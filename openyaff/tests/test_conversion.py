import molmod
import numpy as np

from openyaff import Configuration, ExplicitConversion, \
        OpenMMForceFieldWrapper, YaffForceFieldWrapper

from systems import get_system


def test_simple_covalent_nonperiodic():
    system, pars = get_system('alanine')
    configuration = Configuration(system, pars)
    conversion = ExplicitConversion()
    seed_kind = 'covalent'
    seed_mm = conversion.apply(configuration, seed_kind=seed_kind)
    seed_yaff = configuration.create_seed(kind=seed_kind)

    wrapper_mm = OpenMMForceFieldWrapper.from_seed(seed_mm, 'Reference')
    wrapper_yaff = YaffForceFieldWrapper.from_seed(seed_yaff)
    assert not wrapper_yaff.periodic # system should not be considered periodic
    assert not wrapper_mm.periodic # system should not be considered periodic

    pos = system.pos.copy()
    for i in range(100):
        pos += np.random.uniform(-1, 1, size=pos.shape) * 2
        energy_mm, forces_mm = wrapper_mm.evaluate(pos / molmod.units.angstrom)

        energy, forces = wrapper_yaff.evaluate(pos / molmod.units.angstrom)
        np.testing.assert_almost_equal(
                energy_mm,
                energy,
                )
        np.testing.assert_almost_equal(
                forces_mm,
                forces,
                )


def test_simple_covalent_periodic():
    system, pars = get_system('cobdp')
    configuration = Configuration(system, pars)
    # this test fails for small unit cells
    configuration.supercell = [2, 2, 2]
    conversion = ExplicitConversion()
    seed_kind = 'covalent'
    seed_yaff = configuration.create_seed(kind=seed_kind)
    seed_mm = conversion.apply(configuration, seed_kind=seed_kind)

    wrapper_mm = OpenMMForceFieldWrapper.from_seed(seed_mm, 'Reference')
    wrapper_yaff = YaffForceFieldWrapper.from_seed(seed_yaff)
    assert wrapper_yaff.periodic
    assert wrapper_mm.periodic

    system = seed_yaff.system
    pos = system.pos.copy()
    rvecs = system.cell._get_rvecs().copy()
    for i in range(20):
        dpos = np.random.uniform(-1.0, 1.0, size=pos.shape) * 2.0
        drvecs = np.random.uniform(-0.1, 0.1, size=rvecs.shape) * 3
        drvecs[0, 1] = 0.0
        drvecs[0, 2] = 0.0
        drvecs[1, 2] = 0.0
        energy_mm, forces_mm = wrapper_mm.evaluate(
                (pos + dpos) / molmod.units.angstrom,
                (rvecs + drvecs) / molmod.units.angstrom,
                )

        energy, forces = wrapper_yaff.evaluate(
                (pos + dpos) / molmod.units.angstrom,
                (rvecs + drvecs) / molmod.units.angstrom,
                )
        #delta = np.linalg.norm(forces - forces_mm, axis=1)
        #for i in range(forces.shape[0]):
        #    if delta[i] > 1e-7:
        #        print(i, system.ffatypes[system.ffatype_ids[i]])
        #        print(forces[i], forces_mm[i])
        np.testing.assert_almost_equal(
                energy_mm,
                energy,
                decimal=6,
                )
        np.testing.assert_almost_equal(
                forces_mm,
                forces,
                decimal=6,
                )


def test_simple_dispersion_nonperiodic():
    system, pars = get_system('alanine')
    configuration = Configuration(system, pars)
    # YAFF and OpenMM use a different switching function. If it is disabled,
    # the results between both are identical up to 6 decimals
    conversion = ExplicitConversion()
    seed_kind = 'dispersion'
    seed_mm = conversion.apply(configuration, seed_kind=seed_kind)
    seed_yaff = configuration.create_seed(kind=seed_kind)

    wrapper_mm = OpenMMForceFieldWrapper.from_seed(seed_mm, 'Reference')
    wrapper_yaff = YaffForceFieldWrapper.from_seed(seed_yaff)
    assert not wrapper_yaff.periodic # system should not be considered periodic
    assert not wrapper_mm.periodic # system should not be considered periodic

    pos = system.pos.copy()
    for i in range(30):
        dpos = np.random.uniform(-1.0, 1.0, size=pos.shape)
        energy_mm, forces_mm = wrapper_mm.evaluate(
                (pos + dpos) / molmod.units.angstrom,
                )

        energy, forces = wrapper_yaff.evaluate(
                (pos + dpos) / molmod.units.angstrom,
                )
        np.testing.assert_almost_equal(
                energy_mm,
                energy,
                decimal=6,
                )
        np.testing.assert_almost_equal(
                forces_mm,
                forces,
                decimal=6,
                )

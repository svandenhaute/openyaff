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

    pos = system.pos.copy()
    for i in range(30):
        pos += np.random.uniform(-0.3, 0.3, size=pos.shape)
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
    conversion = ExplicitConversion()
    seed_kind = 'covalent'
    seed_mm = conversion.apply(configuration, seed_kind=seed_kind)
    seed_yaff = configuration.create_seed(kind=seed_kind)

    wrapper_mm = OpenMMForceFieldWrapper.from_seed(seed_mm, 'Reference')
    wrapper_yaff = YaffForceFieldWrapper.from_seed(seed_yaff)

    pos = system.pos.copy()
    rvecs = system.cell._get_rvecs().copy()
    for i in range(15): # fails for large displacements
        pos += np.random.uniform(-0.3, 0.3, size=pos.shape)
        drvecs = np.random.uniform(-0.1, 0.1, size=rvecs.shape)
        drvecs[0, 1] = 0.0
        drvecs[0, 2] = 0.0
        drvecs[1, 2] = 0.0
        rvecs += drvecs
        energy_mm, forces_mm = wrapper_mm.evaluate(
                pos / molmod.units.angstrom,
                rvecs / molmod.units.angstrom,
                )

        energy, forces = wrapper_yaff.evaluate(
                pos / molmod.units.angstrom,
                rvecs / molmod.units.angstrom,
                )
        np.testing.assert_almost_equal(
                energy_mm,
                energy,
                )
        np.testing.assert_almost_equal(
                forces_mm,
                forces,
                )

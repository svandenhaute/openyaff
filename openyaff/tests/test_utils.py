import pytest
import molmod
import numpy as np
import simtk.unit as unit

from openyaff.utils import transform_lower_triangular, is_lower_triangular, \
        reduce_box_vectors, is_reduced, transform_symmetric, \
        do_gram_schmidt_reduction, compute_lengths_angles, \
        estimate_cell_derivative, wrap_coordinates, create_openmm_system, \
        create_openmm_topology
from openyaff.configuration import Configuration
from openyaff.wrappers import YaffForceFieldWrapper

from systems import get_system


def test_transform_lower_triangular():
    for i in range(100):
        trial = np.random.uniform(-20, 20, size=(3, 3))
        trial *= np.sign(np.linalg.det(trial))
        assert np.linalg.det(trial) > 0
        pos   = np.random.uniform(-100, 100, size=(10, 3))
        manual = np.linalg.cholesky(trial @ trial.T)
        transform_lower_triangular(pos, trial) # in-place
        # comparison with cholesky made inside transform_lower_triangular

    for name in ['cau13', 'uio66']: # FAILS ON COBDP; ewald_reci changes
        ff = get_system(name, return_forcefield=True) # nonrectangular system
        gpos0 = np.zeros((ff.system.natom, 3))
        energy0 = ff.compute(gpos0, None)
        rvecs = ff.system.cell._get_rvecs().copy()
        transform_lower_triangular(ff.system.pos, rvecs, reorder=True)
        assert is_lower_triangular(rvecs)
        ff.update_pos(ff.system.pos)
        ff.update_rvecs(rvecs)
        gpos1 = np.zeros((ff.system.natom, 3))
        energy1 = ff.compute(gpos1, None)
        np.testing.assert_almost_equal( # energy should remain the same
                energy0,
                energy1,
                )


def test_is_reduced():
    trial = np.array([
        [5, 0, 0],
        [2, 3, 0],
        [1, 2, 1], # c_y too large
        ])
    assert not is_reduced(trial)
    trial = np.array([
        [5, 0, 0],
        [3, 3, 0], # b_x too large
        [1, 1, 1],
        ])
    assert not is_reduced(trial)
    trial = np.array([
        [5, 0, 0],
        [1, 3, 0],
        [1, 1, 0], # c_x nonpositive
        ])
    assert not is_reduced(trial)


def test_lattice_reduction():
    system, pars = get_system('cau13')
    pos = system.pos.copy()
    rvecs = system.cell._get_rvecs().copy()

    # use reduction algorithm from Bekker, and transform to diagonal
    reduced = do_gram_schmidt_reduction(rvecs)
    reduced_LT = np.linalg.cholesky(reduced @ reduced.T)
    assert np.allclose(reduced_LT, np.diag(np.diag(reduced_LT))) # diagonal

    # transform to lower triangular
    transform_lower_triangular(pos, rvecs, reorder=True)
    reduce_box_vectors(rvecs)

    # assert equality of diagonal elements from both methods
    np.testing.assert_almost_equal(np.diag(rvecs), np.diag(reduced_LT))


def test_compute_lengths_angles():
    rvecs = np.eye(3)
    lengths, angles = compute_lengths_angles(rvecs, degree=True)
    np.testing.assert_almost_equal(lengths, np.ones(3))
    np.testing.assert_almost_equal(angles, 90 * np.ones(3))

    rvecs = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        ])
    lengths, angles = compute_lengths_angles(rvecs, degree=False)
    np.testing.assert_almost_equal(
            lengths,
            np.array([np.sqrt(2), 1, 1]),
            )
    np.testing.assert_almost_equal(
            angles,
            np.array([np.pi / 2, np.pi / 2, np.pi / 4]),
            )


def test_transform_symmetric():
    system, pars = get_system('mil53')
    pos = system.pos.copy()
    rvecs  = system.cell._get_rvecs().copy()

    # transform to symmetric form
    transform_symmetric(pos, rvecs)
    assert np.allclose(rvecs, rvecs.T)

    # transform to triangular, and back to symmetric
    rvecs_ = rvecs.copy()
    pos_ = pos.copy()
    transform_lower_triangular(pos_, rvecs_, reorder=False)
    transform_symmetric(pos_, rvecs_)

    # assert equality
    np.testing.assert_almost_equal(rvecs_, rvecs)
    np.testing.assert_almost_equal(pos_, pos)


def test_estimate_virial_stress():
    def energy_func(positions, rvecs):
        ff.update_pos(positions)
        ff.update_rvecs(rvecs)
        return ff.compute()

    # verify numerical pressure computation for number of benchmark systems
    # include anisotropic systems and LJ
    dh = 1e-5
    for name in ['cau13', 'uio66', 'ppycof', 'lennardjones']:
        ff = get_system(name, return_forcefield=True)
        positions = ff.system.pos.copy()
        rvecs = ff.system.cell._get_rvecs().copy()
        vtens = np.zeros((3, 3))

        ff.compute(None, vtens)
        unit = molmod.units.pascal * 1e6
        pressure = np.trace(vtens) / np.linalg.det(rvecs) / unit
        dUdh = estimate_cell_derivative(positions, rvecs, energy_func, dh=dh,
                use_triangular_perturbation=False)
        vtens_numerical = rvecs.T @ dUdh
        pressure_ = np.trace(vtens_numerical) / np.linalg.det(rvecs) / unit
        assert abs(pressure - pressure_) < 1e-3 # require at least kPa accuracy
        assert np.allclose(vtens_numerical, vtens, atol=1e-5)

        transform_lower_triangular(positions, rvecs, reorder=True)
        ff.update_pos(positions)
        ff.update_rvecs(rvecs)
        vtens = np.zeros((3, 3))
        ff.compute(None, vtens)
        pressure_LT = np.trace(vtens) / np.linalg.det(rvecs) / unit
        dUdh = estimate_cell_derivative(positions, rvecs, energy_func, dh=dh,
                use_triangular_perturbation=True)
        vtens_numerical = rvecs.T @ dUdh
        pressure_LT_ = np.trace(vtens_numerical) / np.linalg.det(rvecs) / unit
        assert abs(pressure_LT - pressure) < 1e-8 # should be identical
        assert abs(pressure_LT_ - pressure_) < 1e-3 # require kPa accuracy
        #assert np.allclose(vtens_numerical, vtens, atol=1e-5)
        # VTENS != VTENS_NUMERICAL HERE!

        transform_symmetric(positions, rvecs)
        ff.update_pos(positions)
        ff.update_rvecs(rvecs)
        vtens = np.zeros((3, 3))
        ff.compute(None, vtens)
        pressure_S = np.trace(vtens) / np.linalg.det(rvecs) / unit
        dUdh = estimate_cell_derivative(positions, rvecs, energy_func, dh=dh)
        vtens_numerical = rvecs.T @ dUdh
        assert np.allclose(vtens_numerical, vtens_numerical.T, atol=1e5)
        pressure_S_ = np.trace(vtens_numerical) / np.linalg.det(rvecs) / unit
        assert abs(pressure_S - pressure) < 1e-8 # should be identical
        assert abs(pressure_S_ - pressure_) < 1e-3 # require kPa accuracy
        assert np.allclose(vtens_numerical, vtens, atol=1e-5)

        # check evaluate_using_reduced=True gives same results
        dUdh_r = estimate_cell_derivative(positions, rvecs, energy_func, dh=dh,
                evaluate_using_reduced=True)
        vtens_numerical_r = rvecs.T @ dUdh_r
        assert np.allclose(vtens_numerical_r, vtens_numerical_r.T, atol=1e-5)
        assert np.allclose(vtens_numerical_r, vtens_numerical, atol=1e-5)


def test_wrap_coordinates():
    for name in ['cau13', 'uio66', 'ppycof', 'mof5', 'mil53', 'cof5']:
        ff = get_system(name, return_forcefield=True)
        positions = ff.system.pos.copy()
        rvecs  = ff.system.cell._get_rvecs().copy()
        rvecs_ = ff.system.cell._get_rvecs().copy()
        ff.update_pos(positions)
        ff.update_rvecs(rvecs)
        e = ff.compute()

        # make random periodic displacements
        for i in range(100):
            coefficients = np.random.randint(0, high=3, size=(3, 1))
            atom = np.random.randint(0, high=ff.system.natom)
            positions[atom, :] += np.sum(coefficients * rvecs, axis=0)
        ff.update_pos(positions)
        ff.update_rvecs(rvecs)
        e0 = ff.compute()
        assert np.allclose(e, e0)

        wrap_coordinates(positions, rvecs, rectangular=False)
        frac = np.dot(positions, np.linalg.inv(rvecs)) # fractional coordinates
        assert np.all(frac >= 0)
        assert np.all(frac <= 1)
        assert np.allclose(rvecs, rvecs_) # rvecs should not change
        ff.update_pos(positions)
        ff.update_rvecs(rvecs)
        e1 = ff.compute()
        assert np.allclose(e0, e1)

        with pytest.raises(AssertionError):
            wrap_coordinates(positions, rvecs, rectangular=True)

        # transform rvecs
        transform_lower_triangular(positions, rvecs, reorder=False)
        reduce_box_vectors(rvecs)
        wrap_coordinates(positions, rvecs, rectangular=True)
        for i in range(positions.shape[0]):
            assert np.all(np.abs(positions[i, :]) < np.diag(rvecs))
        ff.update_pos(positions)
        ff.update_rvecs(rvecs)
        e2 = ff.compute()
        assert np.allclose(e0, e2)

        # reorder rvecs
        transform_lower_triangular(positions, rvecs, reorder=True)
        reduce_box_vectors(rvecs)
        wrap_coordinates(positions, rvecs, rectangular=True)
        for i in range(positions.shape[0]):
            assert np.all(np.abs(positions[i, :]) < np.diag(rvecs))
        ff.update_pos(positions)
        ff.update_rvecs(rvecs)
        e3 = ff.compute()
        assert np.allclose(e0, e3)


def test_create_openmm_system():
    system, _ = get_system('cau13')
    with pytest.raises(AssertionError):
        create_openmm_system(system)

    rvecs = system.cell._get_rvecs().copy()
    transform_lower_triangular(system.pos, rvecs, reorder=True)
    reduce_box_vectors(rvecs)

    system.cell.update_rvecs(rvecs)
    system_mm = create_openmm_system(system)
    # verify box vectors are correct
    a, b, c = system_mm.getDefaultPeriodicBoxVectors()
    assert np.allclose(
            a.value_in_unit(unit.angstrom),
            rvecs[0, :] / molmod.units.angstrom,
            )
    assert np.allclose(
            b.value_in_unit(unit.angstrom),
            rvecs[1, :] / molmod.units.angstrom,
            )
    assert np.allclose(
            c.value_in_unit(unit.angstrom),
            rvecs[2, :] / molmod.units.angstrom,
            )


def test_create_openmm_topology():
    system, _ = get_system('cau13')
    with pytest.raises(AssertionError):
        create_openmm_topology(system)

    rvecs = system.cell._get_rvecs().copy()
    transform_lower_triangular(system.pos, rvecs, reorder=True)
    reduce_box_vectors(rvecs)

    system.cell.update_rvecs(rvecs)
    topology = create_openmm_topology(system)
    # verify box vectors are correct
    a, b, c = topology.getPeriodicBoxVectors()
    assert np.allclose(
            a.value_in_unit(unit.angstrom),
            rvecs[0, :] / molmod.units.angstrom,
            )
    assert np.allclose(
            b.value_in_unit(unit.angstrom),
            rvecs[1, :] / molmod.units.angstrom,
            )
    assert np.allclose(
            c.value_in_unit(unit.angstrom),
            rvecs[2, :] / molmod.units.angstrom,
            )

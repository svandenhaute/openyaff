import molmod
import numpy as np

from openyaff.utils import transform_lower_diagonal, is_lower_diagonal, \
        do_lattice_reduction
from openyaff.configuration import Configuration
from openyaff.wrappers import YaffForceFieldWrapper

from systems import get_system


def test_transform_lower_diagonal():
    for i in range(100):
        trial = np.random.uniform(-20, 20, size=(3, 3))
        trial *= np.sign(np.linalg.det(trial))
        assert np.linalg.det(trial) > 0
        pos   = np.random.uniform(-100, 100, size=(10, 3))
        manual = np.linalg.cholesky(trial @ trial.T)
        transform_lower_diagonal(pos, trial) # in-place
        # comparison with cholesky made inside transform_lower_diagonal

    ff = get_system('cobdp', return_forcefield=True) # nonrectangular system
    gpos0 = np.zeros((ff.system.natom, 3))
    energy0 = ff.compute(gpos0, None)
    rvecs = ff.system.cell._get_rvecs().copy()
    # test fails for COBDP if reordering=True!
    transform_lower_diagonal(ff.system.pos, rvecs, reorder=False)
    ff.update_pos(ff.system.pos)
    ff.update_rvecs(rvecs)
    gpos1 = np.zeros((ff.system.natom, 3))
    energy1 = ff.compute(gpos1, None)
    np.testing.assert_almost_equal( # energy should remain the same
            energy0,
            energy1,
            )
    np.testing.assert_raises( # gpos different because of rotation
            AssertionError,
            np.testing.assert_array_almost_equal,
            gpos0,
            gpos1,
            )


def test_is_lower_diagonal():
    trial = np.array([
        [5, 0, 0],
        [2, 3, 0],
        [1, 2, 1], # c_y too large
        ])
    assert not is_lower_diagonal(trial)
    trial = np.array([
        [5, 0, 0],
        [3, 3, 0], # b_x too large
        [1, 1, 1],
        ])
    assert not is_lower_diagonal(trial)
    trial = np.array([
        [5, 0, 0],
        [1, 3, 0],
        [1, 1, 0], # c_x nonpositive
        ])
    assert not is_lower_diagonal(trial)


def test_lattice_reduction():
    system, pars = get_system('cobdp')
    rvecs = system.cell._get_rvecs().copy()
    assert not is_lower_diagonal(rvecs)
    configuration = Configuration(system, pars)
    seed = configuration.create_seed('full')
    wrapper = YaffForceFieldWrapper.from_seed(seed)
    positions = seed.system.pos.copy() / molmod.units.angstrom
    rvecs = seed.system.cell._get_rvecs().copy() / molmod.units.angstrom
    energy0 = wrapper.evaluate(positions, rvecs=rvecs, do_forces=False)

    # compute fractional coordinates; displace all particles to primitive cell
    frac = np.dot(positions, np.linalg.inv(rvecs))
    frac -= np.floor(frac)
    assert np.all(frac >= 0)
    assert np.all(frac <= 1)

    # recreate unit cell by multiplying with old cell, and evaluate energy
    pos_generated = np.dot(frac, rvecs)
    energy1 = wrapper.evaluate(pos_generated, rvecs=rvecs, do_forces=False)
    np.testing.assert_almost_equal(energy0, energy1)

    # transform to orthogonal lattice basis
    reduced = do_lattice_reduction(rvecs)
    delta = np.random.uniform(-1, 1, size=(3, 3))
    np.testing.assert_almost_equal( # still primitive cell; volume unchanged
            np.linalg.det(reduced),
            np.linalg.det(rvecs),
            )
    # displace all particles into new cell
    frac_ = np.dot(positions, np.linalg.inv(reduced))
    frac_ -= np.floor(frac_)
    assert np.all(frac_ >= 0)
    assert np.all(frac_ <= 1)
    pos_generated = np.dot(frac_, reduced) # should represent same unit cell

    # transform back to original primitive cell and assert no positions changed
    frac__ = np.dot(pos_generated, np.linalg.inv(rvecs))
    frac__ -= np.floor(frac__)
    assert np.all(frac__ >= 0)
    assert np.all(frac__ <= 1)
    np.allclose(frac, frac__)

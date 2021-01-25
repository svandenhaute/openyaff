import numpy as np

from openyaff.utils import transform_lower_diagonal, check_reduced_form

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
    transform_lower_diagonal(ff.system.pos, rvecs)
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


def test_check_reduced_form():
    trial = np.array([
        [5, 0, 0],
        [2, 3, 0],
        [1, 2, 1], # c_y too large
        ])
    assert not check_reduced_form(trial)
    trial = np.array([
        [5, 0, 0],
        [3, 3, 0], # b_x too large
        [1, 1, 1],
        ])
    assert not check_reduced_form(trial)
    trial = np.array([
        [5, 0, 0],
        [1, 3, 0],
        [1, 1, 0], # c_x nonpositive
        ])
    assert not check_reduced_form(trial)

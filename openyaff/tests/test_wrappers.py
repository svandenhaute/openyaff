import numpy as np
import molmod

from openyaff import YaffForceFieldWrapper

from systems import get_system


def test_yaff_wrapper_periodic():
    ff = get_system('cobdp', return_forcefield=True)
    wrapper = YaffForceFieldWrapper(ff)

    pos = ff.system.pos.copy()
    rvecs = ff.system.cell._get_rvecs().copy()

    nstates = 10
    pos_ = np.zeros((nstates,) + pos.shape)
    rvecs_ = np.zeros((nstates,) + (3, 3))
    energy_ = np.zeros(10)
    gpos_ = np.zeros(pos_.shape)
    for i in range(10):
        pos += np.random.uniform(-1, 1, size=pos.shape) / 5
        rvecs += np.random.uniform(-0.5, 0.5, size=rvecs.shape) / 5
        pos_[i, :] = pos[:] / molmod.units.angstrom
        rvecs_[i, :] = rvecs[:] / molmod.units.angstrom
        ff.update_pos(pos)
        ff.update_rvecs(rvecs)
        energy_[i] = ff.compute(gpos_[i, :], None)

    energy_w, force_w = wrapper.evaluate(pos_, rvecs=rvecs_, do_forces=True)
    np.testing.assert_almost_equal(
            energy_ / molmod.units.kjmol,
            energy_w,
            decimal=6, # test sometimes fail for 7
            )
    np.testing.assert_almost_equal(
            - gpos_ / molmod.units.kjmol * molmod.units.angstrom,
            force_w,
            decimal=6, # test sometimes fail for 7
            )


def test_yaff_wrapper_nonperiodic():
    ff = get_system('alanine', return_forcefield=True)
    wrapper = YaffForceFieldWrapper(ff)

    pos = ff.system.pos.copy()

    nstates = 10
    pos_ = np.zeros((nstates,) + pos.shape)
    energy_ = np.zeros(10)
    gpos_ = np.zeros(pos_.shape)
    for i in range(10):
        pos += np.random.uniform(-1, 1, size=pos.shape) / 5
        pos_[i, :] = pos[:] / molmod.units.angstrom
        ff.update_pos(pos)
        energy_[i] = ff.compute(gpos_[i, :], None)

    energy_w, force_w = wrapper.evaluate(pos_, rvecs=None, do_forces=True)
    np.testing.assert_almost_equal(
            energy_ / molmod.units.kjmol,
            energy_w,
            decimal=6, # test sometimes fail for 7
            )
    np.testing.assert_almost_equal(
            - gpos_ / molmod.units.kjmol * molmod.units.angstrom,
            force_w,
            decimal=6, # test sometimes fail for 7
            )

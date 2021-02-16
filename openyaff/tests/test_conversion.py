import molmod
import pytest
import numpy as np

from openyaff import Configuration, ExplicitConversion, \
        OpenMMForceFieldWrapper, YaffForceFieldWrapper
from openyaff.utils import reduce_box_vectors

from systems import get_system
from conftest import assert_tol


def test_periodic():
    systems    = ['uio66', 'cau13', 'mil53', 'ppycof', 'cof5', 'mof5']
    platforms  = ['Reference']
    seed_kinds = ['covalent', 'dispersion', 'electrostatic']

    # systematic constant offset in dispersion energy for COFs, unclear why

    tolerance = {
            ('Reference', 'covalent'): 1e-6, # some MM3 terms have error 1e-7
            ('Reference', 'dispersion'): 1e-3,
            ('Reference', 'electrostatic'): 1e-3,
            #('CUDA', 'covalent'): 1e-3,
            #('CUDA', 'dispersion'): 1e-3,
            #('CUDA', 'electrostatic'): 1e-3,
            }

    nstates   = 5
    disp_ampl = 0.5
    box_ampl  = 0.5

    for name in systems:
        for platform in platforms:
            for kind in seed_kinds:
                system, pars = get_system(name)
                configuration = Configuration(system, pars)
                tol = tolerance[(platform, kind)]

                # YAFF and OpenMM use a different switching function. If it is disabled,
                # the results between both are identical up to 6 decimals
                configuration.switch_width = 0.0 # disable switching
                configuration.rcut = 10.0 # request cutoff of 10 angstorm
                configuration.cell_interaction_radius = 15.0
                #supercell = configuration.determine_supercell(rcut)
                #configuration.supercell = list(supercell) # set required supercell
                configuration.update_properties(
                        configuration.write(),
                        )
                conversion = ExplicitConversion(pme_error_thres=1e-5)
                seed_mm = conversion.apply(configuration, seed_kind=kind)
                seed_yaff = configuration.create_seed(kind=kind)

                wrapper_mm = OpenMMForceFieldWrapper.from_seed(seed_mm, platform)
                wrapper_yaff = YaffForceFieldWrapper.from_seed(seed_yaff)
                assert wrapper_yaff.periodic # system should not be considered periodic
                assert wrapper_mm.periodic # system should not be considered periodic

                pos = seed_yaff.system.pos.copy()
                rvecs = seed_yaff.system.cell._get_rvecs().copy()
                for i in range(nstates):
                    dpos = np.random.uniform(-disp_ampl, disp_ampl, size=pos.shape)
                    drvecs = np.random.uniform(-box_ampl, box_ampl, size=rvecs.shape)
                    drvecs[0, 1] = 0
                    drvecs[0, 2] = 0
                    drvecs[1, 2] = 0
                    tmp = rvecs + drvecs
                    reduce_box_vectors(tmp)
                    energy_mm, forces_mm = wrapper_mm.evaluate(
                            (pos + dpos) / molmod.units.angstrom,
                            rvecs=tmp / molmod.units.angstrom,
                            )
                    energy, forces = wrapper_yaff.evaluate(
                            (pos + dpos) / molmod.units.angstrom,
                            rvecs=tmp / molmod.units.angstrom,
                            )
                    assert_tol(energy, energy_mm, tol)
                    assert_tol(forces, forces_mm, 10 * tol)


def test_nonperiodic():
    systems    = ['alanine']
    platforms  = ['Reference']
    seed_kinds = ['covalent', 'dispersion', 'electrostatic']

    tolerance = {
            ('Reference', 'covalent'): 1e-6,
            ('Reference', 'dispersion'): 1e-6,
            ('Reference', 'electrostatic'): 1e-6,
            #('Cuda', 'covalent'): 1e-5,
            #('Cuda', 'dispersion'): 1e-5,
            #('Cuda', 'electrostatic'): 1e-5,
            }

    nstates   = 10
    disp_ampl = 1.0
    box_ampl  = 1.0

    for name in systems:
        for platform in platforms:
            for kind in seed_kinds:
                system, pars = get_system(name)
                configuration = Configuration(system, pars)
                tol = tolerance[(platform, kind)]

                conversion = ExplicitConversion()
                seed_mm = conversion.apply(configuration, seed_kind=kind)
                seed_yaff = configuration.create_seed(kind=kind)

                wrapper_mm = OpenMMForceFieldWrapper.from_seed(seed_mm, platform)
                wrapper_yaff = YaffForceFieldWrapper.from_seed(seed_yaff)
                assert not wrapper_yaff.periodic # system should not be considered periodic
                assert not wrapper_mm.periodic # system should not be considered periodic

                pos = seed_yaff.system.pos.copy()
                for i in range(nstates):
                    dpos = np.random.uniform(-disp_ampl, disp_ampl, size=pos.shape)
                    energy_mm, forces_mm = wrapper_mm.evaluate(
                            (pos + dpos) / molmod.units.angstrom,
                            )
                    energy, forces = wrapper_yaff.evaluate(
                            (pos + dpos) / molmod.units.angstrom,
                            )
                    assert_tol(energy, energy_mm, tol)
                    assert_tol(forces, forces_mm, 10 * tol)


def test_check_compatibility():
    system, _ = get_system('lennardjones')
    conversion = ExplicitConversion()
    seed_kind = 'dispersion'

    # generate pars with unsupported prefix
    pars_unsupported = """
    MM3CAP:UNIT SIGMA angstrom
    MM3CAP:UNIT EPSILON kcalmol
    MM3CAP:SCALE 1 1.0
    MM3CAP:SCALE 2 1.0
    MM3CAP:SCALE 3 1.0

    # ---------------------------------------------
    # KEY      ffatype  SIGMA  EPSILON  ONLYPAULI
    # ---------------------------------------------

    MM3CAP:PARS      C     2.360   0.116      0"""

    with pytest.raises(AssertionError):
        configuration = Configuration(system, pars_unsupported)

    # generate pars with unsupported scaling
    pars_unsupported = """
    LJ:UNIT SIGMA angstrom
    LJ:UNIT EPSILON kcalmol
    LJ:SCALE 1 0.5
    LJ:SCALE 2 1.0
    LJ:SCALE 3 1.0

    # ---------------------------------------------
    # KEY      ffatype  SIGMA  EPSILON  ONLYPAULI
    # ---------------------------------------------

    LJ:PARS      C     2.360   0.116      0"""

    configuration = Configuration(system, pars_unsupported)
    with pytest.raises(AssertionError):
        seed_mm = conversion.apply(configuration, seed_kind=seed_kind)


def test_write_annotate(tmp_path):
    path_config = tmp_path / 'config.yml'
    ExplicitConversion().write(path_config)
    with open(path_config, 'r') as f:
        content = f.read()
    assert content == """conversion:
  kind: explicit
  pme_error_thres: 1.0e-05
"""
    ExplicitConversion.annotate(path_config)

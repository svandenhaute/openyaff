import molmod
import pytest
import numpy as np
from lxml import etree

import simtk.openmm as mm
import simtk.openmm.app
import simtk.unit as unit

from openyaff import Configuration, ExplicitConversion, \
        OpenMMForceFieldWrapper, YaffForceFieldWrapper
from openyaff.utils import reduce_box_vectors

from systems import get_system
from conftest import assert_tol


def test_implicit_nonperiodic()
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


def test_save_load_pdb(tmp_path):
    system, pars = get_system('mil53')
    configuration = Configuration(system, pars)

    # YAFF and OpenMM use a different switching function. If it is disabled,
    # the results between both are identical up to 6 decimals
    configuration.switch_width = 0.0 # disable switching
    configuration.rcut = 10.0 # request cutoff of 10 angstorm
    configuration.interaction_radius = 11.0
    #configuration.update_properties(configuration.write())

    conversion = ExplicitConversion(pme_error_thres=5e-4)
    seed_mm = conversion.apply(configuration, seed_kind='all')
    seed_yaff = configuration.create_seed(kind='all')

    wrapper_mm = OpenMMForceFieldWrapper.from_seed(seed_mm, 'Reference')
    wrapper_yaff = YaffForceFieldWrapper.from_seed(seed_yaff)
    assert wrapper_yaff.periodic # system should not be considered periodic
    assert wrapper_mm.periodic # system should not be considered periodic

    positions = seed_yaff.system.pos.copy() / molmod.units.angstrom
    rvecs = seed_yaff.system.cell._get_rvecs().copy() / molmod.units.angstrom

    e0, f0 = wrapper_mm.evaluate(positions, rvecs, do_forces=True)
    e1, f1 = wrapper_yaff.evaluate(positions, rvecs, do_forces=True)
    assert np.allclose(e0, e1, rtol=1e-3)

    path_pdb = tmp_path / 'top.pdb'
    seed_yaff.save_topology(path_pdb) # stores current positions and box vectors
    pdb = mm.app.PDBFile(str(path_pdb))
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    a, b, c  = pdb.getTopology().getPeriodicBoxVectors()
    rvecs = np.array([
        a.value_in_unit(unit.angstrom),
        b.value_in_unit(unit.angstrom),
        c.value_in_unit(unit.angstrom)])

    e2, f2 = wrapper_mm.evaluate(positions, rvecs, do_forces=True)
    e3, f3 = wrapper_yaff.evaluate(positions, rvecs, do_forces=True)
    assert np.allclose(e2, e3, rtol=1e-3)
    assert np.allclose(e1, e3, rtol=1e-4) # rounding errors during saving pdb
    assert np.allclose(e0, e2, rtol=1e-4)


def test_periodic():
    systems    = ['uio66', 'cau13', 'mil53', 'ppycof', 'cof5', 'mof5']
    platforms  = ['Reference']
    seed_kinds = ['covalent', 'dispersion', 'electrostatic']

    # systematic constant offset in dispersion energy for COFs, unclear why

    tolerance = {
            ('Reference', 'covalent'): 1e-6, # some MM3 terms have error 1e-7
            ('Reference', 'dispersion'): 1e-2, # some MM3 terms have error 1e-3
            ('Reference', 'electrostatic'): 1e-3,
            #('CUDA', 'covalent'): 1e-3,
            #('CUDA', 'dispersion'): 1e-3,
            #('CUDA', 'electrostatic'): 1e-3,
            }

    nstates   = 5
    disp_ampl = 0.3
    box_ampl  = 0.3

    for name in systems:
        for platform in platforms:
            for kind in seed_kinds:
                system, pars = get_system(name)
                configuration = Configuration(system, pars)
                tol = tolerance[(platform, kind)]

                # YAFF and OpenMM use a different switching function. If it is disabled,
                # the results between both are identical up to 6 decimals
                configuration.switch_width = 0.0 # disable switching
                configuration.rcut = 13.0 # request cutoff of 13 angstorm
                configuration.interaction_radius = 15.0
                configuration.update_properties(configuration.write())
                conversion = ExplicitConversion(pme_error_thres=5e-4)
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


@pytest.mark.skip(reason='removing ludicrous mode')
def test_serialize_aggregate_nonperiodic(tmp_path):
    system, pars = get_system('alanine')
    configuration = Configuration(system, pars)
    conversion = ExplicitConversion()
    seed      = conversion.apply(configuration)
    wrapper   = OpenMMForceFieldWrapper.from_seed(seed, 'Reference')
    seed_l    = conversion.apply_ludicrous(configuration)
    wrapper_l = OpenMMForceFieldWrapper.from_seed(seed_l, 'Reference')

    positions = system.pos / molmod.units.angstrom
    energy, forces = wrapper.evaluate(positions)
    energy_l, forces_l = wrapper_l.evaluate(positions)
    assert_tol(energy, energy_l, 1e-5)
    assert_tol(forces, forces_l, 1e-5)


@pytest.mark.skip(reason='removing ludicrous mode')
def test_serialize_aggregate_nonperiodic(tmp_path):
    system, pars = get_system('cau13')
    configuration = Configuration(system, pars)
    configuration.switch_width = 0.0 # disable switching
    configuration.rcut = 10.0 # request cutoff of 10 angstorm
    configuration.interaction_radius = 15.0
    configuration.update_properties(configuration.write())
    yaff_seed = configuration.create_seed('covalent')
    positions = yaff_seed.system.pos / molmod.units.angstrom
    rvecs = yaff_seed.system.cell._get_rvecs() / molmod.units.angstrom

    conversion = ExplicitConversion()
    seed      = conversion.apply(configuration)
    wrapper   = OpenMMForceFieldWrapper.from_seed(seed, 'Reference')
    seed_l    = conversion.apply_ludicrous(configuration)
    wrapper_l = OpenMMForceFieldWrapper.from_seed(seed_l, 'Reference')

    energy, forces = wrapper.evaluate(positions, rvecs)
    energy_l, forces_l = wrapper_l.evaluate(positions, rvecs)
    assert_tol(energy, energy_l, 1e-3)
    assert_tol(forces, forces_l, 1e-3)


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
    BLAAA:UNIT SIGMA angstrom
    BLAAA:UNIT EPSILON kcalmol
    BLAAA:SCALE 1 1.0
    BLAAA:SCALE 2 1.0
    BLAAA:SCALE 3 1.0

    # ---------------------------------------------
    # KEY      ffatype  SIGMA  EPSILON  ONLYPAULI
    # ---------------------------------------------

    BLAAA:PARS      C     2.360   0.116      0"""

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

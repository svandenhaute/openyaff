import molmod
import yaff
import tempfile
import numpy as np
import simtk.openmm as mm
import simtk.openmm.app
from pathlib import Path


# specifying absolute path ensures tests may be executed from any directory
here = Path(__file__).parent


def lennardjones(natoms=40, volume=1000):
    """Generate lennard jones system

    Parameters
    ----------

    natoms : int, optional
        specifies the number of atoms in the box

    volume : float [A ** 3]
        specifies the volume of the periodic box

    """
    # generate random non-overlapping positions in 
    pos        = np.zeros((natoms, 3))
    box_length = volume ** (1 / 3)
    rvecs      = box_length * np.eye(3) * molmod.units.angstrom
    begin      = box_length * 0.1 * molmod.units.angstrom
    end        = box_length * 0.9 * molmod.units.angstrom

    # do not allow atoms to be closer than 2 angstrom
    distance_thres = 2 * molmod.units.angstrom
    i = 0
    while i < natoms:
        trial = np.random.uniform(begin, end, (3,))
        too_close = False
        for j in range(i):
            if np.linalg.norm(pos[j, :] - trial) < distance_thres:
                too_close = True
        if not too_close: # assign positions and move to next atom
            pos[i, :] = trial
            i += 1

    system = yaff.System(
            12 * np.ones(natoms, dtype=int),
            pos,
            ffatypes=['C'],
            ffatype_ids=np.zeros(natoms, dtype=int),
            rvecs=rvecs,
            )
    # random LJ parameters; SCALE should equal 1.0 as no bonds are defined
    pars = """
    LJ:UNIT SIGMA angstrom
    LJ:UNIT EPSILON kcalmol
    LJ:SCALE 1 1.0
    LJ:SCALE 2 1.0
    LJ:SCALE 3 1.0

    # ---------------------------------------------
    # KEY      ffatype  SIGMA  EPSILON  ONLYPAULI
    # ---------------------------------------------

    LJ:PARS      C     2.360   0.116      0"""
    return system, pars


def methane(return_forcefield=False):
    """Generate box of methane molecules from YAFF input files"""
    path_system = str(here / 'methane' / 'system.chk')
    path_pars = str(here / 'methane' / 'pars.txt')
    system = yaff.System.from_file(path_system)
    with open(path_pars, 'r') as f:
        pars = f.read()
    return system, pars


def mil53(return_forcefield=False):
    """Generate MIL53 system from YAFF input files"""
    path_system = str(here / 'mil53' / 'system.chk')
    path_pars = str(here / 'mil53' / 'pars.txt')
    system = yaff.System.from_file(path_system)
    with open(path_pars, 'r') as f:
        pars = f.read()
    return system, pars


def cof5(return_forcefield=False):
    """Generate COF5 system from YAFF input files"""
    path_system = str(here / 'cof5' / 'system.chk')
    path_pars = str(here / 'cof5' / 'pars.txt')
    system = yaff.System.from_file(path_system)
    with open(path_pars, 'r') as f:
        pars = f.read()
    return system, pars


def cau13(return_forcefield=False):
    """Generate CAU13 system from YAFF input files"""
    path_system = str(here / 'cau13' / 'system.chk')
    path_pars = str(here / 'cau13' / 'pars.txt')
    system = yaff.System.from_file(path_system)
    with open(path_pars, 'r') as f:
        pars = f.read()
    return system, pars


def ppycof(return_forcefield=False):
    """Generate PPYCOF system from YAFF input files"""
    path_system = str(here / 'ppycof' / 'system.chk')
    path_pars = str(here / 'ppycof' / 'pars.txt')
    system = yaff.System.from_file(path_system)
    with open(path_pars, 'r') as f:
        pars = f.read()
    return system, pars


def uio66(return_forcefield=False):
    """Generate UIO66 system from YAFF input files"""
    path_system = str(here / 'uio66' / 'system.chk')
    path_pars = str(here / 'uio66' / 'pars.txt')
    system = yaff.System.from_file(path_system)
    with open(path_pars, 'r') as f:
        pars = f.read()
    return system, pars


def mof5(return_forcefield=False):
    """Generate MOF5 system from YAFF input files"""
    path_system = str(here / 'uio66' / 'system.chk')
    path_pars = str(here / 'uio66' / 'pars.txt')
    system = yaff.System.from_file(path_system)
    with open(path_pars, 'r') as f:
        pars = f.read()
    return system, pars


def alanine(return_forcefield=False):
    path_system = str(here / 'alanine' / 'system.chk')
    path_pars = str(here / 'alanine' / 'pars.txt')
    system = yaff.System.from_file(path_system)
    with open(path_pars, 'r') as f:
        pars = f.read()
    return system, pars


def polymer():
    path_system = str(here / 'polymer' / 'system.chk')
    path_pars = str(here / 'polymer' / 'pars.txt')
    system = yaff.System.from_file(path_system)
    with open(path_pars, 'r') as f:
        pars = f.read()
    pdb = mm.app.PDBFile(str(here / 'polymer' / 'topology.pdb'))
    return (system, pdb.getTopology()), pars


def get_system(name, return_forcefield=False, **kwargs):
    if name == 'lennardjones':
        system, pars = lennardjones(**kwargs)
    elif name == 'mil53':
        system, pars = mil53(**kwargs)
    elif name == 'cof5':
        system, pars = cof5(**kwargs)
    elif name == 'mof5':
        system, pars = mof5(**kwargs)
    elif name == 'cau13':
        system, pars = cau13(**kwargs)
    elif name == 'ppycof':
        system, pars = ppycof(**kwargs)
    elif name == 'uio66':
        system, pars = uio66(**kwargs)
    elif name == 'alanine':
        system, pars = alanine(**kwargs)
    elif name == 'methane':
        system, pars = methane(**kwargs)
    elif name == 'polymer':
        system, pars = polymer(**kwargs) # system -> (system, topology)
    else:
        raise NotImplementedError

    if return_forcefield: # return force field
        with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tf:
            tf.write(pars)
        tf.close()
        parameters = yaff.Parameters.from_file(tf.name)
        forcefield = yaff.ForceField.generate(system, parameters)
        return forcefield
    else:
        return system, pars

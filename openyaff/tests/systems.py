import molmod
import yaff
import numpy as np


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


def get_system(name, **kwargs):
    if name == 'lennardjones':
        return lennardjones(**kwargs)
    else:
        raise NotImplementedError

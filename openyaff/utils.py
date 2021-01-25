import yaff
import molmod
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm


def check_reduced_form(rvecs):
    """Returns whether rvecs is in reduced form

    OpenMM puts requirements on the components of the box vectors.
    Essentially, rvecs has to be a lower triangular positive definite matrix
    where additionally (a_x > 2*b_x), (a_x > 2*c_x), and (b_y > 2*c_y).

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    return (rvecs[0, 0] > 2 * rvecs[1, 0] and # b mostly along y axis
            rvecs[0, 0] > 2 * rvecs[2, 0] and # z mostly along z axis
            rvecs[1, 1] > 2 * rvecs[2, 1] and # z mostly along z axis
            rvecs[0, 0] > 0 and # positive volumes
            rvecs[1, 1] > 0 and
            rvecs[2, 2] > 0 and
            rvecs[0, 1] == 0 and # lower triangular
            rvecs[0, 2] == 0 and
            rvecs[1, 2] == 0)


def determine_rcut(rvecs):
    """Determines the maximum allowed cutoff radius of rvecs

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    if not check_reduced_form(rvecs):
        raise ValueError('Box vectors are not in reduced form')
    else:
        return min([
                rvecs[0, 0],
                rvecs[1, 1],
                rvecs[2, 2],
                ]) / 2


def transform_lower_diagonal(pos, rvecs):
    """Transforms coordinate axes such that cell matrix is lower diagonal

    The transformation is derived from the QR decomposition and performed
    in-place

    Parameters
    ----------

    pos : array_like
        (natoms, 3) array containing atomic positions

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    q, r = np.linalg.qr(rvecs.T)
    flip_vectors = np.eye(3) * np.diag(np.sign(r)) # reflections after rotation
    rotation = np.linalg.inv(q.T) @ flip_vectors # full (improper) rotation
    pos[:]   = pos @ rotation
    rvecs[:] = rvecs @ rotation
    assert np.allclose(rvecs, np.linalg.cholesky(rvecs @ rvecs.T))
    rvecs[0, 1] = 0
    rvecs[0, 2] = 0
    rvecs[1, 2] = 0


def compute_lengths_angles(rvecs):
    """Computes and returns the box vector lengths and angles"""
    raise NotImplementedError


def yaff_generate(seed):
    """Generates a yaff.ForceField instance based on a seed

    Parameters
    ----------

    seed : openyaff.YaffSeed
        seed for the yaff force field wrapper

    """
    system = seed.system
    parameters = seed.parameters
    ff_args = seed.ff_args
    yaff.apply_generators(system, parameters, ff_args)
    return yaff.ForceField(system, ff_args.parts, ff_args.nlist)


#def openmm_generate(seed)
#    """Generates a yaff.ForceField instance based on a seed
#
#    Parameters
#    ----------
#
#    seed : tuple of (mm.System,)
#        seed for the OpenMM force field wrapper. For now, this is simply an
#        OpenMM system object in which all forces are added by an
#        openyaff.ExplicitConversion.apply() call.
#
#    """
#    pass


def create_openmm_system(system):
    """Creates an OpenMM system object from a yaff.System object

    Parameters
    ----------

    system : yaff.System
        yaff System for which to generate an OpenMM variant

    """
    system.set_standard_masses()
    system_mm = mm.System()
    if system.cell.nvec == 3:
        rvecs = system.cell._get_rvecs() / molmod.units.angstrom / 10.0 * unit.nanometer
        system_mm.setDefaultPeriodicBoxVectors(*rvecs)
    for i in range(system.pos.shape[0]):
        system_mm.addParticle(system.masses[i] / molmod.units.amu * unit.dalton)
    return system_mm

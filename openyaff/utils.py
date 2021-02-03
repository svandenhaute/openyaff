import yaff
import molmod
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm


def is_lower_triangular(rvecs):
    """Returns whether rvecs is in standardized lower diagonal form

    OpenMM puts requirements on the components of the box vectors.
    Essentially, rvecs has to be a lower triangular positive definite matrix
    where additionally (a_x > 2*|b_x|), (a_x > 2*|c_x|), and (b_y > 2*|c_y|).

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    return (rvecs[0, 0] > abs(2 * rvecs[1, 0]) and # b mostly along y axis
            rvecs[0, 0] > abs(2 * rvecs[2, 0]) and # z mostly along z axis
            rvecs[1, 1] > abs(2 * rvecs[2, 1]) and # z mostly along z axis
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
    if not is_lower_triangular(rvecs):
        raise ValueError('Box vectors are not in reduced form')
    else:
        return min([
                rvecs[0, 0],
                rvecs[1, 1],
                rvecs[2, 2],
                ]) / 2


def transform_lower_triangular(pos, rvecs, reorder=False):
    """Transforms coordinate axes such that cell matrix is lower diagonal

    The transformation is derived from the QR decomposition and performed
    in-place. Because the lower triangular form puts restrictions on the size
    of off-diagonal elements, lattice vectors are by default reordered from
    largest to smallest; this feature can be disabled using the reorder
    keyword.

    Parameters
    ----------

    pos : array_like
        (natoms, 3) array containing atomic positions

    rvecs : array_like
        (3, 3) array with box vectors as rows

    reorder

    """
    if reorder: # reorder box vectors as k, l, m with |k| >= |l| >= |m|
        norms = np.linalg.norm(rvecs, axis=1)
        ordering = np.argsort(norms)[::-1] # largest first
        a = rvecs[ordering[0], :].copy()
        b = rvecs[ordering[1], :].copy()
        c = rvecs[ordering[2], :].copy()
        rvecs[0, :] = a[:]
        rvecs[1, :] = b[:]
        rvecs[2, :] = c[:]
    q, r = np.linalg.qr(rvecs.T)
    flip_vectors = np.eye(3) * np.diag(np.sign(r)) # reflections after rotation
    rotation = np.linalg.inv(q.T) @ flip_vectors # full (improper) rotation
    pos[:]   = pos @ rotation
    rvecs[:] = rvecs @ rotation
    assert np.allclose(rvecs, np.linalg.cholesky(rvecs @ rvecs.T))
    rvecs[0, 1] = 0
    rvecs[0, 2] = 0
    rvecs[1, 2] = 0

    # replace c and b with shortest possible vectors to ensure 
    # b_y > |2 c_y|
    # b_x > |2 c_x|
    # a_x > |2 b_x|
    rvecs[2, :] = rvecs[2, :] - rvecs[1, :] * np.round(rvecs[2, 1] / rvecs[1, 1])
    rvecs[2, :] = rvecs[2, :] - rvecs[0, :] * np.round(rvecs[2, 0] / rvecs[0, 0])
    rvecs[1, :] = rvecs[1, :] - rvecs[0, :] * np.round(rvecs[1, 0] / rvecs[0, 0])
    assert is_lower_triangular(rvecs)


def compute_lengths_angles(rvecs):
    """Computes and returns the box vector lengths and angles"""
    raise NotImplementedError


def do_lattice_reduction(rvecs):
    """Transforms a triclinic cell into a rectangular cell

    The algorithm is described in Bekker (1997). Essentially, it performs a
    Gram-Schmidt orthogonalization of the existing lattice vectors, after first
    reordering them from longest to shortest.

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows. These need not be in lower
        diagonal form.

    """
    # reorder box vectors as k, l, m with |k| >= |l| >= |m|
    norms = np.linalg.norm(rvecs, axis=1)
    ordering = np.argsort(norms)[::-1] # largest first
    # define original lattice basis vectors and their norms
    k = rvecs[ordering[0], :]
    l = rvecs[ordering[1], :]
    m = rvecs[ordering[2], :]
    k_ = k / np.linalg.norm(k)
    l_ = l / np.linalg.norm(k)
    m_ = m / np.linalg.norm(k)
    # define orthogonal basis for the same lattice.
    u = k
    v = l - np.dot(l, k_) * k_
    kvecl_ = np.cross(k, l) / np.linalg.norm(np.cross(k, l))
    w = np.dot(m, kvecl_) * kvecl_
    # assert orthogonality of new basis
    np.testing.assert_almost_equal(np.dot(u, v), 0.0)
    np.testing.assert_almost_equal(np.dot(u, w), 0.0)
    np.testing.assert_almost_equal(np.dot(v, w), 0.0)
    reduced = np.stack((u, v, w), axis=0)
    reduced *= np.sign(np.linalg.det(reduced)) # ensure positive determinant
    return reduced


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

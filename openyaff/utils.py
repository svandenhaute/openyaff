import yaff
import numpy as np


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

    seed : tuple of (yaff.System, yaff.Parameters, yaff.FFArgs)
        seed for the force field

    """
    yaff.apply_generators(*seed)
    ff_args = seed[2]
    system = seed[0]
    return yaff.ForceField(system, ff_args.parts, ff_args.nlist)


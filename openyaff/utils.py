import yaff
import molmod
import numpy as np
import simtk.unit as unit
import simtk.openmm.app
import simtk.openmm as mm

from datetime import datetime


def is_lower_triangular(rvecs):
    """Returns whether rvecs are in lower triangular form

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    return (rvecs[0, 0] > 0 and # positive volumes
            rvecs[1, 1] > 0 and
            rvecs[2, 2] > 0 and
            rvecs[0, 1] == 0 and # lower triangular
            rvecs[0, 2] == 0 and
            rvecs[1, 2] == 0)


def is_reduced(rvecs):
    """Returns whether rvecs are in reduced form

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
            is_lower_triangular(rvecs))


def transform_lower_triangular(pos, rvecs, reorder=False):
    """Transforms coordinate axes such that cell matrix is lower diagonal

    The transformation is derived from the QR decomposition and performed
    in-place. Because the lower triangular form puts restrictions on the size
    of off-diagonal elements, lattice vectors are by default reordered from
    largest to smallest; this feature can be disabled using the reorder
    keyword.
    The box vector lengths and angles remain exactly the same.

    Parameters
    ----------

    pos : array_like
        (natoms, 3) array containing atomic positions

    rvecs : array_like
        (3, 3) array with box vectors as rows

    reorder : bool
        whether box vectors are reordered from largest to smallest

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


def transform_symmetric(pos, rvecs):
    """Transforms coordinate axes such that cell matrix is lower diagonal

    Parameters
    ----------

    pos : array_like
        (natoms, 3) array containing atomic positions

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    U, s, Vt = np.linalg.svd(rvecs)
    rot_mat = np.dot(Vt.T, U.T)
    rvecs[:] = np.dot(rvecs, rot_mat)
    pos[:] = np.dot(pos, rot_mat)


def determine_rcut(rvecs):
    """Determines the maximum allowed cutoff radius of rvecs

    The maximum cutoff radius should be determined based on the reduced cell

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    rvecs_ = rvecs.copy()
    if not is_reduced(rvecs_):
        reduce_box_vectors(rvecs_)
    return min([
            rvecs_[0, 0],
            rvecs_[1, 1],
            rvecs_[2, 2],
            ]) / 2


def reduce_box_vectors(rvecs):
    """Uses linear combinations of box vectors to obtain the reduced form

    The reduced form of a cell matrix is lower triangular, with additional
    constraints that enforce vector b to lie mostly along the y-axis and vector
    c to lie mostly along the z axis.

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows. These should already by in
        lower triangular form.

    """
    # simple reduction algorithm only works on lower triangular cell matrices
    assert is_lower_triangular(rvecs)
    # replace c and b with shortest possible vectors to ensure 
    # b_y > |2 c_y|
    # b_x > |2 c_x|
    # a_x > |2 b_x|
    rvecs[2, :] = rvecs[2, :] - rvecs[1, :] * np.round(rvecs[2, 1] / rvecs[1, 1])
    rvecs[2, :] = rvecs[2, :] - rvecs[0, :] * np.round(rvecs[2, 0] / rvecs[0, 0])
    rvecs[1, :] = rvecs[1, :] - rvecs[0, :] * np.round(rvecs[1, 0] / rvecs[0, 0])


def compute_lengths_angles(rvecs, degree=False):
    """Computes and returns the box vector lengths and angles"""
    lengths = np.linalg.norm(rvecs, axis=1)
    cosalpha = np.sum(rvecs[1, :] * rvecs[2, :]) / (lengths[1] * lengths[2])
    cosbeta  = np.sum(rvecs[0, :] * rvecs[2, :]) / (lengths[0] * lengths[2])
    cosgamma = np.sum(rvecs[1, :] * rvecs[0, :]) / (lengths[1] * lengths[0])
    if degree:
        conversion = np.pi / 180.0
    else:
        conversion = 1.0
    alpha = np.arccos(cosalpha) / conversion
    beta  = np.arccos(cosbeta ) / conversion
    gamma = np.arccos(cosgamma) / conversion
    return lengths, np.array([alpha, beta, gamma])


def do_gram_schmidt_reduction(rvecs):
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


def create_openmm_topology(system):
    """Creates OpenMM Topology from yaff System instance"""
    system.set_standard_masses()
    top = mm.app.Topology()
    chain = top.addChain()
    res = top.addResidue('res', chain)
    elements = []
    atoms = []
    for i in range(system.natom):
        mass = system.masses[i] / molmod.units.amu * unit.dalton
        elements.append(
                mm.app.Element.getByMass(mass),
                )
    for i in range(system.natom):
        element = elements[i]
        name = str(i)
        atoms.append(top.addAtom(
                name,
                element,
                res,
                ))
    for bond in system.bonds:
        top.addBond(atoms[bond[0]], atoms[bond[1]])
    u = molmod.units.nanometer # should be of type 'float', not 'unit'
    top.setPeriodicBoxVectors([
            system.cell._get_rvecs()[0] / u,
            system.cell._get_rvecs()[1] / u,
            system.cell._get_rvecs()[2] / u,
            ])
    return top


def add_header_to_config(path_yml):
    """Adds header with information to config file"""
    message = """
===============================================================================
===============================================================================

The configuration is divided into three parts:

  yaff:
      This section combines all settings related to the force field
      generation in YAFF. The available keywords depend on the specific
      system and force field.
      For purely covalent and nonperiodic systems, all information on the
      force field is contained in the parameter file and hence no additional
      settings need to be specified here. In all other cases, this section
      is nonempty and requires additional input from the user.


  conversion:
      This section combines settings related to the actual conversion
      or to additional parameters that are required by OpenMM.


  validations:
      This section contains all information regarding the validation
      procedure that should be performed after converting the force field.
      Each type of validation (e.g. singlepoint) has its own set of
      options explained below.

===============================================================================
==============================================================================="""
    lines = message.splitlines()
    pre = 'THIS FILE IS GENERATED BY OPENYAFF AT ' + str(datetime.now())
    lines = [pre] + lines
    for i in range(len(lines)):
        lines[i] = '#' + lines[i]
    with open(path_yml, 'r') as f:
        content = f.read()
    with open(path_yml, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')
        f.write(content)


def estimate_cell_derivative(positions, rvecs, energy_func, dh=1e-5,
        use_triangular_perturbation=False, evaluate_using_reduced=False):
    """Approximates the virial stress using a finite difference scheme

    Finite differences are only considered in nonzero cell matrix components.
    I.e. for lower triangular cells, upper triagonal zeros are not perturbed.

    Parameters
    ----------

    positions : array-like [angstrom]
        current atomic positions

    rvecs : array-like [angstrom]
        current rvecs

    energy_func : function
        function which accepts positions and rvecs as arguments (in angstrom)
        and returns the energy in kJ/mol

    dh : float [angstrom]
        determines box vector increments

    use_triangular_perturbation : bool
        determines whether finite differences are computed in all nine
        components or only in the six lower triangular components.

    """
    fractional = np.dot(positions, np.linalg.inv(rvecs))
    dUdh = np.zeros((3, 3))
    if use_triangular_perturbation:
        indices = [
                (0, 0),
                (1, 0), (1, 1),
                (2, 0), (2, 1), (2, 2),
                ]
    else:
        indices = [
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1), (1, 2),
                (2, 0), (2, 1), (2, 2),
                ]


    for i in range(3):
        for j in range(3):
            if (i, j) in indices:
                rvecs_ = rvecs.copy()
                rvecs_[i, j] -= dh / 2

                pos_tmp   = np.dot(fractional, rvecs_)
                rvecs_tmp = rvecs_.copy()
                if evaluate_using_reduced:
                    transform_lower_triangular(pos_tmp, rvecs_tmp)
                    reduce_box_vectors(rvecs_tmp)
                E_minus = energy_func(
                        pos_tmp,
                        rvecs_tmp,
                        )

                rvecs_[i, j] += dh
                pos_tmp   = np.dot(fractional, rvecs_)
                rvecs_tmp = rvecs_.copy()
                if evaluate_using_reduced:
                    transform_lower_triangular(pos_tmp, rvecs_tmp)
                    reduce_box_vectors(rvecs_tmp)
                E_pluss = energy_func(
                        pos_tmp,
                        rvecs_tmp,
                        )
                dUdh[i, j] = (E_pluss - E_minus) / dh
    return dUdh


def get_scale_index(definition):
    """Computes the scale index of a SCALE parameter definition"""
    scale_index = 0
    for line in definition.lines:
        # scaling is last part of string
        scale = float(line[1].split(' ')[-1])
        if scale == 0.0:
            scale_index += 1
    assert scale_index <= 3
    return scale_index


class Colors:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def log_header(name, logger, width=90, spacing=6):
    """Creates larger header with given title

    Parameters
    ----------

    name : str
        name to be displayed in the header

    logger : logging.Logger
        logger instance to be used

    width : int
        total width of the header, in number of characters

    spacing : int
        spacing around the title, in number of spaces

    """
    msg = str(name)
    msg += ' ' * (len(msg) % 2)
    leftright = (width - len(msg) - 2 * spacing) // 2

    first = '$' * width
    second = '$'*leftright + ' ' * (2 * spacing + len(msg)) + '$'*leftright
    third = '$'*leftright + ' '*spacing + msg + ' '*spacing + '$'*leftright
    fourth = second
    fifth = first
    logger.info(Colors.BOLD + first + Colors.END)
    logger.info(Colors.BOLD + second + Colors.END)
    logger.info(Colors.BOLD + third + Colors.END)
    logger.info(Colors.BOLD + fourth + Colors.END)
    logger.info(Colors.BOLD + fifth + Colors.END)

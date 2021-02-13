import logging
import tempfile
import yaff
import molmod
import yaml
import numpy as np
from datetime import datetime

from openyaff.utils import determine_rcut, transform_lower_triangular, \
        compute_lengths_angles, is_lower_triangular, is_reduced, \
        reduce_box_vectors, log_header
from openyaff.seeds import YaffSeed
from openyaff.generator import COVALENT_PREFIXES, DISPERSION_PREFIXES, \
        ELECTROSTATIC_PREFIXES
import openyaff.generator


logger = logging.getLogger(__name__) # logging per module


class Configuration:
    """Represents a configuration of a YAFF system and force field

    Class attributes
    ----------------

    properties : list of str
        list of all supported properties

    """
    properties = [
            'supercell',
            'rcut',
            'switch_width',
            'tailcorrections',
            'ewald_alphascale',
            'ewald_gcutscale',
            ]

    def __init__(self, system, pars):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object for which to generate a base configuration

        pars : str
            string containing the contents of a YAFF force field parameters file

        """
        self.periodic = (system.cell._get_nvec() == 3)
        if self.periodic:
            rvecs = system.cell._get_rvecs().copy()
            transform_lower_triangular(system.pos, rvecs, reorder=True)
            reduce_box_vectors(rvecs)
            assert is_reduced(rvecs)
            system.cell.update_rvecs(rvecs)
        self.system = system

        # generate parameters object to keep track of prefixes 
        with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tf:
            tf.write(pars)
        tf.close()
        parameters = yaff.Parameters.from_file(tf.name)
        self.parameters = parameters
        self.prefixes = [key for key, _ in parameters.sections.items()]

        # check whether all prefixes are supported
        for prefix in self.prefixes:
            _all = (COVALENT_PREFIXES +
                    DISPERSION_PREFIXES +
                    ELECTROSTATIC_PREFIXES)
            assert prefix in _all, 'I do not know prefix {}'.format(prefix)

        # use setters to initialize properties to default YAFF values
        # if properties are not applicable, they are initialized to None
        self.initialize_properties()

    def create_seed(self, kind='all'):
        """Creates a seed for constructing a yaff.ForceField object

        The returned seed contains all objects that are required in order to
        generate the force field object unambiguously. Specifically, this
        involves a yaff.System object, a yaff.FFArgs object with the correct
        parameter settings, and a yaff.Parameters object that contains the
        actual force field parameters.
        Because tests are typically performed on isolated parts of a force
        field, it is possible to generate different seeds corresponding
        to different parts -- this is done using the kind parameter. Allowed
        values are:

                - all:
                    generates the entire force field

                - covalent
                    generates only the covalent part of the force field.

                - nonbonded
                    generates only the nonbonded part of the force field,
                    including both dispersion and electrostatics.

                - dispersion
                    generates only the dispersion part of the force field,
                    which is basically the nonbonded part minus the
                    electrostatics.

                - electrostatic
                    generates only the electrostatic part

        Parameters
        ----------

        kind : str, optional
            specifies the kind of seed to be created. Allowed values are
            'all', 'covalent', 'nonbonded', 'dispersion', 'electrostatic'

        """
        assert kind in ['all', 'covalent', 'nonbonded', 'dispersion', 'electrostatic']
        parameters = self.parameters.copy()
        if kind == 'all':
            pass # do nothing, all prefixes should be retained
        elif kind == 'covalent':
            # pop all dispersion and electrostatic prefixes:
            for key in (DISPERSION_PREFIXES + ELECTROSTATIC_PREFIXES):
                parameters.sections.pop(key, None) # returns None if not present
        elif kind == 'nonbonded':
            # retain only dispersion and electrostatic
            sections = {}
            for key in (DISPERSION_PREFIXES + ELECTROSTATIC_PREFIXES):
                section = parameters.sections.get(key, None)
                if section is not None: # only add if present
                    sections[key] = section
            parameters = yaff.Parameters(sections)
        elif kind == 'dispersion':
            # retain only dispersion
            sections = {}
            for key in DISPERSION_PREFIXES:
                section = parameters.sections.get(key, None)
                if section is not None: # only add if present
                    sections[key] = section
            parameters = yaff.Parameters(sections)
        elif kind == 'electrostatic':
            # retain only electrostatic
            sections = {}
            for key in ELECTROSTATIC_PREFIXES:
                section = parameters.sections.get(key, None)
                if section is not None: # only add if present
                    sections[key] = section
            parameters = yaff.Parameters(sections)
        else:
            raise NotImplementedError(kind + ' not known.')

        # construct FFArgs instance and set properties
        ff_args = yaff.FFArgs()
        if self.periodic and tuple(self.supercell) != (1, 1, 1):
            # generate supercell based on reduced rvecs, and apply reduction
            # again if necessary
            system = self.system.supercell(*self.supercell)
            rvecs = system.cell._get_rvecs().copy()
            reduce_box_vectors(rvecs)
            system.cell.update_rvecs(rvecs)
        else:
            system = self.system

        if self.rcut is not None:
            ff_args.rcut = self.rcut * molmod.units.angstrom
        else:
            ff_args.rcut = 1e10 # humongous value; for nonperiodic systems

        if self.switch_width is not None and (self.switch_width != 0.0):
            ff_args.tr = yaff.Switch3(self.switch_width * molmod.units.angstrom)
        else:
            ff_args.tr = None

        if self.tailcorrections is not None:
            ff_args.tailcorrections = self.tailcorrections

        if self.ewald_alphascale is not None:
            ff_args.alpha_scale = self.ewald_alphascale

        if self.ewald_gcutscale is not None:
            ff_args.gcut_scale = self.ewald_gcutscale
        return YaffSeed(system, parameters, ff_args)

    def determine_supercell(self, rcut):
        """Determines the smallest supercell for which rcut is possible

        Since OpenMM does not allow particles to interact with their
        periodic copies, the maximum allowed interaction range (often equal to
        the cutoff range of the nonbonded interactions) is determined by the
        cell geometry. This function inspects the unit cell and supercells
        of the system to compute the maximum allowed rcut for each option.
        The supercell tuple is constructed based on the *reduced form* of the
        initial cell as stored in the system .chk.

        Parameters
        ----------

        rcut : float [angstrom]
            desired cutoff radius

        """
        rcut *= molmod.units.angstrom
        rvecs = self.system.cell._get_rvecs()
        assert is_lower_triangular(rvecs)
        assert is_reduced(rvecs)
        current_rcut = 0
        i, j, k = (1, 1, 1)
        while (k < 20) and (current_rcut < rcut): # c vector is last
            j = 1
            while (j < 20) and (current_rcut < rcut): # b vector second 
                i = 1
                while (i < 20) and (current_rcut < rcut): # a vector first
                    supercell = (i, j, k)
                    rvecs_ = np.array(supercell)[:, np.newaxis] * rvecs
                    try:
                        # compute reduced form to evaluate max rcut
                        transform_lower_triangular(
                                np.zeros((1, 3)), # dummy pos
                                rvecs_,
                                reorder=True,
                                )
                        current_rcut = determine_rcut(rvecs_)
                    except ValueError:
                        pass # invalid box vectors, move on to next
                    i += 1
                j += 1
            k += 1
        return list(supercell)

    def get_prefixes(self, kind):
        """Returns the prefixes belonging to a specific kind

        Parameters
        ----------

        kind : str
            kind of interactions. This is either 'covalent', 'dispersion',
            'electrostatic', 'nonbonded', 'all'.

        """
        prefixes = []
        if kind == 'covalent':
            target = COVALENT_PREFIXES
        elif kind == 'dispersion':
            target = DISPERSION_PREFIXES
        elif kind == 'electrostatic':
            target = ELECTROSTATIC_PREFIXES
        elif kind == 'nonbonded':
            target = DISPERSION_PREFIXES + ELECTROSTATIC_PREFIXES
        elif kind == 'all':
            target = (COVALENT_PREFIXES +
                      DISPERSION_PREFIXES +
                      ELECTROSTATIC_PREFIXES)
        for prefix in self.prefixes:
            if prefix in target:
                prefixes.append(prefix)
        return prefixes

    def log_config(self):
        """Logs information about the current configuration"""
        log_header('force field configuration', logger)
        logger.info('')
        logger.info('')
        if self.periodic:
            natom     = np.prod(np.array(self.supercell)) * self.system.natom
        else:
            natom = self.system.natom
        config = {}
        for name in self.properties:
            value = getattr(self, name)
            if value is not None: # if property is applicable
                config[name] = value
        config['number of atoms'] = natom
        min_length = max([len(key) for key, _ in config.items()])
        spacing = 3
        for key, value in config.items():
            line = key
            line += ' ' * (min_length - len(key) + spacing)
            line += ':'
            line += ' ' * spacing
            line += '{}'.format(value)
            logger.info(line)
        logger.info('')
        logger.info('')

    def log_system(self):
        """Logs information about this system"""
        log_header('system information', logger)
        logger.info('')
        logger.info('')
        natom = self.system.natom
        if self.periodic:
            system_type = 'periodic'
        else:
            system_type = 'non-periodic'
        logger.info('number of atoms:     {}'.format(natom))
        logger.info('system type:         ' + system_type)
        logger.info('')
        if self.periodic:
            rvecs = self.system.cell._get_rvecs() / molmod.units.angstrom
            transform_lower_triangular(
                    np.zeros((1, 3)), # dummy pos
                    rvecs,
                    reorder=True,
                    )
            lengths, angles = compute_lengths_angles(rvecs, degree=True)
            logger.info('reduced (!) box vectors (in angstrom):')
            logger.info('\ta: {}'.format(rvecs[0, :]))
            logger.info('\tb: {}'.format(rvecs[1, :]))
            logger.info('\tc: {}'.format(rvecs[2, :]))
            logger.info('')
            logger.info('reduced box lengths (in angstrom):')
            logger.info('\ta: {:.4f}'.format(lengths[0]))
            logger.info('\tb: {:.4f}'.format(lengths[1]))
            logger.info('\tc: {:.4f}'.format(lengths[2]))
            logger.info('')
            logger.info('reduced box angles (in degrees):')
            logger.info('\talpha: {:.4f}'.format(angles[0]))
            logger.info('\tbeta : {:.4f}'.format(angles[1]))
            logger.info('\tgamma: {:.4f}'.format(angles[2]))
            logger.info('')
        logger.info('found {} prefixes:'.format(len(self.prefixes)))
        for prefix in self.prefixes:
            logger.info('\t' + prefix)
        logger.info('')
        logger.info('')

    def write(self, path_config=None):
        """Generates the .yml contents and optionally saves it to a file

        If the file already exists, then the contents of the 'yaff' key are
        overwritten with the current values

        Parameters
        ----------

        path_config : pathlib.Path, optional
            specifies the location of the output .yml file

        """
        config = {}
        for name in self.properties:
            value = getattr(self, name)
            if value is not None: # if property is applicable
                config[name] = value

        final = {'yaff': config}
        if path_config is not None:
            assert path_config.suffix == '.yml'
            if path_config.exists():
                # load contents and look for 'yaff' to replace
                with open(path_config, 'r') as f:
                    loaded_config = yaml.load(f, Loader=yaml.FullLoader)
                    # dict needs to be initialized when reading 'empty' file
                    if loaded_config is None:
                        loaded_config = {}
                    loaded_config['yaff'] = config
                final = loaded_config
            with open(path_config, 'w') as f:
                yaml.dump(final, f, default_flow_style=False)
        return final

    @staticmethod
    def from_files(path_system, path_pars, path_config=None):
        """Constructs a Configuration based on system and parameter files

        If, optionally, a config .yml file is specified, than the settings in
        that file are verified and loaded into the newly created object

        Parameters
        ----------

        path_system : pathlib.Path
            specifies the location of the YAFF .chk file

        path_pars : pathlib.Path
            specifies the location of the force field parameters .txt file

        path_config : pathlib.Path, optional
            specifies the location of the .yml configuration file

        """
        # load system and generate generic force field
        system = yaff.System.from_file(str(path_system))
        with open(path_pars, 'r') as f:
            pars = f.read()
        configuration = Configuration(system, pars)

        if path_config: # loads .yml contents and calls update_properties()
            with open(path_config, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            configuration.update_properties(config)
        return configuration

    def initialize_properties(self):
        """Initializes properties to YAFF defaults"""
        try:
            self.supercell = [1, 1, 1] # default supercell setting
        except ValueError:
            pass

        try:
            self.rcut = yaff.FFArgs().rcut / molmod.units.angstrom
        except ValueError:
            pass

        try:
            self.switch_width = 4.0 # 4.0 angstrom default
        except ValueError:
            pass

        try:
            self.tailcorrections = yaff.FFArgs().tailcorrections
        except ValueError:
            pass

        try:
            self.ewald_alphascale = yaff.FFArgs().alpha_scale
        except ValueError:
            pass

        try:
            self.ewald_gcutscale = yaff.FFArgs().gcut_scale
        except ValueError:
            pass

    def update_properties(self, config):
        """Updates property values based on values in .yml file

        Parameters
        ----------

        config : dict
            dictionary, e.g. loaded from .yml file
        """
        determine_supercell = False
        for name in self.properties:
            if name in config['yaff'].keys():
                if name == 'supercell': # special treatment
                    if tuple(config['yaff'][name]) == (-1, -1, -1):
                        determine_supercell = True
                        config['yaff'][name] = [1, 1, 1] # dummy
                # following should not raise anything
                setattr(self, name, config['yaff'][name])

        if determine_supercell: # determine supercell after all properties are set
            self.supercell = self.determine_supercell(self.rcut)

    @staticmethod
    def annotate(path_yml):
        """Annotates a .yml file with comments regarding the current system"""
        message = """ YAFF

    Below is a list of possible keywords for this section. Their applicability
    depends on the specific system (whether it is periodic, whether it contains
    charges, ...). Some keywords that appear in the list below may therefore
    not actually be valid for this specific configuration.

    supercell:
        determines the supercell to use. OpenMM does not allow interactions
        to reach further than half the *shortest* cell vector in its reduced
        representation. NOTE: THE SUPERCELL TUPLE REFERS TO THE CELL VECTORS
        IN THE REDUCED REPRESENTATION, NOT TO THE ORIGINAL CELL VECTORS.
        For most nanoporous materials and cutoff ranges (10 - 15 angstrom),
        this implies that validation with OpenMM is almost always performed on
        supercells of those in the .chk input.
        By default, OpenYAFF will suggest the smallest possible unit cell that
        is compatible with the default cutoff distance. Other supercells and
        their corresponding largest cutoff distance are logged when executing
        `openyaff configure`.
        (see http://docs.openmm.org/latest/userguide/theory.html#periodic-boundary-conditions
        for more information on the reduced cell representation in OpenMM).
        )

    rcut:
        cutoff distance of the nonbonded interactions.
        (default: 10 angstrom)

    switch_width:
        distance over which dispersion interactions are smoothed to 0.
        (default: 4 angstrom)

    tailcorrections:
        whether to use tailcorrections for the dispersion interactions.
        (default: False)

    ewald_alphascale:
        alpha scale parameter for the ewald summation in case of periodic
        electrostatics. (see yaff.pes.generator.FFArgs for more information
        and default values)

    ewald_gcutscale:
        gcut scale parameter for the ewald summation in case of periodic
        electrostatics. (see yaff.pes.generator.FFArgs for more information
        and default values)"""
        comments = message.splitlines()
        for i in range(len(comments)):
            comments[i] = '#' + comments[i]
        comments = ['\n\n'] + comments

        with open(path_yml, 'r') as f:
            content = f.read()
        lines = content.splitlines()

        index = None
        for i, line in enumerate(lines):
            if line.startswith('yaff'):
                assert index is None
                index = i

        assert index is not None
        lines = lines[:index] + comments + lines[index:]
        with open(path_yml, 'w') as f:
            f.write('\n'.join(lines))

    @property
    def supercell(self):
        """Returns the supercell tuple"""
        return self._supercell

    @supercell.setter
    def supercell(self, value):
        if self.periodic:
            assert len(value) == 3        # systems are 3D periodic
            self._supercell = list(value) # store as list because of pyyaml
        else: # property not applicable
            self._supercell = None
            raise ValueError('Cannot set supercell because system is aperiodic')

    @property
    def rcut(self):
        """Returns rcut in angstrom"""
        return self._rcut

    @rcut.setter
    def rcut(self, value):
        """Sets the rcut parameter

        A ValueError is raised if no nonbonded interactions are found

        Parameters
        ----------

        value : float [angstrom]
            desired cutoff radius

        """
        # rcut applicable only to nonbonded force parts:
        if (self.periodic and (len(self.get_prefixes('nonbonded')) > 0)):
            assert type(value) == float
            self._rcut = value
            return True
        else: # property not applicable
            self._rcut = None
            raise ValueError('Cannot set rcut for this system')

    @property
    def switch_width(self):
        """Returns the width of the switching function
        """
        return self._switch_width

    @switch_width.setter
    def switch_width(self, value):
        """Sets the width of the switching function

        A ValueError is raised if no nonbonded interactions are found or if
        the system is not periodic.

        Parameters
        ----------

        value : float [angstrom]
            desired width of switching function. A width of zero implies a hard
            cutoff. (YAFF default value is 4 angstrom)

        """
        if (self.periodic and (len(self.get_prefixes('dispersion')) > 0)):
            assert type(value) == float
            self._switch_width = value
            return True
        else: # property not applicable
            self._switch_width = None
            raise ValueError('Cannot set switch width.')

    @property
    def tailcorrections(self):
        """Returns whether tailcorrections are enabled"""
        return self._tailcorrections

    @tailcorrections.setter
    def tailcorrections(self, value):
        """Sets the tailcorrections parameter

        A ValueError is raised if no dispersion nonbonded interactions are found

        Parameters
        ----------

        value : bool
            enables or disables tail corrections

        """
        # tailcorrections apply only to dispersion nonbonded force parts in
        # periodic systems:
        if (self.periodic and (len(self.get_prefixes('dispersion')) > 0)):
            assert type(value) == bool
            self._tailcorrections = value
            return True
        else: # property not applicable
            self._tailcorrections = None
            raise ValueError('Cannot set tailcorrections for this system')

    @property
    def ewald_alphascale(self):
        """Returns the alpha_scale parameter for the ewald summation"""
        return self._ewald_alphascale

    @ewald_alphascale.setter
    def ewald_alphascale(self, value):
        """Sets the alpha_scale parameter for the ewald summation

        A ValueError is raised if no Ewald sum is present in the force field

        Parameters
        ----------

        value : float, dimensionless
            desired alpha scale

        """
        # ewald parameters apply only to periodic systems with electrostatics
        if ('FIXQ' in self.prefixes) and self.periodic:
            assert type(value) == float
            self._ewald_alphascale = value
        else:
            self._ewald_alphascale = None
            raise ValueError('Cannot set ewald parameters for this system')

    @property
    def ewald_gcutscale(self):
        """Returns the gcut_scale parameter for the ewald summation"""
        return self._ewald_gcutscale

    @ewald_gcutscale.setter
    def ewald_gcutscale(self, value):
        """Sets the gcut_scale parameter for the ewald summation

        A ValueError is raised if no Ewald sum is present in the force field

        Parameters
        ----------

        value : float, dimensionless
            desired gcut scale

        """
        # ewald parameters apply only to periodic systems with electrostatics
        if ('FIXQ' in self.prefixes) and self.periodic:
            assert type(value) == float
            self._ewald_gcutscale = value
        else:
            self._ewald_gcutscale = None
            raise ValueError('Cannot set ewald parameters for this system')

import logging
import tempfile
import yaff
import molmod

from configparser import ConfigParser

from openyaff.utils import determine_rcut


logger = logging.getLogger(__name__) # logging per module


class Configuration:
    """Represents a configuration of a YAFF system and force field

    Class attributes
    ----------------

    config_layout : dict

    """
    config_layout = { # specifies properties and their locations in config
            'rcut': ('yaff', 'rcut'),
            'tailcorrections': ('yaff', 'tailcorrections'),
            'ewald_parameters': ('yaff', 'ewald_parameters'),
            }

    def __init__(self, system, pars):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object for which to generate a base configuration

        pars : str
            string containing the contents of a YAFF force field parameters file

        """
        # sanity check to ensure force field is nonempty
        with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tf:
            tf.write(pars)
        tf.close()
        parameters = yaff.Parameters.from_file(tf.name)
        ff = yaff.ForceField.generate(system, parameters)
        assert ff.compute() != 0.0

        self.forcefield = ff
        self.parameters = parameters
        self.periodic   = (ff.system.cell._get_nvec() == 3)
        self.prefixes = [key for key, _ in parameters.sections.items()]

        # use setters to initialize properties to default YAFF values
        # if properties are not applicable, they are initialized to None
        try:
            self.rcut = yaff.FFArgs().rcut / molmod.units.angstrom
        except ValueError:
            pass

        try:
            self.tailcorrections = yaff.FFArgs().tailcorrections
        except ValueError:
            pass

        try:
            self.ewald_parameters = (yaff.FFArgs().alpha_scale,
                    yaff.FFArgs().gcut_scale)
        except ValueError:
            pass

        # TODO
        # save .ini file
        # create system, ffargs for force field generation
        # evaluate method

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
        if (('MM3' in self.prefixes) or ('LJ' in self.prefixes)  or
            ('FIXQ' in self.prefixes)):
            assert type(value) == float
            self._rcut = value
            return True
        else: # property not applicable
            self._rcut = None
            raise ValueError('Cannot set rcut for this system')

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
        # tailcorrections apply only to dispersion nonbonded force parts:
        if ('MM3' in self.prefixes or 'LJ' in self.prefixes):
            assert type(value) == bool
            self._tailcorrections = value
            return True
        else: # property not applicable
            self._tailcorrections = None
            raise ValueError('Cannot set tailcorrections for this system')

    @property
    def ewald_parameters(self):
        """Returns ewald parameters (alpha_scale, gcut_scale)"""
        return self._ewald_parameters

    @ewald_parameters.setter
    def ewald_parameters(self, value):
        """Sets the ewald parameters

        A ValueError is raised if no Ewald sum is present in the force field

        Parameters
        ----------

        value : tuple of float
            tuple of floats (alpha_scale, gcut_scale)

        """
        # ewald parameters apply only to periodic systems with electrostatics
        if 'FIXQ' in self.prefixes and self.periodic:
            assert len(value) == 2 # value is tuple of (alpha, gcut)
            self._ewald_parameters = value
        else:
            self._ewald_parameters = None
            raise ValueError('Cannot set ewald parameters for this system')

    def determine_supercell(self, rcut):
        """Determines the smallest supercell for which rcut is possible

        Since OpenMM does not allow particles to interact with their
        periodic copies, the maximum allowed interaction range (often equal to
        cutoff range of the nonbonded interactions) is determined by the cell
        geometry. This evaluates the current cell and supercells to compute
        the maximum allowed rcut for each option.

        Parameters
        ----------

        rcut : float
            desired cutoff radius

        """
        rvecs = self.forcefield.system.cell._get_rvecs()
        current_rcut = 0
        i, j, k = (1, 1, 1)
        while k < 10: # c vector is last to increase
            while j < 10: # b vector second to increase
                while i < 10: # a vector first to increase
                    while current_rcut < rcut:
                        supercell = (i, j, k)
                        rvecs_ = np.array(supercell)[:, np.newaxis] * rvecs
                        try:
                            current_rcut = determine_rcut(reduced)
                        except ValueError:
                            pass # invalid box vectors, move on to next
        return supercell


    def log(self):
        """Logs information about this configuration"""
        pass

    def write(self, path_config=None):
        """Generates the .ini contents and optionally saves it to a file

        Parameters
        ----------

        path_config : pathlib.Path, optional
            specifies the location of the output .ini file

        """
        config = ConfigParser()
        for name, location in self.config_layout.items():
            assert len(location) == 2 # no nested ini
            config[location[0]] = {} # initialize dicts
        for name, location in self.config_layout.items():
            value = getattr(self, name)
            if value is not None: # if property is applicable
                config[location[0]][location[1]] = str(value)

        if path_config is not None:
            assert path_config.suffix == '.ini'
            with open(path_config, 'w') as f:
                config.write(f)
        return config


    @staticmethod
    def from_files(path_system, path_pars):
        """Initializes a config .ini file with default values

        The system and force field objects are created, and the relevant
        key value pairs are added to the configparser and saved to an .ini
        file specified by path_config.

        Parameters
        ----------

        path_system : pathlib.Path
            specifies the location of the YAFF .chk file

        path_pars : pathlib.Path
            specifies the location of the force field parameters .txt file

        """
        # load system and generate generic force field
        system = yaff.System.from_file(str(path_system))
        with open(path_pars, 'r') as f:
            pars = path_pars.read()
        configuration = Configuration(system, pars)
        return configuration

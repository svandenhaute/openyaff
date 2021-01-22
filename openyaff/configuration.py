import logging
import tempfile
import configparser
import yaff


logger = logging.getLogger(__name__) # logging per module


class Configuration:
    """Represents a configuration of a YAFF system and force field

    Class attributes
    ----------------

    yaff_defaults : dict
        loads the default settings of the force field generator in YAFF

    supported_config_entries : list of str
        enumerates allowed keys in the YAFF section of the config

    """
    yaff_defaults = {
            'rcut': yaff.FFArgs().rcut,
            'alpha_scale': yaff.FFArgs().alpha_scale,
            'gcut_scale': yaff.FFArgs().gcut_scale,
            'skin': yaff.FFArgs().skin,
            'smooth_ei': yaff.FFArgs().smooth_ei,
            'reci_ei': yaff.FFArgs().reci_ei,
            'nlow': yaff.FFArgs().nlow,
            'nhigh': yaff.FFArgs().nhigh,
            'tailcorrections': yaff.FFArgs().tailcorrections,
            }
    supported_config_entries = [
            'rcut',             # cutoff radius of real space ei and dispersion
            'supercell',        # supercell to use
            'tailcorrections',  # tailcorrections for dispersion
            #'ewald_params',     # parameters of ewald summation
            ]

    def __init__(self, system, pars, config=None):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object for which to generate a base configuration

        pars : str
            string containing the contents of a YAFF force field parameters file

        config : ConfigParser, optional
            contains existing configuration

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

        # generate default config for this system
        default_config = configparser.ConfigParser()
        part_names = [part.name for part in ff.parts]
        prefixes   = [key for key, _ in parameters.sections.items()]
        if ('MM3' in prefixes) or ('LJ' in prefixes):
            self.set_rcut(default_config)
            self.set_tailcorrections(default_config)
        if ('FIXQ' in prefixes) and (ff.system.cell._get_nvec() == 3):
            self.set_ewald_parameters(default_config)


        # TODO
        # list of force parts -> generate force fields for each separately
        # inspect force field and determine which options are appropriate
        # save .ini file
        # cell alignment options with supercell
        # create system, ffargs for force field generation
        # evaluate method

    def set_rcut(self, config):
        """Adds entries for rcut and supercell settings

        Since OpenMM does not allow particles to interact with their
        periodic copies, the maximum allowed interaction range (often equal to
        cutoff range of the nonbonded interactions) is determined by the cell
        geometry. This evaluates the current cell and its supercells to compute
        the maximum allowed rcut for each option.

        Parameters
        ----------

        config : ConfigParser
            config parser for which to add this option

        """
        pass

    def set_tailcorrections(self, default_config):
        """Adds entry to enable tail corrections (default: off)

        Parameters
        ----------

        config : ConfigParser
            config parser for which to add this option

        """
        pass

    def set_ewald_parameters(self, default_config):
        """Adds entry for ewald summation parameters

        Parameters
        ----------

        config : ConfigParser
            config parser for which to add this option

        """
        pass


    def log(self):
        """Logs information about this configuration"""
        pass

    def write(self):
        """Generates the .ini contents and optionally saves it to a file

        Parameters
        ----------

        path_config : pathlib.Path
            specifies the location of the output .ini file

        """
        with open(path_config, 'w') as f:
            f.write(self.config)

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

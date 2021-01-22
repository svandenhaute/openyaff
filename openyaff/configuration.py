import tempfile
import yaff

class Configuration:
    """Represents a configuration of a YAFF system and force field"""

    def __init__(self, system, pars):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object for which to generate a base configuration

        pars : str
            string containing the contents of a YAFF force field parameters file

        """
        self.system = system
        self.pars   = pars

        # write pars to temporary file and generate ForceField
        with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tf:
            tf.write(pars)
        tf.close()
        ff = yaff.ForceField.generate(system, [tf.name])
        assert ff.compute() != 0.0

    @staticmethod
    def configure(path_system, path_pars, path_config):
        """Initializes a configuration .ini file

        Parameters
        ----------

        path_system : Path
            specifies the location of the YAFF .chk file

        path_pars : Path
            specifies the location of the force field parameters .txt file

        path_config : Path
            specifies the location of the output .ini file that will be created

        """
        # load system and generate generic force field
        system = yaff.System.from_file(str(path_system))
        with open(path_pars, 'r') as f:
            pars = path_pars.read()
        configuration = Configuration(system, pars)


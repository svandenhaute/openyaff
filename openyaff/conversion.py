import yaml
import numpy as np

from openyaff.utils import create_openmm_system
from openyaff.generator import AVAILABLE_PREFIXES, apply_generators_mm
from openyaff.seeds import OpenMMSeed


class Conversion:
    """Base class for conversion objects"""
    kind = None
    properties = []

    def __init__(self):
        """Constructor; should initialize properties to default values"""
        pass

    def write(self, path_config=None):
        """Generates the .yml contents and optionally saves it to a file

        If the file already exists, then the contents of the 'conversion' key
        are overwritten with the current values

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
        config['kind'] = self.kind

        final = {'conversion': config}
        if path_config is not None:
            assert path_config.suffix == '.yml'
            if path_config.exists():
                # load contents and look for 'conversion' to replace
                with open(path_config, 'r') as f:
                    loaded_config = yaml.load(f, Loader=yaml.FullLoader)
                    loaded_config['conversion'] = config
                final = loaded_config
            with open(path_config, 'w') as f:
                yaml.dump(final, f, default_flow_style=False)
        return final

    @staticmethod
    def annotate(path_yml):
        """Annotates a .yml file with comments regarding the current system"""
        message = """ CONVERSION

        Below is a list of possible keywords for this section.

        kind:
            specifies the kind of conversion to apply. Currently, only
            explicit conversions are supported.
            (default: explicit)

        pme_error_thres:
            specifies the error threshold for the PME evaluation (only
            relevant for periodic systems). For more information on the PME
            evaluation, see openmm.org."""
        comments = message.splitlines()
        for i in range(len(comments)):
            comments[i] = '#' + comments[i]
        comments = ['\n\n'] + comments

        with open(path_yml, 'r') as f:
            content = f.read()
        lines = content.splitlines()

        index = None
        for i, line in enumerate(lines):
            if line.startswith('conversion'):
                assert index is None
                index = i

        assert index is not None
        lines = lines[:index] + comments + lines[index:]
        with open(path_yml, 'w') as f:
            f.write('\n'.join(lines))


class ExplicitConversion(Conversion):
    """Defines the explicit conversion procedure from YAFF to OpenMM

    In this procedure, each YAFF generator is extended with function calls
    to an OpenMM system object such that the force field generation occurs
    synchronously between OpenMM and YAFF.

    """
    kind = 'explicit'
    properties = [
            'pme_error_thres',
            ]

    def __init__(self, pme_error_thres=1e-5):
        self.pme_error_thres = pme_error_thres

    def check_compatibility(self, configuration):
        """Checks compatibility of current settings with a given configuration

        The following checks are performed:

            generator availability:
                all prefixes in the configuration should have a generator in
                openyaff.generator

            noninteger scalings:
                noninteger scalings in the dispersion or electrostatics is
                currently not supported

        Parameters
        ----------

        configuration : openyaff.Configuration
            configuration for which to check compatibility

        """
        # check available generators
        for key, _ in configuration.parameters.sections.items():
            assert key in AVAILABLE_PREFIXES, ('I do not have a generator '
                    'for key {}'.format(key))
        nonbonded_prefixes = []
        nonbonded_prefixes += configuration.dispersion_prefixes
        nonbonded_prefixes += configuration.electrostatic_prefixes
        for key, _ in configuration.parameters.sections.items():
            if key in nonbonded_prefixes: # can contain scalings
                definition = _['SCALE']
                # line == [line_no, line_content]
                for line in definition.lines:
                    # scaling is last part of string
                    scale = float(line[1].split(' ')[-1])
                    assert (scale == 0.0) or (scale == 1.0)

    def apply(self, configuration, seed_kind='full'):
        """Converts a yaff configuration into an OpenMM seed

        Begins with a call to check_compatibility, and an assertion error is
        raised if the configuration is not compatible.
        The system object in the yaff seed is used to create a corresponding
        openmm system object, after which apply_generators_mm is called.

        Parameters
        ----------

        configuration : openyaff.Configuration
            configuration for which to check compatibility

        seed_kind : str
            specifies the kind of seed to be converted

        """
        # raise AssertionError if not compatible
        self.check_compatibility(configuration)
        yaff_seed = configuration.create_seed(kind=seed_kind)
        system_mm = create_openmm_system(yaff_seed.system)
        kwargs = {}

        # if system is periodic and contains electrostatis; compute PME params
        if (configuration.ewald_alphascale is not None and
                seed_kind in ['full', 'electrostatic', 'nonbonded']):
            alpha = configuration.ewald_alphascale
            delta = np.exp(-(alpha) ** 2) / 2
            if delta > self.pme_error_thres:
                kwargs['delta'] = delta
            else:
                kwargs['delta'] = self.pme_error_thres
        apply_generators_mm(yaff_seed, system_mm, **kwargs)
        openmm_seed = OpenMMSeed(system_mm)
        return openmm_seed

    @property
    def pme_error_thres(self):
        """Returns the error threshold for the PME calculation"""
        return self._pme_error_thres

    @pme_error_thres.setter
    def pme_error_thres(self, value):
        self._pme_error_thres = value


def load_conversion(path_config):
    with open(path_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    assert 'conversion' in list(config.keys())
    config_conversion = config['conversion']
    assert 'kind' in list(config_conversion.keys()), ('the configuration file'
            ' should specify the kind of conversion to use')
    conversion_cls = {}
    for x in list(globals().values()):
        if isinstance(x, type) and issubclass(x, Conversion):
            conversion_cls[x.kind] = x
    assert config_conversion['kind'] in list(conversion_cls.keys())
    kind = config_conversion.pop('kind')
    return conversion_cls[kind](**config_conversion)

import yaml
import numpy as np

from openyaff.utils import create_openmm_system
from openyaff.generator import AVAILABLE_PREFIXES, apply_generators_mm
from openyaff.seeds import OpenMMSeed


class Conversion:
    """Base class for conversion objects"""
    kind = None


class ExplicitConversion(Conversion):
    """Defines the explicit conversion procedure from YAFF to OpenMM

    In this procedure, each YAFF generator is extended with function calls
    to an OpenMM system object such that the force field generation occurs
    synchronously between OpenMM and YAFF.

    """
    kind = 'explicit'

    def __init__(self, pme_error_thres=1e-5):
        """Constructor

        Parameters
        ----------

        pme_error_thres : float
            determines the threshold for the error tolerance in the
            NonbondedForce PME evaluation

        """
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


def load_conversion(path_ini):
    with open(path_ini, 'r') as f:
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

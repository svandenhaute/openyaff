import yaml
import logging

from openyaff.wrappers import YaffForceFieldWrapper, OpenMMForceFieldWrapper


logger = logging.getLogger(__name__) # logging per module


class Validation:
    """Base class to implement validation procedures of conversions

    Class attributes
    ----------------

    """
    name = None
    wrapper_class_yaff = None
    wrapper_class_openmm = None

    def __init__(self, platforms=['Reference'], separate_parts=True):
        """Constructor

        Parameters
        ----------

        platforms : list of str
            determines the OpenMM platforms for which to run the tests

        separate_parts : bool
            specifies whether this test should be performed for each
            part separately

        """
        self.platforms = platforms
        self.separate_parts = separate_parts

    def run(self, configuration, conversion):
        """Validates the conversion of a given configuration

        Depending on the number of platforms to validate and whether individual
        force parts are treated separately, different wrappers are involved in
        the validation. They are stored in a dictionary with a layout like this:

            'yaff':
                'covalent': wrapper
                'dispersion': wrapper
                'electrostatic': wrapper

            'openmm':
                ('covalent', 'Reference'): wrapper
                ('dispersion', 'Reference'): wrapper
                ('electrostatic', 'Reference'): wrapper

        If parts are not separated, then it could for example be given by

            'yaff':
                'full': wrapper

            'openmm':
                ('full', CUDA): wrapper

        The actual validation is performed by the _internal_validate method
        which is implemented by subclasses.

        Parameters
        ----------

        configuration : openyaff.Configuration
            configuration of the YAFF force field to convert and validate

        conversion : openyaff.Conversion
            conversion instance that creates the OpenMM and YAFF seeds which
            should be validated against each other.

        """
        wrappers = {
                'yaff': {},
                'openmm': {},
                }
        results = {
                'yaff': {},
                'openmm': {},
                }
        self.log()
        for platform in self.platforms:
            if self.separate_parts: # generate wrapper for each part of the FF
                seed_kinds = ['covalent', 'dispersion', 'electrostatic']
            else:
                seed_kinds = ['full']
            for kind in seed_kinds:
                seed_yaff = configuration.create_seed(kind=kind)
                seed_mm = conversion.apply(configuration, seed_kind=kind)
                wrapper_yaff = self.wrapper_class_yaff.from_seed(seed_yaff)
                wrapper_mm = self.wrapper_class_openmm.from_seed(
                        seed_mm,
                        platform,
                        )
                result_yaff, result_mm = self._internal_validate(
                        wrapper_yaff,
                        wrapper_mm,
                        )
                wrappers['yaff'][kind] = wrapper_yaff
                wrappers['openmm'][kind] = wrapper_mm

        successful = self.parse_results(results) # does logging and comparison
        return successful

    def log(self):
        """Logs information prior to running the validation"""
        logger.info('')
        n = 20
        logger.info('=' * n + '  ' + self.name + '  ' + '=' * n)
        logger.info('')
        logger.info('validating the following OpenMM platforms:')
        for platform in self.platforms:
            logger.info('\t - ' + platform)
        if self.separate_parts:
            logger.info('covalent, dispersion and electrostatic contributions'
                    'are validated separately')

    def _internal_validate(self, wrapper_yaff, wrapper_mm):
        """Performs validation and returns a dictionary with results

        Parameters
        ----------

        wrapper_yaff : YaffForceFieldWrapper

        wrapper_mm : OpenMMForceFieldWrapper

        """
        raise NotImplementedError


class SinglePointValidation(Validation):
    """Implements a single point validation of energy and forces"""

    name = 'singlepoint'
    wrapper_class_yaff   = YaffForceFieldWrapper
    wrapper_class_openmm = OpenMMForceFieldWrapper


def load_validations(path_yml):
    with open(path_yml, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    configs_validation = config['validations'] # list of configs

    validation_cls = {}
    for x in list(globals().values()):
        if isinstance(x, type) and issubclass(x, Validation):
            validation_cls[x.name] = x

    validations = []
    for _ in configs_validation:
        name = list(_.keys())[0]
        kwargs = _[name]
        assert name in list(validation_cls.keys())
        validations.append(
                validation_cls[name](**kwargs),
                )
    return validations

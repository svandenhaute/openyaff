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
    properties = [
            'platforms',
            'separate_parts',
            ]

    def __init__(self, platforms=['Reference'], separate_parts=True, **kwargs):
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
        for key, value in kwargs.items():
            assert key in self.properties
            setattr(self, key, value)

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
        self.log()
        for platform in self.platforms:
            if self.separate_parts: # generate wrapper for each part of the FF
                seed_kinds = ['covalent', 'dispersion', 'electrostatic']
            else:
                seed_kinds = ['full']
            for kind in seed_kinds:
                self._internal_validate(platform, kind)

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

        final = {'validations': {self.name: config}}
        if path_config is not None:
            assert path_config.suffix == '.yml'
            if path_config.exists():
                # load contents and look for 'yaff' key, replace contents
                with open(path_config, 'r') as f:
                    loaded_config = yaml.load(f, Loader=yaml.FullLoader)
                if 'validations' in loaded_config.keys():
                    loaded_config['validations'][self.name] = config
                else:
                    loaded_config['validations'] = {self.name: config}
                final = loaded_config
            with open(path_config, 'w') as f:
                yaml.dump(final, f, default_flow_style=False)
        return final

    @property
    def platforms(self):
        return self._platforms

    @platforms.setter
    def platforms(self, value):
        assert isinstance(value, list)
        for key in value:
            assert key in ['Reference', 'CPU', 'CUDA', 'OpenCL']
        self._platforms = list(value)

    @property
    def separate_parts(self):
        return self._separate_parts

    @separate_parts.setter
    def separate_parts(self, value):
        assert isinstance(value, bool)
        self._separate_parts = value

    @staticmethod
    def annotate(path_yml):
        raise NotImplementedError


class SinglePointValidation(Validation):
    """Implements a single point validation of energy and forces"""

    name = 'singlepoint'

    @staticmethod
    def annotate(path_yml):
        """Annotates a .yml file with comments regarding the current system"""
        message = """ VALIDATION

        The validation generally consists of a series of individual validation
        experiments. Each experiment (e.g. a single point or stress validation)
        has its own keywords.

        singlepoint:
            performs a series of single point calculations on randomly generated
            states, and compares forces and energies. Allowed keywords for this
            experiment are:

                tol:
                    relative tolerance on energy and forces between YAFF and
                    OpenMM.
                    (default: 1e-5)"""
        comments = message.splitlines()
        for i in range(len(comments)):
            comments[i] = '#' + comments[i]
        comments = ['\n\n'] + comments

        with open(path_yml, 'r') as f:
            content = f.read()
        lines = content.splitlines()

        index = None
        for i, line in enumerate(lines):
            if line.startswith('validations'):
                assert index is None
                index = i

        assert index is not None
        lines = lines[:index] + comments + lines[index:]
        with open(path_yml, 'w') as f:
            f.write('\n'.join(lines))


def load_validations(path_yml):
    with open(path_yml, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    validation_cls = {}
    for x in list(globals().values()):
        if isinstance(x, type) and issubclass(x, Validation):
            validation_cls[x.name] = x

    validations = []
    if 'validations' in list(config.keys()):
        for name, kwargs in config['validations'].items():
            assert name in list(validation_cls.keys())
            validations.append(
                    validation_cls[name](**kwargs),
                    )
    return validations

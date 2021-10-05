import yaml
import logging
import numpy as np

from molmod.units import angstrom

from openyaff.wrappers import YaffForceFieldWrapper, OpenMMForceFieldWrapper
from openyaff.utils import log_header, reduce_box_vectors


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
                'all': wrapper

            'openmm':
                ('all', CUDA): wrapper

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
                seed_kinds = ['all']
            for kind in seed_kinds:
                self._internal_validate(
                        configuration,
                        conversion,
                        platform,
                        kind,
                        )

    def log(self):
        """Logs information prior to running the validation"""
        log_header(
                self.name + ' validation',
                logger,
                )
        logger.info('')
        logger.info('')
        logger.info('validating the following OpenMM platforms:')
        for platform in self.platforms:
            logger.info('\t\t' + platform)

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
        config['kind'] = self.name

        final = {'validations': [config]}
        if path_config is not None:
            assert path_config.suffix == '.yml'
            if path_config.exists():
                # load contents and look for 'yaff' key, replace contents
                with open(path_config, 'r') as f:
                    loaded_config = yaml.load(f, Loader=yaml.FullLoader)
                if 'validations' in loaded_config.keys():
                    loaded_config['validations'].append(config)
                else:
                    loaded_config['validations'] = [config]
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
        """Annotates a .yml file with comments regarding the current system"""
        message = """ VALIDATION

    The validation generally consists of a series of individual validation
    experiments. Each experiment (e.g. a single point or stress validation)
    has its own keywords.

        - singlepoint:

            performs a series of single point calculations on randomly generated
            states, and compares forces and energies. Allowed keywords for this
            experiment are:
                nstates: the number of random states to generate
                box_ampl: the amplitude of perturbations in box vectors
                disp_ampl: the amplitude of perturbations in atomic coordinates

        - stress:

            performs a series of numerical stress calculations on randomly
            generated states. Alowed keywords for this experiment are:
                nstates: the number of random states to generate
                box_ampl: the amplitude of perturbations in box vectors
                disp_ampl: the amplitude of perturbations in atomic coordinates
                dh: change in box vectors used in the finite difference
                approximation of the virial stress"""
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


class RandomStateValidation(Validation):
    """Subclass in which validation is performed over randomly generated states

    """
    properties = [
            'platforms',
            'separate_parts',
            'nstates',
            'disp_ampl',
            'box_ampl',
            ]

    def __init__(self, nstates=10, disp_ampl=0.5, box_ampl=1.0, **kwargs):
        """Constructor

        Parameters
        ----------

        nstates : int
            number of states to evaluate

        disp_ampl : float [angstrom]
            displacement amplitude for each particle. For each state and for
            each particle within each state, a random (uniform) displacement
            is added to its current positions. Choosing the amplitude too large
            may lead to extremely unstable states and large (but irrelevant)
            numerical errors.

        box_ampl : float [angstrom]
            determines amplitude of uniform displacements in box vector
            components.

        """
        self.nstates = nstates
        self.disp_ampl = disp_ampl
        self.box_ampl = box_ampl
        super().__init__(**kwargs)


class SinglePointValidation(RandomStateValidation):
    """Validates energy and force calculations"""
    name = 'singlepoint'

    def _internal_validate(self, configuration, conversion, platform, kind):
        """Performs single point validations"""
        # perform conversion, initialize arrays and wrappers
        seed_yaff = configuration.create_seed(kind)
        seed_mm   = conversion.apply(configuration, seed_kind=kind)
        energy = np.zeros((4, self.nstates)) # stores energies and rel error
        forces = np.zeros((3, self.nstates, seed_yaff.system.natom, 3))
        wrapper_yaff = YaffForceFieldWrapper.from_seed(seed_yaff)
        wrapper_mm   = OpenMMForceFieldWrapper.from_seed(seed_mm, platform)

        # generate states
        states = []
        positions = seed_yaff.system.pos.copy() / angstrom
        if configuration.box is not None:
            rvecs = seed_yaff.system.cell._get_rvecs().copy() / angstrom
        for i in range(self.nstates):
            delta = 2 * self.disp_ampl * np.random.uniform(size=positions.shape)
            state = (positions + delta,)
            if configuration.box is not None:
                delta = 2 * self.box_ampl * np.random.uniform(size=rvecs.shape)
                delta[0, 1] = 0
                delta[0, 2] = 0
                delta[1, 2] = 0
                drvecs = rvecs + delta
                reduce_box_vectors(drvecs) # possibly no longer reduced
                state += (drvecs,)
            states.append(state)
        logger.info('')
        logger.info('\t\tPLATFORM: {} \t\t INTERACTION: {}'.format(platform, kind))
        logger.info('-' * 91)
        prefixes = configuration.get_prefixes(kind)
        if len(prefixes) > 0: # ignore empty parts
            nspaces = 10
            last = 5
            header = '     YAFF [kJ/mol]'
            header += nspaces * ' '
            header += '  OpenMM [kJ/mol]'
            header += nspaces * ' '
            header += '   delta [kJ/mol]'
            header += last * ' '
            header += 'relative error'
            logger.info(header)

            for i, state in enumerate(states):
                energy[0, i], forces[0, i] = wrapper_yaff.evaluate(
                        *state,
                        do_forces=True,
                        )
                energy[1, i], forces[1, i] = wrapper_mm.evaluate(
                        *state,
                        do_forces=True,
                        )
                energy[2, i] = energy[1, i] - energy[0, i]
                energy[3, i] = np.abs(energy[2, i]) / np.abs(energy[0, i])
                line = ' {:17.4f}'.format(energy[0, i])
                line += nspaces * ' '
                line += '{:17.4f}'.format(energy[1, i])
                line += nspaces * ' '
                line += '{:17.4f}'.format(energy[2, i])
                line += last * ' '
                line += '{:14.4e}'.format(energy[3, i])
                logger.info(line)

            df = np.abs(forces[1, :] - forces[0, :])
            error = np.linalg.norm(df, axis=2)
            norm = np.mean(np.linalg.norm(forces[0, :], axis=2))
            logger.info('')
            nspaces = 4
            line = '\tFORCES RELATIVE ERROR: \t'
            line += 'mean={:.1e}'.format(np.mean(error) / norm)
            line += nspaces * ' '
            line += 'median={:.1e}'.format(np.median(error) / norm)
            line += nspaces * ' '
            line += 'min={:.1e}'.format(np.min(error) / norm)
            line += nspaces * ' '
            line += 'max={:.1e}'.format(np.max(error) / norm)
            logger.info(line)
            logger.info('-' * 91)
            logger.info('')
            logger.info('')
        else:
            logger.info('\tno {} interactions present'.format(kind))

    def log(self):
        Validation.log(self)
        logger.info('based on single point calculations over {} '
                'random states:'.format(self.nstates))
        logger.info('\t\tparticle displacement amplitude: {} angstrom'.format(
                self.disp_ampl))
        logger.info('\t\tbox vector displacement amplitude: {} angstrom'.format(
                self.box_ampl))
        logger.info('')


class StressValidation(RandomStateValidation):
    """Validates the numerical calculation of the virial stress"""
    name = 'numerical stress'
    properties = [
            'platforms',
            'separate_parts',
            'nstates',
            'disp_ampl',
            'box_ampl',
            'dh',
            ]

    def __init__(self, dh=1e-6, nstates=1, **kwargs):
        self.dh = dh
        super().__init__(**kwargs)
        self.nstates = nstates # override default value of parent

    def _internal_validate(self, configuration, conversion, platform, kind):
        """Calculates the numerical stress over a series of states"""
        assert configuration.box is not None, ('cannot compute numerical stress'
                ' for nonperiodic systems')

        # perform conversion, initialize arrays and wrappers
        seed_yaff = configuration.create_seed(kind)
        seed_mm   = conversion.apply(configuration, seed_kind=kind)
        stress = np.zeros((3, 6, self.nstates)) # stores energies and rel error
        wrapper_yaff = YaffForceFieldWrapper.from_seed(seed_yaff)
        wrapper_mm   = OpenMMForceFieldWrapper.from_seed(seed_mm, platform)

        # generate states
        states = []
        positions = seed_yaff.system.pos.copy() / angstrom
        rvecs = seed_yaff.system.cell._get_rvecs().copy() / angstrom
        for i in range(self.nstates):
            delta = 2 * self.disp_ampl * np.random.uniform(size=positions.shape)
            state = (positions + delta,)
            delta = 2 * self.box_ampl * np.random.uniform(size=rvecs.shape)
            delta[0, 1] = 0
            delta[0, 2] = 0
            delta[1, 2] = 0
            drvecs = rvecs + delta
            reduce_box_vectors(drvecs) # possibly no longer reduced
            state += (drvecs,)
            states.append(state)
        logger.info('')
        logger.info('')
        logger.info('\t\tPLATFORM: {} \t\t INTERACTION: {}'.format(platform, kind))
        logger.info('-' * 90)
        prefixes = configuration.get_prefixes(kind)
        if len(prefixes) > 0: # ignore empty parts
            nspaces = 4
            header = ' ' * (9) + 'YAFF [kJ/angstrom**3]'
            header += nspaces * ' '
            header += '  OpenMM [kJ/angstrom**3]'
            header += nspaces * ' '
            header += '   delta [kJ/angstrom**3]'
            logger.info(header)
            for i, state in enumerate(states):
                stress_yaff = wrapper_yaff.compute_stress(
                        *state,
                        dh=self.dh,
                        use_symmetric=True,
                        )
                stress_mm = wrapper_mm.compute_stress(
                        *state,
                        dh=self.dh,
                        use_symmetric=True,
                        )
                # symmetrize and print six components
                stress_yaff = (stress_yaff + stress_yaff.T) / 2
                stress_mm = (stress_mm + stress_mm.T) / 2
                nspaces = 9
                start = 3
                value_yaff = np.trace(stress_yaff) / 3
                value_mm   = np.trace(stress_mm) / 3
                line = 'PRESSURE:' + start * ' '
                line += '{:17.4f}'.format(value_yaff)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_mm)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_yaff - value_mm)
                logger.info(line)

                value_yaff = stress_yaff[0, 0] / 3
                value_mm   = stress_mm[0, 0] / 3
                line = 'sigma_xx:' + start * ' '
                line += '{:17.4f}'.format(value_yaff)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_mm)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_yaff - value_mm)
                logger.info(line)

                value_yaff = stress_yaff[1, 1] / 3
                value_mm   = stress_mm[1, 1] / 3
                line = 'sigma_yy:' + start * ' '
                line += '{:17.4f}'.format(value_yaff)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_mm)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_yaff - value_mm)
                logger.info(line)

                value_yaff = stress_yaff[2, 2] / 3
                value_mm   = stress_mm[2, 2] / 3
                line = 'sigma_zz:' + start * ' '
                line += '{:17.4f}'.format(value_yaff)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_mm)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_yaff - value_mm)
                logger.info(line)

                value_yaff = stress_yaff[1, 2] / 3
                value_mm   = stress_mm[1, 2] / 3
                line = 'sigma_yz:' + start * ' '
                line += '{:17.4f}'.format(value_yaff)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_mm)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_yaff - value_mm)
                logger.info(line)

                value_yaff = stress_yaff[0, 2] / 3
                value_mm   = stress_mm[0, 2] / 3
                line = 'sigma_xz:' + start * ' '
                line += '{:17.4f}'.format(value_yaff)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_mm)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_yaff - value_mm)
                logger.info(line)

                value_yaff = stress_yaff[0, 1] / 3
                value_mm   = stress_mm[0, 1] / 3
                line = 'sigma_xy:' + start * ' '
                line += '{:17.4f}'.format(value_yaff)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_mm)
                line += nspaces * ' '
                line += '{:17.4f}'.format(value_yaff - value_mm)
                logger.info(line)
                logger.info('')


        else:
            logger.info('\tno {} interactions present'.format(kind))


def load_validations(path_yml):
    with open(path_yml, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    validation_cls = {}
    for x in list(globals().values()):
        if isinstance(x, type) and issubclass(x, Validation):
            validation_cls[x.name] = x

    validations = []
    if 'validations' in list(config.keys()):
        assert isinstance(config['validations'], list)
        for kwargs in config['validations']:
            assert 'kind' in kwargs.keys()
            name = kwargs.pop('kind')
            validations.append(
                    validation_cls[name](**kwargs),
                    )
    return validations

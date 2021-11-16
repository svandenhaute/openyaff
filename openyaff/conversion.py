import yaml
import numpy as np
import logging
import xml.etree.ElementTree as ET
import networkx as nx
import simtk.openmm as mm
import simtk.openmm.app
import simtk.unit as unit
import molmod
import tempfile

from openyaff.utils import create_openmm_system, get_scale_index
from openyaff.generator import COVALENT_PREFIXES, DISPERSION_PREFIXES, \
        ELECTROSTATIC_PREFIXES, apply_generators_to_system, \
        apply_generators_to_xml
from openyaff.seeds import OpenMMSeed


logger = logging.getLogger(__name__)


class Conversion:
    """Base class for conversion objects"""
    kind = None
    properties = [
            'pme_error_thres',
            ]

    def __init__(self, pme_error_thres=1e-5):
        self.pme_error_thres = pme_error_thres

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
        #_all = COVALENT_PREFIXES + DISPERSION_PREFIXES + ELECTROSTATIC_PREFIXES
        #for key, _ in configuration.parameters.sections.items():
        #    assert key in _all, ('I do not have a generator '
        #            'for key {}'.format(key))
        nonbonded_prefixes = configuration.get_prefixes('nonbonded')
        for key, _ in configuration.parameters.sections.items():
            if key in nonbonded_prefixes: # can contain scalings
                definition = _['SCALE']
                # line == [line_no, line_content]
                for line in definition.lines:
                    # scaling is last part of string
                    scale = float(line[1].split(' ')[-1])
                    assert (scale == 0.0) or (scale == 1.0)

    def determine_exclusion_policy(self, configuration):
        """Determines the implementation of the nonbonded exclusions

        Some OpenMM platforms (CUDA and CPU) force all nonbonded interactions
        to use the same exclusions. However, force fields for e.g. nanoporous
        materials often employ different exclusion policies for the dispersion
        and electrostatic interactions (dispersion often 0 0 1, electrostatic
        usually 1 1 1). If this is the case, then it is necessary to:

            (i) create a CustomNonbondedForce with exclusions as given in the
            parameter file
            (ii) create NonbondedForce for electrostatics, with exclusions as
            given in the created CustomNonbondedForce object (which will
            generally differ from the ones given in the parameter file).
            (iii) add CustomBondForce objects for each wrong exclusion
            in NonbondedForce as compensation.

        While this approach will work on all platforms (i.e. also the Reference
        platform) it is generally not the best choice in terms of efficiency.

        Parameters
        ----------

        configuration : Configuration
            configuration for which to determine the exclusion implementation

        Returns
        -------

        exclusion_policy : str
            regular policy or manual

        dispersion_scale_index : int
            this int is passed onto the various generators and will determine
            the built-in exclusions that will be used. Possible differences
            in the electrostatic scalings are then implemented manually using
            bonded compensation forces.

        """
        dispersion    = configuration.get_prefixes('dispersion')
        electrostatic = configuration.get_prefixes('electrostatic')

        # evaluate exclusion policy based on the scalings in the
        # dispersion prefixes, and whether they differ from the electrostatics
        # the scaling index for both dispersion and electrostatics is returned
        exclusion_policy = 'regular'
        dispersion_scale_index = None
        if len(dispersion) == 0: # no dispersion present, use regular policy
            pass
        elif len(dispersion) >= 1: # check for difference with electrostatic
            ref = None # verify all dispersion interactions have the same exclusions
            for prefix in dispersion:
                index_ = get_scale_index(
                        configuration.parameters.sections[prefix]['SCALE'],
                        )
                if dispersion_scale_index is not None:
                    assert (dispersion_scale_index == index_)
                else:
                    dispersion_scale_index = index_
            # iterate over present electrostatics and check if exclusions differ
            for prefix in electrostatic:
                scale_index = get_scale_index(
                        configuration.parameters.sections[prefix]['SCALE'],
                        )
                if dispersion_scale_index != scale_index:
                    exclusion_policy = 'manual'
        return exclusion_policy, dispersion_scale_index

    @property
    def pme_error_thres(self):
        """Returns the error threshold for the PME calculation"""
        return self._pme_error_thres

    @pme_error_thres.setter
    def pme_error_thres(self, value):
        self._pme_error_thres = value


class ExplicitConversion(Conversion):
    """Defines the explicit conversion procedure from YAFF to OpenMM

    In this procedure, each YAFF generator is extended with function calls
    to an OpenMM system object such that the force field generation occurs
    synchronously between OpenMM and YAFF. This is the safest option, but the
    output .xml file may become excessively large for systems with many atoms,
    as each interaction is encoded manually in the file.

    """
    kind = 'explicit'

    def apply(self, configuration, seed_kind='all'):
        """Converts a yaff configuration into an OpenMM seed

        Begins with a call to check_compatibility, and an assertion error is
        raised if the configuration is not compatible.
        The system object in the yaff seed is used to create a corresponding
        openmm system object, after which apply_generators_to_system is called.

        Parameters
        ----------

        configuration : openyaff.Configuration
            configuration for which to check compatibility

        platform : str
            OpenMM platform for which to perform the conversion because
            some conversion options (such as the exclusion policy) are platform
            dependent.

        seed_kind : str
            specifies the kind of seed to be converted

        """
        # raise AssertionError if not compatible
        self.check_compatibility(configuration)
        policy, dispersion_scale_index = self.determine_exclusion_policy(configuration)
        logger.debug('exclusion policy: ' + policy)
        logger.debug('dispersion scale index: {}'.format(dispersion_scale_index))
        yaff_seed = configuration.create_seed(kind=seed_kind)
        logger.debug('creating OpenMM System object')
        system_mm = create_openmm_system(yaff_seed.system)
        dummy = mm.HarmonicBondForce()
        periodic = configuration.box is not None
        dummy.setUsesPeriodicBoundaryConditions(periodic)
        system_mm.addForce(dummy) # add empty periodic force
        kwargs = {}

        # if system is periodic and contains electrostatis; compute PME params
        if (configuration.ewald_alphascale is not None and
                seed_kind in ['all', 'electrostatic', 'nonbonded']):
            alpha = configuration.ewald_alphascale
            delta = np.exp(-(alpha) ** 2) / 2
            if delta > self.pme_error_thres:
                kwargs['delta'] = delta
            else:
                kwargs['delta'] = self.pme_error_thres
        # add exclusion policy to kwargs
        kwargs['exclusion_policy'] = policy
        kwargs['dispersion_scale_index'] = dispersion_scale_index

        apply_generators_to_system(yaff_seed, system_mm, **kwargs)
        openmm_seed = OpenMMSeed(system_mm)
        return openmm_seed


class ImplicitConversion(Conversion):
    """Converts a YAFF System and Parameters object into an OpenMM ForceField

    This conversion is faster and therefore ideally suited for large systems.
    Its functionality is somewhat restricted as compared to the
    ExplicitConversion class.

    """
    kind = 'implicit'

    def check_compatibility(self, configuration):
        """Checks whether all parameter sections are supported

        The implicit conversion scheme is less flexible in its support for
        unconventional interactions: exotic cross terms or dispersion
        interactions.

        """
        scalings = {}
        nonbonded_prefixes = configuration.get_prefixes('nonbonded')
        if len(nonbonded_prefixes) > 0:
            for key in nonbonded_prefixes:
                scalings[key] = {}
                i = 1
                for line in configuration.parameters.sections[key]['SCALE']:
                    scale = float(line[1].split(' ')[-1])
                    scalings[key][i] = scale
                    i += 1

            # if nonbonded_prefixes are (LJ and FIXQ):
            #   allow noninteger scaling of 1-4 interaction
            #   require that all other scalings are 0
            # else:
            #   require all scalings to be exactly the same
            #   only a scale of 0 or 1 is allowed
            if tuple(sorted(nonbonded_prefixes)) == ('FIXQ', 'LJ'):
                assert scalings['FIXQ'][1] == 0.0
                assert scalings['FIXQ'][2] == 0.0
                assert scalings['FIXQ'][3] <= 1.0
                assert scalings['LJ'][1] == 0.0
                assert scalings['LJ'][2] == 0.0
                assert scalings['LJ'][3] <= 1.0
            else:
                for i, key in enumerate(nonbonded_prefixes):
                    if i == 0: # check whether scalings are 0 or 1
                        for j in [1, 2, 3]:
                            assert (scalings[key][j] == 0.0) or (scalings[key][j] == 1.0)
                    else: # check whether scaling is equal to i == 0 scalin
                        assert scalings[key][1] == scalings[nonbonded_prefixes[0]][1]
                        assert scalings[key][2] == scalings[nonbonded_prefixes[0]][2]
                        assert scalings[key][3] == scalings[nonbonded_prefixes[0]][3]
        return scalings

    def apply(self, configuration, seed_kind='all'):
        """Generates force field XML tree"""
        scalings = self.check_compatibility(configuration)

        # initialize XML object
        forcefield = ET.Element('ForceField')
        sys = configuration.system
        sys.set_standard_masses()

        # add atom types
        atom_types_xml = ET.Element('AtomTypes')
        for i in range(len(sys.ffatypes)):
            index = np.nonzero(sys.ffatype_ids == i)[0][0]
            atom_type = sys.ffatypes[i]
            mass      = sys.masses[index] / molmod.units.amu
            element   = molmod.periodic.periodic[sys.numbers[index]].symbol
            attrib = {
                    'name': atom_type,
                    'class': atom_type,
                    'element': element,
                    'mass': str(mass), # xml expects str, not float
                    }
            atom_types_xml.append(ET.Element('Type', attrib=attrib))
        forcefield.append(atom_types_xml)


        # add template definitions
        residues_xml = ET.Element('Residues')
        for i, template in enumerate(configuration.templates):
            residue_xml = ET.Element('Residue', {'name': 'T' + str(i)})
            types = nx.get_node_attributes(template, 'atom_type')

            # generate atom names within template
            count = np.ones(118, dtype=np.int32)
            index_to_name = {}
            for node in template.nodes():
                if node < configuration.ext_node_count:
                    number = sys.numbers[node]
                    e = mm.app.Element.getByAtomicNumber(number)
                    atom_name = e.symbol + str(count[number])
                    index_to_name[node] = atom_name
                    count[number] += 1

            for node in template.nodes():
                if node < configuration.ext_node_count:
                    atom_name = index_to_name[node]
                    atom_type = types[node]
                    e = ET.Element('Atom', {'name': atom_name, 'type': atom_type})
                    residue_xml.append(e)
            for j, (atom0, atom1) in enumerate(template.edges()):
                is_external0 = (atom0 >= configuration.ext_node_count)
                is_external1 = (atom1 >= configuration.ext_node_count)
                if (not is_external0) and (not is_external1):
                    # this is a real bond
                    attrib = {
                            'atomName1': index_to_name[atom0],
                            'atomName2': index_to_name[atom1],
                            }
                    e = ET.Element('Bond', attrib=attrib)
                    residue_xml.append(e)
                elif (is_external0 != is_external1):
                    # this is an external bond
                    if is_external0:
                        atom = atom1
                    else:
                        atom = atom0
                    attrib = {
                            'atomName': index_to_name[atom],
                            }
                    e = ET.Element('ExternalBond', attrib=attrib)
                    residue_xml.append(e)
                else:
                    raise ValueError('Encountered a bond between two external'
                            ' nodes: {} and {}'.format(atom0, atom1))

            residues_xml.append(residue_xml)
        forcefield.append(residues_xml)

        # get kwargs for apply_generators_to_xml from configuration
        if len(configuration.get_prefixes('nonbonded')) > 0:
            if configuration.box is not None:
                nonbondedMethod = mm.app.PME
                nonbondedCutoff = configuration.rcut
                if configuration.switch_width is not None:
                    useSwitchingFunction = True
                    switchingDistance = configuration.rcut - configuration.switch_width
                else:
                    useSwitchingFunction = False
                    switchingDistance = 0.0
                useLongRangeCorrection = configuration.tailcorrections
            else:
                nonbondedMethod = mm.app.NoCutoff
                nonbondedCutoff = 0.0
                switchingDistance  = 0.0
                useSwitchingFunction = 0
                useLongRangeCorrection = 0
        else: # set dummy values
            nonbondedMethod = mm.app.NoCutoff
            nonbondedCutoff = 0.0
            useSwitchingFunction = False
            switchingDistance = 0.0
            useLongRangeCorrection=False

        # generate yaff_seed and apply generators to the XML object
        yaff_seed = configuration.create_seed(kind=seed_kind) # unnecessary?
        forces = apply_generators_to_xml(
                yaff_seed,
                forcefield,
                nonbondedMethod=nonbondedMethod,
                nonbondedCutoff=nonbondedCutoff,
                useSwitchingFunction=useSwitchingFunction,
                switchingDistance=switchingDistance,
                useLongRangeCorrection=useLongRangeCorrection,
                scalings=scalings,
                )
        for force in forces:
            forcefield.append(force)
        forcefield = ET.ElementTree(element=forcefield) # convert to tree
        ET.indent(forcefield)
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as tf:
            forcefield.write(tf, encoding='unicode')
        tf.close()
        #pars = ET.tostring(forcefield.getroot(), encoding='unicode')
        #print(pars)
        ff = mm.app.ForceField(tf.name)

        # create OpenMM system object
        topology, _ = configuration.create_topology()
        system = ff.createSystem(
                topology,
                nonbondedMethod=nonbondedMethod,
                nonbondedCutoff=nonbondedCutoff / 10, # random value
                ignoreExternalBonds=False,
                removeCMMotion=False,
                switchDistance=switchingDistance / 10,
                )
        if configuration.box is not None:
            # dirty fixes!
            dummy = mm.HarmonicBondForce()
            dummy.setUsesPeriodicBoundaryConditions(True)
            system.addForce(dummy) # add empty force to define periodicity
            for force in system.getForces():
                try:
                    force.setUsesPeriodicBoundaryConditions(True)
                    logger.critical('force {} set to use PBCs'.format(force))
                except:
                    continue
        return OpenMMSeed(system, forcefield)


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

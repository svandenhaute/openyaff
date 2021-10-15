import logging
import tempfile
import yaff
import molmod
import yaml
import numpy as np
import networkx as nx
from datetime import datetime

import simtk.openmm as mm
import simtk.openmm.app
import simtk.unit as unit

from openyaff.utils import determine_rcut, transform_lower_triangular, \
        compute_lengths_angles, is_lower_triangular, is_reduced, \
        reduce_box_vectors, log_header, wrap_coordinates, \
        find_smallest_supercell
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
            'interaction_radius',
            'rcut',
            'supercell',
            'switch_width',
            'tailcorrections',
            'ewald_alphascale',
            'ewald_gcutscale',
            ]

    def __init__(self, system, pars, topology=None):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object for which to generate a base configuration

        pars : str
            string containing the contents of a YAFF force field parameters file

        topology : openmm Topology or None
            OpenMM topology which defines the residues present in the system

        """
        if (system.cell is not None) and (system.cell._get_nvec() == 3):
            self.box = system.cell._get_rvecs() / molmod.units.angstrom
        else:
            self.box = None # nonperiodic system does not have box vectors
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

        # match the residue templates to atoms in the system. Templates are
        # determined based on an existing Topology object or are otherwise
        # inferred based on the system connectivity.
        self.topology = topology
        if topology is not None: # make sure system and topology are consistent
            assert system.natom == len(list(topology.atoms()))
            for i, atom in zip(range(system.natom), topology.atoms()):
                assert system.numbers[i] == atom.element._atomic_number
            assert system.bonds.shape[0] == len(list(topology.bonds()))
            bonds_tuples = [tuple(sorted(list(bond))) for bond in system.bonds]
            for bond in topology.bonds():
                indices_top = tuple(sorted((bond[0].index, bond[1].index)))
                bonds_tuples.remove(indices_top)
            assert len(bonds_tuples) == 0
        self.templates, self.residues = self.define_templates()

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
        if self.box is not None:
            supercell = self.determine_supercell()
            system = self.system.supercell(*supercell)
            # apply reduction
            rvecs = system.cell._get_rvecs().copy()
            transform_lower_triangular(system.pos, rvecs, reorder=True)
            reduce_box_vectors(rvecs)
            wrap_coordinates(system.pos, rvecs)
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

    def create_topology(self):
        """Creates the topology for the current configuration"""
        topology = mm.app.Topology()
        chain = topology.addChain()

        natoms = self.system.natom
        count = 0
        if self.box is not None:
            supercell = self.determine_supercell()

            # compute box vectors and allocate positions array
            box = np.array(supercell)[:, np.newaxis] * self.box
            positions = np.zeros((np.prod(supercell) * natoms, 3))

            # construct map of atom indices to residues
            atom_index_mapping = {}
            for template, residues in self.residues.items():
                for i, residue in enumerate(residues):
                    for j, atom in enumerate(residue):
                        atom_index_mapping[atom] = (template, i, j)
            included_atoms = list(atom_index_mapping.keys())
            assert tuple(sorted(included_atoms)) == tuple(range(natoms))

            def name_residue(image, template, residue_index):
                """Defines the name of a specific residue"""
                return 'CELL' + str(image) + '_T' + str(template) + '_R' + str(i)

            atoms_list = [] # necessary for adding bonds to topology
            for image, index in enumerate(np.ndindex(tuple(supercell))):

                # initialize residues and track them in a dict
                current_residues = {}
                for template, residues in self.residues.items():
                    for i, residue in enumerate(residues):
                        name = name_residue(image, template, i)
                        residue = topology.addResidue(
                                name=name,
                                chain=chain,
                                id=name,
                                )
                        current_residues[(template, i)] = residue

                # add atoms to corresponding residue in topology (in their
                # original order)
                for j in range(natoms):
                    key = atom_index_mapping[j]
                    template = key[0]
                    residue_index = key[1]
                    atom_index = key[2]
                    e = mm.app.Element.getByAtomicNumber(self.system.numbers[j])
                    atom_name = 'giggle'
                    atom = topology.addAtom(
                            name=atom_name,
                            element=e,
                            residue=current_residues[(template, residue_index)]
                            )
                    atoms_list.append(atom)

                # generate positions for this image
                image_pos = self.system.pos / molmod.units.angstrom
                translate = np.dot(np.array(index), self.box).reshape(1, 3)
                image_pos += translate
                start = (image) * natoms
                stop  = (image + 1) * natoms
                positions[start : stop, :] = image_pos.copy()

            # apply cell reduction and wrap coordinates (similar to create_seed)
            transform_lower_triangular(positions, box, reorder=True)
            reduce_box_vectors(box)
            wrap_coordinates(positions, box)
            topology.setPeriodicBoxVectors(box * unit.angstrom)

            # add bonds from supercell system object
            system = self.system.supercell(*supercell)
            for bond in system.bonds:
                topology.addBond(
                        atoms_list[bond[0]],
                        atoms_list[bond[1]],
                        )

        else: # similar workflow, but without the supercell generation
            positions = self.system.pos / molmod.units.angstrom

            # construct map of atom indices to residues
            atom_index_mapping = {}
            for template, residues in self.residues.items():
                for i, residue in enumerate(residues):
                    for j, atom in enumerate(residue):
                        atom_index_mapping[atom] = (template, i, j)
            included_atoms = list(atom_index_mapping.keys())
            assert tuple(sorted(included_atoms)) == tuple(range(natoms))

            def name_residue(template, residue_index):
                """Defines the name of a specific residue"""
                return 'T' + str(template) + '_R' + str(i)

            atoms_list = [] # necessary for adding bonds to topology

            # initialize residues and track them in a dict
            current_residues = {}
            for template, residues in self.residues.items():
                for i, residue in enumerate(residues):
                    name = name_residue(template, i)
                    residue = topology.addResidue(
                            name=name,
                            chain=chain,
                            id=name,
                            )
                    current_residues[(template, i)] = residue

            # add atoms to corresponding residue in topology (in their
            # original order)
            for j in range(natoms):
                key = atom_index_mapping[j]
                template = key[0]
                residue_index = key[1]
                atom_index = key[2]
                e = mm.app.Element.getByAtomicNumber(self.system.numbers[j])
                atom_name = 'giggle'
                atom = topology.addAtom(
                        name=atom_name,
                        element=e,
                        residue=current_residues[(template, residue_index)]
                        )
                atoms_list.append(atom)

            # add bonds from system object
            for bond in self.system.bonds:
                topology.addBond(
                        atoms_list[bond[0]],
                        atoms_list[bond[1]],
                        )
        return topology, positions

    def define_templates(self):
        """Defines residue templates for this system based on its topology

        If an initial partitioning of the system into residues is available
        (as a topology object in self.topology), then
        this function first matches the given residues to the system and
        verifies that each atom is assigned to a residue. If no initial residue
        partitioning is given, then a suitable choice of residues is inferred
        based on the connectivity and periodicity of the system.
        Once residues are defined, they are used to construct templates based
        on the atom types as defined in the yaff System object.

        """
        graph = nx.Graph() # graph of entire system
        for i in range(self.system.natom):
            graph.add_node(i, kind=self.system.numbers[i])
        if self.system.bonds is None:
            self.system.bonds = np.zeros((0, 2))
        for bond in self.system.bonds:
            graph.add_edge(bond[0], bond[1])

        if self.topology is None:
            # infer residues from bond connectivity and periodicity
            # for framework materials (whose 'molecular' graph is periodic),
            # it is necessary to remove the bonds from system which cross the
            # unit cell boundaries.
            #
            # 1. (if periodic) wrap positions into current box
            # 2. separate system into connected graphs
            # for each connected graph:
            # 3. (if periodic) identify and remove cross-unit-cell bonds
            # 4. create residue based on remaining bonds

            if self.box is not None:
                positions = self.system.pos / molmod.units.angstrom
                wrap_coordinates(positions, self.box, rectangular=False)

            components = [graph.subgraph(c).copy() for c in nx.algorithms.connected_components(graph)]
            logger.debug('found {} connected components'.format(len(components)))
            thres = 4.0 # maximum distance of covalent bond 
            if self.box is not None:
                for i in range(len(components)):
                    # avoid edge case where no bonds/edges are present in the
                    # component
                    if len(list(components[i].edges())) > 0:
                        pass
                    else:
                        continue
                    indices = np.array(components[i].edges())
                    lengths = np.linalg.norm(
                            positions[indices[:, 1], :] - positions[indices[:, 0], :],
                            axis=1,
                            ) # find external bonds using nonperiodic distances
                    external = lengths > thres
                    # copy component, remove edges, and check connectivity
                    tmp = components[i].copy()
                    for j in range(len(external)):
                        if external[j]: # remove this edge
                            tmp.remove_edge(indices[j, 0], indices[j, 1])
                    if nx.is_connected(tmp):
                        # if a component remains connected after removing
                        # 'external' edges, it should be modified.
                        components[i] = tmp
                    else:
                        # if a component becomes disconnected after removing
                        # 'external' edges, it does not require modifications
                        pass
            else:
                # no need to do anything for nonperiodic system
                pass
        else:
            # construct components from residues defined in topology
            components = []
            for i, residue in enumerate(self.topology.residues()):
                index_set = set()
                for atom in residue.atoms():
                    index_set.add(atom.index)
                components.append(graph.subgraph(index_set))

        # fill residues dict
        residues = {} # maps template index to list of residues
        i = 0
        while i < len(components):
            residues[i] = [tuple(components[i].nodes())]
            j = i + 1
            while j < len(components): # loop over remaining components
                match = nx.is_isomorphic(
                        components[i],
                        components[j],
                        node_match=lambda x, y: x['kind'] == y['kind'],
                        )
                if match:
                    residues[i].append(tuple(components[j].nodes()))
                    components.pop(j)
                else:
                    j += 1
            i += 1

        # only unique components are retained; these are the templates
        templates = components
        logger.debug('found {} templates:'.format(len(templates)))
        for index, residue_list in residues.items():
            logger.debug('\t{} residues from template {}'.format(
                len(residue_list),
                index,
                ))

        # add force field atom types to templates
        types = [self.system.ffatypes[i] for i in self.system.ffatype_ids]
        types_dict = {i: {'atom_type': types[i]} for i in range(self.system.natom)}
        for template in templates:
            nx.set_node_attributes(template, types_dict)
        return templates, residues

    def determine_supercell(self):
        """Determines supercell of the system"""
        assert self.box is not None
        if self.supercell == 'auto':
            return find_smallest_supercell(
                    self.box,
                    self.interaction_radius,
                    )
        else: # supercell is hardcoded
            assert isinstance(self.supercell, list)
            assert len(self.supercell) == 3
            return list(self.supercell)

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
        if self.box is not None:
            supercell = self.determine_supercell()
            natom = np.prod(np.array(supercell)) * self.system.natom
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
        if self.box is not None:
            system_type = 'periodic'
        else:
            system_type = 'non-periodic'
        logger.info('number of atoms:     {}'.format(natom))
        logger.info('system type:         ' + system_type)
        logger.info('')
        if self.box is not None:
            lengths, angles = compute_lengths_angles(self.box, degree=True)
            logger.info('initial box vectors (in angstrom):')
            logger.info('\ta: {}'.format(self.box[0, :]))
            logger.info('\tb: {}'.format(self.box[1, :]))
            logger.info('\tc: {}'.format(self.box[2, :]))
            logger.info('')
            logger.info('initial box lengths (in angstrom):')
            logger.info('\ta: {:.4f}'.format(lengths[0]))
            logger.info('\tb: {:.4f}'.format(lengths[1]))
            logger.info('\tc: {:.4f}'.format(lengths[2]))
            logger.info('')
            logger.info('initial box angles (in degrees):')
            logger.info('\talpha: {:.4f}'.format(angles[0]))
            logger.info('\tbeta : {:.4f}'.format(angles[1]))
            logger.info('\tgamma: {:.4f}'.format(angles[2]))
            logger.info('')
            logger.info('')
            rvecs = np.array(self.box) # create copy
            transform_lower_triangular(np.zeros((1, 3)), rvecs, reorder=True)
            reduce_box_vectors(rvecs)
            lengths, angles = compute_lengths_angles(rvecs, degree=True)
            logger.info('REDUCED box vectors (in angstrom):')
            logger.info('\ta: {}'.format(rvecs[0, :]))
            logger.info('\tb: {}'.format(rvecs[1, :]))
            logger.info('\tc: {}'.format(rvecs[2, :]))
            logger.info('')
            logger.info('REDUCED box lengths (in angstrom):')
            logger.info('\ta: {:.4f}'.format(lengths[0]))
            logger.info('\tb: {:.4f}'.format(lengths[1]))
            logger.info('\tc: {:.4f}'.format(lengths[2]))
            logger.info('')
            logger.info('REDUCED box angles (in degrees):')
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
    def from_files(chk=None, txt=None, pdb=None, yml=None):
        """Constructs a Configuration based on system and parameter files

        If, optionally, a config .yml file is specified, then the settings in
        that file are verified and loaded into the newly created object

        Parameters
        ----------

        chk : pathlib.Path
            specifies the location of the YAFF .chk file

        txt : pathlib.Path
            specifies the location of the force field parameters .txt file

        pdb : pathlib.Path, optional
            specifies the location of an OpenMM Topology .pdb file used to
            define residues within the system.

        yml : pathlib.Path, optional
            specifies the location of an existing .yml configuration file. If
            not specified, the default settings for the current system are used.

        """
        # pdb and yml are optional
        assert chk is not None
        assert txt is not None

        # load yaff System object; load pars as string; load residues in pdb
        system = yaff.System.from_file(str(chk))
        with open(txt, 'r') as f:
            pars = f.read()
        if (pdb is not None) and pdb.exists():
            topology = mm.app.PDBFile(str(pdb)).getTopology()
        else:
            topology = None
        configuration = Configuration(system, pars, topology)

        if yml is not None: # loads existing config
            with open(yml, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            configuration.update_properties(config)
        return configuration

    def initialize_properties(self):
        """Initializes properties to YAFF defaults"""
        rcut_default = yaff.FFArgs().rcut / molmod.units.angstrom
        try:
            self.interaction_radius = rcut_default
        except ValueError:
            pass

        try:
            self.rcut = rcut_default
        except ValueError:
            pass

        try:
            self.supercell = 'auto' # default is smallest possible supercell
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
        # order of property setting matters due to dependencies!
        # first set the desired interaction radius
        # then set the cutoff, and require that it is smaller than the radius
        # then set the supercell; if a manual supercell is requested, verify
        # that it is large enough for the requested radius.
        for name in self.properties:
            if name in config['yaff'].keys():
                setattr(self, name, config['yaff'][name])

    @staticmethod
    def annotate(path_yml):
        """Annotates a .yml file with comments regarding the current system"""
        message = """ YAFF

    Below is a list of possible keywords for this section. Their applicability
    depends on the specific system (whether it is periodic, whether it contains
    charges, ...). Some keywords that appear in the list below may therefore
    not actually be valid for this specific configuration.

    interaction_radius:
        if the supercell keyword is set to 'auto', then the supercell is
        determined as the smallest possible supercell that can accomodate
        interactions of this size. This value should be slightly larger than the
        nonbonded cutoff to ensure that the periodic box remains sufficiently
        large for the employed cutoff throughout a constant pressure MD
        simulation (as in that case, the box volume will fluctuate).

    supercell:
        determines the supercell to use. OpenMM does not allow interactions
        to reach further than half the *shortest* cell vector in its reduced
        representation.
        For most nanoporous materials and cutoff ranges (10 - 15 angstrom),
        this implies that validation with OpenMM is almost always performed on
        a supercell of the original unit cell in the .chk input.
        If this is set to 'auto', then OpenYAFF will use the smallest possible
        supercell that is compatible with the chosen interaction radius.
        If this is a list of three integers, then these represent the number
        of cell replicas in the first, second and third box vector directions.
        Other supercells and their corresponding largest cutoff distance are
        logged when executing `openyaff configure`.
        (see http://docs.openmm.org/latest/userguide/theory.html#periodic-boundary-conditions
        for more information on the reduced cell representation in OpenMM).

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
        if self.box is not None:
            if value == 'auto':
                pass
            elif len(value) == 3: # requested cell needs to be large enough
                supercell = find_smallest_supercell(
                        self.box,
                        self.interaction_radius,
                        )
                box = np.array(value)[:, np.newaxis] * self.box
                allowed_rcut = determine_rcut(box)
                if allowed_rcut < self.interaction_radius:
                    raise ValueError('The requested supercell {} only supports'
                            ' interactions up to {} angstrom, while the '
                            'requested interaction radius is set to {} '
                            'angstrom. The smallest possible supercell for this'
                            ' system is {}.'.format(
                                value,
                                allowed_rcut,
                                self.interaction_radius,
                                supercell,
                                ))
            else:
                raise AssertionError
            self._supercell = value
        else: # property not applicable
            self._supercell = None
            raise ValueError('Cannot use supercell keyword because system is '
                    'nonperiodic')

    @property
    def interaction_radius(self):
        return self._cell_radius

    @interaction_radius.setter
    def interaction_radius(self, value):
        if self.box is not None:
            self._cell_radius = value
        else: # property not applicable
            self._cell_radius = None
            raise ValueError('Cannot use interaction_radius keyword '
                    'because system is nonperiodic')

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
        if (self.box is not None) and (len(self.get_prefixes('nonbonded')) > 0):
            assert type(value) == float
            self._rcut = value
            assert self.interaction_radius is not None
            if self.rcut > self.interaction_radius:
                logger.debug('increasing interaction_radius to rcut')
                self.interaction_radius = self.rcut
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
        if (self.box is not None) and (len(self.get_prefixes('dispersion')) > 0):
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
        if (self.box is not None) and (len(self.get_prefixes('dispersion')) > 0):
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
        if ('FIXQ' in self.prefixes) and (self.box is not None):
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
        if ('FIXQ' in self.prefixes) and (self.box is not None):
            assert type(value) == float
            self._ewald_gcutscale = value
        else:
            self._ewald_gcutscale = None
            raise ValueError('Cannot set ewald parameters for this system')

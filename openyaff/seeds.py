import logging
import simtk.openmm as mm
from lxml import etree


logger = logging.getLogger(__name__) # logging per module


class YaffSeed:
    """Simple datastructure to represent a seed for YAFF force fields

    Seeds contain all the data and parameters necessary to construct a force
    field unambiguously.

    """

    def __init__(self, system, parameters, ff_args):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            yaff system necessary to apply the generators

        parameters : yaff.Parameters
            contains actual force field parameters

        ff_args : yaff.FFArgs
            contains meta parameters such as the cutoff radius for nonbonded
            interactions

        """
        self.system = system
        self.parameters = parameters
        self.ff_args = ff_args


class OpenMMSeed:
    """Simple datastructure to represent a seed for OpenMM force fields

    Seeds contain all the data and parameters necessary to construct a force
    field unambiguously.
    Because the XmlSerializer does not work well with very large systems --
    one million atoms or more -- it is alternatively possible to construct an
    OpenMMSeed based on several different (smaller) OpenMM System instances
    which each describe a different part of the force field.

    """

    def __init__(self, system, parts=None):
        """Constructor

        Parameters
        ----------

        system : mm.System or None
            contains the particle properties and all forces present in the
            system.

        parts : list of mm.System or None
            contains different system instances, where each item contains
            different forces (all else being equal)

        """
        self.system = system
        self.parts = parts

    def serialize(self, path_xml=None):
        if self.system is not None: # conventional serialization
            xml = mm.XmlSerializer.serialize(self.system)
            assert isinstance(xml, str), ('xml is of type {} but should be str,'
                    ' try converting in ludicrous mode'.format(type(xml)))
        else: # generate xml for each part and merge
            assert self.parts is not None
            tree_list = []
            for part in self.parts:
                tmp = mm.XmlSerializer.serialize(part)
                assert isinstance(tmp, str), ('xml is of type {} but should '
                        'be str; serialization failed.'.format(type(tmp)))
                tree = etree.fromstring(tmp)
                tree_list.append(tree)
            # merge all forces into 
            #base = tree_list[0]
            #base_forces = None
            #    for child in base:
            #        if child.tag == 'Forces':
            #            base_forces = child
            forces = etree.Element('Forces')
            for tree in tree_list:
                for child in tree:
                    if child.tag == 'Forces':
                        for force in child: # iterate over all forces
                            forces.insert(0, force)
            base = tree_list[0]
            for i, child in enumerate(base):
                if child.tag == 'Forces':
                    base[i] = forces # replace with total forces
            xml_binary = etree.tostring(base)
            xml = xml_binary.decode('utf-8')

        if path_xml is not None: # write xml
            if path_xml.exists():
                path_xml.unlink() # remove file if it exists
            with open(path_xml, 'w') as f:
                f.write(xml)
        return xml

    def get_system(self):
        if self.system is not None:
            return self.system
        else:
            xml = self.serialize() # merges separate parts
            system = mm.XmlSerializer.deserialize(xml)
            assert isinstance(system, mm.System)
            return system

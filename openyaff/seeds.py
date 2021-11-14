import molmod
import tempfile
import logging
import simtk.openmm as mm
import simtk.openmm.app
import simtk.unit as unit
from lxml import etree
import xml.etree.ElementTree as ET

from openyaff.utils import create_openmm_topology


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
        # in some cases, the ffatypes attribute is a numpy array 
        system.ffatypes = list(system.ffatypes)
        self.system = system
        self.parameters = parameters
        self.ff_args = ff_args

    def save_topology(self, path_pdb=None):
        """Saves topology of YAFF system and its positions/box vectors"""
        raise NotImplementedError
        #topology = create_openmm_topology(self.system)
        #if self.system.cell.nvec != 0: # check box vectors are included
        #    assert topology.getPeriodicBoxVectors() is not None
        #if path_pdb is not None:
        #    if path_pdb.exists():
        #        path_pdb.unlink()
        #    u = molmod.units.angstrom / unit.angstrom
        #    mm.app.PDBFile.writeFile(
        #            topology,
        #            self.system.pos / u,
        #            open(path_pdb, 'w+'),
        #            keepIds=True,
        #            )
        #return topology


class OpenMMSeed:
    """Simple datastructure to represent a seed for OpenMM force fields

    Seeds contain all the data and parameters necessary to construct a force
    field unambiguously.
    Because the XmlSerializer does not work well with very large systems --
    one million atoms or more -- it is alternatively possible to construct an
    OpenMMSeed based on several different (smaller) OpenMM System instances
    which each describe a different part of the force field.

    """

    def __init__(self, system, forcefield_xml=None):
        """Constructor

        Parameters
        ----------

        system : mm.System
            contains the particle properties and all forces present in the
            system.

        forcefield_xml : ET.ElementTree or None
            XML representation of an OpenMM ForceField object

        """
        self.system = system
        self.forcefield_xml = forcefield_xml

    def serialize_system(self, path_xml=None):
        # save system
        xml = mm.XmlSerializer.serialize(self.system)
        assert isinstance(xml, str), ('xml is of type {} but should be str,'
                ' try converting in ludicrous mode'.format(type(xml)))
        with open(path_xml, 'w+') as f:
            f.write(xml)
        return xml

    def serialize_forcefield(self, path_xml=None):
        assert self.forcefield_xml is not None
        with open(path_xml, 'w+') as f:
            self.forcefield_xml.write(f, encoding='unicode')
        with open(path_xml, 'r') as f:
            xml = f.read()
        return xml

    def get_system(self):
        return self.system

    #@classmethod
    #def from_forcefield(cls, forcefield, configuration):
    #    """Constructs an OpenMM System object from a force field"""
    #    topology, _ = configuration.create_topology()

    #    # get kwargs for createSystem from configuration
    #    if configuration.box is not None:
    #        nonbondedMethod = mm.app.PME # cannot use integers
    #        nonbondedCutoff = configuration.rcut * unit.angstrom
    #        switchDistance  = configuration.switch_width * unit.angstrom
    #    else:
    #        nonbondedMethod = mm.app.NoCutoff
    #        nonbondedCutoff = None
    #        switchDistance  = None

    #    #ET.indent(forcefield)
    #    #pars = ET.tostring(forcefield.getroot(), encoding='unicode')
    #    #print(pars)

    #    # create temporary xml file and generate force field object
    #    with tempfile.NamedTemporaryFile(delete=False, mode='w') as tf:
    #        forcefield.write(tf, encoding='unicode')
    #    tf.close()

    #    ff = mm.app.ForceField(tf.name)
    #    ff.getMatchingTemplates(topology, ignoreExternalBonds=True)
    #    if configuration.box is None:
    #        dummy_cutoff = 1e6
    #    else:
    #        dummy_cutoff = configuration.rcut
    #    system = ff.createSystem(
    #            topology,
    #            #nonbondedMethod=nonbondedMethod,
    #            nonbondedCutoff=dummy_cutoff, # random value
    #            ignoreExternalBonds=True,
    #            removeCMMotion=False,
    #            #switchDistance=switchDistance,
    #            )

    #    #with open('generated_system.xml', 'w+') as f:
    #    #    f.write(mm.XmlSerializer.serialize(system))
    #    return cls(system)

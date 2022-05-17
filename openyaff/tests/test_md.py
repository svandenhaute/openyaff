import molmod
import simtk.unit as unit
import simtk.openmm.app
import simtk.openmm as mm

from openyaff import Configuration, ExplicitConversion
from openyaff.utils import create_openmm_topology

from systems import get_system


def test_short_simulation(tmp_path):
    system, pars = get_system('uio66')
    configuration = Configuration(system, pars)

    # conversion
    conversion = ExplicitConversion()
    openmm_seed = conversion.apply(configuration)
    system = openmm_seed.get_system() # necessary to create Simulation object
    topology, positions = configuration.create_topology()
    a, b, c = topology.getPeriodicBoxVectors()

    # instantiate simulation for each platform
    platforms = ['Reference', 'CPU', 'CUDA', 'OpenCL']
    for name in platforms:
        integrator = mm.LangevinMiddleIntegrator(
                300 * unit.kelvin, # temperature
                0.1 / unit.picosecond, # friction coefficient
                0.5 * unit.femtosecond, # step size
                )
        try:
            platform = mm.Platform.getPlatformByName(name)
        except mm.OpenMMException:
            continue
        simulation = mm.app.Simulation(
                topology,
                system,
                integrator,
                platform,
                )
        simulation.context.setPositions(positions)
        #simulation.context.setPeriodicBoxVectors(box[0], box[1], box[2])
        simulation.context.setPeriodicBoxVectors(a, b, c)
        simulation.step(20)

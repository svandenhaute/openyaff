from openyaff import Configuration

from systems import get_system


def test_initialize(tmp_path):
    system, pars = get_system('lennardjones')
    configuration = Configuration(system, pars)

    # write defaults
    config = configuration.write()
    print(config)

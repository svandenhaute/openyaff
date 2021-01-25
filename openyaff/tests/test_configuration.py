from openyaff import Configuration

from systems import get_system


def test_initialize_write(tmp_path):
    system, pars = get_system('lennardjones')
    configuration = Configuration(system, pars)

    # write defaults
    path_config = tmp_path / 'config.yml'
    config = configuration.write(path_config)
    with open(path_config, 'r') as f:
        content = f.read()
    assert content == """yaff:
  rcut: 10.0
  supercell: [1, 1, 1]
  switch_width: 4.0
  tailcorrections: false
""" # whitespace matters


def test_nonperiodic(tmp_path):
    system, pars = get_system('alanine')
    configuration = Configuration(system, pars)

    # write defaults
    path_config = tmp_path / 'config.yml'
    config = configuration.write(path_config)
    with open(path_config, 'r') as f:
        content = f.read()
    assert content == """yaff: {rcut: 10.0, switch_width: 4.0}\n"""


def test_update_properties(tmp_path):
    system, pars = get_system('alanine')
    configuration = Configuration(system, pars)

    config = configuration.write()
    config['yaff']['rcut'] = 15.0
    configuration.update_properties(config)
    assert configuration.rcut == 15.0


def test_supercell():
    system, pars = get_system('cobdp')
    configuration = Configuration(system, pars)
    supercell = configuration.determine_supercell(10)
    assert tuple(supercell) == (4, 1, 3)

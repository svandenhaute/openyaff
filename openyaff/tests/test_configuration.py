import numpy as np

from openyaff import Configuration
from openyaff.utils import yaff_generate

from systems import get_system, here


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
  supercell:
  - 1
  - 1
  - 1
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
    assert content == """yaff: {}\n"""


def test_update_properties(tmp_path):
    system, pars = get_system('cobdp')
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


def test_from_files():
    path_system = here / 'cobdp' / 'system.chk'
    path_pars   = here / 'cobdp' / 'pars.txt'
    path_config = here / 'cobdp' / 'config.yml'
    configuration = Configuration.from_files(
            path_system,
            path_pars,
            path_config,
            )


def test_create_seed_periodic():
    system, pars = get_system('cobdp')
    configuration = Configuration(system, pars)
    # change parameters randomly
    configuration.supercell = [3, 1, 1]
    configuration.rcut = 12.0
    configuration.switch_width = 5.0
    configuration.tailcorrections = False

    seed_covalent = configuration.create_seed(kind='covalent')
    ff = yaff_generate(seed_covalent)
    energy_covalent = ff.compute()

    seed_dispersion = configuration.create_seed(kind='dispersion')
    ff = yaff_generate(seed_dispersion)
    energy_dispersion = ff.compute()

    seed_electrostatic = configuration.create_seed(kind='electrostatic')
    ff = yaff_generate(seed_electrostatic)
    energy_electrostatic = ff.compute()

    seed_nonbonded = configuration.create_seed(kind='nonbonded')
    ff = yaff_generate(seed_nonbonded)
    energy_nonbonded = ff.compute()

    seed_full = configuration.create_seed(kind='full')
    ff = yaff_generate(seed_full)
    energy_full = ff.compute()

    assert abs(energy_covalent) > 0.0
    assert abs(energy_dispersion) > 0.0
    assert abs(energy_electrostatic) > 0.0
    np.testing.assert_almost_equal(
            energy_nonbonded,
            energy_dispersion + energy_electrostatic,
            )
    np.testing.assert_almost_equal(
            energy_full,
            energy_covalent + energy_nonbonded,
            )


def test_create_seed_nonperiodic():
    system, pars = get_system('alanine')
    configuration = Configuration(system, pars)

    seed_covalent = configuration.create_seed(kind='covalent')
    ff = yaff_generate(seed_covalent)
    energy_covalent = ff.compute()

    seed_dispersion = configuration.create_seed(kind='dispersion')
    ff = yaff_generate(seed_dispersion)
    energy_dispersion = ff.compute()

    seed_electrostatic = configuration.create_seed(kind='electrostatic')
    ff = yaff_generate(seed_electrostatic)
    energy_electrostatic = ff.compute()

    seed_nonbonded = configuration.create_seed(kind='nonbonded')
    ff = yaff_generate(seed_nonbonded)
    energy_nonbonded = ff.compute()

    seed_full = configuration.create_seed(kind='full')
    ff = yaff_generate(seed_full)
    energy_full = ff.compute()

    assert abs(energy_covalent) > 0.0
    assert abs(energy_dispersion) > 0.0
    assert abs(energy_electrostatic) > 0.0
    np.testing.assert_almost_equal(
            energy_nonbonded,
            energy_dispersion + energy_electrostatic,
            )
    np.testing.assert_almost_equal(
            energy_full,
            energy_covalent + energy_nonbonded,
            )

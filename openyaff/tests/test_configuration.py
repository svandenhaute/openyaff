import numpy as np
import pytest

from openyaff import Configuration
from openyaff.utils import yaff_generate

from systems import get_system, here


def test_initialize_periodic(tmp_path):
    system, pars = get_system('lennardjones')
    configuration = Configuration(system, pars)
    configuration.log_system()

    # write defaults
    path_config = tmp_path / 'config.yml'
    config = configuration.write(path_config)
    with open(path_config, 'r') as f:
        content = f.read()
    assert content == """yaff:
  interaction_radius: 10.0
  rcut: 10.0
  supercell: auto
  switch_width: 4.0
  tailcorrections: false
""" # whitespace matters


def test_initialize_nonperiodic(tmp_path):
    system, pars = get_system('alanine')
    configuration = Configuration(system, pars)
    configuration.log_system()

    # write defaults
    path_config = tmp_path / 'config.yml'
    config = configuration.write(path_config)
    with open(path_config, 'r') as f:
        content = f.read()
    assert content == """yaff: {}\n"""


def test_update_properties(tmp_path):
    system, pars = get_system('mil53')
    configuration = Configuration(system, pars)

    config = configuration.write()
    config['yaff']['rcut'] = 15.0
    config['yaff']['interaction_radius'] = 15.0
    configuration.update_properties(config)
    assert configuration.rcut == 15.0


def test_from_files(tmp_path):
    system, pars = get_system('mil53')
    configuration = Configuration(system, pars)

    configuration.write(tmp_path / 'config.yml')
    system.to_file(str(tmp_path / 'system.chk'))
    with open(tmp_path / 'pars.txt', 'w+') as f:
        f.write(pars)

    path_system = tmp_path / 'system.chk'
    path_pars   = tmp_path / 'pars.txt'
    path_config = tmp_path / 'config.yml'
    configuration = Configuration.from_files(
            path_system,
            path_pars,
            path_config,
            )


def test_create_seed_periodic():
    system, pars = get_system('cau13')
    configuration = Configuration(system, pars)
    # change parameters randomly
    with pytest.raises(ValueError): # cell is too small
        configuration.supercell = [3, 1, 1]
    configuration.rcut = 12.0
    assert configuration.interaction_radius == 12.0 # should change too
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

    seed_full = configuration.create_seed(kind='all')
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

    seed_full = configuration.create_seed(kind='all')
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


def test_get_prefixes():
    system, pars = get_system('cau13')
    configuration = Configuration(system, pars)

    prefixes = configuration.get_prefixes('all')
    covalent_prefixes = configuration.get_prefixes('covalent')
    dispersion_prefixes = configuration.get_prefixes('dispersion')
    electrostatic_prefixes = configuration.get_prefixes('electrostatic')
    nonbonded_prefixes = configuration.get_prefixes('nonbonded')

    _ = dispersion_prefixes + electrostatic_prefixes
    assert tuple(sorted(_)) == tuple(sorted(nonbonded_prefixes))
    __ = covalent_prefixes + nonbonded_prefixes
    assert tuple(sorted(__)) == tuple(sorted(prefixes))

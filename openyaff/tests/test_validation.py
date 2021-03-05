from openyaff.cli import initialize, convert, validate

from systems import get_system


def test_cli_cycle(tmp_path):
    # generate input files in temporary folder
    system, pars = get_system('uio66')
    path_system = tmp_path / 'system.chk'
    path_pars = tmp_path / 'pars.txt'
    system.to_file(str(path_system))
    with open(path_pars, 'w+') as f:
        f.write(pars)

    # initialize, write configured system; verify files are created
    initialize(tmp_path, save_xyz=True)
    assert (tmp_path / 'configured_system.xyz').exists()
    assert (tmp_path / 'configured_system.h5').exists()
    assert (tmp_path / 'config.yml').exists()

    # convert all, save topology and system file
    convert(tmp_path, 'all', full=True, ludicrous=False)
    assert (tmp_path / 'topology.pdb').exists()
    assert (tmp_path / 'system.xml').exists()

    # validate
    validate(tmp_path)

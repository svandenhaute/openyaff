from openyaff.cli import initialize, save, convert, validate

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
    input_files = {
            'chk': path_system,
            'txt': path_pars,
            }
    initialize(tmp_path, input_files)
    path_yml = tmp_path / 'config.yml'
    assert path_yml.exists()
    input_files['yml'] = path_yml
    save(tmp_path, input_files, ['xyz', 'h5', 'pdb'])
    assert (tmp_path / 'configured_system.xyz').exists()
    assert (tmp_path / 'configured_system.h5').exists()

    # convert all, save topology and system file
    convert(tmp_path, input_files, 'all', full=True)
    assert (tmp_path / 'topology.pdb').exists()
    assert (tmp_path / 'system.xml').exists()

    # validate
    validate(tmp_path, input_files)

import argparse

from pathlib import Path

from openyaff import Configuration


def get_input_files(path_dir, filetypes):
    """Verifies the existence of precisely one .chk and .txt file and returns
    their paths.

    Parameters
    ----------

    path_dir : pathlib.Path
        path of directory to search in

    filetypes : list of str
        extensions of files to look for, e.g. ['.chk', '.txt']

    Returns
    -------

    path_list : list of pathlib.Path
        list of output paths, of same length as filetypes

    """
    suffixes = [p.suffix for p in path_dir.iterdir()]
    path_list = []
    for filetype in filetypes:
        assert filetype in suffixes # verify correct files are present
        index = suffixes.index(filetype)
        _ = suffixes.pop(index)
        assert filetype not in suffixes # verify uniqueness
        path_list.append(list(path_dir.iterdir())[index])
    return path_list


def test():
    print('works!')


def configure(cwd):
    input_files = get_input_files(cwd, ['.chk', '.txt'])
    path_ini = cwd / 'config.ini'
    configuration = Configuration.from_files(*input_files, path_ini)
    configuration.write(path_ini)


def main():
    parser = argparse.ArgumentParser(
            description='conversion and testing of YAFF force fields to OpenMM-compatible format',
            )
    parser.add_argument(
            'mode',
            action='store',
            help='determines mode of operation: init, compare, convert, test',
            )
    args = parser.parse_args()

    cwd = Path.cwd()
    if args.mode == 'test':
        test()
    elif args.mode == 'configure':
        configure(cwd)
    else:
        raise NotImplementedError

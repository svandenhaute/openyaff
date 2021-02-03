import argparse
import logging
import yaff

from pathlib import Path

from openyaff.utils import add_header_to_config
from openyaff import Configuration, load_conversion, load_validations, \
        ExplicitConversion, SinglePointValidation


yaff.log.set_level(yaff.log.silent)

# enable logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__) # logging per module


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


def validate(cwd):
    input_files = get_input_files(cwd, ['.chk', '.txt'])
    path_yml = cwd / 'config.yml'
    configuration = Configuration.from_files(*input_files, path_yml)
    conversion = load_conversion(path_yml)
    validations = load_validations(path_yml)
    for validation in validations:
        validation.run(configuration, conversion)

def convert(cwd, seed_kind):
    input_files = get_input_files(cwd, ['.chk', '.txt'])
    path_yml = cwd / 'config.yml'
    configuration = Configuration.from_files(*input_files, path_yml)
    conversion = load_conversion(path_yml)
    openmm_seed = conversion.apply(configuration, seed_kind)
    save_openmm_system(openmm_seed.system_mm, cwd / 'system.xml')


def initialize(cwd):
    input_files = get_input_files(cwd, ['.chk', '.txt'])
    path_yml = cwd / 'config.yml'
    logger.info('found the following input files:')
    for file in input_files:
        logger.info(str(file))
    logger.info('')

    if path_yml.exists(): # remove file if it exists
        path_yml.unlink()
    default_classes = [ # classes for which to initialize .yml keywords
            ExplicitConversion,
            SinglePointValidation,
            ]

    # initialize Configuration based on defaults
    configuration = Configuration.from_files(*input_files)
    supercell = configuration.determine_supercell(configuration.rcut)
    configuration.supercell = supercell # smallest usable supercell
    configuration.log()
    configuration.write(path_yml)
    for default in default_classes:
        default().write(path_yml)

    # add annotations to config file for clarity; this cannot be done using
    # pyyaml and hence proceeds manually
    add_header_to_config(path_yml)
    configuration.annotate(path_yml)
    for default in default_classes:
        default().annotate(path_yml)

    logger.info('')
    logger.info('writing configuration file to')
    logger.info(str(path_yml))


def main():
    print('')
    parser = argparse.ArgumentParser(
            description='conversion and testing of YAFF force fields to OpenMM-compatible format',
            )
    parser.add_argument(
            'mode',
            action='store',
            help='specifies mode of operation'
            )
    parser.add_argument(
        '--interaction',
        help='type of interactions to consider: covalent, dispersion, electrostatic, nonbonded, all',
        nargs='?',
        const='all', # default behavior is to include all interactions
        )
    args = parser.parse_args()

    cwd = Path.cwd()
    if args.mode == 'test':
        test()
    elif args.mode == 'initialize':
        initialize(cwd)
    elif args.mode == 'convert':
        seed_kind = args.interaction
        assert seed_kind in ['all', 'covalent', 'dispersion', 'electrostatic', 'all']
        convert(cwd, seed_kind=seed_kind)
        pass
    elif args.mode == 'validate':
        validate(cwd)
    else:
        raise NotImplementedError

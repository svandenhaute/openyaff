import argparse
import logging
import yaff

from pathlib import Path

from openyaff.utils import add_header_to_config, determine_rcut, \
        save_openmm_system
from openyaff import Configuration, load_conversion, load_validations, \
        ExplicitConversion, SinglePointValidation


yaff.log.set_level(yaff.log.silent)


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
    files = list(path_dir.iterdir())
    suffixes = [file.suffix for file in files]
    path_list = []
    for filetype in filetypes:
        assert filetype in suffixes # verify correct files are present
        index = suffixes.index(filetype)
        _ = suffixes.pop(index)
        assert filetype not in suffixes # verify uniqueness
        path_list.append(files.pop(index))
    return path_list


def test():
    print('works!')


def validate(cwd):
    input_files = get_input_files(cwd, ['.chk', '.txt'])
    path_yml = cwd / 'config.yml'
    configuration = Configuration.from_files(*input_files, path_yml)
    # TODO: check rcut setting
    conversion = load_conversion(path_yml)
    validations = load_validations(path_yml)
    for validation in validations:
        validation.run(configuration, conversion)


def convert(cwd, seed_kind):
    input_files = get_input_files(cwd, ['.chk', '.txt'])
    path_yml = cwd / 'config.yml'
    configuration = Configuration.from_files(*input_files, path_yml)
    configuration.log_config()
    conversion = load_conversion(path_yml)
    openmm_seed = conversion.apply(configuration, seed_kind)
    path_xml = cwd / 'system.xml'
    logger.info('saving OpenMM System object to ')
    logger.info(path_xml)
    save_openmm_system(openmm_seed.system_mm, path_xml)


def initialize(cwd, save_reduced=False):
    input_files = get_input_files(cwd, ['.chk', '.txt'])
    path_yml = cwd / 'config.yml'
    path_xyz = cwd / 'reduced.xyz'
    path_h5 = cwd / 'reduced.h5'
    logger.info('found the following input files:')
    for file in input_files:
        logger.info(str(file.name))
    logger.info('')

    if path_yml.exists(): # remove file if it exists
        path_yml.unlink()
    default_classes = [ # classes for which to initialize .yml keywords
            ExplicitConversion,
            SinglePointValidation,
            ]

    # initialize Configuration based on defaults
    configuration = Configuration.from_files(*input_files)
    # check if rcut should be determined 
    nonbonded_prefixes = configuration.get_prefixes('nonbonded')
    if configuration.periodic and (len(nonbonded_prefixes) > 0):
        supercell = configuration.determine_supercell(configuration.rcut)
        configuration.supercell = supercell # smallest usable supercell
    configuration.log_system()
    configuration.log_config()
    logger.info('')
    logger.info('writing configuration file to')
    logger.info(str(path_yml))
    configuration.write(path_yml)
    for default in default_classes:
        default().write(path_yml)

    # add annotations to config file for clarity; this cannot be done using
    # pyyaml and hence proceeds manually
    add_header_to_config(path_yml)
    configuration.annotate(path_yml)
    for default in default_classes:
        default().annotate(path_yml)

    if save_reduced:
        if path_xyz.exists():
            path_xyz.unlink()
        if path_h5.exists():
            path_h5.unlink()
        logger.info('saving YAFF system with reduced box vectors to files')
        logger.info(str(path_xyz))
        configuration.system.to_file(str(path_xyz))
        logger.info(str(path_h5))
        configuration.system.to_file(str(path_h5))


def main():
    print('')
    parser = argparse.ArgumentParser(
            description='conversion and testing of YAFF force fields to OpenMM-compatible format',
            allow_abbrev=False,
            )
    parser.add_argument(
            'mode',
            action='store',
            choices=['test', 'initialize', 'convert', 'validate'],
            default='test',
            help='specifies mode of operation',
            )
    parser.add_argument(
            '-i',
            '--interaction',
            action='store',
            default='all',
            choices=['all', 'covalent', 'dispersion', 'electrostatic', 'nonbonded'],
            help='type of interactions to consider',
            )
    parser.add_argument(
            '-s',
            '--save-reduced',
            help='save YAFF system with reduced box vectors',
            action='store_true',
            default=False,
            )
    parser.add_argument(
            '-d',
            '--debug',
            action='store_true',
            default=False,
            help='enables debug mode',
            )
    args = parser.parse_args()

    # enable logging
    logging_format = '%(name)s - %(message)s'
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    cwd = Path.cwd()
    if args.mode == 'test':
        test()
    elif args.mode == 'initialize':
        initialize(cwd, args.save_reduced)
    elif args.mode == 'convert':
        seed_kind = args.interaction
        assert seed_kind in ['all', 'covalent', 'dispersion', 'electrostatic', 'all']
        convert(cwd, seed_kind=seed_kind)
        pass
    elif args.mode == 'validate':
        validate(cwd)
    else:
        raise NotImplementedError

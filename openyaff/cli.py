import molmod
import argparse
import logging
import yaff
import simtk.openmm as mm
import simtk.unit as unit
import simtk.openmm.app
import numpy as np

from pathlib import Path

from openyaff.utils import add_header_to_config, determine_rcut, \
        create_openmm_topology
from openyaff import Configuration, load_conversion, load_validations, \
        ExplicitConversion, SinglePointValidation, StressValidation, \
        OpenMMForceFieldWrapper


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
    configuration.log_config()
    conversion = load_conversion(path_yml)
    validations = load_validations(path_yml)
    for validation in validations:
        validation.run(configuration, conversion)


def convert(cwd, seed_kind, full, ludicrous):
    input_files = get_input_files(cwd, ['.chk', '.txt'])
    path_yml = cwd / 'config.yml'
    configuration = Configuration.from_files(*input_files, path_yml)
    configuration.log_config()
    conversion = load_conversion(path_yml)

    path_xml = cwd / 'system.xml'
    logger.info('saving OpenMM System object to ')
    logger.info(path_xml)
    if not ludicrous:
        openmm_seed = conversion.apply(configuration, seed_kind)
    else:
        logger.info('converting in ludicrous mode...')
        assert seed_kind == 'all' # only useful situation to use ludicrous
        openmm_seed = conversion.apply_ludicrous(configuration)
    openmm_seed.serialize(path_xml)

    if full: # write additional files
        yaff_seed = configuration.create_seed(seed_kind)
        topology = create_openmm_topology(yaff_seed.system)
        if yaff_seed.system.cell.nvec != 0: # check box vectors are included
            assert topology.getPeriodicBoxVectors() is not None
        u = molmod.units.angstrom / unit.angstrom
        mm.app.PDBFile.writeFile(
                topology,
                yaff_seed.system.pos / u,
                open(cwd / 'topology.pdb', 'w+'),
                keepIds=True,
                )


def save(cwd, file_formats):
    input_files = get_input_files(cwd, ['.chk', '.txt'])
    path_yml = cwd / 'config.yml'
    configuration = Configuration.from_files(*input_files, path_yml)


    logger.info('saving system with current supercell configuration in the'
            ' following formats:')
    for file_format in file_formats:
        logger.info('\t\t' + file_format)
    logger.info('cell vectors are stored in reduced form')
    seed = configuration.create_seed()
    for file_format in file_formats:
        path_file = cwd / ('configured_system.' + file_format)
        if path_file.exists():
            path_file.unlink()
        if (file_format == 'xyz') or (file_format == 'h5'):
            seed.system.to_file(str(path_file))
        elif file_format == 'pdb':
            topology = create_openmm_topology(seed.system)
            if seed.system.cell.nvec != 0: # check box vectors are included
                assert topology.getPeriodicBoxVectors() is not None
            u = molmod.units.angstrom / unit.angstrom
            mm.app.PDBFile.writeFile(
                    topology,
                    seed.system.pos / u,
                    open(path_file, 'w+'),
                    keepIds=True,
                    )


def initialize(cwd):
    input_files = get_input_files(cwd, ['.chk', '.txt'])
    path_yml = cwd / 'config.yml'
    path_xyz = cwd / 'configured_system.xyz'
    path_h5 = cwd / 'configured_system.h5'
    logger.info('found the following input files:')
    for file in input_files:
        logger.info(str(file.name))
    logger.info('')

    if path_yml.exists(): # remove file if it exists
        path_yml.unlink()
    default_classes = [ # classes for which to initialize .yml keywords
            ExplicitConversion,
            SinglePointValidation,
            StressValidation,
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


def main():
    print('')
    parser = argparse.ArgumentParser(
            description='conversion and testing of YAFF force fields to OpenMM-compatible format',
            allow_abbrev=False,
            )
    parser.add_argument(
            'mode',
            action='store',
            choices=['test', 'initialize', 'save', 'convert', 'validate'],
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
            '-d',
            '--debug',
            action='store_true',
            default=False,
            help='enables debug mode',
            )
    parser.add_argument(
            '--xyz',
            action='store_true',
            default=False,
            help='save configured system in XYZ file format',
            )
    parser.add_argument(
            '--h5',
            action='store_true',
            default=False,
            help='save configured system in HDF5 file format',
            )
    parser.add_argument(
            '--pdb',
            action='store_true',
            default=False,
            help='save configured system in PDB file format',
            )
    parser.add_argument(
            '-f',
            '--full',
            action='store_true',
            default=False,
            help='write topology and state files necessary to run simulations',
            )
    parser.add_argument(
            '-l',
            '--ludicrous',
            action='store_true',
            default=False,
            help=('enable ludicrous mode. This is strongly recommended for'
                 ' systems containing over one million atoms'),
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
        initialize(cwd)
    elif args.mode == 'save':
        file_formats = []
        if args.xyz:
            file_formats.append('xyz')
        if args.h5:
            file_formats.append('h5')
        if args.pdb:
            file_formats.append('pdb')
        save(cwd, file_formats)
    elif args.mode == 'convert':
        seed_kind = args.interaction
        assert seed_kind in ['all', 'covalent', 'dispersion', 'electrostatic']
        convert(
                cwd,
                seed_kind=seed_kind,
                full=args.full,
                ludicrous=args.ludicrous,
                )
        pass
    elif args.mode == 'validate':
        validate(cwd)
    else:
        raise NotImplementedError

import molmod
import argparse
import logging
import yaff
import simtk.openmm as mm
import simtk.unit as unit
import simtk.openmm.app
import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path

from openyaff.utils import add_header_to_config, determine_rcut, \
        create_openmm_topology
from openyaff import Configuration, load_conversion, load_validations, \
        ExplicitConversion, SinglePointValidation, StressValidation, \
        OpenMMForceFieldWrapper


yaff.log.set_level(yaff.log.silent)


logger = logging.getLogger(__name__) # logging per module


def validate(cwd, input_files):
    assert 'chk' in input_files
    assert 'txt' in input_files
    assert 'yml' in input_files
    configuration = Configuration.from_files(**input_files)
    configuration.log_config()
    conversion = load_conversion(input_files['yml'])
    validations = load_validations(input_files['yml'])
    for validation in validations:
        validation.run(configuration, conversion)


def convert(cwd, input_files, seed_kind, full):
    assert 'chk' in input_files
    assert 'txt' in input_files
    assert 'yml' in input_files
    configuration = Configuration.from_files(**input_files)
    configuration.log_config()
    conversion = load_conversion(input_files['yml'])

    if conversion.kind == 'explicit':
        path_xml = cwd / 'system.xml'
        logger.info('saving OpenMM System object to ')
        logger.info(path_xml)
        openmm_seed = conversion.apply(configuration, seed_kind)
        openmm_seed.serialize_system(path_xml)
    elif conversion.kind == 'implicit':
        openmm_seed = conversion.apply(configuration, seed_kind)
        path_xml = cwd / 'ff.xml'
        logger.info('saving OpenMM ForceField object to ')
        logger.info(path_xml)
        openmm_seed.serialize_forcefield(path_xml)
        path_xml = cwd / 'system.xml'
        logger.info('saving OpenMM ForceField object to ')
        logger.info(path_xml)
        openmm_seed.serialize_system(path_xml)

    if full: # write additional files
        topology, positions = configuration.create_topology()
        path_pdb = cwd / 'topology.pdb'
        if path_pdb.exists():
            path_pdb.unlink()
        mm.app.PDBFile.writeFile(
                topology,
                positions * unit.angstrom,
                open(path_pdb, 'w+'),
                keepIds=True,
                )


def save(cwd, input_files, file_formats):
    assert 'chk' in input_files
    assert 'txt' in input_files
    assert 'yml' in input_files
    configuration = Configuration.from_files(**input_files)

    logger.info('saving configured system in the following formats:')
    for file_format in file_formats:
        logger.info('\t\t' + file_format)
    logger.info('If the system is periodic, its box vectors are stored in reduced form')
    seed = configuration.create_seed()
    for file_format in file_formats:
        path_file = cwd / ('configured_system.' + file_format)
        if path_file.exists():
            path_file.unlink()
        if (file_format == 'xyz') or (file_format == 'h5'):
            seed.system.to_file(str(path_file))
        elif file_format == 'pdb':
            topology, positions = configuration.create_topology()
            path_pdb = cwd / 'topology.pdb'
            if path_pdb.exists():
                path_pdb.unlink()
            mm.app.PDBFile.writeFile(
                    topology,
                    positions * unit.angstrom,
                    open(path_pdb, 'w+'),
                    keepIds=True,
                    )


def initialize(cwd, input_files):
    assert 'chk' in input_files
    assert 'txt' in input_files
    path_yml = cwd / 'config.yml'
    if path_yml.exists(): # remove file if it exists
        path_yml.unlink()
    default_classes = [ # classes for which to initialize .yml keywords
            ExplicitConversion,
            SinglePointValidation,
            StressValidation,
            ]

    # initialize Configuration based on defaults; log
    configuration = Configuration.from_files(**input_files)
    configuration.log_system()
    configuration.log_config()
    logger.info('writing configuration file to')
    logger.info(str(path_yml))
    configuration.write(path_yml)
    for default in default_classes:
        default().write(path_yml)

    # add annotations to config file for clarity; this cannot be done using
    # pyyaml and therefore proceeds manually
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
            choices=['initialize', 'save', 'convert', 'validate'],
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
            'input_files',
            type=argparse.FileType('r'),
            nargs='+',
            action='store',
            help='input files to consider',
            )
    args = parser.parse_args()

    # enable logging
    logging_format = '%(name)s - %(message)s'
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    # organize input files per extension
    input_files = {}
    for file in args.input_files:
        path = Path(file.name)
        input_files[path.suffix[1:]] = path # remove initial point from ext
    logger.info('using the following input files:')
    for file in input_files.values():
        logger.info(str(file.name))
    logger.info('')

    cwd = Path.cwd()
    if args.mode == 'initialize':
        initialize(cwd, input_files)
    elif args.mode == 'save':
        file_formats = []
        if args.xyz:
            file_formats.append('xyz')
        if args.h5:
            file_formats.append('h5')
        if args.pdb:
            file_formats.append('pdb')
        save(cwd, input_files, file_formats)
    elif args.mode == 'convert':
        seed_kind = args.interaction
        assert seed_kind in ['all', 'covalent', 'dispersion', 'electrostatic']
        convert(
                cwd,
                input_files=input_files,
                seed_kind=seed_kind,
                full=args.full,
                )
        pass
    elif args.mode == 'validate':
        validate(cwd, input_files)
    else:
        raise NotImplementedError

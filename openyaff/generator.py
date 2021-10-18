import yaff
import logging
import numpy as np
import xml.etree.ElementTree as ET
import molmod
import simtk.unit as unit
import simtk.openmm as mm


logger = logging.getLogger(__name__) # logging per module


class ValenceMirroredGenerator(yaff.ValenceGenerator):
    """Class to generate OpenMM and YAFF force field simultaneously

    This subclass only overwrites two parent methods:

        __call__:
            the OpenMM force object is generated just before apply() is
            called.

        apply:
            each time a vterm is added to part_valence, the coresponding
            indices and pars are added to the OpenMM force object

    """

    def __call__(self, system, parsec, ff_args, **kwargs):
        '''Add contributions to the force field from a ValenceGenerator

           **Arguments:**

           system
                The System object for which a force field is being prepared

           parse
                An instance of the ParameterSection class

           ff_ars
                An instance of the FFargs class
        '''
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, self.nffatype)
        periodic = (system.cell.nvec == 3)
        force = self.get_force(periodic)
        if len(par_table) > 0:
            self.apply(par_table, system, ff_args, force)
        return force

    def apply(self, par_table, system, ff_args, force):
        '''Generate terms for the system based on the par_table

           **Arguments:**

           par_table
                A dictionary with tuples of ffatypes is keys and lists of
                parameters as values.

           system
                The system for which the force field is generated.

           ff_args
                An instance of the FFArgs class.

            force
                OpenMM force instance
        '''
        if system.bonds is None:
            raise ValueError('The system must have bonds in order to define valence terms.')
        part_valence = ff_args.get_part_valence(system)
        for indexes in self.iter_indexes(system):
            # We do not want terms where at least one atom index is smaller than
            # nlow, as this is (should be) an excluded interaction
            if min(indexes)<ff_args.nlow:
                # Check that this term indeed features only atoms with index<nlow
                assert max(indexes)<ff_args.nlow
                continue
            # We do not want terms where at least one atom index is higher than
            # or equal to nhigh, as this is (should be) an excluded interaction
            if ff_args.nhigh!=-1 and max(indexes)>=ff_args.nhigh:
                # Check that this term indeed features only atoms with index<nlow
                assert min(indexes)>=ff_args.nhigh
                continue
            key = tuple(system.get_ffatype(i) for i in indexes)
            par_list = par_table.get(key, [])
            for pars in par_list:
                #vterm = self.get_vterm(pars, indexes)
                #assert vterm is not None # neccessary?
                #part_valence.add_term(vterm)
                self.add_term_to_force(force, pars, indexes)

    def add_term_to_force(self, force, pars, indexes):
        raise NotImplementedError

    def parse_xml(self, system, section, ff_args, **kwargs):
        self.check_suffixes(section)
        conversions = self.process_units(section['UNIT'])
        par_table = self.process_pars(section['PARS'], conversions, self.nffatype)
        self.clean_par_table(par_table)
        forces = self._internal_parse(par_table)
        return forces

    def clean_par_table(self, par_table):
        # based on self.iter_equiv_keys_and_pars
        raise NotImplementedError

    def _internal_parse(self, par_table):
        raise NotImplementedError


class BondGenerator(ValenceMirroredGenerator):
    par_info = [('K', float), ('R0', float)]
    nffatype = 2
    ICClass = yaff.Bond
    VClass = None

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_bonds()

    def clean_par_table(self, par_table):
        for key in list(par_table.keys()):
            if key[::-1] in par_table:
                par_table.pop(key)


class BondHarmGenerator(BondGenerator):
    prefix = 'BONDHARM'
    VClass = yaff.Harmonic

    def get_force(self, periodic):
        force = mm.HarmonicBondForce()
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'K': molmod.units.kjmol / molmod.units.angstrom ** 2 / 100,
                'R0': molmod.units.angstrom * 10,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole / unit.nanometer ** 2,
                'R0': unit.nanometer,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        R0 = pars[1] / conversion['R0'] * conversion_mm['R0']
        force.addBond(int(indexes[0]), int(indexes[1]), R0, K)

    def _internal_parse(self, par_table):
        force_xml = ET.Element('HarmonicBondForce')
        for bond, pars in par_table.items():
            k      = pars[0][0]
            length = pars[0][1]
            attrib = {
                    'type1' : bond[0],
                    'type2' : bond[1],
                    'length': str(length / molmod.units.nanometer),
                    'k'     : str(k / molmod.units.kjmol * molmod.units.nanometer ** 2),
                    }
            e = ET.Element('Bond', attrib=attrib)
            force_xml.append(e)
        return [force_xml]


class MM3QuarticGenerator(BondGenerator):
    prefix = 'MM3QUART'
    VClass = yaff.MM3Quartic

    def get_force(self, periodic):
        energy = '0.5 * K * (r - R0)^2 * (1 - CS * (r - R0) * 10 + 7/12 * (CS * (r - R0))^2 * 100);'
        energy += ' CS = 1.349402 * {}'.format(molmod.units.angstrom)
        force = mm.CustomBondForce(energy)
        force.addPerBondParameter('K')
        force.addPerBondParameter('R0')
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'K': molmod.units.kjmol / molmod.units.nanometer ** 2,
                'R0': molmod.units.nanometer,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole / unit.nanometer ** 2,
                'R0': unit.nanometer,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        R0 = pars[1] / conversion['R0'] * conversion_mm['R0']
        force.addBond(
                int(indexes[0]),
                int(indexes[1]),
                [K, R0],
                )


class Poly4Generator(ValenceMirroredGenerator):
    nffatype = 2
    prefix = 'POLY4'
    ICClass = yaff.Bond
    VClass = yaff.Poly4
    par_info = [('C0', float), ('C1', float), ('C2', float), ('C3', float), ('C4', float), ('R0', float)]

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_bonds()

    def get_force(self, periodic):
        energy =  '  C0'
        energy += '+ C1 * (r - R0)^1'
        energy += '+ C2 * (r - R0)^2'
        energy += '+ C3 * (r - R0)^3'
        energy += '+ C4 * (r - R0)^4'
        force = mm.CustomBondForce(energy)
        force.addPerBondParameter('C0')
        force.addPerBondParameter('C1')
        force.addPerBondParameter('C2')
        force.addPerBondParameter('C3')
        force.addPerBondParameter('C4')
        force.addPerBondParameter('R0')
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'C0': molmod.units.kjmol / molmod.units.angstrom ** 0,
                'C1': molmod.units.kjmol / molmod.units.angstrom ** 1,
                'C2': molmod.units.kjmol / molmod.units.angstrom ** 2,
                'C3': molmod.units.kjmol / molmod.units.angstrom ** 3,
                'C4': molmod.units.kjmol / molmod.units.angstrom ** 4,
                'R0': molmod.units.angstrom,
                }
        conversion_mm = {
                'C0': unit.kilojoule_per_mole / unit.angstrom ** 0,
                'C1': unit.kilojoule_per_mole / unit.angstrom ** 1,
                'C2': unit.kilojoule_per_mole / unit.angstrom ** 2,
                'C3': unit.kilojoule_per_mole / unit.angstrom ** 3,
                'C4': unit.kilojoule_per_mole / unit.angstrom ** 4,
                'R0': unit.angstrom,
                }
        C0 = pars[0] / conversion['C0'] * conversion_mm['C0']
        C1 = pars[1] / conversion['C1'] * conversion_mm['C1']
        C2 = pars[2] / conversion['C2'] * conversion_mm['C2']
        C3 = pars[3] / conversion['C3'] * conversion_mm['C3']
        C4 = pars[4] / conversion['C4'] * conversion_mm['C4']
        R0 = pars[5] / conversion['R0'] * conversion_mm['R0']
        force.addBond(
                int(indexes[0]),
                int(indexes[1]),
                [C0, C1, C2, C3, C4, R0],
                )


class BendGenerator(ValenceMirroredGenerator):
    nffatype = 3
    ICClass = None
    VClass = yaff.Harmonic

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_angles()

    def clean_par_table(self, par_table):
        pass


class BendAngleHarmGenerator(BendGenerator):
    par_info = [('K', float), ('THETA0', float)]
    prefix = 'BENDAHARM'
    ICClass = yaff.BendAngle

    def get_force(self, periodic):
        force = mm.HarmonicAngleForce()
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'K': molmod.units.kjmol / molmod.units.rad ** 2,
                'THETA0': molmod.units.deg,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole / unit.radians ** 2,
                'THETA0': unit.degrees,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        THETA0 = pars[1] / conversion['THETA0'] * conversion_mm['THETA0']
        force.addAngle(
                int(indexes[0]),
                int(indexes[1]),
                int(indexes[2]),
                THETA0,
                K,
                )

    def _internal_parse(self, par_table):
        force_xml = ET.Element('HarmonicAngleForce')
        for bend, pars in par_table.items():
            k     = pars[0][0]
            angle = pars[0][1]
            attrib = {
                    'type1' : bend[0],
                    'type2' : bend[1],
                    'type3' : bend[2],
                    'angle': str(angle / molmod.units.rad),
                    'k'     : str(k / molmod.units.kjmol * molmod.units.rad ** 2),
                    }
            e = ET.Element('Angle', attrib=attrib)
            force_xml.append(e)
        return [force_xml]


class MM3BendGenerator(BendGenerator):
    nffatype = 3
    par_info = [('K', float), ('THETA0', float)]
    ICClass = yaff.BendAngle
    VClass = yaff.MM3Bend
    prefix = 'MM3BENDA'

    def get_force(self, periodic):
        #energy = '0.5*K*(theta-R0)^2*(1-0.14*(theta-R0)+5.6*10^(-5)*(theta-R0)^2-7*10^(-7)*(theta-R0)^3+2.2*10^(-8)*(theta-R0)^4); '
        energy = '0.5*K*x^2*(1-0.014*x+5.6*10^(-5)*x^2-7*10^(-7)*x^3+2.2*10^(-8)*x^4) * {}; '.format(molmod.units.deg ** 2)
        energy += 'x=(theta-R0) / {}'.format(molmod.units.deg)
        force = mm.CustomAngleForce(energy)
        force.addPerAngleParameter('K')
        force.addPerAngleParameter('R0')
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'K': molmod.units.kjmol / molmod.units.rad ** 2,
                'R0': molmod.units.rad,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole / unit.radians ** 2,
                'R0': unit.radians,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        R0 = pars[1] / conversion['R0'] * conversion_mm['R0']
        force.addAngle(
                int(indexes[0]),
                int(indexes[1]),
                int(indexes[2]),
                [K, R0],
                )


class BendCosGenerator(BendGenerator):
    par_info = [('M', int), ('A', float), ('PHI0', float)]
    prefix = 'BENDCOS'
    ICClass = yaff.BendAngle
    VClass = yaff.Cosine

    def get_force(self, periodic):
        energy = 'A/2*(1 - cos(M*(theta - PHI0)))'
        force = mm.CustomAngleForce(energy)
        force.addPerAngleParameter('M')
        force.addPerAngleParameter('A')
        force.addPerAngleParameter('PHI0')
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'A': molmod.units.kjmol,
                'PHI0': molmod.units.deg,
                'M': 1,
                }
        conversion_mm = {
                'A': unit.kilojoule_per_mole,
                'PHI0': unit.degrees,
                'M': unit.dimensionless,
                }
        M = pars[0] / conversion['M'] * conversion_mm['M']
        A = pars[1] / conversion['A'] * conversion_mm['A']
        PHI0 = pars[2] / conversion['PHI0'] * conversion_mm['PHI0']
        force.addAngle(
                int(indexes[0]),
                int(indexes[1]),
                int(indexes[2]),
                [M, A, PHI0],
                )


class BendCosHarmGenerator(BendGenerator):
    par_info = [('K', float), ('COS0', float)]
    prefix = 'BENDCHARM'
    ICClass = yaff.BendCos

    def get_force(self, periodic):
        energy = 'K/2*(cos(theta) - COS0)^2'
        force = mm.CustomAngleForce(energy)
        force.addPerAngleParameter('K')
        force.addPerAngleParameter('COS0')
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'K': molmod.units.kjmol,
                'COS0': 1.0,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole,
                'COS0': 1.0,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        COS0 = pars[1] / conversion['COS0'] * conversion_mm['COS0']
        force.addAngle(
                int(indexes[0]),
                int(indexes[1]),
                int(indexes[2]),
                [K, COS0],
                )


class OopDistGenerator(ValenceMirroredGenerator):
    nffatype = 4
    par_info = [('K', float), ('D0', float)]
    prefix = 'OOPDIST'
    ICClass = yaff.OopDist
    VClass = yaff.Harmonic
    allow_superposition = False

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield (key[2], key[0], key[1], key[3]), pars
        yield (key[1], key[2], key[0], key[3]), pars
        yield (key[2], key[1], key[0], key[3]), pars
        yield (key[1], key[0], key[2], key[3]), pars
        yield (key[0], key[2], key[1], key[3]), pars

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopDist term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                yield neighbours[0],neighbours[1],neighbours[2],atom

    def get_force(self, periodic):
        dist = OopDistGenerator._get_dist()
        energy = '0.5 * K * ({} - D0)^2'.format(dist)
        force = mm.CustomCompoundBondForce(4, energy)
        force.addPerBondParameter('K')
        force.addPerBondParameter('D0')
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    @staticmethod
    def _get_dist():
        dist = 'distance(p2, p4) * sin(angle(p4, p2, p3)) * sin(dihedral(p1, p2, p3, p4))'
        return dist

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'K': molmod.units.kjmol / molmod.units.nanometer ** 2,
                'D0': molmod.units.nanometer,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole / unit.nanometer ** 2,
                'D0': unit.nanometer,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        D0 = pars[1] / conversion['D0'] * conversion_mm['D0']
        force.addBond(
                [
                    int(indexes[0]),
                    int(indexes[1]),
                    int(indexes[2]),
                    int(indexes[3])],
                [K, D0],
                )


class SquareOopDistGenerator(ValenceMirroredGenerator):
    nffatype = 4
    par_info = [('K', float), ('D0', float)]
    prefix = 'SQOOPDIST'
    ICClass = yaff.pes.iclist.SqOopDist
    VClass = yaff.Harmonic
    allow_superposition = False

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield (key[2], key[0], key[1], key[3]), pars
        yield (key[1], key[2], key[0], key[3]), pars
        yield (key[2], key[1], key[0], key[3]), pars
        yield (key[1], key[0], key[2], key[3]), pars
        yield (key[0], key[2], key[1], key[3]), pars

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopDist term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                yield neighbours[0],neighbours[1],neighbours[2],atom

    def get_force(self, periodic):
        dist = OopDistGenerator._get_dist()
        energy = '0.5 * K * (({})^2 - D0)^2'.format(dist)
        force = mm.CustomCompoundBondForce(4, energy)
        force.addPerBondParameter('K')
        force.addPerBondParameter('D0')
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'K': molmod.units.kjmol / molmod.units.nanometer ** 4,
                'D0': molmod.units.nanometer ** 2,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole / unit.nanometer ** 4,
                'D0': unit.nanometer ** 2,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        D0 = pars[1] / conversion['D0'] * conversion_mm['D0']
        force.addBond(
                [
                    int(indexes[0]),
                    int(indexes[1]),
                    int(indexes[2]),
                    int(indexes[3])],
                [K, D0],
                )


class OopCosGenerator(ValenceMirroredGenerator):
    nffatype = 4
    par_info = [('A', float)]
    prefix = 'OOPCOS'
    ICClass = yaff.OopCos
    VClass = yaff.Chebychev1
    allow_superposition = True

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield (key[1], key[0], key[2], key[3]), pars

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopCos term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                #Yield a term for all three out-of-plane angles
                #with atom as center atom
                yield neighbours[0],neighbours[1],neighbours[2],atom
                yield neighbours[1],neighbours[2],neighbours[0],atom
                yield neighbours[2],neighbours[0],neighbours[1],atom

    def get_vterm(self, pars, indexes):
        ic = yaff.OopCos(*indexes)
        return yaff.Chebychev1(pars[0], ic)

    def get_force(self, periodic):
        cosangle = OopCosGenerator._get_cosangle()
        energy = '0.5 * A * (1 - {})'.format(cosangle)
        force = mm.CustomCompoundBondForce(4, energy)
        force.addPerBondParameter('A')
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    @staticmethod
    def _get_cosangle():
        dist = 'distance(p2, p3) * sin(angle(p3, p2, p4)) * sin(dihedral(p1, p2, p4, p3))'
        sinangle = '({}) / distance(p3, p4)'.format(dist)
        cosangle = 'sqrt(1 - ({})^2)'.format(sinangle)
        return cosangle

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'A': molmod.units.kjmol,
                }
        conversion_mm = {
                'A': unit.kilojoule_per_mole,
                }
        A = pars[0] / conversion['A'] * conversion_mm['A']
        force.addBond(
                [
                    int(indexes[0]),
                    int(indexes[1]),
                    int(indexes[2]),
                    int(indexes[3])],
                [A],
                )
        assert len(pars) == 1


class TorsionGenerator(ValenceMirroredGenerator):
    nffatype = 4
    par_info = [('M', int), ('A', float), ('PHI0', float)]
    prefix = 'TORSION'
    ICClass = yaff.DihedAngle
    VClass = yaff.Cosine
    allow_superposition = True

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_dihedrals()

    def get_vterm(self, pars, indexes):
        # A torsion term with multiplicity m and rest value either 0 or pi/m
        # degrees, can be treated as a polynomial in cos(phi). The code below
        # selects the right polynomial.
        if pars[2] == 0.0 and pars[0] == 1:
            ic = yaff.DihedCos(*indexes)
            return yaff.Chebychev1(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/1)<1e-6 and pars[0] == 1:
            ic = yaff.DihedCos(*indexes)
            return yaff.Chebychev1(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 2:
            ic = yaff.DihedCos(*indexes)
            return yaff.Chebychev2(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/2)<1e-6 and pars[0] == 2:
            ic = yaff.DihedCos(*indexes)
            return yaff.Chebychev2(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 3:
            ic = yaff.DihedCos(*indexes)
            return yaff.Chebychev3(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/3)<1e-6 and pars[0] == 3:
            ic = yaff.DihedCos(*indexes)
            return yaff.Chebychev3(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 4:
            ic = yaff.DihedCos(*indexes)
            return yaff.Chebychev4(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/4)<1e-6 and pars[0] == 4:
            ic = yaff.DihedCos(*indexes)
            return yaff.Chebychev4(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 6:
            ic = yaff.DihedCos(*indexes)
            return yaff.Chebychev6(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/6)<1e-6 and pars[0] == 6:
            ic = yaff.DihedCos(*indexes)
            return yaff.Chebychev6(pars[1], ic, sign=1)
        else:
            return yaff.ValenceGenerator.get_vterm(self, pars, indexes)

    def get_force(self, periodic):
        energy = 'A/2*(1 - cos(M*(theta - PHI0)))'
        force = mm.CustomTorsionForce(energy)
        force.addPerTorsionParameter('M')
        force.addPerTorsionParameter('A')
        force.addPerTorsionParameter('PHI0')
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'A': molmod.units.kjmol,
                'PHI0': molmod.units.deg,
                'M': 1,
                }
        conversion_mm = {
                'A': unit.kilojoule_per_mole,
                'PHI0': unit.degrees,
                'M': unit.dimensionless,
                }
        M = pars[0]
        A = pars[1] / conversion['A'] * conversion_mm['A']
        PHI0 = pars[2] / conversion['PHI0'] * conversion_mm['PHI0']
        force.addTorsion(
                int(indexes[0]),
                int(indexes[1]),
                int(indexes[2]),
                int(indexes[3]),
                [M, A, PHI0],
                )

    def clean_par_table(self, par_table):
        for key in list(par_table.keys()):
            if key[::-1] in par_table:
                par_table.pop(key)

    def _internal_parse(self, par_table):
        force_xml = ET.Element('PeriodicTorsionForce')
        for tors, pars in par_table.items():
            attrib = {
                    'type1' : tors[0],
                    'type2' : tors[1],
                    'type3' : tors[2],
                    'type4' : tors[3],
                    }
            for i, pars_ in enumerate(pars):
                M    = pars_[0]
                A    = pars_[1]
                PHI  = pars_[2]
                periodicity = M
                k = A / 2
                phase = np.pi + M * PHI
                attrib['periodicity' + str(i + 1)] = str(periodicity)
                attrib['k' + str(i + 1)] = str(k / molmod.units.kjmol)
                attrib['phase' + str(i + 1)] = str(phase / molmod.units.rad)
            e = ET.Element('Proper', attrib=attrib)
            force_xml.append(e)
        return [force_xml]


class TorsionPolySixGenerator(TorsionGenerator):
    prefix = 'TORSCPOLYSIX'
    par_info = [
            ('C1', float),
            ('C2', float),
            ('C3', float),
            ('C4', float),
            ('C5', float),
            ('C6', float),
            ]
    ICClass = None
    VClass = None

    def get_vterm(self, pars, indexes):
        pass

    def get_force(self, periodic):
        energy = 'C6 * cos(theta)^6 + '
        energy += 'C5 * cos(theta)^5 + '
        energy += 'C4 * cos(theta)^4 + '
        energy += 'C3 * cos(theta)^3 + '
        energy += 'C2 * cos(theta)^2 + '
        energy += 'C1 * cos(theta)^1'
        force = mm.CustomTorsionForce(energy)
        force.addPerTorsionParameter('C1')
        force.addPerTorsionParameter('C2')
        force.addPerTorsionParameter('C3')
        force.addPerTorsionParameter('C4')
        force.addPerTorsionParameter('C5')
        force.addPerTorsionParameter('C6')
        force.setUsesPeriodicBoundaryConditions(periodic)
        force.setForceGroup(0)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'C1': molmod.units.kjmol,
                'C2': molmod.units.kjmol,
                'C3': molmod.units.kjmol,
                'C4': molmod.units.kjmol,
                'C5': molmod.units.kjmol,
                'C6': molmod.units.kjmol,
                }
        conversion_mm = {
                'C1': unit.kilojoule_per_mole,
                'C2': unit.kilojoule_per_mole,
                'C3': unit.kilojoule_per_mole,
                'C4': unit.kilojoule_per_mole,
                'C5': unit.kilojoule_per_mole,
                'C6': unit.kilojoule_per_mole,
                }
        c = conversion['C1'] * conversion_mm['C1']
        C1 = pars[0] / c
        C2 = pars[1] / c
        C3 = pars[2] / c
        C4 = pars[3] / c
        C5 = pars[4] / c
        C6 = pars[5] / c
        force.addTorsion(
                int(indexes[0]),
                int(indexes[1]),
                int(indexes[2]),
                int(indexes[3]),
                [C1, C2, C3, C4, C5, C6],
                )


class ValenceCrossMirroredGenerator(yaff.ValenceCrossGenerator):
    """Class to generate OpenMM and YAFF force field simultaneously

    This subclass only overwrites two parent methods:

        __call__:
            the OpenMM force object is generated just before apply() is
            called.

        apply:
            each time a vterm is added to part_valence, the coresponding
            indices and pars are added to the OpenMM force object

    """

    def __call__(self, system, parsec, ff_args, **kwargs):
        '''Add contributions to the force field from a ValenceCrossGenerator

           **Arguments:**

           system
                The System object for which a force field is being prepared

           parse
                An instance of the ParameterSection class

           ff_ars
                An instance of the FFargs class
        '''
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, self.nffatype)
        if len(par_table) > 0:
            forces = self.apply(par_table, system, ff_args)
            return forces

    def apply(self, par_table, system, ff_args):
        '''Generate terms for the system based on the par_table

           **Arguments:**

           par_table
                A dictionary with tuples of ffatypes is keys and lists of
                parameters as values.

           system
                The system for which the force field is generated.

           ff_args
                An instance of the FFArgs class.
        '''
        if system.bonds is None:
            raise ValueError('The system must have bonds in order to define valence cross terms.')
        part_valence = ff_args.get_part_valence(system)
        vterms = []
        ics = []
        for i in range(6):
            for j in range(i+1,6):
                VClass = self.__class__.__dict__['VClass%i%i' %(i,j)]
                if VClass is not None:
                    vterms.append([i,j,VClass])
                    if i not in ics: ics.append(i)
                    if j not in ics: ics.append(j)
        ics = sorted(ics)
        #dict for get_indexes routines
        get_indexes = {
            0: self.get_indexes0, 1: self.get_indexes1, 2: self.get_indexes2,
            3: self.get_indexes3, 4: self.get_indexes4, 5: self.get_indexes5,
        }

        periodic = (system.cell.nvec == 3)
        # dictionaries with keys (i, j)
        forces, conversions = self.get_forces(vterms, periodic)
        for indexes in self.iter_indexes(system):
            if min(indexes)<ff_args.nlow:
                # Check that this term indeed features only atoms with index<nlow
                assert max(indexes)<ff_args.nlow
                continue
            # We do not want terms where at least one atom index is higher than
            # or equal to nhigh, as this is (should be) an excluded interaction
            if ff_args.nhigh!=-1 and max(indexes)>=ff_args.nhigh:
                # Check that this term indeed features only atoms with index<nlow
                assert min(indexes)>=ff_args.nhigh
                continue
            key = tuple(system.get_ffatype(i) for i in indexes)
            par_list = par_table.get(key, [])
            for pars in par_list:
                for i, j, VClass_ij in vterms:
                    ICClass_i = self.__class__.__dict__['ICClass%i' %i]
                    assert ICClass_i is not None, 'IC%i has no ICClass defined' %i
                    ICClass_j = self.__class__.__dict__['ICClass%i' %j]
                    assert ICClass_i is not None, 'IC%i has no ICClass defined' %j
                    K_ij = pars[vterms.index([i,j,VClass_ij])]
                    rv_i = pars[len(vterms)+ics.index(i)]
                    rv_j = pars[len(vterms)+ics.index(j)]
                    args_ij = (K_ij, rv_i, rv_j, ICClass_i(*get_indexes[i](indexes)), ICClass_j(*get_indexes[j](indexes)))
                    #print('=======')
                    #print(ICClass_i.__name__, ICClass_j.__name__)
                    #print(key, indexes)

                    #part_valence.add_term(VClass_ij(*args_ij))
                    self.add_term_to_forces(forces, conversions, (i, j), indexes, K_ij, rv_i, rv_j)
            #break
        #count = 0
        #for force in forces.values():
        #    count += force.getNumBonds()
        #print('Counting {} bonds in total'.format(count))
        return list(forces.values())

    def get_forces(self, vterms):
        """Returns the appropriate force object"""
        raise NotImplementedError

    def add_term_to_force(self, forces, key, indexes, *pars):
        """Adds interaction to OpenMM force object"""
        raise NotImplementedError


class CrossGenerator(ValenceCrossMirroredGenerator):
    prefix = 'CROSS'
    par_info = [('KSS', float), ('KBS0', float), ('KBS1', float), ('R0', float), ('R1', float), ('THETA0', float)]
    nffatype = 3
    ICClass0 = yaff.Bond
    ICClass1 = yaff.Bond
    ICClass2 = yaff.BendAngle
    ICClass3 = None
    ICClass4 = None
    ICClass5 = None
    VClass01 = yaff.Cross
    VClass02 = yaff.Cross
    VClass03 = None
    VClass04 = None
    VClass05 = None
    VClass12 = yaff.Cross
    VClass13 = None
    VClass14 = None
    VClass15 = None
    VClass23 = None
    VClass24 = None
    VClass25 = None
    VClass34 = None
    VClass35 = None
    VClass45 = None

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], (pars[0], pars[2], pars[1], pars[4], pars[3], pars[5])

    def iter_indexes(self, system):
        return system.iter_angles()

    def get_indexes0(self, indexes):
        return indexes[:2]

    def get_indexes1(self, indexes):
        return indexes[1:]

    def get_indexes2(self, indexes):
        return indexes

    def get_forces(self, vterms, periodic):
        forces = {}
        conversions = {}
        for i, j, VClass_ij in vterms:
            ICClass_i = self.__class__.__dict__['ICClass%i' %i]
            ICClass_j = self.__class__.__dict__['ICClass%i' %j]
            if i == 0 and j == 1:
                energy = 'K*(distance(p1, p2) - RV0)*(distance(p2, p3) - RV1)'
                force = mm.CustomCompoundBondForce(3, energy)
                ic0_conversion = molmod.units.angstrom * 10
                ic1_conversion = molmod.units.angstrom * 10
                mm_ic0_conversion = unit.nanometer
                mm_ic1_conversion = unit.nanometer
                #key = (
                #        ('Bond', 1, 2),
                #        ('Bond', 1, 2),
                #        )
            elif i == 0 and j == 2:
                energy = 'K*(distance(p1, p2) - RV0)*(angle(p1, p2, p3) - RV1)'
                force = mm.CustomCompoundBondForce(3, energy)
                ic0_conversion = molmod.units.angstrom * 10
                ic1_conversion = 1.0
                mm_ic0_conversion = unit.nanometer
                mm_ic1_conversion = unit.radians
            elif i == 1 and j == 2:
                energy = 'K*(distance(p2, p3) - RV0)*(angle(p1, p2, p3) - RV1)'
                force = mm.CustomCompoundBondForce(3, energy)
                ic0_conversion = molmod.units.angstrom * 10
                ic1_conversion = 1.0
                mm_ic0_conversion = unit.nanometer
                mm_ic1_conversion = unit.radians
            else:
                raise NotImplementedError
            force.addPerBondParameter('K')
            force.addPerBondParameter('RV0')
            force.addPerBondParameter('RV1')
            force.setUsesPeriodicBoundaryConditions(periodic)
            force.setForceGroup(0)
            key = (i, j)
            forces[key] = force
            conversion = {
                    'K': molmod.units.kjmol / (ic0_conversion * ic1_conversion),
                    'RV0': ic0_conversion,
                    'RV1': ic1_conversion,
                    }
            conversion_mm = {
                    'K': unit.kilojoule_per_mole / (mm_ic0_conversion * mm_ic1_conversion),
                    'RV0': mm_ic0_conversion,
                    'RV1': mm_ic1_conversion,
                    }
            conversions[key] = (dict(conversion), dict(conversion_mm))
        return forces, conversions

    def add_term_to_forces(self, forces, conversions, key, indexes, *pars):
        assert(len(pars) == 3)
        force = forces[key]
        conversion, conversion_mm = conversions[key]
        particles = [int(index) for index in indexes]
        K = pars[0] / conversion['K'] * conversion_mm['K']
        RV0 = pars[1] / conversion['RV0'] * conversion_mm['RV0']
        RV1 = pars[2] / conversion['RV1'] * conversion_mm['RV1']
        force.addBond(particles, [K, RV0, RV1])


class CustomNonbondedForceGenerator:
    """Base class to add dispersion interactions to an OpenMM System"""

    def __init__(self, periodic, energy_expr, particle_params, global_params):
        """Constructor

        Parameters
        ----------

        periodic : bool
            whether the system is periodic

        energy_expr : str
            interaction energy as a function of interparticle distance and
            parameters

        particle_params : list of str
            list of particle parameters

        global_params : list of str
            list of global parameters

        """
        self.periodic = periodic
        force = mm.CustomNonbondedForce(energy_expr)
        for param in particle_params:
            force.addPerParticleParameter(param)
        for param in global_params:
            force.addGlobalParameter(param)
        force.setForceGroup(1)
        self.force = force

    def apply_exclusions(self, natom, scale_index, iterators):
        """Adds exclusions

        Parameters
        ----------

        natom : int
            number of atoms in the system

        scale_index : int
            determines index of exclusions. 0 is no exclusions, 3 is
            1-4 exclusion

        iterators : list of neighs
            list of system.neighs1/2/3 objects to represent connectivity

        """
        for k in range(scale_index):
            iterator = iterators[k]
            for i in range(natom):
                for j in iterator[i]:
                    if i < j:
                        self.force.addExclusion(i, int(j))

    def set_rcut(self, rcut=None):
        if self.periodic:
            assert rcut is not None
            self.force.setCutoffDistance(
                    rcut / molmod.units.nanometer * unit.nanometer,
                    )
            self.force.setNonbondedMethod(2)
        else:
            # only allowed for periodic systems
            assert rcut > 1e3 # should be extremely large
            self.force.setNonbondedMethod(0)

    def set_truncation(self, tr=None):
        if self.periodic:
            if tr is not None:
                self.force.setUseSwitchingFunction(True)
                self.force.setSwitchingDistance(
                        tr.width / molmod.units.nanometer * unit.nanometer,
                        )
            else:
                self.force.setUseSwitchingFunction(False)
        else:
            pass # doesn't matter in nonperiodic systems

    def set_tailcorrections(self, tail=False):
        if tail:
            assert self.periodic
        self.force.setUseLongRangeCorrection(tail)

    def add_particles(self, system):
        raise NotImplementedError


class MM3ForceGenerator(CustomNonbondedForceGenerator):

    def add_particles(self, system, sigmas, epsilons):
        for i in range(system.pos.shape[0]):
            parameters = [
                    sigmas[i] / molmod.nanometer * unit.nanometer,
                    epsilons[i] / molmod.units.kjmol * unit.kilojoule_per_mole,
                    ]
            self.force.addParticle(parameters)


class LJForceGenerator(CustomNonbondedForceGenerator):

    def add_particles(self, system, sigmas, epsilons):
        for i in range(system.pos.shape[0]):
            parameters = [
                    sigmas[i] / molmod.nanometer * unit.nanometer,
                    epsilons[i] / molmod.units.kjmol * unit.kilojoule_per_mole,
                    ]
            self.force.addParticle(parameters)


class LJCrossForceGenerator(CustomNonbondedForceGenerator):

    def add_particles(self, system):
        for i in range(system.natom):
            self.force.addParticle([system.ffatype_ids[i]])


class MM3Generator(yaff.NonbondedGenerator):
    prefix = 'MM3'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float), ('ONLYPAULI', int)]

    def __call__(self, system, parsec, ff_args, dispersion_scale_index=None,
            **kwargs):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        forces = self.apply(par_table, scale_table, system, ff_args,
                dispersion_scale_index)
        return forces

    def apply(self, par_table, scale_table, system, ff_args,
            dispersion_scale_index):
        # Prepare the atomic parameters
        sigmas = np.zeros(system.natom)
        epsilons = np.zeros(system.natom)
        onlypaulis = np.zeros(system.natom, np.int32)
        for i in range(system.natom):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) > 2:
                raise TypeError('Superposition should not be allowed for non-covalent terms.')
            elif len(par_list) == 1:
                sigmas[i], epsilons[i], onlypaulis[i] = par_list[0]

        for i in range(len(onlypaulis)):
            assert(onlypaulis[0] == 0)
        # Prepare the global parameters
        #scalings = yaff.Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        #part_pair = ff_args.get_part_pair(yaff.PairPotMM3)
        #if part_pair is not None:
        #    raise RuntimeError('Internal inconsistency: the MM3 part should not be present yet.')

        #pair_pot = yaff.PairPotMM3(sigmas, epsilons, onlypaulis, ff_args.rcut, ff_args.tr)
        #nlist = ff_args.get_nlist(system)
        #part_pair = yaff.ForcePartPair(system, nlist, scalings, pair_pot)
        #ff_args.parts.append(part_pair)

        ####################################################################
        # OPENMM DISPERSION
        ####################################################################

        # to avoid diverging energies, the MM3 potential is cutoff for distances
        # below 0.6 angstrom, and replaced by a linear potential such that
        # energy and forces are continuous. This is important when restarting
        # simulations which involve the MC barostat

        #mm3 = 'epsilon * (1.84 * 100000.0 * exp(-12.0 * r / sigma) - 2.25 * (sigma / r)^6)'
        #r_switch = 0.08 # in nanometer
        #deriv = ('(epsilon * ' # evaluate derivative at r_switch
        #    '(1.84 * 100000.0 * exp(-12.0 * {} / sigma) * (-12.0) / sigma '
        #    '- 2.25 * (sigma)^6 * (-6) * (1 / {})^7))'.format(r_switch, r_switch))
        #value = ('(epsilon * (1.84 * 100000.0 * exp(-12.0 * {} / sigma)'
        #        ' - 2.25 * (sigma / {})^6))'.format(r_switch, r_switch))
        #linear = '({}) + ({}) * (r - ({}))'.format(value, deriv, r_switch)
        #energy = '({}) * step(r - ({})) + ({}) * step(({}) - r); '.format(
        #        mm3,
        #        r_switch,
        #        linear,
        #        r_switch,
        #        )
        logger.critical('Do not use the default MM3 interaction because '
                'it diverges to minus infinity at very short distances. This '
                'may be problematic when using any of the Monte Carlo '
                'barostats. To resolve this, replace the prefix MM3 with '
                'MM3CAP')
        energy = 'epsilon * (1.84 * 100000.0 * exp(-12.0 * r / sigma) - 2.25 * (sigma / r)^6); '
        energy += 'epsilon=sqrt(EPSILON1 * EPSILON2); sigma=SIGMA1 + SIGMA2;'
        periodic = not (system.cell.nvec == 0)
        scale_index = 0
        for key, value in scale_table.items():
            assert(value == 0.0 or value == 1.0)
            if value == 0.0:
                scale_index += 1

        mmgen = MM3ForceGenerator(
                periodic,
                energy,
                ['SIGMA', 'EPSILON'],
                [],
                )

        mmgen.add_particles(system, sigmas, epsilons)
        mmgen.set_rcut(ff_args.rcut)
        mmgen.set_truncation(ff_args.tr)
        mmgen.set_tailcorrections(ff_args.tailcorrections)
        mmgen.apply_exclusions(
                system.natom,
                scale_index,
                [system.neighs1, system.neighs2, system.neighs3],
                )
        return [mmgen.force]


class MM3CAPGenerator(yaff.NonbondedGenerator):
    prefix = 'MM3CAP'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float), ('ONLYPAULI', int)]

    def __call__(self, system, parsec, ff_args, dispersion_scale_index=None,
            **kwargs):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        forces = self.apply(par_table, scale_table, system, ff_args,
                dispersion_scale_index)
        return forces

    def apply(self, par_table, scale_table, system, ff_args,
            dispersion_scale_index):
        # Prepare the atomic parameters
        sigmas = np.zeros(system.natom)
        epsilons = np.zeros(system.natom)
        onlypaulis = np.zeros(system.natom, np.int32)
        for i in range(system.natom):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) > 2:
                raise TypeError('Superposition should not be allowed for non-covalent terms.')
            elif len(par_list) == 1:
                sigmas[i], epsilons[i], onlypaulis[i] = par_list[0]

        for i in range(len(onlypaulis)):
            assert(onlypaulis[0] == 0)

        ####################################################################
        # OPENMM DISPERSION
        ####################################################################

        mm3 = 'epsilon * (1.84 * 100000.0 * exp(-12.0 * r / sigma) - 2.25 * (sigma / r)^6)'
        r_switch = '(0.355114 * sigma)'
        linear = 'epsilon * (5799.303156-12182.86986*r/sigma)'
        energy = '({}) * step(r - ({})) + ({}) * step(({}) - r); '.format(
                mm3,
                r_switch,
                linear,
                r_switch,
                )
        energy += 'epsilon=sqrt(EPSILON1 * EPSILON2); sigma=SIGMA1 + SIGMA2;'
        periodic = not (system.cell.nvec == 0)
        scale_index = 0
        for key, value in scale_table.items():
            assert(value == 0.0 or value == 1.0)
            if value == 0.0:
                scale_index += 1

        mmgen = MM3ForceGenerator(
                periodic,
                energy,
                ['SIGMA', 'EPSILON'],
                [],
                )

        mmgen.add_particles(system, sigmas, epsilons)
        mmgen.set_rcut(ff_args.rcut)
        mmgen.set_truncation(ff_args.tr)
        mmgen.set_tailcorrections(ff_args.tailcorrections)
        mmgen.apply_exclusions(
                system.natom,
                scale_index,
                [system.neighs1, system.neighs2, system.neighs3],
                )
        return [mmgen.force]


class LJGenerator(yaff.NonbondedGenerator):
    prefix = 'LJ'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float)]

    def __call__(self, system, parsec, ff_args, dispersion_scale_index=None,
            **kwargs):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        forces = self.apply(par_table, scale_table, system, ff_args,
                dispersion_scale_index)
        return forces

    def apply(self, par_table, scale_table, system, ff_args,
            dispersion_scale_index):
        # Prepare the atomic parameters
        sigmas = np.zeros(system.natom)
        epsilons = np.zeros(system.natom)
        for i in range(system.natom):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) > 2:
                raise TypeError('Superposition should not be allowed for non-covalent terms.')
            elif len(par_list) == 1:
                sigmas[i], epsilons[i] = par_list[0]

        # Prepare the global parameters
        #scalings = yaff.Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        ## Get the part. It should not exist yet.
        #part_pair = ff_args.get_part_pair(yaff.PairPotLJ)
        #if part_pair is not None:
        #    raise RuntimeError('Internal inconsistency: the LJ part should not be present yet.')

        #pair_pot = yaff.PairPotLJ(sigmas, epsilons, ff_args.rcut, ff_args.tr)
        #nlist = ff_args.get_nlist(system)
        #part_pair = yaff.ForcePartPair(system, nlist, scalings, pair_pot)
        #ff_args.parts.append(part_pair)

        energy = '4.0 * epsilon * ((sigma / r)^12 - (sigma / r)^6); '
        energy += 'epsilon=sqrt(EPSILON1 * EPSILON2); sigma=(SIGMA1 + SIGMA2) / 2;'
        periodic = not (system.cell.nvec == 0)
        scale_index = 0
        for key, value in scale_table.items():
            assert(value == 0.0 or value == 1.0)
            if value == 0.0:
                scale_index += 1

        mmgen = LJForceGenerator(
                periodic,
                energy,
                ['SIGMA', 'EPSILON'],
                [],
                )

        mmgen.add_particles(system, sigmas, epsilons)
        mmgen.set_rcut(ff_args.rcut)
        mmgen.set_truncation(ff_args.tr)
        mmgen.set_tailcorrections(ff_args.tailcorrections)
        mmgen.apply_exclusions(
                system.natom,
                scale_index,
                [system.neighs1, system.neighs2, system.neighs3],
                )
        return [mmgen.force]


class LJCrossGenerator(yaff.NonbondedGenerator):
    prefix = 'LJCROSS'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float)]

    def __call__(self, system, parsec, ff_args, dispersion_scale_index=None,
            **kwargs):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 2)
        scale_table = self.process_scales(parsec['SCALE'])
        forces = self.apply(par_table, scale_table, system, ff_args,
                dispersion_scale_index)
        return forces

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def apply(self, par_table, scale_table, system, ff_args,
            dispersion_scale_index):
        # Prepare the atomic parameters
        #nffatypes = system.ffatype_ids.max() + 1
        #sigmas = np.ones([nffatypes, nffatypes]) # SAFE DEFAULT VALUE
        #epsilons = np.zeros([nffatypes, nffatypes])
        #for i in range(system.natom):
        #    for j in range(system.natom):
        #        ffa_i, ffa_j = system.ffatype_ids[i], system.ffatype_ids[j]
        #        key = (system.get_ffatype(i), system.get_ffatype(j))
        #        par_list = par_table.get(key, [])
        #        if len(par_list) > 2:
        #            raise TypeError('Superposition should not be allowed for non-covalent terms.')
        #        elif len(par_list) == 1:
        #            sigmas[ffa_i,ffa_j], epsilons[ffa_i,ffa_j] = par_list[0]
        #        elif len(par_list) == 0:
        #            pass
        nffatypes = len(system.ffatypes)
        sigmas   = np.zeros((nffatypes, nffatypes))
        epsilons = np.zeros((nffatypes, nffatypes))
        for i in range(nffatypes):
            for j in range(nffatypes):
                key = (system.ffatypes[i], system.ffatypes[j])
                pars_list = par_table.get(
                        key,
                        [],
                        )
                #assert len(pars_list[0]) == 2
                sigmas[i, j] = pars_list[0][0]
                epsilons[i, j] = pars_list[0][1]

        # Prepare the global parameters
        #scalings = yaff.Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        ## Get the part. It should not exist yet.
        #part_pair = ff_args.get_part_pair(yaff.PairPotLJCross)
        #if part_pair is not None:
        #    raise RuntimeError('Internal inconsistency: the LJCross part should not be present yet.')

        #pair_pot = yaff.PairPotLJCross(system.ffatype_ids, epsilons, sigmas, ff_args.rcut, ff_args.tr)
        #nlist = ff_args.get_nlist(system)
        #part_pair = yaff.ForcePartPair(system, nlist, scalings, pair_pot)
        #ff_args.parts.append(part_pair)


        # create dictionary with list of atom indices per ffatype
        assert np.allclose(sigmas, sigmas.T)
        assert np.allclose(epsilons, epsilons.T)
        atoms_per_ffatype = {}
        for i in range(system.natom):
            ffa = system.ffatypes[system.ffatype_ids[i]]
            if ffa in atoms_per_ffatype.keys():
                atoms_per_ffatype[ffa].append(i)
            else:
                atoms_per_ffatype[ffa] = [i]

        count = 0
        for key, value in atoms_per_ffatype.items():
            count += len(value)
        assert count == system.natom

        # FASTEST APPROACH (ON REFERENCE PLATFORM)
        sigma = 'sigma=0.0'
        u = molmod.units.nanometer
        for i in range(nffatypes):
            for j in range(i, nffatypes):
                sigma += ' + ('
                sigma += 'delta(FFATYPE1+FFATYPE2 - {}) * '.format(i + j)
                sigma += 'delta(FFATYPE1*FFATYPE2 - {}) * '.format(i * j)
                sigma += '({}))'.format(sigmas[i, j] / u)
        sigma += '; '
        epsilon = 'epsilon=0.0'
        u = molmod.units.kjmol
        for i in range(nffatypes):
            for j in range(i, nffatypes):
                epsilon += ' + ('
                epsilon += 'delta(FFATYPE1+FFATYPE2 - {}) * '.format(i + j)
                epsilon += 'delta(FFATYPE1*FFATYPE2 - {}) * '.format(i * j)
                epsilon += '({}))'.format(epsilons[i, j] / u)
        energy = '4.0 * epsilon * ((sigma / r)^12 - (sigma / r)^6); '
        energy += sigma
        energy += epsilon
        periodic = not (system.cell.nvec == 0)
        scale_index = 0
        for key, value in scale_table.items():
            assert(value == 0.0 or value == 1.0)
            if value == 0.0:
                scale_index += 1
        mmgen = LJCrossForceGenerator(
                periodic,
                energy,
                ['FFATYPE'],
                [],
                )

        mmgen.add_particles(system)
        mmgen.set_rcut(ff_args.rcut)
        mmgen.set_truncation(ff_args.tr)
        mmgen.set_tailcorrections(ff_args.tailcorrections)
        mmgen.apply_exclusions(
                system.natom,
                scale_index,
                [system.neighs1, system.neighs2, system.neighs3],
                )
        return [mmgen.force]


class FixedChargeGenerator(yaff.NonbondedGenerator):
    prefix = 'FIXQ'
    suffixes = ['UNIT', 'SCALE', 'ATOM', 'BOND', 'DIELECTRIC']
    par_info = [('Q0', float), ('P', float), ('R', float)]

    def __call__(self, system, parsec, ff_args, delta=None,
            dispersion_scale_index=None, exclusion_policy=None, **kwargs):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        atom_table = self.process_atoms(parsec['ATOM'], conversions)
        bond_table = self.process_bonds(parsec['BOND'], conversions)
        scale_table = self.process_scales(parsec['SCALE'])
        dielectric = self.process_dielectric(parsec['DIELECTRIC'])
        forces = self.apply(
                atom_table,
                bond_table,
                scale_table,
                dielectric,
                system,
                ff_args,
                delta,
                dispersion_scale_index,
                exclusion_policy,
                )
        return forces

    def process_atoms(self, pardef, conversions):
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 3:
                pardef.complain(counter, 'should have 3 arguments')
            ffatype = words[0]
            if ffatype in result:
                pardef.complain(counter, 'has an atom type that was already encountered earlier')
            try:
                charge = float(words[1])*conversions['Q0']
                radius = float(words[2])*conversions['R']
            except ValueError:
                pardef.complain(counter, 'contains a parameter that can not be converted to a floating point number')
            result[ffatype] = charge, radius
        return result

    def process_bonds(self, pardef, conversions):
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 3:
                pardef.complain(counter, 'should have 3 arguments')
            key = tuple(words[:2])
            if key in result:
                pardef.complain(counter, 'has a combination of atom types that were already encountered earlier')
            try:
                charge_transfer = float(words[2])*conversions['P']
            except ValueError:
                pardef.complain(counter, 'contains a parameter that can not be converted to floating point numbers')
            result[key] = charge_transfer
            result[key[::-1]] = -charge_transfer
        return result

    def process_dielectric(self, pardef):
        result = None
        for counter, line in pardef:
            if result is not None:
                pardef.complain(counter, 'is redundant. The DIELECTRIC suffix may only occur once')
            words = line.split()
            if len(words) != 1:
                pardef.complain(counter, 'must have one argument')
            try:
                result = float(words[0])
            except ValueError:
                pardef.complain(counter, 'must have a floating point argument')
        return result

    def apply(self, atom_table, bond_table, scale_table, dielectric, system,
            ff_args, delta, dispersion_scale_index, exclusion_policy):
        if system.charges is None:
            system.charges = np.zeros(system.natom)
        system.charges[:] = 0.0
        system.radii = np.zeros(system.natom)

        # compute the charges
        for i in range(system.natom):
            pars = atom_table.get(system.get_ffatype(i))
            if pars is not None:
                charge, radius = pars
                system.charges[i] += charge
                system.radii[i] = radius
        for i0, i1 in system.iter_bonds():
            ffatype0 = system.get_ffatype(i0)
            ffatype1 = system.get_ffatype(i1)
            if ffatype0 == ffatype1:
                continue
            charge_transfer = bond_table.get((ffatype0, ffatype1))
            if charge_transfer is None:
                pass
            else:
                system.charges[i0] += charge_transfer
                system.charges[i1] -= charge_transfer

        # prepare other parameters
        #scalings = yaff.Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        ## Setup the electrostatic pars
        #assert(dielectric == 1.0)
        #ff_args.add_electrostatic_parts(system, scalings, dielectric)

        ##############################################################
        # OPENMM ELECTROSTATICS
        # 
        # The evaluation of the electrostatic interactions depends on
        # 
        # (i)   the charges in the system (via system.charges)
        # (ii)  radii of the charge distributions (via system.radii)
        # (iii) the desired exclusions (via the scale_index)
        # (iv)  the exclusion policy and the exclusions of the dispersions
        # 
        # The NonbondedForce supports point charge interactions, such that if
        # the charge radii are nonzero, an additional CustomNonbondedForce
        # should be added to compensate for the difference.
        # Next, exclusions should be added for both forces. Because each
        # nonbonded force in the system (Custom or Default) should have the
        # same exclusions, the scale_index from the dispersion interaction is
        # used to add exclusions to the NonbondedForce and CustomNonbondedForce
        # If electrostatic_scale_index is different from dispersion_scale_index
        # then additional compensation forces should be added.
        ##############################################################

        # make asssertions
        if system.cell.nvec != 0:
            periodic = True
            rcut = ff_args.rcut / molmod.units.nanometer * unit.nanometer
            assert system.cell.nvec == 3 # only 3D periodicity supported
            assert ff_args.reci_ei == 'ewald'
        else: # no cutoff allowed
            periodic = False
            rcut = None

        # generate force objects and get expression for interaction energy
        nonbonded, gaussian, energy_expr = FixedChargeGenerator.get_forces(
                system.charges,
                system.radii,
                periodic=periodic,
                )

        # exclusions
        electrostatic_scale_index = 0
        for key, value in scale_table.items():
            assert(value == 0.0 or value == 1.0)
            if value == 0.0:
                electrostatic_scale_index += 1
        exclusion_force = FixedChargeGenerator.add_exclusions(
                system,
                dispersion_scale_index,
                electrostatic_scale_index,
                exclusion_policy,
                energy_expr,
                nonbonded=nonbonded,
                gaussian=gaussian,
                )

        # set method of evaluation based on periodicity and add rcut
        if periodic:
            assert delta is not None
            nonbonded.setNonbondedMethod(4) # PME
            nonbonded.setEwaldErrorTolerance(delta)
            nonbonded.setExceptionsUsePeriodicBoundaryConditions(True)
            nonbonded.setCutoffDistance(rcut)
            if gaussian is not None:
                gaussian.setNonbondedMethod(2) # CutoffPeriodic
                gaussian.setCutoffDistance(rcut)
            if exclusion_force is not None:
                exclusion_force.setUsesPeriodicBoundaryConditions(True)
        else:
            if gaussian is not None:
                assert nonbonded is None # no PME needed
                gaussian.setNonbondedMethod(0)
            else:
                nonbonded.setNonbondedMethod(0) # nonperiodic, no cutoff
            if exclusion_force is not None:
                exclusion_force.setUsesPeriodicBoundaryConditions(False)

        # return force(s)
        forces = []
        if nonbonded is not None:
            forces.append(nonbonded)
        if gaussian is not None:
            forces.append(gaussian)
        if exclusion_force is not None:
            forces.append(exclusion_force)
        return forces

    @staticmethod
    def get_forces(charges, radii, periodic):
        """Generates the relevant OpenMM `Force` objects

        See the following resources for electrostatic interaction expressions
        and a concise explanation of the Ewald summation method:

                10.1021/ct5009069
                http://micro.stanford.edu/mediawiki/images/4/46/Ewald_notes.pdf

        The correction CustomNonbondedForce represents the interaction between
        (gaussian - delta)_particle1 and (gaussian - delta)_particle2, and
        contains four basic contributions:

            (i)   delta delta:  cprod / r
            (ii)  delta gauss:  (-1.0) * cprod / r * erf(A2 * r)
            (iii) gauss delta:  (-1.0) * cprod / r * erf(A1 * r)
            (iv)  gauss gauss:  cprod / r * erf(A12)

        Parameters
        ----------

        charges : array_like (atomic units)
            array of atomic charges

        radii : array_like (atomic units)
            array of charge radii

        alpha : float
            alpha parameter of the ewald summation

        periodic : bool
            whether or not the system is periodic

        Returns
        -------

        nonbonded : mm.NonbondedForce or None
            used if the system is periodic or if the charge radii are 0.

        gaussian : mm.CustomNonbondedForce or None
            used to compensate for nonzero charge radii

        energy_expr : str
            expression for the energy between two particles, as function of the
            radii and charges. This is later used to generate bonded forces
            that compensate for exclusions.

        """
        natom = len(charges)
        assert natom == len(radii)
        coulomb_const = 1.0 / molmod.units.kjmol / molmod.units.nanometer
        cprod_expr = 'cprod=charge1*charge2*' + str(coulomb_const) + '; '
        if np.any(radii > 0.0):
            assert np.all(radii > 0.0)
            energy_expr = '( cprod / r * erf(A12 * r) ); '
            energy_expr += "A12=1/radius1*1/radius2/sqrt(1/radius1^2 + 1/radius2^2); "
            energy_expr += cprod_expr
            if periodic: # need PME --> NonbondedForce
                nonbonded = mm.NonbondedForce() # is short-range!
                gaussian_expr = '( (-1.0) * cprod / r + ' # compensate points
                gaussian_expr += ' cprod / r * erf(A12 * r) ); ' # add gaussian
            else: # omit NonbondedForce, and use gaussian charges directly
                nonbonded = None
                gaussian_expr = '( cprod / r * erf(A12 * r) );'
            gaussian_expr += "A12=1/radius1*1/radius2/sqrt(1/radius1^2 + 1/radius2^2); "
            gaussian_expr += cprod_expr
            gaussian = mm.CustomNonbondedForce(gaussian_expr)
            gaussian.addPerParticleParameter("charge")
            gaussian.addPerParticleParameter("radius")
        else: # all point charges
            energy_expr = '( cprod / r );'
            energy_expr += cprod_expr
            assert np.all(radii == 0.0) # all zero or all nonzero
            gaussian = None
            nonbonded = mm.NonbondedForce()

        u_charge = molmod.units.coulomb / unit.coulomb
        u_radius = molmod.units.nanometer / unit.nanometer
        if nonbonded is not None:
            for i in range(natom):
                nonbonded.addParticle(
                        charges[i] / u_charge,
                        0 * unit.nanometer, # DISPERSION NOT COMPUTED HERE
                        0 * unit.kilocalories_per_mole,
                        )
            nonbonded.setForceGroup(1) # real space contribution
            nonbonded.setReciprocalSpaceForceGroup(2)
        if gaussian is not None:
            for i in range(natom):
                gaussian.addParticle([
                    charges[i] / u_charge,
                    radii[i] / u_radius,
                    ])
            gaussian.setForceGroup(1)
        return nonbonded, gaussian, energy_expr

    @staticmethod
    def add_exclusions(system, dispersion_scale, electrostatic_scale,
            policy, energy_expr, nonbonded=None, gaussian=None):
        """Adds compensating exclusion forces to nonbonded and/or gaussian

        Depending on the difference between dispersion_scale and
        electrostatic_scale, exclusions are either added or removed from the
        system using CustomBondForce instances.

        Parameters
        ----------

        system : yaff.System
            system instance that contains bond information

        dispersion_scale : int
            determines exclusions of dispersion interactions. These are
            a fortiori included in every (Custom)NonbondedForce. Allowed values
            are 0 (no exclusions), 1, 2, 3 (1-4 exclusions).

        electrostatic_scale : int
            determines exclusions of electrostatic interactions.

        policy : str
            policy of exclusion interactions. If the dispersion and
            electrostatic exclusions are exactly the same, then it is possible
            to implement them using the traditional method, i.e. by calling
            addException or addExclusion. Otherwise, is is asserted that the
            policy is set to manual, in which case exclusions are handled
            using bonded compensation forces for the electrostatics (as that
            is the most stable of the two).

        energy_expr : str
            electrostatic interaction energy expression between two particles
            (i.e. either point-point or gaussian-gaussian).

        nonbonded : mm.NonbondedForce or None
            contains point charge interactions

        gaussian : mm.CustomNonbondedForce or None
            contains gaussian charge interactions

        """
        # add exclusions according to dispersion_scale
        # (this is mandatory on some platforms and is therefore always done)
        exclusions = []
        for i in range(system.natom):
            if dispersion_scale > 0:
                for j in system.neighs1[i]:
                    if i < j:
                        exclusions.append((i, int(j)))
            if dispersion_scale > 1:
                for j in system.neighs2[i]:
                    if i < j:
                        exclusions.append((i, int(j)))
            if dispersion_scale > 2:
                for j in system.neighs3[i]:
                    if i < j:
                        exclusions.append((i, int(j)))
        for exclusion in exclusions:
            if nonbonded is not None:
                nonbonded.addException(
                        *exclusion,
                        0 * unit.coulomb,
                        0 * unit.nanometer,
                        0 * unit.kilocalories_per_mole,
                        )
            if gaussian is not None:
                gaussian.addExclusion(
                        *exclusion,
                        )

        # add or remove interactions based on difference between two scales
        dscale = dispersion_scale - electrostatic_scale
        # dscale positive: too many exclusions were added
        # dscale negative: not enough exclusions were added
        exclusion_expr = '({}) * '.format(np.sign(dscale)) + energy_expr
        if dscale != 0:
            bonded = mm.CustomBondForce(exclusion_expr)
            bonded.addPerBondParameter('charge1')
            bonded.addPerBondParameter('charge2')
            bonded.addPerBondParameter('radius1')
            bonded.addPerBondParameter('radius2')
        else:
            bonded = None

        if bonded is not None:
            iterators = []
            iterators_to_add = abs(dscale)
            if dscale < 0:
                index = dispersion_scale + 1 # start at next
            else:
                index = dispersion_scale
            while iterators_to_add:
                if index == 3:
                    iterators.append(system.neighs3)
                    iterators_to_add -= 1
                elif index == 2:
                    iterators.append(system.neighs2)
                    iterators_to_add -= 1
                elif index == 1:
                    iterators.append(system.neighs1)
                    iterators_to_add -= 1
                elif index == 0:
                    pass
                else:
                    raise ValueError
                index -= int(np.sign(dscale))
            u_charge = molmod.units.coulomb / unit.coulomb
            u_radius = molmod.units.nanometer / unit.nanometer
            for iterator in iterators:
                for i in range(system.natom):
                    for j in iterator[i]:
                        if i < j:
                            parameters = [
                                    system.charges[i] / u_charge,
                                    system.charges[j] / u_charge,
                                    system.radii[i] / u_radius,
                                    system.radii[j] / u_radius,
                                    ]
                            bonded.addBond(i, int(j), parameters)
        return bonded


def apply_generators_to_system(yaff_seed, system_mm, **kwargs):
    """Adds forces to an OpenMM system object based on a YAFF seed

    Parameters
    ----------

    yaff_seed : openyaff.YaffSeed
        yaff seed for which to generate and add forces

    system_mm : mm.System
        OpenMM System object. It is assumed that this object does not contain
        any forces when this function is called. Each generator will add one or
        more forces to this object.

    """
    system = yaff_seed.system
    parameters = yaff_seed.parameters
    ff_args = yaff_seed.ff_args
    generators = {}
    for x in globals().values():
        if isinstance(x, type) and issubclass(x, yaff.Generator) and x.prefix is not None:
            generators[x.prefix] = x()

    # Go through all the sections of the parameter file and apply the
    # corresponding generator.
    total_forces = []
    for prefix, section in parameters.sections.items():
        logger.debug('applying prefix {}'.format(prefix))
        generator = generators.get(prefix)
        if generator is None:
            raise NotImplementedError('No implementation for prefix {}'.format(
                prefix))
        else:
            forces = generator(system, section, ff_args, **kwargs)
            if isinstance(forces, list):
                total_forces += forces
            else:
                total_forces += [forces]
    if total_forces is not None:
        # double check periodicity for each force; if yaff system is periodic
        # then all forces in openmm should use PBCs.
        periodic = (yaff_seed.system.cell.nvec == 3)
        for force in total_forces:
            assert force.usesPeriodicBoundaryConditions() == periodic
            system_mm.addForce(force)
    return system_mm


def apply_generators_to_xml(yaff_seed, forcefield, **kwargs):
    """Constructs an OpenMM force field in .xml format

    Parameters
    ----------

    yaff_seed : openyaff.YaffSeed
        yaff seed for which to generate and add forces

    forcefield : xml.etree.ElementTree.Element
        Object used to represent the contents of the .xml file. The root
        element 'ForceField' contains an element 'AtomTypes', an element
        'Residues', and a variety of force elements.

    """
    system     = yaff_seed.system
    #ff_args    = yaff_seed.ff_args
    ff_args    = None
    parameters = yaff_seed.parameters

    generators = {}
    for x in globals().values():
        if isinstance(x, type) and issubclass(x, yaff.Generator) and x.prefix is not None:
            generators[x.prefix] = x()

    forces = []
    for prefix, section in parameters.sections.items():
        logger.debug('applying prefix {}'.format(prefix))
        generator = generators.get(prefix)
        if generator is None:
            raise NotImplementedError('No implementation for prefix {}'.format(
                prefix))
        if prefix not in ['LJ', 'FIXQ']: # treated separately
            forces += generator.parse_xml(system, section, ff_args, **kwargs)

    # add nonbonded
    prefixes_nb = [p for p in parameters.sections.keys() if p in ['LJ', 'FIXQ']]
    particle_pars = {} # ffatype as key; (q, sigma, epsilon) as value
    for atom_type in system.ffatypes:
        particle_pars[atom_type] = [0.0, 4.0, 0.0] # safe initialization
    for p in prefixes_nb:
        section = parameters.sections[p]
        if p == 'LJ': # parse LJ data using LJ generator
            generator = yaff.pes.generator.LJGenerator()
            generator.check_suffixes(section)
            conversions = generator.process_units(section['UNIT'])
            par_table = generator.process_pars(section['PARS'], conversions, 1)
            scale_table = generator.process_scales(section['SCALE'])
            for atom_type in system.ffatypes:
                sigma, epsilon = par_table[(atom_type,)][0]
                particle_pars[atom_type][1] = sigma
                particle_pars[atom_type][2] = epsilon
        elif p == 'FIXQ':
            generator = yaff.pes.generator.FixedChargeGenerator()
            generator.check_suffixes(section)
            conversions = generator.process_units(section['UNIT'])
            atom_table = generator.process_atoms(section['ATOM'], conversions)
            bond_table = generator.process_bonds(section['BOND'], conversions)
            scale_table = generator.process_scales(section['SCALE'])

            # first get regular charges
            charges = np.zeros(len(system.ffatypes))
            for i, atom_type in enumerate(system.ffatypes):
                charge, radius = atom_table[atom_type]
                assert radius == 0.0
                charges[i] = charge

            # apply bond charge increments
            for bond_type, transfer in bond_table.items():
                if transfer is None:
                    continue
                index0 = system.ffatypes.index(bond_type[0])
                index1 = system.ffatypes.index(bond_type[1])
                charges[index0] += transfer
                charges[index1] -= transfer

            for i, atom_type in enumerate(system.ffatypes):
                particle_pars[atom_type][0] = charges[i]
        else:
            raise ValueError('Unexpected nonbonded prefix {}'.format(p))

    attrib = {
            'coulomb14scale': str(kwargs['scalings']['FIXQ'][3]),
            'lj14scale': str(kwargs['scalings']['LJ'][3]),
            'cutoff': str(kwargs['nonbondedCutoff'] / 10),
            'useSwitchingFunction': str(kwargs['useSwitchingFunction']),
            'useLongRangeCorrection': str(kwargs['useLongRangeCorrection']),
            'nonbondedMethod': str(kwargs['nonbondedMethod']),
            'switchingDistance': str(kwargs['switchingDistance'] / 10),
            }
    nbforce = ET.Element('NonbondedForce', attrib=attrib)
    for atom_type in system.ffatypes:
        attrib = {
                'type': atom_type,
                'charge': str(particle_pars[atom_type][0]),
                'sigma': str(particle_pars[atom_type][1] / molmod.units.nanometer),
                'epsilon': str(particle_pars[atom_type][2] / molmod.units.kjmol),
                }
        nbforce.append(ET.Element('Atom', attrib=attrib))
    forces.append(nbforce)
    return forces

# determine supported covalent, dispersion and electrostatic prefixes
COVALENT_PREFIXES      = []
DISPERSION_PREFIXES    = []
ELECTROSTATIC_PREFIXES = ['FIXQ'] # FIXQ is only electrostatic prefix
for x in list(globals().values()):
    if isinstance(x, type) and issubclass(x, yaff.Generator) and x.prefix is not None:
        if (issubclass(x, ValenceMirroredGenerator) or
                issubclass(x, ValenceCrossMirroredGenerator)):
            COVALENT_PREFIXES.append(x.prefix)
        elif x.prefix in ELECTROSTATIC_PREFIXES:
            pass # ELECTROSTATIC_PREFIXES already determined
        else:
            DISPERSION_PREFIXES.append(x.prefix)

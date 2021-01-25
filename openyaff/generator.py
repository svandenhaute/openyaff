import yaff
import numpy as np
import molmod
import simtk.unit as unit
import simtk.openmm as mm


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

    def __call__(self, system, parsec, ff_args):
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
                vterm = self.get_vterm(pars, indexes)
                assert vterm is not None # neccessary?
                part_valence.add_term(vterm)
                self.add_term_to_force(force, pars, indexes)

    def add_term_to_force(self, force, pars, indexes):
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


class BondHarmGenerator(BondGenerator):
    prefix = 'BONDHARM'
    VClass = yaff.Harmonic

    def get_force(self, periodic):
        force = mm.HarmonicBondForce()
        force.setUsesPeriodicBoundaryConditions(periodic)
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


class BendGenerator(ValenceMirroredGenerator):
    nffatype = 3
    ICClass = None
    VClass = yaff.Harmonic

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_angles()


class BendAngleHarmGenerator(BendGenerator):
    par_info = [('K', float), ('THETA0', float)]
    prefix = 'BENDAHARM'
    ICClass = yaff.BendAngle

    def get_force(self, periodic):
        force = mm.HarmonicAngleForce()
        force.setUsesPeriodicBoundaryConditions(periodic)
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
        return force

    @staticmethod
    def _get_dist():
        dist = 'distance(p2, p4) * cos(2 * atan(1.0000) - angle(p4, p2, p3)) * sin(dihedral(p1, p2, p3, p4))'
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


#class SquareOopDistGenerator(ValenceMirroredGenerator):
#    nffatype = 4
#    par_info = [('K', float), ('D0', float)]
#    prefix = 'SQOOPDIST'
#    ICClass = yaff.SqOopDist
#    VClass = yaff.Harmonic
#    allow_superposition = False
#
#    def iter_equiv_keys_and_pars(self, key, pars):
#        yield key, pars
#        yield (key[2], key[0], key[1], key[3]), pars
#        yield (key[1], key[2], key[0], key[3]), pars
#        yield (key[2], key[1], key[0], key[3]), pars
#        yield (key[1], key[0], key[2], key[3]), pars
#        yield (key[0], key[2], key[1], key[3]), pars
#
#    def iter_indexes(self, system):
#        #Loop over all atoms; if an atom has 3 neighbors,
#        #it is candidate for an OopDist term
#        for atom in system.neighs1.keys():
#            neighbours = list(system.neighs1[atom])
#            if len(neighbours)==3:
#                yield neighbours[0],neighbours[1],neighbours[2],atom
#
#    def get_force(self):
#        dist = OopDistGenerator._get_dist()
#        energy = '0.5 * K * (({})^2 - D0)^2'.format(dist)
#        force = mm.CustomCompoundBondForce(4, energy)
#        force.addPerBondParameter('K')
#        force.addPerBondParameter('D0')
#        force.setUsesPeriodicBoundaryConditions(True)
#        return force
#
#    def add_term_to_force(self, force, pars, indexes):
#        conversion = {
#                'K': molmod.units.kjmol / molmod.units.nanometer ** 4,
#                'D0': molmod.units.nanometer ** 2,
#                }
#        conversion_mm = {
#                'K': unit.kilojoule_per_mole / unit.nanometer ** 4,
#                'D0': unit.nanometer ** 2,
#                }
#        K = pars[0] / conversion['K'] * conversion_mm['K']
#        D0 = pars[1] / conversion['D0'] * conversion_mm['D0']
#        force.addBond(
#                [
#                    int(indexes[0]),
#                    int(indexes[1]),
#                    int(indexes[2]),
#                    int(indexes[3])],
#                [K, D0],
#                )


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
        #force = mm.PeriodicTorsionForce()
        #force.setUsesPeriodicBoundaryConditions(True)
        energy = 'A/2*(1 - cos(M*(theta - PHI0)))'
        force = mm.CustomTorsionForce(energy)
        force.addPerTorsionParameter('M')
        force.addPerTorsionParameter('A')
        force.addPerTorsionParameter('PHI0')
        force.setUsesPeriodicBoundaryConditions(periodic)
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

    def __call__(self, system, parsec, ff_args):
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
                    part_valence.add_term(VClass_ij(*args_ij))
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


def apply_generators_mm(yaff_seed, system_mm):
    """Adds forces to an OpenMM system object based on a YAFF seed

    Parameters
    ----------

    yaff_seed : openyaff.YaffSeed
        yaff seed for which to generate and add forces

    system_mm : mm.System
        OpenMM System object. It is assumed that this object does not contain
        any forces when this function is called.

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
        generator = generators.get(prefix)
        if generator is None:
            raise NotImplementedError('No implementation for prefix {}'.format(
                prefix))
        else:
            forces = generator(system, section, ff_args)
            if isinstance(forces, list):
                total_forces += forces
            else:
                total_forces += [forces]
    if total_forces is not None:
        for force in total_forces:
            system_mm.addForce(force)
    return system_mm


AVAILABLE_PREFIXES = []
for x in list(globals().values()):
    if isinstance(x, type) and issubclass(x, yaff.Generator) and x.prefix is not None:
        AVAILABLE_PREFIXES.append(x.prefix)

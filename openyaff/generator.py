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


class MM3Generator(yaff.NonbondedGenerator):
    prefix = 'MM3'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float), ('ONLYPAULI', int)]

    def __call__(self, system, parsec, ff_args, **kwargs):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        forces = self.apply(par_table, scale_table, system, ff_args)
        return forces

    def apply(self, par_table, scale_table, system, ff_args):
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
        scalings = yaff.Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(yaff.PairPotMM3)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the MM3 part should not be present yet.')

        pair_pot = yaff.PairPotMM3(sigmas, epsilons, onlypaulis, ff_args.rcut, ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = yaff.ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)

        energy = 'epsilon * (1.84 * 100000 * exp(-12 * r / sigma) - 2.25 * (sigma / r)^6)'
        step = ' * step({} - r);'.format(ff_args.rcut / molmod.units.nanometer)
        definitions = 'epsilon=sqrt(EPSILON1 * EPSILON2); sigma=SIGMA1 + SIGMA2;'
        force = mm.CustomNonbondedForce(energy + '; ' + definitions)
        force.addPerParticleParameter('SIGMA')
        force.addPerParticleParameter('EPSILON')
        for i in range(system.pos.shape[0]):
            parameters = [
                    sigmas[i] / molmod.nanometer * unit.nanometer,
                    epsilons[i] / molmod.units.kjmol * unit.kilojoule_per_mole,
                    ]
            force.addParticle(parameters)
        force.setCutoffDistance(ff_args.rcut / molmod.units.nanometer * unit.nanometer)
        force.setNonbondedMethod(2)

        # TAIL CORRECTIONS
        if ff_args.tailcorrections:
            force.setUseLongRangeCorrection(True)

        # SET SWITCHING IF NEEDED
        if ff_args.tr is not None:
            width = ff_args.tr.width
            force.setSwitchingDistance((ff_args.rcut - width) / molmod.units.nanometer * unit.nanometer)
            force.setUseSwitchingFunction(True)

        # EXCLUSIONS
        scale_index = 0
        for key, value in scale_table.items():
            assert(value == 0.0 or value == 1.0)
            if value == 0.0:
                scale_index += 1
        for i in range(system.natom):
            if scale_index > 0:
                for j in system.neighs1[i]:
                    if i < j:
                        force.addExclusion(i, int(j))
            if scale_index > 1:
                for j in system.neighs2[i]:
                    if i < j:
                        force.addExclusion(i, int(j))
            if scale_index > 2:
                for j in system.neighs3[i]:
                    if i < j:
                        force.addExclusion(i, int(j))
        return [force]

        # COMPENSATE FOR EXCLUSIONS
        #scale_index = 0
        #for key, value in scale_table.items():
        #    assert(value == 0.0 or value == 1.0)
        #    if value == 0.0:
        #        scale_index += 1

        #exclusion_force = self.get_exclusion_force(energy + step + '; ' + definitions)
        #for i in range(system.natom):
        #    if scale_index > 0:
        #        for j in system.neighs1[i]:
        #            if i < j:
        #                self.add_exclusion(sigmas, epsilons, i, j, exclusion_force)
        #    if scale_index > 1:
        #        for j in system.neighs2[i]:
        #            if i < j:
        #                self.add_exclusion(sigmas, epsilons, i, j, exclusion_force)
        #    if scale_index > 2:
        #        raise NotImplementedError
        #return [force, exclusion_force]

    @staticmethod
    def get_exclusion_force(energy, periodic):
        """Returns force object to account for exclusions"""
        force = mm.CustomBondForce('-1.0 * ' + energy)
        force.addPerBondParameter('SIGMA1')
        force.addPerBondParameter('EPSILON1')
        force.addPerBondParameter('SIGMA2')
        force.addPerBondParameter('EPSILON2')
        force.setUsesPeriodicBoundaryConditions(periodic)
        return force

    @staticmethod
    def add_exclusion(sigmas, epsilons, i, j, force):
        """Adds a bond between i and j"""
        SIGMA1 = sigmas[i] / (molmod.units.angstrom * 10) * unit.nanometer
        EPSILON1 = epsilons[i] / molmod.units.kcalmol * unit.kilocalories_per_mole
        SIGMA2 = sigmas[j] / (molmod.units.angstrom * 10) * unit.nanometer
        EPSILON2 = epsilons[j] / molmod.units.kcalmol * unit.kilocalories_per_mole
        parameters = [
                SIGMA1,
                EPSILON1,
                SIGMA2,
                EPSILON2,
                ]
        force.addBond(int(i), int(j), parameters)


class LJGenerator(yaff.NonbondedGenerator):
    prefix = 'LJ'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float)]

    def __call__(self, system, parsec, ff_args, **kwargs):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        forces = self.apply(par_table, scale_table, system, ff_args)
        return forces

    def apply(self, par_table, scale_table, system, ff_args):
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
        scalings = yaff.Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(yaff.PairPotLJ)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the LJ part should not be present yet.')

        pair_pot = yaff.PairPotLJ(sigmas, epsilons, ff_args.rcut, ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = yaff.ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)

        energy = '4.0 * epsilon * ((sigma / r)^12 - (sigma / r)^6)'
        step = ' * step({} - r);'.format(ff_args.rcut / molmod.units.nanometer)
        definitions = 'epsilon=sqrt(EPSILON1 * EPSILON2); sigma=(SIGMA1 + SIGMA2) / 2;'
        force = mm.CustomNonbondedForce(energy + '; ' + definitions)
        force.addPerParticleParameter('SIGMA')
        force.addPerParticleParameter('EPSILON')
        for i in range(system.pos.shape[0]):
            parameters = [
                    sigmas[i] / molmod.nanometer * unit.nanometer,
                    epsilons[i] / molmod.units.kjmol * unit.kilojoule_per_mole,
                    ]
            force.addParticle(parameters)
        force.setCutoffDistance(ff_args.rcut / molmod.units.nanometer * unit.nanometer)
        if system.cell.nvec == 3: # if system is periodic
            force.setNonbondedMethod(2)
        else: # system is not periodic, use CutOffNonPeriodic
            force.setNonbondedMethod(1)

        # TAIL CORRECTIONS; only make sense if system is periodic
        if ff_args.tailcorrections and system.cell.nvec == 3:
            force.setUseLongRangeCorrection(True)

        # SET SWITCHING IF NEEDED
        if ff_args.tr is not None:
            width = ff_args.tr.width
            force.setSwitchingDistance((ff_args.rcut - width) / molmod.units.nanometer * unit.nanometer)
            force.setUseSwitchingFunction(True)

        # EXCLUSIONS
        scale_index = 0
        for key, value in scale_table.items():
            assert(value == 0.0 or value == 1.0)
            if value == 0.0:
                scale_index += 1
        for i in range(system.natom):
            if scale_index > 0:
                for j in system.neighs1[i]:
                    if i < j:
                        force.addExclusion(i, int(j))
            if scale_index > 1:
                for j in system.neighs2[i]:
                    if i < j:
                        force.addExclusion(i, int(j))
            if scale_index > 2:
                for j in system.neighs3[i]:
                    if i < j:
                        force.addExclusion(i, int(j))
        return [force]


class FixedChargeGenerator(yaff.NonbondedGenerator):
    prefix = 'FIXQ'
    suffixes = ['UNIT', 'SCALE', 'ATOM', 'BOND', 'DIELECTRIC']
    par_info = [('Q0', float), ('P', float), ('R', float)]

    def __call__(self, system, parsec, ff_args, delta=None, **kwargs):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        atom_table = self.process_atoms(parsec['ATOM'], conversions)
        bond_table = self.process_bonds(parsec['BOND'], conversions)
        scale_table = self.process_scales(parsec['SCALE'])
        dielectric = self.process_dielectric(parsec['DIELECTRIC'])
        forces = self.apply(atom_table, bond_table, scale_table, dielectric, system, ff_args, delta)
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

    def apply(self, atom_table, bond_table, scale_table, dielectric, system, ff_args, delta):
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
        scalings = yaff.Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Setup the electrostatic pars
        assert(dielectric == 1.0)
        ff_args.add_electrostatic_parts(system, scalings, dielectric)

        force = mm.NonbondedForce()
        for i in range(system.pos.shape[0]):
            force.addParticle(
                    system.charges[i] / molmod.units.coulomb * unit.coulomb,
                    0 * unit.nanometer, # DISPERSION NOT COMPUTED IN THIS FORCE
                    0 * unit.kilocalories_per_mole, # DISPERSION NOT COMPUTED
                    )
        rcut = ff_args.rcut / (molmod.units.nanometer) * unit.nanometer
        force.setCutoffDistance(rcut)
        if system.cell.nvec == 3:
            key = 'pme_error_thres'
            assert delta is not None # specifies delta
            force.setNonbondedMethod(4) # PME
            force.setEwaldErrorTolerance(delta)
        else:
            force.setNonbondedMethod(0) # nonperiodic cutoff not allowed

        # COMPENSATE FOR GAUSSIAN CHARGES
        if np.any(system.radii[:] !=0.0):
            alpha = ff_args.alpha_scale / rcut
            gaussian_force = self.get_force(
                    alpha,
                    reci_ei=ff_args.reci_ei,
                    )
            for i in range(system.pos.shape[0]):
                parameters = [
                        system.charges[i] / molmod.units.coulomb * unit.coulomb,
                        system.radii[i] / molmod.units.nanometer * unit.nanometer,
                        ]
                gaussian_force.addParticle(parameters)
            gaussian_force.setCutoffDistance(rcut)
            gaussian_force.setNonbondedMethod(2)
        else:
            gaussian_force = None


        # EXCLUSIONS
        scale_index = 0
        for key, value in scale_table.items():
            assert(value == 0.0 or value == 1.0)
            if value == 0.0:
                scale_index += 1
        for i in range(system.natom):
            if scale_index > 0:
                for j in system.neighs1[i]:
                    if i < j:
                        if gaussian_force is not None:
                            gaussian_force.addExclusion(i, int(j))
                        force.addException(i, int(j),
                                0 * unit.coulomb, 0 * unit.nanometer, 0 * unit.kilocalories_per_mole)
            if scale_index > 1:
                for j in system.neighs2[i]:
                    if i < j:
                        if gaussian_force is not None:
                            gaussian_force.addExclusion(i, int(j))
                        force.addException(i, int(j),
                                0 * unit.coulomb, 0 * unit.nanometer, 0 * unit.kilocalories_per_mole)
            if scale_index > 2:
                for j in system.neighs3[i]:
                    if i < j:
                        if gaussian_force is not None:
                            gaussian_force.addExclusion(i, int(j))
                        force.addException(i, int(j),
                                0 * unit.coulomb, 0 * unit.nanometer, 0 * unit.kilocalories_per_mole)

        if np.any(system.radii[:] !=0.0):
            if ff_args.reci_ei == 'ignore':
                forces = [gaussian_force]
            elif ff_args.reci_ei == 'ewald':
                forces = [force, gaussian_force]
            else:
                raise NotImplementedError
        else:
            if ff_args.reci_ei == 'ignore':
                raise NotImplementedError
            elif ff_args.reci_ei == 'ewald':
                forces = [force]
            else:
                raise NotImplementedError
        return forces

    @staticmethod
    def get_force(ALPHA, reci_ei='ewald'):
        """Creates a short-ranged electrostatic force object that compensates
        for the gaussian charge distribution.

        Arguments
        ---------
            ALPHA (float):
                the 'alpha' parameter of the gaussians used for the sum in reciprocal space.
            reci_ei (string):
                specifies whether the reciprocal sum is included. If it is not,
                than no compensation is required.
        """
        #coulomb_const = 8.9875517887 * 1e9 # in units of J * m / C2
        #coulomb_const *= 1.0e9 # in units of J * nm / C2
        #coulomb_const *= molmod.constants.avogadro / 1000 # in units of kJmol * nm / C2
        #coulomb_const /= (1 / 1.602176634e-19) ** 2
        # coulomb_const : 1.38935456e2
        coulomb_const = 1.0 / molmod.units.kjmol / molmod.units.nanometer
        E_S_test = "cprod / r * erfc(ALPHA * r); "
        # SUBTRACT SHORT-RANGE CONTRIBUTION
        E_S = "- cprod / r * erfc(ALPHA * r) "
        # ADD SHORT-RANGE GAUSS CONTRIBUTION
        E_Sg = "cprod / r * (erf(A12 * r) - erf(ALPHA * r)); "
        definitions = "cprod=charge1*charge2*" + str(coulomb_const) + "; "
        definitions += "A12=1/radius1*1/radius2/sqrt(1/radius1^2 + 1/radius2^2); "
        definitions += "A1 = 1/radius1*ALPHA/sqrt(1/radius1^2 + ALPHA^2); "
        definitions += "A2 = 1/radius2*ALPHA/sqrt(1/radius2^2 + ALPHA^2); "
        if reci_ei == 'ewald':
            energy = E_S + " + " + E_Sg + definitions
        elif reci_ei == 'ignore':
            energy = E_Sg + definitions
        else:
            raise NotImplementedError
        #energy = E_Sg + definitions
        #energy = E_S_test
        #energy += "cprod=charge1*charge2*" + str(coulomb_const) + "; "
        force = mm.CustomNonbondedForce(energy)
        force.addPerParticleParameter("charge")
        force.addPerParticleParameter("radius")
        force.addGlobalParameter("ALPHA", ALPHA)
        return force


def apply_generators_mm(yaff_seed, system_mm, **kwargs):
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


AVAILABLE_PREFIXES = []
for x in list(globals().values()):
    if isinstance(x, type) and issubclass(x, yaff.Generator) and x.prefix is not None:
        AVAILABLE_PREFIXES.append(x.prefix)

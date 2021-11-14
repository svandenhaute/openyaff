import molmod
import logging
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm

from openyaff.utils import yaff_generate, estimate_cell_derivative, \
        transform_symmetric


logger = logging.getLogger(__name__) # logging per module


class ForceFieldWrapper:
    """Provides a standardized interface for energy and force computations

    Subclasses wrap around yaff.ForceField and mm.Context objects

    """
    def __init__(self, periodic):
        self.periodic = periodic

    def evaluate(self, positions, rvecs=None, do_forces=True):
        """Computes energy, forces and stress if available

        Parameters
        ----------

        positions : array_like [angstrom]
            numpy array specifying atomic positions. Both 2D and 3D arrays are
            accepted:
                - (natoms, 3): specifies atom positions of a single state
                - (nstates, natoms, 3): specifies trajectory of states;
                  evaluations are performed over each state in the trajectory.

        rvecs : array_like [angstrom], optional
            if the system is periodic, rvecs are required to specify the
            periodic box. Its shape should agree with that of positions

        do_forces : bool, optional
            specifies whether to compute and return forces

        Returns
        -------

        energy : 1darray or float [kJ/mol]
            potential energy over trajectory (or state)

        force : array_like [kJ/(mol angstrom)], if do_forces == True
            forces over trajectory (or state)

        """
        if self.periodic:
            assert rvecs is not None
            assert len(positions.shape) == len(rvecs.shape)
        else:
            assert rvecs is None
        if len(positions.shape) == 2:
            return self._internal_evaluate(positions, rvecs, do_forces)
        else:
            nstates, natoms = positions.shape[:2]
            energy = np.zeros(nstates)
            if do_forces:
                forces = np.zeros((nstates, natoms, 3))
            for i in range(positions.shape[0]):
                if self.periodic:
                    rvecs_ = rvecs[i]
                else:
                    rvecs_ = None
                energy[i], f = self._internal_evaluate(
                        positions[i],
                        rvecs_,
                        do_forces,
                        )
                if do_forces:
                    forces[i, :] = f[:]
            if do_forces:
                return energy, forces
            else:
                return energy

    def compute_stress(self, positions, rvecs, dh=1e-6, use_symmetric=True):
        """Computes the virial stress using a finite difference scheme"""
        def energy_func(pos, cell):
            return self._internal_evaluate(pos, cell, do_forces=False)

        if not use_symmetric:
            # use triangular perturbations if cell is lower triangular
            if (rvecs[0, 1] == 0.0 and
                rvecs[0, 2] == 0.0 and
                rvecs[1, 2] == 0.0):
                use_triangular_perturbation = True
            else:
                use_triangular_perturbation = False
            dUdh = estimate_cell_derivative(
                    positions,
                    rvecs,
                    energy_func,
                    dh=dh,
                    use_triangular_perturbation=use_triangular_perturbation,
                    evaluate_using_reduced=True, # necessary for OpenMM wrapper
                    )
            stress = (rvecs.T @ dUdh) / np.linalg.det(rvecs)
        else:
            pos_tmp = positions.copy()
            rvecs_tmp = rvecs.copy()
            transform_symmetric(pos_tmp, rvecs_tmp)
            dUdh = estimate_cell_derivative(
                    pos_tmp,
                    rvecs_tmp,
                    energy_func,
                    dh=dh,
                    use_triangular_perturbation=False,
                    evaluate_using_reduced=True, # necessary for OpenMM wrapper
                    )
            stress = (rvecs_tmp.T @ dUdh) / np.linalg.det(rvecs_tmp)
            if not np.allclose(stress, stress.T, atol=1e-5, rtol=1e-4):
                logger.debug('numerical stress is not fully symmetric; '
                        'try changing the finite difference step dh')
                logger.debug('{}'.format(stress))
        return stress

    def _internal_evaluate(self, positions, rvecs, do_forces):
        raise NotImplementedError

    @classmethod
    def from_seed(cls, seed):
        """Generates the wrapper from a seed"""
        raise NotImplementedError


class YaffForceFieldWrapper(ForceFieldWrapper):
    """Wrapper for Yaff force field evaluations"""

    def __init__(self, ff):
        """Constructor

        Parameters
        ----------

        ff : yaff.ForceField
            force field used to evaluate energies

        """
        periodic = not (ff.system.cell.nvec == 0)
        if periodic:
            # does not support 1D or 2D periodicity
            assert ff.system.cell.nvec == 3
        self.ff = ff
        super().__init__(periodic)

    def _internal_evaluate(self, positions, rvecs, do_forces):
        """Wraps call to ff.compute()

        Parameters
        ----------

        positions : array_like [angstrom]
            2darray of atomic positions

        rvecs : array_like [angstrom] or None
            box vectors in case of periodic system

        do_forces : bool
            whether to compute and return forces

        Returns
        -------

        energy : float [kJ/mol]
            potential energy

        force : array_like [kJ/(mol angstrom)], if do_forces == True
            forces

        """
        self.ff.update_pos(positions * molmod.units.angstrom)
        if rvecs is not None:
            self.ff.update_rvecs(rvecs * molmod.units.angstrom)
        if do_forces:
            gpos = np.zeros(positions.shape)
        else:
            gpos = None
        energy = self.ff.compute(gpos=gpos, vtens=None)
        energy /= molmod.units.kjmol
        if do_forces:
            return energy, -gpos / molmod.units.kjmol * molmod.units.angstrom
        else:
            return energy

    @classmethod
    def from_seed(cls, seed):
        """Generates wrapper from a seed

        Parameters
        ----------

        seed : tuple of (yaff.System, yaff.Parameters, yaff.FFArgs)
            seed for the yaff force field wrapper

        """
        ff = yaff_generate(seed)
        return cls(ff)


class OpenMMForceFieldWrapper(ForceFieldWrapper):
    """Wrapper for OpenMM force evaluations"""

    def __init__(self, system_mm, platform_name):
        """Constructor

        Parameters
        ----------

        system_mm : mm.System
            OpenMM System object containing all Force objects

        platform_name : str
            OpenMM platform on which the computations should be performed.

        """
        self.periodic = system_mm.usesPeriodicBoundaryConditions()
        # integrator is necessary to create context
        integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
        platform = mm.Platform.getPlatformByName(platform_name)
        logger.debug('platform supports double precision: {}'.format(
            platform.supportsDoublePrecision()))
        if platform_name in ['CUDA', 'OpenCL']:
            platform.setPropertyDefaultValue('Precision', 'Double')
        self.context = mm.Context(system_mm, integrator, platform)


    def _internal_evaluate(self, positions, rvecs, do_forces):
        """Computes energy and forces

        Positions (and optionally box vectors) are updated in the context,
        and get state is called to obtain energy and forces

        """
        self.context.setPositions(positions * unit.angstrom)
        if self.periodic:
            self.context.setPeriodicBoxVectors(
                    rvecs[0, :] * unit.angstrom,
                    rvecs[1, :] * unit.angstrom,
                    rvecs[2, :] * unit.angstrom,
                    )
        #self.context.reinitialize(preserveState=True)
        state = self.context.getState(
                getEnergy=True,
                getForces=do_forces,
                enforcePeriodicBox=True,
                )
        energy = state.getPotentialEnergy().in_units_of(unit.kilojoule_per_mole)
        energy_ = energy.value_in_unit(energy.unit)
        if not do_forces:
            return energy_
        else:
            forces = state.getForces(asNumpy=True)
            forces_ = forces.value_in_unit(unit.kilojoule_per_mole / unit.angstrom)
            return energy_, forces_

    @classmethod
    def from_seed(cls, seed, platform_name):
        """Generates wrapper from OpenMM seed

        Parameters
        ----------

        seed : tuple of (mm.System,)
            seed for the OpenMM force field wrapper. For now, this is simply an
            OpenMM system object in which all forces are added by an
            openyaff.ExplicitConversion.apply() call.

        platform_name : str
            OpenMM platform on which the computations should be performed.

        """
        return cls(seed.system, platform_name)

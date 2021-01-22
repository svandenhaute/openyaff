import molmod
import numpy as np


class ForceFieldWrapper:
    """Provides a standardized interface for energy and force computations

    Subclasses wrap around yaff.ForceField and mm.Context objects

    """
    def __init__(self, periodic):
        pass

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
        if self.is_periodic:
            assert rvecs is not None
            assert len(positions.shape) == len(rvecs.shape)
        else:
            assert rvecs is None
        if len(positions.shape) == 2:
            return _internal_evaluate(positions, rvecs, do_forces)
        else:
            nstates, natoms = positions.shape[:2]
            energy = np.zeros(nstates)
            if do_forces:
                forces = np.zeros((nstates, natoms, 3))
            for i in range(positions.shape[0]):
                energy[i], f = self._internal_evaluate(
                        positions,
                        rvecs,
                        do_forces,
                        )
                if do_forces:
                    forces[i, :] = f[:]
            if do_forces:
                return energy, forces
            else:
                return energy

    def estimate_stress(self, positions, rvecs):
        """Estimates the virial stress using a finite difference scheme"""
        raise NotImplementedError

    def _internal_evaluate(self, positions, rvecs, do_forces):
        raise NotImplementedError


class YAFFWrapper(ForceFieldWrapper):
    """Wrapper for a yaff.ForceField object"""

    def __init__(self, ff):
        """Constructor

        Parameters
        ----------

        ff : yaff.ForceField
            force field used to evaluate energies

        """
        is_periodic = not (ff.system.cell is None)
        super().__init__(is_periodic)
        if is_periodic: # currently only supports 3D periodic systems
            assert ff.system.cell._get_nvec() == 3
        self.ff = ff

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
        self.ff.update_rvecs(rvecs * molmod.units.angstrom)
        if do_forces:
            gpos = np.zeros(positions.shape)
        else:
            gpos = None
        energy = self.ff.compute(gpos=gpos, vtens=None)
        energy /= molmod.units.kjmol
        if do_forces:
            return energy, forces / molmod.units.kjmol * molmod.units.angstrom
        else:
            return energy

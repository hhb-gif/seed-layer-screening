"""CHGNet calculator implementation."""

from typing import Any, Dict

import numpy as np
from pymatgen.core import Structure

from .base import CalculatorBase


class CHGNetCalculator(CalculatorBase):
    """CHGNet machine learning potential calculator."""

    def __init__(self, **kwargs):
        """Initialize CHGNet calculator.

        Args:
            **kwargs: Additional arguments passed to CHGNet (currently unused)
        """
        from chgnet.model import StructOptimizer

        self.relaxer = StructOptimizer()
        self._calculator = None

    def _get_ase_calculator(self):
        """Get ASE calculator from CHGNet (lazy initialization)."""
        if self._calculator is None:
            from chgnet.model.model import CHGNet
            model = CHGNet.load()
            self._calculator = model
        return self._calculator

    def relax(
        self,
        structure: Structure,
        fmax: float = 0.05,
        steps: int = 500,
        relax_cell: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Relax structure using CHGNet.

        Args:
            structure: pymatgen Structure to relax
            fmax: Force convergence criterion in eV/Å
            steps: Maximum number of relaxation steps
            relax_cell: Whether to relax cell parameters
            verbose: Whether to print relaxation progress

        Returns:
            Dictionary with final_structure, energy, and trajectory
        """
        result = self.relaxer.relax(
            structure,
            fmax=fmax,
            steps=steps,
            relax_cell=relax_cell,
            verbose=verbose,
        )

        final_structure = result["final_structure"]
        trajectory = result.get("trajectory")
        energy = trajectory.energies[-1] if trajectory else None

        return {
            "final_structure": final_structure,
            "energy": energy,
            "trajectory": trajectory,
        }

    def get_energy(self, structure: Structure) -> float:
        """Get total energy using CHGNet.

        Args:
            structure: pymatgen Structure

        Returns:
            Total energy in eV
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        atoms.calc = self._get_ase_calculator()
        return atoms.get_potential_energy()

    def get_forces(self, structure: Structure) -> np.ndarray:
        """Get forces using CHGNet.

        Args:
            structure: pymatgen Structure

        Returns:
            Forces array of shape (N, 3) in eV/Å
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        atoms.calc = self._get_ase_calculator()
        return atoms.get_forces()

"""Abstract base class for ML potential calculators."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from pymatgen.core import Structure


class CalculatorBase(ABC):
    """Abstract base class for machine learning potential calculators.

    All calculator implementations must inherit from this class
    and implement the abstract methods.
    """

    @abstractmethod
    def relax(
        self,
        structure: Structure,
        fmax: float = 0.05,
        steps: int = 500,
        relax_cell: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Relax a structure using the ML potential.

        Args:
            structure: pymatgen Structure to relax
            fmax: Force convergence criterion in eV/Å
            steps: Maximum number of relaxation steps
            relax_cell: Whether to relax cell parameters
            verbose: Whether to print relaxation progress

        Returns:
            Dictionary with keys:
                - final_structure: Relaxed pymatgen Structure
                - energy: Total energy in eV
                - trajectory: Optional list of energies during relaxation
        """
        ...

    @abstractmethod
    def get_energy(self, structure: Structure) -> float:
        """Get total energy of a structure.

        Args:
            structure: pymatgen Structure

        Returns:
            Total energy in eV
        """
        ...

    @abstractmethod
    def get_forces(self, structure: Structure) -> np.ndarray:
        """Get forces on atoms in a structure.

        Args:
            structure: pymatgen Structure

        Returns:
            Forces array of shape (N, 3) in eV/Å
        """
        ...

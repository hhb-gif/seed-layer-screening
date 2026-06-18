"""MACE calculator implementation."""

from typing import Any, Dict

import numpy as np
from pymatgen.core import Structure

from .base import CalculatorBase


class MACECalculator(CalculatorBase):
    """MACE machine learning potential calculator.

    Uses MACE-MPA-0 (or other MACE foundation models) via the ASE interface.
    Requires ``mace-torch`` to be installed.
    """

    def __init__(
        self,
        model: str = "medium-mpa-0",
        device: str = "cuda",
        default_dtype: str = "float64",
        dispersion: bool = False,
        **kwargs,
    ):
        """Initialize MACE calculator.

        Args:
            model: Model name or path. Common choices:
                - ``"medium-mpa-0"`` (default, recommended)
                - ``"medium"`` (old MACE-MP-0)
                - ``"small"``, ``"large"``
                - path to a custom model file
            device: ``"cuda"`` or ``"cpu"``
            default_dtype: ``"float64"`` (more accurate) or ``"float32"`` (faster)
            dispersion: Whether to enable D3 dispersion correction
            **kwargs: Reserved for future use
        """
        from mace.calculators import mace_mp

        self._calculator = mace_mp(
            model=model,
            device=device,
            default_dtype=default_dtype,
            dispersion=dispersion,
        )

    def _to_ase(self, structure: Structure):
        """Convert pymatgen Structure to ASE Atoms with calculator attached."""
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor().get_atoms(structure)
        atoms.calc = self._calculator
        return atoms

    @staticmethod
    def _to_pmg(atoms) -> Structure:
        """Convert ASE Atoms back to pymatgen Structure."""
        from pymatgen.io.ase import AseAtomsAdaptor

        return AseAtomsAdaptor().get_structure(atoms)

    def relax(
        self,
        structure: Structure,
        fmax: float = 0.05,
        steps: int = 500,
        relax_cell: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Relax structure using MACE + ASE BFGS.

        Args:
            structure: pymatgen Structure to relax
            fmax: Force convergence criterion in eV/Å
            steps: Maximum number of relaxation steps
            relax_cell: Whether to relax cell parameters
            verbose: Whether to print relaxation progress

        Returns:
            Dictionary with keys ``final_structure``, ``energy``, ``trajectory``
        """
        from ase.filters import ExpCellFilter
        from ase.optimize import BFGS

        atoms = self._to_ase(structure)

        # Optionally wrap with cell filter for NPT-like relaxation
        target = ExpCellFilter(atoms) if relax_cell else atoms

        opt = BFGS(target, logfile=None)
        opt.run(fmax=fmax, steps=steps)

        # Extract results
        final_structure = self._to_pmg(atoms)
        energy = atoms.get_potential_energy()

        return {
            "final_structure": final_structure,
            "energy": energy,
            "trajectory": None,  # MACE trajectory not captured by default
        }

    def get_energy(self, structure: Structure) -> float:
        """Get total energy of a structure.

        Args:
            structure: pymatgen Structure

        Returns:
            Total energy in eV
        """
        atoms = self._to_ase(structure)
        return atoms.get_potential_energy()

    def get_forces(self, structure: Structure) -> np.ndarray:
        """Get forces on atoms in a structure.

        Args:
            structure: pymatgen Structure

        Returns:
            Forces array of shape (N, 3) in eV/Å
        """
        atoms = self._to_ase(structure)
        return atoms.get_forces()

    def get_ase_calculator(self):
        """Return the underlying MACE ASE calculator.

        Returns:
            MACE ASE calculator instance
        """
        return self._calculator

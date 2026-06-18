"""Interface energy calculation via layer-by-layer extrapolation."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base import BaseStep

logger = logging.getLogger(__name__)


class InterfaceStep(BaseStep):
    """Step 4.5: Interface energy via seed-metal-seed sandwich extrapolation."""

    step_dir_name = "06_interface"

    def run(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate interface energy for each material.

        Args:
            input_data: List of dicts from LatticeStep

        Returns:
            Same list with interface_energy appended to each dict
        """
        if not input_data:
            logger.warning("No materials for interface energy calculation")
            return []

        iconfig = self.config.get("interface", {})
        max_layers = iconfig.get("max_metal_layers", 5)
        slab_thickness = iconfig.get("slab_thickness", 5)
        vacuum = iconfig.get("vacuum", 15.0)
        fmax = iconfig.get("fmax", 0.05)
        steps = iconfig.get("steps", 500)

        for item in input_data:
            material_id = item["material_id"]
            logger.info(f"Calculating interface energy for {material_id}")
            step_dir = self.get_material_step_dir(self.output_dir / material_id)

            try:
                result = self._calc_interface(
                    item, step_dir,
                    max_layers=max_layers,
                    slab_thickness=slab_thickness,
                    vacuum=vacuum,
                    fmax=fmax,
                    steps=steps,
                )
                if result:
                    item["interface_energy"] = result
                    logger.info(
                        f"  {material_id}: γ = {result['interface_energy_eV_per_A2']:.4f} eV/Å²"
                    )
            except Exception as e:
                logger.warning(f"Interface energy failed for {material_id}: {e}")
                item["interface_energy"] = None

        return input_data

    def _calc_interface(
        self, item: Dict, step_dir: Path,
        max_layers: int, slab_thickness: float,
        vacuum: float, fmax: float, steps: int,
    ) -> Dict[str, Any]:
        """Calculate interface energy for one material.

        Returns:
            Dict with interface_energy_eV_per_A2, bulk_energy_per_layer_eV, R2, etc.
        """
        raise NotImplementedError("Subclasses must implement _calc_interface")

    def _build_sandwich(
        self, seed_slab, ref_bulk, miller,
        film_scale_a, film_scale_b, ref_scale_a, ref_scale_b,
        n_metal_layers, slab_thickness, vacuum,
    ):
        """Build seed-metal-seed sandwich structure.

        Args:
            seed_slab: pymatgen Slab (best_slab from lattice step)
            ref_bulk: pymatgen Structure (reference metal bulk)
            miller: tuple, Miller indices (same for both seed and metal)
            film_scale_a/b: int, seed slab supercell scaling
            ref_scale_a/b: int, metal slab supercell scaling
            n_metal_layers: int, number of metal layers
            slab_thickness: float, slab thickness in Å
            vacuum: float, vacuum thickness in Å

        Returns:
            ase.Atoms: seed-metal-seed sandwich with selective dynamics
        """
        from ase import Atoms
        from ase.build import make_supercell
        from ase.constraints import FixAtoms
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core.surface import SlabGenerator

        adaptor = AseAtomsAdaptor()

        # 1. Build seed slab supercell
        seed_ase = adaptor.get_atoms(seed_slab)
        P_seed = [[film_scale_a, 0, 0], [0, film_scale_b, 0], [0, 0, 1]]
        seed_ase = make_supercell(seed_ase, P_seed)

        # 2. Build metal slab
        slabgen = SlabGenerator(
            ref_bulk, miller, slab_thickness, vacuum,
            center_slab=True, primitive=False
        )
        metal_slabs = slabgen.get_slabs()
        if not metal_slabs:
            raise ValueError(f"No slabs generated for ref metal {miller}")
        metal_slab = metal_slabs[0]

        metal_ase = adaptor.get_atoms(metal_slab)
        P_ref = [[ref_scale_a, 0, 0], [0, ref_scale_b, 0], [0, 0, 1]]
        metal_ase = make_supercell(metal_ase, P_ref)

        # 3. Strip vacuum from seed slab (keep only atomic region)
        seed_positions = seed_ase.get_positions()
        seed_cell = seed_ase.get_cell()
        z_seed = seed_positions[:, 2]
        seed_height = z_seed.max() - z_seed.min()
        # Shift seed atoms so bottom is at z=0
        seed_positions[:, 2] -= z_seed.min()
        seed_ase.set_positions(seed_positions)
        seed_cell[2, 2] = seed_height + vacuum
        seed_ase.set_cell(seed_cell)

        # 4. Strip vacuum from metal slab, keep only n layers
        metal_positions = metal_ase.get_positions()
        z_metal = metal_positions[:, 2]
        # Sort atoms by z, take only the bottom n layers
        z_sorted = np.sort(np.unique(np.round(z_metal, 2)))
        if n_metal_layers > len(z_sorted):
            raise ValueError(
                f"Requested {n_metal_layers} layers but metal slab has {len(z_sorted)}"
            )
        z_cutoff = z_sorted[n_metal_layers - 1] + 0.5  # include atoms in the target layer
        metal_mask = metal_positions[:, 2] <= z_cutoff
        metal_indices = np.where(metal_mask)[0]
        metal_atoms = metal_ase[metal_indices]
        # Shift so bottom is at z=0
        z_metal_min = metal_atoms.get_positions()[:, 2].min()
        metal_positions = metal_atoms.get_positions()
        metal_positions[:, 2] -= z_metal_min
        metal_atoms.set_positions(metal_positions)

        # 5. Create mirror seed (flip z)
        mirror_positions = seed_ase.get_positions().copy()
        mirror_positions[:, 2] = seed_height - mirror_positions[:, 2]
        mirror_ase = seed_ase.copy()
        mirror_ase.set_positions(mirror_positions)

        # 6. Stack: seed + metal + mirror
        # seed: z = [0, seed_height]
        # metal: z = [seed_height, seed_height + metal_height]
        # mirror: z = [seed_height + metal_height, 2*seed_height + metal_height]
        metal_height = metal_atoms.get_positions()[:, 2].max()

        # Shift metal to sit on top of seed
        metal_pos = metal_atoms.get_positions()
        metal_pos[:, 2] += seed_height
        metal_atoms.set_positions(metal_pos)

        # Shift mirror to sit on top of metal
        mirror_pos = mirror_ase.get_positions()
        mirror_pos[:, 2] += seed_height + metal_height
        mirror_ase.set_positions(mirror_pos)

        # Combine
        sandwich = seed_ase + metal_atoms + mirror_ase

        # Set cell with vacuum on top
        total_height = 2 * seed_height + metal_height
        cell = seed_ase.get_cell().copy()
        cell[2, 2] = total_height + vacuum
        sandwich.set_cell(cell)
        sandwich.set_pbc(True)

        # 7. Fix seed layers (bottom and top)
        n_seed = len(seed_ase)
        n_mirror = len(mirror_ase)
        seed_indices = list(range(n_seed)) + list(range(n_seed + len(metal_atoms), len(sandwich)))
        sandwich.set_constraint(FixAtoms(indices=seed_indices))

        return sandwich

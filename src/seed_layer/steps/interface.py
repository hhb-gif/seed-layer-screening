"""Interface energy calculation via layer-by-layer extrapolation."""

import json
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
                    gamma = result.get("interface_energy_eV_per_A2")
                    if gamma is not None:
                        logger.info(f"  {material_id}: γ = {gamma:.4f} eV/Å²")
                    else:
                        logger.warning(f"  {material_id}: interface energy could not be computed")
            except Exception as e:
                logger.warning(f"Interface energy failed for {material_id}: {e}")
                item["interface_energy"] = None

        return input_data

    def _calc_interface(
        self, item: Dict, step_dir: Path,
        max_layers: int, slab_thickness: float,
        vacuum: float, fmax: float, steps: int,
    ) -> Dict[str, Any]:
        """Calculate interface energy for one material."""
        from pymatgen.io.ase import AseAtomsAdaptor
        from ..io import save_structure_cif

        adaptor = AseAtomsAdaptor()
        material_id = item["material_id"]
        miller = item["miller"]

        # Get structures from lattice step
        seed_slab = item.get("best_slab")
        relaxed_bulk = item.get("relaxed_bulk")
        if seed_slab is None or relaxed_bulk is None:
            raise ValueError("Missing best_slab or relaxed_bulk from lattice step")

        # Get scaling factors
        film_scale_a = item.get("film_scale_a", 1)
        film_scale_b = item.get("film_scale_b", 1)
        ref_scale_a = item.get("ref_scale_a", 1)
        ref_scale_b = item.get("ref_scale_b", 1)

        # Build reference metal structure (not the seed's relaxed_bulk!)
        ref_struct = self.build_ref_structure()
        ref_bulk = self.calculator.relax(
            ref_struct, fmax=fmax, steps=steps, relax_cell=True
        )["final_structure"]

        # Relax seed slab separately to get E_seed
        logger.info("Relaxing seed slab...")
        seed_relaxed = self.calculator.relax(
            seed_slab, fmax=fmax, steps=steps, relax_cell=False
        )
        e_seed = seed_relaxed["energy"]
        seed_struct_relaxed = seed_relaxed["final_structure"]
        save_structure_cif(seed_struct_relaxed, step_dir / "seed_slab.cif")

        # Save metal bulk
        save_structure_cif(ref_bulk, step_dir / "metal_bulk.cif")

        # Metal slab uses same Miller index as seed layer
        # (the interface plane must match on both sides)

        # Build max-thickness metal slab, then slice for each n
        from pymatgen.core.surface import SlabGenerator
        slabgen = SlabGenerator(
            ref_bulk, miller, slab_thickness, vacuum,
            center_slab=True, primitive=False
        )
        metal_slabs = slabgen.get_slabs()
        if not metal_slabs:
            raise ValueError(f"No metal slabs generated for {miller}")
        metal_slab_pmg = metal_slabs[0]

        # Convert to ASE and expand
        from ase.build import make_supercell
        metal_ase = adaptor.get_atoms(metal_slab_pmg)
        P_ref = [[ref_scale_a, 0, 0], [0, ref_scale_b, 0], [0, 0, 1]]
        metal_ase = make_supercell(metal_ase, P_ref)

        # Count available metal layers
        z_metal = metal_ase.get_positions()[:, 2]
        z_layers = np.sort(np.unique(np.round(z_metal, 2)))
        actual_max = min(max_layers, len(z_layers))
        if actual_max < 1:
            raise ValueError("No metal layers available")

        logger.info(f"Metal slab has {len(z_layers)} layers, computing up to {actual_max}")

        # Calculate area from seed slab lattice (2D cross product magnitude)
        seed_cell = adaptor.get_atoms(seed_slab).get_cell()
        area = np.linalg.norm(np.cross(seed_cell[0][:2], seed_cell[1][:2]))

        # Layer-by-layer calculation
        energies = {}
        for n in range(1, actual_max + 1):
            logger.info(f"  Computing n={n}/{actual_max}...")
            try:
                sandwich = self._build_sandwich(
                    seed_slab, ref_bulk, miller,
                    film_scale_a, film_scale_b, ref_scale_a, ref_scale_b,
                    n_metal_layers=n,
                    slab_thickness=slab_thickness,
                    vacuum=vacuum,
                )
                # Relax metal atoms only (seed is fixed by constraint)
                result = self.calculator.relax(
                    adaptor.get_structure(sandwich),
                    fmax=fmax, steps=steps, relax_cell=False,
                )
                e_total = result["energy"]
                energies[str(n)] = round(e_total, 6)

                # Save relaxed structure
                save_structure_cif(
                    result["final_structure"],
                    step_dir / f"sandwich_n{n}.cif",
                )
                logger.info(f"    n={n}: E = {e_total:.4f} eV")
            except Exception as e:
                logger.warning(f"    n={n} failed: {e}")
                energies[str(n)] = None

        # Linear fit
        valid_ns = [int(k) for k, v in energies.items() if v is not None]
        valid_es = [energies[str(n)] for n in valid_ns]

        result_dict = {
            "material_id": material_id,
            "seed_slab_energy_eV": round(e_seed, 6),
            "area_A2": round(area, 4),
            "energies_per_n": energies,
        }

        if len(valid_ns) >= 2:
            coeffs = np.polyfit(valid_ns, valid_es, 1)
            slope = coeffs[0]  # bulk energy per layer
            intercept = coeffs[1]  # 2*E_seed + E_interface
            # R² calculation
            predicted = np.polyval(coeffs, valid_ns)
            ss_res = np.sum((np.array(valid_es) - predicted) ** 2)
            ss_tot = np.sum((np.array(valid_es) - np.mean(valid_es)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Interface energy: γ = (intercept - 2*E_seed) / (2*A)
            interface_energy = (intercept - 2 * e_seed) / (2 * area)

            result_dict["interface_energy_eV_per_A2"] = round(interface_energy, 6)
            result_dict["bulk_energy_per_layer_eV"] = round(slope, 6)
            result_dict["R2"] = round(r2, 4)
            result_dict["intercept_eV"] = round(intercept, 6)

            if r2 < 0.95:
                logger.warning(f"  Low R² = {r2:.3f} for {material_id}")

            # Generate plot
            self._plot_interface(
                step_dir, valid_ns, valid_es, slope, intercept, r2,
                interface_energy, area, material_id,
            )
        else:
            logger.warning(f"  Not enough data points for linear fit ({len(valid_ns)})")
            result_dict["interface_energy_eV_per_A2"] = None
            result_dict["bulk_energy_per_layer_eV"] = None
            result_dict["R2"] = None

        # Save JSON
        with open(step_dir / "interface.json", "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        return result_dict

    def _plot_interface(
        self, step_dir, ns, energies, slope, intercept, r2,
        interface_energy, area, material_id,
    ):
        """Generate E(n) plot with linear fit."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        # Data points
        ax.scatter(ns, energies, c="blue", s=60, zorder=5, label="Calculated")

        # Fit line
        x_fit = np.linspace(min(ns) - 0.5, max(ns) + 0.5, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, "r--", linewidth=1.5, label="Linear fit")

        # Mark intercept
        ax.scatter([0], [intercept], c="red", s=80, marker="*", zorder=5)

        # Labels
        ax.set_xlabel("Metal layers (n)", fontsize=12)
        ax.set_ylabel("Total energy (eV)", fontsize=12)
        ax.set_title(f"Interface Energy: {material_id}", fontsize=14)

        # Annotation
        textstr = (
            f"γ = {interface_energy:.4f} eV/Å²\n"
            f"slope = {slope:.4f} eV/layer\n"
            f"R² = {r2:.4f}"
        )
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(step_dir / "interface_plot.png", dpi=150)
        plt.close()

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

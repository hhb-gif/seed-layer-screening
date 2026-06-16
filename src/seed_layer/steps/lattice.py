"""Step 3: Lattice mismatch calculation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator, generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .base import BaseStep
from ..io import save_json, save_structure_cif, save_lattice_params, create_material_dir, create_miller_dir

logger = logging.getLogger(__name__)


class LatticeStep(BaseStep):
    """Calculate lattice mismatch between materials and Li."""

    def run(self, stable_ids: List[str]) -> List[Dict[str, Any]]:
        """Run lattice matching for stable materials.

        Args:
            stable_ids: List of stable material IDs

        Returns:
            List of dicts with material_id, best_miller, mismatch_pct
        """
        results = []
        api_key = self.config.api.get("mp_api_key", "")
        li_miller = tuple(self.config.lattice.get("li_miller", [1, 1, 0]))

        with MPRester(api_key) as mpr:
            # Build Li reference slab
            li_entries = mpr.get_entries_in_chemsys(["Li"])
            li_struct = sorted(li_entries, key=lambda e: e.energy_per_atom)[0].structure

            slabgen = SlabGenerator(
                li_struct,
                li_miller,
                self.config.surface.get("slab_thickness", 15.0),
                self.config.surface.get("vacuum", 15.0),
                center_slab=True,
                primitive=True,
            )
            li_slab = slabgen.get_slabs()[0]

            for mp_id in stable_ids:
                try:
                    best = self._calc_lattice_match(mp_id, li_slab, mpr)
                    if best:
                        results.append(best)
                        logger.info(f"  {mp_id}: best mismatch = {best['mismatch_pct']:.2f}%")
                except Exception as e:
                    logger.error(f"  {mp_id}: Error - {e}")

        return results

    def _calc_lattice_match(self, mp_id: str, li_slab, mpr) -> Optional[Dict]:
        """Calculate lattice mismatch for a single material.

        Args:
            mp_id: Material ID
            li_slab: Li reference slab
            mpr: MPRester instance

        Returns:
            Dict with best match info, or None if failed
        """
        structure = mpr.get_structure_by_material_id(mp_id)
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

        # Relax bulk
        result = self.calculator.relax(
            structure,
            fmax=self.config.relaxation.get("fmax_bulk", 0.05),
            steps=self.config.relaxation.get("steps_bulk", 500),
            relax_cell=True,
            verbose=False,
        )
        relaxed_structure = result["final_structure"]

        # Save bulk structures
        mat_dir = create_material_dir(self.output_dir, mp_id)
        save_structure_cif(structure, mat_dir / "bulk.cif")
        save_structure_cif(relaxed_structure, mat_dir / "bulk_relaxed.cif")
        save_lattice_params(relaxed_structure, mat_dir / "lattice_params.json")

        # Generate all low-index slabs
        try:
            slabs = generate_all_slabs(
                relaxed_structure,
                max_index=1,
                min_slab_size=self.config.surface.get("slab_thickness", 15.0),
                min_vacuum_size=self.config.surface.get("vacuum", 15.0),
                center_slab=True,
                primitive=True,
            )
        except Exception as e:
            logger.warning(f"  {mp_id}: Slab generation failed - {e}")
            return None

        best_mismatch = float("inf")
        best_info = None

        for slab in slabs:
            miller = slab.miller_index
            mismatch, info = self._compute_mismatch(li_slab.lattice, slab.lattice)

            if mismatch < best_mismatch:
                best_mismatch = mismatch
                best_info = {
                    "material_id": mp_id,
                    "miller": miller,
                    "mismatch_pct": round(mismatch, 3),
                    "mismatch_a_pct": round(info.get("mismatch_a", 0), 3),
                    "mismatch_b_pct": round(info.get("mismatch_b", 0), 3),
                    "angle_diff_deg": round(info.get("angle_diff", 0), 2),
                }

        # Save mismatch data
        if best_info:
            miller_dir = create_miller_dir(mat_dir, best_info["miller"])
            save_json(best_info, miller_dir / "mismatch.json")

        return best_info

    def _compute_mismatch(self, li_lattice, film_lattice) -> Tuple[float, Optional[Dict]]:
        """Compute minimum lattice mismatch.

        Args:
            li_lattice: Li slab lattice
            film_lattice: Film slab lattice

        Returns:
            Tuple of (mismatch_pct, info_dict)
        """
        max_scale = self.config.lattice.get("max_scale", 8)
        angle_tolerance = self.config.lattice.get("angle_tolerance", 180.0)

        a_f = np.array(film_lattice.matrix[0][:2])
        b_f = np.array(film_lattice.matrix[1][:2])
        a_li = np.array(li_lattice.matrix[0][:2])
        b_li = np.array(li_lattice.matrix[1][:2])

        len_a_f, len_b_f = np.linalg.norm(a_f), np.linalg.norm(b_f)
        len_a_li, len_b_li = np.linalg.norm(a_li), np.linalg.norm(b_li)

        def get_angle(v1, v2):
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

        angle_diff = abs(get_angle(a_f, b_f) - get_angle(a_li, b_li))

        best = float("inf")
        best_info = None

        for n1 in range(1, max_scale + 1):
            for m1 in range(1, max_scale + 1):
                ma = abs(n1 * len_a_f - m1 * len_a_li) / (m1 * len_a_li) * 100
                if ma > best:
                    continue

                for n2 in range(1, max_scale + 1):
                    for m2 in range(1, max_scale + 1):
                        mb = abs(n2 * len_b_f - m2 * len_b_li) / (m2 * len_b_li) * 100
                        max_mm = max(ma, mb)

                        if max_mm < best and angle_diff <= angle_tolerance:
                            best = max_mm
                            best_info = {
                                "mismatch_a": ma,
                                "mismatch_b": mb,
                                "film_scale_a": n1,
                                "film_scale_b": n2,
                                "li_scale_a": m1,
                                "li_scale_b": m2,
                                "angle_diff": angle_diff,
                            }

        if best == float("inf"):
            return 999.0, None

        return best, best_info

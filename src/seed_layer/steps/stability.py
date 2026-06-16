"""Step 2: Electrochemical stability screening."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition

from .base import BaseStep
from ..io import save_json, save_structure_cif, create_material_dir

logger = logging.getLogger(__name__)


class StabilityStep(BaseStep):
    """Screen materials for electrochemical stability against Li."""

    def run(self, materials: List[Dict[str, Any]]) -> List[str]:
        """Run stability screening.

        Args:
            materials: List of material dicts with 'material_id' and 'formula'

        Returns:
            List of stable material IDs
        """
        stable_ids = []
        api_key = self.config.api.get("mp_api_key", "")

        with MPRester(api_key) as mpr:
            # Get Li reference energy
            li_entries = mpr.get_entries_in_chemsys(["Li"])
            li_entry = sorted(li_entries, key=lambda e: e.energy_per_atom)[0]
            e_li = li_entry.energy_per_atom

            for mat in materials:
                mp_id = mat["material_id"]
                formula = mat["formula"]

                try:
                    passed, details = self._check_stability(mp_id, mpr, e_li)

                    # Save results
                    mat_dir = create_material_dir(self.output_dir, mp_id)
                    save_json(
                        {
                            "material_id": mp_id,
                            "formula": formula,
                            "passed": passed,
                            "min_dE": details.get("min_dE"),
                            "details": details.get("details"),
                        },
                        mat_dir / "stability.json",
                    )

                    if passed:
                        stable_ids.append(mp_id)
                        logger.info(f"  {mp_id} ({formula}): PASSED")
                    else:
                        logger.info(f"  {mp_id} ({formula}): FAILED - {details.get('details')}")

                except Exception as e:
                    logger.error(f"  {mp_id}: Error - {e}")

        return stable_ids

    def _check_stability(self, mp_id: str, mpr: MPRester, e_li: float) -> Tuple[bool, Dict]:
        """Check if material is stable against Li.

        Args:
            mp_id: Materials Project ID
            mpr: MPRester instance
            e_li: Li reference energy per atom

        Returns:
            Tuple of (passed, details_dict)
        """
        # Get material entry
        docs = mpr.summary.search(
            material_ids=[mp_id],
            fields=["material_id", "elements", "formula_pretty"],
        )
        if not docs:
            return False, {"details": "Not found in MP"}

        elements = [el.symbol for el in docs[0].elements]
        formula = docs[0].formula_pretty
        chemsys = "-".join(sorted(elements))

        entries = mpr.get_entries_in_chemsys(chemsys)
        target = [e for e in entries if e.entry_id == mp_id]
        if not target:
            target = [e for e in entries if e.composition.reduced_formula == formula]
        if not target:
            return False, {"details": "Entry not found"}

        mat_entry = sorted(target, key=lambda e: e.energy_per_atom)[0]
        e_mat = mat_entry.energy_per_atom
        comp_mat = mat_entry.composition

        # Build phase diagram with Li
        all_elements = list(set(elements + ["Li"]))
        full_chemsys = "-".join(sorted(all_elements))
        all_entries = mpr.get_entries_in_chemsys(full_chemsys)
        pd_phase = PhaseDiagram(all_entries)

        # Check if material is stable
        stable_ids = {e.entry_id for e in pd_phase.stable_entries}
        if mat_entry.entry_id not in stable_ids:
            return False, {"details": "Material itself unstable"}

        # Scan composition line to Li
        frac = comp_mat.fractional_composition
        min_dE = 0.0
        n_steps = 100

        for j in range(1, n_steps):
            t = j / n_steps
            comp_dict = {el.symbol: (1 - t) * f for el, f in frac.items()}
            comp_dict["Li"] = comp_dict.get("Li", 0) + t
            comp = Composition(comp_dict)

            try:
                e_hull = pd_phase.get_hull_energy(comp)
            except Exception:
                continue

            e_react = (1 - t) * e_mat + t * e_li
            dE = e_hull - e_react

            if dE < min_dE:
                min_dE = dE

        tolerance = 0.05  # eV/atom
        if min_dE < -tolerance:
            return False, {"min_dE": min_dE, "details": f"Reacts with Li: dE={min_dE:.3f}"}

        return True, {"min_dE": min_dE, "details": f"Stable: dE={min_dE:.3f}"}

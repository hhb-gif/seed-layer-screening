"""Step 2: Electrochemical stability screening."""

import logging
from typing import Any, Dict, List, Tuple

from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element

from .base import BaseStep
from ..io import save_json, create_material_dir

logger = logging.getLogger(__name__)


class StabilityStep(BaseStep):
    """Screen materials for electrochemical stability against working ion."""

    step_dir_name = "01_stability"

    def run(self, materials: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Run stability screening.

        Args:
            materials: List of material dicts with 'material_id' and 'formula'

        Returns:
            Tuple of (stable_ids, stability_results)
            - stable_ids: List of stable material IDs
            - stability_results: Dict mapping material_id to {"min_dE", "e_above_hull"}
        """
        stable_ids = []
        stability_results = {}
        api_key = self.config.api.get("mp_api_key", "")

        with MPRester(api_key) as mpr:
            for mat in materials:
                mp_id = mat["material_id"]
                formula = mat.get("formula", mp_id)

                try:
                    passed, details = self._check_stability(mp_id, mpr)

                    mat_dir = create_material_dir(self.output_dir, mp_id)
                    step_dir = self.get_material_step_dir(mat_dir)
                    save_json(
                        {
                            "material_id": mp_id,
                            "formula": formula,
                            "passed": passed,
                            "min_dE": details.get("min_dE"),
                            "details": details.get("details"),
                        },
                        step_dir / "stability.json",
                    )

                    stability_results[mp_id] = {
                        "min_dE": details.get("min_dE"),
                        "e_above_hull": details.get("e_above_hull"),
                    }

                    if passed:
                        stable_ids.append(mp_id)
                        logger.info(f"  {mp_id} ({formula}): PASSED")
                    else:
                        logger.info(f"  {mp_id} ({formula}): FAILED - {details.get('details')}")

                except Exception as e:
                    logger.error(f"  {mp_id}: Error - {e}")

        return stable_ids, stability_results

    def _check_stability(self, mp_id: str, mpr: MPRester) -> Tuple[bool, Dict]:
        """Check if material is stable against working ion.

        Args:
            mp_id: Materials Project ID
            mpr: MPRester instance

        Returns:
            Tuple of (passed, details_dict)
        """
        working_ion = self.config.working_ion
        # Get material info from summary (more reliable than thermo entries)
        docs = mpr.summary.search(
            material_ids=[mp_id],
            fields=["material_id", "elements", "formula_pretty",
                    "formation_energy_per_atom", "energy_above_hull"],
        )
        if not docs:
            return False, {"details": "Not found in MP", "e_above_hull": None}

        doc = docs[0]
        elements = [el.symbol for el in doc.elements]
        formula = doc.formula_pretty
        e_above_hull = doc.energy_above_hull

        # Basic stability check: energy above hull
        if e_above_hull is None:
            return False, {"details": "No energy_above_hull data", "e_above_hull": None}

        # Check if material contains working ion (relevant for seed layer)
        has_ion = working_ion in elements

        # Use energy_above_hull as proxy for stability
        # Materials with e_above_hull = 0 are on the convex hull (thermodynamically stable)
        tolerance = 0.05  # eV/atom

        if e_above_hull > tolerance:
            return False, {
                "min_dE": -e_above_hull,
                "e_above_hull": e_above_hull,
                "details": f"Unstable: e_above_hull={e_above_hull:.4f} eV/atom",
            }

        # For materials containing working ion, check if they react with more ion
        if has_ion:
            # Build phase diagram from entries
            try:
                all_elements = sorted(set(elements + [working_ion]))
                chemsys = "-".join(all_elements)
                entries = mpr.get_entries_in_chemsys(
                    chemsys, compatible_only=False,
                )

                # Filter out entries that are raw dicts (deserialization failures)
                valid_entries = []
                for e in entries:
                    try:
                        _ = e.composition
                        _ = e.energy_per_atom
                        valid_entries.append(e)
                    except (AttributeError, TypeError):
                        continue

                if len(valid_entries) < 2:
                    logger.warning(f"  {mp_id}: Not enough valid entries for phase diagram, using e_above_hull")
                    return True, {
                        "min_dE": -e_above_hull,
                        "e_above_hull": e_above_hull,
                        "details": f"Stable (e_above_hull={e_above_hull:.4f}, no PD check)",
                    }

                pd_phase = PhaseDiagram(valid_entries)

                # Find our entry
                target = None
                for e in valid_entries:
                    try:
                        if e.entry_id == mp_id:
                            target = e
                            break
                    except AttributeError:
                        continue

                if target is None:
                    # Try matching by formula
                    for e in valid_entries:
                        try:
                            if e.composition.reduced_formula == formula:
                                target = e
                                break
                        except AttributeError:
                            continue

                if target is None:
                    return True, {
                        "min_dE": -e_above_hull,
                        "e_above_hull": e_above_hull,
                        "details": f"Stable (e_above_hull={e_above_hull:.4f}, entry not in PD)",
                    }

                # Scan composition line from material to pure working ion
                comp_mat = target.composition
                frac = comp_mat.fractional_composition

                # Get working ion reference energy
                ion_entries = [e for e in valid_entries
                               if len(e.composition.elements) == 1
                               and e.composition.elements[0] == Element(working_ion)]
                if ion_entries:
                    e_ion = min(e.energy_per_atom for e in ion_entries)
                else:
                    e_ion = -1.9  # Fallback approximate energy per atom

                e_mat = target.energy_per_atom
                min_dE = 0.0
                n_steps = 100

                for j in range(1, n_steps):
                    t = j / n_steps
                    comp_dict = {el.symbol: (1 - t) * f for el, f in frac.items()}
                    comp_dict[working_ion] = comp_dict.get(working_ion, 0) + t
                    comp = Composition(comp_dict)

                    try:
                        e_hull = pd_phase.get_hull_energy(comp)
                    except Exception:
                        continue

                    e_react = (1 - t) * e_mat + t * e_ion
                    dE = e_hull - e_react

                    if dE < min_dE:
                        min_dE = dE

                if min_dE < -tolerance:
                    return False, {
                        "min_dE": min_dE,
                        "e_above_hull": e_above_hull,
                        "details": f"Reacts with {working_ion}: dE={min_dE:.3f}",
                    }

                return True, {"min_dE": min_dE, "e_above_hull": e_above_hull, "details": f"Stable: dE={min_dE:.3f}"}

            except Exception as e:
                logger.warning(f"  {mp_id}: Phase diagram failed ({e}), using e_above_hull")
                return e_above_hull <= tolerance, {
                    "min_dE": -e_above_hull,
                    "e_above_hull": e_above_hull,
                    "details": f"e_above_hull={e_above_hull:.4f} (PD check failed)",
                }

        # For materials without working ion, just check e_above_hull
        return e_above_hull <= tolerance, {
            "min_dE": -e_above_hull,
            "e_above_hull": e_above_hull,
            "details": f"e_above_hull={e_above_hull:.4f} eV/atom",
        }

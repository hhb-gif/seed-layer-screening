"""Step 4: Adsorption energy calculation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseStep
from ..io import save_structure_cif, save_json, create_material_dir

logger = logging.getLogger(__name__)


class AdsorptionStep(BaseStep):
    """Calculate Li adsorption energies on material surfaces."""

    def run(self, matched_materials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run adsorption energy calculation.

        Args:
            matched_materials: List of dicts with material_id, miller, mismatch_pct

        Returns:
            Dict mapping material_id to adsorption results
        """
        if not matched_materials:
            logger.warning("No materials provided for adsorption calculation")
            return {}

        results = {}
        li_ref_energy = self._get_li_reference_energy()

        for item in matched_materials:
            material_id = item['material_id']
            miller = item['miller']
            logger.info(f"Calculating adsorption for {material_id} {miller}")

            try:
                material_results = self._calc_adsorption_for_material(
                    material_id, miller, li_ref_energy
                )
                results[material_id] = {
                    'miller': miller,
                    'adsorption_energies': material_results,
                    'status': 'success'
                }

                # Save intermediate results
                material_dir = create_material_dir(self.output_dir, material_id)
                save_json(
                    results[material_id],
                    material_dir / "adsorption.json"
                )

            except Exception as e:
                logger.error(f"Failed adsorption for {material_id}: {e}")
                results[material_id] = {
                    'miller': miller,
                    'adsorption_energies': [],
                    'status': f'error: {e}'
                }

        return results

    def _get_li_reference_energy(self) -> float:
        """Calculate Li reference energy (bcc Li bulk relaxation).

        Returns:
            Energy per atom in eV
        """
        try:
            from pymatgen.core import Structure, Lattice

            # Create bcc Li structure
            li_bcc = Structure(
                Lattice.cubic(3.49),
                ["Li"], [[0, 0, 0]]
            )

            # Relax with calculator
            result = self.calculator.relax(
                li_bcc,
                fmax=self.config.get('relaxation', 'fmax_bulk', 0.05),
                steps=self.config.get('relaxation', 'steps_bulk', 500),
                relax_cell=True
            )

            energy = result["energy"]
            n_atoms = len(result["final_structure"])
            ref_energy = energy / n_atoms

            logger.info(f"Li reference energy: {ref_energy:.4f} eV/atom")
            return ref_energy

        except Exception as e:
            logger.error(f"Failed to calculate Li reference energy: {e}")
            # Fallback to typical value
            return -1.90  # eV/atom for bcc Li

    def _calc_adsorption_for_material(
        self,
        material_id: str,
        miller: tuple,
        li_ref_energy: float
    ) -> List[Dict[str, Any]]:
        """Calculate adsorption energies for a material at multiple coverages.

        Args:
            material_id: Materials Project ID
            miller: Miller indices
            li_ref_energy: Li reference energy per atom

        Returns:
            List of adsorption energy results
        """
        try:
            from pymatgen.core import Structure
            from pymatgen.core.surface import SlabGenerator
            from pymatgen.analysis.adsorption import AdsorbateSiteFinder
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.build import make_supercell
            from ase.constraints import FixAtoms
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return []

        # Get config values
        slab_thickness = self.config.get('surface', 'slab_thickness', 10)
        vacuum = self.config.get('surface', 'vacuum', 15)
        supercell = self.config.get('adsorption', 'supercell', [3, 3, 1])
        coverages = self.config.get('adsorption', 'coverages', [0.25, 0.5, 0.75, 1.0])
        adsorbate_height = self.config.get('adsorption', 'adsorbate_height', 2.0)
        li_area_per_atom = self.config.get('adsorption', 'li_area_per_atom', 5.0)

        fmax_bulk = self.config.get('relaxation', 'fmax_bulk', 0.05)
        steps_bulk = self.config.get('relaxation', 'steps_bulk', 500)
        fmax_slab = self.config.get('relaxation', 'fmax_slab', 0.05)
        steps_slab = self.config.get('relaxation', 'steps_slab', 500)
        fmax_adsorb = self.config.get('relaxation', 'fmax_adsorb', 0.05)
        steps_adsorb = self.config.get('relaxation', 'steps_adsorb', 500)

        # Get structure from MP API
        from mp_api.client import MPRester
        api_key = self.config.get('api', 'api_key', '')

        with MPRester(api_key) as mpr:
            structure = mpr.get_structure_by_material_id(material_id)

        # Relax bulk
        logger.info("Relaxing bulk structure...")
        result_bulk = self.calculator.relax(
            structure, fmax=fmax_bulk, steps=steps_bulk, relax_cell=True
        )
        bulk = result_bulk["final_structure"]

        # Generate slab
        logger.info("Generating slab...")
        slabgen = SlabGenerator(
            bulk, miller, slab_thickness, vacuum,
            center_slab=True, primitive=False
        )
        slabs = slabgen.get_slabs()
        if not slabs:
            raise ValueError(f"No slabs generated for {miller}")
        slab = slabs[0]

        # Create supercell
        adaptor = AseAtomsAdaptor()
        ase_slab = adaptor.get_atoms(slab)
        P = [[supercell[0], 0, 0],
             [0, supercell[1], 0],
             [0, 0, supercell[2]]]
        ase_slab = make_supercell(ase_slab, P)

        # Fix bottom atoms
        z_vals = ase_slab.get_positions()[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        threshold = z_min + (z_max - z_min) * 0.33
        fixed = [i for i, z in enumerate(z_vals) if z < threshold]
        if len(fixed) < 4:
            fixed = list(np.argsort(z_vals)[:max(4, len(ase_slab) // 4)])
        ase_slab.set_constraint(FixAtoms(indices=fixed))

        # Relax clean slab
        logger.info("Relaxing clean slab...")
        slab_pmg = adaptor.get_structure(ase_slab)
        result_clean = self.calculator.relax(
            slab_pmg, fmax=fmax_slab, steps=steps_slab, relax_cell=False
        )
        clean_struct = result_clean["final_structure"]
        e_clean = result_clean["energy"]

        # Get surface height and area
        clean_atoms = adaptor.get_atoms(clean_struct)
        clean_atoms.set_constraint(FixAtoms(indices=fixed))
        z_all = clean_atoms.get_positions()[:, 2]
        movable_z = [z_all[i] for i in range(len(clean_atoms)) if i not in fixed]
        surface_z = np.percentile(movable_z, 90) if len(movable_z) > 10 else max(movable_z)

        cell = clean_atoms.get_cell()
        area = abs(cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0])

        # Save clean slab
        material_dir = create_material_dir(self.output_dir, material_id)
        save_structure_cif(clean_struct, material_dir / "slab_clean.cif")

        # Calculate adsorption energies at different coverages
        results = []
        for cov in coverages:
            if cov == 0:
                continue

            n_li = max(1, int(area / li_area_per_atom * cov))
            logger.info(f"  Coverage {cov} ML: {n_li} Li atoms")

            # Place Li atoms in grid
            ads_atoms = clean_atoms.copy()
            nx = int(np.sqrt(n_li * cell[1, 1] / cell[0, 0])) + 1
            ny = int(n_li / nx) + 1
            dx = cell[0, 0] / (nx + 1)
            dy = cell[1, 1] / (ny + 1)

            added = 0
            for ix in range(1, nx + 1):
                for iy in range(1, ny + 1):
                    if added >= n_li:
                        break
                    ads_atoms.append('Li')
                    ads_atoms.positions[-1] = [
                        ix * dx, iy * dy, surface_z + adsorbate_height
                    ]
                    added += 1
                if added >= n_li:
                    break
            ads_atoms.set_constraint(FixAtoms(indices=fixed))

            # Relax adsorption system
            ads_pmg = adaptor.get_structure(ads_atoms)
            result_ads = self.calculator.relax(
                ads_pmg, fmax=fmax_adsorb, steps=steps_adsorb, relax_cell=False
            )
            e_ads = result_ads["energy"]

            # Calculate adsorption energy
            # E_ads = (E_slab+Li - E_clean - n_Li * E_Li_ref) / n_Li
            e_ads_per_li = (e_ads - e_clean - n_li * li_ref_energy) / n_li

            # Save relaxed adsorption structure
            save_structure_cif(
                result_ads["final_structure"],
                material_dir / f"slab_adsorbed_{cov}ML.cif"
            )

            results.append({
                'coverage_ML': round(cov, 3),
                'n_Li': n_li,
                'E_clean_eV': round(e_clean, 3),
                'E_ads_system_eV': round(e_ads, 3),
                'E_ads_eV': round(e_ads_per_li, 4),
                'status': 'success'
            })

        return results

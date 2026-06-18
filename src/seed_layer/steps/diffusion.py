"""Step 5: NEB diffusion barrier calculation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseStep
from ..io import save_structure_cif, save_json, create_material_dir, ensure_structure

logger = logging.getLogger(__name__)


class DiffusionStep(BaseStep):
    """Calculate working ion diffusion barriers using CI-NEB."""

    step_dir_name = "04_diffusion"

    def run(self, adsorption_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run diffusion barrier calculation.

        Args:
            adsorption_results: Adsorption results by material ID

        Returns:
            Dict mapping material_id to diffusion results
        """
        if not adsorption_results:
            logger.warning("No adsorption results provided for diffusion calculation")
            return {}

        # Get top N materials from scoring
        neb_top_n = self.config.get('diffusion', 'neb_top_n', 10)

        # Sort by adsorption energy (most negative = best)
        sorted_materials = self._sort_materials_by_adsorption(adsorption_results)
        top_materials = sorted_materials[:neb_top_n]

        results = {}
        for item in top_materials:
            material_id = item['material_id']
            logger.info(f"Calculating diffusion for {material_id}")

            try:
                diffusion_result = self._calc_diffusion_for_material(
                    material_id, item['miller'],
                    slab_structure=item.get('slab_structure'),
                )
                results[material_id] = diffusion_result

                # Save intermediate results
                material_dir = create_material_dir(self.output_dir, material_id)
                step_dir = self.get_material_step_dir(material_dir)
                save_json(diffusion_result, step_dir / "diffusion.json")

            except Exception as e:
                logger.error(f"Failed diffusion for {material_id}: {e}")
                results[material_id] = {
                    'miller': item['miller'],
                    'paths': [],
                    'status': f'error: {e}'
                }

        return results

    def _sort_materials_by_adsorption(
        self, adsorption_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Sort materials by best adsorption energy.

        Args:
            adsorption_results: Results from AdsorptionStep

        Returns:
            Sorted list of materials with their best adsorption energy
        """
        sorted_list = []

        for material_id, data in adsorption_results.items():
            if data.get('status') != 'success':
                continue

            energies = data.get('adsorption_energies', [])
            if not energies:
                continue

            # Find most negative adsorption energy
            valid_energies = [
                e['E_ads_eV'] for e in energies
                if e.get('status') == 'success' and e.get('E_ads_eV') is not None
            ]

            if valid_energies:
                best_energy = min(valid_energies)
                sorted_list.append({
                    'material_id': material_id,
                    'miller': data.get('miller', (1, 0, 0)),
                    'best_adsorption_energy': best_energy,
                    'slab_structure': data.get('clean_slab_struct'),
                })

        # Sort by adsorption energy (most negative first)
        sorted_list.sort(key=lambda x: x['best_adsorption_energy'])
        return sorted_list

    def _calc_diffusion_for_material(
        self, material_id: str, miller: tuple, slab_structure=None
    ) -> Dict[str, Any]:
        """Calculate diffusion barriers for a material.

        Args:
            material_id: Materials Project ID
            miller: Miller indices
            slab_structure: Pre-computed slab Structure from adsorption step

        Returns:
            Dict with diffusion barrier results
        """
        try:
            from pymatgen.core import Structure
            from pymatgen.core.surface import SlabGenerator
            from pymatgen.analysis.adsorption import AdsorbateSiteFinder
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.constraints import FixAtoms
            from ase.optimize import BFGS
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return {'miller': miller, 'paths': [], 'status': f'import error: {e}'}

        # Get config values
        slab_thickness = self.config.get('surface', 'slab_thickness', 10)
        vacuum = self.config.get('surface', 'vacuum', 15)
        adsorbate_height = self.config.get('adsorption', 'adsorbate_height', 2.0)

        fmax_bulk = self.config.get('relaxation', 'fmax_bulk', 0.05)
        steps_bulk = self.config.get('relaxation', 'steps_bulk', 500)
        fmax_slab = self.config.get('relaxation', 'fmax_slab', 0.05)
        steps_slab = self.config.get('relaxation', 'steps_slab', 500)

        neb_n_images = self.config.get('diffusion', 'neb_n_images', 5)
        neb_fmax = self.config.get('diffusion', 'neb_fmax', 0.05)
        neb_steps = self.config.get('diffusion', 'neb_steps', 200)
        neb_climb = self.config.get('diffusion', 'neb_climb', True)
        ion_displacement_min = self.config.get('diffusion', 'ion_displacement_min', 0.5)
        working_ion = self.config.working_ion

        # Get or compute slab structure
        if slab_structure is not None:
            logger.info("Using pre-computed slab from adsorption step")
            slab = slab_structure.copy()
            slab.make_supercell([2, 2, 1])  # Smaller supercell for NEB
        else:
            logger.info("Fallback: fetching and computing slab for NEB...")
            from mp_api.client import MPRester
            import gc
            api_key = self.config.get('api', 'mp_api_key', '')

            with MPRester(api_key) as mpr:
                structure = ensure_structure(mpr.get_structure_by_material_id(material_id))

            # Relax bulk
            logger.info("Relaxing bulk structure...")
            result_bulk = self.calculator.relax(
                structure, fmax=fmax_bulk, steps=steps_bulk, relax_cell=True
            )
            bulk = result_bulk["final_structure"]

            # Generate slab
            logger.info("Generating slab for NEB...")
            slabgen = SlabGenerator(
                bulk, miller, slab_thickness, vacuum,
                center_slab=True, primitive=True
            )
            slabs = slabgen.get_slabs()
            if not slabs:
                raise ValueError(f"No slabs generated for {miller}")
            slab = slabs[0].copy()
            slab.make_supercell([2, 2, 1])  # Smaller supercell for NEB

        adaptor = AseAtomsAdaptor()
        ase_slab = adaptor.get_atoms(slab)

        # Fix bottom atoms
        z_vals = ase_slab.get_positions()[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        threshold = z_min + (z_max - z_min) * 0.33
        fixed = [i for i, z in enumerate(z_vals) if z < threshold]
        if len(fixed) < 2:
            fixed = list(np.argsort(z_vals)[:max(2, len(ase_slab) // 3)])
        ase_slab.set_constraint(FixAtoms(indices=fixed))

        # Relax clean slab
        logger.info("Relaxing clean slab...")
        slab_pmg = adaptor.get_structure(ase_slab)
        result_clean = self.calculator.relax(
            slab_pmg, fmax=fmax_slab, steps=steps_slab, relax_cell=False
        )
        clean_slab = result_clean["final_structure"]

        # Generate adsorption sites
        logger.info("Generating adsorption sites...")
        asf = AdsorbateSiteFinder(clean_slab)
        ads_sites = asf.find_adsorption_sites(distance=adsorbate_height)

        # Manual top sites as fallback
        surface_z = max(s.coords[2] for s in clean_slab)
        surface_atoms = [s for s in clean_slab if abs(s.coords[2] - surface_z) < 1.0]
        manual_top = [
            a.coords + np.array([0, 0, adsorbate_height])
            for a in surface_atoms
        ]

        top_sites = ads_sites.get('top', []) or manual_top
        bridge_sites = ads_sites.get('bridge', [])
        hollow_sites = ads_sites.get('hollow', [])

        # Relax working ion at each site type
        def relax_ion_at_site(site):
            atoms = adaptor.get_atoms(clean_slab)
            atoms.append(working_ion)
            atoms.positions[-1] = site
            atoms.set_constraint(FixAtoms(indices=list(range(len(atoms) - 1))))

            # Set calculator
            atoms.calc = self.calculator.get_ase_calculator()

            dyn = BFGS(atoms, logfile=None)
            dyn.run(fmax=0.05, steps=100)
            return atoms

        opt = {}
        for stype, sites in [('top', top_sites), ('bridge', bridge_sites), ('hollow', hollow_sites)]:
            logger.info(f"  Processing {stype} sites ({len(sites)} found)")
            if sites:
                try:
                    opt[stype] = relax_ion_at_site(sites[0])
                except Exception as e:
                    logger.warning(f"  Failed to relax {stype} site: {e}")
                    opt[stype] = None

        # Build paths
        paths = []
        if opt.get('top') and opt.get('bridge'):
            paths.append(('top_to_bridge', opt['top'], opt['bridge']))
        if opt.get('bridge') and opt.get('hollow'):
            paths.append(('bridge_to_hollow', opt['bridge'], opt['hollow']))
        if opt.get('hollow') and opt.get('top'):
            paths.append(('hollow_to_top', opt['hollow'], opt['top']))

        if not paths:
            logger.warning("No diffusion paths found")
            return {
                'miller': miller,
                'paths': [],
                'status': 'no paths found'
            }

        # Run NEB for each path
        material_dir = create_material_dir(self.output_dir, material_id)
        step_dir = self.get_material_step_dir(material_dir)
        results = []

        for path_name, init_atoms, final_atoms in paths:
            logger.info(f"  Calculating path: {path_name}")

            # Unify cells
            final_atoms.set_cell(init_atoms.get_cell(), scale_atoms=True)

            # Check ion displacement
            ion_init = init_atoms[-1].position
            ion_final = final_atoms[-1].position
            ion_dist = np.linalg.norm(ion_init - ion_final)
            logger.info(f"    Ion displacement: {ion_dist:.3f} Å")

            if ion_dist < ion_displacement_min:
                logger.warning(f"    Ion displacement too small, skipping")
                results.append({
                    'path': path_name,
                    'ion_displacement_A': round(ion_dist, 3),
                    'barrier_eV': None,
                    'status': f'displacement too small ({ion_dist:.2f}Å)'
                })
                continue

            # Build NEB images
            images = [init_atoms]
            for _ in range(neb_n_images - 2):
                img = init_atoms.copy()
                img.calc = self.calculator.get_ase_calculator()
                images.append(img)
            images.append(final_atoms)

            # Run NEB
            try:
                from ase.mep import SingleCalculatorNEB
                from ase.optimize import BFGS

                neb = SingleCalculatorNEB(images, climb=neb_climb, method='improvedtangent')
                neb.interpolate(method='idpp')

                opt_neb = BFGS(neb, logfile=None)
                logger.info("    Running NEB optimization...")
                opt_neb.run(fmax=neb_fmax, steps=neb_steps)

                # Get energies
                energies = [img.get_potential_energy() for img in images]
                barrier = max(energies) - energies[0]
                logger.info(f"    Barrier = {barrier:.4f} eV")

                # Save NEB path structures
                path_dir = step_dir / f"neb_{path_name}"
                path_dir.mkdir(exist_ok=True)
                for i, img in enumerate(images):
                    img_pmg = adaptor.get_structure(img)
                    save_structure_cif(img_pmg, path_dir / f"image_{i}.cif")

                results.append({
                    'path': path_name,
                    'ion_displacement_A': round(ion_dist, 3),
                    'barrier_eV': round(barrier, 4),
                    'energies': [round(e, 4) for e in energies],
                    'status': 'success'
                })

            except Exception as e:
                logger.error(f"    NEB failed: {e}")
                results.append({
                    'path': path_name,
                    'ion_displacement_A': round(ion_dist, 3),
                    'barrier_eV': None,
                    'status': f'NEB failed: {e}'
                })

        return {
            'miller': miller,
            'paths': results,
            'status': 'success'
        }

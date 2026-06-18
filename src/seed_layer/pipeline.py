"""Main pipeline orchestrator for seed layer screening."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig
from .calculators import create_calculator, CalculatorBase
from .io import (
    save_json,
    save_structure_cif,
    save_lattice_params,
    create_material_dir,
)

logger = logging.getLogger(__name__)


class SeedLayerPipeline:
    """Main pipeline for seed layer material screening."""

    def __init__(self, config: PipelineConfig, output_dir: Path, tag: Optional[str] = None):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
            output_dir: Base output directory
            tag: Optional tag for output directory naming
        """
        self.config = config
        self.output_dir = output_dir
        self.tag = tag

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = timestamp if not tag else f"{timestamp}_{tag}"
        self.run_dir = output_dir / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize calculator
        self.calculator = create_calculator(config.calculator)

        # Setup logging
        self._setup_logging()

        # Save run config snapshot
        self._save_run_config()

        logger.info(f"Pipeline initialized. Output: {self.run_dir}")

    def _setup_logging(self):
        """Setup logging to file and console."""
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # File handler
        fh = logging.FileHandler(log_dir / "run.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)

    def _save_run_config(self):
        """Save current config to run directory."""
        import yaml

        config_path = self.run_dir / "run_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "api": self.config.api,
                    "working_ion": self.config.working_ion,
                    "ref_structure_id": self.config.ref_structure_id,
                    "ref_miller": list(self.config.ref_miller),
                    "screening": self.config.screening,
                    "lattice": self.config.lattice,
                    "surface": self.config.surface,
                    "calculator": self.config.calculator,
                    "relaxation": self.config.relaxation,
                    "adsorption": self.config.adsorption,
                    "diffusion": self.config.diffusion,
                    "interface": self.config.interface,
                    "scoring": self.config.scoring,
                    "output": self.config.output,
                },
                f,
                default_flow_style=False,
                allow_unicode=True,
            )

    def run(self, materials_file: Optional[str] = None, skip_neb: bool = False):
        """Run the full screening pipeline.

        Args:
            materials_file: Path to materials list file (optional)
            skip_neb: Whether to skip NEB diffusion calculation
        """
        from .steps.stability import StabilityStep
        from .steps.lattice import LatticeStep
        from .steps.interface import InterfaceStep
        from .steps.adsorption import AdsorptionStep
        from .steps.diffusion import DiffusionStep

        # Step 1: Fetch materials
        logger.info("Step 1: Fetching materials pool...")
        materials = self._fetch_materials(materials_file)
        logger.info(f"Found {len(materials)} candidate materials")

        # Step 2: Stability screening
        logger.info("Step 2: Electrochemical stability screening...")
        stability_step = StabilityStep(self.config, self.calculator, self.run_dir)
        stable_materials, stability_results = stability_step.run(materials)
        logger.info(f"{len(stable_materials)} materials passed stability screening")

        # Step 3: Lattice matching
        logger.info("Step 3: Lattice mismatch calculation...")
        lattice_step = LatticeStep(self.config, self.calculator, self.run_dir)
        matched_materials = lattice_step.run(stable_materials)
        logger.info(f"{len(matched_materials)} materials passed lattice matching")

        # Step 3.5: Interface energy
        logger.info("=" * 50)
        logger.info("Step 3.5: Interface Energy")
        logger.info("=" * 50)
        interface_step = InterfaceStep(self.config, self.calculator, self.run_dir)
        matched_materials = interface_step.run(matched_materials)

        # Step 4: Adsorption energy
        logger.info("Step 4: Adsorption energy calculation...")
        adsorption_step = AdsorptionStep(self.config, self.calculator, self.run_dir)
        adsorption_results = adsorption_step.run(matched_materials)
        logger.info(f"Adsorption calculated for {len(adsorption_results)} materials")

        # Step 5: Diffusion barrier (optional)
        if not skip_neb:
            logger.info("Step 5: NEB diffusion barrier calculation...")
            diffusion_step = DiffusionStep(self.config, self.calculator, self.run_dir)
            diffusion_results = diffusion_step.run(adsorption_results)
            logger.info(f"Diffusion calculated for {len(diffusion_results)} materials")
        else:
            logger.info("Step 5: Skipping NEB calculation")
            diffusion_results = {}

        # Collect interface results
        interface_results = {}
        for mat in matched_materials:
            mp_id = mat["material_id"]
            iface = mat.get("interface_energy")
            if iface:
                interface_results[mp_id] = iface

        # Generate summary
        logger.info("Generating summary report...")
        self._generate_summary(stable_materials, matched_materials, adsorption_results, diffusion_results, interface_results, stability_results)

        logger.info("Pipeline complete!")

    def _fetch_materials(self, materials_file: Optional[str] = None) -> List[dict]:
        """Fetch materials from MP API or file.

        Args:
            materials_file: Path to materials list file

        Returns:
            List of material dictionaries, each with at least 'material_id'
        """
        if materials_file:
            return self._read_materials_file(materials_file)
        return self._fetch_from_mp_api()

    def _read_materials_file(self, materials_file: str) -> List[dict]:
        """Read material IDs from a text file.

        Supports one material_id per line. Lines starting with '#' or
        blank lines are skipped.

        Args:
            materials_file: Path to materials list file

        Returns:
            List of material dicts with 'material_id' key
        """
        path = Path(materials_file)
        if not path.exists():
            raise FileNotFoundError(f"Materials file not found: {materials_file}")

        materials = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    materials.append({"material_id": stripped})

        logger.info(f"Read {len(materials)} materials from {materials_file}")
        return materials

    def _fetch_from_mp_api(self) -> List[dict]:
        """Fetch candidate materials from Materials Project API.

        Uses screening config to query MP for candidate materials.

        Returns:
            List of material dicts with 'material_id' and 'formula' keys
        """
        from mp_api.client import MPRester

        api_key = self.config.api.get("mp_api_key", "")
        screening = self.config.screening

        energy_above_hull_max = screening.get("energy_above_hull_max", 0.10)
        n_elements = screening.get("n_elements", [2, 3])
        exclude = set(screening.get("elements_to_exclude", []))

        logger.info(
            f"Querying MP API: energy_above_hull <= {energy_above_hull_max}, "
            f"n_elements in {n_elements}, excluding {exclude}"
        )

        materials = []
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                energy_above_hull=(0, energy_above_hull_max),
                num_elements=tuple(n_elements),
                fields=["material_id", "formula_pretty", "elements"],
            )

            for doc in docs:
                elements = {el.symbol for el in doc.elements}
                if elements & exclude:
                    continue
                materials.append({
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                })

        logger.info(f"Fetched {len(materials)} candidate materials from MP API")
        return materials

    def _generate_summary(
        self,
        stable: List[str],
        matched: List[dict],
        adsorption: dict,
        diffusion: dict,
        interface: dict = None,
        stability: dict = None,
    ):
        """Generate summary CSV.

        Args:
            stable: List of stable material IDs
            matched: List of matched material dicts (from LatticeStep)
            adsorption: Adsorption results by material ID
            diffusion: Diffusion results by material ID
            interface: Interface energy results by material ID
            stability: Stability results by material ID (with e_above_hull)
        """
        from .reporting import generate_summary_csv

        max_mismatch = self.config.get('lattice', 'max_mismatch', 8.0)

        df = generate_summary_csv(
            output_dir=self.run_dir,
            stable_ids=stable,
            matched_materials=matched,
            adsorption_results=adsorption,
            diffusion_results=diffusion,
            interface_results=interface,
            stability_results=stability,
            max_mismatch=max_mismatch,
        )
        logger.info(f"Summary: {len(df)} materials in final report")

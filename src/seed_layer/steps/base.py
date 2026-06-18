"""Base class for screening steps."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from ..config import PipelineConfig
from ..calculators.base import CalculatorBase

logger = logging.getLogger(__name__)


class BaseStep(ABC):
    """Base class for all screening steps."""

    # Subclasses set this (e.g. "01_stability", "02_lattice")
    step_dir_name: str = ""

    def __init__(self, config: PipelineConfig, calculator: CalculatorBase, output_dir: Path):
        """Initialize step.

        Args:
            config: Pipeline configuration
            calculator: ML potential calculator
            output_dir: Output directory for this run
        """
        self.config = config
        self.calculator = calculator
        self.output_dir = output_dir

    def get_material_step_dir(self, material_dir: Path) -> Path:
        """Get step-specific subdirectory under a material directory.

        Args:
            material_dir: Material directory (e.g. run_id/mp-12345)

        Returns:
            Path to step subdirectory (e.g. run_id/mp-12345/01_stability)
        """
        if not self.step_dir_name:
            return material_dir
        step_dir = material_dir / self.step_dir_name
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def build_ref_structure(self):
        """Get reference metal structure from Materials Project.

        If ref_structure_id is set in config, uses that directly.
        Otherwise auto-detects the most stable structure for the working ion.
        """
        from mp_api.client import MPRester

        ion = self.config.working_ion
        api_key = self.config.api.get("mp_api_key", "")

        with MPRester(api_key) as mpr:
            # Priority: explicit ref_structure_id from config
            ref_id = getattr(self.config, "ref_structure_id", None)
            if ref_id:
                return mpr.get_structure_by_material_id(ref_id)
            # Fallback: auto-lookup most stable structure for this element
            ids = mpr.get_materials_ids(ion)
            if not ids:
                raise ValueError(f"No structures found for '{ion}' in MP.")
            return mpr.get_structure_by_material_id(ids[0])

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the screening step.

        Args:
            input_data: Input from previous step

        Returns:
            Output data for next step
        """
        ...

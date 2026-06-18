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

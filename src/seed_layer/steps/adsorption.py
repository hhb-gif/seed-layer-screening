"""Step 4: Adsorption energy calculation."""

import logging
from typing import Any, Dict, List

from .base import BaseStep

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
        # TODO: Migrate from output7/seed_layer.py lines 657-832
        logger.warning("AdsorptionStep not yet implemented")
        return {}

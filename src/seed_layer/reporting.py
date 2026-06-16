"""Report generation for seed layer screening."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def generate_summary_csv(
    output_dir: Path,
    stable_ids: List[str],
    matched_materials: List[Dict[str, Any]],
    adsorption_results: Dict[str, Any],
    diffusion_results: Dict[str, Any],
) -> pd.DataFrame:
    """Generate summary CSV from all step results.

    Args:
        output_dir: Run output directory
        stable_ids: List of stable material IDs
        matched_materials: Lattice matching results
        adsorption_results: Adsorption results by material ID
        diffusion_results: Diffusion results by material ID

    Returns:
        Summary DataFrame
    """
    rows = []

    for mat in matched_materials:
        mp_id = mat["material_id"]
        row = {
            "material_id": mp_id,
            "miller": str(mat.get("miller")),
            "mismatch_pct": mat.get("mismatch_pct"),
        }

        # Add adsorption data if available
        if mp_id in adsorption_results:
            ads = adsorption_results[mp_id]
            row["E_ads_eV"] = ads.get("E_ads_eV")
            row["coverage_ML"] = ads.get("coverage_ML")

        # Add diffusion data if available
        if mp_id in diffusion_results:
            diff = diffusion_results[mp_id]
            row["barrier_eV"] = diff.get("barrier_eV")
            row["diffusion_path"] = diff.get("path")

        rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate scores
    if not df.empty:
        df = _calculate_scores(df)

    # Save to CSV
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Summary saved to {csv_path}")

    return df


def _calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite scores.

    Args:
        df: DataFrame with screening results

    Returns:
        DataFrame with score columns added
    """
    # TODO: Implement scoring logic from output7/seed_layer.py
    df["score"] = 0.0
    return df

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
    interface_results: Dict[str, Any] = None,
    stability_results: Dict[str, Any] = None,
    max_mismatch: float = 8.0,
    w_lattice: float = 0.15,
    w_adsorption: float = 0.25,
    w_diffusion: float = 0.25,
    w_stability: float = 0.10,
    w_interface: float = 0.25,
) -> pd.DataFrame:
    """Generate summary CSV from all step results.

    Args:
        output_dir: Run output directory
        stable_ids: List of stable material IDs
        matched_materials: Lattice matching results
        adsorption_results: Adsorption results by material ID
        diffusion_results: Diffusion results by material ID
        interface_results: Interface energy results by material ID
        stability_results: Stability results by material ID (with e_above_hull)
        max_mismatch: Maximum lattice mismatch threshold (%)
        w_lattice: Weight for lattice mismatch score
        w_adsorption: Weight for adsorption energy score
        w_diffusion: Weight for diffusion barrier score
        w_stability: Weight for stability score
        w_interface: Weight for interface energy score

    Returns:
        Summary DataFrame
    """
    rows = []

    for mat in matched_materials:
        mp_id = mat["material_id"]

        # Skip unstable materials
        if mp_id not in stable_ids:
            continue

        row = {
            "material_id": mp_id,
            "formula": mat.get("formula", ""),
            "miller": str(mat.get("miller")),
            "mismatch_pct": mat.get("mismatch_pct"),
        }

        # Add adsorption data if available
        if mp_id in adsorption_results:
            ads_data = adsorption_results[mp_id]
            if ads_data.get('status') == 'success':
                energies = ads_data.get('adsorption_energies', [])
                if energies:
                    # Find best adsorption energy (closest to -0.5 eV)
                    valid_energies = [
                        e for e in energies
                        if e.get('status') == 'success' and e.get('E_ads_eV') is not None
                    ]
                    if valid_energies:
                        best = min(valid_energies, key=lambda x: abs(x['E_ads_eV'] + 0.5))
                        row["E_ads_eV"] = best['E_ads_eV']
                        row["coverage_ML"] = best['coverage_ML']

        # Add diffusion data if available
        if mp_id in diffusion_results:
            diff_data = diffusion_results[mp_id]
            if diff_data.get('status') == 'success':
                paths = diff_data.get('paths', [])
                if paths:
                    # Find lowest barrier
                    valid_paths = [
                        p for p in paths
                        if p.get('status') == 'success' and p.get('barrier_eV') is not None
                    ]
                    if valid_paths:
                        best_path = min(valid_paths, key=lambda x: x['barrier_eV'])
                        row["barrier_eV"] = best_path['barrier_eV']
                        row["diffusion_path"] = best_path['path']

        # Add interface energy data if available
        if interface_results and mp_id in interface_results:
            iface_data = interface_results[mp_id]
            if iface_data and iface_data.get("interface_energy_eV_per_A2") is not None:
                row["interface_energy_eV_per_A2"] = iface_data["interface_energy_eV_per_A2"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate scores
    if not df.empty:
        df = _calculate_scores(
            df, max_mismatch=max_mismatch,
            w_lattice=w_lattice, w_adsorption=w_adsorption,
            w_diffusion=w_diffusion, w_stability=w_stability,
            w_interface=w_interface,
            stability_results=stability_results,
        )

    # Save to CSV
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Summary saved to {csv_path}")

    return df


def _calculate_scores(
    df: pd.DataFrame,
    max_mismatch: float = 8.0,
    w_lattice: float = 0.15,
    w_adsorption: float = 0.25,
    w_diffusion: float = 0.25,
    w_stability: float = 0.10,
    w_interface: float = 0.25,
    stability_results: Dict[str, Any] = None,
) -> pd.DataFrame:
    """Calculate composite scores based on screening results.

    Scoring formula:
    - S_adsorption = exp(-((E_ads + 0.5)^2) / 0.08)  # Optimal around -0.5 eV
    - S_lattice = max(0, 1.0 - mismatch / max_mismatch)  # Linear penalty
    - S_diffusion = max(0, 1.0 - barrier / 1.0)  # Linear penalty, 1 eV reference
    - S_interface = exp(-gamma / 0.05)  # Lower interface energy is better
    - Final score = weighted average of available components

    Args:
        df: DataFrame with screening results
        max_mismatch: Maximum lattice mismatch threshold (%)
        w_lattice: Weight for lattice mismatch score
        w_adsorption: Weight for adsorption energy score
        w_diffusion: Weight for diffusion barrier score
        w_stability: Weight for stability score
        w_interface: Weight for interface energy score

    Returns:
        DataFrame with score columns added
    """
    import numpy as np

    if df.empty:
        return df

    # Initialize score columns
    df = df.copy()
    df['S_lattice'] = np.nan
    df['S_adsorption'] = np.nan
    df['S_diffusion'] = np.nan
    df['S_stability'] = np.nan
    df['S_interface'] = np.nan
    df['score'] = 0.0
    df['score_mode'] = 'unknown'

    # Calculate lattice mismatch score
    if 'mismatch_pct' in df.columns:
        valid_lattice = df['mismatch_pct'].notna()
        df.loc[valid_lattice, 'S_lattice'] = (
            1.0 - df.loc[valid_lattice, 'mismatch_pct'] / max_mismatch
        ).clip(lower=0)

    # Calculate adsorption energy score
    if 'E_ads_eV' in df.columns:
        valid_ads = df['E_ads_eV'].notna()
        if valid_ads.any():
            # Optimal adsorption energy is around -0.5 eV
            df.loc[valid_ads, 'S_adsorption'] = np.exp(
                -((df.loc[valid_ads, 'E_ads_eV'] + 0.5) ** 2) / 0.08
            )

    # Calculate diffusion barrier score
    if 'barrier_eV' in df.columns:
        valid_diff = df['barrier_eV'].notna()
        if valid_diff.any():
            df.loc[valid_diff, 'S_diffusion'] = (
                1.0 - df.loc[valid_diff, 'barrier_eV'] / 1.0
            ).clip(lower=0)

    # Calculate interface energy score
    if 'interface_energy_eV_per_A2' in df.columns:
        valid_iface = df['interface_energy_eV_per_A2'].notna()
        if valid_iface.any():
            # Lower interface energy is better
            # Normalize: S = exp(-gamma / gamma_ref), gamma_ref = 0.05 eV/Å²
            df.loc[valid_iface, 'S_interface'] = np.exp(
                -df.loc[valid_iface, 'interface_energy_eV_per_A2'] / 0.05
            ).clip(0, 1)

    # Calculate stability score from e_above_hull
    # S = exp(-e_above_hull / 0.05): e_above_hull=0 → S=1.0, 0.05 → S≈0.37
    if stability_results:
        for idx in df.index:
            mp_id = df.loc[idx, 'material_id']
            if mp_id in stability_results:
                eah = stability_results[mp_id].get('e_above_hull')
                if eah is not None:
                    df.loc[idx, 'S_stability'] = np.exp(-eah / 0.05)

    # Calculate composite score
    for idx in df.index:
        terms = []
        mode_parts = []

        # Adsorption score (always used if available)
        if pd.notna(df.loc[idx, 'S_adsorption']):
            terms.append((w_adsorption, df.loc[idx, 'S_adsorption']))
            mode_parts.append('ads')

        # Lattice score
        if pd.notna(df.loc[idx, 'S_lattice']):
            terms.append((w_lattice, df.loc[idx, 'S_lattice']))
            mode_parts.append('lat')

        # Diffusion score
        if pd.notna(df.loc[idx, 'S_diffusion']):
            terms.append((w_diffusion, df.loc[idx, 'S_diffusion']))
            mode_parts.append('neb')

        # Interface score
        if pd.notna(df.loc[idx, 'S_interface']):
            terms.append((w_interface, df.loc[idx, 'S_interface']))
            mode_parts.append('iface')

        # Stability score
        if pd.notna(df.loc[idx, 'S_stability']):
            terms.append((w_stability, df.loc[idx, 'S_stability']))
            mode_parts.append('stab')

        # Calculate weighted average
        if terms:
            weight_sum = sum(w for w, _ in terms)
            score = sum(w * s for w, s in terms) / weight_sum if weight_sum > 0 else 0.0
            df.loc[idx, 'score'] = round(score, 4)
            df.loc[idx, 'score_mode'] = '_'.join(mode_parts) if mode_parts else 'none'

    # Sort by score descending
    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    return df

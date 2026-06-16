"""Input/Output utilities for structure files and data."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from pymatgen.core import Structure


def save_structure_cif(structure: Structure, path: Union[str, Path]) -> Path:
    """Save structure to CIF format.

    Args:
        structure: pymatgen Structure
        path: Output file path

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(path))
    return path


def save_structure_xyz(structure: Structure, path: Union[str, Path]) -> Path:
    """Save structure to XYZ format.

    Args:
        structure: pymatgen Structure
        path: Output file path

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [str(len(structure))]
    lattice = structure.lattice
    lines.append(
        f"Lattice=\"{lattice.a} {lattice.b} {lattice.c} "
        f"{lattice.alpha} {lattice.beta} {lattice.gamma}\""
    )

    for site in structure:
        coords = site.coords
        lines.append(f"{site.specie} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}")

    path.write_text("\n".join(lines) + "\n")
    return path


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        path: Output file path

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle numpy types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=convert)

    return path


def save_lattice_params(structure: Structure, path: Union[str, Path]) -> Path:
    """Save lattice parameters to JSON.

    Args:
        structure: pymatgen Structure
        path: Output file path

    Returns:
        Path to saved file
    """
    lattice = structure.lattice
    data = {
        "a": round(lattice.a, 6),
        "b": round(lattice.b, 6),
        "c": round(lattice.c, 6),
        "alpha": round(lattice.alpha, 4),
        "beta": round(lattice.beta, 4),
        "gamma": round(lattice.gamma, 4),
        "volume": round(lattice.volume, 4),
    }
    return save_json(data, path)


def create_material_dir(base_dir: Path, material_id: str) -> Path:
    """Create directory for a material.

    Args:
        base_dir: Base output directory
        material_id: Materials Project ID (e.g., 'mp-12345')

    Returns:
        Path to material directory
    """
    mat_dir = base_dir / material_id
    mat_dir.mkdir(parents=True, exist_ok=True)
    return mat_dir


def create_miller_dir(material_dir: Path, miller: Tuple[int, int, int]) -> Path:
    """Create directory for a miller index.

    Args:
        material_dir: Material directory path
        miller: Miller index tuple (h, k, l)

    Returns:
        Path to miller directory
    """
    miller_str = f"({miller[0]}{miller[1]}{miller[2]})"
    miller_dir = material_dir / miller_str
    miller_dir.mkdir(parents=True, exist_ok=True)
    return miller_dir


def create_neb_path_dir(miller_dir: Path, path_name: str) -> Path:
    """Create directory for a NEB path.

    Args:
        miller_dir: Miller index directory
        path_name: NEB path name (e.g., 'neb_top_to_bridge')

    Returns:
        Path to NEB path directory
    """
    neb_dir = miller_dir / path_name
    neb_dir.mkdir(parents=True, exist_ok=True)
    return neb_dir

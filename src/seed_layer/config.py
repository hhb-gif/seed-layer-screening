"""Configuration loader for seed layer pipeline."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PipelineConfig:
    """Pipeline configuration loaded from YAML."""

    api: Dict[str, Any] = field(default_factory=dict)
    working_ion: str = "Li"
    ref_structure_id: str = None
    ref_miller: tuple = (1, 1, 0)
    screening: Dict[str, Any] = field(default_factory=dict)
    lattice: Dict[str, Any] = field(default_factory=dict)
    surface: Dict[str, Any] = field(default_factory=dict)
    calculator: Dict[str, Any] = field(default_factory=lambda: {"type": "chgnet", "kwargs": {}})
    relaxation: Dict[str, Any] = field(default_factory=dict)
    adsorption: Dict[str, Any] = field(default_factory=dict)
    diffusion: Dict[str, Any] = field(default_factory=dict)
    interface: Dict[str, Any] = field(default_factory=lambda: {
        "max_metal_layers": 5,
        "slab_thickness": 5.0,
        "vacuum": 15.0,
        "fmax": 0.05,
        "steps": 500,
    })
    scoring: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """Get config value by section and key."""
        section_data = getattr(self, section, {})
        if key is None:
            return section_data
        return section_data.get(key, default)


def _substitute_env_vars(value: Any) -> Any:
    """Replace ${VAR} patterns with environment variable values."""
    if isinstance(value, str):
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        result = value
        for var_name in matches:
            env_value = os.environ.get(var_name, "")
            result = result.replace(f"${{{var_name}}}", env_value)
        return result
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def load_config(config_path: str) -> PipelineConfig:
    """Load configuration from YAML file with environment variable substitution.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        PipelineConfig instance
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Substitute environment variables
    config = _substitute_env_vars(raw_config)

    ref_miller = tuple(config.get("ref_miller", [1, 1, 0]))

    return PipelineConfig(
        api=config.get("api", {}),
        working_ion=config.get("working_ion", "Li"),
        ref_structure_id=config.get("ref_structure_id", None),
        ref_miller=ref_miller,
        screening=config.get("screening", {}),
        lattice=config.get("lattice", {}),
        surface=config.get("surface", {}),
        calculator=config.get("calculator", {"type": "chgnet", "kwargs": {}}),
        relaxation=config.get("relaxation", {}),
        adsorption=config.get("adsorption", {}),
        diffusion=config.get("diffusion", {}),
        interface=config.get("interface", {
            "max_metal_layers": 5,
            "slab_thickness": 5.0,
            "vacuum": 15.0,
            "fmax": 0.05,
            "steps": 500,
        }),
        scoring=config.get("scoring", {}),
        output=config.get("output", {}),
    )

# Seed Layer Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the seed layer screening pipeline from a single 1800-line file into a modular architecture with YAML config, pluggable calculators, and structured output.

**Architecture:** Modular Python package with YAML configuration, abstract calculator interface for ML potentials, and per-material/per-miller output directories. Each screening step is an independent module.

**Tech Stack:** Python 3.12+, PyYAML, pymatgen, CHGNet, ASE, pandas, numpy

---

## File Structure

```
seed-layer-screening/
├── configs/
│   ├── default.yaml              # Main config with all parameters
│   ├── docker.yaml               # Docker environment overrides
│   └── hpc.yaml                  # HPC platform overrides
├── data/
│   ├── test_materials.txt        # Keep existing
│   └── sample_materials.txt      # Keep existing
├── output/                       # Runtime output (gitignored)
├── src/
│   ├── main.py                   # CLI entry point
│   └── seed_layer/
│       ├── __init__.py
│       ├── config.py             # YAML loader → PipelineConfig dataclass
│       ├── calculators/
│       │   ├── __init__.py       # Factory function
│       │   ├── base.py           # CalculatorBase ABC
│       │   └── chgnet.py         # CHGNet implementation
│       ├── pipeline.py           # Main orchestrator
│       ├── steps/
│       │   ├── __init__.py
│       │   ├── stability.py      # Step 2
│       │   ├── lattice.py        # Step 3
│       │   ├── adsorption.py     # Step 4
│       │   └── diffusion.py      # Step 5
│       ├── io.py                 # Structure file I/O
│       └── reporting.py          # Report generation
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_calculators.py
│   ├── test_io.py
│   └── test_reporting.py
├── .gitignore
└── README.md
```

---

## Task 1: Project Structure & Config Files

**Files:**
- Create: `configs/default.yaml`
- Create: `configs/docker.yaml`
- Create: `configs/hpc.yaml`
- Modify: `.gitignore`

- [ ] **Step 1: Create configs directory and default.yaml**

```yaml
# configs/default.yaml
# 锂金属电池种子层材料筛选 - 默认配置
# 所有能量单位 eV，长度单位 Å

# ── API ──
api:
  mp_api_key: "${MP_API_KEY}"

# ── 材料池初筛 ──
screening:
  energy_above_hull_max: 0.10
  n_elements: [2, 3]
  elements_to_exclude:
    - Pt
    - Pd
    - Rh
    - Ir
    - Ru
    - Os
    - Re
    - Tc
    - Hg
    - Pb
    - Cd
    - As
    - Tl
    - Be
    - U
    - Th
    - Pu
    - Np
    - Am
    - Cm
    - Ra
    - Po
    - In
    - Ga
    - Ge
    - Sb
    - Bi
    - La
    - Ce
    - Pr
    - Nd
    - Pm
    - Sm
    - Eu
    - Gd
    - Tb
    - Dy
    - Ho
    - Er
    - Tm
    - Yb
    - Lu
    - Sc
    - Y

# ── 晶格匹配 ──
lattice:
  li_miller: [1, 1, 0]
  max_mismatch: 8.0
  angle_tolerance: 180.0
  max_scale: 8

# ── 表面模型 ──
surface:
  vacuum: 15.0
  slab_thickness: 15.0
  default_miller: [0, 0, 1]
  supercell: [2, 2, 1]

# ── 势函数 ──
calculator:
  type: "chgnet"
  kwargs: {}

# ── 弛豫参数 ──
relaxation:
  fmax_bulk: 0.05
  fmax_slab: 0.10
  fmax_adsorb: 0.05
  steps_bulk: 500
  steps_slab: 200
  steps_adsorb: 250

# ── 吸附能 ──
adsorption:
  coverages: [0.25, 0.5, 0.75, 1.0]
  adsorption_min: -0.8
  adsorption_max: -0.2
  adsorbate_height: 1.8
  li_area_per_atom: 8.0

# ── NEB 扩散 ──
diffusion:
  neb_n_images: 7
  neb_fmax: 0.10
  neb_steps: 200
  neb_climb: true
  li_displacement_min: 0.2
  neb_top_n: 50

# ── 打分权重 ──
scoring:
  w_lattice: 0.3
  w_adsorption: 0.4
  w_diffusion: 0.3

# ── 输出控制 ──
output:
  save_structures: true
  save_trajectories: false
  save_neb_trajectories: true
```

- [ ] **Step 2: Create docker.yaml**

```yaml
# configs/docker.yaml
# Docker 环境配置 - 覆盖默认值

# 继承默认配置，只覆盖需要改的
api:
  mp_api_key: "${MP_API_KEY}"

output:
  save_structures: true
  save_trajectories: false
  save_neb_trajectories: true
```

- [ ] **Step 3: Create hpc.yaml**

```yaml
# configs/hpc.yaml
# 超算平台配置 - 覆盖默认值

# 可以调大并行数、改路径等
calculator:
  type: "chgnet"
  kwargs: {}
```

- [ ] **Step 4: Update .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Output (runtime generated)
output/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

- [ ] **Step 5: Commit**

```bash
git add configs/ .gitignore
git commit -m "feat: add YAML config files for all parameters"
```

---

## Task 2: Config Module

**Files:**
- Create: `src/seed_layer/__init__.py`
- Create: `src/seed_layer/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create package __init__.py**

```python
# src/seed_layer/__init__.py
"""Seed layer screening pipeline for lithium metal batteries."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Write failing test for config loading**

```python
# tests/test_config.py
import pytest
import yaml
import tempfile
from pathlib import Path
from seed_layer.config import load_config, PipelineConfig


def test_load_config_from_yaml():
    """Test loading config from YAML file."""
    config_data = {
        "api": {"mp_api_key": "test-key"},
        "screening": {
            "energy_above_hull_max": 0.10,
            "n_elements": [2, 3],
            "elements_to_exclude": ["Pt", "Pd"],
        },
        "calculator": {"type": "chgnet", "kwargs": {}},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    config = load_config(config_path)

    assert config.api["mp_api_key"] == "test-key"
    assert config.screening["energy_above_hull_max"] == 0.10
    assert config.calculator["type"] == "chgnet"


def test_env_variable_substitution():
    """Test that ${VAR} is replaced with environment variable."""
    import os
    os.environ["TEST_API_KEY"] = "my-secret-key"

    config_data = {
        "api": {"mp_api_key": "${TEST_API_KEY}"},
        "calculator": {"type": "chgnet", "kwargs": {}},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    config = load_config(config_path)
    assert config.api["mp_api_key"] == "my-secret-key"

    del os.environ["TEST_API_KEY"]


def test_pipeline_config_dataclass():
    """Test PipelineConfig creation."""
    config = PipelineConfig(
        api={"mp_api_key": "test"},
        screening={"energy_above_hull_max": 0.1},
        calculator={"type": "chgnet", "kwargs": {}},
    )

    assert config.api["mp_api_key"] == "test"
    assert config.calculator["type"] == "chgnet"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd "E:/Claude code project/seed-layer-screening" && python -m pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'seed_layer'"

- [ ] **Step 4: Implement config module**

```python
# src/seed_layer/config.py
"""Configuration loader for seed layer pipeline."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class PipelineConfig:
    """Pipeline configuration loaded from YAML."""

    api: Dict[str, Any] = field(default_factory=dict)
    screening: Dict[str, Any] = field(default_factory=dict)
    lattice: Dict[str, Any] = field(default_factory=dict)
    surface: Dict[str, Any] = field(default_factory=dict)
    calculator: Dict[str, Any] = field(default_factory=lambda: {"type": "chgnet", "kwargs": {}})
    relaxation: Dict[str, Any] = field(default_factory=dict)
    adsorption: Dict[str, Any] = field(default_factory=dict)
    diffusion: Dict[str, Any] = field(default_factory=dict)
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

    return PipelineConfig(
        api=config.get("api", {}),
        screening=config.get("screening", {}),
        lattice=config.get("lattice", {}),
        surface=config.get("surface", {}),
        calculator=config.get("calculator", {"type": "chgnet", "kwargs": {}}),
        relaxation=config.get("relaxation", {}),
        adsorption=config.get("adsorption", {}),
        diffusion=config.get("diffusion", {}),
        scoring=config.get("scoring", {}),
        output=config.get("output", {}),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd "E:/Claude code project/seed-layer-screening" && python -m pytest tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/seed_layer/__init__.py src/seed_layer/config.py tests/test_config.py
git commit -m "feat: add config module with YAML loader"
```

---

## Task 3: Calculator Interface

**Files:**
- Create: `src/seed_layer/calculators/__init__.py`
- Create: `src/seed_layer/calculators/base.py`
- Create: `src/seed_layer/calculators/chgnet.py`
- Create: `tests/test_calculators.py`

- [ ] **Step 1: Write failing test for calculator base**

```python
# tests/test_calculators.py
import pytest
import numpy as np
from seed_layer.calculators.base import CalculatorBase


def test_calculator_base_is_abstract():
    """Test that CalculatorBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        CalculatorBase()


def test_create_calculator_factory():
    """Test factory function creates correct calculator type."""
    from seed_layer.calculators import create_calculator

    # This will fail until we implement chgnet calculator
    config = {"type": "chgnet", "kwargs": {}}
    calc = create_calculator(config)
    assert isinstance(calc, CalculatorBase)
```

- [ ] **Step 2: Implement CalculatorBase**

```python
# src/seed_layer/calculators/base.py
"""Abstract base class for ML potential calculators."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from pymatgen.core import Structure


class CalculatorBase(ABC):
    """Abstract base class for machine learning potential calculators.

    All calculator implementations must inherit from this class
    and implement the abstract methods.
    """

    @abstractmethod
    def relax(
        self,
        structure: Structure,
        fmax: float = 0.05,
        steps: int = 500,
        relax_cell: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Relax a structure using the ML potential.

        Args:
            structure: pymatgen Structure to relax
            fmax: Force convergence criterion in eV/Å
            steps: Maximum number of relaxation steps
            relax_cell: Whether to relax cell parameters
            verbose: Whether to print relaxation progress

        Returns:
            Dictionary with keys:
                - final_structure: Relaxed pymatgen Structure
                - energy: Total energy in eV
                - trajectory: Optional list of energies during relaxation
        """
        ...

    @abstractmethod
    def get_energy(self, structure: Structure) -> float:
        """Get total energy of a structure.

        Args:
            structure: pymatgen Structure

        Returns:
            Total energy in eV
        """
        ...

    @abstractmethod
    def get_forces(self, structure: Structure) -> np.ndarray:
        """Get forces on atoms in a structure.

        Args:
            structure: pymatgen Structure

        Returns:
            Forces array of shape (N, 3) in eV/Å
        """
        ...
```

- [ ] **Step 3: Implement CHGNet calculator**

```python
# src/seed_layer/calculators/chgnet.py
"""CHGNet calculator implementation."""

from typing import Any, Dict

import numpy as np
from pymatgen.core import Structure

from .base import CalculatorBase


class CHGNetCalculator(CalculatorBase):
    """CHGNet machine learning potential calculator."""

    def __init__(self, **kwargs):
        """Initialize CHGNet calculator.

        Args:
            **kwargs: Additional arguments passed to CHGNet (currently unused)
        """
        from chgnet.model import StructOptimizer

        self.relaxer = StructOptimizer()
        self._calculator = None

    def _get_ase_calculator(self):
        """Get ASE calculator from CHGNet (lazy initialization)."""
        if self._calculator is None:
            from chgnet.model.model import CHGNet
            model = CHGNet.load()
            self._calculator = model
        return self._calculator

    def relax(
        self,
        structure: Structure,
        fmax: float = 0.05,
        steps: int = 500,
        relax_cell: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Relax structure using CHGNet.

        Args:
            structure: pymatgen Structure to relax
            fmax: Force convergence criterion in eV/Å
            steps: Maximum number of relaxation steps
            relax_cell: Whether to relax cell parameters
            verbose: Whether to print relaxation progress

        Returns:
            Dictionary with final_structure, energy, and trajectory
        """
        result = self.relaxer.relax(
            structure,
            fmax=fmax,
            steps=steps,
            relax_cell=relax_cell,
            verbose=verbose,
        )

        final_structure = result["final_structure"]
        trajectory = result.get("trajectory")
        energy = trajectory.energies[-1] if trajectory else None

        return {
            "final_structure": final_structure,
            "energy": energy,
            "trajectory": trajectory,
        }

    def get_energy(self, structure: Structure) -> float:
        """Get total energy using CHGNet.

        Args:
            structure: pymatgen Structure

        Returns:
            Total energy in eV
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        atoms.calc = self._get_ase_calculator()
        return atoms.get_potential_energy()

    def get_forces(self, structure: Structure) -> np.ndarray:
        """Get forces using CHGNet.

        Args:
            structure: pymatgen Structure

        Returns:
            Forces array of shape (N, 3) in eV/Å
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        atoms.calc = self._get_ase_calculator()
        return atoms.get_forces()
```

- [ ] **Step 4: Implement factory function**

```python
# src/seed_layer/calculators/__init__.py
"""Calculator factory and base class."""

from typing import Any, Dict

from .base import CalculatorBase


def create_calculator(config: Dict[str, Any]) -> CalculatorBase:
    """Create calculator instance from config.

    Args:
        config: Calculator configuration with 'type' and 'kwargs'

    Returns:
        CalculatorBase implementation

    Raises:
        ValueError: If calculator type is unknown
    """
    calc_type = config.get("type", "chgnet")
    kwargs = config.get("kwargs", {})

    if calc_type == "chgnet":
        from .chgnet import CHGNetCalculator
        return CHGNetCalculator(**kwargs)
    else:
        raise ValueError(f"Unknown calculator type: {calc_type}")


__all__ = ["CalculatorBase", "create_calculator"]
```

- [ ] **Step 5: Run tests**

Run: `cd "E:/Claude code project/seed-layer-screening" && python -m pytest tests/test_calculators.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/seed_layer/calculators/ tests/test_calculators.py
git commit -m "feat: add calculator interface with CHGNet implementation"
```

---

## Task 4: IO Module (Structure Saving)

**Files:**
- Create: `src/seed_layer/io.py`
- Create: `tests/test_io.py`

- [ ] **Step 1: Write failing test for IO module**

```python
# tests/test_io.py
import pytest
import json
import tempfile
from pathlib import Path
from seed_layer.io import (
    save_structure_cif,
    save_structure_xyz,
    save_json,
    create_material_dir,
    create_miller_dir,
)


def test_save_json():
    """Test saving JSON data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {"energy": -10.5, "passed": True}
        path = Path(tmpdir) / "test.json"
        save_json(data, path)

        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["energy"] == -10.5
        assert loaded["passed"] is True


def test_create_material_dir():
    """Test creating material directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        mat_dir = create_material_dir(base, "mp-12345")

        assert mat_dir.exists()
        assert mat_dir.name == "mp-12345"


def test_create_miller_dir():
    """Test creating miller index directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        miller_dir = create_miller_dir(base, (1, 1, 0))

        assert miller_dir.exists()
        assert miller_dir.name == "(110)"
```

- [ ] **Step 2: Implement IO module**

```python
# src/seed_layer/io.py
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
```

- [ ] **Step 3: Run tests**

Run: `cd "E:/Claude code project/seed-layer-screening" && python -m pytest tests/test_io.py -v`
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add src/seed_layer/io.py tests/test_io.py
git commit -m "feat: add IO module for structure and data saving"
```

---

## Task 5: Pipeline Orchestrator

**Files:**
- Create: `src/seed_layer/pipeline.py`

- [ ] **Step 1: Implement pipeline orchestrator**

```python
# src/seed_layer/pipeline.py
"""Main pipeline orchestrator for seed layer screening."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

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
                    "screening": self.config.screening,
                    "lattice": self.config.lattice,
                    "surface": self.config.surface,
                    "calculator": self.config.calculator,
                    "relaxation": self.config.relaxation,
                    "adsorption": self.config.adsorption,
                    "diffusion": self.config.diffusion,
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
        from .steps.adsorption import AdsorptionStep
        from .steps.diffusion import DiffusionStep

        # Step 1: Fetch materials
        logger.info("Step 1: Fetching materials pool...")
        materials = self._fetch_materials(materials_file)
        logger.info(f"Found {len(materials)} candidate materials")

        # Step 2: Stability screening
        logger.info("Step 2: Electrochemical stability screening...")
        stability_step = StabilityStep(self.config, self.calculator, self.run_dir)
        stable_materials = stability_step.run(materials)
        logger.info(f"{len(stable_materials)} materials passed stability screening")

        # Step 3: Lattice matching
        logger.info("Step 3: Lattice mismatch calculation...")
        lattice_step = LatticeStep(self.config, self.calculator, self.run_dir)
        matched_materials = lattice_step.run(stable_materials)
        logger.info(f"{len(matched_materials)} materials passed lattice matching")

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

        # Generate summary
        logger.info("Generating summary report...")
        self._generate_summary(stable_materials, matched_materials, adsorption_results, diffusion_results)

        logger.info("Pipeline complete!")

    def _fetch_materials(self, materials_file: Optional[str] = None) -> List[dict]:
        """Fetch materials from MP API or file.

        Args:
            materials_file: Path to materials list file

        Returns:
            List of material dictionaries
        """
        # TODO: Implement materials fetching
        # For now, return placeholder
        return []

    def _generate_summary(
        self,
        stable: List[str],
        matched: List[str],
        adsorption: dict,
        diffusion: dict,
    ):
        """Generate summary CSV.

        Args:
            stable: List of stable material IDs
            matched: List of matched material IDs
            adsorption: Adsorption results by material ID
            diffusion: Diffusion results by material ID
        """
        # TODO: Implement summary generation
        pass
```

- [ ] **Step 2: Commit**

```bash
git add src/seed_layer/pipeline.py
git commit -m "feat: add pipeline orchestrator skeleton"
```

---

## Task 6: Screening Steps

**Files:**
- Create: `src/seed_layer/steps/__init__.py`
- Create: `src/seed_layer/steps/base.py`
- Create: `src/seed_layer/steps/stability.py`
- Create: `src/seed_layer/steps/lattice.py`
- Create: `src/seed_layer/steps/adsorption.py`
- Create: `src/seed_layer/steps/diffusion.py`

- [ ] **Step 1: Create step base class**

```python
# src/seed_layer/steps/base.py
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

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the screening step.

        Args:
            input_data: Input from previous step

        Returns:
            Output data for next step
        """
        ...
```

- [ ] **Step 2: Create steps __init__.py**

```python
# src/seed_layer/steps/__init__.py
"""Screening steps for seed layer pipeline."""

from .base import BaseStep
from .stability import StabilityStep
from .lattice import LatticeStep
from .adsorption import AdsorptionStep
from .diffusion import DiffusionStep

__all__ = [
    "BaseStep",
    "StabilityStep",
    "LatticeStep",
    "AdsorptionStep",
    "DiffusionStep",
]
```

- [ ] **Step 3: Implement StabilityStep (from output7/seed_layer.py lines 407-517)**

```python
# src/seed_layer/steps/stability.py
"""Step 2: Electrochemical stability screening."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram

from .base import BaseStep
from ..io import save_json, save_structure_cif, create_material_dir

logger = logging.getLogger(__name__)


class StabilityStep(BaseStep):
    """Screen materials for electrochemical stability against Li."""

    def run(self, materials: List[Dict[str, Any]]) -> List[str]:
        """Run stability screening.

        Args:
            materials: List of material dicts with 'material_id' and 'formula'

        Returns:
            List of stable material IDs
        """
        stable_ids = []
        api_key = self.config.api.get("mp_api_key", "")

        with MPRester(api_key) as mpr:
            # Get Li reference energy
            li_entries = mpr.get_entries_in_chemsys(["Li"])
            li_entry = sorted(li_entries, key=lambda e: e.energy_per_atom)[0]
            e_li = li_entry.energy_per_atom

            for mat in materials:
                mp_id = mat["material_id"]
                formula = mat["formula"]

                try:
                    passed, details = self._check_stability(mp_id, mpr, e_li)

                    # Save results
                    mat_dir = create_material_dir(self.output_dir, mp_id)
                    save_json(
                        {
                            "material_id": mp_id,
                            "formula": formula,
                            "passed": passed,
                            "min_dE": details.get("min_dE"),
                            "details": details.get("details"),
                        },
                        mat_dir / "stability.json",
                    )

                    if passed:
                        stable_ids.append(mp_id)
                        logger.info(f"  {mp_id} ({formula}): PASSED")
                    else:
                        logger.info(f"  {mp_id} ({formula}): FAILED - {details.get('details')}")

                except Exception as e:
                    logger.error(f"  {mp_id}: Error - {e}")

        return stable_ids

    def _check_stability(self, mp_id: str, mpr: MPRester, e_li: float) -> Tuple[bool, Dict]:
        """Check if material is stable against Li.

        Args:
            mp_id: Materials Project ID
            mpr: MPRester instance
            e_li: Li reference energy per atom

        Returns:
            Tuple of (passed, details_dict)
        """
        # Get material entry
        docs = mpr.summary.search(
            material_ids=[mp_id],
            fields=["material_id", "elements", "formula_pretty"],
        )
        if not docs:
            return False, {"details": "Not found in MP"}

        elements = [el.symbol for el in docs[0].elements]
        formula = docs[0].formula_pretty
        chemsys = "-".join(sorted(elements))

        entries = mpr.get_entries_in_chemsys(chemsys)
        target = [e for e in entries if e.entry_id == mp_id]
        if not target:
            target = [e for e in entries if e.composition.reduced_formula == formula]
        if not target:
            return False, {"details": "Entry not found"}

        mat_entry = sorted(target, key=lambda e: e.energy_per_atom)[0]
        e_mat = mat_entry.energy_per_atom
        comp_mat = mat_entry.composition

        # Build phase diagram with Li
        all_elements = list(set(elements + ["Li"]))
        full_chemsys = "-".join(sorted(all_elements))
        all_entries = mpr.get_entries_in_chemsys(full_chemsys)
        pd_phase = PhaseDiagram(all_entries)

        # Check if material is stable
        stable_ids = {e.entry_id for e in pd_phase.stable_entries}
        if mat_entry.entry_id not in stable_ids:
            return False, {"details": "Material itself unstable"}

        # Scan composition line to Li
        frac = comp_mat.fractional_composition
        min_dE = 0.0
        n_steps = 100

        for j in range(1, n_steps):
            t = j / n_steps
            comp_dict = {el.symbol: (1 - t) * f for el, f in frac.items()}
            comp_dict["Li"] = comp_dict.get("Li", 0) + t
            comp = Composition(comp_dict)

            try:
                e_hull = pd_phase.get_hull_energy(comp)
            except Exception:
                continue

            e_react = (1 - t) * e_mat + t * e_li
            dE = e_hull - e_react

            if dE < min_dE:
                min_dE = dE

        tolerance = 0.05  # eV/atom
        if min_dE < -tolerance:
            return False, {"min_dE": min_dE, "details": f"Reacts with Li: dE={min_dE:.3f}"}

        return True, {"min_dE": min_dE, "details": f"Stable: dE={min_dE:.3f}"}
```

- [ ] **Step 4: Implement LatticeStep (from output7/seed_layer.py lines 521-653)**

```python
# src/seed_layer/steps/lattice.py
"""Step 3: Lattice mismatch calculation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator, generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .base import BaseStep
from ..io import save_json, save_structure_cif, save_lattice_params, create_material_dir, create_miller_dir

logger = logging.getLogger(__name__)


class LatticeStep(BaseStep):
    """Calculate lattice mismatch between materials and Li."""

    def run(self, stable_ids: List[str]) -> List[Dict[str, Any]]:
        """Run lattice matching for stable materials.

        Args:
            stable_ids: List of stable material IDs

        Returns:
            List of dicts with material_id, best_miller, mismatch_pct
        """
        results = []
        api_key = self.config.api.get("mp_api_key", "")
        li_miller = tuple(self.config.lattice.get("li_miller", [1, 1, 0]))

        with MPRester(api_key) as mpr:
            # Build Li reference slab
            li_entries = mpr.get_entries_in_chemsys(["Li"])
            li_struct = sorted(li_entries, key=lambda e: e.energy_per_atom)[0].structure

            slabgen = SlabGenerator(
                li_struct,
                li_miller,
                self.config.surface.get("slab_thickness", 15.0),
                self.config.surface.get("vacuum", 15.0),
                center_slab=True,
                primitive=True,
            )
            li_slab = slabgen.get_slabs()[0]

            for mp_id in stable_ids:
                try:
                    best = self._calc_lattice_match(mp_id, li_slab, mpr)
                    if best:
                        results.append(best)
                        logger.info(f"  {mp_id}: best mismatch = {best['mismatch_pct']:.2f}%")
                except Exception as e:
                    logger.error(f"  {mp_id}: Error - {e}")

        return results

    def _calc_lattice_match(self, mp_id: str, li_slab, mpr) -> Optional[Dict]:
        """Calculate lattice mismatch for a single material.

        Args:
            mp_id: Material ID
            li_slab: Li reference slab
            mpr: MPRester instance

        Returns:
            Dict with best match info, or None if failed
        """
        structure = mpr.get_structure_by_material_id(mp_id)
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

        # Relax bulk
        result = self.calculator.relax(
            structure,
            fmax=self.config.relaxation.get("fmax_bulk", 0.05),
            steps=self.config.relaxation.get("steps_bulk", 500),
            relax_cell=True,
            verbose=False,
        )
        relaxed_structure = result["final_structure"]

        # Save bulk structures
        mat_dir = create_material_dir(self.output_dir, mp_id)
        save_structure_cif(structure, mat_dir / "bulk.cif")
        save_structure_cif(relaxed_structure, mat_dir / "bulk_relaxed.cif")
        save_lattice_params(relaxed_structure, mat_dir / "lattice_params.json")

        # Generate all low-index slabs
        try:
            slabs = generate_all_slabs(
                relaxed_structure,
                max_index=1,
                min_slab_size=self.config.surface.get("slab_thickness", 15.0),
                min_vacuum_size=self.config.surface.get("vacuum", 15.0),
                center_slab=True,
                primitive=True,
            )
        except Exception as e:
            logger.warning(f"  {mp_id}: Slab generation failed - {e}")
            return None

        best_mismatch = float("inf")
        best_info = None

        for slab in slabs:
            miller = slab.miller_index
            mismatch, info = self._compute_mismatch(li_slab.lattice, slab.lattice)

            if mismatch < best_mismatch:
                best_mismatch = mismatch
                best_info = {
                    "material_id": mp_id,
                    "miller": miller,
                    "mismatch_pct": round(mismatch, 3),
                    "mismatch_a_pct": round(info.get("mismatch_a", 0), 3),
                    "mismatch_b_pct": round(info.get("mismatch_b", 0), 3),
                    "angle_diff_deg": round(info.get("angle_diff", 0), 2),
                }

        # Save mismatch data
        if best_info:
            miller_dir = create_miller_dir(mat_dir, best_info["miller"])
            save_json(best_info, miller_dir / "mismatch.json")

        return best_info

    def _compute_mismatch(self, li_lattice, film_lattice) -> Tuple[float, Optional[Dict]]:
        """Compute minimum lattice mismatch.

        Args:
            li_lattice: Li slab lattice
            film_lattice: Film slab lattice

        Returns:
            Tuple of (mismatch_pct, info_dict)
        """
        max_scale = self.config.lattice.get("max_scale", 8)
        angle_tolerance = self.config.lattice.get("angle_tolerance", 180.0)

        a_f = np.array(film_lattice.matrix[0][:2])
        b_f = np.array(film_lattice.matrix[1][:2])
        a_li = np.array(li_lattice.matrix[0][:2])
        b_li = np.array(li_lattice.matrix[1][:2])

        len_a_f, len_b_f = np.linalg.norm(a_f), np.linalg.norm(b_f)
        len_a_li, len_b_li = np.linalg.norm(a_li), np.linalg.norm(b_li)

        def get_angle(v1, v2):
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

        angle_diff = abs(get_angle(a_f, b_f) - get_angle(a_li, b_li))

        best = float("inf")
        best_info = None

        for n1 in range(1, max_scale + 1):
            for m1 in range(1, max_scale + 1):
                ma = abs(n1 * len_a_f - m1 * len_a_li) / (m1 * len_a_li) * 100
                if ma > best:
                    continue

                for n2 in range(1, max_scale + 1):
                    for m2 in range(1, max_scale + 1):
                        mb = abs(n2 * len_b_f - m2 * len_b_li) / (m2 * len_b_li) * 100
                        max_mm = max(ma, mb)

                        if max_mm < best and angle_diff <= angle_tolerance:
                            best = max_mm
                            best_info = {
                                "mismatch_a": ma,
                                "mismatch_b": mb,
                                "film_scale_a": n1,
                                "film_scale_b": n2,
                                "li_scale_a": m1,
                                "li_scale_b": m2,
                                "angle_diff": angle_diff,
                            }

        if best == float("inf"):
            return 999.0, None

        return best, best_info
```

- [ ] **Step 5: Create placeholder AdsorptionStep and DiffusionStep**

```python
# src/seed_layer/steps/adsorption.py
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
```

```python
# src/seed_layer/steps/diffusion.py
"""Step 5: NEB diffusion barrier calculation."""

import logging
from typing import Any, Dict

from .base import BaseStep

logger = logging.getLogger(__name__)


class DiffusionStep(BaseStep):
    """Calculate Li diffusion barriers using CI-NEB."""

    def run(self, adsorption_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run diffusion barrier calculation.

        Args:
            adsorption_results: Adsorption results by material ID

        Returns:
            Dict mapping material_id to diffusion results
        """
        # TODO: Migrate from output7/seed_layer.py lines 836-1081
        logger.warning("DiffusionStep not yet implemented")
        return {}
```

- [ ] **Step 6: Commit**

```bash
git add src/seed_layer/steps/
git commit -m "feat: add screening step classes (stability, lattice implemented; adsorption, diffusion placeholder)"
```

---

## Task 7: Reporting Module

**Files:**
- Create: `src/seed_layer/reporting.py`

- [ ] **Step 1: Implement reporting module**

```python
# src/seed_layer/reporting.py
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
```

- [ ] **Step 2: Commit**

```bash
git add src/seed_layer/reporting.py
git commit -m "feat: add reporting module for summary generation"
```

---

## Task 8: CLI Entry Point

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Implement CLI entry point**

```python
# src/main.py
"""CLI entry point for seed layer screening pipeline."""

import argparse
import sys
from pathlib import Path

from seed_layer.config import load_config
from seed_layer.pipeline import SeedLayerPipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed layer material screening for lithium metal batteries"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for output directory naming",
    )
    parser.add_argument(
        "--materials",
        type=str,
        default=None,
        help="Path to materials list file",
    )
    parser.add_argument(
        "--skip-neb",
        action="store_true",
        help="Skip NEB diffusion calculation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted run (not yet implemented)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode (offline, no API needed)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Setup output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Run pipeline
    if args.demo:
        print("Demo mode not yet implemented")
        sys.exit(0)

    pipeline = SeedLayerPipeline(config, output_dir, tag=args.tag)
    pipeline.run(materials_file=args.materials, skip_neb=args.skip_neb)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: add CLI entry point"
```

---

## Task 9: Testing & Integration

- [ ] **Step 1: Create tests __init__.py**

```python
# tests/__init__.py
"""Test package for seed layer pipeline."""
```

- [ ] **Step 2: Run all tests**

Run: `cd "E:/Claude code project/seed-layer-screening" && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Test CLI help**

Run: `cd "E:/Claude code project/seed-layer-screening" && python src/main.py --help`
Expected: Show help message with all arguments

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py
git commit -m "test: add test infrastructure"
```

---

## Task 10: Cleanup & Documentation

- [ ] **Step 1: Remove old code**

```bash
rm -rf src/seed_layer_pipeline.py src/seed_layer_pipeline_improved.py output7/
```

- [ ] **Step 2: Update README.md**

Update README.md to reflect new project structure and usage.

- [ ] **Step 3: Create docs/parameters_cn.md**

Create Chinese documentation for all parameters.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "docs: update README and add Chinese parameter documentation"
```

---

## Summary

| Task | Description | Status |
|------|-------------|--------|
| 1 | Project structure & config files | - [ ] |
| 2 | Config module (YAML loader) | - [ ] |
| 3 | Calculator interface | - [ ] |
| 4 | IO module | - [ ] |
| 5 | Pipeline orchestrator | - [ ] |
| 6 | Screening steps | - [ ] |
| 7 | Reporting module | - [ ] |
| 8 | CLI entry point | - [ ] |
| 9 | Testing & integration | - [ ] |
| 10 | Cleanup & documentation | - [ ] |

**Note:** Tasks 6 (AdsorptionStep, DiffusionStep) contain placeholders that need migration from output7/seed_layer.py. These are marked with TODO comments.

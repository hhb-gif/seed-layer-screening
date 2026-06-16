# Seed Layer Pipeline Refactor Design

**Date**: 2026-06-16
**Status**: Draft
**Author**: Claude Code + User

## 1. Overview

### Problem

Current pipeline has several issues:
- Single 1800-line file with hardcoded parameters
- No config file support (parameters in code)
- Output structure is flat, no intermediate data saved
- ML potential (CHGNet) is hardcoded, hard to switch
- No room for adding new screening steps

### Solution

Refactor into modular architecture:
- YAML config files for all parameters
- Pluggable calculator interface (CHGNet, MACE, etc.)
- Hierarchical output by material ID / miller index
- Extensible Step system for future screening criteria

### Success Criteria

1. All parameters externalized to YAML (zero hardcoded values in pipeline code)
2. Switch calculator by changing config only
3. All intermediate structures saved (bulk, slab, adsorption sites, NEB paths)
4. New screening steps can be added without modifying existing code

---

## 2. Project Structure

```
seed-layer-screening/
├── configs/                      # Configuration files
│   ├── default.yaml              # Default parameters
│   ├── docker.yaml               # Docker environment
│   └── hpc.yaml                  # HPC platform
├── data/                         # Input data (not in version control for real runs)
│   ├── test_materials.txt
│   └── sample_materials.txt
├── output/                       # Runtime output (auto-created, gitignored)
│   └── 20260616_143052_test/     # timestamp_tag format
│       ├── run_config.yaml       # Parameter snapshot
│       ├── summary.csv           # Final ranking
│       ├── logs/
│       │   └── run.log
│       └── mp-XXXXXXX/           # Per-material output
│           └── (hkl)/            # Per-miller-index output
├── src/
│   └── seed_layer/
│       ├── __init__.py
│       ├── config.py             # YAML → PipelineConfig loader
│       ├── calculators/          # Pluggable ML potentials
│       │   ├── __init__.py       # Factory function
│       │   ├── base.py           # CalculatorBase ABC
│       │   └── chgnet.py         # CHGNet implementation
│       ├── pipeline.py           # Main orchestrator
│       ├── steps/                # Screening steps (extensible)
│       │   ├── __init__.py
│       │   ├── stability.py      # Step 2: Electrochemical stability
│       │   ├── lattice.py        # Step 3: Lattice mismatch
│       │   ├── adsorption.py     # Step 4: Adsorption energy
│       │   └── diffusion.py      # Step 5: NEB diffusion barrier
│       ├── io.py                 # Structure file I/O (CIF, XYZ, JSON)
│       └── reporting.py          # Report generation
├── main.py                       # CLI entry point
├── docs/
│   ├── parameters_cn.md          # Chinese parameter documentation
│   └── superpowers/specs/        # Design docs
├── .gitignore
└── README.md
```

---

## 3. Configuration

### Design Principles

- **Zero hardcoded parameters** in pipeline/steps code
- All values read from YAML config
- Environment variable support: `${MP_API_KEY}` replaced at runtime
- CLI flags override config file (for `--resume`, `--skip-neb`, etc.)

### Config File Example

```yaml
# configs/default.yaml — Chinese comments for user understanding

api:
  mp_api_key: "${MP_API_KEY}"

screening:
  energy_above_hull_max: 0.10   # eV/atom
  n_elements: [2, 3]
  elements_to_exclude: [Pt, Pd, Rh, Ir, Ru, Os, ...]

lattice:
  max_mismatch: 8.0             # %

calculator:
  type: "chgnet"                # Options: chgnet, mace, nequip, dpa
  kwargs: {}                    # Calculator-specific parameters

relaxation:
  fmax_bulk: 0.05               # eV/Å
  fmax_slab: 0.10
  fmax_adsorb: 0.05
  steps_bulk: 500
  steps_slab: 200
  steps_adsorb: 250

scoring:
  w_lattice: 0.3
  w_adsorption: 0.4
  w_diffusion: 0.3
```

### Code Comments

- **Config files**: Chinese comments (user requirement)
- **Source code**: English comments (advisor requirement)
- **Separate doc**: `docs/parameters_cn.md` explains each parameter in Chinese

---

## 4. Calculator Interface

### Abstract Base Class

```python
# src/seed_layer/calculators/base.py

class CalculatorBase(ABC):
    @abstractmethod
    def relax(self, structure: Structure, fmax: float, steps: int,
              relax_cell: bool = True, verbose: bool = False) -> dict:
        """Relax structure. Returns {"final_structure", "energy", "trajectory"}."""
        ...
    
    @abstractmethod
    def get_energy(self, structure: Structure) -> float:
        """Get total energy in eV."""
        ...
    
    @abstractmethod
    def get_forces(self, structure: Structure) -> np.ndarray:
        """Get forces in eV/Å, shape (N, 3)."""
        ...
```

### Factory Pattern

```python
# src/seed_layer/calculators/__init__.py

def create_calculator(config: dict) -> CalculatorBase:
    calc_type = config["type"]
    kwargs = config.get("kwargs", {})
    
    if calc_type == "chgnet":
        from .chgnet import CHGNetCalculator
        return CHGNetCalculator(**kwargs)
    elif calc_type == "mace":
        from .mace import MACCalculator
        return MACCalculator(**kwargs)
    else:
        raise ValueError(f"Unknown calculator: {calc_type}")
```

### Key Principle

**Pipeline and steps code NEVER import specific calculator implementations.**

All calls go through the abstract interface:
```python
# ✅ Correct
result = self.calculator.relax(structure, fmax=0.05, steps=500)

# ❌ Wrong - never in pipeline/steps
from chgnet.model import StructOptimizer
```

---

## 5. Output Structure

### Directory Layout

```
output/20260616_143052_test/
├── run_config.yaml              # Full parameter snapshot
├── summary.csv                  # Final ranking (all materials)
├── logs/
│   └── run.log                  # Execution log
│
├── mp-1052023/                  # ── Per Material ──
│   ├── bulk.cif                 # Original bulk structure (from MP)
│   ├── bulk_relaxed.cif         # Relaxed bulk
│   ├── lattice_params.json      # {a, b, c, alpha, beta, gamma}
│   ├── stability.json           # {passed, min_dE, details}
│   │
│   └── (110)/                   # ── Per Miller Index ──
│       ├── slab.cif             # Original slab
│       ├── slab_relaxed.cif     # Relaxed slab
│       ├── mismatch.json        # Lattice mismatch data
│       │
│       ├── Li_top_0_opt.cif     # Optimized adsorption structures
│       ├── Li_bridge_0_opt.cif
│       ├── Li_hollow_0_opt.cif
│       ├── adsorption.json      # Adsorption energies per coverage
│       │
│       ├── neb_top_to_bridge/   # ── Per NEB Path ──
│       │   ├── init.xyz         # Initial state
│       │   ├── final.xyz        # Final state
│       │   ├── barrier.json     # {path, barrier_eV, li_displacement}
│       │   └── neb.traj         # Optional: full NEB trajectory
│       │
│       └── neb_bridge_to_hollow/
│           └── ...
│
├── mp-10736/
│   └── ...
```

### File Formats

| Data | Format | Reason |
|------|--------|--------|
| Structures | CIF | Standard crystallographic format, widely supported |
| NEB frames | XYZ | Lightweight, ASE compatible |
| Numerical data | JSON | Human-readable, easy to parse |
| Summary table | CSV | Easy to open in Excel/pandas |

### Output Control (in config)

```yaml
output:
  save_structures: true         # Save CIF/XYZ files
  save_trajectories: false      # Save relaxation trajectories (large)
  save_neb_trajectories: true   # Save NEB trajectories
```

---

## 6. Extensibility: Adding New Steps

### Current Steps

| Step | File | Purpose |
|------|------|---------|
| Step 1 | (in pipeline.py) | Material pool fetch |
| Step 2 | stability.py | Electrochemical stability |
| Step 3 | lattice.py | Lattice mismatch |
| Step 4 | adsorption.py | Adsorption energy |
| Step 5 | diffusion.py | NEB diffusion barrier |

### Adding a New Step (e.g., Interface Energy)

1. Create `src/seed_layer/steps/interface_energy.py`
2. Add config section in YAML:
   ```yaml
   interface_energy:
     some_parameter: 1.0
   ```
3. Add step call in `pipeline.py`:
   ```python
   if self.config.get("interface_energy"):
       from .steps.interface_energy import InterfaceEnergyStep
       step = InterfaceEnergyStep(self.config, self.calculator)
       step.run()
   ```
4. Update `io.py` to save new data types if needed
5. Update `reporting.py` to add new columns to summary.csv

**No changes to existing step files required.**

### Summary Integration

Each step writes its results to:
- Per-material JSON (e.g., `interface_energy.json`)
- The step returns a DataFrame with its key results

`reporting.py` merges all step DataFrames into `summary.csv`:
```python
def generate_summary(output_dir: str) -> pd.DataFrame:
    # Merge: stability + lattice + adsorption + diffusion + ...
    # Each step contributes columns: stability_passed, mismatch_pct, E_ads_eV, ...
    # New steps just add more columns
```

---

## 7. CLI Interface

```bash
# Basic usage
python src/main.py --config configs/default.yaml

# With tag
python src/main.py --config configs/default.yaml --tag "LiAl2Ni_test"

# Resume interrupted run
python src/main.py --config configs/default.yaml --resume

# Skip expensive steps
python src/main.py --config configs/default.yaml --skip-neb

# Demo mode (offline, no API/calculator needed)
python src/main.py --config configs/default.yaml --demo

# Specify materials file
python src/main.py --config configs/default.yaml --materials data/test_materials.txt
```

### CLI Overrides

| Flag | Config Equivalent | Notes |
|------|-------------------|-------|
| `--resume` | N/A | Runtime behavior, not a parameter |
| `--skip-neb` | N/A | Runtime behavior |
| `--tag` | N/A | Output directory naming |
| `--materials` | N/A | Input data source |
| `--api-key` | `api.mp_api_key` | Override for convenience |

---

## 8. Implementation Order

1. **Structure Setup** — Create directories, config files, main.py entry point
2. **Core Modules** — config.py, calculators/base.py, calculators/chgnet.py, io.py
3. **Step Migration** — Migrate Step 2-5 from output7/seed_layer.py, each in own file
4. **Pipeline & Reporting** — pipeline.py (orchestrator), reporting.py, summary.csv
5. **Testing** — Demo mode → small real test → verify output structure

---

## 9. Open Questions

1. ~~Config file format~~ → YAML
2. ~~Output directory naming~~ → timestamp_tag
3. ~~Calculator interface~~ → Abstract base class + factory
4. ~~Extensibility~~ → Modular steps

## 10. References

- Current code: `output7/seed_layer.py` (1819 lines, most complete version)
- Reference output: `C:\Users\10660\Desktop\科研项目\材料\方向二\计算电化学稳定性\step3_diffusion`

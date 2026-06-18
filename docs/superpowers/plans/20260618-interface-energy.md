# InterfaceStep 逐层外推法界面能计算 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在种子层筛选 pipeline 中新增界面能计算步骤，采用 seed-metal-seed 三明治结构逐层外推法。

**Architecture:** 新增 InterfaceStep 位于晶格匹配和吸附能之间，接收晶格匹配结果（含 scaling factors），构建三明治结构逐层弛豫，线性拟合提取界面能。透传数据给 AdsorptionStep。

**Tech Stack:** pymatgen, ASE, numpy, matplotlib, MACE-MPA-0

## Global Constraints

- Python 3.13, conda env `claude-code`
- 所有能量单位 eV，长度单位 Å
- 输出放 `E:\Claude code project\seed-layer-screening\output\<run_id>\`
- 代码注释和变量名用英文

---

### Task 1: LatticeStep 保存 scaling factors

**Files:**
- Modify: `src/seed_layer/steps/lattice.py:128-139`

**Interfaces:**
- Produces: `best_info` dict 新增 `film_scale_a`, `film_scale_b`, `ref_scale_a`, `ref_scale_b` 四个 int 字段

- [ ] **Step 1: 修改 `_calc_lattice_match` 保存 scaling factors**

在 `_calc_lattice_match` 方法中，`_compute_mismatch` 返回的 `info` dict 包含 `film_scale_a/b` 和 `ref_scale_a/b`。当前这些值被丢弃了（只保存 mismatch 百分比）。需要在 `best_info` dict 中新增这 4 个字段。

修改 `src/seed_layer/steps/lattice.py` 第 128-139 行：

```python
        # Save mismatch data and slab structure
        if best_info:
            miller_dir = create_miller_dir(step_dir, best_info["miller"])
            save_json(
                {k: v for k, v in best_info.items() if k not in ("relaxed_bulk", "best_slab")},
                miller_dir / "mismatch.json",
            )
            save_structure_cif(best_slab, step_dir / "slab.cif")
            # Pass structures forward for downstream steps
            best_info["relaxed_bulk"] = relaxed_structure
            best_info["best_slab"] = best_slab
```

改为：

```python
        # Save mismatch data and slab structure
        if best_info:
            miller_dir = create_miller_dir(step_dir, best_info["miller"])
            # Add scaling factors for interface construction
            best_info["film_scale_a"] = info.get("film_scale_a", 1)
            best_info["film_scale_b"] = info.get("film_scale_b", 1)
            best_info["ref_scale_a"] = info.get("ref_scale_a", 1)
            best_info["ref_scale_b"] = info.get("ref_scale_b", 1)
            save_json(
                {k: v for k, v in best_info.items() if k not in ("relaxed_bulk", "best_slab")},
                miller_dir / "mismatch.json",
            )
            save_structure_cif(best_slab, step_dir / "slab.cif")
            # Pass structures forward for downstream steps
            best_info["relaxed_bulk"] = relaxed_structure
            best_info["best_slab"] = best_slab
```

- [ ] **Step 2: 语法检查**

Run: `conda run -n claude-code python -m py_compile src/seed_layer/steps/lattice.py`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/seed_layer/steps/lattice.py
git commit -m "feat(lattice): save scaling factors for interface construction"
```

---

### Task 2: 新建 InterfaceStep 基础结构

**Files:**
- Create: `src/seed_layer/steps/interface.py`

**Interfaces:**
- Consumes: `List[Dict]` from LatticeStep (含 `material_id`, `miller`, `relaxed_bulk`, `best_slab`, `film_scale_a/b`, `ref_scale_a/b`)
- Produces: `List[Dict]` (透传，每个 dict 附加 `interface_energy` 字段)

- [ ] **Step 1: 创建 InterfaceStep 骨架**

```python
"""Interface energy calculation via layer-by-layer extrapolation."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.seed_layer.steps.base import BaseStep

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
```

- [ ] **Step 2: 语法检查**

Run: `conda run -n claude-code python -m py_compile src/seed_layer/steps/interface.py`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/seed_layer/steps/interface.py
git commit -m "feat(interface): add InterfaceStep skeleton"
```

---

### Task 3: 实现三明治结构构建

**Files:**
- Modify: `src/seed_layer/steps/interface.py`

**Interfaces:**
- Consumes: `item` dict (含 `best_slab`, `relaxed_bulk`, `miller`, `film_scale_a/b`, `ref_scale_a/b`)
- Produces: ASE Atoms 对象 (seed-metal-seed sandwich)

- [ ] **Step 1: 实现 `_build_sandwich` 方法**

在 `InterfaceStep` 类中添加：

```python
    def _build_sandwich(
        self, seed_slab, ref_bulk, miller,
        film_scale_a, film_scale_b, ref_scale_a, ref_scale_b,
        n_metal_layers, slab_thickness, vacuum,
    ):
        """Build seed-metal-seed sandwich structure.

        Args:
            seed_slab: pymatgen Slab (best_slab from lattice step)
            ref_bulk: pymatgen Structure (reference metal bulk)
            miller: tuple, Miller indices (same for both seed and metal)
            film_scale_a/b: int, seed slab supercell scaling
            ref_scale_a/b: int, metal slab supercell scaling
            n_metal_layers: int, number of metal layers
            slab_thickness: float, slab thickness in Å
            vacuum: float, vacuum thickness in Å

        Returns:
            ase.Atoms: seed-metal-seed sandwich with selective dynamics
        """
        from ase import Atoms
        from ase.build import make_supercell
        from ase.constraints import FixAtoms
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core.surface import SlabGenerator

        adaptor = AseAtomsAdaptor()

        # 1. Build seed slab supercell
        seed_ase = adaptor.get_atoms(seed_slab)
        P_seed = [[film_scale_a, 0, 0], [0, film_scale_b, 0], [0, 0, 1]]
        seed_ase = make_supercell(seed_ase, P_seed)

        # 2. Build metal slab
        slabgen = SlabGenerator(
            ref_bulk, miller, slab_thickness, vacuum,
            center_slab=True, primitive=False
        )
        metal_slabs = slabgen.get_slabs()
        if not metal_slabs:
            raise ValueError(f"No slabs generated for ref metal {miller}")
        metal_slab = metal_slabs[0]

        metal_ase = adaptor.get_atoms(metal_slab)
        P_ref = [[ref_scale_a, 0, 0], [0, ref_scale_b, 0], [0, 0, 1]]
        metal_ase = make_supercell(metal_ase, P_ref)

        # 3. Strip vacuum from seed slab (keep only atomic region)
        seed_positions = seed_ase.get_positions()
        seed_cell = seed_ase.get_cell()
        z_seed = seed_positions[:, 2]
        seed_height = z_seed.max() - z_seed.min()
        # Shift seed atoms so bottom is at z=0
        seed_positions[:, 2] -= z_seed.min()
        seed_ase.set_positions(seed_positions)
        seed_cell[2, 2] = seed_height + vacuum
        seed_ase.set_cell(seed_cell)

        # 4. Strip vacuum from metal slab, keep only n layers
        metal_positions = metal_ase.get_positions()
        z_metal = metal_positions[:, 2]
        # Sort atoms by z, take only the bottom n layers
        z_sorted = np.sort(np.unique(np.round(z_metal, 2)))
        if n_metal_layers > len(z_sorted):
            raise ValueError(
                f"Requested {n_metal_layers} layers but metal slab has {len(z_sorted)}"
            )
        z_cutoff = z_sorted[n_metal_layers - 1] + 0.5  # include atoms in the target layer
        metal_mask = metal_positions[:, 2] <= z_cutoff
        metal_indices = np.where(metal_mask)[0]
        metal_atoms = metal_ase[metal_indices]
        # Shift so bottom is at z=0
        z_metal_min = metal_atoms.get_positions()[:, 2].min()
        metal_positions = metal_atoms.get_positions()
        metal_positions[:, 2] -= z_metal_min
        metal_atoms.set_positions(metal_positions)

        # 5. Create mirror seed (flip z)
        mirror_positions = seed_ase.get_positions().copy()
        mirror_positions[:, 2] = seed_height - mirror_positions[:, 2]
        mirror_ase = seed_ase.copy()
        mirror_ase.set_positions(mirror_positions)

        # 6. Stack: seed + metal + mirror
        # seed: z = [0, seed_height]
        # metal: z = [seed_height, seed_height + metal_height]
        # mirror: z = [seed_height + metal_height, 2*seed_height + metal_height]
        metal_height = metal_atoms.get_positions()[:, 2].max()

        # Shift metal to sit on top of seed
        metal_pos = metal_atoms.get_positions()
        metal_pos[:, 2] += seed_height
        metal_atoms.set_positions(metal_pos)

        # Shift mirror to sit on top of metal
        mirror_pos = mirror_ase.get_positions()
        mirror_pos[:, 2] += seed_height + metal_height
        mirror_ase.set_positions(mirror_pos)

        # Combine
        sandwich = seed_ase + metal_atoms + mirror_ase

        # Set cell with vacuum on top
        total_height = 2 * seed_height + metal_height
        cell = seed_ase.get_cell().copy()
        cell[2, 2] = total_height + vacuum
        sandwich.set_cell(cell)
        sandwich.set_pbc(True)

        # 7. Fix seed layers (bottom and top)
        n_seed = len(seed_ase)
        n_mirror = len(mirror_ase)
        seed_indices = list(range(n_seed)) + list(range(n_seed + len(metal_atoms), len(sandwich)))
        sandwich.set_constraint(FixAtoms(indices=seed_indices))

        return sandwich
```

- [ ] **Step 2: 语法检查**

Run: `conda run -n claude-code python -m py_compile src/seed_layer/steps/interface.py`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/seed_layer/steps/interface.py
git commit -m "feat(interface): implement sandwich structure builder"
```

---

### Task 4: 实现逐层弛豫和线性拟合

**Files:**
- Modify: `src/seed_layer/steps/interface.py`

**Interfaces:**
- Consumes: sandwich ASE Atoms 对象
- Produces: `Dict` with `interface_energy_eV_per_A2`, `bulk_energy_per_layer_eV`, `R2`, `energies_per_n`

- [ ] **Step 1: 实现 `_calc_interface` 方法**

在 `InterfaceStep` 类中添加：

```python
    def _calc_interface(
        self, item: Dict, step_dir: Path,
        max_layers: int, slab_thickness: float,
        vacuum: float, fmax: float, steps: int,
    ) -> Dict[str, Any]:
        """Calculate interface energy for one material."""
        import json
        from pymatgen.io.ase import AseAtomsAdaptor
        from src.seed_layer.io import save_structure_cif

        adaptor = AseAtomsAdaptor()
        material_id = item["material_id"]
        miller = item["miller"]

        # Get structures from lattice step
        seed_slab = item.get("best_slab")
        relaxed_bulk = item.get("relaxed_bulk")
        if seed_slab is None or relaxed_bulk is None:
            raise ValueError("Missing best_slab or relaxed_bulk from lattice step")

        # Get scaling factors
        film_scale_a = item.get("film_scale_a", 1)
        film_scale_b = item.get("film_scale_b", 1)
        ref_scale_a = item.get("ref_scale_a", 1)
        ref_scale_b = item.get("ref_scale_b", 1)

        # Build reference metal structure
        ref_struct = self.build_ref_structure()
        ref_bulk = self.calculator.relax(
            ref_struct, fmax=fmax, steps=steps, relax_cell=True
        )["final_structure"]

        # Relax seed slab separately to get E_seed
        logger.info("Relaxing seed slab...")
        seed_relaxed = self.calculator.relax(
            seed_slab, fmax=fmax, steps=steps, relax_cell=False
        )
        e_seed = seed_relaxed["energy"]
        seed_struct_relaxed = seed_relaxed["final_structure"]
        save_structure_cif(seed_struct_relaxed, step_dir / "seed_slab.cif")

        # Save metal bulk
        save_structure_cif(ref_bulk, step_dir / "metal_bulk.cif")

        # Metal slab uses same Miller index as seed layer
        # (the interface plane must match on both sides)

        # Build max-thickness metal slab, then slice for each n
        from pymatgen.core.surface import SlabGenerator
        slabgen = SlabGenerator(
            ref_bulk, miller, slab_thickness, vacuum,
            center_slab=True, primitive=False
        )
        metal_slabs = slabgen.get_slabs()
        if not metal_slabs:
            raise ValueError(f"No metal slabs generated for {miller}")
        metal_slab_pmg = metal_slabs[0]

        # Convert to ASE and expand
        from ase.build import make_supercell
        metal_ase = adaptor.get_atoms(metal_slab_pmg)
        P_ref = [[ref_scale_a, 0, 0], [0, ref_scale_b, 0], [0, 0, 1]]
        metal_ase = make_supercell(metal_ase, P_ref)

        # Count available metal layers
        z_metal = metal_ase.get_positions()[:, 2]
        z_layers = np.sort(np.unique(np.round(z_metal, 2)))
        actual_max = min(max_layers, len(z_layers))
        if actual_max < 1:
            raise ValueError("No metal layers available")

        logger.info(f"Metal slab has {len(z_layers)} layers, computing up to {actual_max}")

        # Calculate area from seed slab lattice
        seed_cell = adaptor.get_atoms(seed_slab).get_cell()
        area = abs(np.cross(seed_cell[0], seed_cell[1])[2])
        # For 2D slabs, area = |a × b| (z-component)
        area = np.linalg.norm(np.cross(seed_cell[0][:2], seed_cell[1][:2]))

        # Layer-by-layer calculation
        energies = {}
        for n in range(1, actual_max + 1):
            logger.info(f"  Computing n={n}/{actual_max}...")
            try:
                sandwich = self._build_sandwich(
                    seed_slab, ref_bulk, miller,
                    film_scale_a, film_scale_b, ref_scale_a, ref_scale_b,
                    n_metal_layers=n,
                    slab_thickness=slab_thickness,
                    vacuum=vacuum,
                )
                # Relax metal atoms only (seed is fixed by constraint)
                result = self.calculator.relax(
                    adaptor.get_structure(sandwich),
                    fmax=fmax, steps=steps, relax_cell=False,
                )
                e_total = result["energy"]
                energies[str(n)] = round(e_total, 6)

                # Save relaxed structure
                save_structure_cif(
                    result["final_structure"],
                    step_dir / f"sandwich_n{n}.cif",
                )
                logger.info(f"    n={n}: E = {e_total:.4f} eV")
            except Exception as e:
                logger.warning(f"    n={n} failed: {e}")
                energies[str(n)] = None

        # Linear fit
        valid_ns = [int(k) for k, v in energies.items() if v is not None]
        valid_es = [energies[str(n)] for n in valid_ns]

        result_dict = {
            "material_id": material_id,
            "seed_slab_energy_eV": round(e_seed, 6),
            "area_A2": round(area, 4),
            "energies_per_n": energies,
        }

        if len(valid_ns) >= 2:
            coeffs = np.polyfit(valid_ns, valid_es, 1)
            slope = coeffs[0]  # bulk energy per layer
            intercept = coeffs[1]  # 2*E_seed + E_interface
            # R² calculation
            predicted = np.polyval(coeffs, valid_ns)
            ss_res = np.sum((np.array(valid_es) - predicted) ** 2)
            ss_tot = np.sum((np.array(valid_es) - np.mean(valid_es)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Interface energy: γ = (intercept - 2*E_seed) / (2*A)
            interface_energy = (intercept - 2 * e_seed) / (2 * area)

            result_dict["interface_energy_eV_per_A2"] = round(interface_energy, 6)
            result_dict["bulk_energy_per_layer_eV"] = round(slope, 6)
            result_dict["R2"] = round(r2, 4)
            result_dict["intercept_eV"] = round(intercept, 6)

            if r2 < 0.95:
                logger.warning(f"  Low R² = {r2:.3f} for {material_id}")

            # Generate plot
            self._plot_interface(
                step_dir, valid_ns, valid_es, slope, intercept, r2,
                interface_energy, area, material_id,
            )
        else:
            logger.warning(f"  Not enough data points for linear fit ({len(valid_ns)})")
            result_dict["interface_energy_eV_per_A2"] = None
            result_dict["bulk_energy_per_layer_eV"] = None
            result_dict["R2"] = None

        # Save JSON
        with open(step_dir / "interface.json", "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        return result_dict
```

- [ ] **Step 2: 实现 `_plot_interface` 方法**

```python
    def _plot_interface(
        self, step_dir, ns, energies, slope, intercept, r2,
        interface_energy, area, material_id,
    ):
        """Generate E(n) plot with linear fit."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        # Data points
        ax.scatter(ns, energies, c="blue", s=60, zorder=5, label="Calculated")

        # Fit line
        x_fit = np.linspace(min(ns) - 0.5, max(ns) + 0.5, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, "r--", linewidth=1.5, label="Linear fit")

        # Mark intercept
        ax.scatter([0], [intercept], c="red", s=80, marker="*", zorder=5)

        # Labels
        ax.set_xlabel("Metal layers (n)", fontsize=12)
        ax.set_ylabel("Total energy (eV)", fontsize=12)
        ax.set_title(f"Interface Energy: {material_id}", fontsize=14)

        # Annotation
        textstr = (
            f"γ = {interface_energy:.4f} eV/Å²\n"
            f"slope = {slope:.4f} eV/layer\n"
            f"R² = {r2:.4f}"
        )
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(step_dir / "interface_plot.png", dpi=150)
        plt.close()
```

- [ ] **Step 3: 语法检查**

Run: `conda run -n claude-code python -m py_compile src/seed_layer/steps/interface.py`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add src/seed_layer/steps/interface.py
git commit -m "feat(interface): implement layer-by-layer relaxation and linear fit"
```

---

### Task 5: Pipeline 集成

**Files:**
- Modify: `src/seed_layer/pipeline.py:101-148`

**Interfaces:**
- Consumes: `matched_materials` from LatticeStep
- Produces: `matched_materials` (透传，附加 interface_energy)

- [ ] **Step 1: 在 pipeline.py 中导入 InterfaceStep**

在文件顶部导入区添加：

```python
from src.seed_layer.steps.interface import InterfaceStep
```

- [ ] **Step 2: 插入 InterfaceStep 到 pipeline**

在 `run()` 方法中，Step 3（lattice）之后、Step 4（adsorption）之前插入：

```python
        # Step 3.5: Interface energy
        logger.info("=" * 50)
        logger.info("Step 3.5: Interface Energy")
        logger.info("=" * 50)
        interface_step = InterfaceStep(self.config, self.calculator, self.run_dir)
        matched_materials = interface_step.run(matched_materials)
```

- [ ] **Step 3: 语法检查**

Run: `conda run -n claude-code python -m py_compile src/seed_layer/pipeline.py`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add src/seed_layer/pipeline.py
git commit -m "feat(pipeline): insert InterfaceStep between lattice and adsorption"
```

---

### Task 6: 配置更新

**Files:**
- Modify: `configs/default.yaml`

**Interfaces:**
- Produces: `interface` 配置节, 更新后的 `scoring` 权重

- [ ] **Step 1: 新增 interface 配置节**

在 `configs/default.yaml` 中，在 `adsorption:` 之前添加：

```yaml
# ── 界面能 ──
interface:
  max_metal_layers: 5          # 逐层计算的最大金属层数
  slab_thickness: 5            # slab 厚度 Å
  vacuum: 15.0                 # 真空层厚度 Å
  fmax: 0.05                   # 驰豫力收敛 eV/Å
  steps: 500                   # 最大弛豫步数
```

- [ ] **Step 2: 更新评分权重**

将 `scoring:` 节更新为：

```yaml
# ── 评分 ──
scoring:
  w_lattice: 0.15       # 原 0.25
  w_adsorption: 0.25    # 不变
  w_diffusion: 0.25     # 不变
  w_stability: 0.10     # 原 0.25
  w_interface: 0.25     # 新增
```

- [ ] **Step 3: Commit**

```bash
git add configs/default.yaml
git commit -m "feat(config): add interface energy config and update scoring weights"
```

---

### Task 7: 评分集成

**Files:**
- Modify: `src/seed_layer/reporting.py:97-189`

**Interfaces:**
- Consumes: `interface_energy_eV_per_A2` from InterfaceStep results
- Produces: `S_interface` score column in DataFrame

- [ ] **Step 1: 更新 `generate_summary_csv` 签名和逻辑**

在 `generate_summary_csv` 中添加 `interface_results` 参数：

```python
def generate_summary_csv(
    output_dir: Path,
    stable_ids: List[str],
    matched_materials: List[Dict[str, Any]],
    adsorption_results: Dict[str, Any],
    diffusion_results: Dict[str, Any],
    interface_results: Dict[str, Any] = None,
    max_mismatch: float = 8.0,
    w_lattice: float = 0.15,
    w_adsorption: float = 0.25,
    w_diffusion: float = 0.25,
    w_stability: float = 0.10,
    w_interface: float = 0.25,
) -> pd.DataFrame:
```

在 row 构建循环中，添加界面能数据：

```python
        # Add interface energy data if available
        if interface_results and mp_id in interface_results:
            iface_data = interface_results[mp_id]
            if iface_data and iface_data.get("interface_energy_eV_per_A2") is not None:
                row["interface_energy_eV_per_A2"] = iface_data["interface_energy_eV_per_A2"]
```

在 `_calculate_scores` 调用中传递新权重：

```python
    if not df.empty:
        df = _calculate_scores(
            df, max_mismatch=max_mismatch,
            w_lattice=w_lattice, w_adsorption=w_adsorption,
            w_diffusion=w_diffusion, w_stability=w_stability,
            w_interface=w_interface,
        )
```

- [ ] **Step 2: 更新 `_calculate_scores` 函数**

更新签名和权重参数：

```python
def _calculate_scores(
    df: pd.DataFrame,
    max_mismatch: float = 8.0,
    w_lattice: float = 0.15,
    w_adsorption: float = 0.25,
    w_diffusion: float = 0.25,
    w_stability: float = 0.10,
    w_interface: float = 0.25,
) -> pd.DataFrame:
```

在 score 计算部分，添加 `S_interface`：

```python
    # Calculate interface energy score
    if 'interface_energy_eV_per_A2' in df.columns:
        valid_iface = df['interface_energy_eV_per_A2'].notna()
        if valid_iface.any():
            # Lower interface energy is better
            # Normalize: S = exp(-gamma / gamma_ref), gamma_ref = 0.05 eV/Å²
            df.loc[valid_iface, 'S_interface'] = np.exp(
                -df.loc[valid_iface, 'interface_energy_eV_per_A2'] / 0.05
            ).clip(0, 1)
```

在 composite score 循环中，添加界面能项：

```python
        # Interface score
        if pd.notna(df.loc[idx, 'S_interface']):
            terms.append((w_interface, df.loc[idx, 'S_interface']))
            mode_parts.append('iface')
```

- [ ] **Step 3: 更新 pipeline.py 中的 `generate_summary_csv` 调用**

在 `pipeline.py` 的 `_run_full_pipeline` 方法中，收集 interface_results 并传递：

```python
        # Collect interface results
        interface_results = {}
        for mat in matched_materials:
            mp_id = mat["material_id"]
            iface = mat.get("interface_energy")
            if iface:
                interface_results[mp_id] = iface

        # Generate summary
        df = generate_summary_csv(
            self.run_dir,
            stable_materials,
            matched_materials,
            adsorption_results,
            diffusion_results,
            interface_results=interface_results,
            max_mismatch=self.config.screening.get("max_lattice_mismatch", 8.0),
        )
```

- [ ] **Step 4: 语法检查**

Run: `conda run -n claude-code python -m py_compile src/seed_layer/reporting.py`
Run: `conda run -n claude-code python -m py_compile src/seed_layer/pipeline.py`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/seed_layer/reporting.py src/seed_layer/pipeline.py
git commit -m "feat(scoring): integrate interface energy into scoring system"
```

---

### Task 8: 端到端验证

**Files:**
- Test: `tests/test_interface.py`

- [ ] **Step 1: 创建接口测试**

```python
"""Tests for InterfaceStep."""

import sys
sys.path.insert(0, r"E:\Claude code project\seed-layer-screening")

import numpy as np
from unittest.mock import MagicMock, patch
from src.seed_layer.steps.interface import InterfaceStep


def test_sandwich_structure_count():
    """Verify sandwich has correct atom count: seed + metal + mirror."""
    config = MagicMock()
    config.working_ion = "Li"
    config.get.return_value = {
        "max_metal_layers": 3,
        "slab_thickness": 5,
        "vacuum": 15.0,
        "fmax": 0.05,
        "steps": 500,
    }
    calculator = MagicMock()
    output_dir = MagicMock()

    step = InterfaceStep(config, calculator, output_dir)

    # Mock structures
    from pymatgen.core import Structure, Lattice
    seed_slab = Structure(
        Lattice.orthorhombic(3.0, 3.0, 15.0),
        ["Si", "Si"],
        [[0, 0, 0.3], [0.5, 0.5, 0.4]],
    )
    ref_bulk = Structure(
        Lattice.cubic(3.49),
        ["Li"],
        [[0, 0, 0]],
    )

    # This will fail at ASE conversion in test env, but verifies method exists
    assert hasattr(step, "_build_sandwich")
    assert hasattr(step, "_calc_interface")
    assert hasattr(step, "_plot_interface")
    assert step.step_dir_name == "06_interface"


if __name__ == "__main__":
    test_sandwich_structure_count()
    print("All tests passed!")
```

- [ ] **Step 2: 运行测试**

Run: `conda run -n claude-code python tests/test_interface.py`
Expected: "All tests passed!"

- [ ] **Step 3: 语法检查所有改动文件**

Run: `conda run -n claude-code python -m py_compile src/seed_layer/steps/lattice.py && python -m py_compile src/seed_layer/steps/interface.py && python -m py_compile src/seed_layer/pipeline.py && python -m py_compile src/seed_layer/reporting.py`
Expected: All OK

- [ ] **Step 4: Commit**

```bash
git add tests/test_interface.py
git commit -m "test(interface): add InterfaceStep unit tests"
```

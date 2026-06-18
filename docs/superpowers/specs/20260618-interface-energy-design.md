# InterfaceStep 逐层外推法界面能计算设计

## 概述

在种子层筛选 pipeline 中新增界面能计算步骤，采用逐层外推法：构建 seed-metal-seed 三明治结构，逐层增加金属层数 n，固定种子层弛豫金属层，拟合 E(n) 线性方程，截距提取界面能。

## Pipeline 位置

```
初筛 → 稳定性 → 晶格匹配 → 【界面能】→ 吸附能 → 扩散 → 评分
```

插入在 Step 3（晶格匹配）之后、Step 4（吸附能）之前。

## 数据流

- **输入**：晶格匹配结果 `List[Dict]`，每个 dict 含 `material_id`, `miller`, `relaxed_bulk`, `best_slab` + 新增的 scaling factors
- **输出**：同样结构的 `List[Dict]`（透传），每个 dict 附加 `interface_energy` 字段
- AdsorptionStep 接口不变，无需改动

## 界面结构构建

### 三明治结构（seed-metal-seed）

1. 从晶格匹配拿到 `best_slab`（种子层 primitive slab）+ scaling factors
2. 种子层 slab 转 ASE → 用 `film_scale_a × film_scale_b` 扩超胞
3. 金属体相从 `build_ref_structure()` 获取 → `SlabGenerator(ref_miller)` 切面 → 用 `ref_scale_a × ref_scale_b` 扩超胞
4. 种子层 slab 去真空，翻转得到镜像
5. 堆叠：`seed | metal(n层) | seed(镜像)`
6. 用 `selective_dynamics` 固定两层种子层原子

### 逐层计算

- 先构建 max_metal_layers 厚度的金属 slab
- 对 n = 1, 2, ..., max_metal_layers：裁剪前 n 层 → 堆叠 → 弛豫 → 记录能量
- 每个 n 值的弛豫结构保存到 `06_interface/mp-xxx/` 目录

### 弛豫策略

- 固定两层种子层（`FixAtoms`）
- 只弛豫金属层原子
- `relax_cell=False`（保持晶胞不变）

## 界面能提取

### 线性拟合

对 n = 1, 2, ..., N，计算三明治总能量 E_total(n)，拟合：

```
E_total(n) = b + n × k
```

- 斜率 k = 金属体相每层能量（排除形变）
- 截距 b = 2 × E_seed_slab + E_interface

### 界面能计算

```
γ = (b - 2 × E_seed_slab) / (2A)
```

- E_seed_slab：单独弛豫种子层 slab 的能量
- A：三明治超胞截面积（从晶格向量叉乘得到）
- 2A：两个界面的总面积

### 拟合质量

- R² < 0.95 时 log warning
- 如果只有 1 个数据点（max_metal_layers=1），跳过拟合，直接输出单点能量

## 输出

### 目录结构

```
06_interface/
  mp-xxx/
    seed_slab.cif          # 种子层 slab（弛豫后）
    metal_slab.cif         # 金属 slab
    sandwich_n1.cif        # n=1 三明治结构
    sandwich_n2.cif        # n=2 三明治结构
    ...
    interface.json         # 界面能数据
    interface_plot.png     # E(n) 散点 + 拟合线
```

### interface.json 格式

```json
{
  "material_id": "mp-xxx",
  "interface_energy_eV_per_A2": 0.0123,
  "bulk_energy_per_layer_eV": -4.567,
  "R2": 0.998,
  "seed_slab_energy_eV": -123.45,
  "area_A2": 45.67,
  "energies_per_n": {
    "1": -150.12,
    "2": -154.89,
    "3": -159.45,
    "4": -164.01,
    "5": -168.57
  }
}
```

### 图表

- X 轴：金属层数 n
- Y 轴：E_total (eV)
- 散点：各 n 值的计算能量
- 拟合直线：E = b + n×k
- 标注：截距 b、斜率 k、R²、界面能 γ

## 配置

### default.yaml 新增

```yaml
interface:
  max_metal_layers: 5
  slab_thickness: 5        # Å
  vacuum: 15.0             # Å
  fmax: 0.05               # eV/Å
  steps: 500
```

### 评分权重调整

```yaml
scoring:
  w_lattice: 0.15       # 原 0.25
  w_adsorption: 0.25    # 不变
  w_diffusion: 0.25     # 不变
  w_stability: 0.10     # 原 0.25
  w_interface: 0.25     # 新增
```

总和 = 1.0。界面能、吸附能、扩散度三个核心指标权重最高。

## LatticeStep 改动

在 `_calc_lattice_match` 返回的 best_info dict 中新增 4 个字段：

```python
"film_scale_a": info["film_scale_a"],
"film_scale_b": info["film_scale_b"],
"ref_scale_a": info["ref_scale_a"],
"ref_scale_b": info["ref_scale_b"],
```

同时写入 `mismatch.json`（可序列化）。

## 异常处理

### 结构构建失败
- 种子层 slab 转 ASE 失败 → 跳过该材料，log warning
- 金属 slab 生成失败（ref_miller 不兼容）→ 跳过该材料
- 堆叠后原子数异常（< 10 或 > 10000）→ 跳过

### 弛豫失败
- 单个 n 值弛豫不收敛 → 记录为 None，拟合时排除
- 所有 n 值都失败 → 该材料界面能设为 None

### 拟合异常
- 数据点 < 2 → 跳过拟合，输出单点能量
- R² < 0.95 → log warning，结果仍保存
- 截距 > 0（物理上不合理）→ log warning

## 实现清单

| # | 文件 | 改动 |
|---|------|------|
| 1 | `steps/lattice.py` | 保存 scaling factors 到 result dict 和 mismatch.json |
| 2 | `steps/interface.py` | 新建 InterfaceStep 类 |
| 3 | `pipeline.py` | 插入 InterfaceStep 到 Step 3 和 Step 4 之间 |
| 4 | `configs/default.yaml` | 新增 interface 配置节，调整评分权重 |
| 5 | `reporting.py` | 评分计算中加入界面能 |

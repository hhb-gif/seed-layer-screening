# 参数说明（中文）

本文档说明 `configs/default.yaml` 中所有配置参数的含义、类型、默认值和调优建议。

---

## API 设置

```yaml
api:
  mp_api_key: "${MP_API_KEY}"
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `mp_api_key` | string | Materials Project API Key。支持 `${MP_API_KEY}` 引用环境变量，也可直接填写字符串。 |

---

## 材料池初筛 (screening)

```yaml
screening:
  energy_above_hull_max: 0.10
  n_elements: [2, 3]
  elements_to_exclude: [...]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `energy_above_hull_max` | float | 0.10 | 能量高于凸包的最大值（eV/atom）。值越小，筛选出的材料热力学稳定性越高。0.1 eV 以内通常认为是亚稳态或稳定相。 |
| `n_elements` | list[int] | [2, 3] | 允许的元素种类数。[2, 3] 表示只保留二元和三元化合物。 |
| `elements_to_exclude` | list[str] | (见配置) | 排除的元素列表。包括贵金属（Pt, Pd 等）、有毒元素（Hg, Pb, Cd 等）、放射性元素（U, Th 等）和稀土元素（La-Lu）。 |

---

## 晶格匹配 (lattice)

```yaml
lattice:
  li_miller: [1, 1, 0]
  max_mismatch: 8.0
  angle_tolerance: 180.0
  max_scale: 8
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `li_miller` | list[int] | [1, 1, 0] | 锂金属的参考晶面 Miller 指数。Li 为 BCC 结构，(110) 是最密排面。 |
| `max_mismatch` | float | 8.0 | 最大允许晶格失配百分比。一般 <10% 可接受，<5% 为良好匹配。 |
| `angle_tolerance` | float | 180.0 | 晶格角度容差（度）。180 表示不做角度限制。 |
| `max_scale` | int | 8 | 超胞缩放上限。控制晶格匹配搜索时尝试的最大超胞尺寸。 |

---

## 表面模型 (surface)

```yaml
surface:
  vacuum: 15.0
  slab_thickness: 15.0
  default_miller: [0, 0, 1]
  supercell: [2, 2, 1]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vacuum` | float | 15.0 | 真空层厚度（Angstrom）。需足够大以避免周期性镜像相互作用，通常 12-20 A。 |
| `slab_thickness` | float | 15.0 | slab 模型厚度（Angstrom）。需足够厚以体现体相性质，通常 10-20 A。 |
| `default_miller` | list[int] | [0, 0, 1] | 默认切割面 Miller 指数。当材料没有特别指定时使用。 |
| `supercell` | list[int] | [2, 2, 1] | 表面超胞尺寸。[2,2,1] 表示 a、b 方向各扩大 2 倍，c 方向不变。 |

---

## 势函数 (calculator)

```yaml
calculator:
  type: "chgnet"
  kwargs: {}
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | string | "chgnet" | 势函数类型。目前支持 "chgnet"，未来可扩展 "mace"、"m3gnet" 等。 |
| `kwargs` | dict | {} | 传递给势函数构造器的额外参数。 |

---

## 弛豫参数 (relaxation)

```yaml
relaxation:
  fmax_bulk: 0.05
  fmax_slab: 0.10
  fmax_adsorb: 0.05
  steps_bulk: 500
  steps_slab: 200
  steps_adsorb: 250
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fmax_bulk` | float | 0.05 | 体相结构弛豫的力收敛标准（eV/Angstrom）。越小越精确但越慢。 |
| `fmax_slab` | float | 0.10 | slab 弛豫的力收敛标准。表面模型原子数多，可适当放宽。 |
| `fmax_adsorb` | float | 0.05 | 吸附体系弛豫的力收敛标准。吸附能对结构敏感，需较严格。 |
| `steps_bulk` | int | 500 | 体相弛豫最大步数。 |
| `steps_slab` | int | 200 | slab 弛豫最大步数。 |
| `steps_adsorb` | int | 250 | 吸附体系弛豫最大步数。 |

---

## 吸附能 (adsorption)

```yaml
adsorption:
  coverages: [0.25, 0.5, 0.75, 1.0]
  adsorption_min: -0.8
  adsorption_max: -0.2
  adsorbate_height: 1.8
  li_area_per_atom: 8.0
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `coverages` | list[float] | [0.25, 0.5, 0.75, 1.0] | 覆盖度列表。表示 Li 原子占表面位点的比例。0.25 = 25% 覆盖。 |
| `adsorption_min` | float | -0.8 | 吸附能下限（eV）。低于此值认为吸附过强（可能形成合金）。 |
| `adsorption_max` | float | -0.2 | 吸附能上限（eV）。高于此值（更正）认为吸附太弱，无法引导沉积。 |
| `adsorbate_height` | float | 1.8 | Li 吸附原子初始高度（Angstrom），相对于表面。 |
| `li_area_per_atom` | float | 8.0 | Li 金属表面每个原子的面积（Angstrom^2），用于计算覆盖度。 |

---

## NEB 扩散 (diffusion)

```yaml
diffusion:
  neb_n_images: 7
  neb_fmax: 0.10
  neb_steps: 200
  neb_climb: true
  li_displacement_min: 0.2
  neb_top_n: 50
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `neb_n_images` | int | 7 | NEB 路径中的中间 image 数量。越多结果越平滑，但计算量线性增长。 |
| `neb_fmax` | float | 0.10 | NEB 弛豫的力收敛标准（eV/Angstrom）。 |
| `neb_steps` | int | 200 | NEB 弛豫最大步数。 |
| `neb_climb` | bool | true | 是否启用 climbing image NEB。启用后能更精确找到鞍点。 |
| `li_displacement_min` | float | 0.2 | Li 原子最小位移阈值（Angstrom）。位移太小的路径跳过。 |
| `neb_top_n` | int | 50 | 只对排名前 N 的材料做 NEB 计算。控制计算量。 |

---

## 打分权重 (scoring)

```yaml
scoring:
  w_lattice: 0.3
  w_adsorption: 0.4
  w_diffusion: 0.3
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `w_lattice` | float | 0.3 | 晶格匹配度得分权重。失配越小得分越高。 |
| `w_adsorption` | float | 0.4 | 吸附能得分权重。吸附能越接近理想窗口得分越高。 |
| `w_diffusion` | float | 0.3 | 扩散势垒得分权重。势垒越低，Li 扩散越容易，得分越高。 |

三个权重之和应为 1.0。吸附能权重最高，因为它最直接反映种子层引导 Li 沉积的能力。

---

## 输出控制 (output)

```yaml
output:
  save_structures: true
  save_trajectories: false
  save_neb_trajectories: true
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_structures` | bool | true | 是否保存弛豫后的结构文件（CIF/POSCAR）。 |
| `save_trajectories` | bool | false | 是否保存弛豫过程的轨迹文件。开启会占用较多磁盘空间。 |
| `save_neb_trajectories` | bool | true | 是否保存 NEB 路径轨迹。用于可视化扩散路径。 |

---

## 调参建议

### 计算精度 vs 速度

- 放宽 `fmax_*`（如 0.1 -> 0.2）可加速弛豫，但牺牲精度
- 减少 `neb_n_images`（如 7 -> 5）可大幅减少 NEB 计算时间
- 减小 `neb_top_n` 只对最 promising 的材料做 NEB

### 筛选严格度

- 降低 `energy_above_hull_max`（如 0.1 -> 0.05）筛选更稳定的材料
- 降低 `max_mismatch`（如 8% -> 5%）要求更好的晶格匹配
- 调整 `adsorption_min/max` 收窄吸附能窗口

### 内存优化

- 减小 `supercell`（如 [2,2,1] -> [1,1,1]）降低 slab 模型原子数
- 减小 `vacuum` 和 `slab_thickness` 降低体系总原子数

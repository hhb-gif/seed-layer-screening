# 基于机器学习势的高通量锂金属电池种子层材料筛选

## 项目简介

锂金属负极被认为是下一代高能量密度电池的理想选择，但枝晶生长问题严重制约其实际应用。本项目通过**高通量计算筛选**，寻找适合作为"种子层"的材料——在锂表面沉积一层薄层，引导锂均匀沉积、抑制枝晶生长。

筛选流程基于 Materials Project 数据库，结合 **CHGNet 机器学习势**和 **CI-NEB 扩散计算**，从数千种候选材料中逐步筛选出最优种子层材料。

## 筛选流程

```
Step 1  材料池初筛    → 从 MP 数据库拉取，过滤有害元素，保留二/三元化合物
Step 2  电化学稳定性  → 相图法检查与 Li 接触是否反应
Step 3  晶格匹配度    → 各低指数面 vs Li(110)，失配 < 8%
Step 4  CHGNet 吸附能 → 构建 slab，放 Li 原子，弛豫算能量差
Step 5  CI-NEB 扩散   → 2×2 超胞，7 image，BFGS 优化扩散势垒
```

每步层层筛选，下一层只计算上一层达标的材料。

## 目录结构

```
seed-layer-screening/
├── README.md
├── .gitignore
├── src/                                ← 代码
│   ├── seed_layer_pipeline.py          ← 原版主程序
│   ├── seed_layer_pipeline_improved.py ← 改进版主程序
│   └── debug/
│       ├── debug_lattice.py            ← 晶格匹配调试
│       ├── debug_lattice2.py
│       └── debug_neb.py                ← NEB 计算调试
└── data/                               ← 示例材料列表
    ├── test_materials.txt
    └── sample_materials.txt
```

运行产出（不纳入版本控制）按以下结构组织：

```
output_x/                   ← 每次运行一个文件夹
├── 原材料池/
│   ├── run_config.json     ← 运行参数记录
│   └── step1_materials_pool.csv
├── 计算结果/
│   ├── step2_stability.csv
│   ├── step3_lattice_match.csv
│   ├── step4_adsorption.csv
│   └── step5_diffusion.csv
└── 评分/
    ├── final_ranking.csv
    └── final_report.txt
```

## 运行方法

### 环境要求

- Python 3.12+
- conda 环境 `materials_searching`（含 pymatgen, chgnet, mp-api, ASE, pandas）
- Materials Project API Key

### 全量运行

```bash
cd seed-layer-screening/src
MP_API_KEY=<your_key> python seed_layer_pipeline_improved.py --output output_1
```

### 指定材料列表

```bash
python seed_layer_pipeline_improved.py --materials ../data/test_materials.txt --output output_1
```

### 断点续跑

```bash
python seed_layer_pipeline_improved.py --output output_1 --resume
```

### 离线逻辑验证（Demo 模式）

```bash
python seed_layer_pipeline_improved.py --demo --output demo_output
```

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--output` | 输出目录名（默认 `output`） |
| `--resume` | 断点续跑，跳过已有 CSV 的步骤 |
| `--materials` | 自定义材料列表（每行一个 mp-id） |
| `--skip-neb` | 跳过 Step 5 NEB 计算 |
| `--skip-lattice` | 跳过 Step 3，稳定性通过后直接算吸附 |
| `--demo` | 离线模式，不依赖 MP API 和 CHGNet |
| `--api-key` | MP API Key（或通过环境变量 `MP_API_KEY` 设置） |

## 原版 vs 改进版

| 对比项 | 原版 | 改进版 |
|--------|------|--------|
| 评分归一化 | 缺失项混入分母 | 按已有项归一化 |
| 排序逻辑 | 不一致 | NEB 加入后重新排序 |
| Miller 指数解析 | `eval()` | `ast.literal_eval()`（更安全） |
| 离线验证 | 不支持 | `--demo` 模式 |
| 输出结构 | 平铺 | 分类子目录 + run_config |

## 当前结果

10 个候选材料 → 3 个通过稳定性筛选：

| 排名 | 材料 | 综合得分 | 吸附能 | 晶格失配 | 扩散势垒 |
|------|------|---------|--------|---------|---------|
| 1 | LiAl2Ni | 0.892 | -0.608 eV | 1.4% | 0.000 eV |
| 2 | LiZn2Ni | 0.804 | -0.657 eV | 1.5% | 0.116 eV |
| 3 | Li2AlAg | 0.613 | -0.597 eV | 1.1% | 1.101 eV |

## 参考文献

- CHGNet: Deng, B. et al. *CHGNet as a universal neural network potential for charge-informed atomistic modelling.* Nat. Mach. Intell. 5, 1031–1041 (2023).
- Materials Project: Jain, A. et al. *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation.* APL Mater. 1, 011002 (2013).

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
├── pyproject.toml              ← 项目元数据 & 依赖
├── configs/
│   └── default.yaml            ← 默认运行配置
├── src/
│   ├── main.py                 ← CLI 入口
│   └── seed_layer/             ← 核心包
│       ├── config.py           ← 配置加载
│       ├── pipeline.py         ← 流水线编排
│       ├── io.py               ← I/O 工具
│       ├── reporting.py        ← 报告生成
│       ├── calculators/        ← 势函数抽象层
│       │   ├── base.py         ← Calculator 基类
│       │   └── chgnet.py       ← CHGNet 实现
│       └── steps/              ← 筛选步骤
│           ├── base.py         ← Step 基类
│           ├── stability.py    ← Step 2: 电化学稳定性
│           ├── lattice.py      ← Step 3: 晶格匹配
│           ├── adsorption.py   ← Step 4: 吸附能
│           └── diffusion.py    ← Step 5: NEB 扩散
├── data/                       ← 示例材料列表
│   ├── test_materials.txt
│   └── sample_materials.txt
├── tests/                      ← 单元测试
└── output/                     ← 运行产出（不纳入版本控制）
```

## 快速开始

### 环境要求

- Python 3.10+
- conda 环境 `materials_searching`（含 pymatgen, chgnet, mp-api, ASE, pandas）
- Materials Project API Key

### 安装

```bash
cd seed-layer-screening
pip install -e ".[dev]"
```

### 运行

使用默认配置：

```bash
# 设置 API Key
export MP_API_KEY=<your_key>

# 运行
python src/main.py --config configs/default.yaml
```

指定标签（输出目录命名为 `output/<tag>/`）：

```bash
python src/main.py --config configs/default.yaml --tag trial1
```

指定材料列表：

```bash
python src/main.py --config configs/default.yaml --materials data/test_materials.txt
```

跳过 NEB 扩散计算：

```bash
python src/main.py --config configs/default.yaml --skip-neb
```

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件路径（默认 `configs/default.yaml`） |
| `--tag` | 输出目录标签 |
| `--materials` | 自定义材料列表文件（每行一个 mp-id） |
| `--skip-neb` | 跳过 Step 5 NEB 计算 |
| `--resume` | 断点续跑（预留） |
| `--demo` | 离线演示模式（预留） |

### 配置文件

所有运行参数通过 YAML 配置文件控制，包括 API 设置、筛选阈值、计算参数、打分权重等。详见 [参数说明文档（中文）](docs/parameters_cn.md)。

示例配置见 `configs/default.yaml`。

## 架构设计

项目采用模块化架构，核心设计原则：

- **Step 抽象**：每个筛选步骤继承 `StepBase`，实现 `run()` 和 `skip()` 接口，支持独立测试和灵活组合
- **Calculator 抽象**：势函数通过 `CalculatorBase` 解耦，可轻松替换为 MACE、M3GNet 等其他 ML 势
- **YAML 配置驱动**：所有参数集中在配置文件中，便于复现和对比实验
- **Pipeline 编排**：`SeedLayerPipeline` 按顺序执行各 Step，处理数据传递和断点逻辑

## 测试

```bash
pytest tests/ -v
```

## 当前结果

10 个候选材料 -> 3 个通过稳定性筛选：

| 排名 | 材料 | 综合得分 | 吸附能 | 晶格失配 | 扩散势垒 |
|------|------|---------|--------|---------|---------|
| 1 | LiAl2Ni | 0.892 | -0.608 eV | 1.4% | 0.000 eV |
| 2 | LiZn2Ni | 0.804 | -0.657 eV | 1.5% | 0.116 eV |
| 3 | Li2AlAg | 0.613 | -0.597 eV | 1.1% | 1.101 eV |

## 参考文献

- CHGNet: Deng, B. et al. *CHGNet as a universal neural network potential for charge-informed atomistic modelling.* Nat. Mach. Intell. 5, 1031-1041 (2023).
- Materials Project: Jain, A. et al. *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation.* APL Mater. 1, 011002 (2013).

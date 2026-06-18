# 基于机器学习势的高通量锂金属电池种子层材料筛选

## 项目简介

锂金属负极被认为是下一代高能量密度电池的理想选择，但枝晶生长问题严重制约其实际应用。本项目通过**高通量计算筛选**，寻找适合作为"种子层"的材料——在锂表面沉积一层薄层，引导锂均匀沉积、抑制枝晶生长。

筛选流程基于 Materials Project 数据库，结合 **MACE-MPA-0 机器学习势**和 **CI-NEB 扩散计算**，从数千种候选材料中逐步筛选出最优种子层材料。支持 Li、Zn、Mg、Na 等多种工作离子，改一行配置即可切换。

## 筛选流程

```
Step 1  材料池初筛    → 从 MP 数据库拉取，过滤有害元素，保留二/三元化合物
Step 2  电化学稳定性  → 相图法检查与工作离子接触是否反应
Step 3  晶格匹配度    → 各低指数面 vs 参考金属，失配 < 8%
Step 3.5 界面能       → 种子层-金属-种子层三明治，逐层外推法
Step 4  吸附能        → 构建 slab，放工作离子，弛豫算能量差
Step 5  CI-NEB 扩散   → 2×2 超胞，7 image，BFGS 优化扩散势垒
```

每步层层筛选，下一层只计算上一层达标的材料。

## 评分体系

综合评分由 5 个指标加权计算：

| 指标 | 权重 | 公式 |
|------|------|------|
| S_adsorption | 25% | exp(-((E_ads + 0.5)²) / 0.08) |
| S_diffusion | 25% | max(0, 1.0 - barrier / 1.0) |
| S_interface | 25% | exp(-γ / 0.05) |
| S_lattice | 15% | max(0, 1.0 - mismatch / max_mismatch) |
| S_stability | 10% | exp(-e_above_hull / 0.05) |

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
│       ├── config.py           ← 配置加载（支持 ${ENV_VAR}）
│       ├── pipeline.py         ← 流水线编排
│       ├── io.py               ← I/O 工具
│       ├── reporting.py        ← 报告生成 + 评分
│       ├── calculators/        ← 势函数抽象层
│       │   ├── base.py         ← Calculator 基类
│       │   └── mace.py         ← MACE-MPA-0 实现
│       └── steps/              ← 筛选步骤
│           ├── base.py         ← Step 基类（含 build_ref_structure）
│           ├── stability.py    ← Step 2: 电化学稳定性
│           ├── lattice.py      ← Step 3: 晶格匹配
│           ├── interface.py    ← Step 3.5: 界面能
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
- conda 环境 `claude-code`（含 pymatgen, mace-torch, mp-api, ASE, pandas）
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

指定工作离子（如锌）：

```yaml
# configs/zinc.yaml
working_ion: Zn
```

```bash
python src/main.py --config configs/zinc.yaml
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

- **Step 抽象**：每个筛选步骤继承 `BaseStep`，实现 `run()` 接口，支持独立测试和灵活组合
- **Calculator 抽象**：势函数通过 `CalculatorBase` 解耦，可轻松替换为 MACE、CHGNet 等其他 ML 势
- **YAML 配置驱动**：所有参数集中在配置文件中，支持 `${ENV_VAR}` 环境变量替换
- **Pipeline 编排**：`SeedLayerPipeline` 按顺序执行各 Step，处理数据传递和断点逻辑
- **多工作离子支持**：改 `working_ion` 配置即可切换 Li/Zn/Mg/Na 等，参考金属自动从 MP 拉取

## 测试

```bash
pytest tests/ -v
```

## 参考文献

- MACE: Batatia, I. et al. *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields.* NeurIPS (2022).
- MACE-MPA-0: Batatia, I. et al. *A foundation model for atomistic simulation.* arXiv:2401.00096 (2024).
- Materials Project: Jain, A. et al. *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation.* APL Mater. 1, 011002 (2013).

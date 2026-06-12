# Seed Layer Screening

高通量筛选锂金属电池种子层材料的计算项目。

## Output7

- **完成时间**: 2026-06-06
- **材料总数**: 22,325 个（全流程筛选）
- **最终排名**: 1,552 个材料
- **Top1材料**: HfAl3 (mp-568718), 得分 0.9978

### 评分维度

| 维度 | 说明 |
|------|------|
| S_lattice | 晶格匹配度 |
| S_adsorption | 吸附强度 |
| S_diffusion | 扩散势垒 |
| score | 综合得分 |

### 文件结构

```
output7/
├── 原材料池/
│   ├── run_config.json
│   └── step1_materials_pool.csv
├── 计算结果/
│   ├── step2_stability.csv
│   ├── step3_lattice_match.csv
│   ├── step4_adsorption.csv
│   └── step5_diffusion.csv
├── 评分/
│   ├── final_ranking.csv
│   ├── screening_summary.csv
│   └── final_report.txt
├── seed_layer.py
└── run_wrapper.py
```

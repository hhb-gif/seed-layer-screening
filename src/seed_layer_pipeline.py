#!/usr/bin/env python3
"""
锂金属电池种子层材料高通量筛选 — 一站式 Pipeline
=================================================
一个文件跑完全流程：
  材料池获取 → 电化学稳定性 → 晶格失配度 → CHGNet 吸附能 → CI-NEB 扩散势垒 → 打分排名

用法：
  python seed_layer_pipeline.py                        # 从头跑
  python seed_layer_pipeline.py --resume               # 断点续跑
  python seed_layer_pipeline.py --skip-neb             # 跳过扩散计算
  python seed_layer_pipeline.py --materials my.txt     # 自定义材料列表

依赖：chgnet, pymatgen, mp-api, ase, pandas, numpy, scipy, matplotlib
"""

import os
import sys
import csv
import re
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

# ── 材料科学库 ──
from mp_api.client import MPRester
from pymatgen.core import Structure, Composition, Lattice
from pymatgen.core.surface import SlabGenerator, generate_all_slabs
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

# ── CHGNet ──
from chgnet.model import StructOptimizer
from chgnet.model.model import CHGNet

# ── ASE ──
from ase import Atoms
from ase.io import write
from ase.constraints import FixAtoms
from ase.mep import NEB as ASE_NEB, SingleCalculatorNEB
from ase.optimize import BFGS

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════╗
# ║                   参数配置（改这里就行）                        ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class PipelineConfig:
    """所有可调参数集中管理。改数字就行，不用翻代码。"""

    # ── API ──
    mp_api_key: str = ""                        # MP_API_KEY 环境变量，或直接填
    output_dir: str = "output"                  # 所有结果放这里

    # ── 材料池初筛 ──
    energy_above_hull_max: float = 0.05         # eV/atom。凸包上能量，越小越稳定
    # 要排除的元素（贵金属、有毒、放射性、稀土）
    elements_to_exclude: Tuple[str, ...] = (
        "Pt", "Pd", "Rh", "Ir", "Ru", "Os", "Re", "Tc",   # 贵金属
        "Hg", "Pb", "Cd", "As", "Tl", "Be",                # 有毒
        "U", "Th", "Pu", "Np", "Am", "Cm", "Ra", "Po",    # 放射性
        "In", "Ga", "Ge", "Sb", "Bi",                      # 昂贵/稀有
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",   # 稀土
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Sc", "Y",
    )

    # ── 晶格匹配 ──
    li_miller: Tuple[int, int, int] = (1, 1, 0)  # Li 参考表面
    max_mismatch: float = 8.0                     # 最大允许失配度 (%)
    angle_tolerance: float = 180.0               # 不限制角度（允许旋转匹配），仅长度匹配
    max_scale: int = 8                            # 扩胞倍数上限

    # ── 表面模型 ──
    vacuum: float = 15.0                          # 真空层厚度 (Å)
    slab_thickness: float = 12.0                  # slab 最小厚度 (Å)
    default_miller: Tuple[int, int, int] = (0, 0, 1)  # 默认晶面
    supercell: Tuple[int, int, int] = (2, 2, 1)   # 超胞尺寸

    # ── CHGNet 弛豫 ──
    fmax_bulk: float = 0.05                       # 体相力收敛 (eV/Å)
    fmax_slab: float = 0.10                       # slab 力收敛
    fmax_adsorb: float = 0.05                     # 吸附体系力收敛
    steps_bulk: int = 500                         # 体相最大步数
    steps_slab: int = 200                         # slab 最大步数
    steps_adsorb: int = 250                       # 吸附最大步数

    # ── 吸附能 ──
    coverages: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)  # 覆盖度 (ML)
    adsorption_min: float = -0.8                  # 吸附能下限 (eV)
    adsorption_max: float = -0.2                  # 吸附能上限 (eV)
    adsorbate_height: float = 1.8                 # Li 初始高度 (Å)
    li_area_per_atom: float = 8.0                 # Li 原子面积估计 (Å²)

    # ── NEB 扩散 ──
    neb_n_images: int = 7                         # NEB 图像数
    neb_fmax: float = 0.10                        # NEB 收敛标准 (eV/Å)
    neb_steps: int = 200                          # NEB 最大步数
    neb_climb: bool = True                        # 是否使用 climbing image
    li_displacement_min: float = 0.2              # 最小有效锂位移 (Å)
    neb_top_n: int = 30                           # 只对打分最高的 N 个材料跑 NEB

    # ── 打分权重 ──
    w_lattice: float = 0.3                        # 晶格匹配权重
    w_adsorption: float = 0.4                     # 吸附能权重
    w_diffusion: float = 0.3                      # 扩散势垒权重

    # ── 内部用（不需要改）──
    _adaptor: AseAtomsAdaptor = field(default_factory=AseAtomsAdaptor, repr=False)
    _relaxer: Optional[StructOptimizer] = field(default=None, repr=False)
    _li_ref_energy: Optional[float] = field(default=None, repr=False)


# ╔══════════════════════════════════════════════════════════════╗
# ║                     工具函数                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def count_elements(formula: str) -> int:
    """计算化学式中不同元素的种类数。如 LiCoO2 → 3"""
    return len(set(re.findall(r'[A-Z][a-z]?', formula)))


def load_materials_file(filepath: str) -> List[str]:
    """从文件加载 mp-id 列表。支持 # 注释，自动处理编码。"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"材料文件不存在: {filepath}")

    for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
        try:
            with open(path, 'r', encoding=encoding) as f:
                materials = []
                for line in f:
                    line = line.split('#')[0].strip()
                    if line.startswith('mp-'):
                        materials.append(line)
                if materials:
                    return list(dict.fromkeys(materials))  # 去重保序
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件: {filepath}")


def save_checkpoint(df: pd.DataFrame, filepath: str, dedup_cols: list = None):
    """追加模式保存 checkpoint。dedup_cols 指定去重列，默认 ['material_id']。"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
        if dedup_cols is not None:
            df = df.drop_duplicates(subset=dedup_cols, keep='last')
        elif 'material_id' in df.columns:
            # 只有单列主键时才按 material_id 去重
            pass  # 不自动去重，由调用方决定
    df.to_csv(path, index=False)


def load_checkpoint(filepath: str) -> pd.DataFrame:
    """读取 checkpoint，不存在则返回空 DataFrame。"""
    path = Path(filepath)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def completed_ids(filepath: str, col: str = 'material_id') -> set:
    """读取已完成 checkpoint 里的 material_id 集合。"""
    df = load_checkpoint(filepath)
    if df.empty or col not in df.columns:
        return set()
    return set(df[col].dropna().tolist())


def print_step_header(step: int, total: int, name: str, current: int, total_items: int):
    """打印步骤进度头。"""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  [{step}/{total}] {name}  —  进度 {current}/{total_items}")
    print(f"{bar}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                     主 Pipeline 类                           ║
# ╚══════════════════════════════════════════════════════════════╝

class SeedLayerPipeline:
    """锂金属电池种子层材料高通量筛选"""

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.output = Path(config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)

        # API Key
        api_key = config.mp_api_key or os.environ.get("MP_API_KEY", "")
        if not api_key:
            print("⚠ 未设置 MP_API_KEY。请设置环境变量或在 config 中填写。")
            print("  免费注册: https://materialsproject.org/api")
            api_key = "PLACEHOLDER"
        self.api_key = api_key

        # CHGNet 只初始化一次
        print("正在加载 CHGNet 模型（仅一次）...")
        self.cfg._relaxer = StructOptimizer()
        print("  CHGNet 加载完成。")

    # ── Step 1: 获取候选材料池 ──────────────────────────────────

    def step1_fetch_materials(self, materials_file: str = None) -> pd.DataFrame:
        """
        Step 1: 获取候选材料池。
        如果提供了 materials_file，直接加载；否则从 MP API 拉取。
        """
        ckpt = self.output / "step1_materials_pool.csv"
        step_name = "材料池获取"

        # 如果指定了材料文件，直接从文件加载
        if materials_file:
            mp_ids = load_materials_file(materials_file)
            print(f"\n从文件加载 {len(mp_ids)} 个材料: {materials_file}")
            results = []
            with MPRester(self.api_key) as mpr:
                for i, mp_id in enumerate(mp_ids, 1):
                    print(f"  获取材料信息 ({i}/{len(mp_ids)}): {mp_id}")
                    try:
                        structure = mpr.get_structure_by_material_id(mp_id)
                        formula = structure.composition.reduced_formula
                        results.append({
                            'material_id': mp_id,
                            'formula': formula,
                            'n_elements': count_elements(formula),
                        })
                    except Exception as e:
                        print(f"    获取失败: {e}")
                        results.append({
                            'material_id': mp_id,
                            'formula': '',
                            'n_elements': 0,
                        })
            df = pd.DataFrame(results)
        else:
            # 从 MP API 大范围搜索
            done = completed_ids(ckpt)
            if done:
                print(f"\n[Step 1] 断点续跑：已有 {len(done)} 个材料，跳过。")
                return load_checkpoint(ckpt)

            print("\n[Step 1] 从 Materials Project 搜索候选材料...")
            print(f"  条件: energy_above_hull < {self.cfg.energy_above_hull_max} eV")

            with MPRester(self.api_key) as mpr:
                docs = mpr.summary.search(
                    energy_above_hull=(0, self.cfg.energy_above_hull_max),
                    fields=["material_id", "formula_pretty", "energy_above_hull"]
                )
                df = pd.DataFrame([{
                    'material_id': d.material_id,
                    'formula': d.formula_pretty,
                    'energy_above_hull': d.energy_above_hull,
                    'n_elements': count_elements(d.formula_pretty),
                } for d in docs])

            print(f"  MP 返回 {len(df)} 个材料")

        # 过滤有害元素
        def has_excluded_element(formula: str) -> bool:
            for elem in self.cfg.elements_to_exclude:
                if len(elem) == 1:
                    if re.search(rf'(?<![A-Za-z]){elem}(?![a-z])', formula):
                        return True
                elif elem in formula:
                    return True
            return False

        before = len(df)
        df = df[~df['formula'].apply(has_excluded_element)]
        print(f"  去除有害元素后: {len(df)} (过滤 {before - len(df)} 个)")

        # 只保留二元和三元化合物
        df = df[df['n_elements'].isin([2, 3])]
        print(f"  保留二/三元化合物: {len(df)}")

        # 如果来自文件，不需要 energy_above_hull 列
        if 'energy_above_hull' not in df.columns:
            df['energy_above_hull'] = None

        save_checkpoint(df, ckpt)
        print(f"  ✓ Step 1 完成，{len(df)} 个候选材料 → {ckpt}")
        return df

    # ── Step 2: 电化学稳定性 ────────────────────────────────────

    def step2_stability(self) -> pd.DataFrame:
        """
        Step 2: 电化学稳定性筛选。
        用相图法判断材料与锂接触是否反应。
        """
        ckpt = self.output / "step2_stability.csv"
        step_name = "电化学稳定性"

        # 读上游结果
        df_in = load_checkpoint(self.output / "step1_materials_pool.csv")
        if df_in.empty:
            print("[Step 2] 没有上游数据，请先运行 step1")
            return pd.DataFrame()

        mp_ids = df_in['material_id'].tolist()
        done = completed_ids(ckpt)
        remaining = [m for m in mp_ids if m not in done]
        total = len(mp_ids)

        print(f"\n[Step 2] {step_name}: {len(remaining)} 待处理 / {total} 总计")

        if not remaining:
            return load_checkpoint(ckpt)

        with MPRester(self.api_key) as mpr:
            # 获取 Li 参考
            print("  获取 Li 参考...")
            li_entries = mpr.get_entries_in_chemsys(["Li"])
            li_entry = sorted(li_entries, key=lambda e: e.energy_per_atom)[0]
            e_li = li_entry.energy_per_atom

            for i, mp_id in enumerate(remaining, 1):
                print_step_header(2, 5, step_name, i, len(remaining))
                try:
                    passed, detail = self._check_stability(mp_id, mpr, e_li)
                except Exception as e:
                    passed, detail = False, f"错误: {e}"

                row = pd.DataFrame([{
                    'material_id': mp_id,
                    'formula': df_in[df_in['material_id']==mp_id]['formula'].values[0],
                    'passed': passed,
                    'min_dE_eV': detail if isinstance(detail, float) else None,
                    'details': str(detail),
                }])
                save_checkpoint(row, ckpt)

        df_out = load_checkpoint(ckpt)
        passed = len(df_out[df_out['passed'] == True])
        print(f"\n  ✓ Step 2 完成: {passed}/{len(df_out)} 通过稳定性筛选")
        return df_out

    def _check_stability(self, mp_id: str, mpr, e_li: float) -> Tuple[bool, str]:
        """判断单个材料与 Li 是否反应（内部函数）。"""
        # 获取材料 entry
        try:
            docs = mpr.summary.search(material_ids=[mp_id],
                                      fields=["material_id", "elements", "formula_pretty"])
        except AttributeError:
            docs = mpr.materials.summary.search(material_ids=[mp_id],
                                      fields=["material_id", "elements", "formula_pretty"])
        if not docs:
            return False, "MP 中未找到"

        elements = [el.symbol for el in docs[0].elements]
        formula = docs[0].formula_pretty
        chemsys = "-".join(sorted(elements))

        entries = mpr.get_entries_in_chemsys(chemsys)
        target = [e for e in entries if e.entry_id == mp_id]
        if not target:
            target = [e for e in entries if e.composition.reduced_formula == formula]
        if not target:
            return False, "未找到 entry"
        mat_entry = sorted(target, key=lambda e: e.energy_per_atom)[0]
        e_mat = mat_entry.energy_per_atom
        comp_mat = mat_entry.composition

        # 构建含 Li 的相图
        all_elements = list(set(elements + ["Li"]))
        full_chemsys = "-".join(sorted(all_elements))
        all_entries = mpr.get_entries_in_chemsys(full_chemsys)
        pd_phase = PhaseDiagram(all_entries)

        # 检查材料本身是否稳定
        stable_ids = {e.entry_id for e in pd_phase.stable_entries}
        if mat_entry.entry_id not in stable_ids:
            return False, "材料本身不稳定（不在凸包上）"

        # 扫描材料→Li 成分连线
        frac = comp_mat.fractional_composition
        min_dE = 0.0
        n_steps = 100
        for j in range(1, n_steps):
            t = j / n_steps
            comp_dict = {el.symbol: (1-t)*frac for el, frac in frac.items()}
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
            return False, f"与 Li 反应: min dE = {min_dE:.3f} eV/atom"
        return True, f"稳定: min dE = {min_dE:.3f} eV/atom"

    # ── Step 3: 晶格失配度 ─────────────────────────────────────

    def step3_lattice_match(self) -> pd.DataFrame:
        """
        Step 3: 晶格失配度筛选。
        计算材料各低指数面与 Li(110) 面的晶格失配度。
        """
        ckpt = self.output / "step3_lattice_match.csv"
        step_name = "晶格失配度"

        # 只取通过稳定性的材料
        df_stable = load_checkpoint(self.output / "step2_stability.csv")
        if df_stable.empty or 'passed' not in df_stable.columns:
            print("[Step 3] 没有稳定性数据")
            return pd.DataFrame()
        df_stable = df_stable[df_stable['passed'] == True]
        if df_stable.empty:
            print("[Step 3] 没有通过稳定性的材料")
            return pd.DataFrame()

        mp_ids = df_stable['material_id'].tolist()
        done = completed_ids(ckpt)
        remaining = [m for m in mp_ids if m not in done]

        print(f"\n[Step 3] {step_name}: {len(remaining)} 待处理 / {len(mp_ids)} 总计")

        if not remaining:
            return load_checkpoint(ckpt)

        with MPRester(self.api_key) as mpr:
            # 构建 Li(110) slab（不弛豫，用标准 bcc Li）
            print("  构建 Li(110) 参考表面...")
            li_entries = mpr.get_entries_in_chemsys(["Li"])
            li_struct = sorted(li_entries, key=lambda e: e.energy_per_atom)[0].structure
            # 注意：不对 Li 做 CHGNet 弛豫，bcc 单原子胞会被过度变形
            slabgen = SlabGenerator(
                li_struct, self.cfg.li_miller,
                self.cfg.slab_thickness, self.cfg.vacuum,
                center_slab=True, primitive=True
            )
            li_slab = slabgen.get_slabs()[0]

            for i, mp_id in enumerate(remaining, 1):
                print_step_header(3, 5, step_name, i, len(remaining))
                try:
                    rows = self._calc_lattice_mismatch(mp_id, li_slab, mpr)
                except Exception as e:
                    rows = [{'material_id': mp_id, 'miller': '(0,0,0)',
                             'mismatch_pct': 999.0, 'details': f"错误: {e}"}]
                save_checkpoint(pd.DataFrame(rows), ckpt)

        df_out = load_checkpoint(ckpt)
        # 只统计有效数据
        valid = df_out[df_out['mismatch_pct'] < 900]
        passed = len(valid[valid['mismatch_pct'] < self.cfg.max_mismatch])
        print(f"\n  ✓ Step 3 完成: {passed} 个材料通过晶格匹配")
        return df_out

    def _calc_lattice_mismatch(self, mp_id: str, li_slab, mpr) -> List[Dict]:
        """计算单个材料各表面的晶格失配度（内部函数）。"""
        # 获取结构并弛豫体相
        structure = mpr.get_structure_by_material_id(mp_id)
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
        result = self.cfg._relaxer.relax(
            structure, fmax=self.cfg.fmax_bulk, steps=self.cfg.steps_bulk,
            relax_cell=True, verbose=False
        )
        structure = result["final_structure"]

        # 生成所有低指数面
        try:
            slabs = generate_all_slabs(
                structure, max_index=1,
                min_slab_size=self.cfg.slab_thickness,
                min_vacuum_size=self.cfg.vacuum,
                center_slab=True, primitive=True
            )
        except Exception as e:
            return [{'material_id': mp_id, 'miller': '(0,0,0)',
                     'mismatch_pct': 999.0, 'details': f"Slab生成失败: {e}"}]

        results = []
        for slab in slabs:
            miller = slab.miller_index
            mismatch, info = self._compute_mismatch(li_slab.lattice, slab.lattice)
            results.append({
                'material_id': mp_id,
                'miller': str(miller),
                'mismatch_pct': round(mismatch, 3) if mismatch < 900 else 999.0,
                'mismatch_a_pct': round(info.get('mismatch_a', 0), 3) if info else 999.0,
                'mismatch_b_pct': round(info.get('mismatch_b', 0), 3) if info else 999.0,
                'angle_diff_deg': round(info.get('angle_diff', 0), 2) if info else 0,
                'details': (f"film({info['film_scale_a']},{info['film_scale_b']}) "
                           f"Li({info['li_scale_a']},{info['li_scale_b']})") if info else "无匹配"
            })
        return results

    def _compute_mismatch(self, li_lattice, film_lattice) -> Tuple[float, Optional[Dict]]:
        """计算两个二维晶格的最小失配度（含角度匹配）。"""
        a_f = np.array(film_lattice.matrix[0][:2])
        b_f = np.array(film_lattice.matrix[1][:2])
        a_li = np.array(li_lattice.matrix[0][:2])
        b_li = np.array(li_lattice.matrix[1][:2])

        len_a_f, len_b_f = np.linalg.norm(a_f), np.linalg.norm(b_f)
        len_a_li, len_b_li = np.linalg.norm(a_li), np.linalg.norm(b_li)

        # 角度差
        def get_angle(v1, v2):
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        angle_diff = abs(get_angle(a_f, b_f) - get_angle(a_li, b_li))

        best = float('inf')
        best_info = None
        for n1 in range(1, self.cfg.max_scale + 1):
            for m1 in range(1, self.cfg.max_scale + 1):
                ma = abs(n1 * len_a_f - m1 * len_a_li) / (m1 * len_a_li) * 100
                if ma > best:
                    continue
                for n2 in range(1, self.cfg.max_scale + 1):
                    for m2 in range(1, self.cfg.max_scale + 1):
                        mb = abs(n2 * len_b_f - m2 * len_b_li) / (m2 * len_b_li) * 100
                        max_mm = max(ma, mb)
                        if max_mm < best and angle_diff <= self.cfg.angle_tolerance:
                            best = max_mm
                            best_info = {
                                'mismatch_a': ma, 'mismatch_b': mb,
                                'film_scale_a': n1, 'film_scale_b': n2,
                                'li_scale_a': m1, 'li_scale_b': m2,
                                'angle_diff': angle_diff,
                            }
        if best == float('inf'):
            return 999.0, None
        return best, best_info

    # ── Step 4: CHGNet 吸附能 ──────────────────────────────────

    def step4_adsorption(self, skip_lattice: bool = False) -> pd.DataFrame:
        """
        Step 4: CHGNet 吸附能计算。
        skip_lattice=True 时跳过晶格匹配，稳定性通过即用默认 (001) 面。
        """
        ckpt = self.output / "step4_adsorption.csv"
        step_name = "吸附能计算"

        if skip_lattice:
            # 直接从稳定性结果取材料，用默认晶面
            df_in = load_checkpoint(self.output / "step2_stability.csv")
            if df_in.empty or 'passed' not in df_in.columns:
                print("[Step 4] 没有稳定性数据")
                return pd.DataFrame()
            mp_ids = df_in[df_in['passed'] == True]['material_id'].tolist()
            miller_map = {m: self.cfg.default_miller for m in mp_ids}
        else:
            # 取晶格匹配通过的材料（mismatch < 8%），每人取最佳面
            df_in = load_checkpoint(self.output / "step3_lattice_match.csv")
            if df_in.empty or 'mismatch_pct' not in df_in.columns:
                print("[Step 4] 没有晶格匹配数据")
                return pd.DataFrame()
            df_in = df_in[df_in['mismatch_pct'] < self.cfg.max_mismatch]
            if df_in.empty:
                print("[Step 4] 没有通过晶格匹配的材料")
                return pd.DataFrame()
            best = df_in.loc[df_in.groupby('material_id')['mismatch_pct'].idxmin()]
            mp_ids = best['material_id'].tolist()
            miller_map = {}
            for _, row in best.iterrows():
                miller_map[row['material_id']] = eval(row['miller'])

        if not mp_ids:
            return pd.DataFrame()

        done = completed_ids(ckpt)
        remaining = [m for m in mp_ids if m not in done]
        print(f"\n[Step 4] {step_name}: {len(remaining)} 待处理 / {len(mp_ids)} 总计")

        if not remaining:
            return load_checkpoint(ckpt)

        # 获取 Li 参考能量（只算一次）
        if self.cfg._li_ref_energy is None:
            self.cfg._li_ref_energy = self._get_li_reference_energy()

        with MPRester(self.api_key) as mpr:
            for i, mp_id in enumerate(remaining, 1):
                print_step_header(4, 5, step_name, i, len(remaining))
                miller = miller_map.get(mp_id, self.cfg.default_miller)
                try:
                    rows = self._calc_adsorption(mp_id, miller, mpr)
                except Exception as e:
                    rows = [{'material_id': mp_id, 'coverage_ML': 0,
                             'E_ads_eV': None, 'status': f'错误: {e}'}]
                save_checkpoint(pd.DataFrame(rows), ckpt)

        df_out = load_checkpoint(ckpt)
        print(f"\n  ✓ Step 4 完成: {len(df_out)} 条吸附能记录")
        return df_out

    def _get_li_reference_energy(self) -> float:
        """计算 Li 参考能量（bcc Li 体相弛豫）。"""
        from pymatgen.core import Structure, Lattice
        li_bcc = Structure(
            Lattice.cubic(3.49),
            ["Li"], [[0, 0, 0]]
        )
        result = self.cfg._relaxer.relax(
            li_bcc, fmax=self.cfg.fmax_bulk, steps=self.cfg.steps_bulk,
            relax_cell=True, verbose=False
        )
        energy = result["trajectory"].energies[-1]
        n_atoms = len(result["final_structure"])
        return energy / n_atoms

    def _calc_adsorption(self, mp_id: str, miller: tuple, mpr) -> List[Dict]:
        """计算单个材料的多覆盖度吸附能（内部函数）。"""
        # 1. 获取体相并弛豫
        structure = mpr.get_structure_by_material_id(mp_id)
        result_bulk = self.cfg._relaxer.relax(
            structure, fmax=self.cfg.fmax_bulk, steps=self.cfg.steps_bulk,
            relax_cell=True, verbose=False
        )
        bulk = result_bulk["final_structure"]

        # 2. 生成 slab
        slabgen = SlabGenerator(
            bulk, miller, self.cfg.slab_thickness, self.cfg.vacuum,
            center_slab=True, primitive=False
        )
        slab = slabgen.get_slabs()[0]

        # 3. 扩胞
        from ase.build import make_supercell
        ase_slab = self.cfg._adaptor.get_atoms(slab)
        P = [[self.cfg.supercell[0], 0, 0],
             [0, self.cfg.supercell[1], 0],
             [0, 0, self.cfg.supercell[2]]]
        ase_slab = make_supercell(ase_slab, P)

        # 4. 固定底部原子
        z_vals = ase_slab.get_positions()[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        threshold = z_min + (z_max - z_min) * 0.33
        fixed = [i for i, z in enumerate(z_vals) if z < threshold]
        if len(fixed) < 4:
            fixed = list(np.argsort(z_vals)[:max(4, len(ase_slab)//4)])
        ase_slab.set_constraint(FixAtoms(indices=fixed))

        # 5. 弛豫清洁表面
        slab_pmg = self.cfg._adaptor.get_structure(ase_slab)
        result_clean = self.cfg._relaxer.relax(
            slab_pmg, fmax=self.cfg.fmax_slab, steps=self.cfg.steps_slab,
            relax_cell=False, verbose=False
        )
        clean_struct = result_clean["final_structure"]
        e_clean = result_clean["trajectory"].energies[-1]
        clean_atoms = self.cfg._adaptor.get_atoms(clean_struct)
        clean_atoms.set_constraint(FixAtoms(indices=fixed))

        # 6. 获取表面高度和面积
        z_all = clean_atoms.get_positions()[:, 2]
        movable_z = [z_all[i] for i in range(len(clean_atoms)) if i not in fixed]
        surface_z = np.percentile(movable_z, 90) if len(movable_z) > 10 else max(movable_z)

        cell = clean_atoms.get_cell()
        area = abs(cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0])

        # 7. 逐覆盖度计算吸附能
        results = []
        for cov in self.cfg.coverages:
            n_li = max(1, int(area / self.cfg.li_area_per_atom * cov))
            if cov == 0:
                continue

            # 放置 Li 原子（网格排列）
            ads_atoms = clean_atoms.copy()
            nx = int(np.sqrt(n_li * cell[1, 1] / cell[0, 0])) + 1
            ny = int(n_li / nx) + 1
            dx, dy = cell[0, 0] / (nx + 1), cell[1, 1] / (ny + 1)
            added = 0
            for ix in range(1, nx + 1):
                for iy in range(1, ny + 1):
                    if added >= n_li:
                        break
                    ads_atoms.append('Li')
                    ads_atoms.positions[-1] = [ix * dx, iy * dy, surface_z + self.cfg.adsorbate_height]
                    added += 1
                if added >= n_li:
                    break
            ads_atoms.set_constraint(FixAtoms(indices=fixed))

            # 弛豫吸附体系
            ads_pmg = self.cfg._adaptor.get_structure(ads_atoms)
            result_ads = self.cfg._relaxer.relax(
                ads_pmg, fmax=self.cfg.fmax_adsorb, steps=self.cfg.steps_adsorb,
                relax_cell=False, verbose=False
            )
            e_ads = result_ads["trajectory"].energies[-1]

            # 计算吸附能: E_ads = (E_slab+Li - E_clean - n_Li * E_Li_ref) / n_Li
            e_ads_per_li = (e_ads - e_clean - n_li * self.cfg._li_ref_energy) / n_li

            results.append({
                'material_id': mp_id,
                'formula': bulk.composition.reduced_formula,
                'coverage_ML': round(cov, 3),
                'n_Li': n_li,
                'E_clean_eV': round(e_clean, 3),
                'E_ads_system_eV': round(e_ads, 3),
                'E_ads_eV': round(e_ads_per_li, 4),
                'status': 'success',
            })

        return results

    # ── Step 5: CI-NEB 扩散势垒 ─────────────────────────────────

    def step5_diffusion(self) -> pd.DataFrame:
        """
        Step 5: CI-NEB 扩散势垒。
        只对打分 Top N 材料计算 Li 在表面的扩散势垒。
        """
        ckpt = self.output / "step5_diffusion.csv"
        step_name = "扩散势垒"

        # 先打分，取 Top N
        ranked = self._score_and_rank()
        if ranked.empty:
            print("[Step 5] 没有可评分的材料")
            return pd.DataFrame()

        top_materials = ranked.head(self.cfg.neb_top_n)
        mp_ids = top_materials['material_id'].tolist()
        done = completed_ids(ckpt)
        remaining = [m for m in mp_ids if m not in done]

        print(f"\n[Step 5] {step_name}: Top {self.cfg.neb_top_n} 中 {len(remaining)} 待处理")

        if not remaining:
            return load_checkpoint(ckpt)

        with MPRester(self.api_key) as mpr:
            for i, mp_id in enumerate(remaining, 1):
                print_step_header(5, 5, step_name, i, len(remaining))
                # 找该材料最佳表面
                df_mat = load_checkpoint(self.output / "step3_lattice_match.csv")
                df_mat = df_mat[df_mat['material_id'] == mp_id]
                if df_mat.empty:
                    continue
                best_row = df_mat.loc[df_mat['mismatch_pct'].idxmin()]
                miller = eval(best_row['miller'])
                try:
                    rows = self._calc_diffusion_one(mp_id, miller, mpr)
                except Exception as e:
                    rows = [{'material_id': mp_id, 'path': 'N/A',
                             'barrier_eV': None, 'status': f'错误: {e}'}]
                for r in rows:
                    row_df = pd.DataFrame([r])
                    save_checkpoint(row_df, ckpt)

        df_out = load_checkpoint(ckpt)
        print(f"\n  ✓ Step 5 完成: {len(df_out)} 条扩散记录")
        return df_out

    def _calc_diffusion_one(self, mp_id: str, miller: tuple, mpr) -> List[Dict]:
        """计算单个材料一个表面的 Li 扩散势垒（内部函数）。"""
        # 1. 获取结构并构建 slab（复用 step4 逻辑但不用超胞，NE 需要较小体系）
        structure = mpr.get_structure_by_material_id(mp_id)
        result_bulk = self.cfg._relaxer.relax(
            structure, fmax=self.cfg.fmax_bulk, steps=self.cfg.steps_bulk,
            relax_cell=True, verbose=False
        )
        bulk = result_bulk["final_structure"]

        slabgen = SlabGenerator(
            bulk, miller, self.cfg.slab_thickness, self.cfg.vacuum,
            center_slab=True, primitive=True
        )
        slab = slabgen.get_slabs()[0]
        slab.make_supercell([2, 2, 1])  # 2×2 超胞，位点间距翻倍
        ase_slab = self.cfg._adaptor.get_atoms(slab)

        # 固定底部
        z_vals = ase_slab.get_positions()[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        threshold = z_min + (z_max - z_min) * 0.33
        fixed = [i for i, z in enumerate(z_vals) if z < threshold]
        if len(fixed) < 2:
            fixed = list(np.argsort(z_vals)[:max(2, len(ase_slab)//3)])
        ase_slab.set_constraint(FixAtoms(indices=fixed))

        # 弛豫清洁表面
        slab_pmg = self.cfg._adaptor.get_structure(ase_slab)
        result_clean = self.cfg._relaxer.relax(
            slab_pmg, fmax=self.cfg.fmax_slab, steps=self.cfg.steps_slab,
            relax_cell=False, verbose=False
        )
        clean_slab = result_clean["final_structure"]

        # 2. 生成吸附位点
        asf = AdsorbateSiteFinder(clean_slab)
        ads_sites = asf.find_adsorption_sites(distance=self.cfg.adsorbate_height)

        # 手动补充缺失位点
        surface_z = max(s.coords[2] for s in clean_slab)
        surface_atoms = [s for s in clean_slab if abs(s.coords[2] - surface_z) < 1.0]
        manual_top = [a.coords + np.array([0, 0, self.cfg.adsorbate_height]) for a in surface_atoms]

        # 3. 构建位点列表
        top_sites = ads_sites.get('top', []) or manual_top
        bridge_sites = ads_sites.get('bridge', [])
        hollow_sites = ads_sites.get('hollow', [])

        # 4. 优化每个位点的一个 Li
        def relax_li_at_site(site):
            atoms = self.cfg._adaptor.get_atoms(clean_slab)
            atoms.append('Li')
            atoms.positions[-1] = site
            atoms.set_constraint(FixAtoms(indices=list(range(len(atoms)-1))))
            atoms.calc = self.cfg._relaxer.calculator
            dyn = BFGS(atoms, logfile=None)
            dyn.run(fmax=0.05, steps=100)
            return atoms

        opt = {}
        for stype, sites in [('top', top_sites), ('bridge', bridge_sites), ('hollow', hollow_sites)]:
            if sites:
                try:
                    opt[stype] = relax_li_at_site(sites[0])
                except Exception:
                    opt[stype] = None

        # 5. 构建路径
        paths = []
        if opt.get('top') and opt.get('bridge'):
            paths.append(('top_to_bridge', opt['top'], opt['bridge']))
        if opt.get('bridge') and opt.get('hollow'):
            paths.append(('bridge_to_hollow', opt['bridge'], opt['hollow']))
        if opt.get('hollow') and opt.get('top'):
            paths.append(('hollow_to_top', opt['hollow'], opt['top']))

        if not paths:
            return [{'material_id': mp_id, 'path': 'N/A',
                     'barrier_eV': None, 'status': '没有可计算路径'}]

        # 6. 跑 NEB
        results = []
        for path_name, init_atoms, final_atoms in paths:
            # 统一晶胞
            final_atoms.set_cell(init_atoms.get_cell(), scale_atoms=True)
            li_init = init_atoms[-1].position
            li_final = final_atoms[-1].position
            li_dist = np.linalg.norm(li_init - li_final)

            if li_dist < self.cfg.li_displacement_min:
                results.append({'material_id': mp_id, 'path': path_name,
                                'barrier_eV': None, 'status': f'Li位移过小({li_dist:.2f}Å)'})
                continue

            # 构建 NEB 图像（每个 image 必须有独立的 calculator）
            images = [init_atoms]
            for _ in range(self.cfg.neb_n_images - 2):
                img = init_atoms.copy()
                img.calc = self.cfg._relaxer.calculator
                images.append(img)
            images.append(final_atoms)

            neb = SingleCalculatorNEB(images, climb=self.cfg.neb_climb, method='improvedtangent')
            neb.interpolate(method='idpp')
            opt_neb = BFGS(neb, logfile=None)
            try:
                opt_neb.run(fmax=self.cfg.neb_fmax, steps=self.cfg.neb_steps)
                energies = [img.get_potential_energy() for img in images]
                barrier = max(energies) - energies[0]
                results.append({'material_id': mp_id, 'path': path_name,
                                'barrier_eV': round(barrier, 4), 'status': 'success'})
            except Exception as e:
                results.append({'material_id': mp_id, 'path': path_name,
                                'barrier_eV': None, 'status': f'NEB失败: {e}'})

        return results

    # ── 打分排名 ─────────────────────────────────────────────

    def _score_and_rank(self) -> pd.DataFrame:
        """
        根据前三步结果打分排序。
        稳定性不过 → 淘汰
        晶格失配 > 阈值 → 淘汰
        吸附能正常区间的按公式打分
        """
        # 读取各步骤结果
        df_ads = load_checkpoint(self.output / "step4_adsorption.csv")
        df_lat = load_checkpoint(self.output / "step3_lattice_match.csv")
        df_sta = load_checkpoint(self.output / "step2_stability.csv")

        if df_ads.empty:
            return pd.DataFrame()

        # 过滤：只取稳定性通过的材料
        stable_ids = set(df_sta[df_sta['passed']==True]['material_id'])
        valid_ids = stable_ids.copy()  # 默认所有人有效

        # 如果有晶格匹配数据，进一步过滤
        if not df_lat.empty and 'mismatch_pct' in df_lat.columns:
            df_lat_pass = df_lat[df_lat['mismatch_pct'] < self.cfg.max_mismatch]
            match_ids = set(df_lat_pass['material_id'])
            valid_ids &= match_ids
            # 每人取最佳晶格失配度
            best_lat = df_lat_pass.loc[df_lat_pass.groupby('material_id')['mismatch_pct'].idxmin()]
            best_lat = best_lat.set_index('material_id')['mismatch_pct']
        else:
            best_lat = pd.Series(dtype=float)  # 空 Series

        # 每人取各覆盖度平均吸附能，再取最接近 -0.5 eV 的
        df_ads_valid = df_ads[df_ads['material_id'].isin(valid_ids) & (df_ads['E_ads_eV'].notna())]
        if df_ads_valid.empty:
            return pd.DataFrame()

        # 取吸附能最接近 -0.5 eV 的覆盖度
        def best_ads_e(group):
            group = group.copy()
            group['dist'] = abs(group['E_ads_eV'] + 0.5)
            return group.loc[group['dist'].idxmin()]

        best_ads = df_ads_valid.groupby('material_id', group_keys=False).apply(best_ads_e)
        # groupby 后 material_id 已是索引，无需再 set_index

        # 构建打分表
        rows = []
        for mp_id in valid_ids:
            if mp_id not in best_ads.index:
                continue
            e_ads = best_ads.loc[mp_id, 'E_ads_eV']
            formula = best_ads.loc[mp_id, 'formula']

            # 晶格分：有数据就用，没有就跳过
            has_lattice = mp_id in best_lat.index
            S_adsorption = np.exp(-((e_ads + 0.5) ** 2) / 0.08)
            if has_lattice:
                mismatch = best_lat[mp_id]
                S_lattice = max(0, 1.0 - mismatch / self.cfg.max_mismatch)
                score = self.cfg.w_lattice * S_lattice + self.cfg.w_adsorption * S_adsorption
            else:
                mismatch = None
                S_lattice = None
                score = S_adsorption

            rows.append({
                'material_id': mp_id, 'formula': formula,
                'mismatch_pct': mismatch if has_lattice else None,
                'E_ads_eV': e_ads,
                'S_lattice': round(S_lattice, 4) if S_lattice is not None else None,
                'S_adsorption': round(S_adsorption, 4),
                'score': round(score, 4),
            })

        df_score = pd.DataFrame(rows).sort_values('score', ascending=False)
        return df_score

    def score_and_report(self):
        """生成最终排名和报告。"""
        ranked = self._score_and_rank()
        if ranked.empty:
            print("没有可排名的材料")
            return

        # 如果有扩散数据，补充分数
        df_neb = load_checkpoint(self.output / "step5_diffusion.csv")
        if not df_neb.empty:
            df_neb = df_neb[df_neb['barrier_eV'].notna()]
            neb_best = df_neb.loc[df_neb.groupby('material_id')['barrier_eV'].idxmin()]
            neb_best = neb_best.set_index('material_id')['barrier_eV']

            for i, row in ranked.iterrows():
                mp_id = row['material_id']
                if mp_id in neb_best.index:
                    barrier = neb_best[mp_id]
                    S_diff = max(0, 1.0 - barrier / 1.0)
                    ranked.at[i, 'barrier_eV'] = round(barrier, 4)
                    ranked.at[i, 'S_diffusion'] = round(S_diff, 4)
                    ranked.at[i, 'score'] = round(
                        row['score'] * (1 - self.cfg.w_diffusion) + self.cfg.w_diffusion * S_diff, 4
                    )

        # 保存
        ckpt = self.output / "final_ranking.csv"
        ranked.to_csv(ckpt, index=False)
        print(f"\n✓ 最终排名: {ckpt}")

        # 可读报告
        report = self.output / "final_report.txt"
        with open(report, 'w', encoding='utf-8') as f:
            f.write("锂金属电池种子层材料筛选 — 最终报告\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Top {min(20, len(ranked))} 候选材料:\n\n")
            for i, (_, row) in enumerate(ranked.head(20).iterrows(), 1):
                formula = row.get('formula', '') or ''
                f.write(f"  #{i:2d}  {row['material_id']:15s}  {str(formula):20s}  "
                        f"Score={row['score']:.4f}  "
                        f"E_ads={row['E_ads_eV']:.3f} eV")
                if row.get('mismatch_pct') and pd.notna(row['mismatch_pct']):
                    f.write(f"  Mismatch={row['mismatch_pct']:.1f}%")
                if row.get('barrier_eV') and pd.notna(row.get('barrier_eV')):
                    f.write(f"  Barrier={row['barrier_eV']:.3f} eV")
                f.write("\n")
        print(f"✓ 可读报告: {report}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                     命令行入口                                ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="锂金属电池种子层材料高通量筛选")
    parser.add_argument("--materials", type=str, default=None,
                        help="自定义材料列表文件（每行一个 mp-id）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑（默认行为，每步自动跳过已完成）")
    parser.add_argument("--skip-neb", action="store_true",
                        help="跳过 NEB 扩散计算（前四步跑完即打分）")
    parser.add_argument("--skip-lattice", action="store_true",
                        help="跳过晶格匹配，稳定性通过后直接算吸附")
    parser.add_argument("--output", type=str, default="output",
                        help="输出目录 (默认: output)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Materials Project API Key")
    args = parser.parse_args()

    # API Key 优先级：命令行 > 环境变量 > 空
    api_key = args.api_key or os.environ.get("MP_API_KEY", "")
    config = PipelineConfig(
        output_dir=args.output,
        mp_api_key=api_key,
    )

    print("=" * 60)
    print("  锂金属电池种子层材料高通量筛选 Pipeline")
    print("=" * 60)
    print(f"  输出目录: {config.output_dir}")
    print(f"  材料文件: {args.materials or 'MP API 自动搜索'}")
    print(f"  跳过 NEB: {'是' if args.skip_neb else '否'}")
    print("=" * 60)

    pipeline = SeedLayerPipeline(config)

    # Step 1: 材料池
    pipeline.step1_fetch_materials(args.materials)

    # Step 2: 电化学稳定性
    pipeline.step2_stability()

    # Step 3: 晶格失配度
    if not args.skip_lattice:
        pipeline.step3_lattice_match()

    # Step 4: 吸附能
    pipeline.step4_adsorption(skip_lattice=args.skip_lattice)

    # Step 5: 扩散势垒
    if not args.skip_neb:
        pipeline.step5_diffusion()

    # 最终打分排名
    pipeline.score_and_report()

    print("\n" + "=" * 60)
    print("  Pipeline 完成！查看结果: final_ranking.csv / final_report.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()

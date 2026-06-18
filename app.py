"""Streamlit main application for seed layer screening system."""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from seed_layer.config import PipelineConfig, load_config, save_config
from seed_layer.config import _config_to_dict as config_to_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_default_config() -> Path:
    """Return path to configs/default.yaml (project root)."""
    return Path(__file__).parent / "configs" / "default.yaml"


def _init_session_state():
    """Load config into st.session_state on first run."""
    if "config_loaded" in st.session_state:
        return

    config_path = _find_default_config()
    cfg = load_config(str(config_path))

    st.session_state["config_path"] = str(config_path)
    st.session_state["config"] = cfg
    st.session_state["config_dict"] = config_to_dict(cfg)

    # Convenience copies for sidebar widgets
    st.session_state["working_ion"] = [cfg.working_ion]
    st.session_state["screening_source"] = "MP 数据库"
    st.session_state["screening_elements"] = ""
    st.session_state["screening_n_elements"] = cfg.screening.get("n_elements", [2, 3])
    st.session_state["calculator_type"] = cfg.calculator.get("type", "chgnet")
    st.session_state["mace_model"] = cfg.calculator.get("kwargs", {}).get("model", "medium-mpa-0")
    st.session_state["fmax"] = cfg.interface.get("fmax", 0.05)
    st.session_state["steps"] = cfg.interface.get("steps", 500)
    st.session_state["use_database_eah"] = False
    st.session_state["max_e_above_hull"] = cfg.screening.get("energy_above_hull_max", 0.1)

    # Section 5: Slab Parameters
    st.session_state["slab_thickness"] = cfg.interface.get("slab_thickness", 5.0)
    st.session_state["vacuum"] = cfg.interface.get("vacuum", 15.0)

    # Section 6: Diffusion
    st.session_state["use_neb"] = True
    st.session_state["diffusion_supercell"] = "2,2,1"

    # Section 7: Interface Energy
    st.session_state["max_metal_layers"] = cfg.interface.get("max_metal_layers", 5)

    # Section 8: Scoring Weights
    scoring = getattr(cfg, "scoring", None) or {}
    st.session_state["w_adsorption"] = scoring.get("w_adsorption", 0.25)
    st.session_state["w_diffusion"] = scoring.get("w_diffusion", 0.25)
    st.session_state["w_interface"] = scoring.get("w_interface", 0.25)
    st.session_state["w_lattice"] = scoring.get("w_lattice", 0.15)
    st.session_state["w_stability"] = scoring.get("w_stability", 0.10)

    # Section 9: Output
    st.session_state["output_formats"] = ["json", "csv"]

    st.session_state["config_loaded"] = True


def _build_config_from_session() -> PipelineConfig:
    """Reconstruct PipelineConfig from current session_state values."""
    cfg = st.session_state["config"]

    # Update fields from sidebar widgets
    ions = st.session_state.get("working_ion", ["Li"])
    cfg.working_ion = ions[0] if ions else "Li"

    cfg.screening["energy_above_hull_max"] = st.session_state.get("max_e_above_hull", 0.1)

    calc_type = st.session_state.get("calculator_type", "chgnet")
    cfg.calculator["type"] = calc_type
    if calc_type == "mace":
        cfg.calculator.setdefault("kwargs", {})["model"] = st.session_state.get("mace_model", "medium-mpa-0")

    cfg.interface["fmax"] = st.session_state.get("fmax", 0.05)
    cfg.interface["steps"] = st.session_state.get("steps", 500)

    # Section 5: Slab Parameters
    cfg.interface["slab_thickness"] = st.session_state.get("slab_thickness", 5.0)
    cfg.interface["vacuum"] = st.session_state.get("vacuum", 15.0)

    # Section 7: Interface Energy
    cfg.interface["max_metal_layers"] = st.session_state.get("max_metal_layers", 5)

    # Section 8: Scoring Weights
    cfg.scoring = {
        "w_adsorption": st.session_state.get("w_adsorption", 0.25),
        "w_diffusion": st.session_state.get("w_diffusion", 0.25),
        "w_interface": st.session_state.get("w_interface", 0.25),
        "w_lattice": st.session_state.get("w_lattice", 0.15),
        "w_stability": st.session_state.get("w_stability", 0.10),
    }

    # Section 9: Output (store for downstream use)
    cfg.output["formats"] = st.session_state.get("output_formats", ["json", "csv"])

    return cfg


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="种子层筛选系统",
    page_icon="🔋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — deep-blue theme with amber accents
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Import Fira Sans + Fira Code */
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Fira Sans', sans-serif;
}

/* Code blocks */
code, pre, .stCode {
    font-family: 'Fira Code', monospace !important;
}

/* Sidebar section headers - left accent bar */
.sidebar .stSubheader,
[data-testid="stSidebar"] .stSubheader {
    border-left: 3px solid #3B82F6;
    padding-left: 0.75rem;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

/* Card style for main area sections */
.main .stTabs [data-baseweb="tab-panel"] {
    background: #1E293B;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid rgba(59, 130, 246, 0.15);
    margin-top: 0.5rem;
}

/* Dataframe dark theme */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}

/* Button hover */
.stButton > button {
    border-radius: 8px;
    transition: all 0.2s ease;
    font-family: 'Fira Sans', sans-serif;
}
.stButton > button:hover {
    border-color: #3B82F6;
    box-shadow: 0 0 12px rgba(59, 130, 246, 0.3);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #1E293B;
    border-radius: 10px;
    padding: 1rem;
    border: 1px solid rgba(59, 130, 246, 0.12);
}

/* Divider color */
hr {
    border-color: rgba(59, 130, 246, 0.2) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-family: 'Fira Sans', sans-serif;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

_init_session_state()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("🔋 种子层筛选系统")
st.caption("高通量筛选锂金属电池种子层材料 — Streamlit 控制台")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ 配置")

    # -- Load / Save --
    st.text_input(
        "配置文件路径",
        value=st.session_state.get("config_path", ""),
        key="config_path_display",
        disabled=True,
    )

    col_save, col_reload = st.columns(2)
    with col_save:
        if st.button("💾 保存", use_container_width=True):
            cfg = _build_config_from_session()
            save_config(cfg, st.session_state["config_path"])
            st.session_state["config"] = cfg
            st.session_state["config_dict"] = config_to_dict(cfg)
            st.toast("配置已保存", icon="✅")
    with col_reload:
        if st.button("🔄 重载", use_container_width=True):
            cfg = load_config(st.session_state["config_path"])
            st.session_state["config"] = cfg
            st.session_state["config_dict"] = config_to_dict(cfg)
            # Reset sidebar widget defaults
            st.session_state["working_ion"] = [cfg.working_ion]
            st.session_state["calculator_type"] = cfg.calculator.get("type", "chgnet")
            st.session_state["mace_model"] = cfg.calculator.get("kwargs", {}).get("model", "medium-mpa-0")
            st.session_state["fmax"] = cfg.interface.get("fmax", 0.05)
            st.session_state["steps"] = cfg.interface.get("steps", 500)
            st.session_state["max_e_above_hull"] = cfg.screening.get("energy_above_hull_max", 0.1)
            # Reset Sections 5-9
            st.session_state["slab_thickness"] = cfg.interface.get("slab_thickness", 5.0)
            st.session_state["vacuum"] = cfg.interface.get("vacuum", 15.0)
            st.session_state["use_neb"] = True
            st.session_state["diffusion_supercell"] = "2,2,1"
            st.session_state["max_metal_layers"] = cfg.interface.get("max_metal_layers", 5)
            scoring = getattr(cfg, "scoring", None) or {}
            st.session_state["w_adsorption"] = scoring.get("w_adsorption", 0.25)
            st.session_state["w_diffusion"] = scoring.get("w_diffusion", 0.25)
            st.session_state["w_interface"] = scoring.get("w_interface", 0.25)
            st.session_state["w_lattice"] = scoring.get("w_lattice", 0.15)
            st.session_state["w_stability"] = scoring.get("w_stability", 0.10)
            st.session_state["output_formats"] = ["json", "csv"]
            st.toast("配置已重载", icon="🔄")

    st.divider()

    # -- Section 1: Target Metal --
    st.subheader("🎯 目标金属")
    st.multiselect(
        "金属元素",
        options=["Li", "Na", "K", "Mg", "Ca", "Zn", "Al"],
        key="working_ion",
    )

    st.divider()

    # -- Section 2: Candidate Materials --
    st.subheader("📦 候选材料")
    st.radio(
        "来源",
        options=["MP 数据库", "自定义列表"],
        key="screening_source",
        horizontal=True,
    )
    st.number_input(
        "最大材料数 (max_materials)",
        min_value=1,
        max_value=1000,
        step=1,
        key="screening_max_materials",
    )
    if st.session_state.get("screening_source") == "自定义列表":
        st.text_input(
            "元素列表（逗号分隔，如 Li,Fe,P）",
            key="screening_elements",
        )

    st.divider()

    # -- Section 3: Calculation Parameters --
    st.subheader("🧮 计算参数")
    st.selectbox(
        "势函数 (calculator)",
        options=["mace", "m3gnet", "chgnet"],
        key="calculator_type",
    )
    if st.session_state.get("calculator_type") == "mace":
        st.text_input(
            "MACE 模型",
            key="mace_model",
        )
    st.number_input(
        "力收敛阈值 (fmax, eV/Å)",
        min_value=0.001,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        key="fmax",
    )
    st.number_input(
        "最大弛豫步数 (steps)",
        min_value=10,
        max_value=10000,
        step=50,
        key="steps",
    )

    st.divider()

    # -- Section 4: Stability Screening --
    st.subheader("🧪 稳定性筛选")
    st.checkbox(
        "使用数据库 Eh（替代 DFT 计算）",
        key="use_database_eah",
    )
    st.number_input(
        "最大能量高于凸包 (eV/atom)",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="max_e_above_hull",
    )

    st.divider()

    # -- Section 5: Slab Parameters --
    st.subheader("📐 平板参数")
    st.number_input(
        "平板厚度 (Å)",
        min_value=1.0,
        max_value=20.0,
        step=0.5,
        key="slab_thickness",
    )
    st.number_input(
        "真空层厚度 (Å)",
        min_value=5.0,
        max_value=50.0,
        step=1.0,
        key="vacuum",
    )

    st.divider()

    # -- Section 6: Diffusion --
    st.subheader("🔀 扩散")
    st.checkbox(
        "启用 NEB 扩散计算",
        key="use_neb",
    )
    st.text_input(
        "扩散超胞 (如 2,2,1)",
        key="diffusion_supercell",
    )

    st.divider()

    # -- Section 7: Interface Energy --
    st.subheader("🔗 界面能")
    st.number_input(
        "最大金属层数",
        min_value=1,
        max_value=20,
        step=1,
        key="max_metal_layers",
    )

    st.divider()

    # -- Section 8: Scoring Weights --
    st.subheader("⚖️ 评分权重")
    st.slider(
        "w_adsorption",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="w_adsorption",
    )
    st.slider(
        "w_diffusion",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="w_diffusion",
    )
    st.slider(
        "w_interface",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="w_interface",
    )
    st.slider(
        "w_lattice",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="w_lattice",
    )
    st.slider(
        "w_stability",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="w_stability",
    )

    # Live weight sum indicator
    total_weight = (
        st.session_state.get("w_adsorption", 0.25) +
        st.session_state.get("w_diffusion", 0.25) +
        st.session_state.get("w_interface", 0.25) +
        st.session_state.get("w_lattice", 0.15) +
        st.session_state.get("w_stability", 0.10)
    )
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"权重总和: {total_weight:.2f}（建议 = 1.00）")
    else:
        st.success(f"权重总和: {total_weight:.2f}")

    st.divider()

    # -- Section 9: Output --
    st.subheader("📤 输出")
    st.multiselect(
        "输出格式",
        options=["json", "csv", "cif"],
        key="output_formats",
    )

# ---------------------------------------------------------------------------
# Pipeline runner helpers
# ---------------------------------------------------------------------------

def _start_pipeline(config_path: str, output_dir: str) -> subprocess.Popen:
    """Start pipeline as a background subprocess.

    Args:
        config_path: Absolute path to the YAML config file.
        output_dir: Output directory for pipeline results.

    Returns:
        subprocess.Popen object for the running pipeline.
    """
    cmd = [
        sys.executable, "-m", "seed_layer.pipeline",
        "--config", config_path,
        "--output-dir", output_dir,
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(Path(__file__).parent / "src"),
    )
    return process


def _stop_pipeline():
    """Terminate the running pipeline process if one exists."""
    proc = st.session_state.get("pipeline_process")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        st.session_state["pipeline_status"] = "stopped"


def _read_process_output(proc: subprocess.Popen) -> list:
    """Read all available stdout lines from a running process (non-blocking).

    Args:
        proc: The subprocess.Popen object.

    Returns:
        List of new lines read since last call.
    """
    import select
    import os

    new_lines = []
    # Use os.read with select for non-blocking read on Unix,
    # or readline on Windows (blocking but acceptable for Streamlit reruns)
    if os.name == "nt":
        # Windows: read whatever is available
        import msvcrt
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = msvcrt.get_osfhandle(proc.stdout.fileno())
        while True:
            # Peek at the pipe to see if data is available
            avail = ctypes.c_ulong(0)
            kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None)
            if avail.value == 0:
                break
            line = proc.stdout.readline()
            if line:
                new_lines.append(line.rstrip("\n"))
            else:
                break
    else:
        # Unix: use select for non-blocking read
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0)
            if not ready:
                break
            line = proc.stdout.readline()
            if line:
                new_lines.append(line.rstrip("\n"))
            else:
                break
    return new_lines


def _find_latest_run(output_dir: str) -> dict | None:
    """Find the latest run directory and load summary.csv."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    run_dirs = sorted(output_path.glob("run_*"), reverse=True)
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.csv"
        if summary_path.exists():
            return {
                "run_dir": run_dir,
                "summary_path": summary_path,
            }
    return None


# ---------------------------------------------------------------------------
# Main area — Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_run, tab_results = st.tabs(["总览", "运行", "结果"])

# -- Tab 1: Overview --
with tab_overview:
    st.subheader("当前配置")
    st.json(st.session_state.get("config_dict", {}))

    st.divider()
    st.subheader("导出配置")

    cfg = st.session_state.get("config")
    if cfg:
        yaml_str = yaml.dump(
            config_to_dict(cfg),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        st.download_button(
            label="下载 YAML",
            data=yaml_str,
            file_name="seed_layer_config.yaml",
            mime="text/yaml",
            use_container_width=True,
        )

# -- Tab 2: Run Pipeline --
with tab_run:
    # Initialize session state keys
    if "pipeline_process" not in st.session_state:
        st.session_state["pipeline_process"] = None
    if "pipeline_status" not in st.session_state:
        st.session_state["pipeline_status"] = "idle"
    if "log_lines" not in st.session_state:
        st.session_state["log_lines"] = []

    # -- Run configuration --
    st.subheader("运行配置")

    col_cfg, col_out = st.columns(2)
    with col_cfg:
        st.text_input(
            "配置文件",
            value=st.session_state.get("config_path", ""),
            key="run_config_path",
            disabled=True,
        )
    with col_out:
        output_dir = st.text_input(
            "输出目录",
            key="run_output_dir",
        )

    # -- Control buttons --
    st.divider()
    col_start, col_stop, col_status = st.columns([1, 1, 2])

    with col_start:
        start_disabled = st.session_state["pipeline_status"] == "running"
        if st.button(
            "▶ 开始运行",
            disabled=start_disabled,
            use_container_width=True,
            type="primary",
        ):
            # Save config before running
            cfg = _build_config_from_session()
            save_config(cfg, st.session_state["config_path"])
            st.session_state["config"] = cfg
            st.session_state["config_dict"] = config_to_dict(cfg)

            # Launch subprocess
            proc = _start_pipeline(
                st.session_state["config_path"],
                output_dir,
            )
            st.session_state["pipeline_process"] = proc
            st.session_state["pipeline_status"] = "running"
            st.session_state["log_lines"] = []
            st.rerun()

    with col_stop:
        stop_disabled = st.session_state["pipeline_status"] != "running"
        if st.button(
            "⏹ 停止",
            disabled=stop_disabled,
            use_container_width=True,
        ):
            _stop_pipeline()
            st.session_state["log_lines"].append("[SYSTEM] Pipeline stopped by user")
            st.rerun()

    # -- Status indicator --
    with col_status:
        status = st.session_state["pipeline_status"]
        proc = st.session_state.get("pipeline_process")

        if status == "running" and proc and proc.poll() is None:
            st.info("Pipeline is running...")
        elif status == "running" and proc and proc.poll() is not None:
            # Process finished between reruns
            exit_code = proc.returncode
            if exit_code == 0:
                st.session_state["pipeline_status"] = "success"
            else:
                st.session_state["pipeline_status"] = "error"
            st.session_state["pipeline_exit_code"] = exit_code
        elif status == "success":
            st.success(f"Pipeline completed successfully (exit code 0)")
        elif status == "error":
            code = st.session_state.get("pipeline_exit_code", "?")
            st.error(f"Pipeline failed (exit code {code})")
        elif status == "stopped":
            st.warning("Pipeline was stopped by user")
        else:
            st.caption("Ready to run")

    # -- Log display --
    st.divider()
    st.subheader("运行日志")

    # Read new output from running process
    if st.session_state["pipeline_status"] == "running":
        proc = st.session_state.get("pipeline_process")
        if proc and proc.poll() is None:
            new_lines = _read_process_output(proc)
            st.session_state["log_lines"].extend(new_lines)
        elif proc and proc.poll() is not None:
            # Drain remaining output
            remaining = proc.stdout.read()
            if remaining:
                for line in remaining.splitlines():
                    st.session_state["log_lines"].append(line.rstrip("\n"))
            exit_code = proc.returncode
            if exit_code == 0:
                st.session_state["pipeline_status"] = "success"
                st.session_state["log_lines"].append("[SYSTEM] Pipeline finished successfully")
            else:
                st.session_state["pipeline_status"] = "error"
                st.session_state["log_lines"].append(f"[SYSTEM] Pipeline failed with exit code {exit_code}")
            st.rerun()

    # Display logs
    log_text = "\n".join(st.session_state.get("log_lines", []))
    if log_text:
        st.code(log_text, language=None)
    else:
        st.caption("No output yet. Start the pipeline to see logs here.")

    # Auto-refresh while running
    if st.session_state["pipeline_status"] == "running":
        time.sleep(2)
        st.rerun()

# -- Tab 3: Results --
with tab_results:
    st.subheader("最新结果")

    # Refresh button
    if st.button("🔄 Refresh", key="results_refresh"):
        st.rerun()

    result_info = _find_latest_run("output")

    if result_info is None:
        st.info("No results found. Run the pipeline first to generate output.")
    else:
        st.caption(f"Run directory: `{result_info['run_dir']}`")

        try:
            df = pd.read_csv(result_info["summary_path"])

            # Score table (sortable)
            st.divider()
            st.subheader("评分表格")
            st.dataframe(df, use_container_width=True)

            # Bar chart: composite_score per material
            score_col = None
            for candidate in ["composite_score", "score", "total_score"]:
                if candidate in df.columns:
                    score_col = candidate
                    break

            id_col = None
            for candidate in ["material_id", "formula", "material", "name"]:
                if candidate in df.columns:
                    id_col = candidate
                    break

            if score_col and id_col:
                st.divider()
                st.subheader("综合评分排名")

                # Sort by composite score descending, show top 20
                df_sorted = df.sort_values(score_col, ascending=False)
                chart_df = df_sorted.head(20)

                fig = go.Figure(go.Bar(
                    x=chart_df[score_col],
                    y=chart_df[id_col],
                    orientation='h',
                    marker=dict(
                        color=chart_df[score_col],
                        colorscale=[[0, '#1E3A8A'], [0.5, '#3B82F6'], [1, '#F59E0B']],
                        line=dict(width=0),
                    ),
                    text=chart_df[score_col].round(3),
                    textposition='outside',
                    textfont=dict(family='Fira Code', size=11, color='#E2E8F0'),
                ))
                fig.update_layout(
                    title=dict(text='综合评分排名', font=dict(family='Fira Sans', size=16, color='#E2E8F0')),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title='Score', gridcolor='rgba(59,130,246,0.1)', color='#94A3B8'),
                    yaxis=dict(autorange='reversed', color='#E2E8F0'),
                    height=max(400, len(chart_df) * 35),
                    margin=dict(l=120, r=40, t=50, b=40),
                    font=dict(family='Fira Sans'),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Sub-score radar chart (if available)
            sub_score_cols = [c for c in df.columns if c.startswith('S_')]
            if sub_score_cols and id_col:
                st.divider()
                st.subheader("分项评分雷达图")

                # Show radar for top 5 materials
                df_sorted = df.sort_values(score_col, ascending=False) if score_col else df
                top5 = df_sorted.head(5)
                labels = [c.replace('S_', '').capitalize() for c in sub_score_cols]

                fig = go.Figure()
                colors = ['#3B82F6', '#F59E0B', '#10B981', '#EF4444', '#8B5CF6']
                for i, (_, row) in enumerate(top5.iterrows()):
                    values = [row[c] if pd.notna(row[c]) else 0 for c in sub_score_cols]
                    values.append(values[0])  # close the polygon
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=labels + [labels[0]],
                        name=str(row[id_col]),
                        line=dict(color=colors[i % len(colors)], width=2),
                        fill='toself',
                        opacity=0.15,
                    ))
                fig.update_layout(
                    title=dict(text='Top 5 材料分项评分', font=dict(family='Fira Sans', size=16, color='#E2E8F0')),
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(gridcolor='rgba(59,130,246,0.15)', color='#94A3B8', range=[0, 1]),
                        angularaxis=dict(color='#E2E8F0'),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Fira Sans'),
                    height=450,
                    showlegend=True,
                    legend=dict(font=dict(color='#E2E8F0')),
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.error(f"Failed to load results: {exc}")

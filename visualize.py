"""
可视化模块 - 多叶片对比可视化
功能：
1. 单叶片时序载荷曲线（批量出图）
2. 多叶片极限载荷极值对比柱状图（通用，支持EOG/NTM等所有风况）
3. 多叶片疲劳损伤分布对比图
4. 整轮合成载荷时序与包络图
5. 面板级载荷云图（展向-时间）
6. 载荷谱（雨流计数结果）
"""

import os
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

from config import config, VisualizeConfig

logger = logging.getLogger(__name__)


# ============================================================
# 绘图基础设置
# ============================================================
def setup_plot_style(cfg: Optional[VisualizeConfig] = None):
    """设置matplotlib绘图风格"""
    cfg = cfg or config.visualize
    plt.rcParams["font.sans-serif"] = [cfg.font_family, "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = cfg.dpi
    plt.rcParams["savefig.dpi"] = cfg.dpi
    plt.rcParams["axes.grid"] = cfg.grid
    plt.rcParams["grid.alpha"] = 0.3


def save_figure(fig, name: str, cfg: Optional[VisualizeConfig] = None):
    """保存图片"""
    cfg = cfg or config.visualize
    path = os.path.join(config.path.figure_dir, f"{name}.{cfg.fig_format}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"图片保存: {path}")
    return path


# ============================================================
# 单叶片时序载荷图
# ============================================================
def plot_blade_timeseries(blade_df: pd.DataFrame, blade_id: str,
                          load_types: Optional[List[str]] = None,
                          title_suffix: str = "") -> str:
    """
    绘制单叶片时序载荷曲线
    blade_df: 包含 time, total_normal, total_tangential, azimuth 等
    """
    setup_plot_style()
    if load_types is None:
        load_types = ["total_normal", "total_tangential"]

    time_col = "time" if "time" in blade_df.columns else config.global_ch.time_col
    t = blade_df[time_col].values

    fig, axes = plt.subplots(len(load_types), 1, figsize=(12, 4 * len(load_types)),
                             sharex=True)
    if len(load_types) == 1:
        axes = [axes]

    colors = config.visualize.colors
    for i, lt in enumerate(load_types):
        if lt in blade_df.columns:
            axes[i].plot(t, blade_df[lt].values, color=colors[i % len(colors)],
                         linewidth=0.8, label=lt)
            axes[i].set_ylabel(lt)
            axes[i].legend(loc="upper right")
            # 标注峰值
            vals = blade_df[lt].values
            idx_max = np.argmax(vals)
            idx_min = np.argmin(vals)
            axes[i].annotate(f"max={vals[idx_max]:.1f}",
                             xy=(t[idx_max], vals[idx_max]),
                             fontsize=8, color="red")
            axes[i].annotate(f"min={vals[idx_min]:.1f}",
                             xy=(t[idx_min], vals[idx_min]),
                             fontsize=8, color="blue")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{blade_id} 载荷时序 {title_suffix}", fontsize=14)
    plt.tight_layout()
    return save_figure(fig, f"{blade_id}_timeseries")


def plot_all_blades_timeseries(df: pd.DataFrame, mapper,
                               load_type: str = "total_normal") -> str:
    """
    绘制所有叶片同一载荷的对比时序图
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=config.visualize.figsize)
    time_col = config.global_ch.time_col
    t = df[time_col].values
    colors = config.visualize.colors

    for i, bid in enumerate(mapper.blade_total_loads):
        bdf = mapper.get_blade_total_load_df(df, bid)
        if not bdf.empty and load_type in bdf.columns:
            ax.plot(t[:len(bdf)], bdf[load_type].values,
                    label=bid, color=colors[i % len(colors)], linewidth=0.8)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(load_type)
    ax.set_title(f"多叶片{load_type}载荷时序对比")
    ax.legend()
    plt.tight_layout()
    return save_figure(fig, f"all_blades_{load_type}_timeseries")


# ============================================================
# 极限载荷极值对比图（通用，支持EOG/NTM等所有风况）
# ============================================================
def plot_eog_extremes_comparison(summary_df: pd.DataFrame,
                                 metric: str = "combined_peak_value") -> str:
    """
    绘制多叶片极限载荷极值对比柱状图
    （函数名保留eog前缀以向后兼容）
    """
    setup_plot_style()
    if summary_df.empty or metric not in summary_df.columns:
        return ""

    fig, ax = plt.subplots(figsize=config.visualize.figsize)
    blades = summary_df["blade_id"].values
    values = summary_df[metric].values

    bars = ax.bar(blades, values, color=config.visualize.colors[:len(blades)])
    ax.set_ylabel(metric)
    ax.set_title(f"多叶片极限载荷极值对比 ({metric})")

    # 数值标注
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    return save_figure(fig, f"eog_{metric}_comparison")


# ============================================================
# 疲劳损伤对比图
# ============================================================
def plot_fatigue_damage_comparison(summary_df: pd.DataFrame,
                                   metric: str = "combined_load_damage") -> str:
    """
    绘制多叶片疲劳损伤对比图
    """
    setup_plot_style()
    if summary_df.empty or metric not in summary_df.columns:
        return ""

    fig, ax = plt.subplots(figsize=config.visualize.figsize)
    blades = summary_df["blade_id"].values
    values = summary_df[metric].values

    colors = ["#d62728" if v == max(values) else "#1f77b4" for v in values]
    bars = ax.bar(blades, values, color=colors)
    ax.set_ylabel("Miner Damage")
    ax.set_title(f"多叶片疲劳损伤对比 ({metric})")
    ax.set_yscale("log")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2e}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    return save_figure(fig, f"fatigue_{metric}_comparison")


def plot_rainflow_spectrum(ranges: np.ndarray, counts: np.ndarray,
                           blade_id: str, channel: str = "") -> str:
    """
    绘制雨流计数载荷谱
    """
    setup_plot_style()
    if len(ranges) == 0:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 载荷范围-循环次数柱状图
    ax1.bar(ranges, counts, width=np.diff(ranges).mean() * 0.8 if len(ranges) > 1 else 1.0,
            color="#1f77b4", alpha=0.7)
    ax1.set_xlabel("Load Range")
    ax1.set_ylabel("Cycle Counts")
    ax1.set_title(f"{blade_id} {channel} 载荷谱")
    ax1.set_yscale("log")

    # 累积分布
    sorted_idx = np.argsort(ranges)
    cum_counts = np.cumsum(counts[sorted_idx])
    ax2.plot(ranges[sorted_idx], cum_counts, "o-", color="#ff7f0e")
    ax2.set_xlabel("Load Range")
    ax2.set_ylabel("Cumulative Cycles")
    ax2.set_title("累积循环次数")

    plt.tight_layout()
    return save_figure(fig, f"{blade_id}_{channel}_rainflow_spectrum")


# ============================================================
# 整轮合成载荷图
# ============================================================
def plot_synthesis_timeseries(synth_df: pd.DataFrame) -> str:
    """
    绘制整轮合成载荷时序图
    """
    setup_plot_style()
    if synth_df.empty:
        return ""

    time_col = "time" if "time" in synth_df.columns else config.global_ch.time_col
    t = synth_df[time_col].values

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 合力
    if "global_F_mag" in synth_df.columns:
        axes[0].plot(t, synth_df["global_F_mag"].values, "b-", linewidth=0.8, label="|F|")
        axes[0].set_ylabel("Total Force [N]")
        axes[0].legend()

    # X/Y分量
    if "global_Fx" in synth_df.columns:
        axes[1].plot(t, synth_df["global_Fx"].values, "r-", linewidth=0.8, label="Fx")
        axes[1].plot(t, synth_df["global_Fy"].values, "g-", linewidth=0.8, label="Fy")
        axes[1].set_ylabel("Force Components [N]")
        axes[1].legend()

    # 倾覆弯矩
    if "overturning_moment" in synth_df.columns:
        axes[2].plot(t, synth_df["overturning_moment"].values, "k-", linewidth=0.8)
        axes[2].set_ylabel("Overturning Moment [Nm]")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("整轮合成载荷时序", fontsize=14)
    plt.tight_layout()
    return save_figure(fig, "wheel_synthesis_timeseries")


# ============================================================
# 面板级载荷云图
# ============================================================
def plot_panel_heatmap(panel_df: pd.DataFrame, blade_id: str,
                       load_type: str = "normal_force") -> str:
    """
    绘制面板级载荷云图（时间-展向）
    panel_df: 列 time, PAN_0..PAN_n
    """
    setup_plot_style()
    panel_cols = [c for c in panel_df.columns if c.startswith("PAN_")]
    if not panel_cols:
        return ""

    time_col = "time"
    t = panel_df[time_col].values
    loads = panel_df[panel_cols].values.T  # (panels, time)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(t, np.arange(len(panel_cols)), loads,
                       shading="auto", cmap="jet")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Panel Index (展向)")
    ax.set_title(f"{blade_id} {load_type} 展向-时间分布")
    plt.colorbar(im, ax=ax, label=load_type)
    plt.tight_layout()
    return save_figure(fig, f"{blade_id}_{load_type}_heatmap")


def plot_panel_spanwise_distribution(panel_df: pd.DataFrame, blade_id: str,
                                     load_type: str = "normal_force") -> str:
    """
    绘制峰值时刻的展向载荷分布
    """
    setup_plot_style()
    panel_cols = [c for c in panel_df.columns if c.startswith("PAN_")]
    if not panel_cols:
        return ""

    loads = panel_df[panel_cols].values
    # 找整体最大载荷时刻
    max_idx = np.unravel_index(np.argmax(np.abs(loads)), loads.shape)
    peak_row = loads[max_idx[0]]

    fig, ax = plt.subplots(figsize=config.visualize.figsize)
    ax.plot(range(len(panel_cols)), peak_row, "o-", color="#1f77b4", linewidth=2)
    ax.fill_between(range(len(panel_cols)), peak_row, alpha=0.3)
    ax.set_xlabel("Panel Index (展向位置)")
    ax.set_ylabel(load_type)
    ax.set_title(f"{blade_id} 峰值时刻展向载荷分布 (t={panel_df['time'].iloc[max_idx[0]]:.2f}s)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return save_figure(fig, f"{blade_id}_{load_type}_spanwise")


# ============================================================
# 批量可视化入口
# ============================================================
def generate_all_plots(df: pd.DataFrame, mapper, eog_results: Dict,
                       fatigue_results: Dict, synth_results: Dict) -> List[str]:
    """
    生成所有可视化图表
    返回图片路径列表
    """
    logger.info("开始生成可视化图表...")
    paths = []

    # 1. 多叶片时序对比
    for lt in ["total_normal", "total_tangential"]:
        p = plot_all_blades_timeseries(df, mapper, lt)
        if p:
            paths.append(p)

    # 2. 极限载荷极值对比（兼容eog_results参数名）
    if eog_results and "summary_df" in eog_results and not eog_results["summary_df"].empty:
        for metric in ["combined_peak_value", "total_normal_max", "total_tangential_max"]:
            p = plot_eog_extremes_comparison(eog_results["summary_df"], metric)
            if p:
                paths.append(p)

    # 3. 疲劳损伤对比
    if "summary_df" in fatigue_results and not fatigue_results["summary_df"].empty:
        for metric in ["combined_load_damage", "total_normal_damage"]:
            p = plot_fatigue_damage_comparison(fatigue_results["summary_df"], metric)
            if p:
                paths.append(p)

    # 4. 雨流谱（每个叶片）
    for bid, fres in fatigue_results.get("blade_results", {}).items():
        for ch_name, ch_data in fres.items():
            if ch_name.startswith("panel_"):
                continue
            if "spectrum_ranges" in ch_data:
                p = plot_rainflow_spectrum(
                    ch_data["spectrum_ranges"], ch_data["spectrum_counts"],
                    bid, ch_name
                )
                if p:
                    paths.append(p)

    # 5. 整轮合成载荷
    if "synthesis_df" in synth_results and not synth_results["synthesis_df"].empty:
        p = plot_synthesis_timeseries(synth_results["synthesis_df"])
        if p:
            paths.append(p)

    # 6. 面板级云图（每个叶片法向力）
    if config.visualize.panel_heatmap:
        for bid in mapper.blade_total_loads:
            panel_df = mapper.get_blade_panel_load_df(df, bid, "normal_force")
            if not panel_df.empty:
                p = plot_panel_heatmap(panel_df, bid, "normal_force")
                if p:
                    paths.append(p)
                p = plot_panel_spanwise_distribution(panel_df, bid, "normal_force")
                if p:
                    paths.append(p)

    logger.info(f"可视化完成: 生成 {len(paths)} 张图")
    return paths

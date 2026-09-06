"""
极限载荷分析模块（通用化，支持EOG/NTM/EWS等所有风况类型）

功能：
1. 单叶片独立极值统计（总切向/法向载荷、面板分布载荷）
2. 整轮综合载荷极值（合力、倾覆弯矩）
3. 最不利叶片筛查
4. 叶片载荷不均匀系数
5. 工况自适应极值段定位（瞬态阵风切片 / 稳态全量极值 / 滑动窗口）

与旧版eog_analysis.py的区别：
- 命名通用化（EOG→Extreme）
- 支持所有风况类型，不再假设只有EOG
- 极值段检测策略根据工况自动选择
- 保留完全向后兼容（run_eog_analysis = run_extreme_analysis别名）
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from config import config, ExtremeLoadConfig
from utils import compute_statistics, find_peak_time, combined_load

logger = logging.getLogger(__name__)


# ============================================================
# 单叶片极限载荷分析
# ============================================================
class BladeExtremeAnalyzer:
    """单叶片极限载荷分析器（通用，适用于所有风况）"""

    def __init__(self, blade_id: str, cfg: Optional[ExtremeLoadConfig] = None):
        self.blade_id = blade_id
        self.cfg = cfg or config.extreme
        self.results: Dict[str, Any] = {}

    def analyze_total_loads(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析叶片总载荷（切向+法向）
        df: 包含 time, total_tangential, total_normal, azimuth, pitch 的DataFrame
        """
        result = {"blade_id": self.blade_id}
        time_col = "time" if "time" in df.columns else config.global_ch.time_col

        # 各通道统计
        for load_type in ["total_tangential", "total_normal"]:
            if load_type in df.columns:
                s = df[load_type]
                stats = compute_statistics(s, self.cfg.stats_metrics)
                result[f"{load_type}_stats"] = stats

                # 峰值时间
                t_peak, v_peak = find_peak_time(s, df[time_col], mode="abs_max")
                result[f"{load_type}_peak_time"] = t_peak
                result[f"{load_type}_peak_value"] = v_peak

                # 对应方位角
                if "azimuth" in df.columns:
                    idx = np.argmax(np.abs(s.values))
                    result[f"{load_type}_peak_azimuth"] = float(df["azimuth"].iloc[idx])

        # 合成载荷（法向+切向矢量和）
        if "total_tangential" in df.columns and "total_normal" in df.columns:
            combined = combined_load(df["total_normal"].values, df["total_tangential"].values)
            combined_s = pd.Series(combined, name="combined_load")
            stats = compute_statistics(combined_s, self.cfg.stats_metrics)
            result["combined_load_stats"] = stats
            t_peak, v_peak = find_peak_time(combined_s, df[time_col], mode="max")
            result["combined_peak_time"] = t_peak
            result["combined_peak_value"] = v_peak
            if "azimuth" in df.columns:
                idx = np.argmax(combined)
                result["combined_peak_azimuth"] = float(df["azimuth"].iloc[idx])

        self.results.update(result)
        return result

    def analyze_panel_loads(self, panel_df: pd.DataFrame,
                            load_type: str = "normal_force") -> Dict[str, Any]:
        """
        分析面板级分布载荷极值
        panel_df: 列名为 time, PAN_0, PAN_1, ... 的DataFrame
        load_type: normal_force / tangential_force / pitching_moment
        """
        result = {"blade_id": self.blade_id, "load_type": load_type}
        panel_cols = [c for c in panel_df.columns if c.startswith("PAN_")]
        if not panel_cols:
            return result

        # 每个面板的极值
        panel_extremes = []
        for col in panel_cols:
            s = panel_df[col]
            stats = compute_statistics(s, ("max", "min", "mean", "abs_max"))
            panel_extremes.append({
                "panel": col,
                "max": stats["max"],
                "min": stats["min"],
                "mean": stats["mean"],
                "abs_max": stats["abs_max"],
            })

        result["panel_extremes"] = pd.DataFrame(panel_extremes)

        # 最危险面板（abs_max最大）
        worst_idx = np.argmax([p["abs_max"] for p in panel_extremes])
        result["worst_panel"] = panel_extremes[worst_idx]["panel"]
        result["worst_panel_value"] = panel_extremes[worst_idx]["abs_max"]

        # 沿展向的载荷分布（峰值时刻）
        time_col = "time"
        if time_col in panel_df.columns:
            # 找整体最大载荷时刻
            all_vals = panel_df[panel_cols].values
            max_idx = np.unravel_index(np.argmax(np.abs(all_vals)), all_vals.shape)
            result["peak_time"] = float(panel_df[time_col].iloc[max_idx[0]])
            result["peak_distribution"] = panel_df[panel_cols].iloc[max_idx[0]].to_dict()

        self.results[f"panel_{load_type}"] = result
        return result

    def get_summary_row(self) -> Dict[str, Any]:
        """获取汇总行（用于多叶片对比表）"""
        row = {"blade_id": self.blade_id}
        for key in ["total_tangential_stats", "total_normal_stats", "combined_load_stats"]:
            if key in self.results:
                prefix = key.replace("_stats", "")
                stats = self.results[key]
                row[f"{prefix}_max"] = stats.get("max", np.nan)
                row[f"{prefix}_min"] = stats.get("min", np.nan)
                row[f"{prefix}_peak_to_peak"] = stats.get("peak_to_peak", np.nan)
                row[f"{prefix}_rms"] = stats.get("rms", np.nan)
        if "combined_peak_value" in self.results:
            row["combined_peak_value"] = self.results["combined_peak_value"]
            row["combined_peak_time"] = self.results["combined_peak_time"]
        return row


# ============================================================
# 整轮极限载荷综合分析
# ============================================================
class WheelExtremeAnalyzer:
    """整轮（多叶片合成）极限载荷分析（通用，适用于所有风况）"""

    def __init__(self, cfg: Optional[ExtremeLoadConfig] = None):
        self.cfg = cfg or config.extreme
        self.results: Dict[str, Any] = {}

    def analyze_global_loads(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析整机全局载荷（X/Y/Z向力和力矩）
        df: 包含全局载荷通道的DataFrame
        """
        gc = config.global_ch
        result = {}
        time_col = config.global_ch.time_col

        load_channels = {
            "thrust_x": gc.thrust_x_col,
            "thrust_y": gc.thrust_y_col,
            "thrust_z": gc.thrust_z_col,
            "moment_x": gc.moment_x_col,
            "moment_y": gc.moment_y_col,
            "moment_z": gc.moment_z_col,
        }

        for name, col in load_channels.items():
            if col in df.columns:
                s = df[col]
                stats = compute_statistics(s, self.cfg.stats_metrics)
                result[f"{name}_stats"] = stats
                t_peak, v_peak = find_peak_time(s, df[time_col], mode="abs_max")
                result[f"{name}_peak_time"] = t_peak
                result[f"{name}_peak_value"] = v_peak

        # 合力幅值
        fx = df[gc.thrust_x_col].values if gc.thrust_x_col in df.columns else None
        fy = df[gc.thrust_y_col].values if gc.thrust_y_col in df.columns else None
        fz = df[gc.thrust_z_col].values if gc.thrust_z_col in df.columns else None
        if fx is not None and fy is not None and fz is not None:
            total_force = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
            stats = compute_statistics(pd.Series(total_force), self.cfg.stats_metrics)
            result["total_force_stats"] = stats
            result["total_force_peak"] = float(np.max(total_force))

        # 合成倾覆弯矩（X-Y平面）
        mx = df[gc.moment_x_col].values if gc.moment_x_col in df.columns else None
        my = df[gc.moment_y_col].values if gc.moment_y_col in df.columns else None
        if mx is not None and my is not None:
            overturning = np.sqrt(mx ** 2 + my ** 2)
            stats = compute_statistics(pd.Series(overturning), self.cfg.stats_metrics)
            result["overturning_moment_stats"] = stats
            result["overturning_moment_peak"] = float(np.max(overturning))

        # 气动性能（含最大/最小值发生时刻）
        for name, col in [("power", gc.inst_power_col), ("torque", gc.inst_torque_col),
                           ("thrust", gc.inst_thrust_col)]:
            if col in df.columns:
                s = df[col]
                stats = compute_statistics(s, self.cfg.stats_metrics)
                result[f"{name}_stats"] = stats
                # 最大值发生时刻
                t_max, v_max = find_peak_time(s, df[time_col], mode="max")
                result[f"{name}_max_time"] = t_max
                result[f"{name}_max_value"] = v_max
                # 最小值发生时刻
                t_min, v_min = find_peak_time(s, df[time_col], mode="min")
                result[f"{name}_min_time"] = t_min
                result[f"{name}_min_value"] = v_min

        self.results.update(result)
        return result

    def analyze_blade_imbalance(self, blade_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析叶片间载荷不均匀度
        blade_results: {blade_id: BladeExtremeAnalyzer结果}
        """
        result = {}
        metric = self.cfg.critical_blade_metric

        # 收集各叶片峰值（过滤NaN，使用abs_max，缺失时回退到max）
        combined_peaks = {}
        normal_peaks = {}
        tangential_peaks = {}
        for bid, res in blade_results.items():
            if "combined_peak_value" in res:
                v = res["combined_peak_value"]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    combined_peaks[bid] = float(v)
            if "total_normal_stats" in res:
                stats = res["total_normal_stats"]
                v = stats.get("abs_max", stats.get("max", np.nan))
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    normal_peaks[bid] = float(v)
            if "total_tangential_stats" in res:
                stats = res["total_tangential_stats"]
                v = stats.get("abs_max", stats.get("max", np.nan))
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    tangential_peaks[bid] = float(v)

        # 不均匀系数（安全计算，避免除零和NaN）
        def _safe_imbalance(peaks_dict):
            if not peaks_dict:
                return 0.0
            vals = np.array(list(peaks_dict.values()), dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                return 0.0
            mean_val = np.mean(vals)
            if mean_val == 0 or np.isnan(mean_val):
                return 0.0
            return float(np.std(vals) / mean_val)

        if combined_peaks:
            result["combined_peak_imbalance"] = _safe_imbalance(combined_peaks)
            result["combined_peak_max_blade"] = max(combined_peaks, key=combined_peaks.get)
            result["combined_peak_min_blade"] = min(combined_peaks, key=combined_peaks.get)

        result["normal_peak_imbalance"] = _safe_imbalance(normal_peaks)
        result["tangential_peak_imbalance"] = _safe_imbalance(tangential_peaks)

        # 最不利叶片
        if self.cfg.find_critical_blade:
            if metric == "combined" and combined_peaks:
                result["critical_blade"] = max(combined_peaks, key=combined_peaks.get)
                result["critical_blade_load"] = max(combined_peaks.values())
            elif metric == "total_normal" and normal_peaks:
                result["critical_blade"] = max(normal_peaks, key=normal_peaks.get)
                result["critical_blade_load"] = max(normal_peaks.values())
            elif metric == "total_tangential" and tangential_peaks:
                result["critical_blade"] = max(tangential_peaks, key=tangential_peaks.get)
                result["critical_blade_load"] = max(tangential_peaks.values())

        self.results["imbalance"] = result
        return result


# ============================================================
# 极限载荷分析主入口（通用化）
# ============================================================
def run_extreme_analysis(df: pd.DataFrame, mapper,
                         extreme_segments: Optional[List[Tuple[float, float]]] = None,
                         condition_type: str = ""
                         ) -> Dict[str, Any]:
    """
    运行完整极限载荷分析（通用，支持EOG/NTM/EWS等所有风况）

    参数:
        df: 全量预处理后的DataFrame
        mapper: ChannelMapper实例
        extreme_segments: 极值段时间范围列表，None=全量分析
                          （由ConditionSlicer.detect_extreme_segments()生成）
        condition_type: 工况类型（如EOG/NTM），用于日志记录
    返回: {blade_results, wheel_results, summary_df, condition_type}
    """
    cond_label = condition_type or "通用"
    logger.info(f"开始极限载荷分析 (工况: {cond_label})...")
    time_col = config.global_ch.time_col

    # 保留全量数据：整轮气动性能（推力/扭矩/功率）统计必须基于全量时程，
    # 与全量时序图表一致；极值段切片仅用于叶片极值载荷提取。
    df_full = df
    # 如果指定了极值段，截取数据
    if extreme_segments:
        masks = []
        for s, e in extreme_segments:
            masks.append((df[time_col] >= s) & (df[time_col] <= e))
        if masks:
            df = df[np.logical_or.reduce(masks)].copy()
        logger.info(f"极值段数据: {len(df)}行 ({len(extreme_segments)}个时段)")
    else:
        logger.info(f"全量极值分析: {len(df)}行")

    # 1. 单叶片分析
    blade_results = {}
    summary_rows = []
    for blade_id in mapper.blade_total_loads:
        blade_df = mapper.get_blade_total_load_df(df, blade_id)
        if blade_df.empty:
            continue
        analyzer = BladeExtremeAnalyzer(blade_id)
        res = analyzer.analyze_total_loads(blade_df)
        blade_results[blade_id] = res
        summary_rows.append(analyzer.get_summary_row())

        # 面板级分析（法向力分布）
        panel_df = mapper.get_blade_panel_load_df(df, blade_id, "normal_force")
        if not panel_df.empty:
            analyzer.analyze_panel_loads(panel_df, "normal_force")

    # 2. 整轮分析（使用全量数据，保证推力/扭矩/功率统计与时序图表一致）
    wheel_analyzer = WheelExtremeAnalyzer()
    wheel_results = wheel_analyzer.analyze_global_loads(df_full)
    imbalance = wheel_analyzer.analyze_blade_imbalance(blade_results)
    wheel_results.update(imbalance)

    # 3. 汇总表
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    logger.info(f"极限载荷分析完成: {len(blade_results)}叶片, "
                f"最不利叶片={imbalance.get('critical_blade', 'N/A')}")

    return {
        "blade_results": blade_results,
        "wheel_results": wheel_results,
        "summary_df": summary_df,
        "condition_type": condition_type,
    }


# ============================================================
# 向后兼容别名（旧代码仍可使用）
# ============================================================
# 类名别名
BladeEOGAnalyzer = BladeExtremeAnalyzer
WheelEOGAnalyzer = WheelExtremeAnalyzer

# 函数别名：run_eog_analysis 调用 run_extreme_analysis
def run_eog_analysis(df, mapper, eog_segments=None):
    """
    旧版EOG分析入口（向后兼容）
    内部调用通用化的run_extreme_analysis
    """
    return run_extreme_analysis(df, mapper, extreme_segments=eog_segments, condition_type="EOG")

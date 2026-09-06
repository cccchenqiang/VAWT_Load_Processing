"""
多叶片载荷合成模块 - 垂直轴风轮专属
功能：
1. 多叶片空间矢量合成（整机合力、倾覆弯矩）
2. 叶片载荷不平衡度计算
3. 旋转周期内载荷脉动特征统计
4. 面板级展向载荷积分（总载荷验证）
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from config import config, SynthesisConfig
from utils import compute_statistics

logger = logging.getLogger(__name__)


# ============================================================
# 叶片载荷合成
# ============================================================
class BladeSynthesis:
    """多叶片载荷合成器"""

    def __init__(self, cfg: Optional[SynthesisConfig] = None):
        self.cfg = cfg or config.synthesis

    def synthesize_total_loads(self, blade_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合成多叶片总载荷（法向+切向）
        blade_data: {blade_id: DataFrame with time, total_normal, total_tangential, azimuth}
        返回: 包含合成载荷的DataFrame
        """
        if not blade_data:
            return pd.DataFrame()

        # 以第一个叶片的时间为基准
        ref_id = list(blade_data.keys())[0]
        ref_df = blade_data[ref_id]
        time_col = "time" if "time" in ref_df.columns else config.global_ch.time_col

        result = pd.DataFrame({"time": ref_df[time_col].values})

        # 各叶片法向/切向之和（标量叠加）
        total_normal_sum = np.zeros(len(ref_df))
        total_tangential_sum = np.zeros(len(ref_df))
        normal_list = []
        tangential_list = []

        for bid, bdf in blade_data.items():
            if "total_normal" in bdf.columns:
                # 对齐长度
                n = min(len(bdf), len(result))
                total_normal_sum[:n] += bdf["total_normal"].values[:n]
                normal_list.append(bdf["total_normal"].values[:n])
            if "total_tangential" in bdf.columns:
                n = min(len(bdf), len(result))
                total_tangential_sum[:n] += bdf["total_tangential"].values[:n]
                tangential_list.append(bdf["total_tangential"].values[:n])

        result["total_normal_sum"] = total_normal_sum
        result["total_tangential_sum"] = total_tangential_sum
        result["combined_sum"] = np.sqrt(total_normal_sum ** 2 + total_tangential_sum ** 2)

        # 矢量合成（考虑方位角，将法向/切向转换到全局坐标）
        # 垂直轴风轮：法向沿径向，切向沿周向
        # F_global_x = sum(F_normal * cos(azimuth) - F_tangential * sin(azimuth))
        # F_global_y = sum(F_normal * sin(azimuth) + F_tangential * cos(azimuth))
        fx = np.zeros(len(ref_df))
        fy = np.zeros(len(ref_df))
        for bid, bdf in blade_data.items():
            n = min(len(bdf), len(result))
            if "azimuth" in bdf.columns:
                az = np.deg2rad(bdf["azimuth"].values[:n])
                fn = bdf["total_normal"].values[:n] if "total_normal" in bdf.columns else np.zeros(n)
                ft = bdf["total_tangential"].values[:n] if "total_tangential" in bdf.columns else np.zeros(n)
                fx[:n] += fn * np.cos(az) - ft * np.sin(az)
                fy[:n] += fn * np.sin(az) + ft * np.cos(az)

        result["global_Fx"] = fx
        result["global_Fy"] = fy
        result["global_F_mag"] = np.sqrt(fx ** 2 + fy ** 2)

        # 倾覆弯矩（假设力作用在叶片高度中心）
        if self.cfg.compute_overturning_moment:
            h = self.cfg.blade_height / 2.0
            result["overturning_moment"] = result["global_F_mag"] * h

        return result

    def compute_imbalance(self, blade_data: Dict[str, pd.DataFrame],
                          load_type: str = "total_normal") -> Dict[str, Any]:
        """
        计算叶片间载荷不平衡度
        """
        result = {}
        method = self.cfg.imbalance_method

        # 收集各叶片载荷时序
        load_series = {}
        for bid, bdf in blade_data.items():
            if load_type in bdf.columns:
                load_series[bid] = bdf[load_type].values

        if not load_series:
            return result

        # 对齐长度
        min_len = min(len(v) for v in load_series.values())
        loads = np.array([v[:min_len] for v in load_series.values()])

        # 逐时刻不平衡度
        if method == "std_over_mean":
            mean_per_t = np.mean(loads, axis=0)
            std_per_t = np.std(loads, axis=0)
            imbalance_per_t = np.where(mean_per_t != 0, std_per_t / np.abs(mean_per_t), 0)
        elif method == "max_minus_min":
            imbalance_per_t = np.max(loads, axis=0) - np.min(loads, axis=0)
        elif method == "max_over_min":
            min_per_t = np.min(np.abs(loads), axis=0)
            imbalance_per_t = np.where(min_per_t != 0,
                                       np.max(np.abs(loads), axis=0) / min_per_t, 0)
        else:
            mean_per_t = np.mean(loads, axis=0)
            std_per_t = np.std(loads, axis=0)
            imbalance_per_t = np.where(mean_per_t != 0, std_per_t / np.abs(mean_per_t), 0)

        result["imbalance_series"] = imbalance_per_t
        result["imbalance_mean"] = float(np.mean(imbalance_per_t))
        result["imbalance_max"] = float(np.max(imbalance_per_t))
        result["imbalance_std"] = float(np.std(imbalance_per_t))

        # 各叶片均值/极值对比
        blade_stats = {}
        for bid, vals in load_series.items():
            s = pd.Series(vals)
            blade_stats[bid] = compute_statistics(s, ("max", "min", "mean", "std", "rms"))
        result["blade_statistics"] = blade_stats

        return result

    def analyze_rotational_pulsation(self, df: pd.DataFrame,
                                     rpm: np.ndarray,
                                     load_col: str = "global_F_mag") -> Dict[str, Any]:
        """
        分析旋转周期内载荷脉动特征
        """
        result = {}
        if load_col not in df.columns or len(rpm) == 0:
            return result

        vals = df[load_col].values
        time_col = "time" if "time" in df.columns else config.global_ch.time_col
        t = df[time_col].values if time_col in df.columns else np.arange(len(vals))

        # 平均转速
        avg_rpm = float(np.mean(rpm))
        result["avg_rpm"] = avg_rpm
        if avg_rpm > 0:
            rot_freq_hz = avg_rpm / 60.0
            result["rotational_freq_hz"] = rot_freq_hz
            result["rotational_period_sec"] = 1.0 / rot_freq_hz

            # FFT分析（找主要脉动频率）
            n = len(vals)
            if n > 10:
                dt = np.median(np.diff(t))
                if dt > 0:
                    freqs = np.fft.rfftfreq(n, d=dt)
                    fft_vals = np.abs(np.fft.rfft(vals - np.mean(vals)))
                    # 找前5个峰值频率
                    top_idx = np.argsort(fft_vals)[-5:][::-1]
                    result["dominant_frequencies"] = [
                        {"freq_hz": float(freqs[i]),
                         "amplitude": float(fft_vals[i]),
                         "order": float(freqs[i] / rot_freq_hz) if rot_freq_hz > 0 else 0}
                        for i in top_idx
                    ]

        # 统计
        stats = compute_statistics(pd.Series(vals), ("max", "min", "mean", "std", "peak_to_peak"))
        result["statistics"] = stats
        # 脉动率（std/mean）
        result["pulsation_ratio"] = float(stats["std"] / stats["mean"]) if stats["mean"] != 0 else 0

        return result

    def integrate_panel_loads(self, panel_df: pd.DataFrame,
                              panel_height_df: Optional[pd.DataFrame] = None
                              ) -> Dict[str, Any]:
        """
        面板级分布载荷沿展向积分，验证总载荷
        panel_df: 列 PAN_0..PAN_n 的分布载荷 [N/m]
        panel_height_df: 各面板高度 [m]，用于积分权重
        """
        result = {}
        panel_cols = [c for c in panel_df.columns if c.startswith("PAN_")]
        if not panel_cols:
            return result

        # 简化积分：梯形法（假设等间距面板）
        loads = panel_df[panel_cols].values  # (time, n_panels)

        if panel_height_df is not None:
            h_cols = [c for c in panel_height_df.columns if c.startswith("PAN_")]
            if h_cols:
                heights = panel_height_df[h_cols].values
                # 用第一行的高度作为面板位置
                z = heights[0] if len(heights) > 0 else np.arange(len(panel_cols))
            else:
                z = np.arange(len(panel_cols))
        else:
            z = np.arange(len(panel_cols))

        # 梯形积分
        integrated = np.trapz(loads, x=z, axis=1)
        result["integrated_load_series"] = integrated
        result["integrated_mean"] = float(np.mean(integrated))
        result["integrated_max"] = float(np.max(integrated))
        result["panel_count"] = len(panel_cols)

        return result


# ============================================================
# 便捷函数
# ============================================================
def synthesize_all_blades(df: pd.DataFrame, mapper) -> Dict[str, Any]:
    """
    一键合成所有叶片载荷
    返回: {synthesis_df, imbalance, pulsation}
    """
    synth = BladeSynthesis()

    # 提取各叶片数据
    blade_data = {}
    for bid in mapper.blade_total_loads:
        bdf = mapper.get_blade_total_load_df(df, bid)
        if not bdf.empty:
            blade_data[bid] = bdf

    if not blade_data:
        return {}

    # 1. 载荷合成
    synth_df = synth.synthesize_total_loads(blade_data)

    # 2. 不平衡度
    imbalance_normal = synth.compute_imbalance(blade_data, "total_normal")
    imbalance_tangential = synth.compute_imbalance(blade_data, "total_tangential")

    # 3. 旋转脉动
    rpm_col = config.global_ch.rpm_col
    rpm = df[rpm_col].values if rpm_col in df.columns else np.array([])
    pulsation = synth.analyze_rotational_pulsation(synth_df, rpm, "global_F_mag")

    logger.info(f"叶片合成完成: {len(blade_data)}叶片, "
                f"不平衡度(法向)={imbalance_normal.get('imbalance_mean', 0):.4f}")

    return {
        "synthesis_df": synth_df,
        "imbalance_normal": imbalance_normal,
        "imbalance_tangential": imbalance_tangential,
        "pulsation": pulsation,
    }

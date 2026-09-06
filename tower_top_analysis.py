"""
塔顶载荷分析模块
校核对象：风轮几何中心的塔架顶点载荷

功能：
1. 塔顶水平推力（Momentary_Aerodynamic_Thrust_[N]）
2. 塔顶扭矩（Momentary_Aerodynamic_Torque_[Nm]）
3. 倾覆弯矩（全局X/Y向力矩合成 sqrt(Mx²+My²)）
4. 极限统计（max/min/mean + 发生时刻）
5. 疲劳等效载荷（Wöhler指数加权，塔架默认m=3）

说明：塔顶"合力"输出已取消——各风机半径/结构不同，推力与扭矩量纲不同，
强行矢量合成需要机型特定的折算系数，容易产生误导性结果，因此塔顶校核仅输出推力/扭矩/倾覆弯矩。

与叶片校核的区别：
- 叶片校核：法向/切向/合成载荷，m=10（复合材料）
- 塔顶校核：推力/扭矩/倾覆弯矩，m=3（焊接钢结构）
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import config, TowerTopConfig
from utils import compute_statistics, find_peak_time
from fatigue_analysis import analyze_channel_fatigue, equivalent_fatigue_load

logger = logging.getLogger(__name__)


class TowerTopAnalyzer:
    """塔顶载荷分析器"""

    def __init__(self, cfg: Optional[TowerTopConfig] = None):
        self.cfg = cfg or config.tower_top

    def extract_channels(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """提取塔顶载荷通道时序"""
        gc = config.global_ch
        channels = {}

        # 推力（直接复用）
        tc = self.cfg.thrust_col or gc.inst_thrust_col
        if tc in df.columns:
            channels["thrust"] = df[tc]

        # 扭矩
        tq = self.cfg.torque_col or gc.inst_torque_col
        if tq in df.columns:
            channels["torque"] = df[tq]

        # 倾覆弯矩（全局X/Y力矩合成）
        if self.cfg.use_global_moment:
            mx_col = self.cfg.moment_x_col or gc.moment_x_col
            my_col = self.cfg.moment_y_col or gc.moment_y_col
            if mx_col in df.columns and my_col in df.columns:
                mx = df[mx_col].values
                my = df[my_col].values
                channels["overturning_moment"] = pd.Series(
                    np.sqrt(mx ** 2 + my ** 2), index=df.index)

        # 注：塔顶合力（推力与扭矩矢量合成）已取消——每台风机半径/结构不同，折算系数因人而异，
        # 强行合成会引入误导性结果。塔顶校核仅输出推力与扭矩两个独立通道。
        return channels

    def analyze_extreme(self, channels: Dict[str, pd.Series],
                        time: np.ndarray) -> Dict[str, Dict]:
        """逐通道极限统计（含发生时刻）"""
        result = {}
        time_s = pd.Series(time)
        for name, s in channels.items():
            stats = compute_statistics(s, self.cfg.stats_metrics)
            t_max, v_max = find_peak_time(s, time_s, mode="max")
            t_min, v_min = find_peak_time(s, time_s, mode="min")
            result[name] = {
                "max": float(stats["max"]),
                "min": float(stats["min"]),
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "rms": float(stats["rms"]),
                "peak_to_peak": float(stats["peak_to_peak"]),
                "max_time": float(t_max) if t_max is not None else None,
                "min_time": float(t_min) if t_min is not None else None,
            }
        return result

    def analyze_fatigue(self, channels: Dict[str, pd.Series],
                        duration: float) -> Dict[str, Dict]:
        """逐通道疲劳等效载荷（Wöhler指数m加权）"""
        result = {}
        m = self.cfg.fatigue_m
        for name, s in channels.items():
            try:
                # 复用现有疲劳分析：雨流计数 + DEL
                fr = analyze_channel_fatigue(s.values, m=m,
                                             design_life_sec=config.fatigue.design_life_sec,
                                             duration=duration)
                result[name] = {
                    "del": float(fr.get("del", 0.0)),
                    "damage": float(fr.get("damage", 0.0)),
                    "cycle_count": int(fr.get("cycle_count", 0)),
                    "m": m,
                }
            except Exception as e:
                logger.warning(f"塔顶疲劳分析[{name}]失败: {e}")
                result[name] = {"del": 0.0, "damage": 0.0, "cycle_count": 0, "m": m}
        return result

    def analyze(self, df: pd.DataFrame, duration: Optional[float] = None,
                load_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        完整塔顶载荷分析
        df: 预处理后的DataFrame
        duration: 数据时长[s]（疲劳计算用），默认取时间列范围
        load_data: 若提供，直接使用该时序数据（batch用）
        """
        time_col = config.global_ch.time_col
        if time_col not in df.columns:
            return {"channels": {}, "extreme": {}, "fatigue": {}, "error": "无时间列"}

        time = df[time_col].values
        if duration is None:
            duration = float(time[-1] - time[0]) if len(time) > 1 else 1.0

        channels = self.extract_channels(df)
        extreme = self.analyze_extreme(channels, time)
        fatigue = self.analyze_fatigue(channels, duration)

        # 汇总表
        rows = []
        for name in channels:
            ex = extreme.get(name, {})
            fa = fatigue.get(name, {})
            rows.append({
                "通道": name,
                "最大值": ex.get("max"),
                "最大值时刻": ex.get("max_time"),
                "最小值": ex.get("min"),
                "最小值时刻": ex.get("min_time"),
                "均值": ex.get("mean"),
                "标准差": ex.get("std"),
                "等效疲劳载荷": fa.get("del"),
                "疲劳指数m": fa.get("m"),
            })
        summary_df = pd.DataFrame(rows)

        return {
            "channels": {k: v.values.tolist() for k, v in channels.items()},
            "time": time.tolist(),
            "extreme": extreme,
            "fatigue": fatigue,
            "summary_df": summary_df,
        }


def run_tower_top_analysis(df: pd.DataFrame, duration: Optional[float] = None) -> Dict:
    """塔顶载荷分析主入口"""
    analyzer = TowerTopAnalyzer()
    return analyzer.analyze(df, duration=duration)


# 供load_envelope复用的轻量接口
def tower_top_envelope_inputs(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """提取塔顶校核载荷时序（供批处理包络用）"""
    analyzer = TowerTopAnalyzer()
    channels = analyzer.extract_channels(df)
    return {k: v.values for k, v in channels.items()}

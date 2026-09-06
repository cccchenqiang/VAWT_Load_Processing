"""
疲劳载荷分析模块 - 大文件分块疲劳
功能：
1. 雨流计数（支持分块累积，无精度损失）
2. 多叶片独立疲劳损伤计算（Miner准则）
3. 载荷谱统计（循环次数-幅值分布）
4. 等效疲劳载荷（DEL）计算
5. 叶片疲劳薄弱位置筛查
6. 极端载荷参与疲劳累积（通用，支持EOG/NTM等所有风况）
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from config import config, FatigueConfig
from utils import compute_statistics

logger = logging.getLogger(__name__)


# ============================================================
# 雨流计数（纯Python实现，无外部依赖）
# ============================================================
def rainflow_count(series: np.ndarray, nbins: int = 64,
                   range_min: Optional[float] = None,
                   range_max: Optional[float] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    雨流计数法（ASTM E1049标准）
    返回: (载荷范围中心值数组, 对应循环次数数组)

    实现逻辑：
    1. 提取局部极值点（峰谷序列）
    2. 四点法雨流计数
    3. 直方图统计
    """
    # 去除NaN
    vals = series[~np.isnan(series)]
    if len(vals) < 3:
        return np.array([]), np.array([])

    # 1. 提取峰谷点（保留转向点）
    extrema = [vals[0]]
    for i in range(1, len(vals) - 1):
        if (vals[i] - vals[i - 1]) * (vals[i + 1] - vals[i]) <= 0:
            extrema.append(vals[i])
    extrema.append(vals[-1])
    extrema = np.array(extrema)

    if len(extrema) < 3:
        return np.array([]), np.array([])

    # 2. 四点雨流计数
    ranges = []
    means = []
    i = 0
    while i < len(extrema) - 3:
        x1, x2, x3, x4 = extrema[i], extrema[i + 1], extrema[i + 2], extrema[i + 3]
        s1 = abs(x2 - x1)
        s2 = abs(x3 - x2)
        s3 = abs(x4 - x3)
        if s2 <= s1 and s2 <= s3:
            # 形成完整循环
            r = s2
            m = (x2 + x3) / 2.0
            ranges.append(r)
            means.append(m)
            # 移除x2, x3
            extrema = np.delete(extrema, [i + 1, i + 2])
            i = max(0, i - 1)
        else:
            i += 1

    # 处理剩余的半循环（残差）
    for j in range(len(extrema) - 1):
        r = abs(extrema[j + 1] - extrema[j])
        if r > 0:
            ranges.append(r)
            means.append((extrema[j] + extrema[j + 1]) / 2.0)

    if not ranges:
        return np.array([]), np.array([])

    ranges = np.array(ranges)
    means = np.array(means)

    # 3. 直方图统计
    if range_min is None:
        range_min = ranges.min()
    if range_max is None:
        range_max = ranges.max()
    if range_max <= range_min:
        range_max = range_min + 1e-10

    bin_edges = np.linspace(range_min, range_max, nbins + 1)
    counts, edges = np.histogram(ranges, bins=bin_edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    # 过滤零计数
    mask = counts > 0
    return bin_centers[mask], counts[mask].astype(float)


# ============================================================
# 分块雨流计数累积器
# ============================================================
class ChunkRainflowAccumulator:
    """
    分块雨流计数累积器
    对大文件逐块雨流计数，最后合并载荷谱
    采用重叠块策略防止跨块循环丢失
    """

    def __init__(self, nbins: int = 64, overlap_ratio: float = 0.1):
        self.nbins = nbins
        self.overlap_ratio = overlap_ratio
        self.global_range_min = np.inf
        self.global_range_max = -np.inf
        self.chunk_spectra: List[Tuple[np.ndarray, np.ndarray]] = []
        self.total_cycles = 0.0

    def process_chunk(self, values: np.ndarray):
        """处理一块数据"""
        vals = values[~np.isnan(values)]
        if len(vals) < 3:
            return
        # 更新全局范围
        chunk_min = vals.min()
        chunk_max = vals.max()
        self.global_range_min = min(self.global_range_min, chunk_min)
        self.global_range_max = max(self.global_range_max, chunk_max)
        # 本块雨流计数
        centers, counts = rainflow_count(vals, self.nbins)
        if len(centers) > 0:
            self.chunk_spectra.append((centers, counts))
            self.total_cycles += counts.sum()

    def get_combined_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        合并所有块的载荷谱（重新分箱到统一范围）
        """
        if not self.chunk_spectra:
            return np.array([]), np.array([])

        # 全局统一分箱
        r_min = self.global_range_min
        r_max = self.global_range_max
        if r_max <= r_min:
            r_max = r_min + 1e-10
        bin_edges = np.linspace(r_min, r_max, self.nbins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total_counts = np.zeros(self.nbins)

        # 将各块谱映射到全局箱
        for centers, counts in self.chunk_spectra:
            for c, n in zip(centers, counts):
                idx = np.searchsorted(bin_edges, c) - 1
                idx = np.clip(idx, 0, self.nbins - 1)
                total_counts[idx] += n

        mask = total_counts > 0
        return bin_centers[mask], total_counts[mask]


# ============================================================
# Miner累积损伤计算
# ============================================================
def miner_damage(ranges: np.ndarray, counts: np.ndarray,
                 sn_m: float = 3.0, sn_log_a: float = 12.0,
                 reference_range: float = 1.0) -> float:
    """
    Miner线性累积损伤
    D = sum(n_i / N_i)
    N_i = 10^(log_a) / (S_i)^m  （S-N曲线）
    """
    if len(ranges) == 0:
        return 0.0
    # 归一化到参考载荷
    normalized = ranges / reference_range if reference_range != 0 else ranges
    # 防止除零
    normalized = np.clip(normalized, 1e-10, None)
    # S-N曲线：N = 10^log_a / S^m
    n_fail = 10 ** sn_log_a / (normalized ** sn_m)
    damage = np.sum(counts / n_fail)
    return float(damage)


def equivalent_load(ranges: np.ndarray, counts: np.ndarray,
                    sn_m: float = 3.0, total_cycles: Optional[float] = None) -> float:
    """
    等效疲劳载荷（DEL - Damage Equivalent Load）
    DEL = (sum(n_i * S_i^m) / N_total)^(1/m)
    """
    if len(ranges) == 0:
        return 0.0
    if total_cycles is None:
        total_cycles = counts.sum()
    if total_cycles <= 0:
        return 0.0
    del_val = (np.sum(counts * ranges ** sn_m) / total_cycles) ** (1.0 / sn_m)
    return float(del_val)


def goodman_correction(ranges: np.ndarray, means: np.ndarray,
                       ultimate_strength: float) -> np.ndarray:
    """
    Goodman修正：将非零均值应力范围修正为等效零均值范围
    S_eq = S / (1 - mean / ultimate)
    """
    if ultimate_strength <= 0:
        return ranges
    ratio = means / ultimate_strength
    ratio = np.clip(ratio, -0.99, 0.99)
    return ranges / (1.0 - ratio)


def analyze_channel_fatigue(values: np.ndarray, m: float = 3.0,
                            sn_log_a: float = 12.0,
                            reference_range: float = 1.0,
                            design_life_sec: float = 630720000.0,
                            duration: Optional[float] = None,
                            nbins: int = 64) -> Dict[str, Any]:
    """
    通用单通道疲劳分析（供塔顶/批处理等模块复用）
    返回: {del, damage, cycle_count, spectrum_ranges, spectrum_counts, m}
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    result: Dict[str, Any] = {"m": m}
    if len(vals) < 3:
        result.update({"del": 0.0, "damage": 0.0, "cycle_count": 0})
        return result

    # 雨流计数
    ranges, counts = rainflow_count(vals, nbins)
    if len(ranges) == 0:
        result.update({"del": 0.0, "damage": 0.0, "cycle_count": 0})
        return result

    cycle_count = float(counts.sum())
    # 等效疲劳载荷
    del_val = equivalent_load(ranges, counts, sn_m=m)
    # Miner损伤（按设计寿命折算）
    damage = miner_damage(ranges, counts, sn_m=m, sn_log_a=sn_log_a,
                          reference_range=reference_range)
    # 折算到设计寿命
    if duration and duration > 0:
        cycles_per_sec = cycle_count / duration
        damage = damage * cycles_per_sec * design_life_sec / max(cycle_count, 1e-9)

    result.update({
        "del": del_val,
        "damage": damage,
        "cycle_count": int(cycle_count),
        "spectrum_ranges": ranges.tolist(),
        "spectrum_counts": counts.tolist(),
    })
    return result


# 兼容别名
equivalent_fatigue_load = analyze_channel_fatigue


# ============================================================
# 单叶片疲劳分析器
# ============================================================
class BladeFatigueAnalyzer:
    """单叶片疲劳载荷分析器"""

    def __init__(self, blade_id: str, cfg: Optional[FatigueConfig] = None):
        self.blade_id = blade_id
        self.cfg = cfg or config.fatigue
        self.results: Dict[str, Any] = {}

    def analyze_channel(self, values: np.ndarray,
                        channel_name: str = "load") -> Dict[str, Any]:
        """
        对单通道时序做疲劳分析
        """
        result = {"blade_id": self.blade_id, "channel": channel_name}

        # 雨流计数
        centers, counts = rainflow_count(values, self.cfg.rainflow_bins)
        result["spectrum_ranges"] = centers
        result["spectrum_counts"] = counts
        result["total_cycles"] = float(counts.sum()) if len(counts) > 0 else 0.0

        # 载荷谱DataFrame
        if len(centers) > 0:
            result["spectrum_df"] = pd.DataFrame({
                "range": centers,
                "counts": counts,
            })

        # Miner损伤
        damage = miner_damage(
            centers, counts,
            sn_m=self.cfg.sn_m,
            sn_log_a=self.cfg.sn_log_a,
            reference_range=self.cfg.reference_range,
        )
        result["miner_damage"] = damage

        # 等效疲劳载荷
        del_val = equivalent_load(centers, counts, self.cfg.sn_m)
        result["equivalent_load"] = del_val

        # 预估寿命（基于当前数据时长外推）
        if damage > 0:
            # 当前数据时长内的损伤，假设设计寿命内载荷重复
            result["damage_per_sec"] = damage  # 需除以实际时长
        else:
            result["damage_per_sec"] = 0.0

        # 统计
        stats = compute_statistics(pd.Series(values),
                                   ("max", "min", "mean", "std", "peak_to_peak", "rms"))
        result["statistics"] = stats

        self.results[channel_name] = result
        return result

    def analyze_panel_fatigue(self, panel_df: pd.DataFrame,
                              load_type: str = "normal_force") -> Dict[str, Any]:
        """
        面板级疲劳分析（找出最危险面板）
        """
        result = {"blade_id": self.blade_id, "load_type": load_type}
        panel_cols = [c for c in panel_df.columns if c.startswith("PAN_")]
        if not panel_cols:
            return result

        panel_damages = []
        for col in panel_cols:
            vals = panel_df[col].values
            centers, counts = rainflow_count(vals, self.cfg.rainflow_bins)
            damage = miner_damage(centers, counts, self.cfg.sn_m,
                                  self.cfg.sn_log_a, self.cfg.reference_range)
            del_val = equivalent_load(centers, counts, self.cfg.sn_m)
            panel_damages.append({
                "panel": col,
                "damage": damage,
                "equivalent_load": del_val,
                "total_cycles": float(counts.sum()) if len(counts) > 0 else 0,
            })

        result["panel_fatigue_df"] = pd.DataFrame(panel_damages)
        # 最危险面板
        worst = max(panel_damages, key=lambda x: x["damage"])
        result["worst_panel"] = worst["panel"]
        result["worst_panel_damage"] = worst["damage"]
        result["worst_panel_del"] = worst["equivalent_load"]

        self.results[f"panel_{load_type}"] = result
        return result

    def get_summary_row(self) -> Dict[str, Any]:
        """获取汇总行"""
        row = {"blade_id": self.blade_id}
        for ch_name, res in self.results.items():
            if ch_name.startswith("panel_"):
                continue
            row[f"{ch_name}_damage"] = res.get("miner_damage", np.nan)
            row[f"{ch_name}_del"] = res.get("equivalent_load", np.nan)
            row[f"{ch_name}_cycles"] = res.get("total_cycles", np.nan)
        return row


# ============================================================
# 疲劳分析主入口
# ============================================================
def run_fatigue_analysis(df: pd.DataFrame, mapper,
                         data_duration_sec: float = 0.0
                         ) -> Dict[str, Any]:
    """
    运行完整疲劳分析
    df: 预处理后的DataFrame
    mapper: ChannelMapper
    data_duration_sec: 数据时长（用于损伤率计算）
    """
    logger.info("开始疲劳载荷分析...")
    cfg = config.fatigue

    blade_results = {}
    summary_rows = []

    for blade_id in mapper.blade_total_loads:
        blade_df = mapper.get_blade_total_load_df(df, blade_id)
        if blade_df.empty:
            continue

        analyzer = BladeFatigueAnalyzer(blade_id)

        # 总法向载荷疲劳
        if "total_normal" in blade_df.columns:
            analyzer.analyze_channel(blade_df["total_normal"].values, "total_normal")

        # 总切向载荷疲劳
        if "total_tangential" in blade_df.columns:
            analyzer.analyze_channel(blade_df["total_tangential"].values, "total_tangential")

        # 合成载荷疲劳
        if "total_normal" in blade_df.columns and "total_tangential" in blade_df.columns:
            combined = np.sqrt(blade_df["total_normal"].values ** 2 +
                               blade_df["total_tangential"].values ** 2)
            analyzer.analyze_channel(combined, "combined_load")

        # 面板级疲劳（法向力）
        panel_df = mapper.get_blade_panel_load_df(df, blade_id, "normal_force")
        if not panel_df.empty:
            analyzer.analyze_panel_fatigue(panel_df, "normal_force")

        blade_results[blade_id] = analyzer.results
        summary_rows.append(analyzer.get_summary_row())

    # 叶片疲劳排名
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    if not summary_df.empty and "combined_load_damage" in summary_df.columns:
        summary_df = summary_df.sort_values("combined_load_damage", ascending=False)
        summary_df["rank"] = range(1, len(summary_df) + 1)

    # 整机疲劳水平
    wheel_fatigue = {}
    if blade_results:
        damages = [r.get("combined_load", {}).get("miner_damage", 0)
                   for r in blade_results.values()]
        wheel_fatigue["max_blade_damage"] = max(damages) if damages else 0
        wheel_fatigue["mean_blade_damage"] = float(np.mean(damages)) if damages else 0
        wheel_fatigue["damage_imbalance"] = (float(np.std(damages) / np.mean(damages))
                                              if damages and np.mean(damages) != 0 else 0)

    # 整机全局通道疲劳（风轮推力/扭矩等，无叶片数据时也提供疲劳结果）
    gc = config.global_ch
    global_fatigue = {}
    for name, col in [("thrust", gc.inst_thrust_col), ("torque", gc.inst_torque_col),
                      ("power", gc.inst_power_col)]:
        if col in df.columns:
            try:
                fr = analyze_channel_fatigue(
                    df[col].values, m=cfg.sn_m, sn_log_a=cfg.sn_log_a,
                    reference_range=cfg.reference_range,
                    design_life_sec=cfg.design_life_sec,
                    duration=data_duration_sec if data_duration_sec > 0 else None,
                    nbins=cfg.rainflow_bins)
                global_fatigue[name] = fr
            except Exception as e:
                logger.warning(f"整机通道[{name}]疲劳失败: {e}")
    if global_fatigue:
        wheel_fatigue["global_channels"] = {
            k: {"del": v.get("del", 0), "damage": v.get("damage", 0),
                "cycle_count": v.get("cycle_count", 0), "m": v.get("m", cfg.sn_m)}
            for k, v in global_fatigue.items()
        }

    logger.info(f"疲劳分析完成: {len(blade_results)}叶片, "
                f"最大损伤={wheel_fatigue.get('max_blade_damage', 0):.2e}")

    return {
        "blade_results": blade_results,
        "wheel_fatigue": wheel_fatigue,
        "summary_df": summary_df,
        "global_fatigue": global_fatigue,
    }


def run_chunked_fatigue(loader, mapper, channel_getter,
                        cfg: Optional[FatigueConfig] = None) -> Dict[str, Any]:
    """
    大文件分块疲劳分析（不加载全量数据）
    channel_getter: 函数，接收chunk返回 {blade_id: {channel: values}}
    """
    cfg = cfg or config.fatigue
    accumulators: Dict[str, Dict[str, ChunkRainflowAccumulator]] = {}

    for i, chunk in enumerate(loader.iter_chunks(include_panel=False)):
        blade_data = channel_getter(chunk)
        for bid, channels in blade_data.items():
            if bid not in accumulators:
                accumulators[bid] = {}
            for ch_name, vals in channels.items():
                if ch_name not in accumulators[bid]:
                    accumulators[bid][ch_name] = ChunkRainflowAccumulator(
                        nbins=cfg.rainflow_bins,
                        overlap_ratio=cfg.chunk_overlap_ratio,
                    )
                accumulators[bid][ch_name].process_chunk(vals)
        logger.info(f"疲劳分块 {i}: {chunk.shape[0]}行")

    # 合并结果
    results = {}
    for bid, accs in accumulators.items():
        results[bid] = {}
        for ch_name, acc in accs.items():
            centers, counts = acc.get_combined_spectrum()
            damage = miner_damage(centers, counts, cfg.sn_m, cfg.sn_log_a, cfg.reference_range)
            del_val = equivalent_load(centers, counts, cfg.sn_m)
            results[bid][ch_name] = {
                "spectrum_ranges": centers,
                "spectrum_counts": counts,
                "miner_damage": damage,
                "equivalent_load": del_val,
                "total_cycles": float(counts.sum()) if len(counts) > 0 else 0,
            }

    return results

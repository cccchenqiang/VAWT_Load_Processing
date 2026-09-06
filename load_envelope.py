"""
载荷包络引擎 - 多工况(DLC)批处理核心

目标：跨所有工况找到能包络所有载荷的载荷，用于部件校核

功能：
1. 极限载荷包络：逐通道跨工况取 max/min + 来源工况 + 发生时刻
2. 组合载荷包络：多载荷分量同时作用的最不利组合（叶片法向+切向合成）
3. 疲劳等效包络：Wöhler指数加权（载荷取m次幂加权合成）
   DEL_equiv = (Σ w_i · DEL_i^m)^(1/m)
4. 叶片校核载荷：每叶片法向/切向/合成（m=10复合材料）
5. 塔顶校核载荷：推力/扭矩/倾覆弯矩（m=3焊接钢结构）
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import config
from fatigue_analysis import analyze_channel_fatigue
from tower_top_analysis import TowerTopAnalyzer

logger = logging.getLogger(__name__)


# ============================================================
# 极限包络计算
# ============================================================
def extreme_envelope(case_results: List[Dict]) -> pd.DataFrame:
    """
    跨工况极限载荷包络
    case_results: [{case_id, channel_stats: {通道: {max/min/max_time/min_time/...}}}]
    返回DataFrame：每通道上限/下限/来源工况/发生时刻
    """
    # 收集所有通道
    all_channels = set()
    for cr in case_results:
        all_channels.update(cr.get("channel_stats", {}).keys())

    rows = []
    for ch in sorted(all_channels):
        max_val, max_src, max_t = -np.inf, "-", None
        min_val, min_src, min_t = np.inf, "-", None
        for cr in case_results:
            stats = cr.get("channel_stats", {}).get(ch)
            if not stats:
                continue
            vmax = stats.get("max")
            vmin = stats.get("min")
            if vmax is not None and not np.isnan(vmax) and vmax > max_val:
                max_val = vmax
                max_src = cr["case_id"]
                max_t = stats.get("max_time")
            if vmin is not None and not np.isnan(vmin) and vmin < min_val:
                min_val = vmin
                min_src = cr["case_id"]
                min_t = stats.get("min_time")
        if max_val == -np.inf:
            max_val = np.nan
        if min_val == np.inf:
            min_val = np.nan
        rows.append({
            "通道": ch,
            "上限": max_val,
            "上限来源": max_src,
            "上限时刻": max_t,
            "下限": min_val,
            "下限来源": min_src,
            "下限时刻": min_t,
        })
    return pd.DataFrame(rows)


# ============================================================
# Wöhler指数加权疲劳包络
# ============================================================
def fatigue_envelope(case_results: List[Dict], m: float = 3.0) -> pd.DataFrame:
    """
    跨工况Wöhler指数加权疲劳等效包络
    DEL_equiv = (Σ w_i · DEL_i^m)^(1/m)
    case_results: [{case_id, weight, channel_del: {通道: DEL}, ...}]
    """
    all_channels = set()
    for cr in case_results:
        all_channels.update(cr.get("channel_del", {}).keys())

    rows = []
    for ch in sorted(all_channels):
        weighted = 0.0
        total_w = 0.0
        details = []
        for cr in case_results:
            del_i = cr.get("channel_del", {}).get(ch)
            if del_i is None or np.isnan(del_i):
                continue
            w = cr.get("weight", 1.0)
            weighted += w * (abs(del_i) ** m)
            total_w += w
            details.append(f"{cr['case_id']}:{del_i:.1f}(w{w})")
        del_equiv = (weighted / total_w) ** (1.0 / m) if total_w > 0 else 0.0
        rows.append({
            "通道": ch,
            "等效疲劳载荷": del_equiv,
            "Wöhler指数m": m,
            "工况明细": "; ".join(details),
        })
    return pd.DataFrame(rows)


# ============================================================
# 叶片校核载荷包络（垂直轴：侧面固定，法向+切向+合成）
# ============================================================
def blade_check_envelope(case_results: List[Dict], num_blades: int = 3,
                         m: float = 10.0) -> Dict[str, pd.DataFrame]:
    """
    叶片校核载荷包络（逐叶片）
    每叶片：法向(Total_Normal_Load)、切向(Total_Tangential_Load)、合成(sqrt(N²+T²))
    极限包络 + 疲劳等效包络
    """
    results = {}
    for n in range(1, num_blades + 1):
        bid = f"Blade_{n}"
        # 收集该叶片各工况的通道数据
        normal_ch = f"{bid} 法向载荷"
        tangential_ch = f"{bid} 切向载荷"
        combined_ch = f"{bid} 合成载荷"

        # 极限包络（重新组织channel_stats结构）
        sub_cases = []
        for cr in case_results:
            sub = {
                "case_id": cr["case_id"],
                "weight": cr.get("weight", 1.0),
                "channel_stats": {},
                "channel_del": {},
            }
            cs = cr.get("channel_stats", {})
            cd = cr.get("channel_del", {})
            for k in (normal_ch, tangential_ch, combined_ch):
                if k in cs:
                    sub["channel_stats"][k] = cs[k]
                if k in cd:
                    sub["channel_del"][k] = cd[k]
            sub_cases.append(sub)

        extreme = extreme_envelope(sub_cases)
        fatigue = fatigue_envelope(sub_cases, m=m)
        results[bid] = {
            "extreme": extreme,
            "fatigue": fatigue,
        }
    return results


# ============================================================
# 塔顶校核载荷包络
# ============================================================
def tower_top_check_envelope(case_results: List[Dict],
                             m: float = 3.0) -> Dict[str, pd.DataFrame]:
    """
    塔顶校核载荷包络（仅塔顶通道）
    通道：塔顶-thrust/torque/overturning_moment/resultant
    """
    # 仅筛选塔顶通道
    tower_cases = []
    for cr in case_results:
        sub = {
            "case_id": cr["case_id"],
            "weight": cr.get("weight", 1.0),
            "channel_stats": {k: v for k, v in cr.get("channel_stats", {}).items()
                              if k.startswith("塔顶-")},
            "channel_del": {k: v for k, v in cr.get("channel_del", {}).items()
                            if k.startswith("塔顶-")},
        }
        tower_cases.append(sub)
    extreme = extreme_envelope(tower_cases)
    fatigue = fatigue_envelope(tower_cases, m=m)
    return {
        "extreme": extreme,
        "fatigue": fatigue,
    }


# ============================================================
# 单工况分析：提取包络所需通道统计
# ============================================================
def analyze_case_for_envelope(df: pd.DataFrame, mapper, info,
                              case_id: str, weight: float = 1.0,
                              condition_type: str = "",
                              blade_m: Optional[float] = None,
                              tower_m: Optional[float] = None) -> Dict:
    """
    从单个工况DataFrame提取包络所需数据（不重复跑完整分析）
    提取：
    - 叶片校核通道（法向/切向/合成，逐叶片）的极值统计 + DEL
    - 塔顶校核通道（推力/扭矩/倾覆弯矩/合力）的极值统计 + DEL
    """
    time_col = config.global_ch.time_col
    if time_col not in df.columns:
        return {}
    # B4修复：叶片/塔顶疲劳指数 m 可配置（默认取批处理配置）
    blade_m = blade_m or config.batch.fatigue_m.get("blade", 10)
    tower_m = tower_m or config.batch.fatigue_m.get("tower", 3)

    time = df[time_col].values
    duration = float(time[-1] - time[0]) if len(time) > 1 else 1.0

    channel_stats = {}
    channel_del = {}

    # 1. 叶片校核通道
    for bid in mapper.blade_total_loads:
        bdf = mapper.get_blade_total_load_df(df, bid)
        if bdf.empty:
            continue
        tn = bdf["total_normal"].values if "total_normal" in bdf.columns else None
        tt = bdf["total_tangential"].values if "total_tangential" in bdf.columns else None
        if tn is None and tt is None:
            continue

        # 法向
        if tn is not None:
            _collect_channel(bid + " 法向载荷", tn, time, duration, channel_stats, channel_del,
                             fatigue_m_blade=blade_m)
        # 切向
        if tt is not None:
            _collect_channel(bid + " 切向载荷", tt, time, duration, channel_stats, channel_del,
                             fatigue_m_blade=blade_m)
        # 合成
        if tn is not None and tt is not None:
            comb = np.sqrt(tn ** 2 + tt ** 2)
            _collect_channel(bid + " 合成载荷", comb, time, duration, channel_stats, channel_del,
                             fatigue_m_blade=blade_m)

    # 2. 塔顶校核通道
    tower = TowerTopAnalyzer()
    tch = tower.extract_channels(df)
    for name, s in tch.items():
        chname = f"塔顶-{name}"
        _collect_channel(chname, s.values, time, duration, channel_stats, channel_del,
                         fatigue_m_blade=tower_m)

    return {
        "case_id": case_id,
        "weight": weight,
        "condition_type": condition_type,
        "channel_stats": channel_stats,
        "channel_del": channel_del,
    }


def _collect_channel(name, values, time, duration, channel_stats, channel_del,
                     fatigue_m_blade: float = 10.0, fatigue_m_tower: float = 3.0):
    """收集单通道极值统计和DEL
    B4修复：疲劳指数 m 使用调用方传入的值（叶片通道用 blade_m，塔顶通道用 tower_m）
    """
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return
    stats = {
        "max": float(np.nanmax(vals)),
        "min": float(np.nanmin(vals)),
        "mean": float(np.nanmean(vals)),
        "std": float(np.nanstd(vals)),
    }
    # 发生时刻
    if len(time) == len(vals) and len(vals) > 0:
        idx_max = np.nanargmax(vals)
        idx_min = np.nanargmin(vals)
        stats["max_time"] = float(time[idx_max])
        stats["min_time"] = float(time[idx_min])
    channel_stats[name] = stats

    # 疲劳DEL：叶片通道用 blade_m，塔顶通道用 tower_m
    m = fatigue_m_tower if "塔顶-" in name else fatigue_m_blade
    try:
        fr = analyze_channel_fatigue(vals, m=m,
                                     design_life_sec=config.fatigue.design_life_sec,
                                     duration=duration)
        channel_del[name] = float(fr.get("del", 0.0))
    except Exception as e:
        logger.warning(f"通道[{name}]疲劳计算失败: {e}")
        channel_del[name] = 0.0


# ============================================================
# 主入口：生成完整包络
# ============================================================
def build_envelope(case_results: List[Dict], num_blades: int = 3,
                   blade_m: float = 10.0, tower_m: float = 3.0) -> Dict:
    """
    生成完整载荷包络
    case_results: analyze_case_for_envelope() 的返回值列表
    """
    if not case_results:
        return {"error": "无工况数据"}

    # 叶片校核包络
    blade_envelope = blade_check_envelope(case_results, num_blades=num_blades, m=blade_m)
    # 塔顶校核包络
    tower_envelope = tower_top_check_envelope(case_results, m=tower_m)
    # 全局通道包络（所有通道汇总）
    global_extreme = extreme_envelope(case_results)
    global_fatigue = fatigue_envelope(case_results, m=tower_m)

    return {
        "blade_envelope": blade_envelope,
        "tower_envelope": tower_envelope,
        "global_extreme": global_extreme,
        "global_fatigue": global_fatigue,
        "num_cases": len(case_results),
        "blade_m": blade_m,
        "tower_m": tower_m,
    }

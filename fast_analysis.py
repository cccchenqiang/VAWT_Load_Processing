"""
fast_analysis.py — OpenFAST/AeroDyn .out 结果分析模块（独立于 QBlade 分析）
============================================================================
基于 fast_io 解析出的 FAST 数据，进行与 QBlade 类似的载荷分析：
1. 风轮载荷分析：推力 / 扭矩 / 功率 / 转速 / TSR（容错缺列）
2. 叶片节点载荷极值（若输出 B*N*Fx/Fy/Fn/Ft 等分布力）
3. 时序数据（降采样，供前端绘图）

容错原则：即使输出文件缺少部分变量，也能继续分析已存在的变量。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from fast_io import FastFileInfo, FASTChannelMapper, load_fast_data


# ============================================================
# 统计工具（与 QBlade 侧一致口径）
# ============================================================
def _stats(vals: np.ndarray) -> Dict[str, float]:
    """极值统计（全量数据）"""
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"max": None, "min": None, "mean": None, "std": None, "rms": None}
    return {
        "max": float(np.max(v)),
        "min": float(np.min(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "rms": float(np.sqrt(np.mean(v ** 2))),
    }


def _peak_time(vals: np.ndarray, t: np.ndarray, mode: str = "max") -> Optional[float]:
    """返回最值发生时刻"""
    if vals is None or len(vals) == 0 or t is None or len(t) != len(vals):
        return None
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return None
    idx = int(np.argmax(v)) if mode == "max" else int(np.argmin(v))
    return round(float(t[idx]), 4)


# ============================================================
# 主分析入口
# ============================================================
def run_fast_analysis(filepath: str, info: Optional[FastFileInfo] = None,
                      max_pts: int = 1200) -> Dict:
    """运行 FAST .out 文件分析，返回 JSON 友好结果字典。

    参数:
        filepath: .out 文件路径
        info: 已解析的 FastFileInfo（None=自动）
        max_pts: 时序降采样点数上限
    返回: results dict
    """
    if info is None:
        info = parse_fast_header_local(filepath)

    df = load_fast_data(filepath, header=info)
    if df.empty:
        return {"error": f"文件无有效数据: {filepath}", "file_info": _file_info(info)}

    mapper = FASTChannelMapper(info.columns, info.num_blades, info.num_nodes)
    t = df["Time"].to_numpy(dtype=float) if "Time" in df.columns else None

    results = {
        "file_info": _file_info(info),
        "overview": {},
        "rotor": {},
        "blades": {},
        "timeseries": {},
        "channels": mapper.global_channels,
        "missing": [],   # 期望输出但文件中缺失的变量（提示用户补输出）
    }

    # ---- 功率 ----
    if "Momentary_Aerodynamic_Power_[W]" in mapper.channels:
        col = mapper.channels["Momentary_Aerodynamic_Power_[W]"]
        if col in df.columns:
            v = df[col].to_numpy(dtype=float)
            results["rotor"]["power_stats"] = _stats(v)
            results["rotor"]["power_max_time"] = _peak_time(v, t)
            results["rotor"]["power_min_time"] = _peak_time(v, t, "min")
            results["overview"]["power_mean_kw"] = round(float(np.nanmean(v)) / 1000, 2)

    # ---- 转速 / TSR ----
    for sys_name, key in [("Rotational_Speed_[rpm]", "rpm_stats"),
                          ("Tip_Speed_Ratio_[-]", "tsr_stats")]:
        if sys_name in mapper.channels:
            col = mapper.channels[sys_name]
            if col in df.columns:
                v = df[col].to_numpy(dtype=float)
                results["rotor"][key] = _stats(v)

    # ---- 推力（水平合力优先，其次 z 分量；均可缺省跳过）----
    thrust_cols = mapper.get_horiz_force_cols()
    thrust_vec = None
    if thrust_cols:
        fx, fy = thrust_cols
        if fx in df.columns and fy in df.columns:
            fxv = df[fx].to_numpy(dtype=float)
            fyv = df[fy].to_numpy(dtype=float)
            thrust_vec = np.hypot(fxv, fyv)
            thrust_src = "sqrt(Fxh^2+Fyh^2)"
    if thrust_vec is None:
        tcol = mapper.get_thrust_col()
        if tcol and tcol in df.columns:
            thrust_vec = df[tcol].to_numpy(dtype=float)
            thrust_src = tcol
    if thrust_vec is not None:
        results["rotor"]["thrust_stats"] = _stats(thrust_vec)
        results["rotor"]["thrust_max_time"] = _peak_time(thrust_vec, t)
        results["rotor"]["thrust_min_time"] = _peak_time(thrust_vec, t, "min")
        results["rotor"]["thrust_source"] = thrust_src
        results["overview"]["thrust_mean"] = round(float(np.nanmean(thrust_vec)), 1)

    # ---- 扭矩（绕旋转轴力矩优先；无则用 P/ω 计算）----
    torque_vec = None
    torque_src = None
    mz_col = mapper.channels.get("Momentary_Aerodynamic_Torque_[Nm]")
    if mz_col and mz_col in df.columns:
        torque_vec = df[mz_col].to_numpy(dtype=float)
        torque_src = mz_col
    if torque_vec is None:
        # 尝试 P / ω
        pcol = mapper.channels.get("Momentary_Aerodynamic_Power_[W]")
        rcol = mapper.channels.get("Rotational_Speed_[rpm]")
        if pcol in df.columns and rcol in df.columns:
            p = df[pcol].to_numpy(dtype=float)
            rpm = df[rcol].to_numpy(dtype=float)
            omega = np.abs(rpm) * 2.0 * np.pi / 60.0
            with np.errstate(divide="ignore", invalid="ignore"):
                torque_vec = np.where(omega > 1e-6, p / np.where(omega > 1e-6, omega, 1.0), np.nan)
            torque_src = "P / omega"
    if torque_vec is not None:
        tv = np.asarray(torque_vec, dtype=float)
        results["rotor"]["torque_stats"] = _stats(tv)
        results["rotor"]["torque_max_time"] = _peak_time(tv, t)
        results["rotor"]["torque_min_time"] = _peak_time(tv, t, "min")
        results["rotor"]["torque_source"] = torque_src
        results["overview"]["torque_mean"] = round(float(np.nanmean(tv)), 1)

    # ---- 叶片节点载荷（若输出 B*N*Fx/Fy/Fn/Ft 等）----
    results["blades"] = _analyze_blade_nodes(df, mapper)

    # ---- 缺失提示：期望输出但未输出的 rotor/叶片变量 ----
    _fill_missing(df, mapper, results)

    # ---- 时序（降采样，供前端画图）----
    results["timeseries"] = _make_timeseries(df, mapper, max_pts)

    return results


def _analyze_blade_nodes(df, mapper: FASTChannelMapper) -> Dict:
    """叶片节点载荷极值分析（容错：无节点载荷列则返回空）"""
    blades: Dict[str, Dict] = {}
    for key, node in mapper.blade_node_cols.items():
        # key 形如 Blade_1_Node_1
        parts = key.split("_")
        blade_key = f"Blade_{parts[1]}"
        node_num = int(parts[3])
        if blade_key not in blades:
            blades[blade_key] = {"nodes": {}, "suffixes_available": []}
        node_vals = {}
        for suffix, col in node.items():
            if col in df.columns:
                v = df[col].to_numpy(dtype=float)
                node_vals[suffix] = _stats(v)
                blades[blade_key]["suffixes_available"].append(suffix)
        if node_vals:
            blades[blade_key]["nodes"][str(node_num)] = node_vals
    # 去重 suffixes
    for b in blades.values():
        b["suffixes_available"] = sorted(set(b["suffixes_available"]))
    return blades


def _fill_missing(df, mapper: FASTChannelMapper, results: Dict):
    """提示期望输出但缺失的关键变量（帮助用户修正 OutList 输出设置）"""
    wanted = ["RtAeroFxh", "RtAeroFyh", "RtAeroFzh",
              "RtAeroMxh", "RtAeroMyh", "RtAeroMzh",
              "RtAeroPwr", "RtSpeed", "RtTSR"]
    missing = []
    for w in wanted:
        if not mapper._norm.get(w) and w not in df.columns:
            missing.append(w)
    # 叶片载荷：任一叶片任一节点 Fx/Fy 缺失即提示
    blade_sample = ["B1N1Fx", "B1N1Fy", "B1N1Fn", "B1N1Ft", "B1N1Mm"]
    for w in blade_sample:
        if not mapper._norm.get(w) and w not in df.columns:
            missing.append(w)
    results["missing"] = missing


def _make_timeseries(df, mapper: FASTChannelMapper, max_pts: int) -> Dict:
    """降采样时序，供前端绘图（含 rotor 载荷通道）"""
    t = df["Time"].to_numpy(dtype=float)
    n = len(t)
    step = max(1, n // max_pts)
    # 采样索引含首尾
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    ch: Dict[str, List[float]] = {}
    # 优先输出系统通道
    sys_map = {
        "Momentary_Aerodynamic_Thrust_[N]": "thrust",
        "Momentary_Aerodynamic_Torque_[Nm]": "torque",
        "Momentary_Aerodynamic_Power_[W]": "power",
        "Rotational_Speed_[rpm]": "rpm",
        "Tip_Speed_Ratio_[-]": "tsr",
        "Abs_Meas._Wind_Vel._at_Hub_[m/s]": "wind_speed",
    }
    # 推力向量（组合水平合力优先）
    thrust_vec = None
    if mapper.get_horiz_force_cols():
        fx, fy = mapper.get_horiz_force_cols()
        if fx in df.columns and fy in df.columns:
            thrust_vec = np.hypot(df[fx].to_numpy(dtype=float), df[fy].to_numpy(dtype=float))
    if thrust_vec is None:
        tcol = mapper.get_thrust_col()
        if tcol and tcol in df.columns:
            thrust_vec = df[tcol].to_numpy(dtype=float)
    # 扭矩向量（绕轴力矩优先，其次 P/ω）
    torque_vec = None
    mz_col = mapper.channels.get("Momentary_Aerodynamic_Torque_[Nm]")
    if mz_col and mz_col in df.columns:
        torque_vec = df[mz_col].to_numpy(dtype=float)
    if torque_vec is None:
        pcol = mapper.channels.get("Momentary_Aerodynamic_Power_[W]")
        rcol = mapper.channels.get("Rotational_Speed_[rpm]")
        if pcol in df.columns and rcol in df.columns:
            p = df[pcol].to_numpy(dtype=float)
            rpm = df[rcol].to_numpy(dtype=float)
            omega = np.abs(rpm) * 2.0 * np.pi / 60.0
            with np.errstate(divide="ignore", invalid="ignore"):
                torque_vec = np.where(omega > 1e-6, p / np.where(omega > 1e-6, omega, 1.0), np.nan)
    for sys_name, key in sys_map.items():
        if key == "thrust":
            if thrust_vec is not None:
                ch[key] = [round(float(x), 3) for x in thrust_vec[idx]]
            continue
        if key == "torque":
            if torque_vec is not None:
                ch[key] = [round(float(x), 3) for x in torque_vec[idx]]
            continue
        col = mapper.channels.get(sys_name)
        if col and col in df.columns:
            v = df[col].to_numpy(dtype=float)
            ch[key] = [round(float(x), 3) for x in v[idx]]
    return {"time": [round(float(x), 3) for x in t[idx]], "channels": ch}


def _file_info(info: FastFileInfo) -> Dict:
    return {
        "filename": info.filename,
        "size_mb": round(info.file_size_mb, 2),
        "num_rows": info.num_rows,
        "num_cols": info.num_columns,
        "num_blades": info.num_blades,
        "num_nodes": info.num_nodes,
        "sample_rate": round(info.sample_rate_hz, 1),
        "duration": round(info.total_time, 2),
        "generator": info.generator,
        "condition_type": "FAST",
    }


def parse_fast_header_local(filepath: str) -> FastFileInfo:
    """本地解析（避免循环依赖）"""
    from fast_io import parse_fast_header
    return parse_fast_header(filepath)


# ============================================================
# 通道向量计算（供前端时序接口复用）
# ============================================================
FAST_CHANNEL_LABELS = {
    "thrust": "风轮推力",
    "torque": "风轮扭矩",
    "power": "风轮功率",
    "rpm": "转速",
    "tsr": "叶尖速比",
    "wind_speed": "风速",
}


def fast_channel_vector(df, mapper: FASTChannelMapper, key: str):
    """返回 FAST 通道全量向量 (values: np.ndarray, label: str)

    key 可取：
      - 特殊键 thrust/torque/power/rpm/tsr/wind_speed（自动计算/组合）
      - 或 df 原始列名（直接返回该列）
    若通道不可用返回 (None, label)。
    """
    if key == "thrust":
        if mapper.get_horiz_force_cols():
            fx, fy = mapper.get_horiz_force_cols()
            if fx in df.columns and fy in df.columns:
                return np.hypot(df[fx].to_numpy(dtype=float), df[fy].to_numpy(dtype=float)), "风轮推力"
        tcol = mapper.get_thrust_col()
        if tcol and tcol in df.columns:
            return df[tcol].to_numpy(dtype=float), "风轮推力"
        return None, "风轮推力"
    if key == "torque":
        mz = mapper.channels.get("Momentary_Aerodynamic_Torque_[Nm]")
        if mz and mz in df.columns:
            return df[mz].to_numpy(dtype=float), "风轮扭矩"
        pcol = mapper.channels.get("Momentary_Aerodynamic_Power_[W]")
        rcol = mapper.channels.get("Rotational_Speed_[rpm]")
        if pcol in df.columns and rcol in df.columns:
            p = df[pcol].to_numpy(dtype=float)
            rpm = df[rcol].to_numpy(dtype=float)
            omega = np.abs(rpm) * 2.0 * np.pi / 60.0
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(omega > 1e-6, p / np.where(omega > 1e-6, omega, 1.0), np.nan), "风轮扭矩"
        return None, "风轮扭矩"
    sys_map = {
        "power": ("Momentary_Aerodynamic_Power_[W]", "风轮功率"),
        "rpm": ("Rotational_Speed_[rpm]", "转速"),
        "tsr": ("Tip_Speed_Ratio_[-]", "叶尖速比"),
        "wind_speed": ("Abs_Meas._Wind_Vel._at_Hub_[m/s]", "风速"),
    }
    if key in sys_map:
        sys_name, label = sys_map[key]
        col = mapper.channels.get(sys_name)
        if col and col in df.columns:
            return df[col].to_numpy(dtype=float), label
        return None, label
    # 原始列名
    if key in df.columns:
        return df[key].to_numpy(dtype=float), key
    return None, key

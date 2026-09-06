"""数据质量与坐标/符号约定校验。"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from config import config
from utils import check_data_quality, find_wind_speed_col


def validate_load_data(df: pd.DataFrame, mapper=None) -> Dict[str, Any]:
    """检查载荷数据是否具备可靠分析所需的时间、通道和坐标信息。"""
    quality = check_data_quality(df)
    warnings = []
    errors = []
    checks = []

    time_col = config.global_ch.time_col
    time = df[time_col].to_numpy(dtype=float) if time_col in df.columns else np.array([])
    if time_col not in df.columns:
        errors.append(f"缺少时间通道: {time_col}")
    elif time.size == 0:
        errors.append("时间通道没有数据")
    else:
        finite_time = time[np.isfinite(time)]
        if finite_time.size != time.size:
            errors.append("时间通道包含 NaN 或 Inf")
        if time.size > 1:
            dt = np.diff(time)
            positive_dt = dt[dt > 0]
            if positive_dt.size == 0:
                errors.append("时间通道不是递增序列")
            else:
                median_dt = float(np.median(positive_dt))
                bad_dt = int(np.count_nonzero(dt <= 0))
                irregular = int(np.count_nonzero(
                    np.abs(dt - median_dt) > max(median_dt * 1e-3, 1e-12)
                ))
                checks.append({
                    "name": "时间步长",
                    "status": "warning" if bad_dt or irregular else "ok",
                    "median_dt_s": median_dt,
                    "sample_rate_hz": 1.0 / median_dt,
                    "non_positive_count": bad_dt,
                    "irregular_count": irregular,
                })
                if bad_dt:
                    errors.append(f"时间步长存在 {bad_dt} 个非正值")
                elif irregular:
                    warnings.append(f"时间步长存在 {irregular} 个明显不规则点")

    if quality["nan_count"]:
        warnings.append(f"包含 {quality['nan_count']} 个 NaN")
    if quality["inf_count"]:
        warnings.append(f"包含 {quality['inf_count']} 个 Inf")

    wind_col = find_wind_speed_col(df.columns)
    checks.append({
        "name": "风速通道",
        "status": "ok" if wind_col else "warning",
        "channel": wind_col,
    })
    if not wind_col:
        warnings.append("未找到可识别的风速通道，工况分类和阵风检测可能不可用")

    global_cfg = config.global_ch
    vector_cols = [
        global_cfg.thrust_x_col,
        global_cfg.thrust_y_col,
        global_cfg.thrust_z_col,
        global_cfg.moment_x_col,
        global_cfg.moment_y_col,
        global_cfg.moment_z_col,
    ]
    present_vector = [col for col in vector_cols if col in df.columns]
    missing_vector = [col for col in vector_cols if col not in df.columns]
    checks.append({
        "name": "全局坐标载荷",
        "status": "ok" if len(present_vector) == len(vector_cols) else "warning",
        "present": present_vector,
        "missing": missing_vector,
    })
    if missing_vector:
        warnings.append(f"全局力/力矩通道不完整，缺少 {len(missing_vector)} 个通道")

    scalar_thrust = global_cfg.inst_thrust_col
    if scalar_thrust in df.columns and global_cfg.thrust_x_col in df.columns:
        scalar = pd.to_numeric(df[scalar_thrust], errors="coerce").to_numpy()
        axial = pd.to_numeric(df[global_cfg.thrust_x_col], errors="coerce").to_numpy()
        valid = np.isfinite(scalar) & np.isfinite(axial)
        if np.count_nonzero(valid) > 3:
            corr = float(np.corrcoef(scalar[valid], axial[valid])[0, 1])
            checks.append({
                "name": "推力符号一致性提示",
                "status": "warning" if corr < 0 else "ok",
                "correlation": corr,
                "note": "仅作提示，最终正负号必须依据机型坐标定义确认",
            })
            if corr < -0.5:
                warnings.append("标量推力与全局 X 向力呈负相关，请确认坐标正方向和符号约定")

    azimuth = {}
    if mapper is not None:
        for blade_id, channels in mapper.blade_total_loads.items():
            col = channels.get("azimuth")
            if not col or col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").to_numpy()
            finite = values[np.isfinite(values)]
            if finite.size < 2:
                continue
            jumps = np.abs(np.diff(finite))
            wrap_jumps = int(np.count_nonzero(jumps > 180.0))
            azimuth[blade_id] = {
                "channel": col,
                "min_deg": float(np.min(finite)),
                "max_deg": float(np.max(finite)),
                "wrap_count": wrap_jumps,
            }
            if np.any(jumps > 360.0):
                warnings.append(f"{blade_id} 方位角存在超过 360 度的跳变")
        if azimuth:
            checks.append({"name": "叶片方位角", "status": "ok", "blades": azimuth})

    return {
        "status": "error" if errors else ("warning" if warnings else "ok"),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "quality": quality,
    }

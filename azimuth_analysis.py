"""垂直轴风轮方位角分箱、周期和叶片相位分析。"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def analyze_azimuth_loads(df: pd.DataFrame, mapper, bins: int = 36) -> Dict[str, Any]:
    """按方位角分箱统计叶片载荷，并计算一阶周期和叶片相位差。"""
    if bins < 4 or bins > 360:
        raise ValueError("方位角分箱数必须在 4 到 360 之间")

    blade_results: Dict[str, Any] = {}
    phase_inputs = []
    edges = np.linspace(0.0, 360.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    for blade_id, channels in mapper.blade_total_loads.items():
        az_col = channels.get("azimuth")
        if not az_col or az_col not in df.columns:
            continue
        azimuth = pd.to_numeric(df[az_col], errors="coerce").to_numpy(dtype=float) % 360.0
        load_columns = {
            name: channels.get(name)
            for name in ("total_normal", "total_tangential")
            if channels.get(name) in df.columns
        }
        if not load_columns:
            continue

        valid = np.isfinite(azimuth)
        if not np.any(valid):
            continue
        bin_index = np.clip(np.digitize(azimuth[valid], edges, right=False) - 1, 0, bins - 1)
        stats: Dict[str, Any] = {}
        numeric_values: Dict[str, np.ndarray] = {}
        for name, column in load_columns.items():
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)[valid]
            finite = np.isfinite(values)
            numeric_values[name] = values
            records = []
            for idx, center in enumerate(centers):
                selected = values[(bin_index == idx) & finite]
                records.append({
                    "azimuth_deg": float(center),
                    "count": int(selected.size),
                    "mean": float(np.mean(selected)) if selected.size else None,
                    "std": float(np.std(selected)) if selected.size else None,
                    "min": float(np.min(selected)) if selected.size else None,
                    "max": float(np.max(selected)) if selected.size else None,
                })
            stats[name] = records

        combined = None
        if "total_normal" in numeric_values and "total_tangential" in numeric_values:
            combined = np.sqrt(
                numeric_values["total_normal"] ** 2
                + numeric_values["total_tangential"] ** 2
            )
            stats["combined"] = _bin_records(combined, bin_index, centers)

        periodic = {}
        for name, values in numeric_values.items():
            periodic[name] = _first_harmonic(azimuth[valid], values)
        if combined is not None:
            periodic["combined"] = _first_harmonic(azimuth[valid], combined)

        blade_results[blade_id] = {
            "azimuth_column": az_col,
            "bins": bins,
            "statistics": stats,
            "periodic": periodic,
        }
        phase_values = combined if combined is not None else numeric_values.get("total_normal")
        phase_inputs.append((blade_id, azimuth[valid], phase_values))

    phase = _phase_relationships(phase_inputs)
    return {
        "bins": bins,
        "blade_results": blade_results,
        "phase_relationships": phase,
    }


def _bin_records(values: np.ndarray, bin_index: np.ndarray,
                 centers: np.ndarray) -> List[Dict[str, Any]]:
    records = []
    finite = np.isfinite(values)
    for idx, center in enumerate(centers):
        selected = values[(bin_index == idx) & finite]
        records.append({
            "azimuth_deg": float(center),
            "count": int(selected.size),
            "mean": float(np.mean(selected)) if selected.size else None,
            "std": float(np.std(selected)) if selected.size else None,
            "min": float(np.min(selected)) if selected.size else None,
            "max": float(np.max(selected)) if selected.size else None,
        })
    return records


def _first_harmonic(azimuth: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(azimuth) & np.isfinite(values)
    if np.count_nonzero(valid) < 4:
        return {"mean": 0.0, "amplitude": 0.0, "phase_deg": 0.0, "samples": int(np.count_nonzero(valid))}
    angle = np.deg2rad(azimuth[valid])
    design = np.column_stack((np.ones(angle.size), np.cos(angle), np.sin(angle)))
    coeff, _, _, _ = np.linalg.lstsq(design, values[valid], rcond=None)
    amplitude = float(np.hypot(coeff[1], coeff[2]))
    phase = float(np.rad2deg(np.arctan2(-coeff[2], coeff[1])) % 360.0)
    return {
        "mean": float(coeff[0]),
        "amplitude": amplitude,
        "phase_deg": phase,
        "samples": int(np.count_nonzero(valid)),
    }


def _phase_relationships(inputs) -> List[Dict[str, Any]]:
    """用每个叶片的一阶谐波相位计算相对相位差。"""
    phases = []
    for blade_id, azimuth, values in inputs:
        harmonic = _first_harmonic(azimuth, values)
        phases.append((blade_id, harmonic["phase_deg"]))
    if not phases:
        return []
    reference_id, reference_phase = phases[0]
    return [
        {
            "reference_blade": reference_id,
            "blade": blade_id,
            "phase_difference_deg": float((phase - reference_phase + 180.0) % 360.0 - 180.0),
        }
        for blade_id, phase in phases
    ]

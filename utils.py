"""
工具函数模块
内存监控、统计计算、日志配置、数据导出等
"""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from functools import wraps

import numpy as np
import pandas as pd

from config import config


# ============================================================
# 日志配置
# ============================================================
def setup_logger(name: str = "VAWT_Load", level: str = "INFO") -> logging.Logger:
    """配置日志"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ============================================================
# 内存监控
# ============================================================
def get_memory_usage_mb() -> float:
    """获取当前进程内存使用 [MB]"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def memory_limit_check(limit_mb: float) -> bool:
    """检查是否超过内存上限"""
    usage = get_memory_usage_mb()
    if usage > limit_mb:
        logging.getLogger(__name__).warning(
            f"内存使用 {usage:.1f}MB 超过上限 {limit_mb:.1f}MB"
        )
        return False
    return True


# ============================================================
# 计时装饰器
# ============================================================
def timer(func):
    """函数计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.getLogger(__name__).info(f"{func.__name__} 耗时: {elapsed:.2f}s")
        return result
    return wrapper


# ============================================================
# 统计计算
# ============================================================
def compute_statistics(series: pd.Series,
                       metrics: Tuple[str, ...] = ("max", "min", "mean", "std",
                                                    "peak_to_peak", "rms")) -> Dict[str, float]:
    """
    计算时序信号的统计指标
    """
    vals = series.dropna().values
    if len(vals) == 0:
        return {m: np.nan for m in metrics}

    stats = {}
    for m in metrics:
        if m == "max":
            stats["max"] = float(np.max(vals))
        elif m == "min":
            stats["min"] = float(np.min(vals))
        elif m == "mean":
            stats["mean"] = float(np.mean(vals))
        elif m == "std":
            stats["std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        elif m == "peak_to_peak":
            stats["peak_to_peak"] = float(np.max(vals) - np.min(vals))
        elif m == "rms":
            stats["rms"] = float(np.sqrt(np.mean(vals ** 2)))
        elif m == "median":
            stats["median"] = float(np.median(vals))
        elif m == "abs_max":
            stats["abs_max"] = float(np.max(np.abs(vals)))
        elif m == "kurtosis":
            stats["kurtosis"] = float(pd.Series(vals).kurtosis())
        elif m == "skewness":
            stats["skewness"] = float(pd.Series(vals).skew())
    return stats


def find_extrema_indices(series: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    找到局部极大值和极小值的索引
    order: 极值点左右各需要多少个点比它小/大
    """
    from scipy.signal import argrelextrema
    vals = series.values
    max_idx = argrelextrema(vals, np.greater, order=order)[0]
    min_idx = argrelextrema(vals, np.less, order=order)[0]
    return max_idx, min_idx


def find_peak_time(series: pd.Series, time_col: pd.Series,
                   mode: str = "max") -> Tuple[float, float]:
    """
    找到峰值对应的时间和值
    mode: max(最大值), min(最小值), abs_max(绝对值最大)
    """
    vals = series.values
    t = time_col.values
    if mode == "max":
        idx = np.argmax(vals)
    elif mode == "min":
        idx = np.argmin(vals)
    elif mode == "abs_max":
        idx = np.argmax(np.abs(vals))
    else:
        idx = np.argmax(vals)
    return float(t[idx]), float(vals[idx])


# ============================================================
# 数据导出
# ============================================================
def export_to_excel(data_dict: Dict[str, pd.DataFrame],
                    filepath: str, sheet_name: str = "Sheet1"):
    """
    导出数据到Excel（多Sheet）
    data_dict: {sheet_name: DataFrame}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for name, df in data_dict.items():
            # Sheet名最长31字符
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    logger = logging.getLogger(__name__)
    logger.info(f"Excel导出: {filepath} ({len(data_dict)}个Sheet)")


def export_dict_to_csv(data: Dict[str, Any], filepath: str):
    """导出字典到CSV（统计结果用）"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame([data])
    df.to_csv(filepath, index=False, encoding="utf-8-sig")


def save_dataframe(df: pd.DataFrame, filepath: str, fmt: str = "parquet"):
    """保存DataFrame（支持parquet/csv/pickle）"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(filepath, index=False)
    elif fmt == "csv":
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
    elif fmt == "pickle":
        df.to_pickle(filepath)
    else:
        df.to_csv(filepath, index=False, encoding="utf-8-sig")


# ============================================================
# 数值工具
# ============================================================
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法"""
    if b == 0 or np.isnan(b):
        return default
    return a / b


def normalize_to_reference(values: np.ndarray, reference: float) -> np.ndarray:
    """归一化到参考值"""
    if reference == 0:
        return values
    return values / reference


def combined_load(normal: np.ndarray, tangential: np.ndarray) -> np.ndarray:
    """
    合成载荷（法向+切向的矢量和）
    适用于垂直轴风轮叶片总载荷合成
    """
    return np.sqrt(normal ** 2 + tangential ** 2)


def magnitude_3d(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """三向力合成幅值"""
    return np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)


# ============================================================
# ============================================================
# 风速列探测（B3修复：支持不同 QBlade 版本的列名差异）
# ============================================================
def find_wind_speed_col(columns) -> Optional[str]:
    """
    在列名列表中查找风速列（供工况分类/阵风检测/时序图使用）。
    优先级：
      1. Abs_Inflow_Vel._at_Hub_[m/s]（入流风速，四列文件/精简文件通常只有此列）
      2. config.global_ch.wind_vel_hub_col（Abs_Meas._Wind_Vel._at_Hub_[m/s]）
      3. 其他包含 Abs.*(Inflow|Wind).*Vel.*Hub 的列
    返回列名；找不到返回 None。
    """
    import re
    cols = list(columns)
    inflow = "Abs_Inflow_Vel._at_Hub_[m/s]"
    preferred = [inflow]
    try:
        preferred.append(config.global_ch.wind_vel_hub_col)
    except Exception:
        pass
    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        if re.search(r"Abs.*(Inflow|Wind).*Vel.*Hub", c, re.IGNORECASE):
            return c
    return None


# 数据校验
# ============================================================
def validate_load_data(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
    """校验数据是否包含必需列"""
    missing = [c for c in required_cols if c not in df.columns]
    return len(missing) == 0, missing


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """数据质量检查"""
    report = {
        "rows": len(df),
        "cols": len(df.columns),
        "nan_count": int(df.isna().sum().sum()),
        "nan_cols": df.columns[df.isna().any()].tolist(),
        "inf_count": int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum()),
        "time_range": None,
    }
    time_col = config.global_ch.time_col
    if time_col in df.columns and not df.empty:
        report["time_range"] = [float(df[time_col].iloc[0]), float(df[time_col].iloc[-1])]
        time_values = pd.to_numeric(df[time_col], errors="coerce")
        if len(time_values) > 1:
            dt = time_values.diff().dropna()
            report["time_step_median"] = float(dt[dt > 0].median()) if (dt > 0).any() else None
            report["non_positive_time_steps"] = int((dt <= 0).sum())
    else:
        report["time_step_median"] = None
        report["non_positive_time_steps"] = 0
    return report

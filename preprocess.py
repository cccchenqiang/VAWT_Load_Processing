"""
预处理模块 - 适配大文件+多叶片
分块预处理 + 逐叶片处理，低内存占用
功能：缺失值填充、异常值剔除、低通滤波、去趋势、瞬态峰值保护
"""

import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from config import config, PreprocessConfig

logger = logging.getLogger(__name__)


# ============================================================
# 单通道预处理
# ============================================================
def fill_missing(series: pd.Series, method: str = "interpolate") -> pd.Series:
    """缺失值填充"""
    if method == "ffill":
        return series.ffill().bfill()
    elif method == "bfill":
        return series.bfill().ffill()
    elif method == "interpolate":
        return series.interpolate(method="linear", limit_direction="both")
    elif method == "drop":
        return series.dropna()
    else:
        return series.interpolate(method="linear", limit_direction="both")


def remove_outliers(series: pd.Series, sigma: float = 5.0) -> pd.Series:
    """
    异常值剔除（基于标准差）
    超过sigma倍标准差的值设为NaN后插值
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series
    mask = np.abs(series - mean) > sigma * std
    if mask.any():
        series = series.copy()
        series[mask] = np.nan
        series = fill_missing(series, "interpolate")
        logger.debug(f"剔除 {mask.sum()} 个异常值 (sigma={sigma})")
    return series


def lowpass_filter(series: pd.Series, sample_rate: float,
                   cutoff_hz: float = 5.0, order: int = 4) -> pd.Series:
    """
    巴特沃斯低通滤波
    sample_rate: 采样率 [Hz]
    """
    if sample_rate <= 0 or cutoff_hz <= 0:
        return series
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff_hz / nyq
    if normal_cutoff >= 1.0:
        return series
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    # filtfilt零相位滤波
    filtered = signal.filtfilt(b, a, series.values)
    return pd.Series(filtered, index=series.index, name=series.name)


def detrend_series(series: pd.Series) -> pd.Series:
    """去趋势（线性）"""
    return pd.Series(signal.detrend(series.values), index=series.index, name=series.name)


def preserve_transient_peaks(original: pd.Series, filtered: pd.Series,
                             sigma: float = 3.0) -> pd.Series:
    """
    瞬态峰值保护：极端载荷峰值（如EOG阵风）不被滤波抹平
    在原始信号峰值处，用原始值替换滤波值
    """
    mean = original.mean()
    std = original.std()
    if std == 0:
        return filtered
    peak_mask = np.abs(original - mean) > sigma * std
    result = filtered.copy()
    result[peak_mask] = original[peak_mask]
    return result


# ============================================================
# 单DataFrame预处理（一块数据）
# ============================================================
def preprocess_dataframe(df: pd.DataFrame,
                         cfg: Optional[PreprocessConfig] = None,
                         sample_rate: float = 0.0,
                         exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    对一个DataFrame块进行预处理
    只处理数值列，跳过时间列和非载荷列
    """
    cfg = cfg or config.preprocess
    exclude_cols = exclude_cols or [config.global_ch.time_col, config.global_ch.timestep_col]

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    process_cols = [c for c in numeric_cols if c not in exclude_cols]

    for col in process_cols:
        s = df[col]
        # 1. 缺失值
        s = fill_missing(s, cfg.missing_method)
        # 2. 异常值
        s = remove_outliers(s, cfg.outlier_sigma)
        # 3. 低通滤波
        if cfg.enable_filter and sample_rate > 0:
            original = s.copy()
            s = lowpass_filter(s, sample_rate, cfg.filter_cutoff_hz, cfg.filter_order)
            # 瞬态峰值保护
            if cfg.preserve_transient_peaks:
                s = preserve_transient_peaks(original, s, cfg.transient_threshold_sigma)
        # 4. 去趋势
        if cfg.detrend:
            s = detrend_series(s)
        df[col] = s.values

    return df


# ============================================================
# 分块预处理器（大文件用）
# ============================================================
class ChunkPreprocessor:
    """
    大文件分块预处理器
    逐块读取 -> 预处理 -> 回调处理（不累积全量数据）
    支持重叠块修正（滤波边界效应）
    """

    def __init__(self, loader, cfg: Optional[PreprocessConfig] = None):
        """
        loader: BigFileLoader实例
        """
        self.loader = loader
        self.cfg = cfg or config.preprocess
        self.sample_rate = 0.0
        self._init_sample_rate()

    def _init_sample_rate(self):
        """初始化采样率"""
        try:
            t = self.loader.get_time_array()
            if len(t) > 1:
                dt = np.median(np.diff(t))
                self.sample_rate = 1.0 / dt if dt > 0 else 0.0
        except Exception as e:
            logger.warning(f"采样率初始化失败: {e}")
            self.sample_rate = 120.0  # 默认QBlade典型采样率

    def process_stream(self, callback, include_panel: bool = True,
                       usecols: Optional[List[str]] = None):
        """
        流式分块预处理
        callback(chunk_df, chunk_index): 处理每块预处理后的数据
        """
        for i, chunk in enumerate(self.loader.iter_chunks(include_panel=include_panel,
                                                          usecols=usecols)):
            processed = preprocess_dataframe(
                chunk, self.cfg, self.sample_rate
            )
            callback(processed, i)
            logger.info(f"预处理块 {i}: {processed.shape[0]}行")

    def process_and_collect(self, include_panel: bool = True,
                            usecols: Optional[List[str]] = None,
                            max_rows: int = 0) -> pd.DataFrame:
        """
        分块预处理并收集结果（适用于中等大小文件）
        max_rows: 最大收集行数，0=不限制
        """
        results = []
        total = 0
        for i, chunk in enumerate(self.loader.iter_chunks(include_panel=include_panel,
                                                          usecols=usecols)):
            processed = preprocess_dataframe(chunk, self.cfg, self.sample_rate)
            results.append(processed)
            total += len(processed)
            if max_rows > 0 and total >= max_rows:
                break
        if not results:
            return pd.DataFrame()
        df = pd.concat(results, ignore_index=True)
        logger.info(f"分块预处理完成: {df.shape[0]}行 x {df.shape[1]}列")
        return df


# ============================================================
# 多叶片并行预处理
# ============================================================
def preprocess_blade_data(blade_data: Dict[str, pd.DataFrame],
                          cfg: Optional[PreprocessConfig] = None,
                          sample_rate: float = 0.0) -> Dict[str, pd.DataFrame]:
    """
    逐叶片独立预处理（避免整机平均化掩盖单叶片极端载荷）
    blade_data: {blade_id: DataFrame}
    """
    cfg = cfg or config.preprocess
    result = {}
    for blade_id, df in blade_data.items():
        logger.info(f"预处理 {blade_id}: {df.shape[0]}行")
        result[blade_id] = preprocess_dataframe(df, cfg, sample_rate)
    return result


def resample_align(blade_data: Dict[str, pd.DataFrame],
                   target_rate: float = 0.0) -> Dict[str, pd.DataFrame]:
    """
    多叶片时序对齐重采样
    确保各叶片时间轴一致
    """
    if not blade_data:
        return blade_data
    # 以第一个叶片的时间轴为基准
    ref_id = list(blade_data.keys())[0]
    ref_df = blade_data[ref_id]
    time_col = "time" if "time" in ref_df.columns else config.global_ch.time_col

    if target_rate <= 0:
        return blade_data

    result = {}
    t_target = np.arange(ref_df[time_col].iloc[0],
                         ref_df[time_col].iloc[-1],
                         1.0 / target_rate)
    for blade_id, df in blade_data.items():
        resampled = pd.DataFrame({time_col: t_target})
        for col in df.columns:
            if col == time_col:
                continue
            resampled[col] = np.interp(t_target, df[time_col].values, df[col].values)
        result[blade_id] = resampled
    return result

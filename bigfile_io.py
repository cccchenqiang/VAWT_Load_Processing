"""
大文件读写模块 - QBlade仿真输出适配
核心功能：
1. QBlade ASCII文件解析（2行注释头 + 表头 + 科学计数法数据）
2. 分块惰性读取（GB级文件不OOM）
3. 多叶片通道自动解析与映射
4. 面板级分布载荷提取
5. 工况标识自动提取
6. 按需列读取（1526列中只取需要的）
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Iterator, Any

import numpy as np
import pandas as pd

from config import config, BigFileConfig, BladeConfig

logger = logging.getLogger(__name__)


# ============================================================
# QBlade文件元信息解析
# ============================================================
class QBladeFileInfo:
    """QBlade仿真输出文件的元信息"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_size_mb: float = 0.0
        self.header_line: str = ""
        self.columns: List[str] = []
        self.num_columns: int = 0
        self.num_data_rows: int = 0
        self.condition_name: str = ""       # 工况名称，如 "zhognneng20 Turb"
        self.condition_type: str = ""       # 工况类型，如 "EOG"
        self.creation_time: str = ""
        self.comment_lines: List[str] = []
        self.data_start_row: int = 0        # 数据起始行（0-based）
        self.header_row: int = 0            # 表头行索引
        self.num_blades: int = 0
        self.num_panels: int = 0
        self.time_step: float = 0.0
        self.total_time: float = 0.0
        self.sample_rate_hz: float = 0.0

    def __repr__(self):
        return (f"QBladeFileInfo({os.path.basename(self.filepath)}, "
                f"{self.num_columns}cols, {self.num_data_rows}rows, "
                f"{self.num_blades}blades, {self.condition_type})")


def normalize_qblade_column(name: str) -> str:
    """QBlade列名归一化，统一到系统内部命名：

    1) 波浪号~（QBlade导出的空格占位符）→ 下划线_（不同导出设置下同一通道可能是
       Time~[s] 或 Time_[s]，归一化后与 config.global_ch 等下划线版列名对齐）
    2) 新导出格式叶片编号 BLD_n（可带结构后缀 +STR_X_n）→ 旧格式 Blade_n：
       Total_Tangential_Load_BLD_1+STR_X_1_[N] → Total_Tangential_Load_Blade_1_[N]
       以便与 config.blade 的 Total_*_Load_Blade_{n}_[N] pattern 匹配。
    """
    name = name.replace("~", "_")
    # 先处理带结构后缀的 BLD_n+STR_X_n
    name = re.sub(r"BLD_(\d+)\+STR_X_\d+", r"Blade_\1", name)
    # 再处理纯 BLD_n
    name = re.sub(r"BLD_(\d+)", r"Blade_\1", name)
    return name


def parse_qblade_header(filepath: str, cfg: Optional[BigFileConfig] = None) -> QBladeFileInfo:
    """
    解析QBlade文件头部信息，不加载数据
    只读取前几行 + 统计行数，内存友好
    """
    cfg = cfg or config.bigfile
    info = QBladeFileInfo(filepath)
    info.file_size_mb = os.path.getsize(filepath) / 1024 / 1024

    with open(filepath, "r", encoding=cfg.encoding, errors="replace") as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line.rstrip("\n").rstrip("\r"))
            if i >= 10:  # 前10行足够解析头部
                break

    # 解析注释行和工况行
    data_row_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            data_row_idx = i + 1
            continue
        # QBlade文件创建信息行
        if stripped.startswith("Results Output File"):
            info.comment_lines.append(stripped)
            # 提取创建时间
            m = re.search(r'on\s+(.+)$', stripped)
            if m:
                info.creation_time = m.group(1).strip()
            data_row_idx = i + 1
            continue
        # 工况标识行，如 "EOG: zhognneng20 Turb" / "NTM: xxx" / "zhognneng-e-store: xxx"
        # 兼容小写、连字符等自定义工况前缀（如 "zhognneng-e-store:"）
        if re.match(r'^[A-Za-z][A-Za-z0-9_-]*\s*:', stripped):
            parts = stripped.split(":", 1)
            info.condition_type = parts[0].strip()
            info.condition_name = parts[1].strip() if len(parts) > 1 else ""
            info.comment_lines.append(stripped)
            data_row_idx = i + 1
            continue
        # 第一个非注释行就是表头
        if "\t" in stripped or "," in stripped:
            info.header_row = i
            info.header_line = stripped
            sep = "\t" if "\t" in stripped else ","
            info.columns = [normalize_qblade_column(c.strip()) for c in stripped.split(sep)]
            info.num_columns = len(info.columns)
            data_row_idx = i + 1
            break

    info.data_start_row = data_row_idx

    # 统计数据行数（快速计数，不加载）
    with open(filepath, "r", encoding=cfg.encoding, errors="replace") as f:
        total_lines = sum(1 for _ in f)
    info.num_data_rows = total_lines - info.data_start_row

    # 自动检测叶片数量和面板数
    blade_nums = set()
    panel_nums = set()
    for col in info.columns:
        m_b = re.search(r'Blade_(\d+)', col)
        if m_b:
            blade_nums.add(int(m_b.group(1)))
        m_p = re.search(r'PAN_(\d+)', col)
        if m_p:
            panel_nums.add(int(m_p.group(1)))

    info.num_blades = max(blade_nums) if blade_nums else 0
    info.num_panels = (max(panel_nums) + 1) if panel_nums else 0

    # 更新配置中的叶片数
    if info.num_blades > 0:
        config.blade.num_blades = info.num_blades
        config.blade.blade_ids = [f"Blade_{n}" for n in range(1, info.num_blades + 1)]
    if info.num_panels > 0:
        config.blade.num_panels = info.num_panels

    logger.info(f"文件解析完成: {info}")
    return info


# ============================================================
# 通道映射构建
# ============================================================
class ChannelMapper:
    """
    多叶片通道映射器
    自动从1526列中归类出：全局通道、各叶片总载荷通道、各叶片面板级通道
    """

    def __init__(self, columns: List[str], blade_cfg: Optional[BladeConfig] = None):
        self.columns = columns
        self.blade_cfg = blade_cfg or config.blade
        self.num_blades = self.blade_cfg.num_blades or 0
        self.num_panels = self.blade_cfg.num_panels

        # 全局通道（不含Blade_和PAN_的列）
        self.global_channels: List[str] = []
        # 叶片总载荷: {blade_id: {channel_type: col_name}}
        self.blade_total_loads: Dict[str, Dict[str, str]] = {}
        # 叶片面板级载荷: {blade_id: {channel_type: [col_pan0, col_pan1, ...]}}
        self.blade_panel_loads: Dict[str, Dict[str, List[str]]] = {}
        # 叶片气动参数
        self.blade_aero: Dict[str, Dict[str, List[str]]] = {}

        self._build_mapping()

    def _build_mapping(self):
        """构建完整的通道映射"""
        # 1. 全局通道
        for col in self.columns:
            if "Blade_" not in col and "PAN_" not in col:
                self.global_channels.append(col)

        # 2. 逐叶片映射
        for n in range(1, self.num_blades + 1):
            blade_id = f"Blade_{n}"
            self.blade_total_loads[blade_id] = {}
            self.blade_panel_loads[blade_id] = {}
            self.blade_aero[blade_id] = {}

            # 总载荷
            self._map_single_col(blade_id, "total_tangential",
                                 self.blade_cfg.total_tangential_pattern.format(n=n))
            self._map_single_col(blade_id, "total_normal",
                                 self.blade_cfg.total_normal_pattern.format(n=n))
            self._map_single_col(blade_id, "azimuth",
                                 self.blade_cfg.azimuth_pattern.format(n=n))
            self._map_single_col(blade_id, "pitch",
                                 self.blade_cfg.pitch_pattern.format(n=n))

            # 面板级载荷
            self._map_panel_cols(blade_id, "normal_force",
                                 self.blade_cfg.panel_normal_force_pattern, n)
            self._map_panel_cols(blade_id, "tangential_force",
                                 self.blade_cfg.panel_tangential_force_pattern, n)
            self._map_panel_cols(blade_id, "pitching_moment",
                                 self.blade_cfg.panel_pitching_moment_pattern, n)
            self._map_panel_cols(blade_id, "height",
                                 self.blade_cfg.panel_height_pattern, n)

            # 面板级气动参数（可选）
            self._map_panel_cols(blade_id, "lift_coeff",
                                 self.blade_cfg.panel_lift_coeff_pattern, n)
            self._map_panel_cols(blade_id, "drag_coeff",
                                 self.blade_cfg.panel_drag_coeff_pattern, n)
            self._map_panel_cols(blade_id, "aoa",
                                 self.blade_cfg.panel_aoa_pattern, n)
            self._map_panel_cols(blade_id, "reynolds",
                                 self.blade_cfg.panel_re_pattern, n)

    def _map_single_col(self, blade_id: str, key: str, pattern: str):
        """映射单列（总载荷类）"""
        if pattern in self.columns:
            self.blade_total_loads[blade_id][key] = pattern

    def _map_panel_cols(self, blade_id: str, key: str, pattern: str, n: int):
        """映射面板级多列"""
        cols = []
        for p in range(self.num_panels):
            col = pattern.format(n=n, p=p)
            if col in self.columns:
                cols.append(col)
        if cols:
            self.blade_panel_loads[blade_id][key] = cols

    def get_load_columns(self, include_panel: bool = True) -> List[str]:
        """
        获取所有需要加载的载荷列名（用于usecols减少内存）
        include_panel: 是否包含面板级分布载荷（1526列中大部分是面板级）
        """
        cols = set(self.global_channels)
        for blade_id in self.blade_total_loads:
            cols.update(self.blade_total_loads[blade_id].values())
            if include_panel:
                for panel_cols in self.blade_panel_loads[blade_id].values():
                    cols.update(panel_cols)
        # 保持原始列顺序
        return [c for c in self.columns if c in cols]

    def get_blade_total_load_df(self, df: pd.DataFrame, blade_id: str) -> pd.DataFrame:
        """从全量DataFrame中提取单叶片总载荷数据"""
        cols = self.blade_total_loads.get(blade_id, {})
        if not cols:
            return pd.DataFrame()
        time_col = config.global_ch.time_col
        if time_col not in df.columns:
            return pd.DataFrame()
        # 只保留实际存在的列
        existing = {k: v for k, v in cols.items() if v in df.columns}
        if not existing:
            return pd.DataFrame()
        result = df[[time_col] + list(existing.values())].copy()
        result.columns = ["time"] + list(existing.keys())
        return result

    def get_blade_panel_load_df(self, df: pd.DataFrame, blade_id: str,
                                load_type: str = "normal_force") -> pd.DataFrame:
        """提取单叶片某类型面板级分布载荷（行=时间，列=面板）"""
        cols = self.blade_panel_loads.get(blade_id, {}).get(load_type, [])
        if not cols:
            return pd.DataFrame()
        # 只保留实际存在于df中的列（--no-panel模式下面板列未加载）
        existing_cols = [c for c in cols if c in df.columns]
        if not existing_cols:
            return pd.DataFrame()
        time_col = config.global_ch.time_col
        if time_col not in df.columns:
            return pd.DataFrame()
        result = df[[time_col] + existing_cols].copy()
        result.columns = ["time"] + [f"PAN_{i}" for i in range(len(existing_cols))]
        return result

    def summary(self) -> str:
        """映射摘要"""
        total_panel_cols = sum(
            len(v) for b in self.blade_panel_loads.values() for v in b.values()
        )
        return (f"通道映射: {len(self.global_channels)}全局列, "
                f"{self.num_blades}叶片, "
                f"{sum(len(v) for v in self.blade_total_loads.values())}总载荷列, "
                f"{total_panel_cols}面板级列")


# ============================================================
# 大文件分块读取器
# ============================================================
class BigFileLoader:
    """
    QBlade大文件分块读取器
    - 惰性迭代，不一次性加载全量数据
    - 支持按需列选择（usecols）
    - 自动跳过注释行
    - 科学计数法自动解析
    """

    def __init__(self, filepath: str, cfg: Optional[BigFileConfig] = None):
        self.filepath = filepath
        self.cfg = cfg or config.bigfile
        self.info = parse_qblade_header(filepath, self.cfg)
        self.mapper = ChannelMapper(self.info.columns)
        self._all_columns = self.info.columns

    @property
    def num_rows(self) -> int:
        return self.info.num_data_rows

    @property
    def num_cols(self) -> int:
        return self.info.num_columns

    def _get_usecols(self, columns: Optional[List[str]] = None) -> Optional[List[str]]:
        """确定要加载的列"""
        if columns is None:
            return None  # 全部加载
        return [c for c in columns if c in self._all_columns]

    def iter_chunks(self,
                    chunksize: Optional[int] = None,
                    usecols: Optional[List[str]] = None,
                    include_panel: bool = True) -> Iterator[pd.DataFrame]:
        """
        分块迭代读取数据
        chunksize: 每块行数，默认用配置值
        usecols: 指定列名列表，None=自动选择载荷列
        include_panel: 是否包含面板级列（仅usecols为None时生效）
        """
        cs = chunksize or self.cfg.chunksize

        # 确定列
        if usecols is None:
            usecols = self.mapper.get_load_columns(include_panel=include_panel)

        # pandas分块读取
        reader = pd.read_csv(
            self.filepath,
            sep=self.cfg.sep,
            skiprows=self.info.data_start_row - 0,  # data_start_row已经是数据行
            header=None,
            names=self._all_columns,
            usecols=usecols if usecols else None,
            chunksize=cs,
            encoding=self.cfg.encoding,
            engine="c",
            dtype=np.float64,
            na_values=["", " ", "NaN", "nan", "INF", "inf"],
            low_memory=False,
        )

        for chunk in reader:
            # 重置索引
            chunk = chunk.reset_index(drop=True)
            yield chunk

    def load_all(self, usecols: Optional[List[str]] = None,
                 include_panel: bool = True) -> pd.DataFrame:
        """
        全量加载（仅适用于中小文件）
        大文件请使用 iter_chunks
        """
        if usecols is None:
            usecols = self.mapper.get_load_columns(include_panel=include_panel)
        usecols = self._get_usecols(usecols)
        if not usecols:
            return pd.DataFrame()

        # Read directly instead of materializing all chunks and then
        # concatenating them, which temporarily keeps two full copies.
        df = pd.read_csv(
            self.filepath,
            sep=self.cfg.sep,
            skiprows=self.info.data_start_row,
            header=None,
            names=self._all_columns,
            usecols=usecols,
            encoding=self.cfg.encoding,
            engine="c",
            dtype=np.float64,
            na_values=["", " ", "NaN", "nan", "INF", "inf"],
            low_memory=False,
        )
        if df.empty:
            return pd.DataFrame(columns=usecols)
        df = df.reset_index(drop=True)
        logger.info(f"全量加载完成: {df.shape[0]}行 x {df.shape[1]}列")
        return df

    def load_time_range(self, t_start: float, t_end: float,
                        usecols: Optional[List[str]] = None,
                        include_panel: bool = True) -> pd.DataFrame:
        """按时间范围加载数据（EOG阵风段提取用）"""
        time_col = config.global_ch.time_col
        if usecols is None:
            usecols = self.mapper.get_load_columns(include_panel=include_panel)
        if time_col not in usecols:
            usecols = [time_col] + usecols

        result_chunks = []
        for chunk in self.iter_chunks(usecols=usecols):
            mask = (chunk[time_col] >= t_start) & (chunk[time_col] <= t_end)
            if mask.any():
                result_chunks.append(chunk[mask].copy())
            # 提前终止：已经超过结束时间
            if chunk[time_col].iloc[-1] > t_end:
                break

        if not result_chunks:
            return pd.DataFrame()
        return pd.concat(result_chunks, ignore_index=True)

    def get_time_array(self) -> np.ndarray:
        """只读取时间列（快速获取时间轴）"""
        time_col = config.global_ch.time_col
        df = pd.read_csv(
            self.filepath,
            sep=self.cfg.sep,
            skiprows=self.info.data_start_row,
            header=None,
            names=self._all_columns,
            usecols=[time_col],
            encoding=self.cfg.encoding,
            dtype=np.float64,
        )
        t = df[time_col].values
        if len(t) > 1:
            self.info.time_step = float(np.median(np.diff(t)))
            self.info.total_time = float(t[-1] - t[0])
            self.info.sample_rate_hz = 1.0 / self.info.time_step if self.info.time_step > 0 else 0
        return t


# ============================================================
# 工况智能切片器（通用化，支持EOG/NTM/EWS等所有风况）
# ============================================================
class ConditionSlicer:
    """
    从长时序中智能切片，支持所有风况类型：
    - 瞬态工况（EOG/EWS/ECD等）：阵风瞬态段检测（风速变化率法）
    - 稳态工况（NTM/NWP等）：全量极值 或 滑动窗口极值
    - 自动模式：根据工况标识自动选择策略

    策略选择逻辑：
    - extreme_strategy="auto" → 根据condition_type自动判断
    - extreme_strategy="eog_surge" → 强制使用阵风切片（适合瞬态工况）
    - extreme_strategy="full" → 强制使用全量极值（适合任何工况）
    - extreme_strategy="sliding_window" → 滑动窗口极值（适合长时序稳态）
    """

    def __init__(self, loader: BigFileLoader):
        self.loader = loader
        self.info = loader.info

    def classify_condition(self) -> str:
        """
        分类工况类型
        返回: "transient"(瞬态) / "steady"(稳态) / "unknown"(未知)
        """
        cond_type = (self.info.condition_type or "").upper()
        transient_types = config.extreme.transient_condition_types
        steady_types = config.extreme.steady_condition_types

        if cond_type in transient_types:
            return "transient"
        if cond_type in steady_types:
            return "steady"
        # 未识别的工况：根据风速变化率特征自动判断
        return self._auto_classify_by_wind()

    def _auto_classify_by_wind(self) -> str:
        """根据风速变化率特征自动判断工况类型"""
        from utils import find_wind_speed_col
        wind_col = find_wind_speed_col(self.info.columns) or config.global_ch.wind_vel_hub_col
        time_col = config.global_ch.time_col
        try:
            df = self.loader.load_all(usecols=[time_col, wind_col], include_panel=False)
            if df.empty or len(df) < 10:
                return "unknown"
            v = df[wind_col].values
            t = df[time_col].values
            dv_dt = np.abs(np.gradient(v, t))
            # 如果风速变化率的最大值远大于均值，说明有明显瞬态（阵风）
            if np.mean(dv_dt) > 0 and np.max(dv_dt) / np.mean(dv_dt) > 5:
                return "transient"
            return "steady"
        except Exception:
            return "unknown"

    def detect_extreme_segments(self,
                                 strategy: Optional[str] = None,
                                 min_duration_sec: float = 1.0
                                 ) -> List[Tuple[float, float]]:
        """
        通用极值段检测（根据工况类型自动选择策略）

        参数:
            strategy: 检测策略，None=使用配置中的extreme_strategy
            min_duration_sec: 最小时段长度
        返回: [(t_start, t_end), ...]，空列表表示使用全量数据
        """
        strategy = strategy or config.extreme.extreme_strategy
        cond_class = self.classify_condition()
        cond_type = self.info.condition_type or "UNKNOWN"

        logger.info(f"极值检测: 工况={cond_type}({cond_class}), 策略={strategy}")

        if strategy == "auto":
            if cond_class == "transient":
                strategy = "eog_surge"
            elif cond_class == "steady":
                strategy = "full"  # 稳态工况默认全量极值
            else:
                strategy = "full"  # 未知工况默认全量极值

        if strategy == "eog_surge":
            segments = self.detect_eog_segments(min_duration_sec=min_duration_sec)
            if not segments:
                logger.info("未检测到阵风段，回退到全量极值")
            return segments
        elif strategy == "sliding_window":
            return self.detect_sliding_window_segments()
        elif strategy == "full":
            return []  # 空列表表示全量数据
        else:
            logger.warning(f"未知策略 '{strategy}'，使用全量极值")
            return []

    def detect_eog_segments(self,
                            surge_rate_threshold: Optional[float] = None,
                            min_duration_sec: float = 1.0) -> List[Tuple[float, float]]:
        """
        检测瞬态阵风段（适用于EOG/EWS/ECD等瞬态工况）
        基于风速变化率阈值，返回 [(t_start, t_end), ...]
        """
        from config import ExtremeLoadConfig
        from utils import find_wind_speed_col
        ext_cfg = config.extreme
        threshold = surge_rate_threshold or ext_cfg.wind_surge_rate_threshold

        # 读取风速和时间（兼容 Inflow/Meas 列名差异）
        wind_col = find_wind_speed_col(self.info.columns) or config.global_ch.wind_vel_hub_col
        time_col = config.global_ch.time_col
        df = self.loader.load_all(usecols=[time_col, wind_col], include_panel=False)
        if df.empty:
            return []

        t = df[time_col].values
        v = df[wind_col].values

        # 计算风速变化率
        dv_dt = np.gradient(v, t)
        # 阵风触发：变化率超过阈值
        surge_mask = np.abs(dv_dt) > threshold

        # 合并连续触发段
        segments = []
        in_surge = False
        seg_start = 0
        for i in range(len(surge_mask)):
            if surge_mask[i] and not in_surge:
                in_surge = True
                seg_start = i
            elif not surge_mask[i] and in_surge:
                in_surge = False
                duration = t[i - 1] - t[seg_start]
                if duration >= min_duration_sec:
                    # 前后扩展窗口
                    ext = ext_cfg.extreme_window_sec
                    s = max(0, t[seg_start] - ext)
                    e = min(t[-1], t[i - 1] + ext)
                    segments.append((float(s), float(e)))
        if in_surge:
            duration = t[-1] - t[seg_start]
            if duration >= min_duration_sec:
                ext = ext_cfg.extreme_window_sec
                segments.append((float(max(0, t[seg_start] - ext)), float(t[-1])))

        logger.info(f"检测到 {len(segments)} 个瞬态阵风段")
        return segments

    def detect_sliding_window_segments(self,
                                        window_sec: Optional[float] = None,
                                        step_sec: Optional[float] = None
                                        ) -> List[Tuple[float, float]]:
        """
        滑动窗口极值段检测（适用于NTM等长时序稳态工况）
        将整个时序划分为固定窗口，每个窗口独立统计极值
        返回 [(t_start, t_end), ...]
        """
        ext_cfg = config.extreme
        window_sec = window_sec or ext_cfg.sliding_window_sec
        step_sec = step_sec or ext_cfg.sliding_step_sec

        time_col = config.global_ch.time_col
        t = self.loader.get_time_array()
        if len(t) < 2:
            return []

        t_total = t[-1] - t[0]
        segments = []
        t_start = t[0]
        while t_start < t[-1]:
            t_end = min(t_start + window_sec, t[-1])
            segments.append((float(t_start), float(t_end)))
            t_start += step_sec
            if t_end >= t[-1]:
                break

        logger.info(f"滑动窗口: {len(segments)} 个窗口 (窗口={window_sec}s, 步长={step_sec}s)")
        return segments

    def get_steady_state_mask(self, exclude_segments: List[Tuple[float, float]],
                              t: np.ndarray) -> np.ndarray:
        """获取稳态段掩码（排除瞬态阵风段）"""
        mask = np.ones(len(t), dtype=bool)
        for s, e in exclude_segments:
            mask &= ~((t >= s) & (t <= e))
        return mask


# ============================================================
# 便捷函数
# ============================================================
def load_qblade_file(filepath: str, include_panel: bool = True) -> Tuple[pd.DataFrame, QBladeFileInfo, ChannelMapper]:
    """
    一键加载QBlade文件（中小文件用，大文件用BigFileLoader.iter_chunks）
    返回: (DataFrame, 文件信息, 通道映射器)
    """
    loader = BigFileLoader(filepath)
    df = loader.load_all(include_panel=include_panel)
    return df, loader.info, loader.mapper


def quick_inspect(filepath: str) -> str:
    """快速检查文件信息（不加载数据）"""
    info = parse_qblade_header(filepath)
    mapper = ChannelMapper(info.columns)
    lines = [
        f"文件: {os.path.basename(filepath)}",
        f"大小: {info.file_size_mb:.2f} MB",
        f"工况: {info.condition_type} - {info.condition_name}",
        f"创建时间: {info.creation_time}",
        f"数据: {info.num_data_rows} 行 x {info.num_columns} 列",
        f"叶片数: {info.num_blades}, 面板数: {info.num_panels}",
        mapper.summary(),
    ]
    return "\n".join(lines)

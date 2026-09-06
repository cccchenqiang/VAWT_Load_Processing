"""
fast_io.py — OpenFAST / AeroDyn .out 时程文件读取模块
=====================================================
独立于 QBlade（bigfile_io）的 FAST 结果解析，不影响原 QBlade 读取流程。

功能：
1. parse_fast_header()         解析 .out 头部（列名/单位/行数/采样率/生成信息）
2. load_fast_data()            加载 .out 数据为 DataFrame（固定宽度/空白分隔自适应）
3. AERODYN_VARS               内置 AeroDyn 子表变量定义（rotor 载荷 + 叶片节点载荷）
4. load_aerodyn_vars_from_xlsx 从 OutListParameters.xlsx 的 AeroDyn 子表加载变量定义
5. FASTChannelMapper           FAST 列名 → 系统通道映射（容错：缺列不报错）

容错原则：即使输出文件缺少部分变量，也能继续分析已存在的变量。
"""

from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================
# 1. AeroDyn 变量定义（内置，预处理自 OutListParameters.xlsx 的 AeroDyn 子表）
# ============================================================

# ---- 转子(Rotor)载荷变量：对应系统"风轮载荷"分析 ----
# key: 变量名; value: (单位, 描述)
AERODYN_ROTOR_VARS: Dict[str, Tuple[str, str]] = {
    "RtSpeed":    ("rpm",  "Rotor speed"),
    "RtTSR":      ("-",    "Rotor tip-speed ratio"),
    "RtAeroFxh":  ("N",    "Total rotor aerodynamic load (force in x direction, hub coord)"),
    "RtAeroFyh":  ("N",    "Total rotor aerodynamic load (force in y direction, hub coord)"),
    "RtAeroFzh":  ("N",    "Total rotor aerodynamic load (force in z direction, hub coord)"),
    "RtAeroMxh":  ("N-m",  "Total rotor aerodynamic load (moment in x direction, hub coord)"),
    "RtAeroMyh":  ("N-m",  "Total rotor aerodynamic load (moment in y direction, hub coord)"),
    "RtAeroMzh":  ("N-m",  "Total rotor aerodynamic load (moment in z direction, hub coord)"),
    "RtAeroPwr":  ("W",    "Rotor aerodynamic power"),
    "RtArea":     ("m^2",  "Rotor swept area"),
    "RtAeroCp":   ("-",    "Rotor aerodynamic power coefficient"),
    "RtAeroCq":   ("-",    "Rotor aerodynamic torque coefficient"),
    "RtAeroCt":   ("-",    "Rotor aerodynamic thrust coefficient"),
    "RtAeroFxg":  ("N",    "Total rotor aerodynamic load (force in x direction, global coord)"),
    "RtAeroFyg":  ("N",    "Total rotor aerodynamic load (force in y direction, global coord)"),
    "RtAeroFzg":  ("N",    "Total rotor aerodynamic load (force in z direction, global coord)"),
    "RtAeroMxg":  ("N-m",  "Total rotor aerodynamic load (moment in x direction, global coord)"),
    "RtAeroMyg":  ("N-m",  "Total rotor aerodynamic load (moment in y direction, global coord)"),
    "RtAeroMzg":  ("N-m",  "Total rotor aerodynamic load (moment in z direction, global coord)"),
    "RtVAvgxh":   ("m/s",  "Rotor-disk-averaged relative wind velocity (x-component)"),
    "RtVAvgyh":   ("m/s",  "Rotor-disk-averaged relative wind velocity (y-component)"),
    "RtVAvgzh":   ("m/s",  "Rotor-disk-averaged relative wind velocity (z-component)"),
    "RtSkew":     ("deg",  "Rotor inflow-skew angle"),
}

# ---- 叶片节点载荷变量：用于叶片载荷积分 ----
# 模板: 前缀 + 节点号。如 B1N1Fx = 叶片1节点1法向(平面)分布力 N/m
# 命名规则: B{b}N{n}{后缀}，b=叶片号，n=节点号
AERODYN_BLADE_SUFFIXES: Dict[str, Tuple[str, str]] = {
    "Fx":  ("N/m",    "Normal force (to plane) per unit length"),
    "Fy":  ("N/m",    "Tangential force (to plane) per unit length"),
    "Fn":  ("N/m",    "Normal force (to chord) per unit length"),
    "Ft":  ("N/m",    "Tangential force (to chord) per unit length"),
    "Fl":  ("N/m",    "Lift force per unit length"),
    "Fd":  ("N/m",    "Drag force per unit length"),
    "Mm":  ("N-m/m",  "Pitching moment per unit length"),
    "VRel":("m/s",    "Relative wind speed"),
    "Alpha":("deg",   "Angle of attack"),
    "Cl":  ("-",      "Lift force coefficient"),
    "Cd":  ("-",      "Drag force coefficient"),
    "Cm":  ("-",      "Pitching moment coefficient"),
    "Vindx":("m/s",   "Axial induced wind velocity"),
    "Vindy":("m/s",   "Tangential induced wind velocity"),
}

# ---- 入流/环境/运动变量（AeroDyn_driver 常用）----
AERODYN_ENV_VARS: Dict[str, Tuple[str, str]] = {
    "HWindSpeedX": ("m/s", "Horizontal wind speed X (driver input)"),
    "HWindSpeedY": ("m/s", "Horizontal wind speed Y (driver input)"),
    "HWindSpeedZ": ("m/s", "Horizontal wind speed Z (driver input)"),
    "ShearExp":    ("-",   "Power law wind shear exponent"),
    "Azimuth":     ("deg", "Azimuth angle"),
    "Yaw":         ("deg", "Yaw angle"),
    "RotSpeed":    ("rpm", "Rotor speed (driver input)"),
    "BldPitch1":   ("deg", "Blade 1 pitch angle"),
    "BldPitch2":   ("deg", "Blade 2 pitch angle"),
    "BldPitch3":   ("deg", "Blade 3 pitch angle"),
    "PtfmSurge":   ("m",   "Platform surge DOF"),
    "PtfmSway":    ("m",   "Platform sway DOF"),
    "PtfmHeave":   ("m",   "Platform heave DOF"),
    "PtfmRoll":    ("deg", "Platform roll DOF"),
    "PtfmPitch":   ("deg", "Platform pitch DOF"),
    "PtfmYaw":     ("deg", "Platform yaw DOF"),
}


def build_aerodyn_var_dict(num_blades: int = 3, num_nodes: int = 18) -> Dict[str, Tuple[str, str]]:
    """生成完整 AeroDyn 变量定义字典（rotor + 环境 + 各叶片各节点载荷）

    num_blades: 叶片数（默认3）
    num_nodes: 每叶片节点数（默认18，AeroDyn_driver 实测）
    返回: {变量名: (单位, 描述)}
    """
    d: Dict[str, Tuple[str, str]] = {}
    d.update(AERODYN_ROTOR_VARS)
    d.update(AERODYN_ENV_VARS)
    for b in range(1, num_blades + 1):
        for n in range(1, num_nodes + 1):
            for suffix, (unit, desc) in AERODYN_BLADE_SUFFIXES.items():
                name = f"B{b}N{n:03d}{suffix}" if n >= 100 else f"B{b}N{n}{suffix}"
                # ad_driver 用 3 位节点号 AB1N001Vindx；标准 FAST 用 B1N1Vindx
                if n >= 10:
                    name = f"B{b}N{n}{suffix}"
                d[name] = (unit, f"{desc} at Blade {b}, Node {n}")
    # 兼容 AeroDyn_driver 的 AB 前缀命名（AB1N001Vindx）
    for b in range(1, num_blades + 1):
        for n in range(1, num_nodes + 1):
            for suffix, (unit, desc) in AERODYN_BLADE_SUFFIXES.items():
                d[f"AB{b}N{n:03d}{suffix}"] = (unit, f"{desc} at Blade {b}, Node {n} (AeroDyn_driver)")
    return d


# 默认内置定义（3叶片 × 18节点）
AERODYN_VARS: Dict[str, Tuple[str, str]] = build_aerodyn_var_dict()


def load_aerodyn_vars_from_xlsx(xlsx_path: str) -> Dict[str, Tuple[str, str]]:
    """从 OutListParameters.xlsx 的 AeroDyn 子表加载变量定义（可选，增强内置字典）

    返回: {变量名: (单位, 描述)}
    """
    try:
        import pandas as pd
    except ImportError:
        return {}
    out: Dict[str, Tuple[str, str]] = {}
    try:
        df = pd.read_excel(xlsx_path, sheet_name="AeroDyn", header=0)
    except Exception:
        return out
    if "Name" not in df.columns:
        return out
    for _, row in df.iterrows():
        name = row.get("Name")
        if not isinstance(name, str) or not name.strip():
            continue
        unit = row.get("Units", "")
        desc = row.get("Description", "")
        out[name.strip()] = (str(unit) if pd.notna(unit) else "", str(desc) if pd.notna(desc) else "")
    # AeroDyn_Nodes 子表（通用节点变量）
    try:
        df2 = pd.read_excel(xlsx_path, sheet_name="AeroDyn_Nodes", header=0)
        if "Name" in df2.columns:
            for _, row in df2.iterrows():
                name = row.get("Name")
                if isinstance(name, str) and name.strip():
                    unit = row.get("Units", "")
                    desc = row.get("Description", "")
                    out[name.strip()] = (str(unit) if pd.notna(unit) else "",
                                         str(desc) if pd.notna(desc) else "")
    except Exception:
        pass
    if out:
        AERODYN_VARS.update(out)  # 用 xlsx 官方定义覆盖/补充内置字典
    return out


# ============================================================
# 2. .out 头部信息
# ============================================================
@dataclass
class FastFileInfo:
    filepath: str
    filename: str = ""
    file_size_mb: float = 0.0
    columns: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    num_rows: int = 0          # 数据行数
    num_columns: int = 0
    sample_rate_hz: float = 0.0
    total_time: float = 0.0
    creation_time: str = ""
    generator: str = ""        # 生成信息（版本、时间）
    num_blades: int = 0
    num_nodes: int = 0
    # 各变量在文件中的实际列索引（缺失变量不在其中）
    col_index: Dict[str, int] = field(default_factory=dict)

    @property
    def condition_type(self) -> str:
        """工况类型（如 EOG/NTM/ECG），由生成信息推断"""
        return "FAST"

    @property
    def condition_name(self) -> str:
        return ""

    def is_present(self, name: str) -> bool:
        return name in self.col_index


def _is_header_line(line: str) -> bool:
    """判断是否为数据开始后的头部注释行（以 '-----' 或 '-------' 开头）"""
    return line.startswith("-----")


def _find_colname_unit_rows(lines: List[str], start: int = 0):
    """在头部行中定位 列名行 与 单位行 的索引

    规则（OpenFAST/AeroDyn_driver 通用）：
      - 列名行：以空白+Time 开头，且行内不含 '('
      - 单位行：列名行的下一行，以 '(' 开头
    返回 (colname_row_idx, unit_row_idx, data_start_idx) 或 (None, None, None)
    """
    n = len(lines)
    for i in range(start, n):
        line = lines[i].rstrip("\n").rstrip("\r")
        # 跳过注释行与空行
        if not line.strip() or _is_header_line(line):
            continue
        # 列名行：第一个 token 是 Time
        toks = line.split()
        if toks and toks[0] == "Time" and "(" not in line:
            # 下一行应为单位行
            if i + 1 < n:
                u = lines[i + 1].rstrip("\n").rstrip("\r")
                if u.lstrip().startswith("("):
                    return i, i + 1, i + 2
            return i, None, i + 1
        # 兼容列名行前面有 'Time' 但首 token 带空格的情况已由 split 处理
    return None, None, None


def parse_fast_header(filepath: str) -> FastFileInfo:
    """解析 .out 文件头部（不加载全部数据，内存友好）"""
    info = FastFileInfo(filepath=filepath, filename=os.path.basename(filepath))
    info.file_size_mb = os.path.getsize(filepath) / 1024 / 1024
    if not os.path.exists(filepath):
        return info

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        # 读取前 200 行找头部；同时记录生成信息
        head_lines = []
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            head_lines.append(line)
            s = line.strip()
            if s.startswith("Predictions were generated") or "generated on" in s:
                info.generator = s
            elif s.startswith("-----") or "OpenFAST" in s or "FAST v" in s:
                if "OpenFAST" in s or "FAST v" in s:
                    info.generator = (info.generator + " | " if info.generator else "") + s
        # 数据行数（从数据开始处统计）
        col_row, unit_row, data_start = _find_colname_unit_rows(head_lines)
        if col_row is None:
            # 头部可能超过200行（极少数），退化为整行扫描前 5000 行
            head_lines2 = head_lines
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f2:
                    for _ in range(5000):
                        ln = f2.readline()
                        if not ln:
                            break
                        head_lines2.append(ln)
            except Exception:
                pass
            col_row, unit_row, data_start = _find_colname_unit_rows(head_lines2)

    if col_row is None:
        raise ValueError(f"无法识别 .out 文件列名行: {filepath}")

    col_line = head_lines[col_row].rstrip("\n").rstrip("\r")
    unit_line = head_lines[unit_row].rstrip("\n").rstrip("\r") if unit_row is not None else ""
    info.columns = col_line.split()
    if unit_line.strip():
        info.units = unit_line.split()
    else:
        info.units = [""] * len(info.columns)
    info.num_columns = len(info.columns)
    # 列名→索引
    for i, c in enumerate(info.columns):
        info.col_index[c] = i

    # 统计数据行数 + 采样率 + 时长
    n_data = 0
    t0 = t_last = None
    # 从文件头重新定位数据起始（避免重复读）
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or _is_header_line(line):
                continue
            if n_data == 0:
                # 数据第一行：尝试解析时间
                try:
                    t0 = float(s.split()[0])
                except Exception:
                    continue
            n_data += 1
            t_last = s.split()[0]
    info.num_rows = n_data
    try:
        if n_data > 1 and t0 is not None and t_last is not None:
            tn = float(t_last)
            info.total_time = tn - t0
            # 采样率：用前两个数据点估计，需重新读一次取第二行
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                vals = []
                for line in f:
                    s = line.strip()
                    if not s or _is_header_line(line):
                        continue
                    try:
                        vals.append(float(s.split()[0]))
                    except Exception:
                        continue
                    if len(vals) >= 2:
                        break
            if len(vals) >= 2 and vals[1] > vals[0]:
                dt = vals[1] - vals[0]
                info.sample_rate_hz = round(1.0 / dt, 3) if dt > 0 else 0.0
    except Exception:
        pass

    # 叶片/节点推断：从列名 AB{b}N{nnn} 或 B{b}N{n}
    blades, nodes = set(), set()
    for c in info.columns:
        m = re.match(r"^AB?(\d+)N(\d+)", c)
        if m:
            blades.add(int(m.group(1)))
            nodes.add(int(m.group(2)))
    info.num_blades = max(blades) if blades else 0
    info.num_nodes = max(nodes) if nodes else 0
    return info


# ============================================================
# 3. 数据加载
# ============================================================
def load_fast_data(filepath: str, usecols: Optional[List[str]] = None,
                   header: Optional[FastFileInfo] = None) -> "pd.DataFrame":
    """加载 .out 数据为 DataFrame（列名 = FAST 变量名）

    参数:
        filepath: .out 文件路径
        usecols:  仅加载这些列（None=全部）
        header:   已解析的 FastFileInfo（None=自动解析）
    返回: pandas DataFrame
    """
    import pandas as pd
    if header is None:
        header = parse_fast_header(filepath)
    cols = header.columns
    if usecols:
        cols = [c for c in cols if c in usecols]
    # 列索引集合
    keep = {header.col_index[c] for c in cols if c in header.col_index}

    rows: List[List[float]] = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or _is_header_line(line):
                continue
            toks = s.split()
            if len(toks) != header.num_columns:
                continue
            if keep:
                sel = [toks[i] for i in sorted(keep)]
            else:
                sel = toks
            try:
                vals = [float(x) for x in sel]
            except Exception:
                continue
            rows.append(vals)
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=[c for c in cols if header.col_index[c] in keep] if keep else cols)
    return df


# ============================================================
# 4. FAST → 系统通道映射
# ============================================================
class FASTChannelMapper:
    """把 FAST 变量名映射到系统内部通道，容错缺列。

    系统通道命名沿用 QBlade 内部约定（Momentary_Aerodynamic_* 等），
    便于复用前端时序/统计/导出等通用能力。
    """

    def __init__(self, columns: List[str], num_blades: int = 0, num_nodes: int = 0):
        self.columns = list(columns)
        self.num_blades = num_blades or 0
        self.num_nodes = num_nodes or 0
        # 名称规范化：AB1N001Vindx → B1N1Vindx（去AB前缀、去节点前导零）
        self._norm: Dict[str, str] = {}
        for c in columns:
            self._norm[c] = self._normalize_name(c)
        self._build_mapping()

    @staticmethod
    def _normalize_name(name: str) -> str:
        m = re.match(r"^AB?(\d+)N(\d+)(.*)$", name)
        if m:
            b, n, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
            return f"B{b}N{n}{suffix}"
        return name

    def _build_mapping(self):
        """建立系统通道 → 可用FAST列名的映射（只含实际存在的列）"""
        self.global_channels: List[str] = []   # 供时序下拉/导出的全量通道（规范化名）
        self.time_col = "Time_[s]" if "Time" in self._norm.values() else None
        self.channels: Dict[str, str] = {}     # 系统通道 → 实际列名

        def norm2col(norm_name: str) -> Optional[str]:
            for c, n in self._norm.items():
                if n == norm_name:
                    return c
            return None

        # 时间
        tc = norm2col("Time")
        if tc:
            self.channels["Time"] = tc
            self.time_col = tc
        # 转速 / TSR
        for sys_name, fast_name in [
            ("Rotational_Speed_[rpm]", "RtSpeed"),
            ("Rotational_Speed_[rpm]", "RotSpeed"),
            ("Tip_Speed_Ratio_[-]", "RtTSR"),
        ]:
            c = norm2col(fast_name)
            if c and sys_name not in self.channels:
                self.channels[sys_name] = c
        # 功率
        c = norm2col("RtAeroPwr")
        if c:
            self.channels["Momentary_Aerodynamic_Power_[W]"] = c
        # 推力（水平合力）与扭矩（绕 z 轴力矩）→ 系统通道
        fx = norm2col("RtAeroFxh") or norm2col("RtAeroFxg")
        fy = norm2col("RtAeroFyh") or norm2col("RtAeroFyg")
        mz = norm2col("RtAeroMzh") or norm2col("RtAeroMzg")
        fz = norm2col("RtAeroFzh") or norm2col("RtAeroFzg")
        if fx and fy:
            self.channels["Momentary_Aerodynamic_Thrust_[N]"] = "__combo_horiz_F__"
            self._fx_col, self._fy_col = fx, fy
        elif fz:
            self.channels["Momentary_Aerodynamic_Thrust_[N]"] = fz
        if mz:
            self.channels["Momentary_Aerodynamic_Torque_[Nm]"] = mz
        # 风速（AeroDyn_driver 输入）
        wx = norm2col("HWindSpeedX")
        if wx:
            self.channels["Abs_Meas._Wind_Vel._at_Hub_[m/s]"] = wx

        # 全量可用通道（规范化后的列名，供下拉选择）
        for c in self.columns:
            self.global_channels.append(self._norm[c])

        # 叶片节点载荷通道
        self.blade_node_cols: Dict[str, Dict[str, str]] = {}
        if self.num_blades:
            for b in range(1, self.num_blades + 1):
                for n in range(1, self.num_nodes + 1):
                    d: Dict[str, str] = {}
                    for suffix in ["Fx", "Fy", "Fn", "Ft", "Fl", "Fd", "Mm"]:
                        col = norm2col(f"B{b}N{n}{suffix}")
                        if col:
                            d[suffix] = col
                    if d:
                        self.blade_node_cols[f"Blade_{b}_Node_{n}"] = d

    def get_thrust_col(self) -> Optional[str]:
        v = self.channels.get("Momentary_Aerodynamic_Thrust_[N]")
        if v == "__combo_horiz_F__":
            return None  # 需要组合计算
        return v

    def get_horiz_force_cols(self) -> Optional[Tuple[str, str]]:
        if "Momentary_Aerodynamic_Thrust_[N]" in self.channels and \
           self.channels["Momentary_Aerodynamic_Thrust_[N]"] == "__combo_horiz_F__":
            return (self._fx_col, self._fy_col)
        return None

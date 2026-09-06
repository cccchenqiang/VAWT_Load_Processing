"""
全局配置模块
垂直轴风轮(VAWT)载荷数据处理 - QBlade仿真输出适配
所有叶片数量、通道命名、大文件读取参数、工况切片参数集中配置
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ============================================================
# 1. 路径配置
# ============================================================
@dataclass
class PathConfig:
    """输出路径配置"""
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    output_dir: str = os.path.join(base_dir, "output")
    blade_single_dir: str = os.path.join(base_dir, "output", "blade_single")
    wheel_total_dir: str = os.path.join(base_dir, "output", "wheel_total")
    compare_dir: str = os.path.join(base_dir, "output", "compare")
    figure_dir: str = os.path.join(base_dir, "output", "figures")

    def __post_init__(self):
        for d in [self.output_dir, self.blade_single_dir,
                  self.wheel_total_dir, self.compare_dir, self.figure_dir]:
            os.makedirs(d, exist_ok=True)


# ============================================================
# 2. 叶片与通道配置
# ============================================================
@dataclass
class BladeConfig:
    """叶片配置 - 自动从数据中检测，也可手动指定"""
    # 叶片数量（None=自动检测）
    num_blades: Optional[int] = None
    # 叶片编号列表（自动生成 Blade_1, Blade_2, ...）
    blade_ids: List[str] = field(default_factory=list)
    # 每叶片面板数（QBlade中PAN_0到PAN_19，共20个）
    num_panels: int = 20

    # ---- 叶片总载荷通道（QBlade命名规则）----
    # 总切向载荷 [N]
    total_tangential_pattern: str = "Total_Tangential_Load_Blade_{n}_[N]"
    # 总法向载荷 [N]
    total_normal_pattern: str = "Total_Normal_Load_Blade_{n}_[N]"
    # 方位角 [deg]
    azimuth_pattern: str = "Azimuthal_Position_Blade_{n}_[deg]"
    # 桨距角 [deg]
    pitch_pattern: str = "Pitch_Angle_Blade_{n}_[deg]"

    # ---- 面板级分布载荷通道 ----
    # 法向力分布 [N/m]
    panel_normal_force_pattern: str = "Normal_Force_Blade_{n}_PAN_{p}_[N/m]"
    # 切向力分布 [N/m]
    panel_tangential_force_pattern: str = "Tangential_Force_Blade_{n}_PAN_{p}_[N/m]"
    # 俯仰力矩分布 [Nm/m]
    panel_pitching_moment_pattern: str = "Pitching_Moment_Blade_{n}_PAN_{p}_[Nm/m]"
    # 面板高度 [m]
    panel_height_pattern: str = "Height_Blade_{n}_PAN_{p}_[m]"

    # ---- 面板级气动系数（可选分析）----
    panel_lift_coeff_pattern: str = "Lift_Coefficient_Blade_{n}_PAN_{p}_[-]"
    panel_drag_coeff_pattern: str = "Drag_Coefficient_Blade_{n}_PAN_{p}_[-]"
    panel_aoa_pattern: str = "Angle_of_Attack_at_0.25c_Blade_{n}_PAN_{p}_[deg]"
    panel_re_pattern: str = "Reynolds_Number_Blade_{n}_PAN_{p}_[-]"

    def get_blade_id(self, n: int) -> str:
        """获取叶片编号字符串，如 Blade_1"""
        return f"Blade_{n}"

    def detect_blades_from_columns(self, columns: List[str]) -> int:
        """从列名中自动检测叶片数量"""
        import re
        blade_nums = set()
        for col in columns:
            m = re.search(r'Blade_(\d+)', col)
            if m:
                blade_nums.add(int(m.group(1)))
        if blade_nums:
            self.num_blades = max(blade_nums)
            self.blade_ids = [self.get_blade_id(n) for n in range(1, self.num_blades + 1)]
        return self.num_blades or 0


# ============================================================
# 3. 整机/全局通道配置
# ============================================================
@dataclass
class GlobalChannelConfig:
    """整机全局通道（非叶片专属）"""
    # 时间
    time_col: str = "Time_[s]"
    timestep_col: str = "Timestep_[s]"

    # 运行参数
    rpm_col: str = "Rotational_Speed_[rpm]"
    tsr_col: str = "Tip_Speed_Ratio_[-]"

    # 风速
    wind_vel_hub_col: str = "Abs_Meas._Wind_Vel._at_Hub_[m/s]"
    wind_vel_rotor_col: str = "Abs_Meas._Wind_Vel._Rotor_Avg._[m/s]"

    # 气动性能（瞬时值）
    inst_power_col: str = "Momentary_Aerodynamic_Power_[W]"
    inst_torque_col: str = "Momentary_Aerodynamic_Torque_[Nm]"
    inst_thrust_col: str = "Momentary_Aerodynamic_Thrust_[N]"

    # 气动性能（平均值/滤波值）
    avg_power_col: str = "Aerodynamic_Power_[W]"
    avg_torque_col: str = "Aerodynamic_Torque_[Nm]"
    avg_thrust_col: str = "Aerodynamic_Thrust_[N]"

    # 整机力/力矩（全局坐标系）
    thrust_x_col: str = "Momentary_Thrust_in_X_g_Direction_[N]"
    thrust_y_col: str = "Momentary_Thrust_in_Y_g_Direction_[N]"
    thrust_z_col: str = "Momentary_Thrust_in_Z_g_Direction_[N]"
    moment_x_col: str = "Momentary_Moment_in_X_g_Direction_[Nm]"
    moment_y_col: str = "Momentary_Moment_in_Y_g_Direction_[Nm]"
    moment_z_col: str = "Momentary_Moment_in_Z_g_Direction_[Nm]"

    # 系数
    cp_col: str = "Power_Coefficient_[-]"
    ct_col: str = "Thrust_Coefficient_[-]"
    cq_col: str = "Torque_Coefficient_[-]"


# ============================================================
# 4. 大文件读取配置
# ============================================================
@dataclass
class BigFileConfig:
    """大文件分块读取配置"""
    # 分块行数（每块读取的数据行数）
    chunksize: int = 10000
    # 文件编码
    encoding: str = "utf-8"
    # 注释行标识（以这些开头的行跳过，除了工况标识行）
    comment_prefixes: tuple = ("Results Output File",)
    # 工况标识行前缀列表（用于提取工况名称，支持EOG/NTM/EWS/ECD等IEC标准工况）
    condition_prefixes: tuple = ("EOG:", "NTM:", "EWS:", "ECD:", "NWP:", "EWM:",
                                 "DLC:", "TURB:", "STEADY:", "TRANSIENT:")
    # 兼容旧版：单工况前缀（已弃用，保留用于向后兼容）
    condition_prefix: str = "EOG:"
    # 表头所在行索引（0-based，跳过注释行后的第一行）
    header_row_offset: int = 0
    # 分隔符
    sep: str = "\t"
    # 科学计数法数据
    scientific_notation: bool = True
    # 内存上限（MB），超过则自动减小分块
    memory_limit_mb: float = 512.0
    # 是否启用跨块峰值校验（EOG分析用）
    cross_block_peak_check: bool = True


# ============================================================
# 5. 预处理配置
# ============================================================
@dataclass
class PreprocessConfig:
    """数据预处理配置"""
    # 缺失值处理：ffill(前向填充), bfill(后向填充), interpolate(插值), drop(删除)
    missing_method: str = "interpolate"
    # 异常值阈值（标准差倍数）
    outlier_sigma: float = 5.0
    # 是否启用低通滤波
    enable_filter: bool = True
    # 巴特沃斯低通滤波截止频率 [Hz]
    filter_cutoff_hz: float = 5.0
    # 滤波阶数
    filter_order: int = 4
    # 是否去趋势
    detrend: bool = False
    # EOG瞬态保护：滤波时保留峰值（不滤除阵风冲击）
    preserve_transient_peaks: bool = True
    # 瞬态峰值保护阈值（标准差倍数）
    transient_threshold_sigma: float = 3.0


# ============================================================
# 6. 极限载荷分析配置（通用，支持EOG/NTM/EWS等所有风况）
# ============================================================
@dataclass
class ExtremeLoadConfig:
    """
    极限载荷分析配置（通用化，支持所有风况类型）
    - EOG/EWS/ECD等瞬态工况：使用阵风切片策略
    - NTM/NWP等稳态工况：使用全量极值或滑动窗口极值
    """
    # 极值检测策略：auto(自动根据工况选择), eog_surge(EOG阵风切片),
    #              full(全量极值), sliding_window(滑动窗口极值)
    extreme_strategy: str = "auto"
    # EOG阵风触发：风速变化率阈值 [m/s per s]（仅eog_surge策略用）
    wind_surge_rate_threshold: float = 1.0
    # EOG阵风触发：载荷变化率阈值（相对均值的倍数）
    load_surge_factor: float = 2.0
    # 极值统计窗口（阵风前后各多少秒，仅eog_surge策略用）
    extreme_window_sec: float = 2.0
    # 滑动窗口大小 [秒]（仅sliding_window策略用）
    sliding_window_sec: float = 3.0
    # 滑动窗口步长 [秒]
    sliding_step_sec: float = 1.0
    # 统计指标
    stats_metrics: tuple = ("max", "min", "mean", "std", "peak_to_peak", "rms", "abs_max")
    # 是否计算最不利叶片
    find_critical_blade: bool = True
    # 最不利叶片评判指标（可用 total_normal, total_tangential, combined）
    critical_blade_metric: str = "combined"
    # 瞬态工况类型（这些工况使用eog_surge策略）
    transient_condition_types: tuple = ("EOG", "EWS", "ECD", "EWM", "TRANSIENT")
    # 稳态工况类型（这些工况使用full或sliding_window策略）
    steady_condition_types: tuple = ("NTM", "NWP", "STEADY", "TURB")


# 向后兼容别名
EOGConfig = ExtremeLoadConfig


# ============================================================
# 7. 疲劳分析配置
# ============================================================
@dataclass
class FatigueConfig:
    """疲劳载荷分析配置"""
    # 雨流计数配置
    rainflow_bins: int = 64
    # 是否分块雨流计数（大文件用）
    chunked_rainflow: bool = True
    # 分块重叠比例（防止跨块循环丢失）
    chunk_overlap_ratio: float = 0.1
    # Miner累积损伤 - S-N曲线参数（默认钢材，可修改）
    sn_m: float = 3.0          # S-N曲线斜率倒数
    sn_log_a: float = 12.0     # S-N曲线截距（log10）
    # 参考应力范围（用于归一化，单位与载荷一致）
    reference_range: float = 1.0
    # 等效疲劳载荷计算时长 [秒]（如设计寿命20年=630720000s）
    design_life_sec: float = 630720000.0
    #  Goodman修正是否启用
    goodman_correction: bool = False
    # 材料极限强度（Goodman修正用）
    ultimate_strength: float = 1.0
    # 是否包含极端载荷段参与疲劳累积（通用名，原include_eog_in_fatigue）
    include_extreme_in_fatigue: bool = True


# ============================================================
# 8. 叶片合成配置
# ============================================================
@dataclass
class SynthesisConfig:
    """多叶片载荷合成配置"""
    # 是否启用整轮合成
    enable_synthesis: bool = True
    # 不平衡度计算方式：std/mean, max/min, (max-min)/mean
    imbalance_method: str = "std_over_mean"
    # 合成坐标系：global(全局), rotational(旋转)
    coordinate_system: str = "global"
    # 是否计算倾覆弯矩
    compute_overturning_moment: bool = True
    # 风轮半径 [m]（用于力矩合成，需根据实际机型修改）
    rotor_radius: float = 5.0
    # 叶片高度 [m]
    blade_height: float = 10.0


# ============================================================
# 9. 可视化配置
# ============================================================
@dataclass
class VisualizeConfig:
    """可视化配置"""
    # 图片格式
    fig_format: str = "png"
    # 图片DPI
    dpi: int = 150
    # 图片尺寸 (宽, 高) 英寸
    figsize: tuple = (12, 6)
    # 中文字体（Windows下用SimHei，Linux下用WenQuanYi）
    font_family: str = "SimHei"
    # 是否显示网格
    grid: bool = True
    # 颜色方案
    colors: tuple = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    # 是否批量生成单叶片时序图
    batch_blade_timeseries: bool = True
    # 是否生成面板载荷云图
    panel_heatmap: bool = True


# ============================================================
# 10. 运行模式配置
# ============================================================
@dataclass
class RunConfig:
    """运行模式配置"""
    # 分析模式：extreme(极限载荷), fatigue(疲劳), both(两者)
    # 兼容旧版：eog 等同于 extreme
    analysis_mode: str = "both"
    # 是否启用并行处理（多叶片并行）
    enable_parallel: bool = False
    # 并行工作进程数
    n_workers: int = 2
    # 日志级别：DEBUG, INFO, WARNING, ERROR
    log_level: str = "INFO"
    # 是否保存中间结果（预处理后的数据）
    save_intermediate: bool = False
    # 中间结果格式：parquet, csv, pickle
    intermediate_format: str = "parquet"


# ============================================================
# 11. 塔顶载荷分析配置
# ============================================================
@dataclass
class TowerTopConfig:
    """
    塔顶载荷分析配置
    校核对象：风轮几何中心的塔架顶点载荷
    - 塔顶水平推力：Momentary_Aerodynamic_Thrust_[N]
    - 塔顶扭矩：Momentary_Aerodynamic_Torque_[Nm]
    - 倾覆弯矩：由全局X/Y向力矩合成 sqrt(Mx²+My²)
    - 塔顶合力：推力与扭矩合成的校核值
    """
    # 塔顶推力通道（直接复用全局通道）
    thrust_col: str = "Momentary_Aerodynamic_Thrust_[N]"
    # 塔顶扭矩通道
    torque_col: str = "Momentary_Aerodynamic_Torque_[Nm]"
    # 倾覆弯矩合成：使用全局X/Y向力矩
    use_global_moment: bool = True
    moment_x_col: str = "Momentary_Moment_in_X_g_Direction_[Nm]"
    moment_y_col: str = "Momentary_Moment_in_Y_g_Direction_[Nm]"
    # 疲劳Wöhler指数（塔架=焊接钢结构，m=3）
    fatigue_m: float = 3.0
    # 是否计算塔顶合力（已取消：每台风机半径/结构不同，折算系数因人而异，输出会误导）
    compute_resultant: bool = False
    # 统计指标
    stats_metrics: tuple = ("max", "min", "mean", "std", "rms", "peak_to_peak")


# ============================================================
# 12. 批处理配置（多工况DLC）
# ============================================================
@dataclass
class BatchConfig:
    """
    多工况批处理配置（IEC 61400 DLC）
    目标：跨所有工况找到能包络所有载荷的载荷，用于部件校核
    """
    # 并发处理数（大文件每份几百MB~2GB，建议2-3）
    n_workers: int = 2
    # 工况清单配置文件（cases.yaml），不提供则自动扫描目录+自动识别
    cases_file: str = ""
    # 部件-疲劳Wöhler指数映射
    fatigue_m: dict = field(default_factory=lambda: {
        "blade": 10,        # 叶片（复合材料）
        "tower": 3,         # 塔架（焊接钢结构）
        "mechanical": 4,    # 机械零部件（螺栓、轴承）
    })
    # 设计寿命 [秒]（默认20年）
    design_life_sec: float = 630720000.0
    # 工况类型→DLC编号默认映射（自动识别时使用）
    dlc_map: dict = field(default_factory=lambda: {
        "EOG": "1.3", "EWM": "1.3", "NTM": "1.2", "NWP": "1.1",
        "EWS": "1.4", "ECD": "2.3", "STEADY": "1.1", "TURB": "1.2",
    })
    # 是否包含面板级数据（批处理默认不加载以省内存，除非需要叶片分布载荷）
    include_panel: bool = False


# ============================================================
# 全局配置实例
# ============================================================
class Config:
    """统一配置入口"""
    def __init__(self):
        self.path = PathConfig()
        self.blade = BladeConfig()
        self.global_ch = GlobalChannelConfig()
        self.bigfile = BigFileConfig()
        self.preprocess = PreprocessConfig()
        self.extreme = ExtremeLoadConfig()
        self.eog = self.extreme  # 向后兼容：eog 指向 extreme
        self.fatigue = FatigueConfig()
        self.synthesis = SynthesisConfig()
        self.visualize = VisualizeConfig()
        self.run = RunConfig()
        self.tower_top = TowerTopConfig()
        self.batch = BatchConfig()

    def summary(self) -> str:
        """输出配置摘要"""
        lines = [
            "=" * 60,
            "VAWT载荷处理系统 - 配置摘要",
            "=" * 60,
            f"分析模式: {self.run.analysis_mode}",
            f"极限载荷策略: {self.extreme.extreme_strategy}",
            f"叶片数量: {self.blade.num_blades or '自动检测'}",
            f"每叶片面板数: {self.blade.num_panels}",
            f"大文件分块: {self.bigfile.chunksize} 行/块",
            f"内存上限: {self.bigfile.memory_limit_mb} MB",
            f"滤波截止频率: {self.preprocess.filter_cutoff_hz} Hz",
            f"疲劳S-N斜率(m): {self.fatigue.sn_m}",
            f"设计寿命: {self.fatigue.design_life_sec/31536000:.0f} 年",
            f"并行处理: {'开启' if self.run.enable_parallel else '关闭'}",
            "=" * 60,
        ]
        return "\n".join(lines)


# 全局单例
config = Config()

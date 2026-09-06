"""
EOG极端阵风分析模块（向后兼容层）

本模块已通用化为 extreme_load_analysis.py，支持EOG/NTM/EWS等所有风况类型。
保留此文件仅为向后兼容，所有类和函数均从extreme_load_analysis导入。

新代码请使用:
    from extreme_load_analysis import run_extreme_analysis, BladeExtremeAnalyzer, WheelExtremeAnalyzer
"""

from extreme_load_analysis import (
    BladeExtremeAnalyzer as BladeEOGAnalyzer,
    WheelExtremeAnalyzer as WheelEOGAnalyzer,
    run_extreme_analysis as run_eog_analysis,
    run_extreme_analysis,
    BladeExtremeAnalyzer,
    WheelExtremeAnalyzer,
)

__all__ = [
    "BladeEOGAnalyzer",
    "WheelEOGAnalyzer",
    "run_eog_analysis",
    "BladeExtremeAnalyzer",
    "WheelExtremeAnalyzer",
    "run_extreme_analysis",
]

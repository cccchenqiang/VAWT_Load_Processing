"""
垂直轴风轮载荷数据处理系统 - 图形界面 (Tkinter)
基于已有分析模块，提供可视化操作界面

运行方式：
    python gui.py
"""

import os
import sys
import threading
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# matplotlib嵌入Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 导入已有分析模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from bigfile_io import BigFileLoader, ConditionSlicer, quick_inspect
from preprocess import preprocess_dataframe
from extreme_load_analysis import run_extreme_analysis
from fatigue_analysis import run_fatigue_analysis
from blade_synthesis import synthesize_all_blades
from utils import export_to_excel


# ============================================================
# 全局样式
# ============================================================
COLOR_BG = "#f0f2f5"
COLOR_PANEL = "#ffffff"
COLOR_PRIMARY = "#1f77b4"
COLOR_ACCENT = "#ff7f0e"
COLOR_DANGER = "#d62728"
COLOR_SUCCESS = "#2ca02c"
COLOR_TEXT = "#2c3e50"
COLOR_MUTED = "#7f8c8d"


class LoadAnalysisGUI:
    """载荷分析GUI主窗口"""

    def __init__(self, root):
        self.root = root
        self.root.title("垂直轴风轮载荷数据处理系统")
        self.root.geometry("1280x820")
        self.root.minsize(1024, 680)
        self.root.configure(bg=COLOR_BG)

        # 状态变量
        self.filepath = tk.StringVar(value="")
        self.analysis_mode = tk.StringVar(value="both")
        self.include_panel = tk.BooleanVar(value=False)
        self.filter_enabled = tk.BooleanVar(value=True)
        self.filter_cutoff = tk.DoubleVar(value=5.0)
        self.sn_m = tk.DoubleVar(value=3.0)
        self.sn_log_a = tk.DoubleVar(value=12.0)
        self.status_text = tk.StringVar(value="就绪")
        self.progress_val = tk.DoubleVar(value=0)
        self.is_running = False

        # 分析结果缓存
        self.results = None
        self.df = None
        self.mapper = None
        self.loader = None

        self._build_ui()
        self._setup_style()

    def _setup_style(self):
        """设置ttk样式"""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Panel.TFrame", background=COLOR_PANEL)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Microsoft YaHei", 9))
        style.configure("Panel.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Microsoft YaHei", 9))
        style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_TEXT,
                        font=("Microsoft YaHei", 14, "bold"))
        style.configure("Subtitle.TLabel", background=COLOR_BG, foreground=COLOR_MUTED,
                        font=("Microsoft YaHei", 9))
        style.configure("TButton", font=("Microsoft YaHei", 9), padding=6)
        style.configure("Primary.TButton", font=("Microsoft YaHei", 10, "bold"),
                        padding=8, background=COLOR_PRIMARY, foreground="white")
        style.map("Primary.TButton", background=[("active", "#1565c0")])
        style.configure("TEntry", padding=4)
        style.configure("TCheckbutton", background=COLOR_BG, font=("Microsoft YaHei", 9))
        style.configure("TRadiobutton", background=COLOR_BG, font=("Microsoft YaHei", 9))
        style.configure("Horizontal.TProgressbar", thickness=16)
        style.configure("TNotebook", background=COLOR_BG, tabmargins=[2, 5, 2, 0])
        style.configure("TNotebook.Tab", padding=[16, 8], font=("Microsoft YaHei", 9))
        style.configure("Treeview", font=("Microsoft YaHei", 9), rowheight=26)
        style.configure("Treeview.Heading", font=("Microsoft YaHei", 9, "bold"))

    def _build_ui(self):
        """构建界面"""
        # 顶部标题栏
        header = ttk.Frame(self.root, style="TFrame")
        header.pack(fill="x", padx=16, pady=(12, 8))
        ttk.Label(header, text="垂直轴风轮载荷数据处理系统", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="  QBlade仿真输出 | 极限载荷(EOG/NTM通用) + 疲劳分析 | 多叶片解耦",
                  style="Subtitle.TLabel").pack(side="left", pady=(6, 0))

        # 主内容区：左侧控制面板 + 右侧结果区
        main = ttk.Frame(self.root, style="TFrame")
        main.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        self._build_control_panel(main)
        self._build_result_area(main)

        # 底部状态栏
        status_bar = ttk.Frame(self.root, style="TFrame")
        status_bar.pack(fill="x", padx=16, pady=(0, 12))
        ttk.Label(status_bar, textvariable=self.status_text, style="Subtitle.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(status_bar, variable=self.progress_val,
                                         maximum=100, style="Horizontal.TProgressbar")
        self.progress.pack(side="right", fill="x", expand=True, padx=(16, 0))

    def _build_control_panel(self, parent):
        """左侧控制面板"""
        panel = ttk.Frame(parent, style="Panel.TFrame", width=300)
        panel.pack(side="left", fill="y", padx=(0, 12))
        panel.pack_propagate(False)

        # 文件选择
        self._section_label(panel, "1. 数据文件")
        file_frame = ttk.Frame(panel, style="Panel.TFrame")
        file_frame.pack(fill="x", padx=12, pady=(4, 8))
        self.file_entry = ttk.Entry(file_frame, textvariable=self.filepath, font=("Microsoft YaHei", 9))
        self.file_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="浏览...", command=self._browse_file, width=8).pack(side="left", padx=(6, 0))

        # 文件信息显示
        self.file_info_text = scrolledtext.ScrolledText(
            panel, height=5, font=("Consolas", 8), wrap="word",
            bg="#f8f9fa", relief="flat", padx=8, pady=6
        )
        self.file_info_text.pack(fill="x", padx=12, pady=(0, 8))
        self.file_info_text.insert("1.0", "请选择QBlade仿真输出文件(.txt)...")
        self.file_info_text.config(state="disabled")

        # 分析模式
        self._section_label(panel, "2. 分析模式")
        mode_frame = ttk.Frame(panel, style="Panel.TFrame")
        mode_frame.pack(fill="x", padx=12, pady=(4, 8))
        ttk.Radiobutton(mode_frame, text="极限载荷分析 (EOG/NTM通用)", variable=self.analysis_mode,
                        value="extreme").pack(anchor="w", pady=2)
        ttk.Radiobutton(mode_frame, text="疲劳载荷分析", variable=self.analysis_mode,
                        value="fatigue").pack(anchor="w", pady=2)
        ttk.Radiobutton(mode_frame, text="两者都做 (推荐)", variable=self.analysis_mode,
                        value="both").pack(anchor="w", pady=2)

        # 高级参数
        self._section_label(panel, "3. 高级参数")
        param_frame = ttk.Frame(panel, style="Panel.TFrame")
        param_frame.pack(fill="x", padx=12, pady=(4, 8))

        ttk.Checkbutton(param_frame, text="加载面板级分布载荷 (内存大)",
                        variable=self.include_panel).pack(anchor="w", pady=2)
        ttk.Checkbutton(param_frame, text="启用低通滤波",
                        variable=self.filter_enabled).pack(anchor="w", pady=2)

        cutoff_row = ttk.Frame(param_frame, style="Panel.TFrame")
        cutoff_row.pack(fill="x", pady=2)
        ttk.Label(cutoff_row, text="滤波截止(Hz):", style="Panel.TLabel", width=12).pack(side="left")
        ttk.Entry(cutoff_row, textvariable=self.filter_cutoff, width=8).pack(side="left")

        sn_row1 = ttk.Frame(param_frame, style="Panel.TFrame")
        sn_row1.pack(fill="x", pady=2)
        ttk.Label(sn_row1, text="S-N斜率m:", style="Panel.TLabel", width=12).pack(side="left")
        ttk.Entry(sn_row1, textvariable=self.sn_m, width=8).pack(side="left")

        sn_row2 = ttk.Frame(param_frame, style="Panel.TFrame")
        sn_row2.pack(fill="x", pady=2)
        ttk.Label(sn_row2, text="S-N截距logA:", style="Panel.TLabel", width=12).pack(side="left")
        ttk.Entry(sn_row2, textvariable=self.sn_log_a, width=8).pack(side="left")

        # 运行按钮
        btn_frame = ttk.Frame(panel, style="Panel.TFrame")
        btn_frame.pack(fill="x", padx=12, pady=(12, 8))
        self.run_btn = ttk.Button(btn_frame, text="▶ 开始分析", style="Primary.TButton",
                                  command=self._start_analysis)
        self.run_btn.pack(fill="x")

        # 导出按钮
        export_frame = ttk.Frame(panel, style="Panel.TFrame")
        export_frame.pack(fill="x", padx=12, pady=(0, 8))
        self.export_btn = ttk.Button(export_frame, text="📊 导出Excel报表",
                                     command=self._export_results, state="disabled")
        self.export_btn.pack(fill="x")

        # 日志区
        self._section_label(panel, "运行日志")
        self.log_text = scrolledtext.ScrolledText(
            panel, height=8, font=("Consolas", 8), wrap="word",
            bg="#1e1e1e", fg="#d4d4d4", relief="flat", padx=8, pady=6
        )
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    def _section_label(self, parent, text):
        """分区标题"""
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="x", padx=12, pady=(10, 0))
        tk.Frame(frame, bg=COLOR_PRIMARY, width=3, height=16).pack(side="left", padx=(0, 6))
        ttk.Label(frame, text=text, style="Panel.TLabel",
                  font=("Microsoft YaHei", 10, "bold")).pack(side="left")

    def _build_result_area(self, parent):
        """右侧结果展示区"""
        result_frame = ttk.Frame(parent, style="TFrame")
        result_frame.pack(side="left", fill="both", expand=True)

        # 概览卡片
        self._build_overview_cards(result_frame)

        # Notebook多标签页
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill="both", expand=True, pady=(8, 0))

        # 标签页1: 极限载荷
        self.eog_frame = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.eog_frame, text="  极限载荷分析  ")
        self._build_eog_tab()

        # 标签页2: 疲劳分析
        self.fatigue_frame = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.fatigue_frame, text="  疲劳分析  ")
        self._build_fatigue_tab()

        # 标签页3: 叶片合成
        self.synth_frame = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.synth_frame, text="  叶片合成  ")
        self._build_synth_tab()

        # 标签页4: 时序图表
        self.chart_frame = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.chart_frame, text="  时序图表  ")
        self._build_chart_tab()

    def _build_overview_cards(self, parent):
        """概览指标卡片"""
        cards_frame = ttk.Frame(parent, style="TFrame")
        cards_frame.pack(fill="x")

        self.cards = {}
        card_defs = [
            ("torque", "平均扭矩", "Nm", COLOR_PRIMARY),
            ("thrust", "平均推力", "N", COLOR_ACCENT),
            ("power", "平均功率", "kW", COLOR_SUCCESS),
            ("critical", "最不利叶片", "-", COLOR_DANGER),
        ]

        for i, (key, title, unit, color) in enumerate(card_defs):
            card = tk.Frame(cards_frame, bg=COLOR_PANEL, highlightthickness=1,
                            highlightcolor="#e0e0e0")
            card.pack(side="left", fill="x", expand=True, padx=(0 if i == 0 else 8, 0))
            card.config(height=72)
            card.pack_propagate(False)

            tk.Label(card, text=title, bg=COLOR_PANEL, fg=COLOR_MUTED,
                     font=("Microsoft YaHei", 9)).pack(anchor="w", padx=12, pady=(8, 0))
            value_label = tk.Label(card, text=f"-- {unit}", bg=COLOR_PANEL, fg=color,
                                   font=("Microsoft YaHei", 16, "bold"))
            value_label.pack(anchor="w", padx=12, pady=(2, 0))
            self.cards[key] = value_label

    def _build_eog_tab(self):
        """EOG极值标签页"""
        # 上下布局：上表格下图
        top = ttk.Frame(self.eog_frame, style="TFrame")
        top.pack(fill="both", expand=True)

        # 表格
        table_frame = ttk.Frame(top, style="Panel.TFrame")
        table_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        ttk.Label(table_frame, text="叶片极限载荷极值汇总", style="Panel.TLabel",
                  font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        self.eog_tree = ttk.Treeview(table_frame, show="headings", height=6)
        self.eog_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        eog_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.eog_tree.yview)
        eog_scroll.pack(side="right", fill="y", pady=(30, 10))
        self.eog_tree.configure(yscrollcommand=eog_scroll.set)

        # 图表
        chart_frame = ttk.Frame(top, style="Panel.TFrame", width=420)
        chart_frame.pack(side="right", fill="both")
        chart_frame.pack_propagate(False)
        ttk.Label(chart_frame, text="极限载荷极值对比", style="Panel.TLabel",
                  font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.eog_fig = Figure(figsize=(4, 3.5), dpi=100, facecolor=COLOR_PANEL)
        self.eog_canvas = FigureCanvasTkAgg(self.eog_fig, master=chart_frame)
        self.eog_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # 整机载荷信息
        bottom = ttk.Frame(self.eog_frame, style="Panel.TFrame")
        bottom.pack(fill="x", pady=(8, 0))
        ttk.Label(bottom, text="整机载荷统计", style="Panel.TLabel",
                  font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.eog_global_text = scrolledtext.ScrolledText(
            bottom, height=4, font=("Consolas", 9), wrap="word",
            bg="#f8f9fa", relief="flat", padx=10, pady=6
        )
        self.eog_global_text.pack(fill="x", padx=10, pady=(0, 10))

    def _build_fatigue_tab(self):
        """疲劳分析标签页"""
        top = ttk.Frame(self.fatigue_frame, style="TFrame")
        top.pack(fill="both", expand=True)

        # 表格
        table_frame = ttk.Frame(top, style="Panel.TFrame")
        table_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        ttk.Label(table_frame, text="叶片疲劳损伤汇总", style="Panel.TLabel",
                  font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.fatigue_tree = ttk.Treeview(table_frame, show="headings", height=6)
        self.fatigue_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 图表
        chart_frame = ttk.Frame(top, style="Panel.TFrame", width=420)
        chart_frame.pack(side="right", fill="both")
        chart_frame.pack_propagate(False)
        ttk.Label(chart_frame, text="疲劳损伤对比", style="Panel.TLabel",
                  font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.fatigue_fig = Figure(figsize=(4, 3.5), dpi=100, facecolor=COLOR_PANEL)
        self.fatigue_canvas = FigureCanvasTkAgg(self.fatigue_fig, master=chart_frame)
        self.fatigue_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # 载荷谱信息
        bottom = ttk.Frame(self.fatigue_frame, style="Panel.TFrame")
        bottom.pack(fill="x", pady=(8, 0))
        ttk.Label(bottom, text="载荷谱与损伤详情", style="Panel.TLabel",
                  font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.fatigue_detail_text = scrolledtext.ScrolledText(
            bottom, height=4, font=("Consolas", 9), wrap="word",
            bg="#f8f9fa", relief="flat", padx=10, pady=6
        )
        self.fatigue_detail_text.pack(fill="x", padx=10, pady=(0, 10))

    def _build_synth_tab(self):
        """叶片合成标签页"""
        # 不平衡度表格
        top = ttk.Frame(self.synth_frame, style="Panel.TFrame")
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="多叶片载荷不平衡度", style="Panel.TLabel",
                  font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.synth_text = scrolledtext.ScrolledText(
            top, height=5, font=("Consolas", 9), wrap="word",
            bg="#f8f9fa", relief="flat", padx=10, pady=6
        )
        self.synth_text.pack(fill="x", padx=10, pady=(0, 10))

        # 合成载荷图表
        chart_frame = ttk.Frame(self.synth_frame, style="Panel.TFrame")
        chart_frame.pack(fill="both", expand=True)
        ttk.Label(chart_frame, text="整轮合成载荷时序", style="Panel.TLabel",
                  font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.synth_fig = Figure(figsize=(8, 3.5), dpi=100, facecolor=COLOR_PANEL)
        self.synth_canvas = FigureCanvasTkAgg(self.synth_fig, master=chart_frame)
        self.synth_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _build_chart_tab(self):
        """时序图表标签页"""
        # 通道选择
        ctrl = ttk.Frame(self.chart_frame, style="Panel.TFrame")
        ctrl.pack(fill="x", pady=(0, 8))
        ttk.Label(ctrl, text="选择通道:", style="Panel.TLabel").pack(side="left", padx=(10, 6), pady=10)
        self.chart_channel = tk.StringVar(value="Momentary_Aerodynamic_Torque_[Nm]")
        self.chart_combo = ttk.Combobox(ctrl, textvariable=self.chart_channel, width=42,
                                        state="readonly")
        self.chart_combo.pack(side="left", pady=10)
        ttk.Button(ctrl, text="绘制", command=self._draw_chart).pack(side="left", padx=(8, 10), pady=10)

        # 图表
        chart_frame = ttk.Frame(self.chart_frame, style="Panel.TFrame")
        chart_frame.pack(fill="both", expand=True)
        self.chart_fig = Figure(figsize=(10, 4.5), dpi=100, facecolor=COLOR_PANEL)
        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, master=chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))
        toolbar = NavigationToolbar2Tk(self.chart_canvas, chart_frame)
        toolbar.update()

    # ============================================================
    # 事件处理
    # ============================================================
    def _browse_file(self):
        """选择文件"""
        path = filedialog.askopenfilename(
            title="选择QBlade仿真输出文件",
            filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if path:
            self.filepath.set(path)
            self._preview_file(path)

    def _preview_file(self, path):
        """预览文件信息"""
        try:
            info = quick_inspect(path)
            self.file_info_text.config(state="normal")
            self.file_info_text.delete("1.0", "end")
            self.file_info_text.insert("1.0", info)
            self.file_info_text.config(state="disabled")
            self._log(f"文件已加载: {os.path.basename(path)}")
        except Exception as e:
            self.file_info_text.config(state="normal")
            self.file_info_text.delete("1.0", "end")
            self.file_info_text.insert("1.0", f"文件解析失败: {e}")
            self.file_info_text.config(state="disabled")

    def _start_analysis(self):
        """开始分析"""
        if self.is_running:
            return
        path = self.filepath.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("提示", "请先选择有效的数据文件！")
            return

        self.is_running = True
        self.run_btn.config(state="disabled", text="分析中...")
        self.export_btn.config(state="disabled")
        self.progress_val.set(0)

        # 后台线程运行
        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self):
        """后台运行分析"""
        try:
            path = self.filepath.get().strip()
            mode = self.analysis_mode.get()
            include_panel = self.include_panel.get()

            # 更新配置
            config.preprocess.enable_filter = self.filter_enabled.get()
            config.preprocess.filter_cutoff_hz = self.filter_cutoff.get()
            config.fatigue.sn_m = self.sn_m.get()
            config.fatigue.sn_log_a = self.sn_log_a.get()

            self._set_status("正在解析文件...")
            self._set_progress(5)
            self._log(f"开始分析: {os.path.basename(path)}")
            self._log(f"模式: {mode}, 面板数据: {'是' if include_panel else '否'}")

            # 1. 加载
            self.loader = BigFileLoader(path)
            self.mapper = self.loader.mapper
            info = self.loader.info
            self._log(f"文件: {info.num_data_rows}行 x {info.num_columns}列, "
                      f"{info.num_blades}叶片, {info.num_panels}面板")

            self._set_status("正在加载数据...")
            self._set_progress(15)
            t_array = self.loader.get_time_array()
            sample_rate = info.sample_rate_hz
            self._log(f"采样率: {sample_rate:.1f} Hz, 时长: {info.total_time:.1f}s")

            self.df = self.loader.load_all(include_panel=include_panel)
            self._log(f"数据加载: {self.df.shape[0]}行 x {self.df.shape[1]}列")

            self._set_status("正在预处理...")
            self._set_progress(30)
            self.df = preprocess_dataframe(self.df, config.preprocess, sample_rate)
            self._log("预处理完成")

            # 更新图表通道下拉框
            self._update_channel_combo()

            results = {}

            # 2. 极限载荷分析（通用，支持EOG/NTM等所有风况）
            if mode in ("extreme", "eog", "both"):
                self._set_status(f"正在进行极限载荷分析 ({info.condition_type})...")
                self._set_progress(45)
                slicer = ConditionSlicer(self.loader)
                cond_class = slicer.classify_condition()
                self._log(f"工况分类: {info.condition_type} -> {cond_class}")
                extreme_segments = slicer.detect_extreme_segments()
                if extreme_segments:
                    self._log(f"检测到极值段: {len(extreme_segments)}个")
                else:
                    self._log("使用全量数据进行极值统计")
                extreme_results = run_extreme_analysis(
                    self.df, self.mapper,
                    extreme_segments=extreme_segments if extreme_segments else None,
                    condition_type=info.condition_type,
                )
                results["extreme"] = extreme_results
                results["eog"] = extreme_results  # 向后兼容
                self._log(f"极限载荷分析完成, 最不利叶片: "
                          f"{extreme_results['wheel_results'].get('critical_blade', 'N/A')}")

            # 3. 疲劳分析
            if mode in ("fatigue", "both"):
                self._set_status("正在进行疲劳分析...")
                self._set_progress(65)
                fatigue_results = run_fatigue_analysis(self.df, self.mapper, info.total_time)
                results["fatigue"] = fatigue_results
                self._log(f"疲劳分析完成, 最大损伤: "
                          f"{fatigue_results['wheel_fatigue'].get('max_blade_damage', 0):.2e}")

            # 4. 叶片合成
            self._set_status("正在进行叶片合成...")
            self._set_progress(80)
            synth_results = synthesize_all_blades(self.df, self.mapper)
            results["synthesis"] = synth_results
            self._log("叶片合成完成")

            self.results = results
            self._set_progress(95)

            # 5. 更新UI
            self.root.after(0, self._update_results_ui)

            self._set_status("分析完成！")
            self._set_progress(100)
            self._log("=" * 40)
            self._log("全部分析完成！")

        except Exception as e:
            self._log(f"错误: {e}")
            self._log(traceback.format_exc())
            self._set_status(f"分析失败: {e}")
            messagebox.showerror("分析错误", f"{e}\n\n详见日志区。")
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.run_btn.config(state="normal", text="▶ 开始分析"))

    def _update_results_ui(self):
        """更新结果UI（主线程）"""
        if not self.results:
            return

        # 概览卡片
        self._update_overview_cards()

        # EOG
        if "eog" in self.results:
            self._update_eog_tab()

        # 疲劳
        if "fatigue" in self.results:
            self._update_fatigue_tab()

        # 合成
        if "synthesis" in self.results:
            self._update_synth_tab()

        self.export_btn.config(state="normal")

    def _update_overview_cards(self):
        """更新概览卡片"""
        gc = config.global_ch
        if self.df is not None:
            if gc.inst_torque_col in self.df.columns:
                t_mean = self.df[gc.inst_torque_col].mean()
                self.cards["torque"].config(text=f"{t_mean:.0f} Nm")
            if gc.inst_thrust_col in self.df.columns:
                f_mean = self.df[gc.inst_thrust_col].mean()
                self.cards["thrust"].config(text=f"{f_mean:.0f} N")
            if gc.inst_power_col in self.df.columns:
                p_mean = self.df[gc.inst_power_col].mean() / 1000
                self.cards["power"].config(text=f"{p_mean:.1f} kW")

        if "eog" in self.results:
            critical = self.results["eog"]["wheel_results"].get("critical_blade", "-")
            self.cards["critical"].config(text=critical)

    def _update_eog_tab(self):
        """更新EOG标签页"""
        eog = self.results["eog"]
        df = eog["summary_df"]
        if df.empty:
            return

        # 表格
        cols = list(df.columns)
        self.eog_tree["columns"] = cols
        for col in cols:
            self.eog_tree.heading(col, text=col)
            width = max(80, min(140, len(col) * 8))
            self.eog_tree.column(col, width=width, anchor="center")
        self.eog_tree.delete(*self.eog_tree.get_children())
        for _, row in df.iterrows():
            values = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
            self.eog_tree.insert("", "end", values=values)

        # 柱状图
        self.eog_fig.clear()
        ax = self.eog_fig.add_subplot(111)
        if "combined_peak_value" in df.columns:
            blades = df["blade_id"].values
            values = df["combined_peak_value"].values
            colors = [COLOR_DANGER if v == max(values) else COLOR_PRIMARY for v in values]
            bars = ax.bar(blades, values, color=colors, alpha=0.85)
            ax.set_ylabel("合成载荷峰值 [N]", fontsize=9)
            ax.set_title("各叶片极限载荷合成载荷峰值", fontsize=10)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        self.eog_fig.tight_layout()
        self.eog_canvas.draw()

        # 整机载荷
        wheel = eog["wheel_results"]
        self.eog_global_text.config(state="normal")
        self.eog_global_text.delete("1.0", "end")
        lines = []
        for key in ["total_force_stats", "overturning_moment_stats",
                    "thrust_x_stats", "moment_z_stats"]:
            if key in wheel:
                stats = wheel[key]
                lines.append(f"{key}: max={stats.get('max', 0):.1f}, "
                             f"mean={stats.get('mean', 0):.1f}, "
                             f"std={stats.get('std', 0):.1f}")
        if "critical_blade" in wheel:
            lines.append(f"最不利叶片: {wheel['critical_blade']} "
                         f"(峰值={wheel.get('critical_blade_load', 0):.1f} N)")
        if "combined_peak_imbalance" in wheel:
            lines.append(f"叶片间峰值不平衡度: {wheel['combined_peak_imbalance']:.4f}")
        self.eog_global_text.insert("1.0", "\n".join(lines))
        self.eog_global_text.config(state="disabled")

    def _update_fatigue_tab(self):
        """更新疲劳标签页"""
        fat = self.results["fatigue"]
        df = fat["summary_df"]
        if df.empty:
            return

        cols = list(df.columns)
        self.fatigue_tree["columns"] = cols
        for col in cols:
            self.fatigue_tree.heading(col, text=col)
            width = max(80, min(130, len(col) * 8))
            self.fatigue_tree.column(col, width=width, anchor="center")
        self.fatigue_tree.delete(*self.fatigue_tree.get_children())
        for _, row in df.iterrows():
            values = [f"{v:.2e}" if isinstance(v, float) and abs(v) < 0.01
                      else f"{v:.2f}" if isinstance(v, float) else str(v)
                      for v in row]
            self.fatigue_tree.insert("", "end", values=values)

        # 损伤对比图
        self.fatigue_fig.clear()
        ax = self.fatigue_fig.add_subplot(111)
        if "combined_load_damage" in df.columns:
            blades = df["blade_id"].values
            values = df["combined_load_damage"].values
            colors = [COLOR_DANGER if v == max(values) else COLOR_ACCENT for v in values]
            bars = ax.bar(blades, values, color=colors, alpha=0.85)
            ax.set_ylabel("Miner Damage", fontsize=9)
            ax.set_title("各叶片疲劳损伤 (合成载荷)", fontsize=10)
            ax.set_yscale("log")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2e}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        self.fatigue_fig.tight_layout()
        self.fatigue_canvas.draw()

        # 详情
        self.fatigue_detail_text.config(state="normal")
        self.fatigue_detail_text.delete("1.0", "end")
        wf = fat["wheel_fatigue"]
        lines = [
            f"整机最大叶片损伤: {wf.get('max_blade_damage', 0):.4e}",
            f"整机平均叶片损伤: {wf.get('mean_blade_damage', 0):.4e}",
            f"损伤不平衡度: {wf.get('damage_imbalance', 0):.4f}",
            f"S-N曲线参数: m={config.fatigue.sn_m}, logA={config.fatigue.sn_log_a}",
        ]
        self.fatigue_detail_text.insert("1.0", "\n".join(lines))
        self.fatigue_detail_text.config(state="disabled")

    def _update_synth_tab(self):
        """更新叶片合成标签页"""
        synth = self.results["synthesis"]
        self.synth_text.config(state="normal")
        self.synth_text.delete("1.0", "end")
        lines = []
        for key in ["imbalance_normal", "imbalance_tangential"]:
            if key in synth:
                imb = synth[key]
                lines.append(f"[{key}]")
                lines.append(f"  不平衡度(均值): {imb.get('imbalance_mean', 0):.4f}")
                lines.append(f"  不平衡度(最大): {imb.get('imbalance_max', 0):.4f}")
                if "blade_statistics" in imb:
                    for bid, stats in imb["blade_statistics"].items():
                        lines.append(f"  {bid}: mean={stats.get('mean', 0):.1f}, "
                                     f"max={stats.get('max', 0):.1f}")
        if "pulsation" in synth:
            p = synth["pulsation"]
            lines.append(f"[旋转脉动] 平均转速={p.get('avg_rpm', 0):.1f}rpm, "
                         f"脉动率={p.get('pulsation_ratio', 0):.4f}")
        self.synth_text.insert("1.0", "\n".join(lines))
        self.synth_text.config(state="disabled")

        # 合成载荷时序图
        self.synth_fig.clear()
        if "synthesis_df" in synth and not synth["synthesis_df"].empty:
            sdf = synth["synthesis_df"]
            t = sdf["time"].values
            ax1 = self.synth_fig.add_subplot(211)
            if "global_F_mag" in sdf.columns:
                ax1.plot(t, sdf["global_F_mag"].values, color=COLOR_PRIMARY, linewidth=0.8)
                ax1.set_ylabel("合力 [N]", fontsize=9)
                ax1.set_title("整轮合成载荷时序", fontsize=10)
            ax1.tick_params(labelsize=8)
            ax1.grid(alpha=0.3)

            ax2 = self.synth_fig.add_subplot(212)
            if "overturning_moment" in sdf.columns:
                ax2.plot(t, sdf["overturning_moment"].values, color=COLOR_DANGER, linewidth=0.8)
                ax2.set_ylabel("倾覆弯矩 [Nm]", fontsize=9)
            ax2.set_xlabel("Time [s]", fontsize=9)
            ax2.tick_params(labelsize=8)
            ax2.grid(alpha=0.3)
        self.synth_fig.tight_layout()
        self.synth_canvas.draw()

    def _update_channel_combo(self):
        """更新图表通道下拉框"""
        if self.df is None:
            return
        channels = [c for c in self.df.columns if not c.startswith("PAN_")]
        self.chart_combo["values"] = channels
        if channels:
            default = "Momentary_Aerodynamic_Torque_[Nm]"
            self.chart_channel.set(default if default in channels else channels[0])

    def _draw_chart(self):
        """绘制时序图表"""
        if self.df is None:
            messagebox.showinfo("提示", "请先运行分析！")
            return
        channel = self.chart_channel.get()
        if channel not in self.df.columns:
            return

        self.chart_fig.clear()
        ax = self.chart_fig.add_subplot(111)
        t = self.df[config.global_ch.time_col].values
        vals = self.df[channel].values

        ax.plot(t, vals, color=COLOR_PRIMARY, linewidth=0.7, alpha=0.9)
        ax.axhline(y=np.mean(vals), color=COLOR_ACCENT, linestyle="--",
                   linewidth=1, label=f"均值={np.mean(vals):.1f}")
        ax.fill_between(t, vals, alpha=0.1, color=COLOR_PRIMARY)

        # 标注极值
        idx_max = np.argmax(vals)
        idx_min = np.argmin(vals)
        ax.annotate(f"max={vals[idx_max]:.1f}", xy=(t[idx_max], vals[idx_max]),
                    fontsize=8, color=COLOR_DANGER,
                    arrowprops=dict(arrowstyle="->", color=COLOR_DANGER, lw=0.8))
        ax.annotate(f"min={vals[idx_min]:.1f}", xy=(t[idx_min], vals[idx_min]),
                    fontsize=8, color=COLOR_SUCCESS,
                    arrowprops=dict(arrowstyle="->", color=COLOR_SUCCESS, lw=0.8))

        ax.set_xlabel("Time [s]", fontsize=10)
        ax.set_ylabel(channel, fontsize=9)
        ax.set_title(f"{channel} 时序曲线", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
        self.chart_fig.tight_layout()
        self.chart_canvas.draw()
        self._log(f"绘制图表: {channel}")

    def _export_results(self):
        """导出Excel报表"""
        if not self.results:
            return
        path = filedialog.asksaveasfilename(
            title="导出分析结果",
            defaultextension=".xlsx",
            initialfile=f"载荷分析结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            filetypes=[("Excel文件", "*.xlsx")]
        )
        if not path:
            return
        try:
            sheets = {}
            ext_data = self.results.get("extreme") or self.results.get("eog")
            if ext_data:
                if not ext_data["summary_df"].empty:
                    sheets["极限载荷叶片极值"] = ext_data["summary_df"]
            if "fatigue" in self.results:
                fat = self.results["fatigue"]
                if not fat["summary_df"].empty:
                    sheets["疲劳损伤汇总"] = fat["summary_df"]
                for bid, bres in fat.get("blade_results", {}).items():
                    for ch, chdata in bres.items():
                        if ch.startswith("panel_"):
                            continue
                        if "spectrum_df" in chdata:
                            sheets[f"{bid}_{ch}_谱"] = chdata["spectrum_df"]
            if "synthesis" in self.results:
                synth = self.results["synthesis"]
                if "synthesis_df" in synth and not synth["synthesis_df"].empty:
                    sheets["合成载荷时序"] = synth["synthesis_df"]

            export_to_excel(sheets, path)
            self._log(f"已导出: {path}")
            messagebox.showinfo("导出成功", f"结果已导出到:\n{path}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    # ============================================================
    # 工具方法
    # ============================================================
    def _log(self, msg):
        """写日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _set_status(self, text):
        self.status_text.set(text)

    def _set_progress(self, val):
        self.progress_val.set(val)


def main():
    root = tk.Tk()
    # Windows高DPI适配
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = LoadAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

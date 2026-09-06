# 垂直轴风轮(VAWT)载荷数据处理系统

适配 **QBlade 仿真输出大文件**，支持 **EOG 极端阵风分析** + **疲劳载荷分析**，多叶片解耦与合成。

## 功能特性

- **大文件低内存处理**：分块惰性读取，GB级文件不OOM
- **QBlade格式自动解析**：2行注释头 + 1526列 + 科学计数法 + 制表符分隔
- **多叶片解耦分析**：单叶片独立极值/疲劳计算，最不利叶片筛查
- **整轮耦合合成**：多叶片矢量合成、不平衡度、倾覆弯矩、旋转脉动
- **面板级分布载荷**：20面板展向载荷云图、最危险面板定位
- **EOG智能切片**：风速变化率自动检测阵风段
- **分块雨流计数**：超长时序疲劳分析无精度损失
- **Miner损伤+DEL**：S-N曲线参数可配置，支持Goodman修正

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 图形界面（推荐）

```bash
# 一键启动（Windows，自动检查/安装依赖）
双击 start.bat

# 或手动启动 Web GUI（浏览器中操作，无需Tkinter，推荐）
python web_gui.py
# 自动打开浏览器 http://127.0.0.1:8080

# 桌面GUI（Tkinter，需Python自带Tcl/Tk）
python gui.py
```

Web GUI功能：文件预览、分析模式选择、参数配置、实时进度、EOG/疲劳/合成/时序4个标签页、ECharts交互式图表、**载荷数据导出（Excel/CSV/TXT）**、**分析结果报告导出（PDF/Word）**。

分析完成后，Web GUI 会显示数据质量与坐标校验摘要，包括时间步长、NaN/Inf、
风速通道、全局力/力矩通道、推力符号相关性和叶片方位角跳变。符号相关性只是
自动提示，最终正负号仍需根据风轮坐标系定义确认。

> Windows 启动时如果出现 `OpenBLAS error: Memory allocation still failed`，
> 优先使用 `start.bat` 启动；程序默认将 BLAS/OpenMP 线程限制为 1，避免科学计算库初始化时申请过多线程内存。
> 也可以在 PowerShell 中先执行 `$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"`，
> 再运行 `python .\web_gui.py`。如确实需要提高并发，可先设置 `$env:VAWT_BLAS_THREADS="2"`；如果仍失败，请检查当前环境的 NumPy/Pandas/Matplotlib 是否安装在同一个 Python 中。

### 3. 打包迁移到其他电脑

```bash
# 1. 将整个 VAWT_Load_Processing 文件夹拷贝到目标电脑
# 2. 目标电脑安装 Python 3.10+（安装时勾选 Add Python to PATH）
# 3. 安装依赖：
pip install -r requirements.txt
# 4. 启动：
python web_gui.py
# 或直接双击 start.bat（自动完成步骤3-4）
```

### 3. 命令行运行

```bash
# 完整分析（EOG + 疲劳）
python main.py --input ../load_data_sample.txt --mode both

# 仅EOG极值分析（不加载面板级数据，省内存）
python main.py --input ../load_data_sample.txt --mode eog --no-panel

# 仅疲劳分析（大文件分块模式）
python main.py --input ../load_data_sample.txt --mode fatigue --chunked
```

### 4. 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input/-i` | QBlade载荷文件路径（必填） | - |
| `--mode/-m` | 分析模式: eog/fatigue/both | both |
| `--no-panel` | 不加载面板级分布载荷（1526列→约70列） | False |
| `--chunked` | 分块模式（>100MB文件推荐） | False |
| `--chunksize` | 分块行数 | 10000 |
| `--output/-o` | 输出目录 | ./output |
| `--log-level` | 日志级别 | INFO |

## 项目结构

```
VAWT_Load_Processing/
├── config.py           # 全局配置（叶片数、通道、S-N曲线、滤波参数等）
├── bigfile_io.py       # 大文件IO：QBlade解析、分块读取、通道映射、工况切片
├── preprocess.py       # 预处理：缺失值、异常值、低通滤波、瞬态峰值保护
├── eog_analysis.py     # EOG分析：单叶片极值、整轮载荷、不平衡度、最不利叶片
├── fatigue_analysis.py # 疲劳分析：雨流计数、Miner损伤、DEL、分块累积
├── blade_synthesis.py  # 叶片合成：矢量合成、倾覆弯矩、旋转脉动、展向积分
├── visualize.py        # 可视化：时序图、对比柱状图、载荷谱、云图
├── utils.py            # 工具函数：统计、内存监控、数据导出
├── main.py             # 主入口（命令行）
├── gui.py              # 桌面GUI（Tkinter）
├── web_gui.py          # Web GUI（http.server + ECharts，推荐）
├── check_thrust_torque.py  # 推力/扭矩专项分析脚本
├── requirements.txt    # Python依赖
├── README.md           # 本文件
└── output/             # 输出目录
    ├── blade_single/   # 单叶片独立结果
    ├── wheel_total/    # 整轮合成结果
    ├── compare/        # 多叶片对比报表
    └── figures/        # 可视化图表
```

## 适配的QBlade数据格式

```
行0: Results Output File created with QBlade on 31.08.2026 at 09:10:56  (注释)
行1: EOG: zhognneng20 Turb                                          (工况标识)
行2: Time_[s]  Timestep_[s]  Azimuthal_Position_Blade_1_[deg]  ...  (表头, 制表符分隔)
行3+: 0.00000E+00  8.33333E-03  2.69995E+02  ...                     (数据, 科学计数法)
```

- 总列数：1526列（含3叶片×20面板的气动/载荷分布）
- 关键载荷通道：
  - `Total_Tangential_Load_Blade_N_[N]`：叶片总切向载荷
  - `Total_Normal_Load_Blade_N_[N]`：叶片总法向载荷
  - `Normal_Force_Blade_N_PAN_M_[N/m]`：面板法向力分布
  - `Tangential_Force_Blade_N_PAN_M_[N/m]`：面板切向力分布
  - `Pitching_Moment_Blade_N_PAN_M_[Nm/m]`：面板俯仰力矩分布
  - `Momentary_Thrust/Moment_in_X/Y/Z_g_Direction`：整机力/力矩

## 配置说明

所有参数集中在 `config.py`，关键配置：

### 叶片配置
```python
BladeConfig.num_blades = None      # None=自动检测
BladeConfig.num_panels = 20        # 每叶片面板数
```

### 大文件读取
```python
BigFileConfig.chunksize = 10000    # 每块行数
BigFileConfig.memory_limit_mb = 512 # 内存上限
```

### 疲劳S-N曲线
```python
FatigueConfig.sn_m = 3.0           # S-N斜率倒数
FatigueConfig.sn_log_a = 12.0      # S-N截距(log10)
FatigueConfig.design_life_sec = 630720000  # 20年
```

### 预处理
```python
PreprocessConfig.filter_cutoff_hz = 5.0   # 低通滤波截止频率
PreprocessConfig.preserve_transient_peaks = True  # EOG峰值保护
```

### IEC DLC 批处理

批处理支持从文件名自动识别 `EOG/NTM/EWS/ECD/EWM/NWP/STEADY/TURB`
等工况类型，并按 `config.batch.dlc_map` 映射到默认 DLC 编号。使用清单文件时，
可提供以下字段：

```csv
file,condition_type,dlc,label,weight,extreme
run_eog.txt,EOG,1.3,EOG设计工况,1,true
run_ntm.txt,NTM,1.2,NTM运行工况,2,false
```

批处理结果包含叶片、塔顶和全局载荷的跨工况极限/疲劳包络，同时保留每个
DLC 的来源工况、发生时刻、权重和失败原因。

## 输出说明

### Excel报表
- `EOG_极值分析结果.xlsx`：叶片极值汇总、整机载荷、各叶片详细统计
- `疲劳分析结果.xlsx`：叶片疲劳汇总、各通道载荷谱、面板疲劳排名
- `叶片合成结果.xlsx`：合成载荷时序、不平衡度统计、旋转脉动分析

### 可视化图表（output/figures/）
- `all_blades_total_normal_timeseries.png`：多叶片法向载荷对比
- `eog_combined_peak_value_comparison.png`：EOG极值对比
- `fatigue_combined_load_damage_comparison.png`：疲劳损伤对比
- `Blade_N_normal_force_heatmap.png`：面板载荷展向-时间云图
- `wheel_synthesis_timeseries.png`：整轮合成载荷时序

## 扩展开发

### 添加新的载荷通道
在 `config.py` 的 `BladeConfig` 中添加列名模式，`ChannelMapper` 会自动匹配。

### 自定义疲劳S-N曲线
修改 `FatigueConfig.sn_m` 和 `sn_log_a`，或在 `fatigue_analysis.py` 的 `miner_damage()` 中传入自定义参数。

### 新增分析模块
1. 在项目目录新建 `.py` 文件
2. 在 `main.py` 的 `run_pipeline()` 中添加调用步骤
3. 在 `export_results()` 中添加结果导出

## 注意事项

1. **内存优化**：1526列全量加载约需1-2GB内存，建议加 `--no-panel` 只加载总载荷
2. **分块模式**：>100MB文件建议加 `--chunked`，疲劳分析支持分块无精度损失；Web GUI 默认不加载面板级数据以降低内存占用
3. **EOG检测**：基于风速变化率，若仿真为稳态EOG（风速阶跃），可能检测不到明显阵风段，此时使用全量极值
4. **采样率**：自动从时间列计算，QBlade典型为120Hz（dt=8.33ms）
5. **叶片数**：自动从列名检测 `Blade_N`，支持任意叶片数
#   V A W T _ L o a d _ P r o c e s s i n g  
 
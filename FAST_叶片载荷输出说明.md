# OpenFAST / AeroDyn 输出文件中"叶片载荷"参数说明

> 适用范围：垂直轴风轮（VAWT）使用 OpenFAST 的 **AeroDyn 模块**单独计算并输出的 `.out` 时程文件。
> 对应附件：`ad_driver.out`（当前输出设置**未包含叶片载荷**）、`OutListParameters.xlsx`（变量字典，重点看 **AeroDyn** 子表）。
> 说明：本系统（`fast_io.py` / `fast_analysis.py`）已内置 AeroDyn 变量定义，输出文件**缺列也能继续分析已有的变量**。

---

## 一、为什么当前 `ad_driver.out` 算不了叶片载荷

对上传的 `ad_driver.out` 实测解析结果：

- 共 **188 列**，包含：`Time`、`Case`、`HWindSpeedX/Y/Z`、`ShearExp`、平台运动、`RotSpeed`、`BldPitch1~3`、
  `RtAeroPwr`（风轮功率）、`RtSpeed`、`RtArea`、`RtAeroCt/Cq/Cp`、`RtTSR`，
  以及 3 叶片 × 18 节点的气动参数 `AB1NxxxVindx / Alpha / Cl`。
- **缺失**的关键变量（系统自动检测并提示）：
  - 风轮整体力/力矩：`RtAeroFxh / Fyh / Fzh`（推力分量）、`RtAeroMxh / Myh / Mzh`（力矩分量）；
  - 叶片节点载荷：`B*N*Fx / Fy / Fn / Ft / Mm`。

当前文件只有"气动系数/诱导速度/攻角"（用于气动分析），**没有分布力载荷**，因此无法直接积分得到叶片力与弯矩。

---

## 二、要计算叶片载荷，需在 AeroDyn `OutList` 中补充输出的变量

按 OpenFAST 变量命名规则（`OutListParameters.xlsx` → **AeroDyn** 子表，`B{b}N{n}{后缀}`，
其中 `b`=叶片号、`n`=节点号），建议在 `AeroDyn` 的 `OutList` 中输出以下**分布力（N/m）**变量：

| 变量模板（叶片1节点1示例） | 单位 | 含义 | 用途 |
|---|---|---|---|
| `B1N1Fx` | N/m | 法向力（垂直于叶片平面）分布载荷 | 积分 → 叶片法向总力 / 根部挥舞弯矩 |
| `B1N1Fy` | N/m | 切向力（叶片平面内）分布载荷 | 积分 → 叶片切向总力（驱动扭矩贡献） |
| `B1N1Fn` | N/m | 法向力（垂直弦向）分布载荷 | 气动法向分量（与 Cx 对应） |
| `B1N1Ft` | N/m | 切向力（沿弦向）分布载荷 | 气动切向分量（与 Cy 对应） |
| `B1N1Fl` | N/m | 升力分布载荷 | 升力分量 |
| `B1N1Fd` | N/m | 阻力分布载荷 | 阻力分量 |
| `B1N1Mm` | N-m/m | 俯仰力矩分布载荷 | 积分 → 叶片扭转载荷 |

> 对 VAWT 推荐至少输出 **`B*N*Fx`（法向）+ `B*N*Fy`（切向）** 两组，
> 它们直接对应 QBlade 输出里的叶片法向 / 切向总载荷，可用于极限与疲劳分析。

### 风轮整体载荷（算塔顶 / 整机载荷）

| 变量 | 单位 | 含义 |
|---|---|---|
| `RtAeroFxh` | N | 风轮总气动力 x 分量（轮毂坐标系） |
| `RtAeroFyh` | N | 风轮总气动力 y 分量 |
| `RtAeroFzh` | N | 风轮总气动力 z 分量 |
| `RtAeroMxh` | N-m | 风轮总气动力矩 x 分量 |
| `RtAeroMyh` | N-m | 风轮总气动力矩 y 分量 |
| `RtAeroMzh` | N-m | 风轮总气动力矩 z 分量（绕旋转轴的扭矩） |
| `RtAeroPwr` | W | 风轮气动功率 |
| `RtSpeed` | rpm | 风轮转速 |
| `RtTSR` | - | 叶尖速比 |

> 与 QBlade 系统通道的对应关系（本系统已内置映射）：
> - **风轮推力** ≈ `sqrt(RtAeroFxh² + RtAeroFyh²)`（VAWT 水平面内合力，即法向推力）；
> - **风轮扭矩** ≈ `RtAeroMzh`（绕垂直旋转轴的力矩）；若未输出 `RtAeroMzh`，系统自动用 `RtAeroPwr / ω` 反算（ω = RtSpeed × 2π/60）；
> - **风轮功率** ≈ `RtAeroPwr`。

---

## 三、从分布力到"叶片总载荷"的计算方法

`B*N*Fx / Fy` 是**单位长度分布力**（N/m），要得到叶片总载荷需沿叶片展向积分：

```
叶片法向总力  F_N = ∫₀^L  f_x(r) dr   ≈  Σᵢ  f_x(rᵢ) · Δrᵢ
叶片切向总力  F_T = ∫₀^L  f_y(r) dr   ≈  Σᵢ  f_y(rᵢ) · Δrᵢ
叶片根部挥舞弯矩 M_B = ∫₀^L  f_x(r) · r dr   （r 为距叶片固定点距离）
```

**前提**：需要知道每个节点的**展向位置 / 节点间距**（Δr）。
AeroDyn 输出文件本身不含节点几何坐标，因此：
1. 若节点等距，可在程序内按"叶片长度 / (节点数-1)"近似；
2. 更准确做法：读取 AeroDyn 输入文件（`*_AD.dat`）中的节点位置，或在程序配置中填写各节点展向坐标。

> 本系统当前对叶片部分输出**节点级载荷极值表**（各节点 max/min/mean/std，无需几何即可给出）。
> 若要输出"叶片合成总载荷 / 叶片疲劳损伤"，需在 FAST 侧补充输出上述分布力，并提供节点几何。

---

## 四、AeroDyn_driver 命名与标准 OpenFAST 命名的对应

`ad_driver.out`（AeroDyn_driver 单模块运行）与完整 OpenFAST（`FAST` 主程序）的节点列命名略有差异，本系统已做**命名归一化**兼容两套：

| AeroDyn_driver（当前文件） | 标准 OpenFAST | 说明 |
|---|---|---|
| `AB1N001Vindx` | `B1N1Vindx` | 叶片1节点1 轴向诱导速度 |
| `AB1N001Alpha` | `B1N1Alpha` | 叶片1节点1 攻角 |
| `AB1N001Cl` | `B1N1Cl` | 叶片1节点1 升力系数 |
| `AB3N018Fy`（若输出） | `B3N18Fy` | 叶片3节点18 切向分布力 |

即：去掉 `A` 前缀、节点号去掉前导零，即可与 `OutListParameters.xlsx` 的标准名对应。
系统在分析时对两种命名都能识别。

---

## 五、推荐的 OutList 配置示例（AeroDyn 模块）

```
OutList
RtAeroFxh RtAeroFyh RtAeroFzh
RtAeroMxh RtAeroMyh RtAeroMzh
RtAeroPwr RtSpeed RtTSR
B1N1Fx B1N1Fy B1N2Fx B1N2Fy ... B1N18Fx B1N18Fy
B2N1Fx B2N1Fy ... B2N18Fx B2N18Fy
B3N1Fx B3N1Fy ... B3N18Fx B3N18Fy
```

输出后，本系统即可在 **FAST 载荷** 页提供：风轮推力/扭矩/功率极值、时序图、叶片节点载荷极值表；
后续补充节点几何后，可进一步扩展叶片合成总载荷与叶片疲劳损伤分析。

---

*说明：以上变量名、单位与描述均摘自 `OutListParameters.xlsx` 的 **AeroDyn** 子表，并与实测 `ad_driver.out` 列核对一致。*

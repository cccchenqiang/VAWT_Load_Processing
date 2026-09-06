"""风轮推力与扭矩输出快速分析"""
import numpy as np
import pandas as pd

fpath = r"D:\1worksfiles\py\laoshan\ultrapost\load_data_sample.txt"

# 读取表头
with open(fpath, encoding="utf-8") as f:
    lines = f.readlines()
header = lines[2].strip().split("\t")

cols_needed = [
    "Time_[s]",
    "Momentary_Aerodynamic_Torque_[Nm]",
    "Momentary_Aerodynamic_Thrust_[N]",
    "Aerodynamic_Torque_[Nm]",
    "Aerodynamic_Thrust_[N]",
    "Momentary_Aerodynamic_Power_[W]",
    "Aerodynamic_Power_[W]",
    "Rotational_Speed_[rpm]",
    "Abs_Meas._Wind_Vel._at_Hub_[m/s]",
    "Power_Coefficient_[-]",
    "Torque_Coefficient_[-]",
    "Thrust_Coefficient_[-]",
]

df = pd.read_csv(
    fpath, sep="\t", skiprows=3, header=None,
    names=header, usecols=cols_needed,
    encoding="utf-8", dtype=np.float64,
)

print("=" * 70)
print("风轮推力与扭矩输出分析")
print("=" * 70)
print(f"数据时长: {df['Time_[s]'].iloc[-1]:.2f} s, 采样点数: {len(df)}")
print(f"平均转速: {df['Rotational_Speed_[rpm]'].mean():.2f} rpm")
print(f"平均风速: {df['Abs_Meas._Wind_Vel._at_Hub_[m/s]'].mean():.2f} m/s")
print()

# 瞬时扭矩
t_inst = df["Momentary_Aerodynamic_Torque_[Nm]"]
print("--- 瞬时气动扭矩 Momentary_Aerodynamic_Torque [Nm] ---")
print(f"  最大值: {t_inst.max():.2f} Nm")
print(f"  最小值: {t_inst.min():.2f} Nm")
print(f"  均值:   {t_inst.mean():.2f} Nm")
print(f"  标准差: {t_inst.std():.2f} Nm")
print(f"  峰峰值: {t_inst.max() - t_inst.min():.2f} Nm")
print(f"  RMS:    {np.sqrt(np.mean(t_inst**2)):.2f} Nm")
print(f"  波动系数(std/mean): {t_inst.std() / t_inst.mean() * 100:.1f}%")

# 平均扭矩
t_avg = df["Aerodynamic_Torque_[Nm]"]
print()
print("--- 平均气动扭矩 Aerodynamic_Torque [Nm] (滑动平均) ---")
print(f"  最大值: {t_avg.max():.2f} Nm")
print(f"  最小值: {t_avg.min():.2f} Nm")
print(f"  均值:   {t_avg.mean():.2f} Nm")
print(f"  终值(稳态): {t_avg.iloc[-1]:.2f} Nm")

# 瞬时推力
f_inst = df["Momentary_Aerodynamic_Thrust_[N]"]
print()
print("--- 瞬时气动推力 Momentary_Aerodynamic_Thrust [N] ---")
print(f"  最大值: {f_inst.max():.2f} N")
print(f"  最小值: {f_inst.min():.2f} N")
print(f"  均值:   {f_inst.mean():.2f} N")
print(f"  标准差: {f_inst.std():.2f} N")
print(f"  峰峰值: {f_inst.max() - f_inst.min():.2f} N")
print(f"  RMS:    {np.sqrt(np.mean(f_inst**2)):.2f} N")
print(f"  波动系数(std/mean): {f_inst.std() / f_inst.mean() * 100:.1f}%")

# 平均推力
f_avg = df["Aerodynamic_Thrust_[N]"]
print()
print("--- 平均气动推力 Aerodynamic_Thrust [N] ---")
print(f"  最大值: {f_avg.max():.2f} N")
print(f"  最小值: {f_avg.min():.2f} N")
print(f"  均值:   {f_avg.mean():.2f} N")
print(f"  终值(稳态): {f_avg.iloc[-1]:.2f} N")

# 功率
p_inst = df["Momentary_Aerodynamic_Power_[W]"]
p_avg = df["Aerodynamic_Power_[W]"]
print()
print("--- 气动功率 Power ---")
print(f"  瞬时均值: {p_inst.mean():.2f} W = {p_inst.mean()/1000:.2f} kW")
print(f"  平均终值: {p_avg.iloc[-1]:.2f} W = {p_avg.iloc[-1]/1000:.2f} kW")
print(f"  瞬时峰值: {p_inst.max():.2f} W = {p_inst.max()/1000:.2f} kW")

# 系数
print()
print("--- 性能系数 ---")
print(f"  Cp (功率系数): 均值={df['Power_Coefficient_[-]'].mean():.4f}, 终值={df['Power_Coefficient_[-]'].iloc[-1]:.4f}")
print(f"  Ct (推力系数): 均值={df['Thrust_Coefficient_[-]'].mean():.4f}, 终值={df['Thrust_Coefficient_[-]'].iloc[-1]:.4f}")
print(f"  Cq (扭矩系数): 均值={df['Torque_Coefficient_[-]'].mean():.4f}, 终值={df['Torque_Coefficient_[-]'].iloc[-1]:.4f}")

# 校验 P = T * omega
omega = df["Rotational_Speed_[rpm]"].mean() * 2 * np.pi / 60
p_from_t = t_inst.mean() * omega
print()
print("--- 校验: P = T * omega ---")
print(f"  平均角速度 omega = {omega:.4f} rad/s")
print(f"  T_mean * omega = {p_from_t:.2f} W")
print(f"  P_inst均值     = {p_inst.mean():.2f} W")
print(f"  相对误差: {abs(p_from_t - p_inst.mean()) / p_inst.mean() * 100:.2f}%")

# 分段统计：前10s vs 后10s（EOG阵风通常在后半段）
print()
print("=" * 70)
print("分段对比: 0-10s (启动/稳态) vs 10-20s (EOG阵风段)")
print("=" * 70)
mask1 = df["Time_[s]"] < 10
mask2 = df["Time_[s]"] >= 10
for label, mask in [("0-10s", mask1), ("10-20s", mask2)]:
    print(f"\n[{label}]")
    print(f"  扭矩: 均值={t_inst[mask].mean():.1f} Nm, 峰值={t_inst[mask].max():.1f} Nm, 波动={t_inst[mask].std():.1f} Nm")
    print(f"  推力: 均值={f_inst[mask].mean():.1f} N,  峰值={f_inst[mask].max():.1f} N,  波动={f_inst[mask].std():.1f} N")
    print(f"  功率: 均值={p_inst[mask].mean()/1000:.2f} kW, 峰值={p_inst[mask].max()/1000:.2f} kW")
    print(f"  风速: 均值={df['Abs_Meas._Wind_Vel._at_Hub_[m/s]'][mask].mean():.2f} m/s, 峰值={df['Abs_Meas._Wind_Vel._at_Hub_[m/s]'][mask].max():.2f} m/s")

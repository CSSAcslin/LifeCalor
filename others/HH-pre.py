import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 霍奇金-赫胥黎模型的常数和参数
# 数值基于乌贼巨大轴突的实验数据
C_m = 1.0  # 膜电容 (uF/cm^2)

g_Na = 120.0  # 钠离子最大电导 (mS/cm^2)
g_K = 36.0  # 钾离子最大电导 (mS/cm^2)
g_L = 0.3  # 漏电最大电导 (mS/cm^2)

E_Na = 50.0  # 钠离子平衡电位 (mV)
E_K = -77.0  # 钾离子平衡电位 (mV)
E_L = -54.387  # 漏电平衡电位 (mV)


# 门控变量的电压依赖性速率常数 (α 和 β)
def alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))


def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))


def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


# 离子电流计算
def I_Na(V, m, h):
    return g_Na * m ** 3 * h * (V - E_Na)


def I_K(V, n):
    return g_K * n ** 4 * (V - E_K)


def I_L(V):
    return g_L * (V - E_L)


# 外部刺激电流
def I_ext(t):
    # 在 10ms 到 11ms 之间施加一个 10 uA/cm^2 的电流脉冲
    if 10.0 < t < 11.0:
        return 10.0
    return 0.0


# 描述HH模型微分方程组的函数
def hodgkin_huxley_model(y, t):
    V, n, m, h = y

    # 计算膜电位的变化率
    dVdt = (I_ext(t) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m

    # 计算门控变量的变化率
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h

    return [dVdt, dndt, dmdt, dhdt]


# 设置仿真的时间和初始条件
t = np.arange(0.0, 50.0, 0.01)  # 仿真时间从0到50ms，步长0.01ms

# 初始膜电位和门控变量值 (静息状态)
V0 = -65.0
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
y0 = [V0, n0, m0, h0]

# 使用scipy的odeint求解微分方程组
solution = odeint(hodgkin_huxley_model, y0, t)
V = solution[:, 0]

# 绘制膜电位随时间的变化
plt.figure(figsize=(12, 6))
plt.plot(t, V, label='Membrane Potential (V)')
plt.title('Hodgkin-Huxley Model Simulation of an Action Potential')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.grid(True)
plt.legend()
plt.show()
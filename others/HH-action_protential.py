import numpy as np
import matplotlib.pyplot as plt


def simulate_hh_neuron(duration_ms=100.0, dt=0.01, current_base=10.0, noise_sigma=5.0):
    """
    模拟霍奇金-赫胥黎神经元在一段时间内的活动。

    参数:
    - duration_ms: 模拟总时长 (毫秒)
    - dt: 时间步长 (毫秒)，建议 <= 0.01 以保证数值稳定性
    - current_base: 基础注入电流强度 (uA/cm^2)，控制平均发放频率
    - noise_sigma: 电流噪声的标准差，模拟突触背景噪声，使放电更真实不规则

    返回:
    - time: 时间数组
    - V: 膜电位数组
    - spikes: 动作电位发生的时刻列表 (简单阈值检测)
    """

    # --- 1. 初始化参数 ---
    num_steps = int(duration_ms / dt)
    time = np.linspace(0, duration_ms, num_steps)

    # HH 模型常数
    C_m = 1.0
    g_Na, E_Na = 120.0, 50.0
    g_K, E_K = 36.0, -77.0
    g_L, E_L = 0.3, -54.387

    # 初始化状态变量 (静息电位附近)
    V = -65.0
    m = 0.05
    h = 0.6
    n = 0.32

    # 预分配数组以提升速度
    V_trace = np.zeros(num_steps)

    # 生成带有随机噪声的输入电流 (Input Current = Base + Noise)
    # 这种噪声模拟了真实神经元接收到的不规则突触输入
    I_input = current_base + np.random.normal(0, noise_sigma, num_steps)

    # --- 2. 辅助函数 (由于需要高性能，直接内嵌在循环中或使用Numba加速更好，这里为了可读性保留函数形式) ---
    # 为了代码简洁和性能平衡，这里直接写出alpha/beta公式
    # 注意：为了防止除以0的错误，通常会在分母加极小值或处理特殊点，这里简化处理

    spikes = []
    spike_threshold = 0.0  # 用于检测发放时刻
    has_spiked = False

    # --- 3. 主循环 (使用欧拉法积分) ---
    for i in range(num_steps):
        # 记录当前电压
        V_trace[i] = V

        # 简单的尖峰检测逻辑
        if V > spike_threshold and not has_spiked:
            spikes.append(time[i])
            has_spiked = True
        if V < spike_threshold - 10:
            has_spiked = False

        # --- 计算速率常数 alpha 和 beta ---
        # 技巧：利用 numpy 的标量运算，虽然是在循环里，但比调用函数稍快

        # Alpha n, Beta n
        if abs(V + 55.0) < 1e-6:  # 防止除零
            alpha_n = 0.1
        else:
            alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
        beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

        # Alpha m, Beta m
        if abs(V + 40.0) < 1e-6:
            alpha_m = 1.0
        else:
            alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
        beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

        # Alpha h, Beta h
        alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

        # --- 更新门控变量 (m, h, n) ---
        # dx/dt = alpha * (1-x) - beta * x
        # x_new = x_old + dt * dx/dt
        n = n + dt * (alpha_n * (1.0 - n) - beta_n * n)
        m = m + dt * (alpha_m * (1.0 - m) - beta_m * m)
        h = h + dt * (alpha_h * (1.0 - h) - beta_h * h)

        # --- 更新膜电位 V ---
        # dV/dt = (I_ext - I_Na - I_K - I_L) / Cm
        I_Na_val = g_Na * m ** 3 * h * (V - E_Na)
        I_K_val = g_K * n ** 4 * (V - E_K)
        I_L_val = g_L * (V - E_L)

        V = V + dt * (I_input[i] - I_Na_val - I_K_val - I_L_val) / C_m

    return time, V_trace, spikes, I_input



if __name__ == '__main__':
    # ==========================================
    # 调用示例：模拟 500ms (0.5秒) 的神经活动
    # ==========================================

    # 场景 1: 较低的输入，偶尔发放 (模拟背景活动)
    t1, v1, spikes1, i1 = simulate_hh_neuron(duration_ms=500, current_base=5.0, noise_sigma=3.0)

    # 场景 2: 较高的输入，频繁发放 (模拟受刺激状态)
    t2, v2, spikes2, i2 = simulate_hh_neuron(duration_ms=500, current_base=7.0, noise_sigma=5.0)

    # 绘图
    plt.figure(figsize=(12, 10))

    # 图 1: 低频/随机发放
    plt.subplot(2, 1, 1)
    plt.plot(t1, v1, 'b-', lw=1)
    plt.title(f'Scenario 1: Low Input (Sparse Spiking) - {len(spikes1)} spikes')
    plt.ylabel('Membrane Potential (mV)')
    plt.grid(True, alpha=0.3)

    # 图 2: 高频发放
    plt.subplot(2, 1, 2)
    plt.plot(t2, v2, 'r-', lw=1)
    plt.title(f'Scenario 2: High Input (Frequent Spiking with Noise) - {len(spikes2)} spikes')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 输出一些统计信息
    print(f"场景 2 模拟时长: 500ms")
    print(f"发放脉冲数: {len(spikes2)}")
    print(f"平均发放频率: {len(spikes2) / 0.5} Hz")
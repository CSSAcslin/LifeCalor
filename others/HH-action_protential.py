import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
import pandas as pd
import pywt
import scipy.signal as signal
from scipy.fft import fft, fftfreq


def generate_ou_noise(n_steps, dt, tau=15.0, sigma=4.0, mean_input=3.0):
    """
    生成 Ornstein-Uhlenbeck (O-U) 有色噪声。
    这是模拟生物神经元突触背景噪声或膜电位波动的黄金标准。

    参数:
    - tau: 噪声的相关时间常数 (ms)，模拟通道状态的持续时间。
    - sigma: 噪声的标准差 (波动幅度)。
    - mean_input: 基础电流输入的均值。
    """
    # n_steps = len(t)
    # I_noise = np.zeros(n_steps)
    # I_noise[0] = mean_input
    #
    # # O-U 过程公式: dI = -(I - mean)/tau * dt + sigma * sqrt(2/tau) * dW
    # # dW 是维纳过程增量 (高斯随机数 * sqrt(dt))
    # sqrt_dt = np.sqrt(dt)
    # drift_factor = np.sqrt(2 / tau)
    #
    # for i in range(1, n_steps):
    #     # 这里的随机项模拟了环境的随机扰动
    #     dW = np.random.normal(0, 1) * sqrt_dt
    #     dI = -(I_noise[i - 1] - mean_input) / tau * dt + sigma * drift_factor * dW
    #     I_noise[i] = I_noise[i - 1] + dI

    I_noise = np.zeros(n_steps)
    I_noise[0] = mean_input

    sqrt_dt = np.sqrt(dt)
    drift_factor = np.sqrt(2 / tau)
    # 预生成所有随机数
    random_nums = np.random.normal(0, 1, n_steps)

    for i in range(1, n_steps):
        dI = -(I_noise[i - 1] - mean_input) / tau * dt + sigma * drift_factor * random_nums[i] * sqrt_dt
        I_noise[i] = I_noise[i - 1] + dI
    return I_noise


def generate_pulse_signal(
        total_duration_ms: float = 1000.0,
        time_resolution_ms: float = 0.01,
        pulse_period_ms: float = 700.0,
        pulse_width_ms: float = 5.0,
        amplitude_range: Tuple[float, float] = (5.0, 6.0),
        amplitude_step: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成周期性递减脉冲信号

    参数:
    total_duration_ms: 总时长(毫秒)
    time_resolution_ms: 时间分辨率(毫秒)
    pulse_period_ms: 脉冲周期(毫秒)
    pulse_width_ms: 脉冲宽度(毫秒)
    amplitude_range: 幅值范围(最小值, 最大值)
    amplitude_step: 幅值变化步长

    返回:
    time_array: 时间数组(毫秒)
    signal: 信号幅值数组
    """

    # 生成时间序列
    t = np.arange(0, total_duration_ms, time_resolution_ms)

    # 初始化信号数组
    signal = np.zeros_like(t)

    # 计算总点数
    n_points = len(t)

    # 计算每个脉冲的时间点索引
    pulse_period_points = int(pulse_period_ms / time_resolution_ms)
    pulse_width_points = int(pulse_width_ms / time_resolution_ms)

    # 计算幅值变化范围
    min_amp, max_amp = amplitude_range
    n_steps = int((max_amp - min_amp) / amplitude_step) + 1

    # 生成幅值序列(从0到10，再回到0，不断循环)
    amplitudes = []
    direction = -1  # 初始方向为递减
    current_amp = min_amp

    while len(amplitudes) < (total_duration_ms / pulse_period_ms) + 1:
        if direction == -1:  # 递减
            while current_amp >= min_amp - 1e-10:  # 考虑浮点误差
                amplitudes.append(current_amp)
                current_amp -= amplitude_step
            current_amp = min_amp
            direction = 1
        else:  # 递增
            while current_amp <= max_amp + 1e-10:  # 考虑浮点误差
                amplitudes.append(current_amp)
                current_amp += amplitude_step
            current_amp = max_amp
            direction = -1

    # 生成脉冲信号
    for i in range(0, n_points, pulse_period_points):
        pulse_idx = i // pulse_period_points

        if pulse_idx < len(amplitudes):
            amp = amplitudes[pulse_idx]

            # 设置脉冲区域的值
            start_idx = i
            end_idx = min(i + pulse_width_points, n_points)

            signal[start_idx:end_idx] = amp

    return signal


def simulate_pc12_spontaneous(duration_ms=1000, dt_sim=0.01, target_fs=2000):
    """
    模拟 PC12 细胞的自发动作电位。
    参数:
    - duration_ms: 模拟时长 (ms)
    - dt_sim: 物理模拟的步长，必须很小 (建议 0.01 ms)
    - target_fs: 目标采样率 (Hz)。
      例如 2000Hz 意味着每秒记录2000个点 (即每0.5ms记录一次)。
      对于动作电位分析，1000Hz-5000Hz 通常足够。
    """
    total_sim_steps = int(duration_ms / dt_sim)

    # 计算需要每隔多少步保存一次 (downsample_factor)
    # 目标时间间隔 dt_rec = 1000 / target_fs (ms)
    dt_rec = 1000.0 / target_fs
    skip_step = int(np.round(dt_rec / dt_sim))

    # 实际记录的数据长度
    n_record = int(total_sim_steps / skip_step)

    # 2. 初始化输出数组 (只分配需要记录的大小)
    t_record = np.linspace(0, duration_ms, n_record)
    V_record = np.zeros(n_record)

    # 3. 生成全分辨率的噪声 (物理层需要连续的噪声)
    I_full = generate_ou_noise(total_sim_steps, dt_sim,
                                         tau=10.0, sigma=1.5, mean_input=1.0)
    I_full = generate_pulse_signal(duration_ms, dt_sim)

    # HH 参数
    C_m = 1.0
    g_Na, E_Na = 120.0, 50.0
    g_K, E_K = 36.0, -77.0
    g_L, E_L = 0.3, -54.387

    # 初始状态
    V = -65.0
    m, h, n = 0.05, 0.6, 0.32

    # 4. 主循环
    rec_idx = 0  # 记录索引

    for i in range(total_sim_steps):
        # --- HH 物理计算 (高频) ---
        if abs(V + 55.0) < 1e-6:
            an = 0.1
        else:
            an = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
        bn = 0.125 * np.exp(-(V + 65.0) / 80.0)

        if abs(V + 40.0) < 1e-6:
            am = 1.0
        else:
            am = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
        bm = 4.0 * np.exp(-(V + 65.0) / 18.0)

        ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
        bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

        n += dt_sim * (an * (1.0 - n) - bn * n)
        m += dt_sim * (am * (1.0 - m) - bm * m)
        h += dt_sim * (ah * (1.0 - h) - bh * h)

        I_Na = g_Na * m ** 3 * h * (V - E_Na)
        I_K = g_K * n ** 4 * (V - E_K)
        I_L = g_L * (V - E_L)

        V += dt_sim * (I_full[i] - I_Na - I_K - I_L) / C_m

        # --- 数据记录 (低频) ---
        # 只有当步数是 skip_step 的倍数时才记录
        if i % skip_step == 0 and rec_idx < n_record:
            V_record[rec_idx] = V
            rec_idx += 1

    return t_record, V_record,np.arange(0,duration_ms,dt_sim), I_full

def quality_cwt(sig,fs,totalscales,wavelet):
    cparam = 2 * pywt.central_frequency(wavelet) * totalscales
    scales = cparam / np.arange(totalscales, 1, -1)
    coefficients, frequencies = pywt.cwt(sig, scales, wavelet, sampling_period=1.0 / fs)
    return np.abs(coefficients), frequencies

def analyze_signals(sig, fs, target_freq=150, wavelet = 'cmor1.5-1'):
    """
    对信号进行STFT和CWT变换，并提取目标频率幅值
    """
    N = len(sig)
    # xf = fft.rfftfreq(N, 1 / fs)
    # yf = fft.rfft(sig)
    # psd_fft = np.abs(yf) ** 2 / N  # 简单的功率谱估计


    # --- 1. STFT (短时傅里叶变换) ---
    # nperseg 决定了频率分辨率。fs=1500, nperseg=256 -> 分辨率约为 5.8Hz
    # window = signal.get_window(('gaussian', 128 / 6), 128, fftbins=False)
    window = 'hann'
    f_stft, t_stft, Zxx = signal.stft(sig, fs, window=window ,nperseg=128, noverlap=120, nfft= fs, return_onesided=True)

    # 计算功率谱密度 PSD (取模的平方)
    psd_stft = np.abs(Zxx) ** 2

    # 提取 150Hz 处的幅值
    # 找到频率数组 f_stft 中最接近 target_freq 的索引
    idx_stft = np.argmin(np.abs(f_stft - target_freq))
    # 提取该频率随时间变化的幅值 (取模)
    amp_trace_stft = np.abs(Zxx[idx_stft, :])

    # --- 2. CWT (连续小波变换) ---
    psd_cwt, freqs_cwt = quality_cwt(sig,fs,256,wavelet)
    target_freqs = np.linspace(target_freq - 1 // 2, target_freq + 1 // 2,
                               1)  # totalscales//4
    scales = pywt.frequency2scale(wavelet, target_freqs * 1.0 / fs)
    coefficients, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1.0 / fs)

    magnitude_avg = np.mean(np.abs(coefficients), axis=0) / 3.16 # 这个倍数是有公式能算出来的

    return {
        # 'fft': (extract_150hz_amplitude(sig,fs)),
        'stft': (t_stft, f_stft, psd_stft, amp_trace_stft),
        'cwt': (np.arange(len(sig)) / fs, freqs_cwt, psd_cwt, magnitude_avg)
    }


# def extract_150hz_amplitude(signal, sampling_rate, frame_count=50):
#     """
#     将时序信号分割成指定帧数，对每帧进行FFT，提取150Hz分量的幅值
#
#     参数:
#     ----------
#     signal : numpy.ndarray
#         一维时序信号数组
#     sampling_rate : float
#         采样率(Hz)
#     frame_count : int, optional
#         要分割的帧数，默认为50
#
#     返回:
#     ----------
#     numpy.ndarray
#         包含每帧150Hz分量幅值的数组，长度为frame_count
#     """
#
#     # 输入验证
#     if not isinstance(signal, np.ndarray):
#         raise ValueError("输入信号必须为numpy ndarray")
#     if signal.ndim != 1:
#         raise ValueError("输入信号必须为一维数组")
#     if len(signal) < frame_count:
#         raise ValueError("信号长度必须大于或等于帧数")
#     if sampling_rate <= 0:
#         raise ValueError("采样率必须大于0")
#
#     # 计算每帧的长度
#     total_length = len(signal)
#     frame_length = total_length // frame_count
#
#     # 如果信号长度不能被帧数整除，截断尾部多余的部分
#     signal = signal[:frame_length * frame_count]
#
#     # 目标频率
#     target_freq = 150.0  # Hz
#
#     # 初始化结果数组
#     amplitudes = np.zeros(frame_count)
#
#     for i in range(frame_count):
#         # 提取当前帧
#         start_idx = i * frame_length
#         end_idx = (i + 1) * frame_length
#         frame = signal[start_idx:end_idx]
#
#         # 计算FFT
#         # fft_result = fft(frame)
#
#         # 获取FFT的频率分量
#         freqs = fftfreq(frame_length, 1 / sampling_rate)
#
#         # 找到最接近150Hz的频率索引
#         # 取正频率部分
#         pos_freqs = freqs[:frame_length // 2]
#         # pos_fft = fft_result[:frame_length // 2]
#
#         # 找到最接近150Hz的频率索引
#         freq_idx = np.argmin(np.abs(pos_freqs - target_freq))
#
#         # 获取该频率对应的幅值（取模）
#         amplitude = np.abs(pos_fft[freq_idx])
#
#         # 由于FFT结果是对称的，需要处理幅值缩放
#         # 对于实际信号，需要乘以2（除了直流分量和Nyquist频率）
#         if 0 < freq_idx < len(pos_freqs) - 1:
#             amplitude = 2 * amplitude / frame_length
#         else:
#             amplitude = amplitude / frame_length
#
#         amplitudes[i] = amplitude
#
#     return np.arange(len(amplitudes))/10,amplitudes-2

def gen_sine_modulation(fs, duration, freq, amp=0.1):
    """
    生成叠加用的正弦波
    """
    total_samples = int(fs * duration)
    t = np.arange(total_samples) / fs
    return amp * np.sin(2 * np.pi * freq * t)

def gen_random_noise(fs, duration,amp , level=0.05):
    """
    生成高斯白噪声
    """
    total_samples = int(fs * duration)
    return np.random.normal(amp, level, total_samples)


def normalize_0_1(array):
    """归一化到 [0, 1] 范围"""
    array_min = array.min()
    array_max = array.max()

    # 处理全相同值的情况
    if array_max == array_min:
        return np.zeros_like(array, dtype=np.float32)

    return (array - array_min) / (array_max - array_min)


def normalize_neg1_1(array):
    """归一化到 [-1, 1] 范围"""
    array_min = array.min()
    array_max = array.max()
    array_abs_max = max(abs(array_min), abs(array_max))

    if array_abs_max == 0:
        return np.zeros_like(array, dtype=np.float32)

    return array / array_abs_max

if __name__ == '__main__':
    WAVELET = 'cmor1-1'
    target_freq = 60
    fs = 600
    duration = 5

    # base_sig = gen_random_noise(fs, duration, amp =1 , level=0.3)
    # outer_noise_sig = gen_random_noise(fs, duration, amp=200, level=1)
    sine_sig = gen_sine_modulation(fs, duration, target_freq, amp=600)
    # full_signal = base_sig * sine_sig + outer_noise_sig
    full_signal = sine_sig
    time, v_trace,t_input, i_input = simulate_pc12_spontaneous(duration_ms=5000, target_fs=fs)

    # plt.figure(figsize=(14, 8))
    #
    # # 绘制输入的波动电流
    # plt.subplot(2, 1, 1)
    # plt.plot(t_input, i_input, color='green', lw=1, alpha=0.7)
    # plt.title('Simulated Membrane Current Fluctuation (Ornstein-Uhlenbeck Process)', fontsize=12)
    # plt.ylabel(r'Input Current ($\mu A/cm^2$)')
    # # 画一条虚线表示大概的触发阈值 (这只是经验值，不是硬性界限)
    # plt.axhline(y=2.5, color='red', linestyle='--', alpha=0.5, label='Approx. Firing Threshold')
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.2)
    #
    # # 绘制产生的动作电位
    # plt.subplot(2, 1, 2)
    # plt.plot(time, v_trace, color='black', lw=1)
    # plt.title('Simulated PC12 Spontaneous Activity', fontsize=12)
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Membrane Potential (mV)')
    # plt.ylim(-80, 50)
    # plt.grid(True, alpha=0.2)
    #
    # plt.tight_layout()
    # plt.show()

    # after_modulate =  normalize_neg1_1(full_signal) + normalize_0_1(v_trace)*0.1

    # 设置通用参数
    fs = 600  # 采样频率 (Hz) - 与您的设置匹配
    duration = 5  # 信号时长 (秒)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # 时间向量

    # 1. 生成纯净的调制信号 (作为基准)
    fc = 90  # 载波频率 (Hz) - 调制频率
    fm = 5  # 调制信号频率 (Hz)
    carrier = np.cos(2 * np.pi * fc * t)  # 60Hz载波
    modulating = np.ones(int(fs * duration),dtype=np.float32)
    clean_modulated = carrier * modulating  # 调幅信号

    # 2. 情景1：调制信号受到瞬态脉冲扰动
    impulse_times = [1.43, 3.06,4.3,4.12]  # 脉冲发生时刻（秒）
    impulse_signal = clean_modulated.copy()
    for impulse_time in impulse_times:
        idx = np.argmin(np.abs(t - impulse_time))
        impulse_width = int(0.01 * fs)  # 脉冲宽度10ms
        # 高斯脉冲
        impulse = 2.0 * np.exp(-50 * (t[idx:idx + impulse_width] - t[idx]) ** 2)
        impulse_signal[idx:idx + impulse_width] += impulse
    time = np.linspace(0, duration, int(fs * duration), endpoint=False)
    after_modulate = impulse_signal

    results = analyze_signals(after_modulate, fs, target_freq=target_freq, wavelet=WAVELET)

    t_stft, f_stft, psd_stft, trace_stft = results['stft']
    t_cwt, f_cwt, psd_cwt, trace_cwt = results['cwt']
    # t_fft, f_fft = results['fft']

    # total_time = 5.0
    # t_full = np.arange(len(after_modulate)) / 1000

    # 创建5行1列的布局
    # sharex=False，因为 FFT 的 x轴是频率，不能和时间的 x轴共享
    fig, axes = plt.subplots(4, 1, figsize=(18, 16), constrained_layout=True)

    # === 1. 原始信号 (Time) ===
    ax1 = axes[0]
    ax1.plot(time, after_modulate, color='blue', alpha=1, linewidth=0.8, label='Final Signal')
    ax1.set_title('1. Time Domain Signal')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')

    # # === 2. FFT 频谱 (Frequency) - 这是一个频域图，X轴不同 ===
    # ax2 = axes[1]
    # ax2.semilogy(f_fft, psd_fft, color='darkorange', linewidth=1)
    # ax2.set_title('2. FFT Power Spectrum (Global)')
    # ax2.set_ylabel('Power (Log Scale)')
    # ax2.set_xlabel('Frequency (Hz)')
    # ax2.set_xlim(0, min(400,fs//2))  # 重点关注低频区
    # # 标记目标频率
    # ax2.axvline(target_freq, color='red', linestyle='--', alpha=0.6, label=f'Target {target_freq}Hz')
    # ax2.legend()

    # === 3. STFT 谱图 (Time-Freq) ===
    ax3 = axes[1]
    # 使用 shading='gouraud' 可以让图像更平滑，且视觉上对齐更准
    mesh_stft = ax3.pcolormesh(t_stft, f_stft, 10 * np.log10(psd_stft + 1e-10),
                               shading='gouraud', cmap='viridis')
    ax3.set_title('3. STFT Spectrogram')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_ylim(0, min(400, fs //2))
    ax3.axhline(target_freq, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(mesh_stft, ax=ax3, label='dB')

    # === 4. CWT 谱图 (Time-Freq) ===
    ax4 = axes[2]
    # CWT 的 x 轴直接使用 t_cwt (即 t_full)
    mesh_cwt = ax4.pcolormesh(t_cwt, f_cwt, psd_cwt, shading='auto', cmap='viridis')
    ax4.set_title('4. CWT Scalogram')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_ylim(1, min(400, fs//2))
    ax4.axhline(target_freq, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(mesh_cwt, ax=ax4, label='Mag')

    # === 5. 目标频率幅值提取 (Time) ===
    ax5 = axes[3]
    ax5.plot(t_stft, trace_stft, label='STFT Trace', color='red', linewidth=1.5, alpha=0.8)
    ax5.plot(t_cwt, trace_cwt, label='CWT Trace', color='purple', linewidth=1, alpha=0.8)
    # ax5.plot(t_fft, f_fft, label='FFT Trace', color='green', linewidth=1, alpha=0.8, linestyle=':')
    # ax5.plot(time, after_modulate, label='Base (Rect)', color='blue', linewidth=1, alpha=0.2, linestyle='--')
    ax5.set_title(f'5. Extracted Amplitude @ {target_freq}Hz')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Magnitude')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # === 关键步骤：强制对齐所有时间轴 ===
    # 将 ax1, ax3, ax4, ax5 的 X 轴锁定在一起
    # ax2 是 FFT，不参与时间轴锁定
    time_axes = [ax1, ax3, ax4, ax5]

    # 1. 统一设置显示范围
    for ax in time_axes:
        ax.set_xlim(4, 4.5)

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import scipy.signal as signal
from scipy.fft import fft, fftfreq


# ==========================================
# 第一部分：信号生成函数 (拆解版)
# ==========================================

def gen_base_square_wave(fs, duration, density=1.0):
    """
    生成基础随机矩形波 (开:1, 闭:0.3)
    """
    total_samples = int(fs * duration)
    sig = np.zeros(total_samples)

    # 转换为采样点数
    min_on = int(20e-3 * fs)
    max_on = int(70e-3 * fs)

    # density 控制间隔密度
    min_off = int((50e-3 * fs) / density)
    max_off = int((100e-3 * fs) / density)

    if min_off < 1: min_off = 1
    if max_off <= min_off: max_off = min_off + 1

    idx = 0
    while idx < total_samples:
        # 生成开信号 (值为1)
        on_len = np.random.randint(min_on, max_on)
        end_on = min(idx + on_len, total_samples)
        sig[idx:end_on] = 1.0

        idx = end_on
        if idx >= total_samples: break

        # 生成闭信号 (值为0.3)
        off_len = np.random.randint(min_off, max_off)
        end_off = min(idx + off_len, total_samples)
        sig[idx:end_off] = 0.3

        idx = end_off

    return sig


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


def create_composite_signal(fs=1500, duration=5, mod_freq=150, density=1.0, mod_amp=1.0, noise_amp = 2, noise_level=0.05):
    """
    组合上述三个步骤
    """
    # 1. 基础信号
    base_sig = gen_base_square_wave(fs, duration, density)
    # 2. 正弦调制
    sine_sig = gen_sine_modulation(fs, duration, mod_freq, amp=mod_amp)
    # 3. 噪声
    noise_sig = gen_random_noise(fs, duration, amp = noise_amp, level=noise_level)

    # 叠加
    final_sig = base_sig * sine_sig + noise_sig

    return base_sig, final_sig


# ==========================================
# 第二部分：信号分析 (STFT 和 CWT)
# ==========================================
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
    f_stft, t_stft, Zxx = signal.stft(sig, fs, window=window ,nperseg=100, noverlap=99, nfft= fs, return_onesided=True)

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
        'fft': (extract_150hz_amplitude(sig,fs)),
        'stft': (t_stft, f_stft, psd_stft, amp_trace_stft),
        'cwt': (np.arange(len(sig)) / fs, freqs_cwt, psd_cwt, magnitude_avg)
    }


def extract_150hz_amplitude(signal, sampling_rate, frame_count=50):
    """
    将时序信号分割成指定帧数，对每帧进行FFT，提取150Hz分量的幅值

    参数:
    ----------
    signal : numpy.ndarray
        一维时序信号数组
    sampling_rate : float
        采样率(Hz)
    frame_count : int, optional
        要分割的帧数，默认为50

    返回:
    ----------
    numpy.ndarray
        包含每帧150Hz分量幅值的数组，长度为frame_count
    """

    # 输入验证
    if not isinstance(signal, np.ndarray):
        raise ValueError("输入信号必须为numpy ndarray")
    if signal.ndim != 1:
        raise ValueError("输入信号必须为一维数组")
    if len(signal) < frame_count:
        raise ValueError("信号长度必须大于或等于帧数")
    if sampling_rate <= 0:
        raise ValueError("采样率必须大于0")

    # 计算每帧的长度
    total_length = len(signal)
    frame_length = total_length // frame_count

    # 如果信号长度不能被帧数整除，截断尾部多余的部分
    signal = signal[:frame_length * frame_count]

    # 目标频率
    target_freq = 150.0  # Hz

    # 初始化结果数组
    amplitudes = np.zeros(frame_count)

    for i in range(frame_count):
        # 提取当前帧
        start_idx = i * frame_length
        end_idx = (i + 1) * frame_length
        frame = signal[start_idx:end_idx]

        # 计算FFT
        fft_result = fft(frame)

        # 获取FFT的频率分量
        freqs = fftfreq(frame_length, 1 / sampling_rate)

        # 找到最接近150Hz的频率索引
        # 取正频率部分
        pos_freqs = freqs[:frame_length // 2]
        pos_fft = fft_result[:frame_length // 2]

        # 找到最接近150Hz的频率索引
        freq_idx = np.argmin(np.abs(pos_freqs - target_freq))

        # 获取该频率对应的幅值（取模）
        amplitude = np.abs(pos_fft[freq_idx])

        # 由于FFT结果是对称的，需要处理幅值缩放
        # 对于实际信号，需要乘以2（除了直流分量和Nyquist频率）
        if 0 < freq_idx < len(pos_freqs) - 1:
            amplitude = 2 * amplitude / frame_length
        else:
            amplitude = amplitude / frame_length

        amplitudes[i] = amplitude

    return np.arange(len(amplitudes))/10,amplitudes-2


# ==========================================
# 第三部分：绘图逻辑
# ==========================================

def plot_all_results(base_sig, final_sig, analysis_res, fs, target_freq, wavelet):
    t_stft, f_stft, psd_stft, trace_stft = analysis_res['stft']
    t_cwt, f_cwt, psd_cwt, trace_cwt = analysis_res['cwt']
    t_fft, f_fft = analysis_res['fft']

    total_time = len(final_sig) / fs
    t_full = np.arange(len(final_sig)) / fs

    # 创建5行1列的布局
    # sharex=False，因为 FFT 的 x轴是频率，不能和时间的 x轴共享
    fig, axes = plt.subplots(4, 1, figsize=(18, 16), constrained_layout=True)

    # === 1. 原始信号 (Time) ===
    ax1 = axes[0]
    ax1.plot(t_full, base_sig, color='red', alpha=1, linestyle='--', label='Base (Rect)')
    ax1.plot(t_full, final_sig, color='blue', alpha=0.5, linewidth=0.8, label='Final Signal')
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
    ax3.set_ylim(0, min(400,fs//2))
    ax3.axhline(target_freq, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(mesh_stft, ax=ax3, label='dB')

    # === 4. CWT 谱图 (Time-Freq) ===
    ax4 = axes[2]
    # CWT 的 x 轴直接使用 t_cwt (即 t_full)
    mesh_cwt = ax4.pcolormesh(t_cwt, f_cwt, psd_cwt, shading='auto', cmap='viridis')
    ax4.set_title('4. CWT Scalogram')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_ylim(1, min(400,fs//2))
    ax4.axhline(target_freq, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(mesh_cwt, ax=ax4, label='Mag')

    # === 5. 目标频率幅值提取 (Time) ===
    ax5 = axes[3]
    ax5.plot(t_stft, trace_stft, label='STFT Trace', color='red', linewidth=1.5, alpha=0.8)
    ax5.plot(t_cwt, trace_cwt, label='CWT Trace', color='purple', linewidth=1, alpha=0.8)
    ax5.plot(t_fft, f_fft, label='FFT Trace', color='green', linewidth=1, alpha=0.8,linestyle=':')
    ax5.plot(t_full, base_sig, label='Base (Rect)', color='blue', linewidth=1, alpha=0.2, linestyle='--')
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
        ax.set_xlim(0.2, total_time-0.2)

    # 2. 启用 sharex 机制 (手动链接)
    # 这样你在交互式窗口缩放 ax1 时，ax3/4/5 也会跟着动
    # ax1.get_shared_x_axes().joined(ax1, ax3, ax4, ax5)

    plt.show()


# ==========================================
# 主程序执行
# ==========================================
if __name__ == "__main__":
    # 参数设定
    FS = 1500
    DURATION = 2.0  # 为了绘图清晰，这里生成2秒数据，你可以改为5秒
    MOD_FREQ = 150
    DENSITY = 0.5
    WAVELET = 'cmor1.5-1'
    noise_amp = 2
    noise_level = 0.3
    mod_amp = 2

    # 1. 生成信号
    base_sig = gen_base_square_wave(FS, DURATION, DENSITY)
    # 2. 正弦调制
    sine_sig = gen_sine_modulation(FS, DURATION, MOD_FREQ, amp=mod_amp)
    # 3. 噪声
    inner_noise_sig = gen_random_noise(FS, DURATION, amp =2 , level=noise_level)
    outer_noise_sig = gen_random_noise(FS, DURATION, amp=15, level=3)
    # 叠加
    full_signal0 = (base_sig + inner_noise_sig) * sine_sig + outer_noise_sig

    full_signal = full_signal0 - np.mean(full_signal0[0:300])

    # 2. 分析信号
    results = analyze_signals(full_signal, FS, target_freq=MOD_FREQ, wavelet=WAVELET)

    # 3. 绘图
    plot_all_results(base_sig+2, full_signal0, results, FS, target_freq=MOD_FREQ, wavelet=WAVELET)

    # 4. 保存
    # df1 = pd.DataFrame({
    #     'time': np.arange(len(full_signal0)) / FS,
    #     'channel_simulate':base_sig,
    #     'inner_noise' : inner_noise_sig,
    #     'sine_signal' : sine_sig,
    #     'outer_noise' : outer_noise_sig,
    #     'origin_signal': full_signal0,
    #     'stft': results['stft'][3][1:],
    #     'cwt': results['cwt'][3],
    # })
    #
    # # 导出为CSV
    # df1.to_csv('simulate_compare_all.csv', index=False)
    #
    # df2 = pd.DataFrame({'fft_time':results['fft'][0],
    #                     'fft':results['fft'][1],})
    # df2.to_csv('simulate_compare_all.csv', index=False)
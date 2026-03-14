import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import scipy.signal as signal
from scipy.fft import fft, fftfreq


def load_npy_file(file_path, flatten=False, dtype=None, order='C'):
    """
    加载.npy文件并可选地展平数组

    参数:
    ----------
    file_path : str
        .npy文件路径
    flatten : bool, 默认False
        如果为True，则将多维数组展平为一维
    dtype : numpy.dtype, 可选
        指定返回数组的数据类型
    order : {'C', 'F', 'A'}, 可选
        展平时的顺序：
        - 'C': 按行（C风格）展平
        - 'F': 按列（Fortran风格）展平
        - 'A': 保留原始顺序

    返回:
    -------
    numpy.ndarray
        加载的数组数据
    """
    # 检查文件是否存在
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 检查文件扩展名
    if not file_path.lower().endswith('.npy'):
        print(f"警告: 文件扩展名不是 .npy: {file_path}")

    try:
        # 加载.npy文件
        data = np.load(file_path, allow_pickle=True)

        # 打印原始数据信息
        print(f"原始数据形状: {data.shape}")
        print(f"原始数据类型: {data.dtype}")
        print(f"原始数组维度: {data.ndim}D")

        # 数据类型转换
        if dtype is not None:
            data = data.astype(dtype)
            print(f"转换后数据类型: {data.dtype}")

        # 展平处理
        if flatten:
            original_shape = data.shape
            data = data.flatten(order=order)
            print(f"展平后形状: {data.shape} (原始形状: {original_shape})")

        return data

    except Exception as e:
        raise ValueError(f"加载.npy文件时出错: {e}")

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
        'fft': (extract_amplitude(sig,fs,target_freq)),
        'stft': (t_stft, f_stft, psd_stft, amp_trace_stft),
        'cwt': (np.arange(len(sig)) / fs, freqs_cwt, psd_cwt, magnitude_avg)
    }

def extract_amplitude(signal, sampling_rate,target_freq, frame_count=50):
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
    target_freq = target_freq  # Hz

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

def plot_all_results(final_sig, analysis_res, fs, target_freq, wavelet):
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
    ax1.plot(t_full, final_sig, color='blue', alpha=0.5, linewidth=0.8, label='Signal')
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
    # ax5.plot(t_fft, f_fft, label='FFT Trace', color='green', linewidth=1, alpha=0.8,linestyle=':')
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


def arrays_to_csv(time_array, signal_array, save_path,
                  header=None, index=False, time_col_name='Time',
                  signal_col_name='Signal', verbose=True):
    """
    将两个等长一维数组组合并保存为CSV文件

    参数:
    ----------
    time_array : np.ndarray
        时间序列数组
    signal_array : np.ndarray
        信号序列数组
    save_path : str
        保存的CSV文件路径
    header : bool or list, 默认None
        是否包含表头，True使用默认列名，或传入列表自定义
    index : bool, 默认False
        是否包含索引列
    time_col_name : str, 默认'Time'
        时间列的名称
    signal_col_name : str, 默认'Signal'
        信号列的名称
    verbose : bool, 默认True
        是否显示详细信息

    返回:
    -------
    pd.DataFrame
        创建的数据框
    """
    # 输入验证
    if not isinstance(time_array, np.ndarray) or not isinstance(signal_array, np.ndarray):
        raise TypeError("输入必须是numpy数组")

    if time_array.ndim != 1 or signal_array.ndim != 1:
        raise ValueError("输入必须是一维数组")

    if len(time_array) != len(signal_array):
        raise ValueError(f"数组长度不一致: time={len(time_array)}, signal={len(signal_array)}")

    if len(time_array) == 0:
        raise ValueError("数组不能为空")

    # 创建数据框
    if header is None or header is False:
        # 无表头
        data_dict = {0: time_array, 1: signal_array}
        df = pd.DataFrame(data_dict)
    else:
        # 有表头
        if header is True:
            # 使用默认列名
            col_names = [time_col_name, signal_col_name]
        else:
            # 使用自定义列名
            if len(header) != 2:
                raise ValueError("表头列表必须包含两个元素")
            col_names = header

        data_dict = {col_names[0]: time_array, col_names[1]: signal_array}
        df = pd.DataFrame(data_dict)

    # 保存为CSV
    save_dir = os.path.dirname(save_path)
    if save_dir:  # 如果指定了目录
        os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）

    df.to_csv(save_path, index=index, header=(header is not False and header is not None))

    if verbose:
        print(f"数据已保存到: {save_path}")
        print(f"文件大小: {os.path.getsize(save_path) / 1024:.2f} KB")
        print(f"数据形状: {df.shape}")
        print(f"时间范围: {time_array.min():.6f} 到 {time_array.max():.6f}")
        print(f"信号范围: {signal_array.min():.6f} 到 {signal_array.max():.6f}")

        # 预览数据
        print("\n数据预览:")
        print(df.head())

    return df

# ==========================================
# 主程序执行
# ==========================================
if __name__ == "__main__":
    # 参数设定
    FS = 1000
    DURATION = 4.0  # 为了绘图清晰，这里生成2秒数据，你可以改为5秒
    MOD_FREQ = 50
    DENSITY = 0.5
    WAVELET = 'cmor1.5-1'
    file_name = 'tri_phasechange_4s1000hz50'

    full_signal = load_npy_file(fr"H:\Newera\programing\simulate_data\{file_name}.npy")

    # 2. 分析信号
    results = analyze_signals(full_signal, FS, target_freq=MOD_FREQ, wavelet=WAVELET)

    # 3. 绘图
    plot_all_results(full_signal, results, FS, target_freq=MOD_FREQ, wavelet=WAVELET)

    # arrays_to_csv(results['stft'][0],results['stft'][3], fr"H:\Newera\dataprocessing\simulate-findAP\{file_name}-stft.csv")
    # arrays_to_csv(results['cwt'][0], results['cwt'][3], fr"H:\Newera\dataprocessing\simulate-findAP\{file_name}-cwt.csv")

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
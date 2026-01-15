import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
import pandas as pd
import os

# ==========================================
# 0. 基础设置与信号生成
# ==========================================
# 创建保存结果的文件夹
if not os.path.exists('comparison_results'):
    os.makedirs('comparison_results')

# # 信号参数
fs = 1500  # 采样率高一点，保证波形细腻
duration = 1
t = np.arange(int(fs * duration)) / fs
sig = np.zeros_like(t)

# 生成 0.1s 的方波 (居中)
# 0.5s总长，0.1s宽 -> 从 0.2s 到 0.3s
start_idx = int(0.4 * fs)
end_idx = int(0.6 * fs)
sig[start_idx:end_idx] = 0.7
sig0 = sig + 0.3

sig = sig0 * np.sin(2 * np.pi * 150 * t)
# fs = 1500  # 采样频率
# t = np.arange(0, 1, 1 / fs)  # 1秒时间
# f_target = 150
#
# s1 = np.sin(2 * np.pi * f_target * t)
# mask = (t >= 0.4) & (t <= 0.6)
# s1[mask] = 3 * s1[mask]
# sig = s1 + 0.1 * np.random.randn(len(t))

def normalize_01(data):
    """将数据归一化到 0 ~ 1 范围"""
    d_min = np.min(data)
    d_max = np.max(data)
    if d_max - d_min == 0:
        return data # 防止除以0
    return (data - d_min) / (d_max - d_min)

# 定义一个通用的绘图和保存函数
def save_and_plot(title, filename_base, df_results, t_axis):
    """
    绘制曲线并保存 CSV
    """
    # 1. 保存 CSV
    csv_path = f'comparison_results/{filename_base}.csv'
    # 将时间轴作为第一列加入
    df_export = df_results.copy()
    df_export.insert(0, 'Time', t_axis)
    df_export.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # 2. 绘图 (单独一张图)
    plt.figure(figsize=(10, 6))

    # 画原始信号作为参考 (灰色填充)
    plt.plot(t_axis, sig0, color='gray', alpha=0.2, label='Original Rect')

    for col in df_results.columns:
        plt.plot(t_axis, df_results[col], label=col, linewidth=1.5, alpha=0.7)

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Magnitude (Envelope Recovery)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    img_path = f'comparison_results/{filename_base}.png'
    plt.savefig(img_path, dpi=150)
    plt.show()


# 辅助函数：将不同时间轴的数据插值对齐到原始时间轴 t
def interp_to_t(t_new, y_new, t_orig):
    return np.interp(t_orig, t_new, y_new)


# ==========================================
# 1. STFT 不同步长 (Step Size / Hop Length)
# ==========================================
# 固定窗口长度 128
nperseg_fixed = 64
steps = [1, 8,16, 32, 63]
results_1 = {}

print("Processing 1: STFT Step Sizes...")
for step in steps:
    noverlap = nperseg_fixed - step
    f, t_s, Zxx = signal.stft(sig, fs, window='hann', nperseg=nperseg_fixed, noverlap=noverlap,nfft=fs, return_onesided=True)
    # 计算150
    idx_stft = np.argmin(np.abs(f - 150))
    # 提取该频率随时间变化的幅值 (取模)
    amp_trace_stft = np.abs(Zxx[idx_stft, :])
    # 插值对齐
    results_1[f'Step={step}'] = (interp_to_t(t_s, amp_trace_stft, t))*2

save_and_plot('1. STFT Comparison: Step Sizes (Window=128)', '1_stft_step_size', pd.DataFrame(results_1), t)

# ==========================================
# 2. STFT 不同窗口大小 (Window Size)
# ==========================================
# 固定重叠率为 50% (step = size / 2)
sizes = [512, 256, 128, 64, 32]
results_2 = {}

print("Processing 2: STFT Window Sizes...")
for size in sizes:
    noverlap = size // 2
    f, t_s, Zxx = signal.stft(sig, fs, window='hann', nperseg=size, noverlap=noverlap,nfft=fs, return_onesided=True)
    # 计算150
    idx_stft = np.argmin(np.abs(f - 150))
    # 提取该频率随时间变化的幅值 (取模)
    amp_trace_stft = np.abs(Zxx[idx_stft, :])
    # 插值对齐
    results_2[f'WinSize={size}'] = (interp_to_t(t_s, amp_trace_stft, t))*2

save_and_plot('2. STFT Comparison: Window Sizes (Overlap=50%)', '2_stft_window_size', pd.DataFrame(results_2), t)

# ==========================================
# 3. STFT 不同窗口类型 (Window Type)
# ==========================================
# 固定 size=128, step=64
types = ['hann', 'hamming', 'gaussian', 'boxcar', 'blackman']
results_3 = {}

print("Processing 3: STFT Window Types...")
for w_type in types:
    # 处理 gaussian 需要的参数 tuple
    if w_type == 'gaussian':
        window_arg = ('gaussian', 20)  # std=20
        label = 'gaussian(std=20)'
    elif w_type == 'boxcar':
        window_arg = 'boxcar'
        label = 'rect (boxcar)'
    else:
        window_arg = w_type
        label = w_type

    f, t_s, Zxx = signal.stft(sig, fs, window=window_arg, nperseg=128, noverlap=124,nfft=fs, return_onesided=True)
    idx_stft = np.argmin(np.abs(f - 150))
    # 提取该频率随时间变化的幅值 (取模)
    amp_trace_stft = np.abs(Zxx[idx_stft, :])
    results_3[label] = (interp_to_t(t_s, amp_trace_stft, t))*2

save_and_plot('3. STFT Comparison: Window Types', '3_stft_window_types', pd.DataFrame(results_3), t)

# ==========================================
# 4. CWT 不同小波类型 (Wavelet Family)
# ==========================================
# 定义 CWT 通用参数
target_freqs = np.linspace(150 - 1 // 2, 150 + 1 // 2,
                               1)  # totalscales//4
  # 尺度范围
wavelets = ['morl', 'cmor1-1', 'cgau4', 'shan1-1.0']
results_4 = {}

print("Processing 4: CWT Wavelet Families...")
for wav in wavelets:
    scales = pywt.frequency2scale(wav, target_freqs * 1.0 / fs)
    coefs, _ = pywt.cwt(sig, scales, wav, sampling_period=1 / fs)
    # 取所有尺度的平均幅值
    coefs_corrected = coefs / np.sqrt(scales[:, None])
    envelope = np.mean(np.abs(coefs_corrected), axis=0)
    results_4[wav] = envelope  # CWT 长度和原始信号一致，无需插值

save_and_plot('4. CWT Comparison: Wavelet Families', '4_cwt_families', pd.DataFrame(results_4), t)

# ==========================================
# 5. CWT cmor: 不同中心频率 (Center Freq)
# ==========================================
# Bandwidth 固定 1.5, Center 变化
centers = [3, 2, 1, 0.5]
results_5 = {}

print("Processing 5: CWT cmor Center Frequencies...")
for c in centers:
    w_name = f'cmor1-{c}'
    scales = pywt.frequency2scale(w_name, target_freqs * 1.0 / fs)
    coefs, _ = pywt.cwt(sig, scales, w_name, sampling_period=1 / fs)
    coefs = coefs /( np.sqrt(scales[:, None]))

    results_5[w_name] = np.mean(np.abs(coefs), axis=0)*2

save_and_plot('5. CWT cmor: Varying Center Freq (B=1.5)', '5_cwt_cmor_center', pd.DataFrame(results_5), t)

# ==========================================
# 6. CWT cmor: 不同带宽 (Bandwidth)
# ==========================================
# Center 固定 1.0, Bandwidth 变化
bandwidths = [5, 3, 1.5, 1, 0.5]
results_6 = {}

print("Processing 6: CWT cmor Bandwidths...")
for b in bandwidths:
    w_name = f'cmor{b}-1.0'
    scales = pywt.frequency2scale(w_name, target_freqs * 1.0 / fs)
    coefs, _ = pywt.cwt(sig, scales, w_name, sampling_period=1 / fs)
    coefs = coefs / np.sqrt(scales[:, None])
    results_6[w_name] = np.mean(np.abs(coefs), axis=0)*2

save_and_plot('6. CWT cmor: Varying Bandwidth (C=1.0)', '6_cwt_cmor_bandwidth', pd.DataFrame(results_6), t)

print("\nDone! All plots displayed and CSVs saved in 'comparison_results' folder.")
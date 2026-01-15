import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import warnings

warnings.filterwarnings('ignore')

# 设置参数
fs = 1500  # 采样频率
t = np.arange(0, 1, 1 / fs)  # 1秒时间
f_target = 150  # 目标频率


# 信号生成函数
def generate_signals():
    """生成多种对比信号"""
    signals = []

    # 1. 纯150Hz正弦波，中间有幅值突变
    s1 = np.sin(2 * np.pi * f_target * t)
    mask = (t >= 0.4) & (t <= 0.6)
    s1[mask] = 2 * s1[mask]
    s1 = s1 + 0.1 * np.random.randn(len(t))
    signals.append((s1, "纯150Hz正弦波，中间段幅值加倍"))

    # 2. 频率在150Hz附近跳变
    s2 = np.zeros_like(t)
    mask1 = t <= 0.3
    s2[mask1] = np.sin(2 * np.pi * 150 * t[mask1])
    mask2 = (t > 0.3) & (t <= 0.7)
    s2[mask2] = np.sin(2 * np.pi * 155 * t[mask2])
    mask3 = t > 0.7
    s2[mask3] = np.sin(2 * np.pi * 150 * t[mask3])
    s2 = s2 + 0.1 * np.random.randn(len(t))
    signals.append((s2, "频率跳变: 150Hz→155Hz→150Hz"))

    # 3. 多分量信号，150Hz持续存在，155Hz间歇出现
    s3 = np.sin(2 * np.pi * 150 * t)
    mask = (t >= 0.2) & (t <= 0.5)
    s3[mask] = s3[mask] + 0.8 * np.sin(2 * np.pi * 155 * t[mask])
    s3 = s3 + 0.1 * np.random.randn(len(t))
    signals.append((s3, "150Hz持续 + 155Hz(0.2-0.5秒)"))

    # 4. 150Hz瞬时脉冲信号
    s4 = np.zeros_like(t)
    pulse_start = int(0.45 * fs)
    pulse_end = int(0.46 * fs)  # 10ms脉冲
    t_pulse = t[pulse_start:pulse_end]
    s4[pulse_start:pulse_end] = np.sin(2 * np.pi * f_target * t_pulse) * \
                                np.hanning(len(t_pulse))  # 加窗减少泄漏
    s4 = s4 + 0.05 * np.random.randn(len(t))
    signals.append((s4, "150Hz瞬时脉冲(0.45-0.46秒)"))

    # 5. 150Hz幅值调制信号
    s5 = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * f_target * t)
    s5 = s5 + 0.1 * np.random.randn(len(t))
    signals.append((s5, "150Hz幅值调制(5Hz调制频率)"))

    # 6. 频率快速变化: 在150Hz附近快速摆动
    s6 = np.sin(2 * np.pi * (f_target + 10 * np.sin(2 * np.pi * 20 * t)) * t)
    s6 = s6 + 0.1 * np.random.randn(len(t))
    signals.append((s6, "频率快速摆动: 150±10Hz, 20Hz摆动速率"))

    # 7. 150Hz + 300Hz谐波，150Hz间歇性消失
    s7 = np.sin(2 * np.pi * 300 * t)  # 始终有300Hz
    mask = (t < 0.3) | (t > 0.7)
    s7[mask] = s7[mask] + np.sin(2 * np.pi * 150 * t[mask])  # 150Hz只在两端有
    s7 = s7 + 0.1 * np.random.randn(len(t))
    signals.append((s7, "300Hz持续 + 150Hz间歇(两端有，中间无)"))

    # 8. 频率斜坡变化
    s8 = np.zeros_like(t)
    for i in range(len(t)):
        freq = 140 + 20 * t[i]  # 从140Hz线性增加到160Hz
        s8[i] = np.sin(2 * np.pi * freq * t[i])
    s8 = s8 + 0.1 * np.random.randn(len(t))
    signals.append((s8, "频率斜坡: 140Hz→160Hz"))

    return signals


# STFT计算函数
def compute_stft(signal_data, fs, target_freq, nperseg=128, noverlap=127, nfft=1500):
    """计算STFT并提取目标频率的幅值"""
    f_stft, t_stft, Zxx = signal.stft(
        signal_data, fs, window='hann',
        nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    freq_idx = np.argmin(np.abs(f_stft - target_freq))
    magnitude_stft = np.abs(Zxx[freq_idx, :])
    return t_stft, magnitude_stft, f_stft, Zxx


# CWT计算函数
def compute_cwt(signal_data, fs, target_freq, wavelet='cmor3-3'):
    """计算CWT并提取目标频率的幅值"""
    match = re.match(r'([a-zA-Z]+)(\d+\.\d+|\d+)-(\d+\.\d+|\d+)', wavelet)

    if match:
        wavelet_name = match.group(1)  # 'cmor'
        fb = float(match.group(2))  # '3'
        fc = float(match.group(3))

    # 计算对应目标频率的尺度
    scale = fc * fs / target_freq

    # 计算CWT
    scales = [scale]  # 只计算目标频率对应的尺度
    coefficients, frequencies = pywt.cwt(
        signal_data, scales, wavelet, sampling_period=1 / fs
    )
    magnitude_cwt = np.abs(coefficients[0, :])

    return t, magnitude_cwt, coefficients


# 改进的绘图函数
def plot_comparison_enhanced(signal_data, signal_title, fs, target_freq=150, nperseg=128):
    """绘制信号及其STFT/CWT对比分析结果"""
    fig = plt.figure(figsize=(16, 10))

    # 1. 原始信号
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(t, signal_data, 'k-', linewidth=0.8)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.set_title(f'原始信号: {signal_title}')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])

    # 2. 频谱分析
    ax2 = plt.subplot(4, 2, 2)
    freqs = np.fft.rfftfreq(len(signal_data), 1 / fs)
    fft_vals = np.abs(np.fft.rfft(signal_data))
    ax2.plot(freqs[:500], fft_vals[:500], 'b-', linewidth=1)
    ax2.axvline(x=target_freq, color='r', linestyle='--', alpha=0.5, label=f'目标频率: {target_freq}Hz')
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('幅值')
    ax2.set_title('信号频谱')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0, 500])

    # 计算STFT和CWT
    t_stft, mag_stft, f_stft, Zxx_stft = compute_stft(signal_data, fs, target_freq, nperseg=nperseg)
    t_cwt, mag_cwt, coeff_cwt = compute_cwt(signal_data, fs, target_freq)

    # 3. STFT和CWT结果对比 (直接叠加)
    ax3 = plt.subplot(4, 2, (3, 4))
    # 归一化以便比较
    mag_stft_norm = mag_stft / np.max(mag_stft) if np.max(mag_stft) > 0 else mag_stft
    mag_cwt_norm = mag_cwt / np.max(mag_cwt) if np.max(mag_cwt) > 0 else mag_cwt

    ax3.plot(t_stft, mag_stft_norm, 'b-', linewidth=2, label='STFT (汉宁窗, 窗长128)')
    ax3.plot(t_cwt, mag_cwt_norm, 'r-', linewidth=1, alpha=0.8, label='CWT (cmor3-3)')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('归一化幅值')
    ax3.set_title('STFT与CWT提取的150Hz幅值对比')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim([0, 1])

    # 添加分辨率说明
    stft_time_res = nperseg / fs
    fb, fc = 3, 3
    sigma_t = np.sqrt(fb) / (np.pi * fc)
    cwt_scale = fc * fs / target_freq
    cwt_time_res = cwt_scale * sigma_t / fs

    res_text = f'STFT时间分辨率: {stft_time_res:.4f}s\nCWT时间分辨率: {cwt_time_res:.6f}s\nCWT/STFT分辨率比: {cwt_time_res / stft_time_res:.3f}'
    ax3.text(0.02, 0.98, res_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 4. STFT频谱图 (150Hz附近)
    ax4 = plt.subplot(4, 2, 5)
    freq_range = (f_stft >= target_freq - 30) & (f_stft <= target_freq + 30)
    pcm = ax4.pcolormesh(t_stft, f_stft[freq_range], np.abs(Zxx_stft)[freq_range, :],
                         shading='gouraud', cmap='viridis')
    ax4.set_ylabel('频率 (Hz)')
    ax4.set_xlabel('时间 (s)')
    ax4.set_title(f'STFT频谱图 ({target_freq}Hz附近)')
    plt.colorbar(pcm, ax=ax4, label='幅值')
    ax4.axhline(y=target_freq, color='r', linestyle='--', alpha=0.7)

    # 5. CWT复数系数的实部和虚部
    ax5 = plt.subplot(4, 2, 6)
    ax5.plot(t, np.real(coeff_cwt[0, :]), 'g-', linewidth=0.5, label='实部')
    ax5.plot(t, np.imag(coeff_cwt[0, :]), 'orange', linewidth=0.5, alpha=0.7, label='虚部')
    ax5.set_xlabel('时间 (s)')
    ax5.set_ylabel('系数值')
    ax5.set_title('CWT复数系数')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_xlim([0, 1])

    # 6. 细节对比: 放大感兴趣的区域
    ax6 = plt.subplot(4, 2, 7)
    # 自动选择变化最大的区域
    mag_diff = np.abs(np.diff(mag_stft_norm))
    if len(mag_diff) > 0:
        max_change_idx = np.argmax(mag_diff)
        zoom_center = t_stft[max_change_idx] if max_change_idx < len(t_stft) else 0.5
    else:
        zoom_center = 0.5

    zoom_start = max(0, zoom_center - 0.1)
    zoom_end = min(1, zoom_center + 0.1)

    zoom_mask_stft = (t_stft >= zoom_start) & (t_stft <= zoom_end)
    zoom_mask_cwt = (t >= zoom_start) & (t <= zoom_end)

    if np.any(zoom_mask_stft) and np.any(zoom_mask_cwt):
        ax6.plot(t_stft[zoom_mask_stft], mag_stft_norm[zoom_mask_stft], 'b-',
                 linewidth=2, label='STFT', marker='o', markersize=3)
        ax6.plot(t[zoom_mask_cwt], mag_cwt_norm[zoom_mask_cwt], 'r-',
                 linewidth=1, alpha=0.8, label='CWT')
        ax6.set_xlabel('时间 (s)')
        ax6.set_ylabel('归一化幅值')
        ax6.set_title(f'细节对比 ({zoom_start:.2f}s 到 {zoom_end:.2f}s)')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, '无显著变化区域', ha='center', va='center', transform=ax6.transAxes)

    # 7. 统计指标
    ax7 = plt.subplot(4, 2, 8)
    ax7.axis('off')

    # 计算统计指标
    stft_smoothness = np.std(np.diff(mag_stft))
    cwt_smoothness = np.std(np.diff(mag_cwt))

    # 计算峰度（衡量尖锐程度）
    from scipy.stats import kurtosis
    stft_kurt = kurtosis(mag_stft) if len(mag_stft) > 3 else 0
    cwt_kurt = kurtosis(mag_cwt) if len(mag_cwt) > 3 else 0

    stats_text = f"统计指标对比:\n\n"
    stats_text += f"平滑度(差值标准差):\n"
    stats_text += f"  STFT: {stft_smoothness:.6f}\n"
    stats_text += f"  CWT:  {cwt_smoothness:.6f}\n"
    stats_text += f"  比值(STFT/CWT): {stft_smoothness / cwt_smoothness:.2f}\n\n"
    stats_text += f"峰度(衡量分布尖锐度):\n"
    stats_text += f"  STFT: {stft_kurt:.2f}\n"
    stats_text += f"  CWT:  {cwt_kurt:.2f}\n\n"
    stats_text += f"最大幅值:\n"
    stats_text += f"  STFT: {np.max(mag_stft):.4f}\n"
    stats_text += f"  CWT:  {np.max(mag_cwt):.4f}"

    ax7.text(0.1, 0.5, stats_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))

    plt.suptitle(f'信号分析: {signal_title}', fontsize=16, y=1.02)
    plt.tight_layout()

    return fig, t_stft, mag_stft, t_cwt, mag_cwt


# 主程序
signals = generate_signals()
selected_signals = [0, 1, 2, 3, 4, 5, 6, 7]  # 选择要显示的所有信号

print("=" * 80)
print("STFT与CWT对比分析实验")
print("=" * 80)
print(f"采样频率: {fs} Hz")
print(f"目标频率: {f_target} Hz")
print(f"STFT参数: 汉宁窗, 窗长128, 步长1, 变换长度1500")
print(f"CWT参数: cmor3-3小波")
print("=" * 80)

# 分析每个信号
for i in selected_signals:
    sig, title = signals[i]
    print(f"\n分析信号 {i + 1}: {title}")

    fig, t_stft, mag_stft, t_cwt, mag_cwt = plot_comparison_enhanced(sig, title, fs, f_target)

    # 保存图形
    fig.savefig(f'signal_analysis_{i + 1}.png', dpi=150, bbox_inches='tight')
    plt.show()

# 创建汇总对比图
print("\n" + "=" * 80)
print("汇总对比: STFT与CWT特性总结")
print("=" * 80)

# 选择最具代表性的3个信号
summary_signals = [signals[0], signals[3], signals[5]]  # 幅值突变、瞬时脉冲、频率摆动
summary_titles = ["幅值突变", "瞬时脉冲", "频率快速摆动"]

fig_summary, axes = plt.subplots(3, 2, figsize=(14, 12))

for idx, ((sig, title), sig_type) in enumerate(zip(summary_signals, summary_titles)):
    # 计算STFT和CWT
    t_stft, mag_stft, f_stft, Zxx_stft = compute_stft(sig, fs, f_target)
    t_cwt, mag_cwt, coeff_cwt = compute_cwt(sig, fs, f_target)

    # 归一化
    mag_stft_norm = mag_stft / np.max(mag_stft) if np.max(mag_stft) > 0 else mag_stft
    mag_cwt_norm = mag_cwt / np.max(mag_cwt) if np.max(mag_cwt) > 0 else mag_cwt

    # 绘制对比
    ax1 = axes[idx, 0]
    ax1.plot(t, sig, 'k-', linewidth=0.8, alpha=0.7)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.set_title(f'信号 {idx + 1}: {sig_type}\n{title}')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])

    ax2 = axes[idx, 1]
    ax2.plot(t_stft, mag_stft_norm, 'b-', linewidth=2, label='STFT')
    ax2.plot(t_cwt, mag_cwt_norm, 'r-', linewidth=1, alpha=0.8, label='CWT')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('归一化幅值')
    ax2.set_title('STFT vs CWT 提取的150Hz幅值')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0, 1])

    # 添加评价
    stft_smooth = np.std(np.diff(mag_stft))
    cwt_smooth = np.std(np.diff(mag_cwt))

    if idx == 0:
        evaluation = "STFT: 平滑，清晰显示幅值变化\nCWT: 有震荡，但响应更快"
    elif idx == 1:
        evaluation = "STFT: 脉冲被展宽，能量分散\nCWT: 脉冲定位精确，但震荡明显"
    elif idx == 2:
        evaluation = "STFT: 平滑变化，频率响应慢\nCWT: 快速震荡，反映瞬时频率变化"

    ax2.text(0.02, 0.98, evaluation, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.suptitle('STFT与CWT方法对比总结', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('stft_cwt_comparison_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# 不同STFT窗长对比
print("\n" + "=" * 80)
print("STFT不同窗长对比")
print("=" * 80)

# 使用信号1
sig1, title1 = signals[0]

fig_windows, axes = plt.subplots(2, 3, figsize=(15, 8))
window_lengths = [64, 128, 256, 512, 1024, 2048]
window_names = ['窗长64\n(高时间分辨)', '窗长128\n(平衡)', '窗长256\n(高频率分辨)',
                '窗长512\n(很高频率分辨)', '窗长1024\n(极高频率分辨)', '窗长2048\n(最高频率分辨)']

for idx, (window_len, name) in enumerate(zip(window_lengths, window_names)):
    ax = axes[idx // 3, idx % 3]

    # 计算STFT
    t_stft, mag_stft, f_stft, Zxx_stft = compute_stft(
        sig1, fs, f_target, nperseg=window_len, noverlap=window_len - 1
    )

    # 计算CWT作为对比
    t_cwt, mag_cwt, coeff_cwt = compute_cwt(sig1, fs, f_target)

    # 归一化
    mag_stft_norm = mag_stft / np.max(mag_stft) if np.max(mag_stft) > 0 else mag_stft
    mag_cwt_norm = mag_cwt / np.max(mag_cwt) if np.max(mag_cwt) > 0 else mag_cwt

    # 绘制
    ax.plot(t_stft, mag_stft_norm, 'b-', linewidth=2, label='STFT')
    ax.plot(t_cwt, mag_cwt_norm, 'r-', linewidth=1, alpha=0.7, label='CWT')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('归一化幅值')
    ax.set_title(f'{name}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])

    # 添加时间分辨率
    time_res = window_len / fs
    ax.text(0.02, 0.95, f'时间分辨: {time_res:.4f}s',
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

plt.suptitle('STFT不同窗长对时间-频率分辨率的影响 (信号1: 幅值突变)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('stft_window_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 输出结论
print("\n" + "=" * 80)
print("实验结论总结")
print("=" * 80)
print("1. STFT特点:")
print("   - 时间分辨率固定: 窗长越长，频率分辨率越高，时间分辨率越低")
print("   - 结果平滑稳定，适合分析幅值包络")
print("   - 对频率选择性好，适合分析特定频率成分")
print("   - 对瞬时脉冲响应差（能量被展宽）")
print()
print("2. CWT特点:")
print("   - 时间分辨率自适应: 高频处时间分辨率高，低频处时间分辨率低")
print("   - 结果包含高频震荡，反映瞬时相位/频率变化")
print("   - 对瞬时事件定位精确")
print("   - 频率选择性较差，受邻近频率影响大")
print()
print("3. 方法选择建议:")
print("   - 分析特定频率的平滑幅值变化: 使用STFT")
print("   - 检测瞬态事件或快速变化: 使用CWT")
print("   - 需要精确频率定位: 使用STFT")
print("   - 需要高时间分辨率: 使用CWT")
print("   - 分析多分量信号: 使用STFT，或用CWT配合合适小波参数")
print()
print("4. 参数选择:")
print("   - STFT: 根据需要在时间分辨率和频率分辨率间权衡")
print("   - CWT: 选择合适的小波参数(fb值)平衡时间和频率分辨率")
print("=" * 80)
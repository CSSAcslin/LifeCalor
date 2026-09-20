import numpy as np
import pywt


def reduce_cwt_coefficients(coefficients, scales):
    """Reduce scale x time coefficients without constructing a scale result cube."""
    coefficients = np.asarray(coefficients)
    scales = np.asarray(scales, dtype=np.float64)
    if coefficients.ndim != 2:
        raise ValueError("CWT coefficients must have shape scale x time")
    if scales.ndim != 1 or coefficients.shape[0] != scales.size:
        raise ValueError("CWT scale count does not match coefficient rows")
    if scales.size == 0 or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("CWT scales must be finite positive values")

    normalized = 2.0 * np.abs(coefficients) / np.sqrt(scales[:, None])
    return np.mean(normalized, axis=0).astype(np.float32, copy=False)


def cwt_axes(*, target_freq, scale_range, total_scales, wavelet, fps):
    if int(total_scales) < 1:
        raise ValueError("CWT total_scales must be at least 1")
    if float(fps) <= 0:
        raise ValueError("CWT fps must be positive")

    half_width = max(0.0, float(scale_range)) / 2.0
    target_frequencies = np.linspace(
        float(target_freq) - half_width,
        float(target_freq) + half_width,
        int(total_scales),
    )
    if np.any(target_frequencies <= 0):
        raise ValueError("CWT target frequency range must stay above 0 Hz")
    scales = pywt.frequency2scale(wavelet, target_frequencies / float(fps))
    return np.asarray(scales), target_frequencies


def cwt_frequency_trace(
    values,
    *,
    target_freq,
    scale_range,
    total_scales,
    wavelet,
    fps,
):
    scales, _ = cwt_axes(
        target_freq=target_freq,
        scale_range=scale_range,
        total_scales=total_scales,
        wavelet=wavelet,
        fps=fps,
    )
    coefficients, frequencies = pywt.cwt(
        np.asarray(values),
        scales,
        wavelet,
        sampling_period=1.0 / float(fps),
    )
    return reduce_cwt_coefficients(coefficients, scales), scales, frequencies

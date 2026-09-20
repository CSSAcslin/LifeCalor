from __future__ import annotations

import numpy as np
import pywt

from compute.algorithms import stft_frequency_trace
from compute.algorithms.cwt import cwt_axes


def _pixel_traces(block):
    array = np.asarray(block)
    if array.ndim != 3:
        raise ValueError(f"STFT 计算块必须为 THW，实际 shape={array.shape}")
    time_length, height, width = array.shape
    return array.transpose(1, 2, 0).reshape(height * width, time_length), height, width


def stft_block_cpu(block, params, *, compute_dtype, output_dtype):
    traces, height, width = _pixel_traces(block)
    traces = traces.astype(np.dtype(compute_dtype), copy=False)
    frequencies, times, magnitude, selected = stft_frequency_trace(
        traces,
        fs=params["fps"],
        window=params["window"],
        nperseg=params["window_size"],
        noverlap=params["noverlap"],
        nfft=params["nfft"],
        target_freq=params["target_freq"],
        scale_range=params.get("scale_range", 0.0),
    )
    output = magnitude.T.reshape(magnitude.shape[-1], height, width)
    return (
        output.astype(np.dtype(output_dtype), copy=False),
        np.asarray(frequencies),
        np.asarray(times),
        np.asarray(selected, dtype=np.intp),
    )


def cwt_block_cpu(block, params, *, compute_dtype, output_dtype):
    traces, height, width = _pixel_traces(block)
    traces = traces.astype(np.dtype(compute_dtype), copy=False)
    scales, _ = cwt_axes(
        target_freq=params["target_freq"],
        scale_range=params.get("scale_range", 0.0),
        total_scales=params["total_scales"],
        wavelet=params["wavelet"],
        fps=params["fps"],
    )
    coefficients, frequencies = pywt.cwt(
        traces,
        scales,
        params["wavelet"],
        sampling_period=1.0 / float(params["fps"]),
        axis=-1,
    )
    normalized = 2.0 * np.abs(coefficients) / np.sqrt(scales[:, None, None])
    magnitude = np.mean(normalized, axis=0)
    output = magnitude.T.reshape(magnitude.shape[-1], height, width)
    return (
        output.astype(np.dtype(output_dtype), copy=False),
        np.asarray(scales),
        np.asarray(frequencies),
    )

def fft2_block_cpu(block, *, compute_dtype, output_dtype=None):
    array = np.asarray(block).astype(np.dtype(compute_dtype), copy=False)
    result = np.fft.fft2(array, axes=(-2, -1))
    return result.astype(np.dtype(output_dtype or result.dtype), copy=False)

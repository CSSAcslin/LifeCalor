from dataclasses import dataclass
from math import ceil, floor

import numpy as np
import pywt


SUPPORTED_CWT_WAVELETS = (
    "cmor1-1.0",
    "cmor1.5-1.0",
    "cmor3-3",
    "cmor8-3",
    "cgau8",
    "mexh",
    "morl",
)


@dataclass(frozen=True)
class CwtKernelBank:
    wavelet: str
    precision: int
    scales: np.ndarray
    frequencies: np.ndarray
    kernels: tuple[np.ndarray, ...]
    complex_cwt: bool
    input_dtype: str
    output_dtype: str
    pywavelets_version: str


def normalize_cwt_wavelet(wavelet):
    normalized = str(wavelet).strip()
    if normalized not in SUPPORTED_CWT_WAVELETS:
        raise ValueError(f"CWT 小波尚未验证: {normalized or '<empty>'}")
    return normalized


def _cwt_input_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind in "biu":
        return np.dtype("float64")
    if dtype == np.dtype("float16"):
        return np.dtype("float32")
    return dtype


def build_cwt_kernel_bank(scales, wavelet, input_dtype, *, precision=10):
    normalized = normalize_cwt_wavelet(wavelet)
    scales = np.atleast_1d(np.asarray(scales, dtype=np.float64))
    if scales.size == 0 or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("CWT scales must be finite positive values")
    precision = int(precision)
    if precision < 1:
        raise ValueError("CWT kernel precision must be positive")

    wavelet_object = pywt.ContinuousWavelet(normalized)
    data_dtype = _cwt_input_dtype(input_dtype)
    complex_dtype = np.result_type(data_dtype, np.complex64)
    output_dtype = complex_dtype if wavelet_object.complex_cwt else data_dtype
    if data_dtype.kind == "c":
        output_dtype = data_dtype

    integrated, coordinates = pywt.integrate_wavelet(
        wavelet_object, precision=precision
    )
    if wavelet_object.complex_cwt:
        integrated = np.conj(integrated)
    kernel_dtype = complex_dtype if integrated.dtype.kind == "c" else data_dtype
    integrated = np.asarray(integrated, dtype=kernel_dtype)
    coordinates = np.asarray(coordinates, dtype=data_dtype.type(0).real.dtype)
    step = coordinates[1] - coordinates[0]

    kernels = []
    for scale in scales:
        indices = np.arange(
            scale * (coordinates[-1] - coordinates[0]) + 1
        ) / (scale * step)
        indices = indices.astype(int)
        if indices[-1] >= integrated.size:
            indices = indices[indices < integrated.size]
        kernel = np.asarray(integrated[indices][::-1])
        kernel.setflags(write=False)
        kernels.append(kernel)

    frozen_scales = scales.copy()
    frozen_scales.setflags(write=False)
    frequencies = np.atleast_1d(
        np.asarray(
            pywt.scale2frequency(wavelet_object, frozen_scales, precision),
            dtype=np.float64,
        )
    )
    frequencies.setflags(write=False)
    return CwtKernelBank(
        wavelet=normalized,
        precision=precision,
        scales=frozen_scales,
        frequencies=frequencies,
        kernels=tuple(kernels),
        complex_cwt=bool(wavelet_object.complex_cwt),
        input_dtype=data_dtype.name,
        output_dtype=np.dtype(output_dtype).name,
        pywavelets_version=str(pywt.__version__),
    )


def iter_cwt_coefficients(values, kernel_bank):
    data = np.asarray(values, dtype=np.dtype(kernel_bank.input_dtype))
    if data.ndim not in {1, 2}:
        raise ValueError("CWT kernel reference expects time or batch x time input")
    traces = data[None, :] if data.ndim == 1 else data
    output_dtype = np.dtype(kernel_bank.output_dtype)
    for scale, kernel in zip(kernel_bank.scales, kernel_bank.kernels):
        convolved = np.stack(
            [np.convolve(trace, kernel) for trace in traces],
            axis=0,
        )
        coefficient = -np.sqrt(scale) * np.diff(convolved, axis=-1)
        if output_dtype.kind != "c":
            coefficient = coefficient.real
        difference = (coefficient.shape[-1] - traces.shape[-1]) / 2.0
        if difference > 0:
            coefficient = coefficient[..., floor(difference):-ceil(difference)]
        elif difference < 0:
            raise ValueError(f"Selected scale of {scale} too small.")
        yield scale, coefficient.astype(output_dtype, copy=False)


def cwt_coefficients_from_kernels(values, kernel_bank):
    data = np.asarray(values)
    squeeze = data.ndim == 1
    coefficients = [
        coefficient
        for _, coefficient in iter_cwt_coefficients(values, kernel_bank)
    ]
    result = np.stack(coefficients, axis=0)
    return result[:, 0, :] if squeeze else result


def reduce_cwt_traces_from_kernels(
    values, kernel_bank, *, output_dtype="float32"
):
    data = np.asarray(values)
    squeeze = data.ndim == 1
    traces = data[None, :] if squeeze else data
    accumulator_dtype = (
        np.float64 if np.dtype(output_dtype).itemsize > 4 else np.float32
    )
    accumulator = np.zeros(traces.shape, dtype=accumulator_dtype)
    count = 0
    for scale, coefficient in iter_cwt_coefficients(values, kernel_bank):
        accumulator += (
            2.0 * np.abs(coefficient) / np.sqrt(scale)
        ).astype(accumulator_dtype, copy=False)
        count += 1
    if count == 0:
        raise ValueError("CWT kernel bank is empty")
    result = (accumulator / count).astype(np.dtype(output_dtype), copy=False)
    return result[0] if squeeze else result


def cwt_quality_spectrum(
    values,
    scales,
    wavelet,
    *,
    fps,
    compute_dtype=None,
    output_dtype="float32",
):
    data = np.asarray(values)
    compute_dtype = np.dtype(compute_dtype or _cwt_input_dtype(data.dtype))
    bank = build_cwt_kernel_bank(scales, wavelet, compute_dtype)
    coefficients = cwt_coefficients_from_kernels(
        data.astype(compute_dtype, copy=False), bank
    )
    spectrum = np.abs(coefficients).astype(np.dtype(output_dtype), copy=False)
    return spectrum, np.asarray(bank.frequencies) * float(fps)


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
    wavelet = normalize_cwt_wavelet(wavelet)
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
    wavelet = normalize_cwt_wavelet(wavelet)
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

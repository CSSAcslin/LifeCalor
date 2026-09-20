from numbers import Real

import numpy as np
from scipy import signal


def select_frequency_indices(frequencies, target_freq, scale_range=0.0):
    frequencies = np.asarray(frequencies)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("STFT frequency axis must be a non-empty 1D array")

    if isinstance(target_freq, Real):
        target = float(target_freq)
        half_width = max(0.0, float(scale_range)) / 2.0
        if half_width:
            indices = np.flatnonzero(
                (frequencies >= target - half_width)
                & (frequencies <= target + half_width)
            )
            if indices.size:
                return indices
        return np.asarray([int(np.argmin(np.abs(frequencies - target)))])

    indices = np.asarray(target_freq, dtype=np.intp).reshape(-1)
    if indices.size == 0:
        raise ValueError("At least one STFT frequency index is required")
    if np.any(indices < 0) or np.any(indices >= frequencies.size):
        raise IndexError("STFT frequency index is outside the transform axis")
    return indices


def _amplitude_factors(frequencies, selected, *, real_input, fs, nfft):
    factors = np.ones(len(selected), dtype=np.float64)
    if not real_input:
        return factors

    selected_freqs = np.asarray(frequencies)[selected]
    edge = np.isclose(selected_freqs, 0.0)
    if int(nfft) % 2 == 0:
        edge |= np.isclose(np.abs(selected_freqs), float(fs) / 2.0)
    factors[~edge] = 2.0
    return factors


def stft_frequency_trace(
    values,
    *,
    fs,
    window,
    nperseg,
    noverlap,
    nfft,
    target_freq,
    scale_range=0.0,
):
    """Compute the frozen STFT amplitude contract for one or many traces."""
    array = np.asarray(values)
    if array.ndim not in (1, 2):
        raise ValueError("STFT input must have shape T or batch x T")
    if array.shape[-1] < 1:
        raise ValueError("STFT input cannot be empty")

    real_input = not np.iscomplexobj(array)
    frequencies, times, coefficients = signal.stft(
        array,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=real_input,
        scaling="spectrum",
        axis=-1,
    )
    selected = select_frequency_indices(frequencies, target_freq, scale_range)
    selected_coefficients = np.take(coefficients, selected, axis=-2)
    factors = _amplitude_factors(
        frequencies,
        selected,
        real_input=real_input,
        fs=fs,
        nfft=nfft,
    )
    factor_shape = (1,) * (selected_coefficients.ndim - 2) + (-1, 1)
    magnitude = np.mean(
        np.abs(selected_coefficients) * factors.reshape(factor_shape),
        axis=-2,
    )
    return frequencies, times, magnitude.astype(np.float32, copy=False), selected

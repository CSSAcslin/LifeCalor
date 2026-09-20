from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class LifetimeFitResult:
    parameters: np.ndarray
    lifetime: object
    r_squared: float
    physical_signal: np.ndarray


def single_exponential(time, amplitude, lifetime, baseline):
    return amplitude * np.exp(-time / lifetime) + baseline


def double_exponential(time, amplitude1, lifetime1, amplitude2, lifetime2, baseline):
    return (
        amplitude1 * np.exp(-time / lifetime1)
        + amplitude2 * np.exp(-time / lifetime2)
        + baseline
    )


def _failed_result(model_type, physical_signal):
    if model_type == "single":
        parameters = np.full(3, np.nan)
        lifetime = np.nan
    else:
        parameters = np.full(5, np.nan)
        lifetime = (np.nan, np.nan)
    return LifetimeFitResult(parameters, lifetime, np.nan, physical_signal)


def _r_squared(observed, predicted):
    residual = np.sum((observed - predicted) ** 2, dtype=np.float64)
    total = np.sum((observed - np.mean(observed)) ** 2, dtype=np.float64)
    if not np.isfinite(total) or total <= np.finfo(np.float64).eps:
        return np.nan
    return float(1.0 - residual / total)


def fit_lifetime(data_type, time_series, time_points, params, model_type="single"):
    signal_values = np.asarray(time_series)
    times = np.asarray(time_points)
    if np.iscomplexobj(signal_values):
        raise ValueError("Lifetime fitting requires real-valued observations")
    if signal_values.ndim != 1 or times.ndim != 1 or signal_values.size != times.size:
        raise ValueError("Lifetime signal and time axis must be matching 1D arrays")
    if signal_values.size < 3:
        raise ValueError("Lifetime fitting requires at least three samples")
    if model_type not in {"single", "double"}:
        raise ValueError(f"Unsupported lifetime model: {model_type}")

    physical_signal = (
        np.abs(signal_values)
        if data_type in {"central negative", "central positive"}
        else signal_values
    )
    if np.any(~np.isfinite(physical_signal)) or np.any(~np.isfinite(times)):
        return _failed_result(model_type, physical_signal)

    max_index = int(np.argmax(physical_signal))
    if params["from_start_cal"]:
        decay_signal = physical_signal
        decay_time = times
    else:
        decay_signal = physical_signal[max_index:]
        decay_time = times[max_index:] - times[max_index]
    if decay_signal.size < 3:
        return _failed_result(model_type, physical_signal)

    peak_min, peak_max = params["peak_range"]
    if not peak_min <= max_index <= peak_max:
        size = 3 if model_type == "single" else 5
        lifetime = 0.0 if model_type == "single" else (0.0, 0.0)
        return LifetimeFitResult(np.zeros(size), lifetime, np.nan, physical_signal)

    span = float(decay_time[-1] - decay_time[0])
    if span <= 0:
        return _failed_result(model_type, physical_signal)
    amplitude_guess = float(np.max(decay_signal) - np.min(decay_signal))
    lifetime_guess = span / 5.0
    baseline_guess = float(np.min(decay_signal))
    tau_min, tau_max = params["tau_range"]

    try:
        if model_type == "single":
            fitted, _ = curve_fit(
                single_exponential,
                decay_time,
                decay_signal,
                p0=[amplitude_guess, lifetime_guess, baseline_guess],
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
            )
            tau = float(fitted[1])
            if not tau_min < tau < tau_max:
                return LifetimeFitResult(fitted, 0.0, np.nan, physical_signal)
            r_squared = _r_squared(decay_signal, single_exponential(decay_time, *fitted))
            lifetime = tau if np.isfinite(r_squared) and r_squared > params["r_squared_min"] else 0.0
            return LifetimeFitResult(fitted, lifetime, r_squared, physical_signal)

        fitted, _ = curve_fit(
            double_exponential,
            decay_time,
            decay_signal,
            p0=[amplitude_guess, lifetime_guess, amplitude_guess / 2.0, lifetime_guess * 2.0, baseline_guess],
            bounds=([0, 0, 10, 10, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]),
        )
        tau1, tau2 = float(fitted[1]), float(fitted[3])
        if not (tau_min < tau1 < tau_max or tau_min < tau2 < tau_max):
            return LifetimeFitResult(fitted, (0.0, 0.0), np.nan, physical_signal)
        r_squared = _r_squared(decay_signal, double_exponential(decay_time, *fitted))
        lifetime = (
            (tau1, tau2)
            if np.isfinite(r_squared) and r_squared > params["r_squared_min"]
            else (0.0, 0.0)
        )
        return LifetimeFitResult(fitted, lifetime, r_squared, physical_signal)
    except (RuntimeError, ValueError, TypeError, FloatingPointError):
        return _failed_result(model_type, physical_signal)


def has_correlated_window(time_series, time_points, threshold=0.8, max_window=10):
    values = np.asarray(time_series)
    times = np.asarray(time_points)
    if values.ndim != 1 or times.ndim != 1 or values.size != times.size:
        return False
    window_size = min(int(max_window), values.size // 2)
    if window_size < 2:
        return False

    for start in range(values.size - window_size + 1):
        value_window = values[start:start + window_size]
        time_window = times[start:start + window_size]
        if np.any(~np.isfinite(value_window)) or np.std(value_window) == 0:
            continue
        correlation = np.corrcoef(time_window, value_window)[0, 1]
        if np.isfinite(correlation) and abs(correlation) >= float(threshold):
            return True
    return False

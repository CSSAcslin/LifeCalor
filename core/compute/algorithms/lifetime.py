from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType

import numpy as np
from scipy.optimize import curve_fit


class LifetimeFitStatus(IntEnum):
    NOT_EVALUATED = 0
    SUCCESS = 1
    NO_CORRELATED_WINDOW = 2
    NONFINITE_INPUT = 3
    INVALID_WINDOW = 4
    PEAK_REJECTED = 5
    FIT_FAILED = 6
    TAU_REJECTED = 7
    R_SQUARED_REJECTED = 8


LIFETIME_SOLVER_CONTRACT = MappingProxyType({
    "solver": "scipy.optimize.curve_fit",
    "method": "trf",
    "compute_dtype": "float64",
    "ftol": 1e-8,
    "xtol": 1e-8,
    "gtol": 1e-8,
    "jacobian": "scipy_finite_difference",
    "single_bounds": ((-np.inf, 0.0, -np.inf), (np.inf, np.inf, np.inf)),
    "double_bounds": (
        (0.0, 0.0, 10.0, 10.0, -np.inf),
        (np.inf, np.inf, np.inf, np.inf, np.inf),
    ),
})


@dataclass(frozen=True)
class LifetimeFitResult:
    parameters: np.ndarray
    lifetime: object
    r_squared: float
    physical_signal: np.ndarray
    status: LifetimeFitStatus = LifetimeFitStatus.SUCCESS


def single_exponential(time, amplitude, lifetime, baseline):
    return amplitude * np.exp(-time / lifetime) + baseline


def double_exponential(time, amplitude1, lifetime1, amplitude2, lifetime2, baseline):
    return (
        amplitude1 * np.exp(-time / lifetime1)
        + amplitude2 * np.exp(-time / lifetime2)
        + baseline
    )


def lifetime_model_and_jacobian(time, parameters, *, model_type="single"):
    times = np.asarray(time, dtype=np.float64)
    values = np.asarray(parameters, dtype=np.float64)
    expected = 3 if model_type == "single" else 5 if model_type == "double" else 0
    if expected == 0:
        raise ValueError(f"Unsupported lifetime model: {model_type}")
    if times.ndim != 1 or values.shape != (expected,):
        raise ValueError(
            f"{model_type} lifetime Jacobian expects time (T,) and parameters ({expected},)"
        )

    if model_type == "single":
        amplitude, lifetime, baseline = values
        if lifetime <= 0:
            raise ValueError("Lifetime must be positive")
        exponential = np.exp(-times / lifetime)
        model = amplitude * exponential + baseline
        jacobian = np.column_stack((
            exponential,
            amplitude * exponential * times / lifetime ** 2,
            np.ones_like(times),
        ))
        return model, jacobian

    amplitude1, lifetime1, amplitude2, lifetime2, baseline = values
    if lifetime1 <= 0 or lifetime2 <= 0:
        raise ValueError("Lifetimes must be positive")
    exponential1 = np.exp(-times / lifetime1)
    exponential2 = np.exp(-times / lifetime2)
    model = amplitude1 * exponential1 + amplitude2 * exponential2 + baseline
    jacobian = np.column_stack((
        exponential1,
        amplitude1 * exponential1 * times / lifetime1 ** 2,
        exponential2,
        amplitude2 * exponential2 * times / lifetime2 ** 2,
        np.ones_like(times),
    ))
    return model, jacobian


def _failed_result(
    model_type,
    physical_signal,
    status=LifetimeFitStatus.FIT_FAILED,
):
    if model_type == "single":
        parameters = np.full(3, np.nan)
        lifetime = np.nan
    else:
        parameters = np.full(5, np.nan)
        lifetime = (np.nan, np.nan)
    return LifetimeFitResult(parameters, lifetime, np.nan, physical_signal, status)


def _interior_initial(value, lower):
    if not np.isfinite(lower):
        return float(value)
    margin = max(1e-9, abs(float(lower)) * 1e-9)
    return max(float(value), float(lower) + margin)


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
        return _failed_result(
            model_type, physical_signal, LifetimeFitStatus.NONFINITE_INPUT
        )

    max_index = int(np.argmax(physical_signal))
    if params["from_start_cal"]:
        decay_signal = physical_signal
        decay_time = times
    else:
        decay_signal = physical_signal[max_index:]
        decay_time = times[max_index:] - times[max_index]
    if decay_signal.size < 3:
        return _failed_result(
            model_type, physical_signal, LifetimeFitStatus.INVALID_WINDOW
        )

    peak_min, peak_max = params["peak_range"]
    if not peak_min <= max_index <= peak_max:
        size = 3 if model_type == "single" else 5
        lifetime = 0.0 if model_type == "single" else (0.0, 0.0)
        return LifetimeFitResult(
            np.zeros(size),
            lifetime,
            np.nan,
            physical_signal,
            LifetimeFitStatus.PEAK_REJECTED,
        )

    span = float(decay_time[-1] - decay_time[0])
    if span <= 0:
        return _failed_result(
            model_type, physical_signal, LifetimeFitStatus.INVALID_WINDOW
        )
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
                bounds=LIFETIME_SOLVER_CONTRACT["single_bounds"],
                method=LIFETIME_SOLVER_CONTRACT["method"],
                ftol=LIFETIME_SOLVER_CONTRACT["ftol"],
                xtol=LIFETIME_SOLVER_CONTRACT["xtol"],
                gtol=LIFETIME_SOLVER_CONTRACT["gtol"],
            )
            tau = float(fitted[1])
            if not tau_min < tau < tau_max:
                return LifetimeFitResult(
                    fitted,
                    0.0,
                    np.nan,
                    physical_signal,
                    LifetimeFitStatus.TAU_REJECTED,
                )
            r_squared = _r_squared(decay_signal, single_exponential(decay_time, *fitted))
            accepted = np.isfinite(r_squared) and r_squared > params["r_squared_min"]
            lifetime = tau if accepted else 0.0
            status = (
                LifetimeFitStatus.SUCCESS
                if accepted
                else LifetimeFitStatus.R_SQUARED_REJECTED
            )
            return LifetimeFitResult(
                fitted, lifetime, r_squared, physical_signal, status
            )

        double_lower, double_upper = LIFETIME_SOLVER_CONTRACT["double_bounds"]
        initial = [
            _interior_initial(amplitude_guess, double_lower[0]),
            _interior_initial(lifetime_guess, double_lower[1]),
            _interior_initial(amplitude_guess / 2.0, double_lower[2]),
            _interior_initial(lifetime_guess * 2.0, double_lower[3]),
            baseline_guess,
        ]
        fitted, _ = curve_fit(
            double_exponential,
            decay_time,
            decay_signal,
            p0=initial,
            bounds=(double_lower, double_upper),
            method=LIFETIME_SOLVER_CONTRACT["method"],
            ftol=LIFETIME_SOLVER_CONTRACT["ftol"],
            xtol=LIFETIME_SOLVER_CONTRACT["xtol"],
            gtol=LIFETIME_SOLVER_CONTRACT["gtol"],
        )
        tau1, tau2 = float(fitted[1]), float(fitted[3])
        if not (tau_min < tau1 < tau_max or tau_min < tau2 < tau_max):
            return LifetimeFitResult(
                fitted,
                (0.0, 0.0),
                np.nan,
                physical_signal,
                LifetimeFitStatus.TAU_REJECTED,
            )
        r_squared = _r_squared(decay_signal, double_exponential(decay_time, *fitted))
        accepted = np.isfinite(r_squared) and r_squared > params["r_squared_min"]
        lifetime = (tau1, tau2) if accepted else (0.0, 0.0)
        status = (
            LifetimeFitStatus.SUCCESS
            if accepted
            else LifetimeFitStatus.R_SQUARED_REJECTED
        )
        return LifetimeFitResult(
            fitted, lifetime, r_squared, physical_signal, status
        )
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

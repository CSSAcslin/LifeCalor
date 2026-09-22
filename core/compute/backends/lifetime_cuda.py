from __future__ import annotations

import numpy as np

from compute.algorithms.lifetime import (
    LIFETIME_SOLVER_CONTRACT,
    LifetimeFitStatus,
    has_correlated_window,
)


def _load_cupy():
    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError(
            f"CuPy/CUDA 寿命后端加载失败: {type(exc).__name__}: {exc}"
        ) from exc
    return cp


def _prepare_batch(block, time_points, data_type, fit_params, model_type):
    array = np.asarray(block)
    times = np.asarray(time_points, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != times.size:
        raise ValueError("寿命 CUDA 块必须为与时间轴匹配的 THW 数据")
    if np.iscomplexobj(array):
        raise ValueError("寿命拟合不接受复数输入")
    traces = array.reshape(array.shape[0], -1).T.astype(np.float64, copy=False)
    if data_type in {"central negative", "central positive"}:
        traces = np.abs(traces)
    count, length = traces.shape
    parameter_count = 3 if model_type == "single" else 5
    values = np.zeros((count, length), dtype=np.float64)
    relative_times = np.zeros((count, length), dtype=np.float64)
    mask = np.zeros((count, length), dtype=bool)
    initial = np.zeros((count, parameter_count), dtype=np.float64)
    status = np.full(
        count, int(LifetimeFitStatus.NOT_EVALUATED), dtype=np.uint8
    )
    peak_min, peak_max = fit_params["peak_range"]
    lower = np.asarray(
        LIFETIME_SOLVER_CONTRACT[
            "single_bounds" if model_type == "single" else "double_bounds"
        ][0],
        dtype=np.float64,
    )
    valid = []
    for index, trace in enumerate(traces):
        if np.any(~np.isfinite(trace)) or np.any(~np.isfinite(times)):
            status[index] = int(LifetimeFitStatus.NONFINITE_INPUT)
            continue
        if not has_correlated_window(trace, times):
            status[index] = int(LifetimeFitStatus.NO_CORRELATED_WINDOW)
            continue
        peak = int(np.argmax(trace))
        if not peak_min <= peak <= peak_max:
            status[index] = int(LifetimeFitStatus.PEAK_REJECTED)
            continue
        start = 0 if fit_params["from_start_cal"] else peak
        sample_count = length - start
        if sample_count < 3:
            status[index] = int(LifetimeFitStatus.INVALID_WINDOW)
            continue
        decay_time = times[start:] - (0.0 if start == 0 else times[start])
        if float(decay_time[-1] - decay_time[0]) <= 0:
            status[index] = int(LifetimeFitStatus.INVALID_WINDOW)
            continue
        decay = trace[start:]
        values[index, :sample_count] = decay
        relative_times[index, :sample_count] = decay_time
        mask[index, :sample_count] = True
        amplitude = float(np.max(decay) - np.min(decay))
        lifetime = float((decay_time[-1] - decay_time[0]) / 5.0)
        baseline = float(np.min(decay))
        if model_type == "single":
            guess = np.asarray([amplitude, lifetime, baseline])
        else:
            guess = np.asarray([
                amplitude,
                lifetime,
                amplitude / 2.0,
                lifetime * 2.0,
                baseline,
            ])
        finite_lower = np.isfinite(lower)
        margin = np.maximum(1e-9, np.abs(lower) * 1e-9)
        guess[finite_lower] = np.maximum(
            guess[finite_lower], lower[finite_lower] + margin[finite_lower]
        )
        initial[index] = guess
        valid.append(index)
    return values, relative_times, mask, initial, status, np.asarray(valid)


def _parameters_from_unconstrained(cp, unconstrained, lower):
    finite = cp.isfinite(lower)
    softplus = cp.maximum(unconstrained, 0) + cp.log1p(
        cp.exp(-cp.abs(unconstrained))
    )
    parameters = cp.where(finite, lower + softplus, unconstrained)
    derivative = cp.where(finite, 1.0 / (1.0 + cp.exp(-unconstrained)), 1.0)
    return parameters, derivative


def _unconstrained_from_parameters(cp, parameters, lower):
    finite = cp.isfinite(lower)
    distance = cp.maximum(parameters - lower, 1e-12)
    inverse = cp.where(
        distance > 20.0,
        distance,
        cp.log(cp.expm1(distance)),
    )
    return cp.where(finite, inverse, parameters)


def _model_jacobian(cp, times, parameters, model_type):
    if model_type == "single":
        amplitude, lifetime, baseline = (
            parameters[:, index:index + 1] for index in range(3)
        )
        exponential = cp.exp(-times / lifetime)
        model = amplitude * exponential + baseline
        jacobian = cp.stack((
            exponential,
            amplitude * exponential * times / lifetime ** 2,
            cp.ones_like(times),
        ), axis=-1)
        return model, jacobian
    amplitude1 = parameters[:, 0:1]
    lifetime1 = parameters[:, 1:2]
    amplitude2 = parameters[:, 2:3]
    lifetime2 = parameters[:, 3:4]
    baseline = parameters[:, 4:5]
    exponential1 = cp.exp(-times / lifetime1)
    exponential2 = cp.exp(-times / lifetime2)
    model = amplitude1 * exponential1 + amplitude2 * exponential2 + baseline
    jacobian = cp.stack((
        exponential1,
        amplitude1 * exponential1 * times / lifetime1 ** 2,
        exponential2,
        amplitude2 * exponential2 * times / lifetime2 ** 2,
        cp.ones_like(times),
    ), axis=-1)
    return model, jacobian


def _solve_batch(cp, values, times, mask, initial, lower, model_type):
    unconstrained = _unconstrained_from_parameters(cp, initial, lower)
    damping = cp.full(initial.shape[0], 1e-3, dtype=cp.float64)
    converged = cp.zeros(initial.shape[0], dtype=cp.bool_)
    mask_float = mask.astype(cp.float64)
    last_loss = cp.full(initial.shape[0], cp.inf, dtype=cp.float64)
    for _ in range(100):
        parameters, transform_derivative = _parameters_from_unconstrained(
            cp, unconstrained, lower
        )
        model, jacobian = _model_jacobian(
            cp, times, parameters, model_type
        )
        residual = (values - model) * mask_float
        loss = cp.sum(residual ** 2, axis=1)
        transformed_jacobian = (
            jacobian * transform_derivative[:, None, :] * mask_float[:, :, None]
        )
        gradient = cp.einsum("ntp,nt->np", transformed_jacobian, residual)
        hessian = cp.einsum(
            "ntp,ntq->npq", transformed_jacobian, transformed_jacobian
        )
        diagonal = cp.maximum(
            cp.diagonal(hessian, axis1=1, axis2=2), 1e-12
        )
        indices = cp.arange(initial.shape[1])
        hessian[:, indices, indices] += damping[:, None] * diagonal
        try:
            step = cp.linalg.solve(hessian, gradient[..., None])[..., 0]
        except Exception:
            step = cp.matmul(cp.linalg.pinv(hessian), gradient[..., None])[..., 0]
        candidate = unconstrained + step
        candidate_parameters, _ = _parameters_from_unconstrained(
            cp, candidate, lower
        )
        candidate_model, _ = _model_jacobian(
            cp, times, candidate_parameters, model_type
        )
        candidate_loss = cp.sum(
            ((values - candidate_model) * mask_float) ** 2, axis=1
        )
        accepted = candidate_loss < loss
        unconstrained = cp.where(accepted[:, None], candidate, unconstrained)
        damping = cp.where(
            accepted,
            cp.maximum(damping * 0.3, 1e-12),
            cp.minimum(damping * 10.0, 1e12),
        )
        step_small = cp.max(cp.abs(step), axis=1) <= (
            1e-8 * (1.0 + cp.max(cp.abs(unconstrained), axis=1))
        )
        loss_small = cp.abs(last_loss - candidate_loss) <= (
            1e-8 * (1.0 + candidate_loss)
        )
        gradient_small = cp.max(cp.abs(gradient), axis=1) <= 1e-8
        converged |= accepted & (step_small | loss_small | gradient_small)
        last_loss = cp.where(accepted, candidate_loss, loss)
        all_converged = cp.all(converged)
        if hasattr(all_converged, "get"):
            all_converged = all_converged.get()
        if bool(all_converged):
            break
    parameters, _ = _parameters_from_unconstrained(cp, unconstrained, lower)
    model, _ = _model_jacobian(cp, times, parameters, model_type)
    residual_sum = cp.sum(((values - model) * mask_float) ** 2, axis=1)
    sample_count = cp.sum(mask_float, axis=1)
    mean = cp.sum(values * mask_float, axis=1) / sample_count
    total = cp.sum(((values - mean[:, None]) * mask_float) ** 2, axis=1)
    r_squared = cp.where(total > cp.finfo(cp.float64).eps, 1 - residual_sum / total, cp.nan)
    return parameters, r_squared, converged


def lifetime_fit_block_cuda(
    block,
    time_points,
    data_type,
    fit_params,
    *,
    model_type="single",
    compute_dtype="float64",
    output_dtype="float64",
    device_index=0,
):
    if np.dtype(compute_dtype) != np.dtype("float64"):
        raise ValueError("寿命 CUDA 求解当前仅验证 float64")
    cp = _load_cupy()
    values, times, mask, initial, status, valid = _prepare_batch(
        block, time_points, data_type, fit_params, model_type
    )
    parameter_count = initial.shape[1]
    parameters = np.zeros_like(initial)
    r_squared = np.zeros(initial.shape[0], dtype=np.float64)
    fitted_mask = np.zeros(initial.shape[0], dtype=bool)
    if valid.size:
        bounds = LIFETIME_SOLVER_CONTRACT[
            "single_bounds" if model_type == "single" else "double_bounds"
        ]
        with cp.cuda.Device(int(device_index)):
            gpu_parameters, gpu_r_squared, converged = _solve_batch(
                cp,
                cp.asarray(values[valid]),
                cp.asarray(times[valid]),
                cp.asarray(mask[valid]),
                cp.asarray(initial[valid]),
                cp.asarray(bounds[0], dtype=cp.float64),
                model_type,
            )
            fitted = cp.asnumpy(gpu_parameters)
            scores = cp.asnumpy(gpu_r_squared)
            did_converge = cp.asnumpy(converged)
        parameters[valid] = fitted
        fitted_mask[valid] = did_converge
        r_squared[valid] = np.where(np.isfinite(scores), scores, 0.0)
        tau_min, tau_max = fit_params["tau_range"]
        for local, source_index in enumerate(valid):
            if not did_converge[local]:
                status[source_index] = int(LifetimeFitStatus.FIT_FAILED)
                continue
            taus = (
                (fitted[local, 1],)
                if model_type == "single"
                else (fitted[local, 1], fitted[local, 3])
            )
            if not any(tau_min < tau < tau_max for tau in taus):
                status[source_index] = int(LifetimeFitStatus.TAU_REJECTED)
            elif not (
                np.isfinite(scores[local])
                and scores[local] > fit_params["r_squared_min"]
            ):
                status[source_index] = int(
                    LifetimeFitStatus.R_SQUARED_REJECTED
                )
            else:
                status[source_index] = int(LifetimeFitStatus.SUCCESS)

    height, width = np.asarray(block).shape[1:]
    shape = (height, width)
    success = status == int(LifetimeFitStatus.SUCCESS)
    outputs = {
        "r_squared_map": r_squared.reshape(shape),
        "fit_status": status.reshape(shape),
    }
    if model_type == "single":
        outputs["lifetime_map"] = np.where(
            success, parameters[:, 1], 0.0
        ).reshape(shape).astype(output_dtype)
    else:
        names = (
            "amplitude1_map",
            "tau1_map",
            "amplitude2_map",
            "tau2_map",
            "baseline_map",
        )
        for index, name in enumerate(names):
            outputs[name] = np.where(
                fitted_mask, parameters[:, index], 0.0
            ).reshape(shape).astype(output_dtype)
    return outputs

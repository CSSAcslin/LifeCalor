from __future__ import annotations

import numpy as np

from compute.algorithms.cwt import build_cwt_kernel_bank, cwt_axes
from compute.algorithms.stft import select_frequency_indices
from compute.backends.lifetime_cuda import lifetime_fit_block_cuda


def _load_cuda():
    try:
        import cupy as cp
        from cupyx.scipy import signal as cupy_signal
    except Exception as exc:
        raise RuntimeError(f"CuPy/CUDA 后端加载失败: {type(exc).__name__}: {exc}") from exc
    return cp, cupy_signal


def cuda_self_test(device_index=0):
    cp, _ = _load_cuda()
    try:
        with cp.cuda.Device(int(device_index)):
            device = cp.cuda.Device()
            free_memory, total_memory = cp.cuda.runtime.memGetInfo()
            values = cp.arange(64, dtype=cp.float32)
            transformed = cp.fft.rfft(values)
            checksum = float(cp.asnumpy(cp.abs(transformed).sum()))
            properties = cp.cuda.runtime.getDeviceProperties(device.id)
            raw_name = properties.get("name", b"NVIDIA GPU")
            name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
            driver = str(cp.cuda.runtime.driverGetVersion())
            runtime = str(cp.cuda.runtime.runtimeGetVersion())
        return {
            "device_id": f"nvidia:{int(device_index)}",
            "name": name,
            "free_memory_bytes": int(free_memory),
            "total_memory_bytes": int(total_memory),
            "driver": driver,
            "runtime": runtime,
            "cupy": str(cp.__version__),
            "checksum": checksum,
        }
    except Exception as exc:
        raise RuntimeError(f"CUDA 快速自检失败: {type(exc).__name__}: {exc}") from exc


def stft_block_cuda(block, params, *, compute_dtype, output_dtype, device_index=0):
    cp, cupy_signal = _load_cuda()
    array = np.asarray(block)
    if array.ndim != 3:
        raise ValueError(f"STFT 计算块必须为 THW，实际 shape={array.shape}")
    time_length, height, width = array.shape
    traces = array.transpose(1, 2, 0).reshape(height * width, time_length)
    try:
        with cp.cuda.Device(int(device_index)):
            gpu_values = cp.asarray(traces, dtype=np.dtype(compute_dtype))
            gpu_window = cp.asarray(params["window"], dtype=gpu_values.real.dtype)
            real_input = not np.iscomplexobj(array)
            frequencies, times, coefficients = cupy_signal.stft(
                gpu_values,
                fs=params["fps"],
                window=gpu_window,
                nperseg=params["window_size"],
                noverlap=params["noverlap"],
                nfft=params["nfft"],
                return_onesided=real_input,
                scaling="spectrum",
                axis=-1,
            )
            host_frequencies = cp.asnumpy(frequencies)
            selected = select_frequency_indices(
                host_frequencies,
                params["target_freq"],
                params.get("scale_range", 0.0),
            )
            selected_coefficients = cp.take(coefficients, cp.asarray(selected), axis=-2)
            factors = np.ones(selected.size, dtype=np.float64)
            if real_input:
                selected_frequencies = host_frequencies[selected]
                edge = np.isclose(selected_frequencies, 0.0)
                if int(params["nfft"]) % 2 == 0:
                    edge |= np.isclose(
                        np.abs(selected_frequencies), float(params["fps"]) / 2.0
                    )
                factors[~edge] = 2.0
            factor_shape = (1,) * (selected_coefficients.ndim - 2) + (-1, 1)
            magnitude = cp.mean(
                cp.abs(selected_coefficients)
                * cp.asarray(
                    factors, dtype=selected_coefficients.real.dtype
                ).reshape(factor_shape),
                axis=-2,
            )
            output = magnitude.T.reshape(magnitude.shape[-1], height, width)
            host_output = cp.asnumpy(output.astype(np.dtype(output_dtype), copy=False))
            host_times = cp.asnumpy(times)
    except Exception:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        raise
    return host_output, host_frequencies, host_times, np.asarray(selected, dtype=np.intp)


def cwt_block_cuda(
    block,
    params,
    *,
    compute_dtype,
    output_dtype,
    device_index=0,
):
    cp, cupy_signal = _load_cuda()
    array = np.asarray(block)
    if array.ndim != 3:
        raise ValueError(f"CWT 计算块必须为 THW，实际 shape={array.shape}")
    if np.any(~np.isfinite(array)):
        raise ValueError(
            "CWT CUDA 原型暂不接受 NaN/Inf；请使用 CPU 参考路径以保持局部传播语义"
        )
    time_length, height, width = array.shape
    traces = array.transpose(1, 2, 0).reshape(height * width, time_length)
    scales, _ = cwt_axes(
        target_freq=params["target_freq"],
        scale_range=params.get("scale_range", 0.0),
        total_scales=params["total_scales"],
        wavelet=params["wavelet"],
        fps=params["fps"],
    )
    bank = build_cwt_kernel_bank(
        scales,
        params["wavelet"],
        np.dtype(compute_dtype),
        precision=int(params.get("wavelet_precision", 10)),
    )
    accumulator_dtype = (
        cp.float64 if np.dtype(output_dtype).itemsize > 4 else cp.float32
    )
    try:
        with cp.cuda.Device(int(device_index)):
            gpu_values = cp.asarray(traces, dtype=np.dtype(compute_dtype))
            magnitude_sum = cp.zeros(
                (traces.shape[0], time_length), dtype=accumulator_dtype
            )
            for scale, kernel in zip(bank.scales, bank.kernels):
                gpu_kernel = cp.asarray(kernel)[None, :]
                convolved = cupy_signal.fftconvolve(
                    gpu_values, gpu_kernel, mode="full", axes=-1
                )
                coefficients = -cp.sqrt(scale) * cp.diff(convolved, axis=-1)
                if np.dtype(bank.output_dtype).kind != "c":
                    coefficients = coefficients.real
                difference = (coefficients.shape[-1] - time_length) / 2.0
                if difference > 0:
                    left = int(np.floor(difference))
                    right = int(np.ceil(difference))
                    coefficients = coefficients[..., left:-right]
                elif difference < 0:
                    raise ValueError(f"Selected scale of {scale} too small.")
                magnitude_sum += (
                    2.0 * cp.abs(coefficients) / cp.sqrt(scale)
                ).astype(accumulator_dtype, copy=False)
            reduced = magnitude_sum / len(bank.scales)
            output = reduced.T.reshape(time_length, height, width)
            host_output = cp.asnumpy(
                output.astype(np.dtype(output_dtype), copy=False)
            )
    except Exception:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        raise
    return (
        host_output,
        np.asarray(bank.scales),
        np.asarray(bank.frequencies) * float(params["fps"]),
    )


def cwt_quality_trace_cuda(
    values,
    scales,
    wavelet,
    *,
    fps,
    compute_dtype,
    output_dtype,
    device_index=0,
):
    cp, cupy_signal = _load_cuda()
    data = np.asarray(values)
    if data.ndim != 1:
        raise ValueError(f"CWT 质量分析要求一维时间曲线，实际 shape={data.shape}")
    if np.any(~np.isfinite(data)):
        raise ValueError(
            "CWT CUDA 质量分析暂不接受 NaN/Inf，请使用 CPU 参考路径"
        )
    bank = build_cwt_kernel_bank(scales, wavelet, np.dtype(compute_dtype))
    rows = []
    with cp.cuda.Device(int(device_index)):
        gpu_values = cp.asarray(data, dtype=np.dtype(compute_dtype))[None, :]
        for scale, kernel in zip(bank.scales, bank.kernels):
            convolved = cupy_signal.fftconvolve(
                gpu_values, cp.asarray(kernel)[None, :], mode="full", axes=-1
            )
            coefficient = -cp.sqrt(scale) * cp.diff(convolved, axis=-1)
            if np.dtype(bank.output_dtype).kind != "c":
                coefficient = coefficient.real
            difference = (coefficient.shape[-1] - data.size) / 2.0
            if difference > 0:
                left = int(np.floor(difference))
                right = int(np.ceil(difference))
                coefficient = coefficient[..., left:-right]
            elif difference < 0:
                raise ValueError(f"Selected scale of {scale} too small.")
            rows.append(cp.abs(coefficient[0]))
        spectrum = cp.stack(rows, axis=0)
        host = cp.asnumpy(
            spectrum.astype(np.dtype(output_dtype), copy=False)
        )
    return host, np.asarray(bank.frequencies) * float(fps)


def lifetime_model_jacobian_cuda_prototype(
    time,
    parameters,
    *,
    model_type="single",
    device_index=0,
):
    cp, _ = _load_cuda()
    expected = 3 if model_type == "single" else 5 if model_type == "double" else 0
    if expected == 0:
        raise ValueError(f"Unsupported lifetime model: {model_type}")
    host_time = np.asarray(time, dtype=np.float64)
    host_parameters = np.asarray(parameters, dtype=np.float64)
    if host_time.ndim != 1 or host_parameters.shape != (expected,):
        raise ValueError(
            f"{model_type} lifetime prototype expects time (T,) and parameters ({expected},)"
        )
    with cp.cuda.Device(int(device_index)):
        times = cp.asarray(host_time)
        values = cp.asarray(host_parameters)
        if model_type == "single":
            amplitude, lifetime, baseline = values
            exponential = cp.exp(-times / lifetime)
            model = amplitude * exponential + baseline
            jacobian = cp.stack((
                exponential,
                amplitude * exponential * times / lifetime ** 2,
                cp.ones_like(times),
            ), axis=-1)
        else:
            amplitude1, lifetime1, amplitude2, lifetime2, baseline = values
            exponential1 = cp.exp(-times / lifetime1)
            exponential2 = cp.exp(-times / lifetime2)
            model = (
                amplitude1 * exponential1
                + amplitude2 * exponential2
                + baseline
            )
            jacobian = cp.stack((
                exponential1,
                amplitude1 * exponential1 * times / lifetime1 ** 2,
                exponential2,
                amplitude2 * exponential2 * times / lifetime2 ** 2,
                cp.ones_like(times),
            ), axis=-1)
        return cp.asnumpy(model), cp.asnumpy(jacobian)


def fft2_block_cuda(block, *, compute_dtype, output_dtype=None, device_index=0):
    cp, _ = _load_cuda()
    with cp.cuda.Device(int(device_index)):
        values = cp.asarray(block, dtype=np.dtype(compute_dtype))
        result = cp.fft.fft2(values, axes=(-2, -1))
        if output_dtype is not None:
            result = result.astype(np.dtype(output_dtype), copy=False)
        return cp.asnumpy(result)


def spatiotemporal_convolution_block_cuda_prototype(
    block,
    kernel,
    *,
    origin=0,
    compute_dtype="float32",
    output_dtype="float32",
    device_index=0,
):
    cp, _ = _load_cuda()
    try:
        from cupyx.scipy import ndimage as cupy_ndimage
    except Exception as exc:
        raise RuntimeError(
            f"CuPy ndimage 后端加载失败: {type(exc).__name__}: {exc}"
        ) from exc
    with cp.cuda.Device(int(device_index)):
        values = cp.asarray(block, dtype=np.dtype(compute_dtype))
        weights = cp.asarray(kernel, dtype=np.dtype(compute_dtype))
        result = cupy_ndimage.convolve(
            values, weights, mode="constant", cval=0.0, origin=origin
        )
        return cp.asnumpy(result.astype(np.dtype(output_dtype), copy=False))


_CUDA_HANDLERS = {
    "stft_block": stft_block_cuda,
    "cwt_block": cwt_block_cuda,
    "cwt_quality_trace": cwt_quality_trace_cuda,
    "lifetime_fit_block": lifetime_fit_block_cuda,
    "fft2_block": fft2_block_cuda,
}

_CUDA_PROTOTYPE_HANDLERS = {
    "lifetime_single_prototype": lifetime_model_jacobian_cuda_prototype,
    "lifetime_double_prototype": lifetime_model_jacobian_cuda_prototype,
    "spatiotemporal_convolution_block": spatiotemporal_convolution_block_cuda_prototype,
}


def cuda_capability_self_test(device_index=0):
    """Validate the runtime plus representative real and complex CWT kernels."""
    info = cuda_self_test(device_index)
    from compute.backends.cpu import cwt_block_cpu

    fps = 64
    time = np.arange(64, dtype=np.float32) / fps
    real_trace = np.sin(2.0 * np.pi * 8.0 * time).astype(np.float32)
    complex_trace = np.exp(2j * np.pi * 8.0 * time).astype(np.complex64)
    for wavelet, trace in (("morl", real_trace), ("cmor1-1.0", complex_trace)):
        params = {
            "target_freq": 8.0,
            "scale_range": 2.0,
            "total_scales": 3,
            "wavelet": wavelet,
            "fps": fps,
        }
        block = np.broadcast_to(trace[:, None, None], (64, 1, 1)).copy()
        expected, _, _ = cwt_block_cpu(
            block, params, compute_dtype=trace.dtype.name, output_dtype="float32"
        )
        actual, _, _ = cwt_block_cuda(
            block,
            params,
            compute_dtype=trace.dtype.name,
            output_dtype="float32",
            device_index=device_index,
        )
        scale = max(float(np.max(np.abs(expected))), 1.0)
        if not np.allclose(actual, expected, rtol=5e-4, atol=5e-6 * scale):
            difference = float(np.max(np.abs(actual - expected)))
            raise RuntimeError(
                f"CUDA CWT 数值自检失败: wavelet={wavelet}, max_error={difference}"
            )
    fit_params = {
        "from_start_cal": True,
        "r_squared_min": 0.4,
        "peak_range": (0, 200),
        "tau_range": (1e-3, 100.0),
    }
    lifetime_time = np.linspace(0.0, 30.0, 160)
    single = (
        8.0 * np.exp(-lifetime_time / 2.5) + 0.5
    ).reshape(-1, 1, 1)
    double = (
        50.0 * np.exp(-lifetime_time / 3.0)
        + 20.0 * np.exp(-lifetime_time / 15.0)
        + 2.0
    ).reshape(-1, 1, 1)
    single_result = lifetime_fit_block_cuda(
        single,
        lifetime_time,
        None,
        fit_params,
        model_type="single",
        device_index=device_index,
    )
    double_result = lifetime_fit_block_cuda(
        double,
        lifetime_time,
        None,
        fit_params,
        model_type="double",
        device_index=device_index,
    )
    if not np.isclose(single_result["lifetime_map"][0, 0], 2.5, rtol=1e-4):
        raise RuntimeError("CUDA 单指数寿命快速自检失败")
    if not np.allclose(
        sorted((
            double_result["tau1_map"][0, 0],
            double_result["tau2_map"][0, 0],
        )),
        (3.0, 15.0),
        rtol=1e-4,
    ):
        raise RuntimeError("CUDA 双指数寿命快速自检失败")
    info["supported_algorithms"] = (
        "stft",
        "cwt",
        "cwt_quality",
        "lifetime_single",
        "lifetime_double",
    )
    info["detail"] = "CUDA FFT 与 CWT 数值快速自检通过"
    return info


def get_cuda_handler(handler_id, *, allow_prototype=False):
    handlers = dict(_CUDA_HANDLERS)
    if allow_prototype:
        handlers.update(_CUDA_PROTOTYPE_HANDLERS)
    try:
        return handlers[str(handler_id)]
    except KeyError as exc:
        raise ValueError(f"未注册的 CUDA handler: {handler_id}") from exc


# Kept for the stage-A validation import path.
cwt_block_cuda_prototype = cwt_block_cuda

from __future__ import annotations

import numpy as np

from compute.algorithms.stft import select_frequency_indices


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

def fft2_block_cuda(block, *, compute_dtype, output_dtype=None, device_index=0):
    cp, _ = _load_cuda()
    with cp.cuda.Device(int(device_index)):
        values = cp.asarray(block, dtype=np.dtype(compute_dtype))
        result = cp.fft.fft2(values, axes=(-2, -1))
        if output_dtype is not None:
            result = result.astype(np.dtype(output_dtype), copy=False)
        return cp.asnumpy(result)

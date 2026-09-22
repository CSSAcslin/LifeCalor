from __future__ import annotations

import multiprocessing as mp
import time
import traceback

from tasks.model import CancellationToken, TaskCancelled

from .model import CapabilityStatus, DeviceCapability


class CudaBackendError(RuntimeError):
    def __init__(self, message, *, out_of_memory=False, details=""):
        super().__init__(message)
        self.out_of_memory = bool(out_of_memory)
        self.details = str(details or "")


def _cuda_worker_main(connection, device_index):
    try:
        while True:
            message = connection.recv()
            operation = message.get("operation")
            if operation == "shutdown":
                return
            try:
                from compute.backends.cuda import (
                    cuda_capability_self_test,
                    get_cuda_handler,
                )
                from compute.registry import get_algorithm_spec

                if operation == "self_test":
                    result = cuda_capability_self_test(device_index)
                elif operation == "execute":
                    spec = get_algorithm_spec(message["algorithm_id"])
                    if spec.cuda_handler is None:
                        raise ValueError(
                            f"{spec.algorithm_id} 尚无已验证的 CUDA handler"
                        )
                    handler = get_cuda_handler(spec.cuda_handler)
                    result = handler(
                        device_index=device_index,
                        **dict(message.get("payload") or {}),
                    )
                else:
                    raise ValueError(f"未知 CUDA worker 操作: {operation}")
                connection.send({"ok": True, "result": result})
            except BaseException as exc:
                details = traceback.format_exc()
                connection.send({
                    "ok": False,
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": details,
                    "out_of_memory": "OutOfMemory" in type(exc).__name__
                    or "out of memory" in str(exc).lower(),
                })
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class CudaWorkerClient:
    """One isolated CUDA process with task-aware cancellation polls."""

    def __init__(self, device_index=0):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_cuda_worker_main,
            args=(child, int(device_index)),
            name="LifeCalor-CUDA",
            daemon=True,
        )
        self._process.start()
        child.close()
        self.device_index = int(device_index)
        self._closed = False

    def request(self, operation, *, token=None, timeout=None, **payload):
        if self._closed or not self._process.is_alive():
            raise CudaBackendError("CUDA worker 未运行")
        token = token or CancellationToken()
        self._connection.send({"operation": operation, **payload})
        started = time.monotonic()
        while True:
            if token.is_cancelled:
                self.terminate()
                raise TaskCancelled("CUDA 任务已取消")
            if self._connection.poll(0.05):
                response = self._connection.recv()
                if response.get("ok"):
                    return response.get("result")
                raise CudaBackendError(
                    response.get("message", "CUDA worker 执行失败"),
                    out_of_memory=response.get("out_of_memory", False),
                    details=response.get("details", ""),
                )
            if timeout is not None and time.monotonic() - started > float(timeout):
                self.terminate()
                raise CudaBackendError(f"CUDA worker 超时（{float(timeout):.1f} 秒）")
            if not self._process.is_alive():
                raise CudaBackendError(
                    f"CUDA worker 异常退出，exitcode={self._process.exitcode}"
                )

    def stft(self, block, params, *, compute_dtype, output_dtype, token=None):
        return self.execute(
            "stft", token=token, block=block, params=params,
            compute_dtype=compute_dtype, output_dtype=output_dtype,
        )

    def execute(
        self,
        algorithm_id,
        *,
        task_id="",
        attempt_id=0,
        block_id=0,
        token=None,
        timeout=300.0,
        **payload,
    ):
        return self.request(
            "execute",
            algorithm_id=str(algorithm_id),
            task_id=str(task_id),
            attempt_id=int(attempt_id),
            block_id=str(block_id),
            payload=payload,
            token=token,
            timeout=timeout,
        )
    def self_test(self, timeout=15.0):
        return self.request("self_test", timeout=timeout)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                self._connection.send({"operation": "shutdown"})
                self._process.join(2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(2.0)
        finally:
            self._connection.close()

    def terminate(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(2.0)
        finally:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()
        return False


def probe_cuda_capability_isolated(device_index=0, timeout=45.0):
    try:
        with CudaWorkerClient(device_index) as worker:
            info = worker.self_test(timeout=timeout)
        return DeviceCapability(
            device_id=info["device_id"], kind="gpu", name=info["name"],
            vendor="NVIDIA", status=CapabilityStatus.AVAILABLE,
            total_memory_bytes=int(info["total_memory_bytes"]),
            free_memory_bytes=int(info["free_memory_bytes"]),
            driver=str(info["driver"]),
            backend=f"CuPy {info['cupy']} / CUDA {info['runtime']}",
            detail=str(info.get("detail") or "CUDA 快速自检通过"),
            supported_algorithms=tuple(info.get("supported_algorithms") or ("stft",)),
        )
    except Exception as exc:
        return DeviceCapability(
            device_id=f"nvidia:{int(device_index)}", kind="gpu", name="NVIDIA GPU",
            vendor="NVIDIA", status=CapabilityStatus.FAILED, detail=str(exc),
        )

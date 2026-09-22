import logging
import shutil
import uuid
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from scipy.ndimage import convolve
from scipy.optimize import curve_fit
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QElapsedTimer
from DataManager import *

from diagnostics import AppError, format_exception_details
from tasks import CancellationToken, TaskCancelled, TaskContextQueue, task_scoped
from compute import (
    BackendPreference, CapabilityStatus, ComputeRequest, PrecisionPolicy, ResourceBudget,
    compatibility_metadata, execution_metadata, plan_compute, resolve_precision,
)
from compute.algorithms import fit_lifetime, has_correlated_window
from compute.algorithms.lifetime_pipeline import run_lifetime_pipeline, spatial_kernel
from compute.capabilities import (
    mark_algorithm_verified,
    update_cached_device_capability,
)
from compute.worker import probe_cuda_capability_isolated


class LifetimeCalculator:
    """
    载流子寿命计算类（此类为静态方法类，不要放别的进来）
    """
    _cal_params = {
        'from_start_cal':False,
        'r_squared_min': 0.4,
        'peak_range': (0, 50),
        'tau_range': (1e-3, 1e2)
    }

    @classmethod
    def set_cal_parameters(cls,cal_set_params):
        """更新参数"""
        from_start_cal = cal_set_params['from_start_cal']
        r_squared_min = cal_set_params['r_squared_min']
        peak_range = (cal_set_params['peak_min'], cal_set_params['peak_max'])
        tau_range = (cal_set_params['tau_min'], cal_set_params['tau_max'])
        cls._cal_params['from_start_cal'] = from_start_cal
        cls._cal_params['r_squared_min'] = r_squared_min
        cls._cal_params['peak_range'] = peak_range
        cls._cal_params['tau_range'] = tau_range
        pass

    @staticmethod
    def single_exponential(t, A, tau, C):
        """单指数衰减模型"""
        return A * np.exp(-t / tau) + C

    @staticmethod
    def double_exponential(t, A1, tau1, A2, tau2, C):
        """双指数衰减模型"""
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

    @staticmethod
    def calculate_lifetime(data_type, time_series, time_points, arg_bundle=None, model_type='single'):
        """使用冻结后的 CPU 参考契约计算一条寿命曲线。"""
        # 兼容旧代码把 model_type 作为第 4 个位置参数传入的方式。
        if isinstance(arg_bundle, str) and model_type == 'single':
            model_type = arg_bundle
            arg_bundle = None
        params = arg_bundle if isinstance(arg_bundle, dict) else LifetimeCalculator._cal_params
        result = fit_lifetime(
            data_type,
            time_series,
            time_points,
            params,
            model_type=model_type,
        )
        return (
            result.parameters,
            result.lifetime,
            result.r_squared,
            result.physical_signal,
        )

    @staticmethod
    def analyze_region(data, time_points,mask, model_type='single'):
        """
        分析特定区域的载流子寿命

        参数:
            data: 3D numpy数组 (time, height, width)
            time_points: 时间点序列
            center: (y, x) 中心坐标
            shape: 'square' 或 'circle'
            size: 区域大小 (正方形边长或圆形半径)
            model_type: 衰减模型类型

        返回:
            avg_curve: 平均时间曲线
            lifetime: 计算得到的寿命
            fit_curve: 拟合曲线
        """
        global lifetime, fit_curve, phy_signal, r_squared
        data_type = ((getattr(data,'parameters')if isinstance(data, Data) else getattr(data,'out_processed')) or {}).get('data_type', None)

        # 计算区域平均时间曲线
        source = data.data_origin if isinstance(data, Data) else data.data_processed
        region_data = source[:, mask]
        avg_curve = np.mean(region_data, axis=1)

        # 计算寿命
        if model_type == 'single':
            popt, lifetime, r_squared, phy_signal = LifetimeCalculator.calculate_lifetime(data_type, avg_curve, time_points, model_type='single')
            if LifetimeCalculator._cal_params['from_start_cal']: # 从头算
                fit_curve = LifetimeCalculator.single_exponential(
                time_points, popt[0], popt[1], popt[2])
            else: # 从最大值算
                fit_curve = LifetimeCalculator.single_exponential(
                time_points[np.argmax(phy_signal):] - time_points[np.argmax(phy_signal)],
                popt[0], popt[1], popt[2])
        elif model_type == 'double':
            popt, lifetime, r_squared, phy_signal = LifetimeCalculator.calculate_lifetime(data_type, avg_curve, time_points, model_type='double')
            if LifetimeCalculator._cal_params['from_start_cal']: # 从头算
                fit_curve = LifetimeCalculator.double_exponential(
                time_points, popt[0], popt[1], popt[2], popt[3], popt[4])
            else: # 从最大值算
                fit_curve = LifetimeCalculator.double_exponential(
                time_points[np.argmax(phy_signal):] - time_points[np.argmax(phy_signal)],
                popt[0], popt[1], popt[2], popt[3], popt[4])
        return lifetime, fit_curve, phy_signal, r_squared

    @staticmethod
    def apply_custom_kernel(data, kernel_type='smooth',half_size = 2):
        """
         应用自定义卷积核
        参数:
            data: 输入数据（2D数组）
            kernel_type: 卷积核类型，可选 'smooth', 'gaussian', 'sharpen', 'edge', 'laplacian', 'average'
            half_size: 半卷积核大小（奇数，默认2）

        返回:
            卷积后的数据
        """
        # 确保卷积核大小为奇数
        size = half_size * 2 - 1
        if kernel_type == 'smooth':
            if size == 3:
                kernel = np.array([
                    [0.1, 0.1, 0.1],
                    [0.1, 0.2, 0.1],
                    [0.1, 0.1, 0.1]
                ])
            else:
                kernel = np.zeros((size, size))
                center = size // 2

                # 计算最大距离（从中心到角落）
                max_dist = np.sqrt(2 * (center ** 2))

                # 填充核值
                for i in range(size):
                    for j in range(size):
                        # 计算到中心的距离
                        dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)

                        # 线性插值计算权重
                        if dist == 0:  # 中心点
                            weight = 2
                        else:
                            # 从中心权重线性递减到边缘权重
                            weight = 2 - 1 * (dist / max_dist)

                        kernel[i, j] = weight

                # 归一化核，使所有元素和为1
                kernel /= np.sum(kernel)

        elif kernel_type == 'gaussian':
            # 高斯核
            sigma = max(0.3 * ((size - 1) * 0.5 - 1) + 0.8, 0.1)  # 自动计算sigma

            ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
            kernel = kernel / np.sum(kernel)  # 归一化

        elif kernel_type == 'sharpen':
            # 锐化核
            if size == 3:
                kernel = np.array([
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]
                ])
            else:
                # 对于其他尺寸，创建中心突出、周围负值的核
                kernel = -np.ones((size, size)) / (size * size - 1)
                center = size // 2
                kernel[center, center] = 2  # 中心权重

        elif kernel_type == 'edge':
            # 边缘检测核（Sobel变体）
            if size == 3:
                kernel = np.array([
                    [-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]
                ])
            else:
                # 对于大尺寸，使用拉普拉斯近似
                kernel = np.ones((size, size))
                center = size // 2
                kernel[center, center] = - (size * size - 1)

        elif kernel_type == 'laplacian':
            # 拉普拉斯算子（二阶微分）
            if size == 3:
                kernel = np.array([
                    [0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]
                ])
            else:
                # 扩展的拉普拉斯核
                kernel = np.zeros((size, size))
                center = size // 2
                # 设置中心值
                kernel[center, center] = -4
                # 设置四邻域
                if center > 0:
                    kernel[center - 1, center] = 1
                    kernel[center + 1, center] = 1
                    kernel[center, center - 1] = 1
                    kernel[center, center + 1] = 1

        elif kernel_type == 'average':
            kernel = np.ones((size, size)) / (size * size)

        else:  # 默认不处理
            return data

        # 边界处理采用镜像模式
        try:
            return convolve(data, kernel, mode='mirror')
        except Exception as e:
            logging.error(f"卷积操作失败: {e}")
            return data

    @staticmethod
    def gaussian_func(x, a, mu, sigma):
        """高斯函数"""
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def gaussian_fit(x, y):
        """高斯拟合"""
        # 初始猜测参数 [幅值, 均值, 标准差]
        a_guess = max(abs(y))
        mu_guess = np.median(x)
        sigma_guess = (max(x) - min(x)) / 4

        popt, pcov = curve_fit(
            LifetimeCalculator.gaussian_func,
            x, y,
            p0=[a_guess, mu_guess, sigma_guess]
        )
        return popt, pcov


class CalculationThread(QObject):
    """仅在线程中使用，目前未加锁（仍无必要）"""
    processed_result = pyqtSignal(ProcessedData)
    calculating_progress_signal = pyqtSignal(int, int)
    stop_thread_signal = pyqtSignal()
    update_status = pyqtSignal(str,str)
    processing_error_signal = pyqtSignal(object)
    processing_cancelled_signal = pyqtSignal()
    compute_plan_signal = pyqtSignal(object)


    def __init__(self):
        super().__init__()
        logging.info('计算线程已载入')
        self._is_calculating = False
        self.cancellation_token = CancellationToken()
        self.task_id = None
        self._task_contexts = TaskContextQueue()

    def set_cancellation_token(self, token):
        self.cancellation_token = token or CancellationToken()
        self._is_calculating = True
    def set_task_context(self, task_id, token=None):
        self.task_id = str(task_id or "") or None
        if token is not None:
            self.cancellation_token = token

    def enqueue_task_context(self, task_id, token=None):
        return self._task_contexts.enqueue(task_id, token)

    def _activate_task_context(self):
        context = self._task_contexts.next()
        if context is None:
            return True
        self.task_id = context.task_id or None
        self.cancellation_token = context.token
        self._is_calculating = True
        if context.token.is_cancelled:
            self.processing_cancelled_signal.emit()
            return False
        return True

    def cancel(self):
        self._is_calculating = False
        self.cancellation_token.cancel()

    def _raise_if_cancelled(self):
        self.cancellation_token.raise_if_cancelled()
        if not self._is_calculating:
            raise TaskCancelled("寿命计算已取消")

    def _report_failure(self, title, stage, exc, data=None):
        self.processing_error_signal.emit(AppError(
            title, str(exc), stage=stage, severity="error", original=exc,
            details=format_exception_details(exc, stage, data),
            context={"data": data} if data is not None else {},
            task_id=self.task_id,
        ))

    def _run_planned_lifetime(
        self,
        data,
        time_points,
        model_type,
        pre_cov,
        pre_size,
        is_multipro,
        cpu_num,
        compute_options,
    ):
        options = dict(compute_options or {})
        values = object.__getattribute__(data, "__dict__")
        storage_name = (
            "_data_origin_storage" if isinstance(data, Data)
            else "_data_processed_storage"
        )
        source = values.get(storage_name)
        if source is None:
            source = data.data_origin if isinstance(data, Data) else data.data_processed
        shape = tuple(int(value) for value in source.shape)
        if len(shape) != 3:
            raise ValueError(f"寿命热图要求 THW 三维数据，实际 shape={shape}")
        algorithm = "lifetime_single" if model_type == "single" else "lifetime_double"
        backend = BackendPreference(options.get("backend", "auto"))
        precision = PrecisionPolicy(options.get("precision", "compatibility"))
        precision_info = resolve_precision(algorithm, source.dtype, precision)
        workspace_per_pixel = shape[0] * (
            np.dtype(source.dtype).itemsize
            + np.dtype(precision_info.compute_dtype).itemsize
        )
        request = ComputeRequest(
            task_id=self.task_id or f"lifetime-{uuid.uuid4().hex}",
            attempt_id=0,
            algorithm=algorithm,
            data_id=str(getattr(data, "timestamp", getattr(data, "name", "data"))),
            shape=shape,
            dtype=source.dtype,
            axes="THW",
            source=source,
            parameters={
                "output_shape": shape[1:],
                "workspace_bytes_per_spatial_item": workspace_per_pixel,
                "max_spatial_items": 64,
                "model_type": model_type,
                "output_multiplier": 3 if model_type == "single" else 7,
            },
            backend=backend,
            precision=precision,
        )
        cache_dir = Path(
            options.get("cache_directory") or get_array_store().config.cache_dir
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        preferred_device = str(options.get("preferred_device", "") or "")
        try:
            device_index = (
                int(preferred_device.rsplit(":", 1)[-1])
                if preferred_device else 0
            )
        except ValueError:
            device_index = 0
        capabilities = list(options.get("capabilities", ()))
        capability = next(
            (
                device for device in capabilities
                if device.kind == "gpu"
                and device.status is CapabilityStatus.AVAILABLE
                and algorithm in device.supported_algorithms
                and (
                    not preferred_device
                    or device.device_id == preferred_device
                )
            ),
            None,
        )
        if backend is BackendPreference.GPU and capability is None:
            capability = probe_cuda_capability_isolated(
                device_index, timeout=45.0
            )
            update_cached_device_capability(capability)
            capabilities = [
                device for device in capabilities
                if device.device_id != capability.device_id
            ]
            capabilities.append(capability)
            if capability.status is not CapabilityStatus.AVAILABLE:
                logging.warning(
                    "CUDA 寿命拟合自检未通过: algorithm=%s detail=%s",
                    algorithm,
                    capability.detail,
                )
        device_limit = 0
        if (
            capability is not None
            and capability.status is CapabilityStatus.AVAILABLE
        ):
            percent = min(
                90, max(10, int(options.get("gpu_memory_percent", 70)))
            )
            reserve = max(
                1024 ** 3, int(capability.total_memory_bytes * 0.1)
            )
            available = max(0, capability.free_memory_bytes - reserve)
            device_limit = min(
                int(capability.total_memory_bytes * percent / 100),
                available,
            )
        effective_workers = (
            max(1, int(options.get("cpu_workers", cpu_num or 1)))
            if is_multipro else 1
        )
        budget = ResourceBudget(
            host_limit_bytes=max(
                256, int(options.get("host_memory_limit_mb", 4096))
            ) * 1024 ** 2,
            device_limit_bytes=device_limit,
            disk_free_bytes=int(shutil.disk_usage(cache_dir).free),
            cpu_workers=effective_workers,
        )
        plan = plan_compute(
            request,
            budget,
            capabilities=tuple(capabilities),
            default_backend=BackendPreference.AUTO,
            allow_cpu_fallback=bool(options.get("allow_cpu_fallback", True)),
            disk_output_threshold_bytes=max(
                1, int(options.get("cache_threshold_mb", 512))
            ) * 1024 ** 2,
        )
        self.compute_plan_signal.emit(plan)
        source_metadata = data.out_processed if isinstance(data, ProcessedData) else data.parameters
        data_type = (source_metadata or {}).get("data_type")
        kernel = spatial_kernel(pre_cov, pre_size) if pre_cov is not None else None
        workers = max(1, int(plan.cpu_workers))

        def report_progress(current, total, message):
            self._raise_if_cancelled()
            self.calculating_progress_signal.emit(int(current), int(total))

        result = run_lifetime_pipeline(
            plan,
            data_type=data_type,
            time_points=time_points,
            fit_params=LifetimeCalculator._cal_params.copy(),
            model_type=model_type,
            pre_kernel=kernel,
            cpu_workers=workers,
            token=self.cancellation_token,
            progress=report_progress,
            cache_dir=cache_dir,
            allow_cpu_fallback=bool(
                options.get("allow_cpu_fallback", True)
            ),
            device_index=device_index,
        )
        if result.plan.actual_backend == "gpu":
            verified = mark_algorithm_verified(
                f"nvidia:{device_index}", algorithm
            )
            logging.info(
                "GPU 算法实际任务验证通过: algorithm=%s device=%s "
                "backend=%s model=%s",
                algorithm,
                (
                    verified.name
                    if verified is not None
                    else f"nvidia:{device_index}"
                ),
                (
                    verified.backend
                    if verified is not None
                    else "CuPy/CUDA"
                ),
                model_type,
            )
        if result.plan is not plan:
            self.compute_plan_signal.emit(result.plan)
        metadata = execution_metadata(
            result.plan,
            execution=(
                "bounded_multiprocess"
                if result.plan.actual_backend == "cpu" and workers > 1
                else f"bounded_{result.plan.actual_backend}"
            ),
            model=model_type,
            cpu_workers=workers,
            pre_convolution=pre_cov or "none",
            fallback_reason=result.fallback_reason,
            max_gpu_oom_retries=3,
        )
        return result, metadata
    @pyqtSlot(object, float, np.ndarray, str)
    @task_scoped
    def region_analyze(self,data,time_unit,mask,model_type):
        """分析选定区域"""
        logging.info("开始计算选区载流子寿命...")
        self.calculating_progress_signal.emit(1, 3)
        try:
            # 获取参数
            time_points = data.time_point * time_unit
            self.calculating_progress_signal.emit(2, 3)

            # 执行区域分析
            lifetime, fit_curve, phy_signal, r_squared = LifetimeCalculator.analyze_region(
                data, time_points,mask, model_type)

            self.processed_result.emit(ProcessedData(data.timestamp,
                                                       f'{data.name}@r-lft',
                                                       'ROI_lifetime',
                                                       time_point=time_points,
                                                       data_processed=fit_curve,
                                                       out_processed={'phy_signal': phy_signal,
                                                                      'lifetime': lifetime,
                                                                      'fit_curve': fit_curve,
                                                                      'r_squared': r_squared,
                                                                      'model_type': model_type,
                                                                      'boundary': {'min':data.datamin, 'max':data.datamax},
                                                                      **(data.out_processed if isinstance(data,ProcessedData) else data.parameters),
                                                                      'compute': compatibility_metadata(
                                                                          'lifetime',
                                                                          (data.data_processed if isinstance(data, ProcessedData) else data.data_origin).dtype,
                                                                          backend='cpu', execution='curve_fit', model=model_type,
                                                                      )}))
            logging.info("计算完成!")
            self.calculating_progress_signal.emit(3, 3)
        except TaskCancelled:
            self.processing_cancelled_signal.emit()
        except Exception as e:
            self._report_failure("寿命计算失败", "寿命计算", e, data)
        finally:
            self._is_calculating = False
            self.stop_thread_signal.emit()

    @pyqtSlot(object, float, str, str, int, str, int, bool, int, object)
    @task_scoped
    def distribution_analyze(
        self,
        data,
        time_unit,
        model_type,
        pre_cov=None,
        pre_size=None,
        post_cov=None,
        post_size=None,
        is_multipro=False,
        cpu_num=0,
        compute_options=None,
    ):
        """Analyze the full lifetime map with bounded task-owned spatial blocks."""
        self._is_calculating = True
        try:
            time_points = data.time_point * time_unit
            result, metadata = self._run_planned_lifetime(
                data,
                time_points,
                model_type,
                pre_cov,
                pre_size,
                is_multipro,
                cpu_num,
                compute_options,
            )
            lifetime_map = result.lifetime_map
            r_squared_map = result.r_squared_map
            named_outputs = dict(result.named_outputs)
            if post_cov is not None:
                lifetime_values = (
                    lifetime_map.load(mmap_mode="r")
                    if hasattr(lifetime_map, "load")
                    else lifetime_map
                )
                lifetime_map_cov = LifetimeCalculator.apply_custom_kernel(
                    lifetime_values, post_cov, post_size
                )
                primary_field = (
                    "lifetime_map" if model_type == "single" else "tau1_map"
                )
                named_outputs[primary_field] = lifetime_map_cov
                if model_type == "double":
                    tau2_values = named_outputs["tau2_map"]
                    if hasattr(tau2_values, "load"):
                        tau2_values = tau2_values.load(mmap_mode="r")
                    named_outputs["tau2_map"] = (
                        LifetimeCalculator.apply_custom_kernel(
                            tau2_values, post_cov, post_size
                        )
                    )
                logging.info("后卷积完成")
            else:
                lifetime_map_cov = lifetime_map
            inherited = data.out_processed if isinstance(data, ProcessedData) else data.parameters
            result_fields = (
                ("lifetime_map", "r_squared_map", "fit_status")
                if model_type == "single"
                else (
                    "tau1_map",
                    "tau2_map",
                    "amplitude1_map",
                    "amplitude2_map",
                    "baseline_map",
                    "r_squared_map",
                    "fit_status",
                )
            )
            self.processed_result.emit(ProcessedData(
                data.timestamp,
                f"{data.name}@d-lft",
                "lifetime_distribution",
                time_point=np.array([0]),
                data_processed=lifetime_map_cov,
                out_processed={
                    **(inherited or {}),
                    **named_outputs,
                    "model_type": model_type,
                    "result_fields": result_fields,
                    "active_result_field": result_fields[0],
                    "compute": metadata,
                },
            ))
        except TaskCancelled:
            self.processing_cancelled_signal.emit()
        except Exception as exc:
            self._report_failure("寿命计算失败", "寿命计算", exc, data)
        finally:
            self._is_calculating = False
            self.stop_thread_signal.emit()
    @pyqtSlot(object, float, float, float, str)
    @task_scoped
    def diffusion_calculation(self,frame_data,time_unit,space_unit,timestamp,name):
        # 存储拟合方差结果 [时间, 方差]
        self._is_calculating = True

        sigma_results = np.zeros((2, len(frame_data)))
        time_series = []
        loading_bar_value =0 #进度条
        total_l = len(frame_data)
        fitting_result = []
        signal_series = []
        if self._is_calculating:  # 线程关闭控制（目前仅针对循环计算）
            for i, (frame_idx, data) in enumerate(frame_data.items()):
                if self.cancellation_token.is_cancelled or not self._is_calculating:
                    self.processing_cancelled_signal.emit()
                    return False
                positions = data[:, 0] * space_unit # 此处合并单位长度
                intensities = data[:, 1]

                # 高斯拟合
                try:
                    popt, pcov = LifetimeCalculator.gaussian_fit(positions, intensities)
                    fit_curve = LifetimeCalculator.gaussian_func(positions, *popt)

                    # 保存拟合结果
                    sigma_results[0, i] = frame_idx * time_unit # 此处合并单位时间
                    sigma_results[1, i] = popt[2] ** 2  # 保存方差
                    time_series.append(frame_idx * time_unit)  # 时间点单独算
                    fitting_result.append(np.stack((positions,fit_curve),axis=0))
                    signal_series.append(np.stack((positions,intensities),axis=0))
                except Exception as e:
                    logging.error(f"拟合失败,报错{e}")
                    self.update_status.emit(f'扩散拟合失败:{e}', 'error')
                    self.stop_thread_signal.emit()
                loading_bar_value += 1
                self.calculating_progress_signal.emit(loading_bar_value, total_l)
        else:
            logging.info("计算终止")
            self.calculating_progress_signal.emit(total_l, total_l)  # 进度条更新
            self.stop_thread_signal.emit()  # 目前来说，计算终止也会关闭线程，后续可考虑分开命令
            return

        dif_data_dict = {
            'sigma': np.stack(sigma_results, axis=0),
            'signal': np.stack(signal_series, axis=0),
            'fitting': np.stack(fitting_result, axis=0),
            'time_series': np.stack(time_series, axis=0),
        }
        self.processed_result.emit(ProcessedData(timestamp,
                                                 f'{name}@dif',
                                                 'diffusion',
                                                 time_point=dif_data_dict['time_series'],
                                                 data_processed=dif_data_dict['signal'],
                                                 out_processed=dif_data_dict))
        self._is_calculating = False
        self.stop_thread_signal.emit()

    @pyqtSlot(object, float, str, str, int, str, int, bool, int, object)
    @task_scoped
    def heat_transfer_calculation(
        self,
        data,
        time_unit,
        model_type,
        pre_cov=None,
        pre_size=None,
        post_cov=None,
        post_size=None,
        is_multipro=False,
        cpu_num=0,
        compute_options=None,
    ):
        """Calculate heat transfer from an existing or newly fitted lifetime map."""
        self._is_calculating = True
        try:
            metadata = None
            if isinstance(data, ProcessedData) and data.type_processed == "lifetime_distribution":
                lifetime_map = data.data_processed
                r_squared_map = data.out_processed.get("r_squared_map")
            else:
                result, metadata = self._run_planned_lifetime(
                    data,
                    data.time_point * time_unit,
                    model_type,
                    pre_cov,
                    pre_size,
                    is_multipro,
                    cpu_num,
                    compute_options,
                )
                lifetime_map = result.lifetime_map
                r_squared_map = result.r_squared_map

            if hasattr(lifetime_map, "load"):
                lifetime_map = lifetime_map.load(mmap_mode="r")
            with np.errstate(divide="ignore", invalid="ignore"):
                heat_transfer = np.where(lifetime_map >= 0.1, 42.72 / lifetime_map, 0)
            heat_transfer_cov = (
                LifetimeCalculator.apply_custom_kernel(heat_transfer, post_cov, post_size)
                if post_cov is not None
                else heat_transfer
            )
            inherited = data.out_processed if isinstance(data, ProcessedData) else data.parameters
            out_processed = {
                "heat_transfer_map": heat_transfer_cov,
                "r_squared_map": r_squared_map,
                **(inherited or {}),
            }
            if metadata is not None:
                out_processed["compute"] = {
                    **metadata,
                    "derived_result": "heat_transfer",
                }
            self.processed_result.emit(ProcessedData(
                data.timestamp,
                f"{data.name}@heat",
                "heat_transfer",
                time_point=np.array([0]),
                data_processed=heat_transfer_cov,
                out_processed=out_processed,
            ))
            return True
        except TaskCancelled:
            self.processing_cancelled_signal.emit()
        except Exception as exc:
            self._report_failure("传热计算失败", "传热计算", exc, data)
        finally:
            self._is_calculating = False
            self.stop_thread_signal.emit()
    @pyqtSlot(object,str, object)
    @task_scoped
    def easy_process(self,data,ptype, mask = None):
        if isinstance(data, ProcessedData):
            origin_data = data.data_processed
        else:
            origin_data = data.data_origin
        time_point = data.time_point
        name = f'{data.name}@avg'
        if mask is not None: # 内置mask是仅仅用于快速计算的
            origin_data = np.where(mask[np.newaxis,:,:],origin_data,0)
            name = f'{data.name}@ROI_avg'
        if ptype in ['avg', 'mean']:
            data_processed = np.mean(origin_data, axis=(1, 2))
            new_data = ProcessedData(data.timestamp,
                                     name,
                                     'signal_average',
                                     time_point=time_point,
                                     data_processed=np.column_stack((time_point, data_processed)),)
        else:
            return False

        self.processed_result.emit(new_data)
        return True

    def lifetime_map_cal(self, aim_data, data_type, time_points, model_type):
        """Compatibility helper for direct single-exponential map calls."""
        try:
            if model_type != 'single':
                raise ValueError(
                    "该兼容接口只返回单个寿命图；双指数请使用具名多结果流水线。"
                )

            values = np.asarray(aim_data)
            times = np.asarray(time_points)
            if values.ndim != 3 or values.shape[0] != times.size:
                raise ValueError("寿命热图输入必须为与时间轴匹配的 THW 数据")

            _, height, width = values.shape
            lifetime_map = np.zeros((height, width), dtype=np.float64)
            r_squared_map = np.zeros((height, width), dtype=np.float64)
            logging.info("开始拟合热图...")

            completed = 0
            total = height * width
            for row in range(height):
                self._raise_if_cancelled()
                for column in range(width):
                    time_series = values[:, row, column]
                    lifetime = 0.0
                    r_squared = 0.0
                    if has_correlated_window(time_series, times):
                        _, lifetime, r_squared, _ = LifetimeCalculator.calculate_lifetime(
                            data_type,
                            time_series,
                            times,
                            model_type=model_type,
                        )

                    if np.isfinite(lifetime):
                        lifetime_map[row, column] = lifetime
                    if np.isfinite(r_squared):
                        r_squared_map[row, column] = r_squared
                    completed += 1
                    self.calculating_progress_signal.emit(completed, total)

            logging.info("计算完成!")
            return lifetime_map, r_squared_map
        except Exception as exc:
            return exc
        finally:
            self.stop_thread_signal.emit()

    def stop(self):
        self._is_calculating = False

    def force_stop(self):
        self._is_calculating = False
        self.stop_thread_signal.emit()

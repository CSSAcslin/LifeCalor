import math
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from scipy.ndimage import convolve
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QElapsedTimer
from DataManager import *
from multiprocessing import shared_memory, Pool
import traceback


# --- 必须放在类外的 Worker 函数 ---
def _lifetime_fit_worker(
        shm_in_name, shape_in, dtype_in,
        shm_out_name, shape_out, dtype_out,
        row_range,
        time_points,
        data_type, model_type, fit_params  # 显式传入参数
):
    """
    独立进程运行的寿命拟合函数
    """
    # 1. 连接共享内存
    existing_shm_in = shared_memory.SharedMemory(name=shm_in_name)
    existing_shm_out = shared_memory.SharedMemory(name=shm_out_name)

    try:
        # 2. 重构数组
        # 输入: [Time, Height, Width]
        aim_data = np.ndarray(shape_in, dtype=dtype_in, buffer=existing_shm_in.buf)
        # 输出: [Height, Width]
        lifetime_map = np.ndarray(shape_out, dtype=dtype_out, buffer=existing_shm_out.buf)

        start_row, end_row = row_range
        height, width = shape_out

        # 3. 参数解包
        r_squared_min = fit_params['r_squared_min']
        peak_range = fit_params['peak_range']
        tau_range = fit_params['tau_range']
        from_start_cal = fit_params['from_start_cal']

        # 4. 逐像素计算 (仅计算分配到的行)
        # 这里的循环是在 C 级别的进程中运行，利用多核并行
        processed_count = 0

        for i in range(start_row, end_row):
            for j in range(width):
                try:
                    time_series = aim_data[:, i, j]

                    # --- A. Pearson 校验 (保留原有逻辑) ---
                    # 优化：提前判断全零或无效数据
                    if np.max(np.abs(time_series)) < 1e-6:
                        lifetime_map[i, j] = 0
                        continue

                    window_size = min(10, len(time_points) // 2)
                    is_valid_signal = False

                    # 快速检查：如果没有明显的衰减趋势，也许可以跳过？
                    # 这里保留你的滑动窗口逻辑，但它其实挺慢的，建议将来考虑优化
                    pr_list = []
                    for k in range(len(time_series) - window_size):
                        window = time_series[k:k + window_size]
                        time_window = time_points[k:k + window_size]
                        # 注意：pearsonr 在常数输入下会报警告或返回 nan
                        if np.std(window) == 0:
                            r = 0
                        else:
                            r, _ = pearsonr(time_window, window)

                        if abs(r) >= 0.8:
                            is_valid_signal = True
                            break  # 只要有一段满足，就认为有效，直接跳出循环节省时间

                    if not is_valid_signal:
                        lifetime_map[i, j] = 0
                        continue

                    # --- B. 准备拟合数据 ---
                    # 逻辑复刻 LifetimeCalculator.calculate_lifetime
                    if data_type in ['central negative', 'central positive']:
                        phy_signal = np.abs(time_series)
                    else:
                        phy_signal = time_series

                    max_idx = np.argmax(phy_signal)

                    if not from_start_cal:
                        decay_signal = phy_signal[max_idx:]
                        decay_time = time_points[max_idx:] - time_points[max_idx]
                    else:
                        decay_signal = phy_signal
                        decay_time = time_points

                    # 极短数据保护
                    if len(decay_signal) < 3:
                        lifetime_map[i, j] = 0
                        continue

                    # --- C. 执行 curve_fit ---
                    A_guess = np.max(decay_signal) - np.min(decay_signal)
                    tau_guess = (decay_time[-1] - decay_time[0]) / 5 if (decay_time[-1] - decay_time[0]) != 0 else 1
                    C_guess = np.min(decay_signal)

                    lifetime = 0

                    # 单指数
                    if model_type == 'single':
                        if peak_range[0] <= max_idx <= peak_range[1]:
                            try:
                                popt, _ = curve_fit(
                                    _single_exp_func,  # 使用模块级函数，避免 pickle 问题
                                    decay_time,
                                    decay_signal,
                                    p0=[A_guess, tau_guess, C_guess],
                                    bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                                    maxfev=1000  # 限制迭代次数防止死循环
                                )
                                tau = popt[1]
                                if tau_range[0] < tau < tau_range[1]:
                                    # R方校验
                                    y_pred = _single_exp_func(decay_time, *popt)
                                    ss_res = np.sum((decay_signal - y_pred) ** 2)
                                    ss_tot = np.sum((decay_signal - np.mean(decay_signal)) ** 2)
                                    if ss_tot != 0:
                                        r2 = 1 - (ss_res / ss_tot)
                                        if r2 > r_squared_min:
                                            lifetime = tau
                            except:
                                pass  # 拟合失败 lifetime 保持 0

                    # 双指数 (略，结构类似，为了代码简洁先只写单指数，你需要的话可以把你的双指数逻辑拷进来)
                    elif model_type == 'double':
                        # ... 你的双指数逻辑 ...
                        pass

                    lifetime_map[i, j] = lifetime

                except Exception:
                    # 单个像素失败不影响整体
                    lifetime_map[i, j] = 0

            # (可选) 可以在这里通过 Queue 发送 row 进度，但通常不需要那么细
            pass

    except Exception as e:
        traceback.print_exc()
    finally:
        existing_shm_in.close()
        existing_shm_out.close()


# 为了 worker 能调用，必须定义在顶层
def _single_exp_func(t, A, tau, C):
    return A * np.exp(-t / tau) + C


class LifetimeCalculator:
    """
    载流子寿命计算类（此类为静态方法类，不要放别的进来）
    """
    _cal_params = {
        'from_start_cal':False,
        'r_squared_min': 0.4,
        'peak_range': (0.0, 1000.0),
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
    def calculate_lifetime(data_type, time_series, time_points, arg_bundle = None, model_type='single'):
        """
        计算载流子寿命
        """
        params = LifetimeCalculator._cal_params
        from_start_cal = params['from_start_cal']
        r_squared_min = params['r_squared_min']
        peak_range = params['peak_range']
        tau_range = params['tau_range']
        

        # 获得具有实际意义的信号序列
        if data_type in ['central negative', 'central positive']:
            phy_signal = np.abs(time_series)
            max_idx = np.argmax(phy_signal)
            if not from_start_cal:
                decay_signal = phy_signal[max_idx:] # 全部正置 且从最大值之后开始拟合
                decay_time = time_points[max_idx:] - time_points[max_idx]
            else:
                decay_signal = phy_signal
                decay_time = time_points
        elif data_type == 'sif':
            phy_signal = time_series
            max_idx = np.argmax(phy_signal)
            if not from_start_cal:
                decay_signal = phy_signal[max_idx:] # 全部正置 且从最大值之后开始拟合
                decay_time = time_points[max_idx:] - time_points[max_idx]
            else:
                decay_signal = phy_signal
                decay_time = time_points
        else: # 不可能走到这里，我只是觉得代码高亮不舒服 所以加的 (在alpha 1.10.0时遇到了这个问题，于是乎打脸了）
            phy_signal = time_series
            max_idx = np.argmax(phy_signal)
            if not from_start_cal:
                decay_signal = phy_signal[max_idx:]  # 全部正置 且从最大值之后开始拟合
                decay_time = time_points[max_idx:] - time_points[max_idx]
            else:
                decay_signal = phy_signal
                decay_time = time_points

        # 初始猜测
        A_guess = np.max(decay_signal) - np.min(decay_signal)
        tau_guess = (decay_time[-1] - decay_time[0]) / 5
        C_guess = np.min(decay_signal)

        try:
            if model_type == 'single':
                # 单指数拟合
                if peak_range[0] <= max_idx <= peak_range[1]: # 峰值位置筛选
                    popt, pcov = curve_fit(
                        LifetimeCalculator.single_exponential,
                        decay_time,
                        decay_signal,
                        p0=[A_guess, tau_guess, C_guess],
                        bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))
                    if popt[1] <= tau_range[0] or popt[1] >= tau_range[1]: # 寿命大小筛选
                        lifetime = 0
                        r_squared = np.nan
                    else:
                        # 计算R方
                        y_pred = LifetimeCalculator.single_exponential(decay_time, *popt)
                        ss_res = np.sum((decay_signal - y_pred) ** 2)
                        ss_tot = np.sum((decay_signal - np.mean(decay_signal)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        if r_squared <= r_squared_min: # R方筛选
                            lifetime = 0
                        else:
                            lifetime = popt[1]  # tau
                else:
                    lifetime = 0
                    r_squared = np.nan
                    popt = [0,0,0]
                return popt, lifetime, r_squared, phy_signal

            elif model_type == 'double':
                # 双指数拟合
                if peak_range[0] <= max_idx <= peak_range[1]:  # 峰值位置筛选
                    A2_guess = A_guess / 2
                    tau2_guess = tau_guess * 2

                    popt, pcov = curve_fit(
                        LifetimeCalculator.double_exponential,
                        decay_time,
                        decay_signal,
                        p0=[A_guess, tau_guess, A2_guess, tau2_guess, C_guess],
                        bounds=([0, 0, 10, 10, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]))

                    # 计算平均寿命
                    A1, tau1, A2, tau2, C = popt
                    # 计算R方
                    if (tau1 <= tau_range[0] or tau1 >= tau_range[1]) and (tau2 <= tau_range[0] or tau2 >= tau_range[1]):  # 寿命大小筛选
                        tau1,tau2 = (0,0)
                        r_squared = np.nan
                    else:
                        y_pred = LifetimeCalculator.double_exponential(decay_time, *popt)
                        ss_res = np.sum((decay_signal - y_pred) ** 2)
                        ss_tot = np.sum((decay_signal - np.mean(decay_signal)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        if r_squared <= r_squared_min: # R方筛选
                            tau1,tau2 = (0,0)
                        else:
                            tau1,tau2 = (popt[1], popt[3]) # tau1 and 2
                else:
                    tau1,tau2 = (0,0)
                    r_squared = np.nan
                    popt = [0, 0, 0,0,0]
                return popt, (tau1, tau2), r_squared, phy_signal

        except Exception as e:
            # 拟合失败时返回NaN
            if model_type == 'single':
                return [np.nan, np.nan, np.nan], np.nan, np.nan ,phy_signal
            else:
                return [np.nan, np.nan, np.nan, np.nan, np.nan], (np.nan, np.nan), np.nan ,phy_signal

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
        region_data = data.data_origin[:, mask]
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
    cal_running_status = pyqtSignal(bool)
    processed_result = pyqtSignal(ProcessedData)
    calculating_progress_signal = pyqtSignal(int, int)
    stop_thread_signal = pyqtSignal()
    update_status = pyqtSignal(str,str)


    def __init__(self):
        super().__init__()
        logging.info('计算线程已载入')
        self._is_calculating = False

    @pyqtSlot(object, float, np.ndarray, str)
    def region_analyze(self,data,time_unit,mask,model_type):
        """分析选定区域"""
        logging.info("开始计算选区载流子寿命...")
        self.calculating_progress_signal.emit(1, 3)
        self.cal_running_status.emit(True)
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
                                                                      **(data.out_processed if isinstance(data,ProcessedData) else data.parameters)}))
            logging.info("计算完成!")
            self.calculating_progress_signal.emit(3, 3)
        except Exception as e:
            self.update_status.emit(f'区域数据拟合出错:{e}','error')
        finally:
            self._is_calculating = False
            self.cal_running_status.emit(False)
            self.stop_thread_signal.emit()

    @pyqtSlot(object, float, str, str, int, str, int, bool, int)
    def distribution_analyze(self,data,time_unit,model_type,pre_cov = None,pre_size = None,post_cov = None,post_size = None,is_multipro = False,cpu_num = 0):
        """分析全图载流子寿命"""
        self._is_calculating = True
        self.cal_running_status.emit(True)
        try:

            time_points = data.time_point * time_unit
            data_type = data.parameters['data_type'] if data.parameters is not None and 'data_type' in data.parameters else None
            aim_data = data.data_origin
            T = data.timelength

            if pre_cov is not None:
                for t in range(T):
                    frame = aim_data[t, :, :]
                    smoothed_frame = LifetimeCalculator.apply_custom_kernel(frame, kernel_type=pre_cov,half_size=pre_size)
                    aim_data[t, :, :] = smoothed_frame
                    self.calculating_progress_signal.emit(t, T)
                self.calculating_progress_signal.emit(T,T)
                logging.info("预卷积完成，下面开始计算")

            if is_multipro:
                lifetime_map = self._run_multiprocess_lifetime_cal(aim_data, data_type, time_points, model_type, cpu_num)
            else:
                lifetime_map = self.lifetime_map_cal(aim_data,data_type,time_points,model_type)
            if isinstance(lifetime_map, np.ndarray):
                pass
            else:
                raise Exception(lifetime_map)

            # 显示结果
            if post_cov is not None:
                lifetime_map_cov = LifetimeCalculator.apply_custom_kernel(lifetime_map,post_cov,post_size)
                logging.info("后卷积完成")
            else:
                lifetime_map_cov = lifetime_map
            self.processed_result.emit(ProcessedData(data.timestamp,
                                                      f'{data.name}@d-lft',
                                                      'lifetime_distribution',
                                                     time_point=np.array([0]),
                                                      data_processed=lifetime_map_cov,
                                                     out_processed={'lifetime_map': lifetime_map,
                                                                    **(data.out_processed if isinstance(data,ProcessedData) else data.parameters)}))
        except Exception as e:
            self.update_status.emit(f'区域数据拟合出错:{e}','error')
        finally:
            self._is_calculating = False
            self.cal_running_status.emit(False)
            self.stop_thread_signal.emit()

    @pyqtSlot(object, float, float, float, str)
    def diffusion_calculation(self,frame_data,time_unit,space_unit,timestamp,name):
        # 存储拟合方差结果 [时间, 方差]
        self._is_calculating = True
        self.cal_running_status.emit(True)

        sigma_results = np.zeros((2, len(frame_data)))
        time_series = []
        loading_bar_value =0 #进度条
        total_l = len(frame_data)
        fitting_result = []
        signal_series = []
        if self._is_calculating:  # 线程关闭控制（目前仅针对循环计算）
            for i, (frame_idx, data) in enumerate(frame_data.items()):
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
            self.cal_running_status.emit(False)
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
        self.cal_running_status.emit(False)
        self.stop_thread_signal.emit()

    @pyqtSlot(object, float, str, str, int, str, int, bool, int)
    def heat_transfer_calculation(self,data,time_unit,model_type,pre_cov = None,pre_size = None,post_cov = None,post_size = None,is_multipro = False,cpu_num = 0):
        """计算传热系数"""
        self._is_calculating = True
        self.cal_running_status.emit(True)
        try:
            if isinstance(data, ProcessedData) and data.type_processed == "lifetime_distribution":
                aim_data = data.data_processed
                # 向量化计算，极快，无需多进程
                with np.errstate(divide='ignore', invalid='ignore'):
                    heat_transfer = np.where(aim_data >= 0.1, 42.72 / aim_data, 0)
                if post_cov is not None:
                    heat_transfer_cov = LifetimeCalculator.apply_custom_kernel(heat_transfer, post_cov, post_size)
                    logging.info("后卷积完成")
                else:
                    heat_transfer_cov = heat_transfer
                self.processed_result.emit(ProcessedData(data.timestamp,
                                                         f'{data.name}@heat',
                                                         'heat_transfer',
                                                         time_point=np.array([0]),
                                                         data_processed=heat_transfer_cov,
                                                         out_processed={'heat_transfer_map': heat_transfer,
                                                                        **(data.out_processed if isinstance(data,ProcessedData) else data.parameters)}))
                return True
            time_points = data.time_point * time_unit
            data_type = data.parameters.get('data_type')
            aim_data = data.data_origin.copy()
            T = data.timelength

            if pre_cov is not None:
                for t in range(T):
                    frame = aim_data[t, :, :]
                    smoothed_frame = LifetimeCalculator.apply_custom_kernel(frame, kernel_type=pre_cov,half_size=pre_size)
                    aim_data[t, :, :] = smoothed_frame
                    self.calculating_progress_signal.emit(t, T)
                self.calculating_progress_signal.emit(T,T)
                logging.info("预卷积完成，下面开始传热计算")

            # 拟合计算
            if is_multipro:
                lifetime_map = self._run_multiprocess_lifetime_cal(aim_data, data_type, time_points, model_type,
                                                                   cpu_num)
            else:
                lifetime_map = self.lifetime_map_cal(aim_data, data_type, time_points, model_type)

            if isinstance(lifetime_map, np.ndarray):
                pass
            else:
                raise Exception(lifetime_map)
            # 向量化计算传热系数
            with np.errstate(divide='ignore', invalid='ignore'):
                heat_transfer = np.where(lifetime_map >= 0.1, 42.72 / lifetime_map, 0)

            # 后卷积
            if post_cov is not None:
                heat_transfer_cov = LifetimeCalculator.apply_custom_kernel(heat_transfer,post_cov,post_size)
                logging.info("后卷积完成")
            else:
                heat_transfer_cov = heat_transfer
            self.processed_result.emit(ProcessedData(data.timestamp,
                                                     f'{data.name}@heat',
                                                     'heat_transfer',
                                                     time_point=np.array([0]),
                                                     data_processed=heat_transfer_cov,
                                                     out_processed={'heat_transfer_map': heat_transfer, }))
        except Exception as e:
            self.update_status.emit(f'传热计算出错:{e}', 'error')
        finally:
            self._is_calculating = False
            self.cal_running_status.emit(False)
            self.stop_thread_signal.emit()

    @pyqtSlot(object,str, object)
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

    def lifetime_map_cal(self,aim_data,data_type,time_points,model_type):
        """纯寿命热图计算"""
        try:
            T, height, width = aim_data.shape
            lifetime_map = np.zeros((height, width))
            logging.info("开始拟合热图...")

            loading_bar_value = 0  # 进度条
            total_l = height * width
            for i in range(height):
                if self._is_calculating:  # 线程关闭控制（目前仅针对长时计算）
                    for j in range(width):
                        time_series = aim_data[:, i, j]
                        # 用皮尔逊系数判断噪音(滑动窗口法)
                        window_size = min(10, len(time_points) // 2)
                        pr = []
                        for k in range(len(time_series) - window_size):
                            window = time_series[k:k + window_size]
                            time_window = time_points[k:k + window_size]
                            r, _ = pearsonr(time_window, window)
                            pr.append(r)
                            if abs(r) >= 0.8:
                                _, lifetime, r_squared, _ = LifetimeCalculator.calculate_lifetime(data_type, time_series,
                                                                                                  time_points, model_type)
                                continue
                            else:
                                pass
                        if np.all(np.abs(pr) < 0.8):
                            lifetime = np.nan
                        else:
                            pass
                        lifetime_map[i, j] = lifetime if not np.isnan(lifetime) else 0
                        loading_bar_value += 1
                        self.calculating_progress_signal.emit(loading_bar_value, total_l)
                else:
                    logging.info("计算终止")
                    self.calculating_progress_signal.emit(total_l, total_l)  # 进度条更新
                    return "线程终止"
            logging.info("计算完成!")
            return lifetime_map
        except Exception as e:
            return e
        finally:
            self.cal_running_status.emit(False)
            self.stop_thread_signal.emit()  # 目前来说，计算终止也会关闭线程，后续可考虑分开命令

    def _run_multiprocess_lifetime_cal(self, aim_data, data_type, time_points, model_type, cpu_num):
        """核心：驱动多进程进行 curve_fit"""
        logging.info("将开启并行加速，进度条不代表实时进度，请稍等。。。")
        shm_in = None
        shm_out = None
        pool = None

        try:
            T, height, width = aim_data.shape
            total_pixels = height * width

            logging.info(f"开始多进程拟合: 尺寸 {width}x{height}, 帧数 {T}")

            # 1. 创建共享内存
            # 输入数据 SHM
            shm_in = shared_memory.SharedMemory(create=True, size=aim_data.nbytes)
            shm_in_arr = np.ndarray(aim_data.shape, dtype=aim_data.dtype, buffer=shm_in.buf)
            shm_in_arr[:] = aim_data[:]  # 复制数据

            # 输出结果 SHM (2D map)
            output_shape = (height, width)
            output_dtype = np.float64  # lifetime通常用float64
            # 计算字节数
            out_size = int(np.prod(output_shape) * np.dtype(output_dtype).itemsize)
            shm_out = shared_memory.SharedMemory(create=True, size=out_size)
            shm_out_arr = np.ndarray(output_shape, dtype=output_dtype, buffer=shm_out.buf)
            shm_out_arr[:] = 0  # 初始化

            # 2. 获取当前的拟合参数 (从 LifetimeCalculator 获取)
            # 必须显式传递，因为子进程看不到主进程修改后的 _cal_params
            fit_params = LifetimeCalculator._cal_params.copy()

            # 3. 配置任务
            # 按行(row)分割任务，避免切片太碎
            num_cores = cpu_num
            rows_per_core = math.ceil(height / num_cores)

            tasks = []
            for i in range(num_cores):
                start_row = i * rows_per_core
                end_row = min((i + 1) * rows_per_core, height)
                if start_row >= end_row: break

                tasks.append((
                    shm_in.name, aim_data.shape, aim_data.dtype,
                    shm_out.name, output_shape, output_dtype,
                    (start_row, end_row),
                    time_points,
                    data_type, model_type, fit_params
                ))

            self.calculating_progress_signal.emit(0, 100)  # 假进度，表示开始

            # 4. 启动进程池
            pool = Pool(processes=len(tasks))
            result_async = pool.starmap_async(_lifetime_fit_worker, tasks)
            self.calculating_progress_signal.emit(1, 100)

            # 5. 等待完成 (为了简单，这里用简单的轮询或者直接wait)
            # 由于拟合非常耗时，且这里没法像STFT那样精确数像素(因为在C里循环)，
            # 我们可以做一个简单的等待动画
            while not result_async.ready():
                if not self._is_calculating:  # 支持中途取消
                    pool.terminate()
                    return None
                QThread.msleep(100)
                # 可以在这里发信号让进度条滚来滚去

            result_async.get()  # 检查错误

            # 6. 取回结果
            lifetime_map = shm_out_arr.copy()  # Copy out
            self.calculating_progress_signal.emit(99, 100)
            logging.info("多进程拟合完成")
            return lifetime_map

        except Exception as e:
            raise e
        finally:
            self.calculating_progress_signal.emit(100, 100)
            if pool:
                pool.close()
                pool.join()
            if shm_in:
                shm_in.close()
                shm_in.unlink()
            if shm_out:
                shm_out.close()
                shm_out.unlink()

    def stop(self):
        self._is_calculating = False

    def force_stop(self):
        self._is_calculating = False
        self.cal_running_status.emit(False)
        self.stop_thread_signal.emit()
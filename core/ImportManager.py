import glob
import re
import logging
import os
from inspect import Parameter, signature

import numpy as np
import tifffile as tiff
import sif_parser
import cv2
import h5py
from PIL import Image

from typing import List, Union, Optional, Callable

from DataManager import *

from PyQt5.QtCore import QObject


class ImportManager(QObject):
    """本类仅包含导入数据时的数据处理"""
    update_status = pyqtSignal(str, str)
    update_QMessageBox = pyqtSignal(str, str, str)
    processing_progress_signal = pyqtSignal(int, int) # 进度槽
    import_finished = pyqtSignal(Data)
    def __init__(self):
        super().__init__()
        self.type_all ={
            'tif_series': self.load_and_sort_tiff,
            'sif_folder': self.load_and_sort_sif,
            'avi_EM': self.load_avi,
            'tif_EM': self.load_tiff,
        }
        self.abortion = False
        logging.info("数据导入线程已启动")

    @pyqtSlot(str,str,dict)
    def import_dispatch(self,import_type:str, filepath:str, kwargs_dict: Dict[str, Any]):

        if import_type in self.type_all:
            try:
                handler = self.type_all[import_type]
                self._call_with_kwargs(handler,filepath,kwargs_dict)
            except Exception as e:
                logging.error(f"处理信号 '{import_type}' 时出错: {e}")
        else:
            logging.error("导入类型不支持")

    def _call_with_kwargs(self, handler: Callable, filepath: str, kwargs_dict: Dict[str, Any]):
        """
        动态调用处理器函数，自动解包参数字典
        参数:
            handler: 处理器函数
            filepath: 文件路径
            kwargs_dict: 参数字典
        返回:
            处理器函数的返回值
        """
        # 获取处理器函数的参数签名
        sig = signature(handler)
        params = list(sig.parameters.keys())

        # 检查是否需要 filepath 参数
        if len(params) > 0 and params[0] == 'self':
            params = params[1:]  # 去掉 self

        # 构建调用参数字典
        call_kwargs = {}

        # 如果有参数，第一个参数通常是 filepath
        if params and 'filepath' in kwargs_dict:
            # 如果函数签名包含 filepath 参数，则使用提供的 filepath
            if 'filepath' in params:
                call_kwargs['filepath'] = filepath
            # 如果函数期望第一个参数是 filepath，但不叫 'filepath'
            elif len(params) > 0:
                call_kwargs[params[0]] = filepath
        elif params:
            # 如果 kwargs_dict 中没有 filepath，但函数有参数
            call_kwargs[params[0]] = filepath

        # 添加其他参数
        for key, value in kwargs_dict.items():
            if key != 'filepath':  # filepath 已经处理过了
                if key in params:
                    call_kwargs[key] = value
                elif any(p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    # 如果函数接受 **kwargs，添加所有额外参数
                    if 'kwargs' not in call_kwargs:
                        call_kwargs['kwargs'] = {}
                    call_kwargs['kwargs'][key] = value

        # 调用处理器函数
        return handler(**call_kwargs)

    """tiff"""
    def load_and_sort_tiff(self, filepath ,current_group, time_step, space_step, time_unit, space_unit):
        # 因为tiff存在两种格式，n,p
        files = []
        find = filepath + '/*.tiff'
        if current_group != '不区分':
            self.tiff_type = current_group
            for f in glob.glob(find):
                match = re.search(r'(\d+)([a-zA-Z]+)\.tiff', f)
                if match and match.group(2) == current_group:
                    files.append((int(match.group(1)), f))
            if not files: # 当找不到任何tiff时，使其能够寻找tif结尾的文件
                find_tif = filepath + '/*.tif'
                for f in glob.glob(find_tif):
                    match = re.search(r'(\d+)([a-zA-Z]+)\.tif', f)
                    if match and match.group(2) == current_group:
                        files.append((int(match.group(1)), f))
        if current_group == '不区分':
            self.tiff_type = 'np'
            find_tif =  filepath + '/*.tiff'
            for f in glob.glob(find_tif):
                # 提取数字并排序
                num_groups = re.findall(r'\d+', f)
                last_num = int(num_groups[-1]) if num_groups else 0
                files.append((last_num, f))
            if not files:  # 当找不到任何tiff时，使其能够寻找tif结尾的文件
                find_tif = filepath + '/*.tif'
                for f in glob.glob(find_tif):
                    # 提取数字并排序
                    num_groups = re.findall(r'\d+', f)
                    last_num = int(num_groups[-1]) if num_groups else 0
                    files.append((last_num, f))

        tiff_files = sorted(files, key=lambda x: x[0])
        if not tiff_files or tiff_files == []:
            self.update_status.emit("文件夹中没有目标TIFF文件", 'warning')
            self.update_QMessageBox.emit("warning", "导入错误", "文件夹中没有目标TIFF文件")
            return
        # 数据处理合并
        try:
            images_original = []
            vmax_array = []
            vmin_array = []
            vmean_array = []
            for _, fpath in files:
                img_data = tiff.imread(fpath)
                vmax_array.append(np.max(img_data))
                vmin_array.append(np.min(img_data))
                vmean_array.append(np.mean(img_data))
                images_original.append(img_data)
            #   以最值为边界
            vmax = np.max(vmax_array)
            vmin = np.min(vmin_array)
            filename = os.path.basename(filepath)

            images_show, data_type, max_mean, phy_max, phy_min = self.process_data(images_original, vmax, vmin, vmean_array)

            self.import_finished.emit(Data(np.stack(images_original, axis=0),
                                    np.arange(len(images_show)),
                                    'tif_series',
                                    np.stack(images_show, axis=0),
                                    parameters={
                                        'file_path': filepath,
                                        'vmax_array':vmax_array,
                                        'vmin_array':vmin_array,
                                        'data_type':data_type,
                                        'time_step':time_step,
                                        'time_unit':time_unit,
                                        'space_step':space_step,
                                        'space_unit':space_unit,
                                    },
                                    name=filename))

        except Exception as e:
            self.update_QMessageBox.emit('error','导入错误',f'无法读取TIFF文件:{e}')
            self.update_status.emit("无法读取TIFF文件", 'warning')

    @staticmethod
    def process_data(data, max_all, min_all, vmean_array):
        process_show = []
        if np.abs(min_all) > np.abs(max_all):
            # n-type 信号中心为黑色，最强值为负
            data_type = 'central negative'
            for every_data in data:
                normalized_data = (every_data - min_all) / (max_all - min_all)
                process_show.append(normalized_data)
            max_mean = np.min(vmean_array)
            phy_max = -min_all
            phy_min = -max_all
        else:
            # p-type 信号中心为白色，最强值为正
            data_type = 'central positive'
            for every_data in data:
                normalized_data = (max_all - every_data) / (max_all - min_all)
                process_show.append(normalized_data)
            max_mean = np.max(vmean_array)
            phy_max = max_all
            phy_min = min_all
        return process_show, data_type, max_mean, phy_max, phy_min

    """sif"""
    def load_and_sort_sif(self, foldpath, normalize_type, time_step, space_step, time_unit, space_unit):
        time_data = {}  # 存储时间点数据
        background = None  # 存储背景数据

        try:
            for filename in os.listdir(foldpath):
                if filename.endswith('.sif'):
                    filepath = os.path.join(foldpath, filename)
                    name = os.path.splitext(filename)[0]  # 去除扩展名

                    # 检查是否是背景文件（文件名包含 "no"）
                    if name.lower() == 'no':
                        background = sif_parser.np_open(filepath)[0][0]
                        continue

                    # 否则尝试提取时间点（文件名中的数字）
                    match = re.search(r'(\d+)', name)
                    if match:
                        time = int(match.group(1))
                        data = sif_parser.np_open(filepath)[0][0]
                        time_data[time] = data
                else:
                    self.update_status.emit("文件夹中没有目标SIF文件",'warning')
                    self.update_QMessageBox.emit('error','导入错误',"文件夹中没有目标SIF文件,请确认选择的文件格式是否匹配")
                    logging.warning("文件夹中没有目标SIF文件")
                    return False

            # 检查是否找到背景
            if background is None:
                raise logging.error("未找到背景文件（文件名应包含 'no'）")

            # 按时间排序
            self.sif_sorted_times = sorted(time_data.keys())

            # 创建三维数组（时间, 高度, 宽度）并减去背景
            sample_data = next(iter(time_data.values()))
            self.sif_data_original = np.zeros((len(self.sif_sorted_times), *sample_data.shape), dtype=np.float32)

            for i, time in enumerate(self.sif_sorted_times):
                self.sif_data_original[i] = (time_data[time] - background)/background

            normalized = self.normalize_data(self.sif_data_original,normalize_type)
            self.import_finished.emit(Data(np.stack(self.sif_data_original , axis=0),
                                    np.stack(self.sif_sorted_times,axis=0),
                                    'sif',
                                    np.stack(normalized, axis=0),
                                    name=os.path.basename(foldpath),
                                    parameters={
                                        'data_type': 'sif_folder',
                                        'normalize_type': normalize_type,
                                        'file_path': foldpath,
                                        'time_unit': time_unit,
                                    }
                                    ))
            return True
        except Exception as e:
            logging.error(f"处理SIF序列时出错: {str(e)}")

    """avi"""
    def load_avi(self,filepath,fps, time_step, space_step, time_unit, space_unit):
        """
        处理AVI视频文件，返回包含视频数据和元信息的字典
        返回字典:
            - data_origin: 原始视频帧数据 (n_frames, height, width)
            - images: 归一化后的视频帧数据
            - time_points: 时间点数组
            - data_type: 数据类型标识 ('video')
            - boundary: 最大最小值边界
            - fps: 视频帧率
            - frame_size: 视频帧尺寸 (width, height)
            - duration: 视频时长(秒)
        """
        file_name = os.path.basename(filepath)
        # 读取视频文件
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {filepath}")

        # 获取视频元信息
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        # codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        duration = frame_count / fps if fps > 0 else 0

        # 读取所有帧
        frames = []
        loading_bar_value = 0  # 进度条
        total_l = frame_count+1
        self.processing_progress_signal.emit(loading_bar_value, total_l)
        while not self.abortion:
            ret, frame = cap.read()
            if not ret:
                break
            # 转换为灰度图(如果原始是彩色)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            loading_bar_value += 1
            self.processing_progress_signal.emit(loading_bar_value, total_l)

        cap.release()

        if not frames:
            raise ValueError("视频中没有读取到有效帧")

        # 转换为numpy数组
        frames_array = np.stack(frames, axis=0)

        # 归一化处理
        normalized_frames = self.normalize_data(frames_array)

        self.processing_progress_signal.emit(loading_bar_value+1, total_l)
        avi_data = Data(frames_array,
                                    np.arange(len(frames)) / fps if fps > 0 else np.arange(len(frames)),
                                    'avi_EM',
                                    normalized_frames,
                                    parameters={'fps': fps,
                                                'frame_size': (width, height),
                                                'duration': duration,
                                                'file_path': filepath,
                                                'space_step': space_step,
                                                'space_unit': space_unit,},
                                    name=file_name)

        self.import_finished.emit(avi_data)

    """TIF"""
    def load_tiff(self,filepath,fps, time_step, space_step, time_unit, space_unit):
        try:
            tiff_files = []
            file_name = os.path.basename(filepath)
            for f in os.listdir(filepath):
                if f.lower().endswith(('.tif', '.tiff')):
                    # 提取数字并排序
                    num_groups = re.findall(r'\d+', f)
                    last_num = int(num_groups[-1]) if num_groups else 0
                    tiff_files.append((last_num, f))

            # 按最后一组数字排序
            tiff_files.sort(key=lambda x: x[0])
            frame_numbers, file_names = zip(*tiff_files) if tiff_files else ([], [])

            logging.info(f"找到{len(file_names)}个TIFF文件")

            # 检查数字连续性（使用已排序的frame_numbers）
            unique_nums = sorted(set(frame_numbers))
            is_continuous = (
                    len(unique_nums) == len(frame_numbers) and
                    (unique_nums[-1] - unique_nums[0] + 1) == len(unique_nums)
            )

            # 读取图像数据
            frames = []
            total_files = len(file_names)
            self.processing_progress_signal.emit(0, total_files)

            for i, filename in enumerate(file_names):
                if self.abortion:
                    break

                img_path = os.path.join(filepath, filename)
                img = tiff.imread(img_path)

                if img is None:
                    logging.warning(f"无法读取文件: {filename}")
                    continue

                frames.append(img)
                self.processing_progress_signal.emit(i + 1, total_files)

            if not frames:
                raise ValueError("没有有效图像数据被读取")

            # 转换为numpy数组
            frames_array = np.stack(frames, axis=0)
            height, width = frames[0].shape

            # 计算统计信息
            normalized_frames = self.normalize_data(frames_array)

            # 生成时间点
            if is_continuous:
                time_points = (np.array(frame_numbers) - frame_numbers[0]) / fps
                logging.info("使用文件名数字作为时间序列")
            else:
                time_points = np.arange(len(frames)) / fps
                logging.info("使用默认顺序作为时间序列")

            tiff_data = Data(frames_array,
                                         time_points,
                                         'tif_EM',
                                         normalized_frames,
                                         parameters={'fps':fps,
                                                    'frame_size': (width, height),
                                                    'original_files': tiff_files,
                                                    'file_path': filepath,
                                                     'space_step': space_step,
                                                     'space_unit': space_unit,},
                                         name = file_name)
            self.import_finished.emit(tiff_data)

        except Exception as e:
            logging.error(f"处理TIFF序列时出错: {str(e)}")
            self.processing_progress_signal.emit(1, 1)

    @staticmethod
    def normalize_data(
            data: np.ndarray,
            method: str = 'linear',
            low: float = 10,
            high: float = 100,
            k: Optional[float] = None,
            clip_limit: float = 0.03,
            eps: float = 1e-6
    ) -> np.ndarray:
        """
        多种归一化方法可选
        Parameters:
            method:
                'linear'    - 线性归一化 (min-max)
                'sigmoid'  - Sigmoid归一化
                'percentile'- 百分位裁剪归一化 (默认)
                'log'      - 对数归一化
                'clahe'    - 自适应直方图均衡化
            low/high: 百分位裁剪的上下界（method='percentile'时生效）
            k: Sigmoid的斜率系数（method='sigmoid'时生效，None则自动计算）
            clip_limit: CLAHE的裁剪限制（method='clahe'时生效）
            eps: 对数归一化的微小增量（method='log'时生效）
        """
        if method == 'linear':
            # 线性归一化
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        elif method == 'sigmoid':
            # Sigmoid归一化
            mu = np.median(data)
            std = np.std(data)
            k = 10 / std if k is None else k
            centered = data - mu
            return 1 / (1 + np.exp(-k * centered))

        elif method == 'percentile':
            # 百分位裁剪归一化
            plow = np.percentile(data, low)
            phigh = np.percentile(data, high)
            clipped = np.clip(data, plow, phigh)
            return (clipped - plow) / (phigh - plow)

        elif method == 'log':
            # 对数归一化
            logged = np.log(data + eps)
            return (logged - np.min(logged)) / (np.max(logged) - np.min(logged))

        # elif method == 'clahe':
        #     # CLAHE自适应直方图均衡化
        #     return equalize_adapthist(data, clip_limit=clip_limit)
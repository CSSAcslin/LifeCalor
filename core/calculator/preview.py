from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from ArrayCache import ArrayRef
from DataManager import Data, ProcessedData
from diagnostics import AppError, format_exception_details
from .engine import CalculationEngine


@dataclass(frozen=True)
class PreviewFrame:
    image: np.ndarray
    description: str
    minimum: object
    maximum: object


class PreviewTaskSignals(QObject):
    completed = pyqtSignal(int, object)
    failed = pyqtSignal(int, object)


class CalculatorPreviewTask(QRunnable):
    """Read exactly one calculator preview frame outside the GUI thread."""

    MAX_PREVIEW_PIXELS = 4 * 1024 * 1024

    def __init__(self, request_id, source, payload_key, slice_text, axes, frame_index):
        super().__init__()
        # Python owns the signal object; defer QRunnable destruction until the
        # queued GUI callback has returned.
        self.setAutoDelete(False)
        self.request_id = int(request_id)
        self.source = source
        self.payload_key = payload_key
        self.slice_text = str(slice_text or "")
        self.axes = str(axes or "")
        self.frame_index = int(frame_index)
        self.signals = PreviewTaskSignals()

    @staticmethod
    def _source_value(source, payload_key):
        if payload_key is not None:
            value = (getattr(source, "out_processed", None) or {})[payload_key]
        elif isinstance(source, ProcessedData):
            value = source.__dict__.get("_data_processed_storage")
            if value is None:
                value = source.data_processed
        elif isinstance(source, Data):
            value = source.__dict__.get("_data_origin_storage")
            if value is None:
                value = source.data_origin
        else:
            value = source
        if isinstance(value, ArrayRef):
            value = value.load(mmap_mode="r")
        return np.asarray(value)

    def run(self):
        try:
            value = self._source_value(self.source, self.payload_key)
            if self.slice_text:
                index = CalculationEngine.slice_tuple(
                    CalculationEngine._parse_slice(self.slice_text), value.ndim
                )
                value = value[index]
            if value.ndim == 3 and self.axes.startswith("T"):
                if not 0 <= self.frame_index < value.shape[0]:
                    raise IndexError(
                        f"帧号 {self.frame_index} 超出范围 0~{value.shape[0] - 1}"
                    )
                sample = value[self.frame_index]
                description = f"帧 {self.frame_index} / {value.shape[0] - 1}"
            elif value.ndim == 2:
                sample = value
                description = "二维数据"
            else:
                raise ValueError(f"预览只支持 HW 或 THW 数据，当前 shape={value.shape}")
            if np.iscomplexobj(sample):
                sample = np.abs(sample)
                description += " · complex 幅值"
            if sample.size > self.MAX_PREVIEW_PIXELS:
                stride = int(np.ceil(np.sqrt(sample.size / self.MAX_PREVIEW_PIXELS)))
                sample = sample[::stride, ::stride]
                description += f" ? 预览抽样 {stride}x"
            image = np.array(sample, copy=True, order="C")
            if image.size == 0:
                raise ValueError("当前预览帧为空")
            result = PreviewFrame(
                image=image,
                description=description,
                minimum=np.nanmin(image),
                maximum=np.nanmax(image),
            )
            self.signals.completed.emit(self.request_id, result)
        except Exception as exc:
            context = (
                f"source_name: {getattr(self.source, 'name', '')}\n"
                f"payload_key: {self.payload_key}\n"
                f"shape: {getattr(self.source, 'datashape', None)}\n"
                f"dtype: {getattr(self.source, 'datatype', None)}\n"
                f"axes: {self.axes}\n"
                f"slice: {self.slice_text or 'None'}\n"
                f"frame_index: {self.frame_index}"
            )
            details = format_exception_details(exc, "计算器帧预览", self.source)
            error = AppError(
                "预览读取失败",
                str(exc),
                stage="计算器帧预览",
                severity="warning",
                details=f"{context}\n{details}",
                original=exc,
            )
            self.signals.failed.emit(self.request_id, error)

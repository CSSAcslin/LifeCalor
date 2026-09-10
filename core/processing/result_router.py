from __future__ import annotations

import logging

from DataManager import ProcessedData
from diagnostics import AppError


class ResultRouter:
    """Route ProcessedData results without growing a MainWindow match block."""

    def __init__(self, window, processing_controller):
        self.window = window
        self.processing = processing_controller
        self._handlers = {
            "ROI_lifetime": self._roi_lifetime,
            "lifetime_distribution": self._lifetime_distribution,
            "diffusion": self._diffusion,
            "heat_transfer": self._heat_transfer,
            "stft_quality": self._stft_quality,
            "cwt_quality": self._cwt_quality,
            "ROI_stft": self._roi_stft,
            "ROI_cwt": self._roi_cwt,
            "Accumulated_time_amplitude_map": self._accumulated,
            "Single_channel_signal": self._single_channel,
            "signal_average": self._signal_average,
            "Roi_applied": self._roi_applied,
            "Heartbeat": self._heartbeat,
            "Basic_math": lambda _data: logging.info("对数据的基础运算完毕！"),
            "Multi_data_math": lambda _data: logging.info("多数据运算完成"),
            "data_cropped": lambda _data: logging.info("对数据的切片完成！"),
            "EM_pre_processed": lambda _data: None,
            "2D_Fourier_transform": lambda _data: None,
            "2D_Inverse_Fourier_transform": lambda _data: None,
        }

    def register(self, process_type, handler):
        self._handlers[str(process_type)] = handler

    def route(self, data):
        if not isinstance(data, ProcessedData):
            error = AppError(
                "处理结果类型无效", f"收到无法分发的处理结果：{type(data).__name__}",
                stage="结果分发", severity="warning", details=repr(data),
            )
            self.window._handle_processing_error(error)
            return False
        self.window.processed_data = data
        process_type = data.type_processed
        self.processing.complete_for_result(process_type)
        handler = self._handlers.get(process_type)
        if handler is None:
            logging.info("处理结果已保存，暂无专用展示器: %s", process_type)
            return True
        handler(data)
        return True

    def _roi_lifetime(self, data):
        self.window.result_display.display_lifetime_curve(data, self.window.time_unit_combo.currentText())

    def _lifetime_distribution(self, data):
        self.window.result_display.display_distribution_map(data, "指数衰减寿命分布图")

    def _diffusion(self, data):
        self.window.result_display.display_diffusion_coefficient(data)

    def _heat_transfer(self, data):
        self.window.result_display.display_distribution_map(data, "传热系数分布图")

    def _stft_quality(self, data):
        self.window.stft_quality_btn.setEnabled(True)
        self.window.result_display.quality_avg(data)

    def _cwt_quality(self, data):
        logging.info("请稍等，出图会有点慢")
        self.window.cwt_quality_btn.setEnabled(True)
        self.window.result_display.quality_avg(data)

    def _roi_stft(self, _data):
        self.window.stft_process_btn.setEnabled(True)

    def _roi_cwt(self, _data):
        self.window.cwt_process_btn.setEnabled(True)

    def _accumulated(self, _data):
        self.window.atam_btn.setEnabled(True)

    def _single_channel(self, data):
        self.window.tDgf_btn.setEnabled(True)
        self.window.sscs_btn.setEnabled(True)
        if data.out_processed.get("thr_known"):
            self.window.result_display.single_channel(data, True)
            return
        threshold = int(data.out_processed.get("thr", 0))
        mean_signal = data.out_processed.get("mean_signal")
        self.window.time_slider_vertical.setVisible(True)
        if mean_signal is not None:
            self.window.time_slider_vertical.setMaximum(int(mean_signal.max() * 10 + 21))
        self.window.update_result_display(threshold * 10, reuse_current=False)

    def _signal_average(self, data):
        values = data.data_processed
        self.window.result_display.plot_time_series(data.time_point, values[:, 1])
        self.window.graph_plot.plot_data(values, name=data.name)

    @staticmethod
    def _roi_applied(_data):
        logging.info("ROI应用完成")

    def _heartbeat(self, data):
        logging.info("心肌细胞处理完成，开始作图")
        self.window.result_display.display_heartbeat(data)
        logging.info("所有图绘制完成")
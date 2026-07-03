from __future__ import annotations

import logging

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ExtraDialog import DataExportDialog, DataSavingPop
from .policy import can_export_em_data, prepare_dataframe_for_export
from .workflow import save_dataframe


class ExportController:
    def __init__(self, window):
        self.window = window

    def export_image(self):
        current_index = self.window.result_display.currentIndex()
        if current_index < 0:
            QMessageBox.warning(self.window, "导出失败", "没有可导出的图像")
            return

        tab = self.window.result_display.widget(current_index)
        canvas = tab.findChild(FigureCanvas)
        if not canvas:
            QMessageBox.warning(self.window, "导出失败", "未找到图像画布")
            return

        try:
            path, _ = QFileDialog.getSaveFileName(
                self.window, "保存图像", "", "PNG(*.png);;JPEG(*.jpg);;TIFF图像 (*.tif *.tiff);;所有文件(*.*)"
            )
            if path:
                canvas.figure.savefig(path, dpi=300)
                QMessageBox.information(self.window, "导出成功", f"图像已保存至:\n{path}")
                logging.info(f"导出成功,图像已保存至:{path}")
        except Exception as exc:
            logging.info(f"数据未保存: {exc}")

    def export_data(self):
        result_display = self.window.result_display
        if result_display.current_dataframe is None:
            logging.warning("没有数据可以导出")
            self.window.update_status("准备就绪")
            return

        dialog = DataSavingPop(self.window)
        file_path = None
        self.window.update_status("数据导出ing", 'working')
        if dialog.exec_():
            is_fitting = dialog.fitting_check.isChecked()
            has_header = dialog.index_check.isChecked()
            file_path, _ = QFileDialog.getSaveFileName(
                self.window, "保存数据", "", "CSV文件 (*.csv);;文本文件 (*.txt)"
            )
        else:
            is_fitting = False
            has_header = True

        if file_path:
            dataframe = prepare_dataframe_for_export(
                result_display.current_dataframe,
                result_display.current_mode,
                include_fitting=is_fitting,
            )
            task_state = self.window.task_states["export"]
            if save_dataframe(dataframe, file_path, has_header, task_state):
                logging.info("数据已保存")
            else:
                logging.info(f"数据未保存: {task_state.error}")
            self.window.update_status("准备就绪", 'idle')
            return

        logging.info("数据未保存")
        self.window.update_status("准备就绪", 'idle')

    def export_em_data(self, result):
        processed_data = self.window.processed_data
        if processed_data is None:
            logging.warning('请先加载并处理数据')
            return
        if not can_export_em_data(processed_data.type_processed):
            QMessageBox.warning(self.window, '提示', '请先变换处理数据')
            return

        dialog = DataExportDialog(datatypes=['tif', 'avi', 'gif', 'png'])
        if dialog.exec_():
            directory = dialog.directory
            prefix = dialog.text_edit.text().strip()
            filetype = dialog.type_combo.currentText()
            duration = dialog.duration_input.value()
            self.window.mass_export_signal.emit(
                processed_data.data_processed,
                directory,
                prefix,
                filetype,
                True,
                {'duration': duration},
            )

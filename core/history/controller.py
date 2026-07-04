from __future__ import annotations

import logging

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox

from DataManager import ArrayLoadWorker, collect_array_refs
from ExtraDialog import DataViewAndSelectPop


class HistoryController:
    def __init__(self, window):
        self.window = window
        self.cache_load_thread = None
        self.cache_load_worker = None

    def data_history_view(self):
        if self.window.data is None:
            logging.warning('暂无导入数据历史')
            return
        dialog = DataViewAndSelectPop(datadict=self.window.get_data_all())
        if dialog.exec_():
            selected_timestamp, _ = dialog.get_selected_timestamp()
            selected_data = self.window.data.find_history(selected_timestamp)
            self.load_cached_history_async(selected_data, 'data')

    def process_history_view(self):
        if self.window.processed_data is None:
            logging.warning('暂无处理数据历史')
            return
        dialog = DataViewAndSelectPop(processed_datadict=self.window.get_processed_data_all())
        if dialog.exec_():
            selected_timestamp, _ = dialog.get_selected_timestamp()
            selected_data = self.window.processed_data.find_history(selected_timestamp)
            self.load_cached_history_async(selected_data, 'processed_data')

    def load_cached_history_async(self, target, attr_name):
        if target is None:
            logging.warning("未找到可读取的历史数据")
            return
        if not collect_array_refs(target):
            setattr(self.window, attr_name, target)
            logging.info(f"当前数据焦点已更新至{target.name}")
            return

        self.window.update_status("正在读取缓存数据", 'working')
        self.window.update_progress(0, 1000)
        self.cache_load_thread = QThread()
        self.cache_load_worker = ArrayLoadWorker(target)
        self.window.cache_load_thread = self.cache_load_thread
        self.window.cache_load_worker = self.cache_load_worker
        self.cache_load_worker.moveToThread(self.cache_load_thread)
        self.cache_load_thread.started.connect(self.cache_load_worker.run)
        self.cache_load_worker.progress_signal.connect(self.window.cache_progress_signal.emit)
        self.cache_load_worker.finished_signal.connect(lambda loaded, attr=attr_name: self.finish_cached_history_load(loaded, attr))
        self.cache_load_worker.error_signal.connect(self.cache_load_failed)
        self.cache_load_worker.finished_signal.connect(self.cache_load_thread.quit)
        self.cache_load_worker.error_signal.connect(self.cache_load_thread.quit)
        self.cache_load_thread.finished.connect(self.cache_load_worker.deleteLater)
        self.cache_load_thread.finished.connect(self.cache_load_thread.deleteLater)
        self.cache_load_thread.start()

    def finish_cached_history_load(self, loaded, attr_name):
        setattr(self.window, attr_name, loaded)
        logging.info(f"当前数据焦点已更新至{loaded.name}")
        self.window.update_progress(1000, 1000)
        self.window.update_status("缓存数据读取完成", 'idle')

    def cache_load_failed(self, message):
        logging.error(f"缓存数据读取失败: {message}")
        QMessageBox.critical(self.window, "缓存读取失败", str(message))
        self.window.update_progress(-1)
        self.window.update_status("缓存读取失败", 'failed')

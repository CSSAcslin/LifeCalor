from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox

from DataManager import ArrayLoadWorker, Data, ProcessedData, collect_array_refs, force_cache_arrays
from ExtraDialog import DataViewAndSelectPop
from .dialog import HistoryCacheManagerDialog
from .manifest import HistoryManifestStore, build_manifest_item, cache_status_for_history_item


class HistoryController:
    def __init__(self, window):
        self.window = window
        self.cache_load_thread = None
        self.cache_load_worker = None
        self.history_cache_dialog = None

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

    def history_cache_manager(self):
        store = self.manifest_store()
        manifest_items = []
        for item in store.load().get("items", []):
            item = dict(item)
            item["file_status"] = store.validate_item_files(item)
            manifest_items.append(item)

        dialog = HistoryCacheManagerDialog(
            params=self.window.tool_params,
            current_items=self.current_history_items(),
            manifest_items=manifest_items,
            cache_summary=self.cache_summary(),
            parent=self.window,
        )
        self.history_cache_dialog = dialog
        dialog.select_history_requested.connect(self.select_history_item)
        dialog.force_cache_requested.connect(self.force_cache_history_item)
        dialog.cleanup_orphans_requested.connect(self.cleanup_orphans)
        dialog.clear_cache_requested.connect(self.window.clear_array_cache_files)
        self.window.update_status("历史与缓存管理", 'working')
        if dialog.exec_():
            params = dialog.get_params()
            self.window.update_param('tool', 'cache_directory', params['cache_directory'])
            self.window.update_param('tool', 'cache_threshold_mb', params['cache_threshold_mb'])
            self.window.update_param('tool', 'cache_cleanup_startup', params['cache_cleanup_startup'])
            self.window.apply_cache_settings()
            logging.info("历史与缓存设置已更新")
        self.window.update_status("准备就绪", 'idle')

    def manifest_store(self) -> HistoryManifestStore:
        cache_dir = self.window.tool_params.get("cache_directory") or self.window.default_cache_directory()
        return HistoryManifestStore(Path(cache_dir))

    def cache_summary(self):
        cache_dir = Path(self.window.tool_params.get("cache_directory") or self.window.default_cache_directory())
        files = list(cache_dir.glob("*.npy")) if cache_dir.exists() else []
        return {
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files if path.exists()),
        }

    def current_history_items(self):
        items = []
        for item in list(Data.history):
            items.append(self.summarize_history_item(item, "Data"))
        for item in list(ProcessedData.history):
            items.append(self.summarize_history_item(item, "ProcessedData"))
        items.reverse()
        return items

    def summarize_history_item(self, item, kind):
        status = cache_status_for_history_item(item)
        if status["cached_count"] and status["memory_bytes"]:
            cache_state = "部分缓存"
        elif status["cached_count"]:
            cache_state = "已缓存"
        else:
            cache_state = "内存"
        return {
            "kind": kind,
            "name": getattr(item, "name", ""),
            "shape": getattr(item, "datashape", ""),
            "dtype": str(getattr(item, "datatype", "")),
            "timestamp": getattr(item, "timestamp", ""),
            "cache_state": cache_state,
            "cached_bytes": status["cached_bytes"],
            "memory_bytes": status["memory_bytes"],
        }

    def find_history_item(self, kind, timestamp):
        if kind == "Data":
            return Data.find_history(timestamp), "data"
        if kind == "ProcessedData":
            return ProcessedData.find_history(timestamp), "processed_data"
        return None, None

    def select_history_item(self, kind, timestamp):
        target, attr_name = self.find_history_item(kind, timestamp)
        self.load_cached_history_async(target, attr_name)

    def force_cache_history_item(self, kind, timestamp):
        target, _ = self.find_history_item(kind, timestamp)
        if target is None:
            QMessageBox.warning(self.window, "缓存落盘", "未找到选中的历史数据")
            return
        self.window.update_status("正在强制缓存历史数据", 'working')
        try:
            force_cache_arrays(target)
            manifest_item = build_manifest_item(target, kind)
            self.manifest_store().upsert(manifest_item)
            logging.info("历史数据已强制缓存: %s", getattr(target, "name", ""))
            QMessageBox.information(self.window, "缓存落盘", "历史数据已保存为可恢复缓存")
            if self.history_cache_dialog is not None:
                self.history_cache_dialog.refresh_current_items(self.current_history_items())
                store = self.manifest_store()
                items = []
                for item in store.load().get("items", []):
                    item = dict(item)
                    item["file_status"] = store.validate_item_files(item)
                    items.append(item)
                self.history_cache_dialog.refresh_manifest_items(items)
        except Exception as exc:
            logging.exception("强制缓存历史数据失败")
            QMessageBox.critical(self.window, "缓存落盘失败", str(exc))
        finally:
            self.window.update_status("准备就绪", 'idle')

    def cleanup_orphans(self):
        deleted = self.window.cleanup_array_cache_orphans()
        removed_manifest_items = self.manifest_store().remove_missing_items()
        QMessageBox.information(self.window, "缓存清理", f"已清理缓存文件 {deleted} 个，移除失效索引 {removed_manifest_items} 条")

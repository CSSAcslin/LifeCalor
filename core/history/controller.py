from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMessageBox

from DataManager import ArrayLoadWorker, Data, ProcessedData, clear_array_cache, collect_array_refs, force_cache_arrays, get_array_store
from ExtraDialog import DataViewAndSelectPop
from .dialog import HistoryCacheManagerDialog
from .manifest import HistoryManifestStore, array_refs_for_manifest, build_manifest_item, cache_status_for_history_item, restore_history_item
from diagnostics import AppError, report_exception, report_warning, show_app_error
from tasks.model import CancellationToken, TaskCancelled


class CachePersistWorker(QObject):
    progress_signal = pyqtSignal(object, object, str)
    finished_signal = pyqtSignal(object)
    cancelled_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, target, cancellation_token=None):
        super().__init__()
        self.target = target
        self.cancellation_token = cancellation_token or CancellationToken()

    def cancel(self):
        self.cancellation_token.cancel()

    @pyqtSlot()
    def run(self):
        try:
            refs = force_cache_arrays(self.target, cancellation_token=self.cancellation_token, progress_callback=self.progress_signal.emit)
            self.cancellation_token.raise_if_cancelled()
            self.finished_signal.emit(refs)
        except TaskCancelled:
            self.cancelled_signal.emit()
        except Exception as exc:
            logging.exception("缓存落盘线程失败", extra={"lifecalor_user_reported": True})
            self.error_signal.emit(str(exc))


class CacheMaintenanceWorker(QObject):
    progress_signal = pyqtSignal(object, object, str)
    finished_signal = pyqtSignal(object)
    cancelled_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, mode, active_refs=None, cancellation_token=None):
        super().__init__()
        self.mode = mode
        self.active_refs = active_refs or []
        self.cancellation_token = cancellation_token or CancellationToken()

    def cancel(self):
        self.cancellation_token.cancel()

    @pyqtSlot()
    def run(self):
        try:
            store = get_array_store(progress_callback=self.progress_signal.emit)
            if self.mode == "clear":
                deleted = store.clear_all(self.cancellation_token)
            else:
                deleted = store.cleanup_orphans(self.active_refs, self.cancellation_token)
            self.finished_signal.emit({"mode": self.mode, "deleted": deleted})
        except TaskCancelled:
            self.cancelled_signal.emit()
        except Exception as exc:
            logging.exception("缓存维护任务失败", extra={"lifecalor_user_reported": True})
            self.error_signal.emit(str(exc))


class HistoryController:
    def __init__(self, window):
        self.window = window
        self.cache_load_thread = None
        self.cache_load_worker = None
        self.history_cache_dialog = None
        self.cache_write_thread = None
        self.cache_write_worker = None
        self.cache_maintenance_threads = {}

    def data_history_view(self):
        if self.window.data is None:
            logging.warning("暂无导入数据历史")
            return
        dialog = DataViewAndSelectPop(datadict=self.window.get_data_all())
        if dialog.exec_():
            selected_timestamp, _ = dialog.get_selected_timestamp()
            selected_data = self.window.data.find_history(selected_timestamp)
            self.load_cached_history_async(selected_data, 'data')

    def process_history_view(self):
        if self.window.processed_data is None:
            logging.warning("暂无处理数据历史")
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
            logging.info("当前数据焦点已更新至%s", getattr(target, "name", ""))
            return

        self.window.update_status("正在读取缓存数据", "working")
        self.window.update_progress(0, 1000)
        task = self.window.task_coordinator.create_task(
            "读取缓存数据", "cache_read", cancel_callback=None
        )
        task.start()
        self.window.task_coordinator.task_updated.emit(task)
        self.cache_load_thread = QThread()
        self.cache_load_worker = ArrayLoadWorker(target, cancellation_token=task.token)
        task.cancel_callback = self.cache_load_worker.cancel
        self.window.cache_load_thread = self.cache_load_thread
        self.window.cache_load_worker = self.cache_load_worker
        self.cache_load_worker.moveToThread(self.cache_load_thread)
        self.cache_load_thread.started.connect(self.cache_load_worker.run)
        self.cache_load_worker.progress_signal.connect(self.window.cache_progress_signal.emit)
        self.cache_load_worker.progress_signal.connect(lambda current, total, message, task_id=task.task_id: self.window.task_coordinator.progress(task_id, current, total, message))
        self.cache_load_worker.finished_signal.connect(lambda loaded, attr=attr_name: self.finish_cached_history_load(loaded, attr, task.task_id))
        self.cache_load_worker.error_signal.connect(lambda message, task_id=task.task_id: self.cache_load_failed(message, task_id))
        self.cache_load_worker.cancelled_signal.connect(lambda task_id=task.task_id: self.cache_load_cancelled(task_id))
        self.cache_load_worker.finished_signal.connect(self.cache_load_thread.quit)
        self.cache_load_worker.error_signal.connect(self.cache_load_thread.quit)
        self.cache_load_worker.cancelled_signal.connect(self.cache_load_thread.quit)
        self.cache_load_thread.finished.connect(self.cache_load_worker.deleteLater)
        self.cache_load_thread.finished.connect(self.cache_load_thread.deleteLater)
        self.cache_load_thread.finished.connect(self._clear_cache_load_handles)
        self.cache_load_thread.start()

    def _clear_cache_load_handles(self):
        self.cache_load_thread = None
        self.cache_load_worker = None
        self.window.cache_load_thread = None
        self.window.cache_load_worker = None

    def cancel_cached_history_load(self):
        if self.cache_load_worker is None:
            logging.info("没有正在读取的缓存任务")
            return False
        self.cache_load_worker.cancel()
        self.window.update_status("正在取消缓存读取", "working")
        return True

    def finish_cached_history_load(self, loaded, attr_name, task_id=None):
        setattr(self.window, attr_name, loaded)
        logging.info("当前数据焦点已更新至%s", getattr(loaded, "name", ""))
        self.window.update_progress(1000, 1000)
        self.window.update_status("缓存数据读取完成", "idle")
        if task_id is not None:
            self.window.task_coordinator.complete(task_id, "缓存数据读取完成")

    def cache_load_cancelled(self, task_id=None):
        logging.info("缓存数据读取已取消")
        self.window.update_progress(-1)
        self.window.update_status("缓存读取已取消", "idle")
        if task_id is not None:
            self.window.task_coordinator.cancelled(task_id, "缓存读取已取消")

    def cache_load_failed(self, message, task_id=None):
        if task_id is None:
            show_app_error(self.window, AppError("缓存读取失败", str(message), stage="缓存读取", severity="error"))
        self.window.update_progress(-1)
        self.window.update_status("缓存读取失败", "failed")
        if task_id is not None:
            self.window.task_coordinator.fail(task_id, str(message))

    def history_cache_manager(self):
        dialog = HistoryCacheManagerDialog(
            params=self.window.tool_params,
            current_items=self.current_history_items(),
            manifest_items=self.manifest_items(),
            cache_summary=self.cache_summary(),
            parent=self.window,
        )
        self.history_cache_dialog = dialog
        dialog.select_history_requested.connect(self.select_history_item)
        dialog.force_cache_requested.connect(self.force_cache_history_item)
        dialog.delete_history_requested.connect(self.delete_current_history_item)
        dialog.cleanup_orphans_requested.connect(self.cleanup_orphans)
        dialog.clear_cache_requested.connect(self.clear_all_cache)
        dialog.recover_manifest_requested.connect(self.restore_manifest_item)
        dialog.delete_manifest_requested.connect(self.delete_manifest_item)
        dialog.refresh_requested.connect(self.refresh_cache_dialog)
        dialog.cancel_load_requested.connect(self.cancel_cached_history_load)
        self.window.update_status("历史与缓存管理", "working")
        if dialog.exec_():
            params = dialog.get_params()
            old_directory = self.window.tool_params.get("cache_directory") or self.window.default_cache_directory()
            new_directory = params["cache_directory"] or self.window.default_cache_directory()
            self.window.update_param("tool", "cache_directory", new_directory)
            self.window.update_param("tool", "cache_threshold_mb", params["cache_threshold_mb"])
            self.window.update_param("tool", "cache_cleanup_startup", params["cache_cleanup_startup"])
            self.window.apply_cache_settings()
            if Path(old_directory) != Path(new_directory):
                self.handle_cache_directory_change(old_directory, new_directory)
            logging.info("历史与缓存设置已更新")
        self.window.update_status("准备就绪", "idle")

    def handle_cache_directory_change(self, old_directory, new_directory):
        message = (
            f"缓存目录已切换到当前目录:\n{new_directory}\n\n"
            f"旧缓存仍保留在旧缓存目录:\n{old_directory}\n"
            "如需恢复旧缓存，可切回旧目录查看可恢复历史。"
        )
        logging.info("缓存目录切换: old=%s new=%s。旧缓存保留。", old_directory, new_directory)
        QMessageBox.information(self.window, "缓存目录已切换", message)

    def manifest_store(self) -> HistoryManifestStore:
        cache_dir = self.window.tool_params.get("cache_directory") or self.window.default_cache_directory()
        return HistoryManifestStore(Path(cache_dir))

    def manifest_items(self):
        store = self.manifest_store()
        store.remove_missing_items()
        items = []
        for item in store.load().get("items", []):
            item = dict(item)
            item["file_status"] = store.validate_item_files(item)
            items.append(item)
        return items

    def refresh_cache_dialog(self):
        if self.history_cache_dialog is None:
            return
        self.history_cache_dialog.refresh_current_items(self.current_history_items())
        self.history_cache_dialog.refresh_manifest_items(self.manifest_items())

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

    def delete_current_history_item(self, kind, timestamp):
        target, attr_name = self.find_history_item(kind, timestamp)
        if target is None:
            report_warning(self.window, "删除历史", "未找到选中的当前历史")
            return False
        answer = QMessageBox.question(
            self.window,
            "删除当前历史",
            "仅从本次历史列表中移除该项，不会删除已落盘的可恢复缓存。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False
        history = Data.history if kind == "Data" else ProcessedData.history
        self._remove_unique_history(history, target)
        if getattr(self.window, attr_name, None) is target:
            setattr(self.window, attr_name, None)
        self.refresh_cache_dialog()
        logging.info("已删除当前历史: %s", getattr(target, "name", ""))
        return True

    def force_cache_history_item(self, kind, timestamp):
        target, _ = self.find_history_item(kind, timestamp)
        if target is None:
            logging.warning("缓存落盘：未找到选中的历史数据")
            self.window.update_status("未找到选中的历史数据", "warning")
            return
        self.window.update_status("正在强制缓存历史数据", "working")
        task = self.window.task_coordinator.create_task("缓存历史数据", "cache_write")
        task.start()
        self.window.task_coordinator.task_updated.emit(task)
        self.cache_write_thread = QThread()
        self.cache_write_worker = CachePersistWorker(target, task.token)
        task.cancel_callback = self.cache_write_worker.cancel
        self.cache_write_worker.moveToThread(self.cache_write_thread)
        self.cache_write_thread.started.connect(self.cache_write_worker.run)
        self.cache_write_worker.progress_signal.connect(lambda current, total, message: self.window.task_coordinator.progress(task.task_id, current, total, message))
        self.cache_write_worker.finished_signal.connect(lambda _refs: self._finish_force_cache(target, kind, task.task_id))
        self.cache_write_worker.cancelled_signal.connect(lambda: self.window.task_coordinator.cancelled(task.task_id, "缓存落盘已取消"))
        self.cache_write_worker.error_signal.connect(lambda message: self._fail_force_cache(target, task.task_id, message))
        self.cache_write_worker.finished_signal.connect(self.cache_write_thread.quit)
        self.cache_write_worker.cancelled_signal.connect(self.cache_write_thread.quit)
        self.cache_write_worker.error_signal.connect(self.cache_write_thread.quit)
        self.cache_write_thread.finished.connect(self.cache_write_worker.deleteLater)
        self.cache_write_thread.finished.connect(self.cache_write_thread.deleteLater)
        self.cache_write_thread.start()

    def _finish_force_cache(self, target, kind, task_id):
        manifest_item = build_manifest_item(target, kind)
        self.manifest_store().upsert(manifest_item)
        self.window.task_coordinator.complete(task_id, "历史数据缓存完成")
        logging.info("历史数据已强制缓存: %s", getattr(target, "name", ""))
        self.refresh_cache_dialog()

    def _fail_force_cache(self, target, task_id, message):
        self.window.task_coordinator.fail(task_id, message)

    def cleanup_orphans(self):
        manifest = self.manifest_store().load()
        active_refs = collect_array_refs(Data.history) + collect_array_refs(ProcessedData.history)
        active_refs += array_refs_for_manifest(manifest)
        return self._start_cache_maintenance("cleanup", active_refs)

    def clear_all_cache(self):
        Data.clear_history(remove_cache=False)
        ProcessedData.clear_history(remove_cache=False)
        return self._start_cache_maintenance("clear")

    def _start_cache_maintenance(self, mode, active_refs=None):
        title = "清除全部缓存" if mode == "clear" else "清理孤立缓存"
        task = self.window.task_coordinator.create_task(title, f"cache_{mode}")
        task.start()
        self.window.task_coordinator.task_updated.emit(task)
        thread = QThread()
        worker = CacheMaintenanceWorker(mode, active_refs, task.token)
        task.cancel_callback = worker.cancel
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress_signal.connect(lambda current, total, message: self.window.task_coordinator.progress(task.task_id, current, total, message))
        worker.finished_signal.connect(lambda result: self._finish_cache_maintenance(task.task_id, result))
        worker.cancelled_signal.connect(lambda: self.window.task_coordinator.cancelled(task.task_id, f"{title}已取消"))
        worker.error_signal.connect(lambda message: self.window.task_coordinator.fail(task.task_id, message))
        worker.finished_signal.connect(thread.quit)
        worker.cancelled_signal.connect(thread.quit)
        worker.error_signal.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda task_id=task.task_id: self.cache_maintenance_threads.pop(task_id, None))
        self.cache_maintenance_threads[task.task_id] = (thread, worker)
        thread.start()
        return task.task_id

    def _finish_cache_maintenance(self, task_id, result):
        removed_manifest_items = 0
        if result["mode"] == "clear":
            removed_manifest_items = self.manifest_store().clear_items()
        else:
            removed_manifest_items = self.manifest_store().remove_missing_items()
        self.refresh_cache_dialog()
        message = f"缓存维护完成：删除文件 {result['deleted']} 个，移除索引 {removed_manifest_items} 条"
        self.window.task_coordinator.complete(task_id, message)
        logging.info(message)

    def restore_manifest_item(self, item_id):
        store = self.manifest_store()
        manifest = store.load()
        item = next((entry for entry in manifest.get("items", []) if entry.get("id") == item_id), None)
        if item is None:
            report_warning(self.window, "恢复历史", "未找到可恢复历史索引")
            return None
        status = store.validate_item_files(item)
        if not status["ok"]:
            report_warning(self.window, "恢复历史", "缓存文件缺失，无法恢复该历史项")
            return None
        try:
            restored = restore_history_item(item)
            if item.get("kind") == "Data":
                self._append_unique_history(Data.history, restored)
                attr_name = "data"
            elif item.get("kind") == "ProcessedData":
                self._append_unique_history(ProcessedData.history, restored)
                attr_name = "processed_data"
            else:
                raise ValueError(f"不支持恢复的历史类型: {item.get('kind')}")
            logging.info("已恢复历史缓存: %s", getattr(restored, "name", ""))
            self.refresh_cache_dialog()
            self.load_cached_history_async(restored, attr_name)
            QMessageBox.information(self.window, "恢复历史", "历史数据已恢复，并正在设为当前数据")
            return restored
        except Exception as exc:
            report_exception(self.window, "恢复历史失败", str(exc), exc, stage="恢复历史缓存")
            return None

    def delete_manifest_item(self, item_id):
        answer = QMessageBox.question(
            self.window,
            "删除可恢复历史",
            "是否同时删除该索引独占的缓存文件？\n选择 No 将只删除索引，保留缓存文件。",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No,
        )
        if answer == QMessageBox.Cancel:
            return None
        result = self.manifest_store().delete_item(item_id, delete_cache_files=(answer == QMessageBox.Yes))
        self.refresh_cache_dialog()
        if result["removed"]:
            QMessageBox.information(self.window, "删除可恢复历史", f"已删除索引，删除独占缓存文件 {result['deleted_files']} 个")
        else:
            report_warning(self.window, "删除可恢复历史", "未找到选中的可恢复历史索引")
        return result

    @staticmethod
    def _append_unique_history(history, item):
        timestamp = getattr(item, "timestamp", None)
        serial_number = getattr(item, "serial_number", None)
        for existing in list(history):
            if getattr(existing, "timestamp", None) == timestamp or getattr(existing, "serial_number", None) == serial_number:
                history.remove(existing)
                break
        history.append(item)

    @staticmethod
    def _remove_unique_history(history, item):
        timestamp = getattr(item, "timestamp", None)
        serial_number = getattr(item, "serial_number", None)
        for existing in list(history):
            if existing is item or getattr(existing, "timestamp", None) == timestamp or getattr(existing, "serial_number", None) == serial_number:
                history.remove(existing)
                return True
        return False

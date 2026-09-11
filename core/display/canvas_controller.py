import logging

from PyQt5.QtWidgets import QMessageBox

from DataManager import Data, ImagingData
from ExtraDialog import DataViewAndSelectPop


class DisplayCanvasController:
    def __init__(self, window):
        self.window = window

    def _apply_colormap_mode(self, data_display):
        window = self.window
        data_display.colormode = window.tool_params["colormap"] if window.tool_params["use_colormap"] else None
        return data_display

    def _create_display_data(self, data):
        return self._apply_colormap_mode(ImagingData.create_image(data))

    def add_new_canvas(self, assign_data=None):
        window = self.window
        image_display = window.image_display
        if window.data is None and window.processed_data is None:
            logging.warning("请先导入或处理数据")
            return False
        if len(image_display.display_canvas) >= 4:
            logging.warning("已达到最大显示区域数量 (4)，请先删除或覆盖一个画布")
            return False

        window.update_progress(-1)
        data_display = None
        if assign_data is not None:
            data_display = self._create_display_data(assign_data)
            if isinstance(assign_data, Data):
                logging.info("数据选择成功（原初）")
            else:
                logging.info("数据选择成功（处理）")
        else:
            dialog = DataViewAndSelectPop(
                datadict=window.get_data_all(),
                processed_datadict=window.get_processed_data_all(),
                add_canvas=True,
            )
            if dialog.exec_():
                selected_timestamp, selected_table = dialog.get_selected_timestamp()
                if selected_table == "data" and window.data is not None:
                    for data in window.data.history:
                        if data.timestamp == selected_timestamp:
                            data_display = self._create_display_data(data)
                            logging.info("数据选择成功（原初）")
                            break
                elif window.processed_data is not None:
                    for data in window.processed_data.history:
                        if data.timestamp == selected_timestamp:
                            data_display = self._create_display_data(data)
                            logging.info("数据选择成功（处理）")
                            break

        added = image_display.add_canvas(data_display) if data_display is not None else False
        if added:
            window.focus_canvas = added.id
            window.canvas_signal_connect()
        window.update_status("准备就绪", "idle")
        return added

    def load_image(self, data_type="original", other_params: str = None, origin_data=None):
        window = self.window
        image_display = window.image_display
        if data_type != "original":
            return None

        source_data = origin_data if origin_data is not None else window.data
        if source_data is None:
            logging.warning("没有可显示的数据")
            return False

        data_display = self._create_display_data(source_data)
        window.imaging_main = data_display
        if not image_display.display_canvas:
            added = image_display.add_canvas(data_display)
            if not added:
                return False
            window.canvas_signal_connect()
        else:
            target = image_display.current_canvas() or image_display.display_canvas[-1]
            choice = self._ask_canvas_action(target)
            if choice == "overwrite":
                if not image_display.replace_canvas(target.id, data_display):
                    return False
            elif choice == "new":
                added = image_display.add_canvas(data_display)
                if not added:
                    return False
                window.canvas_signal_connect()
            else:
                return False

        shape = getattr(window.data, "datashape", ())
        if len(shape) >= 3:
            window.region_x_input.setMaximum(shape[1])
            window.region_y_input.setMaximum(shape[2])
        return True

    def _ask_canvas_action(self, target_canvas):
        msg_box = QMessageBox(self.window)
        msg_box.setWindowTitle("画布操作")
        msg_box.setText(f"当前画布：{target_canvas.windowTitle()}")
        msg_box.setInformativeText("覆盖只替换当前画布；新建会保留已有画布。")
        overwrite_btn = msg_box.addButton("覆盖当前", QMessageBox.ActionRole)
        new_btn = msg_box.addButton("新建画布", QMessageBox.ActionRole)
        cancel_btn = msg_box.addButton("暂不显示", QMessageBox.RejectRole)
        if len(self.window.image_display.display_canvas) >= 4:
            new_btn.setEnabled(False)
            new_btn.setToolTip("最多同时保留四个画布")
        msg_box.exec_()
        if msg_box.clickedButton() == overwrite_btn:
            return "overwrite"
        if msg_box.clickedButton() == new_btn:
            return "new"
        if msg_box.clickedButton() == cancel_btn:
            return "hide"
        return None

    def upgrade_and_imaging(self, data, key: str):
        processed_data = data.upgrade_processed(key)
        if processed_data is not None:
            self.window.processed_data = processed_data
            self.window.add_new_canvas(self.window.processed_data)
            return None
        return logging.error("导入成像失败")

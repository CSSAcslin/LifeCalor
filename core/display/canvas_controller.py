import logging

from PyQt5.QtWidgets import QMessageBox

from DataManager import Data, ImagingData
from ExtraDialog import DataViewAndSelectPop


class DisplayCanvasController:
    def __init__(self, window):
        self.window = window

    def _apply_colormap_mode(self, data_display):
        window = self.window
        data_display.colormode = window.tool_params['colormap'] if window.tool_params['use_colormap'] else None
        return data_display

    def _create_display_data(self, data):
        return self._apply_colormap_mode(ImagingData.create_image(data))

    def add_new_canvas(self, assign_data=None):
        window = self.window
        if window.data is None and window.processed_data is None:
            logging.warning('请先导入或处理数据')
            return
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
                if selected_table == 'data' and window.data is not None:
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
        if data_display is not None:
            window.image_display.add_canvas(data_display)
            window.focus_canvas = window.image_display.cursor_id
        window.canvas_signal_connect()
        window.update_status("准备就绪", 'idle')

    def load_image(self, data_type='original', other_params: str = None, origin_data=None):
        window = self.window
        if len(window.image_display.display_canvas) == 0:
            if data_type == 'original':
                window.imaging_main = ImagingData.create_image(window.data)
            window.image_display.add_canvas(window.imaging_main)
            window.canvas_signal_connect()
        else:
            if data_type == 'original':
                choice = self._ask_canvas_action()
                if choice == 'overwrite':
                    window.image_display.del_canvas(-1)
                    window.imaging_main = ImagingData.create_image(window.data)
                    window.add_new_canvas(origin_data)
                elif choice == 'new':
                    window.add_new_canvas(origin_data)
                elif choice == 'hide':
                    return False

        window.region_x_input.setMaximum(window.data.datashape[1])
        window.region_y_input.setMaximum(window.data.datashape[2])
        return None

    def _ask_canvas_action(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("画布操作")
        msg_box.setText("请选择是否要覆盖当前画布或新建画布")
        overwrite_btn = msg_box.addButton("覆盖", QMessageBox.ActionRole)
        new_btn = msg_box.addButton("新建", QMessageBox.ActionRole)
        hide_btn = msg_box.addButton("隐藏", QMessageBox.ActionRole)
        msg_box.exec_()
        if msg_box.clickedButton() == overwrite_btn:
            return 'overwrite'
        if msg_box.clickedButton() == new_btn:
            return 'new'
        if msg_box.clickedButton() == hide_btn:
            return 'hide'
        return None

    def upgrade_and_imaging(self, data, key: str):
        processed_data = data.upgrade_processed(key)
        if processed_data is not None:
            self.window.processed_data = processed_data
            self.window.add_new_canvas(self.window.processed_data)
            return None
        return logging.error("导入成像失败")

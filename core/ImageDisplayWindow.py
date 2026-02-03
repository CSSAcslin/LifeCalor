import logging
import time
from enum import Enum
from math import atan2, pi, cos, sin

import numpy as np
from PyQt5.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QTransform, QIcon, QPen, QBrush, QColor, QPainterPath, \
    QLinearGradient, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QToolBar, QAction, QDockWidget, QStyle,
                             QGraphicsRectItem, QActionGroup, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsItem,
                             QGraphicsPathItem, QMenu, QInputDialog, QColorDialog, QToolButton, QDialogButtonBox,
                             QDialog, QMessageBox, QGraphicsTextItem, QSizePolicy, QCheckBox, QGraphicsObject
                             )
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QSize, QTimer, QDateTime, QLineF, QPointF, QPoint


from DataManager import ImagingData, ColorMapManager, PublicEasyMethod
from ExtraDialog import ROIInfoDialog, ColorMapDialog, DataExportDialog, ParamsResetDialog
from widget.AdvancedTimeline import AdvancedTimeline

import matplotlib.cm as cm

class ImageDisplayWindow(QMainWindow):
    """图像显示管理"""
    add_canvas_signal = pyqtSignal()
    draw_result_signal = pyqtSignal(str, int, object,dict)
    params_update_signal = pyqtSignal(dict)
    image_style_change_signal = pyqtSignal(object,dict)
    image_export_signal = pyqtSignal(object, str, str, str, bool, dict)
    def __init__(self, params,parent=None):
        super().__init__(parent)
        self.display_canvas = []
        self.cursor_id = 0
        self.current_tool = None
        self.anchor_active = False
        self.actions_all = {}
        self.tool_parameters = params
        self.parent = parent

        self.init_tool_bars()

    def init_tool_bars(self):
        Canvas_bar = QToolBar('Canvas')
        add_canvas = QAction(QIcon(":icons/icon_add.svg"),'Add', self)
        add_canvas.setStatusTip("Add new canvas")
        add_canvas.triggered.connect(lambda: self.add_canvas_signal.emit())
        Canvas_bar.addAction(add_canvas)

        del_canvas = QAction(QIcon(":icons/icon_del.svg"),'Del', self)
        del_canvas.setStatusTip("Delete the latest canvas")
        del_canvas.triggered.connect(self.del_canvas)
        Canvas_bar.addAction(del_canvas)

        cursor = QAction(QIcon(":icons/icon_cursor.svg"),'Cursor', self)
        cursor.setStatusTip("Cursor")
        cursor.triggered.connect(self.cursor)
        Canvas_bar.addAction(cursor)

        Canvas_bar.setIconSize(QSize(36, 36))
        self.addToolBar(Canvas_bar)

        self.Drawing_bar = QToolBar('Drawing')
        self.drawing_action_group = QActionGroup(self)
        self.drawing_action_group.setExclusive(True)
        self.Drawing_bar.setIconSize(QSize(36, 36)) # 已经在样式表设置,样式表设置不管用

        # 创建所有绘图工具
        self.create_drawing_action(self.Drawing_bar,'Pen', "Draw the pen",'画笔')
        self.create_drawing_action(self.Drawing_bar,'Line', "Draw the line",'直线')
        self.create_drawing_action(self.Drawing_bar,'Rect', "Draw the rect",'矩形')
        self.create_drawing_action(self.Drawing_bar,'Ellipse', "Draw the ellipse",'椭圆')

        self.create_drawing_action(self.Drawing_bar,'Eraser', "Draw the eraser",'橡皮擦')
        self.create_drawing_action(self.Drawing_bar,'Fill', "Draw the fill",'填充')

        self.create_drawing_action(self.Drawing_bar,'Anchor', "Anchor",'光标')
        self.create_drawing_action(self.Drawing_bar,'V-line', "Draw the vector line",'向量选区')
        self.create_drawing_action(self.Drawing_bar,'V-rect', "Draw the vector rect",'矩形选区')

        self.Drawing_bar.addSeparator()
        self.create_drawing_action(self.Drawing_bar,'Color', "Set the Style",'样式') # 原本色彩设置现改为样式设置
        self.create_drawing_action(self.Drawing_bar, 'Accept', "Accept Roi and check canvas", '确认')
        self.create_drawing_action(self.Drawing_bar, 'Reset', "Reset all", '重置')
        self.create_drawing_action(self.Drawing_bar,'Export','Export your canvas','导出')

        # # 添加默认选中项（可选）
        # draw_pen.setChecked(True)
        self.addToolBar(self.Drawing_bar)

    def create_drawing_action(self, toolbar, name, statustip,tooltip, slot = None):
        """创建绘图工具动作并添加上下文菜单"""
        # 创建动作
        icon_dict = {
            'Pen': ':icons/icon_pen.svg',
            'Line': ':icons/icon_line.svg',
            'Rect': ':icons/icon_rect.svg',
            'Ellipse': ':icons/icon_ellipse.svg',
            'Eraser': ':icons/icon_eraser.svg',
            'Fill': ':icons/icon_fill.svg',
            'Color': ':icons/icon_color.svg',
            'Anchor': ':icons/icon_anchor.svg',
            'V-line': ':icons/icon_v-line.svg',
            'V-rect': ':icons/icon_v-rect.svg',
            'Accept': ':icons/icon_accept.svg',
            'Cancel': ':icons/icon_cancel.svg',
            'Reset': ':icons/icon_reset.svg',
            'Export': ':icons/icon_export.svg',
        }

        action = QAction(QIcon(icon_dict[name]), name, self)
        action.setStatusTip(statustip)
        action.setToolTip(tooltip)
        if name not in ['Color','Accept', 'Cancel','Reset','Export']:
            action.triggered.connect(lambda checked: self.set_tools(name if checked else None))
            action.setCheckable(True)
        elif name == 'Color':
            action.triggered.connect(lambda checked: self.set_color_style_dialog())
        elif name == 'Accept':
            action.triggered.connect(lambda checked: self.show_roi_info_dialog())
        elif name == 'Reset':
            action.triggered.connect(lambda checked: self.reset_all_canvas())
        elif name == 'Export':
            action.triggered.connect(lambda checked: self.export_canvas_dialog())

        # 添加到动作组和工具栏
        self.drawing_action_group.addAction(action)
        toolbar.addAction(action)
        self.actions_all[name] = action

        # 创建自定义按钮并替换默认按钮
        button = ToolButtonWithMenu(name, self)
        button.setDefaultAction(action)
        button.contextMenuRequested.connect(self.show_tool_context_menu)

        # 替换工具栏中的默认按钮
        for default_button in self.Drawing_bar.findChildren(QToolButton):
            if default_button.defaultAction() == action:
                self.Drawing_bar.removeAction(action)
                self.Drawing_bar.addWidget(button)
                break

        return action

    def show_tool_context_menu(self, tool_name, button):
        """显示工具的上下文菜单"""
        menu = QMenu(self)

        # 根据工具类型添加不同的菜单项
        if tool_name in ['Pen', 'Line']:
            width_action = menu.addAction(f"设置画笔大小 (当前: {self.tool_parameters['pen_size']}像素)")
            width_action.triggered.connect(lambda: self.set_pen_size())

            pen_color_action = menu.addAction("设置颜色")
            pen_color_action.triggered.connect(lambda: self.set_pen_color())

        elif tool_name in ['Rect', 'Ellipse']:
            width_action = menu.addAction(f"设置画笔大小 (当前: {self.tool_parameters['pen_size']}像素)")
            width_action.triggered.connect(lambda: self.set_pen_size())

            pen_color_action = menu.addAction("设置边框颜色")
            pen_color_action.triggered.connect(lambda: self.set_pen_color())

            fill_action = menu.addAction(
                f"填充颜色 (当前: {'是' if self.tool_parameters['auto_fill'] else '否'})")
            fill_action.triggered.connect(lambda: self.toggle_fill_shape())

            fill_color_action = menu.addAction("设置填充颜色")
            fill_color_action.triggered.connect(lambda: self.set_fill_color())

        elif tool_name == 'Eraser':
            size_action = menu.addAction(f"设置大小 (当前: {self.tool_parameters['pen_size']}像素)")
            size_action.triggered.connect(lambda: self.set_pen_size())

        elif tool_name == 'Fill':
            fill_color_action = menu.addAction("设置填充颜色")
            fill_color_action.triggered.connect(lambda: self.set_fill_color())

        elif tool_name in  ['V-line','V-rect','Anchor']:
            if tool_name == 'V-line':
                width_action = menu.addAction(f"设置选区宽度 (当前: {self.tool_parameters['vector_width']}像素)")
                width_action.triggered.connect(lambda: self.set_vector_width())
            elif tool_name == 'Anchor':
                width_action = menu.addAction(f"设置快速提取功能")
                width_action.triggered.connect(lambda: self.set_anchor_select())

            vector_color_action = menu.addAction("设置矢量颜色")
            vector_color_action.triggered.connect(lambda: self.set_vector_color())
        else:
            return False

        # 显示菜单
        menu.exec_(button.mapToGlobal(QPoint(0, button.height())))
        # 更新设置
        if not self.display_canvas:
            return None
        for canvas in self.display_canvas:
            canvas.set_toolset(self.tool_parameters)
        return True

    def set_pen_size(self):
        """设置工具宽度"""
        dialog = WidthSliderDialog(
            self,
            current_value=self.tool_parameters['pen_size'],
            min_value=1,
            max_value=50,
            title=f"设置画笔大小"
        )

        if dialog.exec_() == QDialog.Accepted:
            self.tool_parameters['pen_size'] = dialog.get_value()
            self.params_update_signal.emit(self.tool_parameters)

    def set_vector_width(self):
        """设置工具宽度"""
        dialog = WidthSliderDialog(
            self,
            current_value=self.tool_parameters['vector_width'],
            min_value=1,
            max_value=50,
            title=f"设置选区宽度"
        )

        if dialog.exec_() == QDialog.Accepted:
            self.tool_parameters['vector_width'] = dialog.get_value()
            self.params_update_signal.emit(self.tool_parameters)

    def set_pen_color(self):
        """选择画笔颜色"""
        color = QColorDialog.getColor(
            QColor(self.tool_parameters['pen_color']),
            self,
            f"选择画笔颜色"
        )
        if color.isValid():
            self.tool_parameters['pen_color'] = color.name()
            self.params_update_signal.emit(self.tool_parameters)

    def set_vector_color(self):
        """选择矢量工具颜色"""
        color = QColorDialog.getColor(
            QColor(self.tool_parameters['vector_color']),
            self,
            f"选择画笔颜色"
        )
        if color.isValid():
            self.tool_parameters['vector_color'] = color.name()
            self.params_update_signal.emit(self.tool_parameters)

    def toggle_fill_shape(self):
        """切换形状填充状态(这个就还没实装）"""
        self.tool_parameters['auto_fill'] = not self.tool_parameters['auto_fill']
        self.params_update_signal.emit(self.tool_parameters)

    def set_fill_color(self):
        """选择画笔颜色"""
        color = QColorDialog.getColor(
            QColor(self.tool_parameters['fill_color']),
            self,
            f"选择画笔颜色", QColorDialog.ShowAlphaChannel
        )
        if color.isValid():
            self.tool_parameters['fill_color'] = color.name()
            self.params_update_signal.emit(self.tool_parameters)

    def set_anchor_select(self):
        """锚点快速提取功能设置"""
        self.dialog = AnchorSelectDialog(self.tool_parameters, self)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def set_cursor_id(self,cursor_id):
        self.cursor_id = cursor_id
        if not self.display_canvas:
            logging.warning("请先创建图像画板")
            return
        # if self.current_tool is not None:
        #     self.display_canvas[self.cursor_id].set_drawing_tool(self.current_tool)
        if self.anchor_active:
            self.display_canvas[self.cursor_id].set_anchor_mode(True)

    def add_canvas(self,data):
        """新增图像显示画布"""
        if len(self.display_canvas) >= 4:
            logging.warning("已达到最大显示区域数量 (4)")
            return False
        canvas_id = len(self.display_canvas)
        self.cursor_id = canvas_id
        # self.display_data.append(data)
        data.canvas_num = canvas_id
        new_canvas = SubImageDisplayWidget(name=f'{canvas_id}-{data.source_name}',canvas_id=canvas_id,data=data,args_dict=self.tool_parameters,parent=self)
        self.display_canvas.append(new_canvas)
        self.add_dock(self.display_canvas[-1])
        if self.tool_parameters['use_colormap']:
            self.image_style_change_signal.emit(self.display_canvas[-1].data, self.tool_parameters)
            # self.display_canvas[-1].set_toolset(self.tool_parameters)
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.display_canvas[-1])

    def _remove_single_canvas(self, canvas_id):
        """删除单个画布"""
        # 从布局中移除并删除DockWidget
        for dock in self.findChildren(QDockWidget):
            if hasattr(dock, 'id') and dock.id == canvas_id:
                self.removeDockWidget(dock)
                dock.deleteLater()
                break

        # 从display_canvas列表中移除
        for i, canvas in enumerate(self.display_canvas):
            if canvas.id == canvas_id:
                del self.display_canvas[i]
                break

        return True

    def del_canvas(self,canvas_id = False):
        """删除画布
             None - 删除最后一个画布
              int - 删除指定ID的画布
               -1 - 删除所有画布
        """
        if not self.display_canvas:
            return False
        # 删除最后添加的画布
        if canvas_id is False: # 不能改为not 否则0也会被判定
            canvas_id = self.display_canvas[-1].id
        # 删除所有画布
        elif canvas_id == -1: # 全部清除
            all_ids = [c.id for c in self.display_canvas]
            for cid in all_ids:
                self._remove_single_canvas(cid)
            self.parent.canvas_signal_connect()
            return True

        # 删除单个canvas_id画布
        self._remove_single_canvas(canvas_id)
        for i, canvas in enumerate(self.display_canvas):
            if canvas.id > canvas_id:
                canvas.id -= 1
        self.parent.canvas_signal_connect()
        return True

    def add_dock(self, dock):
        """根据区域数量更新布局"""
        # 获取所有DockWidget并按id排序

        dock_count = len(self.display_canvas)

            # 根据目标DockWidget数量重新布局
        if dock_count == 1:
            self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        elif dock_count == 2:
            self.addDockWidget(Qt.RightDockWidgetArea, dock)

        elif dock_count == 3:
            # 创建左侧区域
            self.addDockWidget(Qt.LeftDockWidgetArea, dock)
            # # 垂直分割左侧区域
            self.splitDockWidget(self.display_canvas[0], dock, Qt.Vertical)
            # # 添加右侧区域
            # self.addDockWidget(Qt.RightDockWidgetArea, target_docks[2])

        elif dock_count >= 4:
            # 只处理前4个
            # docks = target_docks[:4]
            # # 创建左侧区域
            # self.addDockWidget(Qt.LeftDockWidgetArea, docks[0])
            # # 垂直分割左侧区域
            # self.splitDockWidget(docks[0], docks[1], Qt.Vertical)
            # 添加右侧区域
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            # # 垂直分割右侧区域
            # self.splitDockWidget(docks[2], docks[3], Qt.Vertical)

    def reset_all_canvas(self):
        if not self.display_canvas:
            logging.warning("没有画布可以重置")
            return
        for canvas in self.display_canvas:
            canvas.clear_anchor()
            canvas.clear_vector_line()
            canvas.clear_vector_rect()
            canvas.clear_draw_layer()
            canvas.clear_fast_selection()
            canvas.reset_view()

    def set_tools(self,tool_name:str):
        if not self.display_canvas:
            logging.warning("请先创建图像画板")
            return
        for canvas in self.display_canvas:
            canvas.set_drawing_tool(tool_name)
        self.current_tool = tool_name
        if tool_name == 'Anchor':
            self.anchor_active = not self.anchor_active
            action = self.actions_all.get('Anchor')
            action.setChecked(self.anchor_active)
            for canvas in self.display_canvas:
                canvas.set_anchor_mode(self.anchor_active)
        else:
            self.anchor_active = False

    def cursor(self):
        if not self.display_canvas:
            self.display_canvas[self.cursor_id].set_drawing_tool(None)
            self.anchor_active = False
                # 清除所有画板的十字标
            for canvas in self.display_canvas:
                canvas.set_anchor_mode(self.anchor_active)
        for action in self.actions_all.values():
            action.setChecked(False)

    def get_draw_roi(self,canvas_id = None):
        """获取绘制的roi"""
        if canvas_id is None:
            canvas_id = self.cursor_id
        draw_layer = self.display_canvas[canvas_id].draw_roi
        bool_mask = draw_layer >0

        return draw_layer.copy(), bool_mask

    def get_all_canvas_info(self):
        """收集所有画布的信息"""
        canvas_info = []
        if not self.display_canvas:
            QMessageBox.warning(self,"图像错误","当前并没有画布和数据")
            return False
        for canvas in self.display_canvas:
            info = {
                'canvas_id': canvas.id,
                'image_name': canvas.windowTitle(),
                'image_size': canvas.data.framesize if hasattr(canvas, 'data') else (0, 0),
                'ROIs': []
            }

            # 收集矢量矩形ROI
            if hasattr(canvas, 'v_rect_roi') and canvas.v_rect_roi:
                (x, y), width, height = canvas.v_rect_roi
                info['ROIs'].append({
                    'type': 'v_rect',
                    'position': (x, y),
                    'size': (width, height)
                })

            # 收集矢量线ROI
            if hasattr(canvas, 'vector_line') and canvas.vector_line:
                line = canvas.vector_line.line()
                info['ROIs'].append({
                    'type': 'v_line',
                    'start': (line.x1(), line.y1()),
                    'end': (line.x2(), line.y2()),
                    'width': canvas.vector_width
                })

            # 收集锚点ROI
            if hasattr(canvas, 'anchor_pos') and canvas.anchor_pos:
                x, y = canvas.anchor_pos
                anchor_dict = {
                    'type': 'anchor',
                    'position': (x, y)
                }
                if canvas.anchor_mask is not None:
                    anchor_dict['anchor_mask'] = np.count_nonzero(canvas.anchor_mask)
                info['ROIs'].append(anchor_dict)

            # 收集像素ROI
            if hasattr(canvas, 'draw_roi') and np.any(canvas.draw_roi):
                draw_layer = canvas.draw_roi
                info['ROIs'].append({
                    'type': 'pixel_roi',
                    'counts': np.count_nonzero(canvas.draw_roi),
                    'draw_mask' : draw_layer.copy(),
                    'bool_mask' : draw_layer > 0,
                })

            canvas_info.append(info)

        return canvas_info

    def show_roi_info_dialog(self):
        """显示ROI信息的对话框"""
        if not self.display_canvas:
            QMessageBox.warning(self, "图像错误", "当前没有显示任何图像画布")
            return

        dialog = ROIInfoDialog(self.get_all_canvas_info(), parent=self)
        if dialog.exec_():
            aim_id = dialog.canvas_id
            if dialog.roi_type == 'pixel_roi':
                self.draw_result_signal.emit('pixel_roi',aim_id,self.get_draw_roi(aim_id),dialog.selected_roi_info)
            elif dialog.roi_type == 'v_line':
                self.draw_result_signal.emit('v_line', aim_id, self.display_canvas[aim_id].vector_line,dialog.selected_roi_info)
            elif dialog.roi_type == 'v_rect':
                self.draw_result_signal.emit('v_rect', aim_id, self.display_canvas[aim_id].v_rect_roi,dialog.selected_roi_info)
            else:
                pass

    def set_color_style_dialog(self):
        """这是设置伪彩色的"""
        info = []
        for canvas in self.display_canvas:
            info.append(f'{canvas.id}-{canvas.windowTitle()}')

        dialog = ColorMapDialog(self,ColorMapManager().get_colormap_names(),info,params = self.tool_parameters)
        if dialog.exec_() == QDialog.Accepted:
            tool_dict = self.tool_parameters.copy()
            tool_dict.update(dialog.get_value())
            canvas = dialog.canvas_index
            if canvas == -1:
                self.tool_parameters.update(tool_dict) # 水平
                self.params_update_signal.emit(self.tool_parameters) # 向上
                if not self.display_canvas:
                    return None
                for canvas in self.display_canvas:
                    self.image_style_change_signal.emit(canvas.data, tool_dict) # 去处理数据
                    canvas.set_toolset(self.tool_parameters) # 向下
            else:
                if not self.display_canvas:
                    return None
                self.image_style_change_signal.emit(self.display_canvas[canvas].data, tool_dict)
                self.display_canvas[canvas].set_toolset(tool_dict)
        pass

    def update_canvas_by_stamp(self,data):
        """根据data溯源所在canvas"""
        if not self.display_canvas:
            return
        try:
            canvas = next(canvas for canvas in self.display_canvas if canvas.data.timestamp == data.timestamp)
            canvas.update_after_set()
        except Exception as e:
            raise IndexError(f"canvas 未找到:{e}")

    def export_canvas_dialog(self):
        """导出画布数据的对话框"""
        if not self.display_canvas:
            QMessageBox.warning(self, "图像错误", "当前没有显示任何图像画布")
            return
        info = []
        is_temporal = []
        for canvas in self.display_canvas:
            info.append(f'{canvas.id}-{canvas.windowTitle()}')
            is_temporal.append(canvas.is_temporal)
        dialog = DataExportDialog(self,export_type='canvas',canvas_info=info,is_temporal = is_temporal)
        if dialog.exec_():
            directory = dialog.directory
            prefix = dialog.text_edit.text().strip()
            filetype = dialog.type_combo.currentText()
            canvas_id = dialog.canvas_selector.currentIndex()
            arg_dict = dialog.get_values()
            arg_dict.update({'max_bound': self.display_canvas[canvas_id].max_value,
                             'min_bound': self.display_canvas[canvas_id].min_value,
                             'cmap': self.display_canvas[canvas_id].colormap})
            self.image_export_signal.emit(self.display_canvas[canvas_id].data,
                                          directory,prefix,filetype,
                                          self.display_canvas[canvas_id].is_temporal,
                                          arg_dict)
            logging.info(f"开始导出图像数据{info[canvas_id]}")


class SubImageDisplayWidget(QDockWidget):
    """子图像显示部件"""
    mouse_position_signal = pyqtSignal(int, int, int, object,float)
    mouse_clicked_signal = pyqtSignal(int, int, int)
    current_canvas_signal = pyqtSignal(int)
    draw_result_signal = pyqtSignal(str,int,object,dict)
    get_fast_selection = pyqtSignal(object, np.ndarray, str, str)

    def __init__(self, parent=None,canvas_id = None,name = None, data :ImagingData = None, args_dict :dict = None):
        super().__init__(name, parent)
        self.parent_window = parent
        self.id = canvas_id
        self.data = data
        self.current_image = None
        self.mouse_pos = None
        self.current_time_idx = 0
        self.max_time_idx = self.data.totalframes if self.data.is_temporary else 0
        self.is_temporal = self.data.is_temporary
        self.drag_start_pos = None  # 拖动起始位置
        self.last_scale = None
        self.initial_scale = None
        self.min_scale = 0.1
        self.max_scale = 30.0
        self.min_value = data.imagemin
        self.max_value = data.imagemax
        self.draw_roi = np.zeros(data.framesize, dtype=np.uint8) # ROI结果（像素）
        self.data_layer = None # 数据层
        self.draw_layer = None # 绘图层
        self.top_pixmap = None # 顶层绘图层的pixmap
        self.draw_layer_opacity = 0.8 # 绘图层透明度

        # 伪彩色控制（参数设置与toolset合并）
        self.colorbar_item = None  # 颜色条图形项
        self.colorbar_width = 3  # 颜色条宽度
        self.colorbar_padding = 5  # 颜色条边距
        self.color_map_manager = ColorMapManager()  # 伪彩色管理器
        self.colormap = None

        # 工具响应
        self.drawing_tool = None
        self.drawing = False
        self.start_pos = None
        self.end_pos = None
        self.rect_item = None # 存储向量矩形
        self.movable_rect_item = None # 可拖动的向量矩形
        self.temp_pixmap = None # 临时像素画布
        self.v_rect_roi = None # 矢量矩形蒙版结果（左上角坐标（x,y), 宽度, 高度）
        self.anchor_active = False
        self.anchor_item = None  # 存储十字标图形项
        self.anchor_pos = None  # 存储十字标位置
        self.anchor_mask = None # 存储光标蒙版
        self.line_item = None  # 向量线模版
        self.vector_line = None # 向量线item

        # 绘图设置
        self.args_dict = args_dict
        self.set_toolset(self.args_dict)

        # 播放相关
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.auto_play_update)
        self.play_start_time = 0
        self.play_paused_time = 0
        self.is_playing = False

        self.init_ui()
        self.map_view = False

    def init_ui(self):
        widget = QWidget(self)
        layout = QVBoxLayout(self)

        # 创建图形视图和场景
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        # self.graphics_view.setMinimumSize(350, 350)
        self.data_layer = QGraphicsPixmapItem()
        self.draw_layer = QGraphicsPixmapItem()
        self.scene.addItem(self.data_layer)
        self.scene.addItem(self.draw_layer)

        layout.addWidget(self.graphics_view)

        # 视图设置
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.graphics_view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.graphics_view.setResizeAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setTransform(QTransform())
        # 鼠标功能响应
        self.graphics_view.wheelEvent = self.wheel_event  # 滚轮缩放
        self.graphics_view.mouseMoveEvent = self.mouse_move_event
        self.graphics_view.mousePressEvent = self.mouse_press_event
        self.graphics_view.mouseReleaseEvent = self.mouse_release_event

        slider_layout = QHBoxLayout()
        # 自动播放按钮
        self.start_button = QPushButton()
        self.start_button.setIcon(QIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay)))
        self.start_button.setIconSize(QSize(12, 12))
        self.start_button.setFixedSize(16,16)
        self.start_button.clicked.connect(self.start_auto_play)
        # 暂停播放按钮
        self.pause_button = QPushButton()
        self.pause_button.setIcon(QIcon(QApplication.style().standardIcon(QStyle.SP_MediaPause)))
        self.pause_button.setIconSize(QSize(12, 12))
        self.pause_button.setFixedSize(16, 16)
        self.pause_button.clicked.connect(self.pause_auto_play)
        self.pause_button.setEnabled(False)  # 初始不可用
        # 重置播放按钮
        self.reset_button = QPushButton()
        self.reset_button.setIcon(QIcon(QApplication.style().standardIcon(QStyle.SP_MediaSkipBackward)))
        self.reset_button.setIconSize(QSize(12, 12))
        self.reset_button.setFixedSize(16, 16)
        self.reset_button.clicked.connect(self.reset_auto_play)
        self.reset_button.setEnabled(False)  # 初始不可用

        # 全新时间轴
        # self.time_slider = QSlider(Qt.Horizontal)
        # self.time_slider.setMinimum(0)
        # self.time_slider.setMaximum(self.max_time_idx-1)
        self.time_slider = AdvancedTimeline(total_frames=self.max_time_idx,fps=self.data.fps,time_point=self.data.time_point)
        self.time_slider.rightClicked.connect(self.on_timeline_right_click)
        self.time_label = QLabel(f"{self.current_time_idx}/{self.max_time_idx-1}")
        slider_layout.addWidget(self.start_button)
        slider_layout.addWidget(self.pause_button)
        slider_layout.addWidget(self.reset_button)
        slider_layout.addWidget(self.time_slider)
        slider_layout.addWidget(self.time_label)
        if self.data.is_temporary :
            layout.addLayout(slider_layout)
            self.time_slider.positionChanged.connect(self.update_time_slice)
        widget.setLayout(layout)
        self.setWidget(widget)

        self.add_overlay_label("请移动鼠标")

    # 添加覆盖标签的方法
    def add_overlay_label(self, text):
        """添加覆盖文本标签"""
        # 如果已有标签，先删除
        if hasattr(self, 'overlay_label') and self.overlay_label:
            self.remove_overlay_label()

        # 创建新标签
        self.overlay_label = QLabel(text, self.graphics_view)
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: rgba(100, 100, 100, 180);
                font: Noto Sans;
                font-size: 60px;
                font-weight: bold;
            }
        """)
        self.overlay_label.setGeometry(10, 10, 350, 300)
        self.overlay_label.setAttribute(Qt.WA_TransparentForMouseEvents)  # 允许鼠标穿透
        self.overlay_label.show()

    def remove_overlay_label(self):
        """移除覆盖标签"""
        if hasattr(self, 'overlay_label') and self.overlay_label:
            self.overlay_label.deleteLater()
            self.overlay_label = None

    def set_drawing_tool(self, tool):
        """设置当前绘图工具"""
        self.drawing_tool = tool
        self.drawing = False
        self.start_pos = None
        self.end_pos = None

        # 清除前序画板
        if tool in ["V-rect",'V-line',"Anchor"]:
            self.clear_draw_layer()
            if tool == "Anchor":
                self.clear_anchor()
            elif tool == "V-rect":
                self.clear_vector_rect()
            elif tool == "V-line":
                self.clear_vector_line()

    def set_toolset(self,args_dict:dict):
        """初始化和更新参数"""
        self.pen_size = args_dict["pen_size"]
        self.pen_color = args_dict["pen_color"]
        self.fill_color = args_dict["fill_color"]
        self.auto_fill = args_dict.get("auto_fill", False)
        self.vector_color = args_dict["vector_color"]
        self.angle_step = args_dict["angle_step"]
        self.vector_width = args_dict["vector_width"]
        self.colormap = args_dict["colormap"]  # 默认伪彩色方案
        self.use_colormap = args_dict["use_colormap"]  # 是否使用伪彩色
        self.min_value = args_dict["min_value"]  # 伪彩色最小值
        self.max_value = args_dict["max_value"]  # 伪彩色最大值
        self.auto_boundary_set = args_dict["auto_boundary_set"]
        if self.auto_boundary_set:
            self.auto_colormap_range()
        self.update_after_set()

    def update_after_set(self):
        """更新设置后的更新"""
        if self.use_colormap and hasattr(self,'graphics_view'):
            self.update_time_slice(self.current_time_idx)
            self.add_colorbar()
            self.graphics_view.resize(self.width(), self.height())
            # self.graphics_view.resetTransform()
            # self.graphics_view.fitInView(self.data_layer, Qt.KeepAspectRatio)
        elif not self.use_colormap and hasattr(self,'graphics_view'):
            self.update_time_slice(self.current_time_idx)
            # logging.info("若未更新，滑动时间轴即可更新")
            if self.colorbar_item:
                self.scene.removeItem(self.colorbar_item)
                self.colorbar_item = None
                self.scene.removeItem(self.min_label)
                self.scene.removeItem(self.max_label)
            self.graphics_view.resize(self.width(), self.height())
            # self.graphics_view.resetTransform()
            # self.graphics_view.fitInView(self.data_layer, Qt.KeepAspectRatio)

    def auto_colormap_range(self):
        """自动设置伪彩色范围"""
        if self.data is not None:
            self.min_value = self.data.imagemin
            self.max_value = self.data.imagemax

    def set_anchor_mode(self, active):
        """设置 anchor 模式"""
        self.anchor_active = active
        if not active:
            self.clear_anchor()

    def clear_anchor(self):
        """清除十字标"""
        if self.anchor_item:
            for item in self.anchor_item:
                self.scene.removeItem(item)
            self.anchor_item = None
            self.anchor_pos = None
        if self.anchor_mask is not None:
            self.anchor_mask = None

    def clear_vector_line(self):
        """清除当前矢量直线"""
        if hasattr(self, 'vector_line') and self.vector_line:
            self.scene.removeItem(self.vector_line)
            self.vector_line = None
            self.line_item = None

    def clear_vector_rect(self):
        """清除矢量矩形"""
        if self.v_rect_roi:
            self.scene.removeItem(self.rect_item)
            self.rect_item = None
        if self.movable_rect_item:
            self.scene.removeItem(self.movable_rect_item)
            self.movable_rect_item = None
        self.v_rect_roi = None

    def clear_draw_layer(self):
        """清除绘制层"""
        if hasattr(self, 'top_pixmap') and self.top_pixmap is not None:
            self.top_pixmap.fill(Qt.transparent)
            self.draw_layer.setPixmap(self.top_pixmap)
            self.temp_pixmap = None
            self.draw_roi = np.zeros(self.data.framesize, dtype=np.uint8)

    def wheel_event(self, event: QWheelEvent):
        """滚轮缩放实现"""
        if not self.map_view:
            self.display_image()
            self.remove_overlay_label()
        if not hasattr(self, 'current_image'):
            return

        zoom_step  = 1.25
        if event.angleDelta().y() > 0:  # 放大
            new_scale = min(self.last_scale * zoom_step, self.max_scale)
        else:  # 缩小
            new_scale = max(self.last_scale / zoom_step, self.min_scale)

        if new_scale == self.last_scale:
            return

        # 获取鼠标在视图和场景中的位置
        mouse_view_pos = event.pos()
        old_scene_pos = self.graphics_view.mapToScene(mouse_view_pos)

        # 应用新缩放因子
        self.graphics_view.setTransform(QTransform.fromScale(new_scale, new_scale))
        self.last_scale = new_scale  # 必须更新缩放因子记录

        # 仅在放大时调整视口位置
        if new_scale > self.initial_scale:
            # 计算缩放后的鼠标位置差
            new_view_pos = self.graphics_view.mapFromScene(old_scene_pos)
            delta = new_view_pos - mouse_view_pos
            # 通过滚动条补偿位置变化
            self.graphics_view.horizontalScrollBar().setValue(
                self.graphics_view.horizontalScrollBar().value() + delta.x()
            )
            self.graphics_view.verticalScrollBar().setValue(
                self.graphics_view.verticalScrollBar().value() + delta.y()
            )

    def mouse_press_event(self, event):
        """鼠标点击事件处理"""
        if not hasattr(self, 'current_image'):
            return
        if self.drawing_tool == 'V-rect' and event.button() == Qt.LeftButton:
            item = self.graphics_view.itemAt(event.pos())
            # 如果点击的是我们的自定义ROI对象，则交由对象自己处理（拖动），不开启新绘图
            if isinstance(item, ResizableRectROI):
                QGraphicsView.mousePressEvent(self.graphics_view, event)
                return

        if event.button() == Qt.MidButton :
            # 中键按下：准备拖动
            self.drag_start_pos = event.pos()
            self.graphics_view.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            # 左键点击：发射坐标信号
            if self.drawing_tool in ['V-rect','V-line', 'Rect', 'Ellipse', 'Line', 'Pen', 'Eraser','Fill']:
                self.drawing = True
                self.start_pos = self.graphics_view.mapToScene(event.pos())
                self.end_pos = self.start_pos

                if self.drawing_tool == 'V-rect':
                    self.clear_draw_layer() # 重置图层
                    self.clear_vector_line()
                    self.clear_vector_rect()
                    rect = QRectF(self.start_pos, self.end_pos)
                    self.rect_item = QGraphicsRectItem(rect)
                    self.rect_item.setPen(QPen(QColor(self.vector_color), 0.2, Qt.SolidLine,Qt.SquareCap ,Qt.MiterJoin))
                    self.scene.addItem(self.rect_item)

                elif self.drawing_tool == 'V-line':
                    self.clear_draw_layer() # 重置图层
                    self.clear_vector_line()  # 一次只能保留一条直线
                    self.clear_vector_rect()
                    self.line_item = QLineF(self.start_pos, self.end_pos)
                    self.vector_line = VectorLineROI(self.line_item)
                    self.vector_line.setPen(QPen(QColor(self.vector_color), 0.2, Qt.SolidLine,Qt.SquareCap ,Qt.MiterJoin))
                    self.vector_line.setWidth(self.vector_width)
                    self.scene.addItem(self.vector_line)
                    pass

                elif self.drawing_tool == 'Fill':
                    # 填充工具
                    pos = self.graphics_view.mapToScene(event.pos())
                    self.fill_at_point(pos.toPoint())
                    self.drawing = False
                    return

            elif self.drawing_tool == 'Anchor' and self.anchor_active:
                # 光标模式
                pos = self.graphics_view.mapToScene(event.pos())
                x_int , y_int = int(pos.x()), int(pos.y())
                x, y = x_int+0.5, y_int+0.5
                h, w = self.current_image.shape[0],self.current_image.shape[1]

                if 0 <= x < w and 0 <= y < h:
                    self.current_canvas_signal.emit(self.id)  # 改为只有当绘图操作时才更新cursor
                    # 清除现有十字标
                    self.clear_anchor()
                    if self.args_dict['anchor_select']:
                        v_color = Qt.red
                    else:
                        v_color = QColor(self.vector_color)

                    # 创建新的十字标
                    pen = QPen(v_color, 0.1, Qt.SolidLine)
                    # 水平线
                    h_line = QGraphicsLineItem(x-1, y,x+1, y)
                    h_line.setPen(pen)
                    h_line.setZValue(60)
                    # 垂直线
                    v_line = QGraphicsLineItem(x, y-1, x, y + 1)
                    v_line.setPen(pen)
                    v_line.setZValue(60)
                    circle = QGraphicsEllipseItem(x-0.4, y-0.4, 0.8, 0.8)
                    circle.setPen(pen)
                    circle.setZValue(60)

                    # 添加到场景
                    self.scene.addItem(h_line)
                    self.scene.addItem(v_line)
                    self.scene.addItem(circle)

                    # 存储十字标位置
                    self.anchor_pos = (x_int, y_int)
                    self.anchor_item = [h_line, v_line,circle]

                    # 获取并发射图像数据
                    self.get_value(y_int, x_int)

                    if self.anchor_active and self.data.is_temporary and self.args_dict['anchor_select']:
                        # anchor模式下取值快速绘图
                        self.anchor_mask = PublicEasyMethod.quick_mask(self.data.framesize,
                                                                                shape= self.args_dict['anchor_shape'],
                                                                                size=self.args_dict['anchor_size'],
                                                                                center = (y_int,x_int))
                        self.add_fast_selection(x_int,y_int,self.anchor_mask)
                        method = self.args_dict['anchor_method']
                        self.get_fast_selection.emit(self.data,self.anchor_mask, method , f'canvas{self.id}-({x_int},{y_int}){method}')
                        logging.info(f'取{x, y}的{self.args_dict['anchor_method']}绘图')
                    return

            else: # 无工具选中的纯单机模式
                pos = self.graphics_view.mapToScene(event.pos())
                x, y = int(pos.x()), int(pos.y())
                h, w = self.current_image.shape[0],self.current_image.shape[1]
                if 0 <= x < w and 0 <= y < h:
                    self.mouse_clicked_signal.emit(x, y,self.id)
                    self.current_canvas_signal.emit(self.id)  # 改为只有当绘图操作时才更新cursor

        QGraphicsView.mousePressEvent(self.graphics_view, event)

    def mouse_move_event(self, event):
        """鼠标移动事件处理"""
        if not self.map_view:
            self.display_image()
            self.remove_overlay_label()
        if not hasattr(self, 'current_image'):
            return

        if self.drag_start_pos is not None:
            delta = event.pos() - self.drag_start_pos
            self.graphics_view.horizontalScrollBar().setValue(
                self.graphics_view.horizontalScrollBar().value() - delta.x())
            self.graphics_view.verticalScrollBar().setValue(
                self.graphics_view.verticalScrollBar().value() - delta.y())
            self.drag_start_pos = event.pos()

        # 当 anchor 激活时，禁用动态获取鼠标位置功能

        # 坐标获取
        move_pos = self.graphics_view.mapToScene(event.pos())
        self.x_img, self.y_img = int(move_pos.x()), int(move_pos.y())
        h_img, w_img = self.current_image.shape[0], self.current_image.shape[1]

        # 鼠标值显示逻辑
        if 0 <= self.x_img < w_img and 0 <= self.y_img < h_img:
            self.mouse_pos = (self.x_img, self.y_img)
            if not self.anchor_active:
                self.get_value(self.y_img, self.x_img)

        # 绘图模式
        if self.drawing and self.drawing_tool != 'Anchor':
            # 1. 获取原始坐标
            raw_end_pos = self.graphics_view.mapToScene(event.pos())

            # 2. 获取图像边界
            h_img, w_img = self.current_image.shape[:2]

            # 3. 无论是否按 Shift，都先将坐标限制在 [0, w] 和 [0, h] 之间
            clamped_x = max(0, min(raw_end_pos.x(), w_img))
            clamped_y = max(0, min(raw_end_pos.y(), h_img))
            self.end_pos = QPointF(clamped_x, clamped_y)

            shift_pressed = QApplication.keyboardModifiers() == Qt.ShiftModifier
            if self.drawing_tool == 'V-rect':
                rect = QRectF(self.start_pos, self.end_pos).normalized()
                if shift_pressed: # 限制正方形
                    # 计算当前宽高
                    w = abs(self.end_pos.x() - self.start_pos.x())
                    h = abs(self.end_pos.y() - self.start_pos.y())
                    size = min(w, h)

                    # 判断象限方向
                    dx = 1 if self.end_pos.x() >= self.start_pos.x() else -1
                    dy = 1 if self.end_pos.y() >= self.start_pos.y() else -1

                    # 重新生成正方形 End Point (保持起点不变)
                    new_end_x = self.start_pos.x() + size * dx
                    new_end_y = self.start_pos.y() + size * dy

                    # 再次限制边界（防止 Shift 扩展出边界）
                    new_end_x = max(0, min(new_end_x, w_img))
                    new_end_y = max(0, min(new_end_y, h_img))

                    rect = QRectF(self.start_pos, QPointF(new_end_x, new_end_y)).normalized()

                self.rect_item.setRect(rect)

            elif self.drawing_tool == 'V-line':
                if self.vector_line:
                    self.line_item.setP2(self.end_pos)
                    self.vector_line.setLine(self.line_item)
                    self.vector_line.updateWidthPath()
                return

            elif self.drawing_tool in ['Pen', 'Eraser', 'Line', 'Rect','Ellipse']:
                temp_pixmap = QPixmap(self.top_pixmap)
                self.temp_pixmap = self._draw_on_pixmap(temp_pixmap, self.start_pos.toPoint(), move_pos.toPoint())
                self.draw_layer.setPixmap(temp_pixmap)

                if self.drawing_tool in ["Pen", "Eraser"]:
                    self.temp_pixmap = temp_pixmap
                    self.top_pixmap = temp_pixmap
                    self.start_pos = move_pos

        QGraphicsView.mouseMoveEvent(self.graphics_view, event)

    def mouse_release_event(self, event):
        """鼠标释放事件（结束拖动）"""
        if event.button() == Qt.MidButton:
            self.drag_start_pos = None
            self.graphics_view.setCursor(Qt.ArrowCursor)

        elif event.button() == Qt.LeftButton and self.drawing:
            # 完成绘图
            self.drawing = False
            if self.drawing_tool == 'V-rect':
                # 使用之前的 clamp 逻辑获取最终坐标
                raw_end_pos = self.graphics_view.mapToScene(event.pos())
                img_h, img_w = self.current_image.shape[:2]

                x2 = max(0, min(int(raw_end_pos.x()), img_w))
                y2 = max(0, min(int(raw_end_pos.y()), img_h))

                if self.rect_item:
                    # 获取标准化后的矩形 (已经包含了 Shift 逻辑的结果)
                    final_rect = self.rect_item.rect()
                    self.scene.removeItem(self.rect_item)
                    self.rect_item = None

                    x, y, w, h = final_rect.x(), final_rect.y(), final_rect.width(), final_rect.height()

                    # 再次取整确保像素对齐
                    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))

                    if w > 0 and h > 0:
                        if self.movable_rect_item:
                            self.scene.removeItem(self.movable_rect_item)

                        # --- 实例化新的 ResizableRectROI ---
                        self.movable_rect_item = ResizableRectROI(x, y, w, h, (img_w, img_h), callback=self.on_roi_moved)
                        self.movable_rect_item.setZValue(100)
                        self.scene.addItem(self.movable_rect_item)
                        self.movable_rect_item.setSelected(True)

                        # 发送信号
                        self.on_roi_moved(((x, y), w, h))
                return

            if self.drawing_tool in ['V-line', 'Anchor']:
                pass
            else:
                self.top_pixmap = self.temp_pixmap
                self.draw_layer.setPixmap(self.top_pixmap)
                self.update_draw_layer_array() # 仅在绘制像素时储存
            self.current_canvas_signal.emit(self.id) # 改为只有当绘图操作时才更新cursor

        QGraphicsView.mouseReleaseEvent(self.graphics_view, event)

    def _draw_on_pixmap(self, pixmap, from_point, to_point):
        """在绘图层上绘制（用于Pen和Eraser）"""
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(self.pen_color), self.pen_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        # 检测 Shift 按键
        shift_pressed = QApplication.keyboardModifiers() == Qt.ShiftModifier

        if self.drawing_tool == 'Pen':
            painter.drawLine(from_point, to_point)
        elif self.drawing_tool == 'Eraser':
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setPen(QPen(Qt.transparent, self.pen_size))
            painter.drawLine(from_point, to_point)

        elif self.drawing_tool == "Line":
            if shift_pressed:
                # --- 角度约束逻辑（45°倍数）---
                dx = to_point.x() - from_point.x()
                dy = to_point.y() - from_point.y()
                length = (dx ** 2 + dy ** 2) ** 0.5  # 直线长度

                if length > 0:
                    angle = atan2(dy, dx)  # 原始角度（弧度）
                    constrained_angle = round(angle / self.angle_step) * self.angle_step  # 锁定到最近的45°倍数

                    # 修正终点坐标
                    to_point = QPointF(
                        from_point.x() + length * cos(constrained_angle),
                        from_point.y() + length * sin(constrained_angle)
                    ).toPoint()
            painter.drawLine(from_point, to_point)

        elif self.drawing_tool in ["Rect", "Ellipse"]:
            # 设置填充模式
            if self.auto_fill:
                painter.setBrush(QBrush(QColor(self.fill_color)))
            else:
                painter.setBrush(Qt.NoBrush)

            rect = QRectF(from_point, to_point).normalized()

            # 统一处理 Shift 约束（正方形/正圆）
            if shift_pressed:
                size = min(rect.width(), rect.height())
                # 根据绘制方向调整正方形/圆的位置
                if to_point.x() >= from_point.x() and to_point.y() >= from_point.y():
                    rect = QRectF(from_point, from_point + QPointF(size, size))
                elif to_point.x() < from_point.x() and to_point.y() < from_point.y():
                    rect = QRectF(from_point, from_point - QPointF(size, size))
                elif to_point.x() >= from_point.x() and to_point.y() < from_point.y():
                    # 右上方向
                    rect = QRectF(from_point.x(), from_point.y() - size, size, size)
                else:
                    # 左下方向
                    rect = QRectF(from_point.x() - size, from_point.y(), size, size)

            if self.drawing_tool == "Rect":
                painter.drawRect(rect)
            else:
                painter.drawEllipse(rect)

        painter.end()
        return pixmap

    def on_roi_moved(self, roi_data):
        """当拖动矩形结束时调用此函数"""
        self.v_rect_roi = roi_data

    def fill_at_point(self, point):
        """在指定点进行填充"""
        # 创建临时图像
        image = self.top_pixmap.toImage()

        # 获取点击位置的颜色
        target_color = image.pixelColor(point)

        # 如果颜色已经是填充色，则不操作
        if target_color == QColor(self.fill_color):
            return

        # 执行填充算法
        self.flood_fill(image, point.x(), point.y(), target_color, self.fill_color)

        # 更新pixmap
        self.top_pixmap = QPixmap.fromImage(image)
        self.draw_layer.setPixmap(self.top_pixmap)
        # 将pixmap储存
        self.update_draw_layer_array()

    @staticmethod
    def flood_fill(image, x, y, target_color, fill_color):
        """洪水填充算法实现"""
        # 使用非递归方式实现，避免堆栈溢出
        pixels = [(x, y)]
        width = image.width()
        height = image.height()

        while pixels:
            x, y = pixels.pop()

            # 边界检查
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            current_color = image.pixelColor(x, y)

            # 检查是否需要填充
            if current_color != target_color:
                continue

            # 设置填充颜色
            image.setPixelColor(x, y, QColor(fill_color))

            # 添加相邻像素
            pixels.append((x + 1, y))
            pixels.append((x - 1, y))
            pixels.append((x, y + 1))
            pixels.append((x, y - 1))

    def closeEvent(self, event):
        """重写关闭事件"""
        self.parent_window.del_canvas(self.id)
        super().closeEvent(event)

    """下面是播放和图像更新的设置"""
    def start_auto_play(self):
        if self.max_time_idx <= 1:
            return False # 没有足够的帧进行播放

        if not self.is_playing:
            if self.play_paused_time == 0:            # 如果是第一次播放或重置后播放
                self.play_start_time = QDateTime.currentMSecsSinceEpoch()
            else:                # 如果是暂停后继续播放，调整开始时间
                self.play_start_time = QDateTime.currentMSecsSinceEpoch() - self.play_paused_time
            self.is_playing = True
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.reset_button.setEnabled(True)

            # 计算帧间隔时间（毫秒）
            total_time = 15000  # 15秒
            frame_interval = max(1, 1000 // self.data.fps) if self.data.fps is not None else max(1, total_time // self.max_time_idx)
            self.play_timer.start(frame_interval)

    def pause_auto_play(self):
        """暂停自动播放"""
        if self.is_playing:
            self.is_playing = False
            self.play_paused_time = QDateTime.currentMSecsSinceEpoch() - self.play_start_time
            self.play_timer.stop()
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)

    def reset_auto_play(self):
        """重置自动播放"""
        self.play_timer.stop()
        self.is_playing = False
        self.play_start_time = 0
        self.play_paused_time = 0
        self.current_time_idx = 0
        self.time_slider.set_current_frame(0)
        self.time_label.setText(f"{self.current_time_idx}/{self.max_time_idx - 1}")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.reset_button.setEnabled(False)

    def auto_play_update(self):
        """定时器回调，更新帧显示"""
        if not self.is_playing:
            return

        # 计算当前应该显示的帧索引
        elapsed = QDateTime.currentMSecsSinceEpoch() - self.play_start_time
        position_in_cycle = elapsed % 15000

        # 根据周期位置计算帧索引
        target_idx = int(self.max_time_idx * position_in_cycle / 15000)
        target_idx = min(self.max_time_idx - 1, target_idx)

        # 只有当帧索引变化时才更新显示
        if target_idx != self.current_time_idx:
            self.current_time_idx = target_idx
            self.time_slider.set_current_frame(target_idx)
            self.time_label.setText(f"{self.current_time_idx}/{self.max_time_idx - 1}")

    def update_time_slice(self,idx=0):
        if not 0 <= idx <= self.max_time_idx:
            raise ValueError('idx out of range(impossible Fault)')
        self.current_time_idx = idx
        self.time_label.setText(f"{self.current_time_idx}/{self.max_time_idx - 1}")
        if self.use_colormap:  # 是否伪彩
            if self.data.colormode == self.colormap:  # 是否模式匹配
                image_data = self.data.image_data[idx] if self.data.is_temporary else self.data.image_data  # 是否时间分辨
            else:
                image_data = self.data.image_backup[idx] if self.data.is_temporary else self.data.to_uint8()
        else:
            image_data = self.data.image_data[idx] if self.data.is_temporary else self.data.image_data
        self.update_display(image_data)

    def on_timeline_right_click(self, frame, in_selection, global_pos):
        """处理时间轴右键点击"""
        menu = QMenu(self)

        # 1. 创建菜单项
        if in_selection == 'handle': # 拖动句柄
            # 计算选区长度
            sel_len = self.time_slider.selection_end - self.time_slider.selection_start

            crop_action = QAction(QIcon.fromTheme("edit-cut"), f"裁剪图像(至{sel_len} 帧)", self)
            crop_action.triggered.connect(self.crop_image)
            menu.addAction(crop_action)

            # 你可以在这里加更多功能，比如 "导出选区"
            strong_crop_action = QAction(f"强劲裁剪数据(至{sel_len} 帧)", self)
            strong_crop_action.triggered.connect(self.crop_data)
            menu.addAction(strong_crop_action)
        else:
            # 视频数据参数编辑
            reset_params = QAction("重设参数", self)
            reset_params.triggered.connect(self.reset_params)
            menu.addAction(reset_params)

        # 2. 弹出菜单
        menu.exec_(global_pos)

    def crop_image(self):
        """执行裁剪逻辑"""
        # 1. 获取当前选区
        start = self.time_slider.selection_start
        end = self.time_slider.selection_end
        length = end - start

        if length <= 0:
            QMessageBox.warning(self, "无法裁剪", "选区长度无效。")
            return

        # 2. 弹出确认对话框 (重要！防止误操作)
        reply = QMessageBox.question(
            self,
            "确认裁剪",
            f"确定要裁剪图像数据吗？\n保留范围: {start} - {end}\n\n注意：此操作将丢弃设定区间外的所有数据！\n只对图像有效，不修改源数据，关闭后需要重新裁剪！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # 3. 调用数据类的裁剪方法
                # 注意：end 是闭区间索引，切片通常是左闭右开，所以传 end+1 或者根据你的业务逻辑调整
                # Timeline 闭区间 [start, end]，所以切片应该是 [start : end + 1]
                self.data.trim_time(start, end + 1)

                # 4. 更新 UI 状态
                self.refresh_trimmed_image()

                QMessageBox.information(self, "成功", "数据裁剪完成。")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"裁剪失败: {str(e)}")

    def crop_data(self):
        """裁剪母数据"""
        # 1. 获取当前选区
        start = self.time_slider.selection_start
        end = self.time_slider.selection_end
        length = end - start

        if length <= 0:
            QMessageBox.warning(self, "无法裁剪", "选区长度无效。")
            return

        # 2. 弹出确认对话框 (重要！防止误操作)
        reply = QMessageBox.question(
            self,
            "确认强劲裁剪",
            f"确定要裁剪源数据吗？\n保留范围: {start} - {end}\n\n注意：此操作将同时修改图像和源数据！\n 源数据会被裁剪！！！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            reply_confirm = QMessageBox.question(
                self,
                '再次确认',
                "源数据会被裁剪！无法撤销！\n（原始的文件不会被修改，可以重新导入）",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply_confirm == QMessageBox.Yes:
                try:
                    # 3. 调用数据类的裁剪方法
                    # 注意：end 是闭区间索引，切片通常是左闭右开，所以传 end+1 或者根据你的业务逻辑调整
                    # Timeline 闭区间 [start, end]，所以切片应该是 [start : end + 1]
                    self.data.trim_time(start, end + 1)
                    # 4. 更新 UI 状态
                    self.refresh_trimmed_image()
                    # 5. 裁剪父数据
                    parent_data = self.data.parent_data()
                    if parent_data is not None:
                        parent_data.trim_time(start, end + 1)
                    else:
                        logging.error("图像数据已裁剪，但未找到源数据")
                        raise KeyError('其源数据已不存在于当前内存中！')

                    QMessageBox.information(self, "成功", "数据裁剪完成。")

                except Exception as e:
                    QMessageBox.critical(self, "错误", f"裁剪失败: {str(e)}")

    def reset_params(self):
        """重置核心参数"""
        data = self.data.parent_data()
        self.dialog = ParamsResetDialog(data, image_data=self.data)
        self.dialog.updata_param_signal.connect(self.parent_window.parent.update_param) # 这个适合父类调用吧
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.accepted.connect(self.reset_from_params)

    def refresh_trimmed_image(self):
        """数据改变后刷新整个界面"""
        # 1. 更新最大帧数记录
        self.max_time_idx = self.data.totalframes
        self.current_time_idx = 0  # 重置到开头

        # 2. 更新时间轴控件
        self.time_slider.update_data_range(self.max_time_idx-1)

        # 3. 更新标签
        self.time_label.setText(f"0/{self.max_time_idx - 1}")

        # 4. 刷新图像显示
        self.update_time_slice(0)

    def reset_from_params(self):
        """参数更新后对全局的修改"""
        self.current_time_idx = 0
        self.time_slider.set_fps(self.data.fps)
        self.update_time_slice(0)

    def display_image(self):
        """显示图像数据 (使用QPixmap)并记录当前时间索引
        ROI_applied 暂时放弃"""
        try:
            if self.use_colormap: # 是否伪彩
                if self.data.colormode == self.colormap: # 是否模式匹配
                    image_data = self.data.image_data[0] if self.data.is_temporary else self.data.image_data # 是否时间分辨
                else:
                    image_data = self.data.image_backup[0] if self.data.is_temporary else self.data.image_backup
            else:
                image_data = self.data.image_data[0] if self.data.is_temporary else self.data.image_data
        except Exception as e:
            raise  ValueError(f'nodata(impossible Fault):{e}')

        self.graphics_view.resize(self.width(), self.height())
        self.scene.clear()
        self.current_time_idx = 0
        # 创建QImage并转换为QPixmap
        if self.use_colormap and self.data.colormode != self.colormap:
            # 应用伪彩色映射（预览处理）
            image_data = self.color_map_manager.apply_colormap(
                image_data,
                self.colormap,
                self.min_value,
                self.max_value
            )

            # 创建QImage
            height, width, _ = image_data.shape
            qimage = QImage(image_data.data, width, height,
                            image_data.strides[0], QImage.Format_RGBA8888)
        elif self.use_colormap and self.data.colormode == self.colormap:
            height, width, _ = image_data.shape
            qimage = QImage(image_data.data, width, height,
                            image_data.strides[0], QImage.Format_RGBA8888)
        else:
            height, width = image_data.shape
            qimage = QImage(image_data, image_data.shape[1], image_data.shape[0],
                            image_data.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        # 显示图像
        self.current_image = image_data
        self.data_layer = self.scene.addPixmap(pixmap)
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.data_layer, Qt.KeepAspectRatio)
        self.last_scale = self.graphics_view.transform().m11()
        self.initial_scale = self.graphics_view.transform().m11()
        self.map_view = True
        # 绘制层
        self.top_pixmap = QPixmap(width, height)
        self.top_pixmap.fill(Qt.transparent)
        self.draw_layer = self.scene.addPixmap(self.top_pixmap)
        self.draw_layer.setOpacity(self.draw_layer_opacity)
        self.draw_layer.setZValue(1)  # 确保绘图层在数据层之上

        self.add_colorbar()

    def update_display(self, image_data):
        """仅更新图像数据，不改变视图状态"""
        self.current_image = image_data

        if self.anchor_pos:
            x, y = self.anchor_pos
            # 获取并发射图像数据
            self.get_value(y,x)
        if self.use_colormap and self.data.colormode != self.colormap:
            # 应用伪彩色映射（预览处理）
            image_data = self.color_map_manager.apply_colormap(
                image_data,
                self.colormap,
                self.min_value,
                self.max_value
            )

            # 创建QImage
            height, width, _ = image_data.shape
            qimage = QImage(image_data.data, width, height,
                            image_data.strides[0], QImage.Format_RGBA8888)
        elif self.use_colormap and self.data.colormode == self.colormap:
            height, width, _ = image_data.shape
            qimage = QImage(image_data.data, width, height,
                            image_data.strides[0], QImage.Format_RGBA8888)
        else:
            qimage = QImage(image_data, image_data.shape[1], image_data.shape[0],
                            image_data.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        # 直接更新现有pixmap，避免重置场景，其它层保持不变
        self.data_layer.setPixmap(pixmap)

    def add_colorbar(self):
        """添加颜色条到场景"""
        # 移除现有的颜色条
        if self.colorbar_item:
            self.scene.removeItem(self.colorbar_item)
            self.colorbar_item = None
            try:
                self.scene.removeItem(self.min_label)
                self.scene.removeItem(self.max_label)
            finally:
                pass

        if not self.use_colormap:
            return

        # 创建颜色条
        height = self.data_layer.pixmap().height() if self.data_layer.pixmap() else 256
        width = self.colorbar_width
        h_point = 0
        w_point = self.data_layer.pixmap().width() + self.colorbar_padding
        # size = max(height // 50 +1 ,1)
        size = 1

        # 创建渐变
        gradient = QLinearGradient(0, 0, 0, height)

        # 获取colormap
        cmap = self.color_map_manager.matplotlib_cmaps.get(self.colormap, cm.jet)

        # 添加颜色停止点
        for i in range(256):
            pos = i / 255.0
            r, g, b, _ = cmap(pos)
            gradient.setColorAt(1.0 - pos, QColor(int(r * 255), int(g * 255), int(b * 255)))

        # 创建颜色条图形项
        self.colorbar_item = self.scene.addRect(
            w_point,
            h_point,
            width,
            height,
            brush=QBrush(gradient),
            pen=QPen(Qt.NoPen)
        )

        # 添加文本标签
        if self.min_value is not None and self.max_value is not None:

            font = QFont("Source Han Serif")
            font.setPointSize(size)
            font.setLetterSpacing(QFont.AbsoluteSpacing, -0.2)
            # 最小值标签
            self.min_label = self.scene.addText(f"min={self.min_value:.1f}",font=font)
            self.min_label.setPos(w_point + width-8, h_point+ height-6+size)
            self.min_label.setDefaultTextColor(Qt.black)
            # 最大值标签
            self.max_label = self.scene.addText(f"max={self.max_value:.1f}",font=font)
            self.max_label.setPos(w_point + width-8, h_point -5-size)
            self.max_label.setDefaultTextColor(Qt.black)

    def add_fast_selection(self, y: int, x: int, mask: np.ndarray, color=None, auto_clear=True):
        """
        以独立图元形式显示布尔蒙版区域和中心点
        :param x: 蒙版中心的 x 坐标
        :param y: 蒙版中心的 y 坐标
        :param mask: 二维布尔 ndarray
        :param color: (可选) 指定颜色 QColor，默认为 vector_color
        :param auto_clear: 是否自动清除上一次调用的蒙版显示（默认为 True）
        """
        # 1. 初始化存储列表（如果没有的话）
        if not hasattr(self, 'mask_overlay_items'):
            self.mask_overlay_items = []

        # 2. 如果需要，清除旧的蒙版显示
        if auto_clear:
            self.clear_fast_selection()

        # 3. 准备颜色
        c = color if color else QColor(self.vector_color)

        # 4. 创建 BGRA 缓冲区
        h, w = mask.shape
        # 初始化全透明 buffer
        buffer = np.zeros((h, w, 4), dtype=np.uint8)

        # 4.1 填充蒙版区域 (半透明)
        # 这里的 mask 尺寸即图像尺寸，直接映射
        # Qt ARGB32 (Little Endian) -> B G R A
        buffer[mask] = [c.blue(), c.green(), c.red(), 120]

        # 4.2 标记中心点像素 (红色，不透明)，如果是anchor模式就不标记了
        # 确保坐标在图像范围内
        if 0 <= x < w and 0 <= y < h and not self.anchor_active:
            buffer[y, x] = [0, 0, 255, 150]  # Red=255, Alpha=255 (BGRA顺序: B=0, G=0, R=255, A=255)

        # 5. 生成 QImage 和 QPixmap
        # copy() 是必须的，防止 buffer 被垃圾回收导致图像花屏
        qimg = QImage(buffer.data, w, h, w * 4, QImage.Format_ARGB32).copy()
        pixmap = QPixmap.fromImage(qimg)

        # 6. 创建图元并添加到场景
        mask_item = QGraphicsPixmapItem(pixmap)

        # 关键修正：因为 mask 也就是整张图的大小，所以位置直接设为 (0,0)
        mask_item.setPos(0, 0)

        # 设置层级 (ZValue)，保证覆盖在数据层之上
        mask_item.setZValue(50)

        self.scene.addItem(mask_item)
        self.mask_overlay_items.append(mask_item)

    def clear_fast_selection(self):
        """清所有临时显示的蒙版图元"""
        if hasattr(self, 'mask_overlay_items') and self.mask_overlay_items:
            for item in self.mask_overlay_items:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            self.mask_overlay_items.clear()

    def get_value(self,y,x):
        value = self.current_image[y,x]
        if self.data.is_temporary:
            original_value = self.data.image_backup[self.current_time_idx][y, x]
        else:
            original_value = self.data.image_backup[y, x]
        # 发射信号(需要主窗口连接此信号)
        self.mouse_position_signal.emit(x, y, self.current_time_idx, value, original_value)

    def update_draw_layer_array(self):
        """将顶部图层QPixmap转换为二维数组"""
        image = self.top_pixmap.toImage()
        height, width = self.draw_roi.shape

        for y in range(height):
            for x in range(width):
                if x < image.width() and y < image.height():
                    color = image.pixelColor(x, y)
                    # 按照透明度设置蒙版
                    if color.alpha() == 0:  # 完全透明
                        self.draw_roi[y, x] = 0
                    else:
                        self.draw_roi[y, x] = color.alpha() / 255

    def reset_view(self):
        """手动重置视图（缩放和平移）"""
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.last_scale = self.graphics_view.transform().m11()


class Handle(Enum):
    NONE = 0
    TOP_LEFT = 1
    TOP = 2
    TOP_RIGHT = 3
    RIGHT = 4
    BOTTOM_RIGHT = 5
    BOTTOM = 6
    BOTTOM_LEFT = 7
    LEFT = 8
    CENTER = 9  # 移动


class ResizableRectROI(QGraphicsRectItem):
    """
    仿PS风格的可调整矩形ROI
    - 支持8个方向调整大小
    - 支持像素级吸附
    - 限制在图像边界内
    """

    def __init__(self, x, y, w, h, limit_size, callback=None):
        # 初始化：始终保持 rect 为 (0, 0, w, h)，通过 setPos 移动
        super().__init__(0, 0, w, h)
        self.setPos(x, y)
        self.limit_w, self.limit_h = limit_size
        self.callback = callback

        # 标志位
        self.setFlags(QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        # 样式参数
        self.handle_size = 8  # 句柄大小
        self.handle_space = 0  # 句柄与线的间距

        # 画笔设置 (Cosmetic=True 保证线条宽度不随缩放改变)
        self.pen_outline = QPen(Qt.white, 1, Qt.DashLine)
        self.pen_outline.setCosmetic(True)
        self.pen_shadow = QPen(Qt.black, 1, Qt.SolidLine)  # 黑底白虚线，高对比度
        self.pen_shadow.setCosmetic(True)

        self.handle_brush = QBrush(Qt.white)
        self.handle_pen = QPen(Qt.black, 1)
        self.handle_pen.setCosmetic(True)

        # 交互状态
        self.current_handle = Handle.NONE
        self.mouse_press_pos = None
        self.mouse_press_rect = None
        self.is_resizing = False

    def paint(self, painter, option, widget=None):
        rect = self.rect()

        # 1. 绘制高对比度边框 (黑实线垫底，白虚线在通过)
        fill_color = QColor(255, 255, 0, 50)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(self.pen_shadow)
        painter.drawRect(rect)
        painter.setPen(self.pen_outline)
        painter.drawRect(rect)

        # 2. 如果被选中或悬停，绘制8个控制手柄
        if self.isSelected() or self.isUnderMouse():
            self._draw_handles(painter, rect)

    def _draw_handles(self, painter, rect):
        """绘制8个方块句柄"""
        painter.setPen(self.handle_pen)
        painter.setBrush(self.handle_brush)

        # 计算关键点坐标
        l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
        cx, cy = rect.center().x(), rect.center().y()
        hs = self.handle_size / 2  # 半宽，用于居中

        # 定义8个点的位置
        points = [
            (l, t), (cx, t), (r, t),  # 上三
            (r, cy),  # 右
            (r, b), (cx, b), (l, b),  # 下三
            (l, cy)  # 左
        ]

        # 转换回视图坐标系绘制，或者直接画在场景坐标
        # 这里为了简单，我们利用 Cosmetic 特性，实际上 handle 大小在视图缩放时视觉大小不变会更好
        # 但 QGraphicsItem paint 是在局部坐标系。为了让句柄看起来大小固定，
        # 我们通常需要根据 view 的缩放反向计算，或者简单点直接画。
        # 这里为了保持像素贴合，我们暂且按局部坐标画，可以根据 scale 动态调整大小（高阶做法），
        # 现阶段先画固定像素大小。

        # 获取当前的缩放比例，保持句柄视觉大小一致
        scale = 1.0
        if self.scene() and self.scene().views():
            scale = self.scene().views()[0].transform().m11()

        offset = (self.handle_size / 2) / scale
        draw_size = self.handle_size / scale

        for x, y in points:
            # 以点为中心绘制矩形
            h_rect = QRectF(x - offset, y - offset, draw_size, draw_size)
            painter.drawRect(h_rect)

    def hoverMoveEvent(self, event):
        """悬停时改变鼠标图标"""
        pos = event.pos()
        handle = self._get_handle_at(pos)
        self._set_cursor(handle)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        """记录初始状态"""
        if event.button() == Qt.LeftButton:
            self.current_handle = self._get_handle_at(event.pos())
            self.mouse_press_pos = event.scenePos()  # 记录场景绝对坐标
            self.mouse_press_rect = self.rect()
            self.mouse_press_item_pos = self.pos()
            self.is_resizing = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """处理拖动和缩放"""
        if not self.is_resizing:
            return

        cur_pos = event.scenePos()

        # 1. 移动模式 (点击在中心区域)
        if self.current_handle == Handle.CENTER:
            delta = cur_pos - self.mouse_press_pos
            # 像素吸附：对目标位置取整
            new_x = int(self.mouse_press_item_pos.x() + delta.x())
            new_y = int(self.mouse_press_item_pos.y() + delta.y())

            # 边界检查
            w, h = self.rect().width(), self.rect().height()
            new_x = max(0, min(new_x, self.limit_w - w))
            new_y = max(0, min(new_y, self.limit_h - h))

            self.setPos(new_x, new_y)

        # 2. 调整大小模式 (点击在句柄)
        elif self.current_handle != Handle.NONE:
            self._interactive_resize(cur_pos)

    def mouseReleaseEvent(self, event):
        self.is_resizing = False
        self.current_handle = Handle.NONE
        self.setCursor(Qt.ArrowCursor)

        # 动作结束，发射信号
        self._notify_change()
        super().mouseReleaseEvent(event)

    def _get_handle_at(self, pos):
        """判断鼠标位置对应的句柄"""
        rect = self.rect()
        l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
        cx, cy = rect.center().x(), rect.center().y()

        # 获取视图缩放比例以计算容差
        scale = 1.0
        if self.scene() and self.scene().views():
            scale = self.scene().views()[0].transform().m11()
        tol = (self.handle_size / scale)  # 容差范围

        # 检查顺序：先检查角，再检查边，最后中心
        if self._near(pos.x(), l, tol) and self._near(pos.y(), t, tol): return Handle.TOP_LEFT
        if self._near(pos.x(), r, tol) and self._near(pos.y(), t, tol): return Handle.TOP_RIGHT
        if self._near(pos.x(), r, tol) and self._near(pos.y(), b, tol): return Handle.BOTTOM_RIGHT
        if self._near(pos.x(), l, tol) and self._near(pos.y(), b, tol): return Handle.BOTTOM_LEFT

        if self._near(pos.y(), t, tol) and l < pos.x() < r: return Handle.TOP
        if self._near(pos.y(), b, tol) and l < pos.x() < r: return Handle.BOTTOM
        if self._near(pos.x(), l, tol) and t < pos.y() < b: return Handle.LEFT
        if self._near(pos.x(), r, tol) and t < pos.y() < b: return Handle.RIGHT

        if rect.contains(pos): return Handle.CENTER
        return Handle.NONE

    def _near(self, v1, v2, tol):
        return abs(v1 - v2) <= tol

    def _set_cursor(self, handle):
        """根据句柄设置光标"""
        cursors = {
            Handle.NONE: Qt.ArrowCursor,
            Handle.CENTER: Qt.SizeAllCursor,
            Handle.TOP_LEFT: Qt.SizeFDiagCursor,
            Handle.BOTTOM_RIGHT: Qt.SizeFDiagCursor,
            Handle.TOP_RIGHT: Qt.SizeBDiagCursor,
            Handle.BOTTOM_LEFT: Qt.SizeBDiagCursor,
            Handle.TOP: Qt.SizeVerCursor,
            Handle.BOTTOM: Qt.SizeVerCursor,
            Handle.LEFT: Qt.SizeHorCursor,
            Handle.RIGHT: Qt.SizeHorCursor,
        }
        self.setCursor(cursors.get(handle, Qt.ArrowCursor))

    def _interactive_resize(self, mouse_pos):
        """根据句柄逻辑计算新的 rect 和 pos"""
        # 所有的计算都基于像素吸附 (int)

        # 1. 获取当前 Item 的绝对位置 (item_x, item_y) 和 原始尺寸
        orig_rect = self.mouse_press_rect
        orig_pos = self.mouse_press_item_pos

        # 目标边界（场景坐标）
        target_l = orig_pos.x() + orig_rect.left()
        target_r = orig_pos.x() + orig_rect.right()
        target_t = orig_pos.y() + orig_rect.top()
        target_b = orig_pos.y() + orig_rect.bottom()

        # 鼠标当前场景坐标（吸附）
        mx, my = int(mouse_pos.x()), int(mouse_pos.y())

        # 2. 根据句柄更新边界
        h = self.current_handle
        shift_pressed = QApplication.keyboardModifiers() == Qt.ShiftModifier
        is_corner = h in [Handle.TOP_LEFT, Handle.TOP_RIGHT, Handle.BOTTOM_LEFT, Handle.BOTTOM_RIGHT]

        if shift_pressed and is_corner:
            # 1. 确定锚点（Anchor）：即拖动角点的"对角点"，它是固定的
            pivot_x = target_r if h in [Handle.TOP_LEFT, Handle.BOTTOM_LEFT] else target_l
            pivot_y = target_b if h in [Handle.TOP_LEFT, Handle.TOP_RIGHT] else target_t

            # 2. 计算鼠标相对于锚点的向量
            dx = mx - pivot_x
            dy = my - pivot_y

            # 3. 取长边作为正方形边长
            max_len = max(abs(dx), abs(dy))

            # 4. 根据鼠标相对于锚点的方向，重新计算修正后的鼠标位置
            # sign 用来保持拖动翻转时的方向正确性
            sign_x = 1 if dx >= 0 else -1
            sign_y = 1 if dy >= 0 else -1

            mx = int(pivot_x + max_len * sign_x)
            my = int(pivot_y + max_len * sign_y)

        if h in [Handle.LEFT, Handle.TOP_LEFT, Handle.BOTTOM_LEFT]:
            target_l = min(mx, target_r - 1)  # 保证宽度至少1
        if h in [Handle.RIGHT, Handle.TOP_RIGHT, Handle.BOTTOM_RIGHT]:
            target_r = max(mx, target_l + 1)
        if h in [Handle.TOP, Handle.TOP_LEFT, Handle.TOP_RIGHT]:
            target_t = min(my, target_b - 1)
        if h in [Handle.BOTTOM, Handle.BOTTOM_LEFT, Handle.BOTTOM_RIGHT]:
            target_b = max(my, target_t + 1)

        # 3. 全局边界限制 (0 ~ limit)
        target_l = max(0, target_l)
        target_t = max(0, target_t)
        target_r = min(self.limit_w, target_r)
        target_b = min(self.limit_h, target_b)

        # 4. 转换回 item 坐标 (setPos + setRect)
        new_w = target_r - target_l
        new_h = target_b - target_t

        self.setPos(target_l, target_t)
        self.setRect(0, 0, new_w, new_h)
        self.update()  # 强制重绘手柄

    def _notify_change(self):
        if self.callback:
            rect = self.rect()
            roi_data = (
                (int(self.pos().x()), int(self.pos().y())),
                int(rect.width()),
                int(rect.height())
            )
            self.callback(roi_data)


class VectorLineROI(QGraphicsLineItem):
    """向量直线的方法"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.width = 5  # 默认线宽
        self.pen = QPen(Qt.blue, 1, Qt.SolidLine)
        self.setPen(self.pen)

        # 用于显示宽度的路径
        self.width_path = QGraphicsPathItem()
        self.width_path.setParentItem(self)
        self.width_path.setPen(QPen(Qt.transparent))
        self.width_path.setBrush(QBrush(QColor(255,255,0,50)))
        self.updateWidthPath()

    def setWidth(self, width):
        self.width = max(1, width)
        self.updateWidthPath()

    def updateWidthPath(self):
        """更新显示线宽的路径"""
        line = self.line()
        if line.isNull():
            return

        # 计算垂直于线的向量
        dx = line.x2() - line.x1()
        dy = line.y2() - line.y1()
        length = np.sqrt(dx * dx + dy * dy)
        if length == 0:
            return

        # 单位法向量
        nx = -dy / length
        ny = dx / length

        # 计算宽度路径的四个角点
        half_width = self.width / 2
        p1 = QPointF(line.x1() + nx * half_width, line.y1() + ny * half_width)
        p2 = QPointF(line.x1() - nx * half_width, line.y1() - ny * half_width)
        p3 = QPointF(line.x2() - nx * half_width, line.y2() - ny * half_width)
        p4 = QPointF(line.x2() + nx * half_width, line.y2() + ny * half_width)

        # 创建路径
        path = QPainterPath()
        path.moveTo(p1)
        path.lineTo(p2)
        path.lineTo(p3)
        path.lineTo(p4)
        path.closeSubpath()

        self.width_path.setPath(path)

    # def mouseMoveEvent(self, event):
    #     super().mouseMoveEvent(event)
    #     self.updateWidthPath()

    def getPixelValues(self, data, spatial_scale=1.0, temporal_scale=1.0):
        """
        获取直线ROI覆盖的像素值
        返回: [t, x, 2] 数组，其中最后一维是[位置, 平均值]
        """
        line = self.line()
        if line.isNull() :
            return None

        data_origin = data.data_origin
        # 计算直线参数
        x0, y0 = line.x1(), line.y1()
        x1, y1 = line.x2(), line.y2()
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx * dx + dy * dy)

        if length == 0:
            return None

        # 单位方向向量和法向量
        ux = dx / length
        uy = dy / length
        nx = -uy
        ny = ux

        # 采样参数
        step = spatial_scale
        num_samples = int(length / step) + 1
        half_width = self.width / 2

        # 准备输出数组
        t_dim = data.timelength
        result = np.zeros((t_dim, num_samples, 2))

        for i in range(num_samples):
            t = i * step
            if t > length:
                t = length

            # 记录位置信息
            result[:, i, 0] = t

            # 直线上的中心点
            cx = x0 + ux * t
            cy = y0 + uy * t

            # 收集宽度方向上的像素
            for w in np.linspace(-half_width, half_width, int(self.width) + 1):
                px = int(round(cx + nx * w))
                py = int(round(cy + ny * w))

                # 检查是否在图像范围内
                if 0 <= px < data_origin.shape[2] and 0 <= py < data_origin.shape[1]:
                    # 计算时间序列上的平均值
                    result[:, i, 1] += data_origin[:, py, px]

            # 计算宽度方向上的平均值
            if self.width > 0:
                result[:, i, 1] /= (int(self.width) + 1)

        return result


class ToolButtonWithMenu(QToolButton):
    """自定义工具按钮，内置上下文菜单功能"""
    contextMenuRequested = pyqtSignal(str, QToolButton)  # 信号：工具名称, 按钮对象

    def __init__(self, tool_name, parent=None):
        super().__init__(parent)
        self.tool_name = tool_name
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.handle_context_menu)

    def handle_context_menu(self, pos):
        """处理上下文菜单请求"""
        self.contextMenuRequested.emit(self.tool_name, self)


class WidthSliderDialog(QDialog):
    """宽度设置对话框，使用滑动条"""

    def __init__(self, parent=None, current_value=2, min_value=1, max_value=50, title="设置宽度"):
        super().__init__(parent)
        self.setWindowTitle(title)

        layout = QVBoxLayout()

        # 创建滑动条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_value, max_value)
        self.slider.setValue(current_value)
        self.slider.valueChanged.connect(self.update_label)

        # 显示当前值的标签
        self.value_label = QLabel(f"当前值: {current_value}")
        self.value_label.setAlignment(Qt.AlignCenter)

        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(self.value_label)
        layout.addWidget(self.slider)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def update_label(self, value):
        self.value_label.setText(f"当前值: {value}")

    def get_value(self):
        return self.slider.value()


class AnchorSelectDialog(QDialog):
    """锚点快速提取功能设置"""
    def __init__(self,param, parent=None):
        super().__init__()
        self.setWindowTitle("快速提取并绘制功能设置")
        self.setWindowFlags(Qt.Dialog |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowContextHelpButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.param = param
        self.parent = parent
        self.shape_items = ['square','circle']
        self.method_items = ['mean', 'max', 'min', 'median', 'quantile_075', 'std', 'sum', 'var']
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        active_layout = QHBoxLayout()
        active_layout.addWidget(QLabel("是否开启快速提取功能："))
        self.active_check = QCheckBox()
        active_layout.addWidget(self.active_check)
        self.active_check.setChecked(self.param["anchor_select"])

        self.region_shape_combo = QComboBox()
        self.region_shape_combo.addItems(["正方形", "圆形"])
        self.region_shape_combo.setCurrentIndex(self.shape_items.index(self.param["anchor_shape"]))
        self.region_size_input = QSpinBox()
        self.region_size_input.setMinimum(1)
        self.region_size_input.setMaximum(1000)
        self.region_size_input.setValue(self.param['anchor_size'])

        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("区域形状:"))
        shape_layout.addWidget(self.region_shape_combo)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("区域大小:"))
        size_layout.addWidget(self.region_size_input)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("提取方法"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(['平均值', '最大值', '最小值', '中位值', '0.75分位数', '标准差', '求和', '方差'])
        self.method_combo.setCurrentIndex(self.method_items.index(self.param["anchor_method"]))
        method_layout.addWidget(self.method_combo)

        layout.addLayout(active_layout)
        layout.addLayout(shape_layout)
        layout.addLayout(size_layout)
        layout.addLayout(method_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def accept(self):
        self.param["anchor_select"] = self.active_check.isChecked()
        self.param["anchor_shape"] = self.shape_items[self.region_shape_combo.currentIndex()]
        self.param["anchor_size"] = self.region_size_input.value()
        self.parent.params_update_signal.emit(self.param)
        self.parent.tool_parameters = self.param
        self.close()


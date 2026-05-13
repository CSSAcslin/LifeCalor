import logging
import math
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import ( QDialog, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGroupBox, QWidget, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor,  QVector4D

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from DataManager import Data, ProcessedData
from widget.DoubleSlider import DoubleSlider


class TextOverlay(QWidget):
    """轴刻度"""
    def __init__(self, parent=None, gl_view=None):
        super().__init__(parent)
        self.gl_view = gl_view
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 鼠标穿透，不挡拖拽
        self.ticks = []

    def update_ticks(self, ticks):
        self.ticks = ticks
        self.update()

    def paintEvent(self, ev):
        if not self.gl_view: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(30, 30, 30))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        for pos3d, text in self.ticks:
            pos2d = self.gl_view._project_to_screen(pos3d)
            if pos2d:
                # 给文字画一个半透明的白色底板，防止被 3D 网格遮挡看不清
                fm = painter.fontMetrics()
                rect = fm.boundingRect(text)
                rect.translate(int(pos2d[0]) + 10, int(pos2d[1]) + 5)
                painter.fillRect(rect.adjusted(-2, -2, 2, 2), QColor(255, 255, 255, 200))
                painter.drawText(int(pos2d[0]) + 10, int(pos2d[1]) + 5, text)


class InteractiveGLView(gl.GLViewWidget):
    """OpenGL视图"""
    boundsDragged = pyqtSignal(int, bool, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 设为 MATLAB 风格的高级浅灰背景，防止纯白刺眼
        self.setBackgroundColor(245, 245, 245)
        self.mousePos = None
        self.dragging_info = None
        self.handles_pos = {}
        self.overlay = TextOverlay(self, self)
        self.scale_xyz = (1.0, 1.0, 1.0)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.overlay.resize(self.size())  # 尺寸与 3D 画布同步

    def update(self):
        super().update()
        if hasattr(self, 'overlay'):
            self.overlay.update()  # 每次视角变动时，刷新 2D 刻度位置

    def set_handles(self, handles_dict):
        self.handles_pos = handles_dict

    def mousePressEvent(self, ev):
        self.mousePos = ev.pos()
        if ev.button() == Qt.LeftButton:
            closest_dist = float('inf')
            closest_key = None
            for key, pos3d in self.handles_pos.items():
                scr_pos = self._project_to_screen(pos3d)
                if scr_pos:
                    dist = math.hypot(scr_pos[0] - ev.pos().x(), scr_pos[1] - ev.pos().y())
                    if dist < 40 and dist < closest_dist:  # 增加抓取容差到 40 像素
                        closest_dist = dist
                        closest_key = key
            if closest_key:
                self.dragging_info = closest_key

    def mouseMoveEvent(self, ev):
        if not self.mousePos: return
        diff = ev.pos() - self.mousePos

        if ev.buttons() == Qt.RightButton:
            self.opts['azimuth'] -= diff.x() * 0.5
            self.opts['elevation'] += diff.y() * 0.5
            self.update()

        elif ev.buttons() == Qt.MidButton:
            self.pan(diff.x(), diff.y(), 0, relative='view-upright')

        elif ev.buttons() == Qt.LeftButton and self.dragging_info:
            axis_idx, is_max = self.dragging_info
            axis_vec = [0, 0, 0]
            step = max(1.0, self.handles_pos[self.dragging_info][axis_idx] * 0.01 + 1.0)
            axis_vec[axis_idx] = step

            p0 = self._project_to_screen(self.handles_pos[self.dragging_info])
            p1 = self._project_to_screen(np.array(self.handles_pos[self.dragging_info]) + np.array(axis_vec))

            if p0 and p1:
                vx, vy = p1[0] - p0[0], p1[1] - p0[1]
                pixel_len = math.hypot(vx, vy)
                if pixel_len > 0.1:
                    dot = (diff.x() * vx + diff.y() * vy) / pixel_len
                    delta_3d = (dot / pixel_len) * step
                    self.boundsDragged.emit(axis_idx, is_max, delta_3d)

        self.mousePos = ev.pos()

    def mouseReleaseEvent(self, ev):
        self.dragging_info = None

    def _project_to_screen(self, pos3d):

        sx, sy, sz = self.scale_xyz
        scaled_pos = (pos3d[0]*sx, pos3d[1]*sy, pos3d[2]*sz)
        try:
            vp = (0, 0, self.width(), self.height())
            pm = self.projectionMatrix(region=vp, viewport=vp)
        except TypeError:
            pm = self.projectionMatrix()

        vm = self.viewMatrix()
        vec4 = QVector4D(scaled_pos[0], scaled_pos[1], scaled_pos[2], 1.0)
        clip = pm * vm * vec4

        if clip.w() == 0: return None
        ndc_x, ndc_y = clip.x() / clip.w(), clip.y() / clip.w()
        sx = (ndc_x + 1.0) * self.width() / 2.0
        sy = (1.0 - ndc_y) * self.height() / 2.0
        return (sx, sy)


# ==========================================
# 4. 主界面提取器 Dialog
# ==========================================
class SpatialExtractor(QDialog):
    """空间数据切片器"""
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.aim_data = data
        self.data = data.data_origin if hasattr(data, 'data_origin') else data.data_processed
        self.shape = data.datashape
        self.ndim = data.ndim
        self.extracted_data = None

        self.bounds_float = [[0.0, s - 1.0] for s in self.shape]
        self.sliders = []
        self._is_updating = False

        self.setWindowTitle(f"🌌 空间数据提取器 - {'3D' if self.ndim == 3 else '2D'}")
        self.resize(1100, 700)

        self.init_ui()
        self.update_visualization()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # ============== 左侧控制面板 ==============
        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignTop)

        control_layout.addWidget(QLabel(f"<h2>数据边界: {self.shape}</h2>"))
        control_layout.addWidget(QLabel("<span style='color:#e74c3c; font-weight:bold;'>交互指南:</span><br>"
                                        "<b>左键</b> 拖拽滑块 或 空间红点<br>"
                                        "<b>右键</b> 旋转视角 | <b>中键</b> 平移视角"))

        axis_names = ['T (Z轴/深度)', 'H (Y轴/高度)', 'W (X轴/宽度)'] if self.ndim == 3 else ['H (Y轴/高度)',
                                                                                              'W (X轴/宽度)']
        for i, dim_size in enumerate(self.shape):
            group = QGroupBox(f"调节轴: {axis_names[i]}")
            l = QVBoxLayout()
            slider = DoubleSlider(0, dim_size - 1)
            slider.rangeChanged.connect(lambda low, high, idx=i: self._on_slider_changed(idx, low, high))
            l.addWidget(slider)
            group.setLayout(l)
            control_layout.addWidget(group)
            self.sliders.append(slider)

        control_layout.addStretch()
        opt_group = QGroupBox("🛠️ 数据输出与转置")
        opt_layout = QVBoxLayout()

        self.combo_boxes = []
        for i in range(self.ndim):
            h_lay = QHBoxLayout()
            h_lay.addWidget(QLabel(f"输出 第 {i} 维 (原{axis_names[i]}轴):"))
            cb = QComboBox()
            cb.addItems([f"{name.split(' ')[0]}" for name in axis_names])
            cb.setCurrentIndex(i)
            h_lay.addWidget(cb)
            self.combo_boxes.append(cb)
            opt_layout.addLayout(h_lay)

        self.chk_squeeze = QCheckBox("自动 Squeeze (降维厚度1)")
        self.chk_squeeze.setChecked(True)
        opt_layout.addWidget(self.chk_squeeze)
        opt_group.setLayout(opt_layout)
        control_layout.addWidget(opt_group)

        time_group = QGroupBox("⏰ 时间轴修改")
        time_layout = QVBoxLayout()
        time_layout.addWidget(QLabel(self.format_time_axis_info(self.aim_data.time_point)))
        time_layout1 = QHBoxLayout()
        self.amend_check = QCheckBox("是否修改时间轴")
        time_layout.addWidget(self.amend_check)
        time_layout1.addWidget(QLabel("与时间轴对应的轴是："))
        self.time_axis = QComboBox()
        self.time_axis.addItems(axis_names)
        time_layout1.addWidget(self.time_axis)
        time_layout.addLayout(time_layout1)
        time_group.setLayout(time_layout)
        control_layout.addWidget(time_group)

        control_layout.addStretch()
        btn_confirm = QPushButton("✅ 确认提取")
        btn_confirm.setStyleSheet(
            "padding: 12px; font-weight: bold; background-color: #2b78e4; color: white; font-size: 14px; border-radius: 4px;")
        btn_confirm.clicked.connect(self.accept_extraction)
        control_layout.addWidget(btn_confirm)

        left_widget = QWidget()
        left_widget.setLayout(control_layout)
        left_widget.setFixedWidth(320)
        main_layout.addWidget(left_widget)

        # ============== 右侧绘图区 ==============
        right_layout = QVBoxLayout()
        if self.ndim == 2:
            self.init_2d_view(right_layout)
        else:
            self.init_3d_view(right_layout)
        main_layout.addLayout(right_layout, stretch=1)

    def init_2d_view(self, layout):
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.invertY(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)

        H, W = self.shape
        self.plot_widget.addItem(pg.PlotDataItem(x=[0, W, W, 0, 0], y=[0, 0, H, H, 0], pen=pg.mkPen('k', width=2)))
        self.roi = pg.RectROI(pos=[0, 0], size=[W, H], pen=pg.mkPen(color=(0, 100, 255), width=3))
        self.roi.addScaleHandle([1, 1], [0, 0]);
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.plot_widget.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self._on_roi_changed)

    def init_3d_view(self, layout):
        # 顶部坐标系说明
        legend = QLabel(
            "🔴 <b>X轴</b> (宽度 W) &nbsp;&nbsp;&nbsp; 🟢 <b>Y轴</b> (高度 H) &nbsp;&nbsp;&nbsp; 🔵 <b>Z轴</b> (深度 T)")
        legend.setAlignment(Qt.AlignCenter)
        legend.setStyleSheet("font-size: 14px; padding: 5px; background: #E0E0E0; border-radius: 5px;")
        layout.addWidget(legend)
        self.gl_widget = InteractiveGLView()

        T, H, W = self.shape
        scales = [1.0, 1.0, 1.0]  # 默认物理比例为 1:1
        max_dim, min_dim = max(W, H, T), min(W, H, T)
        def dis_power(aim_dim):
            return np.log10(aim_dim / min_dim)
        dis_power_max = dis_power(max_dim)
        # 如果长宽比超过 3 倍，启动视图压缩
        if min_dim > 0 and dis_power_max > 0.5:
            for i, dim in enumerate([W, H, T]):
                # 指数压缩：10000的长度会被压缩为不到 1000 的视觉长度，但依然保持最长
                if dis_power(dim) >= 0.5:
                    v_dim = min_dim * (dim / min_dim) ** 0.2
                    scales[i] = v_dim / dim

        self.gl_widget.scale_xyz = tuple(scales)
        ws, hs, ts = scales
        vW, vH, vT = W * ws, H * hs, T * ts  # 获取视觉映射后的长宽高

        self.gl_widget.opts['distance'] = max(vW, vH, vT) * 1.8
        # 调整初始视角，更符合直觉的等轴测俯视
        self.gl_widget.opts['elevation'] = 25
        self.gl_widget.opts['azimuth'] = 45
        self.gl_widget.boundsDragged.connect(self._on_3d_drag)
        layout.addWidget(self.gl_widget, stretch=1)

        # ================== 1. 背景网格 ==================
        grid_color = pg.mkColor(180, 180, 180, 150)
        def spacing(size):
            return max((int(math.log10(size+1)) - 1),1 ) * 10

        # 底部网格 (XY 面)
        grid_z = gl.GLGridItem(color=grid_color)
        grid_z.setSize(vW, vH, 0)
        grid_z.setSpacing(spacing(vW), spacing(vH), 0)
        grid_z.translate(vW / 2, vH / 2, 0)
        self.gl_widget.addItem(grid_z)

        # 背部网格 (XZ 面)
        grid_y = gl.GLGridItem(color=grid_color)
        grid_y.setSize(vW, vT, 0)
        grid_y.setSpacing(spacing(vW), spacing(vT), 0)
        grid_y.rotate(90, 1, 0, 0)
        grid_y.translate(vW / 2, 0, vT / 2)
        self.gl_widget.addItem(grid_y)

        # 侧边网格 (YZ 面)
        grid_x = gl.GLGridItem(color=grid_color)
        grid_x.setSize(vH, vT, 0)
        grid_x.setSpacing(spacing(vH), spacing(vT), 0)
        grid_x.rotate(90, 0, 1, 0)
        grid_x.rotate(90, 1, 0, 0)
        grid_x.translate(0, vH / 2, vT / 2)
        self.gl_widget.addItem(grid_x)

        # 全局边界线框 (暗灰色)
        outer_box = gl.GLBoxItem(size=pg.Vector(vW, vH, vT), color=(120, 120, 120, 200))
        self.gl_widget.addItem(outer_box)

        # ================== 2. 轴与刻度 (Ticks) ==================
        # 三色加粗主坐标轴
        self.gl_widget.addItem(
            gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [vW * 1.1, 0, 0]]), color=pg.mkColor(255, 0, 0, 255), width=3,
                              antialias=True))
        self.gl_widget.addItem(
            gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, vH * 1.1, 0]]), color=pg.mkColor(0, 180, 0, 255), width=3,
                              antialias=True))
        self.gl_widget.addItem(
            gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, vT * 1.1]]), color=pg.mkColor(0, 0, 255, 255), width=3,
                              antialias=True))

        # 物理刻度线与 2D 文字映射
        ticks_data = []
        tick_lines = []
        for x in np.linspace(0, W, 6):
            ticks_data.append(((x, 0, 0), f"{int(x)}"))
            tick_lines.extend([[x*ws, 0, 0], [x*ws, -vH * 0.03, 0]])  # 画出小短线
        for y in np.linspace(0, H, 6):
            if y == 0: continue
            ticks_data.append(((0, y, 0), f"{int(y)}"))
            tick_lines.extend([[0, y*hs, 0], [-vW * 0.03, y*hs, 0]])
        for z in np.linspace(0, T, 6):
            if z == 0: continue
            ticks_data.append(((0, 0, z), f"{int(z)}"))
            tick_lines.extend([[0, 0, z*ts], [-vW * 0.03, 0, z*ts]])

        self.gl_widget.overlay.update_ticks(ticks_data)
        self.gl_widget.addItem(
            gl.GLLinePlotItem(pos=np.array(tick_lines), mode='lines', color=pg.mkColor(50, 50, 50, 255), width=2))

        # ================== 3. 透明水晶体 & 线框 ==================
        # 构造单位立方体的顶点
        verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                         dtype=float)
        faces = np.array(
            [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7],
             [2, 7, 6], [3, 0, 4], [3, 4, 7]])

        # 内层：【核心修复】绝对纯净的半透明蓝色水晶体 (使用 0.0~1.0 的浮点数限定)
        self.inner_mesh = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=verts, faces=faces),
                                        color=(0.0, 0.4, 1.0, 0.25),  # 浮点数：R=0, G=0.4, B=1.0, 透明度=0.25
                                        shader=None,  # 取消阴影遮罩，让其完全剔透
                                        glOptions='translucent',
                                        drawEdges=False)
        self.gl_widget.addItem(self.inner_mesh)

        # 外层：锋利的深蓝色轮廓边框
        edge_indices = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
        self.inner_box_outline = gl.GLLinePlotItem(pos=verts[edge_indices], mode='lines',
                                                   color=pg.mkColor(0, 50, 200, 255), width=4.0, antialias=True)
        self.gl_widget.addItem(self.inner_box_outline)

        # ================== 4. 真正纯正的大红点 ==================
        # 抛弃 numpy array 色彩映射，直接使用 pg.mkColor 传递整数 rgba，确保万无一失
        self.handles_scatter = gl.GLScatterPlotItem(size=16, color=pg.mkColor(255, 0, 0, 255), pxMode=True)
        self.gl_widget.addItem(self.handles_scatter)

    def _on_slider_changed(self, idx, low, high):
        if self._is_updating: return
        self._is_updating = True
        self.bounds_float[idx] = [float(low), float(high)]
        self.update_visualization()
        self._is_updating = False

    def _on_roi_changed(self):
        if self._is_updating: return
        self._is_updating = True
        pos, size = self.roi.pos(), self.roi.size()
        H, W = self.shape
        w1, w2 = max(0, pos.x()), min(W - 1, pos.x() + size.x())
        h1, h2 = max(0, pos.y()), min(H - 1, pos.y() + size.y())

        self.bounds_float[0] = [h1, h2]
        self.bounds_float[1] = [w1, w2]
        self.sliders[0].setValues(int(h1), int(h2))
        self.sliders[1].setValues(int(w1), int(w2))
        self._is_updating = False

    def _on_3d_drag(self, axis_idx, is_max, delta_3d):
        ctrl_idx = 2 - axis_idx
        limit_max = self.shape[ctrl_idx] - 1
        current_val = self.bounds_float[ctrl_idx][1 if is_max else 0]
        new_val = current_val + delta_3d

        if is_max:
            new_val = max(self.bounds_float[ctrl_idx][0] + 0.1, min(new_val, limit_max))
        else:
            new_val = max(0, min(new_val, self.bounds_float[ctrl_idx][1] - 0.1))

        self.bounds_float[ctrl_idx][1 if is_max else 0] = new_val
        self.sliders[ctrl_idx].setValues(int(self.bounds_float[ctrl_idx][0]), int(self.bounds_float[ctrl_idx][1]))
        self.update_visualization()

    def update_visualization(self):
        if self.ndim == 2:
            (h1, h2), (w1, w2) = self.bounds_float
            self.roi.setPos((w1, h1), update=False)
            self.roi.setSize((w2 - w1, h2 - h1), update=False)
        else:
            (t1, t2), (h1, h2), (w1, w2) = self.bounds_float
            ws, hs, ts = self.gl_widget.scale_xyz  # 获取视觉压缩系数

            # 使用 3D Transform 矩阵进行极速缩放和平移，避免重新计算 Mesh 顶点
            tr = pg.Transform3D()
            tr.scale(ws, hs, ts)
            tr.translate(w1, h1, t1)
            tr.scale(max(0.1, w2 - w1), max(0.1, h2 - h1), max(0.1, t2 - t1))

            # 同步应用给 水晶玻璃体 和 锐利边框
            self.inner_mesh.setTransform(tr)
            self.inner_box_outline.setTransform(tr)

            # 计算 6 个面的中心红点
            handles_pos = {
                (0, False): (w1, (h1 + h2) / 2, (t1 + t2) / 2),
                (0, True): (w2, (h1 + h2) / 2, (t1 + t2) / 2),
                (1, False): ((w1 + w2) / 2, h1, (t1 + t2) / 2),
                (1, True): ((w1 + w2) / 2, h2, (t1 + t2) / 2),
                (2, False): ((w1 + w2) / 2, (h1 + h2) / 2, t1),
                (2, True): ((w1 + w2) / 2, (h1 + h2) / 2, t2)
            }
            pts = np.array(list(handles_pos.values()))

            pts_scaled = pts * np.array([ws, hs, ts])

            self.handles_scatter.setData(pos=pts_scaled)
            self.gl_widget.set_handles(handles_pos)

    def accept_extraction(self):
        ranges = [[int(min_v), int(max_v)] for min_v, max_v in self.bounds_float]
        slices = tuple(slice(r[0], r[1] + 1) for r in ranges)
        sub_data = self.data[slices]

        target_axes = [cb.currentIndex() for cb in self.combo_boxes]
        if len(set(target_axes)) != self.ndim:
            QtWidgets.QMessageBox.warning(self, "轴错误", "导出的轴有重复，请确保每个维度不同！")
            return

        sub_data = np.transpose(sub_data, target_axes)
        if self.chk_squeeze.isChecked():
            sub_data = np.squeeze(sub_data)

        time_point = self.aim_data.time_point
        if self.amend_check.isChecked():
            try:
                r_t = self.bounds_float[self.time_axis.currentIndex()]
                time_point = self.aim_data.time_point[int(r_t[0]):int(r_t[1])+1]
            except IndexError as e:
                QtWidgets.QMessageBox.warning(self, "数值错误", "对时间轴切片时，发现超出原本时间轴刻度")
                return

        ori_data = self.aim_data
        self.extracted_data = ProcessedData(ori_data.timestamp,
                                            f'{ori_data.name}@cropped',
                                            'data_cropped',
                                            time_point=time_point,
                                            data_processed=sub_data,
                                            out_processed ={'crop_slice': slices ,**ori_data.out_processed, **ori_data.parameters})
        self.accept()

    @staticmethod
    def format_time_axis_info(data):
        """
        返回时间轴信息。
        """
        if data is None:
            return '该数据无时间轴'

        if not isinstance(data, np.ndarray) or data.ndim != 1:
            return '该数据时间轴信息：无效数据'

        n = len(data)
        if n == 0:
            return '该数据时间轴信息：无数据点'
        if n == 1:
            return f"该数据时间轴信息：{data[0]:.1f}，共计{n}个点"

        # 一般情况：至少有两个点
        return (f"该数据时间轴信息：{data[0]:.1f}, {data[1]:.1f} ... \n"
                f"{data[-2]:.1f}, {data[-1]:.1f}，共计{n}个点")
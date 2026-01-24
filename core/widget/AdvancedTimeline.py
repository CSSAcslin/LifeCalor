from PyQt5.QtCore import pyqtSignal, Qt, QRectF, QPointF, QPoint
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush
from PyQt5.QtWidgets import QWidget, QSizePolicy


class AdvancedTimeline(QWidget):
    """高阶时间轴组件"""
    # 信号：当前帧改变，区间改变
    positionChanged = pyqtSignal(int)
    selectionChanged = pyqtSignal(int, int)
    rightClicked = pyqtSignal(int, str, QPoint)

    # 定义交互模式常量
    MODE_NONE = 0
    MODE_MOVE_HEAD = 1  # 拖动播放头
    MODE_MOVE_START = 2  # 拖动起点
    MODE_MOVE_END = 3  # 拖动终点
    MODE_PAN = 4  # 平移视图
    MODE_MOVE_SELECTION = 5 # 移动选区

    def __init__(self, total_frames=1000, fps=0, parent=None, time_point = None):
        super().__init__(parent)

        # --- 数据模型 ---
        self.total_frames = total_frames - 1
        self.fps = fps
        self.time_point = time_point
        self.current_frame = 0
        self.selection_start = 0
        self.selection_end = total_frames

        # --- 视图控制 ---
        # zoom_level: 每个帧占用的像素数 (pixels per frame)
        self.zoom_level = 0.5
        # scroll_offset: 视图最左侧对应的像素偏移量（用于平移）
        self.scroll_offset = 0

        # 样式配置
        # self.bg_color = QColor(40, 40, 40)  # 深灰背景
        # self.ruler_bg_color = QColor(55, 55, 55)  # 刻度尺背景
        # self.tick_color = QColor(180, 180, 180)  # 刻度颜色
        # self.cursor_color = QColor(62, 166, 255)  # Pr 风格蓝
        # self.select_color = QColor(62, 166, 255, 60)  # 半透明蓝色选区
        self.bg_color = Qt.transparent  # 背景透明
        self.ruler_color = QColor(240, 240, 240)  # 刻度尺背景(浅灰)
        self.tick_color = QColor(80, 80, 80)  # 刻度线(深灰)
        self.text_color = QColor(0, 0, 0)  # 文字(黑色)
        # 游标和选区颜色
        self.cursor_color = QColor(0, 120, 215)  # 蓝色游标
        self.select_border_color = QColor(0, 120, 215)
        self.select_fill_color = QColor(0, 120, 215, 50)  # 浅蓝半透明填充
        self.handle_center_color = QColor(255, 255, 255, 200)  # 手柄颜色(白)
        self.handle_bg_color = QColor(0, 120, 215, 200)  # 手柄背景(深蓝)

        # 鼠标交互状态
        self.interaction_mode = self.MODE_NONE
        self.last_mouse_x = 0
        self.handle_width = 10  # 拖动句柄的感应宽度
        self.drag_start_pos = 0  # 记录点击时的帧位置
        self.drag_selection_len = 0  # 记录区间长度

        # 设置高度策略
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(30)
        self.setMouseTracking(True)
        self.initialized = False  # 标记是否已完成初始化布局

    # 核心逻辑：坐标映射
    def frame_to_x(self, frame):
        return (frame * self.zoom_level) - self.scroll_offset

    def x_to_frame(self, x):
        return (x + self.scroll_offset) / self.zoom_level

    def get_content_width(self):
        return self.total_frames * self.zoom_level

    def get_min_zoom(self):
        """计算能够显示完整时间轴的最小缩放比例"""
        if self.width() <= 0: return 0.001
        return self.width() / self.total_frames

    def get_max_zoom(self):
        """计算最大缩放比例。"""
        if self.width() <= 0: return 1.0  # 防止除以0
        min_visible_frames = 3.0 # 最少可见帧数
        return self.width() / min_visible_frames

    def clamp_view(self):
        """限制视图缩放比例和范围"""
        # 1. 限制最小缩放 (只能放大，不能比"全视图"更小)
        min_z = self.get_min_zoom()
        max_z = self.get_max_zoom()
        if max_z < min_z:
            max_z = min_z
        if self.zoom_level < min_z:
            self.zoom_level = min_z
        elif self.zoom_level > max_z:
            self.zoom_level = max_z

        # 2. 限制滚动范围
        content_w = self.get_content_width()
        view_w = self.width()

        max_offset = content_w - view_w

        # 如果内容宽度小于视图宽度（通常因为限制了min_zoom，这种情况只会正好相等）
        if max_offset < 0:
            max_offset = 0

        # 强制限制
        if self.scroll_offset < 0: self.scroll_offset = 0
        if self.scroll_offset > max_offset: self.scroll_offset = max_offset

    # 事件处理
    def resizeEvent(self, event):
        """修改点 3: 窗口大小改变时保持视图适应，初次显示时铺满全屏"""
        super().resizeEvent(event)
        if not self.initialized and self.width() > 0:
            # 初始化：将缩放设置为正好填满屏幕
            self.zoom_level = self.get_min_zoom()
            self.scroll_offset = 0
            self.initialized = True
        else:
            # 窗口调整大小时，确保不越界
            self.clamp_view()
        self.update()

    # 绘图事件
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        ruler_h = int(h * 0.5)  # 刻度尺占上面50%

        # 1. 绘制背景
        painter.fillRect(0, 0, w, h, self.bg_color)
        painter.fillRect(0, 0, w, ruler_h, self.ruler_color)

        # 2. 绘制选中区域 (Selection Range)
        x_start = self.frame_to_x(self.selection_start)
        x_end = self.frame_to_x(self.selection_end)

        # 绘制区域填充
        sel_rect = QRectF(x_start, ruler_h, x_end - x_start, h - ruler_h)
        painter.fillRect(sel_rect, self.select_fill_color)
        pen_border = QPen(self.select_border_color)
        pen_border.setWidth(2)
        painter.setPen(pen_border)

        # 选区拖动手柄
        selection_width = x_end - x_start
        handle_w = 24
        handle_h = 14

        if selection_width > handle_w + 10:
            center_x = (x_start + x_end) / 2
            # 手柄位于轨道区域的垂直居中位置
            center_y = ruler_h + (h - ruler_h) / 2

            # 绘制手柄背景 (圆角矩形)
            handle_rect = QRectF(center_x - handle_w / 2, center_y - handle_h / 2, handle_w, handle_h)
            painter.setBrush(QBrush(self.handle_bg_color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(handle_rect, 4, 4)

            # 绘制三条杠 (汉堡菜单样式)
            painter.setPen(QPen(self.handle_center_color, 2))
            line_h = 8
            line_y_top = center_y - line_h / 2
            line_y_btm = center_y + line_h / 2

            # 画三条竖线
            for offset in [-4, 0, 4]:
                px = center_x + offset
                painter.drawLine(QPointF(px, line_y_top), QPointF(px, line_y_btm))

        # 起点线
        if 0 <= x_start <= w:
            painter.drawLine(QPointF(x_start, ruler_h), QPointF(x_start, h))
            # 画一个小三角形或者加粗显示作为"把手"
            painter.fillRect(QRectF(x_start, h - 6, 6, 6), self.select_border_color)  # 右下角标记

        # 终点线
        if 0 <= x_end <= w:
            painter.drawLine(QPointF(x_end, ruler_h), QPointF(x_end, h))
            painter.fillRect(QRectF(x_end - 6, h - 6, 6, 6), self.select_border_color)  # 左下角标记

        # 4. 绘制刻度
        painter.setPen(self.tick_color)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        # 动态步长逻辑
        target_pixel_step = 120
        step_frames = max(1, int(target_pixel_step / self.zoom_level))

        # 优化刻度显示的数值逻辑 (1, 5, 10, 25, 50, 100...)
        scales = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]
        for s in scales:
            if step_frames <= s:
                step_frames = s
                break

        start_frame = int(self.x_to_frame(0))
        end_frame = int(self.x_to_frame(w)) + 1

        # 刻度对齐
        first_tick = (start_frame // step_frames) * step_frames

        for f in range(first_tick, end_frame + step_frames, step_frames):
            if f > self.total_frames: break
            x = self.frame_to_x(f)

            # 主刻度
            painter.drawLine(QPointF(x, 0), QPointF(x, ruler_h))

            # 文字
            painter.setPen(self.text_color)
            if self.fps > 0:
                seconds = f / self.fps
                minutes = int(seconds // 60)
                remaining_seconds = int(seconds % 60)
                frames = int((seconds - int(seconds)) * self.fps)
                time_str = f"{minutes:02d}:{remaining_seconds:02d}:{frames:0{len(str(self.fps))}d}"
            elif self.time_point is not None:
                val = self.time_point[f]
                time_str = str(val)
            else:
                time_str = str(f)
            painter.drawText(QPointF(x + 4, ruler_h - 4), time_str)
            painter.setPen(self.tick_color)

        # 5. 绘制当前帧游标 (Playhead)
        cursor_x = self.frame_to_x(self.current_frame)
        if 0 <= cursor_x <= w:
            pen = QPen(self.cursor_color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(QPointF(cursor_x, 0), QPointF(cursor_x, h))

            # 游标头
            painter.setBrush(QBrush(self.cursor_color))
            poly = [
                QPointF(cursor_x - 5, 0),
                QPointF(cursor_x + 5, 0),
                QPointF(cursor_x, ruler_h / 2)
            ]
            painter.drawPolygon(*poly)

    # 鼠标交互
    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        start_x = self.frame_to_x(self.selection_start)
        end_x = self.frame_to_x(self.selection_end)
        frame = self.x_to_frame(x)
        ruler_h = int(self.height() * 0.5)

        # 中键平移
        if event.button() == Qt.MiddleButton:
            self.interaction_mode = self.MODE_PAN
            self.last_mouse_x = x
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            # 判断点击位置

            # 1. 检查是否点击了 Start 把手 (且在刻度尺下方)

            if y > ruler_h and abs(x - start_x) < self.handle_width:
                self.interaction_mode = self.MODE_MOVE_START

            # 2. 检查是否点击了 End 把手
            elif y > ruler_h and abs(x - end_x) < self.handle_width:
                self.interaction_mode = self.MODE_MOVE_END

            # 点击选区拖动句柄
            elif y > ruler_h and start_x + self.handle_width < x < end_x - self.handle_width:
                self.interaction_mode = self.MODE_MOVE_SELECTION
                self.setCursor(Qt.SizeAllCursor)  # 十字光标

                # 记录点击瞬间的鼠标帧位置，用于计算偏移
                self.drag_start_pos = frame
                # 记录当前的起点，后续计算都是基于这个起点的偏移
                self.drag_selection_base = self.selection_start
                # 记录长度，用于计算终点
                self.drag_selection_len = self.selection_end - self.selection_start

            # 3. 点击刻度尺区域 -> 移动播放头
            elif y <= ruler_h:
                self.interaction_mode = self.MODE_MOVE_HEAD
                self.set_current_frame(int(frame))  # 立即跳转
                self.positionChanged.emit(self.current_frame)

            # 4. 点击其他空白区域 -> 默认也移动播放头，或者你可以改为"清除选区"
            else:
                # 这里为了操作方便，点击下方空白处通常不移动游标，防止误触把手
                # 但根据pr逻辑，点击哪里都会移动游标，除非拖动了选区
                self.interaction_mode = self.MODE_MOVE_HEAD
                self.set_current_frame(int(frame))
                self.positionChanged.emit(self.current_frame)

            self.update()

        if event.button() == Qt.RightButton:
            if y > ruler_h and start_x + self.handle_width < x < end_x - self.handle_width:
                in_selection = 'handle'
            else:
                in_selection = 'other'
            self.rightClicked.emit(int(frame), in_selection, event.globalPos()) # 右键截取数据

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        ruler_h = self.height() * 0.5

        # --- 悬停状态更新光标样式 ---
        if self.interaction_mode == self.MODE_NONE:
            start_x = self.frame_to_x(self.selection_start)
            end_x = self.frame_to_x(self.selection_end)

            if y > ruler_h:
                # 左右边缘
                if abs(x - start_x) < self.handle_width or abs(x - end_x) < self.handle_width:
                    self.setCursor(Qt.SizeHorCursor)
                # --- 新增: 鼠标在选区中间悬停时显示手型 ---
                elif start_x < x < end_x:
                    self.setCursor(Qt.OpenHandCursor)
                # -------------------------------------
                else:
                    self.setCursor(Qt.ArrowCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

        # --- 拖拽处理 ---
        if self.interaction_mode == self.MODE_PAN:
            delta = x - self.last_mouse_x
            self.scroll_offset -= delta
            self.clamp_view()
            self.last_mouse_x = x
            self.update()

        elif self.interaction_mode == self.MODE_MOVE_HEAD:
            frame = int(self.x_to_frame(x))
            frame = max(0, min(frame, self.total_frames))
            self.set_current_frame(frame)
            self.positionChanged.emit(self.current_frame)

        elif self.interaction_mode == self.MODE_MOVE_START:
            frame = int(self.x_to_frame(x))
            # 限制：不能小于0，不能大于 End
            frame = max(0, min(frame, self.selection_end))
            self.selection_start = frame
            self.selectionChanged.emit(self.selection_start, self.selection_end)
            self.update()

        elif self.interaction_mode == self.MODE_MOVE_END:
            frame = int(self.x_to_frame(x))
            # 限制：不能小于 Start，不能大于总帧数
            frame = max(self.selection_start, min(frame, self.total_frames))
            self.selection_end = frame
            self.selectionChanged.emit(self.selection_start, self.selection_end)
            self.update()

        elif self.interaction_mode == self.MODE_MOVE_SELECTION:
            current_frame = self.x_to_frame(x)

            # 计算鼠标移动了多少帧
            delta = current_frame - self.drag_start_pos

            # 计算新的 Start
            new_start = self.drag_selection_base + delta
            new_end = new_start + self.drag_selection_len

            # 边界限制 (Clamping)
            if new_start < 0:
                new_start = 0
                new_end = self.drag_selection_len
            elif new_end > self.total_frames:
                new_end = self.total_frames
                new_start = self.total_frames - self.drag_selection_len

            # 应用更新
            self.selection_start = int(new_start)
            self.selection_end = int(new_end)
            self.selectionChanged.emit(self.selection_start, self.selection_end)
            self.update()

    def mouseReleaseEvent(self, event):
        self.interaction_mode = self.MODE_NONE
        self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        angle = event.angleDelta().y()
        mouse_x = event.x()
        mouse_frame_before = self.x_to_frame(mouse_x)

        zoom_factor = 1.15
        if angle > 0:
            target_zoom = self.zoom_level * zoom_factor
        else:
            target_zoom = self.zoom_level / zoom_factor

        # 1. 立即计算新的 Offset 以保持鼠标位置不动
        # new_x = frame * new_zoom - new_offset
        # new_offset = frame * new_zoom - mouse_x
        min_z = self.get_min_zoom()
        max_z = self.get_max_zoom()

        # 限制 target_zoom 在 [min_z, max_z] 之间
        target_zoom = max(min_z, min(max_z, target_zoom))

        if abs(target_zoom - self.zoom_level) < 0.00001:
            return

        # 5. 应用新的缩放
        self.zoom_level = target_zoom

        self.scroll_offset = (mouse_frame_before * self.zoom_level) - mouse_x

        # 2. 应用严格的边界限制 (这也处理了缩放过小的问题)
        self.clamp_view()
        self.update()

    # 公共接口 API
    def set_current_frame(self, frame):
        if 0 <= frame <= self.total_frames:
            self.current_frame = frame
            # 如果游标跑出屏幕，自动跟随滚动
            x = self.frame_to_x(frame)
            if x < 0 or x > self.width():
                self.scroll_offset = (frame * self.zoom_level) - (self.width() / 2)
                self.clamp_view()
            self.update()
            self.positionChanged.emit(self.current_frame)

    def set_selection(self, start, end):
        self.selection_start = start
        self.selection_end = end
        self.update()

    def set_fps(self, fps):
        """
        更新 FPS 并重绘标尺
        """
        self.fps = max(0, fps)  # 保证不为负数即可
        self.update()

    def set_time_point(self, time_point):
        """
        设置时间点数组。
        :param time_point: numpy 数组或列表，长度应对应 total_frames
        """
        self.time_point = time_point
        self.update() # 触发重绘

    def update_data_range(self, new_total_frames):
        """当外部数据发生变化时调用"""
        self.total_frames = new_total_frames
        self.current_frame = 0
        self.selection_start = 0
        self.selection_end = new_total_frames if new_total_frames > 0 else 0

        # 重置视图状态
        self.zoom_level = self.get_min_zoom()
        self.scroll_offset = 0
        self.update()
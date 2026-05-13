from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPropertyAnimation, pyqtProperty, pyqtSignal, QEasingCurve, QRectF, QSize
from PyQt5.QtGui import QPainter, QColor, QBrush


class TriStateSwitch(QWidget):
    """三状态切换组件"""
    valueChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._value = 0
        self._anim_progress = 0.0

        # === 核心修改 1: 设定固定的宽高比 ===
        # 3.0 表示宽度是高度的 3 倍。如果想扁一点就改大，想圆一点就改小
        self._aspect_ratio = 3.0

        # 颜色配置
        self._color_left = QColor("#34C759")
        self._color_mid = QColor("#E5E5EA")
        self._color_right = QColor("#007AFF")
        self._handle_color = QColor("#FFFFFF")

        # === 核心修改 2: 告诉 Layout 高度依赖宽度 ===
        # 使用 Preferred 策略，允许伸缩
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHeightForWidth(True)  # 关键：激活 HeightForWidth 模式
        self.setSizePolicy(sizePolicy)

        self._animation = QPropertyAnimation(self, b"anim_progress", self)
        self._animation.setEasingCurve(QEasingCurve.OutQuint)
        self._animation.setDuration(350)

    # === 核心修改 3: 正确实现 heightForWidth ===
    def heightForWidth(self, width):
        # 只要宽度确定了，高度必须是宽度的 1/ratio
        return int(width / self._aspect_ratio)

    def sizeHint(self):
        return QSize(80, int(80 / self._aspect_ratio))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    # --- 属性与接口 (保持不变) ---
    @pyqtProperty(float)
    def anim_progress(self):
        return self._anim_progress

    @anim_progress.setter
    def anim_progress(self, p):
        self._anim_progress = p
        self.update()

    def setValue(self, val):
        if val not in [0, 1, 2]: return
        if self._value == val: return
        self._value = val
        self.start_animation()
        self.valueChanged.emit(val)

    def value(self):
        return self._value

    # --- 交互逻辑 (需要微调点击区域判定) ---
    def mousePressEvent(self, event):
        # 依然根据宽度分三份，因为宽度是主导
        w = self.width()
        click_x = event.x()
        if click_x < w / 3:
            target = 0
        elif click_x < (w / 3) * 2:
            target = 1
        else:
            target = 2
        self.setValue(target)

    def start_animation(self):
        self._animation.stop()
        self._animation.setStartValue(self._anim_progress)
        self._animation.setEndValue(float(self._value))
        self._animation.start()

    def get_bg_color(self):
        # (颜色算法保持不变)
        p = self._anim_progress
        if p <= 1.0:
            ratio = p
            r = self._color_left.red() + ratio * (self._color_mid.red() - self._color_left.red())
            g = self._color_left.green() + ratio * (self._color_mid.green() - self._color_left.green())
            b = self._color_left.blue() + ratio * (self._color_mid.blue() - self._color_left.blue())
            return QColor(int(r), int(g), int(b))
        else:
            ratio = p - 1.0
            r = self._color_mid.red() + ratio * (self._color_right.red() - self._color_mid.red())
            g = self._color_mid.green() + ratio * (self._color_right.green() - self._color_mid.green())
            b = self._color_mid.blue() + ratio * (self._color_right.blue() - self._color_mid.blue())
            return QColor(int(r), int(g), int(b))

    # === 核心修改 4: 绘图逻辑彻底解耦 ===
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        # 获取控件真实宽度
        full_width = self.width()
        full_height = self.height()

        # 1. 计算【逻辑绘图区域】
        # 我们不信 Layout 给的高度，我们自己算：基于宽度和比例
        logic_height = full_width / self._aspect_ratio

        # 2. 如果 Layout 强行拉大了高度，我们把绘图区域垂直居中
        # 这样即使控件被拉成正方形，开关依然是长条形，不会变形
        offset_y = (full_height - logic_height) / 2

        # 定义一个 "干净" 的矩形，所有的绘图都基于这个 rect
        draw_rect = QRectF(0, offset_y, full_width, logic_height)

        # 半径 = 逻辑高度的一半
        radius = logic_height / 2

        # 3. 绘制背景 (使用 draw_rect 而不是 self.rect())
        bg_color = self.get_bg_color()
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(draw_rect, radius, radius)

        # 4. 绘制锚点 (同样基于 draw_rect)
        painter.setBrush(QBrush(QColor(0, 0, 0, 30)))
        dot_size = logic_height * 0.15
        center_y = offset_y + logic_height / 2  # 垂直中心

        centers_x = [logic_height / 2, full_width / 2, full_width - logic_height / 2]
        for cx in centers_x:
            painter.drawEllipse(QRectF(cx - dot_size / 2, center_y - dot_size / 2, dot_size, dot_size))

        # 5. 绘制滑块
        margin = 4
        # 注意：如果控件特别小，margin可能会导致负数，做个保护
        safe_margin = min(margin, logic_height * 0.1)

        handle_radius = (logic_height - 2 * safe_margin) / 2

        left_cx = safe_margin + handle_radius
        mid_cx = full_width / 2
        right_cx = full_width - safe_margin - handle_radius

        if self._anim_progress <= 1.0:
            ratio = self._anim_progress
            cur_cx = left_cx + ratio * (mid_cx - left_cx)
        else:
            ratio = self._anim_progress - 1.0
            cur_cx = mid_cx + ratio * (right_cx - mid_cx)

        # 绘制阴影 (Y轴也要加上 offset_y)
        shadow_offset = logic_height * 0.05
        painter.setBrush(QBrush(QColor(0, 0, 0, 60)))
        painter.drawEllipse(QRectF(cur_cx - handle_radius,
                                   center_y - handle_radius + shadow_offset,
                                   handle_radius * 2, handle_radius * 2))

        # 绘制实体
        painter.setBrush(QBrush(self._handle_color))
        painter.drawEllipse(QRectF(cur_cx - handle_radius,
                                   center_y - handle_radius,
                                   handle_radius * 2, handle_radius * 2))


# # --- 测试窗口 ---
# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("修复宽高比的三段开关")
#         self.resize(500, 400)
#
#         layout = QVBoxLayout(self)
#
#         label = QLabel("随意拖动窗口大小\n开关永远保持 3:1 比例\n即使布局空间是正方形，开关也不会变形")
#         label.setAlignment(Qt.AlignCenter)
#         layout.addWidget(label)
#
#         # 模拟一个可能把控件拉坏的布局环境
#         container = QWidget()
#         container.setStyleSheet("background-color: #EEE; border: 1px dashed gray;")
#         v_layout = QVBoxLayout(container)
#
#         self.switch_btn = TriStateSwitch()
#         v_layout.addWidget(self.switch_btn)
#
#         # 添加一个弹簧，测试拉伸
#         v_layout.addStretch()
#
#         layout.addWidget(container)
#
#         # 绑定测试
#         self.switch_btn.valueChanged.connect(lambda v: print(f"Mode: {v}"))
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
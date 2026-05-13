from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSpinBox


# class DoubleSlider(QWidget):
#     """双端滑块组件"""
#     rangeChanged = pyqtSignal(int, int)
#
#     def __init__(self, minimum, maximum):
#         super().__init__()
#         self.min_val = minimum
#         self.max_val = maximum
#         self.low = minimum
#         self.high = maximum
#         self.setMinimumHeight(40)
#         self.active_handle = None
#
#     def setValues(self, low, high):
#         self.low = max(self.min_val, min(low, self.high))
#         self.high = min(self.max_val, max(high, self.low))
#         self.update()
#
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.Antialiasing)
#         w, h = self.width(), self.height()
#         cy = h // 2
#
#         painter.setPen(Qt.NoPen)
#         painter.setBrush(QColor(220, 220, 220))
#         painter.drawRoundedRect(10, cy - 3, w - 20, 6, 3, 3)
#
#         x1, x2 = self._val_to_x(self.low), self._val_to_x(self.high)
#         painter.setBrush(QColor(43, 120, 228))
#         painter.drawRoundedRect(int(x1), cy - 3, int(x2 - x1), 6, 3, 3)
#
#         painter.setPen(QPen(QColor(150, 150, 150), 1))
#         painter.setBrush(QColor(255, 255, 255))
#         painter.drawEllipse(int(x1) - 8, cy - 8, 16, 16)
#         painter.drawEllipse(int(x2) - 8, cy - 8, 16, 16)
#
#         painter.setPen(QColor(50, 50, 50))
#         painter.drawText(int(x1) - 10, cy + 20, str(int(self.low)))
#         painter.drawText(int(x2) - 10, cy + 20, str(int(self.high)))
#
#     def _val_to_x(self, val):
#         span = self.max_val - self.min_val
#         return 10 + (val - self.min_val) / span * (self.width() - 20) if span > 0 else 10
#
#     def _x_to_val(self, x):
#         span = self.max_val - self.min_val
#         val = self.min_val + (x - 10) / (self.width() - 20) * span
#         return max(self.min_val, min(self.max_val, round(val)))
#
#     def mousePressEvent(self, ev):
#         if ev.button() == Qt.LeftButton:
#             val = self._x_to_val(ev.pos().x())
#             self.active_handle = 'low' if abs(val - self.low) < abs(val - self.high) else 'high'
#
#     def mouseMoveEvent(self, ev):
#         if self.active_handle:
#             val = self._x_to_val(ev.pos().x())
#             if self.active_handle == 'low' and val <= self.high:
#                 self.low = val
#             elif self.active_handle == 'high' and val >= self.low:
#                 self.high = val
#             self.update()
#             self.rangeChanged.emit(int(self.low), int(self.high))
#
#     def mouseReleaseEvent(self, ev):
#         self.active_handle = None


class DoubleSlider(QWidget):
    rangeChanged = pyqtSignal(int, int)

    def __init__(self, minimum, maximum):
        super().__init__()
        self.min_val = minimum
        self.max_val = maximum
        self.low = minimum
        self.high = maximum
        self.active_handle = None
        self._is_updating = False

        self.setMinimumHeight(40)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 左侧起点输入框
        self.spin_low = QSpinBox()
        self.spin_low.setRange(minimum, maximum)
        self.spin_low.setValue(minimum)
        self.spin_low.setKeyboardTracking(False)

        # 右侧终点输入框
        self.spin_high = QSpinBox()
        self.spin_high.setRange(minimum, maximum)
        self.spin_high.setValue(maximum)
        self.spin_high.setKeyboardTracking(False)

        # 布局：左框 + 弹性留白(给画笔画滑块用) + 右框
        self.layout.addWidget(self.spin_low)
        self.layout.addStretch(1)
        self.layout.addWidget(self.spin_high)

        self.spin_low.valueChanged.connect(self._on_spin_changed)
        self.spin_high.valueChanged.connect(self._on_spin_changed)

    def setValues(self, low, high):
        self._is_updating = True
        self.low = max(self.min_val, min(low, self.high))
        self.high = min(self.max_val, max(high, self.low))
        self.spin_low.setValue(int(self.low))
        self.spin_high.setValue(int(self.high))
        self._is_updating = False
        self.update()

    def _on_spin_changed(self):
        if self._is_updating: return
        low, high = self.spin_low.value(), self.spin_high.value()
        # 约束：左边不能大于右边
        if low > high:
            if self.sender() == self.spin_low:
                high = low
                self.spin_high.setValue(high)
            else:
                low = high
                self.spin_low.setValue(low)

        self.low, self.high = low, high
        self.update()
        self.rangeChanged.emit(self.low, self.high)

    # ============ 核心计算与绘制 ============
    def _get_slider_rect(self):
        # 动态计算左右两个输入框中间的留白区域边界
        left = self.spin_low.geometry().right() + 12
        right = self.spin_high.geometry().left() - 12
        return left, right

    def _val_to_x(self, val):
        left, right = self._get_slider_rect()
        span = self.max_val - self.min_val
        return left + (val - self.min_val) / span * (right - left) if span > 0 else left

    def _x_to_val(self, x):
        left, right = self._get_slider_rect()
        span = self.max_val - self.min_val
        if right <= left: return self.min_val
        val = self.min_val + (x - left) / (right - left) * span
        return max(self.min_val, min(self.max_val, round(val)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        left, right = self._get_slider_rect()
        if right <= left: return  # 空间不足不绘制

        cy = self.height() // 2

        # 1. 画灰色底槽
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(220, 220, 220))
        painter.drawRoundedRect(left, cy - 3, right - left, 6, 3, 3)

        # 2. 画蓝色激活槽
        x1, x2 = self._val_to_x(self.low), self._val_to_x(self.high)
        painter.setBrush(QColor(43, 120, 228))
        painter.drawRoundedRect(int(x1), cy - 3, int(x2 - x1), 6, 3, 3)

        # 3. 画圆形手柄
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(int(x1) - 8, cy - 8, 16, 16)
        painter.drawEllipse(int(x2) - 8, cy - 8, 16, 16)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            val = self._x_to_val(ev.pos().x())
            self.active_handle = 'low' if abs(val - self.low) < abs(val - self.high) else 'high'

    def mouseMoveEvent(self, ev):
        if self.active_handle:
            val = self._x_to_val(ev.pos().x())
            if self.active_handle == 'low' and val <= self.high:
                self.low = val
            elif self.active_handle == 'high' and val >= self.low:
                self.high = val

            self._is_updating = True
            self.spin_low.setValue(int(self.low))
            self.spin_high.setValue(int(self.high))
            self._is_updating = False

            self.update()
            self.rangeChanged.emit(int(self.low), int(self.high))

    def mouseReleaseEvent(self, ev):
        self.active_handle = None
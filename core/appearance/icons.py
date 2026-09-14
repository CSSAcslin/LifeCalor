from __future__ import annotations

from PyQt5.QtCore import QByteArray, Qt
from PyQt5.QtGui import QIcon, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer


_CONTROL_SHAPES = {
    "minimize": '<path d="M28 74 H92"/>',
    "maximize": '<rect x="29" y="29" width="62" height="62" rx="3"/>',
    "restore": (
        '<path d="M38 44 V27 H92 V81 H75"/>'
        '<rect x="27" y="39" width="54" height="54" rx="3"/>'
    ),
    "close": '<path d="M31 31 L89 89 M89 31 L31 89"/>',
    "sun": (
        '<circle cx="60" cy="60" r="21"/>'
        '<path d="M60 14 V25 M60 95 V106 M14 60 H25 M95 60 H106 '
        'M27 27 L35 35 M85 85 L93 93 M93 27 L85 35 M35 85 L27 93" fill="none"/>'
    ),
    "moon": '<path d="M78 23 A40 40 0 1 0 98 82 A34 34 0 0 1 78 23 Z"/>',
}


def window_control_icon(role, tokens, size=18):
    """Render compact window controls in the existing mint-outline icon language."""
    shape = _CONTROL_SHAPES[role]
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">
      <g fill="{tokens['selection']}" stroke="{tokens['text']}"
         stroke-width="8" stroke-linecap="round" stroke-linejoin="round">
        {shape}
      </g>
    </svg>
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(painter)
    painter.end()
    return QIcon(pixmap)

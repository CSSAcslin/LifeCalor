from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from .service import FrameRenderService
from .renderer import FrameRenderParams


class _SimpleSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self._callbacks):
            callback(*args)


class FrameRenderWorker(QObject):
    rendered = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, service: FrameRenderService | None = None, cache_capacity: int = 12):
        super().__init__()
        self.service = service or FrameRenderService(cache_capacity=cache_capacity)
        if self.rendered is None:
            self.rendered = _SimpleSignal()
        if self.failed is None:
            self.failed = _SimpleSignal()

    @pyqtSlot(int, object, int, object)
    def render(self, request_id: int, source, frame_index: int, params: FrameRenderParams | None = None) -> None:
        try:
            rendered = self.service.render_source(source, frame_index, params or FrameRenderParams())
        except Exception as exc:
            self.failed.emit(request_id, str(exc))
            return
        self.rendered.emit(request_id, rendered)

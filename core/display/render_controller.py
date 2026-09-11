import logging

from PyQt5.QtCore import QObject, QThread, pyqtSlot

from .worker import FrameRenderWorker


class RenderRequestState:
    def __init__(self):
        self.render_in_flight = False
        self.pending_frame_index = None
        self.frame_render_request_id = 0
        self.latest_frame_render_request_id = 0

    def start_or_queue(self, frame_index):
        if self.render_in_flight:
            self.pending_frame_index = frame_index
            return None
        self.render_in_flight = True
        return frame_index

    def next_request_id(self):
        self.frame_render_request_id += 1
        self.latest_frame_render_request_id = self.frame_render_request_id
        return self.frame_render_request_id

    def is_stale(self, request_id):
        return request_id != self.latest_frame_render_request_id

    def complete_current(self):
        self.render_in_flight = False

    def reset(self):
        self.pending_frame_index = None
        self.render_in_flight = False

    def pop_pending(self):
        pending = self.pending_frame_index
        self.pending_frame_index = None
        return pending


class RenderController(QObject):
    """Own one canvas renderer and retire it without blocking the GUI thread."""

    def __init__(self, canvas, cache_capacity=12):
        super().__init__(canvas)
        self.canvas = canvas
        self.state = RenderRequestState()
        self.cache_capacity = cache_capacity
        self._retiring = False
        self._retirement_finished = False
        self._retire_callbacks = []

    @property
    def is_retiring(self):
        return self._retiring

    def start_worker(self):
        canvas = self.canvas
        canvas.frame_render_thread = QThread(canvas)
        canvas.frame_render_worker = FrameRenderWorker(cache_capacity=self.cache_capacity)
        canvas.frame_render_worker.moveToThread(canvas.frame_render_thread)
        canvas.frame_render_requested.connect(canvas.frame_render_worker.render)
        canvas.frame_render_worker.rendered.connect(self.on_rendered)
        canvas.frame_render_worker.failed.connect(self.on_failed)
        canvas.frame_render_thread.finished.connect(canvas.frame_render_worker.deleteLater)
        canvas.frame_render_thread.start()

    def stop_worker(self, wait_ms=None, on_finished=None):
        """Request retirement and return immediately; wait_ms is retained for API compatibility."""
        if self._retirement_finished:
            if on_finished is not None:
                on_finished()
            return True
        if on_finished is not None:
            self._retire_callbacks.append(on_finished)
        if self._retiring:
            return False

        self._retiring = True
        canvas = self.canvas
        self.state.reset()
        if hasattr(canvas, "frame_render_requested") and hasattr(canvas, "frame_render_worker"):
            canvas._safe_disconnect(canvas.frame_render_requested, canvas.frame_render_worker.render)
            canvas._safe_disconnect(canvas.frame_render_worker.rendered, self.on_rendered)
            canvas._safe_disconnect(canvas.frame_render_worker.failed, self.on_failed)

        thread = getattr(canvas, "frame_render_thread", None)
        if thread is None or not thread.isRunning():
            self._finish_retirement()
            return True

        thread.finished.connect(self._finish_retirement)
        thread.requestInterruption()
        thread.quit()
        return False

    @pyqtSlot()
    def _finish_retirement(self):
        if self._retirement_finished:
            return
        self._retirement_finished = True
        callbacks = self._retire_callbacks
        self._retire_callbacks = []
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logging.exception("画布渲染线程退役回调失败")

    def request_frame_render(self, idx):
        canvas = self.canvas
        if getattr(canvas, "_is_closing", False) or self._retiring:
            return
        frame_index = self.state.start_or_queue(idx)
        if frame_index is None:
            logging.debug("Coalesced frame render request on canvas %s: pending frame %s", canvas.id, idx)
            return
        self._start_frame_render(frame_index)

    def _start_frame_render(self, idx):
        canvas = self.canvas
        if getattr(canvas, "_is_closing", False) or self._retiring:
            return
        frame_index = idx if canvas.data.is_temporary else 0
        canvas.set_render_status("rendering", f"画布 {canvas.id} 渲染第 {idx + 1} 帧")
        request_id = self.state.next_request_id()
        logging.debug("Start frame render canvas=%s request=%s frame=%s", canvas.id, request_id, frame_index)
        canvas.frame_render_requested.emit(
            request_id,
            canvas.data.display_source,
            frame_index,
            canvas.render_params_for_display(),
        )

    def start_pending_frame_render(self):
        canvas = self.canvas
        if getattr(canvas, "_is_closing", False) or self._retiring:
            self.state.pending_frame_index = None
            return
        pending_idx = self.state.pop_pending()
        if pending_idx is not None:
            self._start_frame_render(pending_idx)

    def on_rendered(self, request_id, rendered):
        canvas = self.canvas
        if getattr(canvas, "_is_closing", False) or self._retiring:
            return
        self.state.complete_current()
        if self.state.is_stale(request_id):
            logging.debug("Discard stale rendered frame canvas=%s request=%s", canvas.id, request_id)
            self.start_pending_frame_render()
            return
        canvas.update_display(rendered.image)
        canvas.set_render_status("completed", f"画布 {canvas.id} 渲染完成")
        self.start_pending_frame_render()

    def on_failed(self, request_id, message):
        canvas = self.canvas
        if getattr(canvas, "_is_closing", False) or self._retiring:
            return
        self.state.complete_current()
        if self.state.is_stale(request_id):
            logging.debug(
                "Discard stale render failure canvas=%s request=%s: %s",
                canvas.id,
                request_id,
                message,
            )
            self.start_pending_frame_render()
            return
        logging.error("Frame render failed on canvas %s: %s", canvas.id, message)
        canvas.set_render_status("failed", f"画布 {canvas.id} 渲染失败: {message}")
        self.start_pending_frame_render()

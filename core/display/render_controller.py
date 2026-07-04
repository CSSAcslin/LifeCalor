import logging

from PyQt5.QtCore import QThread

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


class RenderController:
    def __init__(self, canvas, cache_capacity=12):
        self.canvas = canvas
        self.state = RenderRequestState()
        self.cache_capacity = cache_capacity

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

    def stop_worker(self, wait_ms=5000):
        canvas = self.canvas
        self.state.reset()
        if hasattr(canvas, 'frame_render_requested') and hasattr(canvas, 'frame_render_worker'):
            canvas._safe_disconnect(canvas.frame_render_requested, canvas.frame_render_worker.render)
            canvas._safe_disconnect(canvas.frame_render_worker.rendered, self.on_rendered)
            canvas._safe_disconnect(canvas.frame_render_worker.failed, self.on_failed)
        if hasattr(canvas, 'frame_render_thread') and canvas.frame_render_thread.isRunning():
            canvas.frame_render_thread.quit()
            if not canvas.frame_render_thread.wait(wait_ms):
                logging.warning(f'Frame render thread did not stop within {wait_ms} ms on canvas {canvas.id}')

    def request_frame_render(self, idx):
        canvas = self.canvas
        if getattr(canvas, '_is_closing', False):
            return
        frame_index = self.state.start_or_queue(idx)
        if frame_index is None:
            logging.debug(f'Coalesced frame render request on canvas {canvas.id}: pending frame {idx}')
            return
        self._start_frame_render(frame_index)

    def _start_frame_render(self, idx):
        canvas = self.canvas
        if getattr(canvas, '_is_closing', False):
            return
        frame_index = idx if canvas.data.is_temporary else 0
        canvas.set_render_status('rendering', f'画布 {canvas.id} 渲染第 {idx + 1} 帧')
        request_id = self.state.next_request_id()
        logging.debug(f'Start frame render canvas={canvas.id} request={request_id} frame={frame_index}')
        canvas.frame_render_requested.emit(
            request_id,
            canvas.data.display_source,
            frame_index,
            canvas.render_params_for_display(),
        )

    def start_pending_frame_render(self):
        canvas = self.canvas
        if getattr(canvas, '_is_closing', False):
            self.state.pending_frame_index = None
            return
        pending_idx = self.state.pop_pending()
        if pending_idx is not None:
            self._start_frame_render(pending_idx)

    def on_rendered(self, request_id, rendered):
        canvas = self.canvas
        if getattr(canvas, '_is_closing', False):
            return
        self.state.complete_current()
        if self.state.is_stale(request_id):
            logging.debug(f'Discard stale rendered frame canvas={canvas.id} request={request_id}')
            self.start_pending_frame_render()
            return
        canvas.update_display(rendered.image)
        canvas.set_render_status('completed', f'画布 {canvas.id} 渲染完成')
        self.start_pending_frame_render()

    def on_failed(self, request_id, message):
        canvas = self.canvas
        if getattr(canvas, '_is_closing', False):
            return
        self.state.complete_current()
        if self.state.is_stale(request_id):
            logging.debug(f'Discard stale render failure canvas={canvas.id} request={request_id}: {message}')
            self.start_pending_frame_render()
            return
        logging.error(f'Frame render failed on canvas {canvas.id}: {message}')
        canvas.set_render_status('failed', f'画布 {canvas.id} 渲染失败: {message}')
        self.start_pending_frame_render()

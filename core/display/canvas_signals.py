def disconnect_canvas_signal(signal, slot):
    try:
        signal.disconnect(slot)
    except (TypeError, RuntimeError):
        return


class CanvasSignalBinder:
    def __init__(self, window):
        self.window = window

    def rebind_all(self):
        window = self.window
        window.roi_pick.clear()
        for canvas in window.image_display.display_canvas:
            self._rebind(canvas)
            window.roi_pick.addItem(canvas.windowTitle())

    def _rebind(self, canvas):
        window = self.window
        image_display = window.image_display
        proc_thread = window.proc_thread
        pairs = [
            (canvas.mouse_position_signal, window._handle_hover),
            (canvas.mouse_clicked_signal, window._handle_click),
            (canvas.current_canvas_signal, image_display.set_cursor_id),
            (canvas.draw_result_signal, window.draw_result),
            (canvas.get_fast_selection, proc_thread.get_fast_selection),
            (canvas.get_value_distribution, proc_thread.get_value_distribution),
            (canvas.sync_progress_signal, image_display.on_canvas_sync_progress),
            (canvas.sync_playback_signal, image_display.on_canvas_sync_playback),
        ]
        for signal, slot in pairs:
            disconnect_canvas_signal(signal, slot)
            signal.connect(slot)

# LifeCalor Display Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the display-layer refactor by moving canvas wiring, display data creation, and render lifecycle coordination out of `MainWindow.py` / large widget methods while preserving the current large-data behavior.

**Architecture:** Keep `MainWindow` as the UI entry and signal hub, but delegate display-specific work to small controllers under `core/display/`. Do not change algorithm outputs, `Data` / `ProcessedData` semantics, cache-on-disk policy, or the user-facing display workflow in this round.

**Tech Stack:** PyQt5, numpy, existing `display.source/renderer/cache/service/worker`, `unittest`, architecture tests based on source inspection plus pure unit tests for extracted policy classes.

---

## File Structure

- Create `core/display/status.py`: pure mapping for render status messages and status-bar behavior.
- Create `core/display/canvas_signals.py`: connect/disconnect one canvas to MainWindow slots without touching render-worker internal signals.
- Create `core/display/canvas_controller.py`: create, replace, append, and focus image canvases; wrap existing dialog/data-selection flow.
- Create `core/display/render_controller.py`: own per-canvas async render request state currently embedded in `SubImageDisplayWidget`.
- Modify `core/display/__init__.py`: export new display helpers without importing GUI-heavy controllers unless needed.
- Modify `core/MainWindow.py`: keep only thin wrappers for display actions, similar to `SelectionController`, `ExportController`, `HistoryController`.
- Modify `core/ImageDisplayWindow.py`: delegate render lifecycle to `RenderController`; keep widget drawing, scene, ROI, and UI controls local.
- Add or extend `tests/test_display_architecture.py`: lock boundaries so MainWindow and widgets do not grow back.
- Add `tests/test_display_status.py`, `tests/test_canvas_signals.py`, `tests/test_render_controller.py`: pure or mostly fake-object tests.

---

### Task 1: Extract Render Status Policy

**Files:**
- Create: `core/display/status.py`
- Modify: `core/MainWindow.py`
- Test: `tests/test_display_status.py`
- Test: `tests/test_display_architecture.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_display_status.py
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from display.status import render_status_update


class DisplayStatusTests(unittest.TestCase):
    def test_only_failed_render_status_updates_main_status(self):
        self.assertEqual(render_status_update("failed", "bad frame"), ("bad frame", "failed"))
        self.assertIsNone(render_status_update("rendering", "rendering frame"))
        self.assertIsNone(render_status_update("completed", "done"))


if __name__ == "__main__":
    unittest.main()
```

Add to `tests/test_display_architecture.py`:

```python
def test_mainwindow_delegates_render_status_policy(self):
    source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
    self.assertIn("from display.status import render_status_update", source)
    handler = source[source.index("def handle_render_status"):source.index("def update_status")]
    self.assertIn("render_status_update(status, message)", handler)
    self.assertNotIn("status == 'failed'", handler)
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
H:\Newera\programing\.venv\Scripts\python.exe -m unittest tests.test_display_status tests.test_display_architecture -v
```

Expected: import or assertion failure because `display.status` does not exist and MainWindow still has inline status logic.

- [ ] **Step 3: Implement minimal policy**

```python
# core/display/status.py
def render_status_update(status: str, message: str):
    if status == "failed":
        return message, "failed"
    return None
```

Modify `core/MainWindow.py`:

```python
from display.status import render_status_update


def handle_render_status(self, status, message):
    update = render_status_update(status, message)
    if update is None:
        return
    text, state = update
    self.update_status(text, state)
```

- [ ] **Step 4: Verify and commit**

Run:

```bash
H:\Newera\programing\.venv\Scripts\python.exe -m unittest tests.test_display_status tests.test_display_architecture -v
H:\Newera\programing\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Commit:

```bash
git add core/display/status.py core/MainWindow.py tests/test_display_status.py tests/test_display_architecture.py
git commit -m "refactor: extract display status policy"
```

---

### Task 2: Extract Canvas Signal Wiring

**Files:**
- Create: `core/display/canvas_signals.py`
- Modify: `core/MainWindow.py`
- Test: `tests/test_canvas_signals.py`
- Test: `tests/test_display_architecture.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_canvas_signals.py
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from display.canvas_signals import disconnect_canvas_signal


class FakeSignal:
    def __init__(self):
        self.disconnected = False
    def disconnect(self, slot):
        self.disconnected = True


class CanvasSignalTests(unittest.TestCase):
    def test_disconnect_ignores_unconnected_signal(self):
        class BrokenSignal:
            def disconnect(self, slot):
                raise TypeError("not connected")
        disconnect_canvas_signal(BrokenSignal(), object())

    def test_disconnect_calls_signal_disconnect(self):
        signal = FakeSignal()
        slot = object()
        disconnect_canvas_signal(signal, slot)
        self.assertTrue(signal.disconnected)


if __name__ == "__main__":
    unittest.main()
```

Add to `tests/test_display_architecture.py`:

```python
def test_mainwindow_delegates_canvas_signal_wiring(self):
    source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
    self.assertIn("from display.canvas_signals import CanvasSignalBinder", source)
    self.assertIn("self.canvas_signal_binder = CanvasSignalBinder(self)", source)
    block = source[source.index("def canvas_signal_connect"):source.index("def add_new_canvas")]
    self.assertIn("self.canvas_signal_binder.rebind_all()", block)
    self.assertNotIn("canvas.mouse_position_signal.connect", block)
```

- [ ] **Step 2: Implement binder**

```python
# core/display/canvas_signals.py
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
            (canvas.sync_progress_signal, image_display.on_canvas_sync_progress),
            (canvas.sync_playback_signal, image_display.on_canvas_sync_playback),
        ]
        for signal, slot in pairs:
            disconnect_canvas_signal(signal, slot)
            signal.connect(slot)
```

Modify `MainWindow.__init__` after `self.history_controller`:

```python
self.canvas_signal_binder = CanvasSignalBinder(self)
```

Modify wrappers:

```python
def disconnect_canvas_signal(self, signal, slot):
    return disconnect_canvas_signal(signal, slot)


def canvas_signal_connect(self):
    return self.canvas_signal_binder.rebind_all()
```

- [ ] **Step 3: Verify and commit**

Run:

```bash
H:\Newera\programing\.venv\Scripts\python.exe -m unittest tests.test_canvas_signals tests.test_display_architecture -v
H:\Newera\programing\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Commit:

```bash
git add core/display/canvas_signals.py core/MainWindow.py tests/test_canvas_signals.py tests/test_display_architecture.py
git commit -m "refactor: extract canvas signal binder"
```

---

### Task 3: Extract MainWindow Canvas Creation Controller

**Files:**
- Create: `core/display/canvas_controller.py`
- Modify: `core/MainWindow.py`
- Test: `tests/test_display_architecture.py`

- [ ] **Step 1: Write failing architecture test**

```python
def test_mainwindow_delegates_canvas_creation_controller(self):
    source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
    self.assertIn("from display.canvas_controller import DisplayCanvasController", source)
    self.assertIn("self.display_canvas_controller = DisplayCanvasController(self)", source)
    add_block = source[source.index("def add_new_canvas"):source.index("def upgrade_and_imaging")]
    self.assertIn("self.display_canvas_controller.add_new_canvas(assign_data)", add_block)
    self.assertNotIn("DataViewAndSelectPop", add_block)
    self.assertNotIn("ImagingData.create_image", add_block)
    upgrade_block = source[source.index("def upgrade_and_imaging"):source.index("def _handle_hover")]
    self.assertIn("self.display_canvas_controller.upgrade_and_imaging", upgrade_block)
```

- [ ] **Step 2: Move existing logic with no behavior change**

Create `core/display/canvas_controller.py` and move the body of current `MainWindow.add_new_canvas()` plus the data-display creation parts of `upgrade_and_imaging()` into methods on `DisplayCanvasController`. Use `self.window` for all existing dependencies.

Key signatures:

```python
class DisplayCanvasController:
    def __init__(self, window):
        self.window = window

    def add_new_canvas(self, assign_data=None):
        # Move existing MainWindow.add_new_canvas body here unchanged.
        # Replace self.foo with window.foo where needed.
        # End by calling window.canvas_signal_connect().

    def upgrade_and_imaging(self, origin_data=None):
        # Move current MainWindow upgrade/display refresh flow here unchanged.
```

Modify `MainWindow` wrappers:

```python
from display.canvas_controller import DisplayCanvasController

self.display_canvas_controller = DisplayCanvasController(self)


def add_new_canvas(self, assign_data=None):
    return self.display_canvas_controller.add_new_canvas(assign_data)


def upgrade_and_imaging(self, origin_data=None):
    return self.display_canvas_controller.upgrade_and_imaging(origin_data)
```

- [ ] **Step 3: Verify and commit**

Run:

```bash
H:\Newera\programing\.venv\Scripts\python.exe -m unittest tests.test_display_architecture tests.test_display_source tests.test_data_models -v
H:\Newera\programing\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Commit:

```bash
git add core/display/canvas_controller.py core/MainWindow.py tests/test_display_architecture.py
git commit -m "refactor: extract display canvas controller"
```

---

### Task 4: Extract Per-Canvas Render Lifecycle Controller

**Files:**
- Create: `core/display/render_controller.py`
- Modify: `core/ImageDisplayWindow.py`
- Test: `tests/test_render_controller.py`
- Test: `tests/test_display_architecture.py`

- [ ] **Step 1: Write behavior tests for coalescing and stale-result handling**

```python
# tests/test_render_controller.py
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from display.render_controller import RenderRequestState


class RenderControllerTests(unittest.TestCase):
    def test_busy_request_is_coalesced_to_latest_frame(self):
        state = RenderRequestState()
        first = state.start_or_queue(1)
        second = state.start_or_queue(2)
        third = state.start_or_queue(3)
        self.assertEqual(first, 1)
        self.assertIsNone(second)
        self.assertIsNone(third)
        self.assertEqual(state.pending_frame_index, 3)

    def test_stale_result_is_ignored(self):
        state = RenderRequestState()
        request_id = state.next_request_id()
        newer = state.next_request_id()
        self.assertTrue(state.is_stale(request_id))
        self.assertFalse(state.is_stale(newer))


if __name__ == "__main__":
    unittest.main()
```

Add architecture test:

```python
def test_subimage_widget_delegates_render_lifecycle(self):
    source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
    self.assertIn("from display.render_controller import RenderController", source)
    self.assertIn("self.render_controller = RenderController(self)", source)
    self.assertIn("self.render_controller.request_frame_render(idx)", source)
```

- [ ] **Step 2: Implement render controller around existing signals**

```python
# core/display/render_controller.py
import logging
from PyQt5.QtCore import QThread
from .renderer import FrameRenderParams
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

    def request_frame_render(self, idx):
        frame_index = self.state.start_or_queue(idx)
        if frame_index is None:
            logging.debug(f"Coalesced frame render request on canvas {self.canvas.id}: pending frame {idx}")
            return
        self._start_frame_render(frame_index)

    def _start_frame_render(self, idx):
        canvas = self.canvas
        frame_index = max(0, min(idx, canvas.data.frame_count - 1))
        canvas.set_render_status("rendering", f"画布 {canvas.id} 渲染第 {frame_index + 1} 帧")
        request_id = self.state.next_request_id()
        canvas.frame_render_requested.emit(
            request_id,
            canvas.data.display_source,
            frame_index,
            canvas.render_params_for_display(),
        )

    def on_rendered(self, request_id, rendered):
        canvas = self.canvas
        self.state.complete_current()
        if getattr(canvas, "_is_closing", False):
            return
        if self.state.is_stale(request_id):
            self._start_pending_frame_render()
            return
        canvas.update_display(rendered.image)
        canvas.set_render_status("completed", f"画布 {canvas.id} 渲染完成")
        self._start_pending_frame_render()

    def on_failed(self, request_id, message):
        canvas = self.canvas
        self.state.complete_current()
        if getattr(canvas, "_is_closing", False):
            return
        if self.state.is_stale(request_id):
            self._start_pending_frame_render()
            return
        logging.error(f"Frame render failed on canvas {canvas.id}: {message}")
        canvas.set_render_status("failed", f"画布 {canvas.id} 渲染失败: {message}")
        self._start_pending_frame_render()

    def _start_pending_frame_render(self):
        pending = self.state.pop_pending()
        if pending is not None:
            self.request_frame_render(pending)
```

Modify `SubImageDisplayWidget` to delegate `request_frame_render`, `on_frame_rendered`, `on_frame_render_failed`, and worker startup to `RenderController` while preserving existing public method names as wrappers during this round.

- [ ] **Step 3: Verify and commit**

Run:

```bash
H:\Newera\programing\.venv\Scripts\python.exe -m unittest tests.test_render_controller tests.test_display_architecture tests.test_frame_render_worker -v
H:\Newera\programing\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Commit:

```bash
git add core/display/render_controller.py core/ImageDisplayWindow.py tests/test_render_controller.py tests/test_display_architecture.py
git commit -m "refactor: extract canvas render controller"
```

---

### Task 5: Final Display Regression Guardrails

**Files:**
- Modify: `tests/test_display_architecture.py`
- Modify: `tests/test_mainwindow_architecture.py`
- Optional Modify: `readme.md` if current architecture notes are already present there.

- [ ] **Step 1: Add boundary tests**

```python
def test_mainwindow_display_methods_remain_thin(self):
    source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
    add_block = source[source.index("def add_new_canvas"):source.index("def upgrade_and_imaging")]
    self.assertLess(len(add_block.splitlines()), 8)
    self.assertNotIn("DataViewAndSelectPop", add_block)
    self.assertNotIn("ImagingData.create_image", add_block)


def test_image_display_render_methods_remain_wrappers(self):
    source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
    request_block = source[source.index("def request_frame_render"):source.index("def _start_frame_render")]
    self.assertIn("self.render_controller.request_frame_render(idx)", request_block)
    self.assertLess(len(request_block.splitlines()), 8)
```

- [ ] **Step 2: Run complete verification**

Run:

```bash
H:\Newera\programing\.venv\Scripts\python.exe -m py_compile core\MainWindow.py core\ImageDisplayWindow.py core\display\status.py core\display\canvas_signals.py core\display\canvas_controller.py core\display\render_controller.py
H:\Newera\programing\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass, no whitespace warnings.

- [ ] **Step 3: Manual smoke checklist with real data**

Run the App manually and verify:

```text
1. Start app.
2. Import small TIFF/AVI data.
3. First frame appears without moving mouse.
4. Slider changes frame.
5. Pseudocolor toggles and updates current frame.
6. Add canvas from current data.
7. Add canvas from processed data/history.
8. Delete one canvas, then delete all canvases.
9. Import large data as first import.
10. Import large data as second import and choose add/replace canvas.
11. Confirm no stuck render status text; only failures appear in main status.
```

- [ ] **Step 4: Commit final guardrails**

```bash
git add tests/test_display_architecture.py tests/test_mainwindow_architecture.py readme.md
git commit -m "test: lock display refactor boundaries"
```

---

## Risk Controls

- Do not change `Data`, `ProcessedData`, `ArrayCache`, STFT/CWT, or lifetime calculation in this round.
- Do not change the cache threshold or disk-cache format.
- Keep public methods such as `add_new_canvas`, `canvas_signal_connect`, `request_frame_render`, and `display_image` as wrappers so existing signal connections keep working.
- Commit after each task; if real-data testing finds a regression, revert the smallest display-specific commit.
- Any crash during real-data import/display should first be isolated to canvas creation, signal binding, or render lifecycle based on the task boundary.

## Completion Criteria

- `MainWindow.py` loses another substantial chunk of display/canvas orchestration code.
- `ImageDisplayWindow.py` still owns drawing UI and scene items, but no longer owns the render request state machine inline.
- Full test suite passes.
- Real large-data smoke test remains as good as the current working behavior: first import, second import, frame slider, pseudocolor, add/replace canvas all work.

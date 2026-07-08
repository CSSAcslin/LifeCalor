# LifeCalor 1.0.5 Error Handling And Anchor Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Standardize user-facing error reporting for the most failure-prone paths and add anchor ROI value distribution statistics for the current displayed frame.

**Architecture:** Add a lightweight diagnostics module for structured logging and QMessageBox display, then wire it into MainWindow/DataProcessor/canvas paths. Add a reusable DataProcessor value-distribution function and route the anchor quick-extract option through PlotGraphWidget's existing histogram-capable graph area.

**Tech Stack:** PyQt5, NumPy, pyqtgraph, unittest/pytest-compatible tests.

---

### Task 1: Diagnostics Foundation

**Files:**
- Create: `core/diagnostics/__init__.py`
- Create: `core/diagnostics/reporter.py`
- Test: `tests/test_error_reporting.py`

- [x] Create an `AppError` dataclass with title, message, stage, severity, details, and optional original exception.
- [x] Add `format_exception_details(exc, stage=None, data=None)` so logs include exception type, data name, shape, dtype, and traceback.
- [x] Add `show_app_error(parent, error)` and `report_exception(parent, title, message, exc, stage=None, data=None, severity="critical")`.
- [x] Test formatting without requiring a real GUI dialog.

### Task 2: Route Existing Error Hotspots

**Files:**
- Modify: `core/MainWindow.py`
- Modify: `core/history/controller.py`
- Modify: `core/ImageDisplayWindow.py`
- Modify: `core/DataProcessor.py`
- Test: `tests/test_error_reporting.py`

- [x] Replace `MainWindow.processed_result` ad-hoc computation error popup with `show_app_error`.
- [x] Replace history cache critical failure popups with `report_exception` or `show_app_error`.
- [x] Replace anchor quick-extract failures in `DataProcessor.get_fast_selection` with structured logging and a plot error signal payload.
- [x] Keep existing visible behavior where possible: warnings remain warnings, critical failures remain critical.

### Task 3: Anchor Value Distribution Algorithm

**Files:**
- Modify: `core/DataProcessor.py`
- Test: `tests/test_anchor_distribution.py`

- [x] Add `DataProcessor.value_distribution_from_frame(frame, mask, bins="auto")`.
- [x] Return a two-column histogram array `[bin_center, count]` and metadata including pixel count, min, max, mean, std, median, dtype, and bins.
- [x] Handle NaN/Inf by dropping non-finite values.
- [x] For complex arrays, compute distribution on magnitude and include `value_mode="abs_complex"` in metadata.
- [x] Raise readable `ValueError` for empty ROI, mismatched mask shape, or non-2D frame.

### Task 4: Anchor UI And Signal Wiring

**Files:**
- Modify: `core/ImageDisplayWindow.py`
- Modify: `core/display/canvas_signals.py`
- Modify: `core/DataProcessor.py`
- Test: `tests/test_anchor_distribution.py`
- Test: `tests/test_display_architecture.py`

- [x] Add `value_distribution` to anchor quick-extract method options and Chinese label `值分布统计`.
- [x] Add a dedicated `get_value_distribution` signal carrying `(data, mask, current_time_idx, name)`.
- [x] Emit the new signal only for the distribution method; keep time-series quick extraction unchanged.
- [x] Connect the new signal in `CanvasSignalBinder`.
- [x] In DataProcessor, use `data.display_source.get_frame(frame_idx)` when available, otherwise fall back to `data.image_backup[frame_idx]`.

### Task 5: PlotGraph Histogram Integration

**Files:**
- Modify: `core/PlotGraphWidget.py`
- Test: `tests/test_anchor_distribution.py`

- [x] Allow plot calls to pass `analysis_mode="hist"` and set the graph to histogram mode before plotting.
- [x] Preserve existing histogram mode behavior for normal time-series curves.
- [x] Add labels for value/count when plotting a precomputed distribution.

### Task 6: Version, Log, Verification, Commit

**Files:**
- Modify: version location used by the app
- Modify: `updatinglog.md`

- [x] Bump patch version to `1.0.5`.
- [x] Add a 1.0.5 updatinglog entry covering error handling and anchor value distribution.
- [x] Run focused tests for diagnostics, distribution, display architecture.
- [x] Run the full existing test suite.
- [x] Commit only relevant files, excluding data/spec/temp/unrelated files.

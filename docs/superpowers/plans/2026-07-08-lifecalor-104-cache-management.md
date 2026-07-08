# LifeCalor 1.0.4 Cache Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make history/cache management mature enough for long-term real-data use by improving text quality, cache-directory switching, cancellable cache restore, selected item deletion, and sortable cache tables.

**Architecture:** Keep `MainWindow` as a thin entry point and put cache/history behavior in `core/history/controller.py`, UI controls in `core/history/dialog.py`, and manifest helpers in `core/history/manifest.py`. Keep array loading in `DataManager.ArrayLoadWorker` with cooperative cancellation between array loads.

**Tech Stack:** PyQt5, Python unittest, existing ArrayCache/DataManager/history modules.

---

### Task 1: Add Regression Tests

**Files:**
- Modify: `tests/test_history_cache_manager.py`
- Modify: `tests/test_cache_settings_architecture.py`

- [ ] Add tests/assertions for manifest deletion, current history deletion, cancellable load architecture, sortable tree items, cache directory switch notification, and visible refresh behavior.
- [ ] Run `python -m unittest tests.test_history_cache_manager tests.test_cache_settings_architecture -v` and confirm new tests fail before implementation.

### Task 2: Manifest and Cache Directory Helpers

**Files:**
- Modify: `core/history/manifest.py`
- Modify: `core/history/controller.py`

- [ ] Add helpers to remove a manifest item by id and delete only exclusive cache files for that item.
- [ ] Add a cache directory switch handler that applies the new directory without migrating old cache files and informs the user that old caches remain in their original folder.
- [ ] Run history cache tests.

### Task 3: Cancellable Cache Restore

**Files:**
- Modify: `core/DataManager.py`
- Modify: `core/history/controller.py`
- Modify: `core/history/dialog.py`

- [ ] Add cooperative cancellation to `ArrayLoadWorker` and `materialize_cached_arrays`.
- [ ] Add a cancel signal/button in history cache dialog and route it through `HistoryController.cancel_cached_history_load`.
- [ ] Ensure cancellation exits cleanly without setting current data focus.
- [ ] Run cache settings and history cache tests.

### Task 4: Delete Selected History Items

**Files:**
- Modify: `core/history/dialog.py`
- Modify: `core/history/controller.py`
- Modify: `core/history/manifest.py`

- [ ] Add delete buttons for current history and recoverable history.
- [ ] Current history deletion removes only in-memory history by default.
- [ ] Recoverable history deletion removes manifest item and optionally deletes exclusive cache files.
- [ ] Refresh lists after every delete.
- [ ] Run history cache tests.

### Task 5: Sorting and Chinese Text Cleanup

**Files:**
- Modify: `core/history/dialog.py`
- Modify: `core/ArrayCache.py`
- Modify: `core/DataManager.py`
- Modify: `core/history/controller.py`

- [ ] Add sortable tree item support with numeric sort values for byte/timestamp/status columns.
- [ ] Enable sorting on current and recoverable history trees.
- [ ] Replace obvious replacement-character乱码 in cache/history user-facing strings and logs.
- [ ] Run a script/test check for replacement characters in cache/history files.

### Task 6: Version, Log, Verification, Commit

**Files:**
- Modify: `core/MainWindow.py`
- Modify: `updatinglog.md`
- Modify: tests as needed

- [ ] Bump version to `1.0.4`.
- [ ] Add `1.0.4` changelog entry.
- [ ] Run `python -m py_compile core\MainWindow.py core\DataManager.py core\history\manifest.py core\history\dialog.py core\history\controller.py`.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Commit only relevant files with pathspecs; exclude unrelated `data/`, spec files, and scripts.

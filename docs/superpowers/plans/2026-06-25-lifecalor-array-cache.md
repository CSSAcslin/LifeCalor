# LifeCalor Data Cache Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace deep-copied large arrays in LifeCalor history with configurable `.npy` cache references while preserving free historical data selection and minimizing algorithm changes.

**Architecture:** Keep `Data` and `ProcessedData` as the public objects used by the rest of the app. Add a focused cache layer that stores large ndarrays as `.npy` files and exposes them through array-like references resolved at data-object boundaries. Small arrays and metadata stay in memory. Large primary arrays (`Data.data_origin`, `Data.image_import`, `ProcessedData.data_processed`) and large secondary arrays inside `out_processed` all become `ArrayRef` references in historical snapshots. Primary arrays are the highest priority because they are usually the largest data in the application.

**Tech Stack:** Python stdlib, NumPy `.npy`/`np.load(..., mmap_mode='r+')`, PyQt5 `QSettings` and existing progress/status signals, unittest.

---

## Non-Negotiable Data Scope

This migration must cover both primary data arrays and secondary result arrays:

- `Data.data_origin`: cache if `nbytes` exceeds the configured threshold.
- `Data.image_import`: cache if `nbytes` exceeds the configured threshold.
- `ProcessedData.data_processed`: cache if `nbytes` exceeds the configured threshold.
- `ProcessedData.out_processed`: cache individual ndarray values if their `nbytes` exceeds the configured threshold.
- Small metadata values and small arrays stay in memory.

The implementation is incomplete if only `out_processed` is cached.

## File Structure

- Create `core/ArrayCache.py`: `ArrayRef`, `ArrayCacheConfig`, `ArrayStore`, `array_nbytes`, `should_cache_array`, `resolve_array`, `cache_large_arrays_in_mapping`.
- Create `tests/test_array_cache.py`: pure unit tests for dtype-aware byte estimation, configurable threshold, `.npy` save/load, memmap resolution, and `out_processed` mapping caching.
- Modify `core/DataManager.py`: integrate cache references into `Data` and `ProcessedData`, replace history `deepcopy` with cache-aware snapshots, resolve arrays in properties, avoid persisting `unfolded_data` as a permanent large history array.
- Modify `core/DataProcessor.py`: replace long-lived `out_processed['unfolded_data']` storage with `get_unfolded_data(data)` helper or local temporary view; preserve temporary reshape for STFT/CWT speed.
- Modify `core/MainWindow.py`: add cache settings actions, expose cache directory and threshold settings, wire progress/status messages for large cache writes/loads.
- Modify `readme.md`: document cache directory, 512MB default threshold, and `.npy` cache behavior.
- Add tests to existing `tests/test_data_models.py` or new `tests/test_data_cache_integration.py`: verify historical free selection still returns usable `Data`/`ProcessedData` objects.

---

### Task 1: Array Cache Foundation

**Files:**
- Create: `core/ArrayCache.py`
- Create: `tests/test_array_cache.py`

- [ ] **Step 1: Write failing tests for dtype-aware size and threshold**

```python
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from ArrayCache import array_nbytes, should_cache_array, ArrayCacheConfig


class ArrayCacheTests(unittest.TestCase):
    def test_array_nbytes_respects_dtype(self):
        self.assertEqual(array_nbytes(np.zeros((4,), dtype=np.uint8)), 4)
        self.assertEqual(array_nbytes(np.zeros((4,), dtype=np.float32)), 16)
        self.assertEqual(array_nbytes(np.zeros((4,), dtype=np.complex128)), 64)

    def test_should_cache_array_uses_configurable_threshold(self):
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=16)
        self.assertFalse(should_cache_array(np.zeros((4,), dtype=np.float32), config))
        self.assertTrue(should_cache_array(np.zeros((5,), dtype=np.float32), config))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest discover -s tests -p "test_array_cache.py" -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'ArrayCache'`.

- [ ] **Step 3: Implement minimal `ArrayCacheConfig`, `array_nbytes`, and `should_cache_array`**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CACHE_THRESHOLD_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ArrayCacheConfig:
    cache_dir: Path
    threshold_bytes: int = DEFAULT_CACHE_THRESHOLD_BYTES

    def ensure_dir(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir


def array_nbytes(value: Any) -> int:
    return int(value.nbytes) if isinstance(value, np.ndarray) else 0


def should_cache_array(value: Any, config: ArrayCacheConfig) -> bool:
    return isinstance(value, np.ndarray) and array_nbytes(value) > int(config.threshold_bytes)
```

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest discover -s tests -p "test_array_cache.py" -v`

Expected: PASS.

---

### Task 2: ArrayRef `.npy` Persistence

**Files:**
- Modify: `core/ArrayCache.py`
- Modify: `tests/test_array_cache.py`

- [ ] **Step 1: Add failing tests for `.npy` write, memmap load, and metadata**

```python
from ArrayCache import ArrayStore, ArrayRef, resolve_array

    def test_store_writes_npy_and_resolves_memmap(self):
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=1)
        store = ArrayStore(config)
        source = np.arange(12, dtype=np.float32).reshape(3, 4)

        ref = store.put_array(source, owner_id="abc", field_name="data_processed")
        loaded = resolve_array(ref, mmap_mode="r+")

        self.assertIsInstance(ref, ArrayRef)
        self.assertEqual(ref.shape, (3, 4))
        self.assertEqual(ref.dtype, "float32")
        self.assertEqual(ref.nbytes, source.nbytes)
        self.assertTrue(ref.path.exists())
        self.assertTrue(isinstance(loaded, np.memmap))
        np.testing.assert_array_equal(loaded, source)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m unittest discover -s tests -p "test_array_cache.py" -v`

Expected: FAIL because `ArrayStore`, `ArrayRef`, or `resolve_array` is missing.

- [ ] **Step 3: Implement `ArrayRef`, `ArrayStore`, and `resolve_array`**

```python
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass(frozen=True)
class ArrayRef:
    path: Path
    shape: tuple
    dtype: str
    nbytes: int
    created_at: float
    field_name: str

    def load(self, mmap_mode: str = "r+"):
        return np.load(self.path, mmap_mode=mmap_mode)


class ArrayStore:
    def __init__(self, config: ArrayCacheConfig, progress_callback: ProgressCallback = None):
        self.config = config
        self.progress_callback = progress_callback
        self.config.ensure_dir()

    def put_array(self, array: np.ndarray, owner_id: str, field_name: str) -> ArrayRef:
        self.config.ensure_dir()
        safe_field = str(field_name).replace("/", "_").replace("\\", "_")
        path = self.config.cache_dir / f"{owner_id}_{safe_field}_{uuid.uuid4().hex}.npy"
        if self.progress_callback:
            self.progress_callback(0, int(array.nbytes), f"正在写入缓存: {safe_field}")
        np.save(path, array)
        if self.progress_callback:
            self.progress_callback(int(array.nbytes), int(array.nbytes), f"缓存写入完成: {safe_field}")
        return ArrayRef(path=path, shape=tuple(array.shape), dtype=str(array.dtype), nbytes=int(array.nbytes), created_at=time.time(), field_name=safe_field)


def resolve_array(value, mmap_mode: str = "r+"):
    if isinstance(value, ArrayRef):
        return value.load(mmap_mode=mmap_mode)
    return value
```

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest discover -s tests -p "test_array_cache.py" -v`

Expected: PASS.

---

### Task 3: Cache Large Arrays Inside `out_processed`

**Files:**
- Modify: `core/ArrayCache.py`
- Modify: `tests/test_array_cache.py`

- [ ] **Step 1: Add failing test for mapping caching**

```python
from ArrayCache import cache_large_arrays_in_mapping

    def test_cache_large_arrays_in_mapping_keeps_small_metadata(self):
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=8)
        store = ArrayStore(config)
        mapping = {
            "fps": 360,
            "whole_mean": np.arange(2, dtype=np.float32),
            "unfolded_data": np.arange(12, dtype=np.float32).reshape(3, 4),
        }

        cached = cache_large_arrays_in_mapping(mapping, store, owner_id="abc")

        self.assertEqual(cached["fps"], 360)
        self.assertIsInstance(cached["whole_mean"], np.ndarray)
        self.assertIsInstance(cached["unfolded_data"], ArrayRef)
        np.testing.assert_array_equal(resolve_array(cached["unfolded_data"]), mapping["unfolded_data"])
```

- [ ] **Step 2: Run focused test to verify failure**

Run: `python -m unittest discover -s tests -p "test_array_cache.py" -v`

Expected: FAIL because `cache_large_arrays_in_mapping` is missing.

- [ ] **Step 3: Implement mapping helper**

```python
def cache_large_arrays_in_mapping(mapping: dict, store: ArrayStore, owner_id: str) -> dict:
    cached = {}
    for key, value in (mapping or {}).items():
        if should_cache_array(value, store.config):
            cached[key] = store.put_array(value, owner_id=owner_id, field_name=f"out_processed_{key}")
        else:
            cached[key] = value
    return cached
```

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest discover -s tests -p "test_array_cache.py" -v`

Expected: PASS.

---

### Task 4: Cache-Aware Data History Snapshots

**Files:**
- Modify: `core/DataManager.py`
- Create: `tests/test_data_cache_integration.py`

- [ ] **Step 1: Add failing integration tests for historical selection**

```python
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from ArrayCache import ArrayCacheConfig, ArrayRef
from DataManager import Data, ProcessedData, configure_array_cache


class DataCacheIntegrationTests(unittest.TestCase):
    def setUp(self):
        Data.clear_history()
        ProcessedData.clear_history()
        configure_array_cache(ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=8))

    def test_data_history_caches_large_data_origin_but_remains_array_accessible(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        data = Data(source, np.arange(3), "test", source.mean(axis=0))

        history_item = Data.history[-1]

        self.assertIsInstance(history_item._data_origin_storage, ArrayRef)
        np.testing.assert_array_equal(history_item.data_origin, source)
        np.testing.assert_array_equal(data.data_origin, source)

    def test_processed_history_caches_large_data_processed_and_out_processed_arrays(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        processed = ProcessedData(
            1.0,
            "processed",
            "ROI_stft",
            time_point=np.arange(3),
            data_processed=source,
            out_processed={"fps": 360, "large_extra": source.copy()},
        )

        history_item = ProcessedData.history[-1]

        self.assertIsInstance(history_item._data_processed_storage, ArrayRef)
        self.assertIsInstance(history_item.out_processed["large_extra"], ArrayRef)
        np.testing.assert_array_equal(history_item.data_processed, source)
        np.testing.assert_array_equal(history_item.out_processed_array("large_extra"), source)
```

- [ ] **Step 2: Run integration test to verify failure**

Run: `python -m unittest discover -s tests -p "test_data_cache_integration.py" -v`

Expected: FAIL because `configure_array_cache`, `_data_origin_storage`, `_data_processed_storage`, or `out_processed_array` is missing.

- [ ] **Step 3: Add cache configuration and storage fields to `DataManager.py`**

Add imports:

```python
from ArrayCache import (
    ArrayCacheConfig,
    ArrayStore,
    ArrayRef,
    cache_large_arrays_in_mapping,
    resolve_array,
    should_cache_array,
)
```

Add module-level config:

```python
_ARRAY_CACHE_CONFIG = ArrayCacheConfig(cache_dir=Path.cwd() / ".lifecalor_cache", threshold_bytes=512 * 1024 * 1024)


def configure_array_cache(config: ArrayCacheConfig) -> None:
    global _ARRAY_CACHE_CONFIG
    _ARRAY_CACHE_CONFIG = config
    _ARRAY_CACHE_CONFIG.ensure_dir()


def get_array_store(progress_callback=None) -> ArrayStore:
    return ArrayStore(_ARRAY_CACHE_CONFIG, progress_callback=progress_callback)
```

- [ ] **Step 4: Convert `Data` array fields to cache-aware properties**

Replace public dataclass fields carefully:

```python
_data_origin_storage: object = field(init=False, repr=False, default=None)
_image_import_storage: object = field(init=False, repr=False, default=None)
```

In `Data.__post_init__`, before `_recalculate()`:

```python
self._data_origin_storage = self.data_origin
self._image_import_storage = self.image_import
```

Add properties after `_recalculate()`:

```python
@property
def data_origin(self):
    return resolve_array(self._data_origin_storage)

@data_origin.setter
def data_origin(self, value):
    self._data_origin_storage = value

@property
def image_import(self):
    return resolve_array(self._image_import_storage)

@image_import.setter
def image_import(self, value):
    self._image_import_storage = value
```

Add snapshot helper:

```python
def _history_snapshot(self):
    snapshot = copy.copy(self)
    store = get_array_store()
    if should_cache_array(snapshot.data_origin, store.config):
        snapshot._data_origin_storage = store.put_array(snapshot.data_origin, str(snapshot.timestamp), "data_origin")
    if should_cache_array(snapshot.image_import, store.config):
        snapshot._image_import_storage = store.put_array(snapshot.image_import, str(snapshot.timestamp), "image_import")
    return snapshot
```

Replace `Data.history.append(copy.deepcopy(self))` with `Data.history.append(self._history_snapshot())`.

Replace `Data.history[i] = copy.deepcopy(self)` with `Data.history[i] = self._history_snapshot()`.

- [ ] **Step 5: Convert `ProcessedData.data_processed` and `out_processed` to cache-aware storage**

Add field:

```python
_data_processed_storage: object = field(init=False, repr=False, default=None)
```

In `ProcessedData.__post_init__`, before metadata calculation:

```python
self._data_processed_storage = self.data_processed
```

Add property:

```python
@property
def data_processed(self):
    return resolve_array(self._data_processed_storage)

@data_processed.setter
def data_processed(self, value):
    self._data_processed_storage = value

def out_processed_array(self, key: str):
    return resolve_array(self.out_processed[key])
```

Add snapshot helper:

```python
def _history_snapshot(self):
    snapshot = copy.copy(self)
    store = get_array_store()
    if should_cache_array(snapshot.data_processed, store.config):
        snapshot._data_processed_storage = store.put_array(snapshot.data_processed, str(snapshot.timestamp), "data_processed")
    snapshot.out_processed = cache_large_arrays_in_mapping(snapshot.out_processed, store, owner_id=str(snapshot.timestamp))
    return snapshot
```

Replace `ProcessedData.history.append(copy.deepcopy(self))` and `ProcessedData.history[i] = copy.deepcopy(self)` with `_history_snapshot()`.

- [ ] **Step 6: Run integration tests**

Run: `python -m unittest discover -s tests -p "test_data_cache_integration.py" -v`

Expected: PASS.

---

### Task 5: Remove Long-Lived `unfolded_data` From Persistent Results

**Files:**
- Modify: `core/DataProcessor.py`
- Modify: `core/DataManager.py`
- Create or modify: `tests/test_unfolded_data_policy.py`

- [ ] **Step 1: Add failing tests for temporary unfolded data helper**

```python
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataProcessor import get_unfolded_data


class UnfoldedDataPolicyTests(unittest.TestCase):
    def test_get_unfolded_data_uses_existing_cache_when_present(self):
        existing = np.arange(12, dtype=np.float32).reshape(3, 4)
        data = SimpleNamespace(out_processed={"unfolded_data": existing})
        self.assertIs(get_unfolded_data(data), existing)

    def test_get_unfolded_data_rebuilds_from_three_dimensional_data(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        data = SimpleNamespace(out_processed={}, data_processed=source, data_origin=None)
        unfolded = get_unfolded_data(data)
        self.assertEqual(unfolded.shape, (4, 3))
        np.testing.assert_array_equal(unfolded, source.reshape((3, 4)).T)
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m unittest discover -s tests -p "test_unfolded_data_policy.py" -v`

Expected: FAIL because `get_unfolded_data` is missing.

- [ ] **Step 3: Implement helper and use it in STFT/CWT**

Add near top of `core/DataProcessor.py`:

```python
def get_unfolded_data(data):
    out_processed = getattr(data, "out_processed", {}) or {}
    if "unfolded_data" in out_processed:
        return out_processed["unfolded_data"]
    source = getattr(data, "data_processed", None)
    if source is None:
        source = getattr(data, "data_origin", None)
    if source is None:
        raise ValueError("无法展开数据：缺少 data_processed/data_origin")
    T, H, W = source.shape
    return source.reshape((T, H * W)).T
```

Replace repeated fallback blocks in `quality_stft`, `python_stft`, `quality_cwt`, and `python_cwt` with:

```python
unfolded_data = get_unfolded_data(data)
```

- [ ] **Step 4: Stop writing `unfolded_data` into preprocessed history**

In `MassDataProcessor.pre_process`, do not add `unfolded_data` to `out_processed`. Keep local calculation only if needed for immediate processing; otherwise omit it.

Replace:

```python
'unfolded_data': unfolded_data,
```

with no persistent entry. Keep `bg_frame`, `fps`, and metadata.

- [ ] **Step 5: Run focused tests**

Run: `python -m unittest discover -s tests -p "test_unfolded_data_policy.py" -v`

Expected: PASS.

---

### Task 6: Cache Settings UI and Progress Hooks

**Files:**
- Modify: `core/MainWindow.py`
- Modify: `core/DataManager.py`
- Create: `tests/test_cache_settings_architecture.py`

- [ ] **Step 1: Add architecture tests for settings and progress integration**

```python
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"


class CacheSettingsArchitectureTests(unittest.TestCase):
    def test_mainwindow_exposes_cache_settings(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertIn("cache_threshold_mb", source)
        self.assertIn("cache_directory", source)
        self.assertIn("configure_array_cache", source)
        self.assertIn("choose_cache_directory", source)

    def test_data_manager_accepts_progress_callback(self):
        source = (CORE / "DataManager.py").read_text(encoding="utf-8")
        self.assertIn("set_array_cache_progress_callback", source)
        self.assertIn("progress_callback", source)
```

- [ ] **Step 2: Run architecture test to verify failure**

Run: `python -m unittest discover -s tests -p "test_cache_settings_architecture.py" -v`

Expected: FAIL because UI/config hooks are missing.

- [ ] **Step 3: Add DataManager progress callback hook**

Add module-level callback:

```python
_ARRAY_CACHE_PROGRESS_CALLBACK = None


def set_array_cache_progress_callback(callback):
    global _ARRAY_CACHE_PROGRESS_CALLBACK
    _ARRAY_CACHE_PROGRESS_CALLBACK = callback


def get_array_store(progress_callback=None) -> ArrayStore:
    return ArrayStore(_ARRAY_CACHE_CONFIG, progress_callback=progress_callback or _ARRAY_CACHE_PROGRESS_CALLBACK)
```

- [ ] **Step 4: Add MainWindow cache settings initialization**

In `MainWindow.init_params`, add tool/cache defaults:

```python
'cache_directory': str(Path.cwd() / '.lifecalor_cache'),
'cache_threshold_mb': 512,
```

After params are loaded, call:

```python
self.apply_cache_settings()
```

Add methods:

```python
def apply_cache_settings(self):
    cache_dir = Path(self.tool_params.get('cache_directory') or Path.cwd() / '.lifecalor_cache')
    threshold_mb = int(self.tool_params.get('cache_threshold_mb', 512))
    configure_array_cache(ArrayCacheConfig(cache_dir=cache_dir, threshold_bytes=threshold_mb * 1024 * 1024))
    set_array_cache_progress_callback(self.cache_progress_update)


def cache_progress_update(self, current, total, message):
    self.update_status(message, 'working')
    self.update_progress(current, total)


def choose_cache_directory(self):
    directory = QFileDialog.getExistingDirectory(self, "选择缓存目录", self.tool_params.get('cache_directory', ''))
    if directory:
        self.update_param('tool', 'cache_directory', directory)
        self.apply_cache_settings()
        logging.info(f"缓存目录已更新: {directory}")
```

- [ ] **Step 5: Add menu action**

In `setup_menus`, add an action under existing settings/tool menu or data menu:

```python
cache_dir_action = tools_menu.addAction('设置缓存目录')
cache_dir_action.triggered.connect(self.choose_cache_directory)
```

If no `tools_menu` variable exists, add it to the data menu to avoid broader UI restructuring.

- [ ] **Step 6: Run architecture test**

Run: `python -m unittest discover -s tests -p "test_cache_settings_architecture.py" -v`

Expected: PASS.

---

### Task 7: Full Regression and Documentation

**Files:**
- Modify: `readme.md`
- All touched source/tests.

- [ ] **Step 1: Update README cache notes**

Add:

```markdown
## 数据缓存

LifeCalor uses `.npy` cache files for large arrays. The default threshold is 512 MB and is computed from `ndarray.nbytes`, so dtype differences such as `uint8`, `float32`, and `complex128` are reflected automatically. The cache directory and threshold are user-configurable from the app settings.

Historical `Data` and `ProcessedData` entries remain selectable. Large arrays in historical entries may be backed by cache references and are resolved to NumPy arrays or memmaps when accessed.
```

- [ ] **Step 2: Run all tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 3: Compile touched modules**

Run: `python -m py_compile core\ArrayCache.py core\DataManager.py core\DataProcessor.py core\MainWindow.py`

Expected: exit code 0.

- [ ] **Step 4: Inspect diff**

Run: `git diff --stat`

Expected: changes are limited to cache modules, data manager, processor, main window settings, tests, and docs.

- [ ] **Step 5: Commit**

```bash
git add core\ArrayCache.py core\DataManager.py core\DataProcessor.py core\MainWindow.py readme.md tests\test_array_cache.py tests\test_data_cache_integration.py tests\test_unfolded_data_policy.py tests\test_cache_settings_architecture.py
git commit -m "feat: add configurable array cache for large data"
```

---

## Self-Review Notes

- Covers configurable 512MB threshold and dtype-aware size through `ndarray.nbytes`.
- Explicitly covers primary arrays (`data_origin`, `image_import`, `data_processed`) and secondary arrays (`out_processed` ndarray values).
- Preserves free historical data selection by keeping public `Data` and `ProcessedData` access patterns.
- Avoids broad algorithm rewrites; STFT/CWT still receive array-like NumPy data.
- Removes persistent `unfolded_data` while preserving temporary reshape for computation speed.
- Includes cache directory UI and progress hooks for large cache writes/loads.
- Uses `.npy` and memmap for fast NumPy-native storage without new dependencies.


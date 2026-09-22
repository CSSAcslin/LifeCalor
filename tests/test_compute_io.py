import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from ArrayCache import ArrayRef
from compute.executor import run_bounded_cpu
from compute.io import ComputeNpySink, ComputeOutputSet, TaskProgressReporter
from compute.model import BackendPreference, ComputeRequest, PrecisionPolicy, ResourceBudget
from compute.planner import plan_compute
from tasks.model import CancellationToken, TaskCancelled


class _Coordinator:
    def __init__(self):
        self.updates = []

    def progress(self, task_id, current, total, message):
        self.updates.append((task_id, current, total, message))


class ComputeIoTests(unittest.TestCase):
    def test_multi_output_set_commits_independent_named_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = {
                "tau1_map": ((2, 3), "float64"),
                "fit_status": ((2, 3), "int16"),
            }
            sink = ComputeOutputSet(directory, "fit", 2, outputs)
            sink.write_block("tau1_map", (slice(None), slice(None)), np.full((2, 3), 2.5))
            sink.write_block("fit_status", (slice(None), slice(None)), np.ones((2, 3), dtype=np.int16))

            references = sink.commit()

            self.assertEqual(set(references), set(outputs))
            np.testing.assert_allclose(np.load(references["tau1_map"].path), 2.5)
            np.testing.assert_array_equal(np.load(references["fit_status"].path), 1)

    def test_block_output_is_invisible_until_commit_and_returns_array_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = None
            with ComputeNpySink(
                directory, "task-1", 0, "result", (4, 3), "float32"
            ) as sink:
                sink.write_block(slice(0, 2), np.ones((2, 3), dtype=np.float32))
                self.assertFalse(any(Path(directory).glob("task-1_result_*.npy")))
                sink.write_block(slice(2, 4), np.full((2, 3), 2, dtype=np.float32))
                ref = sink.commit()
                final_path = ref.path

            self.assertTrue(final_path.exists())
            self.assertEqual(ref.shape, (4, 3))
            np.testing.assert_array_equal(
                np.load(final_path),
                np.vstack((np.ones((2, 3)), np.full((2, 3), 2))).astype(np.float32),
            )
            self.assertFalse((Path(directory) / ".compute_tasks").exists())

    def test_failure_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "尺寸不匹配"):
                with ComputeNpySink(
                    directory, "task-2", 1, "result", (4, 3), "float32"
                ) as sink:
                    sink.write_block(slice(0, 2), np.ones((1, 3), dtype=np.float32))

            self.assertFalse(list(Path(directory).rglob("*.npy")))

    def test_cancelled_write_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            token = CancellationToken()
            with self.assertRaises(TaskCancelled):
                with ComputeNpySink(
                    directory, "task-3", 0, "result", (2, 2), "float32", token=token
                ) as sink:
                    token.cancel()
                    sink.write_block(slice(0, 1), np.ones((1, 2), dtype=np.float32))

            self.assertFalse(list(Path(directory).rglob("*.npy")))

    def test_task_progress_reporter_scales_large_byte_counts(self):
        coordinator = _Coordinator()
        reporter = TaskProgressReporter(coordinator, "task-4")
        reporter(2 ** 49, 2 ** 50, "写入")
        self.assertEqual(coordinator.updates, [("task-4", 5000, 10000, "写入")])


    def test_bounded_executor_reads_array_ref_and_streams_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.npy"
            source = np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3)
            np.save(source_path, source)
            ref = ArrayRef(source_path, source.shape, "float32", source.nbytes, 0, "source")
            item = ComputeRequest(
                task_id="task-5",
                attempt_id=0,
                algorithm="lifetime",
                data_id="data-5",
                shape=source.shape,
                dtype=source.dtype,
                axes="THW",
                source=ref,
                parameters={"output_shape": source.shape[1:]},
                backend=BackendPreference.CPU,
                precision=PrecisionPolicy.COMPATIBILITY,
            )
            plan = plan_compute(
                item,
                ResourceBudget(host_limit_bytes=256 * 1024 * 1024),
                disk_output_threshold_bytes=1,
            )
            result_ref = run_bounded_cpu(
                plan,
                lambda block, _input_index, _output_index: block.mean(axis=0),
                cache_dir=directory,
            )

            self.assertIsInstance(result_ref, ArrayRef)
            np.testing.assert_allclose(np.load(result_ref.path), source.mean(axis=0))

    def test_bounded_executor_keeps_small_output_in_memory(self):
        source = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
        item = ComputeRequest(
            task_id="task-6",
            attempt_id=0,
            algorithm="cwt",
            data_id="data-6",
            shape=source.shape,
            dtype=source.dtype,
            axes="THW",
            source=source,
            parameters={},
            backend=BackendPreference.CPU,
            precision=PrecisionPolicy.COMPATIBILITY,
        )
        plan = plan_compute(
            item,
            ResourceBudget(host_limit_bytes=256 * 1024 * 1024),
        )
        result = run_bounded_cpu(
            plan,
            lambda block, _input_index, _output_index: block * 2,
        )
        np.testing.assert_array_equal(result, source * 2)
    def test_processed_history_reuses_committed_array_ref(self):
        from DataManager import ProcessedData

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "committed.npy"
            values = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
            np.save(path, values)
            ref = ArrayRef(path, values.shape, "float32", values.nbytes, 0, "data_processed")
            ProcessedData.history.clear()

            ProcessedData(0.0, "bounded", "test", data_processed=ref)

            snapshot = ProcessedData.history[-1]
            storage = object.__getattribute__(snapshot, "__dict__")["_data_processed_storage"]
            self.assertIs(storage, ref)
            self.assertEqual(list(Path(directory).glob("*.npy")), [path])
if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from dataio.npy import copy_npy_to_memory, write_npy_atomic
from tasks.model import CancellationToken, TaskCancelled


class ChunkedNpyIoTests(unittest.TestCase):
    def test_round_trip_preserves_dtype_shape_and_values(self):
        for dtype in (np.int16, np.float32, np.complex64):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                source = np.arange(60, dtype=np.float32).reshape(5, 3, 4).astype(dtype)
                if np.issubdtype(dtype, np.complexfloating):
                    source = source + source * 1j
                path = Path(directory) / "value.npy"
                events = []
                write_npy_atomic(path, source, progress=lambda *event: events.append(event), chunk_bytes=24)
                restored = copy_npy_to_memory(path, progress=lambda *event: events.append(event), chunk_bytes=24)
                self.assertEqual(restored.dtype, source.dtype)
                self.assertEqual(restored.shape, source.shape)
                np.testing.assert_array_equal(restored, source)
                self.assertGreater(len(events), 4)

    def test_cancelled_write_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cancelled.npy"
            token = CancellationToken()

            def cancel_after_first_chunk(current, total, _message):
                if 0 < current < total:
                    token.cancel()

            with self.assertRaises(TaskCancelled):
                write_npy_atomic(path, np.arange(1000), cancel_after_first_chunk, token, chunk_bytes=64)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).glob("*.writing.npy")), [])

    def test_cancelled_read_does_not_return_partial_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npy"
            np.save(path, np.arange(1000))
            token = CancellationToken()

            def cancel_after_first_chunk(current, total, _message):
                if 0 < current < total:
                    token.cancel()

            with self.assertRaises(TaskCancelled):
                copy_npy_to_memory(path, cancel_after_first_chunk, token, chunk_bytes=64)


if __name__ == "__main__":
    unittest.main()

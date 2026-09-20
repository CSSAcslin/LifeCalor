import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import Data, ProcessedData
from DataProcessor import MassDataProcessor


class ComputeIntegrationTests(unittest.TestCase):
    def setUp(self):
        Data.history.clear()
        ProcessedData.history.clear()

    @staticmethod
    def source_data():
        time = np.arange(128, dtype=np.float64) / 128.0
        trace = np.sin(2 * np.pi * 16 * time).astype(np.float32)
        values = np.empty((128, 2, 2), dtype=np.float32)
        values[:, 0, 0] = trace
        values[:, 0, 1] = trace * 0.5
        values[:, 1, 0] = trace * 0.25
        values[:, 1, 1] = trace * 0.75
        return ProcessedData(
            1.0,
            "source",
            "EM_pre",
            time_point=time,
            data_processed=values,
            out_processed={"fps": 128, "data_type": "central positive"},
        )

    def test_serial_stft_emits_contract_metadata_and_matching_axis(self):
        processor = MassDataProcessor()
        results = []
        errors = []
        processor.processed_result.connect(results.append)
        processor.processing_error_signal.connect(errors.append)

        success = processor.python_stft(
            self.source_data(),
            16.0,
            0,
            128,
            64,
            32,
            64,
            "hann",
            False,
            4,
            1,
        )

        self.assertTrue(success)
        self.assertFalse(errors)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.data_processed.shape[0], result.time_point.size)
        self.assertEqual(result.data_processed.shape[1:], (2, 2))
        self.assertEqual(result.data_processed.dtype, np.float32)
        self.assertEqual(result.out_processed["compute"]["algorithm"], "stft")
        self.assertEqual(result.out_processed["compute"]["execution"], "bounded_cpu")
        self.assertEqual(result.out_processed["compute"]["actual_backend"], "cpu")
        self.assertIn("chunk_shape", result.out_processed["compute"])

    def test_preprocess_emits_bounded_contract_result(self):
        source = np.arange(8 * 2 * 2, dtype=np.uint16).reshape(8, 2, 2) + 1
        data = Data(
            data_origin=source,
            time_point=np.arange(8, dtype=np.float64),
            format_import="unit",
            image_import=source,
            parameters={"fps": 8},
            name="raw",
        )
        processor = MassDataProcessor()
        results = []
        errors = []
        processor.processed_result.connect(results.append)
        processor.processing_error_signal.connect(errors.append)

        success = processor.pre_process(data, bg_num=3, unfold=True)

        self.assertTrue(success)
        self.assertFalse(errors)
        result = results[0]
        background = np.median(source[:3].astype(np.float32), axis=0)
        expected = (source.astype(np.float32) - background) / background
        np.testing.assert_allclose(result.data_processed, expected, rtol=1e-6)
        self.assertEqual(result.out_processed["compute"]["execution"], "bounded_cpu")
        self.assertEqual(
            result.out_processed["compute"]["background_reduction"], "exact_median"
        )
    def test_multiscale_cwt_emits_one_thw_result(self):
        processor = MassDataProcessor()
        results = []
        errors = []
        processor.processed_result.connect(results.append)
        processor.processing_error_signal.connect(errors.append)

        success = processor.python_cwt(
            self.source_data(),
            target_freq=16.0,
            fps=128,
            totalscales=5,
            wavelet="morl",
            cwt_scale_range=4.0,
        )

        self.assertTrue(success)
        self.assertFalse(errors)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.data_processed.shape, (128, 2, 2))
        self.assertEqual(result.data_processed.dtype, np.float32)
        self.assertEqual(result.out_processed["compute"]["algorithm"], "cwt")
        self.assertEqual(result.out_processed["compute"]["execution"], "bounded_cpu")
        self.assertEqual(
            result.out_processed["compute"]["scale_reduction"],
            "normalized_mean",
        )


if __name__ == "__main__":
    unittest.main()

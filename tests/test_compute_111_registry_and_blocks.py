import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.blocking import iter_block_plans
from compute.backends.cuda import get_cuda_handler
from compute.registry import ExecutionMode, get_algorithm_spec, user_algorithm_keys
from compute.settings import ALGORITHM_KEYS


class ComputeRegistryTests(unittest.TestCase):
    def test_settings_keys_are_derived_from_registry(self):
        self.assertEqual(ALGORITHM_KEYS, user_algorithm_keys())
        self.assertNotIn("spatiotemporal_convolution", ALGORITHM_KEYS)

    def test_registry_describes_outputs_and_trusted_handlers(self):
        spec = get_algorithm_spec("lifetime_double")
        self.assertEqual(spec.input_axes, "THW")
        self.assertEqual(spec.execution_mode, ExecutionMode.MULTI_OUTPUT)
        self.assertEqual(
            tuple(output.name for output in spec.outputs),
            (
                "tau1_map", "tau2_map", "amplitude1_map",
                "amplitude2_map", "baseline_map", "r_squared_map", "fit_status",
            ),
        )
        self.assertEqual(get_algorithm_spec("stft").cuda_handler, "stft_block")
        self.assertEqual(get_algorithm_spec("cwt").cuda_handler, "cwt_block")

    def test_unknown_algorithm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知计算算法"):
            get_algorithm_spec("not-an-algorithm")

    def test_cuda_dispatch_only_exposes_validated_handlers_by_default(self):
        self.assertTrue(callable(get_cuda_handler("stft_block")))
        self.assertTrue(callable(get_cuda_handler("cwt_block")))


class BlockPlanTests(unittest.TestCase):
    def test_asymmetric_halo_is_clipped_and_only_global_edges_are_padded(self):
        plans = tuple(iter_block_plans(
            (7,), core_shape=(3,), halo_before=(1,), halo_after=(2,)
        ))

        self.assertEqual(len(plans), 3)
        self.assertEqual(plans[0].read_slices, (slice(0, 5),))
        self.assertEqual(plans[0].pad_width, ((1, 0),))
        self.assertEqual(plans[0].local_core_slices, (slice(1, 4),))

        self.assertEqual(plans[1].read_slices, (slice(2, 7),))
        self.assertEqual(plans[1].pad_width, ((0, 1),))
        self.assertEqual(plans[1].local_core_slices, (slice(1, 4),))

        self.assertEqual(plans[2].read_slices, (slice(5, 7),))
        self.assertEqual(plans[2].pad_width, ((0, 2),))
        self.assertEqual(plans[2].local_core_slices, (slice(1, 2),))

    def test_even_kernel_halo_can_be_directional(self):
        plan = next(iter_block_plans(
            (8, 9), core_shape=(8, 9),
            halo_before=(1, 2), halo_after=(2, 1),
        ))
        self.assertEqual(plan.pad_width, ((1, 2), (2, 1)))


if __name__ == "__main__":
    unittest.main()

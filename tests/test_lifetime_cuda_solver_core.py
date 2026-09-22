import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.algorithms.lifetime import LIFETIME_SOLVER_CONTRACT
from compute.backends.lifetime_cuda import _prepare_batch, _solve_batch


FIT_PARAMS = {
    "from_start_cal": True,
    "r_squared_min": 0.4,
    "peak_range": (0, 300),
    "tau_range": (1e-3, 100.0),
}


class LifetimeCudaSolverCoreTests(unittest.TestCase):
    def _solve(self, trace, times, model_type):
        block = np.broadcast_to(trace[:, None, None], (trace.size, 1, 1))
        values, relative_times, mask, initial, status, valid = _prepare_batch(
            block, times, None, FIT_PARAMS, model_type
        )
        bounds = LIFETIME_SOLVER_CONTRACT[
            "single_bounds" if model_type == "single" else "double_bounds"
        ]
        parameters, scores, converged = _solve_batch(
            np,
            values[valid],
            relative_times[valid],
            mask[valid],
            initial[valid],
            np.asarray(bounds[0]),
            model_type,
        )
        self.assertEqual(valid.tolist(), [0])
        self.assertTrue(bool(converged[0]))
        self.assertGreater(float(scores[0]), 0.999999)
        return parameters[0]

    def test_single_solver_core_recovers_analytic_parameters(self):
        times = np.linspace(0, 20, 200)
        parameters = self._solve(
            8 * np.exp(-times / 2.5) + 0.5, times, "single"
        )
        np.testing.assert_allclose(parameters, (8.0, 2.5, 0.5), rtol=1e-4)

    def test_double_solver_core_respects_bounds_and_recovers_taus(self):
        times = np.linspace(0, 60, 400)
        parameters = self._solve(
            50 * np.exp(-times / 3.0)
            + 20 * np.exp(-times / 15.0)
            + 2,
            times,
            "double",
        )
        self.assertGreaterEqual(parameters[2], 10.0)
        self.assertGreaterEqual(parameters[3], 10.0)
        np.testing.assert_allclose(
            sorted((parameters[1], parameters[3])),
            (3.0, 15.0),
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()

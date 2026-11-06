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

    def test_double_batch_avoids_saturating_lower_bound_transform(self):
        times = np.linspace(0.0, 30.0, 200)
        expected = np.asarray([
            [30.0, 2.5, 15.0, 12.0, 0.0],
            [35.0, 2.7, 16.0, 13.0, 1.0],
            [40.0, 2.9, 17.0, 14.0, 2.0],
            [45.0, 3.1, 18.0, 15.0, 3.0],
        ])
        traces = np.asarray([
            first * np.exp(-times / tau1) + second * np.exp(-times / tau2) + baseline
            for first, tau1, second, tau2, baseline in expected
        ])
        block = traces.T.reshape(times.size, 2, 2)
        values, relative_times, mask, initial, _status, valid = _prepare_batch(
            block, times, None, FIT_PARAMS, "double"
        )
        parameters, scores, converged = _solve_batch(
            np, values[valid], relative_times[valid], mask[valid], initial[valid],
            np.asarray(LIFETIME_SOLVER_CONTRACT["double_bounds"][0]), "double",
        )
        self.assertTrue(np.all(converged))
        np.testing.assert_allclose(parameters, expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(scores, 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

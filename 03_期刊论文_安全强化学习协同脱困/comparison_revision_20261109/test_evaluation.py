import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import EvaluationConfig, evaluate


class EvaluationChecks(unittest.TestCase):
    def test_elapsed_time_holding_and_no_early_twentieth_sample(self):
        scene = dict(name="stationary_goal", p=np.array([[0.0, 0.0]]), goal=np.array([[0.0, 0.0]]))
        with tempfile.TemporaryDirectory() as directory:
            row = evaluate(scene, lambda position: np.zeros_like(position), Path(directory) / "run.npz")
        self.assertEqual(row["status"], "arrived")
        self.assertEqual(row["steps"], 21)
        self.assertAlmostEqual(row["holding_duration"], 1.0)

    def test_plant_preserves_outward_controller_command(self):
        scene = dict(name="boundary", p=np.array([[3.99, 0.0]]), goal=np.array([[-3.0, 0.0]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.npz"
            row = evaluate(scene, lambda position: np.array([[0.5, 0.0]]), path, EvaluationConfig(horizon=0.1))
            with np.load(path) as archive:
                self.assertAlmostEqual(archive["states"][-1, 0, 0], 4.04)
                np.testing.assert_array_equal(archive["actions"][:, 0, 0], np.array([0.5, 0.5]))
        self.assertEqual(row["status"], "horizon_reached")

    def test_goal_window_resets_when_robot_leaves(self):
        scene = dict(name="leave_goal", p=np.array([[0.0, 0.0]]), goal=np.array([[0.0, 0.0]]))
        calls = []

        def command(position):
            calls.append(1)
            return np.array([[0.5 if len(calls) == 10 else -0.5 if len(calls) == 11 else 0.0, 0.0]])

        with tempfile.TemporaryDirectory() as directory:
            row = evaluate(scene, command, Path(directory) / "run.npz")
        self.assertAlmostEqual(row["arrival"], 0.55)
        self.assertAlmostEqual(row["duration"], 1.55)


if __name__ == "__main__":
    unittest.main()

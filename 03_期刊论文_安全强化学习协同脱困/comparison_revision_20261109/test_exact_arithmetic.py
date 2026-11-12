import json
import runpy
import time
import unittest
from pathlib import Path

import numpy as np

from exact_arithmetic import ExactRational, PythonFraction
from exact_arithmetic import enable


HERE = Path(__file__).resolve().parent


class ExactArithmeticChecks(unittest.TestCase):
    def test_decimal_and_scientific_notation(self):
        for text in ("0.0000001", "1e-15", "1.001", "-3.45", "19/515"):
            converted = ExactRational(text)
            expected = PythonFraction(text)
            self.assertEqual(int(converted.numerator), expected.numerator)
            self.assertEqual(int(converted.denominator), expected.denominator)

    def test_same_frozen_certificate_outputs(self):
        runpy.run_path(str(HERE.parent / "experiments_sota_rl" / "run_sota_baselines.py"))
        import safety_checks
        checked = 0
        started = time.perf_counter()
        for pattern in ("H8_local0_f60*", "H12_local0*", "H20_local0*", "H25_local0*"):
            filename = sorted((HERE / "external_egp_s1").glob(pattern + "_egp_final.npz"))[0]
            row = json.loads(filename.with_suffix(".json").read_text(encoding="utf-8"))
            with np.load(filename) as archive:
                states = archive["states"]
            for index in sorted({0, min(10, len(states) - 2), len(states) - 2}):
                safety_checks.Fraction = PythonFraction
                reference = safety_checks.exact_affine_segment_certificate(states[index], states[index + 1], 10, 1, row["lambda_required"])
                safety_checks.Fraction = ExactRational
                accelerated = safety_checks.exact_affine_segment_certificate(states[index], states[index + 1], 10, 1, row["lambda_required"])
                self.assertEqual(reference, accelerated)
                checked += 1
        safety_checks.Fraction = PythonFraction
        (HERE / "exact_arithmetic_validation.json").write_text(json.dumps(dict(
            matched_certificate_outputs=checked, scales=[8,12,20,25],
            arithmetic="Identical exact rational values; same frozen certificate algorithm; GMP arithmetic backend",
            elapsed_seconds=time.perf_counter() - started), indent=2), encoding="utf-8")

    def test_collision_short_circuit_matches_boolean_certificate(self):
        runpy.run_path(str(HERE.parent / "experiments_sota_rl" / "run_sota_baselines.py"))
        import safety_checks
        import audit_trajectories
        safety_checks.Fraction = PythonFraction
        path = sorted((HERE / "external_dgppo_s2").glob("H8_local0_f60*.npz"))[0]
        row = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path) as archive:
            states = archive["states"]
        expected = [safety_checks.exact_affine_segment_certificate(states[index], states[index + 1], 10, 1, row["lambda_required"])["certified"] for index in range(12)]
        enable(short_circuit=True)
        actual = [audit_trajectories.exact_affine_segment_certificate(states[index], states[index + 1], 10, 1, row["lambda_required"])["certified"] for index in range(12)]
        self.assertEqual(expected, actual)
        self.assertTrue(any(not value for value in actual))


if __name__ == "__main__":
    unittest.main()

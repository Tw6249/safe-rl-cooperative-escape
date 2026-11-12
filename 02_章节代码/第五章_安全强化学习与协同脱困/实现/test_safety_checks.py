import unittest
import numpy as np
from fractions import Fraction
from safety_checks import affine_segment_bounds, configuration_metrics, exact_affine_segment_certificate


class SafetyChecksTest(unittest.TestCase):
    def test_rational_certificate_and_strict_threshold_boundary(self):
        a = [[-1, 0], [1, 0]]
        self.assertTrue(exact_affine_segment_certificate(a, a, 3, 2, Fraction(1, 2))['certified'])
        self.assertFalse(exact_affine_segment_certificate(a, a[::-1], 3, 2, Fraction(1, 2))['collision_safe'])
        # This certificate intentionally requires a strict spectral margin.
        self.assertFalse(exact_affine_segment_certificate(a, a, 3, 2, Fraction(50, 81))['strict_spectral_certificate'])

    def test_rational_reduced_matrix_retains_nonorthonormal_gram(self):
        for n in [4, 6]:
            a = np.array([[i % 3, i // 3] for i in range(n)])
            lam = configuration_metrics(a, 4)['lambda2']
            self.assertTrue(exact_affine_segment_certificate(a, a, 4, 1, lam/2)['certified'])
            self.assertFalse(exact_affine_segment_certificate(a, a, 4, 1, lam*1.1)['strict_spectral_certificate'])

    def test_safe_endpoints_do_not_hide_midstep_collision(self):
        p0 = np.array([[-1., 0.], [1., 0.]])
        p1 = -p0
        self.assertEqual(configuration_metrics(p0, 3)['min_distance'], 2)
        self.assertEqual(configuration_metrics(p1, 3)['min_distance'], 2)
        self.assertEqual(affine_segment_bounds(p0, p1, 3)['min_distance'], 0)

    def test_known_spectrum_and_rigid_translation(self):
        p0 = np.array([[-1., 0.], [1., 0.]])
        bounds = affine_segment_bounds(p0, p0 + [5., -7.], 3)
        self.assertAlmostEqual(bounds['min_distance'], 2)
        self.assertAlmostEqual(bounds['lambda2_lower'], 50/81)

    def test_cutoff_and_degenerate_segment(self):
        p = np.array([[0., 0.], [3., 0.], [8., 0.]])
        self.assertEqual(affine_segment_bounds(p, p, 3)['lambda2_lower'], 0)

    def test_segment_bounds_against_independent_dense_evaluation(self):
        rng = np.random.default_rng(20261007)
        for n in [2, 4, 6]:
            for _ in range(20):
                a, b = rng.normal(size=(2, n, 2))
                bounds = affine_segment_bounds(a, b, 3)
                for s in np.linspace(0, 1, 101):
                    m = configuration_metrics((1-s)*a+s*b, 3)
                    self.assertGreaterEqual(m['min_distance'] + 1e-12, bounds['min_distance'])
                    self.assertGreaterEqual(m['lambda2'] + 1e-12, bounds['lambda2_lower'])


if __name__ == '__main__':
    unittest.main()

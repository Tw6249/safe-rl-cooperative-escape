"""Independent audit of saved trajectories: re-certify EVERY executed segment from the stored endpoints.

For each segment: exact rational collision/spectral certificate (safety_checks), exact per-robot displacement bound
(speed <= 0.5), and workspace [-4,4]^2 at the endpoints.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse, json
import numpy as np
from safety_checks import exact_affine_segment_certificate, affine_segment_bounds
from pair_common import dump


def audit(npz, lam):
    z = np.load(npz)
    st = z['states']
    ts = np.array(z['times'])
    bad = []
    n_seg = len(st) - 1
    min_d = 1e9
    min_l = 1e9
    for k in range(n_seg):
        p, q = st[k], st[k + 1]
        dt = float(ts[k + 1] - ts[k])
        cert = exact_affine_segment_certificate(p, q, 10., 1., lam)
        b = affine_segment_bounds(p, q, 10.)
        min_d = min(min_d, b['min_distance'])
        min_l = min(min_l, b['lambda2_lower'])
        lim = (F(str(.5)) * F(repr(dt))) ** 2
        speed_ok = all(sum((F(repr(float(q[i, j]))) - F(repr(float(p[i, j])))) ** 2 for j in range(2)) <= lim
                       for i in range(len(p)))
        if not (cert['certified'] and speed_ok and np.abs(q).max() <= 4. + 1e-12):
            bad.append(k)
    return n_seg, bad, min_d, min_l

"""Scenarios: the frozen original nine (scenarios.json) and seeded triangular-lattice scenes.

Lattice scenes: spacing 1.001 (strictly feasible), goals = a permutation of the same lattice sites;
lambda_req = 2.5 for N=7, 0.6*lambda_2(initial) otherwise.
"""
from pathlib import Path
import json
import numpy as np
from safety_checks import pair_indices, laplacian_from_weights, weights_from_squared_distance

HERE = Path(__file__).resolve().parent
S = 1.001
RADIUS = 2.5
SIZES = [(12, 'local', 4), (12, 'perm', 2), (20, 'local', 4), (20, 'perm', 2), (25, 'local', 4)]


def original_cases():
    data = json.loads((HERE / 'scenarios.json').read_text(encoding='utf8'))
    return [dict(name=c['name'], p=np.array(c['p']), goal=np.array(c['goal']), lam=c['lam']) for c in data['cases']]


def lattice(n):
    pts = [(S * (a + b / 2), S * b * np.sqrt(3) / 2) for a in range(-4, 5) for b in range(-4, 5)]
    pts = sorted(pts, key=lambda q: (round(q[0] ** 2 + q[1] ** 2, 6), np.arctan2(q[1], q[0])))
    return np.array(pts[:n])


def lambda2(p, radius=10.):
    i, j = pair_indices(len(p))
    rel = p[i] - p[j]
    d2 = np.sum(rel * rel, axis=1)
    return float(np.linalg.eigvalsh(laplacian_from_weights(len(p), weights_from_squared_distance(d2, radius)))[1])


def adjacent_pairs(p):
    i, j = pair_indices(len(p))
    d = np.linalg.norm(p[i] - p[j], axis=1)
    return [(int(a), int(b)) for a, b, dd in zip(i, j, d) if dd < 1.2 * S]


def lattice_cases(seed=20261012):
    rng = np.random.default_rng(seed)
    cases = []
    for n, kind, count, swaps in [(7, 'local', 4, 2), (7, 'perm', 4, 0), (12, 'local', 6, None), (19, 'local', 4, None)]:
        p = lattice(n)
        adj = adjacent_pairs(p)
        for k in range(count):
            perm = np.arange(n)
            if kind == 'perm':
                perm = rng.permutation(n)
            else:
                m = swaps if swaps else int(rng.integers(1, 4))
                used = set()
                for _ in range(m):
                    for _try in range(50):
                        a, b = adj[int(rng.integers(len(adj)))]
                        if a not in used and b not in used:
                            perm[[a, b]] = perm[[b, a]]
                            used |= {a, b}
                            break
            lam = 2.5 if n == 7 else round(0.6 * lambda2(p), 3)
            cases.append(dict(name=f'H{n}_{kind}{k}', p=p, goal=p[perm], lam=lam, n=n, perm=perm.tolist()))
    return cases


def all_cases(seed=20261012):
    pool = {c['name']: c for c in original_cases()}
    pool.update({c['name']: c for c in lattice_cases(seed)})
    return pool


def effective_margin(case, m0):
    """Discretization margin may never exclude the goal: m = min(m0, (lambda2(g)-lambda_req)/2)."""
    return max(0., min(m0, .5 * (lambda2(case['goal']) - case['lam'])))


def scenes_for(seed):
    """Cross-scale evaluation scenes (N in {12, 20, 25}, local swaps and full permutations)."""
    rng = np.random.default_rng(seed)
    out = {}
    for n, kind, count in SIZES:
        p = lattice(n)
        adj = adjacent_pairs(p)
        for k in range(count):
            perm = np.arange(n)
            if kind == 'perm':
                perm = rng.permutation(n)
            else:
                m = int(rng.integers(1, 4))
                used = set()
                for _ in range(m):
                    for _try in range(50):
                        a, b = adj[int(rng.integers(len(adj)))]
                        if a not in used and b not in used:
                            perm[[a, b]] = perm[[b, a]]
                            used |= {a, b}
                            break
            name = f'H{n}_{kind}{k}_s{seed}'
            out[name] = dict(name=name, p=p, goal=p[perm], lam=round(0.6 * lambda2(p), 3), n=n, perm=perm.tolist())
    return out

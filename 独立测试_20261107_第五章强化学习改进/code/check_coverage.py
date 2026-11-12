"""Phase-0 necessity check for upgrade 1 (closed-neighbourhood critic): how much of the team does the closed
neighbourhood N_i^+ (2.5 m sensing graph, the same graph as filter L) already contain?  If N_i^+ is almost the whole
team at the training scale, the block reward R_i = mean_{j in N_i^+} r_j is almost the team reward.

    python -B -X utf8 check_coverage.py --out ../results/phase0_coverage
"""
import paths  # noqa: F401
import argparse, json
from pathlib import Path
import numpy as np
import sarl_env as E


def coverage(p):
    n = len(p)
    dist = np.linalg.norm(p[:, None] - p[None], axis=2)
    nb = (dist <= E.RADIUS)                       # includes i itself -> closed neighbourhood
    return nb.sum(1) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=3000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = {}
    by_n = {}
    rng = np.random.default_rng(777001)          # statistic only; not a training or test seed
    for k in range(a.draws):
        sc = E.train_scene(rng)
        c0 = coverage(sc['p'])
        cg = coverage(sc['goal'])
        by_n.setdefault(sc['n'], []).append(np.concatenate([c0, cg]))
    allc = np.concatenate([np.concatenate(v) for v in by_n.values()])
    res['train'] = dict(draws=a.draws, mean=float(allc.mean()), median=float(np.median(allc)),
                        p10=float(np.quantile(allc, .1)), frac_ge_0_8=float(np.mean(allc >= 0.8)),
                        frac_full=float(np.mean(allc >= 1 - 1e-12)),
                        by_n={int(n): dict(count=len(v), mean=float(np.concatenate(v).mean()),
                                           frac_full=float(np.mean(np.concatenate(v) >= 1 - 1e-12)))
                              for n, v in sorted(by_n.items())})
    from scenes import lattice
    res['lattice_t0'] = {int(n): dict(mean=float(coverage(lattice(n)).mean()),
                                      min=float(coverage(lattice(n)).min()),
                                      max=float(coverage(lattice(n)).max())) for n in (8, 12, 20, 25)}
    (out / 'coverage.json').write_text(json.dumps(res, indent=1), encoding='utf8')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

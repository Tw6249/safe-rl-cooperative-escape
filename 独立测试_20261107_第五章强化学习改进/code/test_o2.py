"""Checks for idea 6 (README §3.3), run before any closed-loop job:
1. every observation column is either reflection-even or reflection-odd, and the odd set equals PSEUDO_NODE/EDGE;
2. O2Policy is exactly reflection-equivariant (omega -> -omega, rho and psi unchanged);
3. the thesis actor itself is not (size of its chirality term).

    python -B -X utf8 test_o2.py --out ../results/phase0_o2check
"""
import paths  # noqa: F401
import argparse, json
from pathlib import Path
import numpy as np
import torch
import sarl_env as E
import sarl_policy as P
import exp_common as X


def obs_pair(scene, rng):
    n = scene['n']
    fa, fb = E.ResidualFlow(scene, .05, 0., .02, False), E.ResidualFlow(X.mirror_case(scene), .05, 0., .02, False)
    v = rng.normal(0, .2, (n, 2)); dl = rng.normal(0, .1, (n, 2))
    M = np.array([1., -1.])
    fa.u_prev, fa.delta_prev = v, dl
    fb.u_prev, fb.delta_prev = v * M, dl * M
    oa, _ = fa.observe(scene['p'])
    ob, _ = fb.observe(scene['p'] * M)
    return oa, ob


def classify(a, b):
    out = []
    for c in range(a.shape[1]):
        if np.allclose(a[:, c], b[:, c], atol=1e-5):
            out.append('even' if not np.allclose(a[:, c], 0, atol=1e-7) else 'zero')
        elif np.allclose(a[:, c], -b[:, c], atol=1e-5):
            out.append('odd')
        else:
            out.append('neither')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rng = np.random.default_rng(777002)
    actor = P.Actor('full'); actor.load_state_dict(torch.load(str(paths.THESIS_CKPT), map_location='cpu')['actor'])
    actor.eval()
    base = P.NumpyPolicy(actor, True)
    o2 = X.O2Policy(base)
    cls_x, cls_e, cls_xa, cls_eaa = {}, {}, {}, {}
    err_o2, chir = [], []
    for k in range(40):
        sc = E.train_scene(rng)
        oa, ob = obs_pair(sc, rng)
        for name, store in (('x', cls_x), ('ea', cls_e), ('xa', cls_xa), ('eaa', cls_eaa)):
            for c, lab in enumerate(classify(oa[name], ob[name])):
                store.setdefault(c, set()).add(lab)
        na, ea_ = o2(oa); nb, eb = o2(ob)
        err_o2.append(max(np.abs(na[:, 0] + nb[:, 0]).max(), np.abs(na[:, 1] - nb[:, 1]).max(), np.abs(ea_ - eb).max()))
        ba, bea = base(oa); bb, beb = base(ob)
        chir.append(dict(omega_asym=float(np.abs(ba[:, 0] + bb[:, 0]).max()),
                         omega_scale=float(np.abs(ba[:, 0]).max()),
                         omega_even_part=float(np.abs(0.5 * (ba[:, 0] + bb[:, 0])).mean())))
    odd_x = sorted(c for c, s in cls_x.items() if s <= {'odd', 'zero'} and 'odd' in s)
    odd_e = sorted(c for c, s in cls_e.items() if s <= {'odd', 'zero'} and 'odd' in s)
    bad = {nm: {c: sorted(s) for c, s in st.items() if 'neither' in s or ('odd' in s and 'even' in s)}
           for nm, st in (('x', cls_x), ('ea', cls_e), ('xa', cls_xa), ('eaa', cls_eaa))}
    alg_odd = [c for st in (cls_xa, cls_eaa) for c, s in st.items() if 'odd' in s]
    res = dict(odd_node=odd_x, odd_edge=odd_e, expected_node=X.PSEUDO_NODE, expected_edge=X.PSEUDO_EDGE,
               mixed_or_neither=bad, algebraic_odd=alg_odd, o2_max_equivariance_error=float(max(err_o2)),
               thesis_actor_chirality=dict(
                   omega_asym_median=float(np.median([c['omega_asym'] for c in chir])),
                   omega_scale_median=float(np.median([c['omega_scale'] for c in chir])),
                   omega_even_part_mean=float(np.mean([c['omega_even_part'] for c in chir]))),
               passed=bool(odd_x == X.PSEUDO_NODE and odd_e == X.PSEUDO_EDGE and not alg_odd
                           and all(not v for v in bad.values()) and max(err_o2) < 1e-5))
    (out / 'o2check.json').write_text(json.dumps(res, indent=1), encoding='utf8')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

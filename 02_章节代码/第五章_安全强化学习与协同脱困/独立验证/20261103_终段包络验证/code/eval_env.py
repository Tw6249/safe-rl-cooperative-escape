"""Test whether multiplying the residual of ResidualFlow by the closed-neighbourhood goal envelope
    rho_i = tanh( max_{j in N_i^+} |e_j| / eps_g )
eliminates the 2-5 cm terminal hovering of rl_full on N=20/25 without retraining.
Imports frozen code unchanged; writes results to a new directory."""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
from functools import partial
import argparse, json, sys, time
import multiprocessing as mp
import numpy as np

HERE = Path(__file__).resolve().parent
SARL = HERE.parents[1] / '20261102_安全强化学习' / 'code'
sys.path.insert(0, str(SARL))
import sarl_env as E
import evaluate as EV

EPS_G = 0.05


class EnvelopedResidualFlow(E.ResidualFlow):
    """Same as ResidualFlow, except the residual is multiplied by rho_i = tanh(max_{j in N_i^+} |e_j| / EPS_G)
    so that once a robot and its 1-hop neighbours are within EPS_G of their goals, the residual smoothly vanishes."""
    def compose(self, p, geo, gate, node_a, edge_a):
        e, I, J, d, deg, ne = geo['e'], geo['I'], geo['J'], geo['d'], geo['deg'], geo['ne']
        nmax = ne.copy()
        np.maximum.at(nmax, I, ne[J])
        rho = np.tanh(nmax / EPS_G)
        eff_gate = gate * rho
        return super().compose(p, geo, eff_gate, node_a, edge_a)


_W = {}


def init_worker():
    import sysperf
    sysperf.configure()
    import torch
    torch.set_num_threads(1)
    _W.update(torch=torch, actors={})


def actor_for(path, variant):
    if path not in _W['actors']:
        import sarl_policy as P
        a = P.Actor(variant)
        ck = _W['torch'].load(path, map_location='cpu')
        a.load_state_dict(ck['actor'])
        a.eval()
        _W['actors'][path] = a
    return _W['actors'][path]


def run_one(job):
    import pa_bridge
    from audit_trajectories import audit
    from pair_common import dump
    import sarl_policy as P
    case = job['case']
    case['p'], case['goal'] = np.asarray(case['p'], float), np.asarray(case['goal'], float)
    out = Path(job['out'])
    d = out / 'runs'
    pa_bridge.PAOracle = partial(EV.budget_oracle(E), settle='local', diagnostic=False)
    pa_bridge.BraidFlow = EnvelopedResidualFlow if job['envelope'] else E.ResidualFlow
    pa_bridge.BraidFlow.gate_kind = job['gate']
    pa_bridge.BraidFlow.policy = P.NumpyPolicy(actor_for(job['ckpt'], job['variant']), True)
    r = pa_bridge.run_pa(case, d, 0.0, job['rounds'], E.RADIUS, job['horizon'], job['tag'], clip_level=True)
    npz = d / f"{case['name']}_{job['tag']}.npz"
    n_seg, bad, md, ml = audit(npz, case['lam']) if r['steps'] else (0, [], None, None)
    r.update(audit_segments=n_seg, audit_failed=len(bad), audit_min_distance=md, audit_min_lambda2=ml,
             final_max_err=float(np.linalg.norm(np.load(npz)['states'][-1] - case['goal'], axis=1).max()))
    dump(out / 'rows' / f"{case['name']}_{job['tag']}.json", r)
    print(job['tag'], case['name'], r['status'], 'arrival', r['arrival'], 'final_err %.4f' % r['final_max_err'],
          'audit_bad', r['audit_failed'], 'cuts', r['step_cuts'], 'wall %.0fs' % r['wall_seconds'], flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261017)
    ap.add_argument('--only', default=None)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    import sysperf
    sysperf.configure()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'runs').mkdir(); (out / 'rows').mkdir()
    scenes = E.scenes_for(a.seed)
    names = sorted(scenes) if a.only is None else a.only.split(',')
    ckpt = str((SARL.parent / 'results' / 'train_full02' / 'best.pt').resolve())
    jobs = [dict(case={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in scenes[s].items()},
                 out=str(out), ckpt=ckpt, variant='full', gate='mu', envelope=True, rounds=8, horizon=60.0,
                 tag='rl_full_env_B8') for s in names]
    with mp.get_context('spawn').Pool(min(a.workers, len(jobs)), initializer=init_worker) as pool:
        rows = pool.map(run_one, jobs, chunksize=1)
    (out / 'summary.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf8')
    print('ALL DONE', sum(r['status'] == 'arrived' for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
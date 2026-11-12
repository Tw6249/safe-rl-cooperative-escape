"""Step 0 definitive test on seed 20261105 (16 scenes, all using train_full02 weights + envelope rho_i + gate=1):
Compare against saved wfull_gone_B8 (original unsmoothed pre-round algebraic features):
  1. alg_zero : knock out all 12 algebraic features (xa = 0, eaa = 0) in closed loop.
  2. alg_fixed: fix both defects without retraining:
                (a) extract duals AFTER the B=8 NPA rounds at the final negotiated shares s_last,
                (b) apply temporal EMA smoothing (gamma=0.85) to duals, shares s0, and edge ledger z.
"""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
from functools import partial
import multiprocessing as mp
import numpy as np

HERE = Path(__file__).resolve().parent
V = HERE.parents[1]
SARL = V / '20261102_安全强化学习' / 'code'
ENVC = V / '20261103_终段包络验证' / 'code'
sys.path.insert(0, str(SARL))
sys.path.insert(0, str(ENVC))
import sarl_env as E
import evaluate as EV
import eval_env as EE

CK = (V / '20261102_安全强化学习' / 'results' / 'train_full02' / 'best.pt').resolve()
SEED = 20261105
_W = {}


class PostRoundShieldOracle(E.ShieldOracle):
    """Same Filter L (every velocity u and share s is bit-identical to ShieldOracle), except after the B rounds finish,
    we re-evaluate refit_one at the final negotiated shares s_last so self.duals reflects post-negotiation shadow prices."""
    def solve(self, p, nominal):
        if self.count >= EV.MAX_SAMPLES:
            raise EV.StepBudget(f'step budget {EV.MAX_SAMPLES} samples reached')
        u = super().solve(p, nominal)
        if self.s_last is not None:
            from pa_bridge import build_rows
            A, b, part, ws, _, _, _ = build_rows(p, self.lam_eff, self.clip_level)
            self.duals[:] = 0.0
            for a in range(self.n):
                self.refit_one(a, A, b, ws, nominal, self.s_last)
        return u


class Step0Flow(EE.EnvelopedResidualFlow):
    mode = 'alg_zero'  # 'alg_zero' or 'alg_fixed'

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.xa_ema = None
        self.eaa_ema = {}

    def observe(self, p):
        obs, geo = super().observe(p)
        # Fix gate=1 (as in wfull_gone_B8, so only the algebraic features xa/eaa differ)
        obs['gate'] = np.ones_like(obs['gate'])
        obs['x'][:, 2] = 1.0
        obs['ea'][:, 11] = 1.0
        if self.mode == 'alg_zero':
            obs['xa'][:] = 0.0
            obs['eaa'][:] = 0.0
        elif self.mode == 'alg_fixed':
            if self.xa_ema is None:
                self.xa_ema = obs['xa'].copy()
            else:
                self.xa_ema = 0.85 * self.xa_ema + 0.15 * obs['xa']
            obs['xa'] = self.xa_ema.astype(np.float32)
            ei = obs['ei']
            for q in range(ei.shape[1]):
                key = (int(ei[0, q]), int(ei[1, q]))
                prev = self.eaa_ema.get(key)
                cur = obs['eaa'][q].copy()
                val = cur if prev is None else (0.85 * prev + 0.15 * cur)
                self.eaa_ema[key] = val
                obs['eaa'][q] = val
        return obs, geo


def init_worker():
    import sysperf
    sysperf.configure()
    import torch
    torch.set_num_threads(1)
    import sarl_policy as P
    a = P.Actor('full')
    ck = torch.load(str(CK), map_location='cpu')
    a.load_state_dict(ck['actor'])
    a.eval()
    _W.update(actor=a, scenes=E.NCL.C29.scenes_for(SEED))


def run_job(job):
    import pa_bridge
    import sarl_policy as P
    from audit_trajectories import audit
    from pair_common import dump
    out = Path(job['out'])
    d = out / 'runs'
    case = _W['scenes'][job['scene']]
    mode = job['mode']
    tag = f'{mode}_B8'
    if mode == 'alg_fixed':
        pa_bridge.PAOracle = partial(PostRoundShieldOracle, settle='local', diagnostic=False)
    else:
        pa_bridge.PAOracle = partial(EV.budget_oracle(E), settle='local', diagnostic=False)
    Step0Flow.gate_kind = 'mu'
    Step0Flow.mode = mode
    Step0Flow.policy = P.NumpyPolicy(_W['actor'], True)
    pa_bridge.BraidFlow = Step0Flow
    r = pa_bridge.run_pa(case, d, 0.0, 8, E.RADIUS, 60.0, tag, clip_level=True)
    npz = d / f"{case['name']}_{tag}.npz"
    n_seg, bad, md, ml = audit(npz, case['lam']) if r['steps'] else (0, [], None, None)
    r.update(audit_segments=n_seg, audit_failed=len(bad), audit_min_distance=md, audit_min_lambda2=ml,
             final_max_err=float(np.linalg.norm(np.load(npz)['states'][-1] - case['goal'], axis=1).max()),
             tag=tag, mode=mode)
    dump(out / 'rows' / f"{case['name']}_{tag}.json", r)
    print(f"{case['name']:22s} {tag:14s} {r['status']:15s} arr={str(round(r['arrival'],2)) if r['arrival'] else 'None':6s} err={r['final_max_err']*100:5.2f}cm audit_bad={r['audit_failed']} wall={r['wall_seconds']:.0f}s", flush=True)
    return r


if __name__ == '__main__':
    import sysperf
    sysperf.configure()
    out = (HERE.parent / 'results' / 'run01').resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'runs').mkdir(exist_ok=True)
    (out / 'rows').mkdir(exist_ok=True)
    scenes = E.NCL.C29.scenes_for(SEED)
    jobs = [dict(scene=s, mode=m, n=scenes[s]['n'], out=str(out))
            for s in sorted(scenes) for m in ('alg_zero', 'alg_fixed')]
    jobs.sort(key=lambda j: -j['n'])
    print('jobs', len(jobs), flush=True)
    with mp.get_context('spawn').Pool(16, initializer=init_worker) as pool:
        rows = pool.map(run_job, jobs, chunksize=1)
    (out / 'summary.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf8')
    print('ALL DONE', sum(r['status'] == 'arrived' for r in rows), '/', len(rows), flush=True)

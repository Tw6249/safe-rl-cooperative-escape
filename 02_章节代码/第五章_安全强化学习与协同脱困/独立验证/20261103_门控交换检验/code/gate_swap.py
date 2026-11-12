"""Gate-swap test on seed 20261105 (README §2): same weights, gate swapped at execution. Same runner conditions as
20261103_包络公平检验/code/fair_eval.py. Frozen code is imported, never edited."""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
from functools import partial
import argparse, hashlib, json, platform, sys, time
import multiprocessing as mp
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
V = HERE.parents[1]
SARL = V / '20261102_安全强化学习' / 'code'
ENVC = V / '20261103_终段包络验证' / 'code'
sys.path.insert(0, str(SARL))
sys.path.insert(0, str(ENVC))
CK = {v: (V / '20261102_安全强化学习' / 'results' / d / 'best.pt').resolve()
      for v, d in (('full', 'train_full02'), ('geom', 'train_geom02'), ('full_int', 'train_fullint02'))}
SEED = 20261105
GROUPS = {'wfull_gint': dict(variant='full', gate='int'),
          'wfull_gone': dict(variant='full', gate='one'),
          'wfullint_gmu': dict(variant='full_int', gate='mu'),
          'wgeom_gmu': dict(variant='geom', gate='mu'),
          'rl_full_env': dict(variant='full', gate='mu'),            # reproduction of run01 (one scene)
          'rl_full_int_env': dict(variant='full_int', gate='int')}   # reproduction of run01 (one scene)
REPRO_SCENE = 'H12_local2_s20261105'
_W = {}


def init_worker():
    import sysperf
    sysperf.configure()
    import torch
    torch.set_num_threads(1)
    import sarl_env as E
    _W.update(E=E, torch=torch, actors={}, scenes=None)


def actor_for(variant):
    if variant not in _W['actors']:
        import sarl_policy as P
        a = P.Actor(variant)
        ck = _W['torch'].load(str(CK[variant]), map_location='cpu')
        assert ck['variant'] == variant, (ck['variant'], variant)
        a.load_state_dict(ck['actor'])
        a.eval()
        _W['actors'][variant] = a
    return _W['actors'][variant]


def flow_class(gate):
    import eval_env as EE
    if gate != 'one':
        return EE.EnvelopedResidualFlow

    class OneGateFlow(EE.EnvelopedResidualFlow):
        """Gate fixed at 1: the residual multiplier and the gate values the policy reads (node column 2, edge
        column 11 of sarl_env.ResidualFlow.observe) are set to 1. The envelope rho_i still applies."""
        def observe(self, p):
            obs, geo = super().observe(p)
            obs['gate'] = obs['gate'] * 0 + 1
            obs['x'][:, 2] = 1.0
            obs['ea'][:, 11] = 1.0
            return obs, geo
    return OneGateFlow


def run_job(job_out):
    job, out = job_out
    E = _W['E']
    import numpy as np
    import pa_bridge
    import evaluate as EV
    import sarl_policy as P
    from audit_trajectories import audit
    from pair_common import dump
    g = GROUPS[job['group']]
    if _W['scenes'] is None:
        _W['scenes'] = E.NCL.C29.scenes_for(SEED)
    case = _W['scenes'][job['scene']]
    d = Path(out) / 'runs'
    try:
        pa_bridge.PAOracle = partial(EV.budget_oracle(E), settle='local', diagnostic=False)
        cls = flow_class(g['gate'])
        cls.gate_kind = 'mu' if g['gate'] == 'one' else g['gate']
        cls.policy = P.NumpyPolicy(actor_for(g['variant']), True)
        pa_bridge.BraidFlow = cls
        r = pa_bridge.run_pa(case, d, 0.0, 8, E.RADIUS, 60.0, job['tag'], clip_level=True)
        if r['status'] == 'solver_failure' and 'StepBudget' in str(r.get('failure')):
            r['status'] = 'step_budget'
        npz = d / f"{case['name']}_{job['tag']}.npz"
        n_seg, bad, md, ml = audit(npz, case['lam']) if r['steps'] else (0, [], None, None)
        r.update(audit_segments=n_seg, audit_failed=len(bad), audit_min_distance=md, audit_min_lambda2=ml)
        if r['steps']:
            r.update(EV.freeze_stats(npz, np.asarray(case['goal'])),
                     final_max_err=float(np.linalg.norm(np.load(npz)['states'][-1] - case['goal'], axis=1).max()))
    except BaseException as exc:                                     # every failure is kept as a result
        r = dict(scene=case['name'], status='job_error', failure=repr(exc))
    r.update(job=job, tag=job['tag'], n=case['n'])
    dump(Path(out) / 'rows' / f"{case['name']}_{job['tag']}.json", r)
    print(case['name'], job['tag'], r.get('status'), r.get('arrival'), 'audit_bad', r.get('audit_failed'),
          'cuts', r.get('step_cuts'), 'err %.4f' % (r.get('final_max_err') or -1),
          'wall %.0f' % r.get('wall_seconds', 0), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()
    import sysperf
    perf = sysperf.configure()
    import numpy as np
    import sarl_env as E
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for sub in ('runs', 'rows', 'source'):
        (out / sub).mkdir()
    sc = E.NCL.C29.scenes_for(SEED)
    jobs = []
    for gname in ('wfull_gint', 'wfull_gone', 'wfullint_gmu', 'wgeom_gmu'):
        for s in sorted(sc):
            jobs.append(dict(seed=SEED, scene=s, group=gname, tag=f'{gname}_B8', n=sc[s]['n']))
    for gname in ('rl_full_env', 'rl_full_int_env'):
        jobs.append(dict(seed=SEED, scene=REPRO_SCENE, group=gname, tag=f'{gname}_B8', n=sc[REPRO_SCENE]['n']))
    jobs.sort(key=lambda j: -j['n'])
    files = [HERE / 'gate_swap.py', ENVC / 'eval_env.py'] + [SARL / f for f in
             ('evaluate.py', 'sarl_env.py', 'sarl_policy.py', 'sysperf.py')]
    hashes = {}
    for f in files:
        bts = f.read_bytes()
        (out / 'source' / f.name).write_bytes(bts)
        hashes[f.name] = hashlib.sha256(bts).hexdigest()
    for v, pth in CK.items():
        hashes['ckpt/' + v] = hashlib.sha256(pth.read_bytes()).hexdigest()
    (out / 'protocol.json').write_text(json.dumps(dict(
        seed=SEED, groups=GROUPS, repro_scene=REPRO_SCENE, ckpt={k: str(v) for k, v in CK.items()}, eps_g=0.05,
        rounds=8, horizon=60.0, jobs=jobs, source_sha256=hashes, python=sys.version, platform=platform.platform(),
        numpy=np.__version__, process_settings=perf, started=time.strftime('%Y-%m-%d %H:%M:%S')),
        indent=1, ensure_ascii=False, default=str), encoding='utf8')
    print('jobs', len(jobs), flush=True)
    with mp.get_context('spawn').Pool(min(a.workers, len(jobs)), initializer=init_worker) as pool:
        rows = pool.map(run_job, [(j, str(out)) for j in jobs], chunksize=1)
    (out / 'summary.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf8')
    print('ALL DONE', sum(r.get('status') == 'arrived' for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()

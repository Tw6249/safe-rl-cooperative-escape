"""Fair envelope ablation (E2, seed 20261017) and fresh-seed test (seed 20261105), README §2.
Every group goes through the same runner, filter L and audit. Frozen code is imported, never edited."""
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

GROUPS = {'rl_full_env': dict(kind='rl', variant='full', gate='mu', env=True),
          'rl_geom_env': dict(kind='rl', variant='geom', gate='int', env=True),
          'rl_full_int_env': dict(kind='rl', variant='full_int', gate='int', env=True),
          'rl_full': dict(kind='rl', variant='full', gate='mu', env=False),
          'heur': dict(kind='const', gate='int', env=False),
          'myopic': dict(kind='myopic', kappa=0.0)}
PLAN = {'e2': (20261017, ['rl_geom_env', 'rl_full_int_env']),
        'fresh': (20261105, ['rl_full_env', 'rl_geom_env', 'rl_full_int_env', 'rl_full', 'heur', 'myopic'])}
_W = {}


def init_worker():
    import sysperf
    sysperf.configure()
    import torch
    torch.set_num_threads(1)
    import sarl_env as E
    _W.update(E=E, torch=torch, actors={}, scenes={})


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


def run_job(job_out):
    job, out = job_out
    E = _W['E']
    import numpy as np
    import pa_bridge
    import evaluate as EV
    import eval_env as EE
    from audit_trajectories import audit
    from pair_common import dump
    g = GROUPS[job['group']]
    if job['seed'] not in _W['scenes']:
        _W['scenes'][job['seed']] = E.scenes_for(job['seed'])
    case = _W['scenes'][job['seed']][job['scene']]
    d = Path(out) / 'runs'
    try:
        pa_bridge.PAOracle = partial(EV.budget_oracle(E), settle='local', diagnostic=False)
        if g['kind'] == 'myopic':
            pa_bridge.BraidFlow = E.GoalProjectionFlow
            kappa = g['kappa']
        else:
            cls = EE.EnvelopedResidualFlow if g['env'] else E.ResidualFlow
            cls.gate_kind = g['gate']
            if g['kind'] == 'const':
                cls.policy = E.constant_policy(1.0)
            else:
                import sarl_policy as P
                cls.policy = P.NumpyPolicy(actor_for(g['variant']), True)
            pa_bridge.BraidFlow = cls
            kappa = 0.0
        r = pa_bridge.run_pa(case, d, kappa, 8, E.RADIUS, 60.0, job['tag'], clip_level=True)
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
    print(job['set'], case['name'], job['tag'], r.get('status'), r.get('arrival'), 'audit_bad',
          r.get('audit_failed'), 'cuts', r.get('step_cuts'), 'err %.4f' % (r.get('final_max_err') or -1),
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
    jobs, scen = [], {}
    for set_name, (seed, groups) in PLAN.items():
        sc = E.scenes_for(seed)
        scen[set_name] = {k: dict(n=v['n'], perm=v['perm'], lam=v['lam']) for k, v in sc.items()}
        for gname in groups:
            for s in sorted(sc):
                jobs.append(dict(set=set_name, seed=seed, scene=s, group=gname, tag=f'{gname}_B8', n=sc[s]['n']))
    jobs.sort(key=lambda j: -j['n'])                 # large scenes first so the pool drains evenly
    files = [HERE / 'fair_eval.py', ENVC / 'eval_env.py'] + [SARL / f for f in
             ('evaluate.py', 'sarl_env.py', 'sarl_policy.py', 'sysperf.py')]
    hashes = {}
    for f in files:
        bts = f.read_bytes()
        (out / 'source' / f.name).write_bytes(bts)
        hashes[f.name] = hashlib.sha256(bts).hexdigest()
    for v, pth in CK.items():
        hashes['ckpt/' + v] = hashlib.sha256(pth.read_bytes()).hexdigest()
    (out / 'protocol.json').write_text(json.dumps(dict(
        plan=PLAN, groups=GROUPS, ckpt={k: str(v) for k, v in CK.items()}, eps_g=0.05, rounds=8, horizon=60.0,
        jobs=jobs, scenes=scen, source_sha256=hashes, python=sys.version, platform=platform.platform(),
        numpy=np.__version__, process_settings=perf, started=time.strftime('%Y-%m-%d %H:%M:%S')),
        indent=1, ensure_ascii=False, default=str), encoding='utf8')
    print('jobs', len(jobs), flush=True)
    with mp.get_context('spawn').Pool(min(a.workers, len(jobs)), initializer=init_worker) as pool:
        rows = pool.map(run_job, [(j, str(out)) for j in jobs], chunksize=1)
    (out / 'summary.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf8')
    print('ALL DONE', sum(r.get('status') == 'arrived' for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()

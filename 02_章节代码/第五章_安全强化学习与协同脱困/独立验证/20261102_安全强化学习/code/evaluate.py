"""Closed-loop evaluation through the frozen runner pa_bridge.run_pa and the frozen audit (README §5).

    D:\\miniconda\\python.exe -X utf8 -W ignore evaluate.py --out ../results/<new-dir> --set val --groups myopic,heur
        [--rounds 8] [--horizon 60] [--workers 12] [--ckpt full=<path>,geom=<path>,static=<path>] [--only name,...]

Groups (nominal -> filter):
  myopic     GoalProjectionFlow, kappa = 0 (goal projection)                       -> L, B rounds
  heur       ResidualFlow, constant w = 1, r = psi = 0, gate 'int'                  -> L, B rounds
  heur_st    same nominal                                                          -> static split (B = 0)
  rl_full    ResidualFlow, actor 'full' (deterministic), gate 'mu'                  -> L, B rounds
  rl_geom    ResidualFlow, actor 'geom' (deterministic), gate 'int'                 -> L, B rounds
  rl_static  ResidualFlow, actor 'static' (deterministic), gate 'int'               -> static split (B = 0)
Sets: val (seed 20261102), e1 (seed 20261103; both lattice_scenes, N = 8, f in {0.60, 0.85}), e2 (the 16 scenes of
seed 20261017 from scenes.scenes_for, N = 12/20/25, f = 0.6).
"""
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
sys.path.insert(0, str(HERE))

GROUPS = {'myopic': dict(kind='myopic', kappa=0.0, static=False),
          'heur': dict(kind='const', gate='int', static=False), 'heur_st': dict(kind='const', gate='int', static=True),
          'rl_full': dict(kind='rl', variant='full', gate='mu', static=False),
          'rl_geom': dict(kind='rl', variant='geom', gate='int', static=False),
          'rl_static': dict(kind='rl', variant='static', gate='int', static=True),
          'rl_full_int': dict(kind='rl', variant='full_int', gate='int', static=False)}
SETS = dict(val=20261102, dev_val01=20261102, e1=20261103, e2=20261017)


def scene_set(name):
    """val: 24 local + 12 permutation tasks at f = 0.60 (revised 2026-11-02 before formal training, README §3.2);
    e1: 20 local + 12 permutation tasks, each at f = 0.60 and f = 0.90; dev_val01: the first validation design
    (12 + 6 tasks at f = 0.60 and 0.85) used by results/dev_val_baselines01."""
    import sarl_env as E
    if name == 'val':
        return E.lattice_scenes(SETS[name], n=8, n_local=24, n_perm=12, fracs=(0.6,), tag='v')
    if name == 'dev_val01':
        return E.lattice_scenes(SETS['val'], n=8, n_local=12, n_perm=6, fracs=(0.6, 0.85))
    if name == 'e1':
        return E.lattice_scenes(SETS[name], n=8, n_local=20, n_perm=12, fracs=(0.6, 0.9))
    return E.scenes_for(SETS[name]), 0


_W = {}
MAX_SAMPLES = 3600            # step budget: 3 x the nominal 60 s / 0.05 s samples (README §5, added before any test)


class StepBudget(RuntimeError):
    pass


def budget_oracle(E):
    class BudgetOracle(E.ShieldOracle):
        """ShieldOracle that refuses a sample beyond MAX_SAMPLES; run_pa then stops the run (relabelled step_budget).
        Needed because run_pa only bounds physical time: repeated step halving by the certificate can make a run
        practically endless (dev_val_baselines01: two heur_st runs exceeded 60 min of wall time)."""
        def solve(self, p, nominal):
            if self.count >= MAX_SAMPLES:
                raise StepBudget(f'step budget {MAX_SAMPLES} samples reached')
            return super().solve(p, nominal)
    return BudgetOracle


def init_worker():
    import sysperf
    sysperf.configure()
    import torch
    torch.set_num_threads(1)
    import sarl_env as E
    _W.update(E=E, torch=torch, actors={})


def actor_for(path, variant):
    if path not in _W['actors']:
        import sarl_policy as P
        a = P.Actor(variant)
        ck = _W['torch'].load(path, map_location='cpu')
        assert ck['variant'] == variant, (ck['variant'], variant)
        a.load_state_dict(ck['actor'])
        a.eval()
        _W['actors'][path] = a
    return _W['actors'][path]


def freeze_stats(npz, goal):
    import numpy as np
    z = np.load(npz)
    st, ts = z['states'], np.array(z['times'])
    if len(st) < 2:
        return dict(frozen_robot_frac=None, min_pair_distance=None)
    dt = np.diff(ts)[:, None]
    sp = np.linalg.norm(np.diff(st, axis=0), axis=2) / dt
    away = np.linalg.norm(st[:-1] - goal[None], axis=2) > 0.05
    frozen = away & (sp < 0.005)                         # not at goal and speed < 0.01 U (README §5)
    return dict(frozen_robot_frac=float(frozen.sum() / max(away.sum(), 1)),
                frozen_any_frac=float(np.mean(frozen.any(axis=1))),
                away_robot_steps=int(away.sum()))


def run_job(job_out):
    job, out = job_out
    E = _W['E']
    import numpy as np
    import pa_bridge
    from audit_trajectories import audit
    from pair_common import dump
    g = GROUPS[job['group']]
    case = _W['scenes'][job['scene']] if 'scenes' in _W else None
    if case is None:
        _W['scenes'] = scene_set(job['set'])[0]
        case = _W['scenes'][job['scene']]
    d = Path(out) / 'runs'
    rounds = 0 if g['static'] else job['rounds']
    try:
        pa_bridge.PAOracle = partial(budget_oracle(E), settle='local', diagnostic=True)
        if g['kind'] == 'myopic':
            pa_bridge.BraidFlow = E.GoalProjectionFlow
            kappa = g['kappa']
        else:
            pa_bridge.BraidFlow = E.ResidualFlow
            E.ResidualFlow.gate_kind = g['gate']
            if g['kind'] == 'const':
                E.ResidualFlow.policy = E.constant_policy(1.0)
            else:
                import sarl_policy as P
                E.ResidualFlow.policy = P.NumpyPolicy(actor_for(job['ckpt'], g['variant']), True)
            kappa = 0.0
        r = pa_bridge.run_pa(case, d, kappa, rounds, E.RADIUS, job['horizon'], job['tag'], clip_level=True)
        if r['status'] == 'solver_failure' and 'StepBudget' in str(r.get('failure')):
            r['status'] = 'step_budget'
        npz = d / f"{case['name']}_{job['tag']}.npz"
        n_seg, bad, md, ml = audit(npz, case['lam']) if r['steps'] else (0, [], None, None)
        r.update(audit_segments=n_seg, audit_failed=len(bad), audit_min_distance=md, audit_min_lambda2=ml)
        if r['steps']:
            r.update(freeze_stats(npz, np.asarray(case['goal'])))
            tr = json.loads((d / f"{case['name']}_{job['tag']}_trace.json").read_text(encoding='utf8'))
            settled = [x.get('nodes_settled', 0) for x in tr]
            r.update(settle_ticks=int(sum(s > 0 for s in settled)), settle_nodes_total=int(sum(settled)),
                     median_npa_seconds=float(np.median([x.get('npa_seconds', np.nan) for x in tr])),
                     gate_open_mean=float(np.mean([x.get('gate_open', 0) for x in tr])),
                     residual_norm_mean=float(np.mean([x.get('residual_norm', 0.0) for x in tr])),
                     final_max_err=float(np.linalg.norm(np.load(npz)['states'][-1] - case['goal'], axis=1).max()))
    except BaseException as exc:                                     # every failure is kept as a result
        r = dict(scene=case['name'], status='job_error', failure=repr(exc))
    r.update(job=job, n=case['n'], frac=case.get('frac'), kind=case.get('kind'))
    dump(Path(out) / 'rows' / f"{case['name']}_{job['tag']}.json", r)
    print(job['group'], case['name'], job['tag'], r.get('status'), r.get('arrival'), 'audit_bad',
          r.get('audit_failed'), 'cuts', r.get('step_cuts'), 'frozen %.3f' % (r.get('frozen_robot_frac') or 0),
          'wall %.0f' % r.get('wall_seconds', 0), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', required=True, choices=list(SETS))
    ap.add_argument('--groups', required=True)
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--horizon', type=float, default=60.0)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--only', default=None)
    ap.add_argument('--fracs', default=None, help='comma list of spectral levels to keep, e.g. 0.6')
    a = ap.parse_args()
    import sysperf
    perf = sysperf.configure()
    import numpy as np, scipy, cvxpy as cp, torch
    import sarl_env as E
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for sub in ('runs', 'rows', 'source'):
        (out / sub).mkdir()
    ck = dict(kv.split('=', 1) for kv in a.ckpt.split(',') if kv)
    groups = a.groups.split(',')
    scenes, redrawn = scene_set(a.set)
    names = sorted(scenes) if a.only is None else a.only.split(',')
    if a.fracs:
        keep = {float(x) for x in a.fracs.split(',')}
        names = [s for s in names if scenes[s].get('frac') in keep]
    jobs = []
    for gname in groups:
        g = GROUPS[gname]
        if g['kind'] == 'rl':
            assert g['variant'] in ck, f'checkpoint for {g["variant"]} missing'
        for s in names:
            tag = f"{gname}_B{0 if g['static'] else a.rounds}"
            jobs.append(dict(set=a.set, scene=s, group=gname, rounds=a.rounds, horizon=a.horizon, tag=tag,
                             ckpt=str(Path(ck[g['variant']]).resolve()) if g['kind'] == 'rl' else None))
    jobs.sort(key=lambda j: (-scenes[j['scene']]['n'], j['group'] not in ('myopic', 'heur_st')))
    files = [HERE / f for f in ('evaluate.py', 'sarl_env.py', 'sarl_policy.py', 'sysperf.py')]
    hashes = {}
    for f in files:
        bts = f.read_bytes()
        (out / 'source' / f.name).write_bytes(bts)
        hashes[f.name] = hashlib.sha256(bts).hexdigest()
    for v, pth in ck.items():
        hashes['ckpt/' + v] = hashlib.sha256(Path(pth).read_bytes()).hexdigest()
    (out / 'protocol.json').write_text(json.dumps(dict(
        set=a.set, seed=SETS[a.set], redrawn=redrawn, groups=groups, rounds=a.rounds, horizon=a.horizon,
        ckpt={k: str(Path(v).resolve()) for k, v in ck.items()}, jobs=jobs, source_sha256=hashes,
        scenes={k: dict(goal=v['goal'], lam=v['lam'], perm=v['perm'], frac=v.get('frac')) for k, v in scenes.items()
                if k in names},
        python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        cvxpy=cp.__version__, torch=torch.__version__, process_settings=perf,
        started=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1, ensure_ascii=False, default=lambda x: x.tolist()),
        encoding='utf8')
    print('jobs', len(jobs), flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(a.workers, len(jobs)), initializer=init_worker) as pool:
        rows = pool.map(run_job, [(j, str(out)) for j in jobs], chunksize=1)
    (out / 'summary.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf8')
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()

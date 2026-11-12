"""Closed-loop evaluation of this folder (README §3-§5). Same runner, filter and audit as the thesis evidence:
frozen pa_bridge.run_pa, filter L = evaluate.budget_oracle(ShieldOracle) with settle='local', diagnostic=False,
B = 8, 60 s, frozen exact-rational audit. Nothing in this file changes a physical parameter or a controller output.

    python -B -X utf8 eval_ext.py --plan identity|phaseA|phaseC --out ../results/<new-dir> [--workers 8]

Sets:  E1n  lattice_scenes(20261109, N = 8, 20 local + 12 perm, f = 0.6)     (fresh seed)
       E2n  closed_loop.scenes_for(20261110), N = 12/20/25, f = 0.6        (fresh seed)
       SYM  two exactly mirror-symmetric swaps (N = 2 head-on, N = 4 rhombus)
       F05  scenes_for(20261105) (already used; identity check only)
        rl_o2 (same + O2Policy). Any job can run on the mirrored scene (mirror=True).
"""
import paths  # noqa: F401
from pathlib import Path
from functools import partial
import argparse, hashlib, json, platform, sys, time
import multiprocessing as mp

SEEDS = dict(E1n=20261109, E2n=20261110, F05=20261105)
_W = {}


def sym_scenes():
    import numpy as np
    from scenes import lambda2
    S = 1.001
    h = S * np.sqrt(3) / 2
    out = {}
    for name, p, perm in (('S2_headon', np.array([[0., 0.], [S, 0.]]), [1, 0]),
                          ('S4_rhombus', np.array([[0., 0.], [S, 0.], [S / 2, h], [S / 2, -h]]), [1, 0, 2, 3])):
        out[name] = dict(name=name, p=p, goal=p[perm].copy(), lam=round(0.6 * lambda2(p), 3), n=len(p), perm=perm,
                         kind='sym', frac=0.6)
    return out


def scene_set(name):
    import sarl_env as E
    if name == 'E1n':
        return E.lattice_scenes(SEEDS['E1n'], n=8, n_local=20, n_perm=12, fracs=(0.6,), tag='n')[0]
    if name == 'SYM':
        return sym_scenes()
    sc = E.NCL.C29.scenes_for(SEEDS[name])
    for v in sc.values():
        v.setdefault('kind', 'perm' if '_perm' in v['name'] else 'local')
        v.setdefault('frac', 0.6)
    return sc


def init_worker():
    import sysperf
    sysperf.configure()
    import torch
    torch.set_num_threads(1)
    import sarl_env as E
    _W.update(E=E, torch=torch, actors={}, scenes={})


def actor_for(path):
    if path not in _W['actors']:
        import sarl_policy as P
        ck = _W['torch'].load(path, map_location='cpu')
        a = P.Actor(ck['variant'])
        a.load_state_dict(ck['actor'])
        a.eval()
        _W['actors'][path] = a
    return _W['actors'][path]


def run_job(job_out):
    job, out = job_out
    E = _W['E']
    import numpy as np
    import pa_bridge
    import evaluate as EV
    import exp_common as X
    import sarl_policy as P
    from audit_trajectories import audit
    from pair_common import dump
    if job['set'] not in _W['scenes']:
        _W['scenes'][job['set']] = scene_set(job['set'])
    case = dict(_W['scenes'][job['set']][job['scene']])
    if job['mirror']:
        case = X.mirror_case(case)
    d = Path(out) / 'runs'
    t0 = time.time()
    try:
        pa_bridge.PAOracle = partial(EV.budget_oracle(E), settle='local', diagnostic=False)
        if job['group'] == 'myopic':
            pa_bridge.BraidFlow = E.GoalProjectionFlow
            kappa = 0.0
        else:
            cls = X.ExtFlow
            cls.gate_kind = job['gate']
            cls.use_env = job['envelope']
            base = P.NumpyPolicy(actor_for(job['ckpt']), True)
            cls.policy = X.RecPolicy(X.O2Policy(base) if job['group'] == 'rl_o2' else base)
            pa_bridge.BraidFlow = cls
            kappa = 0.0
        r = pa_bridge.run_pa(case, d, kappa, 8, E.RADIUS, 60.0, job['tag'], clip_level=True)
        if r['status'] == 'solver_failure' and 'StepBudget' in str(r.get('failure')):
            r['status'] = 'step_budget'
        npz = d / f"{case['name']}_{job['tag']}.npz"
        n_seg, bad, md, ml = audit(npz, case['lam']) if r['steps'] else (0, [], None, None)
        r.update(audit_segments=n_seg, audit_failed=len(bad), audit_min_distance=md, audit_min_lambda2=ml)
        if r['steps']:
            r.update(EV.freeze_stats(npz, np.asarray(case['goal'])))
            r.update(X.smoothness(npz))
            r['final_max_err'] = float(np.linalg.norm(np.load(npz)['states'][-1] - case['goal'], axis=1).max())
            if job['group'].startswith('rl'):
                tr = json.loads((d / f"{case['name']}_{job['tag']}_trace.json").read_text(encoding='utf8'))
                for key in ('interv_mean', 'nom_mean', 'nomc_mean', 'sat_w', 'sat_r', 'sat_psi', 'absw_mean',
                            'w_mean', 'lat_nom_moving', 'lat_u_moving', 'gate_mean'):
                    vals = [x[key] for x in tr if key in x]
                    r['tr_' + key] = float(np.mean(vals)) if vals else None
    except BaseException as exc:                                    # every failure is kept as a result
        r = dict(scene=case['name'], status='job_error', failure=repr(exc))
    r.update(job=job, tag=job['tag'], n=case['n'], kind=case.get('kind'), base_scene=job['scene'],
             mirror=job['mirror'], wall_job=time.time() - t0)
    dump(Path(out) / 'rows' / f"{case['name']}_{job['tag']}.json", r)
    print(job['set'], case['name'], job['tag'], r.get('status'), r.get('arrival'), 'audit_bad', r.get('audit_failed'),
          'cuts', r.get('step_cuts'), 'err %.4f' % (r.get('final_max_err') or -1), 'wall %.0f' % r['wall_job'],
          flush=True)
    return r


def ckpts_from(spec):
    return dict(kv.split('=', 1) for kv in spec.split(',') if kv)


def plan_jobs(plan, ck):
    """ck: label -> checkpoint path. Returns the job list of a plan (README §4)."""
    T = str(paths.THESIS_CKPT)
    jobs = []

    def add(set_name, groups, mirror=False):
        for s in sorted(scene_set(set_name)):
            for g, label, path in groups:
                tag = f"{label}_B8"
                jobs.append(dict(set=set_name, scene=s, group=g, label=label, tag=tag, mirror=mirror, ckpt=path,
                                 gate='mu', envelope=True))

    if plan == 'identity':
        jobs = [dict(set='F05', scene='H12_local2_s20261105', group='rl', label='rl_full_env', tag='rl_full_env_B8',
                     mirror=False, ckpt=T, gate='mu', envelope=True)]
    elif plan == 'phaseA':
        add('SYM', [('myopic', 'myopic', None), ('rl', 'thesis', T), ('rl_o2', 'thesis_o2', T)])
        for st in ('E1n', 'E2n'):
            add(st, [('rl', 'thesis', T), ('rl_o2', 'thesis_o2', T)])
            add(st, [('rl', 'thesis', T)], mirror=True)
        add('E1n', [('rl_o2', 'thesis_o2', T)], mirror=True)
    elif plan == 'phaseC':                     # README §6 step 4 (renamed from 'phaseB' after phaseA started)
        groups = [('rl', label, str(Path(p).resolve())) for label, p in sorted(ck.items())]
        for st in ('E1n', 'E2n'):
            add(st, groups)
    else:
        raise ValueError(plan)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ckpt', default='', help='phaseC: label=path,...')
    a = ap.parse_args()
    import sysperf
    perf = sysperf.configure()
    import numpy as np, scipy, cvxpy as cp, torch
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for sub in ('runs', 'rows', 'source'):
        (out / sub).mkdir()
    ck = ckpts_from(a.ckpt)
    jobs = plan_jobs(a.plan, ck)
    sizes = {}
    for j in jobs:
        sizes.setdefault(j['set'], {})
    for st in sizes:
        sizes[st] = {k: v['n'] for k, v in scene_set(st).items()}
    jobs.sort(key=lambda j: -sizes[j['set']][j['scene']])          # largest scenes first
    files = [paths.HERE / f for f in ('eval_ext.py', 'exp_common.py', 'paths.py')] + \
            [paths.SARL / f for f in ('evaluate.py', 'sarl_env.py', 'sarl_policy.py', 'sysperf.py')] + \
            [paths.C5 / '独立验证' / '20261103_终段包络验证' / 'code' / 'eval_env.py']
    hashes = {}
    for f in files:
        bts = f.read_bytes()
        (out / 'source' / f.name).write_bytes(bts)
        hashes[f.name] = hashlib.sha256(bts).hexdigest()
    for p in sorted({j['ckpt'] for j in jobs if j['ckpt']}):
        hashes['ckpt:' + p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (out / 'protocol.json').write_text(json.dumps(dict(
        plan=a.plan, seeds=SEEDS, jobs=jobs, source_sha256=hashes, rounds=8, horizon=60.0, eps_g=0.05,
        python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        cvxpy=cp.__version__, torch=torch.__version__, process_settings=perf, workers=a.workers,
        started=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1, ensure_ascii=False, default=str), encoding='utf8')
    print('jobs', len(jobs), flush=True)
    with mp.get_context('spawn').Pool(min(a.workers, len(jobs)), initializer=init_worker) as pool:
        rows = pool.map(run_job, [(j, str(out)) for j in jobs], chunksize=1)
    (out / 'summary.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf8')
    (out / 'done.json').write_text(json.dumps(dict(finished=time.strftime('%Y-%m-%d %H:%M:%S'), jobs=len(rows))),
                                   encoding='utf8')
    print('ALL DONE', sum(r.get('status') == 'arrived' for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()

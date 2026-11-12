"""MAPPO training of the gated residual nominal behind filter L (README §4-§6).

    D:\\miniconda\\python.exe -X utf8 -W ignore train_ppo.py --variant full --out ../results/<new-dir>

Every episode runs the frozen pipeline: ResidualFlow nominal -> ShieldOracle (filter L, B rounds; B = 0 is the static
split) -> exact-rational certified_step. Rollout workers are single-threaded CPU processes; the update runs on the GPU.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import argparse, hashlib, json, platform, sys, time
import multiprocessing as mp
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TRAIN_SEED, VAL_SEED = 20261102, 20261102          # training stream ([seed, iteration, slot]); validation set seed
HP = dict(variants={'full': dict(gate='mu', rounds=2), 'geom': dict(gate='int', rounds=2),
                    'static': dict(gate='int', rounds=0),
                    'full_int': dict(gate='int', rounds=2)},     # extra ablation, README §8.4

          episodes_per_iter=24, tmax=500, val_tmax=600, budget_steps=960_000, val_every=8,
          gamma=0.995, lam=0.95, lr=1e-3, epochs=5, minibatches=8, clip=0.2, target_kl=0.03, vf_coef=0.5,
          max_grad_norm=0.5, std0=0.4, gate_mask=0.01,
          reward=dict(progress=1.0, time=1e-3, stall=0.02, success=1.0, eps_p=0.1, eps_u=0.05,
                      stall_u=0.005, stall_steps=60, success_tol=0.01),
          val=dict(n=8, n_local=24, n_perm=12, fracs=[0.6], tag='v'))

# ------------------------------------------------------------------------------------------------- rollout worker
_W = {}


def init_worker(variant):
    import sysperf
    sysperf.configure()
    import torch
    torch.set_num_threads(1)
    import sarl_env as E
    import sarl_policy as P
    from pair_common import certified_step
    _W.update(E=E, P=P, torch=torch, cert=certified_step, variant=variant, oracles={},
              actor=P.Actor(variant, HP['std0']))


def run_episode(task):
    E, P, torch, cert = _W['E'], _W['P'], _W['torch'], _W['cert']
    import numpy as np
    variant = _W['variant']
    cfg = HP['variants'][variant]
    rw = HP['reward']
    actor = _W['actor']
    actor.load_state_dict(task['state'])
    actor.eval()
    scene = task['scene'] if task.get('scene') is not None else E.train_scene(
        np.random.default_rng(task['scene_seed']), tag=f"_{task['scene_seed'][1]}_{task['scene_seed'][2]}")
    n = scene['n']
    flow = E.ResidualFlow(scene, 0.05, 0.0, 0.02, False)
    flow.gate_kind = cfg['gate']
    key = (n, cfg['rounds'])
    orc = _W['oracles'].get(key)
    if orc is None:
        orc = _W['oracles'][key] = E.ShieldOracle(n, scene['lam'] + flow.margin, cfg['rounds'], E.RADIUS,
                                                  clip_level=True, settle='local', diagnostic=False)
    orc.reset(scene['lam'] + flow.margin)
    flow.oracle = orc
    pol = P.NumpyPolicy(actor, task['det'], np.random.default_rng(task['seed']))
    E.ResidualFlow.policy = pol
    g = scene['goal']
    p = scene['p'].copy()
    D0 = float(np.linalg.norm(p - g, axis=1).sum())
    steps, quiet, reason, success_t = [], 0, 'truncated', None
    min_d, min_l, cuts_total, settle_nodes = np.inf, np.inf, 0, 0
    t0 = time.perf_counter()
    for k in range(task['tmax']):
        try:
            u, info = flow.control(p)
        except Exception as exc:
            reason = 'filter:' + type(exc).__name__
            if pol.record and len(pol.record) > len(steps):
                pol.record.pop()
            if steps:
                steps[-1]['done'] = 1.0
            break
        obs = flow.last_obs
        mu = np.linalg.norm(orc.duals, axis=1)
        settle_nodes += orc.last.get('nodes_settled', 0)
        try:
            z, dt, cuts, bnd = cert(p, u, scene['lam'], 0.05)
        except Exception:
            reason = 'certification_stop'
            steps.append(dict(obs=obs, r=0.0, done=1.0))
            break
        cuts_total += cuts
        min_d, min_l = min(min_d, bnd['min_distance']), min(min_l, bnd['lambda2_lower'])
        e0 = np.linalg.norm(p - g, axis=1)
        e1 = np.linalg.norm(z - g, axis=1)
        un = np.linalg.norm(u, axis=1)
        stall = np.tanh(e0 / rw['eps_p']) * np.exp(-un ** 2 / rw['eps_u'] ** 2) * np.log1p(mu ** 2)
        r = rw['progress'] * (e0.sum() - e1.sum()) / D0 - rw['time'] - rw['stall'] * float(stall.mean())
        done = 0.0
        if e1.max() <= rw['success_tol']:
            r += rw['success']
            done, reason, success_t = 1.0, 'success', (k + 1) * 0.05
        quiet = quiet + 1 if un.max() < rw['stall_u'] else 0
        if not done and quiet >= rw['stall_steps']:
            done, reason = 1.0, 'stall'
        steps.append(dict(obs=obs, r=float(r), done=done))
        p = z
        if done:
            break
    boot = None
    if reason == 'truncated':
        try:
            boot, _ = flow.observe(p)
        except Exception:
            reason = 'truncated_noboot'
            if steps:
                steps[-1]['done'] = 1.0
    stats = dict(scene=scene['name'], n=n, kind=scene.get('kind'), frac=scene.get('frac'), reason=reason,
                 success=reason == 'success', success_time=success_t, steps=len(steps),
                 ret=float(sum(s['r'] for s in steps)), final_err_ratio=float(np.linalg.norm(p - g, axis=1).sum() / D0),
                 max_final_err=float(np.linalg.norm(p - g, axis=1).max()), min_distance=float(min_d),
                 min_lambda2_lower=float(min_l), lam=scene['lam'], cuts=int(cuts_total), settle_nodes=int(settle_nodes),
                 wall=time.perf_counter() - t0)
    if task['det']:
        return dict(stats=stats)
    rec = pol.record[:len(steps)]
    return dict(stats=stats, steps=steps, rec=rec, boot=boot)


# ------------------------------------------------------------------------------------------------- learner
def assemble(results, variant, device):
    import numpy as np, torch
    from sarl_policy import actor_inputs
    X, XA, EA, EAA, EI, GATE, GLOB, AN, AE, LP = [], [], [], [], [], [], [], [], [], []
    ncount, ecount, R, DONE, EP = [], [], [], [], []
    boots = []
    off = 0
    for epi, res in enumerate(results):
        for s, rc in zip(res['steps'], res['rec'] + [None] * (len(res['steps']) - len(res['rec']))):
            o = s['obs']
            n, E_ = len(o['x']), o['ei'].shape[1]
            X.append(o['x']); XA.append(o['xa']); EA.append(o['ea']); EAA.append(o['eaa'])
            EI.append(o['ei'] + off); GATE.append(o['gate']); GLOB.append(o['glob'])
            if rc is None:                                   # certification-stop step: action not executed
                AN.append(np.zeros((n, 2), np.float32)); AE.append(np.zeros(E_, np.float32))
                LP.append(np.zeros(n, np.float32)); GATE[-1] = np.zeros(n, np.float32)
            else:
                AN.append(rc['an']); AE.append(rc['ae']); LP.append(rc['logp'].astype(np.float32))
            ncount.append(n); ecount.append(E_); R.append(s['r']); DONE.append(s['done']); EP.append(epi)
            off += n
        boots.append(res['boot'])
    t = lambda a, dt=torch.float32: torch.as_tensor(np.concatenate(a) if isinstance(a, list) else a, dtype=dt,
                                                    device=device)
    B = dict(x=t(X), xa=t(XA), ea=t(EA), eaa=t(EAA), ei=torch.as_tensor(np.concatenate(EI, 1), dtype=torch.long,
                                                                        device=device),
             gate=t(GATE), glob=torch.as_tensor(np.stack(GLOB), dtype=torch.float32, device=device),
             an=t(AN), ae=t(AE), lp=t(LP), ncount=torch.as_tensor(ncount, device=device),
             ecount=torch.as_tensor(ecount, device=device), r=np.array(R), done=np.array(DONE), ep=np.array(EP))
    B['nstart'] = torch.cumsum(B['ncount'], 0) - B['ncount']
    B['estart'] = torch.cumsum(B['ecount'], 0) - B['ecount']
    return B, boots


def gather(B, gid, variant):
    import torch
    nc, ec = B['ncount'][gid], B['ecount'][gid]
    G = len(gid)
    ar = torch.arange(G, device=gid.device)
    nb = torch.repeat_interleave(ar, nc)
    eb = torch.repeat_interleave(ar, ec)
    new_ns = torch.cumsum(nc, 0) - nc
    new_es = torch.cumsum(ec, 0) - ec
    nidx = B['nstart'][gid][nb] + (torch.arange(int(nc.sum()), device=gid.device) - new_ns[nb])
    eidx = B['estart'][gid][eb] + (torch.arange(int(ec.sum()), device=gid.device) - new_es[eb])
    ei = B['ei'][:, eidx] - B['nstart'][gid][eb] + new_ns[eb]
    xc = torch.cat([B['x'][nidx], B['xa'][nidx]], 1)
    eac = torch.cat([B['ea'][eidx], B['eaa'][eidx]], 1)
    if variant in ('full', 'full_int'):
        xa, eaa = xc, eac
    else:
        xa, eaa = B['x'][nidx], B['ea'][eidx]
    return dict(xa=xa, eaa=eaa, xc=xc, eac=eac, ei=ei, nb=nb, G=G, glob=B['glob'][gid], an=B['an'][nidx],
                ae=B['ae'][eidx], lp=B['lp'][nidx], gate=B['gate'][nidx])


def boot_values(critic, boots, device):
    import numpy as np, torch
    out = []
    for o in boots:
        if o is None:
            out.append(0.0)
            continue
        x = torch.as_tensor(np.concatenate([o['x'], o['xa']], 1), device=device)
        ea = torch.as_tensor(np.concatenate([o['ea'], o['eaa']], 1), device=device)
        ei = torch.as_tensor(o['ei'], dtype=torch.long, device=device)
        nb = torch.zeros(len(o['x']), dtype=torch.long, device=device)
        g = torch.as_tensor(o['glob'][None], device=device)
        out.append(float(critic(x, ei, ea, nb, g, 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=list(HP['variants']))
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--budget', type=int, default=HP['budget_steps'])
    ap.add_argument('--iter-episodes', type=int, default=HP['episodes_per_iter'])
    ap.add_argument('--val-every', type=int, default=HP['val_every'])
    ap.add_argument('--smoke', action='store_true', help='tiny run for code checks (not evidence)')
    a = ap.parse_args()
    import sysperf
    perf = sysperf.configure()
    import numpy as np, torch
    import sarl_env as E
    import sarl_policy as P
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source').mkdir()
    files = [HERE / 'train_ppo.py', HERE / 'sarl_env.py', HERE / 'sarl_policy.py', HERE / 'sysperf.py']
    hashes = {}
    for f in files:
        bts = f.read_bytes()
        (out / 'source' / f.name).write_bytes(bts)
        hashes[f.name] = hashlib.sha256(bts).hexdigest()
    frozen = [E.BASE.parent / '基础依赖' / 'code' / 'ledger_npa.py',
              E.BASE.parent / '基础依赖' / 'code' / 'block_npa.py',
              E.BASE.parent / '基础依赖' / 'code' / 'npa_oracle.py',
              E.BASE.parent / '基础依赖' / 'code' / 'pa_bridge.py',
              E.BASE.parent / '基础依赖' / 'code' / 'pair_common.py',
              E.BASE.parent / '基础依赖' / 'code' / 'safety_checks.py']
    for f in frozen:
        hashes['frozen/' + f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(TRAIN_SEED)
    actor = P.Actor(a.variant, HP['std0']).to(device)
    critic = P.Critic().to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=HP['lr'], eps=1e-5)
    val_scenes, redrawn = E.lattice_scenes(VAL_SEED, **{k: v for k, v in HP['val'].items() if k != 'fracs'},
                                           fracs=tuple(HP['val']['fracs']))
    val_list = list(val_scenes.values())
    if a.smoke:
        val_list = val_list[:4]
    meta = dict(variant=a.variant, hp=HP, workers=a.workers, budget=a.budget, iter_episodes=a.iter_episodes,
                train_seed=TRAIN_SEED, val_seed=VAL_SEED, val_scenes=[s['name'] for s in val_list],
                val_redrawn=redrawn, source_sha256=hashes, device=str(device),
                python=sys.version, platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                started=time.strftime('%Y-%m-%d %H:%M:%S'), smoke=a.smoke, process_settings=perf)
    (out / 'protocol.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf8')
    log = open(out / 'train_log.jsonl', 'a', encoding='utf8')
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(a.workers, initializer=init_worker, initargs=(a.variant,))
    total, it, best = 0, 0, None
    gam, lam = HP['gamma'], HP['lam']

    def validate(tag_it):
        sd_ = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        vt = [dict(state=sd_, scene=s, seed=[VAL_SEED, 0], det=True, tmax=HP['val_tmax']) for s in val_list]
        vres = [r['stats'] for r in pool.map(run_episode, vt, chunksize=1)]
        succ = [s['success'] for s in vres]
        times = [s['success_time'] for s in vres if s['success']]
        score = float(np.mean(succ)) - 0.001 * (float(np.mean(times)) if times else 30.0)
        (out / f'val_{tag_it:04d}.json').write_text(json.dumps(vres, indent=1), encoding='utf8')
        return vres, dict(success=float(np.mean(succ)), mean_time=float(np.mean(times)) if times else None,
                          score=score,
                          by_frac={str(f): float(np.mean([s['success'] for s in vres if s['frac'] == f]))
                                   for f in HP['val']['fracs']},
                          reasons={k: sum(s['reason'] == k for s in vres) for k in sorted(set(s['reason'] for s in vres))},
                          min_distance=float(min(s['min_distance'] for s in vres)),
                          min_lambda2_margin=float(min(s['min_lambda2_lower'] - s['lam'] for s in vres)))

    try:
        _, v0 = validate(9999)                      # initial (zero-mean) policy = goal projection; logged as iter -1
        best = (v0['score'], -1)
        torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), iter=-1, variant=a.variant),
                   out / 'best.pt')
        log.write(json.dumps(dict(iter=-1, env_steps=0, val=v0, best_iter=-1)) + '\n')
        log.flush()
        print(f"it  -1 initial policy | VAL {v0['success']:.3f} t={v0['mean_time']} {v0['by_frac']}", flush=True)
        while total < a.budget:
            t_it = time.perf_counter()
            sd = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
            tasks = [dict(state=sd, scene=None, scene_seed=[TRAIN_SEED, it, k], seed=[TRAIN_SEED, it, k, 1], det=False,
                          tmax=HP['tmax']) for k in range(a.iter_episodes)]
            results = pool.map(run_episode, tasks, chunksize=1)
            t_roll = time.perf_counter() - t_it
            results = [r for r in results if r['steps']]
            B, boots = assemble(results, a.variant, device)
            Gn = len(B['r'])
            with torch.no_grad():
                gid = torch.arange(Gn, device=device)
                V = torch.zeros(Gn, device=device)
                for ch in torch.split(gid, 4096):
                    m = gather(B, ch, a.variant)
                    V[ch] = critic(m['xc'], m['ei'], m['eac'], m['nb'], m['glob'], m['G'])
                V = V.cpu().numpy()
            vb = boot_values(critic, boots, device)
            adv = np.zeros(Gn)
            ret = np.zeros(Gn)
            for epi in range(len(results)):
                idx = np.flatnonzero(B['ep'] == epi)
                last_gae, next_v = 0.0, vb[epi]
                for q in idx[::-1]:
                    nonterm = 1.0 - B['done'][q]
                    delta = B['r'][q] + gam * next_v * nonterm - V[q]
                    last_gae = delta + gam * lam * nonterm * last_gae
                    adv[q] = last_gae
                    next_v = V[q]
                ret[idx] = adv[idx] + V[idx]
            advn = (adv - adv.mean()) / (adv.std() + 1e-8)
            A_t = torch.as_tensor(advn, dtype=torch.float32, device=device)
            R_t = torch.as_tensor(ret, dtype=torch.float32, device=device)
            kls, pls, vls, clipf, stop_epoch = [], [], [], [], None
            for ep in range(HP['epochs']):
                perm = torch.randperm(Gn, device=device)
                ep_kl = []
                for ch in torch.chunk(perm, HP['minibatches']):
                    m = gather(B, ch, a.variant)
                    lp_new = actor.log_prob(m['xa'], m['ei'], m['eaa'], m['an'], m['ae'])
                    mask = (m['gate'] > HP['gate_mask']).float()
                    ratio = torch.exp(lp_new - m['lp'])
                    An = A_t[ch][m['nb']]
                    pg = -torch.min(ratio * An, torch.clamp(ratio, 1 - HP['clip'], 1 + HP['clip']) * An)
                    lpi = (pg * mask).sum() / mask.sum().clamp(min=1.0)
                    v = critic(m['xc'], m['ei'], m['eac'], m['nb'], m['glob'], m['G'])
                    lv = ((v - R_t[ch]) ** 2).mean()
                    loss = lpi + HP['vf_coef'] * lv
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), HP['max_grad_norm'])     # separately, so that
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), HP['max_grad_norm'])    # the critic cannot shrink the actor step
                    opt.step()
                    with torch.no_grad():
                        lr_ = lp_new - m['lp']
                        kl = float((((torch.exp(lr_) - 1) - lr_) * mask).sum() / mask.sum().clamp(min=1.0))
                        ep_kl.append(kl)
                        clipf.append(float((((ratio - 1).abs() > HP['clip']).float() * mask).sum()
                                           / mask.sum().clamp(min=1.0)))
                    pls.append(float(lpi)); vls.append(float(lv))
                kls.append(float(np.mean(ep_kl)))
                if kls[-1] > HP['target_kl']:
                    stop_epoch = ep
                    break
            steps_it = sum(r['stats']['steps'] for r in results)
            total += steps_it
            st = [r['stats'] for r in results]
            sn, se = actor.stds()
            row = dict(iter=it, env_steps=total, steps=steps_it, episodes=len(st), rollout_s=t_roll,
                       iter_s=time.perf_counter() - t_it,
                       success=float(np.mean([s['success'] for s in st])),
                       reasons={k: sum(s['reason'] == k for s in st) for k in sorted(set(s['reason'] for s in st))},
                       ret=float(np.mean([s['ret'] for s in st])),
                       err_ratio=float(np.mean([s['final_err_ratio'] for s in st])),
                       min_distance=float(min(s['min_distance'] for s in st)),
                       min_lambda2_margin=float(min(s['min_lambda2_lower'] - s['lam'] for s in st)),
                       cuts=int(sum(s['cuts'] for s in st)), kl=kls, stop_epoch=stop_epoch,
                       pi_loss=float(np.mean(pls)), v_loss=float(np.mean(vls)), clipfrac=float(np.mean(clipf)),
                       std_node=sn.detach().cpu().tolist(), std_edge=se.detach().cpu().tolist(),
                       gate_frac=float((B['gate'] > HP['gate_mask']).float().mean()))
            if (it + 1) % a.val_every == 0 or total >= a.budget:
                _, row['val'] = validate(it)
                score = row['val']['score']
                if best is None or score > best[0]:
                    best = (score, it)
                    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), iter=it, variant=a.variant),
                               out / 'best.pt')
                row['best_iter'] = best[1]
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), opt=opt.state_dict(), iter=it,
                            variant=a.variant, env_steps=total), out / 'last.pt')
            if it % 8 == 0:
                torch.save(dict(actor=actor.state_dict(), iter=it, variant=a.variant), out / f'ckpt_{it:04d}.pt')
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(f"it {it:3d} steps {total:7d} succ {row['success']:.2f} ret {row['ret']:+.3f} "
                  f"reasons {row['reasons']} kl {kls[-1]:.4f} std {row['std_node']} gate {row['gate_frac']:.2f} "
                  f"roll {t_roll:.0f}s it {row['iter_s']:.0f}s" + (f" | VAL {row['val']['success']:.3f} "
                  f"t={row['val']['mean_time']} {row['val']['by_frac']}" if 'val' in row else ''), flush=True)
            it += 1
            if a.smoke and it >= 2:
                break
    finally:
        pool.close()
        pool.join()
        log.close()
    (out / 'done.json').write_text(json.dumps(dict(iters=it, env_steps=total, best=best,
                                                   finished=time.strftime('%Y-%m-%d %H:%M:%S'))), encoding='utf8')
    print('DONE', it, total, best, flush=True)


if __name__ == '__main__':
    main()

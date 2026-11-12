"""MAPPO of the thesis (20261102_安全强化学习/code/train_ppo.py, variant 'full', gate 'mu', B = 2) with two switches
from 独立测试.md (README §3 of this folder):

  --critic local   upgrade 1: closed-neighbourhood critic. Per-robot value V_i (same GNN, no pooling) fitted to the
                   block reward R_i = mean_{j in N_i^+} r_j (N_i^+ = closed 2.5 m neighbourhood, the graph of filter L),
                   per-robot GAE, per-robot advantage in the PPO ratio. r_j is the per-robot split of the team reward
                   (mean_j r_j == team reward exactly).
  --proj LAMBDA    upgrade 2: shield-projection regulariser added to the actor loss,
                   LAMBDA * mean_i || Pi_U(u_nom_i(mu_theta)) - stop_grad(u_i) ||^2,
                   u_nom from the policy mean through the same composition as sarl_env.ResidualFlow.compose
                   (straight-through clip), u_i = executed filter output of that sample, Pi_U = speed-ball projection.

With --critic team --proj 0 the update is the thesis update (same HP, same code path). Rollouts, filter, certificate,
validation set and model-selection rule are unchanged; only the training seed is new (--seed).

    python -B -X utf8 -W ignore train_ext.py --seed 20261107 --critic team --proj 0 --out ../results/train_<name>
"""
import paths  # noqa: F401
from pathlib import Path
import argparse, hashlib, json, platform, sys, time
import multiprocessing as mp

VAL_SEED = 20261102                    # thesis validation set (development data, used for model selection only)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('train_ppo_frozen', str(paths.SARL / 'train_ppo.py'))
TP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(TP)           # frozen thesis trainer: HP dictionary, assemble/gather/boot helpers
HP = TP.HP

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
    """train_ppo.run_episode, plus per-robot reward split, positions, executed velocities and intervention."""
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
    min_d, min_l, cuts_total, settle_nodes, interv = np.inf, np.inf, 0, 0, []
    t0 = time.perf_counter()
    for k in range(task['tmax']):
        p_k = p.copy()
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
        interv.append(float(np.linalg.norm(flow.delta_prev, axis=1).mean()))
        mu = np.linalg.norm(orc.duals, axis=1)
        settle_nodes += orc.last.get('nodes_settled', 0)
        try:
            z, dt, cuts, bnd = cert(p, u, scene['lam'], 0.05)
        except Exception:
            reason = 'certification_stop'
            steps.append(dict(obs=obs, r=0.0, done=1.0, ri=np.zeros(n, np.float32),
                              pos=p_k.astype(np.float32), u=u.astype(np.float32)))
            break
        cuts_total += cuts
        min_d, min_l = min(min_d, bnd['min_distance']), min(min_l, bnd['lambda2_lower'])
        e0 = np.linalg.norm(p - g, axis=1)
        e1 = np.linalg.norm(z - g, axis=1)
        un = np.linalg.norm(u, axis=1)
        stall = np.tanh(e0 / rw['eps_p']) * np.exp(-un ** 2 / rw['eps_u'] ** 2) * np.log1p(mu ** 2)
        r = rw['progress'] * (e0.sum() - e1.sum()) / D0 - rw['time'] - rw['stall'] * float(stall.mean())
        ri = rw['progress'] * n * (e0 - e1) / D0 - rw['time'] - rw['stall'] * stall     # mean_j ri_j == r
        done = 0.0
        if e1.max() <= rw['success_tol']:
            r += rw['success']
            ri = ri + rw['success']
            done, reason, success_t = 1.0, 'success', (k + 1) * 0.05
        quiet = quiet + 1 if un.max() < rw['stall_u'] else 0
        if not done and quiet >= rw['stall_steps']:
            done, reason = 1.0, 'stall'
        steps.append(dict(obs=obs, r=float(r), done=done, ri=ri.astype(np.float32), pos=p_k.astype(np.float32),
                          u=u.astype(np.float32)))
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
                 interv_mean=float(np.mean(interv)) if interv else None, wall=time.perf_counter() - t0)
    if task['det']:
        return dict(stats=stats)
    rec = pol.record[:len(steps)]
    return dict(stats=stats, steps=steps, rec=rec, boot=boot, goal=g.astype(np.float32))


# ------------------------------------------------------------------------------------------------- learner extras
def build_critic(kind):
    import torch
    import torch.nn as nn
    import sarl_policy as P

    class NodeCritic(nn.Module):
        """Upgrade 1: same GNN as the thesis critic, per-robot value head (no pooling): V_i = f(h_i, glob)."""
        def __init__(self):
            super().__init__()
            fn, fe = P.NODE_GEOM + P.NODE_ALG, P.EDGE_GEOM + P.EDGE_ALG
            self.gnn = P.GNN(fn, fe)
            self.head = nn.Sequential(nn.Linear(P.H + 4, P.H), nn.SiLU(), nn.Linear(P.H, P.H), nn.SiLU(),
                                      nn.Linear(P.H, 1))

        def forward(self, x, ei, ea, batch, glob, n_graphs):
            h = self.gnn(x, ei, ea)
            return self.head(torch.cat([h, glob[batch]], 1)).squeeze(-1)

    return P.Critic() if kind == 'team' else NodeCritic()


def extras(results):
    """Per-node arrays aligned with assemble(): individual rewards, positions, goals, executed velocities, and the
    block reward R_i = (r_i + sum_{j in N_i} r_j) / (1 + |N_i|) on the observation graph of the same sample."""
    import numpy as np
    RI, POS, GOAL, UEX, RB = [], [], [], [], []
    for res in results:
        for s in res['steps']:
            o = s['obs']
            n = len(o['x'])
            I, J = o['ei'][0], o['ei'][1]
            ri = s['ri'].astype(np.float64)
            num = ri + np.bincount(I, weights=ri[J], minlength=n)
            den = 1.0 + np.bincount(I, minlength=n)
            RI.append(ri); RB.append(num / den)
            POS.append(s['pos']); GOAL.append(res['goal']); UEX.append(s['u'])
    return (np.concatenate(RI), np.concatenate(RB), np.concatenate(POS), np.concatenate(GOAL), np.concatenate(UEX))


def proj_loss(actor_out, m, POS_t, GOAL_t, UEX_t, nidx, E):
    """Upgrade 2 regulariser on one minibatch. actor_out = (mu_n, mu_e) of the minibatch graphs."""
    import torch
    mu_n, mu_e = actor_out
    st = lambda v: v + (torch.clamp(v, -1.0, 1.0) - v).detach()          # straight-through clip
    w, rr, psi = st(mu_n[:, 0]), st(mu_n[:, 1]), st(mu_e)
    p = POS_t[nidx]
    e = p - GOAL_t[nidx]
    Je = torch.stack([-e[:, 1], e[:, 0]], 1)
    src, dst = m['ei'][0], m['ei'][1]
    dv = p[src] - p[dst]
    d = torch.linalg.norm(dv, dim=1).clamp(min=1e-9)
    away = dv / d[:, None]
    nn_ = p.shape[0]
    pb = torch.zeros_like(e).index_add_(0, src, psi[:, None] * away)
    deg = torch.zeros(nn_, device=p.device).index_add_(0, src, torch.ones_like(d))
    pb = pb / deg.clamp(min=1.0)[:, None]
    res = E.A_ROT * w[:, None] * Je + E.A_RAD * rr[:, None] * e + E.A_PSI * E.U * pb
    nom = -e + m['gate'][:, None] * res
    nrm = torch.linalg.norm(nom, dim=1, keepdim=True).clamp(min=1e-12)
    nomc = nom * torch.clamp(E.U / nrm, max=1.0)
    return ((nomc - UEX_t[nidx]) ** 2).sum(1).mean()


def gather_idx(B, gid):
    import torch
    nc = B['ncount'][gid]
    G = len(gid)
    ar = torch.arange(G, device=gid.device)
    nb = torch.repeat_interleave(ar, nc)
    new_ns = torch.cumsum(nc, 0) - nc
    return B['nstart'][gid][nb] + (torch.arange(int(nc.sum()), device=gid.device) - new_ns[nb])


def actor_forward_logp(actor, x, ei, ea, an, ae):
    """Actor.log_prob, also returning the means (one forward pass)."""
    import torch
    mu_n, mu_e = actor(x, ei, ea)
    sn, se = actor.stds()
    lpn = torch.distributions.Normal(mu_n, sn).log_prob(an).sum(-1)
    lpe = torch.distributions.Normal(mu_e, se[0]).log_prob(ae)
    return lpn.index_add(0, ei[0], lpe), (mu_n, mu_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='full', choices=list(HP['variants']))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--critic', default='team', choices=['team', 'local'])
    ap.add_argument('--proj', type=float, default=0.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--budget', type=int, default=HP['budget_steps'])
    ap.add_argument('--iter-episodes', type=int, default=HP['episodes_per_iter'])
    ap.add_argument('--val-every', type=int, default=HP['val_every'])
    ap.add_argument('--smoke', action='store_true', help='tiny run for code checks (not evidence)')
    ap.add_argument('--grad-probe-every', type=int, default=1, help='log |grad L_proj| / |grad L_PPO| every k iters')
    a = ap.parse_args()
    import sysperf
    perf = sysperf.configure()
    import numpy as np, torch
    import sarl_env as E
    import sarl_policy as P
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source').mkdir()
    files = [paths.HERE / 'train_ext.py', paths.HERE / 'paths.py', paths.SARL / 'train_ppo.py',
             paths.SARL / 'sarl_env.py', paths.SARL / 'sarl_policy.py', paths.SARL / 'sysperf.py']
    hashes = {}
    for f in files:
        bts = f.read_bytes()
        (out / 'source' / f.name).write_bytes(bts)
        hashes[f.name] = hashlib.sha256(bts).hexdigest()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed)
    actor = P.Actor(a.variant, HP['std0']).to(device)
    critic = build_critic(a.critic).to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=HP['lr'], eps=1e-5)
    val_scenes, redrawn = E.lattice_scenes(VAL_SEED, **{k: v for k, v in HP['val'].items() if k != 'fracs'},
                                           fracs=tuple(HP['val']['fracs']))
    val_list = list(val_scenes.values())
    if a.smoke:
        val_list = val_list[:4]
    meta = dict(variant=a.variant, seed=a.seed, critic=a.critic, proj=a.proj, hp=HP, workers=a.workers,
                budget=a.budget, iter_episodes=a.iter_episodes, val_seed=VAL_SEED,
                val_scenes=[s['name'] for s in val_list], val_redrawn=redrawn, source_sha256=hashes,
                device=str(device), python=sys.version, platform=platform.platform(), numpy=np.__version__,
                torch=torch.__version__, started=time.strftime('%Y-%m-%d %H:%M:%S'), smoke=a.smoke,
                process_settings=perf)
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
                          score=score, reasons={k: sum(s['reason'] == k for s in vres)
                                                for k in sorted(set(s['reason'] for s in vres))},
                          min_distance=float(min(s['min_distance'] for s in vres)),
                          min_lambda2_margin=float(min(s['min_lambda2_lower'] - s['lam'] for s in vres)),
                          interv_mean=float(np.mean([s['interv_mean'] for s in vres if s['interv_mean'] is not None])))

    def boot_node_values(boots):
        outv = []
        for o in boots:
            if o is None:
                outv.append(None)
                continue
            x = torch.as_tensor(np.concatenate([o['x'], o['xa']], 1), device=device)
            ea = torch.as_tensor(np.concatenate([o['ea'], o['eaa']], 1), device=device)
            ei = torch.as_tensor(o['ei'], dtype=torch.long, device=device)
            nb = torch.zeros(len(o['x']), dtype=torch.long, device=device)
            g_ = torch.as_tensor(o['glob'][None], device=device)
            with torch.no_grad():
                outv.append(critic(x, ei, ea, nb, g_, 1).cpu().numpy())
        return outv

    try:
        _, v0 = validate(9999)
        best = (v0['score'], -1)
        torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), iter=-1, variant=a.variant),
                   out / 'best.pt')
        log.write(json.dumps(dict(iter=-1, env_steps=0, val=v0, best_iter=-1)) + '\n')
        log.flush()
        print(f"it  -1 initial policy | VAL {v0['success']:.3f} t={v0['mean_time']}", flush=True)
        while total < a.budget:
            t_it = time.perf_counter()
            sd = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
            tasks = [dict(state=sd, scene=None, scene_seed=[a.seed, it, k], seed=[a.seed, it, k, 1], det=False,
                          tmax=HP['tmax']) for k in range(a.iter_episodes)]
            results = pool.map(run_episode, tasks, chunksize=1)
            t_roll = time.perf_counter() - t_it
            results = [r for r in results if r['steps']]
            B, boots = TP.assemble(results, a.variant, device)
            RI, RB, POS, GOAL, UEX = extras(results)
            POS_t = torch.as_tensor(POS, dtype=torch.float32, device=device)
            GOAL_t = torch.as_tensor(GOAL, dtype=torch.float32, device=device)
            UEX_t = torch.as_tensor(UEX, dtype=torch.float32, device=device)
            Gn = len(B['r'])
            Nn = int(B['ncount'].sum())
            ncount = B['ncount'].cpu().numpy()
            nstart = B['nstart'].cpu().numpy()
            with torch.no_grad():
                gid = torch.arange(Gn, device=device)
                if a.critic == 'team':
                    V = torch.zeros(Gn, device=device)
                    for ch in torch.split(gid, 4096):
                        m = TP.gather(B, ch, a.variant)
                        V[ch] = critic(m['xc'], m['ei'], m['eac'], m['nb'], m['glob'], m['G'])
                else:
                    V = torch.zeros(Nn, device=device)
                    for ch in torch.split(gid, 4096):
                        m = TP.gather(B, ch, a.variant)
                        V[gather_idx(B, ch)] = critic(m['xc'], m['ei'], m['eac'], m['nb'], m['glob'], m['G'])
                V = V.cpu().numpy()
            if a.critic == 'team':
                vb = TP.boot_values(critic, boots, device)
                adv = np.zeros(Gn); ret = np.zeros(Gn)
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
            else:
                vbn = boot_node_values(boots)
                adv = np.zeros(Nn); ret = np.zeros(Nn)
                for epi in range(len(results)):
                    idx = np.flatnonzero(B['ep'] == epi)
                    n_ep = int(ncount[idx[0]])
                    rows = np.stack([np.arange(nstart[q], nstart[q] + n_ep) for q in idx])     # (T, n)
                    last_gae = np.zeros(n_ep)
                    next_v = vbn[epi] if vbn[epi] is not None else np.zeros(n_ep)
                    for t in range(len(idx) - 1, -1, -1):
                        q = idx[t]
                        nonterm = 1.0 - B['done'][q]
                        rr_ = rows[t]
                        delta = RB[rr_] + gam * next_v * nonterm - V[rr_]
                        last_gae = delta + gam * lam * nonterm * last_gae
                        adv[rr_] = last_gae
                        next_v = V[rr_]
                    ret[rows.ravel()] = adv[rows.ravel()] + V[rows.ravel()]
            advn = (adv - adv.mean()) / (adv.std() + 1e-8)
            A_t = torch.as_tensor(advn, dtype=torch.float32, device=device)
            R_t = torch.as_tensor(ret, dtype=torch.float32, device=device)
            kls, pls, vls, prl, clipf, stop_epoch, gprobe = [], [], [], [], [], None, None
            for ep in range(HP['epochs']):
                perm = torch.randperm(Gn, device=device)
                ep_kl = []
                for mbi, ch in enumerate(torch.chunk(perm, HP['minibatches'])):
                    m = TP.gather(B, ch, a.variant)
                    nidx = gather_idx(B, ch)
                    lp_new, aout = actor_forward_logp(actor, m['xa'], m['ei'], m['eaa'], m['an'], m['ae'])
                    mask = (m['gate'] > HP['gate_mask']).float()
                    ratio = torch.exp(lp_new - m['lp'])
                    An = A_t[ch][m['nb']] if a.critic == 'team' else A_t[nidx]
                    pg = -torch.min(ratio * An, torch.clamp(ratio, 1 - HP['clip'], 1 + HP['clip']) * An)
                    lpi = (pg * mask).sum() / mask.sum().clamp(min=1.0)
                    v = critic(m['xc'], m['ei'], m['eac'], m['nb'], m['glob'], m['G'])
                    lv = ((v - (R_t[ch] if a.critic == 'team' else R_t[nidx])) ** 2).mean()
                    lproj = proj_loss(aout, m, POS_t, GOAL_t, UEX_t, nidx, E)
                    if ep == 0 and mbi == 0 and (it % a.grad_probe_every == 0):
                        aps = [q for q in actor.parameters()]
                        g1 = torch.autograd.grad(lpi, aps, retain_graph=True, allow_unused=True)
                        g2 = torch.autograd.grad(lproj, aps, retain_graph=True, allow_unused=True)
                        n1 = float(torch.sqrt(sum((x ** 2).sum() for x in g1 if x is not None)))
                        n2 = float(torch.sqrt(sum((x ** 2).sum() for x in g2 if x is not None)))
                        gprobe = dict(grad_ppo=n1, grad_proj_unit=n2, ratio_at_lambda=a.proj * n2 / max(n1, 1e-12),
                                      proj_value=float(lproj))
                    loss = lpi + HP['vf_coef'] * lv + (a.proj * lproj if a.proj > 0 else 0.0)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), HP['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), HP['max_grad_norm'])
                    opt.step()
                    with torch.no_grad():
                        lr_ = lp_new - m['lp']
                        kl = float((((torch.exp(lr_) - 1) - lr_) * mask).sum() / mask.sum().clamp(min=1.0))
                        ep_kl.append(kl)
                        clipf.append(float((((ratio - 1).abs() > HP['clip']).float() * mask).sum()
                                           / mask.sum().clamp(min=1.0)))
                    pls.append(float(lpi)); vls.append(float(lv)); prl.append(float(lproj))
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
                       pi_loss=float(np.mean(pls)), v_loss=float(np.mean(vls)), proj_loss=float(np.mean(prl)),
                       clipfrac=float(np.mean(clipf)), grad_probe=gprobe,
                       interv_mean=float(np.mean([s['interv_mean'] for s in st if s['interv_mean'] is not None])),
                       std_node=sn.detach().cpu().tolist(), std_edge=se.detach().cpu().tolist(),
                       gate_frac=float((B['gate'] > HP['gate_mask']).float().mean()),
                       block_vs_team_reward_corr=float(np.corrcoef(RB, np.repeat(B['r'], ncount))[0, 1])
                       if RB.std() > 0 else None,
                       block_vs_team_reward_corr_nonterminal=float(np.corrcoef(
                           RB[np.repeat(B['done'], ncount) == 0], np.repeat(B['r'], ncount)[np.repeat(B['done'], ncount) == 0])[0, 1])
                       if (np.repeat(B['done'], ncount) == 0).sum() > 2 else None,
                       block_minus_team_rms=float(np.sqrt(np.mean((RB - np.repeat(B['r'], ncount)) ** 2))),
                       team_reward_rms=float(np.sqrt(np.mean(np.repeat(B['r'], ncount) ** 2))))
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
            if it % 16 == 0:
                torch.save(dict(actor=actor.state_dict(), iter=it, variant=a.variant), out / f'ckpt_{it:04d}.pt')
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(f"it {it:3d} steps {total:7d} succ {row['success']:.2f} ret {row['ret']:+.3f} "
                  f"kl {kls[-1]:.4f} proj {row['proj_loss']:.4f} interv {row['interv_mean']:.4f} "
                  f"gp {gprobe['ratio_at_lambda'] if gprobe else None} roll {t_roll:.0f}s it {row['iter_s']:.0f}s"
                  + (f" | VAL {row['val']['success']:.3f} t={row['val']['mean_time']}" if 'val' in row else ''),
                  flush=True)
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

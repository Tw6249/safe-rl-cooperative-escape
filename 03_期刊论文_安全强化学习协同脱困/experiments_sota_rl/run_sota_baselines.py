"""Train and evaluate three representative Safe MARL baselines on the 96 benchmark scenes
(E1: N=8, 32 scenes at f=0.60 + 32 scenes at f=0.90; E2: N in {12,20,25}, 32 scenes across seeds 20261017 and 20261105):

1. Proj-MAPPO + NPA (Shield-Consistent / Action-Projected Safe MARL, Wabersich & Zeilinger 2021 / Liu et al. 2026):
   Uses the trained checkpoint with shield-projection consistency regulariser L_proj (train_cteam_p1_s2/best.pt)
   + completion envelope rho_{i,k} + closed-neighborhood block NPA (B=8).
2. Static-CBF GNN (GCBF+ / DGPPO-style Decentralized Static-Split CBF-QP Shielded GNN, Zhang et al. 2025):
   Uses the GNN policy (train_full02/best.pt) + completion envelope rho_{i,k} + decentralized static-split CBF-QP (B=0).
3. MAPPO-Lagrangian (Primal-Dual Constrained Safe MARL without hard coupled QP shield):
   Trains the identical 2-layer Equivariant GNN actor + task critic + cost critic with dual-ascent Lagrangian multiplier
   penalising collision and algebraic connectivity violations, then evaluates on the 96 scenes with exact rational ZOH
   trajectory auditing.

Strictly read-only on 02_章节代码/ and 独立测试_20261107_第五章强化学习改进/. All outputs are saved under
03_期刊论文_安全强化学习协同脱困/experiments_sota_rl/results/.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
import math
import multiprocessing as mp
import sys
import time
from functools import partial
from math import comb
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMON = ROOT / "02_章节代码" / "00_公共依赖" / "统一模型"
C5 = ROOT / "02_章节代码" / "第五章_安全强化学习与协同脱困"
BASE_CODE = C5 / "基础依赖" / "code"
SARL_CODE = C5 / "独立验证" / "20261102_安全强化学习" / "code"
ENV_CODE = C5 / "独立验证" / "20261103_终段包络验证" / "code"
EXT_DIR = ROOT / "独立测试_20261107_第五章强化学习改进"

sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(BASE_CODE))
sys.path.insert(0, str(SARL_CODE))
sys.path.insert(0, str(ENV_CODE))

import pa_bridge  # noqa: E402
import sarl_env as E  # noqa: E402
import sarl_policy as P  # noqa: E402
import evaluate as EV  # noqa: E402
import eval_env as EE  # noqa: E402
from audit_trajectories import audit  # noqa: E402
from pair_common import dump  # noqa: E402
from safety_checks import laplacian_from_weights, weights_from_squared_distance, pair_indices  # noqa: E402

OUT_DIR = HERE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THESIS_CKPT = C5 / "独立验证" / "20261102_安全强化学习" / "results" / "train_full02" / "best.pt"
PROJ_CKPT = EXT_DIR / "results" / "train_cteam_p1_s2" / "best.pt"
LAG_CKPT = OUT_DIR / "train_mappo_lagrangian" / "best.pt"


# ---------------------------------------------------------------------------
# 1. Fast MAPPO-Lagrangian training (same 2-layer Equivariant GNN architecture)
# ---------------------------------------------------------------------------
def observe_geom_fast(p, goal, lam_req, u_prev, mem):
    """Compute the exact 12D node_geom and 13D edge_geom features of sarl_env.ResidualFlow.observe
    without calling CVXPY."""
    n = len(p)
    e = p - goal
    ne = np.linalg.norm(e, axis=1)
    eh = e / np.maximum(ne, 1e-9)[:, None]
    sg = np.minimum(1.0, ne / 0.1)
    Jeh = np.stack([-eh[:, 1], eh[:, 0]], 1)
    dist = np.linalg.norm(p[:, None] - p[None], axis=2)
    I, J = np.nonzero((dist <= E.RADIUS) & ~np.eye(n, dtype=bool))
    d = dist[I, J]
    r = (p[J] - p[I]) / d[:, None]
    Jr = np.stack([-r[:, 1], r[:, 0]], 1)
    deg = np.bincount(I, minlength=n)
    v = u_prev
    dl = np.zeros_like(v)
    # Proximity conflict signal for gate when no QP solver is used: max(0, 1.15 - min_d)
    clear = np.full(n, 1.5)
    np.minimum.at(clear, I, np.minimum(d - 1.0, 1.5))
    sig = np.maximum(0.0, 0.25 - clear)
    raw = sig.copy()
    np.maximum.at(raw, I, sig[J])
    mem_new = np.maximum(E.GATE_DECAY * mem, raw)
    gate = np.tanh(E.GATE_GAIN * mem_new)

    Ii, Jj = pair_indices(n)
    d2 = np.sum((p[Ii] - p[Jj]) ** 2, axis=1)
    ev = np.linalg.eigvalsh(laplacian_from_weights(n, weights_from_squared_distance(d2, E.RC)))
    lam2 = float(ev[1])
    min_dist = float(np.sqrt(d2.min()))

    x = np.column_stack([
        np.minimum(ne, 4.0), (ne < 0.05).astype(float), gate,
        np.linalg.norm(v, axis=1) / E.U, np.sum(v * eh, 1) / E.U * sg, np.sum(v * Jeh, 1) / E.U * sg,
        np.zeros(n), np.zeros(n), np.zeros(n),
        np.full(n, (lam2 - lam_req) / lam_req), deg / 8.0, clear,
    ])
    ea = np.column_stack([
        d / E.RADIUS, np.clip(d - 1.0, 0.0, 1.5),
        np.sum(r * eh[I], 1) * sg[I], np.sum(r * Jeh[I], 1) * sg[I],
        np.sum(r * eh[J], 1) * sg[J], np.sum(r * Jeh[J], 1) * sg[J], np.minimum(ne[J], 4.0),
        np.sum(eh[I] * eh[J], 1) * sg[I] * sg[J], np.sum(Jeh[I] * eh[J], 1) * sg[I] * sg[J],
        np.sum((v[J] - v[I]) * r, 1) / E.U, np.sum((v[J] - v[I]) * Jr, 1) / E.U, gate[J],
        np.zeros(len(I)),
    ])
    # Closed-neighborhood completion envelope rho_i
    ne_nb = ne.copy()
    if len(I):
        np.maximum.at(ne_nb, I, ne[J])
    rho = np.tanh(ne_nb / 0.05)
    gate_eff = rho  # envelope-gated residual
    obs = dict(
        x=x.astype(np.float32),
        ea=ea.astype(np.float32),
        ei=np.stack([I, J]).astype(np.int64),
        gate=gate_eff.astype(np.float32),
        glob=np.array([lam2, lam_req, float(np.mean(ne < 0.05)), n / 10.0], np.float32),
    )
    geo = dict(e=e, I=I, J=J, d=d, deg=deg, ne=ne, lam2=lam2, min_dist=min_dist)
    return obs, geo, mem_new


class GeomCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = P.GNN(E.NODE_GEOM, E.EDGE_GEOM)
        self.head = nn.Sequential(
            nn.Linear(2 * P.H + 4, P.H), nn.SiLU(),
            nn.Linear(P.H, P.H), nn.SiLU(),
            nn.Linear(P.H, 1),
        )

    def forward(self, x, ei, ea, batch, glob, n_graphs):
        h = self.gnn(x, ei, ea)
        mean, mx = P.scatter_mean_max(h, batch, n_graphs)
        return self.head(torch.cat([mean, mx, glob], 1)).squeeze(-1)


def rollout_lagrangian_episode(actor, scene, rng, det=False, tmax=240):
    """Roll out one episode of unshielded MAPPO-Lagrangian with step size dt=0.05 s."""
    pol = P.NumpyPolicy(actor, det, rng)
    p = scene["p"].copy()
    g = scene["goal"].copy()
    lam_req = scene["lam"]
    n = len(p)
    u_prev = np.zeros((n, 2))
    mem = np.zeros(n)
    D0 = float(np.linalg.norm(p - g, axis=1).sum())
    steps = []
    dt = 0.05
    min_d, min_lam = np.inf, np.inf
    viol_steps = 0
    arrived_hold = 0
    arrived_time = None

    for k in range(tmax):
        obs, geo, mem = observe_geom_fast(p, g, lam_req, u_prev, mem)
        node_a, edge_a = pol(obs)
        e, I, J, d, deg = geo["e"], geo["I"], geo["J"], geo["d"], geo["deg"]
        w = np.clip(node_a[:, 0], -1.0, 1.0)
        rr = np.clip(node_a[:, 1], -1.0, 1.0)
        psi = np.clip(edge_a, -1.0, 1.0)
        Je = np.stack([-e[:, 1], e[:, 0]], 1)
        away = (p[I] - p[J]) / d[:, None] if len(I) else np.zeros((0, 2))
        pb = np.zeros_like(e)
        if len(I):
            np.add.at(pb, I, psi[:, None] * away)
        pb /= np.maximum(deg, 1)[:, None]
        res = E.A_ROT * w[:, None] * Je + E.A_RAD * rr[:, None] * e + E.A_PSI * E.U * pb
        nom = -e + obs["gate"][:, None] * res
        u = E.clip_ball(nom)
        z = np.clip(p + dt * u, -4.0, 4.0)

        # Check safety at next state z
        Ii, Jj = pair_indices(n)
        d2_next = np.sum((z[Ii] - z[Jj]) ** 2, axis=1)
        d_next = np.sqrt(d2_next)
        md = float(d_next.min())
        ev_next = np.linalg.eigvalsh(laplacian_from_weights(n, weights_from_squared_distance(d2_next, E.RC)))
        l2 = float(ev_next[1])
        min_d = min(min_d, md)
        min_lam = min(min_lam, l2)

        # Safety violation cost C_k (collision + connectivity)
        col_pen = float(np.sum(np.maximum(0.0, 1.02 - d_next)))
        con_pen = float(max(0.0, lam_req - l2))
        cost = col_pen + con_pen
        if md < 1.0 or l2 < lam_req:
            viol_steps += 1

        e0 = np.linalg.norm(p - g, axis=1)
        e1 = np.linalg.norm(z - g, axis=1)
        r = (e0.sum() - e1.sum()) / D0 - 1e-3
        done = 0.0
        if e1.max() <= 0.01:
            arrived_hold += 1
            if arrived_hold == 1:
                arrived_time = (k + 1) * dt
            if arrived_hold >= 20:
                r += 1.0
                done = 1.0
        else:
            arrived_hold = 0

        if not det:
            rec = pol.record[-1]
            steps.append(dict(obs=obs, r=float(r), c=float(cost), done=done, rec=rec))
        p = z
        u_prev = u
        if done:
            break

    return dict(
        steps=steps,
        success=bool(arrived_hold >= 20),
        arrival=arrived_time if arrived_hold >= 20 else None,
        min_d=min_d,
        min_lam=min_lam,
        viol_steps=viol_steps,
        final_max_err=float(np.linalg.norm(p - g, axis=1).max()),
    )


def train_mappo_lagrangian():
    out = OUT_DIR / "train_mappo_lagrangian"
    out.mkdir(parents=True, exist_ok=True)
    if LAG_CKPT.exists():
        print("MAPPO-Lagrangian checkpoint already exists:", LAG_CKPT)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20261109)
    rng = np.random.default_rng(20261109)

    actor = P.Actor("geom", std0=0.4).to(device)
    v_critic = GeomCritic().to(device)
    c_critic = GeomCritic().to(device)
    opt = torch.optim.Adam(
        list(actor.parameters()) + list(v_critic.parameters()) + list(c_critic.parameters()),
        lr=1e-3, eps=1e-5,
    )

    lam_lag = 0.5  # initial Lagrangian multiplier
    lr_lag = 0.05
    cost_limit = 0.02
    gamma, gae_lam = 0.995, 0.95

    val_scenes, _ = E.lattice_scenes(20261102, n=8, n_local=12, n_perm=6, fracs=(0.6,))
    val_list = list(val_scenes.values())

    best_score = -1e9
    log_f = open(out / "train_log.jsonl", "w", encoding="utf8")
    t0 = time.time()

    for it in range(60):
        actor_cpu = P.Actor("geom", std0=0.4)
        actor_cpu.load_state_dict({k: v.detach().cpu() for k, v in actor.state_dict().items()})
        actor_cpu.eval()

        episodes = []
        for ep_i in range(20):
            sc = E.train_scene(rng, tag=f"_{it}_{ep_i}")
            episodes.append(rollout_lagrangian_episode(actor_cpu, sc, rng, det=False, tmax=220))

        # Assemble batch
        xs, eas, eis, globs, nbs = [], [], [], [], []
        ans, aes, lps, gates = [], [], [], []
        adv_r_list, ret_r_list, adv_c_list, ret_c_list = [], [], [], []
        node_offset = 0
        graph_idx = 0
        ep_costs, ep_rets = [], []

        for ep in episodes:
            st = ep["steps"]
            if not st:
                continue
            ep_rets.append(sum(s["r"] for s in st))
            ep_costs.append(sum(s["c"] for s in st))
            # Evaluate V_r and V_c
            bx, bea, bei, bglob, bnb = [], [], [], [], []
            no = 0
            for gi, s in enumerate(st):
                o = s["obs"]
                n_i = len(o["x"])
                bx.append(o["x"])
                bea.append(o["ea"])
                bei.append(o["ei"] + no)
                bglob.append(o["glob"])
                bnb.append(np.full(n_i, gi, dtype=np.int64))
                no += n_i
            tx = torch.from_numpy(np.concatenate(bx)).to(device)
            tea = torch.from_numpy(np.concatenate(bea)).to(device)
            tei = torch.from_numpy(np.concatenate(bei, axis=1)).to(device)
            tglob = torch.from_numpy(np.stack(bglob)).to(device)
            tnb = torch.from_numpy(np.concatenate(bnb)).to(device)
            with torch.no_grad():
                vr = v_critic(tx, tei, tea, tnb, tglob, len(st)).cpu().numpy()
                vc = c_critic(tx, tei, tea, tnb, tglob, len(st)).cpu().numpy()

            T_ep = len(st)
            ar, ac = np.zeros(T_ep), np.zeros(T_ep)
            gae_r, gae_c = 0.0, 0.0
            next_vr, next_vc = 0.0, 0.0
            for t in range(T_ep - 1, -1, -1):
                nonterm = 1.0 - st[t]["done"]
                dr = st[t]["r"] + gamma * next_vr * nonterm - vr[t]
                dc = st[t]["c"] + gamma * next_vc * nonterm - vc[t]
                gae_r = dr + gamma * gae_lam * nonterm * gae_r
                gae_c = dc + gamma * gae_lam * nonterm * gae_c
                ar[t], ac[t] = gae_r, gae_c
                next_vr, next_vc = vr[t], vc[t]
            rr_t = ar + vr
            rc_t = ac + vc

            for t, s in enumerate(st):
                o = s["obs"]
                rec = s["rec"]
                n_i = len(o["x"])
                xs.append(o["x"])
                eas.append(o["ea"])
                eis.append(o["ei"] + node_offset)
                globs.append(o["glob"])
                nbs.append(np.full(n_i, graph_idx, dtype=np.int64))
                ans.append(rec["an"])
                aes.append(rec["ae"])
                lps.append(rec["logp"])
                gates.append(o["gate"])
                adv_r_list.append(ar[t])
                ret_r_list.append(rr_t[t])
                adv_c_list.append(ac[t])
                ret_c_list.append(rc_t[t])
                node_offset += n_i
                graph_idx += 1

        X_t = torch.from_numpy(np.concatenate(xs)).to(device)
        EA_t = torch.from_numpy(np.concatenate(eas)).to(device)
        EI_t = torch.from_numpy(np.concatenate(eis, axis=1)).to(device)
        GLOB_t = torch.from_numpy(np.stack(globs)).to(device)
        NB_t = torch.from_numpy(np.concatenate(nbs)).to(device)
        AN_t = torch.from_numpy(np.concatenate(ans)).to(device)
        AE_t = torch.from_numpy(np.concatenate(aes)).to(device)
        LP_t = torch.from_numpy(np.concatenate(lps)).to(device)
        GATE_t = torch.from_numpy(np.concatenate(gates)).to(device)

        adv_r_np = np.array(adv_r_list, dtype=np.float32)
        adv_c_np = np.array(adv_c_list, dtype=np.float32)
        adv_r_np = (adv_r_np - adv_r_np.mean()) / (adv_r_np.std() + 1e-8)
        adv_c_np = (adv_c_np - adv_c_np.mean()) / (adv_c_np.std() + 1e-8)
        # Combined Lagrangian advantage: A_lag = (A_r - lam_lag * A_c) / (1 + lam_lag)
        adv_comb = (adv_r_np - lam_lag * adv_c_np) / (1.0 + lam_lag)
        A_node = torch.from_numpy(adv_comb).to(device)[NB_t]
        RR_t = torch.from_numpy(np.array(ret_r_list, dtype=np.float32)).to(device)
        RC_t = torch.from_numpy(np.array(ret_c_list, dtype=np.float32)).to(device)

        for _ep_ppo in range(4):
            lp_new = actor.log_prob(X_t, EI_t, EA_t, AN_t, AE_t)
            ratio = torch.exp(lp_new - LP_t)
            mask = (GATE_t > 0.01).float()
            pg = -torch.min(ratio * A_node, torch.clamp(ratio, 0.8, 1.2) * A_node)
            l_pi = (pg * mask).sum() / mask.sum().clamp(min=1.0)
            v_pred = v_critic(X_t, EI_t, EA_t, NB_t, GLOB_t, graph_idx)
            c_pred = c_critic(X_t, EI_t, EA_t, NB_t, GLOB_t, graph_idx)
            l_v = ((v_pred - RR_t) ** 2).mean() + ((c_pred - RC_t) ** 2).mean()
            loss = l_pi + 0.5 * l_v
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(v_critic.parameters(), 0.5)
            nn.utils.clip_grad_norm_(c_critic.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                kl = float((((torch.exp(lp_new - LP_t) - 1.0) - (lp_new - LP_t)) * mask).sum() / mask.sum().clamp(min=1.0))
            if kl > 0.03:
                break

        # Dual ascent on Lagrangian multiplier
        mean_cost_step = float(np.mean(ep_costs)) / max(1.0, float(np.mean([len(e["steps"]) for e in episodes])))
        lam_lag = float(np.clip(lam_lag + lr_lag * (mean_cost_step - cost_limit), 0.0, 15.0))

        if (it + 1) % 10 == 0 or it == 59:
            actor_cpu.load_state_dict({k: v.detach().cpu() for k, v in actor.state_dict().items()})
            v_runs = [rollout_lagrangian_episode(actor_cpu, s, rng, det=True, tmax=400) for s in val_list]
            succ = float(np.mean([r["success"] for r in v_runs]))
            viol = float(np.mean([r["viol_steps"] for r in v_runs]))
            err = float(np.mean([r["final_max_err"] for r in v_runs]))
            score = succ - 0.1 * err - 0.002 * viol
            if score > best_score:
                best_score = score
                torch.save(
                    dict(actor=actor.state_dict(), iter=it, variant="geom", lam_lag=lam_lag),
                    LAG_CKPT,
                )
            print(
                f"[MAPPO-Lagrangian] it={it:2d} ret={np.mean(ep_rets):+.3f} cost/step={mean_cost_step:.4f} "
                f"lam_lag={lam_lag:.2f} | val_succ={succ:.3f} val_err={err:.3f} val_viol_steps={viol:.1f}",
                flush=True,
            )
            log_f.write(json.dumps(dict(
                iter=it, ret=float(np.mean(ep_rets)), cost_step=mean_cost_step, lam_lag=lam_lag,
                val_succ=succ, val_err=err, val_viol=viol,
            )) + "\n")
            log_f.flush()

    log_f.close()
    print(f"Finished training MAPPO-Lagrangian in {time.time() - t0:.1f}s -> {LAG_CKPT}", flush=True)


# ---------------------------------------------------------------------------
# 2. Closed-loop evaluation of the 3 Safe RL baselines on E1 (64) + E2 (32)
# ---------------------------------------------------------------------------
_W = {}


def all_96_scenes():
    sc_e1, _ = E.lattice_scenes(20261103, n=8, n_local=20, n_perm=12, fracs=(0.6, 0.9))
    sc_e2_a = E.scenes_for(20261017)
    sc_e2_b = E.scenes_for(20261105)
    for v in sc_e2_a.values():
        v["bench"] = "E2"
        v.setdefault("frac", 0.6)
    for v in sc_e2_b.values():
        v["bench"] = "E2"
        v.setdefault("frac", 0.6)
    for v in sc_e1.values():
        v["bench"] = "E1"
    out = {}
    out.update(sc_e1)
    out.update(sc_e2_a)
    out.update(sc_e2_b)
    return out


def init_eval_worker():
    import sysperf
    sysperf.configure()
    torch.set_num_threads(1)
    _W["scenes"] = all_96_scenes()
    _W["actors"] = {}


def get_actor(path):
    p_str = str(path)
    if p_str not in _W["actors"]:
        ck = torch.load(p_str, map_location="cpu")
        a = P.Actor(ck["variant"])
        a.load_state_dict(ck["actor"])
        a.eval()
        _W["actors"][p_str] = a
    return _W["actors"][p_str]


def eval_unshielded_lagrangian_case(case, actor, d_runs, tag):
    """Evaluate unshielded MAPPO-Lagrangian on a benchmark scene over 60 s (dt=0.05 s, ZOH)
    and run the exact rational trajectory auditor `audit()` on the saved trajectory."""
    pol = P.NumpyPolicy(actor, True)
    p = np.asarray(case["p"], float).copy()
    g = np.asarray(case["goal"], float).copy()
    lam_req = float(case["lam"])
    n = len(p)
    u_prev = np.zeros((n, 2))
    mem = np.zeros(n)
    dt = 0.05
    states = [p.copy()]
    times = [0.0]
    arrived_hold = 0
    arrival = None
    status = "horizon_reached"
    quiet = 0

    for k in range(1200):
        obs, geo, mem = observe_geom_fast(p, g, lam_req, u_prev, mem)
        node_a, edge_a = pol(obs)
        e, I, J, d, deg = geo["e"], geo["I"], geo["J"], geo["d"], geo["deg"]
        w = np.clip(node_a[:, 0], -1.0, 1.0)
        rr = np.clip(node_a[:, 1], -1.0, 1.0)
        psi = np.clip(edge_a, -1.0, 1.0)
        Je = np.stack([-e[:, 1], e[:, 0]], 1)
        away = (p[I] - p[J]) / d[:, None] if len(I) else np.zeros((0, 2))
        pb = np.zeros_like(e)
        if len(I):
            np.add.at(pb, I, psi[:, None] * away)
        pb /= np.maximum(deg, 1)[:, None]
        res = E.A_ROT * w[:, None] * Je + E.A_RAD * rr[:, None] * e + E.A_PSI * E.U * pb
        nom = -e + obs["gate"][:, None] * res
        u = E.clip_ball(nom)
        p = np.clip(p + dt * u, -4.0, 4.0)
        u_prev = u
        t_now = round((k + 1) * dt, 6)
        states.append(p.copy())
        times.append(t_now)

        err_max = float(np.linalg.norm(p - g, axis=1).max())
        if err_max <= 0.01:
            arrived_hold += 1
            if arrived_hold == 1:
                arrival = t_now
            if arrived_hold >= 20:
                status = "arrived"
                break
        else:
            arrived_hold = 0

        if float(np.linalg.norm(u)) < 1e-6:
            quiet += 1
            if quiet >= 100:
                status = "stalled"
                break
        else:
            quiet = 0

    npz_path = d_runs / f"{case['name']}_{tag}.npz"
    np.savez_compressed(npz_path, states=np.array(states), times=np.array(times), goal=g)
    n_seg, bad, md, ml = audit(npz_path, lam_req)
    return dict(
        scene=case["name"],
        bench=case["bench"],
        n=case["n"],
        frac=case.get("frac", 0.6),
        tag=tag,
        status=status,
        arrival=arrival if status == "arrived" else None,
        steps=len(times) - 1,
        step_cuts=0,
        audit_segments=n_seg,
        audit_failed=len(bad),
        audit_min_distance=md,
        audit_min_lambda2=ml,
        final_max_err=float(np.linalg.norm(states[-1] - g, axis=1).max()),
    )


def run_eval_job(job):
    case = dict(_W["scenes"][job["scene"]])
    d_runs = OUT_DIR / "eval_runs"
    d_rows = OUT_DIR / "eval_rows"
    row_file = d_rows / f"{case['name']}_{job['tag']}.json"
    if row_file.exists():
        return json.loads(row_file.read_text(encoding="utf8"))

    t0 = time.time()
    if job["mode"] == "lagrangian_unshielded":
        actor = get_actor(LAG_CKPT)
        r = eval_unshielded_lagrangian_case(case, actor, d_runs, job["tag"])
    else:
        actor = get_actor(job["ckpt"])
        pa_bridge.PAOracle = partial(EV.budget_oracle(E), settle="local", diagnostic=False)
        EE.EnvelopedResidualFlow.gate_kind = job["gate"]
        EE.EnvelopedResidualFlow.eps_g = 0.05
        EE.EnvelopedResidualFlow.policy = P.NumpyPolicy(actor, True)
        pa_bridge.BraidFlow = EE.EnvelopedResidualFlow
        r = pa_bridge.run_pa(case, d_runs, 0.0, job["rounds"], E.RADIUS, 60.0, job["tag"], clip_level=True)
        if r["status"] == "solver_failure" and "StepBudget" in str(r.get("failure")):
            r["status"] = "step_budget"
        npz_path = d_runs / f"{case['name']}_{job['tag']}.npz"
        n_seg, bad, md, ml = audit(npz_path, case["lam"]) if r["steps"] else (0, [], None, None)
        r.update(
            bench=case["bench"],
            n=case["n"],
            frac=case.get("frac", 0.6),
            tag=job["tag"],
            audit_segments=n_seg,
            audit_failed=len(bad),
            audit_min_distance=md,
            audit_min_lambda2=ml,
            final_max_err=float(np.linalg.norm(np.load(npz_path)["states"][-1] - case["goal"], axis=1).max())
            if r["steps"] else None,
        )
    r["wall_s"] = round(time.time() - t0, 2)
    dump(row_file, r)
    print(
        f"[{job['tag']}] {case['name']} -> {r['status']} arr={r.get('arrival')} "
        f"cuts={r.get('step_cuts')} audit_bad={r.get('audit_failed')} ({r['wall_s']}s)",
        flush=True,
    )
    return r


def main():
    train_mappo_lagrangian()

    (OUT_DIR / "eval_runs").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_rows").mkdir(parents=True, exist_ok=True)

    scenes = all_96_scenes()
    jobs = []
    for sname in sorted(scenes, key=lambda k: (-scenes[k]["n"], k)):
        # 1. Proj-MAPPO + NPA (B=8)
        jobs.append(dict(scene=sname, mode="shielded", ckpt=str(PROJ_CKPT), gate="mu", rounds=8, tag="rl_proj_env_B8"))
        # 2. GCBF+/DGPPO-style Decentralized Static-CBF GNN (B=0)
        jobs.append(dict(scene=sname, mode="shielded", ckpt=str(THESIS_CKPT), gate="mu", rounds=0, tag="rl_static_cbf_B0"))
        # 3. MAPPO-Lagrangian (unshielded constrained MARL)
        jobs.append(dict(scene=sname, mode="lagrangian_unshielded", ckpt=str(LAG_CKPT), gate="int", rounds=0, tag="rl_mappo_lag"))

    print(f"Running {len(jobs)} evaluation jobs across 96 scenes...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(16, initializer=init_eval_worker) as pool:
        rows = pool.map(run_eval_job, jobs, chunksize=1)

    (OUT_DIR / "summary_sota_rl.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf8")
    print("Completed all SOTA Safe RL evaluations.", flush=True)


if __name__ == "__main__":
    main()

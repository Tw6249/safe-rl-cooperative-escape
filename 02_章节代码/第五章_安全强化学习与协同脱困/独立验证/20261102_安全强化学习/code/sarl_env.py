"""Shielded closed loop of the safe-RL experiment (README of the parent directory).

Frozen code is imported unchanged:
  pa_bridge.build_rows / run_pa / lambda2_data / GoalProjectionFlow
  ledger_npa.LedgerNPAOracle                     filter L (closed-neighbourhood NPA + edge-ledger settlement)
  pair_common.certified_step, audit              exact-rational segment certificate and trajectory audit
New in this file: ShieldOracle (L with a bounded block cache and the recorded duals of the local refits),
ResidualFlow (gated residual nominal input), scene generators, observation features.
"""
from pathlib import Path
from collections import OrderedDict
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]                                                    # 02_章节代码/第五章_安全强化学习与协同脱困/独立验证
sys.path.insert(0, str(BASE.parent / '基础依赖' / 'code'))
from ledger_npa import LedgerNPAOracle                                    # noqa: E402
from block_npa import NeighbourhoodBlock                                  # noqa: E402
import numpy as np                                                        # noqa: E402
import pa_bridge                                                          # noqa: E402
from pa_bridge import build_rows, U, RC, GoalProjectionFlow               # noqa: E402
from scenes import lattice, lambda2, adjacent_pairs, effective_margin, RADIUS, scenes_for  # noqa: E402
from safety_checks import laplacian_from_weights, weights_from_squared_distance, pair_indices  # noqa: E402

GATE_DECAY, GATE_GAIN = 0.97, 4.0     # gate memory per sample and tanh gain (fixed before any run, README §4)
A_ROT, A_RAD, A_PSI = 2.0, 1.5, 2.0   # residual scales (README §4)
NODE_GEOM, EDGE_GEOM, NODE_ALG, EDGE_ALG = 12, 13, 5, 7


# ---------------------------------------------------------------------------------------------------------- filter
class ShieldOracle(LedgerNPAOracle):
    """Filter L, unchanged in every decision. Added only: (i) the block cache keeps the 256 most recently used
    problems (a rebuilt block is the same problem; 20261029_第三章性质检验 used the same bound), (ii) the dual vector
    of every robot's last successful local refit in a sample is recorded (observation only), (iii) the diagnostic
    central QP can be switched off (training only; it never drives the robots), (iv) reset() for reuse in training."""
    CACHE = 256

    def __init__(self, *a, diagnostic=True, **k):
        super().__init__(*a, **k)
        self.block_cache = OrderedDict()
        self.diagnostic = diagnostic
        self.duals = np.zeros((self.n, self.n))       # rows of member_rows(a): [spectral, collision with o != a]
        self.s_last = None

    def reset(self, lam_eff):
        self.lam_eff = lam_eff
        self.s_prev = self.s_last = None
        self.ledger = {}
        self.count, self.max_residual, self.last = 0, 0.0, {}
        self.totals = dict(accepted=0, rejected=0, transfer_scalars=0, rounds_accepted=0, rounds_rejected=0,
                           stars_rejected=0, inherit_ok=0, inherit_fail=0)
        self.duals[:] = 0.0

    def block(self, i, nbs, edge_set):
        members = (i,) + tuple(nbs)
        mset = set(members)
        pairs = tuple(sorted(e for e in edge_set if e[0] in mset and e[1] in mset))
        key = (members, pairs)
        bp = self.block_cache.get(key)
        if bp is None:
            bp = self.block_cache[key] = NeighbourhoodBlock(members, pairs, self.n, self.eta)
            if len(self.block_cache) > self.CACHE:
                self.block_cache.popitem(last=False)
        else:
            self.block_cache.move_to_end(key)
        return bp

    def refit_one(self, a, A, b, ws, nominal, s):
        ua, why = super().refit_one(a, A, b, ws, nominal, s)
        if ua is not None:
            dv = self.refits[a].problem.constraints[0].dual_value
            if dv is not None:
                self.duals[a] = np.asarray(dv, float).reshape(-1)
        return ua, why

    def central(self, A, b, part, ws, nominal):
        if not self.diagnostic:
            return None, None
        return super().central(A, b, part, ws, nominal)

    def solve(self, p, nominal):
        self.duals[:] = 0.0
        self.block_seconds = []                       # timing list of the parent, emptied per sample
        return super().solve(p, nominal)


# ---------------------------------------------------------------------------------------------------------- scenes
def tri_sites(spacing, k=7):
    return np.array([(spacing * (a + b / 2), spacing * b * np.sqrt(3) / 2) for a in range(-k, k + 1)
                     for b in range(-k, k + 1)])


def adjacent(p, spacing):
    I, J = np.triu_indices(len(p), 1)
    d = np.linalg.norm(p[I] - p[J], axis=1)
    return [(int(a), int(b)) for a, b, dd in zip(I, J, d) if dd < 1.2 * spacing]


def swap_perm(rng, n, adj, m):
    """Same rule as closed_loop.scenes_for: m disjoint adjacent swaps (50 tries each)."""
    perm = np.arange(n)
    used = set()
    for _ in range(m):
        for _try in range(50):
            a, b = adj[int(rng.integers(len(adj)))]
            if a not in used and b not in used:
                perm[[a, b]] = perm[[b, a]]
                used |= {a, b}
                break
    return perm


def cycle_perm(rng, n, adj):
    A = set(adj)
    tris = [(a, b, c) for (a, b) in adj for c in range(b + 1, n) if (a, c) in A and (b, c) in A]
    a, b, c = tris[int(rng.integers(len(tris)))]
    perm = np.arange(n)
    perm[[a, b, c]] = perm[[b, c, a]]
    return perm


def spectral_gap(p):
    lam2, gap, _ = pa_bridge.lambda2_data(p)
    return lam2, gap


def train_scene(rng, tag=''):
    """Training distribution (README §3.1): N in 6..10, triangular sites with spacing in [1.001, 1.12] around a random
    centre, random rotation, goals = local swaps / 3-cycle / full permutation, lambda_req = f * lambda_2, f in
    [0.55, 0.90]. Rejected: identity task, spectral gap < 1e-3 (scalar Fiedler row needs a simple root)."""
    for _ in range(200):
        n = int(rng.choice([6, 7, 8, 9, 10], p=[.1, .2, .3, .25, .15]))
        S = float(rng.uniform(1.001, 1.12))
        c = rng.uniform(-.5, .5, 2) * S
        pts = tri_sites(S)
        d = np.linalg.norm(pts - c, axis=1)
        order = np.lexsort((np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]), np.round(d, 6)))
        p = pts[order[:n]]
        th = float(rng.uniform(0, 2 * np.pi))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        p = (p - p.mean(axis=0)) @ R.T
        adj = adjacent(p, S)
        kind = str(rng.choice(['local', 'cycle', 'perm'], p=[.55, .25, .2]))
        if kind == 'local':
            perm = swap_perm(rng, n, adj, int(rng.integers(1, 4)))
        elif kind == 'cycle':
            perm = cycle_perm(rng, n, adj)
        else:
            perm = rng.permutation(n)
        if np.array_equal(perm, np.arange(n)):
            continue
        lam2, gap = spectral_gap(p)
        I, J = np.triu_indices(n, 1)
        if gap < 1e-3 or np.linalg.norm(p[I] - p[J], axis=1).min() < 1.0005 or np.abs(p).max() > 3.5:
            continue
        f = float(rng.uniform(.55, .90))
        return dict(name=f'T{n}_{kind}{tag}', p=p, goal=p[perm].copy(), lam=round(f * lam2, 3), n=n,
                    perm=perm.tolist(), kind=kind, spacing=S, frac=f)
    raise RuntimeError('no admissible training scene in 200 draws')


def lattice_scenes(seed, n=8, n_local=16, n_perm=8, fracs=(0.6, 0.85), tag=''):
    """Evaluation family (README §3.2): frozen scenes.lattice(n) sites (spacing 1.001), tasks drawn as in
    closed_loop.scenes_for (1-3 adjacent swaps; uniform permutations), duplicate or identity tasks redrawn (count
    recorded), every task at each spectral level lambda_req = round(f * lambda_2(p), 3). `tag` is inserted after
    H{n} in the names (the revised validation set uses 'v', so that its names never coincide with the first design)."""
    rng = np.random.default_rng(seed)
    p = lattice(n)
    adj = adjacent_pairs(p)
    tasks, redrawn = [], 0
    for kind, count in (('local', n_local), ('perm', n_perm)):
        k = 0
        while k < count:
            perm = swap_perm(rng, n, adj, int(rng.integers(1, 4))) if kind == 'local' else rng.permutation(n)
            if np.array_equal(perm, np.arange(n)) or any(np.array_equal(perm, t[2]) for t in tasks):
                redrawn += 1
                continue
            tasks.append((kind, k, perm))
            k += 1
    lam2 = lambda2(p)
    out = {}
    for kind, k, perm in tasks:
        for f in fracs:
            name = f'H{n}{tag}_{kind}{k}_f{int(round(100 * f))}_s{seed}'
            out[name] = dict(name=name, p=p.copy(), goal=p[perm].copy(), lam=round(f * lam2, 3), n=n,
                             perm=perm.tolist(), kind=kind, frac=f)
    return out, redrawn


# ---------------------------------------------------------------------------------------------------------- nominal
def clip_ball(v):
    nv = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.minimum(1.0, U / np.maximum(nv, 1e-12))


def pair_rows(n, I, J):
    """Row index (1 + triu position) of the collision row of pair {I, J}, vectorised."""
    a, b = np.minimum(I, J), np.maximum(I, J)
    return 1 + a * (2 * n - a - 1) // 2 + (b - a - 1)


class ResidualFlow:
    """run_pa-compatible controller (same constructor and control() contract as GoalProjectionFlow).

    u_nom_i = -e_i + gamma_{i,k}^eff [A_ROT w_i J e_i + A_RAD r_i e_i + A_PSI U mean_{j in N_i} psi_ij (p_i - p_j)/d_ij],
    e_i = p_i - g_i, (w, r, psi) = clip(policy output, -1, 1). gamma_{i,k}^gate in [0, 1) is the conflict-memory gate:
    m_{i,k} <- max(GATE_DECAY m_{i,k-1}, max_{j in N_{i,k}^+} c_{j,k-1}),
    gamma_{i,k}^gate = tanh(GATE_GAIN m_{i,k}); c_{j,k-1} = ||mu_{j,k-1}||_2 (local refit duals, gate='mu') or ||delta_{j,k-1}||_2 (gate='int').
    The policy is a class attribute `policy(obs) -> (node_action (n,2), edge_action (E,))`, set by the caller."""
    policy = None
    gate_kind = 'int'

    def __init__(self, case, dt=.05, kappa=0., margin=.02, curv=False):
        self.goal = np.asarray(case['goal'], float)
        self.n = n = len(self.goal)
        self.kappa, self.margin = kappa, effective_margin(case, margin)
        self.lam_req = case['lam']
        self.mem = np.zeros(n)
        self.u_prev = np.zeros((n, 2))
        self.delta_prev = np.zeros((n, 2))
        self.oracle = None
        self.last_obs = None

    # --- observation ------------------------------------------------------------------------------------------
    def observe(self, p):
        n, g = self.n, self.goal
        e = p - g
        ne = np.linalg.norm(e, axis=1)
        eh = e / np.maximum(ne, 1e-9)[:, None]
        sg = np.minimum(1.0, ne / 0.1)                         # fades goal-frame features near the goal
        Jeh = np.stack([-eh[:, 1], eh[:, 0]], 1)
        dist = np.linalg.norm(p[:, None] - p[None], axis=2)
        I, J = np.nonzero((dist <= RADIUS) & ~np.eye(n, dtype=bool))      # directed edges i -> j
        d = dist[I, J]
        r = (p[J] - p[I]) / d[:, None]
        Jr = np.stack([-r[:, 1], r[:, 0]], 1)
        deg = np.bincount(I, minlength=n)
        v, dl = self.u_prev, self.delta_prev
        duals = self.oracle.duals if self.oracle is not None else np.zeros((n, n))
        sig = np.linalg.norm(duals, axis=1) if self.gate_kind == 'mu' else np.linalg.norm(dl, axis=1)
        raw = sig.copy()
        np.maximum.at(raw, I, sig[J])
        self.mem = np.maximum(GATE_DECAY * self.mem, raw)
        gate = np.tanh(GATE_GAIN * self.mem)
        # exact Laplacian of the frozen channel (spectral information is given by assumption, as for filter L)
        Ii, Jj = pair_indices(n)
        d2 = np.sum((p[Ii] - p[Jj]) ** 2, axis=1)
        ev, vec = np.linalg.eigh(laplacian_from_weights(n, weights_from_squared_distance(d2, RC)))
        lam2 = float(ev[1])
        clear = np.full(n, 1.5)
        np.minimum.at(clear, I, np.minimum(d - 1.0, 1.5))
        x = np.column_stack([
            np.minimum(ne, 4.0), (ne < 0.05).astype(float), gate,
            np.linalg.norm(v, axis=1) / U, np.sum(v * eh, 1) / U * sg, np.sum(v * Jeh, 1) / U * sg,
            np.linalg.norm(dl, axis=1) / U, np.sum(dl * eh, 1) / U * sg, np.sum(dl * Jeh, 1) / U * sg,
            np.full(n, (lam2 - self.lam_req) / self.lam_req), deg / 8.0, clear])
        ea = np.column_stack([
            d / RADIUS, np.clip(d - 1.0, 0.0, 1.5),
            np.sum(r * eh[I], 1) * sg[I], np.sum(r * Jeh[I], 1) * sg[I],
            np.sum(r * eh[J], 1) * sg[J], np.sum(r * Jeh[J], 1) * sg[J], np.minimum(ne[J], 4.0),
            np.sum(eh[I] * eh[J], 1) * sg[I] * sg[J], np.sum(Jeh[I] * eh[J], 1) * sg[I] * sg[J],
            np.sum((v[J] - v[I]) * r, 1) / U, np.sum((v[J] - v[I]) * Jr, 1) / U, gate[J],
            np.linalg.norm(dl[J], axis=1) / U])
        # algebraic features of Chapters 2-4 (filter internals of the previous sample)
        A, b, part, ws, _, _, _ = build_rows(p, self.lam_req + self.margin, True)
        s = self.oracle.s_last if (self.oracle is not None and self.oracle.s_last is not None) else None
        s0 = s[:, 0] if s is not None else np.zeros(n)
        a0n = np.linalg.norm(A[:, 0], axis=1)
        norm0 = U * a0n + np.abs(b[:, 0]) + 1e-3
        pos = 1 + J - (J > I)                                    # position of row (i, j) in member_rows(i)
        posj = 1 + I - (I > J)
        xa = np.column_stack([
            np.log1p(np.abs(duals[:, 0])), np.log1p(np.sum(np.abs(duals[:, 1:]), axis=1)),
            np.clip(s0 / norm0, -3, 3), np.clip((U * a0n - b[:, 0] - s0) / norm0, -3, 3),
            np.log1p(np.linalg.norm(duals, axis=1))])
        led = self.oracle.ledger if self.oracle is not None else {}
        z = np.zeros((len(I), 2))
        for q, (i, j) in enumerate(zip(I, J)):
            ent = led.get((min(i, j), max(i, j)))
            if ent is not None:
                z[q] = ent if i < j else -ent
        # Unit mean-square spectral coordinates (Eq. 5.10): tilde_phi_i = sqrt(N)*phi_i, tilde_Phi_i = sqrt(N)*Phi_i
        phi_tilde = np.sqrt(n) * vec[:, 1]
        Phi_tilde = np.sqrt(n) * vec[:, 1:3]
        eaa = np.column_stack([
            np.log1p(np.abs(duals[I, pos])), np.log1p(np.abs(duals[J, posj])), np.tanh(2 * z[:, 0]),
            np.tanh(2 * z[:, 1]), np.abs(phi_tilde[I] - phi_tilde[J]),
            np.linalg.norm(Phi_tilde[I] - Phi_tilde[J], axis=1), np.sum(Phi_tilde[I] * Phi_tilde[J], 1)])
        obs = dict(x=x.astype(np.float32), xa=xa.astype(np.float32), ea=ea.astype(np.float32),
                   eaa=eaa.astype(np.float32), ei=np.stack([I, J]).astype(np.int64), gate=gate.astype(np.float32),
                   glob=np.array([lam2, self.lam_req, float(np.mean(ne < 0.05)), n / 10.0], np.float32))
        geo = dict(e=e, I=I, J=J, d=d, deg=deg, ne=ne)
        return obs, geo

    # --- nominal ---------------------------------------------------------------------------------------------
    def compose(self, p, geo, gate, node_a, edge_a):
        e, I, J, d, deg = geo['e'], geo['I'], geo['J'], geo['d'], geo['deg']
        w = np.clip(node_a[:, 0], -1, 1)
        rr = np.clip(node_a[:, 1], -1, 1)
        psi = np.clip(edge_a, -1, 1)
        Je = np.stack([-e[:, 1], e[:, 0]], 1)
        away = (p[I] - p[J]) / d[:, None]
        pb = np.zeros_like(e)
        np.add.at(pb, I, psi[:, None] * away)
        pb /= np.maximum(deg, 1)[:, None]
        res = A_ROT * w[:, None] * Je + A_RAD * rr[:, None] * e + A_PSI * U * pb
        return -e + gate[:, None] * res, res

    def control(self, p):
        obs, geo = self.observe(p)
        node_a, edge_a = type(self).policy(obs)
        nominal, res = self.compose(p, geo, obs['gate'], node_a, edge_a)
        u = self.oracle.solve(p, nominal)
        self.delta_prev = u - clip_ball(nominal)
        self.u_prev = u.copy()
        self.last_obs = obs
        gate = obs['gate']
        W = float(.5 * np.sum(geo['e'] ** 2))
        return u, dict(V=W, W=W, u_norm=float(np.linalg.norm(u)), gate_mean=float(gate.mean()),
                       gate_max=float(gate.max()), gate_open=int(np.sum(gate > 0.5)),
                       residual_norm=float(np.linalg.norm(gate[:, None] * res)),
                       intervention=float(np.linalg.norm(self.delta_prev)))


def constant_policy(w=1.0):
    """Fixed-chirality circulation heuristic: w = const, r = 0, psi = 0 (README §5, group H)."""
    def pol(obs):
        n = len(obs['x'])
        return np.column_stack([np.full(n, w), np.zeros(n)]), np.zeros(obs['ei'].shape[1])
    return pol

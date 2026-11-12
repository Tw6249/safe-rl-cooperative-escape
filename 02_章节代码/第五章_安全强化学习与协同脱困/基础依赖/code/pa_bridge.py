"""Linearized spectral and collision barrier rows (build_rows), sensing-graph PA base class, myopic goal-projection
nominal controller, and closed-loop runner (run_pa) with exact-rational segment certification.
"""
from pathlib import Path
import sys, time
sys.dont_write_bytecode = True
import numpy as np
import scipy
import cvxpy as cp
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FROZEN = HERE
sys.path.insert(0, str(FROZEN))
from pair_common import certified_step, dump          # noqa: E402
from audit_trajectories import audit                  # noqa: E402
from scenes import all_cases, effective_margin        # noqa: E402

RC = 10.0                  # polynomial channel w=(1-d^2/RC^2)_+^2
GAMMA_C = GAMMA_S = 2.0    # alpha_collision = alpha_spectral
U = 0.5 - 1e-7
WS_LIMIT = 4.0
WS_A = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
SOLVE = dict(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=300)
CERT_TOL = 2e-8
GAP_FLOOR = 1e-6


class SeedInfeasible(RuntimeError):
    pass


class DegenerateSpectrum(RuntimeError):
    pass


class GoalProjectionFlow:
    """Myopic goal-projection nominal controller: u_nom = -(p - goal)."""

    def __init__(self, case, dt=.05, kappa=0., margin=.02, curv=False):
        self.goal = np.asarray(case['goal'], float)
        self.n = len(self.goal)
        self.kappa = 0.0
        self.margin = effective_margin(case, margin)
        self.oracle = None

    def control(self, p):
        e = p - self.goal
        u = self.oracle.solve(p, -e)
        W = float(0.5 * np.sum(e ** 2))
        return u, dict(V=W, W=W, u_norm=float(np.linalg.norm(u)), grad_norm=float(np.linalg.norm(e)))


BraidFlow = GoalProjectionFlow


def lambda2_data(p):
    n = len(p)
    I, J = np.triu_indices(n, 1)
    rel = p[I] - p[J]
    d2 = np.sum(rel * rel, axis=1)
    base = np.maximum(0.0, 1.0 - d2 / RC**2)
    w = base**2
    lap = np.zeros((n, n))
    lap[I, J] = -w
    lap[J, I] = -w
    lap[np.arange(n), np.arange(n)] = w_sum(n, I, J, w)
    ev, vec = np.linalg.eigh(lap)
    gap = ev[2] - ev[1] if n > 2 else np.inf
    phi = vec[:, 1]
    grad_w = -4.0 / RC**2 * base[:, None] * rel
    contrib = ((phi[I] - phi[J]) ** 2)[:, None] * grad_w
    g = np.zeros((n, 2))
    np.add.at(g, I, contrib)
    np.add.at(g, J, -contrib)
    return float(ev[1]), float(gap), g


def w_sum(n, I, J, w):
    s = np.zeros(n)
    np.add.at(s, I, w)
    np.add.at(s, J, w)
    return s


def build_rows(p, lam_eff, clip_level=False):
    """Rows a_i u_i + b_i <= 0 summing over participants. Row 0: Fiedler row; rows 1..P: pair collision."""
    n = len(p)
    I, J = np.triu_indices(n, 1)
    P = len(I)
    m = 1 + P
    A = np.zeros((n, m, 2))
    b = np.zeros((n, m))
    part = np.zeros((n, m), bool)
    lam2, gap, g = lambda2_data(p)
    if n > 2 and gap <= GAP_FLOOR:
        raise DegenerateSpectrum(f'lambda_3-lambda_2={gap:.3e}')
    h = max(lam2 - lam_eff, 0.0) if clip_level else lam2 - lam_eff
    A[:, 0] = -g
    b[:, 0] = -GAMMA_S * h / n
    part[:, 0] = True
    for r, (i, j) in enumerate(zip(I, J), 1):
        rel = p[i] - p[j]
        hc = float(rel @ rel) - 1.0
        A[i, r] = -2.0 * rel
        A[j, r] = 2.0 * rel
        b[i, r] = b[j, r] = -GAMMA_C * hc / 2.0
        part[[i, j], r] = True
    ws = np.column_stack([-2 * (WS_LIMIT - p[:, 0]), -2 * (WS_LIMIT + p[:, 0]),
                          -2 * (WS_LIMIT - p[:, 1]), -2 * (WS_LIMIT + p[:, 1])])
    return A, b, part, ws, lam2, gap, (I, J)


class EdgeProblem:
    """Two-endpoint proximal update (Chapter 3): joint hard rows, shared rows carry one transfer variable each."""

    def __init__(self, i, j, m, shared, eta):
        self.i, self.j, self.shared, self.eta_value = i, j, shared, eta
        k = len(shared)
        self.ui, self.uj, self.delta = cp.Variable(2), cp.Variable(2), cp.Variable(k)
        self.Ai, self.Aj = cp.Parameter((m, 2)), cp.Parameter((m, 2))
        self.ci, self.cj = cp.Parameter(m), cp.Parameter(m)
        self.ni, self.nj = cp.Parameter(2), cp.Parameter(2)
        self.wi, self.wj = cp.Parameter(4), cp.Parameter(4)
        E = np.zeros((m, k))
        E[shared, np.arange(k)] = 1.0
        cons = [self.Ai @ self.ui + self.ci + E @ self.delta <= 0,
                self.Aj @ self.uj + self.cj - E @ self.delta <= 0,
                WS_A @ self.ui + self.wi <= 0, WS_A @ self.uj + self.wj <= 0,
                cp.norm(self.ui) <= U, cp.norm(self.uj) <= U]
        obj = (0.5 * cp.sum_squares(self.ui - self.ni) + 0.5 * cp.sum_squares(self.uj - self.nj)
               + cp.sum_squares(self.delta) / (2 * eta))
        self.problem = cp.Problem(cp.Minimize(obj), cons)

    def solve(self, A, b, part, ws, nominal, s):
        i, j = self.i, self.j
        for A_p, c_p, idx in ((self.Ai, self.ci, i), (self.Aj, self.cj, j)):
            mask = part[idx]
            A_p.value = np.where(mask[:, None], A[idx], 0.0)
            c_p.value = np.where(mask, b[idx] + s[idx], -1.0)
        self.ni.value, self.nj.value = nominal[i], nominal[j]
        self.wi.value, self.wj.value = ws[i], ws[j]
        self.problem.solve(**SOLVE)
        if self.problem.status != cp.OPTIMAL:
            raise RuntimeError(f'edge status {self.problem.status}')
        pen = float(self.delta.value @ self.delta.value) / (2 * self.eta_value)
        return self.ui.value.copy(), self.uj.value.copy(), self.delta.value.copy(), pen


class PAOracle:
    """Base safety filter oracle: solve(p, nominal)."""

    def __init__(self, n, lam_eff, sweeps, comm_radius, eta=2.0, clip_level=False):
        self.n, self.lam_eff, self.sweeps, self.comm_radius, self.eta = n, lam_eff, sweeps, comm_radius, eta
        self.clip_level = clip_level
        self.m = 1 + n * (n - 1) // 2
        I, J = np.triu_indices(n, 1)
        self.pair_row = {(int(i), int(j)): r for r, (i, j) in enumerate(zip(I, J), 1)}
        self.edge_cache = {}
        self.count = 0
        self.max_residual = 0.0
        self.last = {}
        self.totals = dict(accepted=0, rejected=0, transfer_scalars=0)

    def comm_edges(self, p):
        n = self.n
        dist = np.linalg.norm(p[:, None] - p[None], axis=2)
        mst = minimum_spanning_tree(dist + 1e-12 * (1 - np.eye(n))).toarray()
        edges = {(i, j) for i in range(n) for j in range(i + 1, n)
                 if dist[i, j] <= self.comm_radius or mst[i, j] > 0 or mst[j, i] > 0}
        return sorted(edges)

    def edge_problem(self, i, j):
        key = (i, j)
        if key not in self.edge_cache:
            self.edge_cache[key] = EdgeProblem(i, j, self.m, [0, self.pair_row[key]], self.eta)
        return self.edge_cache[key]

    def residuals(self, A, b, part, ws, u, s):
        loc = np.einsum('nmd,nd->nm', A, u) + b + s
        return dict(local=float(max(0., np.max(loc[part]))),
                    glob=float(max(0., np.max(np.sum(np.where(part, np.einsum('nmd,nd->nm', A, u) + b, 0.), axis=0)))),
                    ws=float(max(0., np.max(np.einsum('ij,nj->ni', WS_A, u) + ws))),
                    ball=float(max(0., np.max(np.linalg.norm(u, axis=1) - U))),
                    alloc=float(np.max(np.abs(np.sum(np.where(part, s, 0.), axis=0)))))

    def central(self, A, b, part, ws, nominal):
        u = cp.Variable((self.n, 2))
        cons = [cp.norm(u, axis=1) <= U, u @ WS_A.T + ws <= 0]
        for r in range(self.m):
            idx = np.flatnonzero(part[:, r])
            cons.append(sum(A[i, r] @ u[i] + b[i, r] for i in idx) <= 0)
        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(u - nominal)), cons)
        try:
            prob.solve(**SOLVE)
        except Exception:
            return None, None
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or u.value is None:
            return None, None
        return u.value.copy(), float(prob.value)

    def solve(self, p, nominal):
        t0 = time.perf_counter()
        A, b, part, ws, lam2, gap, _ = build_rows(p, self.lam_eff, self.clip_level)
        if getattr(self, 'exact', False):
            uc, jc = self.central(A, b, part, ws, nominal)
            if uc is None:
                raise RuntimeError('exact scalar QP failed')
            self.count += 1
            self.last = dict(lambda2=lam2, eigengap=gap, pa_cost=jc, central_cost=jc, cost_gap=0.0, u_diff=0.0,
                             accepted=0, rejected=0, comm_edges=0, nonneighbor_pairs=0, residual={},
                             filter_seconds=time.perf_counter() - t0,
                             spectral_row_value=float(np.sum(np.einsum('nd,nd->n', A[:, 0], uc) + b[:, 0])),
                             spectral_allocation_max=0.0)
            return uc
        if np.max(b[part]) > 1e-12 or np.max(ws) > 1e-12:
            raise SeedInfeasible(f'zero seed infeasible: max b={np.max(b[part]):.3e}')
        n = self.n
        u = np.zeros((n, 2))
        s = np.zeros((n, self.m))
        edges = self.comm_edges(p)
        cost = lambda x: 0.5 * float(np.sum((x - nominal) ** 2))
        acc = rej = 0
        scalars = 0
        for sweep in range(self.sweeps):
            for (i, j) in (edges if sweep % 2 == 0 else edges[::-1]):
                ep = self.edge_problem(i, j)
                try:
                    ui, uj, dl, pen = ep.solve(A, b, part, ws, nominal, s)
                except Exception:
                    rej += 1
                    continue
                uu, ss = u.copy(), s.copy()
                uu[i], uu[j] = ui, uj
                ss[i, ep.shared] += dl
                ss[j, ep.shared] -= dl
                rr = self.residuals(A, b, part, ws, uu, ss)
                if max(rr.values()) > CERT_TOL or cost(uu) + pen > cost(u) + CERT_TOL:
                    rej += 1
                    continue
                u, s = uu, ss
                acc += 1
                scalars += len(ep.shared)
        rr = self.residuals(A, b, part, ws, u, s)
        self.max_residual = max(self.max_residual, rr['local'], rr['glob'], rr['ws'], rr['ball'])
        uc, jc = self.central(A, b, part, ws, nominal)
        self.count += 1
        self.totals['accepted'] += acc
        self.totals['rejected'] += rej
        self.totals['transfer_scalars'] += scalars
        if uc is None:
            self.totals['reference_failed'] = self.totals.get('reference_failed', 0) + 1
        self.last = dict(lambda2=lam2, eigengap=gap, pa_cost=cost(u),
                         central_cost=jc if uc is not None else float('nan'),
                         cost_gap=cost(u) - jc if uc is not None else float('nan'),
                         u_diff=float(np.linalg.norm(u - uc)) if uc is not None else float('nan'),
                         accepted=acc, rejected=rej, comm_edges=len(edges),
                         nonneighbor_pairs=self.n * (self.n - 1) // 2 - len(edges),
                         residual=rr, filter_seconds=time.perf_counter() - t0,
                         spectral_row_value=float(np.sum(np.einsum('nd,nd->n', A[:, 0], u) + b[:, 0])),
                         spectral_allocation_max=float(np.max(np.abs(s[:, 0]))))
        return u


class SensingGraphPA(PAOracle):
    """PA filter with transactions on pairs within the sensing radius R_sense = 2.5 m."""

    def comm_edges(self, p):
        n = self.n
        dist = np.linalg.norm(p[:, None] - p[None], axis=2)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if dist[i, j] <= self.comm_radius]
        adj = np.zeros((n, n))
        for i, j in edges:
            adj[i, j] = adj[j, i] = 1
        self._components = int(connected_components(adj, directed=False)[0])
        return edges

    def solve(self, p, nominal):
        u = super().solve(p, nominal)
        self.last['comm_components'] = self._components
        return u


def run_pa(case, out, kappa, sweeps, comm_radius, horizon, tag, dt0=0.05, margin_cap=0.02,
           checkpoint=None, budget=None, clip_level=False, exact_scalar=False):
    out = Path(out)
    p = case['p'].copy()
    g = case['goal']
    n = len(p)
    ctl = BraidFlow(case, dt0, kappa, margin_cap, False)
    lam_eff = case['lam'] + ctl.margin
    ctl.oracle = PAOracle(n, lam_eff, sweeps, comm_radius, clip_level=clip_level)
    ctl.oracle.exact = exact_scalar
    start = time.perf_counter()
    ts, states, trace = [0.], [p.copy()], []
    status, failure, arrival, cuts_total, min_step = 'horizon_reached', None, None, 0, dt0
    quiet = 0
    wall_before = 0.0
    chunks = 1
    while ts[-1] < horizon - 1e-9:
        t = ts[-1]
        try:
            u, info = ctl.control(p)
        except SeedInfeasible as exc:
            status, failure = 'seed_infeasible', repr(exc)
            break
        except DegenerateSpectrum as exc:
            status, failure = 'degenerate_spectrum', repr(exc)
            break
        except Exception as exc:
            status, failure = 'solver_failure', repr(exc)
            break
        try:
            z, dt, cuts, bnd = certified_step(p, u, case['lam'], min(dt0, horizon - t))
        except Exception as exc:
            status, failure = 'certification_stop', repr(exc)
            break
        cuts_total += cuts
        min_step = min(min_step, dt)
        info.update(t=t, dt=dt, cuts=cuts, min_distance=bnd['min_distance'], lambda2_lower=bnd['lambda2_lower'],
                    **{k: v for k, v in ctl.oracle.last.items()})
        trace.append(info)
        ts.append(t + dt)
        states.append(z.copy())
        p = z
        if np.linalg.norm(p - g, axis=1).max() <= .01:
            if arrival is None:
                arrival = t + dt
            if t + dt - arrival >= 1 - 1e-9:
                status = 'arrived'
                break
        else:
            arrival = None
        quiet = quiet + 1 if info['u_norm'] < 1e-6 else 0
        if quiet >= 100:
            status = 'stalled'
            break
    name = f"{case['name']}_{tag}"
    np.savez_compressed(out / (name + '.npz'), states=np.array(states), goal=g, times=ts)
    dump(out / (name + '_trace.json'), trace)
    V = np.array([x['V'] for x in trace]) if trace else np.array([np.nan])
    W = np.array([x['W'] for x in trace]) if trace else np.array([np.nan])
    row = dict(scene=case['name'], n=n, tag=tag, kappa=kappa, sweeps=sweeps, comm_radius=comm_radius,
               lambda_required=case['lam'], margin=ctl.margin, status=status, failure=failure,
               duration=ts[-1], arrival=arrival if status == 'arrived' else None, steps=len(trace),
               step_cuts=cuts_total, min_step=min_step,
               W_initial=float(W[0]), W_peak=float(np.nanmax(W)), W_final=float(W[-1]),
               V_initial=float(V[0]), V_final=float(V[-1]),
               max_V_increase=float(max(np.diff(V).max(), 0.)) if len(V) > 1 else 0.,
               min_distance=min((x['min_distance'] for x in trace), default=None),
               min_lambda2_lower=float(np.nanmin([x['lambda2_lower'] for x in trace])) if trace else None,
               min_eigengap=float(min(x['eigengap'] for x in trace)) if trace else None,
               min_spectral_slack=float(min(x['lambda2'] - lam_eff for x in trace)) if trace else None,
               pa_transactions_accepted=ctl.oracle.totals['accepted'], pa_transactions_rejected=ctl.oracle.totals['rejected'],
               transfer_scalars=ctl.oracle.totals['transfer_scalars'],
               max_filter_residual=ctl.oracle.max_residual,
               reference_failed=ctl.oracle.totals.get('reference_failed', 0),
               max_cost_gap=float(np.nanmax([x['cost_gap'] for x in trace])) if trace else None,
               median_cost_gap=float(np.nanmedian([x['cost_gap'] for x in trace])) if trace else None,
               max_u_diff=float(np.nanmax([x['u_diff'] for x in trace])) if trace else None,
               mean_comm_edges=float(np.mean([x['comm_edges'] for x in trace])) if trace else None,
               mean_nonneighbor_angle_pairs=float(np.mean([x['nonneighbor_pairs'] for x in trace])) if trace else None,
               median_filter_seconds=float(np.median([x['filter_seconds'] for x in trace])) if trace else None,
               wall_seconds=wall_before + time.perf_counter() - start)
    row['clip_level'] = bool(clip_level)
    row['exact_scalar'] = bool(exact_scalar)
    dump(out / (name + '.json'), row)
    return row

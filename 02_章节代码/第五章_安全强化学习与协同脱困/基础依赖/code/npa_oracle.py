"""Thesis Chapter-3 NPA (Neighborhood Proximal Allocation) as the Chapter-5 safety filter on the scalar rows.

Algorithm source (01_论文写作/独立章节/第三章_持久分配/sections):
  04_algorithm.tex  eq. (ch3star)       closed-neighbourhood proximal subproblem of every centre i
                    eq. (ch3starweight) beta_i = 1 / max_{k in N_i^+}(d_k + 1),  gamma_k = 1 - sum_{i in N_k^+ & A} beta_i
                    eq. (ch3starcommit) fused commit of decisions and allocations
                    Table ch3algorithm  'verify': candidate certificates and local objective not worse than the anchor
  06_control.tex    eq. (ch3reuse)-(ch3refit) cross-sample allocation inheritance with a local refit at every node;
                    if (ch3reuse) fails, fall back to the zero-speed seed (u = 0, s = 0).
Round mechanics follow the Chapter-3 evidence code run_chapter3.run_star_pa_graph: every non-isolated centre solves,
a failed solve leaves the active set, fusion by beta weights, round-level residual + descent check with CERT_TOL.
That function cannot be called directly (it is written for the Chapter-3 frozen benchmark: box inputs and a balance
equality); compute_star_betas is imported from it unchanged.

Chapter-5 data (unchanged from the bridge, pa_bridge.build_rows with clip_level=True): simple-root Fiedler row 0,
pairwise collision rows, private sets |u_i| <= U and workspace rows, l_i = 0.5 |u_i - nominal_i|^2, eta = 2.
Communication graph G_c: the 2.5 m sensing graph (pa_bridge.SensingGraphPA.comm_edges). Rows shared by an edge
(i, j): part[i] & part[j] = {spectral row 0, collision row of (i, j)}.
"""
from pathlib import Path
import sys, time
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LOCAL = HERE.parent
CH3 = ROOT / '01_论文写作' / '独立章节' / '第三章_持久分配' / 'experiments'
sys.path.insert(0, str(HERE))
import numpy as np                                                       # noqa: E402
import cvxpy as cp                                                       # noqa: E402
from pa_bridge import build_rows, SeedInfeasible, WS_A, U, SOLVE, CERT_TOL, SensingGraphPA  # noqa: E402
_saved_path = list(sys.path)
sys.path.insert(0, str(CH3))
import run_chapter3                                                      # noqa: E402,F401
sys.path[:] = _saved_path                    # its own path insertions must not shadow Chapter-5 module names
from run_chapter3 import compute_star_betas                              # noqa: E402

ACCEPTED_STATUS = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)                    # as run_chapter3.solve


def member_rows(a, n, pair_row):
    """Rows robot a participates in: spectral row 0, then its collision rows (a, o), o != a, increasing o."""
    return [0] + [pair_row[(min(a, o), max(a, o))] for o in range(n) if o != a]


def pos_of_pair(a, o):
    """Index of the collision row (a, o) inside member_rows(a)."""
    return 1 + (o if o < a else o - 1)


class StarProblem:
    """Subproblem (ch3star) of centre i with neighbours nbs. Decisions u_hat_k for k in N_i^+; one transfer vector
    delta_{i->j} in R^2 per neighbour (component 0: spectral row, component 1: collision row (i, j))."""

    def __init__(self, i, nbs, n, eta):
        self.i, self.nbs, self.eta = i, tuple(nbs), eta
        self.members = (i,) + self.nbs
        k = len(self.members)
        self.u = cp.Variable((k, 2))
        self.d = [cp.Variable(2) for _ in self.nbs]
        self.A = [cp.Parameter((n, 2)) for _ in self.members]
        self.c = [cp.Parameter(n) for _ in self.members]              # b + s on the member's rows
        self.nom = cp.Parameter((k, 2))
        self.ws = cp.Parameter((k, 4))
        inflow = 0
        for q, j in enumerate(self.nbs):
            E = np.zeros((n, 2))
            E[0, 0] = E[pos_of_pair(i, j), 1] = 1.0                      # R_i^{ij}
            inflow = inflow + E @ self.d[q]
        cons = [self.A[0] @ self.u[0] + self.c[0] + inflow <= 0]          # centre: s_i + sum_j R_i^{ij} delta
        for q, j in enumerate(self.nbs):
            F = np.zeros((n, 2))
            F[0, 0] = F[pos_of_pair(j, i), 1] = 1.0                      # R_j^{ij}
            cons.append(self.A[q + 1] @ self.u[q + 1] + self.c[q + 1] - F @ self.d[q] <= 0)   # s_j - R delta
        for q in range(k):
            cons += [WS_A @ self.u[q] + self.ws[q] <= 0, cp.norm(self.u[q]) <= U]
        obj = 0.5 * cp.sum_squares(self.u - self.nom) + sum(cp.sum_squares(d) for d in self.d) / (2 * eta)
        self.problem = cp.Problem(cp.Minimize(obj), cons)

    def solve(self, A, b, ws, nominal, s, rows_idx):
        for q, a in enumerate(self.members):
            r = rows_idx[a]
            self.A[q].value = A[a, r]
            self.c[q].value = b[a, r] + s[a, r]
        idx = list(self.members)
        self.nom.value = nominal[idx]
        self.ws.value = ws[idx]
        self.problem.solve(**SOLVE)
        if self.problem.status not in ACCEPTED_STATUS or self.u.value is None:
            raise RuntimeError(f'star status {self.problem.status}')
        d = np.array([x.value for x in self.d]).reshape(len(self.nbs), 2)
        return self.u.value.copy(), d, float(np.sum(d * d)) / (2 * self.eta)


class RefitProblem:
    """Local refit (ch3refit) of robot a at a fixed allocation."""

    def __init__(self, n):
        self.u = cp.Variable(2)
        self.A, self.c = cp.Parameter((n, 2)), cp.Parameter(n)
        self.nom, self.ws = cp.Parameter(2), cp.Parameter(4)
        self.problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(self.u - self.nom)),
                                  [self.A @ self.u + self.c <= 0, WS_A @ self.u + self.ws <= 0, cp.norm(self.u) <= U])


class NPAOracle(SensingGraphPA):
    """run_pa-compatible filter: `sweeps` is the NPA round budget B per sample; seed in {'zero', 'inherit'}."""

    def __init__(self, n, lam_eff, sweeps, comm_radius, eta=2.0, clip_level=False, seed='zero'):
        super().__init__(n, lam_eff, sweeps, comm_radius, eta=eta, clip_level=clip_level)
        assert seed in ('zero', 'inherit')
        self.rounds, self.seed_mode = sweeps, seed
        self.rows_idx = [np.array(member_rows(a, n, self.pair_row)) for a in range(n)]
        self.star_cache = {}
        self.refits = [RefitProblem(n) for _ in range(n)]
        self.s_prev = None
        self.history = None                         # set to a list to record every round (checks only)
        self.totals.update(rounds_accepted=0, rounds_rejected=0, stars_rejected=0, inherit_ok=0, inherit_fail=0)

    def refit_all(self, A, b, part, ws, nominal, s):
        """(ch3refit) at every node; on failure returns None and records why in self.refit_failure (logging only)."""
        u = np.zeros((self.n, 2))
        self.refit_failure = None
        for a in range(self.n):
            rp, r = self.refits[a], self.rows_idx[a]
            rp.A.value, rp.c.value = A[a, r], b[a, r] + s[a, r]
            rp.nom.value, rp.ws.value = nominal[a], ws[a]
            try:
                rp.problem.solve(**SOLVE)
            except Exception as exc:
                self.refit_failure = dict(node=a, reason=f'solver exception {exc!r}'[:200])
                return None
            if rp.problem.status not in ACCEPTED_STATUS or rp.u.value is None:
                # smallest row slack reachable inside the disc |u| <= U: < 0 means (ch3reuse) is violated by a row
                need = b[a, r] + s[a, r] - U * np.linalg.norm(A[a, r], axis=1)
                worst = int(np.argmax(need))
                self.refit_failure = dict(node=a, reason=str(rp.problem.status), worst_row=int(r[worst]),
                                          worst_row_min_value=float(need[worst]),
                                          worst_row_allocation=float(s[a, r[worst]]))
                return None
            u[a] = rp.u.value
        rr = self.residuals(A, b, part, ws, u, s)
        if max(rr.values()) > CERT_TOL:
            self.refit_failure = dict(node=None, reason='residual', residual={k: float(v) for k, v in rr.items()})
            return None
        return u

    def solve(self, p, nominal):
        t0 = time.perf_counter()
        A, b, part, ws, lam2, gap, _ = build_rows(p, self.lam_eff, self.clip_level)
        if np.max(b[part]) > 1e-12 or np.max(ws) > 1e-12:
            raise SeedInfeasible(f'zero seed infeasible: max b={np.max(b[part]):.3e}')
        n = self.n
        cost = lambda x: 0.5 * float(np.sum((x - nominal) ** 2))
        u, s, seed_kind = np.zeros((n, 2)), np.zeros((n, self.m)), 'zero'
        self.refit_failure = None
        if self.seed_mode == 'inherit' and self.s_prev is not None:
            ur = self.refit_all(A, b, part, ws, nominal, self.s_prev)
            if ur is not None:
                u, s, seed_kind = ur, self.s_prev.copy(), 'inherit'
                self.totals['inherit_ok'] += 1
            else:
                seed_kind = 'zero_fallback'
                self.totals['inherit_fail'] += 1
        seed_cost = cost(u)
        edges = self.comm_edges(p)
        neighbors, degrees, betas = compute_star_betas(n, edges)
        acc_r = rej_r = stars_acc = stars_rej = scalars = 0
        for _ in range(self.rounds):
            props = {}
            for i in range(n):
                nbs = tuple(neighbors[i])
                if not nbs:
                    continue
                sp = self.star_cache.get((i, nbs))
                if sp is None:
                    sp = self.star_cache[(i, nbs)] = StarProblem(i, nbs, n, self.eta)
                try:
                    uS, dS, pen = sp.solve(A, b, ws, nominal, s, self.rows_idx)
                except Exception:
                    stars_rej += 1
                    continue
                mem = list(sp.members)
                if 0.5 * float(np.sum((uS - nominal[mem]) ** 2)) + pen > \
                        0.5 * float(np.sum((u[mem] - nominal[mem]) ** 2)) + CERT_TOL:
                    stars_rej += 1                                  # not better than the anchor: not active
                    continue
                props[i] = (sp, uS, dS, pen)
            if not props:
                rej_r += 1
                continue
            uu, wsum, ss, pen_round = np.zeros_like(u), np.zeros(n), s.copy(), 0.0
            for i, (sp, uS, dS, pen) in props.items():
                bi = betas[i]
                for q, k in enumerate(sp.members):
                    uu[k] += bi * uS[q]
                    wsum[k] += bi
                for q, j in enumerate(sp.nbs):
                    r = self.pair_row[(min(i, j), max(i, j))]
                    ss[i, 0] += bi * dS[q, 0]
                    ss[j, 0] -= bi * dS[q, 0]
                    ss[i, r] += bi * dS[q, 1]
                    ss[j, r] -= bi * dS[q, 1]
                pen_round += bi * pen
            gamma = 1.0 - wsum
            assert gamma.min() >= -1e-12, gamma.min()
            uu += gamma[:, None] * u
            rr = self.residuals(A, b, part, ws, uu, ss)
            ok = not (max(rr.values()) > CERT_TOL or cost(uu) + pen_round > cost(u) + CERT_TOL)
            if self.history is not None:
                self.history.append(dict(accepted=ok, cost_before=cost(u), cost_after=cost(uu), penalty=pen_round,
                                         residual=max(rr['local'], rr['glob'], rr['ws'], rr['ball']),
                                         conservation=rr['alloc'], gamma_min=float(gamma.min()), active=len(props)))
            if not ok:
                rej_r += 1
                stars_rej += len(props)
                continue
            u, s = uu, ss
            acc_r += 1
            stars_acc += len(props)
            scalars += sum(2 * len(x[0].nbs) for x in props.values())
        self.s_last = s.copy()
        if self.seed_mode == 'inherit':
            self.s_prev = self.s_last
        npa_seconds = time.perf_counter() - t0
        rr = self.residuals(A, b, part, ws, u, s)
        self.max_residual = max(self.max_residual, rr['local'], rr['glob'], rr['ws'], rr['ball'])
        uc, jc = self.central(A, b, part, ws, nominal)              # diagnostic reference only
        self.count += 1
        self.totals['accepted'] += stars_acc
        self.totals['rejected'] += stars_rej
        self.totals['transfer_scalars'] += scalars
        self.totals['rounds_accepted'] += acc_r
        self.totals['rounds_rejected'] += rej_r
        self.totals['stars_rejected'] += stars_rej
        if uc is None:
            self.totals['reference_failed'] = self.totals.get('reference_failed', 0) + 1
        self.last = dict(lambda2=lam2, eigengap=gap, pa_cost=cost(u),
                         central_cost=jc if uc is not None else float('nan'),
                         cost_gap=cost(u) - jc if uc is not None else float('nan'),
                         u_diff=float(np.linalg.norm(u - uc)) if uc is not None else float('nan'),
                         accepted=stars_acc, rejected=stars_rej, comm_edges=len(edges),
                         nonneighbor_pairs=self.n * (self.n - 1) // 2 - len(edges),
                         residual=rr, filter_seconds=time.perf_counter() - t0, npa_seconds=npa_seconds,
                         spectral_row_value=float(np.sum(np.einsum('nd,nd->n', A[:, 0], u) + b[:, 0])),
                         spectral_allocation_max=float(np.max(np.abs(s[:, 0]))),
                         comm_components=self._components, seed_kind=seed_kind, seed_cost=seed_cost,
                         inherit_failure=self.refit_failure,
                         rounds_accepted=acc_r, rounds_rejected=rej_r,
                         max_degree=max(degrees.values()) if degrees else 0)
        return u

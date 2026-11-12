"""Closed-neighbourhood clearing NPA (README): the N1 filter with the star subproblem replaced by a block G_i^+ = (N_{i,k}^+, E_i) over the
closed neighbourhood. Transferable rows: the spectral row (zero-sum change w_q over the members) and the collision rows
of member pairs e in E_i that are communication neighbours (one antisymmetric change sigma_{q,e} t_e each). Everything else (centre weight vartheta_i commit,
per-block and round checks, cross-sample inheritance with local refit and zero-speed fallback) is NPAOracle's.
"""
from pathlib import Path
import sys, time
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
NPA_CODE = HERE
sys.path.insert(0, str(HERE))
import numpy as np                                                       # noqa: E402
import cvxpy as cp                                                       # noqa: E402
from npa_oracle import (NPAOracle, build_rows, SeedInfeasible, WS_A, U, SOLVE, CERT_TOL,   # noqa: E402
                        ACCEPTED_STATUS, compute_star_betas, pos_of_pair)


class NeighbourhoodBlock:
    def __init__(self, members, pairs, n, eta):
        self.members, self.pairs, self.eta = tuple(members), tuple(pairs), eta
        k, P = len(self.members), len(self.pairs)
        self.u = cp.Variable((k, 2))
        self.w = cp.Variable(k)
        self.t = cp.Variable(P) if P else None
        self.A = [cp.Parameter((n, 2)) for _ in self.members]
        self.c = [cp.Parameter(n) for _ in self.members]
        self.nom = cp.Parameter((k, 2))
        self.ws = cp.Parameter((k, 4))
        e0 = np.zeros(n)
        e0[0] = 1.0
        cons = [cp.sum(self.w) == 0]
        for q, a in enumerate(self.members):
            expr = self.A[q] @ self.u[q] + self.c[q] + e0 * self.w[q]
            if P:
                E = np.zeros((n, P))
                for p, (x, y) in enumerate(self.pairs):
                    if a == x:
                        E[pos_of_pair(x, y), p] = 1.0
                    elif a == y:
                        E[pos_of_pair(y, x), p] = -1.0
                if E.any():
                    expr = expr + E @ self.t
            cons += [expr <= 0, WS_A @ self.u[q] + self.ws[q] <= 0, cp.norm(self.u[q]) <= U]
        pen = cp.sum_squares(self.w) + (2 * cp.sum_squares(self.t) if P else 0)
        self.problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(self.u - self.nom) + pen / (2 * eta)), cons)

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
            raise RuntimeError(f'block status {self.problem.status}')
        w = self.w.value.copy()
        t = self.t.value.copy() if self.pairs else np.zeros(0)
        return self.u.value.copy(), w, t, float(w @ w + 2 * t @ t) / (2 * self.eta)


class BlockNPAOracle(NPAOracle):
    """`sweeps` = round budget B; schedule 'jacobi' (beta commit, the Chapter-3 rule) or 'sequential' (one centre
    at a time, weight 1, alternating order; one round = every centre once; offline comparison only)."""

    def __init__(self, n, lam_eff, sweeps, comm_radius, eta=2.0, clip_level=False, seed='zero', schedule='jacobi'):
        super().__init__(n, lam_eff, sweeps, comm_radius, eta=eta, clip_level=clip_level, seed=seed)
        assert schedule in ('jacobi', 'sequential')
        self.schedule = schedule
        self.block_cache = {}
        self.block_seconds = []

    def block(self, i, nbs, edge_set):
        members = (i,) + tuple(nbs)
        mset = set(members)
        pairs = tuple(sorted(e for e in edge_set if e[0] in mset and e[1] in mset))
        key = (members, pairs)
        bp = self.block_cache.get(key)
        if bp is None:
            bp = self.block_cache[key] = NeighbourhoodBlock(members, pairs, self.n, self.eta)
        return bp

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
        edge_set = set(tuple(e) for e in edges)
        neighbors, degrees, betas = compute_star_betas(n, edges)
        acc_r = rej_r = blocks_acc = blocks_rej = 0

        def propose(i, u, s):
            nbs = tuple(neighbors[i])
            bp = self.block(i, nbs, edge_set)
            tb = time.perf_counter()
            uS, w, t, pen = bp.solve(A, b, ws, nominal, s, self.rows_idx)
            self.block_seconds.append(time.perf_counter() - tb)
            mem = list(bp.members)
            if 0.5 * float(np.sum((uS - nominal[mem]) ** 2)) + pen > \
                    0.5 * float(np.sum((u[mem] - nominal[mem]) ** 2)) + CERT_TOL:
                return None
            return bp, uS, w, t, pen

        def apply(uu, ss, bp, uS, w, t, weight):
            for q, k in enumerate(bp.members):
                uu[k] += weight * uS[q]
                ss[k, 0] += weight * w[q]
            for q, (x, y) in enumerate(bp.pairs):
                r = self.pair_row[(x, y)]
                ss[x, r] += weight * t[q]
                ss[y, r] -= weight * t[q]

        for rnd in range(self.rounds):
            if self.schedule == 'jacobi':
                props = {}
                for i in range(n):
                    if not neighbors[i]:
                        continue
                    try:
                        pr = propose(i, u, s)
                    except Exception:
                        pr = None
                    if pr is None:
                        blocks_rej += 1
                        continue
                    props[i] = pr
                if not props:
                    rej_r += 1
                    continue
                uu, wsum, ss, pen_round = np.zeros_like(u), np.zeros(n), s.copy(), 0.0
                for i, (bp, uS, w, t, pen) in props.items():
                    apply(uu, ss, bp, uS, w, t, betas[i])
                    for k in bp.members:
                        wsum[k] += betas[i]
                    pen_round += betas[i] * pen
                gamma = 1.0 - wsum
                assert gamma.min() >= -1e-12, gamma.min()
                uu += gamma[:, None] * u
                rr = self.residuals(A, b, part, ws, uu, ss)
                if max(rr.values()) > CERT_TOL or cost(uu) + pen_round > cost(u) + CERT_TOL:
                    rej_r += 1
                    blocks_rej += len(props)
                    continue
                u, s = uu, ss
                acc_r += 1
                blocks_acc += len(props)
            else:
                order = range(n) if rnd % 2 == 0 else range(n - 1, -1, -1)
                any_acc = False
                for i in order:
                    if not neighbors[i]:
                        continue
                    try:
                        pr = propose(i, u, s)
                    except Exception:
                        pr = None
                    if pr is None:
                        blocks_rej += 1
                        continue
                    bp, uS, w, t, pen = pr
                    uu, ss = u.copy(), s.copy()
                    mem = list(bp.members)
                    uu[mem] = 0.0
                    apply(uu, ss, bp, uS, w, t, 1.0)
                    rr = self.residuals(A, b, part, ws, uu, ss)
                    if max(rr.values()) > CERT_TOL or cost(uu) + pen > cost(u) + CERT_TOL:
                        blocks_rej += 1
                        continue
                    u, s = uu, ss
                    blocks_acc += 1
                    any_acc = True
                acc_r += int(any_acc)
                rej_r += int(not any_acc)
            if self.history is not None:
                rr_h = self.residuals(A, b, part, ws, u, s)
                self.history.append(dict(cost=cost(u), residual=max(rr_h['local'], rr_h['glob'], rr_h['ws'],
                                                                    rr_h['ball']), conservation=rr_h['alloc']))
        self.s_last = s.copy()
        if self.seed_mode == 'inherit':
            self.s_prev = self.s_last
        npa_seconds = time.perf_counter() - t0
        rr = self.residuals(A, b, part, ws, u, s)
        self.max_residual = max(self.max_residual, rr['local'], rr['glob'], rr['ws'], rr['ball'])
        uc, jc = self.central(A, b, part, ws, nominal)              # diagnostic reference only
        self.count += 1
        self.totals['accepted'] += blocks_acc
        self.totals['rejected'] += blocks_rej
        self.totals['rounds_accepted'] += acc_r
        self.totals['rounds_rejected'] += rej_r
        if uc is None:
            self.totals['reference_failed'] = self.totals.get('reference_failed', 0) + 1
        self.last = dict(lambda2=lam2, eigengap=gap, pa_cost=cost(u),
                         central_cost=jc if uc is not None else float('nan'),
                         cost_gap=cost(u) - jc if uc is not None else float('nan'),
                         u_diff=float(np.linalg.norm(u - uc)) if uc is not None else float('nan'),
                         accepted=blocks_acc, rejected=blocks_rej, comm_edges=len(edges),
                         nonneighbor_pairs=self.n * (self.n - 1) // 2 - len(edges),
                         residual=rr, filter_seconds=time.perf_counter() - t0, npa_seconds=npa_seconds,
                         spectral_row_value=float(np.sum(np.einsum('nd,nd->n', A[:, 0], u) + b[:, 0])),
                         spectral_allocation_max=float(np.max(np.abs(s[:, 0]))),
                         comm_components=self._components, seed_kind=seed_kind, seed_cost=seed_cost,
                         inherit_failure=self.refit_failure, rounds_accepted=acc_r, rounds_rejected=rej_r,
                         max_degree=max(degrees.values()) if degrees else 0)
        return u

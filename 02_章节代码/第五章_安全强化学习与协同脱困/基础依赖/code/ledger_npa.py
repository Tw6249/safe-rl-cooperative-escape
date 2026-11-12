"""Edge-record (varpi_e) settlement on the closed-neighbourhood NPA (group M of 20261029_全邻域NPA); README of this directory.

Every allocation change of a committed block is also booked on the communication edge that carries it:
  spectral transfer vartheta_i w_k of block i to member k  -> edge (i, k) in E_i, credited to k;
  collision transfer vartheta_i t_e on member pair e=(x, y) in E_i -> edge (x, y), +t_e at x (lower index), -t_e at y,
so that s_{ia} = sum_{e ni i} sigma_{i,e} varpi_{e,a} row by row (sigma_{i,e} in {+1, -1}: signed edge-node orientation, +1 at the lower index). Each edge's entry is held by its
two ends only, which makes undoing it a one-hop operation.

  settle='global' (group G, implementation check): bookkeeping only. Inheritance fallback and round check are M's
                  global ones, so u and s must be bit-identical to M.
  settle='local'  (group L): at the start of a sample, entries of edges that left the 2.5 m graph are undone; a robot
                  whose local refit fails undoes all its edges (its allocation becomes 0, so u = 0 is feasible) and
                  its partners refit again; each robot settles at most once per sample. Rounds keep only the per-block
                  checks (block descent as in M, and every member's own residual with the new shares); the global
                  round check and the global refit residual are evaluated for the record and never act.
"""
from pathlib import Path
import sys, time
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BLOCK_CODE = HERE
sys.path.insert(0, str(HERE))
from block_npa import (BlockNPAOracle, build_rows, SeedInfeasible, WS_A, U, SOLVE, CERT_TOL,   # noqa: E402
                       ACCEPTED_STATUS, compute_star_betas)
import numpy as np                                                       # noqa: E402


class LedgerNPAOracle(BlockNPAOracle):

    def __init__(self, n, lam_eff, sweeps, comm_radius, eta=2.0, clip_level=False, settle='local'):
        super().__init__(n, lam_eff, sweeps, comm_radius, eta=eta, clip_level=clip_level, seed='inherit',
                         schedule='jacobi')
        assert settle in ('global', 'local')
        self.settle = settle
        self.ledger = {}                                    # edge (x, y), x < y -> [spectral, collision] booked at x

    # ---- ledger ------------------------------------------------------------------------------------------------
    def undo(self, s, e):
        z = self.ledger.pop(e)
        x, y = e
        r = self.pair_row[e]
        s[x, 0] -= z[0]
        s[y, 0] += z[0]
        s[x, r] -= z[1]
        s[y, r] += z[1]

    def drift(self, s):
        S = np.zeros_like(s)
        for (x, y), z in self.ledger.items():
            r = self.pair_row[(x, y)]
            S[x, 0] += z[0]
            S[y, 0] -= z[0]
            S[x, r] += z[1]
            S[y, r] -= z[1]
        return float(np.max(np.abs(S - s))) if s.size else 0.0

    # ---- local refit of one robot (ch3refit) with its own residual check ------------------------------------------
    def refit_one(self, a, A, b, ws, nominal, s):
        rp, r = self.refits[a], self.rows_idx[a]
        rp.A.value, rp.c.value = A[a, r], b[a, r] + s[a, r]
        rp.nom.value, rp.ws.value = nominal[a], ws[a]
        try:
            rp.problem.solve(**SOLVE)
        except Exception as exc:
            return None, f'solver exception {exc!r}'[:200]
        if rp.problem.status not in ACCEPTED_STATUS or rp.u.value is None:
            return None, str(rp.problem.status)
        ua = rp.u.value.copy()
        res = max(float(np.max(A[a, r] @ ua + b[a, r] + s[a, r])), float(np.max(WS_A @ ua + ws[a])),
                  float(np.linalg.norm(ua) - U))
        if res > CERT_TOL:
            return None, f'residual {res:.3e}'
        return ua, None

    def member_residual(self, bp, A, b, part, ws, uS, s, w, t):
        mem = list(bp.members)
        ssb = s[mem].copy()
        ssb[:, 0] += w
        pos = {k: q for q, k in enumerate(mem)}
        for q, (x, y) in enumerate(bp.pairs):
            r = self.pair_row[(x, y)]
            ssb[pos[x], r] += t[q]
            ssb[pos[y], r] -= t[q]
        loc = np.einsum('kmd,kd->km', A[mem], uS) + b[mem] + ssb
        return max(float(np.max(np.where(part[mem], loc, -np.inf))), float(np.max(uS @ WS_A.T + ws[mem])),
                   float(np.max(np.linalg.norm(uS, axis=1) - U)))

    # ---- one control sample ---------------------------------------------------------------------------------------
    def solve(self, p, nominal):
        t0 = time.perf_counter()
        A, b, part, ws, lam2, gap, _ = build_rows(p, self.lam_eff, self.clip_level)
        if np.max(b[part]) > 1e-12 or np.max(ws) > 1e-12:
            raise SeedInfeasible(f'zero seed infeasible: max b={np.max(b[part]):.3e}')
        n = self.n
        cost = lambda x: 0.5 * float(np.sum((x - nominal) ** 2))
        u, s, seed_kind = np.zeros((n, 2)), np.zeros((n, self.m)), 'zero'
        self.refit_failure = None
        edges = self.comm_edges(p)
        edge_set = set(tuple(e) for e in edges)
        led = dict(broken_settled=0, nodes_settled=0, cascade_depth=0, refit_solves=0, settled_refit_failed=0,
                   settle_leftover=0.0, global_refit_would_fail=None)
        if self.s_prev is not None and self.settle == 'global':
            ur = self.refit_all(A, b, part, ws, nominal, self.s_prev)
            if ur is not None:
                u, s, seed_kind = ur, self.s_prev.copy(), 'inherit'
                self.totals['inherit_ok'] += 1
            else:
                seed_kind = 'zero_fallback'
                self.totals['inherit_fail'] += 1
                self.ledger = {}
        elif self.s_prev is not None:
            s = self.s_prev.copy()
            for e in [e for e in self.ledger if e not in edge_set]:
                self.undo(s, e)
                led['broken_settled'] += 1
            settled, pending, depth = set(), list(range(n)), 0
            while pending:
                failed = []
                for a in pending:
                    ua, why = self.refit_one(a, A, b, ws, nominal, s)
                    led['refit_solves'] += 1
                    if ua is None:
                        failed.append(a)
                        if self.refit_failure is None:
                            self.refit_failure = dict(node=a, reason=why)
                    else:
                        u[a] = ua
                if not failed:
                    break
                depth += 1
                touched = set()
                for a in failed:
                    for e in [e for e in self.ledger if a in e]:
                        self.undo(s, e)
                        touched.update(e)
                    led['settle_leftover'] = max(led['settle_leftover'], float(np.max(np.abs(s[a]))))
                    s[a] = 0.0                              # removes the rounding left by the undo (recorded above)
                    settled.add(a)
                for a in failed:
                    ua, _ = self.refit_one(a, A, b, ws, nominal, s)
                    led['refit_solves'] += 1
                    if ua is None:
                        ua = np.zeros(2)
                        led['settled_refit_failed'] += 1
                    u[a] = ua
                pending = sorted(touched - settled)
            led['nodes_settled'], led['cascade_depth'] = len(settled), depth
            seed_kind = 'inherit' if not settled else ('all_settled' if len(settled) == n else 'partial')
            rr0 = self.residuals(A, b, part, ws, u, s)
            led['global_refit_would_fail'] = bool(max(rr0.values()) > CERT_TOL)
        seed_cost = cost(u)
        neighbors, degrees, betas = compute_star_betas(n, edges)
        acc_r = rej_r = blocks_acc = blocks_rej = would_reject = member_rejects = 0

        def propose(i, u, s):
            nonlocal member_rejects
            nbs = tuple(neighbors[i])
            bp = self.block(i, nbs, edge_set)
            tb = time.perf_counter()
            uS, w, t, pen = bp.solve(A, b, ws, nominal, s, self.rows_idx)
            self.block_seconds.append(time.perf_counter() - tb)
            mem = list(bp.members)
            if 0.5 * float(np.sum((uS - nominal[mem]) ** 2)) + pen > \
                    0.5 * float(np.sum((u[mem] - nominal[mem]) ** 2)) + CERT_TOL:
                return None
            if self.settle == 'local' and self.member_residual(bp, A, b, part, ws, uS, s, w, t) > CERT_TOL:
                member_rejects += 1
                return None
            return bp, uS, w, t, pen

        def apply(uu, ss, book, i, bp, uS, w, t, weight):
            for q, k in enumerate(bp.members):
                uu[k] += weight * uS[q]
                ss[k, 0] += weight * w[q]
                if k != i:
                    e = (min(i, k), max(i, k))
                    book.setdefault(e, np.zeros(2))[0] += weight * w[q] if k == e[0] else -weight * w[q]
            for q, (x, y) in enumerate(bp.pairs):
                r = self.pair_row[(x, y)]
                ss[x, r] += weight * t[q]
                ss[y, r] -= weight * t[q]
                book.setdefault((x, y), np.zeros(2))[1] += weight * t[q]

        for rnd in range(self.rounds):
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
            uu, wsum, ss, pen_round, book = np.zeros_like(u), np.zeros(n), s.copy(), 0.0, {}
            for i, (bp, uS, w, t, pen) in props.items():
                apply(uu, ss, book, i, bp, uS, w, t, betas[i])
                for k in bp.members:
                    wsum[k] += betas[i]
                pen_round += betas[i] * pen
            gamma = 1.0 - wsum
            assert gamma.min() >= -1e-12, gamma.min()
            uu += gamma[:, None] * u
            rr = self.residuals(A, b, part, ws, uu, ss)
            global_reject = max(rr.values()) > CERT_TOL or cost(uu) + pen_round > cost(u) + CERT_TOL
            if global_reject and self.settle == 'global':
                rej_r += 1
                blocks_rej += len(props)
                continue
            would_reject += int(global_reject)
            u, s = uu, ss
            for e, z in book.items():
                self.ledger.setdefault(e, np.zeros(2))[:] += z
            acc_r += 1
            blocks_acc += len(props)
        self.s_last = s.copy()
        self.s_prev = self.s_last
        npa_seconds = time.perf_counter() - t0
        led['ledger_drift'] = self.drift(s)
        led['ledger_edges'] = len(self.ledger)
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
                         max_degree=max(degrees.values()) if degrees else 0,
                         would_reject_rounds=would_reject, member_rejects=member_rejects, **led)
        return u

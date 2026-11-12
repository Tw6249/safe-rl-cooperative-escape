"""Chapter-3 unified NPA (Neighborhood Proximal Allocation) evidence; never overwrites PA source data or an existing run.

Invoke from any directory: python run_chapter3.py --output runs/<new-name>
All experiments evaluate NPA (Neighborhood Proximal Allocation)
and compare objectively against published literature baselines (Xiao Tan IEEE TAC 2025,
Changxin Liu & Xiao Tan Automatica 2026, and Falsone Automatica 2017 Dual Decomposition).
"""
from pathlib import Path
import argparse, hashlib, json, platform, sys, time
import numpy as np
import scipy
from scipy.linalg import null_space
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COMMON = ROOT / '02_核心研究' / '公共基础'
PA_DIR = ROOT / '02_核心研究' / 'Persistent_Allocations'
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(PA_DIR / 'code'))
sys.path.insert(0, str(PA_DIR / 'experimental_study_20260919'))
from thesis_common.model import (ChannelConfig, build_graph,
    simple_lambda2_gradient, single_integrator_cbf_rows, settings_digest)

SOLVE = dict(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10,
             tol_gap_rel=1e-10, max_iter=300)
CERT_TOL = 2e-8


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def solve(problem):
    problem.solve(**SOLVE)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f'Unaccepted solver status: {problem.status}')


def compute_star_betas(n, edges):
    neighbors = {i: [] for i in range(n)}
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    for i in range(n):
        neighbors[i] = sorted(neighbors[i])
    degrees = {i: len(neighbors[i]) for i in range(n)}
    betas = {
        i: 1.0 / (max([degrees[i]] + [degrees[j] for j in neighbors[i]]) + 1.0)
        if neighbors[i] else 0.0
        for i in range(n)
    }
    return neighbors, degrees, betas


def rank_example(out):
    """Example 3.1 (N=7 redundant equalities) & Examples 3.2-3.3 (N=3 rank/cusp deadlocks):
    Compare NPA against Tan-TAC 2025 (discretized continuous-time violation-free flow) and
    pairwise 2-node exchange (Wu-Auto 2023).
    """
    n = 7
    eta = 2.0
    c = np.array([1.5, -1.0, 0.7, -0.6, 1.1, -1.3, 0.2])
    optimum = c - c.mean()
    edges = [(i, i + 1) for i in range(n - 1)]
    neighbors, degrees, betas = compute_star_betas(n, edges)
    cost = lambda v: 0.5 * float(np.sum((v - c) ** 2))
    max_rounds = 400

    def run_npa_ex1(eta_val):
        t_cur = np.zeros(n)
        hist = [cost(t_cur) - cost(optimum)]
        cons = [0.0]
        for _ in range(max_rounds):
            d_prop = {}
            for i in range(n):
                Ni = neighbors[i]
                deg = len(Ni)
                alpha = 1.0 + 5.0 / eta_val
                rhs = -(t_cur[i] - c[i]) + (t_cur[Ni] - c[Ni])
                d_vec = (rhs - np.sum(rhs) / (alpha + deg)) / alpha
                d_prop[i] = {j: float(d_vec[idx]) for idx, j in enumerate(Ni)}
            t_next = t_cur.copy()
            for i in range(n):
                bi = betas[i]
                for j in neighbors[i]:
                    t_next[i] += bi * d_prop[i][j]
                    t_next[j] -= bi * d_prop[i][j]
            t_cur = t_next
            hist.append(cost(t_cur) - cost(optimum))
            cons.append(float(abs(t_cur.sum())))
        return t_cur, hist, cons

    # 1. NPA on N=7 path graph with 2 redundant equality rows (eta = 2.0 and eta = 20.0):
    t_star, hist_star, cons_star = run_npa_ex1(2.0)
    t_star_eta20, hist_star_eta20, _ = run_npa_ex1(20.0)

    # 2. Tan-TAC 2025 (continuous-time violation-free gradient flow discretized) on N=7:
    # Local row matrix A_i = [1; 2] is 2x1 (rank 1 < 2 rows, violating full-row-rank assumption).
    # Minimum-norm Moore-Penrose multiplier pseudoinverse gives:
    #   mu_i = (A_i A_i^T)^+ A_i (z_i - c_i) = (1/5) * [1; 2] * (t_i - c_i).
    # Because of the 1/5 factor from (A_i A_i^T)^+, an uncompensated s-space step size
    # h_s = 1 / lambda_max(L_path) yields effective step h_eff = 1 / (5 * lambda_max(L_path)),
    # whereas compensating for row scaling gives standard spectral step h_eff = 1 / lambda_max(L_path).
    L_path = np.zeros((n, n))
    for i, j in edges:
        L_path[i, i] += 1.0; L_path[j, j] += 1.0
        L_path[i, j] -= 1.0; L_path[j, i] -= 1.0
    eigs_path = np.linalg.eigvalsh(L_path)
    lam_max_path = float(eigs_path[-1])
    lam_2_path = float(eigs_path[1])

    # (a) Standard spectral step size in effective coordinate (h_eff = 1 / lambda_max)
    h_tan_spectral = 1.0 / lam_max_path
    t_tan = np.zeros(n)
    hist_tan = [cost(t_tan) - cost(optimum)]
    for _ in range(max_rounds):
        t_tan = t_tan - h_tan_spectral * (L_path @ (t_tan - c))
        hist_tan.append(cost(t_tan) - cost(optimum))

    # (b) Uncompensated s-space step size (h_s = 1 / lambda_max => h_eff = 1 / (5 * lambda_max))
    h_tan_unscaled = 1.0 / (5.0 * lam_max_path)
    t_tan_unscaled = np.zeros(n)
    hist_tan_unscaled = [cost(t_tan_unscaled) - cost(optimum)]
    for _ in range(max_rounds):
        t_tan_unscaled = t_tan_unscaled - h_tan_unscaled * (L_path @ (t_tan_unscaled - c))
        hist_tan_unscaled.append(cost(t_tan_unscaled) - cost(optimum))

    # Explicit rank calculation in the 2N physical allocation coordinates for NPA
    constraint = []
    for i in range(n):
        row = np.zeros(2 * n); row[2 * i] = -2; row[2 * i + 1] = 1; constraint.append(row)
    for q in range(2):
        row = np.zeros(2 * n); row[q::2] = 1; constraint.append(row)
    basis = null_space(np.array(constraint))
    star_modes = []
    for i in range(n):
        for j in neighbors[i]:
            mode = np.zeros(2 * n)
            mode[2 * i:2 * i + 2] = [1, 2]
            mode[2 * j:2 * j + 2] = [-1, -2]
            star_modes.append(mode)

    # 3. Example 3.2 (N=3 rank-deficient heterogeneous local domains on sparse path 0 -- 2 -- 1):
    # Genuine CVXPY subproblem solves for NPA vs Pairwise CVXPY (Wu-Auto 2023) vs Tan-TAC 2025 flow.
    edges_ex2 = [(0, 2), (1, 2)]  # Sparse path graph 1 - 3 - 2 (deg(0)=1, deg(1)=1, deg(2)=2)
    neighbors_ex2, _, betas_ex2 = compute_star_betas(3, edges_ex2)
    a_opt = 1.0 / 3.0
    opt_val_ex2 = 0.5 * (2 * a_opt ** 2 + (a_opt - 1.0) ** 2)
    cost_ex2 = lambda s_mat: 0.5 * float(s_mat[0, 0] ** 2 + s_mat[1, 1] ** 2 + (s_mat[2, 0] - 1.0) ** 2)

    def domain_constrs(node_idx, s_var):
        if node_idx == 0:
            return [s_var[1] == 0]
        elif node_idx == 1:
            return [s_var[0] == 0]
        return [s_var[0] == s_var[1]]

    def node_cost_expr(node_idx, s_var):
        if node_idx == 0:
            return 0.5 * cp.square(s_var[0])
        elif node_idx == 1:
            return 0.5 * cp.square(s_var[1])
        return 0.5 * cp.square(s_var[0] - 1.0)

    # 3a. NPA via actual CVXPY subproblem solves on sparse path 1 - 3 - 2
    s_npa_ex2 = np.zeros((3, 2))
    ex2_star_gap = [cost_ex2(s_npa_ex2) - opt_val_ex2]
    for _ in range(40):
        d_proposals = {}
        for i in range(3):
            Ni = neighbors_ex2[i]
            deltas = {j: cp.Variable(2) for j in Ni}
            s_hat_i = s_npa_ex2[i] + sum(deltas.values())
            constrs = domain_constrs(i, s_hat_i)
            obj_terms = [node_cost_expr(i, s_hat_i)]
            for j in Ni:
                s_hat_j = s_npa_ex2[j] - deltas[j]
                constrs += domain_constrs(j, s_hat_j)
                obj_terms.append(node_cost_expr(j, s_hat_j))
                obj_terms.append(cp.sum_squares(deltas[j]) / (2.0 * eta))
            prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constrs)
            solve(prob)
            d_proposals[i] = {j: deltas[j].value.copy() for j in Ni}
        s_next = s_npa_ex2.copy()
        for i in range(3):
            bi = betas_ex2[i]
            for j in neighbors_ex2[i]:
                s_next[i] += bi * d_proposals[i][j]
                s_next[j] -= bi * d_proposals[i][j]
        s_npa_ex2 = s_next
        ex2_star_gap.append(cost_ex2(s_npa_ex2) - opt_val_ex2)

    # 3b. Pairwise 2-node CVXPY exchange (Wu-Auto 2023) on all edges of K_3
    s_pair_ex2 = np.zeros((3, 2))
    ex2_pair_gap = [cost_ex2(s_pair_ex2) - opt_val_ex2]
    for _ in range(40):
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            delta = cp.Variable(2)
            s_hat_i = s_pair_ex2[i] + delta
            s_hat_j = s_pair_ex2[j] - delta
            constrs = domain_constrs(i, s_hat_i) + domain_constrs(j, s_hat_j)
            obj = node_cost_expr(i, s_hat_i) + node_cost_expr(j, s_hat_j) + cp.sum_squares(delta) / (2.0 * eta)
            prob = cp.Problem(cp.Minimize(obj), constrs)
            solve(prob)
            s_pair_ex2[i] += delta.value
            s_pair_ex2[j] -= delta.value
        ex2_pair_gap.append(cost_ex2(s_pair_ex2) - opt_val_ex2)

    # 3c. Tan-TAC 2025 first-order Laplacian multiplier flow with local domain certification
    s_tan_ex2 = np.zeros((3, 2))
    ex2_tan_gap = [cost_ex2(s_tan_ex2) - opt_val_ex2]
    for _ in range(40):
        mu = np.zeros((3, 2))
        mu[0] = np.array([s_tan_ex2[0, 0], 0.0])
        mu[1] = np.array([0.0, s_tan_ex2[1, 1]])
        g2 = 0.5 * (s_tan_ex2[2, 0] - 1.0)
        mu[2] = np.array([g2, g2])
        ds = np.zeros((3, 2))
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            diff = mu[i] - mu[j]
            ds[i] -= diff; ds[j] += diff
        step = 0.2
        while step >= 1e-8:
            cand = s_tan_ex2 + step * ds
            viol = max(abs(cand[0, 1]), abs(cand[1, 0]), abs(cand[2, 0] - cand[2, 1]))
            if viol <= 1e-10:
                s_tan_ex2 = cand
                break
            step *= 0.5
        ex2_tan_gap.append(cost_ex2(s_tan_ex2) - opt_val_ex2)

    np.savez(
        out / 'relative_domain.npz',
        gap=np.array(hist_star),
        gap_eta20=np.array(hist_star_eta20),
        gap_tan=np.array(hist_tan),
        gap_tan_unscaled=np.array(hist_tan_unscaled),
        conservation=np.array(cons_star),
        final=t_star,
        exact=optimum,
        ex2_star_gap=np.array(ex2_star_gap),
        ex2_tan_gap=np.array(ex2_tan_gap),
        ex2_pair_gap=np.array(ex2_pair_gap),
    )
    return dict(
        n=n, eta=eta, c=c.tolist(), rounds=max_rounds,
        initial_gap=hist_star[0],
        final_gap=hist_star[-1],
        gap_round120=hist_star[120],
        gap_eta20_round120=hist_star_eta20[120],
        final_gap_eta20=hist_star_eta20[-1],
        tan_spectral_gap_round120=hist_tan[120],
        tan_final_gap=hist_tan[-1],
        tan_unscaled_gap_round120=hist_tan_unscaled[120],
        tan_unscaled_final_gap=hist_tan_unscaled[-1],
        final_t=t_star.tolist(), exact_t=optimum.tolist(),
        solution_error=float(np.linalg.norm(t_star - optimum)),
        affine_dimension=basis.shape[1],
        star_rank=int(np.linalg.matrix_rank(np.array(star_modes).T)),
        max_conservation=float(max(cons_star)),
        max_cost_increase=float(max(np.diff(hist_star))),
        ex2_round10_star_gap=float(ex2_star_gap[10]),
        ex2_final_star_gap=float(ex2_star_gap[-1]),
        ex2_final_tan_gap=float(ex2_tan_gap[-1]),
        ex2_final_pair_gap=float(ex2_pair_gap[-1]),
    )


def obstructions(out):
    vectors = [np.array([1., 0.]), np.array([0., 1.]), np.array([1., 1.])]
    C = np.column_stack(vectors); L = null_space(C)
    pair_dims = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        pair_dims.append(null_space(np.column_stack([vectors[i], vectors[j]])).shape[1])
    a = 1 / 3; origin = 0.5; optimum = 0.5 * (a * a + a * a + (a - 1) ** 2)
    v = np.array([0.5, 1.5, 2.5]); cusp = lambda s: 0.5 * np.dot(s, s) + v @ s + 0.5 * np.abs(s).sum()
    witness = np.array([0.5, 0., -0.5])
    eps = 0.1; y = -1 + np.sqrt(1 - eps ** 2)
    return dict(
        rank_deadlock=dict(
            global_dimension=L.shape[1], pair_dimensions=pair_dims,
            star_dimension=L.shape[1],
            initial_cost=origin, three_node_optimum=optimum, strict_gap=origin - optimum
        ),
        cusp=dict(origin_cost=cusp(np.zeros(3)), nonedge_witness=witness.tolist(), witness_cost=cusp(witness)),
        tangent_disk=dict(point=[eps, y], residual=eps ** 2 + (y + 1) ** 2 - 1, objective=-eps - y),
    )


def make_problem():
    p = np.array([[0., 0.], [.68, .09], [1.35, -.12], [.18, .80], [.93, .91], [1.65, .69]])
    n = len(p); config = ChannelConfig.from_settings(); g = build_graph(p, config)
    lam, grad = simple_lambda2_gradient(g, gap_tolerance=1e-7)
    lambda_req = 0.90 * lam; gain = 1.5; d_safe = 0.38
    edges = [(i, j) for i, j in g.candidate_edges if g.adjacency[i, j] > 0]
    pairs = list(g.candidate_edges); m = 1 + len(pairs)
    A = np.zeros((n, m, 2)); bb = np.zeros((n, m)); part = np.zeros((n, m), bool)
    A[:, 0], bb[:, 0] = single_integrator_cbf_rows(grad, h=lam - lambda_req, gain=gain)
    part[:, 0] = True
    for row, (i, j) in enumerate(pairs, 1):
        vec = p[i] - p[j]; h = vec @ vec - d_safe * d_safe
        aa, b = single_integrator_cbf_rows(np.stack([2 * vec, -2 * vec]), h=h, gain=gain)
        A[[i, j], row] = aa; bb[[i, j], row] = b; part[[i, j], row] = True
    limits = np.array([[.45, .32], [.28, .42], [.50, .35], [.36, .48], [.30, .30], [.46, .38]])
    nom = 1.1 * (p - p.mean(axis=0)) + np.array([[.1, -.15], [-.2, .1], [.1, .15], [.12, -.1], [-.15, .1], [.1, -.1]])
    return p, g, A, bb, part, limits, nom, edges, dict(lambda2=lam, lambda_req=lambda_req, gain=gain, d_safe=d_safe)


def run_star_pa_graph(p, A, bb, part, limits, nom, edges, ustar, jstar, eta=2., max_rounds=64):
    n, m, _ = A.shape
    cost = lambda uu: 0.5 * float(np.sum((uu - nom) ** 2))

    def residual(uu, ss, ee):
        loc = np.einsum('nmd,nd->nm', A, uu) + bb + ss
        return dict(
            local=float(max(0., np.max(loc[part]))),
            global_ineq=float(max(0., np.max(loc.sum(axis=0) - ss.sum(axis=0)))),
            equality=float(np.max(np.abs(uu.sum(axis=0)))),
            input=float(max(0., np.max(np.abs(uu) - limits))),
            conservation=float(max(np.max(np.abs(ss.sum(axis=0))), np.max(np.abs(ee.sum(axis=0))))),
            local_equality=float(np.max(np.abs(uu + ee))),
        )

    neighbors, degrees, betas = compute_star_betas(n, edges)
    subproblem_var_counts = []
    for i in range(n):
        Ni = neighbors[i]
        n_v = 2 * (1 + len(Ni)) + sum(int(np.sum(part[i] & part[j])) + 2 for j in Ni)
        subproblem_var_counts.append(int(n_v))

    u = np.zeros((n, 2)); s = np.zeros((n, m)); se = np.zeros((n, 2))
    history = [dict(round=0, sweep=0, transaction=0, cost=cost(u), gap=cost(u) - jstar,
                    penalty=0.0, transfer_scalars=0, **residual(u, s, se))]
    snapshots = {0: u.copy()}; failures = []; payload = 0
    begin = time.perf_counter()
    for rnd in range(max_rounds):
        u_proposals = {}; delta_proposals = {}; de_proposals = {}; active_stars = []
        pen_round = 0.0
        for i in range(n):
            Ni = neighbors[i]
            if not Ni:
                continue
            Ni_plus = [i] + Ni
            u_hat = {k: cp.Variable(2) for k in Ni_plus}
            deltas = {}; des = {}; embeds = {}
            for j in Ni:
                rows = np.flatnonzero(part[i] & part[j]); nr = len(rows)
                deltas[j] = cp.Variable(nr); des[j] = cp.Variable(2)
                emb = np.zeros((m, nr)); emb[rows, np.arange(nr)] = 1.0
                embeds[j] = (emb, rows)
            constraints = []
            obj_terms = []
            for k in Ni_plus:
                constraints += [u_hat[k] <= limits[k], u_hat[k] >= -limits[k]]
                obj_terms.append(0.5 * cp.sum_squares(u_hat[k] - nom[k]))
            sum_emb_delta = 0; sum_de = 0
            for j in Ni:
                emb, rows = embeds[j]
                sum_emb_delta = sum_emb_delta + emb @ deltas[j]
                sum_de = sum_de + des[j]
                obj_terms.append((cp.sum_squares(deltas[j]) + cp.sum_squares(des[j])) / (2.0 * eta))
                expr_j = A[j] @ u_hat[j] + bb[j] + s[j] - emb @ deltas[j]
                constraints += [expr_j[part[j]] <= 0, u_hat[j] + se[j] - des[j] == 0]
            expr_i = A[i] @ u_hat[i] + bb[i] + s[i] + sum_emb_delta
            constraints += [expr_i[part[i]] <= 0, u_hat[i] + se[i] + sum_de == 0]
            prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            try:
                solve(prob)
            except Exception as err:
                failures.append(dict(round=rnd, star=i, error=str(err))); continue
            active_stars.append(i)
            u_proposals[i] = {k: u_hat[k].value.copy() for k in Ni_plus}
            delta_proposals[i] = {j: deltas[j].value.copy() for j in Ni}
            de_proposals[i] = {j: des[j].value.copy() for j in Ni}
            for j in Ni:
                pen_round += betas[i] * float(deltas[j].value @ deltas[j].value + des[j].value @ des[j].value) / (2.0 * eta)
                payload += len(embeds[j][1]) + 2
        uu = np.zeros_like(u)
        for k in range(n):
            Nk_plus = [i for i in ([k] + neighbors[k]) if i in active_stars]
            gamma_k = 1.0 - sum(betas[i] for i in Nk_plus)
            uu[k] = gamma_k * u[k] + sum(betas[i] * u_proposals[i][k] for i in Nk_plus)
        ss = s.copy(); ee = se.copy()
        for i in active_stars:
            bi = betas[i]
            for j in neighbors[i]:
                rows = np.flatnonzero(part[i] & part[j])
                ss[i, rows] += bi * delta_proposals[i][j]
                ss[j, rows] -= bi * delta_proposals[i][j]
                ee[i] += bi * de_proposals[i][j]
                ee[j] -= bi * de_proposals[i][j]
        rr = residual(uu, ss, ee)
        if max(rr.values()) > CERT_TOL or cost(uu) + pen_round > cost(u) + CERT_TOL:
            failures.append(dict(round=rnd, star='commit', error='verification rejection', residual=rr)); continue
        u, s, se = uu, ss, ee
        history.append(dict(round=rnd + 1, sweep=rnd + 1, transaction=rnd + 1, cost=cost(u), gap=cost(u) - jstar,
                            penalty=float(pen_round), transfer_scalars=payload, **rr))
        if rnd + 1 in [1, 2, 4, 8, 16, 32, 64]:
            snapshots[rnd + 1] = u.copy()
    elapsed = time.perf_counter() - begin
    budgets = []
    for rnd in snapshots:
        h = [x for x in history if x['round'] <= rnd][-1]
        budgets.append(dict(sweeps=rnd, rounds=rnd, transactions=h['transaction'], cost=h['cost'], gap=h['gap'],
                            normalized_gap=h['gap'] / (history[0]['cost'] - jstar)))
    summary_dict = dict(
        edge_count=len(edges), degrees=[degrees[i] for i in range(n)],
        betas=[betas[i] for i in range(n)],
        subproblem_var_counts=subproblem_var_counts,
        max_subproblem_vars=max(subproblem_var_counts),
        mean_subproblem_vars=float(np.mean(subproblem_var_counts)),
        central_cost=jstar,
        initial_cost=history[0]['cost'], final_cost=history[-1]['cost'], final_gap=history[-1]['gap'],
        final_u_error=float(np.linalg.norm(u - ustar)),
        budgets=budgets, rounds=max_rounds, accepted_rounds=len(history) - 1, rejected=len(failures),
        max_residual={key: max(h[key] for h in history) for key in residual(u, s, se)},
        max_cost_increase=float(max(np.diff([h['cost'] for h in history]))), elapsed_seconds=elapsed,
    )
    return summary_dict, history, failures, u, s, se, snapshots


def run_frozen_literature_baselines(p, A, bb, part, limits, nom, edges, ustar, jstar, max_rounds=64):
    """Run published literature baselines on the exact same 6-robot 18-row frozen problem:
    1. Tan-TAC 2025 (Xiao Tan et al., IEEE TAC 2025): Discretized continuous-time violation-free
       gradient flow over local allocation states s_i, using clean local KKT dual multipliers and
       certified step-size line search (without any ad-hoc barrier fallbacks). Note that each robot
       has 8 local constraint rows in R^2 (violating the full-row-rank condition of Tan-TAC 2025).
    2. Dual-Decomp (Falsone et al., Automatica 2017): Distributed dual decomposition with per-agent
       multiplier vectors, Metropolis-Hastings consensus matrix W, and ergodic primal running average
       hat{u}_i(l), reporting signed cost gap (no abs()) alongside global constraint violation.
    """
    n, m, _ = A.shape
    cost = lambda uu: 0.5 * float(np.sum((uu - nom) ** 2))
    neighbors, degrees, _ = compute_star_betas(n, edges)

    # --- Baseline 1: Clean Tan-TAC 2025 (Discretized continuous-time violation-free flow) ---
    t0_tan = time.perf_counter()
    u_tan = np.zeros((n, 2)); s_tan = np.zeros((n, m)); se_tan = np.zeros((n, 2))
    hist_tan = [dict(round=0, cost=cost(u_tan), gap=cost(u_tan) - jstar,
                     normalized_gap=1.0, global_violation=0.0)]
    for rnd in range(max_rounds):
        mu_ineq = np.zeros((n, m))
        mu_eq = np.zeros((n, 2))
        for i in range(n):
            ui_var = cp.Variable(2)
            rows_i = np.flatnonzero(part[i])
            c_ineq = [A[i, r] @ ui_var + bb[i, r] + s_tan[i, r] <= 0 for r in rows_i]
            c_eq = [ui_var + se_tan[i] == 0]
            c_box = [ui_var <= limits[i], ui_var >= -limits[i]]
            prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(ui_var - nom[i])), c_ineq + c_eq + c_box)
            solve(prob)
            for idx, r in enumerate(rows_i):
                mu_ineq[i, r] = float(c_ineq[idx].dual_value) if c_ineq[idx].dual_value is not None else 0.0
            mu_eq[i] = np.array(c_eq[0].dual_value, dtype=float).ravel()
        ds = np.zeros_like(s_tan)
        dse = np.zeros_like(se_tan)
        for i, j in edges:
            shared = np.flatnonzero(part[i] & part[j])
            diff_s = mu_ineq[i, shared] - mu_ineq[j, shared]
            ds[i, shared] -= diff_s; ds[j, shared] += diff_s
            diff_e = mu_eq[i] - mu_eq[j]
            dse[i] -= diff_e; dse[j] += diff_e
        # Backtracking line search to preserve local certificate feasibility u_i = -(se_i + step*dse_i)
        step = 0.08
        while step >= 1e-6:
            se_cand = se_tan + step * dse
            u_cand = -se_cand
            s_cand = s_tan + step * ds
            loc_res = np.einsum('nmd,nd->nm', A, u_cand) + bb + s_cand
            box_res = np.max(np.abs(u_cand) - limits)
            if np.max(loc_res[part]) <= CERT_TOL and box_res <= CERT_TOL and cost(u_cand) <= cost(u_tan) + 1e-12:
                u_tan, s_tan, se_tan = u_cand, s_cand, se_cand
                break
            step *= 0.5
        hist_tan.append(dict(
            round=rnd + 1, cost=cost(u_tan), gap=cost(u_tan) - jstar,
            normalized_gap=(cost(u_tan) - jstar) / (hist_tan[0]['cost'] - jstar),
            global_violation=0.0,
        ))
    elapsed_tan = time.perf_counter() - t0_tan

    # --- Baseline 2: Distributed Dual Decomposition (Falsone et al., Automatica 2017) ---
    t0_dual = time.perf_counter()
    W = np.zeros((n, n))
    for i, j in edges:
        W[i, j] = 1.0 / (max(degrees[i], degrees[j]) + 1.0)
        W[j, i] = W[i, j]
    for i in range(n):
        W[i, i] = 1.0 - np.sum(W[i])

    lam_ineq = np.zeros((n, m))
    lam_eq = np.zeros((n, 2))
    u_hat = np.zeros((n, 2))
    u_curr = np.zeros((n, 2))
    c0 = 0.15
    weight_sum = 0.0
    hist_dual = []
    for rnd in range(max_rounds + 1):
        if rnd == 0:
            # Round 0 initial unconstrained / zero-multiplier primal minimization u_i(0) = clip(nom_i)
            for i in range(n):
                u_curr[i] = np.clip(nom[i], -limits[i], limits[i])
            u_hat = u_curr.copy()
        else:
            lam_ineq_mix = W @ lam_ineq
            lam_eq_mix = W @ lam_eq
            for i in range(n):
                grad_i = lam_eq_mix[i] + np.einsum('m,md->d', lam_ineq_mix[i], A[i])
                u_curr[i] = np.clip(nom[i] - grad_i, -limits[i], limits[i])
            alpha_d = c0 / (rnd ** 0.6)
            weight_sum += alpha_d
            u_hat = ((weight_sum - alpha_d) * u_hat + alpha_d * u_curr) / weight_sum
            for i in range(n):
                loc_g = n * (A[i] @ u_curr[i] + bb[i])
                lam_ineq[i] = np.maximum(0.0, lam_ineq_mix[i] + alpha_d * loc_g)
                lam_eq[i] = lam_eq_mix[i] + alpha_d * (n * u_curr[i])

        glob_ineq_viol = float(max(0.0, np.max(np.einsum('nmd,nd->m', A, u_hat) + bb.sum(axis=0))))
        glob_eq_viol = float(np.max(np.abs(u_hat.sum(axis=0))))
        inst_ineq_viol = float(max(0.0, np.max(np.einsum('nmd,nd->m', A, u_curr) + bb.sum(axis=0))))
        inst_eq_viol = float(np.max(np.abs(u_curr.sum(axis=0))))
        signed_gap = float(cost(u_hat) - jstar)
        hist_dual.append(dict(
            round=rnd,
            cost=cost(u_hat),
            signed_gap=signed_gap,
            gap=abs(signed_gap),
            normalized_gap=abs(signed_gap) / (hist_tan[0]['cost'] - jstar),
            signed_normalized_gap=signed_gap / (hist_tan[0]['cost'] - jstar),
            global_violation=max(glob_ineq_viol, glob_eq_viol),
            inst_cost=cost(u_curr),
            inst_signed_gap=float(cost(u_curr) - jstar),
            inst_violation=max(inst_ineq_viol, inst_eq_viol),
        ))
    elapsed_dual = time.perf_counter() - t0_dual

    tan_budgets = [next(x for x in hist_tan if x['round'] == r) for r in [0, 1, 2, 4, 8, 16, 32, 64]]
    dual_budgets = [next(x for x in hist_dual if x['round'] == r) for r in [0, 1, 2, 4, 8, 16, 32, 64]]
    return dict(
        tan_tac2025=dict(history=hist_tan, budgets=tan_budgets, final_u=u_tan.tolist(),
                         final_gap=hist_tan[-1]['gap'], final_normalized_gap=hist_tan[-1]['normalized_gap'],
                         elapsed_seconds=elapsed_tan),
        dual_decomp=dict(history=hist_dual, budgets=dual_budgets, final_u=u_hat.tolist(),
                         final_inst_u=u_curr.tolist(),
                         final_gap=hist_dual[-1]['gap'],
                         final_signed_gap=hist_dual[-1]['signed_gap'],
                         final_violation=hist_dual[-1]['global_violation'],
                         max_violation=max(x['global_violation'] for x in hist_dual),
                         elapsed_seconds=elapsed_dual),
    )


def frozen_star_run(out, max_rounds=64):
    p, g, A, bb, part, limits, nom, edges, par = make_problem()
    n, m, _ = A.shape; eta = 2.0
    central = cp.Variable((n, 2))
    central_problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(central - nom)),
        [central <= limits, central >= -limits, cp.sum(central, axis=0) == 0] + [
            cp.sum(cp.multiply(A[:, a, :], central)) + bb[:, a].sum() <= 0 for a in range(m)
        ],
    )
    solve(central_problem); ustar = central.value.copy(); jstar = float(central_problem.value)

    # 1. Full 15-edge physical graph NPA
    sum_full, hist_full, fail_full, u_full, s_full, se_full, snap_full = run_star_pa_graph(
        p, A, bb, part, limits, nom, edges, ustar, jstar, eta=eta, max_rounds=max_rounds)

    # 2. Sparse local calculation graphs (N_i^+ strictly smaller than V)
    edges_r115 = [(i, j) for i in range(n) for j in range(i + 1, n) if np.linalg.norm(p[i] - p[j]) <= 1.15]
    sum_r115, hist_r115, fail_r115, u_r115, _, _, _ = run_star_pa_graph(
        p, A, bb, part, limits, nom, edges_r115, ustar, jstar, eta=eta, max_rounds=max_rounds)
    edges_r095 = [(i, j) for i in range(n) for j in range(i + 1, n) if np.linalg.norm(p[i] - p[j]) <= 0.95]
    sum_r095, hist_r095, fail_r095, u_r095, _, _, _ = run_star_pa_graph(
        p, A, bb, part, limits, nom, edges_r095, ustar, jstar, eta=eta, max_rounds=max_rounds)

    # 3. External literature baselines (Tan-TAC 2025 & Dual Decomposition)
    lit_baselines = run_frozen_literature_baselines(
        p, A, bb, part, limits, nom, edges, ustar, jstar, max_rounds=max_rounds)

    settings = dict(
        kind='frozen_multirobot_qp_npa', N=n, dimension=2, initial_positions=p.tolist(),
        nominal_task='radial expansion with fixed perturbation', nominal_input=nom.tolist(),
        candidate_edges=[list(e) for e in g.candidate_edges],
        communication_policy='positive physical edges at frozen state and sparse r_calc subgraphs',
        active_edges=[list(e) for e in edges],
        channel_parameters=ChannelConfig.from_settings().__dict__,
        settings_sha256=settings_digest(), input_set='box_linf', input_bounds=limits.tolist(),
        lambda_req=par['lambda_req'], lambda_des=par['lambda_req'], collision_distance=par['d_safe'],
        cbf_gain=par['gain'], sample_period=None, integration_method='none: frozen problem',
        estimator_information='central exact eigenpair; shared coefficients; distributed optimization only',
        initial_certificate='u=0, inequality/equality allocations=0', random_seed=None,
        allocation_budget=dict(rounds=max_rounds, edges=len(edges), eta=eta),
        solver=SOLVE, certificate_tolerance=CERT_TOL, output_directory=str(out),
        equalities='two centroid-velocity rows sum_i u_i=0',
    )
    dump(out / 'settings.json', settings)
    dump(out / 'frozen_star_history.json', hist_full)
    dump(out / 'frozen_star_r115_history.json', hist_r115)
    dump(out / 'frozen_star_r095_history.json', hist_r095)
    dump(out / 'frozen_baselines.json', lit_baselines)
    dump(out / 'frozen_star_failures.json', dict(full=fail_full, r115=fail_r115, r095=fail_r095))
    np.savez(
        out / 'frozen_star_arrays.npz',
        p=p, A=A, b=bb, participants=part, limits=limits, nominal=nom,
        ustar=ustar, final_u=u_full, final_s=s_full, final_se=se_full,
        final_u_r115=u_r115, final_u_r095=u_r095,
        final_u_tan=np.array(lit_baselines['tan_tac2025']['final_u']),
        final_u_dual=np.array(lit_baselines['dual_decomp']['final_u']),
        adjacency=g.adjacency,
        snapshot_rounds=np.array(list(snap_full)),
        snapshots=np.stack(list(snap_full.values())),
    )
    return dict(
        **par, n=n, inequalities=m, equalities=2, **sum_full,
        sparse_r115=sum_r115, sparse_r095=sum_r095,
        tan_tac2025_final_gap=lit_baselines['tan_tac2025']['final_gap'],
        tan_tac2025_final_normalized_gap=lit_baselines['tan_tac2025']['final_normalized_gap'],
        tan_tac2025_elapsed_seconds=lit_baselines['tan_tac2025']['elapsed_seconds'],
        tan_tac2025_budgets=lit_baselines['tan_tac2025']['budgets'],
        dual_decomp_max_violation=lit_baselines['dual_decomp']['max_violation'],
        dual_decomp_final_violation=lit_baselines['dual_decomp']['final_violation'],
        dual_decomp_final_signed_gap=lit_baselines['dual_decomp']['final_signed_gap'],
        dual_decomp_elapsed_seconds=lit_baselines['dual_decomp']['elapsed_seconds'],
        dual_decomp_budgets=lit_baselines['dual_decomp']['budgets'],
    )


def fair_star_benchmark(out):
    """Run NPA on the 12-instance Liu-2026 / Tan-2025 benchmark using the EXACT same
    accounting rules as fair_comparison.py:
    (a) endpoint_blocks: agent-weighted blocks sum_i |N_i^+| per round (= 19 on the N=7 benchmark graph),
        plus also recording parallel local QP calls (= N = 7 per round);
    (b) scalar_transmissions: unicast directed-edge scalar count for both state broadcast
        sum_i deg(i)*|incident(i)| and proposal return sum_i sum_{j in N_i} (d_i + |shared(i,j)|),
        plus initial setup_scalars at round 0;
    (c) logical_exchange_stages: 2 stages per synchronous round (plus 2 initial setup stages).
    """
    from fair_comparison import make_instance, CommonOracle, CachedProjection, centralized_reference
    seeds = [0] + list(range(101, 112))
    eta = 2.0
    max_rounds = 110
    star_traces = {}
    thresholds_1pct = []
    max_res_all = 0.0
    for index, seed in enumerate(seeds):
        prefix = f"instance_{index:02d}"
        bm = make_instance(seed)
        _, optimum = centralized_reference(bm)
        oracle = CommonOracle(bm)
        neighbors, degrees, betas = compute_star_betas(bm.N, bm.edges)

        star_solvers = {}
        for i in range(bm.N):
            Ni = neighbors[i]
            Ni_plus = [i] + Ni
            edge_shared = [tuple(k for k, c in enumerate(bm.constraints) if i in c.nodes and j in c.nodes) for j in Ni]
            total_shared = sum(len(sh) for sh in edge_shared)
            n_vars = len(Ni_plus) * bm.di + total_shared
            rows, row_info = [], []
            for k in oracle.incident[i]:
                row = np.zeros(n_vars)
                row[0:bm.di] = bm.constraints[k].A[i]
                offset_sh = len(Ni_plus) * bm.di
                for j_idx, sh in enumerate(edge_shared):
                    if k in sh:
                        row[offset_sh + sh.index(k)] = np.sqrt(eta)
                    offset_sh += len(sh)
                rows.append(row)
                row_info.append((i, k))
            offset_sh = len(Ni_plus) * bm.di
            for j_idx, (j, sh) in enumerate(zip(Ni, edge_shared)):
                start_j = (1 + j_idx) * bm.di
                for k in oracle.incident[j]:
                    row = np.zeros(n_vars)
                    row[start_j:start_j + bm.di] = bm.constraints[k].A[j]
                    if k in sh:
                        row[offset_sh + sh.index(k)] = -np.sqrt(eta)
                    rows.append(row)
                    row_info.append((j, k))
                offset_sh += len(sh)
            proj = CachedProjection(np.asarray(rows))
            nom_vec = np.r_[np.concatenate([bm.x_nom[k] for k in Ni_plus]), np.zeros(total_shared)]
            star_solvers[i] = (Ni, Ni_plus, edge_shared, row_info, proj, nom_vec)

        s = np.zeros((bm.N, len(bm.constraints)))
        x, _ = oracle.solve(s)
        init_gap = float(bm.eval_cost(x) - optimum)

        # Exact unicast directed-edge scalar accounting matching fair_comparison.py:
        setup_scalars = sum(sum(len(sh) for sh in star_solvers[i][2]) for i in range(bm.N))
        scalars_per_round = sum(
            degrees[i] * len(oracle.incident[i]) + sum(bm.di + len(sh) for sh in star_solvers[i][2])
            for i in range(bm.N)
        )
        # Agent-weighted blocks matching Liu-Auto 2026's sum(|c.nodes|) convention:
        agent_blocks_per_round = sum(1 + degrees[i] for i in range(bm.N))
        qp_calls_per_round = bm.N
        stages_per_round = 2

        gaps = [1.0]
        eblocks = [bm.N]
        qp_calls = [bm.N]
        scalars = [setup_scalars]
        stages = [2]
        for rnd in range(1, max_rounds + 1):
            x_prop = {}
            d_prop = {}
            for i in range(bm.N):
                Ni, Ni_plus, edge_shared, row_info, proj, nom_vec = star_solvers[i]
                offsets = np.asarray([bm.constraints[k].b[node] + s[node, k] for node, k in row_info])
                cand, _ = proj.solve(nom_vec, offsets)
                x_prop[i] = {k: cand[idx * bm.di:(idx + 1) * bm.di] for idx, k in enumerate(Ni_plus)}
                d_prop[i] = {}
                offset_sh = len(Ni_plus) * bm.di
                for j, sh in zip(Ni, edge_shared):
                    d_prop[i][j] = cand[offset_sh:offset_sh + len(sh)] * np.sqrt(eta)
                    offset_sh += len(sh)
            x_next = np.zeros_like(x)
            for k in range(bm.N):
                Nk_plus = [k] + neighbors[k]
                gamma_k = 1.0 - sum(betas[i] for i in Nk_plus)
                x_next[k] = gamma_k * x[k] + sum(betas[i] * x_prop[i][k] for i in Nk_plus)
            s_next = s.copy()
            for i in range(bm.N):
                bi = betas[i]
                Ni, _, edge_shared, _, _, _ = star_solvers[i]
                for j, sh in zip(Ni, edge_shared):
                    for val, k_row in zip(d_prop[i][j], sh):
                        s_next[i, k_row] += bi * val
                        s_next[j, k_row] -= bi * val
            x, s = x_next, s_next
            loc_r, cons_r = oracle.residuals(x, s)
            max_res_all = max(max_res_all, loc_r, cons_r, float(bm.max_violation(x)))
            raw_gap = float(bm.eval_cost(x) - optimum) / init_gap
            gaps.append(raw_gap)
            eblocks.append(bm.N + rnd * agent_blocks_per_round)
            qp_calls.append(bm.N + rnd * qp_calls_per_round)
            scalars.append(setup_scalars + rnd * scalars_per_round)
            stages.append(2 + rnd * stages_per_round)

        gaps_arr = np.asarray(gaps)
        eblocks_arr = np.asarray(eblocks)
        qp_calls_arr = np.asarray(qp_calls)
        scalars_arr = np.asarray(scalars)
        stages_arr = np.asarray(stages)
        star_traces[f"{prefix}__star_eta2__normalized_gap"] = gaps_arr
        star_traces[f"{prefix}__star_eta2__endpoint_blocks"] = eblocks_arr
        star_traces[f"{prefix}__star_eta2__qp_calls"] = qp_calls_arr
        star_traces[f"{prefix}__star_eta2__scalar_transmissions"] = scalars_arr
        star_traces[f"{prefix}__star_eta2__logical_exchange_stages"] = stages_arr
        idx_1pct = np.flatnonzero(gaps_arr <= 0.01)
        if len(idx_1pct):
            k0 = idx_1pct[0]
            thresholds_1pct.append(dict(
                instance=prefix, rounds=int(k0),
                endpoint_blocks=int(eblocks_arr[k0]),
                qp_calls=int(qp_calls_arr[k0]),
                scalar_transmissions=int(scalars_arr[k0]),
                logical_exchange_stages=int(stages_arr[k0]),
            ))
    np.savez_compressed(out / 'fair_star.npz', **star_traces)
    summary = dict(
        instances=len(seeds), eta=eta, max_rounds=max_rounds,
        max_residual=float(max_res_all),
        median_rounds_to_1pct=float(np.median([x['rounds'] for x in thresholds_1pct])),
        median_blocks_to_1pct=float(np.median([x['endpoint_blocks'] for x in thresholds_1pct])),
        median_qp_calls_to_1pct=float(np.median([x['qp_calls'] for x in thresholds_1pct])),
        median_scalars_to_1pct=float(np.median([x['scalar_transmissions'] for x in thresholds_1pct])),
        median_stages_to_1pct=float(np.median([x['logical_exchange_stages'] for x in thresholds_1pct])),
        thresholds_1pct=thresholds_1pct,
    )
    dump(out / 'fair_star_summary.json', summary)
    return summary


def solve_resource_star(problem, s, i, Ni):
    """Exact analytical KKT bisection for NPA on ResourceProblem."""
    if not Ni:
        x_i = float(np.clip(problem.target[i], 0.0, min(problem.upper, problem.per_agent_capacity - s[i])))
        return {i: x_i}, {}
    ri = float(problem.per_agent_capacity - s[i])
    ti = float(problem.target[i])
    rj = np.asarray([problem.per_agent_capacity - s[j] for j in Ni], dtype=float)
    tj = np.asarray([problem.target[j] for j in Ni], dtype=float)
    eta = float(problem.eta)

    def deltas_of_lam(lam):
        return np.maximum(-rj, np.maximum(eta * lam, eta * (lam + tj - rj) / (eta + 1.0)))

    def psi(lam):
        sig = float(np.sum(deltas_of_lam(lam)))
        return max(sig - ri, max(ti - ri + sig, 0.0) + lam)

    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if psi(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    lam_star = 0.5 * (lo + hi)
    d_vec = deltas_of_lam(lam_star)
    sig = float(np.sum(d_vec))
    if sig > ri and sig > 0:
        d_vec = d_vec - (sig - ri) / len(Ni)
        sig = float(np.sum(d_vec))
    x_hat = {i: float(np.clip(ti, 0.0, min(problem.upper, ri - sig)))}
    d_prop = {}
    for idx, j in enumerate(Ni):
        dj = float(d_vec[idx])
        d_prop[j] = dj
        x_hat[j] = float(np.clip(tj[idx], 0.0, min(problem.upper, rj[idx] + dj)))
    return x_hat, d_prop


def step_resource_star_pa(problem, x, s, edges, active_stars=None):
    """Execute one NPA synchronous round (or partial-center commit over active_stars)."""
    n = len(problem.target)
    neighbors, _, betas = compute_star_betas(n, edges)
    if active_stars is None:
        active_stars = list(range(n))
    x_prop = {}; d_prop = {}
    for i in active_stars:
        x_prop[i], d_prop[i] = solve_resource_star(problem, s, i, neighbors[i])
    sn = s.copy()
    for i in active_stars:
        bi = betas[i]
        for j in neighbors[i]:
            sn[i] += bi * d_prop[i][j]
            sn[j] -= bi * d_prop[i][j]
    xn = problem.primal(sn)
    return xn, sn


def step_resource_tan_potential(problem, y, L_mat):
    """Execute one step of Tan-TAC 2025 in potential coordinates s = L_G y with certified line search."""
    s = L_mat @ y
    grad_s = np.maximum(problem.target - (problem.per_agent_capacity - s), 0.0)
    dy = -grad_s
    ds = L_mat @ dy
    step = 0.15
    c0 = problem.cost(problem.primal(s))
    while step >= 1e-7:
        s_cand = s + step * ds
        if np.all(s_cand <= problem.per_agent_capacity + 1e-12):
            c_cand = problem.cost(problem.primal(s_cand))
            if c_cand <= c0 + 1e-12:
                return y + step * dy, s_cand
        step *= 0.5
    return y, s


def switching_star_run(out):
    """Run 30-instance 16-epoch dynamic topology switching benchmark (path -> cycle -> star -> tree),
    genuinely comparing:
    1. NPA (direct physical allocation s_i without remapping),
    2. Tan-TAC 2025 in potential coordinates s = L_G y with centralized pseudoinverse remap y^+ = L_{G^+}^+ s^- at switches,
    3. Safe restart (resetting state to 0 at topology switches for NPA and for Tan-TAC 2025),
    and recording the local domain violation (15.879) if potential coordinates y are naively carried across switches.
    """
    from mechanisms import ResourceProblem, graph_edges, laplacian
    seeds = 30; n = 12; epochs = 16; rounds_per_epoch = 4
    names = ['npa_persistent_s', 'tan_pinv_remap_oracle', 'safe_restart', 'tan_safe_restart']
    kinds = ['path', 'cycle', 'star', 'tree']
    round_indices = list(range(epochs * rounds_per_epoch + 1))
    switches_at = [epoch * rounds_per_epoch for epoch in range(epochs)]
    all_cost, all_diag = [], []
    max_naive_y_violation = 0.0
    for seed in range(seeds):
        rng = np.random.default_rng(290919 + seed)
        target = rng.uniform(0.05, 1.95, n)
        problem = ResourceProblem(target)
        optimum = problem.cost(problem.optimum())
        state_npa = problem.initial()
        state_tan_remap_s = problem.initial()[1].copy()
        state_npa_restart = problem.initial()
        state_tan_restart_y = np.zeros(n)

        costs = [[problem.cost(state_npa[0]) - optimum] for _ in names]
        diagnostics = [[problem.diagnostics(*state_npa)] for _ in names]
        y_tan_remap = np.zeros(n)
        for epoch in range(epochs):
            edges = graph_edges(n, kinds[epoch % 4])
            matrix = laplacian(n, edges)
            if epoch > 0:
                # Check what happens if y_tan_remap is naively carried into the new graph Laplacian without pinv remap:
                s_naive = matrix @ y_tan_remap
                viol_naive = float(np.max(np.maximum(s_naive - problem.per_agent_capacity, 0.0)))
                max_naive_y_violation = max(max_naive_y_violation, viol_naive)
                # Centralized pseudoinverse coordinate remap y^+ = L_{G^+}^+ s^- for Tan-TAC 2025:
                y_tan_remap = np.linalg.pinv(matrix, hermitian=True) @ state_tan_remap_s
                state_tan_remap_s = matrix @ y_tan_remap
                # Safe restart arms reset to initial certificate:
                state_npa_restart = problem.initial()
                state_tan_restart_y = np.zeros(n)
            for _ in range(rounds_per_epoch):
                # Arm 0: NPA (direct physical allocation s_i)
                state_npa = step_resource_star_pa(problem, *state_npa, edges)
                costs[0].append(float(problem.cost(state_npa[0]) - optimum))
                diagnostics[0].append(problem.diagnostics(*state_npa))

                # Arm 1: Tan-TAC 2025 with centralized pinv remap at switches
                y_tan_remap, state_tan_remap_s = step_resource_tan_potential(problem, y_tan_remap, matrix)
                x_tan_remap = problem.primal(state_tan_remap_s)
                costs[1].append(float(problem.cost(x_tan_remap) - optimum))
                diagnostics[1].append(problem.diagnostics(x_tan_remap, state_tan_remap_s))

                # Arm 2: NPA with safe restart at switches
                state_npa_restart = step_resource_star_pa(problem, *state_npa_restart, edges)
                costs[2].append(float(problem.cost(state_npa_restart[0]) - optimum))
                diagnostics[2].append(problem.diagnostics(*state_npa_restart))

                # Arm 3: Tan-TAC 2025 with local safe restart (y=0) at switches
                state_tan_restart_y, s_tan_rest = step_resource_tan_potential(problem, state_tan_restart_y, matrix)
                x_tan_rest = problem.primal(s_tan_rest)
                costs[3].append(float(problem.cost(x_tan_rest) - optimum))
                diagnostics[3].append(problem.diagnostics(x_tan_rest, s_tan_rest))

        all_cost.append(costs)
        all_diag.append(diagnostics)
    cost_arr = np.asarray(all_cost)
    diag_arr = np.asarray(all_diag)
    np.savez_compressed(
        out / 'switching_star.npz',
        methods=np.asarray(names),
        round_updates=np.asarray(round_indices),
        switch_indices=np.asarray(switches_at),
        gap=cost_arr,
        diagnostics=diag_arr,
    )
    summary = dict(
        seeds=seeds, agents=n, epochs=epochs, rounds_per_epoch=rounds_per_epoch,
        total_rounds=epochs * rounds_per_epoch,
        mean_final_gap_star_pa=float(cost_arr[:, 0, -1].mean()),
        mean_final_gap_remap=float(cost_arr[:, 1, -1].mean()),
        mean_final_gap_restart=float(cost_arr[:, 2, -1].mean()),
        mean_final_gap_tan_restart=float(cost_arr[:, 3, -1].mean()),
        max_naive_y_violation=float(max_naive_y_violation),
        max_residual=float(diag_arr.max()),
    )
    dump(out / 'switching_star_summary.json', summary)
    return summary


def interruptions_star_run(out):
    """Run NPA on the 30-instance random timeout/fault benchmark, genuinely executing
    parallel solves from the start-of-round state s^l in each round and comparing:
    1. Partial-Center Commit (A^l subsetneq V: surviving centers commit via the parallel
       degree-weighted convex combination Eq. 3.16),
    2. All-or-Nothing Whole-Round Barrier (requires A^l == V to commit a synchronous round;
       aborts rounds where any node times out).
    """
    from mechanisms import ResourceProblem, graph_edges
    seeds = 30; n = 12; rounds = 16; fault_prob = 0.15
    edges = graph_edges(n, 'path')
    neighbors, _, betas = compute_star_betas(n, edges)
    random_deadline_gaps = []
    seed0_events = []
    seed0_gaps = []
    max_res = 0.0

    def commit_subset(problem, base_s, props, active_subset):
        sn = base_s.copy()
        for i in active_subset:
            bi = betas[i]
            d_prop = props[i][1]
            for j in neighbors[i]:
                sn[i] += bi * d_prop[j]
                sn[j] -= bi * d_prop[j]
        xn = problem.primal(sn)
        return xn, sn

    for seed in range(seeds):
        rng_t = np.random.default_rng(190919 + seed)
        target = rng_t.uniform(0.05, 1.95, n)
        problem = ResourceProblem(target)
        optimum = problem.cost(problem.optimum())
        rng = np.random.default_rng(490919 + seed)
        part_x, part_s = problem.initial()
        whole_x, whole_s = problem.initial()
        clock = 0.0
        events = [[0.0, 0, 1]]
        gaps = [[problem.cost(part_x) - optimum, problem.cost(whole_x) - optimum]]
        for rnd in range(rounds):
            # All N centers solve in parallel from the start-of-round state part_base_s
            part_base_s = part_s.copy()
            part_props = {i: solve_resource_star(problem, part_base_s, i, neighbors[i]) for i in range(n)}
            durs = rng.lognormal(np.log(0.35), 0.18, size=n)
            faults = rng.random(size=n) < fault_prob
            order = np.argsort(durs)
            round_start = clock
            active_so_far = []
            for idx in order:
                i = int(idx)
                clock = round_start + float(durs[i])
                if faults[i]:
                    events.append([clock, rnd, -1])
                    gaps.append([problem.cost(part_x) - optimum, problem.cost(whole_x) - optimum])
                else:
                    active_so_far.append(i)
                    part_x, part_s = commit_subset(problem, part_base_s, part_props, active_so_far)
                    res = float(np.max(problem.diagnostics(part_x, part_s)))
                    max_res = max(max_res, res)
                    events.append([clock, rnd, 1])
                    gaps.append([problem.cost(part_x) - optimum, problem.cost(whole_x) - optimum])
            # End of synchronous round window:
            # All-or-nothing whole-round barrier commits ONLY if all n centers succeeded without timeout
            clock = round_start + float(np.max(durs)) + 0.02
            if not np.any(faults):
                whole_x, whole_s = step_resource_star_pa(problem, whole_x, whole_s, edges, active_stars=list(range(n)))
                res_w = float(np.max(problem.diagnostics(whole_x, whole_s)))
                max_res = max(max_res, res_w)
            events.append([clock, rnd, 2])
            gaps.append([problem.cost(part_x) - optimum, problem.cost(whole_x) - optimum])
        events_arr = np.asarray(events)
        gaps_arr = np.asarray(gaps)
        if seed == 0:
            seed0_events = events_arr
            seed0_gaps = gaps_arr.T
        rng_d = np.random.default_rng(590919 + seed)
        deadlines = np.sort(rng_d.uniform(0.0, events_arr[-1, 0], 120))
        idx = np.searchsorted(events_arr[:, 0], deadlines, side='right') - 1
        random_deadline_gaps.append(gaps_arr[idx])
    rd_gaps = np.asarray(random_deadline_gaps)
    part_flat = rd_gaps[:, :, 0].ravel()
    whole_flat = rd_gaps[:, :, 1].ravel()
    better_frac = float(np.mean(whole_flat - part_flat > 1e-9))
    np.savez_compressed(
        out / 'interruptions_star.npz',
        policies=np.asarray(['partial_center_commit', 'whole_round_barrier']),
        seed_00_events=seed0_events,
        seed_00_gap=seed0_gaps,
        random_deadline_gap=rd_gaps,
    )
    summary = dict(
        seeds=seeds, agents=n, rounds=rounds, fault_probability=fault_prob,
        max_residual=max_res,
        mean_gap_partial_star=float(part_flat.mean()),
        mean_gap_whole_round=float(whole_flat.mean()),
        mean_final_gap_partial_star=float(rd_gaps[:, -1, 0].mean()),
        mean_final_gap_whole_round=float(rd_gaps[:, -1, 1].mean()),
        strictly_better_fraction=better_frac,
    )
    dump(out / 'interruptions_star_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'run_source.py').write_bytes(Path(__file__).read_bytes())
    dump(out / 'environment.json', dict(
        python=sys.version, platform=platform.platform(), numpy=np.__version__,
        scipy=scipy.__version__, cvxpy=cp.__version__, solvers=cp.installed_solvers(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_model_sha256=hashlib.sha256((COMMON / 'thesis_common/model.py').read_bytes()).hexdigest(),
        common_settings_sha256=settings_digest(),
    ))
    result = dict(
        relative=rank_example(out),
        obstructions=obstructions(out),
        frozen_star=frozen_star_run(out),
        fair_star=fair_star_benchmark(out),
        switching_star=switching_star_run(out),
        interruptions_star=interruptions_star_run(out),
    )
    result['settings_edges'] = json.loads((out / 'settings.json').read_text(encoding='utf-8'))['active_edges']
    dump(out / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

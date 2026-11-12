"""Comprehensively upgrade and re-render all active figures across the PhD dissertation
using the 2.0 publication visual standard (thesis_plot_style.py).

Read-only on all experimental data (.npz, .json, .csv), frozen scripts, and .tex tables.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys
sys.dont_write_bytecode = True

import csv
import importlib.util
import json
from pathlib import Path
from statistics import median
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import thesis_plot_style as tps
from thesis_plot_style import (
    apply_style, save_figure, panel, grid,
    halo_stroke, halo_text, halo_line,
    add_trajectory_arrows, draw_robot_nodes, shared_top_legend,
    COLORS, NODE_PALETTE, MM, FIEDLER,
)


# ==============================================================================
# 1. CHAPTER 2 FIGURES
# ==============================================================================
def upgrade_chapter2():
    print('=== Upgrading Chapter 2 figures ===')
    ch2 = ROOT / '01_论文写作' / '独立章节' / '第二章_分布式谱估计'
    exp_dir = ch2 / 'experiments'
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))

    # 1.1 gain_threshold (figures/v5/gain_threshold)
    import render_concepts_v5 as rc5
    rc5.gain_threshold()
    print('  [Ch2] gain_threshold done')

    # 1.2 weak_connection & information_flow (figures/v2/*)
    import render_concepts_v2 as rc2
    rc2.main()
    print('  [Ch2] weak_connection & information_flow done')

    # 1.3 reference_plane & reference_trajectory (figures/v6/*)
    import render_v6 as rv6
    rv6.fig_reference_plane()
    rv6.fig_reference_trajectory()
    print('  [Ch2] reference_plane & reference_trajectory done')

    # 1.4 gain_scan, comparison_failure, switch_comparison (figures/v7/*)
    import render_v7 as rv7
    rv7.fig_gain_scan()

    R = rv7.cmp_records()
    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(150 * MM, 64 * MM), layout='constrained')
    for m in ['nodiv_lc', 'ratio_lc', 'yang', 'zhang']:
        rs = rv7.cmp_rows(R, 'dense', 'biased', m)
        rv7.band(axs[0], rs, 'error', m)
        rv7.band(axs[1], rs, 'energy', m, floor=1e-300)
    axs[0].axhline(1e-3, color=COLORS['reference'], lw=0.85, ls=':')
    axs[0].set(xscale='log', yscale='log', xlim=(0.01, 120), ylim=(1e-7, 1e3),
               xlabel=r'算法时间 $t$', ylabel=r'相对谱值误差 $E_\lambda$')
    axs[1].set(xscale='log', yscale='log', xlim=(0.01, 120), ylim=(1e-12, 3),
               xlabel=r'算法时间 $t$', ylabel=r'均方幅值 $\|\boldsymbol{x}\|_2^2/N$')
    axs[0].legend(fontsize=8.0, loc='lower left')
    for ax, tag, title in zip(axs, 'ab', ['谱值误差', '均方幅值']):
        grid(ax)
        panel(ax, tag, title)
    save_figure(fig, rv7.FIG / 'comparison_failure')
    plt.close(fig)

    TS = 15.0
    apply_style()
    fig, axs = plt.subplots(2, 2, figsize=(150 * MM, 118 * MM), layout='constrained')
    keys = ['est', 'error', 'direction', 'mean_fraction']
    for a, ev in rv7.SW:
        rs = [r for r in rv7.load(ev / 'switch/summary.json') if r['arm'] == a and r['rtol'] == 2e-8]
        ds = [np.load(ev / 'switch' / (r['name'] + '.npz')) for r in rs]
        t = np.r_[ds[0]['t_pre'], ds[0]['t_post']]
        for ax, k in zip(axs.flat, keys):
            if k == 'est':
                y = np.array([np.r_[np.median(d['pre_est'], axis=1), np.median(d['post_est'], axis=1)] for d in ds])
            else:
                y = np.array([np.r_[d['pre_' + k], d['post_' + k]] for d in ds])
            q = np.quantile(y, [0.25, 0.5, 0.75], axis=0)
            if k in ('error', 'direction'):
                q[q < 1e-9] = np.nan
            ext = a.endswith('external')
            ax.plot(t, q[1], color=rv7.SW_COLOR[a], ls=rv7.SW_STYLE[a], lw=1.05 if ext else 1.45, alpha=0.55 if ext else 1.0)
            ax.fill_between(t, q[0], q[2], color=rv7.SW_COLOR[a], alpha=0.07, lw=0)
    lam_sw = rv7.load(rv7.EV6 / 'switch/manifest.json')['spectra']
    axs[0, 0].plot([12, TS], [lam_sw[0][1]] * 2, color=COLORS['reference'], ls=':', lw=1.05)
    axs[0, 0].plot([TS, 22], [lam_sw[1][1]] * 2, color=COLORS['reference'], ls=':', lw=1.05)
    for ax in axs.flat:
        ax.axvline(TS, color=COLORS['reference'], ls=':', lw=0.9)
        ax.set(xlim=(12, 22), xlabel=r'算法时间 $t$')
        grid(ax)
    axs[0, 0].set(ylim=(0, 28), ylabel=r'节点读数中位数')
    axs[0, 1].set(yscale='log', ylim=(1e-7, 1e3), ylabel=r'相对谱值误差 $E_\lambda$')
    axs[0, 1].axhline(1e-3, color=COLORS['reference'], ls=':', lw=0.8)
    axs[1, 0].set(yscale='log', ylim=(1e-7, 2), ylabel=r'完整方向误差 $E_v$')
    axs[1, 0].axhline(1e-3, color=COLORS['reference'], ls=':', lw=0.8)
    axs[1, 1].set(ylim=(-0.04, 1.06), ylabel=r'均值能量占比 $\zeta_{\mathrm{av}}^2/m_{\zeta,2}$')
    for ax, tag, title in zip(axs.flat, 'abcd', ['真值突变与读数响应', '各节点最大谱值误差', '方向选择', '固定基线转入一致性模态']):
        panel(ax, tag, title)
    handles = [
        Line2D([0], [0], color=rv7.SW_COLOR[a], ls=rv7.SW_STYLE[a],
               alpha=0.55 if a.endswith('external') else 1.0, label=rv7.SW_LABEL[a])
        for a, _ in rv7.SW
    ]
    fig.legend(handles=handles, loc='outside upper center', ncol=3, fontsize=8.4, frameon=False)
    save_figure(fig, rv7.FIG / 'switch_comparison')
    plt.close(fig)
    print('  [Ch2] gain_scan, comparison_failure, switch_comparison done')

    # 1.5 switch_topology (figures/v4/switch_topology)
    from switch_suite_v4 import OUT as SW_OUT
    M_v4 = json.loads((SW_OUT / 'manifest.json').read_text(encoding='utf-8'))
    FIG4 = ch2 / 'figures' / 'v4'
    representative = np.load(SW_OUT / 'proposed_s1200.npz')
    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(150 * MM, 50 * MM), layout='constrained')
    for k, ax in enumerate(axs):
        p = np.array(M_v4['graphs'][k]['positions'])
        L = representative['L_pre' if k == 0 else 'L_post']
        for i in range(len(p)):
            for j in range(i):
                if L[i, j] < -1e-14:
                    w = min(1.0, float(-L[i, j]))
                    ax.plot(p[[i, j], 0], p[[i, j], 1], color=COLORS['edge'],
                            alpha=0.14 + 0.30 * w, lw=0.35 + 0.55 * w, zorder=1)
        ax.scatter(p[:8, 0], p[:8, 1], s=32, c=COLORS['method'], edgecolors='white',
                   linewidths=0.85, label='原有节点', zorder=4)
        if k:
            ax.scatter(p[8:, 0], p[8:, 1], s=28, c=COLORS['comparison'], marker='^',
                       edgecolors='white', linewidths=0.75, label='新增节点', zorder=5)
        ax.set(xlim=(-1.02, 1.02), ylim=(-0.52, 0.52),
               xlabel=r'水平位置 $p_x$ ($\mathrm{m}$)', ylabel=r'纵向位置 $p_y$ ($\mathrm{m}$)', aspect='equal')
        panel(ax, 'ab'[k], f"$N={len(p)}$, $\\lambda_2={M_v4['graphs'][k]['spectrum'][1]:.3f}$")
        grid(ax)
    axs[1].legend(loc='lower right', fontsize=8.2)
    save_figure(fig, FIG4 / 'switch_topology')
    plt.close(fig)
    print('  [Ch2] switch_topology done')

    # 1.6 full_dac_v8 (figures/v7/full_dac_v8)
    _spec_v8 = importlib.util.spec_from_file_location(
        'make_ch2_figures_v8',
        ROOT / '01_论文写作' / '整篇论文' / 'evidence' / '20261101_写作重塑' / 'make_ch2_figures_v8.py',
    )
    m_v8 = importlib.util.module_from_spec(_spec_v8)
    _spec_v8.loader.exec_module(m_v8)
    m_v8.main()
    print('  [Ch2] full_dac_v8 done')


# ==============================================================================
# 2. CHAPTER 3 FIGURES
# ==============================================================================
def upgrade_chapter3():
    print('=== Upgrading Chapter 3 figures ===')
    ch3 = ROOT / '01_论文写作' / '独立章节' / '第三章_持久分配'
    fig_dir = ch3 / 'figures'
    if str(ch3) not in sys.path:
        sys.path.insert(0, str(ch3))

    # 2.1 concept_coupling, concept_joint_feasibility, concept_persistent
    import plot_concepts as pc3
    pc3.coupling_dilemma()
    pc3.joint_feasibility()
    pc3.persistent_coordinates()
    print('  [Ch3] concept_coupling, concept_joint_feasibility, concept_persistent done')

    # 2.2 pa_mechanism_block_v2
    import plot_mechanism_block_v2 as pmb
    pmb.draw()
    print('  [Ch3] pa_mechanism_block_v2 done')

    # 2.3 frozen_scene_v2
    run_honest = ch3 / 'experiments/runs/20261025_npa_honest_01'
    block_npz = ch3 / 'experiments/runs/20261030_npa_block_01/jobs/e31_block_K6_64.npz'
    summary = json.loads((run_honest / 'summary.json').read_text(encoding='utf-8'))
    star_data = np.load(run_honest / 'frozen_star_arrays.npz')
    block = np.load(block_npz)
    u_npa = block['final_u']
    p = star_data['p']
    adj = star_data['adjacency']
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(150 * MM, 74 * MM), layout='constrained')
    for i, j in summary['settings_edges']:
        ax[0].plot(p[[i, j], 0], p[[i, j], 1], color=COLORS['edge'], lw=0.45 + 1.15 * adj[i, j], alpha=0.55, zorder=1)
    ax[0].scatter(p[:, 0], p[:, 1], color=COLORS['method'], edgecolors='white', linewidths=0.9, s=34, zorder=4)
    for i in range(len(p)):
        halo_text(ax[0], p[i, 0] + 0.04, p[i, 1] + 0.05, str(i + 1), fontsize=8.6, zorder=5)
    for vv, color, label, ls in [
        (star_data['nominal'], COLORS['comparison'], '标称输入', '--'),
        (star_data['ustar'], COLORS['reference'], '集中最优', '--'),
        (star_data['final_u_tan'], COLORS['third'], 'Tan 等 (2025) 输入', ':'),
        (u_npa, COLORS['method'], 'NPA 输入', '-'),
    ]:
        for origin, v in zip(p, vv):
            end = origin + v / 2.5
            ax[0].plot([origin[0], end[0]], [origin[1], end[1]], color=color, lw=1.25, ls=ls, zorder=3.5)
            ax[0].annotate('', xy=end, xytext=origin + v / 2.5 * 0.82,
                           arrowprops=dict(arrowstyle='-|>,head_length=0.42,head_width=0.25',
                                           color=color, lw=1.1, shrinkA=0, shrinkB=0), zorder=3.8)
        ax[0].plot([], [], color=color, ls=ls, lw=1.35, label=label)
    ax[0].set_aspect('equal')
    ax[0].set(xlabel=r'水平位置 $x$ ($\mathrm{m}$)', ylabel=r'纵向位置 $y$ ($\mathrm{m}$)',
              xlim=(-0.4, 2.2), ylim=(-0.42, 1.35))
    panel(ax[0], 'a', '冻结图与控制输入方向对比')
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.12), ncol=2, handlelength=1.8, fontsize=8.2, frameon=False)
    grid(ax[0])

    vals = np.linalg.eigvalsh(np.diag(adj.sum(axis=1)) - adj)
    ax[1].bar(range(1, len(vals) + 1), vals, width=0.58, color=COLORS['method'], edgecolor='white', linewidth=0.6, zorder=3)
    ax[1].axhline(summary['frozen_star']['lambda_req'], ls='--', color=COLORS['reference'], lw=1.2, label=r'安全阈值 $\lambda_{\mathrm{req}}$')
    ax[1].annotate(r'$\lambda_2$', xy=(2, vals[1]), xytext=(2.65, vals[1] + 0.55),
                   fontsize=9, color=COLORS['ink'],
                   arrowprops=dict(arrowstyle='->', color=COLORS['spine'], lw=0.8))
    ax[1].set(xlabel=r'特征值序号 $q$', ylabel=r'拉普拉斯特征值 $\lambda_q$', xticks=range(1, len(vals) + 1))
    panel(ax[1], 'b', '加权拉普拉斯谱')
    ax[1].legend(loc='upper left', fontsize=8.4)
    grid(ax[1])
    save_figure(fig, fig_dir / 'frozen_scene_v2')
    plt.close(fig)
    print('  [Ch3] frozen_scene_v2 done')

    # 2.4 frozen_anytime_v2
    import plot_block_evidence_v2 as pbev2
    apply_style()
    inp = pbev2.Inputs()
    rec_dummy = {'figures': [], 'tables': []}
    pbev2.frozen_figure(fig_dir / 'frozen_anytime_v2', inp, rec_dummy)
    print('  [Ch3] frozen_anytime_v2 done')

    # 2.5 optimization_comparison, persistent_switching, anytime_interruptions
    import plot_evidence as pev
    apply_style()
    rec_ev = []
    pev.switching(rec_ev)
    pev.interruption(rec_ev)
    pev.fair_comparison(rec_ev)
    print('  [Ch3] optimization_comparison, persistent_switching, anytime_interruptions done')

    # 2.6 convergence_examples_block_v4 & closed_loop_settle_v4 (Upgraded GridSpec!)
    _spec_c3v4 = importlib.util.spec_from_file_location(
        'make_ch3_figures_v4',
        ROOT / '01_论文写作' / '整篇论文' / 'evidence' / '20261101_写作重塑' / 'make_ch3_figures_v4.py',
    )
    m_c3v4 = importlib.util.module_from_spec(_spec_c3v4)
    _spec_c3v4.loader.exec_module(m_c3v4)
    m_c3v4.make_examples_v4()

    # Upgraded GridSpec version of closed_loop_settle_v4 (eliminating cramped 1x3 layout)
    sys_ch3 = ROOT / '03_系统验证' / '20261030_第三章闭环重做'
    run = sys_ch3 / 'results' / 'run01'
    MA = m_c3v4.MA
    rows = MA.load(run)
    ARMS = ('L', 'M')
    ARM_STYLE = {
        'L': dict(color=COLORS['method'], ls='-', marker='o', mfc=COLORS['method'], label='NPA (本地结清)'),
        'M': dict(color=COLORS['comparison'], ls='--', marker='s', mfc='white', label='NPA (全网回退)'),
    }
    GAP_FLOOR = MA.GAP_FLOOR
    CERT_TOL = MA.CERT_TOL

    apply_style()
    fig = plt.figure(figsize=(150 * MM, 88 * MM), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.0, 1.0])
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_ecdf = fig.add_subplot(gs[0, 1])
    ax_marg = fig.add_subplot(gs[1, 1])

    # Panel (a): Spacious 6-robot planar motion trajectories and final connectivity topology
    d_est = np.load(run / 'runs' / 'B_T2_est_L_B8.npz')
    P = d_est['P']
    goals = d_est['goals']
    N = P.shape[1]
    r_com = 1.8
    P_end = P[-1]

    for i in range(N):
        for j in range(i + 1, N):
            dist = float(np.linalg.norm(P_end[i] - P_end[j]))
            if dist < r_com:
                w = np.exp(-dist ** 2 / 1.5 ** 2)
                ax_traj.plot(
                    [P_end[i, 0], P_end[j, 0]], [P_end[i, 1], P_end[j, 1]],
                    color=COLORS['edge'], lw=0.6 + 1.3 * w, alpha=0.48, zorder=2,
                )

    for i in range(N):
        c = NODE_PALETTE[i]
        ax_traj.plot(
            [P_end[i, 0], goals[i, 0]], [P_end[i, 1], goals[i, 1]],
            color=c, ls=':', lw=0.8, alpha=0.55, zorder=1.5,
        )
        ax_traj.plot(P[:, i, 0], P[:, i, 1], color=c, lw=1.45, zorder=3)
        add_trajectory_arrows(ax_traj, P[:, i, :], color=c, fractions=(0.52,), size=8.2, zorder=3.6)

    draw_robot_nodes(
        ax_traj,
        start_xy=P[0],
        end_xy=P_end,
        goal_xy=goals,
        colors=NODE_PALETTE[:N],
        ms_start=4.5,
        ms_end=5.4,
        ms_goal=5.0,
        label_nodes=True,
        label_offset=(0.0, 0.11),
    )

    h_traj = [
        Line2D([0], [0], color=COLORS['spine'], marker='o', mfc='white', mew=1.15, lw=0, ms=4.5, label='起点'),
        Line2D([0], [0], color=COLORS['method'], marker='o', mfc=COLORS['method'], mec='white', mew=0.9, lw=1.4, ms=5.0, label='轨迹与终态'),
        Line2D([0], [0], color=COLORS['spine'], marker='x', mew=1.2, lw=0, ms=4.8, label='拉伸目标'),
        Line2D([0], [0], color=COLORS['edge'], lw=1.5, alpha=0.65, label='终态通信边'),
    ]
    ax_traj.set_aspect('equal')
    ax_traj.set(
        xlim=(-1.65, 3.35), ylim=(-0.52, 1.58),
        xlabel=r'水平位置 $x$ ($\mathrm{m}$)', ylabel=r'纵向位置 $y$ ($\mathrm{m}$)',
    )
    ax_traj.legend(handles=h_traj, fontsize=8.0, ncol=2, loc='upper left', handlelength=1.4, columnspacing=0.8)
    grid(ax_traj)
    panel(ax_traj, 'a', '6 机运动轨迹与终态拓扑')

    # Panel (b): Per-step cost gap ECDF
    xs = np.logspace(np.log10(GAP_FLOOR), 0, 500)
    inputs = set()
    lows = []
    for Bn, alpha_b, lw_b in ((2, 0.55, 1.15), (8, 1.0, 1.55)):
        for arm in ARMS:
            st = ARM_STYLE[arm]
            g = np.concatenate(MA.arm_gaps(run, arm, Bn, inputs))
            F = MA.ecdf_on_log(g, xs)
            lows.append(float(F[0]))
            lab = f"{st['label']} ($B={Bn}$)"
            ax_ecdf.plot(xs, F, color=st['color'], ls=st['ls'], lw=lw_b, alpha=alpha_b, label=lab)
    ylo = np.floor(min(lows) * 20) / 20 - 0.05
    ax_ecdf.set(
        xscale='log', xlim=(GAP_FLOOR, 1), ylim=(ylo, 1.01),
        xlabel=r'每拍代价差 $J_k-J_k^\star$', ylabel='累计比例',
    )
    ax_ecdf.legend(fontsize=7.8, loc='lower right', handlelength=1.5)
    grid(ax_ecdf)
    panel(ax_ecdf, 'b', r'每拍代价差分布 ($B=2,8$)')

    # Panel (c): Connectivity safety margin and settlement events
    for arm in ('M', 'L'):
        nm = f'B_T2_true_{arm}_B8'
        d = np.load(run / 'runs' / f'{nm}.npz')
        st = ARM_STYLE[arm]
        y = d['audit_lambda2_min'] - rows[nm]['lambda_req']
        ax_marg.plot(d['t'], y, color=st['color'], ls=st['ls'], lw=1.35)
        ev = np.flatnonzero(d['log_kind'] != 'inherit')
        if arm == 'L':
            fail = np.flatnonzero(d['s1_spectral'] > CERT_TOL)
            if fail.size:
                cuts = np.flatnonzero(np.diff(fail) > 1)
                for a0, a1 in zip(np.r_[fail[0], fail[cuts + 1]], np.r_[fail[cuts], fail[-1]]):
                    ax_marg.axvspan(d['t'][a0], d['t'][a1] + d['dt'][a1], color=COLORS['failure'], alpha=0.13, lw=0)
            br = np.flatnonzero(d['log_broken_settled'] > 0)
            ax_marg.plot(d['t'][br], y[br], ls='none', marker='|', ms=8.5, mew=1.3, color=st['color'], label='断边结清')
        ax_marg.plot(
            d['t'][ev], y[ev], ls='none', marker='o' if arm == 'L' else 's',
            ms=5.2, mfc='none', mew=1.2, color=st['color'],
            label='本地结清' if arm == 'L' else '全网回退',
        )
    ax_marg.fill_between([], [], color=COLORS['failure'], alpha=0.15, label='零输入不可行')
    ax_marg.set_yscale('log')
    ax_marg.set(
        xlim=(0, 30),
        xlabel=r'时间 $t$ ($\mathrm{s}$)',
        ylabel=r'连通裕度 $\min\,\lambda_2-\lambda_{\mathrm{req}}$',
    )
    ax_marg.legend(loc='upper right', fontsize=7.6, ncol=2, handlelength=1.2, columnspacing=0.8)
    grid(ax_marg)
    panel(ax_marg, 'c', '连通裕度与结清事件')

    save_figure(fig, fig_dir / 'closed_loop_settle_v4')
    plt.close(fig)
    print('  [Ch3] convergence_examples_block_v4 & closed_loop_settle_v4 (GridSpec) done')


# ==============================================================================
# 3. CHAPTER 4 & CHAPTER 6 PHYSICAL FIGURES
# ==============================================================================
def upgrade_chapter4():
    print('=== Upgrading Chapter 4 figures ===')
    ch4 = ROOT / '01_论文写作' / '独立章节' / '第四章_子空间连通安全'
    ch6 = ROOT / '01_论文写作' / '独立章节' / '第六章_系统验证'
    fig4 = ch4 / 'figures'
    fig6 = ch6 / 'figures'

    # 3.1 concept_rectangle & concept_cone (with clean cage-free 3D Lorentz cone!)
    exp4 = ch4 / 'experiments'
    _spec_pc4 = importlib.util.spec_from_file_location('plot_concepts_ch4', exp4 / 'plot_concepts.py')
    pc4 = importlib.util.module_from_spec(_spec_pc4)
    _spec_pc4.loader.exec_module(pc4)
    pc4.rectangle_crossing()

    # Upgraded concept_cone: eliminate ugly mplot3d grey wireframe cage
    M = np.array([[1.0, 2.0], [2.0, 1.0]])
    z = pc4.jsoc(M)
    zs = z[1:] / z[0]
    apply_style()
    fig = plt.figure(figsize=(150 * MM, 68 * MM), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.05])

    ax = fig.add_subplot(gs[0], projection='3d', computed_zorder=False)
    ax.set_proj_type('ortho')
    ax.view_init(elev=19, azim=-58)
    h = np.linspace(0, 1.65, 36)
    th = np.linspace(0, 2 * np.pi, 100)
    H, T = np.meshgrid(h, th)
    ax.plot_surface(H * np.cos(T), H * np.sin(T), H, color=COLORS['method'], alpha=0.18, lw=0,
                    rstride=1, cstride=1, zorder=2)
    for hh in (0.5, 1.0, 1.5):
        ax.plot(hh * np.cos(th), hh * np.sin(th), hh + 0 * th, color=COLORS['method'], lw=0.75, alpha=0.7, zorder=3)
    for ang in np.linspace(0, 2 * np.pi, 13)[:-1]:
        ax.plot([0, 1.65 * np.cos(ang)], [0, 1.65 * np.sin(ang)], [0, 1.65],
                color=COLORS['method'], lw=0.45, alpha=0.42, zorder=2.5)

    # Tangent half-space plane z0 + z1 >= 0
    yy, zz = np.meshgrid(np.linspace(-1.45, 1.85, 2), np.linspace(0, 1.85, 2))
    ax.plot_surface(-zz, yy, zz, color=COLORS['comparison'], alpha=0.13, lw=0, zorder=1)
    ax.plot([-1.85, 0], [-1.45, -1.45], [1.85, 0], color=COLORS['comparison'], lw=0.8, ls='--', alpha=0.7, zorder=3)

    # Clean custom 3D coordinate axes (no wireframe box!)
    ax_kw = dict(color=COLORS['spine'], lw=0.85, zorder=1)
    ax.plot([-1.7, 1.85], [0, 0], [0, 0], **ax_kw)
    ax.plot([0, 0], [-1.7, 2.05], [0, 0], **ax_kw)
    ax.plot([0, 0], [0, 0], [0, 2.05], color=COLORS['spine'], lw=1.0, zorder=4)
    ax.text(1.98, 0, -0.05, r'$z_1$', fontsize=9.2, color=COLORS['ink'], ha='center')
    ax.text(0, 2.18, -0.05, r'$z_2$', fontsize=9.2, color=COLORS['ink'], ha='center')
    ax.text(0, 0, 2.15, r'$z_0$', fontsize=9.2, color=COLORS['ink'], ha='center')

    ax.scatter(*z[[1, 2, 0]], s=46, color=COLORS['failure'], marker='x', depthshade=False, zorder=6, linewidths=1.9)
    ax.plot([0, z[1]], [0, z[2]], [0, z[0]], color=COLORS['failure'], lw=1.0, ls=':', zorder=5)
    ax.text(z[1] + 0.12, z[2] + 0.08, z[0] + 0.08, r'$\mathcal{J}_{\mathrm{soc}}(M)$', fontsize=8.8, color=COLORS['ink'])
    ax.text(-1.55, -1.65, 0.55, r'$z_0+z_1\geq 0$', fontsize=8.5, color=COLORS['ink'])
    ax.text(0.32, -0.15, 1.72, r'$\mathcal{Q}_3$', fontsize=10.0, color=COLORS['method'])
    ax.set_xlim(-1.75, 1.85)
    ax.set_ylim(-1.75, 2.15)
    ax.set_zlim(0, 2.10)
    ax.set_box_aspect((1, 1.12, 0.82), zoom=1.18)
    ax.set_axis_off()
    panel(ax, 'a', '二维半正定锥即 Lorentz 锥', pad=2)

    ax = fig.add_subplot(gs[1])
    for ang in np.linspace(0, np.pi, 37)[:-1]:
        nvec = np.array([np.cos(2 * ang), np.sin(2 * ang)])
        tvec = np.array([-nvec[1], nvec[0]])
        base = -nvec
        seg = np.array([base - 2.6 * tvec, base + 2.6 * tvec])
        ax.plot(*seg.T, color=COLORS['grid'], lw=0.55, zorder=1)
    ax.add_patch(Circle((0, 0), 1, fc=COLORS['method'], alpha=0.20, ec=COLORS['method'], lw=1.4, zorder=2))
    halo_text(ax, 0.0, 0.0, '半正定\n切片', fontsize=8.6, color=COLORS['ink'], ha='center', va='center', zorder=4)
    ax.axvspan(-1, 2.5, color=COLORS['comparison'], alpha=0.06, lw=0, zorder=0)
    ax.axvline(-1, color=COLORS['comparison'], ls='--', lw=1.35, zorder=3)
    halo_text(ax, -0.88, -1.72, r'只检验 $v=e_1$：' + '\n' + r'接受 $z_1\geq -1$', fontsize=8.4, color=COLORS['ink'])
    u = zs / np.linalg.norm(zs)
    tv = np.array([-u[1], u[0]])
    tline = np.array([u - 1.9 * tv, u + 1.9 * tv])
    ax.plot(*tline.T, color=COLORS['failure'], lw=1.4, zorder=3)
    ax.plot(*zs, 'x', color=COLORS['failure'], ms=8, mew=1.9, zorder=5)
    halo_text(ax, zs[0] + 0.18, zs[1] + 0.04, r'$\mathcal{J}_{\mathrm{soc}}(M)/z_0$', fontsize=8.6, va='center')
    halo_text(ax, 0.38, 1.15, '最不利方向拒绝', fontsize=8.4)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'归一化分量 $z_1/z_0$')
    ax.set_ylabel(r'归一化分量 $z_2/z_0$')
    grid(ax)
    panel(ax, 'b', r'$z_0=1$ 切片：全部方向半平面之交')
    save_figure(fig, fig4 / 'concept_cone')
    plt.close(fig)
    print('  [Ch4] concept_rectangle & concept_cone done')

    # 3.2 crossing_results, frozen_recovery, physical_records
    src_data = ROOT / '02_核心研究/Fiedler_Subspace/current_version/distributed_fiedler_violation_free_soc/paper_data'
    exp7_dir = ROOT / '03_系统验证/20261024_跨章闭环补强/exp7_StarPA全量升级'
    exp7_res = exp7_dir / 'results/npa_honest_01'
    raw_dir = ROOT / '02_核心研究/Fiedler_Subspace/最新实物实验/动捕数据记录/run_20260821_194553_719606(实机实验)'
    if str(exp7_dir / 'code') not in sys.path:
        sys.path.insert(0, str(exp7_dir / 'code'))

    cross = pd.read_csv(src_data / 'core/rectangle_crossing.csv')
    apply_style()
    fig, axs = plt.subplots(2, 1, figsize=(150 * MM, 102 * MM), layout='constrained', sharex=True)
    g = cross[(cross.seed == 1) & (cross.method == 'ours-r2-block')]
    axs[0].plot(g.step, g.lambda2_true, color=COLORS['reference'], ls='--', lw=1.4, label=r'$\lambda_2$')
    axs[0].plot(g.step, g.lambda3_true, color=COLORS['third'], ls='-.', lw=1.4, label=r'$\lambda_3$')
    axs[0].set(ylabel=r'有序谱值 $\lambda_q$')
    axs[0].legend(ncol=2, loc='upper right')
    panel(axs[0], 'a', '矩形几何诱发的低频谱交叉')
    for method, label, color, ls in [
        ('ours-r2-block', '二维块 (本文)', COLORS['method'], '-'),
        ('ours-r1-ablation', '单列消融', COLORS['comparison'], '--'),
        ('yang-2010-single-fiedler', 'Yang 等 (2010) 方法', COLORS['third'], '-.'),
    ]:
        g = cross[(cross.seed == 1) & (cross.method == method) & (cross.step >= 60)]
        axs[1].plot(g.step, g.lambda2_estimate_mean - g.lambda2_true, label=label, color=color, ls=ls, lw=1.45)
    axs[1].axhline(0, color=COLORS['reference'], lw=0.75)
    axs[1].set(xlabel=r'构型样本 $k$', ylabel='谱值有符号误差')
    axs[1].legend(ncol=3, fontsize=8.4, loc='lower right')
    panel(axs[1], 'b', '同一初值下的估计误差')
    for ax in axs:
        for v in range(60, 361, 60):
            ax.axvline(v, color=COLORS['edge'], lw=0.6, ls=':')
        grid(ax)
    save_figure(fig, fig4 / 'crossing_results')
    plt.close(fig)

    # frozen_recovery (Chapter 4)
    from plot_exp7 import render_ch4_core_figures
    render_ch4_core_figures(exp7_res, [fig4])
    print('  [Ch4] crossing_results & frozen_recovery done')

    # physical_records (used in Chapter 5/6 physical section, saved to both fig4 and fig6)
    raw = pd.read_csv(raw_dir / 'timeseries.csv')
    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(150 * MM, 74 * MM), layout='constrained')
    p_starts, p_ends = [], []
    agent_cols = [COLORS['method'], COLORS['comparison'], COLORS['third'], COLORS['fourth']]
    for idx, ((agent, g), color, ls) in enumerate(zip(
        raw.groupby('agent'),
        agent_cols,
        ['-', '--', '-.', ':'],
    )):
        i = int(agent[-1])
        e = np.hypot(g.local_x_m - g.reference_x_m, g.local_y_m - g.reference_y_m)
        traj = np.column_stack([g.local_x_m.to_numpy(), g.local_y_m.to_numpy()])
        p_starts.append(traj[0])
        p_ends.append(traj[-1])
        axs[0].plot(traj[:, 0], traj[:, 1], color=color, ls=ls, lw=1.45, label=f'机器人 {i}', zorder=3)
        add_trajectory_arrows(axs[0], traj, color=color, fractions=(0.55,), size=8.0, zorder=3.6)
        axs[1].plot(g.local_elapsed_s, e, color=color, ls=ls, lw=1.35, label=f'机器人 {i}')
    draw_robot_nodes(
        axs[0],
        start_xy=np.array(p_starts),
        end_xy=np.array(p_ends),
        colors=agent_cols,
        ms_start=4.5,
        ms_end=5.2,
        label_nodes=True,
        label_offset=(0.0, 0.11),
    )
    axs[0].set(
        xlabel=r'局部水平位置 $x$ ($\mathrm{m}$)',
        ylabel=r'局部纵向位置 $y$ ($\mathrm{m}$)',
        xlim=(-1.12, 1.12), ylim=(-1.12, 1.12),
        aspect='equal',
    )
    panel(axs[0], 'a', '769 条本地报告中的运动轨迹')
    axs[1].set(xlabel=r'各节点本地经过时间 $t$ ($\mathrm{s}$)', ylabel=r'位置跟踪误差 ($\mathrm{m}$)')
    panel(axs[1], 'b', '按本地时间记录的跟踪误差')
    axs[1].legend(fontsize=8.4, ncol=2, loc='upper right')
    for ax in axs:
        grid(ax)
    save_figure(fig, fig4 / 'physical_records')
    save_figure(fig, fig6 / 'physical_records')
    plt.close(fig)
    print('  [Ch4/Ch6] physical_records done')

    # 3.3 topology_margins_v2, topology_geometry_v2, estimated_pipeline_v2,
    #     ch4_pa_closed_loop_v3, c4_npa_block_settle_b2B_L2_v2
    c4_code = ROOT / '02_章节代码' / '第四章_子空间连通安全' / '20261030_ConicNPA块与结清' / 'code'
    if str(c4_code) not in sys.path:
        sys.path.insert(0, str(c4_code))
    _spec_v2 = importlib.util.spec_from_file_location('c4_assets_v2', c4_code / 'make_thesis_assets_v2.py')
    V2 = importlib.util.module_from_spec(_spec_v2)
    _spec_v2.loader.exec_module(V2)

    # Override V2.save_fig to use thesis_plot_style.save_figure directly into fig4
    V2.save_fig = lambda f, stem: save_figure(f, fig4 / stem)
    V2.write_tex = lambda name, text: None  # Do not touch any .tex tables!

    # Render topology_margins_v2 and upgraded topology_geometry_v2
    top = V2.rcsv(V2.TOPO / 'topology_formation.csv')
    trb = V2.trace('e43_01', 'topology_block_v1_T5')
    P_conic = V2.reconstruct_positions(trb, V2.R7.TOPOLOGY_INITIAL, 0.025)
    marg = np.array([x['exact_global_matrix_margin'] for x in trb])
    conn = np.array([x['connectivity_safety_margin'] for x in trb])

    apply_style()
    fig, axs = plt.subplots(2, 1, figsize=(150 * MM, 102 * MM), sharex=True, layout='constrained')
    t_b = np.array([(x['step'] + 1) * 0.025 for x in trb])
    series = [
        ('conic', 'Conic NPA', COLORS['method'], '-'),
        ('centralized', '集中式精确 SOCP', COLORS['reference'], '--'),
        ('nominal', '标称控制', COLORS['comparison'], '-.'),
        ('aggregate', '一致性参数聚合', COLORS['fourth'], ':'),
    ]
    for key, label, color, ls in series:
        if key == 'conic':
            axs[0].plot(t_b, conn, label=label, color=color, ls=ls, lw=1.55)
            axs[1].plot(t_b, marg, label=label, color=color, ls=ls, lw=1.55)
        else:
            g = top[top.method == key]
            axs[0].plot(g.time_s, g.connectivity_safety_margin, label=label, color=color, ls=ls, lw=1.35)
            if key == 'centralized':
                axs[1].plot(g.time_s, g.exact_global_matrix_margin, label=label, color=color, ls=ls, lw=1.35)
    axs[0].axhline(0, color=COLORS['failure'], ls=':', lw=1.0)
    axs[0].set(ylabel=r'采样连通裕度 $\lambda_2-\lambda_{\mathrm{req}}$')
    axs[0].legend(ncol=2, fontsize=8.2, loc='upper right')
    panel(axs[0], 'a', '强目标场景的采样连通裕度')
    axs[1].axhline(0, color=COLORS['failure'], ls=':', lw=1.0)
    axs[1].set(xlabel=r'物理时间 $t$ ($\mathrm{s}$)', ylabel='真实矩阵最小特征值')
    axs[1].legend(ncol=2, fontsize=8.2, loc='upper right')
    panel(axs[1], 'b', '执行输入对应的精确矩阵裕度')
    for ax in axs:
        grid(ax)
    save_figure(fig, fig4 / 'topology_margins_v2')
    plt.close(fig)

    # Upgraded topology_geometry_v2 with cool slate edges, trajectory arrows & white-ring markers
    tra = V2.rnpz(V2.TOPO / 'topology_trajectories.npz')
    scen = V2.rcsv(V2.TOPO / 'topology_scenario.csv')
    goals = scen[scen.configuration == 'goal'][['x_m', 'y_m']].to_numpy()
    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(150 * MM, 76 * MM), layout='constrained', sharex=True, sharey=True)
    lim = {'x': (-26.0, 40.0), 'y': (-24.0, 26.0)}
    for ax, path, tag, title in zip(
        axs,
        [np.asarray(tra['nominal']), P_conic],
        ['a', 'b'],
        ['标称控制：末态分裂', 'Conic NPA 滤波：末态连通'],
    ):
        edges, _ = V2.VF.performance_edges_and_gradients(
            path[-1], V2.R7.complete_topology(6), sigma=1, channel_parameters=V2.R7.MoxChannelParameters()
        )
        for i, j, w in edges:
            if w > 0:
                ax.plot(
                    path[-1, [i, j], 0], path[-1, [i, j], 1],
                    color=COLORS['edge'], lw=0.6 + 1.8 * w, alpha=0.58, zorder=2,
                )
        for i in range(6):
            ax.plot(path[:, i, 0], path[:, i, 1], color=COLORS['method'], lw=1.25, alpha=0.82, zorder=3)
            add_trajectory_arrows(ax, path[:, i, :], color=COLORS['method'], fractions=(0.55,), size=7.8, zorder=3.5)
            halo_text(ax, path[-1, i, 0] + 1.1, path[-1, i, 1] + 1.3, str(i + 1), fontsize=8.5, zorder=5.5)
        ax.scatter(
            path[0, :, 0], path[0, :, 1],
            facecolors='white', edgecolors=COLORS['reference'], linewidths=1.1, s=26, label='初态', zorder=4,
        )
        ax.scatter(
            path[-1, :, 0], path[-1, :, 1],
            color=COLORS['method'], edgecolors='white', linewidths=1.0, s=34, label='末态', zorder=5,
        )
        ax.scatter(
            goals[:, 0], goals[:, 1],
            marker='D', s=24, facecolors='white', edgecolors=COLORS['comparison'], linewidths=1.15, label='目标', zorder=3.8,
        )
        ax.set(
            aspect='equal',
            xlabel=r'水平位置 $x$ ($\mathrm{m}$)',
            ylabel=r'纵向位置 $y$ ($\mathrm{m}$)',
            xlim=lim['x'], ylim=lim['y'],
        )
        grid(ax)
        panel(ax, tag, title)
    axs[0].legend(loc='lower right', fontsize=8.2)
    save_figure(fig, fig4 / 'topology_geometry_v2')
    plt.close(fig)

    # estimated_pipeline_v2, c4_npa_block_settle_b2B_L2_v2, ch4_pa_closed_loop_v3
    V2.pipeline_assets()

    def _save_ablation_b2B(f, stem):
        if stem == 'c4_npa_block_settle_b2B_L2_v2':
            f.axes[0].legend(loc='center right', bbox_to_anchor=(1.0, 0.52), fontsize=7.2)
        save_figure(f, fig4 / stem)

    V2.save_fig = _save_ablation_b2B
    V2.fig_ablation_b2B()

    D = V2.closed_loop_assets()
    SINGLE = [f'B2_{t}_single_{r}_npaL' for t in 'AB' for r in ('r1_L8', 'r5_L2')]
    _orig_trace = V2.trace
    V2.trace = lambda run, name: _orig_trace(run, name + '_eta8' if (run == 'e44_01' and name in SINGLE) else name)
    def _save_closed_loop_v3(f, stem):
        f.set_size_inches(150 * MM, 116 * MM)
        for ax_i in f.axes:
            leg = ax_i.get_legend()
            if leg is not None:
                for txt_i in leg.get_texts():
                    txt_i.set_text(txt_i.get_text().replace(r'\rho=1', r'\rho_{\mathrm{est}}=1').replace(r'\rho=5', r'\rho_{\mathrm{est}}=5'))
                    txt_i.set_fontsize(7.8)
        save_figure(f, fig4 / ('ch4_pa_closed_loop_v3' if stem == 'ch4_pa_closed_loop_v2' else stem), layout='P4')
    V2.save_fig = _save_closed_loop_v3
    V2.fig_closed_loop(D)
    print('  [Ch4] topology_margins_v2, topology_geometry_v2, estimated_pipeline_v2, c4_npa_block_settle_b2B_L2_v2, ch4_pa_closed_loop_v3 done')


# ==============================================================================
# 4. CHAPTER 5 FIGURES
# ==============================================================================
def upgrade_chapter5_and_6():
    print('=== Upgrading Chapter 5 figures ===')
    ch5_script = ROOT / '01_论文写作' / '独立章节' / '第五章_安全强化学习与协同脱困' / 'make_thesis_figures.py'
    spec = importlib.util.spec_from_file_location('make_ch5_thesis_figures', ch5_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.fig1_swap_trajectories()
    mod.fig2_time_evolution()
    mod.fig3_perm_and_gate()
    print('  [Ch5] standalone Chapter 5 figures done')


if __name__ == '__main__':
    upgrade_chapter2()
    upgrade_chapter3()
    upgrade_chapter4()
    upgrade_chapter5_and_6()
    print('=== ALL THESIS FIGURES UPGRADED SUCCESSFULLY ===')

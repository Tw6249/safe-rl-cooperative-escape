"""Generate publication-grade vector figures (both Chinese and English editions)
for the standalone journal paper on EGP-NPA (Chapter 5).

Strictly read-only on frozen experimental data in 02_章节代码/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STYLE_DIR = ROOT / "01_论文写作" / "绘图规范"
sys.path.insert(0, str(STYLE_DIR))
from thesis_plot_style import (  # noqa: E402
    apply_style, new_figure, save_figure, panel, grid,
    halo_text, halo_line, add_trajectory_arrows,
    draw_robot_nodes, shared_top_legend,
    COLORS, MM,
)

OUT_ZH = HERE / "figures_zh"
OUT_EN = HERE / "figures_en"
OUT_ZH.mkdir(parents=True, exist_ok=True)
OUT_EN.mkdir(parents=True, exist_ok=True)

V5 = ROOT / "02_章节代码" / "第五章_安全强化学习与协同脱困"
AUDIT7 = V5 / "实现" / "results" / "mobile_robot_local_minimum"
IND = V5 / "独立验证"
S = IND / "20261102_安全强化学习" / "results"
G = IND / "20261103_终段包络验证" / "results"
FAIR = IND / "20261103_包络公平检验" / "results" / "run01"
SWAP = IND / "20261103_门控交换检验" / "results" / "run01"
STEP0 = IND / "20261104_代数特征拔插头与平滑检验" / "results" / "run01"

V6 = ROOT / "02_章节代码" / "第六章_虚拟物理仿真与系统验证" / "results"
V6_A = V6 / "run_20261104_v2" / "exp2_gnn_npa_mujoco"
V6_B = V6 / "run_20261104_v4" / "exp2_gnn_npa_mujoco"
SOTA_RUNS = HERE / "experiments_sota_rl" / "results" / "eval_runs"
MATCHED = ROOT / "独立测试_20261107_第五章强化学习改进" / "results"


def matched_run_dir(scene, tag):
    candidates = [MATCHED / part / "runs" for part in ("phaseC_part1", "phaseC_part2")]
    found = [directory for directory in candidates if (directory / f"{scene}_{tag}.npz").exists()]
    if len(found) != 1:
        raise ValueError(f"Expected one matched trajectory: {scene}/{tag}")
    return found[0]


def draw_comm_edges(ax, pos, r_comm=2.5, r_c=10.0):
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d <= r_comm:
                w = max(0.0, 1.0 - (d / r_c) ** 2) ** 2
                ax.plot(
                    [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color=COLORS["edge"], lw=0.55 + 0.75 * w, alpha=0.38, zorder=2.0,
                )


def fig1_seven_geometry(lang: str = "zh") -> None:
    """P3 figure: (a) 7-robot strict local minimum configuration & KKT critical velocity witness;
    (b) analytical safe connected reachability path; (c) task error W(p(s)) energy barrier profile."""
    apply_style()
    out_dir = OUT_ZH if lang == "zh" else OUT_EN
    fig, axes = new_figure("P3")
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    p_bar = np.array([
        [-1.0, 0.0],
        [-0.5, np.sqrt(3) / 2.0],
        [-0.5, -np.sqrt(3) / 2.0],
        [0.0, 0.0],
        [0.5, np.sqrt(3) / 2.0],
        [0.5, -np.sqrt(3) / 2.0],
        [1.0, 0.0],
    ])
    z_star = np.array([
        [-15.0, 15.0 * np.sqrt(3)],
        [-132.0, 132.0 * np.sqrt(3)],
        [-105.0, -15.0 * np.sqrt(3)],
        [90.0, 206.0 * np.sqrt(3)],
        [177.0, 177.0 * np.sqrt(3)],
        [-105.0, -225.0 * np.sqrt(3)],
        [90.0, -290.0 * np.sqrt(3)],
    ]) / 496.0

    # Panel (a): Contact graph, safety disks, and critical-cone optimal velocity witness z_*
    for i in range(7):
        for j in range(i + 1, 7):
            if abs(np.linalg.norm(p_bar[i] - p_bar[j]) - 1.0) < 1e-5:
                ax1.plot(
                    [p_bar[i, 0], p_bar[j, 0]], [p_bar[i, 1], p_bar[j, 1]],
                    color=COLORS["reference"], ls="--", lw=0.95, alpha=0.65, zorder=2.0,
                )
    for idx in range(7):
        is_swap = idx in (3, 6)
        col = COLORS["method"] if idx == 3 else (COLORS["comparison"] if idx == 6 else "#94A3B8")
        circ = plt.Circle(
            p_bar[idx], 0.5, facecolor=col, edgecolor=col,
            alpha=0.13 if is_swap else 0.08, lw=0.75, ls="-" if is_swap else ":", zorder=1.5,
        )
        ax1.add_patch(circ)
        ax1.plot(
            p_bar[idx, 0], p_bar[idx, 1], marker="o", ms=6.2 if is_swap else 5.2,
            mfc=col, mec="white", mew=1.1, zorder=4.5,
        )
        # Critical-cone velocity vector arrow
        v = z_star[idx] * 0.55
        arr_col = COLORS["method"] if idx == 3 else (COLORS["comparison"] if idx == 6 else COLORS["third"])
        ax1.annotate(
            "", xy=(p_bar[idx, 0] + v[0], p_bar[idx, 1] + v[1]), xytext=(p_bar[idx, 0], p_bar[idx, 1]),
            arrowprops=dict(arrowstyle="-|>", color=arr_col, lw=1.35, mutation_scale=8.5),
            zorder=5.0,
        )
        dx, dy = (0.0, -0.24) if idx in (0, 3, 6) else (0.0, 0.22 if p_bar[idx, 1] > 0 else -0.24)
        halo_text(ax1, p_bar[idx, 0] + dx, p_bar[idx, 1] + dy, f"{idx + 1}", ha="center", va="center", fontsize=8.3)

    ax1.set_xlim(-1.68, 1.68)
    ax1.set_ylim(-1.48, 1.48)
    ax1.set_aspect("equal")
    if lang == "zh":
        panel(ax1, "a", "七机接触构型与临界锥速度")
        ax1.set_xlabel(r"水平位置 $x$ ($\mathrm{m}$)")
        ax1.set_ylabel(r"纵向位置 $y$ ($\mathrm{m}$)")
    else:
        panel(ax1, "a", "7-Robot Contact & Critical Velocity")
        ax1.set_xlabel(r"Horizontal Position $x$ ($\mathrm{m}$)")
        ax1.set_ylabel(r"Vertical Position $y$ ($\mathrm{m}$)")
    grid(ax1)

    # Panel (b): Analytical reachability witness path
    wit = np.load(AUDIT7 / "reachability_witness.npz")
    states, goal, w_path = wit["states"], wit["goal"], wit["W"]
    for k in range(7):
        if k not in (3, 6):
            ax2.plot(states[:, k, 0], states[:, k, 1], color=COLORS["third"], lw=1.05, ls="--", alpha=0.65, zorder=2.8)
            ax2.plot(p_bar[k, 0], p_bar[k, 1], marker="o", ms=4.2, mfc="#94A3B8", mec="white", mew=0.85, zorder=4.0)
    for k, col in ((3, COLORS["method"]), (6, COLORS["comparison"])):
        halo_line(ax2, states[:, k, 0], states[:, k, 1], color=col, lw=1.75, zorder=3.8)
        add_trajectory_arrows(ax2, states[:, k, :], color=col, fractions=(0.45,), size=8.5, zorder=4.2)
    draw_robot_nodes(
        ax2, start_xy=p_bar[[3, 6]], end_xy=states[-1, [3, 6]], goal_xy=goal[[3, 6]],
        colors=[COLORS["method"], COLORS["comparison"]], ms_start=4.6, ms_end=5.4, ms_goal=5.0,
    )
    ax2.set_xlim(-3.4, 3.4)
    ax2.set_ylim(-2.95, 2.95)
    if lang == "zh":
        panel(ax2, "b", "满足连通与避碰的解析可达轨迹")
        ax2.set_ylabel(r"纵向 $y$ ($\mathrm{m}$)")
    else:
        panel(ax2, "b", "Safe Connected Reachability Path")
        ax2.set_ylabel(r"$y$ ($\mathrm{m}$)")
    grid(ax2)

    # Panel (c): Task error W(p(s)) energy barrier profile
    s_param = np.linspace(0.0, 3.0, len(w_path))
    halo_line(ax3, s_param, w_path, color=COLORS["method"], lw=1.65)
    ax3.axhline(1.0, color=COLORS["reference"], ls="--", lw=1.05)
    ax3.set_xlim(0, 3.0)
    ax3.set_ylim(0, 29.5)
    if lang == "zh":
        halo_text(ax3, 2.92, 2.2, r"初态局部极小 $W(\bar{\boldsymbol{p}})=1$", ha="right", va="bottom", fontsize=8.0)
        panel(ax3, "c", "可达路径上的任务误差能量壁垒")
        ax3.set_xlabel(r"路径参数 $s$")
        ax3.set_ylabel(r"误差 $W(\boldsymbol{p}(s))$")
    else:
        halo_text(ax3, 2.92, 2.2, r"Local Min. $W(\bar{\boldsymbol{p}})=1$", ha="right", va="bottom", fontsize=8.0)
        panel(ax3, "c", "Task Error Energy Barrier Profile")
        ax3.set_xlabel(r"Path Parameter $s$")
        ax3.set_ylabel(r"Error $W(\boldsymbol{p}(s))$")
    grid(ax3)

    h_r4, = plt.plot([], [], color=COLORS["method"], lw=1.75)
    h_r7, = plt.plot([], [], color=COLORS["comparison"], lw=1.75)
    h_yd, = plt.plot([], [], color=COLORS["third"], lw=1.2, ls="--")
    h_ct, = plt.plot([], [], color=COLORS["reference"], lw=0.95, ls="--")
    if lang == "zh":
        shared_top_legend(
            fig, [h_r4, h_r7, h_yd, h_ct],
            ["节点 4 轨迹/速度", "节点 7 轨迹/速度", "静止邻居退让方向", "接触边 ($d_{ij}=1\\,\\mathrm{m}$)"],
            ncol=4,
        )
    else:
        shared_top_legend(
            fig, [h_r4, h_r7, h_yd, h_ct],
            ["Robot 4 Path/Vel.", "Robot 7 Path/Vel.", "Neighbor Yielding", "Contact Edge ($1\\,\\mathrm{m}$)"],
            ncol=4,
        )
    save_figure(fig, out_dir / "fig1_seven_geometry", layout="P3")
    plt.close(fig)


def fig2_training_curves(lang: str = "zh") -> None:
    """1x2 P2 training convergence figure."""
    apply_style()
    out_dir = OUT_ZH if lang == "zh" else OUT_EN
    fig, (ax1, ax2) = new_figure("P2")

    if lang == "zh":
        specs = [
            ("EGP-NPA (本文方法)", "train_full02", COLORS["method"], "-", 1.75, True),
            ("纯几何观测图策略", "train_geom02", COLORS["comparison"], "--", 1.40, False),
            ("滤波修正量门控策略", "train_fullint02", COLORS["third"], "-.", 1.40, False),
        ]
    else:
        specs = [
            ("EGP-NPA (Proposed)", "train_full02", COLORS["method"], "-", 1.75, True),
            ("Pure Geometric GNN", "train_geom02", COLORS["comparison"], "--", 1.40, False),
            ("Intervention-Gated GNN", "train_fullint02", COLORS["third"], "-.", 1.40, False),
        ]
    handles, labels = [], []
    w = 7
    for lab, folder, col, ls, lw, is_main in specs:
        raw = [
            json.loads(line)
            for line in (S / folder / "train_log.jsonl").read_text(encoding="utf8").splitlines()
            if line.strip()
        ]
        tr = [r for r in raw if "ret" in r]
        steps = np.array([r["env_steps"] / 1e4 for r in tr])
        rets = np.array([r["ret"] for r in tr])
        ax1.plot(steps, rets, color=col, lw=0.55, alpha=0.22, zorder=2.0)
        sm_ret = np.convolve(rets, np.ones(w) / w, mode="valid")
        sm_step = steps[w - 1:]
        if is_main:
            ln = halo_line(ax1, sm_step, sm_ret, color=col, ls=ls, lw=lw, label=lab, zorder=3.8)[0]
        else:
            ln = ax1.plot(sm_step, sm_ret, color=col, ls=ls, lw=lw, label=lab, zorder=3.2)[0]
        handles.append(ln)
        labels.append(lab)

        vals = [(r["env_steps"] / 1e4, 100.0 * r["val"]["success"]) for r in raw if "val" in r]
        vx, vy = zip(*vals)
        if is_main:
            halo_line(ax2, vx, vy, color=col, ls=ls, lw=lw, zorder=3.8)
        else:
            ax2.plot(vx, vy, color=col, ls=ls, lw=lw, zorder=3.2)
        ax2.plot(vx[::2], vy[::2], marker="o", ls="none", ms=3.5, mfc=col, mec="white", mew=0.75, zorder=4.2)

    ax1.set_xlim(0, 97)
    ax1.set_ylim(0.55, 1.98)
    ax2.axhline(19.44, color=COLORS["reference"], ls="--", lw=1.05)
    ax2.set_xlim(0, 97)
    ax2.set_ylim(10, 105)

    if lang == "zh":
        ax1.set_xlabel(r"环境交互步数 ($\times 10^4$)")
        ax1.set_ylabel(r"回合累积奖励 $R_{\mathrm{ep}}$")
        panel(ax1, "a", "训练回合累积奖励收敛曲线")
        halo_text(ax2, 94.0, 23.0, r"初始零残差策略 ($19.4\%$)", ha="right", va="bottom", fontsize=8.0)
        ax2.set_xlabel(r"环境交互步数 ($\times 10^4$)")
        ax2.set_ylabel(r"验证集达标率 ($\%$)")
        panel(ax2, "b", "验证集确定性评估达标率演化")
    else:
        ax1.set_xlabel(r"Environment Steps ($\times 10^4$)")
        ax1.set_ylabel(r"Episode Return $R_{\mathrm{ep}}$")
        panel(ax1, "a", "Training Episode Return Convergence")
        halo_text(ax2, 94.0, 23.0, r"Zero-Residual Init. ($19.4\%$)", ha="right", va="bottom", fontsize=8.0)
        ax2.set_xlabel(r"Environment Steps ($\times 10^4$)")
        ax2.set_ylabel(r"Validation Arrival Rate ($\%$)")
        panel(ax2, "b", "Validation Deterministic Arrival Rate")
    grid(ax1)
    grid(ax2)
    shared_top_legend(fig, handles, labels, ncol=3)
    save_figure(fig, out_dir / "fig2_training_curves", layout="P2")
    plt.close(fig)


def fig3_swap_trajectories(lang: str = "zh") -> None:
    """2x2 P4 spatial trajectory & topology comparison on N=20 interior pair swap."""
    apply_style()
    out_dir = OUT_ZH if lang == "zh" else OUT_EN
    scene = "H20_local1_s20261105"
    if lang == "zh":
        specs = [
            ("a", FAIR / "runs", "rl_full_env_B8", "EGP-NPA (13.45 s 达标)"),
            ("b", FAIR / "runs", "heur_B8", "切向环流启发式 (27.25 s 达标)"),
            ("c", SOTA_RUNS, "rl_proj_env_B8", "投影一致性强化学习 (活锁于 1.01 m)"),
            ("d", SOTA_RUNS, "rl_static_cbf_B0", "静态平摊护盾图策略 (冻结于 1.00 m)"),
        ]
    else:
        specs = [
            ("a", FAIR / "runs", "rl_full_env_B8", "EGP-NPA (Arrived at 13.45 s)"),
            ("b", FAIR / "runs", "heur_B8", "Circulation Heuristic (27.25 s)"),
            ("c", SOTA_RUNS, "rl_proj_env_B8", "Proj-MAPPO + NPA (Stalled 1.01 m)"),
            ("d", SOTA_RUNS, "rl_static_cbf_B0", "Static-Split GNN (Frozen 1.00 m)"),
        ]
    fig, axes = new_figure("P4")
    for ax, (tag_let, run_dir, mtag, title) in zip(axes.flat, specs):
        z = np.load(run_dir / f"{scene}_{mtag}.npz")
        st, goal = z["states"], z["goal"]
        p0, pend = st[0], st[-1]
        n = len(p0)
        movers = [i for i in range(n) if np.linalg.norm(p0[i] - goal[i]) > 1e-6]
        stat = [i for i in range(n) if i not in movers]

        draw_comm_edges(ax, p0, r_comm=1.55)
        for rank, i in enumerate(movers):
            c = COLORS["method"] if rank == 0 else COLORS["comparison"]
            circ = plt.Circle(pend[i], 0.5, facecolor=c, edgecolor=c, alpha=0.10, lw=0.7, ls="--", zorder=1.5)
            ax.add_patch(circ)

        disp_max = np.linalg.norm(st[:, stat, :] - p0[None, stat, :], axis=2).max(axis=0)
        for idx_s, i in enumerate(stat):
            if disp_max[idx_s] > 0.025:
                ax.plot(st[:, i, 0], st[:, i, 1], color=COLORS["third"], lw=1.15, alpha=0.80, zorder=3.0)
            ax.plot(
                p0[i, 0], p0[i, 1], marker="o", ms=4.2,
                mfc=(COLORS["third"] if disp_max[idx_s] > 0.025 else "#CBD5E1"),
                mec="white", mew=0.9, zorder=4.2,
            )

        mover_colors = [COLORS["method"], COLORS["comparison"]]
        for rank, i in enumerate(movers):
            c = mover_colors[rank]
            halo_line(ax, st[:, i, 0], st[:, i, 1], color=c, lw=1.85, zorder=3.8)
            add_trajectory_arrows(ax, st[:, i, :], color=c, fractions=(0.45,), size=8.5, zorder=4.1)
        draw_robot_nodes(
            ax, start_xy=p0[movers], end_xy=pend[movers], goal_xy=goal[movers],
            colors=mover_colors, ms_start=4.6, ms_end=5.4, ms_goal=5.0,
        )

        lim = 2.95
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        panel(ax, tag_let, title)
        if tag_let in ("c", "d"):
            ax.set_xlabel(r"水平位置 $x$ ($\mathrm{m}$)" if lang == "zh" else r"Horizontal Position $x$ ($\mathrm{m}$)")
        if tag_let in ("a", "c"):
            ax.set_ylabel(r"纵向位置 $y$ ($\mathrm{m}$)" if lang == "zh" else r"Vertical Position $y$ ($\mathrm{m}$)")
        grid(ax)

    h_m1, = plt.plot([], [], color=COLORS["method"], lw=1.85)
    h_m2, = plt.plot([], [], color=COLORS["comparison"], lw=1.85)
    h_st, = plt.plot([], [], marker="o", ls="-", color=COLORS["third"], lw=1.15, mfc=COLORS["third"], mec="white", ms=4.5)
    h_id, = plt.plot([], [], marker="o", ls="none", mfc="#CBD5E1", mec="white", ms=4.5)
    h_gl, = plt.plot([], [], marker="x", ls="none", color=COLORS["ink"], ms=5.0, mew=1.25)
    if lang == "zh":
        shared_top_legend(
            fig, [h_m1, h_m2, h_st, h_id, h_gl],
            ["互换节点 1 轨迹", "互换节点 2 轨迹", "退让邻居轨迹", "静止邻居", "目标位置"],
            ncol=5,
        )
    else:
        shared_top_legend(
            fig, [h_m1, h_m2, h_st, h_id, h_gl],
            ["Mover 1", "Mover 2", "Yielding Neighbor", "Idle Neighbor", "Goal"],
            ncol=5,
        )
    save_figure(fig, out_dir / "fig3_swap_trajectory", layout="P4")
    plt.close(fig)


def fig4_time_evolution(lang: str = "zh") -> None:
    """2x2 P4 closed-loop time response comparison on N=20 interior swap."""
    apply_style()
    out_dir = OUT_ZH if lang == "zh" else OUT_EN
    sc_e2 = "H20_local1_s20261017"
    if lang == "zh":
        entries = [
            ("EGP-NPA", G / "e2_full_env_B8" / "runs", f"{sc_e2}_rl_full_env_B8", COLORS["method"], "-", 1.75, True),
            ("投影一致 RL", SOTA_RUNS, f"{sc_e2}_rl_proj_env_B8", COLORS["fifth"], (0, (3, 1, 1, 1)), 1.40, False),
            ("无包络消融", S / "test_e2_B8" / "runs", f"{sc_e2}_rl_full_B8", COLORS["comparison"], "--", 1.35, False),
            ("切向环流", S / "test_e2_B8" / "runs", f"{sc_e2}_heur_B8", COLORS["third"], "-.", 1.35, False),
            ("静态平摊图策略", SOTA_RUNS, f"{sc_e2}_rl_static_cbf_B0", COLORS["fourth"], ":", 1.45, False),
        ]
    else:
        entries = [
            ("EGP-NPA", G / "e2_full_env_B8" / "runs", f"{sc_e2}_rl_full_env_B8", COLORS["method"], "-", 1.75, True),
            ("Proj-MAPPO", SOTA_RUNS, f"{sc_e2}_rl_proj_env_B8", COLORS["fifth"], (0, (3, 1, 1, 1)), 1.40, False),
            ("w/o Envelope", S / "test_e2_B8" / "runs", f"{sc_e2}_rl_full_B8", COLORS["comparison"], "--", 1.35, False),
            ("Circulation", S / "test_e2_B8" / "runs", f"{sc_e2}_heur_B8", COLORS["third"], "-.", 1.35, False),
            ("Static-Split GNN", SOTA_RUNS, f"{sc_e2}_rl_static_cbf_B0", COLORS["fourth"], ":", 1.45, False),
        ]
    fig, axes = new_figure("P4")
    ax_err, ax_lam, ax_dist, ax_vel = axes.flat
    handles, labels = [], []

    for lab, d, stem, col, ls, lw, is_main in entries:
        z = np.load(d / f"{stem}.npz")
        tr = json.loads((d / f"{stem}_trace.json").read_text(encoding="utf8"))
        st, goal, ts = z["states"], z["goal"], np.array(z["times"])
        t_tr = np.array([x["t"] for x in tr])
        err_max = np.linalg.norm(st - goal[None], axis=2).max(axis=1)
        lam2 = np.array([x["lambda2"] for x in tr])
        mdist = np.array([x["min_distance"] for x in tr])
        u_rms = np.array([x["u_norm"] for x in tr]) / np.sqrt(st.shape[1])

        plot_fn = (lambda ax, x, y, **kw: halo_line(ax, x, y, **kw)) if is_main else (lambda ax, x, y, **kw: ax.plot(x, y, **kw))
        ln = plot_fn(ax_err, ts, err_max, color=col, ls=ls, lw=lw, label=lab)[0]
        plot_fn(ax_lam, t_tr, lam2, color=col, ls=ls, lw=lw)
        plot_fn(ax_dist, t_tr, mdist, color=col, ls=ls, lw=lw)
        plot_fn(ax_vel, t_tr, u_rms, color=col, ls=ls, lw=lw)
        handles.append(ln)
        labels.append(lab)

    ax_err.axhline(0.01, color=COLORS["reference"], ls="--", lw=1.1)
    ax_err.set_yscale("log")
    ax_err.set_xlim(0, 60)
    ax_err.set_ylim(2e-3, 1.5)

    row0 = json.loads((G / "e2_full_env_B8" / "rows" / f"{sc_e2}_rl_full_env_B8.json").read_text(encoding="utf8"))
    lam_req = row0["lambda_required"]
    ax_lam.axhline(lam_req, color=COLORS["reference"], ls="--", lw=1.1)
    ax_lam.set_xlim(0, 60)
    ax_lam.set_ylim(lam_req - 0.5, 16.8)

    ax_dist.axhline(1.0, color=COLORS["reference"], ls="--", lw=1.1)
    ax_dist.set_xlim(0, 60)
    ax_dist.set_ylim(0.992, 1.035)

    ax_vel.set_xlim(0, 60)
    ax_vel.set_ylim(0, 0.22)

    if lang == "zh":
        halo_text(ax_err, 58.0, 0.0125, r"达标阈值 $0.01\ \mathrm{m}$", ha="right", va="bottom", fontsize=8.2)
        panel(ax_err, "a", "全网最大位置误差收敛响应")
        ax_err.set_ylabel(r"最大位置误差 $\max_i\|\boldsymbol{e}_i\|_2$ ($\mathrm{m}$)")
        halo_text(ax_lam, 58.0, lam_req + 0.25, r"安全下界 $\lambda_{\mathrm{req}}$", ha="right", va="bottom", fontsize=8.2)
        panel(ax_lam, "b", "代数连通度安全保持响应")
        ax_lam.set_ylabel(r"代数连通度 $\lambda_2(\boldsymbol{L}_{\mathrm{p}})$")
        halo_text(ax_dist, 58.0, 0.9965, r"安全距离 $d_{\mathrm{safe}}=1.0\ \mathrm{m}$", ha="right", va="top", fontsize=8.2)
        panel(ax_dist, "c", "最小机间几何距离演化")
        ax_dist.set_xlabel(r"时间 $t$ ($\mathrm{s}$)")
        ax_dist.set_ylabel(r"最小距离 $\min_{i<j}\ d_{ij}$ ($\mathrm{m}$)")
        panel(ax_vel, "d", "全网均方根控制速度响应")
        ax_vel.set_xlabel(r"时间 $t$ ($\mathrm{s}$)")
        ax_vel.set_ylabel(r"均方根速度 $\|\boldsymbol{u}\|_2/\sqrt{N}$ ($\mathrm{m/s}$)")
    else:
        halo_text(ax_err, 58.0, 0.0125, r"Tolerance $0.01\ \mathrm{m}$", ha="right", va="bottom", fontsize=8.2)
        panel(ax_err, "a", "Maximum Position Error Convergence")
        ax_err.set_ylabel(r"Max Error $\max_i\|\boldsymbol{e}_i\|_2$ ($\mathrm{m}$)")
        halo_text(ax_lam, 58.0, lam_req + 0.25, r"Threshold $\lambda_{\mathrm{req}}$", ha="right", va="bottom", fontsize=8.2)
        panel(ax_lam, "b", "Algebraic Connectivity Maintenance")
        ax_lam.set_ylabel(r"Connectivity $\lambda_2(\boldsymbol{L}_{\mathrm{p}})$")
        halo_text(ax_dist, 58.0, 0.9965, r"Safety Bound $d_{\mathrm{safe}}=1.0\ \mathrm{m}$", ha="right", va="top", fontsize=8.2)
        panel(ax_dist, "c", "Minimum Inter-Robot Distance")
        ax_dist.set_xlabel(r"Time $t$ ($\mathrm{s}$)")
        ax_dist.set_ylabel(r"Min Distance $\min_{i<j}\ d_{ij}$ ($\mathrm{m}$)")
        panel(ax_vel, "d", "Team RMS Control Speed")
        ax_vel.set_xlabel(r"Time $t$ ($\mathrm{s}$)")
        ax_vel.set_ylabel(r"RMS Speed $\|\boldsymbol{u}\|_2/\sqrt{N}$ ($\mathrm{m/s}$)")

    for ax in axes.flat:
        grid(ax)
    shared_top_legend(fig, handles, labels, ncol=5)
    save_figure(fig, out_dir / "fig4_time_evolution", layout="P4")
    plt.close(fig)


def fig5_perm_and_gate(lang: str = "zh") -> None:
    """1x2 P2E figure: (a) 20-robot random permutation trajectories; (b) fixed-weight gate & feature knockout."""
    apply_style()
    out_dir = OUT_ZH if lang == "zh" else OUT_EN
    fig, (ax1, ax2) = new_figure("P2E", geometry="left")

    sc_perm = "H20_perm1_s20261105"
    z_p = np.load(FAIR / "runs" / f"{sc_perm}_rl_full_env_B8.npz")
    st_p, goal_p = z_p["states"], z_p["goal"]
    p0_p, pend_p = st_p[0], st_p[-1]
    n_p = len(p0_p)
    draw_comm_edges(ax1, pend_p, r_comm=1.55)
    disp_tot = np.linalg.norm(pend_p - p0_p, axis=1)
    top3 = list(np.argsort(-disp_tot)[:3])
    hi_colors = [COLORS["method"], COLORS["comparison"], COLORS["third"]]
    bg_idx = [i for i in range(n_p) if i not in top3]
    for i in bg_idx:
        ax1.plot(st_p[:, i, 0], st_p[:, i, 1], color="#64748B", lw=0.70, alpha=0.30, zorder=2.6)
        ax1.plot(pend_p[i, 0], pend_p[i, 1], marker="o", ms=3.8, mfc="#94A3B8", mec="white", mew=0.9, zorder=4.0)
    for rank, i in enumerate(top3):
        c = hi_colors[rank]
        halo_line(ax1, st_p[:, i, 0], st_p[:, i, 1], color=c, lw=1.75, zorder=3.8)
        add_trajectory_arrows(ax1, st_p[:, i, :], color=c, fractions=(0.48,), size=8.0, zorder=4.1)
    draw_robot_nodes(
        ax1, start_xy=p0_p[top3], end_xy=pend_p[top3], goal_xy=goal_p[top3],
        colors=hi_colors, ms_start=4.4, ms_end=5.2, ms_goal=4.8,
    )
    lim_p = 3.1
    ax1.set_xlim(-lim_p, lim_p)
    ax1.set_ylim(-lim_p, lim_p)
    ax1.set_aspect("equal")

    sc_g = "H12_local0_s20261105"
    if lang == "zh":
        g_specs = [
            ("EGP-NPA (影子价格门控)", FAIR, f"{sc_g}_rl_full_env_B8", COLORS["method"], "-", 1.75, True),
            ("单包络常开门控 ($\\gamma_{i,k}^{\\mathrm{gate}}\\equiv 1$)", SWAP, f"{sc_g}_wfull_gone_B8", COLORS["third"], "--", 1.4, False),
            ("滤波修正量门控替换", SWAP, f"{sc_g}_wfull_gint_B8", COLORS["fifth"], "-.", 1.35, False),
            ("敲除内部代数特征", STEP0, f"{sc_g}_alg_zero_B8", COLORS["comparison"], ":", 1.5, False),
        ]
    else:
        g_specs = [
            ("EGP-NPA (Shadow-Price Gate)", FAIR, f"{sc_g}_rl_full_env_B8", COLORS["method"], "-", 1.75, True),
            ("Envelope-Only ($\\gamma_{i,k}^{\\mathrm{gate}}\\equiv 1$)", SWAP, f"{sc_g}_wfull_gone_B8", COLORS["third"], "--", 1.4, False),
            ("Intervention Gate Swap", SWAP, f"{sc_g}_wfull_gint_B8", COLORS["fifth"], "-.", 1.35, False),
            ("Algebraic Feature Knockout", STEP0, f"{sc_g}_alg_zero_B8", COLORS["comparison"], ":", 1.5, False),
        ]
    for lab, d, stem, col, ls, lw, is_main in g_specs:
        z = np.load(d / "runs" / f"{stem}.npz")
        e = np.linalg.norm(z["states"] - z["goal"][None], axis=2).max(axis=1)
        t = np.array(z["times"])
        if is_main:
            halo_line(ax2, t, e, color=col, ls=ls, lw=lw, label=lab)
        else:
            ax2.plot(t, e, color=col, ls=ls, lw=lw, label=lab)
    ax2.axhline(0.01, color=COLORS["reference"], ls="--", lw=1.1)
    ax2.set_yscale("log")
    ax2.set_xlim(0, 14)
    ax2.set_ylim(1.5e-3, 1.5)
    ax2.legend(loc="upper right", fontsize=8.0)

    if lang == "zh":
        panel(ax1, "a", "20 机器人全排列穿插轨迹 (EGP-NPA)")
        ax1.set_xlabel(r"水平位置 $x$ ($\mathrm{m}$)")
        ax1.set_ylabel(r"纵向位置 $y$ ($\mathrm{m}$)")
        halo_text(ax2, 13.5, 0.0125, r"达标阈值 $0.01\ \mathrm{m}$", ha="right", va="bottom", fontsize=8.0)
        panel(ax2, "b", "门控替换与代数特征敲除响应")
        ax2.set_xlabel(r"时间 $t$ ($\mathrm{s}$)")
        ax2.set_ylabel(r"最大位置误差 $\max_i\|\boldsymbol{e}_i\|_2$ ($\mathrm{m}$)")
    else:
        panel(ax1, "a", "20-Robot Full Permutation (EGP-NPA)")
        ax1.set_xlabel(r"Horizontal Position $x$ ($\mathrm{m}$)")
        ax1.set_ylabel(r"Vertical Position $y$ ($\mathrm{m}$)")
        halo_text(ax2, 13.5, 0.0125, r"Tolerance $0.01\ \mathrm{m}$", ha="right", va="bottom", fontsize=8.0)
        panel(ax2, "b", "Gate Swap & Algebraic Knockout")
        ax2.set_xlabel(r"Time $t$ ($\mathrm{s}$)")
        ax2.set_ylabel(r"Max Error $\max_i\|\boldsymbol{e}_i\|_2$ ($\mathrm{m}$)")
    grid(ax1)
    grid(ax2)
    save_figure(fig, out_dir / "fig5_perm_and_gate", layout="P2E")
    plt.close(fig)


def fig6_mujoco_verification(lang: str = "zh") -> None:
    """2x2 P4 figure: Zero-shot virtual-physical transfer in MuJoCo with online division-free spectral estimation."""
    apply_style()
    out_dir = OUT_ZH if lang == "zh" else OUT_EN
    fig, axs = new_figure("P4")
    M, C, T3 = COLORS["method"], COLORS["comparison"], COLORS["third"]
    REF = COLORS["reference"]

    if lang == "zh":
        arms = [
            ("gnn_online_phys", "EGP-NPA + 在线谱估计 + 物理接口", M, "-", 1.75),
            ("gnn_static_phys", "EGP-NPA + 静态平摊 ($B=0$)", T3, "-.", 1.35),
            ("gnn_ideal_npa", "EGP-NPA + 无补偿单积分器滤波", C, "--", 1.35),
            ("myopic_phys_npa", "短视比例吸引 + 物理接口", COLORS["edge"], ":", 1.35),
        ]
    else:
        arms = [
            ("gnn_online_phys", "EGP-NPA + Online Est. + Phys-NPA", M, "-", 1.75),
            ("gnn_static_phys", "EGP-NPA + Static Split ($B=0$)", T3, "-.", 1.35),
            ("gnn_ideal_npa", "EGP-NPA + Ideal Single-Integrator", C, "--", 1.35),
            ("myopic_phys_npa", "Myopic Attraction + Phys-NPA", COLORS["edge"], ":", 1.35),
        ]

    handles, labels = [], []
    for ax, case, tag, title_zh, title_en in [
        (axs[0, 0], "H8_local0_f60_s20261103", "a", r"$N=8$ 二阶底盘局部交换误差", r"$N=8$ Chassis Local Swap Error"),
        (axs[0, 1], "H12_local0_s20261017", "b", r"$N=12$ 零样本跨规模交换误差", r"$N=12$ Zero-Shot Cross-Scale Error"),
    ]:
        for arm, label, col, ls, lw in arms:
            tr = np.load(V6_A / f"{case}_{arm}.npz")["trace"]
            if arm == "gnn_online_phys":
                ln = halo_line(ax, tr[:, 0], tr[:, 4], label=label, color=col, ls=ls, lw=lw, zorder=4)[0]
            else:
                ln = ax.plot(tr[:, 0], tr[:, 4], label=label, color=col, ls=ls, lw=lw, zorder=3)[0]
            if tag == "a":
                handles.append(ln)
                labels.append(label)
        ax.axhline(0.025, color=REF, ls="--", lw=1.05)
        ax.set_ylabel(r"最大目标误差 ($\mathrm{m}$)" if lang == "zh" else r"Max Goal Error ($\mathrm{m}$)")
        panel(ax, tag, title_zh if lang == "zh" else title_en)
        grid(ax)

    ax = axs[1, 0]
    tr8 = np.load(V6_A / "H8_local0_f60_s20261103_gnn_online_phys.npz")["trace"]
    tr12 = np.load(V6_A / "H12_local0_s20261017_gnn_online_phys.npz")["trace"]
    ax.plot(tr8[:, 0], tr8[:, 1], color=M, lw=1.75, label=r"$N=8$ 真值" if lang == "zh" else r"$N=8$ True $\lambda_2$")
    ax.plot(tr8[:, 0], tr8[:, 2], color=M, ls="--", lw=1.2, label=r"$N=8$ 在线估计" if lang == "zh" else r"$N=8$ Online $\hat{\lambda}_2$")
    ax.plot(tr12[:, 0], tr12[:, 1], color=C, lw=1.75, label=r"$N=12$ 真值" if lang == "zh" else r"$N=12$ True $\lambda_2$")
    ax.plot(tr12[:, 0], tr12[:, 2], color=C, ls="--", lw=1.2, label=r"$N=12$ 在线估计" if lang == "zh" else r"$N=12$ Online $\hat{\lambda}_2$")
    ax.set_xlabel(r"物理时间 $t$ ($\mathrm{s}$)" if lang == "zh" else r"Physical Time $t$ ($\mathrm{s}$)")
    ax.set_ylabel(r"代数连通度 $\lambda_2$" if lang == "zh" else r"Connectivity $\lambda_2$")
    ax.legend(loc="center", ncol=2, fontsize=8.0)
    panel(ax, "c", "在线无除法谱估计实时跟踪" if lang == "zh" else "Online Division-Free Spectral Tracking")
    grid(ax)

    ax = axs[1, 1]
    tr_id = np.load(V6_B / "H8_local0_f60_s20261103_gnn_ideal_npa.npz")["trace"]
    tr_v2 = np.load(V6_A / "H8_local0_f60_s20261103_gnn_online_phys.npz")["trace"]
    tr_v3 = np.load(V6_B / "H8_local0_f60_s20261103_gnn_online_phys.npz")["trace"]
    if lang == "zh":
        ax.plot(tr_id[:, 0], tr_id[:, 3], color=C, ls="--", lw=1.35, label="无物理补偿 (违约 4764 步)")
        ax.plot(tr_v2[:, 0], tr_v2[:, 3], color=T3, ls="-.", lw=1.35, label="前瞻补偿档 (7.71 s 达标)")
        halo_line(ax, tr_v3[:, 0], tr_v3[:, 3], color=M, ls="-", lw=1.75, label="保守内缩档 (0 违约，通道闭合)")
    else:
        ax.plot(tr_id[:, 0], tr_id[:, 3], color=C, ls="--", lw=1.35, label="Ideal-NPA (4764 Viol. Steps)")
        ax.plot(tr_v2[:, 0], tr_v2[:, 3], color=T3, ls="-.", lw=1.35, label="Regime A (Arrived 7.71 s)")
        halo_line(ax, tr_v3[:, 0], tr_v3[:, 3], color=M, ls="-", lw=1.75, label="Regime B (0 Viol., Pinched)")
    ax.axhline(1.000, color=REF, ls=":", lw=1.15)
    ax.set_xlabel(r"物理时间 $t$ ($\mathrm{s}$)" if lang == "zh" else r"Physical Time $t$ ($\mathrm{s}$)")
    ax.set_ylabel(r"最小距离 $\min\ d_{ij}$ ($\mathrm{m}$)" if lang == "zh" else r"Min Distance $\min\ d_{ij}$ ($\mathrm{m}$)")
    ax.set_ylim(0.975, 1.10)
    ax.legend(loc="upper right", fontsize=7.9)
    panel(ax, "d", "安全裕度内缩与通道可通行性权衡" if lang == "zh" else "Margin Shrinkage vs. Passability")
    grid(ax)

    shared_top_legend(fig, handles, labels, ncol=2)
    save_figure(fig, out_dir / "fig6_mujoco_verification", layout="P4")
    plt.close(fig)


if __name__ == "__main__":
    for language in ("zh", "en"):
        fig1_seven_geometry(language)
        fig2_training_curves(language)
        fig3_swap_trajectories(language)
        fig4_time_evolution(language)
        fig5_perm_and_gate(language)
        fig6_mujoco_verification(language)
    print("Successfully generated all 6 ZH figures and 6 EN figures.")

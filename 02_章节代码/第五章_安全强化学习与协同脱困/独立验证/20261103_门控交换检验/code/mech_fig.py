"""Mechanism figures from saved per-sample traces (no simulation). Writes to results/figs01.

A: H20_local1_s20261105, the three learned variants of 20261103_包络公平检验/run01 (each with its training gate).
B: same scene, columns = weights, series = gate at execution (needs results/run01 of this directory)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
V = HERE.parents[1]
FAIR = V / '20261103_包络公平检验' / 'results' / 'run01'
SWAP = HERE.parent / 'results' / 'run01'
OUT = HERE.parent / 'results' / 'figs01'
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a']


def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def series(d, scene, tag):
    tr = json.loads((d / 'runs' / f'{scene}_{tag}_trace.json').read_text(encoding='utf8'))
    z = np.load(d / 'runs' / f'{scene}_{tag}.npz')
    st, goal = z['states'], z['goal']
    n = st.shape[1]
    stat = np.linalg.norm(st[0] - goal, axis=1) < 1e-6
    err = np.linalg.norm(st[:-1] - goal[None], axis=2)              # state at the start of each traced sample
    row = json.loads((d / 'rows' / f'{scene}_{tag}.json').read_text(encoding='utf8'))
    return dict(t=np.array([x['t'] for x in tr]), gate=np.array([x['gate_mean'] for x in tr]),
                res=np.array([x['residual_norm'] for x in tr]) / np.sqrt(n),
                u=np.array([x['u_norm'] for x in tr]) / np.sqrt(n),
                err=err.max(axis=1), moved=(err[:, stat] > 0.025).sum(axis=1), n_stat=int(stat.sum()), row=row)


MEASURES = [('gate', '门控均值（乘包络前）', '', (0, 1.05)),
            ('res', '策略残差经门控后的大小（每台 RMS，未乘包络）', 'm/s', None),
            ('u', '滤波器 L 输出、实际执行的速度（每台 RMS）', 'm/s', None),
            ('moved', '目标在原地、当前偏离目标 > 2.5 cm 的台数（共 18 台）', '台', None),
            ('err', '最大目标误差', 'm', (0, 1.15))]


def outcome(r):
    return f"到达 {r['arrival']:.2f} s" if r['status'] == 'arrived' else f"{r['duration']:.0f} s 未到达"


def fig_a(scene, tmax=20.0):
    runs = [('rl_full + 包络（μ 门控）', 'rl_full_env_B8'), ('rl_geom + 包络（修正量门控）', 'rl_geom_env_B8'),
            ('rl_full_int + 包络（修正量门控）', 'rl_full_int_env_B8')]
    data = [(lab, series(FAIR, scene, tag)) for lab, tag in runs]
    fig, axes = plt.subplots(len(MEASURES), 1, figsize=(9.2, 11.5), sharex=True, facecolor=SURF)
    for ax, (key, title, unit, ylim) in zip(axes, MEASURES):
        style(ax)
        for (lab, s), col in zip(data, SLOTS):
            m = s['t'] <= tmax
            ax.plot(s['t'][m], s[key][m], color=col, lw=1.8)
        ax.set_title(title, loc='left', color=INK, fontsize=9.5, pad=4)
        ax.set_ylabel(unit, color=INK2, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        if key == 'moved':
            ax.set_ylim(0, data[0][1]['n_stat'] + 1)
    axes[-1].axhline(0.01, color=INK2, lw=1, ls=(0, (4, 3)))
    axes[-1].text(tmax * 0.995, 0.04, '到达阈值 1 cm', ha='right', va='bottom', fontsize=8, color=INK2)
    axes[-1].set_xlabel('时间 t（s）', color=INK2)
    axes[-1].set_xlim(0, tmax)
    handles = [plt.Line2D([], [], color=c, lw=2.2) for c in SLOTS]
    fig.legend(handles, [f'{lab}：{outcome(s["row"])}' for lab, s in data], loc='upper left', ncol=1, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.05, 0.975))
    fig.suptitle(f'{scene}：三个学习检查点（各用训练时的门控）在同一场景的逐步记录', x=0.012, y=0.995, ha='left',
                 color=INK, fontsize=11.5, fontweight='bold')
    fig.text(0.012, 0.006, '数据：20261103_包络公平检验/results/run01 的逐步 trace，未重新仿真。门控与残差是 trace 记录的乘包络前数值；'
             '执行速度是滤波器 L 的输出。图只显示前 20 s，两个修正量门控的运行一直持续到 60 s。', fontsize=8.2, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.915))
    fig.savefig(OUT / f'mech_{scene.split("_s")[0]}_checkpoints.png', dpi=140, facecolor=SURF)
    plt.close(fig)


def fig_b(scene, tmax=20.0):
    cols = [('rl_full 权重', [('μ（训练时）', FAIR, 'rl_full_env_B8'), ('修正量 int', SWAP, 'wfull_gint_B8'),
                            ('恒为 1', SWAP, 'wfull_gone_B8')]),
            ('rl_full_int 权重', [('μ', SWAP, 'wfullint_gmu_B8'), ('修正量 int（训练时）', FAIR, 'rl_full_int_env_B8')]),
            ('rl_geom 权重', [('μ', SWAP, 'wgeom_gmu_B8'), ('修正量 int（训练时）', FAIR, 'rl_geom_env_B8')])]
    color = {'μ': SLOTS[0], '修正': SLOTS[1], '恒为': SLOTS[2]}
    rows = [('u', '执行速度，每台 RMS（m/s）'), ('moved', '原地机器人偏离 > 2.5 cm 的台数'), ('err', '最大目标误差（m）')]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(14, 8.4), sharex=True, sharey='row', facecolor=SURF)
    for ci, (title, runs) in enumerate(cols):
        for ri, (key, ylab) in enumerate(rows):
            ax = axes[ri, ci]
            style(ax)
            for lab, d, tag in runs:
                s = series(d, scene, tag)
                m = s['t'] <= tmax
                ax.plot(s['t'][m], s[key][m], color=color[lab[:2] if lab[0] != 'μ' else 'μ'], lw=1.8,
                        label=f'{lab}：{outcome(s["row"])}')
            if ci == 0:
                ax.set_ylabel(ylab, color=INK2, fontsize=9)
            if ri == 0:
                ax.set_title(title, loc='left', color=INK, fontsize=10.5)
                ax.legend(loc='upper right', frameon=False, fontsize=8.3, labelcolor=INK)
            if ri == len(rows) - 1:
                ax.set_xlabel('时间 t（s）', color=INK2)
                ax.axhline(0.01, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.set_xlim(0, tmax)
    fig.suptitle(f'{scene}：同一权重只换执行时的门控（颜色 = 门控：蓝 μ、橙 修正量、绿 恒为 1）', x=0.012, y=0.995,
                 ha='left', color=INK, fontsize=11.5, fontweight='bold')
    fig.text(0.012, 0.008, '数据：本目录 results/run01 与 20261103_包络公平检验/results/run01 的逐步记录，全部带终段包络、B=8。'
             '只画前 20 s。', fontsize=8.3, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT / f'mech_{scene.split("_s")[0]}_gate_swap.png', dpi=140, facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    OUT.mkdir(parents=True, exist_ok=True)
    fig_a('H20_local1_s20261105')
    if (SWAP / 'rows' / 'H20_local1_s20261105_wfull_gint_B8.json').exists():
        for sc in ('H20_local1_s20261105', 'H25_local1_s20261105'):
            fig_b(sc)
    print('written', OUT)

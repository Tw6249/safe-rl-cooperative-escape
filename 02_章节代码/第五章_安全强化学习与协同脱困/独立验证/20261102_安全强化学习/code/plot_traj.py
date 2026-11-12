"""Trajectory small multiples for one task (README §8); reads saved states only, no new simulation.

    D:\\miniconda\\python.exe -X utf8 plot_traj.py --runs ../results/<d1>,../results/<d2> --scene H8_local3_f60_s20261103
Robots whose goal differs from their start are drawn with reference categorical slots 1-3 (validated all-pairs, light
mode); robots that stay are muted grey. Use a task with at most three moving robots."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a']
NAMES = {'myopic_B8': '短视 + L', 'heur_B8': '环流启发式 + L', 'heur_st_B0': '环流启发式 + 静态分摊',
         'rl_full_B8': 'RL-Full + L', 'rl_geom_B8': 'RL-Geom + L', 'rl_static_B0': 'RL-Static + 静态分摊',
         'rl_full_B2': 'RL-Full + L (B=2)', 'rl_full_int_B8': 'RL-full_int + L'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--order', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    order = a.order.split(',')
    found = {}
    for d in a.runs.split(','):
        for tag in order:
            f = Path(d) / 'runs' / f'{a.scene}_{tag}.npz'
            if f.exists():
                found[tag] = (np.load(f), json.loads((Path(d) / 'rows' / f'{a.scene}_{tag}.json').read_text(encoding='utf8')))
    tags = [t for t in order if t in found]
    fig, axes = plt.subplots(1, len(tags), figsize=(3.1 * len(tags), 3.4), facecolor=SURF)
    axes = np.atleast_1d(axes)
    z0 = found[tags[0]][0]
    p0, goal = z0['states'][0], z0['goal']
    movers = [i for i in range(len(p0)) if np.linalg.norm(p0[i] - goal[i]) > 1e-9]
    assert len(movers) <= 3, f'{len(movers)} moving robots; pick a task with at most three'
    lim = np.abs(np.concatenate([found[t][0]['states'].reshape(-1, 2) for t in tags])).max() + 0.7
    for ax, tag in zip(axes, tags):
        z, row = found[tag]
        st = z['states']
        ax.set_facecolor(SURF)
        for i in range(st.shape[1]):
            c = SLOTS[movers.index(i)] if i in movers else MUTED
            ax.plot(st[:, i, 0], st[:, i, 1], color=c, lw=2 if i in movers else 1.2, zorder=3 if i in movers else 2)
            ax.add_patch(Circle(st[-1, i], 0.5, facecolor='none', edgecolor=c, lw=1.0, zorder=4))
            if i in movers:
                ax.plot(*goal[i], marker='x', color=c, ms=7, mew=2, zorder=5)
        status = row.get('status')
        sub = f"到达 {row['arrival']:.1f} s" if status == 'arrived' else {'horizon_reached': '60 s 未到达', 'stalled': '停滞',
                                                                          'certification_stop': '证书停止',
                                                                          'step_budget': '步数上限'}.get(status, status)
        ax.set_title(f'{NAMES.get(tag, tag)}\n{sub}', color=INK, fontsize=9.5, loc='left')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.01, 0.01, '彩色：需要移动的机器人（× 为其目标）；灰色：原地任务；圆为终态位置（半径 0.5 m，即安全距离的一半）',
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'traj_{a.scene}.png', dpi=200, facecolor=SURF, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(out / f'traj_{a.scene}.pdf', facecolor=SURF, bbox_inches='tight', pad_inches=0.15)
    print('saved', out / f'traj_{a.scene}.png')


if __name__ == '__main__':
    main()

"""Learning curves (README §8) from train_log.jsonl of the training runs; no new simulation.

    D:\\miniconda\\python.exe -X utf8 plot_training.py --runs full=../results/<d1>,geom=../results/<d2>,static=../results/<d3>
        --out ../results/<fig-dir>
Palette: reference categorical slots 1-3 (validated all-pairs, light mode); aqua is below 3:1 on the surface, so every
line is direct-labelled at its end and the numbers are in curves.json (table view)."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
COL = {'full': '#2a78d6', 'geom': '#eb6834', 'full_int': '#1baf7a'}      # slot 3 reassigned: RL-Static was dropped (README §8.4)
LABEL = {'full': 'Full（几何+代数，μ 门控）', 'geom': 'Geom（仅几何，修正量门控）', 'full_int': 'full_int（几何+代数，修正量门控）'}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smooth', type=int, default=5)
    a = ap.parse_args()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    runs = dict(kv.split('=', 1) for kv in a.runs.split(','))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), facecolor=SURF)
    table = {}
    for v, d in runs.items():
        allrows = [json.loads(x) for x in (Path(d) / 'train_log.jsonl').read_text(encoding='utf8').splitlines() if x]
        rows = [r for r in allrows if 'success' in r]
        steps = np.array([r['env_steps'] for r in rows]) / 1e3
        succ = np.array([r['success'] for r in rows])
        k = min(a.smooth, len(succ))
        sm = np.convolve(succ, np.ones(k) / k, mode='valid')
        vs = [(r['env_steps'] / 1e3, r['val']['success']) for r in allrows if 'val' in r]
        table[v] = dict(env_steps_k=steps.tolist(), train_success=succ.tolist(),
                        val=[dict(env_steps_k=x, success=y) for x, y in vs])
        # all runs end at the same value (1.0), so end-of-line direct labels would collide; the legend carries identity
        axes[0].plot(steps[k - 1:], sm, color=COL[v], lw=2, label=LABEL[v])
        if vs:
            x, y = zip(*vs)
            axes[1].plot(x, y, color=COL[v], lw=2, marker='o', ms=6, mec=SURF, mew=1.5, label=LABEL[v])
    for ax, t in zip(axes, ['(a) 训练回合成功率（{} 轮滑动平均）'.format(a.smooth), '(b) 验证集成功率（确定性策略，36 个场景）']):
        style(ax)
        ax.set_title(t, color=INK, fontsize=10, loc='left')
        ax.set_xlabel('环境步数（千步）', color=INK2, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower right')
    axes[0].set_ylabel('成功率', color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'learning_curves.png', dpi=200, facecolor=SURF)
    fig.savefig(out / 'learning_curves.pdf', facecolor=SURF)
    (out / 'curves.json').write_text(json.dumps(table, indent=1), encoding='utf8')
    print('saved', out)


if __name__ == '__main__':
    main()

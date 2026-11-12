"""Arrival-time strip plot per group (README §8) from evaluate.py rows; no new simulation.

    D:\\miniconda\\python.exe -X utf8 plot_results.py --runs ../results/<d1>,../results/<d2> --out ../results/<fig-dir>
One dot per task; tasks that did not arrive within the horizon are drawn in the right-hand column '未到达' in muted ink.
Groups are identified by the axis labels, so a single series colour (reference slot 1) is used; the numbers are in
the README tables (table view)."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURF, INK, INK2, MUTED, GRID, AXIS, C1 = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#2a78d6'
NAMES = {'myopic_B8': '短视 + L', 'heur_B8': '环流启发式 + L', 'heur_st_B0': '环流启发式 + 静态分摊',
         'rl_full_B8': 'RL-Full + L', 'rl_geom_B8': 'RL-Geom + L', 'rl_static_B0': 'RL-Static + 静态分摊',
         'rl_full_B2': 'RL-Full + L (B=2)', 'myopic_B2': '短视 + L (B=2)', 'heur_B2': '环流启发式 + L (B=2)',
         'rl_full_int_B8': 'RL-full_int + L'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--order', required=True)
    ap.add_argument('--frac', type=float, default=None)
    ap.add_argument('--horizon', type=float, default=60.0)
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    rows = []
    for d in a.runs.split(','):
        rows += [json.loads(f.read_text(encoding='utf8')) for f in sorted((Path(d) / 'rows').glob('*.json'))]
    if a.frac is not None:
        rows = [r for r in rows if r.get('frac') == a.frac]
    order = a.order.split(',')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 0.45 * len(order) + 1.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    rng = np.random.default_rng(0)
    xfail = a.horizon + 6.0
    table = {}
    for k, tag in enumerate(order[::-1]):
        sel = [r for r in rows if r['job']['tag'] == tag]
        arr = np.array([r['arrival'] for r in sel if r.get('status') == 'arrived'])
        nf = len(sel) - len(arr)
        jit = rng.uniform(-0.18, 0.18, len(arr))
        ax.scatter(arr, k + jit, s=22, color=C1, edgecolors=SURF, linewidths=1.0, zorder=3)
        if nf:
            ax.scatter(np.full(nf, xfail) + rng.uniform(-1.2, 1.2, nf), k + rng.uniform(-0.18, 0.18, nf), s=22,
                       color=MUTED, edgecolors=SURF, linewidths=1.0, zorder=3)
        med = float(np.median(arr)) if len(arr) else None
        if med is not None:
            ax.plot([med, med], [k - 0.3, k + 0.3], color=INK, lw=2, zorder=4)
        ax.annotate(f'{len(arr)}/{len(sel)}', (a.horizon + 13.5, k), color=INK2, fontsize=9, va='center')
        table[tag] = dict(arrived=int(len(arr)), total=len(sel), median=med)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAMES.get(t, t) for t in order[::-1]], color=INK2, fontsize=9)
    ax.set_xlim(0, a.horizon + 18)
    ax.set_xticks(list(range(0, int(a.horizon) + 1, 10)) + [xfail])
    ax.set_xticklabels([str(x) for x in range(0, int(a.horizon) + 1, 10)] + ['未到达'], color=INK2, fontsize=9)
    ax.set_xlabel('到达时间 (s)；竖线为到达者中位数；右侧为到达数', color=INK2, fontsize=9)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', colors=MUTED)
    ax.grid(True, axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if a.title:
        ax.set_title(a.title, color=INK, fontsize=10, loc='left')
    fig.tight_layout()
    stem = 'arrival_strip' + (f'_f{int(round(100 * a.frac))}' if a.frac is not None else '')
    fig.savefig(out / f'{stem}.png', dpi=200, facecolor=SURF)
    fig.savefig(out / f'{stem}.pdf', facecolor=SURF)
    (out / f'{stem}.json').write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding='utf8')
    print('saved', out / stem)


if __name__ == '__main__':
    main()

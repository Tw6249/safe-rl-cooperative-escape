"""Cross-scale E2 report.
    D:\\miniconda\\python.exe -X utf8 e2_report.py --runs ../results/test_e2_B8,../results/test_e2_B8_fullint
        --reference-l ../../20261029_边账本结清/results/run01 --out ../results/test_e2_B8_report
Writes by_size.md / by_size.json (per group x N x task kind; per-N paired comparisons, exact binomial on discordant pairs and
time differences on tasks both arrived; end state of every run that did not arrive, read from the saved trajectory) and
fig_e2_strip.png (one panel per N, one dot per task; non-arrivals in muted ink at the right; groups are named by the axis
labels so a single series colour is used)."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_report import load, load_reference, paired, fmt
from diag_traces import settle_time

SURF, INK, INK2, MUTED, GRID, AXIS, C1 = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#2a78d6'
NAMES = {'rl_full_B8': 'RL-Full + L', 'rl_geom_B8': 'RL-Geom + L',
         'rl_full_int_B8': 'RL-full_int + L', 'heur_B8': '环流启发式 + L', 'myopic_B8': '短视 + L'}
KIND = {'local': '局部交换', 'perm': '置换'}


def end_state(run_dir, scene, tag):
    """Final and closest max_i|p_i-g_i| of a run and the time after which it stays within 5 cm (None if never)."""
    z = np.load(Path(run_dir) / 'runs' / f'{scene}_{tag}.npz')
    st, goal, times = z['states'], z['goal'], z['times']
    m = min(len(st), len(times))
    err = np.linalg.norm(st[:m] - goal[None], axis=2).max(axis=1)
    k = int(err.argmin())
    return dict(final_max_err=float(err[-1]), min_max_err=float(err[k]), t_min=float(times[k]),
                t_5cm=settle_time(st[:m], goal, times[:m], 0.05))


def status_str(sel):
    st = {}
    for r in sel:
        st[r.get('status')] = st.get(r.get('status'), 0) + 1
    return ', '.join(f'{k} {v}' for k, v in sorted(st.items(), key=lambda x: str(x[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-l', default=None)
    ap.add_argument('--pairs', default='')
    ap.add_argument('--horizon', type=float, default=60.0)
    a = ap.parse_args()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    rows = []
    for d in a.runs.split(','):
        for r in load([d]):
            r['_dir'] = d
            rows.append(r)
    if a.reference_l:
        names = {r['scene'] for r in rows}
        rows += [r for r in load_reference(a.reference_l) if r['scene'] in names]
    for r in rows:
        r['_kind'] = 'perm' if 'perm' in r['scene'] else 'local'
    sizes = sorted({r['n'] for r in rows})
    order = [t for t in a.order.split(',') if any(r['_tag'] == t for r in rows)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. per group x N x kind
    md = ['# E2 分规模汇总', '',
          '| 组 | N | 任务 | 场景 | 到达 | 状态 | 到达时间中位数 (s) | 到达时间 (s) | 冻结机器人拍占比中位数 | 审计失败段 | 切步 | 最小间距 | min(λ₂下界−λ_req) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    groups = {}
    for t in order:
        for n in sizes:
            for k in ('local', 'perm'):
                sel = [r for r in rows if r['_tag'] == t and r['n'] == n and r['_kind'] == k]
                if not sel:
                    continue
                arr = [r['arrival'] for r in sel if r.get('status') == 'arrived']
                fr = [r.get('frozen_robot_frac') for r in sel if r.get('frozen_robot_frac') is not None]
                md_ = min((r.get('audit_min_distance') or r.get('min_distance')) for r in sel)
                sl = min(((r.get('audit_min_lambda2') or r.get('min_lambda2_lower')) - r['lambda_required']) for r in sel)
                g = dict(n_scenes=len(sel), arrived=len(arr), statuses=status_str(sel),
                         arrival_median=float(np.median(arr)) if arr else None, arrivals=sorted(round(x, 2) for x in arr),
                         frozen_median=float(np.median(fr)) if fr else None,
                         audit_failed=sum(r.get('audit_failed') or 0 for r in sel), cuts=sum(r.get('step_cuts') or 0 for r in sel),
                         min_distance=md_, min_slack=sl)
                groups[f'{t}|{n}|{k}'] = g
                md.append(f"| {t} | {n} | {KIND[k]} | {len(sel)} | {len(arr)} | {g['statuses']} | {fmt(g['arrival_median'])} | "
                          f"{', '.join(f'{x:.2f}' for x in g['arrivals']) or '—'} | {fmt(g['frozen_median'], 3)} | {g['audit_failed']} | "
                          f"{g['cuts']} | {md_:.9f} | {sl:.4f} |")
    md += ['', '## 各组按规模的到达数', '', '| 组 | ' + ' | '.join(f'N={n}' for n in sizes) + ' | 合计 |', '|---|' + '---|' * (len(sizes) + 1)]
    totals = {}
    for t in order:
        cells, tot = [], [0, 0]
        for n in sizes:
            sel = [r for r in rows if r['_tag'] == t and r['n'] == n]
            k = sum(r.get('status') == 'arrived' for r in sel)
            cells.append(f'{k}/{len(sel)}')
            tot[0] += k
            tot[1] += len(sel)
        totals[t] = dict(by_n={str(n): c for n, c in zip(sizes, cells)}, total=f'{tot[0]}/{tot[1]}')
        md.append(f"| {t} | {' | '.join(cells)} | {tot[0]}/{tot[1]} |")

    # 2. paired comparisons per N
    pr = {}
    for pq in [x for x in a.pairs.split(',') if x]:
        x, y = pq.split(':')
        pr[pq] = {}
        md += ['', f'## 配对 {x} − {y}', '', '| N | 配对数 | 前者到达 | 后者到达 | 仅前者 | 仅后者 | 精确二项 p | 都到达 | 时间差中位数 (s) | 平均 | 前者快/慢 |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
        for n in sizes + ['all']:
            sub = [r for r in rows if n == 'all' or r['n'] == n]
            s = paired(sub, x, y)['all']
            pr[pq][str(n)] = s
            md.append(f"| {n} | {s['pairs']} | {s['a_arrived']} | {s['b_arrived']} | {s['a_only']} | {s['b_only']} | "
                      f"{s['exact_binomial_p']:.3g} | {s['both']} | {fmt(s['dt_median'])} | {fmt(s['dt_mean'])} | "
                      f"{s['a_faster']}/{s['a_slower']} |")

    # 3. end state of every run that did not arrive (our runs only; the reference arrived everywhere it is cited)
    md += ['', '## 未到达运行的终态', '', '| 组 | 场景 | 状态 | 终态最大误差 (m) | 最近时最大误差 (m) | 最近时刻 (s) | 到 5 cm (s) | 冻结拍占比 | 切步 | 墙钟 (s) |',
           '|---|---|---|---|---|---|---|---|---|---|']
    na = []
    for r in sorted(rows, key=lambda r: (order.index(r['_tag']) if r['_tag'] in order else 99, r['n'], r['scene'])):
        if r.get('status') == 'arrived' or '_dir' not in r:
            continue
        e = end_state(r['_dir'], r['scene'], r['_tag'])
        e.update(tag=r['_tag'], scene=r['scene'], status=r.get('status'), frozen=r.get('frozen_robot_frac'),
                 cuts=r.get('step_cuts'), wall=r.get('wall_seconds'))
        na.append(e)
        md.append(f"| {r['_tag']} | {r['scene']} | {r.get('status')} | {e['final_max_err']:.3f} | {e['min_max_err']:.3f} | "
                  f"{e['t_min']:.2f} | {fmt(e['t_5cm'])} | {fmt(e['frozen'], 3)} | {e['cuts']} | {fmt(e['wall'], 0)} |")
    (out / 'by_size.md').write_text('\n'.join(md) + '\n', encoding='utf8')
    (out / 'by_size.json').write_text(json.dumps(dict(groups=groups, totals=totals, pairs=pr, non_arrivals=na), indent=1,
                                                 ensure_ascii=False), encoding='utf8')
    print('\n'.join(md))

    # 4. strip figure, one panel per N
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.6 * len(sizes) + 1.0, 0.45 * len(order) + 1.5), facecolor=SURF, sharey=True)
    rng = np.random.default_rng(0)
    xfail = a.horizon + 10.0          # far enough from the 60 s tick that the two labels do not touch
    for ax, n in zip(np.atleast_1d(axes), sizes):
        ax.set_facecolor(SURF)
        for k, tag in enumerate(order[::-1]):
            sel = [r for r in rows if r['_tag'] == tag and r['n'] == n]
            arr = np.array([r['arrival'] for r in sel if r.get('status') == 'arrived'])
            nf = len(sel) - len(arr)
            ax.scatter(arr, k + rng.uniform(-0.18, 0.18, len(arr)), s=22, color=C1, edgecolors=SURF, linewidths=1.0, zorder=3)
            if nf:
                ax.scatter(np.full(nf, xfail) + rng.uniform(-1.2, 1.2, nf), k + rng.uniform(-0.18, 0.18, nf), s=22,
                           color=MUTED, edgecolors=SURF, linewidths=1.0, zorder=3)
            if len(arr):
                med = float(np.median(arr))
                ax.plot([med, med], [k - 0.3, k + 0.3], color=INK, lw=2, zorder=4)
            ax.annotate(f'{len(arr)}/{len(sel)}', (a.horizon + 17.0, k), color=INK2, fontsize=8, va='center')
        nl = sum(1 for r in rows if r['n'] == n and r['_tag'] == order[0] and r['_kind'] == 'local')
        npm = sum(1 for r in rows if r['n'] == n and r['_tag'] == order[0] and r['_kind'] == 'perm')
        ax.set_title(f'N={n}（{nl} 局部交换' + (f' + {npm} 置换）' if npm else '）'), color=INK, fontsize=10, loc='left')
        ax.set_xlim(0, a.horizon + 25)
        ax.set_xticks(list(range(0, int(a.horizon) + 1, 20)) + [xfail])
        ax.set_xticklabels([str(x) for x in range(0, int(a.horizon) + 1, 20)] + ['未到达'], color=INK2, fontsize=8)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(AXIS)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', colors=MUTED)
        ax.grid(True, axis='x', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('到达时间 (s)', color=INK2, fontsize=9)
    np.atleast_1d(axes)[0].set_yticks(range(len(order)))
    np.atleast_1d(axes)[0].set_yticklabels([NAMES.get(t, t) for t in order[::-1]], color=INK2, fontsize=9)
    fig.suptitle('E2 零样本跨规模（B=8，60 s）：竖线为到达者中位数，右侧为到达数', color=INK, fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'fig_e2_strip.png', dpi=200, facecolor=SURF)
    fig.savefig(out / 'fig_e2_strip.pdf', facecolor=SURF)
    print('saved', out / 'fig_e2_strip.png')


if __name__ == '__main__':
    main()

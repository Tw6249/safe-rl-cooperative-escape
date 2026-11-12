"""Tables for README §8 from evaluate.py outputs (no new simulation).

    D:\\miniconda\\python.exe -X utf8 make_report.py --runs ../results/<dir1>,../results/<dir2> --out ../results/<report-dir>
"""
from pathlib import Path
import argparse, json
import numpy as np
from scipy.stats import binomtest


def load(dirs):
    rows = []
    for d in dirs:
        for f in sorted((Path(d) / 'rows').glob('*.json')):
            r = json.loads(f.read_text(encoding='utf8'))
            r['_tag'] = r['job']['tag']
            r['_task'] = r['scene'].rsplit('_f', 1)[0] if '_f' in r['scene'] else r['scene']
            rows.append(r)
    return rows


def load_reference(d):
    out = []
    for r in json.loads((Path(d) / 'summary.json').read_text(encoding='utf8')):
        if r['job']['group'] != 'L':
            continue
        # E2 rows carry frac=None (closed_loop.scenes_for cases have no spectral-level field), so the reference is
        # keyed the same way for the pairing in paired()
        out.append(r)
    return out


def fmt(x, k=2):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{k}f}'


def group_table(rows, keys):
    lines = ['| 组 | 谱水平 f | 场景 | 到达 | 状态 | 到达时间中位数 (s) | 冻结机器人拍占比中位数 | 审计失败段 | 切步 | 最小间距 | min(λ₂下界−λ_req) |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    stats = {}
    for tag in keys:
        for f in sorted({r.get('frac') for r in rows if r['_tag'] == tag}, key=lambda v: (v is None, v)):
            sel = [r for r in rows if r['_tag'] == tag and r.get('frac') == f]
            if not sel:
                continue
            arr = [r['arrival'] for r in sel if r.get('status') == 'arrived']
            st = {}
            for r in sel:
                st[r.get('status')] = st.get(r.get('status'), 0) + 1
            fr = [r.get('frozen_robot_frac') for r in sel if r.get('frozen_robot_frac') is not None]
            md = [r.get('audit_min_distance') or r.get('min_distance') for r in sel if r.get('steps')]
            ml = [(r.get('audit_min_lambda2') or r.get('min_lambda2_lower')) - r['lambda_required'] for r in sel
                  if r.get('steps') and (r.get('audit_min_lambda2') or r.get('min_lambda2_lower')) is not None]
            stats[(tag, f)] = dict(n=len(sel), arrived=len(arr), audit_failed=sum(r.get('audit_failed') or 0 for r in sel))
            lines.append(f"| {tag} | {fmt(f)} | {len(sel)} | {len(arr)} | "
                         f"{', '.join(f'{k} {v}' for k, v in sorted(st.items(), key=lambda x: str(x[0])))} | "
                         f"{fmt(float(np.median(arr)) if arr else None)} | {fmt(float(np.median(fr)) if fr else None, 3)} | "
                         f"{sum(r.get('audit_failed') or 0 for r in sel)} | {sum(r.get('step_cuts') or 0 for r in sel)} | "
                         f"{fmt(min(md) if md else None, 9)} | {fmt(min(ml) if ml else None, 4)} |")
    return '\n'.join(lines), stats


def paired(rows, a, b):
    A = {(r['_task'], r.get('frac')): r for r in rows if r['_tag'] == a}
    Bm = {(r['_task'], r.get('frac')): r for r in rows if r['_tag'] == b}
    keys = sorted(set(A) & set(Bm), key=str)
    out = {}
    for f in sorted({k[1] for k in keys}, key=lambda v: (v is None, v)) + ['all']:
        ks = [k for k in keys if f == 'all' or k[1] == f]
        sa = [A[k].get('status') == 'arrived' for k in ks]
        sb = [Bm[k].get('status') == 'arrived' for k in ks]
        bwin = sum(x and not y for x, y in zip(sa, sb))
        lwin = sum(y and not x for x, y in zip(sa, sb))
        p = binomtest(bwin, bwin + lwin, 0.5).pvalue if bwin + lwin else 1.0
        dt = [A[k]['arrival'] - Bm[k]['arrival'] for k, x, y in zip(ks, sa, sb) if x and y]
        out[str(f)] = dict(pairs=len(ks), a_arrived=sum(sa), b_arrived=sum(sb), a_only=bwin, b_only=lwin,
                           exact_binomial_p=float(p), both=len(dt),
                           dt_median=float(np.median(dt)) if dt else None, dt_mean=float(np.mean(dt)) if dt else None,
                           a_faster=sum(x < 0 for x in dt), a_slower=sum(x > 0 for x in dt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', default='')
    ap.add_argument('--reference-l', default=None)
    ap.add_argument('--only-scenes-of', default=None, help='restrict the reference to scenes present in the runs')
    a = ap.parse_args()
    rows = load(a.runs.split(','))
    if a.reference_l:
        names = {r['scene'] for r in rows}
        rows += [r for r in load_reference(a.reference_l) if r['scene'] in names]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    keys = sorted({r['_tag'] for r in rows})
    tab, stats = group_table(rows, keys)
    md = ['# 汇总', '', tab, '']
    pr = {}
    for pq in [x for x in a.pairs.split(',') if x]:
        x, y = pq.split(':')
        pr[pq] = paired(rows, x, y)
        md += [f'## 配对 {x} − {y}', '', '| f | 配对数 | 前者到达 | 后者到达 | 仅前者 | 仅后者 | 精确二项 p | 都到达 | 时间差中位数 (s) | 平均 | 前者快/慢 |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
        for f, s in pr[pq].items():
            md.append(f"| {f} | {s['pairs']} | {s['a_arrived']} | {s['b_arrived']} | {s['a_only']} | {s['b_only']} | "
                      f"{s['exact_binomial_p']:.3g} | {s['both']} | {fmt(s['dt_median'])} | {fmt(s['dt_mean'])} | "
                      f"{s['a_faster']}/{s['a_slower']} |")
        md.append('')
    (out / 'table.md').write_text('\n'.join(md), encoding='utf8')
    (out / 'stats.json').write_text(json.dumps(dict(groups={f'{k[0]}|{k[1]}': v for k, v in stats.items()},
                                                    pairs=pr), indent=1), encoding='utf8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()

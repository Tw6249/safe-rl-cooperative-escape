"""Offline diagnostics from evaluate.py outputs (no new simulation): final-approach timing and the gate / residual /
filter intervention in the last 2 s of every run (README §8).

    D:\\miniconda\\python.exe -X utf8 diag_traces.py --runs ../results/<d1>,../results/<d2> --out ../results/<diag-dir>

t_5cm / t_1cm: first time after which max_i |p_i - g_i| never exceeds 5 cm / 1 cm again (None if it does at the end);
t_1cm - t_5cm is the time spent on the final approach. Gate, residual and intervention are read from the saved per-step
trace (ResidualFlow records them; BraidFlow runs have no gate or residual). Norms over all robots are divided by
sqrt(n), i.e. reported per robot (RMS)."""
from pathlib import Path
import argparse, json
import numpy as np

WINDOW = 40          # samples: 2 s at dt = 0.05
TOLS = (0.05, 0.01)


def settle_time(states, goal, times, tol):
    err = np.linalg.norm(states - goal[None], axis=2).max(axis=1)
    bad = np.flatnonzero(err > tol)
    if len(bad) and bad[-1] == len(err) - 1:
        return None
    return float(times[0 if not len(bad) else bad[-1] + 1])


def diag_one(d, scene, tag, n):
    z = np.load(d / 'runs' / f'{scene}_{tag}.npz')
    tr = json.loads((d / 'runs' / f'{scene}_{tag}_trace.json').read_text(encoding='utf8'))
    st, goal, times = z['states'], z['goal'], z['times']
    m = min(len(st), len(times))
    out = {f't_{int(round(tol * 100))}cm': settle_time(st[:m], goal, times[:m], tol) for tol in TOLS}
    out['approach'] = (out['t_1cm'] - out['t_5cm']) if out['t_1cm'] is not None and out['t_5cm'] is not None else None
    rn = 1.0 / np.sqrt(n)
    last = tr[-WINDOW:]

    def mean(rows, key, scale=1.0):
        v = [x[key] for x in rows if x.get(key) is not None]
        return float(np.mean(v)) * scale if v else None

    out.update(gate_last2s=mean(last, 'gate_mean'), residual_rms_last2s=mean(last, 'residual_norm', rn),
               intervention_rms_last2s=mean(last, 'intervention', rn), u_rms_last2s=mean(last, 'u_norm', rn),
               u_diff_last2s=mean(last, 'u_diff'), gate_all=mean(tr, 'gate_mean'),
               intervention_rms_all=mean(tr, 'intervention', rn), u_diff_all=mean(tr, 'u_diff'))
    go = [x.get('gate_open') for x in tr if x.get('gate_open') is not None]
    out['gate_all_open_frac'] = float(np.mean([g >= n for g in go])) if go else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    per = []
    for d in a.runs.split(','):
        d = Path(d)
        for f in sorted((d / 'rows').glob('*.json')):
            r = json.loads(f.read_text(encoding='utf8'))
            if not r.get('steps'):
                continue
            tag = r['job']['tag']
            x = dict(scene=r['scene'], tag=tag, n=r['n'], frac=r.get('frac'), kind=r.get('kind'), status=r.get('status'),
                     arrival=r.get('arrival'), run_dir=str(d))
            try:
                x.update(diag_one(d, r['scene'], tag, r['n']))
            except FileNotFoundError as exc:
                x['error'] = str(exc)
            per.append(x)
    cols = ['t_5cm', 't_1cm', 'approach', 'gate_last2s', 'residual_rms_last2s', 'intervention_rms_last2s',
            'u_diff_last2s', 'gate_all', 'gate_all_open_frac', 'intervention_rms_all', 'u_diff_all']
    # u_diff is pa_bridge's |u_L - u_centralQP| (distance between the local settlement and the exact central QP),
    # not the filter correction; intervention = |u - Pi_U(u_nom)| recorded by ResidualFlow.
    lines = ['| 组 | f | 场景 | 到达 | 到 5 cm 中位数 (s) | 到 1 cm 中位数 (s) | 最后 5 cm→1 cm 中位数 (s) | 末 2 s 门控均值 | 末 2 s 残差 RMS (m/s) | 末 2 s 滤波修正 RMS (m/s) | 末 2 s ‖u_L−u_QP‖ | 全程门控均值 | 全程门控全开拍占比 | 全程滤波修正 RMS | 全程 ‖u_L−u_QP‖ |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    summary = {}
    for tag in sorted({x['tag'] for x in per}):
        for fr in sorted({x['frac'] for x in per if x['tag'] == tag}, key=lambda v: (v is None, v)):
            sel = [x for x in per if x['tag'] == tag and x['frac'] == fr]
            med = {}
            for c in cols:
                v = [x[c] for x in sel if x.get(c) is not None]
                med[c] = float(np.median(v)) if v else None
                med[c + '_n'] = len(v)
            arrived = sum(x['status'] == 'arrived' for x in sel)
            summary[f'{tag}|{fr}'] = dict(n=len(sel), arrived=arrived, **med)
            fmt = lambda v, k=2: '—' if v is None else f'{v:.{k}f}'
            lines.append(f"| {tag} | {fmt(fr)} | {len(sel)} | {arrived} | {fmt(med['t_5cm'])} ({med['t_5cm_n']}) | "
                         f"{fmt(med['t_1cm'])} ({med['t_1cm_n']}) | {fmt(med['approach'])} | {fmt(med['gate_last2s'], 3)} | "
                         f"{fmt(med['residual_rms_last2s'], 4)} | {fmt(med['intervention_rms_last2s'], 4)} | "
                         f"{fmt(med['u_diff_last2s'], 4)} | {fmt(med['gate_all'], 3)} | {fmt(med['gate_all_open_frac'], 3)} | "
                         f"{fmt(med['intervention_rms_all'], 4)} | {fmt(med['u_diff_all'], 4)} |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'diag_table.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    (out / 'diag_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf8')
    (out / 'diag_per_run.json').write_text(json.dumps(per, indent=1), encoding='utf8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

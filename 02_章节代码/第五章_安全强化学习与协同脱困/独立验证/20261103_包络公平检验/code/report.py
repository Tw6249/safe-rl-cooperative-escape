"""Tables and paired tests for results/run01 exactly as registered in README §3. Reads rows only."""
import json, sys
from math import comb
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figs as F

RUN = HERE.parent / 'results' / 'run01'


def p2(a, b):
    n = a + b
    return float('nan') if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def by_size(D, tags, scenes, lines):
    lines.append('| 组 | N=12 | N=20 | N=25 | 合计 | 切步 | 审计失败 | 未到达的终止误差 (cm) |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for t in tags:
        cells = []
        for n in (12, 20, 25):
            rs = [D[(t, s)] for s in scenes if s.startswith(f'H{n}_')]
            arr = [r['arrival'] for r in rs if r['status'] == 'arrived']
            cells.append(f'{len(arr)}/{len(rs)}' + (f'（中位 {np.median(arr):.2f} s）' if arr else ''))
        rs = [D[(t, s)] for s in scenes]
        fails = ', '.join(f"{100 * r.get('final_max_err', np.nan):.1f}" for r in rs if r['status'] != 'arrived')
        lines.append(f"| {F.LABEL.get(t, t)} | {' | '.join(cells)} | {sum(r['status'] == 'arrived' for r in rs)}/{len(rs)} | "
                     f"{sum(r.get('step_cuts', 0) or 0 for r in rs)} | {sum(r.get('audit_failed', 0) or 0 for r in rs)} | {fails or '—'} |")


def paired(D, a, others, scenes, lines):
    lines.append('| A 对 B | 不一致对 A独到:B独到 | 双侧 p | 共同到达 | Δt 中位 (s) | Δt 均值 (s) | A 更快/更慢 |')
    lines.append('|---|---|---|---|---|---|---|')
    for b in others:
        ao = bo = 0
        dts = []
        for s in scenes:
            xa, xb = D[(a, s)]['status'] == 'arrived', D[(b, s)]['status'] == 'arrived'
            ao += xa and not xb
            bo += xb and not xa
            if xa and xb:
                dts.append(D[(a, s)]['arrival'] - D[(b, s)]['arrival'])
        dts = np.array(dts)
        md = f'{np.median(dts):+.2f}' if len(dts) else '—'
        mn = f'{dts.mean():+.2f}' if len(dts) else '—'
        lines.append(f'| {F.LABEL.get(a, a)} 对 {F.LABEL.get(b, b)} | {ao}:{bo} | {p2(ao, bo):.4g} | {len(dts)} | {md} | {mn} | '
                     f'{int((dts < 0).sum())}/{int((dts > 0).sum())} |')


def main():
    D = F.rows(RUN)
    e2 = F.load_e2()
    sc2 = sorted({s for _, s in e2 if s.endswith('s20261017')}, key=lambda x: (int(x.split('_')[0][1:]), x))
    scf = sorted({s for _, s in D if s.endswith('s20261105')}, key=lambda x: (int(x.split('_')[0][1:]), x))
    L = ['# run01 结果表（由 code/report.py 生成）', '', '## A. E2 公平消融（种子 20261017，样本内）', '']
    by_size(e2, t2, sc2, L)
    L.append('')
    paired(e2, 'rl_full_env_B8', ['rl_geom_env_B8', 'rl_full_int_env_B8'], sc2, L)
    L += ['', f'## B. 新种子 20261105（{len(scf)} 场景，此前未用）', '']
    by_size(D, tf, scf, L)
    L.append('')
    L.append('')
    L.append('补充（非登记项，仅描述）：')
    L.append('')
    txt = '\n'.join(L) + '\n'
    (RUN.parent / 'run01_report.md').write_text(txt, encoding='utf8')
    print(txt)


if __name__ == '__main__':
    main()

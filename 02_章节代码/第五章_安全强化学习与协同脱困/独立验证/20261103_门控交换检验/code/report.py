"""Tables, paired tests (README §3) and the arrival matrix for results/run01. Reads rows/npz only."""
import json, sys
from math import comb
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
V = HERE.parents[1]
FAIR = V / '20261103_包络公平检验' / 'results' / 'run01'
SWAP = HERE.parent / 'results' / 'run01'
sys.path.insert(0, str(V / '20261103_包络公平检验' / 'code'))
import figs as F                                                     # noqa: E402

REPRO = 'H12_local2_s20261105'
NAME = {'rl_full_env_B8': 'rl_full 权重 + μ（训练时门控）', 'wfull_gint_B8': 'rl_full 权重 + 修正量门控',
        'wfull_gone_B8': 'rl_full 权重 + 门控恒为 1',
        'rl_full_int_env_B8': 'rl_full_int 权重 + 修正量（训练时）', 'wfullint_gmu_B8': 'rl_full_int 权重 + μ',
        'rl_geom_env_B8': 'rl_geom 权重 + 修正量（训练时）', 'wgeom_gmu_B8': 'rl_geom 权重 + μ'}
ORDER = list(NAME)


def p2(a, b):
    n = a + b
    return float('nan') if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def load():
    D = {}
    for k, r in F.rows(FAIR).items():
        if k[0] in NAME:
            D[k] = r
    for k, r in F.rows(SWAP).items():
        if k[0] in NAME and k[0] not in ('rl_full_env_B8', 'rl_full_int_env_B8'):
            D[k] = r
    return D


def main():
    D = load()
    sc = sorted({s for _, s in D if s.endswith('s20261105')}, key=lambda x: (int(x.split('_')[0][1:]), x))
    L = ['# 门控交换检验 run01 结果（由 code/report.py 生成）', '']
    L += ['## 复现检查', '']
    for tag in ('rl_full_env_B8', 'rl_full_int_env_B8'):
        a = np.load(SWAP / 'runs' / f'{REPRO}_{tag}.npz')['states']
        b = np.load(FAIR / 'runs' / f'{REPRO}_{tag}.npz')['states']
        same = a.shape == b.shape and bool(np.array_equal(a, b))
        diff = float(np.abs(a[:min(len(a), len(b))] - b[:min(len(a), len(b))]).max())
        L.append(f'- {tag} @ {REPRO}：步数 {len(a)} / {len(b)}，逐位相同：{same}，最大位置差 {diff:.3e} m')
    L += ['', '## 各组到达（16 个场景）', '',
          '| 组 | N=12 | N=20 | N=25 | 合计 | 切步 | 审计失败 | 未到达的场景 |', '|---|---|---|---|---|---|---|---|']
    for t in ORDER:
        rs = [D[(t, s)] for s in sc]
        cells = []
        for n in (12, 20, 25):
            sub = [D[(t, s)] for s in sc if s.startswith(f'H{n}_')]
            arr = [r['arrival'] for r in sub if r['status'] == 'arrived']
            cells.append(f'{len(arr)}/{len(sub)}' + (f'（中位 {np.median(arr):.2f} s）' if arr else ''))
        fails = ', '.join(f"{s.split('_s')[0]}（{D[(t, s)]['status']}，{D[(t, s)].get('final_max_err', np.nan):.2f} m）"
                          for s in sc if D[(t, s)]['status'] != 'arrived')
        L.append(f"| {NAME[t]} | {' | '.join(cells)} | {sum(r['status'] == 'arrived' for r in rs)}/16 | "
                 f"{sum(r.get('step_cuts', 0) or 0 for r in rs)} | {sum(r.get('audit_failed', 0) or 0 for r in rs)} | {fails or '—'} |")

    def paired(pairs, title):
        L.extend(['', title, '', '| A 对 B | 不一致对 A独到:B独到 | 双侧 p | 共同到达 | Δt 中位 (s) | Δt 均值 (s) | A 更快/更慢 |',
                  '|---|---|---|---|---|---|---|'])
        for a, b in pairs:
            ao = bo = 0
            dts = []
            for s in sc:
                xa, xb = D[(a, s)]['status'] == 'arrived', D[(b, s)]['status'] == 'arrived'
                ao += xa and not xb
                bo += xb and not xa
                if xa and xb:
                    dts.append(D[(a, s)]['arrival'] - D[(b, s)]['arrival'])
            dts = np.array(dts)
            md = f'{np.median(dts):+.2f}' if len(dts) else '—'
            mn = f'{dts.mean():+.2f}' if len(dts) else '—'
            L.append(f'| {NAME[a]} 对 {NAME[b]} | {ao}:{bo} | {p2(ao, bo):.4g} | {len(dts)} | {md} | {mn} | '
                     f'{int((dts < 0).sum())}/{int((dts > 0).sum())} |')

    paired([('rl_full_env_B8', 'wfull_gint_B8'), ('wfullint_gmu_B8', 'rl_full_int_env_B8'),
            ('rl_full_env_B8', 'wfull_gone_B8'), ('wgeom_gmu_B8', 'rl_geom_env_B8')],
           '## 登记的配对检验（同一权重，只换门控）')
    paired([('wfull_gint_B8', 'rl_full_int_env_B8'), ('rl_full_env_B8', 'wfullint_gmu_B8')],
           '## 同门控、不同权重')
    txt = '\n'.join(L) + '\n'
    (SWAP.parent / 'run01_report.md').write_text(txt, encoding='utf8')
    print(txt)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['hatch.color'] = F.AXIS
    F.LABEL.update(NAME)
    out = SWAP.parent / 'figs01'
    out.mkdir(exist_ok=True)
    F.matrix(D, ORDER, sc, '门控交换（种子 20261105）：同一权重换门控后，每个场景的到达时间', out / 'swap_matrix.png')


if __name__ == '__main__':
    main()

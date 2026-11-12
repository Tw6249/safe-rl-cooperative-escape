"""Static figures from saved rows/npz only (no simulation). Writes to results/figs01 (or --out)."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
V = HERE.parents[1]
S = V / '20261102_安全强化学习' / 'results'
G = V / '20261103_终段包络验证' / 'results'
LED = V / '20261029_边账本结清' / 'results' / 'run01'
FAIR = HERE.parent / 'results' / 'run01'
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
SEQ = LinearSegmentedColormap.from_list('seq', ['#dfeafa', '#8fb6ea', '#2a78d6', '#163f73'])
NAMES = {'rl_full_env_B8': 'rl_full + 包络',
         'rl_full_B8': 'rl_full（无包络）', 'rl_geom_env_B8': 'rl_geom + 包络', 'rl_geom_B8': 'rl_geom（无包络）',
         'rl_full_int_env_B8': 'rl_full_int + 包络', 'rl_full_int_B8': 'rl_full_int（无包络）',
         'heur_B8': '固定环流 heur', 'myopic_B8': '短视 myopic'}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def rows(d):
    out = {}
    for f in sorted((d / 'rows').glob('*.json')):
        r = json.loads(f.read_text(encoding='utf8'))
        out[(r.get('tag') or r['job']['tag'], r['scene'])] = r
    return out


def load_e2():
    e2 = {}
    for d in (S / 'test_e2_B8', S / 'test_e2_B8_fullint', G / 'e2_full_env_B8'):
        e2.update(rows(d))
    if (FAIR / 'rows').exists():
        e2.update({k: v for k, v in rows(FAIR).items() if k[1].endswith('s20261017')})
    return e2


def matrix(D, tags, scenes, title, path):
    sizes = [s.split('_')[0] for s in scenes]
    fig, ax = plt.subplots(figsize=(14.5, 0.55 * len(tags) + 1.9), facecolor=SURF)
    ax.set_facecolor(SURF)
    vmax = 30.0
    for yi, t in enumerate(tags):
        for xi, s in enumerate(scenes):
            r = D.get((t, s))
            if r is None:
                ax.add_patch(plt.Rectangle((xi + .04, yi + .06), .92, .88, fc=SURF, ec=GRID, lw=.8))
                ax.text(xi + .5, yi + .5, '未跑', ha='center', va='center', fontsize=8, color=MUTED)
                continue
            if r['status'] == 'arrived':
                v = r['arrival']
                c = SEQ(min(v, vmax) / vmax)
                ax.add_patch(plt.Rectangle((xi + .04, yi + .06), .92, .88, fc=c, ec=SURF, lw=2))
                ax.text(xi + .5, yi + .5, f'{v:.1f}', ha='center', va='center', fontsize=8.5,
                        color='white' if v / vmax > .45 else INK)
            else:
                e = 100 * r.get('final_max_err', np.nan)
                ax.add_patch(plt.Rectangle((xi + .04, yi + .06), .92, .88, fc='#f1f0ec', ec=AXIS, lw=.8,
                                           hatch='////'))
                ax.text(xi + .5, yi + .5, f'×\n{e:.0f} cm' if e >= 1.5 else f'×\n{e:.1f} cm', ha='center',
                        va='center', fontsize=7.5, color=INK, linespacing=.95)
    ax.set_xlim(0, len(scenes))
    ax.set_ylim(len(tags), 0)
    ax.set_yticks(np.arange(len(tags)) + .5)
    ax.set_yticklabels([f"{LABEL.get(t, t)}   {sum(D.get((t, s), {}).get('status') == 'arrived' for s in scenes)}/{len(scenes)}"
                        for t in tags], fontsize=9.5, color=INK)
    ax.set_xticks(np.arange(len(scenes)) + .5)
    ax.set_xticklabels([s.split('_')[1] for s in scenes], fontsize=8, color=INK2, rotation=0)
    for xi in range(1, len(scenes)):
        if sizes[xi] != sizes[xi - 1]:
            ax.axvline(xi, color=INK2, lw=1.2)
    for n in sorted(set(sizes), key=sizes.index):
        idx = [i for i, z in enumerate(sizes) if z == n]
        ax.text(np.mean(idx) + .5, -.12, f'N = {n[1:]}', ha='center', va='bottom', fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.suptitle(title, x=0.01, ha='left', color=INK, fontsize=11.5)
    bad = sum(D[(t, s)].get('audit_failed', 0) or 0 for t in tags for s in scenes if (t, s) in D)
    fig.text(0.01, 0.01, '格内数字：到达时间（s，误差 ≤ 1 cm 并保持 1 s）；颜色越深越慢（30 s 封顶）。'
             f'斜线格：60 s 内未到达，下方为终止时最大误差。每格一次运行，共 {bad} 次审计失败。',
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def err_curve(entries, title, path, note=''):
    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=SURF)
    style(ax)
    for (lab, d, stem), col in zip(entries, (C2, C1, C3)):
        z = np.load(d / 'runs' / f'{stem}.npz')
        e = np.linalg.norm(z['states'] - z['goal'][None], axis=2).max(axis=1)
        t = np.array(z['times'])
        ax.plot(t, np.maximum(e, 1e-4), color=col, lw=2)
        ax.annotate(lab, (t[-1], max(e[-1], 1e-4)), xytext=(6, 0), textcoords='offset points', va='center',
                    fontsize=9, color=INK)
    ax.axhline(0.01, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.3, 0.0118, '到达阈值 1 cm', fontsize=8.5, color=INK2)
    ax.set_yscale('log')
    ax.set_xlabel('时间 t（s）', color=INK2)
    ax.set_ylabel('最大目标误差 max_i ‖e_i‖_2（m）', color=INK2)
    ax.set_title(title, loc='left', color=INK, fontsize=11)
    ax.margins(x=0.02)
    xl = ax.get_xlim()
    ax.set_xlim(xl[0], xl[1] * 1.22)
    fig.text(0.01, 0.01, note, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE.parent / 'results' / 'figs01'))
    a = ap.parse_args()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['hatch.color'] = AXIS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    e2 = load_e2()
    sc2 = sorted({s for _, s in e2 if s.endswith('s20261017')}, key=lambda x: (int(x.split('_')[0][1:]), x))
    tags = ['rl_full_env_B8', 'rl_full_B8', 'rl_geom_env_B8', 'rl_geom_B8', 'rl_full_int_env_B8',
            'rl_full_int_B8', 'heur_B8', 'myopic_B8']
    matrix(e2, tags, sc2, 'E2（种子 20261017，已用过的测试种子）：每个场景、每种方法的到达时间', out / 'e2_matrix.png')
    err_curve([('rl_full（无包络）', S / 'test_e2_B8', 'H20_local1_s20261017_rl_full_B8'),
               ('rl_full + 包络', G / 'e2_full_env_B8', 'H20_local1_s20261017_rl_full_env_B8')],
              'H20_local1：无包络时终点悬停在 1 cm 阈值之上', out / 'hover_H20_local1.png')
    e1 = {}
    for d in (S / 'test_e1_f60_B8', S / 'test_e1_f90_B8', G / 'e1_full_env_B8'):
        e1.update(rows(d))
    fr = rows(FAIR) if (FAIR / 'rows').exists() else {}
    scf = sorted({s for _, s in fr if s.endswith('s20261105')}, key=lambda x: (int(x.split('_')[0][1:]), x))
    if scf:
        matrix(fr, ['rl_full_env_B8', 'rl_full_B8', 'rl_geom_env_B8', 'rl_full_int_env_B8', 'heur_B8',
                    'myopic_B8'], scf, '新种子 20261105（运行前登记，此前未用）：每个场景、每种方法的到达时间',
               out / 'fresh_matrix.png')
    print('written', out)


if __name__ == '__main__':
    main()

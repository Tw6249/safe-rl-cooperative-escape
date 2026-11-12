"""Generate paired statistics comparing rl_full_env_B8 against all baselines and ablations on E1 and E2."""
import json
from pathlib import Path
from math import comb
import numpy as np

HERE = Path(__file__).resolve().parent
RES_ENV = HERE.parent / 'results'
RES_SARL = HERE.parents[1] / '20261102_安全强化学习' / 'results'
RES_LED = HERE.parents[1] / '20261029_边账本结清' / 'results' / 'run01'


def binom_p(a, b):
    n = a + b
    if n == 0:
        return None
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_rows(dirs):
    out = {}
    for d in dirs:
        for f in sorted((Path(d) / 'rows').glob('*.json')):
            r = json.loads(f.read_text(encoding='utf8'))
            tag = r.get('tag') or r['job']['tag']
            out[(tag, r['scene'])] = r
    return out


# E2 comparison
e2 = load_rows([RES_SARL / 'test_e2_B8', RES_SARL / 'test_e2_B8_fullint', RES_ENV / 'e2_full_env_B8'])
tags_e2 = ['rl_full_env_B8', 'rl_full_B8', 'rl_geom_B8', 'rl_full_int_B8', 'heur_B8', 'myopic_B8']

scenes_e2 = sorted({s for _, s in e2 if s.endswith('s20261017')}, key=lambda x: (int(x.split('_')[0][1:]), x))

print('=== E2 BY SIZE ===')
for t in tags_e2:
    by_n = {}
    for n in (12, 20, 25):
        sc = [s for s in scenes_e2 if s.startswith(f'H{n}_')]
        arr = [e2[(t, s)]['arrival'] for s in sc if e2[(t, s)]['status'] == 'arrived']
        by_n[n] = f'{len(arr)}/{len(sc)}' + (f' (med {np.median(arr):.2f}s)' if arr else '')
    tot = sum(e2[(t, s)]['status'] == 'arrived' for s in scenes_e2)
    print(f'{t:16s} | N=12: {by_n[12]:18s} | N=20: {by_n[20]:18s} | N=25: {by_n[25]:18s} | Total: {tot}/16')

print('\n=== E2 PAIRED: rl_full_env_B8 vs others ===')
for other in tags_e2[1:]:
    a_arr = b_arr = a_only = b_only = 0
    dts = []
    for s in scenes_e2:
        ra, rb = e2[('rl_full_env_B8', s)], e2[(other, s)]
        xa, xb = ra['status'] == 'arrived', rb['status'] == 'arrived'
        a_arr += xa; b_arr += xb
        a_only += xa and not xb; b_only += xb and not xa
        if xa and xb:
            dts.append(ra['arrival'] - rb['arrival'])
    p = binom_p(a_only, b_only)
    print(f'rl_full_env_B8 vs {other:14s}: arrived {a_arr:2d}:{b_arr:2d} | discordant {a_only:2d}:{b_only:2d} (p={p}) | both={len(dts):2d}, med dt={np.median(dts):+.2f}s, mean dt={np.mean(dts):+.2f}s')

# E1 comparison
e1 = load_rows([RES_SARL / 'test_e1_f60_B8', RES_SARL / 'test_e1_f60_B8_fullint',
                RES_SARL / 'test_e1_f90_B8', RES_SARL / 'test_e1_f90_B8_fullint',
                RES_ENV / 'e1_full_env_B8'])
for f_tag in ('f60', 'f90'):
    sc = sorted({s for _, s in e1 if f'_{f_tag}_' in s})
    print(f'\n=== E1 {f_tag} (32 scenes) ===')
    for other in ['rl_full_B8', 'rl_geom_B8', 'rl_full_int_B8', 'heur_B8', 'myopic_B8', 'heur_st_B0']:
        if (other, sc[0]) not in e1:
            continue
        a_arr = b_arr = a_only = b_only = 0
        dts = []
        for s in sc:
            ra, rb = e1[('rl_full_env_B8', s)], e1[(other, s)]
            xa, xb = ra['status'] == 'arrived', rb['status'] == 'arrived'
            a_arr += xa; b_arr += xb
            a_only += xa and not xb; b_only += xb and not xa
            if xa and xb:
                dts.append(ra['arrival'] - rb['arrival'])
        p = binom_p(a_only, b_only)
        print(f'rl_full_env_B8 vs {other:14s}: arrived {a_arr:2d}:{b_arr:2d} | discordant {a_only:2d}:{b_only:2d} (p={p}) | both={len(dts):2d}, med dt={np.median(dts):+.2f}s, faster {sum(d<0 for d in dts)}/{sum(d>0 for d in dts)}')

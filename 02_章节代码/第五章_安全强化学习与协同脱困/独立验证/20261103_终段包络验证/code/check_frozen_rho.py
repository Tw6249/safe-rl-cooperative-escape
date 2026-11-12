import json, numpy as np
from pathlib import Path
SARL = Path(__file__).resolve().parents[1] / '20261102_安全强化学习' / 'results'
for d, tag in [('test_e2_B8', 'rl_geom_B8'), ('test_e2_B8_fullint', 'rl_full_int_B8')]:
    for f in sorted((SARL / d / 'rows').glob('*_' + tag + '.json')):
        r = json.loads(f.read_text(encoding='utf8'))
        if r['status'] == 'arrived': continue
        sc = r['scene']
        z = np.load(SARL / d / 'runs' / (sc + '_' + tag + '.npz'))
        st, g = z['states'], z['goal']
        min_nmax = 1e9
        min_all_nmax = 1e9
        for p in st:
            dist = np.linalg.norm(p[:, None] - p[None], axis=2)
            e = np.linalg.norm(p - g, axis=1)
            for i in range(len(p)):
                nmax_i = float(e[dist[i] <= 2.5].max())
                min_all_nmax = min(min_all_nmax, nmax_i)
                if e[i] > 0.01:
                    min_nmax = min(min_nmax, nmax_i)
        st_name = r['status']
        print(f'{tag:15s} {sc:22s} status={st_name:15s} min_mover_nmax={min_nmax:.4f}m tanh={np.tanh(min_nmax/0.05):.17f} min_any_nmax={min_all_nmax:.4f}m')

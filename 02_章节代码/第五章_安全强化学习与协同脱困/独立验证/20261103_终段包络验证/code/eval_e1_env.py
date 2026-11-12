"""Evaluate EnvelopedResidualFlow on E1 (seed 20261103, N=8, f=0.60 and f=0.90, 32 tasks each) and compare with E2."""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import json, sys
import multiprocessing as mp
import numpy as np

HERE = Path(__file__).resolve().parent
SARL = HERE.parents[1] / '20261102_安全强化学习' / 'code'
sys.path.insert(0, str(SARL))
sys.path.insert(0, str(HERE))
import sarl_env as E
import eval_env as EE


def main():
    import sysperf
    sysperf.configure()
    out = (HERE.parent / 'results' / 'e1_full_env_B8').resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'runs').mkdir(); (out / 'rows').mkdir()
    scenes, _ = E.lattice_scenes(20261103, n=8, n_local=20, n_perm=12, fracs=(0.6, 0.9))
    ckpt = str((SARL.parent / 'results' / 'train_full02' / 'best.pt').resolve())
    jobs = [dict(case={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in scenes[s].items()},
                 out=str(out), ckpt=ckpt, variant='full', gate='mu', envelope=True, rounds=8, horizon=60.0,
                 tag='rl_full_env_B8') for s in sorted(scenes)]
    with mp.get_context('spawn').Pool(16, initializer=EE.init_worker) as pool:
        rows = pool.map(EE.run_one, jobs, chunksize=1)
    for r, s in zip(rows, sorted(scenes)):
        r['frac'] = scenes[s]['frac']
        r['kind'] = scenes[s]['kind']
    (out / 'summary.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf8')
    for f in (0.6, 0.9):
        sub = [r for r in rows if r['frac'] == f]
        arr = [r['arrival'] for r in sub if r['status'] == 'arrived']
        print(f'E1 f={f}: arrived {len(arr)}/{len(sub)}, median arrival {np.median(arr):.2f}s, audit_failed {sum(r["audit_failed"] for r in sub)}')


if __name__ == '__main__':
    main()
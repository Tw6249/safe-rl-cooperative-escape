"""Quick 3-second check on interior-swap scenes (H20_local2, H20_local3, H25_local0) for rl_geom and rl_full_int
with the exact same neighbourhood envelope rho_i."""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import json, sys, multiprocessing as mp
import numpy as np

HERE = Path(__file__).resolve().parent
SARL = HERE.parents[1] / '20261102_安全强化学习' / 'code'
sys.path.insert(0, str(SARL))
sys.path.insert(0, str(HERE))
import sarl_env as E
import eval_env as EE

if __name__ == '__main__':
    import sysperf
    sysperf.configure()
    out = (HERE.parent / 'results' / 'check_int_freeze_3s').resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'runs').mkdir(exist_ok=True); (out / 'rows').mkdir(exist_ok=True)
    scenes = E.NCL.C29.scenes_for(20261017)
    names = ['H20_local2_s20261017', 'H20_local3_s20261017', 'H25_local0_s20261017']
    ckpts = {
        'rl_geom_env': (str((SARL.parent / 'results' / 'train_geom02' / 'best.pt').resolve()), 'geom', 'int'),
        'rl_full_int_env': (str((SARL.parent / 'results' / 'train_fullint02' / 'best.pt').resolve()), 'full_int', 'int'),
        'rl_full_env': (str((SARL.parent / 'results' / 'train_full02' / 'best.pt').resolve()), 'full', 'mu'),
    }
    jobs = []
    for tag, (ck, var, gate) in ckpts.items():
        for s in names:
            jobs.append(dict(case={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in scenes[s].items()},
                             out=str(out), ckpt=ck, variant=var, gate=gate, envelope=True, rounds=8, horizon=3.0,
                             tag=tag))
    with mp.get_context('spawn').Pool(9, initializer=EE.init_worker) as pool:
        rows = pool.map(EE.run_one, jobs, chunksize=1)
    for r in rows:
        npz = np.load(out / 'runs' / f"{r['scene']}_{r['tag']}.npz")
        st = npz['states']
        sp = np.linalg.norm(np.diff(st, axis=0), axis=2).max(axis=1) / 0.05
        print(f"{r['tag']:16s} {r['scene']:22s} err(t=0->3s): {np.linalg.norm(st[0]-npz['goal'],axis=1).max():.3f} -> {r['final_max_err']:.3f}m | max_speed(t=3s)={sp[-1]:.4f} m/s")
"""Implementation check (README §6.1): ShieldOracle under the frozen run_pa must reproduce the saved filter-L smoke run

    D:\\miniconda\\python.exe -X utf8 -W ignore check_identity.py --out ../results/<new-dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'
from pathlib import Path
from functools import partial
import argparse, json, sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sysperf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    perf = sysperf.configure()
    import numpy as np
    import sarl_env as E
    import pa_bridge
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    ref = E.BASE / '20261029_边账本结清' / 'results' / 'smoke01' / 'runs'
    case = E.NCL.C29.scenes_for(20261016)['H25_local1_s20261016']
    pa_bridge.PAOracle = partial(E.ShieldOracle, settle='local', diagnostic=True)
    r = pa_bridge.run_pa(case, out, 1.0, 8, E.RADIUS, 2.0, 'L_s8', clip_level=True)
    mine, theirs = np.load(out / 'H25_local1_s20261016_L_s8.npz'), np.load(ref / 'H25_local1_s20261016_L_s8.npz')
    tm = json.loads((out / 'H25_local1_s20261016_L_s8_trace.json').read_text(encoding='utf8'))
    tt = json.loads((ref / 'H25_local1_s20261016_L_s8_trace.json').read_text(encoding='utf8'))
    fields = ('seed_kind', 'pa_cost', 'cost_gap', 'accepted', 'rejected', 'nodes_settled', 'cascade_depth',
              'broken_settled', 'member_rejects', 'u_norm', 'V')
    res = dict(samples=len(mine['states']), states_identical=bool(np.array_equal(mine['states'], theirs['states'])),
               times_identical=bool(np.array_equal(mine['times'], theirs['times'])),
               max_state_diff=float(np.max(np.abs(mine['states'] - theirs['states']))),
               trace_fields_identical={f: all(x.get(f) == y.get(f) for x, y in zip(tm, tt)) for f in fields},
               trace_len=(len(tm), len(tt)), status=r['status'], process_settings=perf)
    (out / 'check.json').write_text(json.dumps(res, indent=1), encoding='utf8')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()

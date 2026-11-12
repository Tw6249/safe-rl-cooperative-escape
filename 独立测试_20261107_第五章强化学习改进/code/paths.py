"""Path setup shared by every script of this folder.

The frozen Chapter-5 code is imported read-only from 02_章节代码 (never copied, never edited):
  SARL  = .../独立验证/20261102_安全强化学习/code      sarl_env, sarl_policy, evaluate, sysperf
  基础依赖/code                                       pa_bridge, ledger_npa, block_npa, pair_common, audit ...
Bytecode writing is disabled so that importing does not create files inside the frozen directories.
"""
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # 独立测试_20261107_第五章强化学习改进
WS = ROOT.parent                                       # thesis workspace
C5 = WS / '02_章节代码' / '第五章_安全强化学习与协同脱困'
SARL = C5 / '独立验证' / '20261102_安全强化学习' / 'code'
DEPS = C5 / '基础依赖' / 'code'
THESIS_CKPT = (C5 / '独立验证' / '20261102_安全强化学习' / 'results' / 'train_full02' / 'best.pt').resolve()
# The 2026-11 restructure moved thesis_common from 02_核心研究/公共基础 to 02_章节代码/00_公共依赖/统一模型; the frozen
# Chapter-3 runner (run_chapter3.py, imported by npa_oracle for compute_star_betas) still points at the old path.
# Only the search path is added here; no frozen file is edited.  Bit identity is checked by check_identity.py.
COMMON = WS / '02_章节代码' / '00_公共依赖' / '统一模型'

for _p in (str(COMMON), str(HERE), str(SARL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(HERE) != sys.path[0]:
    sys.path.remove(str(HERE))
    sys.path.insert(0, str(HERE))

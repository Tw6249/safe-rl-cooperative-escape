# 第五章基础依赖：精确有理数安全证书、场景生成与闭邻域 NPA 滤波

本目录存放第五章 EGP-NPA（包络等变图策略与邻域近端分配协同控制）共用的底层模块：

| 文件 | 作用 |
|---|---|
| `code/safety_checks.py`、`code/pair_common.py` | 精确有理数零阶保持执行段碰撞与拉普拉斯谱下界证书（`certified_step`） |
| `code/audit_trajectories.py` | 闭环轨迹逐段精确有理数安全审计（`audit`） |
| `code/scenes.py`、`code/scenarios.json` | 密集三角晶格构型生成（`lattice`）与相邻节点对提取 |
| `code/pa_bridge.py` | 线性化谱连通与避碰约束行构造（`build_rows`）及统一闭环运行器（`run_pa`） |
| `code/npa_oracle.py`、`code/block_npa.py`、`code/ledger_npa.py` | 第 3 章闭邻域块近端分配（`NeighbourhoodBlock`）与边记录本地结清安全滤波器（`LedgerNPAOracle`） |

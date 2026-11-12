# 七机器人严格非目标局部极小反例与基础安全核验

本目录保留第五章第 5.3 节“短视误差单调控制的死锁机理与破局条件”所用的七机器人严格局部极小反例核验脚本与证书数据：

| 保留文件 | 用途 |
|---|---|
| `verify_mobile_robot_local_minimum.py` | 七机器人对称接触构型二阶 KKT 严格局部极小有理证书核验 |
| `results/mobile_robot_local_minimum/` | 七机器人严格局部极小反例的冻结证书与核验输出 |
| `pair_common.py`、`safety_checks.py`、`scenarios.json`、`test_safety_checks.py` | 公共安全优化、零阶保持整段认证与基础场景定义 |

早期成对环流（Pair Circulation）比较脚本与 `results/pair_circulation_20261011/` 已按原相对路径移至 `归档_20261030/03_第五章研究/实现/`。

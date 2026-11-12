# 门控交换检验 run01 结果（由 code/report.py 生成）

## 复现检查

- rl_full_env_B8 @ H12_local2_s20261105：步数 124 / 124，逐位相同：True，最大位置差 0.000e+00 m
- rl_full_int_env_B8 @ H12_local2_s20261105：步数 1201 / 1201，逐位相同：True，最大位置差 0.000e+00 m

## 各组到达（16 个场景）

| 组 | N=12 | N=20 | N=25 | 合计 | 切步 | 审计失败 | 未到达的场景 |
|---|---|---|---|---|---|---|---|
| rl_full 权重 + μ（训练时门控） | 6/6（中位 6.52 s） | 6/6（中位 11.48 s） | 4/4（中位 9.98 s） | 16/16 | 0 | 0 | — |
| rl_full 权重 + 修正量门控 | 6/6（中位 6.82 s） | 6/6（中位 9.35 s） | 4/4（中位 10.43 s） | 16/16 | 0 | 0 | — |
| rl_full 权重 + 门控恒为 1 | 6/6（中位 6.40 s） | 6/6（中位 10.03 s） | 4/4（中位 9.73 s） | 16/16 | 0 | 0 | — |
| rl_full_int 权重 + 修正量（训练时） | 4/6（中位 9.98 s） | 2/6（中位 32.83 s） | 0/4 | 6/16 | 170 | 0 | H12_local2（horizon_reached，1.07 m）, H12_local3（horizon_reached，1.03 m）, H20_local0（horizon_reached，1.00 m）, H20_local1（horizon_reached，1.00 m）, H20_local2（horizon_reached，1.01 m）, H20_local3（horizon_reached，1.00 m）, H25_local0（horizon_reached，1.04 m）, H25_local1（horizon_reached，1.00 m）, H25_local2（certification_stop，1.00 m）, H25_local3（horizon_reached，1.00 m） |
| rl_full_int 权重 + μ | 4/6（中位 10.35 s） | 2/6（中位 31.93 s） | 0/4 | 6/16 | 588 | 0 | H12_local2（horizon_reached，1.07 m）, H12_local3（horizon_reached，1.05 m）, H20_local0（horizon_reached，1.00 m）, H20_local1（horizon_reached，1.00 m）, H20_local2（horizon_reached，1.01 m）, H20_local3（horizon_reached，1.00 m）, H25_local0（horizon_reached，1.06 m）, H25_local1（horizon_reached，1.00 m）, H25_local2（certification_stop，1.00 m）, H25_local3（horizon_reached，1.00 m） |
| rl_geom 权重 + 修正量（训练时） | 6/6（中位 7.05 s） | 4/6（中位 24.35 s） | 1/4（中位 36.65 s） | 11/16 | 0 | 0 | H20_local0（horizon_reached，1.00 m）, H20_local1（horizon_reached，1.02 m）, H25_local1（horizon_reached，1.11 m）, H25_local2（stalled，1.01 m）, H25_local3（horizon_reached，1.02 m） |
| rl_geom 权重 + μ | 6/6（中位 6.55 s） | 6/6（中位 19.38 s） | 1/4（中位 29.15 s） | 13/16 | 0 | 0 | H25_local1（horizon_reached，1.09 m）, H25_local2（horizon_reached，1.01 m）, H25_local3（horizon_reached，1.02 m） |

## 登记的配对检验（同一权重，只换门控）

| A 对 B | 不一致对 A独到:B独到 | 双侧 p | 共同到达 | Δt 中位 (s) | Δt 均值 (s) | A 更快/更慢 |
|---|---|---|---|---|---|---|
| rl_full 权重 + μ（训练时门控） 对 rl_full 权重 + 修正量门控 | 0:0 | nan | 16 | -0.33 | -0.51 | 13/3 |
| rl_full_int 权重 + μ 对 rl_full_int 权重 + 修正量（训练时） | 0:0 | nan | 6 | -0.35 | -0.23 | 4/1 |
| rl_full 权重 + μ（训练时门控） 对 rl_full 权重 + 门控恒为 1 | 0:0 | nan | 16 | +0.10 | +0.20 | 4/11 |
| rl_geom 权重 + μ 对 rl_geom 权重 + 修正量（训练时） | 2:0 | 0.5 | 11 | -0.40 | -1.76 | 9/2 |


| A 对 B | 不一致对 A独到:B独到 | 双侧 p | 共同到达 | Δt 中位 (s) | Δt 均值 (s) | A 更快/更慢 |
|---|---|---|---|---|---|---|
| rl_full 权重 + 修正量门控 对 rl_full_int 权重 + 修正量（训练时） | 10:0 | 0.001953 | 6 | +0.12 | -0.47 | 3/3 |
| rl_full 权重 + μ（训练时门控） 对 rl_full_int 权重 + μ | 10:0 | 0.001953 | 6 | +0.17 | -1.02 | 3/3 |
| rl_full 权重 + 修正量门控 对 rl_geom 权重 + 修正量（训练时） | 5:0 | 0.0625 | 11 | -1.35 | -4.39 | 7/3 |
| rl_full 权重 + μ（训练时门控） 对 rl_geom 权重 + μ | 3:0 | 0.25 | 13 | -0.45 | -4.40 | 7/6 |

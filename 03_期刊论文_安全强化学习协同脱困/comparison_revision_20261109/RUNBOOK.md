# 外部比较复核与重运行

## 当前结果入口

- `external_report.md`：新96任务、两个训练种子、两种方法的完整汇总。
- `external_summary.json`：分规模和训练重复的统计。
- `external_verification.json`：384条轨迹的输入一致性、保持时长、审计覆盖及上游源码核验。
- `results_matched/`：既有384次匹配学习变体结果，完整保留两个种子和两种价值网络。
- `results_exploratory_corrected/`：修正保持时间和位置裁剪后的旧拉格朗日模型96次新评测。

此工作目录沿用2026-11-09启动日期，正式外部训练与审计于2026-11-10完成。

## 结果含义

官方DGPPO在本任务适配下没有学得满足约束的策略；本文方法重训也未保持原单检查点100%到达。结果用于说明这些配置在限定预算上的表现，不作为一般SOTA结论。完整差异见 `PROTOCOL.md`。

## 冻结依赖

上游仓库 `vendor/dgppo`，提交 `bf9ff80ebc7f135121b9fe1658800bdb106b8c20`，学习更新源文件没有修改。其许可保留在仓库内。

DGPPO运行使用本目录 `.venv_dgppo/Scripts/python.exe`（Python 3.12，CPU JAX）。环境包清单在 `requirements_dgppo_locked.txt`。原EGP-NPA评测使用 `D:/miniconda/python.exe`，只读导入工作区章节代码。GMP审计包安装在本目录 `gmp_runtime/`，不改变全局环境。

## 从保存数据复核

在期刊目录执行：

```powershell
python -B -X utf8 comparison_revision_20261109/summarize_matched.py
python -B -X utf8 comparison_revision_20261109/report_external.py
python -B -X utf8 comparison_revision_20261109/verify_external_comparison.py
python -B -X utf8 -m unittest discover -s comparison_revision_20261109 -p test_evaluation.py
python -B -X utf8 -m unittest discover -s comparison_revision_20261109 -p test_exact_arithmetic.py
& '.\comparison_revision_20261109\.venv_dgppo\Scripts\python.exe' -B -X utf8 -m unittest discover -s comparison_revision_20261109 -p test_dgppo_adapter.py
.\build.ps1
```

统计脚本只更新派生报告，不覆盖原模型和轨迹。训练与闭环评测输出目录必须尚不存在。

## 训练入口

以下命令示例必须指定新的输出目录；正式结果已经保存，不必为查看结果重新训练。

```powershell
& '.\comparison_revision_20261109\.venv_dgppo\Scripts\python.exe' -B -X utf8 comparison_revision_20261109/run_official_dgppo.py --mixed --seed 20261107 --budget 480000 --horizon-steps 500 --environments 4 --output comparison_revision_20261109/new_dgppo_run
```

训练代码计入上游更新内部的额外确定性展开，因此120次更新×(4回合×500步×2类展开)=480000步。大小不同的图共享网络参数与优化器状态，使用无RNN选项。最终预算权重为 `models/480000/`。

正式EGP-NPA模型身份在 `egp_final_models.json`，只使用训练最终 `last.pt`，而非按本轮测试选择的最优权重。两个模型原始训练步数分别为481990与484821。

## 评测与审计

```powershell
& '.\comparison_revision_20261109\.venv_dgppo\Scripts\python.exe' -B -X utf8 comparison_revision_20261109/evaluate_official_dgppo.py --training comparison_revision_20261109/new_dgppo_run --output comparison_revision_20261109/new_dgppo_eval
python -B -X utf8 comparison_revision_20261109/evaluate_external_common.py --audit-only --output comparison_revision_20261109/new_dgppo_eval
```

两方法使用 `external_test/scenes.json` 同一输入。DGPPO无护盾执行；EGP-NPA安全分配与在线认证是其控制流程的一部分。所有原始状态、时间及DGPPO命令保存到NPZ，审计与控制分离。

最终完整审计不在最初运行目录的未完成汇总中，而在 `external_audit_egp_s1/`、`external_audit_v2_egp_s2/`、`external_audit_v2_dgppo_s1/`、`external_audit_v2_dgppo_s2/`。切换精确运算后端时只停止了重复的事后审计，所有384次闭环轨迹均先完整生成；原前缀保留。

## 适配历史

`official_dgppo_s1/s2` 是128步窗口适配；`official_dgppo_h500_s1/s2` 是尚未恢复官方代价裕度处理的500步适配。最终正式版本为 `official_dgppo_h500_margin_s1/s2`。两项修订来自原代码和训练窗口核对，均在DGPPO最终新任务性能测试开始之前确定，不根据测试结果调参。所有版本及失败结果均保留，正式版本的独立冻结源码在各运行 `source/` 下。

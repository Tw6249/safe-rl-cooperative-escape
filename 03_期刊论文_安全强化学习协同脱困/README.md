# 独立期刊论文：耦合安全约束下多机器人协同脱困的包络等变图强化学习与邻域近端分配控制

本目录为完全独立的双栏期刊论文工程，聚焦于**全局耦合且相互冲突的标量安全约束（全局标量连通保持 $h_0(\boldsymbol{p})=\lambda_2(\boldsymbol{L}_{\mathrm{p}}(\boldsymbol{p}))-\lambda_{\mathrm{req}}\ge 0$ 与局部成对机间避碰 $h_{ij}^{\mathrm{col}}(\boldsymbol{p})=d_{ij}^2-d_{\mathrm{safe}}^2\ge 0$）下的多机器人防死锁与协同脱困控制**，不涉及重根特征值或子空间谱展开。

## 一、 核心入口与双版本成稿

- **中文控制顶刊双栏版（对标《自动化学报》/《控制理论与应用》）**：
  - 模板与主文件：[cjas_journal.cls](cjas_journal.cls)、[main_zh.tex](main_zh.tex)（章节位于 `sections_zh/01_introduction.tex`..`06_conclusion.tex`）
  - 编译产物：[main_zh.pdf](main_zh.pdf)（同步生成 [main.pdf](main.pdf)，20 页，`Overfull \hbox = 0, Warnings = 0`）
- **英文 IEEE Transactions 双栏版（对标 *IEEE T-RO* / *IEEE TAC*）**：
  - 模板与主文件：`IEEEtran.cls`、[main_en.tex](main_en.tex)（章节位于 `sections_en/01_introduction.tex`..`06_conclusion.tex`）
  - 编译产物：[main_en.pdf](main_en.pdf)（17 页，`Overfull \hbox = 0, Warnings = 0`）

## 二、 标量耦合约束建模与对比实验体系

| 模块 / 对比方法 | 核心设置与实测结果 |
|---|---|
| **标量耦合安全约束建模（第 2 节）** | 采用单重特征值下的全局标量连通障碍函数 $h_0(\boldsymbol{p})$ 与 $P=N(N-1)/2$ 个成对避碰标量障碍函数 $h_{ij}^{\mathrm{col}}(\boldsymbol{p})$，直接定义标量仿射复合安全输入集 $\mathcal{K}_0(\boldsymbol{p})$、定理 1（零阶保持整段连续时间安全保证）及闭邻域块 NPA 线性化不等式组 |
| **EGP-NPA（本文方法）** | $96$ 组主基准（$N=8$ 共 $64$ 组 + $N\in\{12,20,25\}$ 零样本跨规模 $32$ 组）与 $48$ 组独立盲测均取得 $100\%$ 安全达标、$0$ 步长减半与 $0$ 审计违约；$4.8\times 10^5$ 步减半预算双种子重训在新建 $96$ 组测试集上达标 $93/96$ 与 $92/96$ |
| **护盾投影一致性强化学习（`Proj-MAPPO + NPA`，$B=8$）** | 共享两层等变 GNN、完成包络与 $B=8$ 护盾，加入投影一致性正则 $\mathcal{L}_{\mathrm{proj}}$；主基准 $N=8$ 达标 $64/64$（中位慢 $2.88\ \mathrm{s}$），跨规模达标 $17/32\ (53.1\%)$；同预算双种子 $2\times 2$ 析因实验（$384$ 次运行）共同达标中位慢 $+2.00\ \mathrm{s}\ (p=6.2\times 10^{-9})$ |
| **静态配额平摊 CBF 护盾图策略（`Static-Split GNN`，$B=0$）** | 共享等变 GNN 策略与完成包络，底层采用单机静态配额平摊（$B=0, s_{ia,k}\equiv 0$）；$N=8$ 达标 $11/64\ (17.2\%)$，跨规模达标 $1/32\ (3.1\%)$，完整记录零速停滞、步长减半停止与调用预算终止 |
| **离散图障碍函数近端策略优化（`DGPPO`，官方实现任务适配）** | 调用官方 MIT-REALM `DGPPO` 算法，适配至一阶积分器及三项标量耦合安全约束（$500$ 步窗口、$4.8\times 10^5$ 环境转移、双训练种子）；无在线耦合 QP 护盾下在全部 $96$ 组场景上发生毫米级净空交会碰撞（无违约安全达标 $0\%$） |
| **拉格朗日约束强化学习（`MAPPO-Lagrangian`，无硬护盾）** | 原始—对偶自适应乘子训练，按严格 $1.0\ \mathrm{s}$ 保持判据与无裁剪植物评测；名义到达 $96/96$，但全部运行均发生碰撞穿透（合计 $10\,971$ 个违约段，安全达标 $0\%$） |

## 三、 一键绘图与编译命令

在本目录下运行：

```powershell
# 仅编译中英文 PDF 并校验零 Overfull \hbox 与零 Warning
.\build.ps1

# 重新生成全部 6 幅中文矢量图与 6 幅英文矢量图并编译
.\build.ps1 -Figures
```

新增源码和结果全部位于 [comparison_revision_20261109](comparison_revision_20261109/)。该目录名沿用启动日期；工作跨越至2026-11-10。修订前的论文源文件、PDF和图件保存在 `comparison_revision_20261109/manuscript_before_revision/`。原实验结果和冻结模型未覆盖。

## 外部算法实现状态

官方来源：https://github.com/MIT-REALM/dgppo 。上游源码及许可在 `comparison_revision_20261109/vendor/dgppo/`，未改动其学习更新。

使用本地隔离环境运行适配测试：

```powershell
& '.\comparison_revision_20261109\.venv_dgppo\Scripts\python.exe' -B -X utf8 -m unittest discover -s comparison_revision_20261109 -p test_dgppo_adapter.py
```

初期接口验证在 `dgppo_smoke01/`、`dgppo_smoke02/` 和 `dgppo_mixed_smoke03/`，不用于性能结论。最终采用N=6..10任务库、500步展开窗口和官方约束代价变换，两个正式模型分别保存于 `official_dgppo_h500_margin_s1/`、`official_dgppo_h500_margin_s2/`。每个模型计入随机和内部确定性展开共480000次环境转移，选用最终预算检查点；对应EGP-NPA采用相同两个训练种子的 `last.pt`，实际481990/484821步。新测试场景保存在 `external_test/`，不用于选模或调参。

四组各96条轨迹保存在 `external_egp_s1/`、`external_egp_s2/`、`external_dgppo_s1/`、`external_dgppo_s2/`。最终完整独立审计目录为 `external_audit_egp_s1/`、`external_audit_v2_egp_s2/`、`external_audit_v2_dgppo_s1/`、`external_audit_v2_dgppo_s2/`；其他审计前缀保留作历史记录。使用精确GMP有理数运算加速同一证书，碰撞短路仅在精确证明不安全时返回失败，等价性检查见 `test_exact_arithmetic.py`。原数据不覆盖。

结果：EGP-NPA两次为93/96、92/96安全到达（跨规模29/32、28/32）；DGPPO任务适配两次为0/96，均未名义到达并发生碰撞。DGPPO在本配置与预算下未学得满足任务要求的策略，应作为适配负面结果，不支持“优于DGPPO的一般结论”。双方观测、网络、原生奖励、有限库与在线采样、前期调参投入不同，详见协议和正文外部比较小节。仅两个训练重复不作算法级显著性检验。

复核新增比较：

```powershell
python -B -X utf8 comparison_revision_20261109/report_external.py
python -B -X utf8 comparison_revision_20261109/verify_external_comparison.py
```

训练和评测命令、适配历史及重现说明见 [RUNBOOK.md](comparison_revision_20261109/RUNBOOK.md)。此前128步窗口以及未保留官方训练代价变换的500步训练均单独保留，未进行新测试性能选择，不进入正式表格。

历史拉格朗日模型已经按1秒保持和无位置裁剪规则重新运行96个场景，输出在 `results_exploratory_corrected/`；仍为0安全到达，有理数审计失败10971段。原来10929段的记录完整保留。新旧数字差别来自实际执行长度改变及状态积分，不对旧证据回填。

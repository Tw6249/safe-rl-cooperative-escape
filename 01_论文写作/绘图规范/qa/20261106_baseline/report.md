# 图件规范 3.0 检查报告

- 生成时间：2026-11-06T08:30:41
- 命令：`python check_figures.py --layout-map qa/layout_map_20261106.json --pdf ../整篇论文/build/main.pdf --pages --out qa/20261106_baseline`
- 检查的 PDF：`C:\Users\TanWei\Desktop\phd-thesis\01_论文写作\整篇论文\build\main.pdf`（修改时间 2026-11-06T07:42:05，页码偏移 20）
- 构图代号来源：tex = figure 内 `% layout=`；pdf = 图件 PDF 关键字；map = `--layout-map` 人工判读

## 汇总

ERROR 104 条，WARN 76 条。代码后 `!` 为 ERROR，`?` 为 WARN。

| 代码 | 含义 | ERROR | WARN |
|---|---|---:|---:|
| F01 | 位置说明符 | 11 | 0 |
| F02 | 插入缩放 | 10 | 0 |
| F03 | 画布宽度 | 8 | 5 |
| F04 | 画布高度与档位 | 18 | 13 |
| F05 | 每排子图数 | 7 | 0 |
| F06 | 字号下限 | 17 | 0 |
| F07 | 字体 | 5 | 0 |
| F08 | 位图分辨率 | 0 | 2 |
| F09 | 白边与偏心 | 0 | 3 |
| F10 | 题注 | 0 | 18 |
| F11 | 标签 | 0 | 24 |
| F13 | TikZ | 0 | 2 |
| F14 | 同章同档高度不一 | 0 | 6 |
| C01 | placeins[section] | 5 | 0 |
| C02 | 章内浮动参数 | 12 | 0 |
| P01 | 图与首次引用的页距 | 0 | 3 |
| P02 | 页底空白 | 5 | 0 |
| P03 | 页顶空白/孤立浮动页 | 1 | 0 |
| P04 | 图表溢出版心 | 5 | 0 |

## 逐图

| 图 | 标签 | 构图 | 画布 mm | 缩放 | 成品 mm | 字号 中位/最小正文 | <7.8pt | 问题 |
|---|---|---|---|---:|---|---|---:|---|
| 1.1 | `fig:ch1-loop` | D（map） | TikZ |  |  |  |  | F13? |
| 1.2 | `fig:ch1-roadmap` | D（map） | TikZ |  |  |  |  | F13? |
| 2.1 | `fig:ch2-gain-threshold` | P2（map） | 150×62 | 1.0 | 150×62 | 8.8/8.5 | 0% | F04? F10? |
| 2.2 | `fig:weak_connection` | P4（map） | 150×105 | 1.0 | 150×105 | 9.5/8.2 | 0% | F04! F11? |
| 2.3 | `fig:information_flow` | D（map） | 150×52 | 1.0 | 150×52 | 8.8/8.8 | 1% | F06! F11? |
| 2.4 | `fig:ch2-reference-plane` | P2E（map） | 150×68 | 1.0 | 150×68 | 9.5/8.5 | 0% | F04? F09? F10? |
| 2.5 | `fig:reference_trajectory` | P4（map） | 150×112 | 1.0 | 150×112 | 8.0/8.0 | 0% | F04! F11? |
| 2.6 | `fig:full_dac` | P4（map） | 154×114 | 0.974 | 150×111 | 7.8/7.8 | 26% | F02! F03! F04! F06! F10? F11? |
| 2.7 | `fig:gain_scan` | P2（map） | 150×62 | 1.0 | 150×62 | 7.5/8.8 | 30% | F04? F06! F11? |
| 2.8 | `fig:switch_topology` | P2E（map） | 150×50 | 1.0 | 150×50 | 9.5/8.2 | 0% | F04! F11? |
| 2.9 | `fig:switch_comparison` | P4（map） | 150×118 | 1.0 | 150×118 | 8.8/8.4 | 0% | F04? F11? |
| 2.10 | `fig:comparison_failure` | P2（map） | 150×64 | 1.0 | 150×64 | 8.8/8.0 | 0% | F04! F11? |
| 3.1 | `fig:ch3-coupling` | P4（map） | 150×128 | 1.0 | 150×128 | 9.5/8.0 | 0% | F04! F10? |
| 3.2 | `fig:ch3-persistent` | 1x3-data（map） | 150×52 | 1.0 | 150×52 | 8.5/8.0 | 0% | F05! |
| 3.3 | `fig:ch3-joint` | 1x3-data（map） | 150×54 | 1.0 | 150×54 | 8.5/8.0 | 0% | F05! |
| 3.4 | `fig:ch3mechanism` | D（map） | 150×96 | 1.0 | 150×96 | 7.0/8.0 | 0% | F04! F09? F11? |
| 3.5 | `fig:ch3convergence` | P2（map） | 150×62 | 1.0 | 150×62 | 7.2/8.8 | 5% | F04? F06! F11? P01? |
| 3.6 | `fig:ch3scene` | P2E（map） | 150×74 | 1.0 | 150×74 | 9.5/8.2 | 0% | F04! F11? |
| 3.7 | `fig:ch3anytime` | P2（map） | 150×84 | 1.0 | 150×84 | 8.8/8.8 | 23% | F04! F06! F11? |
| 3.8 | `fig:ch3comparison` | V3（map） | 150×148 | 0.84 | 126×124 | 7.4/8.0 | 86% | F01! F02! F04? F06! F11? |
| 3.9 | `fig:ch3switching` | P1W（map） | 150×72 | 1.0 | 150×72 | 7.5/8.8 | 68% | F04! F06! F11? |
| 3.10 | `fig:ch3interruptions` | P2（map） | 150×76 | 1.0 | 150×76 | 8.8/8.8 | 49% | F04! F06! F11? |
| 3.11 | `fig:ch3closedloop` | P3（map） | 150×88 | 0.98 | 147×86 | 8.6/7.8 | 32% | F02! F04? F06! F11? |
| 4.1 | `fig:ch4-rectangle` | 3+2（map） | 150×112 | 1.0 | 150×112 | 8.8/8.0 | 0% | F05! F10? |
| 4.2 | `fig:ch4-cone` | P2E（map） | 150×68 | 1.0 | 150×68 | 9.5/8.5 | 0% | F04? F09? |
| 4.3 | `fig:c4crossing` | V2（map） | 150×102 | 1.0 | 150×102 | 8.8/8.4 | 0% | F04? F11? |
| 4.4 | `fig:c4topmargins` | V2（map） | 153×105 | 0.981 | 150×103 | 8.6/8.0 | 0% | F03? F04? F11? |
| 4.5 | `fig:c4geometry` | P2E（map） | 153×67 | 0.981 | 150×66 | 9.3/8.0 | 0% | F03? F11? |
| 4.6 | `fig:c4recovery` | P4（map） | 150×104 | 1.0 | 150×104 | 7.2/7.8 | 18% | F04! F06! F10? F11? |
| 4.7 | `fig:c4pipeline` | P4（map） | 153×119 | 0.981 | 150×117 | 9.3/8.3 | 0% | F03? F04? F11? P01? |
| 4.8 | `fig:c4paclosedloop` | P4（map） | 153×123 | 0.843 | 129×104 | 5.7/8.0 | 14% | F02! F03! F04! F06! F10? F11? |
| 4.9 | `fig:c4blocksettle` | P4（map） | 153×103 | 0.843 | 129×87 | 8.0/8.0 | 35% | F02! F03! F04! F06! F11? |
| 5.1 | `fig:seven5` | P2E（map） | 149×73 | 1.01 | 150×74 | 9.6/8.9 | 0% | F01! F03? F04! F11? P01? |
| 5.2 | `fig:ch5-swap-traj` | P4（map） | 150×122 | 1.0 | 150×122 | 9.5/8.3 | 0% | F01! F04! F10? |
| 5.3 | `fig:ch5-time-evol` | P4（map） | 150×118 | 1.0 | 150×118 | 9.5/8.4 | 0% | F01! F04? |
| 5.4 | `fig:ch5-perm-gate` | P2E（map） | 150×68 | 1.0 | 150×68 | 8.8/8.0 | 0% | F01! F04? |
| 5.5 | `fig:ch5-npa-rounds` | 1x3-data（map） | 153×57 | 0.981 | 150×56 | 9.3/8.6 | 0% | F01! F03? F05! |
| 6.1 | `fig:ch6-arch-3d` | 2x3-data（map） | 338×191 | 0.443 | 150×85 | 4.7/None | 100% | F01! F02! F03! F05! F06! F07! F08? F10? |
| 6.2 | `fig:ch6-exp1` | 2x3-data（map） | 338×176 | 0.443 | 150×78 | 4.4/None | 100% | F01! F02! F03! F05! F06! F07! F10? |
| 6.3 | `fig:ch6-exp2` | P4（map） | 313×176 | 0.479 | 150×84 | 3.8/None | 100% | F01! F02! F03! F04! F06! F07! F10? |
| 6.4 | `fig:ch6-exp3` | P4（map） | 318×176 | 0.472 | 150×83 | 3.6/None | 100% | F01! F02! F03! F04! F06! F07! F10? |
| 6.5 | `fig:ch6-exp4-hetero` | 2x3-data（map） | 354×194 | 0.424 | 150×82 | 3.9/None | 100% | F01! F02! F03! F05! F06! F07! F08? F10? |

## 明细

### ERROR（104）

**F01 位置说明符**
- 图3.8 fig:ch3comparison：位置说明符为 [!htbp]，应为 [htbp]（独立章节/第三章_持久分配/sections/07_experiments.tex:66）
- 图5.1 fig:seven5：位置说明符为 [tb]，应为 [htbp]（独立章节/第五章_安全强化学习与协同脱困/sections/03_obstruction.tex:69）
- 图5.2 fig:ch5-swap-traj：位置说明符为 [!htbp]，应为 [htbp]（独立章节/第五章_安全强化学习与协同脱困/sections/05_experiments.tex:17）
- 图5.3 fig:ch5-time-evol：位置说明符为 [!htbp]，应为 [htbp]（独立章节/第五章_安全强化学习与协同脱困/sections/05_experiments.tex:22）
- 图5.4 fig:ch5-perm-gate：位置说明符为 [!htbp]，应为 [htbp]（独立章节/第五章_安全强化学习与协同脱困/sections/05_experiments.tex:69）
- 图5.5 fig:ch5-npa-rounds：位置说明符为 [!htbp]，应为 [htbp]（独立章节/第五章_安全强化学习与协同脱困/sections/05_experiments.tex:110）
- 图6.1 fig:ch6-arch-3d：位置说明符为 [!htbp]，应为 [htbp]（独立章节/第六章_虚拟物理仿真与系统验证/sections/02_platform_model.tex:45）
- 图6.2 fig:ch6-exp1：位置说明符为 [!htbp]，应为 [htbp]（独立章节/第六章_虚拟物理仿真与系统验证/sections/05_mujoco_experiments.tex:39）
- 图6.3 fig:ch6-exp2：位置说明符为 [!htbp]，应为 [htbp]（独立章节/第六章_虚拟物理仿真与系统验证/sections/05_mujoco_experiments.tex:86）
- 图6.4 fig:ch6-exp3：位置说明符为 [!htbp]，应为 [htbp]（独立章节/第六章_虚拟物理仿真与系统验证/sections/05_mujoco_experiments.tex:123）
- 图6.5 fig:ch6-exp4-hetero：位置说明符为 [!htbp]，应为 [htbp]（独立章节/第六章_虚拟物理仿真与系统验证/sections/06_hetero_air_ground.tex:119）

**F02 插入缩放**
- 图2.6 fig:full_dac：插入缩放比 0.974（画布 154 mm 宽被压到 150 mm），应为 1:1
- 图3.8 fig:ch3comparison：按系数 0.84\textwidth 插入；应在出图时按档位重出，不在 LaTeX 中缩放
- 图3.11 fig:ch3closedloop：按系数 0.98\textwidth 插入；应在出图时按档位重出，不在 LaTeX 中缩放
- 图4.8 fig:c4paclosedloop：按系数 0.86\textwidth 插入；应在出图时按档位重出，不在 LaTeX 中缩放
- 图4.9 fig:c4blocksettle：按系数 0.86\textwidth 插入；应在出图时按档位重出，不在 LaTeX 中缩放
- 图6.1 fig:ch6-arch-3d：插入缩放比 0.443（画布 338 mm 宽被压到 150 mm），应为 1:1
- 图6.2 fig:ch6-exp1：插入缩放比 0.443（画布 338 mm 宽被压到 150 mm），应为 1:1
- 图6.3 fig:ch6-exp2：插入缩放比 0.479（画布 313 mm 宽被压到 150 mm），应为 1:1
- 图6.4 fig:ch6-exp3：插入缩放比 0.472（画布 318 mm 宽被压到 150 mm），应为 1:1
- 图6.5 fig:ch6-exp4-hetero：插入缩放比 0.424（画布 353 mm 宽被压到 150 mm），应为 1:1

**F03 画布宽度**
- 图2.6 fig:full_dac：画布宽 154.0 mm，应为 150 或 105 mm
- 图4.8 fig:c4paclosedloop：画布宽 153.0 mm，应为 150 或 105 mm
- 图4.9 fig:c4blocksettle：画布宽 153.0 mm，应为 150 或 105 mm
- 图6.1 fig:ch6-arch-3d：画布宽 338.2 mm，应为 150 或 105 mm
- 图6.2 fig:ch6-exp1：画布宽 338.2 mm，应为 150 或 105 mm
- 图6.3 fig:ch6-exp2：画布宽 312.8 mm，应为 150 或 105 mm
- 图6.4 fig:ch6-exp3：画布宽 317.9 mm，应为 150 或 105 mm
- 图6.5 fig:ch6-exp4-hetero：画布宽 353.5 mm，应为 150 或 105 mm

**F04 画布高度与档位**
- 图2.2 fig:weak_connection：P4 档标称高 116 mm，成品 105.0 mm（差 -11.0）
- 图2.5 fig:reference_trajectory：P4 档标称高 116 mm，成品 112.0 mm（差 -4.0）
- 图2.6 fig:full_dac：P4 档标称高 116 mm，成品 111.0 mm（差 -5.0）
- 图2.8 fig:switch_topology：P2E 档标称高 66 mm，成品 50.0 mm（差 -16.0）
- 图2.10 fig:comparison_failure：P2 档标称高 58 mm，成品 64.0 mm（差 +6.0）
- 图3.1 fig:ch3-coupling：P4 档标称高 116 mm，成品 128.0 mm（差 +12.0）
- 图3.4 fig:ch3mechanism：D 档标称高 100 mm，成品 96.0 mm（差 -4.0）
- 图3.6 fig:ch3scene：P2E 档标称高 66 mm，成品 74.0 mm（差 +8.0）
- 图3.7 fig:ch3anytime：P2 档标称高 58 mm，成品 84.0 mm（差 +26.0）
- 图3.9 fig:ch3switching：P1W 档标称高 78 mm，成品 72.0 mm（差 -6.0）
- 图3.10 fig:ch3interruptions：P2 档标称高 58 mm，成品 76.0 mm（差 +18.0）
- 图4.6 fig:c4recovery：P4 档标称高 116 mm，成品 104.0 mm（差 -12.0）
- 图4.8 fig:c4paclosedloop：P4 档标称高 116 mm，成品 103.7 mm（差 -12.3）
- 图4.9 fig:c4blocksettle：P4 档标称高 116 mm，成品 86.8 mm（差 -29.2）
- 图5.1 fig:seven5：P2E 档标称高 66 mm，成品 73.7 mm（差 +7.7）
- 图5.2 fig:ch5-swap-traj：P4 档标称高 116 mm，成品 122.0 mm（差 +6.0）
- 图6.3 fig:ch6-exp2：P4 档标称高 116 mm，成品 84.2 mm（差 -31.8）
- 图6.4 fig:ch6-exp3：P4 档标称高 116 mm，成品 82.9 mm（差 -33.1）

**F05 每排子图数**
- 图3.2 fig:ch3-persistent：一排 3 个带数据坐标轴的子图；数据子图每排最多 2 个，改用 P3/V3 或拆图
- 图3.3 fig:ch3-joint：一排 3 个带数据坐标轴的子图；数据子图每排最多 2 个，改用 P3/V3 或拆图
- 图4.1 fig:ch4-rectangle：首排 3 个带坐标轴的子图；数据子图每排最多 2 个，改用 P3/V3 或拆图
- 图5.5 fig:ch5-npa-rounds：一排 3 个带数据坐标轴的子图；数据子图每排最多 2 个，改用 P3/V3 或拆图
- 图6.1 fig:ch6-arch-3d：2×3 排布且含带数据坐标轴的子图；数据子图每排最多 2 个，改用 P3/V3 或拆图
- 图6.2 fig:ch6-exp1：2×3 排布且含带数据坐标轴的子图；数据子图每排最多 2 个，改用 P3/V3 或拆图
- 图6.5 fig:ch6-exp4-hetero：2×3 排布且含带数据坐标轴的子图；数据子图每排最多 2 个，改用 P3/V3 或拆图

**F06 字号下限**
- 图2.3 fig:information_flow：1/69 处文字（1%）低于 7.8 pt：7 pt
- 图2.6 fig:full_dac：62/239 处文字（26%）低于 7.8 pt：5, 5.2, 7.2, 7.3, 7.4 pt
- 图2.7 fig:gain_scan：48/161 处文字（30%）低于 7.8 pt：5.2, 7.5 pt
- 图3.5 fig:ch3convergence：8/146 处文字（5%）低于 7.8 pt：7.2 pt
- 图3.7 fig:ch3anytime：30/128 处文字（23%）低于 7.8 pt：5.3, 7.4, 7.6 pt
- 图3.8 fig:ch3comparison：94/109 处文字（86%）低于 7.8 pt：4.7, 5.2, 6.7, 7.4 pt
- 图3.9 fig:ch3switching：86/127 处文字（68%）低于 7.8 pt：3.7, 5.2, 7.5 pt
- 图3.10 fig:ch3interruptions：33/68 处文字（49%）低于 7.8 pt：5.1, 7.3 pt
- 图3.11 fig:ch3closedloop：60/187 处文字（32%）低于 7.8 pt：7.4, 7.6 pt
- 图4.6 fig:c4recovery：70/392 处文字（18%）低于 7.8 pt：7.2 pt
- 图4.8 fig:c4paclosedloop：47/346 处文字（14%）低于 7.8 pt：5.2, 7.4 pt
- 图4.9 fig:c4blocksettle：45/130 处文字（35%）低于 7.8 pt：6.1, 7.4 pt
- 图6.1 fig:ch6-arch-3d：235/235 处文字（100%）低于 7.8 pt：3.3, 3.6, 4.2, 4.4, 4.7 pt
- 图6.2 fig:ch6-exp1：379/379 处文字（100%）低于 7.8 pt：2.4, 2.9, 3.1, 3.5, 4.2, 4.4 pt
- 图6.3 fig:ch6-exp2：404/404 处文字（100%）低于 7.8 pt：2.6, 2.7, 3.2, 3.7, 3.8, 4.6, 4.8, 4.9 pt
- 图6.4 fig:ch6-exp3：575/575 处文字（100%）低于 7.8 pt：2.5, 2.6, 3.1, 3.4, 3.6, 3.8, 4.5, 4.7, 4.8 pt
- 图6.5 fig:ch6-exp4-hetero：524/524 处文字（100%）低于 7.8 pt：2, 2.2, 2.3, 2.8, 3, 3.1, 3.2, 3.9, 4, 4.2 pt

**F07 字体**
- 图6.1 fig:ch6-arch-3d：使用禁用字体：DejaVuSans, DejaVuSans-BoldOblique, DejaVuSans-Oblique, MicrosoftYaHei, MicrosoftYaHei-Bold
- 图6.2 fig:ch6-exp1：使用禁用字体：DejaVuSans, DejaVuSans-BoldOblique, DejaVuSans-Oblique, MicrosoftYaHei, MicrosoftYaHei-Bold
- 图6.3 fig:ch6-exp2：使用禁用字体：DejaVuSans, DejaVuSans-Oblique, MicrosoftYaHei, MicrosoftYaHei-Bold
- 图6.4 fig:ch6-exp3：使用禁用字体：DejaVuSans, DejaVuSans-BoldOblique, DejaVuSans-Oblique, MicrosoftYaHei, MicrosoftYaHei-Bold
- 图6.5 fig:ch6-exp4-hetero：使用禁用字体：DejaVuSans, DejaVuSans-BoldOblique, DejaVuSans-Oblique, MicrosoftYaHei, MicrosoftYaHei-Bold

**C01 placeins[section]**
- 第2章：独立章节/第二章_分布式谱估计/packages.tex:2 加载 placeins[section]，会在每节前强制排完浮动体
- 第3章：独立章节/第三章_持久分配/packages.tex:5 加载 placeins[section]，会在每节前强制排完浮动体
- 第4章：独立章节/第四章_子空间连通安全/packages.tex:3 加载 placeins[section]，会在每节前强制排完浮动体
- 第5章：独立章节/第五章_安全强化学习与协同脱困/packages.tex:2 加载 placeins[section]，会在每节前强制排完浮动体
- 第6章：独立章节/第六章_虚拟物理仿真与系统验证/packages.tex:2 加载 placeins[section]，会在每节前强制排完浮动体

**C02 章内浮动参数**
- 第5章：独立章节/第五章_安全强化学习与协同脱困/settings.tex:25 在章内覆盖浮动参数：\renewcommand{\topfraction}{0.93}
- 第5章：独立章节/第五章_安全强化学习与协同脱困/settings.tex:26 在章内覆盖浮动参数：\renewcommand{\bottomfraction}{0.85}
- 第5章：独立章节/第五章_安全强化学习与协同脱困/settings.tex:27 在章内覆盖浮动参数：\renewcommand{\textfraction}{0.06}
- 第5章：独立章节/第五章_安全强化学习与协同脱困/settings.tex:28 在章内覆盖浮动参数：\renewcommand{\floatpagefraction}{0.75}
- 第5章：独立章节/第五章_安全强化学习与协同脱困/settings.tex:29 在章内覆盖浮动参数：\setcounter{topnumber}{3}
- 第5章：独立章节/第五章_安全强化学习与协同脱困/settings.tex:30 在章内覆盖浮动参数：\setcounter{totalnumber}{5}
- 第6章：独立章节/第六章_虚拟物理仿真与系统验证/settings.tex:28 在章内覆盖浮动参数：\renewcommand{\topfraction}{0.93}
- 第6章：独立章节/第六章_虚拟物理仿真与系统验证/settings.tex:29 在章内覆盖浮动参数：\renewcommand{\bottomfraction}{0.85}
- 第6章：独立章节/第六章_虚拟物理仿真与系统验证/settings.tex:30 在章内覆盖浮动参数：\renewcommand{\textfraction}{0.06}
- 第6章：独立章节/第六章_虚拟物理仿真与系统验证/settings.tex:31 在章内覆盖浮动参数：\renewcommand{\floatpagefraction}{0.75}
- 第6章：独立章节/第六章_虚拟物理仿真与系统验证/settings.tex:32 在章内覆盖浮动参数：\setcounter{topnumber}{3}
- 第6章：独立章节/第六章_虚拟物理仿真与系统验证/settings.tex:33 在章内覆盖浮动参数：\setcounter{totalnumber}{5}

**P02 页底空白**
- p17：页底空白 55 mm（非章末页）
- p24：页底空白 37 mm（非章末页）
- p58：页底空白 91 mm（非章末页）
- p74：页底空白 34 mm（非章末页）
- p127：页底空白 35 mm（非章末页）

**P03 页顶空白/孤立浮动页**
- p18：页顶空白 68 mm：疑似只排一张图的孤立浮动页

**P04 图表溢出版心**
- 独立章节/第五章_安全强化学习与协同脱困/sections/05_experiments.tex:42：表格超出版心 20.0 mm（in alignment）
- 独立章节/第五章_安全强化学习与协同脱困/sections/05_experiments.tex:56：表格超出版心 29.3 mm（in alignment）
- 独立章节/第五章_安全强化学习与协同脱困/sections/05_experiments.tex:79：表格超出版心 44.5 mm（in alignment）
- 独立章节/第六章_虚拟物理仿真与系统验证/sections/06_hetero_air_ground.tex:87：表格超出版心 27.2 mm（in alignment）
- 独立章节/第六章_虚拟物理仿真与系统验证/sections/06_hetero_air_ground.tex:108：表格超出版心 74.9 mm（in alignment）

### WARN（76）

**F03 画布宽度**
- 图4.4 fig:c4topmargins：画布宽 153.0 mm，应为 150 或 105 mm
- 图4.5 fig:c4geometry：画布宽 153.0 mm，应为 150 或 105 mm
- 图4.7 fig:c4pipeline：画布宽 153.0 mm，应为 150 或 105 mm
- 图5.1 fig:seven5：画布宽 148.6 mm，应为 150 或 105 mm
- 图5.5 fig:ch5-npa-rounds：画布宽 153.0 mm，应为 150 或 105 mm

**F04 画布高度与档位**
- 图2.1 fig:ch2-gain-threshold：P2 档标称高 58 mm，成品 62.0 mm，在存量容差内；修改时回到标称值
- 图2.4 fig:ch2-reference-plane：P2E 档标称高 66 mm，成品 68.0 mm，在存量容差内；修改时回到标称值
- 图2.7 fig:gain_scan：P2 档标称高 58 mm，成品 62.0 mm，在存量容差内；修改时回到标称值
- 图2.9 fig:switch_comparison：P4 档标称高 116 mm，成品 118.0 mm，在存量容差内；修改时回到标称值
- 图3.5 fig:ch3convergence：P2 档标称高 58 mm，成品 62.0 mm，在存量容差内；修改时回到标称值
- 图3.8 fig:ch3comparison：V3 档标称高 128 mm，成品 124.3 mm，在存量容差内；修改时回到标称值
- 图3.11 fig:ch3closedloop：P3 档标称高 88 mm，成品 86.2 mm，在存量容差内；修改时回到标称值
- 图4.2 fig:ch4-cone：P2E 档标称高 66 mm，成品 68.0 mm，在存量容差内；修改时回到标称值
- 图4.3 fig:c4crossing：V2 档标称高 100 mm，成品 102.0 mm，在存量容差内；修改时回到标称值
- 图4.4 fig:c4topmargins：V2 档标称高 100 mm，成品 102.9 mm，在存量容差内；修改时回到标称值
- 图4.7 fig:c4pipeline：P4 档标称高 116 mm，成品 116.7 mm，在存量容差内；修改时回到标称值
- 图5.3 fig:ch5-time-evol：P4 档标称高 116 mm，成品 118.0 mm，在存量容差内；修改时回到标称值
- 图5.4 fig:ch5-perm-gate：P2E 档标称高 66 mm，成品 68.0 mm，在存量容差内；修改时回到标称值

**F08 位图分辨率**
- 图6.1 fig:ch6-arch-3d：内嵌位图成品分辨率最低 225 ppi；渲染帧/照片须 ≥ 300，数据热图可说明后保留
- 图6.5 fig:ch6-exp4-hetero：内嵌位图成品分辨率最低 236 ppi；渲染帧/照片须 ≥ 300，数据热图可说明后保留

**F09 白边与偏心**
- 图2.4 fig:ch2-reference-plane：画布白边 左5.6 右14.7 上1.0 下3.0 mm；宜 ≤ 3 mm 且左右对称
- 图3.4 fig:ch3mechanism：画布白边 左1.8 右5.1 上3.0 下7.6 mm；宜 ≤ 3 mm 且左右对称
- 图4.2 fig:ch4-cone：画布白边 左11.4 右6.3 上1.0 下1.8 mm；宜 ≤ 3 mm 且左右对称

**F10 题注**
- 图2.1 fig:ch2-gain-threshold：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图2.4 fig:ch2-reference-plane：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图2.6 fig:full_dac：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图3.1 fig:ch3-coupling：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图4.1 fig:ch4-rectangle：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图4.6 fig:c4recovery：短题约 31 字，宜 ≤ 30 字
- 图4.6 fig:c4recovery：完整题注估计 6 行，宜 ≤ 4 行，解释性内容移入正文
- 图4.8 fig:c4paclosedloop：短题约 31 字，宜 ≤ 30 字
- 图4.8 fig:c4paclosedloop：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图5.2 fig:ch5-swap-traj：完整题注估计 6 行，宜 ≤ 4 行，解释性内容移入正文
- 图6.1 fig:ch6-arch-3d：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图6.2 fig:ch6-exp1：短题约 34 字，宜 ≤ 30 字
- 图6.3 fig:ch6-exp2：短题约 36 字，宜 ≤ 30 字
- 图6.3 fig:ch6-exp2：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图6.4 fig:ch6-exp3：短题约 38 字，宜 ≤ 30 字
- 图6.4 fig:ch6-exp3：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图6.5 fig:ch6-exp4-hetero：短题约 36 字，宜 ≤ 30 字
- 图6.5 fig:ch6-exp4-hetero：完整题注估计 7 行，宜 ≤ 4 行，解释性内容移入正文

**F11 标签**
- 图2.2 fig:weak_connection：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图2.3 fig:information_flow：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图2.5 fig:reference_trajectory：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图2.6 fig:full_dac：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图2.7 fig:gain_scan：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图2.8 fig:switch_topology：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图2.9 fig:switch_comparison：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图2.10 fig:comparison_failure：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图3.4 fig:ch3mechanism：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.5 fig:ch3convergence：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.6 fig:ch3scene：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.7 fig:ch3anytime：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.8 fig:ch3comparison：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.9 fig:ch3switching：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.10 fig:ch3interruptions：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.11 fig:ch3closedloop：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图4.3 fig:c4crossing：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.4 fig:c4topmargins：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.5 fig:c4geometry：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.6 fig:c4recovery：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.7 fig:c4pipeline：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.8 fig:c4paclosedloop：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.9 fig:c4blocksettle：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图5.1 fig:seven5：标签未用 fig:ch5- 前缀（存量标签可不改）

**F13 TikZ**
- 图1.1 fig:ch1-loop：TikZ 有 5 个节点用绝对坐标定位，框内文字变长时易重叠；宜改用 positioning 相对定位
- 图1.2 fig:ch1-roadmap：TikZ 有 7 个节点用绝对坐标定位，框内文字变长时易重叠；宜改用 positioning 相对定位

**F14 同章同档高度不一**
- 第2章 P4：同档高度相差 13 mm：图2.2 105，图2.5 112，图2.6 111，图2.9 118
- 第2章 P2E：同档高度相差 18 mm：图2.4 68，图2.8 50
- 第3章 P2：同档高度相差 22 mm：图3.5 62，图3.7 84，图3.10 76
- 第4章 P4：同档高度相差 30 mm：图4.6 104，图4.7 117，图4.8 104，图4.9 87
- 第5章 P2E：同档高度相差 6 mm：图5.1 74，图5.4 68
- 第5章 P4：同档高度相差 4 mm：图5.2 122，图5.3 118

**P01 图与首次引用的页距**
- 图3.5 fig:ch3convergence：图在 p75，距首次引用 p73 为 2 页
- 图4.7 fig:c4pipeline：图在 p116，距首次引用 p114 为 2 页
- 图5.1 fig:seven5：图在 p127，距首次引用 p125 为 2 页


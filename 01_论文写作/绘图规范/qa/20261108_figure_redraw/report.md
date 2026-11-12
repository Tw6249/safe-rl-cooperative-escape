# 图件规范 3.0 检查报告

- 生成时间：2026-11-08T09:16:37
- 命令：`python check_figures.py --pdf ../整篇论文/build/main.pdf --pages --layout-map qa/layout_map_20261106.json --out qa/20261108_figure_redraw`
- 检查的 PDF：`C:\Users\TanWei\Desktop\phd-thesis\01_论文写作\整篇论文\build\main.pdf`（修改时间 2026-11-08T09:16:18，页码偏移 18）
- 构图代号来源：tex = figure 内 `% layout=`；pdf = 图件 PDF 关键字；map = `--layout-map` 人工判读

## 汇总

ERROR 22 条，WARN 53 条。代码后 `!` 为 ERROR，`?` 为 WARN。

| 代码 | 含义 | ERROR | WARN |
|---|---|---:|---:|
| F02 | 插入缩放 | 1 | 0 |
| F03 | 画布宽度 | 1 | 3 |
| F04 | 画布高度与档位 | 10 | 7 |
| F05 | 每排子图数 | 1 | 0 |
| F06 | 字号下限 | 6 | 0 |
| F07 | 字体 | 1 | 0 |
| F08 | 位图分辨率 | 0 | 1 |
| F09 | 白边与偏心 | 0 | 1 |
| F10 | 题注 | 0 | 11 |
| F11 | 标签 | 0 | 18 |
| F13 | TikZ | 0 | 1 |
| F14 | 同章同档高度不一 | 0 | 5 |
| P01 | 图与首次引用的页距 | 1 | 6 |
| P02 | 页底空白 | 1 | 0 |

## 逐图

| 图 | 标签 | 构图 | 画布 mm | 缩放 | 成品 mm | 字号 中位/最小正文 | <7.8pt | 问题 |
|---|---|---|---|---:|---|---|---:|---|
| 1.1 | `fig:ch1-loop` | D（map） | TikZ |  |  |  |  | F13? |
| 1.2 | `fig:ch1-roadmap` | D（tex） | TikZ |  |  |  |  | ✓ |
| 2.1 | `fig:ch2-gain-threshold` | P2（map） | 150×62 | 1.0 | 150×62 | 8.8/8.5 | 0% | F04? F10? |
| 2.2 | `fig:information_flow` | D（map） | 150×52 | 1.0 | 150×52 | 8.8/8.8 | 1% | F06! F11? |
| 2.3 | `fig:full_dac` | P4（map） | 154×114 | 0.974 | 150×111 | 7.8/7.8 | 26% | F02! F03! F04! F06! F10? F11? |
| 2.4 | `fig:switch_comparison` | P4（map） | 150×125 | 1.0 | 150×125 | 8.8/8.0 | 0% | F04! F11? |
| 3.1 | `fig:ch3-coupling` | P4（map） | 150×128 | 1.0 | 150×128 | 9.5/8.0 | 0% | F04! F10? |
| 3.2 | `fig:ch3mechanism` | D（map） | 150×96 | 1.0 | 150×96 | 7.0/8.0 | 0% | F04! F09? F11? |
| 3.3 | `fig:ch3convergence` | P2（map） | 150×62 | 1.0 | 150×62 | 7.2/8.8 | 5% | F04? F06! F11? |
| 3.4 | `fig:ch3scene` | P2E（map） | 150×74 | 1.0 | 150×74 | 9.5/8.2 | 0% | F04! F11? |
| 3.5 | `fig:ch3anytime` | P2（map） | 150×84 | 1.0 | 150×84 | 8.8/8.8 | 23% | F04! F06! F11? |
| 3.6 | `fig:ch3switching` | P1W（map） | 150×72 | 1.0 | 150×72 | 7.5/8.8 | 68% | F04! F06! F11? P01! |
| 3.7 | `fig:ch3closedloop` | D（pdf） | 150×116 | 1.0 | 150×116 | 8.8/8.0 | 0% | F11? |
| 4.1 | `fig:ch4-rectangle` | 3+2（map） | 150×112 | 1.0 | 150×112 | 8.8/8.0 | 0% | F05! F10? |
| 4.2 | `fig:c4crossing` | V2（map） | 150×102 | 1.0 | 150×102 | 8.8/8.4 | 0% | F04? F11? |
| 4.3 | `fig:c4recovery` | P4（map） | 150×104 | 1.0 | 150×104 | 7.2/7.8 | 18% | F04! F06! F10? F11? |
| 4.4 | `fig:c4topmargins` | V2（map） | 153×105 | 0.981 | 150×103 | 8.6/8.0 | 0% | F03? F04? F11? |
| 4.5 | `fig:c4geometry` | P2E（pdf） | 150×66 | 1.0 | 150×66 | 9.5/8.2 | 0% | F11? |
| 4.6 | `fig:c4pipeline` | P4（map） | 153×119 | 0.981 | 150×117 | 9.3/8.3 | 0% | F03? F04? F11? P01? |
| 4.7 | `fig:c4paclosedloop` | P4（pdf） | 150×116 | 1.0 | 150×116 | 7.8/7.8 | 0% | F10? F11? P01? |
| 4.8 | `fig:c4symmetric` | D（pdf） | 150×116 | 1.0 | 150×116 | 8.2/8.2 | 0% | F11? |
| 4.9 | `fig:c4symmetricjcon` | P2（pdf） | 150×58 | 1.0 | 150×58 | 8.8/8.2 | 0% | F11? P01? |
| 5.1 | `fig:seven5` | P2E（map） | 149×73 | 1.01 | 150×74 | 9.6/8.9 | 0% | F03? F04! F11? |
| 5.2 | `fig:ch5-swap-traj` | P4（map） | 150×122 | 1.0 | 150×122 | 9.5/8.3 | 0% | F04! F10? |
| 5.3 | `fig:ch5-time-evol` | P4（map） | 150×118 | 1.0 | 150×118 | 9.5/8.4 | 0% | F04? |
| 5.4 | `fig:ch5-perm-gate` | P2E（map） | 150×68 | 1.0 | 150×68 | 8.8/8.0 | 0% | F04? |
| 6.1 | `fig:ch6-arch-3d` | D（pdf） | 150×116 | 1.0 | 150×116 | 12.0/12.0 | 0% | F07! F08? F10? |
| 6.2 | `fig:ch6-arch-curves` | P2E（pdf） | 150×66 | 1.0 | 150×66 | 9.5/8.5 | 0% | P01? |
| 6.3 | `fig:ch6-exp1` | V3（pdf） | 150×128 | 1.0 | 150×128 | 9.5/8.8 | 0% | ✓ |
| 6.4 | `fig:ch6-exp1b` | V3（pdf） | 150×128 | 1.0 | 150×128 | 9.5/8.8 | 0% | P01? |
| 6.5 | `fig:ch6-exp2` | P4（pdf） | 150×116 | 1.0 | 150×116 | 9.5/8.5 | 0% | F10? P01? |
| 6.6 | `fig:ch6-exp3` | P4（pdf） | 150×116 | 1.0 | 150×116 | 8.8/8.5 | 0% | ✓ |

## 明细

### ERROR（22）

**F02 插入缩放**
- 图2.3 fig:full_dac：插入缩放比 0.974（画布 154 mm 宽被压到 150 mm），应为 1:1

**F03 画布宽度**
- 图2.3 fig:full_dac：画布宽 154.0 mm，应为 150 或 105 mm

**F04 画布高度与档位**
- 图2.3 fig:full_dac：P4 档标称高 116 mm，成品 111.0 mm（差 -5.0）
- 图2.4 fig:switch_comparison：P4 档标称高 116 mm，成品 125.0 mm（差 +9.0）
- 图3.1 fig:ch3-coupling：P4 档标称高 116 mm，成品 128.0 mm（差 +12.0）
- 图3.2 fig:ch3mechanism：D 档标称高 100 mm，成品 96.0 mm（差 -4.0）
- 图3.4 fig:ch3scene：P2E 档标称高 66 mm，成品 74.0 mm（差 +8.0）
- 图3.5 fig:ch3anytime：P2 档标称高 58 mm，成品 84.0 mm（差 +26.0）
- 图3.6 fig:ch3switching：P1W 档标称高 78 mm，成品 72.0 mm（差 -6.0）
- 图4.3 fig:c4recovery：P4 档标称高 116 mm，成品 104.0 mm（差 -12.0）
- 图5.1 fig:seven5：P2E 档标称高 66 mm，成品 73.7 mm（差 +7.7）
- 图5.2 fig:ch5-swap-traj：P4 档标称高 116 mm，成品 122.0 mm（差 +6.0）

**F05 每排子图数**
- 图4.1 fig:ch4-rectangle：首排 3 个带坐标轴的子图；数据子图每排最多 2 个，改用 P3/V3 或拆图

**F06 字号下限**
- 图2.2 fig:information_flow：1/69 处文字（1%）低于 7.8 pt：7 pt
- 图2.3 fig:full_dac：62/239 处文字（26%）低于 7.8 pt：5, 5.2, 7.2, 7.3, 7.4 pt
- 图3.3 fig:ch3convergence：8/146 处文字（5%）低于 7.8 pt：7.2 pt
- 图3.5 fig:ch3anytime：30/128 处文字（23%）低于 7.8 pt：5.3, 7.4, 7.6 pt
- 图3.6 fig:ch3switching：86/127 处文字（68%）低于 7.8 pt：3.7, 5.2, 7.5 pt
- 图4.3 fig:c4recovery：70/392 处文字（18%）低于 7.8 pt：7.2 pt

**F07 字体**
- 图6.1 fig:ch6-arch-3d：使用禁用字体：DejaVuSans, DejaVuSans-Oblique

**P01 图与首次引用的页距**
- 图3.6 fig:ch3switching：图在 p67，距首次引用 p47 达 20 页

**P02 页底空白**
- p72：页底空白 35 mm（非章末页）

### WARN（53）

**F03 画布宽度**
- 图4.4 fig:c4topmargins：画布宽 153.0 mm，应为 150 或 105 mm
- 图4.6 fig:c4pipeline：画布宽 153.0 mm，应为 150 或 105 mm
- 图5.1 fig:seven5：画布宽 148.6 mm，应为 150 或 105 mm

**F04 画布高度与档位**
- 图2.1 fig:ch2-gain-threshold：P2 档标称高 58 mm，成品 62.0 mm，在存量容差内；修改时回到标称值
- 图3.3 fig:ch3convergence：P2 档标称高 58 mm，成品 62.0 mm，在存量容差内；修改时回到标称值
- 图4.2 fig:c4crossing：V2 档标称高 100 mm，成品 102.0 mm，在存量容差内；修改时回到标称值
- 图4.4 fig:c4topmargins：V2 档标称高 100 mm，成品 102.9 mm，在存量容差内；修改时回到标称值
- 图4.6 fig:c4pipeline：P4 档标称高 116 mm，成品 116.7 mm，在存量容差内；修改时回到标称值
- 图5.3 fig:ch5-time-evol：P4 档标称高 116 mm，成品 118.0 mm，在存量容差内；修改时回到标称值
- 图5.4 fig:ch5-perm-gate：P2E 档标称高 66 mm，成品 68.0 mm，在存量容差内；修改时回到标称值

**F08 位图分辨率**
- 图6.1 fig:ch6-arch-3d：内嵌位图成品分辨率最低 100 ppi；渲染帧/照片须 ≥ 300，数据热图可说明后保留

**F09 白边与偏心**
- 图3.2 fig:ch3mechanism：画布白边 左1.8 右5.1 上3.0 下7.6 mm；宜 ≤ 3 mm 且左右对称

**F10 题注**
- 图2.1 fig:ch2-gain-threshold：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图2.3 fig:full_dac：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图3.1 fig:ch3-coupling：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图4.1 fig:ch4-rectangle：完整题注估计 5 行，宜 ≤ 4 行，解释性内容移入正文
- 图4.3 fig:c4recovery：短题约 31 字，宜 ≤ 30 字
- 图4.3 fig:c4recovery：完整题注估计 6 行，宜 ≤ 4 行，解释性内容移入正文
- 图4.7 fig:c4paclosedloop：短题约 31 字，宜 ≤ 30 字
- 图5.2 fig:ch5-swap-traj：完整题注估计 6 行，宜 ≤ 4 行，解释性内容移入正文
- 图6.1 fig:ch6-arch-3d：完整题注估计 7 行，宜 ≤ 4 行，解释性内容移入正文
- 图6.5 fig:ch6-exp2：短题约 36 字，宜 ≤ 30 字
- 图6.5 fig:ch6-exp2：完整题注估计 6 行，宜 ≤ 4 行，解释性内容移入正文

**F11 标签**
- 图2.2 fig:information_flow：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图2.3 fig:full_dac：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图2.4 fig:switch_comparison：标签未用 fig:ch2- 前缀（存量标签可不改）
- 图3.2 fig:ch3mechanism：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.3 fig:ch3convergence：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.4 fig:ch3scene：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.5 fig:ch3anytime：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.6 fig:ch3switching：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图3.7 fig:ch3closedloop：标签未用 fig:ch3- 前缀（存量标签可不改）
- 图4.2 fig:c4crossing：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.3 fig:c4recovery：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.4 fig:c4topmargins：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.5 fig:c4geometry：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.6 fig:c4pipeline：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.7 fig:c4paclosedloop：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.8 fig:c4symmetric：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图4.9 fig:c4symmetricjcon：标签未用 fig:ch4- 前缀（存量标签可不改）
- 图5.1 fig:seven5：标签未用 fig:ch5- 前缀（存量标签可不改）

**F13 TikZ**
- 图1.1 fig:ch1-loop：TikZ 有 5 个节点用绝对坐标定位，框内文字变长时易重叠；宜改用 positioning 相对定位

**F14 同章同档高度不一**
- 第2章 P4：同档高度相差 14 mm：图2.3 111，图2.4 125
- 第3章 P2：同档高度相差 22 mm：图3.3 62，图3.5 84
- 第4章 P4：同档高度相差 13 mm：图4.3 104，图4.6 117，图4.7 116
- 第5章 P2E：同档高度相差 6 mm：图5.1 74，图5.4 68
- 第5章 P4：同档高度相差 4 mm：图5.2 122，图5.3 118

**P01 图与首次引用的页距**
- 图4.6 fig:c4pipeline：图在 p93，距首次引用 p91 为 2 页
- 图4.7 fig:c4paclosedloop：图在 p96，距首次引用 p94 为 2 页
- 图4.9 fig:c4symmetricjcon：图在 p99，距首次引用 p97 为 2 页
- 图6.2 fig:ch6-arch-curves：图在 p126，距首次引用 p124 为 2 页
- 图6.4 fig:ch6-exp1b：图在 p133，距首次引用 p131 为 2 页
- 图6.5 fig:ch6-exp2：图在 p136，距首次引用 p134 为 2 页


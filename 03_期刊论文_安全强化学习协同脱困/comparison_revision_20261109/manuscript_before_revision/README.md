# 期刊论文独立工程：耦合安全约束下多机器人协同脱困的包络等变图强化学习与邻域近端分配控制（EGP-NPA）

本目录是基于博士论文第五章《耦合安全约束下的安全强化学习与协同脱困》完整撰写的**独立自包含学术期刊论文工程**（对冻结实验数据与原有章节代码保持严格只读，不修改任何工作区既有文件）。为兼顾国内外顶刊投递需求，本目录同时提供**中文控制顶刊双栏完整版**（对标《自动化学报》/《控制与决策》）与**英文 IEEE Transactions 双栏完整版**（采用标准 `IEEEtran.cls`，对标 *IEEE Transactions on Robotics* / *IEEE Transactions on Automatic Control*）。

## 一、 核心成品与编译入口

| 版本 | 主入口 TeX | 期刊模板类 | 成品 PDF | 编译方式 | 版面与核验状态 |
|---|---|---|---|---|---|
| **中文顶刊双栏版** | [main_zh.tex](main_zh.tex) | [cjas_journal.cls](cjas_journal.cls) | [main_zh.pdf](main_zh.pdf) / [main.pdf](main.pdf) | `XeLaTeX + BibTeX` | 19 页，6 图 5 表，0 警告、0 越界 |
| **英文 IEEE Transactions 版** | [main_en.tex](main_en.tex) | `IEEEtran.cls` | [main_en.pdf](main_en.pdf) | `pdfLaTeX + BibTeX` | 16 页，6 图 5 表，0 警告、0 越界 |

### 一键编译命令
在 PowerShell 中运行：
```powershell
& '.\03_期刊论文_安全强化学习协同脱困\build.ps1'
# 若需从底层冻结数据重新生成全套中英文矢量图，可加上 -Figures 参数：
& '.\03_期刊论文_安全强化学习协同脱困\build.ps1' -Figures
```

## 二、 章节结构与自包含内容对应

相比学位论文单章，本期刊稿完全剥离了跨章依赖，补齐了自包含的闭邻域近端分配（NPA）及反对称边记录本地结清预备知识、全部定理与命题的完整数学证明、算法伪代码框（算法 1）、48 组盲测析因对照以及二阶全向底盘 `MuJoCo` 虚拟物理闭环验证：

| 节次 | 中文版源文件 (`sections_zh/`) | 英文版源文件 (`sections_en/`) | 核心内容 |
|---|---|---|---|
| **第 1 节 引言** | `01_introduction.tex` | `01_introduction.tex` | 研究背景、三类相关工作综述（分布式耦合安全滤波、CBF 非目标平衡点与几何死锁、多智能体安全强化学习）及四项核心贡献 |
| **第 2 节 问题建模与预备知识** | `02_preliminaries.tex` | `02_preliminaries.tex` | 平面多机固定标签目标模型、有理数闭合多项式信道核、零阶保持整段安全定理（**定理 1** 及完整证明）、闭邻域块 NPA 与反对称边记录一跳本地结清自包含建模 |
| **第 3 节 死锁机理与有理数证书** | `03_obstruction.tex` | `03_obstruction.tex` | 短视误差单调控制（$\dot{W}\le 0$）的法锥锁死机制、七机器人严格非目标局部极小有理数 KKT 证书与球面二次能量壁垒及安全连通可达路径（**定理 2**、完整证明、**图 1**、**注 1**） |
| **第 4 节 EGP-NPA 协同控制方法** | `04_method.tex` | `04_method.tex` | 17 维节点与 20 维有向边旋转不变特征、$\mathrm{SO}(2)\ltimes\mathbb{R}^2$ 与置换等变性及手性破缺必要性证明（**命题 1**）、随时硬安全不变性（**命题 2**）、动态份额转移驱动静止阻挡邻居同向退让的闭式解（**命题 3**）、一跳闭邻域终点完成包络 $\rho_{i,k}$ 一阶消隐性质（**命题 4**）、小规模短轮数 MAPPO 训练与跨规模部署协议（**算法 1 / 表 1**） |
| **第 5 节 闭环实验、消融与物理验证** | `05_experiments.tex` | `05_experiments.tex` | 训练收敛演化（**图 2**）、$N=20$ 密集晶格内部互换轨迹与时间响应（**图 3**、**图 4**）、与三类代表性安全强化学习方法（`Proj-MAPPO + NPA`、`Static-CBF GNN`、`MAPPO-Lagrangian`）及经典基线在 $N=8$ 训练规模（**表 2**，64 组）与 $N\in\{12,20,25\}$ 零样本跨规模迁移（**表 3**，32 组）上的对比、$N=20$ 全排列与固定权重门控替换及特征敲除（**图 5**、**表 4**）、48 组独立盲测析因实验、二阶底盘安全裕度—可通行性权衡定理（**定理 3**）及 `MuJoCo` 全栈物理验证（**表 5**、**图 6**） |
| **第 6 节 结论与展望** | `06_conclusion.tex` | `06_conclusion.tex` | 全文理论与实证总结、方法适用边界与未来研究方向 |

## 三、 出版级矢量图清单（`make_journal_figures.py`）

所有图件均通过 [make_journal_figures.py](make_journal_figures.py) 直接读取 `02_章节代码/` 下已保存的 `.json` 与 `.npz` 冻结实测数据生成，分别输出至 `figures_zh/`（中文标签）与 `figures_en/`（英文标签）：

1. `fig1_seven_geometry.pdf`（`P3` 构图）：(a) 七机双三角对称接触构型与临界锥最优速度见证；(b) 解析安全连通可达路径；(c) 路径任务误差能量壁垒曲线 $W(\boldsymbol{p}(s))$。
2. `fig2_training_curves.pdf`（`P2` 构图）：(a) MAPPO 训练回合累积奖励收敛曲线；(b) $N=8$ 验证集确定性评估达标率演化。
3. `fig3_swap_trajectory.pdf`（`P4` 构图）：$N=20$ 密集晶格内部单对节点互换任务的空间运动轨迹与通信拓扑对比。
4. `fig4_time_evolution.pdf`（`P4` 构图）：$N=20$ 内部互换任务中最大位置误差、代数连通度、最小机间距离与均方根控制速度的闭环时间响应。
5. `fig5_perm_and_gate.pdf`（`P2E` 构图）：(a) $N=20$ 随机全排列空间穿插轨迹；(b) 固定权重 $\theta^\star$ 下的门控替换与代数特征敲除误差收敛响应。
6. `fig6_mujoco_verification.pdf`（`P4` 构图）：全栈 EGP-NPA + 在线无除法谱估计 + 闭邻域 NPA 在 `MuJoCo` 二阶全向底盘中的闭环误差收敛、实时谱跟踪与安全裕度—可通行性权衡对比。

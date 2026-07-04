# Innovation 1 PRESENT 轮数推进研究蓝图

**日期：** 2026-07-04

**状态：** 新研究路线蓝图 / 准备进入 r8 中等诊断实验

**范围：** Innovation 1 的 SPN/PRESENT 分支。本文是 `docs/research/` 下的研究蓝图，回答“为什么接下来要从 r7 机制微调推进到 r8/r9 轮数扩展，以及什么样的方法路线值得继续投入”。具体 run id、GPU、样本规模、seed、gate 和远程命令放在 `docs/experiments/innovation1-present-r8-round-extension-ladder-plan.md`。

## 1. 研究重心调整

前一阶段的主要问题是：

```text
在 Zhang/Wang 2022 PRESENT-80 r7 Case2 m=16 协议下，
SPN/P-layer 对齐的数据表示或网络结构是否真的比普通 MCND 更有用？
```

这个问题目前已有比较清楚的内部证据：

| 路线 | 当前证据 |
|---|---|
| Zhang/Wang-style MCND local anchor | r7、1M/class、seed0，本地同协议 AUC `0.793897025948` |
| `present_nibble_invp_only_spn_only` | r7、1M/class、seed0 AUC `0.797470988906`，seed1 AUC `0.797347588554` |
| InvP attribution controls | r7、1M/class，`DeltaC-only` AUC `0.792064879854`，`shuffled-P` AUC `0.793621524954` |
| 当前允许结论 | `InvP(DeltaC)` 作为 P-layer 对齐 SPN 视图，对 r7 同协议区分器有稳定正向证据和归因支持 |

这说明“SPN 结构适配”不是空想，但如果目标是提高区分轮数，继续只在 r7 做很小的结构微调会越来越慢。下一阶段应把问题改成：

```text
这些 SPN 结构适配信号能不能在 PRESENT r8 仍然保留？
如果 r8 还有弱但稳定的信号，如何通过 pair-set evidence pooling、
multi-query aggregation、curriculum / transfer，把它变成可验证的轮数推进路线？
```

因此，本路线不是放弃 r7 证据，而是把 r7 证据作为已知有效结构先验，向 r8/r9 做“轮数阶梯”。

## 2. 文献锚点与可比性约束

### 2.1 Zhang/Wang 2022 是当前本地严格协议锚点

Zhang 和 Wang 的论文 *Improving Differential-Neural Distinguisher Model For DES, Chaskey, and PRESENT* 明确报告了 PRESENT 的 6/7 轮 MCND 改进。论文摘要说其目标是提高 DES、Chaskey 和 PRESENT 的差分神经区分器预测精度，并对 PRESENT 的 6/7 轮区分器做了提升。

本地 PDF 表 4 中，PRESENT-80 r7 的关键数值为：

| Setting | m=2 | m=4 | m=8 | m=16 |
|---|---:|---:|---:|---:|
| MCND | 0.5503 | 0.5853 | 0.5786 | 0.5818 |
| Case1 | 0.5717 | 0.6054 | 0.6510 | 0.7070 |
| Case2 | 0.5717 | 0.6070 | 0.6559 | **0.7205** |

本项目当前严格参考值采用：

```text
PRESENT-80 r7, Zhang/Wang 2022 Case2 m=16, accuracy = 0.7205
```

注意：这个 `0.7205` 是论文参考值，不等同于本项目已经完全复现论文全部训练规模和官方环境。本项目比较时以本地同协议、同尺度、严格负样本 anchor 为准，同时保留论文值作为外部参考。

### 2.2 Jain/Kohli/Mishra 2021 到 PRESENT 6 轮

Jain、Kohli 和 Mishra 的 *Deep Learning based Differential Distinguisher for Lightweight Block Ciphers* 报告了 PRESENT-80 和 Simeck 的深度学习差分区分器，其中 PRESENT 可到 6 轮，Simeck 可到 7 轮。它说明 PRESENT/SPN 本身比 ARX/SPECK 那条路线更难，不能期待普通网络简单扩大后自然推进到高轮。

### 2.3 r8 结果必须按 taxonomy 比较

现有 `docs/research/spn_structured_nn_research_plan.md` 已经记录：综述类材料中存在若干 PRESENT r8 相关结果，例如 DBitNet、INC、多 pair、`E=D` 或 related-key setting 下的结果。但这些结果不能直接和 Zhang/Wang Case2 `m=16`、strict `encrypted_random_plaintexts`、本项目 `real-vs-random` 设定混为一谈。

因此后续报告必须使用类似 `n-m-T-E` 的可比性标签：

| 维度 | 含义 | 对本项目的约束 |
|---|---|---|
| `n` | 每个样本含多少 ciphertext / pair 信息 | 多 pair 会增加统计信息，不能直接和 single-pair 比 |
| `m` | 输入差分数量 | Zhang/Wang Case2 `m=16` 是当前本地 anchor |
| `T` | 输入/特征类型 | 原始 CT、差分 `delta`、SPN-aware advanced features 要分开 |
| `E` | 区分任务 | `R` real-vs-random 与 `D` differential-vs-differential 不可直接混报 |

本项目 r8 目标应表述为：

```text
在本项目严格同协议链条下，探索 PRESENT r8 是否仍存在可学习的 SPN 结构信号。
```

而不是泛泛宣称：

```text
首次做到 PRESENT r8
```

除非后续完成严格的同 taxonomy 文献复核和 formal evidence gate。

## 3. 核心假设

当前最强内部证据表明，`InvP(DeltaC)` 能把最终输出差分重新对齐到上一轮 S-box cell 视角，让网络更容易看到 SPN 扩散后的局部结构。

新的轮数推进假设是：

```text
高一轮的 PRESENT 信号不会靠单个 r7 微小模块自动变强；
它需要把已经验证有效的 InvP/P-layer 对齐视图，
和 pair-set evidence pooling、multi-query aggregation、curriculum / transfer
组合成轮数阶梯。
```

拆成更可检验的三个子假设：

1. **表示保留假设：** 如果 r8 还有可学习信号，InvP/P-layer 对齐视图应当比普通 Zhang/Wang-style MCND 更容易保留信号。
2. **证据汇聚假设：** r8 单样本信号可能很弱，pair-set consistency 或 frozen score aggregation 能把弱信号变成更稳定的应用级证据。
3. **训练迁移假设：** r7 已学到的 SPN 结构滤波器可能适合作为 r8 curriculum / transfer 的初始化，而不是每次从随机初始化开始硬学。

## 4. 路线阶梯

### Stage A：r8 中等诊断

目标：

```text
先判断 r8 是否还有可学习信号，以及结构适配路线是否比同预算 baseline 更能保留信号。
```

首轮只跑 3 行，避免实验矩阵过大：

| Row | 模型 | 作用 |
|---:|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | 同预算 Zhang/Wang-style baseline |
| 1 | `present_nibble_invp_only_spn_only` | 当前最强 SPN 表示 anchor |
| 2 | `present_nibble_invp_pair_consistency_spn_only` | 检查 pair-set evidence pooling 在 r8 是否更有价值 |

证据等级：

```text
262144/class = medium diagnostic only
```

这一级只用于决定是否值得升到 1M/class，不用于论文级结论。

### Stage B：r8 1M/class 确认

触发条件：

```text
r8 262144/class 中至少一个结构候选明显高于随机，并且超过同预算 baseline。
```

建议判据：

| 情况 | 行动 |
|---|---|
| best candidate AUC `<= 0.52` | 不扩大；考虑 curriculum / transfer 或换数据表示 |
| AUC `0.52 - 0.55` 且超过 baseline | 弱信号；先 seed1 或训练策略诊断 |
| AUC `>= 0.55` 且超过 baseline `+0.005` | 支持升到 1M/class |

这些是研究 gate，不是正式成功声明。

### Stage C：r8 confirmation / r9 weak probe

如果 r8 1M/class 至少两个 seed 稳定正向：

```text
1. 做 r8 multi-seed / attribution / aggregation control。
2. 再启动 r9 weak probe。
```

如果 r8 只在 multi-query 或 pair-set 聚合后有效，必须把 claim 限定为：

```text
application-level evidence
```

而不是 raw single-sample neural distinguisher SOTA。

## 5. 方法候选

### 5.1 InvP/P-layer aligned representation

已完成 r7 证据支持：

```text
DeltaC -> InvP(DeltaC) -> 16 个 4-bit S-box cell token
```

它的优点是低复杂度、可解释、与 PRESENT P-layer 直接对应。r8 首轮必须保留它作为主候选。

### 5.2 Pair-set evidence pooling

r8 的单样本 margin 可能很小。pair-set 路线要回答：

```text
网络是否能从同一个 sample 的 16 个 pair 中学到一致性，
超过单 pair 分数的 frozen aggregation？
```

这条路线不能直接宣称“网络结构创新有效”，除非它超过 frozen single-pair score aggregation control。

### 5.3 Curriculum / transfer

如果 r8 从零训练 AUC 很弱，但 r7 已有稳定结构信号，则下一步不应只继续换小模块，而应测试：

```text
r7 -> r8 curriculum
r7 checkpoint -> r8 fine-tune
r6/r7/r8 mixed-round training
```

这属于训练策略变化，必须和模型/数据结构变化分开做实验。

### 5.4 S-box prior gate 与 DDT graph

当前 S-box transition prior gate 的 r7 run 仍在 watcher 管理中。它属于 r7 机制验证，不应被 r8 路线中断。若它最终表现出明确正信号，可作为 r8 Stage B 或 Stage C 的候选增强模块；若它停止，也不影响 r8 首轮使用已经完成验证的 InvP-only 和 pair-set 候选。

## 6. 证据语言

后续必须区分：

| 证据等级 | 允许说法 | 不允许说法 |
|---|---|---|
| smoke | 代码路径能跑 | 模型有效 |
| 262144/class diagnostic | 有/无中等诊断信号 | 正式成功/失败 |
| 1M/class single-seed | paper-scale strong diagnostic | formal route evidence |
| 1M/class multi-seed + controls | route-level evidence | SOTA/突破，除非文献 taxonomy 已核准 |
| multi-query aggregation | 应用级证据 | raw single-sample SOTA |

## 7. 当前结论

当前最合理的推进不是继续在 r7 上堆大矩阵，而是：

```text
保持 r7 S-box prior gate watcher 不动；
准备并验证 r8 262144/class 三行中等诊断；
如果 r8 有弱信号，再进入 1M/class 或 curriculum / transfer；
如果 r8 无信号，则把重点转向训练迁移或新的 SPN-aware 数据表示。
```

这条路线更贴近用户的核心目标：

```text
做出真正面向 SPN/PRESENT 结构适配的神经网络或数据集表示，
并尽量推进可验证的区分轮数。
```

## 8. 参考来源

- Zhang and Wang, *Improving Differential-Neural Distinguisher Model For DES, Chaskey, and PRESENT*, arXiv:2204.06341: https://arxiv.org/abs/2204.06341
- Jain, Kohli, and Mishra, *Deep Learning based Differential Distinguisher for Lightweight Block Ciphers*, arXiv:2112.05061: https://arxiv.org/abs/2112.05061
- 本项目文档：`docs/research/spn_structured_nn_research_plan.md`
- 本项目证据汇总：`docs/experiments/innovation1-invp-route-level-evidence-summary.md`

# Innovation 1 PRESENT 更高轮次研究策略

**日期：** 2026-07-05

**状态：** 新任务研究蓝图 / r9 from-scratch weak-probe 已完成 / r8 pair-set 1M 正在 watcher 管理下运行 / 等待结果后仲裁下一分支

**范围：** Innovation 1 的 SPN/PRESENT 更高轮次推进。本文回答“如果目标不是只在 r7/r8 同协议下微调，而是尽量推进 PRESENT 可验证轮数，研究路线应该怎么改”。具体 r9 weak-probe 运行计划和结果放在 `docs/experiments/innovation1-present-r9-weak-probe-plan.md`。

## 1. 研究目标

用户的新任务是：

```text
推进更高轮次研究。
```

这里的“更高轮次”不能理解成简单把 `rounds` 从 8 改到 9、10、11 然后硬跑。PRESENT 的高轮 real-vs-random 神经信号会迅速接近随机，盲目增加轮数很容易得到一堆不可解释的 0.50x 结果。

因此本任务的目标应定义为：

```text
围绕 PRESENT/SPN，建立 r8 -> r9 -> r10 的条件式 round-extension ladder，
验证 SPN-aware 数据表示、pair-set evidence pooling、curriculum / transfer、
multi-query aggregation 哪一种路线最可能把弱单样本信号推到更高轮。
```

它不是一个单独的模型实验，而是一个研究阶段。

## 2. 文献边界

当前可稳妥引用的公开锚点：

| 文献 | PRESENT 相关结论 | 对本项目的作用 |
|---|---|---|
| Jain/Kohli/Mishra 2021 | 深度学习差分区分器可到 PRESENT-80 6 轮 | 说明 PRESENT/SPN 标准 DL 区分本来就比 SPECK 路线更难 |
| Zhang/Wang 2022 | PRESENT-80 r7 Case2 `m=16` accuracy `0.7205` | 本项目当前严格同协议 r7 外部 anchor |
| Gohr et al. assessment / entropy-based PRESENT 2026 synthesis | differential PRESENT 常见公开 anchor 仍集中在 r6/r7；有工作整理 r6 `0.712`、r7 `0.563`，后续 entropy-bit-selection 更偏向轻量化和 key recovery，而不是提高 differential 轮数 | 说明 raw differential pair 到 r8/r9 可能很弱，需要结构特征、训练路径或聚合路线 |
| Wu/Guo integral-neural PRESENT 2024 | 通过 `invP / invS` previous-round 数据格式和 DenseNet/MBConv，把 integral-neural PRESENT 推到 8 轮，报告 r8 accuracy `57.32%` | 这是“提高 PRESENT 可验证轮数”最直接的 SPN 数据构造启发，但它属于 integral/multiset setting，不能和 Zhang/Wang differential m=16 直接混报 |
| Generic Partial Decryption / AutoND 系列 | 对 SPN，zero-key partial decryption 可暴露上一轮 truncated/activity 信息；多 pair、更多数据和 partial-decryption 的贡献应分开评估 | 支持把本项目下一步从“换模型”扩展到“partial inverse data representation + controlled attribution” |
| 2026 IoT-friendly SPN framework | 对 SKINNY/MIDORI 这类 SPN，inverse-round/Conv2D-style 数据格式能提高 accuracy 或有效轮数 | 方法启发强，但不是 PRESENT differential 同协议证据 |

本项目不能直接写：

```text
我们要首次做到 PRESENT r8/r9。
```

更严谨的表述是：

```text
我们在 strict encrypted-random-plaintext negatives、Zhang/Wang Case2 m=16、
同协议同尺度证据链下，探索 SPN-structure-adaptive 表示是否能推进
PRESENT real-vs-random neural distinguisher 的可验证轮数。
```

新增边界：

```text
如果切换到 integral/multiset、partial-decryption、entropy-bit-selection、
input-difference search 或 multi-query aggregation，就必须单独标注证据类型。
这类路线可以服务“提高可验证轮数”，但不能直接写成 Zhang/Wang Case2
differential raw single-sample 的同协议提升。
```

## 3. 为什么不能直接冲 r10

如果 r8 还没有完成 retrieved / validated / gated，直接开 r9/r10 会有三个问题：

1. **缺少条件证据。** 不知道 r8 是还有单样本信号、只剩聚合信号，还是已经接近随机。
2. **不可归因。** r9/r10 失败时无法判断是轮数太高、训练策略不对、数据表示不够，还是模型容量不合适。
3. **浪费 GPU。** 当前 r7 S-box prior 和 r8 round-extension 已经分别占用 watcher/GPU 资源，更高轮应该准备好但等待 gate 触发。

所以本阶段采用：

```text
prepare now, launch conditionally
```

而不是：

```text
launch everything now
```

### 3.1 当前 r9 from-scratch 诊断结论

截至 2026-07-05，本阶段已经完成并拉回：

```text
run_id = i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
scale = 262144/class
status = retrieved / validated / plotted / postprocessed / plan-aligned
```

结果：

| Model | AUC | Calibrated accuracy | 解释 |
|---|---:|---:|---|
| `present_zhang_wang_keras_mcnd` | `0.503853519913` | `0.503520965576` | baseline 最高，但仍接近随机 |
| `present_nibble_invp_pair_consistency_spn_only` | `0.502131485177` | `0.502822875977` | 最强候选，低于 baseline |
| `present_nibble_invp_only_spn_only` | `0.500478791655` | `0.501800537109` | 接近随机 |

当前 gate：

```text
decision = stop_from_scratch_r9_r10_plan_curriculum_or_difference_search
```

解释：

```text
这不是 r9 正式失败结论，也不是 r10 不可能。
它只说明在当前 Zhang/Wang Case2 m=16、strict negatives、262144/class、
from-scratch 训练路径下，现有 r7/r8 结构候选没有保留出可扩展的 r9 单样本信号。
```

下一步优先级因此调整为：

```text
1. 等待 active r8 pair-set 1M seed0 返回；
2. postprocess 并与 r9 from-scratch summary 仲裁；
3. 如果 r8 1M 支持 pair-set 路线，优先 r8 seed1 / frozen aggregation control；
4. 如果 r8 1M 不支持继续扩大，启动 r8-to-r9 curriculum；
5. 如果 curriculum 仍近随机，再进入 r9 difference screen 或 integral/inverse-round data route。
```

## 4. 高轮路线假设

### 4.1 单样本弱信号路线

假设：

```text
r9 仍存在微弱但可学习的单样本 real-vs-random 信号。
```

第一候选：

```text
present_nibble_invp_only_spn_only
```

原因：

```text
r7 两个 1M/class seed 已经支持 InvP/P-layer aligned representation；
r8 首轮也把它作为主候选。若 r9 仍有信号，最可能先在这个表示上出现。
```

风险：

```text
r9 单样本 AUC 可能只有 0.50x，262144/class 诊断方差会很大。
```

### 4.2 Pair-set / multi-query 聚合路线

假设：

```text
r9 单个 sample 的 score 很弱，但多个 pair 或多个 query 的 log-odds 聚合仍可形成应用级证据。
```

候选：

```text
present_nibble_invp_pair_consistency_spn_only
frozen single-pair score aggregation
multi-query score aggregation
```

约束：

```text
这类结果只能写 application-level evidence，不能写 raw single-sample SOTA。
```

### 4.3 Curriculum / transfer 路线

假设：

```text
r7/r8 学到的 SPN cell filter、InvP active pattern 或 pair evidence pooling
可以迁移到 r9，比随机初始化更稳定。
```

候选训练方式：

```text
r7 checkpoint -> r8 fine-tune -> r9 fine-tune
r6/r7/r8 mixed-round curriculum -> r9
r8 positive route checkpoint -> r9 weak probe
```

约束：

```text
训练策略变化必须和模型/数据结构变化分开记录。
```

### 4.4 高轮输入差分搜索路线

假设：

```text
当前 Zhang/Wang difference member 0 不一定是 r9/r10 最优输入差分。
```

如果 r8/r9 都弱，下一步应考虑：

```text
固定模型与协议，搜索 PRESENT/SPN 的 high-round candidate input differences；
先做小规模筛选，再对候选差分做 262144/class 验证。
```

这属于 benchmark/difference 变化，必须单独开研究路线，不能和模型结构创新混在一起。

当前已把该分支具体化为一个准备但不立即启动的 r9 screen：

```text
docs/experiments/innovation1-present-r9-difference-screen-plan.md
configs/experiment/innovation1/innovation1_spn_present_r9_difference_screen_65k_seed0.csv
```

这个 screen 固定 `present_nibble_invp_pair_consistency_spn_only`，只改变
`difference_profile / difference_member`。它用于回答“r9 是否需要换输入差分/数据构造”，
不能和同协议模型改进结论混报。

### 4.5 Integral / inverse-round 数据结构路线

假设：

```text
PRESENT r8/r9 的 raw ciphertext-pair differential signal 太弱，
但由 multiset/integral 结构和 inverse permutation / inverse S-box 暴露的
previous-round cell evidence 仍然可学习。
```

文献动机：

```text
Wu/Guo 的 8-round integral-neural PRESENT 不是 Zhang/Wang differential 协议，
但它显示：对 PRESENT 这类 nibble SPN，invP/invS previous-round 表示能把
原始 ciphertext 表示中被扩散稀释的结构重新摆到网络面前。
```

本项目落地方式：

```text
先不做纯论文复现，也不改变 strict encrypted-random-plaintext negatives。
使用项目已有 plaintext_integral_nibble sample_structure 作为 multiset-inspired
生成方式，并比较：

1. raw ciphertext pair bits；
2. InvP/P-layer aligned matrix bits；
3. InvP + structural inverse-S matrix bits。
```

证据语言：

```text
这条路线是 high-round data representation evidence。
它不是 Zhang/Wang Case2 同协议模型提升，也不是 Wu/Guo integral-neural 正式复现。
如果 screen 有正信号，再写 262144/class confirmation 和更严格的 pure-integral
复现/对照计划。
```

## 5. Round-Extension Ladder

### Stage H0：r8 watcher 结果

当前已启动：

```text
i1_present_r8_round_extension_262k_seed0_gpu0_20260704
```

它是 r9/r10 的 gate source。没有它，不启动 r9 medium run。

### Stage H1：r9 local smoke

目的：

```text
只证明 r9 配置、模型 forward、数据生成、metric 路径能跑。
```

允许现在做：

```text
CPU tiny smoke
```

不允许说：

```text
r9 有效或无效
```

### Stage H2：r9 262144/class weak probe

触发条件：

```text
r8 262144/class 完成 retrieved / validated / gate-note，
且 best candidate AUC > 0.52 或者 r8 pair-set/multi-query 显示应用级信号。
```

矩阵保持 3 行：

| Row | Model | Role |
|---:|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | r9 same-budget baseline |
| 1 | `present_nibble_invp_only_spn_only` | strongest SPN representation |
| 2 | `present_nibble_invp_pair_consistency_spn_only` | pair-set weak-signal pooling candidate |

Gate：

| Result | Decision |
|---|---|
| best candidate AUC `<= 0.505` | 不再从零训练推 r10；改做 curriculum / difference search |
| AUC `0.505 - 0.52` | 只算 near-random weak trace；优先多 seed 方差或 aggregation，不升 1M |
| AUC `> 0.52` 且超过 baseline | r9 weak positive，考虑 seed1 或 curriculum-scale |
| AUC `> 0.55` 且超过 baseline `+0.005` | 强诊断，准备 1M/class seed0 |

### Stage H3：r9 1M/class 或 r9 curriculum

只有 H2 支持时才进入。

可选路径：

```text
Path A: from-scratch r9 1M/class
Path B: r8 checkpoint -> r9 fine-tune
Path C: r7/r8/r9 curriculum
```

优先级：

```text
如果 H2 AUC > 0.55：Path A
如果 H2 AUC 0.505-0.55：Path B/C
```

### Stage H4：r10 weak probe

r10 只在以下条件之一成立后启动：

```text
1. r9 262144/class 有 clear weak positive；
2. r9 curriculum / transfer 明显超过 from-scratch；
3. multi-query aggregation 在 r9 已经稳定高于随机并有应用价值。
```

否则 r10 不作为训练任务启动，只保留为研究方向。

### Stage H5：r8 integral/inverse-round data screen

这是并行准备的新高轮数据结构路线，不抢占当前 r8 pairset 1M 和 r9 weak-probe。

计划文件：

```text
docs/experiments/innovation1-present-r8-integral-inverse-feature-screen-plan.md
configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_smoke.csv
configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0.csv
```

触发条件：

```text
1. 当前 GPU watcher 释放资源；
2. r9 weak-probe near-random 或 r8 pairset 显示需要更强数据表示；
3. 用户明确选择高轮数据结构路线优先于继续同协议微调。
```

首轮只跑 r8 `65536/class` screen。若 InvP+Sinv candidate 明显强于 raw/invP
anchor，再进入 `262144/class` confirmation。

## 6. 当前行动

本轮应做：

```text
1. 写本研究蓝图。
2. 写 r9 weak-probe 实验计划。
3. 准备 r9 smoke CSV 和 r9 262144/class conditional CSV。
4. 跑本地解析/结构测试；可选 tiny CPU smoke。
5. 提交并推送。
6. 不启动 r9 remote，等待 r8 watcher gate。
```

这让项目继续向高轮目标前进，同时不破坏现有 r7/r8 远程闭环。

## 6.1 2026-07-05 更新：r8 已触发 r9 weak-probe

`i1_present_r8_round_extension_262k_seed0_gpu0_20260704` 已经 retrieved /
validated / plotted / gate-note generated。结果显示：

```text
best_model = present_nibble_invp_pair_consistency_spn_only
best_auc = 0.552908501064
baseline_auc = 0.540348751209
delta_vs_baseline = +0.012559749855
decision = support_scale_r8_to_1m_seed0
```

这改变当前高轮路线的状态：

```text
H0 r8 watcher result = complete
H1 r9 smoke = passed
H2 r9 262144/class weak probe = launchable
```

新的研究解释：

```text
r8 信号不是 InvP-only 单独保留下来的，而是 pair-set consistency 明显更强。
因此高轮推进的主要候选暂时应从“单样本 InvP-only”转向
“InvP-aligned pair-set evidence pooling”。
```

下一步：

```text
启动 r9 262144/class weak-probe；
并行准备 r8 1M pair-set confirmation；
如果 r9 仍有弱正信号，再决定 r9 seed1 / r8 pair-set seed1 /
frozen aggregation control / curriculum-transfer 哪个优先。
```

## 6.2 2026-07-05 更新：更高轮次阶段的条件分支

当前更高轮次任务不再只是“把 rounds 改大”。新的阶段目标是：

```text
先确认 r8 pair-set 信号是否在 1M/class 保留；
同时观察 r9 weak-probe 是否还有单样本或 pair-set 弱信号；
再根据 gate 选择 r9 seed/scale、r9 curriculum、r9 difference screen，
最后才考虑 r10 weak screen。
```

分支状态：

| 分支 | 文档 | 状态 | 作用 |
|---|---|---|---|
| r9 curriculum | `docs/experiments/innovation1-present-r9-curriculum-from-r8-plan.md` | launched / watcher-managed | r9 from-scratch 弱时测试训练路径 |
| r9 difference screen | `docs/experiments/innovation1-present-r9-difference-screen-plan.md` | prepared / not launched | 测试高轮输入差分是否是瓶颈 |
| r8 integral/inverse feature screen | `docs/experiments/innovation1-present-r8-integral-inverse-feature-screen-plan.md` | launched / watcher-managed | 测试 high-round 数据结构是否比 raw pair 更能暴露 SPN previous-round signal |
| r8 pair-mixer | `docs/experiments/innovation1-present-pair-mixer-consistency-plan.md` | prepared / not launched | 测试 pair embedding 交互是否增强高轮信号 |
| r10 conditional weak-probe | `docs/experiments/innovation1-present-r10-conditional-weak-probe-plan.md` | planned only | 只有 r9 gate 支持时才创建资产 |

r10 的规则尤其要严格：

```text
没有 r9 retrieved / validated / gate-noted 结果前，不创建 r10 remote config，
不占 GPU，不写 r10 成功/失败。
```

这样更像一个高轮研究阶梯，而不是并行乱开一堆不可归因的高轮训练。

## 6.3 2026-07-05 更新：当前执行状态

本地 bounded monitor 检查显示，当前 active high-round watcher 是：

```text
i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705
  status = running
  results_jsonl_line_count = 0 / 2
  stage = training
  latest local progress = epoch 5 / 22 on row 1
  best_checkpoint_metric_so_far = 0.5018549287342466
  evidence status = running, not final

i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705
  status = launched / watcher-managed
  expected_rows = 3
  launch evidence = started.marker / readiness / progress observed remotely
  local watcher = running, waiting for next artifact sync
  evidence status = running, not final
```

当前主线程策略：

```text
不 SSH 轮询；
不启动 r9 difference screen / r8 pair-mixer / r10；
等待两个 active watcher 中任一结果 ready 后，先 route-specific postprocess，
再按下面的仲裁规则选择下一步。
```

这意味着更高轮次研究已经进入“训练路径 vs 数据结构”双候选诊断阶段。
在其中一个结果 ready 前继续抢跑 r9 difference / r10 会降低可归因性。

仲裁规则：

```text
1. 若 r9 curriculum best AUC > r9 from-scratch best AUC + 0.003：
   支持 training-path hypothesis，准备 seed1 或 attribution control；
2. 若 r9 curriculum 仍接近随机，但 r8 integral/inverse 有明显正信号：
   优先 262144/class integral/inverse confirmation；
3. 若两者都弱：
   不开 r10 from-scratch，转 r9 difference screen 或新的 SPN-aware data route；
4. 若两者互相矛盾：
   先写 variance / protocol / attribution 分析，不盲目扩大 GPU。
```

## 7. 参考来源

- Jain, Kohli, and Mishra, *Deep Learning based Differential Distinguisher for Lightweight Block Ciphers*, arXiv:2112.05061: https://arxiv.org/abs/2112.05061
- Zhang and Wang, *Improving Differential-Neural Distinguisher Model For DES, Chaskey, and PRESENT*, arXiv:2204.06341: https://arxiv.org/abs/2204.06341
- Wu and Guo, *Improved integral neural distinguisher model for lightweight cipher PRESENT*, local paper note: `papers/innovation_one/grobid_md/improved-integral-neural-distinguisher-model-for-lightweight-cipher-present.md`
- Entropy-based PRESENT key-recovery/distinguisher synthesis, local text: `papers/innovation_one/text/2026_present_entropy_nd.txt`
- Generic Partial Decryption / AutoND feature-engineering route, local note: `papers/innovation_one/grobid_md/generic-partial-decryption-as-feature-engineering-for-neural-distinguishers.md`
- IoT-friendly SPN framework, local text: `papers/innovation_one/text/2026_liu_spn_iot_friendly_neural_distinguisher_framework.txt`
- 本项目综述：`docs/research/spn_structured_nn_research_plan.md`
- 本项目 r8 阶梯：`docs/research/innovation1-present-round-extension-research-plan.md`

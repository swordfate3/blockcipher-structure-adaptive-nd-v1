# 创新2 E94：RECTANGLE-80 四轮 7/8/9-bit 嵌套 cube 单调标签门

日期：2026-07-19

状态：已完成 / `pass` / 允许 E95 无训练机制门

## 1. 研究问题

E88 已经在 192 个 8-bit cube 上确认 RECTANGLE-80 四轮严格 unit-output 标签足够宽，E93 则停止了
当前 unit-profile benchmark 上继续枚举网络。E94 不训练模型，只回答一个新的前置问题：

```text
同一个输入 cube 从 7 bit 扩到 8 bit、再扩到 9 bit 时，是否能得到 sound、足够宽、
chain-disjoint 且无法被维度/位置一元捷径解释的严格标签？
```

若答案为是，下一次实验才允许设计 `Monotone Cube-Lattice Operator`；若答案为否，不得用
dimension embedding、Transformer、GNN 或更多 epoch 绕过标签门。

## 2. 数学机制

设 `A` 是活动 bit 集合，`B` 是它的超集。对固定 key 和 `B` 外的 inactive offset，`B`-cube 的
输出 XOR 可以写成所有 `B \ A` 取值对应的 `A`-coset cube XOR 之和。因此：

```text
若 A 对所有 key 和所有 inactive offset 都平衡，则任意 B ⊇ A 也平衡。
```

所以 universal-balance 标签沿 cube 子集格单调：

```text
p(balance | d7) <= p(balance | d8) <= p(balance | d9)
```

这不是经验假设。E94 必须验证实现中不存在 `d7 positive -> d8 negative`、
`d8 positive -> d9 negative` 或 `d7 positive -> d9 negative`；unknown 不能当 negative。

## 3. 冻结协议

```text
cipher                    = final RECTANGLE-80 published specification
rounds                    = 4
anchor                    = E88, 192 frozen 8-bit structures
chain count               = 192
dimensions                = 7 / 8 / 9 active bits
outputs                    = 64 unit output bits
witness keys              = E88 same 16 deterministic 80-bit keys
inactive offsets          = 8 / chain node
training                  = no
remote                    = no
```

对 E88 的每个 8-bit 活动集 `A8`：

```text
removed bit = A8[chain_index mod 8]
A7          = A8 without removed bit
added bit   = sorted([0..63] \ A8)[chain_index mod 56]
A9          = A8 union added bit
```

三节点共享同一个 `chain_id`；`chain_index % 4 == 0` 为 validation，其余为 train。拆分单位是整条链，
严禁将同一条链的不同维度放入不同 split。

## 4. 标签语义

每个 `(chain, dimension, output_bit)` 只有三种原始状态：

```text
positive = full-cube monomial 不在 sound active-variable ANF-support over-approximation 中
negative = 一个具体 scheduled 80-bit key 与 inactive offset 使实际 unit cube XOR = 1
unknown  = 尚无 positive certificate，也未找到 concrete negative witness
```

允许使用上面的数学定理做 sound positive closure：`d7 positive` 可将 unresolved 的 d8/d9 标为
inherited positive，`d8 positive` 可将 unresolved 的 d9 标为 inherited positive。closure 不得覆盖
negative；一旦冲突，协议直接 fail。E88 的 d8 原始三态标签必须逐项重放一致。

## 5. 同预算控制与门槛

E94 不比较神经网络。锚点是 E88 的同一密码、轮数、key bank、offset 数和 192 条结构定义；唯一变量是
相邻活动维度及其严格嵌套关系。

协议门：

```text
E88 status/decision/hash/192 structures replay
final RECTANGLE vector path and scalar witnesses validate
all chains satisfy A7 proper-subset A8 proper-subset A9
all three nodes in a chain use the same split
all positive rows have direct or inherited sound certificate
all negative rows retain a concrete key/offset witness
positive/negative conflicts = 0
```

单调门：

```text
d7 positive -> d8 negative violations = 0
d8 positive -> d9 negative violations = 0
d7 positive -> d9 negative violations = 0
positive prevalence is nondecreasing from d7 to d8 to d9 after closure
```

宽度门：

```text
每个维度 raw positive >= 128
每个维度 raw negative >= 128
每个维度至少 24 个 mixed-output chains
至少 32 条 chain 含一个跨维度状态变化的 resolved output
matched train 每维每类 >= 96
matched validation 每维每类 >= 32
matched validation 覆盖 >= 8 chains、>= 16 output bits
```

抗捷径门：

```text
matched duplicate rows = 0
每个 dimension 内 structure/output class delta = 0
structure-disjoint validation 上最强 unary baseline
  (dimension / output bit / removed bit / added bit / active-position mean) <= 0.65
```

## 6. 裁决

```text
协议或 soundness 失败:
  status   = fail
  decision = innovation2_rectangle80_nested_cube_monotonic_protocol_invalid

标签宽度、matching 或抗捷径不足:
  status   = hold
  decision = innovation2_rectangle80_nested_cube_monotonic_labels_not_ready

全部通过:
  status   = pass
  decision = innovation2_rectangle80_nested_cube_monotonic_labels_ready
```

通过后下一步不是远程或长训练，而是本地确定性机制门：比较真实 nesting、打乱 nesting 和错误 superset
关系在同容量单调聚合表示上的 margin。只有该机制门通过，才开放两轮
`Monotone Cube-Lattice Operator` readiness；两轮通过后才允许 30 轮 seed0，再按门决定 seed1。

## 7. 计划产物

```text
run_id = i2_rectangle80_r4_nested_cube_monotonic_readiness_20260719
output = outputs/local_audits/i2_rectangle80_r4_nested_cube_monotonic_readiness_20260719

atlas.jsonl
chains.json
matched_nested_contrast.csv
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

本实验只能支持“相邻 cube 维度严格标签与单调机制可用/不可用”的结论，不构成神经收益、高轮区分器、
积分攻击、跨密码泛化或 SOTA。

## 8. 实际结果

E88 的 `gate.json`、`atlas.jsonl` 和 `structures.json` 分别按预注册 SHA-256 重放一致，192 条链全部满足
严格 `A7 ⊂ A8 ⊂ A9`，train/validation 以整条链拆分。三档原始严格标签为：

| 维度 | positive | negative | unknown | resolved positive prevalence |
|---:|---:|---:|---:|---:|
| 7-bit | 9,259 | 2,466 | 563 | 0.789680 |
| 8-bit | 9,995 | 1,791 | 502 | 0.848040 |
| 9-bit | 10,650 | 1,195 | 443 | 0.899114 |

本次不需要用定理补齐 unknown：`inherited_positive_d8=0`、`inherited_positive_d9=0`。直接证书本身已经满足：

```text
d7 positive -> d8 negative = 0
d8 positive -> d9 negative = 0
d7 positive -> d9 negative = 0
```

36 个按维度分层抽取的 concrete negative witness 全部由标量 RECTANGLE-80 重算通过；向量化 fixture
也为 `16/16`。跨维度至少出现一个 resolved `negative -> positive` 状态变化的链为 `191/192`。

匹配后容量：

| 维度 | train 每类 | validation 每类 | validation chains | validation output bits |
|---:|---:|---:|---:|---:|
| 7-bit | 1,566 | 488 | 48 | 52 |
| 8-bit | 1,208 | 388 | 48 | 41 |
| 9-bit | 834 | 294 | 48 | 36 |

所有 dimension 内 chain/output class delta 为零，重复行数为零。dimension、output bit、removed bit、
added bit 和 active-position mean 的 structure-disjoint 验证 AUC 均为 `0.500`。

## 9. 裁决

```text
status   = pass
decision = innovation2_rectangle80_nested_cube_monotonic_labels_ready
training = no
remote   = no
```

E94 证明 RECTANGLE-80 四轮上存在 sound、宽、chain-disjoint、无一元捷径且符合 cube 格单调定理的
7/8/9-bit unit-output 标签。这是新的输出任务/标签机制证据，不是神经网络收益，也不改变 E93 对第三
SPN 正式神经结果仍未确认的结论。

## 10. 推荐下一步：E95

下一次唯一允许的实验是本地无训练确定性机制门：

```text
question = 正确的 A7 ⊂ A8 ⊂ A9 关系，是否比独立节点与错误关系提供可归因的标签信息？
anchor   = E94 matched rows、冻结 chain split、相同 r1-r3 prefix summaries
variable = 只改变 chain relation

rows:
  independent_dimension = 三个维度独立、没有跨维度边
  true_nesting          = E94 真实 A7 ⊂ A8 ⊂ A9
  shuffled_nesting      = split 内打乱 chain 对应，保持维度与边际分布
  wrong_superset        = 连接到不包含 A7/A8 的同维度节点
  unconstrained_control = 同输入、同特征数，但不施加单调投影

training = no neural training; deterministic ridge/isotonic fitting only
scale    = 192 chains, all E94 matched edges
seeds    = none; fixed deterministic permutations
epochs   = 0
device   = local CPU
remote   = no
```

E95 必须保持 train/validation chain-disjoint，并要求真实 nesting 相对 independent、shuffled 和 wrong
superset 的 structure-disjoint AUC margin 均至少 `+0.03`，同时不劣于同容量 unconstrained control
超过 `0.01`。通过才开放两轮 `Monotone Cube-Lattice Operator`；失败则关闭该路线，不增加
embedding、hidden width、网络层数、epoch，也不远程扩规模。

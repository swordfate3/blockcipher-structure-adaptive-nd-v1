# 创新2 E22：SKINNY-64/64 8轮相邻活动 pair kernel 多样性审计

日期：2026-07-17
状态：已完成 / hold

## 1. 研究问题

E20/E21 已分别精确复现 SKINNY-64/64 7轮18维和8轮1维论文 kernel，但已知
fixture 本身不足以构造结构条件化输出预测任务。E22 只改变活动 cell pair 的
位置，回答：

> 固定 SKINNY-64/64 8轮、two-active-cell 宽度、密钥、inactive context 和 raw
> 输出格式后，16个循环相邻 cell pair 是否产生多个稳定且不同的 joint kernel？

如果只有论文 `(14,15)` anchor 非平凡，则结构字段几乎等同于一个身份开关，无法
形成非退化的 `structure + output mask -> balance property` 学习问题。

## 2. 固定协议

```text
cipher / rounds = SKINNY-64/64 / 8
state = 4 x 4 row-major nibbles
output feature = raw 64-bit parity, MSB-first
structures = 16 cyclic adjacent pairs:
  (0,1),(1,2),...,(14,15),(15,0)
paper anchor = (14,15) -> span{b28 xor b44 xor b60}
same-budget control = (0,1)
inactive context = one deterministic random 64-bit base plaintext
plaintexts per structure per key = 16^2 = 256
discovery keys = 64
validation keys = 64 fresh and disjoint
joint keys = 128 unique, disjoint from all E20/E21 keys
training = none
device = local CPU
```

所有16个结构复用同一 base plaintext 和同一密钥顺序，唯一变量是活动 pair。
E22 是本地 sampled-key geometry readiness，不是 paper-scale、全密钥证明或神经
训练。

## 3. 有限密钥秩亏控制

一个随机 `64 x 64` GF(2) 矩阵本身有明显概率不是 full rank，因此 discovery 或
validation half 单独出现非零 kernel 不能作为多样性证据。E22 的非平凡结构定义
必须使用128-key joint matrix：

```text
nontrivial structure := joint_nullity > 0
```

每个 discovery、validation、joint basis 都需通过自己的 `M u = 0` 校验；joint
basis 必须显式在 discovery 和 validation 两半上分别复核。另报告 discovery basis
在 validation 的存活比例，以显示 half-level 候选中有多少只是有限密钥噪声。

子空间签名使用确定性 RREF/kernel basis，不枚举高维 span。

## 4. 同预算锚点与控制

- `(14,15)` 的 discovery、validation 两半必须都包含论文一维
  `b28 xor b44 xor b60` span，128-key joint kernel 必须精确等于该 span；half
  可以包含额外有限密钥秩亏方向，但这些方向不能进入 joint anchor。
- `(0,1)` 作为 E21 同预算控制继续报告，但不预设必须 full rank；其实际结果由
  joint kernel 裁决。
- E22 密钥必须与 E20 seed `8201`、E21 seed `9201` 的两组768把密钥完全互斥。
- Appendix B 的 SKINNY-64/64 32轮公开向量必须继续通过。

## 5. Readiness 与推进门槛

协议门：

```text
public vector passes
exact 16 pair list and anchor/control ownership pass
128 keys unique; 64/64 halves disjoint
E22 keys disjoint from E20/E21
parity_rows.npy shape = 16 x 128, dtype = uint64
all bases validate their matrices and both halves
all metrics finite
```

推进门全部满足：

```text
anchor paper span validates on discovery and validation halves
anchor joint rank/nullity = 63/1
anchor joint span = span{b28 xor b44 xor b60}
nontrivial joint-kernel structures >= 4
distinct nontrivial joint-kernel signatures >= 4
mean discovery-basis validation survival over nontrivial structures >= 0.50
```

裁决：

```text
pass:
  decision = innovation2_skinny_r8_geometry_kernel_diversity_ready
  next = E23 structure-mask label construction and shortcut audit

hold:
  decision = innovation2_skinny_r8_geometry_kernel_not_diverse
  next = stop cyclic adjacent-pair expansion; rank literature-backed alternatives

fail:
  decision = innovation2_skinny_r8_geometry_protocol_invalid
  next = repair anchor, key ownership, pair enumeration, cache, or GF(2) logic
```

任何分支都保持 `training=no`、`remote_scale=no`。只有 pass 才允许把 E22 kernel
转成标签表；hold 时不得通过增加网络、seed 或远程样本掩盖结构族退化。

## 6. 产物与可视化

run id：

```text
i2_skinny64_r8_adjacent_pair_kernel_diversity_128keys_seed0_20260717
```

```text
outputs/local_audits/<run_id>/
  results.jsonl
  kernel_basis.csv
  keys.npy
  parity_rows.npy
  metadata.json
  progress.jsonl
  gate.json
  curves.svg
  visual_qa_passed.marker
```

可视化必须显示16个 pair 的 discovery/validation/joint nullity、joint signature
类别、论文 anchor 和 control 位置、`>=4` 多样性门槛及中文裁决。生成后必须执行
`visual-qa-redraw` 像素检查，完成结果后刷新最近结果索引。

## 7. 通过后的边界

E22 通过只表示8轮相邻 pair 产生足够多的稳定输出子空间。E23 仍需把 kernel 转成
结构-mask 标签，并在任何神经训练前执行 structure/mask prevalence、身份与重量
边际、bit-pattern、group-disjoint、fresh-key 和有限密钥噪声控制。只有残差交互
可重复，才进入真正的结构条件化积分输出预测模型。

## 8. 2026-07-17 E22 结果与校准修正

真实运行完成全部 `16 x 128 x 256 = 524288` 次8轮加密。使用固定 base plaintext
`0x4731457D741EBC76`，E22 的128把密钥与 E20/E21 两组768把密钥完全不相交；
公开32轮向量、16-pair所有权、缓存形状、所有 GF(2) basis 和 joint basis 两半复核
全部通过。

第一次裁决返回 `protocol_invalid`，原因不是密码或缓存错误，而是预注册内部存在
矛盾：第三节明确指出64-key half 会随机秩亏、只能由128-key joint kernel 裁决，
但第五节仍要求 anchor 两个 half 都必须恰好 `63/1`。本次 anchor 实际为：

```text
discovery: rank/nullity = 63/1, span = paper span
validation: rank/nullity = 62/2, contains paper span plus one random half-only direction
joint:      rank/nullity = 63/1, span = paper span
```

因此校准修正为“两半都包含论文方向，joint 必须精确等于论文一维 span”。多样性
阈值、结构、密钥、结果数据和其他门槛均未改变，也没有重复加密；直接从冻结的
`keys.npy` 与 `parity_rows.npy` 重新裁决。

128-key joint matrix 只有3个非平凡结构：

| 活动 pair | joint rank/nullity | 稳定输出 mask | discovery -> validation |
|---|---:|---|---:|
| `(12,13)` | `63/1` | `b20 xor b36 xor b52` | `1/1` |
| `(13,14)` | `63/1` | `b24 xor b40 xor b56` | `1/1` |
| `(14,15)` | `63/1` | `b28 xor b44 xor b60` | `1/1` |

其余13个 pair 的 joint matrix 均 full rank、nullity为0。half-level 出现的多个高
汉明重量 basis 全部没有跨 half 存活，验证了联合矩阵控制的必要性。

最终指标：

```text
nontrivial joint-kernel structures = 3 / 16   (gate >= 4)
distinct nontrivial signatures     = 3        (gate >= 4)
mean discovery-basis survival      = 1.000    (gate >= 0.50)
```

最终裁决：

```text
status = hold
decision = innovation2_skinny_r8_geometry_kernel_not_diverse
training = no
remote_scale = no
```

这不是“8轮没有输出性质”。三个稳定方向真实存在，并形成连续平移模式；但数量未
达到预注册的非退化标签门，不能直接把16-pair表交给神经网络，也不继续机械扩所有
循环相邻 pair。

权威产物：

```text
outputs/local_audits/
  i2_skinny64_r8_adjacent_pair_kernel_diversity_128keys_seed0_20260717/
```

最终 `curves.svg` 渲染为 `1800 x 830` 并通过 `visual-qa-redraw`。16个 pair 标签、
three-way nullity、128-key签名类别、anchor/control、多样性门槛和 hold 裁决均无
重叠、裁切、缺字或轴范围歧义。

## 9. 推荐下一步：E23 底行 complete-pair closure fresh-key 审判

三个存活结构不是任意散点，而恰好是4x4状态底行 cells `{12,13,14,15}` 的三个
相邻 pair，输出 mask 也按4 bit平移。E23 不扩全部120个 two-cell 组合，而是用
全新密钥检验这个由 E22 提出的最小结构假设：底行完整无序 pair 族是否闭合。

```text
target structures = C({12,13,14,15}, 2) = 6 pairs:
  known from E22 = (12,13),(13,14),(14,15)
  new holdout geometry = (12,14),(12,15),(13,15)
same-budget negative control = (0,1)
rounds / feature = 8 / raw 64-bit parity MSB-first
keys = 128 fresh = 64 discovery + 64 validation
keys disjoint from E20/E21/E22
plaintexts per structure per key = 256
training = none
execution = local CPU with disk-backed parity cache and progress JSONL
```

E23 必须首先在新密钥上精确复现 E22 的三个已知 joint kernel，然后裁决三个未见
非相邻 pair。建议推进门为底行6个 pair 中至少5个具有非零稳定 joint kernel、
至少5种签名、平均 half 存活率 `>=0.50`，且 `(0,1)` 不产生同一族方向。通过后
才进入结构-mask shortcut audit；失败则停止 SKINNY r8 bottom-row geometry 路线，
转 Hwang 的其他正文主案例或更高信息量结构，不训练网络、不上远程 GPU。

# 创新2 E23：SKINNY-64/64 8轮底行 complete-pair closure fresh-key 审判

日期：2026-07-17
状态：已完成 / hold

## 1. 研究问题

E22 在16个循环相邻 two-active-cell 结构中只找到三个稳定 joint kernel：

```text
(12,13) -> b20 xor b36 xor b52
(13,14) -> b24 xor b40 xor b56
(14,15) -> b28 xor b44 xor b60
```

三个结构恰好覆盖 SKINNY 4x4 状态底行 `{12,13,14,15}` 的三个相邻 pair，
输出方向也按4 bit平移。E23 只检验由 E22 产生的最小新假设：

> 底行全部6个无序 two-active-cell pair 是否形成稳定、非退化且位置相关的输出
> kernel 家族？

E23 不扩展全部120个 two-cell 组合，也不训练神经网络。它先判断是否存在足以构造
`structure + output mask -> balance property` 标签表的底行结构族。

## 2. 固定协议

```text
cipher / rounds = SKINNY-64/64 / 8
state layout = 4 x 4 row-major nibbles
output feature = raw 64-bit parity, MSB-first
bottom-row structures =
  (12,13),(12,14),(12,15),(13,14),(13,15),(14,15)
same-budget negative control = (0,1)
inactive context = one deterministic random 64-bit base plaintext
plaintexts per structure per key = 16^2 = 256
discovery keys = 64
validation keys = 64 fresh and disjoint
joint keys = 128 unique
key generation seed = 11201
base plaintext seed = 11202
training = none
device = local CPU
```

E23 的128把密钥必须与 E20 seed `8201` 的768把、E21 seed `9201` 的768把和
E22 seed `10201` 的128把完全互斥。所有7个结构复用同一 base plaintext、密钥
顺序与预算，唯一变量是活动 cell pair。

## 3. 冻结 anchor 与控制

三个 E22 anchor 使用全新密钥复验：

```text
(12,13) expected mask = 0x0000080008000800 = b20 xor b36 xor b52
(13,14) expected mask = 0x0000008000800080 = b24 xor b40 xor b56
(14,15) expected mask = 0x0000000800080008 = b28 xor b44 xor b60
```

每个 expected mask 必须分别满足：

```text
discovery 64-key matrix contains expected direction
validation 64-key matrix contains expected direction
joint 128-key matrix has rank/nullity = 63/1
joint kernel exactly equals expected one-dimensional span
```

两个 half 可以出现额外的有限密钥随机秩亏方向；这些方向不得进入128-key joint
anchor。负对照 `(0,1)` 的 joint kernel 不得包含上述任一 anchor direction。

## 4. Readiness 与推进门

协议门：

```text
SKINNY Appendix B 32-round public vector passes
exact six bottom-row pairs plus one control
128 keys unique; discovery/validation halves disjoint
E23 keys disjoint from E20/E21/E22
parity_rows.npy shape = 7 x 128, dtype = uint64
all computed bases validate their own matrices
all joint bases validate on both halves
all metrics finite
```

推进门全部满足：

```text
three known E22 anchors reproduce exactly on fresh keys
bottom-row nontrivial joint kernels >= 5 / 6
bottom-row distinct joint signatures >= 5
mean discovery-basis validation survival >= 0.50
control (0,1) excludes all three anchor-family directions
```

裁决：

```text
pass:
  decision = innovation2_skinny_r8_bottom_row_pair_family_ready
  next = E24 structure-mask label construction and shortcut audit

hold, known anchors fail:
  decision = innovation2_skinny_r8_bottom_row_anchor_not_reproduced
  next = stop SKINNY r8 bottom-row geometry and rank other Hwang main cases

hold, diversity/control fail:
  decision = innovation2_skinny_r8_bottom_row_pair_family_not_closed
  next = stop SKINNY r8 bottom-row geometry and rank other Hwang main cases

fail:
  decision = innovation2_skinny_r8_bottom_row_protocol_invalid
  next = repair pair ownership, key ownership, cache, public vector, or GF(2) logic
```

任何分支都保持 `training=no`、`remote_scale=no`。不得通过增加网络、seed 或远程
样本掩盖 anchor 不稳定、结构族退化或负对照泄漏。

## 5. 产物与执行

run id：

```text
i2_skinny64_r8_bottom_row_pair_closure_128keys_seed0_20260717
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

执行命令：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/audit-innovation2-skinny-r8-bottom-row-closure \
  --run-id i2_skinny64_r8_bottom_row_pair_closure_128keys_seed0_20260717 \
  --output-root outputs/local_audits/i2_skinny64_r8_bottom_row_pair_closure_128keys_seed0_20260717
```

`keys.npy` 与 `parity_rows.npy` 是冻结的磁盘回放证据；完成后必须从它们重新调用
纯裁决函数并证明重放 `gate.json` 完全一致。曲线必须经过 `visual-qa-redraw` 的真实
像素检查，随后刷新最近结果索引。

## 6. 通过后的 E24 边界

E23 通过仍不允许直接声称神经网络学到了积分传播。E24 必须把稳定 kernel 转为
正负 structure-mask 标签，并同时审计：

```text
label prevalence
structure identity and mask identity
mask weight / bit-pattern
structure-only and mask-only baselines
group-disjoint context/structure/mask splits
accuracy and AUC
fresh-key label stability
```

只有神经候选在同预算强控制上保留可重复的结构-mask交互残差，才允许进入真正的
结构条件化积分输出性质预测训练。

## 7. 2026-07-17 E23 结果

真实运行完成全部 `7 x 128 x 256 = 229376` 次8轮加密。固定 base plaintext 为
`0x9081EA11C2F7D507`；E23 的128把密钥与 E20/E21/E22 完全互斥。公开32轮向量、
结构所有权、密钥所有权、缓存形状、全部 GF(2) basis 及 joint basis 两半复核均
通过。

三个 E22 anchor 在全新密钥上全部精确复现：

| 活动 pair | discovery | validation | joint | 稳定输出方向 |
|---|---:|---:|---:|---|
| `(12,13)` | `63/1` | `63/1` | `63/1` | `b20 xor b36 xor b52` |
| `(13,14)` | `63/1` | `63/1` | `63/1` | `b24 xor b40 xor b56` |
| `(14,15)` | `61/3` | `63/1` | `63/1` | `b28 xor b44 xor b60` |

其中 `(14,15)` discovery half 的另外两个高重量方向没有进入 validation/joint，
属于预期的64-row随机秩亏，不影响 anchor 精确复现。

三个 E23 holdout pair 的结果：

| 活动 pair | joint rank/nullity | 稳定输出方向 | 结论 |
|---|---:|---|---|
| `(12,14)` | `64/0` | 无 | 空 kernel |
| `(12,15)` | `63/1` | `b16 xor b32 xor b48` | 新稳定方向 |
| `(13,15)` | `64/0` | 无 | 空 kernel |

负对照 `(0,1)` 的 joint matrix 为 `64/0`，且不包含三个 anchor-family 方向。最终
指标为：

```text
bottom-row nontrivial joint kernels = 4 / 6  (gate >= 5)
distinct joint signatures           = 4      (gate >= 5)
mean half survival                  = 0.667  (gate >= 0.50)
control contains anchor family      = false
```

最终裁决：

```text
status = hold
decision = innovation2_skinny_r8_bottom_row_pair_family_not_closed
training = no
remote_scale = no
```

这不是“8轮不存在输出性质”：四个方向在128把 fresh keys 上稳定，其中三个还完成了
独立的 E22 -> E23 复验。但底行 complete-pair 假设不成立，稳定结构和签名都只达到
`4/6`，低于冻结的 `5/6` 非退化门。因此停止 SKINNY 8轮 bottom-row pair geometry，
不扩全部120个 pair、不增加 seed、不训练网络。

冻结 `keys.npy` 与 `parity_rows.npy` 重放得到的 gate 与权威 `gate.json` 完全一致。
最终 `curves.svg` 渲染为 `1800 x 844`，并通过 `visual-qa-redraw`：中文标题、图例、
7个 pair 标签、两面板坐标、底部指标与裁决均无重叠、裁切、缺字或含义歧义。

## 8. 推荐下一步：E24 SKINNY 7轮 all-single-cell geometry

E23 已否定8轮底行 complete-pair 闭合，但 Hwang 正文最强的直接 kernel 主案例仍是
7轮单活动 cell 的18维完整输出空间。E20 只比较论文 cell15 与 control cell0，尚未
检查其余14个 cell。因此下一步固定同一7轮协议，只改变单活动 cell 的位置：

```text
research question = 16个单活动 cell 是否产生多个稳定且不同的7轮 joint kernel
same-budget anchor = cell15 的 Hwang Table 2 18维 exact span
same-budget control = cell0，不得包含 Hwang span
one variable = active cell 0..15
keys = 128 fresh = 64 discovery + 64 validation
key seed / base seed = 12201 / 12202
keys disjoint from E20/E21/E22/E23
plaintexts per structure per key = 16
feature = raw 64-bit parity, MSB-first
training = none
execution = local CPU
advance = nontrivial structures >= 6，distinct signatures >= 4，mean survival >= 0.50
pass next = E25 structure-mask label construction and shortcut audit
hold next = stop SKINNY position geometry; reproduce Hwang SPECK7 or Midori nonlinear case
forbidden = reopen r8 pair enumeration, add seed to move the E23 gate, neural/remote scale
```

选择 E24 而不是直接实现 Midori 或11轮 MILP，是因为它复用已经由公开向量和 E20
验证的 SKINNY 实现，唯一变量清楚，且能以约 `16 x 128 x 16 = 32768` 次加密判断
7轮正文主案例能否形成非退化结构标签族。

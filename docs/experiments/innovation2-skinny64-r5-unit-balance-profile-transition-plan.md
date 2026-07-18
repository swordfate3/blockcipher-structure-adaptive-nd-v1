# 创新2 E82：SKINNY-64五轮严格单位输出平衡谱标签过渡计划

日期：2026-07-19

状态：已完成 / pass / 无神经训练

## 1. 研究问题

E81四轮协议全部通过，但`5563/71/510`的positive/negative/unknown分布使resolved positive
prevalence达到`0.987398`，只有10/96个mixed结构。它同时失败raw class、prevalence、mixed和
checkerboard结构覆盖，不值得机械扩到192结构。

E82只回答：

```text
保持同一批96个8-bit cube和相同严格标签协议，
只将SKINNY-64/64目标轮数4 -> 5，
是否能把unit-output平衡谱从“几乎全平衡”推进到正负可训练过渡区？
```

## 2. 冻结锚点与唯一变量

锚点：

```text
i2_skinny64_r4_unit_balance_profile_readiness_20260719
status   = hold
decision = innovation2_skinny64_unit_balance_profile_not_ready
```

E82必须读取E81的`gate.json`、`structures.json`、`metadata.json`并记录SHA-256。以下逐项不变：

```text
structure_count       = 96
structure_seed        = 20260718
witness_keys          = 16
key_seed              = 7401
offsets_per_structure = 8
offset_seed           = 17401
match_attempts        = 64
split                 = structure_index % 4
active_dimension      = 8
output_bits           = 64
strict label semantics and all gates
```

唯一变量：`rounds = 4 -> 5`。E82重新生成标签，不使用E81标签训练或选择结构。

## 3. 标签与前缀

标签仍为：

```text
positive = 五轮输出中完整8变量cube单项式不在sound ANF support over-approx
negative = 一个真实scheduled 64-bit key和inactive offset给出unit XOR one
unknown  = 未证明也未找到反例
```

为未来readiness保留r1-r4前缀，每轮13维，共`52`维；E82不解释或训练这些特征。若标签门通过，
下一实验只使用最后的r4切片13维测试同一4,795参数算子思想，不能直接复用PRESENT/GIFT权重。

## 4. 协议门

除E81全部密码、向量化、support、witness和split门外，还要求：

1. E81来源hash存在且run/status/decision匹配；
2. 96个结构定义逐项重放；
3. 五轮向量化加密与标量`Skinny64(rounds=5)`一致；
4. 标签shape为`96 x 64`，前缀shape为`96 x 64 x 52`；
5. E81与E82的rounds字段确实为4和5。

任一失败均为protocol invalid。

## 5. 宽度、捷径与裁决

完全复用E81/GIFT E74门：raw每类至少256、resolved prevalence在`[0.10,0.90]`、mixed至少
32、matched train至少150/class、validation至少50/class、结构/输出覆盖、最强一元AUC不超过
0.65且行列class delta为0。

```text
if protocol invalid:
  status   = fail
  decision = innovation2_skinny64_r5_unit_balance_profile_transition_protocol_invalid
elif width or shortcut fails:
  status   = hold
  decision = innovation2_skinny64_r5_unit_balance_profile_transition_not_ready
else:
  status   = pass
  decision = innovation2_skinny64_r5_unit_balance_profile_transition_ready
```

## 6. 产物

```text
run_id = i2_skinny64_r5_unit_balance_profile_transition_20260719
output = outputs/local_audits/i2_skinny64_r5_unit_balance_profile_transition_20260719

atlas.jsonl
matched_unit_contrast.csv
structures.json
profile_targets.npy
profile_observed.npy
prefix_features.npy
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

## 7. 推荐分支

```text
pass:
  下一步 = E83 SKINNY r4-only三行两轮本地readiness

hold且raw正负已宽、仅matching容量不足:
  下一步 = 计算192结构packing上界后再裁决是否扩结构

hold且positive不足、unknown主导或仍严重单类化:
  下一步 = 停止SKINNY 8-bit unit-profile跨轮扫描，转新的sound标签表示

fail:
  下一步 = 修复E81锚点、五轮向量化、support或witness协议
```

E82不训练、不远程；结果不得写成五轮神经区分器或攻击。

## 8. 完成结果

运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/audit-innovation2-skinny64-r5-unit-balance-profile-transition \
  --protocol e82 \
  --run-id i2_skinny64_r5_unit_balance_profile_transition_20260719 \
  --anchor-root outputs/local_audits/i2_skinny64_r4_unit_balance_profile_readiness_20260719 \
  --output-root outputs/local_audits/i2_skinny64_r5_unit_balance_profile_transition_20260719
```

E81来源hash、run/status/decision、四轮协议与96个结构定义全部重放；五轮向量化、support、
52维前缀、24个negative witness和split协议全部通过。

结果：

```text
positive / negative / unknown = 4080 / 1208 / 856
resolved positive prevalence  = 0.771558
mixed structures              = 64 / 96
distinct signatures           = 59

train checkerboard            = 584 / 584
validation checkerboard       = 162 / 162
train / validation structures = 47 / 17
validation output bits        = 59
strongest unary AUC           = 0.5
```

所有width和shortcut门通过。唯一改变的第五轮确实把四轮`98.74%`平衡偏置推进到正负可训练过渡区。

```text
status   = pass
decision = innovation2_skinny64_r5_unit_balance_profile_transition_ready
training = no
remote   = no
```

`curves.svg`最终1600x812像素通过`visual-qa-redraw`，标题、三态数量、配平柱值、0.65门线、
裁决和证据范围无重叠、裁切、缺字或歧义，已记录`visual_qa_passed.marker`。最近结果索引刷新通过。

## 9. 证据支持的推荐下一步

E82开放E83两轮本地readiness，但SKINNY线性层不是PRESENT/GIFT的一对一permutation。下一候选
必须用真实`ShiftRows + MixColumns`的稀疏多前驱图，并与独立node、同入度错误图及公平
train-only拓扑ridge比较。标签通过不等于任意图网络都可训练，也不直接开放30轮或远程。

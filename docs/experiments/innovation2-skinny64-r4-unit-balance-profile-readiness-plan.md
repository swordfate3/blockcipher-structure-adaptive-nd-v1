# 创新2 E81：SKINNY-64四轮严格单位输出平衡谱标签readiness计划

日期：2026-07-19

状态：已完成 / hold / 无神经训练

## 1. 研究问题

E80已经确认同一r3-only Prefix-Guided Nodewise Profile Operator在PRESENT-80和GIFT-64
两个真实SPN上分别获得双seed拓扑归因。第三密码不能直接训练，因为现有SKINNY E20-E24只提供
少量论文kernel或位置几何，E42的ready label family仍为0。

E81只回答：

```text
SKINNY-64/64四轮、8-bit输入cube、64个单位输出bit，
能否形成全称正类 + 具体反例负类的宽、structure-disjoint、抗一元边际捷径标签库？
```

## 2. 同预算锚点与唯一变量

same-budget anchor为GIFT E74：

```text
structures             = 96
active dimension       = 8 bits
witness keys           = 16
inactive offsets       = 8 / structure
checkerboard attempts  = 64
split                   = structure_index % 4
rounds                  = 4
```

E81复用E74的`structure_seed=20260718`、`key_seed=7401`和`offset_seed=17401`，因此96个
活动结构定义逐项相同。唯一方法变量是：

```text
GIFT-64 128-bit key / S-box / P-layer
    -> SKINNY-64/64 64-bit tweakey / S-box / ShiftRows / MixColumns
```

不改变标签语义、split、matching、门槛或输出bit数量。

## 3. 严格标签语义

对每个输入结构和输出bit：

```text
positive = 完整8变量cube单项式不在四轮active-variable ANF support over-approx中
negative = 存在一个真实scheduled 64-bit key和inactive offset，使该输出bit的cube XOR为1
unknown  = 两者均未建立
```

正类是sound充分条件，不把有限key全平衡当全称证明；负类保存key和offset并由标量SKINNY实现
抽样复验；unknown不得改成negative。

## 4. 实现与密码协议门

必须通过：

1. SKINNY-64/64公开Appendix B向量；
2. 4-bit S-box ANF重构16个输入；
3. 向量化四轮加密与`Skinny64.encrypt`逐项一致；
4. ShiftRows和MixColumns support传播坐标与公开实现一致；
5. 所有positive有support缺失证书；
6. 所有negative保存真实64-bit key与inactive offset；
7. 抽样24个negative witness用标量实现复验；
8. train/validation结构严格不交叉。

任一失败均为协议失败，不解释标签数量。

## 5. 冻结宽度与反捷径门

与GIFT E74完全相同：

```text
raw positive >= 256
raw negative >= 256
resolved positive prevalence in [0.10, 0.90]
mixed structures >= 32
distinct ternary signatures >= 4
matched train >= 150/class
matched validation >= 50/class
matched total structures >= 32
validation structures >= 8
validation output bits >= 16
strongest unary validation AUC <= 0.65
duplicate edges = 0
per-structure class delta = 0
per-output-bit class delta = 0
```

checkerboard允许读取标签做配平选择，因此其AUC只用于神经benchmark readiness，不是无偏总体性能估计。

## 6. 裁决

```text
if protocol invalid:
    status   = fail
    decision = innovation2_skinny64_unit_balance_profile_protocol_invalid
elif width or shortcut gate fails:
    status   = hold
    decision = innovation2_skinny64_unit_balance_profile_not_ready
else:
    status   = pass
    decision = innovation2_skinny64_unit_balance_profile_ready
```

`pass`只开放相同4,795参数的independent/true-P/fair-corrupted-P两轮本地readiness；不直接开放
30轮、远程或新架构。若96结构只因checkerboard容量不足而hold，先计算packing上界，再决定是否
冻结192结构扩展；不得事后降低门槛。

## 7. 产物与运行位置

```text
run_id = i2_skinny64_r4_unit_balance_profile_readiness_20260719
output = outputs/local_audits/i2_skinny64_r4_unit_balance_profile_readiness_20260719

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

这是本地标签readiness，无训练、无远程GPU。

## 8. 推荐分支

```text
pass:
  下一步 = SKINNY r3-only 4,795参数三行、两轮、本地readiness

hold且仅matching容量不足、192结构packing上界可过:
  下一步 = 只扩结构96 -> 192，其他协议不变

hold且标签语义/稳定性/边际失败:
  下一步 = 停止当前SKINNY unit-profile，审计新的sound标签表示

fail:
  下一步 = 修复SKINNY坐标、向量化、support或witness协议，禁止训练
```

无论结果如何，不回到PRESENT/GIFT同benchmark继续加层、加seed或机械扩网络。

## 9. 完成结果

运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/audit-innovation2-skinny64-unit-balance-profile-readiness \
  --run-id i2_skinny64_r4_unit_balance_profile_readiness_20260719 \
  --output-root outputs/local_audits/i2_skinny64_r4_unit_balance_profile_readiness_20260719
```

密码协议全部通过：公开32轮向量、S-box ANF、20项四轮向量化/标量对拍、一轮精确ANF
47项对子集夹具、24个negative witness标量复验和structure-disjoint split均通过。

原始标签：

```text
positive = 5563
negative =   71
unknown  =  510
resolved positive prevalence = 0.987398
mixed structures             = 10 / 96
distinct ternary signatures  = 20
```

checkerboard：

```text
train      = 46 / 46, structures = 7, output bits = 21
validation = 18 / 18, structures = 3, output bits = 18
unary AUC  = 0.5 for global/output-bit/active-bit
row/column class delta = 0 / 0
```

标签确实严格且无一元捷径，但四轮SKINNY下8-bit cube几乎都能由support缺失证明平衡，只有
71个具体非平衡反例。失败的不只是matching容量：raw negative、resolved prevalence、mixed
structures、matched class数和结构覆盖同时未过门。因此不允许按GIFT E75方式机械扩到192结构。

最终裁决：

```text
status   = hold
decision = innovation2_skinny64_unit_balance_profile_not_ready
training = no
remote   = no
next     = stop r4 expansion and audit an r5 label-distribution transition
```

## 10. 可视化与验证

`curves.svg`展示三态atlas、配平checkerboard和一元边际。首次像素检查发现裁决说明仍写成
“先审计packing”，与真实gate分支不一致；修正为“不扩到192结构，下一步审计五轮标签分布”，并把
混合英文的密钥语义改成中文。最终1600x812像素通过`visual-qa-redraw`，无重叠、裁切、缺字、
图例冲突、数值遮挡或裁决歧义，已记录`visual_qa_passed.marker`。

验证：

```text
focused pytest              = 6 passed
protocol_checks             = all true
shortcut_checks             = all true
negative witness validation = 24 / 24
result rows                 = 3
result index                = pass, E81 is 001
```

## 11. 证据支持的推荐下一步

E81说明四轮位置处在“平衡性质过多、非平衡反例过少”的标签区间。仅扩结构可能增加总数，却不解决
`98.74%` resolved positive prevalence和只有10个mixed结构的根本失衡。下一实验E82只把目标轮数
从4改为5，复用同96结构、密钥/offset预算、严格证书、split、matching和门槛，审计标签分布是否
进入可训练过渡区。

E82仍不训练。若五轮形成宽、配平且抗捷径标签，才实现SKINNY的`r4-only`前缀算子readiness；
若五轮又变成positive不足或unknown占主导，则停止当前8-bit unit-profile跨轮路线，转向新的
sound标签表示，而不是扫描更多轮数或增加神经容量。

# 创新2 E74：GIFT-64四轮严格unit平衡谱标签readiness计划

日期：2026-07-18

状态：已完成 / hold / 标签有效但checkerboard容量未过门

## 1. 研究问题

E73在PRESENT-80四轮、8-bit活动cube上确认了r3-only平衡谱算子，但不能直接宣称跨密码有效。
E74先回答GIFT-64是否能形成同语义、同输出结构的严格标签benchmark：

```text
input  = GIFT拓扑 + 4 rounds + 8-bit coordinate cube
output = 64个unit bit是否对所有128-bit key和所有inactive offset保持XOR平衡
```

E74不训练网络。只有标签宽度、反捷径和证书门通过，才允许下一实验比较GIFT专属r3-only
true-P、independent和fair-corrupted-P。

## 2. 三态严格标签

```text
positive = sound active-variable ANF support over-approximation中不存在完整8变量单项式
negative = 具体GIFT-64 key和inactive offset使256个cube密文的指定输出bit XOR为1
unknown  = 当前证书与witness bank都无法裁决，不参与训练
```

GIFT每轮顺序必须按源码与规范执行：`SubCells -> PermBits -> AddRoundKey/constant`。轮密钥与
常数只引入活动变量的常数项，但负类必须使用真实128-bit key schedule和标量`Gift64.encrypt`
复验。不得把有限key零失败当positive。

## 3. 冻结atlas

```text
cipher/rounds             = GIFT-64 / 4
active dimension          = 8 coordinate bits
structures                = 96
  coordinate nibble pairs = 24
  deterministic random    = 72
outputs                   = 64 unit bits
witness keys              = 16 deterministic 128-bit keys
offsets per structure     = 8
split                     = structure index mod 4
checkerboard attempts     = 64
execution                 = local CPU / no training
```

实现门必须包括：GIFT零key/零plaintext官方28轮向量`F62BC3EF34F775AC`、S-box ANF重构16个
输入、四轮vectorized/scalar逐值一致、P-layer为64位置换、抽样negative witness标量复验100%。

## 4. 反捷径benchmark与门

train与validation分别构造2x2 checkerboard，使每个selected structure和output bit内部正负
精确平衡；split先冻结，结构不重叠。原始atlas禁止直接训练。

标签宽度门：

```text
raw positive/negative                         each >= 256
resolved positive prevalence                  in [0.10, 0.90]
mixed structures                              >= 32
distinct ternary signatures                   >= 4
matched train positive/negative                each >= 150
matched validation positive/negative           each >= 50
matched total/validation structures            >= 32 / 8
matched validation output bits                 >= 16
strongest validation unary marginal AUC        <= 0.65
duplicate edges / structure delta / bit delta  = 0
```

## 5. 产物、裁决与下一步

```text
run_id = i2_gift64_r4_unit_balance_profile_readiness_20260718
output = outputs/local_audits/i2_gift64_r4_unit_balance_profile_readiness_20260718
```

全部通过：

```text
decision = innovation2_gift64_unit_balance_profile_ready
next = GIFT r3-only Prefix-Guided Profile Operator最小三行readiness
```

标签或matching失败只说明当前GIFT四轮benchmark/provider未就绪；不得扩大有限key投票、把unknown
改成negative、直接套PRESENT checkpoint或启动远程训练。E74不是高轮区分器、攻击或SOTA。

## 6. 实际执行结果

执行命令：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/audit-innovation2-gift64-unit-balance-profile-readiness \
  --run-id i2_gift64_r4_unit_balance_profile_readiness_20260718 \
  --output-root outputs/local_audits/i2_gift64_r4_unit_balance_profile_readiness_20260718
```

原始`96 x 64 = 6144`个三态坐标：

```text
positive = 3022
negative = 1381
unknown  = 1741
resolved positive prevalence = 0.686350
mixed structures             = 72 / 96
distinct ternary signatures  = 91
```

密码学与实现协议全部通过：官方28轮零向量一致，GIFT S-box ANF重构`16/16`，四轮
vectorized/scalar fixture一致`20/20`，抽样negative witness标量复验`24/24`，正类与已发现
反例冲突为0，train/validation结构互斥。

checkerboard与一元边际控制也有效：

```text
train      = 92 positive / 92 negative / 48 structures / 40 output bits
validation = 28 positive / 28 negative / 15 structures / 24 output bits
duplicate edges / structure delta / output delta = 0 / 0 / 0
global / output-bit / active-bit AUC              = 0.5 / 0.5 / 0.5
```

未通过的预注册门只有：

```text
matched train each class >= 150       false (92)
matched validation each class >= 50   false (28)
```

因此正式裁决为：

```text
status   = hold
decision = innovation2_gift64_unit_balance_profile_not_ready
training = no
remote   = no
```

这不是GIFT标签语义失败：raw正负宽度、签名、多结构混合、证书、反例和反捷径均已过门；失败点是
冻结96结构下2x2 checkerboard的可用边容量不足。基于同一atlas的标签盲上界为train 318边、
validation 100边，而预注册总边门分别为300和100，几乎要求用尽所有理论可配平边，缺少稳健余量。

## 7. 推荐下一步

下一实验只改变一个变量：把确定性结构库从`96`扩大到`192`，继续使用4轮、8-bit cube、
`16 keys x 8 offsets`、相同split、相同checkerboard算法和完全相同的正负宽度/反捷径门。E74的
96结构结果作为同协议锚点，不增加有限key投票，不降低门槛，也不训练网络。

若192结构使train/validation各达到`150/150`和`50/50`且全部协议/反捷径门保持通过，才开放
GIFT r3-only的`independent / true-P / fair-corrupted-P`两轮本地readiness；若仍失败，关闭当前
GIFT四轮unit-profile神经迁移，转SKINNY标签provider或新的sound标签表示，而不是继续机械扩结构。

产物：

```text
outputs/local_audits/i2_gift64_r4_unit_balance_profile_readiness_20260718/
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
```

`curves.svg`已按`visual-qa-redraw`渲染为1600x812像素检查，未发现字体重叠、裁剪、缺字、
图例冲突、坐标歧义或不可读标题。

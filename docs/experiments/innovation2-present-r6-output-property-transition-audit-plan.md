# 创新2 E7：PRESENT r6 输出性质活动宽度过渡区审计

**日期：** 2026-07-17
**状态：** 计划冻结 / 待本地执行
**主任务：** 结构条件积分输出性质预测

## 1. 研究问题

PRESENT r5 的单活动 nibble 输出性质预测已完成 E0-E6，但主要 top-k 收益被
输出位置先验解释；相同 16 明文结构到 r6 时，跨密钥 parity 已接近随机。

本审计不训练神经网络，只回答 benchmark 是否适合训练：把一个完整活动 nibble
扩为两个完整活动 nibble，是否能在 r6 形成结构间可排序的平衡率差异，并且该
差异在扣除输出位置边际后仍存在？

```text
同预算锚点：1 个活动 nibble，16 个明文
唯一变量：  2 个活动 nibble，256 个明文
固定项：    PRESENT-80 / r6 / 输出 nibble 与非零 4-bit mask / 随机独立密钥
```

## 2. 一条数据与标签

一条结构输入包含：

```text
active_nibbles      16-bit multi-hot
output_nibble       16-bit one-hot
output_mask         15-bit one-hot（0001..1111）
fixed_plaintext     64 bit，所有活动 nibble 清零
```

它不包含密钥、最终密文或任何输出统计。对每把随机密钥 `K`，先枚举活动 nibble
的全部取值，再对指定输出掩码求结构内 XOR：

```text
q(K,S,u) = XOR_{x in S} <u, E_K^6(x)>
```

`q=0` 表示该结构在这一把密钥下观察为平衡。网络的未来目标是根据结构描述预测
`P_K[q=0 | S,u]` 或给候选排序，不是预测某个密文 bit，也不是判断结构/随机类别。

## 3. 冻结本地审计

```text
rounds                    = 6
seed                      = 0
active_nibble_counts      = 1, 2
structures per width      = 64
keys per structure        = 256
output masks              = 1..15
device                    = local CPU / NumPy vectorized PRESENT
training                  = none
```

两种活动宽度分别使用独立、可复现的结构集合，但使用同一组 32 把密钥。输出：

```text
results.jsonl
structure_rates.csv
position_priors.csv
progress.jsonl
gate.json
curves.svg
```

## 4. 指标与控制

每种活动宽度报告：总体 `q=1` 比例、结构平衡率均值/标准差、全平衡结构比例、
全随机附近结构比例、最高四分位平衡率，以及扣除同宽度训练输出位置均值后的
结构残差标准差。观测率标准差还必须扣除有限密钥导致的二项采样方差；不能把
`sqrt(p(1-p)/keys)` 量级的纯观测噪声误判为结构可学习性。

控制包括：

- 1-nibble 同轮数同密钥锚点；
- 输出位置边际先验；
- active-position 组合边际摘要；
- output-mask 边际摘要；
- 标量 PRESENT fixture 与向量化 parity 逐样本对拍；
- 所有活动 nibble 在固定上下文中必须清零。

本阶段不训练 MLP，因此不产生 AUC，也不允许把标签分布本身称为神经结果。

## 5. Advance / stop gate

两活动 nibble benchmark 只有同时满足以下条件才进入输出预测训练：

```text
0.05 <= overall q1 rate <= 0.95
noise-corrected structure balance-rate std >= 0.03
noise-corrected output-position residual std >= 0.02
at least 10% structures have 0.05 < balance rate < 0.95
vectorized/scalar parity cross-check = pass
```

裁决：

- 全门通过：冻结 r6 两活动 nibble 的 geometry-disjoint 训练矩阵；候选 MLP 必须
  同预算比较线性、标签打乱、输出位置/活动位置/mask 先验和分布匹配控制。
- 若几乎全平衡：不训练；只把轮数提高到 r7，保持两活动 nibble 做一次过渡审计。
- 若仍近随机且残差不足：不训练、不扩样；改审计活动 bit 宽度 5--7 的细粒度
  过渡区，而不是远程放大同一无信号标签。
- 无论结果如何，本阶段都不启动远程 GPU、H0 seed2、r9 二分类或跨密码扩展。

## 6. 通过后的训练计划边界

若本地 benchmark gate 通过，下一训练实验只允许 3 个模型：结构 MLP、同输入
线性、标签打乱 MLP；位置/活动/mask 先验作为不训练控制。先用 seed0 本地小规模
readiness，只有候选在 geometry-disjoint test 上超过所有边际和匹配控制，才准备
seed1 或远程规模。正式规模、epochs 和远程参数必须在该结果后另行冻结。

## 7. 执行中统计修正

首次按 `32` 把密钥生成标签后，观测结构标准差约等于
`sqrt(0.25/32)=0.088` 的纯二项采样噪声，原 gate 会把有限密钥噪声误判为
真实结构差异。该结果在进入最近结果索引和训练决策前作废。

修正版在不查看任何神经指标的前提下冻结为每结构 `256` 把密钥，并同时报告
观测标准差、估计采样噪声标准差以及非负的超额结构标准差。位置残差同样扣除
组内均值所对应的采样噪声。只有噪声校正指标通过第 5 节门槛才允许训练。

## 8. 修正版结果与裁决

修正版已按 `64 structures/width × 256 keys/structure` 在本地完成。向量化
PRESENT parity 与标量实现对拍通过，所有输出位置均覆盖，活动 nibble 的固定
上下文清零，所有指标有限。

| 活动宽度 | q=1 比例 | 观测结构标准差 | 估计采样噪声 | 超额结构标准差 | 超额位置残差标准差 |
|---|---:|---:|---:|---:|---:|
| 1 nibble / 16 明文 | `0.487488` | `0.043849` | `0.030936` | `0.031075` | `0.019058` |
| 2 nibbles / 256 明文 | `0.480347` | `0.065891` | `0.030770` | `0.058265` | `0.018802` |

两活动 nibble 的总体结构差异在扣除采样噪声后仍存在，但进一步扣除输出位置
边际后只剩 `0.018802`，没有达到冻结的 `0.02` 门槛；`93.75%` 的结构平衡率
仍落在 `0.4--0.6`，top quartile 平均平衡率仅 `0.598389`。因此裁决为：

```text
status = hold
decision = innovation2_r6_two_nibble_output_prediction_benchmark_not_ready
training = no
remote_scale = no
```

这不是“创新2输出预测失败”，而是说明把活动宽度从 4 bit 一步跳到 8 bit 仍没有
得到独立于输出位置的足够残差信号。当前不得训练 MLP，否则大概率再次复制 E6
的位置先验捷径。

权威产物：

```text
outputs/local_audits/
  i2_present_r6_output_property_transition_width1_width2_seed0_20260717/
```

最终 `curves.svg` 经 `visual-qa-redraw` 以 `1800×870` 像素检查，中文 glyph、
标题、轴、图例、门槛线和裁决说明均无重叠或裁切；通过标记为
`visual_qa_passed.marker`。`scripts/index-results` 刷新通过，本结果为
`outputs/00_RECENT_RESULTS.md` 的 `001`。

## 9. 下一步固定计划

下一步仍是标签 benchmark 审计，不训练网络：

```text
research question          = r6 的 5/6/7 活动 bit 是否形成可学习过渡区
same-budget anchor         = 本次 4-bit 与 8-bit 噪声校正结果
one variable               = active bit width 5, 6, 7
structures per width       = 64
keys per structure         = 256
seed                       = 0
rounds                     = 6
plaintexts per structure   = 32, 64, 128
execution                  = local NumPy vectorized PRESENT; no neural training
required controls          = output-position / active-position / mask marginals,
                             finite-key noise correction, scalar cross-check
advance gate               = excess structure std >= 0.03 and
                             excess position-matched residual std >= 0.02
stop gate                  = no width clears both gates
remote_scale               = no
```

若某个宽度过门，选择残差最大的单一宽度冻结 geometry-disjoint 三模型训练；若
全部不过门，停止 r6 当前结构描述路线，不做 seed1、不扩大密钥数、不远程训练，
回到 r5 方法学章节或重新设计更强的积分结构定义。

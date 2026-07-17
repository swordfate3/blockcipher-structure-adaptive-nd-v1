# 创新2 E8：PRESENT r6 细粒度活动 bit 输出性质过渡审计

**日期：** 2026-07-17
**状态：** 已完成 / hold / 停止 r6 当前单 mask 训练路线
**前置裁决：** E7 两活动 nibble 的位置残差不足

## 1. 研究问题

E7 表明从 4 个活动 bit 直接跳到 8 个活动 bit 后，PRESENT r6 的结构平衡率
虽然有总体差异，但扣除有限密钥噪声与输出位置先验后的残差只有 `0.018802`，
不足以启动神经训练。

E8 只改变活动 bit 宽度，审计 5、6、7 个活动 bit 是否形成可重复的输出性质
过渡区：既不是近似常数，也不能被活动位置、输出位置和输出 mask 的简单边际
解释。

## 2. 数据与标签

每条结构包含：

```text
active_bits        = 64-bit multi-hot mask，恰有 5/6/7 个活动位
output_nibble      = 0..15
output_mask        = 1..15
fixed_plaintext    = 64 bit，所有活动位清零
plaintext set size = 2^active_bit_width
```

对每把 PRESENT-80 密钥，枚举结构中的全部明文并计算：

```text
q(K,S,u) = XOR_{x in S} <u, E_K^6(x)>
```

输入不含密钥、密文或输出统计。未来网络目标仍是预测跨密钥平衡概率或候选
排序，不是结构/随机二分类。

## 3. 冻结规模

```text
rounds                    = 6
active bit widths         = 5, 6, 7
plaintexts per structure  = 32, 64, 128
structures per width      = 64
keys per structure        = 256
key halves                = 128 + 128, mutually disjoint
seed                      = 0
training                  = none
execution                 = local NumPy vectorized PRESENT
```

输出：

```text
results.jsonl
structure_rates.csv
marginal_predictions.csv
progress.jsonl
gate.json
metadata.json
curves.svg
```

## 4. 强边际控制

每种宽度独立构造以下结构字段：

```text
64 active-bit indicators
16 output-nibble indicators
15 output-mask indicators
```

使用固定 `ridge alpha=1.0`、4-fold cross-fitting 的加性边际模型。每个结构的
预测只来自不包含该结构的 fold 训练数据。两组 128-key 标签独立拟合并独立产生
残差，避免同一批密钥噪声同时进入拟合与评估。

主要可重复信号使用两半协方差估计：

```text
cross-half structure std = sqrt(max(0, cov(rate_half0, rate_half1)))
cross-half residual std  = sqrt(max(0, cov(residual_half0, residual_half1)))
```

同时报告两半 residual Pearson correlation。独立密钥采样噪声期望协方差为零，
因此不能仅凭单半观测标准差放行。

## 5. Advance / stop gate

某一活动宽度只有同时满足以下条件才通过：

```text
0.05 <= overall q1 rate <= 0.95
cross-half structure std >= 0.03
cross-half combined-marginal residual std >= 0.02
cross-half combined-marginal residual correlation >= 0.20
at least 10% structures have 0.05 < balance rate < 0.95
vectorized/scalar parity cross-check = pass
```

若多个宽度通过，选择 combined-marginal residual std 最大者；差值小于 `0.002`
时选择明文复杂度更低的宽度。

## 6. 后续分支

- 至少一个宽度通过：冻结所选宽度的 geometry-disjoint 本地训练矩阵，只训练
  结构 MLP、同输入线性和标签打乱 MLP；另保留边际 ridge 与分布匹配控制。
- 全部不过：停止 PRESENT r6 当前结构描述路线，不加 seed、不增加密钥、不远程
  训练；保留 r5 方法学章节，后续必须重新设计积分结构而不是调网络。
- 审计无效：修数据、密钥拆分或向量化对拍，不解释任何标签指标。

本实验不允许启动远程 GPU、H0 seed2、r8/r9 二分类扩展或跨密码扩展。

## 7. 实际结果

E8 已按冻结规模在本地完成。三种宽度共享同一组 256 把 PRESENT-80 密钥；
两半各 128 把且互斥。活动上下文清零、输出位置与全部非零 4-bit mask 覆盖、
标量/向量化 parity 对拍和所有有限指标检查均通过。

| 活动 bit | q=1 比例 | 观测平衡率 std | 两半结构 std | 输出位置残差 std | 组合边际残差 std | 组合残差相关 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | `0.499939` | `0.031299` | `0.000000` | `0.006060` | `0.022468` | `0.197728` |
| 6 | `0.501404` | `0.028188` | `0.000000` | `0.000000` | `0.000000` | `-0.301313` |
| 7 | `0.504272` | `0.036876` | `0.013687` | `0.012657` | `0.035282` | `0.276179` |

三种宽度的结构平衡率都 `100%` 落在 `0.4--0.6`。5/6 bit 的原始两半结构
协方差非正；7 bit 的两半结构标准差只有 `0.013687`，没有达到 `0.03`。

7 bit 的组合边际残差标准差 `0.035282`、相关 `0.276179` 看似过门，但原始
结构本身不跨两半密钥复现。该现象可能来自相同字段设计和 ridge 正则造成的共享
模型偏差，不能反向证明标签含有可学习结构信号。因此 gate 正确要求“原始结构
复现”和“边际残差复现”同时成立，7 bit 仍不得进入训练。

正式裁决：

```text
status = hold
decision = innovation2_r6_active_bit_transition_benchmark_not_ready
passing_active_bit_widths = []
selected_active_bit_width = null
training = no
seed1 = no
remote_scale = no
```

权威产物：

```text
outputs/local_audits/
  i2_present_r6_output_property_active_bits5_6_7_seed0_20260717/
```

最终 `curves.svg` 经 `visual-qa-redraw` 以 `1800×779` 像素检查，中文 glyph、
标题、轴、阈值、数据标签、图例和底部裁决均无重叠或裁切；通过标记为
`visual_qa_passed.marker`。最近结果索引刷新通过，本结果为
`outputs/00_RECENT_RESULTS.md` 的 `001`。

## 8. 推荐下一步

停止的范围是：`PRESENT r6 + 当前结构描述 + 单个 4-bit 输出 mask 概率`。
这不否定创新2的输出性质预测目标。E8 说明单 mask 近似随机后，继续调 MLP 没有
标签基础；下一步应先审计更强的输出对象：64-bit 输出 parity 向量所定义的平衡
mask 子空间。

推荐 E9 只做稳定子空间 readiness，不训练神经网络：

```text
input structures           = 复用 E8 的 5/6/7 active-bit structures
output observable          = 64-bit XOR parity vector per key
keys                       = 128 discovery + 128 validation
baseline                   = GF(2) empirical kernel, Hwang et al. 2026
primary evidence           = dim ker(M_discovery), dim ker(M_validation),
                             dim ker([M_discovery; M_validation])
advance                    = 至少一个宽度存在跨两半稳定的非平凡 joint kernel
stop                       = 所有结构 joint kernel 都为零维
execution                  = local, no neural training, no remote scale
```

只有稳定非平凡子空间存在，才讨论“结构 -> kernel dimension / balanced-mask basis”
的神经预测；GF(2) kernel 必须保留为强非神经基线，不能把已有 kernel 方法包装成
本项目首创。

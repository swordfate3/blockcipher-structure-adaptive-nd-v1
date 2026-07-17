# 创新2 E10：PRESENT 7轮 Hwang kernel bit-order readiness 计划

**状态：** E11b pass / Hwang four-dimensional kernel reproduced / E12 pending

**日期：** 2026-07-17

**密码：** PRESENT-80

**轮数：** 7

**任务：** 文献积分输出平衡性质复现，不做积分结构/随机结构二分类
**论文锚点：** Hwang et al. 2026 Appendix A.1, Table 8

## 1. 研究问题

Hwang et al. 报告：PRESENT 7轮、最后16个输入 bit 全活动时，输出 parity
matrix 的 kernel 包含四个平衡线性组合：

```text
b0
b4 XOR b12
b16 XOR b48
b20 XOR b28 XOR b52 XOR b60
```

本项目 PRESENT 实现以 Python 整数的 LSB 为 bit 0，但论文的“最后16 bit”和
`b0...b63` 可能使用相反的状态书写方向。E10 只回答：哪一种输入方向和输出
编号映射能在当前实现中复现上述四个输出平衡性质。

这一步是创新2输出预测数据集的协议校准。它不是神经网络实验，也不是把积分
结构和随机结构做二分类。

## 2. 固定协议

四个候选协议由两个输入方向和两个输出编号方向的笛卡尔积组成：

```text
输入活动方向 A: project bits 0..15
输入活动方向 B: project bits 48..63

输出编号方向 A: paper bi -> project bit i
输出编号方向 B: paper bi -> project bit (63-i)
```

共同固定项：

```text
cipher                  = PRESENT-80
rounds                  = 7
active bits             = exactly 16
plaintexts per structure= 2^16 = 65536
fixed inactive context  = all zero
seed                    = 0
keys                    = 8 total
discovery/validation    = 4 + 4 disjoint keys
key_chunk_size          = 1
training                = none
remote                  = no
```

8-key 本地运行只是 bit-order readiness，不是所有密钥证明，也不是论文完整
kernel 规模复现。

## 3. 同预算强基线与控制

强基线直接计算每把密钥下的完整输出 XOR word：

```text
P(K,S) = XOR_{x in S} E_K^7(x)
```

对于输出 mask `u`，平衡检查为：

```text
<u, P(K,S)> = 0
```

必须检查：

```text
1. 四个论文 mask 在 discovery 和 validation 密钥半上分别为零；
2. GF(2) empirical kernel 确实包含四个论文 mask；
3. scalar 与 vectorized PRESENT 输出 XOR 对齐；
4. parity word 不是所有密钥都等于零；
5. 至少一个确定性同权重非论文控制 mask 出现非零 parity。
```

控制 mask 从四种论文权重 `1,2,2,4` 出发，使用固定 seed 生成，不得在看到
结果后更换。控制 mask 不得落入四个论文基张成的子空间。

## 4. Readiness 门槛

每个候选协议分别判定：

```text
paper_masks_zero_on_discovery      = true
paper_masks_zero_on_validation     = true
paper_masks_in_joint_kernel        = true
nonzero_output_parity_word_exists  = true
nonpaper_control_failure_exists    = true
```

全局 readiness 还要求：

```text
four_protocol_candidates_present   = true
keys_halves_disjoint               = true
vectorized_matches_scalar          = true
all_metrics_finite                 = true
```

裁决：

```text
恰好一个候选通过:
  status   = pass
  decision = innovation2_present_r7_hwang_bitorder_ready

多个候选通过:
  status   = hold
  decision = innovation2_present_r7_hwang_bitorder_ambiguous

没有候选通过:
  status   = hold
  decision = innovation2_present_r7_hwang_bitorder_not_reproduced

实现/标量校验失败:
  status   = fail
  decision = innovation2_present_r7_hwang_protocol_invalid
```

## 5. 产物

```text
outputs/local_audits/
  i2_present_r7_hwang_kernel_last16_bitorder_readiness_seed0_20260717/
    results.jsonl
    mask_checks.csv
    progress.jsonl
    gate.json
    metadata.json
    curves.svg
```

`curves.svg` 必须通过 `visual-qa-redraw` 的渲染像素检查，检查中文字体、标题、
候选标签、数值标签、图例、裁决文字、重叠和裁切。

## 6. 后续决策

若恰好一个候选通过，下一步不是立即训练神经网络，而是冻结该 bit-order，设计
更大且互斥的 discovery/validation key 集合，复现论文 kernel 维数和四个基的
稳定性。该阶段若达到中等规模，必须先实现磁盘缓存、参数匹配复用和持久进度，
再走远程 GPU 工作流。

只有文献结构在更大新密钥集合上仍稳定后，才构造真正的创新2训练任务：输入
结构描述与候选输出 mask，输出该 mask 的跨密钥平衡概率/稳定性/排序。所需强
基线是直接 GF(2) kernel 和训练集边际先验，所需控制包括标签打乱、mask 匹配和
结构字段匹配。

若多个方向通过，先增加校准密钥或核对论文状态布局，不选择最方便的方向。若
全部不通过，停止训练和扩样，审计论文轮定义、最终 whitening、bit 编号及活动
集合定义；不得事后修改预注册 mask 迎合当前实现。

## 7. 2026-07-17 E10 结果

本地 readiness 已按冻结协议完成：

```text
run_id = i2_present_r7_hwang_kernel_last16_bitorder_readiness_seed0_20260717
keys = 8 total = 4 discovery + 4 validation
plaintexts_per_structure = 65536
training = no
```

四个候选的 joint 结果：

| 输入活动方向 | 输出编号 | 论文 mask 失败次数 | 控制 mask 失败次数 | joint kernel dim | 候选通过 |
|---|---|---:|---:|---:|---|
| `0..15` | direct | `0` | `17` | `56` | yes |
| `0..15` | reflected | `18` | `15` | `56` | no |
| `48..63` | direct | `0` | `17` | `56` | yes |
| `48..63` | reflected | `17` | `9` | `56` | no |

四个 readiness 校验全部通过：密钥半互斥、四个候选齐全、标量/向量化完整
输出 XOR 一致、指标有限。两种 direct 输入方向的 parity word 均非全零，且非论文
同权重控制 mask 均出现非零，所以论文 mask 的零失败不是“整个输出 XOR word
恒零”造成的。

按预注册门槛，多于一个候选通过，因此裁决保持为：

```text
status = hold
decision = innovation2_present_r7_hwang_bitorder_ambiguous
passing_candidates = [low_0_15__direct, high_48_63__direct]
training = no
remote_scale = no
```

论文索引审计进一步确认：Hwang 声明沿用所引用前序工作的索引，而 Wu/Guo 对
同一 PRESENT 积分结构明确写为“明文最右侧 bit 活动”。结合本项目 Python 整数
以 LSB 为 bit 0，论文意图对应 `low_0_15__direct`。`high_48_63__direct` 的通过
应记录为当前 8-key 样本下的额外对称现象，不能反向改写论文结构。

8 行 parity matrix 的秩最多为 8，因此当前 joint kernel 维数至少是 56；本次
观测 `dim=56` 只能证明四个论文 mask 被包含，不能证明经验 kernel 已收敛为论文
报告的四维空间。

权威产物：

```text
outputs/local_audits/
  i2_present_r7_hwang_kernel_last16_bitorder_readiness_seed0_20260717/
```

最终 `curves.svg` 经 `visual-qa-redraw` 渲染为 `1800×853` 像素检查；标题、
中文 glyph、坐标、四个候选标签、数据标签和裁决文字无重叠、裁切或结构歧义。

## 8. 推荐下一步：E11 论文四维 kernel 收敛审判

下一实验固定已由论文语义解析的 `low_0_15__direct`，只把独立密钥数从 8 增至
128，不再同时改变活动结构、轮数、输出 mask 或固定上下文：

```text
question                = empirical joint kernel 是否收敛到论文四维空间
cipher / rounds         = PRESENT-80 / 7
active bits             = project bits 0..15
output mapping          = paper bi -> project bit i
fixed inactive context  = all zero
keys                    = 128 total
discovery / validation  = 64 + 64 disjoint
key generation seed     = seed + 3301
E10 key overlap         = none; verify against seed + 2301 keys
plaintexts per key      = 65536
key_chunk_size          = 1
seed                    = 0
training                = none
execution               = local vectorized protocol audit
same-budget anchor      = direct GF(2) empirical kernel
```

唯一推进门槛：

```text
scalar/vectorized cross-check                  = pass
four paper masks vanish on both key halves     = true
joint kernel dimension                         = 4
joint kernel span equals four paper-mask span  = true
```

若四个 mask 仍稳定但 kernel 维数大于 4，结果只说明采样密钥不足；下一步重新冻结
256-key 审计并评估是否转远程，不得训练神经网络。若任一论文 mask 在新密钥半上
失败，停止路线并审计论文具体 plaintext context、轮边界和作者实现。只有四维 kernel
在互斥新密钥上复现，才开始设计“结构 + candidate mask -> 跨密钥平衡稳定性/排序”
的神经输出预测数据集，并要求直接 kernel、边际先验、mask 匹配和标签打乱控制。

## 9. 2026-07-17 E11 结果

E11 已使用与 E10 完全互斥的128把密钥完成：

| 密钥集合 | 密钥数 | rank | kernel dim | 论文 mask 失败 | 控制 mask 失败 |
|---|---:|---:|---:|---:|---:|
| discovery | 64 | `57` | `7` | `0` | `125` |
| validation | 64 | `56` | `8` | `0` | `136` |
| joint | 128 | `57` | `7` | `0` | `261` |

所有 readiness 检查通过，包括两个密钥半的标量/向量化完整输出 XOR 对齐。
四个论文 mask 在128把新密钥上零失败，控制 mask 大量失败，因此论文输出性质
获得强经验支持。但 joint kernel 仍有7维，比论文报告多3维，未通过“恰好四维且
空间相等”的门槛：

```text
status = hold
decision = innovation2_present_r7_hwang_kernel_underconstrained
training = no
remote_scale = no
```

论文第4.1节明确固定同一个 plaintext multiset `S`，每一行只采样独立密钥；因此
全零固定上下文并未违反其形式协议。论文同时说明所有实验使用 `m=10^3` 发现密钥
和 `m'=10^6` hold-out 密钥，本项目128-key E11 远低于论文验证规模，不能把多出的
三维直接解释为论文错误或新的所有密钥性质。

## 10. E11b 高16位同预算对照

E10 中 `high_48_63__direct` 也包含四个论文 mask。为区分“低位结构尚未采样收敛”
与“论文 last16 实际映射到高位”，E11b 固定与 E11 完全相同的128把密钥、轮数、
固定上下文、输出映射和计算代码，只改变一个变量：

```text
active bits = project bits 48..63
```

门槛仍为四个论文 mask 在两半零失败、joint kernel dimension `4` 且空间恰好等于
论文四维 span。若高位达到四维而低位为七维，优先选择高位作为论文协议映射；若
两者都高于四维，再设计更大密钥收敛审判，不进入神经训练。

## 11. 2026-07-17 E11b 结果与最终映射裁决

E11b 使用与 E11 完全相同的128把密钥，只把活动输入改为项目 bits `48..63`：

| 密钥集合 | 密钥数 | rank | kernel dim | 论文 mask 失败 | 控制 mask 失败 |
|---|---:|---:|---:|---:|---:|
| discovery | 64 | `59` | `5` | `0` | `116` |
| validation | 64 | `58` | `6` | `0` | `133` |
| joint | 128 | `60` | `4` | `0` | `249` |

joint kernel 的 canonical basis 精确为：

```text
0x0000000000000001 = b0
0x0000000000001010 = b4 XOR b12
0x0001000000010000 = b16 XOR b48
0x1010000010100000 = b20 XOR b28 XOR b52 XOR b60
```

因此同预算对照给出明确裁决：

```text
status = pass
decision = innovation2_present_r7_hwang_kernel_reproduced
input_orientation = high_48_63
joint_kernel_dimension = 4
joint_kernel_equals_paper_span = true
training = no
remote_scale = no
```

此前仅根据 Wu/Guo“最右侧活动 bit”文字推断项目 `0..15` 是 Hwang last16 的映射，
证据不足。E11/E11b 的同密钥、同预算实验表明：项目 `48..63` 才能精确复现
Hwang Table 8 的四维 kernel；项目 `0..15` 保留七维，是另一个固定结构，不能
当作该论文协议。创新2后续必须冻结 `high_48_63 + direct output mapping`。

E11 与 E11b 两张 `curves.svg` 均经 `visual-qa-redraw` 以 `1800×866` 像素
检查，标题、中文 glyph、坐标、图例、数据标签和裁决文字无重叠或裁切。

## 12. 推荐下一步：E12 16-bit 活动块 kernel 多样性 readiness

论文单一结构已经校准，但一个结构不能训练“结构条件化输出预测”模型。E12 先
回答：改变16-bit 连续活动块位置后，是否得到多个在互斥密钥上稳定且不同的输出
kernel，从而形成真正有标签多样性的预测问题。

冻结矩阵：

```text
cipher / rounds          = PRESENT-80 / 7
active blocks            = bits 0..15, 16..31, 32..47, 48..63
fixed inactive context   = all zero
output feature           = full 64-bit ciphertext XOR word
keys per structure       = 128 = 64 discovery + 64 validation
key generation seed      = seed + 3301, same across structures
plaintexts per structure = 65536 per key
key_chunk_size           = 1
seed                     = 0
training                 = none
same-budget anchor       = direct GF(2) empirical kernel
```

这里只改变活动块位置。高16位 Hwang 结构是正校准 anchor，低16位 E11 是已知
七维对照；中间两个活动块是新候选。

推进门槛：

```text
Hwang high16 joint kernel remains exact four-dimensional paper span
all structures have nonzero parity words and scalar/vectorized agreement
each reported joint basis validates on both key halves
at least two distinct joint-kernel signatures
at least two structures have nontrivial joint kernels
```

通过后才设计结构/mask 样本表，并在训练前检查：标签不能由活动块位置或 mask
Hamming weight 的单字段边际直接解释；必须保留直接 kernel、训练集字段边际、
mask 匹配和标签打乱控制。若四个块只有同一个 kernel，下一步改为固定高16位并
变化非活动上下文；不得用重复标签直接训练网络。

## 13. 2026-07-17 E12 结果

E12 已按冻结矩阵完成：

| 活动块 | discovery dim | validation dim | joint dim | joint kernel |
|---|---:|---:|---:|---|
| `0..15` | `7` | `8` | `7` | Hwang 四维 span + 3个额外方向 |
| `16..31` | `4` | `4` | `4` | 精确 Hwang 四维 span |
| `32..47` | `5` | `4` | `4` | 精确 Hwang 四维 span |
| `48..63` | `5` | `6` | `4` | 精确 Hwang 四维 span |

所有四个结构的 joint basis 均在 discovery/validation 两个密钥半上验证通过，
标量/向量化完整输出 XOR 一致，高16位论文 anchor 仍精确四维。四个结构共有两个
不同 joint-kernel 签名，且四个结构均有非平凡 kernel，因此 E12 裁决为：

```text
status = pass
decision = innovation2_present_r7_active_block_kernel_diversity_ready
distinct_joint_kernel_signatures = 2
nontrivial_joint_kernel_structures = 4
training = no
remote_scale = no
```

该结果证明活动结构能够改变稳定输出性质，但差异全部集中在 `0..15` 活动块：
另外三个活动块共享同一个 Hwang 四维 kernel。因此活动块位置本身可能成为完全
足够的捷径，E12 通过不等于神经模型 readiness。

权威产物：

```text
outputs/local_audits/
  i2_present_r7_active_block_kernel_diversity_128keys_seed0_20260717/
```

最终 `curves.svg` 经 `visual-qa-redraw` 渲染为 `1800×850` 像素检查；标题、
中文 glyph、坐标、图例、四个活动块标签、数据标签和裁决文字无重叠或裁切。

## 14. 推荐下一步：E13 结构-mask 标签与边际捷径审计

E13 不重新加密，也不训练神经网络。它从 E12 的四个 joint kernel 构造完整标签
表，判断任务是否只是记忆活动块或 mask 身份。

冻结候选 mask 集合：

```text
positive candidates = 四个结构 joint canonical basis 的并集
extra candidates    = 固定 seed 生成的同 Hamming-weight 非 kernel 控制
structures          = 四个 E12 活动块
label               = candidate mask 是否属于该结构的 joint kernel span
```

对每个 `(active_block, candidate_mask)` 组合生成一行，至少比较：

```text
global positive-rate baseline
active-block-only marginal baseline
mask-identity-only marginal baseline
mask-weight-only marginal baseline
two-field additive block+mask baseline
label-shuffle control
```

readiness 门槛：标签同时包含正负类；至少两个 mask 在不同结构间翻转；活动块单字段
和 mask-weight 单字段不能完美预测；必须明确报告 mask-identity 与 block+mask 加性
基线能解释多少标签。若 mask identity 或简单加性边际已经近乎完美，下一步不是
训练 MLP，而是扩大结构族（优先变化固定上下文或非连续活动几何）以制造真正的
结构-mask 交互。只有边际控制后仍有可重复残差，才设计神经输出预测模型。

## 15. 2026-07-17 E13 结果

E13 从 E12 的稳定 kernel 构造了 `4 structures × 16 masks = 64` 条确定性标签：

```text
positive candidate masks = 8
matched negative controls = 8
positive label rate       = 0.3125
cross-structure flipping masks = 4
```

无需神经网络的基线：

| 基线 | Accuracy | AUC | Brier |
|---|---:|---:|---:|
| 全局正例率 | `0.6875` | `0.5000` | `0.214844` |
| 活动块边际 | `0.6875` | `0.6091` | `0.203125` |
| mask 身份边际 | `0.9375` | `0.9727` | `0.046875` |
| mask 权重边际 | `0.6875` | `0.5955` | `0.206055` |
| 活动块 + mask 权重加性 | `0.59375` | `0.4614` | `0.241367` |
| 活动块 + mask 身份加性 | `0.9375` | `0.9091` | `0.073804` |
| 标签打乱加性控制 | `0.59375` | `0.4432` | `0.275658` |

所有预注册门槛通过，最高非 oracle 准确率 `0.9375 < 0.98`，因此：

```text
status = pass
decision = innovation2_output_label_interaction_ready
training = no
remote_scale = no
```

这不是“神经网络已准备好”。mask 身份边际 AUC 已达 `0.9727`，说明大部分标签
仍由 mask 自身解释；当前只有4个结构和4个翻转 mask。E13 只支持扩大结构族，
不支持在64行标签上训练 MLP。

权威产物：

```text
outputs/local_audits/
  i2_present_r7_structure_mask_label_readiness_seed0_20260717/
```

最终 `curves.svg` 经 `visual-qa-redraw` 渲染为 `1800×867` 像素检查；标题、
中文 glyph、坐标、基线名称、数值标签、0.98停止线和裁决文字无重叠或裁切。

## 16. 推荐下一步：E14 循环滑动16-bit活动几何扩展

E14 保持 PRESENT 7轮、16个活动 bit、128把密钥和全零固定上下文，只把四个
非重叠活动块扩展为16个 nibble-aligned 循环窗口：

```text
start nibble = 0..15
active nibbles = start, start+1, start+2, start+3 (mod 16)
active bits = corresponding 16 state bits
keys per structure = 128 = 64 discovery + 64 validation
key generation seed = seed + 3301
plaintexts per structure per key = 65536
key_chunk_size = 4
training = none
same-budget anchor = direct GF(2) empirical kernel
```

只改变活动几何。`start=12` 必须继续精确复现 Hwang 高16位四维 kernel；原 E12
的 `start=0,4,8,12` 是嵌入式回归 anchor。

推进门槛：

```text
all joint bases validate on discovery and validation halves
all structures have nonzero parity words
Hwang start=12 anchor remains exact
at least 4 distinct joint-kernel signatures
at least 8 structures have nontrivial joint kernels
```

通过后重跑 E13 标签/边际审计，要求 mask 身份和结构+mask加性基线在新增结构上
仍低于停止线，并用 geometry-disjoint split 冻结训练/验证结构。若滑动窗口仍只有
1--2个签名或 mask 身份继续接近完美，转向变化固定上下文或非连续活动几何，仍不
训练神经网络。

## 17. 2026-07-17 E14 结果

16个循环窗口全部完成，readiness 校验全部通过。joint kernel 只在四个4-nibble
边界对齐的起点非平凡：

| 起点 | 活动 nibbles | discovery dim | validation dim | joint dim | joint签名 |
|---:|---|---:|---:|---:|---|
| `0` | `0,1,2,3` | `7` | `8` | `7` | 低位额外三维 + Hwang span |
| `4` | `4,5,6,7` | `4` | `4` | `4` | Hwang span |
| `8` | `8,9,10,11` | `5` | `4` | `4` | Hwang span |
| `12` | `12,13,14,15` | `5` | `6` | `4` | Hwang span |

其余12个起点的 joint kernel 维数均为 `0`。部分非对齐窗口在单独的64-key half
中出现1--3维经验 kernel，但在联合128把密钥后全部消失，说明互斥密钥半成功
过滤了有限采样假方向。

最终只有三种 joint 签名：空 kernel、Hwang 四维、低位七维；非平凡结构仅4个。
因此裁决为：

```text
status = hold
decision = innovation2_cyclic_geometry_kernel_diversity_insufficient
distinct_joint_kernel_signatures = 3
nontrivial_joint_kernel_structures = 4
training = no
remote_scale = no
```

这排除了“把连续16 bit 每次平移一个 nibble 就能自然获得丰富标签”的路线。稳定
输出性质依赖4-nibble边界对齐结构，下一步应使用 PRESENT P-layer 拓扑变换而不是
继续机械滑动。

权威产物：

```text
outputs/local_audits/
  i2_present_r7_cyclic_geometry_kernel_diversity_128keys_seed0_20260717/
```

最终 `curves.svg` 经 `visual-qa-redraw` 渲染为 `1800×843` 像素检查；标题、
中文 glyph、坐标、16个起点、签名标签、Hwang anchor 线和裁决文字无重叠或裁切。

## 18. 推荐下一步：E15 PRESENT 拓扑感知活动几何

E15 固定同一轮数、密钥、上下文和16-bit活动宽度，构造12个由 PRESENT 结构直接
定义的几何，而不是任意滑动：

```text
family A: 四个边界对齐4-nibble块，start = 0,4,8,12
family B: family A 每个结构经过 PRESENT P-layer 一次映射
family C: family A 每个结构经过 PRESENT P-layer 两次映射
```

实施前静态去重确认：原计划的 family D 四个 nibble-column 与 family B 完全
相同，不能重复计为额外结构。P-layer 三次映射回到原集合，因此四个基础块各有
`P^0/P^1/P^2` 三个唯一几何，共12个。

P-layer 映射按当前实现的精确 bit permutation：

```text
p(i) = (16*i mod 63), i < 63
p(63) = 63
```

冻结预算：

```text
structures = 12 unique
keys per structure = 128 = 64 + 64
plaintexts per structure per key = 65536
key generation seed = seed + 3301
key_chunk_size = 4
training = none
```

门槛继续要求 Hwang 原始结构精确四维、所有 joint basis 验证两半、至少4个不同
joint签名和至少6个非平凡结构。通过后才用 E13 同一 mask/边际基线重建标签表；
否则转固定高16位并变化 inactive context，或停止 PRESENT r7 多结构预测路线。

## 19. 2026-07-17 E15 结果

12个拓扑几何全部完成，readiness 校验全部通过。结果严格按 `P` 的幂次分层：

| 基础块 | `P^0` joint dim | `P^1` joint dim | `P^2` joint dim |
|---:|---:|---:|---:|
| `0` | `7` | `0` | `0` |
| `4` | `4` | `0` | `0` |
| `8` | `4` | `0` | `0` |
| `12` | `4` | `0` | `0` |

`block12_p0` 继续精确复现 Hwang 四维 span；`block04_p0` 和 `block08_p0`
也等于该 span，`block00_p0` 保留 E11 已见的额外三维。8个经过 `P^1/P^2`
得到的非连续活动几何，其联合128把密钥 kernel 全部降为零。单个64-key half
偶尔出现1--2维经验方向，但另一半不支持，不能算稳定输出性质。

最终仍只有空 kernel、Hwang 四维和低位七维三种签名，非平凡结构仍是原来的
四个边界对齐块：

```text
status = hold
decision = innovation2_topology_geometry_kernel_diversity_insufficient
distinct_joint_kernel_signatures = 3
nontrivial_joint_kernel_structures = 4
training = no
remote_scale = no
```

因此 E15 排除了“将已知活动块沿 PRESENT P-layer 轨道移动即可产生丰富输出标签”
的假设。当前证据仍不足以构造神经训练集，不能把12个几何直接送入模型。

权威产物：

```text
outputs/local_audits/
  i2_present_r7_topology_geometry_kernel_diversity_128keys_seed0_20260717/
```

最终 `curves.svg` 经 `visual-qa-redraw` 渲染为 `1800×849` 像素检查；标题、
中文 glyph、双面板、12个拓扑标签、数值标注、签名类别和裁决文字均无重叠或
裁切。

## 20. 推荐下一步：E16 高16位活动集合的固定上下文审计

E16 不再改变活动集合，固定已精确复现论文 kernel 的 project bits `48..63`，
只改变其余 `0..47` 位的固定明文上下文。研究问题是：同一个活动子空间平移到
不同 affine coset 后，稳定输出 kernel 是否会改变，从而产生可学习的
`context + output mask -> balanced` 标签交互。

冻结协议：

```text
cipher / rounds = PRESENT-80 / 7
active bits = 48..63
contexts = 16
context 0 = all-zero Hwang anchor
contexts 1..15 = seed+4401 生成的确定性非零48-bit常量
keys per context = 128 = 64 discovery + 64 validation
key generation seed = seed + 3301
plaintexts per context per key = 65536
key_chunk_size = 4
training = none
same-budget anchor = direct GF(2) empirical kernel
```

只改变 inactive context；轮数、活动宽度、密钥、密钥划分和 kernel 算法保持
不变。推进门槛冻结为：

```text
zero-context kernel continues to contain the Hwang four-dimensional span
all joint bases validate on both key halves
at least 4 distinct joint-kernel signatures
at least 8 contexts have nontrivial joint kernels
```

通过后进入 E17，在16个 context 上重建候选 mask 标签，并要求 context 边际、
mask identity、mask weight 和 context+mask 加性基线不能解释标签，之后才讨论
神经网络。若 E16 仍只有一个 Hwang kernel 或少于4个签名，则停止 PRESENT r7
多结构输出预测路线；不增加 context 数、不增加密钥数，也不启动远程训练。

## 21. 2026-07-17 E16 结果

16个固定上下文全部完成，全部 readiness 校验通过。全零 context 精确复现 Hwang
四维 kernel，另一个非零 context 的标量/向量 XOR 也完全一致。联合128把密钥下：

| joint维数 | context 数 | kernel 类型 |
|---:|---:|---|
| `4` | `9` | 精确 Hwang 四维 span |
| `5` | `5` | Hwang span 加一个 context-dependent 方向 |
| `6` | `2` | Hwang span 加两个 context-dependent 方向 |

五维结构中有三种不同的额外方向模式，六维结构共享一种模式；连同原始 Hwang
四维，共形成5种 joint kernel 签名。所有16个 context 都非平凡，且所有扩展
kernel 都仍包含 Hwang 四维子空间。

```text
status = pass
decision = innovation2_inactive_context_kernel_diversity_ready
distinct_joint_kernel_signatures = 5
nontrivial_joint_kernel_contexts = 16
training = no
remote_scale = no
```

这支持 E16 的核心假设：同一个16-bit活动子空间平移到不同 affine coset 后，
指定输出 mask 的平衡性质可以改变。它是结构条件输出性质信号，但目前仍是128把
采样密钥上的 GF(2) kernel 审计，不是神经结果，也不是全密钥证明。

权威产物：

```text
outputs/local_audits/
  i2_present_r7_inactive_context_kernel_diversity_128keys_seed0_20260717/
```

第一次渲染中左图图例遮挡 context 14 的验证曲线高点；`visual-qa-redraw` 将图例
移到坐标轴上方后重新渲染为 `1800×836`。最终标题、中文 glyph、图例、两图坐标、
16个 context、kernel 曲线、签名标签、anchor 线和裁决文字均无重叠或裁切。

## 22. 推荐下一步：E17 context-mask 标签捷径审计

E17 不重新加密，直接消费 E16 已验证的16个 joint kernel。候选 mask 由所有
context 的 kernel basis 并集构成；当前并集有9个唯一方向。为每个方向生成一个
相同 Hamming weight、且位于所有 source kernel 之外的确定性负控制，共18个
候选 mask：

```text
contexts = 16
positive basis-union masks = 9
matched negative controls = 9
label rows = 16 * 18 = 288
label(context, mask) = 1 iff mask belongs to that context joint kernel
training = none
```

必须报告以下无需神经网络的基线：

```text
global positive rate
context identity marginal
context Hamming-weight marginal
mask identity marginal
mask Hamming-weight marginal
context + mask identity additive LOOCV ridge
48 context bits + 64 mask bits linear LOOCV ridge
label-shuffle bitwise linear control
direct GF(2) oracle
```

推进门槛：标签两类齐全；至少3个 mask 跨 context 翻转；至少4种 context 标签
签名；context 与 mask 单边边际准确率低于 `0.95/0.98`；context+mask身份加性
准确率低于 `0.98`；bitwise 线性准确率低于 `0.95`。所有 context/mask 边际、
身份加性和 bitwise 线性捷径的 AUC 还必须低于 `0.95`，不能用固定 `0.5`
阈值下的 accuracy 掩盖接近完美的排序能力。通过只表示可以进入独立 fresh-key
label 稳定性验证，仍不直接训练。若任一强捷径越线，则先重构候选 mask 或
context family，不启动本地或远程神经网络。

## 23. 2026-07-17 E17 结果与门控纠正

E17 从 E16 的9个 basis-union mask 和9个同权重全局负控制构造了 `16×18=288`
行标签。标签两类、5种 context 签名和5个跨 context 翻转 mask 均存在，正例率为
`0.274306`。但无需神经网络的强基线已经接近完美排序：

| 基线 | accuracy | AUC |
|---|---:|---:|
| 全局正例率 | `0.725694` | `0.500000` |
| context身份边际 | `0.725694` | `0.591030` |
| context汉明重量边际 | `0.725694` | `0.580673` |
| mask身份边际 | `0.947917` | `0.978227` |
| mask汉明重量边际 | `0.725694` | `0.680243` |
| context+mask身份加性 | `0.947917` | `0.967991` |
| 48+64位模式线性 | `0.944444` | `0.975410` |
| 标签打乱位模式线性 | `0.701389` | `0.529829` |

初始 accuracy-only 门槛曾产生 provisional pass，但在结果索引和提交前被作废。
固定 `0.5` 阈值使三个主要捷径的 accuracy 略低于停止线，却掩盖了
`0.968--0.978` 的 AUC。门控修正为同时要求所有非 oracle 捷径 AUC `<0.95`
后，最终裁决为：

```text
status = hold
decision = innovation2_context_label_shortcut_dominated
training = no
remote_scale = no
```

这不是 E16 context-dependent kernel 信号被否定，而是当前候选表构造不合格：
4个 Hwang basis mask 对全部 context 恒正，9个匹配控制对全部 context 恒负，
mask 身份因此几乎直接给出标签。不能用这个288行表训练神经网络。

权威产物：

```text
outputs/local_audits/
  i2_present_r7_context_mask_label_readiness_seed0_20260717/
```

最终图同时展示 accuracy 和 AUC。第一次双指标渲染的 `0.948` 标签与 `0.95`
停止线冲突；`visual-qa-redraw` 将高值标签移入条形内部后，以 `1800×853`
重新检查通过。标题、中文 glyph、16个 context、8组双指标、停止线、图例和裁决
文字均无重叠或裁切。

## 24. 推荐下一步：E17b 等流行率翻转-mask 标签审计

E17b 保持 E16 context、kernel 和所有基线算法不变，只重构候选 mask。对16个
joint kernel 的完整非零 span 做静态统计：

```text
union nonzero masks = 79
common-to-all masks = 15
flipping masks = 64
membership in 1 context = 16 masks
membership in 2 contexts = 16 masks
membership in 4 contexts = 32 masks
```

冻结选择全部32个“恰在4/16个 context 中平衡”的 mask，去掉恒正公共子空间和
恒负控制，不按结果再挑 mask：

```text
contexts = 16
candidate masks = 32
label rows = 16 * 32 = 512
per-mask positive rate = 4 / 16 = 0.25
global positive rate = 0.25
training = none
```

同一流行率使 mask-identity marginal 不能仅凭每个 mask 的总体正例率排序标签。
E17b 继续报告 context/mask身份与重量边际、context+mask身份加性、48+64位模式
线性和标签打乱控制。readiness 要求32个 mask 全部跨 context 翻转、至少4种
context 标签签名、完整512行且无恒定标签。推进门槛冻结为所有 context/mask
边际 AUC `<0.75`，身份加性与 bitwise 线性 AUC `<0.75`；accuracy 同时报告但
不再作为唯一裁决依据。通过后才进入 E18 fresh-key 稳定性验证，仍不训练；失败
则停止当前 context-mask 表并重新选择结构族。

## 25. 2026-07-17 E17b 结果

E17b 从完整 span 自动得到32个 membership=`4/16` 的 mask，完整生成512行；
readiness 全部通过。mask 身份泄漏被成功消除：mask identity 和 mask weight 的
AUC 都降为 `0.5`。但正标签集中在少数 context：

```text
context 5,7,10,13: 16 / 32 positive
context 12,14:      32 / 32 positive
other 10 contexts:   0 / 32 positive
```

基线结果：

| 基线 | accuracy | AUC |
|---|---:|---:|
| 全局正例率 | `0.750000` | `0.500000` |
| context身份边际 | `0.875000` | `0.958333` |
| context汉明重量边际 | `0.812500` | `0.901042` |
| mask身份边际 | `0.750000` | `0.500000` |
| mask汉明重量边际 | `0.750000` | `0.500000` |
| context+mask身份加性 | `0.750000` | `0.916667` |
| 48+64位模式线性 | `0.750000` | `0.916667` |
| 标签打乱位模式线性 | `0.750000` | `0.449402` |

因此最终裁决为：

```text
status = hold
decision = innovation2_equal_prevalence_context_label_shortcut_dominated
training = no
remote_scale = no
```

完整 kernel union 的64个翻转 mask 实际只有4种 context incidence pattern，每种
16个 mask：`{15}`、`{12,14}`、`{5,10,12,14}`、`{7,12,13,14}`。因此继续在
当前16个 context 上挑 mask，无法让其余9个 exact-Hwang context 产生额外正例。

权威产物：

```text
outputs/local_audits/
  i2_present_r7_equal_prevalence_context_mask_readiness_seed0_20260717/
```

第一次渲染错误复用了 E17 的 `0.95/0.98` 线；`visual-qa-redraw` 将 renderer
参数化并只保留 E17b 正确的 `0.75` AUC 线，最终以 `1800×853` 检查通过。标题、
32/16/0 context bars、双指标、标签、门槛线、图例和裁决均无重叠、裁切或误导。

## 26. 推荐下一步：E17c 双轴 group-disjoint 捷径审计

E17/E17b 的 LOOCV 每次只留一行，训练数据仍包含同一个 context 的其他 mask，
因此 context identity marginal 是随机 pair split 的泄漏上界，不能代表未见
context 泛化。E17c 不改变512行标签，唯一变量是评估拆分：

```text
context folds = 4 deterministic folds over 16 contexts
mask folds = 4 deterministic folds over 32 masks
context-disjoint: hold one context fold, predict its all masks
mask-disjoint: hold one mask fold, predict it on all contexts
dual-disjoint: for all 16 context-fold × mask-fold pairs,
               train without either held group and predict their intersection
features = 48 context bits + 64 mask bits
model = ridge linear, alpha = 1.0
training = no neural network
```

每种协议必须让全部512行恰好获得一次 out-of-group prediction，训练/测试标签两类
齐全，并报告 accuracy、Brier 和 AUC。标签打乱使用相同 dual-disjoint folds。
推进门槛冻结为 context-disjoint、mask-disjoint 和 dual-disjoint bitwise AUC 均
`<0.75`，且 label-shuffle dual AUC 在 `[0.35,0.65]`。通过表示当前高 AUC 主要
来自 row-wise 泄漏，可进入 E18 fresh-key 稳定性验证；若任一 group-disjoint
真实标签基线 AUC `>=0.75`，则当前 context/mask 位模式本身已经形成可泛化捷径，
停止该标签族且不训练神经网络。

## 27. 2026-07-17 E17c 结果

首次随机平衡四折产生了若干单类 dual test block，协议门正确返回 `fail`；该输出
未索引、未解释。修复只按“每个 train/test block 同时有0/1”搜索平衡折，不读取
或优化 AUC，随后同一512行、同一 ridge 和同一门槛重新运行。全部512行在每种
group protocol 下恰好获得一次组外预测。

最终指标：

| 拆分 | accuracy | Brier | raw AUC | directional AUC |
|---|---:|---:|---:|---:|
| 逐行 LOOCV | `0.750000` | `0.067941` | `0.916667` | `0.916667` |
| context-disjoint | `0.625000` | `0.266785` | `0.367188` | `0.632812` |
| mask-disjoint | `0.859375` | `0.064018` | `0.950623` | `0.950623` |
| context+mask dual-disjoint | `0.625000` | `0.266583` | `0.366740` | `0.633260` |
| 标签打乱 dual-disjoint | `0.750000` | `0.195840` | `0.473490` | `0.526510` |

context-disjoint 和 dual-disjoint 均低于 `0.75`，说明 E17b 的 context identity
高分很大程度来自逐行拆分泄漏；但 mask-disjoint directional AUC 仍为 `0.950623`，
位模式捷径可泛化到未见 mask。不能只选择通过的拆分来缩窄创新任务，因此联合
裁决仍为：

```text
status = hold
decision = innovation2_group_disjoint_shortcut_generalizes
training = no
remote_scale = no
```

权威产物：

```text
outputs/local_audits/
  i2_present_r7_context_mask_group_disjoint_readiness_seed0_20260717/
```

最终 `curves.svg` 经 `visual-qa-redraw` 渲染为 `1800×799`；原始/方向无关 AUC、
`0.75` 线、accuracy/Brier、五种拆分、图例和裁决均无重叠、裁切或方向歧义。

## 28. 推荐下一步：E18 64-context fresh-key kernel 扩展

当前16个 context 只有4种翻转 incidence pattern，继续重排同一标签矩阵已经没有
信息增益。E18 同时完成 fresh-key 验证和结构族扩展：保留 E16 的16个 context
作为精确回归 anchor，再加入48个确定性新 context；活动集合、轮数和 kernel
算法不变。

```text
cipher / rounds = PRESENT-80 / 7
active bits = 48..63
contexts = 64 = 16 E16 anchors + 48 new low48 constants
new-context generation seed = seed + 7401
fresh keys = 128 = 64 discovery + 64 validation
fresh-key seed = seed + 8801
fresh keys disjoint from E16 seed+3301 keys
plaintexts per context per key = 65536
key_chunk_size = 4
training = none
```

只改变 context 数量和密钥证据集；不改变候选 mask、标签定义或轮数。门槛：

```text
zero-context kernel continues to contain the Hwang four-dimensional span
all joint bases validate both fresh-key halves
all 16 E16 context signatures reproduce exactly on fresh keys
at least 8 distinct joint-kernel signatures across 64 contexts
at least 24 contexts contain directions beyond the Hwang span
```

通过后才基于64-context fresh-key joint kernels 重建 span 标签，并重新执行
context/mask 双轴 group-disjoint 审计；失败则停止 PRESENT r7 inactive-context
输出预测分支。E18 是本地向量化协议审计，不是神经训练，不启动远程 GPU；不得
把64个 context 或128把采样密钥描述为 paper-scale 或全密钥证明。

## 29. 2026-07-17 E18 结果

64个 context、128把全新密钥全部完成。fresh keys 与 E16 的128把密钥完全不相交，
两个64-key half、两个标量 XOR anchor、所有 joint basis 和 Hwang 四维 span 校验
全部通过。64个 context 的 fresh-key joint 维数分布为：

```text
dim 4: 42 contexts
dim 5: 19 contexts
dim 6:  2 contexts
dim 7:  1 context
```

所有64个 context 都继续包含 Hwang 四维子空间，说明论文 basis 本身稳定；但 E16
的16个完整经验 kernel 签名只有 `7/16` 在全新密钥上精确复现。全64个 context
形成9种 fresh-key 签名，22个包含 Hwang 之外的方向，低于预注册的24个门槛。

初始 readiness 错误要求 zero context 的经验 kernel 必须恰好四维；fresh keys
下它为五维，但仍完整包含 Hwang basis。这是应由签名稳定性门裁决的有限密钥额外
方向，不是协议故障。该过强校准在不重复加密的情况下改为 Hwang span containment，
重新裁决后最终结果为：

```text
status = hold
decision = innovation2_context_kernel_fresh_key_unstable
reproduced_e16_context_signatures = 7 / 16
distinct_joint_kernel_signatures = 9
contexts_with_directions_beyond_hwang = 22
training = no
remote_scale = no
```

因此停止 PRESENT r7 inactive-context **严格 kernel membership 二分类标签**路线。
E16 的 context-dependent 额外方向在单一128-key池上真实存在，但不足以作为跨新
密钥池稳定的确定性标签；E17/E17b/E17c 的标签表不得进入神经训练。

权威产物：

```text
outputs/local_audits/
  i2_present_r7_fresh_expanded_context_kernel_128keys_seed0_20260717/
```

第一次 E18 渲染的图例遮挡签名类别9和曲线数据；`visual-qa-redraw` 将两组图例移到
坐标轴上方，以 `1800×820` 重新检查通过。标题、64-context轴、E16/新增边界、
三条维数曲线、9类签名、图例和裁决无重叠或裁切。

## 30. 推荐下一步：E19 跨密钥平衡概率标签审计

E18 表明二元“是否属于所有采样密钥的 joint kernel”对额外方向过于脆弱，但每个
`(context, output mask)` 在不同密钥上的平衡频率仍可能是可重复的连续输出性质。
E19 只改变标签目标：从严格 membership 改为跨密钥平衡概率，不改变 PRESENT 7轮、
64个 context、明文集合或密钥协议。

```text
contexts = E18 frozen 64
keys = E18 frozen 128 = 64 discovery + 64 validation
candidate masks = 16 output nibbles * 15 nonzero 4-bit masks = 240
cells = 64 * 240 = 15360
per-key label = parity(mask & integral_output_xor_word) == 0
cell target = balanced-key rate in each 64-key half
training = none
```

E19 重新计算一次 E18 的确定性 XOR words 并持久化为小型 `xor_words.npy`，以后所有
概率标签和控制都从该缓存派生，不再重复约5亿次块加密。必须报告：

```text
discovery/validation rate Pearson correlation
discovery/validation interaction-residual correlation
mean absolute half-rate difference
context marginal, mask identity, mask weight and additive residual variance
finite-key binomial noise estimate and nonnegative excess variance
label-shuffle and context-shuffle controls
```

推进门槛冻结为：两半 rate correlation `>=0.25`；两半去除 context/mask 加性边际
后的 residual correlation `>=0.20`；mean absolute half-rate difference `<=0.15`；
validation residual standard deviation `>=0.05`；有限密钥噪声修正后的 interaction
excess variance严格大于0；两个 shuffle control residual correlation均 `<0.10`。
通过后才设计连续值/排序神经预测，仍不直接远程训练；失败则停止 PRESENT r7
context 输出概率分支，转文献支持的其他密码/轮数结构族。

## 31. 2026-07-17 E19 结果

E19 完整重放 E18 的64个 context、128把密钥和全部 joint-kernel 签名，并将
`64 x 128` 个积分输出 XOR word 持久化为 `xor_words.npy`。从该缓存构造
`16` 个输出 nibble乘以每个 nibble 的 `15` 个非零 mask，共 `240` 个候选
mask、`64 x 240 = 15360` 个 `(context, mask)` cell。没有训练神经网络。

协议检查全部通过：E18 的64个 joint 签名全部复现，发现/验证各64把密钥互斥，
mask 网格完整，XOR 缓存形状和类型正确，zero context 与首个新增 context 的
标量实现复核一致。最终指标为：

| 指标 | 结果 | 预注册门槛 |
|---|---:|---:|
| 原始 balance-rate 两半相关 | `0.7063695522` | `>= 0.25` |
| interaction 残差两半相关 | `0.0012802112` | `>= 0.20` |
| 两半 rate 平均绝对差 | `0.0678202311` | `<= 0.15` |
| 验证 interaction 残差标准差 | `0.0599370044` | `>= 0.05` |
| 验证 interaction 残差方差 | `0.0035924445` | 诊断量 |
| 二项有限密钥噪声方差估计 | `0.0037270501` | 诊断量 |
| 噪声修正 interaction excess variance | `0.0000000000` | `> 0` |
| context-shuffle 残差相关 | `0.0089938952` | `abs < 0.10` |
| label-shuffle 残差相关 | `-0.0037839902` | `abs < 0.10` |

原始平衡率在两半密钥之间高度相关，但去除 context 与 mask 加性边际后，真正的
`context x mask` 交互相关只剩 `0.00128`；验证残差方差还低于有限64-key采样的
二项噪声估计，噪声修正后的额外交互方差为零。也就是说，可重复的是稳定的
context/mask 边际，不是创新2需要学习的结构条件交互。因此最终裁决为：

```text
status = hold
decision = innovation2_balance_rate_interaction_not_reproducible
strict kernel membership = stop for PRESENT r7 contexts
cross-key balance probability = stop for PRESENT r7 contexts
neural training = no
remote_scale = no
```

这是对冻结的64-context、240-mask、两组64把采样密钥的本地协议审计，不是神经
模型结果、paper-scale 训练或全密钥空间证明。它足以否决“从当前标签表直接训练或
机械增加 context/key/mask”这一下一步，但不能外推为所有 PRESENT 积分输出性质
均不存在。

权威产物：

```text
outputs/local_audits/
  i2_present_r7_context_mask_balance_rate_128keys_seed0_20260717/
```

`curves.svg` 的散点层已栅格化，文件由约 `2.4 MB` 降至约 `53 KB`。最终以
`1800 x 864` 像素执行 `visual-qa-redraw` 检查，标题、双面板、门槛标记、裁决文字
无重叠或裁切，中文字体完整，坐标和门槛含义明确。

## 32. 推荐下一步：E20 SKINNY-64/64 论文 kernel 协议就绪审计

E19 后停止 PRESENT r7 inactive-context 路线，不继续增加密钥、context、mask、
seed，也不训练连续值网络。下一步转向 Hwang 2026 正文主案例 SKINNY-64/64；
PRESENT 在该论文中只是附录案例，而 SKINNY 与神经积分输出性质的联系更直接。

Hwang Section 5.1 给出了可执行的同预算锚点：SKINNY-64/64、7轮、单个活动明文
cell、MSB-first 64-bit parity feature、`10^6` data，经验矩阵的 `(rank, nullity)`
应为 `(46, 18)`，Table 2 给出完整18维 kernel basis；8轮则只剩
`b28 xor b44 xor b60` 一个 basis。当前源码扫描未发现 keyed SKINNY-64/64 实现，
因此 E20 先做密码与协议复现，不做训练。

E20 的可执行顺序冻结为：

```text
research question = 项目能否逐项复现 Hwang SKINNY-64/64 7轮 exact kernel
same-budget anchor = Hwang Section 5.1 / Table 2 的 64-bit parity fixture
one variable = cipher 从 PRESENT-80 切换为 SKINNY-64/64
implementation = keyed SKINNY-64/64 + 权威测试向量 + 明确 tweakey/round 约定
bit order = 4x4 nibble state，论文 MSB-first，必须用 exact basis 裁决
readiness gate = 权威向量通过，rank=46，nullity=18，18个 Table 2 basis 全部一致
training = none
execution = 本地、磁盘缓存、可恢复进度；未过 readiness 不上远程 GPU
advance = 通过后才设计 structure/mask 输出属性标签及边际/group/fresh-key 控制
stop = exact fixture 无法复现时先审计协议差异，不扩样本、不换网络掩盖问题
```

这个顺序保持创新2的任务不变：预测“给定结构和输出 mask 的积分输出性质”，而不
退回“积分数据与随机数据二分类”。

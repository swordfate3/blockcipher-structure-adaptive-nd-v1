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

E15 固定同一轮数、密钥、上下文和16-bit活动宽度，构造16个由 PRESENT 结构直接
定义的几何，而不是任意滑动：

```text
family A: 四个边界对齐4-nibble块，start = 0,4,8,12
family B: family A 每个结构经过 PRESENT P-layer 一次映射
family C: family A 每个结构经过 PRESENT P-layer 两次映射
family D: 四个 nibble-column，{c,c+4,c+8,c+12}, c=0..3
```

P-layer 映射按当前实现的精确 bit permutation：

```text
p(i) = (16*i mod 63), i < 63
p(63) = 63
```

冻结预算：

```text
structures = 16
keys per structure = 128 = 64 + 64
plaintexts per structure per key = 65536
key generation seed = seed + 3301
key_chunk_size = 4
training = none
```

门槛继续要求 Hwang 原始结构精确四维、所有 joint basis 验证两半、至少4个不同
joint签名和至少8个非平凡结构。通过后才用 E13 同一 mask/边际基线重建标签表；
否则转固定高16位并变化 inactive context，或停止 PRESENT r7 多结构预测路线。

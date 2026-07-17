# 创新2 E36：小状态SPN拓扑标签可识别性审计计划

日期：2026-07-18

状态：已完成 / pass / 标签可识别但独立拓扑样本不足

## 1. 研究问题

E33--E35b已经覆盖GraphGPS、SCGT、cell等变表示、round-shared reasoner和Cipher
Edge-Token Transformer。公平控制下最强CETT真实P-layer dual AUC为`0.671767`，低于
ID边际`0.726528`，且只领先fair-corrupted P`0.006823`。

继续换网络前必须回答：E32b的589个matched cell中，标签是否真的包含足够、组外稳定的
P-layer条件变化，还是主要由round/structure/mask/S-box边际解释？

E36不训练模型，直接审计完整`4 S-box x 4 P-layer`精确标签立方体。

## 2. 数据与选择边界

```text
label source = i2_small_spn_exact_label_width_16ciphers_256keys_seed0_20260718
cell source  = i2_small_spn_matched_contrast_readjudication_20260718
label shape  = 16 x 4 x 14 x 64
selected cells = 589
train block  = S0..S2 x P0..P2
unseen-P     = S0..S2 x P3
unseen-S     = S3 x P0..P2
dual         = S3 x P3
```

selected mask仍只来自九个train topology；E36可以读取heldout标签做审计，但不能反向修改
589个cell或训练任何模型。

## 3. 指标定义

对每个base cell的`4x4`标签矩阵`Y[s,p]`计算：

```text
train P-sensitive any-S:
  S0..S2中至少一个固定S下，P0..P2标签发生变化

train P-sensitive all-S:
  S0..S2每一个固定S下，P0..P2标签都发生变化

heldout P3 novel any-S / all-S:
  对S0..S2，P3标签是否不同于至少一个P0..P2

dual P-effect:
  在S3下，P3标签是否不同于至少一个P0..P2

SxP interaction:
  GF(2)混合二阶差分
  Y[s,p] xor Y[s,0] xor Y[0,p] xor Y[0,0]
  是否对任意(s,p)非零
```

同时统计dual P-effect子集上的`Y[S3,P3]`正负数量，防止只有单类变化。

## 4. 冻结门

以下阈值在读取详细E36统计前冻结：

```text
selected cells                         >= 512
train P-sensitive any-S                >= 192
train P-sensitive all-S                >= 64
heldout P3 novel any-S                 >= 192
heldout P3 novel all-S                 >= 64
dual P-effect cells                    >= 128
dual P-effect target positive          >= 32
dual P-effect target negative          >= 32
train-block SxP interaction cells      >= 128
full 4x4 SxP interaction cells         >= 192
```

这些门要求信号达到百级base-cell宽度，而不是依赖少数位置。任何一项失败都不通过。

## 5. 裁决

全部通过：

```text
decision = innovation2_small_spn_topology_labels_identifiable
```

只允许重新设计一个更干净的benchmark：在train-only条件下显式选择P-sensitive且
interaction-rich cell，并重新做边际/公平控制审判。不得直接恢复网络训练。

宽度或类平衡失败：

```text
decision = innovation2_small_spn_topology_labels_not_identifiable
```

停止当前16-bit合成标签路线；下一步返回确定性候选提供者、不同输入子空间族或不同target，
而不是继续搜索神经网络架构。

协议失败：

```text
decision = innovation2_small_spn_topology_label_audit_protocol_invalid
```

## 6. 产物

```text
run_id = i2_small_spn_topology_label_identifiability_20260718
output = outputs/local_audits/i2_small_spn_topology_label_identifiability_20260718/
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

证据只描述16-bit合成SPN标签可识别性，不是神经结果、真实密码结果或确定性高轮积分证明。

## 7. 实际结果

```text
run_id   = i2_small_spn_topology_label_identifiability_20260718
status   = pass
decision = innovation2_small_spn_topology_labels_identifiable
training = false
```

| 指标 | 实际cell | 最低门 | 比例 |
|---|---:|---:|---:|
| selected cells | 589 | 512 | 1.0000 |
| train P-sensitive any-S | 585 | 192 | 0.9932 |
| train P-sensitive all-S | 227 | 64 | 0.3854 |
| heldout P3 novel any-S | 588 | 192 | 0.9983 |
| heldout P3 novel all-S | 296 | 64 | 0.5025 |
| dual P-effect | 391 | 128 | 0.6638 |
| train S×P interaction | 521 | 128 | 0.8846 |
| full S×P interaction | 579 | 192 | 0.9830 |

dual P-effect 391个cell上的目标类为：

```text
positive = 234
negative = 157
```

所有十个冻结宽度与类平衡门均通过。按轮分解：

```text
2轮 selected=0
3轮 selected=201, P-sensitive=201, dual-effect=151, interaction=194
4轮 selected=373, P-sensitive=369, dual-effect=238, interaction=370
5轮 selected=15,  P-sensitive=15,  dual-effect=2,   interaction=15
```

## 8. 解释与下一步

E36排除了“标签没有P-layer信号”的解释。真正瓶颈是拓扑样本量：E33--E35b虽然有
5301行train label，但独立结构只有`3 S-box x 3 P-layer=9`种，其中只有3个独立P-layer。
其余行只是同一9张图上的round/structure/mask查询。用3个随机permutation学习到第4个
随机permutation的结构规律，统计上严重欠定。

下一步E37先扩展确定性标签族，不训练网络：

```text
4 S-box x 16 P-layer = 64 variants
train        = S0..S2 x P0..P11  = 36 topologies
unseen-S     = S3 x P0..P11      = 12
unseen-P     = S0..S2 x P12..P15 = 12
dual         = S3 x P12..P15      = 4
all 256 toy master keys
rounds/structures/masks保持不变
```

E37必须重新执行train-only matched选择、P-sensitive/interaction宽度、ID边际和公平控制门。
只有扩展benchmark通过，才允许在同一模型中比较GraphGPS与CETT；当前不直接恢复训练。

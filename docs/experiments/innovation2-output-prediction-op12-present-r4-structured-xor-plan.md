# 创新2 OP12：PRESENT四轮预注册多输出bit结构化XOR预测门

日期：2026-07-21

状态：本地实现门通过 / 远程正式门待启动

## 1. 唯一研究问题

OP11在两把独立固定秘密密钥上确认了八个PRESENT三轮真实密文输出bit，并证明专用八输出头优于
完整64输出anchor。OP12只回答：进入PRESENT四轮后，直接预测由这些位置构成的结构化XOR，是否比
继续预测组成bit、同重量几何控制和标签打乱更容易，从而提供“多输出bit XOR提高预测轮数”的证据？

```text
input  = 未见明文P的64个MSB-first bit
C      = PRESENT_K^4(P)
target = XOR_{j in preregistered mask} C[j]
```

每个`0/1`标签都是同一条明文对应的真实四轮密文输出函数值，不是真假样本、平衡/不平衡、积分结构
类别、kernel或关系标签。

## 2. 冻结数据与攻击契约

远程正式门复用OP11的第二把固定未知秘密密钥生成规则和明文生成seed：

```text
cipher                 = PRESENT-80
rounds                 = 4
seed / fixed key       = 1
train rows             = 131072 total plaintext/ciphertext pairs
test rows              = 65536 total disjoint plaintext/ciphertext pairs
input                   = 64 MSB-first plaintext bits
selected positions      = [0, 2, 8, 10, 32, 34, 40, 42]
epochs                  = 100 per model
batch                   = 250
optimizer               = RMSprop
loss                    = raw-output MSE
learning rate           = 0.001
selection               = final epoch
device                  = lxy-a6000 GPU0
```

该任务没有正负类别，不使用`/class`。训练与测试明文必须唯一且零重合；密钥、明文、真实64-bit目标按块
落盘，缓存参数严格匹配后才能复用；四个模型逐epoch保存latest checkpoint并支持恢复。

本地实现门只用`64 train / 64 test / 1 epoch / CPU`，不作性能结论。

## 3. 结构化mask与匹配控制

### 3.1 主要同末轮S-box双bit mask

以下四组已经通过`MSB -> integer -> inverse-P source -> S-box`确定性映射与正向P-layer往返检查：

```text
same_sbox_pair_0_32   = [0, 32]   source S-box 15, roles 3 xor 1
same_sbox_pair_2_34   = [2, 34]   source S-box 13, roles 3 xor 1
same_sbox_pair_8_40   = [8, 40]   source S-box  7, roles 3 xor 1
same_sbox_pair_10_42  = [10, 42]  source S-box  5, roles 3 xor 1
```

### 3.2 同角色四bit mask

```text
same_role4_0_2_8_10       = [0, 2, 8, 10]       four source S-box role-3 bits
same_role4_32_34_40_42    = [32, 34, 40, 42]    four source S-box role-1 bits
```

### 3.3 同重量几何控制

双bit控制保持所用八位置和全家族bit频次完全匹配，但按密文显示nibble邻近而不是逆P后的同S-box配对：

```text
output_nibble_pair_0_2    = [0, 2]
output_nibble_pair_8_10   = [8, 10]
output_nibble_pair_32_34  = [32, 34]
output_nibble_pair_40_42  = [40, 42]
```

四bit控制同样覆盖每个位置一次，但混合两个输出角色，取消“跨四个来源S-box同角色”假设：

```text
mixed_role4_0_2_32_34    = [0, 2, 32, 34]
mixed_role4_8_10_40_42   = [8, 10, 40, 42]
```

不得在四轮结果揭盲后增加、删除、重排mask或用64-bit枚举结果替换这些候选。

## 4. 同预算四行矩阵

```text
selected8_mlp_true_output        输出八个预注册真实密文bit，四轮单bit anchor
structured6_mlp_true_xor         输出六个预注册结构化XOR值
geometry6_mlp_true_xor           输出六个同重量几何控制XOR值
structured6_mlp_label_shuffle    与structured6完全同构，只打乱训练标签行
```

四行使用同一明文、密钥、split、两隐藏层`hidden=1936`骨干、优化器、epoch、batch和初始化规则。三种
XOR模型参数完全匹配；selected8 anchor只差末层输出维数。标签打乱只作用于训练行，测试仍使用真实
结构化XOR标签。

## 5. 派生parity与组成bit基线

对每个结构化mask，从`selected8_mlp_true_output`的对应bit原始分数裁剪到`[0,1]`后计算：

```text
P(XOR=1) = (1 - PRODUCT_j(1 - 2*p_j)) / 2
```

报告派生parity的AUC、阈值准确率和裁剪率；同时报告该mask组成bit中最高的单bit AUC。直接XOR头必须
超过派生parity和最佳组成bit，才支持“直接学习组合函数”而不是仅复述已有单bit信号。

## 6. 逐mask与家族门

一个结构化mask通过，必须同时满足：

```text
direct structured AUC                         >= 0.510
direct accuracy - majority                    >= +0.005
direct AUC - matched shuffled AUC             >= +0.005
direct AUC - paired geometry-control AUC       >= +0.005
direct AUC - selected8-derived parity AUC      >= +0.005
direct AUC - best component-bit AUC            >= +0.002
```

家族门：

```text
same-S-box pair family pass = 至少2/4个主要双bit mask通过全部六门
same-role four-bit pass      = 两个四bit mask均通过全部六门
OP12 pass                    = 任一家族通过
```

smoke只检查协议和执行，不应用性能门。正式数据、标签、bit顺序、mask几何、打乱、结果行、历史行、
checkpoint或指标无效时裁决`protocol_invalid`，只修协议。

## 7. 结果产物与视觉门

每次运行至少生成：

```text
results.jsonl
history.csv
metadata.json
summary.json
gate.json
checkpoint_manifest.json
progress.jsonl
data/cache_metadata.json
curves.svg
```

正式运行预期`26`行结果：八个selected-bit anchor、六个structured、六个geometry control及六个
matched shuffle。最终图必须通过`visual-qa-redraw`像素检查，再写`visual_qa_passed.marker`并刷新
`outputs/00_RECENT_RESULTS.md`和JSON索引。

## 8. 下一步与停止项

若OP12通过，下一步OP13只在seed0第一把固定秘密密钥上使用相同四轮、相同mask、同预算四行矩阵做
独立密钥确认；不得直接进入五轮。只有OP13复现同一家族门，才讨论五轮或四/八bit扩展。

若OP12未通过，停止该结构化XOR的样本、epoch、层数、mask枚举和轮数扩展；保留OP11“三轮易预测
位置跨两密钥成立、专用头减少多任务干扰”为创新2论文结论。禁止按四轮结果后验挑mask、机械增加到
`262144`训练行或绕过控制只报告最高AUC。

## 9. 本地实现门结果

本地使用`64 train / 64 test / 1 epoch / CPU`完成四行端到端实现门：

```text
run_id   = i2_output_prediction_op12_present_r4_structured_xor_smoke_20260721
status   = pass
decision = innovation2_r4_structured_xor_local_smoke_passed
output   = outputs/local_smoke/i2_output_prediction_op12_present_r4_structured_xor_smoke_20260721/
```

PRESENT官方向量、四轮真实密文bit重放、明文唯一/互斥、六个结构mask、六个几何控制、逆P来源、家族
bit频次匹配、训练标签打乱、26行结果、4行训练历史、四checkpoint哈希、直接/派生/组成bit比较字段与
全部有限指标检查均通过。64条测试上的`0/6` mask性能门不作结论，唯一解锁动作是远程正式门。

最终`curves.svg`经过两轮`visual-qa-redraw`像素检查。第一版右上高点和右下柱顶接近图例；扩大两个
面板上方纵轴留白后重新渲染，最终标题、图例、柱体、曲线、mask标签、门槛线和裁决无重叠、遮挡、
裁切、缺字或歧义，记录`visual_qa_passed.marker`。

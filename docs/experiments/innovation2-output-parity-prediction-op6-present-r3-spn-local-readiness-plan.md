# 创新2 OP6：PRESENT三轮真实输出parity的SPN局部表示就绪门

日期：2026-07-21

状态：已完成 / nibble邻接SPN局部网络未通过 / 转精确bit-level路由

## 1. 研究问题

OP5两颗seed在三轮均留下约`0.525`的结构对齐真实密文输出parity弱残差，但当前全连接MLP未过
`0.55/+0.03/+0.03`门。OP6只回答：在标签、数据和预算不变时，按PRESENT nibble与P-layer
依赖组织的局部网络，能否恢复三轮输出预测信号并将收益归因于真实P层拓扑。

```text
input = unseen plaintext P
C = PRESENT_K^3(P)
target_m = XOR_{j in aligned_mask_m} C[j]
```

仍然预测每条明文对应的真实密文输出值，不引入真假、平衡或关系类别。

## 2. 冻结数据与预算

OP6复用OP5 seed0：

```text
cipher                 = PRESENT-80
rounds                 = 3
secret key             = one fixed unknown seed0 key
train/validation/test  = 4096 / 1024 / 2048 disjoint plaintexts
input                  = 64 LSB-first plaintext bits only
target                 = 16 aligned real-ciphertext parity outputs
epochs                 = 5
batch size             = 128
optimizer              = AdamW, lr=1e-3, weight_decay=1e-4
selection              = final epoch
device                 = local CPU
```

不能改变密钥、明文、输出mask、标签、样本、epoch、损失、测试集或指标。

## 3. 候选网络与控制

候选复用仓库现有`PresentPLayerMixerBlock`：

```text
64 plaintext bits
-> 16 nibble tokens（按PRESENT MSB token约定重排）
-> 4-bit to token_dim=28 encoder + learned nibble position embedding
-> 2 x PRESENT P-layer nibble-adjacency mixer block
-> shared per-token output head
-> reverse token order back to 16 LSB-first aligned parity logits
```

两层mixer对应三轮目标最后一个S-box之前的两次S-box/P-layer扩散。候选不输入密钥、密文、标签或
手工计算的中间状态。

四行矩阵：

```text
aligned_parity_mlp                  当前两层全连接MLP同预算锚点
spn_local_true_p                    真实P层邻接候选
spn_local_shuffled_p                确定性错误P层邻接、同参数控制
spn_local_true_p_label_shuffle      只打乱训练标签、测试仍为真实输出
```

`spn_local_true_p`与`spn_local_shuffled_p`必须除固定拓扑buffer外逐参数同初始化；候选参数量须与
MLP的`26896`相差不超过1%，且不得通过更大容量获得优势。

## 4. 协议与性能门

协议检查至少包括：

```text
OP5固定密钥真实输出数据契约全部通过
输入只含明文bit，输出逐项等于真实密文aligned parity
MSB token重排与LSB输出逆重排手工fixture通过
真实P与shuffled P均为固定合法拓扑且不同
true/shuffled模型可训练参数初值逐项相同
true/shuffled参数量完全相同
候选与MLP参数量差 <= 1%
四行训练完成、指标有限、打乱控制测试标签真实
```

只有同时满足以下四项，才裁决SPN局部表示通过：

```text
true-P macro AUC                     >= 0.55
true-P macro AUC - MLP macro AUC    >= +0.03
true-P macro AUC - shuffled-P AUC   >= +0.03
true-P macro AUC - label-shuffle AUC>= +0.03
```

只超过MLP但不超过错误P，属于通用容量/归纳偏置收益，不算PRESENT拓扑归因通过。

## 5. 执行与下一步

```text
run_id = i2_output_parity_prediction_op6_present_r3_spn_local_readiness_seed0_20260721
output = outputs/local_readiness/i2_output_parity_prediction_op6_present_r3_spn_local_readiness_seed0_20260721/
```

本地CPU执行，生成JSONL/CSV/JSON/SVG完整产物并经`visual-qa-redraw`检查。

若四项门全部通过，下一步只在独立固定密钥seed1复验相同矩阵；seed1确认后才重开四轮步进。若
绝对信号提高但拓扑归因未过，只保留通用局部表示诊断并审计精确bit-role路由；若仍在随机附近，
停止该nibble邻接架构，转精确bit-level S-box/P-layer路由，不加样本、epoch、轮数或远程规模。

OP6是三轮本地表示就绪门，不是高轮攻击、论文复现、SOTA或创新2最终上限。

## 6. 正式结果

15项数据、输出、token顺序、拓扑控制、参数配平与训练检查全部通过。候选`26881`参数，MLP
`26896`参数，差异仅`15`个参数；true/shuffled除固定拓扑buffer外初值逐参数相同。

```text
status                             = hold
decision                           = innovation2_output_parity_present_r3_spn_local_not_ready

MLP accuracy                       = 0.517547607
MLP macro AUC                      = 0.526979820
SPN-local true-P accuracy          = 0.504608154
SPN-local true-P macro AUC         = 0.513589961
SPN-local shuffled-P accuracy      = 0.506896973
SPN-local shuffled-P macro AUC     = 0.509065359
SPN-local label-shuffle accuracy   = 0.497314453
SPN-local label-shuffle macro AUC  = 0.495423303

true-P - MLP macro AUC             = -0.013389859
true-P - shuffled-P macro AUC      = +0.004524602
true-P - label-shuffle macro AUC   = +0.018166658
```

粗粒度nibble邻接候选没有恢复三轮信号：绝对AUC未到`0.55`，比MLP低`0.0134`，相对错误P层仅
`+0.0045`。因此不能将小幅高于标签打乱的差值解释为PRESENT拓扑收益，也不应继续增加该模型
深度、样本、epoch或轮数。

失败机制与模型结构一致：现有`PresentPLayerMixerBlock`把一个目标nibble的多个来源token做均值，
保留了“哪些nibble相连”，但抹掉了PRESENT P-layer中“源nibble的第几个bit进入目标nibble的第几个
bit”的角色。三轮对齐parity依赖这些bit-role组合，粗粒度邻接可能比全连接MLP损失更多信息。

下一步只开放精确bit-level路由：每个64-bit位置保持独立通道，在每个S-box内做4-bit局部混合，再
按照真实P层逐bit置换；错误P层必须使用同形的另一合法64-bit置换，参数与初始化完全配平。真实密文
aligned parity标签、seed0数据、MLP锚点、训练预算和四项门槛全部不变。

`curves.svg`经`visual-qa-redraw`渲染为`1800 x 965`像素检查：四模型标签、接近随机的柱值、局部
纵轴、随机基线、差值、裁决和证据边界均无文字重叠、裁切、缺字或语义混淆；验收记录为
`visual_qa_passed.marker`。

# 创新2 OP1：固定密钥密文输出parity预测统一就绪门

日期：2026-07-21

状态：已完成 / 本地CPU就绪门通过 / 连续nibble parity未过信号门

## 1. 研究纠正

创新2的主任务冻结为**输出预测**，不是`real-vs-random`、`structured-vs-random`或ATM关系分类。
一个输出parity虽然取值为`0/1`并可用BCE训练，但标签必须直接来自同一输入的真实密码输出：

```text
C = E_K^r(P)
y_m = XOR_{j in mask_m} C[j]
```

H0 multiset二分类、PG-NBPO整cube平衡标签、跨密钥kernel membership和E98--E106 ATM关系排序
退出创新2主实验轨迹，只保留为历史基线、旁支或失败边界。

## 2. OP1唯一问题

OP1只回答：固定秘密密钥的已知明文攻击设置下，仓库能否用同一批互斥明密文对，正确训练并评价：

1. `plaintext -> 64 ciphertext bits`完整输出预测基线；
2. `plaintext -> 16 nibble parity bits`直接parity输出；
3. 从完整输出模型的bit概率推导出的16个parity概率；
4. 训练标签打乱、测试仍使用真实parity的控制。

本门不判断parity方法优于完整输出，也不声称攻击轮数。它只校准数据、密钥、目标和指标语义。

## 3. 冻结攻击与数据契约

```text
cipher                 = PRESENT-80
rounds                 = 1
secret keys            = 1 fixed unknown key for the complete run
train plaintexts       = 4096
validation plaintexts  = 1024
test plaintexts        = 2048
split                   = plaintext-disjoint
input                   = 64 LSB-first plaintext bits only
full target             = 64 LSB-first ciphertext bits
parity target           = XOR of each contiguous 4-bit output nibble
parity masks            = 16 frozen masks, one per nibble
sample-class label      = none
seed                    = 0
device                  = local CPU
```

训练、验证和测试使用同一把秘密密钥，模拟论文的单密钥已知明文攻击；三者明文严格互斥。模型
输入不含密钥、密文、真假类别、积分统计或标签派生特征。后续不同密钥必须分别训练模型，再汇总
密钥间分布，不能把每行随机密钥而隐藏密钥的矛盾标签送给同一模型。

## 4. 同预算模型与指标

完整输出和直接parity模型使用相同的两层MLP backbone、初始化seed、batch、epoch、优化器和明文
split，仅输出维度与监督目标不同。OP1冻结：

```text
hidden width   = 128
epochs         = 5
batch size     = 128
optimizer      = AdamW
learning rate  = 1e-3
weight decay   = 1e-4
selection      = final epoch
```

完整输出报告64-bit逐bit accuracy、macro AUC和完整64-bit exact match。parity报告16个目标的逐cell
accuracy、macro AUC和16-bit parity-vector exact match。随机parity基线为`0.5`；4-bit或64-bit exact
match不能与parity accuracy直接相减。

从完整输出概率`p_j`推导mask奇parity概率：

```text
P(XOR=1) = (1 - PRODUCT_j(1 - 2*p_j)) / 2
```

后续候选只有同时超过该派生parity、直接线性/MLP基线和标签打乱，才能主张直接parity监督有价值。

## 5. 就绪裁决

`readiness_passed`要求：

```text
PRESENT官方向量仍通过
固定秘密密钥在三个split中一致
train/validation/test明文零重合
输入只等于明文bit
完整输出标签可由标量PRESENT逐行重放
16个parity标签逐项等于完整输出bit XOR
16个mask互异且恰好覆盖64个输出bit
每个split的每个parity目标同时含0和1
三行训练完成且全部指标有限
训练标签打乱控制未修改validation/test真实标签
```

协议失败只修实现，不解释神经性能。通过后根据校准信号选择唯一下一动作：直接parity若已达到
macro AUC `>=0.55`且相对标签打乱`>=+0.03`，才开放多固定密钥`r1--r4`轮数梯；否则OP2只在
一轮比较连续输出nibble与末轮S-box/P-layer对齐的四位置mask，其他数据、密钥、模型和预算不变。
OP2前不启动远程GPU，不执行高轮搜索。

## 6. 产物

```text
run_id = i2_output_parity_prediction_readiness_present_r1_seed0_20260721
output = outputs/local_readiness/i2_output_parity_prediction_readiness_present_r1_seed0_20260721/
```

至少生成`results.jsonl`、`history.csv`、`summary.json`、`gate.json`、`metadata.json`、
`dataset_summary.json`、`masks.csv`、`progress.jsonl`和中文`curves.svg`。图必须经过
`visual-qa-redraw`像素检查，完成后刷新最近结果索引并给出OP2是否开放的证据裁决。

## 7. 正式结果

OP1已完整执行，16项数据、输出语义和训练协议检查全部通过：

```text
status                          = pass
decision                        = innovation2_output_parity_prediction_readiness_passed
full-output bit accuracy        = 0.603141785
full-output bit macro AUC       = 0.639698169
full-derived parity accuracy    = 0.498352051
full-derived parity macro AUC   = 0.498583423
direct parity accuracy          = 0.496917725
direct parity macro AUC         = 0.494962902
shuffled parity accuracy        = 0.502380371
shuffled parity macro AUC       = 0.506306629
```

该结果证明固定密钥、互斥明文、真实64-bit输出、16个输出parity和训练控制已经形成一条干净的
输出预测通路。完整输出逐bit在4096条训练明文、5 epochs下已有明显一轮信号，但连续输出nibble的
直接parity与完整输出派生parity均处于随机附近，直接parity也未超过标签打乱。因此OP1不是parity
方法成功证据，也不开放`r2--r4`。

机制上，PRESENT一轮后的连续4个输出位置来自多个S-box，经XOR后比单bit输出形成更高阶组合。
推荐OP2只改变mask几何：加入“同一末轮S-box四个输出bit经P-layer映射后的位置集合”，与连续
nibble mask做同重量配对；明文、秘密密钥、split、MLP、epoch和指标全部保持不变。只有对齐mask
能够在真实测试明文上超过`0.55` macro AUC且相对标签打乱达到`+0.03`，才讨论多密钥或扩轮。

OP1是本地一轮就绪实验，不是高轮输出预测、正式攻击轮数、论文复现、SOTA或直接parity优势。

最终`curves.svg`已按`visual-qa-redraw`渲染为`1800 x 977`像素检查：标题、任务解释、两面板、
三组柱值、随机基线、图例和底部裁决均无文字重叠、裁切、缺字或语义歧义，验收记录为
`visual_qa_passed.marker`。

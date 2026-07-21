# 创新2 OP2：PRESENT一轮输出parity mask几何校准

日期：2026-07-21

状态：已完成 / S-box与P层对齐mask信号通过 / 只开放独立固定密钥复验

## 1. 唯一研究问题

OP2继续执行固定密钥已知明文下的**真实密文输出预测**，不引入真假样本、平衡类别或关系类别。
本实验只回答：将四位输出parity从“密文连续nibble”改为“同一末轮S-box的四个输出位经
P-layer后的落点”，是否能在同预算下恢复可归因的输出预测信号。

```text
C = PRESENT_K^1(P)
y_m = XOR_{j in mask_m} C[j]
```

每一行标签仍由该行明文的真实密文直接计算，因此目标虽然为`0/1`，语义仍是输出预测。

## 2. 冻结数据与攻击契约

OP2完全复用OP1：

```text
cipher                 = PRESENT-80
rounds                 = 1
secret keys            = 1 fixed unknown key for the complete run
train plaintexts       = 4096
validation plaintexts  = 1024
test plaintexts        = 2048
split                   = plaintext-disjoint
input                   = 64 LSB-first plaintext bits only
seed                    = 0
device                  = local CPU
```

秘密密钥、全部明文、真实密文、split、网络初始化规则、优化器和评价集合必须与OP1确定性一致。
本实验不能通过换密钥、换样本或换split制造差值。

## 3. 唯一变量：四位mask几何

同重量基线为16个连续输出nibble：

```text
contiguous_mask[s] = 0xF << (4*s)
```

候选为16个末轮S-box/P-layer对齐mask：

```text
aligned_mask[s] = Present80.permutation_layer(0xF << (4*s))
```

例如：

```text
contiguous_mask[0] positions = {0, 1, 2, 3}
aligned_mask[0]    positions = {0, 16, 32, 48}
```

PRESENT的P-layer使连续密文四位分别来自多个S-box；对齐mask的四位则来自同一个S-box。
两组各含16个互异四位mask，且各自恰好覆盖全部64个输出位置一次。

## 4. 同预算矩阵

```text
full_output_mlp                 明文 -> 64个真实密文bit
contiguous_parity_mlp           明文 -> 16个连续mask真实parity
aligned_parity_mlp              明文 -> 16个对齐mask真实parity
aligned_parity_label_shuffle    只打乱训练集对齐parity；测试仍用真实parity
```

四行共享两层MLP backbone：

```text
hidden width   = 128
epochs         = 5
batch size     = 128
optimizer      = AdamW
learning rate  = 1e-3
weight decay   = 1e-4
selection      = final epoch
```

完整输出模型还要分别推导连续mask与对齐mask的parity概率，作为辅助基线。比较主指标为测试集
macro AUC，支持指标为cell accuracy、exact match、majority accuracy和每轮验证历史。

## 5. 协议与裁决门

执行前必须验证：

```text
OP1全部固定密钥、明文互斥、真实输出重放检查通过
两组mask均为16个互异weight-4 mask并分别覆盖64位一次
aligned_mask逐项等于P-layer(contiguous_mask)
两组数据的密钥、明文、密文、输入和完整输出标签逐项相同
两组parity均可从真实64-bit密文输出逐项重放
打乱控制只改训练标签，不改验证/测试真实标签
四行训练完成且指标有限
```

只有同时满足以下三项，才裁决`aligned mask geometry supported`：

```text
aligned macro AUC                         >= 0.55
aligned macro AUC - shuffled macro AUC   >= +0.03
aligned macro AUC - contiguous macro AUC >= +0.03
```

否则裁决`mask geometry not calibrated`。不能用accuracy单指标替代macro AUC门，也不能选择性汇报
16个mask中的最好一个。

## 6. 执行、产物与下一步

```text
run_id = i2_output_parity_prediction_op2_mask_geometry_present_r1_seed0_20260721
output = outputs/local_readiness/i2_output_parity_prediction_op2_mask_geometry_present_r1_seed0_20260721/
```

本地CPU执行，生成`results.jsonl`、`history.csv`、`summary.json`、`gate.json`、`metadata.json`、
`dataset_summary.json`、`masks.csv`、`progress.jsonl`和中文`curves.svg`。图像必须经过
`visual-qa-redraw`像素检查，完成后刷新最近结果索引。

若通过，下一步OP3只增加独立固定密钥，保持一轮和全部模型预算不变，确认mask几何收益能否跨密钥
复现；跨密钥确认后才设计`r2--r4`轮数梯。若失败，停止机械增加样本、epoch、密钥和轮数，转为
审计Kimura类输出预测方法的精确输入编码、网络输出和固定密钥协议，再决定是否更换模型表示。

本地一轮结果无论通过或失败，都不是高轮攻击、论文复现或SOTA证据。

## 7. 正式结果

OP2按冻结协议完整执行，16项输出预测、mask几何、配对数据和训练检查全部通过：

```text
status                               = pass
decision                             = innovation2_output_parity_mask_geometry_supported
full-output bit accuracy             = 0.603141785
full-output bit macro AUC            = 0.639698169
full-derived aligned parity accuracy = 0.499694824
full-derived aligned parity AUC      = 0.495724814
contiguous parity accuracy           = 0.496917725
contiguous parity macro AUC          = 0.494962902
aligned parity accuracy              = 0.871490479
aligned parity macro AUC             = 0.961079724
aligned label-shuffle accuracy       = 0.508575439
aligned label-shuffle macro AUC      = 0.514765490
aligned - contiguous macro AUC       = +0.466116822
aligned - shuffled macro AUC         = +0.446314233
```

三项预注册性能门分别为`0.55`、`+0.03`和`+0.03`，实际结果均大幅通过。连续密文nibble与
标签打乱仍在随机附近，而只改变真实密文输出mask的几何后，直接parity预测恢复到`0.9611`
macro AUC；因此该差值可以在本门范围内归因于末轮S-box/P-layer对齐。

完整输出模型逐bit有信号，但用64个边际bit概率和独立假设推导的对齐parity仍为`0.4957` AUC。
直接parity模型的`0.9611`说明它学习了四位联合异或函数，而不是简单复用单bit边际概率。这是支持
“直接预测结构对齐输出函数”的方法证据，但一轮PRESENT的局部S-box函数本身较简单，不能外推到
高轮或跨密钥。

下一步冻结为OP3独立固定密钥确认：只更换确定性秘密密钥与互斥明文样本，保持一轮、两组mask、
两层MLP、初始化配对、4096/1024/2048条split、5 epochs和三项门槛不变。只有seed1也通过且双密钥
联合裁决通过，才设计`r2--r4`轮数梯。不得跳过OP3直接增加轮数、样本或远程GPU规模。

`curves.svg`已由`visual-qa-redraw`渲染为`1800 x 982`像素检查，标题、任务说明、三组柱值、随机
基线、坐标轴、差值、裁决与证据边界均无文字重叠、裁切、缺字或语义歧义；验收记录为
`visual_qa_passed.marker`。

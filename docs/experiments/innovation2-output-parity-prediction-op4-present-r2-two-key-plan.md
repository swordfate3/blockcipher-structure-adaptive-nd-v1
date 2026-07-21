# 创新2 OP4：PRESENT二轮结构对齐输出parity双固定密钥轮数步进

日期：2026-07-21

状态：已完成 / PRESENT二轮双固定密钥门通过 / 只开放三轮同预算门

## 1. 唯一研究问题

OP2--OP3已在两把独立固定密钥下确认PRESENT一轮结构对齐真实密文输出parity可预测。OP4只回答：
将加密轮数从一轮增加到两轮后，同一输出预测任务在完全相同预算下能否保持可归因信号。

```text
input = unseen plaintext P
C = PRESENT_K^2(P)
target_m = XOR_{j in aligned_mask_m} C[j]
```

标签仍是每条明文对应的真实二轮密文输出值，不是`real-vs-random`、平衡类别或关系类别。

## 2. 唯一变量与冻结项

相对OP3唯一改变：

```text
rounds = 1 -> 2
```

冻结：

```text
cipher                    = PRESENT-80
fixed secret key seeds    = 0, 1（每个seed单独训练模型）
train/validation/test     = 4096 / 1024 / 2048 plaintexts per key
split                      = plaintext-disjoint within each key
input                      = 64 LSB-first plaintext bits only
full target                = 64 true ciphertext bits
contiguous parity target   = 16 contiguous weight-4 ciphertext masks
aligned parity target      = 16 last-round S-box/P-layer weight-4 masks
models                     = full / contiguous / aligned / aligned-label-shuffle MLP
hidden width               = 128
epochs                     = 5
batch size                 = 128
optimizer                  = AdamW, lr=1e-3, weight_decay=1e-4
device                     = local CPU
```

连续和对齐parity模型继续共享初始化seed。不能调样本、epoch、容量、学习率、mask、指标或选择最好
输出位置。

## 3. 每密钥门与联合裁决

每把固定密钥独立要求：

```text
aligned macro AUC                         >= 0.55
aligned macro AUC - shuffled macro AUC   >= +0.03
aligned macro AUC - contiguous macro AUC >= +0.03
```

联合门额外要求seed0/seed1密钥不同、跨运行明文零重合、两次输出预测协议完整且两把密钥均通过。
只有全部成立，裁决：

```text
innovation2_output_parity_present_r2_two_key_supported
```

任一密钥协议有效但性能门失败，裁决`r2_two_key_not_supported`。协议、anchor、轮数、密钥或明文
独立性无效时，只修协议，不解释性能。

## 4. 执行与下一步

```text
seed0 run = i2_output_parity_prediction_op4_present_r2_seed0_20260721
joint run = i2_output_parity_prediction_op4_present_r2_seed1_joint_20260721
output    = outputs/local_readiness/<run_id>/
```

先完成seed0单密钥门，再由seed1运行读取seed0冻结anchor并形成双密钥联合门。两次均刷新最近结果
索引；最终双密钥图使用`visual-qa-redraw`像素检查。

若通过，OP5只将轮数从`r2`改为`r3`，继续同预算双密钥衰减门；不增加样本、epoch、容量、密钥数
或远程GPU规模。若未通过，停止机械扩轮，下一步只比较保持真实密文输出标签不变的SPN局部连接
表示与当前全连接MLP，先在二轮本地恢复信号归因。两轮本地结果不是高轮攻击、论文复现或SOTA。

## 5. 正式结果

seed0单密钥门与seed1联合门均按冻结协议完成。最终8项anchor、轮数、密钥、明文独立性与双密钥
性能检查全部通过：

```text
status                                  = pass
decision                                = innovation2_output_parity_present_r2_two_key_supported

seed0 full-output bit macro AUC         = 0.532194374
seed0 aligned parity accuracy           = 0.589050293
seed0 aligned parity macro AUC          = 0.626573292
seed0 contiguous parity macro AUC       = 0.500600010
seed0 aligned label-shuffle macro AUC   = 0.501177670
seed0 aligned - contiguous macro AUC    = +0.125973282
seed0 aligned - shuffled macro AUC      = +0.125395622

seed1 full-output bit macro AUC         = 0.529793857
seed1 full-derived aligned parity AUC   = 0.495162563
seed1 aligned parity accuracy           = 0.588104248
seed1 aligned parity macro AUC          = 0.626496197
seed1 contiguous parity macro AUC       = 0.503019860
seed1 aligned label-shuffle macro AUC   = 0.502785383
seed1 aligned - contiguous macro AUC    = +0.123476337
seed1 aligned - shuffled macro AUC      = +0.123710813

two-key minimum aligned macro AUC       = 0.626496197
two-key mean aligned macro AUC          = 0.626534744
two-key aligned macro AUC range         = 0.000077095
```

从一轮到二轮，对齐parity平均AUC由`0.9621`降至`0.6265`，说明信号明显随轮数衰减；但二轮两把
密钥仍同时超过`0.55`绝对门，并相对连续mask和打乱控制保持约`+0.124--+0.126`的稳定优势。
跨密钥极差仅`0.000077`，因此二轮信号不是单密钥偶然。

二轮完整密文逐bit AUC只有约`0.53`，用完整输出边际bit概率推导对齐parity也仍在随机附近；直接
对齐parity模型达到`0.6265`，继续支持“直接学习结构对齐联合输出函数”而非从单bit边际概率机械
组合。不过该优势仍是低轮、本地小样本和两把密钥范围内的方法证据。

裁决只开放OP5：将轮数从`r2`改为`r3`，其余两把密钥、数据预算、模型、mask、epoch和门槛全部
不变。不得先增加样本、epoch、容量、密钥数或远程GPU规模。若三轮未过双密钥门，则停止机械扩轮，
转为三轮或最后通过轮的SPN局部表示重设计。

双密钥`curves.svg`已由`visual-qa-redraw`渲染为`1800 x 984`像素检查：标题准确标明PRESENT二轮，
双面板、三种方法、随机基线、数值、汇总、裁决与证据边界均无文字重叠、裁切、缺字或结构歧义；
验收记录为`visual_qa_passed.marker`。

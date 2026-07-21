# 创新2 OP3：PRESENT一轮结构对齐输出parity独立固定密钥确认

日期：2026-07-21

状态：已完成 / 双固定密钥确认通过 / 只开放PRESENT二轮同预算门

## 1. 研究问题

OP2在seed0单固定密钥下发现：直接预测同一末轮S-box经P-layer后的四位置真实密文parity，
macro AUC为`0.961079724`，显著超过连续mask与标签打乱。OP3只回答该信号能否在第二把独立
固定秘密密钥及全新明文上复现。

任务契约不变：

```text
input = unseen plaintext P
C = PRESENT_K^1(P)
target = XOR of four registered ciphertext output positions
```

`0/1`仍是该明文真实密文的输出值，不是样本类别。

## 2. 唯一变量与冻结预算

唯一变化是确定性实验seed从`0`变为`1`，因此生成一把不同的固定80-bit秘密密钥与一套全新明文。

```text
cipher                 = PRESENT-80
rounds                 = 1
experiment seed        = 1
secret keys            = one fixed unknown seed1 key
train/validation/test  = 4096 / 1024 / 2048 plaintexts
cross-run plaintexts   = disjoint from OP2 seed0
input                   = 64 LSB-first plaintext bits only
hidden width            = 128
epochs                  = 5
batch size              = 128
optimizer               = AdamW, lr=1e-3, weight_decay=1e-4
device                  = local CPU
```

模型矩阵、mask定义、配对初始化与OP2相同：完整64-bit输出、连续parity、对齐parity、对齐训练标签
打乱。不能调整网络、样本数、epoch、指标、阈值或选择最好mask。

## 3. 独立性与联合门

执行前额外验证：

```text
OP2 seed0 anchor gate = pass / mask_geometry_supported
seed1 != seed0
seed1 secret key != seed0 secret key
两次运行的全部明文零重合
seed1本身通过OP2全部输出预测与mask配对协议检查
```

seed1仍使用OP2三项门：

```text
aligned macro AUC                         >= 0.55
aligned macro AUC - shuffled macro AUC   >= +0.03
aligned macro AUC - contiguous macro AUC >= +0.03
```

只有seed0、seed1均通过，联合裁决才为：

```text
innovation2_output_parity_mask_geometry_two_key_confirmed
```

若seed1协议有效但任一性能门失败，则裁决`two_key_not_confirmed`并停止轮数扩展；若anchor、密钥、
明文独立性或输出预测协议无效，则只修协议，不解释神经性能。

## 4. 执行与下一步

```text
run_id = i2_output_parity_prediction_op3_independent_key_present_r1_seed1_20260721
anchor = outputs/local_readiness/i2_output_parity_prediction_op2_mask_geometry_present_r1_seed0_20260721/
output = outputs/local_readiness/i2_output_parity_prediction_op3_independent_key_present_r1_seed1_20260721/
```

本地CPU执行，生成与OP2相同的JSONL/CSV/SVG/门控/元数据/数组产物，图经`visual-qa-redraw`
像素检查并刷新最近结果索引。

若双密钥确认通过，下一步OP4只将轮数从`r1`改为`r2`，以seed0/seed1同预算复核信号衰减；不先
增加样本、epoch、模型容量、密钥数或远程GPU规模。若未确认，则停止轮数梯并审计固定密钥输出预测
文献协议。无论结果如何，本门仍不是高轮攻击、论文复现或SOTA证据。

## 5. 正式结果

OP3按冻结协议执行完成。seed0 anchor、seed1、两把秘密密钥不同以及两次运行全部明文零重合等
7项联合检查全部通过：

```text
status                                    = pass
decision                                  = innovation2_output_parity_mask_geometry_two_key_confirmed

seed0 aligned parity macro AUC            = 0.961079724
seed0 contiguous parity macro AUC         = 0.494962902
seed0 aligned label-shuffle macro AUC     = 0.514765490
seed0 aligned - contiguous macro AUC      = +0.466116822
seed0 aligned - shuffled macro AUC        = +0.446314233

seed1 aligned parity accuracy             = 0.872650146
seed1 aligned parity macro AUC            = 0.963068370
seed1 contiguous parity macro AUC         = 0.501500846
seed1 aligned label-shuffle macro AUC     = 0.491663568
seed1 aligned - contiguous macro AUC      = +0.461567524
seed1 aligned - shuffled macro AUC        = +0.471404802

two-key minimum aligned macro AUC         = 0.961079724
two-key mean aligned macro AUC            = 0.962074047
two-key aligned macro AUC range           = 0.001988646
```

第二把固定密钥下，对齐真实密文parity仍达到`0.9631` AUC，连续mask和训练标签打乱仍在随机附近；
两把密钥的对齐AUC极差仅`0.0020`。因此OP2的mask几何信号获得一次严格独立密钥确认，当前结果
不是某一把密钥或某一批明文的偶然现象。

裁决只开放OP4：保持两把固定密钥、两组mask、模型、4096/1024/2048条split、5 epochs和三项
门槛不变，唯一将PRESENT轮数从`r1`改为`r2`，量化结构对齐parity信号的首步轮数衰减。不得先
增加样本、epoch、容量、密钥数量或远程GPU规模。即使双密钥AUC很高，它仍是一轮局部S-box输出
函数的本地小规模证据，不能称为高轮攻击或与主流区分轮数直接比较。

`curves.svg`已由`visual-qa-redraw`渲染为`1800 x 984`像素检查：双面板、标题、输出预测语义、
三种方法、随机基线、数值标签、跨密钥汇总、裁决和证据边界均无文字重叠、裁切、缺字或结构
歧义；验收记录为`visual_qa_passed.marker`。

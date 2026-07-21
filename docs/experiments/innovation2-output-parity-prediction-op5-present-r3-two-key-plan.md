# 创新2 OP5：PRESENT三轮结构对齐输出parity双固定密钥轮数步进

日期：2026-07-21

状态：已完成 / 三轮双固定密钥门未通过 / 停止机械扩轮并转SPN局部表示重设计

## 1. 唯一研究问题

OP4在PRESENT二轮、两把独立固定密钥下得到稳定的结构对齐真实密文输出parity信号，双密钥平均
macro AUC为`0.626534744`。OP5只回答：加到三轮后，同一输出预测任务是否仍超过连续mask与标签
打乱控制。

```text
input = unseen plaintext P
C = PRESENT_K^3(P)
target_m = XOR_{j in aligned_mask_m} C[j]
```

目标仍是每条明文真实三轮密文的输出parity，不是样本类别。

## 2. 唯一变量与冻结项

唯一改变：

```text
rounds = 2 -> 3
```

冻结OP4全部协议：seed0/seed1各自固定秘密密钥与独立模型、每密钥4096/1024/2048条互斥明文、
64-bit明文输入、真实64-bit密文输出、16个连续四位parity、16个末轮S-box/P-layer对齐parity、
完整/连续/对齐/对齐标签打乱四行两层MLP、hidden 128、5 epochs、batch 128、AdamW及相同初始化
配对。不能更换mask、样本、epoch、容量、指标或选择最好输出位置。

## 3. 门控

每把密钥沿用三项门：

```text
aligned macro AUC                         >= 0.55
aligned macro AUC - shuffled macro AUC   >= +0.03
aligned macro AUC - contiguous macro AUC >= +0.03
```

seed0与seed1均通过且联合独立性检查有效，才裁决：

```text
innovation2_output_parity_present_r3_two_key_supported
```

任一密钥未通过则裁决`r3_two_key_not_supported`；仍完整执行两颗seed，以区分稳定衰减与单seed波动。
协议无效时只修协议。

## 4. 执行与下一步

```text
seed0 run = i2_output_parity_prediction_op5_present_r3_seed0_20260721
joint run = i2_output_parity_prediction_op5_present_r3_seed1_joint_20260721
output    = outputs/local_readiness/<run_id>/
```

若双密钥通过，OP6只将轮数从`r3`改为`r4`，继续同预算衰减门。若未通过，停止机械增加轮数、样本、
epoch、容量、密钥数和远程规模；下一步在三轮或最后通过的二轮上，保持真实密文输出标签不变，比较
SPN局部连接表示与当前全连接MLP。最终双密钥图经`visual-qa-redraw`检查并刷新最近结果索引。

三轮本地小规模结果不是高轮攻击、论文复现、SOTA或创新2最终上限证据。

## 5. 正式结果

seed0与seed1均完成，输出预测、轮数、anchor、密钥与跨运行明文独立性协议有效。两颗seed均有
同向弱残差，但均未通过冻结门：

```text
status                                  = hold
decision                                = innovation2_output_parity_present_r3_two_key_not_supported

seed0 full-output bit macro AUC         = 0.502689235
seed0 aligned parity accuracy           = 0.517669678
seed0 aligned parity macro AUC          = 0.527433636
seed0 contiguous parity macro AUC       = 0.501606262
seed0 aligned label-shuffle macro AUC   = 0.497583519
seed0 aligned - contiguous macro AUC    = +0.025827375
seed0 aligned - shuffled macro AUC      = +0.029850118

seed1 full-output bit macro AUC         = 0.505919691
seed1 full-derived aligned parity AUC   = 0.493682161
seed1 aligned parity accuracy           = 0.513244629
seed1 aligned parity macro AUC          = 0.522831564
seed1 contiguous parity macro AUC       = 0.498815204
seed1 aligned label-shuffle macro AUC   = 0.498945845
seed1 aligned - contiguous macro AUC    = +0.024016359
seed1 aligned - shuffled macro AUC      = +0.023885719

two-key minimum aligned macro AUC       = 0.522831564
two-key mean aligned macro AUC          = 0.525132600
two-key aligned macro AUC range         = 0.004602073
```

三轮结果不是完全没有信号：两颗seed的对齐parity均高于连续与打乱控制，且方向一致。但绝对AUC
`0.523--0.527`低于`0.55`，归因差值`+0.024--+0.030`也未稳定达到`+0.03`。因此不能把弱残差
表述为三轮预测成功，也不能据此继续机械运行四轮。

轮数衰减轨迹为：

```text
r1 two-key mean aligned AUC = 0.962074047
r2 two-key mean aligned AUC = 0.626534744
r3 two-key mean aligned AUC = 0.525132600  (gate miss)
```

下一步不改成二分类、不改标签、不加样本或epoch。推荐在三轮与最后通过的二轮上，保持同一真实
密文对齐parity输出目标，比较当前全连接MLP与按PRESENT逆向依赖锥组织的SPN局部连接网络；必须
配同参数量/同预算MLP、错误P层拓扑和标签打乱控制。只有本地三轮恢复到`>=0.55`且相对两类控制
均`>=+0.03`，才重新开放后续轮数梯。

双密钥`curves.svg`经`visual-qa-redraw`首次检查发现裁决文字仍沿用旧的论文协议审计路径；修正为
SPN局部表示重设计后重新渲染`1800 x 984`像素。最终标题、双面板、三组近随机柱值、基线、汇总、
裁决和证据边界无重叠、裁切、缺字或语义不一致；验收记录为`visual_qa_passed.marker`。

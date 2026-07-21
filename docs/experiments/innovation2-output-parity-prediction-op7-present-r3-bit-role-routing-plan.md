# 创新2 OP7：PRESENT三轮真实输出parity的精确bit-role路由门

日期：2026-07-21

状态：已完成 / 精确bit-role路由未通过 / 转确定性依赖锥与布尔函数难度审计

## 1. 唯一研究问题

OP6证明粗粒度nibble邻接平均不能恢复三轮真实密文输出parity。OP7只改变局部网络的路由精度：
保留每个PRESENT bit在S-box内的输入/输出角色，并按公开P-layer逐bit置换，检验精确路由是否比
MLP、错误位级拓扑和标签打乱更有效。

```text
input = unseen plaintext P
C = PRESENT_K^3(P)
target_m = XOR_{j in aligned_mask_m} C[j]
```

标签仍是每条明文真实三轮密文的16个输出parity，不是样本类别。

## 2. 冻结协议

完全复用OP5/OP6 seed0：PRESENT-80三轮、一把固定未知密钥、4096/1024/2048条互斥明文、
64-bit明文输入、16个S-box/P-layer对齐真实密文parity、5 epochs、batch 128、AdamW、final epoch
与同一测试集。不能改变数据、密钥、mask、标签、损失、指标或训练预算。

## 3. 精确bit-role候选

```text
64 plaintext bits
-> shared 1-to-13 channel projection + learned 64-position embedding
-> stage0 nibble/key-position embedding
-> shared 4-bit-role S-box block: LayerNorm + 52->104->52 residual MLP
-> exact 64-bit PRESENT P-layer permutation
-> stage1 nibble/key-position embedding
-> shared 4-bit-role S-box block
-> exact 64-bit PRESENT P-layer permutation
-> per-nibble nonlinear 52->64->1 parity head + learned nibble bias
-> 16 LSB-first aligned parity logits
```

两层S-box/P路由构造第三轮S-box之前的表示，非线性head预测第三轮S-box输出四位parity。stage/nibble
embedding只提供固定位置上下文以吸收未知固定轮密钥差异，不输入真实密钥或中间状态。

错误拓扑为同形合法置换：

```text
wrong_p(i) = (8*i) mod 63, i < 63
wrong_p(63) = 63
```

它与PRESENT真实`p(i)=16*i mod 63`不同，但仍是64-bit双射，不改变参数量或计算深度。

## 4. 同预算矩阵与门

```text
aligned_parity_mlp                    26896参数锚点
bit_role_true_p                       精确真实P层候选
bit_role_wrong_p                      精确错误P层同参数控制
bit_role_true_p_label_shuffle         只打乱训练标签
```

候选与MLP参数差必须`<=1%`；true/wrong除拓扑buffer外参数初值逐项相同。四项性能门不变：

```text
true-P macro AUC                     >= 0.55
true-P macro AUC - MLP macro AUC    >= +0.03
true-P macro AUC - wrong-P AUC      >= +0.03
true-P macro AUC - label-shuffle AUC>= +0.03
```

另需手工fixture证明真实/错误P均为双射、真实路由逐bit等于`Present80.permutation_layer`、S-box
bit-role reshape无换序、输出仍为16个LSB-first真实parity。

## 5. 执行与下一步

```text
run_id = i2_output_parity_prediction_op7_present_r3_bit_role_routing_seed0_20260721
output = outputs/local_readiness/i2_output_parity_prediction_op7_present_r3_bit_role_routing_seed0_20260721/
```

本地CPU执行并生成完整结果与可视化，图经`visual-qa-redraw`检查。

若四项门全部通过，下一步只做独立固定密钥seed1复验；确认后才重开四轮。若只胜MLP而不胜错误P，
保留通用bit-role归纳偏置证据但不主张PRESENT拓扑收益；若仍不胜MLP，则停止当前神经路由设计，
转确定性依赖锥/布尔函数难度审计，决定三轮弱残差是否值得扩大本地样本，而不是直接远程或扩轮。

OP7不是高轮攻击、论文复现、SOTA或创新2最终上限。

## 6. 正式结果

16项真实输出、三轮、bit路由、双射控制、参数配平、初始化和训练检查全部通过。候选`27003`
参数，MLP`26896`参数，差异`0.398%`；true/wrong除固定置换buffer外初值逐参数相同。

```text
status                            = hold
decision                          = innovation2_output_parity_present_r3_bit_role_not_ready

MLP accuracy                      = 0.517547607
MLP macro AUC                     = 0.526979820
bit-role true-P accuracy          = 0.509582520
bit-role true-P macro AUC         = 0.518033257
bit-role wrong-P accuracy         = 0.504150391
bit-role wrong-P macro AUC        = 0.513248717
bit-role label-shuffle accuracy   = 0.498565674
bit-role label-shuffle macro AUC  = 0.496554813

true-P - MLP macro AUC            = -0.008946563
true-P - wrong-P macro AUC        = +0.004784540
true-P - label-shuffle macro AUC  = +0.021478444
```

精确bit-role路由比OP6粗粒度nibble网络的`0.5136`略升至`0.5180`，但仍低于MLP的`0.5270`；
真实P相对错误P仅`+0.0048`，相对标签打乱仅`+0.0215`。因此位角色保留没有形成可归因的
PRESENT拓扑收益，也不开放seed1、四轮、扩样本或远程训练。

OP6和OP7共同说明：三轮弱残差不是通过“更像密码轮函数”的小网络就能直接放大。当前更需要先
回答目标函数本身有多难：每个对齐parity依赖多少明文bit、依赖锥是否已覆盖全部64位、精确代数
次数/单项式增长如何、不同mask难度是否一致，以及4096条样本相对有效真值空间有多稀疏。

下一步冻结为无训练确定性审计，不再换神经架构：逐轮传播16个aligned mask的精确bit依赖集合，
用PRESENT S-box ANF传播代数次数上界与可行的稀疏单项式统计，并对r1/r2/r3形成同一张难度曲线。
该审计决定是开放三轮本地数据斜率，还是认定当前固定密钥明文到parity函数在小预算下不可学习。

`curves.svg`经`visual-qa-redraw`渲染为`1800 x 965`像素检查：精确bit-role/错误P标签、四组近随机
柱值、局部纵轴、随机基线、差值、裁决和证据边界均无重叠、裁切、缺字或任务歧义；验收记录为
`visual_qa_passed.marker`。

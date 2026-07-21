# 创新2 OP8：PRESENT r1--r3真实输出parity精确ANF难度审计

日期：2026-07-21

状态：已完成 / hold（冻结硬上限边界）

## 1. 研究问题

输出预测主线在同预算下出现稳定衰减：r1双密钥平均AUC`0.9621`、r2`0.6265`、r3`0.5251`；
两种SPN网络也未恢复三轮。OP8不训练模型，只回答同一16个结构对齐真实密文parity函数从r1到r3
发生了什么确定性复杂度跃迁，以及4096条训练明文是否已从充分覆盖变为极稀疏覆盖。

## 2. 冻结函数契约

对每个轮数`r in {1,2,3}`与每个末轮S-box索引`m in {0..15}`：

```text
C = PRESENT_K^r(P)
mask_m = Present80.permutation_layer(0xF << (4*m))
f_{r,m}(P) = XOR_{j in mask_m} C[j]
```

使用OP1--OP7 seed0的同一把固定80-bit秘密密钥。密码轮密钥按标量`Present80._update_key`确定性
展开，但网络不可见的密钥在本审计中只作为布尔常数；多项式变量仅为64个明文bit。

## 3. 精确审计

复用E55经过测试的`CappedPolynomialOps`与`required_state_cone`，对全部48个函数精确计算GF(2)
ANF，不使用次数上界替代：

```text
structural dependency cone width by round
exact surviving ANF monomial count after GF(2) cancellation
exact algebraic degree
exact functional variable support from surviving monomials
constant-term presence
training rows / 2^support coverage
training rows / monomial count ratio
```

每个函数使用独立硬上限：

```text
maximum terms   = 500000
maximum seconds = 10
maximum memory  = 1 GiB
```

任一函数超过硬上限时记录`cap_exceeded`，不得提高上限后选择性补跑。每个精确多项式至少在三个确定性
明文上求值，并逐项等于标量PRESENT的同一真实密文parity。

## 4. 冻结神经来源

只读取并验证：

```text
r1 = OP3 two-key confirmed gate
r2 = OP4 two-key supported gate
r3 = OP5 two-key not-supported gate
```

使用各门的双密钥平均aligned parity macro AUC，不重训、不重选mask、不使用OP6/OP7候选指标替换
基础轮数曲线。

## 5. 裁决

`difficulty transition confirmed`要求：

```text
48个函数全部在硬上限内完成
全部确定性赋值与标量PRESENT一致
每轮16个函数的functional support一致
r1/r2/r3 median support严格增加
r1/r2/r3 median exact degree严格增加
r1/r2/r3 median exact monomial count严格增加
r3所有函数support = 64
r3 median monomials > 4096 train rows
r1/r2/r3双密钥平均AUC严格下降
```

通过只说明三轮难度跃迁与数据稀疏是合理机制证据，不证明增加样本一定成功。

## 6. 执行与下一步

```text
run_id = i2_output_parity_prediction_op8_present_r1_r3_exact_anf_difficulty_20260721
output = outputs/local_audits/i2_output_parity_prediction_op8_present_r1_r3_exact_anf_difficulty_20260721/
```

本地CPU执行，生成48行`results.jsonl`、`summary.json`、`gate.json`、`round_summary.csv`、
`progress.jsonl`与中文`curves.svg`，图经`visual-qa-redraw`检查。

若难度跃迁确认，下一步OP9只改变训练行数，使用固定seed0密钥、三轮、同一MLP/真实输出parity、
固定验证测试集和嵌套训练前缀`4096 -> 8192 -> 16384`，配标签打乱；只有AUC与相对控制呈预注册
正斜率才考虑更大本地门。若审计不支持数据稀疏机制或未能在冻结硬上限内完成全部函数，则不开放
该数据斜率，停止该输出mask/固定密钥路线的机械扩样本、扩轮或远程规模。

OP8是确定性函数难度审计，不是训练结果、攻击轮数、论文复现或SOTA。

## 7. 完成结果

冻结执行完成48个目标函数中的33个：

```text
PRESENT r1: 16/16完成，support=4，degree中位数=3，单项式中位数=6
PRESENT r2: 16/16完成，support=16，degree中位数=9，单项式中位数=896
PRESENT r3:  1/16完成，15/16触及500000项硬上限
            唯一完成函数support=64，degree=12，单项式数=47565
```

全部33个已完成函数均通过三次确定性明文赋值与标量PRESENT真实密文parity逐项重放。冻结神经
来源仍是同一输出预测任务，两把独立固定密钥的平均AUC依次为：

```text
r1 = 0.962074047
r2 = 0.626534744
r3 = 0.525132600
```

这里的`0/1`是每条明文对应的真实密文输出parity值，不是正负样本类别。AUC只是衡量这个一比特
输出值是否被正确排序的评价指标，不把任务改成`real-vs-random`、平衡/不平衡或关系分类。

最终裁决：

```text
status   = hold
decision = innovation2_output_parity_exact_anf_difficulty_hard_cap_exceeded
```

这不是`protocol invalid`：来源门、bit顺序与标量重放均有效。它也不是完整确认三轮ANF统计，因为
15个三轮函数未在冻结资源上限内完成。按预注册规则，不提高50万项上限，不开放训练数据斜率，
不扩到四轮，不启动远程规模。

证据目录：

```text
outputs/local_audits/i2_output_parity_prediction_op8_present_r1_r3_exact_anf_difficulty_20260721/
```

图像`curves.svg`已按`visual-qa-redraw`渲染为`1800 x 1202`像素检查；最终版本无文字重叠、
裁切、缺字、图例冲突或“最终项数/执行硬上限”语义混淆。

## 8. 推荐下一步

下一步不做OP9数据扩容，先做“固定密钥真实输出预测论文协议对齐审计”。要核对的唯一关键差异是：
Kimura及后续输出预测论文的训练输入、输出编码、固定/变化密钥、预测单位和评价规则，是否与当前冻结契约
`P -> E_K^r(P)`的真实输出bit/parity逐样本预测一致。审计锚点是当前OP1完整64-bit输出基线与OP2
aligned parity，不引入真假、平衡或关系标签，也不训练新网络。

该审计只解锁一个决定：若论文协议与当前任务一致，按论文表示建立一个同预算、同数据拆分的忠实基线，
再只改变输出表示；若不一致，则保留当前任务为独立主线，重新预注册真实密文输出函数族，不把历史积分、
kernel或ATM分类旁支包装成输出预测。审计完成前禁止机械增加样本、epoch、轮数、模型或远程GPU规模。

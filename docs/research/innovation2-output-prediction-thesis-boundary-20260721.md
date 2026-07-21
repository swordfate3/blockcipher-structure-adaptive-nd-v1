# 创新2真实输出值预测：论文结论与扩轮边界

日期：2026-07-21

状态：OP9--OPA2完成 / 三轮SPN-aware架构独立确认 / OPA3拓扑归因运行中 / OPA4条件式预注册

## 0. 2026-07-22路线更新

本文原先在OP12后关闭的是“四轮结构化XOR + MLP”路线，不是所有四轮真实输出值函数或所有网络
架构。随后在不使用OP12四轮测试结果选模型的前提下，OPA1对第三固定密钥、相同八个预注册三轮
真实输出bit进行了五模型同预算发现屏；OPA2再用第四固定密钥和架构匹配标签打乱独立确认候选。

新增正式证据为：

| 阶段 | 协议 | 结果 | 状态 |
|---|---|---|---|
| OPA1 | seed2，八输出，MLP/LSTM/ResCNN/Transformer/PRESENT-SPN-aware，约390万参数，`2^17/2^16`，100 epochs | MLP `0.531657`，ResCNN `0.588388`，PRESENT-SPN-aware `1.000000` | fallback-retrieved / validated发现证据 |
| OPA2 | seed3，MLP与PRESENT-SPN-aware的true/shuffle四行匹配控制，同预算 | MLP true `0.532262`；SPN true `1.000000`、shuffle `0.500840`；调整增益`+0.465752` | verified result branch / pass |
| OPA3 | seed3，exact-P / identity-P / fixed-wrong-P同参数归因 | 正式远程运行中 | 尚无测试结果，不得预判 |

OPA2把当前最强结论从“专用八输出MLP略优于完整头”推进为“PRESENT-SPN-aware在第三和第四固定
密钥上均明显优于同预算MLP，并在第四密钥上显著超过自身匹配标签打乱”。但OPA2只确认整体架构，
不能证明收益来自精确PRESENT P-layer；该因果主张必须等待OPA3。

条件式OPA4保持八输出任务、第四密钥、明文split、模型和训练预算，只把`rounds=3`改为`4`。它只有
在OPA3正式通过、结果来源验证且图像质检完成后才允许实现。该路线不重新打开OP12结构化XOR，不按
四轮结果后验挑bit，也不允许直接跳到五轮。

## 1. 当前唯一任务定义

创新2当前主任务是固定未知秘密密钥下的真实密码输出值预测：

```text
固定未知秘密密钥 K
输入 = 训练中未见过的明文 P
真实输出 = C = PRESENT_K^r(P)
标签 = C的完整64 bit、预注册单bit，或预注册多个bit的XOR值
```

一个parity标签虽然是`0/1`，它仍是同一条明文真实密文的确定函数值，不是正负样本类别。真假样本
分类、积分平衡分类、kernel判断、cube性质判断和关系成员判断均不是本路线结果。

## 2. OP9--OP12回答了什么

| 阶段 | 唯一问题 | 正式协议 | 结果 |
|---|---|---|---|
| OP9 | Kimura式完整64-bit输出头能否在PRESENT三轮恢复论文信号 | 单固定密钥，`2^17`训练、`2^16`测试、100 epochs | LSTM macro AUC `0.500008`、完整命中`0`，单密钥论文校准不支持 |
| OP10 | 完整输出失败时，是否仍有个别真实密文bit容易预测 | seed0固定密钥，发现集与fresh确认集严格分离 | 位置`0,2,8,10,32,34,40,42`共`8/8`在fresh明文上确认 |
| OP11 | 同八个位置能否跨第二把固定密钥复现，且专用头是否优于完整64输出头 | seed1独立密钥，`2^17/2^16`、100 epochs、匹配shuffle | `8/8`确认；专用头均值AUC `0.530900`，比完整头高`0.008994`，比shuffle高`0.030112` |
| OP12 | 将这些位置组成结构化双bit/四bit XOR，能否把预测推进到PRESENT四轮 | seed1固定密钥，同预算四行矩阵，六个预注册mask与四类强基线 | 平均AUC `0.499943`，`0/6` mask通过，扩轮不支持 |
| OPA1 | 除MLP/LSTM外，位置保持或SPN-aware架构能否更好预测同八个三轮输出bit | seed2固定密钥，五模型参数匹配发现屏 | ResCNN `0.588388`；PRESENT-SPN-aware `1.000000`，进入独立确认 |
| OPA2 | OPA1候选能否在第四密钥同时超过MLP和架构匹配shuffle | seed3固定密钥，MLP/SPN true/shuffle四行矩阵 | SPN true `1.000000`，比MLP高`0.467738`，调整增益`+0.465752`，通过 |

OP9--OP12先形成原始可审计链：按论文协议校准完整输出，从共享64位置扫描中发现易预测位置，用全新
明文确认、换独立秘密密钥复现，再用结构化XOR检验能否增加轮数。OPA1/OPA2随后在不使用OP12四轮
测试结果选模型的前提下，追加同八位置的多架构发现与第四密钥匹配控制确认。两段流程均把候选发现与
确认分离，并为方法主张提供同预算标签打乱或结构控制，不是从最终测试集后验挑最高bit。

## 3. 当前最强正结果

创新2当前最强结果不是完整密文恢复，而是：

> 对PRESENT-80三轮，在第一把固定秘密密钥上发现并用全新明文确认八个易预测真实密文输出bit，
> 再将位置冻结并跨第二、第三和第四固定密钥继续评估。第三密钥的五模型屏中，位置保持ResCNN平均
> AUC为`0.588388`，PRESENT-SPN-aware为`1.000000`；第四密钥的匹配控制确认中，SPN-aware仍为
> `1.000000`，MLP为`0.532262`、SPN标签打乱为`0.500840`，调整后架构增益为`+0.465752`。

这支持两个递进结论：完整输出的多任务干扰会掩盖局部可预测坐标，先发现并冻结易预测位置、再使用
专用小输出头可以恢复三轮真实输出值信号；在同一八输出契约下，保留bit位置并显式模拟SPN局部混合
与P-layer传播的结构网络又显著优于参数匹配MLP。后一结论目前只到整体架构层，精确拓扑归因等待
OPA3，不要求64个bit同时命中，也没有把二分类区分指标冒充输出恢复。

权威证据：

```text
docs/experiments/innovation2-output-prediction-op10-present-r3-easy-bit-discovery-plan.md
docs/experiments/innovation2-output-prediction-op11-present-r3-selected8-independent-key-plan.md
docs/experiments/innovation2-output-prediction-opa1-present-r3-selected8-architecture-screen-plan.md
docs/experiments/innovation2-output-prediction-opa2-conditional-architecture-confirmation-plan.md
outputs/remote_results/i2_output_prediction_op10_present_r3_easy_bit_confirm_gpu0_20260721/
outputs/remote_results/i2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721/
outputs/remote_results_incomplete/i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721_raw_fallback/
outputs/remote_results/i2_output_prediction_opa2_present_r3_selected8_present_spn_key3_gpu0_20260722/
```

## 4. 四轮多bit XOR边界

OP12直接预测四个同末轮S-box双bit XOR和两个同角色四bit XOR，并同时比较：

```text
同重量几何控制
同架构训练标签打乱
从八个单bit概率派生的parity
mask内最佳组成bit
```

正式结果为：

```text
mean direct structured XOR AUC = 0.499943121
mean geometry-control AUC      = 0.500640653
mean matched-shuffle AUC       = 0.500612424
mean derived-parity AUC        = 0.498950059
mean best-component AUC        = 0.506514660
passed masks                   = 0 / 6
```

因此当前证据不支持“对多个易预测密文bit做XOR可以把真实输出预测由三轮提升到四轮”。这不是一般数学
不可能性证明，但在预注册mask、与OP11相同的`2^17/2^16`数据预算、100 epochs、匹配控制和单固定
秘密密钥下，结果已接近
随机且未超过更简单基线。按冻结门不启动OP13、不进入五轮，也不通过后验换mask、加数据、加epoch或
换大模型继续搜索。

权威证据：

```text
docs/experiments/innovation2-output-prediction-op12-present-r4-structured-xor-plan.md
outputs/remote_results/i2_output_prediction_op12_present_r4_structured_xor_key1_gpu0_20260721/
```

## 5. 论文可写与不可写

可以写：

1. 提出“易预测输出坐标发现 + fresh确认 + 独立密钥复现 + 专用多输出头”的真实输出值预测流程；
2. 在PRESENT三轮的两把独立固定秘密密钥上确认同八个真实密文bit；
3. 证明专用八输出头优于同位置完整64输出anchor和同架构shuffle；
4. 用预注册结构化XOR和四类控制给出四轮扩展的清晰负边界。
5. 在第三密钥发现ResCNN和PRESENT-SPN-aware优于MLP，并在第四密钥用匹配shuffle独立确认
   PRESENT-SPN-aware整体架构优势。

不能写：

- 恢复了完整64-bit密文；
- 达到了PRESENT主流七至九轮神经区分或积分攻击轮数；
- 复现了Kimura跨100把密钥的论文结果；
- 多bit XOR普遍优于单bit预测；
- OP12证明所有四轮输出函数均不可预测；
- 三轮AUC可与真假样本区分准确率或SOTA攻击轮数直接比较。
- OPA2已经证明精确P-layer是增益原因，或已经得到四轮输出预测结果。

## 6. 论文建议结构

```text
4.1 固定密钥真实输出值预测任务与威胁模型
4.2 Kimura完整输出协议校准及完整输出失败分析
4.3 易预测输出坐标的发现与全新明文确认
4.4 专用八输出头及独立秘密密钥复现
4.5 五模型同预算筛选与第四密钥匹配控制确认
4.6 PRESENT-SPN-aware结构与P-layer同参数归因
4.7 结构化多bit XOR负边界与条件式八输出四轮扩展
4.8 适用范围、与神经区分/积分预测的区别及停止边界
```

建议核心表格至少包含OP11八个位置的两密钥结果、专用头/完整头/shuffle三方对照、OPA1五模型
同预算结果、OPA2 true/shuffle独立确认，以及OP12六个mask的六门逐项结果。OPA3表格和图只有在
verified result branch回收并完成`visual-qa-redraw`后加入；不得从训练MSE预写测试结论。

## 7. 下一步

OP9--OP12已经整理为可进入论文模板的实验章节初稿：

```text
docs/research/innovation2-output-prediction-thesis-chapter-draft-20260721.md
```

当前唯一开放训练是已经启动的OPA3同参数拓扑归因。OPA3通过后才实现条件式OPA4，在相同八输出
契约下只把轮数从三轮改为四轮；OPA3不通过则立即收束到OPA2整体架构结论。OPA4即使通过也先做
第五固定密钥的四轮独立确认，不直接进入五轮。仅换mask、seed、网络名称或扩大现有预算仍不能绕过
这些冻结门。

SPN-aware方法相对Kimura/Watanabe通用LSTM、Kimura普通Conv1D和Singh去密钥逐轮MLP的暂定创新
边界见：

```text
docs/research/innovation2-spn-aware-output-prediction-novelty-audit-20260722.md
```

该审计只覆盖四篇已验证核心全文，不支持“首次”或穷尽性文献声明；精确拓扑贡献仍以OPA3为必要门。

# 创新2真实输出值预测：论文结论与扩轮边界

日期：2026-07-21

状态：OP9--OP12完成 / 三轮selected-bit方法成立 / 四轮结构化XOR扩轮停止 / 论文实验章节初稿已完成

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

四个阶段形成一条可审计证据链：先按论文协议校准完整输出，再从共享64位置扫描中发现易预测位置，
用全新明文确认，换独立秘密密钥复现，最后用结构化XOR检验能否增加轮数。候选发现与确认分离，且
每次方法主张都有同预算标签打乱或结构控制，不是从最终测试集后验挑最高bit。

## 3. 当前最强正结果

创新2当前最强结果不是完整密文恢复，而是：

> 对PRESENT-80三轮，在第一把固定秘密密钥上发现并用全新明文确认八个易预测真实密文输出bit；
> 将同一位置集合冻结后，在第二把独立固定秘密密钥上八个位置全部复现。专用八输出MLP相对完整
> 64输出头平均AUC提高`0.008993575`，相对同架构标签打乱提高`0.030112164`。

这支持一个适合毕业论文的方法结论：完整输出的多任务干扰会掩盖局部可预测坐标，先发现并冻结易预测
位置，再使用专用小输出头，可以稳定恢复三轮真实输出值信号。该方法不要求64个bit同时命中，也没有
把二分类区分指标冒充输出恢复。

权威证据：

```text
docs/experiments/innovation2-output-prediction-op10-present-r3-easy-bit-discovery-plan.md
docs/experiments/innovation2-output-prediction-op11-present-r3-selected8-independent-key-plan.md
outputs/remote_results/i2_output_prediction_op10_present_r3_easy_bit_confirm_gpu0_20260721/
outputs/remote_results/i2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721/
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

不能写：

- 恢复了完整64-bit密文；
- 达到了PRESENT主流七至九轮神经区分或积分攻击轮数；
- 复现了Kimura跨100把密钥的论文结果；
- 多bit XOR普遍优于单bit预测；
- OP12证明所有四轮输出函数均不可预测；
- 三轮AUC可与真假样本区分准确率或SOTA攻击轮数直接比较。

## 6. 论文建议结构

```text
4.1 固定密钥真实输出值预测任务与威胁模型
4.2 Kimura完整输出协议校准及完整输出失败分析
4.3 易预测输出坐标的发现与全新明文确认
4.4 专用八输出头及独立秘密密钥复现
4.5 结构化多bit XOR四轮扩展实验与强控制
4.6 适用范围、与神经区分/积分预测的区别及停止边界
```

建议核心表格包含OP11八个位置的两密钥结果、专用头/完整头/shuffle三方对照，以及OP12六个mask的
六门逐项结果。图使用已经通过`visual-qa-redraw`的OP11与OP12正式`curves.svg`。

## 7. 下一步

OP9--OP12已经整理为可进入论文模板的实验章节初稿：

```text
docs/research/innovation2-output-prediction-thesis-chapter-draft-20260721.md
```

当前不再为这条输出值预测路线分配远程训练预算。下一步是将初稿并入学校论文模板并统一图号、表号、
参考文献与正式图像；若开题验收要求中间状态逐轮预测，先冻结威胁模型与确定性基线，不从OP12直接
机械扩展。只有独立文献或确定性密码结构分析提出新的、预先限定的四轮输出函数机制，才允许新建实验
计划；仅换mask、seed、网络名称或扩大现有预算不能重新打开路线。

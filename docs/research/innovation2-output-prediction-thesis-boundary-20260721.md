# 创新2真实输出值预测：论文结论与扩轮边界

日期：2026-07-21

状态：OP9--OPB1完成 / OPB1低秩未归因 / OPC1正式远程运行中 / 协议边界已审计 / 四轮关闭

## 权威入口与阅读顺序

截至`2026-07-22`，本文件是创新2当前任务、结果范围和停止边界的唯一权威状态入口。建议按以下顺序
阅读和引用：

1. 本文件：任务契约、正式证据表、可写/不可写结论和下一步；
2. `docs/research/innovation2-output-prediction-thesis-chapter-draft-20260721.md`：可进入论文模板的章节底稿；
3. `docs/research/innovation2-spn-aware-output-prediction-novelty-audit-20260722.md`：核心论文与联网检索限定的新颖性边界；
4. `docs/research/innovation2-opening-proposal-claim-evidence-matrix-20260722.md`：开题承诺逐项覆盖和协议修订；
5. `docs/experiments/innovation2-output-prediction-op9-*.md`至`opa3-*.md`及OPB1计划：逐实验协议、产物与gate；
6. `outputs/00_RECENT_RESULTS.md`：本地可查看的最新结果入口，`001`始终是最新完成结果。

下列文档保留历史价值，但不再定义当前创新2：

| 历史文档 | 历史任务 | 当前用途 |
|---|---|---|
| `innovation2-thesis-ready-conclusion-20260716.md` | PRESENT r5结构条件积分候选排序 | 位置先验负归因与早期评测方法 |
| `innovation2-thesis-consolidation-20260719.md` | PRESENT/GIFT严格平衡谱PG-NBPO | 独立的历史积分性质预测章节 |
| `innovation2-neural-architecture-ranking-20260718.md` | 积分平衡/关系标签上的结构网络排序 | 历史架构探索记录 |
| `innovation2-structure-conditioned-integral-output-prediction-20260715.md` | 结构集合XOR平衡性质预测 | 历史研究蓝图 |

这些历史任务的输入对象、标签、指标和轮数不能与当前“未见明文到真实密文输出值”的结果串成同一条
轮数提升轨迹。

## 0. 2026-07-22路线更新

本文原先在OP12后关闭的是“四轮结构化XOR + MLP”路线，不是所有四轮真实输出值函数或所有网络
架构。随后在不使用OP12四轮测试结果选模型的前提下，OPA1对第三固定密钥、相同八个预注册三轮
真实输出bit进行了五模型同预算发现屏；OPA2再用第四固定密钥和架构匹配标签打乱独立确认候选。

新增正式证据为：

| 阶段 | 协议 | 结果 | 状态 |
|---|---|---|---|
| OPA1 | seed2，八输出，MLP/LSTM/ResCNN/Transformer/PRESENT-SPN-aware，约390万参数，`2^17/2^16`，100 epochs | MLP `0.531657`，ResCNN `0.588388`，PRESENT-SPN-aware `1.000000` | fallback-retrieved / validated发现证据 |
| OPA2 | seed3，MLP与PRESENT-SPN-aware的true/shuffle四行匹配控制，同预算 | MLP true `0.532262`；SPN true `1.000000`、shuffle `0.500840`；调整增益`+0.465752` | verified result branch / pass |
| OPA3 | seed3，exact-P / identity-P / fixed-wrong-P同参数归因 | exact `1.000000`，identity `0.531990`，wrong `1.000000`；exact-wrong `0.000000` | verified result branch / hold |
| OPB1 | seed4，原SPN锚点/低秩拓扑瓶颈exact-P/wrong-P/shuffle四行，同预算 | anchor/exact/wrong均`1.0`，shuffle `0.499477`，exact-wrong `0`，`0/8`归因 | verified result branch / hold / route stopped |
| OPC1 | seed6，ResCNN锚点/SPN-ResCNN exact-P/wrong-P/shuffle四行，同预算 | 本地readiness通过；已从推送提交`286cd0c`在A6000 GPU0启动`2^17/2^16`、100 epochs正式矩阵 | remote running / result pending |
| OPM1 | 全部64个三轮输出bit的反向锥宽与四个S-box坐标布尔复杂度 | 全部输出锥宽`4 -> 16 -> 64`；selected坐标`1/3`均为degree 3、nonlinearity 4，不是唯一最低degree | local deterministic audit / pass |
| OPK1 | 1024个共享明文，256参考密钥与256零重合评估密钥，参考密钥逐明文频率预测未见密钥 | 平均AUC `0.500544`，方向化AUC均值`0.501220`、最大`0.502176`，accuracy-majority `-0.000466` | local deterministic audit / pass |
| 轮间状态审计 | 16把真实PRESENT-80主密钥、31轮、每轮1个校准转移和256个未见转移 | `496/496`子密钥、`126976/126976`下一状态、`4096/4096`完整加密精确恢复 | local deterministic audit / pass |
| 轮间八输出审计 | 完整当前状态、冻结八个下一状态bit、16把密钥、31轮、最多16个校准对 | 四个相关key nibble `1984/1984`唯一精确恢复；平均`3.1154`、最多11个校准对；未见八bit `126976/126976`精确预测 | local deterministic audit / pass |

OPA2把当前最强结论从“专用八输出MLP略优于完整头”推进为“PRESENT-SPN-aware在第三和第四固定
密钥上均明显优于同预算MLP，并在第四密钥上显著超过自身匹配标签打乱”。OPA3进一步证明identity
控制明显较弱，却没有证明精确PRESENT连线优于固定错误双射。当前只能把收益解释为位置保持、局部
4-bit混合和分层跨nibble扩散组成的整体架构效应，不能归因于精确P-layer。

条件式OPA4和OPA5都要求OPA3正式通过。OPA3为`hold`后，两项计划均已关闭且从未实现或启动；当前
没有四轮八输出SPN式网络结果，也不允许直接跳到五轮。

OPB1不是把OPA4/OPA5改名重跑。它来自一个新的方法级假设：旧网络的每位置189维自由嵌入可能使
任意快速混合双射都饱和，因此用每轮64个标量乘共享方向的低秩固定密钥条件建立结构瓶颈，并在全新
seed4同时比较原锚点、exact-P、wrong-P和匹配shuffle。正式结果中低秩exact-P和wrong-P仍同为
`1.0`，exact-wrong为`0`，因此低秩约束没有解决拓扑不可归因问题；该路线已经停止，不做seed5确认
或容量修复，也不开放四轮。

为落实“不能只使用LSTM/MLP”的模型优化要求，OPC1已把OPA1中最强的非异常饱和模型ResCNN作为
锚点，完成三阶段`3+3+4`残差块之间插入固定P-layer重排的SPN-ResCNN混合实现门。候选、普通
ResCNN、错误P和标签打乱四行参数/预算匹配；本地readiness不作性能判断。正式模式被代码硬门控，
OPB1现已有效裁决为低秩瓶颈未归因，因此OPC1已由冻结gate正式授权，并从推送提交`286cd0c`在
A6000 GPU0启动seed6同预算远程矩阵；本地tmux watcher已经接管完成检测和结果回收。它不是与
OPB1并行选优，正式测试AUC和裁决仍待回收。

OPK1进一步限定了“泛化”用语。即使参考与评估密钥使用完全相同的1024个明文，只用参考密钥逐明文
频率预测256把未见密钥时，八bit平均AUC仍为`0.500544`。因此当前跨密钥证据是“冻结位置与协议在
多把独立固定密钥上分别训练后重复成立”，不是一个不看密钥、没有support set的模型对新密钥零适配
泛化。更强主张必须改为support-set条件预测或密钥输入任务，并重新预注册。

OPM1进一步限定了“易预测位置”的机制用语。全部64个三轮输出bit都具有相同的64-bit输入依赖锥；
八个冻结位置经最后一轮inverse P-layer回溯后对应的S-box输出坐标`1/3`均为degree 3，而PRESENT还
存在degree 2坐标，四个坐标的
nonlinearity又同为4。因此当前不能把易预测性简单归因于更窄依赖锥或单S-box坐标明显更简单。更细的
固定密钥函数谱与训练动力学仍未解释；OPC1保持不变，不因该审计后验换位或增加远程实验。

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
| OPA3 | 精确PRESENT连线是否优于identity与固定错误双射 | seed3固定密钥，exact/identity/wrong-P同参数三行矩阵 | exact与wrong均`1.000000`，identity `0.531990`；`0/8`位置满足exact-wrong门，hold |

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
与跨nibble传播的结构网络又显著优于参数匹配MLP。OPA3把该结论限定在整体架构和分层扩散层：精确
PRESENT连线与固定错误双射同分，因此不能声称精确拓扑贡献。不要求64个bit同时命中，也没有把
二分类区分指标冒充输出恢复。

权威证据：

```text
docs/experiments/innovation2-output-prediction-op10-present-r3-easy-bit-discovery-plan.md
docs/experiments/innovation2-output-prediction-op11-present-r3-selected8-independent-key-plan.md
docs/experiments/innovation2-output-prediction-opa1-present-r3-selected8-architecture-screen-plan.md
docs/experiments/innovation2-output-prediction-opa2-conditional-architecture-confirmation-plan.md
docs/experiments/innovation2-output-prediction-opa3-present-r3-topology-attribution-plan.md
docs/experiments/innovation2-output-prediction-opm1-present-r3-selected-output-structural-baseline-audit-plan.md
outputs/remote_results/i2_output_prediction_op10_present_r3_easy_bit_confirm_gpu0_20260721/
outputs/remote_results/i2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721/
outputs/remote_results_incomplete/i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721_raw_fallback/
outputs/remote_results/i2_output_prediction_opa2_present_r3_selected8_present_spn_key3_gpu0_20260722/
outputs/remote_results/i2_output_prediction_opa3_present_r3_selected8_topology_attribution_key3_gpu0_20260722/
outputs/local_audits/i2_output_prediction_opm1_present_r3_selected_output_structural_baseline_audit_20260722/
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
6. 用identity与固定错误双射同参数控制证明跨nibble扩散有用，但精确PRESENT连线并非当前性能的
   唯一解释。

不能写：

- 恢复了完整64-bit密文；
- 达到了PRESENT主流七至九轮神经区分或积分攻击轮数；
- 复现了Kimura跨100把密钥的论文结果；
- 多bit XOR普遍优于单bit预测；
- OP12证明所有四轮输出函数均不可预测；
- 三轮AUC可与真假样本区分准确率或SOTA攻击轮数直接比较。
- OPA2已经证明精确P-layer是增益原因，或已经得到四轮输出预测结果。
- OPA3通过了精确P-layer归因门，或OPA4/OPA5已经运行。

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

OPB1已经完成并确认低秩瓶颈仍不能使真实P超过错误P；该路线停止。当前唯一开放训练OPC1已在远程
运行：在第七固定密钥、相同`2^17/2^16`总数据与100 epochs下，检验真实P的SPN-ResCNN混合是否
同时超过普通ResCNN、错误P和匹配shuffle。OPC1通过只开放新密钥确认，失败则保留ResCNN发现锚点
并停止混合路线；任何分支都不得通过后验换bit、增加深度、数据或epoch绕过门。

开题提出的完整`state_r -> state_(r+1)`临界轮路线已经完成确定性审计。PRESENT完整状态一对即可
恢复当轮子密钥，全部31轮未见转移均100%预测，因此该协议不能定义扩散到随机猜测的临界轮。当前
保留的主任务仍是未见明文到多轮真实密文输出值；轮间动态若继续，必须先改为部分可见或严格跨密钥
协议并重新通过可识别性门。

把下一状态缩减为当前主线冻结的八个bit仍不能修复该问题。在完整当前状态可见时，这八个bit只依赖
轮密钥nibble`[15,13,7,5]`；正式审计中`1984/1984`个nibble均在最多11个、平均约3.12个校准对内
唯一恢复，随后`126976/126976`个未见八bit向量精确预测。因此“完整当前状态 -> selected8下一状态”
同样是部分子密钥恢复协议，不再训练对应神经网络。该否决不扩张到当前看不到内部状态的端到端任务。

SPN-aware方法相对Kimura/Watanabe通用LSTM、Kimura普通Conv1D和Singh去密钥逐轮MLP的暂定创新
边界见：

```text
docs/research/innovation2-spn-aware-output-prediction-novelty-audit-20260722.md
```

该审计只覆盖四篇已验证核心全文，不支持“首次”或穷尽性文献声明；精确拓扑贡献仍以OPA3为必要门。

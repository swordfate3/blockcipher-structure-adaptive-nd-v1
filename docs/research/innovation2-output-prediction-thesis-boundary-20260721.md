# 创新2真实输出值预测：论文结论与扩轮边界

日期：2026-07-21

状态：OP9--OPC1与OPN1完成 / OPD1位置绑定正式矩阵运行中 / 四轮关闭

## 权威入口与阅读顺序

截至`2026-07-22`，本文件是创新2当前任务、结果范围和停止边界的唯一权威状态入口。建议按以下顺序
阅读和引用：

1. 本文件：任务契约、正式证据表、可写/不可写结论和下一步；
2. `docs/research/innovation2-output-prediction-thesis-chapter-draft-20260721.md`：可进入论文模板的章节底稿；
3. `docs/research/innovation2-spn-aware-output-prediction-novelty-audit-20260722.md`：核心论文与联网检索限定的新颖性边界；
4. `docs/research/innovation2-opening-proposal-claim-evidence-matrix-20260722.md`：开题承诺逐项覆盖和协议修订；
5. `docs/experiments/innovation2-output-prediction-op9-*.md`至`opd1-*.md`：逐实验协议、产物与gate；
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
| OPC1 | seed6，ResCNN锚点/SPN-ResCNN exact-P/wrong-P/shuffle四行，同预算 | ResCNN `0.573572`，exact `0.546634`，wrong `0.545181`，shuffle `0.500163`；exact-anchor `-0.026938`，exact-wrong `+0.001453`，`0/8`通过 | verified result branch / hold / route stopped |
| OPN1 | OPC1实际`Flatten + Linear(252*64,8)`输出头，identity/exact/wrong三种最后路由确定性重参数化 | 三种最后路由均可完全吸收进head列排列；最大误差`1.99e-13` | local deterministic audit / pass |
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

为落实“不能只使用LSTM/MLP”的模型优化要求，OPC1把OPA1中最强的非异常饱和模型ResCNN作为锚点，
在`3+3+4`残差阶段后插入固定P-layer，并完成seed6四行同预算正式矩阵。verified result branch显示
普通ResCNN平均AUC为`0.573572`，真实P混合为`0.546634`，错误P为`0.545181`，shuffle为
`0.500163`。真实P低于ResCNN `0.026938`，只高于错误P `0.001453`，`0/8`位置通过联合门，裁决为
`hold`。因此全局头混合路线停止，不做新密钥原样确认或机械扩容，四轮继续关闭。

OPK1进一步限定了“泛化”用语。即使参考与评估密钥使用完全相同的1024个明文，只用参考密钥逐明文
频率预测256把未见密钥时，八bit平均AUC仍为`0.500544`。因此当前跨密钥证据是“冻结位置与协议在
多把独立固定密钥上分别训练后重复成立”，不是一个不看密钥、没有support set的模型对新密钥零适配
泛化。更强主张必须改为support-set条件预测或密钥输入任务，并重新预注册。

OPM1进一步限定了“易预测位置”的机制用语。全部64个三轮输出bit都具有相同的64-bit输入依赖锥；
八个冻结位置经最后一轮inverse P-layer回溯后对应的S-box输出坐标`1/3`均为degree 3，而PRESENT还
存在degree 2坐标，四个坐标的
nonlinearity又同为4。因此当前不能把易预测性简单归因于更窄依赖锥或单S-box坐标明显更简单。更细的
固定密钥函数谱与训练动力学仍未解释；OPC1保持不变，不因该审计后验换位或增加远程实验。

OPN1进一步限定了OPC1的拓扑归因范围。普通ResCNN和两个混合模型都使用全局
`Linear(252*64,8)`头，每个输出读取全部64个最终位置。最后一次P重排可通过线性头列重排精确吸收，
identity、exact与wrong三种映射的数值等价最大误差为`1.99e-13`。因此OPC1若出现exact相对wrong
增益，只能归因于前两个路由及其后的非线性残差阶段，不能把最后路由单列为贡献。OPC1最终没有出现
该增益；OPN1据此只开放一个新的位置绑定head机制设计，不改变已经停止的全局头路线。

OPD1已经把这一机制落实为五行冻结矩阵：普通全局头ResCNN、无P位置头、exact-P位置头、wrong-P
位置头和exact-P标签打乱。八个独立局部head各自只读取对应最终位置的252维特征，整网参数量与全局
头锚点仅差约`0.026%`；exact-P和wrong-P可训练张量初始化逐项相同。seed7本地readiness已通过，
正式`2^17/2^16`、100 epochs任务已从推送提交在A6000运行并由本地watcher自动回收。OPD1尚无正式
AUC或裁决，不能写成精确拓扑正结果；运行期间不修改位置、阈值、模型、数据或控制。

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
| OPB1 | 限制自由位置条件后，exact-P能否超过wrong-P | seed4固定密钥，原锚点/低秩exact/wrong/shuffle四行矩阵 | anchor/exact/wrong均`1.000000`，exact-wrong为`0`，hold |
| OPC1 | 在非饱和ResCNN中加入P路由能否保持性能并恢复拓扑差异 | seed6固定密钥，全局头ResCNN/exact/wrong/shuffle四行矩阵 | ResCNN `0.573572`，exact `0.546634`，wrong `0.545181`；`0/8`联合通过，hold |
| OPN1 | OPC1最后路由在全局头下是否可识别 | identity/exact/wrong确定性重参数化审计 | 三种路由均可由线性head列重排吸收，最大误差`1.99e-13`，pass |
| OPD1 | 位置绑定head能否恢复exact-P相对四类控制的增益 | seed7固定密钥，五行参数匹配矩阵，`2^17/2^16`、100 epochs | readiness通过；远程正式任务运行中，尚无性能裁决 |

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
docs/experiments/innovation2-output-prediction-opn1-spn-rescnn-head-identifiability-audit-plan.md
docs/experiments/innovation2-output-prediction-opc1-present-r3-spn-rescnn-hybrid-plan.md
docs/experiments/innovation2-output-prediction-opd1-position-bound-spn-rescnn-plan.md
outputs/remote_results/i2_output_prediction_op10_present_r3_easy_bit_confirm_gpu0_20260721/
outputs/remote_results/i2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721/
outputs/remote_results_incomplete/i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721_raw_fallback/
outputs/remote_results/i2_output_prediction_opa2_present_r3_selected8_present_spn_key3_gpu0_20260722/
outputs/remote_results/i2_output_prediction_opa3_present_r3_selected8_topology_attribution_key3_gpu0_20260722/
outputs/remote_results/i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_key6_gpu0_20260722/
outputs/local_audits/i2_output_prediction_opm1_present_r3_selected_output_structural_baseline_audit_20260722/
outputs/local_audits/i2_output_prediction_opn1_present_r3_spn_rescnn_head_permutation_identifiability_audit_20260722/
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
7. 用OPN1确定性审计证明全局线性head可吸收最后固定路由，并据此提出参数匹配的位置绑定归因设计；
   OPD1完成前只能写方法与冻结协议，不能写性能收益。

不能写：

- 恢复了完整64-bit密文；
- 达到了PRESENT主流七至九轮神经区分或积分攻击轮数；
- 复现了Kimura跨100把密钥的论文结果；
- 多bit XOR普遍优于单bit预测；
- OP12证明所有四轮输出函数均不可预测；
- 三轮AUC可与真假样本区分准确率或SOTA攻击轮数直接比较。
- OPA2已经证明精确P-layer是增益原因，或已经得到四轮输出预测结果。
- OPA3通过了精确P-layer归因门，或OPA4/OPA5已经运行。
- OPD1正在运行就等于位置绑定head已经恢复精确P-layer贡献。

## 6. 论文建议结构

```text
4.1 固定密钥真实输出值预测任务与威胁模型
4.2 Kimura完整输出协议校准及完整输出失败分析
4.3 易预测输出坐标的发现与全新明文确认
4.4 专用八输出头及独立秘密密钥复现
4.5 五模型同预算筛选与第四密钥匹配控制确认
4.6 PRESENT-SPN-aware结构与P-layer同参数归因
4.7 SPN-ResCNN全局头负结果与输出头可识别性审计
4.8 位置绑定输出头及五类匹配控制
4.9 结构化多bit XOR负边界与条件式八输出四轮扩展
4.10 适用范围、与神经区分/积分预测的区别及停止边界
```

建议核心表格至少包含OP11八个位置的两密钥结果、专用头/完整头/shuffle三方对照、OPA1五模型
同预算结果、OPA2 true/shuffle独立确认，以及OP12六个mask的六门逐项结果。OPA3、OPB1和OPC1的
正式图已经从verified result branch回收并通过`visual-qa-redraw`；OPD1必须等正式归档与真实图像
质检后才能加入，不得从训练MSE预写测试结论。

## 7. 下一步

OP9--OP12已经整理为可进入论文模板的实验章节初稿：

```text
docs/research/innovation2-output-prediction-thesis-chapter-draft-20260721.md
```

OPC1已经完成并裁决为`hold`：真实P混合模型既未达到`0.55`，又低于普通ResCNN且没有超过错误P，
全局头混合路线停止；保留ResCNN锚点，不做seed7原样确认或四轮扩展。OPN1已证明最后一次路由可被
全局head吸收，因此下一研究问题不是加深当前模型，而是参数匹配的位置绑定输出头能否让最终路由
可识别。OPD1已保持PRESENT三轮、八个冻结bit、`2^17/2^16`和100 epochs不变，以普通ResCNN、无P
位置头、wrong-P和shuffle为控制，通过本地readiness并启动远程正式任务。当前唯一动作是等待watcher
回收并按冻结平均门和`4/8`逐bit联合门裁决；通过只开放全新固定密钥原样确认，失败则停止位置绑定，
两种分支均不直接开放四轮。

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

该审计只覆盖四篇已验证核心全文，不支持“首次”或穷尽性文献声明。OPA3已经否定原SPN-aware模型的
精确拓扑归因；当前位置绑定机制是否建立新的精确拓扑贡献，只能由OPD1冻结控制矩阵裁决。

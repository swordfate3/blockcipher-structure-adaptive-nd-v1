# 创新2：开题报告承诺到当前证据矩阵

日期：2026-07-22

状态：PDF原文复核完成 / OP9--OPA3与可识别性审计已映射 / OPB1正式训练中

## 1. 使用方式

本矩阵把开题报告中的输出预测承诺逐项映射到当前实现和证据，防止出现两类错误：

1. 用真实密文单bit/八bit的三轮结果冒充完整64-bit密文恢复、高轮攻击或跨密码结论；
2. 在完整`state_r -> state_(r+1)`已经被确定性基线100%解决时，继续训练多个神经网络并把结果
   解释成扩散临界轮。

页码同时给出PDF物理页和报告正文页。PDF原件：

```text
docs/24032307046_廖锡粤_开题附件.pdf
```

当前创新2任务、结果和停止边界仍以
`docs/research/innovation2-output-prediction-thesis-boundary-20260721.md`为权威入口；本矩阵只负责
开题对齐。

## 2. 承诺--证据矩阵

| 开题报告承诺 | 精确来源 | 当前实现 | 权威证据 | 覆盖判断 | 论文安全表述 | 不能声称 | 下一动作 |
|---|---|---|---|---|---|---|---|
| 使用神经网络预测真实密文输出bit | PDF物理第3页/报告第1页：“预测分组密码的密文输出比特” | 固定未知秘密密钥；输入未见明文；目标为真实PRESENT三轮完整64-bit或冻结八bit | OP9、OP10、OP11 | **部分完成**：八个冻结bit完成；完整64-bit未恢复 | “建立PRESENT三轮真实密文输出bit预测流程，并确认八个易预测坐标” | 恢复完整密文；输出预测普遍成功 | 等待OPB1；保留selected-bit任务，不以后验换bit |
| 比较ResNet、时间序列网络、Transformer等不同架构 | PDF物理第3页/报告第1页 | OPA1同预算比较MLP、六层LSTM、位置保持ResCNN、Transformer、PRESENT-SPN-aware | OPA1五模型结果；OPA2第四密钥确认 | **发现屏完成，独立确认部分完成** | “同预算发现屏显示架构归纳偏置显著影响八bit预测；SPN-aware在另一密钥上超过MLP和匹配shuffle” | 五类模型均已跨密钥确认；ResCNN/Transformer本身构成创新 | OPB1检验结构瓶颈；新的模型必须有机制和fresh-key确认 |
| 比较预测精度 | PDF物理第3页/报告第1页 | 每bit AUC、阈值accuracy、majority baseline、MSE、完整输出exact match | OP9--OPA3正式结果 | **当前PRESENT三轮任务完成** | 报告每个输出bit和模型的AUC/accuracy/majority差值，并保留完整输出exact-match负结果 | 与神经区分器AUC或不同论文攻击轮数直接比较 | OPB1继续使用同一八bit和同一预算 |
| 比较收敛速度 | PDF物理第3页/报告第1页 | 现有100-epoch运行保存逐epoch训练MSE和checkpoint | OP9、OPA1--OPA3 `history.csv` | **部分完成**：有训练历史，没有统一逐epoch held-out指标和time-to-threshold | “记录了训练损失演化，但尚未形成严格收敛速度排名” | 已证明某网络收敛更快；仅凭最终AUC推断收敛 | 后续跨模型矩阵增加不参与选模的统一epoch-time与held-out评估协议 |
| 比较泛化能力 | PDF物理第3页/报告第1页 | 训练/测试明文严格不重合；seed0--3代表不同固定秘密密钥且各自重新训练 | OP10 fresh确认、OP11第二密钥、OPA1第三密钥、OPA2/OPA3第四密钥 | **部分完成**：未见明文和多固定密钥证据存在，但不是单模型跨密钥零适配 | “同一冻结输出位置在多把独立固定密钥的逐密钥训练协议下复现” | 一个模型泛化到未知新密钥；五模型都已跨密钥稳定 | OPB1使用第五密钥；正式通过后用第六密钥不改协议确认 |
| 建立“网络架构--随机猜测临界轮数”关系 | PDF物理第3页/报告第1页 | 当前主任务正式证据集中在PRESENT三轮；OP12四轮预测的是不同XOR目标 | OP9--OPA3、OP12 | **缺失** | “当前只建立三轮任务的架构比较和四轮XOR负边界，尚未形成同目标临界轮曲线” | 已达到PRESENT主流七至九轮；OP12等价于同八bit四轮结果 | 只有新架构通过fresh-key归因后，才预注册同目标四轮；不跳五轮 |
| 捕捉“上一轮密文 -> 下一轮密文”的动态 | PDF物理第7页/报告第5页 | 对完整64-bit内部状态建立确定性PRESENT可识别性审计 | 完整轮间状态可识别性审计，16 keys × 31 rounds | **协议不可识别为临界轮** | “完整状态对一对即可恢复当轮子密钥，故该协议测量子密钥可识别性而非累计扩散” | 神经网络在该协议上的高准确率表示结构脆弱；准确率会随轮数回到随机 | 不训练完整状态轮间模型；若保留轮间研究，先改成部分可见或严格跨密钥并重做确定性门 |
| 将“明文--最终密文”重构为“中间轮状态--下一轮状态” | PDF物理第14页/报告第12页 | 已审计完整状态版本；当前有效主线仍为`P -> E_K^r(P)`真实输出 | 可识别性审计；OP9--OPB1 | **原细化方案需修订** | “先审计轮间任务的可识别性，发现完整状态版本退化，因此保留多轮端到端真实输出并研究受限轮间观测” | 已按原方案完成神经轮间预测；确定性审计否定所有轮间研究 | 论文方法章节明确说明协议修订理由和适用范围 |
| 研究RNN/LSTM、门控Transformer等序列模型 | PDF物理第14页/报告第12页 | Kimura式LSTM与Transformer已在三轮输出任务评估，均近随机；不是唯一模型 | OP9、OPA1 | **基线覆盖完成，优化主张未完成** | “通用序列模型是论文/架构基线；当前数据支持结构感知和位置保持模型优先” | LSTM失败意味着输出预测失败；换更大Transformer必然扩轮 | 不重复无机制的大模型枚举；优先结构瓶颈或SPN-ResCNN |
| 用输出预测增加分组密码安全评估维度 | PDF物理第19页/报告第17页，正式创新（2） | 真实输出任务、论文协议校准、易预测坐标发现、独立密钥确认、多架构与控制归因已形成闭环 | OP9--OPA3、OPB1计划、论文边界文档 | **PRESENT三轮窄范围实质支持** | “在神经区分器之外建立真实密文输出值预测评估流程，并给出PRESENT三轮的正结果与四轮负边界” | 已形成通用分组密码安全边界；已达到SOTA攻击轮数 | 完成OPB1裁决；再决定同目标四轮或收束PRESENT章节 |
| 跨密码/跨结构输出预测评估 | PDF物理第3页/报告第1页“分组密码”，物理第19页/报告第17页系统创新 | 当前权威真实输出值路线只覆盖PRESENT-80 | 当前无同协议GIFT/AES/SPECK/SM4真实输出预测结果 | **缺失** | “PRESENT作为SPN机制研究对象，跨密码有效性尚待验证” | 对所有SPN或所有分组密码成立 | PRESENT方法稳定后选择一个非PRESENT密码做最小同协议验证，不同时扩模型和轮数 |
| 大规模实验形成安全边界 | PDF物理第3--4页/报告第1--2页 | OP9--OPA3使用`2^17`训练、`2^16`测试、100 epochs；OPB1同规模运行中 | 各正式实验计划和远程归档 | **部分完成**：达到Kimura单密钥样本预算，不是跨多轮/跨密码大规模边界 | “完成PRESENT三轮固定密钥论文协议量级实验” | paper-scale多密钥复现；百万/类；完整临界轮图 | 精确报告train/test totals、密钥数、epoch和目标，不使用`/class`术语 |

## 3. 当前论文级结论

截至OPB1正式结果回收前，创新2最稳妥的论文主张是：

> 本文在固定未知秘密密钥下建立真实密文输出值预测评估流程。通过论文协议校准、输出坐标发现、
> fresh明文确认、独立密钥复现和同预算多架构控制，发现PRESENT三轮存在八个可预测输出bit；位置
> 保持ResCNN和SPN式分层扩散网络优于通用LSTM/Transformer，其中SPN式网络的整体架构收益得到
> 独立密钥及标签打乱控制支持，但精确P-layer尚未超过固定错误双射。对开题提出的完整轮间状态
> 预测，确定性审计证明一对状态即可恢复当轮子密钥，因此修订为多轮端到端真实输出预测和受限
> 轮间观测，而不使用退化协议定义随机猜测临界轮。

这段结论同时保留了创新2的正贡献和负边界，没有把协议修订写成方向失败。

## 4. 当前执行顺序

1. 等待OPB1第五固定密钥正式结果由tmux watcher自动回收；
2. 若OPB1同时保持输出预测能力并超过wrong-P/shuffle，使用第六固定密钥确认，不立即扩轮；
3. 若只有归因但性能下降，只允许一次训练隔离的低秩表达修复；
4. 若未归因，停止瓶颈路线，优先独立设计SPN-ResCNN混合候选；
5. 不训练完整`state_r -> state_(r+1)`多架构矩阵；
6. PRESENT方法边界稳定后，再选择一个非PRESENT密码验证跨密码性。

## 5. 权威证据入口

```text
docs/research/innovation2-output-prediction-thesis-boundary-20260721.md
docs/research/innovation2-output-prediction-thesis-chapter-draft-20260721.md
docs/research/innovation2-spn-aware-output-prediction-novelty-audit-20260722.md
docs/experiments/innovation2-present-next-round-full-state-identifiability-audit-plan.md
docs/experiments/innovation2-output-prediction-opb1-present-r3-topology-bottleneck-plan.md
outputs/local_audits/i2_present_next_round_full_state_identifiability_audit_20260722/
```

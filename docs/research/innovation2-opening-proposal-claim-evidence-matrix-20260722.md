# 创新2：开题报告承诺到当前证据矩阵

日期：2026-07-22

状态：PDF原文复核完成 / OPC1 hold并停止 / 位置绑定head新假设待预注册

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
| 使用神经网络预测真实密文输出bit | PDF物理第3页/报告第1页：“预测分组密码的密文输出比特” | 固定未知秘密密钥；输入未见明文；目标为真实PRESENT三轮完整64-bit或冻结八bit | OP9--OPC1 | **部分完成**：八个冻结bit完成；完整64-bit未恢复；OPC1全局头混合未过门 | “建立PRESENT三轮真实密文输出bit预测流程，并确认八个易预测坐标” | 恢复完整密文；输出预测普遍成功 | 保留ResCNN，预注册位置绑定head机制 |
| 比较ResNet、时间序列网络、Transformer等不同架构 | PDF物理第3页/报告第1页 | OPA1五模型；OPA2/OPA3确认归因；OPB1低秩瓶颈；OPC1 SPN-ResCNN；OPN1输出头审计 | OPA1--OPC1、OPN1 verified/local evidence | **多架构筛选完成；全局头SPN-ResCNN低于ResCNN且未超过wrong-P** | “架构归纳偏置显著影响八bit预测，但精确拓扑贡献尚未建立” | ResCNN/Transformer本身构成创新；OPC1是正结果 | 只改为参数匹配位置绑定head，保留全部控制 |
| 比较预测精度 | PDF物理第3页/报告第1页 | 每bit AUC、阈值accuracy、majority baseline、MSE、完整输出exact match | OP9--OPB1正式结果 | **当前PRESENT三轮任务完成并继续模型归因** | 报告每个输出bit和模型的AUC/accuracy/majority差值，并保留完整输出exact-match负结果 | 与神经区分器AUC或不同论文攻击轮数直接比较 | OPC1继续使用同一八bit和同一预算 |
| 比较收敛速度 | PDF物理第3页/报告第1页 | 现有100-epoch运行保存逐epoch训练MSE和checkpoint | OP9、OPA1--OPA3 `history.csv` | **部分完成**：有训练历史，没有统一逐epoch held-out指标和time-to-threshold | “记录了训练损失演化，但尚未形成严格收敛速度排名” | 已证明某网络收敛更快；仅凭最终AUC推断收敛 | 后续跨模型矩阵增加不参与选模的统一epoch-time与held-out评估协议 |
| 比较泛化能力 | PDF物理第3页/报告第1页 | 训练/测试明文严格不重合；seed0--3逐密钥重训；OPK1用256参考/256未见密钥审计key-blind目标稳定性 | OP10/OP11、OPA1--OPA3、OPK1 | **边界已明确**：有未见明文和逐密钥重复性；key-blind零样本新密钥AUC `0.500544` | “冻结位置和协议在多把独立固定密钥上分别训练后复现；无密钥零适配任务没有稳定标签” | 一个模型已泛化到未知新密钥；五模型均跨密钥稳定 | OPB1继续固定密钥确认；更强泛化必须另行设计support-set条件协议 |
| 建立“网络架构--随机猜测临界轮数”关系 | PDF物理第3页/报告第1页 | 当前主任务正式证据集中在PRESENT三轮；OP12四轮预测的是不同XOR目标 | OP9--OPA3、OP12 | **缺失** | “当前只建立三轮任务的架构比较和四轮XOR负边界，尚未形成同目标临界轮曲线” | 已达到PRESENT主流七至九轮；OP12等价于同八bit四轮结果 | 只有新架构通过fresh-key归因后，才预注册同目标四轮；不跳五轮 |
| 捕捉“上一轮密文 -> 下一轮密文”的动态 | PDF物理第7页/报告第5页 | 对完整64-bit内部状态建立确定性PRESENT可识别性审计 | 完整轮间状态可识别性审计，16 keys × 31 rounds | **协议不可识别为临界轮** | “完整状态对一对即可恢复当轮子密钥，故该协议测量子密钥可识别性而非累计扩散” | 神经网络在该协议上的高准确率表示结构脆弱；准确率会随轮数回到随机 | 不训练完整状态轮间模型；若保留轮间研究，先改成部分可见或严格跨密钥并重做确定性门 |
| 将“明文--最终密文”重构为“中间轮状态--下一轮状态” | PDF物理第14页/报告第12页 | 已审计完整下一状态和冻结八个下一状态bit；当前有效主线仍为`P -> E_K^r(P)`真实输出 | 完整状态与selected8可识别性审计；OP9--OPB1 | **原细化方案需修订** | “完整当前状态版本会退化为全部或局部轮密钥恢复，因此保留多轮端到端真实输出；轮间动态必须采用更严格受限观测或跨密钥协议” | 已按原方案完成神经轮间预测；确定性审计否定所有部分状态或跨密钥研究 | 不训练当前状态可见的selected8网络；论文方法章节解释协议修订理由 |
| 研究RNN/LSTM、门控Transformer等序列模型 | PDF物理第14页/报告第12页 | Kimura式LSTM与Transformer已在三轮输出任务评估，均近随机；不是唯一模型 | OP9、OPA1 | **基线覆盖完成，优化主张未完成** | “通用序列模型是论文/架构基线；当前数据支持结构感知和位置保持模型优先” | LSTM失败意味着输出预测失败；换更大Transformer必然扩轮 | 不重复无机制的大模型枚举；优先结构瓶颈或SPN-ResCNN |
| 用输出预测增加分组密码安全评估维度 | PDF物理第19页/报告第17页，正式创新（2） | 真实输出任务、协议校准、易预测坐标、独立密钥、多架构与两级控制归因已形成闭环 | OP9--OPB1、协议审计、论文边界文档 | **PRESENT三轮窄范围实质支持** | “在神经区分器之外建立真实密文输出值预测评估流程，并给出PRESENT三轮正结果、拓扑归因边界与四轮XOR负边界” | 已形成通用分组密码安全边界；已达到SOTA攻击轮数 | 完成OPC1，再决定确认、跨密码验证或论文收束 |
| 跨密码/跨结构输出预测评估 | PDF物理第3页/报告第1页“分组密码”，物理第19页/报告第17页系统创新 | 当前权威真实输出值路线只覆盖PRESENT-80 | 当前无同协议GIFT/AES/SPECK/SM4真实输出预测结果 | **缺失** | “PRESENT作为SPN机制研究对象，跨密码有效性尚待验证” | 对所有SPN或所有分组密码成立 | PRESENT方法稳定后选择一个非PRESENT密码做最小同协议验证，不同时扩模型和轮数 |
| 大规模实验形成安全边界 | PDF物理第3--4页/报告第1--2页 | OP9--OPC1使用`2^17`训练、`2^16`测试、100 epochs | 各正式实验计划和远程归档 | **部分完成**：达到Kimura单密钥样本预算，不是跨多轮/跨密码大规模边界 | “完成PRESENT三轮固定密钥论文协议量级实验” | paper-scale多密钥复现；百万/类；完整临界轮图 | 精确报告train/test totals、密钥数、epoch和目标，不使用`/class`术语 |

## 3. 当前论文级结论

截至OPB1正式结果回收后，创新2最稳妥的论文主张是：

> 本文在固定未知秘密密钥下建立真实密文输出值预测评估流程。通过论文协议校准、输出坐标发现、
> fresh明文确认、独立密钥复现和同预算多架构控制，发现PRESENT三轮存在八个可预测输出bit；位置
> 保持ResCNN和SPN式分层扩散网络优于通用LSTM/Transformer，其中SPN式网络的整体架构收益得到
> 独立密钥及标签打乱控制支持，但OPA3与OPB1均未使精确P-layer超过固定错误双射。完整和八输出
> 轮间状态审计分别退化为全部或局部轮密钥恢复；OPK1还证明无密钥零样本新密钥目标不稳定。因此
> 本文保留固定密钥多轮端到端真实输出预测，不使用退化协议定义随机猜测临界轮，也不把逐密钥重训
> 复现包装成零样本密钥泛化。

这段结论同时保留了创新2的正贡献和负边界，没有把协议修订写成方向失败。

## 4. 当前执行顺序

1. OPB1低秩瓶颈与OPC1全局头混合路线均已停止，不再调整rank、深度、数据、epoch或错误P；
2. 保留OPC1普通ResCNN `0.573572`为新矩阵锚点；
3. OPN1已证明最后一次P重排可被全局head吸收；下一项只改为参数匹配的位置绑定head，并保留
   无路由、wrong-P和shuffle控制；
4. 新head先过确定性可识别性与本地readiness，再决定同规模远程训练，不立即扩轮；
5. 不训练完整`state_r -> state_(r+1)`多架构矩阵；
6. PRESENT方法边界稳定后，再选择一个非PRESENT密码验证跨密码性。

## 5. 权威证据入口

```text
docs/research/innovation2-output-prediction-thesis-boundary-20260721.md
docs/research/innovation2-output-prediction-thesis-chapter-draft-20260721.md
docs/research/innovation2-spn-aware-output-prediction-novelty-audit-20260722.md
docs/experiments/innovation2-present-next-round-full-state-identifiability-audit-plan.md
docs/experiments/innovation2-output-prediction-opb1-present-r3-topology-bottleneck-plan.md
docs/experiments/innovation2-output-prediction-opc1-present-r3-spn-rescnn-hybrid-plan.md
outputs/local_audits/i2_present_next_round_full_state_identifiability_audit_20260722/
outputs/remote_results/i2_output_prediction_opb1_present_r3_topology_bottleneck_key4_gpu0_20260722/
```

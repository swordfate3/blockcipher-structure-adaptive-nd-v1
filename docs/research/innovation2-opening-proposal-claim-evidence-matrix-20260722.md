# 创新2：开题报告承诺到当前证据矩阵

日期：2026-07-22

状态：PDF原文复核完成 / PRESENT三轮正结果与OPF1四轮同协议边界已形成 / OPF2 `2^20`规模审判运行中

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
| 使用神经网络预测真实密文输出bit | PDF物理第3页/报告第1页：“预测分组密码的密文输出比特” | 固定未知秘密密钥；输入未见明文；目标为真实PRESENT完整64-bit、八个预注册bit或确定性parity；OPF1/2保持同八bit进入四轮 | OP9--OPF1 verified结果；OPF2运行记录 | **部分完成**：三轮八bit正结果已确认；完整64-bit和简单八bit parity未通过；四轮`2^17`同协议未过门，`2^20`仍在审判 | “建立PRESENT固定密钥真实密文输出预测流程，确认三轮八个易预测坐标，并测量同目标进入四轮后的衰减” | 恢复完整密文；输出预测普遍成功；简单XOR一定增强信号；运行中OPF2已经成功 | 等待OPF2正式裁决，保持完整输出、选定位和组合输出三种标签边界 |
| 比较ResNet、时间序列网络、Transformer等不同架构 | PDF物理第3页/报告第1页 | OPA1五模型；OPA2/OPA3确认归因；OPB1/OPC1/OPN1结构审计；OPD1位置绑定head；OPF1/2四轮同五模型尺度比较 | OPA1--OPF1 verified/local/fallback evidence；OPF2运行记录 | **多架构筛选完成且开始同架构轮数比较**：三轮位置绑定有效但exact/wrong持平；四轮`2^17`五模型均接近随机，OPF2只检验训练规模 | “架构归纳偏置显著影响PRESENT三轮八bit预测；精确拓扑贡献未建立，四轮规模效应仍待裁决” | ResCNN/Transformer本身构成创新；OPD1 exact单行证明精确P；OPF2中间训练损失支持成功 | 不再枚举无机制网络；OPF2 hold后只允许预注册OPF3共享轮SPN，pass则只换密钥确认 |
| 比较预测精度 | PDF物理第3页/报告第1页 | 每bit AUC、阈值accuracy、majority baseline、MSE、完整输出exact match和模型/控制差值 | OP9--OPF1正式或fallback结果 | **三轮与四轮`2^17`结果已形成**：OPD1 exact/wrong差`+0.000022`，精确P归因失败；OPF1 exact平均AUC `0.513755`、对shuffle `+0.012821`、`0/8`过门 | “逐bit报告AUC、accuracy-majority、shuffle及错误结构差值；三轮正结果与四轮同协议hold分开表述” | 与神经区分器AUC或不同论文攻击轮数直接比较；把accuracy约0.51写成可预测成功 | OPF2完成后沿用冻结联合门；跨密码完整输出另报告BAPavg和exact match |
| 比较收敛速度 | PDF物理第3页/报告第1页 | 现有100-epoch运行保存逐epoch训练MSE和checkpoint | OP9、OPA1--OPA3 `history.csv` | **部分完成**：有训练历史，没有统一逐epoch held-out指标和time-to-threshold | “记录了训练损失演化，但尚未形成严格收敛速度排名” | 已证明某网络收敛更快；仅凭最终AUC推断收敛 | 后续跨模型矩阵增加不参与选模的统一epoch-time与held-out评估协议 |
| 比较泛化能力 | PDF物理第3页/报告第1页 | 训练/测试明文严格不重合；多把固定密钥分别训练；OPK1审计key-blind目标稳定性；OPF2-C1已拆分独立`key_seed` | OP10/OP11、OPA1--OPF1、OPK1；OPF2-C1条件计划 | **边界已明确但四轮新密钥待条件授权**：三轮有逐密钥重训复现；key-blind零样本新密钥AUC `0.500544`；OPF1仅单密钥hold | “同一冻结协议可在独立固定密钥上分别训练确认；这不是一个checkpoint零样本泛化任意密钥” | 一个模型已泛化到未知新密钥；OPF1单密钥hold已经构成跨密钥四轮边界 | OPF2 pass才只换`key_seed=8`确认；hold关闭C1并转OPF3 |
| 建立“网络架构--随机猜测临界轮数”关系 | PDF物理第3页/报告第1页 | OPD1与OPF1保持同一seed7密钥、明文划分、八个输出bit、五模型和`2^17/2^16`预算，只把PRESENT轮数从3改为4；OPF2只把训练规模增至`2^20` | OPD1、OPF1 verified archives；OPF2运行记录 | **部分完成**：OPD1三轮候选AUC `0.999996`；OPF1四轮为`0.513755`、对shuffle `+0.012821`且`0/8`过门，形成单密钥同预算三至四轮经验边界；OPF2尚未揭盲 | “在seed7固定密钥和`2^17`训练预算下，同网络同目标从三轮到四轮信号显著衰减；`2^20`规模是否改变裁决仍待OPF2” | 四轮数学不可预测；所有网络上限为三轮；OPF2 readiness代表四轮恢复；达到主流七至九轮 | 回收并裁决OPF2；pass只换`key_seed=8`确认，hold才运行预注册OPF3，invalid只修协议；不跳五轮 |
| 捕捉“上一轮密文 -> 下一轮密文”的动态 | PDF物理第7页/报告第5页 | 对完整64-bit内部状态建立确定性PRESENT可识别性审计 | 完整轮间状态可识别性审计，16 keys × 31 rounds | **协议不可识别为临界轮** | “完整状态对一对即可恢复当轮子密钥，故该协议测量子密钥可识别性而非累计扩散” | 神经网络在该协议上的高准确率表示结构脆弱；准确率会随轮数回到随机 | 不训练完整状态轮间模型；若保留轮间研究，先改成部分可见或严格跨密钥并重做确定性门 |
| 将“明文--最终密文”重构为“中间轮状态--下一轮状态” | PDF物理第14页/报告第12页 | 已审计完整下一状态和冻结八个下一状态bit；当前有效主线仍为`P -> E_K^r(P)`真实输出 | 完整状态与selected8可识别性审计；OP9--OPB1 | **原细化方案需修订** | “完整当前状态版本会退化为全部或局部轮密钥恢复，因此保留多轮端到端真实输出；轮间动态必须采用更严格受限观测或跨密钥协议” | 已按原方案完成神经轮间预测；确定性审计否定所有部分状态或跨密钥研究 | 不训练当前状态可见的selected8网络；论文方法章节解释协议修订理由 |
| 研究RNN/LSTM、门控Transformer等序列模型 | PDF物理第14页/报告第12页 | Kimura式LSTM与Transformer已在三轮输出任务评估，均近随机；不是唯一模型 | OP9、OPA1 | **基线覆盖完成，优化主张未完成** | “通用序列模型是论文/架构基线；当前数据支持结构感知和位置保持模型优先” | LSTM失败意味着输出预测失败；换更大Transformer必然扩轮 | 不重复无机制的大模型枚举；优先结构瓶颈或SPN-ResCNN |
| 用输出预测增加分组密码安全评估维度 | PDF物理第19页/报告第17页，正式创新（2） | 真实输出任务、协议校准、易预测坐标、独立密钥、多架构、错误结构、shuffle与轮数控制已形成PRESENT闭环 | OP9--OPF1、OPN1、协议审计、论文边界文档 | **PRESENT三轮窄范围实质支持，四轮边界正在重审** | “在神经区分器之外建立真实密文输出值预测流程，给出PRESENT三轮正结果、精确拓扑归因边界、简单XOR负边界及四轮同协议衰减” | 已形成通用分组密码安全边界；达到SOTA攻击轮数；OPD1证明精确P；OPF1证明所有四轮失败 | 回收OPF2后执行唯一分支，再用GIFT、SPECK、DES检验跨结构方法 |
| 跨密码/跨结构输出预测评估 | PDF物理第3页/报告第1页“分组密码”，物理第19页/报告第17页系统创新 | 权威性能证据仍只覆盖PRESENT-80；GIFT数据/发现CLI、SPECK数据/模型/训练CLI、DES数据/模型已完成条件单元测试 | 多结构执行蓝图及GX1、ARX1、FEISTEL1条件计划；当前无同协议GIFT/SPECK/DES/AES/SM4性能结果 | **缺失性能证据但实现准备已推进** | “PRESENT作为SPN机制研究对象；GIFT、SPECK和DES已冻结同任务条件协议，跨结构性能有效性尚待依次验证” | 单元测试等于跨结构成功；对所有SPN、ARX、Feistel或所有分组密码成立 | 先闭环OPF2唯一分支，再依次执行GIFT、SPECK、DES；不并行占用GPU或跨门启动 |
| 大规模实验形成安全边界 | PDF物理第3--4页/报告第1--2页 | OP9--OPF1使用`2^17`训练、`2^16`测试、100 epochs；OPF2正在运行`2^20`训练、`2^16`测试、100 epochs五模型矩阵 | 各正式实验计划、verified远程归档和OPF2 watcher记录 | **部分完成**：三轮与四轮`2^17`单密钥证据已形成；百万总训练行的OPF2尚未完成，不是结果或跨密码边界 | “完成PRESENT三轮论文协议量级实验和四轮同预算边界；四轮百万总训练行规模审判正在执行” | paper-scale多密钥复现；`2^20`是每类样本；运行中任务已经支持结论；完整跨密码临界轮图 | 精确报告train/test总量、密钥数、epoch和目标；等待OPF2正式gate，不使用`/class`术语 |

## 3. 当前论文级结论

PRESENT三轮方法证据以OPD1及此前独立密钥实验为基础，最稳妥的论文主张是：

> 本文在固定未知秘密密钥下建立真实密文输出值预测评估流程。通过论文协议校准、输出坐标发现、
> fresh明文确认、独立密钥复现和同预算多架构控制，发现PRESENT三轮存在八个可预测输出bit；位置
> 保持ResCNN和SPN式分层扩散网络优于通用LSTM/Transformer，其中SPN式网络的整体架构收益得到
> 独立密钥及标签打乱控制支持，但OPA3与OPB1均未使精确P-layer超过固定错误双射。完整和八输出
> 轮间状态审计分别退化为全部或局部轮密钥恢复；OPK1还证明无密钥零样本新密钥目标不稳定。因此
> 本文保留固定密钥多轮端到端真实输出预测，不使用退化协议定义随机猜测临界轮，也不把逐密钥重训
> 复现包装成零样本密钥泛化。OPC1全局头混合未过门，OPN1证明最后固定路由可被全局head吸收；
> OPD1位置绑定使exact-P与wrong-P都近乎完美，但二者只差`+0.000022`，所以仍不支持精确拓扑归因。

OPF1随后补齐了同一seed7密钥、同一明文划分、同八个输出bit、同五模型和同`2^17/2^16`预算下只增加
一轮的比较：真实P位置头平均AUC从三轮`0.999996158`降到四轮`0.513755358`，对matched shuffle只有
`+0.012821215`，且`0/8`输出bit过门。因此当前可增加“该单密钥预算下三至四轮经验边界”，但不能写成
所有网络或所有四轮输出函数的上限。OPF2正在只把训练总行数扩到`2^20`进行规模审判；正式gate回收前，
论文结论不得写入OPF2方向或数值。

这些结论同时保留创新2的正贡献和负边界，没有把协议修订、精确拓扑归因失败或运行中实验写成方向失败。

## 4. 当前执行顺序

1. 由本地watcher回收并裁决OPF2正式`2^20/2^16`五模型结果，不读取中间指标改门；
2. OPF2 pass只开放OPF2-C1，保持data/model/shuffle seed7并只把`key_seed`改为8；
3. OPF2 hold关闭C1，只开放已盲预注册的OPF3共享逐轮SPN本地readiness；
4. OPF2 invalid只修复来源、缓存、切分或产物协议，不运行任何条件分支；
5. PRESENT当前分支闭环后依次执行GIFT-64、SPECK32/64和DES真实输出预测，不并行占用远程GPU；
6. 不恢复完整`state_r -> state_(r+1)`退化协议、简单八bit全XOR路线或无机制网络枚举，也不跳到五轮。

## 5. 权威证据入口

```text
docs/research/innovation2-output-prediction-thesis-boundary-20260721.md
docs/research/innovation2-output-prediction-thesis-chapter-draft-20260721.md
docs/research/innovation2-spn-aware-output-prediction-novelty-audit-20260722.md
docs/research/innovation2-multistructure-output-prediction-execution-blueprint-20260723.md
docs/experiments/innovation2-present-next-round-full-state-identifiability-audit-plan.md
docs/experiments/innovation2-output-prediction-opb1-present-r3-topology-bottleneck-plan.md
docs/experiments/innovation2-output-prediction-opc1-present-r3-spn-rescnn-hybrid-plan.md
docs/experiments/innovation2-output-prediction-opn1-spn-rescnn-head-identifiability-audit-plan.md
docs/experiments/innovation2-output-prediction-opd1-position-bound-spn-rescnn-plan.md
docs/experiments/innovation2-output-prediction-opf1-present-r4-position-bound-boundary-plan.md
docs/experiments/innovation2-output-prediction-opf2-present-r4-2p20-scale-adjudication-plan.md
docs/research/innovation2-cross-cipher-output-prediction-route-ranking-20260722.md
outputs/local_audits/i2_present_next_round_full_state_identifiability_audit_20260722/
outputs/remote_results/i2_output_prediction_opb1_present_r3_topology_bottleneck_key4_gpu0_20260722/
outputs/remote_results/i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_key6_gpu0_20260722/
outputs/remote_results/i2_output_prediction_opf1_present_r4_position_bound_spn_rescnn_key7_gpu0_20260722/
```

# 创新2跨密码真实输出预测路线排名

日期：2026-07-22

状态：条件研究蓝图 / OPD1完成前不实现、不训练、不启动

## 1. 决策问题

创新2当前已经在PRESENT-80三轮完成真实密文输出bit发现、fresh明文确认、独立固定密钥复现和
多架构控制，但开题中的“跨密码/跨结构输出预测评估”仍缺少同一任务协议下的第二种密码证据。

本审计只回答：OPD1正式裁决完成后，如果仍需要增加一条最有论文价值、且不重复堆叠PRESENT三轮
模型的实验路线，应优先选择哪种密码，以及最小可归因协议是什么。

它不是新实验计划。OPD1结果、正式SVG、结果索引和论文边界文档没有闭环前，不得据此实现runner、
运行本地性能屏或占用远程GPU。

## 2. 候选排名

| 排名 | 候选 | 与当前方法的一致性 | 已有邻近论文 | 仓库准备度 | 裁决 |
|---:|---|---|---|---|---|
| 1 | GIFT-64三轮 | 同为64-bit、16个4-bit S-box和bit permutation，可直接检验SPN式位置保持与分层扩散是否跨密码成立 | 当前已核验核心输出预测论文未覆盖GIFT；该未命中不是首次性证明 | `Gift64`、官方零向量、标量/向量化加密和真实/错误P映射均已有验证 | 首选 |
| 2 | SPECK32/64三轮 | 可检验方法是否跨到ARX，但需要全新的word/rotation/carry结构模型，不能复用SPN主张 | Jeong 2024已报告`2^22`训练、300 epochs下FCNN/BiLSTM完整输出结果 | 官方向量、标量/NumPy/Torch实现均已有验证 | 只作外部协议校准备选 |
| 3 | AES/SAES | 有跨论文可比性，但block宽度、byte结构、轮函数和训练规模同时变化 | Jeong 2024/2026已覆盖 | 仓库有密码实现，但没有当前selected-output结构适配 | 暂缓 |
| 4 | SM4/其他Feistel-like | 可扩大结构范围，但与PRESENT方法变量距离过大 | 与当前核心输出预测文献链不直接对齐 | 有密码实现，缺少同任务结构网络 | 暂缓 |

历史创新1的GIFT真假样本神经区分结果，以及创新2早期GIFT积分平衡谱/关系标签结果，任务、标签和指标
均不同，不能作为这里的真实输出值证据，也不能直接提供候选bit。

## 3. 为什么优先GIFT-64

GIFT-64是最小的跨密码方法检验，而不是因为预期它更容易得到高AUC：

```text
PRESENT-80：64-bit block / 4-bit S-box / bit permutation / 80-bit key
GIFT-64：   64-bit block / 4-bit S-box / bit permutation / 128-bit key
```

两者保持block宽度和SPN基本粒度不变，但S-box、P-layer和key schedule不同。若同一“全64位扫描 ->
冻结易预测位置 -> fresh确认 -> 独立密钥selected-output模型比较”流程在GIFT也成立，可以支持方法层面的
跨SPN可重复性；若不成立，则给出PRESENT特异性边界。两种结果都比继续枚举PRESENT错误P、增加epoch
或机械扩样本更有论文信息量。

当前实现准备证据包括：

```text
Gift64(rounds=28, key=0).encrypt(0) = 0xF62BC3EF34F775AC
Gift64支持1--28轮、64-bit明文和128-bit固定秘密密钥
GIFT P-layer / inverse P-layer均为64位置双射
现有测试已验证真实P索引与Gift64.inverse_permutation_layer一致
四轮向量化加密已在多把密钥、多条明文上逐项匹配标量实现
```

这些证据只说明密码语义和结构适配可复用，不说明GIFT三轮存在可预测输出bit。

## 4. OPD1后的条件分支

### 4.1 OPD1通过

先执行OPD1预注册的全新固定密钥原样五行确认。确认完成前不启动GIFT。只有位置绑定exact-P相对
全局头、无P、wrong-P和shuffle再次过门，才把“位置绑定精确拓扑候选”带入GIFT适配。

### 4.2 OPD1未通过

停止PRESENT位置绑定路线，不用GIFT救援或改写OPD1。GIFT若随后获准，只测试更保守的整体流程：

```text
易预测输出位置发现与fresh确认
+ selected-output减少多任务干扰
+ SPN式位置保持/局部混合/跨nibble扩散
```

此分支不得声称精确P-layer已经得到支持；exact/wrong-P只能继续作为归因控制。优先级低于先更新
创新2论文边界和章节草稿，只有论文确实需要第二密码证据时才占用远程实验槽。

## 5. GIFT最小同协议路线

### Phase GX0：确定性与本地readiness

只验证实现，不解释AUC：

```text
cipher / rounds       = GIFT-64 / 3
input                  = 64 MSB-first plaintext bits
target                 = 同一明文的64个真实三轮密文bit
key protocol           = 单把固定未知128-bit秘密密钥，逐密钥重新训练
sample classification  = false
local rows / epochs    = 64 train + 64 discovery + 64 fresh / 1
```

必须验证官方向量、三轮标量回放、bit顺序、P/inverse-P往返、不同seed密钥、三个明文split零重合、
真实标签、标签打乱只作用训练行、磁盘缓存、checkpoint和结果闭环。GX0失败只修协议，不远程训练。

### Phase GX1：全64位发现与fresh确认

为避免把PRESENT位置硬套给GIFT，GX1必须重新扫描全部64个GIFT输出位置：

```text
train / discovery / fresh = 2^17 / 2^16 / 2^16 total pairs
epochs / batch            = 100 / 250
loss / optimizer          = 与PRESENT校准锚点一致的MSE / RMSprop
models                    = full64 MLP true + 参数匹配full64 MLP shuffle
candidate_limit           = 8，在fresh标签读取前冻结
```

候选仍按保守联合分数排序：

```text
min(AUC-0.5, accuracy-majority, true AUC-shuffle AUC)
```

候选在fresh集上必须保持方向，并同时满足`AUC >= 0.510`、`accuracy-majority >= +0.005`和
`true-shuffle >= +0.005`。至少`4/8`冻结候选通过才开放GX2；否则停止GIFT输出位置路线，不更换seed、
模型、轮数、候选容量、数据或epoch补救。

### Phase GX2：独立密钥架构确认

GX2使用第二把固定秘密密钥和GX1冻结位置，不重新选bit。为保持矩阵精简，先比较三个true模型：

```text
selected8 MLP
selected8位置保持ResCNN
selected8 GIFT-SPN-aware
```

只有某个非MLP候选在同预算下超过MLP平均AUC至少`+0.003`、至少`4/8`位置增益达到`+0.002`，才在
第三把固定密钥执行三行确认：`MLP true / 候选 true / 候选 matched-shuffle`。错误P归因是确认后的
独立单变量阶段，不能把六个模型塞进同一个发现矩阵，也不能在GX2测试集上调P映射。

## 6. 证据与停止边界

GIFT路线即使全部通过，也只能支持：

> 同一真实输出值预测流程在两个64-bit轻量SPN上发现并独立确认了选定位输出信号，且结构感知候选
> 相对通用锚点获得受控增益。

不能支持：

- 一个模型零适配泛化到新密码或新密钥；
- PRESENT checkpoint直接迁移到GIFT；
- 完整64-bit密文恢复；
- 四轮以上或主流攻击轮数；
- exact GIFT P-layer因果贡献，除非独立wrong-P归因门通过；
- 文献首次或SOTA。

GIFT GX1/GX2不与Jeong的SPECK `2^22/300 epochs`数值直接比较。若后续选择SPECK，目标应是严格
复核Jeong完整输出BAPavg协议，而不是把它包装成PSA-SOP跨SPN验证。

## 7. 当前推荐动作

```text
now                     = 等待并闭环OPD1正式结果
do_not_launch           = GX0 / GX1 / GX2 / SPECK / Jeong scale
after OPD1 pass         = 先做OPD1 fresh-key原样确认
after OPD1 hold         = 先更新论文边界；再决定是否需要GIFT跨密码证据
cross-cipher priority   = GIFT-64 > SPECK32/64 > AES/SAES > SM4/其他
```

权威参考：

```text
docs/research/innovation2-output-prediction-thesis-boundary-20260721.md
docs/research/innovation2-output-prediction-paper-protocol-audit-20260721.md
docs/research/innovation2-spn-aware-output-prediction-novelty-audit-20260722.md
sources/research_innovation2_spn_aware_output_prediction_web_20260722.md
src/blockcipher_nd/ciphers/spn/gift.py
src/blockcipher_nd/ciphers/arx/speck.py
```

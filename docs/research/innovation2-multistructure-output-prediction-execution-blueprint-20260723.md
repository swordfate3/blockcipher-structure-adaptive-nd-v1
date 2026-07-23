# 创新2多结构真实输出预测执行蓝图

日期：2026-07-23

状态：方法级目标冻结 / PRESENT OPF2运行中 / GIFT、SPECK与DES条件实现已准备但性能实验尚未开放

## 1. 目标任务

本研究面向SPN、Feistel和ARX三类分组密码，研究固定未知秘密密钥条件下的多轮真实密文输出预测：

```text
固定未知密钥 K
未见明文 P
真实 r 轮密文 C = E_K^r(P)

输入  = P的公开编码
标签  = C的完整输出、预注册输出bit，或这些输出bit的确定性组合
```

模型不得读取秘密密钥、轮密钥、中间状态或测试标签。每把固定密钥独立训练；跨密钥证据指同一冻结协议在
新密钥上重新训练后复现，不指一个checkpoint零样本预测任意密钥。

主任务不是真假样本、积分平衡、cube/kernel成员、关系成立或密码/随机分类。二元损失和AUC只是在真实
输出bit为`0/1`时使用的评估工具，不能改变标签语义。停止的dense DDT输入路线不恢复；DDT或路径搜索最多
作为候选位置或解释先验，不能替代真实明文到真实密文输出的主任务。

## 2. 三类合法输出目标

| 目标 | 标签 | 主指标 | 使用边界 |
|---|---|---|---|
| 完整输出 | `E_K^r(P)`全部bit | 逐bit AUC/accuracy、完整输出exact match、论文协议指标 | 作为Kimura/Jeong等论文锚点，不要求每个新结构都先恢复完整密文 |
| 预注册选定位 | 测试前冻结的`E_K^r(P)[S]` | 逐bit与平均AUC、majority margin、selected-vector exact match | 当前主方法；发现集和fresh确认集必须分离 |
| 确定性组合 | 例如`xor(E_K^r(P)[S])` | AUC、accuracy、相对组合多数基线 | 仅在强同协议逐bit锚点后测试；不能改写成积分或真假分类 |

默认优先级为“选定位输出 -> 完整输出论文校准 -> 有机制依据的组合输出”。失败的低预算parity配置不能否定
从强逐bit模型派生的组合输出；组合输出也不能取代逐bit主证据。

## 3. 统一数据与公平性合同

每个正式实验必须冻结并记录：

```text
cipher / rounds / key_seed
data seed / model seed / shuffle seed
train / validation / discovery / fresh-test总行数
明文split、逐文件SHA256和零重合证明
目标bit顺序及其冻结时间
epochs / batch / optimizer / loss / checkpoint selection
模型参数量、运行时间和磁盘缓存元数据
```

同一矩阵的候选、通用锚点、错误结构和标签打乱行必须使用相同明文、真实测试标签、初始化规则、batch顺序、
预算和指标。`key_seed`必须与数据/模型seed分离；换新密钥时只改变密钥和由它产生的真实输出。

正式远程数据必须按chunk落盘并支持参数匹配复用/恢复。训练、验证和测试明文不得重合；发现输出位置时，
必须在读取fresh-test标签前冻结位置、数量和排序规则。

## 4. 统一裁决层级

现有OPF2、OPF2-C1和OPF3使用各自已经冻结的门，不得修改。新分支默认预注册以下三层门；若某个论文
协议需要不同指标，只能在揭盲前写入具体实验计划。

### 4.1 输出可预测门

```text
候选平均AUC >= 0.55
候选 - 匹配标签打乱平均AUC >= 0.03
候选平均accuracy-majority >= 0.005
至少一半冻结bit同时满足：
  AUC >= 0.55
  candidate-shuffle >= 0.015
  accuracy-majority >= 0.005
```

### 4.2 结构增益门

```text
结构候选在两把独立固定未知密钥上都优于最强同预算通用锚点
两把密钥合计平均AUC增益 >= 0.005
至少一半冻结bit的增益方向一致
错误结构控制不能取得与正确结构同等的稳定增益
```

错误结构只负责归因；若真实输出候选通过而正确/错误结构持平，可保留“输出可预测”结论，但不能宣称精确
拓扑贡献。

### 4.3 轮数推进门

```text
当前轮两把独立密钥通过 -> 才允许增加一轮
单密钥通过、新密钥失败 -> 保留单密钥条件结果，不增加轮数
首个完整失败轮 -> 只允许一个预注册、同预算、单变量结构假设
结构假设仍失败 -> 冻结最后通过轮/首个失败轮，不做机械扩样或追加epoch
```

### 4.4 网络选择与临界轮汇总合同

每个可比较证据单元冻结为：

```text
(structure, cipher, rounds, target_kind, target_spec,
 train/test budget, epochs, protocol_version)
```

只有同一证据单元内的模型可以直接按数值排名。完整输出、选定位和确定性组合不能互相替代；不同密码、
不同输出宽度或不同预算的AUC/BAPavg也不能直接排成“跨结构排行榜”。跨结构汇总只使用下列共同裁决字段：

```text
output_status            = confirmed / single_key_only / hold / invalid / not_run
candidate_minus_shuffle  = 同任务主指标差值
candidate_minus_generic  = 同预算最强通用锚点差值
candidate_minus_wrong    = 匹配错误结构差值
independent_keys_passed  = 0 / 1 / 2
last_confirmed_round     = 两把密钥通过的最高相邻已测轮
first_complete_hold      = 紧邻last_confirmed_round且基础候选和唯一预注册结构假设均未通过的首轮
```

推荐网络按以下顺序确定：

1. 候选必须先在两把独立固定未知密钥上通过输出可预测门；单密钥通过只能记录条件结果；
2. 在同密码、同轮数、同目标和同预算内，先按具体计划冻结的论文主指标，再按macro/mean AUC比较；
3. 若结构候选通过结构增益门，推荐该密码结构对应的结构网络；
4. 若输出门通过但正确/错误结构持平，只能推荐实测最强通用或“结构归因未定”网络；
5. 若所有有效协议模型都未通过，填写“当前协议未建立可预测网络”，不能填写“密码不可预测”。

临界轮只由相邻轮、同目标、同预算和两密钥证据组成。`last_confirmed_round`不能由低轮论文数值、单密钥、
readiness或事后选bit填充；`first_complete_hold`必须在协议有效、控制完整且唯一预注册结构假设也未通过后填写。
若中间轮未测，临界轮只能写区间，不能跨过缺口取最大轮数。

最终论文表每个密码至少包含：

```text
结构 | 密码 | 输出目标 | 预算 | 独立密钥数 | 推荐网络
最后确认通过轮 | 首个完整hold轮/区间 | generic差值 | wrong差值 | shuffle差值
完整输出exact match或selected-vector exact match | 证据状态 | 产物路径
```

## 5. 当前主干：PRESENT四轮

当前唯一运行中的性能任务是：

```text
run_id  = i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_key7_gpu0_20260722
cipher  = PRESENT-80 / 4轮 / 固定未知key seed7
input   = 64个MSB-first明文bit
target  = [0,2,8,10,32,34,40,42]八个真实密文bit
train   = 2^20 total
test    = 2^16 total
epochs  = 100 / 五模型
status  = running；正式gate未知
```

裁决后只执行一个分支：

```text
OPF2 pass
  -> OPF2-C1：保持data/model/shuffle seed7，只将key_seed改为8
  -> 两把密钥都通过后，才预注册PRESENT r5同预算边界实验

OPF2 hold且协议完整
  -> OPF3：共享四步SPN轮递推器，保持2^20/2^16和100 epochs
  -> 若通过，再只更换独立密钥确认；若失败，关闭PRESENT r4扩展

OPF2 invalid
  -> 只修复OPF2，不运行C1、OPF3、r5或跨密码性能实验
```

OPF3共享轮模型与独立`key_seed`数据合同已经在盲态实现并通过结构单元测试；readiness、生成数据和训练
仍由OPF2正式裁决授权。实现完成只缩短条件分支准备时间，不是PRESENT四轮性能证据。

## 6. SPN分支：PRESENT -> GIFT-64 -> AES

### 6.1 GIFT-64机制确认

研究问题：PRESENT形成的“输出位置发现、位置保持表示和结构控制”流程能否在另一种64-bit真实SPN上复现？

GX1的冻结条件协议见：

```text
docs/experiments/
  innovation2-output-prediction-gx1-gift64-r3-full64-discovery-conditional-plan.md
```

```text
GX0 local readiness：64/64/64，1 epoch，只验证实现
GX1 discovery：GIFT-64 r3，2^17训练、2^16发现、2^16 fresh
GX1 rows：full64通用锚点true + 参数匹配shuffle
唯一变量：密码从PRESENT换为GIFT；不复用PRESENT输出位置
GX2 screen：selected8 MLP / 位置保持ResCNN / GIFT-SPN-aware
GX2 confirm：新固定密钥上的最强通用锚点 / 候选 / 候选shuffle
```

GX1在fresh标签读取前冻结最多八个位置；至少四个位置通过输出门才开放GX2。新密钥不得重选bit。失败时
停止GIFT位置路线，不以更多seed、epoch、候选容量或更低轮数救援同一主张。

条件开放前已经完成GIFT-64真实输出数据适配器、full64 MLP/matched-shuffle发现裁决核心和CLI编排的
确定性单元测试；尚未运行`64/64/64` readiness、生成正式数据或训练。该状态只证明GX1执行链可准备，
不证明GIFT三轮存在易预测位置。

### 6.2 AES最终SPN覆盖

只有GIFT证明方法至少跨两个SPN后才进入AES。AES表示按`4x4 byte state`组织，候选显式分离SubBytes局部
混合、ShiftRows和MixColumns扩散；通用byte-ResCNN为同预算锚点，identity/wrong ShiftRows或错误扩散
顺序为归因控制，另有匹配标签打乱。

冻结条件协议见：

```text
docs/experiments/
  innovation2-output-prediction-aes1-aes128-r1-r2-full-output-conditional-plan.md
```

AES先以`2^17/2^16、100 epochs`定位低轮正控制和边界区间；只在最后通过轮与首个失败轮使用完整结构
矩阵。若边界存在可信数据稀缺解释，才另行预注册`2^20/2^16`单变量规模审判。

## 7. Feistel分支：DES -> SM4

### 7.1 DES机制校准

研究问题：显式保持左右分支角色和轮间交换的共享递推网络，是否比同预算通用网络更能预测真实DES输出？

FEISTEL1的冻结条件协议见：

```text
docs/experiments/
  innovation2-output-prediction-feistel1-des-r2-r3-full-output-conditional-plan.md
```

```text
input：未见64-bit明文，公开IP后保持左右32-bit分支
target：同一输入的完整64-bit真实轮减DES密文
anchors：Jeong论文族FCNN与两半三层BiLSTM
candidate：公开IP/E/P/FP、共享F主体、显式swap的Feistel递推网络
controls：wrong F-input branch / matched label shuffle
initial scale：2^20训练、2^15测试、100 epochs
keys：key_seed31首次矩阵 + key_seed32独立确认
```

先在r2执行FCNN、两半BiLSTM与共享Feistel递推候选的同预算screen，再加入错误F输入和匹配shuffle归因，
并只更换独立密钥确认；r2闭环后才测试r3边界。若`2^20`通用锚点没有恢复公开信号，必须先执行
`2^22/2^15、300 epochs`论文规模校准，不能把小预算失败写成DES二轮不可预测。不得把Innovation 1的
DES真假区分结果当作本任务输出预测证据。

条件开放前已经完成轮减DES标量/NumPy真实输出数据合同、磁盘缓存、FCNN/BiLSTM和正确/错误F输入递推器
的结构测试。正确/错误候选只改变F读取`R_t`或`L_t`；二轮候选`3877312`参数、三轮候选`3890560`
参数，均在BiLSTM `3780672`参数的`±5%`内。matched-shuffle训练、裁决、CLI和readiness仍未实现，
不得把结构测试写成DES输出预测结果。

### 7.2 SM4最终Feistel-like覆盖

SM4必须按四个32-bit字和`X_(i+4)=X_i xor F(X_(i+1),X_(i+2),X_(i+3),rk_i)`组织，不能复用DES
左右两半映射。候选为四字共享轮递推网络；通用word-ResCNN为锚点；错误字序、错误递推依赖和标签打乱
为控制。一至三轮分别保留96、64、32个原始明文直通bit，因此不计作神经输出预测正证据；从首个零直通的
r4开始，以`2^17/2^16、100 epochs`校准，再按统一轮数推进门决定是否进行`2^20`边界审判。

冻结条件协议见：

```text
docs/experiments/
  innovation2-output-prediction-sm4-1-r4-r5-full-output-conditional-plan.md
```

## 8. ARX分支：SPECK32/64

研究问题：显式编码字角色、公开旋转和模加进位传播，能否在真实SPECK输出预测中超过论文协议与通用网络？

ARX1的冻结条件协议见：

```text
docs/experiments/
  innovation2-output-prediction-arx1-speck32-r3-full-output-conditional-plan.md
```

分两步保持矩阵精简：

```text
ARX-A screen（同预算true rows）：
  Jeong论文族FCNN锚点
  Jeong论文族两word BiLSTM锚点
  rotation/carry-aware共享递推候选

ARX-B attribution（只保留最强通用锚点与候选）：
  strongest generic true
  rotation/carry-aware true
  wrong-rotation或carry-ablation true
  candidate matched-shuffle
```

第一阶段使用`2^20/2^15、100 epochs`训练完整32-bit输出，并在第二固定未知密钥上原样确认。候选通过且
新密钥复现后，才允许为了论文可比性另行执行`2^22/2^15`、300 epochs的完整输出协议；该大规模行必须
报告BAPavg、逐bit AUC和完整输出exact match，不能只用selected-bit AUC冒充完整密文复现。若之后需要
selected-bit路线，必须另设discovery/fresh分离计划，不能从本实验测试集事后挑位置。

条件开放前已经完成SPECK32/64真实输出数据、FCNN/BiLSTM、正确/错误rotation-carry候选、完整输出指标、
A1/A2训练与CLI执行链的确定性测试；尚未运行readiness或正式训练。ARX1-C论文规模runner和正式远程包
仍由A/B结果授权，不在揭盲前机械补齐。

## 9. 执行顺序与算力边界

```text
1. 回收并裁决PRESENT OPF2
2. 自动执行OPF2-C1或OPF3唯一授权分支
3. GIFT-64跨SPN方法确认
4. SPECK32/64 ARX论文锚点与结构归因
5. DES Feistel机制校准
6. AES与SM4开题算法最终覆盖
7. 汇总“密码结构 -> 推荐网络 -> 最后通过轮/首个失败轮”选择规则
```

本地只做确定性审计、`64--256`行、1 epoch readiness和绘图验证。`2^17`及以上性能实验使用远程A6000，
必须从推送提交启动，使用`G:\\lxy`磁盘缓存、进度与恢复，由本地tmux watcher自动回收；主任务不SSH轮询。

### 9.1 当前实现与证据状态

| 结构/密码 | 冻结输出任务 | 当前实现 | 当前证据状态 | 下一授权门 |
|---|---|---|---|---|
| SPN / PRESENT-80 | r4八个预注册真实密文bit | OPF2五模型正式矩阵；OPF3共享轮模型盲态就绪；独立`key_seed`就绪 | OPF2远程运行中，正式gate未知 | OPF2 pass走C1；hold走OPF3；invalid只修协议 |
| SPN / GIFT-64 | r3完整64-bit发现后冻结最多八bit并fresh确认 | 数据、matched MLP/shuffle发现核心、候选冻结和CLI编排已测试 | 未运行readiness或性能训练 | PRESENT当前分支闭环 |
| ARX / SPECK32/64 | r3完整32-bit真实密文 | 数据、FCNN、BiLSTM、rotation-carry、wrong rotation、指标和A1/A2执行链已测试 | 未运行readiness或性能训练 | PRESENT与GIFT分支依次闭环 |
| Feistel / DES | r2完整64-bit真实密文，随后r3边界 | 数据缓存、FCNN、BiLSTM、正确/错误F输入共享递推器已测试 | 未实现训练/裁决/CLI，未运行readiness | PRESENT、GIFT与SPECK分支依次闭环 |
| SPN / AES-128 | 条件低轮完整/选定位真实输出 | 仅条件计划和已有密码实现 | 无同任务readiness或性能结果 | GIFT形成跨SPN证据后开放 |
| Feistel-like / SM4 | r4起完整128-bit真实输出 | 仅条件计划和已有密码实现 | 无同任务readiness或性能结果 | DES闭环后开放 |

表中“已测试”均指确定性实现或结构单元测试，不是性能screen。只有本地/远程结果产物通过来源、缓存、指标、
控制和独立密钥门后，才能填入最后通过轮、首个失败轮或推荐网络。

2026-07-23在目录重构完成后重新执行了跨结构条件实现回归：

```text
GIFT-64：数据、完整输出发现/候选冻结、CLI合同
SPECK32/64：数据、通用/ARX结构模型、训练、指标、A1/A2 CLI合同
DES：数据缓存、FCNN/BiLSTM、正确/错误F输入共享递推器合同

result = 56 passed
```

该回归没有生成训练数据、checkpoint、AUC/BAPavg、SVG或可索引结果。它只证明后续条件分支没有因目录
整理而失效；DES的matched shuffle训练、裁决与CLI仍必须等到PRESENT、GIFT和SPECK依次闭环后实现。

## 10. 每个完成实验的必备产物

```text
results.jsonl：逐bit/逐模型正式结果
history.csv：逐epoch训练历史
gate.json：协议、来源、输出、结构和轮数裁决
metadata/cache manifest：密钥、split、hash、缓存与恢复信息
curves.svg：中文可理解的正式图
实验记录：问题、同预算锚点、唯一变量、结果、边界和下一动作
```

正式图必须渲染为像素并通过`visual-qa-redraw`，检查重叠、裁切、标题歧义、曲线可分性、图例和坐标轴。
完成结果在同一轮刷新`outputs/00_RECENT_RESULTS.md/json`；代码、配置、测试和文档范围提交并推送。

## 11. 最终可支持的论文贡献

只有SPN、Feistel和ARX都至少获得一个真实标准密钥算法的同任务、独立密钥证据后，才能形成：

> 固定未知密钥下的多轮真实密文输出预测评估方法；针对SPN、Feistel和ARX构造结构适配网络，并通过
> 通用锚点、错误结构、标签打乱和独立密钥控制，识别易预测输出坐标，测量经验可预测临界轮，形成
> “密码结构—网络架构—可预测轮数”的可执行选择规则。

在此之前，PRESENT、GIFT、DES或SPECK的单项结果只能按各自密码、密钥、轮数、目标和预算报告，不能
外推为整类密码结论、完整密文恢复、高轮攻击突破、SOTA或跨密钥通用模型。

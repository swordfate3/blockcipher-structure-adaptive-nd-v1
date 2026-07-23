# 创新2多结构真实输出预测执行蓝图

日期：2026-07-23

状态：方法级目标冻结 / PRESENT OPF2运行中 / 跨结构性能实验尚未启动

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

OPF3共享轮模型与独立`key_seed`数据合同可以盲态实现和测试，但readiness、生成正式数据和训练仍由OPF2
正式裁决授权。

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
input：未见明文的左右32-bit分支
target：发现后冻结的真实DES密文bit
anchor：普通ResCNN或同预算MLP中的最强者
candidate：双分支、共享轮函数、显式swap的Feistel递推网络
controls：wrong F-input branch / matched label shuffle
initial scale：2^20训练、2^15测试、100 epochs
keys：发现密钥 + 独立确认密钥
```

先在r2执行FCNN、两半BiLSTM与共享Feistel递推候选的同预算screen，再加入错误F输入和匹配shuffle归因，
并只更换独立密钥确认；r2闭环后才测试r3边界。若`2^20`通用锚点没有恢复公开信号，必须先执行
`2^22/2^15、300 epochs`论文规模校准，不能把小预算失败写成DES二轮不可预测。不得把Innovation 1的
DES真假区分结果当作本任务输出预测证据。

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

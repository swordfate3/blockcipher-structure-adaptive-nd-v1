# 创新2 ARX1：SPECK32/64三轮完整输出与结构网络条件计划

日期：2026-07-23

状态：条件计划冻结 / 数据、模型、指标、A1/A2训练与CLI执行链已实现 / 未运行readiness、未训练、未启动

## 1. 条件授权与任务边界

ARX1是创新2的ARX真实输出预测锚点，不是现有SPECK积分平衡、kernel成员或真假差分区分实验的延续。
它只在以下前置路线闭环后开放readiness：

```text
PRESENT当前OPF2条件分支完成并裁决
GIFT GX1及其被授权的GX2分支完成、hold或协议修复闭环
```

GIFT是否成功不决定SPECK是否值得执行；顺序约束只用于避免同时占用远程GPU和混淆三种结构假设。
在ARX1开放前只允许协议审计、确定性数据/网络单元测试和静态配置，不得生成正式数据、运行性能screen或
启动远程任务。

主任务冻结为：

```text
固定但不提供给网络的64-bit秘密密钥 K
输入 = 未见32-bit明文 P
标签 = 同一输入的真实32-bit三轮密文 E_K^3(P)
```

每个输出`0/1`是密文bit的真实数值，不是正负样本类别。历史SPECK Hwang、积分、固定位置关系、kernel或
Innovation 1区分结果均不能作为本计划的输出预测来源、基线或成功证据。

## 2. 已核验论文锚点

Jeong等，*Comprehensive Neural Cryptanalysis on Block Ciphers Using Different Encryption Methods*，
Mathematics 2024，DOI `10.3390/math12131936`，开放全文已经本地核验。其Encryption Emulation任务与
本计划语义一致：明文bit输入，完整真实密文bit输出。

SPECK32/64相关协议为：

```text
train / test       = 2^22 / 2^15 total pairs
rounds             = 3
models             = FCNN / three-layer BiLSTM-256
BiLSTM input       = 两个16-bit半块组成长度2序列
epochs / batch     = 300 / 128
loss / optimizer   = BCE / AdamW
learning rate      = 0.001
metric             = 完整输出平均逐bit准确率 BAPavg
Table 3 r3 EE      = FCNN 0.587 / BiLSTM 0.883
```

2024数据生成段没有单独把密钥冻结规则写得足够精确；2026同作者后续论文明确把论文族描述为固定未知秘密
密钥KPA。因而本计划使用显式固定未知密钥，但只能称“Jeong论文族协议校准”，不能声称逐细节完全复现。

Jeong结果没有标签打乱、错误旋转、进位消融或ARX结构网络。使用FCNN、BiLSTM、逐bit准确率或完整输出
本身都不是本项目的新意；可检验增量是正确ARX算子相对通用锚点和错误结构控制的受控增益。

## 3. 为什么分成同预算screen与论文规模确认

直接把五个模型都放到`2^22/300 epochs`会一次改变架构、控制和巨大计算预算，也会浪费失败路线的GPU。
只在`2^17/100 epochs`失败又不足以反驳已发表的`2^22/300`正结果。因此ARX1分成：

```text
ARX1-A：2^20/2^15、100 epochs，同预算架构和归因screen
ARX1-B：第二固定未知密钥，同一ARX1-A协议确认
ARX1-C：只有A/B通过后，2^22/2^15、300 epochs论文规模比较
ARX1-R：只有r3两密钥通过后，按相同预算测试r4临界轮
```

ARX1-A失败只能关闭当前`2^20/100`结构候选，不能写成SPECK三轮或Jeong论文规模失败。若所有通用锚点
也接近随机，先审计实现与论文差异；不得用更多模型名、事后调epoch或选择容易bit绕过门。

## 4. 冻结数据与密钥协议

ARX1-A：

```text
cipher                  = SPECK32/64
rounds                  = 3
block / key             = 32 / 64 bits
data/model/shuffle seed = 21
key_seed                = 21
secret key derivation   = Random(1_210_000 + key_seed).getrandbits(64)
train plaintext RNG     = numpy.default_rng(1_220_000 + seed)
test plaintext RNG      = numpy.default_rng(1_230_000 + seed)
input                    = 32个MSB-first明文bit，并保留(x,y)两个16-bit字角色
target                   = 32个MSB-first真实密文bit
train / test             = 2^20 / 2^15 total
epochs / batch           = 100 / 128
loss / optimizer / lr    = BCE / AdamW / 0.001
selection                = final epoch
data chunk rows          = 4096
sample classification    = false
```

ARX1-B保持全部字段，只将`key_seed=22`。明文、特征、初始化、batch顺序和shuffle置换必须逐值相同；仅秘密
密钥和由它生成的真实密文targets改变。

ARX1-C保持`seed=21, key_seed=21`和测试集，唯一预算变化为：

```text
train rows = 2^22
epochs     = 300
```

原`2^20`训练明文必须是`2^22`训练集前缀，测试明文必须保留且不能进入扩展训练段。测试集与训练集逐值
零重合。ARX1-C只用于论文规模比较，不替代ARX1-B的独立密钥证据。

训练与测试使用相互独立的确定性明文RNG流。生成时先冻结并持久化测试明文保留集，再让所有训练规模都排除
同一保留集；磁盘数组布局仍保持`train || test`。这样ARX1-C扩展训练前缀时，测试明文可以逐值保持不变且
不会被扩展训练段吞入。仅使用单个连续流，或虽使用两条流但先生成训练再对测试避碰，都无法同时满足这些
约束。两条RNG流的状态与训练/测试完成行数必须分别写入metadata并独立恢复。

所有正式数据按chunk写入`plaintexts.npy/features.npy/full_targets.npy/cache_metadata.json`，metadata冻结
cipher、rounds、两个seed、64-bit key hex、MSB顺序、两个16-bit字角色、split、completed train/test rows、
两条RNG state和参数匹配恢复。

## 5. 三个网络家族

### 5.1 论文FCNN锚点

```text
32 -> 512 -> 1024 -> 512 -> 32 sigmoid logits/output
```

实现必须按论文正文核对激活、归一化和输出定义；不能仅凭层宽猜测后称精确复现。未明确细节写入
`paper-family approximation`字段。

### 5.2 论文BiLSTM锚点

```text
输入reshape为 [batch, 2 words, 16 bits]
三层bidirectional LSTM，hidden size 256
最终映射到32个真实密文输出bit
```

它是当前最强外部同任务锚点。不得把普通bit序列LSTM、PRESENT六层LSTM或现有SPECK差分LSTM当作等价实现。

正文明确三层、每方向hidden 256、两半长度2输入和Sigmoid输出，但没有说明如何把顶层双向序列压缩到
32-bit输出头。当前实现采用顶层正向/反向最终hidden拼接，并在协议元数据中把该选择、权重初始化和
checkpoint selection标为论文未说明字段。因此它是可审计的`Jeong 2024 paper-family approximation`，
不是逐细节精确复现。

### 5.3 rotation/carry-aware候选

候选只测试一个ARX机制假设：每一步保持左右16-bit字角色，显式执行公开旋转坐标，并使用共享的可训练
carry-scan单元近似模加进位传播；XOR融合后递推三次。每轮仅允许独立位置上下文吸收固定未知轮密钥差异，
共享主体不读取密钥或中间真实状态。

冻结正确结构：

```text
x route = ROR7(x)
y route = ROL2(y)
mix     = modular-add carry scan + XOR fusion
steps   = 3 shared recurrent applications
head    = 32个位置绑定输出logit
```

冻结错误旋转控制使用`ROR5 / ROL6`。它保持第一个旋转与16互素、第二个旋转与16的最大公因数为2，避免
identity控制带来的明显混合强度差异。错误行与正确行参数量、初始化和算子次序完全相同，只替换旋转常数。

候选总参数量在实现时预注册并限制为BiLSTM锚点的`±5%`；若精确结构无法在该范围内实现，先更新计划解释
容量差异，不得根据性能揭盲后调宽度。

当前静态实现采用`channels=400`。每轮先对MSB-first `x` token执行公开ROR，再把`ROR(x) || y`按
LSB到MSB顺序送入共享单层GRU carry-scan；加法融合器读取`ROR(x)、y、carry`，三轮各自只有16个x位置
context用于吸收固定未知轮密钥差异。随后对y执行公开ROL，并用共享XOR融合器与新x结合。三步复用同一组
carry/addition/XOR权重，最终使用32组位置绑定输出权重。模型不读取秘密密钥或真实中间状态。

## 6. 分阶段精简矩阵

### ARX1-A1：三行true架构screen

```text
speck32_full32_fcnn_true_output
speck32_full32_bilstm_true_output
speck32_full32_rotation_carry_true_output
```

只选择最强通用锚点，不在A1宣称结构归因。

### ARX1-A2：两条新增归因控制

复用A1完全相同的候选checkpoint和正式结果，只新增：

```text
speck32_full32_wrong_rotation_carry_true_output
speck32_full32_rotation_carry_label_shuffle
```

逻辑矩阵共五行，但正确候选不重复训练。shuffle只打乱训练标签行，测试标签始终是真实密文；wrong rotation
只改变`7/2 -> 5/6`。若A1来源、checkpoint或hash不完整，A2不得运行。

### ARX1-B：独立密钥确认

新密钥只运行四行：

```text
strongest_generic_true
rotation_carry_true
wrong_rotation_carry_true
rotation_carry_label_shuffle
```

最强通用锚点在A1的预注册选择规则为：先比较完整输出`BAPavg`，再比较macro AUC，仍相同时优先论文BiLSTM；
不得读取A2结果后更换锚点。

### ARX1-C：论文规模比较

只运行：

```text
strongest_generic_true
rotation_carry_true
rotation_carry_label_shuffle
```

wrong rotation的两密钥归因已经在A/B回答，不在论文规模重复消耗GPU。

## 7. 指标与裁决门

每行必须报告：

```text
32个逐bit threshold accuracy / majority accuracy / accuracy-majority
32个逐bit AUC和MSE/BCE
BAPavg = 32个逐bitthreshold accuracy的平均
macro AUC
完整32-bit exact-match count/rate
参数量、训练时间、最终epoch和无效输出率
```

ARX1-A/B输出可预测门：

```text
候选BAPavg >= 0.55
候选macro AUC >= 0.55
候选 - matched shuffle BAPavg >= 0.03
候选 - matched shuffle macro AUC >= 0.03
至少16/32 bit同时满足：
  AUC >= 0.55
  accuracy-majority >= 0.005
  candidate-shuffle AUC >= 0.015
```

结构增益门：

```text
候选在key_seed21和22上都优于各自最强通用锚点
两密钥合计候选 - generic BAPavg >= 0.005
至少16/32 bit的候选相对generic增益方向一致
正确旋转在两密钥上都优于wrong rotation
两密钥合计correct - wrong BAPavg >= 0.003
```

输出可预测门可以在精确结构归因失败时单独成立；此时只能保留通用/候选输出信号，不得宣称正确SPECK
旋转提供独特增益。

ARX1-C论文规模门：

```text
来源、预算和论文字段完整
matched shuffle保持接近无信息基线
报告FCNN/BiLSTM相对公开0.587/0.883的绝对差值
候选相对同预算最强generic不退化，或保持ARX1-A/B的正增益
```

公开数值只作参考，不作为协议有效性的硬等式；单密钥实现即使达到`0.883`也不是论文多场景或逐细节复现。

## 8. 临界轮推进

ARX1-A/B两把密钥都通过后，下一步只把`rounds=3 -> 4`，其余使用ARX1-A预算和四行确认矩阵。若r4
失败，则SPECK经验边界记为“最后确认通过r3 / 首个完整失败r4”。

若r3未通过但协议有效，向下测试r2；若r2仍未通过，再测r1。这样定位最后通过轮，不同时枚举r1--r4。
任何轮数只在两把密钥确认后推进；单密钥通过不增加轮数。

允许在完整32-bit训练结果出来后把逐bit排名作为分析，但selected-bit新实验必须使用独立discovery/fresh
分离计划，测试前冻结位置，并重新加入匹配shuffle。不得从ARX1测试集事后挑bit再把AUC写成确认结果。

## 9. readiness与远程执行

条件开放后，本地CPU只运行：

```text
train / test = 64 / 64
epochs       = 1
models       = A1三行 + A2两行
```

readiness验证官方SPECK32/64向量`0x6574694C -> 0xA86842F2`、32-bit MSB顺序、两个16-bit字角色、真实
密文回放、固定密钥、split零重合、匹配shuffle、wrong rotation唯一变化、参数门、checkpoint、恢复和结果
行数；随机小样本指标不解释。

正式`2^20`和`2^22`全部使用远程A6000、推送提交、run-owned干净clone和`G:\\lxy`磁盘缓存。启动后只
做一次有界确认，随后由本地tmux watcher等待、校验并回收。正式SVG必须中文解释协议、完整输出指标、逐bit
分布和控制差值，并通过`visual-qa-redraw`像素检查。

## 10. 实现缺口与停止边界

当前仓库已有正确`Speck32_64`标量实现、官方向量测试及多种SPECK差分/积分工具。现已新增本计划专用的
固定未知密钥真实输出数据适配器：

```text
src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_data.py
tests/test_innovation2_speck32_output_prediction_data.py
```

它冻结ARX1-A/B/C的规模与key_seed，使用`Random(1_210_000 + key_seed)`生成一把64-bit秘密密钥，将每个
32-bit明文按MSB-first编码为`x_msw || y_lsw`两个16-bit字，并把同一明文的真实三轮SPECK32/64密文作为
32-bit标签。缓存先冻结测试保留集，再生成排除该集合的训练前缀；磁盘逐chunk保存训练/测试各自的完成行数
和RNG state，参数不匹配、缺失metadata或部分数组均fail closed。

确定性测试覆盖：

```text
官方22轮向量0x6574694C -> 0xA86842F2
MSB-first明文/密文回放和x/y字角色
训练、测试内部唯一且零重合
只更换key_seed时明文/features逐值相同、真实targets改变
小训练集是扩展训练集前缀且测试集逐值不变
扩展训练段与冻结测试集零重合
完成缓存复用、参数错配和部分缓存拒绝
测试保留集首个chunk后故障注入、双RNG状态恢复与完整数组逐值复现
```

验证命令与结果：

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_innovation2_speck32_output_prediction_data.py \
  tests/test_innovation2_speck_hwang_parity.py
22 passed

UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_data.py \
  tests/test_innovation2_speck32_output_prediction_data.py
All checks passed
```

现已按2024正文实现两个通用锚点及协议元数据：

```text
src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_models.py
tests/test_innovation2_speck32_output_prediction_models.py

FCNN参数量   = 1,087,520
BiLSTM参数量 = 3,731,488
```

FCNN严格采用正文的`32 -> 512 -> 1024 -> 512 -> 32`，三个隐藏层各接BatchNorm与ReLU，输出使用
Sigmoid。BiLSTM把MSB-first输入重排为`[batch, 2, 16]`的`x_msw || y_lsw`字序列，采用三层双向
LSTM、每方向hidden 256和32维Sigmoid输出头。模型协议直接声明真实密文输出任务、BCE/AdamW/0.001计划、
DOI和`paper_exact_reproduction=false`；未报告的初始化、checkpoint选择、dropout及BiLSTM序列压缩不被
伪装成论文原样实现。

联合验证：

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_innovation2_speck32_output_prediction_data.py \
  tests/test_innovation2_speck32_output_prediction_models.py \
  tests/test_innovation2_speck_hwang_parity.py
29 passed

UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_data.py \
  src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_models.py \
  tests/test_innovation2_speck32_output_prediction_data.py \
  tests/test_innovation2_speck32_output_prediction_models.py
All checks passed
```

现已新增独立的SPECK32概率输出评估原语：

```text
src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_metrics.py
tests/test_innovation2_speck32_output_prediction_metrics.py
```

它按论文公式冻结`p <= 0.5 -> 0, p > 0.5 -> 1`，而不是沿用其他分类器的`>= 0.5`规则。每个MSB
索引`0..31`报告整数bit、`x_msw/y_lsw`字角色、accuracy、majority、accuracy-majority、AUC、BCE、MSE
和非法概率率；完整输出摘要报告BAPavg、macro AUC、BCE/MSE、majority BAPavg及32-bit exact-match
count/rate。标签必须是非空、有限的`[rows, 32]`二值真实输出数组，错误形状、NaN或非二值标签fail closed。

数据、模型与指标联合验证：

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_innovation2_speck32_output_prediction_data.py \
  tests/test_innovation2_speck32_output_prediction_models.py \
  tests/test_innovation2_speck32_output_prediction_metrics.py \
  tests/test_innovation2_speck_hwang_parity.py
37 passed
```

现已实现正确/错误rotation/carry静态模型合同：

```text
src/blockcipher_nd/tasks/innovation2/speck32_rotation_carry_model.py
tests/test_innovation2_speck32_rotation_carry_model.py

channels                     = 400
candidate parameters         = 3,732,032
BiLSTM anchor parameters     = 3,731,488
candidate / BiLSTM ratio     = 1.0001457863
within BiLSTM +/-5%          = true
correct rotations            = ROR7 / ROL2
wrong rotations              = ROR5 / ROL6
```

正确与错误模型在相同seed下所有参数和buffer逐值相同，旋转常数不进入state dict；错误行只替换两个公开旋转
常数。匹配shuffle行复用正确结构，只在训练时固定置换训练标签，测试仍使用真实SPECK输出。测试还把
`0x1234`的MSB-first token旋转结果还原为整数，并与项目标量`ror/rol`逐值比较，避免张量方向自洽但密码
语义写反。

全部ARX1静态单元联合验证：

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_innovation2_speck32_output_prediction_data.py \
  tests/test_innovation2_speck32_output_prediction_models.py \
  tests/test_innovation2_speck32_output_prediction_metrics.py \
  tests/test_innovation2_speck32_rotation_carry_model.py \
  tests/test_innovation2_speck_hwang_parity.py
47 passed
```

现已实现A1/A2训练、checkpoint、来源与裁决统一执行链：

```text
src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_training.py
src/blockcipher_nd/cli/run_innovation2_speck32_output_prediction.py
scripts/run-innovation2-speck32-output-prediction
tests/test_innovation2_speck32_output_prediction_training.py
tests/test_innovation2_speck32_output_prediction_cli.py
```

执行链现在冻结并验证以下合同：

```text
A1仅训练FCNN、BiLSTM和正确rotation/carry三个true-output模型
A2要求A1 bundle、来源manifest和final checkpoint逐项SHA256匹配
A2复用A1正确候选结果/checkpoint，只新增wrong rotation和matched shuffle
正确、错误和shuffle候选的initial state SHA256、参数量和batch schedule必须一致
shuffle采用固定非identity训练标签置换；测试始终使用真实SPECK密文标签
每个模型保存latest/final checkpoint、config/source/state/schedule hash
中断恢复要求history epoch从1连续到最终epoch
正式AdamW weight_decay显式冻结为0.01，并标记为论文未报告字段
readiness缩小candidate channels只验证执行；正式channels=400参数门另行静态验证
```

统一命令支持`readiness`和冻结的`arx1_a`两种模式，输出`results.jsonl`、`history.csv`、`gate.json`、
`metadata.json`、source/A1/A2 bundle、checkpoint manifest和模型协议。ARX1-A单密钥裁决预先实现输出门、
最强通用锚点冻结、论文协议未校准分支和当前候选停止分支；随机readiness指标不参与性能判断。

联合验证：

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_innovation2_speck32_output_prediction_data.py \
  tests/test_innovation2_speck32_output_prediction_models.py \
  tests/test_innovation2_speck32_output_prediction_metrics.py \
  tests/test_innovation2_speck32_rotation_carry_model.py \
  tests/test_innovation2_speck32_output_prediction_training.py \
  tests/test_innovation2_speck32_output_prediction_cli.py \
  tests/test_innovation2_speck_hwang_parity.py
56 passed

UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_data.py \
  src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_models.py \
  src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_metrics.py \
  src/blockcipher_nd/tasks/innovation2/speck32_rotation_carry_model.py \
  src/blockcipher_nd/tasks/innovation2/speck32_output_prediction_training.py \
  src/blockcipher_nd/cli/run_innovation2_speck32_output_prediction.py \
  tests/test_innovation2_speck32_output_prediction_data.py \
  tests/test_innovation2_speck32_output_prediction_models.py \
  tests/test_innovation2_speck32_output_prediction_metrics.py \
  tests/test_innovation2_speck32_rotation_carry_model.py \
  tests/test_innovation2_speck32_output_prediction_training.py \
  tests/test_innovation2_speck32_output_prediction_cli.py
All checks passed

UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/run-innovation2-speck32-output-prediction --help
exit 0
```

本次没有调用runner，没有生成readiness、性能screen或正式数据，因此没有可索引实验结果、SVG或可解释
指标，也不触发结果索引和`visual-qa-redraw`。当前推荐动作仍是保持ARX1性能执行关闭：先回收并裁决
PRESENT OPF2，再完成其唯一授权分支和GIFT GX1/GX2闭环；之后才运行本地`64/64、1 epoch` ARX1
readiness。readiness全部协议门通过后，才从推送提交在远程A6000运行key_seed21的`2^20/2^15、100
epochs`五行逻辑矩阵；单密钥通过后只更换`key_seed=22`确认。当前不得提前实现依赖A1性能选择的ARX1-B
最终四行编排，不得生成正式SPECK数据或训练模型。

仍未实现且当前未获执行授权的部分为：

```text
ARX1-B基于A1最强generic冻结结果的四行独立密钥runner
ARX1-C 2^22/300论文规模条件runner
正式远程启动/监控/回收包
正式结果SVG及visual-qa-redraw
```

明确禁止：

```text
不得复用积分/kernel标签或真假差分数据
不得把2^20失败写成2^22/300论文规模失败
不得根据A1结果新增LSTM、Transformer或调候选宽度
不得在同一实验同时改轮数、数据量、epoch和模型
不得用随机密文、变化密钥混合训练或测试集挑bit
不得在两密钥r3通过前进入r4
不得把BAPavg与PRESENT selected-bit AUC直接拼成SOTA比较
```

## 11. 结果闭环

每个完成阶段必须生成JSONL、history CSV、gate、metadata/cache manifest、checkpoint manifest和SVG；更新
本记录中的运行ID、来源、指标、差值、claim scope和下一动作，刷新`outputs/00_RECENT_RESULTS.md/json`，
完成像素级可视化检查、范围提交和推送。

最终允许的ARX结论强度依次为：

```text
单密钥输出信号
-> 两密钥完整输出预测确认
-> 正确rotation/carry相对通用与wrong结构的两密钥增益
-> r3/r4经验临界轮
-> 2^22/300论文规模相对比较
```

任何较低层证据都不能提前表述为跨ARX通用方法、完整攻击、论文复现或SOTA。

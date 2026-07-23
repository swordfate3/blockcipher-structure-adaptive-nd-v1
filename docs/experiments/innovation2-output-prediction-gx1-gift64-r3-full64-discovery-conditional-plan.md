# 创新2 GX1：GIFT-64三轮全64-bit输出位置发现条件计划

日期：2026-07-23

状态：条件计划冻结 / GIFT数据与发现裁决核心单元测试完成 / 未授权readiness、性能训练或远程启动

## 1. 条件授权

GX1是创新2从PRESENT扩展到第二个真实SPN的第一项同任务实验，不是当前OPF2的并行模型搜索。只有
PRESENT四轮当前分支按冻结裁决树闭环后才开放GX0/GX1 readiness：

```text
OPF2 pass -> OPF2-C1完成并裁决后开放
OPF2 hold -> OPF3完成并裁决；若OPF3通过还须完成新密钥确认后开放
OPF2 invalid -> 保持关闭，先修复OPF2
```

PRESENT分支闭环不要求四轮必须成功。若四轮最终hold，PRESENT三轮已完成的真实输出正证据仍可作为
方法来源；GX1负责检验流程是否跨到GIFT，不能用GIFT结果回写或救援PRESENT四轮。

在条件开放前只允许：

1. 编写GIFT输出预测数据/来源适配器和确定性单元测试。
2. 复用现有64-bit MLP、逐bit指标和候选冻结原语。
3. 准备配置、来源门和本计划要求的静态验证。

不得生成GX1正式数据、运行readiness、训练模型或启动远程任务。

## 2. 研究问题与唯一变化

研究问题：

> 在固定未知128-bit秘密密钥下，使用与PRESENT输出位置发现相同的真实明文到真实密文任务、数据预算、
> 优化器、损失和fresh冻结协议，GIFT-64三轮是否存在至少四个能在全新明文上复现的易预测输出bit？

相对PRESENT方法锚点，唯一研究变量是密码算法及其真实输出：

```text
PRESENT-80 r3 -> GIFT-64 r3
80-bit key     -> 128-bit key
PRESENT S/P    -> GIFT S/P和key schedule
```

不复用PRESENT的输出位置、密钥、checkpoint、P层映射或候选排名。GIFT必须重新扫描全部64个真实输出bit。
GX1不是结构网络比较；GIFT-SPN-aware、错误P和独立密钥架构归因属于候选通过后的GX2。

## 3. 冻结数据协议

```text
cipher                  = GIFT-64
rounds                  = 3
block / key             = 64 / 128 bits
data/model/shuffle seed = 11
key_seed                = 11
secret key derivation   = Random(1_110_000 + key_seed).getrandbits(128)
input                    = 64个MSB-first明文bit
target                   = 同一明文的64个MSB-first真实三轮密文bit
sample classification   = false

train rows               = 131072 = 2^17 total
discovery rows           = 65536  = 2^16 total
fresh rows               = 65536  = 2^16 total
epochs / batch           = 100 / 250
optimizer / lr           = RMSprop / 0.001
loss                     = raw-output MSE
checkpoint selection     = final epoch
data chunk rows          = 4096
```

source数据顺序冻结为`train || discovery`。fresh使用独立明文RNG流，且必须与source全部`196608`条明文
逐值零重合。三组明文内部也必须唯一。正式缓存必须逐chunk写入：

```text
data/plaintexts.npy
data/features.npy
data/full_targets.npy
data/cache_metadata.json
fresh_data/plaintexts.npy
fresh_data/features.npy
fresh_data/full_targets.npy
fresh_data/cache_metadata.json
```

metadata必须冻结cipher、rounds、两个seed、128-bit密钥hex、bit顺序、split、completed_rows、RNG state和
参数匹配复用。任何字段不匹配时拒绝复用，不得覆盖已有缓存。

## 4. 两行发现矩阵

GX1只比较一个通用全输出模型与它的匹配标签控制：

```text
gift64_full64_mlp_true_output
  64 -> 1936 -> 1936 -> 64
  真实训练标签

gift64_full64_mlp_label_shuffle
  完全相同的网络、初始化、参数量、batch顺序和测试标签
  只对训练标签行做固定随机置换
```

不沿用OP10中“MLP真值相对LSTM shuffle”的跨架构控制。GX1的shuffle必须与MLP架构匹配，避免把模型差异
误写成输出信号。两行测试始终使用未打乱的真实GIFT密文输出。

## 5. 候选冻结与fresh门

训练结束后，先在discovery split计算每个MSB索引`0..63`：

```text
AUC
threshold accuracy
majority accuracy
accuracy-majority
true AUC - matched-shuffle AUC
MSE
```

每个bit的保守选择分数为：

```text
min(AUC - 0.5, accuracy-majority, true AUC - shuffle AUC)
```

discovery资格门：

```text
AUC >= 0.510
accuracy-majority >= 0.005
true-shuffle AUC >= 0.005
```

按保守分数、AUC和MSB索引确定性排序，最多冻结八个候选。必须先写出并哈希：

```text
candidates.json
candidates.sha256
event = candidates_frozen_before_fresh_generation
```

然后才允许生成和读取fresh数据。每个冻结候选在fresh上使用相同三门；至少`4/8`冻结候选通过才裁决：

```text
status   = pass
decision = innovation2_gift64_r3_true_output_bits_fresh_confirmed
next     = GX2 selected8 architecture screen
```

若少于四个通过：

```text
status   = hold
decision = innovation2_gift64_r3_true_output_bits_not_confirmed
next     = close GIFT output-position route and update cross-SPN boundary
```

任一来源、bit顺序、密钥、split、缓存、checkpoint、候选冻结顺序或结果行门失败均为`protocol_invalid`，只
修协议，不解释性能。

## 6. 输出与完整密文边界

GX1必须同时报告全64-bit模型的：

```text
macro per-bit AUC
bit match
full-output exact match count/rate
逐bit discovery/fresh指标
```

候选门只支持“GIFT-64三轮存在fresh确认的选定位真实输出预测信号”。即使出现完整密文exact match，也只能
按本实验单密钥、规模和协议报告；GX1不支持跨密钥、结构模型增益、四轮、完整攻击或SOTA主张。

## 7. 已审计的实现边界

现有代码不能直接改密码名复用：

```text
output_prediction_kimura_lstm.py
  数据生成、128/80-bit密钥、scalar replay和official vector写死PRESENT

output_bit_discovery.py
  source门、fresh加密、claim scope和模型名称写死PRESENT/OP9
```

允许复用：

```text
ParameterMatchedOutputMlp
full_output_metrics和训练循环的数学定义
per_bit_metric_rows
候选保守排序原则
GIFT官方零向量与现有vectorized encrypt_gift_words
```

实现采用窄的GIFT适配器，不在GX1中把PRESENT、GIFT、SPECK、AES一次性重构为通用框架。新适配器必须
显式声明`cipher=GIFT-64`和128-bit密钥，使用现有向量化GIFT实现生成chunk，并逐chunk抽样与标量
`Gift64.encrypt`回放。不得修改历史OP9/OP10缓存序列化、checkpoint hash或结果语义。

## 8. readiness与远程路径

条件开放后先运行本地CPU readiness：

```text
train / discovery / fresh = 64 / 64 / 64
epochs / batch            = 1 / 32
models                    = 2
expected training rows    = 2
expected history rows     = 2
expected checkpoints      = 2
expected per-bit rows     = 64 * 2 * 2 = 256
```

readiness只验证官方向量、向量/标量一致、128-bit密钥、MSB顺序、真实标签、三split零重合、匹配shuffle、
候选先冻结、磁盘恢复、产物和中文SVG；随机小样本指标不解释。SVG必须渲染为像素并通过
`visual-qa-redraw`。

readiness通过后范围提交并推送，从精确推送提交在A6000 GPU0启动正式GX1。远程run root、数据缓存、
checkpoint、progress和结果必须位于`G:\\lxy`。启动后只做一次有界确认，随后交给本地tmux watcher自动
等待、验证、回收、绘图和索引。

## 9. 明确禁止

```text
不得在PRESENT当前分支闭环前运行GX1 readiness或性能训练
不得复用PRESENT八个输出位置或checkpoint
不得在fresh结果后更换候选、数量、排序或阈值
不得用跨架构shuffle替代匹配MLP shuffle
不得因为GX1 hold追加2^20、300 epochs、更多seed或降低到r2救援
不得把历史GIFT积分/关系标签或Innovation 1真假区分结果写成真实输出证据
不得把单密钥selected-bit结果写成完整密文恢复或跨SPN结构结论
```

## 10. 完成后的下一动作

```text
GX1 pass
  -> 预注册GX2：selected8 MLP / 位置保持ResCNN / GIFT-SPN-aware同预算screen
  -> 冻结最强候选后，在key_seed=12且data/model/shuffle seed仍为11的新密钥上确认

GX1 hold
  -> 关闭GIFT输出位置路线
  -> 保留PRESENT适用边界，继续ARX/SPECK独立同任务校准

GX1 invalid
  -> 只修复GIFT数据、来源、冻结或结果协议
```

任何完成结果都必须更新本记录、生成JSONL/CSV/SVG/gate、刷新`outputs/00_RECENT_RESULTS.md/json`，并给出
由门和控制直接支持的下一动作。

## 11. 条件开放前的实现进度

已新增窄范围GIFT真实输出数据适配器：

```text
src/blockcipher_nd/tasks/innovation2/gift64_output_prediction_data.py
tests/test_innovation2_gift64_output_prediction_data.py
```

并新增两行匹配MLP训练、逐bit发现、候选哈希冻结和fresh `4/8`裁决核心：

```text
src/blockcipher_nd/tasks/innovation2/gift64_output_prediction_discovery.py
tests/test_innovation2_gift64_output_prediction_discovery.py
```

已验证：

```text
正式GX1数据字段和2^17/2^16/2^16规模冻结
128-bit key_seed独立于明文seed
只更换key_seed时plaintext/features逐值相同、真实targets改变
GIFT官方零向量及source/fresh标量真实密文回放
source train/discovery唯一且零重合
fresh在candidate SHA256冻结后生成并与全部source明文零重合
磁盘分块缓存完成态复用与参数不匹配拒绝
```

验证命令和结果：

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_innovation2_gift64_output_prediction_data.py
5 passed

UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  src/blockcipher_nd/tasks/innovation2/gift64_output_prediction_data.py \
  tests/test_innovation2_gift64_output_prediction_data.py
All checks passed

UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_innovation2_output_prediction_kimura_lstm.py \
  tests/test_innovation2_output_bit_discovery.py \
  tests/test_innovation2_gift64_output_prediction_data.py \
  tests/test_innovation2_gift64_output_prediction_discovery.py
26 passed

UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  src/blockcipher_nd/tasks/innovation2/gift64_output_prediction_data.py \
  src/blockcipher_nd/tasks/innovation2/gift64_output_prediction_discovery.py \
  tests/test_innovation2_gift64_output_prediction_data.py \
  tests/test_innovation2_gift64_output_prediction_discovery.py
All checks passed
```

发现核心的微型训练测试只使用人工`8x64`数组，没有生成GIFT样本或解释指标。这些是数据与裁决核心的单元
测试；候选SHA256由规范JSON内容重新计算，错误摘要和缺失初始化公平性证据均fail closed。这些不是GX0
readiness或GX1性能结果。命令行编排、正式配置、SVG和远程启动包仍未实现；它们继续由本计划第1节的
PRESENT分支闭环条件授权。

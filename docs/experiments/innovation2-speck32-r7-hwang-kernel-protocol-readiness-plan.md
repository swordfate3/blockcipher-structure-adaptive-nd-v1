# 创新2 E25：SPECK32/64 Hwang Table 7 kernel 协议与分块执行就绪计划

日期：2026-07-17
状态：E27位置族过窄 / E28不执行 / E27-N实现与本地readiness通过 / 待远程精确枚举

## 1. 路线来源

E24 在 SKINNY-64/64 7轮找到4个完全稳定且不同的18维位置 kernel，但只有
`4/16` 个结构非平凡，未达到冻结的 `6/16` 标签宽度门。按 E24 计划停止 SKINNY
位置几何，转向 Hwang et al. 2026 Section 6.2 / Table 7 的另一个正文主案例：
SPECK32/64 的线性输出平衡 kernel。

这仍然保持创新2任务定义：

```text
input  = cipher / rounds / integral input structure / fixed context / output mask
target = output mask是否属于跨密钥稳定平衡kernel
```

不是把完整积分 multiset 与随机 multiset 做二分类。

## 2. 一手来源核验

Hwang 的文献[32]已从 TOSC 出版页面核实并下载：

```text
Senpeng Wang, Bin Hu, Jie Guan, Kai Zhang, Tairong Shi
Exploring Secret Keys in Searching Integral Distinguishers Based on Division Property
IACR Transactions on Symmetric Cryptology, 2020(3), 288-304
canonical landing DOI = 10.46586/tosc.v2020.i3.288-304
PDF embedded legacy DOI = 10.13154/tosc.v2020.i3.288-304
```

权威本地文件：

```text
papers/innovation_two/pdf/2020_wang_exploring_secret_keys_integral.pdf
papers/innovation_two/text/2020_wang_exploring_secret_keys_integral.txt
sources/research_innovation2_speck_wang_secret_keys_landing_20260717.html
```

出版页面标题、作者、年份、主题和 Hwang bibliography 一致。PDF 为17页、未加密，
正文 Section 4.1 / Table 2 给出缺失的输入结构。

## 3. 已解决的协议字段

Wang 2020 明确写明：

```text
input variables x^0_5 and x^0_6 are constant
all other 30 plaintext bits are active
table form = (aaaaaaaaaaaaaaaa, aaaaaaaaaccaaaaa)
```

其位编号定义为 `x^i_31 ... x^i_0`，所以两个固定位置是整个32-bit plaintext 的
LSB-first integer bits `5` 和 `6`；并不是每个16-bit word各固定一个 bit。项目
`Speck32_64` 的 plaintext 布局为高16 bit `x`、低16 bit `y`，与表格左右字顺序
一致。因此冻结：

```text
active bit mask = 0xFFFFFF9F
fixed bit mask  = 0x00000060
fixed contexts  = bits(6,5) in {00,01,10,11}
primary paper context = 00 unless an exact author artifact specifies otherwise
output bit order = LSB-first; paper bit b maps directly to integer bit (1 << b)
```

Table 2 的 `c` 表示任意固定常量而不是必须为零。主复现先用 `00`，后续只有在主
anchor 通过时才把四种 fixed contexts 作为结构族变量。

## 4. 论文输出 kernel anchors

Hwang Table 7 冻结的标准轮边界结果：

```text
6-round rank/nullity expected = 23/9
6-round basis, LSB-first:
  b2 xor b18
  b3 xor b19
  b4 xor b20
  b5 xor b21
  b6 xor b22
  b7 xor b23
  b8 xor b24
  b9 xor b25
  b16

7-round rank/nullity expected = 31/1
7-round basis, LSB-first:
  b2 xor b9 xor b16 xor b18 xor b25
```

Wang Table 1 将通过末轮仿射/round-boundary shift 得到的版本记为7轮，Hwang
Table 1 进一步列为7/8轮；Hwang Table 7 本身明确标注6/7轮。E25 首先复现项目
标准 `Speck32_64(rounds=6/7)` 的 Table 7，不把边界平移版本混入同一 gate。

## 5. 密码实现 anchor

项目现有实现已核对官方 SPECK32/64 向量：

```text
key        = 0x1918111009080100
plaintext  = 0x6574694C
22-round C = 0xA86842F2
```

E25 测试必须把该向量、key word顺序、`ROR7 / ADD / XOR key / ROL2 / XOR` 轮函数
和 output word packing 固定下来。

## 6. 为什么尚未启动

每个 parity row 都要 XOR 完整的 `2^30 = 1073741824` 个 chosen plaintext 输出。
即使只做本地128把 sampled-key readiness，也需要：

```text
128 x 2^30 = 137438953472 encryptions per round setting
6-round + 7-round = 274877906944 encryptions
```

纯 Python、一次性内存生成或本地 CPU 机械枚举均不可接受。Hwang 还声明 kernel
阶段 `m=10^3`、独立 validation `m'=10^6`；当前项目不会把128-key诊断冒充其
论文规模或全密钥证明。

## 7. 分块实现 readiness

在任何全结构运行前，必须实现：

```text
vectorized uint32 SPECK32/64 encryption backend
assignment index -> active-mask plaintext mapping
chunked XOR reduction without materializing 2^30 rows
per-key parity row durable cache
completed-key bitmap / metadata / progress.jsonl
parameter-matched resume for rounds, key, fixed context and chunk size
scalar-vs-vectorized equality on small random batches
small-cube chunked-vs-exhaustive parity equality
official 22-round vector
6/7-round paper mask evaluation in LSB-first order
```

远程输出必须位于：

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>\
```

不得在生成完整 `2^30` 结构后才写第一个持久产物。

## 8. 分阶段门控

### Phase A：本地实现 smoke

```text
active bits <= 16
2-4 keys
6/7 rounds
NumPy chunked parity == scalar exhaustive parity
all cache/resume/progress tests pass
execution = local CPU
```

### Phase B：远程单 key exact-structure timing gate

```text
active bits = exact 30
fixed bits 5,6 = 00
keys = 1 fixed readiness key
rounds = 6 and 7
backend = remote GPU vectorized chunks
required = exact completed parity rows, peak memory, throughput, ETA, resume proof
```

只有 Phase B 在合理时间内完成且7轮论文 mask 对该 key 为0，才设计 sampled-key
matrix。单 key 通过不是 kernel 复现。

### Phase C：sampled-key exact-kernel readiness

Phase B 已证明单个 exact row 约需7秒，因此冻结最低可判定规模：

```text
seed = 0
key generation seed = 25031
discovery keys = 32
validation keys = 32 fresh
anchor fixed bits = {5,6}, fixed context = 00
anchor rounds = 6,7
same-budget position control fixed bits = {0,1}, fixed context = 00
control rounds = 7
backend = torch_int32 / CUDA
chunk size = 2^24
```

密钥由冻结 seed 确定性生成，前32把只用于 discovery，后32把只用于 validation，
两组不得重叠。控制只把两个相邻固定 bit 从论文位置 `{5,6}` 移到同一个16-bit word
内的 `{0,1}`；其余30 bit全活动，使用完全相同的64把密钥、完整 `2^30` 枚举、
chunk size、GPU后端和 GF(2) 评价。不得在看到结果后更换控制位置。

Phase C advance 必须同时满足：

```text
6轮 joint rank-nullity 为 23/9，joint kernel精确等于Hwang九维span
6轮 Hwang 9个预注册方向分别在 discovery 与 validation 全部成立
7轮 joint rank-nullity 为 31/1，joint kernel精确等于mask 0x02050204
7轮 Hwang预注册方向分别在 discovery 与 validation 成立
控制结构的 joint kernel 不包含 0x02050204
所有缓存完成，第二次执行生成0行，所有 GF(2) basis 回代验证通过
```

discovery 和 validation 各自的经验 rank/nullity、经验 discovery basis 在 validation
的存活率仍完整记录，但不要求32行的7轮半矩阵单独达到31秩。原因是即使真实
row-space 维数为31，随机32行恰好张成完整31维空间的概率也不够高；把半矩阵偶然
欠秩设为失败门会制造假阴性。论文复现的硬门是64把 joint 的精确空间，以及预注册
Hwang 方向在两个不相交密钥半集分别成立。

若 anchor 通过但控制也包含论文7轮 mask，裁决为 `hold`：结构位置特异性不足，
禁止把该 mask 直接做神经标签。若 anchor 未恢复论文 rank/nullity/span，也裁决为
`hold` 并停止机械增加密钥；先审计轮边界、key ownership 和论文协议。Phase C 通过
后也不直接训练，下一审判是四种固定 context `{00,01,10,11}` 的 kernel 多样性与
structure-mask shortcut audit。

## 9. Phase A 后的历史裁决

```text
status = protocol_resolved / phase_a_pass
training = no
remote_scale = no
next = implement CUDA chunk backend and run Phase B remote single-key timing gate
forbidden = guess active mask, local 2^30 CPU run, neural training, remote launch before cache gate
```

OpenAlex 查询在2026-07-17因日预算耗尽返回结构化 rate-limit 错误；Crossref 和
TOSC canonical landing/PDF 已足以完成标题、作者、DOI、年份、主题和协议核验。原始
查询响应保留在 `sources/`，没有把失败的 OpenAlex 响应当作论文证据。

## 10. 2026-07-17 Phase A 实施结果

已新增：

```text
src/blockcipher_nd/tasks/innovation2/speck_hwang_parity.py
tests/test_innovation2_speck_hwang_parity.py
```

实现包括：

```text
NumPy uint32 batch SPECK32/64 encryption
generic assignment-to-active-bit plaintext mapping
exact bits5/6-fixed fast mapping for the 30-active-bit structure
chunked ciphertext XOR reduction
6/7-round Hwang LSB-first basis masks
per-round/per-key parity_rows.npy
completed.npy durable bitmap
strict metadata-matched resume
row/chunk progress callback
```

聚焦验证共22项通过：官方22轮向量、257个随机 plaintext 在1/6/7/22轮的
NumPy/标量逐项相等、4种 chunk size 的小 cube parity 与 scalar exhaustive 完全
一致、缓存第二次运行不重算、参数变化 fail-closed。

真实 Phase A smoke：

```text
active bits = 16
assignments per key = 2^16
keys = 4
rounds = 6,7
parity rows = 8
chunk size = 4096
first rows_generated = 8
resume rows_generated = 0
completed bitmap = all true
elapsed = 0.99 seconds including process startup and resume check
```

按 assignment 数量做最粗线性外推，`2^30 / 2^16 = 16384`，本地 CPU 每个 exact
row 可能达到数十分钟；该估计只用于否决本地机械运行，不能替代 Phase B 的远程
GPU实测。下一步新增 CUDA `uint32/int64-safe` chunk backend，并继续要求与 NumPy
在随机 batch、小 cube parity、缓存 resume 上逐项相等。通过后从已推送 commit 在
`G:\lxy` 启动一把固定 key 的6/7轮 exact-structure timing gate；在完成标记、
throughput、显存、parity row 和恢复证据齐全前，不设计多 key matrix。

## 11. Phase B 远程执行与结果

CUDA 后端、单 key CLI、结果 gate、Windows launcher/run script 和本地 monitor 已
实现。冻结运行：

```text
run_id = i2_speck32_hwang_phase_b_singlekey_gpu0_20260717
physical GPU = 0
key = 0x1918111009080100
rounds = 6,7
active bits = all except integer bits 5,6
fixed bits 5,6 = 00
assignments per row = 2^30
chunk size = 2^24 = 16777216
backend = torch_int32
device = cuda
remote root = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>
```

Phase B pass 必须同时满足：

```text
official SPECK32/64 vector passes
CUDA visible and exact 30-bit structure metadata passes
6/7-round rows both complete and resume generates zero rows
all Hwang Table 7 masks evaluate balanced for the fixed key
timing and peak-memory evidence present for both rows
max elapsed <= 1800 seconds per row
max allocated GPU memory <= 40 GiB
verified result archive and SHA256 manifest retrieved
```

单 key 的 parity matrix nullity 固定至少31，因此 pass 只允许进入“预注册最低
32+32 fresh-key matrix”，不允许声称9维/1维 kernel 已复现。若 timing 或显存超门，
直接枚举路线 hold，转 Midori 或代数/bit-sliced cube-sum backend。

Phase B 已从提交 `a1cb0388c1ae5e629b66c3dd19140d04d9709789` 在远程
RTX A6000 完成，并通过 verified result branch 与 SHA256 manifest 回收：

```text
run_id = i2_speck32_hwang_phase_b_singlekey_gpu0_20260717
assignments per row = 2^30
6轮 parity = 0x078C078C
6轮 elapsed = 7.067094900005031 seconds
6轮 Hwang 9 masks = all balanced
7轮 parity = 0x984D867D
7轮 elapsed = 6.942849299986847 seconds
7轮 mask 0x02050204 = balanced
peak allocated GPU memory = 939524096 bytes (896 MiB)
resume rows generated = 0
archive verification = pass
decision = innovation2_speck_hwang_phase_b_single_key_timing_ready
```

权威本地证据：

```text
outputs/remote_results/i2_speck32_hwang_phase_b_singlekey_gpu0_20260717/
```

该结果只证明 exact `2^30` 路线的单 key 协议、论文 mask 和计算可行性，不是多 key
kernel 复现、论文规模、神经训练或新密码分析性质。

## 12. Phase C 冻结执行

```text
run_id = i2_speck32_hwang_phase_c_32plus32_gpu0_20260717
physical GPU = 0
seed = 0
key generation seed = 25031
discovery keys = 32
validation keys = 32 fresh
anchor = fixed bits {5,6}=00, rounds {6,7}
control = fixed bits {0,1}=00, rounds {7}
assignments per key/round/role = 2^30
chunk size = 2^24
backend = torch_int32
device = cuda
expected exact rows = 64 * 2 + 64 = 192
estimated anchor time = about 15 minutes from Phase B timing
estimated control time = about 7-8 minutes
execution = remote GPU from an exact pushed commit and run-owned clean clone
retrieval = local tmux watcher plus verified result archive
postprocess = local-only tmux validation / plot / index watcher
local plot = scripts/plot-innovation2-speck-hwang-phase-c -> curves.svg
local validation = scripts/validate-innovation2-speck-hwang-phase-c
training = no
```

本阶段唯一改变的实验变量是固定 bit 位置；不修改密码实现、轮边界、密钥组、枚举
规模、输出 bit order 或 kernel 评价。Phase C 结果生成后必须刷新最近结果索引；如
生成图表，必须经过 `visual-qa-redraw` 的像素级检查后才能完成结果处理。

## 13. Phase C 完成结果与裁决

Phase C 从精确提交 `700ac88a4c250fb43ff076ce043c79a575faf95d` 在远程
RTX A6000 完成。verified archive、SHA256 manifest、source commit、缓存元数据、
64把密钥 split、192条 row timing 和本地 GF(2) 重算全部通过：

```text
validation.local.status = pass
validation.local.errors = []
manifest_verified = true
source_commit_matches = true
remote_gate_matches_recomputation = true
resume rows generated = 0
timing rows = 192/192
```

精确 kernel 结果：

```text
anchor，固定 bit {5,6}=00，6轮：
  discovery rank/nullity = 23/9
  validation rank/nullity = 23/9
  joint rank/nullity = 23/9
  joint basis = Hwang Table 7 九维 span（精确相等）
  discovery 9个方向在 validation 全部存活

anchor，固定 bit {5,6}=00，7轮：
  discovery rank/nullity = 30/2
  validation rank/nullity = 31/1
  joint rank/nullity = 31/1
  joint basis = {0x02050204}（精确等于 Hwang Table 7）
  discovery 的2个经验方向中仅论文方向在 validation 存活

position control，固定 bit {0,1}=00，7轮：
  discovery rank/nullity = 31/1
  validation rank/nullity = 31/1
  joint rank/nullity = 32/0
  joint kernel = empty
  Hwang mask 0x02050204 不成立
```

7轮 discovery 的额外一维是32行有限样本的伪方向；它在不相交 validation 中消失，
joint 正好收敛到论文一维。这验证了 Phase C 预注册时“不要求每个32-key半矩阵单独
满31秩、以64-key joint精确空间为硬门”的统计处理，而不是事后放宽门槛。

运行证据：

```text
anchor r6 mean/max row seconds = 6.9587 / 7.0377
anchor r7 mean/max row seconds = 7.1258 / 7.2069
control r7 mean/max row seconds = 7.1077 / 7.1938
summed row time = 1356.30 seconds（约22分36秒）
max allocated GPU memory = 1073741824 bytes（1 GiB）
```

最终裁决：

```text
status = pass
decision = innovation2_speck_hwang_phase_c_kernel_reproduced
training = no
remote mechanical scale = no
next = E26 SPECK fixed-context kernel invariance/diversity audit
```

这证明 Hwang/Wang 的 SPECK32/64 6轮九维、7轮一维输出平衡 kernel 在32把发现密钥
与32把全新验证密钥的 exact `2^30` 结构上复现，而且移动固定位置的同预算控制不含
该 kernel。它不是论文 `10^3/10^6` 密钥规模、全密钥证明、神经结果或新积分性质。

权威证据：

```text
outputs/remote_results/i2_speck32_hwang_phase_c_32plus32_gpu0_20260717/
outputs/00_RECENT_RESULTS.md entry 001
```

真实 `curves.svg` 已渲染为像素检查，中文字体、标题、图例、柱状分离、七项 gate、
裁决文字和导出边界均无重叠、裁切或歧义，`visual_qa_passed.marker` 已记录。

## 14. 下一审判 E26：四种固定 context kernel 不变性/多样性

### 14.1 研究问题与同预算 anchor

Phase C 只有一个正结构 `{5,6}=00` 和一个负位置控制 `{0,1}=00`，尚不足以构造
结构条件输出标签：模型可能只记住固定位置。E26 只改变 `{5,6}` 两个固定 bit 的取值，
判断 `00/01/10/11` 是产生不同稳定 kernel 的有效结构变量，还是应视为不影响标签的
nuisance context。

Wang 2020 Section 4.1 只写明 `x^0_5`、`x^0_6` 为 constant，并在 Table 2 用
`c` 表示 constant bit；论文没有报告四种具体固定值的逐项实验，也没有直接声明该
kernel 对 `00/01/10/11` 不变。Hwang 2026 的 affine chosen-set 定义要求先把固定
public constants 代入 derived function；其输入侧扩展定理还特别说明，“区分器对任意
固定常量成立”是必要假设，不能由 active/fixed pattern 自动推出。因此 E26 若得到
context-invariant，是本项目对论文结构语义的实验闭合证据，不应写成原论文已逐项给出；
若得到 context-dependent，也不与 Wang 的 `c=constant` 记号矛盾。

同预算 anchor 是 Phase C 的 context `00`，必须逐字节复用 verified cache：

```text
anchor parity_rows SHA256 = 3a6df2692fd428938cf8d30e16521947efd1b3242dfc62f288094d7f5187637f
anchor metadata SHA256 = 67138d81e04240b99f42046d2dd6e64a44a8b1586562947176360339a33afe00
keys = Phase C 相同64把，保持 paired comparison
rounds = 6,7
fixed positions = {5,6}
```

### 14.2 唯一变量、规模与计算优化

```text
唯一变量 = fixed context value
contexts = 00（verified baseline）, 01, 10, 11
discovery/validation = Phase C相同32+32把密钥
exact assignments per row = 2^30
new exact enumeration = contexts 01和10 x rounds 6和7 x 64 keys = 256 rows
context11 = parity00 xor parity01 xor parity10
direct context11 crosscheck = 第一把key x rounds 6和7 = 2 rows
backend/device/chunk = torch_int32 / CUDA / 2^24
execution = remote GPU
estimated new row time = about 30-31 minutes
postprocess = local-only validation / plot / index watcher
local validation = scripts/validate-innovation2-speck-hwang-contexts
local plot = scripts/plot-innovation2-speck-hwang-contexts -> curves.svg
training/epochs = none
```

推导 `parity00 xor parity01 xor parity10 xor parity11 = 0` 来自四个 context 对完整
`2^32` 明文空间的分割，以及任意轮数 SPECK 仍为置换、全输出空间 XOR 为0。该推导
必须先在小结构 exhaustive fixture 通过，再以 context11 的一把密钥6/7轮直接完整
枚举交叉验证；不能只因公式看似正确就省略实现校准。

### 14.3 裁决门与后续分支

共同 readiness gate：baseline SHA/metadata/keys 精确匹配，01/10缓存完整且 resume
生成0行，context11 两个 direct crosscheck 通过，所有 split basis 回代有效，远程归档
与本地重算一致。

结果分支预注册：

```text
A. context_invariant:
   四个 context 的6轮 joint kernel均精确等于论文九维；
   四个 context 的7轮 joint kernel均精确等于0x02050204；
   论文方向在每个 context 的 discovery/validation 分别成立。
   -> 固定值不是标签变量；下一步扩展 fixed-position family，保留 context-matched controls。

B. context_dependent_stable:
   至少两个 context 形成不同、跨两半稳定的非平凡 joint kernel。
   -> context可能提供标签多样性；下一步先做 mask/context边际与组外捷径审计，仍不训练。

C. unstable_or_invalid:
   joint kernel不能跨两半稳定、baseline不匹配或推导交叉验证失败。
   -> hold，审计协议；不增加密钥、不训练。
```

禁止路线：把四个 context 当四个训练样本直接训练；把相同 kernel 说成标签多样性；
跳过位置族与 mask-matched shortcut baseline；机械扩大到更多密钥而不先解释 E26 分支。

## 15. E26 完成结果与裁决

E26 从精确提交 `9e8f3ea35d2a0b691f702791064e7867247270a2` 在远程
RTX A6000 完成。verified archive、SHA256 manifest、Phase C baseline 哈希、258条
row timing、01/10磁盘缓存、context11推导与两条直接 exact crosscheck、本地八个
GF(2) kernel 重算全部通过：

```text
validation.local.status = pass
validation.local.errors = []
manifest_verified = true
phase_c_baseline_verified = true
source_commit_matches = true
remote_gate_matches_recomputation = true
timing rows = 258/258
resume rows generated = 0 for 01 / 10 / 11_direct
small partition fixture = pass
context11 direct r6/r7 crosscheck = pass / pass
```

四种固定值的精确结果：

```text
6轮：
  context 00 discovery/validation/joint nullity = 9/9/9
  context 01 discovery/validation/joint nullity = 9/9/9
  context 10 discovery/validation/joint nullity = 9/9/9
  context 11 discovery/validation/joint nullity = 9/9/9
  四个 joint kernel 均精确等于 Hwang Table 7 九维 span

7轮：
  context 00 discovery/validation/joint nullity = 2/1/1
  context 01 discovery/validation/joint nullity = 1/1/1
  context 10 discovery/validation/joint nullity = 2/1/1
  context 11 discovery/validation/joint nullity = 1/2/1
  四个 joint kernel 均精确等于 {0x02050204}
```

7轮不同半集出现的第二个经验方向位置不同，并且都没有在联合64把密钥中存活；这与
Phase C 的有限样本诊断一致。硬证据是四个 context 的 joint rank/nullity 均为
`31/1`，且论文方向分别在每个 context 的 discovery 与 validation 成立。

汇总：

```text
exact paper-span context/round cells = 8/8
paper directions valid in both halves = 8/8
distinct joint signatures at r6 = 1
distinct joint signatures at r7 = 1
nontrivial contexts at r6/r7 = 4/4 and 4/4
summed new exact-row time = 1817.35 seconds（约30分17秒）
max allocated GPU memory = 1073741824 bytes（1 GiB）
```

最终裁决：

```text
status = pass
decision = innovation2_speck_hwang_context_invariant
fixed context value = nuisance, not a label variable
training = no
next = E27 SPECK fixed-position kernel family and shortcut readiness
```

这补齐了 Wang 2020 没有逐项报告的固定值语义：在本项目 exact `2^30`、32+32把
paired sampled keys 下，`{5,6}` 的四种固定值共享相同6/7轮 kernel。它不是论文
`10^3/10^6` 密钥规模、全密钥证明、神经结果或新积分性质。

权威证据：

```text
outputs/remote_results/i2_speck32_hwang_contexts_32plus32_gpu0_20260717/
outputs/00_RECENT_RESULTS.md entry 001
```

真实 `curves.svg` 已通过 `visual-qa-redraw` 像素检查：6/7轮双面板、四个 context、
三个 key split、论文期望线、有限样本第二维和底部裁决均无重叠、裁切或歧义；
`visual_qa_passed.marker` 已记录。

## 16. 下一审判 E27：相邻固定位置 family 筛选

### 16.1 研究问题与同预算结构族

E26 证明 fixed value 不提供标签多样性。E27 固定 `context=00`，只移动两个相邻固定
bit 的位置，测试7轮论文 mask `0x02050204` 是否只属于 `{5,6}`，还是在一组位置上
形成可用于结构条件输出预测的正负 family。

结构族冻结为两个16-bit word 内的全部相邻 pair：

```text
low word  = {(0,1), (1,2), ..., (14,15)}
high word = {(16,17), (17,18), ..., (30,31)}
total positions = 30
active bits = 其余30 bit
fixed context = 00
rounds = 7
target mask = 0x02050204
```

所有位置具有完全相同的活动宽度、明文数、轮数、密钥顺序、后端和评价；唯一变量是
fixed pair start。Phase C `{5,6}` 是 verified positive anchor，`{0,1}` 是 verified
negative control，必须复用其64-key cache与哈希，不重新挑选 anchor/control。

### 16.2 分阶段规模与动态预算门

```text
Phase S screen keys = Phase C key indices 0..7
screen positions = 除anchor/control外其余28个相邻pair
screen exact rows = 28 * 8 = 224
screen pass = paper mask在8把key上全部平衡
false-positive expectation under random parity = 28 / 2^8 = 0.109375

Phase V candidate keys = indices 8..63（24 discovery remainder + 32 validation）
per candidate additional exact rows = 56
full candidate evidence = 64 paired keys
```

候选数若 `<=8`，验证全部候选。若 screen 候选 `>8`，不得事后挑最漂亮的位置；使用
冻结的确定性选择：分别按 pair start 升序取 low-word 前4个和 high-word 前4个，缺额
再按全局升序补齐，并把“候选过宽”作为单独指标。预计基础 screen 约26分钟，每个
进入 Phase V 的新候选约增加6.5分钟。执行路径为远程 GPU、逐位置磁盘缓存、可恢复
进度和 exact pushed commit；本地只做小结构映射与 gate readiness。

### 16.3 裁决与停止门

共同 readiness：Phase C anchor/control baseline SHA与key顺序匹配；30个位置唯一且覆盖
两字；screen/validation缓存完整并可恢复；所有 paper-mask checks 与候选 joint kernel
本地重算一致。

```text
advance_position_family:
  64-key稳定正位置 >= 4（含anchor）
  sampled negative位置 >= 8（含control）
  正位置覆盖low/high两个word
  每个稳定正位置的joint kernel包含0x02050204
  -> E28 mask-matched position-group holdout与捷径审计，仍不训练。

narrow_position_family:
  稳定正位置为2或3，或只在一个word出现
  -> hold；评估非相邻、旋转等价位置族，不机械增加密钥。

anchor_only_or_invalid:
  只有{5,6}正，anchor/control不复现，或缓存/推导无效
  -> 停止当前SPECK位置标签路线；不训练。
```

禁止：用8-key screen 命中直接当最终正标签；把 fixed value 四种取值重复计作样本；
对 screen 失败位置继续机械补64把密钥；在 position/mask group-disjoint shortcut gate
之前训练任何神经网络。

### 16.4 E27 实现与本地 readiness

E27 实现冻结为：

```text
run_id = i2_speck32_hwang_positions_gpu0_20260717
remote device = RTX A6000 GPU0
screen cache = cache/positionXX/screen/{metadata.json,parity_rows.npy,completed.npy}
validation cache = cache/positionXX/validation/{metadata.json,parity_rows.npy,completed.npy}
verified result branch = results/i2_speck32_hwang_positions_gpu0_20260717
```

Phase C control 哈希已在实现前冻结：

```text
control parity SHA256 = 34486c570a630544ce3ca9fccf1297629bc7924fb6ec19adcf939a4b97b485ca
control metadata SHA256 = 79d31905668c8121ca8ee2f30fc2fd6bd87d981bdafadf544143e182b5d2b1d3
```

本地 readiness 只验证映射、冻结 baseline、密钥顺序和动态候选规则，不运行 `2^30`
枚举。2026-07-17 就绪结果：

```text
status = pass
mapping_fixture_valid = true
phase_c_anchor_and_control_verified = true
phase_c_keys = 64
position_pairs = 30
screen_keys = 8
max_validation_candidates = 8
training_performed = false
```

聚焦测试覆盖：30个位置边界、相邻 bit 映射、候选超过8个时的 low/high word 冻结
选择、advance/narrow/anchor-only 三分支、只对screen候选生成56-key缓存、E25/E26
回归、结果索引，以及远程 `G:\lxy`、`cmd.exe /c`、无 delayed expansion、verified
archive、独立验证、绘图和 `visual_qa_pending` handoff。远程结果必须经本地 manifest、
baseline SHA、cache-to-aggregate parity、密钥顺序、GF(2) gate 独立重算后才能裁决。

### 16.5 条件下一门 E28：位置 × mask 标签宽度与组外捷径

E28 只在 E27 得到 `innovation2_speck_hwang_position_family_advance` 且本地独立
验证通过后执行。它是本地纯后处理，不新增明文枚举、不增加密钥、不训练网络。
输入冻结为 E27 中拥有完整64-key parity 的位置：Phase C anchor、control，以及
进入 Phase V 的候选。8-key-only screen 失败位置不得伪装成64-key确定标签。

每个完整位置先重算 joint kernel。候选 output mask 使用有界、确定性规则：

```text
1. 收集每个joint kernel的全部非零basis mask；
2. 收集同一kernel内任意两个不同basis mask的pairwise XOR；
3. 去重并按整数升序排序；
4. 只保留在至少1个、但非全部完整位置kernel中成立的flipping mask；
5. 不加入“在所有位置恒负”的人工mask，因为mask identity会直接泄漏标签。
```

这不是完整枚举高维 kernel span；它是固定的 basis+pairwise 标签宽度诊断。若某个
kernel 只有更高阶组合才产生额外方向，E28 不据此声称不存在，只报告当前有界候选
不足。标签网格定义为：

```text
label(position, mask) = 1 iff mask belongs to that position's 64-key joint kernel
```

readiness 下限：

```text
full-evidence positions >= 8
flipping masks >= 8
distinct position label signatures >= 4
positive rate in [0.10, 0.90]
target mask 0x02050204 included and flips across positions
complete position x mask grid
```

同一标签网格必须报告 global、position identity、mask identity、position/mask weight、
identity-additive 和 32+32 bitwise linear 基线的 accuracy、Brier、AUC。随后冻结三类
group-disjoint 评价：

```text
position-disjoint：整组留出位置；
mask-disjoint：整组留出mask；
dual-disjoint：测试cell的位置和mask均未在训练cell出现；
control：相同dual folds上的标签打乱。
```

fold 只能用固定 seed0、按整数排序后的确定性分组；每个 train/test fold 必须两类齐全，
否则 readiness 失败，不得事后换 seed 或合并最难 fold。方向无关捷径指标为
`max(AUC, 1-AUC)`，避免反向预测被错误当成无信号。

裁决预注册：

```text
advance_label_grid:
  readiness全部通过；
  position identity、mask identity、additive、bitwise线性AUC均<0.75；
  position/mask/dual-disjoint directional AUC均<0.75；
  shuffle dual directional AUC<=0.65。
  -> E29 fresh-key稳定性与结构族扩展；仍不直接训练。

label_grid_shortcut_dominated:
  标签宽度足够，但任一真实组外捷径AUC>=0.75。
  -> 停止当前SPECK相邻位置×mask表，不挑选通过的单一split。

label_grid_too_narrow:
  完整位置<8、flipping mask<8、签名<4或类别/fold不完整。
  -> 不训练；若E27是narrow分支则优先审判ROR7-to-addition真实对齐与offset-1错位控制，
     若E27虽过数量门但mask宽度不足，同样转拓扑耦合结构族而不是机械加密钥。
```

即使 E28 通过，E27 最多10个完整位置仍不足以直接支撑神经模型的结构组外训练。
E29 必须先用新密钥确认标签，并把独立结构数量扩展到可形成稳定 train/validation/test
组的规模；禁止把同一位置的不同密钥复制成多个“独立结构样本”。

### 16.6 条件窄分支 E27-N：ROR7-to-addition 拓扑对齐对照

E27-N 只在以下任一条件成立时执行：E27 裁决为 `narrow_position_family`；或 E27
过数量门但 E28 裁决为 `label_grid_too_narrow`。若 E27 为 `anchor_only`，当前
SPECK位置路线直接停止，不得用 E27-N 绕过停止门。若 E28 已通过标签宽度与捷径门，
优先做 E29 fresh-key，不运行 E27-N。

SPECK32/64 项目整数表示中，低16 bit为 `y`，高16 bit为 `x`。一轮首先计算
`ROR7(x) + y`，所以进入同一模加 bit lane 的真实输入关系为：

```text
true topology pair(i) = { y_i, x_(i+7 mod 16) }
project bits          = { i, 16 + ((i+7) mod 16) }
i                     = 0..15
```

同预算错位控制只把高字索引偏移减1：

```text
offset-minus-one control(i) = { y_i, x_(i+6 mod 16) }
project bits                = { i, 16 + ((i+6) mod 16) }
i                           = 0..15
```

两组各16个 pair，组内和组间均按 `(family, i)` 唯一标识；所有 pair 固定两个 bit
为 `00`，其余30 bit活动。保持 E27 的密码、7轮、目标 mask `0x02050204`、Phase C
64把密钥、CUDA backend、chunk、输出 bit order 和 exact `2^30` 明文不变。唯一
研究变量是跨 word pair 是否符合真实 `ROR7 -> addition` lane 对齐。

分阶段预算：

```text
Phase S:
  true 16 + control 16
  Phase C key indices 0..7
  exact rows = 32 * 8 = 256
  screen pass = target mask在8把key全部平衡
  random-parity expected hits = 32 / 2^8 = 0.125

Phase V:
  true候选按i升序最多取4个
  control候选按i升序最多取4个
  若某组不足4个，不从另一组补齐名额
  selected candidate增加indices 8..63共56个exact row
  max additional rows = 8 * 56 = 448
```

不得按两组合并排序后只挑真实拓扑候选；每组独立最多4个能保持同预算归因。screen
失败 pair 不补密钥。预计 Phase S 约30分钟，Phase V 每个候选约6.5分钟，上限总时长
约82分钟；远程GPU、逐 pair 磁盘缓存、resume、verified branch、本地独立重算和视觉
QA规则与 E27 相同。

裁决：

```text
topology_aligned_family:
  64-key稳定true位置 >= 2；
  64-key稳定control位置 = 0；
  true screen hits - control screen hits >= 2；
  每个稳定true位置的joint kernel包含目标mask。
  -> 构造拓扑pair × output-mask标签宽度门；仍不训练。

topology_not_specific:
  任一control候选在64-key稳定，或true/control screen命中差<2。
  -> hold/stop；真实对齐没有超过错位控制，不继续换offset挑结果。

topology_no_signal:
  true family没有64-key稳定位置。
  -> 停止当前SPECK固定pair路线，不增加密钥、不测试其余14个offset。

protocol_invalid:
  key/cache/backend/mapping/timing/GF(2)任一门失败。
  -> 只修协议，不解释信号。
```

禁止把 `same lane`、`ROL2/XOR` 对齐或其他 offset 同时加入该矩阵。它们是不同假设，
只有当前真实 ROR7-to-addition family 明确失败且论文路线仍有独立理由时才能另行
预注册；不得事后扫描16个offset并报告最好一个。

## 17. E27 完成结果与裁决

E27 从精确 pushed commit `41d60a1b73c2018a09b2cfae7a9ccc44ca256d9f` 在远程
RTX A6000 GPU0 完成，并从 verified result branch 回收。本地独立验证结果：

```text
status = pass
errors = []
manifest_verified = true
phase_c_baselines_verified = true
source_commit_matches = true
remote_gate_matches_recomputation = true
timing rows = 280/280
all cache roles complete = true
resume rows generated = 0
```

动态预算实际执行为：

```text
28个新相邻位置 x 8 screen keys = 224 exact rows
screen新候选 = position start 6，即fixed bits {6,7}
Phase V = {6,7} x 56 remaining keys = 56 exact rows
total new exact rows = 280
assignments per row = 2^30
summed row time = 2001.108052 seconds（约33分21秒）
max allocated GPU memory = 1073741824 bytes（1 GiB）
```

8-key screen 中，除已验证 anchor `{5,6}` 外，只有相邻 pair `{6,7}` 达到 `8/8`
平衡。它补到完整64把密钥后仍稳定：

```text
fixed {5,6}:
  discovery/validation/joint nullity = 2/1/1
  joint rank = 31
  joint kernel = {0x02050204}

fixed {6,7}:
  discovery/validation/joint nullity = 1/2/1
  joint rank = 31
  joint kernel = {0x02050204}
```

两处半集中的额外经验方向均未进入 joint。其余28个位置被计为 sampled negatives：
`{0,1}` 是 Phase C 完整64-key负控制，其他27个是8-key screen failures，不能夸大为
完整64-key否定。高16-bit word的15个相邻 pair 全部screen失败；稳定正位置只覆盖
低 word。

最终门控：

```text
stable positive positions = [5, 6]
stable positive count = 2 < 4
sampled negative count = 28 >= 8
positive words = [low]
screen candidates excluding anchor/control = [6]
candidate overflow = false
status = hold
decision = innovation2_speck_hwang_position_family_narrow
training = no
```

因此 E27 证明论文7轮 mask 不只属于精确的 `{5,6}`，相邻平移一位的 `{6,7}` 也在
同一64-key协议下成立；但它没有形成覆盖两个word的宽位置族，而且两个正位置的
joint kernel 完全相同。这个证据不足以构造预注册 E28 所需的至少8个完整位置和8个
flipping masks，所以 E28 不执行，不能用空标签网格制造“通过”。下一步按结果揭晓前
冻结的16.6节执行 E27-N：ROR7-to-addition真实跨word对齐 family 与 offset-minus-one
同预算错位控制。

权威证据：

```text
outputs/remote_results/i2_speck32_hwang_positions_gpu0_20260717/
outputs/00_RECENT_RESULTS.md entry 001
```

真实 `curves.svg` 已按自然尺寸 `1461x743` 渲染到像素并通过
`visual-qa-redraw`：标题、中文解释、30个位置标签、双word面板、anchor/control
注释、8-key阈值线、裁决和证据范围无重叠、裁切、缺字或结构歧义；marker 已转换为
`visual_qa_passed.marker`。

## 18. E27-N 实现与远程就绪证据

E27-N 已按16.6节冻结协议实现，远程运行ID为：

```text
i2_speck32_hwang_topology_pairs_gpu0_20260717
```

实现保持单一变量：真实 `{y_i, x_(i+7)}` 对齐与错位
`{y_i, x_(i+6)}` 控制各16个lane；两组独立按lane升序最多选择4个8-key候选，
补齐同一Phase C剩余56把密钥。缓存按 `family/lane/phase` 落盘，并在独立校验中核对
密钥序列、活动位、固定明文、后端、chunk、设备、聚合数组、完成位图、resume行数、
动态计时身份、Phase C冻结SHA256、source commit及远程门控重算。

本地就绪证据：

```text
focused tests = 28 passed
Phase C readiness = pass
Phase C keys = 64
families x lanes = 2 x 16
screen keys = 8
max validation candidates = 4 per family
training performed = false
```

合成裁决图已按自然尺寸 `1444x756` 渲染成像素并执行 `visual-qa-redraw`。标题、说明、
四类点图例、双面板、阈值线、lane标签、指标、裁决和证据范围均无重叠、裁切、缺字或
结构歧义。该图只验证绘图实现，不是实验结果；真实远程结果回收后仍必须重新执行视觉
QA，才能将 `visual_qa_pending.marker` 转为通过标记。

远程执行门仍保持：只能从精确 pushed commit 启动 run-owned clean clone；GPU缓存、
日志、archive均位于 `G:\lxy`；结果分支、SHA256、本地独立验证、结果索引和真实图像
QA全部完成前，不得把 E27-N 报告为完成，也不得启动E28或神经网络训练。

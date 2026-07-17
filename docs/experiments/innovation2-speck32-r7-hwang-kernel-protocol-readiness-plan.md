# 创新2 E25：SPECK32/64 Hwang Table 7 kernel 协议与分块执行就绪计划

日期：2026-07-17
状态：Phase C 通过并验证回收 / SPECK 6、7轮论文 kernel 精确复现 / E26 context 审计待实施

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

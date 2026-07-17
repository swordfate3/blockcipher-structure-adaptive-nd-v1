# 创新2 E20：SKINNY-64/64 7轮 Hwang exact-kernel 协议就绪审计

日期：2026-07-17
状态：已完成 / pass

## 1. 研究问题

PRESENT-80 7轮 inactive-context 的严格 kernel membership 与跨密钥平衡概率
路线已分别在 E18、E19 被 fresh-key 和 interaction-residual 门槛暂缓。E20 不再
扩大 PRESENT 的密钥、context 或 mask，而是切换到 Hwang 2026 正文主案例
SKINNY-64/64，回答一个更前置且可证伪的问题：

> 本项目能否在正确的 keyed SKINNY-64/64、活动 cell 和 MSB-first 输出位序下，
> 精确复现论文7轮 raw-bit parity matrix 的 `rank=46/nullity=18` 及完整18维
> kernel span？

E20 是密码实现与论文协议 readiness，不训练神经网络，也不声称提出 kernel 方法。

## 2. 权威来源与已知锚点

本轮使用三个互补来源：

1. Beierle et al. 2016 SKINNY 原始规范 Appendix B 的 SKINNY-64-64 向量：

```text
plaintext  = 0x06034F957724D19D
key / TK1  = 0xF5269826FC681238
ciphertext = 0xBB39DFB2429B8AC7
rounds     = 32
```

2. Zhang et al. 2026 Section 4.3 / Distinguisher 1：7轮 SKINNY-64/64，活动
第15个 plaintext cell；已知8个两比特线性组合来自 Hwang basis `0..3` 与
`10..13`。
3. Hwang et al. 2026 Section 5.1 / Table 2：raw 64-bit parity feature 使用
MSB-first 位编号；7轮经验矩阵 `(rank, nullity)=(46,18)`，完整 basis 为：

```text
(b4,b52)             (b19,b35,b51)
(b5,b53)             (b24,b56)
(b6,b54)             (b25,b57)
(b7,b55)             (b26,b58)
(b8,b44,b56,b60)     (b27,b59)
(b9,b45,b57,b61)     (b28,b44,b60)
(b16,b32,b48)        (b29,b45,b61)
(b17,b33,b49)        (b30,b46,b62)
(b18,b34,b50)        (b31,b47,b63)
```

论文公开的匿名代码链接在当前环境返回 `not_connected`，因此不把作者代码对拍
列为已经完成的校验。E20 必须依靠原始规范向量、论文正文和本项目独立实现三者
闭环，并把这一来源限制保留在 metadata/claim scope 中。

## 3. 固定协议

```text
cipher = SKINNY-64/64, TK1 only
rounds = 7
state = 4 x 4 row-major nibbles
bit order = global MSB-first, paper b0 maps to integer bit 63
target active cell = 15
control active cell = 0
inactive context = one deterministic random 64-bit base plaintext
plaintexts per key per role = 16
discovery keys = 512
validation keys = 256 fresh and disjoint
total keys = 768 unique 64-bit TK1 values
feature = XOR of raw 64 ciphertext bits over each 16-text multiset
training = none
device = local CPU
```

target 与 control 必须复用完全相同的 base plaintext、密钥顺序、轮数和输出格式；
唯一变量是活动 cell。密钥和 base plaintext 由冻结 seed 确定性生成。

Hwang 报告的 `10^6` data 是论文规模。E20 的 `512+256` 是低成本 sampled-key
readiness，足以检查64列经验矩阵是否达到论文签名，但不是 paper-scale 或全密钥
证明。若 readiness 通过，后续仍需按独立问题决定是否增加论文规模证据。

## 4. 同预算控制

E20 同时要求：

- target cell15 的 discovery、validation、joint 三个 kernel 都与论文18维 span
  完全相等；只检查8个已知 mask 不够。
- active-cell0 控制在同预算下不得复现论文 span；预期 joint matrix full rank、
  nullity为0。
- global LSB-first、cell顺序反转、cell内bit反转三种错误位序不得与 target joint
  kernel span 相等。
- discovery basis 必须逐向量通过 validation，所有 basis 都必须通过各自矩阵的
  `M u = 0` 校验。

子空间比较只使用 GF(2) rank 与 basis containment，不枚举18维 span。

## 5. Readiness 门槛

协议有效条件：

```text
public 32-round SKINNY-64/64 vector passes
512 discovery and 256 validation keys are unique and disjoint
target/control parity-row caches have shape 2 x 768 and uint64 dtype
all discovery/validation/joint bases validate their matrices
all metrics are finite and row ownership is complete
```

推进条件全部满足：

```text
target discovery rank/nullity = 46/18
target validation rank/nullity = 46/18
target joint rank/nullity = 46/18
target discovery/validation/joint spans = Hwang Table 2 span
all 18 discovery basis directions survive validation
control joint rank/nullity = 64/0
control joint span != Hwang Table 2 span
all three wrong bit-order mappings != target joint span
```

裁决：

```text
pass:
  decision = innovation2_skinny_r7_hwang_kernel_reproduced
  next = E21 SKINNY-64/64 8轮 two-active-cell exact-kernel readiness

hold:
  decision = innovation2_skinny_r7_hwang_kernel_not_reproduced
  next = 审计活动cell、轮边界、state/TK1排列和论文数据所有权；不训练

fail:
  decision = innovation2_skinny_r7_hwang_protocol_invalid
  next = 修复密码向量、缓存、GF(2)或拆分协议；不解释研究信号
```

## 6. 产物与执行路径

预期 run id：

```text
i2_skinny64_r7_hwang_kernel_readiness_768keys_seed0_20260717
```

产物：

```text
outputs/local_audits/<run_id>/
  results.jsonl
  kernel_basis.csv
  keys.npy
  parity_rows.npy
  metadata.json
  progress.jsonl
  gate.json
  curves.svg
  visual_qa_passed.marker
```

`curves.svg` 必须展示 target/control 的 discovery、validation、joint rank/nullity、
论文门槛和 span equality，标题使用中文解释。生成后必须执行 `visual-qa-redraw`
像素检查。结果完成后刷新 `outputs/00_RECENT_RESULTS.md/.json`。

E20 只在本地运行。readiness 未通过时禁止启动神经训练、远程 GPU、更多 seed 或
机械扩大密钥数。

## 7. 通过后的研究边界

E20 通过只证明本项目能重现一个已知的7轮 SKINNY 输出 balance space。它不是
创新结果。下一步 E21 必须先复现 Zhang/Hwang 的8轮 two-active-cell 单方向
`b28 xor b44 xor b60` fixture；只有高轮 fixture 也稳定，才允许构造多个
`(integral structure, output mask)` 标签，并重新执行 marginal、group-disjoint、
fresh-key 和有限密钥噪声控制。创新2的目标仍是结构条件化积分输出性质预测，
不是 structured-vs-random 二分类。

## 8. 2026-07-17 E20 结果

本项目独立实现的 TK1 SKINNY-64/64 首先通过原始规范 Appendix B 的32轮公开
测试向量。随后使用冻结的 base plaintext `0x4F24390FFC0B613B` 和768把唯一密钥，
对 target cell15 与同预算 control cell0 分别生成 raw 64-bit parity rows。

最终矩阵结果：

| 角色 | split | keys | rank | nullity | discovery basis 在 validation 存活 |
|---|---|---:|---:|---:|---:|
| target cell15 | discovery | `512` | `46` | `18` | `18/18` |
| target cell15 | validation | `256` | `46` | `18` | 不适用 |
| target cell15 | joint | `768` | `46` | `18` | 不适用 |
| control cell0 | discovery | `512` | `64` | `0` | `0/0` |
| control cell0 | validation | `256` | `64` | `0` | 不适用 |
| control cell0 | joint | `768` | `64` | `0` | 不适用 |

target 的 discovery、validation、joint 三个18维 kernel span 均与 Hwang
Table 2 完全相等，所有18个论文 basis 在三个 split 上都满足 `M u = 0`。cell0
控制不接受论文 span，且为 full rank。三种错误位序控制结果均不相等：

```text
global LSB-first equality       = false
reversed cell-order equality   = false
bit-reversed-in-cell equality  = false
```

所有协议、缓存、basis 验证和信号门槛通过，最终裁决为：

```text
status = pass
decision = innovation2_skinny_r7_hwang_kernel_reproduced
training = no
remote_scale = no
next_adjudication = E21 SKINNY r8 Hwang kernel readiness
```

这说明本项目已经正确对齐 SKINNY-64/64 的轮函数、TK1 排列、活动 cell15、
MSB-first 输出位序和 Hwang raw-bit kernel 协议。cell0 控制排除了“任何活动 cell
都会机械出现18维 kernel”的解释。它仍只是 `512 discovery + 256 validation`
sampled-key readiness，不是论文 `10^6` data 复现、全密钥证明、神经训练结果或新
balance property。

权威产物：

```text
outputs/local_audits/
  i2_skinny64_r7_hwang_kernel_readiness_768keys_seed0_20260717/
```

`curves.svg` 的第一次像素检查发现两组图例压住柱体和数值；渲染器已将图例移到
两个坐标轴上方的独立区域。最终 `1800 x 830` 渲染通过 `visual-qa-redraw`：中文
字体、标题、副标题、split 归属、rank/nullity 门槛、span 比较和裁决说明均无重叠、
裁切、缺字或歧义。

## 9. 推荐下一步：E21 SKINNY-64/64 8轮 two-active-cell exact kernel

E20 是已知7轮 fixture，不能直接支持高轮结构条件预测。E21 只测试论文给出的下
一个高轮锚点：Zhang Distinguisher 2 / Hwang Table 2(b)。协议冻结为：

```text
cipher / rounds = SKINNY-64/64 / 8
target active plaintext cells = 14,15
plaintexts per key = 16^2 = 256
output feature = raw 64-bit parity, MSB-first
paper kernel basis = (b28,b44,b60)
expected rank/nullity = 63/1
discovery / validation keys = 512 / 256, fresh and disjoint from E20
same-budget control = r8 active cells 0,1 with identical keys/context
training = none
execution = local CPU with keys.npy, parity_rows.npy and progress JSONL
```

推进门要求 target 的 discovery、validation、joint 都为 `63/1` 且唯一 span 与
`b28 xor b44 xor b60` 完全相等；control 不得复现该 span，错误位序不得匹配。
通过后进入 E22，构造多个 r8 active-cell geometry 与 output-mask 的标签矩阵并先
做 marginal、group-disjoint、fresh-key 和有限密钥噪声审计。失败则先审计8轮
round boundary、two-cell enumeration 和论文数据所有权，不增加网络、seed 或远程
预算。

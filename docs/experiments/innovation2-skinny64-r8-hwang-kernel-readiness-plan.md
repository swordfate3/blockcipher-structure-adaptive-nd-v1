# 创新2 E21：SKINNY-64/64 8轮 two-active-cell exact-kernel 就绪审计

日期：2026-07-17
状态：已完成 / pass

## 1. 研究问题

E20 已在本项目独立复现 SKINNY-64/64 7轮、活动 cell15 的 Hwang 18维 raw-bit
kernel，并由同预算 cell0 控制和错误位序控制确认协议。E21 继续检查论文给出的
下一个高轮 fixture：

> 当 plaintext cells 14、15 同时遍历全部 `16^2=256` 个取值时，本项目能否在
> 8轮后精确复现唯一 balance direction `b28 xor b44 xor b60`，即经验矩阵
> `rank=63/nullity=1`？

来源为 Zhang et al. 2026 Distinguisher 2 与 Hwang et al. 2026 Table 2(b)。E21
仍是 known-fixture readiness，不训练网络，也不把已知性质写成新创新。

## 2. 固定协议

```text
cipher = SKINNY-64/64, TK1 only
rounds = 8
state = 4 x 4 row-major nibbles
output bit order = global MSB-first
target active cells = 14,15
control active cells = 0,1
inactive context = one deterministic random 64-bit base plaintext
plaintexts per key per role = 256
discovery keys = 512
validation keys = 256 fresh and disjoint
total keys = 768 unique 64-bit TK1 values
keys = fresh and disjoint from E20's 768 keys
feature = XOR of raw 64 ciphertext bits over each 256-text multiset
paper basis = (b28,b44,b60)
training = none
device = local CPU
```

target 与 control 复用完全相同的 base plaintext、密钥顺序、轮数和输出格式；唯一
变量是活动 cell pair。E21 的 `512+256` sampled-key evidence 是 readiness，不是
Hwang 的 `10^6` data、paper-scale 或全密钥证明。

## 3. 同预算控制

- target discovery、validation、joint 三个 kernel 必须都精确等于论文一维 span。
- control `(0,1)` 不要求 full rank，因为论文没有给该结构的预期 kernel；但它不得
  接受 `b28 xor b44 xor b60`，也不得与论文 span 相等。
- global LSB-first、cell顺序反转、cell内bit反转三种错误映射不得匹配 target
  joint kernel。
- discovery 的唯一 basis 必须在 validation 全部存活，所有 basis 必须通过各自
  矩阵的 `M u = 0` 校验。
- SKINNY Appendix B 32轮公开向量必须继续通过；E20/E21 密钥集合必须互斥。

## 4. Readiness 与裁决门槛

协议门：

```text
public SKINNY-64/64 vector passes
E21 768 keys unique; discovery/validation disjoint
E21 keys disjoint from deterministic E20 key set
target/control cache shape = 2 x 768, dtype = uint64
all bases validate and all metrics are finite
paper basis rank = 1
```

推进门：

```text
target discovery rank/nullity = 63/1
target validation rank/nullity = 63/1
target joint rank/nullity = 63/1
target discovery/validation/joint spans = {(b28,b44,b60)}
discovery basis validation survivors = 1/1
control joint does not validate the paper basis
all wrong bit-order spans do not match target joint kernel
```

裁决：

```text
pass:
  decision = innovation2_skinny_r8_hwang_kernel_reproduced
  next = E22 r8 active-cell geometry x output-mask label readiness

hold:
  decision = innovation2_skinny_r8_hwang_kernel_not_reproduced
  next = 审计 two-cell enumeration、round boundary、位序和论文数据所有权

fail:
  decision = innovation2_skinny_r8_hwang_protocol_invalid
  next = 修复密码向量、密钥所有权、缓存或 GF(2) 实现
```

所有分支均保持 `training=no`、`remote_scale=no`。E21 不通过时禁止增加网络、seed
或远程预算掩盖协议问题。

## 5. 产物

run id：

```text
i2_skinny64_r8_hwang_kernel_readiness_768keys_seed0_20260717
```

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

图表必须用中文说明8轮、two-active-cell、target/control、三个 key split、`63/1`
门槛和 span equality。生成后执行 `visual-qa-redraw` 像素检查，完成结果后刷新
`outputs/00_RECENT_RESULTS.md/.json`。

## 6. 通过后的下一步边界

E21 通过仍只是已知8轮性质复现。E22 才允许围绕8轮构造多个 active-cell geometry
与 output mask 的结构条件标签；在任何神经训练前，必须先证明标签不被 structure
或 mask 边际、位模式、group split、有限密钥噪声解释，并在 fresh keys 上稳定。
最终目标仍是预测给定结构与输出 mask 的积分输出性质，不是 structured-vs-random
二分类。

## 7. 2026-07-17 E21 结果

E21 使用与 E20 完全不相交的768把新密钥和固定 base plaintext
`0xDEB4B7AD29B18D62`，分别计算 target cells14+15 与同预算 control cells0+1 的
8轮 raw-bit parity matrix。每个结构、每把密钥完整遍历256个明文。

结果：

| 角色 | split | keys | rank | nullity | discovery basis 在 validation 存活 |
|---|---|---:|---:|---:|---:|
| target cells14+15 | discovery | `512` | `63` | `1` | `1/1` |
| target cells14+15 | validation | `256` | `63` | `1` | 不适用 |
| target cells14+15 | joint | `768` | `63` | `1` | 不适用 |
| control cells0+1 | discovery | `512` | `64` | `0` | `0/0` |
| control cells0+1 | validation | `256` | `64` | `0` | 不适用 |
| control cells0+1 | joint | `768` | `64` | `0` | 不适用 |

target 三个 split 的唯一 basis 完全一致：

```text
mask = 0x0000000800080008
paper bits = b28 xor b44 xor b60
```

它与 Hwang Table 2(b) 的一维 span 在 discovery、validation、joint 三组均完全
相等。control 不接受该 mask，三种错误位序映射也均不匹配；Appendix B 向量、
E20/E21 密钥互斥、缓存形状、basis 校验和所有协议门全部通过。

最终裁决：

```text
status = pass
decision = innovation2_skinny_r8_hwang_kernel_reproduced
training = no
remote_scale = no
next_adjudication = E22 SKINNY r8 structure-mask readiness
```

E21 把本项目的已知积分输出性质锚点从7轮推进到8轮，并确认不是 cells0+1 的通用
two-active-cell 现象。它仍只是 `512 discovery + 256 validation` sampled-key
readiness，不是论文 `10^6` data、全密钥证明、神经模型结果或新 balance property。

权威产物：

```text
outputs/local_audits/
  i2_skinny64_r8_hwang_kernel_readiness_768keys_seed0_20260717/
```

第一次图像像素检查发现 nullity 面板继承 E20 的 `0..22` 纵轴，使关键的 `1 vs 0`
被压在基线附近。renderer 已改为按论文目标自适应 nullity 范围，E21 使用
`0..2.5`。最终 `1800 x 830` 渲染通过 `visual-qa-redraw`：标题、图例、数值、
门槛、split、span 和裁决说明无重叠、裁切、缺字或误导轴范围。

## 8. 推荐下一步：E22 8轮 active-cell geometry kernel 多样性审计

E20/E21 证明两个已知 fixture 可复现，但一个7轮18维标签和一个8轮1维标签仍不足
以训练“结构 + output mask -> balance property”模型。E22 的单一研究问题是：

> 固定 SKINNY-64/64 8轮与 two-active-cell 宽度，只移动活动 cell pair，是否能在
> fresh keys 上得到多个稳定、不同的输出 kernel，从而形成非退化标签族？

建议冻结：

```text
structures = 16个循环相邻 cell pairs:
             (0,1),(1,2),...,(14,15),(15,0)
anchor = (14,15) must reproduce {(b28,b44,b60)}
same-budget control = (0,1)
keys = 128 fresh = 64 discovery + 64 validation, disjoint from E20/E21
plaintexts per structure per key = 256
feature = raw 64-bit parity, MSB-first
training = none
execution = local CPU, disk-backed parity_rows.npy + keys.npy + progress JSONL
```

推进门应同时要求 anchor exact、所有 basis 在各自 half 上有效、至少4个 pair 具有
非零 joint kernel、至少4种不同 joint-kernel 签名，并报告 discovery basis 的
validation 存活率。通过后才从这些 kernel 构造结构-mask 标签并做边际/位模式/
group-disjoint/fresh-key 捷径审计；若只有论文 anchor 或少量对产生非平凡 kernel，
则停止机械移动 pair，不训练网络，也不扩远程规模。

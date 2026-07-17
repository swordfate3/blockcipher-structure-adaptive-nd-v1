# 创新2 E24：SKINNY-64/64 7轮单活动 cell kernel 几何多样性审计

日期：2026-07-17
状态：已完成 / hold

## 1. 研究问题

E20 已精确复现 Hwang Section 5.1 / Table 2 的 SKINNY-64/64 7轮、活动
`cell15`、18维 raw-bit kernel，并确认同预算 `cell0` 为 full rank。E23 则否定了
8轮底行 complete-pair 闭合假设。E24 停止8轮 pair 路线，回到论文正文信息量最高
的7轮单活动 cell 主案例，只改变活动位置：

> 在固定7轮、单活动 cell 宽度、密钥、inactive context 和 raw parity 格式时，
> 16个活动 cell 是否产生多个稳定且不同的 joint kernel？

如果只有 `cell15` 非平凡，结构字段仍近似一个身份开关，不能构造非退化的
`structure + output mask -> balance property` 预测任务。

## 2. 固定协议

```text
cipher / rounds = SKINNY-64/64 / 7
state layout = 4 x 4 row-major nibbles
structures = single active cell 0..15
paper anchor = cell15 -> Hwang Table 2 exact 18-dimensional span
same-budget control = cell0
output feature = raw 64-bit parity, MSB-first
inactive context = one deterministic random 64-bit base plaintext
plaintexts per structure per key = 16
discovery keys = 64
validation keys = 64 fresh and disjoint
joint keys = 128 unique
key generation seed = 12201
base plaintext seed = 12202
training = none
device = local CPU
```

所有16个结构复用同一 base plaintext、密钥顺序和预算，唯一变量是活动 cell。
E24 密钥必须与 E20 seed `8201`、E21 seed `9201`、E22 seed `10201` 和 E23
seed `11201` 完全互斥。

## 3. Anchor、有限密钥控制与签名

`cell15` 的 Hwang 18维 basis 必须分别在 discovery、validation 两半成立，128-key
joint kernel 必须满足：

```text
rank/nullity = 46/18
joint span exactly equals Hwang Table 2 span
```

64-key half 可以包含额外随机秩亏方向，只有128-key joint kernel 计入稳定多样性。
每个 basis 必须通过自己的 `M u = 0` 校验，joint basis 还需分别在两个 half 上
复核。`cell0` joint kernel 不得包含任一 Hwang anchor basis direction。

签名使用确定性 GF(2) kernel basis 的十六进制序列，不枚举高维 span。另报告
discovery basis 在 validation 的存活比例。

## 4. Readiness 与推进门

协议门：

```text
SKINNY Appendix B 32-round public vector passes
exact active cells 0..15 and anchor/control ownership
Hwang Table 2 basis rank = 18
128 keys unique; discovery/validation halves disjoint
E24 keys disjoint from E20/E21/E22/E23
parity_rows.npy shape = 16 x 128, dtype = uint64
all bases validate their matrices and both halves
all metrics finite
```

推进门全部满足：

```text
cell15 expected span validates on both halves
cell15 joint rank/nullity = 46/18
cell15 joint span exactly equals Hwang 18-dimensional span
nontrivial joint-kernel structures >= 6 / 16
distinct nontrivial joint-kernel signatures >= 4
mean discovery-basis validation survival >= 0.50
cell0 excludes every Hwang anchor direction
```

裁决：

```text
pass:
  decision = innovation2_skinny_r7_single_cell_kernel_diversity_ready
  next = E25 structure-mask label construction and shortcut audit

hold, anchor fails:
  decision = innovation2_skinny_r7_single_cell_anchor_not_reproduced
  next = audit fresh-key stability; do not train

hold, diversity/control fails:
  decision = innovation2_skinny_r7_single_cell_kernel_not_diverse
  next = stop SKINNY position geometry; reproduce Hwang SPECK7 main case

fail:
  decision = innovation2_skinny_r7_single_cell_protocol_invalid
  next = repair vector, active-cell ownership, keys, cache, or GF(2) logic
```

任何分支都保持 `training=no`、`remote_scale=no`。不得重新打开 E23 的8轮 pair
枚举，也不得通过增加 seed 或网络移动冻结门槛。

## 5. 产物与执行

run id：

```text
i2_skinny64_r7_single_cell_geometry_128keys_seed0_20260717
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

执行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/audit-innovation2-skinny-r7-single-cell-diversity \
  --run-id i2_skinny64_r7_single_cell_geometry_128keys_seed0_20260717 \
  --output-root outputs/local_audits/i2_skinny64_r7_single_cell_geometry_128keys_seed0_20260717
```

冻结缓存必须可重放出完全相同的 gate。最终 SVG 必须执行 `visual-qa-redraw` 真实
像素检查并刷新最近结果索引。

## 6. Claim 边界

E24 只是本地128把 sampled-key 的结构几何 readiness。即使通过，也不等于
paper-scale、全密钥证明、神经训练、7轮预测突破或新积分性质证明。E25 仍须构造
正负 structure-mask 标签，并以 accuracy 和 AUC 同时审计 structure-only、
mask-only、bit-pattern、prevalence 与 group-disjoint 捷径。

## 7. 2026-07-17 E24 结果

真实运行完成 `16 x 128 x 16 = 32768` 次7轮加密。固定 base plaintext 为
`0xACE6837A36C7A5B3`；E24 的128把密钥与 E20/E21/E22/E23 全部互斥。公开向量、
16-cell所有权、密钥拆分、Hwang basis rank、缓存形状、全部 GF(2) basis 与 joint
basis 两半复核均通过。

只有底行四个 cell 的128-key joint kernel 非平凡：

| 活动 cell | discovery | validation | joint | 稳定签名 |
|---:|---:|---:|---:|---|
| `12` | `46/18` | `46/18` | `46/18` | 独立18维签名 A |
| `13` | `46/18` | `46/18` | `46/18` | 独立18维签名 B |
| `14` | `46/18` | `46/18` | `46/18` | 独立18维签名 C |
| `15` | `46/18` | `46/18` | `46/18` | Hwang Table 2 exact span |

`cell0..11` 的 joint matrix 全部为 `64/0`。这些位置在单个64-key half 偶尔出现
1-2个高重量随机方向，但没有任何方向跨 half 存活。`cell15` 在 discovery、
validation、joint 三个 split 均精确等于 Hwang 18维 span；`cell0` 不包含任何
Hwang direction。

最终指标：

```text
nontrivial joint-kernel structures = 4 / 16  (gate >= 6)
distinct joint signatures          = 4       (gate >= 4)
mean discovery-basis survival      = 1.000   (gate >= 0.50)
control contains anchor direction  = false
```

最终裁决：

```text
status = hold
decision = innovation2_skinny_r7_single_cell_kernel_not_diverse
training = no
remote_scale = no
```

这项结果比“只有一个论文 anchor”更强：底行四个位置形成四个不同且完全跨 half
稳定的18维输出子空间，证明7轮 kernel 对活动位置敏感。但非平凡结构仍只有
`4/16`，低于预注册的 `6/16` 标签宽度门。因此停止 SKINNY 单活动位置几何，不加
seed、不训练网络，也不重新打开8轮 pair 枚举。

冻结 `keys.npy` 与 `parity_rows.npy` 必须重放得到完全相同的 gate。最终
`curves.svg` 采用标明的 symlog 低值放大轴，同时显示0-2维 half噪声与18维稳定
kernel；以 `1800 x 845` 像素通过 `visual-qa-redraw`，无重叠、裁切、缺字、尺度
歧义或图例冲突。

## 8. 推荐下一步：E25 Hwang SPECK32/64 协议就绪审计

Hwang Section 6.2 / Table 7 是下一个正文主案例：标准32-bit parity feature 下，
6轮 kernel 有9维 basis，7轮存在唯一方向
`b2 xor b9 xor b16 xor b18 xor b25`；Table 1 还按末轮仿射边界记为7/8轮。该案例
可以增加 ARX 算法覆盖并保持“输入结构 -> 输出 mask性质”的任务定义。

但 Hwang 正文只说复用 Wang et al. 2020 文献[32] 的 `2^30` chosen-plaintext
结构，没有列出30个活动 bit。直接猜测“除两个固定 bit 外全部活动”会改变积分结构
和轮边界，不能作为论文复现。因此 E25 的第一步不是加密或训练，而是协议审计：

```text
exact mismatch to resolve = 文献[32]的30个活动bit、两个固定bit取值、左右16-bit字
                             顺序、LSB-first输出编号、6/7与7/8轮边界
same-budget anchor = Hwang Table 7(a) 9维与Table 7(b) 1维 exact kernel
implementation anchor = 项目Speck32_64公开向量和key-word顺序测试
execution = 先获取/解析文献[32]；协议明确后再设计可恢复的2^30结构生成
advance = exact active mask与round boundary均有一手来源，且小规模结构生成校验通过
stop = 不用猜测mask启动2^30枚举，不训练网络，不上远程GPU
decision unlocked = 是否能执行Hwang SPECK主案例及后续跨结构标签族构造
```

由于 `2^30` plaintext/key 远大于当前本地审计，协议明确后还必须先设计分块 XOR、
磁盘进度与断点恢复；在这些条件满足前不得启动实验。

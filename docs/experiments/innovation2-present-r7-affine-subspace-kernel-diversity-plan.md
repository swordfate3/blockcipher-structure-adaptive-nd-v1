# 创新2 E30：PRESENT-80 7轮16维线性子空间 kernel 多样性就绪计划

日期：2026-07-17
状态：冻结128-key完整审计已完成 / hold / 随机orientation路线停止

## 1. 研究问题

E27-N 已按冻结门停止SPECK固定pair路线。PRESENT坐标活动块曾产生稳定kernel，
但后续被位置/mask捷径和fresh-key不稳定性限制；SKINNY坐标位置族也过窄。
E30只改变一个变量：输入集合的16维子空间orientation。

```text
anchor    = 4个坐标16维子空间（连续16 bit活动）
candidate = 32个确定性随机、RREF规范化且互不重复的16维线性子空间
cipher    = PRESENT-80
rounds    = 7
```

网络仍不训练。目标是判断一般线性子空间能否形成足够多、跨密钥稳定且不同的完整输出
joint kernel，从而构造新的 `subspace x output-mask` 标签benchmark。

## 2. 文献与创新边界

Carlet等关于S-box积分抗性的工作直接以k维affine space的division property为对象；
Zhang等2026神经启发积分工作也从线性子空间plaintext集合产生候选。因而“使用线性/
仿射子空间”本身不是首创点。E30的候选价值仅在于：为结构条件输出性质预测构造
orientation-disjoint、跨密钥经验kernel标签，并在训练前执行强捷径门。

有限密钥joint kernel不是全密钥证明；E30不得宣称发现确定性积分区分器，也不得把
神经候选生成加传统验证包装为首创。

## 3. 冻结协议

```text
subspace dimension        = 16
plaintexts per structure  = 2^16 = 65536
random orientations       = 32
coordinate anchors        = 4
keys                      = 128 = 64 discovery + 64 validation
key seed                  = 13001
subspace seed             = 13002
translation               = 0（本门只审计orientation，不混入affine offset）
output                     = 64-bit ciphertext XOR parity, project bit order
execution                  = local CPU, NumPy vectorization
disk cache                 = basis/points/parity/completed/metadata per structure
training                   = none
```

64-bit输出若只使用 `32+32` keys，joint矩阵恰为随机64×64方阵，纯随机秩亏也很常见，
会制造假非平凡kernel。因此本门冻结为 `64+64`；不得为缩短运行改回32-key half。

随机16维子空间必须用GF(2)满秩basis生成，并转换成唯一RREF规范形式后去重；不得把同一
子空间的不同basis计作独立结构。坐标锚点与随机候选也必须RREF后互斥。每个结构的
`2^16` plaintext points、64把密钥parity、进度和完成位图逐结构落盘，可参数匹配续跑。

## 4. 同预算基线与控制

四个坐标锚点分别活动bit `0..15`、`16..31`、`32..47`、`48..63`，与随机orientation
保持相同维度、明文数、密钥、轮数和GF(2)核计算。不得改变输出bit order、密钥定义、
discovery/validation切分或kernel算法。

本门只审计标签存在性与多样性，不以“随机orientation优于坐标锚点”作为神经性能
结论。下一门若开放，必须再加入basis重量、bit覆盖率、position/mask identity、线性
边际、标签打乱和orientation-disjoint拆分。

## 5. 裁决门

每个结构分别报告discovery、validation和joint rank/nullity、joint kernel规范签名，
以及基不变的half-retention：`joint_nullity / min(discovery_nullity,
validation_nullity)`。不得使用“某个任意discovery basis向量是否逐个存活”替代子空间
交维度，因为该值依赖消元basis选择。

```text
affine_kernel_family_ready:
  32个随机orientation中非平凡joint kernel >= 8；
  随机orientation的不同非零joint签名 >= 4；
  非平凡候选平均half-retention >= 0.50；
  64把密钥、RREF唯一性、缓存、bit order和GF(2)回代全部通过。
  -> 进入E31 subspace x mask标签宽度与orientation-disjoint捷径审计；仍不训练。

affine_kernel_family_too_sparse:
  协议有效，但任一数量/多样性/存活门未过。
  -> 停止PRESENT随机orientation benchmark；不增加seed、不调维度挑结果。

protocol_invalid:
  basis不满秩/RREF重复、坐标锚点重叠、密钥/缓存/bit order/GF(2)任一失败。
  -> 只修协议，不解释kernel信号。
```

## 6. 执行与停止边界

预计总加密量为 `36 x 128 x 65536 = 301,989,888` 个PRESENT加密，属于本地
readiness/audit，不使用远程GPU。先做2个随机子空间、4把密钥的实现smoke；smoke
通过后自动运行冻结36结构审计。

禁止路线：

```text
不重新打开SPECK offset扫描；
不扩SKINNY全部120个pair；
不在E30改变subspace维度、round或translation；
不从E30挑最好structure直接训练；
不把经验kernel称为确定性积分证明。
```

权威产物规划：

```text
outputs/local_audits/i2_present_r7_linear_subspace_kernel_diversity_128keys_seed0_20260717/
  results.jsonl
  gate.json
  metadata.json
  progress.jsonl
  bases.npy
  parity_rows.npy
  curves.svg
  visual_qa_passed.marker
```

## 7. 实现与 smoke 就绪证据

E30已实现RREF规范化、随机满秩子空间去重、`2^16` 点枚举、PRESENT-80向量化、
逐结构磁盘缓存、64-bit输出parity、GF(2)三段kernel、half-intersection retention、
裁决和中文可视化。聚焦测试 `63 passed`。

预注册smoke使用4个坐标锚点、2个随机orientation和4把密钥，结果：

```text
status = pass
decision = innovation2_present_linear_subspace_readiness_passed
all 11 readiness checks = true
resume generated rows = 0
official PRESENT vector + vectorized path = pass
training performed = false
```

4-key smoke的nullity和签名受行数下界支配，只能证明实现就绪，不作为标签证据，也不
进入最近实验索引。下一步按本计划原样运行32个随机orientation和 `64+64` 把密钥；
不得根据smoke指标调整seed、维度、轮数或门槛。

## 8. 冻结完整审计结果

权威run：

```text
i2_present_r7_linear_subspace_kernel_diversity_128keys_seed0_20260717
```

实际完成规模与预注册一致：

```text
structures                  = 36 = 4 coordinate + 32 random
keys                        = 128 = 64 discovery + 64 validation
plaintexts/structure/key    = 65536
total PRESENT encryptions   = 301989888
training_performed          = false
```

11项协议与resume门全部通过。四个同预算坐标锚点保留了非平凡joint kernel：

| 结构 | discovery nullity | validation nullity | joint nullity | half retention |
|---|---:|---:|---:|---:|
| `coordinate_0` | 8 | 8 | 8 | 1.0 |
| `coordinate_1` | 4 | 4 | 4 | 1.0 |
| `coordinate_2` | 4 | 5 | 4 | 1.0 |
| `coordinate_3` | 7 | 5 | 5 | 1.0 |

因此本次负结果不是PRESENT实现、bit order、GF(2)消元或锚点校准失败。与之相对，
32个随机orientation的完整结果是：

```text
nontrivial joint kernels             = 0 / 32
distinct nonzero joint signatures    = 0
mean half-intersection retention      = 0.0
```

冻结裁决：

```text
status       = hold
decision     = innovation2_present_linear_subspace_kernel_family_too_sparse
training     = no
remote_scale = no
```

科学解释限于：PRESENT-80 r7的坐标16维子空间保留经验输出kernel，但任意随机
`GL(64,2)` orientation没有形成可训练的跨密钥kernel族。不得外推成“所有仿射
子空间都没有积分性质”，也不得把有限128-key证据称为全密钥证明。

结果已进入 `outputs/00_RECENT_RESULTS.md` 的 `001`。`curves.svg` 已按原生交付尺寸
对应的 `1469x790` 像素执行 `visual-qa-redraw`：标题、说明、图例、坐标轴、36个结构
ID、底部指标和裁决均无重叠、裁切或歧义。

## 9. 推荐下一步

停止随机orientation，不改seed、维度或轮数继续挑选。E15已经完成并否定了坐标块的
P-layer一阶/二阶轨道扩展，因此也不能把“topology-preserving”简单解释成再次运行
P-layer orbit。

下一步先做确定性候选提供者契约审计：检查division-property、monomial-prediction和
algebraic-transition-matrix工具能否输出与本项目目标完全相同的
`input affine set x linear output mask` 标签，而不是更宽的高阶输出单项式或广义积分
关系。该审计将决定是构建高轮PRESENT候选标签atlas，还是改用可精确枚举的小状态
SPN生成训练标签并执行跨密码拓扑迁移。两条路线都必须先过标签宽度、fresh-key和
组外捷径门，之后才开放神经训练。

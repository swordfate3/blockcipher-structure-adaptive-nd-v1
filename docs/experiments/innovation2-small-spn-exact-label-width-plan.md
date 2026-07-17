# 创新2 E32：小状态SPN全密钥精确标签宽度与组外捷径审计计划

日期：2026-07-18

状态：完整审计已完成 / hold / 标签宽但被ID边际解释

## 1. 研究问题

E31确认当前公开确定性提供者不能直接给出本项目需要的完整0/1标签：CLAASP-MP当前
依赖不可用的Gurobi环境；ATM预计算只给key-independent unknown constant或广义关系，
且未命中不是完备负类。E32不改变target，而是在可完整枚举的小状态SPN上建立精确
标签门：

```text
input  = SPN S-box/P-layer topology + rounds + coordinate affine set + linear mask
label  = 对冻结toy key family的全部256把master key，集合输出XOR是否恒为0
```

研究问题是：这种精确标签能否同时满足正负类宽度、跨cipher变体差异和组外反捷径门，
从而值得在E33训练small GraphGPS与SCGT。

## 2. 与真实密码结果的边界

E32是算法学习benchmark readiness，不是PRESENT/GIFT/SKINNY攻击结果。16-bit状态、
8-bit toy master key和合成S-box/P-layer只用于让输入集合及整个key space都可穷举。

允许的结论：

```text
结构条件积分性质预测在精确可验证的小SPN族上是否具有可学标签宽度。
```

禁止的结论：

```text
真实PRESENT高轮已被区分；
toy key family代表80-bit/128-bit实际key schedule；
在合成SPN上过门等于在真实密码上有效；
GraphGPS/SCGT已经有效（E32不训练网络）。
```

## 3. 冻结合成密码族

```text
state bits            = 16 = 4 nibbles
S-box variants        = 4
P-layer variants      = 4
cipher variants       = 16 Cartesian products
rounds                = 2,3,4,5
master keys           = all 0..255
round-key derivation  = fixed deterministic 8->16 bit nonlinear expansion
training              = none
```

`S0`固定为PRESENT 4-bit S-box；`S1..S3`由固定seed生成互异的4-bit双射。`P0`固定为
16-bit transpose式SPN bit permutation；`P1..P3`由另一固定seed生成互异的16-bit
permutation。所有查表和permutation必须写入`metadata.json`，不得运行后换seed挑结果。

每个round key由master-key byte、round index和固定8-bit双射确定，且对每个round保持
256把master key的映射可复现。该key schedule只定义toy全密钥范围，不宣称安全性。

## 4. 输入结构与输出mask

输入集合全部以inactive bits为0的coordinate linear set表示：

```text
4个 single-active-nibble 结构，dimension=4
6个 active-nibble-pair 结构，dimension=8
4个 active-nibble-triple 结构，dimension=12
total structures/cipher/round = 14
```

每个结构完整枚举 `2^dimension` 个plaintext，不抽样。固定64个非零线性输出mask：

```text
16 single-bit masks
4 full-nibble parity masks
16 deterministic weight-2 masks
28 deterministic weight-3..8 masks
```

mask集合在全部cipher变体间相同，便于S-box/P-layer组外比较；随机部分固定seed并去重。

完整标签格：

```text
16 ciphers x 4 rounds x 14 structures x 64 masks = 57344 labels
```

每个`cipher/round/structure`先生成256行16-bit输出XOR word，再对64个mask计算线性奇偶。
标签为1仅当256把master key全部为0；不是“多数key平衡”。

## 5. 组外拆分

变体按S-box/P-layer轴冻结拆分：

```text
train topology block = S0..S2 x P0..P2       (9 variants)
unseen S-box         = S3 x P0..P2           (3 variants)
unseen P-layer       = S0..S2 x P3           (3 variants)
dual unseen          = S3 x P3               (1 variant)
```

E32不训练神经网络，但在每个heldout split上计算只使用train block标签的平滑lookup
边际基线：global、mask-only、round+mask、structure+mask和
round+structure+mask。若这些ID边际已经高AUC解释heldout标签，图网络没有必要。

## 6. 同预算控制与就绪检查

1. `S0/P0`标量与向量化加密逐值一致；
2. 每个S-box和P-layer都是双射且16个组合唯一；
3. 256把master key全部覆盖且每个结构无抽样；
4. 14个输入结构和64个mask固定、互异；
5. 每行标签由all-key parity直接重算；
6. 保存`labels.npy`、结构/mask清单、metadata和progress；
7. 相同参数resume不得重新生成已完成cipher/round/structure块；
8. 不使用神经模型、标签增强或人工类别平衡。

## 7. 裁决门

```text
small_spn_exact_label_family_ready:
  protocol checks全部通过；
  positive labels >= 5000；
  negative labels >= 5000；
  distinct 64-mask label signatures >= 64；
  四个split各自positive与negative均 >= 256；
  同一(round,structure,mask)在16个cipher中发生标签变化的cell >= 10%；
  unseen-S和unseen-P最强ID边际AUC <= 0.80；
  dual-unseen最强ID边际AUC <= 0.75。
  -> E33实现deterministic baseline、small GraphGPS、SCGT三行同预算训练矩阵。

small_spn_exact_label_shortcut_dominated:
  标签宽度存在，但任一组外ID边际超过停止线。
  -> 不训练图网络；重新设计合成cipher拆分或结构/mask族，不增加模型容量。

small_spn_exact_label_too_narrow:
  正负数量、split覆盖、签名或cipher interaction不足。
  -> 停止当前toy family；不换seed挑结果。

protocol_invalid:
  双射、标量/向量、全key覆盖、完整结构枚举、缓存或标签重算失败。
  -> 只修协议，不解释标签。
```

## 8. 规模与执行位置

最大单块为 `256 keys x 4096 plaintexts` 的16-bit状态，按
`cipher x round x structure`逐块向量化并立即落盘。总标签只有57344，属于本地CPU
精确审计；不使用远程GPU。先以`2 S-box x 2 P-layer x rounds 2,3`运行实现smoke，
smoke只证明协议；通过后自动运行冻结16变体完整审计。

权威run规划：

```text
outputs/local_audits/i2_small_spn_exact_label_width_16ciphers_256keys_seed0_20260718/
```

## 9. 下一步网络矩阵

只有ready分支才创建E33训练计划。E33固定同一标签表、拆分、参数预算、epoch和seed，
只比较：

```text
1. deterministic/marginal baseline
2. small GraphGPS
3. SCGT = AllSet basis encoder + GraphGPS cipher encoder + mask query
```

必须另加label shuffle、topology shuffle、basis-row permutation/RREF和dual-disjoint控制。
若GraphGPS不优于边际，不实现更重的TokenGT；若SCGT不优于GraphGPS，不保留AllSet分支。

## 10. 完整结果

权威run：

```text
i2_small_spn_exact_label_width_16ciphers_256keys_seed0_20260718
```

所有10项协议门通过，缓存resume生成块数为0。冻结标签格完整产生：

```text
total labels                     = 57344
positive labels                  = 30595
negative labels                  = 26749
distinct 64-mask signatures      = 126
cipher-variable base cells       = 710 / 3584 = 0.198103
```

四个split均有充足正负类，全部标签宽度门通过。但只用9个训练拓扑标签构造的
`round+structure+mask` lookup在heldout上的AUC为：

```text
unseen S-box     = 0.987248
unseen P-layer   = 0.983952
dual unseen      = 0.986291
```

三项均远高于预注册停止线。因此裁决是：

```text
status       = hold
decision     = innovation2_small_spn_exact_label_shortcut_dominated
training     = no
remote_scale = no
```

这说明精确标签数量已经不是瓶颈；同一轮数、coordinate结构和绝对mask身份在不同
合成cipher间过于一致，图网络即使高分也可能只是重放该边际。

`curves.svg` 已按SVG原生交付尺寸对应的约 `1448x776` 像素执行
`visual-qa-redraw`。标题、两面板、数值标签、五种边际、停止线图例、裁决和证据范围
均无重叠或裁切。

## 11. 推荐下一步

不改变cipher seed、不重新加密。使用现有精确标签做E32b matched-contrast重裁决：只看
9个train topology，对每个 `(round,structure,mask)` 统计9个标签；仅保留train内部
同时有正负类的cell，即positive count在 `1..8`。选择规则不得读取三个heldout split。

预审显示该规则保留589个base cell，并把三种heldout的最强train-derived ID边际降到
`0.776/0.743/0.727`。这些数字只用于冻结E32b门，必须由独立postprocess产物重算、
验证和索引后，才决定是否开放E33训练。

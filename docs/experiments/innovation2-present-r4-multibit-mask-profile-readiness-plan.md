# 创新2 E69：PRESENT四轮多bit linear-mask profile非平凡性审计计划

日期：2026-07-18

状态：已完成 / hold / componentwise主导，decoder关闭

## 1. 研究问题

E68确认了64个unit output bit上的`Prefix-Guided Nodewise Profile Operator`。E69判断E43其余
236个多bit linear mask是否提供新的mask-query学习问题，还是仅由unit标签的简单组合决定。

```text
families = nibble / player_pair / same_nibble_pair / adjacent_nibble_pair
input    = 8-bit活动结构 + 多bit linear output mask
target   = 该mask是否对所有key/offset保持XOR平衡
```

E69不训练，不增加decoder，不改变E43标签。

## 2. 决定性语义分解

对每个多bit mask，读取同一structure下对应unit bits的E43三态标签：

```text
trivial positive    = multi-bit positive 且所有component unit均为positive
nontrivial positive = multi-bit positive 但至少一个component unit非positive
negative            = E43 concrete combined-mask witness
```

`all component units positive`作为强确定性分数，在完整matched validation上和每个mask family
分别计算AUC。若所有正类都是trivial，或该基线近乎完美，则当前provider没有提供多bit消去
关系，禁止训练mask-query decoder。

## 3. 冻结重排与强基线

每个family独立使用E43 checkerboard selector，之后合并：

```text
split      = E43 structure index mod 4
edge reuse = 每个(structure, mask)最多一次
balance    = 每个structure和mask在各自split内正负精确平衡
unknown    = 不参与，不写成negative
```

在相同行上使用train-only ridge比较static set、fair-corrupted P reachability、true P
reachability和ANF r1--r3 prefix；但componentwise unit-status是首要语义控制。

## 4. 预注册门

协议与宽度：

```text
E43 source/hash/96x300标签重放                pass
四个非unit family覆盖全部236 masks            pass
combined train positive/negative              each >= 250
combined validation positive/negative         each >= 80
combined train/validation structures           >= 48 / >= 16
每个family train positive/negative             each >= 40
每个family validation positive/negative        each >= 12
validation masks均在对应family train出现        yes
duplicate edges                               0
每个structure/mask class delta                0
global/mask/family/active边际最强AUC            <= 0.55
```

非平凡性门：

```text
每个split nontrivial positive fraction         >= 0.10
每个family validation nontrivial positives     >= 8
componentwise all-positive validation AUC       <= 0.80
```

信号门：combined ANF prefix或true topology AUC至少`0.60`。全部通过才允许轻量mask-query
decoder readiness；非平凡性失败则关闭当前多bit扩展，不用更大Transformer补救。

## 5. 产物、范围和下一步

```text
run_id = i2_present_r4_multibit_mask_profile_readiness_20260718
output = outputs/local_audits/i2_present_r4_multibit_mask_profile_readiness_20260718
execution = local exact audit / no training / no remote GPU
```

通过也只说明PRESENT-80四轮多bit严格标签适合本地decoder readiness；失败则保留E68 unit
profile方法成果，并把下一瓶颈明确为需要cancellation-aware的严格positive provider。

## 6. 2026-07-18实际结果

E43 `96 x 300`标签、236个非unit mask、四个family、三态语义和source hash全部重放通过。
完整原始多bit positive分解：

| family | raw positive | nontrivial positive |
|---|---:|---:|
| nibble | 179 | 0 |
| player_pair | 608 | 0 |
| same_nibble_pair | 1151 | 0 |
| adjacent_nibble_pair | 316 | 0 |
| **combined** | **2254** | **0** |

这不是抽样偶然：E43当前positive certificate要求mask中每个选中输出bit的full-cube单项式都
缺失，因此multi-bit positive按构造就等价于所有component unit均为positive。

family独立checkerboard只得到：

```text
train       = 90 positive + 90 negative，34 structures，51 masks
validation  = 20 positive + 20 negative， 8 structures，15 masks
```

`nibble`没有可匹配rectangle，`adjacent_nibble_pair`在validation也没有rectangle；宽度门同样
失败。combined train-only ridge在validation的AUC：

```text
componentwise all-unit-positive = 1.000
static set                       = 0.600
fair-corrupted P reachability    = 0.535
true P reachability              = 0.735
ANF r1--r3 prefix                = 0.520
```

虽然正确P可达仍有`0.735`信号，但语义强基线已经完美解释标签，不能把它升级为新decoder路线：

```text
status   = hold
decision = innovation2_present_multibit_profile_componentwise_dominated
training = no
remote   = no
```

## 7. 裁决与推荐下一步

停止当前multi-bit mask-query decoder，不增加Transformer、query attention或远程规模。E68的
unit-output双seed结果保持有效，因为它预测的是每个unit性质本身，不依赖这个后验组合。

若继续多bit路线，标签提供者必须能严格证明“至少一个unit component并非universally
balanced，但combined linear mask仍因GF(2)消去而universally balanced”的positive；当前E43
support-absence provider做不到。下一结构实验优先转为E70跨活动维度unit-profile标签readiness：
先审计PRESENT四轮4-bit/12-bit coordinate cube是否有严格正负宽度、边际控制和与E68相同的
64-node输入契约，再决定是否测试已确认operator的跨维度迁移。

最终`curves.svg`经`visual-qa-redraw`渲染为`2200 x 1167`像素检查；family正类分解、零值、
componentwise AUC、确定性基线、裁决与范围均无重叠、裁切或歧义。

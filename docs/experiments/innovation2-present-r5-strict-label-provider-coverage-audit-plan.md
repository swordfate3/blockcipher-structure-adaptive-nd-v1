# 创新2 E52：PRESENT五轮严格标签证书提供者覆盖审计计划

日期：2026-07-18

状态：已完成 / hold / 五轮严格标签库未就绪

## 1. 研究问题

E43--E51已经完成PRESENT-80四轮严格标签上的架构筛选，最终确定性ANF/degree约`0.69`强于
所有纯神经与神经残差候选。继续在四轮换网络不再是有效研究投入。

E52审计真正的轮数瓶颈：能否在保持创新2原始“指定输出mask是否对所有key和inactive
offset保持积分XOR平衡”语义下，为PRESENT-80五轮产生足够的可证明正类和具体反例负类，
并构造可训练、无一元捷径的structure-disjoint benchmark。

E52不训练神经网络，不使用远程GPU。

## 2. 固定语义与来源

```text
cipher            = PRESENT-80
rounds            = 5
active dimension  = 8 bits
output query      = 64-bit nonzero linear mask
positive          = 对所有80-bit key和所有inactive offset，masked cube XOR恒为0
negative          = 存在具体(key, inactive offset)使masked cube XOR为1
unknown           = 当前提供者既不能证明positive，也没有找到negative witness
```

不得把有限密钥经验平衡率、固定key零XOR、随机负密文、结构化vs随机分类或单样本mask parity
替代严格标签。E43四轮atlas只作来源与宽度锚点，不混入五轮训练数据。

## 3. 提供者候选与单变量顺序

先完成文献/实现审计并冻结可执行提供者，不并行混合多个新算法。候选优先级：

```text
P0 = 当前sound active-variable ANF support superset（复现失败锚点）
P1 = 更精确的division-property / monomial-prediction certificate
P2 = P1无法落地时的可验证MILP/SAT cube-coefficient zero certificate
```

P1/P2必须输出可独立复验的证书或完整solver witness，不接受只有“solver says balanced”的不透明
布尔值。使用外部库前先核对论文语义、PRESENT bit order、key schedule、round convention和许可证；
不得把截断差分、普通division trail absence或固定key ANF误报为全密钥全offset证书。

只在P0覆盖审计完成后选择一个P1实现；P1失败才考虑P2。每次只改变证书提供者，structure、
mask、negative witness和匹配协议保持固定。

## 4. 冻结候选池与预算

```text
structure seed       = 20260718
structures           = 96（复用E43 8-bit coordinate cubes）
output masks          = 300（复用E43 mask集合）
candidate pairs       = 28800
negative witness bank = 16 keys x 8 offsets/structure，沿用E43确定性种子
scalar recheck        = 至少32个negative + 全部抽样positive证书边界fixture
device                = local CPU
```

若P1/P2完整池预计超过本地合理预算，先在固定`16 structures x 64 masks`子集做coverage readiness；
只有positive率、negative率和证书复验均非零才扩大到完整池。不得因计算慢直接改用远程训练。

## 5. 标签与反捷径门

完整池最低宽度：

```text
positive structures同时含negative       >= 48
可构造train matched checkerboard         >= 400 positive + 400 negative
可构造validation checkerboard            >= 118 positive + 118 negative
train/validation structures              disjoint
global/mask/family/active-bit unary AUC   <= 0.65
duplicate(structure, mask)                = 0
```

正确性门：

```text
PRESENT官方向量与round convention          pass
每个positive均有sound可序列化证书           pass
抽样positive独立边界/solver复验             pass
抽样negative标量加密复验masked XOR=1        pass
provider timeout/error与unknown分开记录      pass
```

全部通过：

```text
decision = innovation2_present_r5_strict_label_bank_ready
next     = 先做确定性shortcut/feature attribution，再冻结首个五轮神经矩阵
```

覆盖不足：停止五轮神经训练，记录最强提供者的positive/negative/unknown与失败原因；不得用
经验标签填充。协议无效：只修bit order、round/key、证书或复验协议，不解释覆盖率。

## 6. 产物与下一动作

```text
outputs/local_audits/i2_present_r5_strict_label_provider_coverage_20260718/

provider_manifest.json
structures.json
masks.json
labels.csv
certificates.jsonl
witnesses.jsonl
matched_contrast.csv（仅在门通过时）
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

E52完成后必须给出：最强提供者、覆盖率、证书/反例正确性、是否可训练、下一步是否开放网络、
明确关闭的替代语义与机械扩展。声明范围仅为五轮严格标签提供者覆盖，不是神经结果、五轮
区分器、新攻击或SOTA。

## 7. 正式执行结果

权威run：

```text
i2_present_r5_strict_label_provider_coverage_20260718
```

P0完整执行了冻结的`96 x 300 = 28800`候选池：

```text
positive                              = 0
negative                              = 27446
unknown                               = 1354
positive / negative mixed structures = 0
support size min / max                = 256 / 256
fully saturated output supports       = 6144 / 6144
sampled negative scalar recheck       = 32 / 32 pass
```

P0的正确性门全部通过，但五轮每一个`structure x output bit`都已包含全部256个活动变量单项式。
因此support overapprox无法证明任何full-cube monomial缺失，无法构造train/validation
checkerboard。这是标签提供者覆盖失败，不是五轮性质不存在，也不是神经网络失败。

P1静态与运行环境审计：

```text
CLAASP source commit                  = f2239d639ae5c4a013947ce9121c6f4464584758
PRESENT-80 configurable-round model  = present
full superpoly API                    = present
independent cube-sum verifier         = present
non-Gurobi monomial backend           = absent
Sage                                  = 9.5 available
bitstring in Sage runtime             = absent
gurobipy                              = absent
Gurobi license                        = not checked because package is absent
preinstalled relevant Docker image    = absent in the read-only image audit
```

## 8. P1语义修正

五轮目标要求对所有inactive offset成立。CLAASP-MP的
`find_keycoeff_of_cube_monomial_of_specific_output_bit`会把全部非cube明文位固定为0，只能
覆盖零offset，不能作为当前正类证书。

真正匹配目标的是`find_superpoly_of_specific_output_bit`：它保留非cube明文变量和key变量。
对于多bit output mask，必须把所选输出bit的完整superpoly在GF(2)上异或，并证明所得多项式
恒为0。执行前还必须对拍CLAASP的MSB-first output index与本项目LSB-first bit编号。当前仅完成
源码语义核对，没有执行CLAASP-MP，不能声称复现其结果。

## 9. 裁决

```text
status       = hold
decision     = innovation2_present_r5_strict_label_bank_not_ready
training     = no
remote_scale = no
```

明确关闭：有限密钥经验平衡标签、零offset key-coefficient标签、PRESENT四轮继续调网络、
seed1、更长epoch和五轮标签门前的远程GPU扩展。E52不是神经结果、五轮区分器、攻击或SOTA。

## 10. 推荐下一步：E53开放3SDP求解门

本机Sage 9.5的`MixedIntegerLinearProgram`已用`GLPKBackend`通过一变量binary MILP fixture，
所以不把“没有Gurobi”直接当成路线终点。E53只改变provider实现，保持E52 structure、mask、
witness bank和门槛不变：

```text
question       = 开放MILP后端能否正确实现3SDP trail枚举与GF(2)消去
anchor         = E52 P0 sound support overapprox
readiness      = PRESENT 1--2轮exact-ANF positive/negative与bit-order fixtures
candidate      = Sage/GLPK上的最小3SDP provider或可验证的等价开放实现
first scale    = 16 structures x 64 masks，P0固定子集
device         = local CPU
seeds/epochs   = not applicable
advance        = positive/negative均非零，证书复验全过，且相对P0新增严格正类
stop           = 只能做2SDP可达性、不能处理trail奇偶消去，或fixture不一致
remote/training= no
```

只有E53先过fixture、再过固定子集覆盖门，才扩大五轮完整池；完整池达到E52宽度与反捷径门后，
才重新开放五轮网络排名。首个网络矩阵仍应是确定性baseline、最强简洁神经锚点和一个结构候选，
不得提前枚举GraphGPS、Transformer、NBFNet或远程规模。

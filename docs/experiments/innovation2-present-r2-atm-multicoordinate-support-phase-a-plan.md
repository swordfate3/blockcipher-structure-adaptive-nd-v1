# 创新2 E61 Phase A：PRESENT两轮ATM多坐标消去支撑门计划

日期：2026-07-18

状态：已完成 / 暂缓 / 完整支撑路线关闭

## 1. 研究问题

E59和E60合计32条单坐标query均为严格constant，无法形成神经二分类标签。E61不继续扩大
singleton扫描，而是把每个坐标`(u,v)`表示成独立轮密钥上的完整GF(2) key-polynomial support，
判断多个坐标的support对称差能否形成：

```text
positive = 非零key-monomial support完全消去，relation对所有独立轮密钥为constant
negative = 对称差后保留具体非零odd key monomial，并可逐坐标精确重放
```

Phase A只审计标签oracle和低重量relation存在性，不实现或训练神经网络。

## 2. 冻结坐标池

固定PRESENT两轮、输出cell 0和输入cell 0：

```text
input exponent u   = 0..15，覆盖cell 0的全部4-bit monomial
output exponent v  = 1..15，覆盖output cell 0的全部非空monomial
coordinate count   = 16 x 15 = 240
order              = u升序，再按v升序
```

这个笛卡尔积在读取标签前冻结，不按SAT结果挑位置。它同时覆盖`wt(u)=0..4`和`wt(v)=1..4`，
但只改变关系表示，不改变PRESENT轮函数、独立轮密钥语义或严格证书定义。

## 3. Provider与预算

沿用E60已校准的ATM原生PySAT adapter：

```text
rounds / independent key additions = 2 / 3
key variables                       = 192
wall-clock cap                      = 60 seconds total
projected-key cap                   = 2^12/coordinate
trail-model cap                     = 2^16/key/coordinate
execution                           = local CPU child worker
remote GPU                          = no
```

对每个坐标必须枚举包括key exponent 0在内的全部projected key masks。每个mask的trail数量奇偶
必须在cap内精确求出；只保存odd masks构成canonical support。任何projected/trail cap、worker超时
或未完成枚举都标为`unknown`，不得当作空support。

worker在模型完成后写`model.json`，每完成一个坐标立即追加`coordinate_supports.jsonl`和
`progress.jsonl`，父进程只解释完整落盘行。

## 4. 同预算锚点与唯一变化

```text
anchor       = E60，两轮、同SAT adapter、同60秒/2^12/2^16预算
one change   = 单坐标constant/key-dependent -> 导出完整key-polynomial support
fixed        = cipher、rounds、key model、cap语义、bit order、unknown定义
```

E60的16条锥内/锥外查询仍是来源校准，不与新池混成训练行。

## 5. Phase A门

```text
E60 gate与16/16 scalar validation通过               pass
model rounds/key additions/key variables正确         pass
completed coordinates                               >= 64/240
unknown fraction                                    <= 0.25
exact key-dependent coordinate supports             >= 16
all saved odd masks independently replay odd         pass
nontrivial nullspace relations of size 2--4          >= 4
matched same-size strict negative pairs              >= 4
all negative relation witnesses replay XOR odd        pass
```

低重量positive必须由至少两个不同坐标组成，禁止`c XOR c`、重复坐标或空relation。negative必须
保持relation size，并尽量匹配输入重量多重集、输出重量多重集和坐标位置；不能用random、
`not in basis`或unknown冒充。

## 6. 标签泄漏与强基线

完整support只属于离线标签oracle和证书，不进入未来神经输入。否则确定性对称差直接得到标签，
神经比较没有意义。未来可见输入仅包含：

```text
relation coordinate set + PRESENT S-box/P-layer graph + rounds
```

Phase B首先审计的廉价基线为relation size、输入/输出重量多重集、依赖锥签名、位置ID和单坐标
constant/key-dependent边际。exact support oracle只报告生成成本与正确性，不作为可见特征AUC。

## 7. 推进与停止

Phase A通过后，Phase B才扩大并构造至少`256 positive + 256 negative`，使用coordinate-disjoint、
input-monomial-disjoint和output-monomial-disjoint拆分，要求unknown不超过5%且最强廉价边际AUC
不超过0.80。之后才开放固定三行网络矩阵：

```text
cheap deterministic marginal baseline
coordinate-set DeepSets
Relation-Cipher Cross-Attention (RCCA)
```

RCCA的唯一结构增量是：relation坐标query token对共享PRESENT typed graph做cross-attention，再做
集合级置换不变pooling。同预算控制必须包含label-shuffle、wrong-P-layer和relation-token shuffle。

若Phase A没有足够key-dependent atoms、低重量nullspace relation或严格matched negative，则停止
当前两轮ATM多坐标路线；不增加SAT cap、不转远程GPU、不训练网络，也不把exact oracle输出泄漏给
模型。训练scale、seed和epoch在标签门通过前均为不适用。

## 8. 预注册运行

```text
run_id = i2_present_r2_atm_multicoordinate_support_phase_a_20260718
output = outputs/local_audits/i2_present_r2_atm_multicoordinate_support_phase_a_20260718
```

## 9. 正式结果

父进程在冻结60秒总墙钟终止worker，durable产物保留8条完整落盘坐标：

```text
planned / completed / exact / unknown = 240 / 8 / 7 / 233
exact constant / key-dependent         = 0 / 7
positive cancellation relations        = 0
matched strict negative pairs           = 0
worker                                  = timeout
decision                                = innovation2_atm_r2_multicoordinate_support_runtime_not_ready
```

来源、坐标池和两轮独立轮密钥模型检查全部通过；7条exact坐标的全部odd masks也逐项重放为odd。
失败来自支撑与trail复杂度，不是来源或实现契约无效：

```text
u=0, output weight 1: 41 nonzero terms, approximately 0.11--0.13 s/query
u=0, output weight 2: 1763 nonzero terms, approximately 7.73--7.98 s/query
u=0, v=0x7 weight 3: first projected mask exceeds 2^16 trail cap, 31.45 s -> unknown
```

因此没有达到64条完成、16条key-dependent atoms、4条positive relation或4组matched negative
中的任何宽度门。8条落盘不是可训练数据，也不能从“前7条都是key-dependent”外推240坐标。

权威产物：

```text
outputs/local_audits/i2_present_r2_atm_multicoordinate_support_phase_a_20260718/gate.json
outputs/local_audits/i2_present_r2_atm_multicoordinate_support_phase_a_20260718/results.jsonl
outputs/local_audits/i2_present_r2_atm_multicoordinate_support_phase_a_20260718/coordinate_supports.jsonl
outputs/local_audits/i2_present_r2_atm_multicoordinate_support_phase_a_20260718/progress.jsonl
outputs/local_audits/i2_present_r2_atm_multicoordinate_support_phase_a_20260718/curves.svg
```

## 10. 裁决与推荐下一步

裁决为`hold`：关闭当前PRESENT两轮ATM完整key-polynomial支撑矩阵，不提高projected/trail cap，
不转远程GPU，不只保留低次数输出后宣称标签门通过，也不训练DeepSets或RCCA。

下一优先实验改为E62“小型SPN严格多坐标relation架构门”。研究问题是：在项目已有可穷尽、
无unknown的small-SPN精确标签源上，relation-coordinate set与真实S-box/P-layer图做cross-attention，
是否比同预算DeepSets和ID/重量边际稳定提升。唯一变化是标签oracle从不可扩展的PRESENT ATM完整
支撑换成可穷尽small-SPN；任务语义仍是多个坐标GF(2)关系，严格正负必须由全枚举证明。

E62应先生成`256/class`、structure-disjoint与relation-coordinate-disjoint标签，执行廉价边际和
wrong-P-layer门；只有标签门通过才训练`DeepSets vs RCCA`，固定2 seed、40 epochs、本地小规模。
如果RCCA没有逐seed超过DeepSets和最强边际，关闭该架构；如果通过，只能声称小型SPN方法证据，
不能自动迁移为PRESENT-80高轮结果。

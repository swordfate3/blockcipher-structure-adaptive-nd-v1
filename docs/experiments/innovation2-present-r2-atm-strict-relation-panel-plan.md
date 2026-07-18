# 创新2 E59：PRESENT两轮ATM严格relation标签面板readiness计划

日期：2026-07-18

状态：完成 / hold / singleton面板单一constant类别

## 1. 研究问题

E58-A证明ATM原生projected-SAT机制在S-box/toy上精确；E58-B证明九轮单候选在60秒内不可
执行。E59不重新增加九轮预算，而是测试同一provider在两轮真实PRESENT round function、独立
轮密钥模型下，能否形成一个同时包含严格constant和strict key-dependent的可执行标签面板。

```text
input  = rounds=2 + precursor exponent u + output monomial exponent v
label1 = exact枚举全部projected key monomial后无非零项（constant）
label0 = 具体非零key exponent的trail parity为odd并独立重放
unknown= projected/trail cap或总墙钟触发，或未形成完整证书
```

这是新广义relation任务的provider readiness，不是原五轮linear-mask balance任务。

## 2. 冻结面板

```text
cipher round function = PRESENT S-box + P-layer
rounds                = 2
key additions         = 3 independent 64-bit keys
input u               = 0xFFFFFFFFFFFFFFF0 (wt=60)
output v              = unit bits e0..e15
queries               = 16
wall-clock cap        = 60 seconds total
projected-key cap     = 2^12/query
trail-model cap       = 2^16/key/query
execution             = local CPU child worker
training              = no
remote                = no
```

worker必须在模型构造后写`model.json`，每完成一条query立即追加`panel.jsonl`。父进程超时后只
读取已完整落盘的行，不解释正在计算的query。

## 3. 同预算锚点与唯一变量

```text
source/adapter = E58-A commit与256/256 S-box calibration
failed anchor  = E58-B r9 singleton 60-second timeout
one variable   = rounds 9 -> 2
fixed          = basis semantics、SAT backend、key model、negative witness定义
```

不改变candidate后验顺序，不用`not in basis`作负类，不用有限密钥采样补标签。

## 4. Readiness门

```text
E58-A source/calibration gate通过                 pass
model rounds/key additions/key variables正确       pass
completed queries                                >= 12/16
unknown fraction                                 <= 0.25
strict constant rows                             >= 4
strict key-dependent rows                        >= 4
all negative witnesses nonzero and replay odd     pass
all cap outcomes remain unknown                   pass
```

若通过，只开放Phase B完整`16 input nibbles x 64 unit outputs = 1024`标签宽度与捷径审计；
仍不训练网络。若只有一个类别或简单P-layer reachability完全解释标签，停止该面板并重新设计
multi-coordinate cancellation relation，不直接实现Cross-Attention。

## 5. 预注册的网络结构（标签门后）

只有Phase B至少获得`256`严格constant、`256`严格key-dependent、structure-disjoint拆分和
边际匹配后，才开放：

```text
0. deterministic reachability + weight + position baseline
1. coordinate-set DeepSets
2. Relation-Cipher Cross-Attention (RCCA)
```

RCCA把每个`(u,v)`作为query token，对轮共享PRESENT typed graph做cross-attention，再对relation
坐标集合做置换不变pooling。唯一创新变量是query-conditioned cipher interaction；同预算、
同split、同seed、同epoch，并要求超过DeepSets、deterministic、label-shuffle和wrong-P-layer。

当前不实现RCCA，因为16-query readiness不构成训练数据。

## 6. 2026-07-18实际结果

权威run：

```text
i2_present_r2_atm_strict_relation_panel_20260718
```

模型与16条query均在本地完整执行：

```text
rounds / key additions / key vars = 2 / 3 / 192
CNF clauses / maximum variable    = 2080 / 576
model build                       = 0.00735 s
completed                         = 16 / 16
explicit unknown / timeout        = 0 / 0
strict constant                   = 16
strict key-dependent              = 0
projected nonzero key masks seen  = 0 for every query
median query time                 = 0.000609 s
```

每条constant都由“非零key monomial projected model为空”得到完整证书，不是有限密钥经验标签。
但面板是单一类别，未通过至少4条strict negative的冻结宽度门。

```text
status   = hold
decision = innovation2_atm_r2_strict_relation_panel_not_ready
training = no
remote   = no
```

推荐下一步不是直接扩大`e0..e15`到全部输出，也不是把`not in basis`标负类。应先根据两轮P-layer
依赖锥预注册同时覆盖cone内外的输出panel，并加入exact reachability强基线；若标签仍被可达性
完全解释，则singleton任务停止，转向由两个以上坐标GF(2)抵消定义的multi-coordinate relation。

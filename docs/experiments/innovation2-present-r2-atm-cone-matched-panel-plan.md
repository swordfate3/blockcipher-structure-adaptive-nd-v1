# 创新2 E60：PRESENT两轮ATM依赖锥匹配严格标签面板计划

日期：2026-07-18

状态：已完成 / 暂缓 / 单坐标路线关闭

## 1. 研究问题

E59的`wt(u)=60`远超两轮unit output最高代数次数`3^2=9`，16条全部exact constant。E60不
后验扫描64个输出位置，而是从PRESENT真实P-layer反向依赖锥构造同重量锥内/锥外配对，判断：

```text
1. provider能否产生两类严格标签；
2. 标签是否只是由简单依赖锥可达性完全解释；
3. singleton relation是否值得形成RCCA benchmark。
```

## 2. 冻结query构造

固定两轮output monomial `v=e0`。按真实P-layer反向两轮得到输入cone：

```text
inverse-P(e0) -> round2 S-box cell0
round2 cell0 inputs -> round1 source cells 0,1,2,3
two-round input cone = bits 0..15
```

对重量`w=1..8`各构造一对：

```text
inside(w)  = bits {0,...,w-1}
outside(w) = bits {0,...,w-2} union {16}
```

两条重量完全相同，唯一变量是最后一个输入bit在真实依赖锥内还是锥外。总计16条query，顺序
固定为`inside(1), outside(1), ..., inside(8), outside(8)`，不读取标签选query。

## 3. 标签与预算

沿用E59完全相同的严格标签契约：

```text
rounds / independent key additions = 2 / 3
wall-clock cap                     = 60 seconds total
projected-key cap                  = 2^12/query
trail-model cap                    = 2^16/key/query
constant                           = exhaustive no nonzero key monomial
key-dependent                      = nonzero odd key exponent + exact replay
unknown                            = any cap/incomplete outcome
```

本地CPU，不训练，不使用远程GPU。

## 4. 强基线与推进门

只在resolved行上评估：

```text
degree-only baseline       = wt(u)
cone-membership baseline   = all input bits inside the exact two-round cone
```

readiness：

```text
completed queries                         >= 12/16
unknown fraction                          <= 0.25
strict constant / key-dependent           each >= 4
all negatives replay odd                  pass
each weight keeps inside/outside pair      pass
degree-only AUC                           <= 0.65
cone-membership strongest-direction AUC   <= 0.80
```

若两类存在但cone AUC超过`0.80`，裁决为shortcut-dominated：停止singleton relation，不扩大
1024-query，不训练RCCA，转multi-coordinate GF(2) cancellation relation。只有两类和反捷径门
同时通过，才进入完整标签宽度审计。

## 5. 网络结构边界

E60不实现网络。RCCA仍冻结为：relation坐标query token对轮共享SPN typed graph做cross-attention，
集合级置换不变pooling；必须同预算超过deterministic、DeepSets、label-shuffle和wrong-P-layer。
singleton shortcut门失败时，这个模型没有研究价值，不能靠隐藏reachability字段或扩大模型补救。

## 6. 正式结果

冻结面板在本地CPU约2秒内完成：

```text
completed queries               = 16/16
strict constant                 = 16
strict key-dependent            = 0
unknown                         = 0
scalar-validated constant rows  = 16/16
scalar key sets                 = 3
decision                        = innovation2_atm_r2_cone_matched_panel_width_not_ready
```

SAT模型包含2080条CNF、192个独立轮密钥变量和三次key addition。每条constant不仅通过
“不存在非零key monomial”的穷尽证书，还重放key exponent 0得到常数值，并在三组固定独立
轮密钥上执行完整标量系数计算。第一条`inside(1)`恒为1，其余15条恒为0；16条均与SAT重放
一致。因此全constant不是worker超时、cap耗尽或依赖锥查询构造错误。

两类标签没有同时出现，`degree-only`和`cone-membership` AUC没有定义，不能把空缺AUC解释为
反捷径通过。E60没有神经训练、远程GPU、PRESENT-80真实主密钥标签、攻击或SOTA结论。

权威产物：

```text
outputs/local_audits/i2_present_r2_atm_cone_matched_panel_20260718/gate.json
outputs/local_audits/i2_present_r2_atm_cone_matched_panel_20260718/results.jsonl
outputs/local_audits/i2_present_r2_atm_cone_matched_panel_20260718/panel.jsonl
outputs/local_audits/i2_present_r2_atm_cone_matched_panel_20260718/summary.json
outputs/local_audits/i2_present_r2_atm_cone_matched_panel_20260718/curves.svg
```

## 7. 裁决与推荐下一步

裁决为`hold`：停止singleton relation扫描，不扩大到1024条，不实现或训练RCCA、DeepSets，
也不提高SAT cap或转远程GPU。原因是从E59高重量面板到E60低重量锥内/锥外面板，32条单坐标
查询都没有产生严格key-dependent类；扩大模型无法修复缺失标签。

下一实验是E61多坐标GF(2) cancellation relation。研究问题是：多个`(u,v)`坐标的key-polynomial
支撑做对称差后，能否构造严格零支撑positive与带具体odd key-monomial的negative。E61沿用E60
两轮独立轮密钥exact provider和相同硬cap，只把单个坐标改为2--4坐标集合；同relation size、
输入/输出重量和依赖锥签名配对，先比较deterministic symmetric-difference baseline。只有严格
正负各至少256、unknown不超过5%、组外拆分无泄漏且边际AUC不超过0.80，才开放三行同预算矩阵：

```text
deterministic cancellation baseline
coordinate-set DeepSets
Relation-Cipher Cross-Attention
```

E61标签门仍在本地CPU执行；没有标签门证据前，训练轮数、seed和远程路径均为不适用。

# 创新2 E55：PRESENT三轮query-cone sparse-ANF硬cap增长门计划

日期：2026-07-18

状态：完成 / hold / 三轮query触发500万项硬cap

## 1. 研究问题

E54证明全key/全offset full superpoly必须保留136个变量，dense GF(2) tensor至少有`2^136`
表项，不可执行。E53-A的稀疏exact ANF在一轮、二轮仍可计算，但完整输出项数从1907增长到
4352830，fixture最大superpoly从13增长到53392。

E55测试最后一个不改变标签语义的开放provider可能性：只沿指定output bit/mask的反向依赖锥
计算三轮exact sparse ANF，并使用硬单项式、内存和时间cap。它不计算全部64个输出，也不把
cap中止解释成unknown label之外的结论。

E55不训练网络，不使用远程GPU，不执行五轮标签池。

## 2. 固定协议

```text
cipher             = PRESENT-80
rounds             = 3
symbolic variables = 64 plaintext + 80 master-key
queries            = 8 unit output bits + 4 frozen multi-bit masks
active cubes       = E53-A中已对拍的一、二轮fixture活动集合
max unique terms   = 5,000,000 / query
max wall time      = 60 seconds / query
max resident memory= 4 GiB
device             = local CPU
```

12个query在运行前固定为E53-A二轮fixture：

```text
unit positive anchors = r2_positive_00..03
unit negative anchors = r2_negative_00..03
multi-mask anchors     = r2_multi_mask_00..03
```

也就是不根据三轮term count或标签后验挑选容易query。三轮继续使用各anchor的active bits与
output mask，只把rounds从2改为3。

先用同一query-cone实现重放E53-A全部一轮fixture和选定二轮fixture；任何hash、term count、标量
加密或multi-mask XOR不一致均为协议失败，不进入三轮。

## 3. 推进门

```text
r1/r2 calibration exact agreement       pass
all 12 r3 queries finish within hard cap pass
positive and negative exact labels       both nonzero
negative concrete witnesses              scalar pass
positive superpoly hashes/certificates   serializable
key and inactive variables               remain symbolic
median and max term growth               reported
```

全部通过后，下一实验才以相同cap运行四轮query-cone；四轮过门后才允许五轮固定`16x64`子集。
任何三轮query超过500万项、60秒或4GiB即停止当前sparse provider，不增加cap、不转远程GPU。

## 4. 控制与停止线

```text
control 1 = E53-A full exact ANF r1/r2
control 2 = multi-mask component XOR
control 3 = wrong P-layer query cone
control 4 = zero-offset-only coefficient，必须标记语义不匹配
```

E55失败后，当前开放全变量provider家族关闭；创新2保留PRESENT四轮严格标签方法学结果、确定性
ANF/degree约0.69与最强纯神经0.561979，以及五轮provider不可执行边界。不得重新开始四轮网络
枚举或用经验标签训练五轮网络。

## 5. 产物

```text
outputs/local_audits/i2_present_r3_query_cone_sparse_anf_growth_20260718/

query_manifest.json
progress.jsonl
certificates.jsonl
witnesses.jsonl
results.jsonl
gate.json
summary.json
metadata.json
curves.svg
visual_qa_passed.marker
```

## 6. 2026-07-18实际结果

权威run：

```text
i2_present_r3_query_cone_sparse_anf_growth_20260718
```

E53-A校准先执行完成：全部20个一轮fixture与冻结的12个二轮fixture，共`32/32`行。每行
superpoly项数/hash一致，unit输出多项式hash一致，query输出随机赋值与标量PRESENT一致；故意
identity P-layer控制被识别，zero-offset-only控制被拒绝。因此三轮增长不是位序或实现错配。

三轮按冻结顺序执行：

| query | 二轮来源 | 状态 | query内最大项数 | 三轮superpoly项数 | 耗时 |
|---|---|---|---:|---:|---:|
| `Q00` | `r2_positive_00` | 完成 / negative | `2,149,131` | `12` | `4.2843s` |
| `Q01` | `r2_positive_01` | 完成 / negative | `1,417,246` | `12` | `2.7638s` |
| `Q02` | `r2_positive_02` | 完成 / negative | `1,775,929` | `13` | `3.1409s` |
| `Q03` | `r2_positive_03` | `term_cap_exceeded` | `5,000,001` | unknown | `3.2048s` |
| `Q04--Q11` | 其余冻结query | 按停止线跳过 | - | unknown | - |

三个已完成query的具体负类反例均通过标量PRESENT复验。Q03在远低于60秒时已经越过500万项，
峰值驻留内存约`1.002 GiB`；这表明首个失败原因是exact稀疏项组合爆炸，不是时间耗尽或缺少
GPU。Q03部分多项式没有被当作标签，后续query也没有执行。

```text
status   = hold
decision = innovation2_present_r3_query_cone_sparse_anf_hard_cap_exceeded
training = no
remote   = no
```

最终`curves.svg`按`visual-qa-redraw`渲染为`2261x1145`像素检查。首次预览右侧校准标签拥挤、
底部下一步残留英文；改为横向校准条并映射中文裁决后重绘，标题、坐标、门线、Q00--Q11、
校准项、裁决和证据范围均无重叠、裁切、缺字或歧义，并记录`visual_qa_passed.marker`。

## 7. 推荐下一步

按预注册停止线关闭当前exact full-variable provider家族：不提高500万项/60秒/4GiB cap，不转
远程GPU，不使用Q03部分结果，不以经验平衡补正类，不进入四轮/五轮query-cone，也不重新开始
E43四轮神经架构枚举。

下一实验不是训练，而是E56“广义积分关系输出预测契约审计”：核验项目已有Algebraic
Transition Matrices预计算basis能否给出PRESENT真实key schedule下、可序列化且正负可复验的
`input exponent/subspace + output monomial relation`标签，并明确它与当前linear-mask XOR=0任务
的可逆/不可逆映射。只有存在足够、非捷径、group-disjoint标签，才为关系查询任务重新开放
最小神经矩阵；否则创新2冻结为四轮严格标签方法学结果与五轮provider不可执行边界。

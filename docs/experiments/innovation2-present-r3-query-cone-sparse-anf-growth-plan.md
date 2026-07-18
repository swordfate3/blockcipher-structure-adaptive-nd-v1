# 创新2 E55：PRESENT三轮query-cone sparse-ANF硬cap增长门计划

日期：2026-07-18

状态：计划冻结 / 待实现

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

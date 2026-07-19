# 创新2 E97：PRESENT五轮非平凡GF(2)消去标签提供器可行性计划

日期：2026-07-19

状态：已完成 / 暂缓 / 当前provider研究停止

## 1. 研究问题

E96确认当前没有新的神经结构具备训练资格。排名最高但仍缺前置条件的新路线是
`cancellation-aware Mask-Query Hypergraph`，其唯一解锁条件是：存在可执行、sound、
不依赖有限密钥投票，并能证明非平凡GF(2)消去正类的真实SPN标签提供器。

E97只回答：

```text
现有exact-ANF、3SDP、ATM、完整枚举和componentwise提供器中，
是否至少有一个能在冻结本地复杂度上限内，为真实PRESENT五轮未决查询
给出非平凡GF(2) cancellation positive的严格证书？
```

不训练网络，不修改标签语义，不把低轮或小型合成SPN证书外推到PRESENT高轮。

## 2. 同预算锚点

重放以下8个冻结gate及SHA256，不重新解释部分结果：

| 来源 | 作用 |
|---|---|
| E52 | PRESENT r5 all-key/all-offset严格标签池与P0/P1语义 |
| E53-A | 一、二轮exact-ANF与GF(2) trail parity校准 |
| E53-B | GLPK逐解blocking正确性与扩展性边界 |
| E54 | 136保留变量的full-superpoly最终边界 |
| E55 | 三轮query-cone sparse-ANF硬cap |
| E61-A | ATM两轮多坐标key-support消去与运行边界 |
| E64 | 小型SPN非平凡消去存在性及singleton捷径宽度 |
| E69 | PRESENT r4多bit标签componentwise domination |

唯一变化是把这些证据放入一个统一、预先冻结的提供器契约和12-query面板；不改变任何
历史cap、solver、round、标签或negative定义。

## 3. 十二个冻结查询

从E52的`1354`个unknown中确定性选择：

```text
structures = cube_000, cube_004, cube_008
families   = nibble, player_pair, same_nibble_pair, adjacent_nibble_pair
selection  = 每个structure/family中mask_index最小的unknown
total      = 3 x 4 = 12
rounds     = PRESENT-80 r5
active dimension = 8
```

这些查询当前既没有正证书，也没有负见证。选择unknown而不是negative，避免把已存在具体反例的
查询错误地交给另一个提供器“证明为正”。每个正类必须证明：指定多bit linear mask的完整
superpoly在所有80-bit主密钥和所有inactive plaintext offset上恒为零。

## 4. 提供器矩阵

冻结比较6类路线：

```text
P0 active-variable support absence
full-superpoly sparse exact ANF
GLPK per-solution 3SDP
ATM key-support cancellation
small-SPN exact enumeration
componentwise multibit composition
```

每类同时检查：目标语义匹配、证书sound、当前可执行、能表达非平凡消去、复杂度低于冻结cap、
真实PRESENT五轮正证书数量。有限密钥经验kernel不进入候选矩阵。

## 5. 冻结裁决门

```text
advance:
  至少一个提供器匹配PRESENT-80 r5 all-key/all-offset linear-mask语义；
  证书sound且不使用有限密钥投票；
  能表达非平凡GF(2)消去；
  在既有500万项/60秒/4GiB等冻结本地cap内可执行；
  12-query面板至少产生1个非平凡严格positive。
  -> 只扩充标签宽度并做确定性机制门，仍不直接训练网络。

stop:
  语义匹配的exact提供器仍越过cap，或可执行正类仍被singleton/componentwise解释，
  或只有小型合成SPN/独立轮密钥证书。
  -> 停止当前provider研究，冻结高轮输出预测为标签未就绪，转论文收束。

protocol_invalid:
  任一来源hash、run_id、status、decision或12-query选择漂移。
  -> 只修审计，不解释科学结果。
```

## 6. 执行与产物

```text
run_id = i2_present_r5_cancellation_provider_feasibility_20260719
device = local CPU
epochs = 0
seeds = none
training = no
remote = no
output = outputs/local_audits/i2_present_r5_cancellation_provider_feasibility_20260719/
```

产物必须包含：

```text
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
source_hashes.json
query_panel.csv
provider_portfolio.csv
curves.svg
visual_qa_passed.marker
```

完成后必须更新本节的正式结果与证据驱动的下一步，并刷新`outputs/00_RECENT_RESULTS.md/json`。

## 7. 正式结果

权威run：

```text
i2_present_r5_cancellation_provider_feasibility_20260719
```

8个冻结来源的`run_id/status/decision/SHA256`共33项检查全部通过；12-query面板也通过
结构数、mask family、多bit、unknown状态、无有限密钥投票和固定选择键等7项协议检查。

提供器资格结果：

| 提供器 | 目标语义 | sound | 可执行 | 识别相消 | 低于cap | 真实PRESENT r5 |
|---|---:|---:|---:|---:|---:|---:|
| P0支撑缺失 | 是 | 是 | 是 | 否 | 是 | 是 |
| full-superpoly sparse ANF | 是 | 是 | 否 | 是 | 否 | 是 |
| GLPK逐解3SDP | 否 | 是 | 否 | 是 | 否 | 否 |
| ATM key-support消去 | 否 | 是 | 否 | 是 | 否 | 否 |
| 小型SPN完整枚举 | 否 | 是 | 是 | 是 | 是 | 否 |
| 多bit逐分量组合 | 否 | 是 | 是 | 否 | 是 | 否 |

关键计数：

```text
providers audited                         = 6
target-semantics matching providers       = 2
cancellation-aware providers              = 4
fully eligible providers                  = 0
PRESENT r5 strict nontrivial positives    = 0
frozen queries resolved                   = 0 / 12
```

两个语义匹配的路线各缺一项不可替代条件：P0完整执行了`28800`个候选，但支撑饱和、正类为0且
不能识别trail相消；full-superpoly sparse ANF能够表达正确证书，但E55在三轮第4个query已达到
`5,000,001`项，越过冻结500万项硬cap。因此不存在同时满足语义、sound、可执行、相消感知、
真实PRESENT和复杂度上限的提供器。

冻结裁决：

```text
status   = hold
decision = innovation2_present_cancellation_provider_not_feasible_under_frozen_caps
training = no
remote   = no
```

## 8. 证据范围与推荐下一步

E97不是“PRESENT不存在非平凡积分相消”的数学否定；它只证明当前已实现和已审计的开放provider
在冻结语义与本地cap内不能生成可用于监督学习的真实PRESENT五轮非平凡正标签。小型SPN完整
枚举确实存在244条非平凡正类，但它们不能迁移成PRESENT证书；E69的PRESENT r4多bit正类又全部
被unit状态组合解释。

因此停止当前provider研究：不提高exact-ANF、GLPK、ATM的时间/内存/term cap，不用有限密钥
投票替代全密钥标签，不训练Mask-Query Hypergraph，也不启动PRESENT 7--9轮输出预测远程任务。

下一步转论文收束，正式贡献保留为PRESENT/GIFT分别训练的r3-only Profile Operator双密码双seed
拓扑归因；同时把E52--E55与E97整理为“高轮严格标签生成边界”。若未来出现新的sound provider
算法或可验证证书实现，必须另立新计划并先重过E97同等语义和复杂度门，而不是继续调当前网络。

可视化`curves.svg`已通过`visual-qa-redraw`：按最终SVG渲染为`2412 x 1234`像素，标题、说明、
提供器矩阵、query状态、资格漏斗和底部裁决均无字体重叠、裁切、缺字、颜色或类别歧义。

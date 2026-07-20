# 创新2 E102：PRESENT ATM逐候选可恢复搜索器fixture门

日期：2026-07-20

状态：预注册 / 待实现与执行

## 1. 研究问题

E101确认公开ATM的PRESENT九/十轮notebook有明确高轮调用，但原`search_integral_properties()`使用
阻塞`Pool.map`，只在完整split结束后写pickle/stats。单个公开R9 split历史耗时0.75至6.61小时，
不可在没有持续进度和参数匹配恢复时直接重跑。E102回答：

```text
能否在不改变候选生成、Avec语义和GF(2)关系构造的前提下，
把官方搜索改造成逐候选原子落盘、可验证恢复的route-owned runner？
```

E102只运行本地CPU小fixture，不构造PRESENT九/十轮模型，不执行缺失split搜索，不训练网络，
不启动远程任务。

## 2. 冻结来源

```text
source E101 gate = 056366dcbe692306c830284348b0a28fbbd4dc685c9dc671237c7bf1a5519933
E101 decision    = innovation2_present_high_round_resumable_runner_required
ATM repository   = https://github.com/michielverbauwhede/AlgebraicTransitionMatrices
ATM commit       = b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
search source    = Modelling/Search.py
search sha256    = 5d9a5c117d7f0940473c15dded0e3243dc06a6182fe584733e0badc8928f459d
```

官方算法的数学阶段保持不变：按weight生成候选`(u,v)`，调用`Avec(uv) -> (key_dependent, W)`，
将空`W`的key-independent候选加入weight-1 basis，其余key-independent支撑进入GF(2) nullspace。

## 3. 同预算对照与唯一变量

```text
anchor = 同一3-bit输入/3-bit输出确定性Avec fixture，无中断执行至完成
candidate = 同一fixture完成第1个新候选并原子落盘后受控中断，再用相同参数恢复至完成
one variable = 逐候选持久化与恢复边界
fixed = 候选顺序、Avec真值、输入/输出宽度、permutation模式、GF(2)结果规范化
workers = 1（fixture避免进程调度混入归因；runner接口保留多进程增量返回）
seeds/epochs = 不适用
device = local CPU
```

fixture至少产生3个候选，且必须同时覆盖：直接basis候选、进入`WUV`后参与nullspace的候选、
key-dependent丢弃候选。测试中的调用计数器必须能证明已完成候选在恢复后没有再次调用。

## 4. runner持久化契约

每个run目录必须包含：

```text
metadata.json                 冻结参数、来源hash、runner版本和parameter_hash
started.marker                启动后立即写入
progress.jsonl                run/layer/candidate/resume/complete事件
candidate_results/*.json      单个(u,v)的规范化Avec结果，临时文件后os.replace
result.json                   规范化最终关系集，临时文件后os.replace
complete.marker               result写完并重新校验后最后写入
```

恢复规则：

- 现有`metadata.json`的parameter hash不一致时必须拒绝，不能静默覆盖；
- 只有结构、候选和hash均通过的完整候选文件可以复用；半写临时文件不能算完成；
- 已完成候选必须跳过，未完成候选允许重算；
- 每层候选规范排序，最终关系规范排序，避免进程完成顺序影响证据；
- 多worker只能在父进程接收结果后落盘；worker不得竞争写同一结果文件；
- oracle内部Manager cache仍可在进程重启后重建，这是明确的性能降级，不改变已完成候选的复用语义。

## 5. 冻结裁决门

`fixture_passed`要求全部满足：

```text
E101 gate和官方Search.py hash重放通过；
anchor与恢复run的规范化result.json逐字节相等；
两者的数学关系集相等；
恢复run确实发生受控中断且随后记录resume事件；
中断前完成候选的调用次数在恢复后不增加；
parameter mismatch被拒绝；
损坏或不完整候选文件不被复用；
started/progress/per-candidate/result/complete均存在且hash通过；
complete.marker只在原子result完成之后出现；
至少一个basis、WUV和key-dependent路径被fixture覆盖。
```

通过后的唯一动作是另立E103“真实ATM低成本兼容性门”：使用冻结官方构模和真实`Avec`运行一个
小型、资源封顶的低轮/缩小候选检查，验证pickle、多进程Manager和bitarray环境兼容；仍不直接跑
R9 `(3,3,3)`。只有E103通过，才允许预注册R9缺失split生成。

`runner_protocol_invalid`：来源、结果相等、调用复用、hash、原子写或参数拒绝任一失败。只修runner。

`fixture_insufficient`：实现通过通用文件测试但未覆盖官方搜索的basis/WUV/key-dependent三条路径。
扩充fixture，不启动真实搜索。

## 6. 计划产物

```text
run_id = i2_present_atm_resumable_search_runner_fixture_20260720
output = outputs/local_audits/i2_present_atm_resumable_search_runner_fixture_20260720/
```

产物包括`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`、
`anchor/`、`resumed/`、`fixture_calls.csv`、`artifact_contract.csv`和`curves.svg`。可视化必须经
`visual-qa-redraw`像素检查。完成后刷新最近结果索引并写入正式结果与证据支持的下一步。

## 7. 正式结果

待执行。

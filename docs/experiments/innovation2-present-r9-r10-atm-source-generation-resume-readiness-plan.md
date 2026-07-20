# 创新2 E101：PRESENT九/十轮ATM新来源生成与恢复就绪审计

日期：2026-07-20

状态：预注册 / 待执行只读审计

## 1. 研究问题

E99确认九轮公开ATM正关系存在强坐标可学习性；E100停止当前PRESENT拓扑归因。继续高轮路线所缺的
不是第四个模型，而是独立或新生成关系集。冻结公开ATM仓库只有一个PRESENT语料目录：R9 notebook
声明9个split但公开8个，R10 notebook声明9个split但公开0个。E101回答：

```text
冻结公开源码是否提供可执行、参数明确、可持久化恢复的R9缺失split或R10关系生成路径，
使后续可以形成“新生成复核集”，而不冒充论文公开结果或盲跑不可恢复的长任务？
```

E101不执行ATM搜索、不训练网络、不生成negative、不启动远程任务。

## 2. 冻结来源

```text
source E100 gate = 6875aa3995c5b1e7b26ad549cb50b65f4b90bb866f50bf0f15b6ed739cc296ef
E100 decision    = innovation2_present_r9_coordinate_identity_anchor_remains_best
ATM repository   = https://github.com/michielverbauwhede/AlgebraicTransitionMatrices
ATM commit       = b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
notebook         = Ciphers/PRESENT/PRESENT.ipynb
search source    = Modelling/Search.py
oracle source    = Tools/AvecImplementations.py
environment      = requirements.txt + bitarrays/README.md
published data   = 8 R9 pickle/stats pairs
```

第三方pickle仍只通过项目受限loader读取；E101解析notebook JSON、Python AST、requirements文本和stats，
不执行notebook代码。

## 3. 结构化审计项

### 3.1 声明与公开覆盖

必须提取并验证：

```text
R9 splits = (1,7,1),(1,6,2),(1,5,3),(2,6,1),(2,5,2),(2,4,3),(3,5,1),(3,4,2),(3,3,3)
R10 splits= (1,8,1),(1,7,2),(1,6,3),(2,7,1),(2,6,2),(2,5,3),(3,6,1),(3,5,2),(3,4,3)
limit     = 2^10
threads   = 36
```

逐split报告notebook声明、pickle、stats、论文公开状态；缺失项保持`declared_without_public_result`，
不得写成零维、搜索失败或negative evidence。

### 3.2 历史成本锚点

解析8个R9 stats，报告时间和oracle calls的min/median/max以及split映射。当前只读观察锚点为：

```text
minimum time         = 2714.433652 s
maximum time         = 23786.157261 s
minimum oracle calls = 2666126
maximum oracle calls = 220841578547
```

这些是冻结公开机器/环境的历史记录，不直接预测当前本地或远程墙钟时间。

### 3.3 生成与恢复契约

检查notebook和`search_integral_properties()`是否具备：

- split/round/limit/thread写入元数据；
- 搜索开始marker和持续`progress.jsonl`；
- 每个候选`(u,v)`或每个weight层的原子结果缓存；
- 中断后参数匹配复用，跳过已完成候选；
- Manager内存cache的可选持久化或明确性能降级语义；
- 完成前不把半成品命名为正式pickle；
- 完成marker、stats、hash和安全结果转换；
- 不依赖阻塞整个候选列表后才返回的`Pool.map`作为唯一进度边界。

### 3.4 环境契约

检查`pybind11/ortools/python-sat/galois/numpy`是否固定版本，bitarray扩展是否有可验证构建命令、产物
hash或ABI元数据，以及项目当前环境能否只读发现这些模块。环境缺失不是科学失败，但会关闭直接生成。

## 4. 冻结裁决门

`generation_ready`要求同时满足：

```text
来源commit/notebook/source hash通过；
R9/R10 split、limit、threads和文件覆盖重放通过；
至少一个未公开高轮split有明确生成调用；
route-owned runner具备逐候选/逐层持久化、参数匹配resume、progress、原子完成和安全转换；
依赖版本与bitarray ABI/build可重放；
小型fixture已经验证中断恢复不会重复已完成候选且结果等于无中断搜索。
```

通过只允许另立生成执行计划：项目产物必须在受控本地路径或远程`G:\lxy`，先生成R9 `(3,3,3)`，
不得直接并行九个R10 split。新结果称`locally generated confirmation set`，不是published result或
independent publication。

`resumable_runner_required`：源码声明高轮split且数学调用明确，但只在末尾写结果、无可靠progress/
resume，或环境未固定。下一步只实现route-owned resumable runner与小型fixture；不启动长搜索。

`generation_resource_closed`：即使恢复契约可满足，来源审计证明目标在当前明确资源cap外，或生成结果
不能形成新的严格source-heldout正例。正式收束为E99九轮通用坐标关系识别证据。

`protocol_invalid`：冻结来源、notebook解析、stats或AST重放失败。只修审计。

## 5. 执行与产物

```text
run_id   = i2_present_r9_r10_atm_source_generation_resume_readiness_20260720
device   = local CPU
training = no
search   = no
remote   = no
output   = outputs/local_audits/i2_present_r9_r10_atm_source_generation_resume_readiness_20260720/
```

产物包括`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`、
`split_coverage.csv`、`historical_costs.csv`、`resume_contract.csv`、`environment_contract.csv`、
`source_hashes.json`、`curves.svg`和`visual_qa_passed.marker`。完成后刷新最近结果索引并给出唯一可执行
下一步。

## 6. 正式结果

待执行。

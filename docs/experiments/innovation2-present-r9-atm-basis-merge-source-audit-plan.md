# 创新2 E98-A：PRESENT九轮ATM基底合并与缺失split来源审计

日期：2026-07-20

状态：已完成 / 来源差异已解释 / E99仍关闭

## 1. 研究问题

E98在冻结公开ATM结果中重放出：

```text
unique serialized relations = 470
recomputed union GF(2) rank = 468
strict file-heldout positives = 24
```

E98-A只回答：

```text
470与468的差异究竟来自项目解析错误、单个公开基底内部依赖、
跨split基底合并依赖，还是作者公开merge实现没有真正应用GF(2)行化简？
notebook声明但没有公开结果的(3,3,3) split处于什么状态？
```

不训练网络，不生成负类，不运行缺失split的高成本oracle搜索，不修改第三方源码。

## 2. 冻结来源

```text
repository = https://github.com/michielverbauwhede/AlgebraicTransitionMatrices
commit     = b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
files      = 8 published R9-complex-oracle-*.pkl
source     = Ciphers/PRESENT/PRESENT.ipynb
             Ciphers/PRESENT/data_analysis.ipynb
             Tools/BasisTools.py
paper      = 2023 Beyne/Verbauwhede ATM paper, PRESENT section and Table 1
loader     = project RestrictedUnpickler
```

第三方pickle只能通过项目受限反序列化器读取。源码只做AST/文本审计；不执行notebook搜索。

## 3. 唯一变量与同预算锚点

同预算锚点是E98的相同8文件、470关系和468秩。唯一新增变量是**保留每个序列化关系的来源与
消元组合系数**，从而恢复导致秩亏2的精确GF(2)依赖；同时对比作者`merge_bases()`源码、
notebook声明split、实际文件、保存输出和论文Table 1。

必须分别计算：

```text
每个文件：serialized basis count vs independent GF(2) rank
八文件并集：deduplicated count vs independent GF(2) rank
依赖基：每条dependency包含哪些关系、坐标、来源文件
merge源码：row_reduce()返回值是否被赋回矩阵
split：声明9个、data_analysis使用多少、论文Table 1报告多少、实际结果多少
```

## 4. 冻结正确性控制

1. 小型GF(2) fixture包含两个独立向量和一个两者XOR，必须恢复秩2和一条三成员依赖；
2. 关系顺序使用canonical coordinate排序，依赖输出必须确定；
3. 每条恢复依赖逐坐标XOR必须为空；
4. 8个公开文件SHA256必须与E98一致；
5. 每个文件单独的重算秩必须等于notebook保存的`dimension`和pickle元素数；
6. `merge_bases`必须通过AST区分“调用row_reduce”与“使用row_reduce返回值”；
7. 不把未公开`(3,3,3)`结果推测成零维、失败或negative evidence。

## 5. 预注册裁决门

```text
audit_pass_source_mismatch_explained:
  冻结commit/文件/hash/受限解析全部通过；
  8个单文件基底各自满秩；
  八文件并集稳定重放470个关系、秩468；
  恢复恰好2条独立GF(2)依赖且逐坐标XOR为0；
  作者merge函数丢弃row_reduce返回值；
  data_analysis的470输出等于去重计数而不是重算秩；
  notebook声明9个split，但数据分析、论文表和结果目录只有8个，(3,3,3)无结果。
  -> 更正公开语料契约：当前可执行证据为468维span，不再把470称作重算维数；
     E99仍关闭，因为E98的24条严格留出正例宽度不变。

protocol_invalid:
  任一冻结hash、commit、安全解析、依赖重放或源码AST检查失败。
  -> 只修审计，不解释论文或高轮路线。

not_explained:
  470/468差异不能由可复算依赖和merge语义解释。
  -> 保持E98来源契约未决，禁止E99。
```

无论哪个分支，本实验都不直接开放神经训练。E99只有在后续独立来源/关系族审计同时满足至少
6个合格留出组、每组至少8条、总计至少64条严格未见正例后才可能开放。

## 6. 执行与产物

```text
run_id   = i2_present_r9_atm_basis_merge_source_audit_20260720
device   = local CPU
epochs   = 0
training = no
remote   = no
```

输出目录：

```text
outputs/local_audits/i2_present_r9_atm_basis_merge_source_audit_20260720/
```

产物必须包含：

```text
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
source_hashes.json
file_ranks.csv
dependencies.csv
dependency_members.csv
split_coverage.csv
curves.svg
visual_qa_passed.marker
```

完成后更新本文件正式结果、刷新最近结果索引，并给出证据驱动的下一步。

## 7. 正式结果

权威run：

```text
i2_present_r9_atm_basis_merge_source_audit_20260720
```

冻结commit、3个源码hash、8个pickle hash、论文文本hash、受限pickle解析和E98 gate hash全部
通过。8个公开split逐文件重算结果如下：

| split | 序列化元素 | GF(2)秩 | 秩亏 |
|---|---:|---:|---:|
| 1-7-1 | 455 | 455 | 0 |
| 1-6-2 | 420 | 420 | 0 |
| 1-5-3 | 338 | 338 | 0 |
| 2-6-1 | 425 | 425 | 0 |
| 2-5-2 | 401 | 401 | 0 |
| 2-4-3 | 331 | 331 | 0 |
| 3-5-1 | 338 | 338 | 0 |
| 3-4-2 | 331 | 331 | 0 |

因此单个搜索结果没有秩亏，差异只发生在跨split合并。3039个序列化引用去重后为470个不同
关系，它们在673个坐标上只张成468维空间。带来源消元恢复出恰好两条独立依赖：

```text
dependency_000 = 3个二项关系，逐坐标XOR为0
dependency_001 = 4个二项关系，逐坐标XOR为0
union nullity  = 2
```

公开`Tools/BasisTools.py::merge_bases()`中的`M.row_reduce()`以表达式语句调用，没有把返回
矩阵赋回`M`。在当前可执行`galois 0.4.11`中，`row_reduce()`返回新矩阵而不原地修改；因而
冻结`data_analysis.ipynb`保存的`470`与去重关系数完全相等，而不等于正确重算秩`468`。
仓库`requirements.txt`没有固定galois版本，所以本裁决限定为冻结公开源码在当前可执行环境的
可复算结论，不声称覆盖所有历史环境；但470维声明不能由冻结8文件的正确GF(2)消元重放。

split覆盖也确认：`PRESENT.ipynb`声明9个九轮split，保存输出只有8个；`data_analysis.ipynb`、
论文Table 1、结果pickle和stats同样只覆盖8个。`(3,3,3)`没有pickle、stats、保存输出或论文表
记录，必须保持`declared_without_public_result`，不能解释成零维、搜索失败或negative evidence。

冻结裁决：

```text
status   = pass
decision = innovation2_present_r9_atm_public_merge_count_not_rank
correct public corpus span rank = 468
E99      = closed
training = no
remote   = no
missing split generation = no
```

## 8. 证据范围与推荐下一步

E98-A解决了470/468来源契约阻塞，但没有增加任何正例。E98仍只有24条严格文件留出正例、1个
达到8条的合格组，不能训练九轮神经模型。也不生成`(3,3,3)`：它最多增加一个来源组，单独
无法把E98提升到至少6个合格组，而且现有最重公开split已经达到约2208亿oracle calls。

下一步执行E98-B：按密码结构对468维正关系做orbit-disjoint关系族覆盖审计，冻结循环位移、
nibble移位、P-layer共轭和坐标支撑不重合控制。只有该审计得到至少6个互斥关系族、每族至少8条、
总计至少64条严格正例，并证明train/heldout不存在relation、orbit或support-coordinate泄漏，才允许
设计E99本地正例-未标注排序；否则高轮神经路线保持关闭。

可视化`curves.svg`已通过`visual-qa-redraw`：最终SVG渲染为`2500 x 1344`像素，标题、说明、
三组坐标轴、标签、图例、缺失split标注和底部裁决均无重叠、裁切、缺字或含义歧义。

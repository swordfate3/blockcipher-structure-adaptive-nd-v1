# 创新2 E54：PRESENT五轮GF(2) transition tensor变量消元宽度审计计划

日期：2026-07-18

状态：完成 / hold / full-superpoly最终语义边界不可行

## 1. 研究问题

E53-A建立了完整PRESENT-80一、二轮exact ANF oracle；E53-B证明Sage/GLPK逐解blocking在低
复杂度S-box query上精确，但最重`v=15` query需要枚举1792个raw term choices，在冻结10秒内
无法完成。当前环境也没有PySAT、CryptoMiniSat、Z3、BDD或model counter。

E54不再逐解计数，而是把每个S-box、XOR、COPY、permutation和key-schedule关系表示为GF(2)
局部transition tensor，通过变量消元直接计算全局trail parity。第一门只审计因子图宽度与内存
上界，避免实现后才发现五轮收缩不可行。

E54不训练网络，不使用远程GPU，不执行五轮标签池。

## 2. 固定语义

```text
cipher             = PRESENT-80
target rounds      = 5
input variables    = 64 plaintext + 80 master-key exponent bits
positive semantics = full cube coefficient polynomial对key/inactive变量恒为0
local coefficients = exact GF(2) monomial-transition parity，不是trail existence
only change        = per-solution GLPK blocking -> exact factor/tensor contraction
```

## 3. Phase 0：保留边界维度硬门

内部变量可以消元，但目标full superpoly必须保留：

```text
80 master-key exponent variables
64 - 8 = 56 inactive plaintext exponent variables
total retained boundary variables = 136
```

先报告dense tensor的`2^136`表项与bit-packed/1-byte内存。如果最终边界本身超过26变量或4GiB，
直接停止dense tensor路线；内部min-fill宽度再小也不能改变最终输出张量的136维语义。

同预算语义控制：零offset key-coefficient保留80个变量但不覆盖all-offset；固定key+offset虽无自由
边界变量，但不覆盖all-key/all-offset。两者只作不可替代性控制，不能当可行候选。

同时读取E53-A exact oracle，报告r1/r2完整输出ANF总项数和fixture最大superpoly项数，判断稀疏
表示是否值得另立受限增长门；不得把两轮稀疏自动外推成五轮可行。

## 4. Phase A：因子图与宽度审计（仅Phase 0通过后）

逐轮构造PRESENT真实DAG，包括round-key XOR、16个4-bit S-box、P-layer、80-bit key schedule和
final whitening。每个局部factor必须先与E53-A S-box transition和一轮exact ANF fixture对拍。

至少比较三种确定性消元顺序：

```text
min-fill
min-degree
query-aware min-fill（目标output/cube变量最后消元）
```

每轮报告：变量数、factor数、primal graph边数、induced width、最大中间factor变量数、预计
`2^width`表项、按1 byte/bit-packed两种估计的峰值内存。随机顺序只作压力对照，不参与选择。

## 5. 推进门

```text
PRESENT bit order / key schedule / round convention pass
all-key/all-offset retained boundary variables <= 26
dense final-boundary peak <= 4 GiB
local S-box tensor == E53-A exact 256-entry transition parity
r1 tensor result == E53-A exact ANF fixture
r2 selected fixture == E53-A exact ANF fixture
best deterministic order max factor variables <= 26
estimated dense peak <= 4 GiB
order与宽度计算可复现
```

全部通过后，下一实验才实现实际tensor contraction，并先重放一、二轮全部fixture；再通过才允许
五轮固定`16x64`子集。若五轮宽度超过门槛，停止当前全变量tensor路线，不通过稀疏结果猜测、
更长timeout、远程GPU或经验标签绕过。

## 6. 控制与停止线

```text
control 1 = E53-A exact ANF oracle
control 2 = E53-B existence-only与GLPK blocking结果
control 3 = 故意错误P-layer，必须被r1/r2 fixture识别
control 4 = key schedule删除/独立轮密钥替代，必须被协议门识别
```

宽度门失败时，五轮严格标签路线冻结为“当前开放provider不可执行”；创新2保留四轮方法学结果
与确定性/神经负结果边界。不得因此重新开启四轮网络枚举。

## 7. 产物

```text
outputs/local_audits/i2_present_r5_transition_tensor_boundary_audit_20260718/

factor_manifest.json
elimination_orders.jsonl
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

## 8. 2026-07-18实际结果

权威run：

```text
i2_present_r5_transition_tensor_boundary_audit_20260718
```

E53-A来源、144个符号输入、全key/全inactive-offset语义和一、二轮fixture覆盖检查全部通过。
8-bit活动cube留下`64 - 8 = 56`个inactive plaintext变量；再加80个PRESENT-80 master-key
变量，目标full superpoly的最终边界必须保留136个变量：

```text
required retained variables = 136
dense entries               = 2^136
                            = 87112285931760246646623899502532662132736
bit-packed memory           = 1.0141204801825835e31 GiB
frozen gate                 = 26 variables / 4 GiB
```

四条边界路线中，只有`all key + all inactive offset`保持目标语义，但它不可执行；固定key、
zero-offset或固定赋值都改变全称标签语义，不能作为替代。由于最终输出边界本身已经失败，内部
factor graph和min-fill顺序按计划没有构造；更小的内部treewidth也无法缩小必须输出的136维
full-superpoly边界。

E53-A稀疏证据也不能直接外推五轮：完整输出ANF项数从一轮`1907`增长到二轮`4352830`
（`2282.5537x`），fixture最大superpoly从`13`增长到`53392`（`4107.0769x`），且没有可靠
五轮稀疏上界。因此正式裁决为：

```text
status   = hold
decision = innovation2_present_r5_transition_tensor_boundary_infeasible
training = no
remote   = no
```

最终`curves.svg`经`visual-qa-redraw`以`2261x1189`像素渲染检查。首次预览的右图增长率注释
与图表重叠，已把增长率移入两行子图标题；重绘后标题、轴标签、图例、数值、裁决和证据范围
均无重叠、裁切、缺字或歧义，并记录`visual_qa_passed.marker`。

## 9. 推荐下一步

关闭以下路线：136变量dense tensor、以zero-offset key coefficient替代all-offset、固定key或
固定offset替代全称标签、边界失败后继续内部min-fill、严格标签前训练神经网络。

只执行一次E55本地CPU硬cap门：沿冻结output query的反向依赖锥计算PRESENT三轮exact sparse
ANF，先重放E53-A一、二轮fixture，再运行8个unit-bit与4个multi-bit query；每query固定最多
`5,000,000`项、`60`秒和`4 GiB`。任何query越界即关闭当前开放全变量provider家族，不增加
cap、不转远程GPU、不重新枚举四轮网络。只有12个三轮query全部完成、严格正负标签均非零且
证书/反例复验通过，才以相同cap进入四轮query-cone门。

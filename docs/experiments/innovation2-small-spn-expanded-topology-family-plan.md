# 创新2 E37：扩展小状态SPN拓扑族与可泛化benchmark计划

日期：2026-07-18

状态：已完成 / pass / 扩展benchmark开放最小神经归因矩阵

## 1. 研究问题

E36已经证明原`4 S-box x 4 P-layer`精确标签包含宽P-layer条件信号，但E33--E35b的
5301条训练行只来自`3 x 3=9`张训练拓扑，其中独立训练P-layer仅3个。从3个随机
permutation外推第4个随机permutation严重欠定，不能继续用换网络结构解决。

E37只改变一个变量：把确定性P-layer族从4个扩展到16个。目标是判断在12个独立训练
P-layer下，是否能形成同时满足标签宽度、组外拓扑效应、反ID捷径和公平控制契约的
合成benchmark。本轮不训练神经网络。

## 2. 冻结数据协议

```text
state                     = 16-bit synthetic SPN
S-box family              = E32原4个，SBOX_SEED=32001
P-layer family            = E32同一生成序列的前16个，PLAYER_SEED=32002
rounds                    = 2,3,4,5
master keys               = 全部256个toy key
coordinate structures     = E32原14个
output masks              = E32原64个非零线性mask
label                     = 对全部256个key，完整输入结构上的输出线性mask XOR均为0
training                  = false
execution                 = local deterministic audit
```

`P0..P3`必须与E32逐字节一致。E37使用独立任务、缓存metadata和gate，不放宽
`SmallSpnAuditConfig`的E32冻结契约。

标签张量：

```text
[4 S-box, 16 P-layer, 4 rounds, 14 structures, 64 masks]
= 64 variants x 4 x 14 x 64
= 229376 boolean labels
```

精确输出XOR按key保存为磁盘缓存：

```text
cache_metadata.json
parity_words.npy
labels.npy
completed.npy
progress.jsonl
```

相同参数恢复时必须生成0个新block。

## 3. 冻结拓扑划分

```text
train        = S0..S2 x P0..P11  = 36 topologies / 12 independent P-layers
unseen-S     = S3 x P0..P11      = 12 topologies
unseen-P     = S0..S2 x P12..P15 = 12 topologies / 4 heldout P-layers
dual         = S3 x P12..P15      = 4 topologies
```

variant顺序固定为S-box优先、P-layer次序：`s0p0..s0p15,s1p0..s3p15`。

## 4. Train-only matched选择

对每个`round x structure x mask` base cell，只读取36个train topology标签。选择规则在
读取heldout标签前冻结：

```text
9 <= train positive topology count <= 27
```

即每个入选cell在train中至少有9个正类和9个负类。heldout标签只用于选择完成后的审计，
不得改变selected mask或阈值。

## 5. 指标与冻结门

### 5.1 协议门

必须同时满足：

```text
4个S-box、16个P-layer均为唯一双射
前4个P-layer与E32完全一致
64 variants与4 x 16顺序一致
全部256个key、4轮、14结构、64 mask覆盖
parity可重新计算labels
scalar/vectorized加密一致
cache恢复生成0个新block
selected mask只依赖36个train topology
```

### 5.2 宽度与类别门

```text
selected base cells                         >= 256
train positive / negative label rows        各 >= 3000
unseen-S positive / negative                各 >= 768
unseen-P positive / negative                各 >= 768
dual positive / negative                    各 >= 192
distinct 64-topology label patterns         >= 128
至少2个轮数各有                           >= 32 selected cells
```

### 5.3 拓扑可识别性门

在selected cell上计算：

```text
train P-sensitive any-S fraction            >= 0.75
train P-sensitive all-S fraction            >= 0.20
heldout-P novel any-S fraction              >= 0.50
dual heldout-P effect fraction              >= 0.40
dual P-effect target positive rows          >= 192
dual P-effect target negative rows          >= 192
train SxP interaction fraction              >= 0.50
full SxP interaction fraction               >= 0.70
```

`SxP interaction`仍使用GF(2)混合二阶差分：

```text
Y[s,p] xor Y[s,0] xor Y[0,p] xor Y[0,0]
```

### 5.4 ID边际与公平控制门

只用train标签拟合`global / mask-only / round-mask / structure-mask /
round-structure-mask`五个确定性边际，在heldout split上计算AUC：

```text
unseen-S strongest marginal AUC             <= 0.80
unseen-P strongest marginal AUC             <= 0.80
dual strongest marginal AUC                 <= 0.75
```

公平错误拓扑使用每个P-layer自身的固定destination-cell rotation；不得跨variant roll：

```text
corrupt(P)[source] = rotate_destination_cell(P[source], +1)
```

所有corrupted P必须仍为合法唯一permutation；`P12..P15`的corrupted版本不得等于任何
true或corrupted train P，保证heldout family身份不被换成train-seen topology。本轮只验证
控制构造契约；其预测归因在E37通过后的神经矩阵中同预算执行。

## 6. 裁决

全部门通过：

```text
decision = innovation2_small_spn_expanded_topology_benchmark_ready
```

只开放一个最小网络矩阵：同预算确定性ID基线、最强历史cell-equivariant GraphGPS、CETT、
fair-corrupted topology和label shuffle。先比较GraphGPS与CETT，不同时引入新模型家族。

标签宽度、拓扑效应或ID边际失败：

```text
decision = innovation2_small_spn_expanded_topology_benchmark_not_ready
```

停止当前随机P-layer合成benchmark，返回结构化P-layer族、目标定义或真实密码标签提供者；
不得机械增加网络容量、epoch或P-layer数量。

协议或公平控制失败：

```text
decision = innovation2_small_spn_expanded_topology_protocol_invalid
```

先修复生成、缓存、split或控制协议，不解释标签结果。

## 7. 计划产物

```text
run_id = i2_small_spn_expanded_topology_4s16p_256keys_20260718
output = outputs/local_audits/i2_small_spn_expanded_topology_4s16p_256keys_20260718/

cache_metadata.json
parity_words.npy
labels.npy
completed.npy
selected_mask.npy
selected_cells.csv
variants.csv
structures.csv
masks.csv
results.jsonl
gate.json
metadata.json
summary.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

若E37完成，文档必须补充实际指标、裁决、证据范围和可执行下一步。结果只属于16-bit合成
SPN benchmark，不是PRESENT/GIFT/SKINNY高轮结果、实际密码攻击或神经网络突破。

## 8. 实际结果

```text
run_id   = i2_small_spn_expanded_topology_4s16p_256keys_20260718
status   = pass
decision = innovation2_small_spn_expanded_topology_benchmark_ready
training = false
```

全部协议、宽度、拓扑与ID边际门通过。核心宽度：

```text
selected base cells             = 320       (gate >= 256)
distinct topology patterns      = 314       (gate >= 128)
per-round selected              = [0,154,165,1]
supported rounds >=32 cells     = 2         (gate >= 2)

train        positive/negative  = 6746 / 4774
unseen-S     positive/negative  = 2606 / 1234
unseen-P     positive/negative  = 2345 / 1495
dual         positive/negative  =  981 /  299
```

拓扑可识别性：

| 指标 | 实际 | 冻结门 |
|---|---:|---:|
| train P-sensitive any-S | 1.000000 | 0.75 |
| train P-sensitive all-S | 0.996875 | 0.20 |
| heldout-P novel any-S | 1.000000 | 0.50 |
| dual P-effect cells | 0.984375 | 0.40 |
| train S×P interaction | 0.981250 | 0.50 |
| full S×P interaction | 1.000000 | 0.70 |

dual P-effect行的目标正负为`961 / 299`，均超过冻结`192`门。train-only确定性ID边际：

```text
unseen-S strongest marginal AUC = 0.688198  (gate <= 0.80)
unseen-P strongest marginal AUC = 0.648753  (gate <= 0.80)
dual strongest marginal AUC     = 0.684393  (gate <= 0.75)
```

公平控制契约也全部通过：64个corrupted P仍为唯一双射，destination lane保持不变，
heldout corrupted P不等于任何true或corrupted train P。缓存恢复生成0个block，labels可从
parity words逐字重算，scalar/vectorized加密一致。

最终图已经按`visual-qa-redraw`渲染为像素并二次检查。第一次发现左图图例遮挡第一根柱的
数值标签，移动到绘图区上方后复查通过；无文字重叠、裁切、图例歧义或误导轴范围。

## 9. 解释与下一步

E37说明原E33--E35b的主要限制确实包含独立拓扑样本不足。扩到12个训练P-layer后，
train-only matched标签同时具备宽P敏感性、宽S×P交互和不被简单ID边际解释的组外集合。
这只是benchmark readiness，不代表GraphGPS或CETT已经学会积分性质。

下一步E38只运行一个本地同预算归因矩阵：

```text
research question:
  在12个独立训练P-layer上，显式edge-token是否比neighbor-gather GraphGPS更能外推？

staged models:
  Phase A: deterministic ID anchor + GraphGPS true P seed0/1
           + CETT true P seed0/1 + CETT label shuffle seed0
  Phase B: 仅对Phase A过门的最强候选运行fair-corrupted P seed0/1

frozen budget:
  hidden64 / 3 blocks or Transformer layers / 4 heads / 40 epochs
  AdamW lr1e-3 / weight decay1e-4 / batch128
  fit/validation只切train base cells，heldout不选模型
  local CPU；不使用远程GPU
```

Phase A主候选必须在两颗seed上超过同split最强ID边际，且dual均值超过ID至少`0.03`；
label-shuffle dual必须`<=0.60`。都不过门则不运行Phase B。Phase B要求true dual均值超过
自身公平错误拓扑至少`0.03`；CETT相对GraphGPS达到`+0.01`才保留显式edge-token增量。
否则停止当前两种结构，不加层、不加epoch、不上远程规模，转向有向边对关系或结构化
P-layer族，而不是继续按网络名称枚举。

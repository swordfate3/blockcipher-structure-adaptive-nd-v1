# 创新2：uKNIT-BC 拓扑双 cell 线性代数积分核轮数普查

日期：2026-07-30
状态：已完成 / hold（未超过 r4）

## 1. 研究问题与锚点

单活动 cell 的 `128 discovery + 128 validation` 普查表明：r3 的16个位置均为
64维全平衡；r4 只有 cells `0,2,3,4,5,6` 保留非零核；r5-r11 全部满秩。因此
单 cell 的 sampled-key 边界为 r4。

本实验只改变一个变量：把积分结构从一个活动4-bit cell扩大为两个活动4-bit
cell，每个 multiset 从16个明文增加到256个明文。问题是：

> 第一轮扩散拓扑内强耦合的双 cell 结构能否把 uKNIT 的稳定线性积分核从 r4
> 延长到 r5 或更高？

## 2. 拓扑候选与控制

根据 `UKNIT_LINEAR_TARGET_SOURCES[0]`，输入 cells 形成四个第一轮共享输出支撑的
四元组：

```text
G0 = {0,1,2,3}
G1 = {4,5,6,7}
G2 = {8,9,10,11}
G3 = {12,13,14,15}
```

每个组内部6个无序 pair 全部扫描，共24个 topology-coherent pairs。另设4个跨组
边界控制 `(3,4),(7,8),(11,12),(15,0)`。控制使用相同密钥、上下文、轮数、明文数
和输出特征；唯一差别是两活动 cell 是否属于同一首轮拓扑组。

## 3. 固定协议

```text
cipher = uKNIT-BC, 64-bit block, 128-bit key
round semantics = prefix AddRoundKey -> S-box -> GF(2) linear layer
calibration = r1
target rounds = r4,r5,r6,r7,r8
structures = 24 within-group pairs + 4 cross-group controls
plaintexts per multiset = 16^2 = 256
feature = raw 64-bit ciphertext XOR parity
discovery = 128 fresh unique keys + fresh inactive contexts
validation = 128 disjoint fresh keys + fresh inactive contexts
seed = 0; training = none; device = local CPU
```

本实验使用与单 cell 普查不同的确定性随机流，避免重复使用其发现或验证数据。

## 4. 裁决门

协议门：uKNIT公开向量、key唯一性与拆分、cache shape/dtype、全部 GF(2) basis、
r1 `rank/nullity=0/64` 和所有目标随机输出控制满秩必须同时通过。

一个 pair/round 的稳定核定义与单 cell 完全相同：joint nullity非零、discovery存在
候选、候选在128个独立validation试验中存活、joint basis在两半均成立、随机控制
joint nullity为零。

推进门：

```text
pass/extension:
  最高稳定轮 >= r5
  next = 只确认最高轮最强2-4个pair，1000+1000 trials，seed0/seed1

hold/no extension:
  只有r4或没有稳定核
  next = 不机械加密钥；比较三活动cell与256维cell-VDS的成本/潜力后单选一路

fail:
  任一协议门失败
  next = 修复枚举、轮边界、缓存所有权或GF(2)实现
```

即使通过，这仍是 sampled-key key-independent integral distinguisher，不是完整
密钥恢复结论。只有确认后才能另行设计 partial-sum/末轮子密钥猜测复杂度。

## 5. 产物

```text
run_id = i2_uknit_topology_pair_linear_integral_round_census_256trials_seed0_20260730
outputs/local_audits/<run_id>/
  results.jsonl
  round_summary.csv
  kernel_basis.csv
  keys.npy
  base_plaintexts.npy
  parity_rows.npy
  random_control_rows.npy
  metadata.json
  progress.jsonl
  gate.json
  curves.svg
```

完成后写回实测结果和推荐下一步，刷新最近结果索引，并对 SVG 执行
`visual-qa-redraw` 像素检查。

## 6. 2026-07-30 实测结果

干净复跑完成全部256个试验，`progress.jsonl` 包含128个严格递增的2-trial chunk，
从 `2` 连续到 `256`，最后事件为 `run_done`。全部协议门通过：公开12轮向量、
28个结构的所有权、256把唯一随机密钥、发现/验证互斥、cache shape/dtype、r1
校准、GF(2) basis校验和同预算随机输出控制均有效。

| 轮数 | 组内 pair 稳定非零 | 跨组控制稳定非零 | 最大 nullity | 随机输出秩亏 |
|---:|---:|---:|---:|---:|
| r4 | 24/24 | 2/4 | 46 | 0/28 |
| r5 | 0/24 | 0/4 | 0 | 0/28 |
| r6 | 0/24 | 0/4 | 0 | 0/28 |
| r7 | 0/24 | 0/4 | 0 | 0/28 |
| r8 | 0/24 | 0/4 | 0 | 0/28 |

r4 最大的两个平衡空间来自 pairs `5+7` 与 `6+7`，joint nullity 均为46；组内
24个 pair 全部稳定，跨组控制仅 `3+4` 与 `15+0` 稳定。但 r5 开始所有28个结构
全部 `rank/nullity=64/0`，所以双 cell 并未把单 cell 的支持轮数推进到5轮。

裁决：

```text
status = hold
decision = innovation2_uknit_pair_linear_kernel_no_round_extension
highest_supported_round = r4
topology_coherent_pair_extended_beyond_r4 = false
```

该结果排除了“只把积分输入从16个明文扩大到256个明文，就能自然增加一轮”的
路线，但不等于 uKNIT 在5轮后不存在其他积分性质。推荐下一步优先采用论文明确
展示过增量价值的256维 cell-VDS 输出表示：冻结 `5+7`,`6+7`,`0+1`,`15+0`
四个结构，只把64维线性输出改为每 cell 的非线性单项式特征。必须剔除16个恒定
单项式，按240个有效维度计算，并使用至少512 discovery + 512 validation，使
随机控制不会因行数少于特征维数而必然产生非零核。若 r5 仍满秩，再比较三活动
cell 的4096明文成本，不直接增加网络或远程预算。

权威产物：

```text
outputs/local_audits/
  i2_uknit_topology_pair_linear_integral_round_census_256trials_seed0_20260730/
```

第一次像素检查发现图底部把26个 r4 pair 连成一行，虽未裁切但过密难读。renderer
改为汇总“组内24/24、跨组2/4、最大维数46（5+7,6+7）”，完整列表保留在
`results.jsonl` 与 `gate.json`。最终1896x1253像素图通过
`visual-qa-redraw`：中文字体、标题、28行热图、分组边界、柱图、图例、坐标和
结论均无重叠、裁切、缺字或误导色阶。

# 创新2 E5：PRESENT r5 候选筛选的 4096 把全新密钥富集验证

**日期：** 2026-07-16
**状态：** 已完成；预注册门通过，但 E6 证明主要收益受输出位置先验解释
**执行位置：** 本地 CPU
**远程 GPU：** 禁止

## 研究问题

E4 的结构 MLP 在 128 个训练中未出现的 geometry 上选出了经验平衡率更高
的 top-16。E5 检验这个优势能否在一批完全独立、规模扩大 16 倍的密钥上
保持，并且优于同预算线性、确定性结构启发式和固定随机选择。

```text
research question = E4 MLP top-16 是否在 4096 把全新密钥上仍显著富集高平衡率结构？
same-budget anchor = E4 linear_same_input top-16
required controls = label-free P-layer reachability top-16 + fixed-seed random top-16
one changed variable = candidate selector
candidate budget = 每种方法 16 个结构
training = 0；只读 E4 ranking.csv，不重训、不改分数
```

## 文献与声明边界

EUROCRYPT 2026 已有神经积分候选生成与 split-search 验证；kernel 工作也已
使用多密钥 parity matrix 和独立密钥验证。E5 不主张首创这两种范式。

当前环境没有可验证的 SAT/MILP/Sage 工具，保守的活动单项式上界又在
PRESENT r5 上全部返回未知。因此 E5 是 fresh-key 统计验证，不是
Split-and-Cancel、division-property 完整搜索或所有密钥证明。

## 冻结输入

```text
ranking root = outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_ranking_seed0
source root  = outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_seed0
source rows  = 128 geometry-disjoint test structures
rounds       = PRESENT-80 r5
fresh keys   = 4096
key seed     = 2026071601
random selector seed = 20260716
```

fresh keys 必须与 E4 train、validation、calibration、test、stability 的全部
密钥互斥。所有选择器使用完全相同的 4096 把 fresh keys。

## 四种选择器

1. `structure_mlp`：按 E4 candidate predicted q1 rate 从低到高取 16 个。
2. `linear_same_input`：按 E4 anchor predicted q1 rate 从低到高取 16 个。
3. `p_layer_reachability`：不读取标签或模型分数。从活动 nibble 的 4 个 bit
   出发，每轮按“同 S-box nibble 全连接，再过 PRESENT P-layer”传播，记录
   output mask 各 bit 首次可达轮数；按 `最低首次可达轮数`、`平均首次可达
   轮数` 降序，再按 mask Hamming weight 升序和 structure id 升序取 16 个。
4. `fixed_random`：`numpy.default_rng(20260716)` 无放回选择 16 个，选择后
   按 structure id 排序。

允许不同选择器命中同一结构；产物必须报告交集，不能把重叠结构重复计算
成独立证据。

## 观测与统计量

对每个被选结构和每把 fresh key，完整枚举 16 个活动 nibble 取值，计算：

```text
q(K,S,u) = XOR_{x in S} <u, E_K^5(x)>
balance  = 1 - q
```

每个结构记录 q1 count、balance rate、Wilson 95% 区间。若 4096 把密钥
零次观察到 q1，则额外记录 95% 单侧失败率上界：

```text
upper_q1 = 1 - 0.05 ** (1 / 4096) ~= 0.000731
```

该上界依赖均匀独立随机密钥假设，只是统计界，不是“对所有密钥平衡”。

## Readiness

先运行 `top_k=4, fresh_keys=64`：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate-innovation2-integral-fresh-keys \
  --run-id i2_present_r5_integral_fresh_key_enrichment_readiness_seed0 \
  --ranking-root outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_ranking_seed0 \
  --source-root outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_seed0 \
  --top-k 4 --fresh-keys 64 --key-seed 2026071601 --random-selector-seed 20260716 \
  --gate-mode fresh-key-smoke \
  --output-root outputs/local_smoke/i2_present_r5_integral_fresh_key_enrichment_readiness_seed0
```

readiness 必须验证：source gate 为 E4 geometry pass、128 行/geometry 唯一、
四选择器各 4 个、fresh keys 与所有历史 key 互斥、向量化 parity 与标量
`integral_mask_parity` 在多结构/多密钥上逐项一致、CSV/JSONL/SVG/gate/
progress 产物齐全。

## 完整 E5

readiness 通过后运行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate-innovation2-integral-fresh-keys \
  --run-id i2_present_r5_integral_fresh_key_enrichment_4096_seed0 \
  --ranking-root outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_ranking_seed0 \
  --source-root outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_seed0 \
  --top-k 16 --fresh-keys 4096 --key-seed 2026071601 --random-selector-seed 20260716 \
  --gate-mode fresh-key-enrichment \
  --output-root outputs/local_diagnostic/i2_present_r5_integral_fresh_key_enrichment_4096_seed0
```

## 冻结门槛

```text
MLP mean balance - linear mean balance        >= +0.03
MLP mean balance - reachability mean balance  >= +0.03
MLP mean balance - random mean balance        >= +0.10
MLP minimum structure balance rate            >= 0.75
MLP zero-observed-failure structures           >= 1
```

全部通过：`innovation2_integral_fresh_key_enrichment_passed`，冻结创新2实验
证据并进入毕业论文写作，不启动远程扩样本。

前三个富集门通过但没有零失败结构：
`innovation2_integral_fresh_key_ranking_only`，只保留排序贡献。

任一主要对照优势失败：
`innovation2_integral_fresh_key_enrichment_not_confirmed`，E4 仍保留为
256-key geometry 诊断，但不得把 4096-key 稳健性写入论文。

## 禁止路线

- 不重训 E4，不重新选择 seed，不根据 4096-key 结果修改选择器。
- 不把零失败称为确定性积分、精确认证或所有密钥证明。
- 不安装未经验证的求解器来临时替换统计实验。
- 不启动远程 GPU，不机械增加结构、epochs 或模型规模。

## 完成结果

readiness：

```text
run_id = i2_present_r5_integral_fresh_key_enrichment_readiness_seed0
status = pass
decision = innovation2_integral_fresh_key_implementation_ready
readiness checks = 11/11 true
```

完整 E5：

```text
run_id = i2_present_r5_integral_fresh_key_enrichment_4096_seed0
status = pass
decision = innovation2_integral_fresh_key_enrichment_passed

selector                    mean balance    minimum    zero failures
structure MLP               0.956604004     0.841797   8
linear same input           0.894393921     0.715576   2
P-layer reachability        0.848632812     0.542725   1
fixed random                0.802154541     0.501465   2

MLP - linear                +0.062210083
MLP - reachability          +0.107971191
MLP - random                +0.154449463
```

五个冻结门全部通过。8 个 MLP 零观察失衡结构在 4096 把密钥下均有
`q1=0`，每个结构的 95% 单侧 q1 上界为 `0.000731113`；这仍不是所有密钥
证明。

完成后检查发现，8 个零失败候选全部属于 `output_nibble=0`，触发 E6
post-hoc 位置先验审计。E6 表明训练位置先验和位置匹配线性基线几乎解释
全部收益。因此 E5 的正确最终解释是“MLP 能富集高平衡率候选，但该全局
富集不能归因为独有神经结构交互”，不能单独作为神经方法优势结论。

产物：

```text
outputs/local_smoke/i2_present_r5_integral_fresh_key_enrichment_readiness_seed0/
outputs/local_diagnostic/i2_present_r5_integral_fresh_key_enrichment_4096_seed0/
```

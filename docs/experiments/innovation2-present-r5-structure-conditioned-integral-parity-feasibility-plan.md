# Innovation 2 PRESENT r5 结构条件积分 Parity 可行性实验

**日期：** 2026-07-15

**状态：** E0-E3 本地诊断完成 / 双 seed 排序效用确认，待几何组合留出

**Run ID：** `i2_present_r5_structure_integral_parity_feasibility_seed0`

## 研究问题

在固定 PRESENT-80 r5、网络看不到密钥和最终密文的条件下，活动位置、
输出位置、输出掩码与固定明文上下文的非线性交互，能否预测未见结构、
未见密钥上的积分掩码 parity 概率，并优于相同输入的线性模型？

## 假设

PRESENT 的 S-box/P-layer 扩散使 r5 处于全平衡向随机 parity 过渡的临界
区。两层 MLP 可以学习位置与掩码的非线性交互；单层线性模型不能充分
表达这些交互，标签打乱 MLP 应回到 AUC 约 `0.5`。

## 冻结数据协议

| 字段 | Smoke | Local diagnostic |
|---|---:|---:|
| cipher | PRESENT-80 | PRESENT-80 |
| rounds | 5 | 5 |
| active values / structure | 16 | 16 |
| train structures | 64 | 512 |
| validation structures | 32 | 128 |
| test structures | 32 | 128 |
| train keys / structure | 4 | 16 |
| validation keys / structure | 8 | 32 |
| test keys / structure | 8 | 32 |
| seed | 0 | 0 |

每个 split 独立生成结构和 80-bit 密钥，三组结构签名不得重复，三组
密钥集合不得相交。一个结构在该 split 的每把密钥下产生一行 `q`
观测；密钥不进入特征。

标签：

```text
q(K,S,u) = XOR_{v=0}^{15} <u, E_K^5(base with active_nibble=v)>
```

特征固定为 111 维：活动位置 one-hot、输出位置 one-hot、非零 4-bit
输出掩码 one-hot、活动 nibble 清零后的 64-bit 固定上下文。

## 三模型矩阵

| 角色 | 模型 | 输入 | 唯一变化 |
|---|---|---|---|
| anchor | `linear_same_input` | 111 维结构向量 | 单层线性 logit |
| candidate | `structure_mlp` | 同一 111 维向量 | 两层 64-unit ReLU MLP |
| control | `structure_mlp_shuffled_labels` | 同一向量 | 只打乱训练标签 |

训练超参数对三行冻结一致：

```text
epochs              = smoke 3 / diagnostic 20
batch_size           = 256
optimizer            = Adam
learning_rate        = 0.001
weight_decay         = 0.0001
loss                 = BCEWithLogitsLoss
checkpoint_metric    = val_auc
restore_best         = true
device               = cpu
```

候选与 shuffled control 参数量完全一致。validation 只选择 checkpoint；
最终裁决使用独立 test split。

## 指标

主指标：

```text
test observation AUC
candidate AUC - linear AUC
candidate AUC - shuffled-control AUC
```

结构级支持指标：对每个 test 结构聚合 32 把密钥的真实 `q=1` 比例，
比较模型概率的 MAE/Brier，并报告全局训练先验的常数预测误差。

普通 accuracy 只作支持指标，因为 r5 的 `q` 类别不平衡。

## Readiness Gate

Smoke 只检查实现，不做性能结论：

- 三个 split 都同时包含 `q=0` 和 `q=1`；
- 结构签名和密钥集合跨 split 互斥；
- 三个模型完成训练并产生有限指标；
- shuffled control 只打乱 train 标签；
- JSONL、CSV、SVG、gate 和进度产物可生成。

满足后自动进入冻结的 local diagnostic；不满足则修协议，不扩样。

## Diagnostic Advance Gate

候选只有同时满足以下条件才保留：

```text
candidate test AUC                  >= 0.60
candidate - linear test AUC         >= +0.02
candidate - shuffled test AUC       >= +0.05
candidate structure-rate MAE        <= linear structure-rate MAE - 0.02
```

裁决：

| 结果 | 决定 |
|---|---|
| 全部通过 | 保留路线；下一步做 seed1、几何组合留出和经典积分基线 |
| AUC 有信号但线性差值不足 | 结构存在简单可解释信号；先设计确定性/线性基线，不扩 MLP |
| 候选不胜 shuffled 或 AUC < 0.60 | 停止当前表示；审计目标噪声和结构编码 |
| shuffled 明显偏离 0.5 | 视为拆分/泄漏风险，结果无效 |

## 禁止项

本轮不得：

```text
把单密钥 q=0 写成确定性积分
把输出密文或已计算 parity 放入输入
混入 r1-r4/r6-r7 制造轮数捷径
根据 test 指标调超参数再重报同一实验
在本地门控前启动远程 GPU
把本地 512-structure 诊断称为正式训练或论文规模
```

## 计划命令

Smoke：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run-innovation2-integral-property \
  --run-id i2_present_r5_structure_integral_parity_smoke_seed0 \
  --train-structures 64 --validation-structures 32 --test-structures 32 \
  --train-keys 4 --validation-keys 8 --test-keys 8 \
  --epochs 3 --seed 0 --device cpu --gate-mode smoke \
  --output-root outputs/local_smoke/i2_present_r5_structure_integral_parity_smoke_seed0
```

Local diagnostic：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run-innovation2-integral-property \
  --run-id i2_present_r5_structure_integral_parity_feasibility_seed0 \
  --train-structures 512 --validation-structures 128 --test-structures 128 \
  --train-keys 16 --validation-keys 32 --test-keys 32 \
  --epochs 20 --seed 0 --device cpu --gate-mode diagnostic \
  --output-root outputs/local_diagnostic/i2_present_r5_structure_integral_parity_feasibility_seed0
```

## 预期产物

```text
results.jsonl
progress.jsonl
dataset_summary.json
structure_rates.csv
gate.json
curves.svg
history.csv
```

## 结果记录

### Smoke readiness

```text
run_id = i2_present_r5_structure_integral_parity_smoke_seed0
train/validation/test structures = 64 / 32 / 32
train/validation/test keys       = 4 / 8 / 8
epochs                           = 3
status                           = pass
decision                         = innovation2_integral_property_implementation_ready
```

所有 split 同时包含 `q=0/1`，结构和密钥跨 split 互斥，三模型及
JSONL/CSV/SVG/gate/progress 产物完整。Smoke 指标只用于实现检查，不作
性能结论。

### Local diagnostic

按本计划冻结命令运行：

```text
run_id = i2_present_r5_structure_integral_parity_feasibility_seed0
train/validation/test structures = 512 / 128 / 128
train/validation/test keys       = 16 / 32 / 32
train/validation/test rows       = 8192 / 4096 / 4096
epochs                           = 20
test q=1 rate                    = 0.291259765625
```

结果：

| 角色 | 模型 | 参数量 | Test AUC | Test accuracy | 结构概率 MAE | 结构概率相关 |
|---|---|---:|---:|---:|---:|---:|
| anchor | `linear_same_input` | 112 | `0.620888903` | `0.709228516` | `0.122736339` | `0.516846335` |
| candidate | `structure_mlp` | 11,393 | `0.655288384` | `0.706787109` | `0.110356920` | `0.645444436` |
| control | `structure_mlp_shuffled_labels` | 11,393 | `0.548262066` | `0.708740234` | `0.146503033` | `0.184441717` |

预声明差值：

```text
candidate - linear AUC       = +0.034399481  PASS (>= +0.02)
candidate - shuffled AUC     = +0.107026318  PASS (>= +0.05)
candidate test AUC           =  0.655288384  PASS (>= 0.60)
linear MAE - candidate MAE   = +0.012379419  MISS (< +0.02)
shuffled AUC near chance     =  0.548262066  PASS (within 0.45--0.55)
```

门控：

```text
status   = hold
decision = innovation2_integral_property_redesign_before_scale
```

### 解释

候选在未见结构、未见密钥上明显胜过相同输入线性模型和标签打乱控制，
因此“结构字段含有可泛化排序信息”得到一个正向本地诊断。候选把结构
概率相关从 `0.5168` 提升到 `0.6454`，也优于训练全局先验 MAE
`0.1502`。

但本方法声称预测的是跨密钥平衡概率，不只是把 `q=1` 样本排在前面。
候选结构级 MAE 只比线性改善 `0.0124`，未达到冻结的 `0.02` 门槛；
因此不得升级为稳定概率预测、确定性积分发现或远程扩展证据。

### 下一步 E1：独立校准/标签噪声审判

研究问题：当前 MAE 缺口来自 MLP 概率未校准，还是 32 把 test 密钥
给出的结构率本身噪声过高？

执行协议：

```text
same-budget anchor = 本 E0 的 linear 与 uncalibrated MLP
one variable       = 增加独立 calibration split，并只拟合 affine logit calibration
train/val/test     = 保持 512/128/128 结构和 16/32/32 密钥
calibration split = 新增 128 个结构、32 把独立密钥
stability audit    = 对 E0 test 的同 128 个结构增加到 256 把新密钥，仅重估标签率
models             = linear / MLP / shuffled 不变，不做新架构搜索
```

E1 readiness：保存 validation/calibration/test observation logits 与结构率；
checkpoint 只能由 validation 选择，校准器只能由 calibration 拟合，test
只评价一次。

E1 advance gate：

```text
calibrated MLP 256-key structure-rate MAE <= 0.09
calibrated MLP MAE improvement vs calibrated linear >= +0.015
MLP AUC advantage vs linear remains >= +0.02
32-key vs 256-key observed-rate MAE <= 0.05
```

若只校准即可过门，进入 seed1/几何组合留出；若 256-key 标签稳定但 MLP
仍不过门，增加 PRESENT P-layer reachability 特征；若标签本身不稳定，
改为区间/排序目标。禁止直接增加训练结构、epochs 或远程 GPU。

### 产物

```text
outputs/local_smoke/i2_present_r5_structure_integral_parity_smoke_seed0/
outputs/local_diagnostic/i2_present_r5_structure_integral_parity_feasibility_seed0/
```

本结果的声明范围仅为 PRESENT r5 本地结构条件、密钥省略的积分 parity
概率诊断；不是确定性积分证明，不是正式输出预测结果，也不是论文规模。

## E1 冻结执行矩阵（2026-07-16）

E1 使用独立 Run ID，不覆盖 E0 产物：

```text
readiness run_id  = i2_present_r5_integral_parity_calibration_smoke_seed0
diagnostic run_id = i2_present_r5_integral_parity_calibration_seed0
```

正式本地诊断冻结为：

| Split | 结构数 | 每结构密钥数 | 用途 |
|---|---:|---:|---|
| train | 512 | 16 | 与 E0 相同的模型拟合 |
| validation | 128 | 32 | 只选 checkpoint |
| calibration | 128 | 32 | 只拟合单调 affine logit 的斜率和偏置 |
| test | 128 | 32 | 与 E0 相同的独立测试观测与 AUC |
| stability | 与 test 相同 128 个 | 256 把全新密钥 | 只重估结构真实 `q=1` 概率 |

三模型仍为 `linear_same_input`、`structure_mlp` 和
`structure_mlp_shuffled_labels`；训练规模、20 epochs、batch 256、Adam、
学习率和权重衰减全部与 E0 相同。校准映射为：

```text
z_cal = a * logit(p_raw) + b, a > 0
p_cal = sigmoid(z_cal)
```

`a>0` 保证校准不改变排序，因此不能靠后处理制造 AUC 提升。validation、
calibration、test 和 stability 的 observation logits/概率写入独立 CSV；
32-key 与 256-key 结构率及各模型 raw/calibrated 预测写入结构率 CSV。

Readiness smoke 只检查 split 所有权、密钥互斥、test/stability 结构一致、
有限校准参数和产物完整，不作性能结论。通过后直接运行本地 E1：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run-innovation2-integral-property \
  --run-id i2_present_r5_integral_parity_calibration_seed0 \
  --train-structures 512 --validation-structures 128 --test-structures 128 \
  --train-keys 16 --validation-keys 32 --test-keys 32 \
  --calibration-structures 128 --calibration-keys 32 \
  --stability-test-keys 256 \
  --epochs 20 --seed 0 --device cpu --gate-mode calibration \
  --output-root outputs/local_diagnostic/i2_present_r5_integral_parity_calibration_seed0
```

E1 advance gate 保持此前预注册数值，并明确全部在 256-key 稳定率上评价：

```text
calibrated MLP 256-key structure-rate MAE       <= 0.09
calibrated linear MAE - calibrated MLP MAE      >= +0.015
raw MLP test AUC - raw linear test AUC          >= +0.02
32-key vs 256-key observed structure-rate MAE    <= 0.05
```

全过才进入 seed1 与几何组合留出。若标签稳定但校准后 MLP 仍不过门，
下一步只增加 PRESENT P-layer reachability 特征并保持本矩阵；若标签不
稳定，则停止点概率回归，改为区间或排序目标。不得直接增加训练结构、
epochs、seed 或远程 GPU。

## E1 完成结果

Readiness smoke 与冻结的本地 E1 均已完成：

```text
readiness = i2_present_r5_integral_parity_calibration_smoke_seed0
diagnostic = i2_present_r5_integral_parity_calibration_seed0
train / validation / calibration / test rows = 8192 / 4096 / 4096 / 4096
stability rows = 128 structures x 256 fresh keys = 32768
epochs / seed / device = 20 / 0 / cpu
```

数据所有权检查全部通过：train、validation、calibration、test 的结构
互斥，五组密钥互斥，stability 精确复用 test 的 128 个结构但使用 256
把全新密钥。三模型的校准斜率均为正且指标有限。

| 模型 | Test AUC | 校准前 256-key MAE | 校准后 256-key MAE | 校准 slope |
|---|---:|---:|---:|---:|
| linear | `0.620888903` | `0.108694189` | `0.111128089` | `1.242609116` |
| MLP | `0.655288384` | `0.102821873` | `0.095799242` | `0.662279627` |
| shuffled MLP | `0.548262066` | `0.136237099` | `0.139737508` | `0.447884718` |

冻结门控：

```text
MLP calibrated 256-key MAE <= 0.09             MISS  0.095799242
linear MAE - MLP MAE >= +0.015                 PASS +0.015328847
MLP AUC - linear AUC >= +0.02                  PASS +0.034399481
32-key vs 256-key observed-rate MAE <= 0.05    MISS  0.059875488
shuffled AUC within 0.45--0.55                 PASS  0.548262066

status   = hold
decision = innovation2_integral_rate_target_unstable
```

校准确实将 MLP 的 calibration log loss 从 `0.513216589` 降到
`0.506776186`，并将其 256-key MAE 从 `0.102821873` 降到
`0.095799242`；同时它仍明显优于线性和打乱标签控制。因此 E1 没有否定
结构信号，而是否定了“用 32-key 观测率作为精确点目标后直接扩规模”这
条路径。不得把本结果写成确定性积分发现或正式输出预测结果。

完成产物：

```text
outputs/local_smoke/i2_present_r5_integral_parity_calibration_smoke_seed0/
outputs/local_diagnostic/i2_present_r5_integral_parity_calibration_seed0/
```

其中本地诊断保存 3 行模型 JSONL、128 行结构率 CSV、135168 行
validation/calibration/test/stability observation 预测 CSV、训练曲线、历史、
进度、数据摘要和门控。

## 推荐下一步 E2：不确定性感知排序效用审判

E1 的正向证据是排序 AUC 和相对线性 MAE 优势，负向证据是点概率绝对
误差与 32-key 标签稳定性。因此 E2 不重训、不增加数据，先复用 E1 的
独立 256-key stability 标签判断网络能否用于“优先筛出更可能平衡的积分
结构”。

```text
research question = MLP 能否比线性/打乱控制更好地排序高平衡概率结构？
same-budget anchor = calibrated linear_same_input
required control = calibrated shuffled-label MLP + global test prior
one changed variable = 裁决目标从点概率 MAE 改为稳定率排序/候选筛选效用
data = 复用 E1 的 128 个 test 结构与 256-key stability 率
training = 无；不得重新选 checkpoint 或调超参数
execution = 本地只读后处理
```

预先冻结 E2 主门：

```text
MLP Spearman(stable q1 rate) - linear Spearman        >= +0.05
MLP lowest-q1 top16 observed balance - global mean    >= +0.05
MLP lowest-q1 top16 observed balance - linear top16   >= +0.03
shuffled top16 advantage vs global                     <= +0.02
```

四门全过才设计新的 seed1/几何组合留出确认；若只相关性过门但 top16 不
过门，路线只保留为解释性统计结果；若连相关性优势也不过门，停止当前
111-bit 表示，再单变量加入 P-layer reachability 特征。E2 不得把现有
test 结果包装为独立复验，也不得启动远程 GPU。

冻结执行命令：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate-innovation2-integral-ranking \
  --run-id i2_present_r5_integral_parity_ranking_utility_seed0 \
  --source-root outputs/local_diagnostic/i2_present_r5_integral_parity_calibration_seed0 \
  --output-root outputs/local_diagnostic/i2_present_r5_integral_parity_ranking_utility_seed0
```

E2 只能读取 E1 的 `structure_rates.csv` 与 `gate.json`，并固定产生：

```text
results.jsonl    # 线性、MLP、打乱标签 MLP 三行排序效用
ranking.csv      # 128 个结构的预测分数、排名、top-16 选择和 256-key 真值
gate.json        # 四个预注册门槛与下一步
curves.svg       # 中文 Spearman/top-16 对比图
progress.jsonl   # 只含后处理开始/结束，不得出现训练事件
```

## E2 完成结果

E2 已在冻结的 E1 test/stability 证据上完成。它没有重训、重新选择
checkpoint 或改变结构；输入仍是 E1 的 128 个 test 结构，真值仍由每个
结构 256 把新密钥的 `q=1` 率给出。`q1` 越低表示该结构越常保持 XOR
平衡，因此三个模型都按预测 `q1` 从低到高选择 top-16。

| 模型 | Spearman（预测 q1 vs 真实 q1） | top-16 真实平衡率 | 相对全局 |
|---|---:|---:|---:|
| 同输入线性 | `0.592055455` | `0.872802734` | `+0.156188965` |
| 结构 MLP | `0.685425839` | `0.922119141` | `+0.205505371` |
| 打乱标签 MLP | `0.200834464` | `0.695556641` | `-0.021057129` |

全体 128 个结构的真实平均平衡率为 `0.716613770`。冻结门控全部通过：

```text
MLP - linear Spearman                         = +0.093370384  PASS >= +0.05
MLP top-16 balance - global                  = +0.205505371  PASS >= +0.05
MLP top-16 balance - linear top-16           = +0.049316406  PASS >= +0.03
shuffled top-16 balance - global             = -0.021057129  PASS <= +0.02

status   = pass
decision = innovation2_integral_ranking_utility_advance_independent_confirmation
```

这个结果支持“当前 111-bit 结构表示具有候选排序/筛选效用”，但仍只来自
seed0 的一个本地 test split。`0.9221` 是 top-16 在 256 把抽样密钥上的
平均平衡率，不是 16 个确定性积分，也不是对所有密钥的证明。

完成产物：

```text
outputs/local_diagnostic/i2_present_r5_integral_parity_ranking_utility_seed0/
```

## 推荐下一步 E3：独立 seed1 排序确认

E3 只改变一个变量：`seed 0 -> seed 1`。结构/密钥生成、512/128/128/128
结构预算、16/32/32/32/256 把密钥、20 epochs、网络、线性/打乱控制、
校准和 E2 top-16 门槛全部保持不变。先重新运行 seed1 E1，再对其全新
128 个 test 结构和 256-key stability 标签运行相同 E2 后处理。

```text
research question = seed0 的排序/top-16 效用能否在独立 seed1 重现？
same-budget anchor = seed1 linear_same_input
required control = seed1 shuffled-label MLP + seed1 global prior
one changed variable = seed
execution = local CPU；不得启动远程 GPU
advance gate = seed1 再次通过 E2 四门
stop gate = candidate 不胜 linear、打乱控制异常，或 top-16 不过门
```

只有 seed1 再次通过，才把 seed0/seed1 合并裁决并进入 active/output/mask
几何组合留出；不得把几何留出与 seed 复验混在同一实验里。

冻结执行命令：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run-innovation2-integral-property \
  --run-id i2_present_r5_integral_parity_calibration_seed1 \
  --train-structures 512 --validation-structures 128 --test-structures 128 \
  --train-keys 16 --validation-keys 32 --test-keys 32 \
  --calibration-structures 128 --calibration-keys 32 \
  --stability-test-keys 256 \
  --epochs 20 --seed 1 --device cpu --gate-mode calibration \
  --output-root outputs/local_diagnostic/i2_present_r5_integral_parity_calibration_seed1

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate-innovation2-integral-ranking \
  --run-id i2_present_r5_integral_parity_ranking_utility_seed1 \
  --source-root outputs/local_diagnostic/i2_present_r5_integral_parity_calibration_seed1 \
  --output-root outputs/local_diagnostic/i2_present_r5_integral_parity_ranking_utility_seed1

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/summarize-innovation2-integral-ranking \
  --run-id i2_present_r5_integral_parity_ranking_utility_joint_seed0_seed1 \
  --source-gates \
    outputs/local_diagnostic/i2_present_r5_integral_parity_ranking_utility_seed0/gate.json \
    outputs/local_diagnostic/i2_present_r5_integral_parity_ranking_utility_seed1/gate.json \
  --output-root outputs/local_diagnostic/i2_present_r5_integral_parity_ranking_utility_joint_seed0_seed1
```

## E3 完成结果

seed1 严格保持 E1/E2 的全部预算和门槛，只改变 `seed=1`。seed1 的点概率
校准结果仍因 32-key/256-key 观测率不稳定而 held：

```text
candidate calibrated 256-key MAE               = 0.075657689  PASS <= 0.09
linear MAE - candidate MAE                     = 0.019570280  PASS >= 0.015
candidate AUC - linear AUC                     = 0.024616301  PASS >= 0.02
32-key vs 256-key observed-rate MAE             = 0.061187744  MISS <= 0.05
shuffled AUC                                    = 0.514466854  PASS

decision = innovation2_integral_rate_target_unstable
```

这再次说明精确点概率目标不稳定，但不阻止冻结的排序效用审判。seed1 E2
四门全部通过：

| 指标 | seed0 | seed1 | 冻结门槛 |
|---|---:|---:|---:|
| MLP Spearman | `0.685425839` | `0.782359110` | 支持指标 |
| MLP-linear Spearman | `+0.093370384` | `+0.112546944` | `>= +0.05` |
| MLP top-16 平衡率 | `0.922119141` | `0.963134766` | 支持指标 |
| MLP top-16 - 全局 | `+0.205505371` | `+0.245941162` | `>= +0.05` |
| MLP top-16 - linear top-16 | `+0.049316406` | `+0.047607422` | `>= +0.03` |
| shuffled top-16 - 全局 | `-0.021057129` | `+0.003997803` | `<= +0.02` |

联合裁决：

```text
candidate Spearman mean / range     = 0.733892474 / [0.685425839, 0.782359110]
candidate top-16 balance mean/range = 0.942626953 / [0.922119141, 0.963134766]
candidate-linear top-16 gain min    = 0.047607422
control-global top-16 gain max      = 0.003997803

status   = pass
decision = innovation2_integral_ranking_utility_two_seed_confirmed
```

完成产物：

```text
outputs/local_diagnostic/i2_present_r5_integral_parity_calibration_seed1/
outputs/local_diagnostic/i2_present_r5_integral_parity_ranking_utility_seed1/
outputs/local_diagnostic/i2_present_r5_integral_parity_ranking_utility_joint_seed0_seed1/
```

结论边界：双 seed 结果确认当前结构 MLP 在随机未见结构上具有排序和
top-16 候选筛选效用；它仍不是几何组合留出、远程/论文规模结果，也没有
证明 top-16 中任一结构对所有密钥确定平衡。

## 推荐下一步 E4：active/output/mask 几何组合留出

E4 只改变结构拆分规则，不改网络、标签、训练规模、密钥预算、epochs、
校准或排序门槛：将 `(active_nibble, output_nibble, output_mask)` 三元组作为
geometry id，训练、validation、calibration、test 四组 geometry 完全互斥；
每个 geometry 只采一个固定上下文，stability 仍复用 test geometry/结构并
更换 256 把新密钥。

```text
research question = 排序效用能否泛化到训练中从未出现的位置/掩码组合？
same-budget anchor = geometry-held-out linear_same_input
required control = geometry-held-out shuffled-label MLP + global prior
one changed variable = structure split: random-disjoint -> geometry-disjoint
scale = 512/128/128/128 structures; 16/32/32/32/256 keys; 20 epochs
seed = 0
execution = local CPU
readiness = geometry/key ownership互斥、标签双类、三模型及产物完整
advance gate = 原 E2 四门全部通过
stop gate = MLP不胜linear、shuffled超过+0.02或top-16任一主门失败
forbidden = 增加结构/epochs/seeds、改标签、启动远程GPU
```

实现时只给现有 runner 增加显式
`--structure-split-mode geometry-disjoint`，默认值保持当前 random-disjoint，
不得同时加入 P-layer 特征。冻结命令形状为：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run-innovation2-integral-property \
  --run-id i2_present_r5_integral_parity_geometry_holdout_seed0 \
  --structure-split-mode geometry-disjoint \
  --train-structures 512 --validation-structures 128 --test-structures 128 \
  --train-keys 16 --validation-keys 32 --test-keys 32 \
  --calibration-structures 128 --calibration-keys 32 \
  --stability-test-keys 256 \
  --epochs 20 --seed 0 --device cpu --gate-mode calibration \
  --output-root outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_seed0
```

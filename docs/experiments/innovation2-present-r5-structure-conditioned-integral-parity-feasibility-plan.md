# Innovation 2 PRESENT r5 结构条件积分 Parity 可行性实验

**日期：** 2026-07-15

**状态：** Smoke 通过 / 本地诊断完成 / held，校准后再扩展

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

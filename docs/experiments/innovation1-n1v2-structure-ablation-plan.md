# Innovation1 N1-v2 结构消融执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `skills/blockcipher-auto-research/SKILL.md` for experiment evidence gates, and use Karpathy-style coding discipline for implementation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 N1-v1 `present_nibble_paligned_mcnd` 只有小正向的基础上，验证 PRESENT nibble/P-layer 结构信息是否能通过 gated/cross interaction 产生稳定、可归因的增益。

**Architecture:** 不改 benchmark，不改 Zhang/Wang official protocol。先实现最小可归因模型族：SPN-only、late-fusion N1-v1、P-layer gated fusion N1-v2、shuffled-P ablation。先在 `262144/class` 做结构诊断，只有 true P-layer gated 明显优于 baseline / late-fusion / shuffled-P，才推进 1M 和多 seed。

**Tech Stack:** Python 3.10+, PyTorch, project matrix runner `scripts/train`, disk-backed dataset cache, JSONL/CSV/SVG artifacts, remote Windows A6000 `lxy-a6000`.

---

## 0. 背景与当前证据

已完成的 paper-scale seed0 诊断：

```text
run_id = i1_spn_present_mcnd_r7_1m_seed0_gpu1_retry1_20260626
scale = 1000000/class
strict_negative = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
checkpoint_metric = val_auc
```

结果：

| model | accuracy | calibrated_accuracy | AUC | loss | best_epoch |
|---|---:|---:|---:|---:|---:|
| `present_zhang_wang_keras_mcnd` | 0.715258 | 0.718335 | 0.793910375814 | 0.548134426178 | 17 |
| `present_nibble_paligned_mcnd` | 0.718995 | 0.719028 | 0.794619119358 | 0.543621080994 | 20 |

delta：

```text
accuracy_delta = +0.003737
calibrated_accuracy_delta = +0.000693
auc_delta = +0.000708743544
loss_delta = -0.004513345184
```

当前判断：

```text
N1-v1 = positive but weak
```

这不是失败，但不足以支撑强创新。下一步目标不是盲目上更大规模，而是回答：

```text
1. 结构信息是否真的贡献了增益？
2. true P-layer 是否优于 shuffled P-layer？
3. SPN view 应该是 late-fusion 附加特征，还是应该 gate / 调制 raw MCND 表征？
```

## 1. 不变量

所有诊断必须保持：

```text
cipher = PRESENT-80
rounds = 7
pairs_per_sample = 16
feature_encoding = ciphertext_pair_bits
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
loss = mse
optimizer = adam
weight_decay = 0.00001
lr_scheduler = official_cyclic
learning_rate = 0.0001
max_learning_rate = 0.002
checkpoint_metric = val_auc
restore_best_checkpoint = true
early_stopping_patience = 8
early_stopping_min_delta = 0.0001
batch_size = 1024
hidden_bits = 32
```

禁止事项：

```text
不改 validation data
不改 labels
不改 negative sample definition
不改 metric computation
不把 random-ciphertext negatives 和 strict negatives 混报
不把 multi-query aggregation 当 raw single-sample SOTA
```

## 2. 模型族定义

### N0: Zhang/Wang official baseline

```text
model_key = present_zhang_wang_keras_mcnd
role = same-budget baseline
```

### N1-v1: Late-fusion nibble/P-layer aligned MCND

已有：

```text
model_key = present_nibble_paligned_mcnd
role = current weak-positive innovation baseline
```

结构：

```text
raw_embedding = Zhang/Wang MCND encode(bits)
spn_embedding = token mixer over Delta C + InvP(Delta C)
classifier(concat(raw_embedding, spn_mean, spn_max))
```

### N1-S: SPN-only control

新增：

```text
model_key = present_nibble_paligned_spn_only
```

目的：

```text
验证 Delta C + InvP(Delta C) SPN view 自己有没有 real-vs-random 区分力。
```

设计：

```text
reuse PresentNibblePAlignedMCNDDistinguisher SPN view encoder
remove raw_branch
classifier(spn_mean, spn_max)
```

判定：

```text
如果 SPN-only 接近 chance，则 SPN view 本身太弱，只能作为辅助 gate。
如果 SPN-only 接近 N0，则结构 view 是强信号，后续值得强化。
```

### N1-v2: P-layer gated MCND

新增：

```text
model_key = present_nibble_paligned_gated_mcnd
```

目的：

```text
把 SPN view 从 late-fusion 附加特征升级为 raw MCND 表征的 gate / modulation。
```

最小结构：

```text
raw_embedding = Zhang/Wang MCND encode(bits)
spn_embedding = encode_spn_pairs(Delta C + InvP(Delta C))
gate = sigmoid(MLP(mean(spn_embedding), max(spn_embedding)))
gated_raw = raw_embedding * (1 + gate_scale * gate)
classifier(concat(gated_raw, spn_mean, spn_max))
```

默认参数：

```text
gate_scale = 0.25
blocks = 5
spn_mixer_depth = 2
activation = relu
norm = layernorm
```

为什么用 `1 + gate_scale * gate`：

```text
保留 raw MCND 主通道，不让未训练 gate 在初期直接关闭 raw 表征。
gate_scale 小，避免只是靠更大参数量硬推。
```

### N1-v2-shuffled: Shuffled P-layer gated ablation

新增：

```text
model_key = present_nibble_shuffled_paligned_gated_mcnd
```

目的：

```text
检验增益是否来自真实 PRESENT P-layer 对齐，而不是来自多一个分支/参数量。
```

设计：

```text
与 N1-v2 完全相同
唯一差异：InvP index 替换为固定 deterministic shuffled permutation
```

shuffle 规则：

```text
使用固定 seed = 20260627 对 64 个 bit position 生成 permutation
buffer 持久为模型常量
不在训练中随机变化
```

判定：

```text
如果 true-P <= shuffled-P，则不能声称 P-layer topology 有贡献。
如果 true-P 明显 > shuffled-P，结构创新才可归因。
```

## 3. 实现计划

### Task 1: 加模型单元测试

**Files:**

- Modify: `tests/test_project_structure.py`

- [ ] **Step 1: 添加模型构建和 forward 测试**

在 `tests/test_project_structure.py` 增加：

```python
def test_present_nibble_paligned_ablation_models_build_and_forward():
    input_bits = 16 * 128
    features = torch.randint(0, 2, (4, input_bits), dtype=torch.float32)

    for model_key in [
        "present_nibble_paligned_spn_only",
        "present_nibble_paligned_gated_mcnd",
        "present_nibble_shuffled_paligned_gated_mcnd",
    ]:
        model = build_model(
            model_key=model_key,
            input_bits=input_bits,
            hidden_bits=32,
            pair_bits=128,
            model_options={"blocks": 2, "spn_mixer_depth": 1, "activation": "relu", "norm": "layernorm"},
        )
        logits = model(features)
        assert logits.shape == (4, 1)
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_project_structure.py::test_present_nibble_paligned_ablation_models_build_and_forward -q
```

Expected:

```text
FAIL with unknown model key or missing class
```

### Task 2: 抽出 SPN view encoder 并实现三种变体

**Files:**

- Modify: `src/blockcipher_nd/models/structure/spn/present_nibble_paligned_mcnd.py`
- Modify: `src/blockcipher_nd/models/structure/spn/__init__.py`

- [ ] **Step 1: 在现有文件内增加共享基类**

将当前 `PresentNibblePAlignedMCNDDistinguisher` 内部的 SPN view 逻辑整理成内部 helper class：

```python
class _PresentNibblePAlignedSpnEncoder(nn.Module):
    def __init__(..., p_alignment: str = "true") -> None:
        ...
```

支持：

```text
p_alignment = "true"
p_alignment = "shuffled"
```

输出：

```python
def forward(self, features: torch.Tensor) -> torch.Tensor:
    """Return per-pair SPN embeddings with shape [batch, pairs, embedding_bits]."""
```

- [ ] **Step 2: 保持 N1-v1 外部行为不变**

`PresentNibblePAlignedMCNDDistinguisher` 继续支持当前构造参数和 forward 输出，不改变已有 plan。

- [ ] **Step 3: 添加 SPN-only 类**

新增：

```python
class PresentNibblePAlignedSpnOnlyDistinguisher(nn.Module):
    ...
```

forward：

```python
spn_pair_embeddings = self.spn_encoder(features)
spn_mean = spn_pair_embeddings.mean(dim=1)
spn_max = spn_pair_embeddings.max(dim=1).values
return self.classifier(torch.cat([spn_mean, spn_max], dim=1))
```

- [ ] **Step 4: 添加 gated MCND 类**

新增：

```python
class PresentNibblePAlignedGatedMCNDDistinguisher(nn.Module):
    ...
```

forward：

```python
raw_embedding = self.raw_branch.encode(features)
spn_pair_embeddings = self.spn_encoder(features)
spn_mean = spn_pair_embeddings.mean(dim=1)
spn_max = spn_pair_embeddings.max(dim=1).values
gate = torch.sigmoid(self.gate(torch.cat([spn_mean, spn_max], dim=1)))
gated_raw = raw_embedding * (1.0 + self.gate_scale * gate)
return self.classifier(torch.cat([gated_raw, spn_mean, spn_max], dim=1))
```

- [ ] **Step 5: 添加 shuffled-P gated 类**

新增：

```python
class PresentNibbleShuffledPAlignedGatedMCNDDistinguisher(
    PresentNibblePAlignedGatedMCNDDistinguisher
):
    ...
```

只改变 `p_alignment="shuffled"`。

- [ ] **Step 6: 更新 `__all__` 和 package exports**

在 `src/blockcipher_nd/models/structure/spn/__init__.py` 导出：

```python
PresentNibblePAlignedSpnOnlyDistinguisher
PresentNibblePAlignedGatedMCNDDistinguisher
PresentNibbleShuffledPAlignedGatedMCNDDistinguisher
```

### Task 3: 注册模型 key

**Files:**

- Modify: `src/blockcipher_nd/registry/model_families/spn.py`

- [ ] **Step 1: 导入新类**

添加：

```python
PresentNibblePAlignedSpnOnlyDistinguisher,
PresentNibblePAlignedGatedMCNDDistinguisher,
PresentNibbleShuffledPAlignedGatedMCNDDistinguisher,
```

- [ ] **Step 2: 添加 build branches**

新增 model keys：

```text
present_nibble_paligned_spn_only
present_nibble_paligned_gated_mcnd
present_nibble_shuffled_paligned_gated_mcnd
```

每个分支接受：

```text
blocks
spn_token_dim
spn_mixer_depth
token_mlp_ratio
activation
norm
dropout
initial_kernel_sizes
residual_kernel_size
gate_scale
```

`gate_scale` 只对 gated 模型有效，默认 `0.25`。

- [ ] **Step 3: 运行单测**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_project_structure.py::test_present_nibble_paligned_ablation_models_build_and_forward -q
```

Expected:

```text
1 passed
```

### Task 4: 加 262k 结构消融矩阵

**Files:**

- Create: `configs/experiment/innovation1/innovation1_spn_present_n1v2_structure_ablation_r7_262k.csv`
- Modify: `tests/test_project_structure.py`

- [ ] **Step 1: 创建 CSV**

Rows：

```text
1. present_zhang_wang_keras_mcnd
2. present_nibble_paligned_mcnd
3. present_nibble_paligned_spn_only
4. present_nibble_paligned_gated_mcnd
5. present_nibble_shuffled_paligned_gated_mcnd
```

共同字段：

```text
cipher = PRESENT-80
structure = SPN
rounds = 7
seed = 0
samples_per_class = 262144
pairs_per_sample = 16
feature_encoding = ciphertext_pair_bits
negative_mode = encrypted_random_plaintexts
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
key_rotation_interval = 0
sample_structure = zhang_wang_case2_official_mcnd
integral_active_nibble = 0
difference_profile = present_zhang_wang2022_mcnd
difference_member = 0
loss = mse
learning_rate = 0.0001
optimizer = adam
weight_decay = 0.00001
lr_scheduler = official_cyclic
max_learning_rate = 0.002
checkpoint_metric = val_auc
restore_best_checkpoint = true
early_stopping_patience = 8
early_stopping_min_delta = 0.0001
```

Evidence language：

```text
MEDIUM 262144/class structure ablation for I1-SPN-001 N1-v2; not formal reproduction or breakthrough evidence.
```

- [ ] **Step 2: 添加 plan 解析测试**

新增测试：

```python
def test_present_n1v2_262k_structure_ablation_plan_is_same_protocol():
    plan = "configs/experiment/innovation1/innovation1_spn_present_n1v2_structure_ablation_r7_262k.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_paligned_mcnd",
        "present_nibble_paligned_spn_only",
        "present_nibble_paligned_gated_mcnd",
        "present_nibble_shuffled_paligned_gated_mcnd",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
```

- [ ] **Step 3: 运行 plan 测试**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_project_structure.py::test_present_n1v2_262k_structure_ablation_plan_is_same_protocol -q
```

Expected:

```text
1 passed
```

### Task 5: 本地 smoke

**Files:**

- Create: `configs/experiment/innovation1/innovation1_spn_present_n1v2_structure_ablation_smoke.csv`

- [ ] **Step 1: 创建 smoke CSV**

创建：

```text
configs/experiment/innovation1/innovation1_spn_present_n1v2_structure_ablation_smoke.csv
```

Rows 与 262k 结构消融一致：

```text
1. present_zhang_wang_keras_mcnd
2. present_nibble_paligned_mcnd
3. present_nibble_paligned_spn_only
4. present_nibble_paligned_gated_mcnd
5. present_nibble_shuffled_paligned_gated_mcnd
```

差异字段：

```text
samples_per_class = 8
model_options for baseline / N1-v1 / gated variants use {"blocks":1,"spn_mixer_depth":1,"activation":"relu","norm":"layernorm"}
evidence includes SMOKE only
```

- [ ] **Step 2: 跑 CPU smoke**

Run：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_n1v2_structure_ablation_smoke.csv \
  --epochs 1 \
  --batch-size 8 \
  --hidden-bits 16 \
  --device cpu \
  --output outputs/smoke/n1v2_structure_ablation_smoke.jsonl
```

Expected：

```text
5 rows written
no exception
```

### Task 6: 远程 262k 结构消融

**Files:**

- Create: `configs/remote/innovation1_spn_present_n1v2_structure_ablation_r7_262k_gpu1_YYYYMMDD.json`
- Update: `docs/experiments/innovation1-n1v2-structure-ablation-plan.md` launch record

- [ ] **Step 1: 创建 remote config**

建议 run id：

```text
i1_spn_n1v2_ablation_r7_262k_seed0_gpu1_YYYYMMDD
```

配置：

```json
{
  "run_id": "i1_spn_n1v2_ablation_r7_262k_seed0_gpu1_YYYYMMDD",
  "task_name": "i1_spn_n1v2_ablation_r7_262k_seed0_gpu1_YYYYMMDD",
  "plan": "configs\\experiment\\innovation1\\innovation1_spn_present_n1v2_structure_ablation_r7_262k.csv",
  "expected_rows": 5,
  "device": "cuda:1",
  "epochs": 20,
  "batch_size": 1024,
  "hidden_bits": 32,
  "learning_rate": 0.0001,
  "optimizer": "adam",
  "weight_decay": 0.00001,
  "loss": "mse",
  "lr_scheduler": "official_cyclic",
  "max_learning_rate": 0.002,
  "checkpoint_metric": "val_auc",
  "restore_best_checkpoint": true,
  "early_stopping_patience": 8,
  "early_stopping_min_delta": 0.0001,
  "dataset_cache": true,
  "dataset_cache_root": "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\shared_dataset_cache",
  "dataset_cache_chunk_size": 8192,
  "dataset_cache_workers": 4,
  "branch": "main",
  "project_id": "blockcipher-structure-adaptive-nd",
  "repo_url": "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git",
  "result_sync": "local_tmux_monitor_scp_fallback",
  "claim_scope": "MEDIUM 262144/class structure ablation only; not formal or breakthrough evidence",
  "launch_policy": "use pushed GitHub commit; keep artifacts under G:\\lxy; cmd.exe /c only; tmux monitor retrieves results automatically; shared dataset cache required"
}
```

- [ ] **Step 2: 启动前 audit**

检查：

```text
cmd.exe /c present
cmd.exe /k absent
all project-generated paths under G:\lxy
dataset_cache_root uses G:\lxy\...\shared_dataset_cache
repo_url uses git@github.com with dedicated remote key
```

- [ ] **Step 3: 启动远程 run**

遵守 remote workflow：

```text
launch from pushed commit
use run-owned clean clone under G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>
start local tmux monitor
do not main-thread SSH-poll after handoff unless monitor unhealthy
```

- [ ] **Step 4: 完成后自动文档**

结果一旦 gate 通过并拉回：

```text
update this docs/experiments plan automatically
include JSONL metrics, deltas, true-P vs shuffled-P, keep/discard decision
commit and push
```

## 4. 262k 判定门槛

Primary metric：

```text
AUC
```

Secondary：

```text
calibrated_accuracy
loss
best_epoch
train/val gap
```

比较关系：

```text
N0 = present_zhang_wang_keras_mcnd
N1-v1 = present_nibble_paligned_mcnd
N1-S = present_nibble_paligned_spn_only
N1-v2 = present_nibble_paligned_gated_mcnd
N1-v2-shuffled = present_nibble_shuffled_paligned_gated_mcnd
```

继续条件：

```text
N1-v2 AUC >= N0 AUC + 0.002
and N1-v2 AUC >= N1-v1 AUC + 0.001
and N1-v2 AUC >= N1-v2-shuffled AUC + 0.001
and N1-v2 calibrated_accuracy >= N0 calibrated_accuracy
```

弱继续条件：

```text
N1-v2 AUC > all alternatives
but margin < 0.001
```

动作：

```text
repeat 262k with seed1 before 1M
```

停止或降级条件：

```text
N1-v2 <= N1-v1
or true-P <= shuffled-P
or SPN-only near chance and gated does not improve N0
```

动作：

```text
do not scale N1-v2 to 1M
document as negative/weak route
shift to DDT transition consistency or pair-set trail-family consistency
```

## 5. 1M / 多 seed 门槛

只有 262k 满足继续条件才开：

```text
I1-SPN-001 N1-v2 1M seed0
```

1M seed0 继续条件：

```text
N1-v2 AUC >= N0 AUC + 0.0015
or calibrated_accuracy >= N0 calibrated_accuracy + 0.0015
and true-P ablation remains better than shuffled-P at 262k
```

如果 1M seed0 仍只是 `+0.0005 ~ +0.0010`：

```text
do not call breakthrough
run seed1 only if training curves show stable lower loss and better calibration
```

正式结论至少需要：

```text
1M/class
seeds = 0,1,2
same protocol
strict negatives
mean/std reported
paired comparison against N0
```

## 6. 产物要求

每个非 smoke run 必须保留：

```text
plan CSV
remote config JSON
source commit
remote run dir
local retrieved dir
results JSONL
history CSV
curves SVG
result_gate.txt
progress JSONL
stdout/stderr
git/gpu/torch evidence
```

若远程因缺 `matplotlib` 画图失败：

```text
training gate passed => result still valid
retrieve JSONL/logs
regenerate curves/history locally with scripts/plot-results
record plotting failure in docs if it affects artifacts
```

## 7. 执行顺序

- [x] Task 1: 加模型构建/forward 失败测试。
- [x] Task 2: 实现 SPN-only、gated true-P、gated shuffled-P。
- [x] Task 3: 注册新 model keys。
- [x] Task 4: 加 262k structure ablation CSV 和解析测试。
- [x] Task 5: CPU smoke 验证 5 行模型都能训练 1 epoch。
- [x] Task 6: 提交并推送代码/config/docs。
- [x] Task 7: 远程启动 262k structure ablation。
- [x] Task 8: 结果完成后自动更新 `docs/experiments/`，再决定是否上 1M。

## 8. 262k seed0 结果记录

```text
run_id = i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627
status = completed remotely, fallback-retrieved locally, plan-aligned
remote = lxy-a6000 GPU0
source_commit = 822236302166b5add6f89063971704f9e2f8a1b5
scale = 262144/class
rounds = 7
seed = 0
pairs_per_sample = 16
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
checkpoint_metric = val_auc
local_dir = outputs/remote_results/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627/
```

远程状态：

```text
result_lines = 5
stderr = empty
result_gate = result_lines=5, expected_rows=5
```

本地 gate：

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_n1v2_structure_ablation_r7_262k.csv \
  --results outputs/remote_results/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627.jsonl \
  --expected-rows 5 \
  --output outputs/remote_results/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627_local_result_gate.json
```

结果：

```text
status = pass
plan_rows = 5
result_rows = 5
missing_result_keys = []
unexpected_result_keys = []
field_mismatches = []
```

本地重绘：

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/remote_results/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627.jsonl \
  --output outputs/remote_results/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627_curves.svg \
  --history-csv outputs/remote_results/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627_history.csv \
  --title i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627
```

指标：

| model | role | accuracy | calibrated_accuracy | AUC | loss | epochs_ran | best_epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| `present_zhang_wang_keras_mcnd` | N0 baseline | 0.676277 | 0.711231 | 0.784541 | 0.612181 | 14 | 6 |
| `present_nibble_paligned_mcnd` | N1-v1 late fusion | 0.673466 | 0.710648 | 0.784299 | 0.611594 | 13 | 5 |
| `present_nibble_paligned_spn_only` | N1-S SPN-only | 0.716358 | 0.716434 | 0.791488 | 0.547521 | 20 | 14 |
| `present_nibble_paligned_gated_mcnd` | N1-v2 true-P gated | 0.702484 | 0.710701 | 0.784897 | 0.566700 | 14 | 6 |
| `present_nibble_shuffled_paligned_gated_mcnd` | N1-v2 shuffled control | 0.668278 | 0.710266 | 0.784281 | 0.622337 | 14 | 6 |

N1-v2 true-P gated deltas：

| comparison | delta_accuracy | delta_calibrated_accuracy | delta_AUC |
|---|---:|---:|---:|
| vs N0 baseline | +0.026207 | -0.000530 | +0.000356 |
| vs N1-v1 late fusion | +0.029018 | +0.000053 | +0.000598 |
| vs shuffled control | +0.034206 | +0.000435 | +0.000615 |
| vs SPN-only | -0.013874 | -0.005733 | -0.006591 |

门槛判定：

```text
N1-v2 AUC >= N0 AUC + 0.002       false, observed +0.000356
N1-v2 AUC >= N1-v1 AUC + 0.001    false, observed +0.000598
N1-v2 AUC >= shuffled AUC + 0.001 false, observed +0.000615
N1-v2 calibrated_accuracy >= N0   false, observed -0.000530
```

结论：

```text
N1-v2 true-P gated = weak positive diagnostic, not enough to scale to 1M as the main route.
SPN-only = strongest row in this 262k seed0 ablation.
```

解释：

```text
true-P gated 比 N0、N1-v1、shuffled control 的 AUC 都略高，但幅度没有达到预设门槛；
calibrated_accuracy 也没有超过 N0。
因此不能把 N1-v2 gated 写成有效结构创新。

更重要的信号是 SPN-only 明显高于所有 MCND/gated 组合。
这说明 PRESENT nibble/P-layer structure view 本身可能有真实区分信号，
但当前的 gate/fusion 方式没有把这个信号稳定注入 MCND 主干。
下一步应优先研究 SPN-only 为什么有效，以及如何围绕 SPN transition view 设计主干，
而不是直接把 N1-v2 gated 扩到 1M。
```

当前状态：

```text
classification = medium diagnostic result, not formal evidence
decision = do not scale current N1-v2 gated to 1M
next_action = analyze SPN-only route and design N1-S/N2 transition-consistency follow-up
```

## 9. 当前推荐决策

当前不建议：

```text
直接继续 N1-v1 1M 多 seed
直接上 full SPN-GNN
直接修改 benchmark
直接把当前 N1-v2 gated 扩到 1M
```

当前建议：

```text
围绕 262k 消融里最强的 SPN-only 结果继续做可归因分析。
优先验证 SPN-only 的信号来源、泛化稳定性和与 MCND 融合失败原因。
下一轮不改 benchmark，仍保持 strict negative 和 Zhang/Wang Case2 official MCND 数据构造。
```

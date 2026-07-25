# Innovation 1 Dialga-128 Runtime-E5 D6 Gated-Residual Plan

Date: 2026-07-25

## Status

```text
phase = completed local architecture diagnostic
status = hold
decision = innovation1_dialga_runtime_e5_d6_not_supported
source_d3 = completed same-budget Runtime-E4 anchor
source_d4 = fifth-round data-depth loss isolated
source_d5 = no single-bit difference candidate
remote_scale = prohibited
```

## Research Question

Can a runtime-parameterized SPN model preserve an independently useful
no-topology prediction while using the supplied Dialga topology only as a
bounded residual correction, thereby improving prefix-r5 discrimination and
retaining identifiable correct-topology margins?

D4 showed that Runtime-E4 can consume the shifted rounds-3/4 window on r4 data,
but r5 data destroys most of the signal. D5 found no replacement single-bit
difference that passed both fixed keys. D6 therefore changes exactly one
variable: the architecture by which topology enters the prediction.

## Candidate Architecture

`RuntimeE5GatedResidualSpnDistinguisher` reuses one Runtime-E4 encoder and
computes:

```text
base = E4.encode(ciphertext pairs, relation_mode=independent)
topology = E4.encode(ciphertext pairs, relation_mode=true)
residual = topology - base
logit = E4.classifier(base) + tanh(gate) * tanh(residual_head(residual))
```

The scalar gate is initialized to exactly zero. The base logit is always added
directly and is never replaced by the topology path. The residual correction is
bounded. Under the no-topology role, `topology == base` and the residual is
exactly zero for every input regardless of the learned gate. The implementation
therefore returns a literal zero correction for `relation_mode=independent`;
this prevents residual-head biases from turning a zero residual into a learned
topology-free offset.

This architecture combines a cipher-agnostic-style independent prediction path
with the project's runtime topology conditioning. It does not add DDT, trail,
partial-decryption, guessed-key or cipher-name features.

## Functional Readiness Gate

Before training:

1. For random binary pair inputs, forcing `gate=0` must make the full output
   exactly equal to the independent base logit.
2. The base logits must be finite and nonconstant across distinct samples.
3. In `relation_mode=independent`, correct and corrupted linear topologies must
   produce exactly equal outputs and an exactly zero residual representation.
4. In `relation_mode=true`, correct and corrupted topologies must produce a
   nonzero residual difference.
5. Cell relabeling with the corresponding input-bit relabeling must preserve
   logits, and gradients must remain finite.
6. All three registry roles must have identical `492644`-parameter geometry and
   preserve the Dialga descriptor/control fingerprints.

## Frozen Training Matrix

| Role | Model | Runtime intervention |
| --- | --- | --- |
| correct | `runtime_spn_e5_gated_residual_true` | exact rounds 3/4 topology |
| corrupted | `runtime_spn_e5_gated_residual_corrupted` | deterministic corruption seed `20260725` |
| no topology | `runtime_spn_e5_gated_residual_independent` | identity cell relation |

Frozen protocol:

```text
cipher = Dialga-128 prefix-r5
input difference = 0x40
pairs/sample = 4
train = 2048/class
validation = 1024/class
seeds = 0, 1
epochs = 10
loss = MSE
optimizer = Adam, lr 1e-4, weight decay 1e-5
train key = 0
validation key = 0x11 repeated over 32 bytes
negative definition = encrypted random plaintext pairs
runtime window = round_start 3, rounds 2
dataset cache = exact completed D3 cache
device = local CPU
```

The completed D3 Runtime-E4 correct rows are the same-budget anchors:

```text
seed0 = 0.5073299408
seed1 = 0.4933295250
```

## Gate

Every seed must independently satisfy:

```text
correct AUC >= 0.520
correct - corrupted >= +0.005
correct - no topology >= +0.005
correct - D3 Runtime-E4 correct >= +0.010
```

The gate also requires six complete rows, exact cache reuse, equal geometry,
best-checkpoint restoration, complete ten-epoch histories, finite learned gate
values, source-D3 gate recomputation, exact descriptor/control fingerprints and
strict encrypted-random-plaintext negatives.

## Completed Result

Run:

```text
run_id = i1_dialga128_runtime_e5_d6_r5_2048_seed0_seed1_20260725
result_index = 001
plan_validation = pass, 6/6 rows
protocol_checks = 20/20 pass
cache_audit = 12/12 train/validation reuses, 0 generation events
visual_qa_redraw = pass after moving the middle-panel legend away from a value label
```

Best-checkpoint AUC:

| Seed | Correct topology | Corrupted topology | No topology | Old D3 correct |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.538021088 | 0.524413586 | 0.544159889 | 0.507329941 |
| 1 | 0.498243809 | 0.511754513 | 0.519388676 | 0.493329525 |

Per-seed margins:

| Seed | Correct - corrupted | Correct - no topology | Correct - old D3 |
| ---: | ---: | ---: | ---: |
| 0 | +0.013607502 | -0.006138802 | +0.030691147 |
| 1 | -0.013510704 | -0.021144867 | +0.004914284 |

Learned bounded topology gates:

| Seed | Correct topology | Corrupted topology | No topology |
| ---: | ---: | ---: | ---: |
| 0 | +0.000443830 | +0.000723509 | 0.000000000 |
| 1 | -0.000281767 | -0.000017275 | 0.000000000 |

The first attempt was stopped after 3/6 rows because the no-topology role
exposed a protocol bug: a biased residual head mapped the exact zero residual to
a nonzero correction. That incomplete run is retained only under
`outputs/invalid_runs/` and is not indexed as evidence. After the no-topology
path was made identically equal to the independent base, all six rows were
rerun from scratch. The final no-topology gates remained exactly zero.

The valid D6 matrix does not support the architecture on Dialga prefix-r5.
Seed0 improves over the old correct-topology D3 row and beats the corrupted
control, but loses to the no-topology base. Seed1 remains below 0.5, loses to
both controls, and improves over old D3 by less than +0.010. The very small
learned gates further show that the correct-topology residual was not a stable
source of the recovered seed0 signal.

## Evidence-Backed Next Action

Run D7 as a regression on the already strong Dialga prefix-r4 D1 mechanism.
This is not a retry of the failed five-round route and not a scale-up.

```text
research_question = can unchanged Runtime-E5 preserve the known D1 r4 mechanism?
same_budget_anchor = completed D1 Runtime-E4 correct/corrupted/no-topology rows
one_changed_variable = Runtime-E4 -> Runtime-E5 gated residual architecture
data = exact D1 prefix-r4 cache, difference 0x40, 4 pairs/sample
scale = 2048/class train, 1024/class validation
seeds = 0, 1
epochs = 10
execution = local CPU
required_controls = correct, corrupted, no topology
```

D1 correct-topology anchors are `0.958416939` and `0.958679199`. D7 advances
only if both E5 correct rows remain within `0.010` of the matching D1 anchor and
each beats corrupted and no-topology by at least `0.005`. A pass retains E5 as
a mechanically sound architecture but does not reopen Dialga r5; a miss means
discard E5 and return Innovation 1 architecture work to the supported Runtime-E4
and cross-cipher evidence. Do not change difference, data, pairs, keys, epochs,
negative definition, metric, or launch a remote run.

## Decision Routes

- Full pass: freeze both correct D6 best checkpoints and run a same-checkpoint
  correct/corrupted/no-topology swap before any sample increase.
- Candidate improves D3 but misses topology margins: retain the independent-base
  improvement only; audit the learned gate and same-checkpoint interventions,
  but do not call it topology attribution.
- Candidate does not improve both seeds: stop the Dialga prefix-r5 E5 branch and
  test the architecture only on the strong r4 mechanism as a regression/control
  before deciding whether the general Runtime-E5 implementation is retained.
- Protocol failure: repair only implementation or source binding with all data,
  models, seeds and thresholds frozen.

Observed route: `candidate does not improve both seeds`; follow the D7 r4
regression route above, then retain or discard E5 without another prefix-r5 run.

## Explicitly Blocked

- No input-difference, sample, pair, epoch, key, negative or metric change.
- No additional architectures or more than the three required D6 roles.
- No remote GPU launch or medium/formal scale-up.
- No Dialga attack, paper reproduction, SOTA or universal-SPN claim.

## Execution

```bash
RUN_ID=i1_dialga128_runtime_e5_d6_r5_2048_seed0_seed1_20260725
RUN_ROOT=outputs/local_diagnostic/${RUN_ID}
D3_ROOT=outputs/local_diagnostic/i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e5_d6_r5_2048_seed0_seed1.csv \
  --device cpu \
  --dataset-cache-root "${D3_ROOT}/cache" \
  --dataset-cache-chunk-size 1024 \
  --dataset-cache-workers 1 \
  --checkpoint-output-dir "${RUN_ROOT}/checkpoints" \
  --progress-output "${RUN_ROOT}/progress.jsonl" \
  --output "${RUN_ROOT}/results.jsonl"

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e5_d6_r5_2048_seed0_seed1.csv \
  --results "${RUN_ROOT}/results.jsonl" \
  --expected-rows 6

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-runtime-spn-dialga-d6 \
  --run-id "${RUN_ID}" \
  --run-root "${RUN_ROOT}" \
  --d3-root "${D3_ROOT}"
```

After the result, render and inspect the Chinese SVG with `visual-qa-redraw`,
refresh both recent-result indexes, record the metrics and evidence-backed next
action here, then run focused regression before commit and push.

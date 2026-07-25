# Innovation 1 Dialga-128 Runtime-E5 D7 Four-Round Regression Plan

Date: 2026-07-25

## Status

```text
phase = completed local architecture regression
status = hold
decision = innovation1_dialga_runtime_e5_d7_r4_regression_not_supported
source_d1 = completed Runtime-E4 four-round pass
source_d6 = completed Runtime-E5 five-round hold
remote_scale = prohibited
```

## Research Question

Can the unchanged Runtime-E5 gated-residual architecture preserve the already
established Dialga-128 prefix-r4 Runtime-E4 mechanism under the exact D1 data,
training budget and topology controls?

D6 did not support Runtime-E5 at prefix-r5. Before discarding the architecture,
D7 tests whether E5 is mechanically capable of learning the strong four-round
mechanism that Runtime-E4 already learned. This is an architecture regression,
not a retry of the five-round route and not a new distinguishing claim.

## Single Changed Variable

```text
anchor architecture = Runtime-E4 recurrent-window backbone
candidate architecture = Runtime-E5 independent base + bounded topology residual
all data and training fields = exact D1
```

Required roles:

| Role | Model | Runtime intervention |
| --- | --- | --- |
| correct | `runtime_spn_e5_gated_residual_true` | exact rounds 2/3 topology |
| corrupted | `runtime_spn_e5_gated_residual_corrupted` | deterministic corruption seed `20260725` |
| no topology | `runtime_spn_e5_gated_residual_independent` | exact independent base; zero topology correction |

## Frozen Protocol

```text
cipher = Dialga-128 prefix-r4
input difference = 0x40
pairs/sample = 4 independent ciphertext pairs
train = 2048/class, 4096 total rows
validation = 1024/class, 2048 total rows
seeds = 0, 1
epochs = 10
loss = MSE
optimizer = Adam, lr 1e-4, weight decay 1e-5
train key = 0
validation key = 0x11 repeated over 32 bytes
negative definition = encrypted random plaintext pairs
runtime window = round_start 2, rounds 2
dataset cache = exact completed D1 cache, reuse only
device = local CPU
```

The same-budget D1 Runtime-E4 correct-topology anchors are:

```text
seed0 = 0.958416939
seed1 = 0.958679199
```

## Readiness And Protocol Gate

D7 requires all six rows, exact Runtime-E5 models and `492644`-parameter
geometry, complete ten-epoch best-checkpoint histories, finite learned topology
gates, strict encrypted-random-plaintext negatives and exact D1 descriptor
fingerprints. The no-topology gate must remain exactly zero.

The source D1 results, persisted gate and validation must replay exactly. All
12 train/validation cache events must reuse the four exact D1 cache leaves,
with no data-generation event.

## Research Gate

Each seed must independently satisfy all four checks:

```text
E5 correct AUC >= 0.520
E5 correct - E5 corrupted >= +0.005
E5 correct - E5 no topology >= +0.005
E5 correct - D1 E4 correct >= -0.010
```

Decision routes:

- Full pass: retain Runtime-E5 as a mechanically sound optional architecture,
  but keep Dialga prefix-r5 closed and prioritize supported Runtime-E4
  cross-cipher work.
- Any research-gate miss: discard Runtime-E5 gated residuals, retain Runtime-E4
  as the supported runtime backbone and run no further Dialga E5 experiment.
- Any protocol miss: repair only the failed frozen check and rerun unchanged.

## Explicitly Blocked

- No remote GPU launch or sample increase.
- No new difference, pair count, key, epoch, negative definition or metric.
- No reopening Dialga prefix-r5 after a four-round pass.
- No claim that retaining a D1 signal means E5 improves Runtime-E4.
- No Dialga attack, paper reproduction, SOTA or universal-SPN claim.

## Execution

```bash
RUN_ID=i1_dialga128_runtime_e5_d7_r4_2048_seed0_seed1_20260725
RUN_ROOT=outputs/local_diagnostic/${RUN_ID}
D1_ROOT=outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e5_d7_r4_2048_seed0_seed1.csv \
  --device cpu \
  --dataset-cache-root "${D1_ROOT}/cache" \
  --dataset-cache-chunk-size 1024 \
  --dataset-cache-workers 1 \
  --checkpoint-output-dir "${RUN_ROOT}/checkpoints" \
  --progress-output "${RUN_ROOT}/progress.jsonl" \
  --output "${RUN_ROOT}/results.jsonl"

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e5_d7_r4_2048_seed0_seed1.csv \
  --results "${RUN_ROOT}/results.jsonl" \
  --expected-rows 6

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-runtime-spn-dialga-d7 \
  --run-id "${RUN_ID}" \
  --run-root "${RUN_ROOT}" \
  --d1-root "${D1_ROOT}"
```

After completion, render and inspect the Chinese SVG through
`visual-qa-redraw`, refresh both recent-result indexes, record the metrics and
evidence-backed next action here, then run focused regression before commit and
push.

## Completed Result

```text
run_id = i1_dialga128_runtime_e5_d7_r4_2048_seed0_seed1_20260725
result_index = 001
plan_validation = pass, 6/6 rows
protocol_checks = 23/23 pass
research_checks = 6/8 pass
cache_audit = 12/12 train/validation reuses, 0 generation events
history = 60/60 epoch rows
visual_qa_redraw = pass after moving the left legend below the plot
```

Best-checkpoint AUC:

| Seed | E5 correct | E5 corrupted | E5 no topology | D1 E4 correct |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.920428753 | 0.898289680 | 0.497210026 | 0.958416939 |
| 1 | 0.917288303 | 0.887289047 | 0.503405571 | 0.958679199 |

Per-seed margins:

| Seed | Correct - corrupted | Correct - no topology | Correct - D1 |
| ---: | ---: | ---: | ---: |
| 0 | +0.022139072 | +0.423218727 | -0.037988186 |
| 1 | +0.029999256 | +0.413882732 | -0.041390896 |

Learned bounded topology gates:

| Seed | Correct topology | Corrupted topology | No topology |
| ---: | ---: | ---: | ---: |
| 0 | -0.078133568 | -0.071212649 | 0.000000000 |
| 1 | +0.070230946 | +0.055406161 | 0.000000000 |

Correct topology beats corrupted and no-topology controls on both seeds, so
the supplied Dialga topology remains useful. However, both correct rows lose
far more than the allowed `0.010` relative to the matching D1 Runtime-E4
anchors: `-0.037988` and `-0.041391`. The gated residual therefore weakens the
known four-round mechanism even where the task is easy enough for Runtime-E4.
This is sufficient to discard Runtime-E5 as the supported backbone without
reopening its five-round branch.

## Evidence-Backed Next Action

Return to Runtime-E4 and run an X3 readiness audit for a new held-out
cross-cipher target: frozen GIFT-64 Runtime-E4 backbone to Dialga-128 target
head. This directly advances the method-level claim across both block size and
heterogeneous runtime topology, whereas another Dialga architecture tweak does
not.

The readiness question is exact:

```text
can the completed GIFT Runtime-E4 source states load strictly into the
Dialga-128 Runtime-E4 adapter while preserving source hashes, frozen-backbone
ownership, nonconstant target features and classifier-only gradients?
```

Readiness reuses the immutable GIFT X2 source checkpoints and exact Dialga D1
cache. It generates no dataset and performs no performance interpretation.

If and only if readiness passes, freeze an X3 local matrix:

```text
research question = does a GIFT-trained Runtime-E4 backbone retain useful
                    Dialga information after training only the target head?
same-budget anchor = D1 end-to-end Dialga Runtime-E4 rows on the same cache
one changed variable = full Dialga training -> frozen GIFT backbone + new head
roles = correct source/correct target, corrupted source/correct target,
        correct source/corrupted target, random frozen backbone/correct target
data = exact D1 prefix-r4 cache, difference 0x40, 4 pairs/sample
scale = 2048/class train, 1024/class validation
seeds = 0, 1
epochs = 5
execution = local CPU
advance gate = both candidates AUC >= 0.55 and exceed all three controls by
               at least +0.005
```

If strict cross-block-size loading, frozen ownership, feature variation or
classifier-only gradients fail, stop X3 before training and retain the
existing GIFT-to-SKINNY X2 result as the current cross-cipher boundary. Do not
add an adapter, resize parameters, regenerate data, unfreeze the backbone or
launch remote scale as a readiness repair.

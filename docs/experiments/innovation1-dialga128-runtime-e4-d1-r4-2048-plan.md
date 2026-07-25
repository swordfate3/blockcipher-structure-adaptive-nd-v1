# Innovation 1 Dialga-128 Runtime-E4 D1 R4 Plan

Date: 2026-07-25

## Status

```text
implementation = paper vectors and runtime equivalence passed
training = completed and plan-aligned
gate = pass; both seeds satisfy the absolute and both attribution margins
decision = innovation1_dialga_runtime_e4_d1_two_seed_supported
scale = local diagnostic, not formal evidence
remote_scale = prohibited at D1
```

## Research Question

Can the existing cipher-name-free Runtime-E4 network consume Dialga-128's
externally supplied non-contiguous S-box cells and heterogeneous GF(2) linear
topology, with correct topology outperforming matched corrupted and no-topology
controls?

The one changed variable is the runtime structure relation. Cipher data,
network parameter shapes, optimizer, keys, seeds, epochs and checkpoint rule
are identical across roles.

## Correctness Prerequisite

The Dialga implementation must pass before training:

- four published 16/20-round ciphertext vectors;
- all sixteen published per-round Dialga-128-dagger states;
- exact inverses for SubCell, every linear layer and every round function;
- exact runtime/non-contiguous-cell SubCell equivalence;
- exact runtime/native GF(2) equivalence for all four round types;
- descriptor/factory equivalence, cell relabeling equivariance, 128-bit forward,
  and strict state-dict reuse from the same 64-bit Runtime-E4 geometry.

The completed implementation gate is `32 passed` in
`tests/test_dialga128.py`. This is correctness/readiness evidence, not AUC
evidence.

## Frozen Panel

Three roles are repeated at seeds 0 and 1:

| Role | Model key | Runtime relation | Purpose |
| --- | --- | --- | --- |
| correct | `runtime_spn_e4_equivariant_true` | exact Dialga rounds 2 and 3 | candidate |
| corrupted | `runtime_spn_e4_equivariant_corrupted` | deterministic source-bit corruption | wrong-topology control |
| no-topology | `runtime_spn_e4_equivariant_independent` | no inverse topology, identity cell adjacency | baseline/control |

Frozen protocol:

```text
cipher                    = Dialga-128 20-round parent, prefix r4
tweak                     = fixed zero, 128 bits
runtime descriptor        = configs/runtime/spn/dialga128.json
runtime window            = round_start 2, rounds 2
runtime round types       = R2 and R3
round window mode         = recurrent_window
cell input                = state_triplet
S-box context             = edge_gate
input difference          = 0x40 (generic readiness seed, not literature-backed)
pairs_per_sample          = 4 independent pairs
samples_per_class         = 2048
validation                = 1024/class through the standard runner
seeds                     = 0, 1
epochs                    = 10
loss                      = MSE
optimizer                 = Adam, lr 1e-4, weight decay 1e-5
checkpoint                = best validation AUC, restored
negative definition       = encrypted random plaintext pairs
train key                 = 0x00 repeated 32 bytes
validation key            = 0x11 repeated 32 bytes
execution                 = local CPU diagnostic with disk-backed cache
```

Plan CSV:

```text
configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e4_d1_r4_2048_seed0_seed1.csv
```

## Preregistered Gate

Protocol validity requires exactly six result rows, 60 epoch-history rows,
equal parameter geometry, 4096 train rows and 2048 validation rows per role,
strict encrypted-random-plaintext negatives, the frozen keys and difference,
the exact descriptor hash/window metadata, disk-backed datasets and restored
best-AUC checkpoints.

D1 advances only if both seeds satisfy:

```text
correct AUC >= 0.520
correct - corrupted AUC >= +0.005
correct - no-topology AUC >= +0.005
```

This threshold tests whether Dialga's correct runtime topology provides a
replicated local inductive-bias advantage. It does not establish a Dialga
attack, high-round distinguisher, formal cross-cipher result or SOTA claim.

## Execution

```bash
RUN_ID=i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725
RUN_ROOT=outputs/local_diagnostic/${RUN_ID}

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e4_d1_r4_2048_seed0_seed1.csv \
  --device cpu \
  --dataset-cache-root "${RUN_ROOT}/cache" \
  --dataset-cache-chunk-size 1024 \
  --dataset-cache-workers 1 \
  --checkpoint-output-dir "${RUN_ROOT}/checkpoints" \
  --progress-output "${RUN_ROOT}/progress.jsonl" \
  --output "${RUN_ROOT}/results.jsonl"
```

After training: validate six rows, apply the frozen D1 gate, render a Chinese
validation-only SVG, run `visual-qa-redraw`, refresh both recent-result indexes,
and document the evidence-backed next action.

## Decision Routes

- If D1 passes, do not scale. Freeze both correct best checkpoints and run a
  same-checkpoint structure-swap audit before any new training.
- If correct AUC is useful but controls tie, classify the signal as data/model
  evidence without topology attribution and redesign locally.
- If all roles remain near chance, screen only the input difference at a tiny
  fixed budget; do not change the network and difference simultaneously.
- If one seed passes and one fails, hold for instability and do not average the
  failure away.
- Do not add DDT/trail/partial-decryption features or move this `2048/class`
  panel to the remote GPU.

## Recommended Next Action

Freeze the two `correct` best checkpoints and run an inference-only,
same-checkpoint structure-swap audit. For each seed, keep the exact validation
features, labels, weights and head fixed, then evaluate the checkpoint with the
correct, corrupted and no-topology Dialga runtime structures. This is the
required next attribution gate before any new training or scale increase.

Do not increase samples, pairs, epochs or rounds, and do not move Dialga to the
remote GPU until this audit shows that changing only the inference-time
structure changes the predictions and preserves the correct-structure AUC
advantage in both seeds.

## Completed D1 Result

Run:

```text
run_id = i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725
status = pass
decision = innovation1_dialga_runtime_e4_d1_two_seed_supported
result_rows = 6
history_rows = 60
```

Validation AUC and preregistered margins:

| Seed | Correct | Corrupted | No topology | Correct - corrupted | Correct - no topology |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.958417 | 0.936310 | 0.497210 | +0.022107 | +0.461207 |
| 1 | 0.958679 | 0.937817 | 0.503406 | +0.020863 | +0.455274 |

Both seeds exceeded the frozen `0.520` correct-AUC floor and both `+0.005`
control margins. Protocol validation also passed: the rows match the frozen
CSV, all roles use equal parameter geometry and the same data/training
protocol, negatives are encrypted random plaintext pairs, datasets are disk
backed, and every reported AUC is the restored best validation checkpoint.

The strong absolute AUC is evidence that Runtime-E4 can consume this Dialga
prefix-r4 input and the between-role margins are replicated training-time
attribution evidence. The corrupted role also remains strong, so D1 alone does
not prove that the frozen checkpoints functionally depend on the exact runtime
structure. The same-checkpoint swap above is therefore mandatory.

Claim scope remains limited to a Dialga-128 prefix-r4, two-seed,
`2048/class` local diagnostic. This is not formal scale, a Dialga attack, a
paper reproduction, SOTA evidence or a universal-SPN result.

Artifacts:

```text
outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725/results.jsonl
outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725/history.csv
outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725/gate.json
outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725/validation.json
outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725/curves.svg
```

The final SVG passed rendered-pixel visual QA after the legend and summary
labels were changed to identify both the seed and the correct/corrupted/no-
topology role. No overlap, clipping, ambiguous title, unreadable curve group,
misleading axis or incomplete legend remained in the inspected output.

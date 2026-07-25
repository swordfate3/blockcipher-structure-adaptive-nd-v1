# Innovation 1 Dialga-128 Runtime-E4 D3 R5 Adjacent-Window Plan

Date: 2026-07-25

## Status

```text
question = preregistered
training = completed and plan-aligned after a documented seed1 resume
gate = hold; both seeds fail the absolute and topology-control margins
decision = innovation1_dialga_runtime_e4_d3_adjacent_window_not_replicated
scale = local diagnostic, not formal evidence
remote_scale = prohibited
```

## Research Question

Does the Dialga Runtime-E4 topology advantage from D1/D2 replicate when the
same model and data budget move to the next valid two-round window?

D1 trained on Dialga-128 prefix r4 and supplied runtime rounds 2/3. D2 then
showed that the frozen D1 checkpoints functionally used the supplied topology.
D3 changes exactly one research variable: the encryption prefix and matching
runtime window advance by one round.

```text
D1/D2 anchor: prefix r4, runtime rounds 2/3, round_start 2
D3 candidate: prefix r5, runtime rounds 3/4, round_start 3
```

The architecture, difference, data, keys, seeds, optimizer, epochs and all
three topology roles remain fixed. D3 therefore tests cross-window replication,
not extra capacity or extra data.

## Same-Budget Anchor And Controls

The strongest same-protocol anchor is completed D1:

| Seed | Correct | Corrupted | No topology | Correct - corrupted | Correct - no topology |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.958417 | 0.936310 | 0.497210 | +0.022107 | +0.461207 |
| 1 | 0.958679 | 0.937817 | 0.503406 | +0.020863 | +0.455274 |

D3 repeats the same three roles at seeds 0 and 1:

| Role | Model key | Runtime relation | Purpose |
| --- | --- | --- | --- |
| correct | `runtime_spn_e4_equivariant_true` | exact Dialga rounds 3/4 | candidate |
| corrupted | `runtime_spn_e4_equivariant_corrupted` | deterministic wrong GF(2) topology | attribution control |
| no-topology | `runtime_spn_e4_equivariant_independent` | identity cell adjacency, no inverse topology | baseline |

## Frozen Protocol

```text
cipher                    = Dialga-128 20-round parent, prefix r5
tweak                     = fixed zero, 128 bits
runtime descriptor        = configs/runtime/spn/dialga128.json
runtime window            = round_start 3, rounds 2
runtime round types       = R3 and R4
architecture              = RuntimeE4, unchanged from D1
round_window_mode         = recurrent_window
cell_input_mode           = state_triplet
sbox_context_mode         = edge_gate
processor_steps           = 2
pair_embedding_dim        = 128
dropout                   = 0.0
input difference          = 0x40
pairs_per_sample          = 4 independent pairs
samples_per_class         = 2048
train rows                = 4096 per role
validation rows           = 2048 per role, 1024/class
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
configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e4_d3_r5_2048_seed0_seed1.csv
```

This `2048/class` run is a local diagnostic. It is not formal training, a
Dialga attack, paper-scale evidence, SOTA evidence or a universal-SPN result.

## Readiness And Frozen Gate

Before training, the parsed six-row plan must show exactly two seeds and three
roles, prefix r5, runtime `round_start=3`, two loaded heterogeneous transitions,
equal RuntimeE4 parameter geometry, strict encrypted-random-plaintext
negatives, the frozen 256-bit keys, and disk-cache-capable standard-runner
semantics. Existing Dialga public-vector and runtime-equivalence tests must
remain green.

Protocol validity after training requires exactly six result rows and 60 epoch
history rows, 4096 train rows and 2048 validation rows per role, equal parameter
geometry, the exact descriptor/window metadata, disk-backed train/validation
datasets and restored best-AUC checkpoints.

D3 advances only if both seeds independently satisfy:

```text
correct AUC >= 0.520
correct - corrupted AUC >= +0.005
correct - no-topology AUC >= +0.005
```

Do not average away a failing seed. Passing means only that the Dialga topology
advantage replicated on one adjacent window at local diagnostic scale.

## Execution

```bash
RUN_ID=i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725
RUN_ROOT=outputs/local_diagnostic/${RUN_ID}

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e4_d3_r5_2048_seed0_seed1.csv \
  --device cpu \
  --dataset-cache-root "${RUN_ROOT}/cache" \
  --dataset-cache-chunk-size 1024 \
  --dataset-cache-workers 1 \
  --checkpoint-output-dir "${RUN_ROOT}/checkpoints" \
  --progress-output "${RUN_ROOT}/progress.jsonl" \
  --output "${RUN_ROOT}/results.jsonl"

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e4_d3_r5_2048_seed0_seed1.csv \
  --results "${RUN_ROOT}/results.jsonl" \
  --expected-rows 6

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-runtime-spn-dialga-d3 \
  --run-id "${RUN_ID}" \
  --run-root "${RUN_ROOT}"
```

After adjudication, render a Chinese validation-only SVG, inspect the rendered
pixels with `visual-qa-redraw`, refresh both recent-result indexes, and update
this record with metrics, claim scope and the evidence-backed next action.

## Decision Routes

- If both seeds pass, freeze both correct D3 best checkpoints and run an
  inference-only same-checkpoint correct/corrupted/no-topology swap on the exact
  D3 validation features. Do not scale before that audit.
- If either seed misses the AUC floor or either control margin, classify the
  D1/D2 effect as not yet cross-window stable. Stop Dialga scale advancement and
  diagnose locally without changing RuntimeE4 and data scale simultaneously.
- If the protocol gate fails, repair only the failed frozen condition and rerun
  the invalid row; do not interpret the AUC values.
- Do not increase samples, pairs, epochs or seeds; do not add DDT, trail,
  partial-decryption or guessed-key features; do not launch a remote GPU run.

## Preregistered Next Action

The next action is determined by the frozen gate rather than the absolute AUC
alone. A pass requires the D3 same-checkpoint structure-swap audit because
separately trained controls cannot by themselves prove inference-time topology
use. A hold stops mechanical Dialga scaling because the same architecture and
budget would then have failed its adjacent-window replication test.

Artifacts:

```text
outputs/local_diagnostic/i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725/results.jsonl
outputs/local_diagnostic/i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725/history.csv
outputs/local_diagnostic/i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725/gate.json
outputs/local_diagnostic/i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725/validation.json
outputs/local_diagnostic/i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725/curves.svg
outputs/local_diagnostic/i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725/visual_qa_passed.marker
```

## Completed D3 Result

Run:

```text
run_id = i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725
status = hold
decision = innovation1_dialga_runtime_e4_d3_adjacent_window_not_replicated
result_rows = 6
history_rows = 60
```

Validation AUC and preregistered margins:

| Seed | Correct | Corrupted | No topology | Correct - corrupted | Correct - no topology |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.507330 | 0.504382 | 0.544160 | +0.002948 | -0.036830 |
| 1 | 0.493330 | 0.528280 | 0.519389 | -0.034950 | -0.026059 |

All protocol checks passed: the merged result contains exactly the frozen six
rows, both seeds and all three roles; the data/training protocol and parameter
geometry match; prefix r5 uses runtime `round_start=3` with two heterogeneous
transitions; negatives are encrypted random plaintext pairs; datasets are disk
backed; and every result restores the best validation-AUC checkpoint with ten
complete history rows.

The first session completed all three seed0 rows but was interrupted during an
incomplete seed1 row. Seed1 was restarted from its exact three frozen CSV rows
and reused the parameter-matched disk cache. The original seed0 and restarted
seed1 JSONL files were preserved and merged only after verifying six unique
`(seed, model)` keys and exact agreement with the full plan. Provenance and
SHA-256 values are recorded in `resume_manifest.json`; `validate-results`
subsequently returned `status=pass`, six expected rows and no errors.

Research-wise, both seeds missed the `0.520` correct-AUC floor. Seed0 also
missed the corrupted margin and trailed no-topology by `0.036830`; seed1
trailed corrupted by `0.034950` and no-topology by `0.026059`. Thus the strong
D1/D2 prefix-r4 effect did not replicate on the adjacent prefix-r5 window. The
two controls retaining occasional above-chance AUC does not rescue the topology
claim because the correct structure is not the winning condition.

The rendered Chinese validation-only SVG passed `visual-qa-redraw` at 1600 px:
no text overlap, clipping, missing glyphs, ambiguous legend, unreadable curve
group, misleading AUC range or incomplete summary table remained.

Claim scope is limited to a Dialga-128 prefix-r5, two-seed, `2048/class` local
diagnostic. This does not erase D1/D2, but narrows them to window-specific
evidence. It is not formal scale, a Dialga attack, paper reproduction, SOTA
evidence or a universal-SPN result.

## Evidence-Backed Next Action

Do not train another D3 variant and do not increase data. Preregister D4 as a
training-free `2 x 2` factorial audit using each seed's frozen D1 correct
checkpoint:

| Validation data | Runtime window | Role |
| --- | --- | --- |
| D1 prefix-r4 | `round_start=2` | exact D1/D2 anchor |
| D1 prefix-r4 | `round_start=3` | structure-window-only intervention |
| D3 prefix-r5 | `round_start=2` | data-depth-only intervention |
| D3 prefix-r5 | `round_start=3` | coupled adjacent-window condition |

Keep the checkpoint, labels, keys, pair organization and evaluation code fixed
within each seed. Perform no optimization, checkpoint selection or new data
generation. Require exact D1 anchor reproduction, strict state-dict loading,
shared feature/label hashes within each data split and nonzero probability
changes for the structure intervention.

Decision unlocked by D4:

- If changing only D1 data from prefix r4 to r5 collapses AUC under both
  structures, treat the current `0x40` differential data route as the limiting
  factor; run only a tiny preregistered input-difference screen before any
  architecture change.
- If D1 prefix-r4 data collapses when only `round_start` changes, or D3 data
  recovers under the old window but not the matching new window, prioritize a
  shared residual/gated topology processor that cannot erase the independent
  representation; compare it against current RuntimeE4 and no-topology at the
  same local budget.
- If both factors independently degrade AUC, stop Dialga scale advancement and
  address data and topology in separate experiments, one variable at a time.

D4 remains local and inference-only. Remote GPU use, larger samples, more
epochs, additional seeds, DDT/trail/partial-decryption features and Dialga
attack claims are explicitly blocked.

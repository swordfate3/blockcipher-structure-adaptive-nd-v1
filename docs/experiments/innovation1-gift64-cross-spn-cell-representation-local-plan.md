# Innovation 1 GIFT-64 Cross-SPN Cell Representation Local Plan

**Date:** 2026-07-06

**Status:** local diagnostic planned / no remote launch

## Question

The strongest completed Innovation 1 SPN evidence remains the PRESENT r7
InvP/P-layer aligned route:

```text
present_nibble_invp_only_spn_only
two_seed_1000000_class_positive_with_attribution_control
```

Recent local branches do not justify simply adding more near-neighbor PRESENT
models:

```text
near-neighbor neural ensemble = weak positive below gate
r8 learned pair-mixer = hold
SGP / deterministic InvP aggregate statistics = hold
```

This diagnostic asks a narrower representation question:

```text
Does a generic inverse-permutation SPN-aligned ciphertext-pair representation
show a local advantage on GIFT-64 over same-budget raw and C||C'||DeltaC
controls?
```

The goal is not to claim a GIFT result. The goal is to test whether SPN-aligned
representation is a PRESENT-specific artifact or a plausible cross-SPN route
that could later become a genuinely diverse expert source.

## Why This Comes Before A Larger Ensemble

The user's diverse-network idea remains valid, but the current pool is too
near-neighbor-heavy. Combining raw PRESENT MCND, PRESENT InvP-only, and PRESENT
DDT graph variants mainly averages related views of the same task. A useful
expert pool needs a non-neighbor candidate with its own weak-positive signal and
low error overlap.

GIFT-64 is a small local probe for that requirement:

- It is SPN-like, but its bit permutation differs from PRESENT.
- The repository already supports `GIFT-64`.
- The generic `ciphertext_pair_xor_spn_aligned_bits` encoder uses the cipher's
  own `inverse_permutation_layer`, not PRESENT hardcoding.
- The generic `spn_token_mixer_pairset` model can consume all three planned
  GIFT feature widths.

## Protocol

Config:

```text
configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_cell_repr_local.csv
```

Fixed benchmark fields:

| Field | Value |
|---|---|
| Cipher | `GIFT-64` |
| Rounds | `6` |
| Seed | `0` |
| Samples per class | `2048` |
| Pairs per sample | `4` |
| Negative mode | `encrypted_random_plaintexts` |
| Sample structure | `independent_pairs` |
| Difference profile | `gift64_shen2024_spn_screen`, member `0` |
| Train key | `0x00000000000000000000000000000000` |
| Validation key | `0x11111111111111111111111111111111` |

Rows:

| Row | Model | Feature encoding | Purpose |
|---:|---|---|---|
| 0 | `mlp` | `ciphertext_pair_bits` | raw capacity anchor |
| 1 | `spn_token_mixer_pairset` | `ciphertext_pair_bits` | token-mixer capacity control |
| 2 | `spn_token_mixer_pairset` | `ciphertext_pair_xor_bits` | generic differential representation control |
| 3 | `spn_token_mixer_pairset` | `ciphertext_pair_xor_spn_aligned_bits` | generic SPN inverse-permutation candidate |

Only the model/feature route changes. Labels, negative sampling, validation key,
sample structure, input difference, metric computation, and plan-alignment logic
are unchanged.

## Gate

This is a local diagnostic only. Treat outcomes as route-selection evidence:

| Result | Decision |
|---|---|
| SPN-aligned row beats `ciphertext_pair_xor_bits` token mixer by `>= +0.01` AUC and beats raw token mixer | keep as cross-SPN representation candidate for a larger local repeat |
| SPN-aligned row improves by `0 < delta < +0.01` AUC | weak positive; repeat locally before any remote plan |
| Raw token mixer wins | model capacity, not SPN representation; do not scale aligned route |
| `ciphertext_pair_xor_bits` wins | generic DeltaC representation is enough; do not claim inverse-permutation gain |
| All rows near random | hold GIFT cross-SPN route and return to PRESENT data/difference construction |

No remote launch should be created from this diagnostic alone.

## Claim Scope

This is not formal training, not a publication-scale GIFT result, not a
breakthrough claim, and not ensemble evidence. It is a small local screen for a
possible non-neighbor expert family.

## Planned Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_cell_repr_local.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 16 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --output outputs/local_smoke/i1_gift64_cross_spn_cell_repr_2048/results.jsonl \
  --progress-output outputs/local_smoke/i1_gift64_cross_spn_cell_repr_2048/progress.jsonl
```

Planned validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_cell_repr_local.csv \
  --results outputs/local_smoke/i1_gift64_cross_spn_cell_repr_2048/results.jsonl \
  --expected-rows 4
```

## 2026-07-06 Local Result

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_cell_repr_local.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 16 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --output outputs/local_smoke/i1_gift64_cross_spn_cell_repr_2048/results.jsonl \
  --progress-output outputs/local_smoke/i1_gift64_cross_spn_cell_repr_2048/progress.jsonl
```

Artifacts:

```text
outputs/local_smoke/i1_gift64_cross_spn_cell_repr_2048/results.jsonl
outputs/local_smoke/i1_gift64_cross_spn_cell_repr_2048/progress.jsonl
outputs/local_smoke/i1_gift64_cross_spn_cell_repr_2048/curves.svg
outputs/local_smoke/i1_gift64_cross_spn_cell_repr_2048/history.csv
```

Plan-alignment verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_cell_repr_local.csv \
  --results outputs/local_smoke/i1_gift64_cross_spn_cell_repr_2048/results.jsonl \
  --expected-rows 4
```

Result:

```text
status = pass
result_rows = 4
field_mismatches = []
```

Metrics:

| Row | Model | Feature encoding | AUC | Accuracy | Calibrated accuracy | Loss | Best epoch |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `mlp` | `ciphertext_pair_bits` | `0.5167593955993652` | `0.51318359375` | `0.52099609375` | `0.6928995829075575` | `2` |
| 1 | `spn_token_mixer_pairset` | `ciphertext_pair_bits` | `0.5172939300537109` | `0.50390625` | `0.52587890625` | `0.6930988002568483` | `3` |
| 2 | `spn_token_mixer_pairset` | `ciphertext_pair_xor_bits` | `0.4951457977294922` | `0.4921875` | `0.5087890625` | `0.6937781348824501` | `3` |
| 3 | `spn_token_mixer_pairset` | `ciphertext_pair_xor_spn_aligned_bits` | `0.5226421356201172` | `0.5087890625` | `0.5263671875` | `0.6927917245775461` | `3` |

Deltas:

```text
SPN-aligned - C||C'||DeltaC token mixer AUC = +0.027496337890625
SPN-aligned - raw token mixer AUC = +0.00534820556640625
SPN-aligned - raw MLP AUC = +0.005882740020751953
raw token mixer - raw MLP AUC = +0.0005345344543457031
```

Decision:

```text
weak_cross_spn_aligned_positive_keep_for_local_repeat
no_remote_launch_from_current_gift64_diagnostic
do_not_use_as_ensemble_expert_until_repeat_or_medium_gate
```

Interpretation:

The generic SPN-aligned representation is the best row in this small GIFT-64
diagnostic and clears the `+0.01` gate against the `C||C'||DeltaC` control.
It also beats the same model on raw pairs, but only by about `+0.0053` AUC.

This supports one narrow next step: a local repeat with more seeds or a slightly
larger diagnostic budget. It does not support a remote launch, a GIFT result
claim, or immediate inclusion as a qualified diverse-ensemble expert.

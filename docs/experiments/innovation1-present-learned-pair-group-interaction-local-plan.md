# Innovation 1 PRESENT Learned Pair/Group Interaction Local Plan

**Date:** 2026-07-06

**Status:** local diagnostic planned / no remote launch

## Question

The SGP and deterministic InvP aggregate-stat audits did not produce a stable
projection or statistic route:

```text
sgp_stable_axis_hold
sgp_grouped_axis_hold
invp_global_stats_hold
invp_group_distribution_hold
```

However, the strongest completed Innovation 1 SPN evidence is still the r7
InvP/P-layer aligned route:

```text
present_nibble_invp_only_spn_only
two_seed_1000000_class_positive_with_attribution_control
```

This local diagnostic asks:

```text
Can a learned pair/group interaction over the same InvP representation show
any r8 local signal beyond the simple InvP-only same-input anchor?
```

## Protocol

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_learned_pair_group_interaction_r8_local.csv
```

Fixed benchmark fields:

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `8` |
| Seed | `0` |
| Samples per class | `512` |
| Pairs per sample | `16` |
| Feature encoding | `ciphertext_pair_bits` |
| Negative mode | `encrypted_random_plaintexts` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Difference profile | `present_zhang_wang2022_mcnd`, member `0` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |

Rows:

| Row | Model | Purpose |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | same-input InvP-only anchor |
| 1 | `present_nibble_invp_pair_consistency_spn_only` | pair-level evidence pooling candidate |
| 2 | `present_nibble_invp_pair_mixer_consistency_spn_only` | learned cross-pair interaction candidate |

Only the pair/group interaction path changes. Labels, negative sampling,
validation key, sample structure, feature encoding, and metrics are unchanged.

## Gate

This is a small local diagnostic. Treat outcomes as route-selection evidence
only:

| Result | Decision |
|---|---|
| Pair-mixer or pair-consistency beats InvP-only by `>= +0.01` AUC | keep learned interaction for a larger local repeat |
| Best interaction row beats InvP-only by `0 < delta < +0.01` AUC | weak positive; repeat locally before any remote plan |
| Both interaction rows `<=` InvP-only | stop this r8 learned interaction branch for now |
| All rows near random | confirms r8 fixed-difference bottleneck; return to representation/data search |

No remote launch should be created from this diagnostic alone.

## Claim Scope

This is not formal training, not a publication-scale result, not an ensemble
result, and not a breakthrough claim. It is a local test of whether learned
pair/group interaction deserves another small experiment slot after handwritten
InvP aggregation failed.

## 2026-07-06 Local Result

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_learned_pair_group_interaction_r8_local.csv \
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
  --output outputs/local_smoke/i1_present_r8_learned_pair_group_interaction_512/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_learned_pair_group_interaction_512/progress.jsonl
```

Artifacts:

```text
outputs/local_smoke/i1_present_r8_learned_pair_group_interaction_512/results.jsonl
outputs/local_smoke/i1_present_r8_learned_pair_group_interaction_512/progress.jsonl
outputs/local_smoke/i1_present_r8_learned_pair_group_interaction_512/curves.svg
outputs/local_smoke/i1_present_r8_learned_pair_group_interaction_512/history.csv
```

Plan-alignment verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_learned_pair_group_interaction_r8_local.csv \
  --results outputs/local_smoke/i1_present_r8_learned_pair_group_interaction_512/results.jsonl \
  --expected-rows 3
```

Result:

```text
status = pass
result_rows = 3
field_mismatches = []
```

Metrics:

| Model | AUC | Accuracy | Calibrated accuracy | Loss | Best epoch |
|---|---:|---:|---:|---:|---:|
| `present_nibble_invp_only_spn_only` | `0.51275634765625` | `0.5078125` | `0.5390625` | `0.693090058863163` | `1` |
| `present_nibble_invp_pair_consistency_spn_only` | `0.5184173583984375` | `0.5` | `0.53125` | `0.6935298591852188` | `1` |
| `present_nibble_invp_pair_mixer_consistency_spn_only` | `0.5105438232421875` | `0.5234375` | `0.529296875` | `0.6930325329303741` | `3` |

Deltas:

```text
pair-consistency - InvP-only AUC = +0.0056610107421875
pair-mixer - InvP-only AUC = -0.0022125244140625
pair-mixer - pair-consistency AUC = -0.00787353515625
```

Decision:

```text
weak_pair_consistency_positive_below_local_gate
stop_pair_mixer_remote_launch_from_current_evidence
do_not_launch_prepared_262k_pair_mixer_package
```

Interpretation:

The simple pair-consistency row has a weak local AUC edge over InvP-only, but it
does not meet the `+0.01` local keep gate. The learned cross-pair mixer is below
both InvP-only and pair-consistency in this diagnostic. This does not support
remote launch or a larger pair-mixer matrix under the current r8 protocol.

Next action:

```text
Do not expand r8 pair-mixer now. Keep pair-consistency only as a weak local
diagnostic. Move the next experiment slot toward a different SPN
representation/data route, such as cross-SPN cell-representation sanity or a
new controlled data construction.
```

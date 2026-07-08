# Innovation 1 PRESENT r8 Leakage Audit Plan

**Date:** 2026-07-08

**Status:** planned local 512/class seed0 audit

## Question

The PRESENT r8 trail-position route reaches very high AUC under the
matched-negative integral protocol. This audit asks whether that strength
survives three direct checks:

```text
1. strict independent random-plaintext negatives
2. per-row key rotation
3. weaker feature encodings before InvS/DDT beamstats/trail-position
```

## Fixed Scope

This is a local diagnostic only, not formal SPN/PRESENT evidence and not a
publication claim.

```text
cipher = PRESENT-80
rounds = 8
seed = 0
samples_per_class = 512
pairs_per_sample = 16
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
difference_profile = present_zhang_wang2022_mcnd
difference_member = 0
integral_active_nibble = 0
negative_mode = encrypted_random_plaintexts
```

Plan:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_leakage_audit_512_seed0.csv
```

## Rows

| Row | Purpose | Sample structure | Key rotation | Feature | Model |
|---:|---|---|---:|---|---|
| 0 | matched-negative global anchor | `plaintext_integral_nibble_difference_matched_negative` | `0` | full beamstats | `present_pairset_global_stats` |
| 1 | matched-negative trail anchor | `plaintext_integral_nibble_difference_matched_negative` | `0` | full beamstats | `present_trail_position_stats_pairset` |
| 2 | strict-negative global | `plaintext_integral_nibble_strict_random_negative` | `0` | full beamstats | `present_pairset_global_stats` |
| 3 | strict-negative trail | `plaintext_integral_nibble_strict_random_negative` | `0` | full beamstats | `present_trail_position_stats_pairset` |
| 4 | strict-negative global per-row key | `plaintext_integral_nibble_strict_random_negative` | `1` | full beamstats | `present_pairset_global_stats` |
| 5 | strict-negative trail per-row key | `plaintext_integral_nibble_strict_random_negative` | `1` | full beamstats | `present_trail_position_stats_pairset` |
| 6 | weak xor ablation | matched-negative | `0` | `ciphertext_xor_bits` | `present_pairset_global_stats` |
| 7 | P-layer aligned ablation | matched-negative | `0` | `ciphertext_xor_spn_paligned_bits` | `present_pairset_global_stats` |
| 8 | InvS/P-layer ablation | matched-negative | `0` | `present_pair_xor_paligned_sinv_cell_matrix_bits` | `present_pairset_global_stats` |

## Gate

Interpretation rules:

```text
If strict-negative trail stays very high:
  the high signal is not mainly caused by matched-negative construction.

If strict-negative trail collapses:
  the previous result is tied to the matched-negative integral protocol.

If per-row key rotation collapses while strict fixed-key survives:
  fixed-key dependence is a leading risk.

If weak xor / paligned / sinv ablations are already high:
  the signal starts before DDT beamstats and trail-position.

If only full beamstats trail-position is high:
  the signal is mainly in public DDT/trail-position derived structure.
```

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_leakage_audit_512_seed0.csv \
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
  --output outputs/local_smoke/i1_present_r8_leakage_audit_512_seed0/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_leakage_audit_512_seed0/progress.jsonl
```

Validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_leakage_audit_512_seed0.csv \
  --results outputs/local_smoke/i1_present_r8_leakage_audit_512_seed0/results.jsonl \
  --expected-rows 9
```

## 512/Class Seed0 Result

Run artifacts:

```text
results = outputs/local_smoke/i1_present_r8_leakage_audit_512_seed0/results.jsonl
progress = outputs/local_smoke/i1_present_r8_leakage_audit_512_seed0/progress.jsonl
curves = outputs/local_smoke/i1_present_r8_leakage_audit_512_seed0/curves.svg
history = outputs/local_smoke/i1_present_r8_leakage_audit_512_seed0/history.csv
cache = outputs/local_cache/i1_present_r8_leakage_audit_512_seed0
```

Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_leakage_audit_512_seed0.csv \
  --results outputs/local_smoke/i1_present_r8_leakage_audit_512_seed0/results.jsonl \
  --expected-rows 9

status = pass
field_mismatches = []
```

Validation note:

```text
The first validation attempt exposed a result-alignment bug: protocol audit
rows that differed only by sample_structure/key_rotation_interval were treated
as duplicate keys. The validator was fixed to include pairs_per_sample,
negative_mode, sample_structure, integral_active_nibble, and
key_rotation_interval in its alignment key.
```

Metrics:

| Row | Protocol | Key rotation | Feature | Model | AUC | Accuracy | Calibrated accuracy |
|---:|---|---:|---|---|---:|---:|---:|
| 0 | matched-negative | `0` | full beamstats | global | `0.762863159` | `0.529297` | `0.701172` |
| 1 | matched-negative | `0` | full beamstats | trail-position | `0.977752686` | `0.611328` | `0.925781` |
| 2 | strict random negative | `0` | full beamstats | global | `0.837478638` | `0.578125` | `0.761719` |
| 3 | strict random negative | `0` | full beamstats | trail-position | `0.999984741` | `0.599609` | `0.998047` |
| 4 | strict random negative | `1` | full beamstats | global | `0.849136353` | `0.582031` | `0.783203` |
| 5 | strict random negative | `1` | full beamstats | trail-position | `0.999969482` | `0.591797` | `0.998047` |
| 6 | matched-negative | `0` | ciphertext xor | global | `0.000000000` | `0.500000` | `0.500000` |
| 7 | matched-negative | `0` | P-layer aligned xor | global | `0.523498535` | `0.525391` | `0.542969` |
| 8 | matched-negative | `0` | P-layer + InvS | global | `0.569618225` | `0.529297` | `0.558594` |

Interpretation:

```text
strict random-negative check:
  The trail-position candidate does not collapse. It improves from
  0.977752686 under matched-negative to 0.999984741 under strict independent
  random-plaintext negatives at this small 512/class seed0 diagnostic.

key-rotation check:
  Per-row rotating keys also do not collapse the strict-negative full
  beamstats trail-position result: AUC remains 0.999969482.

feature-ablation check:
  P-layer aligned xor and P-layer+InvS global-stat rows remain weak
  (0.5235 and 0.5696 AUC), so the high positive-orientation signal does not
  appear from those weaker derived views alone. The ciphertext-xor row reports
  AUC 0.0, which is a direction pathology rather than clean chance evidence;
  it should be followed by an oriented-AUC or deterministic sign-control audit
  before being interpreted.
```

Decision:

```text
support_followup_strict_negative_keyrot_medium
```

Claim scope:

```text
This is a 512/class seed0 local audit only. It reduces the likelihood that the
trail-position result is solely caused by matched-negative construction or
fixed train/validation keys, but it does not prove a formal PRESENT r8 claim.
The next useful audit is a lean 2048/class seed0+seed1 repeat with strict
negative, per-row key rotation, full beamstats trail-position/global rows, and
an oriented-AUC check for ciphertext_xor_bits.
```

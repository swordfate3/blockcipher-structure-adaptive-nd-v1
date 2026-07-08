# Innovation 1 PRESENT r8 Leakage Audit Plan

**Date:** 2026-07-08

**Status:** local diagnostics completed through same-difference 2048/class seed0+seed1 audit

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

## 2048/Class Seed0+Seed1 Follow-Up Plan

Purpose:

```text
Repeat the strict random-negative and per-row key-rotation checks at 2048/class
on two seeds, and run the weak feature ablations under strict negatives rather
than matched negatives.
```

Plan:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_leakage_audit_2048_seed0_seed1.csv
```

Rows per seed:

| Role | Sample structure | Key rotation | Feature | Model |
|---|---|---:|---|---|
| strict full global | `plaintext_integral_nibble_strict_random_negative` | `0` | full beamstats | `present_pairset_global_stats` |
| strict full trail | `plaintext_integral_nibble_strict_random_negative` | `0` | full beamstats | `present_trail_position_stats_pairset` |
| strict full global per-row key | `plaintext_integral_nibble_strict_random_negative` | `1` | full beamstats | `present_pairset_global_stats` |
| strict full trail per-row key | `plaintext_integral_nibble_strict_random_negative` | `1` | full beamstats | `present_trail_position_stats_pairset` |
| strict xor global | `plaintext_integral_nibble_strict_random_negative` | `0` | `ciphertext_xor_bits` | `present_pairset_global_stats` |
| strict PAligned global | `plaintext_integral_nibble_strict_random_negative` | `0` | `ciphertext_xor_spn_paligned_bits` | `present_pairset_global_stats` |
| strict InvS global | `plaintext_integral_nibble_strict_random_negative` | `0` | `present_pair_xor_paligned_sinv_cell_matrix_bits` | `present_pairset_global_stats` |

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_leakage_audit_2048_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_leakage_audit_2048_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_leakage_audit_2048_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_leakage_audit_2048_seed0_seed1/progress.jsonl
```

Validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_leakage_audit_2048_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_leakage_audit_2048_seed0_seed1/results.jsonl \
  --expected-rows 14
```

## 2048/Class Seed0+Seed1 Result

Run artifacts:

```text
results = outputs/local_smoke/i1_present_r8_leakage_audit_2048_seed0_seed1/results.jsonl
progress = outputs/local_smoke/i1_present_r8_leakage_audit_2048_seed0_seed1/progress.jsonl
curves = outputs/local_smoke/i1_present_r8_leakage_audit_2048_seed0_seed1/curves.svg
history = outputs/local_smoke/i1_present_r8_leakage_audit_2048_seed0_seed1/history.csv
cache = outputs/local_cache/i1_present_r8_leakage_audit_2048_seed0_seed1
```

Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_leakage_audit_2048_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_leakage_audit_2048_seed0_seed1/results.jsonl \
  --expected-rows 14

status = pass
field_mismatches = []
```

Metrics:

| Seed | Role | Key rotation | Feature | Model | AUC | Oriented AUC | Accuracy | Calibrated accuracy |
|---:|---|---:|---|---|---:|---:|---:|---:|
| 0 | strict full global | `0` | full beamstats | global | `0.969344139` | `0.969344139` | `0.906250` | `0.911133` |
| 0 | strict full trail | `0` | full beamstats | trail-position | `1.000000000` | `1.000000000` | `1.000000` | `1.000000` |
| 0 | strict full global per-row key | `1` | full beamstats | global | `0.969796181` | `0.969796181` | `0.905273` | `0.910156` |
| 0 | strict full trail per-row key | `1` | full beamstats | trail-position | `0.999984741` | `0.999984741` | `0.999512` | `0.999512` |
| 0 | strict xor global | `0` | ciphertext xor | global | `0.000000000` | `1.000000000` | `0.500000` | `0.500000` |
| 0 | strict PAligned global | `0` | P-layer aligned xor | global | `0.661911011` | `0.661911011` | `0.512695` | `0.625488` |
| 0 | strict InvS global | `0` | P-layer + InvS | global | `0.731975079` | `0.731975079` | `0.520508` | `0.678223` |
| 1 | strict full global | `0` | full beamstats | global | `0.968549728` | `0.968549728` | `0.904785` | `0.914551` |
| 1 | strict full trail | `0` | full beamstats | trail-position | `0.999997139` | `0.999997139` | `0.998535` | `0.999023` |
| 1 | strict full global per-row key | `1` | full beamstats | global | `0.976821899` | `0.976821899` | `0.901367` | `0.918945` |
| 1 | strict full trail per-row key | `1` | full beamstats | trail-position | `0.999995232` | `0.999995232` | `0.998047` | `0.999023` |
| 1 | strict xor global | `0` | ciphertext xor | global | `0.000000000` | `1.000000000` | `0.500000` | `0.500000` |
| 1 | strict PAligned global | `0` | P-layer aligned xor | global | `0.614149094` | `0.614149094` | `0.591309` | `0.593262` |
| 1 | strict InvS global | `0` | P-layer + InvS | global | `0.702373028` | `0.702373028` | `0.646484` | `0.649414` |

Interpretation:

```text
strict random-negative + full beamstats:
  The trail-position candidate is stable across both seeds, with AUC
  1.000000000 and 0.999997139. The same-input global control is also very
  strong at about 0.969 on both seeds. This means the strict-negative protocol
  is much easier than the matched-negative protocol for the full feature route.

per-row key rotation:
  Per-row key rotation does not collapse the result. The full trail-position
  rows remain 0.999984741 and 0.999995232, while global controls remain about
  0.970-0.977.

weak feature ablation:
  `ciphertext_xor_bits` reports AUC 0.0 on both seeds, but oriented AUC is 1.0.
  This is not clean chance evidence; it means the learned score ranks labels
  perfectly in the opposite direction. PAligned and InvS views are positive but
  much weaker than full beamstats.
```

Decision:

```text
strict_negative_keyrot_survives_but_protocol_too_easy
```

Claim scope:

```text
This result reduces the likelihood of direct label leakage, fixed-key leakage,
or matched-negative-only artifacts. However, it also shows that strict random
negative is a much easier protocol: even raw ciphertext XOR carries a perfect
sign-flipped ranking under the current trained global-stat row. Therefore this
is not evidence for a standard PRESENT r8 breakthrough. It is evidence that
the current positive structure versus independent-random negative split is
too easy for publication-style claims.
```

Next action:

```text
1. Add an oriented-AUC/sign-control audit for weak-feature rows, especially
   ciphertext_xor_bits.
2. Build a stricter same-difference negative: negative plaintext pairs should
   keep the same input difference or matched output-difference budget while
   breaking the specific trail/integral alignment.
3. Treat matched-negative, not strict random-negative, as the more meaningful
   control family for evaluating trail-position residual value.
4. Do not scale strict random-negative to remote medium runs; it is now a
   diagnostic for leakage/protocol ease, not a candidate benchmark.
```

## Same-Difference Random-Negative Follow-Up

Purpose:

```text
The strict random-negative protocol was too easy because positives used
fixed-difference pairs while negatives used independent random plaintext pairs.
This follow-up keeps the same input difference in both classes:

positive:
  one row is an integral active-nibble set of P, P xor Delta pairs.

negative:
  each pair is also P, P xor Delta, but every P is independently random, so
  the 16-pair integral active-nibble sweep is broken.

This asks whether the model still separates integral/trail structure when the
obvious fixed-difference-vs-random-difference shortcut is removed.
```

Plan:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_same_difference_audit_2048_seed0_seed1.csv
```

Run artifacts:

```text
results = outputs/local_smoke/i1_present_r8_same_difference_audit_2048_seed0_seed1/results.jsonl
progress = outputs/local_smoke/i1_present_r8_same_difference_audit_2048_seed0_seed1/progress.jsonl
curves = outputs/local_smoke/i1_present_r8_same_difference_audit_2048_seed0_seed1/curves.svg
history = outputs/local_smoke/i1_present_r8_same_difference_audit_2048_seed0_seed1/history.csv
cache = outputs/local_cache/i1_present_r8_same_difference_audit_2048_seed0_seed1
```

Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_same_difference_audit_2048_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_same_difference_audit_2048_seed0_seed1/results.jsonl \
  --expected-rows 14

status = pass
field_mismatches = []
```

Metrics:

| Seed | Role | Key rotation | Feature | Model | AUC | Oriented AUC | Accuracy | Calibrated accuracy |
|---:|---|---:|---|---|---:|---:|---:|---:|
| 0 | same-diff full global | `0` | full beamstats | global | `0.971157074` | `0.971157074` | `0.914551` | `0.916504` |
| 0 | same-diff full trail | `0` | full beamstats | trail-position | `1.000000000` | `1.000000000` | `0.999512` | `1.000000` |
| 0 | same-diff full global per-row key | `1` | full beamstats | global | `0.970283508` | `0.970283508` | `0.904297` | `0.911133` |
| 0 | same-diff full trail per-row key | `1` | full beamstats | trail-position | `0.999978065` | `0.999978065` | `0.999023` | `0.999512` |
| 0 | same-diff xor global | `0` | ciphertext xor | global | `0.000000000` | `1.000000000` | `0.500000` | `0.500000` |
| 0 | same-diff PAligned global | `0` | P-layer aligned xor | global | `0.634251118` | `0.634251118` | `0.514648` | `0.603516` |
| 0 | same-diff InvS global | `0` | P-layer + InvS | global | `0.720172882` | `0.720172882` | `0.533203` | `0.670898` |
| 1 | same-diff full global | `0` | full beamstats | global | `0.974644661` | `0.974644661` | `0.909668` | `0.932129` |
| 1 | same-diff full trail | `0` | full beamstats | trail-position | `0.999991417` | `0.999991417` | `0.997559` | `0.998535` |
| 1 | same-diff full global per-row key | `1` | full beamstats | global | `0.978249550` | `0.978249550` | `0.906250` | `0.927246` |
| 1 | same-diff full trail per-row key | `1` | full beamstats | trail-position | `0.999996185` | `0.999996185` | `0.998047` | `0.999023` |
| 1 | same-diff xor global | `0` | ciphertext xor | global | `0.000000000` | `1.000000000` | `0.500000` | `0.500000` |
| 1 | same-diff PAligned global | `0` | P-layer aligned xor | global | `0.600473404` | `0.600473404` | `0.561035` | `0.574219` |
| 1 | same-diff InvS global | `0` | P-layer + InvS | global | `0.694131374` | `0.694131374` | `0.637207` | `0.639160` |

Interpretation:

```text
same-difference control:
  Removing the fixed-difference-vs-random-difference shortcut does not collapse
  the full beamstats route. Trail-position remains almost perfect on both
  seeds, and per-row key rotation still does not collapse it.

weak feature audit:
  The weak rows remain suspicious. ciphertext_xor_bits still gives AUC 0.0
  with oriented AUC 1.0, which means a perfect reversed score ordering exists
  even though the ordinary thresholded accuracy is 0.5. PAligned and InvS are
  not perfect, but remain above chance.

meaning:
  The high score is not explained only by matched-negative construction,
  fixed train/validation key, or fixed-difference-vs-random-difference
  separation. But it is still not clean evidence for a publication-style
  PRESENT r8 distinguisher, because very weak public XOR-derived views still
  expose class structure.
```

Decision:

```text
same_difference_full_signal_survives_but_weak_xor_control_still_suspicious
```

Next action:

```text
Do not launch a larger remote version of this exact full-feature protocol yet.
First add a deterministic / non-neural feature audit that explains the
ciphertext_xor_bits sign-flipped separation. The next useful question is not
"can the network score 0.999 again"; it is "which simple statistic already
separates the labels, and can a protocol variant remove that statistic while
preserving meaningful trail-position residual signal?"
```

## Deterministic Weak-XOR Audit Result

Artifacts:

```text
strict seed0 = outputs/local_audits/i1_present_r8_weak_xor_deterministic_strict_seed0_2048.json
strict seed1 = outputs/local_audits/i1_present_r8_weak_xor_deterministic_strict_seed1_2048.json
same-diff seed0 = outputs/local_audits/i1_present_r8_weak_xor_deterministic_samediff_seed0_2048.json
same-diff seed1 = outputs/local_audits/i1_present_r8_weak_xor_deterministic_samediff_seed1_2048.json
matched seed0 = outputs/local_audits/i1_present_r8_weak_xor_deterministic_matched_seed0_2048.json
matched seed1 = outputs/local_audits/i1_present_r8_weak_xor_deterministic_matched_seed1_2048.json
```

Metrics:

| Protocol | Seed | `left_right_column_sum_l1_mean` AUC | `pair_xor_column_sum_variance` AUC | Decision |
|---|---:|---:|---:|---|
| strict random negative | 0 | `1.000000000` | `0.997581720` | exact multiset shortcut |
| strict random negative | 1 | `1.000000000` | `0.997450113` | exact multiset shortcut |
| same-difference random negative | 0 | `1.000000000` | `0.997866392` | exact multiset shortcut |
| same-difference random negative | 1 | `1.000000000` | `0.997518897` | exact multiset shortcut |
| matched negative | 0 | `0.500000000` | `0.890203834` | multiset shortcut removed; pair-XOR variance remains |
| matched negative | 1 | `0.500000000` | `0.882089019` | multiset shortcut removed; pair-XOR variance remains |

Interpretation:

```text
strict and same-difference random negatives:
  Positive rows have left/right ciphertext multisets that are identical across
  the 16 integral pairs. Negative rows do not. Therefore the mean absolute
  difference between left and right bit-column sums is 0 for positives and
  nonzero for negatives, giving a perfect deterministic separator. These
  protocols are too easy and should not be scaled.

matched negative:
  The exact left/right column-sum shortcut is removed: both positive and
  negative rows have left_right_column_sum_l1_mean = 0. But a weaker hand-coded
  statistic, pair_xor_column_sum_variance, still reaches about 0.88-0.89 AUC.
  Future trail-position claims should be compared against this deterministic
  baseline at the same scale.
```

## Matched-Negative Residual Follow-Up Plan

Purpose:

```text
The deterministic weak-XOR audit found that strict and same-difference random
negative rows are perfectly separated by a left/right ciphertext column-sum
multiset statistic. The matched-negative protocol removes that exact shortcut,
but still has a strong pair_xor_column_sum_variance deterministic baseline
around 0.88-0.89 AUC at 2048/class.

This follow-up asks whether the neural full beamstats/trail-position rows add
residual value beyond that hand-coded weak-XOR statistic under the more
meaningful matched-negative protocol.
```

Plan:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_matched_residual_audit_2048_seed0_seed1.csv
```

Gate:

```text
If trail-position is only near the deterministic pair_xor_column_sum_variance
baseline, treat the high score as mostly explained by weak-XOR pair variance.

If trail-position clearly exceeds that deterministic baseline on both seeds,
keep it as a residual candidate, but still report it as local diagnostic
evidence rather than formal PRESENT r8 evidence.

If per-row key rotation collapses, treat fixed-key dependence as unresolved.
```

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_matched_residual_audit_2048_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_matched_residual_audit_2048_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_matched_residual_audit_2048_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_matched_residual_audit_2048_seed0_seed1/progress.jsonl
```

## Matched-Negative Residual Result

Run artifacts:

```text
results = outputs/local_smoke/i1_present_r8_matched_residual_audit_2048_seed0_seed1/results.jsonl
progress = outputs/local_smoke/i1_present_r8_matched_residual_audit_2048_seed0_seed1/progress.jsonl
curves = outputs/local_smoke/i1_present_r8_matched_residual_audit_2048_seed0_seed1/curves.svg
history = outputs/local_smoke/i1_present_r8_matched_residual_audit_2048_seed0_seed1/history.csv
cache = outputs/local_cache/i1_present_r8_matched_residual_audit_2048_seed0_seed1
```

Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_matched_residual_audit_2048_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_matched_residual_audit_2048_seed0_seed1/results.jsonl \
  --expected-rows 14

status = pass
field_mismatches = []
```

Metrics:

| Seed | Role | Key rotation | Feature | Model | AUC | Oriented AUC | Delta vs deterministic pair-XOR variance |
|---:|---|---:|---|---|---:|---:|---:|
| 0 | deterministic baseline | `0` | pair-XOR column variance | hand-coded | `0.890203834` | `0.890203834` | `0.000000000` |
| 0 | matched full global | `0` | full beamstats | global | `0.866194725` | `0.866194725` | `-0.024009109` |
| 0 | matched full trail | `0` | full beamstats | trail-position | `0.999062538` | `0.999062538` | `+0.108858705` |
| 0 | matched full global per-row key | `1` | full beamstats | global | `0.860925198` | `0.860925198` | `-0.029278636` |
| 0 | matched full trail per-row key | `1` | full beamstats | trail-position | `0.998202324` | `0.998202324` | `+0.107998490` |
| 0 | matched xor global | `0` | ciphertext xor | global | `0.000000000` | `1.000000000` | sign-flipped weak feature |
| 0 | matched PAligned global | `0` | P-layer aligned xor | global | `0.591629505` | `0.591629505` | `-0.298574328` |
| 0 | matched InvS global | `0` | P-layer + InvS | global | `0.616469860` | `0.616469860` | `-0.273733974` |
| 1 | deterministic baseline | `0` | pair-XOR column variance | hand-coded | `0.882089019` | `0.882089019` | `0.000000000` |
| 1 | matched full global | `0` | full beamstats | global | `0.878118038` | `0.878118038` | `-0.003970981` |
| 1 | matched full trail | `0` | full beamstats | trail-position | `0.996650696` | `0.996650696` | `+0.114561677` |
| 1 | matched full global per-row key | `1` | full beamstats | global | `0.877559662` | `0.877559662` | `-0.004529357` |
| 1 | matched full trail per-row key | `1` | full beamstats | trail-position | `0.998130798` | `0.998130798` | `+0.116041780` |
| 1 | matched xor global | `0` | ciphertext xor | global | `0.000000000` | `1.000000000` | sign-flipped weak feature |
| 1 | matched PAligned global | `0` | P-layer aligned xor | global | `0.577971458` | `0.577971458` | `-0.304117560` |
| 1 | matched InvS global | `0` | P-layer + InvS | global | `0.591490269` | `0.591490269` | `-0.290598750` |

Interpretation:

```text
The matched-negative protocol removes the exact left/right multiset shortcut,
but a deterministic pair-XOR variance statistic remains strong at about
0.88-0.89 AUC.

The global full-beamstats neural row does not beat that deterministic baseline.
It is at or slightly below the hand-coded statistic on both seeds.

The trail-position full-beamstats row does beat the deterministic baseline on
both seeds by about +0.108 to +0.116 AUC, and per-row key rotation does not
collapse it. This supports trail-position residual value at local diagnostic
scale, but it is still not formal PRESENT r8 evidence.
```

Decision:

```text
matched_trail_position_residual_supported_local_diagnostic
```

Next action:

```text
Keep matched-negative as the main local protocol for trail-position residual
work. Do not scale strict or same-difference random-negative protocols. The
next useful step is a residualized/frozen-score comparison that explicitly
controls for pair_xor_column_sum_variance, then a lean 65k/class or 262k/class
remote run only if that residual check remains positive.
```

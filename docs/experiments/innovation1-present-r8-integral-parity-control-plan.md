# Innovation 1 PRESENT r8 Integral Parity Control Plan

**Date:** 2026-07-05

**Status:** Control A implemented / local matched-negative smoke complete / no remote launch

**Scope:** PRESENT-80 r8 integral/multiset data-construction controls under
strict `encrypted_random_plaintexts` semantics. This plan does not modify the
Zhang/Wang Case2 benchmark and does not claim a neural distinguisher result.

## Why This Plan Exists

The completed r8 integral/inverse feature screen produced:

```text
run_id = i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705
raw integral anchor AUC = 0.999995831400
InvP matrix AUC = 0.513465017546
InvP+Sinv matrix AUC = 0.505787684582
decision = stop_integral_inverse_feature_screen_for_now
```

A deterministic local audit then showed that the raw integral anchor is already
separable without neural training:

```text
audit = integral_pair_xor_parity
samples_per_class = 2048
positive_pair_xor_parity_hw.zero_rate = 1.0
negative_pair_xor_parity_hw.zero_rate = 0.0
best threshold = parity_hw <= 0
accuracy = 1.0
```

The same `1.0` threshold accuracy was observed for:

| Audit artifact | Key split | Seed | Accuracy |
|---|---|---:|---:|
| `outputs/local_audits/r8_integral_raw_anchor_parity_audit_seed7_2048.json` | validation | 7 | `1.0` |
| `outputs/local_audits/r8_integral_raw_anchor_parity_audit_train_seed7_2048.json` | train | 7 | `1.0` |
| `outputs/local_audits/r8_integral_raw_anchor_parity_audit_validation_seed19_2048.json` | validation | 19 | `1.0` |

Interpretation:

```text
The current raw integral anchor is dominated by a deterministic multiset parity
statistic created by the positive-vs-random-right sample construction. It is
not evidence that the current InvP/Sinv matrix architecture should scale.
```

## Current Data-Construction Mismatch

For `sample_structure = plaintext_integral_nibble`:

```text
positive:
  left plaintexts form one active-nibble integral set
  right plaintexts are left plaintexts xor fixed input_difference
  therefore the right side also forms an aligned active-nibble integral set

negative:
  left plaintexts form one active-nibble integral set
  right plaintexts are independent random plaintexts
```

That mismatch makes this statistic trivial:

```text
parity_hw = HW( XOR_over_pairs(C_i xor C'_i) )
```

For the current protocol, `parity_hw <= 0` separates all checked positive and
negative samples.

## Research Question

After removing the trivial positive-vs-random-right integral mismatch, does any
nontrivial PRESENT r8 integral/multiset signal remain?

The answer determines whether to keep a controlled integral data route or
return to a smaller SPN-derived feature probe.

## Non-Goals

This plan does not:

```text
change Zhang/Wang Case2 r7/r8 benchmark rows
claim a neural architecture gain
scale the current raw integral anchor
scale the current InvP/Sinv matrix candidate
fit a model before deterministic controls pass
call 65536/class formal evidence
```

## Candidate Controls

### Control A: Matched Integral Negative

Create a new local-only sample structure such as:

```text
plaintext_integral_nibble_matched_negative
```

Positive:

```text
left = active-nibble integral set from base
right = left xor input_difference
```

Negative:

```text
left = active-nibble integral set from base_a
right = independent active-nibble integral set from base_b
```

Expected parity behavior:

```text
Both classes should have pair_xor parity_hw near zero if the trivial multiset
parity leak is removed.
```

Decision:

```text
Implement this first as a local data-generation/audit control, not a remote
training route.
```

### Control B: Pair-Order Scramble

Keep the current positive and negative construction but randomly permute pair
order on one side before encoding.

Purpose:

```text
Test whether pair alignment, not only integral balance, is carrying the raw
anchor signal.
```

Decision:

```text
Use after Control A if parity no longer separates but a simpler alignment
statistic still might.
```

### Control C: Parity-Removed Feature Mask

Keep the dataset fixed but remove or mask the global pair-xor parity statistic
from deterministic probes.

Decision:

```text
Use only as a diagnostic. A feature mask cannot make the data construction
claim clean by itself.
```

## Gate

Run deterministic audits before any model training:

```text
integral_pair_xor_parity
left_multiset_ciphertext_xor_hw
right_multiset_ciphertext_xor_hw
pair_alignment_parity_or_permutation_checks
```

Decision table:

| Result | Decision |
|---|---|
| Current parity statistic remains `>= 0.99` accurate | invalid control; do not train |
| Any trivial deterministic statistic is `>= 0.90` accurate | fix data construction before training |
| Deterministic statistics fall near chance, but feature probes show signal | write a small controlled feature-probe plan |
| Deterministic statistics and probes are near chance | stop current integral data route |

## Next Concrete Step

Control A has been implemented locally with tests:

```text
sample_structure = plaintext_integral_nibble_matched_negative
test = test_integral_parity_audit_matched_negative_removes_pair_xor_separator
```

Implementation summary:

```text
positive:
  unchanged active-nibble integral set paired by fixed input_difference

matched negative:
  the right side uses the same active-nibble multiset as the left side, but
  with a one-step variant rotation instead of the fixed input_difference pair
```

Local audit at `2048/class`, seed `7`, validation key:

| Class | parity-HW mean | zero rate |
|---|---:|---:|
| Positive | `0.0` | `1.0` |
| Matched negative | `0.0` | `1.0` |

Gate:

```text
best threshold = parity_hw <= 0
accuracy = 0.5
interpretation = parity_statistic_does_not_explain_result_by_itself
```

Verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_project_structure.py::test_integral_parity_audit_detects_plaintext_integral_pair_xor_signal \
  tests/test_project_structure.py::test_integral_parity_audit_matched_negative_removes_pair_xor_separator \
  -q
```

Result:

```text
2 passed
```

Matched-negative smoke/probe:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_integral_matched_negative_probe_smoke.csv
output = outputs/local_smoke/i1_present_r8_integral_matched_negative_probe_smoke/results.jsonl
samples_per_class = 256
validation_samples_per_class = 128
epochs = 3
device = cpu
```

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_integral_matched_negative_probe_smoke.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 32 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --output outputs/local_smoke/i1_present_r8_integral_matched_negative_probe_smoke/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_integral_matched_negative_probe_smoke/progress.jsonl
```

Result:

| Row | Feature | AUC | Calibrated accuracy | Accuracy |
|---|---|---:|---:|---:|
| Raw pair | `ciphertext_pair_bits` | `0.805480957031` | `0.753906250000` | `0.531250000000` |
| InvP matrix | `present_pair_xor_paligned_cell_matrix_bits` | `0.530761718750` | `0.566406250000` | `0.484375000000` |
| InvP+Sinv matrix | `present_pair_xor_paligned_sinv_cell_matrix_bits` | `0.547485351562` | `0.550781250000` | `0.500000000000` |

Plot/history artifacts:

```text
outputs/local_smoke/i1_present_r8_integral_matched_negative_probe_smoke/curves.svg
outputs/local_smoke/i1_present_r8_integral_matched_negative_probe_smoke/history.csv
```

Interpretation:

```text
The explicit pair-xor parity separator is removed, but the raw ciphertext-pair
view still shows a small-scale residual signal in this tiny smoke. The InvP and
InvP+Sinv matrix rows do not show convincing support at this scale.
```

This is not enough for a remote launch. The raw-pair result may reflect another
pair-alignment or construction artifact, small validation variance, or a real
controlled integral signal. It needs seed repeat and additional deterministic
alignment audits first.

Next concrete step:

```text
Run a second local matched-negative probe with a different seed and add a
deterministic pair-alignment audit. Do not launch remote training until those
controls distinguish artifact from residual SPN signal.
```

## Seed Repeat And Pair-Alignment Audit

The second local matched-negative smoke/probe used the same protocol with
`seed = 1`:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_integral_matched_negative_probe_smoke_seed1.csv
output = outputs/local_smoke/i1_present_r8_integral_matched_negative_probe_smoke_seed1/results.jsonl
samples_per_class = 256
validation_samples_per_class = 128
epochs = 3
device = cpu
```

Result:

| Row | Feature | AUC | Calibrated accuracy | Accuracy |
|---|---|---:|---:|---:|
| Raw pair | `ciphertext_pair_bits` | `0.877990722656` | `0.800781250000` | `0.578125000000` |
| InvP matrix | `present_pair_xor_paligned_cell_matrix_bits` | `0.530029296875` | `0.546875000000` | `0.500000000000` |
| InvP+Sinv matrix | `present_pair_xor_paligned_sinv_cell_matrix_bits` | `0.574340820312` | `0.578125000000` | `0.500000000000` |

Plot/history artifacts:

```text
outputs/local_smoke/i1_present_r8_integral_matched_negative_probe_smoke_seed1/curves.svg
outputs/local_smoke/i1_present_r8_integral_matched_negative_probe_smoke_seed1/history.csv
```

Deterministic parity audit for the seed1 plan, generated at `2048/class` with
audit seed `11`, still removes the explicit separator:

```text
artifact = outputs/local_audits/r8_integral_matched_negative_parity_audit_seed11_2048.json
positive_pair_xor_parity_hw.zero_rate = 1.0
negative_pair_xor_parity_hw.zero_rate = 1.0
best threshold = parity_hw <= 0
accuracy = 0.5
interpretation = parity_statistic_does_not_explain_result_by_itself
```

A new pair-alignment audit checks three simple deterministic statistics:

```text
same_index_xor_hw_mean
shifted_index_xor_hw_mean
same_minus_shifted_xor_hw_mean
```

For the same seed1 plan and audit seed `11`:

```text
artifact = outputs/local_audits/r8_integral_matched_negative_alignment_audit_seed11_2048.json
best statistic = same_index_xor_hw_mean
best threshold accuracy = 0.546630859375
positive same_index_xor_hw_mean = 31.9224853515625
negative same_index_xor_hw_mean = 31.95086669921875
```

Interpretation:

```text
The matched-negative raw-pair smoke signal survives a second tiny seed, while
the explicit pair-xor parity leak and the simple pair-order alignment
statistics remain near chance. This makes the residual raw-pair signal worth a
controlled local follow-up, but not a remote launch or architecture claim.
```

Decision:

```text
keep_as_local_control_candidate
```

Next action:

```text
Build a lean deterministic feature audit / shallow-probe control for the raw
matched-negative pair route. The next remote slot should still wait until a
nontrivial controlled statistic or same-budget local probe explains why the raw
pair rows are positive while InvP/Sinv matrix rows remain weak.
```

## Deterministic Feature-Bank Audit

A follow-up deterministic feature-bank audit now explains the raw-pair residual
much more directly. It computes simple scalar summaries from the raw ciphertext
pair set only:

```text
left/right/pair-xor HW mean and std
left/right/pair-xor column-sum variance
left-right column-sum L1/L2 mean
```

For the seed0 matched-negative plan:

```text
artifact = outputs/local_audits/r8_integral_matched_negative_feature_bank_audit_seed7_2048.json
best statistic = pair_xor_column_sum_variance
best threshold = pair_xor_column_sum_variance >= 5.484375
best threshold accuracy = 0.979248046875
positive mean = 7.8383893966674805
negative mean = 3.923461437225342
```

For the seed1 matched-negative plan:

```text
artifact = outputs/local_audits/r8_integral_matched_negative_feature_bank_audit_seed11_2048.json
best statistic = pair_xor_column_sum_variance
best threshold = pair_xor_column_sum_variance >= 5.5380859375
best threshold accuracy = 0.982421875
positive mean = 7.895696640014648
negative mean = 3.936082363128662
```

Interpretation:

```text
The raw matched-negative neural smoke signal is now best understood as a
learned version of a simple deterministic pair-xor column-distribution
variance statistic. The explicit parity leak was removed, but a different
multiset-distribution statistic remains very strong.
```

Decision update:

```text
do_not_scale_as_neural_architecture_result
keep_as_spn_multiset_feature_candidate
```

Next action update:

```text
Turn this into a controlled deterministic SPN/multiset feature route only if it
survives additional controls such as active-nibble variation, input-difference
variation, pair-order scramble, and comparison against Zhang/Wang-compatible
same-budget anchors. Do not launch a remote neural training job for this raw
matched-negative route yet.
```

## Claim Scope

Allowed after this plan's local control:

```text
The original r8 integral raw anchor was dominated by a deterministic multiset
parity statistic, and the matched-negative control either removes or does not
remove that trivial separator.
The matched-negative raw-pair residual is strongly explained by
pair_xor_column_sum_variance at local deterministic-audit scale.
```

Not allowed:

```text
PRESENT r8 neural breakthrough
Zhang/Wang same-protocol improvement
formal evidence
architecture gain
publication-style claim without controlled scale and attribution
```

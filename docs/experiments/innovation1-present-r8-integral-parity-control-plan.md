# Innovation 1 PRESENT r8 Integral Parity Control Plan

**Date:** 2026-07-05

**Status:** Control A implemented / local parity control passes / no remote launch

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

Next concrete step:

```text
Run a local matched-negative smoke/probe before any remote screen. If the model
or deterministic probes are near chance, stop the current integral data route.
If a nontrivial signal remains, write a lean controlled remote screen plan.
```

## Claim Scope

Allowed after this plan's local control:

```text
The original r8 integral raw anchor was dominated by a deterministic multiset
parity statistic, and the matched-negative control either removes or does not
remove that trivial separator.
```

Not allowed:

```text
PRESENT r8 neural breakthrough
Zhang/Wang same-protocol improvement
formal evidence
architecture gain
publication-style claim without controlled scale and attribution
```

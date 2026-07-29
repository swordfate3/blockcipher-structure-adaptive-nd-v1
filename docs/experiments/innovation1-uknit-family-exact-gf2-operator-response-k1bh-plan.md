# Innovation 1 K1-BH Exact GF(2) Operator-Response Audit

**Status:** completed / hold
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_exact_gf2_operator_response_k1bh_audit_replica0_replica1_20260729`

## 1. Research Question

K1-BC, K1-BE and K1-BG successively tested learned token concatenation,
mandatory learned token gating and a fixed full-rank token basis. The final
K1-BG basis retained at least `1.073x` of K1-BE's whole-path effect but reduced
median topology share to `0.569x/0.575x` of K1-BC. The learned edge-message and
pooling family is therefore stopped.

K1-BH resolves the next causal mismatch before another neural architecture:

> When the supplied runtime GF(2) matrices are applied exactly to the existing
> sample Boolean views, does a probe fitted only on the correct operator retain
> label signal that disappears under both wrong operators?

If yes, a future model should consume exact transported states directly. If
not, another message network cannot reasonably be blamed for failing to infer
an operator distinction absent from this data/control surface.

## 2. One Feature Variable

For each ciphertext pair construct three raw bit channels:

```text
left, right, left XOR right
```

Apply each of the two supplied inverse-linear matrices exactly modulo two and
retain four ordered views:

```text
raw
inverse_linear_0(raw)
inverse_linear_1(raw)
inverse_linear_0(inverse_linear_1(raw))
```

Average only across the four ciphertext pairs, then flatten ordered
`bit x view x channel` coordinates. Native bit order, cell membership and bit
role therefore remain represented until after exact transport. The feature
dimensions are `64 x 12 = 768` for uKNIT/Midori and `128 x 12 = 1536` for
Dialga. No S-box, neural layer, trainable edge message, invariant bit pooling,
cipher ID or per-cipher neural parameter is used.

## 3. Frozen Probe Protocol

Use the existing deterministic diagonal Fisher recipe with variance floor
`1e-6` for every replica and cipher.

1. Extract correct-operator features from the existing 4096-row `train_seen`
   cache and fit one correct scorer.
2. Fit one label-shuffled scorer on the identical correct features with the
   frozen permutation seed; preserve exact class counts.
3. On each untouched 2048-row fresh split, evaluate the correct scorer on
   correct, same-summary corrupted, cross-cipher and identity-operator features.
4. Evaluate the shuffled scorer only on correct features.

The wrong operators must not receive separate refits. Reusing the exact
correct-fit weights makes this a same-probe topology intervention rather than a
comparison of independently adapted coordinate systems.

Frozen label-shuffle seeds:

```text
replica0: uKNIT=73100 Midori=73101 Dialga=73102
replica1: uKNIT=73200 Midori=73201 Dialga=73202
```

## 4. Authority And Budget

Bind and rehash K1-BG's config, gate, validation, results, panel results,
summary and geometry. Reuse K1-BG's authority chain to recover the same 18
disk-backed datasets, exact K1-AZ-era runtime structures and both frozen wrong
operators.

```text
uKNIT-BC r5       seeds 3/4
Midori64 r4       seeds 6/7
Dialga-128 r4     seeds 0/1
4 pairs/sample
4096 train rows per replica/cipher
2048 rows per fresh split
neural optimizer steps = 0
local CPU audit only
```

## 5. Required Controls And Gates

Conditions:

```text
correct_operator
same_summary_corrupted_operator
cross_cipher_operator
identity_operator
label_shuffled_correct_operator
```

For every one of twelve `replica x cipher x fresh split` panels require:

```text
correct AUC                         >= 0.55
correct - identity AUC             >= 0.01
correct - same-summary wrong AUC   >= 0.01
correct - cross-cipher wrong AUC   >= 0.01
correct - label-shuffle AUC        >= 0.03
label-shuffle AUC                  <= 0.53
```

Every exact wrong/identity response must also differ from the correct response
by finite, strictly positive RMS. Protocol gates require exact source digests,
72 feature manifests, 12 scorer rows, 60 fresh result rows, frozen feature
dimensions, class-count-preserving shuffles, correct-fit scorer reuse, zero
neural optimizer steps, no data generation and finite scores/AUCs.

## 6. Decisions

- **All gates pass:** preregister a direct exact-GF(2)-transport residual
  readiness design. Do not train or scale yet.
- **Correct AUC fails:** direct transport lacks stable label signal on the
  current cipher/round/difference surface. Return to a bounded benchmark and
  difference-position audit, not another network.
- **Correct signal exists but wrong/identity margins fail:** exact transport is
  predictive but not topology-identifying. Audit the wrong-operator controls
  and representation equivalence before architecture work.
- **Shuffle attribution fails:** hold the route and inspect probe/control
  validity; do not reinterpret the row as structure evidence.
- **Protocol failure:** repair only the failed binding or implementation
  invariant and rerun unchanged.

No outcome authorizes neural training, 16 pairs, more data/epochs/seeds/width,
remote GPU, separate wrong-operator fits, MoE/per-cipher neural modules or
benchmark changes.

## 7. Required Artifacts

Write under `outputs/local_audit/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
feature_manifest.jsonl
scorers.jsonl
results.jsonl
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

The Chinese figure must separate absolute label signal, both topology margins,
identity/shuffle controls and exact-response differences. It must state that
the Fisher fit is a deterministic mechanism probe, not neural accuracy or
formal-scale evidence, and pass a `2700 x 1800` rendered-pixel
`visual-qa-redraw` inspection.

## 8. Completed Result

K1-BH completed the frozen local audit with zero neural updates:

```text
feature manifests = 72 / 72
scorer rows       = 12 / 12
fresh result rows = 60 / 60
protocol errors   = []
status            = hold
decision          = innovation1_uknit_family_k1bh_exact_operator_signal_unstable
```

Correct-operator AUC on the four fresh panels per cipher:

| Cipher / rounds | replica0 same-key | replica0 cross-key | replica1 same-key | replica1 cross-key |
|---|---:|---:|---:|---:|
| uKNIT-BC r5 | `0.499370` | `0.496367` | `0.470592` | `0.514756` |
| Midori-64 r4 | `0.603778` | `0.613360` | `0.613196` | `0.603104` |
| Dialga-128 r4 | `0.993649` | `0.987278` | `0.990101` | `0.991722` |

Midori passed every frozen topology and attribution margin. Its
correct-minus-same-summary margin was `+0.092501` to `+0.121085`, and its
correct-minus-cross-cipher margin was `+0.107615` to `+0.124783`.

Dialga also passed every frozen margin. Its same-summary wrong operator still
retained high AUC (`0.929885-0.938639`), but the correct operator remained
ahead by `+0.048639` to `+0.061837`; its cross-cipher and identity controls
were near chance while the correct response was approximately `0.99`.

uKNIT failed the absolute signal gate on all four panels and failed most
operator margins. The exact bit-response feature therefore cannot serve as a
shared uKNIT-family architecture surface even though the identical primitive
is strongly informative for Midori and Dialga.

## 9. Interpretation Boundary

K1-BH rules out this complete combination:

```text
ordered exact linear GF(2) bit responses
-> mean over four pairs
-> independent bit coordinates
-> diagonal Fisher scorer
```

It does not show that uKNIT lacks a deterministic relation. K1-Q and K1-S
already established `0.806228-0.825591` AUC on these seed3/4 cell11 datasets
with an exact five-stage, position-preserving cell-value histogram. Repeating
the difference-position sweep would therefore ignore stronger, already-bound
evidence. The unresolved mismatch is representation: K1-BH exposes independent
bit means, whereas K1-Q/K1-S preserve the joint four-bit value of every native
cell through nonlinear stages.

This is a local deterministic mechanism result, not neural accuracy, formal
scale, an attack, SOTA evidence, arbitrary-SPN transfer or a model ceiling.

## 10. Recommended Next Action: K1-BI Cell-Joint Response Audit

K1-BI should change one variable only:

```text
K1-BH: bit x view x channel pair means
K1-BI: runtime-cell x view x 16-value categorical histograms
```

For each of the same four exact GF(2) views, use runtime `cell_membership` and
`bit_role` to reconstruct each native 4-bit cell value, one-hot encode its
`0..15` value, and average only across the same four ciphertext pairs. Retain
cell position and view order. Do not add an S-box stage, neural parameters,
cipher ID or per-cipher component in this audit.

The same-budget anchor is K1-BH. Reuse the exact 18 disk caches, two replicas,
three ciphers, `4096` total train rows, `2048` total rows per fresh split,
strict encrypted-random-plaintext negatives, correct-fit Fisher scorer and the
same-summary, cross-cipher, identity and label-shuffle controls. No data
generation, pair expansion, remote GPU or neural optimization is authorized.

Advance only if every one of the twelve fresh panels satisfies the current
K1-BH signal and control gates, and every Midori/Dialga correct-operator AUC is
no more than `0.02` below its K1-BH anchor. If K1-BI passes, preregister a
shared neural residual that consumes exact transported native-cell tokens. If
uKNIT remains below `0.55`, stop linear-only response redesign and bind the
already-proven runtime S-box-aware five-stage cell statistic as the next
family-level primitive; do not return to edge-message pooling or mechanical
scale-up.

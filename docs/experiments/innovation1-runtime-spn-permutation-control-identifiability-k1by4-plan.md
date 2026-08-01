# Innovation 1 Runtime SPN Permutation Control Identifiability K1-BY4

**Date:** 2026-08-01  
**Status:** completed / hold
**Execution:** local CPU, zero neural training, frozen K1-BY3 validation caches

## Research question

K1-BY3 showed that compiled PRESENT inverse primitives substantially improved
PRESENT-80 r7 validation AUC over the no-conditioner anchor, but the correct
target binding did not beat the wrong-target-binding control on seed3. K1-BY4
tests the failed control itself before any additional optimization:

> Is the existing wrong target-cell binding statistically distinguishable from
> the correct PRESENT program after cell order is removed, and is a within-cell
> source-role corruption a stronger equal-geometry permutation control?

This is a mechanism audit. It does not train a network, generate data, change
the benchmark, test more rounds, establish transfer, or support attack/SOTA
claims.

## Frozen authority

The audit reads only the two K1-BY3 validation caches:

| Seed | Rows | Pairs/sample | Feature shape |
|---:|---:|---:|---|
| 2 | `2048` total (`1024/class`) | 16 | `[2048, 2048]` bits |
| 3 | `2048` total (`1024/class`) | 16 | `[2048, 2048]` bits |

The source gate must remain:

```text
status   = hold
decision = innovation1_runtime_spn_k1by3_permutation_attribution_not_supported
```

The configuration freezes SHA-256 digests for the K1-BY3 gate, results,
validation, preflight, plan, PRESENT descriptor and all six cache files. Any
drift invalidates the audit before metrics are interpreted. Source files must
have identical digests before and after execution.

## One variable and controls

The correct compiled program is the reference. Two deterministic corruptions
are compared against it:

| Control | Exact change | Why it is tested |
|---|---|---|
| current wrong target binding | move each complete target-cell edge bundle using frozen seed 11 | reproduces the K1-BY3 control that failed attribution |
| source-role corruption | replace every source role by frozen permutation `[1,3,0,2]` inside its source cell | preserves one-to-one fan-in and the permutation expert, but does not merely relabel complete cells |

The source-role control must keep every inverse-linear matrix row and column
sum equal to one and must keep all 32 linear calls routed to
`linear_permutation` (`linear_gf2=0`). No absolute cell ID is introduced.

## Deterministic taps

For each seed, execute the two compiled inverse stages in the same reverse
order as `OrderedPrimitiveConditioner`. At each stage collect:

1. `inverse_linear`: per-cell 16-bin ciphertext-difference counts immediately
   after the inverse linear layer;
2. `post_inverse_sbox`: the same counts after the inverse S-box.

Each histogram contains integer counts summing to 16 for every sample and
cell. Three comparisons are frozen:

| Metric | Definition | Interpretation |
|---|---|---|
| `multiset_equal_rate` | fraction of samples whose 16 cell histograms are exactly equal after lexicographic sorting removes cell order | `1.0` means invariant cell pooling cannot distinguish the programs at that tap |
| `pooled_summary_l1` | mean absolute difference between concatenated cell-mean and cell-max summaries, divided by 16 pairs | positive values expose a difference to K1-BY3-style invariant summaries |
| `ordered_histogram_l1` | normalized L1 distance before removing cell order | supporting evidence that the programs are not bit/cell-order identical |

The matrix contains exactly `2 seeds x 2 controls x 2 stages x 2 taps = 16`
result rows.

## Preregistered gates

A control is **identifiable at one seed/stage/tap** only when both hold:

```text
multiset_equal_rate <= 0.95
pooled_summary_l1   >= 0.0001
```

The source-role control uniformly dominates the current control only when, at
every seed/stage/tap:

```text
current multiset_equal_rate - source-role multiset_equal_rate >= 0.01
source-role pooled_summary_l1 - current pooled_summary_l1     >= 0.0001
```

No averaging may rescue a failed seed, stage or tap.

## Frozen decision order

1. **Protocol invalid:** repair only the failed source, geometry, histogram or
   artifact invariant and rerun unchanged.
2. **Current wrong binding identifiable everywhere:** choose
   `learned_pooling_audit_required`; replay the frozen K1-BY3 checkpoints and
   locate the first learned tap that erases the already-visible control.
3. **Source-role identifiable and uniformly stronger everywhere:** choose
   `source_role_control_preferred`; reclassify complete-cell target shuffling
   as an inadequate homogeneous-P-layer control, then preregister one
   same-budget K1-BY5 neural row per seed using the frozen source-role control
   against the existing K1-BY3 correct/no-conditioner anchors.
4. **Otherwise:** choose `permutation_expert_hold`; stop neural training and
   remote scale until a structurally identifiable permutation control exists.

Blocked before adjudication: neural training, remote execution, scale/pair/
seed/epoch changes, difference scanning, adding ciphers, adding absolute cell
identity, or changing the K1-BY3 data protocol.

## Planned artifacts

```text
outputs/local_audit/
i1_runtime_spn_permutation_control_identifiability_k1by4_present_r7_seed2_seed3_20260801/
  preflight.json
  results.jsonl
  condition_comparison.csv
  gate.json
  validation.json
  summary.json
  progress.jsonl
  curves.svg
  visual_qa_render_report.json
```

After completion this document must record the measured metrics, frozen
decision, claim boundary and executable next action. The recent-result indexes
must be refreshed in the same turn.

## Completed result

All source, cache, program-geometry and artifact checks passed. The audit read
both frozen validation caches without changing any source digest, produced all
16 expected rows and performed zero optimizer steps.

The two seeds were numerically consistent. `1 - multiset_equal_rate` and
`pooled_summary_l1` are shown below; both must be positive enough at every tap
for a control to be accepted.

| Control | Inverse step | Tap | seed2 multiset change / pooled L1 | seed3 multiset change / pooled L1 |
|---|---:|---|---:|---:|
| current wrong target binding | 1 | inverse linear | `0.000 / 0.000000` | `0.000 / 0.000000` |
| current wrong target binding | 1 | post inverse S-box | `0.000 / 0.000000` | `0.000 / 0.000000` |
| current wrong target binding | 2 | inverse linear | `1.000 / 0.033804` | `1.000 / 0.033601` |
| current wrong target binding | 2 | post inverse S-box | `1.000 / 0.034588` | `1.000 / 0.034649` |
| uniform source-role corruption | 1 | inverse linear | `0.000 / 0.000000` | `0.000 / 0.000000` |
| uniform source-role corruption | 1 | post inverse S-box | `0.000 / 0.000000` | `0.000 / 0.000000` |
| uniform source-role corruption | 2 | inverse linear | `0.000 / 0.000000` | `0.000 / 0.000000` |
| uniform source-role corruption | 2 | post inverse S-box | `0.000 / 0.000000` | `0.000 / 0.000000` |

Both controls changed the ordered histograms substantially: normalized ordered
L1 was approximately `0.414-0.508` for current target binding and
`0.509-0.510` for source-role corruption. Therefore the zero invariant metrics
are not caused by an implementation no-op. They show that these changes are
cell permutations that disappear exactly when cell order is removed.

The current target-binding control stops being equivalent only after the
second compiled inverse transition. The uniform `[1,3,0,2]` source-role change
remains multiset-equivalent through both transitions. It is therefore weaker,
not stronger, for the K1-BY3 invariant representation.

The preregistered decision is:

```text
status       = hold
decision     = innovation1_runtime_spn_k1by4_permutation_expert_hold
remote_scale = no
```

## Interpretation and claim boundary

K1-BY4 explains why a random-fixture output delta was an insufficient K1-BY3
readiness check. A control can change every ordered cell position while leaving
the exact cell-token multiset and mean/max summaries unchanged. On homogeneous
PRESENT layers, complete target-cell movement and one uniform source-role
permutation are automorphism-like changes from the invariant network's point
of view.

This result does not retract the K1-BY3 correct-versus-no-conditioner signal.
It narrows the unsupported attribution claim: the existing controls cannot
test exact P-layer semantics at every recurrent stage. K1-BY4 contains no AUC,
accuracy, training-scale or cross-cipher evidence.

## Executable next action

Preregister K1-BY5 as one more zero-training control audit on the identical
K1-BY3 caches. Change only the source-endpoint bijection. Flatten each source
endpoint as:

```text
u = 4 * source_cell + source_role
```

and use the frozen affine permutation:

```text
u -> (5 * u + 1) mod 64
```

Because `gcd(5,64)=1`, this preserves a global one-to-one P layer. Unlike a
uniform role permutation, the destination source cell depends on the original
role, so complete four-bit cell bundles are split. The model still receives no
absolute cell identity; cell numbers are used only to construct the negative
control.

K1-BY5 must reuse the same seeds, 2048 validation rows per seed, 16 pairs, two
compiled stages, taps, metrics and K1-BY4 thresholds. It runs locally on CPU
with zero epochs and zero optimizer steps.

- If the affine endpoint control changes both the multiset and pooled summary
  at every seed/stage/tap, freeze it and preregister a same-budget neural gate
  with only one new wrong-control training row per seed.
- If any tap remains invariant, stop permutation neural attribution and audit
  the representation contract itself; do not search more random permutations.

Do not increase samples, pairs, epochs, width or seeds; do not launch remotely;
do not add GIFT or another cipher before this control-level question is closed.

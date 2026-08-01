# Innovation 1 Runtime SPN Permutation Expert K1-BY3

**Date:** 2026-08-01
**Status:** completed / hold
**Execution:** local sub-medium diagnostic; local CUDA availability is checked and recorded before optimization

## Research question

K1-BY1 and K1-BY2 established a reproducible uKNIT r5 signal when a runtime
SPN descriptor is compiled into ordered S-box and general GF(2) primitive calls.
That uKNIT window never invoked `linear_permutation`. K1-BY3 asks one new
question:

> Can the same fixed shared parameter geometry use a compiled one-to-one
> permutation program on PRESENT r7, and does the correct target binding beat
> both an identifiable wrong-binding control and the raw-pair/no-conditioner
> anchor on two untouched seeds?

This is a local `2048/class` architecture diagnostic. It is not formal scale,
paper-scale evidence, a Zhang/Wang reproduction, cross-cipher shared-weight
transfer, an attack or a SOTA claim.

## Evidence-surface selection

PRESENT is selected instead of GIFT before K1-BY3 training because the existing
strict-negative local evidence is stronger and already replicated:

| Existing surface | seed0 correct AUC | seed1 correct AUC | Status |
|---|---:|---:|---|
| PRESENT r7 T1, `2048/class`, 16 pairs | `0.664596081` | `0.676282406` | both pass |
| GIFT r6 R2g, `2048/class`, 4 pairs | `0.538176537` | `0.548832417` | both pass, weaker |

PRESENT also has retrieved project-formal `1000000/class` RTG3-B results of
`0.749477538` and `0.749664813` for seed0/1. Those formal results establish that
the PRESENT r7 MCND data surface is not a local-data artifact; they are not a
same-budget architecture baseline for this K1-BY3 diagnostic.

The source authority is frozen to:

```text
K1-BY2 pass decision:
innovation1_runtime_spn_k1by2_fresh_seed_confirmed

PRESENT T1 seed0 pass decision:
innovation1_runtime_spn_present_transfer_seed0_supported

PRESENT T1 seed1 pass decision:
innovation1_runtime_spn_present_transfer_seed1_supported
```

Readiness checks the exact source config/gate/result digests recorded in the
implementation. Source drift invalidates the run before optimizer steps.

## Single variable and controls

Relative to K1-BY2, the runtime descriptor and frozen data surface change from
uKNIT r5/general GF(2) to PRESENT r7/one-to-one P layer. The ordered primitive
model implementation and parameter shapes do not change.

The lean matrix contains exactly:

| Condition | Role |
|---|---|
| `correct_permutation_routing` | compiled PRESENT S-box plus correct P-layer target binding |
| `wrong_permutation_binding` | deterministic target-cell permutation with equal geometry |
| `no_compiler_conditioner` | identical model geometry with structure residual disabled |

PRESENT repeats the same S-box and P layer every round. Rotating a two-stage
program would therefore be algebraically identical to the correct program, so
wrong order is not an identifiable control and is excluded before training.

## Frozen protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | PRESENT-80 r7 |
| Difference | `present_zhang_wang2022_mcnd:0`, `0x0000000000000009` |
| Sample organization | Zhang/Wang Case2 official MCND |
| Key sampling | fresh random PRESENT-80 key for every pair |
| Train | `2048/class`, `4096` total rows |
| Validation | `1024/class`, `2048` total rows |
| Seeds | `2`, `3` |
| Pairs/sample | `16` ciphertext pairs |
| Input width | `2048` bits/sample, `128` bits/pair |
| Negative definition | encrypted random plaintexts under fresh per-pair keys |
| Runtime program | two repeated PRESENT transitions, descriptor round start `0` |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | restored best validation AUC |
| Parameters | exactly `235780` for every condition |

For each seed the same disk-backed train and validation datasets must be reused
across all three conditions. No difference scan, round scan, key tuning,
architecture tuning or seed-dependent option is allowed.

## Readiness gate

Before optimizer steps require:

1. exact K1-BY2 and PRESENT T1 source digests and pass decisions;
2. exactly six frozen plan rows;
3. all compiled linear calls route to `linear_permutation` and none to
   `linear_gf2`;
4. equal `235780` trainable parameters across conditions;
5. finite `[B, 1]` outputs and finite nonzero gradients;
6. correct and wrong-binding semantic digests differ;
7. correct and wrong-binding fixture outputs differ;
8. only the no-conditioner model disables the primitive residual;
9. no model consumes cipher identity or absolute cell/bit identity.

## Result gate

For seed2 and seed3 separately require:

```text
correct AUC                    >= 0.550
correct - no conditioner       >= +0.010
correct - wrong binding        >= +0.005
```

Every clause must pass on both seeds. Averaging cannot rescue a failed seed.

## Decisions

- **Both seeds pass:** the shared compiled contract has independent GF(2) and
  permutation-expert evidence; next isolate deterministic inverse execution
  from learned descriptors under the same PRESENT budget.
- **Correct signal passes but a margin fails:** hold permutation attribution;
  inspect only the failed control without scaling or changing data.
- **Either correct signal fails:** hold this learned permutation expert and
  compare deterministic compiled features with the historical PRESENT T1
  representation before redesigning the interface.
- **Protocol invalid:** repair only the failed invariant and rerun unchanged.

Blocked: remote execution, more samples, more epochs, more pairs, GIFT addition,
difference scanning, model widening, transfer claims and publication claims.

## Planned artifacts

```text
run_id = i1_runtime_spn_permutation_expert_k1by3_present_r7_16pair_2048_seed2_seed3_20260801
```

The completed run must include preflight, disk caches, progress, six
checkpoints, results, validation, gate, summary, condition CSV, history CSV,
Chinese SVG, plot report and rendered-pixel `visual-qa-redraw` evidence. Refresh
both recent-result indexes and append the measured result and an executable next
action here before reporting completion.

## Completed result

Readiness passed before optimization. Every model had `235780` trainable
parameters and the compiled program used exactly:

```text
sbox4_table       = 32 calls
linear_permutation= 32 calls
linear_gf2        = 0 calls
```

All six rows completed, reused the frozen disk caches and passed plan/result
validation.

| Condition | seed2 AUC | seed3 AUC |
|---|---:|---:|
| correct permutation routing | `0.683736801` | `0.665543556` |
| wrong target binding | `0.664277077` | `0.672483444` |
| no compiler conditioner | `0.543799877` | `0.527235508` |

The attribution margins were:

| Margin | seed2 | seed3 |
|---|---:|---:|
| correct - wrong binding | `+0.019459724` | `-0.006939888` |
| correct - no conditioner | `+0.139936924` | `+0.138308048` |

Correct-route best validation accuracy was `0.636718750` for seed2 and
`0.630371094` for seed3. AUC remains the preregistered primary metric; the raw
`0.5` threshold accuracy for seed3 was uncalibrated and does not replace its
best-threshold accuracy or AUC.

The signal floor and no-conditioner margin passed on both seeds. The
wrong-binding margin passed seed2 but failed seed3 because wrong binding was
`0.006939888` AUC higher than correct binding. The frozen decision is:

```text
status       = hold
decision     = innovation1_runtime_spn_k1by3_permutation_attribution_not_supported
remote_scale = no
```

## Interpretation

The strong and repeated correct-versus-no-conditioner margin supports a narrow
claim: compiled PRESENT inverse-primitive features add substantial signal to
the shared raw-pair backbone at this local budget. K1-BY3 does not support the
stronger claim that the learned representation identifies the exact PRESENT
target-cell binding.

The failed control is mechanistically plausible. The network excludes absolute
cell identity and aggregates cell tokens with attention, mean and maximum
pooling. PRESENT applies the same S-box to all cells and repeats one P layer.
Moving complete target-cell edge bundles can therefore preserve much of the
multiset seen by the invariant aggregator even though the compiled program SHA
and a random-fixture output differ. A nonzero fixture delta proved only that the
control was not bit-exact; it did not prove that the control was statistically
identifiable after invariant pooling.

This does not invalidate the completed metrics. It limits their attribution and
shows that the readiness identifiability check was too weak for a homogeneous
permutation SPN.

## Recommended next action

Do not retrain, scale, add GIFT or weaken the gate. Preregister K1-BY4 as a
zero-training representation-identifiability audit using the frozen K1-BY3
cache and compiled programs. For both seeds, compare correct and wrong binding
at these deterministic taps:

```text
inverse-linear per-cell difference histograms
post-inverse-Sbox per-cell difference histograms
cell-token multiset after sorting away cell order
mean/max invariant summaries at each of the two stages
```

Also construct, without training, one within-cell source-role corruption that
preserves one-to-one fan-in and `linear_permutation` routing but cannot be
reduced to moving complete cell bundles. Require it to change the deterministic
multiset and pooled summaries on both cached validation seeds.

- If current wrong binding is multiset-equivalent while source-role corruption
  is identifiable, reclassify wrong binding as an invalid attribution control
  for homogeneous P layers and use the frozen role-corruption control in the
  next same-budget neural gate.
- If current wrong binding is already identifiable before pooling, inspect the
  trained correct/wrong checkpoints tap by tap and redesign only the first
  learned pooling stage where the distinction disappears.
- If neither corruption is identifiable, hold the current permutation expert;
  no further neural training or remote scale is justified.

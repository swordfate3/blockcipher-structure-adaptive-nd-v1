# Innovation 1 uKNIT-Family Partial-State Round Calibration K1-P

**Date:** 2026-07-28  
**Status:** planned / protocol frozen  
**Execution:** local CPU, disk-backed data, deterministic zero-neural-training audit

## 1. Question

K1-O found that the exact five-stage, position-preserving histogram fitted the
uKNIT-BC r5 discovery rows (`AUC 0.775-0.783`) but fell to `0.510-0.527` on
fresh same-key and cross-key rows. Before another neural redesign, K1-P changes
one variable only:

> Does the frozen `input_difference = 0x40` carry reproducible partial-state
> signal at uKNIT r3 or r4 and then lose it by r5, or is the current data and
> difference convention already uncalibrated at lower rounds?

This is not neural training and is not a difference search.

## 2. Source Authority

K1-P binds its r5 anchor to:

```text
outputs/local_audit/
  i1_uknit_family_ctspn_exact_partial_state_signal_audit_k1o_20260728/
```

| Artifact | SHA-256 |
|---|---|
| `gate.json` | `e4da0e2c02404cd8a65457f4c07d8f0b7b8767f17faf6cba48c08faad2d031f1` |
| `results.jsonl` | `345572ef9a311144ba42dbaf6e856f2c78242e620adc38f8340639ecfe842c25` |
| `validation.json` | `9883037941c8b281f014ea95709f08d8365f3e042709d6dc0957e3d95b3457f3` |
| `feature_manifest.jsonl` | `b6a8fe931c3a37a626d5897c4feba066b40913526fdaa1fc850f60e1e8a2a6af` |
| `scorer_manifest.jsonl` | `55b6af9c99d4b80b68455a7ac11779e3f08b6ecae37f09166f3540685ac81ec5` |

The source must retain `status = hold`, validation `status = pass`, all
protocol checks, and decision:

```text
innovation1_uknit_family_ctspn_k1o_current_differential_signal_not_supported
```

The r5 result rows and feature/scorer identities are reused exactly. K1-P must
not regenerate, refit, or reinterpret the r5 anchor.

## 3. Frozen Protocol

| Field | Value |
|---|---|
| Cipher | uKNIT-BC only |
| Rounds | `r3`, `r4`, `r5` |
| Seeds | `0`, `1` |
| Training/discovery | `2048/class`, `4096` total rows |
| Same-key fresh | `1024/class`, `2048` total rows |
| Cross-key | `1024/class`, `2048` total rows |
| Pairs per sample | `4` |
| Input difference | `0x40` |
| Negative definition | encrypted random plaintexts |
| Train key | all-zero 128-bit key |
| Validation key | all-one 128-bit key |
| Dataset seeds | train `seed`; cross-key `seed+10000`; same-key `seed+20000` |
| Scorer | diagonal Fisher/LDA, variance floor `1e-6` |
| Neural parameters / epochs / optimizer steps | `0 / 0 / 0` |

r3 and r4 use new local disk-backed caches with `features.npy`, `labels.npy`,
`metadata.json`, and progress events. r5 reuses K1-O exactly.

## 4. One Variable And Runtime Alignment

Only cipher round count changes. The two-transition inverse window is aligned
to the final two available transitions:

| Cipher rounds | `runtime_round_start` | `runtime_rounds` |
|---:|---:|---:|
| 3 | 1 | 2 |
| 4 | 2 | 2 |
| 5 | 3 | 2 |

Every round retains the same five boundaries:

```text
ciphertext
-> inverse linear slot 1
-> inverse S-box slot 1
-> inverse linear slot 0
-> inverse S-box slot 0
```

For each boundary and native cell, record the four-pair frequency of difference
values `0..15`.

## 5. Lean Panel

Evaluate only:

| View | Features | Role |
|---|---:|---|
| `exact_five_stage_position_histogram` | `1280` | candidate |
| `raw_position_histogram` | `256` | ciphertext-only anchor |
| `label_shuffled_exact_position_histogram` | `1280` | fitted-noise control |

For each round and seed, fit each scorer on train-seen rows only. Evaluate the
unchanged fitted scorer on both fresh splits. AUC orientation remains fixed by
`mean1 - mean0`; no threshold, feature, split, seed, or regularizer selection is
allowed.

## 6. Validity And Advance Gates

The run is valid only if:

1. all six task rows match the frozen round/window map and data protocol;
2. all twelve r3/r4 caches exist, are metadata-bound, and have exact row counts;
3. all six r5 candidate/raw/label-shuffle scorer-result triplets exactly replay
   K1-O hashes and metrics;
4. exactly `54` result rows exist (`3 rounds x 2 seeds x 3 splits x 3 views`);
5. exactly `54` feature rows and `18` scorer rows exist;
6. exact and label-shuffle features share their digest within each dataset;
7. no neural training, checkpoint, optimizer step, or epoch occurs.

Apply these gates independently to both seeds and both fresh splits at r3/r4:

```text
exact AUC                  >= 0.550
exact - raw                >= +0.010
exact - label-shuffled     >= +0.030
```

r5 is the frozen failed anchor; its metrics are not required to pass.

## 7. Decisions

- **r4 passes, r5 fails:** identify r5 as the current `0x40` signal-loss
  boundary. Next preregister an independent r5 difference discovery followed
  by confirmation on untouched seeds/keys. Do not redesign the network first.
- **r3 passes, r4 fails:** identify the boundary before r4. Suspend r5 network
  experiments and discover a round-calibrated difference for r4 before r5.
- **r3 and r4 both fail:** audit `0x40` construction, integer/bit ordering,
  plaintext XOR, round invocation, key binding, and runtime window alignment
  against uKNIT test vectors before any difference or model search.
- **A lower round passes only some seed/split gates:** classify the route as
  key/split unstable and localize the mismatch before scaling.
- **r3 and r4 both pass:** the current difference has a lower-round signal and
  r5 is the first observed failed anchor; proceed to independent r5 difference
  discovery/confirmation.
- **Protocol invalid:** repair only the violated source, cache, task, feature,
  scorer, or artifact invariant and rerun unchanged.

## 8. Artifacts

```text
run_id = i1_uknit_family_ctspn_partial_state_round_calibration_k1p_r3_r4_r5_seed0_seed1_20260728
output = outputs/local_audit/<run_id>/
```

Required artifacts:

```text
preflight.json
dataset_manifest.jsonl
feature_manifest.jsonl
scorer_manifest.jsonl
results.jsonl
round_calibration.csv
validation.json
gate.json
summary.json
progress.jsonl
curves.svg
plot_report.json
visual_qa_passed.marker
```

The chart must explain in Chinese whether `0x40` is visible at r3/r4 and where
it becomes unsupported. It must pass rendered-pixel `visual-qa-redraw`, and the
completed result must refresh both recent-result indexes.

## 9. Claim Boundary And Forbidden Routes

K1-P is a two-seed local `2048/class` mechanism calibration. It cannot establish
formal scale, an attack, SOTA, a uKNIT ceiling, or arbitrary-SPN transfer.
Inside K1-P do not add a neural model, remote scale, MoE, DDT/trail input,
difference search, more pairs, more samples, more seeds, or more epochs.

## 10. Recommended Next Action

Execute K1-P locally from this frozen plan. The result must choose exactly one
of the decision branches above and name the next same-budget experiment or
protocol audit. No architecture change is authorized until the round boundary
is resolved.

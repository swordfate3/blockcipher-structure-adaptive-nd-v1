# Innovation 1 uKNIT-Family Difference-Position Discovery K1-Q

**Date:** 2026-07-28
**Status:** completed / two r5 positions confirmed on untouched seeds and keys
**Execution:** local CPU, disk-backed data, deterministic zero-neural-training audit

## 1. Question

> At uKNIT r5, can moving the same single-bit role used by `0x40` to another
> native input cell recover reproducible exact partial-state signal?

K1-P established that `0x40` is strongly visible at r3/r4 but unsupported at
r5. K1-Q tests the narrower remaining explanation that the fifth-round loss is
position-dependent. It is not another architecture experiment.

## 2. Same-Budget Anchor And One Variable

The anchor is the K1-P/K1-O uKNIT r5 protocol:

- four ciphertext pairs per sample;
- strict encrypted-random-plaintext negatives;
- fixed-key training, same-key fresh validation, and cross-key validation;
- exact five-stage position histogram and diagonal Fisher/LDA scorer;
- raw ciphertext position histogram as the mandatory attribution control;
- no neural parameters, optimizer steps, epochs, or checkpoint selection.

The only research variable is the input-difference position. The `0x40` bit is
runtime bit index 6, native cell 1, `bit_role=1`. K1-Q keeps `bit_role=1` and
tests the corresponding physical bit in every one of the sixteen native
cells:

```text
bit_index(cell) = 4 * cell + 2
input_difference(cell) = 1 << bit_index(cell)
```

This yields the frozen candidate set:

| Cell | Bit index | Input difference | Role |
|---:|---:|---:|---:|
| 0 | 2 | `0x0000000000000004` | 1 |
| 1 | 6 | `0x0000000000000040` | 1, K1-P anchor |
| 2 | 10 | `0x0000000000000400` | 1 |
| 3 | 14 | `0x0000000000004000` | 1 |
| 4 | 18 | `0x0000000000040000` | 1 |
| 5 | 22 | `0x0000000000400000` | 1 |
| 6 | 26 | `0x0000000004000000` | 1 |
| 7 | 30 | `0x0000000040000000` | 1 |
| 8 | 34 | `0x0000000400000000` | 1 |
| 9 | 38 | `0x0000004000000000` | 1 |
| 10 | 42 | `0x0000040000000000` | 1 |
| 11 | 46 | `0x0000400000000000` | 1 |
| 12 | 50 | `0x0004000000000000` | 1 |
| 13 | 54 | `0x0040000000000000` | 1 |
| 14 | 58 | `0x0400000000000000` | 1 |
| 15 | 62 | `0x4000000000000000` | 1 |

Do not test other bit roles inside K1-Q. A wider bit-value scan would mix
position and differential-value effects.

## 3. Frozen Data Protocol

### 3.1 Discovery

| Field | Value |
|---|---|
| Cipher / rounds | uKNIT-BC / 5 |
| Candidate positions | all 16 cells above |
| Seed | 2 |
| Train | `1024/class`, `2048` total rows per position |
| Same-key fresh | `512/class`, `1024` total rows per position |
| Cross-key | `512/class`, `1024` total rows per position |
| Train key | `0x22222222222222222222222222222222` |
| Validation key | `0x33333333333333333333333333333333` |
| Pairs per sample | 4 |
| Negative definition | encrypted random plaintext pairs |
| Cache | parameter-bound `metadata.json`, `features.npy`, `labels.npy` |

The discovery plan is:

```text
configs/experiment/innovation1/
innovation1_uknit_family_ctspn_difference_position_discovery_k1q_seed2.csv
```

For each position, fit the exact and raw diagonal Fisher scorers on its own
training split and evaluate both fresh splits. Rank positions by the minimum
fresh exact AUC, then the minimum exact-minus-raw margin. The anchor remains in
the table but is not eligible to occupy one of the two new-candidate slots.

A non-anchor position is selectable only if both fresh splits satisfy:

```text
exact AUC       >= 0.550
exact - raw     >= +0.010
```

Select at most two positions. Ties are resolved by smaller native-cell index.

### 3.2 Untouched Confirmation

Confirmation is generated only from the frozen selection rule. It evaluates
the selected positions plus the `0x40` anchor on both untouched seeds:

| Seed | Train key | Validation key |
|---:|---|---|
| 3 | `0x44444444444444444444444444444444` | `0x55555555555555555555555555555555` |
| 4 | `0x66666666666666666666666666666666` | `0x77777777777777777777777777777777` |

For every position and seed:

```text
train            = 2048/class = 4096 total rows
same-key fresh   = 1024/class = 2048 total rows
cross-key        = 1024/class = 2048 total rows
```

Confirmation adds the deterministic label-shuffled Fisher scorer. A selected
position confirms only if every seed and fresh split satisfies:

```text
exact AUC                  >= 0.550
exact - raw                >= +0.010
exact - label-shuffled     >= +0.030
```

The anchor is evaluated for context but cannot advance merely by outperforming
its older seed0/1 values.

## 4. Artifact Contract

K1-Q must emit:

```text
preflight.json
selection.json
dataset_manifest.jsonl
feature_manifest.jsonl
scorer_manifest.jsonl
results.jsonl
difference_position.csv
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
plot_report.json
```

Every result row records phase, cell, bit index, hexadecimal/integer input
difference, seed, split, view, row count, AUC, hashes, key scope, pair count,
negative definition, and zero-training fields. Dataset manifests must bind
every cache to its exact parameters and payload files.

## 5. Decision Table

1. **Protocol invalid:** repair only the failed plan, cache, feature, scorer,
   split, or artifact invariant; do not interpret metrics.
2. **No discovery candidate:** stop mechanical position scanning. Preregister a
   DDT/trail-guided input-difference ranking audit; DDT/trail data may rank
   differences but is not a neural-network input.
3. **Discovery candidate but no confirmation:** classify the discovery result as
   selection noise or key/seed instability and stop position scanning.
4. **At least one confirmed position:** return to the K1-N-derived neural model
   at the same `2048/class`, seed3/4 protocol and compare exactly four rows:
   exact structure, wrong S-box, no S-box, and no topology.

`remote_scale = no` for every K1-Q branch. Confirmation is a data-signal gate,
not evidence of a neural distinguisher, attack, uKNIT ceiling, transfer result,
or publication-scale result.

## 6. Blocked Actions

Inside K1-Q do not add another bit role, multi-bit differences, more pairs,
more samples, more seeds, neural training, MoE, remote execution, DDT/trail
features, candidate-specific thresholds, or post-result gate changes.

## 7. Completed Result

```text
status = pass
decision = innovation1_uknit_family_ctspn_k1q_confirmed_r5_difference_position_supported
remote_scale = no
validation status = pass
errors = []
dataset rows = 66
feature rows = 150
scorer rows = 50
result rows = 150
training rows = 0
neural parameters = 0
optimizer steps = 0
epochs = 0
```

### 7.1 Discovery

The frozen seed2 ranking selected cell11 and cell0. The original `0x40`
position was near chance and was excluded from selection:

| Cell / difference | Same-key exact AUC | Cross-key exact AUC | Minimum exact - raw | Outcome |
|---|---:|---:|---:|---|
| 11 / `0x0000400000000000` | `0.939445` | `0.960564` | `+0.462402` | selected rank 1 |
| 0 / `0x0000000000000004` | `0.647923` | `0.624786` | `+0.133270` | selected rank 2 |
| 1 / `0x0000000000000040` | `0.503689` | `0.520794` | `-0.006351` | anchor failed |

Eight additional non-anchor positions passed the discovery threshold, but the
preregistered cap prevented post-hoc expansion beyond the top two.

### 7.2 Untouched Confirmation

Both selected positions passed every absolute and attribution gate on seed3/4
with new train/validation key pairs:

| Cell | Seed | Same-key exact AUC | Cross-key exact AUC |
|---:|---:|---:|---:|
| 11 | 3 | `0.825591` | `0.809460` |
| 11 | 4 | `0.806228` | `0.820532` |
| 0 | 3 | `0.665880` | `0.618431` |
| 0 | 4 | `0.634613` | `0.644200` |
| 1 (`0x40` anchor) | 3 | `0.518620` | `0.504206` |
| 1 (`0x40` anchor) | 4 | `0.515451` | `0.504639` |

Worst-case confirmation margins were:

| Cell | Minimum exact - raw | Minimum exact - label shuffle | Confirmed |
|---:|---:|---:|---|
| 11 | `+0.293386` | `+0.278514` | yes |
| 0 | `+0.111885` | `+0.136022` | yes |
| 1 (`0x40` anchor) | `+0.009514` | `+0.003362` | no |

This supports the position-dependent explanation: the K1-N/K1-O route was
tested on an r5 input difference whose local-state signal had already
collapsed, while two same-role positions preserve strong signal across unseen
seeds and keys. It does not yet show that a neural model can learn that signal
or that correct runtime structure beats neural controls.

The final `curves.svg` passed rendered-pixel `visual-qa-redraw` at `1920x1344`.
The first rendering was rejected because a failed `+0.0095` raw margin rounded
to `+0.010` and could be mistaken for a pass. The final rendering uses four
decimal places for confirmation margins and has no text overlap, clipping,
missing glyphs, incomplete legend, structural ambiguity, or insufficient curve
separation.

## 8. Recommended Next Action: K1-R Neural Attribution

K1-R should test one question only:

> Can the K1-N-derived trainable model exploit the confirmed cell11 r5 signal,
> and is its performance specifically attributable to the correct uKNIT S-box
> and diffusion structure?

Use only the strongest confirmed difference
`0x0000400000000000`, retain the K1-Q seed3/4 disk-backed datasets and keys,
and keep `2048/class` train, `1024/class` per fresh split, four pairs, ten
epochs, strict negatives, optimizer, checkpoint rule, and runtime window fixed.
Change only the structure condition. The lean matrix is eight rows:

```text
seed3/4 x {
  exact composition,
  wrong S-box semantics,
  no S-box semantics,
  no topology
}
```

The same-budget no-topology row is the primary neural anchor. Require the
exact row on both fresh splits and both seeds to reach AUC `>=0.550`, exceed
no-topology by `>=+0.010`, and exceed both S-box controls by `>=+0.005` before
claiming a uKNIT-specific structure contribution. Keep execution local at this
budget. Do not add cell0, more positions, more pairs, MoE, remote scale, or
another architecture until this attribution gate is adjudicated.

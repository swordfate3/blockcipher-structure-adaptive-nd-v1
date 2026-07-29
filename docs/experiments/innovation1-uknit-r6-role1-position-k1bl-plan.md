# Innovation 1 uKNIT r6 Role-1 Position Boundary K1-BL

**Date:** 2026-07-29
**Status:** completed / hold / no confirmed r6 role1 position
**Execution:** local CPU; zero neural updates; disk-backed cache

## 1. Research Question

The completed uKNIT-only evidence now establishes two different facts:

1. K1-Q found that the original `0x40` r5 difference had collapsed because of
   its physical position, while cell11 role1
   `0x0000400000000000` retained fresh deterministic signal.
2. K1-U then trained the uKNIT r5 specialist at remote `65536/class`. The exact
   model reached cross-key AUC `0.974540495/0.967867357`, while the wrong-S-box
   control remained at `0.503900695/0.505827348`.

Therefore r5 is not the current random boundary. K1-BL asks the next narrow
question:

> When only the uKNIT prefix increases from r5 to r6, does any input difference
> with the already verified `bit_role=1` retain reproducible exact five-stage
> signal across unseen seeds and keys?

K1-BL is a data-signal gate before neural training. It is not a neural result,
formal scale, attack, SOTA comparison, or proof that r6 is random.

## 2. Same-Budget Anchor And One Variable

The same-budget anchor is K1-Q. K1-BL retains:

```text
cipher                = uKNIT-BC
candidate family      = one role-1 bit in each of 16 native cells
discovery seed        = 2
confirmation seeds    = 3, 4
train/validation keys = the exact K1-Q key pairs
pairs/sample          = 4 independent ciphertext pairs
negative definition   = encrypted random plaintexts
feature               = exact five-stage native-cell histogram
controls              = raw ciphertext histogram, label-shuffled scorer
metric                = fresh same-key and cross-key AUC
```

The only research variable relative to K1-Q is the evaluated prefix round:

```text
K1-Q anchor: uKNIT r5, runtime window rounds 3..4
K1-BL:       uKNIT r6, runtime window rounds 4..5
```

Moving the runtime window is required alignment, not a second hypothesis: the
five-stage view must continue to expose the final two completed uKNIT
transitions.

## 3. Frozen Difference Matrix

For native cell `c in [0,15]`:

```text
bit_index(c)       = 4*c + 2
input_difference  = 1 << bit_index(c)
active_bit_role   = 1
```

This produces all sixteen candidates from `0x4` through
`0x4000000000000000`. Cell11 `0x0000400000000000` is the mandatory r5 anchor:
it is always confirmed at r6 even if it misses the discovery threshold.

The frozen discovery matrix is:

```text
configs/experiment/innovation1/
  innovation1_uknit_ctspn_r6_role1_position_k1bl_seed2.csv
```

Do not add another bit role, a multi-bit difference, another round, more pairs,
or a trainable model inside K1-BL.

## 4. Discovery And Confirmation

### Discovery

```text
seed                 = 2
train                = 1024/class, 2048 total rows per position
same-key fresh       = 512/class, 1024 total rows per position
cross-key validation = 512/class, 1024 total rows per position
```

Rank the fifteen non-anchor cells by:

1. minimum fresh exact AUC;
2. minimum fresh exact-minus-raw margin;
3. smaller cell index as the frozen tie break.

A non-anchor cell is selectable only if both fresh splits satisfy:

```text
exact AUC   >= 0.550
exact - raw >= +0.010
```

Select at most two. Cell11 is recorded in the same ranking but is confirmed
independently of its discovery status.

### Untouched confirmation

Confirm cell11 and the selected non-anchor cells on seed3/4:

```text
train                = 2048/class, 4096 total rows
same-key fresh       = 1024/class, 2048 total rows
cross-key validation = 1024/class, 2048 total rows
```

A position confirms only if every seed and fresh split satisfies:

```text
exact AUC              >= 0.550
exact - raw            >= +0.010
exact - label-shuffled >= +0.030
```

The label-shuffled scorer uses the same exact feature values and changes only
the fit-label permutation.

## 5. Result Decisions

1. **Protocol invalid:** repair only the failed plan, cache, split, feature,
   scorer, selection, or artifact invariant and rerun unchanged.
2. **At least one r6 position confirms:** freeze the strongest confirmed
   difference. Run the uKNIT-only r6 16-pair neural matrix at `2048/class` with
   exact, wrong-S-box and invariant controls. Do not change the difference while
   comparing networks.
3. **No role1 position confirms:** do not call r6 random. Run a second frozen
   position gate over the remaining single-bit roles `0`, `2`, and `3`.
4. **All 64 single-bit positions fail:** preregister a DDT/differential-trail
   guided multi-bit difference ranking. Trail information may choose candidate
   inputs but must not enter the neural input.
5. **Single-bit and trail-guided families both fail fresh confirmation:** only
   then describe r5-to-r6 as the observed boundary for the searched difference
   families. This still is not proof that every possible r6 distinguisher is
   random.

If the r6 local neural matrix passes, advance it to remote `65536/class` with
disk-backed cache and the same controls. Continue to r7 only after the retrieved
r6 result is plan-aligned. The same ladder repeats until a round fails the full
difference-search and neural-confirmation sequence.

## 6. Required Artifacts

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
visual_qa_passed.marker
```

All data caches must contain `metadata.json`, `features.npy`, `labels.npy`, and
durable progress. After completion, refresh both recent-result indexes and
update this document with metrics, decision, evidence scope, and the exact next
action.

## 7. Prohibited Interpretations

- K1-BL is not neural training.
- A failed role1 scan does not establish that uKNIT r6 is random.
- A passing deterministic feature does not establish that a neural model can
  learn it.
- Local `1024/class` and `2048/class` evidence is not formal or paper scale.
- Do not compare its AUC directly with the designers' 7-round classical
  distinguisher or 10-round key-recovery attacks; the protocols and claims are
  different.

## 8. Completed Result

K1-BL completed locally with no neural training. All protocol checks passed:

```text
dataset rows = 54
feature rows = 114
scorer rows  = 38
result rows  = 114
training rows / optimizer steps / epochs = 0 / 0 / 0
validation status = pass
errors = []
```

No non-anchor role1 position passed the discovery floor. The strongest two
discovery rows by worst fresh AUC were:

| Cell / difference | Worst fresh exact AUC | Worst exact - raw | Outcome |
|---|---:|---:|---|
| 4 / `0x0000000000040000` | `0.517750` | `+0.009949` | failed both frozen thresholds |
| 0 / `0x0000000000000004` | `0.502968` | `+0.019474` | failed AUC floor |

The mandatory r5 cell11 anchor also failed r6 confirmation:

| Seed | Same-key exact AUC | Cross-key exact AUC |
|---:|---:|---:|
| 3 | `0.513282` | `0.520939` |
| 4 | `0.477784` | `0.507025` |

Its worst fresh margins were `-0.018483` over raw ciphertext and `-0.018261`
over the label-shuffled scorer. Therefore:

```text
status = hold
decision = innovation1_uknit_ctspn_k1bl_no_confirmed_r6_role1_difference
remote_scale = no
```

This result closes only the r6 `bit_role=1` single-bit position family. It does
not show that every r6 input difference is random and does not adjudicate the
uKNIT specialist network because no r6 role1 data signal passed the prerequisite
gate.

The rendered `curves.svg` passed `visual-qa-redraw` after increasing the left
margin to expose the cell11 row label and replacing the inherited K1-Q stop text
with the correct role1-only hold decision.

## 9. Executable Next Action

Run K1-BM over all remaining r6 single-bit positions:

```text
one variable        = active bit role, role1 -> roles0/2/3
candidates          = 16 cells x 3 roles = 48 input differences
same-budget anchor  = completed K1-BL role1 ranking
discovery           = seed2, 1024/class train, 512/class per fresh split
confirmation        = seed3/4, 2048/class train, 1024/class per fresh split
pairs               = 4
negative definition = encrypted random plaintexts
feature / controls  = exact five-stage / raw / label-shuffled
execution           = local CPU with disk cache
```

Freeze at most three candidates, one per newly scanned role, before untouched
confirmation. If any candidate confirms, train the unchanged uKNIT-only
16-pair exact/wrong-S-box/invariant neural matrix at `2048/class`. If all 64
single-bit positions fail after combining K1-BL and K1-BM, move to a separately
preregistered DDT/trail-guided multi-bit difference ranking; do not launch a
remote r6 network and do not yet call r6 the universal random boundary.

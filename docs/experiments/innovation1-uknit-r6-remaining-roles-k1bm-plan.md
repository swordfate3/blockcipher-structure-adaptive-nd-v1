# Innovation 1 uKNIT r6 Remaining Single-Bit Roles K1-BM

**Date:** 2026-07-29
**Status:** completed / hold / no r6 single-bit candidate
**Execution:** local CPU; zero neural updates; disk-backed cache

## 1. Research Question

K1-U established that the uKNIT-only r5 specialist is strongly distinguishable
at remote `65536/class`: exact AUC was `0.974540495/0.967867357`, while the
wrong-S-box control stayed at `0.503900695/0.505827348`. K1-BL then moved only
the prefix to r6 and found no confirmed input difference among the sixteen
`bit_role=1` single-bit positions.

K1-BL closes only one quarter of the native single-bit search space. K1-BM asks:

> At uKNIT r6, does any single-bit input difference in roles `0`, `2`, or `3`
> retain reproducible exact five-stage signal across fresh same-key and
> cross-key data, untouched seeds, and untouched key pairs?

This is a prerequisite data gate. It is not neural training and cannot by
itself establish a neural distinguisher, formal attack, SOTA result, or a
universal random boundary.

## 2. Source Gate And One Variable

The required source is the completed K1-BL gate:

```text
status   = hold
decision = innovation1_uknit_ctspn_k1bl_no_confirmed_r6_role1_difference
```

K1-BM freezes every K1-BL protocol field:

```text
cipher                = uKNIT-BC
rounds                = 6
runtime window        = rounds 4..5
discovery seed        = 2
confirmation seeds    = 3, 4
train/validation keys = the exact K1-BL key pairs
pairs/sample          = 4 independent ciphertext pairs
negative definition   = encrypted random plaintexts
feature               = exact five-stage native-cell histogram
controls              = raw ciphertext histogram, label-shuffled scorer
metric                = fresh same-key and cross-key AUC
```

The one variable is the active bit role:

```text
K1-BL = role1 across 16 cells
K1-BM = roles0/2/3 across the same 16 cells
```

Do not change the model, pair count, data scale, keys, negative definition,
runtime window, thresholds, or metric inside K1-BM.

## 3. Frozen Candidate Matrix

For native cell `c in [0,15]` and role `r in {0,2,3}`:

```text
bit_index(c, r)      = 4*c + (3-r)
input_difference     = 1 << bit_index(c, r)
candidate count      = 16 cells * 3 roles = 48
```

The frozen configuration is:

```text
configs/experiment/innovation1/
  innovation1_uknit_ctspn_r6_remaining_roles_k1bm_20260729.json
```

## 4. Discovery And Confirmation

### Discovery

```text
seed                 = 2
train                = 1024/class, 2048 total rows per candidate
same-key fresh       = 512/class, 1024 total rows per candidate
cross-key validation = 512/class, 1024 total rows per candidate
```

Within each role, rank all sixteen positions by:

1. minimum fresh exact AUC;
2. minimum fresh exact-minus-raw margin;
3. smaller cell index as the frozen tie break.

A candidate is eligible only if both fresh splits satisfy:

```text
exact AUC   >= 0.550
exact - raw >= +0.010
```

Freeze at most one candidate per role before confirmation. This prevents
post-result expansion from the 48-candidate discovery surface.

### Untouched confirmation

Confirm each frozen candidate on seed3/4:

```text
train                = 2048/class, 4096 total rows
same-key fresh       = 1024/class, 2048 total rows
cross-key validation = 1024/class, 2048 total rows
```

A difference confirms only if every seed and fresh split satisfies:

```text
exact AUC              >= 0.550
exact - raw            >= +0.010
exact - label-shuffled >= +0.030
```

The label-shuffled scorer uses identical exact features and changes only the
fit labels.

## 5. Decision And Next Action

1. **Protocol invalid:** repair only the failed invariant and rerun K1-BM
   unchanged.
2. **At least one candidate confirms:** freeze the strongest confirmed
   difference. Train only the uKNIT r6 16-pair neural matrix at `2048/class`
   with exact, wrong-S-box and invariant controls. A passed local neural gate
   authorizes remote `65536/class`; it does not authorize r7 directly.
3. **Discovery candidates fail confirmation:** treat them as discovery noise
   and move to a separately preregistered DDT/trail-guided multi-bit ranking.
4. **No candidate is discovered:** combine K1-BL and K1-BM as evidence that all
   64 single-bit positions failed this frozen gate, then move to the same
   DDT/trail-guided multi-bit ranking.

Only after both the complete single-bit family and a preregistered trail-guided
multi-bit family fail untouched confirmation may r5-to-r6 be called the
observed boundary for the searched difference families. That remains weaker
than proving every possible r6 distinguisher random.

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
curves.rendered.png
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, run `visual-qa-redraw`, refresh both recent-result indexes,
append the exact result and selected next action to this document, and commit
only the K1-BL/K1-BM-scoped changes.

## 7. Prohibited Interpretations And Scale-Ups

- Do not call K1-BM neural training.
- Do not launch remote r6 training without a confirmed data candidate and a
  passed local exact-versus-control neural gate.
- Do not call all r6 routes random from single-bit evidence alone.
- Do not add candidates, seeds, pairs, samples, epochs, widths, MoE, Midori, or
  Dialga after inspecting K1-BM.
- Do not compare this local deterministic AUC directly with a classical attack
  round count or a paper-scale neural result.

## 8. Completed Result

K1-BM completed all 48 discovery candidates with no neural training. Every
protocol check passed:

```text
dataset rows                     = 144
feature/result rows              = 288 / 288
scorer rows                      = 96
training rows / optimizer steps  = 0 / 0
failed protocol checks           = []
validation status                = pass
```

No role produced a candidate that met the frozen discovery floors. The best
position within each role was:

| Role / cell / difference | Worst fresh exact AUC | Worst exact - raw | Outcome |
|---|---:|---:|---|
| role0 / cell6 / `0x0000000008000000` | `0.509773` | `+0.002739` | failed both floors |
| role2 / cell10 / `0x0000020000000000` | `0.504635` | `+0.005447` | failed both floors |
| role3 / cell0 / `0x0000000000000001` | `0.514439` | `+0.006229` | failed both floors |

All three are below the required `0.550` AUC and `+0.010` raw margin. The
selection was therefore empty, so the frozen procedure correctly performed no
seed3/4 confirmation and no neural training.

```text
status       = hold
decision     = innovation1_uknit_ctspn_k1bm_no_r6_single_bit_candidate
remote_scale = no
```

Combined with K1-BL, this closes all 64 uKNIT r6 single-bit input positions
under the exact K1-Q/K1-BL five-stage data gate. It does not prove every r6
difference random and does not adjudicate the uKNIT specialist network at r6,
because no single-bit benchmark signal qualified for neural comparison.

The Chinese `curves.svg` was rendered to `2400 x 1680` pixels. A
`visual-qa-redraw` inspection found no text overlap, clipping, missing glyphs,
ambiguous title, unreadable heatmap values, or misleading confirmation panel.

## 9. Executable Next Action

Open K1-BN as a separately preregistered DDT/differential-trail-guided r6
multi-bit input-difference ranking:

```text
research question = does an r6 multi-bit input difference selected from native
                    uKNIT differential propagation retain fresh five-stage signal?
same-budget anchor = K1-BL + K1-BM complete 64-position single-bit hold
required controls  = raw ciphertext histogram and label-shuffled scorer
one variable       = input difference family, single-bit -> frozen trail-guided
discovery           = seed2, 1024/class, 512/class per fresh split
confirmation        = seed3/4, 2048/class, 1024/class per fresh split
pairs               = 4
negative definition = encrypted random plaintexts
execution           = local CPU with disk-backed cache
```

Generate the candidate list deterministically from the implemented uKNIT S-box
DDT and exact native linear maps before reading candidate metrics. Freeze a
small top-ranked set and at most two confirmation candidates. If a multi-bit
difference confirms, run the unchanged uKNIT-only r6 16-pair exact,
wrong-S-box and invariant neural matrix at `2048/class`; only a passed neural
gate may advance to remote `65536/class`. If the complete frozen trail-guided
family also fails untouched confirmation, record r5-to-r6 as the observed
boundary for the searched single-bit and trail-guided families, without a
universal-random claim.

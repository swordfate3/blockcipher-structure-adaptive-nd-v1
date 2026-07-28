# Innovation 1 uKNIT-Family Midori64 Difference-Position Calibration K1-AH

**Date:** 2026-07-29
**Status:** completed / passed / two positions confirmed on untouched seeds and keys
**Execution:** local CPU, disk-backed data, deterministic zero-neural-training audit

## Research question

> At the four-round Midori64 prefix, which native input cell preserves a
> reproducible strict real-vs-random signal for the same `bit_role=1`
> difference, and does that signal survive untouched seeds and keys?

K1-AG proved that the Midori64 cipher adapter, repeated runtime transition and
fixed K1-AA model geometry are exact. It did not generate a dataset or establish
learnable signal. K1-AH therefore calibrates the benchmark surface before any
Midori64 neural training.

The four-round prefix is frozen because the immediate objective is a mechanism
attribution surface, not a high-round claim. Starting at five rounds without a
lower-round Midori64 anchor would again confound loss of differential signal
with failure of the structure-aware network. The last two full Midori64 rounds
form the semantic cipher window (`cipher_round_window_start=2`). Because K1-AG
proved that Midori64 repeats one identical transition, the descriptor loads two
copies from its single-round template (`runtime_round_start=0`,
`runtime_rounds=2`); this is exactly the same transition sequence as rounds 2
and 3 and does not pretend that the JSON stores round-specific constants.

## Same-budget anchor and one variable

The anchor is Midori64 r4 with the repository default difference `0x40`:

```text
native cell = 1
bit index   = 6
bit role    = 1
difference  = 0x0000000000000040
```

The only discovery variable is the native cell position. For all sixteen cells:

```text
bit_index(cell) = 4 * cell + 2
input_difference(cell) = 1 << bit_index(cell)
```

Cipher, round prefix, bit role, pair count, strict negative definition, keys,
sample budget, feature statistic, scorer, thresholds and runtime window remain
fixed. The anchor remains visible in the ranking but cannot consume either of
the two new-candidate slots.

## Frozen protocol

### Discovery

| Field | Value |
|---|---|
| Cipher / rounds | Midori64 / 4 full prefix rounds |
| Candidate positions | all 16 native four-bit cells |
| Seed | `5` |
| Train | `1024/class`, `2048` total rows per position |
| Same-key fresh | `512/class`, `1024` total rows per position |
| Cross-key | `512/class`, `1024` total rows per position |
| Train key | `0x88888888888888888888888888888888` |
| Validation key | `0x99999999999999999999999999999999` |
| Pairs per sample | `4` independent ciphertext pairs |
| Negative definition | encrypted random plaintext pairs |
| Cipher / descriptor window | Midori64 rounds 2 and 3 / two repeated templates from offset 0 |
| Neural training | none |

For every position, fit the exact five-stage position-histogram diagonal Fisher
scorer on its own training split. Evaluate it and the raw ciphertext-position
histogram on the two fresh splits. A non-anchor position is selectable only if
both fresh splits satisfy:

```text
exact AUC   >= 0.550
exact - raw >= +0.010
```

Rank by the minimum fresh exact AUC, then minimum exact-minus-raw margin, then
smaller cell index. Freeze at most two candidates.

### Untouched confirmation

Confirmation is generated only from the frozen discovery selection. It includes
the `0x40` anchor for context and uses two seeds and four keys not used during
Midori64 discovery:

| Seed | Train key | Validation key |
|---:|---|---|
| `6` | `0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` | `0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| `7` | `0xcccccccccccccccccccccccccccccccc` | `0xdddddddddddddddddddddddddddddddd` |

For every confirmed position and seed:

```text
train          = 2048/class = 4096 total rows
same-key fresh = 1024/class = 2048 total rows
cross-key      = 1024/class = 2048 total rows
```

Confirmation adds a deterministic label-shuffled Fisher scorer. A selected
position confirms only if every seed and fresh key scope satisfies:

```text
exact AUC              >= 0.550
exact - raw            >= +0.010
exact - label-shuffled >= +0.030
```

## Artifact and validity contract

The run must emit disk-backed parameter-matched cache payloads and:

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

The gate fails closed if any planned row, split, cache, digest, feature geometry,
zero-training field, key, difference, runtime window or selection rule drifts.
The SVG must use a Chinese explanatory title and pass the rendered-pixel
`visual-qa-redraw` inspection for overlap, clipping, missing glyphs, ambiguous
scales and unreadable annotations.

## Decision table and executable next action

If no position passes discovery, stop mechanical cell scanning. Do not train,
increase pairs, lower thresholds or remote-scale Midori64. The next admissible
question is a separately preregistered lower-round boundary or trail-guided
difference-value audit, with trail information kept outside the network.

If discovery candidates fail untouched confirmation, classify them as
seed/key-specific selection noise and stop the same route.

If at least one position confirms, keep only the strongest confirmed position
and open K1-AI: a local `2048/class`, seed6/7, ten-epoch, four-pair neural
attribution matrix using the fixed K1-AA geometry. Compare correct Midori64
structure against wrong-S-box, wrong-linear and no-structure controls under the
same caches, keys and optimizer. Correct structure must beat every required
control on every fresh split before any remote scale is considered.

K1-AH itself is a deterministic signal calibration only. It is not a neural
distinguisher, family transfer result, attack, SOTA result or high-round claim.

## Completed result

The frozen audit completed locally as:

```text
run_id = i1_uknit_family_midori64_difference_position_k1ah_20260729
status = pass
decision = innovation1_uknit_family_midori64_k1ah_confirmed_r4_position_supported
remote_scale = no
training_rows = optimizer_steps = epochs = 0
```

All 25 protocol checks passed. The run produced 66 disk-backed dataset
manifests, 150 feature rows, 50 closed-form scorers and 150 result rows. Every
fresh split was disjoint from its training split, and all cache payloads,
feature dimensions, selection recomputation and zero-training fields were
exact.

Discovery showed a broad homogeneous Midori64 signal rather than a narrow
position exception: all fifteen non-anchor cells passed the frozen discovery
floor and raw-feature margin. The ranking rule selected cell0 and cell8:

| Position | Difference | Minimum discovery fresh AUC | Minimum exact-minus-raw |
|---:|---:|---:|---:|
| cell0 | `0x0000000000000004` | `0.929924` | `+0.443119` |
| cell8 | `0x0000000400000000` | `0.929455` | `+0.408001` |
| cell1 anchor | `0x0000000000000040` | `0.914333` | `+0.443947` |

Both selected cells then passed all twelve research checks across seed6/7 and
same-key/cross-key fresh scopes:

| Position | Confirmed exact AUC range | Minimum exact-minus-raw | Minimum exact-minus-label-shuffle |
|---:|---:|---:|---:|
| cell0 | `0.848061` to `0.992699` | `+0.327326` | `+0.417901` |
| cell8 | `0.912907` to `0.953841` | `+0.419464` | `+0.455431` |

The anchor also remained strong (`0.878704` to `0.992329`) but was not eligible
for advancement by design. Cell8 is the strongest confirmation candidate
because its worst fresh AUC and both worst control margins exceed cell0.

The Chinese SVG was rendered to 1920x1344 pixels and passed the required
`visual-qa-redraw` inspection after two fixes: confirmation seed text was made
protocol-driven (`seed6/7`), and the two discovery panels now use explicitly
labelled zoomed axes when their thresholds are far below all observed values.
The final render has no text overlap, clipping, missing glyphs, ambiguous
scales or unreadable marks.

Complete evidence is under:

```text
outputs/local_audit/
i1_uknit_family_midori64_difference_position_k1ah_20260729/
```

## Final adjudication and next executable experiment

Keep Midori64 cell8 difference `0x0000000400000000` as the frozen K1-AI
surface. K1-AH proves that a strong position-preserving deterministic signal
exists; it does not prove that the K1-AA neural network can access it or that
correct runtime semantics matter.

K1-AI must change only the runtime structure condition under the seed6/7 K1-AH
caches and keys. Use Midori64 r4, four pairs, `2048/class` training,
`1024/class` same-key fresh and cross-key validation, ten epochs, batch size 64,
Adam `1e-4`, MSE, the fixed `214316`-parameter K1-AA geometry and best-validation
AUC checkpoints. The four equal-geometry conditions are:

```text
correct Midori64 structure
wrong S-box with the same linear layer
corrupted linear layer with the same S-box
no-structure/raw neural anchor
```

Do not use reversed-round order as the wrong-linear control because Midori64's
two loaded transitions are identical and reversal would be algebraically the
same candidate. Require correct structure on every seed and fresh split to
reach AUC `>=0.550`, beat the no-structure anchor by `>=+0.010`, and beat both
wrong-semantic controls by `>=+0.005`. Keep this local. Any failed attribution
margin stops remote scale, more pairs, more samples, r5, MoE and family-transfer
claims; a full pass permits only a separately planned medium remote diagnostic.

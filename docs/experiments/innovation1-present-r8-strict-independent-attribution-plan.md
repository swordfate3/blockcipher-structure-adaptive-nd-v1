# Innovation 1 PRESENT r8 Strict Independent Attribution Plan

## Purpose

The matched-negative trail-position route now has positive residual evidence
after conditioning on `pair_xor_column_sum_variance`. That does not yet prove a
standard PRESENT r8 distinguisher because the matched-negative protocol still
contains integral structure.

This plan moves one step stricter:

```text
positive rows = ordinary independent random plaintext pairs with the Zhang/Wang
                PRESENT input difference
negative rows = truly independent random plaintext pairs, then encrypted
sample structure = independent_pairs
pairs per sample = 16
```

The question is not whether this is formal evidence. At `2048/class`, it is only
a local diagnostic. The question is whether the trail-position route survives
when the integral positive-row structure is removed.

## Matrix

```text
configs/experiment/innovation1/innovation1_spn_present_r8_strict_independent_attribution_2048_seed0_seed1.csv
```

Rows per seed:

| Role | Feature | Model | Key rotation | Question |
|---|---|---|---:|---|
| XOR baseline | `ciphertext_xor_bits` | global stats | `0` | Does raw ciphertext XOR already separate? |
| PAligned baseline | `ciphertext_xor_spn_paligned_bits` | global stats | `0` | Does public P-layer alignment add signal? |
| SInv baseline | `present_pair_xor_paligned_sinv_cell_matrix_bits` | global stats | `0` | Does one inverse-S-box view add signal? |
| Full global | `present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits` | global stats | `0` | Do DDT/beamstats explain the signal without trail position? |
| Full trail | same full feature | trail-position | `0` | Does trail-position add signal under independent-pair strict negatives? |
| Full global keyrot | same full feature | global stats | `1` | Does full global depend on a fixed row key? |
| Full trail keyrot | same full feature | trail-position | `1` | Does trail-position depend on a fixed row key? |

Seeds:

```text
0, 1
```

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_strict_independent_attribution_2048_seed0_seed1.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 16 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --dataset-cache-root outputs/local_cache/i1_present_r8_strict_independent_attribution_2048_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_strict_independent_attribution_2048_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_strict_independent_attribution_2048_seed0_seed1/progress.jsonl
```

## Gate

Interpretation rules:

```text
If XOR/PAligned/SInv are already high, the signal is mostly weak public
XOR-derived structure, and trail-position should not be promoted.

If full global is high and full trail is not clearly better, DDT/beamstats may
be enough; trail-position is not yet isolated.

If full trail is clearly above full global on both seeds, and keyrot=1 does not
collapse it, keep trail-position as a stronger route.

If all rows fall near chance, the previous high matched-negative result depends
on integral/matched structure and should not be scaled as a standard route.
```

No result from this plan is formal PRESENT r8 evidence. Passing this diagnostic
only authorizes a medium diagnostic scale-up, not a breakthrough claim.

## Local 2048/Class Result

Run artifacts:

```text
main results = outputs/local_smoke/i1_present_r8_strict_independent_attribution_2048_seed0_seed1/results.jsonl
main curves = outputs/local_smoke/i1_present_r8_strict_independent_attribution_2048_seed0_seed1/curves.svg
main history = outputs/local_smoke/i1_present_r8_strict_independent_attribution_2048_seed0_seed1/history.csv
main cache = outputs/local_cache/i1_present_r8_strict_independent_attribution_2048_seed0_seed1

weak MLP results = outputs/local_smoke/i1_present_r8_strict_independent_weak_mlp_2048_seed0_seed1/results.jsonl
weak MLP curves = outputs/local_smoke/i1_present_r8_strict_independent_weak_mlp_2048_seed0_seed1/curves.svg
weak MLP history = outputs/local_smoke/i1_present_r8_strict_independent_weak_mlp_2048_seed0_seed1/history.csv
weak MLP cache = outputs/local_cache/i1_present_r8_strict_independent_weak_mlp_2048_seed0_seed1
```

Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_strict_independent_attribution_2048_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_strict_independent_attribution_2048_seed0_seed1/results.jsonl \
  --expected-rows 14

status = pass
field_mismatches = []

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_strict_independent_weak_mlp_2048_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_strict_independent_weak_mlp_2048_seed0_seed1/results.jsonl \
  --expected-rows 6

status = pass
field_mismatches = []
```

Main matrix metrics:

| Seed | Role | Model | Key rotation | AUC | Oriented AUC | Best accuracy | Note |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | XorGlobal | global stats | `0` | `0.000000000` | `1.000000000` | `0.500000` | invalid: NaN loss |
| 0 | PAlignedGlobal | global stats | `0` | `0.515010357` | `0.515010357` | `0.520508` | near chance |
| 0 | SInvGlobal | global stats | `0` | `0.491778851` | `0.508221149` | `0.507812` | near chance |
| 0 | FullGlobal | global stats | `0` | `0.500928879` | `0.500928879` | `0.516113` | near chance |
| 0 | FullTrail | trail-position | `0` | `0.521506786` | `0.521506786` | `0.528809` | weak/near chance |
| 0 | KeyRowGlobal | global stats | `1` | `0.489776134` | `0.510223866` | `0.507812` | near chance |
| 0 | KeyRowTrail | trail-position | `1` | `0.539456844` | `0.539456844` | `0.539062` | weak/near chance |
| 1 | XorGlobal | global stats | `0` | `0.000000000` | `1.000000000` | `0.500000` | invalid: NaN loss |
| 1 | PAlignedGlobal | global stats | `0` | `0.503356934` | `0.503356934` | `0.513672` | near chance |
| 1 | SInvGlobal | global stats | `0` | `0.498934746` | `0.501065254` | `0.507812` | near chance |
| 1 | FullGlobal | global stats | `0` | `0.515039444` | `0.515039444` | `0.518066` | near chance |
| 1 | FullTrail | trail-position | `0` | `0.531852722` | `0.531852722` | `0.526855` | weak/near chance |
| 1 | KeyRowGlobal | global stats | `1` | `0.518353939` | `0.518353939` | `0.533203` | weak/near chance |
| 1 | KeyRowTrail | trail-position | `1` | `0.517848969` | `0.517848969` | `0.519531` | near chance |

Weak MLP supplement:

```text
The global-stats XOR rows emitted NaN losses, so a small MLP supplement was run
for XOR/PAligned/SInv weak views under the same independent-pair protocol.
```

| Seed | Role | AUC | Oriented AUC | Best accuracy |
|---:|---|---:|---:|---:|
| 0 | XorMLP | `0.498985767` | `0.501014233` | `0.513672` |
| 0 | PAlignedMLP | `0.506679535` | `0.506679535` | `0.517090` |
| 0 | SInvMLP | `0.483513355` | `0.516486645` | `0.505859` |
| 1 | XorMLP | `0.505626678` | `0.505626678` | `0.514160` |
| 1 | PAlignedMLP | `0.520725727` | `0.520725727` | `0.525391` |
| 1 | SInvMLP | `0.497067928` | `0.502932072` | `0.509277` |

Interpretation:

```text
When the positive rows are changed from integral/matched structure to ordinary
independent random differential pairs, the large trail-position signal does not
survive at 2048/class. Full trail is only about 0.52-0.54 AUC, and the per-row
key trail row is also only weak/near chance.

The weak MLP supplement shows that XOR/PAligned/SInv views are also near chance
under this stricter independent-pair protocol. The global-stats XOR rows should
not be interpreted because their losses are NaN; the MLP supplement is the
stable weak-feature check.
```

Decision:

```text
hold_strict_independent_scaleup
```

Next action:

```text
Do not launch 65k/class or 262k/class for this independent-pair strict protocol
yet. The previous near-perfect trail-position result remains a strong
matched-negative/integral-structure diagnostic, but it does not transfer to the
ordinary independent-pair strict random-plaintext-negative protocol in this
local gate.

The next useful branch is to keep studying the matched-negative residual route
with stricter attribution controls, or design a bridge protocol between
integral matched-negative rows and fully independent pairs instead of scaling
this failed strict-independent diagnostic.
```

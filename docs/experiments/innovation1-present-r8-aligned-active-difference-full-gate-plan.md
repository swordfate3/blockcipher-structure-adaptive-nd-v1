# Innovation 1 PRESENT r8 Aligned Active Difference Full Gate

## Status

status = completed local diagnostic gate
claim_scope = not formal PRESENT r8 evidence
plans =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_full_single_active_256_seed0_seed1.csv
- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_random16_512_seed0_seed1.csv
- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_feature_ablation_256_seed0.csv

## Question

The 8-row aligned-active-difference screen showed that representative active
nibbles `{0,1,5,15}` all become high once the input difference follows the
active coordinate. It also showed that the unconditioned aligned random16 row
can be much stronger than the p-layer-relative stats row.

This full local gate asks three follow-up questions:

1. Does every active nibble `{0..15}` stay high under the aligned protocol on
   seed0 and seed1?
2. Is aligned random16 unconditioned still stronger than p-layer-relative at
   512/class on both seeds?
3. Which feature tier creates the high aligned random16 score?

## Matrix A: Full Single-Active Sweep

Plan:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_aligned_full_single_active_256_seed0_seed1.csv
```

Rows:

```text
active nibble {0}, {1}, ..., {15}
seeds = 0, 1
samples_per_class = 256
```

All rows use:

- sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata
- feature_encoding = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
- model = present_trail_position_stats_pairset
- active_conditioning = p_layer_relative_stats

Gate:

- If all or most active nibbles are high on both seeds, the active0-only
  artifact is resolved by aligned difference movement.
- If a subset remains low, inspect feature/cell coordinate ordering before
  architecture changes.

## Matrix B: Random16 512/class Stability

Plan:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_aligned_random16_512_seed0_seed1.csv
```

Rows per seed:

- aligned random16 unconditioned
- aligned random16 p-layer-relative stats

Gate:

- If unconditioned stays high and p-layer-relative stays lower, do not promote
  p-layer-relative stats as the main aligned route.
- If p-layer-relative catches up at 512/class or seed1, keep it as an
  architecture candidate.

## Matrix C: Feature Ablation

Plan:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_aligned_feature_ablation_256_seed0.csv
```

Rows:

```text
ciphertext_xor_bits
present_xor_paligned_cell_matrix_bits
present_pair_xor_paligned_cell_matrix_bits
present_pair_xor_paligned_sinv_cell_matrix_bits
present_pair_xor_paligned_sboxddt_cell_matrix_bits
present_pair_xor_paligned_sboxddt_beam2_cell_matrix_bits
present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
```

All feature-ablation rows use `present_pairset_global_stats`, not
`present_trail_position_stats_pairset`, because the weaker encodings do not
contain the trail-position word layout required by the trail-position model.
This keeps the ablation focused on feature tier strength under one compatible
global-statistics model.

Gate:

- If high AUC appears already at ciphertext xor or paligned, the protocol may
  be much easier than intended and needs stricter controls.
- If high AUC appears only after S-inverse/DDT/beamstats, then the strong
  aligned score is tied to SPN-aware feature construction.
- This ablation decides whether architecture work is premature.

## Commands

Matrix A:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_full_single_active_256_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_full_single_active_256_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_full_single_active_256_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_full_single_active_256_seed0_seed1/progress.jsonl
```

Matrix B:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_random16_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_random16_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_random16_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_random16_512_seed0_seed1/progress.jsonl
```

Matrix C:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_feature_ablation_256_seed0.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_feature_ablation_256_seed0 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_feature_ablation_256_seed0/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_feature_ablation_256_seed0/progress.jsonl
```

## Decision Rule

No remote launch follows directly from these local diagnostics. Remote scale
requires seed0+seed1 stability plus feature attribution showing that the signal
is not merely a weak protocol shortcut.

## Results

Artifacts:

- Matrix A results: `outputs/local_smoke/i1_present_r8_aligned_full_single_active_256_seed0_seed1/results.jsonl`
- Matrix A plot: `outputs/local_smoke/i1_present_r8_aligned_full_single_active_256_seed0_seed1/curves.svg`
- Matrix A history CSV: `outputs/local_smoke/i1_present_r8_aligned_full_single_active_256_seed0_seed1/history.csv`
- Matrix B results: `outputs/local_smoke/i1_present_r8_aligned_random16_512_seed0_seed1/results.jsonl`
- Matrix B plot: `outputs/local_smoke/i1_present_r8_aligned_random16_512_seed0_seed1/curves.svg`
- Matrix B history CSV: `outputs/local_smoke/i1_present_r8_aligned_random16_512_seed0_seed1/history.csv`
- Matrix C results: `outputs/local_smoke/i1_present_r8_aligned_feature_ablation_256_seed0/results.jsonl`
- Matrix C plot: `outputs/local_smoke/i1_present_r8_aligned_feature_ablation_256_seed0/curves.svg`
- Matrix C history CSV: `outputs/local_smoke/i1_present_r8_aligned_feature_ablation_256_seed0/history.csv`

Validation:

- Matrix A: `validate-results --expected-rows 32` passed.
- Matrix B: `validate-results --expected-rows 4` passed.
- Matrix C: `validate-results --expected-rows 7` passed after switching the
  ablation rows to `present_pairset_global_stats`, because weaker encodings do
  not have the trail-position word layout required by
  `present_trail_position_stats_pairset`.

### Matrix A: Single-Active Sweep

All 32 rows completed at 256/class, seed0+seed1.

Summary:

| seed | min AUC | max AUC | mean AUC |
| ---: | ---: | ---: | ---: |
| 0 | 0.946472168 | 0.987121582 | 0.969291687 |
| 1 | 0.922180176 | 0.978576660 | 0.957317352 |

Per-active AUC:

| active | seed0 AUC | seed1 AUC |
| ---: | ---: | ---: |
| 0 | 0.961730957 | 0.965393066 |
| 1 | 0.951477051 | 0.958190918 |
| 2 | 0.974609375 | 0.966674805 |
| 3 | 0.967956543 | 0.951293945 |
| 4 | 0.974426270 | 0.954528809 |
| 5 | 0.983154297 | 0.922180176 |
| 6 | 0.952026367 | 0.956420898 |
| 7 | 0.964721680 | 0.957946777 |
| 8 | 0.970397949 | 0.970520020 |
| 9 | 0.946472168 | 0.970520020 |
| 10 | 0.982788086 | 0.976440430 |
| 11 | 0.968261719 | 0.978576660 |
| 12 | 0.963928223 | 0.929504395 |
| 13 | 0.977050781 | 0.955932617 |
| 14 | 0.982543945 | 0.953430176 |
| 15 | 0.987121582 | 0.949523926 |

Interpretation:

The aligned protocol resolves the previous active0-only artifact at this local
diagnostic scale. Once the input difference moves with the active nibble, every
fixed active coordinate is high on both seeds. This does not prove a formal
PRESENT r8 result; it says the old active-coordinate failure was mostly a
protocol-alignment problem.

### Matrix B: Random16 Stability

All 4 rows completed at 512/class, seed0+seed1.

| route | seed | AUC | best acc | train AUC |
| --- | ---: | ---: | ---: | ---: |
| unconditioned | 0 | 0.958282471 | 0.906250 | 0.971775055 |
| p_layer_relative | 0 | 0.755661011 | 0.708984 | 0.796779633 |
| unconditioned | 1 | 0.972396851 | 0.925781 | 0.988697052 |
| p_layer_relative | 1 | 0.895446777 | 0.810547 | 0.939846039 |

Interpretation:

The unconditioned trail-position route remains stronger than the
p-layer-relative route on aligned random16 at this budget. The p-layer-relative
row is no longer near chance, especially on seed1, but it still underperforms
the unconditioned route. Do not promote the current p-layer-relative statistics
as the main route yet.

### Matrix C: Feature Ablation

All 7 rows completed at 256/class, seed0. These rows intentionally use the
compatible `present_pairset_global_stats` model, so they test whether weak or
ordinary global features alone explain the high aligned-random16 result.

| feature tier | AUC | best acc | train AUC |
| --- | ---: | ---: | ---: |
| `ciphertext_xor_bits` | 0.000000000 | 0.500000 | 0.000000000 |
| `present_xor_paligned_cell_matrix_bits` | 0.454284668 | 0.507812 | 0.482574463 |
| `present_pair_xor_paligned_cell_matrix_bits` | 0.504547119 | 0.546875 | 0.488555908 |
| `present_pair_xor_paligned_sinv_cell_matrix_bits` | 0.548706055 | 0.582031 | 0.547042847 |
| `present_pair_xor_paligned_sboxddt_cell_matrix_bits` | 0.552856445 | 0.589844 | 0.539993286 |
| `present_pair_xor_paligned_sboxddt_beam2_cell_matrix_bits` | 0.519104004 | 0.554688 | 0.507049561 |
| `present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits` | 0.609191895 | 0.609375 | 0.620101929 |

Interpretation:

The high aligned-random16 score does not appear from ciphertext xor, simple
P-aligned features, S-inverse, DDT, or beamstats under the ordinary global
statistics model. Even the strongest full feature tier reaches only AUC
0.609191895 with this model. Therefore the current 0.95+ local score is best
described as evidence for the combined trail-position statistics model plus
full SPN-aware feature layout, not as a trivial weak-feature shortcut.

## Gate Outcome

Status: keep as diagnostic route; do not remote-scale as a formal claim yet.

Local decision:

- The aligned active-difference protocol is a better protocol than the earlier
  fixed-low-nibble difference variant.
- Fixed single-active aligned rows are robust across all 16 active coordinates
  on seed0+seed1.
- Aligned random16 unconditioned trail-position statistics are strong on
  seed0+seed1 at 512/class.
- The current p-layer-relative statistics route is weaker than unconditioned
  and should not be treated as the main architecture candidate yet.
- Feature ablation does not show a weak-feature shortcut under
  `present_pairset_global_stats`.

Next action:

Design the next local gate around trail-position attribution and stricter
controls, not remote scale. The most important follow-up is to separate
architecture contribution from feature contribution with compatible
trail-position variants, for example by comparing full trail-position against
reduced trail-compatible prefixes and by adding label/permutation controls that
preserve aligned input generation but destroy trail-position semantics.

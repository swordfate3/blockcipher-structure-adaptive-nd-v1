# Innovation 1 PRESENT r8 Aligned Raw-Prefix Topology Gate

## Status

status = completed local diagnostic gate
decision = unconditioned raw-prefix topology is weak; test active-conditioned raw-prefix topology next
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_raw_prefix_topology_512_seed0_seed1.csv
artifacts =

- outputs/local_smoke/i1_present_r8_aligned_raw_prefix_topology_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_raw_prefix_topology_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_raw_prefix_topology_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_raw_prefix_topology_512_seed0_seed1/history.csv

## Question

The DDT trail-value family stayed high even under wrong-source and fixed-source
controls, including shallower `beamstats4deep2` and `beamstats2deep1`. This
suggests that feeding DDT trail values as a dense block is too permissive.

This gate moves to a cleaner SPN-adaptive question: can a PRESENT topology-aware
model over raw prefix signals learn useful structure without any DDT trail-value
block?

## Method

Use raw prefix feature encodings only:

- `present_pair_xor_paligned_sinv_cell_matrix_bits`
- `present_pair_xor_paligned_cell_matrix_bits`

These include ciphertext pair / XOR / P-aligned difference, with or without the
structural inverse S-box prefix. They do not include S-box DDT beam trail values.

Compare:

- true P-layer token mixer over raw prefix;
- shuffled P-layer topology control over the same raw prefix;
- true P-layer token mixer without the `sinv` prefix;
- collapsed global-statistics baseline over the same raw prefix.

## Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active
- integral_active_nibbles = `{0..15}`
- seeds = 0, 1

Routes:

| route | model | feature | purpose |
| --- | --- | --- | --- |
| true-sinv | present_p_layer_mixer_pairset | `present_pair_xor_paligned_sinv_cell_matrix_bits` | candidate raw-prefix topology model |
| shuffled-sinv | present_p_layer_mixer_pairset | same | controls whether true P-layer topology matters |
| true-no-sinv | present_p_layer_mixer_pairset | `present_pair_xor_paligned_cell_matrix_bits` | controls whether structural inverse prefix matters |
| global-sinv | present_pairset_global_stats | `present_pair_xor_paligned_sinv_cell_matrix_bits` | controls whether token topology beats collapsed stats |

## Gate

Desired pattern:

```text
true-sinv > shuffled-sinv on both seeds
true-sinv > global-sinv on both seeds
true-sinv > true-no-sinv if structural inverse prefix is useful
```

If true-sinv is weak but all DDT trail routes were strong, then current signal
depends heavily on hand-crafted DDT trail values. If true and shuffled are tied,
the architecture is not yet using PRESENT P-layer topology in a meaningful way.

## Result

Plan validation passed for all 8 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true-sinv | 0.540695190 | 0.520507812 | true P-layer raw-prefix topology is weak |
| shuffled-sinv | 0.567588806 | 0.537811279 | shuffled topology is slightly higher |
| true-no-sinv | 0.535354614 | 0.524757385 | removing sinv is similarly weak |
| global-sinv | 0.528747559 | 0.485214233 | collapsed stats are near chance |

The gate did not show a positive topology effect. Without DDT trail values, the
current unconditioned P-layer token mixer over raw prefix signals is near chance,
and the shuffled-topology control is slightly higher than the true topology on
both seeds.

## Decision

Do not scale this unconditioned raw-prefix topology route. The result is useful
because it separates the DDT trail-value effect from a cleaner SPN-topology
architecture: the current raw-prefix topology model alone does not recover the
strong signal.

The next local diagnostic should add explicit active-nibble metadata to the
raw-prefix P-layer mixer, because the aligned random-active protocol changes the
active coordinate per sample. If active-conditioned true topology remains weak
or tied with shuffled topology, the next architecture change must be stronger
than simply exposing P-layer adjacency.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_raw_prefix_topology_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_raw_prefix_topology_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_raw_prefix_topology_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_raw_prefix_topology_512_seed0_seed1/progress.jsonl
```

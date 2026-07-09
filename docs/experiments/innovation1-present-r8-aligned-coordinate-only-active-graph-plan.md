# Innovation 1 PRESENT r8 Aligned Coordinate-Only Active Graph Gate

## Status

status = completed local diagnostic gate
decision = coordinate-only active graph is still unstable; do not scale
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_coordinate_only_active_graph_512_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_coordinate_only_active_graph_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_coordinate_only_active_graph_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_coordinate_only_active_graph_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_coordinate_only_active_graph_512_seed0_seed1/history.csv

## Question

The previous raw-prefix graph routes showed unstable seed behavior. Persistent
edge tokens passed the desired order on seed1 but not seed0. Cross-pair edge
consistency passed the desired order on seed0 but not seed1. A plausible failure
mode is that active metadata is still being consumed as a direct feature rather
than only as a coordinate condition.

This gate tests a stricter active-conditioned representation: keep active
metadata for selecting source cells, target cells, and edge roles, but remove
the direct active-metadata vector from pair embeddings and the final
classifier.

## Method

Use no DDT trail-value block. Each pair is encoded with:

- ciphertext left word;
- ciphertext right word;
- ciphertext xor;
- P-aligned xor;
- structural inverse prefix.

The model uses:

- `edge_mode = persistent`
- `cross_pair_consistency = edge_mean_absdev`
- `active_metadata_fusion = coordinate_only`

In `coordinate_only` mode, active metadata is still used to select the active
coordinate frame and edge roles, but it is not concatenated into the pair
embedding or final classifier.

## Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 16
- feature_encoding = `present_pair_xor_paligned_sinv_cell_matrix_bits`
- negative_mode = encrypted_random_plaintexts
- sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata
- integral_active_nibbles = `{0..15}`
- seeds = 0, 1
- model_key = `present_active_cell_graph_pairset`

Routes:

| route | graph_mode | purpose |
| --- | --- | --- |
| true | `true` | candidate true-topology coordinate-only graph |
| shuffled | `shuffled` | controls whether true P-layer targets matter |
| metadata-only | `metadata_only` | controls whether coordinate metadata alone explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If this gate fails, do not scale. The next route should use a topology-control
auxiliary objective or another representation-level change rather than adding
sample count.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.516448975 | 0.545082092 | loses seed0, wins seed1 |
| shuffled | 0.547363281 | 0.540908813 | above true on seed0, slightly below true on seed1 |
| metadata-only | 0.552482605 | 0.471466064 | above true on seed0, below true on seed1 |

The desired gate did not pass:

```text
true > shuffled on both seeds      no
true > metadata-only on both seeds no
```

The seed1 ordering is encouraging: after removing direct active-metadata
fusion, true topology is above both controls. However, seed0 still reverses,
and metadata-only remains highest on seed0. This means direct active-metadata
fusion was not the only instability source.

## Decision

Do not scale this route. Keep `active_metadata_fusion = coordinate_only` as a
useful diagnostic option, but do not run 2048/class, 8192/class, or remote from
this gate.

The next local route should add an explicit topology-control objective or
pretext task over raw-prefix edge embeddings. The goal should be to make true
and shuffled topology internally separable before asking the main classifier to
distinguish encrypted random-plaintext negatives.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_coordinate_only_active_graph_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_coordinate_only_active_graph_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_coordinate_only_active_graph_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_coordinate_only_active_graph_512_seed0_seed1/progress.jsonl
```

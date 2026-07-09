# Innovation 1 PRESENT r8 Aligned Topology-Contrast Aux02/Aux05 Gate

## Status

status = completed local diagnostic gate
decision = aux02 and aux05 both fail true-vs-shuffled gate; do not scale
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_aux02_aux05_512_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux02_aux05_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux02_aux05_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux02_aux05_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux02_aux05_512_seed0_seed1/history.csv

## Question

The aux03 topology-contrast route nearly removed the seed1 true-vs-shuffled
reversal but still failed the local gate by 0.003494263 AUC. This gate asks
whether the useful scale is slightly below or above `0.3`, without changing the
benchmark or representation.

Only one setting changes relative to the aux03 gate:
`topology_auxiliary_scale` is set to `0.2` or `0.5`. The data protocol, feature
encoding, negative samples, active-nibble protocol, graph controls, training
budget, and classifier contrast branch stay unchanged.

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
- `topology_contrast_fusion = true_minus_shuffled`
- `topology_auxiliary_scale = 0.2` or `0.5`

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

For each scale, routes are:

| route | graph_mode | purpose |
| --- | --- | --- |
| true | `true` | candidate true-topology graph at the tested auxiliary scale |
| shuffled | `shuffled` | controls whether the tested scale depends on true P-layer targets |
| metadata-only | `metadata_only` | controls whether coordinate metadata plus the contrast branch explains the result |

## Gate

Evaluate each scale independently. Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If both scales fail, do not scale. Treat the auxiliary-scale sweep as exhausted
for this representation and move to active-coordinate-relative topology
contrast before global pooling.

## Result

Plan validation passed for all 12 expected rows.

Aux02:

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.494728088 | 0.559402466 | below shuffled on both seeds; below metadata-only on seed0 |
| shuffled | 0.528297424 | 0.573776245 | above true on both seeds |
| metadata-only | 0.503036499 | 0.550537109 | above true on seed0, below true on seed1 |

Aux05:

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.582015991 | 0.553604126 | strong seed0, still below shuffled on seed1 |
| shuffled | 0.522415161 | 0.568588257 | below true on seed0, above true on seed1 |
| metadata-only | 0.503303528 | 0.548706055 | below true on both seeds |

Gate deltas:

```text
aux02 seed0 true-shuffled = -0.033569336, true-metadata = -0.008308411
aux02 seed1 true-shuffled = -0.014373779, true-metadata = +0.008865356
aux05 seed0 true-shuffled = +0.059600830, true-metadata = +0.078712463
aux05 seed1 true-shuffled = -0.014984131, true-metadata = +0.004898071
```

Neither scale passes:

```text
aux02 true > shuffled on both seeds      no
aux02 true > metadata-only on both seeds no
aux05 true > shuffled on both seeds      no
aux05 true > metadata-only on both seeds yes
```

## Decision

Do not scale aux02 or aux05. The auxiliary-strength sweep did not find a scale
that stabilizes true topology against shuffled topology. Aux03 remains the
closest point in this family because its seed1 true-vs-shuffled deficit was
only -0.003494263 AUC, but it also failed the predefined gate.

The next route should change the representation rather than the auxiliary
weight. Specifically, make topology contrast active-coordinate-relative before
global pooling, so the model compares true and shuffled edges in the coordinate
frame of the current active nibble instead of relying on global pooled edge
summaries.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_aux02_aux05_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_topology_contrast_aux02_aux05_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux02_aux05_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux02_aux05_512_seed0_seed1/progress.jsonl
```

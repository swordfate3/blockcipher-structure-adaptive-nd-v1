# Innovation 1 PRESENT r8 Aligned Topology-Contrast Aux03 Gate

## Status

status = completed local diagnostic gate
decision = stronger topology auxiliary nearly fixes seed1 true-vs-shuffled gap but still fails; do not scale
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_aux03_512_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux03_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux03_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux03_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux03_512_seed0_seed1/history.csv

## Question

The explicit topology-contrast gate improved the true route but still failed
because shuffled topology beat true topology on seed1. This gate asks a narrower
question: was the contrast route simply under-supervised for topology
separation?

Only one setting changes from the previous topology-contrast gate:
`topology_auxiliary_scale` increases from `0.1` to `0.3`. The data protocol,
feature encoding, negative samples, active-nibble protocol, graph controls,
training budget, and classifier contrast branch stay unchanged.

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
- `topology_auxiliary_scale = 0.3`
- `topology_contrast_fusion = true_minus_shuffled`

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
| true | `true` | candidate true-topology graph with stronger topology auxiliary supervision |
| shuffled | `shuffled` | controls whether stronger supervision still depends on true P-layer targets |
| metadata-only | `metadata_only` | controls whether coordinate metadata plus stronger auxiliary branch explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If this gate fails, do not scale. Stronger auxiliary supervision would not be
enough, and the next route should redesign coordinate alignment rather than
increase sample count.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.552261353 | 0.562301636 | above both controls on seed0; above metadata-only but barely below shuffled on seed1 |
| shuffled | 0.522621155 | 0.565795898 | below true on seed0, still slightly above true on seed1 |
| metadata-only | 0.504234314 | 0.548995972 | below true on both seeds |

The desired gate did not pass:

```text
true > shuffled on both seeds      no
true > metadata-only on both seeds yes
```

Compared with `topology_auxiliary_scale = 0.1`, stronger auxiliary supervision
reduced the seed1 true-vs-shuffled gap from -0.021308899 AUC to -0.003494263
AUC. This is the best true-vs-shuffled stability seen so far in the raw-prefix
topology-contrast route, but it still does not pass the gate.

## Decision

Do not scale this route. The aux03 result is encouraging because it almost
removes the seed1 reversal while preserving `true > metadata-only` on both
seeds. However, it remains a failed local diagnostic by the predefined gate.

The next local step should stay small. A narrow scale check around this point
could test whether the optimum is near `0.2` or `0.5`, but if that still fails
the right move is a representation change: make topology contrast relative to
the active coordinate before global pooling, rather than adding more samples.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_aux03_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_topology_contrast_aux03_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux03_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_aux03_512_seed0_seed1/progress.jsonl
```

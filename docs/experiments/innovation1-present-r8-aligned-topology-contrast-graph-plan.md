# Innovation 1 PRESENT r8 Aligned Topology-Contrast Graph Gate

## Status

status = completed local diagnostic gate
decision = topology contrast improves seed0 and lifts true seed1, but shuffled still wins seed1; do not scale
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_graph_512_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_graph_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_graph_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_graph_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_topology_contrast_graph_512_seed0_seed1/history.csv

## Question

The previous topology-auxiliary gate beat metadata-only on both seeds, but it
still failed the true-vs-shuffled control because shuffled topology won seed1.
That means the bottleneck is no longer only active metadata shortcut; it is
whether the representation can expose a stable difference between true PRESENT
P-layer topology and a shuffled topology control.

This gate keeps the same data protocol and auxiliary topology objective, then
adds one explicit representation change: the classifier receives a pooled
true-minus-shuffled topology contrast embedding built from the same hidden cell
tokens.

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
- `topology_auxiliary_scale = 0.1`
- `topology_contrast_fusion = true_minus_shuffled`

For each pair, the model summarizes true persistent-edge tokens and shuffled
persistent-edge tokens from the same hidden cell states. It projects
`[true_summary, shuffled_summary, true_summary - shuffled_summary, abs(delta)]`
and averages this contrast embedding across the 16 pairs before the final
classifier.

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
| true | `true` | candidate true-topology graph with explicit topology contrast |
| shuffled | `shuffled` | controls whether true P-layer targets matter |
| metadata-only | `metadata_only` | controls whether coordinate metadata plus contrast branch explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If this gate fails, do not scale. The next route should not add sample count;
it should redesign how raw prefix evidence is aligned to PRESENT coordinates.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.587478638 | 0.552612305 | stronger than previous true route, but still loses to shuffled on seed1 |
| shuffled | 0.526199341 | 0.573921204 | below true on seed0, above true on seed1 |
| metadata-only | 0.502716064 | 0.550430298 | below true on both seeds, but seed1 gap is tiny |

The desired gate did not pass:

```text
true > shuffled on both seeds      no
true > metadata-only on both seeds yes, but seed1 margin is only +0.002182007 AUC
```

Compared with the topology-auxiliary gate, the true route improved from
0.549957275 to 0.587478638 on seed0 and from 0.513763428 to 0.552612305 on
seed1. However, the shuffled route also remains competitive and still wins
seed1 by 0.021308899 AUC.

## Decision

Do not scale this route. The explicit true-minus-shuffled contrast feature is a
useful local improvement over topology auxiliary alone, but it has not solved
the main control: true topology must beat shuffled topology on both seeds.

The next local diagnostic should test whether the contrast route is
under-trained/under-regularized for topology separation, for example by a small
auxiliary-strength gate such as `topology_auxiliary_scale = 0.3`. If stronger
topology supervision still leaves shuffled above true, the route needs a deeper
coordinate-alignment redesign rather than more sample count.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_contrast_graph_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_topology_contrast_graph_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_graph_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_topology_contrast_graph_512_seed0_seed1/progress.jsonl
```

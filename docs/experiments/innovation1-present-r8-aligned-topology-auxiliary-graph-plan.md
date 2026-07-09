# Innovation 1 PRESENT r8 Aligned Topology-Auxiliary Graph Gate

## Status

status = completed local diagnostic gate
decision = topology auxiliary improves seed0 but remains unstable; do not scale
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_auxiliary_graph_512_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_topology_auxiliary_graph_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_auxiliary_graph_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_topology_auxiliary_graph_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_topology_auxiliary_graph_512_seed0_seed1/history.csv

## Question

Raw-prefix graph routes have been unstable: true topology wins one seed and
loses another across persistent edge tokens, cross-pair consistency, and
coordinate-only active metadata. This suggests the representation may not
separate true PRESENT topology from shuffled topology before the main
distinguisher head is trained.

This gate adds a small topology-control auxiliary objective. During each
forward pass, the model builds both true and shuffled persistent-edge summaries
from the same hidden cell tokens and trains an auxiliary head to distinguish
them. The main distinguisher task and labels remain unchanged.

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

The auxiliary objective is applied inside the model and added to the standard
binary distinguisher loss by the trainer when `last_auxiliary_loss` is present.
It is a local representation diagnostic, not a new evidence scale.

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
| true | `true` | candidate true-topology graph with topology auxiliary |
| shuffled | `shuffled` | controls whether true P-layer targets matter |
| metadata-only | `metadata_only` | controls whether coordinate metadata plus auxiliary objective explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If this gate fails, do not scale. The next route should change the topology
auxiliary design or representation rather than adding sample count.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.549957275 | 0.513763428 | wins seed0, loses to shuffled on seed1 |
| shuffled | 0.519607544 | 0.555465698 | below true on seed0, above true on seed1 |
| metadata-only | 0.502563477 | 0.475112915 | below true on both seeds |

The desired gate did not pass:

```text
true > shuffled on both seeds      no
true > metadata-only on both seeds yes
```

Compared with the coordinate-only gate, the topology auxiliary objective fixed
the seed0 metadata-only reversal and pushed true above metadata-only on both
seeds. However, shuffled topology still beats true topology on seed1.

## Decision

Do not scale this route. This is the strongest raw-prefix graph diagnostic so
far because it passes the metadata-only control on both seeds, but it still
fails the true-vs-shuffled topology control. Treat it as a useful direction,
not as a scale-ready candidate.

The next local route should focus specifically on the true-vs-shuffled
separation, for example by:

- increasing the topology auxiliary influence in a small scale sweep;
- using a pairwise true-minus-shuffled contrast feature rather than only an
  auxiliary loss;
- delaying the main classifier until the topology-control branch has learned a
  separable representation.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_topology_auxiliary_graph_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_topology_auxiliary_graph_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_topology_auxiliary_graph_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_topology_auxiliary_graph_512_seed0_seed1/progress.jsonl
```

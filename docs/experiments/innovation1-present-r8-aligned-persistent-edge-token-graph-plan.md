# Innovation 1 PRESENT r8 Aligned Persistent Edge-Token Graph Gate

## Status

status = completed local diagnostic gate
decision = persistent edge-token graph is unstable; do not scale
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_persistent_edge_token_graph_512_seed0_seed1.csv

artifacts =

- outputs/local_smoke/i1_present_r8_aligned_persistent_edge_token_graph_512_seed0_seed1/results.jsonl
- outputs/local_smoke/i1_present_r8_aligned_persistent_edge_token_graph_512_seed0_seed1/progress.jsonl
- outputs/local_smoke/i1_present_r8_aligned_persistent_edge_token_graph_512_seed0_seed1/curves.svg
- outputs/local_smoke/i1_present_r8_aligned_persistent_edge_token_graph_512_seed0_seed1/history.csv

## Question

The previous dynamic active cell graph used raw prefix features and no dense DDT
trail-value block, but it only passed active-source messages for the current
active nibble. It was encouraging on seed0 and failed on seed1.

This gate tests whether persistent PRESENT P-layer edge tokens give the model a
better SPN coordinate representation: all public P-layer cell relations are
encoded as edge tokens, and active metadata only marks each edge's relative
role. The model still must learn from raw local evidence rather than from DDT
trail-value features.

## Method

Use no DDT trail-value block. Each pair is encoded with:

- ciphertext left word;
- ciphertext right word;
- ciphertext xor;
- P-aligned xor;
- structural inverse prefix.

The model then splits each 64-bit PRESENT word into 16 cell tokens. In
`edge_mode = persistent`, it also creates persistent edge tokens for every
PRESENT P-layer source-cell to target-cell relation. For each sample, the
16-bit active-nibble metadata is used only as coordinate conditioning:

- role 0 = unrelated persistent edge;
- role 1 = edge touches a current active target cell;
- role 2 = edge leaves the current active source cell;
- role 3 = edge enters the current active source cell.

This is intentionally different from the high-AUC DDT trail route: the edge
tokens describe public topology and local raw-prefix cell evidence, not
candidate DDT beam values.

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
- model option `edge_mode = persistent`

Routes:

| route | graph_mode | purpose |
| --- | --- | --- |
| true | `true` | candidate persistent PRESENT P-layer edge-token graph |
| shuffled | `shuffled` | controls whether true P-layer targets matter |
| metadata-only | `metadata_only` | controls whether active metadata alone explains the result |

## Gate

Desired pattern:

```text
true > shuffled on both seeds
true > metadata-only on both seeds
```

If the gate passes, run a slightly larger local/GPU screen such as 2048/class
or 8192/class before considering remote scale. If true ties or loses to either
control, do not scale; redesign the representation locally.

## Result

Plan validation passed for all 6 expected rows.

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| true | 0.519042969 | 0.503952026 | loses seed0, wins seed1 |
| shuffled | 0.529899597 | 0.466140747 | above true on seed0, below true on seed1 |
| metadata-only | 0.532554626 | 0.465972900 | above true on seed0, below true on seed1 |

The desired gate did not pass:

```text
true > shuffled on both seeds      no
true > metadata-only on both seeds no
```

This is better than pure crash/no-signal because seed1 shows the desired
ordering, but seed0 reverses it and metadata-only is highest on seed0. The
result does not justify remote scale or a larger fixed-active run.

## Decision

Do not scale this route. Keep `edge_mode = persistent` as a diagnostic baseline,
but treat this local result as another sign that raw-prefix topology alone is
not yet extracting a stable PRESENT r8 signal.

The next local route should add a stronger learning signal for topology without
feeding DDT trail values as labels or dense main input. Candidate directions:

- topology-control auxiliary head over raw-prefix edge tokens;
- cross-pair edge consistency tokens before pooling;
- stricter separation between active coordinate conditioning and final
  metadata shortcut.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_persistent_edge_token_graph_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_persistent_edge_token_graph_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_persistent_edge_token_graph_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_persistent_edge_token_graph_512_seed0_seed1/progress.jsonl
```

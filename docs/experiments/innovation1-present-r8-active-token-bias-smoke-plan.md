# Innovation 1 PRESENT r8 Active Token Bias Smoke

## Status

status = deferred local smoke
claim_scope = not formal PRESENT r8 evidence
scale = 128/class seed0
plan = configs/experiment/innovation1/innovation1_spn_present_r8_active_token_bias_smoke_128_seed0.csv

defer_reason = the single-active sweep showed that only active nibble 0 is high
under the current protocol, because the active coordinate moves but the
Zhang/Wang input difference remains fixed at low nibble 0

## Question

The statistics-only active-conditioning routes improved narrow active sets but
did not transfer to random-active or heldout-active. This smoke tests a more
architectural route:

Can a PRESENT P-layer token mixer use active-nibble metadata as token-level role
information rather than as a final appended statistic?

## Model Change Under Test

Model:

```text
present_p_layer_mixer_pairset
```

Active-conditioned option:

```text
active_conditioning = p_layer_active_token_bias
metadata_bits = 16
```

For each sample, the model tags PRESENT cells as:

```text
2 = active S-box source cell
1 = direct P-layer target cell
0 = other cell
```

For active nibble 0, feature cell 15 is the source role and cells 11, 7, 3 are
direct P-layer target roles.

## Matrix

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 128
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- feature_encoding = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
- seed = 0

Rows:

| Row | Active set | Conditioning | Purpose |
| --- | --- | --- | --- |
| unconditioned random16 | 0..15 | none | Same-budget graph/token baseline. |
| active-token random16 | 0..15 | token role bias | Tests full random-active conditioning. |
| active-token active4 | 0..3 | token role bias | Tests four-coordinate curriculum. |
| active-token heldout4to4 | train 0..3, validation 4..7 | token role bias | Tests transfer to unseen active coordinates. |

## Gate

This is a smoke only. Continue to a 512/class seed0+seed1 gate only if at least
one active-token row is clearly above the unconditioned random16 baseline and
does not only reproduce fixed-active behavior.

If all rows stay near chance, do not remote-scale this route. Redesign the token
model or revisit the feature route locally.

## Commands

Training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_token_bias_smoke_128_seed0.csv \
  --epochs 3 \
  --batch-size 32 \
  --hidden-bits 16 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --dataset-cache-root outputs/local_cache/i1_present_r8_active_token_bias_smoke_128_seed0 \
  --dataset-cache-chunk-size 128 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_active_token_bias_smoke_128_seed0/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_active_token_bias_smoke_128_seed0/progress.jsonl
```

Validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_token_bias_smoke_128_seed0.csv \
  --results outputs/local_smoke/i1_present_r8_active_token_bias_smoke_128_seed0/results.jsonl \
  --expected-rows 4
```

Plot:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_r8_active_token_bias_smoke_128_seed0/results.jsonl \
  --output outputs/local_smoke/i1_present_r8_active_token_bias_smoke_128_seed0/curves.svg \
  --history-csv outputs/local_smoke/i1_present_r8_active_token_bias_smoke_128_seed0/history.csv \
  --title i1_present_r8_active_token_bias_smoke_128_seed0
```

## Deferral

Do not run this smoke as the immediate next step. It would mix an architecture
change with a now-suspect protocol. First run an aligned-active-difference
protocol where active nibble `k` also moves the input difference to
`0x9 << (4*k)`.

After that protocol audit, rerun this smoke only if the aligned random-active
or heldout-active question still needs an architecture answer.

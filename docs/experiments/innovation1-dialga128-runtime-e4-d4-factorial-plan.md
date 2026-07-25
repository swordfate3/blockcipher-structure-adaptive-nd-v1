# Innovation 1 Dialga-128 Runtime-E4 D4 Factorial Audit Plan

Date: 2026-07-25

## Status

```text
phase = preregistered inference-only diagnosis
source_d1 = completed, pass
source_d3 = completed, hold
training = prohibited
data_generation = prohibited
remote_scale = prohibited
```

## Research Question

D1 and D2 showed strong two-seed Dialga prefix-r4 discrimination and functional
use of the exact rounds-2/3 runtime topology. D3 moved both the encrypted data
to prefix-r5 and the runtime window to rounds 3/4; the correct model then fell
to chance on both seeds. D4 separates those two changes.

Each seed's frozen D1 correct-topology best checkpoint is evaluated in a `2 x 2`
panel:

| Validation data | Runtime window | Condition | Variable changed from anchor |
| --- | --- | --- | --- |
| D1 prefix-r4 | rounds 2/3 | `r4_w2` | none; exact D1/D2 anchor |
| D1 prefix-r4 | rounds 3/4 | `r4_w3` | runtime window only |
| D3 prefix-r5 | rounds 2/3 | `r5_w2` | encrypted data depth only |
| D3 prefix-r5 | rounds 3/4 | `r5_w3` | both factors |

This is an inference audit. It does not train, select, calibrate or modify a
checkpoint and it does not generate a new example.

## Frozen Protocol

```text
cipher = Dialga-128
checkpoint source = D1 correct Runtime-E4 best checkpoint, one per seed
model = runtime_spn_e4_equivariant_true
cell input = state_triplet
S-box context = edge_gate
round window mode = recurrent_window
processor steps = 2
pairs per sample = 4
input difference = 0x40
validation = exact D1 r4 and D3 r5 1024/class caches
validation total = 2048 rows per cell and seed
negative definition = encrypted random plaintext pairs
seeds = 0, 1
metric = AUC
device = local CPU
```

Both persisted D1 and D3 gates must exactly equal gates recomputed from their
source `results.jsonl`. D1 must remain `pass`; D3 must remain `hold` with the
adjacent-window-not-replicated decision. D4 records SHA256 values for both
source result/gate pairs, both caches, both runtime windows and both D1
checkpoints.

## Frozen Interpretation Rule

For each seed, let the D1 anchor be `A = AUC(r4_w2)`. A condition retains the
anchor signal only when it preserves at least half of the anchor's AUC excess
over random:

```text
retention threshold = 0.5 + 0.5 * (A - 0.5)
```

This relative threshold prevents an AUC barely above `0.52` from being called a
replication of a roughly `0.958` anchor. D4 also reports the two data effects,
two window effects and the difference-in-differences interaction:

```text
data effect at w2   = AUC(r5_w2) - AUC(r4_w2)
data effect at w3   = AUC(r5_w3) - AUC(r4_w3)
window effect at r4 = AUC(r4_w3) - AUC(r4_w2)
window effect at r5 = AUC(r5_w3) - AUC(r5_w2)
interaction         = window effect at r5 - window effect at r4
```

The route is chosen before reading D4 metrics:

- Data-depth loss: both seeds retain `r4_w3`, lose `r5_w2`, and also lose the
  r5 signal relative to the `r4_w3` context. Next run is a tiny fixed-network
  input-difference screen on prefix-r5.
- Runtime-window loss: both seeds retain `r5_w2` but lose `r4_w3`. Next change
  is a residual/gated runtime processor that cannot erase the independent
  representation.
- Both factors: both seeds lose `r4_w3` and `r5_w2`. Diagnose one factor at a
  time; do not combine a new difference with a new network.
- Joint interaction: both single-factor cells retain signal but `r5_w3` does
  not. Redesign the runtime processor before any scale-up.
- Stable frozen transfer: all three non-anchor cells retain signal. Treat D3 as
  a training/optimization instability and audit optimization without changing
  the cipher protocol.
- Mixed seeds: no route is selected; retain the per-seed effects and
  preregister one independent validation-key replication.

## Protocol Gate

D4 is valid only if:

1. Exactly eight rows exist: two seeds by four conditions.
2. Each seed uses one D1 correct best checkpoint across all four cells, while
   the two seeds use distinct checkpoints.
3. Both runtime windows load exactly two heterogeneous transitions from the
   same checked Dialga descriptor and have distinct window fingerprints.
4. The two window rows on the same data source share feature, label and
   metadata hashes; r4 and r5 features differ while their aligned labels match.
5. The `r4_w2` AUC reproduces D1/D2 within `1e-12`.
6. All metrics and hashes are valid, the model has `442466` parameters, and no
   training or data generation occurs.

## Explicitly Blocked

- No D1 or D3 checkpoint reselection, retraining or calibration.
- No new validation examples, keys, differences, pairs, epochs or seeds.
- No remote GPU launch and no sample-scale increase.
- No DDT, trail, partial-decryption or guessed-key features.
- No claim of a Dialga attack, formal cross-cipher result, SOTA result or
  universal SPN breakthrough.

## Execution

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/audit-runtime-spn-dialga-d4 \
  --run-id i1_dialga128_runtime_e4_d4_factorial_20260725 \
  --d1-root outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725 \
  --d3-root outputs/local_diagnostic/i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725 \
  --output-root outputs/local_audits/i1_dialga128_runtime_e4_d4_factorial_20260725 \
  --device cpu
```

After execution, render and inspect the SVG with `visual-qa-redraw`, refresh
both recent-result indexes, record the metrics and evidence-backed next action
here, then run the focused regression suite before commit and push.

## Completed Result

```text
run_id = i1_dialga128_runtime_e4_d4_factorial_20260725
status = pass
decision = innovation1_dialga_runtime_e4_d4_data_depth_isolated
diagnosis = fifth_round_data_signal_loss
result_rows = 8
training_performed = false
data_generation_performed = false
```

| Seed | r4_w2 anchor | r4_w3 window only | r5_w2 data only | r5_w3 both |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.958417 | 0.927548 | 0.528886 | 0.520816 |
| 1 | 0.958679 | 0.928645 | 0.526758 | 0.529995 |

The new runtime window retained `93.27%` and `93.45%` of the D1 AUC excess
above random on seeds 0 and 1. In contrast, prefix-r5 data under the old
window retained only `6.30%` and `5.83%`. Moving from r4 to r5 reduced AUC by
`-0.429531` and `-0.431921` at the old window, while changing the runtime
window on r4 reduced AUC by only `-0.030869` and `-0.030034`.

The same pattern holds in the fourth cell: both r5 windows remain near random.
All 18 protocol checks passed. Each seed reused one exact D1 best checkpoint
across four cells; the r4 and r5 caches were reused without generation; the
aligned label hashes match; both persisted source gates exactly match their
recomputed gates; and `r4_w2` reproduces D1/D2 within `1e-12`.

The evidence therefore isolates the primary D3 failure to the encrypted
prefix-r5 data under input difference `0x40`, not to a general inability of the
frozen Runtime-E4 processor to consume the rounds-3/4 Dialga window. This does
not prove that the current processor is optimal; it says that redesigning it
first would address the weaker factor.

The final SVG was rendered to a 1800-pixel PNG and inspected with
`visual-qa-redraw`. Chinese glyphs, title, decision line, legends, axes, eight
AUC labels and ten factor-effect labels have no overlap, clipping, ambiguity or
missing content. The dynamic AUC range keeps both the r4 separation and r5
near-chance values readable.

Artifacts:

```text
outputs/local_audits/i1_dialga128_runtime_e4_d4_factorial_20260725/results.jsonl
outputs/local_audits/i1_dialga128_runtime_e4_d4_factorial_20260725/progress.jsonl
outputs/local_audits/i1_dialga128_runtime_e4_d4_factorial_20260725/validation.json
outputs/local_audits/i1_dialga128_runtime_e4_d4_factorial_20260725/gate.json
outputs/local_audits/i1_dialga128_runtime_e4_d4_factorial_20260725/summary.json
outputs/local_audits/i1_dialga128_runtime_e4_d4_factorial_20260725/curves.svg
outputs/local_audits/i1_dialga128_runtime_e4_d4_factorial_20260725/visual_qa_passed.marker
```

## Recommended Next Action

Run a small local prefix-r5 input-difference screen with Runtime-E4 and the
runtime window frozen. The research question is whether another single-bit or
single-cell input difference preserves a usable five-round signal before any
network redesign or sample increase.

Use the D4 `r5_w2` cell as the exact same-budget anchor. Change only the input
difference; retain Dialga prefix-r5, `2048/class` train, `1024/class`
validation, four pairs, two seeds, ten epochs, encrypted-random-plaintext
negatives, disk cache, Runtime-E4 geometry and the correct/corrupted/no-topology
controls. First run a deterministic cipher-only shortlist over inexpensive
candidate differences, then train at most the top two candidates plus the
`0x40` anchor. Advance only if both seeds achieve `correct AUC >= 0.520` and
both topology margins `>= +0.005`; otherwise stop the Dialga r5 difference
route and implement the residual/gated topology processor as a separate
single-variable experiment.

Do not launch remote GPU work, increase samples, add DDT/trail features, or
change the input difference and network in the same experiment.

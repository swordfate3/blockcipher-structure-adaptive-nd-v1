# Innovation 1 RECTANGLE-80 RuntimeE4 RCT2 Medium Seed0 Plan

Date: 2026-07-25

## Status

```text
stage             = protocol and adjudicator frozen before training
run_id            = i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725
execution         = remote GPU only
launch status     = held behind pushed-SHA verification and active remote lane
result            = none
claim scope       = single-seed medium topology confirmation only
```

RCT1 passed both local seeds, so RCT2 asks one narrower question: does the
correct RECTANGLE runtime descriptor remain better than both equal-parameter
controls when only the data scale increases?

## Authority And Same-Budget Anchor

Source gate:

```text
run id   = i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725
gate     = outputs/local_diagnostic/i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725/gate.json
status   = pass
decision = innovation1_runtime_spn_rectangle_noncontiguous_attribution_supported
```

The RCT1 seed0 row is the same-protocol anchor. RCT2 changes only:

```text
training samples   = 2048/class -> 65536/class
validation samples = 1024/class -> 32768/class
execution device   = local CPU readiness -> remote A6000 GPU
```

The device change is an execution requirement, not a research intervention.
The same implementation, deterministic data construction and metric code must
be used.

## Frozen Protocol

```text
cipher                    = RECTANGLE-80
rounds                    = 6
input difference          = 0x0000002100010020
difference profile        = rectangle80_weng_repo_best_trail_r6
train key                 = 0x00000000000000000000
validation key            = 0x11111111111111111111
seed                       = 0
train rows                 = 131072 total = 65536/class
validation rows            = 65536 total = 32768/class
pairs per sample           = 4 independent ciphertext pairs
feature encoding           = ciphertext_pair_bits = 512 bits/sample
negative definition        = encrypted random plaintexts
epochs / batch             = 5 / 64
loss / optimizer           = MSE / Adam
learning rate              = 0.0001
weight decay               = 0.00001
checkpoint                 = best validation AUC
train evaluation interval  = every epoch
```

Model options remain byte-for-byte equivalent to RCT1:

```json
{
  "runtime_structure_path": "configs/runtime/spn/rectangle64.json",
  "runtime_rounds": 2,
  "processor_steps": 2,
  "pair_embedding_dim": 128,
  "dropout": 0.0,
  "sbox_context_mode": "late_pair"
}
```

Matrix:

```text
configs/experiment/innovation1/innovation1_spn_rectangle80_runtime_e4_medium_rct2_65536_seed0.csv
```

| Role | Model key | Intervention |
| --- | --- | --- |
| correct | `runtime_spn_e4_equivariant_true` | exact external cell, S-box and ShiftRow descriptor |
| corrupted | `runtime_spn_e4_equivariant_corrupted` | deterministic full-bit linear-topology corruption |
| no topology | `runtime_spn_e4_equivariant_independent` | retain cell/S-box metadata but remove linear relations |

All rows must keep `442466` trainable parameters.

## Remote Storage And Publication Preconditions

The run must use a clean run-owned clone at an exact GitHub-verified commit.
Do not launch from a local-only commit, dirty overlay or existing dirty remote
clone. All generated project artifacts stay under:

```text
source = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725\source
run    = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725
```

Before launch, the generated Windows path must pass these checks:

```text
launcher              = cmd.exe /c
project paths         = only G:\lxy
dataset cache         = features.npy + labels.npy + metadata.json or equivalent
cache generation      = durable chunk progress before training
cache reuse           = exact parameter-matched identity
cache chunk/workers   = 1024 / 1
progress              = progress.jsonl under the run root
checkpoints/logs      = under the run root
source HEAD           = exact pushed commit
```

## Fail-Closed Result Gate

The RCT2 adjudicator is:

```text
scripts/gate-runtime-spn-rectangle-medium
```

It rejects an incomplete matrix, any protocol drift, non-disk or non-`G:\lxy`
cache, wrong descriptor hash, unequal parameter geometry, missing five-epoch
history, non-finite metrics, or a reported result that does not replay the
best validation-AUC checkpoint.

After all protocol checks pass, seed0 advances only when:

```text
correct AUC >= 0.55
correct - corrupted AUC >= +0.005
correct - no-topology AUC >= +0.005
```

`pass` opens an identical seed1 confirmation only after verified retrieval,
local redraw, `visual-qa-redraw`, documentation and result-index refresh.
`hold` stops the scale ladder and returns to a single local diagnosis.
`fail` permits evidence/protocol repair only. Do not rescue a weak result by
increasing pairs, epochs, rounds or moving directly to `262144/class`.

## Queue Decision

At freeze time, the local tmux process list shows active Innovation 1 watchers
for SKINNY RTG3-A seed0/seed1 and uKNIT U3/U4. RCT2 therefore remains prepared
but unlaunched. This prevents GPU contention and does not count as a failed or
running result. Launch assets and the local retrieval watcher must be verified
after the exact source commit is published and the active remote lane becomes
available.

## Implementation Readiness

The frozen matrix parses as exactly three seed0 rows with the required model
keys, sample scale, four-pair input, difference profile and runtime descriptor.
Synthetic gate fixtures verify both paths:

```text
valid remote-cache + five-epoch evidence = pass
non-G:\lxy cache or missing history       = fail closed
remote --no-plot postprocessing           = writes gate/validation/summary/history
```

The RCT2 SVG template was rendered to `1824 x 864` pixels and inspected with
`visual-qa-redraw`. The Chinese title and protocol explanation, AUC labels,
margin labels, legends, thresholds, axes and evidence caption have no
unintended overlap, clipping, missing glyph or ambiguous association. This is
a template check only; the retrieved real result must undergo a fresh rendered
pixel inspection before `visual_qa_passed.marker` is created.

# Innovation 1 RECTANGLE-80 RuntimeE4 Non-Contiguous Attribution RCT1 Plan

Date: 2026-07-25

## Status

```text
stage       = completed and adjudicated
run_id      = i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725
execution   = local readiness diagnostic
remote GPU  = no
status      = pass
decision    = innovation1_runtime_spn_rectangle_noncontiguous_attribution_supported
claim scope = RECTANGLE-80 r6 non-contiguous-cell runtime-topology readiness only
```

## Research Question

Can the same cipher-name-free RuntimeE4 backbone used for PRESENT, GIFT and
SKINNY exploit RECTANGLE's externally supplied non-contiguous bitsliced cells
and ShiftRow topology, without changing its trainable parameter geometry?

RCT1 changes the target cipher and runtime descriptor, not the neural backbone.
It compares the correct descriptor against a deterministic corrupted topology
and a no-linear-topology control at exactly the same data and training budget.

## Frozen Difference

The design paper confirms the final revised S-box, row rotations and the
six-round best single-trail weight of 18, but its NIST copy does not publish a
concrete input difference. The public supplementary search repository by Weng,
Zhang, Peng and Ding contains the exact six-round trail:

```text
repository commit = df0dac19726350838c43d7a254575f2f0f6ba18a
artifact           = Experimental result/BestTrails/RECTANGLE/DiffTrail_64.txt
artifact SHA-256   = 46247df28e81155e8ac5eeb650d5239415595d64e54fdae904bc0c6aa965d7d6
RNUM_6             = Bn 18
artifact input     = cell14:0x6, cell3:0x5
```

The artifact prints cells from 15 down to 0. RECTANGLE is rotationally
symmetric across its 16 columns, so rotate cell 14 to cell 0. Cell 3 then maps
to cell 5. With project physical index `16 * row + column`, the frozen external
input difference is:

```text
cell0 = 0x6 -> physical bits 16 and 32
cell5 = 0x5 -> physical bits 5 and 37
input difference = 0x0000002100010020
profile = rectangle80_weng_repo_best_trail_r6
```

The raw web lookup, exact URLs, hashes and mapping audit are stored in
`sources/research_rectangle80_differential_neural_readiness_20260725.json`.

## Frozen Protocol

```text
cipher                    = RECTANGLE-80
rounds                    = 6
train key                 = 0x00000000000000000000
validation key            = 0x11111111111111111111
train samples             = 2048/class = 4096 total per row
validation samples        = 1024/class = 2048 total per row
seeds                     = 0, 1
pairs per sample          = 4 independent ciphertext pairs
feature encoding          = ciphertext_pair_bits = 512 bits/sample
negative definition       = encrypted random plaintexts
epochs                    = 5
batch size                = 64
loss / optimizer          = MSE / Adam
learning rate             = 0.0001
weight decay              = 0.00001
checkpoint                = best validation AUC
dataset storage           = disk-backed cache
```

The fixed model options are:

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

## Three Equal-Budget Rows Per Seed

| Role | Model key | Intervention |
| --- | --- | --- |
| correct | `runtime_spn_e4_equivariant_true` | exact external cell, S-box and ShiftRow descriptor |
| corrupted | `runtime_spn_e4_equivariant_corrupted` | deterministic full-bit linear-topology corruption |
| no topology | `runtime_spn_e4_equivariant_independent` | retain cell/S-box metadata but remove linear relations |

All six rows must have equal parameter and trainable-parameter counts. The
descriptor SHA must be
`904241dc1d42470188b5ed6a1c080a24191433cfc065f8838cdbe06ba2a2a4cd`.

## Preregistered Gate

Protocol failure occurs if any row, seed, key width, difference, cache,
negative definition, feature width, model option, descriptor SHA, runtime mode,
parameter geometry or finite AUC check fails.

Both seeds must independently satisfy:

```text
correct AUC >= 0.55
correct - corrupted AUC >= +0.005
correct - no-topology AUC >= +0.005
```

Decisions:

```text
pass = both seeds pass all three research checks;
       freeze a remote 65536/class seed0 confirmation plan when the active
       remote lane is available
hold = protocol is valid but either seed misses a signal/control gate;
       do not scale, and diagnose whether the failure is signal strength,
       cell orientation or topology identifiability
fail = repair only the failed protocol check and rerun the frozen matrix
```

## Outputs

```text
outputs/local_diagnostic/i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725/
  results.jsonl
  progress.jsonl
  validation.json
  gate.json
  summary.json
  history.csv
  curves.svg
  checkpoints/
  cache/
```

After the result completes, validate all six rows, run the RCT1 gate, render and
pixel-inspect the SVG with `visual-qa-redraw`, refresh both recent-result
indexes, record the metrics and evidence-backed next action here, then commit
and push the scoped files. Do not launch a remote RECTANGLE run unless both
local seeds pass.

## Completed Results

All six rows completed and passed the frozen protocol checks. The validation
and gate both report `status = pass`.

| Seed | Correct topology AUC | Corrupted topology AUC | No-topology AUC | Correct - corrupted | Correct - no topology |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7914686203 | 0.6456823349 | 0.6500158310 | +0.1457862854 | +0.1414527893 |
| 1 | 0.7656726837 | 0.6471195221 | 0.6468963623 | +0.1185531616 | +0.1187763214 |

Both seeds exceed the preregistered `0.55` correct-topology signal gate and
both `+0.005` attribution margins. The equal-parameter corrupted and
no-linear-topology rows remain substantially below the correct descriptor.
This supports the narrow claim that the cipher-name-free RuntimeE4 backbone
can consume RECTANGLE's externally supplied non-contiguous cell assignment and
ShiftRow topology. It does not establish paper-scale performance,
cross-cipher weight reuse, universal SPN support, an attack, or SOTA.

Completed artifacts:

```text
outputs/local_diagnostic/i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725/results.jsonl
outputs/local_diagnostic/i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725/validation.json
outputs/local_diagnostic/i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725/gate.json
outputs/local_diagnostic/i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725/summary.json
outputs/local_diagnostic/i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725/history.csv
outputs/local_diagnostic/i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725/curves.svg
outputs/local_diagnostic/i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725/visual_qa_passed.marker
```

The final SVG was rendered to `1824 x 899` pixels and passed
`visual-qa-redraw`: Chinese glyphs, titles, legends, data labels, thresholds,
axes and the evidence caption are readable, with no unintended overlap,
clipping or ambiguous association. The first render's left-panel legend/data
collision was repaired by moving that legend into the figure band and using
horizontal AUC labels.

## Adjudication And Next Action

```text
RCT1 decision = keep the RECTANGLE non-contiguous descriptor route
remote now    = wait for the active Innovation 1 remote lane
next run      = RECTANGLE-80 r6, 65536/class train, seed0 only
controls      = correct / corrupted / no-linear-topology
fixed         = difference, keys, 4 pairs/sample, 512-bit input, RuntimeE4,
                model options, 5 epochs, optimizer, loss and checkpoint rule
execution     = remote GPU with disk-backed cache/progress/reuse under G:\lxy
```

The next research question is whether the large local attribution margins
survive a medium-scale confirmation. Change only the train/validation sample
budget; retain all three equal-budget rows and every other RCT1 protocol field.
Run seed0 first. Advance to an identical seed1 confirmation only if seed0 has a
valid archive, `correct AUC >= 0.55`, `correct - corrupted >= +0.005`, and
`correct - no topology >= +0.005`. Hold and return to local diagnosis if any
signal/control gate fails. Do not mechanically increase pairs, epochs, rounds,
or proceed to `262144/class` before that decision.

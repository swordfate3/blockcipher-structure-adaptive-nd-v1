# Innovation 1 PRESENT Runtime-E4 RTG3-B Formal Seed1 Plan

```text
status       = prepared / conditional / not launched
run_id       = i1_rtg3b_present80_one_to_one_formal_1000000_seed1_20260726
execution    = remote lxy-a6000 GPU0 only after the seed0 publication gate passes
dependency   = complete locally verified RTG3-B seed0 formal pass
one variable = seed 0 -> 1
```

## Research Question

Does the PRESENT-80 r7 correct one-to-one runtime topology advantage replicate
under seed1 at the identical `1000000/class` project-formal budget?

This package may be prepared while seed0 runs, but it has no authority to
contact the remote workstation until seed0 is retrieved from its verified
result branch and passes local result validation, the frozen research gate and
rendered-pixel visual QA. A local successor watcher waits only on those local
artifacts. A seed0 hold or protocol failure stops the successor.

## Same-Budget Anchor And One Variable

The same-budget anchor is RTG3-B seed0. The only research variable is:

```text
seed = 0 -> 1
```

Run identity and descriptive evidence fields change accordingly. Cipher,
rounds, difference, keys, pair organization, negative definition, model keys,
model options, topology controls, loss, optimizer, epochs, checkpoint rule,
data totals and thresholds remain equal at the plan-field level.

## Frozen Protocol

```text
cipher                    = PRESENT-80
rounds                    = 7
difference profile        = present_zhang_wang2022_mcnd
sample structure          = zhang_wang_case2_official_mcnd
train key                 = 0x00000000000000000000
validation key            = 0x11111111111111111111
train                     = 2000000 total = 1000000/class
validation                = 1000000 total = 500000/class
pairs/sample              = 16 = 2048 raw ciphertext-pair bits
negative                  = encrypted random plaintexts
models                    = correct / deterministic corrupted / no topology
processor steps           = 2
pair embedding dimension  = 128
S-box context             = late_pair
parameters                = equal across all three rows
loss                      = MSE
optimizer                 = Adam, learning rate 0.0001
weight decay              = 0.00001
epochs                    = 5/model
checkpoint                = best validation AUC
seed                      = 1
device                    = remote CUDA only
```

All three roles must reuse byte-identical disk-backed train and validation
arrays inside the seed1 run. The seed0 cache must not be reused because the
random seed is the sole research variable.

## Conditional Launch Gate

Before seed1 can contact the remote, require all of:

1. Seed0 is retrieved from its verified result branch with exact run id,
   phase `rtg3b`, seed `0`, pass status and decision
   `innovation1_runtime_spn_present_formal_seed0_supported`.
2. Every seed0 protocol and research check is true and every AUC/margin is
   finite.
3. Local plan validation reports exactly three result rows with no errors;
   the result roles are correct/corrupted/no-topology at `1000000/class`.
4. Local history contains exactly fifteen rows: five ordered epochs for each
   of the three models.
5. `visual-qa-redraw` has inspected the rendered seed0 chart and written
   `visual_qa_passed.marker` beside it.
6. The seed1 CSV differs from seed0 only by seed and descriptive identity.
7. Protected PRESENT data, model and training paths are unchanged from seed0
   training commit `233b2e2986578bb66bb95055f380a3ed21cbff1d`.
8. Remote readiness verifies three rows, CUDA, disk cache, progress,
   checkpoints and all frozen fields under `G:\lxy`.
9. Every seed1 source asset is committed, byte-matches the worktree and the
   exact commit equals the live `origin/main` SHA.

The remote launcher independently checks the archived seed0 gate before
scheduling seed1. No successful SSH process exit is treated as proof of
training start; the local monitor requires the exact started marker.

## Result Gate

Seed1 passes only if protocol validation succeeds and:

```text
correct AUC >= 0.520
correct - corrupted topology >= +0.005
correct - no topology >= +0.005
```

After verified retrieval and visual QA:

```text
seed0 pass + seed1 pass = close C3 with two-seed one-to-one topology support
seed1 research hold      = stop RTG3-B; do not rescue by changing scale or model
protocol failure         = repair only the failed invariant before interpretation
```

A two-seed pass unlocks only the separately preregistered C5 local comparison
between unchanged Runtime-E4 and the parameter-shape-matched periodic-orbit
candidate. It is not a Zhang/Wang reproduction, attack, SOTA, breakthrough,
correct-S-box proof, unseen-cipher transfer result, or universal-SPN evidence.

## Artifacts

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\
  i1_rtg3b_present80_one_to_one_formal_1000000_seed1_20260726\
    cache\
    checkpoints\
    logs\progress.jsonl
    results\results.jsonl
    source\results_archive\<run_id>\

outputs/remote_results/<run_id>/
  results.jsonl
  validation.local.json
  gate.local.json
  history.csv
  curves.svg
  visual_qa_passed.marker
```

After completion, update this record with exact metrics, margins, provenance,
claim scope and the evidence-backed C5-or-stop decision. Refresh both recent
result indexes in the same turn.

## Blocked Routes

Do not launch before the conditional gate passes. Do not modify the running
seed0 watcher, run seed1 locally, reuse the seed0 cache, change pairs, keys,
negatives, model geometry, optimizer, epochs, thresholds, or launch from an
unpublished commit or source overlay. Do not begin C5 until both C3 seeds are
complete and jointly interpreted.

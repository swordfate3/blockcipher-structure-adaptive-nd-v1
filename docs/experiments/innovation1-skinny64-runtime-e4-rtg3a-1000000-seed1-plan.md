# Innovation 1 SKINNY RTG3-A Runtime E4 1000000/Class Seed1 Plan

Date: 2026-07-25

## Status

```text
stage       = prepared / conditional / not launched
run_id      = i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_20260725
execution   = remote lxy-a6000 GPU0 only after the local publication gate passes
dependency  = completed and locally verified RTG3-A seed0 formal pass
one variable = seed 0 -> 1
claim scope = second project-formal seed confirmation only
```

This package is prepared while seed0 runs, but it has no authority to contact
the remote workstation until seed0 is retrieved from its verified result
branch and passes local result validation, checkpoint replay, research gates
and rendered-pixel visual QA. A local successor watcher waits only on those
local artifacts. A seed0 hold writes a stopped marker; a protocol failure
repairs evidence only. Neither state may launch seed1.

## Research Question

Does the SKINNY-64/64 r7 correct runtime general-GF(2) topology advantage
replicate under seed1 at the identical `1000000/class` project-formal budget?

The same-budget anchor is RTG3-A seed0. The only research variable is:

```text
seed = 0 -> 1
```

Run identity and descriptive evidence fields change accordingly. Cipher,
rounds, difference, keys, input pairs, negative definition, model keys, model
options, topology controls, optimizer, loss, batch, epochs, checkpoint rule,
data totals and research thresholds remain byte-for-byte equivalent at the
plan-field level.

## Frozen Protocol

```text
cipher                   = SKINNY-64/64
rounds                   = 7
difference               = 0x2000
train key                = 0x0000000000000000
validation key           = 0x1111111111111111
train                    = 2000000 total = 1000000/class
validation               = 1000000 total = 500000/class
pairs/sample             = 4 independent ciphertext pairs = 512 input bits
negative                 = encrypted random plaintexts
models                   = correct / deterministic corrupted / no topology
parameters               = 442466 per model
epochs                   = 5/model
batch                    = 64
optimizer                = Adam, learning rate 1e-4
loss                     = MSE
weight decay             = 1e-5
checkpoint               = best validation AUC
seed                     = 1
device                   = remote CUDA only
```

All three roles must reuse byte-identical disk-backed train and validation
arrays. Parameter-matched cache reuse is allowed only inside this run-owned
seed1 cache after complete metadata equality. No seed0 cache is reused because
the random seed is the sole research variable.

## Conditional Publication Gate

Before seed1 can contact the remote, require all of the following:

1. Seed0 is retrieved from the verified result branch and has exact run id,
   phase `rtg3a`, seed `0`, pass status and decision
   `innovation1_rtg3a_skinny_formal_seed0_supported`.
2. Every seed0 protocol and research check is true; all three AUCs and both
   margins are finite.
3. Local validation reports exactly three result rows with no errors, and the
   result roles are correct/corrupted/no-topology at `1000000/class`.
4. Local history contains exactly fifteen rows: five ordered epochs for each
   of the three models.
5. Checkpoint verification reports exactly three expected files, strict model
   loading, complete metadata/history/final-metric equality, correct parameter
   count and selected best checkpoint for every file.
6. `visual-qa-redraw` has inspected rendered pixels and written
   `visual_qa_passed.marker` beside the seed0 chart.
7. Seed1 differs from seed0 only by seed and descriptive identity; protected
   data, model and training paths are unchanged from seed0 training commit
   `131e6387e164fb8679b1bcb9bf46887cf049fca0`.
8. Remote readiness verifies three rows, CUDA, disk cache, progress,
   checkpoints and all frozen training fields under `G:\lxy`.
9. Every seed1 source asset is committed, byte-matches the worktree and the
   exact commit is published to `origin/main`.

The remote launcher repeats the exact seed0 remote gate identity and all
protocol/research checks before scheduling seed1. The local publication gate
is authoritative for the stronger checkpoint and visual evidence.

## Result And Joint Gates

Seed1 research pass requires:

```text
correct AUC >= 0.55
correct - corrupted AUC >= +0.005
correct - no-topology AUC >= +0.005
```

Protocol validity requires exactly three rows, fifteen history rows, three
strictly replayed checkpoints, equal parameter geometry, strict encrypted
random-plaintext negatives, parameter-matched disk caches, exact source
revision and immutable archive SHA verification.

After retrieval and local visual QA, synthesize seed0 and seed1 with:

```text
scripts/gate-runtime-spn-skinny-medium-joint --phase rtg3a
```

Decisions:

```text
seed0 pass + seed1 pass = two-seed project-formal support for this SKINNY route
either research hold     = stop mechanical scale-up; audit fixed-protocol dynamics
protocol failure         = repair evidence only; do not interpret AUC
```

Even a two-seed pass is method evidence for this frozen SKINNY experiment. It
is not paper-scale AutoND reproduction, a key-recovery attack, SOTA,
breakthrough evidence or proof of universal SPN adaptation. The next research
step must compare this same runtime mechanism on another preregistered SPN or
test a separately isolated cross-cipher hypothesis; it must not mechanically
increase SKINNY sample count.

## Required Artifacts

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\<seed1_run_id>\
  cache\
  checkpoints\
  logs\progress.jsonl
  results\results.jsonl
  source\results_archive\<seed1_run_id>\

outputs/remote_results/<seed1_run_id>/
  results.jsonl
  validation.local.json
  gate.local.json
  checkpoint-verification.local.json
  history.local.csv
  curves.svg
  visual_qa_passed.marker
```

After a completed result, refresh both recent-result indexes and update this
record with exact metrics, deltas, claim scope and the evidence-backed next
action.

## Blocked Routes

Do not launch seed1 before every publication check passes. Do not modify the
running seed0 watcher. Do not run seed1 locally, reuse the seed0 cache, change
the difference, keys, pair count, negative semantics, model geometry,
optimizer, epochs, thresholds or checkpoint rule. Do not start Route B or a
larger SKINNY scale as a rescue.

# Innovation 1 SKINNY RTG2-B Runtime E4 262144/Class Seed1 Plan

Date: 2026-07-24

## Status

```text
stage       = local launch authorization passed; remote launch pending republished source gate
run_id      = i1_rtg2b_skinny64_general_gf2_scale_262144_seed1_20260724
execution   = remote lxy-a6000 only
dependency  = satisfied by exact plan-aligned RTG2-B seed0 pass
seed0 gate  = pass / innovation1_rtg2b_skinny_scale_seed0_supported
launch gate = pass / innovation1_rtg2b_seed1_remote_launch_authorized
claim scope = medium architecture/protocol diagnostic only
```

This plan prepared the second-seed confirmation while seed0 was running. Seed0
has now completed and passed at `0.649229395` AUC with margins `+0.045667696`
over corrupted topology and `+0.139039457` over no topology. The local launch
gate consumed the completed, locally validated, visually checked seed0 result
retrieved from its verified result branch and authorized the exact unchanged
seed1 protocol. Before remote contact, the gate is rerun against the final
pushed source commit so the launch SHA includes this completed-result record.

## Research Question

If seed0 passes all three RTG2-B gates, does the same correct general-GF(2)
runtime topology retain its two-control advantage under seed1 at the identical
`262144/class` budget?

The same-budget anchor is RTG2-B seed0. The only research variable is the random
seed:

```text
anchor    = seed0
candidate = seed1
```

No model, feature, data protocol, optimizer, epoch, batch, checkpoint, negative
definition, key, difference, or control semantics may change.

## Frozen Protocol

```text
cipher                    = SKINNY-64/64
rounds                    = 7
difference profile        = skinny64_gohr2022_single_key
difference member         = 0
input difference          = 0x2000
seed                      = 1
train key                 = 0x0000000000000000
validation key            = 0x1111111111111111
train                     = 524288 total = 262144/class
validation                = 262144 total = 131072/class
pairs/sample              = 4 independent ciphertext pairs
feature                   = ciphertext_pair_bits
negative                  = encrypted random plaintexts
models                    = correct / deterministic corrupted / no topology
parameters/model          = 442466 trainable
runtime processor steps   = 2
pair embedding            = 128
S-box context             = late_pair
epochs                    = 5/model
batch                     = 64
optimizer                 = Adam, learning rate 1e-4
loss                      = MSE
weight decay              = 1e-5
learning-rate scheduler   = none
checkpoint                = best validation AUC
device                    = remote CUDA, physical GPU0
```

Matrix:

```text
configs/experiment/innovation1/
  innovation1_spn_skinny64_runtime_e4_scale_rtg2b_262144_seed1.csv
```

The three rows share the same cached dataset, parameter geometry and training
budget. They differ only in runtime topology mode: correct, deterministic
corrupted, or no linear topology.

## Conditional Launch Gate

Before any SSH contact, require all of the following:

1. Seed0 has exact run id
   `i1_rtg2b_skinny64_general_gf2_scale_262144_seed0_20260724`.
2. Seed0 gate status is `pass` with decision
   `innovation1_rtg2b_skinny_scale_seed0_supported`.
3. Every seed0 protocol and research check is true, all AUCs and margins are
   finite, and local result validation reports exactly three rows.
4. Seed0 came from a verified result branch and has completed pixel-level
   visual QA; raw fallback evidence is insufficient for automatic seed1 launch.
5. The seed1 plan differs from seed0 only in seed and descriptive identity.
6. Protected data/model/training paths are unchanged from seed0 training commit
   `061fd9a3c30cd1089a24e9df241f63964d147d6c`.
7. Remote readiness verifies CUDA, GPU0, disk cache, progress, checkpoints,
   three rows, five epochs, batch 64 and strict negatives under `G:\lxy`.
8. Every seed1 source asset is committed, matches the worktree and is published
   to `origin/main`.

The Windows launcher repeats the exact seed0 gate identity/status/decision check
before creating the seed1 run-owned clone. A failed condition exits without
training.

## Result Gate

After all three models complete five epochs and restore their best checkpoints:

```text
correct AUC >= 0.55
correct AUC - corrupted AUC >= +0.005
correct AUC - no-topology AUC >= +0.005
```

Protocol validity also requires complete histories, equal geometry, strict
negatives, parameter-matched disk caches, exact row counts, source revision,
clean status and SHA-256 verification.

## Decisions

```text
seed0 not pass = do not launch seed1; redesign locally
seed1 pass     = synthesize the two-seed 262144/class evidence
seed1 hold     = stop mechanical scaling and audit training dynamics
protocol fail  = repair evidence only; do not interpret AUC
```

Even two passing seeds remain medium diagnostic evidence. They do not authorize
`1000000/class`, a broad cipher matrix, an attack, SOTA, breakthrough or a
universal-SPN claim without a separate plan and formal evidence gate.

## Completion Artifacts

Require `results.jsonl`, full progress, three checkpoints, validation, gate,
summary, history, source revision, clean status, GPU/torch/readiness logs,
frozen plan/config and `SHA256SUMS` on a verified result branch. After local
retrieval, redraw the chart, run `visual-qa-redraw`, write
`visual_qa_passed.marker`, refresh both recent-result indexes, and update the
seed0 and seed1 experiment records with the two-seed recommendation.

The seed1 watcher must then run the shared joint gate with `--phase rtg2b`,
write the derived synthesis under
`outputs/remote_results_incomplete/i1_rtg2b_skinny64_general_gf2_scale_262144_joint_seed0_seed1_20260724/`,
and refresh the recent-result index again. The joint gate must reject RTG2-A
source gates, duplicate seeds, mismatched phases, altered thresholds and
non-finite metrics.

## Blocked Routes

Do not launch seed1 before seed0 passes. Do not rescue a weak result by changing
the difference, round count, key protocol, pairs, negatives, topology controls,
model geometry, optimizer, epochs, batch or checkpoint rule. Do not mix this
replication with DDT, trail, partial-decryption or uKNIT feature branches.

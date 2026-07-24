# Innovation 1 SKINNY RTG3-A Runtime E4 1000000/Class Seed0 Plan

Date: 2026-07-25

## Status

```text
stage       = running remotely; bounded startup evidence verified
run_id      = i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725
execution   = remote lxy-a6000 GPU0 only
dependency  = RTG2-B 262144/class two-seed pass plus completed post-X2 route audit
source      = 131e6387e164fb8679b1bcb9bf46887cf049fca0 (verified origin/main)
launch gate = pass / innovation1_rtg3a_seed0_remote_launch_authorized
started     = 2026-07-25 00:42:34 +08:00
monitor     = local tmux i1_rtg3a_skinny64_formal_monitor
claim scope = single-seed project-formal scale evidence only
```

The watcher observed the remote launcher return, then confirmed the exact
run-owned `started.marker` inside its bounded 30-attempt window. It now owns
log synchronization, completion detection, immutable archive retrieval,
checkpoint retrieval, local re-adjudication and result indexing. No AUC or
research decision exists until those result artifacts are retrieved locally.

## Research Question

Does the SKINNY-64/64 r7 correct runtime general-GF(2) topology advantage
survive at the project's formal evidence floor of `1000000/class`, when the
network, data protocol, controls and five-epoch training budget are frozen from
RTG2-B?

This is a sample-scale replication, not a new architecture, feature route,
negative definition, published-paper reproduction or attack.

## Same-Budget Anchor And One Variable

The exact anchor is the completed RTG2-B matrix:

| Seed | Correct topology | Corrupted topology | No topology |
| ---: | ---: | ---: | ---: |
| 0 | 0.649229395 | 0.603561698 | 0.510189938 |
| 1 | 0.647782881 | 0.602584307 | 0.513038491 |

RTG3-A changes one research field:

```text
samples_per_class = 262144 -> 1000000
```

Run identity and descriptive evidence fields also change. Cipher, rounds,
difference, keys, four-pair input, negatives, model keys, model options,
optimizer, loss, batch size, epochs, topology controls and checkpoint rule do
not change.

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
seed                     = 0
device                   = remote CUDA only
```

All three roles must reuse byte-identical disk-backed train and validation
arrays. Cache chunks, metadata and progress must be durable under the run-owned
`G:\lxy\blockcipher-structure-adaptive-nd-runs` directory before training
finishes. A parameter-matched cache may resume only when its complete protocol
metadata matches.

## Readiness And Launch Gate

Before SSH contact require:

1. The RTG2-B joint gate and validation both pass with exact identities.
2. The post-X2 route decision selects Route A.
3. The formal CSV differs from RTG2-B only in scale and descriptive identity.
4. Remote readiness confirms three rows, CUDA protocol, disk cache, progress,
   checkpoint and `G:\lxy` ownership.
5. The exact source commit contains and byte-matches every plan, gate, run,
   launch and monitor asset.
6. The exact source commit is published and independently verified on
   `origin/main`.
7. Generated Windows scripts contain `cmd.exe /c`, no `cmd.exe /k`, no delayed
   expansion and no `!` characters.

The launcher must use a clean run-owned clone and exact detached source commit.
One bounded read-only startup confirmation must observe the remote
`started.marker`; the local tmux watcher then owns monitoring and retrieval.

## Result Gate

Protocol failure if any of the three rows, sample totals, cache ownership,
five-epoch histories, strict negatives, equal geometry, source revision,
checkpoint replay or immutable archive checks fail.

Research pass requires:

```text
correct AUC >= 0.55
correct - corrupted AUC >= +0.005
correct - no-topology AUC >= +0.005
```

Decisions:

```text
pass:
  decision    = innovation1_rtg3a_skinny_formal_seed0_supported
  next_action = prepare the identical conditional seed1 plan and publication gate

hold:
  decision    = innovation1_rtg3a_skinny_formal_not_supported
  next_action = stop scale-up and audit fixed-protocol dynamics; do not change
                model, data or thresholds as a rescue

protocol fail:
  decision    = innovation1_rtg3a_skinny_formal_protocol_invalid
  next_action = repair evidence only
```

A seed0 pass is single-seed project-formal evidence. It does not authorize a
two-seed formal conclusion until the identical seed1 result passes and a joint
gate is retrieved, validated, indexed and visually checked.

## Required Artifacts

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>\
  cache\
  checkpoints\
  logs\progress.jsonl
  logs\<run_id>_started.marker
  results\results.jsonl
  source\results_archive\<run_id>\

outputs/remote_results/<run_id>/
  results.jsonl
  validation.local.json
  gate.local.json
  checkpoint-verification.local.json
  history.local.csv
  curves.svg
  visual_qa_passed.marker
```

After retrieval, preserve the remote manifest-owned archive unchanged, write
local re-adjudication to separate `*.local.*` files, run rendered-pixel visual
QA, refresh both recent-result indexes and update this document with exact
metrics and the evidence-backed next action.

## Blocked Routes

Do not launch seed1 before seed0 passes. Do not run Route B concurrently. Do
not change rounds, difference, keys, negative semantics, pair count, topology
corruption, model geometry, optimizer, epochs, threshold or checkpoint rule.
Do not call this paper-scale, an exact reproduction, an attack, SOTA, a
breakthrough or universal-SPN evidence.

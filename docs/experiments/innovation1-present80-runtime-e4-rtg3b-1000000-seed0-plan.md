# Innovation 1 PRESENT Runtime-E4 RTG3-B Formal Seed0 Plan

```text
status = local readiness passed / source publication verification hold
phase = RTG3-B
remote_training = not started
remote_launch = prohibited until readiness and exact source publication pass
```

## Research Question

Does the repaired Runtime-E4 correct-topology advantage on the one-to-one
PRESENT-80 P-layer survive the project's formal evidence floor of
`1000000/class`?

C2 froze the supported method boundary as a runtime GF(2)-topology-aware SPN
neural distinguisher. General-GF(2) attribution already has two-seed project-
formal SKINNY evidence. The one-to-one branch has only local `2048/class`
GIFT/PRESENT diagnostics. RTG3-B closes that evidence-scale imbalance without
changing the architecture or benchmark.

## Anchors

Local PRESENT T1:

| Seed | Correct | Corrupted P-layer | No topology | Correct-corrupted | Correct-no topology |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.664596081` | `0.570662022` | `0.554435253` | `+0.093934059` | `+0.110160828` |
| 1 | `0.676282406` | `0.561504364` | `0.571587086` | `+0.114778042` | `+0.104695320` |

Scale/provenance anchor:

```text
SKINNY RTG3-A train      = 2000000 total = 1000000/class
SKINNY RTG3-A validation = 1000000 total = 500000/class
SKINNY RTG3-A models     = correct / corrupted / no topology
SKINNY RTG3-A epochs     = 5/model
```

The SKINNY artifact is fallback-retrieved. It supplies the project-formal
scale contract, not a publication-style provenance upgrade for PRESENT.

## One Variable

Only train and validation sample scale changes relative to PRESENT T1 seed0.
Network geometry, cipher, round, seed, keys, pair organization, input encoding,
negative definition, loss, optimizer, epochs, checkpoint rule, S-box context
and topology controls remain fixed.

Allowed identity-only changes are `network`, `family`, `evidence`, `literature`
and the run id.

## Frozen Protocol

```text
cipher                    = PRESENT-80
rounds                    = 7
seed                      = 0
difference profile        = present_zhang_wang2022_mcnd
sample structure          = zhang_wang_case2_official_mcnd
train key                 = 0x00000000000000000000
validation key            = 0x11111111111111111111
train                     = 2000000 total = 1000000/class
validation                = 1000000 total = 500000/class
pairs/sample              = 16
input                     = raw ciphertext-pair bits
negative                  = encrypted random plaintexts
models                    = correct / deterministic full-bit corrupted /
                            no topology
processor steps           = 2
pair embedding dimension  = 128
S-box context             = late_pair
parameters                = equal across all three rows
loss                      = MSE
optimizer                 = Adam
learning rate             = 0.0001
weight decay              = 0.00001
epochs                    = 5/model
checkpoint                = best validation AUC
execution                 = remote A6000 GPU0
```

This is a project-formal attribution replication, not the data scale or
training schedule of a named paper reproduction. The strict Zhang/Wang r7
reference remains Case2 `m=16`, accuracy `0.7205`, and has not been reproduced
locally by this route.

## Disk-Backed Data Contract

The generic matrix runner already keys caches by cipher, round, split, seed,
sample counts, pair count, difference, feature encoding, negative mode, sample
structure and key. Because model identity is not part of the dataset cache key,
the three parameter-matched topology rows must reuse one train cache and one
validation cache.

Expected feature storage before filesystem overhead:

```text
input bits/sample     = 16 pairs * 2 ciphertexts * 64 bits = 2048 uint8 values
train features        = 2000000 * 2048 = 4096000000 bytes
validation features   = 1000000 * 2048 = 2048000000 bytes
labels                = about 3000000 bytes total
```

The remote run must use:

```text
features.npy
labels.npy
metadata.json
chunk size = 1024
workers = 1
progress JSONL after every generated chunk
parameter-matched cache reuse for rows 2 and 3
```

All source, cache, progress, checkpoints, logs and results must remain under
`G:\lxy`. Pure in-memory full generation is prohibited.

## Local Readiness Gate

Before any SSH contact, require:

1. C2 validation passes and its decision is the frozen method boundary;
2. repaired GIFT R2G and PRESENT T1 seed0/seed1 topology gates pass;
3. the formal CSV has exactly three rows and differs from T1 only in scale and
   run identity;
4. generic remote readiness passes the `1000000/class`, 16-pair, disk-cache,
   progress and `G:\lxy` policy;
5. generated Windows scripts contain `cmd.exe /c`, never `cmd.exe /k`, and no
   generated project path outside `G:\lxy`;
6. the exact source commit contains all required assets and matches the
   worktree;
7. the exact source commit is independently verified on `origin/main`;
8. the local launch gate says `should_ssh=true`, `ssh_allowed=true` and
   `launch_authorized=true`.

The current GitHub live-SHA verification is temporarily unavailable. That
state must hold launch even when an earlier normal push command reported
success.

## Local Readiness Result

The completed launch gate is indexed as recent result `001`:

```text
run_id = i1_rtg3b_present80_one_to_one_formal_1000000_seed0_launch_gate_20260726
source_commit = b46afd0eaf40e007c70a3c4d2663b9c0c3818a45
status = hold
decision = innovation1_rtg3b_present_seed0_source_not_live_verified
evidence checks = 4/4 pass
readiness checks = 7/7 pass
publication checks = 0/2 pass
should_ssh = true
ssh_allowed = false
launch_authorized = false
live_remote_sha = null
remote_training = not started
```

The normal `git push origin main` command reported success for
`b46afd0eaf40e007c70a3c4d2663b9c0c3818a45`. An immediate live query and one
bounded eight-attempt push-recovery process then failed with classified
`transient_network` errors. Publication is therefore uncertain rather than
verified. The only authorized next action is to rerun the same launch gate
after GitHub connectivity returns. No SSH, alternate remote, source overlay,
protocol change or new training run is allowed while this gate remains held.

Local evidence:

```text
outputs/local_readiness/
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_launch_gate_20260726/
```

## Research Gate

After complete verified retrieval, all protocol checks must pass and:

```text
correct AUC >= 0.520
correct - corrupted topology >= +0.005
correct - no topology >= +0.005
```

Seed0 pass authorizes only an identical seed1 replication. Seed0 hold stops
RTG3-B; do not add samples, epochs, pairs, architecture changes or threshold
relaxation after seeing the result. A protocol failure allows repair only of
the failed protocol invariant before rerunning.

## Claim Boundary

A two-seed pass would support project-formal one-to-one P-layer topology
attribution with the same Runtime-E4 parameter geometry used by the general-
GF(2) branch. It would not prove zero-step shared-weight transfer, correct S-box
semantics, heterogeneous round-window adaptation, a Zhang/Wang reproduction,
SOTA performance, an attack, a breakthrough or universal arbitrary-SPN
adaptation.

## Artifacts

Planned remote root:

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726\
```

Planned local launch gate:

```text
outputs/local_readiness/
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_launch_gate_20260726/
```

Planned retrieved result:

```text
outputs/remote_results/
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726/
```

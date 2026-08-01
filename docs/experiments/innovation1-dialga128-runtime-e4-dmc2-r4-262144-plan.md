# Innovation 1 Dialga-128 Runtime-E4 DMC2 R4 262144 Plan

Date: 2026-08-01

## Status

```text
phase = running remotely; completion and gate pending
scale = 262144/class scale confirmation, not formal evidence
paper role = last scale gate before DFC1 formal preregistration
```

## Remote Launch Record

```text
launched_at       = 2026-08-01 23:42 CST
source_commit     = 9168cbd8881c539c3db9b5da9a96f83a5c25dc5b
github_main       = exact SHA match verified before launch
remote_host       = lxy-a6000 / DESKTOP-BBLPACJ
physical_gpu      = 0
remote_run_root   = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_dialga128_runtime_e4_dmc2_r4_262144_seed0_seed1_20260801
launch_gate       = pass / innovation1_dialga_dmc2_remote_launch_authorized
durable_start     = source_expected_commit.txt, readiness.txt, started.marker, progress.jsonl
local_monitor     = tmux:i1_dialga_dmc2_262k_monitor
retrieval_status  = running; no result archive or result claim yet
```

The historical standard clone at
`G:\lxy\blockcipher-structure-adaptive-nd` was found dirty and behind its
upstream, so it was excluded without modification. The launch instead created
the run-owned `source` clone directly from GitHub, detached it at the exact
verified commit, and then invoked the committed launcher. The local monitor is
responsible for completion detection, verified-branch retrieval or raw
fallback retrieval, local re-adjudication, result indexing, and the visual-QA
pending marker. The main task must not SSH-poll this run.

## Research Question

Does the Dialga prefix-r4 Runtime-E4 topology margin survive a fourfold increase
from DMC1's `65536/class` to `262144/class` while every model, seed, data,
optimization, and gate field remains frozen?

DMC1 is the same-budget anchor. Its fallback-retrieved remote gate passed every
protocol and research check for both seeds, with correct-minus-corrupted AUC
margins `+0.013189230/+0.014329014`. DMC2 changes only training and validation
scale. It does not change the observed round boundary and does not add final
tests; those belong to DFC1 only after DMC2 passes.

## Frozen Matrix

Two seeds by three models, exactly six rows:

| role | model key | purpose |
| --- | --- | --- |
| correct | `runtime_spn_e4_equivariant_true` | exact Dialga runtime topology |
| corrupted | `runtime_spn_e4_equivariant_corrupted` | deterministic wrong topology |
| generic | `autond_dbitnet2023` | same-protocol generic architecture baseline |

```text
cipher / rounds      = Dialga-128 parent / encrypted prefix-r4
input difference     = 0x40
pairs per sample     = 4 independent ciphertext pairs
train                = 262144/class = 524288 total rows per model/seed
cross-key validation = 65536/class = 131072 total rows per model/seed
fresh final test     = disabled; DFC1-only requirement
seeds                = 0, 1
train key            = 0x00 repeated 32 bytes
validation key       = 0x11 repeated 32 bytes
negative mode        = encrypted_random_plaintexts
sample structure     = independent_pairs
epochs / batch       = 10 / 64
loss / optimizer     = MSE / Adam, lr 1e-4, weight decay 1e-5
checkpoint           = best validation AUC, restored
Runtime-E4 window    = round_start 2, rounds 2
execution            = remote A6000 GPU0, disk-backed cache/progress/reuse
```

The run must create four train/validation caches shared by seed and split,
perform eight parameter-matched cache reuses, restore six nonempty checkpoints,
and emit six result rows with sixty epoch-history rows. All files remain under
`G:\lxy`.

## Preregistered Gate

For each seed independently:

```text
correct AUC >= 0.900
correct - corrupted AUC >= +0.005
correct - AutoND AUC >= +0.010
```

Protocol validity additionally requires the exact pushed source commit, frozen
six-row plan, four disk caches, eight cache reuses, six checkpoints, exact row
counts, fixed keys/difference/negatives, and no independent-final-test fields.
A seed cannot be rescued by averaging.

## Decisions

- Pass: preregister DFC1 at at least `1000000/class`, multiple seeds, and five
  fresh independent final-test repeats under a third fixed key.
- Hold: stop mechanical scaling and inspect restored-best histories and cache
  equivalence. Do not change rounds, pairs, models, or thresholds.
- Invalid: repair only the failed source, plan, cache, checkpoint, or result
  binding and rerun DMC2 unchanged under a new unique run id if needed.

## Claim Boundary

DMC2 is a two-seed `262144/class` scale confirmation. It is not formal or
paper-scale evidence, not a full-round Dialga result, not a key-recovery result,
not a SOTA claim, and not evidence for a single universal SPN network. Only a
subsequent DFC1 run with at least `1000000/class` and independent final tests can
serve as the formal Dialga performance row in the manuscript.

## Executable Next Action

Run focused plan/gate/package/readiness tests, commit and push the exact assets,
execute the fail-closed launch gate against GitHub `main`, launch the clean
run-owned clone on physical GPU0 using `cmd.exe /c`, confirm one durable start
artifact, and leave completion/retrieval to the local tmux watcher.

# Innovation 1 PRESENT Runtime-E4 RTG3-B Formal Seed0 Plan

```text
status = retry1 completed / verified-branch retrieved / locally re-adjudicated
phase = RTG3-B
original_remote_training = stopped after duplicate-instance contamination
original_remote_launch_source = 233b2e2986578bb66bb95055f380a3ed21cbff1d
retry_run_id = i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_20260727
retry_research_protocol = unchanged
retry_training_source = 3a34c74e2c5e3240bf3b4a84b91ec8a2e08fc706
postprocess_recovery = 902ad01378f6a88906e108b4aca798676800c098
result = pass / innovation1_runtime_spn_present_formal_seed0_supported
visual_qa = pass / visual-qa-redraw
next = identical 1000000/class seed1 replication
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

If GitHub live-SHA verification is unavailable, the gate must hold launch even
when a normal push command reports success. RTG3-B advanced only after a later
live query returned the exact source commit.

## Launch Result

The original completed launch gate was indexed on 2026-07-26:

```text
run_id = i1_rtg3b_present80_one_to_one_formal_1000000_seed0_launch_gate_20260726
source_commit = 233b2e2986578bb66bb95055f380a3ed21cbff1d
status = pass
decision = innovation1_rtg3b_present_seed0_remote_launch_authorized
evidence checks = 4/4 pass
readiness checks = 7/7 pass
publication checks = 2/2 pass
should_ssh = true
ssh_allowed = true
launch_authorized = true
live_remote_sha = 233b2e2986578bb66bb95055f380a3ed21cbff1d
remote_launcher_returned = 2026-07-26T14:18:19+08:00
bounded_start_confirmation = pass at 2026-07-26T14:18:39+08:00
remote_training = started
tmux_session = i1_rtg3b_present80_formal_seed0_monitor
```

The first source-publication attempt for `b46afd0e` was temporarily unverifiable
after eight classified `transient_network` failures. The scoped status commit
`233b2e29` then pushed normally, a live query matched its exact SHA, and the
full fail-closed gate passed before any SSH contact. The remote launcher and
run-owned clean clone both checked out that exact commit under `G:\lxy`.

The original local tmux monitor received the remote started marker after eight
bounded checks and owned log synchronization until the failed marker appeared.
It terminated with `remote_failed.marker`; it did not retrieve a verified
result branch.

The original seed1 successor observed the seed0 failure and terminated with
`seed1_not_launched.marker`. The repaired retry1 seed1 package remains
fail-closed and may launch only after complete retry1 seed0 retrieval,
rendered-pixel visual QA, exact seed-only plan equivalence, unchanged protected
training paths, remote disk-cache readiness and a live `origin/main` SHA equal
to the pinned retry source commit.

```text
seed1 plan = docs/experiments/
  innovation1-present80-runtime-e4-rtg3b-1000000-seed1-plan.md
successor watcher = configs/remote/generated/
  monitor_i1_rtg3b_seed1_after_seed0_retry1_20260727.sh
seed1 tmux = i1_rtg3b_present80_formal_seed1_retry1_monitor
```

Local evidence:

```text
outputs/local_readiness/
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_launch_gate_20260726/
outputs/remote_results_incomplete/
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726_monitor/
```

## Original Run Protocol Incident

The original seed0 run is invalid research evidence. The launcher created a
one-time task for `23:59` and also invoked it immediately with `schtasks /Run`.
The first training instance crossed midnight, so the still-active time trigger
started a second instance against the same run root:

```text
task = I1_RTG3B_PRESENT80_S0_GPU0
scheduled trigger = 2026-07-26 23:59:00
duplicate PID = 21084
duplicate creation = 2026-07-26 23:59:08
shared paths = progress / results / checkpoints
```

The first instance had already produced three result rows and
`validate-results` reported `status=pass`, but the post-training gate then
failed before archiving:

```text
ModuleNotFoundError: No module named 'blockcipher_nd'
```

The second instance subsequently truncated `results.jsonl` to zero bytes and
restarted `progress.jsonl`. Because both instances owned the same evidence
paths, neither the original three-row file nor the remaining checkpoints can
be authenticated as a complete single-writer run. The scheduled task was
ended, the duplicate process was confirmed absent and the seed1 successor
stopped without launching.

A pre-overwrite monitor snapshot is retained only as diagnostic context:

| Role | AUC |
| --- | ---: |
| correct topology | `0.749477538094` |
| corrupted topology | `0.601527151514` |
| no topology | `0.597290895286` |

```text
correct - corrupted = +0.147950386580
correct - no topology = +0.152186642808
```

These values do not pass or fail C3. They must not be reported as a retrieved,
plan-aligned formal result. The raw post-incident snapshot is preserved under:

```text
outputs/remote_results_incomplete/
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726_protocol_invalid_20260727/
```

## Retry1 Protocol Repair

Retry1 keeps cipher, round, seed, keys, data arrays, pair organization, models,
loss, optimizer, epochs, checkpoint rule and research thresholds unchanged.
Only run ownership and post-processing execution are repaired:

1. a new run id gives results, progress, checkpoints, logs and archive an
   independent root;
2. an atomic `run.lock` directory permits only one writer and any existing
   started marker fails closed before evidence paths are touched;
3. after immediate `schtasks /Run`, the future one-time trigger is disabled
   before the launcher returns;
4. `PYTHONPATH=%SOURCE_ROOT%\src` makes the gate CLI importable;
5. the original seed0 cache may be reused only after metadata SHA-256, NPY
   shapes, dtypes and label counts match the frozen protocol.

```text
retry run = i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_20260727
retry launch gate = i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_launch_gate_20260727
retry cache source = original seed0 cache, validated read/reuse only
retry results/checkpoints/logs = independent retry1 root
```

The retry remains a protocol repair, not a new research variable. A clean
seed0 pass authorizes only the matching retry1 seed1 replication. A research
hold stops C3; another protocol failure permits repair only of the failed
invariant.

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

Original protocol-invalid remote root:

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726\
```

Retry1 remote root:

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_20260727\
```

Retry1 local launch gate:

```text
outputs/local_readiness/
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_launch_gate_20260727/
```

Planned retry1 retrieved result:

```text
outputs/remote_results/
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_20260727/
```

## Retry1 Post-Processing Recovery

Retry1 completed all three five-epoch model rows and remote plan validation,
but the no-plot gate entrypoint still imported Matplotlib at module import time.
The remote `torch310` environment does not contain that optional plotting
dependency, so training stopped only in post-processing:

```text
correct topology AUC   = 0.749477538094
corrupted topology AUC = 0.601527151514
no-topology AUC        = 0.597290895286
remote validate-results status = pass, 3/3 rows
remote failure = ModuleNotFoundError: No module named 'matplotlib'
```

This is an active protocol recovery, not a new experiment and not authority to
retrain. Commit `8d05e8bc14dbe9bee8f8fb8cf4deeffa40accf26` defers all plotting
imports until `render_present_transfer_svg` is actually called. A subprocess
regression test blocks every Matplotlib import and proves that `--no-plot`
still writes the gate, summary, validation, history and progress artifacts.

The bounded recovery script is:

```text
configs/remote/generated/
  recover_i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_20260727.cmd
```

It must verify the original training commit and exact Matplotlib failure,
rerun plan validation and gate-only adjudication, strictly replay all three
existing checkpoints, create an immutable SHA-256 archive, and push the normal
`results/<run-id>` branch. It contains no training command. Only after verified
branch retrieval, local re-adjudication, rendered-pixel visual QA and result
index refresh may seed0 be treated as complete and the identical seed1 launch
gate be evaluated.

## Retrieved Retry1 Seed0 Result

Recovery commit `902ad01378f6a88906e108b4aca798676800c098`
ran only the frozen validation, gate, checkpoint-replay, archive and result-
branch steps against the existing retry1 artifacts. It did not regenerate data,
resume an optimizer or retrain any model. The normal verified result branch was
then retrieved under:

```text
outputs/remote_results/
  i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_20260727/
```

The immutable manifest verifies every archived file after CRLF normalization.
Local plan validation reports exactly three of three rows with no missing,
unexpected, duplicate or mismatched keys. All three best-validation-AUC
checkpoints load strictly into the frozen `442466`-parameter geometry and
replay their recorded histories and final metrics exactly.

| Role | Validation AUC | Validation accuracy | Difference from correct |
| --- | ---: | ---: | ---: |
| Correct PRESENT P-layer | `0.749477538094` | `0.680433` | - |
| Deterministically corrupted P-layer | `0.601527151514` | `0.568519` | `+0.147950386580` |
| No linear topology | `0.597290895286` | `0.566848` | `+0.152186642808` |

Every protocol check and all three frozen research checks pass. Relative to the
local T1 seed0 anchor, correct-topology AUC rises from `0.664596081` to
`0.749477538094`, while correct-minus-corrupted grows from `+0.093934059` to
`+0.147950386580` and correct-minus-no-topology grows from `+0.110160828` to
`+0.152186642808`. The result therefore survives the scale increase with a
large topology-attribution margin rather than merely crossing the absolute
signal threshold.

The final retrieved SVG was rendered to `1940 x 993` pixels and inspected with
`visual-qa-redraw` at full composition and readable left/right detail views.
Chinese glyphs, title, protocol subtitle, decision line, values, axes, random
baseline, attribution threshold and claim-scope caption are readable without
overlap, clipping or an ambiguous scale. No redraw was required, and
`visual_qa_pending.marker` was replaced by `visual_qa_passed.marker`.

This is strong single-seed project-formal evidence that the Runtime-E4 model
uses the correct externally supplied one-to-one PRESENT topology. It is an
attribution result, not an absolute-performance breakthrough: accuracy
`0.680433` is below the older PRESENT-specific InvP-only seed0 result
`0.721264`, and the strict Zhang/Wang `0.7205` reference has not been reproduced
under this route. It is also not paper-scale evidence or proof of correct S-box
semantics, unseen-cipher transfer or arbitrary-SPN adaptation.

Evidence-backed next action:

```text
question = does the seed0 topology advantage replicate at seed1?
anchor = this verified retry1 seed0 result
change = seed only, 0 -> 1
fixed = PRESENT-80 r7, keys, difference, 1000000/class train,
        500000/class validation, 16 pairs, three controls, Runtime-E4,
        442466 parameters, 5 epochs, optimizer, loss and checkpoint rule
execution = remote A6000 through the strict seed1 publication/launch gate
advance = valid archive/checkpoints, correct >= 0.520,
          correct-corrupted >= 0.005 and correct-no-topology >= 0.005
stop = research hold stops RTG3-B; protocol failure permits evidence repair only
blocked = no extra samples, pairs, epochs, model rescue or C5 before seed1 closes C3
```

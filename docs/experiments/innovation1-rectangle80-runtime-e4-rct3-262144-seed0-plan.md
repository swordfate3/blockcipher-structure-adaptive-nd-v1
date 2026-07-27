# Innovation 1 RECTANGLE-80 RuntimeE4 RCT3 Scale Seed0 Plan

Date: 2026-07-27

## Status

```text
stage         = conditionally prepared; not launched
run_id        = i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727
execution     = remote GPU1 only
authority     = completed, retrieved, plan-aligned RCT2 seed0 pass plus visual QA
current gate  = closed while RCT2 remains running or visual-QA-pending
claim scope   = single-seed 262144/class scale diagnostic; not formal evidence
```

## Research Question

If the correct RECTANGLE runtime topology passes RCT2 at `65536/class`, does
the same candidate preserve its absolute signal and attribution margins when
only the training and validation sample counts increase by four?

This is a scale-stability test, not a new architecture, feature, differential,
optimizer or benchmark experiment. A result is interpretable only if the RCT2
source result has been retrieved from its verified result branch, replayed
locally, rendered, and approved through `visual-qa-redraw`.

## Same-Budget Anchor And One Variable

Source run:

```text
i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725
```

RCT3 changes only:

```text
train samples/class      = 65536 -> 262144
validation samples/class = 32768 -> 131072
run identity/evidence text
```

The execution device remains the remote A6000 GPU1. No seed1 row is added at
this stage because the user selected the fast single-seed ladder.

## Frozen Protocol

```text
cipher                    = RECTANGLE-80
rounds                    = 6
input difference          = 0x0000002100010020
difference profile        = rectangle80_weng_repo_best_trail_r6
train key                 = 0x00000000000000000000
validation key            = 0x11111111111111111111
seed                       = 0
train rows                 = 524288 total = 262144/class
validation rows            = 262144 total = 131072/class
pairs per sample           = 4 independent ciphertext pairs
feature encoding           = ciphertext_pair_bits = 512 bits/sample
negative definition        = encrypted random plaintexts
epochs / batch             = 5 / 64
loss / optimizer           = MSE / Adam
learning rate              = 0.0001
weight decay               = 0.00001
checkpoint                 = restored best validation AUC
train evaluation interval  = every epoch
```

Runtime model options remain byte-for-byte identical to RCT2:

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
configs/experiment/innovation1/innovation1_spn_rectangle80_runtime_e4_scale_rct3_262144_seed0.csv
```

| Role | Model key | Intervention |
| --- | --- | --- |
| correct | `runtime_spn_e4_equivariant_true` | exact external cell, S-box and ShiftRow descriptor |
| corrupted | `runtime_spn_e4_equivariant_corrupted` | deterministic full-bit linear-topology corruption |
| no topology | `runtime_spn_e4_equivariant_independent` | retain cell/S-box metadata but remove linear relations |

All rows must retain `442466` trainable parameters.

## Conditional Launch Gate

The launch gate must recompute the RCT2 local gate from its retrieved
`results.jsonl` rather than trusting a status string. It requires:

1. exact RCT2 run identity, `status=pass`, and the frozen pass decision;
2. recomputed RCT2 gate equals `gate.local.json`;
3. `validation.local.json` reports three aligned rows and no errors;
4. `retrieved_from_verified_result_branch.marker` exists;
5. `visual_qa_passed.marker` exists after rendered-pixel inspection;
6. RCT3 differs from RCT2 only by sample scale and descriptive identity;
7. the RCT3 remote readiness report passes at exactly `262144/class`;
8. all required source assets are committed, match the worktree and are
   published at the captured source commit;
9. protected data, model, runner, training and gate paths are clean.

Until every local check passes:

```text
should_ssh       = false
ssh_allowed      = false or publication-dependent
launch_authorized= false
```

The conditional successor may wait on local RCT2 artifacts. It must stop on an
RCT2 hold/fail and must not contact the remote host until the launch gate emits
all three affirmative booleans.

## Remote Storage And Cache

The run uses a clean run-owned clone at the exact pushed commit:

```text
source = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727\source
run    = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727
```

All project artifacts remain under `G:\lxy`. Before training, the run creates
durable `features.npy`, `labels.npy`, cache metadata and progress events. The
exact parameter-matched cache is reused across correct, corrupted and
no-topology roles. The generated launcher uses `cmd.exe /c`, never `/k`; after
manual `/Run` it immediately disables the future one-time trigger. The run
script must acquire an atomic run-owned directory lock before touching logs,
results, checkpoints or cache, and it refuses any pre-existing started marker.

## Result Gate

After protocol validation, seed0 advances only when:

```text
correct AUC >= 0.55
correct - corrupted AUC >= +0.005
correct - no-topology AUC >= +0.005
```

The result adjudicator is:

```text
scripts/gate-runtime-spn-rectangle-medium --phase rct3
```

It requires exactly three rows, exact RCT3 scale, strict negatives, exact
descriptor/control modes, disk-backed `G:\lxy` cache, five complete epoch
histories, equal parameter geometry, finite metrics and restored best-AUC
checkpoints.

## Decision Routes

- `pass`: retrieve and visually approve the result, then freeze an otherwise
  identical seed0 `1000000/class` project-formal candidate. Do not call RCT3
  formal evidence and do not launch the formal candidate before its own plan.
- `hold`: stop the RECTANGLE scale ladder. Diagnose the failed absolute signal
  or attribution margin locally without increasing samples, pairs, epochs or
  rounds.
- `fail`: repair only the failed protocol/evidence check and rerun the frozen
  row or postprocessing step. Do not interpret invalid metrics.

## Required Artifacts

```text
results.jsonl
progress.jsonl
validation-plan.json
validation.json
gate.json
summary.json
history.csv
git_revision.txt
gpu_info.txt
torch_info.txt
readiness.txt
SHA256SUMS
retrieved_from_verified_result_branch.marker
validation.local.json
gate.local.json
curves.svg
visual_qa_passed.marker
```

The remote archive initially contains `visual_qa_pending.marker`. The local
watcher retrieves and verifies the archive, normalizes CRLF only in the
`SHA256SUMS` input stream, re-adjudicates with `--phase rct3`, renders the SVG,
refreshes both recent-result indexes and leaves the visual decision pending.
Only rendered-pixel `visual-qa-redraw` inspection may replace that marker with
`visual_qa_passed.marker` and complete result handling.

## Explicitly Blocked

- Do not launch before the retrieved RCT2 pass and visual-QA marker.
- Do not add seed1, another model, more pairs, more epochs or another round.
- Do not change the differential, keys, labels, negative definition or metric.
- Do not launch from a dirty clone, unpublished commit or source overlay.
- Do not call `262144/class` formal, paper-scale or definitive ceiling evidence.
- Do not make an attack, SOTA, breakthrough or universal-SPN claim.

## Recommended Next Action

Keep the local conditional successor prepared but closed. When the existing
RCT2 watcher retrieves a valid result, inspect its real `curves.svg` through
`visual-qa-redraw`. If and only if the recomputed RCT2 gate passes, validation
is aligned and the visual marker is approved, allow the successor to execute
the frozen RCT3 launch gate and hand the remote run to its own local watcher.

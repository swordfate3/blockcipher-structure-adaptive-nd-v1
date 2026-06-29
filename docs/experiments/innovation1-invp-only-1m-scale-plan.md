# Innovation 1 InvP-Only 1M Scale Plan

**Date:** 2026-06-29

**Status:** seed0+seed1 completed / retrieved / validated / postprocessed / route-confirmed

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives, `1000000/class` two-seed confirmation evidence. This is route-level confirmation, not a formal breakthrough claim.

## Motivation

Two medium-scale InvP-centered diagnostics now agree on the useful direction:
the SPN-adapted inverse-P nibble view improves over the Zhang/Wang MCND anchor,
while the extra pair-consistency aggregation is effectively tied with the
simpler InvP-only route.

Medium diagnostic evidence:

| Run | Seed | Baseline AUC | InvP-only AUC | Pair-consistency AUC | InvP-only delta over baseline | Pair delta over InvP-only |
|---|---:|---:|---:|---:|---:|---:|
| `i1_invp_centered_r7_262k_seed0_gpu1_20260628` | 0 | 0.784347 | 0.792105 | 0.792800 | +0.007758 | +0.000695 |
| `i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629` | 1 | 0.786113 | 0.792977 | 0.793216 | +0.006864 | +0.000239 |

Interpretation:

```text
The robust signal is the InvP/SPN structural representation.
The pair-consistency pooling is too small to justify the next 1M/class GPU slot.
```

## Research Question

Does the simpler `present_nibble_invp_only_spn_only` route preserve the
medium-scale advantage at `1000000/class`?

Primary comparison anchor:

```text
Completed same-protocol Zhang/Wang 1M seed0 baseline:
run_id = zhang_wang_present_r7_1m_official_cyclic_seed0_20260625
accuracy = 0.715281
calibrated_accuracy = 0.718555
AUC = 0.793897025948
loss = 0.549200775116
```

Secondary reference:

```text
Completed same-matrix p-aligned MCND 1M seed0:
run_id = i1_spn_present_mcnd_r7_1m_seed0_gpu1_retry1_20260626
present_nibble_paligned_mcnd AUC = 0.794619119358
delta over same-run baseline = +0.000708743544
```

## Planned Matrix

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv
```

Rows:

| Rank | Model key | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | strongest simple SPN/InvP candidate |

This is intentionally a single-row run. The same-protocol Zhang/Wang 1M seed0
baseline has already completed, so repeating it would spend GPU time without
adding a new comparison point.

Seed1 confirmation config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed1.csv
```

This file was launched after seed0 cleared the strong `>= +0.003` AUC gate over
the completed Zhang/Wang 1M anchor. It is intentionally single-row and
same-protocol so the only planned change is `seed = 1`. Seed1 has now been
retrieved, validated, and postprocessed; do not relaunch this same confirmation
job.

## Fixed Protocol

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `7` |
| Seed | `0`, `1` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs per sample | `16` |
| Samples per class | `1000000` |
| Feature encoding | `ciphertext_pair_bits` |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Restore best checkpoint | `true` |
| Fast train eval | `--train-eval-interval 0` |

Do not change validation data, labels, negative mode, metric computation,
keying protocol, or Zhang/Wang Case2 sample construction.

## Decision Gates

Primary comparison:

```text
InvP-only 1M AUC - completed Zhang/Wang 1M AUC
```

Gates:

| Condition | Interpretation | Action |
|---|---|---|
| `>= +0.003` AUC over Zhang/Wang 1M anchor | meaningful paper-scale single-seed improvement | run 1M seed1 confirmation |
| `+0.001` to `+0.003` AUC | weak but positive scale survival | run seed1 before claiming route strength |
| within `±0.001` AUC | medium signal mostly collapses at 1M | prefer simpler baseline or revisit architecture |
| below baseline by `> 0.001` AUC | route does not survive scale | discard InvP-only as main 1M candidate |

Claim scope:

```text
1000000/class two-seed confirmation evidence.
Not formal route evidence without the planned multi-seed/attribution write-up.
Not a breakthrough claim.
```

## Execution Plan

1. Add the single-row 1M CSV matrix.
2. Add a remote config for `cuda:1`, `batch_size=1024`, shared disk-backed cache, and `train_eval_interval=0`.
3. Run a small CPU smoke using a dedicated smoke CSV or the existing seed1 smoke pattern; do not use the 1M CSV for smoke.
4. Commit and push plan/config before launch.
5. Launch from the pushed commit into a run-owned clean clone under `G:\lxy\blockcipher-structure-adaptive-nd-runs`.
6. Use `cmd.exe /c`, not `cmd.exe /k`.
7. Use a local tmux monitor to retrieve artifacts automatically.
8. After retrieval, validate plan alignment, generate history/curves locally, update this document, commit, and push.

Conditional seed1 readiness:

| Field | Value |
|---|---|
| Plan CSV | `configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed1.csv` |
| Remote config | `configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json` |
| Intended run ID | `i1_invp_only_r7_1m_seed1_gpu1_20260629` |
| Status | launched / retrieved / validated / postprocessed |
| Launch trigger | seed0 validated AUC delta over Zhang/Wang 1M anchor `>= +0.001` |
| Strong interpretation | seed0 delta `>= +0.003`; launch seed1 as paper-scale confirmation |
| Weak-positive interpretation | seed0 delta `+0.001` to `+0.003`; launch seed1 before any route-strength claim |
| No-launch trigger | seed0 delta `< +0.001`; use the DDT graph conditional plan or return to the baseline route |

The seed1 config is now completed confirmation evidence. Do not report it as
formal route evidence until the route-level summary records the two-seed table,
claim scope, remaining risks, and next formal design decision.

## Remote Launch Record

The 1M InvP-only run has been launched from a pushed GitHub commit and handed
off to a local tmux monitor.

| Field | Value |
|---|---|
| Run ID | `i1_invp_only_r7_1m_seed0_gpu1_20260629` |
| Source commit | `d9454ea17c6e0446ea934a3fb7ebfd6f56720aee` |
| Plan CSV | `configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv` |
| Remote config | `configs/remote/innovation1_spn_present_invp_only_r7_1m_seed0_gpu1_20260629.json` |
| Remote root | `G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_invp_only_r7_1m_seed0_gpu1_20260629` |
| Local monitor | `tmux: monitor_i1_invp_only_1m_20260629` |
| Local retrieval root | `outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/` |
| Device | `cuda:1` |
| Expected rows | `1` |

Launch health:

```text
remote scheduled task = blockcipher_i1_invp_only_1m_20260629
initial logs appeared = yes
progress.jsonl appeared = yes
stdout/stderr files appeared = yes
immediate failed marker = no
monitor status = running
```

Status checkpoint, 2026-06-29:

```text
local retrieval root exists = yes
retrieved result JSONL = no
retrieved done marker = no
retrieved failed marker = no
postprocess allowed = no
main-thread intervention needed = no
```

Current handoff:

```text
The local low-frequency watcher should continue to wait for result_ready and
run postprocess only after the expected 1 result row is present. Do not launch
the prepared seed1 config and do not implement the DDT graph route until the
validated seed0 branch gate chooses exactly one branch.
```

Prepared branch state:

```text
seed1 readiness gate = pass as of 2026-06-29
DDT graph conditional implementation guardrails = documented in
  docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md
```

Bounded local monitor-health check for sub-agent/watchers:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health \
  --run-id i1_invp_only_r7_1m_seed0_gpu1_20260629 \
  --tmux-session monitor_i1_invp_only_1m_20260629 \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv \
  --plan-doc docs/experiments/innovation1-invp-only-1m-scale-plan.md
```

This command reads only local monitor artifacts and performs at most one local
`tmux has-session` check. It must not be used as a main-thread polling loop.
When the status is `result_ready`, the JSON report includes a
`postprocess_command` that sub-agents/watchers can execute directly.
If the status is `completed_missing_results`, a done marker has appeared but
the retrieved JSONL is still missing; do not postprocess until the JSONL exists.
If the status is `results_empty`, the JSONL path exists but has no non-empty
rows; do not postprocess until at least one result row is present.

Post-retrieval gate to run automatically:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv \
  --results outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/results/i1_invp_only_r7_1m_seed0_gpu1_20260629.jsonl \
  --expected-rows 1 \
  --output outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_local_result_gate.json
```

Post-retrieval plotting command:

```bash
MPLCONFIGDIR=/tmp/mplconfig UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/results/i1_invp_only_r7_1m_seed0_gpu1_20260629.jsonl \
  --output outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_curves.svg \
  --history-csv outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_history.csv \
  --title i1_invp_only_r7_1m_seed0_gpu1_20260629
```

Post-retrieval branch gate command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-invp-result \
  --results outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/results/i1_invp_only_r7_1m_seed0_gpu1_20260629.jsonl \
  --output outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_branch_gate.json
```

Equivalent one-command local postprocess:

```bash
MPLCONFIGDIR=/tmp/mplconfig UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-invp-result \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv \
  --results outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/results/i1_invp_only_r7_1m_seed0_gpu1_20260629.jsonl \
  --output-dir outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629 \
  --run-id i1_invp_only_r7_1m_seed0_gpu1_20260629 \
  --expected-rows 1 \
  --update-plan-doc docs/experiments/innovation1-invp-only-1m-scale-plan.md
```

Result interpretation will compare against:

```text
Zhang/Wang 1M anchor AUC = 0.793897025948
p-aligned MCND 1M AUC   = 0.794619119358
```

The branch decision uses the Zhang/Wang 1M anchor as the primary reference.
The postprocess branch gate also reports `auc_delta_vs_paligned_mcnd_1m` as a
secondary context value, but that secondary delta must not by itself trigger
seed1 or DDT graph.

Conditional next step:

```text
If InvP-only 1M beats the Zhang/Wang 1M anchor by >= +0.003 AUC,
run `scripts/check-remote-readiness` on the prepared seed1 remote config, then
launch a 1M seed1 confirmation from the pushed commit.

If InvP-only is weakly positive by +0.001 to +0.003 AUC, run the same remote
readiness gate and launch seed1 before claiming route strength.

If InvP-only is tied within the weak band or below the Zhang/Wang anchor, shift
the next design iteration toward a true SPN-topology / DDT-aware graph backbone
rather than additional pair-consistency pooling.
```

## Retrieved Result Record

<!-- invp-postprocess:i1_invp_only_r7_1m_seed0_gpu1_20260629:start -->
### i1_invp_only_r7_1m_seed0_gpu1_20260629 Postprocess Result

| Field | Value |
|---|---|
| Run ID | `i1_invp_only_r7_1m_seed0_gpu1_20260629` |
| Postprocess status | `pass` |
| Validation status | `pass` |
| Branch status | `pass` |
| AUC | `0.797470988906` |
| Accuracy | `0.721264000000` |
| Calibrated accuracy | `0.721351000000` |
| Loss | `0.540575103607` |
| Delta vs Zhang/Wang 1M AUC | `0.003573962958` |
| Delta vs p-aligned MCND 1M AUC | `0.002851869548` |
| Decision | `launch_invp_seed1_confirmation` |
| Action | `launch_prepared_seed1_1m_config` |
| Next action branch | `seed1_confirmation` |
| Next action readiness command | `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness --config configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json` |
| Next action implementation aliases | `` |
| Next action implementation files | `` |
| Next action implementation checklist | `` |
| Next steps | `Update and commit the experiment plan with this retrieved result.; Run the remote readiness gate: UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness --config configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json; Launch configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json from the pushed commit.; Hand off seed1 monitoring and retrieval to a local tmux watcher or sub-agent.` |
| Claim scope | `1000000/class single-seed gate only; not formal multi-seed evidence and not a breakthrough claim` |
| Results JSONL | `outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/results/i1_invp_only_r7_1m_seed0_gpu1_20260629.jsonl` |
| Validation report | `outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_local_result_gate.json` |
| Branch gate | `outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_branch_gate.json` |
| Curves | `outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_curves.svg` |
| History CSV | `outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_history.csv` |
| Summary JSON | `outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_postprocess_summary.json` |
| Summary Markdown | `outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_postprocess_summary.md` |
<!-- invp-postprocess:i1_invp_only_r7_1m_seed0_gpu1_20260629:end -->

## Seed1 Confirmation Launch Record

The seed0 result cleared the strong single-seed gate:

```text
AUC = 0.797470988906
Delta vs Zhang/Wang 1M AUC = +0.003573962958
Decision = launch_invp_seed1_confirmation
```

Seed1 has therefore been launched as a confirmation run. This is still
paper-scale confirmation evidence, not a formal multi-seed breakthrough claim.

| Field | Value |
|---|---|
| Run ID | `i1_invp_only_r7_1m_seed1_gpu1_20260629` |
| Source commit | `de6c9851d61b74571c132c470f48d3499abe7f8a` |
| Plan CSV | `configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed1.csv` |
| Remote config | `configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json` |
| Remote task | `blockcipher_i1_invp_only_r7_1m_seed1_20260629` |
| Remote root | `G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_invp_only_r7_1m_seed1_gpu1_20260629` |
| Remote script root | `G:\lxy\blockcipher-structure-adaptive-nd\scripts\generated\remote` |
| Local monitor | `tmux: monitor_i1_invp_only_seed1_1m_20260629` |
| Local retrieval root | `outputs/remote_results/i1_invp_only_r7_1m_seed1_gpu1_20260629/` |
| Watcher owner | `sub-agent: Linnaeus` |
| Device | `cuda:1` |
| Expected rows | `1` |

Launch gates:

```text
remote readiness = pass
task scheduler command uses cmd.exe /c = yes
remote project/run artifacts under G:\lxy = yes
dataset cache root = G:\lxy\blockcipher-structure-adaptive-nd-runs\shared_dataset_cache
initial remote logs appeared = yes
launcher logs appeared = yes
local tmux monitor started = yes
```

Bounded local monitor-health check for seed1 watchers:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health \
  --run-id i1_invp_only_r7_1m_seed1_gpu1_20260629 \
  --tmux-session monitor_i1_invp_only_seed1_1m_20260629 \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed1.csv \
  --plan-doc docs/experiments/innovation1-invp-only-1m-scale-plan.md
```

When seed1 is retrieved, validate and compare it against seed0 and the
Zhang/Wang 1M anchor before making any route-strength claim. If seed1 confirms
the strong positive direction, the next planning step is a multi-seed summary
and a route-level evidence table; if seed1 collapses, treat seed0 as a positive
single-seed diagnostic and run an attribution check before scaling further.

## Stop / Continue Criteria

This automation should not run experiments indefinitely. Use these explicit
stop and branch rules after seed1 is retrieved.

### Run-Level Stop

Stop the current remote run and do not postprocess if any of these occur:

```text
failed marker appears
results JSONL is missing after a done marker
results JSONL line count is not the expected row count
validate-results status is not pass
negative_mode, sample_structure, validation key, metric, or plan row differs
from the configured protocol
```

Action:

```text
Record the failure mode, commit the experiment note, and do not launch the next
experiment until the failure is explained or a repair plan is written.
```

### Route-Level Stop Or Continue

Use Zhang/Wang 1M AUC `0.793897025948` as the primary anchor.

| Seed0 result | Seed1 result | Decision |
|---|---|---|
| `>= +0.003` AUC delta | `>= +0.003` AUC delta | confirm strong paper-scale route; write route-level summary and plan multi-seed/formal evidence |
| `>= +0.003` AUC delta | `+0.001` to `+0.003` AUC delta | keep route as positive but not strong; run one more seed or attribution before any claim |
| `>= +0.003` AUC delta | within `±0.001` | stop scaling InvP-only; treat seed0 as unstable positive diagnostic and run attribution/DDT route |
| `>= +0.003` AUC delta | below `-0.001` | stop InvP-only as main candidate; investigate variance/protocol artifacts before further scale |

The route is not a formal claim until it has at least:

```text
two or more retrieved and validated 1000000/class seeds
same protocol and strict encrypted-random-plaintext negatives
documented comparison against Zhang/Wang 1M anchor
no unresolved validation, retrieval, or checkpoint-selection issue
```

### Stage-Level Stop

Pause the automated research loop, rather than launching more GPU jobs, when:

```text
1. A route has two validated 1M/class seeds with consistent positive deltas and
   the next step requires a paper-level multi-seed design decision.
2. A route fails the seed1 confirmation gate and the next step requires a new
   architecture or data-hypothesis document.
3. The only remaining action is waiting for a remote result; in that case the
   watcher/sub-agent owns monitoring and the main thread must stop polling.
4. Evidence is contradictory, incomplete, fallback-retrieved, or not
   plan-aligned; in that case write the limitation first and do not claim
   route strength.
```

Stopping here means stopping the current automation branch, not abandoning the
project. The next branch must start from a written plan with a new hypothesis
and an explicit same-budget comparison.

<!-- invp-postprocess:i1_invp_only_r7_1m_seed1_gpu1_20260629:start -->
### i1_invp_only_r7_1m_seed1_gpu1_20260629 Postprocess Result

| Field | Value |
|---|---|
| Run ID | `i1_invp_only_r7_1m_seed1_gpu1_20260629` |
| Postprocess status | `pass` |
| Validation status | `pass` |
| Branch status | `pass` |
| AUC | `0.797347588554` |
| Accuracy | `0.721599000000` |
| Calibrated accuracy | `0.721855000000` |
| Loss | `0.540748940674` |
| Delta vs Zhang/Wang 1M AUC | `0.003450562606` |
| Delta vs p-aligned MCND 1M AUC | `0.002728469196` |
| Decision | `confirm_invp_two_seed_route` |
| Action | `pause_gpu_scaling_write_route_plan` |
| Next action branch | `formal_multiseed_or_attribution_plan` |
| Next action readiness command | `` |
| Next action implementation aliases | `` |
| Next action implementation files | `` |
| Next action implementation checklist | `` |
| Next steps | `Write route-level evidence summary and decide between formal multi-seed evidence plan or attribution plan before launching more GPU jobs.` |
| Claim scope | `1000000/class two-seed confirmation evidence; not formal route evidence and not a breakthrough claim` |
| Results JSONL | `outputs/remote_results/i1_invp_only_r7_1m_seed1_gpu1_20260629/results/i1_invp_only_r7_1m_seed1_gpu1_20260629.jsonl` |
| Validation report | `outputs/remote_results/i1_invp_only_r7_1m_seed1_gpu1_20260629/i1_invp_only_r7_1m_seed1_gpu1_20260629_local_result_gate.json` |
| Branch gate | `outputs/remote_results/i1_invp_only_r7_1m_seed1_gpu1_20260629/i1_invp_only_r7_1m_seed1_gpu1_20260629_branch_gate.json` |
| Curves | `outputs/remote_results/i1_invp_only_r7_1m_seed1_gpu1_20260629/i1_invp_only_r7_1m_seed1_gpu1_20260629_curves.svg` |
| History CSV | `outputs/remote_results/i1_invp_only_r7_1m_seed1_gpu1_20260629/i1_invp_only_r7_1m_seed1_gpu1_20260629_history.csv` |
| Summary JSON | `outputs/remote_results/i1_invp_only_r7_1m_seed1_gpu1_20260629/i1_invp_only_r7_1m_seed1_gpu1_20260629_postprocess_summary.json` |
| Summary Markdown | `outputs/remote_results/i1_invp_only_r7_1m_seed1_gpu1_20260629/i1_invp_only_r7_1m_seed1_gpu1_20260629_postprocess_summary.md` |
<!-- invp-postprocess:i1_invp_only_r7_1m_seed1_gpu1_20260629:end -->

## Route-Level Confirmation Summary

Both `1000000/class` InvP-only confirmation seeds were retrieved, validated,
postprocessed, and plan-aligned under the same PRESENT-80 r7 Zhang/Wang Case2
protocol with strict encrypted-random-plaintext negatives.

Primary anchor:

```text
Zhang/Wang 1M AUC = 0.793897025948
```

| Run | Seed | Accuracy | Calibrated accuracy | AUC | AUC delta vs Zhang/Wang 1M | AUC delta vs p-aligned MCND 1M | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `i1_invp_only_r7_1m_seed0_gpu1_20260629` | 0 | 0.721264 | 0.721351 | 0.797470988906 | +0.003573962958 | +0.002851869548 | pass |
| `i1_invp_only_r7_1m_seed1_gpu1_20260629` | 1 | 0.721599 | 0.721855 | 0.797347588554 | +0.003450562606 | +0.002728469196 | pass |

Interpretation:

```text
The InvP-only SPN-aligned representation has now passed the two-seed
1000000/class confirmation gate against the Zhang/Wang 1M anchor.
This is route-level confirmation evidence, not a formal breakthrough claim.
```

Immediate branch decision:

```text
Pause automatic GPU scaling for this branch.
Do not relaunch seed1.
Do not start another InvP-only 1M run without a new formal multi-seed or
paper-style attribution plan.
```

Next planning options:

```text
1. Write a formal multi-seed evidence plan for InvP-only, including seed count,
   attribution checks, checkpoint policy, and publication claim wording.
2. Write an attribution plan comparing raw MCND, p-aligned MCND, and InvP-only
   under matched protocol to isolate whether the gain comes from P-layer
   alignment, nibble grouping, architecture, or training variance.
3. If the next hypothesis changes architecture, start a new docs/experiments
   plan for DDT-aware or topology-aware SPN models before launching training.
```

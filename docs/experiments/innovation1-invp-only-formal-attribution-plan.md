# Innovation 1 InvP-Only Formal Attribution Plan

**Date:** 2026-06-30

**Status:** Branch B completed / 1M attribution controls support InvP alignment

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict
encrypted-random-plaintext negatives. This document decides the next stage
after the InvP-only `1000000/class` seed0+seed1 confirmation.

## Stage Decision

The current InvP-only route has enough evidence to stop blind GPU scaling, but
not enough evidence for a formal paper-style claim.

Decision:

```text
Do not launch another InvP-only 1M run immediately.
Do not relaunch seed1.
Do not switch to DDT-graph before preserving the InvP-only evidence chain.
Next stage = formal evidence and attribution planning.
```

This is a stage-level stop, not abandonment of Innovation 1. The purpose is to
turn the current positive route into a reproducible, attributable claim design
before spending the next GPU slot.

## Research Question

Current question:

```text
Can the InvP-only SPN-aligned representation be promoted from two-seed
confirmation evidence into formal route evidence, and what attribution controls
are needed before making a paper-style claim?
```

Current answer:

```text
The retrieved paper-scale attribution controls support true InvP/P-layer
alignment as the useful SPN structure signal. The route can move from
attribution planning into the next-stage choice: formal multi-seed evidence,
baseline variance, or a DDT/topology method-extension route.
```

The hypothesis to preserve:

```text
InvP(DeltaC), organized as PRESENT 4-bit SPN cells, gives the network a useful
structure-aligned view of the last-round differential state. The observed gain
should remain visible under fixed protocol, strict negatives, and matched
same-budget controls.
```

## Current Evidence Table

Primary same-protocol baseline:

```text
run_id = zhang_wang_present_r7_1m_official_cyclic_seed0_20260625
model = present_zhang_wang_keras_mcnd
accuracy = 0.715281
calibrated_accuracy = 0.718555
AUC = 0.793897025948
loss = 0.549200775116
status = retrieved / validated / plan-aligned 1M single-seed anchor
```

Secondary same-protocol internal reference:

```text
run_id = i1_spn_present_mcnd_r7_1m_seed0_gpu1_retry1_20260626
model = present_nibble_paligned_mcnd
AUC = 0.794619119358
status = retrieved / validated / plan-aligned 1M single-seed reference
```

Confirmed InvP-only route:

| Run | Seed | Model | Accuracy | Calibrated accuracy | AUC | AUC delta vs Zhang/Wang 1M | AUC delta vs p-aligned MCND 1M | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `i1_invp_only_r7_1m_seed0_gpu1_20260629` | 0 | `present_nibble_invp_only_spn_only` | 0.721264 | 0.721351 | 0.797470988906 | +0.003573962958 | +0.002851869548 | pass |
| `i1_invp_only_r7_1m_seed1_gpu1_20260629` | 1 | `present_nibble_invp_only_spn_only` | 0.721599 | 0.721855 | 0.797347588554 | +0.003450562606 | +0.002728469196 | pass |

Evidence level:

```text
1000000/class two-seed confirmation evidence.
Not yet formal route evidence.
Not a breakthrough claim.
```

## Existing Attribution Evidence

The `262144/class` SPN-only attribution matrix already supports the idea that
the signal is not only generic XOR information:

| Model | Role | AUC |
|---|---|---:|
| `present_zhang_wang_keras_mcnd` | Zhang/Wang-style baseline | 0.783228 |
| `present_nibble_paligned_spn_only` | DeltaC + true InvP(DeltaC) | 0.790665 |
| `present_nibble_delta_only_spn_only` | DeltaC-only control | 0.782918 |
| `present_nibble_invp_only_spn_only` | InvP-only route | 0.792536 |
| `present_nibble_shuffled_paligned_spn_only` | shuffled-P control | 0.784487 |

Interpretation:

```text
At 262144/class, InvP-only was the strongest attribution row.
Anchor > DeltaC-only and Anchor > shuffled-P also supports true P-layer
alignment as a useful structure signal.
```

This attribution evidence is medium diagnostic only. It is supportive context,
not formal proof.

## Claim Scope

Allowed wording after the current stage:

```text
InvP-only has produced consistent positive two-seed 1000000/class confirmation
evidence over the local Zhang/Wang 1M anchor under the same PRESENT-80 r7
Case2 protocol.
```

Not allowed yet:

```text
breakthrough
SOTA
formal superiority
proof that InvP-only is universally better
claim that DDT/topology-aware graph is unnecessary
claim that the paper reference 0.7205 has been formally surpassed
```

Formal route evidence would require:

```text
multi-seed design beyond two seeds
matched same-protocol baseline variance or a justified fixed anchor
attribution controls at paper scale or a clearly justified medium-scale
  attribution-to-formal bridge
complete artifacts, validation, gate summaries, and claim wording
```

## Planned Controls

The next formal plan must preserve the benchmark:

| Field | Fixed value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `7` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs per sample | `16` |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Restore best checkpoint | `true` |
| Fast train eval | `--train-eval-interval 0` |

Candidate controls for the next meaningful experiment:

| Priority | Control | Purpose | Suggested scale |
|---:|---|---|---|
| 1 | `present_nibble_invp_only_spn_only` seed2/seed3 | estimate route variance | `1000000/class` |
| 2 | `present_zhang_wang_keras_mcnd` seed1 | estimate baseline variance if needed | `1000000/class` |
| 3 | `present_nibble_delta_only_spn_only` | test generic XOR explanation | `262144/class` repeat or `1000000/class` selected control |
| 4 | `present_nibble_shuffled_paligned_spn_only` | test false alignment / parameter count | `262144/class` repeat or `1000000/class` selected control |
| 5 | `present_nibble_ddt_graph` vs shuffled control | test next topology/DDT hypothesis | start at `262144/class` only after a new implementation plan |

## Metric And Gate Policy

Primary metric:

```text
validation AUC
```

Secondary metrics:

```text
calibrated_accuracy
fixed-threshold accuracy
validation loss
train/validation history for overfit inspection
```

Checkpoint policy:

```text
checkpoint_metric = val_auc
restore_best_checkpoint = true
```

Route confirmation gate already met:

```text
seed0 AUC delta vs Zhang/Wang 1M >= +0.003
seed1 AUC delta vs Zhang/Wang 1M >= +0.003
```

Formal-evidence continue gate:

```text
If an additional InvP-only seed is launched, require AUC delta vs Zhang/Wang
1M anchor >= +0.002 on the new seed, or require a baseline-variance analysis
before weakening/strengthening the claim.
```

Attribution continue gate:

```text
InvP-only or true-P aligned route remains above DeltaC-only and shuffled-P
controls by >= +0.001 AUC under matched protocol.
```

Stop gate:

```text
If new large-scale controls show the gain is explained by DeltaC-only,
shuffled-P, or baseline variance, stop formalizing InvP-only as the main route
and move to DDT-aware/topology-aware SPN models with a new plan.
```

## Next Branch Options

### Branch A: Formal Multi-Seed Evidence

Purpose:

```text
Estimate whether the two-seed InvP-only advantage is stable enough for
paper-style route evidence.
```

Lean matrix:

| Row | Model | Seeds | Scale |
|---:|---|---|---|
| 0 | `present_nibble_invp_only_spn_only` | 2, 3 | `1000000/class` |
| 1 | `present_zhang_wang_keras_mcnd` | 1 if baseline variance is needed | `1000000/class` |

Do not launch this branch until a remote config and readiness gate are written.

### Branch B: Paper-Style Attribution

Purpose:

```text
Prove the gain comes from SPN/InvP structure rather than generic difference
features or extra capacity.
```

Lean matrix:

| Row | Model | Role | Scale |
|---:|---|---|---|
| 0 | `present_nibble_invp_only_spn_only` | confirmed route | `1000000/class` seed2 or selected control |
| 1 | `present_nibble_delta_only_spn_only` | generic XOR control | `1000000/class` selected control if GPU budget allows |
| 2 | `present_nibble_shuffled_paligned_spn_only` | false-alignment control | `1000000/class` selected control if GPU budget allows |

This branch is more informative for paper writing than immediately adding more
InvP-only seeds, but it costs more GPU time if all controls are scaled to 1M.

### Branch C: DDT-Aware / Topology-Aware New Route

Purpose:

```text
Test whether explicit S-box DDT priors and P-layer graph topology add
information beyond the now-confirmed InvP-only route.
```

This is a new hypothesis and must start from a separate plan plus local smoke.
First non-smoke scale should be `262144/class`, not 1M.

## Current Stage Action

Chosen action for the next meaningful run:

```text
Branch B: Paper-Style Attribution.
Prepare and launch a lean 1000000/class attribution-control matrix with:
  - present_nibble_delta_only_spn_only
  - present_nibble_shuffled_paligned_spn_only
```

Rationale:

```text
The current InvP-only route already passed the two-seed confirmation gate.
The next valuable work is to test whether the observed gain is really explained
by true InvP/P-layer alignment, rather than generic DeltaC information or false
alignment with extra tokens.
```

Prepared artifacts:

| Field | Value |
|---|---|
| Plan CSV | `configs/experiment/innovation1/innovation1_spn_present_invp_attribution_controls_r7_1m_seed0.csv` |
| Remote config | `configs/remote/innovation1_spn_present_invp_attribution_controls_r7_1m_seed0_gpu0_20260630.json` |
| Run ID | `i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630` |
| Expected rows | `2` |
| Device | `cuda:0` |
| Scale | `1000000/class` |
| Claim scope | paper-scale attribution controls, not breakthrough or formal evidence alone |

Gate:

```text
InvP-only remains structurally attributable if the completed InvP-only seed0/1
AUC stays above both DeltaC-only and shuffled-P controls by >= +0.001 under the
same protocol.

If either control reaches or exceeds the InvP-only band, the current claim must
be weakened: the gain may be explained by generic DeltaC information, false
alignment, extra tokens, or seed variance.
```

## Post-Retrieval Stage-Gate Playbook

Bounded local health check for watcher/sub-agent:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health \
  --run-id i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630 \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_attribution_controls_r7_1m_seed0.csv \
  --plan-doc docs/experiments/innovation1-invp-only-formal-attribution-plan.md \
  --plan-doc docs/experiments/innovation1-invp-route-level-evidence-summary.md \
  --expected-rows 2 \
  --postprocess-kind invp_attribution
```

When the status is `result_ready`, execute the emitted `postprocess_command`.
Do not postprocess `completed_missing_results`, `results_empty`, or
`results_incomplete`.

The attribution-control watcher must run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-invp-attribution-controls \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_attribution_controls_r7_1m_seed0.csv \
  --results outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630.jsonl \
  --output-dir outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630 \
  --run-id i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630 \
  --expected-rows 2 \
  --update-plan-doc docs/experiments/innovation1-invp-only-formal-attribution-plan.md \
  --update-plan-doc docs/experiments/innovation1-invp-route-level-evidence-summary.md
```

The stage decision must be read from:

```text
outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/
  i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630_attribution_gate.json
  i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630_postprocess_summary.json
```

Do not make a manual claim from raw accuracy/AUC alone. Use the attribution gate
decision below.

### Decision A: `support_invp_structural_attribution`

Meaning:

```text
min(InvP-only seed0 AUC, InvP-only seed1 AUC)
  - max(DeltaC-only AUC, shuffled-P AUC)
>= +0.001
```

Interpretation:

```text
The paper-scale control rows support the claim that true InvP/P-layer aligned
SPN representation carries useful signal beyond generic DeltaC and false
alignment controls.
```

Action:

```text
Write a route-level attribution summary under docs/experiments/.
Do not call this a breakthrough or SOTA result.
Decide whether the next paper need is:
  A. formal multi-seed InvP-only variance,
  B. a Zhang/Wang baseline seed1 variance audit,
  C. a topology-aware network route to improve beyond InvP-only.
```

Claim scope allowed:

```text
InvP-only has two-seed 1000000/class positive confirmation and paper-scale
attribution controls supporting the SPN/InvP/P-layer alignment explanation.
```

Current branch note, 2026-07-03:

```text
The DDT graph method-extension branch has completed two 262144/class seeds as
weak diagnostic evidence and is not being promoted to 1M yet. The active
topology-aware network route stopped after seed1 because true-P graph did not
beat InvP-only or shuffled-P controls. Candidate-trail seed0 stopped under its
medium diagnostic gate, and bit-transition-spectrum seed0 also stopped. The
active method-extension route is now watcher-managed trail-family seed0:
i1_trail_family_r7_262k_seed0_gpu1_20260702.
Do not launch trail-family seed1, active-auxiliary seed0, S-box prior seed0, or
other fallback branches until trail-family seed0 is retrieved, validated,
plan-aligned, postprocessed, and gated.
```

### Decision B: `weak_attribution_support`

Meaning:

```text
InvP-only remains above controls, but by less than the +0.001 AUC attribution
margin.
```

Interpretation:

```text
The direction is still positive but too close for a strong route-level
attribution claim.
```

Action:

```text
Do not formalize InvP-only yet.
Prefer one targeted follow-up before switching routes:
  - repeat the control row that came closest to InvP-only,
  - or launch a baseline/InvP variance seed if the control result suggests
    seed variance is dominating the margin.
Keep the matrix lean; do not rerun every historical control.
```

### Decision C: `weaken_invp_structural_attribution`

Meaning:

```text
At least one control reaches or exceeds the InvP-only two-seed confirmation band.
```

Interpretation:

```text
The current InvP-only route may still be useful, but the claim that its gain is
specifically explained by true InvP/P-layer alignment is not supported by the
paper-scale controls.
```

Action:

```text
Stop formalizing InvP-only as the main Innovation 1 structure claim.
Do not launch more InvP-only scale runs as the next move.
Switch to the next SPN structure hypothesis with a fresh plan:
  - DDT-aware cell graph,
  - SPN topology/message-passing backbone,
  - richer candidate-trail/transition consistency representation.
The next new-route non-smoke scale should start at 262144/class, not 1M.
```

In all cases, after postprocess:

```text
Update this document with retrieved metrics and the attribution gate.
Run relevant tests.
Commit and push the docs/tooling changes.
Keep outputs/ artifacts ignored unless explicitly requested.
```

## Completion Checklist

This stage is complete when:

```text
this plan exists under docs/experiments/
the InvP-only seed0/seed1 evidence table is recorded
the claim scope is explicit
the next branch options are explicit
the selected Branch B attribution controls are explicit
relevant verification passes
the plan is committed and pushed
no result_ready artifacts remain unprocessed
no agent-authored tracked edits remain uncommitted
```

## Retrieved Attribution Control Result

<!-- invp-attribution-postprocess:i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630:start -->
### i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630 Attribution Control Result

| Field | Value |
|---|---|
| Run ID | `i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630` |
| Postprocess status | `pass` |
| Validation status | `pass` |
| Attribution status | `pass` |
| Decision | `support_invp_structural_attribution` |
| Action | `write_route_level_attribution_summary` |
| Interpretation | `InvP-only remains above DeltaC-only and shuffled-P controls at paper scale; true InvP/P-layer alignment is supported as the useful SPN structure signal` |
| InvP seed0 AUC | `0.797470988906` |
| InvP seed1 AUC | `0.797347588554` |
| InvP min AUC | `0.797347588554` |
| Max control AUC | `0.793621524954` |
| Attribution margin | `0.003726063600` |
| Required margin | `0.001000000000` |
| Next action branch | `route_level_attribution_summary` |
| Historical next steps from artifact | `Update the experiment plan with this attribution-control result.; Write a route-level summary: InvP-only two-seed confirmation plus paper-scale controls.; Decide whether formal multi-seed evidence or a new DDT/topology route is the next paper need.` |
| Current branch note | DDT graph screening completed as weak diagnostic evidence; topology-aware network seed1 stopped the route because true-P graph did not beat InvP-only or shuffled-P controls; candidate-trail seed0 and bit-transition-spectrum seed0 have both stopped under their medium diagnostic gates; active method-extension is now watcher-managed trail-family seed0 `i1_trail_family_r7_262k_seed0_gpu1_20260702`; do not launch trail-family seed1, active-auxiliary seed0, S-box prior seed0, or another fallback until trail-family seed0 is retrieved, validated, plan-aligned, postprocessed, and gated |
| Claim scope | `1000000/class attribution-control gate against completed InvP-only seed0/seed1; not formal route evidence by itself` |
| Results JSONL | `outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630.jsonl` |
| Validation report | `outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630_local_result_gate.json` |
| Attribution gate | `outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630_attribution_gate.json` |
| Curves | `outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630_curves.svg` |
| History CSV | `outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630_history.csv` |
| Summary JSON | `outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630_postprocess_summary.json` |
| Summary Markdown | `outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630_postprocess_summary.md` |

Control rows:

| Model | AUC | Delta vs Zhang/Wang 1M |
|---|---:|---:|
| `present_nibble_delta_only_spn_only` | `0.792064879854` | `-0.001832146094` |
| `present_nibble_shuffled_paligned_spn_only` | `0.793621524954` | `-0.000275500994` |
<!-- invp-attribution-postprocess:i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630:end -->

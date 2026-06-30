# Innovation 1 InvP Route-Level Evidence Summary

**Date:** 2026-06-30

**Status:** draft / waiting for 1M attribution-control retrieval

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict
encrypted-random-plaintext negatives. This document is the route-level evidence
summary template for the current InvP-only SPN structure route. It must not be
read as a completed formal claim until the pending attribution-control run is
retrieved, validated, postprocessed, and documented.

## Route Claim Under Test

Working claim:

```text
InvP(DeltaC), organized as PRESENT 4-bit SPN cells, is a useful
structure-adaptive representation for PRESENT/SPN neural distinguishers.
```

The claim is intentionally narrow:

```text
This is about a PRESENT-80 r7 same-protocol neural distinguisher route under
Zhang/Wang Case2 m=16 and strict encrypted-random-plaintext negatives.
```

It is not yet a claim that:

```text
InvP-only is universally best.
The method is SOTA.
The paper reference 0.7205 has been formally surpassed.
DDT/topology-aware graph routes are unnecessary.
The result is formal route evidence.
```

## Fixed Protocol

| Field | Value |
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

## Completed Anchor Evidence

Primary same-protocol local Zhang/Wang 1M anchor:

| Run | Model | Seed | Samples/class | Accuracy | Calibrated accuracy | AUC | Loss | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `zhang_wang_present_r7_1m_official_cyclic_seed0_20260625` | `present_zhang_wang_keras_mcnd` | 0 | 1000000 | 0.715281 | 0.718555 | 0.793897025948 | 0.549200775116 | retrieved / validated / plan-aligned |

Secondary same-protocol p-aligned MCND reference:

| Run | Model | Seed | Samples/class | AUC | Status |
|---|---|---:|---:|---:|---|
| `i1_spn_present_mcnd_r7_1m_seed0_gpu1_retry1_20260626` | `present_nibble_paligned_mcnd` | 0 | 1000000 | 0.794619119358 | retrieved / validated / plan-aligned |

## Completed InvP-Only Evidence

| Run | Seed | Model | Accuracy | Calibrated accuracy | AUC | Delta vs Zhang/Wang 1M | Delta vs p-aligned MCND 1M | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `i1_invp_only_r7_1m_seed0_gpu1_20260629` | 0 | `present_nibble_invp_only_spn_only` | 0.721264 | 0.721351 | 0.797470988906 | +0.003573962958 | +0.002851869548 | retrieved / validated / postprocessed / branch-gate pass |
| `i1_invp_only_r7_1m_seed1_gpu1_20260629` | 1 | `present_nibble_invp_only_spn_only` | 0.721599 | 0.721855 | 0.797347588554 | +0.003450562606 | +0.002728469196 | retrieved / validated / postprocessed / branch-gate pass |

Current evidence level:

```text
two-seed 1000000/class positive confirmation evidence
```

Allowed interpretation so far:

```text
The InvP-only SPN-aligned route has a stable positive signal over the local
same-protocol Zhang/Wang 1M anchor across two seeds.
```

Not allowed yet:

```text
formal route evidence
breakthrough
SOTA
proof of structural attribution
```

## Medium-Scale Attribution Context

The completed `262144/class` SPN-only attribution matrix supports the idea that
the useful signal is concentrated in `InvP(DeltaC)` rather than generic XOR
alone:

| Model | Role | AUC |
|---|---|---:|
| `present_zhang_wang_keras_mcnd` | Zhang/Wang-style baseline | 0.783228 |
| `present_nibble_paligned_spn_only` | `DeltaC + InvP(DeltaC)` | 0.790665 |
| `present_nibble_delta_only_spn_only` | DeltaC-only control | 0.782918 |
| `present_nibble_invp_only_spn_only` | InvP-only route | 0.792536 |
| `present_nibble_shuffled_paligned_spn_only` | shuffled-P control | 0.784487 |

This context is diagnostic only. It motivates the 1M attribution controls but
does not by itself prove paper-scale attribution.

## Pending Paper-Scale Attribution Controls

Pending run:

| Field | Value |
|---|---|
| Run ID | `i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630` |
| Plan CSV | `configs/experiment/innovation1/innovation1_spn_present_invp_attribution_controls_r7_1m_seed0.csv` |
| Remote config | `configs/remote/innovation1_spn_present_invp_attribution_controls_r7_1m_seed0_gpu0_20260630.json` |
| Expected rows | `2` |
| Scale | `1000000/class` |
| Status | launched / watcher-managed / waiting for retrieval |

Rows:

| Model | Purpose | Result |
|---|---|---|
| `present_nibble_delta_only_spn_only` | tests generic XOR/nibble difference explanation | pending |
| `present_nibble_shuffled_paligned_spn_only` | tests false alignment / extra-token explanation | pending |

Postprocess command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-invp-attribution-controls \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_attribution_controls_r7_1m_seed0.csv \
  --results outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630.jsonl \
  --output-dir outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630 \
  --run-id i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630 \
  --expected-rows 2 \
  --update-plan-doc docs/experiments/innovation1-invp-only-formal-attribution-plan.md
```

Primary gate:

```text
min(InvP seed0 AUC, InvP seed1 AUC)
  - max(DeltaC-only AUC, shuffled-P AUC)
>= +0.001
```

## Stage Decision Template

### If Attribution Gate Supports InvP

Required evidence:

```text
decision = support_invp_structural_attribution
validation_status = pass
attribution_status = pass
```

Route-level summary wording:

```text
InvP-only has two-seed 1000000/class positive confirmation and paper-scale
attribution controls supporting the SPN/InvP/P-layer alignment explanation.
```

Next choices:

```text
1. Add formal multi-seed variance evidence for InvP-only.
2. Add Zhang/Wang baseline seed1 if baseline variance must be quantified.
3. Start a new DDT/topology-aware route if the paper needs a stronger method
   beyond InvP-only rather than more confirmation.
```

### If Attribution Gate Is Weak

Required evidence:

```text
decision = weak_attribution_support
```

Interpretation:

```text
The route remains positive but the control margin is too small for a strong
structural attribution claim.
```

Next action:

```text
Run one targeted follow-up, usually the closest control row or a variance seed.
Do not expand into a large uncontrolled matrix.
```

### If Attribution Gate Weakens InvP

Required evidence:

```text
decision = weaken_invp_structural_attribution
```

Interpretation:

```text
The InvP-only route may still be useful, but its gain is not clearly explained
by true InvP/P-layer alignment under paper-scale controls.
```

Next action:

```text
Stop formalizing InvP-only as the main Innovation 1 structure claim.
Move to a fresh SPN structure hypothesis:
  - DDT-aware cell graph,
  - SPN topology/message-passing backbone,
  - candidate-trail/transition consistency representation.
Start the new route at 262144/class after smoke/readiness validation.
```

## Current Claim Boundary

Until the attribution-control run is retrieved and postprocessed, the strongest
allowed claim is:

```text
InvP-only has produced stable two-seed 1000000/class positive confirmation over
the local same-protocol Zhang/Wang 1M anchor.
```

The route is not yet:

```text
formal route evidence
paper-ready proof
SOTA
breakthrough
fully attributed
```

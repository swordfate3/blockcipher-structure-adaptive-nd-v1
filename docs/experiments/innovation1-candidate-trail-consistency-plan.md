# Innovation 1 Candidate-Trail Consistency Plan

**Date:** 2026-07-01

**Status:** planned / gated / do not launch while DDT-topology or pair-set
control is active

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict
encrypted-random-plaintext negatives.

## Why This Exists

The old active-pattern screen is archived and not launchable. It used a small
standalone active-statistics route and did not match the current remote workflow
or current official Case2 protocol. However, active S-box patterns and candidate
trail consistency remain useful SPN-structure signals if they are used as
attribution features or auxiliary evidence for a real-vs-random distinguisher.

This plan defines the re-entry route for that idea. It should not revive the old
active-only screen. It should test whether candidate trail / transition
consistency features add real distinguishing signal beyond the current
InvP/DDT/pair-set evidence routes.

## Current Evidence Boundary

Current strongest completed Innovation 1 route:

```text
present_nibble_invp_only_spn_only
evidence = two-seed 1000000/class positive confirmation
attribution = paper-scale controls support true InvP/P-layer alignment
```

Current active route:

```text
i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630
status = watcher-managed running
```

Prepared next route if DDT is tied or negative:

```text
i1_pairset_aggregation_control_r7_262k_seed0_gpu1_20260630
purpose = separate learned cross-pair consistency from frozen single-pair
          score aggregation
```

Candidate-trail consistency is therefore a later branch, not an immediate launch
candidate.

## Trigger

This plan becomes actionable only after one of these conditions:

```text
1. DDT/topology seed0 is retrieved and tied/negative, and pair-set aggregation
   control is also tied/negative or diagnostic-only.

2. DDT/topology is positive, but attribution needs a richer trail-consistency
   diagnostic before any formal method claim.

3. User explicitly chooses candidate-trail / active-pattern feature
   representation as the next route after the current active run is resolved.
```

Do not launch this route while:

```text
i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630
```

is still running.

## Research Question

Holding cipher, rounds, sample structure, negative mode, train/validation keys,
optimizer, scheduler, checkpoint metric, and scale fixed:

```text
Do candidate trail / transition consistency features expose SPN differential
propagation information that is not already captured by InvP-only, DDT graph,
or learned pair-set consistency?
```

## Single Hypothesis

For PRESENT r7 Case2 `m=16`, real differential samples should show more
coherent candidate-transition evidence across S-box cells and across the 16
pairs than encrypted-random-plaintext negatives.

The route should test exactly one factor:

```text
feature route = candidate trail / transition consistency
```

Do not mix this first test with new topology layers, new pair aggregation
training, changed negative samples, changed validation key, changed metric, or
new sample structure.

## Required Protocol

Use the current official protocol:

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `7` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs/sample | `16` |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Dataset storage | disk-backed cache with progress/reuse metadata |

The existing `spn-candidate-evidence` code path currently defaults to the
legacy masked Case2 protocol. That path must be migrated or explicitly
overridden before any current remote launch. A readiness gate must reject any
candidate-trail remote config that still uses:

```text
sample_structure = zhang_wang_case2_mcnd
validation_key = 0xffffffffffffffffffff
key_rotation_interval > 0
```

unless it is clearly labeled as a historical diagnostic that is not comparable
with the current official Case2 evidence chain.

## Feature Sketch

Build deterministic per-sample features from ciphertext pair data and
PRESENT/SPN structure:

```text
per cell:
  DeltaC nibble
  InvP(DeltaC) nibble
  active DeltaC / active InvP flags
  DDT legality / probability summaries
  candidate input-difference support
  top candidate transition margin

across cells:
  active count
  active-position histogram
  high-probability transition count
  impossible/low-probability transition count
  transition entropy or margin

across 16 pairs:
  mean/std/max of candidate consistency scores
  agreement of active positions
  disagreement between likely local transitions
```

These features should be deterministic and cacheable. The first implementation
should remain a diagnostic feature route, not a new large network.

## Minimal Matrix

First non-smoke scale:

```text
262144/class
seed = 0
```

Keep the matrix lean:

| Row | Model/route | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | current supported InvP anchor |
| 1 | `candidate_trail_consistency_linear` | feature-only linear diagnostic |
| 2 | `candidate_trail_consistency_mlp` | small nonlinear feature diagnostic |

Optional control if implementation cost is small:

| Row | Model/route | Role |
|---:|---|---|
| 3 | `candidate_trail_consistency_shuffled_cells` | false active-position/control alignment |

Do not include Zhang/Wang baseline in the first matrix unless a protocol audit
requires it. Compare against the current same-scale InvP anchor and previously
retrieved Zhang/Wang context.

## Gates

Primary metric:

```text
val_auc
```

Continue only if:

```text
best candidate-trail route >= InvP-only same-scale anchor + 0.001 AUC
and calibrated_accuracy is not worse than InvP-only
```

Weak continue:

```text
best candidate-trail route is best but margin < 0.001 AUC
```

Action:

```text
run a second 262144/class seed before any 1M scale
```

Stop:

```text
candidate-trail route <= InvP-only
or shuffled/false-alignment control matches the true route
```

Action:

```text
record as diagnostic/negative evidence and do not scale this branch
```

## Implementation Readiness Checklist

Before any remote launch:

```text
1. Add or migrate a current-protocol candidate-trail feature builder.
2. Ensure disk-backed feature cache writes features.npy / labels.npy or
   equivalent chunks plus metadata and progress JSONL.
3. Ensure parameter-matched cache reuse/resume behavior.
4. Add a tiny local smoke plan under configs/experiment/innovation1/.
5. Add unit tests for deterministic features, cache reuse, and protocol lock.
6. Add postprocess/gate script before remote launch.
7. Add remote config and generated cmd/monitor scripts only after smoke passes.
8. Run scripts/check-remote-readiness before launch.
9. Commit and push before launching from the pushed commit.
```

## Claim Scope

Until at least `1000000/class` multi-seed evidence exists, this route can only
support one of these statements:

```text
candidate-trail consistency smoke passed
candidate-trail consistency medium diagnostic positive
candidate-trail consistency tied with InvP-only
candidate-trail consistency negative under official Case2 protocol
```

It must not be described as:

```text
formal route evidence
breakthrough
SOTA
proof that trail features solve PRESENT r7
```

## Relation To Other Routes

This route is a data/feature representation branch. It should be interpreted
beside, not instead of, the structure-network branches:

```text
InvP-only                -> structure-aligned data representation
DDT graph                -> explicit S-box prior + topology network
pair-set aggregation     -> cross-pair evidence attribution
candidate-trail route    -> transition-consistency feature attribution
```

The useful paper direction is whichever route survives same-protocol scale,
seed, and attribution gates most cleanly.

# Innovation 1 E4-E6 Cross-SPN Evidence Synthesis

**Date:** 2026-07-15

**Status:** completed / reproducible no-new-training postprocess passed

## Question

Which effects survive the E4 typed-topology attribution study and the E5/E6
source-objective interventions?

This synthesis must keep two evidence scales separate:

```text
E4 representation evidence = remote 65536/class medium diagnostic, four cells
E5/E6 objective evidence   = local 8192/class diagnostic, target seeds2/3
new training               = none
pooled confidence interval = forbidden
```

## Inputs

```text
outputs/local_diagnostic/i1_cross_spn_e4_final_synthesis_20260715/gate.json
outputs/local_diagnostic/i1_cross_spn_e5_target_8192_source_seed0/gate.json
outputs/local_diagnostic/i1_cross_spn_e6_target_8192_source_seed0/gate.json
```

All inputs must have `status=pass`, `errors=[]`, applied research decisions,
and their frozen target seed/source seed mappings. E5 and E6 must have exactly
equal scratch and off-transfer AUCs on target seeds2 and 3 before they may be
treated as same-budget objective comparisons.

## Analysis Axis

For E5 topology-identity BCE and E6 functional margin, report each target seed:

- candidate minus off-transfer AUC and paired 95% CI;
- candidate minus objective-specific placebo AUC and paired 95% CI;
- candidate minus scratch AUC and paired 95% CI;
- point and CI gate decisions.

For E4, copy only the already synthesized pass counts and ranges for scratch,
source-topology, and target-topology comparisons. Do not recalculate or combine
E4 rows with E5/E6 rows.

## Gate

```text
typed topology attribution retained = E4 source topology 4/4 and target topology 4/4
source objective retained            = every E5 or E6 target cell beats off,
                                       placebo, and scratch under frozen gates
ordinary transfer retained           = candidate/off transfer beats scratch
                                       with positive CI on both target seeds
```

Expected decision branches:

```text
topology retained, objectives fail -> freeze controlled representation result;
                                      stop source-objective/scale search
topology fails                       -> invalid contradiction; audit E4 input
any objective fully passes           -> invalid contradiction with frozen E5/E6
                                      joint gates; audit input selection
```

## Planned Artifacts

```text
outputs/local_diagnostic/i1_cross_spn_e4_e6_source_objective_synthesis_20260715/
  results.jsonl
  cells.csv
  summary.json
  gate.json
  curves.svg
```

## Recommended Next Action

Run the deterministic postprocess from the three frozen gate files. If the
expected branch is confirmed, update the route verdict and use the synthesis as
the final Innovation 1 method/evidence summary. Do not launch E7, source seed1,
`65536/class`, `262144/class`, or `1000000/class` from this postprocess.

## Completion

The deterministic postprocess completed from the three frozen gate files:

```text
status   = pass
errors   = []
decision = typed_topology_representation_retained_source_objectives_rejected
```

The two evidence scales remain separate:

```text
E4 remote medium representation cells
  source-topology complete gate = 4/4
  target-topology complete gate = 4/4
  scratch-efficiency gate       = 2/4
  minimum source-topology delta = +0.009724145755
  minimum target-topology delta = +0.069083069451

E5/E6 local source-objective cells
  complete objective gate       = 0/4
  candidate > off complete gate = 0/4
  candidate > placebo gate      = 1/4
  candidate > scratch gate      = 4/4
```

The shared target scratch and off AUCs were exactly equal between E5 and E6 on
target seeds2 and 3, so the source-objective comparisons are same-budget and
plan-compatible. No E4 and E5/E6 score rows were pooled.

## Final Innovation 1 Result

```text
keep = cipher-spec-generated shared typed SPN representation
keep = attributable correct source topology and correct target topology
keep = ordinary PRESENT-to-GIFT transfer signal versus scratch at local budget
hold = one-epoch scratch efficiency as conditional E4 evidence only
stop = topology-identity BCE source objective
stop = functional topology-margin source objective
stop = source-objective rescue, E7, and mechanical scale
```

The supported contribution is a controlled representation result: a single
cipher-spec-generated typed SPN operator preserves source and target topology
value across two independently trained PRESENT checkpoints and four remote
GIFT target cells. Two explicit source objectives successfully changed their
intended source behavior but did not improve target adaptation. This is medium
diagnostic plus local controlled evidence, not formal, paper-scale, SOTA,
breakthrough, persistent-superiority, or end-to-end compute evidence.

Artifacts:

```text
outputs/local_diagnostic/i1_cross_spn_e4_e6_source_objective_synthesis_20260715/
  results.jsonl
  cells.csv
  summary.json
  gate.json
  curves.svg
```

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/synthesize-cross-spn-e4-e6 \
  --e4-synthesis outputs/local_diagnostic/i1_cross_spn_e4_final_synthesis_20260715/gate.json \
  --e5-gate outputs/local_diagnostic/i1_cross_spn_e5_target_8192_source_seed0/gate.json \
  --e6-gate outputs/local_diagnostic/i1_cross_spn_e6_target_8192_source_seed0/gate.json \
  --output-dir outputs/local_diagnostic/i1_cross_spn_e4_e6_source_objective_synthesis_20260715
```

## Final Next Action

Freeze the Innovation 1 experiment branch and move to paper-ready method,
limitations, protocol, and variance reporting. Do not launch E7, source seed1,
or any new `65536/class`/`262144/class`/`1000000/class` run from this evidence.
A future training route requires a new literature-backed method hypothesis and
a new adjudication decision, not parameter tuning or scale rescue.

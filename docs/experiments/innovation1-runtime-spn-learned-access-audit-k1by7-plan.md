# Innovation 1 Runtime SPN Learned-Access Audit K1-BY7

**Date:** 2026-08-01
**Status:** retry1 preregistered after protocol-invalid first execution
**Execution:** local CPU, frozen checkpoints, zero neural training

## Research question

K1-BY6 used an identifiable affine wrong P-layer control. Seed2 preferred the
correct PRESENT program by `+0.039343357` AUC, but seed3 reversed the ordering
by `-0.026506424`. K1-BY7 changes no model or data. It asks:

> At which existing internal representation does seed3 first lose a stable
> preference for the correct PRESENT P-layer semantics?

This is a mechanism audit, not a new model comparison or scale experiment.

## Frozen authority

The audit binds K1-BY3 and K1-BY6 plans, results, gates, validations and four
restored-best checkpoints by SHA-256. It reads the same K1-BY3 validation
arrays used by both experiments. Source files must remain byte-identical before
and after execution.

No checkpoint is fine-tuned. Correct and affine models are reconstructed from
the frozen runtime descriptor and loaded strictly from their own checkpoints.
Their full-model logit AUC must replay the source result within `1e-6` before
any internal probe is interpreted.

## Frozen taps

Forward hooks capture five representations in information-flow order without
changing the model forward method:

1. `linear_histogram`: per-stage, per-cell 16-bin difference histogram;
2. `linear_primitive_expert`: learned permutation-expert output;
3. `cell_fusion`: fused linear and S-box primitive cell representation;
4. `pooled_stage_summary`: attention/mean/max stage summary before projection;
5. `pre_classifier_representation`: raw-pair backbone plus gated primitive
   residual immediately before the final classifier.

The two inverse stages remain ordered as executed by the model. Every tap is
flattened per sample only for the fixed diagnostic probe.

## Frozen label-association probe

For each seed, condition and tap:

- even validation indices form the discovery half;
- odd validation indices form the untouched evaluation half;
- both halves contain exactly `512` positive and `512` negative rows;
- discovery features are standardized with epsilon `1e-6`;
- the direction is the standardized positive-class mean minus negative-class
  mean;
- evaluation AUC is computed from projection onto that frozen direction.

This closed-form probe has no gradient, epoch, learned neural parameter or
hyperparameter search. It is only an internal label-association diagnostic.

## Preregistered localization rule

At every tap and seed calculate:

```text
probe margin = correct probe AUC - affine probe AUC
```

The required correct-structure margin is `+0.005`. For seed3, report the first
tap in forward order below this margin. If all five taps pass but the source
logit margin remains negative, localize the loss to the final classifier. No
later tap or seed average may rescue an earlier loss.

Interpretation routes:

- first loss at `linear_histogram`: the chosen difference/data surface does
  not intrinsically favor the exact P-layer representation on seed3;
- first loss at `linear_primitive_expert` or `cell_fusion`: redesign expert
  normalization or semantic fusion, keeping pooling fixed;
- first loss at `pooled_stage_summary`: redesign the invariant aggregation
  contract without adding absolute identity;
- first loss at `pre_classifier_representation` or classifier: audit residual
  gating/readout and checkpoint selection;
- invalid source replay or artifact drift: repair only that invariant and rerun.

## Claim boundary and next action

K1-BY7 can identify a likely information-access failure location, but cannot
establish causal correction, formal-scale performance, transfer or a universal
SPN architecture. The next experiment may change exactly one module selected
by this audit and must retain K1-BY6's data, seeds, budget and affine control.
No GIFT, Dialga, remote scale or larger model is authorized by this audit alone.

Planned artifacts:

```text
outputs/local_audit/
i1_runtime_spn_learned_access_audit_k1by7_present_r7_seed2_seed3_20260801/
  preflight.json
  results.jsonl
  condition_comparison.csv
  gate.json
  validation.json
  summary.json
  progress.jsonl
  curves.svg
  visual_qa_render_report.json
```

## First execution: protocol invalid

The first zero-training execution completed all 20 internal probe rows, but its
fail-closed source-replay gate rejected one row:

```text
seed3 correct source AUC = 0.665543556
seed3 direct-logit replay = 0.665541649
absolute error            = 0.000001907 > 0.000001
```

All source digests, checkpoints, hook shapes, probe rows and balanced splits
passed. The mismatch came from replaying AUC on raw logits while the project
training evaluator records AUC on `float32 sigmoid(logit)` probabilities.
Although sigmoid is monotonic, float32 conversion creates a small number of
ties and therefore changes the rank-based AUC by about two millionths. A
diagnostic replay through the exact evaluator probability path reproduced the
source AUC with zero error.

The invalid first output remains preserved and indexed. Retry1 changes only
the source-metric replay implementation from raw logits to the exact
`float32 sigmoid(logit)` values and assigns a new run id. It does not change
the checkpoints, validation rows, hook taps, probe, thresholds or already
frozen interpretation routes. The first execution's internal probe values are
not used as evidence until retry1 passes the source-replay gate.

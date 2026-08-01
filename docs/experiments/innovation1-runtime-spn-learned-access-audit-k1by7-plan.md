# Innovation 1 Runtime SPN Learned-Access Audit K1-BY7

**Date:** 2026-08-01
**Status:** completed / audit pass / method remains hold
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

## Retry1 completed result

Retry1 passed every frozen source, checkpoint, hook, split, probe and artifact
check. Correct source AUCs replayed exactly; affine replay error was zero on
seed2 and `4.77e-7` on seed3, below the frozen `1e-6` tolerance. No source
file changed, no neural training occurred and optimizer steps remained zero.

The fixed odd-row probe AUCs and correct-minus-affine margins were:

| Tap | seed2 correct / affine | seed2 margin | seed3 correct / affine | seed3 margin |
|---|---:|---:|---:|---:|
| Linear histogram | `0.708202 / 0.684700` | `+0.023502` | `0.692360 / 0.697807` | `-0.005447` |
| Permutation expert | `0.692879 / 0.684353` | `+0.008526` | `0.717205 / 0.723637` | `-0.006432` |
| Cell fusion | `0.682827 / 0.678555` | `+0.004272` | `0.727463 / 0.718857` | `+0.008606` |
| Pooled stage summary | `0.703434 / 0.681049` | `+0.022385` | `0.727249 / 0.705395` | `+0.021854` |
| Pre-classifier representation | `0.701580 / 0.656769` | `+0.044811` | `0.671665 / 0.682938` | `-0.011272` |

Source model AUC margins remained:

```text
seed2 correct - affine = +0.039343357
seed3 correct - affine = -0.026506424
```

The audit decision is:

```text
status        = pass
method_status = hold
decision      = innovation1_runtime_spn_k1by7_first_loss_linear_histogram_identified
remote_scale  = no
```

Seed3 first falls below the `+0.005` correct-structure margin at the earliest
linear histogram tap. The behavior is not monotonic: cell fusion and stage
pooling temporarily restore a positive margin, but the pre-classifier
representation reverses again. Seed2 first misses the strict margin at cell
fusion by only `0.000728`, then recovers strongly before the classifier.

## Interpretation

The current failure cannot be blamed only on invariant cell pooling. On seed3,
the affine wrong inverse-P transformation already exposes slightly stronger
closed-form label association than the correct transformation before the
learned permutation expert. Learned fusion can recover correct-structure
association transiently, but the final residual representation does not retain
it. This supports a representation/access diagnosis, not a formal model
ceiling.

The internal coordinates come from separately trained correct and affine
checkpoints. Therefore their tap-by-tap differences are diagnostic but not a
clean causal estimate of runtime structure: weight adaptation and runtime
choice are still coupled. Immediately redesigning the histogram would risk
repairing a between-checkpoint artifact.

## Recommended next action

Before changing the representation, preregister K1-BY8 as a zero-training
same-checkpoint runtime swap on the same validation caches:

```text
correct learned parameters + correct runtime buffers
correct learned parameters + affine runtime buffers
affine learned parameters  + correct runtime buffers
affine learned parameters  + affine runtime buffers
```

Only runtime buffers change within each weight row. Recompute source AUC and
the same five internal probes. If both correct-weight seeds prefer the correct
runtime, the main issue is independent-training variance and future attribution
must use same-checkpoint controls. If seed3 still prefers the affine runtime at
the linear histogram, change exactly the state-to-histogram representation to
retain relative source-bundle incidence. In either route, do not add samples,
pairs, epochs, width, seeds, ciphers or remote execution.

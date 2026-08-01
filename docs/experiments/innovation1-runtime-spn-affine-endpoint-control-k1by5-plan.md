# Innovation 1 Runtime SPN Affine Endpoint Control K1-BY5

**Date:** 2026-08-01
**Status:** preregistered / pending local deterministic audit
**Execution:** local CPU, zero neural training, inherited K1-BY3 validation caches

## Research question

K1-BY4 established that complete target-cell movement and a uniform source-role
permutation can change ordered PRESENT features while remaining exactly equal
after cell order is removed. K1-BY5 changes one thing:

> Does a global one-to-one source-endpoint permutation that splits every
> four-bit source-cell bundle produce an identifiable negative P-layer control
> at every deterministic K1-BY4 tap?

This is the final preregistered control-construction audit before either a new
neural attribution row or a representation-contract hold. It does not train a
model, generate data, scan controls, increase rounds or claim attack/SOTA
performance.

## Frozen authority and baseline

K1-BY5 binds the completed K1-BY4 config, preflight, results, validation and
gate by SHA-256. K1-BY4 must remain:

```text
status   = hold
decision = innovation1_runtime_spn_k1by4_permutation_expert_hold
```

K1-BY4 in turn binds the original K1-BY3 PRESENT-80 r7 validation caches. This
audit inherits those exact read-only arrays:

| Seed | Validation rows | Pairs/sample | Input bits/sample |
|---:|---:|---:|---:|
| 2 | `2048` total (`1024/class`) | 16 | 2048 |
| 3 | `2048` total (`1024/class`) | 16 | 2048 |

No source digest may change before versus after K1-BY5 execution.

## Single variable

For every compiled edge, flatten its source endpoint as:

```text
u = 4 * source_cell + source_role
```

and replace it with:

```text
u' = (5 * u + 1) mod 64
source_cell' = u' // 4
source_role' = u' mod 4
```

`gcd(5,64)=1`, so this is a bijection over all 64 source bits. It preserves:

- one incoming edge for every target bit;
- one use of every source bit;
- exactly 32 `linear_permutation` calls and zero `linear_gf2` calls;
- the correct S-boxes, stage order, parameter-independent program geometry and
  absence of cipher/absolute-cell identity from the learned model.

Unlike K1-BY4's uniform role permutation, the mapped source cell depends on the
original source role. Every original four-bit cell must map into at least two
different destination source cells. This prevents the corruption from being a
complete-cell relabeling by construction.

## Frozen taps and metrics

For seed2 and seed3, execute the same two inverse stages and collect the same
two integer-count taps:

```text
inverse_linear
post_inverse_sbox
```

The only result condition is the affine endpoint control against the correct
program. The matrix is exactly:

```text
2 seeds x 2 stages x 2 taps = 8 rows
```

Metrics are unchanged from K1-BY4:

- `multiset_equal_rate`: exact cell-histogram multiset equality after removing
  cell order;
- `pooled_summary_l1`: normalized difference in concatenated cell mean/max;
- `ordered_histogram_l1`: supporting position-preserved difference.

## Preregistered gate

The affine control passes only if every seed/stage/tap satisfies both:

```text
multiset_equal_rate <= 0.95
pooled_summary_l1   >= 0.0001
```

No average, later stage or stronger seed may rescue a failed tap.

## Decisions

- **Pass:** freeze the affine control and preregister K1-BY6 at the identical
  PRESENT r7 `2048/class`, 16-pair, seed2/3, 10-epoch protocol. Train only one
  new affine-wrong-control row per seed and compare with the frozen K1-BY3
  correct/no-conditioner anchors. Do not retrain historical anchors.
- **Hold:** stop searching endpoint permutations. Audit whether exact source
  cell identity must be represented before invariant aggregation, while
  retaining width-independent parameter shapes; no optimizer run or remote
  scale is allowed yet.
- **Invalid:** repair only the failed source, bijection, geometry, cache,
  histogram or artifact invariant and rerun unchanged.

Blocked: random permutation search, result-dependent remapping, neural
training, remote execution, scale/pair/seed/epoch changes, difference scanning,
absolute cell IDs as a shortcut, and additional ciphers.

## Planned artifacts

```text
outputs/local_audit/
i1_runtime_spn_affine_endpoint_control_k1by5_present_r7_seed2_seed3_20260801/
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

After completion, record measured metrics, claim boundary and the exact next
route here, then refresh both recent-result indexes before reporting.

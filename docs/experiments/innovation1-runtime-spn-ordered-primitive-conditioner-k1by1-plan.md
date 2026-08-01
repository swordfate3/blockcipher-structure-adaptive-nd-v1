# Innovation 1 Runtime SPN Ordered Primitive Conditioner K1-BY1

**Date:** 2026-08-01
**Status:** completed / pass / fresh-seed confirmation required
**Execution:** local sub-medium diagnostic; use local CUDA when available, otherwise a documented CPU fallback

## Research question

K1-BY0 proved that seven runtime SPN descriptors can be compiled into the same
ordered primitive contract and replayed exactly. It did not perform neural
training or measure differential AUC. K1-BY1 asks the next narrow question:

> When the compiled S-box tables, GF(2) edges and stage order are routed through
> shared learnable primitive experts, does the resulting conditioner improve a
> differential neural distinguisher, and does the improvement depend on the
> correct stage order and target binding?

This is a local `2048/class` diagnostic, not formal training, paper-scale
evidence, unseen-cipher transfer, an attack-round frontier or a SOTA claim.

## Method under test

The fixed runtime descriptor is compiled before model construction:

```text
uKNIT runtime descriptor (rounds 3 and 4)
  -> K1-BY0 ordered compiler
  -> ordered S-box / inverse-linear primitive calls
  -> shared learnable S-box, permutation and GF(2) experts
  -> ordered stage recurrent aggregation
  -> bounded residual fused with a width-independent raw-pair backbone
  -> binary differential logit
```

The primitive experts share parameters across cells and stages. Cipher name,
cipher ID and absolute cell IDs are not model inputs. State width changes only
the number of expert calls, not expert parameter shapes.

## Single variable and controls

All four conditions instantiate the same backbone, classifier, primitive
experts, recurrent aggregator and parameter geometry. Only compiler routing is
changed:

| Condition | Meaning |
|---|---|
| `correct_compiler_routing` | exact K1-BY0 program |
| `wrong_order_routing` | rotate the two distinct uKNIT stages before inverse execution |
| `wrong_target_binding_routing` | deterministically move inverse-linear messages to wrong target cells |
| `no_compiler_conditioner` | retain the identical modules but zero the compiler residual |

The historical K1-BS uKNIT structure expert is an external strong reference,
not a fifth training row. Its cross-key AUCs were `0.902801513` and
`0.932538986`; the three generic K1-BS networks were approximately random.

## Frozen protocol

Train eight independent rows: two seeds times four routing conditions.

| Field | Frozen value |
|---|---|
| Cipher / rounds | uKNIT-BC r5 |
| Difference | cell11 role1, `0x0000400000000000` |
| Train | `2048/class`, `4096` total rows |
| Cross-key validation | `1024/class`, `2048` total rows |
| Seeds | `3`, `4` |
| Pairs/sample | `16` independent ciphertext pairs |
| Input width | `2048` bits/sample, `128` bits/pair |
| Negative definition | encrypted random plaintexts |
| Train/validation keys | frozen K1-Q/K1-BS seed-matched fixed keys |
| Runtime window | round start `3`, two transitions |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | restored best validation AUC |

The same seed must reuse identical disk-backed train and validation datasets
across all four conditions. Difference, keys, labels, negatives, pair count,
sample count, epochs, optimizer and metric are immutable.

## Readiness gate

Before any optimizer step require:

1. the K1-BY0 source config, pass gate and validation digests match exactly;
2. exactly eight frozen tasks exist and all non-model fields are equal per seed;
3. all four models accept `[B, 2048]`, return finite `[B, 1]` logits and have
   finite nonzero gradients;
4. all four conditions have exactly equal trainable parameter counts;
5. correct, wrong-order and wrong-binding programs have the expected semantic
   relationship, and no model consumes cipher identity;
6. local device choice is recorded before training.

Failure prohibits optimization. Repair only the failed invariant and rerun the
same readiness gate.

## Result gate

For each seed separately require:

```text
correct AUC                         >= 0.550
correct - no conditioner            >= +0.010
correct - wrong order               >= +0.005
correct - wrong target binding      >= +0.005
```

Both seeds must satisfy every clause. Seed averaging cannot hide a failed
control. Any wrong-routing control matching or exceeding the correct program
forces `hold`.

## Decisions

- **All clauses pass on both seeds:** keep the compiled primitive conditioner
  and run one fresh-seed local confirmation before any cross-cipher transfer.
- **Signal passes but a routing margin fails:** hold structure attribution;
  inspect only the failed routing mechanism, without increasing width, epochs,
  samples or cipher count.
- **Correct condition is below `0.550`:** discard this conditioner interface
  at the current diagnostic scale; compare its deterministic feature surface
  with the K1-BS structure expert before redesigning.
- **Protocol invalid:** repair the exact plan, source binding, cache, model
  geometry or checkpoint defect and rerun unchanged.

Blocked inside K1-BY1: difference-position scans, pair changes, new ciphers,
round changes, tuning per condition, capacity mismatch, remote execution,
`65536/class` scale-up and publication claims.

## Planned artifacts

```text
run_id = i1_runtime_spn_ordered_primitive_conditioner_k1by1_16pair_2048_seed3_seed4_20260801
```

The completed result root must contain preflight, shared disk caches, progress
JSONL, eight best checkpoints, results JSONL, comparison CSV, gate, validation,
summary, history CSV, Chinese SVG, plot report and rendered-pixel visual QA
evidence. A completed result must refresh both recent-result indexes and this
document must record the measured metrics and the next executable action.

## Completed result

The frozen eight-row local diagnostic completed on 2026-08-01. Evidence root:

```text
outputs/local_diagnostic/
  i1_runtime_spn_ordered_primitive_conditioner_k1by1_16pair_2048_seed3_seed4_20260801/
```

All readiness and result-protocol checks passed. Every condition contained
exactly `235780` trainable parameters, accepted the same 16-pair input, trained
for 10 epochs, restored the best validation-AUC checkpoint and reused the same
seed-matched disk-backed train/validation data.

| Condition | seed3 AUC | seed4 AUC |
|---|---:|---:|
| Correct compiler routing | `0.979398727` | `0.982597828` |
| Wrong stage order | `0.506119251` | `0.505432606` |
| Wrong target binding | `0.500165939` | `0.503971577` |
| No compiler conditioner | `0.498686790` | `0.507903576` |

The correct route's ordinary threshold-0.5 accuracies were `0.925292969` and
`0.939453125`; its best threshold-calibrated accuracies were `0.941894531` on
both seeds. The preregistered margins were:

```text
seed3 correct - no conditioner       = +0.480711937
seed3 correct - wrong order          = +0.473279476
seed3 correct - wrong target binding = +0.479232788

seed4 correct - no conditioner       = +0.474694252
seed4 correct - wrong order          = +0.477165222
seed4 correct - wrong target binding = +0.478626251
```

Both correct-route AUCs exceeded `0.550`; every no-conditioner margin exceeded
`+0.010`, and every wrong-routing margin exceeded `+0.005`. No seed averaging
was needed to rescue a failed clause.

```text
gate status  = pass
decision     = innovation1_runtime_spn_k1by1_compiler_conditioner_supported
remote_scale = no
```

The supported conclusion is narrow but positive: on the frozen uKNIT r5
cross-key diagnostic, a runtime descriptor can be compiled into ordered
primitive calls and connected to one shared neural parameter geometry; the
resulting signal depends strongly on the exact stage order and inverse-linear
target binding. The raw width-independent backbone alone remained at chance.

This result does not yet isolate how much of the gain comes from deterministic
last-two-transition inverse execution versus learned primitive descriptor
encoding. It also exercises only uKNIT's general GF(2) linear expert in this
window: all 32 linear cell calls were `linear_gf2`, so the permutation expert
has not been validated by differential AUC. It is not unseen-cipher transfer,
formal scale, an attack-round frontier or a SOTA result.

The Chinese SVG was rendered to `2400 x 1320` pixels and passed the required
`visual-qa-redraw` inspection. Titles, subtitles, legends, axis labels, rotated
category labels, bar annotations and the final decision line had no overlap,
clipping, missing glyphs or ambiguous scale.

## Executable next action

Run K1-BY2 as a fresh-seed local confirmation before adding another cipher:

```text
research question = does the K1-BY1 routing separation reproduce on untouched keys?
same-budget anchor = K1-BY1 correct compiler routing
one variable       = fresh seed and fixed train/validation key pair
cipher / rounds    = uKNIT-BC r5
difference         = frozen cell11 role1, 0x0000400000000000
train / validation = 2048/class / 1024/class
pairs              = 16 independent ciphertext pairs
seeds              = 5,6
epochs / batch     = 10 / 64
conditions         = correct routing, wrong-order routing, no conditioner
execution          = local CUDA if available, otherwise documented local CPU
```

Require correct AUC `>=0.550`, correct-minus-no-conditioner `>=+0.010` and
correct-minus-wrong-order `>=+0.005` on both fresh seeds. If it passes, advance
to a same-contract PRESENT/GIFT permutation-expert diagnostic followed by a
held-out-cipher transfer experiment. If it fails, hold the route and audit the
seed/key dependence without increasing samples or epochs. Do not launch remote
scale, add model width, change the difference or claim general SPN support from
K1-BY1 alone.

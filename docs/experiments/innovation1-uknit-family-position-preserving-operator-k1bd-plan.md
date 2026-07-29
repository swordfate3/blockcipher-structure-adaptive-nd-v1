# Innovation 1 K1-BD Gradient And Coupling Audit

**Status:** completed / hold / topology-specific coupling redesign required
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_position_preserving_operator_k1bd_gradient_coupling_audit_replica0_replica1_20260729`

## 1. Research Question

K1-BB showed that the actual GF(2) edge set is representationally visible.
K1-BC then trained only that shared encoder, but both replicas lost cross-key
macro AUC relative to K1-AZ and correct topology beat neither wrong-topology
control in any of the twelve fresh panels.

K1-BD asks the mechanism question that must be answered before another model
change:

> Does the ordinary classification loss deliver topology-specific gradients to
> the connected position-preserving path, or are correct and wrong operators
> effectively optimization-equivalent? Separately, do uKNIT, Midori and Dialga
> produce stable shared-gradient conflict or magnitude imbalance?

This is a zero-update audit of evidence already paid for. It does not train a
new network and cannot authorize scale by absolute AUC.

## 2. Frozen Authority

Bind and rehash K1-BC's config, gate, validation, two-row training result,
48-row control result, summary, checkpoint manifest and both selected encoder
checkpoints. Reuse K1-BC's authority loader so that all eighteen K1-AZ datasets,
strict encrypted-random-plaintext negatives, keys, differences, four-pair
features, runtime structures and source checkpoints are rebound exactly.

Require the source facts:

```text
status                         = hold
protocol failures              = 0
training rows                  = 2
control rows                   = 48
topology-control passes        = 0/12 and 0/12
candidate macro delta replica0 = -0.002906005
candidate macro delta replica1 = -0.000461578
```

## 3. Data And Batch Contract

Reuse each replica's complete `train_seen` cache:

| Cipher | Replica 0 seed | Replica 1 seed | Rows | Pairs |
|---|---:|---:|---:|---:|
| uKNIT-BC r5 | 3 | 4 | 2048/class | 4 |
| Midori64 r4 | 6 | 7 | 2048/class | 4 |
| Dialga-128 r4 | 0 | 1 | 2048/class | 4 |

Deterministically shuffle positive and negative indices separately. Every
batch contains 32 unused positive and 32 unused negative rows. Sixty-four
matching batch indices across the three ciphers consume every training row
once without overlap or class-ratio variation.

No data, labels, negatives, keys, differences, pairs or metrics change.

## 4. Frozen Encoder States And Operators

For each replica audit two immutable encoder states:

- `initial_encoder`: reconstruct K1-BC's exact seeded initialization and verify
  its tensor hash against the K1-BC checkpoint payload.
- `selected_encoder`: strictly restore K1-BC's selected encoder and verify both
  file and tensor hashes.

For every batch and encoder state, use one runtime structure and three operator
conditions:

```text
correct_operator
same_summary_corrupted_operator
cross_cipher_operator
```

Only the operator consumed by the new modulation path changes. Encryption,
K1-AZ runtime structure, labels and frozen anchor remain correct.

## 5. Gradient Groups And Metrics

Use K1-BC's unchanged loss:

```text
MSE(sigmoid(logit), binary label)
```

Call automatic differentiation only to read gradients. Construct no optimizer,
perform no update and require model-state hashes and `.grad` buffers to remain
unchanged.

Measure these fixed parameter groups:

```text
connected_all
bit_encoder
token_encoder
edge_message
bit_update
bit_update_norm
pair_projection
structure_projection
```

`connected_all` excludes only `structure_projection`, because source inspection
shows that K1-BC's `sample_modulation` forward never calls that K1-BB readiness
projection. The audit must prove this from gradients rather than assume it.

Record per batch and condition:

- group gradient norm and nonzero parameter coverage;
- within-cipher cosine and relative norm difference between correct and each
  wrong operator;
- cross-cipher cosine under the correct operator;
- loss, output hash and zero-step/state invariants.

On the twelve fresh replica/cipher/split panels, additionally record RMS and
maximum changes in modulation, logits and probabilities for correct versus
disabled and correct versus both wrong operators. These are attribution
diagnostics, not alternate success metrics.

## 6. Frozen Decision Gates

For one wrong-operator condition to be optimization-indistinguishable from the
correct operator in a replica/cipher/state panel, require all three:

```text
median connected-gradient cosine          >= 0.99
frequency of batch cosine >= 0.99          >= 0.90
median relative connected-norm difference <= 0.05
```

The standard classification objective is judged topology-indistinguishable
only if both wrong conditions meet this gate for all three ciphers in both
replicas at the selected encoder state. Initial-state rows explain whether the
indistinguishability was present before training or emerged during training.

A cipher pair has systematic shared conflict in one selected-state replica
when:

```text
median correct-operator cosine <= -0.05
negative-cosine frequency      >= 0.50
```

The shared-conflict route opens only if the same pair passes in both replicas.
A stable magnitude-imbalance route opens only if both selected-state replicas
have maximum/minimum median connected-gradient norm `>=4.0` with the same
dominant cipher.

The disconnected-path finding is supported only when
`structure_projection` has exactly zero gradient in every audited row, while
all connected groups receive finite gradients and the complete probe state is
immutable.

## 7. Decisions

- **Topology-gradient indistinguishability:** ordinary task loss does not teach
  the runtime operator semantics. Next preregister one same-data
  correct-versus-wrong topology ranking auxiliary loss while keeping the
  classifier, datasets and strict negative definition fixed. Freeze or remove
  the disconnected readiness-only projection. Do not add experts.
- **Stable cross-cipher conflict without topology indistinguishability:** test
  one minimal shared-gradient combination rule against unchanged K1-BC. PCGrad
  is eligible only for a pair that passes in both replicas.
- **Stable magnitude imbalance only:** test one fixed normalization rule, not
  PCGrad, MoE or per-cipher modules.
- **Neither:** redesign the modulation coupling so correct versus wrong
  operators produce a stronger downstream intervention before training again.
- **Protocol invalid:** repair only the failed source, batch, hash, row-count or
  state invariant and replay the unchanged audit.

If multiple mechanisms pass, topology-gradient indistinguishability has
priority because K1-BC's defining claim requires causal use of the supplied
operator; optimizer balancing cannot rescue a descriptor the objective treats
as interchangeable.

## 8. Required Artifacts

Write under `outputs/local_audit/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
checkpoint_manifest.json
gradient_norms.jsonl
topology_gradient_pairs.jsonl
cross_cipher_gradient_pairs.jsonl
interventions.jsonl
results.jsonl
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

The Chinese figure must separate topology-gradient similarity, cross-cipher
conflict, parameter-group coverage and downstream intervention. It must state
the zero-training `2048/class/cipher`, four-pair claim scope and pass rendered
pixel inspection through `visual-qa-redraw`.

## 9. Prohibited Actions

Do not run 16 pairs, larger data, more epochs, wider models, extra seeds or a
remote GPU. Do not change labels, negatives, keys, differences, validation,
checkpoint selection or AUC. Do not add cipher IDs, per-cipher heads, adapters,
routers, MoE, experts, loss weighting or PCGrad inside this audit.

## 10. Completed Result

The zero-update audit completed with every source digest, dataset, initial
state, selected checkpoint and runtime operator rebound exactly. All nine
protocol checks passed:

```text
gradient norm rows             = 18432 / 18432
topology-gradient rows         = 1536 / 1536
cross-cipher-gradient rows     = 768 / 768
fresh intervention rows        = 24 / 24
optimizer steps                = 0
state mutation                 = none
persistent .grad buffers       = none
```

### 10.1 Correct And Wrong Topology Gradients

At the selected encoder, the frozen indistinguishability gate passed in six of
twelve replica/cipher/control panels, not all twelve:

| Replica | Cipher | Same-summary cosine / high-frequency / norm delta | Cross-cipher cosine / high-frequency / norm delta |
|---:|---|---|---|
| 0 | uKNIT | `0.984766 / 0.266 / 0.023643` | `0.977735 / 0.203 / 0.030608` |
| 0 | Midori | `0.970510 / 0.047 / 0.042403` | `0.969931 / 0.094 / 0.043117` |
| 0 | Dialga | `0.995945 / 0.766 / 0.006581` | `0.995190 / 0.781 / 0.007982` |
| 1 | uKNIT | `0.999697 / 1.000 / 0.002105` | `0.999600 / 1.000 / 0.002937` |
| 1 | Midori | `0.998960 / 1.000 / 0.003835` | `0.999240 / 0.984 / 0.003392` |
| 1 | Dialga | `0.999650 / 1.000 / 0.003734` | `0.999680 / 0.984 / 0.004473` |

Replica0 therefore still receives operator-dependent task gradients. Replica1
has largely collapsed to operator-independent gradients, but the effect is not
stable enough across both replicas to blame the ordinary classification
objective alone.

### 10.2 Shared-Optimizer Hypotheses Rejected

Selected-state correct-topology gradient cosines were:

| Cipher pair | Replica 0 cosine / negative frequency | Replica 1 cosine / negative frequency |
|---|---|---|
| uKNIT / Midori | `-0.032467 / 0.547` | `-0.027419 / 0.562` |
| uKNIT / Dialga | `+0.003888 / 0.500` | `-0.291224 / 0.781` |
| Midori / Dialga | `+0.018029 / 0.469` | `-0.032545 / 0.516` |

Only uKNIT/Dialga replica1 passed both conflict clauses. No cipher pair passed
in both replicas, so K1-BD does not support PCGrad.

Connected-gradient median norms and max/min ratios were:

| Replica | uKNIT | Midori | Dialga | Max/min |
|---:|---:|---:|---:|---:|
| 0 | `0.0119323` | `0.00591617` | `0.00845388` | `2.0169x` |
| 1 | `0.00358471` | `0.00217942` | `0.00284104` | `1.6448x` |

Both ratios are below the frozen `4.0x` gate. Stable gradient magnitude
imbalance is therefore also rejected.

### 10.3 The Actual Bottleneck

`structure_projection` received exactly zero gradient in all `2304` audited
state/replica/cipher/condition/batch combinations. Its `12672` parameters are
`30.84%` of K1-BC's declared `41088` trainable parameters, but are used only by
the K1-BB readiness embedding and never by K1-BC's classification forward.
Every connected parameter group received finite nonzero gradients.

The connected gradients were strongly concentrated downstream. Across the six
selected-state correct-topology cipher/replica summaries, the median norms
were approximately:

```text
token_encoder       = 3.2e-5
edge_message        = 2.0e-4
pair_projection     = 4.4e-3
```

The fresh intervention audit makes the failure more direct. Median probability
RMS change from enabling the whole new path increased from `0.015426` at the
random initial encoder to `0.035487` at the selected encoder. Yet topology-
specific probability effects shrank:

| Wrong topology | Initial effect / whole-path effect | Selected effect / whole-path effect |
|---|---:|---:|
| Same-summary corrupted | `0.2856%` | `0.0634%` |
| Cross-cipher | `0.3120%` | `0.0669%` |

Thus K1-BC learned a stronger generic sample-conditioned edge modulation while
making the actual supplied operator less relevant. Increasing the global
`0.05` modulation coefficient would amplify the shortcut as well and is not a
supported repair.

## 11. Verdict

```text
status   = hold
decision = innovation1_uknit_family_k1bd_modulation_coupling_redesign_required
```

This result rejects stable cross-cipher conflict and stable norm imbalance as
the primary K1-BC failure mechanism. It supports a topology-specific coupling
bottleneck: edge tokens are present, but the concatenation-based message MLP
and much larger downstream sample projection can learn useful modulation while
largely bypassing which edges were supplied.

The Chinese `curves.svg` was rendered at `2700 x 1800` pixels and passed
`visual-qa-redraw`. The final image has no text overlap, clipping, missing
glyphs, incomplete legend, unreadable label or misleading axis range.

## 12. Executable Next Action: K1-BE Readiness

Preregister a zero-training local readiness gate before another ten-epoch
comparison.

```text
question:
  Can a shared, width-independent edge-message block make the actual edge token
  a mandatory multiplicative controller rather than an optional concatenated
  feature?

same-budget anchor:
  K1-BB representation probes plus K1-BC/K1-BD initial checkpoints, exact
  datasets and correct/wrong operator controls.

one variable:
  replace edge_message([source_state, target_state, token_hidden]) with
  sample_message(source_state, target_state)
    * bounded_token_gate(token_hidden),
  with no sample-only bypass inside the new branch.

parameter discipline:
  one shared block for uKNIT, Midori and Dialga;
  no cipher ID, per-cipher parameter or learned position lookup;
  freeze/remove the disconnected structure_projection from the trainable path;
  keep fixed hidden and output widths across 64- and 128-bit states.

readiness scale:
  local CPU, optimizer steps 0, replicas 0/1, deterministic K1-BD batches and
  all twelve fresh panels; no data generation.

required controls:
  correct operator;
  same-summary corrupted operator;
  compatible cross-cipher operator;
  token gate disabled with exact K1-AZ replay;
  jointly relabeled state/operator with transported positions.

advance gates:
  disabled path exactly replays K1-AZ;
  joint relabel modulation/logit error <= 1e-5;
  all trainable parameters participate in the enabled classification graph;
  correct/wrong modulation and logit differences are finite and nonzero in all
  twelve panels;
  median correct/wrong probability RMS divided by whole-path probability RMS is
  at least 4x K1-BD's matched initial-encoder anchor for each wrong control,
  without changing the classifier or benchmark.

stop gates:
  any disconnected trainable group, sample-only bypass, cipher identity,
  relabeling failure, disabled replay failure, nonfinite value, or failure to
  improve topology-specific intervention over the matched K1-BD anchor.
```

Only a passing K1-BE readiness may open one local K1-BF training comparison at
the unchanged `2048/class/cipher`, four pairs, replicas `0/1`, ten epochs and
frozen K1-AZ anchor. Do not run 16 pairs, larger data, remote GPU, PCGrad,
normalization, MoE, experts or per-cipher modules.

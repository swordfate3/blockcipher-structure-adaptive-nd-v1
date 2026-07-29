# Innovation 1 K1-BE Mandatory Token-Gate Readiness

**Status:** completed / hold
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_mandatory_token_gate_k1be_readiness_20260729`

## 1. Research Question

K1-BD showed that K1-BC strengthened the overall sample-conditioned modulation
while reducing the probability effect of replacing the actual operator. The
topology token and edge-message gradients were tens to hundreds of times below
the final pair projection, and `30.84%` of declared trainable parameters were
not connected to the classification graph.

K1-BE tests one architecture change before any new training:

> Can an actual edge token become a mandatory multiplicative controller of each
> sample edge message, while preserving width-independent shared parameters,
> exact K1-AZ replay and cell-relabeling equivariance?

## 2. One Architecture Variable

Keep K1-BB's actual 18-dimensional edge tokens, bit-state update order, target
aggregation and final pair pooling. Replace only the edge-message formula:

```text
K1-BC:
  message = MLP([source_state, target_state, token_hidden])

K1-BE:
  sample_message = MLP([source_state, target_state])
  message        = sample_message * tanh(token_hidden)
```

There is no sample-only residual inside the new edge branch. If the token path
collapses to zero, the new message collapses rather than silently reverting to
a generic sample message. The K1-BB readiness-only `structure_projection` is
removed, reducing declared trainable parameters from `41088` to `26368`.

No cipher name, ID, lookup table, per-cipher parameter, adapter, router, expert
or MoE is introduced. The same parameter shapes serve 64- and 128-bit states.

## 3. Frozen Authority And Budget

Bind and rehash the completed K1-BD gate, validation, results, interventions,
checkpoint manifest and summary. Reuse K1-BD's authority chain to recover the
same K1-BC config, all eighteen disk-backed datasets, K1-AZ checkpoints,
runtime structures, gate summaries and two wrong-operator controls.

For both initialization seeds `40/41`, evaluate:

```text
uKNIT-BC r5
Midori64 r4
Dialga-128 r4
same-key fresh and cross-key validation
64 fixed rows per panel
4 ciphertext pairs per sample
optimizer steps = 0
```

The matched anchor is a reconstructed K1-BC initial encoder on the exact same
rows, checkpoint, structure and initialization seed.

## 4. Required Controls And Metrics

For every one of twelve panels evaluate:

```text
correct_operator
same_summary_corrupted_operator
cross_cipher_operator
disabled_k1az
joint_relabel
```

Record candidate and matched-anchor RMS/max changes in modulation, logits and
probabilities for correct versus disabled and correct versus both wrong
operators. Define topology share as:

```text
wrong_topology_probability_rms / correct_vs_disabled_probability_rms
```

Jointly relabel the sample state and runtime/operator structure, transport the
native cell positions, and compare modulation and logits with the original
panel.

On one deterministic balanced 64-row training probe for each replica/cipher,
differentiate the unchanged MSE task loss. Require every declared candidate
parameter to appear in the autograd graph; construct no optimizer and leave all
`.grad` buffers empty.

## 5. Frozen Readiness Gates

Protocol requirements:

- source and config digests exact;
- twelve panel rows and six gradient rows complete;
- one fixed `26368`-parameter geometry across all ciphers and widths;
- no cipher identity or per-cipher parameters;
- zero updates, immutable states and finite metrics.

Compatibility requirements:

```text
disabled K1-AZ logit replay delta = 0
joint relabel modulation delta   <= 1e-5
joint relabel logit delta        <= 1e-5
all trainable tensors connected to task loss
```

Topology intervention requirements:

```text
candidate whole-path probability RMS
  >= 0.5 * matched K1-BC initial-anchor whole-path RMS in every panel

median candidate topology share
  >= 4.0 * median matched-anchor topology share
```

The four-times clause is applied independently to same-summary corrupted and
cross-cipher controls. Every panel must also have finite, strictly nonzero
modulation, logit and probability differences for both wrong controls.

## 6. Decisions

- **All gates pass:** retain the non-bypassable multiplicative coupling and
  preregister K1-BF as one local same-budget training comparison against K1-BC.
- **Graph/replay/relabel failure:** repair only the failed implementation
  invariant and replay readiness unchanged.
- **Whole path too weak:** reject the multiplicative block; do not inflate the
  global `0.05` coefficient to rescue it.
- **Topology share fails:** reject this coupling and test a deterministic
  token-conditioned edge basis, not another generic concatenation MLP.

No branch authorizes training, 16 pairs, more data, epochs, seeds or width,
remote GPU, PCGrad, normalization, MoE, experts or per-cipher modules until
readiness passes.

## 7. Required Artifacts

Write under `outputs/local_readiness/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
geometry.json
panel_results.jsonl
gradient_coverage.jsonl
results.jsonl
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

The Chinese figure must separate whole-path retention, topology-share lift,
gradient connectivity and relabel/replay compatibility, state the zero-training
claim scope, and pass a `2700 x 1800` rendered-pixel `visual-qa-redraw` check.

## 8. Completed Result

The frozen readiness ran locally without constructing an optimizer or changing
any model state. All required artifacts are under:

```text
outputs/local_readiness/
  i1_uknit_family_mandatory_token_gate_k1be_readiness_20260729/
```

Protocol and compatibility evidence passed:

```text
panel rows                         = 12/12
gradient rows                      = 6/6
fixed trainable parameters         = 26368
graph-connected tensors            = 22/22 in every probe
disabled K1-AZ replay max delta    = 0
joint relabel modulation max delta <= 1e-5
joint relabel logit max delta      <= 1e-5
whole-path retention panels        = 12/12
minimum whole-path retention ratio = 0.729592
optimizer steps                    = 0
```

The topology-share gate failed both required controls:

```text
same-summary corrupted operator:
  K1-BC initial median share = 0.325211%
  K1-BE median share         = 0.312599%
  K1-BE / K1-BC              = 0.961221x
  required                   = 4.0x

cross-cipher operator:
  K1-BC initial median share = 0.354246%
  K1-BE median share         = 0.376164%
  K1-BE / K1-BC              = 1.061873x
  required                   = 4.0x
```

Final gate:

```text
status   = hold
decision = innovation1_uknit_family_k1be_topology_share_lift_not_supported
```

The mandatory multiplicative path fixed K1-BC's disconnected-parameter defect
and retained sufficient overall effect, but it did not make the classifier
materially more sensitive to which runtime operator was supplied. K1-BF
training is therefore not authorized. Increasing the global modulation scale
would amplify the same topology-independent path and is not a supported repair.

The Chinese SVG was rendered at `2700 x 1800` and passed the
`visual-qa-redraw` pixel inspection with no overlap, clipping, missing glyphs,
ambiguous axes or incomplete legends.

## 9. Executable Next Action

Preregister K1-BG as a zero-training deterministic token-conditioned edge-basis
readiness test.

- **Question:** can a fixed, runtime-token-derived edge basis preserve the same
  non-bypassable message geometry while making wrong-operator interventions
  materially visible?
- **Same-budget anchors:** reconstruct K1-BE and K1-BC initial encoders on the
  exact same twelve panels and initialization seeds `40/41`.
- **One variable:** replace K1-BE's learned `token_encoder -> tanh` gate with a
  deterministic shared basis derived from the existing 18-dimensional real
  edge token. Keep source/target sample-message generation, update order,
  pooling, K1-AZ and `0.05` unchanged.
- **Budget:** uKNIT r5, Midori r4, Dialga r4; four pairs; 64 fixed rows per
  same-key/cross-key panel; replicas `0/1`; zero epochs and zero optimizer steps;
  local CPU readiness only.
- **Required controls:** same-summary corrupted operator, cross-cipher operator,
  disabled K1-AZ, joint cell relabel and all-parameter graph connectivity.
- **Advance gate:** exact replay/relabel/geometry checks pass, every trainable
  tensor is connected, whole-path effect retains at least half of matched K1-BE,
  and each wrong-operator topology-share median reaches the original four-times
  K1-BC threshold while also exceeding matched K1-BE.
- **Stop gate:** any protocol/compatibility failure is repaired unchanged; weak
  whole-path or failed topology-share attribution rejects the fixed basis and
  does not authorize training.

Do not run K1-BF, expand to 16 pairs, increase data/epochs/seeds/width, use a
remote GPU, introduce MoE/per-cipher modules, or change the benchmark while
K1-BG readiness remains unproven.

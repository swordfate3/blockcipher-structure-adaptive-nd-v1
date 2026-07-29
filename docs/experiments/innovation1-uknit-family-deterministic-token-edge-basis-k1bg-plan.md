# Innovation 1 K1-BG Deterministic Token Edge-Basis Readiness

**Status:** completed / hold
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_deterministic_token_edge_basis_k1bg_readiness_20260729`

## 1. Research Question

K1-BE removed the sample-only edge-message bypass, connected all 26368
trainable parameters to the task loss and retained at least 72.96% of the
matched K1-BC whole-path effect. It nevertheless changed median topology share
by only `0.961x` for the same-summary control and `1.062x` for the cross-cipher
control, far below the frozen `4x` requirement.

K1-BG tests one representation change before any training:

> Can a fixed, full-rank edge basis derived only from the actual runtime token
> prevent learned token compression from washing out operator identity, while
> preserving K1-BE's non-bypassable sample-message geometry?

## 2. One Architecture Variable

Keep K1-BE's sample message, edge multiplication, target aggregation, bit
update, pair projection and K1-AZ integration unchanged. Replace only:

```text
K1-BE:
  token_gate = tanh(LayerNorm(ReLU(Linear18x32(token))))

K1-BG:
  P          = first 18 rows of normalized 32x32 Sylvester-Hadamard matrix
  projected  = token @ P
  normalized = projected / RMS(projected)
  edge_basis = tanh(normalized)
```

The fixed projection satisfies `P @ P.T = I18`, so it is full rank and retains
all distances in the existing 18-dimensional token before the bounded
normalization. It is a registered buffer, not a learned parameter. No random
hash, cipher name, ID, lookup table, per-cipher module, adapter, router, expert
or MoE enters the model.

Removing K1-BE's learned token encoder reduces the expected trainable parameter
count from `26368` to `25696`. The same parameter geometry serves 64- and
128-bit states.

## 3. Frozen Authority And Budget

Bind and rehash K1-BE's config, gate, validation, panel results, gradient
coverage, summary and geometry. Reuse its K1-BD authority chain to recover the
same K1-BC config, eighteen disk-backed datasets, K1-AZ checkpoints, runtime
structures, gate summaries and both wrong-operator controls.

For initialization seeds `40/41`, evaluate:

```text
uKNIT-BC r5
Midori64 r4
Dialga-128 r4
same-key fresh and cross-key validation
64 fixed rows per panel
4 ciphertext pairs per sample
optimizer steps = 0
```

Reconstruct matched K1-BG, K1-BE and K1-BC initial encoders on every panel.

## 4. Required Controls And Metrics

For each of twelve panels evaluate:

```text
correct_operator
same_summary_corrupted_operator
cross_cipher_operator
disabled_k1az
joint_relabel
```

For K1-BG, K1-BE and K1-BC, record whole-path probability RMS and correct versus
wrong-operator modulation, logit and probability RMS. Define topology share as
wrong-operator probability RMS divided by whole-path probability RMS.

Also record the fixed projection Gram error, rank and buffer digest. On one
balanced 64-row training probe per replica/cipher, differentiate the unchanged
MSE task loss and require every declared candidate parameter tensor to enter
the autograd graph. Construct no optimizer and leave `.grad` buffers empty.

## 5. Frozen Readiness Gates

Protocol and compatibility:

- source/config digests exact;
- twelve panel rows, six gradient rows and six geometry rows complete;
- fixed `25696`-parameter geometry across all ciphers and widths;
- fixed basis rank `18`, Gram max error at most `1e-6`, no token encoder;
- zero updates, immutable states and finite metrics;
- disabled K1-AZ replay delta exactly zero;
- joint-relabel modulation and logit deltas at most `1e-5`;
- every trainable tensor connected to task loss.

Research gates:

```text
candidate whole-path probability RMS
  >= 0.5 * matched K1-BE whole-path RMS in every panel

median candidate topology share
  >= 4.0 * median matched K1-BC topology share

median candidate topology share
  > median matched K1-BE topology share
```

The last two clauses apply independently to both wrong-operator controls. Every
panel must have finite, strictly nonzero modulation, logit and probability
effects for both controls.

## 6. Decisions

- **All gates pass:** retain the deterministic basis and preregister one local
  `2048/class/cipher`, four-pair, two-replica, ten-epoch training comparison
  against K1-BC and K1-BE.
- **Graph/replay/relabel/basis failure:** repair only that implementation
  invariant and replay readiness unchanged.
- **Whole path too weak:** reject the fixed basis; do not increase `0.05`.
- **Topology share fails:** reject this basis and stop the learned edge-message
  family before considering a different structure-consumption primitive.

No branch authorizes training, 16 pairs, more data, epochs, seeds or width,
remote GPU, PCGrad, normalization, scale inflation, MoE, experts or per-cipher
modules until readiness passes.

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

The Chinese figure must distinguish fixed-basis validity, whole-path retention,
topology-share attribution and compatibility, state the zero-training scope,
and pass a `2700 x 1800` rendered-pixel `visual-qa-redraw` inspection.

## 8. Completed Result

The frozen local readiness completed with zero optimizer steps. Required
artifacts are under:

```text
outputs/local_readiness/
  i1_uknit_family_deterministic_token_edge_basis_k1bg_readiness_20260729/
```

All protocol and compatibility checks passed:

```text
panel rows                          = 12/12
gradient rows                       = 6/6
fixed trainable parameters          = 25696
graph-connected tensors             = 18/18 in every probe
fixed basis rank                    = 18
fixed basis Gram max error          = 5.960464e-8
matched K1-BE/K1-BC replay max      = 0
disabled K1-AZ replay max           = 0
joint relabel modulation max delta  = 1.907349e-6
joint relabel logit max delta       = 1.668930e-6
minimum whole-path / K1-BE ratio    = 1.073147
optimizer steps                     = 0
```

The deterministic basis failed both topology attribution controls and was
worse than both matched anchors:

```text
same-summary corrupted operator:
  K1-BC median topology share = 0.325211%
  K1-BE median topology share = 0.312599%
  K1-BG median topology share = 0.184972%
  K1-BG / K1-BC              = 0.568777x
  K1-BG / K1-BE              = 0.591724x
  required vs K1-BC          = 4.0x

cross-cipher operator:
  K1-BC median topology share = 0.354246%
  K1-BE median topology share = 0.376164%
  K1-BG median topology share = 0.203638%
  K1-BG / K1-BC              = 0.574849x
  K1-BG / K1-BE              = 0.541354x
  required vs K1-BC          = 4.0x
```

Final gate:

```text
status   = hold
decision = innovation1_uknit_family_k1bg_deterministic_basis_topology_lift_not_supported
```

This result rules out a narrow explanation for K1-BE: the topology loss is not
caused only by a learned `18 -> 32` token encoder compressing edge identity.
K1-BG preserved a stronger overall path than K1-BE on every panel, yet made the
wrong-operator share smaller. The learned edge-message followed by target and
pair pooling is therefore stopped as the active structure-consumption family.
No K1-BG training or scale-up is authorized.

The Chinese SVG passed `visual-qa-redraw` after a `2700 x 1800` rendered-pixel
inspection with no overlap, clipping, missing glyphs, ambiguous axes,
insufficient separation or incomplete legends.

## 9. Executable Next Action

Preregister K1-BH as a zero-neural-training exact GF(2) operator-response audit.

- **Question:** before another neural redesign, determine whether applying the
  supplied runtime matrices exactly modulo two to the existing sample Boolean
  views makes the correct operator distinguishable from both frozen wrong
  operators on label-relevant fresh data.
- **Same-budget anchors:** correct operator, same-summary corrupted operator,
  cross-cipher operator, identity/no-transport view and label-shuffled control
  on the exact K1-BG datasets and panels.
- **One variable:** replace learned edge-message transport with direct stagewise
  GF(2) matrix application to the existing twelve per-bit Boolean channels.
  Preserve stage, native cell and bit-role coordinates until after transport.
- **Budget:** uKNIT r5, Midori r4, Dialga r4; four pairs; replicas `0/1`; same
  train-seen, same-key fresh and cross-key caches; no neural optimizer, no data
  generation and local CPU only.
- **Audit probe:** use one frozen deterministic ridge/logistic probe recipe for
  every cipher/operator condition. Fit only on each existing train-seen cache
  and evaluate on untouched fresh splits; report AUC plus correct-minus-control
  margins. This is mechanism evidence, not a shared-network result.
- **Advance gate:** correct exact transport must beat identity, both wrong
  operators and label shuffle on every replica/cipher/fresh split, with a
  preregistered nontrivial margin. Passing opens a direct exact-transport
  residual readiness design; it does not open remote scale.
- **Stop gate:** if wrong operators tie or beat the correct operator, the current
  data/control surface cannot identify runtime topology through direct GF(2)
  action. Stop architecture changes and audit the benchmark/difference surface
  instead.

Do not return to learned token gates, concatenate another generic edge MLP,
increase `0.05`, train K1-BG, expand pairs/data/epochs/seeds/width, use remote
GPU, add MoE/per-cipher neural modules, or change labels/negatives before K1-BH
resolves this exact operator-response mismatch.

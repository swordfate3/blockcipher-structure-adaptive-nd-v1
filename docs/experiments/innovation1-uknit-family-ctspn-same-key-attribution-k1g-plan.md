# Innovation 1 uKNIT-Family CT-SPN Same-Key Attribution K1-G

**Date:** 2026-07-28
**Run ID:** `i1_uknit_family_ctspn_cell_path_hypergraph_same_key_attribution_k1g_20260728`
**Status:** completed / pass / sample-specific hypergraph attribution overfit confirmed
**Execution:** local CPU, zero training

## 1. Question

K1-F cannot pass its frozen family gate because uKNIT seed0 selected validation
AUC is `0.498477`, below the `0.520` floor. Its epoch histories also show rising
training AUC while uKNIT validation stays near chance. However, the current
train-versus-validation language mixes two changes:

```text
training cache   = training plaintext rows + fixed train key 0
validation cache = different plaintext rows + fixed validation key 0x11...11
```

K1-G asks the missing attribution question:

> Does the frozen K1-F hypergraph generalize to fresh plaintext rows under the
> same fixed training key, or is its apparent training signal tied to the exact
> cached rows? If same-key generalization survives, does only the key change break
> the signal?

This audit distinguishes sample memorization, key-specific signal and shared-cell
relation underuse. It does not train or select another checkpoint.

## 2. Frozen Evidence And One New Dataset Split

Reuse all four K1-F best-validation-AUC checkpoints and all eight source caches:

| Split | Rows | Fixed key | RNG seed | Origin |
|---|---:|---|---:|---|
| `train_seen` | 4096 | original train key | `seed` | exact K1-F training cache |
| `same_key_fresh` | 2048 | original train key | `seed + 20000` | newly generated holdout |
| `cross_key_validation` | 2048 | original validation key | `seed + 10000` | exact K1-F validation cache |

The new split preserves cipher, rounds, difference `0x40`, four independent
ciphertext pairs per sample, balanced labels and strict encrypted-random-plaintext
negatives. It changes only the plaintext RNG rows while retaining the training key.
The audit must prove zero exact feature-plus-label row overlap between
`train_seen` and `same_key_fresh` for every cipher and seed.

Panel:

```text
uKNIT-BC prefix-r5, seeds 0 and 1
Dialga-128 prefix-r4, seeds 0 and 1
```

Scale remains a local `2048/class` training-source and `1024/class` holdout
mechanism diagnostic. It is not formal training, an attack, SOTA evidence, transfer
evidence or a uKNIT ceiling.

## 3. Frozen Same-Checkpoint Controls

Each of the four checkpoints is evaluated on all three splits under:

```text
correct_ordered
repeat_last
rotated
corrupted
no_topology
incidence_shuffled
```

The `incidence_shuffled` control retains the exact path-token multiset and changes
only which paths share source, middle and target cells. Every split/condition must
strict-load the same state dictionary, use the same dataset within its panel and
perform zero optimizer steps.

Artifact contract:

```text
4 checkpoints x 3 splits x 6 conditions = 72 inference rows
new training rows                        = 0
optimizer steps                          = 0
```

The original cross-key validation panel must replay all 24 K1-F control AUC values
within `5e-6` and reproduce the exact dataset hashes.

## 4. Attribution Gate

For a cipher/seed/split, correct relation attribution passes only when:

```text
correct_ordered - every one of five controls >= +0.005 AUC
```

For uKNIT fresh/cross-key usefulness also require:

```text
correct_ordered AUC >= 0.520
```

No average may hide a failed uKNIT seed or control. Dialga is a positive/negative
mechanism calibration and cannot override uKNIT.

## 5. Decisions

- **Training and same-key fresh pass; cross-key fails:** confirm a key-specific
  hypergraph signal. Retain shared-cell routing and next test one difference-only
  input bottleneck that removes absolute ciphertext/key cues at the same budget.
- **Training passes; same-key fresh fails:** confirm exact-row/sample overfit. Close
  this learned two-transition hypergraph and next test one exact operator-tied
  latent propagation model at the same budget.
- **Same-key relation margins pass but uKNIT AUC remains below `0.520`:** retain
  only the relation-attribution mechanism evidence. Do not call it sample
  memorization or useful distinction; close the high-capacity predictor and test
  one constrained exact operator-tied model.
- **Training attribution fails:** confirm that the selected checkpoints do not
  reliably use the shared-cell relation even on source rows. Close learned
  incidence conditioning and move to exact operator-tied propagation.
- **All three pass but K1-F held:** audit only the remaining K1-F anchor/control
  failure before changing architecture.
- **Seed-mixed outcome:** stop scale-up and localize the failed seed/control; do not
  promote a mean result.

## 6. Blocked Routes

- No new training, remote launch, larger data, more pairs, epochs, width or seeds.
- No K2 S-box semantics, MoE, DDT, trail, partial decryption or cipher identity.
- No benchmark relabeling, negative-definition change or checkpoint reselection.
- Do not describe this audit as formal scale, an attack, transfer or a uKNIT limit.

## 7. Required Outputs

```text
outputs/local_audit/i1_uknit_family_ctspn_cell_path_hypergraph_same_key_attribution_k1g_20260728/
  preflight.json
  dataset_manifest.jsonl
  progress.jsonl
  results.jsonl
  attribution.csv
  validation.json
  gate.json
  summary.json
  curves.svg
  plot_report.json
  visual_qa_passed.marker
  cache/
```

After completion, render the SVG to pixels and pass `visual-qa-redraw`, update this
record with metrics and the evidence-backed next action, refresh both recent-result
indexes, run focused tests and commit/push the scoped implementation.

## 8. Completed Result

K1-G completed locally on 2026-07-28. It strict-loaded the four K1-F best
checkpoints, reused all eight source caches, created exactly four fresh-same-key
caches and evaluated all seventy-two planned inference rows without training.

```text
status   = pass
decision = innovation1_uknit_family_ctspn_k1g_sample_specific_hypergraph_attribution_overfit_confirmed
result rows       = 72 / 72
training rows     = 0
optimizer steps   = 0
failed protocol checks = []
maximum K1-F cross-key replay AUC delta = 0.0
fresh-same-key exact feature+label overlap with train = 0 rows
```

The uKNIT result was:

| Seed | Split | Correct AUC | Weakest correct-minus-control margin | Correct-minus-incidence-shuffled |
|---:|---|---:|---:|---:|
| 0 | original train rows / train key | `0.797085` | `+0.045632` | `+0.045632` |
| 0 | fresh rows / same train key | `0.493531` | `-0.012529` | `-0.002191` |
| 0 | original validation / validation key | `0.498477` | `-0.001863` | `+0.000207` |
| 1 | original train rows / train key | `0.701712` | `+0.027907` | `+0.027907` |
| 1 | fresh rows / same train key | `0.497610` | `-0.027804` | `+0.000995` |
| 1 | original validation / validation key | `0.521642` | `+0.002453` | `+0.002453` |

Both uKNIT checkpoints strongly attributed the correct shared-cell relation on
their original training rows. The preference disappeared on fresh plaintexts
under the exact same training key: absolute AUC returned to chance and at least
one wrong relation beat the correct relation on both seeds. The key therefore
is not the main missing variable. The learned two-transition hypergraph is tied
to the cached training rows or their specific ciphertext combinations.

Dialga retained high absolute AUC on all three splits (`0.958663` to `0.979754`),
which confirms that the frozen replay preserves its strong task signal. However,
its incidence-shuffle margin stayed at only about `0.0005` to `0.0047`, so even
Dialga does not establish the learned shared-cell relation as the cause of the
high AUC. It remains a calibration and cannot override the uKNIT decision.

## 9. Claim Scope And Next Action

This result confirms sample-specific relation attribution overfit only for the
frozen K1-F checkpoints at local `2048/class` training-source and `1024/class`
holdout diagnostic scale. It is not evidence that uKNIT is indistinguishable at
formal scale, nor an attack, SOTA, arbitrary-SPN transfer or family ceiling claim.

Close the learned two-transition cell/path hypergraph. The next experiment is
K1-H, a same-budget exact operator-tied latent propagation model. It must keep
the ciphertext-pair dataset, keys, strict negatives, rounds, seeds, optimizer,
epochs and checkpoint rule frozen. The one model variable is to replace learned
anonymous path mixing/shared-incidence messages with deterministic runtime
linear-operator transport of cell latents plus a small shared residual update.
The exact operator and an operator-shuffled control must use the same parameter
geometry; no absolute cell or cipher identity may enter the network.

K1-H should remain a local `2048/class`, seeds `0,1`, ten-epoch diagnostic with
uKNIT-BC r5 and Dialga-128 r4. Compare only the exact-operator candidate against
the strongest same-protocol anchor and the necessary operator-shuffled and
no-topology controls. Advance only if both uKNIT seeds reach `AUC >= 0.520`, beat
the anchor by `>= +0.005`, and beat both controls by `>= +0.005`; otherwise close
the exact parameterization. Do not add samples, remote scale, width, epochs,
MoE, K2 S-box semantics, DDT, trail, partial decryption or cipher ID before this
local gate.

Artifacts:

```text
outputs/local_audit/
  i1_uknit_family_ctspn_cell_path_hypergraph_same_key_attribution_k1g_20260728/
```

The final Chinese `curves.svg` was rendered at `1600 x 1020`. The first pixel
inspection found an unreadably compressed Dialga AUC panel; the redraw switched
that panel to an explicitly labelled local y-axis and added annotation
backgrounds. The final artifact passed `visual-qa-redraw` with no unintended
overlap, clipping, missing glyph, ambiguous title or unreadable close curves.

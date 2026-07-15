# Innovation 1 Balanced-Feistel Round-Relation Attribution Plan

**Date:** 2026-07-16

**Status:** r12/r15 seed0 diagnostic complete; easier-round calibration frozen

## Research Question

Under identical raw ciphertext-pair input, data, labels, keys, epochs, and
optimizer budget, does a cipher-correct Feistel previous-round relation improve
SIMON64/128 r12 and SIMECK64/128 r15 fresh-test AUC over both a fixed
branch-shuffled relation control and the strongest current generic baseline?

## Hypothesis

The AND-RX Feistel equations provide key-canceling previous-round difference
channels. A shared pair encoder using the correct cipher-specific relation
should learn the strong-signal paper rounds more reliably than the same network
with left/right roles swapped. If the generic model performs equally well but
the shuffled model drops, the relation is still attributed; if shuffled does
not drop, no structure claim is allowed.

## Frozen Protocol

| Field | SIMON | SIMECK |
|---|---|---|
| cipher | SIMON64/128 | SIMECK64/128 |
| target round | 12 | 15 |
| external Lu accuracy | `0.7117` | `0.7663` |
| plaintext difference | `(0x00000000,0x00000040)` | same |
| pairs/sample | 8 | 8 |
| key sampling | one independent random 128-bit key/sample | same |
| positive | eight fixed-difference pairs | same |
| negative | eight independent plaintext pairs encrypted under row key | same |
| raw input | 1024-bit ciphertext-pair bits | same |

The external accuracies use paper-scale SE-ResNet training and are not project
targets. Project evaluation uses fresh-test AUC.

## Lean Three-Model Matrix

For each cipher:

| Role | Model | Only semantic difference |
|---|---|---|
| candidate | cipher-correct round relation | correct left/right roles and `f` |
| attribution control | fixed branch-shuffled relation | swap left/right before the same `f` |
| same-input anchor | `multiscale_dense_resnet` | generic processing of raw 1024 bits |

Candidate and shuffled control must have identical trainable parameter counts.
All three receive the same raw features; the benchmark and labels do not change.

## Budgets

Readiness:

```text
samples_per_class      = 64
validation total       = 128
fresh test total       = 128 x 1 repeat
seed                   = 0
epochs                 = 2
batch                  = 64
purpose                = mechanics only; no research conclusion
```

Local diagnostic:

```text
samples_per_class      = 2048 (4096 total train rows/model)
validation total       = 4096 (2048/class)
fresh test total       = 8192 (4096/class) x 3 repeats/model
seed                   = 0
epochs                 = 10
batch                  = 128
hidden bits            = 32
loss / optimizer       = MSE / Adam
learning rate / L2     = 1e-4 / 1e-5
checkpoint             = best val_loss
device                 = local CPU
```

This is a local architecture/representation diagnostic, not formal or
paper-scale evidence.

## Readiness Gate

- exact six plan/result rows and no alignment errors;
- correct ciphers, rounds, profiles, eight pairs, strict negatives, and
  `key_rotation_interval=1`;
- all models complete two epochs and one fresh repeat;
- candidate/control parameter counts match per cipher;
- all metrics are finite and artifacts are complete.

Passing readiness automatically authorizes the frozen local seed0 diagnostic.
Readiness AUC is not research evidence.

## Diagnostic Gate

Evaluate each cipher independently:

```text
candidate fresh AUC                         >= 0.55
candidate fresh AUC - shuffled fresh AUC   >= +0.01
candidate fresh AUC - generic fresh AUC    >= -0.005
```

Decisions:

| Result | Decision and next action |
|---|---|
| both ciphers pass all gates | retain balanced-Feistel relation cell; run the same six-row seed1 confirmation |
| exactly one cipher passes | retain a cipher-conditional relation; audit wrong-cipher relation before another seed |
| signal exists but true does not beat shuffled | reject relation attribution; retain strongest raw baseline |
| candidate AUC below `0.55` | stop scale; inspect implementation and use one easier-round calibration before redesign |

## Stopped Actions

- no SM4 8192 run in parallel with this architecture test;
- no related-key, two-difference, RX, polytopic, or staged data;
- no random-ciphertext negatives;
- no extra model rows or hyperparameter sweep after seeing test metrics;
- no remote launch, paper-scale claim, key recovery, or cross-Feistel claim.

## Planned Artifacts

```text
results.jsonl
progress.jsonl
validation.json
curves.svg
history.csv
gate.json
outputs/00_RECENT_RESULTS.md entry
```

Every completed gate must record exact metrics, claim scope, stopped routes,
and an evidence-backed executable next action.

## Completed Readiness And Seed0 Diagnostic

Readiness completed at `64/class` with all six rows plan-aligned, finite, and
candidate/control parameter counts equal at `32,225`. This authorizes only the
frozen local diagnostic; readiness AUC is not research evidence.

The `2048/class`, seed0, 10-epoch diagnostic completed at:

```text
outputs/local_diagnostic/i1_feistel_balanced_round_relation_2048_seed0/
```

The first execution was invalidated after a source/data audit found that
balanced positive row `i` and negative row `i` reused the same deterministic
rotating key. Both memory and disk generation now use global row indices,
rotating-key caches carry `key_rotation_row_indexing=global_dataset_row`, and
the readiness plus both diagnostics were regenerated from invalidated caches.

Valid fresh-test AUC over three independent `8192`-row repeats:

| Cipher | correct relation | shuffled relation | generic | true-shuffled | true-generic |
|---|---:|---:|---:|---:|---:|
| SIMON64/128 r12 | `0.503118654` | `0.500748505` | `0.503139764` | `+0.002370149` | `-0.000021110` |
| SIMECK64/128 r15 | `0.496726533` | `0.497522483` | `0.499559065` | `-0.000795950` | `-0.002832532` |

Decision:

```text
feistel_balanced_relation_not_ready
```

Neither cipher reaches the frozen `0.55` candidate signal gate and neither
reaches the `+0.01` relation-attribution margin. This is a valid small local
diagnostic, not a formal failure or evidence that paper-scale training cannot
work. Remote launch, seed1 confirmation, and mechanical sample scale-up remain
stopped.

## Frozen Easier-Round Calibration

The next question is whether the source-verified relation implementation and
small pair encoder can learn an easier known-signal cell under the identical
budget. Change only the target round:

| Field | SIMON calibration | SIMECK calibration |
|---|---:|---:|
| round | r11 | r14 |
| external Lu accuracy | `0.9181` | `0.9142` |
| samples/class | `2048` | `2048` |
| seed / epochs | seed0 / 10 | seed0 / 10 |
| models | true / shuffled / generic | true / shuffled / generic |

All data, difference, eight-pair grouping, strict negatives, key rotation,
optimizer, model capacity, validation total, and three fresh-test repeats stay
unchanged. Calibration gates per cipher are frozen before training:

```text
candidate AUC                         >= 0.60
candidate AUC - shuffled AUC          >= +0.01
candidate AUC - generic AUC           >= -0.005
```

If both ciphers pass, the formulas and pair encoder are locally calibrated;
the next implementation is a closer Lu SE-ResNet/high-round training-protocol
comparison, not immediate remote scale. If exactly one passes, audit the
non-passing cipher's round function and wrong-cipher control. If neither passes,
stop neural scale-up and perform a row-by-row author-code data/layout parity
audit before changing architecture.

## Completed Easier-Round Calibration

The corrected independent-key calibration completed with all six rows aligned:

| Cipher | correct relation | shuffled relation | generic | true-shuffled | true-generic |
|---|---:|---:|---:|---:|---:|
| SIMON64/128 r11 | `0.549644550` | `0.500767072` | `0.494218598` | `+0.048877478` | `+0.055425952` |
| SIMECK64/128 r14 | `0.574093332` | `0.499780446` | `0.499883960` | `+0.074312886` | `+0.074209372` |

Both ciphers strongly separate the correct relation from the branch-swapped
control, so the formulas are not inert. Neither reaches the preregistered
`0.60` calibration floor, therefore the decision remains:

```text
feistel_balanced_easier_round_not_calibrated
```

The source-code parity audit found data semantics and eight-field order aligned
after the key-index repair, but found a concrete architecture mismatch. The
public Lu network treats the eight pair groups as the Conv1D sequence and all
`8 x 32 = 256` derived bits per pair as channels, uses two pointwise transition
layers, five SE residual blocks, and a position-preserving flatten head. The
current candidate convolves over 32 bit positions inside each pair and then
mean/max-pools the pair set, discarding pair position.

## Frozen Author-Layout Repair

The next local experiment changes only that architecture layout at the easier
SIMON r11/SIMECK r14 calibration cells. Per cipher, compare three rows:

```text
Lu-layout correct relation
Lu-layout branch-shuffled equal-capacity control
current pair-encoder correct-relation anchor
```

Data, labels, corrected global key indexing, input difference, eight pairs,
seed0, `2048/class`, validation/fresh totals, 10 epochs, MSE/Adam, and checkpoint
selection remain fixed. Frozen gates per cipher:

```text
Lu-layout candidate AUC                    >= 0.60
candidate - Lu-layout shuffled             >= +0.01
candidate - current pair-encoder true       >= +0.02
```

If both ciphers pass, run the same three-row layout comparison at r12/r15; if
one passes, keep the layout cipher-conditional; if neither passes, stop local
architecture changes and quantify the data-scale gap before any remote request.

## Completed Author-Layout Repair

The six-row source-layout comparison completed with corrected global key
indexing and exact reuse of the r11/r14 data cache:

| Cipher | Lu-layout true | Lu-layout shuffled | pair-pool true anchor | true-shuffled | true-anchor |
|---|---:|---:|---:|---:|---:|
| SIMON r11 | `0.509407719` | `0.500735253` | `0.549644550` | `+0.008672466` | `-0.040236831` |
| SIMECK r14 | `0.516449024` | `0.503228535` | `0.574093332` | `+0.013220489` | `-0.057644308` |

The Lu-layout candidate/control parameter counts match at `63,777`, but neither
cipher reaches `0.60` or improves over the `32,225`-parameter pair-pool anchor.
Decision:

```text
feistel_lu_layout_not_calibrated
```

The closer source layout is rejected at this local budget. The current
pair-pool correct-relation model remains the strongest implemented Feistel
candidate; no r12/r15 Lu-layout run is authorized.

## Frozen 8192/Class Data-Scarcity Probe

The next and final local scale question is whether the retained pair-pool
relation signal has a positive sample-size slope at the easier rounds. Change
only `samples_per_class` from `2048` to `8192`; keep two roles per cipher:

```text
pair-pool correct relation
pair-pool branch-shuffled equal-capacity control
```

Use seed0, 10 epochs, batch 128, validation `8192/class`, and three fresh-test
repeats of `16384/class`. Freeze the 2048/class correct-relation anchors at
`0.549644550` for SIMON and `0.574093332` for SIMECK. Gates per cipher:

```text
8192 candidate AUC                         >= 0.57
8192 candidate - 8192 shuffled             >= +0.02
8192 candidate - frozen 2048 candidate      >= +0.02
```

If both pass, run an independent seed1 at the same `8192/class` before any
remote proposal. If one passes, retain a cipher-conditional data slope. If
neither passes, stop mechanical scale-up and redesign the representation or
objective; do not jump to `65536/class` or paper scale.

## Completed 8192/Class Seed0 Probe

| Cipher | 8192 true | 8192 shuffled | true-shuffled | gain over 2048 true |
|---|---:|---:|---:|---:|
| SIMON r11 | `0.658869159` | `0.500257527` | `+0.158611632` | `+0.109224609` |
| SIMECK r14 | `0.860501059` | `0.502064635` | `+0.358436424` | `+0.286407727` |

Both ciphers pass signal, attribution, and scale-gain gates. Decision:

```text
feistel_relation_scale_slope_two_cipher_pass
```

The authorized next action is the identical four-row `8192/class`, 10-epoch,
three-repeat matrix with seed1. Seed1 confirms absolute signal and true-versus-
shuffled attribution; it is not a second within-seed scale slope because no
seed1 `2048/class` anchor is being added. No remote or high-round run is
authorized until this independent confirmation is complete.

## Completed Independent Seed1 Confirmation

| Cipher | seed0 true | seed0 shuffled | seed1 true | seed1 shuffled |
|---|---:|---:|---:|---:|
| SIMON r11 | `0.658869159` | `0.500257527` | `0.689276022` | `0.506654760` |
| SIMECK r14 | `0.860501059` | `0.502064635` | `0.855688992` | `0.510210369` |

Seed1 passes the absolute-signal and true-versus-shuffled gates for both
ciphers. The two-seed synthesis is therefore:

```text
corrected independent-key low-round relation signal = confirmed at 8192/class
seed0 within-seed positive sample-size slope         = supported
paper-target r12/r15 signal at 8192/class             = not yet tested
remote or paper-scale claim                           = not authorized
```

## Frozen Paper-Target 8192/Class Probe

Restore only the target round from SIMON r11/SIMECK r14 to r12/r15. Keep the
same seed0, `8192/class`, 10 epochs, data, cache semantics, model, and shuffled
control. Freeze the corrected `2048/class` high-round anchors at
`0.503118654` and `0.496726533`. Gates per cipher:

```text
8192 target-round candidate AUC                    >= 0.55
candidate - shuffled                               >= +0.02
candidate - corrected 2048 target-round candidate >= +0.02
```

If both pass, run the same target-round four-row matrix with seed1. If one
passes, retain a cipher-conditional target-round route. If neither passes, hold
the high-round route and do not request remote or larger sample scale.

## Completed Paper-Target Probe And Route Verdict

| Cipher | target true | target shuffled | true-shuffled | gain over target 2048 |
|---|---:|---:|---:|---:|
| SIMON r12 | `0.505085583` | `0.500342575` | `+0.004743008` | `+0.001966929` |
| SIMECK r15 | `0.511121858` | `0.500361473` | `+0.010760385` | `+0.014395325` |

Neither cipher passes the `0.55` signal, `+0.02` attribution, or `+0.02`
target-round scale-gain gates. Decision:

```text
feistel_target_round_8192_not_ready
```

Final evidence statement for this experiment family:

```text
SIMON r11 / SIMECK r14 relation signal at 8192/class = retained and two-seed confirmed
SIMON r12 / SIMECK r15 relation signal at 8192/class = not ready
Lu pair-axis SE layout at 2048/class                  = rejected versus pair-pool
remote_scale                                          = no
paper_scale_or_SOTA_claim                              = no
```

## Recommended Next Action

Do not add `16k/32k/65k` target-round rows. The next research question is
whether the confirmed easier-round relation representation can transfer to the
target round under an equal-compute curriculum comparison.

Before training, implement a plan-aligned way to give every row the same total
number of optimizer epochs, then freeze three roles per cipher:

```text
true relation: easier-round pretrain -> target-round fine-tune
shuffled relation: easier-round pretrain -> target-round fine-tune
true relation: target-round-only equal-total-epoch control
```

Keep corrected independent per-sample keys, strict negatives, eight pairs,
`8192/class`, seed0, fresh repeats, model capacity, and total optimizer steps
fixed. Advance only if the curriculum true model reaches AUC `0.55`, beats the
curriculum shuffled control by `+0.02`, and beats the equal-compute scratch
control by `+0.01`. Until the equal-compute runner and plan exist, no training,
remote launch, or scale claim is authorized.

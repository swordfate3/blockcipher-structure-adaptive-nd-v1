# Innovation 1 uKNIT-Family CT-SPN Exact Operator-Tied Latent K1-H

**Date:** 2026-07-28
**Status:** completed / hold / operator-tied real-latent mean not supported
**Execution:** local CPU diagnostic only

## 1. Evidence And Question

K1-G separated plaintext-row effects from fixed-key effects. Both K1-F uKNIT
checkpoints strongly preferred the correct shared-cell relation on the original
training cache, but fell to chance and preferred at least one wrong relation on
fresh plaintexts under the same training key. This confirmed sample-specific
hypergraph attribution overfit rather than a primarily key-specific shortcut.

The next model must therefore remove the learned path-set capacity that can bind
to concrete ciphertext/path combinations. It must not add more metadata or a
larger mixer. K1-H asks:

> Can a low-capacity, round-shared residual processor whose message transport is
> fixed directly by each runtime GF(2) linear operator retain correct-operator
> attribution on fresh same-key plaintexts and cross-key validation data?

## 2. Route Ranking

The alternatives are ranked from the current evidence:

1. **Exact operator-tied latent transport:** selected. It removes path identity,
   fixes message routing to the supplied operator and shares one residual update
   across transitions. This directly addresses K1-G's high-capacity sample
   shortcut while preserving runtime structural adaptation.
2. **Difference-only input bottleneck:** not selected. K1-G's fresh-same-key split
   already failed, so changing only absolute ciphertext/key cues does not target
   the observed cause.
3. **Return to the U2-F delta-U query:** not selected. It passed only on prefix-r4;
   U2-H failed both seeds on the aligned prefix-r5 window.
4. **More samples, width, epochs, MoE or K2 S-box semantics:** blocked. K1-F did
   not establish shared-cell attribution on unseen rows, so scaling or adding
   capacity would not test the missing mechanism.

This route uses the established runtime exact-GF(2) representation, but it is a
new learned processor. It must not relabel earlier Runtime-E4, K1 or U2 results as
operator-tied latent evidence.

## 3. One Model Variable

Keep the K1-F ciphertext-pair input, pair handling, invariant pooling and final
classifier protocol. Replace the complete relative-path token and learned
cell/path hypergraph with one bit-latent processor:

```text
per pair (C, C')
  -> shared bit encoder of [C_bit, C'_bit, C_bit xor C'_bit]
  -> for each loaded transition in reverse order:
       exact runtime inverse-GF(2) adjacency transports source latents
       four bit latents in each runtime cell form a shared cell context
       one shared residual update consumes [state, operator message, cell context]
  -> mean/max/RMS pooling over bits and runtime cells
  -> unchanged four-pair attention and classifier family
```

For transition `r`, the non-trainable message is:

```text
m_r[target] = mean({h[source] : inverse_linear_r[target, source] = 1})
```

The binary matrix is the exact supplied runtime inverse operator. The averaging
acts on real-valued latent channels and must not be described as executing GF(2)
algebra on hidden values. "Exact" refers to the routing support, ordering and
operator ownership, not to a cryptographic hidden-state inversion.

Constraints:

- hidden width `32`;
- one residual block reused for both transitions, with tied parameters;
- total learned parameters `<= 150000`;
- no round embedding, path token, absolute cell/bit number, cipher ID or learned
  router;
- 64- and 128-bit instances have identical state-dict keys and tensor shapes;
- global bit/cell relabeling jointly applied to input and operators leaves pooled
  logits invariant within `1e-6`.

## 4. Frozen Protocol

| Field | uKNIT-BC | Dialga-128 |
|---|---:|---:|
| Prefix rounds | 5 | 4 |
| Runtime transitions | 3-4 | 2-3 |
| Training | `2048/class` = 4096 total | `2048/class` = 4096 total |
| Cross-key validation | `1024/class` = 2048 total | `1024/class` = 2048 total |
| Fresh-same-key holdout | `1024/class` = 2048 total | `1024/class` = 2048 total |
| Seeds | 0, 1 | 0, 1 |
| Pairs/sample | 4 | 4 |
| Epochs | 10 | 10 |
| Batch size | 64 | 64 |
| Loss/optimizer | MSE / Adam | MSE / Adam |
| Learning rate/weight decay | `1e-4` / `1e-5` | `1e-4` / `1e-5` |
| Checkpoint | best original cross-key validation AUC | same |
| Negative mode | encrypted random plaintexts | same |

Reuse the exact K1-F/K1-G train, validation and fresh-same-key caches. Do not
regenerate or reselect rows. Training key, validation key, input difference
`0x40`, label balance, sample order and metric remain frozen.

The strongest same-protocol anchor is the selected Runtime-E4 checkpoint from K1,
not K1-F. Strict-load and evaluate that frozen anchor on the same three splits;
do not retrain or reselect it after seeing K1-H.

## 5. Same-Checkpoint Controls

Each K1-H checkpoint is evaluated without training on all three splits under:

```text
exact_ordered       = correct two-transition runtime operators
operator_reversed   = exact operator multiset, reversed order
operator_corrupted  = deterministic non-identity source permutation per operator
no_topology         = identity transport with the same learned parameters
```

Controls change only non-trainable operator buffers. They must retain identical
state dicts, data, labels and classifier parameters. The corrupted control uses a
frozen seed and must preserve matrix dimensions and nonzero target degree.

## 6. Zero-Training Readiness Gate

Before the first optimizer step require:

1. K1-F and K1-G protocol gates pass with the exact decisions recorded above;
2. all four candidate tasks and four frozen Runtime-E4 anchors match cipher,
   rounds, seed, data, keys and training protocol;
3. all twelve K1-F/K1-G caches are reused by exact digest, with no regeneration;
4. candidate geometry is identical across 64-/128-bit ciphers and stays within
   the `150000`-parameter cap;
5. every runtime inverse matrix is binary, invertible and exactly bound to the
   corresponding candidate transport buffer;
6. a unit-latent fixture reproduces the declared source-to-target support for
   every loaded transition;
7. changing only the earlier operator changes fixed-weight logits, proving full
   two-transition consumption;
8. joint global bit/cell relabeling preserves logits within `1e-6`;
9. correct/control instances strict-load the same state dict while retaining
   distinct operator fingerprints and logits;
10. no path, cell index, bit index, cipher identity, S-box, DDT, trail or key
    value enters a learned layer;
11. readiness uses zero dataset training rows and zero optimizer steps.

Any readiness failure permits only repair of the failed invariant. It does not
authorize training, protocol changes or a fallback model.

## 7. Research Gate

For both uKNIT seeds, require on **both** fresh-same-key and cross-key holdouts:

```text
candidate AUC >= 0.520
candidate - frozen Runtime-E4 anchor >= +0.005
candidate - each of three same-checkpoint controls >= +0.005
```

For both Dialga seeds, require on both holdouts:

```text
candidate >= frozen Runtime-E4 anchor - 0.005
candidate - each same-checkpoint control >= +0.005
```

The original training split must also attribute the correct operator over every
control by `>= +0.005`, but training AUC cannot compensate for a holdout failure.
No mean may hide a failed cipher, seed, split or control.

## 8. Decisions

- **All gates pass:** retain the low-capacity operator-tied processor and only
  then plan a separate `65536/class` remote diagnostic with disk-backed caches.
- **Training passes but fresh-same-key fails:** close K1-H as another sample
  shortcut; do not add capacity or data to this parameterization.
- **Fresh-same-key passes but cross-key fails:** classify the remaining signal as
  key-specific and only then reconsider a difference-only input constraint.
- **Absolute AUC passes but controls fail:** reject the structural interpretation.
- **uKNIT passes but Dialga fails:** hold the two-cipher family claim and inspect
  width-normalized pooling; do not average across ciphers.
- **Protocol invalid:** repair and rerun unchanged.

## 9. Blocked Routes And Outputs

No remote launch, more samples, pairs, epochs, width or seeds before the complete
local gate. No MoE, K2 S-box semantics, DDT, trail, partial decryption, cipher ID,
benchmark relabeling or negative-definition change.

Required readiness/training artifacts follow the existing K1-F/K1-G contract:

```text
preflight.json
results.jsonl
controls.jsonl
split_attribution.csv
checkpoint_manifest.json
history.csv
validation.json
gate.json
summary.json
progress.jsonl
curves.svg
plot_report.json
visual_qa_passed.marker
```

Every completed result must refresh `outputs/00_RECENT_RESULTS.md` and its JSON
companion. The result record must state the next evidence-backed action even when
the research gate holds.

## 10. Completed Readiness And Execution

The zero-training readiness gate completed locally on 2026-07-28:

```text
run_id   = i1_uknit_family_ctspn_operator_tied_latent_k1h_readiness_20260728
status   = pass
decision = innovation1_uknit_family_ctspn_k1h_execution_authorized
```

All four candidate rows, four frozen Runtime-E4 anchors, twelve K1-F/K1-G data
caches and both runtime transitions were digest-bound before training. The
64-/128-bit instances had identical learned parameter geometry, the candidate
stayed below the `150000`-parameter cap, relabeling invariance passed within
`1e-6`, and every control strict-loaded the same learned state with a distinct
operator fingerprint. Readiness consumed zero training rows and optimizer
steps.

The authorized diagnostic then completed with the frozen protocol:

```text
run_id = i1_uknit_family_ctspn_operator_tied_latent_k1h_2048_seed0_seed1_20260728
training rows = 4
evaluation rows = 60
failed protocol checks = []
eight training/validation caches reused = true
training/validation cache regeneration = false
```

## 11. Results

### uKNIT-BC r5

| Seed | Split | Candidate AUC | Runtime-E4 anchor | Weakest control margin |
|---:|---|---:|---:|---:|
| 0 | original train rows | `0.519238` | `0.624762` | `-0.000270` |
| 0 | fresh plaintexts / train key | `0.502423` | `0.503880` | `-0.001659` |
| 0 | original cross-key validation | `0.509368` | `0.526651` | `-0.003747` |
| 1 | original train rows | `0.540934` | `0.556674` | `+0.002676` |
| 1 | fresh plaintexts / train key | `0.500109` | `0.514069` | `-0.005531` |
| 1 | original cross-key validation | `0.528292` | `0.528809` | `+0.002344` |

Neither seed passes the frozen `+0.005` margin against every same-checkpoint
operator control. Both fresh-same-key panels remain at chance, and neither seed
beats the frozen Runtime-E4 anchor on both holdouts. Seed1's cross-key AUC is a
weak residual signal, but it is not correct-operator attribution.

### Dialga-128 r4 calibration

| Seed | Split | Candidate AUC | Runtime-E4 anchor | Weakest control margin |
|---:|---|---:|---:|---:|
| 0 | original train rows | `0.566904` | `0.971672` | `+0.002893` |
| 0 | fresh plaintexts / train key | `0.546438` | `0.969792` | `-0.001080` |
| 0 | original cross-key validation | `0.531590` | `0.961386` | `-0.002370` |
| 1 | original train rows | `0.548439` | `0.970026` | `+0.002285` |
| 1 | fresh plaintexts / train key | `0.517300` | `0.962512` | `-0.003134` |
| 1 | original cross-key validation | `0.527280` | `0.960392` | `+0.000610` |

The candidate collapses Dialga's known same-protocol signal from approximately
`0.96-0.97` to `0.52-0.57`. Reversed or corrupted operators are often equal or
better. This calibration failure shows that K1-H's transport bottleneck loses
useful information even where the frozen benchmark has abundant signal; it does
not support a uKNIT-specific ceiling claim.

## 12. Decision And Claim Scope

```text
status   = hold
decision = innovation1_uknit_family_ctspn_k1h_operator_tied_latent_not_supported
```

K1-H removes much of K1-F's sample-specific path capacity, but its transport is
only exact in operator support. It computes a real-valued mean over source
latents rather than the GF(2) XOR/cancellation operation represented by the
runtime matrix. The Dialga collapse makes information loss in this aggregation
the strongest next mechanism to test.

This is a two-seed local `2048/class` training and `1024/class` holdout mechanism
diagnostic. It is not formal scale, an attack, SOTA evidence, arbitrary-SPN
transfer evidence, or proof that uKNIT has reached a ceiling.

## 13. Evidence-Backed Next Action: K1-I

The next question is whether deterministic Boolean features computed by the
exact runtime GF(2) operators can preserve both Dialga's calibrated signal and
uKNIT correct-operator attribution without reintroducing learned path identity.

Keep the same K1-H data caches, two ciphers, rounds, seeds, four pairs, ten
epochs, MSE/Adam settings, strict negatives, fixed keys, split definitions and
Runtime-E4 anchors. Change exactly one mechanism:

```text
K1-H: exact operator support -> real-valued latent source mean
K1-I: exact operator matrix  -> deterministic GF(2) XOR Boolean views
```

For each ciphertext pair, form `[C, C', C xor C']`, apply each loaded inverse
operator and its exact two-transition composition over GF(2), and only then pass
the resulting Boolean channels to a shared bit encoder and the unchanged
invariant pair-pooling/classifier protocol. Retain operator-reversed,
operator-corrupted and no-topology same-checkpoint controls. Do not add inverse
S-box, DDT, trail, key, cipher ID, raw bypass, MoE, extra width, epochs or data.

Run K1-I locally at `2048/class` first. Advance to a separate disk-cached remote
`65536/class` diagnostic only if both uKNIT seeds pass the AUC floor, anchor and
all-control margins on both holdouts while both Dialga seeds retain their frozen
anchors. A failed local gate closes this Boolean-view parameterization and
requires auditing nonlinear S-box/operator composition; it does not authorize a
mechanical scale-up.

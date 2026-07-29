# Innovation 1 K1-AO Multi-Cipher Shared Training

**Status:** completed / hold / redesign before scale
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_multicipher_shared_weight_k1ao_2048_replica0_replica1_20260729`

## 1. Research Question

K1-AO readiness proved that one K1-AK backbone can strictly share its full
state across uKNIT-BC, Midori64 and Dialga-128 and switch runtime descriptors
without mutation. The remaining question is empirical:

> Does joint optimization force one shared representation to prefer the
> correct runtime S-box semantics, while retaining the independently established
> signal of each cipher?

This tests one variable relative to K1-AK/K1-T/K1-N: independently optimized
per-cipher weights become one shared state per replica. The data surfaces,
differences, pairs, negative definition and per-cipher row budgets remain
unchanged.

## 2. Frozen Replicas

| Replica | Init seed | uKNIT-BC r5 | Midori64 r4 | Dialga-128 r4 |
|---:|---:|---:|---:|---:|
| 0 | 30 | seed3 | seed6 | seed0 |
| 1 | 31 | seed4 | seed7 | seed1 |

Each replica owns one K1-AK backbone, one classifier, one Adam optimizer and one
best checkpoint. It must not own cipher-specific parameters, heads, adapters,
experts, normalization states or optimizers.

## 3. Data Contract

Use the 18 cache payloads bound and rehashed by K1-AO readiness:

| Cipher | Difference | Train | Same-key fresh | Cross-key fresh | Input |
|---|---|---:|---:|---:|---|
| uKNIT-BC r5 | cell11 role1, `0x0000400000000000` | 2048/class | 1024/class | 1024/class | 4 pairs, 512 bits |
| Midori64 r4 | cell8 role1, `0x0000000400000000` | 2048/class | 1024/class | 1024/class | 4 pairs, 512 bits |
| Dialga-128 r4 | `0x40` | 2048/class | 1024/class | 1024/class | 4 pairs, 1024 bits |

All negatives are encrypted random plaintexts. Each cipher uses a fixed key in
its training and same-key-fresh cache and a distinct fixed key in its cross-key
cache. No key, label, difference, cipher ID or dataset seed enters the network.

## 4. Shared Optimization Protocol

```text
epochs                         = 10
batch size                     = 64
batches/cipher/epoch           = 64
optimizer steps/epoch          = 3 * 64 = 192
optimizer steps/replica        = 1920
loss                           = MSE(sigmoid(logit), label)
optimizer                      = Adam, lr 1e-4, weight decay 1e-5
checkpoint metric              = min cross-key AUC over the three ciphers
tie-break                      = mean cross-key AUC
```

Within each epoch, independently shuffle each cipher's 4096 training rows, then
alternate one uKNIT batch, one Midori batch and one Dialga batch. Every row is
used once per epoch. The minimum-AUC checkpoint rule prevents Dialga's strong
signal from selecting a checkpoint that abandons uKNIT or Midori.

## 5. Same-Checkpoint Evaluation

Restore the selected checkpoint and perform no optimizer step. For both fresh
splits, evaluate:

```text
3 ciphers x 2 replicas x 2 fresh splits x {
  correct runtime,
  wrong S-box same checkpoint,
  transition branch off same checkpoint
}
= 36 rows
```

The wrong-S-box runtime changes only S-box truth tables. Branch-off uses the
correct descriptor but bypasses the learned transition branch. Model tensor
hashes must remain identical across all three conditions.

## 6. Same-Budget Anchors

The independent anchors are read from frozen control artifacts:

| Cipher / seed | Same-key fresh AUC | Cross-key fresh AUC | Source |
|---|---:|---:|---|
| uKNIT seed3 | 0.735209 | 0.713162 | K1-T exact position histogram |
| uKNIT seed4 | 0.738803 | 0.748229 | K1-T exact position histogram |
| Midori seed6 | 0.668338 | 0.656132 | K1-AK correct structure |
| Midori seed7 | 0.663027 | 0.653863 | K1-AK correct structure |
| Dialga seed0 | 0.967677 | 0.959750 | K1-N exact composition |
| Dialga seed1 | 0.959320 | 0.954737 | K1-N exact composition |

These anchors use different retained architectures because no prior single
architecture has passed all three ciphers. They are the strongest available
same-data, same-row-budget independent baselines; K1-AO must not obtain a
semantic margin by simply destroying useful signal.

## 7. Frozen Gate

Every one of the 12 cipher/replica/fresh-split panels must satisfy:

```text
correct AUC - independent anchor AUC >= -0.010
correct AUC - wrong-S-box AUC       >= +0.005
correct AUC - branch-off AUC        >= +0.005
```

The result passes only if all 36 clauses pass. Macro or micro averages cannot
rescue a failed panel. Protocol validity additionally requires two complete
checkpoints, exactly `1920` optimizer steps per replica, 36 evaluation rows,
zero evaluation optimizer steps, exact cache hashes and immutable state across
same-checkpoint controls.

## 8. Decisions

- **Pass:** shared optimization retains all three signals and makes correct
  runtime semantics identifiable; next prepare a separate `65536/class/cipher`
  remote-readiness audit with disk-backed per-cipher caches.
- **Retention failure only:** shared capacity or gradient competition is the
  blocker. Inspect per-cipher gradient conflict before considering a minimal
  conflict-aware optimizer; do not add experts yet.
- **Semantic failure with retention:** shared weights still admit an
  operator-insensitive solution. Stop sample/pair scale and audit whether the
  K1-AK transition summary is identifiable across the three runtime families.
- **Both fail:** discard K1-AO in its current form and return to representation
  design; do not remotely scale.
- **Protocol failure:** repair only the exact data, step-count, checkpoint or
  evaluation binding mismatch and rerun without interpreting AUC.

## 9. Required Artifacts

Write `preflight.json`, `dataset_manifest.jsonl`, `checkpoint_manifest.json`,
`results.jsonl`, `controls.jsonl`, `history.csv`, `comparison.csv`, `gate.json`,
`validation.json`, `summary.json`, `progress.jsonl`, and a Chinese `curves.svg`
under:

```text
outputs/local_diagnostic/
  i1_uknit_family_multicipher_shared_weight_k1ao_2048_replica0_replica1_20260729/
```

The SVG must pass `visual-qa-redraw`. Refresh both recent-result indexes after
the completed result. No remote execution, 16-pair run, larger budget, MoE,
cipher ID, per-cipher head or expert is authorized by this plan.

## 10. Completed Result

The run completed with two checkpoints, exactly `1920` optimizer steps per
replica, 36 zero-step evaluation rows and all nine protocol checks passing.
The result is therefore valid local diagnostic evidence rather than a crash or
readiness-only artifact.

Cross-key AUC at each selected shared checkpoint was:

| Cipher | Replica | Correct | Independent anchor | Wrong S-box | Branch off |
|---|---:|---:|---:|---:|---:|
| uKNIT-BC r5 | 0 | 0.642729 | 0.713162 | 0.491512 | 0.509686 |
| uKNIT-BC r5 | 1 | 0.688768 | 0.748229 | 0.506380 | 0.512907 |
| Midori64 r4 | 0 | 0.599349 | 0.656132 | 0.550820 | 0.568588 |
| Midori64 r4 | 1 | 0.600397 | 0.653863 | 0.579778 | 0.609721 |
| Dialga-128 r4 | 0 | 0.967916 | 0.959750 | 0.954894 | 0.924642 |
| Dialga-128 r4 | 1 | 0.971022 | 0.954737 | 0.932915 | 0.900323 |

The full same-key and cross-key gate has three distinct outcomes:

- Correct S-box semantics beat the wrong-S-box same-checkpoint control in
  `12/12` panels. The smallest margin was `+0.013021`; all exceeded the frozen
  `+0.005` gate. Shared optimization therefore produced a real S-box
  preference that independent per-cipher training had not established.
- Signal retention passed only `4/12` panels, all four belonging to Dialga.
  uKNIT retention deltas ranged from `-0.085952` to `-0.054718`; Midori ranged
  from `-0.071068` to `-0.053466`, far below the frozen `-0.010` floor.
- Branch-off attribution passed `11/12` panels. Midori replica1 cross-key was
  the exception: correct `0.600397` versus branch-off `0.609721`, a margin of
  `-0.009324`.

The aggregate gate is therefore `hold` with decision
`innovation1_uknit_family_k1ao_shared_training_retention_and_semantics_failed`.
This does not erase the successful `12/12` S-box result; it says that the
current equal-batch shared optimizer obtains that semantic preference by
losing too much uKNIT/Midori strength and does not use the transition branch
reliably in every split.

Artifacts are under:

```text
outputs/local_diagnostic/
  i1_uknit_family_multicipher_shared_weight_k1ao_2048_replica0_replica1_20260729/
```

The Chinese four-panel `curves.svg` was rendered to `2160 x 1344` pixels and
passed the second `visual-qa-redraw` inspection after clipped labels, legend
occlusion and right-boundary spacing were fixed.

## 11. Evidence-Backed Next Action

Do not increase to 16 pairs, add samples or epochs, widen the network, launch a
remote job, or introduce MoE. Those changes would not distinguish a shared
optimization conflict from a representation failure.

Run K1-AP, a zero-update gradient-conflict audit on the two selected K1-AO
checkpoints:

```text
question       = do the three cipher losses request conflicting shared updates?
anchor         = K1-AO correct-runtime gradients at the selected checkpoints
controls       = wrong-S-box and branch-off gradients on the same rows/state
data           = the unchanged 2048/class/cipher training caches
pairs          = 4
replicas       = 0 and 1
batch schedule = 64 deterministic balanced batch triplets per replica
optimizer step = 0
measurements   = pairwise cosine, negative-cosine frequency, gradient norm ratio
```

Advance to one minimal conflict-aware shared optimizer only if both replicas
show systematic correct-runtime conflict, defined before inspection as at least
one cipher pair with median cosine `<= -0.05` and negative-cosine frequency
`>= 0.50`, or a stable median gradient-norm ratio `>= 4.0`. The next training
candidate must keep the same data, model, pairs, epochs, seeds and controls and
change only the optimizer combination rule. If K1-AP does not establish those
conditions, return to the transition representation rather than tuning the
training budget.

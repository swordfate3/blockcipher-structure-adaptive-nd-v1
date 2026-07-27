# Innovation 1 uKNIT-Family CT-SPN Relative-Path Attribution K1-E

**Date:** 2026-07-28
**Run ID:** `i1_uknit_family_ctspn_relative_path_train_validation_attribution_k1e_20260728`
**Status:** completed / pass
**Source:** completed K1-D `hold`

## 1. Question

K1-D removed absolute cell identifiers and composed adjacent transitions into
directed `source -> middle -> target` path tokens, but uKNIT remained below its
same-budget anchors and did not consistently beat corrupted or no-topology
controls. Before introducing another model, K1-E asks one attribution question:

> Did K1-D learn the correct relative path set only on its training rows and lose
> that preference on unseen validation rows, or did anonymous path-set pooling fail
> to attribute the correct topology even on the training cache?

This is an inference-only audit. It changes no model, checkpoint, dataset, label,
negative definition, metric, optimizer or threshold.

## 2. Frozen Source

```text
source run       = i1_uknit_family_ctspn_relative_path_k1d_2048_seed0_seed1_20260728
ciphers          = uKNIT-BC prefix-r5; Dialga-128 prefix-r4
train            = 2048/class = 4096 total rows per cipher/seed
validation       = 1024/class = 2048 total rows per cipher/seed
seeds            = 0, 1
pairs/sample     = 4
source epochs    = 10
negative mode    = encrypted random plaintexts
sample structure = independent pairs
input difference = 0x40
audit training   = none
optimizer steps  = 0
```

The audit must strict-load the four selected best-validation-AUC K1-D checkpoints,
reuse the exact eight disk-backed source caches, and replay every split under:

```text
correct_ordered
repeat_last
rotated
corrupted
no_topology
```

Expected output is `2 ciphers x 2 seeds x 2 splits x 5 conditions = 40` rows.

## 3. Fail-Closed Protocol Gate

Execution is valid only if:

1. the source gate is the exact protocol-clean K1-D `hold` decision;
2. the frozen plan contains exactly the four K1-D candidate rows;
3. four source result rows, twenty source control rows and four checkpoint entries
   bind to the same cipher/seed panel;
4. checkpoint paths remain inside the source run, file hashes match the manifest,
   and all controls bind to the same checkpoint per cipher/seed;
5. all eight source datasets contain `metadata.json`, `features.npy` and
   `labels.npy`, and execution reports eight cache-reuse events with no generation;
6. every condition strict-loads the same learned state, performs no training and
   records zero optimizer steps;
7. validation AUC replays the K1-D source control within `5e-6` CPU tolerance.

Any failed protocol check invalidates the audit and permits only binding repair.

## 4. Attribution Gate

For each cipher, seed and split, correct topology is attributed only if all four
wrong-control margins independently satisfy:

```text
correct AUC - control AUC >= +0.005
```

No macro average may hide a failed uKNIT seed or control.

## 5. Decisions

- **uKNIT training attribution passes but validation attribution fails:** confirm
  another split-specific shortcut. Close K1-D and build a permutation-equivariant
  cell/path hypergraph whose cell identities are routing indices, not numeric input.
- **uKNIT training attribution also fails:** confirm anonymous path-set relation
  collapse. Close K1-D and use the same hypergraph direction, with a readiness gate
  that explicitly proves shared-cell incidence survives pooling.
- **uKNIT training and validation both pass:** treat the K1-D source gate or replay
  binding as inconsistent and audit it before another model.
- **Protocol failure:** repair only the source, cache, checkpoint or replay binding.

In every valid non-success branch, the next candidate may change only path-set
pooling into relation-preserving equivariant message passing. It must retain the
same pair handling, hidden budget, data protocol and five controls.

## 6. Blocked Routes

- No remote launch or mechanical increase in samples, pairs, epochs, width or seeds.
- No absolute cell value, cipher identity, learned router or MoE.
- No K2 S-box truth table, ANF, DDT, trail, partial decryption or guessed key.
- No attack, SOTA, transfer, arbitrary-SPN or uKNIT-ceiling claim.

## 7. Required Artifacts

```text
outputs/local_audit/
  i1_uknit_family_ctspn_relative_path_train_validation_attribution_k1e_20260728/
    preflight.json
    progress.jsonl
    results.jsonl
    attribution.csv
    gate.json
    validation.json
    summary.json
    curves.svg
    plot_report.json
    visual_qa_passed.marker
```

After completion, refresh `outputs/00_RECENT_RESULTS.md` and
`outputs/00_RECENT_RESULTS.json`. The completed result section must report every
uKNIT seed/split margin, claim scope, and the exact next model decision.

## 8. Completed Result

K1-E completed locally on 2026-07-28. It reused all eight source caches, strict-loaded
the four selected K1-D checkpoints, performed no training and produced all forty
planned rows. Every protocol check passed and validation replay was exact:

```text
status   = pass
decision = innovation1_uknit_family_ctspn_k1e_split_specific_relative_path_overfit_confirmed
training rows = 0
optimizer steps = 0
maximum validation replay AUC delta = 0.0
```

uKNIT attribution was:

```text
seed0 train:
  correct AUC                    = 0.773035
  correct - repeat/rotated       = +0.261763 / +0.273560
  correct - corrupted/no-topology = +0.238903 / +0.262763

seed0 validation:
  correct AUC                    = 0.518386
  correct - repeat/rotated       = +0.007745 / +0.024916
  correct - corrupted/no-topology = +0.020204 / -0.014898

seed1 train:
  correct AUC                    = 0.644833
  correct - repeat/rotated       = +0.118379 / +0.148483
  correct - corrupted/no-topology = +0.139260 / +0.153940

seed1 validation:
  correct AUC                    = 0.515869
  correct - repeat/rotated       = +0.026663 / +0.022419
  correct - corrupted/no-topology = -0.000401 / +0.020160
```

Both uKNIT training splits pass all four attribution margins by a wide margin, but
both validation splits fail. Dialga supplies a positive calibration: training AUC
is `0.982605/0.978985` and all training attribution margins pass; validation retains
`0.958499/0.957774`, although repeated-last remains below the margin on both seeds.

The result confirms that anonymous relative paths are expressive enough to fit the
correct uKNIT topology on cached training rows, but the learned preference does not
generalize. K1-D is therefore closed. The next candidate is one same-budget,
permutation-equivariant cell/path hypergraph: cell indices may be used only as
message-routing keys, never as numeric features, and a relation-shuffle readiness
control must preserve the path-token multiset while changing shared-cell incidence.
No remote scaling, additional data, K2 nonlinear conditioning, MoE or cipher router
is authorized before that candidate passes local attribution controls.

Artifacts:

```text
outputs/local_audit/
  i1_uknit_family_ctspn_relative_path_train_validation_attribution_k1e_20260728/
```

The Chinese `curves.svg` was rendered at `1600 x 1020` and passed
`visual-qa-redraw`: no overlap, clipping, missing glyph, ambiguous title,
insufficient near-zero margin separation or misleading axis range was observed.

# Innovation 1 Feistel Implementation Gap Checklist

**Date:** 2026-07-16

## Primitives And Data

- [x] Add `simon64_lu2024_ordinary` fixed difference `(0,0x40)`.
- [x] Add `simeck64_lu2024_ordinary` fixed difference `(0,0x40)`.
- [x] Verify packed left/right orientation against cipher encryption output.
- [x] Test SIMON and SIMECK tensor round functions against integer formulas.
- [x] Test all eight derived word channels against author-source equations.
- [x] Fix and verify global row indexing so all positive and negative samples
      receive distinct per-row keys; invalidate pre-fix rotating-key caches.
- [x] Verify eight pairs share one row key and validation/fresh-test use distinct
      deterministic key streams.

## Model And Attribution

- [x] Add one balanced-Feistel round-relation model under
      `models/structure/feistel/`.
- [x] Support `simon` and `simeck` round functions without separate duplicated
      network implementations.
- [x] Support `true` and fixed `shuffled` branch mappings.
- [x] Assert candidate/control parameter counts are identical.
- [x] Register four explicit model keys in the Feistel registry.
- [x] Keep `multiscale_dense_resnet` as the same-raw-input anchor.

## Experiment And Gate

- [x] Create six-row readiness CSV: two ciphers x true/shuffled/generic.
- [x] Create six-row seed0 local diagnostic CSV at `2048/class`.
- [x] Add plan-alignment and Feistel relation gate logic.
- [x] Require strict negatives, eight pairs, per-row rotating keys, expected
      rounds, profiles, epochs, final repeats, and equal control capacity.
- [x] Generate JSONL, progress, validation, SVG, history, gate, and index entry.

## Evidence Stages

- [x] Mechanism readiness: tensor/channel/registry/CLI smoke checks.
- [x] Local scaled verification: SIMON r12 and SIMECK r15, seed0,
      `2048/class`, 10 epochs.
- [ ] Independent seed1 confirmation only after the seed0 gate passes.
- [ ] Wrong-cipher relation control only after correct-vs-shuffled attribution.
- [ ] Remote medium diagnostic only after local two-seed evidence and disk-cache
      readiness; no remote launch is authorized by the initial plan.
- [x] Easier-round calibration at SIMON r11/SIMECK r14 to separate
      implementation learnability from high-round small-data weakness.
- [x] Audit public author-code field order, negative sampling, pair grouping,
      key sampling, Conv1D axis, SE blocks, and head layout.
- [x] Compare a source-layout five-block SE-ResNet against the current
      pair-encoder and an equal-capacity branch-shuffled control.
- [x] Run one `8192/class` pair-pool true/shuffled data-scarcity probe at the
      easier rounds; require positive signal and scale slope before seed1.
- [x] Confirm the retained easier-round `8192/class` signal and attribution on
      independent seed1 for both SIMON and SIMECK.
- [x] Test the paper target rounds r12/r15 at `8192/class` and stop remote scale
      after both miss the frozen signal/attribution/scale-gain gates.
- [ ] Add an equal-total-epoch low-to-high curriculum versus scratch runner and
      plan before another target-round training experiment.

## Explicitly Not Implemented In This Cell

- Exact Lu SE-ResNet and paper-scale `2e7/2e6` training.
- Related-key, two-difference, RX, staged, or polytopic sample generation.
- Key recovery.
- SM4 8192 scale-only diagnostic.
- Cross-Feistel breakthrough or SOTA claims.

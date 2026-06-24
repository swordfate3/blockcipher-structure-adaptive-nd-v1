# SPN Candidate Evidence Route

**Date:** 2026-06-23

**Goal:** Move Innovation 1 PRESENT/SPN work from active-only or flat bit-matrix scaling toward explicit structure-adaptive evidence: candidate trail confidence, margin, disagreement, and multi-pair consistency.

## Why

The active-pattern route showed that active-nibble structure can be learned, but active-only 24-dimensional summaries did not separate real-vs-random samples. The stopped trail-position medium run also showed that directly scaling heavy `beamstats8/deep4` bit-matrix inputs is too slow before we know which evidence family has signal.

This route extracts small numeric evidence directly from PRESENT S-box-DDT beam statistics:

- per-layer confidence
- per-layer margin
- top-k score entropy
- candidate disagreement
- active count dynamics
- pair-set consistency via mean/std/span across pairs

The first target is not a publication claim. It is a fast screen to decide whether candidate-disagreement and margin evidence deserves a larger structure-adaptive neural model.

## Implemented Entry Points

- `src/blockcipher_nd/features/spn_candidate_evidence.py`
- `experiments/innovation1/run_spn_candidate_evidence_baseline.py`
- `tests/test_spn_candidate_evidence.py`

Default protocol:

- cipher: `PRESENT-80`
- rounds: configurable, target `r7`
- difference profile: `present_zhang_wang2022_mcnd`
- input difference: `0x0000000000000009`
- sample structure: `zhang_wang_case2_mcnd`
- negative mode: `encrypted_random_plaintexts`
- train key: `0x00000000000000000000`
- validation key: `0xffffffffffffffffffff`

## Screen Ladder

1. Local smoke:
   - `rounds=7`
   - `samples_per_class=8..128`
   - purpose: pipeline correctness only

2. Fast screen:
   - `rounds=7`
   - `samples_per_class=4096..8192`
   - `pairs_per_sample=16`
   - compare `linear` and `mlp`
   - purpose: check whether AUC is clearly above chance

3. Evidence screen:
   - `rounds=7`
   - `samples_per_class=65536`
   - at least seeds `0,1`
   - only if fast screen is positive

4. Medium scale:
   - `samples_per_class=262144`
   - only if candidate evidence beats active-only and global stats controls

No formal SPN/PRESENT claim should be made until `>=1000000/class`, multiple seeds, strict negatives, and structure ablations are complete.

## Required Ablations

- candidate evidence full
- remove pair-set std/span consistency
- remove margin features
- remove disagreement features
- active-only baseline
- global stats baseline
- shuffled structure control before publication-style claims

## Stop Rules

- If fast screen stays at chance across linear and MLP, inspect candidate evidence construction before scaling.
- If full candidate evidence does not beat active-only/global stats controls at `65536/class`, do not launch medium scale.
- If a route requires multi-hour cache generation before any signal is visible, profile and redesign the data representation first.

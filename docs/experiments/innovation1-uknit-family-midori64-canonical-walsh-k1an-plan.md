# Innovation 1 uKNIT-Family Midori64 Canonical Walsh K1-AN

**Date:** 2026-07-29
**Status:** completed / hold; canonical representation too restrictive
**Execution:** local CPU only; no remote scale

## 1. Research Question

K1-AL proved that a K1-AK correct checkpoint causally uses both the correct
Midori64 S-box and the S-box-transition branch. K1-AM then retained this
same-checkpoint causality but failed to beat an independently trained
wrong-S-box substitute on three of four fresh panels. The trainable K1-AK
transition path can reinterpret the `16 x 16` transition bins separately for
each runtime, so a wrong S-box can learn another bin relabeling.

K1-AN asks:

> If every runtime S-box is expressed in the same fixed low-degree Walsh basis,
> with no trainable transition encoder or projection, does the correct Midori64
> runtime retain the signal and consistently beat an independently trained
> wrong-S-box model?

## 2. Frozen Sources

K1-AK source:

```text
run_id   = i1_uknit_family_midori64_sbox_transition_k1ak_2048_seed6_seed7_20260729
status   = hold
decision = innovation1_uknit_family_midori64_k1ak_sbox_transition_discrimination_failed
```

K1-AM source:

```text
run_id   = i1_uknit_family_midori64_semantic_contrast_k1am_2048_seed6_seed7_20260729
status   = hold
decision = innovation1_uknit_family_midori64_k1am_semantic_preference_imposed_substitute_unresolved
```

Required K1-AM source digests:

```text
gate        eda28b0116560d3d1fc5f4dcdcd3859e48ff671a5c04844b42d5494b34458178
validation  988f425ab6d2c1f9ccda2b6f1d509a1cf636e7e1150d6a0e8042798eb05fab0a
results     1b67df6adb9ae265cdb1c2dadcc9abf1fbf8be965d2e30d3f250e8106bd7731c
controls    686a642e536a2ac5a1c4e09e1569c6f8d76f1c70d4c060a8a18c5663b4f4f904
```

K1-AN reuses the six K1-AK/K1-AH disk-backed datasets and their exact payload
digests. Source drift makes the run invalid.

## 3. Single Experimental Variable

Retain the K1-AK exact operator-composition base path, edge residual,
classifier, bounded transition gate and runtime descriptor. Replace only:

```text
learned transition path:
  16x16 histogram -> learned Linear(256,20)
                   -> virtual-slot learned projection(40,128)

with:

canonical transition path:
  16x16 histogram -> fixed Walsh correlations W[a,x] * p[x,y] * W[b,y]
                   -> fixed first 64 non-DC mask pairs per transition stage
                   -> mean over native cells
                   -> concatenate two stages into 128 values
                   -> parameter-free layer normalization
```

Mask pairs are sorted once by total Boolean degree and then lexicographically;
the DC pair `(0,0)` is excluded. Exactly `64` coefficients per stage are
frozen. K1-AN must not scan the coefficient count, ordering, normalization or
gate initialization.

The Walsh basis is runtime-independent and meaningful for any 4-bit S-box. It
prevents the transition branch from learning a separate arbitrary relabeling
of the 256 input/output bins for each candidate S-box.

## 4. Lean Six-Model Matrix

Train three equal-state-geometry models per seed:

| Condition | Runtime S-box | Canonical branch |
|---|---|---|
| `correct_structure` | correct Midori64 | enabled |
| `wrong_sbox` | deterministic wrong S-box | enabled |
| `transition_branch_off` | correct Midori64 | disabled in forward pass |

The complete matrix is `2 seeds x 3 conditions = 6 models`. All three
conditions within a seed must start from the same tensor state hash. The branch
off model retains the same state-dict geometry and parameter count; only the
forward-use boolean changes.

## 5. Frozen Protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | Midori64 r4 |
| Difference | cell8 role1, `0x0000000400000000` |
| Seeds | `6`, `7` |
| Pairs per sample | `4` |
| Train | `2048/class`, `4096` total rows per seed |
| Same-key fresh | `1024/class`, `2048` total rows per seed |
| Cross-key fresh | `1024/class`, `2048` total rows per seed |
| Negative definition | encrypted random plaintexts |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam `1e-4`, weight decay `1e-5` |
| Scheduler | none |
| Checkpoint | best cross-key validation AUC |
| Expected optimizer steps | `640` per model |
| Execution | local CPU |

No dataset is generated. Training and validation must use parameter-matched
reuse of the bound disk caches.

## 6. Evaluation And Protocol Gate

Strict-load each best checkpoint on train-seen, same-key fresh and cross-key
fresh data. The final panel contains `2 seeds x 3 conditions x 3 splits = 18`
rows. Require exact plan/source/data hashes; six complete ten-epoch rows; six
best checkpoints; same initialization within seed; no trainable parameter name
containing `transition_encoder`, `transition_projection` or `walsh`; exact 64
mask pairs and fixed Walsh fingerprint; `18/18` zero-training evaluation rows;
exact cross-key AUC replay; exact row counts, input geometry and strict
negative definition; finite metrics; and cache reuse without generation.

Any failed protocol check makes K1-AN invalid.

## 7. Research Gates

Apply every gate independently to seed6/7 and both fresh splits:

```text
canonical correct AUC                                      >= 0.55
canonical correct - K1-AK correct anchor                  >= -0.010
canonical correct - canonical wrong S-box independent      >= +0.005
canonical correct - canonical branch-off independent       >= +0.005
```

Train-seen rows are diagnostic only. Do not average seeds or splits to hide a
failed panel.

## 8. Decisions And Required Next Action

- **All gates pass:** retain the fixed Walsh transition representation and run
  the unchanged representation on one uKNIT-BC/Dialga family-transfer
  attribution panel before any scale.
- **Anchor retained but wrong S-box still substitutes:** stop single-cipher
  semantic regularization; preregister a multi-cipher shared-weight
  identifiability experiment where one backbone must consume correct runtime
  descriptors across uKNIT-BC, Midori64 and Dialga.
- **Branch off matches the candidate:** the canonical transition residual adds
  no usable signal; discard it and retain only the K1-AK/K1-AL causal evidence.
- **Anchor retention fails:** discard the fixed Walsh representation as too
  restrictive for the current budget.
- **Protocol invalid:** repair only the failed binding and rerun unchanged.

Do not add pairs, samples, epochs, seeds, rounds, positions, DDT/trail inputs,
MoE, coefficient scans, family transfer or remote execution inside K1-AN.

## 9. Required Artifacts

```text
run_id = i1_uknit_family_midori64_canonical_walsh_k1an_2048_seed6_seed7_20260729

results.jsonl
controls.jsonl
history.csv
checkpoint_manifest.json
dataset_manifest.jsonl
preflight.json
progress.jsonl
gate.json
validation.json
summary.json
comparison.csv
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, append observed metrics, decision and an executable next
action here, run `visual-qa-redraw`, and refresh both recent-result indexes.

## 10. Observed Result

Execution completed on the frozen six-row matrix. All `6/6` training rows,
`6/6` best checkpoints and `18/18` zero-training evaluation rows were present.
The validation gate passed all `35/35` protocol checks, including exact source
hashes, cache reuse without generation, same tensor initialization within each
seed, distinct seed initializations, identical state geometry, no trainable
Walsh/transition encoder, strict checkpoint replay and exact runtime
interventions.

Fresh AUC results were:

| Seed / split | Correct Walsh | Wrong S-box | Branch off | K1-AK correct | Correct - wrong | Correct - off | Correct - K1-AK |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed6 same-key | `0.640790` | `0.624883` | `0.576325` | `0.668338` | `+0.015908` | `+0.064465` | `-0.027547` |
| seed6 cross-key | `0.626022` | `0.611942` | `0.562415` | `0.656132` | `+0.014080` | `+0.063608` | `-0.030109` |
| seed7 same-key | `0.570003` | `0.589317` | `0.548059` | `0.663027` | `-0.019314` | `+0.021944` | `-0.093024` |
| seed7 cross-key | `0.580518` | `0.562349` | `0.540230` | `0.653863` | `+0.018169` | `+0.040288` | `-0.073345` |

The branch-off margin passed on every fresh panel, so the deterministic Walsh
branch was not ignored. Three of four wrong-S-box margins also passed. However,
the candidate lost `0.0275-0.0930` AUC against K1-AK on every fresh panel, far
outside the frozen `-0.010` retention allowance; seed7 same-key also let the
wrong S-box win by `0.0193`. The fixed 64-coefficient basis therefore removed
too much of the learnable transition signal without making the S-box semantics
fully identifiable.

```text
status   = hold
decision = innovation1_uknit_family_midori64_k1an_canonical_representation_too_restrictive
remote_scale = no
```

K1-AN is discarded. Do not scan Walsh coefficients, increase pairs, samples,
epochs or width, or launch this route remotely.

## 11. Executable Next Action

The next experiment is **K1-AO: multi-cipher shared-weight identifiability
readiness**, starting with a zero-training implementation audit. K1-AM showed
that independently trained wrong-S-box models can absorb runtime semantics;
K1-AN showed that removing the learnable coordinate system destroys too much
signal. K1-AO therefore restores the K1-AK learned transition path and changes
only ownership of the weights: one backbone must serve uKNIT-BC, Midori64 and
Dialga instead of allowing one independently optimized substitute per runtime.

Before any optimizer step, require:

```text
question = can one K1-AK geometry switch runtime descriptors per batch across
           uKNIT-BC, Midori64 and Dialga without cipher IDs or shape branches?

same-budget authorities:
  uKNIT-BC r5 = K1-T/K1-W local 2048/class seed3/4 caches and anchors
  Midori64 r4 = K1-AK local 2048/class seed6/7 caches and anchors
  Dialga r4   = K1-N/K1-W local 2048/class seed0/1 caches and anchors

single implementation variable:
  fixed one-runtime adapter -> runtime-switched shared-weight adapter

readiness gates:
  identical parameter names and shapes for 64-bit and 128-bit states
  one state hash strict-loads for all three runtime descriptors
  correct/wrong/branch-off interventions change the intended runtime only
  no cipher identity, absolute cell identity, key, label or difference metadata
  all six existing train/fresh cache payload hashes replay without generation
```

If readiness passes, preregister two shared training replicas. Replica 0 binds
uKNIT seed3, Midori seed6 and Dialga seed0; replica 1 binds seeds4/7/1. Each
cipher remains at `2048/class`, `4 pairs`, `10 epochs`, encrypted-random-
plaintext negatives and its already confirmed round/difference. Alternate
equal numbers of batches per cipher, compare the shared correct-runtime model
against reused same-budget independent anchors, and evaluate correct,
wrong-S-box and branch-off runtimes at the same shared checkpoint. Require every
cipher/seed fresh panel to retain its anchor within a preregistered tolerance
and require the correct runtime to beat both same-checkpoint controls; do not
average Dialga's high AUC over a failed uKNIT or Midori panel.

If state geometry or runtime switching fails, repair only that exact adapter
mismatch. If a cipher lacks a plan-aligned confirmed data surface, run the
existing difference-position qualification gate for that cipher instead of
training K1-AO. No remote execution, 16-pair expansion, MoE or new feature
family is authorized before this readiness audit passes.

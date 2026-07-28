# Innovation 1 uKNIT-Family CT-SPN Dialga Single-Pair Replay K1-AF

**Date:** 2026-07-29
**Status:** completed / held / Dialga single-pair semantic route closed
**Execution:** local CPU; reuse K1-AC checkpoints and validation caches

## Research question

K1-AE showed that Dialga r4 with sixteen pairs is saturated by the GF(2) base:
the full model reaches about `0.9999`, and turning off both structure residuals
still reaches `0.9986-0.9994`. Dialga r5 is not an evidence-backed replacement:
D3/D4 lost the signal and D5 found no eligible single-bit difference.

K1-AF asks one bounded question before any retraining:

> Does evaluating the frozen K1-AC representation on one ciphertext pair at a
> time create a learnable but non-saturated surface on which the correct S-box
> beats the wrong S-box?

## Single intervention

K1-AA parameters do not depend on pair count. Strictly load each exact K1-AC
best state into identical `214316`-parameter exact and wrong-S-box models whose
input contains one 256-bit Dialga ciphertext pair instead of sixteen pairs.

For every original validation sample, split its sixteen independent pairs into
sixteen single-pair observations. Evaluate all `2048 x 16 = 32768` observations
without choosing a favorable pair position.

Report three scopes:

1. **pooled single-pair AUC:** all 32768 observations; primary metric;
2. **per-position AUC:** sixteen finite-sample stability checks;
3. **mean-query aggregate AUC:** mean the sixteen single-pair probabilities
   back to one score per original sample; application-level support only.

The mean-query aggregate must not be reported as raw single-pair SOTA evidence.

## Frozen protocol

```text
cipher / rounds        = Dialga-128 / 4
source seeds           = 0,1 exact K1-AC best checkpoints
source validation      = exact K1-AC 1024/class cross-key cache
source pairs/sample    = 16
audit pairs/observation= 1
audit input width      = 256 bits
single-pair rows/seed  = 32768
difference             = 0x40
negative definition    = encrypted random plaintexts
conditions             = exact S-box, wrong S-box
training               = prohibited
optimizer steps        = 0
metric                 = AUC
```

Before the full audit, prove on a frozen fixture that direct one-pair inference
matches repeating that same pair sixteen times through the original input
geometry within `1e-6` logits for both conditions. This establishes that the
input-width change implements pair-count reduction rather than a new learned
function.

## Protocol gate

1. exactly 72 rows: two seeds by two conditions by one pooled, sixteen
   per-position and one aggregate scope;
2. all rows within a seed bind the same exact best checkpoint,
   pre-intervention state, source cache and source K1-AE gate;
3. one-pair models retain the exact state geometry and `214316` parameters;
4. direct-versus-repeated fixture error is at most `1e-6` in both conditions;
5. pooled rows contain 32768 observations, per-position rows 2048 and aggregate
   rows 2048;
6. all AUCs and exact-minus-wrong margins are finite;
7. no training, calibration, checkpoint selection or new data generation
   occurs.

## Research gate

For each seed independently require the pooled single-pair scope to satisfy:

```text
exact AUC >= 0.550          # usable signal
exact AUC <= 0.950          # not saturated
exact - wrong S-box >= 0.010
```

No seed or pair-position averaging may hide a failed pooled seed. Per-position
values diagnose variance but do not select training data.

## Decision routes

- **All three gates pass on both seeds:** preregister one local one-pair
  K1-AA exact/wrong-S-box training matrix at the existing `2048/class`, ten
  epochs and two seeds. Do not increase samples or change architecture.
- **Usable and non-saturated but semantic margin fails:** pair aggregation
  caused saturation, but reducing pairs does not restore correct-S-box
  attribution. Keep Dialga as a GF(2) signal-retention calibration only and
  seek another shared-primitive cipher surface; do not train one pair.
- **Pooled AUC remains above `0.950`:** one pair is still saturated; stop the
  Dialga semantic route rather than reducing queries mechanically again.
- **Pooled AUC is below `0.550`:** the frozen representation lacks useful
  one-pair signal. Do not infer that fresh one-pair training would succeed;
  keep the existing uKNIT evidence and close this Dialga reduction route.
- **Protocol invalid:** repair only the replay/source binding and rerun.

Blocked: neural training inside K1-AF, new samples, differences, rounds,
features, networks, MoE, remote scale and attack/SOTA/family-success claims.

## Required artifacts

```text
run_id = i1_uknit_family_ctspn_dialga_single_pair_replay_k1af_20260729
```

Produce result/progress JSONL, validation, gate, summary, per-position CSV and
a Chinese SVG. Apply `visual-qa-redraw`, refresh both result indexes, document
the exact decision and execute only the authorized next route.

## Completed result

The first execution preserved the frozen research metrics but was correctly
classified as protocol-invalid:

```text
output = outputs/local_audits/
         i1_uknit_family_ctspn_dialga_single_pair_replay_k1af_20260729/
status = invalid
failed_protocol_check = direct_repeat_equivalence
maximum float32 direct-versus-repeat logit error = 1.430511474609375e-6
frozen tolerance = 1e-6
```

This was a float32 reduction-order effect in attention, mean and maximum over
sixteen identical pair embeddings. The registered tolerance was not relaxed.
The equivalence proof was isolated from the AUC inference and rerun with exact
checkpoint values converted to float64 on copied fixture models. Formal pooled
inference remained float32. A regression test also proves that the one-pair
inference model is not mutated by the precision-controlled fixture audit.

The unchanged replay completed in a fresh output directory:

```text
output = outputs/local_audits/
         i1_uknit_family_ctspn_dialga_single_pair_replay_k1af_replay_fix_20260729/
status = hold
decision = innovation1_uknit_family_ctspn_k1af_one_pair_semantic_attribution_failed
protocol = pass; 72/72 rows; zero training; zero new data
direct-versus-repeat maximum error = 5.10702591327572e-15
remote_scale = no
```

Primary pooled single-pair results:

| Seed | Exact S-box AUC | Wrong S-box AUC | Exact - wrong |
|---:|---:|---:|---:|
| 0 | `0.803693948` | `0.808116160` | `-0.004422212` |
| 1 | `0.798364131` | `0.800548660` | `-0.002184529` |

Both seeds pass the registered usable-signal floor (`>= 0.550`) and the
non-saturation ceiling (`<= 0.950`), but both fail the correct-S-box semantic
margin (`>= +0.010`). Across the sixteen source pair positions, exact-minus-
wrong ranges from `-0.015838623` to `+0.004790783` for seed0 and from
`-0.008329391` to `+0.004487038` for seed1. Only `3/16` and `5/16` positions
respectively are positive, and no position reaches the semantic gate.

Mean-query aggregation returns to nearly saturated AUC:

```text
seed0 exact = 0.999197006; wrong = 0.999381065
seed1 exact = 0.999414444; wrong = 0.999306679
```

This confirms that aggregating sixteen independent pair scores explains the
near-perfect Dialga task signal, but it is application-level evidence and not
raw single-pair performance. Reducing pair count removes saturation without
making the correct S-box useful for ranking.

The Chinese SVG was rendered to a `1800 x 1022` pixel preview and passed the
`visual-qa-redraw` gate. Close AUC values are shown as explicit tables rather
than overlapping curves; the per-position panel, zero line, `+0.010` gate,
legend, scope disclaimer and all text were checked for overlap and clipping.

## Evidence-backed next action

Do not train a one-pair Dialga model, do not try Dialga r5 again, and do not
increase pair count, sample count, epochs or network capacity. K1-AC through
K1-AF jointly show that Dialga r4 is useful as a GF(2) signal-retention
calibration but not as a correct-S-box semantic adjudication surface.

Retain the positive uKNIT r5 sixteen-pair K1-AB result. The next research
question is whether the same fixed-shape runtime component vocabulary can be
qualified on native `Midori64`, the already-ranked homogeneous holdout that
shares the MANTIS/Midori `Sb0` and MIDORI linear primitive with uKNIT. Execute
this as a separate K1-AG qualification gate before any neural run:

1. implement and verify the Midori64 full cipher and prefix trace against
   public vectors;
2. generate a runtime descriptor and prove exact S-box and GF(2) transition
   reconstruction;
3. prove unchanged K1-AA parameter geometry and relabeling equivariance;
4. only then run a separate difference-position calibration with strict
   encrypted-random-plaintext negatives.

This sequence changes one uncertainty at a time. No Midori AUC, transfer,
family-generalization or remote-scale claim is authorized by K1-AF.

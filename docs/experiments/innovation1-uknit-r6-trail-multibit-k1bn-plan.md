# Innovation 1 uKNIT r6 Trail-Guided Multibit Boundary K1-BN

**Date:** 2026-07-29
**Status:** completed / hold / no r6 multibit discovery candidate
**Execution:** local CPU; deterministic candidate search; zero neural updates

## 1. Research Question

K1-BL and K1-BM completed a frozen scan of all 64 uKNIT r6 single-bit input
positions. None reached the required fresh AUC and exact-minus-raw thresholds.
This closes the single-bit family only. K1-BN asks:

> Does a multibit input difference selected before data inspection from uKNIT's
> exact per-round S-box DDTs and GF(2) linear layers retain reproducible r6
> five-stage signal?

## 2. Candidate Generator

The generator reads the implemented uKNIT tables directly and binds their
canonical SHA-256 digests. It searches two families:

```text
cell_local_multibit:
  all 16 cells x 11 nibble differences with Hamming weight >= 2 = 176

two_cell_low_spread:
  all cell pairs and nonzero nibble differences
  -> deterministic round-0 DDT/linear low-spread prefilter
  -> freeze the best 256 before the six-round search
```

For every pool member, a six-round beam search uses:

```text
beam width                  = 16
DDT outcomes / active cell  = 4
ranking                     = best characteristic log2 probability,
                              then fewer active S-boxes and lower output activity
```

Freeze the top 24 per family, 48 candidates total, before generating any
ciphertext dataset. DDT/trail values select the input difference only; they are
not neural inputs or labels.

## 3. Frozen Data Gate

All K1-BM data fields remain fixed:

```text
cipher / rounds        = uKNIT-BC / 6
runtime feature window = rounds 4..5
pairs/sample           = 4
negative definition    = encrypted random plaintexts
feature / controls     = exact five-stage / raw / label-shuffled
discovery              = seed2, 1024/class train, 512/class per fresh split
confirmation           = seed3/4, 2048/class train, 1024/class per fresh split
```

A discovery candidate must satisfy on both fresh splits:

```text
exact AUC   >= 0.550
exact - raw >= +0.010
```

Freeze at most one candidate per family. A candidate confirms only if every
seed3/4 fresh split also satisfies:

```text
exact AUC              >= 0.550
exact - raw            >= +0.010
exact - label-shuffled >= +0.030
```

## 4. Decisions

1. Protocol failure repairs only the failed invariant and reruns unchanged.
2. A confirmed candidate freezes the strongest difference and authorizes the
   uKNIT-only r6 16-pair exact/wrong-S-box/invariant neural matrix at
   `2048/class`.
3. No confirmed candidate records r5-to-r6 as the observed boundary for all 64
   single-bit positions and the frozen DDT-guided cell-local/two-cell families.
   It does not prove every possible r6 distinguisher random.
4. Remote `65536/class` is allowed only after the local r6 neural matrix passes.

Do not add candidates, another cipher, more data, more pairs, another model,
or different thresholds after reading K1-BN metrics.

## 5. Required Artifacts

```text
preflight.json
candidate_manifest.json
selection.json
dataset_manifest.jsonl
feature_manifest.jsonl
scorer_manifest.jsonl
results.jsonl
difference_candidates.csv
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
plot_report.json
curves.rendered.png
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, run `visual-qa-redraw`, refresh both recent-result indexes,
append metrics and the exact next action here, and commit/push the scoped files.

## 6. Completed Result

K1-BN completed all 48 frozen discovery candidates without neural training.
Every protocol check passed:

```text
dataset rows                     = 144
feature/result rows              = 288 / 288
scorer rows                      = 96
training rows / optimizer steps  = 0 / 0
failed protocol checks           = []
validation status                = pass
```

No candidate met both frozen discovery thresholds. The strongest worst-fresh
AUC within each family was:

| Family / candidate / difference | Worst fresh exact AUC | Worst exact - raw | Outcome |
|---|---:|---:|---|
| cell-local / `cm_c04_db` / `0x00000000000b0000` | `0.506416` | `-0.006287` | failed both floors |
| two-cell / `tc_c00d4_c01d8` / `0x0000000000000084` | `0.507462` | `+0.014385` | failed the `0.550` AUC floor |

The selection was empty, so the frozen procedure correctly performed no
seed3/4 confirmation and no neural training:

```text
status       = hold
decision     = innovation1_uknit_ctspn_k1bn_no_r6_multibit_candidate
remote_scale = no
```

The first rendered heatmap exposed overlapping long trail values such as
`-100.00`. The plotter was corrected to render those integer-valued log2
scores without redundant decimals. The final `2400 x 1680` pixel rendering
passed `visual-qa-redraw`: no title, label, heatmap value, threshold note or
decision text overlaps or clips.

## 7. Boundary Adjudication And Next Action

The combined plan-aligned evidence now establishes this empirical boundary:

```text
uKNIT r5:
  confirmed specialist neural signal at remote 65536/class
  exact AUC = 0.974540 / 0.967867 on seeds 3 / 4
  wrong-S-box AUC = 0.503901 / 0.505827

uKNIT r6:
  all 64 single-bit positions failed the frozen data gate
  all 48 frozen DDT/trail-guided multibit candidates failed the frozen data gate
  no candidate qualified for neural training
```

Therefore r5 is the last stable distinguishable round and r6 is the observed
random boundary for the searched single-bit and preregistered trail-guided
multibit families. This is not a mathematical proof that every one of the
`2^64 - 1` possible input differences, every feature or every future network is
random.

Do not train an r6 network on a benchmark that failed the prerequisite data
gate, do not launch remote r6 scale, and do not expand the K1-BN candidate list
after seeing its results. The next meaningful experiment is a separately
preregistered **r5 formal-scale specialist confirmation** if publication-level
evidence is required: keep the confirmed cell11 difference, four pairs,
exact/wrong-S-box/invariant models, optimizer, keys and epochs fixed; use at
least `1000000/class` and multiple seeds on the remote A6000. That run would
strengthen the last-stable-round claim, not reopen the r6 difference search.

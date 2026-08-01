# Innovation 1 K1-BW Learnable Runtime-SPN Structure Program Gate

**Date:** 2026-08-01  
**Status:** planned / local structure-only diagnostic  
**Run ID:** `i1_runtime_spn_structure_program_pretrain_k1bw_20260801`

## Research Question

K1-BK showed that one concatenated representation is not a shared semantic
solution for uKNIT, Midori and Dialga. uKNIT depends on position-preserving
multi-stage S-box semantics, Midori is dominated by exact linear transport,
and Dialga is nearly saturated by the linear branch. K1-BW therefore does not
train another shared distinguisher. It asks the prerequisite question for a
modular compiler:

> Can one cipher-name-free learnable encoder read actual S-box truth tables,
> actual source-to-target GF(2) edges and transition order, remain invariant to
> cell renaming, and preserve these distinctions on a whole held-out SPN?

Passing is only representation readiness. It does not establish neural
distinguishing accuracy, transfer, a universal SPN model or a publication
claim.

## One New Variable

Compile each runtime descriptor into an ordered structure program:

```text
per-cell S-box truth-table tokens
per-edge GF(2) source/target/bit-role tokens
  -> shared token encoders
  -> per-stage mean/max/RMS aggregation
  -> shared ordered GRU
  -> normalized 64-dimensional structure embedding
```

Cell labels are canonicalized from their actual bit membership before token
construction. No cipher name, cipher ID, block-width embedding, learned cell
lookup, per-cipher head, expert or distinguisher input is available.

## Protocol

Use the two-transition runtime windows already represented by the project for
GIFT, PRESENT, RECTANGLE, SKINNY, Midori, uKNIT and Dialga. Train on the first
six and hold out all Dialga structures from optimization. For each source
structure and four fixed corruption seeds compare:

1. a joint cell relabeling;
2. a source-permuted but edge-count-matched GF(2) operator;
3. an S-box truth-table input permutation;
4. reversed transition order when the two stages differ.

The loss keeps relabelings together and pushes each semantic intervention away
with a fixed cosine-distance margin. Two initialization/training seeds are
required. The same frozen evaluation is recorded before and after training so
an arbitrary random embedding cannot pass solely because it already produces
different hashes.

```text
epochs / seeds       = 160 / {0,1}
corruption seeds     = {11,23,37,53}
optimizer            = AdamW, lr 1e-3, weight decay 1e-5
device               = local CPU because local CUDA is unavailable
data                  = seven public runtime structure descriptors only
cipher samples        = none
neural distinguisher  = none
remote GPU            = prohibited
```

## Gates

Every structure must use identical parameter names and shapes. Every joint
cell relabeling must have cosine similarity at least `0.999999`. On the unseen
Dialga descriptor, every applicable wrong-linear, wrong-S-box and wrong-order
intervention must be at least `0.02` cosine distance farther than the relabel
positive, and the minimum margin must improve by at least `0.005` over the
same seeded untrained encoder. Both seeds must pass with no protocol error.

If the gate passes, K1-BX may attach the frozen structure program embedding to
small primitive adapters around the shared Runtime-E4 differential backbone.
K1-BX must compare correct descriptors against wrong-linear, wrong-S-box,
uniform-mixture and no-structure controls at the same local data budget. If
K1-BW fails, do not train K1-BX, increase structure width or use remote GPU;
inspect which primitive intervention remains unidentified.

## Required Artifacts

The completed local result must contain `preflight.json`, `geometry.json`,
`history.csv`, `results.jsonl`, `gate.json`, `validation.json`, `summary.json`,
`progress.jsonl` and a Chinese `curves.svg`. The figure must pass rendered-pixel
inspection through `visual-qa-redraw`, and the result indexes must be refreshed
before reporting.

## Completed Result

The two-seed structure-only run completed with all protocol checks passing:

```text
seven descriptors loaded            = true
Dialga held out from all optimization = true
identical parameter geometry         = true
cipher name / cipher ID input        = false
joint cell-relabel cosine             = 1.0 / 1.0
failed protocol checks               = []
```

The encoder learned the S-box intervention strongly on the unseen Dialga
descriptor, but did not learn equally strong actual-edge or stage-order
distinctions:

| Seed | Control | Initial minimum margin | Trained minimum margin |
|---:|---|---:|---:|
| 0 | wrong GF(2) connectivity | `0.001494` | `0.007066` |
| 0 | wrong S-box semantics | `0.068991` | `0.357920` |
| 0 | wrong transition order | `0.000529` | `0.001681` |
| 1 | wrong GF(2) connectivity | `0.000422` | `0.002279` |
| 1 | wrong S-box semantics | `0.032352` | `0.138371` |
| 1 | wrong transition order | `0.000289` | `0.002347` |

The preregistered held-out minimum margin and gain therefore failed on both
seeds:

```text
seed0 minimum margin / gain = 0.001681 / 0.001152
seed1 minimum margin / gain = 0.002279 / 0.001668
required margin / gain      = 0.020000 / 0.005000

status       = hold
decision     = innovation1_runtime_spn_k1bw_structure_program_not_ready
remote_scale = no
```

This is a useful decomposition rather than a generic model failure. The shared
truth-table token path generalized to an unseen heterogeneous SPN. The current
globally pooled GF(2) edge path and one-vector GRU summary did not preserve
enough operator and order information on that same holdout. Connecting this
embedding to Runtime-E4 now would repeat the earlier error of letting a strong
S-box channel hide weak linear topology semantics.

The final Chinese figure was rendered at `2700 x 1800` pixels. After changing
the two semantic-margin panels from linear to logarithmic axes and adding
per-cipher value labels, `visual-qa-redraw` found no text overlap, clipping,
missing glyph, legend occlusion or unreadable near-zero margin.

## Evidence-Backed Next Action

Do not start K1-BX differential training yet. Preregister K1-BX0 as a
structure-only linear/order repair that changes one representation variable:

```text
K1-BW edge path: edge token -> global stage pool -> one stage vector
K1-BX0 candidate: edge token -> target-cell message aggregation
                  -> ordered per-cell transition tokens
                  -> cell-set pool only after transition processing
```

Keep the S-box token path, seven descriptors, six training ciphers, Dialga
holdout, two seeds, corruption seeds, optimizer budget and all gates unchanged.
The same K1-BW embedding is the same-budget anchor. Add an edge-token-shuffled
control to prove that any gain comes from actual endpoint binding rather than
extra parameters. If K1-BX0 still misses `0.02` on wrong-linear or wrong-order
controls, stop structure-only pretraining and return to deterministic primitive
routing; do not add width, epochs, ciphers, remote GPU or a neural distinguisher.

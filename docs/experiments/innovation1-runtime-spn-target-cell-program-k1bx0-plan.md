# Innovation 1 K1-BX0 Target-Cell Structure Program Repair

**Date:** 2026-08-01
**Status:** completed / hold / local structure-only diagnostic
**Run ID:** `i1_runtime_spn_target_cell_program_k1bx0_20260801`

## Research Question

K1-BW learned held-out Dialga S-box semantics but failed the actual GF(2)
connectivity and transition-order gates. Its shared edge MLP pooled all edges
into one stage vector before an edge could update the cell that it actually
targets. K1-BX0 changes that one representation decision:

> Does binding every edge message to its real target cell before pooling make
> a cipher-name-free structure program retain unseen linear topology and stage
> order without losing K1-BW's S-box semantics?

This remains structure-only representation research. No ciphertext dataset,
label, differential neural distinguisher or remote training is introduced.

## One Representation Change

K1-BW anchor:

```text
S-box tokens -> global stage pool
GF(2) edge tokens -> global stage pool
stage vectors -> ordered GRU -> structure embedding
```

K1-BX0 candidate:

```text
GF(2) edge token
  -> aggregate at its actual target cell
  -> fuse with that cell's S-box truth-table token
  -> shared GRUCell updates each cell in transition order
  -> pool cells only after the transition update
  -> ordered stage GRU -> structure embedding
```

All learned functions are shared across cells, stages, ciphers and state widths.
The model receives no cipher name, cipher ID, width embedding, per-cipher head
or per-cipher parameter. A new same-checkpoint `wrong_edge_binding` control
keeps every token and weight fixed but assigns edge messages to the wrong target
cells.

## Frozen Protocol

K1-BX0 binds K1-BW's config, result, validation and gate digests. The following
fields are unchanged:

```text
train structures      = GIFT, PRESENT, RECTANGLE, SKINNY, Midori, uKNIT
whole-cipher holdout  = Dialga-128
runtime window        = two transitions per structure
model seeds           = {0,1}
corruption seeds      = {11,23,37,53}
epochs                = 160
optimizer             = AdamW, lr 1e-3, weight decay 1e-5
triplet margin        = 0.12
hidden / output       = 48 / 64
device                = local CPU; local CUDA is unavailable
```

For each structure evaluate a transported joint cell relabeling, wrong GF(2)
operator, wrong S-box, wrong transition order when applicable, and wrong edge
binding. Record the same seeded untrained encoder before optimization.

## Gates

Both model seeds must satisfy every clause:

```text
joint cell-relabel cosine similarity             >= 0.999999
every Dialga wrong-control semantic margin       >= 0.020
every Dialga margin gain over random initialization >= 0.005
wrong-S-box minimum margin                       >= matching K1-BW - 0.020
failed protocol checks                           = []
```

The absolute `0.020` requirement already forces the previously failed
wrong-linear and wrong-order margins to improve materially over K1-BW. Seed
averaging cannot hide a failed seed or control.

## Decisions

- **All gates pass:** authorize K1-BX local differential readiness. Freeze the
  structure encoder and compare one small Runtime-E4 conditioner against
  correct, wrong-linear, wrong-S-box, wrong-binding, uniform and no-structure
  controls.
- **S-box passes but linear/order/binding fails:** reject structure-vector
  pretraining for the current method. Return to deterministic primitive routing
  rather than adding width, epochs, ciphers or remote scale.
- **S-box retention fails:** reject the target-cell repair; it displaced a
  supported K1-BW semantic channel.
- **Protocol failure:** repair only the failed source, geometry or artifact
  invariant and rerun unchanged.

No result directly authorizes medium data, remote GPU, MoE, differential
training, more epochs, more structures or a universal-SPN claim.

## Required Artifacts

Write `preflight.json`, `geometry.json`, `history.csv`, `results.jsonl`,
`gate.json`, `validation.json`, `summary.json`, `progress.jsonl` and a Chinese
`curves.svg` under the local diagnostic output root. The final SVG must pass a
`2700 x 1800` rendered-pixel `visual-qa-redraw` inspection before indexing and
reporting.

## Completed Result

The frozen two-seed, 160-epoch run completed locally with no protocol errors:

```text
validation status = pass
result rows        = 368
history rows       = 320
gate status        = hold
decision           = innovation1_runtime_spn_k1bx0_target_cell_program_not_ready
```

Dialga whole-cipher holdout minima were:

| Model seed | Control | Initial-to-trained gain | Trained margin | Gate |
|---:|---|---:|---:|---|
| 0 | wrong linear operator | +0.021039 | 0.023484 | pass |
| 0 | wrong S-box | +0.699626 | 0.744355 | pass |
| 0 | wrong stage order | +0.005049 | 0.007120 | **fail margin** |
| 0 | wrong target-cell binding | +0.374939 | 0.376031 | pass |
| 1 | wrong linear operator | +0.016286 | 0.020673 | pass |
| 1 | wrong S-box | +1.092889 | 1.156195 | pass |
| 1 | wrong stage order | +0.000741 | 0.001706 | **fail gain and margin** |
| 1 | wrong target-cell binding | +0.306183 | 0.306676 | pass |

Cell-relabel cosine similarity remained `0.9999999404` for seed 0 and
`0.9999998808` for seed 1. The K1-BW wrong-S-box anchors were retained and
improved. Target-cell aggregation therefore repaired endpoint binding and moved
the unseen Dialga linear-operator margins just above `0.020` on both seeds, but
it did not make the two-stage order observable with a stable margin.

The final SVG passed `visual-qa-redraw` after shortening crowded tick labels and
raising the logarithmic-axis headroom. The inspected raster was `2700 x 1800`;
the visual report and pass marker are stored beside the result artifacts.

## Verdict And Recommended Next Action

K1-BX0 does **not** authorize K1-BX differential conditioning. The evidence
rejects the current whole-structure-vector pretraining route: pooling the
ordered cell states into one global embedding still permits stage-order
semantics to collapse, even though S-box, linear-operator and endpoint-binding
interventions are distinguishable.

The next research question is therefore compiler-first rather than another
training scale-up:

> Can a deterministic structure compiler preserve the exact ordered primitive
> schedule and route each stage to shared learned primitive experts, while
> keeping expert parameter shapes independent of cipher identity?

Use K1-BX0 as the fixed failed global-vector anchor. First perform a local,
non-training descriptor audit over the same seven structures. Change only the
representation boundary: emit an ordered primitive sequence instead of one
pooled vector. Required controls are exact descriptor replay, joint cell
relabeling, wrong stage order and wrong target-cell binding. Readiness requires
zero parse/geometry errors, exact schedule reconstruction for all seven
descriptors, relabel equivalence, and deterministic rejection of both wrong
order and wrong binding. Only after that audit passes may one small local
same-budget Runtime-E4 comparison be preregistered with correct routing,
wrong-order routing, wrong-binding routing and no-structure controls.

Do not increase encoder width, epochs, training ciphers or data scale; do not
launch remote training; and do not connect the current K1-BX0 embedding to a
differential backbone. Those actions do not address the failed order gate.

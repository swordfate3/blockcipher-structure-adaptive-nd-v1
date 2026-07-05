# Innovation 1 PRESENT SGP Stable-Axis Audit Plan

**Date:** 2026-07-06

**Status:** local audit ready / no neural training / no remote launch

## Question

Can the new SGP route find PRESENT r8 SPN-sensitive feature axes that are:

```text
weak-positive as a stable composite
stable across local seeds
stronger than shuffled-cell false-family control
```

This is a route-selection audit only. It does not train a neural model and does
not claim a PRESENT r8 result.

## Why This Exists

The independent route recheck in
`docs/research/innovation1-spn-independent-route-recheck-20260706.md` ranked
stable score/feature-axis mining above wider near-neighbor neural ensembles.
Current local evidence says:

```text
projection v2 = unstable hand-picked priors
candidate-trail = stopped at medium scale but has weak local axes
GPD/beamstats = weak local candidate only
aligned integral neural route = below deterministic baseline
near-neighbor ensemble = weak-positive below gate
```

So the next question is not whether to add more models. It is whether any
SPN-derived axis family is stable enough to become a real non-neighbor weak
expert candidate.

## Protocol

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_sgp_stable_axis_audit_r8_local.json
```

Core protocol:

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `8` |
| Samples per class | `2048` |
| Seeds | `0, 1` |
| Key split | `validation` |
| Pairs per sample | `16` |
| Negative mode | `encrypted_random_plaintexts` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Difference profile | `present_zhang_wang2022_mcnd`, member `0` |

Feature source:

```text
candidate_evidence / cell_structured / beam_width=4 / depth=3
candidate_evidence / aggregate / beam_width=4 / depth=3
InvP(delta) bits / ciphertext_xor_spn_paligned_bits
```

Control source:

```text
candidate_evidence / cell_structured_shuffled / beam_width=4 / depth=3
```

The shuffled-cell control is same-dimensional only for the `cell_structured`
source. Aggregate and InvP(delta) source reports use chance (`0.5`) as the
control floor in this local audit.

## Gate

The audit is a candidate only if all are true:

```text
stable-axis composite AUC >= 0.55 at 2048/class
top-k Jaccard >= 0.35 across the two seed probes
composite AUC beats shuffled-cell control by >= 0.01
```

If any gate fails:

```text
decision = sgp_stable_axis_hold
action = do not create SGP projection smoke yet
```

If all gates pass:

```text
decision = sgp_stable_axis_candidate
action = create a lean local projection smoke using sgp_top32_stable,
         with old projection v2 and shuffled/random masks as controls
```

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --sgp-stable-axis-config configs/experiment/innovation1/innovation1_spn_present_sgp_stable_axis_audit_r8_local.json \
  --samples-per-class 2048 \
  --top-k 32 \
  --output outputs/local_audits/i1_present_r8_sgp_stable_axis_audit_2048.json
```

The final source-sweep artifact was written to:

```text
outputs/local_audits/i1_present_r8_sgp_source_sweep_axis_audit_2048.json
```

## 2026-07-06 Local Source Sweep Result

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --sgp-stable-axis-config configs/experiment/innovation1/innovation1_spn_present_sgp_stable_axis_audit_r8_local.json \
  --samples-per-class 2048 \
  --top-k 32 \
  --output outputs/local_audits/i1_present_r8_sgp_source_sweep_axis_audit_2048.json
```

Result:

```text
decision = sgp_stable_axis_hold
best_source = invp_delta_bits
```

| Source | Decision | Min composite AUC | Mean composite AUC | Top-k Jaccard | Control AUC max | Control delta |
|---|---|---:|---:|---:|---:|---:|
| `candidate_cell_structured` | `sgp_stable_axis_hold` | `0.5262441635131836` | `0.5410444736480713` | `0.14285714285714285` | `0.5273032188415527` | `-0.0010590553283691406` |
| `candidate_aggregate` | `sgp_stable_axis_hold` | `0.5145435333251953` | `0.528357982635498` | `0.16363636363636364` | `0.5` | `0.014543533325195312` |
| `invp_delta_bits` | `sgp_stable_axis_hold` | `0.5609222650527954` | `0.5676992535591125` | `0.0` | `0.5` | `0.06092226505279541` |

Interpretation:

```text
candidate_cell_structured: weak and not better than shuffled-cell control
candidate_aggregate: too weak for projection smoke
invp_delta_bits: composite AUC clears 0.55, but raw top-k axes do not overlap
                 across seeds, so raw-axis top32 projection is not stable
```

Decision:

```text
do_not_create_sgp_top32_projection_smoke_yet
do_not_remote_launch_sgp
```

Next better step:

```text
test orbit/grouped stability over InvP(delta) axes
```

Rationale: the InvP(delta) source has weak-positive composite evidence but raw
bit-axis identity is unstable. This may mean the useful signal is stable only
after grouping by SPN cell, pair slot, P-layer orbit, or bit-role, not by exact
flat feature index. Do not weaken the raw-axis Jaccard gate to force a pass.

Engineering note:

```text
multi-source SGP audit currently regenerates candidate evidence in memory and
has no progress output. Do not scale this path until cache/progress support is
added for SGP source sweeps.
```

## 2026-07-06 Grouped/Orbit Follow-Up Result

The raw-axis result suggested a possible grouped signal, so a stricter local
follow-up was added:

```text
config = configs/experiment/innovation1/innovation1_spn_present_sgp_grouped_axis_audit_r8_local.json
artifact = outputs/local_audits/i1_present_r8_sgp_grouped_axis_audit_2048_top4.json
source = invp_delta_bits / ciphertext_xor_spn_paligned_bits
group_schemes = pair_word_cell, word_cell, cell, word_bit_role, p_layer_orbit
top_k = 4
stable_top_k = 4
max_selected_axis_fraction = 0.75
```

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --sgp-grouped-axis-config configs/experiment/innovation1/innovation1_spn_present_sgp_grouped_axis_audit_r8_local.json \
  --samples-per-class 2048 \
  --top-k 4 \
  --output outputs/local_audits/i1_present_r8_sgp_grouped_axis_audit_2048_top4.json
```

Result:

```text
decision = sgp_grouped_axis_hold
best_group_scheme = word_bit_role
```

| Group scheme | Decision | Min composite AUC | Top-k Jaccard | Group count | Mask axes | Mask fraction | Stable groups |
|---|---|---:|---:|---:|---:|---:|---|
| `pair_word_cell` | `sgp_grouped_axis_hold` | `0.5344549417495728` | `0.0` | `512` | `16` | `0.0078125` | `pair1:invp_delta:cell1`, `pair1:delta:cell8`, `pair13:delta:cell11`, `pair3:delta:cell0` |
| `word_cell` | `sgp_grouped_axis_hold` | `0.6075923442840576` | `0.14285714285714285` | `32` | `256` | `0.125` | `delta:cell8`, `invp_delta:cell12`, `delta:cell11`, `invp_delta:cell2` |
| `cell` | `sgp_grouped_axis_hold` | `0.6401443481445312` | `0.14285714285714285` | `16` | `512` | `0.25` | `cell8`, `cell11`, `cell12`, `cell2` |
| `word_bit_role` | `sgp_grouped_axis_hold` | `0.685741662979126` | `0.14285714285714285` | `8` | `1024` | `0.5` | `invp_delta:bit2`, `delta:bit0`, `delta:bit2`, `delta:bit1` |
| `p_layer_orbit` | `sgp_grouped_axis_hold` | `0.5724446773529053` | `0.0` | `48` | `192` | `0.09375` | `invp_delta:orbit6`, `delta:orbit6`, `invp_delta:orbit11`, `delta:orbit11` |

Interpretation:

```text
There is broad weak separation in InvP(delta), especially when grouped by cell
or bit role, but the top groups are not stable across seeds. Pair-slot-aware
groups and P-layer orbit groups are even less stable. The coarse word_bit_role
signal is too broad to be a useful projection expert, and a degenerate full-width
group mask is now explicitly guarded by max_selected_axis_fraction.
```

Decision:

```text
do_not_create_grouped_sgp_projection_smoke
do_not_remote_launch_sgp
do_not_use_grouped_sgp_as_a_diverse_expert_yet
```

Next action:

```text
Retire SGP as the next immediate projection route. Prefer a representation
route that aggregates the broad weak InvP(delta) signal intentionally, such as
learned pair/global statistics or deterministic bit-role/cell distribution
features, and compare it against existing pairset/global-stats anchors before
any remote launch.
```

## Claim Scope

This is a local diagnostic. A positive result only means the route deserves a
small projection smoke. It is not scale evidence, not formal training, not
remote-launch evidence, and not an ensemble result.

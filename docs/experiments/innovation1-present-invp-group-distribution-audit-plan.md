# Innovation 1 PRESENT InvP Group Distribution Audit Plan

**Date:** 2026-07-06

**Status:** local audit planned / no neural training / no remote launch

## Question

Raw/grouped SGP showed broad weak evidence in `InvP(delta)`, but exact axes and
groups were unstable. Existing `present_global_pairset_statistics` then held,
suggesting that generic global activity statistics are too coarse.

This audit asks:

```text
If group identity moves across seeds, do distribution statistics over SPN groups
remain stable enough to become an explicit representation route?
```

## Protocol

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_group_distribution_audit_r8_local.json
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
| Feature source | `ciphertext_xor_spn_paligned_bits` |
| Negative mode | `encrypted_random_plaintexts` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Difference profile | `present_zhang_wang2022_mcnd`, member `0` |

Group schemes:

```text
pair_word_cell
word_cell
cell
word_bit_role
p_layer_orbit
```

For each scheme, compute group activity per sample, then audit distribution
statistics:

```text
activity_mean
activity_std
activity_max
top2_activity_mean
top4_activity_mean
bottom2_activity_mean
bottom4_activity_mean
activity_span
```

## Gate

Keep the route only if:

```text
top-k oriented statistics composite AUC >= 0.60
top-k statistic-name Jaccard >= 0.35 across seeds
best individual statistic AUC >= 0.55
```

If this passes, the next step is a lean local representation smoke against the
existing pairset/global-stat controls. If it fails, the broad InvP(delta) signal
is not captured by this deterministic group-distribution bank.

## Claim Scope

This is a local deterministic feature audit. It is not neural training, not
scale evidence, not a remote launch gate, not an ensemble result, and not a
breakthrough claim.

## 2026-07-06 Local Result

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --invp-group-distribution-config configs/experiment/innovation1/innovation1_spn_present_invp_group_distribution_audit_r8_local.json \
  --samples-per-class 2048 \
  --top-k 16 \
  --output outputs/local_audits/i1_present_r8_invp_group_distribution_audit_2048.json
```

Result:

```text
decision = invp_group_distribution_hold
best_source = invp_delta_group_distribution
artifact = outputs/local_audits/i1_present_r8_invp_group_distribution_audit_2048.json
```

| Metric | Value |
|---|---:|
| Stat feature dim | `40` |
| Best stat AUC min | `0.514545202255249` |
| Composite AUC min | `0.5135400295257568` |
| Composite AUC mean | `0.5136241912841797` |
| Top-k Jaccard min | `0.18518518518518517` |

Interpretation:

```text
The targeted group-distribution bank did not recover the broader grouped-SGP
signal. Simple unsupervised distribution summaries over cell/word-cell/bit-role
and P-layer-orbit activities are too weak and unstable.
```

Decision:

```text
do_not_create_group_distribution_representation_smoke
do_not_remote_launch_group_distribution_stats
do_not_add_this_as_a_diverse_expert
```

Next action:

```text
Stop deterministic InvP(delta) aggregation for now. If continuing this family,
use a learned representation that can model pair/group interactions directly,
or shift the next local slot to data/difference search. Do not keep adding
handwritten aggregate statistics around the same weak evidence.
```

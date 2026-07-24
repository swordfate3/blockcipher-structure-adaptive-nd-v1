# Innovation 1 uKNIT U2-F Runtime Delta-U Query Plan

Date: 2026-07-24

## Status

```text
stage    = planned local diagnostic
run_id   = i1_rtg1_uknit64_runtime_e4_delta_u_query_u2f_2048_seed0_seed1_20260724
training = not started
decision = pending
```

## Research Question

Does a separate, sample-conditioned runtime inverse-S-box difference query add
stable information to the stronger U2-C state-triplet representation without
replacing or averaging away that representation?

U2-C retained the strongest raw state signal. U2-D showed that executing the
correct per-cell inverse S box can expose S-box ownership. U2-E showed that a
fixed mean of the two complete representations weakens the state anchor. U2-F
therefore keeps both U2-C state inputs unchanged and changes only the value of
a third, capacity-matched query token.

## Frozen Protocol

```text
cipher          = uKNIT-BC prefix r4
runtime window  = round_start 2, processor_steps 2
train           = 2048/class
validation      = 1024/class
seeds           = 0,1
epochs          = 10
pairs/sample    = 4
feature         = ciphertext_pair_bits
negative        = encrypted_random_plaintexts
train key       = 0x00000000000000000000000000000000
validation key  = 0x11111111111111111111111111111111
loss            = MSE
optimizer       = Adam, lr 1e-4, weight decay 1e-5
checkpoint      = best validation AUC
device          = local CPU
cache           = reuse the exact U2-B through U2-E disk cache
```

This is a local mechanism diagnostic. It is not formal or paper-scale
evidence, an attack, SOTA, cross-cipher evidence, or a breakthrough.

## One Variable

For ciphertext endpoints `C` and `C'`, compute:

```text
V       = L_inverse(C)
V'      = L_inverse(C')
deltaV  = V xor V'
deltaU  = S_inverse(V) xor S_inverse(V')
```

Every row uses a three-input fusion with identical parameter geometry:

```text
input 1 = unchanged U2-C current-state triplet embedding
input 2 = unchanged U2-C previous-state triplet embedding
input 3 = one four-bit-per-cell query embedding
```

Only input 3 changes:

| Role | Runtime structure | Third query token |
| --- | --- | --- |
| candidate | correct | `deltaU` from the correct runtime inverse S boxes |
| same-budget anchor | correct | `deltaV`, with no inverse-S-box lookup |
| ownership control | S-box assignment shuffled only | `deltaU` from shuffled per-cell inverse S boxes |

The candidate and anchor modes are respectively
`state_triplet_delta_u_query` and `state_triplet_delta_v_query`. Neither may
average, replace, or otherwise alter the first two state-triplet inputs.

Matrix:

```text
configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_delta_u_query_u2f_2048_seed0_seed1.csv
```

## Readiness Gate

Before training require:

1. Candidate, identity anchor and shuffled control have identical state-dict
   geometry and `458850` parameters.
2. The first two fusion inputs are exactly equal between the candidate and
   identity anchor under shared weights and fixed inputs.
3. The identity query equals encoded `deltaV` and does not call the inverse
   S-box operator.
4. The candidate query equals encoded `deltaU` from the runtime descriptor.
5. Correct and shuffled S-box ownership change the candidate query and logits.
6. Pair swap and joint cell relabeling preserve candidate logits.
7. Independent relation mode bypasses runtime inverse operators.
8. Existing difference-only, state-triplet, inverse-S-box-triplet and
   dual-view paths retain their previous geometry and regressions.
9. The six-row plan parses and all relevant runtime-SPN tests pass.

Any failure blocks training without changing the frozen matrix.

## Research Gate

For each seed require:

```text
candidate AUC >= 0.520
candidate AUC - deltaV identity anchor AUC >= +0.005
candidate AUC - shuffled deltaU query AUC >= +0.005
```

Both seeds must pass. A pass opens a same-checkpoint query swap audit before
any scale discussion. A miss closes this exact query-token representation and
returns to the U2-C state-triplet anchor.

## Execution And Outputs

```text
output root = outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2f_2048_seed0_seed1_20260724
cache root  = outputs/local_cache/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724
```

Require `results.jsonl`, `progress.jsonl`, checkpoints, `validation.json`,
`gate.json`, `summary.json`, `history.csv`, `curves.svg` and a visual-QA marker.
After completion, refresh both recent-result indexes.

## Blocked Routes

Do not add DDT or trail probabilities, guessed subkeys, partial decryption,
more pairs, another difference/window, extra epochs, larger data, more seeds,
learned query gates, auxiliary losses, or remote GPU execution inside U2-F.
Do not compare its AUC directly with Liu et al. or promote it as formal
evidence.

## Evidence-Dependent Next Action

- Pass: run one same-checkpoint correct-versus-shuffled query swap audit; do
  not scale first.
- Hold: close the explicit delta-U query representation and keep U2-C as the
  uKNIT state-input anchor.
- Fail: repair only the protocol/readiness mismatch and rerun the unchanged
  six-row matrix; do not interpret its AUC values.

# Innovation 1 Runtime-SPN Multiscale Orbit-Basis Ranking

Date: 2026-07-26

## Evidence Position

C2 froze the safe current method name as `runtime GF(2)-topology-aware SPN
neural distinguisher`. D1 then rejected deterministic elementary-mutation
topology expansion because only SKINNY and uKNIT passed the cipher-balanced
coverage gate; PRESENT, GIFT and RECTANGLE became worse than the relabeling
anchor, and Dialga remained below threshold.

The running C3 formal PRESENT experiment remains the highest-priority evidence.
No new training should compete with or modify it.

## Candidate Route

The candidate is a fixed multiscale inverse-linear orbit:

```text
O_0(x) = x
O_d(x) = L_(r-d)^-1 ... L_(r-2)^-1 L_(r-1)^-1 x
depths = 0, 1, 2, 4, 8
```

For a homogeneous P layer this becomes powers of one inverse operator. For an
ordered heterogeneous window it follows the supplied transitions backward and
cycles only when the requested depth exceeds the frozen window length. The
view count is fixed, so a later shared cell-wise encoder would not change
parameter geometry with cipher width or round-window length.

## Why This Is Not A Closed Route

- It uses exact runtime matrices as operators, not truth-table, ANF, DDT or
  learned descriptor conditioning.
- It has no Adapter, FiLM, MoE, hypernetwork, cipher id or learned router.
- It uses only real supplied topologies, not the rejected D1 synthetic
  elementary mutations.
- It preserves operator depths explicitly instead of adding another typed-GNN
  residual to the same hidden state.
- It changes representation depth, not validation data, labels, negatives,
  optimizer, loss or C3 protocol.

## Literature Position

MixHop provides a verified high-order adjacency-mixing precedent and SIGN
provides a verified fixed multi-operator-view precedent. Liu et al. provides
the closest direct SPN precedent for one inverse-layer-derived view. No checked
SPN neural-distinguisher source evaluates one shared runtime model with a fixed
`0/1/2/4/8` exact GF(2) orbit. This is a bounded literature gap, not a novelty
proof.

Full source record:

```text
sources/research_runtime_spn_multiscale_orbit_basis_20260726.md
```

## Ranked Actions

1. Keep C3 running under its existing monitor and consume only retrieved local
   artifacts.
2. Run C4 as a local zero-training operator/readiness audit.
3. Permit a later same-budget local neural diagnostic only if C4 passes and C3
   has completed; preregister that diagnostic separately.
4. Do not reopen synthetic-topology D2, S-box conditioning, ANF, Adapter, FiLM,
   typed-GNN, MoE, recurrent-size or Transformer routes.

## Claim Boundary

A C4 pass would prove only that the exact view bank is non-collapsed,
topology-sensitive, ordered-window-sensitive and cell-relabel equivariant on
the six known structures. It would not prove learnability, differential signal,
zero-shot transfer, unseen-cipher adaptation, an attack, SOTA or the full
Innovation 1 goal.

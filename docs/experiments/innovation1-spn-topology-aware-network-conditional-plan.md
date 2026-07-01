# Innovation 1 SPN Topology-Aware Network Conditional Plan

**Date:** 2026-07-01

**Status:** conditional next-route plan / not launched

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict
encrypted-random-plaintext negatives. This plan is prepared as the next
network-architecture route if the active DDT graph seed1 variance check does
not produce stable support, or if the project needs a cleaner architecture
claim beyond feature-level DDT priors.

## Why This Plan Exists

The current strongest completed Innovation 1 route is:

```text
model = present_nibble_invp_only_spn_only
scale = 1000000/class
seed0 AUC = 0.797470988906
seed1 AUC = 0.797347588554
claim_scope = two-seed paper-scale positive confirmation, not formal evidence
```

The completed paper-scale attribution controls support the explanation that
true `InvP(DeltaC)` / P-layer alignment carries useful SPN structure:

```text
decision = support_invp_structural_attribution
attribution_margin = 0.003726063600
```

The active DDT graph branch tests whether adding S-box DDT priors and a
topology graph improves beyond this InvP-only anchor. Seed0 was a weak
diagnostic signal:

```text
run_id = i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630
decision = weak_ddt_graph_signal
margin_vs_same_graph_no_ddt_auc = 0.000472726912
required_margin = 0.001
next_action = 262144/class seed1 variance check
```

If seed1 does not turn this into stable support, the next useful question is
not "add more hand-crafted DDT features". It is:

```text
Can the network architecture itself make better use of the already-supported
InvP/P-layer structure?
```

## Research Question

Can an explicit PRESENT topology-aware network, using the same InvP-only
structural input, improve over the current InvP-only token-mixer anchor?

More concretely:

```text
Does fixed PRESENT P-layer message passing over 16 S-box/nibble cells add a
real structure prior beyond the same input processed by a generic token mixer?
```

## Single Hypothesis

Change exactly one factor:

```text
network topology = generic token mixing -> fixed PRESENT P-layer message passing
```

Keep unchanged:

```text
cipher = PRESENT-80
rounds = 7
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
pairs_per_sample = 16
feature family = InvP(DeltaC)-only nibble/cell view
loss = mse
optimizer = adam
lr_scheduler = official_cyclic
checkpoint_metric = val_auc
validation key = 0x11111111111111111111
```

Do not add in this first topology-aware network route:

```text
raw Zhang/Wang MCND branch
DDT priors
candidate trail features
new negative samples
new sample structure
new pair-set aggregation objective
multi-task auxiliary loss
```

## Candidate Model Sketch

Proposed model key:

```text
present_nibble_invp_p_layer_graph_spn_only
```

Input:

```text
raw ciphertext pairs -> DeltaC -> InvP(DeltaC)
```

Per pair:

```text
nodes = 16 PRESENT S-box cells
node feature = 4-bit InvP(DeltaC) nibble
edge/message source = fixed PRESENT P-layer nibble adjacency
```

Network:

```text
shared nibble encoder
2-3 fixed P-layer message-passing blocks
pair-level embeddings
same evidence pooling family as current InvP route
binary classifier
```

Required topology control:

```text
present_nibble_invp_shuffled_p_layer_graph_spn_only
```

This control must keep the same input and model budget but replace true P-layer
adjacency with deterministic shuffled adjacency. No topology claim is allowed
without this control.

Implementation note:

```text
src/blockcipher_nd/models/structure/spn/present_p_layer_mixer.py already has
PresentPLayerMixerBlock and PRESENT nibble adjacency logic. Reuse this instead
of creating a second graph convention.
```

## First Matrix

First non-smoke scale:

```text
262144/class
seed = 0
```

Keep the matrix lean:

| Row | Model | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | current strongest same-input anchor |
| 1 | `present_nibble_invp_p_layer_graph_spn_only` | true-P topology candidate |
| 2 | `present_nibble_invp_shuffled_p_layer_graph_spn_only` | shuffled topology control |

Do not include Zhang/Wang in this matrix. The question is not whether InvP is
better than raw MCND; that is already supported. The question is whether
topology-aware message passing improves the already-supported InvP route.

## Gates

Primary metric:

```text
val_auc
```

Support:

```text
true-P graph AUC >= InvP-only anchor AUC + 0.001
and true-P graph AUC >= shuffled-P graph AUC + 0.001
and calibrated_accuracy is not worse than InvP-only
```

Weak signal:

```text
true-P graph is best but one required margin is < 0.001
```

Action:

```text
run 262144/class seed1 variance check before any 1M run
```

Stop:

```text
true-P graph <= InvP-only anchor
or true-P graph <= shuffled-P graph
```

Action:

```text
record as tied/negative topology evidence; do not scale this architecture.
Switch to candidate-trail / transition-consistency representation or a new
data-structure hypothesis.
```

## Branch Rules

This plan becomes actionable only after the active DDT seed1 result is
retrieved, validated, and postprocessed.

If DDT seed1 gives stable support:

```text
Do not launch this route immediately.
First decide whether to run a lean 1M DDT confirmation matrix.
```

If DDT seed1 is weak, tied, negative, or unstable:

```text
Activate this topology-aware network route.
Implement the model and shuffled control.
Run smoke/readiness.
Commit and push.
Launch 262144/class seed0 from the pushed commit.
Hand off retrieval/postprocess to tmux watcher.
```

If DDT seed1 fails operationally:

```text
Repair or rerun DDT seed1 first unless the failure proves the DDT route is not
actionable. Do not use an infrastructure failure as evidence against DDT.
```

## Local Readiness Checklist

Before any meaningful remote launch:

```text
1. Add model classes and registry keys.
2. Add forward-shape tests for candidate and shuffled control.
3. Add smoke CSV with tiny samples_per_class.
4. Run CPU smoke through scripts/train.
5. Add 262144/class CSV only after smoke passes.
6. Add remote config with disk dataset cache under G:\lxy.
7. Run scripts/check-remote-readiness.
8. Commit and push before remote launch.
```

## Claim Scope

Allowed if 262144/class seed0 passes support gate:

```text
medium diagnostic evidence that fixed PRESENT P-layer message passing may add
value over the current InvP-only token-mixer route.
```

Not allowed:

```text
breakthrough
SOTA
formal route evidence
paper-ready proof
claim that SPN topology-aware networks are generally superior
```

Formal route evidence would still require at least:

```text
1000000/class
multiple seeds
same-protocol baseline and InvP anchor
true-P vs shuffled-P attribution
complete artifacts and documented gates
```

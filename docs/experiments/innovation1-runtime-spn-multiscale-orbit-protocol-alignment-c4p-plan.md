# Innovation 1 Runtime-SPN Multiscale Orbit Protocol-Alignment C4-P Plan

```text
status = completed / pass / protocol-aligned periodic orbit feasible
run_id = i1_runtime_spn_multiscale_orbit_protocol_alignment_c4p_20260726
remote = no
training_rows = 0
optimizer_steps = 0
```

## Research Question

Does the fixed `0,1,2,4,8` inverse-GF(2) orbit remain non-collapsed and
control-sensitive when it is computed from the exact two-transition runtime
windows used by the completed five-cipher joint Runtime-E4 protocol and the
matching PRESENT extension?

C4 used one homogeneous transition for PRESENT, GIFT, RECTANGLE and SKINNY,
all ten uKNIT transitions and one complete four-transition Dialga cycle. The
neural joint and whole-cipher protocols instead load two transitions per
cipher: uKNIT transitions `3,4`, Dialga transitions `2,3`, and two repeated
homogeneous transitions for the other ciphers. C4 therefore establishes a
broad structure-basis property but does not by itself prove that the exact
windows intended for C5 retain that property.

C4-P closes only this protocol-alignment gap. It does not modify or duplicate
the running C3 experiment and does not authorize C5 training before C3 is
complete.

## Frozen Structure Panel

| Cipher | Cipher rounds used by data protocol | Loaded runtime transitions | Start |
| --- | ---: | ---: | ---: |
| PRESENT-80 | 7 | 2 repeated homogeneous transitions | 0 |
| GIFT-64 | 6 | 2 repeated homogeneous transitions | 0 |
| RECTANGLE-80 | 6 | 2 repeated homogeneous transitions | 0 |
| SKINNY-64/64 | 7 | 2 repeated homogeneous transitions | 0 |
| uKNIT-BC | prefix-r5 | 2 heterogeneous transitions | 3 |
| Dialga-128 | prefix-r4 | 2 heterogeneous transitions | 2 |

All structures must be loaded from the same external JSON descriptors used by
the joint Runtime-E4 tasks. The descriptor path, window length and start are
part of the frozen config and artifact rows.

## Operator Semantics

For the loaded inverse-linear window `L`, compute:

```text
O_0 = I
O_d = L_(last-d+1)^-1 ... L_last^-1
depths = 0,1,2,4,8
```

Traversal cycles through the two loaded transitions when `d > 2`. Therefore
`O4` and `O8` are periodic topology-operator powers, analogous to fixed-hop
graph views. They are not literal recovered internal states four or eight
cipher rounds earlier. Any later neural report must preserve this wording.

## One Variable And Anchor

C4-P changes only the structure-window profile relative to C4:

```text
C4  = broad descriptor windows
C4-P = exact two-transition windows intended for the C5 neural protocol
```

Depths, corruption seed, unit-bit probe basis, distance metric, relabeling
test, controls and thresholds remain unchanged. Training data, labels and
neural weights do not exist in this audit.

## Controls

For all six ciphers:

- deterministic corrupted topology;
- no-topology identity views.

For the two windows whose linear matrices are genuinely heterogeneous:

- repeat the last transition at the same window length;
- rotate the two-transition order by one.

Homogeneous ciphers must not receive meaningless repeat/rotate attribution
votes.

## Protocol Gate

Require all of:

1. six descriptor paths, protocol rounds, starts and two-transition windows
   match the frozen config;
2. depths and semantics strings match the frozen periodic-operator contract;
3. all views are binary and full GF(2) rank;
4. depth one is bit-exact with the last loaded inverse transition;
5. correct and corrupted fingerprints are deterministic and distinct;
6. only uKNIT and Dialga receive valid heterogeneous controls;
7. rotating every cell label conjugates every orbit view exactly;
8. complete unit-bit bases and all finite metrics are used;
9. a repeated run produces the same manifest SHA-256;
10. training rows, optimizer steps and remote execution remain zero.

Protocol failure is invalid evidence and permits repair only of the failed
invariant.

## Frozen Research Gate

Every cipher must satisfy:

```text
unique exact views >= 3
new depth-2/4/8 views beyond depths 0/1 >= 1
correct-corrupted support distance >= 0.25
correct-no-topology support distance >= 0.25
```

uKNIT and Dialga must additionally satisfy:

```text
correct-repeat-last support distance >= 0.10
correct-rotated-window support distance >= 0.10
```

Pass unlocks only the C5 plan after C3 completes. Hold closes the fixed
two-transition periodic-orbit candidate; thresholds, windows and depths must
not be tuned after reading the result.

## Claim Boundary

C4-P can prove deterministic representation non-collapse, topology/control
separation, two-transition order sensitivity and cell-relabel equivariance for
the exact intended windows. It cannot prove differential signal, neural
learnability, causal partial decryption, complete uKNIT/Dialga round-schedule
consumption, unseen-cipher transfer, S-box composability, an attack, SOTA or a
breakthrough.

## Planned Artifacts

```text
outputs/local_audit/
  i1_runtime_spn_multiscale_orbit_protocol_alignment_c4p_20260726/
    results.jsonl
    validation.json
    gate.json
    summary.json
    progress.jsonl
```

No visualization is planned because exact per-cipher tables and protocol
checks are the appropriate evidence.

## Evidence-Backed Next Action

If C4-P passes, wait for C3 to close, then preregister C5 with exactly two
trained roles per seed: unchanged Runtime-E4 and a parameter-shape-matched
periodic-orbit candidate. The candidate checkpoint must be evaluated with
correct, corrupted and no-topology orbit structures; uKNIT and Dialga also
require repeat-last and rotated-window orbit controls.

If C4-P holds, do not implement C5, add depths, lengthen windows, increase
samples or launch a remote rescue. Retain C3 as the only active training route
and reassess the representation hypothesis.

## Completed Result

The frozen C4-P audit completed locally on 2026-07-26. It emitted six result
rows and nine progress events. All fourteen protocol checks and all six
research checks passed.

| Cipher | Unique views | New multihop views | Correct-corrupted | Correct-no topology | Correct-repeat-last | Correct-rotated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PRESENT-80 | 3 | 1 | 0.986139 | 0.967742 | n/a | n/a |
| GIFT-64 | 3 | 1 | 0.994106 | 0.623656 | n/a | n/a |
| RECTANGLE-80 | 5 | 3 | 0.988142 | 0.769231 | n/a | n/a |
| SKINNY-64/64 | 5 | 3 | 0.889303 | 0.966667 | n/a | n/a |
| uKNIT-BC | 5 | 3 | 0.721434 | 0.986050 | 0.689303 | 0.719209 |
| Dialga-128 | 5 | 3 | 0.932904 | 0.942609 | 0.529862 | 0.549333 |

The weakest all-cipher correct-versus-control distance was GIFT's
correct-no-topology value `0.623656`, above the frozen `0.25` threshold. The
weakest heterogeneous order control was Dialga's repeat-last distance
`0.529862`, above `0.10`. Every cipher retained at least three unique views and
at least one new view beyond depths zero and one.

```text
validation = pass
protocol checks = 14/14
research checks = 6/6
result rows = 6
training rows = 0
optimizer steps = 0
decision = innovation1_runtime_spn_multiscale_orbit_protocol_alignment_supported
```

Artifacts:

```text
outputs/local_audit/
  i1_runtime_spn_multiscale_orbit_protocol_alignment_c4p_20260726/
```

No visualization was generated. A chart would add no evidence beyond the
exact six-row gate table.

## Final Next Action

C4-P removes the window-alignment blocker but does not override C3 ordering.
Wait for the running PRESENT seed0 C3 result to complete and be locally
verified. If C3 authorizes continuation, first complete its identical seed1
replication. Only after C3 closes may C5 be frozen and implemented with:

```text
train role 1 = unchanged Runtime-E4 anchor
train role 2 = parameter-shape-matched periodic orbit candidate
orbit depths = 0,1,2,4,8
runtime windows = the exact two-transition C4-P windows
training = 2048/class/cipher, validation = 1024/class/cipher
pairs/sample = 4, epochs = 10, seeds = 0,1
negative = encrypted random plaintexts
execution = local sub-medium diagnostic
```

The candidate must share one set of view-processing parameters across all
depths. Frozen candidate checkpoints must then be evaluated with correct,
corrupted and no-topology orbit structures; uKNIT and Dialga additionally
require repeat-last and rotated-window orbit controls. Do not describe depth
four or eight as a recovered internal cipher state, and do not launch remote
scale until the local neural gate passes.

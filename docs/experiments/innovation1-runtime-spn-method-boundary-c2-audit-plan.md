# Innovation 1 Runtime-SPN Method Boundary C2 Audit Plan

```text
status = completed / audit pass / method partial
execution = local zero-training evidence audit
training rows = 0
optimizer steps = 0
remote = no
```

## Research Question

Which parts of the runtime-parameterized SPN method are supported by completed,
path-verified evidence, and which parts remain partial, contradicted, or
missing?

C2 is not another model rescue. It consolidates the completed Runtime-E4,
whole-cipher holdout, S-box identifiability, ANF-operator and topology-only
results into a requirement-by-requirement method boundary. The audit must not
turn a high local AUC or a responsive descriptor into a universal adaptation
claim.

## Frozen Method Goal

The long-term goal remains:

```text
one shared parameter geometry
+ runtime cell partition and S-box descriptors
+ runtime one-to-one or general GF(2) linear topology
-> adaptation to previously unseen SPN structures without changing backbone
   parameter shapes
```

C2 does not redefine that goal. It records how much of it the current evidence
actually proves.

## Evidence Contract

The audit reads only completed `gate.json` files. Every source path, expected
run id and SHA-256 digest is frozen in the C2 configuration. A missing file,
digest mismatch, run-id mismatch or malformed field makes the audit protocol
invalid rather than silently weakening a requirement.

| Evidence id | Role | Provenance boundary |
| --- | --- | --- |
| `runtime_r0` | shared geometry, runtime cells and exact linear operators | implementation readiness only |
| `gift_r2g_seed0/1` | repaired one-to-one GIFT topology attribution | local `2048/class`, per-cipher diagnostic |
| `present_t1_seed0/1` | one-to-one PRESENT topology attribution | local `2048/class`, per-cipher diagnostic |
| `skinny_t2a` | genuine many-source GF(2) data/operator readiness | implementation readiness only |
| `skinny_rtg3a` | two-seed general-GF(2) attribution | fallback-retrieved project-formal `1000000/class`; not paper reproduction |
| `rectangle_h1` | first whole-cipher zero-training holdout | local `2048/class/source` diagnostic |
| `uknit_a6` | heterogeneous round-window zero-training holdout | local `2048/class/source` diagnostic |
| `dialga_a8` | second whole-cipher zero-training holdout | local `2048/class/source` diagnostic |
| `sbox_s1` | frozen-checkpoint S-box responsiveness and identifiability | local zero-training audit |
| `sbox_s2` | exact ANF Boolean-operator test | local `2048/class/source` diagnostic |
| `topology_c1` | S-box-disabled exact-GF(2) topology isolation | local `2048/class/source` diagnostic |

## Requirement Matrix

Each result row has one of four statuses:

```text
supported    direct evidence satisfies the frozen requirement
partial      some required conditions pass, but scale, seeds, ciphers, or
             controls do not support the full requirement
contradicted completed evidence directly fails the current implementation's
             requirement; this is not an impossibility theorem
missing      no completed evidence tests the requirement
```

The audit evaluates exactly these requirements:

| Id | Requirement | Evidence-level pass rule |
| --- | --- | --- |
| `R1` | fixed parameter geometry | R0 says geometry is stable, runtime structure is absent from `state_dict`, and variable widths/pair shapes pass |
| `R2` | runtime cell-membership support | R0 cell relabel equivariance and variable-width contract pass |
| `R3` | exact one-to-one and general-GF(2) operator support | R0 exact inverse, gather equivalence and both linear families pass; SKINNY T2A readiness passes |
| `R4` | formal general-GF(2) topology attribution | SKINNY RTG3A is protocol-valid, `1000000/class`, two distinct passing seeds, and correct topology beats both controls by at least `0.005` |
| `R5` | one-to-one P-layer topology attribution | repaired GIFT R2G and PRESENT each have two seeds where correct topology beats corrupted and no-topology controls by at least `0.005` |
| `R6` | whole-cipher topology sensitivity | supported only when RECTANGLE and Dialga topology controls pass both seeds; partial when both holdouts show at least one passing seed; otherwise contradicted |
| `R7` | stable whole-cipher anchor retention | every preregistered whole-cipher holdout must fully pass across both seeds |
| `R8` | heterogeneous round-window support | uKNIT A6 must have `functional_pass=true` on both seeds with zero target training rows and optimizer steps |
| `R9` | S-box descriptor responsiveness | S1 must report all ten seed/cipher responsiveness checks true |
| `R10` | S-box semantic identifiability | S1 must report source and Dialga identifiability true |
| `R11` | nonlinear S-box operator composability | S2 must be protocol-valid and fully pass exact-versus-identity and exact-versus-input-permuted semantic margins on both seeds |
| `R12` | universal runtime-SPN adaptation | R1-R11 must all be supported; any contradicted composability or transfer requirement contradicts the current implementation's universal claim |

R4 uses a formal project scale, but C2 must preserve the source gate's
provenance: the joint SKINNY artifact is fallback-retrieved and is not a paper
reproduction, attack, SOTA result, breakthrough or arbitrary-SPN proof.

GIFT R2F is deliberately excluded. Its metrics predate the corrected
within-cell bit-role ordering, and the RTG1 record explicitly states that R2F
cannot adjudicate the repaired Runtime-E4 model. C2 uses the repaired, passing
R2G gates for both seeds.

## Frozen Adjudication

The audit itself passes only when:

1. every configured evidence file exists and matches its SHA-256 and run id;
2. all 12 requirement rows are emitted exactly once;
3. every row includes evidence paths, evidence digests, exact checks and a claim
   boundary;
4. training rows, target training rows and optimizer steps created by C2 are
   all zero;
5. the supported and unsupported method summaries are derived from the rows;
6. the universal claim is not marked supported while any required component is
   partial, contradicted or missing.

The expected method-level decision name is:

```text
innovation1_runtime_spn_method_boundary_frozen
```

An audit `status=pass` means the evidence boundary was synthesized correctly.
It does not mean the universal runtime-SPN goal passed. The gate must expose a
separate `method_status` and `universal_runtime_spn_supported` field.

## Artifacts

```text
outputs/local_audit/i1_runtime_spn_method_boundary_c2_20260726/
  results.jsonl
  validation.json
  gate.json
  summary.json
  progress.jsonl
```

No figure is required. Avoiding an ornamental chart keeps C2 focused on the
exact evidence table and avoids implying that unlike evidence scales are a
single comparable curve.

## Decision Routes

If R1-R5 are supported but R7-R12 are not, freeze the thesis-safe method as:

```text
runtime GF(2)-topology-aware SPN neural distinguisher
```

Do not call it:

```text
universal composable SPN neural distinguisher
```

The next research action must stay on the supported exact-GF(2) branch. It may
strengthen one-to-one formal evidence or introduce a preregistered source-
topology-diversity mechanism, but it must not reopen S-box truth-table, ANF,
DDT, inverse-triplet, Adapter, FiLM, MoE or target-supervision rescue routes.
Mechanical increases in C1/S2 samples, epochs or pairs are prohibited.

## Completed C2 Record

Run:

```text
run_id = i1_runtime_spn_method_boundary_c2_20260726
evidence files = 13/13 SHA-256 and run-id verified
requirement rows = 12/12
training rows = 0
optimizer steps = 0
remote = no
validation = pass
decision = innovation1_runtime_spn_method_boundary_frozen
method_status = partial
universal_runtime_spn_supported = false
```

The authoritative GIFT inputs are the repaired R2G seed gates, not R2F. The
audit verified the two R2G digests and retained the RTG1 statement that R2F
predates the corrected within-cell bit-role ordering.

| Id | Result | Exact evidence summary |
| --- | --- | --- |
| `R1` | supported | shared geometry, runtime descriptor outside `state_dict`, variable width/pair shapes all pass |
| `R2` | supported | runtime cell relabel equivariance and four-structure readiness pass |
| `R3` | supported | exact inverse/gather equivalence and one-to-one/general-GF(2) readiness pass |
| `R4` | supported | SKINNY `1000000/class`, two seeds, correct-minus-corrupted `+0.046029/+0.046670`, correct-minus-no-topology `+0.141366/+0.137806` |
| `R5` | supported | repaired GIFT R2G and PRESENT T1 each pass correct-versus-corrupted/no-topology on seeds 0 and 1 |
| `R6` | partial | Dialga topology sensitivity passes both seeds; RECTANGLE target controls pass seed0 but fail seed1 |
| `R7` | contradicted | RECTANGLE H1, uKNIT A6, Dialga A8 and topology-only C1 do not all retain preregistered source/target anchors |
| `R8` | contradicted | uKNIT heterogeneous-window target AUC is `0.510880/0.518289`; both functional gates fail with zero target training |
| `R9` | supported | S1 descriptor responsiveness is `10/10` seed/cipher pairs |
| `R10` | contradicted | S1 source and Dialga correct-S-box identifiability are both false |
| `R11` | contradicted | S2 is responsive but exact ANF fails identity/input-permuted semantic margins on both seeds |
| `R12` | contradicted | stable transfer, heterogeneous-window support and nonlinear composability are not all supported |

Artifacts:

```text
outputs/local_audit/i1_runtime_spn_method_boundary_c2_20260726/results.jsonl
outputs/local_audit/i1_runtime_spn_method_boundary_c2_20260726/validation.json
outputs/local_audit/i1_runtime_spn_method_boundary_c2_20260726/gate.json
outputs/local_audit/i1_runtime_spn_method_boundary_c2_20260726/summary.json
outputs/local_audit/i1_runtime_spn_method_boundary_c2_20260726/progress.jsonl
```

No visualization was generated. The evidence has incompatible scales and
provenance, so a combined curve would be more misleading than the explicit
requirement table.

## Evidence-Backed Next Action

Freeze the thesis-safe current method name as `runtime GF(2)-topology-aware SPN
neural distinguisher`. Do not use `universal composable SPN neural
distinguisher`.

The next bounded experiment is C3, a formal-scale readiness and attribution
replication for the supported one-to-one branch:

```text
question       = does repaired Runtime-E4 PRESENT-80 r7 topology attribution
                 survive the project-formal evidence floor?
local anchor   = PRESENT T1 2048/class seeds 0/1
scale anchor   = SKINNY RTG3A 1000000/class evidence contract
one variable   = train/validation sample scale only
models         = correct topology / deterministic corrupted topology /
                 no topology
train          = 2000000 total = 1000000/class
validation     = 1000000 total = 500000/class
pairs/sample   = 16, unchanged from PRESENT T1
epochs         = 5/model, unchanged from PRESENT T1
seeds          = seed0 first; seed1 only after a plan-aligned seed0 pass
execution      = remote GPU after disk-cache and pushed-commit readiness gates
advance gate   = correct AUC >= 0.520 and both control margins >= +0.005
stop gate      = any protocol/cache/SHA failure, or seed0 misses a research gate
```

C3 must preserve PRESENT T1 round, Case2 sample organization, strict encrypted
random-plaintext negatives, keys, loss, optimizer, checkpoint selection,
runtime geometry and topology corruption. Before launch it must demonstrate
parameter-matched disk-backed train/validation caches, durable progress and
resume behavior under `G:\\lxy`. A pass supports formal one-to-one topology
attribution only; it is not a Zhang/Wang reproduction, SOTA comparison,
zero-step cross-cipher transfer or universal-SPN claim.

Blocked next routes remain S-box/ANF/DDT rescue, Adapter/FiLM/MoE expansion,
target supervision, extra C1 samples/epochs/pairs and any remote run that
changes architecture together with scale.

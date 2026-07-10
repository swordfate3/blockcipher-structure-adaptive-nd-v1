# Innovation 1 SPN Literature Re-Audit

**Date:** 2026-07-10

**Scope:** SPN/PRESENT neural differential distinguishers, structure-aware
input construction, previous-round or inverse-layer features, input/network
co-design, multi-pair organization, related-key settings, and explicit
S-box/P-layer topology.

**Status:** literature route correction / no experiment launched

## Research Question

The audit asks two concrete questions:

1. Which parts of the current Innovation 1 SPN idea are already covered by
   prior work?
2. After the corrected E1 active-cell graph failure and the stopped dense-DDT
   route, what method-level gap remains defensible under the project's strict
   key-independent real-vs-random protocol?

The target project protocol remains distinct from related-key, multi-difference,
integral, and score-distribution settings:

```text
PRESENT-80
strict encrypted-random-plaintext negatives
same benchmark and same-budget controls
primary metric = validation AUC
formal evidence = at least 1000000/class and multiple seeds
```

## Search Method

### Local corpus

The local manifest contains 30 indexed papers. Ten directly relevant SPN,
PRESENT, GIFT, feature-engineering, assessment, or generic-pipeline entries were
screened first, followed by the locally available full text of the most relevant
papers.

Primary local material includes:

- `papers/innovation_one/grobid_md/an-assessment-of-differential-neural-distinguishers.md`
- `papers/innovation_one/grobid_md/a-cipher-agnostic-neural-training-pipeline-with-automated-finding-of-good-input-differences.md`
- `papers/innovation_one/grobid_md/improving-differential-neural-distinguisher-model-for-des-chaskey-and-present.md`
- `papers/innovation_one/grobid_md/a-highly-efficient-neural-distinguisher-framework-for-iot-friendly-lightweight-spn-block-ciphers.md`
- `papers/innovation_one/grobid_md/generic-partial-decryption-as-feature-engineering-for-neural-distinguishers.md`
- `papers/innovation_one/grobid_md/improved-integral-neural-distinguisher-model-for-lightweight-cipher-present.md`
- `papers/innovation_one/grobid_md/key-recovery-attack-on-present-using-an-entropy-based-neural-distinguisher.md`
- `papers/innovation_one/text/2024_sok_neural_differential_cryptanalysis.txt`

### External checks

Searches and metadata checks were run against IACR ePrint, Crossref, Semantic
Scholar, arXiv, OpenAlex, and public GitHub metadata. The raw query responses
are saved under `sources/` with the suffix `20260710`.

The targeted graph-neural searches returned:

```text
IACR query: "graph neural" cryptanalysis -> no result
arXiv exact neural-distinguisher + graph/SPN query -> 0 result
Crossref broad query -> no directly relevant S-box/P-layer graph ND found
Semantic Scholar broad query -> rate limited
OpenAlex broad query -> daily budget exhausted
```

This is evidence of non-detection, not proof that no such paper exists.

### Inclusion and exclusion

Included work had to address at least one of:

- neural differential distinguishers on PRESENT or another SPN cipher;
- structure-aware or previous-round feature engineering;
- input/network joint adaptation;
- multi-pair signal attribution;
- strict evaluation or comparability of neural distinguishers.

Side-channel GNNs, cipher-design networks, unrelated graph-learning papers, and
claims without enough protocol metadata were excluded from the route decision.

## Executive Verdict

The old novelty framing is too broad.

Already covered by prior work:

```text
inverse linear/permutation features
previous-round or partial-decryption features
S-box input/output feature construction
SPN state-matrix formatting
Conv2D over SPN states
U-Net/long-skip feature extraction
Inception and channel-attention neural distinguishers
multi-pair and related-key sample enhancement
entropy/bit-selection for PRESENT
generic cipher-agnostic training and difference search
```

Therefore, none of the following is a defensible novelty claim by itself:

```text
"use InvP(DeltaC)"
"organize PRESENT as S-box cells"
"jointly change the input and network"
"use a larger CNN/U-Net/attention model"
"use multiple ciphertext pairs"
"use previous-round S-box features"
```

The strongest defensible gap is narrower:

> A reproducible, cipher-spec-driven SPN adaptation method that separates the
> value of the representation from the value of the architecture, uses
> same-input and shuffled-structure controls, and tests whether one typed
> method transfers across SPN ciphers under comparable key-independent
> real-vs-random protocols.

No reviewed paper was found to combine all of those controls. This is a
literature gap, not yet a proven effective method.

## Evidence Matrix

| Work | Actual contribution | Setting and comparability | Collision with Innovation 1 | Verification |
|---|---|---|---|---|
| Gohr, Leander, and Neumann, *An Assessment of Differential-Neural Distinguishers* | Relates ND accuracy to distribution distance and shows multi-pair learned models often add little over score aggregation | Multiple ciphers; explicit comparability analysis | Requires aggregation baselines and warns against architecture-only claims | Local full text |
| Bellini et al., *A Cipher-Agnostic Neural Training Pipeline...* | AutoND difference search plus structure-agnostic DBitNet | Standard differential tasks over many ciphers | The project cannot claim the first generic training pipeline | Local full text; TOSC DOI verified |
| Zhang and Wang, *Improving Differential-Neural Distinguisher Model for DES, Chaskey, and PRESENT* | Multiple ciphertext pairs plus Inception-style kernels; PRESENT Case2 provides the current r7 reference | PRESENT standard real-vs-random, but group size and total-pair budgets must be tracked | Strong same-task baseline; multi-pair and Inception are prior art | Local full text; official code/checkpoint reproduced separately |
| Zhu et al., *BCS: A Neural Distinguisher Method...* | ASMCP/ASMOD S-box input/output features plus improved U-Net with long skip connections | Survey classifies PRESENT result as `12-1-A-R`; key recovery on 8-round PRESENT | Direct collision with broad "S-box feature + matching network" novelty | Crossref and Semantic Scholar verified; primary abstract plus survey detail, full text unavailable |
| Guo et al., *Improved Differential Neural Distinguishers for PRESENT and SKINNY* | CNN+MLP interaction, GA-CAM, RAdam and cyclic LR | PRESENT/SKINNY; exact detailed protocol unavailable in this audit | CNN/MLP fusion and channel attention are not new method space | Crossref and Semantic Scholar verified; abstract only |
| Lu et al., *Enhanced Neural Distinguisher Model for Efficient Differential Cryptanalysis* | Input selection from nonlinear-component positions and linear diffusion; multi-pair augmentation; skip connections and ECA | Demonstrated mainly on SIMON and key recovery | Direct collision with broad input/network joint-adaptation wording | Crossref and Semantic Scholar verified; abstract only |
| Liu et al., *A Highly Efficient Neural Distinguisher Framework for IoT-Friendly Lightweight SPN Block Ciphers* | Three inverse-operation data cases; typed `(Cbar,Cbar',DeltaCbar)` state; Conv2D over `3 x 4 x n/4` tensors | Standard real-vs-random on SKINNY and MIDORI, not PRESENT | SPN state formatting, inverse-round views, and Conv2D are prior art | Local full text; DOI `10.1587/transinf.2025EDP7070` |
| Bellini et al., *Generic Partial Decryption as Feature Engineering for Neural Distinguishers* | Automates previous-round feature extraction and separates feature gain from multi-pair gain | Simon, Simeck, Aradi; not a PRESENT benchmark | Generic partial inverse features are prior art; also shows handcrafted features can hurt | Local full text; ePrint 2025/1443 |
| Wu and Guo, *Improved Integral Neural Distinguisher Model for Lightweight Cipher PRESENT* | `InvP` and `InvS` integral representations plus MBConv-DenseNet | Integral/multiset task, not the current standard differential protocol | InvP/InvS feature construction exists, but its task is not directly comparable | Local full text; DOI `10.1186/s42400-024-00258-0` |
| Gauthier-Umana et al., *Key Recovery Attack on PRESENT Using an Entropy-Based Neural Distinguisher* | Entropy-guided bit selection and compact PRESENT network; iterative key recovery | Standard pair task and attack pipeline; lower-round efficiency focus | Bit selection and compact architecture are occupied space | Local full text; DOI `10.1007/s00521-026-11973-9` |
| Ge and Wang, *Improved Related-Key Differential Neural Distinguishers for SPN Block Ciphers* | Invertible-SPN feature enhancement plus related-key sample reuse | Related-key SKINNY/PRESENT; PRESENT four-pair r7/r8/r9 reported as 95.6/72.0/53.7% | Direct collision with InvP/inverse-layer feature novelty, but not directly comparable to the strict single-key task | IACR ePrint 2026/535 page and abstract verified |
| Nguyen et al., *Related-Key Multi-Pair Neural Distinguishers...* | PCA/silhouette signal analysis; explains multi-pair gains as variance reduction and identifies round-dependent signal ceilings | Related-key multi-pair on PRESENT, SIMECK, LEA, HIGHT | Strong negative pressure against claiming learned pair interactions without an aggregation control | IACR ePrint 2026/748 page and abstract verified |
| Nguyen et al., *Improved Neural Distinguisher for PRESENT-80 Using Inception and Efficient Channel Attention...* | Inception, residual connections, ECA, structured related-key multi-pair input | Related-key multi-pair; reports distinguishing ability to deeper rounds | Inception/ECA/multi-pair combination is prior art but not comparable to strict single-key evidence | Crossref, Semantic Scholar, and public GitHub repository metadata verified |

## Main Corrections to the Existing Blueprint

### 1. InvP is an anchor, not the complete novelty

The project's two-seed 1M/class InvP result remains the strongest local
structure-attribution evidence. Literature, however, already contains inverse
permutation, inverse linear layer, inverse S-box, and partial-decryption feature
routes. The valid local claim is therefore:

```text
InvP/P-layer alignment is supported under the project's strict protocol.
```

The invalid claim is:

```text
The project invented SPN inverse-layer feature enhancement.
```

### 2. Input/network joint adaptation is already crowded

BCS, the Liu SPN framework, the 2025 enhanced model, and the PRESENT/SKINNY
attention model all co-design input formats and architectures. The thesis title
can still use "joint adaptation", but the method must be more specific than
that phrase.

### 3. Multi-pair improvement needs a frozen aggregation baseline

The 2022 assessment and ePrint 2026/748 converge on the same warning: more pairs
amplify weak signal through aggregation and variance reduction. A multi-pair
network is not a new structural learner merely because its accuracy increases
with pair count.

Required controls are:

```text
single-pair score aggregation
pair-order permutation
same total pair budget
same validation distribution
same negative definition
```

### 4. The graph gap remains unoccupied but empirically unsupported here

The targeted searches did not find a directly matching S-box/P-layer graph
neural differential distinguisher. That preserves a possible novelty gap.

It does not justify continuing the current E1 implementation. Corrected E1-R1
failed true/shuffled/metadata controls, and the earlier r7 topology-aware model
also failed to beat InvP-only and shuffled-P controls. Novelty and empirical
merit are separate gates.

### 5. DDT literature does not reopen the stopped dense-DDT route

The reviewed literature supports previous-round features and, in some settings,
S-box-derived features. It does not invalidate the project's completed
real-source versus wrong-source/constant-source DDT mismatch failures. Dense DDT
trail values remain stopped unless a genuinely new source-attribution
hypothesis is approved.

## Ranked Research Directions

The scores combine novelty fit, compatibility with current evidence,
falsifiability, and implementation risk. They are route-selection scores, not
claims of verified novelty.

| Rank | Direction | Score | Decision |
|---:|---|---:|---|
| 1 | Controlled SPN component-adaptation benchmark: typed previous-round representation, same-input baseline, shuffled structure, deterministic/aggregation controls | 8.5/10 | Highest defensibility; can yield a positive or rigorous negative contribution |
| 2 | Cipher-spec-generated typed adapter with shared operators and explicit cross-SPN transfer testing | 7.0/10 | Genuine method gap, but current PRESENT topology and GIFT medium diagnostics make it high risk |
| 3 | Literature-aligned BCS/Liu competitor reproduction under the project's strict protocol | 6.5/10 | Necessary comparator and route calibration; not itself novel |
| 4 | Multiple-instance/Deep-Set pair learner over a frozen single-pair scorer | 5.5/10 | Literature-motivated, but ePrint 2026/748 and local pair-pooling results demand a very strict aggregation gate |
| 5 | Continue the current E1 active-cell graph by increasing samples | 2.0/10 | Do not proceed; corrected controls failed |
| 6 | Reopen dense DDT trail input | 1.0/10 | Do not proceed without an explicit new hypothesis |

## Recommended Next Research Gate

Do not treat the Zhang/Wang seed1 baseline as the next innovation experiment;
it is optional evidence completion. Do not scale E1.

Before designing another network, establish a literature-aligned competitor
gate with no DDT input:

```text
fixed task:
  PRESENT-80 r7 Zhang/Wang Case2
  strict encrypted-random-plaintext negatives
  16 pairs/sample

rows:
  1. completed InvP-only anchor
  2. same typed key-independent previous-round input with a small Conv2D/ResNet
  3. only if the exact prior-art recipe is available: BCS/Liu-style matched architecture

controls:
  same input and budget
  shuffled inverse-linear mapping
  DeltaC-only
  fixed aggregation baseline where applicable
```

This gate first answers whether the current project is comparing against the
right structure-aware competitor. A new method should be proposed only after
that comparator is understood. If the prior-art-shaped model does not beat
InvP-only, the thesis should emphasize controlled structural attribution and
cross-protocol benchmarking rather than another architecture claim.

## Limitations

- The BCS, PRESENT/SKINNY GA-CAM, enhanced IoT, VCRIS Inception/ECA, and two
  2026 ePrint papers were verified from authoritative metadata and abstracts,
  but full text was not available for all of them in this environment.
- Semantic Scholar broad search was rate-limited; OpenAlex search had no daily
  budget. Exact-paper Semantic Scholar checks succeeded for several key works.
- IACR and arXiv graph searches found no directly matching work, but database
  indexing and terminology variation prevent an exhaustive non-existence claim.
- Reported round counts and accuracies from related-key, integral, multi-pair,
  and standard real-vs-random settings are not directly comparable.

## Primary Links

- Gohr et al. assessment: <https://eprint.iacr.org/2022/1521>
- AutoND/DBitNet: <https://doi.org/10.46586/tosc.v2023.i3.184-212>
- Zhang/Wang PRESENT model: <https://arxiv.org/abs/2204.06341>
- BCS: <https://doi.org/10.1088/1402-4896/adae63>
- PRESENT/SKINNY GA-CAM: <https://doi.org/10.1088/1402-4896/add29c>
- Enhanced IoT model: <https://doi.org/10.1109/JIOT.2025.3566051>
- Liu SPN framework: <https://doi.org/10.1587/transinf.2025EDP7070>
- GPD: <https://eprint.iacr.org/2025/1443>
- PRESENT integral ND: <https://doi.org/10.1186/s42400-024-00258-0>
- PRESENT entropy ND: <https://doi.org/10.1007/s00521-026-11973-9>
- Related-key SPN feature enhancement: <https://eprint.iacr.org/2026/535>
- Related-key multi-pair signal analysis: <https://eprint.iacr.org/2026/748>
- PRESENT Inception/ECA: <https://doi.org/10.1109/VCRIS68011.2025.11250567>

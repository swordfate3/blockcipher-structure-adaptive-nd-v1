# 创新2 E58：ATM原生PySAT key-monomial见证provider复现计划

日期：2026-07-18

状态：计划冻结 / 待执行

## 1. 研究问题

E56确认公开ATM九轮basis能提供470个“独立轮密钥下constant generalized relation”正类；E57
确认直接precursor标量求值至少需要`2^60`明文，不能用于生成常数或负类。

E58只回答一个问题：作者原生PySAT投影模型能否在不枚举precursor明文的情况下，输出一个可
复验的非零key-monomial，从而证明某个候选relation在**独立轮密钥模型**下严格key-dependent。

它不声称得到actual PRESENT-80 master-key witness，不改变原五轮linear-mask balance任务，也不
训练神经网络。

## 2. 同预算锚点与唯一变量

```text
source anchor       = ATM commit b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
positive anchor     = E56 public 470-relation constant subspace
complexity anchor   = E57 minimum scalar cost 2^60 plaintexts/relation/key
one changed variable= scalar plaintext enumeration -> native projected SAT model
key model           = independent 64-bit round keys, explicitly not PRESENT-80 schedule
device              = local CPU only
remote              = no
training            = no
```

不得同时修改relation语义、bit order、round split、negative定义或公开正类集合。

## 3. 论文到代码契约

```text
Paper Algorithm 1 / Remark 3
  -> Modelling/Search.py::search_integral_properties
  -> Tools/AvecImplementations.py::Avec_unified_model_with_partial_trail_counting_constant

exact key dependence / constant coefficient
  -> Modelling/Trails.py::is_key_dependent
  -> Modelling/Trails.py::get_key_independent_sum
  -> Modelling/Trails.py::get_sum

projected model enumeration
  -> Tools/SATmodelling.py::enum_projected_models
  -> pysat.solvers.Glucose4
```

冻结源码审计已发现两项必须修复或隔离的问题：

```text
get_sum: `if k in V` compares tuple with integer set; corrected adapter must use `km in V`
limited functions: cap exhaustion returns True/1; strict witness path must map exhaustion to unknown
```

不直接修改`/tmp`官方clone；项目侧写最小adapter并保留commit/hash来源。

当前Python 3.13 / OR-Tools 9.15不能pickle作者QMC multiprocessing任务中的`CpModelProto`。
项目adapter仅把prime-implicant与set-cover约束生成改为同算法单进程执行，CP-SAT固定单worker；
官方源码、truth table、目标函数和CNF语义不改，并由256项S-box全真值门约束正确性。

## 4. Phase A：环境与精确语义校准

在`/tmp`隔离环境安装作者冻结requirements并编译`bitarrays`扩展。依赖或编译失败属于provider
环境结果，不得改用GLPK部分枚举冒充成功。

校准顺序：

```text
A0. PRESENT S-box 256个(u,v)系数与直接ATM/ANF真值逐项一致
A1. 无key与单key toy函数的constant/key-dependent分类全枚举一致
A2. adapter返回具体非零key exponent mask与odd-parity trail count
A3. 同一witness独立重放仍为odd；错误mask或无key fixture为even/absent
A4. cap触发明确返回unknown，不得返回strict negative
```

这里是mechanism reproduction，不是论文九轮规模复现。

复现命令（要求冻结ATM clone已在`/tmp/innovation2-e56-atm`）：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python \
  python-sat pybind11 galois ortools

# 按官方 bitarrays/README.md 编译与当前Python ABI匹配的扩展后：
PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/audit-innovation2-atm-native-sat-provider \
  --run-id i2_present_atm_native_sat_provider_phase_a_20260718 \
  --output-root outputs/local_audits/i2_present_atm_native_sat_provider_phase_a_20260718 \
  --atm-root /tmp/innovation2-e56-atm \
  --paper-text papers/innovation_two/text/2023_beyne_verbauwhede_algebraic_transition_matrices.txt \
  --mode audit
```

## 5. Phase B：冻结九轮单候选硬cap探针

Phase A全部通过后，才从公开正relation构造一个边际匹配的确定性单坐标mutation。不得用
`not in basis`直接标负类。

```text
candidate count      = 1
wall-clock cap       = 60 seconds
projected-key cap    = 2^16 candidates
per-key trail cap    = 2^20 models
accepted negative    = explicit nonzero key exponent + exact odd parity across every coordinate
timeout/cap outcome  = unknown
```

若单坐标先产生witness，必须在relation全部坐标上重算同一key exponent并做GF(2) XOR；未检查
取消不能形成relation-level负类。

## 6. 推进与停止门

```text
frozen source and dependency build                    pass
S-box and toy exact fixtures all match                pass
strict witness includes nonzero key exponent          pass
independent replay remains odd                        pass
cap exhaustion is unknown                             pass
nine-round relation-level witness within hard cap     pass
```

若前五项失败：修复adapter或关闭provider，不执行九轮。若只有最后一项失败：记录九轮原生SAT
provider在冻结cap内不可执行，不提高cap、不转远程GPU、不生成训练数据。

只有最后一项通过，才允许扩大到最多`32`个边际匹配候选，目标是获得至少`16`个严格负类做
标签宽度审计；这仍不是神经训练门。

## 7. 下一网络门（仅预注册）

只有独立轮密钥任务得到至少`256`个去重正类、`256`个严格负类，relation-disjoint拆分和
size/input-weight/output-weight边际控制全部通过，才开放三行同预算本地矩阵：

```text
0. deterministic relation-size/weight/marginal baseline
1. coordinate-set DeepSets
2. Relation-Cipher Cross-Attention
```

候选2唯一创新变量是relation token对SPN typed graph的query-conditioned交互；必须超过
deterministic、DeepSets、label-shuffle和wrong-P-layer控制。标签门之前不实现或训练这些网络。

## 8. Phase A实际结果

权威run：

```text
i2_present_atm_native_sat_provider_phase_a_20260718
```

隔离依赖和官方`bitarrays`扩展构建通过。作者QMC在Python 3.13下因OR-Tools对象不能跨进程
pickle而失败；项目adapter采用同算法单进程constraint生成后，正式校准结果为：

```text
PRESENT S-box ATM coefficients = 256/256 exact match
nonzero coefficients           = 90
maximum SAT models/query       = 1
toy F_k(x)=x XOR k witness     = key exponent 0x1, odd parity
independent replay             = exact odd
constant x coefficient         = no key witness
forced low cap                 = unknown
```

来源commit、Glucose4、官方扩展、论文Remark 3和独立轮密钥模型检查全部通过。没有运行九轮、
没有生成训练标签，也没有使用远程GPU。

```text
status   = pass
decision = innovation2_atm_native_sat_mechanism_ready_for_r9_probe
training = no
remote   = no
```

推荐下一步按第5节只运行一个冻结九轮mutation硬cap探针。它必须返回relation全部坐标GF(2)
合并后的非零key exponent与odd parity；单坐标witness、timeout或cap都不能标strict negative。

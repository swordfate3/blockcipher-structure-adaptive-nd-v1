# 创新2 E103：PRESENT S-box 4-bit三轮真实ATM runner兼容性门

日期：2026-07-20

状态：预注册 / 待环境构建与执行

## 1. 研究问题

E102证明route-owned runner在确定性`Avec` fixture上能逐候选原子落盘并正确恢复，但fixture没有调用
官方构模、PySAT、bitarray扩展、Manager cache或真实`Avec`。E103回答：

```text
在一个严格资源封顶的真实ATM小任务上，route-owned runner是否与官方Search.py得到同一GF(2)
关系空间，并能承载官方构模、真实Avec、Manager cache、多进程增量返回与中断恢复？
```

E103对象是PRESENT的4-bit S-box切片：状态只有一个nibble，每轮使用真实PRESENT S-box和独立4-bit
轮密钥，三轮split为`(1,1,1)`，线性层是4-bit恒等连接。它不是64-bit PRESENT轮函数，不包含
P-layer或PRESENT-80 key schedule，不能作为PRESENT三轮、九轮区分器或攻击证据。

## 2. 冻结来源与环境

```text
source E102 gate = 91188f35d5c2fa24c26fcefb4cde1045356f60121ba0e4f5a32c7eef3fa4b738
E102 decision    = innovation2_present_atm_resumable_runner_fixture_passed
ATM commit       = b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
Search.py        = 5d9a5c117d7f0940473c15dded0e3243dc06a6182fe584733e0badc8928f459d
Avec source      = c0a9dd19f4c92e8ceeccde01b6e13de9599e05f2531ffb0c3ce7c31a9aacccae
Components.py    = 28e17729b5c74ee752c2c105b92af952c6215d34d6e4b248eff9d3fc77b06caf
CompoundFunction = b64fe879e707ac89d5eb8a68face50f143ff2f53f636e469b821a35c43fbdee6
IteratedCipher   = 3bfaa5a9f76cb78b604884028b06c87ae643a2dd1c8a8c53fffcf5adc1a73bdb
Trails.py        = 9852c1facd19b35b5edcef2165dab7c43873d25e1fdfa1e7c0999e3f98634346
```

预检已确认当前`uv`解释器为CPython 3.13.12，冻结clone没有已编译`bitarrays.bitset`，直接import
返回`ModuleNotFoundError`。这不是科学失败，而是官方README声明的手工构建前置条件。E103必须使用
冻结`bitarrays/src/bitset.cpp`和当前`pybind11` include编译与当前`EXT_SUFFIX`匹配的扩展，记录：

```text
Python version / cache_tag / EXT_SUFFIX
g++ version and complete build command
bitset.cpp / bitset.hpp hashes
compiled extension path / sha256 / file type
import smoke result
galois / ortools / python-sat / pybind11 / numpy versions
```

模型构造统一调用项目已验证的single-process QMC兼容shim，规避Python 3.13下OR-Tools变量跨进程
pickle失败；官方anchor与candidate共享同一shim和同一构模对象，因此唯一比较变量仍是搜索调度与
持久化。

## 3. 同预算协议

```text
cipher slice = one PRESENT S-box, 4 input bits, 4 output bits
rounds/split = 3 / (1,1,1)
key model = independent 4-bit key addition at each round boundary, matching ATM construction style
limit = 2^6
anchor = frozen official search_integral_properties, 2 worker processes
candidate = route-owned resumable runner, 2 worker processes
controlled interruption = after first parent-persisted candidate, then resume with identical parameters
hard wall-clock cap = 180 seconds for the formal command
device = local CPU
training/seeds/epochs = not applicable
```

两条路线必须共享完全相同的模型、候选规则和真实`Avec`函数，但使用独立Manager cache和调用计数，
防止anchor预热candidate。多进程可能预取尚未落盘的候选；中断后只要求已原子完成的候选不再调用，
未完成或只在worker中计算但未返回父进程的候选允许重算并必须明确计数。

## 4. 比较与裁决门

不同nullspace实现可能输出不同基底，因此“完全相等”定义为同一GF(2)关系空间，而不是要求pickle
字节或基底排列相同。必须在两边关系支撑坐标的并集上验证：

```text
relation-space rank equal
span(anchor) subset span(candidate)
span(candidate) subset span(anchor)
singleton relation sets equal
candidate result is deterministic and checksum-valid
```

`real_atm_compatibility_passed`还要求：

- 所有冻结源码hash、E102 gate、Python ABI和依赖版本记录通过；
- bitset扩展从冻结C++源构建、hash记录且真实import通过；
- 官方anchor和runner均在180秒硬cap内完成且无异常；
- 两者GF(2)关系空间与singleton集合相等；
- runner使用2个worker，真实Manager cache和oracle call list有非零活动；
- 受控中断、resume事件、至少一个候选复用均可证；
- 已完成候选调用次数不增加；预取重算单独报告，不能伪装成恢复失败；
- runner的metadata/started/progress/candidate/result/complete与hash全部通过。

通过只允许另立E104“R9 `(3,3,3)`单split生成计划”，其中必须先完成目标机器相同ABI构建、
36线程资源/磁盘路径审计、逐候选缓存容量估计、超时/取消/回收和本地monitor设计。R10九个split、
并行多split和神经训练继续关闭。

`environment_incompatible`：扩展构建/import、QMC shim、PySAT或Manager失败。只修可复现环境。

`runner_real_oracle_mismatch`：官方与runner关系空间不同、真实多进程失败或恢复不复用。只修runner，
不启动R9。

`resource_cap_hit`：180秒内小任务未完成。停止当前生成升级，重新缩小兼容fixture或收束为E99证据。

## 5. 计划产物

```text
run_id = i2_present_sbox4_r3_real_atm_runner_compatibility_20260720
output = outputs/local_audits/i2_present_sbox4_r3_real_atm_runner_compatibility_20260720/
```

产物包括`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`、
`environment_contract.csv`、`relation_space.csv`、`oracle_calls.csv`、`source_hashes.json`、
`official_anchor/`、`resumed_runner/`、`curves.svg`和`visual_qa_passed.marker`。完成后刷新最近结果索引，
更新本文件正式结果并给出唯一下一动作。

## 6. 正式结果

待执行。

# 创新2 E104：PRESENT R9 ATM `(3,3,3)`单split可恢复生成

日期：2026-07-20

状态：远程 Phase C `stage_001` 运行中 / 最终关系尚未回收

## 1. 研究问题

E103证明route-owned runner能承载真实ATM构模、`Avec`、bitset、PySAT、Manager cache和两进程恢复，
但只用了4-bit PRESENT S-box切片且最终关系空间为空。E104回答：

```text
冻结公开notebook声明但未公开结果的PRESENT九轮split (3,3,3)，能否在目标CPU工作站上按官方
independent-round-key协议安全、可取消、可恢复地生成，并形成新的source-heldout复核关系集？
```

E104不训练神经网络，不使用GPU加速，不生成negative。完成结果只能称
`locally generated confirmation set`，不是论文公开结果、独立publication、PRESENT-80区分器或攻击。

## 2. 冻结来源与协议

```text
source E103 gate = 8a651da1f16824482ae7653229757d2a3dfde25772a6ded8babe69691f5116e3
E103 decision    = innovation2_present_sbox4_real_atm_compatibility_passed
ATM commit       = b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
Search.py hash   = 5d9a5c117d7f0940473c15dded0e3243dc06a6182fe584733e0badc8928f459d
cipher           = full 64-bit PRESENT round function, 16 PRESENT S-boxes + published P-layer
rounds/split     = 9 / (3,3,3)
limit            = 2^10
threads          = 36
key model        = independent 64-bit round keys, exactly matching ATM notebook construction
target public gap= R9 (3,3,3) only
```

不得把该key model写成PRESENT-80 master-key schedule。不得顺带运行R10、其他R9 split、随机候选、
神经模型或参数扫描。

## 3. 远程目标与已知事实

目标机只读预检：

```text
host              = lxy-a6000 / DESKTOP-BBLPACJ
root              = G:\lxy
free space        = 4,921,998,364,672 bytes (~4.92 TB)
base python       = F:\Anaconda\envs\DWT\torch310\python.exe / Python 3.10.20
extension suffix  = .cp310-win_amd64.pyd
compiler          = MSVC cl.exe 19.36.32535
GPU               = not used
```

`G:\lxy\blockcipher-structure-adaptive-nd`含大量历史源码改动，禁止reset或用于E104；
`G:\lxy\blockcipher-structure-adaptive-nd-v1-clean`干净但detached且提交旧，只作只读参考。正式运行使用：

```text
RUN_ID   = i2_present_r9_atm_split333_resumable_generation_20260720
RUN_ROOT = G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%
SOURCE   = %RUN_ROOT%\source
ATM_ROOT = %RUN_ROOT%\atm-source
VENV     = %RUN_ROOT%\venv
OUTPUT   = %RUN_ROOT%\results
LOGS     = %RUN_ROOT%\logs
```

代码从GitHub已推送`main`创建run-owned clean clone；ATM clone必须checkout冻结commit。项目venv使用
Python 3.10.20创建在`G:\lxy`内，不修改共享`torch310`依赖。项目文件、pip cache、编译产物、
日志、临时文件和结果全部留在`G:\lxy`。

## 4. Windows环境与构建契约

E104新增Windows MSVC bitset构建路径，必须记录：

- Python version/cache tag/`EXT_SUFFIX`与`Include`/`libs`；
- `cl.exe`完整版本、`INCLUDE`和`LIB`可用性；
- 冻结`bitset.cpp`/`bitset.hpp` hash；
- `/O2 /LD /std:c++20 /EHsc /DNDEBUG`构建命令；
- `.cp310-win_amd64.pyd` path/hash/size与真实import；
- 固定依赖版本、`pip freeze`和安装日志；
- single-process QMC兼容shim与16个PRESENT S-box转移模型复用。

目标机 Git 全局配置为 `core.autocrlf=true`。冻结来源门因此以冻结 commit 的 Git blob
计算 SHA-256，并单独要求 tracked worktree 干净；Windows 工作树中的 bitset 源码另记录原始
SHA-256，并将 CRLF 规范化为仓库 LF 后复核冻结 SHA-256。该处理只消除平台换行差异，任何
冻结 blob、commit 或 tracked worktree 内容漂移仍会关闭来源门。

依赖锁必须至少包括`numpy`、`pybind11`、`ortools`、`python-sat`和`galois`，且在run-owned venv中
完成import smoke。网络安装失败是环境hold，不得回退到共享环境脏装或把文件写到`C:\Users`。

## 5. 三阶段执行

### 5.1 Phase A：目标机就绪

只运行构建与模型dry-run：

```text
clean source commit = 实现提交的GitHub HEAD
ATM source hashes = 全部冻结匹配
bitset build/import = pass
64-bit PRESENT round function = 16 S-boxes + exact P-layer
split model = f1/f2/f3 each 3 rounds, 64-bit state
model hash/clauses/key bits = recorded
Manager IPC + 2-worker startup = pass
disk free >= 500 GB
```

### 5.2 Phase B：一个真实候选的可复用探针

使用正式36-worker配置启动同一OUTPUT，但`interrupt_after_new_candidates=1`。墙钟硬上限10分钟：

- 若父进程收到一个完整候选，必须已经有checksummed candidate JSON、progress和probe marker；
- 记录候选`(u,v)`、耗时、文件大小、外层调用、内部oracle activity与Manager cache sizes；
- 随后以同参数resume smoke验证该候选不重算，再在再次产生新候选前受控退出；
- 多进程预取但未落盘的候选允许重算，必须透明计数；
- 若10分钟内无完整候选，关闭自动长跑并裁决`candidate_boundary_too_coarse`。

探针文件本身是正式run的可复用缓存，不另起数据分布或候选集。

### 5.3 Phase C：单split分阶段长跑

Phase A/B全部通过后自动启动唯一`(3,3,3)`搜索。每阶段最多12小时，最多3阶段（累计36小时），
每阶段使用同一参数hash和OUTPUT恢复。超时时由run-owned supervisor终止整个子进程树，写stage timeout
marker；下一阶段必须先验证candidate/result hash再恢复。完成时最后写`result.json`和
`complete.marker`，再生成安全关系转换、GF(2) rank、维数、stats和artifact manifest。

历史8个公开R9 split耗时0.75至6.61小时、oracle calls最高220,841,578,547，仅作参考，不能承诺
`(3,3,3)`墙钟。36小时仍未完成则裁决资源上限，不自动无限续跑。

## 6. 远程启动与监控

生成脚本必须位于项目`source\scripts\generated\remote`，使用`cmd.exe /c`且不得出现`cmd.exe /k`；
任何启用delayed expansion的脚本不得含`!`。正式长任务由Task Scheduler启动，launcher和run日志都在
RUN_ROOT。启动命令返回后只做一次有界确认：run目录、started marker、环境/模型readiness或
`progress.jsonl`至少一项存在。

随后启动本地tmux watcher，负责：

```text
等待远程 complete / hold / timeout / failure marker；
低频检查monitor健康，不在主线程SSH轮询；
完成或阶段裁决后从 G:/lxy/blockcipher-structure-adaptive-nd-runs 拉回原始产物；
校验Git提交、参数hash、结果hash、来源hash与manifest；
写入 outputs/remote_results 或 remote_results_incomplete；
触发文档、可视化、visual-qa-redraw和最近结果索引更新。
```

## 7. 冻结裁决门

`split333_generation_passed`要求：

```text
Phase A/B全部通过；
正式source与ATM commit/hash匹配；
36-worker可恢复runner按同一parameter hash完成；
result/complete/stats/manifest原子且hash通过；
公开8个split未被覆盖或混入；
生成关系可安全解析、规范化、GF(2) rank重放；
至少一次fresh-process resume确实复用已完成候选；
远程结果已检索到本地并通过plan-alignment gate。
```

通过后只开放E105：把新`(3,3,3)`关系作为source-heldout确认集，冻结E99模型与阈值做零重训排序
评估，并与公开8-split训练/测试证据分开。不得立即训练新模型或宣称九轮攻击。

`environment_hold`：clean clone、venv、MSVC、bitset、依赖、64-bit构模或IPC失败。只修Phase A。

`candidate_boundary_too_coarse`：首候选10分钟未完成或无法原子缓存。需要候选内部子任务恢复设计，
不启动Phase C。

`resource_cap_hit`：36小时/3阶段仍未完成。停止自动续跑，保留可恢复产物并评估CPU资源或算法优化。

`protocol_invalid`：来源、split、key model、候选、limit、线程、hash或结果转换漂移。作废并修协议。

## 8. 计划产物

远程产物：

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_present_r9_atm_split333_resumable_generation_20260720\
```

本地检索目标：

```text
outputs/remote_results/i2_present_r9_atm_split333_resumable_generation_20260720/
```

产物至少包括环境/源码/模型manifest、started/progress/stage markers、candidate cache manifest、
`result.json`、`gate.json`、`summary.json`、`results.jsonl`、stats、关系空间审计、日志与最终可视化。

## 9. 正式结果

待执行。

### 9.1 Phase A 启动记录

2026-07-20 首次到达真实 readiness 后暴露两个环境问题，均发生在候选探针和长搜索之前：

```text
attempt source = de97bab8dd44eb4a91730cb7f3d7496f8efd50b9
ATM commit     = b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
torch import   = failed through eager innovation2 package import
source gate    = failed because Windows core.autocrlf changed LF bytes to CRLF
MSVC build     = failed because Task Scheduler had cl.exe but no Windows SDK INCLUDE/LIB
probe/search   = not started
```

延迟导入修复 `8073a8f` 与 Windows 来源哈希修复 `e5411d8` 已推送。下一次启动必须同步这些
提交；setup 还必须显式调用已验证存在的 Visual Studio 2022 `vcvars64.bat`，使 `math.h` 等
Windows SDK 头文件对 MSVC 可见。通过 Git blob 来源校验、Phase A 环境门和两个10分钟候选
探针后，才允许进入九轮搜索。

连续重启还暴露了 bootstrap 与 run-owned source 各自追踪 `origin/main` 的竞态：run-owned source
每次落后启动脚本一个提交，使刚修复的环境逻辑没有进入实际进程。后续 setup 必须读取 bootstrap
clone 的精确 HEAD，将 run-owned source 以 `--ff-only` 快进到同一提交，并在 readiness 前验证两者
相同；只比较两个独立的 `origin/main` 状态不足以证明计划对齐。

调度器审计还确认原任务使用 `Interactive` principal，`schtasks /Run` 虽返回成功，但
`LastTaskResult=0x41303`且`LastRunTime`未更新，实际没有启动探针。目标机近期成功的项目任务使用
`SYSTEM / ServiceAccount`。E104 因此改用 SYSTEM 调度，并将 `HOME`、`USERPROFILE`、`TEMP`、`TMP`、
Git global config、known_hosts 与 pip cache 全部固定到 run-owned `G:\lxy` 目录；GitHub 访问继续只
引用预先存在且 SYSTEM ACL 已允许的固定 SSH key。

SYSTEM 首次真实执行后，watcher 回收的 raw fallback 显示一个实例已在精确提交 `8c26d27` 上通过
Phase A，随后重叠的 setup 实例删除了 success marker，并把前一实例生成的 ATM `__pycache__`
判为 dirty。该记录位于：

```text
outputs/remote_results_incomplete/i2_present_r9_atm_split333_resumable_generation_20260720/
```

后续 pipeline 必须在清理 marker 前原子获取 run-owned `pipeline.lock`，重复实例直接返回且不得写
同一候选缓存；ATM 本地 exclude 还必须覆盖 `**/__pycache__/`与`*.pyc`。只有单一 owner 可以进入
Phase A/B/C。

首个真实 `probe_001` 在 `0.84s` 后失败，日志仍是 MSVC 找不到 `math.h`。Phase A 本身已经通过；
根因是 `setup.cmd` 的 `setlocal` 在返回 pipeline 时恢复了环境，导致后续 probe/search 丢失
`INCLUDE/LIB`，同时也丢失 run-owned `HOME/TEMP`。因此 pipeline owner 必须在 Phase A 返回后
再次调用 `vcvars64.bat`，并独立设置 run-owned `HOME/USERPROFILE/TEMP/TMP`。该失败没有执行真实
候选 oracle，也不是 `candidate_boundary_too_coarse` 裁决。

2026-07-20 20:14，本地 tmux watcher 观察到当前 SYSTEM 任务同时存在：

```text
probe_001_done.marker
probe_001_passed.marker
probe_002_done.marker
probe_002_passed.marker
stage_001_started.marker
Task Scheduler state = Running
```

这证明两个 fresh-process 探针门均已通过并自动开放 Phase C；当前是36-worker的首个12小时搜索
阶段。详细 probe JSON、候选坐标、调用数与耗时尚未回收到本地，故本记录暂不填推测值，也不宣称
九轮关系已经生成。watcher 将在 terminal marker 后自动进行 raw fallback 回收。

`stage_001`运行后又暴露了多进程恢复粒度问题：第一层4096个候选中，961个候选已原子落盘，
随后候选数长期不变，但5分钟累计CPU仍增加约1500至2125秒。原runner使用有序`Pool.imap`；
输入顺序靠前的重候选会阻塞父进程接收后续已完成结果，使其只留在内存，阶段超时时无法恢复。
该问题不改变密码学结果，但削弱逐候选持久化契约。修复要求改用`imap_unordered`，让任意worker
完成的候选立即校验并原子写盘；最终层处理仍按冻结候选顺序执行，关系结果保持确定性。

有序返回修复提交`6e2a06e`推送后，旧任务在确认零个匹配run id的Python worker后受控停止；
`pipeline.lock`重命名为带时间戳的中断证据，961个候选、metadata和progress全部保留。bootstrap与
run-owned source随后均对齐到`6e2a06e`。恢复探针成功原子写入第962个候选，但门槛暂时返回hold：

```text
candidate_call_sum       = 38
new_durable_candidates   = 1
candidate_files_after    = 962
internal cache sizes     = [19, 0, 34, 16, 0, 29]
internal oracle_call_sum = 0
```

这是`imap_unordered`首完成候选触发受控中断后，pool终止其余worker造成的内部统计汇总竞态；它不
表示候选未经过真实ATM oracle。探针门因此要求候选级oracle调用非零，并且内部调用计数或共享缓存
活动至少一项非零，同时继续要求恰好新增一个checksummed候选和已有缓存复用。该修复不改变split、
轮数、SAT模型、候选集、参数hash或关系语义。修复推送并重新通过两个恢复探针后，才重新开放
`stage_001`；现有962个候选继续复用。

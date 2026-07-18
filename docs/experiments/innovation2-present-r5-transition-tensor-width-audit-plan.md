# 创新2 E54：PRESENT五轮GF(2) transition tensor变量消元宽度审计计划

日期：2026-07-18

状态：计划冻结 / 待实现

## 1. 研究问题

E53-A建立了完整PRESENT-80一、二轮exact ANF oracle；E53-B证明Sage/GLPK逐解blocking在低
复杂度S-box query上精确，但最重`v=15` query需要枚举1792个raw term choices，在冻结10秒内
无法完成。当前环境也没有PySAT、CryptoMiniSat、Z3、BDD或model counter。

E54不再逐解计数，而是把每个S-box、XOR、COPY、permutation和key-schedule关系表示为GF(2)
局部transition tensor，通过变量消元直接计算全局trail parity。第一门只审计因子图宽度与内存
上界，避免实现后才发现五轮收缩不可行。

E54不训练网络，不使用远程GPU，不执行五轮标签池。

## 2. 固定语义

```text
cipher             = PRESENT-80
target rounds      = 5
input variables    = 64 plaintext + 80 master-key exponent bits
positive semantics = full cube coefficient polynomial对key/inactive变量恒为0
local coefficients = exact GF(2) monomial-transition parity，不是trail existence
only change        = per-solution GLPK blocking -> exact factor/tensor contraction
```

## 3. Phase A：因子图与宽度审计

逐轮构造PRESENT真实DAG，包括round-key XOR、16个4-bit S-box、P-layer、80-bit key schedule和
final whitening。每个局部factor必须先与E53-A S-box transition和一轮exact ANF fixture对拍。

至少比较三种确定性消元顺序：

```text
min-fill
min-degree
query-aware min-fill（目标output/cube变量最后消元）
```

每轮报告：变量数、factor数、primal graph边数、induced width、最大中间factor变量数、预计
`2^width`表项、按1 byte/bit-packed两种估计的峰值内存。随机顺序只作压力对照，不参与选择。

## 4. 推进门

```text
PRESENT bit order / key schedule / round convention pass
local S-box tensor == E53-A exact 256-entry transition parity
r1 tensor result == E53-A exact ANF fixture
r2 selected fixture == E53-A exact ANF fixture
best deterministic order max factor variables <= 26
estimated dense peak <= 4 GiB
order与宽度计算可复现
```

全部通过后，下一实验才实现实际tensor contraction，并先重放一、二轮全部fixture；再通过才允许
五轮固定`16x64`子集。若五轮宽度超过门槛，停止当前全变量tensor路线，不通过稀疏结果猜测、
更长timeout、远程GPU或经验标签绕过。

## 5. 控制与停止线

```text
control 1 = E53-A exact ANF oracle
control 2 = E53-B existence-only与GLPK blocking结果
control 3 = 故意错误P-layer，必须被r1/r2 fixture识别
control 4 = key schedule删除/独立轮密钥替代，必须被协议门识别
```

宽度门失败时，五轮严格标签路线冻结为“当前开放provider不可执行”；创新2保留四轮方法学结果
与确定性/神经负结果边界。不得因此重新开启四轮网络枚举。

## 6. 产物

```text
outputs/local_audits/i2_present_r5_transition_tensor_width_audit_20260718/

factor_manifest.json
elimination_orders.jsonl
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

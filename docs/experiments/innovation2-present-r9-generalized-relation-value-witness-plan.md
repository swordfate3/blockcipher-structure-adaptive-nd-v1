# 创新2 E57：PRESENT九轮广义relation precursor语义与标量witness边界审计计划

日期：2026-07-18

状态：已完成 / hold

## 1. 研究问题

E56确认ATM公开九轮basis有470个去重正relation，但没有严格负类或常数0/1。最初E57计划把
坐标`(u,v)`错误解释为普通monomial `x^u * E_K(x)^v`，由此估计每坐标只需
`2^(64-wt(u))=2--16`个明文，并尝试真实PRESENT-80直接求值。

论文Definition 7、Theorem 6和Algorithm 1实际规定输入坐标使用precursor basis：

```text
pi_u(x) = 1_{Prec(u)}(x)
Prec(u) = {x: x <= u}
relation coordinate = pi_u(x) * mu_v(E_K(x))
```

因此标量求和需要枚举`|Prec(u)|=2^wt(u)`个明文。E57修正为只审计这个语义与数据复杂度边界，
不执行任何大规模加密，不把错误monomial运行解释成密码学不稳定。

## 2. 冻结来源

```text
ATM commit       = b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
relations        = 8份R9 basis的去重union
rounds           = 9
scalar row cap   = 2^24 plaintext evaluations / relation-key
device           = local CPU audit only
training         = no
remote GPU       = no
```

外部pickle继续使用restricted unpickler。E57不安装PySAT/OR-Tools、不运行ATM九轮搜索器。

## 3. 审计指标

对每个去重relation报告：

```text
relation size
每坐标 wt(u), wt(v)
correct precursor plaintexts = sum_{(u,v) in R} 2^wt(u)
wrong monomial plaintexts    = sum_{(u,v) in R} 2^(64-wt(u))
correct / wrong complexity ratio
```

全局报告input weight直方图、relation-size直方图、最小/中位/最大precursor cost，以及用两颗
master key构造一个negative witness的最小标量工作量下界。

## 4. 语义控制

```text
precursor support size for u     = 2^wt(u)
ordinary monomial support size   = 2^(64-wt(u))
pi_u is not x^u                  pass
all input exponents are 64-bit   pass
all output exponents are nonzero pass
```

此前错误`x^u` evaluator得到`0/470`跨key稳定，只能记录为`wrong_basis_diagnostic_rejected`；
不得据此声称ATM relation不适用于PRESENT-80，也不得后验尝试bit reflection。

## 5. 推进门

```text
470 relations and frozen source valid          pass
precursor semantics matches paper/source       pass
maximum scalar plaintext cost <= 2^24          pass
two-key negative witness lower bound <= 2^25   pass
```

若复杂度门失败：

```text
decision = innovation2_present_r9_generalized_relation_scalar_witness_infeasible
training = no
```

这只关闭“直接枚举明文求常数/负类”的E57方案，不证明relation不存在。下一步只能是获得可执行的
algebraic/SAT constant与key-dependence provider，或关闭广义relation监督路线；不能转远程GPU机械
枚举`2^60`数据。

## 6. 产物

```text
outputs/local_audits/i2_present_r9_generalized_relation_precursor_boundary_20260718/

relation_costs.jsonl
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

## 7. 2026-07-18实际结果

权威run：

```text
i2_present_r9_generalized_relation_precursor_boundary_20260718
```

470个去重relation共含693个坐标。来源commit、64-bit指数范围、非零输出指数和precursor基语义
检查全部通过；没有执行标量加密、神经训练或远程任务。

```text
input wt(u) histogram                 = 60:3, 61:40, 62:98, 63:552
relation size histogram               = 1:305, 2:136, 4:29
minimum precursor plaintexts/key      = 1152921504606846976 = 2^60
median precursor plaintexts/key       = 9223372036854775808 = 2^63
maximum precursor plaintexts/key      = 36893488147419103232 = 2^65
minimum two-key witness plaintexts    = 2305843009213693952 = 2^61
frozen local scalar cap               = 16777216 = 2^24
wrong ordinary-monomial estimate      = 2--32 plaintexts/relation-key
```

因此最便宜的单relation、单key直接求值也比冻结cap高`2^36`倍，最便宜的双key负类见证需要
`2^61`次明文求值。此前错误basis运行的`0/470`没有密码学解释力，已从结果证据链排除。

```text
status   = hold
decision = innovation2_present_r9_generalized_relation_scalar_witness_infeasible
training = no
remote   = no
```

## 8. 推荐下一步

下一研究问题是：现有ATM/3SDP/SAT工具中，是否存在无需展开`2^60`个precursor明文、又能输出
relation常数或actual PRESENT-80 master-key依赖见证的可执行provider。

```text
same-budget anchor = E56公开470 relation + E57 precursor复杂度边界
required controls  = source commit、precursor基、round/bit convention、actual key schedule
one variable       = provider后端/表示，不改变标签、拆分或negative定义
scale              = 先用1--2轮exact fixture，再做单个九轮relation硬cap审计
execution          = 本地CPU/现有开源求解器；不启动远程GPU
advance gate       = exact fixture全匹配，并在硬cap内给出至少一个可复验constant或key witness
stop gate          = 只给SAT可达性、只支持独立轮密钥、超cap或不能输出见证
```

通过前不训练DeepSets/Cross-Attention，不把`not in basis`当负类，不按文件ID拆分，不扩大模型。
若没有provider通过，就关闭ATM广义relation监督路线，保留四轮严格标签方法学和五/九轮provider
边界作为创新2的可写结论。

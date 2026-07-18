# ATM原生provider论文到代码映射与缺口

日期：2026-07-18

证据来源：

```text
paper = papers/innovation_two/pdf/2023_beyne_verbauwhede_algebraic_transition_matrices.pdf
text  = papers/innovation_two/text/2023_beyne_verbauwhede_algebraic_transition_matrices.txt
code  = /tmp/innovation2-e56-atm
commit= b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
```

本记录只覆盖E58机制复现，不声称完整复现论文全部算法、九轮运行规模或Table 1。

| 论文内容 | 官方代码 | 当前状态 | E58验证 |
|---|---|---|---|
| precursor basis `pi_u=1[x<=u]` | `Modelling/Trails.py`的input exponent assumptions | 语义已由E57纠正 | S-box/toy fixture |
| Algorithm 1 candidate filtering与kernel | `Modelling/Search.py::search_integral_properties` | 公开九轮pickle已验证 | 不重新搜索470维basis |
| Remark 3 constant relation oracle | `Avec_unified_model_with_partial_trail_counting_constant` | 证明constant，未知0/1值 | 保留为正类契约 |
| exact key dependence | `is_key_dependent` | 返回bool，无证书序列化 | adapter返回key exponent与odd parity |
| key polynomial support | `get_sum` | 存在tuple/int成员检查错误 | adapter修正并加回归测试 |
| projected SAT enumeration | `enum_projected_models` + Glucose4 | 当前项目环境缺`python-sat` | `/tmp`隔离依赖构建 |
| QMC并行CNF生成 | `LogicOptimisation/QMC.py` | Python 3.13无法pickle OR-Tools model vars | 同算法单进程shim + 256项真值门 |
| cap语义 | `*_limited` | cap时返回dependent/1，不能作严格证据 | cap统一映射`unknown` |
| PRESENT轮函数 | `Ciphers/PRESENT/PRESENT.ipynb` | S-box/P-layer一致 | bit-order fixture |
| key模型 | `construct_iterated_cipher` fresh local key vars | 独立轮密钥，不是80-bit schedule | 结果字段强制标注 |
| 论文Table 1 | 40-core、45--396分钟、百万到千亿oracle calls | 未复现 | 仅60秒单候选探针 |

## 当前实现缺口

```text
[x] 冻结依赖可安装，bitarrays可编译
[x] adapter修复get_sum集合类型错误
[x] exact witness包含key exponent、坐标parity和重放记录
[x] S-box 256项与toy key-dependence全真值校准
[ ] 九轮单relation候选在硬cap内给出witness或unknown
[ ] actual PRESENT-80 key schedule provider（E58明确不实现）
[ ] paper-scale 40-core九轮搜索复现（E58明确不实现）
```

## 可解释边界

非零key monomial只证明候选对作者的独立轮密钥变量存在代数依赖。它不能自动迁移成actual
PRESENT-80 master-key schedule下的负类；后者需要把80-bit schedule纳入模型或给出具体master
key见证。反过来，公开独立轮密钥下constant正类在round/bit约定一致时可以限制到actual schedule。

# 创新2 E31：确定性积分标签提供者契约审计计划

日期：2026-07-18

状态：Phase A与Phase B均完成 / hold / 当前标签语义不匹配

## 1. 研究问题

E30的32个随机线性子空间没有非平凡joint kernel；E15的P-layer轨道、E16--E19的
inactive-context路线和E27-N的SPECK位置拓扑族也都未形成可训练标签。E31不再改变
网络，而是审计确定性工具能否提供与创新2当前目标完全一致的标签：

```text
input  = affine input set X + linear output mask u + cipher + rounds
label  = XOR_{x in X} <u, E_K^r(x)> 是否为0，并说明对哪些key成立
```

只比较两个有完整公开代码且已核对论文的提供者：

```text
A = CLAASP-MP / monomial prediction
B = Algebraic Transition Matrices / generalized integral search
```

E31不是神经训练，也不把“工具找到了某种广义积分关系”自动视为当前标签就绪。

## 2. 冻结版本

```text
CLAASP repository:
  https://github.com/Crypto-TII/claasp
  commit = f2239d639ae5c4a013947ce9121c6f4464584758

AlgebraicTransitionMatrices repository:
  https://github.com/michielverbauwhede/AlgebraicTransitionMatrices
  commit = b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
```

外部仓库只在 `/tmp` 做只读审计，不复制其GPL/CC代码进入本项目。项目只保存版本、
协议、统计和自主实现的解析/验证代码。

## 3. Phase A 已确认事实

CLAASP公开commit包含PRESENT-80模型和monomial prediction模块，但模块直接导入
`gurobipy`，docstring明确要求Gurobi license，相关测试全部skip。当前本机：

```text
sage      = available
gurobipy  = unavailable
```

ATM公开commit使用`pybind11/ortools/python-sat/galois/numpy`，含PRESENT 9轮notebook
和8份预计算pickle。pickle只含builtin set/frozenset/tuple/int；Phase B仍必须使用
拒绝任意class加载的restricted unpickler。

静态解析得到：

```text
files                                  = 8
planned notebook splits                = 9
missing precomputed split              = (3,3,3)
unique serialized basis elements       = 470
GF(2) rank of their union               = 468
support coordinates                    = 673
singleton standard-basis elements      = 305
```

论文报告PRESENT r9总空间dimension为470，但当前公开8文件的union rank是468。该差异
未解释前，不得声称本地完整复现论文470维空间。

更重要的语义差异：ATM返回
`sum_x r'(x,F(x)) = constant independent of key`，Remark 3说明搜索本身不确定
constant是0还是1；output exponent重量大于1时还是高阶输出单项式，不是线性mask。

## 4. Phase B 可执行审计

实现一个只读解析器，输入外部ATM `Results/*.pkl` 根目录，输出：

```text
results.jsonl
summary.json
gate.json
metadata.json
progress.jsonl
```

逐提供者检查以下契约字段：

| 字段 | 必须回答的问题 |
|---|---|
| cipher/round/key model | 是否确实是PRESENT-80 keyed r9；独立轮密钥还是实际key schedule |
| input set | 是否能还原成明确的affine set；维度、offset和bit order是什么 |
| output target | 是否为线性mask，还是高阶单项式/多个关系的线性组合 |
| label value | 是确定为0、确定为1，还是只知道key-independent constant |
| key scope | 全密钥证明、独立轮密钥模型，还是有限密钥经验标签 |
| negative label | 未找到是否真能解释为negative，还是仅为搜索不完备 |
| reproducibility | 当前环境是否能在无隐藏license条件下执行最小fixture |

ATM解析器的同预算控制：

1. 分别报告8个split，不把不同basis简单相加当维数；
2. GF(2)计算union rank并回代全部basis；
3. 区分singleton与multi-term relation；
4. 按`wt(input exponent)`和`wt(output exponent)`分层；
5. 只把output weight=1的singleton列为“线性输出候选”；
6. 未知constant和未找到候选不得强行转成0/1标签。

## 5. 裁决门

```text
deterministic_provider_ready:
  至少一个provider可在当前可用环境执行；
  输入集合、bit order、linear output mask和key scope全部明确；
  label明确区分XOR=0与XOR=1；
  negative语义是证明或明确的完备搜索结果；
  至少8个已知正fixture和8个已知负fixture复现。
  -> E32构建PRESENT高轮structure x mask标签atlas；仍先做捷径门。

provider_semantics_mismatch:
  只能得到未知constant、高阶monomial、多项广义关系或不完备“未找到”。
  -> 不作为当前二分类标签；可单列广义积分关系预测扩展，不混写。

provider_runtime_unavailable:
  唯一同目标provider要求当前不可用的商业license/solver。
  -> 不安装后就宣称可复现；评估开放替代或小状态精确标签路线。

protocol_invalid:
  版本、pickle安全、bit order、GF(2)rank或论文映射不一致。
  -> 只修审计，不解释标签宽度。
```

## 6. 执行位置与规模

Phase B只解析8个小型预计算文件并运行GF(2)审计，属于本地CPU任务，不使用远程GPU，
不训练网络，epochs/seeds不适用。外部工具最小fixture若需要新增依赖，必须隔离在
`/tmp`或独立容器，不修改项目主依赖；Gurobi license不作为默认可用条件。

权威run规划：

```text
outputs/local_audits/i2_present_r9_deterministic_provider_contract_20260718/
```

## 7. 停止边界

```text
不把ATM未知constant写成balanced=1；
不把output monomial degree>1写成linear output mask；
不把multi-term generalized relation拆成独立正样本；
不把“未出现在找到的basis”写成确定负类；
不因GraphGPS/AllSet较新就绕过标签门训练；
不在E31安装或使用未授权Gurobi license。
```

## 8. E31后的网络路线

provider通过时，E32先审计标签宽度、fresh-key/证明范围和边际捷径；通过后才比较
deterministic baseline、small GraphGPS与SCGT。两个provider都不满足当前语义时，
下一方案是生成可完整枚举的小状态SPN exact labels，以S-box/P-layer/topology-disjoint
方式拆分，并把PRESENT/GIFT作为跨密码迁移测试；该路线必须另立实验计划，不能在
E31中偷偷改变target。

## 9. Phase B 完整结果

权威run：

```text
i2_present_r9_deterministic_provider_contract_20260718
```

实际输入版本与冻结commit一致；8份pickle全部通过restricted unpickler的builtin
`set/frozenset/tuple/int`形状验证。CLAASP与ATM来源文件、PRESENT模型和预计算文件集
检查全部通过。

ATM重算结果：

```text
result files                         = 8
unique serialized basis elements    = 470
union GF(2) rank                     = 468
support coordinates                 = 673
standard-basis members              = 305
linear-output standard members      = 198
multi-term serialized elements      = 165
```

singleton输入指数重量分布为 `60:1, 61:40, 62:66, 63:198`；输出指数重量分布为
`1:198, 2:66, 3:40, 4:1`。这复核了预审统计，但不改变语义：ATM constant-search只证明
常数跨key不变，未给出它是0还是1；未出现在找到的subspace也不是完备负类。

CLAASP-MP存在selected-output-bit接口，但当前环境没有`gurobipy`，模块和测试都明确
要求Gurobi license。因此两个provider均未通过当前目标的完整契约：

```text
claasp_provider_ready = false
atm_provider_ready    = false
at_least_one_ready    = false

status       = hold
decision     = innovation2_deterministic_provider_semantics_mismatch
training     = no
remote_scale = no
```

可视化 `curves.svg` 已按SVG原生尺寸对应的 `1446x761` 像素执行
`visual-qa-redraw`。第一次检查发现“不满足/未核验”色阶对调，已修复并重新渲染；最终
标题、数值条形图、逐字段契约矩阵、图例、裁决和证据范围无重叠、裁切或颜色歧义。

## 10. 最终推荐

不使用305个singleton直接训练，也不把198个线性输出候选标成`balanced=1`。下一门
E32应保持当前目标语义不变，先生成可完整枚举的小状态SPN精确标签，并检查：

```text
positive/negative label width
fresh-key exact agreement
position/mask marginal shortcut
cipher/topology-disjoint split feasibility
```

只有E32标签门通过，才实现并训练架构排名文档中的small GraphGPS与SCGT。CLAASP-MP
和ATM保留为确定性控制/候选提供者，不作为未经复现的ground truth。

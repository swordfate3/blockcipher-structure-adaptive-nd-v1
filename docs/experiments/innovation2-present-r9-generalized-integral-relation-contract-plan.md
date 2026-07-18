# 创新2 E56：PRESENT九轮广义积分关系输出预测标签契约审计计划

日期：2026-07-18

状态：计划冻结 / 待执行

## 1. 研究问题

E52--E55已经关闭当前PRESENT-80五轮all-key/all-offset linear-mask balance的开放exact provider
家族。E31曾确认Algebraic Transition Matrices（ATM）公开结果含470个序列化basis元素、305个
standard-basis成员和198个linear-output成员，但这些结果不能直接标成`balanced=1`。

E56不偷偷把旧target换成容易二分类的任务，而是单独审计一个扩展契约是否成立：

```text
input  = cipher/key model + rounds + generalized relation R
R      = {(input exponent u, output monomial exponent v), ...}
target = sum_{(u,v) in R} sum_x x^u F_K(x)^v 是否与key无关
```

该target预测的是“广义积分关系membership”，不是常数具体为0/1，也不是原任务的指定linear
output mask是否XOR平衡。只有契约、标签和拆分都过门，才允许把它作为创新2的明确扩展任务并
重新开放最小神经矩阵。

E56不训练网络、不使用远程GPU、不生成经验标签。

## 2. 冻结来源

```text
repository = https://github.com/michielverbauwhede/AlgebraicTransitionMatrices
commit     = b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
cipher     = PRESENT round function
rounds     = 9
files      = 8 public R9-complex-oracle pickle files
missing    = split (3,3,3)
```

外部仓库仅在`/tmp`只读解析，不复制其代码或pickle进项目。pickle继续使用拒绝global class的
restricted unpickler。

## 3. 必须分开的三种结论

```text
A. generalized relation membership
   basis/span中的R是否已证明其和对该key model保持常数

B. zero-valued balance
   上述常数是否已证明等于0

C. original innovation2 target
   指定affine input set与linear output mask在真实PRESENT-80 master-key schedule下是否XOR=0
```

A成立不能推出B；B在独立轮密钥模型成立也不能自动推出C。输出weight大于1的坐标是高阶输出
monomial，multi-term basis是若干坐标的线性关系，也不能拆成多个单独正类。

## 4. 审计内容

### 4.1 cipher与key model

结构化解析`PRESENT.ipynb`和`Construction/IteratedCipher.py`：记录P-layer、9轮拆分、每轮
`key_masks`、局部key变量创建方式，以及是否存在80-bit master key和PRESENT key schedule。
独立轮密钥模型必须明确标记，不能只因round function是PRESENT就写成PRESENT-80标签。

### 4.2 relation正类与常数语义

安全加载8份basis，重算每文件维度、union rank、坐标数、singleton/multi-term、输入/输出
exponent重量分布。报告basis元素是序列化基向量，不把其任意线性组合机械膨胀成独立样本。

### 4.3 negative与拆分

检查公开结果是否包含key-dependent witness、完备complement或negative certificates。统计8文件
之间的basis交集、共同singleton和独有multi-term，判断按文件拆分是否会把相同relation泄漏到
train/validation。`not in found basis`若搜索不完备，只能是unknown。

### 4.4 原任务映射

只把singleton且`wt(v)=1`列为linear-output relation候选，检查其输入指数是否能唯一还原为项目
当前8-bit affine cube及任意inactive offset；检查常数值、bit order和真实master-key schedule。
任何不可逆字段都必须列为映射缺失。

## 5. 推进门

```text
source commit / safe pickle / 8 files             pass
actual PRESENT-80 master-key schedule              pass
generalized positive membership semantics          pass
at least 256 deduplicated positive relations       pass
at least 256 proven key-dependent negatives        pass
negative witnesses independently replayable        pass
train/validation relation-disjoint split feasible  pass
strong size/weight/file marginal controls defined  pass
mapping to original linear-mask balance explicit   pass or extension_only
```

全部通过且原任务映射成立：构造原任务标签atlas。只有`extension_only`通过：可另立“广义积分关系
预测”扩展，但论文和实验名必须与linear-mask balance分开；首个神经矩阵仍需本地小规模。

key model、negative或relation-disjoint任一关键门失败：

```text
decision = innovation2_generalized_relation_label_contract_not_ready
training = no
```

不得用random unknown当严格负类，不通过one-class accuracy、文件ID或relation长度捷径开训。

## 6. 若标签通过后的最小神经矩阵

只预留，不在E56执行：

```text
0. relation size/degree/file-frequency deterministic baseline
1. coordinate-set DeepSets（关系集合不变基线）
2. Relation-Cipher Cross-Attention（u/v query与SPN typed graph交互）
```

候选2的创新变量只允许是“广义relation token与cipher graph的query-conditioned交互”；同预算、
同split、同seed、同epoch，并要求超过强边际与label-shuffle。当前E44 pair-state、MSPN、CGPR
不会自动重开。

## 7. 产物

```text
outputs/local_audits/i2_present_r9_generalized_integral_relation_contract_20260718/

source_contract.json
relation_overlap.json
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

## 8. 2026-07-18实际结果

权威run：

```text
i2_present_r9_generalized_integral_relation_contract_20260718
```

冻结commit、8份pickle、安全builtin形状、PRESENT S-box/P-layer、九轮split和局部key连接检查
全部通过。第一次机器运行因检查器错误要求notebook显式写`assert sum(split)==9`而被判
`protocol_invalid`；该判断没有解释数据。检查器改为AST解析实际`splits`并逐个验证和为9后，
正式重跑得到以下有效结果。

### 8.1 relation宽度与重叠

```text
serialized unique basis elements = 470
union GF(2) rank                 = 468
standard-basis relations        = 305
linear-output standard relations= 198
relation size histogram         = 1:305, 2:136, 4:29
relations common to all 8 files = 316
relations unique to one file    = 24
pairwise intersection           = 316..418
proven key-dependent negatives  = 0
negative witnesses              = 0
```

因此正类membership语义和数量存在，但按公开文件拆分会把至少316个完全相同relation同时放入
train/validation。`not in basis`仍是unknown，不能填充严格负类。

### 8.2 key model与原目标映射

`PRESENT.ipynb`对每个round segment调用`construct_iterated_cipher`，后者按64-bit `key_masks`
创建局部key变量；源码没有80-bit master key或PRESENT key schedule。该模型证明的正relation
对更强的独立轮密钥空间保持常数，但公开结果不提供常数是0还是1，也不提供actual master-key
下key-dependent negative witness。

198个linear-output候选的input exponent重量仍为`60--63`；公开basis没有项目当前8-bit affine
cube、任意inactive offset和linear-mask XOR标签的完整可逆映射。因此不能把它们改名成原创新2
五轮平衡标签。

```text
status   = hold
decision = innovation2_generalized_relation_label_contract_not_ready
training = no
remote   = no
```

最终`curves.svg`按`visual-qa-redraw`渲染为`2261x1165`像素。首次预览的key-model说明仍为
英文内部文本，已改为中文；重绘后标题、三栏图、relation数量、文件重叠、逐项契约、裁决和
证据范围无重叠、裁切、缺字或歧义，并记录`visual_qa_passed.marker`。

## 9. 推荐下一步

不直接训练DeepSets或Relation-Cipher Cross-Attention，不用random unknown补负类，不按文件ID
拆分，也不把独立轮密钥关系称为PRESENT-80 balance标签。

执行E57 precursor语义与标量witness边界门。论文Definition 7和Algorithm 1中的输入基函数是
`pi_u=1_{x<=u}`，不是普通monomial `x^u`。当前`wt(u)=60--63`，所以直接求和实际需要
`|Prec(u)|=2^wt(u)=2^60--2^63`个明文。E57应：

```text
1. 机器核验470个relation的precursor/plain-monomial复杂度；
2. 报告单relation-key最小/中位/最大明文数；
3. 与冻结2^24本地标量cap比较；
4. 若失败，关闭直接标量求常数/negative witness，不执行大规模加密；
5. 仅在有可执行algebraic/SAT provider时才重新讨论广义relation监督标签。
```

此前按`x^u`错误实现得到`0/470`稳定，只是wrong-basis诊断，不能解释ATM relation或后验挑bit
mapping。标量边界失败时，广义relation神经路线继续关闭。

E57现已完成。470个relation的最小/中位/最大precursor标量成本分别为`2^60/2^63/2^65`
明文/单key，最小双key witness成本为`2^61`，远超冻结`2^24` cap。因此直接标量常数与
negative-witness路线关闭；下一步只允许审计可执行的algebraic/SAT provider，不转远程枚举，
也不开放神经矩阵。

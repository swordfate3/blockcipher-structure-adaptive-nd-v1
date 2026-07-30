# uKNIT-BC 线性代数积分区分路线

日期：2026-07-30

## 1. 文献与方法定位

权威来源为 Hwang 等人的 IACR ePrint 2026/340（2026-06-05 修订版）：
*Improving Neural-Inspired Integral Distinguishers via a Linear-Algebraic
Approach*。论文不训练一个真假分类器，而是把多个独立积分试验的密文输出异或结果
写成 GF(2) 矩阵，并直接求右核空间。

对固定明文多重集结构 `S` 和输出特征映射 `Phi`，一次独立试验产生：

```text
P(K,S) = XOR_{x in S} Phi(E_K(x))
```

把 `m` 次新密钥/新多重集试验的 parity vector 叠成矩阵 `M`：

```text
M in GF(2)^(m x d)
B_emp = ker(M) = {u : M u = 0}
```

核中的非零向量 `u` 表示一个候选输出 mask：对发现数据中的所有试验，mask 选中
的输出特征异或恒为零。真正对所有目标密钥与结构实例成立的平衡空间只满足
`B_true subseteq B_emp`，所以发现核不能直接当成证明，必须用完全独立的验证矩阵
`M_tilde` 检查。若随机置换下每个伪候选每次以 `1/2` 概率偶然通过，发现核维数
为 `k`、独立验证行数为 `v`，论文给出的全候选后选择误接受上界为
`2^(k-v)`（这里每个验证多重集只配一把新密钥）。

论文先使用原始输出 bit（64维），再在 AES-like 4-bit cell 密码上扩展到每 cell
16 个 Boolean monomial 的 256维 VDS。uKNIT 首先使用64维线性 parity，因为它最
便于解释、验证和与随机矩阵对照；只有线性核路线完成后才考虑256维 cell-VDS。

## 2. 与项目既有 uKNIT 实验的区别

既有 uKNIT Innovation 1 任务输入若干选择明文差分对，标签为真实差分样本或严格
随机负样本，并用神经网络输出 AUC。本文路线的数据单位完全不同：

```text
一个随机非活动上下文 + 一个活动 cell 的16个取值
  -> 同一把密钥下加密16个明文
  -> 16个密文的64-bit逐位异或
  -> 一条 parity matrix 行
```

标签不是正负样本，AUC 也不是首要指标。主要量是 GF(2) rank、nullity、核 basis、
basis 在新密钥/新上下文上的存活情况和随机输出控制。因此本路线单独归入创新2的
积分输出性质研究，不改变 Innovation 1 的差分 benchmark。

## 3. uKNIT 轮边界

项目实现的 uKNIT-BC 为64-bit状态、16个4-bit cell、128-bit密钥，最多12轮。
`r1-r11` 的 prefix-reduced 轮均执行：

```text
AddRoundKey -> round-dependent S-box -> round-dependent GF(2) linear layer
```

完整12轮另执行末轮 `AddRoundKey -> S-box -> AddRoundKey`。首轮轮数普查只到 r11，
避免把完整加密末轮语义和 prefix transition round 混在同一裁决中；若 r11 仍有
稳定核，再单独审查 r12。

## 4. 分阶段降轮区分研究路线

阶段A扫描单活动 cell：每个积分集合含16个明文，遍历16个输入位置，在 fresh
keys 和 fresh inactive contexts 上构造64维 parity matrix。r1 必须得到全零矩阵
和64维核，作为位序、枚举、异或与轮函数的实现校准；目标轮为 r3-r11。

阶段B只在阶段A没有高轮信号时扩大为相邻双活动 cell：每个集合含256个明文，
保持相同输出特征和独立验证协议。阶段B改变的唯一变量是积分集合维数。

阶段C只对阶段A/B独立验证后仍存在的最高轮候选执行论文级采样：至少1000个发现
试验和1000个新验证试验，并用第二随机种子复现。随后才讨论256维 cell-VDS、
增加前轮/后轮的 partial-sum 密钥恢复，或把核 mask 用作神经网络特征。

这里报告的“支持轮数”首先指存在经验支持的 key-independent integral
distinguisher。它不自动等于完整密钥恢复结论；后者必须另行给出猜测子密钥位数、
数据、时间和内存复杂度。

## 5. 首轮 uKNIT 证据与路线更新

2026-07-30 已完成两级64-bit线性 parity 普查。单活动 cell 在 r4 有6/16个位置
保留非零空间，r5-r11全部满秩；拓扑双 cell 在 r4 有24/24个组内 pair保留非零
空间，但 r5-r8同样全部满秩。因此当前 sampled-key 支持轮数为 r4，增加第二个
活动 cell 没有延长轮数。

下一优先级不是重复增加同一种线性矩阵的密钥数量，而是采用论文在 Midori 上
证明有增量的256维 cell-VDS。为避免16个常数单项式和有限行数制造伪核，uKNIT
实现应把常数坐标单独报告、只在240个非恒定坐标上裁决，并冻结4个已审查结构，
使用至少512+512个独立试验。VDS若仍不能在 r5 产生非恒定关系，再评估三活动
cell的4096明文集合；两种变量不得在同一实验中同时改变。

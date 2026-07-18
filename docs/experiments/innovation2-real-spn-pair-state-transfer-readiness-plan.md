# 创新2 E42：真实SPN pair-state输出性质迁移readiness计划

日期：2026-07-18

状态：计划冻结 / 待标签库与64-bit模型契约审计 / 禁止训练

## 1. 研究问题

E39和E41已经在扩展16-bit合成SPN族上确认结构条件化有向pair-state能利用正确P-layer；
E40没有证明triangle更新是必要增量。迁移到真实64-bit SPN前，必须先回答：

```text
现有PRESENT/SKINNY真实密码产物是否提供足够多、fresh-key稳定、
不被简单边际解释的“结构 + 输出mask -> kernel membership”标签，
并且64x64 pair-state能否在可控内存下保持同一等变和拓扑契约？
```

本阶段只做readiness审计，不训练神经网络。

## 2. 权威证据锚点

合成方法锚点：

```text
E39 triangle mean dual / fair control = 0.716651 / 0.664140
E40 local mean dual / ID              = 0.702438 / 0.684393
E41 local mean dual / fair control    = 0.702438 / 0.583771
```

真实密码标签锚点必须直接读取以下本地产物，不能从文档摘要手工重建：

```text
PRESENT E11b 7轮论文四维kernel复现
PRESENT E12  活动块kernel多样性
PRESENT E16  固定context kernel多样性
PRESENT E17c context/mask group-disjoint捷径审计
PRESENT E18  fresh-key context稳定性
PRESENT E19  跨密钥平衡概率interaction

SKINNY E20  7轮论文18维kernel复现
SKINNY E21  8轮论文一维kernel复现
SKINNY E22--E24 geometry多样性与闭合审计
```

已知历史结论是PRESENT E18 fresh-key不稳定、E17c存在捷径、E19 interaction不足；SKINNY
稳定非平凡几何数量也未过标签宽度门。E42必须重新从JSON/CSV验证这些事实，而不是假设
标签已经可训练。

## 3. 标签库readiness契约

一条候选标签必须统一为：

```text
cipher topology + rounds + active structure + fixed context + output linear mask
target = output mask是否属于互斥密钥半共同验证的经验kernel
```

不得把单把密钥parity、未校正平衡率、结构/随机二分类标签或论文固定mask身份混入同一目标。

某个真实SPN标签族只有全部通过才可训练：

```text
independent structure groups             >= 32
nontrivial joint-kernel structures       >= 8
distinct fresh-key-stable signatures     >= 4
positive label prevalence                in [0.10, 0.90]
discovery labels reproduced on fresh keys>= 0.90
geometry/context group-disjoint split    可构造且每split正负类均存在
mask-only / position-only / context-only strongest AUC <= 0.65
direct GF(2) kernel、bit order、key halves与scalar parity协议全部通过
```

这些是训练readiness下限，不是论文性能门。不得通过复制同一structure的更多mask、key或行数
伪造独立结构组数量。

## 4. 64-bit pair-state模型契约

只做小batch前向/反向和内存测量，不训练标签：

```text
pair tensor shape        = batch x 64 x 64 x hidden
cell size                = 4 bit
topology input           = 64-bit P-layer permutation
structure/mask input     = 64-bit multi-hot
processor candidates     = pair-local / triangle
absolute bit/cell IDs    = none
cell relabel error       <= 1e-6
true/corrupted logit Δ   >= 1e-5
local off-pair influence = 0.0
```

对`hidden=16/32/64`、`batch=1/2/4/8`记录参数量、峰值RSS或CUDA allocated bytes、一次前向和
一次反向耗时。readiness要求至少`hidden32,batch4`在本地安全完成，估算的远程训练配置保留
`>=25%`显存余量。若triangle内存不足，只关闭triangle，不得把local一起判死。

## 5. 训练开放条件与最小矩阵

只有第3节至少一个真实SPN标签族通过且第4节模型契约通过，才另建训练计划。首个训练矩阵
最多四行：

```text
1. strongest train-only marginal/ID baseline
2. pair-local true topology seed0
3. triangle true topology seed0
4. 最强神经候选的fair-corrupted P-layer control seed0
```

固定同一label、group-disjoint split、epoch、batch、checkpoint metric和参数预算。readiness
规模只允许本地小训练；只有真实拓扑候选超过所有边际且领先公平错误拓扑，才计划seed1或
远程中等实验。

## 6. Stop与禁止路线

任一情况都停止训练开放：

```text
没有标签族达到32个独立结构组
fresh-key复现率不足0.90
简单边际AUC高于0.65
group-disjoint split缺正类或负类
64-bit最小模型前向/反向或等变契约失败
```

停止后下一步是构造或复现更宽的确定性真实结构候选提供者，不是增加context、mask、key、
epoch或远程GPU。不得把PRESENT E16的旧context标签、SKINNY少量稳定几何或合成SPN AUC直接
当作真实密码神经训练证据。NBFNet、FloydNet、更深Transformer继续暂缓。

## 7. 执行产物

```text
outputs/local_audits/i2_real_spn_pair_state_transfer_readiness_20260718/

label_sources.json
label_readiness.csv
model_memory.csv
results.jsonl
gate.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

推荐执行路径：本地CPU审计；若本地CUDA可用可补显存测量，但不训练、不连接远程GPU。

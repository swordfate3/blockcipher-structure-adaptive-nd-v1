# 创新2 E106：PRESENT九轮外部关系来源新颖性就绪审计

日期：2026-07-21

状态：计划冻结 / 本地零训练审计待执行

## 1. 研究问题

E105证明新文件split不等于新关系空间：E104的321条关系中318条与公开语料完全相同，剩余3条也
位于公开468维GF(2) span内。E106只回答：

```text
当前已取得并可复核的ATM、论文或独立算法来源中，是否存在一份能直接评估冻结E99模型的
PRESENT九轮关系集，并相对公开468维训练空间提供至少32个独立新维度？
```

E106不训练网络、不执行密码搜索、不生成negative、不启动远程任务。论文报告的输出mask、弱密钥
组合、monomial或其他密码关系不能因为“也是积分结果”就强行转换成ATM `(u,v)`关系。

## 2. 同预算锚点与冻结候选

```text
training anchor = ATM公开8个PRESENT R9 split
anchor rows     = 470 serialized relations
anchor rank     = 468
model protocol  = E99 coordinate_deepsets relation-pool ranking
minimum novelty= 32 GF(2) dimensions outside the anchor span
epochs / seeds  = 0 / none
execution       = local CPU read-only audit
```

候选来源冻结为：

1. E104 ATM R9 `(3,3,3)`：机器可读且同语义；已验证`318`精确重合、`3`精确新关系、`0`新增维度。
2. ATM R10 notebook声明的9个split：公开结果文件为`0`，且轮数改变，不能直接评估R9冻结模型。
3. Hwang et al. 2026 PRESENT R9 kernel：论文给出4个输出balance mask，但对象是PRESENT-80、
   `2^60`输入结构上的64-bit输出mask，不是ATM独立轮密钥`(u,v)`关系。
4. Split-and-Cancel PRESENT：论文给出R9输出组合和弱密钥结果；公开仓库提交
   `aac4ab4d7430e4add3689214c9e69412a89d8fc1`当前只有README“codes soon”，没有机器结果，且
   输出observable/主密钥或弱密钥语义不同。
5. CLAASP-MP：存在通用monomial/superpoly代码方法，但当前语料没有冻结的PRESENT R9关系结果，
   cube/output-monomial语义也不是E99输入。

Hwang和Split-and-Cancel可以成为后续“更换表示或目标”的候选，但不能在E106中转换成E99来源。

## 3. 来源资格控制

每个候选逐项检查：

```text
source identity/hash verified
machine-readable positive relations exist
cipher = PRESENT
rounds = 9
key model = independent 64-bit round keys
relation semantics = ATM canonical set of (u,v) coordinates
exact relation overlap with training = 0
full synchronous-rotation pool overlap with every E99 train fold = 0
GF(2) dimensions outside public span >= 32
```

缺少机器产物时只能记`artifact_unavailable`；语义不同只能记`representation_incompatible`；论文报告
的基底维数不能当作ATM span新增维数。资格判断不得读取任何神经排名指标。

## 4. 裁决门

`external_source_ready`要求至少一个非训练锚点候选通过全部资格控制。通过后才允许另立零适配评价
计划，继续使用冻结E99 checkpoint、2 seeds、6 folds和不更新权重协议。

`external_source_unavailable`表示来源审计有效，但没有候选合格。此时永久停止E99绝对坐标身份的
外部来源路线；不生成ATM R10、不运行其他R9 split、不筛选E104三条关系、不增加epoch或网络宽度。

`protocol_invalid`表示任一冻结hash、公开468维锚点、E104验证或候选分类不能重放。只修审计，
不解释来源路线。

## 5. 产物与下一步

```text
run_id = i2_present_r9_external_relation_source_readiness_20260721
output = outputs/local_audits/i2_present_r9_external_relation_source_readiness_20260721/
```

至少输出`source_matrix.csv`、`results.jsonl`、`summary.json`、`gate.json`、`metadata.json`、
`progress.jsonl`和中文资格矩阵图。图必须经过`visual-qa-redraw`像素检查并写marker，随后刷新最近结果
索引。

若没有合格来源，推荐下一步是单独审计Hwang PRESENT R9四个固定输出mask能否通过现有确定性提供者
复现，且不枚举`2^60`明文。该下一步明确改变目标表示，不能加载E99 checkpoint。Hwang提供者不可行
时停止PRESENT R9输出mask训练路线；Split-and-Cancel仓库在代码/结果实际公开前保持来源hold。

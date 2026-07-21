# 创新2 E106：PRESENT九轮外部关系来源新颖性就绪审计

日期：2026-07-21

状态：已完成 / hold / 停止E99坐标迁移并转论文收束

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

Hwang和Split-and-Cancel是本次来源语义对照，不能在E106中转换成E99来源。E97已经关闭当前
PRESENT高轮输出预测provider研究，因此它们也不自动构成下一项实验。

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

## 6. 正式结果

冻结输入和全部协议检查通过，E106得到：

```text
status                              = hold
decision                            = innovation2_present_r9_external_relation_source_unavailable
candidate_sources                   = 5
eligible_external_sources           = 0
machine_readable_candidate_sources  = 1
same_atm_semantics_candidate_sources= 2
maximum_known_new_dimensions        = 0
minimum_required_new_dimensions     = 32
```

唯一机器可读且同语义的候选是E104 ATM R9 `(3,3,3)`，但其321条关系中318条与公开训练关系完全
重合；余下3条虽然文件层面新，仍未给公开468维span增加任何GF(2)维度。其他候选缺少机器关系、
轮数不符，或采用输出mask、弱密钥、monomial等不同语义，因此不能直接喂给冻结E99模型。

E106是本地零训练来源资格审计，不是新的神经网络结果、PRESENT-80攻击、论文复现或SOTA比较。

## 7. 推荐下一步与停止门

E106没有发现合格来源，因此永久停止E99绝对坐标身份的外部迁移路线；不生成ATM R10、不继续其他
split、不筛选E104三条关系，也不通过加epoch、网络宽度或远程GPU绕过来源问题。

同时遵守E97的provider停止门：不另立Hwang四mask复现实验，不提高exact-ANF、ATM等现有provider
的资源上限，不启动PRESENT七至九轮输出预测训练。默认下一步转论文收束，把E99--E106整理为
“九轮关系学习信号存在，但坐标身份泛化缺少span外独立来源”的证据链，并与E52--E55、E97的
“高轮严格标签生成边界”并列陈述。

只有未来出现以下任一新证据时才另立计划重开：

1. 新的sound provider算法或可验证证书实现，并重新通过E97同等级别的语义和复杂度门；
2. 机器可读、同为PRESENT R9独立轮密钥ATM `(u,v)`语义、与训练及旋转候选零重合，并在公开
   468维span外新增至少32维的独立来源。

最终`curves.svg`已按`visual-qa-redraw`渲染为`1800 x 1037`像素检查：标题、两行说明、六来源乘
六资格矩阵、行列标签、图例和底部三行结论均无字体重叠、裁切、缺字或语义歧义，验收记录为
`visual_qa_passed.marker`。最近结果索引状态见`outputs/00_RECENT_RESULTS.md`。

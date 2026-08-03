# 中文核心投稿与神经区分器报告规范核验

核验日期：2026-08-03
用途：为 `paper/chinese-core-innovation1/` 的最小实验矩阵、写作边界和投稿审计提供外部依据。

## 1. 使用边界

目标期刊尚未选定。本文使用《软件学报》官方投稿指南作为“计算机类中文核心”的一手通用校准，并结合神经差分密码分析领域原始论文与综述确定实验报告规范。下述内容不应被表述为《密码学报》或其他未选定期刊的具体格式要求。

## 2. 中文核心通用校准

权威来源：

- 《软件学报》官方投稿指南：<https://www.jos.org.cn/jos/site/menu/20210909102616001>
- 访问日期：2026-08-03

官方指南中与本文直接相关的要求：

1. 工作内容应与国内外同类工作进行详细讨论和比较；应“查新”“查全”，并注意国内同类工作。
2. 介绍他人工作时应明确标引来源，使审稿人可以判断作者工作的创新程度。
3. 新算法若声称优越，应有数学证明或实验数据对比支撑。
4. 投稿时应完整表达相关工作比较和作者工作；具体字数、页数及格式可在评审通过后按编辑部要求调整。

对本文的验收含义：

- 不能只与 AutoND/DBitNet 比较；必须包含代表性的公开网络架构同协议比较。
- 相关工作应加入国内密码神经网络研究，但不能把不同密码/协议写成直接可比结果。
- 方法贡献必须与正确/错误结构控制、同检查点反事实和同数据基线对应。
- 论文完整性由“主张是否有对应证据”决定，而不是由数据规模是否达到百万决定。

## 3. 神经差分密码分析领域规范

### 3.1 训练、验证与独立测试

来源：Gerault 等，*Survey: Six Years of Neural Differential Cryptanalysis*，IACR ePrint 2024/1300。

- 本地全文：`papers/innovation_one/text/2024_sok_neural_differential_cryptanalysis.txt`
- 原文位置：第 2698-2727 行说明训练随机性、训练/验证/测试集的不同作用，并指出用于模型优化的验证集不能再作为最终未知数据性能表征，应使用 fresh test data。
- 第 2739-2749 行要求：分别报告 train/validation/test 及各自规模；在多份 fresh test 上报告误差范围；报告参数量、FLOPs、每 epoch 训练时间及硬件；公开代码与模型。

对本文的验收含义：

- 现有结果只有冻结跨密钥 validation，没有独立 fresh final test，必须在局限性中明示。
- 两个 seed 只适合报告逐 seed、均值/范围和方向一致性，不适合做可靠显著性检验。
- 参数量已经可报告；FLOPs、训练时间、吞吐和显存尚不完整，应列为非阻断改进项。

### 3.2 基线迁移与超参数公平性

来源：Gohr、Leander、Neumann，*An Assessment of Differential-Neural Distinguishers*，IACR ePrint 2022/1521。

- 本地全文：`papers/innovation_one/text/2022_gohr_leander_neumann_assessment_differential_neural.txt`
- 原文位置：第 1231-1264 行说明超参数显著影响学习能力；LR、L2 penalty 和 filter size 影响最大；迁移到另一密码时至少应进行小范围搜索。

对本文的验收含义：

- K1-CA/K1-CB 是统一 uKNIT 数据和优化预算下的架构适配比较，不是各公开方法的最优复现。
- 不应把统一预算写成“对每个基线绝对公平”或“充分超参数搜索”。
- 这一限制应写入实验设置和局限性，但不应在投稿前临时扩成无边界的大搜索矩阵。

### 3.3 公开 paper-scale 参照

Gohr 2019：

- 本地全文：`papers/innovation_one/text/2019_gohr_speck32_deep_learning.txt`
- 第 498-512 行：生成 `10^7` 个训练样本，训练 200 epochs，保留 `10^6` 个验证样本，并在独立 `10^6` 测试集上评估最佳 validation-loss 网络。
- DOI：<https://doi.org/10.1007/978-3-030-26951-7_6>

AutoND/DBitNet 2023：

- 本地全文：`papers/innovation_one/text/2023_cipher_agnostic_neural_training_pipeline.txt`
- 第 1299-1304 行：说明每轮训练规模 `10^7`、验证规模 `10^6`；第 1633-1641 行说明 10/40 epochs 的比较和五份 fresh `10^6` 测试；第 1838-1840 行报告 PRESENT r9 accuracy 0.5092。
- DOI：<https://doi.org/10.46586/tosc.v2023.i3.184-212>

对本文的验收含义：

- 本文 `262144/class` 对应总训练行数 524288，不能称为 Gohr 或 AutoND 公开协议的精确 paper-scale 复现。
- 本文可以把该规模称为“当前项目最高规模/论文主规模”，并依靠同数据反事实回答结构归因问题。

## 4. 参考文献元数据核验轨迹

| 键 | 核验来源 | 核验结果 |
|---|---|---|
| `gohr2019speck` | Springer DOI/Crossref + 本地正式 PDF | 题名、作者、CRYPTO 2019、页 150-179、DOI 一致 |
| `gohr2022assessment` | IACR ePrint 2022/1521 landing page + 本地 PDF | 题名、三位作者、年份、报告号一致 |
| `bellini2023cipheragnostic` | Crossref DOI + 本地 ToSC PDF | 四位作者、ToSC 2023(3)、页 184-212、DOI 一致 |
| `gerault2024survey` | IACR ePrint 2024/1300 landing page + 本地 PDF | 题名、四位作者、年份、报告号一致 |
| `yu2023sm4` | 本地期刊正式 PDF 首页 | 中文作者、题名、《桂林电子科技大学学报》43(1):75-79 一致；未填写未核验 DOI |
| `zhang2022mcnd` | arXiv API 2204.06341 + 本地 PDF | Liu Zhang、Zilong Wang、题名和 2022 首次提交日期一致 |
| `liu2026spn` | DOI/Crossref + 本地 IEICE PDF | 四位作者、E109-D(2)、页 238-248、DOI 一致 |
| `chen2021multiplepairs` | IACR ePrint 2021/310 landing page + 本地 PDF | 四位作者、题名、年份和报告号一致 |
| `hu2026uknit` | ToSC DOI landing page + 本地正式 PDF | 四位作者、ToSC 2026(2)、DOI 一致；官方元数据未给连续页码，文献表不臆造页码 |
| `banik2025dialga` | Crossref DOI + ToSC 元数据 | 八位作者、ToSC 2025(4)、页 70-124、DOI 一致 |

正式 BibTeX 文件：`paper/chinese-core-innovation1/references.bib`。

## 5. 对当前论文实验矩阵的最终约束

必须完成：

- uKNIT K1-CA/K1-CB 五模型同缓存、双 seed 主表；
- uKNIT K1-U、Dialga D1/D2/DMC2 的机制归因；
- uKNIT K1-BV、Dialga D3 的失败边界；
- 正式参考文献、方法总图、五模型主图和投稿完整性审计。

不再默认开展：

- 百万级训练、第三 seed、五次 fresh test；
- 新网络族、新轮数、新密码或新攻击模型；
- 每个基线的大规模超参数搜索。

这些项目可作为后续增强或针对真实审稿意见的补充，但不是当前论文主张的必要前置条件。

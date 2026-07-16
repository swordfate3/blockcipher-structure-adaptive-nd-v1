# 创新2高轮积分与输出预测论文语料库

**检索日期：** 2026-07-16
**语料状态：** 14 篇全文已验证并提取文本；2 篇直接核心全文访问受阻；
3 篇强基线仅取得权威元数据；1 篇撤稿记录排除
**创新目标：** 在真实 keyed SPN 上，把神经网络用于高轮积分区分、积分
输出组合发现或结构条件积分候选预测，并与确定性搜索和简单统计基线比较

## 1. 结论先行

这批论文把创新2的边界收紧为三个必须同时满足的要求：

1. 不能只做低轮 `PRESENT r5` 结构概率排序；最终必须回答神经方法在
   `PRESENT r8` 或更高轮是否仍有高于强基线的有效信号。
2. 不能把“神经网络提出候选，再交给传统方法验证”写成首创。Zhang 等
   EUROCRYPT 2026 已明确采用神经特征探索并改进积分搜索模型。
3. 必须比较非神经强基线。division property、MILP、monomial prediction、
   algebraic transition matrix、kernel 和 Split-and-Cancel 都可能直接解释或
   超过神经候选。

因此，当前最优先复现锚点仍是 Wu/Guo 2024 的 keyed PRESENT-80
`r5/r6/r7/r8` 积分神经区分协议；项目候选网络只能在同一数据、同一预算和
同一负类定义下与它比较。

## 2. 直接核心论文

| 优先级 | 论文 | 与创新2的关系 | 本地状态 |
| ---: | --- | --- | --- |
| 1 | Wu and Guo, *Improved integral neural distinguisher model for lightweight cipher PRESENT* (2024), DOI `10.1186/s42400-024-00258-0` | 直接的 PRESENT-80 高轮积分神经锚点；报告 r8 accuracy `0.5732` | PDF 与文本已入库 |
| 2 | Zhang et al., *Neural-Inspired Advances in Integral Cryptanalysis* (EUROCRYPT 2026), DOI `10.1007/978-3-032-25333-0_16` | 神经网络作为积分特征探索器，再用改进的经典搜索确认；直接限定首创边界 | PDF 与文本已入库 |
| 3 | Hwang et al., *Improving Neural-Inspired Integral Distinguishers via a Linear-Algebraic Approach* (ePrint 2026/340) | 用跨密钥 parity matrix 的 kernel 寻找 balance-mask 空间，是神经模型必须超过的经验线性代数基线 | 题录和摘要已核验；PDF 被 Cloudflare 拦截 |
| 4 | Zahednejad and Lyu, *An improved integral distinguisher scheme based on neural networks* (2022), DOI `10.1002/int.22895` | Wu/Guo 的直接前序神经积分方法；原文比较不可缺 | 题录已核验；Wiley 全文端点被 Cloudflare 拦截 |
| 5 | Kimura et al., *Output Prediction Attacks on Block Ciphers Using Deep Learning* (2022) | 神经输出预测的基础协议；帮助区分具体输出恢复与积分性质预测 | PDF 与文本已入库 |
| 6 | Kimura et al., *A Deeper Look into Deep Learning-based Output Prediction Attacks Using Weak SPN Block Ciphers* (2023) | 解释输出预测成功与弱 S-box、扩散和经典安全性质的关系 | PDF 与文本已入库 |
| 7 | Watanabe et al., *On the Effects of Neural Network-based Output Prediction Attacks on the Design of Symmetric-key Ciphers* (2024) | 把输出预测现象联系到积分 `ALL/BALANCE` 与扩散；是本项目任务定义的重要边界 | PDF 与文本已入库 |

本地路径：

```text
papers/innovation_two/pdf/
papers/innovation_two/text/
papers/innovation_two/sources/paper_manifest.csv
sources/research_innovation2_paper_manifest_20260716.csv
```

`papers/` 按项目规则不进入 Git；`sources/` 中的同内容 manifest 快照用于
版本化题录、状态、页数与哈希审计。

## 3. 必要强基线

| 论文 | 关键作用 | 本地状态 |
| --- | --- | --- |
| Wu and Wang, *Integral Attacks on Reduced-Round PRESENT* (2013) | PRESENT 手工积分攻击与轮数历史基线 | PDF 与文本已入库 |
| Todo, *Structural Evaluation by Generalized Integral Property* (2015) | division property 基础 | PDF 与文本已入库 |
| Xiang et al., *Applying MILP Method to Searching Integral Distinguishers...* (2016) | 自动搜索到 9-round PRESENT integral distinguisher | PDF 与文本已入库 |
| Derbez and Lambin, *Fast MILP Models for Division Property* (2022) | 更高效的现代 MILP/division-property 基线 | PDF 与文本已入库 |
| Hadipour and Eichlseder, *Integral Cryptanalysis of WARP based on Monomial Prediction* (2022) | 单输出位之外的非线性输出组合与 monomial prediction | PDF 与文本已入库 |
| Beyne and Verbauwhede, *Integral Cryptanalysis Using Algebraic Transition Matrices* (2023) | 以代数转移矩阵寻找广义积分性质，并包含 PRESENT 应用 | PDF 与文本已入库 |
| Hadipour et al., *Improved Search for Integral, Impossible Differential and Zero-Correlation Attacks* (2024) | 现代自动搜索、精度与效率权衡，包含 PRESENT | PDF 与文本已入库 |
| Wang, Hadipour, and Gerhalter, *On Extending Integral Distinguishers* (2026) | Split-and-Cancel；对 PRESENT/GIFT 给出组合输出和弱密钥积分的精确基线 | PDF 与文本已入库 |

三篇题录已核验但未取得开放全文：Todo and Morii 2016 的
*Compact Representation for Division Property*、Wang et al. 2019 的
*Improved Integral Attacks on PRESENT-80*、Hebborn et al. 2021 的
*Strong and Tight Security Guarantees Against Integral Distinguishers*。

## 4. 邻近论文，不可混报

以下论文已经存在于项目中，可用于网络结构或轮数背景比较，但不是积分神经
方法本身，因此没有机械复制进本语料目录：

| 论文 | 已有路径 | 用途边界 |
| --- | --- | --- |
| Zhang and Wang 2022, differential-neural PRESENT | `papers/innovation_one/pdf/2022_zhang_wang_improving_differential_neural_des_chaskey_present.pdf` | PRESENT r6/r7 差分神经区分参照，不是积分标签 |
| Gauthier-Umaña et al. 2026, entropy-based PRESENT ND | `papers/innovation_one/pdf/2026_present_entropy_nd.pdf` | 低参数差分神经与密钥恢复参照，不是积分方法 |
| Liu et al. 2026, IoT-friendly SPN ND framework | `papers/innovation_one/pdf/2026_liu_spn_iot_friendly_neural_distinguisher_framework.pdf` | SPN 网络结构参照，不是积分区分协议 |
| Singh 2025, PRESENT full-round emulation | `papers/innovation_two/pdf/2025_singh_present_full_round_emulation.pdf` | 删除 AddRoundKey 的公开变换仿真，不能当作 keyed PRESENT 高轮攻击 |

## 5. 排除项与访问失败

- *IABC: A neural integral distinguisher for AND-RX Ciphers* 在检索中被标记为
  已撤稿，且未恢复到稳定题录或有效 PDF，故不进入证据语料。
- Hwang 2026 的 IACR 页面、作者、摘要、PDF URL 和 CC BY 4.0 信息均已
  保存，但 PDF 端点返回 Cloudflare challenge HTML。
- Zahednejad 2022 的 Wiley PDF、EPDF 和 full-XML 均返回 Cloudflare
  challenge HTML；Crossref DOI 和题录有效。
- 三个以 `.pdf` 命名的 Springer 响应经 `file` 检查实际为 HTML，已拒绝
  入库。manifest 保留其元数据状态，避免以后误认为全文已经下载。

这不是“所有数据库无遗漏”的系统综述。当前检索覆盖 Crossref、OpenAlex、
arXiv、IACR 页面、ToSC、SpringerOpen 和项目现有论文库；Semantic Scholar
接口返回错误，通用搜索后端因环境网络/认证限制不可用。结论应表述为：
截至 2026-07-16，在可访问的权威来源中，直接核心与主要方法强基线已经
形成可执行语料库，仍有 5 篇已知题录未取得合法开放 PDF。

## 6. 推荐阅读顺序

```text
Wu/Guo 2024
  -> Zahednejad/Lyu 2022（当前仅题录，先读 Wu/Guo 的复述与比较）
  -> Zhang et al. 2026
  -> Hwang et al. 2026（当前先读已保存摘要）
  -> On Extending Integral Distinguishers
  -> Todo 2015 + Xiang 2016
  -> Beyne 2023 + Hadipour 2022/2024
  -> Kimura 2022/2023 + Watanabe 2024
```

前四项决定创新2能否主张“神经高轮积分方法”；中间五项决定强基线和精确
验证边界；最后三项负责解释“具体输出预测”和“积分输出性质预测”的区别。

## 7. 对下一实验的直接影响

下一实验不应继续在 r5 的 111-bit 结构向量上调 MLP，也不应直接尝试 GIFT。
应先按 `docs/experiments/innovation2-present-high-round-integral-neural-anchor-plan.md`
完成 Wu/Guo 数据协议审计，冻结：

```text
PRESENT-80 keyed encryption
r5/r7/r8 ladder
16-plaintext integral multiset
InvP + InvS representation
encrypted-random-plaintext negative
paper anchor + same-input candidate + linear + shuffled + deterministic statistics
```

r8 两个独立 seed 的置信区间高于随机，才能称为达到主流神经积分轮数；
项目候选还必须在同预算下超过 Wu/Guo anchor 和强统计基线，才能主张模型
层面的新贡献。

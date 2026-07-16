# 创新2高轮积分与输出预测论文语料库

**检索日期：** 2026-07-16
**语料状态：** 创新2目录有 23 篇已验证全文；另有 1 篇直接相关的部分
解密特征论文已存在于创新1目录；主语料只剩 1 篇付费出版物未发现合法
开放全文；1 篇撤稿记录排除
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
| 3 | Hwang et al., *Improving Neural-Inspired Integral Distinguishers via a Linear-Algebraic Approach* (ePrint 2026/340) | 用跨密钥 parity matrix 的 kernel 寻找 balance-mask 空间，是神经模型必须超过的经验线性代数基线 | 本机跨项目论文库找到作者稿；PDF 与文本已入库 |
| 4 | Zahednejad and Lyu, *An improved integral distinguisher scheme based on neural networks* (2022), DOI `10.1002/int.22895` | Wu/Guo 的直接前序神经积分方法；原文比较不可缺 | 本机跨项目论文库找到作者上传稿；PDF 与文本已入库 |
| 5 | Kimura et al., *Output Prediction Attacks on Block Ciphers Using Deep Learning* (2022) | 神经输出预测的基础协议；帮助区分具体输出恢复与积分性质预测 | PDF 与文本已入库 |
| 6 | Kimura et al., *A Deeper Look into Deep Learning-based Output Prediction Attacks Using Weak SPN Block Ciphers* (2023) | 解释输出预测成功与弱 S-box、扩散和经典安全性质的关系 | PDF 与文本已入库 |
| 7 | Watanabe et al., *On the Effects of Neural Network-based Output Prediction Attacks on the Design of Symmetric-key Ciphers* (2024) | 把输出预测现象联系到积分 `ALL/BALANCE` 与扩散；是本项目任务定义的重要边界 | PDF 与文本已入库 |

本地路径：

```text
papers/innovation_two/pdf/
papers/innovation_two/text/
papers/innovation_two/sources/paper_manifest.csv
sources/research_innovation2_paper_manifest_20260716.csv
sources/research_innovation2_iacr_*_20260716.html
sources/research_innovation2_eprint_*_20260716.html
```

`papers/` 按项目规则不进入 Git；`sources/` 中的同内容 manifest 快照用于
版本化题录、状态、页数与哈希审计。

## 3. 必要强基线

| 论文 | 关键作用 | 本地状态 |
| --- | --- | --- |
| Wu and Wang, *Integral Attacks on Reduced-Round PRESENT* (2013) | PRESENT 手工积分攻击与轮数历史基线 | PDF 与文本已入库 |
| Todo, *Structural Evaluation by Generalized Integral Property* (2015) | division property 基础 | PDF 与文本已入库 |
| Xiang et al., *Applying MILP Method to Searching Integral Distinguishers...* (2016) | 自动搜索到 9-round PRESENT integral distinguisher | PDF 与文本已入库 |
| Todo and Morii, *Compact Representation for Division Property* (2016) | bit-based/three-subset division property 的紧凑表示基础 | 从手动提供的 CANS 2016 论文集准确抽取 17 页章节；PDF 与文本已入库 |
| Hebborn et al., *Strong and Tight Security Guarantees Against Integral Distinguishers* (2021) | 一般积分抗性的强安全保证，是 Peng 2026 的直接理论前序 | 手动下载 ePrint；PDF 与文本已入库 |
| Derbez and Lambin, *Fast MILP Models for Division Property* (2022) | 更高效的现代 MILP/division-property 基线 | PDF 与文本已入库 |
| Hadipour and Eichlseder, *Integral Cryptanalysis of WARP based on Monomial Prediction* (2022) | 单输出位之外的非线性输出组合与 monomial prediction | PDF 与文本已入库 |
| Beyne and Verbauwhede, *Integral Cryptanalysis Using Algebraic Transition Matrices* (2023) | 以代数转移矩阵寻找广义积分性质，并包含 PRESENT 应用 | PDF 与文本已入库 |
| Hadipour et al., *Improved Search for Integral, Impossible Differential and Zero-Correlation Attacks* (2024) | 现代自动搜索、精度与效率权衡，包含 PRESENT | PDF 与文本已入库 |
| Wang, Hadipour, and Gerhalter, *On Extending Integral Distinguishers* (2026) | Split-and-Cancel；对 PRESENT/GIFT 给出组合输出和弱密钥积分的精确基线 | PDF 与文本已入库 |
| Peng et al., *Delving Deep into Security Guarantees against Integral Distinguishers with Applications to PRESENT, TWINE and LBLOCK* (ePrint 2026/961; DOI `10.1007/s10623-026-01871-5`) | 把 PRESENT 的积分抗性保证从 13 轮收紧到 11 轮，并部分分析 10 轮；这是“不存在区分器”的安全边界，不是 11 轮区分器 | 手动下载 ePrint；PDF 与文本已入库 |
| Bellini et al., *CLAASP-MP: An Automated MILP Framework for Monomial Prediction* (ePrint 2026/735) | 当前自动 monomial prediction/3SDP-woU 工具基线，能解释神经候选是否只是经典代数性质 | 手动下载 ePrint；PDF 与文本已入库 |

主语料只剩一篇付费全文缺口：Wang et al. 2019 的 *Improved Integral
Attacks on PRESENT-80*。出版商付费墙已由用户手动确认；题录与 DOI 保留为
`metadata_only`，不能写成已读全文。Todo and Morii 2016 已从用户提供的
CANS 2016 全书中按物理 PDF 页 33--49 抽取，正文页码为 19--35。

## 4. 邻近论文，不可混报

以下论文已经存在于项目中，可用于网络结构或轮数背景比较，但不是积分神经
方法本身，因此没有机械复制进本语料目录：

| 论文 | 已有路径 | 用途边界 |
| --- | --- | --- |
| Zhang and Wang 2022, differential-neural PRESENT | `papers/innovation_one/pdf/2022_zhang_wang_improving_differential_neural_des_chaskey_present.pdf` | PRESENT r6/r7 差分神经区分参照，不是积分标签 |
| Bellini et al. 2025, generic partial decryption | `papers/innovation_one/pdf/2025_gpd_feature_engineering_nd.pdf` | 与 `InvP + InvS` 的结构特征工程直接相关，但论文任务是差分神经区分；项目副本已验证为 27 页 PDF，故不重复下载 |
| Gauthier-Umaña et al. 2026, entropy-based PRESENT ND | `papers/innovation_one/pdf/2026_present_entropy_nd.pdf` | 低参数差分神经与密钥恢复参照，不是积分方法 |
| Liu et al. 2026, IoT-friendly SPN ND framework | `papers/innovation_one/pdf/2026_liu_spn_iot_friendly_neural_distinguisher_framework.pdf` | SPN 网络结构参照，不是积分区分协议 |
| Singh 2025, PRESENT full-round emulation | `papers/innovation_two/pdf/2025_singh_present_full_round_emulation.pdf` | 删除 AddRoundKey 的公开变换仿真，不能当作 keyed PRESENT 高轮攻击 |
| Carlet 2025, S-box partial integral resistance | `papers/innovation_two/pdf/2025_carlet_sbox_integral_resistance.pdf` | `k` 阶 `t`-degree-sum-freedom 与 division-property 传播的理论背景；不是神经方法或 PRESENT 实验基线 |
| Gerhalter and Eichlseder 2026, PRINCE complex-linear-layer integral resistance | `papers/innovation_two/pdf/2026_gerhalter_prince_integral_resistance.pdf` | 邻近 SPN 的积分抗性与 degree-bound 方法，不是 PRESENT 主结果 |
| Beierle et al. 2025, modular-addition key whitening integral resistance | `papers/innovation_two/pdf/2025_beierle_modular_addition_integral_resistance.pdf` | ARX/模加白化背景，不是当前 keyed SPN 目标 |

Gerhalter/Eichlseder 与 Beierle 等两篇均已完成 PDF、文本、页数和 SHA-256
核验；它们补充方法背景，但均不作为 PRESENT 主结果比较项。

## 5. 排除项与访问失败

- *IABC: A neural integral distinguisher for AND-RX Ciphers* 在检索中被标记为
  已撤稿，且未恢复到稳定题录或有效 PDF，故不进入证据语料。
- Hwang 2026 的 IACR PDF 端点仍返回 Cloudflare challenge HTML，但本机
  另一个积分分析项目中存在同一作者稿；经标题、36 页、PDF 魔数和 SHA-256
  核验后已入库，不再标记为缺口。
- Zahednejad 2022 的 Wiley PDF、EPDF 和 full-XML 仍返回 Cloudflare
  challenge HTML，但本机论文库中的作者上传稿已经核验并入库。
- Hebborn 2021、Peng 2026、CLAASP-MP 2026、Gerhalter/Eichlseder 2026 和
  Beierle et al. 2025 的命令行端点此前返回 Cloudflare HTML；用户手动提供
  的 PDF 已逐篇通过标题、页数、PDF 魔数和 SHA-256 核验，因此不再属于
  全文缺口。
- Todo/Morii 2016 的两份手动下载完全同哈希，均为 755 页 CANS 2016 全书。
  目标章节已准确抽取，两个原始整书文件保存在
  `papers/innovation_two/sources/manual_downloads/`，不重复计入论文数。
- 先前以 `.pdf` 命名的 Springer 命令行响应经 `file` 检查实际为 HTML；
  2019 PRESENT-80 论文又经用户确认需要付费，故保持 `metadata_only`。
- 本轮补检尝试通过 OpenAlex 继续寻找剩余作者稿时，外部网络权限被平台
  审批器明确拒绝；没有通过其他网络通道绕过限制。

这不是“所有数据库无遗漏”的系统综述。当前检索覆盖 Crossref、OpenAlex、
arXiv、IACR ePrint 关键词交叉检索、ToSC、Springer 和项目现有论文库；
Semantic Scholar 有限流，通用搜索后端因环境网络/认证限制不可用。结论应
表述为：截至 2026-07-16，在可访问的权威来源、本机跨项目论文库和用户
手动提供文件中，创新2已经形成 23 篇全文的可执行语料库；直接核心和主要
方法强基线只剩 Wang et al. 2019 一篇付费全文缺口。该缺口不阻塞当前
PRESENT r8 神经积分实验，但引用其具体结果时必须标明未核对原文。

## 6. 推荐阅读顺序

```text
Wu/Guo 2024
  -> Zahednejad/Lyu 2022
  -> Zhang et al. 2026
  -> Hwang et al. 2026
  -> Peng et al. 2026（注意：11 轮是抗性保证，不是区分轮数）
  -> On Extending Integral Distinguishers
  -> Hebborn 2021 + Todo 2015 + Todo/Morii 2016 + Xiang 2016
  -> Beyne 2023 + Hadipour 2022/2024 + CLAASP-MP 2026
  -> Generic Partial Decryption 2025（已有创新1项目副本）
  -> Kimura 2022/2023 + Watanabe 2024
```

前四项决定创新2能否主张“神经高轮积分方法”；Peng、Split-and-Cancel、
division-property、monomial-prediction 和 kernel 路线决定强基线和精确验证
边界；最后的输出预测论文负责解释“具体输出预测”和“积分输出性质预测”的
区别。

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

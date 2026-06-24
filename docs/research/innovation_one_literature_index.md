# 创新一相关论文库索引

## 本地目录

- PDF 全文：`papers/innovation_one/pdf/`
- 抽取文本：`papers/innovation_one/text/`
- 文献状态唯一来源：`papers/innovation_one/sources/paper_manifest.csv`
- 失败/误下载原始页面归档：`papers/innovation_one/sources/failed_downloads/`

目前已下载并验证为真实 PDF 的论文：30 篇。每篇都已用 `pdftotext` 抽取为 `.txt`，便于后续检索和精读。具体下载状态、来源和备注统一看 `paper_manifest.csv`，本文档不重复维护逐篇状态。

## 最贴合创新一的优先阅读顺序

### 第一优先级：定义创新一边界

1. `2019_gohr_speck32_deep_learning.pdf`
   - Gohr 的 SPECK32/64 残差神经区分器，是 ARX/ResNet-BitSlice 的起点。
   - 对创新一的作用：定义 ARX 强基线，后续所有结构匹配都要和它区分开。

2. `2023_cipher_agnostic_neural_training_pipeline.pdf`
   - DBitNet/cipher-agnostic pipeline，自动寻找输入差分，支持多类密码。
   - 对创新一的作用：这是最重要的“边界论文”。我们不能声称首次通用，而应声称结构感知匹配和统一横向比较。

3. `2022_gohr_leander_neumann_assessment_differential_neural.pdf`
   - 评估 differential-neural distinguisher 的方法论论文。
   - 对创新一的作用：支撑统一评价协议、多种子、可比性、避免只报单次精度。

4. `2024_sok_neural_differential_cryptanalysis.pdf`
   - 领域综述，强调 comparability 和 scaling 问题。
   - 对创新一的作用：把“结构感知匹配策略”放到领域缺口里。

### 第二优先级：结构分支证据

5. `2021_benamira_deeper_look_ml_cryptanalysis.pdf`
   - 解释 Gohr 风格模型到底学习了什么。
   - 对创新一的作用：结构匹配不能只靠结果，要有特征证据。

6. `2026_liu_spn_iot_friendly_neural_distinguisher_framework.pdf`
   - 轻量级 SPN 神经区分器框架，明确关注训练数据格式和网络结构。
   - 对创新一的作用：强支撑 SPN 结构需要单独的数据表示和网络设计。

7. `2023_yu_wu_zhang_sm4_conv_resnet_analysis.pdf`
   - 中文 SM4 卷积残差网络分析。
   - 对创新一的作用：直接服务论文中的 SM4 分支，说明 SM4 不能只当普通 SPN 处理。

8. `2020_hou_linear_attack_des_deep_learning.pdf`
   - DES 深度学习线性攻击。
   - 对创新一的作用：支撑 Feistel 结构有独立的深度学习攻击/检测路线。

### 第三优先级：效率、输入表示和解释

9. `2022_reducing_cost_ml_differential_attacks_bit_selection.pdf`
   - bit selection 与 partial ML-distinguisher。
   - 对创新一的作用：支撑数据复杂度和输入特征选择指标。

10. `2021_chen_multiple_ciphertext_pairs_neural_distinguisher.pdf`
    - 多密文对特征的神经区分器。
    - 对创新一的作用：支撑输入表示会影响网络匹配效果。

11. `2023_nnbits_bit_profiling_deep_learning_ensemble.pdf`
    - bit profiling 与 ensemble 解释。
    - 对创新一的作用：后续可作为“为什么某结构适配某网络”的解释模块。

12. `2024_theoretical_explanation_improvement_deep_learning_distinguisher.pdf`
    - 神经启发差分密码分析解释与增强。
    - 对创新一的作用：补充理论解释和可解释性讨论。

13. `2025_gpd_feature_engineering_nd.pdf`
    - Generic Partial Decryption，自动构造部分解密特征。
    - 对创新一的作用：强支撑“结构感知特征工程”不是简单堆模型。

14. `2025_speck_simon_multi_pair_multiscale.pdf`
    - Speck/Simon 多密文对输入、多尺度卷积和 dense residual。
    - 对创新一的作用：提示 MoE 还要和多密文对/强 CNN 数据格式基线比较。

15. `2025_rx_neural_simon_simeck.pdf`
    - Rotational-XOR neural distinguisher。
    - 对创新一的作用：支撑差分类型要按密码结构扩展，不应只有 XOR difference。

16. `2026_polytopic_pdnd_simon_simeck_speck.pdf`
    - Polytopic differential neural distinguishers。
    - 对创新一的作用：支撑多输入/多差分样本结构，是 `polytope_size` schema 的依据。

17. `2026_present_entropy_nd.pdf`
    - PRESENT entropy-based neural distinguisher/key recovery。
    - 对创新一的作用：支撑 SPN/PRESENT 分支的 bit selection 和攻击级扩展。

## 按密码结构分类

| 结构 | 已下载论文 | 对应创新一用途 |
|---|---|---|
| ARX / SPECK | Gohr 2019; Benamira 2021; Bao 2022; Ebrahimi 2022; Chen multiple-pair 2021; Hou et al. 2025; Mirzaali et al. 2026 | ResNet-BitSlice、bit selection、multi-pair features、polytopic inputs |
| AND-RX / SIMON / SIMECK | Lu et al. 2022 arXiv/ePrint; Bao 2022; Liu et al. 2025; Mirzaali et al. 2026 | 补充 ARX 与 AND-RX 差异，支持 RX difference、polytopic、related-key 扩展 |
| SPN / PRESENT | Jain et al. 2020; Jain et al. 2021; Liu et al. 2026; PRESENT entropy 2026 | CNN-SBoxLocal、DBitNet、SPN-specific data format、entropy bit selection |
| Feistel / DES | Hou et al. 2020 | 证明 Feistel 结构有独立神经/深度学习攻击路线 |
| Feistel-like / SM4 | 余玥琳等 2023; Li & Sun 2025 | SM4 分支的中文和差分背景支撑 |
| 通用流水线/比较 | DBitNet 2023; Assessment 2022; SoK 2024; Bellini & Rossi 2020; GPD 2025 | 支撑“统一实验协议”和“结构感知匹配”的论文定位 |

## 可直接支撑的论文表述

创新一建议写成：

> 现有神经网络区分器研究多以单一算法或单一结构为对象，例如 Gohr 针对 SPECK32/64 构造残差网络区分器，后续工作围绕 SPECK/SIMON 等 ARX 或 AND-RX 结构持续优化；另一方面，DBitNet 等密码无关流水线提高了跨算法复用能力，但其目标是构建通用训练流程，而不是显式刻画密码结构特征与网络架构偏好之间的对应关系。为解决不同结构密码实验标准不一、模型选择缺乏结构依据的问题，本文提出结构感知的神经网络区分器匹配策略，在统一数据生成、训练预算和评价指标下，比较 ARX、SPN、Feistel-like 等结构与 ResNet、CNN、DBitNet、RNN/LSTM 等网络之间的适配关系。

## 后续精读任务

1. 先读 Gohr 2019，抽出 SPECK 数据构造、网络结构、评价指标。
2. 再读 DBitNet 2023，抽出它覆盖的密码、输入差分搜索和通用架构边界。
3. 读 Assessment 2022 和 SoK 2024，整理“为什么必须统一实验协议”。
4. 读 PRESENT/SPN 两篇和 2026 SPN 框架，确定 SPN 分支的输入表示。
5. 读 SM4 中文论文和 SM4 2025 差分论文，确定 SM4 在论文里写作 Feistel-like 的依据。

## 状态管理规则

`paper_manifest.csv` 是唯一状态源：

- `downloaded`：必须同时存在 PDF 和抽取后的 text。
- `pending_download`：题名和来源已确认，但全文尚未成功入库。
- `failed_downloads/`：只保存失败下载得到的原始页面或误下载文件，不作为状态源。

更新论文状态时，只修改 manifest；研究文档只引用 manifest 的汇总结果。

# Innovation 1 uKNIT K1-CB Published-Network Paper Comparison Plan

**Date:** 2026-08-03
**Status:** preregistered; waiting for protocol-valid K1-CA completion and exact cache reuse readiness
**Run ID:** `i1_uknit_r5_k1cb_published_comparison_262144_s3s4_20260803`

## 1. Paper Question

在 uKNIT-BC r5 的同一差分、密钥、负样本、pair 数、训练/验证样本和 epoch
预算下，本文 K1-CA 位置不变结构专家相对代表性公开神经网络架构的优势是否在
`262144/class`、seed 3/4 上同时成立？

K1-CB 是论文主表的直接比较实验，不是架构晋级筛选，也不设置“基线低于某值才
通过”的研究门。只要协议完整，所有观察值都进入论文；如果某个公开网络更强，
同样如实报告。

## 2. Frozen Evidence Anchor

K1-CA 原样提供两个不可重训的锚点：

1. `runtime_spn_ct_k1t_position_histogram_invariant`；
2. `autond_dbitnet2023`。

K1-CB 只补三个公开架构适配器：

| Architecture | Model key | Literature role |
|---|---|---|
| Zhang/Wang MCND | `spn_zhang_wang_mcnd_adapter` | 多分支一维卷积与递增核残差网络 |
| Liu raw Case-3 Conv2D | `spn_liu_case3_conv2d_adapter` | 三通道 SPN 状态矩阵 Conv2D-ResNet |
| Gohr-style ResNet | `spn_gohr_style_resnet_pairset_adapter` | 经典差分神经区分残差 CNN 的 64-bit SPN pair-set 适配 |

合并 K1-CA 与 K1-CB 后，论文主表包含本文方法和四个外部网络家族：AutoND、
MCND、Liu Conv2D、Gohr-style ResNet。两个项目内部通用 SPN 网络保留在 K1-BS
诊断表，不进入主规模表。

## 3. Frozen Six-Row Matrix

| Seed | MCND | Liu Conv2D | Gohr-style ResNet |
|---:|---:|---:|---:|
| 3 | train | train | train |
| 4 | train | train | train |

所有六行冻结为：

```text
cipher = uKNIT-BC
rounds = 5
active cell = 11
input difference = 0x0000400000000000
samples_per_class = 262144
train rows per model/seed = 524288
cross-key validation rows per model/seed = 131072
pairs_per_sample = 4
negative mode = encrypted_random_plaintexts
sample structure = independent_pairs
seeds = 3, 4
epochs = 10
batch size = 64
loss = MSE
optimizer = Adam
learning rate = 0.0001
weight decay = 0.00001
LR scheduler = none
checkpoint = restored best validation AUC
final-test repeats = 0
```

该设置优先回答同数据、同训练预算的架构比较，不是逐论文原协议复现，也不是充分
超参数搜索。三种适配器沿用 K1-BZ 已冻结的结构选项；Gohr-style ResNet 沿用
项目已有的 64 通道、7 个残差块、128 维分类头设置。不得在看到 K1-CA 或 K1-CB
结果后修改其中任一网络的配置。

## 4. Exact Cache-Reuse Contract

K1-CB 不得创建任何训练或验证数据。它必须只读复用 K1-CA 位于下列目录的四份
缓存：

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\
  i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803\cache
```

启动前必须检查两个 seed 的 train/validation `features.npy`、`labels.npy` 和
`metadata.json` 均存在，参数和数组形状与 K1-CB 计划完全一致。运行 progress
必须包含 `3 models x 2 seeds x 2 splits = 12` 次 `cache_reuse`，且 `cache_start`
和 `cache_done` 都必须为零。任何缺失或不匹配直接停止；禁止回退为数据生成。

## 5. Protocol And Reporting Gates

协议门要求：

- K1-CA 本地重裁决协议有效，四行、四缓存、四次 AutoND 复用和结果绑定完整；
- K1-CB 恰有六个计划行、六个结果行和六个非空 best checkpoint；
- source commit 与 GitHub 推送 SHA 完全一致；
- 每行均为 `262144/class` 训练、`131072` 总跨密钥验证、4 pairs、10 epochs；
- seed 3/4 的训练和验证密钥与 K1-CA 完全相同；
- 恰有 12 次源缓存复用、零缓存创建、零 final-test cache；
- CUDA、batch 64、best `val_auc` checkpoint 和完整 10-epoch history 均可追溯；
- progress 中存在 `run_done`。

K1-CB 没有预设性能淘汰阈值。协议有效即状态 `complete`，随后计算本文候选相对
AutoND、MCND、Liu Conv2D 和 Gohr-style ResNet 的逐 seed AUC 差值、双 seed
均值与方向一致性。论文不得只选择有利 seed 或删除表现较强的基线。

## 6. Execution Path

K1-CA 继续由现有远程任务和本地 tmux monitor 完成，不中断、不修改运行根。
K1-CB 只能在 K1-CA 完成、回收并确认缓存可复用后，从新的 GitHub 已推送提交在
远程 A6000 上启动。新 run 的 source、logs、checkpoints、results 和 archive
位于自己的 `G:\lxy` run root，但 `dataset_cache_root` 必须指向 K1-CA 的缓存根。

本地启动门必须在任何面向 A6000 的 `scp` 或 `ssh` 前写出并核验
`should_ssh=true`、`ssh_allowed=true` 和 `launch_authorized=true`。该门同时绑定
本地重裁决后的 K1-CA 四行协议、四缓存 manifest、K1-CB 六行计划、论文资源合同、
远程 readiness、受保护工作区和 GitHub `main` 的精确提交。门禁失败时不得远程
接触；即使本地门通过，远程缓存逐文件审计仍须在训练前失败关闭。

启动使用 `cmd.exe /c`，只做一次有界启动确认；后续由本地 tmux monitor 自动
回收。完成后执行结果验证、K1-CA/K1-CB 合并裁决、recent-results 刷新和论文
主表/图更新。图必须完成 `visual-qa-redraw` 像素检查后才能写为完成。

## 7. Evidence Boundary And Final Action

可写主张限定为：在本文冻结的 uKNIT r5 项目协议、双 seed 和
`262144/class` 预算下，本文方法与四种代表性外部架构完成了同数据直接比较。

禁止把结果写成公开论文的精确复现、充分超参数搜索、百万规模正式证据、完整轮
攻击、SOTA、密钥恢复突破或统一网络通吃所有 SPN。

K1-CB 协议有效完成后，不再增加网络、seed、数据、epoch、pair、轮数或最终测试，
直接合并 K1-CA/K1-CB 结果并转入论文定稿。

# Innovation 1 uKNIT K1-CA Invariant/AutoND Paper Closeout Plan

**Date:** 2026-08-03
**Status:** prepared; local readiness required; remote A6000 only
**Run ID:** `i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803`

## 1. Research Question

在不改变 uKNIT-BC 5 轮差分任务、密钥、负样本、优化器、训练轮数和
pair 数的前提下，K1-U 选出的“正确 S 盒语义 + 位置不变聚合”候选，能否在
`262144/class`、两个随机种子上稳定优于同预算 AutoND/DBitNet？

本实验只回答该直接性能确认问题。K1-U 已经通过正确/错误 S 盒对照完成机制
归因，并且位置不变分支在 seed 3/4 上分别达到 `0.977201/0.974682` AUC，略优
于原生位置分支。因此 K1-CA 不重复 wrong-S-box，不重新搜索位置表示，也不把
AutoND 项目协议比较写成其公开代码的精确复现。

## 2. Same-Budget Anchor And One Variable

- 上游候选锚点：K1-U 的
  `runtime_spn_ct_k1t_position_histogram_invariant`，`65536/class`，seed 3/4。
- 同预算基线：`autond_dbitnet2023`。
- 冻结数据任务：uKNIT-BC r5、cell11、输入差分
  `0x0000400000000000`、`4 pairs/sample`、加密随机明文负样本、固定密钥训练、
  跨密钥验证。
- 唯一规模变化：候选从 `65536/class` 提高到 `262144/class`。
- 新增比较轴：在相同数据、训练轮数和优化设置下加入 AutoND/DBitNet。

这不是 K1-BT/K1-BU 的恢复运行。K1-BT 的冲突 run id 和缓存复用失败继续保留
为内部审计；K1-CA 使用全新 run id、K1-U 选出的四 pair 位置不变候选和独立
证据门。

## 3. Frozen Four-Row Matrix

| Row | Seed | Model | Train | Cross-key validation | Pairs | Epochs |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 3 | invariant structure expert | 262144/class | 65536/class | 4 | 10 |
| 2 | 3 | AutoND/DBitNet | 262144/class | 65536/class | 4 | 10 |
| 3 | 4 | invariant structure expert | 262144/class | 65536/class | 4 | 10 |
| 4 | 4 | AutoND/DBitNet | 262144/class | 65536/class | 4 | 10 |

每行训练总行数为 `524288`，验证总行数为 `131072`。两个 seed 的训练/验证
密钥沿用 K1-U：seed 3 使用 `0x44...44` / `0x55...55`，seed 4 使用
`0x66...66` / `0x77...77`。

禁止增加 wrong-S-box、MCND、Liu Conv2D、其他内部 SPN 网络、额外 seed、额外
轮数、额外 pair、额外 epoch、不同差分或不同密钥。

## 4. Dataset Resource Contract

每个 seed 只生成一次训练数据和一次验证数据，然后由 AutoND 复用候选模型的
缓存：

```text
cache creations = 2 seeds x 2 splits = 4
cache reuses    = 2 seeds x 2 splits x 1 baseline = 4
final-test cache creations/reuses = 0
```

四份缓存必须全部位于：

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\
  i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803\cache
```

缓存必须包含 `features.npy`、`labels.npy`、`metadata.json`，以 1024 行分块
生成、单 worker、持续写入 `progress.jsonl`，并按完整参数匹配复用。出现第五份
缓存、少于四次 AutoND 复用、任何 final-test cache 或非 `G:\lxy` 项目路径均
使协议门失效。

## 5. Frozen Gates

协议门要求：

- 四个且仅四个计划行、结果行和非空 best checkpoint；
- source commit 与启动 pin 完全一致，训练来自 GitHub `main` 上的精确提交；
- `262144/class` 训练、`131072` 总验证行、4 pairs、10 epochs；
- `final_test_repeats=0` 且 `final_test_samples_total=null`；
- 四次缓存创建、四次完成、四次 AutoND 参数匹配复用、零 final-test cache；
- 数据与 checkpoint 路径均在本次唯一 run root；
- 磁盘缓存、chunk size 1024、worker 1、CUDA、best `val_auc` checkpoint；
- progress 中存在 `run_done`。

在看到 K1-CA 结果前冻结研究门，每个 seed 同时要求：

```text
invariant candidate AUC >= 0.900
invariant candidate AUC - AutoND AUC >= +0.100
```

两颗 seed 全部通过时，状态为 `pass`，可写为“在本文冻结项目协议和双 seed
`262144/class` 预算下保持显著优势”。任一研究门未通过但协议完整时，状态为
`hold`，如实报告结果，不把 K1-U 的机制归因外推为规模性能优势。

无论 `pass` 或 `hold`，K1-CA 完成后论文实验阶段均封板。只有协议无效时才允许
修复同一四行矩阵的证据绑定；协议有效但研究门未通过，不能通过新增模型、seed、
数据、epoch、pair、轮数、最终测试或超参数搜索救门。

## 6. Execution And Retrieval

- 训练设备：远程 `lxy-a6000` 的物理 GPU0；本地只做解析、模型构建和门控测试。
- 代码来源：精确 GitHub 已推送提交；run-owned detached clean clone。
- 项目数据边界：所有 source、schedule、cache、logs、checkpoints、results 和
  archive 均在 `G:\lxy`。
- Windows 调度：只使用 `cmd.exe /c`，启动后禁用未来的一次性触发。
- 启动后只做一次有界确认：run root、started marker、readiness/progress。
- 后续由本地 tmux monitor 等待，优先回收验证结果分支；不可用时从指定 run root
  原始回收并标为 fallback。
- 回收后校验 `SHA256SUMS`、source commit、4 行结果、本地重裁决并刷新
  `outputs/00_RECENT_RESULTS.md/json`。

远程 `--no-plot` 式后处理不导入 Matplotlib。K1-CA 图表仅在完整结果本地回收后
生成，并按 `visual-qa-redraw` 工作流检查像素渲染，再写入论文。

## 7. Completion And Paper Update

完整回收后在本文件补充 run 状态、来源类型、四行 AUC、逐 seed 优势、门控状态、
归档路径和明确的论文写法。同步更新：

- `paper/chinese-core-innovation1/codex_manuscript.md`
- `paper/chinese-core-innovation1/claim_evidence_matrix.md`
- `outputs/00_RECENT_RESULTS.md`
- `outputs/00_RECENT_RESULTS.json`

推荐下一动作固定为论文定稿：整理 K1-CA 表格与视觉检查通过的图，保留“项目协议、
双 seed、非精确 AutoND 复现、非百万正式规模、非统一网络通吃所有 SPN”的证据
边界，不再启动新的 Innovation 1 训练。

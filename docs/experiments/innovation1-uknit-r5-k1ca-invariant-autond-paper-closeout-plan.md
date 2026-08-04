# Innovation 1 uKNIT K1-CA Invariant/AutoND Paper Closeout Plan

**Date:** 2026-08-03
**Status:** complete; fallback raw retrieval locally validated; protocol and research gates pass; K1-CA training closed
**Run ID:** `i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803`

## 1. Research Question

在不改变 uKNIT-BC 5 轮差分任务、密钥、负样本、优化器、训练轮数和
pair 数的前提下，K1-U 选出的“正确 S 盒语义 + 位置不变聚合”候选，能否在
`262144/class`、两个随机种子上稳定优于同预算 AutoND/DBitNet？

本实验只回答该直接性能确认问题。K1-U 已经通过正确/错误 S 盒对照完成机制
归因，并且位置不变分支在 seed 3/4 上分别达到 `0.977201/0.974682` AUC，略优
于原生位置分支。因此 K1-CA 不重复 wrong-S-box，不重新搜索位置表示，也不把
AutoND 项目协议比较写成其公开代码的精确复现。

2026-08-03 作者进一步明确：K1-CA 的四行矩阵只完成候选与 AutoND 的主规模
锚定，不能以 K1-BZ 的小规模筛选代替论文主表中的同规模公开网络比较。因此
K1-CA 本身仍按已启动提交和四行协议原样完成，不增加或重启任何行；协议有效
完成后允许执行一次独立 K1-CB，复用 K1-CA 的四份训练/验证缓存，只训练
Zhang/Wang MCND、Liu raw Case-3 Conv2D 和 Gohr-style ResNet 三种公开架构
适配器。K1-CB 不改变 K1-CA 的结果或 gate，仅与其结果合并形成论文比较表。

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

无论 `pass` 或 `hold`，K1-CA 的候选/AutoND 性能确认阶段均封板。只有协议无效
时才允许修复同一四行矩阵的证据绑定；协议有效但研究门未通过，不能通过新增
seed、数据、epoch、pair、轮数、最终测试或超参数搜索救门。作者随后授权的
K1-CB 是独立的论文公开网络比较，不是救门：它无论 K1-CA 数值高低都使用同一
冻结协议，只复用缓存并报告三种公开架构的实际结果。

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

K1-CA 完成后的下一动作更新为 K1-CB 公开网络同规模比较；该比较必须先验证 K1-CA
四份缓存完整且参数匹配，并在缺失时失败关闭而非重新生成。K1-CB 完成后再整理
合并表格与视觉检查通过的图，保留“项目协议、双 seed、公开架构适配而非精确
复现、非百万正式规模、非统一网络通吃所有 SPN”的证据边界，并封板转入论文定稿。

## 8. Launch Record

2026-08-03 已从 GitHub `main` 上精确提交
`ad8581a7b790d138de5f270725a872441e4f29e3` 启动。独立 launch gate 为
`pass`，其上游 K1-U SHA、位置不变选择、论文资源合同、四行计划、remote
readiness、source assets、受保护工作区和 GitHub 远端 SHA 检查全部通过。

远程使用物理 GPU0，启动前没有 Python 训练进程，目标 run root 和同名计划任务
均不存在。由于历史约定项目路径及名为 `v1-clean` 的目录都不是 Git 仓库，未对
其执行 pull、reset 或覆盖；改为从 GitHub 建立本次专用干净启动克隆：

```text
G:\lxy\blockcipher-structure-adaptive-nd-k1ca-launch-ad8581a7
```

该克隆 detach 到精确提交后，通过已提交的 `cmd.exe /c` 启动器创建 run-owned
source 和训练根：

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\
  i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803
```

唯一一次启动后远程确认显示 Git revision、Git status、GPU 和 torch 日志已经
创建，Windows 计划任务结果 `0x41301` 表示任务正在运行；确认时 started marker、
readiness 和 progress 尚未生成，因此状态只能记为 `running / pre-readiness`，
不能记为结果完成或训练行已经开始。

本地 tmux 会话 `i1_uknit_k1ca_closeout_monitor` 已接管后续等待和回收，监视根为：

```text
outputs/remote_results_incomplete/
  i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803_monitor/
```

主线程不再 SSH 轮询。下一动作是由 monitor 等待 verified result branch 或 raw
fallback archive，随后自动校验清单、重裁决四行结果并刷新 recent-results 索引。
在完整结果回收前不生成论文数值、图表或优势结论，也不启动任何额外实验。

## 9. Completion Record (2026-08-04)

K1-CA 已在远程 A6000 完成，并从远程运行根以 fallback raw retrieval 方式
回收到本地。回收来源不是完整验证结果分支，因此证据状态必须写为“原始回收并
本地重裁决”，不能简写为 verified result branch。source commit 为：

```text
ad8581a7b790d138de5f270725a872441e4f29e3
```

本地证据入口为：

```text
outputs/remote_results_incomplete/
  i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803/
```

四个冻结行、四个非空检查点、每行完整 10 epoch history、`run_done`、计划一致性
验证和本地 gate 全部通过。缓存合同实际值为四次创建、四次完成、四次 AutoND
参数匹配复用和零 final-test cache，与预注册完全一致。逐 seed 结果如下：

| seed | 位置不变结构专家 AUC | AutoND/DBitNet AUC | 差值 | 结构专家 accuracy | AutoND accuracy |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.978828491 | 0.500461395 | +0.478367096 | 0.968933105 | 0.501846313 |
| 4 | 0.980237826 | 0.501199730 | +0.479038095 | 0.964027405 | 0.500175476 |

本地 gate 状态为 `pass`，决策为
`innovation1_uknit_k1ca_invariant_advantage_supported`。可写范围限定为：在冻结
uKNIT-BC r5、4 pairs/sample、`262144/class`、双 seed 项目协议下，位置不变
结构专家稳定优于 AutoND/DBitNet 适配器。该结果不是 AutoND 公开代码的精确
复现、百万级正式 benchmark、充分超参数搜索、完整轮攻击、SOTA、密钥恢复突破
或统一网络通吃所有 SPN 的证据。

K1-CA 的结果曾授权冻结的 K1-CB 只读复用四份缓存；K1-CB 现亦已完成。下一动作
更新为：停止 K1-CA/K1-CB 及其任何机械扩样，把五模型结果、图表和证据边界写入
论文后进入期刊格式适配与投稿前审计。

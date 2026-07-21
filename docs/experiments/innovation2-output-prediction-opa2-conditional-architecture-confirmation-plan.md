# 创新2 OPA2：PRESENT三轮候选架构第四密钥匹配控制确认

日期：2026-07-21

状态：正式seed3已完成 / verified result branch回收 / 独立架构确认通过

## 1. 启动条件

OPA2不能手工选择架构。唯一合法输入是OPA1正式`gate.json`，并且必须同时满足：

```text
decision = innovation2_selected8_architecture_candidate_requires_confirmation
metrics.selected_candidate_for_phase_b = 一个非MLP候选
metrics.candidate_gates[候选].passed = true
```

任一条件不满足，OPA2不得生成训练矩阵；保留OP11的专用八输出MLP，并停止在OPA1 seed2测试集上
继续选模型或调参。

## 2. 唯一研究问题

> OPA1发现的非MLP候选，能否在第四把独立固定秘密密钥上，同时超过同预算MLP和自身匹配标签打乱
> 控制？

OPA2只确认三轮固定八输出架构，不测试四轮、XOR、完整密文恢复、真假分类或中间状态递推。

## 3. 冻结协议

```text
cipher                 = PRESENT-80
rounds                 = 3
seed / fixed key       = 3，第四把独立固定秘密密钥
train rows             = 131072 total plaintext/ciphertext pairs
test rows              = 65536 total disjoint plaintext/ciphertext pairs
input                   = 64 MSB-first plaintext bits
target positions        = [0, 2, 8, 10, 32, 34, 40, 42]
sample classification   = false
epochs                  = 100 per model
batch                   = 250
optimizer               = RMSprop
learning rate           = 0.001
loss                    = raw-output MSE
selection               = final epoch
```

候选模型的结构和超参数必须与OPA1完全相同。第四密钥上不得重新选择输出位置、候选模型、epoch、损失、
优化器或参数预算。

## 4. 四行矩阵

```text
selected8_mlp_true_output
selected8_mlp_label_shuffle
selected8_<OPA1候选>_true_output
selected8_<OPA1候选>_label_shuffle
```

两条shuffle行使用同一个固定训练行排列；测试标签始终是真实八输出bit。预期32条逐bit结果、400条
epoch历史和4个checkpoint。

## 5. 最终架构门

每个架构至少4/8位置必须同时满足：

```text
true AUC >= 0.510
accuracy - majority >= +0.005
true AUC - same-architecture shuffle AUC >= +0.005
```

候选架构优先还必须满足：

```text
候选至少4/8位置通过三门
mean(candidate true AUC - MLP true AUC) >= +0.003
mean[(candidate true - candidate shuffle) - (MLP true - MLP shuffle)] >= +0.003
至少4/8位置 candidate true AUC - MLP true AUC >= +0.002
```

通过时才可写成“该候选在两把新增独立密钥的受控筛选/确认流程中优于MLP”。未通过则保留MLP，
关闭本轮架构扩展。无论结果如何，OPA2都不自动重开OP12四轮XOR。

## 6. 执行与停止项

本地smoke仅使用`64 train / 64 test / 1 epoch / CPU`验证由OPA1 gate驱动的候选身份、共享shuffle、
缓存、训练、checkpoint、32条结果和中文图，不作性能结论。正式实验只能在A6000运行，并沿用OPA1的
磁盘缓存、断点和verified-result-branch回收规则。

每个OPA2运行必须单独保存`phase_a_gate.json`，并在协议检查中要求该gate的全部`protocol_checks`和
`execution_checks`为真。`summary.json`同时嵌入同一gate，确保候选身份、授权门和最终结果可以离线
复核，不能只依赖命令行参数或聊天记录。

禁止：

- 手工覆盖OPA1候选；
- OPA1未过门时运行OPA2；
- 在seed3重新选择bit或候选；
- 只比较true行而省略两条匹配shuffle；
- 因OPA2通过直接宣称高轮、完整密文恢复或SOTA；
- OPA2失败后增加seed、模型、数据或epoch补救。

## 7. OPA1授权与本地实现门

OPA1 raw-fallback gate已通过完整性验证并唯一选择：

```text
candidate_architecture = present_spn
OPA1 candidate mean AUC = 1.000000000
OPA1 candidate - MLP mean AUC = +0.468343205
OPA1 protocol/execution checks = all pass
```

OPA2本地smoke：

```text
run_id = i2_output_prediction_opa2_present_r3_selected8_present_spn_smoke_20260722
train/test = 64/64 total pairs
epochs = 1
result rows = 32/32
history rows = 4/4
checkpoints = 4/4
protocol/execution checks = all pass
decision = innovation2_selected8_architecture_confirmation_local_smoke_passed
```

smoke图已通过`visual-qa-redraw`像素检查；tiny AUC不作性能结论。正式下一步冻结为第四固定密钥
`seed3`的四行`present_spn/MLP true/shuffle`远程确认。若PRESENT-SPN未同时通过自身shuffle和相对
MLP增益门，则OPA1的AUC=1只保留为未复现的第三密钥发现，最终架构仍为MLP。

## 8. 正式远程启动

正式确认于`2026-07-22`从已推送提交启动：

```text
source commit = 3ddd346e70fb68f5d143a3253639acf10f33f4b3
run_id = i2_output_prediction_opa2_present_r3_selected8_present_spn_key3_gpu0_20260722
remote run root = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opa2_pspn_k3_20260722
physical GPU = 0
status = running
```

一次性启动确认已验证：source commit精确匹配、readiness=`pass`、started marker存在，且
`progress.jsonl`与磁盘缓存`data/cache_metadata.json`均已写入。旧的远程主克隆含历史修改，未被重置或
用于启动；本次使用独立短路径干净克隆。后续结果等待本地tmux watcher
`i2_opa2_pspn_k3_watch_20260722`自动回收，主线程不重复SSH轮询。

## 9. 正式结果与裁决

远程训练于`2026-07-22T02:46:17+08:00`完成，随后由本地watcher从verified result branch
回收。来源和产物完整性为：

```text
source commit = 3ddd346e70fb68f5d143a3253639acf10f33f4b3
result rows = 32/32
history rows = 400/400
checkpoint hashes = 4/4
disk-cache completed rows = 196608/196608
protocol checks = all true
execution checks = all true
OPA2 gate SHA256 = 97943c59a8d88f8bbc1b6845aa6372a8b91d5693e0abd3a23e63b41259601284
```

第四固定密钥的平均结果为：

| 架构 | 真实输出平均AUC | 匹配shuffle平均AUC | true-shuffle | accuracy-majority | 通过bit |
|---|---:|---:|---:|---:|---:|
| MLP | 0.532262231 | 0.498853553 | +0.033408678 | +0.021852493 | 8/8 |
| PRESENT-SPN-aware | 1.000000000 | 0.500839804 | +0.499160196 | +0.499097824 | 8/8 |

候选相对MLP平均AUC增益为`+0.467737769`，扣除两种架构自身shuffle增益后的调整增益仍为
`+0.465751518`；八个预注册位置全部满足候选相对MLP至少`+0.002`的逐bit门。最终：

```text
status = pass
decision = innovation2_selected8_architecture_priority_independently_confirmed
```

允许的结论是：在PRESENT三轮、两把新增独立固定密钥、八个预注册真实密文输出bit和冻结预算下，
PRESENT-SPN-aware架构相对同预算MLP及自身匹配标签打乱得到独立确认。它仍不是四轮、高轮、完整
密文恢复、跨密钥总体统计或SOTA证据。

正式`curves.svg`已经以1920像素宽度渲染并通过`visual-qa-redraw`检查；标题、热图数字、四个面板、
坐标、`AUC=1.0000`标签和底部证据边界均无重叠、裁切或歧义。结果已刷新为
`outputs/00_RECENT_RESULTS.md`的`001`。

### 推荐下一步

OPA2确认的是整个架构，不足以把增益归因给精确PRESENT拓扑。下一步只运行已预注册的OPA3：在
同一第四密钥、同一数据、同一初始化和同一约388万参数预算下，比较精确P-layer、identity-P和固定
错误P。OPA3通过才开放受控PRESENT四轮门；不通过则保留OPA2整体架构结果并停止拓扑与轮数扩展。

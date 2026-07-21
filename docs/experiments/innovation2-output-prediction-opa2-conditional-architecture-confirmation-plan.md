# 创新2 OPA2：PRESENT三轮候选架构第四密钥匹配控制确认

日期：2026-07-21

状态：条件式预注册 / 等待OPA1正式门

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

# 创新2 OPA1：PRESENT三轮固定八输出多架构发现屏

日期：2026-07-21

状态：本地实现门通过 / 远程Phase A已调度、启动尚未确认

## 1. 研究问题

OP11已经在第二把固定秘密密钥上确认，专用八输出MLP可预测八个预注册真实密文bit，平均AUC为
`0.530900037`。OP9只比较过完整64输出的MLP和Kimura式LSTM，不能回答专用八输出任务是否更适合
卷积、全局注意力或PRESENT结构感知网络。

OPA1只回答：

> 在第三把独立固定秘密密钥、相同八个输出位置、相同数据和训练预算下，哪一种网络归纳偏置最值得
> 进入第四把密钥的匹配shuffle独立确认？

OPA1是候选发现屏，不是最终架构裁决。它不测试四轮、XOR、完整密文恢复、真假分类或中间状态递推。

## 2. 冻结数据与训练协议

```text
cipher                 = PRESENT-80
rounds                 = 3
seed / fixed key       = 2，第三把独立固定秘密密钥
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
remote device           = lxy-a6000 physical GPU0
```

该任务没有正负类别，不使用`/class`。八个位置来自seed0 OP10并在seed1 OP11确认；seed2上不得重新
选择或排序。训练和测试明文必须唯一且零重合。明文、完整64-bit目标和秘密密钥按块落盘，缓存参数严格
匹配后才能复用；每个模型逐epoch保存latest checkpoint并支持恢复。

## 3. Phase A五模型矩阵

```text
selected8_mlp_true_output          64 -> 1936 -> 1936 -> 8 MLP anchor
selected8_lstm_true_output         六层hidden300 Kimura式LSTM
selected8_rescnn_true_output       10个残差块、保留绝对位置的一维bit序列CNN
selected8_transformer_true_output  7层全局自注意力编码器
selected8_present_spn_true_output  三个共享S-box局部块 + 精确P-layer token重排
```

五行都预测相同八个真实密文输出bit。本阶段不训练shuffle，避免在候选未知时把五模型矩阵扩大为十行。
参数量以MLP约389万为锚，各候选相对差距必须不超过`3%`。所有模型使用相同数据、epoch、batch、
优化器、学习率、损失和final-epoch选择。

PRESENT-SPN-aware候选把64个明文bit表示为token；每个块先把连续四个bit作为一个S-box单元，共享
局部MLP，再按公开PRESENT P-layer进行固定、不可训练的token重排。它没有看到密钥或中间状态。

## 4. 发现门与选择规则

每个模型、每个位置报告AUC、0.5阈值准确率、多数类准确率、两者差值、MSE和numpy-rint非法率。

非MLP候选只有同时满足以下条件才进入OPA2：

```text
至少4/8位置：AUC >= 0.510 且 accuracy-majority >= +0.005
八位置平均AUC - MLP八位置平均AUC >= +0.003
至少4/8位置：候选AUC - MLP AUC >= +0.002
```

若多个候选通过，先按八位置平均AUC排序；差值小于`0.001`时依次按平均准确率优势、固定架构顺序
`LSTM -> ResCNN -> Transformer -> PRESENT-SPN-aware`打破平局。该顺序在揭盲前冻结。

Phase A最高分只能产生`candidate_for_independent_confirmation`，不得直接写成架构更优。若没有非MLP
候选通过，保留MLP为当前锚点并停止本轮架构扩展；不得从同一seed2测试集反复调超参数。

## 5. 条件式Phase B

若OPA1有候选通过，OPA2使用第四把独立固定秘密密钥，只训练四行：

```text
winner true output
winner matched label shuffle
MLP true output
MLP matched label shuffle
```

只有OPA2同时复现相对MLP增益且胜过同架构shuffle，才允许形成架构优先结论。OPA2不得重新选择
八个bit、候选架构或超参数。

## 6. 实现门、产物和远程规则

本地smoke仅使用`64 train / 64 test / 1 epoch / CPU`检查实现，不作性能结论，预期40条结果、5条
history和5个checkpoint。正式Phase A预期40条结果、500条history和5个checkpoint。每次运行生成：

```text
results.jsonl
history.csv
metadata.json
summary.json
gate.json
checkpoint_manifest.json
progress.jsonl
data/cache_metadata.json
curves.svg（本地检索后生成）
```

smoke通过后范围提交并推送，从推送提交在`G:\lxy`的run-owned干净克隆启动。缓存、checkpoint、日志
和结果全部位于`G:\lxy\blockcipher-structure-adaptive-nd-runs`。启动命令使用`cmd.exe /c`；本地tmux
watcher负责回收，主线程不重复SSH轮询。SVG必须通过`visual-qa-redraw`真实像素检查，随后刷新最近
结果索引。

## 7. 停止项

- 不在seed2重新选八个bit；
- 不混入不同数据量、epoch、损失、优化器或checkpoint选择；
- 不把Phase A最高AUC直接写成架构结论；
- 不一次训练五模型及全部shuffle共十行；
- 不因Phase A结果直接重开已经失败的OP12四轮XOR；
- 不把三轮部分bit预测包装成完整密文恢复、高轮攻击或SOTA。

## 8. 本地实现门结果与下一步

本地运行：

```text
run_id = i2_output_prediction_opa1_present_r3_selected8_architecture_screen_position_preserving_smoke_20260721
train/test = 64/64 total pairs
epochs = 1
device = CPU
result rows = 40/40
history rows = 5/5
checkpoints = 5/5
protocol checks = all pass
decision = innovation2_selected8_architecture_screen_local_smoke_passed
```

五模型参数量：

```text
MLP          = 3,891,368
LSTM         = 3,978,008  (+2.23%)
ResCNN       = 3,955,904  (+1.66%)
Transformer  = 3,939,016  (+1.22%)
PRESENT-SPN  = 3,879,415  (-0.31%)
```

smoke只证明五模型、真实输出标签、seed2缓存、checkpoint、40条结果、裁决和中文可视化链路可执行；
64条测试数据的AUC不作性能解释。SVG已经渲染为像素并通过`visual-qa-redraw`检查，没有文字重叠、
裁切、缺字、含混标题或不可读的近距离曲线。

推荐下一步是从本次实现的推送提交，在A6000启动冻结的`131072/65536/100 epochs`五模型Phase A。
理由是本地门已经排除协议和实现阻塞，而架构优劣问题需要同预算正式数据才能回答。远程结果若没有
非MLP候选通过三项预注册门，则保留MLP并停止在seed2调参；若有候选通过，只开放seed3四行匹配
shuffle确认，不开放四轮XOR、更多模型或机械扩样本。

远程交接记录：

```text
initial source commit = 7d4e45de5a98aa297a585103e8b6542e1ce73e13
repaired source commit = 7ee17cfb7befb4731663f28a2956795b0e4373d0
remote run id = i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721
remote path = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run-id>
initial launch = failed before task creation because Task Scheduler /TR exceeded 261 characters
repair = short wrapper G:\lxy\scheduled-runs\i2_opa1_key2.cmd
retry state = scheduled; bounded confirmation saw repaired pinned commit but not yet started/readiness markers
local watcher = tmux i2_opa1_arch_screen_watch_20260721
```

因此当前仍不得报告为`running`。watcher确认started/readiness或回收结果后，再将状态更新为`running`或
`completed remotely`；主线程不重复SSH轮询。

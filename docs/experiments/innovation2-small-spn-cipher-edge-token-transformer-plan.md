# 创新2 E35：小状态SPN Cipher Edge-Token Transformer计划

日期：2026-07-18

状态：已运行 / 旧rolled-P控制协议无效 / 由E35b公平控制重裁决

## 1. 研究问题

E33、E33-R和E34依次审计了绝对位置GraphGPS、cell等变GraphGPS和按真实轮数循环的
共享GraphGPS。三者都没有稳定超过ID边际和错误P-layer。共同限制是P-layer只作为
neighbor gather索引，输出query不能直接读取一条密码边，也不能建模edge-edge关系。

TokenGT将图编码为`n+m`个node/edge token并使用全局attention。E35不复制大型通用
TokenGT，而是构造固定小图、密码结构专用的候选：

```text
Cipher Edge-Token Transformer（CETT）
```

研究问题：

```text
显式P-layer edge token与output-mask query交互，
能否在未见S-box/P-layer上建立真实拓扑贡献？
```

## 2. 候选结构

每条样本形成37个token：

```text
16 node tokens
16 directed P-layer edge tokens
4 shared-S-box cell relation tokens
1 output-mask query token
```

编码：

```text
node = lane role + [active bit, output-mask bit] + round context
P-edge = MLP(source node, destination node) + directed-edge type
S-box relation = MLP(cell-node mean) + shared truth-table encoding + S-box type
query = mask-weighted node pool + active-weighted node pool + query type
```

37个token送入小型标准Transformer encoder，最终只读取query token输出logit。没有绝对
bit ID、nibble ID、variant ID、LapPE或cipher ID。P-layer边是显式token，不再只是消息
传递的索引。

## 3. 冻结数据与拆分

```text
label source = i2_small_spn_exact_label_width_16ciphers_256keys_seed0_20260718
cell source  = i2_small_spn_matched_contrast_readjudication_20260718
selected base cells = 589
total rows          = 9424
train topology      = S0..S2 x P0..P2
unseen S-box        = S3 x P0..P2
unseen P-layer      = S0..S2 x P3
dual unseen         = S3 x P3
cell split seed     = 33001
```

heldout不参与训练、checkpoint、调参或token设计选择。E35不使用basis encoder；当前坐标
结构的active mask与单位向量basis信息等价，避免同时重开SCGT分支。

## 4. 冻结预算与矩阵

完整预算：

```text
hidden dimension = 64
Transformer layers = 3
attention heads = 4
FFN dimension = 128
dropout = 0.10
optimizer = AdamW
learning rate = 1e-3
weight decay = 1e-4
batch size = 128
epochs = 40
checkpoint = train-domain validation AUC
```

矩阵：

| 行 | seed | 作用 |
|---|---|---|
| CETT + true P-layer | 0,1 | 主候选 |
| CETT + wrong P-layer | 0,1 | 拓扑归因控制 |
| CETT + label shuffle | 0 | 流程控制 |

冻结锚点：

```text
ID边际 dual              = 0.726528
E33-R最佳神经锚点 dual    = 0.711548
E34共享处理器 dual        = 0.683643
```

readiness smoke使用hidden32、2层、8 epochs、相同三行seed0，只验证实现，不参与裁决。
整个实验本地CPU执行，不使用远程GPU。

## 5. 表示契约

必须通过：

1. token数固定为`16+16+4+1=37`；
2. P-edge token的destination由该variant真实/错误P-layer决定；
3. 没有绝对bit/nibble/variant位置embedding；
4. 同时重标号cell、P-layer、active和mask时query logit最大误差`<=1e-6`；
5. 打乱P-layer只能改变edge token关系，不能改变标签、split或其他输入；
6. E33--E34已有模型默认行为保持不变。

## 6. 裁决门

全部通过才保留CETT：

```text
label-shuffle dual AUC <= 0.60
true P 两seed dual AUC均 > 0.726528
true P mean dual >= 0.756528
true P mean dual >= wrong P mean + 0.03
true P unseen-S mean >= 0.765693
true P unseen-P mean >= 0.732532
cell重标号最大logit误差 <= 1e-6
```

通过：`innovation2_small_spn_cipher_edge_token_confirmed`。只开放真实密码迁移readiness，
仍不直接启动远程规模训练。

过边际但不过wrong P：`innovation2_small_spn_cipher_edge_token_not_attributed`。

其他有效失败：`innovation2_small_spn_cipher_edge_token_not_ready`。这将关闭当前合成
benchmark上的神经拓扑架构搜索，回到标签族/任务设计；不继续Mamba、KAN、大型
Transformer或更多edge-token变体。

协议失败：`innovation2_small_spn_cipher_edge_token_protocol_invalid`，只修实现。

## 7. 产物

```text
readiness = outputs/local_smoke/i2_small_spn_cipher_edge_token_smoke_seed0_20260718/
full      = outputs/local_diagnostic/i2_small_spn_cipher_edge_token_seed0_seed1_20260718/
results.jsonl
history.csv
gate.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

证据只属于16-bit合成SPN topology attribution，不能写成PRESENT/GIFT/SKINNY高轮结果、
真实攻击或SOTA比较。

## 8. 原始运行与协议纠正

原始run：

```text
smoke = i2_small_spn_cipher_edge_token_smoke_seed0_20260718
full  = i2_small_spn_cipher_edge_token_seed0_seed1_20260718
```

真实P dual为`0.695197/0.648338`，rolled-P控制为`0.795281/0.850497`，label-shuffle为
`0.484934`。但运行后审计确认rolled-P把heldout P3替换为train-seen P2，控制难度不匹配。
因此原full结果状态改为：

```text
status   = fail
decision = innovation2_small_spn_cipher_edge_token_protocol_invalid
```

真实P低于ID边际的原始数值保留为诊断，但wrong-P差值不得解释。后续执行独立E35b，使用
保持P-family身份的固定destination corruption并从头重跑完整矩阵。

# 创新2 E33：小状态SPN GraphGPS/SCGT 同预算归因计划

日期：2026-07-18

状态：已完成 / hold / 不开放真实密码迁移

## 1. 研究问题

E32b已经构造出9424行train-only matched-contrast标签，最强
`round+structure+mask`边际在unseen-S、unseen-P、dual-unseen上分别为
`0.775693/0.742532/0.726528`，全部低于冻结停止线。E33第一次训练结构网络，回答：

```text
读取真实S-box truth table和P-layer edge的网络，
能否在未见S-box/P-layer上超过同一train-derived ID边际，
且优于错误P-layer与label-shuffle控制？
```

E33只验证合成SPN上的算法迁移，不是实际密码攻击。

## 2. 冻结来源与拆分

```text
label source = i2_small_spn_exact_label_width_16ciphers_256keys_seed0_20260718
cell source  = i2_small_spn_matched_contrast_readjudication_20260718
selected base cells = 589

train topology block = S0..S2 x P0..P2  (9 variants, 5301 rows)
unseen S-box         = S3 x P0..P2       (1767 rows)
unseen P-layer       = S0..S2 x P3       (1767 rows)
dual unseen          = S3 x P3           (589 rows)
```

三个heldout split只评价，不参与cell选择、fit/validation、early stopping或超参数选择。
9个train topology内按固定cell seed把589个base cell拆成80% fit / 20% validation；同一
base cell的9个variant行必须整体进入同一侧，防止cell泄漏。

## 3. 输入

每条样本只使用公开结构描述：

```text
16-bit S-box truth table：16组4-bit input/output
16-bit P-layer directed permutation
round count：2..5
coordinate input-subspace basis / active-bit mask
16-bit linear output mask
```

不得输入标签统计、train_positive_count、ATM/CLAASP输出、cipher ID one-hot或heldout
标签。variant ID只用于索引对应的truth table和edge，不作为embedding。

## 4. 三行主矩阵

### B0：deterministic marginal baseline

不训练。沿用E32b的train-derived `round+structure+mask` lookup：

```text
unseen-S     AUC = 0.775693
unseen-P     AUC = 0.742532
dual-unseen  AUC = 0.726528
```

### N1：small GraphGPS-style + mask query

16个bit node；局部消息包含同S-box cell mean、真实P-layer incoming/outgoing edge；全局
分支使用小型multi-head self-attention。S-box truth table由共享MLP编码并广播；round、
active bit和output mask是node feature。mask-weighted query readout输出一个logit。

### N2：SCGT

复用N1 cipher encoder，增加SetTransformer/AllSet-inspired basis encoder：把规范basis
row作为无序token，self-attention后pool，并注入graph readout。E32结构的basis是active
bit单位向量，因此本门只能判断set branch是否有增益，不能证明一般RREF子空间有效。

若N2不优于N1，不保留basis branch；不得因“更创新”而忽略消融结果。

## 5. 必要控制

```text
C1 = N1-shuffled-P：每个variant使用冻结的错误P-layer映射，其他完全相同
C2 = N1-label-shuffle：只打乱train-domain标签，heldout仍用真标签
```

C1判断收益是否来自真实拓扑；C2判断训练/选择流程是否会制造虚假AUC。控制属于归因
重跑，不扩成更多候选架构。

## 6. 固定预算

```text
hidden dimension       = 64
GraphGPS blocks        = 3
attention heads        = 4
dropout                = 0.10
optimizer              = AdamW
learning rate          = 1e-3
weight decay           = 1e-4
batch size             = 128
epochs                 = 40
checkpoint selection   = train-domain validation AUC
main seeds             = 0,1 for N1 and N2
control seeds          = 0,1 for C1; seed0 for C2
device                 = local CPU by default; CUDA allowed if locally available
```

先运行seed0、8 epochs、hidden32、2 blocks的readiness smoke；smoke只证明数据、forward、
checkpoint和split有效。通过后自动运行冻结完整矩阵。数据只有9424行，属于本地小实验，
不使用远程GPU。

## 7. 裁决门

先计算每行两seed mean和单seed结果：

```text
topology_predictor_ready:
  protocol checks全部通过；
  C2 dual-unseen AUC <= 0.60；
  N1两seed dual-unseen均 > 0.726528；
  N1 mean dual-unseen >= 0.756528（baseline + 0.03）；
  N1 mean dual-unseen >= C1 mean + 0.03；
  N1在unseen-S和unseen-P的mean均不低于各自baseline - 0.01。
  -> 真实topology predictor在合成SPN成立；进入E34真实密码迁移readiness。

topology_signal_not_attributed:
  N1超过边际但未超过C1，或C2异常高。
  -> 不声称密码拓扑贡献；审计表示与拆分。

topology_predictor_not_ready:
  N1未超过dual baseline + 0.03或任一seed不超过baseline。
  -> 停止，不增加层数或训练轮数。

protocol_invalid:
  source、cell拆分、heldout隔离、forward、checkpoint或metric失败。
  -> 只修协议。
```

SCGT单独保留门：

```text
SCGT mean dual-unseen >= N1 mean + 0.01
```

否则记录`basis branch = discard`，但不影响N1自身裁决。

## 8. 产物与停止边界

```text
results.jsonl
history.csv
gate.json
metadata.json
progress.jsonl
curves.svg
```

禁止：

```text
不在看见heldout结果后改hidden/layers/epoch；
不输入cipher ID embedding；
不把SCGT名称当作有效性证据；
不把合成SPN AUC与PRESENT论文准确率横向宣称SOTA；
不在E33直接启动真实密码远程训练。
```

权威完整run规划：

```text
outputs/local_diagnostic/i2_small_spn_graphgps_scgt_seed0_seed1_20260718/
```

## 9. 实际执行与结果

就绪smoke：

```text
run_id   = i2_small_spn_graphgps_scgt_smoke_seed0_20260718
decision = innovation2_small_spn_topology_training_readiness_passed
```

冻结完整矩阵：

```text
run_id = i2_small_spn_graphgps_scgt_seed0_seed1_20260718
rows   = 7
epochs = 40
seeds  = 0,1（label-shuffle仅冻结seed0）
```

heldout AUC 的seed均值：

| 方法 | unseen-S | unseen-P | dual-unseen |
|---|---:|---:|---:|
| ID边际 | 0.775693 | 0.742532 | 0.726528 |
| GraphGPS真实拓扑 | 0.834885 | 0.726330 | 0.682672 |
| SCGT真实拓扑 | 0.819871 | 0.731912 | 0.726947 |
| GraphGPS错误P-layer | 0.826803 | 0.713394 | 0.752444 |
| GraphGPS标签打乱 | 0.549020 | 0.549165 | 0.465781 |

逐seed关键值：

```text
GraphGPS true dual = 0.667115 / 0.698229
SCGT true dual     = 0.733843 / 0.720051
wrong P-layer dual = 0.803457 / 0.701431
```

最终裁决：

```text
status   = hold
decision = innovation2_small_spn_topology_predictor_not_ready
```

GraphGPS在dual-unseen上比ID边际低`-0.043856`，且比错误P-layer控制低
`-0.069772`。SCGT只比ID边际高`+0.000419`，虽然相对GraphGPS高
`+0.044275`并触发预注册的basis-branch内部保留标记，但没有通过整体拓扑归因门。
label-shuffle接近随机，说明失败不是训练流程凭空制造高AUC。

## 10. 证据边界与推荐下一步

E33只证明当前绝对位置GraphGPS/SCGT没有在合成SPN上建立真实拓扑贡献。它不否定
所有图网络，也不支持扩大层数、epoch、样本或seed。真实PRESENT/GIFT/SKINNY迁移门
保持关闭。

下一步执行E33-R单变量表示审计：保持标签、split、预算、优化器和控制不变，只把
`bit+nibble+lane`绝对位置编码替换为保持S-box cell重标号对称性的lane-role编码。
候选必须先通过确定性cell重标号不变性测试，再在dual-unseen上超过ID边际与错误
P-layer控制各`0.03`，否则停止该GraphGPS表示路线。不得把E33-R变成新的架构搜索。

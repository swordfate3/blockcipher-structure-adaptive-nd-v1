# 创新2 E65：PRESENT四轮单位输出平衡谱重排与架构路由计划

日期：2026-07-18

状态：已完成 / pass / prefix引导profile operator readiness开放

## 1. 研究问题

E43--E51把每个`(活动结构, 输出mask)`当作一条二分类查询。该协议语义严格，但一次只预测
一个mask，与创新2“预测输出积分性质”目标仍有距离。E65只改变输出组织方式：从E43原始
atlas抽取64个unit mask，把同一活动结构的标签重排成一个64维平衡谱：

```text
input  = PRESENT-80拓扑 + rounds=4 + 8-bit活动结构
target = 64个输出bit各自是否对所有key/offset保持XOR平衡
value  = positive / negative / unknown
```

E65不重新生成标签，不把unknown写成negative，不训练网络。它先判断unit谱能否形成
structure-disjoint、行列边际平衡、输出坐标覆盖充分的masked multi-output benchmark。

## 2. 冻结来源与单变量

```text
source = E43 i2_present_r4_universal_balance_atlas_20260718
cipher = PRESENT-80
rounds = 4
structures = 96，按E43 index mod 4冻结train/validation
unit output masks = mask_000 ... mask_063
label semantics = E43 sound positive certificate / concrete negative witness / unknown
```

唯一变化是把unit-mask查询重排成结构级64维目标。密码、轮数、活动结构、证书、反例、split、
unknown语义和metric都不改变。E51关闭的是继续枚举逐查询r4 processor；E65是一次明确的
输出契约诊断例外，不为已有失败模型增加容量或训练预算。

## 3. 反捷径重排

只在unit-mask `96 x 64`标签矩阵上重新执行E43冻结的checkerboard selector：

```text
             output bit a   output bit b
structure x       1              0
structure y       0              1
```

每条边最多使用一次；每个被选结构和输出bit在各自split内正负精确平衡。输出数组保存为：

```text
profile_targets.npy   = 96 x 64 int8，未观察位置为-1
profile_observed.npy  = 96 x 64 bool
```

validation使用的每个输出bit必须在train出现，但validation structure必须与train完全互斥。

## 4. 同预算确定性基线

E65在选中的相同行上，用train-only ridge、同一`lambda=1e-3`比较：

```text
static set                 不使用P-layer
fair-corrupted topology    错误P-layer可达特征
true topology              正确P-layer可达特征
ANF r1--r3 prefix          不读取第4轮最终证书
```

这一步只做架构路由，不把确定性AUC称为神经结果。禁止第4轮full-cube oracle、certificate
status、witness、key/offset parity或validation拟合。

## 5. 预注册门

协议与宽度：

```text
E43 source/hash/label/unit-mask replay          pass
profile shape                                  96 x 64
train positive/negative                        each >= 150
validation positive/negative                   each >= 50
train/validation structures                    >= 48 / >= 16
train/validation output coordinates            >= 24 / >= 16
validation outputs subset of train outputs     yes
duplicate edges                                0
each selected structure/output class delta     0
strongest global/mask/active marginal AUC       <= 0.55
```

架构路由：

```text
topology profile route:
  true topology AUC >= 0.60
  true - corrupted >= 0.03
  true >= ANF prefix - 0.02

prefix-guided profile route:
  ANF prefix AUC >= 0.60
  ANF prefix - true topology >= 0.02
```

两条同时满足时选validation AUC更高者。通过只允许E66本地两轮readiness：一次前向输出64个
logit、masked BCE、共享round block和nodewise decoder；比较拓扑无关profile anchor、true-P
operator与同参数fair-corrupted-P control。E66不得直接复用E65确定性标签或最终证书为输入。

不过门则停止该输出谱神经结构，不恢复RCCA、不扩大small-SPN模板、不启动远程GPU。

## 6. 产物与声明范围

```text
run_id = i2_present_r4_unit_balance_profile_readiness_20260718
output = outputs/local_audits/i2_present_r4_unit_balance_profile_readiness_20260718
```

产物包括`matched_unit_contrast.csv`、`profile_targets.npy`、`profile_observed.npy`、
`features.csv`、JSONL/JSON、进度和SVG。声明范围仅为PRESENT-80四轮严格unit标签的多输出
重排与架构路由；不是训练结果、高轮积分区分器、新攻击或SOTA。

## 7. 2026-07-18实际结果

E43的`96 x 64 = 6144`个unit标签逐项重放，source hash、三态标签、64个singleton mask和
structure split全部通过。重排后的masked profile为：

```text
train       178 positive + 178 negative，50个structure，32个output bit
validation   60 positive +  60 negative，18个structure，23个output bit
shared output bits = 23 / 23 validation bits
observed profile coordinates = 476
```

边唯一、每个被选structure和output bit正负精确平衡，global/mask/active边际最强AUC均为
`0.500000`。同一train-only ridge结果：

| 特征族 | train AUC | structure-disjoint validation AUC |
|---|---:|---:|
| static set | `0.577989` | `0.514722` |
| fair-corrupted P可达 | `0.645326` | `0.695694` |
| true P可达 | `0.717144` | `0.704306` |
| ANF r1--r3 prefix | `0.765749` | `0.793611` |

```text
true topology - corrupted topology = +0.008611
ANF prefix - true topology          = +0.089306
```

正确拓扑本身有信号，但未满足`true-corrupted >= 0.03`；ANF prefix超过正确拓扑`0.0893`，
通过预注册prefix-guided route：

```text
status   = pass
decision = innovation2_present_unit_balance_profile_prefix_ready
remote   = no
```

## 8. 裁决与推荐下一步

执行E66两轮本地readiness，测试`Prefix-Guided Nodewise Profile Operator`。固定E65的
`96 x 64`目标、observed mask和structure split，一次前向输出64个logit并用masked BCE。
所有神经行接收相同的安全r1--r3 prefix node features；比较：

```text
1. 每个output node独立的共享MLP（无跨输出消息）
2. true-P共享轮次nodewise profile mixer
3. 同参数fair-corrupted-P profile mixer
```

唯一候选增量是沿输出节点关系交换消息；不得输入第4轮certificate、witness、label派生值，
不得改变E65匹配或split。E66只检查张量契约、masked loss、参数公平、true/corrupted表示差异、
输出重标号等变性和两轮有限训练，不用两轮AUC声称提升。readiness通过后才预注册30轮seed0
正式矩阵，要求同时接近/超过E65 prefix ridge、领先独立MLP并领先错误P；失败则停止该结构。

最终`curves.svg`经`visual-qa-redraw`渲染为`2200 x 1161`像素检查；标题、宽度柱、AUC柱、
阈值线、覆盖说明、裁决和证据边界均无重叠、裁切、缺字或模糊归因。

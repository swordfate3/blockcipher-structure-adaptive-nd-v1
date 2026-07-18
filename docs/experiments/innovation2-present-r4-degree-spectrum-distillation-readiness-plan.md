# 创新2 E49：PRESENT四轮中间degree-spectrum蒸馏readiness计划

日期：2026-07-18

状态：完成 / hold / 中间谱蒸馏未过门

## 1. 研究问题

E45的ANF 1--3轮prefix ridge达到`0.686082`，E48的degree-only ridge达到`0.689170`；E47
MSPN true只有`0.518673`且train AUC为`0.794375`。E48又否定了变量身份碰撞是主要原因。
E49测试一个更窄的机制问题：给MSPN的中间状态加入训练期degree-spectrum辅助监督，能否
让它在structure-disjoint validation上真正学会确定性传播，而不是只记住train structure。

这不是把E45特征作为输入，也不是开放新的大模型。它只改变训练目标，MSPN主干、输入、
balance head、数据split和预算全部保持不变。

## 2. 数据与辅助target

```text
cipher             = PRESENT-80
rounds             = 4
source             = E43 matched checkerboard
train              = 800 rows（400/400）
validation         = 236 rows（118/118）
split              = structure-disjoint，完全沿用E43/E47
input              = active bits + output mask + S-box ANF + P-layer
primary target     = universal balance 0/1 label
auxiliary target   = r1/r2/r3各13维ANF-prefix，共39维
```

每轮13维由`support mean/max/sum/union + degree0..8 distribution`组成，完全复用E45的确定性
定义。只允许从active bits、output mask、S-box ANF和所选P-layer计算；不得读取第4轮
full-cube计数、certificate status、witness、label或最终oracle。

辅助target可在validation上用于只读误差评估，但validation target不得参与梯度、模型选择
或超参数选择。最终balance分类器不得读取39维teacher target或辅助head输出。

## 3. 固定模型与预算

```text
backbone            = E47 MSPN
hidden              = 32
degree channels     = 9
shared steps        = 4
dropout             = 0.10
epochs              = 2
batch               = 32
optimizer           = AdamW(lr=1e-3, weight_decay=1e-4)
seed                = 0
checkpoint          = best validation balance AUC
device              = local CPU
auxiliary loss      = mean MSE over r1/r2/r3 normalized 13-d targets
auxiliary scale     = 0.25（冻结，不调参）
```

每个shared step后用同一个小型共享head从output-mask query pool预测对应13维target。第4步不
产生辅助target。参数增长必须不超过E47 MSPN的`1.15x`，避免用容量解释readiness差异。

## 4. 最小矩阵与控制

只训练三行，并读取E47锚点：

```text
0. E47 MSPN true label-only                         只读锚点
1. MSPN true P + true degree-spectrum distillation 候选
2. MSPN true P + train-row-shuffled spectrum        同预算安慰剂
3. MSPN fair-corrupted P + corrupted spectrum       自洽错误transport控制
```

第2行只在train split内用seed `49001`固定打乱39维target的row对应，保持每列分布、辅助loss
计算、head参数和训练预算；validation仍按真实target报告误差，但不参与训练。第3行从同一
active/mask用fair-corrupted P-layer计算自洽prefix target，避免拿真实teacher故意惩罚错误
transport。三行分类label、初始化seed、batch顺序、optimizer和checkpoint协议相同。

## 5. Readiness门

协议门：

```text
E43/E45/E47/E48 run id、decision、hash与关键metric匹配
teacher shape                               = rows x 3 x 13
teacher features finite and normalized      = pass
final-round oracle/certificate/witness input = absent
teacher/output leakage into balance head    = absent
shared auxiliary-head parameters            <= 0.15 x E47 MSPN
三行均完成2 epochs且loss/gradient/metric finite
```

辅助可学门以validation MSE衡量，先按39维train target的方差标准化，再跨维平均：

```text
true distilled validation normalized MSE        <= 0.90
true distilled - shuffled-target validation MSE <= -0.10
```

两轮balance只作为防退化readiness信号，不声称候选有效：

```text
true distilled validation AUC        >= 0.48
label-only E47锚点                    = 0.518673（只读）
```

全部通过：允许另建E50 30轮seed0正式计划，仍需比较label-only、target-shuffle、fair-corrupted
和E45 ridge。辅助可学门失败：停止degree-spectrum蒸馏和当前证书传播神经路线。辅助可学但
两轮balance退化：检查loss scale一次，但不得从validation搜索scale；只能使用预先定义的
`0.10`敏感性审计，另建计划后执行。

## 6. 执行边界与产物

E49是本地readiness，不使用远程GPU，不运行seed1，不迁移r5，不增加hidden、层数、token、
Transformer或NBFNet。readiness通过也不构成有效神经预测结论。

```text
outputs/local_smoke/i2_present_r4_degree_spectrum_distillation_readiness_seed0_20260718/

results.jsonl
history.csv
teacher_metrics.csv
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

声明范围：PRESENT-80四轮、E43严格标签、两轮本地中间degree-spectrum辅助监督readiness；
不是高轮积分区分器、新攻击、远程规模证据或SOTA。

## 7. 2026-07-18实际结果

权威run：

```text
i2_present_r4_degree_spectrum_distillation_readiness_seed0_20260718
```

E43/E45/E47/E48 source、hash、split、teacher、target shuffle、参数、泄漏、有限梯度和三行两轮
流程检查全部通过。模型contract：

```text
teacher shape                         = 1036 x 3 x 13
auxiliary prediction shape            = 8 x 3 x 13
E47 MSPN parameters                   = 17788
E49 distilled parameters              = 18281
parameter ratio                       = 1.027715
shared auxiliary-head parameters      = 493
auxiliary-head direct balance delta   = 0.0
balance head width unchanged          = true
teacher/prefix/certificate buffers    = absent
```

最佳checkpoint结果：

| 行 | validation balance AUC | validation teacher normalized MSE |
|---|---:|---:|
| E47 label-only只读锚点 | `0.518673` | 不适用 |
| 正确P + 真谱蒸馏 | `0.475151` | `0.797746` |
| 正确P + train-target打乱 | `0.465384` | `0.850731` |
| 错误P + 自洽错误谱 | `0.453246` | `0.735733` |

真谱MSE低于绝对门`0.90`，但相对target打乱只改善：

```text
true - target-shuffle normalized MSE = -0.052985
required                             <= -0.10
```

同时真谱balance AUC为`0.475151`，低于防退化线`0.48`，也低于E47 label-only `0.518673`。
因此不能把MSE下降解释成足够的组外中间状态学习：

```text
status   = hold
decision = innovation2_present_degree_spectrum_not_learned
E50 formal MSPN = no
seed1           = no
remote          = no
```

## 8. 推荐下一步

停止当前证书传播神经路线：不运行30轮蒸馏、不调`auxiliary_scale`、不增加MSPN宽度/深度，
不恢复identity token，也不迁移r5或远程GPU。E45/E48约`0.69`的确定性ANF/degree结果与E44
triangle `0.561979`神经结果应作为两类独立锚点保留。

下一架构问题改为“安全确定性前端 + 神经残差”，而不是“神经网络重新发明传播算法”：

```text
base     = train-only E45 ANF-prefix ridge（冻结）
residual = E44 64-bit pair-state processor
target   = 只学习ridge未解释的balance残差
controls = prefix-only residual + fair-corrupted-P pair residual
```

下一步冻结E50两轮本地实现readiness。E45 39维前缀只来自1--3轮S-box/P-layer/active/mask，
允许作为显式安全特征；第4轮oracle、certificate、witness仍禁止。只有base复现、残差零初始化
等价、参数匹配、错误P敏感、无泄漏和三行有限训练全部通过，才另建30轮正式计划。E50
readiness不以两轮AUC声称提升；NBFNet、Transformer、seed1和远程仍关闭。

最终`curves.svg`按`visual-qa-redraw`渲染为`1800×936`像素；首次曲线图因三行接近而改成
最终checkpoint柱状对比，精确标注AUC/MSE、冻结门、旧锚点和截断纵轴。最终标题、图例、
柱值、坐标、裁决与导出边界无重叠、裁切、缺字或误导性缩放，已记录
`visual_qa_passed.marker`。

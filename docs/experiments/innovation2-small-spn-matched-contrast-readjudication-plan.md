# 创新2 E32b：小状态SPN训练内matched-contrast重裁决计划

日期：2026-07-18

状态：确定性postprocess完成 / pass / E33合成SPN网络比较门开放

## 1. 研究问题

E32的57344个全key精确标签具有足够正负宽度和126个签名，但
`round+structure+mask`边际在未见拓扑上达到约0.984--0.987 AUC。E32b不更改密码、
密钥、结构、mask或标签，只改变一个变量：训练集cell选择规则。

问题是：只保留训练拓扑块内部已经观察到cipher-dependent contrast的cell后，heldout
标签是否仍不能被相同ID边际解释。

## 2. 冻结来源

```text
source run = i2_small_spn_exact_label_width_16ciphers_256keys_seed0_20260718
source artifact = labels.npy + metadata.json + gate.json
source decision = innovation2_small_spn_exact_label_shortcut_dominated
```

必须验证来源shape为 `16 x 4 x 14 x 64`、全部256-key协议门通过、source gate与run id
一致。不得重新运行cipher或改标签。

## 3. 唯一选择规则

沿用E32拆分：9个train topology、3个unseen-S、3个unseen-P和1个dual-unseen。
对每个base cell `(round, structure, mask)`，只用9个train标签计算：

```text
train_positive_count
```

冻结选择：

```text
1 <= train_positive_count <= 8
```

即train内部必须同时含正负类。heldout标签不得参与选择、阈值或门槛设定。该规则保持
原始binary target不变，只删除被train一致边际完全决定的cell。

## 4. 预注册规模与控制

预审仅用于冻结预计规模：

```text
selected base cells       = 589
train label rows          = 5301
unseen-S rows             = 1767
unseen-P rows             = 1767
dual-unseen rows          = 589
```

postprocess必须重新计算并保存：

1. selected cell index与`round/structure/mask`字段；
2. 四个split正负数量；
3. 16-topology label pattern数量；
4. global、mask、round+mask、structure+mask、round+structure+mask AUC；
5. raw E32与matched-contrast最强边际AUC并排；
6. selection只读取train的代码级证明与测试fixture。

## 5. 裁决门

```text
matched_contrast_ready:
  source与选择协议全部有效；
  selected base cells >= 512；
  train positive和negative均 >= 2000；
  unseen-S与unseen-P各类均 >= 500；
  dual-unseen各类均 >= 200；
  distinct 16-topology patterns >= 128；
  unseen-S strongest ID marginal AUC <= 0.80；
  unseen-P strongest ID marginal AUC <= 0.80；
  dual-unseen strongest ID marginal AUC <= 0.75。
  -> 创建E33三行同预算神经训练计划；先本地smoke。

matched_contrast_still_shortcut_dominated:
  宽度通过但任一AUC停止线失败。
  -> 停止当前合成benchmark，不训练网络。

matched_contrast_too_narrow:
  cell、split类别或topology pattern不足。
  -> 停止，不放宽到读取heldout标签选cell。

protocol_invalid:
  source、shape、选择只读train、索引或标签重算失败。
  -> 只修postprocess。
```

## 6. 执行与停止边界

E32b只读取现有`labels.npy`，属于本地CPU确定性重裁决，无训练、无远程GPU、无新数据。

```text
不读取heldout标签决定cell是否保留；
不从1..8改为2..7或3..6挑最好AUC；
不改变停止线；
不把matched selection称为真实密码攻击；
不在E32b同时实现神经模型。
```

权威run规划：

```text
outputs/local_audits/i2_small_spn_matched_contrast_readjudication_20260718/
```

## 7. 完整结果

所有来源、shape、train-only选择和selected索引门通过。冻结规则得到：

```text
selected base cells                = 589
selected total label rows          = 9424
distinct 16-topology patterns      = 336

train            positive/negative = 2742 / 2559
unseen S-box     positive/negative = 1053 / 714
unseen P-layer   positive/negative = 887 / 880
dual unseen      positive/negative = 360 / 229
```

最强train-derived ID边际AUC从原始E32降为：

| split | 原始E32 | matched E32b | 停止线 |
|---|---:|---:|---:|
| unseen S-box | 0.987248 | 0.775693 | 0.80 |
| unseen P-layer | 0.983952 | 0.742532 | 0.80 |
| dual unseen | 0.986291 | 0.726528 | 0.75 |

宽度门与三项捷径门全部通过：

```text
status       = pass
decision     = innovation2_small_spn_matched_contrast_ready
training     = no（E32b本身未训练）
remote_scale = no
```

该pass只开放E33合成SPN神经比较，不开放真实PRESENT/GIFT高轮训练或突破声明。

`curves.svg` 已按原生交付尺寸对应的约 `1424x759` 像素执行
`visual-qa-redraw`。标题、选择规则、split柱状图、原始到matched AUC连线、停止线、图例、
裁决和证据范围均无重叠、裁切或歧义。

## 8. 推荐下一步

创建E33三行同预算计划并先本地smoke：

```text
1. train-derived round+structure+mask deterministic baseline
2. small GraphGPS-style cipher topology encoder + mask query
3. SCGT = AllSet-style structure encoder + GraphGPS topology encoder + mask query
```

训练数据只能使用9个train topology的5301行；unseen-S、unseen-P和dual-unseen只评价。
必须加入label-shuffle与P-layer-shuffle控制。只有真实topology模型在dual-unseen上优于
`0.726528`边际且显著优于shuffled topology，才继续真实密码迁移门。

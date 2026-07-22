# 创新2 OPE1：八个冻结密文bit全异或为一个bit的输出预测可行性门

日期：2026-07-22

状态：本地实现门与r3诊断完成 / hold / r4诊断未授权 / 远程训练停止

## 1. 研究问题

用户提出把此前确认的八个PRESENT真实密文输出bit全部异或为一个bit，再直接预测这个bit：

```text
selected8 = [0, 2, 8, 10, 32, 34, 40, 42]，MSB-first
z(P) = C[0] xor C[2] xor C[8] xor C[10]
       xor C[32] xor C[34] xor C[40] xor C[42]
```

其中`C=PRESENT_K^r(P)`，`K`是固定但不提供给网络的秘密密钥，`P`是测试阶段未见明文。标签`z`虽为
`0/1`，仍是同一条明文真实密文的确定输出值，不是真假样本、积分平衡类别或正负样本分类。

OPE1只回答两个问题：

1. 三轮时，这个八bit parity目标能否被直接网络学到，证明实现和目标至少有校准信号？
2. 保持同一密钥、mask、数据预算、网络和控制不变，只把轮数改为四轮后，信号是否仍存在？

OP12的四个2-bit和两个4-bit XOR正式结果为`0/6`通过、直接平均AUC `0.499943`，所以八bit全XOR是
低先验假设，只允许本地可行性诊断，不能直接启动A6000正式规模。

## 2. 数据与目标

```text
cipher / key protocol = PRESENT-80 / seed1第二固定未知秘密密钥
rounds                = 3与4，分开训练但除轮数外协议相同
train/test            = 4096 / 4096 total明密文对
input                 = 64个MSB-first明文bit
target                = selected8真实密文bit的全XOR值
epochs / batch        = 10 / 128
optimizer / loss      = RMSprop / raw-output MSE
selection             = final epoch
device                = local CPU
sample_classification = false
```

训练和测试明文必须唯一、零重合；真实64-bit密文目标必须由PRESENT实现逐条抽样回放；数据缓存写入
`plaintexts.npy`、`features.npy`、`full_targets.npy`和`cache_metadata.json`，参数匹配时才允许复用。

## 3. 为什么不能在原八bit内重排出几何控制

八个bit全部参与XOR时，交换分组或顺序不会改变函数：

```text
a xor b xor ... xor h = 任意重排后的同一异或值
```

因此“把八个selected bit重新配对”不是控制，而是与候选完全相同的标签。OPE1预注册一个整体右移一位、
相对形状相同但坐标不同的同重量真实输出mask：

```text
candidate = [0, 2, 8, 10, 32, 34, 40, 42]
control   = [1, 3, 9, 11, 33, 35, 41, 43]
```

不得在结果揭盲后换控制mask、枚举64-bit组合或挑最高parity。

## 4. 五模型最小归因矩阵

```text
selected8_mlp_true_output             八个组成bit，生成派生parity与最佳组成bit基线
selected8_parity_mlp_true_output      同目标MLP架构锚点
selected8_parity_rescnn_true_output   直接八bit parity候选
control8_parity_rescnn_true_output    同重量右移mask控制
selected8_parity_rescnn_label_shuffle 与候选同构，只打乱训练标签
```

本地模型使用`hidden=256`的两层MLP和`channels=36, blocks=10`的ResCNN。单输出MLP为`82689`参数，
单输出ResCNN为`82441`参数，差约`0.30%`且必须不超过`0.5%`；三个ResCNN行结构、初始化、训练顺序
和预算完全一致。标签打乱只改变训练
标签顺序，测试标签始终是真实selected8 parity。

从八个组成bit原始分数裁剪到`[0,1]`后计算独立派生parity：

```text
P(z=1) = (1 - PRODUCT_j(1 - 2*p_j)) / 2
```

直接parity必须超过该派生值和最佳组成bit，才支持“网络直接学到组合输出函数”。

## 5. 冻结门

三轮校准至少满足：

```text
direct ResCNN AUC >= 0.520
accuracy - majority >= +0.005
direct - matched shuffle AUC >= +0.010
```

四轮可行性必须同时满足：

```text
direct ResCNN AUC >= 0.510
accuracy - majority >= +0.005
direct - matched shuffle AUC >= +0.005
direct - shifted geometry control AUC >= +0.005
direct - selected8-derived parity AUC >= +0.005
direct - best component-bit AUC >= +0.002
direct - same-task MLP AUC >= +0.002
```

本地结果只是诊断，不是正式失败或paper-scale证据。若三轮校准和四轮联合门都通过，下一步只准备
A6000 `2^17/2^16`、100 epochs、disk-cache正式计划，仍需单独审计后才能启动。若任一门接近随机或被
控制解释，则停止八bit全parity路线，不扩样本、epoch、模型、mask或轮数。

## 6. 产物与下一动作

每个轮数必须输出：

```text
results.jsonl
history.csv
metadata.json
summary.json
gate.json
checkpoint_manifest.json
progress.jsonl
data/cache_metadata.json
curves.svg
```

结果完成后必须执行`visual-qa-redraw`像素检查并刷新`outputs/00_RECENT_RESULTS.md`和JSON索引。最终记录
必须给出r3/r4候选、MLP、shuffle、shifted control、derived parity、best component的AUC和全部差值，
以及证据支持的唯一下一动作。

## 7. 实现门与三轮校准结果

r3与r4的`64/64`、1 epoch CPU实现门均完成：五模型、真实parity回放、右移mask控制、参数匹配、
`12`条结果、`5`条history、`5`个checkpoint及SVG产物全部有效。小样本数值不作性能判断。

随后按冻结计划运行r3本地诊断：

```text
run_id = i2_output_prediction_ope1_present_r3_r4_selected8_parity_r3_diagnostic_seed1_20260722
train/test = 4096 / 4096 total明密文对
models/epochs = 5 / 10
results/history/checkpoints = 12 / 50 / 5
status = hold
decision = innovation2_selected8_parity_r3_not_calibrated
```

主要AUC：

```text
直接八bit parity ResCNN = 0.493084934
同目标MLP               = 0.502388388
右移一位同重量控制       = 0.500595572
匹配标签打乱             = 0.500994665
八个单bit派生parity      = 0.482824027
最佳组成bit              = 0.535317081
```

直接候选差值与准确率门：

```text
candidate - MLP          = -0.009303453
candidate - control      = -0.007510638
candidate - shuffle      = -0.007909730
candidate - derived      = +0.010260907
candidate - best bit     = -0.042232146
accuracy - majority      = -0.021728516
r3 calibration gates     = 0 / 3 passed
```

八个组成bit的MLP AUC约为`0.514--0.535`，但把它们全部异或后，直接ResCNN、同目标MLP和匹配控制均
回到随机附近。候选不仅没有超过shuffle与右移mask，还低于同目标MLP和最佳组成bit；这说明单bit弱信号
没有在八bit全parity中叠加，反而被组合函数消掉。

按预注册停止门，r3校准失败后不运行r4性能诊断，因为更高轮结果不能弥补低一轮同目标校准失败，也
不能授权远程扩样本。当前裁决只覆盖本地`4096/4096`、seed1固定密钥、256-hidden/36-channel的受控
诊断，不是数学上证明所有八bit parity、所有密钥或正式规模永远不可预测。

三张SVG均执行了`visual-qa-redraw`：r3诊断检查`1920x1171`和`1280x781`，r3/r4实现门检查1280宽
渲染；中文标题、AUC局部缩放、差值标签、八个组成bit和逐项门均无重叠、裁切或结构歧义。

## 8. 证据支持的下一动作

```text
selected8 full-parity remote scale = no
r4 diagnostic                       = no
more epochs/models/masks            = no
higher rounds                        = no
```

下一步不再围绕这八个bit机械组合。若继续寻找高轮输出函数，应先做独立的密码结构/代数审计，要求在
看神经测试结果前给出一个机制导出的输出mask，并明确它为什么可能在增加轮数后保留低复杂度；只有该
审计同时提供同重量控制和可验证的低轮校准预期，才新建训练计划。论文当前保留OP11三轮八bit正结果，
把OP12四轮2/4-bit XOR与OPE1三轮8-bit全parity作为“简单异或组合不能提高预测轮数”的两级负边界。

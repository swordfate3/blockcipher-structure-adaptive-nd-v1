# 创新2 OPF2-C1：PRESENT四轮输出预测单变量新密钥条件确认计划

日期：2026-07-23

状态：条件计划冻结 / 数据配置支持准备中 / 未授权readiness、训练或远程启动

## 1. 条件授权

本计划只在以下OPF2正式结果全部成立后开放：

```text
run_id   = i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_key7_gpu0_20260722
status   = pass
decision = innovation2_position_bound_r4_scale_supported
results/history/checkpoints = 40 / 500 / 5
cache rows = 1114112 complete
source commit、SHA256、切分、protocol和execution checks全部通过
```

若OPF2为`hold`或协议无效，本计划保持关闭。`hold`分支只能按已盲预注册的OPF3计划执行，不得把本计划
改造成补充seed、挑bit或追加epoch的救援实验。

## 2. 为什么必须拆分key seed

当前Kimura数据协议中的`seed`同时控制秘密密钥、明文随机流、模型初始化、训练标签打乱和batch顺序。
因此直接从`seed=7`改为`seed=8`会同时改变五类变量，不能称为“原样更换一把固定未知密钥”。

OPF2-C1冻结为：

```text
data/model/shuffle seed = 7
key_seed                = 8

seed7 secret key = ba48ad2b9c5e584afccf
seed8 secret key = 2bfd091f77163138de86
```

`seed=7`继续控制明文、非连续切分、模型初始化、标签打乱和batch顺序；只有PRESENT-80秘密密钥由
`key_seed=8`生成。这样OPF2与OPF2-C1之间唯一改变的研究变量才是固定未知密钥。

## 3. 冻结数据与训练协议

```text
cipher / rounds        = PRESENT-80 / 4
input                  = 64个MSB-first明文bit
targets                = [0, 2, 8, 10, 32, 34, 40, 42]八个真实密文输出bit
sample classification  = false
train rows             = 1048576 = 2^20 total
test rows              = 65536 = 2^16 total
train indices          = [0,131072) U [196608,1114112)
test indices           = [131072,196608)
epochs / batch         = 100 / 250
optimizer / lr         = RMSprop / 0.001
loss                   = raw-output MSE
selection              = final epoch
```

必须逐值证明OPF2与OPF2-C1的`plaintexts.npy`和`features.npy`完全相同，训练/测试索引与以下旧测试
保护hash不变：

```text
OPF1 train plaintext raw SHA256 = eca0f5705c2d9a6b4f0475bfb90e55d2bfa2d5e4d7b8c380b10ab55778a4555a
OPF1 test plaintext raw SHA256  = 5c5410d4c0761f729f5f705d43a7392bf90f6ae0bee65a57321760d515b82fec
```

`full_targets.npy`必须由seed8密钥重新生成并与OPF2目标hash不同。缓存元数据必须同时记录`seed=7`和
`key_seed=8`；默认`key_seed=None`时必须继续复用不含`key_seed`字段的历史缓存，不能破坏OPF2恢复。

## 4. 冻结五模型与确认门

为使新密钥结果与OPF2逐行可比，原五模型完整复用：

```text
全局头ResCNN
无P位置头ResCNN
真实P位置头SPN-ResCNN
错误P位置头SPN-ResCNN
真实P位置头SPN-ResCNN + 训练标签打乱
```

确认主门保持不变：

```text
真实P平均AUC >= 0.55
真实P - 标签打乱平均AUC >= 0.03
真实P平均accuracy-majority >= 0.005
至少4/8 bit同时满足：
  AUC >= 0.55
  candidate - shuffle >= 0.015
  accuracy-majority >= 0.005
```

若OPF2与OPF2-C1均通过，才允许表述为“两把独立固定未知密钥上的PRESENT四轮八bit输出预测得到确认”，
并据此设计五轮条件实验。若OPF2-C1未通过，保留OPF2为单密钥条件结果，停止直接进入五轮和机械增加密钥。
任何明文、初始化、训练顺序、标签、切分、来源门或缓存不一致均使结果协议无效，而不是确认失败。

## 5. 实施与停止边界

OPF2揭盲前只允许：

1. 在Kimura数据配置中增加默认关闭的独立`key_seed`，保持旧缓存和默认训练hash兼容。
2. 单元测试相同数据seed下明文/特征相同、密钥/真实输出不同，以及旧默认缓存可复用。
3. 保留本条件计划，不生成正式数据、不跑readiness、不实现远程启动包。

OPF2正式通过后才允许补齐PositionBound runner的`key_seed=8`传播、五模型readiness、缓存与来源门、
远程配置和启动脚本。readiness必须先验证`40/5/5`小规模产物、初始化逐参数相同、目标逐值回放和中文SVG；
随机小样本指标不得解释。

明确停止：

```text
OPF2未通过时不得运行OPF2-C1
确认前不得进入PRESENT五轮
不得同时改seed、明文、bit位置、模型、epoch、batch、loss或optimizer
不得追加2^22、300 epochs或事后选择更容易的输出bit
```

## 6. 当前下一步

保持OPF2远程训练和本地watcher运行。OPF2正式回收并验证后执行唯一分支：

```text
OPF2 pass -> 实现并运行OPF2-C1 readiness，然后从推送提交远程确认
OPF2 hold -> 关闭OPF2-C1，按OPF3计划执行共享逐轮SPN递推器readiness
OPF2 invalid -> 只修复OPF2协议，不运行任何条件分支
```

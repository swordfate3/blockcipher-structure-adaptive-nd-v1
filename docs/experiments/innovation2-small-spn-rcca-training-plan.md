# 创新2 E63：小型SPN Relation-Cipher Cross-Attention训练计划

日期：2026-07-18

状态：正式Phase A已完成 / 暂缓 / RCCA关闭

## 1. 研究问题

E62已提供2048个全256主密钥严格多坐标relation模板，标签宽度、拓扑敏感性和反捷径门通过。
E63只测试一个架构假设：

```text
relation coordinate queries与真实S-box/P-layer图做对齐cross-attention，
是否比只对coordinate set做DeepSets聚合更能泛化到未见P-layer和双重未见拓扑？
```

不改变E62标签、split、relation模板、证书或heldout所有权。

## 2. 冻结数据与split

```text
source labels = i2_small_spn_multicoordinate_relation_readiness_20260718
source cipher = i2_small_spn_expanded_topology_4s16p_256keys_20260718
relations     = 2048，3轮1024 + 4轮1024，每条2个不同坐标
labels        = 64 variants x 2048 templates
train variants / unseen-S / unseen-P / dual = 36 / 12 / 12 / 4
```

用固定seed `63001`只在2048个relation模板轴做80/20 fit/validation拆分。checkpoint只看36个训练
拓扑上的validation relation AUC；任何heldout topology AUC都不能用于选epoch。

## 3. 同预算模型

共同输入：variant S-box truth table、真实或control P-layer、round、两个relation coordinates；每个
coordinate包含16-bit active-structure indicator和16-bit output-mask indicator。

### DeepSets

每个coordinate的16个bit node只用lane role、active flag、mask flag和round编码；对bit集合做
mean/active/mask pooling形成coordinate token，再经共享`phi`。两个token无序mean pooling，最后
与密码图全局pool拼接分类。它不做coordinate-to-cipher node对齐交互。

### RCCA

用共享小型SPN graph block编码16个cipher node；每个coordinate的16个bit-query与同index cipher
node对齐相加，再对完整cipher node set做multi-head cross-attention。bit-query按
mean/active/mask pooling成coordinate token，两个coordinate token无序mean后分类。

模型不得使用variant ID、relation ID、绝对bit/cell embedding、exact 256-key parity vector、witness
key index或E62边际分数。只保留4种lane role，确保同时重标号cell node、P-layer和coordinate bits
时输出不变。

## 4. 必要控制

```text
DeepSets true-P         = same-information architecture anchor
RCCA true-P             = candidate
RCCA fair-corrupted-P   = topology attribution control
RCCA label-shuffle      = process control
```

wrong-P沿用E37每条P-layer内部destination-cell rotation，保持bijection和destination lane，不把
heldout P替换成train-seen P。label shuffle只打乱fit/validation训练标签，heldout真标签不变。

## 5. Readiness smoke

```text
run_id = i2_small_spn_rcca_readiness_seed0_20260718
device = local CPU
hidden = 32
layers = 2
heads  = 4
epochs = 8
batch  = 128
rows   = DeepSets true / RCCA true / RCCA wrong-P / RCCA label-shuffle，seed0
```

必须通过：

```text
E62 source gate = pass
E37 source/split/cache ownership = pass
relation token swap max logit error <= 1e-6
cell relabel max logit error <= 1e-6
same-weight true/wrong-P fixture max logit delta >= 1e-5
DeepSets and RCCA parameter counts each <= 300000
larger/smaller parameter ratio <= 1.35
four rows train, checkpoint and all AUC finite
```

smoke指标只验证流程，不用于架构收益结论。失败只修实现契约，不扩大数据或训练预算。

## 6. 正式Phase A

readiness通过后自动运行：

```text
run_id = i2_small_spn_rcca_seed0_seed1_20260718
rows   = DeepSets true seed0/1
         RCCA true seed0/1
         RCCA label-shuffle seed0
hidden64 / 2 layers / 4 heads / dropout0.10
40 epochs / batch128 / AdamW lr1e-3 wd1e-4
device = local CPU
```

冻结强锚点：

```text
E62 strongest dual marginal AUC = 0.6858946784
```

进入Phase B wrong-P前必须：

```text
label-shuffle dual <= 0.60
DeepSets and RCCA true each seed dual > 0.6858946784
RCCA each seed dual > paired DeepSets seed
RCCA mean dual >= 0.7158946784
```

失败则关闭RCCA，不增加hidden、layer、epoch、seed或远程GPU。

## 7. Phase B拓扑归因

仅Phase A通过才训练RCCA fair-corrupted-P seed0/1，同hidden、epoch、optimizer和split。最终保留门：

```text
RCCA true - wrong-P dual > 0 for each seed
mean RCCA true - wrong-P dual >= 0.03
mean RCCA true - mean DeepSets dual >= 0.03
```

通过只能声称：RCCA在16-bit合成SPN全密钥严格relation benchmark上获得可归因的拓扑增益。
不自动迁移为PRESENT/GIFT结果、攻击或SOTA。失败则RCCA架构关闭。

## 8. 产物

每个run必须输出：

```text
results.jsonl
history.csv
gate.json
metadata.json
summary.json
progress.jsonl
checkpoints/
curves.svg
visual_qa_passed.marker
```

每个完成run刷新`outputs/00_RECENT_RESULTS.md/json`。全部本地执行，不使用远程GPU。

## 9. Readiness完成记录

```text
run_id = i2_small_spn_rcca_readiness_seed0_20260718
mode   = smoke / local CPU / hidden32 / 8 epochs
status = pass
decision = innovation2_small_spn_rcca_readiness_passed
```

模型与来源契约：

```text
relation token swap max logit error = 2.9802322388e-08
cell relabel max logit error         = 7.4505805969e-08
true/wrong-P max logit delta         = 9.1859698296e-04
DeepSets / RCCA parameters           = 70465 / 79073
parameter ratio                      = 1.122160
fit / validation relations           = 1638 / 410
```

四行均完成训练、checkpoint和全部split评估：

```text
                         unseen-S    unseen-P    dual
DeepSets true seed0      0.779587    0.664612    0.701076
RCCA true seed0          0.899422    0.641073    0.599471
RCCA wrong-P seed0       0.888960    0.638012    0.678291
RCCA label-shuffle       0.462586    0.471497    0.516807
```

这些是8-epoch readiness诊断，不用于RCCA收益结论。RCCA smoke在dual低于DeepSets且wrong-P高于
true-P，提示正式门可能失败，但预注册流程规定readiness只验证协议；不得据此后验取消或修改正式
矩阵。下一步仍按第6节运行40-epoch seed0/1 Phase A，失败则关闭RCCA。

## 10. 正式Phase A完成记录

```text
run_id = i2_small_spn_rcca_seed0_seed1_20260718
mode   = full / local CPU / hidden64 / 40 epochs
status = hold
decision = innovation2_small_spn_rcca_not_ready
```

五行正式结果：

```text
                         unseen-S    unseen-P    dual       best epoch
DeepSets true seed0      0.834995    0.589084    0.603005   37
DeepSets true seed1      0.827578    0.663007    0.678269   39
RCCA true seed0          0.862550    0.422499    0.526025   40
RCCA true seed1          0.724061    0.554897    0.532732   33
RCCA label-shuffle       0.480148    0.466295    0.515766   9
```

同paired seed的dual差值：

```text
seed0 RCCA - DeepSets = -0.076981
seed1 RCCA - DeepSets = -0.145537
RCCA mean dual        = 0.529378
required mean dual    = 0.715895
E62 marginal anchor   = 0.685895
```

正式协议全部有效：token交换和cell重标号误差为`4.47e-08/8.94e-08`，true/wrong-P fixture
差为`1.44e-03`，参数为`259713/293313`，shuffle控制接近随机。失败不能归因于实现不变量、
标签打乱流程、明显参数失衡或提前checkpoint泄漏。

五个Phase A门中只有`label-shuffle <= 0.60`通过；DeepSets与RCCA均未逐seed超过边际，RCCA也
未逐seed超过DeepSets。因此不启动Phase B wrong-P训练，关闭RCCA，不增加hidden、layer、epoch、
seed或远程GPU。

推荐下一步是E64 exact relation decomposition审计，而不是另一种attention：把E62每个relation
positive分成“两坐标各自都balanced”的trivial conjunction与“两坐标各自nonzero但256-bit parity
向量相等”的nontrivial cancellation；同时评估singleton-status确定性基线。只有nontrivial正负在
所有split都有宽度且该基线不过强，才值得设计新的关系算子。否则E62训练任务本身被重新归类为
单坐标组合问题，创新2保留E39 SPN-PRR方法证据并停止多坐标网络搜索。

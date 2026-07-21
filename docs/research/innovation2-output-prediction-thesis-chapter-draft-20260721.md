# 创新2真实输出值预测：论文实验章节初稿

日期：2026-07-21

状态：OP9--OPB1与OPM1证据已合入 / OPC1正式远程运行中 / 可作为当前论文章节底稿

> 2026-07-22更新：本文已经合入OPA1五模型屏、OPA2/OPA3确认归因、OPB1低秩瓶颈和OPK1跨密钥
> 目标审计。OPA3与OPB1均为`hold`：exact-P与fixed-wrong-P均为`AUC=1.0`，所以本文只保留整体
> SPN式架构与分层扩散结论，不主张精确P-layer因果贡献。OPC1 SPN-ResCNN四行正式矩阵正在远程
> 运行，本文只记录其冻结协议，不提前填写结果。OPM1已经排除“selected bit输入锥更窄”与“单S-box
> 坐标明显更简单”这两个粗粒度解释。

## 1. 研究动机与章节定位

神经区分器判断一个样本是否来自特定密码差分分布，而输出预测攻击尝试从一个真实输入直接恢复对应的
密码输出值，两者回答的安全问题不同。本章研究固定未知秘密密钥下的真实输出值预测：攻击者可以通过
密码接口取得训练明文/密文对，但网络不能直接获得秘密密钥；测试阶段给出训练中未见过的明文，要求
网络预测同一固定密钥下的真实密文输出。

Kimura等人的输出预测工作表明，可将明文到密文建模为多输出回归问题，并以完整输出同时命中概率评价
攻击是否成功。[Kimura et al., 2022]在PRESENT三轮上报告：每把固定秘密密钥使用`2^17`条训练明密文
对和`2^16`条测试明文，100 epochs、batch 250、六层LSTM、hidden 300、MSE与RMSprop；跨多把分别
训练的固定密钥模型，完整64-bit密文预测平均成功概率为`2^-1.30`。

完整64-bit同时恢复是很强的目标，但它也可能掩盖部分输出坐标仍然可预测的现象。因此，本章不把完整
输出失败直接解释为所有密文bit均不可预测，而是提出以下分阶段方法：

```text
完整64输出论文协议校准
  -> 64个位置共享扫描
  -> 易预测位置在独立明文上确认
  -> 固定位置在第二把秘密密钥上复现
  -> 专用小输出头与完整输出头、标签打乱对照
  -> 预注册多bit XOR四轮扩展与停止门
  -> 五模型同预算筛选与独立密钥确认
  -> exact/identity/wrong-P及低秩瓶颈归因
  -> ResCNN与SPN路由的同参数混合改造
```

本章已经完成的是`P -> PRESENT_K^r(P)`真实输出预测。它为开题报告中的“神经密文输出预测攻击”提供
固定密钥实验单元和条件式扩轮边界，但尚未实现更强的`state_t -> state_(t+1)`中间状态逐轮迭代攻击。
由于三轮预测单bit、四轮预测结构化XOR，目标函数并不相同，本章也不能冒充同一任务的严格临界轮数
曲线。因此，本章不将当前结果表述为开题报告第二项创新的全部实现。

## 2. 威胁模型与任务定义

设`E_K^r`表示使用固定80-bit秘密密钥`K`的`r`轮PRESENT映射。训练集和测试集分别为：

```text
D_train = {(P_i, C_i) | C_i = E_K^r(P_i)}
D_test  = {(P_j, C_j) | C_j = E_K^r(P_j)}
```

所有明文均唯一，且训练明文与测试明文零重合。输入为明文的64个MSB-first bit：

```text
x_i = bits_msb(P_i) in {0,1}^64
```

本章使用三种输出目标。

完整输出目标：

```text
y_i = bits_msb(E_K^r(P_i)) in {0,1}^64
```

预注册单bit目标：

```text
y_(i,j) = bit_j(E_K^r(P_i)), j in J
```

预注册多bit XOR目标：

```text
z_(i,M) = XOR_{j in M} bit_j(E_K^r(P_i))
```

单bit或XOR目标虽然取值为`0/1`，但`0`和`1`是密码真实输出值，不是“真实样本/随机样本”正负类别。
本章不使用真假样本分类、积分平衡分类、kernel成员分类或关系存在性标签。

## 3. 数据集构造与无泄漏协议

OP9、OP11和OP12均使用磁盘缓存生成明文、输入bit和真实64-bit密文目标。缓存记录密码、轮数、seed、
秘密密钥、bit顺序、训练/测试行数、生成进度和完成状态；参数不一致时禁止复用。正式训练统一使用：

| 项目 | 设置 |
|---|---|
| 密码 | PRESENT-80 |
| 输入编码 | 64个MSB-first明文bit |
| 训练样本 | `131072 = 2^17`条明文/密文对 |
| 测试样本 | `65536 = 2^16`条独立明文/密文对 |
| 样本类别 | 无；不是`/class`任务 |
| 明文约束 | 全局唯一，训练/测试零重合 |
| 密钥协议 | 每个模型训练和测试使用同一固定未知秘密密钥 |
| epoch/batch | 100 / 250 |
| 优化器 | RMSprop，学习率`0.001` |
| 损失 | 原始输出MSE |
| checkpoint选择 | 第100个epoch的最终权重 |
| 训练设备 | 远程NVIDIA RTX A6000，physical GPU0 |

seed0与seed1分别生成两把不同的固定秘密密钥。跨密钥实验不是把多把密钥混入同一训练集，而是为每把
固定密钥独立训练模型后比较冻结的同一组输出位置。这避免相同明文在不同密钥下对应矛盾目标。

OPK1用`256`把参考密钥、`256`把零重合评估密钥和`1024`个完全相同明文专门审计这一边界。参考
密钥逐明文频率在未见密钥上的八bit平均AUC仅为`0.500544`，方向化AUC最大值为`0.502176`。因此
本文的“跨密钥”只表示同一位置和协议在独立固定密钥上分别训练后的重复性，不表示一个只看明文的
模型可对完全未见密钥零适配泛化。后者若不提供密钥或已知明密文support set，目标本身不稳定。

OP10不重新训练模型。它先在OP9未参与训练的`65536`条测试明文上扫描64个输出位置，冻结最多八个
`(输出位置, 模型)`候选及其SHA-256；随后再生成`65536`条与OP9全部明文不重合的新明文，作为fresh
确认集。候选选择和最终确认使用不同明文，从而控制从64个位置中后验选择最大AUC造成的赢家诅咒。

## 4. 网络结构与同预算控制

### 4.1 完整输出论文锚点

Kimura式LSTM将64个明文bit视为长度64、每步一维的序列。六层LSTM的hidden维度为300，最后一个
隐藏状态通过线性层输出64个原始分数：

```text
[batch, 64]
  -> [batch, 64, 1]
  -> LSTM(input=1, hidden=300, layers=6)
  -> Linear(300, 64)
```

参数量为`3,994,864`。同预算MLP使用两层hidden 1936的ReLU骨干：

```text
Linear(64, 1936) -> ReLU
Linear(1936, 1936) -> ReLU
Linear(1936, 64)
```

参数量为`3,999,840`，与LSTM只相差约`0.12%`。OP9另训练结构完全相同、只对训练标签行做固定随机
排列的LSTM控制；测试仍使用真实密文目标。

### 4.2 预注册八输出专用头

OP11沿用相同两层MLP骨干，只把输出层从64维缩减为八个预注册位置：

```text
Linear(64, 1936) -> ReLU
Linear(1936, 1936) -> ReLU
Linear(1936, 8)
```

专用头参数量为`3,891,368`。OP11的三行同预算矩阵为：完整64输出MLP、真实八输出专用MLP、八输出
训练标签打乱MLP。这样可以区分“位置本身可预测”“减少无关输出任务有益”和“模型只利用标签边际”三种
解释。

### 4.3 四轮结构化XOR头

OP12仍使用相同MLP骨干，直接输出六个预注册XOR值，参数量为`3,887,494`。它同时训练：

```text
八个单bit真实输出anchor
六个结构化XOR直接输出头
六个同重量几何控制XOR输出头
六个结构化XOR标签打乱输出头
```

此外，从八输出anchor的单bit分数`p_j`计算独立近似下的派生parity：

```text
P(XOR=1) = (1 - PRODUCT_j(1 - 2*p_j)) / 2
```

直接XOR头必须同时超过几何控制、标签打乱、单bit派生parity和mask内最佳组成bit，才支持“网络学习了
有助于扩轮的组合输出函数”。

## 5. 评价指标与判定门

完整输出任务报告：原始输出MSE、逐bit match、64个bit的macro AUC，以及64个bit同时命中的exact
match。逐bit和XOR任务报告：

```text
AUC
threshold accuracy，阈值固定为0.5
majority accuracy = max(P(y=0), P(y=1))
accuracy - majority
true AUC - matched shuffle AUC
```

AUC在此用于评价真实二值输出函数的排序能力，不改变任务语义；它不能与真假样本神经区分器的AUC
直接比较。准确率必须超过该bit自身的多数值基线，防止输出边际轻微不平衡产生虚假优势。

OP11中一个位置通过需要同时满足：

```text
AUC >= 0.510
accuracy - majority >= 0.005
true AUC - matched shuffle AUC >= 0.005
```

八个位置中至少四个通过，且专用八输出头在同位置上的平均AUC至少比完整64输出头高`0.002`，才支持
跨密钥与专用头结论。

OP12中一个结构化mask通过需要同时满足六门：

```text
direct AUC >= 0.510
accuracy - majority >= 0.005
direct - shuffle AUC >= 0.005
direct - geometry AUC >= 0.005
direct - derived parity AUC >= 0.005
direct - best component-bit AUC >= 0.002
```

同末轮S-box双bit家族要求至少`2/4`个mask通过；同角色四bit家族要求`2/2`均通过。任一家族通过才
允许在另一把固定密钥上复现；否则停止mask、数据、epoch、模型和轮数机械扩展。

## 6. 分阶段实验与结果

实验围绕四个递进研究问题展开：

```text
RQ1：单固定密钥PyTorch实现能否恢复Kimura论文族的PRESENT三轮完整输出信号？
RQ2：完整输出未通过时，是否仍有真实密文bit能在独立明文上确认？
RQ3：冻结位置能否在第二把固定秘密密钥上复现，专用头是否优于完整输出头？
RQ4：预注册多bit XOR能否在四轮同时超过单bit和四类匹配控制？
```

### 6.1 OP9：完整64-bit输出协议校准

| 模型 | bit match | macro AUC | exact match | 测试MSE |
|---|---:|---:|---:|---:|
| Kimura式LSTM真实输出 | 0.499915838 | 0.500007607 | 0 / 65536 | 0.250080714 |
| 参数量匹配MLP真实输出 | 0.498848200 | 0.508066504 | 0 / 65536 | 0.392121862 |
| Kimura式LSTM标签打乱 | 0.500027418 | 0.500013948 | 0 / 65536 | - |

LSTM没有超过标签打乱控制，完整密文一次也未同时命中，与Kimura等人Table 12/C.2的三轮参考
`2^-1.30 ≈ 0.406126`存在数量级差距。因此，当前PyTorch单密钥实现不支持论文级完整输出校准，
停止继续扩大当前完整输出LSTM。该实验使用单把密钥和PyTorch实现，不是原论文Keras框架、超参数
搜索及100把密钥汇总的精确复现。这个结果只否定当前完整64-bit路线，不能推出每个输出位置都不可
预测；同预算MLP的macro AUC为`0.508066504`，提示位置之间可能存在异质性。

### 6.2 OP10：易预测位置发现与fresh确认

OP10在seed0固定秘密密钥上由真实输出MLP选出八个位置，并在全新`65536`条明文上全部复现：

| MSB位置 | 整数bit | nibble/内部bit | fresh准确率 | 超多数类 | fresh AUC |
|---:|---:|---:|---:|---:|---:|
| 0 | 63 | 0 / 0 | 0.519394 | +0.018600 | 0.526567 |
| 2 | 61 | 0 / 2 | 0.517242 | +0.015396 | 0.523485 |
| 8 | 55 | 2 / 0 | 0.512665 | +0.011734 | 0.519002 |
| 10 | 53 | 2 / 2 | 0.515320 | +0.014221 | 0.522252 |
| 32 | 31 | 8 / 0 | 0.518173 | +0.017197 | 0.524560 |
| 34 | 29 | 8 / 2 | 0.518600 | +0.017441 | 0.524725 |
| 40 | 23 | 10 / 0 | 0.518478 | +0.016815 | 0.525110 |
| 42 | 21 | 10 / 2 | 0.514175 | +0.013428 | 0.520558 |

位置只落在MSB-first nibble `0/2/8/10`的内部bit `0/2`，呈现结构化子集。但OP10的打乱对照是LSTM，
而八个位置均由MLP选出，因此OP10只支持fresh明文信号，不能单独支持同架构归因或跨密钥稳定性。

### 6.3 OP11：第二秘密密钥与专用八输出头

八个位置在seed1第二把独立固定秘密密钥上全部通过AUC、准确率和同架构shuffle三门：

| MSB位置 | 专用头AUC | 准确率 | 超多数类 | shuffle AUC | true-shuffle | 完整头AUC | 专用-完整 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.527628 | 0.519730 | +0.015381 | 0.499809 | +0.027819 | 0.522927 | +0.004701 |
| 2 | 0.529898 | 0.520401 | +0.019608 | 0.501564 | +0.028333 | 0.521003 | +0.008894 |
| 8 | 0.534412 | 0.525146 | +0.023483 | 0.502269 | +0.032143 | 0.522549 | +0.011863 |
| 10 | 0.528777 | 0.521301 | +0.020920 | 0.501493 | +0.027284 | 0.518808 | +0.009970 |
| 32 | 0.532778 | 0.522415 | +0.020874 | 0.503602 | +0.029176 | 0.524600 | +0.008178 |
| 34 | 0.534711 | 0.523056 | +0.023056 | 0.501428 | +0.033283 | 0.522988 | +0.011722 |
| 40 | 0.528734 | 0.521423 | +0.019501 | 0.498359 | +0.030375 | 0.518608 | +0.010127 |
| 42 | 0.530262 | 0.521408 | +0.020355 | 0.497779 | +0.032483 | 0.523769 | +0.006493 |

联合结果为：

```text
确认位置数                         = 8 / 8
专用八输出头平均AUC               = 0.530900037
完整64输出头在同位置的平均AUC     = 0.521906462
同架构八输出标签打乱平均AUC       = 0.500787873
专用头 - 完整头                   = +0.008993575
专用头 - 标签打乱                 = +0.030112164
```

同一位置集合从seed0发现并fresh确认后，在seed1独立密钥上`8/8`复现，降低了结果仅由seed0单把
密钥偶然位置选择导致的可能性。专用头在八个位置上均高于完整64输出头，平均差值为`+0.008993575`；
这一受控结果支持“完整输出多任务干扰可能掩盖局部可预测坐标”的解释，但两把密钥不足以建立广泛
跨密钥统计结论。

### 6.4 OP12：四轮结构化多bit XOR扩展

OP12在seed1同一固定密钥上把轮数增至四轮。四个双bit mask通过精确逆P-layer映射配对同一末轮
S-box来源；两个四bit mask分别聚合同一输出角色。结果如下：

| 结构化XOR mask | 直接AUC | 超多数类 | -shuffle | -几何 | -派生 | -最佳bit | 通过 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0⊕32` | 0.502917533 | +0.002365112 | -0.001551650 | +0.002656740 | +0.004221954 | -0.000663199 | 否 |
| `2⊕34` | 0.498873753 | -0.002655029 | -0.002568371 | -0.003399335 | -0.001944928 | -0.006558020 | 否 |
| `8⊕40` | 0.497671918 | -0.004165649 | -0.000297186 | +0.000824416 | +0.003370649 | -0.011056572 | 否 |
| `10⊕42` | 0.500414826 | +0.001312256 | -0.000258315 | -0.001037967 | +0.002387307 | -0.004665088 | 否 |
| `0⊕2⊕8⊕10` | 0.500779890 | -0.000625610 | -0.001756679 | +0.000071366 | +0.002645932 | -0.007948600 | 否 |
| `32⊕34⊕40⊕42` | 0.499000804 | -0.001464844 | +0.002416382 | -0.003300416 | -0.004722545 | -0.008537758 | 否 |

```text
直接结构化XOR平均AUC       = 0.499943121
同重量几何控制平均AUC     = 0.500640653
匹配标签打乱平均AUC       = 0.500612424
单bit派生parity平均AUC    = 0.498950059
最佳组成bit平均AUC        = 0.506514660
通过mask                  = 0 / 6
```

六个mask均未达到绝对AUC或准确率门，也没有同时超过四类强基线。因此，当前证据不支持“把多个
易预测密文bit异或后可以把PRESENT真实输出预测从三轮推进到四轮”。按照预注册停止门，不启动另一
密钥OP13、不进入五轮，也不后验枚举新mask或机械增加数据、epoch与模型容量。

### 6.5 OPA1--OPA3：多模型筛选、独立确认与拓扑归因

OPA1在第三固定秘密密钥上保持八个输出位置、`2^17/2^16`明文对、100 epochs和约390万参数预算
一致，比较MLP、LSTM、ResCNN、Transformer和PRESENT-SPN-aware：

| 模型 | 八位置平均AUC |
|---|---:|
| MLP | 0.531656795 |
| 六层LSTM | 0.500000000 |
| 位置保持ResCNN | 0.588387942 |
| Transformer | 0.499477131 |
| PRESENT-SPN-aware | 1.000000000 |

OPA2随后在第四固定秘密密钥上比较MLP/SPN的true与匹配label-shuffle。SPN true仍为`1.000000000`，
SPN shuffle为`0.500839804`，MLP true为`0.532262231`；扣除各自shuffle后，SPN相对MLP的调整增益为
`+0.465751518`。这确认的是整体架构优势。

OPA3保持参数、初始化、数据、局部nibble混合和训练预算不变，只替换无参数bit排列：

| 拓扑 | 八位置平均AUC |
|---|---:|
| exact PRESENT P-layer | 1.000000000 |
| identity P-layer | 0.531989557 |
| fixed wrong P-layer | 1.000000000 |

exact相对identity在每个位置均高约`0.46--0.47`，但相对wrong-P的八个差值全部为`0.0`，所以正式门为
`hold`、attributed bits=`0/8`。确定性感受野审计显示exact和wrong都按`1 -> 4 -> 16 -> 64`扩展，
identity始终停在4bit。这支持分层跨nibble扩散归纳偏置，不支持精确PRESENT连线是唯一原因。按预
注册停止门，条件式OPA4/OPA5均未实现或启动。

### 6.6 OPB1：低秩位置条件仍未隔离精确拓扑

OPB1删除每个位置完整189维嵌入，改为每轮64个标量乘共享方向，并在第五固定秘密密钥上比较原SPN
锚点、低秩exact-P、低秩wrong-P和匹配shuffle。原锚点、exact-P与wrong-P八位置平均AUC仍全部为
`1.000000000`，shuffle为`0.499476582`；exact-P相对wrong-P差值为`0`，归因bit为`0/8`。

这说明低秩瓶颈保留了三轮输出预测能力，却没有让精确PRESENT连线优于任意快速混合错误双射。因而
低秩路线停止，不做seed5确认或机械容量搜索；下一候选转为以OPA1中非饱和ResCNN为锚点的
SPN-ResCNN混合控制实验。四轮仍保持关闭。

### 6.7 OPC1：SPN-ResCNN混合网络正式矩阵运行中

OPC1只改变一个架构假设：把OPA1中平均AUC为`0.588387942`的非饱和位置保持ResCNN作为普通网络
锚点，在同样10个残差块的`3+3+4`阶段之间插入三次无参数P-layer位置重排。普通ResCNN、exact-P
混合模型和wrong-P混合模型均为`3,955,904`个参数，第四行使用exact-P架构但打乱训练标签。

正式协议冻结为第七把独立固定秘密密钥、`2^17`训练明文、`2^16`测试明文、每个模型100 epochs，
仍预测同八个PRESENT三轮真实密文bit。正式矩阵已经从推送提交`286cd0c`在远程A6000 GPU0启动，
磁盘缓存和readiness通过，本地watcher负责完成检测和结果回收。当前状态仅为`running`；在verified
result branch、hash、协议门和视觉门全部完成前，不报告候选AUC或保留/删除结论。

OPC1必须同时超过普通ResCNN、wrong-P和匹配shuffle，并至少使`4/8`个bit通过逐位门，才允许换
全新固定密钥原样确认。失败或hold则保留ResCNN发现锚点并停止该混合路线，不通过增加深度、数据、
epoch、错误P或输出位置绕过门，也不直接开放四轮。

### 6.8 OPM1：易预测输出位粗粒度结构基线

OPM1对全部64个PRESENT输出bit计算一至三轮精确反向输入依赖锥，并计算四个S-box输出坐标的balance、
ANF degree、nonlinearity和ANF term count。全部64个输出bit的锥宽均为：

```text
一轮 = 4 bit
二轮 = 16 bit
三轮 = 64 bit
```

八个selected位置经最后一轮inverse P-layer回溯后对应LSB-first S-box输出坐标`1/3`。四个坐标全部
平衡、nonlinearity均为`4`；ANF
degree为`[2,3,3,3]`，selected坐标均为degree 3，并非唯一最低degree坐标。因此易预测位置没有更窄
三轮输入依赖锥，也不落在明显更简单的单S-box坐标上。

该确定性结果只排除两个粗粒度充分解释，不能证明结构无关，也不能解释固定密钥函数谱或训练动力学。
它不改变OPC1，不授权后验重选输出位或追加远程训练。

## 7. 结果讨论

### 7.1 完整输出失败不等于局部输出失败

64-bit exact match要求同一样本的64个输出bit全部正确。即使少数位置存在稳定但较弱的预测优势，其余
位置接近随机也会使完整命中概率迅速衰减。OP9的完整命中为0，而OP10/OP11仍在八个冻结位置上获得
跨新明文和第二秘密密钥的稳定AUC，实验结果表明两种结论可以同时成立。

### 7.2 专用头增益支持多任务干扰解释

OP11中专用头与完整头共享输入、两层hidden 1936骨干、训练集、测试集、优化器、epoch和秘密密钥，
主要差异仅是输出维度。八个位置全部获得正向专用头增益，平均增益`+0.008993575`，而同架构shuffle
接近`0.5`。这比单独报告八个最高AUC更能支持输出选择方法的归因。

### 7.3 多bit XOR没有提高可预测轮数

如果多个位置之间存在比单bit更稳定的结构关系，直接XOR头应超过由单bit独立概率派生的parity，并
超过同重量但破坏结构关系的几何控制。OP12中直接XOR平均AUC反而低于几何控制和最佳组成bit，说明
当前预注册组合没有保留足够四轮信号。这个否定结果限制的是本章的具体机制，不是所有可能的四轮输出
函数的数学不可能性。

### 7.4 整体SPN式架构有效不等于精确P-layer被归因

OPA2的匹配shuffle和MLP控制排除了标签顺序及单纯参数量解释，支持整体SPN式架构对三轮八输出任务
有效。OPA3又显示identity不扩散控制明显较弱，说明局部到全局的跨nibble信息传播是重要机制；但是
wrong-P与exact-P完全同分，说明当前网络只需要相同的全局感受野扩展，不要求精确密码连线。论文应
同时报告这两个结果，不能只展示exact-P的`AUC=1.0`。OPB1进一步压缩位置条件容量后exact-P和
wrong-P仍完全同分，说明该负归因不是仅靠原模型的高维位置嵌入造成；OPC1正在检验把拓扑嵌入非饱和
ResCNN能否同时保持输出能力并恢复精确拓扑差异。OPM1还表明八个selected bit与其他输出bit具有
相同三轮锥宽，其S-box坐标也不是唯一低degree坐标；因此论文不能用这两个粗粒度量直接解释易预测性。

## 8. 创新点、证据边界与开题对齐

本章可主张的创新点是：

1. 构建“共享全输出扫描、独立明文确认、第二秘密密钥复现、专用小输出头归因”的真实输出值预测流程；
2. 在PRESENT三轮上发现八个易预测真实密文输出位置，并在第二把固定秘密密钥上`8/8`复现；
3. 显示专用八输出头在同预算下优于完整64输出头和同架构标签打乱控制；
4. 通过预注册结构化XOR、几何控制、派生parity和最佳组成bit给出四轮扩展的明确停止边界。
5. 在同预算五模型发现屏中，SPN式网络优于MLP、LSTM、ResCNN和Transformer候选；随后在第四密钥
   上相对MLP与匹配shuffle独立确认整体架构优势；
6. 用identity与固定错误双射给出机制边界：分层扩散有效，但精确PRESENT P-layer未被因果归因。
7. 用低秩位置条件瓶颈复核上述负归因，排除“只因高维自由位置嵌入导致任意拓扑饱和”这一单一解释。

本章不能主张：

- 恢复了完整64-bit PRESENT密文；
- 复现了Kimura等人跨100把独立固定密钥模型的平均结果；
- 达到了主流PRESENT七至九轮神经区分或积分攻击轮数；
- 多bit XOR普遍优于单bit输出预测；
- OP12证明所有四轮输出函数均不可预测；
- 完成了`state_t -> state_(t+1)`中间状态逐轮迭代预测攻击；
- 可将本章AUC与真假样本区分器AUC直接比较并宣称SOTA。
- OPA3证明精确P-layer是`AUC=1.0`的原因；
- OPB1低秩瓶颈已经隔离出精确P-layer贡献；
- OPC1正在运行就等于已经获得SPN-ResCNN正式正结果；
- OPA4/OPA5已经得到四轮八输出网络结果。

相对开题报告，本章已经覆盖“未知密钥下从输入预测密码真实输出、预注册并检验条件式扩轮机制”的核心
实验思想，但三轮单bit与四轮XOR不是同一目标，只形成PRESENT三轮正结果与结构化XOR四轮负边界，
尚未形成同任务或跨密码临界轮数曲线，也没有实现中间状态递推。因此，它可以作为创新2的一个完整
实验单元和论文小节，不能单独代替开题报告中全部创新2工作。

### 8.1 局限性与有效性威胁

第一，OP10/OP11的专用MLP确认只覆盖两把独立固定秘密密钥；OPA1/OPA2再覆盖第三、第四密钥，OPB1
覆盖第五把密钥，但
OPA1是fallback-retrieved发现证据，OPA2只独立确认SPN相对MLP与shuffle，没有在第四密钥重跑全部
五模型。因此当前仍没有多密钥均值、方差或置信区间。第二，OP9复用了论文报告的网络规模和训练预算，
但框架、初始化、超参数搜索过程和密钥数量不同，只能视为论文族单密钥校准。第三，OP10的MLP候选在
发现和fresh阶段只配有跨架构LSTM标签打乱控制；OP11才补齐同架构MLP shuffle。第四，OP11预测三轮
单bit，OP12预测四轮XOR，OPA4又因OPA3归因未通过而关闭，当前没有同一八输出任务的四轮临界点。
最后，SPN式网络的三轮八输出平均AUC虽达到`1.0`，wrong-P控制也同为`1.0`；该结果既不能解释为精确
P-layer因果贡献，也不代表完整密文恢复、实际密钥恢复能力或主流攻击轮数竞争力。
OPK1还表明无密钥、无校准的key-blind新密钥任务没有稳定标签；本文不将逐密钥重训复现包装为
零样本密钥泛化。带support set的密钥上下文推断属于尚未实现的后续任务。

## 9. 可直接用于摘要或答辩的结论

> 本文针对完整密文输出预测容易受多任务干扰的问题，提出易预测输出坐标发现、专用多输出头和SPN式
> 分层扩散网络。在固定未知秘密密钥的PRESENT三轮任务中，八个候选位置通过独立明文、第二秘密密钥
> 和后续三把固定密钥评估；专用MLP头平均AUC为0.530900，而同预算多模型筛选中的SPN式网络在第三、
> 第四密钥上均达到1.000000，并在第四密钥显著超过MLP和匹配标签打乱。拓扑归因中exact-P与wrong-P
> 同为1.000000、identity为0.531990；第五密钥上的低秩exact-P与wrong-P也同为1.000000，说明分层
> 跨nibble扩散有效，但精确PRESENT连线不是当前性能的唯一解释。四轮结构化XOR的六个目标均未过门，
> 条件式四轮SPN网络因归因门未通过而没有启动。SPN-ResCNN三轮混合归因仍在运行，尚无正式结论。该结果
> 给出三轮选定位输出预测的正证据和精确拓扑/四轮扩展的清晰边界，不是完整密文恢复或高轮攻击突破。

## 10. 可复现证据入口

论文协议与外部参考：

```text
papers/innovation_two/text/2021_kimura_output_prediction_block_ciphers.txt
docs/research/innovation2-output-prediction-paper-protocol-audit-20260721.md
```

正式实验记录：

```text
docs/experiments/innovation2-output-prediction-op9-present-r3-kimura-lstm-plan.md
docs/experiments/innovation2-output-prediction-op10-present-r3-easy-bit-discovery-plan.md
docs/experiments/innovation2-output-prediction-op11-present-r3-selected8-independent-key-plan.md
docs/experiments/innovation2-output-prediction-op12-present-r4-structured-xor-plan.md
docs/experiments/innovation2-output-prediction-opa1-present-r3-selected8-architecture-screen-plan.md
docs/experiments/innovation2-output-prediction-opa2-conditional-architecture-confirmation-plan.md
docs/experiments/innovation2-output-prediction-opa3-present-r3-topology-attribution-plan.md
docs/experiments/innovation2-output-prediction-opb1-present-r3-topology-bottleneck-plan.md
docs/experiments/innovation2-output-prediction-opc1-present-r3-spn-rescnn-hybrid-plan.md
docs/experiments/innovation2-output-prediction-opm1-present-r3-selected-output-structural-baseline-audit-plan.md
```

正式结果：

```text
outputs/remote_results/i2_output_prediction_op9_present_r3_kimura_lstm_2p17_seed0_gpu0_20260721/
outputs/remote_results/i2_output_prediction_op10_present_r3_easy_bit_confirm_gpu0_20260721/
outputs/remote_results/i2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721/
outputs/remote_results/i2_output_prediction_op12_present_r4_structured_xor_key1_gpu0_20260721/
outputs/remote_results_incomplete/i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721_raw_fallback/
outputs/remote_results/i2_output_prediction_opa2_present_r3_selected8_present_spn_key3_gpu0_20260722/
outputs/remote_results/i2_output_prediction_opa3_present_r3_selected8_topology_attribution_key3_gpu0_20260722/
outputs/remote_results/i2_output_prediction_opb1_present_r3_topology_bottleneck_key4_gpu0_20260722/
outputs/local_audits/i2_output_prediction_opm1_present_r3_selected_output_structural_baseline_audit_20260722/
```

正文建议引用已通过像素检查的OP11、OP12、OPA2、OPA3与OPB1正式`curves.svg`。本初稿没有生成新图，
移入论文模板或重新导出图像后，仍需重新执行视觉质量检查。

## 11. 下一步执行计划

当前不再为已失败的四轮结构化XOR、未归因的精确P-layer或已关闭的OPA4/OPA5分配训练预算。OPC1
正式矩阵已在远程运行，后续按
以下顺序推进：

1. 等待watcher回收OPC1 verified result branch，验证源提交、OPB1 gate、`32/400/4/196608`计数、
   archive hash和全部协议/执行门，并对正式SVG执行像素QA；
2. OPC1通过则只换全新固定密钥原样确认；失败或hold则保留ResCNN锚点并停止混合路线；
3. 将最终裁决并入学校论文模板，补齐正式参考文献编号、图号和表号；
4. 用OP11、OP12、OPA2、OPA3、OPB1及通过视觉门的OPC1正式SVG制作统一字号论文图；
5. 在正文中并列报告OPA2整体架构正结果、OPA3/OPB1错误拓扑同分结果和OPC1最终裁决；
6. 只有独立文献或确定性密码结构分析提出新的方法级四轮输出函数机制，才允许新建预注册路线；不得
   通过换wrong-P、seed、阈值、样本、epoch或网络名称重开已关闭的OPA4/OPA5。

参考文献条目（最终格式待论文模板统一）：

```text
Hayato Kimura, Keita Emura, Takanori Isobe, Ryoma Ito, Kazuto Ogawa,
and Toshihiro Ohigashi. Output Prediction Attacks on Block Ciphers using
Deep Learning. 2022.
```

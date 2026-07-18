# 创新2 E52：PRESENT五轮严格标签证书提供者覆盖审计计划

日期：2026-07-18

状态：计划冻结 / 待调研实现

## 1. 研究问题

E43--E51已经完成PRESENT-80四轮严格标签上的架构筛选，最终确定性ANF/degree约`0.69`强于
所有纯神经与神经残差候选。继续在四轮换网络不再是有效研究投入。

E52审计真正的轮数瓶颈：能否在保持创新2原始“指定输出mask是否对所有key和inactive
offset保持积分XOR平衡”语义下，为PRESENT-80五轮产生足够的可证明正类和具体反例负类，
并构造可训练、无一元捷径的structure-disjoint benchmark。

E52不训练神经网络，不使用远程GPU。

## 2. 固定语义与来源

```text
cipher            = PRESENT-80
rounds            = 5
active dimension  = 8 bits
output query      = 64-bit nonzero linear mask
positive          = 对所有80-bit key和所有inactive offset，masked cube XOR恒为0
negative          = 存在具体(key, inactive offset)使masked cube XOR为1
unknown           = 当前提供者既不能证明positive，也没有找到negative witness
```

不得把有限密钥经验平衡率、固定key零XOR、随机负密文、结构化vs随机分类或单样本mask parity
替代严格标签。E43四轮atlas只作来源与宽度锚点，不混入五轮训练数据。

## 3. 提供者候选与单变量顺序

先完成文献/实现审计并冻结可执行提供者，不并行混合多个新算法。候选优先级：

```text
P0 = 当前sound active-variable ANF support superset（复现失败锚点）
P1 = 更精确的division-property / monomial-prediction certificate
P2 = P1无法落地时的可验证MILP/SAT cube-coefficient zero certificate
```

P1/P2必须输出可独立复验的证书或完整solver witness，不接受只有“solver says balanced”的不透明
布尔值。使用外部库前先核对论文语义、PRESENT bit order、key schedule、round convention和许可证；
不得把截断差分、普通division trail absence或固定key ANF误报为全密钥全offset证书。

只在P0覆盖审计完成后选择一个P1实现；P1失败才考虑P2。每次只改变证书提供者，structure、
mask、negative witness和匹配协议保持固定。

## 4. 冻结候选池与预算

```text
structure seed       = 20260718
structures           = 96（复用E43 8-bit coordinate cubes）
output masks          = 300（复用E43 mask集合）
candidate pairs       = 28800
negative witness bank = 16 keys x 8 offsets/structure，沿用E43确定性种子
scalar recheck        = 至少32个negative + 全部抽样positive证书边界fixture
device                = local CPU
```

若P1/P2完整池预计超过本地合理预算，先在固定`16 structures x 64 masks`子集做coverage readiness；
只有positive率、negative率和证书复验均非零才扩大到完整池。不得因计算慢直接改用远程训练。

## 5. 标签与反捷径门

完整池最低宽度：

```text
positive structures同时含negative       >= 48
可构造train matched checkerboard         >= 400 positive + 400 negative
可构造validation checkerboard            >= 118 positive + 118 negative
train/validation structures              disjoint
global/mask/family/active-bit unary AUC   <= 0.65
duplicate(structure, mask)                = 0
```

正确性门：

```text
PRESENT官方向量与round convention          pass
每个positive均有sound可序列化证书           pass
抽样positive独立边界/solver复验             pass
抽样negative标量加密复验masked XOR=1        pass
provider timeout/error与unknown分开记录      pass
```

全部通过：

```text
decision = innovation2_present_r5_strict_label_bank_ready
next     = 先做确定性shortcut/feature attribution，再冻结首个五轮神经矩阵
```

覆盖不足：停止五轮神经训练，记录最强提供者的positive/negative/unknown与失败原因；不得用
经验标签填充。协议无效：只修bit order、round/key、证书或复验协议，不解释覆盖率。

## 6. 产物与下一动作

```text
outputs/local_audits/i2_present_r5_strict_label_provider_coverage_20260718/

provider_manifest.json
structures.json
masks.json
labels.csv
certificates.jsonl
witnesses.jsonl
matched_contrast.csv（仅在门通过时）
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

E52完成后必须给出：最强提供者、覆盖率、证书/反例正确性、是否可训练、下一步是否开放网络、
明确关闭的替代语义与机械扩展。声明范围仅为五轮严格标签提供者覆盖，不是神经结果、五轮
区分器、新攻击或SOTA。

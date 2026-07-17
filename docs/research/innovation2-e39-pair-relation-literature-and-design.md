# 创新2 E39：有向bit-pair关系推理网络文献与设计裁决

日期：2026-07-18

状态：文献核对完成 / 低秩PPGN式关系路径候选优先

## 1. 证据起点

E37证明扩展合成SPN标签不是拓扑无关或ID捷径主导；320个train-only selected cell中，
train P-sensitive、train S×P interaction、dual P-effect和full interaction比例分别为
`1.000000/0.981250/0.984375/1.000000`。

E38在12个独立train P-layer上比较cell-equivariant GraphGPS与CETT。两者未见S-box均值
约`0.90`，但dual均值只有`0.641682/0.629657`，低于确定性ID边际`0.684393`。这把失败
集中到“新S-box条件下组合新P-layer路径”，而不是S-box truth table读取、标签宽度或
训练拓扑数量。

## 2. 已核对的关系推理路线

### 2.1 Higher-order GNN / k-GNN

Morris等人的《Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks》
（AAAI 2019，arXiv:1810.02244）从理论上说明常规message-passing GNN与1-WL表达能力
对应，并提出在节点tuple上运行的k-GNN。它支持“当前node-message表示可能不够强”的判断，
但没有给出本任务需要的低成本有向路径组合算子。

### 2.2 PPGN / 2-FWL矩阵乘法

Maron等人的《Provably Powerful Graph Networks》（NeurIPS 2019，arXiv:1905.11136）
维护二阶张量，并在feature MLP之间加入矩阵乘法。官方摘要与论文说明该精简二阶网络具有
3-WL表达能力，严格强于普通message passing；其2-FWL block把`R[i,k]`与`R[k,j]`配对并
沿中间节点`k`聚合。这与SPN多轮bit路径组合最直接对应。

限制是完整hidden-channel矩阵乘法计算较大。E39采用低秩path channel投影后再做矩阵乘法，
保持三角组合归纳偏置，同时把16节点上的成本固定为`O(N^3 r)`。

### 2.3 Neural Bellman-Ford Network

Zhu等人的《Neural Bellman-Ford Networks: A General Graph Neural Network Framework for
Link Prediction》（NeurIPS 2021，arXiv:2106.06935）把一对节点表示写成路径表示的广义
求和，并用INDICATOR、MESSAGE、AGGREGATE三个学习组件参数化Bellman-Ford递推。它证明
“学习有向关系路径”是成熟机制，也适用于inductive relation prediction。

但NBFNet天然是给定单一source/query的条件消息传递。本任务同时有多个active bit与多个
output-mask bit；逐source运行会改变计算预算与表示边界，所以不作为第一实现。

### 2.4 Graph Transformer Networks

Yun等人的《Graph Transformer Networks》（NeurIPS 2019，arXiv:1911.06455）通过软选择
edge type并组合邻接矩阵学习meta-path。它适合异质图中的edge-type组合，但本任务只有固定
P-edge、同S-box cell等少量关系，且预测条件由active/mask query变化；直接使用GTN容易把
查询相关信息压缩到静态meta-path，不优先。

### 2.5 FloydNet与当前新颖性边界

2026检索发现一篇匿名ICLR 2026投稿FloydNet，使用全局pair tensor与pivot节点更新学习
Floyd-Warshall式关系演算。检索时仍显示匿名under-review，不能当作已验证基线；但它说明
“pair tensor + pivot update”不能被本项目宣称为通用架构首创。

因此E39的可辩护创新范围是：

```text
SPN专用有向bit-pair初始化
+ S-box/P-layer/active/output-mask联合条件
+ 按真实密码轮数执行的共享低秩2-FWL路径更新
+ 输出积分平衡/kernel membership预测
```

不是“发明了pair relation network”。

## 3. 候选排序

| 排名 | 路线 | 与E38失败的对应 | 当前裁决 |
|---:|---|---|---|
| 1 | 低秩PPGN式pair triangle reasoner | 直接组合`i→k→j`，强于node message passing | E39实施 |
| 2 | query-conditioned NBFNet | 有向路径递推强，但多active/mask需多source改造 | E39失败后再审 |
| 3 | GTN meta-path composition | 能组合关系类型，但query条件弱 | 暂缓 |
| 4 | FloydNet pivotal attention | 最接近但匿名、复杂且参数归因困难 | 不首轮复现 |
| 5 | 更深CETT/GraphGPS、Mamba、KAN | 未对应已观测表达缺口 | 停止 |

## 4. E39最小结构

每个样本构造`R0 ∈ R^(16×16×h)`。有序bit pair `(i,j)`包含：

```text
source/destination lane role
i == j
P(i) == j
P(j) == i
i、j是否同S-box cell
i、j是否同lane
active(i)、active(j)
output_mask(i)、output_mask(j)
round embedding
variant S-box truth-table embedding
```

共享路径块先投影到低秩path channels：

```text
L = left(R)   ∈ R^(16×16×r)
Q = right(R)  ∈ R^(16×16×r)
T[i,j,c] = sum_k L[i,k,c] * Q[k,j,c]
R' = R + update(R, project(T))
```

同一块按实际round count运行2、3、4或5次。readout包含全pair、对角、P-edge、
active→mask和mask→active五种池化，不使用绝对bit/cell/variant ID。

## 5. 风险与停止条件

主要风险：

1. pair模型可能只增加容量；因此参数必须不超过E38 GraphGPS的`297409`；
2. triangle update可能对真实/错误P不敏感；readiness必须直接检查同权重logit差异；
3. pair初始化可能破坏cell重标号不变性；误差必须`<=1e-6`；
4. 高train AUC仍可能无法dual泛化；full仍用E37 ID边际和label-shuffle裁决；
5. 若有效但不过fair-corrupted P，不得解释为真实拓扑收益。

E39不过ID门后，不增加hidden、path rank、epoch或远程规模。下一研究选择应在
query-conditioned NBFNet与结构化P-layer族之间重新比较，而不是继续枚举通用网络名称。

## 6. 来源

检索原始结果：

```text
sources/research_spn_pair_relation_reasoning_20260718.json
sources/research_spn_triangle_relation_networks_20260718.json
sources/research_spn_pair_relation_arxiv_metadata_20260718.xml
```

官方元数据核对：

- Morris et al., AAAI 2019, arXiv:1810.02244；
- Maron et al., NeurIPS 2019, arXiv:1905.11136；
- Yun et al., NeurIPS 2019, arXiv:1911.06455；
- Zhu et al., NeurIPS 2021, arXiv:2106.06935。

FloydNet仅按检索时的匿名投稿状态记录，不用于正式性能比较或首创声明。

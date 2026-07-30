# 创新2：uKNIT-BC 线性代数积分核轮数普查

日期：2026-07-30
状态：已完成 / pass（单 cell 边界为 r4）

## 1. 研究问题

复用 Hwang et al. 2026 的 parity-matrix kernel 方法，回答：

> uKNIT-BC 的单活动4-bit cell积分结构，在 fresh keys 与 fresh inactive contexts
> 上能够稳定支持到多少轮的非零64-bit输出平衡 mask？

该实验是训练自由的积分区分器普查，不是现有 uKNIT 差分神经区分器的扩样，也不
使用 AUC。最高通过轮只表示 sampled-key empirical integral distinguisher。

## 2. 固定协议

```text
cipher = uKNIT-BC
block/key = 64/128 bit
round semantics = prefix-reduced AddKey -> S -> GF(2) linear layer
calibration rounds = r1
target rounds = r3,r4,r5,r6,r7,r8,r9,r10,r11
active structures = 16 single active cells, MSB-first cell index 0..15
plaintexts per multiset = 16
inactive context = fresh deterministic random value per trial
feature = raw 64-bit ciphertext XOR parity
discovery trials = 128 fresh unique keys + contexts
validation trials = 128 fresh unique keys + contexts
total trials = 256
seed = 0
training = none
device = local CPU
```

发现和验证复用相同的 cell/round 定义，但密钥与非活动上下文完全分离。所有 cell
在同一 trial 复用同一主密钥和 base plaintext，保证位置对比只改变活动 cell。

## 3. 控制

- `r1` 校准：每个活动 cell 的16个输入经过一轮双射 S-box 与线性层，输出 XOR
  应为全零；joint rank/nullity 必须为 `0/64`。
- 随机输出控制：每个 round/cell 生成同样 `128+128` 行的均匀64-bit parity words，
  joint kernel 应为零维。
- discovery/validation/joint 分别计算 rank、nullity 和 canonical basis。
- discovery basis 必须逐向量通过独立 validation matrix；所有 basis 必须满足
  各自矩阵的 `M u = 0`。

行重排不会改变矩阵核，因此不把 row-shuffle 伪装成有效控制。错误 S-box 也不是
本文方法的同语义控制；随机输出矩阵才直接检验有限行数导致的偶然秩亏。

## 4. 裁决门

协议门：

```text
uKNIT公开完整轮测试向量仍通过
256把128-bit keys唯一
discovery/validation key sets互斥
parity cache shape = 10 rounds x 16 cells x 256 trials
all GF(2) bases validate
r1 all 16 cells rank/nullity = 0/64
all random-control joint nullities = 0
```

一个 round/cell 只有同时满足以下条件才算稳定非平凡核：

```text
joint nullity > 0
discovery nullity > 0
至少一个 discovery basis direction 在全部128个validation trials存活
joint basis在discovery和validation矩阵上都成立
random-control joint nullity = 0
```

输出报告最高通过轮、通过位置、mask、Hamming weight，以及后选择误接受上界
`2^(discovery_nullity - 128)`。`128+128` 是本地轮数 screen，不称论文级复现或
全密钥证明。所有计算都只针对公开算法的本地降轮实现，不连接网络目标、不使用
真实系统凭据，也不涉及未授权访问。

## 5. 下一步分支

```text
若 r7-r11 有稳定核：
  只确认最高轮及相邻一轮，升级到1000 discovery + 1000 validation、seed0/seed1

若最高稳定轮为 r5-r6：
  先做同结构1000+1000确认，再审查相邻双cell是否延长一轮

若最高稳定轮不超过 r4：
  低轮候选虽需保留，但优先转拓扑双cell搜索高轮扩展；不先扩大低轮确认预算

若 r3-r11 全部零核：
  停止单cell机械加密钥；转相邻双cell、256 plaintexts/multiset，同样128+128门

任何分支都不直接启动神经训练、远程GPU或256维VDS。
```

## 6. 产物

```text
run_id = i2_uknit_linear_integral_single_cell_round_census_256trials_seed0_20260730
outputs/local_audits/<run_id>/
  results.jsonl
  round_summary.csv
  kernel_basis.csv
  keys.npy
  base_plaintexts.npy
  parity_rows.npy
  random_control_rows.npy
  metadata.json
  progress.jsonl
  gate.json
  curves.svg
```

结果完成后必须把实测 metrics、claim scope 与证据支持的下一步写回本文，并刷新
`outputs/00_RECENT_RESULTS.md/.json`。图像只有通过 `visual-qa-redraw` 像素审查
后才算完成。

## 7. 2026-07-30 实测结果

全部协议门通过：uKNIT公开12轮向量匹配；256把128-bit密钥唯一且发现/验证集合
互斥；r1 的16个活动位置均为 `rank/nullity=0/64`；全部 GF(2) basis 满足
`M u=0`；r3-r11 的144个同预算随机输出控制均为 `rank/nullity=64/0`。

目标轮结果：

| 轮数 | 稳定非零核位置 | 最大 nullity | 最小 rank | 随机控制秩亏位置 |
|---:|---:|---:|---:|---:|
| r3 | 16/16 | 64 | 0 | 0/16 |
| r4 | 6/16 | 8 | 56 | 0/16 |
| r5 | 0/16 | 0 | 64 | 0/16 |
| r6-r11 | 每轮0/16 | 0 | 64 | 每轮0/16 |

r4 的稳定位置为 `0,2,3,4,5,6`，joint nullity 依次为 `8,6,6,3,1,1`。
每个位置的 discovery、validation、joint nullity 完全一致，所有 discovery basis
在128个 fresh-key/fresh-context validation trials 中100%存活。最弱的一维候选的
后选择随机误接受上界为 `2^-127`，最大八维候选为 `2^-120`。

裁决：

```text
status = pass
decision = innovation2_uknit_linear_kernel_candidate
highest_supported_round = r4
highest_supported_cells = 0,2,3,4,5,6
```

这说明论文方法在 uKNIT 上找到了可解释、跨新密钥和新上下文稳定的低轮输出平衡
子空间，但单活动 cell 没有推进到 r5，更不能称为高轮结果。由于确认 r4 的更多
密钥不会回答高轮问题，推荐下一步改为拓扑双 cell、256明文积分集合；唯一变量是
活动积分维数，仍保持64-bit raw parity和128+128独立验证协议。

权威产物：

```text
outputs/local_audits/
  i2_uknit_linear_integral_single_cell_round_census_256trials_seed0_20260730/
```

`curves.svg` 首次像素检查发现线性色标被 r3 的64维核主导，使 r4 的1--8维差异
颜色过淡。renderer 改用保留真实数值的幂次色标后重新渲染；最终1872x988像素图
通过 `visual-qa-redraw`，中文字体、标题、热图数值、图例、坐标与范围说明均无
重叠、裁切、缺字或歧义。

# 创新2 E9：PRESENT 稳定输出平衡子空间 readiness

**日期：** 2026-07-17
**状态：** 已完成 / hold / r6 当前结构族无稳定子空间
**前置裁决：** E8 停止 r6 当前单 mask 概率训练路线
**强基线：** Hwang et al. 2026 empirical parity-matrix kernel

## 1. 研究问题

E8 证明，PRESENT r6 的 5/6/7 活动 bit 结构在单个输出 mask 上几乎都接近
随机，且没有足够的跨密钥半重复信号。单 mask 失败不等于多个输出 bit 的线性
组合也没有积分性质。

E9 将输出对象从“一个 nibble mask 的平衡概率”升级为完整 64-bit 输出 parity
向量所定义的平衡 mask 子空间。先回答是否存在跨互斥密钥集合稳定的非平凡
子空间；不存在就不训练神经网络。

## 2. Parity matrix 与 kernel

对固定结构 `S` 和密钥 `K`：

```text
P(K,S) = XOR_{x in S} E_K^r(x) in F2^64
```

将 128 把 discovery keys 的 parity word 按行堆叠成 `M0 ∈ F2^(128×64)`，
另 128 把 validation keys 构造 `M1`。输出 mask `u ∈ F2^64` 平衡当且仅当：

```text
M u = 0
```

因此：

```text
B0     = ker(M0)
B1     = ker(M1)
Bjoint = ker([M0; M1]) = B0 ∩ B1
```

`dim(Bjoint)>0` 表示至少存在一个同时通过两组密钥的经验平衡线性 mask；仍不是
所有密钥的确定性证明。

## 3. 冻结矩阵

复用 E8 的结构生成种子和 5/6/7 活动 bit 定义：

```text
rounds                    = 4 known-fixture calibration, 5 reference, 6 target
active bit widths         = 5, 6, 7
structures per width      = 64
keys per structure        = 256
discovery / validation    = 128 / 128 mutually disjoint keys
plaintexts per structure  = 32, 64, 128
seed                      = 0
training                  = none
execution                 = local NumPy PRESENT + exact GF(2) elimination
```

r4 校准使用 16 个单活动 nibble 结构；PRESENT r4 的完整 64-bit 输出 XOR word
应在全部密钥上为零，因此 joint kernel 维数应为 64。r5 随机 5/6/7 bit 只作
参考，不承担必须发现 kernel 的校准职责，也不用于恢复被 E6 否定的 r5 神经
优势声明。r6 是唯一研究目标。

## 4. 指标

每个结构报告：

```text
rank(M0), rank(M1), rank([M0;M1])
dim ker(M0), dim ker(M1), dim ker([M0;M1])
discovery basis vectors surviving validation
joint-kernel canonical basis and mask weights
```

每个轮数/宽度汇总：

```text
nontrivial joint-kernel structure fraction
mean / median / max joint-kernel dimension
distinct nontrivial joint-kernel signatures
mean discovery-basis validation survival fraction
```

GF(2) RREF rank、kernel basis 和 `M u=0` 必须由单元测试与标量 parity fixture
交叉验证。

## 5. Gate

### r4 known-fixture calibration

```text
all 16 r4 single-active-nibble fixtures have joint-kernel dimension = 64
all reported joint basis vectors satisfy both M0 and M1
```

r4 fixture 不过时，E9 为 protocol invalid，不解释 r6。

### r6 target

某一宽度必须同时满足：

```text
nontrivial joint-kernel fraction >= 0.10
nontrivial joint-kernel structures >= 8
distinct nontrivial subspace signatures >= 4
mean discovery-basis validation survival fraction >= 0.50
```

若多个宽度通过，优先选择非平凡比例最高者；比例差小于 `0.05` 时选择明文复杂度
更低的宽度。

## 6. 后续边界

- r4 校准通过且 r6 至少一宽度通过：冻结“结构 -> joint-kernel dimension /
  balanced-mask candidate”预测计划；必须与直接 GF(2) kernel 比较密钥复杂度和
  未见结构泛化。
- r4 通过但 r6 全部不过：停止当前 r6 结构族，不能调神经网络；下一候选必须
  改输入结构族或使用文献中的 VDS/高活动维结构。
- r4 fixture 不过：修复 parity word、GF(2) elimination 或结构协议。

### 执行中校准修正

首次运行把随机 5/6/7 活动 bit 的 r5 非平凡 kernel 比例预注册为校准门，但实际
r5 只有一个 7-bit 结构出现二维 joint kernel，比例 `1/64`。与此同时，GF(2)
rank/basis、两半验证和标量 XOR 对拍全部通过。这说明失败来自校准假设没有已知
性质保证，而不是实现错误。

在不修改已揭盲 r6 指标、结构、密钥或目标门槛的前提下，校准改为密码分析上
已知的 r4 单活动 nibble fixture。标量预检已确认两个不同固定上下文、32 把密钥
下输出 XOR word 全零；正式修正版扩为 16 个 nibble 位置和同一 256 把密钥。

Hwang et al. 已提出经验 parity-matrix kernel。本项目不能主张 kernel 计算本身
为创新；潜在新增点只能是结构条件化、跨结构预测、减少 discovery keys，或预测
经典 kernel 尚未计算的新结构。

本实验不使用远程 GPU，不启动任何神经训练、H0 seed2 或二分类扩展。

## 7. 修正版实际结果

r4 已知 fixture、r5 参考和 r6 目标已使用同一组 256 把密钥完成；discovery 与
validation 各 128 把且互斥。GF(2) rank、RREF kernel basis、joint basis 双半
验证和标量/向量化 64-bit XOR word 对拍全部通过。

| 轮数 / 角色 | 活动 bit | 非平凡 joint-kernel 结构 | 比例 | 平均维数 | 最大维数 |
|---|---:|---:|---:|---:|---:|
| r4 已知 fixture | 4（完整 nibble） | `16/16` | `1.000000` | `64.000000` | `64` |
| r5 参考 | 5 | `0/64` | `0.000000` | `0.000000` | `0` |
| r5 参考 | 6 | `0/64` | `0.000000` | `0.000000` | `0` |
| r5 参考 | 7 | `1/64` | `0.015625` | `0.031250` | `2` |
| r6 目标 | 5 | `0/64` | `0.000000` | `0.000000` | `0` |
| r6 目标 | 6 | `0/64` | `0.000000` | `0.000000` | `0` |
| r6 目标 | 7 | `0/64` | `0.000000` | `0.000000` | `0` |

r4 的 16 个单活动 nibble 结构在全部 256 把密钥下输出 XOR word 都为零，因此
联合 kernel 维数均为 64，校准完整通过。r6 的 192 个目标结构全部 rank 64、
joint kernel 零维，目标 gate 全部失败。

正式裁决：

```text
status = hold
decision = innovation2_r6_stable_balance_subspace_not_found
r4_known_fixture_calibration_pass = true
r6_passing_widths = []
training = no
remote_scale = no
```

这排除了“单 mask 随机但 64-bit 线性组合仍稳定”的补救解释，范围仅限当前随机
5/6/7 活动 bit 结构族。它不排除文献中特制 VDS/活动结构的平衡子空间。

权威产物：

```text
outputs/local_audits/
  i2_present_stable_balance_subspace_r5_r6_bits5_6_7_seed0_20260717/
```

最终 `curves.svg` 经 `visual-qa-redraw` 以 `1800×858` 像素检查，标题、中文
glyph、坐标、图例、数据标签与裁决均无重叠或裁切；通过标记为
`visual_qa_passed.marker`。最近结果索引刷新通过，本结果为
`outputs/00_RECENT_RESULTS.md` 的 `001`。

## 8. 推荐下一步：PRESENT 文献结构校准

Hwang et al. 2026 附录 A.1 已给出直接可核验的 PRESENT 结构：7 轮、最后
16 个输入 bit 全活动，并报告四个稳定 kernel 基：

```text
b0
b4 XOR b12
b16 XOR b48
b20 XOR b28 XOR b52 XOR b60
```

E10 应先做 bit-order readiness：分别解释“最后16 bit”为本项目 LSB 侧
`0..15` 和 MSB 侧 `48..63`，使用流式 key chunks 计算完整 `2^16` 明文集合的
64-bit parity word，并检查上述四个 mask 在互斥密钥半上是否恒为零。只有匹配
文献基的方向才能进入更大 key 数 kernel 复现；两种方向都不匹配时先审计论文
bit numbering 和 PRESENT 状态映射。

这一步仍是强基线复现，不是神经创新。复现成功后，潜在新增问题才是能否根据
结构描述预测 kernel 属性或减少 discovery keys。

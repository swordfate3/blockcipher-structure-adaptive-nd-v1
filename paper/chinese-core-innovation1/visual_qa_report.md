# 主稿图表像素检查记录

检查时间：2026-08-03

检查对象为 `codex_manuscript.md` 实际引用的最终 SVG。所有图片均使用
Inkscape 渲染为约 1920 px 宽的 PNG 后检查，不以 SVG 解析成功代替像素检查。

| 图 | 最终文件 | 渲染尺寸 | 检查结果 |
|---|---|---:|---|
| 图 1 | `figures/fig_method_framework.svg` | 1920 x 912 | 通过；标题、四阶段框架、双专家分支、五条箭头和证据边界无重叠、裁切或流向歧义 |
| 图 2 | `figures/fig_uknit_k1bz_published_architecture_comparison.svg` | 1920 x 1056 | 通过；标题、热图、差值条、阈值和说明文字无重叠或裁切 |
| 图 3 | `figures/fig_uknit_k1u_semantic_position.svg` | 1920 x 1036 | 通过；两幅热图、双 seed 数值、语义/位置门槛和裁决完整可读 |
| 图 4 | `figures/fig_uknit_k1bv_boundary.svg` | 1920 x 1053 | 重绘后通过；以 AUC 热图和两类差值替换含义不明的 accuracy 虚线图 |
| 图 5 | `figures/fig_dialga_d2_same_checkpoint.svg` | 1960 x 1040 | 通过；六个柱、数值、随机水平和坐标标签完整可读 |
| 图 6 | `figures/fig_dialga_dmc2_auc.svg` | 1920 x 1053 | 修改后通过；删除继续 DFC1 的旧裁决，改为 262144/class 论文上限 |

最终六图均未发现文字遮挡、标签裁切、缺字、图例歧义或导出边界问题。
图 4 的结论以 AUC 为主；图 6 仍须与正文一起保留“原始回收、无独立最终
测试、非正式规模复现”的证据限定。

K1-CA/K1-CB 的五模型主图尚未生成，因为对应结果尚未回收。该图属于投稿阻塞项，生成后必须追加一次独立渲染像素检查；本记录不能提前替代其验收。

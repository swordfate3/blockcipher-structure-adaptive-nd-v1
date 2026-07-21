from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


DEFAULT_RESULT_ROOTS = (
    "local_smoke",
    "local_readiness",
    "local_diagnostic",
    "local_audits",
    "smoke",
    "remote_results",
    "remote_results_incomplete",
)

DEFAULT_INDEX_LIMIT = 30
DEFAULT_RETENTION_DAYS = 7
_SECONDS_PER_DAY = 24 * 60 * 60

_SCOPE_PRIORITY = {
    "remote_results": 0,
    "remote_results_incomplete": 1,
    "local_diagnostic": 2,
    "local_readiness": 3,
    "local_audits": 4,
    "local_smoke": 5,
    "smoke": 6,
}

ARTIFACT_LABELS = {
    "curves": "曲线",
    "gate": "门控",
    "validation": "验证",
    "results": "结果",
    "history": "历史",
    "progress": "进度",
}

DECISION_LABELS = {
    "innovation2_present_full_state_next_round_criticality_not_identifiable": (
        "完整PRESENT轮间状态一对即可恢复当轮子密钥，不能测量随机猜测临界轮"
    ),
    "innovation2_present_full_state_next_round_identifiability_protocol_invalid": (
        "PRESENT轮边界、逆映射、密钥调度或可识别性审计产物无效"
    ),
    "innovation2_topology_bottleneck_local_smoke_passed": (
        "PRESENT三轮低秩拓扑瓶颈四行实现门通过，开放第五密钥正式实验"
    ),
    "innovation2_topology_bottleneck_ready_for_independent_confirmation": (
        "拓扑瓶颈兼顾真实输出预测、错误P归因和标签打乱控制，进入第六密钥确认"
    ),
    "innovation2_topology_bottleneck_attributed_with_performance_cost": (
        "拓扑瓶颈超过错误P与标签打乱，但相对原SPN锚点存在性能代价"
    ),
    "innovation2_topology_bottleneck_not_attributed": (
        "低秩位置条件未使真实P超过错误P与标签打乱，停止该瓶颈路线"
    ),
    "innovation2_topology_bottleneck_protocol_invalid": (
        "OPB1的OPA3所有权、数据、低秩模型、控制、训练或产物协议无效"
    ),
    "innovation2_selected8_topology_attribution_local_smoke_passed": (
        "PRESENT三轮真实/identity/错误P-layer同容量归因实现门通过"
    ),
    "innovation2_selected8_present_topology_independently_attributed": (
        "真实PRESENT P-layer在第四固定密钥上超过两个同容量拓扑控制"
    ),
    "innovation2_selected8_present_topology_not_attributed": (
        "PRESENT结构网络有效但真实P-layer未超过同容量拓扑控制"
    ),
    "innovation2_selected8_topology_attribution_protocol_invalid": (
        "P-layer映射、OPA2所有权、数据、训练或产物协议无效"
    ),
    "innovation2_selected8_architecture_confirmation_local_smoke_passed": (
        "第四密钥候选/MLP匹配shuffle确认实现门通过，开放OPA2远程正式确认"
    ),
    "innovation2_selected8_architecture_priority_independently_confirmed": (
        "OPA1候选在第四固定密钥上超过MLP及匹配shuffle，架构优先得到独立确认"
    ),
    "innovation2_selected8_mlp_retained_after_architecture_confirmation": (
        "OPA1候选未通过第四密钥匹配控制确认，保留专用八输出MLP"
    ),
    "innovation2_selected8_architecture_confirmation_protocol_invalid": (
        "第四密钥确认的候选所有权、数据、匹配shuffle、训练或产物协议无效"
    ),
    "innovation2_selected8_architecture_screen_local_smoke_passed": (
        "PRESENT三轮固定八输出五模型实现门通过，开放第三密钥发现屏"
    ),
    "innovation2_selected8_architecture_candidate_requires_confirmation": (
        "发现非MLP架构候选，必须进入第四密钥与匹配shuffle独立确认"
    ),
    "innovation2_selected8_mlp_anchor_retained_after_screen": (
        "非MLP候选未通过预注册增益门，保留专用八输出MLP锚点"
    ),
    "innovation2_selected8_architecture_protocol_invalid": (
        "多架构发现屏的数据、位序、参数预算、训练或产物协议无效"
    ),
    "innovation2_r4_structured_xor_local_smoke_passed": (
        "PRESENT四轮真实输出结构化XOR实现门通过，开放远程正式归因门"
    ),
    "innovation2_r4_structured_xor_supported": (
        "至少一个四轮结构化XOR家族超过单bit与全部匹配控制"
    ),
    "innovation2_r4_structured_xor_not_supported": (
        "四轮结构化XOR未超过单bit与全部匹配控制，停止机械扩展"
    ),
    "innovation2_r4_structured_xor_protocol_invalid": (
        "四轮真实输出XOR的数据、mask、控制、指标、缓存或checkpoint协议无效"
    ),
    "innovation2_selected8_independent_key_local_smoke_passed": (
        "固定八输出bit独立密钥实现门通过，开放OP11远程正式确认"
    ),
    "innovation2_selected8_cross_key_and_dedicated_head_supported": (
        "至少4/8个固定输出bit跨密钥确认，且专用八输出头优于完整输出anchor"
    ),
    "innovation2_selected8_cross_key_supported_without_head_gain": (
        "至少4/8个固定输出bit跨密钥确认，但专用八输出头没有额外增益"
    ),
    "innovation2_selected8_not_cross_key_supported": (
        "固定八输出bit未通过第二把秘密密钥确认，转密钥条件性审计"
    ),
    "innovation2_selected8_independent_key_protocol_invalid": (
        "固定位置、独立密钥、数据拆分、匹配打乱或checkpoint协议无效"
    ),
    "innovation2_output_bit_discovery_local_smoke_passed": (
        "逐bit候选发现与独立fresh确认实现门通过，等待OP9远程checkpoint"
    ),
    "innovation2_true_output_bits_fresh_confirmed": (
        "至少一个PRESENT三轮真实密文输出bit通过全新明文确认"
    ),
    "innovation2_no_true_output_bit_fresh_confirmed": (
        "没有输出bit通过全新明文确认，停止完整输出机械扩展"
    ),
    "innovation2_output_bit_discovery_protocol_invalid": (
        "逐bit发现的checkpoint、bit顺序、数据拆分或候选冻结协议无效"
    ),
    "innovation2_output_prediction_kimura_lstm_local_smoke_passed": (
        "Kimura式完整输出LSTM本地实现门通过，只开放远程单密钥论文规模校准"
    ),
    "innovation2_output_prediction_kimura_lstm_single_key_supported": (
        "PRESENT三轮完整输出LSTM单固定密钥校准通过，只开放第二密钥确认"
    ),
    "innovation2_output_prediction_kimura_lstm_single_key_not_supported": (
        "PRESENT三轮完整输出LSTM单密钥校准未过门，停止论文族机械扩展"
    ),
    "innovation2_output_prediction_kimura_lstm_protocol_invalid": (
        "Kimura式完整输出预测的数据、模型、指标、缓存或checkpoint协议无效"
    ),
    "raw_fallback_incomplete": (
        "远程原始产物已回收，但尚未形成通过验证的结果分支或最终裁决"
    ),
    "feistel_balanced_relation_readiness_passed": (
        "SIMON/SIMECK 轮关系实现就绪，进入 2048/类本地诊断"
    ),
    "feistel_balanced_relation_two_cipher_seed0_pass": (
        "两种密码均通过轮关系归因门，进入 seed1 同预算确认"
    ),
    "feistel_balanced_relation_cipher_conditional": (
        "轮关系收益仅在一种密码成立，先审计错误轮函数对照"
    ),
    "feistel_balanced_signal_without_relation_attribution": (
        "存在区分信号但无法归因于真实轮关系，停止扩规模"
    ),
    "feistel_balanced_relation_not_ready": ("轮关系候选未就绪，先做更低轮公式校准"),
    "feistel_balanced_easier_round_calibrated": (
        "低一轮公式与网络已校准，下一步贴近 Lu SE-ResNet 高轮协议"
    ),
    "feistel_balanced_easier_round_cipher_conditional": (
        "低轮校准仅一种密码通过，先审计另一种轮函数"
    ),
    "feistel_balanced_easier_round_signal_without_attribution": (
        "低轮有信号但真实轮关系无独立贡献，停止该架构扩展"
    ),
    "feistel_balanced_easier_round_not_calibrated": (
        "低一轮仍未校准，转作者代码逐样本数据与布局对拍"
    ),
    "feistel_lu_layout_two_cipher_calibrated": (
        "Lu-SE 布局两种密码均校准，进入高一轮同布局比较"
    ),
    "feistel_lu_layout_cipher_conditional": ("Lu-SE 布局仅一种密码通过，保留条件路线"),
    "feistel_lu_layout_signal_without_architecture_gain": (
        "Lu-SE 布局有信号但未优于旧锚点，停止布局扩展"
    ),
    "feistel_lu_layout_not_calibrated": ("Lu-SE 布局未校准，先量化数据规模缺口"),
    "feistel_relation_scale_slope_two_cipher_pass": (
        "两种密码均有正数据斜率，进入独立 seed1 同规模确认"
    ),
    "feistel_relation_scale_slope_cipher_conditional": (
        "数据斜率仅一种密码成立，只确认通过的密码"
    ),
    "feistel_relation_signal_without_scale_slope": (
        "仍有低轮信号但样本斜率不足，停止机械扩规模"
    ),
    "feistel_relation_scale_probe_not_ready": (
        "8192/类规模探针未就绪，重新评估 Feistel 路线优先级"
    ),
    "feistel_relation_8192_seed1_confirmation_pass": (
        "独立 seed1 信号与轮关系归因通过，进入双 seed 综合"
    ),
    "feistel_relation_8192_seed1_cipher_conditional": (
        "独立 seed1 只确认一种密码，保留条件路线"
    ),
    "feistel_relation_8192_seed1_confirmation_failed": (
        "独立 seed1 未确认，停止扩规模"
    ),
    "feistel_target_round_8192_two_cipher_pass": (
        "论文目标轮两种密码均过门，进入独立 seed1 确认"
    ),
    "feistel_target_round_8192_cipher_conditional": (
        "论文目标轮信号仅一种密码成立，保留条件路线"
    ),
    "feistel_target_round_signal_without_scale_slope": (
        "论文目标轮有信号但规模增益不足，保留低轮证据"
    ),
    "feistel_target_round_8192_not_ready": ("论文目标轮 8192/类未就绪，停止远程扩规模"),
    "feistel_curriculum_readiness_passed": (
        "同总轮次课程训练机制就绪，进入 8192/类本地诊断"
    ),
    "feistel_curriculum_two_cipher_pass": (
        "两种密码均通过课程迁移门，进入独立 seed1 确认"
    ),
    "feistel_curriculum_cipher_conditional": ("课程迁移仅一种密码通过，只确认条件路线"),
    "feistel_curriculum_without_scratch_gain": (
        "课程模型有信号但未胜同轮次从零训练，停止课程路线"
    ),
    "feistel_curriculum_without_relation_attribution": (
        "课程信号无法归因于真实轮关系，保留目标轮对照"
    ),
    "feistel_curriculum_target_signal_not_ready": (
        "课程训练未恢复目标轮信号，转表示或差分重设计"
    ),
    "feistel_curriculum_seed1_confirmation_pass": (
        "SIMECK 独立 seed1 课程迁移通过，进入双 seed 综合"
    ),
    "feistel_curriculum_seed1_confirmation_failed": (
        "SIMECK 独立 seed1 未确认，停止课程路线"
    ),
    "feistel_simeck_curriculum_65k_scale_pass": (
        "SIMECK 65536/类课程信号与规模保持通过，进入同规模 seed1"
    ),
    "feistel_simeck_curriculum_65k_scale_regressed": (
        "SIMECK 课程控制过门但规模性能回退，保留本地双 seed 证据"
    ),
    "feistel_simeck_curriculum_65k_not_ready": (
        "SIMECK 65536/类课程规模门未通过，停止远程扩展"
    ),
    "innovation2_integral_property_implementation_ready": (
        "创新2积分性质预测实现就绪，可进入本地诊断"
    ),
    "innovation2_integral_property_smoke_invalid": "创新2积分性质预测 Smoke 无效，先修协议",
    "innovation2_integral_property_advance_multiseed": (
        "结构条件积分概率门控通过，进入多 seed 与经典基线"
    ),
    "innovation2_integral_property_invalid_control": "控制无效，审计拆分与标签泄漏",
    "innovation2_integral_property_linear_signal_only": (
        "信号可由线性基线解释，先补确定性基线"
    ),
    "innovation2_integral_property_redesign_before_scale": (
        "结构排序有信号但概率误差未过门，校准后再扩展"
    ),
    "innovation2_integral_calibration_implementation_ready": (
        "创新2 E1 校准与标签稳定性实现就绪，可进入本地诊断"
    ),
    "innovation2_integral_geometry_holdout_implementation_ready": (
        "创新2 E4 几何组合留出实现就绪，可进入本地诊断"
    ),
    "innovation2_integral_geometry_holdout_smoke_invalid": (
        "创新2 E4 几何组合留出实现未就绪，先修复数据所有权或产物链"
    ),
    "innovation2_integral_calibration_smoke_invalid": (
        "创新2 E1 Smoke 无效，先修校准协议"
    ),
    "innovation2_integral_calibration_invalid_control": (
        "创新2 E1 控制无效，审计拆分与校准"
    ),
    "innovation2_integral_calibration_advance_seed1_geometry": (
        "校准概率门控通过，进入 seed1 与几何组合留出"
    ),
    "innovation2_integral_rate_target_unstable": (
        "32-key 概率标签不稳定，改用区间或排序目标"
    ),
    "innovation2_integral_calibration_insufficient": (
        "标签稳定但校准仍不足，下一步仅加入 P-layer 可达性特征"
    ),
    "innovation2_integral_ranking_utility_advance_independent_confirmation": (
        "排序与 top-16 筛选效用过门，进入独立 seed1 确认"
    ),
    "innovation2_integral_ranking_utility_independent_confirmation_passed": (
        "独立 seed1 排序与 top-16 效用通过，进入双 seed 联合裁决"
    ),
    "innovation2_integral_ranking_utility_two_seed_confirmed": (
        "双 seed 排序与 top-16 效用确认，进入几何组合留出"
    ),
    "innovation2_integral_ranking_utility_two_seed_not_confirmed": (
        "双 seed 排序效用未确认，停止几何留出与扩规模"
    ),
    "innovation2_integral_ranking_control_not_attributed": (
        "打乱控制也有 top-16 优势，当前筛选效用不可归因"
    ),
    "innovation2_integral_ranking_explanatory_only": (
        "只有排序相关性过门，保留为解释性证据"
    ),
    "innovation2_integral_ranking_redesign_representation": (
        "当前结构表示无稳定排序效用，先加入 P-layer 可达性特征"
    ),
    "innovation2_integral_geometry_holdout_passed": (
        "几何组合留出排序效用通过，进入精确积分认证桥接"
    ),
    "innovation2_integral_geometry_holdout_not_confirmed": (
        "几何组合留出未确认，停止扩规模并重设结构表示"
    ),
    "innovation2_integral_fresh_key_implementation_ready": (
        "创新2 E5 全新密钥富集验证实现就绪，可运行 4096-key 矩阵"
    ),
    "innovation2_integral_fresh_key_smoke_invalid": (
        "创新2 E5 实现未就绪，先修复选择器或批量 parity"
    ),
    "innovation2_integral_fresh_key_protocol_invalid": (
        "创新2 E5 协议无效，不解释选择器指标"
    ),
    "innovation2_integral_fresh_key_enrichment_passed": (
        "创新2 E5 全新密钥候选富集通过，冻结实验并进入论文写作"
    ),
    "innovation2_integral_fresh_key_ranking_only": (
        "创新2 E5 仅排序富集通过，按经验贡献进入论文写作"
    ),
    "innovation2_integral_fresh_key_enrichment_not_confirmed": (
        "创新2 E5 全新密钥富集未确认，保留 E4 诊断边界"
    ),
    "innovation2_integral_position_prior_audit_ready": (
        "创新2 E6 输出位置先验归因实现就绪，可运行 4096-key 审判"
    ),
    "innovation2_integral_position_prior_smoke_invalid": (
        "创新2 E6 实现未就绪，先修复训练重建或位置匹配"
    ),
    "innovation2_integral_position_prior_protocol_invalid": (
        "创新2 E6 协议无效，不解释位置归因指标"
    ),
    "innovation2_integral_neural_interaction_residual_supported": (
        "创新2 E6 神经结构交互残差通过，可按位置感知方法写入论文"
    ),
    "innovation2_integral_position_prior_dominant_with_conditional_residual": (
        "创新2 E6 全局收益由位置先验主导，仅保留位置内神经残差"
    ),
    "innovation2_integral_position_prior_explains_enrichment": (
        "创新2 E6 位置先验解释候选富集，停止神经优势声明"
    ),
    "innovation2_r6_two_nibble_output_prediction_benchmark_ready": (
        "r6 两活动 nibble 输出性质 benchmark 通过，可进入本地训练"
    ),
    "innovation2_r6_two_nibble_almost_always_balanced": (
        "r6 两活动 nibble 几乎总平衡，转 r7 标签过渡审计"
    ),
    "innovation2_r6_two_nibble_output_prediction_benchmark_not_ready": (
        "r6 两活动 nibble 的位置残差不足，转 5--7 活动 bit 审计"
    ),
    "innovation2_output_property_transition_audit_invalid": (
        "输出性质过渡审计无效，先修数据或校验"
    ),
    "innovation2_r6_active_bit_transition_benchmark_ready": (
        "r6 细粒度活动 bit 输出性质 benchmark 通过，可进入本地训练"
    ),
    "innovation2_r6_active_bit_transition_benchmark_not_ready": (
        "r6 的 5--7 活动 bit 均无可重复结构信号，停止当前单 mask 训练路线"
    ),
    "innovation2_r6_active_bit_transition_audit_invalid": (
        "r6 活动 bit 过渡审计无效，先修复数据、密钥拆分或统计校验"
    ),
    "innovation2_r6_stable_balance_subspace_ready": (
        "r6 稳定平衡子空间存在，可进入结构条件 kernel 属性预测"
    ),
    "innovation2_r6_stable_balance_subspace_not_found": (
        "r4 校准通过，r6 当前结构族无稳定平衡子空间"
    ),
    "innovation2_stable_balance_subspace_protocol_invalid": (
        "平衡子空间协议未校准，先修 parity word 或 GF(2) kernel"
    ),
    "innovation2_present_r7_hwang_bitorder_ready": (
        "唯一 bit-order 复现论文输出 mask，可扩大新密钥复核"
    ),
    "innovation2_present_r7_hwang_bitorder_ambiguous": (
        "多个 bit-order 通过，需增加密钥并核对论文状态布局"
    ),
    "innovation2_present_r7_hwang_bitorder_not_reproduced": (
        "当前映射未复现论文输出 mask，停止训练并审计协议"
    ),
    "innovation2_present_r7_hwang_protocol_invalid": (
        "PRESENT 7轮论文 kernel 协议校验无效，先修实现"
    ),
    "innovation2_present_r7_hwang_kernel_reproduced": (
        "128把新密钥复现论文四维输出 kernel，进入结构族扩展"
    ),
    "innovation2_present_r7_hwang_kernel_underconstrained": (
        "论文 mask 稳定，但128把密钥下的经验 kernel 仍高于四维"
    ),
    "innovation2_present_r7_hwang_kernel_not_reproduced": (
        "论文 mask 未在两组新密钥上保持稳定，停止训练并审计协议"
    ),
    "innovation2_present_r7_hwang_convergence_protocol_invalid": (
        "论文四维 kernel 收敛协议无效，先修复实现或密钥拆分"
    ),
    "innovation2_present_r7_active_block_kernel_diversity_ready": (
        "不同活动块产生多个稳定输出 kernel，可构造结构条件标签表"
    ),
    "innovation2_present_r7_active_block_kernel_not_diverse": (
        "活动块位置未形成足够的输出 kernel 多样性"
    ),
    "innovation2_present_r7_active_block_diversity_protocol_invalid": (
        "活动块 kernel 多样性协议无效，先修复结构或论文 anchor"
    ),
    "innovation2_output_label_interaction_ready": (
        "边际基线未完全解释结构-mask标签，可扩大结构族"
    ),
    "innovation2_output_label_shortcut_dominated": (
        "活动块+mask简单边际已解释标签，禁止直接训练神经网络"
    ),
    "innovation2_output_label_readiness_protocol_invalid": (
        "结构-mask标签源证据或构造无效，先修复协议"
    ),
    "innovation2_cyclic_geometry_kernel_diversity_ready": (
        "循环滑动活动几何形成足够多稳定 kernel，可重建扩展标签表"
    ),
    "innovation2_cyclic_geometry_kernel_diversity_insufficient": (
        "循环滑动活动几何的输出 kernel 多样性不足"
    ),
    "innovation2_cyclic_geometry_diversity_protocol_invalid": (
        "循环活动几何协议或 Hwang anchor 校验无效"
    ),
    "innovation2_topology_geometry_kernel_diversity_ready": (
        "P-layer拓扑活动几何形成足够多稳定 kernel，可重建标签表"
    ),
    "innovation2_topology_geometry_kernel_diversity_insufficient": (
        "P-layer拓扑活动几何的输出 kernel 多样性不足"
    ),
    "innovation2_topology_geometry_protocol_invalid": (
        "P-layer拓扑几何协议或 Hwang anchor 校验无效"
    ),
    "innovation2_inactive_context_kernel_diversity_ready": (
        "固定上下文形成足够多稳定 kernel，可重建标签表"
    ),
    "innovation2_inactive_context_kernel_diversity_insufficient": (
        "固定上下文的输出 kernel 多样性不足"
    ),
    "innovation2_inactive_context_protocol_invalid": (
        "固定上下文协议或 Hwang anchor 校验无效"
    ),
    "innovation2_context_label_interaction_ready": (
        "强基线未解释 context-mask 交互，可做 fresh-key 验证"
    ),
    "innovation2_context_label_shortcut_dominated": (
        "简单 context/mask 捷径已解释标签，禁止训练"
    ),
    "innovation2_context_label_readiness_protocol_invalid": (
        "E16源证据或 context-mask 标签构造无效"
    ),
    "innovation2_equal_prevalence_context_label_ready": (
        "等流行率翻转-mask 标签未被强捷径解释，可做 fresh-key 验证"
    ),
    "innovation2_equal_prevalence_context_label_shortcut_dominated": (
        "等流行率翻转-mask 标签仍被简单捷径解释，禁止训练"
    ),
    "innovation2_equal_prevalence_label_protocol_invalid": (
        "E16 span 或等流行率翻转-mask 标签构造无效"
    ),
    "innovation2_group_disjoint_shortcuts_controlled": (
        "组外线性捷径受控，可进入 fresh-key 稳定性验证"
    ),
    "innovation2_group_disjoint_shortcut_generalizes": (
        "至少一种组外拆分仍保留位模式捷径，禁止训练"
    ),
    "innovation2_group_disjoint_protocol_invalid": (
        "context/mask 组外拆分、覆盖或源标签无效"
    ),
    "innovation2_fresh_expanded_context_kernel_ready": (
        "fresh-key稳定且64-context kernel 多样性充足，可重建标签"
    ),
    "innovation2_context_kernel_fresh_key_unstable": (
        "E16 context kernel 签名未在全新密钥上复现"
    ),
    "innovation2_fresh_expanded_context_diversity_insufficient": (
        "fresh-key稳定但新增 context kernel 多样性不足"
    ),
    "innovation2_fresh_context_protocol_invalid": (
        "fresh-key、E16 source 或 Hwang anchor 协议无效"
    ),
    "innovation2_balance_rate_interaction_ready": (
        "跨密钥平衡率 interaction 残差可重复，可设计连续预测"
    ),
    "innovation2_balance_rate_interaction_not_reproducible": (
        "跨密钥平衡率 interaction 弱、噪声化或被控制解释"
    ),
    "innovation2_balance_rate_protocol_invalid": (
        "E18重放、XOR缓存、mask网格或标量校验无效"
    ),
    "innovation2_skinny_r7_hwang_kernel_reproduced": (
        "SKINNY-64/64 7轮 Hwang 18维 kernel 已精确复现"
    ),
    "innovation2_skinny_r7_hwang_kernel_not_reproduced": (
        "SKINNY 7轮论文 kernel 未完整复现，先审计协议"
    ),
    "innovation2_skinny_r7_hwang_protocol_invalid": (
        "SKINNY向量、缓存、拆分或GF(2)协议无效"
    ),
    "innovation2_skinny_r8_hwang_kernel_reproduced": (
        "SKINNY-64/64 8轮 two-active-cell 一维 kernel 已精确复现"
    ),
    "innovation2_skinny_r8_hwang_kernel_not_reproduced": (
        "SKINNY 8轮论文 kernel 未完整复现，先审计协议"
    ),
    "innovation2_skinny_r8_hwang_protocol_invalid": (
        "SKINNY向量、密钥所有权、缓存或GF(2)协议无效"
    ),
    "innovation2_skinny_r8_geometry_kernel_diversity_ready": (
        "SKINNY 8轮相邻活动pair形成足够多稳定kernel，可构造标签"
    ),
    "innovation2_skinny_r8_geometry_kernel_not_diverse": (
        "SKINNY 8轮相邻活动pair的稳定kernel或签名不足"
    ),
    "innovation2_skinny_r8_geometry_protocol_invalid": (
        "SKINNY论文anchor、密钥、缓存或GF(2)协议无效"
    ),
    "innovation2_skinny_r8_bottom_row_pair_family_ready": (
        "SKINNY 8轮底行pair形成足够多稳定kernel，可进入标签捷径审计"
    ),
    "innovation2_skinny_r8_bottom_row_anchor_not_reproduced": (
        "SKINNY 8轮E22已知方向未在全新密钥上全部复现"
    ),
    "innovation2_skinny_r8_bottom_row_pair_family_not_closed": (
        "SKINNY 8轮底行pair稳定kernel仅4/6，未达到闭合门"
    ),
    "innovation2_skinny_r8_bottom_row_protocol_invalid": (
        "SKINNY底行pair、密钥、缓存、公开向量或GF(2)协议无效"
    ),
    "innovation2_skinny_r7_single_cell_kernel_diversity_ready": (
        "SKINNY 7轮单活动cell形成足够多稳定kernel，可进入标签捷径审计"
    ),
    "innovation2_skinny_r7_single_cell_anchor_not_reproduced": (
        "SKINNY 7轮Hwang anchor未在全新密钥上复现"
    ),
    "innovation2_skinny_r7_single_cell_kernel_not_diverse": (
        "SKINNY 7轮稳定位置kernel仅4/16，未达到标签宽度门"
    ),
    "innovation2_skinny_r7_single_cell_protocol_invalid": (
        "SKINNY单活动cell、密钥、缓存、公开向量或GF(2)协议无效"
    ),
    "innovation2_speck_hwang_phase_b_single_key_timing_ready": (
        "SPECK精确2^30单key计时门通过；尚不是多key kernel复现"
    ),
    "innovation2_speck_hwang_phase_b_direct_enumeration_not_scalable": (
        "SPECK精确2^30直接枚举时间或显存不可扩展"
    ),
    "innovation2_speck_hwang_phase_b_protocol_invalid": (
        "SPECK结构、CUDA、缓存、计时或论文mask协议无效"
    ),
    "innovation2_speck_hwang_phase_c_kernel_reproduced": (
        "SPECK 6/7轮论文kernel在32+32新密钥上精确复现，位置控制通过"
    ),
    "innovation2_speck_hwang_phase_c_position_control_not_specific": (
        "SPECK论文mask也在位置控制中成立，结构特异性不足，禁止训练"
    ),
    "innovation2_speck_hwang_phase_c_kernel_not_reproduced": (
        "SPECK 6/7轮论文kernel未在32+32新密钥上精确复现"
    ),
    "innovation2_speck_hwang_phase_c_protocol_invalid": (
        "SPECK Phase C密钥、结构、CUDA缓存、计时或GF(2)协议无效"
    ),
    "innovation2_speck_hwang_context_invariant": (
        "SPECK四种固定值共享相同6/7轮论文kernel；固定值应视为无关上下文"
    ),
    "innovation2_speck_hwang_context_dependent_stable": (
        "SPECK固定值产生多个跨密钥稳定kernel；进入context/mask捷径审计"
    ),
    "innovation2_speck_hwang_context_family_not_stable": (
        "SPECK固定值kernel未形成稳定不变或多样结构族"
    ),
    "innovation2_speck_hwang_context_protocol_invalid": (
        "SPECK context baseline、推导、CUDA缓存、计时或GF(2)协议无效"
    ),
    "innovation2_speck_hwang_position_family_advance": (
        "SPECK固定位置族满足正负标签数量与跨word覆盖；进入组外捷径审计"
    ),
    "innovation2_speck_hwang_position_family_narrow": (
        "SPECK固定位置族过窄；暂停训练并评估非相邻或旋转等价结构"
    ),
    "innovation2_speck_hwang_position_family_anchor_only": (
        "SPECK仅论文固定位置保持稳定；停止当前位置标签路线"
    ),
    "innovation2_speck_hwang_position_family_protocol_invalid": (
        "SPECK位置族baseline、映射、缓存、计时或GF(2)协议无效"
    ),
    "innovation2_speck_position_label_grid_advance": (
        "SPECK位置×mask标签宽度和组外捷径门通过；进入fresh-key与结构扩展"
    ),
    "innovation2_speck_position_label_grid_shortcut_dominated": (
        "SPECK位置×mask标签可被简单位模式捷径泛化解释"
    ),
    "innovation2_speck_position_label_grid_too_narrow": (
        "SPECK位置×mask完整位置、flipping mask或标签签名不足"
    ),
    "innovation2_speck_position_label_protocol_invalid": (
        "SPECK E28来源、kernel标签网格或组外评价协议无效"
    ),
    "innovation2_speck_topology_aligned_family": (
        "SPECK真实ROR7模加对齐形成稳定跨word位置族，超过错位控制"
    ),
    "innovation2_speck_topology_pair_not_specific": (
        "SPECK真实ROR7模加对齐未超过offset-minus-one错位控制"
    ),
    "innovation2_speck_topology_pair_too_narrow": (
        "SPECK真实ROR7模加对齐仅形成一个稳定lane，标签族仍过窄"
    ),
    "innovation2_speck_topology_pair_no_signal": (
        "SPECK真实ROR7模加对齐没有64-key稳定lane"
    ),
    "innovation2_speck_topology_pair_protocol_invalid": (
        "SPECK拓扑pair映射、密钥、缓存、计时或GF(2)协议无效"
    ),
    "innovation2_present_linear_subspace_readiness_passed": (
        "PRESENT 16维线性子空间kernel实现就绪，可运行冻结E30审计"
    ),
    "innovation2_present_linear_subspace_kernel_family_ready": (
        "PRESENT随机orientation形成足够宽的跨密钥稳定kernel族"
    ),
    "innovation2_present_linear_subspace_kernel_family_too_sparse": (
        "PRESENT随机orientation的稳定kernel或签名仍不足"
    ),
    "innovation2_present_linear_subspace_protocol_invalid": (
        "PRESENT线性子空间RREF、密钥、缓存、向量化或GF(2)协议无效"
    ),
    "innovation2_deterministic_provider_ready": (
        "确定性提供者契约完整，可进入高轮structure×linear-mask标签atlas审计"
    ),
    "innovation2_deterministic_provider_semantics_mismatch": (
        "现有确定性结果的常数值、输出函数或负类语义不匹配当前标签"
    ),
    "innovation2_deterministic_provider_runtime_unavailable": (
        "同目标确定性提供者当前依赖不可用运行时或商业求解器"
    ),
    "innovation2_deterministic_provider_protocol_invalid": (
        "确定性提供者版本、安全解析、文件集或GF(2)契约无效"
    ),
    "innovation2_small_spn_exact_label_readiness_passed": (
        "小状态SPN全key精确标签实现就绪，可运行冻结E32审计"
    ),
    "innovation2_small_spn_exact_label_family_ready": (
        "小状态SPN精确标签宽度与组外反捷径门通过，可准备E33网络比较"
    ),
    "innovation2_small_spn_exact_label_shortcut_dominated": (
        "小状态SPN标签可被组外位置或ID边际解释，禁止训练图网络"
    ),
    "innovation2_small_spn_exact_label_too_narrow": (
        "小状态SPN精确标签的正负宽度、签名或跨cipher交互不足"
    ),
    "innovation2_small_spn_exact_label_protocol_invalid": (
        "小状态SPN双射、全key覆盖、缓存、parity或标签协议无效"
    ),
    "innovation2_small_spn_matched_contrast_ready": (
        "训练内matched-contrast宽度与组外反捷径门通过，可准备E33网络比较"
    ),
    "innovation2_small_spn_matched_contrast_still_shortcut_dominated": (
        "matched-contrast标签仍可被组外ID边际解释，停止当前benchmark"
    ),
    "innovation2_small_spn_matched_contrast_too_narrow": (
        "matched-contrast的cell、类别或topology pattern不足"
    ),
    "innovation2_small_spn_matched_contrast_protocol_invalid": (
        "matched-contrast来源、train-only选择、shape或索引协议无效"
    ),
    "innovation2_small_spn_topology_training_readiness_passed": (
        "小状态SPN GraphGPS/SCGT训练实现就绪，可运行冻结两seed归因矩阵"
    ),
    "innovation2_small_spn_topology_predictor_ready": (
        "真实SPN拓扑预测器超过边际与错误拓扑控制，可进入真实密码迁移readiness"
    ),
    "innovation2_small_spn_topology_signal_not_attributed": (
        "神经收益未归因于真实SPN拓扑或label-shuffle控制异常"
    ),
    "innovation2_small_spn_topology_predictor_not_ready": (
        "GraphGPS未稳定超过冻结ID边际，停止当前合成网络路线"
    ),
    "innovation2_small_spn_topology_training_protocol_invalid": (
        "GraphGPS/SCGT来源、split、forward、checkpoint或metric协议无效"
    ),
    "innovation2_small_spn_cell_equivariance_readiness_passed": (
        "cell重标号等变表示与控制路径就绪，可运行冻结两seed归因矩阵"
    ),
    "innovation2_small_spn_cell_equivariance_repair_confirmed": (
        "等变GraphGPS超过边际与错误P-layer，只开放同表示SCGT增益审计"
    ),
    "innovation2_small_spn_cell_equivariance_topology_not_attributed": (
        "等变表示收益仍未归因于真实P-layer，停止拓扑贡献声明"
    ),
    "innovation2_small_spn_cell_equivariance_repair_not_ready": (
        "cell等变修复未过冻结门，停止当前GraphGPS表示路线"
    ),
    "innovation2_small_spn_cell_equivariance_protocol_invalid": (
        "cell重标号等变、来源、split、forward或metric协议无效"
    ),
    "innovation2_small_spn_round_shared_readiness_passed": (
        "按实际轮数循环的共享图处理器就绪，可运行冻结两seed归因矩阵"
    ),
    "innovation2_small_spn_round_shared_reasoner_confirmed": (
        "共享轮处理器超过边际与错误P-layer，只开放同处理器SCGT增益审计"
    ),
    "innovation2_small_spn_round_shared_topology_not_attributed": (
        "共享轮处理器收益未归因于真实P-layer，停止拓扑贡献声明"
    ),
    "innovation2_small_spn_round_shared_reasoner_not_ready": (
        "共享轮处理器未过冻结门，停止合成GraphGPS/looped家族"
    ),
    "innovation2_small_spn_round_shared_protocol_invalid": (
        "共享可变步数、cell等变、来源、split或metric协议无效"
    ),
    "innovation2_small_spn_cipher_edge_token_readiness_passed": (
        "Cipher Edge-Token Transformer与cell不变性就绪，可运行冻结两seed矩阵"
    ),
    "innovation2_small_spn_cipher_edge_token_confirmed": (
        "显式edge-token模型超过边际与错误P-layer，可准备真实密码迁移readiness"
    ),
    "innovation2_small_spn_cipher_edge_token_not_attributed": (
        "edge-token收益未归因于真实P-layer，关闭合成神经拓扑路线"
    ),
    "innovation2_small_spn_cipher_edge_token_not_ready": (
        "edge-token模型未过冻结门，关闭合成架构搜索并返回标签任务设计"
    ),
    "innovation2_small_spn_cipher_edge_token_protocol_invalid": (
        "edge tokenization、cell不变性、来源、split或metric协议无效"
    ),
    "innovation2_small_spn_topology_labels_identifiable": (
        "P-layer条件标签具有百级组外宽度，可先重建公平benchmark"
    ),
    "innovation2_small_spn_topology_labels_not_identifiable": (
        "P-layer条件标签宽度或类平衡不足，停止当前合成标签路线"
    ),
    "innovation2_small_spn_topology_label_audit_protocol_invalid": (
        "拓扑标签来源、shape、variant顺序或train-only选择协议无效"
    ),
    "innovation2_small_spn_expanded_topology_benchmark_ready": (
        "扩展拓扑族的标签宽度、交互、组外边际与公平控制门通过"
    ),
    "innovation2_small_spn_expanded_topology_benchmark_not_ready": (
        "扩展拓扑族仍不支持公平组外学习，停止随机P-layer机械扩展"
    ),
    "innovation2_small_spn_expanded_topology_protocol_invalid": (
        "扩展拓扑缓存、split、train-only选择或公平控制协议无效"
    ),
    "innovation2_small_spn_expanded_neural_screen_readiness_passed": (
        "扩展拓扑GraphGPS/CETT训练流程就绪，可运行两seed候选筛选"
    ),
    "innovation2_small_spn_expanded_neural_candidate_screened": (
        "至少一个候选稳定超过扩展benchmark的ID边际，可进入公平拓扑归因"
    ),
    "innovation2_small_spn_expanded_neural_screen_not_ready": (
        "GraphGPS/CETT均未稳定超过扩展benchmark的ID边际"
    ),
    "innovation2_small_spn_expanded_neural_screen_not_attributed": (
        "扩展拓扑神经筛选的label-shuffle控制异常，暂不选择候选"
    ),
    "innovation2_small_spn_expanded_neural_screen_protocol_invalid": (
        "扩展拓扑神经筛选的来源、split、等变性或metric协议无效"
    ),
    "innovation2_small_spn_pair_relation_readiness_passed": (
        "有向bit-pair路径推理器的关系、等变性与训练流程就绪"
    ),
    "innovation2_small_spn_pair_relation_candidate_screened": (
        "有向bit-pair路径推理器稳定超过ID边际，可进入公平拓扑归因"
    ),
    "innovation2_small_spn_pair_relation_reasoner_not_ready": (
        "有向bit-pair路径推理器未稳定超过扩展benchmark的ID边际"
    ),
    "innovation2_small_spn_pair_relation_not_attributed": (
        "有向bit-pair路径推理器的label-shuffle控制异常"
    ),
    "innovation2_small_spn_pair_relation_protocol_invalid": (
        "有向bit-pair初始化、triangle update、等变性或训练协议无效"
    ),
    "innovation2_small_spn_pair_relation_topology_confirmed": (
        "有向bit-pair路径推理器稳定领先公平错误P-layer控制"
    ),
    "innovation2_small_spn_pair_relation_topology_not_attributed": (
        "有向bit-pair路径推理器未稳定领先公平错误P-layer控制"
    ),
    "innovation2_small_spn_pair_relation_attribution_protocol_invalid": (
        "有向bit-pair拓扑归因的来源、控制、seed、参数或metric协议无效"
    ),
    "innovation2_small_spn_pair_relation_no_triangle_readiness_passed": (
        "同预算no-triangle局部pair消融流程就绪"
    ),
    "innovation2_small_spn_pair_relation_triangle_attributed": (
        "triangle路径组合稳定领先同预算局部pair更新"
    ),
    "innovation2_small_spn_pair_relation_triangle_not_isolated": (
        "triangle未稳定领先同预算局部pair更新，路径贡献尚未隔离"
    ),
    "innovation2_small_spn_pair_relation_no_triangle_not_attributed": (
        "no-triangle消融的label-shuffle控制异常"
    ),
    "innovation2_small_spn_pair_relation_no_triangle_protocol_invalid": (
        "no-triangle消融的来源、pair局部性、参数、等变性或训练协议无效"
    ),
    "innovation2_small_spn_pair_state_topology_confirmed": (
        "局部pair-state稳定领先公平错误P-layer控制"
    ),
    "innovation2_small_spn_pair_state_topology_not_attributed": (
        "局部pair-state未稳定领先公平错误P-layer控制"
    ),
    "innovation2_small_spn_pair_state_topology_control_protocol_invalid": (
        "局部pair-state拓扑控制的来源、控制、seed、参数、局部性或metric协议无效"
    ),
    "innovation2_real_spn_pair_state_transfer_ready": (
        "真实SPN标签族与64-bit pair-state均通过训练readiness"
    ),
    "innovation2_real_spn_pair_state_label_bank_not_ready": (
        "64-bit pair-state就绪，但现有PRESENT/SKINNY标签库不足以训练"
    ),
    "innovation2_real_spn_pair_state_model_not_ready": (
        "真实SPN标签或64-bit pair-state模型契约尚未就绪"
    ),
    "innovation2_real_spn_pair_state_transfer_protocol_invalid": (
        "真实SPN迁移审计的来源、标签聚合、64-bit模型或metric协议无效"
    ),
    "innovation2_present_universal_balance_atlas_ready": (
        "PRESENT四轮证书/反例atlas与checkerboard反捷径门通过"
    ),
    "innovation2_present_universal_balance_atlas_too_narrow": (
        "PRESENT四轮严格标签或checkerboard宽度不足"
    ),
    "innovation2_present_universal_balance_atlas_shortcut_dominated": (
        "PRESENT四轮matched atlas仍可被一元边际解释"
    ),
    "innovation2_present_universal_balance_atlas_protocol_invalid": (
        "PRESENT四轮atlas的ANF、反例、split、mask或证书协议无效"
    ),
    "innovation2_present_pair_state_topology_attributed": (
        "64-bit pair-state在严格PRESENT四轮标签上通过正确P-layer归因"
    ),
    "innovation2_present_pair_state_candidate_not_ready": (
        "64-bit pair-state未超过严格PRESENT四轮候选开放门"
    ),
    "innovation2_present_pair_state_topology_not_attributed": (
        "64-bit pair-state预测信号未归因到正确P-layer"
    ),
    "innovation2_present_pair_state_attribution_protocol_invalid": (
        "PRESENT四轮pair-state的source、模型、控制、metric或训练协议无效"
    ),
    "innovation2_present_mspn_route_ready": (
        "ANF前缀复杂度归因通过，下一网络选择单项式支撑传播网络"
    ),
    "innovation2_present_query_nbfnet_route_ready": (
        "正确P-layer可达归因通过，下一网络选择query-conditioned NBFNet"
    ),
    "innovation2_present_static_set_route_dominant": (
        "静态active-mask集合统计主导，暂停拓扑网络"
    ),
    "innovation2_present_certificate_attribution_unresolved": (
        "非oracle确定性特征均未解释PRESENT四轮弱神经信号"
    ),
    "innovation2_present_certificate_attribution_protocol_invalid": (
        "PRESENT四轮证书复杂度归因的source、特征、标准化或metric协议无效"
    ),
    "innovation2_present_mspn_readiness_passed": (
        "单项式支撑传播网络实现与两轮训练readiness通过"
    ),
    "innovation2_present_mspn_readiness_failed": (
        "单项式支撑传播网络的等变、有限性、参数或训练控制未通过"
    ),
    "innovation2_present_mspn_topology_attributed": (
        "MSPN在严格PRESENT四轮标签上超过pair-state并通过正确P-layer归因"
    ),
    "innovation2_present_mspn_candidate_not_ready": (
        "MSPN未通过严格PRESENT四轮正式候选门"
    ),
    "innovation2_present_mspn_topology_not_attributed": (
        "MSPN预测信号未归因到正确P-layer transport"
    ),
    "innovation2_present_mspn_attribution_protocol_invalid": (
        "MSPN正式归因的source、模型、控制、metric或训练协议无效"
    ),
    "innovation2_present_identity_sketch_route_ready": (
        "变量身份sketch显著超过degree-only与公平控制，可进入身份传播网络readiness"
    ),
    "innovation2_present_exact_monomial_token_route_ready": (
        "精确support身份超过degree-only，下一候选为稀疏单项式token网络"
    ),
    "innovation2_present_support_identity_not_supported": (
        "变量身份未超过degree-only，关闭identity网络并转向中间degree谱学习问题"
    ),
    "innovation2_present_support_identity_protocol_invalid": (
        "support身份审计的source、传播、投影、碰撞或ridge协议无效"
    ),
    "innovation2_present_degree_spectrum_readiness_passed": (
        "中间degree谱可组外学习且balance未退化，可进入30轮正式归因"
    ),
    "innovation2_present_degree_spectrum_not_learned": (
        "真谱未明显优于target打乱，停止证书传播神经路线"
    ),
    "innovation2_present_degree_spectrum_balance_degenerated": (
        "中间degree谱可学但balance退化，只保留冻结loss-scale审计"
    ),
    "innovation2_present_degree_spectrum_protocol_invalid": (
        "degree谱蒸馏的source、teacher、泄漏、模型、控制或训练协议无效"
    ),
    "innovation2_present_cgpr_readiness_passed": (
        "证书引导pair-state残差实现与两轮训练readiness通过"
    ),
    "innovation2_present_cgpr_readiness_failed": (
        "证书引导残差的source、ridge、零等价、拓扑、参数或训练契约失败"
    ),
    "innovation2_present_cgpr_topology_attributed": (
        "CGPR超过ridge、prefix-only与错误P控制，正式拓扑归因通过"
    ),
    "innovation2_present_cgpr_candidate_not_ready": (
        "CGPR未通过正式候选门，停止E43四轮新网络枚举"
    ),
    "innovation2_present_cgpr_pair_residual_not_attributed": (
        "正确P未超过prefix-only残差，pair-state没有独立贡献"
    ),
    "innovation2_present_cgpr_topology_not_attributed": (
        "正确P未超过错误P残差，P-layer贡献未归因"
    ),
    "innovation2_present_cgpr_attribution_protocol_invalid": (
        "CGPR正式归因的source、ridge、模型、控制或训练协议无效"
    ),
    "innovation2_present_r5_strict_label_bank_not_ready": (
        "五轮P0无可证明正类且P1运行环境未就绪，禁止训练新网络"
    ),
    "innovation2_present_r5_strict_label_p1_subset_required": (
        "五轮P0覆盖不足，进入CLAASP-MP完整superpoly固定子集门"
    ),
    "innovation2_present_r5_strict_label_bank_ready": (
        "五轮严格正负标签与反捷径门通过，先做确定性捷径归因"
    ),
    "innovation2_present_r5_strict_label_provider_protocol_invalid": (
        "五轮标签提供者的PRESENT、证书、反例或产物协议无效"
    ),
    "innovation2_present_r5_open_3sdp_exact_oracle_ready": (
        "一、二轮exact ANF与GF(2)消去校准通过，进入GLPK trail枚举器实现"
    ),
    "innovation2_present_r5_open_3sdp_exact_oracle_invalid": (
        "开放3SDP的exact ANF、bit order、fixture、mask XOR或反例协议无效"
    ),
    "innovation2_present_r5_open_3sdp_cancellation_control_failed": (
        "trail奇偶未复现S-box exact ANF，停止开放provider实现"
    ),
    "innovation2_present_r5_open_3sdp_glpk_runtime_not_ready": (
        "exact oracle正确，但Sage/GLPK运行环境未就绪"
    ),
    "innovation2_present_r5_open_3sdp_glpk_blocking_not_scalable": (
        "GLPK低复杂度计数正确但最重S-box query超时，转transition tensor宽度审计"
    ),
    "innovation2_present_r5_open_3sdp_glpk_sbox_enumerator_ready": (
        "代表S-box query均完整复现exact计数，可进入一轮PRESENT电路fixture"
    ),
    "innovation2_present_r5_open_3sdp_glpk_enumerator_invalid": (
        "GLPK约束、blocking、计数完整性或timeout分类无效"
    ),
    "innovation2_present_r5_transition_tensor_boundary_infeasible": (
        "全key/全offset最终边界需136变量，dense tensor不可行，转三轮稀疏ANF硬cap门"
    ),
    "innovation2_present_r5_transition_tensor_internal_width_audit_ready": (
        "完整superpoly边界门通过，可构造PRESENT内部因子图"
    ),
    "innovation2_present_r5_transition_tensor_boundary_protocol_invalid": (
        "E53-A来源或全key/全offset边界计算无效"
    ),
    "innovation2_present_r3_query_cone_sparse_anf_ready": (
        "三轮冻结query全部通过exact sparse-ANF硬cap，可进入同cap四轮门"
    ),
    "innovation2_present_r3_query_cone_sparse_anf_hard_cap_exceeded": (
        "三轮query越过冻结硬cap，关闭当前full-variable sparse provider"
    ),
    "innovation2_present_r3_query_cone_sparse_anf_label_diversity_insufficient": (
        "三轮冻结query缺少严格正负标签多样性，不后验更换query"
    ),
    "innovation2_present_r3_query_cone_sparse_anf_protocol_invalid": (
        "E53-A重放、位序或语义控制失败，三轮结果不可解释"
    ),
    "innovation2_generalized_relation_label_contract_not_ready": (
        "广义relation正类存在，但真实key schedule、严格负类与互斥拆分未就绪"
    ),
    "innovation2_present_r9_generalized_relation_scalar_witness_infeasible": (
        "最小relation已需2^60明文，关闭直接标量常数与negative-witness路线"
    ),
    "innovation2_atm_native_sat_mechanism_ready_for_r9_probe": (
        "原生SAT机制低轮校准通过，进入单个九轮relation硬cap探针"
    ),
    "innovation2_atm_native_sat_r9_wall_clock_cap_exceeded": (
        "九轮原生SAT单候选超过60秒，保持unknown并关闭该exact witness路线"
    ),
    "innovation2_atm_r2_strict_relation_panel_not_ready": (
        "两轮16条查询全部constant，缺少严格负类，禁止训练RCCA"
    ),
    "innovation2_atm_r2_cone_matched_panel_width_not_ready": (
        "依赖锥内外16条单坐标查询全部constant，关闭RCCA并转多坐标GF(2)消去关系"
    ),
    "innovation2_atm_r2_singleton_relation_shortcut_dominated": (
        "单坐标严格标签可被依赖锥捷径解释，转多坐标GF(2)消去关系"
    ),
    "innovation2_atm_r2_cone_matched_panel_protocol_invalid": (
        "两轮依赖锥配对、SAT模型或标量复核无效"
    ),
    "innovation2_atm_r2_multicoordinate_support_runtime_not_ready": (
        "完整key-polynomial支撑60秒仅完成8/240，关闭该exact支撑路线"
    ),
    "innovation2_atm_r2_multicoordinate_support_width_not_ready": (
        "多坐标支撑缺少足够严格消去正负关系，禁止训练RCCA"
    ),
    "innovation2_atm_r2_multicoordinate_support_phase_b_ready": (
        "多坐标严格消去关系通过Phase A，可构建256/class标签atlas"
    ),
    "innovation2_atm_r2_multicoordinate_support_protocol_invalid": (
        "E60来源、坐标池或独立轮密钥支撑协议无效"
    ),
    "innovation2_small_spn_multicoordinate_relation_training_ready": (
        "全256主密钥多坐标relation标签与反捷径门通过，可训练DeepSets与RCCA"
    ),
    "innovation2_small_spn_multicoordinate_relation_width_not_ready": (
        "小型SPN多坐标relation缺少严格标签或拓扑宽度，停止RCCA"
    ),
    "innovation2_small_spn_multicoordinate_relation_shortcut_dominated": (
        "小型SPN多坐标relation被拓扑无关边际解释，停止RCCA"
    ),
    "innovation2_small_spn_multicoordinate_relation_protocol_invalid": (
        "E37缓存、train-only选择或全密钥证书无效"
    ),
    "innovation2_small_spn_rcca_readiness_passed": (
        "DeepSets/RCCA不变量与四行训练流程通过，进入正式双seed矩阵"
    ),
    "innovation2_small_spn_rcca_phase_b_ready": (
        "RCCA正式双seed超过DeepSets与边际，可进入wrong-P拓扑归因"
    ),
    "innovation2_small_spn_rcca_not_ready": (
        "RCCA未稳定超过DeepSets与强边际，关闭该架构"
    ),
    "innovation2_small_spn_rcca_protocol_invalid": (
        "RCCA不变量、来源、参数预算或训练协议无效"
    ),
    "innovation2_small_spn_relation_nontrivial_width_not_ready": (
        "非平凡消去正类极窄，停止多坐标神经网络路线"
    ),
    "innovation2_small_spn_relation_singleton_shortcut_dominated": (
        "singleton平衡状态几乎完全解释relation标签，停止多坐标网络"
    ),
    "innovation2_small_spn_relation_nontrivial_residual_ready": (
        "非平凡GF(2)消去残差宽且未被singleton状态解释，可审计pair-path算子"
    ),
    "innovation2_small_spn_relation_decomposition_protocol_invalid": (
        "E62标签重算、四类分解或negative witness无效"
    ),
    "innovation2_present_unit_balance_profile_topology_ready": (
        "单位输出平衡谱宽且正确拓扑信号过门，可测试逐节点profile operator"
    ),
    "innovation2_present_unit_balance_profile_prefix_ready": (
        "单位输出平衡谱宽且ANF前缀信号过门，可测试prefix引导profile operator"
    ),
    "innovation2_present_unit_balance_profile_signal_not_ready": (
        "单位输出平衡谱缺少非平凡组外信号，停止该神经结构"
    ),
    "innovation2_present_unit_balance_profile_shortcut_dominated": (
        "单位输出平衡谱仍被行列边际解释，禁止训练"
    ),
    "innovation2_present_unit_balance_profile_too_narrow": (
        "单位输出平衡谱宽度或覆盖不足，停止该路线"
    ),
    "innovation2_present_unit_balance_profile_protocol_invalid": (
        "E43重放、谱重排、特征或split协议无效"
    ),
    "innovation2_present_profile_operator_readiness_passed": (
        "prefix引导逐节点平衡谱算子实现与短训练就绪，可进入正式seed0"
    ),
    "innovation2_present_profile_operator_optimization_not_ready": (
        "安全前缀的两轮学习未就绪，停止正式profile训练"
    ),
    "innovation2_present_profile_operator_protocol_invalid": (
        "profile source、等变性、masked loss、参数或训练协议无效"
    ),
    "innovation2_present_profile_operator_neural_gain_attributed": (
        "正确P平衡谱算子超过独立node、错误P和ANF ridge，允许seed1"
    ),
    "innovation2_present_profile_operator_no_ridge_gain": (
        "正确拓扑贡献成立但未超过ANF ridge，不运行seed1"
    ),
    "innovation2_present_profile_operator_relation_not_attributed": (
        "候选未稳定领先独立node或错误P，停止该结构"
    ),
    "innovation2_present_profile_operator_candidate_not_ready": (
        "正式绝对AUC或过拟合门失败，停止该结构"
    ),
    "innovation2_present_profile_operator_attribution_protocol_invalid": (
        "E67 source、contract、metric或正式训练协议无效"
    ),
    "innovation2_present_profile_operator_two_seed_confirmed": (
        "正确P平衡谱算子双seed均超过独立node、错误P和ANF ridge"
    ),
    "innovation2_present_profile_operator_seed_not_replicated": (
        "seed1或联合门未复现seed0结构增益，停止该算子"
    ),
    "innovation2_present_profile_operator_replication_protocol_invalid": (
        "seed0 source、profile、contract或seed1协议无效"
    ),
    "innovation2_present_multibit_mask_query_ready": (
        "多bit linear-mask非平凡标签和信号过门，可测试轻量query decoder"
    ),
    "innovation2_present_multibit_profile_componentwise_dominated": (
        "多bit正类由unit平衡状态组合解释，停止mask-query decoder"
    ),
    "innovation2_present_multibit_profile_marginal_dominated": (
        "多bit标签仍被边际捷径解释，禁止训练decoder"
    ),
    "innovation2_present_multibit_profile_too_narrow": (
        "多bit mask family宽度不足，停止扩展"
    ),
    "innovation2_present_multibit_profile_signal_not_ready": (
        "多bit标签缺少非平凡确定性信号，停止decoder"
    ),
    "innovation2_present_multibit_profile_protocol_invalid": (
        "E43重放、matching、分解或特征协议无效"
    ),
    "innovation2_present_active_dimension_zero_shot_confirmed": (
        "unit-profile算子双seed零样本跨4/12-bit活动维度通过"
    ),
    "innovation2_present_active_dimension_zero_shot_not_confirmed": (
        "unit-profile算子零样本跨活动维度未确认，保留8-bit域内结果"
    ),
    "innovation2_present_active_dimension_transfer_labels_not_ready": (
        "4/12-bit严格unit标签宽度不足，不解释迁移AUC"
    ),
    "innovation2_present_active_dimension_transfer_protocol_invalid": (
        "source、前缀兼容或checkpoint迁移协议无效"
    ),
    "innovation2_present_round_recurrent_readiness_passed": (
        "显式轮序平衡谱算子readiness通过，可进入30轮seed0归因"
    ),
    "innovation2_present_round_recurrent_readiness_not_passed": (
        "显式轮序或正确P增益未过门，停止RR-PGPO"
    ),
    "innovation2_present_round_recurrent_protocol_invalid": (
        "显式轮序算子的source、等变性、参数公平或训练协议无效"
    ),
    "innovation2_present_early_round_skip_candidate_ready": (
        "r1主导经ridge与双seed切片中和确认，可测试early-round skip"
    ),
    "innovation2_present_round_direction_not_confirmed": (
        "轮切片证据不支持新轮序结构，停止轮递归分支"
    ),
    "innovation2_present_round_slice_protocol_invalid": (
        "E65/E67/E68重放或轮切片中和协议无效"
    ),
    "innovation2_present_r3_only_profile_readiness_passed": (
        "r3-only平衡谱算子readiness通过，可进入30轮seed0"
    ),
    "innovation2_present_r3_only_profile_readiness_not_passed": (
        "r3-only质量或拓扑增益未过门，保留完整39维E68"
    ),
    "innovation2_present_r3_only_profile_protocol_invalid": (
        "r3切片、source、参数公平或训练协议无效"
    ),
    "innovation2_present_r3_only_neural_gain_attributed": (
        "r3-only正确P算子保持E67质量与拓扑增益，可进入seed1"
    ),
    "innovation2_present_r3_only_quality_not_retained": (
        "r3-only未保持完整39维E67质量，保留E68"
    ),
    "innovation2_present_r3_only_topology_not_attributed": (
        "r3-only未通过正确P拓扑归因，不进入seed1"
    ),
    "innovation2_present_r3_only_attribution_protocol_invalid": (
        "r3-only readiness、source、contract或30轮协议无效"
    ),
    "innovation2_present_r3_only_two_seed_confirmed": (
        "r3-only参数压缩算子双seed保持质量与正确P拓扑增益"
    ),
    "innovation2_present_r3_only_seed_not_replicated": (
        "r3-only seed1未保持质量，保留完整39维E68"
    ),
    "innovation2_present_r3_only_replication_protocol_invalid": (
        "r3-only seed0/source/contract或seed1协议无效"
    ),
    "innovation2_gift64_unit_balance_profile_ready": (
        "GIFT-64四轮严格unit谱标签与反捷径门通过，可测试r3-only算子"
    ),
    "innovation2_gift64_unit_balance_profile_not_ready": (
        "GIFT-64四轮严格unit谱宽度或反捷径门不足，禁止神经训练"
    ),
    "innovation2_gift64_unit_balance_profile_protocol_invalid": (
        "GIFT-64轮函数、ANF、反例或split协议无效"
    ),
    "innovation2_gift64_unit_balance_profile_expansion_ready": (
        "GIFT-64四轮192结构严格unit谱容量门通过，可测试r3-only算子"
    ),
    "innovation2_gift64_unit_balance_profile_expansion_not_ready": (
        "GIFT-64四轮192结构仍无足够matching容量，关闭当前迁移路线"
    ),
    "innovation2_gift64_unit_balance_profile_expansion_protocol_invalid": (
        "GIFT-64 E74锚点重放或192结构扩展协议无效"
    ),
    "innovation2_gift64_r3_only_profile_readiness_passed": (
        "GIFT-64真实P-layer的r3-only两轮门通过，可进入30轮seed0"
    ),
    "innovation2_gift64_r3_only_prefix_not_sufficient": (
        "GIFT-64第3轮前缀信息不足，只允许审计完整39维算子"
    ),
    "innovation2_gift64_r3_only_profile_readiness_not_passed": (
        "GIFT-64真实P-layer未过两轮归因门，停止r3-only正式训练"
    ),
    "innovation2_gift64_r3_only_profile_protocol_invalid": (
        "GIFT-64 E75来源、拓扑、参数公平或训练协议无效"
    ),
    "innovation2_gift64_topology_interaction_gate_repaired": (
        "公平拓扑ridge与同权重反事实确认真实GIFT P交互，可另立正式计划"
    ),
    "innovation2_gift64_topology_interaction_not_confirmed": (
        "公平确定性或同权重拓扑归因未确认，关闭GIFT r3-only"
    ),
    "innovation2_gift64_topology_interaction_protocol_invalid": (
        "GIFT-64 E75/E76重放、ridge、拓扑变体或checkpoint协议无效"
    ),
    "innovation2_gift64_r3_only_neural_gain_attributed": (
        "GIFT-64真实P的30轮seed0质量与拓扑增益通过，可运行seed1"
    ),
    "innovation2_gift64_r3_only_quality_not_confirmed": (
        "GIFT-64 r3-only绝对质量、过拟合或公平ridge门未过"
    ),
    "innovation2_gift64_r3_only_topology_not_attributed": (
        "GIFT-64 r3-only未稳定领先独立node或错误P，关闭正式路线"
    ),
    "innovation2_gift64_r3_only_attribution_protocol_invalid": (
        "GIFT-64 E75/E77来源、参数公平或30轮seed0协议无效"
    ),
    "innovation2_gift64_r3_only_two_seed_confirmed": (
        "GIFT-64真实P的30轮质量与拓扑增益双seed确认"
    ),
    "innovation2_gift64_r3_only_seed_not_replicated": (
        "GIFT-64 r3-only seed1未独立通过，保留seed0与确定性证据"
    ),
    "innovation2_gift64_r3_only_replication_protocol_invalid": (
        "GIFT-64 E75/E78来源、参数公平或30轮seed1协议无效"
    ),
    "innovation2_cross_spn_r3_profile_method_confirmed_skinny_labels_not_ready": (
        "PRESENT/GIFT双密码方法证据通过，SKINNY严格标签尚未就绪"
    ),
    "innovation2_cross_spn_r3_profile_method_confirmed_third_spn_ready": (
        "PRESENT/GIFT双密码方法证据通过，第三SPN标签可进入训练门"
    ),
    "innovation2_cross_spn_r3_profile_method_not_confirmed": (
        "PRESENT或GIFT逐seed拓扑归因未同时通过"
    ),
    "innovation2_cross_spn_method_synthesis_protocol_invalid": (
        "E73/E79或SKINNY冻结来源、hash与方法契约无效"
    ),
    "innovation2_skinny64_unit_balance_profile_ready": (
        "SKINNY-64四轮严格unit谱标签门通过，可做本地三行readiness"
    ),
    "innovation2_skinny64_unit_balance_profile_not_ready": (
        "SKINNY-64四轮严格unit谱宽度或反捷径门不足，禁止训练"
    ),
    "innovation2_skinny64_unit_balance_profile_protocol_invalid": (
        "SKINNY坐标、向量化、support、反例或split协议无效"
    ),
    "innovation2_skinny64_r5_unit_balance_profile_transition_ready": (
        "SKINNY-64五轮严格unit谱进入可训练过渡区，可做r4-only readiness"
    ),
    "innovation2_skinny64_r5_unit_balance_profile_transition_not_ready": (
        "SKINNY-64五轮严格unit谱仍不够宽，停止当前跨轮扫描"
    ),
    "innovation2_skinny64_r5_unit_balance_profile_transition_protocol_invalid": (
        "E81锚点、五轮向量化、support、反例或split协议无效"
    ),
    "innovation2_skinny64_sparse_profile_readiness_passed": (
        "SKINNY真实稀疏线性图的ridge与两轮神经门通过，可预注册30轮seed0"
    ),
    "innovation2_skinny64_sparse_profile_topology_not_attributed": (
        "公平ridge未归因SKINNY真实线性图，停止当前稀疏算子"
    ),
    "innovation2_skinny64_sparse_profile_readiness_not_passed": (
        "SKINNY稀疏算子两轮未同时超过控制，停止正式训练"
    ),
    "innovation2_skinny64_sparse_profile_readiness_protocol_invalid": (
        "E82来源、稀疏图、参数公平、等变或训练协议无效"
    ),
    "innovation2_skinny64_true_ridge_residual_readiness_passed": (
        "SKINNY真实图残差超过冻结ridge与两类控制，可预注册30轮seed0"
    ),
    "innovation2_skinny64_true_ridge_residual_not_ready": (
        "SKINNY神经残差未超过强ridge与控制，收束当前神经搜索"
    ),
    "innovation2_skinny64_true_ridge_residual_protocol_invalid": (
        "E82/E83来源、ridge、零残差、冻结buffer、图或训练协议无效"
    ),
    "innovation2_shared_profile_operator_readiness_passed": (
        "一套共享算子在PRESENT/GIFT均超过控制，可进入30轮seed0"
    ),
    "innovation2_shared_profile_operator_readiness_not_passed": (
        "共享参数未同时保留双密码质量与拓扑增益，保留独立模型"
    ),
    "innovation2_shared_profile_operator_protocol_invalid": (
        "双密码来源、运行时拓扑、共享参数、公平预算或等变协议无效"
    ),
    "innovation2_shared_profile_operator_seed0_attributed": (
        "共享算子在PRESENT/GIFT均保留30轮质量与拓扑增益，可运行seed1"
    ),
    "innovation2_shared_profile_operator_quality_not_retained": (
        "共享算子未保留至少一个密码的独立模型质量，关闭共享分支"
    ),
    "innovation2_shared_profile_operator_topology_not_attributed": (
        "共享算子未在至少一个密码超过独立或错误拓扑，关闭共享分支"
    ),
    "innovation2_shared_profile_operator_attribution_protocol_invalid": (
        "E85/锚点来源、30轮schedule、动态拓扑或产物协议无效"
    ),
    "innovation2_rectangle80_unit_profile_ready": (
        "RECTANGLE四轮严格unit谱标签门通过，下一步扩到192结构"
    ),
    "innovation2_rectangle80_unit_profile_raw_labels_not_ready": (
        "RECTANGLE四轮原始标签未进入正负过渡区，下一步只改为五轮"
    ),
    "innovation2_rectangle80_unit_profile_matching_not_ready": (
        "RECTANGLE原始标签足够但matching容量不足，下一步只扩结构"
    ),
    "innovation2_rectangle80_unit_profile_protocol_invalid": (
        "RECTANGLE最终版规范、行序、向量化、support或反例协议无效"
    ),
    "innovation2_rectangle80_unit_profile_expansion_ready": (
        "RECTANGLE四轮192结构严格标签容量门通过，可测试r3-only算子"
    ),
    "innovation2_rectangle80_unit_profile_expansion_not_ready": (
        "RECTANGLE四轮192结构仍未保持宽度或反捷径门，关闭当前神经路线"
    ),
    "innovation2_rectangle80_unit_profile_expansion_protocol_invalid": (
        "E87锚点重放或RECTANGLE严格标签协议无效"
    ),
    "innovation2_rectangle80_nested_cube_monotonic_labels_ready": (
        "RECTANGLE四轮7/8/9-bit嵌套cube严格标签门通过，可做无训练单调机制审计"
    ),
    "innovation2_rectangle80_nested_cube_monotonic_labels_not_ready": (
        "嵌套cube标签宽度或匹配容量不足，关闭当前单调神经路线"
    ),
    "innovation2_rectangle80_nested_cube_monotonic_protocol_invalid": (
        "E88重放、嵌套关系、单调closure或反例语义无效"
    ),
    "innovation2_rectangle80_nested_cube_relation_mechanism_ready": (
        "真实cube嵌套关系通过无训练归因门，可测试两轮单调立方格算子"
    ),
    "innovation2_rectangle80_nested_cube_relation_not_attributed": (
        "真实cube嵌套未稳定超过错误关系控制，关闭当前单调神经路线"
    ),
    "innovation2_rectangle80_nested_cube_relation_protocol_invalid": (
        "E94来源、关系映射、容量、拆分或单调投影协议无效"
    ),
    "innovation2_architecture_portfolio_converged_no_new_training_budget": (
        "当前无合格新架构训练候选，停止枚举并转严格provider研究或论文收束"
    ),
    "innovation2_architecture_portfolio_new_candidate_ready": (
        "出现通过标签与机制门的新候选，只允许预注册最高优先级路线"
    ),
    "innovation2_architecture_portfolio_protocol_invalid": (
        "冻结来源或架构候选分类不一致，先修复组合证据"
    ),
    "innovation2_present_cancellation_provider_not_feasible_under_frozen_caps": (
        "冻结cap内无合格非平凡相消provider，停止当前provider研究并转论文收束"
    ),
    "innovation2_present_cancellation_provider_feasible": (
        "严格非平凡相消provider通过，只允许扩标签并先做确定性机制门"
    ),
    "innovation2_present_cancellation_provider_protocol_invalid": (
        "E52-E69来源或12-query面板漂移，先修复provider审计协议"
    ),
    "innovation2_present_r9_pu_ranking_benchmark_not_ready": (
        "九轮ATM关系分组宽度与秩契约未通过，不训练E99并继续关闭远程扩展"
    ),
    "innovation2_present_r9_pu_ranking_ready_for_local_neural_gate": (
        "九轮正例-未标注排序基准通过，只开放E99本地神经排序门"
    ),
    "innovation2_present_r9_pu_ranking_protocol_invalid": (
        "九轮ATM来源或候选池协议无效，只修复审计而不解释科学结果"
    ),
    "innovation2_present_r9_atm_public_merge_count_not_rank": (
        "公开ATM的470是去重计数，正确合并秩为468；来源差异已解释但E99仍关闭"
    ),
    "innovation2_present_r9_atm_basis_merge_audit_protocol_invalid": (
        "ATM冻结来源、依赖恢复或split覆盖审计无效，只修复协议"
    ),
    "innovation2_present_r9_atm_470_468_mismatch_not_explained": (
        "470与468差异仍未解释，保持九轮神经训练关闭"
    ),
    "innovation2_present_r9_atm_support_component_pu_ready": (
        "九轮ATM支撑组件互斥PU数据门通过，只开放E99本地神经排序"
    ),
    "innovation2_present_r9_atm_support_component_pu_not_ready": (
        "九轮ATM支撑组件PU数据门未通过，停止当前公开语料神经路线"
    ),
    "innovation2_present_r9_atm_support_component_pu_protocol_invalid": (
        "九轮ATM支撑组件或未标注候选协议无效，只修复协议"
    ),
    "innovation2_present_r9_atm_support_orbit_pu_ready": (
        "九轮ATM支撑与旋转轨道双互斥PU门通过，可恢复E99本地神经排序"
    ),
    "innovation2_present_r9_atm_support_orbit_pu_not_ready": (
        "九轮ATM旋转轨道泄漏修复门未通过，停止当前公开语料神经路线"
    ),
    "innovation2_present_r9_atm_support_orbit_pu_protocol_invalid": (
        "九轮ATM旋转轨道或候选协议无效，只修复协议"
    ),
    "innovation2_present_r9_pu_topology_neural_signal_confirmed": (
        "九轮PU排序的PRESENT拓扑神经增益双seed确认，只开放远程方案设计"
    ),
    "innovation2_present_r9_pu_generic_neural_signal_only": (
        "九轮PU排序只有通用神经信号，拓扑归因未过并保持远程关闭"
    ),
    "innovation2_present_r9_pu_public_corpus_neural_route_stopped": (
        "九轮公开PU语料神经排序未稳定超过控制，停止当前路线"
    ),
    "innovation2_present_r9_pu_neural_ranking_protocol_invalid": (
        "九轮PU神经排序来源、折、候选或指标协议无效"
    ),
    "innovation2_present_r9_e99_coordinate_checkpoints_frozen": (
        "E99坐标集合模型12个折权重全部复现并冻结，等待E104来源留出评估"
    ),
    "innovation2_present_r9_external_relation_source_unavailable": (
        "暂无满足同语义、零重合和至少32个新增维度的外部来源，停止E99坐标迁移"
    ),
    "innovation2_present_r9_e99_checkpoint_replay_invalid": (
        "E99坐标模型重放未完全复现，禁止读取E104关系"
    ),
    "innovation2_output_parity_prediction_readiness_passed": (
        "固定密钥输出预测通路通过；连续nibble parity随机附近，只开放一轮mask几何校准"
    ),
    "innovation2_output_parity_mask_geometry_supported": (
        "S-box/P层对齐输出parity信号通过，只开放独立固定密钥复验"
    ),
    "innovation2_output_parity_mask_geometry_not_calibrated": (
        "对齐输出parity未过归因门，停止扩规模并转输出预测论文协议审计"
    ),
    "innovation2_output_parity_mask_geometry_protocol_invalid": (
        "固定密钥输出预测或配对mask协议无效，只修复协议"
    ),
    "innovation2_output_parity_mask_geometry_two_key_confirmed": (
        "结构对齐密文输出parity双固定密钥确认，只开放PRESENT二轮同预算门"
    ),
    "innovation2_output_parity_mask_geometry_two_key_not_confirmed": (
        "结构对齐输出parity未获独立密钥确认，停止扩轮并转论文协议审计"
    ),
    "innovation2_output_parity_two_key_protocol_invalid": (
        "双密钥anchor、独立性或输出预测协议无效，只修复协议"
    ),
    "innovation2_output_parity_present_r2_two_key_supported": (
        "PRESENT二轮结构对齐密文输出parity双密钥通过，只开放三轮同预算门"
    ),
    "innovation2_output_parity_present_r2_two_key_not_supported": (
        "PRESENT二轮结构对齐输出parity未通过，停止扩轮并做本地表示重设计"
    ),
    "innovation2_output_parity_present_r3_two_key_supported": (
        "PRESENT三轮结构对齐密文输出parity双密钥通过，只开放四轮同预算门"
    ),
    "innovation2_output_parity_present_r3_two_key_not_supported": (
        "PRESENT三轮结构对齐输出parity未通过，停止机械扩轮并做本地表示重设计"
    ),
    "innovation2_output_parity_present_r3_single_key_not_supported": (
        "PRESENT三轮seed0对齐输出parity未过门，仍执行seed1后停止机械扩轮"
    ),
    "innovation2_output_parity_present_r3_spn_local_attributed": (
        "PRESENT三轮SPN局部网络真实P层增益过门，只开放独立固定密钥复验"
    ),
    "innovation2_output_parity_present_r3_spn_local_generic_gain_only": (
        "PRESENT三轮只有通用局部表示收益，先审计精确bit-role路由"
    ),
    "innovation2_output_parity_present_r3_spn_local_not_ready": (
        "PRESENT三轮nibble邻接网络未恢复信号，转精确bit-level SPN路由"
    ),
    "innovation2_output_parity_present_r3_spn_local_protocol_invalid": (
        "PRESENT三轮输出、token顺序、拓扑控制或训练协议无效"
    ),
    "innovation2_output_parity_present_r3_bit_role_attributed": (
        "PRESENT三轮精确bit-role真实P层增益过门，只开放独立固定密钥复验"
    ),
    "innovation2_output_parity_present_r3_bit_role_generic_gain_only": (
        "PRESENT三轮只有通用bit-role收益，先复核错误P层拓扑控制"
    ),
    "innovation2_output_parity_present_r3_bit_role_not_ready": (
        "PRESENT三轮精确bit-role网络仍未过门，转确定性依赖锥难度审计"
    ),
    "innovation2_output_parity_present_r3_bit_role_protocol_invalid": (
        "PRESENT三轮bit路由、拓扑控制、输出或训练协议无效"
    ),
    "innovation2_output_parity_exact_anf_difficulty_transition_confirmed": (
        "r1--r3真实输出parity精确ANF难度跃迁确认，只开放三轮嵌套数据斜率"
    ),
    "innovation2_output_parity_exact_anf_difficulty_transition_not_confirmed": (
        "精确ANF未支持数据稀疏机制，停止当前mask路线扩样本与扩轮"
    ),
    "innovation2_output_parity_exact_anf_difficulty_protocol_invalid": (
        "精确ANF、标量重放或冻结神经来源协议无效"
    ),
    "innovation2_output_parity_exact_anf_difficulty_hard_cap_exceeded": (
        "部分真实输出parity精确ANF超过冻结硬上限，不提高上限或扩训练"
    ),
    "innovation2_present_r9_identity_true_p_residual_attributed": (
        "九轮坐标身份主干上的真实P残差双seed归因通过，只开放独立来源确认设计"
    ),
    "innovation2_present_r9_identity_residual_capacity_only": (
        "九轮身份残差只有容量收益，真实P未超过错误P控制"
    ),
    "innovation2_present_r9_coordinate_identity_anchor_remains_best": (
        "九轮坐标身份锚点仍最佳，停止当前PRESENT拓扑分支"
    ),
    "innovation2_present_r9_identity_topology_residual_protocol_invalid": (
        "九轮身份拓扑残差的来源、配对模型、fold或指标协议无效"
    ),
    "innovation2_present_high_round_source_generation_ready": (
        "PRESENT高轮新关系生成链路就绪，只允许预注册R9缺失split"
    ),
    "innovation2_present_high_round_resumable_runner_required": (
        "PRESENT高轮源码有生成调用但缺少可靠恢复，先实现runner而不启动长搜索"
    ),
    "innovation2_present_high_round_source_generation_audit_protocol_invalid": (
        "PRESENT高轮来源、notebook、stats或E100重放协议无效"
    ),
    "innovation2_present_atm_resumable_runner_fixture_passed": (
        "ATM逐候选恢复fixture通过，只开放真实ATM低成本兼容性门"
    ),
    "innovation2_present_atm_resumable_runner_fixture_insufficient": (
        "ATM恢复fixture路径覆盖不足，只扩充fixture而不启动真实搜索"
    ),
    "innovation2_present_atm_resumable_runner_protocol_invalid": (
        "ATM恢复runner的来源、等价性、完整性或参数协议无效"
    ),
    "innovation2_present_sbox4_real_atm_compatibility_passed": (
        "真实ATM小切片兼容门通过，只开放R9缺失split的受控计划"
    ),
    "innovation2_present_sbox4_real_atm_environment_incompatible": (
        "真实ATM的bitset、依赖、QMC或Manager运行环境不兼容"
    ),
    "innovation2_present_sbox4_real_atm_runner_mismatch": (
        "真实ATM官方搜索与可恢复runner的空间或恢复协议不一致"
    ),
    "innovation2_present_sbox4_real_atm_resource_cap_hit": (
        "真实ATM小切片超过资源上限，R9生成继续关闭"
    ),
    "innovation2_present_sbox4_real_atm_source_protocol_invalid": (
        "E102或真实ATM冻结来源重放无效"
    ),
    "innovation2_rectangle80_r3_only_profile_readiness_passed": (
        "RECTANGLE真实P层两轮神经门与公平基线门通过，可进入30轮seed0"
    ),
    "innovation2_rectangle80_r3_only_topology_baseline_not_ready": (
        "公平确定性基线未确认RECTANGLE真实P层信息，关闭当前神经路线"
    ),
    "innovation2_rectangle80_r3_only_profile_readiness_not_passed": (
        "RECTANGLE真实P层未同时超过神经控制和公平ridge，停止正式训练"
    ),
    "innovation2_rectangle80_r3_only_profile_protocol_invalid": (
        "E88来源、RECTANGLE cell-major拓扑、参数公平或训练协议无效"
    ),
    "innovation2_rectangle80_r3_only_neural_gain_attributed": (
        "RECTANGLE真实P层30轮seed0质量与拓扑增益通过，可运行seed1"
    ),
    "innovation2_rectangle80_r3_only_quality_not_confirmed": (
        "RECTANGLE真实P层绝对质量、过拟合或公平ridge门未过"
    ),
    "innovation2_rectangle80_r3_only_topology_not_attributed": (
        "RECTANGLE真实P层未稳定领先同参数拓扑控制"
    ),
    "innovation2_rectangle80_r3_only_attribution_protocol_invalid": (
        "E88/E89来源、cell-major拓扑、参数公平或30轮协议无效"
    ),
    "innovation2_rectangle80_row_typed_representation_ready": (
        "RECTANGLE row类型表示通过机制门，可设计容量配平的新算子"
    ),
    "innovation2_rectangle80_row_typed_representation_not_ready": (
        "RECTANGLE row类型未同时超过锚点和错误控制，不训练新网络"
    ),
    "innovation2_rectangle80_row_typed_representation_protocol_invalid": (
        "E88/E90来源或RECTANGLE row类型表示协议无效"
    ),
    "innovation2_rectangle80_row_typed_shift_operator_readiness_passed": (
        "RECTANGLE参数零增量row类型算子通过两轮门，可进入30轮seed0"
    ),
    "innovation2_rectangle80_row_typed_shift_operator_not_ready": (
        "RECTANGLE row类型算子未同时超过无类型和错误控制，关闭该路线"
    ),
    "innovation2_rectangle80_row_typed_shift_operator_protocol_invalid": (
        "E88/E90/E91来源、row通道置换、参数公平或训练协议无效"
    ),
    "innovation2_architecture_boundary_confirmed_third_spn_neural_not_confirmed": (
        "PRESENT/GIFT独立算子保持正式第一，第三SPN神经尚未确认"
    ),
    "innovation2_architecture_boundary_synthesis_protocol_invalid": (
        "创新2冻结来源或神经结构证据分级不一致"
    ),
    "innovation2_generalized_relation_extension_ready": (
        "广义relation扩展标签就绪，但必须与linear-mask balance任务分开"
    ),
    "innovation2_generalized_relation_original_target_ready": (
        "广义relation可映射原PRESENT-80平衡目标，可先构建标签atlas"
    ),
    "innovation2_generalized_relation_contract_protocol_invalid": (
        "ATM来源版本、安全解析或源码契约无效"
    ),
    "innovation2_high_round_integral_two_seed_bridge_confirmed": (
        "两颗 seed 均确认 PRESENT-80 8轮神经信号，准备论文参考规模近似实验"
    ),
    "innovation2_high_round_integral_two_seed_bridge_not_confirmed": (
        "双 seed 信号未共同过门，停止机械扩规模并审计 seed 敏感性"
    ),
    "innovation2_high_round_integral_two_seed_bridge_invalid": (
        "双 seed source、协议或控制证据无效，修复后重新裁决"
    ),
    "innovation2_high_round_integral_paper_reference_candidate_advantage": (
        "PRESENT-80 8轮论文参考规模近似通过，候选优于同预算锚点与强控制"
    ),
    "innovation2_high_round_integral_paper_reference_round_reach_only": (
        "PRESENT-80 8轮论文参考规模信号成立，但未确认候选架构优势"
    ),
    "innovation2_high_round_integral_paper_reference_not_confirmed": (
        "论文参考规模8轮信号未确认，停止机械放大并审计近似参数"
    ),
    "innovation2_high_round_integral_paper_reference_invalid_control": (
        "论文参考规模 source、缓存或控制无效，修复后重新裁决"
    ),
    "innovation2_high_round_integral_paper_reference_plan_mismatch": (
        "论文参考规模运行偏离冻结协议，不纳入结果比较"
    ),
    "innovation2_high_round_integral_two_seed_paper_reference_candidate_advantage_confirmed": (
        "双 seed 论文参考规模候选优势确认，可按限定范围写入论文"
    ),
    "innovation2_high_round_integral_two_seed_paper_reference_round_reach_confirmed": (
        "双 seed 达到 PRESENT-80 8轮，但未确认候选架构优势"
    ),
    "innovation2_high_round_integral_two_seed_paper_reference_seed_variance_hold": (
        "论文参考规模存在 seed 方差，停止机械追加并审计冻结假设"
    ),
    "innovation2_high_round_integral_two_seed_paper_reference_invalid": (
        "双 seed 论文参考规模证据链无效，修复后重新裁决"
    ),
    "e4_typed_topology_attribution_robust_scratch_efficiency_conditional": (
        "类型拓扑归因稳健，短期 scratch 优势仅条件成立"
    ),
    "e4_r5_source_seed_signal_unstable": "独立 source-seed 稳健性未确认，停止正式扩展",
    "e4_r5_target_adaptation_signal_unstable": "单 seed 目标适配信号不稳定",
    "e4_r5_source_seed_gate_pass": "独立 PRESENT source checkpoint 门控通过",
    "e4_r4_two_seed_target_adaptation_efficiency_confirmed": "双 seed 目标适配效率已确认",
    "e4_r4_two_seed_target_adaptation_signal_unstable": "双 seed 目标适配信号不稳定",
    "e4_r4_two_seed_target_adaptation_rejected": "双 seed 目标适配假设未通过",
    "e4_r4_target_adaptation_efficiency_confirmed": "目标适配效率已确认",
    "e4_r4_target_adaptation_signal_unstable": "目标适配信号不稳定",
    "e4_r4_target_adaptation_rejected": "目标适配假设未通过",
    "e4_r3_two_seed_medium_signal_confirmed": "双 seed 中等规模迁移信号已确认",
    "e4_r3_two_seed_medium_signal_unstable": "双 seed 中等规模迁移信号不稳定",
    "e4_r3_seed_signal_preserved": "中等规模迁移信号保持",
    "e4_r3_seed_margin_miss": "中等规模迁移差值未过门槛",
    "two_seed_transfer_signal_confirmed": "双 seed 迁移信号已确认",
    "promote_e4_r2": "进入 E4-R2 检查点迁移实验",
    "promote_e4_transfer_joint_gate": "进入双 seed 联合门控",
    "promote_e4_transfer_seed1": "进入独立 seed1 复验",
    "implementation_ready": "实现就绪",
}

STATUS_LABELS = {
    "pass": "通过",
    "fail": "失败",
    "hold": "暂缓",
    "running": "运行中",
    "results_available": "结果已生成",
    "fallback_retrieved": "原始回收 / 尚未验证",
    "unknown": "状态未知",
}


def build_result_index(
    outputs_root: Path,
    *,
    roots: tuple[str, ...] = DEFAULT_RESULT_ROOTS,
    limit: int = DEFAULT_INDEX_LIMIT,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if retention_days < 0:
        raise ValueError("retention_days must be at least 0")
    entries: list[dict[str, Any]] = []
    for scope in roots:
        scope_root = outputs_root / scope
        if not scope_root.is_dir():
            continue
        for run_root in sorted(path for path in scope_root.iterdir() if path.is_dir()):
            if (run_root / "index_excluded.marker").is_file():
                continue
            entry = _index_run(outputs_root, scope, run_root)
            if entry is not None:
                entries.append(entry)
    deduplicated: dict[str, dict[str, Any]] = {}
    for entry in entries:
        run_id = str(entry["run_id"])
        current = deduplicated.get(run_id)
        if current is None or _scope_priority(entry) < _scope_priority(current):
            deduplicated[run_id] = entry
    entries = list(deduplicated.values())
    entries.sort(
        key=lambda entry: (
            -float(entry["completed_timestamp"]),
            str(entry["scope"]),
            str(entry["run_id"]),
        )
    )
    retained_count = min(limit, len(entries))
    if entries and retention_days > 0:
        cutoff = float(entries[0]["completed_timestamp"]) - (
            retention_days * _SECONDS_PER_DAY
        )
        recent_count = sum(
            float(entry["completed_timestamp"]) >= cutoff for entry in entries
        )
        retained_count = max(retained_count, recent_count)
    ranked = entries[:retained_count]
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank
        entry["rank_label"] = f"{rank:03d}"
    return ranked


def _scope_priority(entry: dict[str, Any]) -> tuple[int, float]:
    scope = str(entry["scope"])
    return (
        _SCOPE_PRIORITY.get(scope, len(_SCOPE_PRIORITY)),
        -float(entry["completed_timestamp"]),
    )


def write_result_index(
    outputs_root: Path,
    *,
    markdown_output: Path | None = None,
    json_output: Path | None = None,
    roots: tuple[str, ...] = DEFAULT_RESULT_ROOTS,
    limit: int = DEFAULT_INDEX_LIMIT,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> dict[str, Any]:
    markdown_path = markdown_output or outputs_root / "00_RECENT_RESULTS.md"
    json_path = json_output or outputs_root / "00_RECENT_RESULTS.json"
    entries = build_result_index(
        outputs_root,
        roots=roots,
        limit=limit,
        retention_days=retention_days,
    )
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        _render_markdown(
            entries,
            outputs_root=outputs_root,
            markdown_output=markdown_path,
            generated_at=generated_at,
            minimum_entries=limit,
            retention_days=retention_days,
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "sort_rule": "gate > validation > results; descending completion time",
                "retention": {
                    "minimum_entries": limit,
                    "days_from_latest_completion": retention_days,
                },
                "roots": list(roots),
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "pass",
        "entries": len(entries),
        "minimum_entries": limit,
        "retention_days": retention_days,
        "markdown": str(markdown_path),
        "json": str(json_path),
    }


def _index_run(
    outputs_root: Path,
    scope: str,
    run_root: Path,
) -> dict[str, Any] | None:
    artifacts = _find_artifacts(outputs_root, run_root)
    completion_key = next(
        (key for key in ("gate", "validation", "results") if key in artifacts),
        None,
    )
    if completion_key is None:
        return None
    completion_path = outputs_root / artifacts[completion_key]
    completed_timestamp = completion_path.stat().st_mtime
    decision_payload = _load_first_json(
        outputs_root,
        artifacts,
        keys=("gate", "validation"),
    )
    status = str(decision_payload.get("status") or "results_available")
    decision = str(decision_payload.get("decision") or "")
    claim_scope = str(decision_payload.get("claim_scope") or "")
    if scope == "remote_results_incomplete" and (
        run_root / "RAW_RETRIEVAL_NOTICE.txt"
    ).is_file():
        status = "fallback_retrieved"
        if not decision:
            decision = "raw_fallback_incomplete"
    return {
        "run_id": run_root.name,
        "display_name": display_name_for_run(run_root.name),
        "scope": scope,
        "status": status,
        "status_display": STATUS_LABELS.get(status, status),
        "decision": decision,
        "decision_display": DECISION_LABELS.get(decision, decision),
        "claim_scope": claim_scope,
        "completed_at": datetime.fromtimestamp(completed_timestamp)
        .astimezone()
        .isoformat(timespec="seconds"),
        "completed_timestamp": completed_timestamp,
        "completion_source": completion_path.name,
        "path": run_root.relative_to(outputs_root).as_posix(),
        "artifacts": artifacts,
    }


def _find_artifacts(outputs_root: Path, run_root: Path) -> dict[str, str]:
    selectors: tuple[tuple[str, Callable[[Path], bool], tuple[str, ...]], ...] = (
        (
            "gate",
            lambda path: path.name in {"gate.local.json", "gate.json"},
            ("gate.local.json",),
        ),
        (
            "validation",
            lambda path: path.name == "validation.json",
            ("validation.json",),
        ),
        (
            "results",
            lambda path: (
                path.suffix == ".jsonl"
                and path.name != "progress.jsonl"
                and (path.name == "results.jsonl" or path.parent.name == "results")
                and path.stat().st_size > 0
            ),
            ("results.jsonl",),
        ),
        (
            "curves",
            lambda path: path.suffix == ".svg" and "curves" in path.stem,
            ("curves.svg",),
        ),
        (
            "history",
            lambda path: path.suffix == ".csv" and "history" in path.stem,
            ("history.csv",),
        ),
        (
            "progress",
            lambda path: path.name == "progress.jsonl",
            ("progress.jsonl",),
        ),
    )
    artifacts: dict[str, str] = {}
    files = [path for path in run_root.rglob("*") if path.is_file()]
    for key, predicate, preferred_names in selectors:
        candidates = [path for path in files if predicate(path)]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda path: (
                0 if path.name in preferred_names else 1,
                len(path.relative_to(run_root).parts),
                path.as_posix(),
            ),
        )
        artifacts[key] = selected.relative_to(outputs_root).as_posix()
    return artifacts


def _load_first_json(
    outputs_root: Path,
    artifacts: dict[str, str],
    *,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    for key in keys:
        relative = artifacts.get(key)
        if relative is None:
            continue
        try:
            payload = json.loads((outputs_root / relative).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def display_name_for_run(run_id: str) -> str:
    if run_id.startswith(
        "i2_present_next_round_full_state_identifiability_audit"
    ):
        return "创新2：PRESENT完整轮间状态预测的子密钥可识别性审计"
    if run_id.startswith("i2_output_prediction_opb1_present_r3_topology_bottleneck"):
        return "创新2 OPB1：PRESENT三轮低秩拓扑瓶颈真实密文输出预测"
    if run_id.startswith("i2_output_prediction_opa3_present_r3_selected8"):
        return "创新2 OPA3：PRESENT三轮真实P-layer与同容量错误拓扑归因"
    if run_id.startswith("i2_output_prediction_opa2_present_r3_selected8"):
        return "创新2 OPA2：PRESENT三轮候选架构与MLP的第四密钥匹配控制确认"
    if run_id.startswith(
        "i2_output_prediction_opa1_present_r3_selected8_architecture_screen"
    ):
        return "创新2 OPA1：PRESENT三轮固定八个真实密文输出bit的五模型架构发现屏"
    if run_id.startswith("i2_output_prediction_op12_present_r4_structured_xor"):
        return "创新2 OP12：PRESENT四轮多输出bit结构化XOR真实值预测"
    if run_id.startswith("i2_output_prediction_op11_present_r3_selected8"):
        return "创新2 OP11：PRESENT三轮固定八个真实密文输出bit的独立密钥确认"
    if run_id.startswith("i2_output_prediction_op10_present_r3_easy_bit"):
        return "创新2 OP10：PRESENT三轮易预测真实密文输出bit发现与独立确认"
    if run_id.startswith("i2_output_prediction_op9_present_r3_kimura_lstm"):
        return "创新2 OP9：PRESENT三轮Kimura式完整64-bit真实密文输出预测"
    if run_id == (
        "i2_output_parity_prediction_op8_present_r1_r3_exact_anf_"
        "difficulty_20260721"
    ):
        return "创新2 OP8：PRESENT r1--r3真实密文输出parity精确ANF难度审计"
    if run_id == (
        "i2_output_parity_prediction_op7_present_r3_bit_role_routing_seed0_20260721"
    ):
        return "创新2 OP7：PRESENT三轮真实密文输出parity精确bit-role路由门"
    if run_id == (
        "i2_output_parity_prediction_op6_present_r3_spn_local_readiness_seed0_20260721"
    ):
        return "创新2 OP6：PRESENT三轮真实密文输出parity SPN局部网络就绪门"
    if run_id == "i2_output_parity_prediction_op5_present_r3_seed0_20260721":
        return "创新2 OP5-A：PRESENT三轮结构对齐密文输出parity seed0门"
    if run_id == (
        "i2_output_parity_prediction_op5_present_r3_seed1_joint_20260721"
    ):
        return "创新2 OP5：PRESENT三轮结构对齐密文输出parity双密钥门"
    if run_id == "i2_output_parity_prediction_op4_present_r2_seed0_20260721":
        return "创新2 OP4-A：PRESENT二轮结构对齐密文输出parity seed0门"
    if run_id == (
        "i2_output_parity_prediction_op4_present_r2_seed1_joint_20260721"
    ):
        return "创新2 OP4：PRESENT二轮结构对齐密文输出parity双密钥门"
    if run_id == (
        "i2_output_parity_prediction_op3_independent_key_present_r1_seed1_20260721"
    ):
        return "创新2 OP3：PRESENT一轮结构对齐密文输出parity独立密钥确认"
    if run_id == (
        "i2_output_parity_prediction_op2_mask_geometry_present_r1_seed0_20260721"
    ):
        return "创新2 OP2：PRESENT一轮真实密文输出parity mask几何校准"
    if run_id == "i2_output_parity_prediction_readiness_present_r1_seed0_20260721":
        return "创新2 OP1：PRESENT一轮固定密钥密文输出parity预测就绪门"
    if run_id == "i1_feistel_balanced_round_relation_readiness_seed0":
        return "创新1 Feistel：SIMON/SIMECK 真实轮关系模型就绪检查"
    if run_id == "i1_feistel_balanced_round_relation_2048_seed0":
        return "创新1 Feistel：SIMON/SIMECK 真实轮关系 2048/类归因诊断"
    if run_id == "i1_feistel_balanced_round_relation_calibration_2048_seed0":
        return "创新1 Feistel：SIMON r11 / SIMECK r14 低一轮公式校准"
    if run_id == "i1_feistel_lu_senet_layout_calibration_2048_seed0":
        return "创新1 Feistel：Lu 源码 SE-ResNet pair 轴布局校准"
    if run_id == "i1_feistel_round_relation_scale_probe_8192_seed0":
        return "创新1 Feistel：SIMON/SIMECK 轮关系 8192/类数据斜率探针"
    if run_id == "i1_feistel_round_relation_scale_probe_8192_seed1":
        return "创新1 Feistel：SIMON/SIMECK 轮关系 8192/类独立 seed1 确认"
    if run_id == "i1_feistel_round_relation_target_round_8192_seed0":
        return "创新1 Feistel：SIMON r12 / SIMECK r15 论文目标轮 8192/类探针"
    if run_id == "i1_feistel_low_to_target_curriculum_readiness_seed0":
        return "创新1 Feistel：低轮到目标轮同总轮次课程训练就绪检查"
    if run_id == "i1_feistel_low_to_target_curriculum_8192_seed0":
        return "创新1 Feistel：SIMON/SIMECK 低轮到目标轮课程迁移 8192/类裁决"
    if run_id == "i1_feistel_low_to_target_curriculum_8192_seed1_simeck":
        return "创新1 Feistel：SIMECK 低轮到目标轮课程迁移 seed1 确认"
    if run_id == "i1_feistel_simeck_curriculum_65k_seed0":
        return "创新1 Feistel：SIMECK 低轮到目标轮课程迁移 65536/类规模裁决"
    if run_id == "i2_present_r5_structure_integral_parity_smoke_seed0":
        return "创新2 E0：PRESENT 5轮结构条件积分平衡概率预测 Smoke"
    if run_id == "i2_present_r5_structure_integral_parity_feasibility_seed0":
        return "创新2 E0：PRESENT 5轮结构条件积分平衡概率可行性诊断"
    if run_id == "i2_present_r5_integral_parity_calibration_smoke_seed0":
        return "创新2 E1：PRESENT 5轮积分平衡概率独立校准 Smoke"
    if run_id == "i2_present_r5_integral_parity_calibration_seed0":
        return "创新2 E1：PRESENT 5轮积分平衡概率校准与标签稳定性诊断"
    if run_id == "i2_present_r5_integral_parity_ranking_utility_seed0":
        return "创新2 E2：PRESENT 5轮积分输出平衡候选排序与 top-16 效用审判"
    if run_id == "i2_present_r5_integral_parity_ranking_utility_joint_seed0_seed1":
        return "创新2 E3：PRESENT 5轮积分输出候选排序双 seed 联合裁决"
    if run_id == (
        "i2_present_r6_output_property_transition_"
        "width1_width2_seed0_20260717"
    ):
        return "创新2 E7：PRESENT 6轮积分输出性质活动宽度过渡审计"
    if run_id == (
        "i2_present_r6_output_property_active_bits5_6_7_seed0_20260717"
    ):
        return "创新2 E8：PRESENT 6轮积分输出性质细粒度活动 bit 审计"
    if run_id == (
        "i2_present_stable_balance_subspace_r5_r6_bits5_6_7_seed0_20260717"
    ):
        return "创新2 E9：PRESENT 输出平衡 mask 子空间稳定性审计"
    if run_id == (
        "i2_present_r7_hwang_kernel_last16_bitorder_readiness_seed0_20260717"
    ):
        return "创新2 E10：PRESENT 7轮论文输出 mask bit-order 校准"
    if run_id == "i2_present_r7_hwang_kernel_convergence_128keys_seed0_20260717":
        return "创新2 E11：PRESENT 7轮论文四维输出 kernel 收敛审判"
    if run_id == (
        "i2_present_r7_hwang_kernel_convergence_high16_128keys_seed0_20260717"
    ):
        return "创新2 E11b：PRESENT 7轮高16位论文 kernel 同预算对照"
    if run_id == (
        "i2_present_r7_active_block_kernel_diversity_128keys_seed0_20260717"
    ):
        return "创新2 E12：PRESENT 7轮活动块输出 kernel 多样性 readiness"
    if run_id == "i2_present_r7_structure_mask_label_readiness_seed0_20260717":
        return "创新2 E13：PRESENT 7轮结构-mask输出标签边际捷径审计"
    if run_id == (
        "i2_present_r7_cyclic_geometry_kernel_diversity_128keys_seed0_20260717"
    ):
        return "创新2 E14：PRESENT 7轮循环活动几何输出 kernel 扩展"
    if run_id == (
        "i2_present_r7_topology_geometry_kernel_diversity_128keys_seed0_20260717"
    ):
        return "创新2 E15：PRESENT 7轮P-layer拓扑活动几何审计"
    if run_id == (
        "i2_present_r7_inactive_context_kernel_diversity_128keys_seed0_20260717"
    ):
        return "创新2 E16：PRESENT 7轮高16位固定上下文 kernel 审计"
    if run_id == (
        "i2_present_r7_context_mask_label_readiness_seed0_20260717"
    ):
        return "创新2 E17：PRESENT 7轮context-mask输出标签捷径审计"
    if run_id == (
        "i2_present_r7_equal_prevalence_context_mask_readiness_seed0_20260717"
    ):
        return "创新2 E17b：PRESENT 7轮等流行率翻转-mask标签审计"
    if run_id == (
        "i2_present_r7_context_mask_group_disjoint_readiness_seed0_20260717"
    ):
        return "创新2 E17c：PRESENT 7轮context/mask双轴组外捷径审计"
    if run_id == (
        "i2_present_r7_fresh_expanded_context_kernel_128keys_seed0_20260717"
    ):
        return "创新2 E18：PRESENT 7轮64-context fresh-key kernel扩展"
    if run_id == (
        "i2_present_r7_context_mask_balance_rate_128keys_seed0_20260717"
    ):
        return "创新2 E19：PRESENT 7轮跨密钥输出平衡概率审计"
    if run_id == "i2_skinny64_r7_hwang_kernel_readiness_768keys_seed0_20260717":
        return "创新2 E20：SKINNY-64/64 7轮 Hwang exact-kernel 就绪审计"
    if run_id == "i2_skinny64_r8_hwang_kernel_readiness_768keys_seed0_20260717":
        return "创新2 E21：SKINNY-64/64 8轮 two-active-cell kernel 就绪审计"
    if run_id == (
        "i2_skinny64_r8_adjacent_pair_kernel_diversity_128keys_seed0_20260717"
    ):
        return "创新2 E22：SKINNY-64/64 8轮相邻活动pair kernel多样性审计"
    if run_id == (
        "i2_skinny64_r8_bottom_row_pair_closure_128keys_seed0_20260717"
    ):
        return "创新2 E23：SKINNY-64/64 8轮底行活动pair kernel闭合审判"
    if run_id == (
        "i2_skinny64_r7_single_cell_geometry_128keys_seed0_20260717"
    ):
        return "创新2 E24：SKINNY-64/64 7轮单活动cell kernel多样性审计"
    if run_id == "i2_speck32_hwang_phase_b_singlekey_gpu0_20260717":
        return "创新2 E25 Phase B：SPECK32/64精确2^30单key GPU计时门"
    if run_id == "i2_speck32_hwang_phase_c_32plus32_gpu0_20260717":
        return "创新2 E25 Phase C：SPECK32/64 32+32密钥精确kernel复现"
    if run_id == "i2_speck32_hwang_contexts_32plus32_gpu0_20260717":
        return "创新2 E26：SPECK32/64四种固定context kernel审计"
    if run_id == "i2_speck32_hwang_positions_gpu0_20260717":
        return "创新2 E27：SPECK32/64相邻固定位置kernel族筛选"
    if run_id == "i2_speck32_hwang_position_labels_seed0_20260717":
        return "创新2 E28：SPECK32/64位置×mask标签宽度与组外捷径审计"
    if run_id == "i2_speck32_hwang_topology_pairs_gpu0_20260717":
        return "创新2 E27-N：SPECK32/64 ROR7模加对齐与错位控制"
    if run_id == "i2_present_r7_linear_subspace_kernel_diversity_128keys_seed0_20260717":
        return "创新2 E30：PRESENT-80 7轮16维线性子空间kernel多样性"
    if run_id == "i2_present_r9_deterministic_provider_contract_20260718":
        return "创新2 E31：PRESENT高轮确定性积分标签提供者契约审计"
    if run_id == "i2_small_spn_exact_label_width_16ciphers_256keys_seed0_20260718":
        return "创新2 E32：16-bit小状态SPN全key精确标签宽度审计"
    if run_id == "i2_small_spn_matched_contrast_readjudication_20260718":
        return "创新2 E32b：小状态SPN训练内matched-contrast重裁决"
    if run_id == "i2_small_spn_graphgps_scgt_smoke_seed0_20260718":
        return "创新2 E33：小状态SPN GraphGPS/SCGT训练就绪smoke"
    if run_id == "i2_small_spn_graphgps_scgt_seed0_seed1_20260718":
        return "创新2 E33：小状态SPN GraphGPS/SCGT两seed拓扑归因"
    if run_id == "i2_small_spn_cell_equivariance_smoke_seed0_20260718":
        return "创新2 E33-R：cell重标号等变表示就绪smoke"
    if run_id == "i2_small_spn_cell_equivariance_seed0_seed1_20260718":
        return "创新2 E33-R：cell重标号等变GraphGPS两seed归因"
    if run_id == "i2_small_spn_round_shared_reasoner_smoke_seed0_20260718":
        return "创新2 E34：共享轮处理器就绪smoke"
    if run_id == "i2_small_spn_round_shared_reasoner_seed0_seed1_20260718":
        return "创新2 E34：共享轮处理器两seed拓扑归因"
    if run_id == "i2_small_spn_cipher_edge_token_smoke_seed0_20260718":
        return "创新2 E35：Cipher Edge-Token Transformer就绪smoke"
    if run_id == "i2_small_spn_cipher_edge_token_seed0_seed1_20260718":
        return "创新2 E35：Cipher Edge-Token Transformer两seed归因"
    if run_id == "i2_small_spn_cipher_edge_token_fair_control_smoke_seed0_20260718":
        return "创新2 E35b：Cipher Edge-Token Transformer公平控制就绪smoke"
    if run_id == "i2_small_spn_cipher_edge_token_fair_control_seed0_seed1_20260718":
        return "创新2 E35b：Cipher Edge-Token Transformer公平控制重裁决"
    if run_id == "i2_small_spn_topology_label_identifiability_20260718":
        return "创新2 E36：小状态SPN拓扑标签可识别性审计"
    if run_id == "i2_small_spn_expanded_topology_4s16p_256keys_20260718":
        return "创新2 E37：4×16小状态SPN扩展拓扑benchmark审计"
    if run_id == "i2_small_spn_expanded_neural_screen_smoke_seed0_20260718":
        return "创新2 E38：扩展拓扑GraphGPS/CETT筛选就绪smoke"
    if run_id == "i2_small_spn_expanded_neural_screen_seed0_seed1_20260718":
        return "创新2 E38：扩展拓扑GraphGPS/CETT两seed候选筛选"
    if run_id == "i2_small_spn_pair_relation_reasoner_smoke_seed0_20260718":
        return "创新2 E39：有向bit-pair路径推理器就绪smoke"
    if run_id == "i2_small_spn_pair_relation_reasoner_seed0_seed1_20260718":
        return "创新2 E39：有向bit-pair路径推理器两seed筛选"
    if run_id == "i2_small_spn_pair_relation_fair_control_seed0_seed1_20260718":
        return "创新2 E39 Phase B：有向bit-pair路径推理器公平拓扑归因"
    if run_id == "i2_small_spn_pair_relation_no_triangle_smoke_seed0_20260718":
        return "创新2 E40：SPN-PRR no-triangle消融就绪smoke"
    if run_id == "i2_small_spn_pair_relation_no_triangle_seed0_seed1_20260718":
        return "创新2 E40：SPN-PRR同预算no-triangle路径归因"
    if run_id == (
        "i2_small_spn_pair_relation_no_triangle_fair_control_seed0_seed1_20260718"
    ):
        return "创新2 E41：局部pair-state公平拓扑归因"
    if run_id == "i2_real_spn_pair_state_transfer_readiness_20260718":
        return "创新2 E42：真实SPN标签与64-bit pair-state迁移readiness"
    if run_id == "i2_present_r4_universal_balance_atlas_20260718":
        return "创新2 E43：PRESENT四轮全称平衡证书/反例atlas"
    if run_id == "i2_present_r4_pair_state_neural_attribution_seed0_20260718":
        return "创新2 E44：PRESENT四轮64-bit pair-state神经归因"
    if run_id == "i2_present_r4_pair_state_neural_attribution_smoke_seed0_20260718":
        return "创新2 E44：PRESENT四轮64-bit pair-state训练就绪smoke"
    if run_id == "i2_present_r4_certificate_complexity_attribution_20260718":
        return "创新2 E45：PRESENT四轮证书复杂度与拓扑特征归因"
    if run_id == "i2_present_r4_mspn_readiness_smoke_seed0_20260718":
        return "创新2 E46：PRESENT四轮MSPN训练就绪smoke"
    if run_id == "i2_present_r4_mspn_neural_attribution_seed0_20260718":
        return "创新2 E47：PRESENT四轮MSPN正式神经归因"
    if run_id == "i2_present_r4_support_identity_collision_20260718":
        return "创新2 E48：PRESENT四轮support身份碰撞审计"
    if run_id == (
        "i2_present_r4_degree_spectrum_distillation_readiness_seed0_20260718"
    ):
        return "创新2 E49：PRESENT四轮中间degree谱蒸馏readiness"
    if run_id == "i2_present_r4_cgpr_readiness_seed0_20260718":
        return "创新2 E50：PRESENT四轮证书引导pair-state残差readiness"
    if run_id == "i2_present_r4_cgpr_neural_attribution_seed0_20260718":
        return "创新2 E51：PRESENT四轮CGPR正式残差与拓扑归因"
    if run_id == "i2_present_r5_strict_label_provider_coverage_20260718":
        return "创新2 E52：PRESENT五轮严格标签提供者覆盖审计"
    if run_id == "i2_present_r5_open_3sdp_exact_anf_phase_a_20260718":
        return "创新2 E53-A：PRESENT开放3SDP exact-ANF与消去校准"
    if run_id == "i2_present_r5_open_3sdp_glpk_blocking_gate_20260718":
        return "创新2 E53-B：PRESENT S-box GLPK逐解blocking扩展性门"
    if run_id == "i2_present_r5_transition_tensor_boundary_audit_20260718":
        return "创新2 E54：PRESENT五轮full-superpoly tensor语义边界审计"
    if run_id == "i2_present_r3_query_cone_sparse_anf_growth_20260718":
        return "创新2 E55：PRESENT三轮query-cone exact sparse-ANF硬cap门"
    if run_id == "i2_present_r9_generalized_integral_relation_contract_20260718":
        return "创新2 E56：PRESENT九轮广义积分relation神经标签契约审计"
    if run_id == "i2_present_r9_generalized_relation_precursor_boundary_20260718":
        return "创新2 E57：PRESENT九轮广义relation precursor标量边界"
    if run_id == "i2_present_atm_native_sat_provider_phase_a_20260718":
        return "创新2 E58-A：ATM原生PySAT见证机制校准"
    if run_id == "i2_present_atm_native_sat_r9_singleton_probe_20260718":
        return "创新2 E58-B：ATM原生SAT九轮严格负类单候选探针"
    if run_id == "i2_present_r2_atm_strict_relation_panel_20260718":
        return "创新2 E59：PRESENT两轮ATM严格relation标签面板"
    if run_id == "i2_present_r2_atm_cone_matched_panel_20260718":
        return "创新2 E60：PRESENT两轮ATM依赖锥匹配标签审计"
    if run_id == "i2_present_r2_atm_multicoordinate_support_phase_a_20260718":
        return "创新2 E61-A：PRESENT两轮ATM多坐标消去支撑门"
    if run_id == "i2_small_spn_multicoordinate_relation_readiness_20260718":
        return "创新2 E62：小型SPN严格多坐标relation readiness"
    if run_id == "i2_small_spn_rcca_readiness_seed0_20260718":
        return "创新2 E63：DeepSets/RCCA训练readiness"
    if run_id == "i2_small_spn_rcca_seed0_seed1_20260718":
        return "创新2 E63：DeepSets/RCCA正式双seed筛选"
    if run_id == "i2_small_spn_relation_decomposition_20260718":
        return "创新2 E64：多坐标relation非平凡消去分解"
    if run_id == "i2_present_r4_unit_balance_profile_readiness_20260718":
        return "创新2 E65：PRESENT四轮单位输出平衡谱readiness"
    if run_id == (
        "i2_present_r4_prefix_guided_profile_operator_readiness_seed0_20260718"
    ):
        return "创新2 E66：PRESENT四轮prefix引导平衡谱算子readiness"
    if run_id == (
        "i2_present_r4_prefix_guided_profile_operator_attribution_seed0_20260718"
    ):
        return "创新2 E67：PRESENT四轮prefix引导平衡谱算子正式归因"
    if run_id == "i2_present_r4_prefix_guided_profile_operator_seed1_20260718":
        return "创新2 E68：PRESENT四轮prefix引导平衡谱算子双seed复核"
    if run_id == "i2_present_r4_multibit_mask_profile_readiness_20260718":
        return "创新2 E69：PRESENT四轮多bit linear-mask profile审计"
    if run_id == "i2_present_r4_active_dimension_zero_shot_transfer_20260718":
        return "创新2 E70：PRESENT四轮unit-profile跨活动维度零样本迁移"
    if run_id == (
        "i2_present_r4_round_recurrent_profile_operator_readiness_seed0_20260718"
    ):
        return "创新2 E71：PRESENT四轮显式轮序平衡谱算子readiness"
    if run_id == "i2_present_r4_round_slice_direction_attribution_20260718":
        return "创新2 E72：PRESENT四轮平衡谱前缀轮切片方向归因"
    if run_id == "i2_present_r4_r3_only_profile_operator_readiness_seed0_20260718":
        return "创新2 E73：PRESENT四轮r3-only平衡谱算子readiness"
    if run_id == "i2_present_r4_r3_only_profile_operator_attribution_seed0_20260718":
        return "创新2 E73：PRESENT四轮r3-only平衡谱算子正式归因"
    if run_id == "i2_present_r4_r3_only_profile_operator_seed1_20260718":
        return "创新2 E73：PRESENT四轮r3-only平衡谱算子双seed复核"
    if run_id == "i2_gift64_r4_unit_balance_profile_readiness_20260718":
        return "创新2 E74：GIFT-64四轮严格单位输出平衡谱标签readiness"
    if run_id == "i2_gift64_r4_unit_balance_profile_192_structures_20260719":
        return "创新2 E75：GIFT-64四轮严格单位输出平衡谱192结构容量复核"
    if run_id == "i2_gift64_r4_r3_only_profile_operator_readiness_seed0_20260719":
        return "创新2 E76：GIFT-64四轮r3-only平衡谱算子readiness"
    if run_id == "i2_gift64_r4_topology_interaction_readjudication_20260719":
        return "创新2 E77：GIFT-64四轮拓扑交互公平基线与同权重再审判"
    if run_id == "i2_gift64_r4_r3_only_profile_operator_attribution_seed0_20260719":
        return "创新2 E78：GIFT-64四轮r3-only平衡谱算子30轮seed0归因"
    if run_id == "i2_gift64_r4_r3_only_profile_operator_seed1_20260719":
        return "创新2 E79：GIFT-64四轮r3-only平衡谱算子30轮双seed确认"
    if run_id == "i2_cross_spn_r3_profile_operator_method_synthesis_20260719":
        return "创新2 E80：PRESENT/GIFT r3-only平衡谱算子方法级综合"
    if run_id == "i2_skinny64_r4_unit_balance_profile_readiness_20260719":
        return "创新2 E81：SKINNY-64四轮严格单位输出平衡谱标签readiness"
    if run_id == "i2_skinny64_r5_unit_balance_profile_transition_20260719":
        return "创新2 E82：SKINNY-64五轮严格单位输出平衡谱标签过渡"
    if run_id == (
        "i2_skinny64_r5_r4_only_sparse_profile_operator_"
        "readiness_seed0_20260719"
    ):
        return "创新2 E83：SKINNY-64五轮r4-only稀疏线性层算子readiness"
    if run_id == (
        "i2_skinny64_r5_true_ridge_sparse_residual_"
        "readiness_seed0_20260719"
    ):
        return "创新2 E84：SKINNY-64五轮真实拓扑ridge引导稀疏残差readiness"
    if run_id == (
        "i2_present_gift_r4_topology_parameterized_shared_profile_operator_"
        "readiness_seed0_20260719"
    ):
        return "创新2 E85：PRESENT/GIFT拓扑参数化共享Profile Operator readiness"
    if run_id == (
        "i2_present_gift_r4_topology_parameterized_shared_profile_operator_"
        "attribution_seed0_20260719"
    ):
        return "创新2 E86：PRESENT/GIFT共享Profile Operator 30轮seed0归因"
    if run_id == "i2_rectangle80_r4_unit_balance_profile_readiness_20260719":
        return "创新2 E87：RECTANGLE-80四轮严格unit平衡谱标签readiness"
    if run_id == "i2_rectangle80_r4_unit_balance_profile_192_structures_20260719":
        return "创新2 E88：RECTANGLE-80四轮严格unit平衡谱192结构容量复核"
    if run_id == (
        "i2_rectangle80_r4_r3_only_profile_operator_readiness_seed0_20260719"
    ):
        return "创新2 E89：RECTANGLE-80四轮r3-only平衡谱算子readiness"
    if run_id == (
        "i2_rectangle80_r4_r3_only_profile_operator_attribution_seed0_20260719"
    ):
        return "创新2 E90：RECTANGLE-80四轮r3-only平衡谱算子30轮seed0归因"
    if run_id == "i2_rectangle80_row_typed_shift_representation_audit_20260719":
        return "创新2 E91：RECTANGLE row-typed ShiftRow表示无训练审计"
    if run_id == (
        "i2_rectangle80_row_typed_shift_operator_readiness_seed0_20260719"
    ):
        return "创新2 E92：RECTANGLE参数零增量Row-Typed Shift Operator readiness"
    if run_id == "i2_neural_architecture_boundary_synthesis_20260719":
        return "创新2 E93：跨SPN神经结构证据与边界综合"
    if run_id == "i2_rectangle80_r4_nested_cube_monotonic_readiness_20260719":
        return "创新2 E94：RECTANGLE-80四轮7/8/9-bit嵌套cube单调标签门"
    if run_id == "i2_rectangle80_r4_nested_cube_relation_mechanism_20260719":
        return "创新2 E95：RECTANGLE-80四轮嵌套cube关系无训练机制门"
    if run_id == "i2_post_e95_architecture_portfolio_boundary_20260719":
        return "创新2 E96：E95后神经架构候选组合边界复核"
    if run_id == "i2_present_r5_cancellation_provider_feasibility_20260719":
        return "创新2 E97：PRESENT五轮非平凡GF(2)消去provider可行性审计"
    if run_id == "i2_present_r9_generalized_relation_pu_ranking_readiness_20260719":
        return "创新2 E98：PRESENT九轮广义积分关系正例-未标注排序就绪审判"
    if run_id == "i2_present_r9_atm_basis_merge_source_audit_20260720":
        return "创新2 E98-A：PRESENT九轮ATM基底合并与缺失split来源审计"
    if run_id == "i2_present_r9_atm_support_component_pu_readiness_20260720":
        return "创新2 E98-B：PRESENT九轮ATM支撑组件互斥PU排序就绪门"
    if run_id == "i2_present_r9_atm_support_rotation_orbit_pu_readiness_20260720":
        return "创新2 E98-C：PRESENT九轮ATM支撑与旋转轨道双互斥PU就绪门"
    if run_id == (
        "i2_present_r9_atm_support_component_pu_neural_ranking_"
        "seed0_seed1_20260720"
    ):
        return "创新2 E99：PRESENT九轮ATM轨道互斥PU神经排序双seed门"
    if run_id == (
        "i2_present_r9_identity_topology_residual_attribution_"
        "seed0_seed1_20260720"
    ):
        return "创新2 E100：PRESENT九轮坐标身份保持拓扑残差归因"
    if run_id == (
        "i2_present_r9_atm_e99_coordinate_checkpoint_replay_"
        "seed0_seed1_20260720"
    ):
        return "创新2 E105-F：PRESENT九轮E99坐标模型权重冻结重放"
    if run_id == "i2_present_r9_atm_split333_resumable_generation_20260720":
        return "创新2 E104：PRESENT九轮ATM缺失(3,3,3) split可恢复生成"
    if run_id == (
        "i2_present_r9_r10_atm_source_generation_resume_readiness_20260720"
    ):
        return "创新2 E101：PRESENT九/十轮ATM新来源生成与恢复就绪审计"
    if run_id == "i2_present_atm_resumable_search_runner_fixture_20260720":
        return "创新2 E102：PRESENT ATM逐候选断点恢复一致性门"
    if run_id == "i2_present_sbox4_r3_real_atm_runner_compatibility_20260720":
        return "创新2 E103：PRESENT S-box 4-bit三轮真实ATM兼容性门"
    if run_id == "i2_present_r9_external_relation_source_readiness_20260721":
        return "创新2 E106：PRESENT九轮外部关系来源新颖性就绪审计"
    if run_id == (
        "i2_present_r8_high_round_integral_bridge_262144_joint_"
        "seed0_seed1_20260716"
    ):
        return "创新2：PRESENT-80 8轮 262144-total 双 seed bridge 联合裁决"
    if run_id == (
        "i2_present_r8_high_round_integral_paper_reference_"
        "2pow21_seed0_gpu0_20260716"
    ):
        return "创新2：PRESENT-80 8轮 2^21-total / 50轮训练论文参考规模近似"
    if run_id == (
        "i2_present_r8_high_round_integral_paper_reference_"
        "2pow21_joint_seed0_seed1"
    ):
        return "创新2：PRESENT-80 8轮论文参考规模双 seed 联合裁决"
    ranking_seed = re.fullmatch(
        r"i2_present_r5_integral_parity_ranking_utility_seed(?P<seed>\d+)",
        run_id,
    )
    if ranking_seed:
        return (
            "创新2 E3：PRESENT 5轮积分输出平衡候选排序独立确认，"
            f"seed {ranking_seed.group('seed')}"
        )
    calibration_seed = re.fullmatch(
        r"i2_present_r5_integral_parity_calibration_seed(?P<seed>\d+)",
        run_id,
    )
    if calibration_seed:
        return (
            "创新2 E3：PRESENT 5轮积分平衡概率与标签稳定性独立诊断，"
            f"seed {calibration_seed.group('seed')}"
        )
    if run_id == "i1_cross_spn_e4_final_synthesis_20260715":
        return "创新1 E4：跨 SPN 类型拓扑四个目标 cell 最终证据综合"
    if run_id == "i1_gift64_cross_spn_source_seed_r5_65536_joint_seed4_seed5":
        return "创新1 E4-R5：独立 PRESENT source-seed 稳健性联合裁决"
    r5_medium = re.fullmatch(
        r"i1_gift64_cross_spn_source_seed_r5_65536_seed(?P<seed>\d+)",
        run_id,
    )
    if r5_medium:
        return (
            "创新1 E4-R5：独立 source checkpoint 的一轮 GIFT-64 适配，"
            f"目标 seed {r5_medium.group('seed')}"
        )
    r5_readiness = re.fullmatch(
        r"i1_gift64_cross_spn_source_seed_r5_readiness_seed(?P<seed>\d+)",
        run_id,
    )
    if r5_readiness:
        return (
            "创新1 E4-R5：独立 source checkpoint 目标适配就绪检查，"
            f"目标 seed {r5_readiness.group('seed')}"
        )
    if run_id == "i1_present_cross_spn_source_seed_r5_8192_seed1":
        return "创新1 E4-R5 Phase A：独立 PRESENT source seed1 门控"
    if run_id == ("i1_gift64_cross_spn_target_adaptation_r4_65536_joint_seed2_seed3"):
        return "创新1 E4-R4：PRESENT → GIFT-64 双 seed 目标适配效率联合裁决"
    r4_medium = re.fullmatch(
        r"i1_gift64_cross_spn_target_adaptation_r4_65536_seed(?P<seed>\d+)",
        run_id,
    )
    if r4_medium:
        return (
            "创新1 E4-R4：PRESENT → GIFT-64 一轮目标适配效率，"
            f"目标 seed {r4_medium.group('seed')}"
        )
    r4_readiness = re.fullmatch(
        r"i1_gift64_cross_spn_target_adaptation_r4_readiness_seed(?P<seed>\d+)",
        run_id,
    )
    if r4_readiness:
        return (
            "创新1 E4-R4：跨 SPN 一轮目标适配实验就绪检查，"
            f"目标 seed {r4_readiness.group('seed')}"
        )
    if run_id == ("i1_gift64_cross_spn_typed_transfer_r3_65536_joint_seed0_seed1"):
        return "创新1 E4-R3：PRESENT → GIFT-64 跨 SPN 双 seed 中等规模联合裁决"
    r3_medium = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r3_65536_seed(?P<seed>\d+)",
        run_id,
    )
    if r3_medium:
        return (
            "创新1 E4-R3：PRESENT → GIFT-64 跨 SPN 中等规模迁移，"
            f"目标 seed {r3_medium.group('seed')}"
        )
    r3_readiness = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r3_readiness_seed(?P<seed>\d+)",
        run_id,
    )
    if r3_readiness:
        return (
            "创新1 E4-R3：跨 SPN 中等规模实验就绪检查，"
            f"目标 seed {r3_readiness.group('seed')}"
        )
    if run_id == "i1_gift64_cross_spn_typed_transfer_r2_joint_seed0_seed1":
        return "创新1 E4-R2：PRESENT → GIFT-64 跨 SPN 双 seed 联合裁决"
    transfer = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r2_seed(?P<seed>\d+)",
        run_id,
    )
    if transfer:
        return (
            "创新1 E4-R2：PRESENT → GIFT-64 跨 SPN 结构迁移，"
            f"目标 seed {transfer.group('seed')}"
        )
    readiness = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r0_seed(?P<seed>\d+)",
        run_id,
    )
    if readiness:
        return f"创新1 E4-R0：GIFT-64 迁移实现就绪检查，seed {readiness.group('seed')}"
    if run_id == "i1_cross_spn_typed_cell_r1_seed0":
        return "创新1 E4-R1：PRESENT/GIFT-64 共享类型单元联合门控"
    if run_id == "i1_cross_spn_typed_cell_r0_seed0":
        return "创新1 E4-R0：PRESENT/GIFT-64 共享类型单元实现就绪门控"
    source = re.fullmatch(
        r"i1_(?P<cipher>present|gift64)_cross_spn_typed_cell_r1_seed(?P<seed>\d+)",
        run_id,
    )
    if source:
        cipher = "PRESENT" if source.group("cipher") == "present" else "GIFT-64"
        return f"创新1 E4-R1：{cipher} 共享类型单元训练，seed {source.group('seed')}"
    return run_id


def _render_markdown(
    entries: list[dict[str, Any]],
    *,
    outputs_root: Path,
    markdown_output: Path,
    generated_at: str,
    minimum_entries: int,
    retention_days: int,
) -> str:
    if retention_days > 0:
        retention_note = (
            f"> 保留规则：至少显示最新 {minimum_entries} 条，并保留距最新完成结果 "
            f"{retention_days} 天内的所有条目；实验密集时会超过 {minimum_entries} 条。"
        )
    else:
        retention_note = f"> 保留规则：显示最新 {minimum_entries} 条。"
    lines = [
        "# 最近实验结果",
        "",
        "> `001` 永远表示最新完成的结果。排序优先使用门控、验证、结果文件的完成时间；重新生成曲线不会改变实验先后顺序。",
        "",
        retention_note,
        "",
        f"更新时间：`{generated_at}`",
        "",
        "| 序号 | 完成时间 | 实验说明 | 位置 | 状态 / 裁决 | 快速查看 |",
        "|---:|---|---|---|---|---|",
    ]
    for entry in entries:
        completed = str(entry["completed_at"]).replace("T", " ")
        status = str(entry["status_display"])
        if entry["decision_display"]:
            status = f"{status} / {entry['decision_display']}"
        links = _artifact_links(
            entry["artifacts"],
            run_path=str(entry["path"]),
            outputs_root=outputs_root,
            markdown_output=markdown_output,
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(entry["rank_label"]),
                    _escape_table(completed),
                    _escape_table(str(entry["display_name"])),
                    _escape_table(str(entry["scope"])),
                    _escape_table(status),
                    links or "-",
                )
            )
            + " |"
        )
    if not entries:
        lines.append("| - | - | 暂无可索引的结果 | - | - | - |")
    lines.extend(
        (
            "",
            "说明：原始实验目录名保持不变，避免破坏配置、文档和门控中的证据路径。",
            "",
        )
    )
    return "\n".join(lines)


def _artifact_links(
    artifacts: dict[str, str],
    *,
    run_path: str,
    outputs_root: Path,
    markdown_output: Path,
) -> str:
    directory_target = os.path.relpath(outputs_root / run_path, markdown_output.parent)
    links = [f"[目录]({Path(directory_target).as_posix()})"]
    for key in ("curves", "gate", "validation", "results", "history", "progress"):
        relative = artifacts.get(key)
        if relative is None:
            continue
        target = os.path.relpath(outputs_root / relative, markdown_output.parent)
        links.append(f"[{ARTIFACT_LABELS[key]}]({Path(target).as_posix()})")
    return " ".join(links)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")

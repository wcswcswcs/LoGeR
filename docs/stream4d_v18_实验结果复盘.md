# Stream4D v18 Signed Boundary Graph 实验结果复盘

日期: 2026-06-09
计划文档: `docs/stream4d_v18_signed_boundary_graph_plan_for_codex.md`
执行日志: `docs/stream4d_v18_执行日志.md`
结果根目录: `Stream3D/outputs/audit`

结论先行: v18 完成了 signed boundary graph 工具链与 Phase0/Phase1/Phase2 诊断，但没有得到可报告方法成功。Phase1 的 graph 巨大粘连 blocker 被非 GT pre-cut 修复；然而 Phase2 GT-only edge oracle 没过最小 gate，最佳 bank16 oracle 只有 `0.434865 / 0.581342 / 0.664137`，AP25 低于 `0.70`，且 node/edge GT coverage 只有约 `0.45/0.41`。按计划停止，未运行 Phase3 non-GT evidence 和 Phase4 partition。

本文件只基于已落盘 artifact 生成；未运行、未写出或不可推断的字段保持 NA/unavailable。

## Phase0: Unified Eval Matrix

输出: `Stream3D/outputs/audit/v18_phase0/unified_eval_matrix_probe5.*`

| row | AP | AP50 | AP25 | pre ratio | union ratio | manifest |
|---|---:|---:|---:|---:|---:|---|
| P0 on S0 | 0.235730 | 0.414306 | 0.537786 | 0.846744 | 0.846744 | True |
| P0 on S1 | 0.399213 | 0.597171 | 0.742535 | 0.045145 | 0.846744 | True |
| O38 own | 0.081038 | 0.219225 | 0.492501 | 0.666809 | 0.666809 | True |
| repair_cmask own | 0.101653 | 0.248464 | 0.494844 | 0.608353 | 0.608353 | True |
| repair_cmask on S1 | 0.102883 | 0.242779 | 0.576250 | 0.045145 | 0.608353 | True |
| P_v6compact on S1 | 0.284832 | 0.503962 | 0.671915 | 0.045145 | 0.041578 | True |

Phase0 作用: 提供 v18 对照背景，不产生新方法结论。

## Phase1: Signed Surfel Graph

| graph | nodes | edges | track mean | uv in01 | cycle p90 | raw largest | largest after pre-cut | pre-cut removed | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| initial k8 | 16384.0 | 195850.4 | 13.284717 | 0.985845 | 3.273727 | NA | 1.000000 | NA | False |
| k4 no-cross | 16384.0 | 117147.0 | 13.284717 | 0.985845 | 3.273727 | NA | 1.000000 | NA | False |
| pre-cut k8 d0.25 | 16384.0 | 195850.4 | 13.284717 | 0.985845 | 3.273727 | 1.000000 | 0.828467 | 0.170913 | True |
| pre-cut k16 d0.15 | 16384.0 | 271744.2 | 13.284717 | 0.985845 | 3.273727 | 1.000000 | 0.681323 | 0.214480 | True |
| grid48 pre-cut k16 d0.15 | 36864.0 | 625660.8 | 13.341786 | 0.988227 | 3.284617 | 1.000000 | 0.813097 | 0.188202 | True |

Phase1 判定:

- 初始 graph 和 k4/no-cross 都失败，证明巨大粘连不是单纯 kNN k 太大或 cross-frame 边导致。
- 新增非 GT `precut_keep` 后，aggregate largest component ratio 从 raw `1.0` 降到 `0.828467` / `0.681323`，Phase1 aggregate gate 通过。
- `precut_keep` 只使用预测 mask disagreement、source RGB discontinuity、UV discontinuity，不读 GT；raw graph 仍保留在 artifact 中。
- `pre-cut k16 d0.15` 作为主 Phase2 graph，因为它同时修复 edge count 与 aggregate 粘连。
- 逐 scene residual 未隐藏: `scene0011_00` track mean `9.640198` 且 largest `0.952698`，`scene0030_00` largest `0.253357`，`scene0081_01` largest `0.961792`。这说明 aggregate 通过不等于所有 scene 都完美。

## Phase2: GT-only Edge Boundary Oracle

Phase2 最小 gate:

- oracle AP >= `0.25`
- oracle AP50 >= `0.50`
- oracle AP25 >= `0.70`
- graph coverage 计划目标: node >= `0.70`, edge >= `0.60`

| run | AP | AP50 | AP25 | node cov | edge cov | exported objects | exported points | AP gate | coverage gate | min gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| bank16 k16 d0.15 main | 0.434865 | 0.581342 | 0.664137 | 0.450122 | 0.410739 | 8.6 | 4848.2 | NA/False after gate fix | NA/False after gate fix | False |
| nn radius 0.08 | 0.420261 | 0.581342 | 0.664137 | 0.467236 | 0.429066 | 8.6 | 4886.8 | NA/False after gate fix | NA/False after gate fix | False |
| export core radius 0.08 | 0.431795 | 0.581342 | 0.664137 | 0.450122 | 0.410739 | 8.6 | 4900.8 | False | False | False |
| oracle min surfels 5 | 0.434865 | 0.581342 | 0.664137 | 0.450122 | 0.410739 | 8.6 | 4848.2 | False | False | False |
| grid48 k16 d0.15 | 0.396002 | 0.504018 | 0.638514 | 0.454085 | 0.425560 | 9.0 | 6457.0 | False | False | False |
| bank16 k8 d0.25 | 0.434865 | 0.581342 | 0.664137 | 0.450122 | 0.414456 | 8.6 | 4845.6 | False | False | False |

注: main/nn rows 是在 gate 拆分修复前生成的 artifact，因此 JSON 中没有 `phase2_oracle_ap_gate` / `phase2_graph_coverage_gate` 字段；按相同阈值重判均为 False。后续 rows 已落盘拆分 gate 字段。

Phase2 判定:

- 最佳 AP/AP50/AP25 是 bank16 oracle: `0.434865 / 0.581342 / 0.664137`。
- AP 和 AP50 达到 oracle AP 阈值，但 AP25 没到 `0.70`。
- 更严重的是 coverage: 最佳 node/edge coverage 只有 `0.467236 / 0.429066`，远低于 `0.70 / 0.60`。
- grid48 将 surfel 数从 `16384` 增至 `36864`，但 coverage 只到 `0.454085 / 0.425560`，AP25 降到 `0.638514`，说明单纯加密 grid 不是当前主要瓶颈。
- export radius、GT label nn radius、小组件阈值都没有修复 AP25。

## Phase3/Phase4

- Phase3 evidence variants: 未运行。
- Phase4 signed graph partition: 未运行。
- cross-support controls: 未运行。
- final method package: 无。

原因: Phase2 GT-only oracle 未达到最小 gate。按计划，oracle 不过不能启动 non-GT solver，也不能报告 D4RT-native signed graph 方法收益。

## Blocker Repair 记录

1. graph 巨大粘连:
   - 现象: initial/k4 graph `largest_graph_component_ratio=1.0`。
   - 修复: 增加非 GT pre-cut。
   - 结果: aggregate largest 降至 `0.681323`，Phase1 aggregate gate pass。

2. Phase2 coverage/AP25 不足:
   - 尝试 GT label nn radius `0.08`: coverage 小幅上升，AP25 不变。
   - 尝试 oracle export core radius `0.08`: exported points 小幅上升，AP25 不变。
   - 尝试 oracle min surfels `5`: exported object 数变化但 AP25 不变。
   - 尝试 grid48: surfel 数增加到 `36864`，但 oracle AP25 下降到 `0.638514`。
   - 尝试 bank16 k8 d0.25 对照: 与 k16 d0.15 几乎一致，说明不是 k16 pre-cut 参数单独导致。

3. 代码审计修复:
   - Phase2 backprojection KDTree 改为每 scene 复用，减少重复构建。
   - oracle export radius 参数化。
   - Phase2 gate 拆成 AP gate 和 graph coverage gate，防止 coverage 未过时误启动 Phase3。

## Insight 与证据链

- v18 的 Phase1 证明 signed surfel graph 不是完全不可构造: graph 节点覆盖、track length、UV/cycle 质量都达标，且巨大粘连可以用非 GT pre-cut 缓解。
- 但 Phase2 证明当前 surfel-to-mesh/object materialization 上界不足。即使用 GT edge labels 直接删除跨 GT 边，oracle 也没达到 AP25 `0.70`，这不是 non-GT classifier 能绕过的问题。
- node/edge GT coverage 是核心短板。`node_gt_label_coverage≈0.45` 意味着大量 surfel 虽然在 2D/track 中有效，但没有可靠落到可评估 GT instance；edge coverage 同样只有约 `0.41-0.43`。
- grid48 修复没有改善 coverage，说明不是“采样点太少”这一单一问题，更像是 UV->depth->mesh 映射、eval bridge materialization、或 object support 从 sparse surfel 到 mesh 点扩张这条链路存在瓶颈。
- 当前不能证明 signed boundary graph 思路无效；只能证明“在当前 bank/export/materialization 设计下，GT oracle 上界不足，不能进入 non-GT solver”。

## 必答问题

1. v18 是否得到可报告方法成功: `False`。
2. Phase1 graph 是否构建成功: `aggregate yes`，但有逐 scene residual。
3. Phase2 GT-only oracle 是否过最小 gate: `False`。
4. 是否运行 Phase3/Phase4: `False`，按计划停止。
5. 是否有 GT 泄漏到 method result: `No method result was produced`；所有 oracle configs 均为 diagnostic-only / forbidden。
6. 当前最强 oracle: bank16 k16 d0.15 或 bank16 k8 d0.25，AP/AP50/AP25 `0.434865 / 0.581342 / 0.664137`。
7. 后续方向: 优先修 surfel-to-mesh / support materialization coverage，例如更可靠的 RGB-D hit association、multi-frame mesh union、mask-region fill 与 component-supported dilation；不建议在当前 oracle gate 未过时继续调 non-GT edge evidence 或 partition solver。

## 审计材料

- Phase0: `Stream3D/outputs/audit/v18_phase0`
- Phase1 main/repairs:
  - `Stream3D/outputs/audit/v18_phase1`
  - `Stream3D/outputs/audit/v18_phase1_repair_k4_nocross`
  - `Stream3D/outputs/audit/v18_phase1_repair_precut_k8`
  - `Stream3D/outputs/audit/v18_phase1_repair_precut_k16_d015`
  - `Stream3D/outputs/audit/v18_phase1_grid48_precut_k16_d015`
- Phase2:
  - `Stream3D/outputs/audit/v18_phase2_precut_k16_d015`
  - `Stream3D/outputs/audit/v18_phase2_repair_nn008`
  - `Stream3D/outputs/audit/v18_phase2_repair_export008`
  - `Stream3D/outputs/audit/v18_phase2_repair_min5`
  - `Stream3D/outputs/audit/v18_phase2_grid48_precut_k16_d015`
  - `Stream3D/outputs/audit/v18_phase2_repair_bank16_k8_d025`
- grid48 carriers: `Stream3D/outputs/v18_d4rt_grid_surfel_field/stream4d_v18_g1_grid48m002_probe5_16f_stride1_gpu67`
- grid48 bank: `Stream3D/outputs/v18_measurement_bank_grid48_cropformer`
- logs: `Stream3D/outputs/audit/v18_logs`
- reportable scan: `Stream3D/outputs/audit/v18_final/oracle_reportable_scan.*`

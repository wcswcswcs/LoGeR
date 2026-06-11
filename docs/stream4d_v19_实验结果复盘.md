# Stream4D v19 4D TubeCover / Materialization 实验结果复盘

日期: 2026-06-09
计划文档: `docs/stream4d_v19_4d_tubecover_materialization_plan_for_codex.md`
执行日志: `docs/stream4d_v19_执行日志.md`
结果根目录: `Stream3D/outputs/audit`

结论先行: v19 完成了 Phase0 复现、Phase1 GT oracle failure decomposition、Phase2A materialization variants 与三组 blocker repair，但没有得到可报告方法成功。最关键的发现是: 已覆盖 mesh 子集上的 Oracle C 可以达到 AP/AP50/AP25 `1.0/1.0/1.0`，但覆盖范围极小；主 M1 的 materialized mesh coverage 只有 `0.019867`，node/edge GT coverage 仍是 `0.450122/0.410739`。因此当前失败主因不是“covered mesh 上不会分 object”，而是 D4RT surfel/tube 与可评估 ScanNet GT object surface 的覆盖和落点不足。按计划停止，未运行 TubeCover、non-GT edge evidence、partition 或 method table。

本文件只基于已落盘 artifact 生成；未运行、未写出或不可推断的字段保持未运行/NA。

## Phase0: v18 可信起点

输出: `Stream3D/outputs/audit/v19_phase0/phase0_reproduction_probe5.*`

| row | AP | AP50 | AP25 | reproduced |
|---|---:|---:|---:|---|
| P0 on S0 | 0.235730 | 0.414306 | 0.537786 | True |
| P0 on S1 | 0.399213 | 0.597171 | 0.742535 | True |
| O38 own | 0.081038 | 0.219225 | 0.492501 | True |
| repair_cmask own | 0.101653 | 0.248464 | 0.494844 | True |
| repair_cmask on S1 | 0.102883 | 0.242779 | 0.576250 | True |
| P_v6compact on S1 | 0.284832 | 0.503962 | 0.671915 | True |

v18 oracle reference:

| row | AP | AP50 | AP25 | node cov | edge cov | min gate |
|---|---:|---:|---:|---:|---:|---|
| bank16 k16 d0.15 | 0.434865 | 0.581342 | 0.664137 | 0.450122 | 0.410739 | False |
| bank16 k8 d0.25 | 0.434865 | 0.581342 | 0.664137 | 0.450122 | 0.414456 | False |
| grid48 k16 d0.15 | 0.396002 | 0.504018 | 0.638514 | 0.454085 | 0.425560 | False |

Phase0 判定: v19 的起点与 v18 结论一致，可进入 failure decomposition / materialization 诊断。

## Phase1/Phase2A: Oracle A/B/C 与 materialization variants

Gate:

- node_gt_label_coverage >= `0.70`
- edge_gt_label_coverage >= `0.60`
- per-GT surfel-covered / covered GT instance ratio >= `0.65`
- covered mesh vertex ratio >= `0.25`
- GT oracle AP/AP50/AP25 >= `0.45/0.60/0.72`

| variant | node cov | edge cov | mesh cov | covered GT inst | Oracle A | Oracle B | Oracle C | Phase2A gate |
|---|---:|---:|---:|---:|---|---|---|---|
| M0 v18 posterior | 0.450122 | 0.410739 | 0.019828 | 0.217783 | 0.531747 / 0.691765 / 0.735294 | 0.434865 / 0.581342 / 0.664137 | 1.000000 / 1.000000 / 1.000000 | False |
| M1 multi-frame hit union | 0.450122 | 0.410739 | 0.019867 | 0.276430 | 0.491969 / 0.696217 / 0.735294 | 0.444403 / 0.647368 / 0.723408 | 1.000000 / 1.000000 / 1.000000 | False |
| M2 dilation r0.03 | 0.450122 | 0.410739 | 0.038123 | 0.283573 | 0.346451 / 0.601698 / 0.711111 | 0.288492 / 0.512341 / 0.656624 | 1.000000 / 1.000000 / 1.000000 | False |
| M3 anchor mask fill | 0.450122 | 0.410739 | 0.032380 | 0.274550 | 0.328369 / 0.590742 / 0.757120 | 0.284319 / 0.503402 / 0.698622 | 1.000000 / 1.000000 / 1.000000 | False |

Oracle interpretation:

- Oracle A 高于 v18 B，说明“按 surfel GT label 分组”在已标注 surfel 上有上界。
- Oracle B 的最佳修复是 M1: `0.444403 / 0.647368 / 0.723408`，AP50/AP25 过阈值，但 AP 差 `0.0056` 未达 `0.45`。
- Oracle C 一直是 `1.0/1.0/1.0`，但它只在极小 covered mesh 子集上成立，不能抵消 coverage gate 失败。
- M2/M3 让 mesh coverage 略升，但 AP 明显下降，说明 naive dilation / mask fill 会引入污染或边界错配。

Failure decomposition aggregate:

| variant | no_surfel_coverage | underfilled | fragmented | export_lost |
|---|---:|---:|---:|---:|
| M0 | 159 | 34 | 8 | 2 |
| M1 | 159 | 34 | 8 | 2 |
| M2 r0.03 | 159 | 35 | 8 | 1 |
| M3 | 159 | 35 | 8 | 1 |

判定: v19 的主要 blocker 是 coverage/materialization，不是 partition solver。

## Blocker Repair 结果

| repair | node cov | edge cov | mesh cov | covered GT inst | Oracle B | gate |
|---|---:|---:|---:|---:|---|---|
| M1 nn radius 0.08 | 0.467236 | 0.429066 | 0.020490 | 0.281693 | 0.412439 / 0.683826 / 0.718042 | False |
| grid48 M1 | 0.454085 | 0.425560 | 0.026224 | 0.288836 | 0.375282 / 0.502732 / 0.690234 | False |
| M2 dilation r0.08 | 0.450122 | 0.410739 | 0.060089 | 0.290716 | 0.141086 / 0.371698 / 0.593569 | False |

修复审计:

1. `nn_radius=0.08`:
   - 提升 node/edge coverage 到 `0.467236/0.429066`，但离 `0.70/0.60` 很远。
   - Oracle B AP 降到 `0.412439`，不是有效修复。

2. grid48:
   - surfel 数增加到 `36864`，raw mesh hit coverage 到 `0.048543`，materialized mesh coverage `0.026224`。
   - node/edge 仍只有 `0.454085/0.425560`，Oracle B 降到 `0.375282/0.502732/0.690234`。
   - 结论: 不是单纯 surfel 密度不足。

3. M2 dilation r0.08:
   - materialized mesh coverage 提升到 `0.060089`，仍远低于 `0.25`。
   - Oracle B 大幅下降到 `0.141086/0.371698/0.593569`。
   - 结论: 大半径空间扩张会带来明显污染，不能作为当前 materialization 修复。

4. 代码/审计修复:
   - 修正 v19 diagnostic manifest 语义，避免 GT oracle 被误标为 method prediction。
   - 修正 M2/M3 coverage 字段，确保 `mesh_vertex_coverage_ratio` 反映 variant materialized coverage。
   - 新增 `test_v19_materialization.py`，覆盖 failure type 和 dilation pure logic。

## Audit

输出:

- `Stream3D/outputs/audit/v19_final/oracle_reportable_scan.*`
- `Stream3D/outputs/audit/v19_final/metric_integrity.*`
- `Stream3D/outputs/audit/v19_logs/audit_*.log`

结果:

- `unittest discover tests`: 36 tests OK。
- reportable scan: 21 configs，全部 diagnostic-only；0 method configs；0 suspicious configs；0 missing manifest；0 missing eval policy；0 `uses_gt_for_prediction`。
- metric integrity: `phase0_pass=True`。

## Phase2B/Phase3/Phase4

- Phase2B TubeCover: 未运行。
- Phase3 Non-GT Signed Edge Evidence: 未运行。
- Phase4 Tube Manifold Partition: 未运行。
- Phase5/Phase7: 未运行。

原因: Phase2A coverage/materialization gate 未过。按计划，不允许启动 non-GT evidence、partition solver 或 method table。

## Insight 与证据链

1. v19 把 v18 的失败进一步拆开了: v18 best edge oracle 不过，不是因为所有 materialized mesh 都无法形成 object。Oracle C 在 covered mesh 子集上为 `1.0/1.0/1.0`，说明已覆盖且按 GT regroup 的局部 mesh 可以被 evaluator 完美接受。

2. 真正短板是覆盖面极小。M1 的 raw UV->depth->mesh hit coverage 只有 `0.038572`，materialized GT-labeled mesh coverage 只有 `0.019867`；covered GT instance ratio 只有 `0.276430`。这解释了为什么 Oracle C 看起来完美但不能推进。

3. surfel GT label coverage 仍停在 v18 水平。M0/M1/M2/M3 的 node/edge coverage 都是 `0.450122/0.410739`；nn radius 0.08 也只有 `0.467236/0.429066`。non-GT classifier 无法弥补未被 label/materialize 的 graph 部分。

4. “更多点”不是自动更好。grid48 增加 surfel 数但 Oracle B 下降；M2 大半径 dilation 增加 mesh coverage 但 AP 明显下降。当前问题不是简单调大 density/radius，而是缺少可靠的 object-support materialization 机制。

5. 当前不能证明 4D TubeCover 或 signed graph partition 思路无效；只能证明在当前 D4RT surfel bank、RGB-D bridge 与 support materialization 下，coverage/materialization gate 未过，不能进入 non-GT solver。

## 必答问题

1. v19 是否得到可报告方法成功: `False`。
2. Phase0 是否复现 v18 起点: `True`。
3. Phase1/Phase2A 是否完成 failure decomposition/materialization comparison: `True`。
4. Coverage/materialization gate 是否通过: `False`。
5. GT oracle gate 是否整体通过: `False`。Oracle C AP 过，但 coverage gate 不过；Oracle B 最佳 M1 AP 未达 `0.45`。
6. 是否运行 Phase2B/Phase3/Phase4: `False`，按 stop rules 停止。
7. 是否有 GT 泄漏到 method result: `No method result was produced`；21 个 v19 config 全部 diagnostic-only / forbidden。
8. 后续方向: 优先修 D4RT surfel generation / RGB-D hit association / ScanNet object-surface coverage，而不是继续调 non-GT edge evidence 或 partition threshold。

## 审计材料

- Phase0: `Stream3D/outputs/audit/v19_phase0`
- Main Phase2A:
  - `Stream3D/outputs/audit/v19_phase2a_M0`
  - `Stream3D/outputs/audit/v19_phase2a_M1`
  - `Stream3D/outputs/audit/v19_phase2a_M2`
  - `Stream3D/outputs/audit/v19_phase2a_M3`
- Repairs:
  - `Stream3D/outputs/audit/v19_phase2a_repair_m1_nn008`
  - `Stream3D/outputs/audit/v19_phase2a_repair_grid48_m1`
  - `Stream3D/outputs/audit/v19_phase2a_repair_m2_r008`
- Final audit: `Stream3D/outputs/audit/v19_final`
- Logs: `Stream3D/outputs/audit/v19_logs`
- Code review packet: `stream4d_v19_code_review_packet_20260609.zip`
- Code review packet sha256: `stream4d_v19_code_review_packet_20260609.sha256`

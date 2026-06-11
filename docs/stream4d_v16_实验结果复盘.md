# Stream4D v16 实验结果复盘

日期: 2026-06-09  
计划文档: `docs/stream4d_v16_decisive_algorithmic_plan_for_codex.md`  
执行日志: `docs/stream4d_v16_执行日志.md`  
结论先行: v16 得到了一个比 v14/v15 更明确的好消息和一个同样明确的 stop gate。好消息是 broad-support C_hybrid slot oracle 从 single candidate `0.256256 / 0.495495 / 0.702512` 提升到 K8 union `0.366996 / 0.634907 / 0.764547`，support 仍有 `52.8088%`，说明“多个 measurements 解释一个 latent object”这条路确实比 measurement-as-object 更有上界。坏消息是 v16 official broad-support gate 仍失败: AP25 未达到 `0.80`；额外 K16/K32 stress repair 也只到 `0.375373 / 0.643887 / 0.769007`。因此 Phase 3/4/5 solver/materialization method 未启动。

## 结果边界

- 所有 AP/AP50/AP25 来自 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt` 或统一 summary JSON。
- `oracle` 使用 GT selection，只是 diagnostic upper bound，不能进入 method table。
- `C_regionlet` K8 slot oracle 很高，但 support 只有 `18.5455%`，不满足 broad-support 要求。
- `C_hybrid` 是本轮最重要 broad-support signal，但 AP25 仍不过线。
- 本轮没有 reportable method result，因此没有四行 method protocol 结果。按 v16 stop condition，不强行启动 solver。

## 审计通过

```text
py_compile: pass
unittest: Ran 30 tests in 1.474s OK
reportable scan:
  num_configs=14
  num_oracle_configs=14
  num_reportable_method_configs=0
  num_diagnostic_only_configs=14
  num_suspicious_configs=0
  num_uses_gt_for_prediction=0
  num_gt_selected_output_and_method_result=0
  num_forbidden_for_method_table_and_method_result=0
  num_alignment_used_for_prediction=0
metric integrity:
  phase0_pass=True
  gt_files_read_by_rescore=False
```

核心审计输出:

- `Stream3D/outputs/audit/v16_phase0/reportable_config_scan_v16_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase0/metric_integrity_v16_probe5.{json,md}`
- `Stream3D/outputs/audit/v16_phase1/three_layer_oracle_matrix_probe5.{json,csv,md}`

## 做了什么修改

1. `Stream3D/tools/diagnose_v15_union_oracle.py`
   - 新增 `--algorithm-name`，使 v16 manifest 明确写 `v16_union_oracle` 等标签。
   - 修正 `--k` 默认行为，避免显式 K 与默认 K 混合。
   - 该修改不改变 greedy union oracle 逻辑，只影响审计标签和实验 K 选择。

2. `Stream3D/tools/summarize_v16_decisive_diagnostics.py`
   - 新增 v16 汇总器，统一读取 v13/v14/v15/v16 输出。
   - 输出三层 oracle/materialization matrix 和 stop gate summary。

3. 复现脚本
   - `Stream3D/scripts/reproduce_v16_phase0_audit_probe5.sh`
   - `Stream3D/scripts/reproduce_v16_phase1_decisive_oracle_probe5.sh`
   - `Stream3D/scripts/reproduce_v16_phase2_measurement_bank_probe5.sh`
   - `Stream3D/scripts/reproduce_v16_phase6_geometry_probe5.sh`

## Phase 1: Candidate Oracle

来自 v13/v14 既有 candidate oracle，纳入 v16 decisive matrix 作为第一层上界。

| primitive | oracle AP | oracle AP50 | oracle AP25 | support% | 结论 |
|---|---:|---:|---:|---:|---|
| C_mask | 0.224691 | 0.453333 | 0.648889 | 60.8842 | broad but weak |
| C_regionlet | 0.338574 | 0.613208 | 0.829643 | 18.5455 | strong but not broad |
| C_surfel | 0.395062 | 0.750000 | 0.993360 | 4.2916 | very tiny |
| C_masklet | 0.183908 | 0.551724 | 0.926056 | 2.1168 | very tiny |
| C_hybrid | 0.256256 | 0.495495 | 0.702512 | 52.8088 | broad but below gate |
| target atom A3 | 0.068627 | 0.117647 | 0.558824 | 3.0295 | weak/tiny export |
| target atom A4 | 0.068627 | 0.117647 | 0.558824 | 3.0469 | weak/tiny export |
| target atom A4 minpts5 | 0.060317 | 0.114286 | 0.542857 | 3.4740 | weak/tiny export |

解读:

1. 单 primitive 仍延续 v14/v15 结论: broad-support 的 C_mask/C_hybrid 不够强，strong oracle 的 C_regionlet/C_surfel support 不够。
2. target atom 的 raw known support 在 v14 提高过，但 exported support 仍只有约 `3%`，作为 object primitive 不成立。

## Phase 1: Slot Oracle

v16 新跑了 K=2/4/8 的 official slot oracle，允许 oracle 从多个 measurements 组成一个 object slot。

| primitive | K | AP | AP50 | AP25 | support% | selected union% | mean best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C_mask | 2 | 0.261716 | 0.524444 | 0.679009 | 60.8842 | 53.1640 | 0.478241 | False |
| C_mask | 4 | 0.275543 | 0.528889 | 0.683460 | 60.8842 | 59.3466 | 0.487736 | False |
| C_mask | 8 | 0.276531 | 0.528889 | 0.687852 | 60.8842 | 60.4630 | 0.489141 | False |
| C_hybrid | 2 | 0.303099 | 0.595873 | 0.751241 | 52.8088 | 41.8660 | 0.540461 | False |
| C_hybrid | 4 | 0.346129 | 0.620470 | 0.760035 | 52.8088 | 47.0744 | 0.566580 | False |
| C_hybrid | 8 | 0.366996 | 0.634907 | 0.764547 | 52.8088 | 50.0378 | 0.577288 | False |
| C_regionlet | 2 | 0.440463 | 0.778924 | 0.967833 | 18.5455 | 11.4649 | 0.661649 | False |
| C_regionlet | 4 | 0.539863 | 0.856832 | 0.978939 | 18.5455 | 14.1117 | 0.714291 | False |
| C_regionlet | 8 | 0.595554 | 0.877258 | 0.979039 | 18.5455 | 16.0752 | 0.737853 | False |

v16 的主要好消息:

- C_hybrid 是 broad support，K8 union oracle 达到 AP50 `0.634907`，明显高于 single C_hybrid AP50 `0.495495`。
- 这说明 v16 plan 的核心表述“latent object explains multiple measurements”有真实诊断上界，不是空想。

v16 的主要失败点:

- C_hybrid K8 AP25 `0.764547` 未达到 v16 gate `0.80`。
- C_regionlet K8 很强，但 support `18.5455%` 低于 broad-support 口径。

## Blocker 修复结果

### C_hybrid K16/K32 Stress

动机: C_hybrid K8 的 AP50 已过线，AP25 接近但未过，因此尝试增加 slot budget。

| variant | K | AP | AP50 | AP25 | support% | selected union% | mean best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C_hybrid stress | 16 | 0.375373 | 0.643887 | 0.769007 | 52.8088 | 51.6155 | 0.580907 | False |
| C_hybrid stress | 32 | 0.377319 | 0.643887 | 0.769007 | 52.8088 | 52.2386 | 0.581851 | False |

结论: 更多 slot budget 有小幅增益，但 AP25 仍只有 `0.769007`。失败不是简单 K 太小。

### C_hybrid min_region_size=50

动机: 检查 `min_region_size=100` 是否过滤了关键小 measurement。

结果: K2/K4/K8 与默认结果相同，best 仍是 `0.366996 / 0.634907 / 0.764547`。说明失败不是 50-100 点候选被过滤造成。

## Phase 2: Measurement Bank

v16 复用 v14 bank16 CropFormer predicted mask bank 重新诊断。

| metric | value | gate |
|---|---:|---|
| num_mask_frames_available | 16.0 | pass |
| num_mask_frames_missing | 0.0 | pass |
| uv_in01_rate | 0.985845 | pass |
| cycle_uv_error_p90 | 3.273727 | pass |
| self_uv_error_p90 | 1.570825 | diagnostic pass |
| unobserved_surfel_ratio | 0.007849 | pass |
| mean_positive_observations_per_surfel | 1.632495 | fail vs 2.5 |
| surfel_positive_observation_rate | 0.992151 | diagnostic high |

解读:

- D4RT/mask-frame density 没有明显坏掉，16/16 frames 可用，unobserved surfels 很低。
- 但每个 surfel 的 mean positive observations 未达到 `2.5`，说明 measurement redundancy 仍不足。
- 这和 Phase1 的现象一致: correspondence/density 不是唯一瓶颈，object-level ownership/slot quality 仍不够。

## Materialization Evidence

v15 owned-mask-region materialization 作为 v16 materialization 证据链纳入 decisive matrix。

| variant | AP | AP50 | AP25 | exported pre% | purity | contamination | mean best IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 component | 0.000132 | 0.000478 | 0.036858 | 5.4552 | 0.585791 | 0.414209 | 0.016472 |
| R0b component radius0.10 | 0.000092 | 0.000475 | 0.036856 | 5.6308 | 0.575856 | 0.424144 | 0.016797 |
| R1 seed_voronoi | 0.000146 | 0.001065 | 0.025048 | 4.5435 | 0.697447 | 0.302553 | 0.016551 |
| R2 boundary_core | 0.000137 | 0.000495 | 0.036890 | 5.3673 | 0.587792 | 0.412208 | 0.016070 |

结论: owned-region direct materialization 还远未能把 dense measurement 转成 object output。R1 purity 较高，但 completeness/export support/AP 都太低。

## Phase 6: D4RT Geometry Diagnostic

| metric | value |
|---|---:|
| num_ok_windows | 5 |
| num_failed_windows | 0 |
| sim3_anchor_count_mean | 431.2 |
| sim3_scale_mean | 0.560101 |
| sim3_scale_min | 0.194515 |
| sim3_scale_max | 0.978872 |
| sim3_residual_median_mean | 0.468208 |
| sim3_residual_p90_mean | 0.859581 |
| sim3_residual_p95_mean | 1.077516 |
| uv_in01_rate_mean | 0.985845 |

解读: D4RT correspondence/geometry 仍是可用信号，但 v16 没有证据支持“直接把 primitive 当 object”能成功。问题更像 object ownership / measurement explanation，而不是纯几何投影坏掉。

## Gate 判定

v16 official broad-support slot gate:

```text
required:
  broad support >= 25%
  slot oracle AP50 >= 0.60
  slot oracle AP25 >= 0.80

best official broad row:
  C_hybrid K8
  AP/AP50/AP25 = 0.366996 / 0.634907 / 0.764547
  support = 52.8088%
  selected union = 50.0378%

pass = False
```

stress repair:

```text
C_hybrid K16
AP/AP50/AP25 = 0.375373 / 0.643887 / 0.769007
pass = False
```

Phase2 bank gate:

```text
uv_in01_rate >= 0.95: pass
cycle_uv_error_p90 <= 5px: pass
unobserved_surfel_ratio <= 0.05: pass
mean_positive_observations_per_surfel >= 2.5: fail (1.632495)
```

最终:

```text
official_broad_slot_gate_pass=False
stress_broad_slot_gate_pass=False
measurement_bank_gate_pass=False
stop_before_solver=True
```

因此:

```text
Phase 3 object-conditioned mask decomposition: not started
Phase 4 global object explanation solver: not started
Phase 5 posterior materialization method: not started
four-row method evaluation: not started, because no reportable method config exists
```

## 主要 Insight

1. v16 的最大正信号是 C_hybrid slot oracle。C_hybrid single oracle AP50 `0.495495`，K8 union AP50 `0.634907`，K16 stress AP50 `0.643887`。这说明 object slot 组合 measurements 的方向确实比继续调 single primitive 更有价值。
2. 但 C_hybrid AP25 卡在 `0.7645-0.7690`，K 增大和 min-region-size 降低都没过 `0.80`。这不是一个简单阈值或预算问题。
3. C_regionlet K8 oracle 很强，`0.595554 / 0.877258 / 0.979039`，但 support 只有 `18.5455%`。它说明局部 regionlet 质量可以很高，但没有解决 broad reconstruction。
4. bank16 已经把 measurement density 做到很满: `16/16` mask frames、`uv_in01_rate=0.985845`、`unobserved_surfel_ratio=0.007849`。然而 mean positive observations/surfel 只有 `1.632495`，object explanation 需要更多可靠正证据或更好的 ownership model。
5. owned-region materialization 仍断裂。最宽 R0b exported pre 也只有 `5.6308%`，AP50 `0.000475`，说明从 2D owned region 回到 object support 的 materialization 仍非常弱。
6. v16 比 v14/v15 更精确地定位了失败层级: 不是 D4RT correspondence 完全没用，也不是 union slot 完全没上界，而是 broad-support slot 的 AP25/完整度和真实 materialization 仍不足。

## 结论

Stream4D v16 在 probe5 上没有得到可报告 method success，也没有启动 solver。最可靠的好消息是: broad-support C_hybrid 的 multi-measurement slot oracle 已经能把 AP50 推过 `0.60`，达到 `0.634907`，stress 到 `0.643887`。这支持 v16 的算法重启方向，即 latent object 应该解释多个 noisy measurements，而不是把单个 mask/regionlet/atom 当 object。

最可靠的负结论是: 这个上界还不够。AP25 最高只有 `0.769007`，低于 v16 `0.80` gate；bank16 的 positive observation redundancy 未达标；owned-region materialization 的 direct AP 接近零。因此继续写 Phase4 global solver 会产生不可靠的 method claim。

下一步如果继续，应集中在两个方向:

1. 把 C_regionlet 的高质量局部 signal 扩成 broad support，而不是继续调 C_hybrid 的 K 或 min-size。
2. 设计真正的 object ownership/materialization，使 C_hybrid slot oracle 的 broad support 能获得 AP25 完整度，而不是仅靠 GT-selected union。

## 审计材料

审计包:

- `stream4d_v16_code_review_packet.zip`
- `stream4d_v16_code_review_packet.sha256`
- `stream4d_v16_filelist.txt`
- `stream4d_v16_ziptest.log`
- `stream4d_v16_git_status.txt`
- `stream4d_v16_git_diff.patch`

核心输出:

- `Stream3D/outputs/audit/v16_phase1/three_layer_oracle_matrix_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase1/c_mask_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase1/c_hybrid_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase1/c_regionlet_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase1/c_hybrid_union_stress_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase1/c_hybrid_union_min50_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase2/measurement_bank_bank16_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase6/d4rt_sim3_residual_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase0/reportable_config_scan_v16_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase0/metric_integrity_v16_probe5.{json,md}`

复现脚本:

- `Stream3D/scripts/reproduce_v16_phase0_audit_probe5.sh`
- `Stream3D/scripts/reproduce_v16_phase1_decisive_oracle_probe5.sh`
- `Stream3D/scripts/reproduce_v16_phase2_measurement_bank_probe5.sh`
- `Stream3D/scripts/reproduce_v16_phase6_geometry_probe5.sh`


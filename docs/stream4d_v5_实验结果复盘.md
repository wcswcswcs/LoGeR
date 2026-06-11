# Stream4D v5 实验结果复盘

日期：2026-06-08（Asia/Singapore）  
计划文件：`docs/stream4d_v5_deep_audit_and_parallel_experiment_plan_for_codex.md`  
执行日志：`docs/stream4d_v5_执行日志.md`  
结果根目录：`Stream3D/outputs/`、`Stream3D/data/evaluation/scannet/`

本复盘只记录真实运行得到的数据。没有运行或没有完整实现的 phase 明确标注为 `not_run` 或 `blocked`。

## 当前结论

```text
v5 本轮没有达成 ScanNet probe5 第一阶段目标。

Phase A 通过：manifest / oracle guard / metric integrity 已补齐并验证。
Phase B 通过：probe5 5/5 场景 96f cache 生成成功，3/5 场景 128f cache 生成成功。
Phase C 部分完成：C0 baseline 复现；observation bank 和 local proposal bank 有真实诊断结果，但未达到 v5 gate。
Phase D blocked：本地 Dynamic Replica 数据缺失，不能报告 official tracking metrics。

Full ScanNet final 未启动，因为 probe5 gate 未通过。
```

v5 第一阶段 gate：

```text
AP   >= 0.32
AP50 >= 0.53
AP25 >= 0.70
#pred <= 300
union in target >= 0.94
```

本轮最接近完整 method 的新增尝试是：

```text
stream4d_v5_localprop_96f_probe5_min2_bestframe_ioc075
AP/AP50/AP25 = 0.11252410871496846 / 0.2853900969507906 / 0.4797727369815028
#pred = 173.20
pre/union = 35.7973%
```

该尝试显著低于 v4.1 current best：

```text
stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
AP/AP50/AP25 = 0.2816154378895367 / 0.4975830336133305 / 0.6902541954477854
#pred = 415.6
```

因此不能进入 full ScanNet final。

## 已完成代码修复审计

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| `Stream3D/tools/prediction_manifest.py` | 新增 manifest helper | 统一记录 `uses_gt`、`support_policy`、`pre_points_policy` 等审计字段 | `py_compile` pass |
| `Stream3D/tools/scan_reportable_configs.py` | 新增 config scanner | 区分 method/oracle/diagnostic/missing-manifest/suspicious | scanner 运行 pass |
| `Stream3D/tools/oracle_candidate_upper_bound.py` | oracle output config 必须包含 `oracle`，写 diagnostic manifest | 防止 GT-read oracle 误入 method 表 | reject test exit=1 |
| `Stream3D/evaluation/evaluate.py` | 默认拒绝 oracle-named eval，新增 `--allow-oracle-eval` | 防止 oracle config 被普通 evaluator 评估 | reject test exit=1 |
| `Stream3D/tools/verify_stream4d_metric_integrity.py` | 加 manifest/oracle/suspicious 统计与 `--require-manifest` | 让 metric audit 覆盖 reportable configs | phase0_pass=True |
| `Stream3D/stream4d/appearance_memory.py` | 新增 `cosine_similarity_01_valid` | missing appearance 不再贡献正 match | unit test pass |
| `Stream3D/stream4d/object_memory_v2.py` | 使用 appearance valid flag | 避免 missing-feature positive match | unit test pass |
| `Stream3D/stream4d/reliable_densifier.py` | 增加 raw/filter/selected/exported 诊断字段 | 让 densify 流程可审计 | `py_compile` pass |
| `Stream3D/tools/d4rt_preflight.py` | 新增 D4RT preflight | 单独验证 ckpt load/build/fake inference | no-copy/local-copy pass |
| `Stream3D/stream4d/d4rt_adapter.py` | 输出 D4RT timing/cache diagnostics；修复 `num_queries` NameError | unblock smoke/cache generation | 2f rerun pass |
| `Stream3D/stream4d/run_scannet.py` | 写 per-window carrier manifest 和 prediction manifest | cache 可审计 | 96f/128f manifests 存在 |
| `Stream3D/tools/export_mask_observation_bank.py` | 写 manifest；summary 文件名增加 scene | 支持 probe5 多场景 observation bank 审计 | 5/5 scenes exported |
| `Stream3D/tools/export_local_proposal_bank.py` | summary 文件名增加 scene | 避免多场景覆盖 summary | 5/5 scenes exported |
| `Stream3D/tools/check_dynamic_replica_env.py` | 增加 `--root` alias | 符合 v5 计划命令 | env checker pass |

最终验证：

```text
py_compile: pass
unit tests: Ran 8 tests in 0.002s, OK
```

## Phase A 复盘：Metric Safety and Code Hygiene

### Reportable config scanner

初始扫描：

```text
num_configs=6
num_configs_missing_manifest=6
num_diagnostic_only_configs=1
num_oracle_configs=1
num_reportable_method_configs=0
num_suspicious_configs=6
num_uses_gt_and_method_result=0
```

修复后扫描非 oracle reportable configs：

```text
num_configs=5
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=5
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

证据：

```text
Stream3D/outputs/audit/v5_reportable_config_scan_initial.json
Stream3D/outputs/audit/v5_reportable_config_scan_method.json
```

注意：

```text
历史 artifact 的 manifest 是 retroactive manifest。
它能防止后续报告时混淆 config 类型，但不能反向证明历史运行当时已经写了 manifest。
```

### oracle guard

正确拒绝测试：

```text
ValueError: --output-config for oracle_candidate_upper_bound must contain 'oracle'. This tool reads GT and any output prediction is diagnostic-only.
ORACLE_GUARD_REJECT_EXIT_STATUS=1
```

evaluator oracle 拒绝测试：

```text
ValueError: Refusing to evaluate oracle-named prediction/TMP/output without --allow-oracle-eval.
EVALUATOR_ORACLE_REJECT_EXIT_STATUS=1
```

证据：

```text
Stream3D/outputs/audit/v5_oracle_guard_reject_test.log
Stream3D/outputs/audit/v5_evaluator_oracle_reject_test.log
```

第一次误用测试说明：

```text
手动测试 config 名为 v5_bad_method_oracle_guard_test，包含 oracle，因此按规则允许写 diagnostic-only oracle artifact。
该产物只作为 GT-read upper-bound diagnostic，不进入任何 method 主表。
```

该 diagnostic aggregate：

```text
mean_best_iou_per_gt=0.614459029012737
mean_num_gt_instances=16.6
mean_num_pred_instances=415.6
mean_num_valid_pred_instances_in_support=58.8
mean_num_oracle_selected=14.4
mean_oracle_selected_iou_ge_0p5=11.6
mean_oracle_selected_iou_ge_0p75=6.8
```

解释：

```text
candidate pool 中确实有不少高 IoU 候选，但只有 GT oracle 才能选出来。
这说明当前主要问题不是完全没有候选，而是无监督 object formation / selection 不能把候选组织成少量正确实例。
该结论只能作为 diagnostic，不是 method 成绩。
```

### Metric integrity

Reportable configs：

```text
evaluator_ap_core_equal_by_hash=True
has_pre_points_load_original=True
has_pre_points_load_current=True
gt_files_read_by_rescore=False
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=5
num_suspicious_configs=0
phase0_pass=True
```

Alignment source configs：

```text
scannet:
  checked=5
  alignment mean_iou=1.0
  alignment min_iou=1.0
  alignment failed=0

stream4d_scannet_32f_ioc075_fixmem:
  checked=5
  alignment mean_iou=1.0
  alignment min_iou=1.0
  alignment failed=0
```

证据：

```text
Stream3D/outputs/audit/v5_metric_integrity_probe5.md
Stream3D/outputs/audit/v5_metric_integrity_probe5.json
Stream3D/outputs/audit/v5_metric_integrity_probe5_alignment_sources.md
Stream3D/outputs/audit/v5_metric_integrity_probe5_alignment_sources.json
```

Phase A 结论：

```text
Phase A 通过。
本轮没有发现非 oracle reportable config 被标记为 uses_gt=true。
oracle 工具和 evaluator 均有硬 guard。
AP core hash 与原版 Stream3D evaluator 的核心函数一致。
```

## Phase B 复盘：D4RT Cache / Multi-window Infrastructure

### D4RT preflight

| Run | ckpt path | torch_load | build_model | fake_encode | fake_decode | 结论 |
|---|---|---:|---:|---:|---:|---|
| no local copy | original ckpt | `40.54081153869629s` | `5.7353949546813965s` | `1.075178623199463s` | `0.22654008865356445s` | pass |
| local copy | `/tmp/stream4d_v5_d4rt_ckpt/opend4rt.ckpt` | `5.030629634857178s` | `5.640040397644043s` | `0.211500883102417s` | `0.07605218887329102s` | pass |

分析：

```text
local checkpoint copy 将 torch.load 从 40.54s 降到 5.03s，说明先前多窗口卡住风险很可能与大 checkpoint 的 filesystem/I/O 有关，而不是 D4RT fake inference 本身不可运行。
```

证据：

```text
Stream3D/outputs/audit/d4rt_preflight_v5_no_copy.md
Stream3D/outputs/audit/d4rt_preflight_v5_local_copy.md
```

### Smoke

| Run | frames | windows | objects | points | hit_rate | total_sec | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| scene0011 2f | `2` | `1` | `30` | `127` | `0.5603448275862069` | `1.0269687175750732` | pass after fix |
| scene0011 32f | `32` | `1` | `208` | `5883` | `0.7142509283496682` | `20.79462480545044` | pass |

修复记录：

```text
第一次 2f smoke 暴露 d4rt_adapter.py 中 num_queries 未定义。
修复后 2f rerun 正常完成，并写出 carriers_window000_manifest.json。
```

### 96f / 128f cache

96f probe5：

| Scene | windows | total_sec | objects | points | hit_rate | uv_in01_mean | visibility_mean | carriers_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | `5` | `100.54125618934631` | `315` | `21396` | `0.6484030216887124` | `0.6294623272742046` | `0.23711047768592836` | `3283.2` |
| scene0030_00 | `5` | `171.30217480659485` | `237` | `47164` | `0.9152797511616297` | `0.5865065958884814` | `0.2819016814231873` | `5684.8` |
| scene0050_00 | `5` | `127.04659223556519` | `243` | `27340` | `0.9491049765179994` | `0.4753550088710122` | `0.2926975518465042` | `4238.4` |
| scene0081_01 | `5` | `138.26458644866943` | `1337` | `16225` | `0.6266542141711748` | `0.681480289576998` | `0.10063766986131668` | `4966.4` |
| scene0591_00 | `5` | `230.90835857391357` | `885` | `40092` | `0.8410287118071549` | `0.659582538768336` | `0.25249509811401366` | `7859.2` |

96f gate：

```text
success_96f=5
gate_96f_5of5=true
```

128f：

| Scene | windows | total_sec | objects | points | hit_rate | uv_in01_mean | visibility_mean | carriers_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | `7` | `145.86376214027405` | `429` | `31008` | `0.6341291365003408` | `0.6097843618093508` | `0.22543639157499587` | `3441.1428571428573` |
| scene0030_00 | `7` | `236.13476848602295` | `263` | `64871` | `0.8826745335828519` | `0.5822063263699236` | `0.28048694133758545` | `5683.428571428572` |
| scene0050_00 | `7` | `166.67303085327148` | `264` | `29992` | `0.9401067478558792` | `0.5435342275892976` | `0.31628135059561047` | `3987.4285714285716` |

128f gate：

```text
success_128f=3
gate_128f_3of5=true
```

Phase B 结论：

```text
Phase B 通过。计划中 Day0/Day1 最大的 D4RT cache blocker 已经明显缓解。
96f probe5 5/5 场景均完成，每个 window 均有 manifest。
128f 已完成 3/5 场景，满足本轮 3/5 sanity 条件。
```

Insight：

```text
D4RT cache 现在不是本轮最主要 blocker。真正的问题已经转移到 object formation：有 carrier、多窗口 local props 和 observation，但简单导出/轻量 proposal 不能形成高质量 instance。
```

证据：

```text
Stream3D/outputs/cache_audit/v5_cache_success_table.csv
Stream3D/outputs/cache_audit/v5_cache_success_summary.json
```

## Phase C 复盘：ScanNet Probe5 Algorithm Core

### C0 fixed baseline rerun

| Config | AP | AP50 | AP25 | pre % | union % | #pred | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `scannet_self_inherit_probe5` | `0.23572958766757215` | `0.41430630093420795` | `0.5377857284834029` | `84.6744` | `84.6744` | `128.2` | reproduced |
| `scannet_on_stream4d_32f_probe5` | `0.3992127932017927` | `0.5971712938711367` | `0.7425353588266108` | `4.5145` | `84.6744` | `128.2` | Stream3D-on-32f diagnostic remains strong |
| `stream4d_32f_self_probe5` | `0.14423832897728478` | `0.28834385159686365` | `0.46471600688468157` | `4.5145` | `4.5145` | `386.0` | weak baseline |
| `stream4d_v3_adapt_recompute_on_32f_probe5` | `0.11446812892596024` | `0.2413345690454124` | `0.4303985171455051` | `4.5145` | `2.9076` | `17.0` | sparse |
| `stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5` | `0.2816154378895367` | `0.4975830336133305` | `0.6902541954477854` | `4.5145` | `4.3406` | `415.6` | current best reproduced |

这些值与计划中记录的 v4.1 baseline 一致到 `<0.002` gate 内。

证据：

```text
Stream3D/data/evaluation/scannet/v5_c0_*_class_agnostic.txt
Stream3D/outputs/audit/v5_metric_integrity_probe5.json
```

分析：

```text
1. Stream3D on Stream4D 32f support 仍然达到 0.399213 / 0.597171 / 0.742535，说明同一 32f support 上并非完全没有足够几何覆盖。
2. Stream4D current best 为 0.281615 / 0.497583 / 0.690254，低于 v5 第一阶段目标 AP>=0.32/AP50>=0.53/AP25>=0.70。
3. 差距主要不是 evaluator 或 manifest 问题，因为 Phase A 已通过，C0 已复现。
```

### C1 Mask Observation Bank

导出结果：

| Scene | raw_unique_observations | exported_observations | dropped_small | union_points | hit_rate | score_mean | score_max |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | `949` | `736` | `213` | `85347` | `0.6858332591333002` | `0.002890825504437089` | `0.04751131311058998` |
| scene0030_00 | `1860` | `1705` | `155` | `153614` | `0.8488804394363615` | `0.00859245378524065` | `0.08983287215232849` |
| scene0050_00 | `1309` | `1280` | `29` | `79341` | `0.9250419823874507` | `0.007641164120286703` | `0.07459677755832672` |
| scene0081_01 | `1893` | `1097` | `796` | `143481` | `0.6092923103602758` | `0.0013570161536335945` | `0.025080906227231026` |
| scene0591_00 | `2657` | `2274` | `383` | `124217` | `0.7088552521370416` | `0.009824611246585846` | `0.1184210553765297` |

Aggregate：

```text
num_scene_summaries=5
total_raw_unique_observations=8668
total_exported_observations=7092
total_dropped_small=1576
mean_hit_rate=0.7555806486908859
```

Scanner：

```text
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=1
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

Raw observation bank 直接作为 prediction 的 AP：

```text
AP/AP50/AP25 = 0.0014692349275190334 / 0.006991623527277158 / 0.029471902888694328
```

Metric integrity：

```text
pre % = 55.2481
union % = 55.2481
#pred = 1418.40
phase0_pass=True
```

解释：

```text
observation bank 成功导出了 5/5 scenes，并且没有 GT/oracle 风险。
但是 raw observation 不能直接当 object instance prediction。
AP 极低且 #pred 极高，说明它只是证据层，不是 proposal/result 层。
```

当前不足：

```text
计划里的 MaskObservation Bank v2 schema 要求 carrier_boundary_distance、projected_3d_centroid/extent、depth_valid_ratio、appearance_valid 等字段。
当前现有 local_props 只提供 frame_id/mask_id/coverage/num_carriers 这一级信息。
因此 C1 是部分完成，不是完整 v2 schema。
```

证据：

```text
Stream3D/outputs/mask_observation_bank_v5/stream4d_v5_obs_bank_96f_probe5_ioc075_aggregate.csv
Stream3D/data/evaluation/scannet/v5_obs_bank_96f_probe5_ioc075_class_agnostic.txt
Stream3D/outputs/audit/v5_metric_integrity_obs_bank_probe5.md
```

### C2 Local Proposal Bank minimal attempt

设置：

```text
debug source = outputs/stream4d_v5_cache_96f_probe5
same_frame_policy = best_per_frame
min_observations = 2
min_frames = 2
support = mask_backproject
```

导出结果：

| Scene | raw_local_proposals | kept_local_proposals | exported_objects | conflict_rate | mask_hit_rate |
|---|---:|---:|---:|---:|---:|
| scene0011_00 | `745` | `119` | `95` | `0.5581523772200153` | `0.7288030877871904` |
| scene0030_00 | `1126` | `302` | `272` | `0.6578927627013746` | `0.8407515770859368` |
| scene0050_00 | `649` | `220` | `144` | `0.6513681089095565` | `0.9173049979133806` |
| scene0081_01 | `2452` | `162` | `135` | `0.5737452915429657` | `0.7024225645623452` |
| scene0591_00 | `2045` | `376` | `220` | `0.6400010446864635` | `0.7062009862292453` |

Aggregate：

```text
num_scene_summaries=5
total_raw_local_proposals=7017.0
total_kept_local_proposals=1179.0
total_exported_objects=866.0
mean_export_conflict_rate=0.6162319170120751
mean_export_mask_hit_rate=0.7790966427156196
```

AP：

```text
AP/AP50/AP25 = 0.11252410871496846 / 0.2853900969507906 / 0.4797727369815028
```

Metric integrity：

```text
pre % = 35.7973
union % = 35.7973
#pred = 173.20
alignment mean/min = 1.0 / 1.0
alignment failed = 0
phase0_pass=True
```

分析：

```text
1. local proposal bank 把 #pred 降到了 173.20，满足 #pred<=300，但 AP/AP50/AP25 明显失败。
2. pre/union 达到 35.7973%，高于 current best 的 4.3406%，但 AP 更差。这说明扩大 support/union 本身不是充分条件。
3. mean conflict rate 约 0.616，说明简单 best-per-frame 和 min2/min2 无法解决 same-frame / support ownership 冲突。
4. 它比 raw observation bank 更接近 object 层，但仍缺计划中的 core/fringe/reject、typed conflict、weak bridge、split quarantine。
```

结论：

```text
该 minimal attempt 没有通过 v5 gate，不能作为 v5 Proposal v2 成功。
它的失败支持计划中的判断：下一步必须实现真正 Boundary-Aware Proposal v2 和 Typed Evidence Graph v2，而不是继续直接导出 local props。
```

证据：

```text
Stream3D/outputs/local_proposal_bank_v5/stream4d_v5_localprop_96f_probe5_min2_bestframe_ioc075_aggregate.csv
Stream3D/data/evaluation/scannet/v5_localprop_96f_probe5_min2_bestframe_ioc075_class_agnostic.txt
Stream3D/outputs/audit/v5_metric_integrity_localprop_probe5.md
```

### C3/C4/C5 状态

```text
C-Exp2 Boundary-Aware Proposal v2: blocked
C-Exp3 Typed Evidence Graph v2: blocked
C-Exp4 Split/Merge-Capable Object Memory: blocked
Full ScanNet final: not_run
```

原因：

```text
当前代码库有 v4.1 相邻组件，例如 local_props、evidence_graph、boundary refine 工具；
但没有完整的 v5 core/fringe/reject proposal 生成、typed edges、pending/split/quarantine lifecycle 实现。

本轮已经做了两个不造假的可执行替代诊断：
1. raw observation bank direct eval
2. local proposal bank min2/bestframe eval

两者均未通过 gate，且 local proposal 的 conflict rate 很高。
```

必须避免的误写：

```text
不能写 v5 Proposal v2 已经完成。
不能写 Typed Evidence Graph v2 已经验证。
不能把 local proposal bank minimal attempt 写成完整 v5 算法。
不能启动 full ScanNet final。
```

Phase C 结论：

```text
Phase C 没有通过 v5 gate。
C0 baseline 复现，说明比较基线稳定。
C1/C2 的真实诊断说明 observation/candidate 足够多，但 object formation 仍失败。
当前主要 blocker 是 v5 算法核心尚未实现到计划要求的粒度，而不是 D4RT cache 或 metric audit。
```

Insight：

```text
Stream4D 的问题不是“没有点”或“没有候选”。observation bank 的 union 可以达到 55.25%，local proposal 的 #pred 可以压到 173.2，但 AP 仍失败。
这说明 object proposal 的质量、冲突处理和一对一 instance assignment 比覆盖率更关键。
```

## Phase D 复盘：Dynamic Replica

环境检查：

```text
data_root_exists=False
split_dir_exists=False
annotation_exists=False
all_required_camera_fields_present=False
usable_scene_count=0
can_report_official_instance_tracking=False
can_report_d4rt_trajectory_metrics=False
can_report_only_qualitative_consistency=False
annotation_error=annotation file missing
```

证据：

```text
Stream3D/outputs/audit/dynamic_replica_env_v5.md
Stream3D/outputs/audit/dynamic_replica_env_v5.json
```

结论：

```text
Phase D blocked by missing local dataset.
没有 RGB/depth/camera/instance/object ID/trajectory 文件可用。
不能报告 official IDF1、MOTA/MOTP、4D IoU、trajectory metrics 或 pseudo consistency。
```

## Final Gate 判断

Probe5 v5 第一阶段 gate：

| 项 | Gate | 本轮最佳可比/新增结果 | 结论 |
|---|---|---:|---|
| AP | `>=0.32` | localprop `0.11252410871496846`; current best `0.2816154378895367` | fail |
| AP50 | `>=0.53` | localprop `0.2853900969507906`; current best `0.4975830336133305` | fail |
| AP25 | `>=0.70` | localprop `0.4797727369815028`; current best `0.6902541954477854` | fail |
| #pred | `<=300` | localprop `173.20`; current best `415.6` | localprop pass, current best fail |
| manifest/metric audit | pass required | pass | pass |
| no oracle config in reportable set | pass required | pass | pass |

最终判断：

```text
v5 本轮没有达到 ScanNet probe5 gate。
不启动 full ScanNet final。
不 claim Stream4D v5 超过 Stream3D。
不 claim dynamic semantic 4D reconstruction/tracking 已验证。
```

## 总分析

```text
1. Phase A 解决了虚假指标风险的核心口子：oracle output/eval 已经有硬 guard，reportable configs 有 manifest，metric integrity 通过。
2. Phase B 解决了 D4RT cache 吞吐 blocker：96f probe5 5/5 完成，128f 3/5 完成，local ckpt copy 显著降低 torch.load 时间。
3. C0 确认 v4.1 current best 仍是 0.281615 / 0.497583 / 0.690254，未达到 v5 第一阶段 gate。
4. C1 observation bank 显示候选和覆盖并不缺，7092 个 observations、pre/union 55.25%，但 raw AP 只有 0.00147。
5. C2 minimal local proposal bank 能把 #pred 压到 173.2，但 AP 只有 0.1125，conflict rate 约 0.616，说明简单 proposal 聚合会损害实例质量。
6. 计划中真正需要的 Boundary-Aware Proposal v2 / Typed Evidence Graph v2 / Split-Merge memory 仍是 blocker，不能用已有 minimal attempt 冒充完成。
7. Dynamic Replica 本地数据缺失，不能造 tracking 指标。
```

## 下一步建议

必须先做，不建议继续 full final：

```text
1. 实现真正 MaskObservation Bank v2 schema：
   carrier_boundary_distance、projected_3d_centroid/extent、depth_valid_ratio、appearance_valid。

2. 实现 Boundary-Aware Proposal v2：
   core/fringe/reject 三类 evidence，记录 fringe/core ratio、conflict point rate、compactness。

3. 实现 Typed Evidence Graph v2：
   positive_track、positive_complement、negative_conflict、negative_ownership、weak_bridge，并输出边类型统计。

4. 在 probe5 fixed 32f support 上重跑 P/G/M 矩阵。

5. 只有当 AP>=0.32、AP50>=0.53、AP25>=0.70、#pred<=300 后，才启动 full ScanNet final。

6. Dynamic Replica 需要先补数据路径，至少使 env checker 返回 usable_scene_count>0。
```

安全表述：

```text
本轮 v5 完成了实验基础设施和可审计性建设，并把 D4RT 多窗口 cache blocker 从主要风险中移除。
但 v5 算法核心还没有过 probe5 gate。当前最重要的科学 blocker 是 object formation，而不是 evaluator、GT 泄漏、D4RT cache 或单纯 coverage。
```


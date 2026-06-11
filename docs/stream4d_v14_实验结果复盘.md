# Stream4D v14 实验结果复盘

日期: 2026-06-09  
计划文档: `docs/stream4d_v14_global_object_explanation_plan_for_codex.md`  
执行日志: `docs/stream4d_v14_执行日志.md`  
结论先行: v14 完成了 Phase 0 审计、Phase 1 failure decomposition、Phase 2 surfel atom primitive、oracle upper bound、Stream3D-on-atom-support attribution、mask-density repair、target-dominant atom-base repair 和 final audit packet，但没有得到可作为正式 method success 的结果。Phase 2 gate 明确失败: 最好的 A3/A4 target-base oracle AP/AP50/AP25 只有 `0.068627 / 0.117647 / 0.558824`，actual pre% 约 `3.03%`；最宽松 minpts5 版本 pre% 也只有 `3.4740%`，远低于 `25%` gate。因此 Phase 3/4 global solver、tune30/final 均未启动。

## 结果边界

- 所有 AP/AP50/AP25 来自 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt` 或统一汇总。
- Oracle 使用 GT selection，只是 diagnostic upper bound，不能进入 method table。
- Failure decomposition 读取 GT，只用于诊断。
- Phase 2 atom candidates 是 diagnostic-only，不是 reportable method result。
- `Stream3D on atom support` 是 diagnostic same-support attribution，不是方法。
- 新增 bank16 masks 来自 CropFormer predicted masks，不是 GT。
- 没有把 `NA`、`nan`、空预测或 oracle 结果改写为有利数字。

## 审计通过

```text
py_compile: pass
pure unittest: Ran 7 tests ... OK
all unittest: Ran 27 tests ... OK
reportable scan: num_configs=61, suspicious=0, uses_gt_for_prediction=0
metric integrity: phase0_pass=True, all_ap_core_equal=True, gt_files_read_by_rescore=False
```

Final scan:

```text
num_diagnostic_only_configs=57
num_oracle_configs=13
num_reportable_method_configs=4
num_suspicious_configs=0
num_uses_gt_for_prediction=0
```

## Phase 0 Baseline

| row | AP | AP50 | AP25 | pre% | conflict | best IoU |
|---|---:|---:|---:|---:|---:|---:|
| P0 Stream3D on S0 | 0.235730 | 0.414306 | 0.537786 | 84.6744 | 0.2213 | 0.6308 |
| P0 Stream3D on S1 | 0.399213 | 0.597171 | 0.742535 | 4.5145 | 0.2213 | 0.7370 |
| B1 own | 0.328439 | 0.629266 | 0.884363 | 3.9861 | 8.4307 | 0.6914 |
| O38 own | 0.081038 | 0.219225 | 0.492501 | 66.6809 | 3.9025 | 0.4611 |
| M13c own | 0.224575 | 0.419119 | 0.781728 | 4.3855 | 55.8958 | 0.6576 |
| M13d own | 0.161109 | 0.427857 | 0.793144 | 2.0522 | 0.0000 | 0.5991 |
| C_surfel own | 0.228316 | 0.460285 | 0.778069 | 4.2916 | 52.7632 | 0.6522 |

继承结论没有改变: v13/v12 的高 AP25 行大多是 tiny-support subset；O38 support broad 但 AP/AP50 不够。

## Phase 1 Failure Decomposition

输出:

- `Stream3D/outputs/audit/v14_failure_decomposition/failure_decomposition_probe5.json`
- `Stream3D/outputs/audit/v14_failure_decomposition/failure_decomposition_probe5.md`
- `Stream3D/outputs/audit/v14_failure_decomposition/failure_decomposition_probe5_source_gt.csv`
- `Stream3D/outputs/audit/v14_failure_decomposition/failure_decomposition_probe5_method_gt.csv`
- `Stream3D/outputs/audit/v14_failure_decomposition/visuals/failure_visuals_manifest.json`

Candidate source diagnostic:

| source | AP | AP50 | AP25 | oracle AP | oracle AP50 | oracle AP25 | support% | best IoU | no/weak/good/high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0.328439 | 0.629266 | 0.884363 | NA | NA | NA | 3.9861 | 0.0260 | 217/12/6/1 |
| O38 | 0.081038 | 0.219225 | 0.492501 | 0.223210 | 0.444444 | 0.635556 | 66.6809 | 0.3000 | 71/35/73/57 |
| C_mask | 0.058281 | 0.161357 | 0.343526 | 0.224691 | 0.453333 | 0.648889 | 60.8842 | 0.2921 | 73/35/75/53 |
| C_regionlet | 0.045679 | 0.122830 | 0.266596 | 0.338574 | 0.613208 | 0.829643 | 18.5455 | 0.0959 | 162/33/37/4 |
| C_surfel | 0.228316 | 0.460285 | 0.778069 | 0.395062 | 0.750000 | 0.993360 | 4.2916 | 0.0265 | 217/10/8/1 |
| C_masklet | 0.062802 | 0.267185 | 0.517357 | 0.183908 | 0.551724 | 0.926056 | 2.1988 | 0.0104 | 228/7/1/0 |
| C_hybrid | 0.023515 | 0.066350 | 0.133871 | 0.256256 | 0.495495 | 0.702512 | 52.8088 | 0.2756 | 67/47/79/43 |
| M13c | 0.224575 | 0.419119 | 0.781728 | NA | NA | NA | 4.3855 | 0.0270 | 217/10/8/1 |
| M13d | 0.161109 | 0.427857 | 0.793144 | NA | NA | NA | 2.0522 | 0.0100 | 228/8/0/0 |

Final method attribution:

| method | support% | pool IoU | method IoU | selected_good | filtered_good | assignment | boundary | weak | no |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M13c | 4.3855 | 0.3231 | 0.0270 | 1 | 52 | 7 | 83 | 37 | 56 |
| M13d | 2.0522 | 0.3231 | 0.0100 | 0 | 56 | 4 | 83 | 37 | 56 |

Pool best source counts:

| source | count |
|---|---:|
| O38 | 416 |
| C_regionlet | 50 |
| M13c | 2 |
| B1 | 4 |

分析:

1. Broad-support candidate pool 中最相关的是 O38/C_mask/C_hybrid，但它们 oracle AP50 只有 `0.4444/0.4533/0.4955`，低于 `0.60`。
2. Tiny primitive 的 oracle 可以很高，例如 C_surfel oracle `0.395062/0.750000/0.993360`，但 support 只有 `4.2916%`。
3. M13c/M13d final method 与 pool best IoU 的差距很大。M13c pool IoU `0.3231`，method IoU `0.0270`；M13d method IoU `0.0100`。
4. M13c 有 `52` 个 filtered_good，M13d 有 `56` 个 filtered_good，说明 final selection/export 没保留可解释候选。
5. 但 broad-support oracle 本身仍不够强，因此不能只把失败归咎于 solver。

## Phase 2 Default Atom Results

输出:

- `Stream3D/outputs/v14_surfel_atom_bank/A*/`
- `Stream3D/outputs/audit/v14_atom_oracle/atom_oracle_matrix_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v14_atom_support/cross_prepoints_audit.{json,csv,md}`

| row | candidate AP/AP50/AP25 | oracle AP/AP50/AP25 | Stream3D-on-support | pre% | atom known% | best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 | 0.055958/0.259918/0.621966 | 0.120915/0.411765/0.852941 | 0.316840/0.445312/0.693015 | 2.2934 | 12.0898 | 0.459438 | False |
| A1 | 0.000288/0.001294/0.089091 | 0.015326/0.068966/0.482759 | 0.376710/0.591133/0.722291 | 2.0994 | 11.4001 | 0.243462 | False |
| A2 | 0.000000/0.000000/0.032591 | 0.000000/0.000000/0.200000 | 0.472356/0.772800/0.827200 | 1.7472 | 9.6252 | 0.102932 | False |
| A3 | 0.000000/0.000000/0.077154 | 0.000000/0.000000/0.200000 | 0.496667/0.843333/0.843333 | 1.6450 | 9.0344 | 0.104866 | False |
| A4 | 0.000000/0.000000/0.077154 | 0.000000/0.000000/0.200000 | 0.473067/0.827200/0.827200 | 1.7203 | 9.5300 | 0.102010 | False |

默认结果的关键事实:

- A0 oracle AP50 `0.411765`，但 actual pre% 只有 `2.2934%`，且 A0 不是计划要求的 A3/A4 broad-support primitive。
- A3/A4 oracle AP50 均为 `0.000000`，actual pre% 约 `1.65-1.72%`。
- Stream3D-on-A3/A4 support 很高，AP50 分别 `0.843333/0.827200`，说明这些 tiny support 上 Stream3D baseline 能分类/切分，但 atom-as-object 本身很差。

## Blocker 修复与结果

### Unknown Merge Repair

动机: default atom export 只输出 source-known atoms，support 太小。尝试导出 unknown atoms，但只作为 diagnostic candidate。

过宽尝试:

```text
min_surfels=1, min_export_surfels=1, fringe_from_neighbors, min_export_points=5
```

结果: 导出超过 3 分钟未完成，终止，记录为 atom/export 规模 blocker。

保守 unknown merge:

| row | candidate | oracle | Stream3D-on-support | pre% | atom known% | best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| A4 unknown merge | 0.000132/0.001190/0.140602 | 0.003175/0.028571/0.400000 | 0.306765/0.450000/0.700340 | 3.1863 | 10.2649 | 0.212128 | False |

结论: unknown export 将 actual pre% 从 `1.7203%` 提到 `3.1863%`，best IoU 从 `0.102010` 提到 `0.212128`，但 oracle AP50 仍只有 `0.028571`。

### Bank16 Mask Density Repair

发现: v12 measurement bank 的 D4RT local frames 为 `0..15`，但每个 scene 只有 `0.png` 和 `10.png` 两帧有 CropFormer mask。本轮用可用 CropFormer 环境生成缺失 `1..9,11..15`，共 70 张 predicted masks，复制到 ScanNet processed cache，并重建:

- `Stream3D/outputs/v14_measurement_bank_bank16_cropformer/`
- `Stream3D/outputs/audit/v14_measurement_bank_bank16/measurement_bank_probe5.{json,csv,md}`
- copied filelist: `Stream3D/outputs/audit/v14_cropformer_bank16_missing_copied_filelist.txt`

Density 对比:

| bank | mask frames avail | missing | unobserved surfel | target positive samples | positive obs rate | negative obs rate | ambiguous |
|---|---:|---:|---:|---:|---:|---:|---:|
| v12 | 2.0 | 14.0 | 0.145764 | 26461.8 | 0.854236 | 0.001477 | 0.012196 |
| v14 bank16 | 16.0 | 0.0 | 0.007849 | 213839.6 | 0.992151 | 0.005811 | 0.047914 |

bank16 source-base result:

| row | candidate | oracle | Stream3D-on-support | pre% | atom known% | best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| bank16 A0 | 0.056986/0.262359/0.627404 | 0.120915/0.411765/0.852941 | 0.316840/0.445312/0.693015 | 2.2936 | 12.0898 | 0.458642 | False |
| bank16 A3 | 0.000000/0.000000/0.057673 | 0.000000/0.000000/0.240000 | 0.491204/0.843333/0.843333 | 1.6761 | 9.0515 | 0.124489 | False |
| bank16 A4 | 0.000000/0.000000/0.057673 | 0.000000/0.000000/0.240000 | 0.473067/0.827200/0.827200 | 1.7172 | 9.5251 | 0.123183 | False |
| bank16 A4 unknown merge | 0.000257/0.001157/0.152601 | 0.006173/0.027778/0.416667 | 0.323428/0.461886/0.707364 | 3.2367 | 10.2539 | 0.213122 | False |

结论: mask density 修复真实降低了 unobserved ratio，但 source-base atom 没利用新增 target masks，因为 atom birth 仍依赖旧 carrier `src_mask_id`。因此 raw source-known support 基本不变。

### Target-Dominant Atom Base Repair

修改: 新增 `base_mode=target_dominant`，每个 surfel 用第一个 positive predicted target mask `(frame_id, mask_id)` 作为 atom birth key；不读 GT。

结果:

| row | candidate | oracle | Stream3D-on-support | pre% | atom known% | best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| bank16 target A3 | 0.002005/0.004992/0.165599 | 0.068627/0.117647/0.558824 | 0.342857/0.494118/0.741176 | 3.0295 | 86.2830 | 0.303492 | False |
| bank16 target A4 | 0.002005/0.004992/0.165599 | 0.068627/0.117647/0.558824 | 0.336866/0.484102/0.760731 | 3.0469 | 88.8538 | 0.302721 | False |
| bank16 target A4 minpts5 | 0.001693/0.004850/0.149635 | 0.060317/0.114286/0.542857 | 0.313843/0.475083/0.727575 | 3.4740 | 88.8538 | 0.276845 | False |

过宽 target-loose:

```text
min_surfels=1, min_export_surfels=1, fringe_from_neighbors, min_export_points=5
```

结果: 2 分 35 秒未完成，终止，记录为 export-scale blocker。

结论:

- target-base repair 是本轮最有信息量的正向修复: raw atom-known support 提升到 `86-89%`，best IoU 提升到约 `0.303`。
- 但 actual exported pre% 仍只有 `3.03-3.47%`，oracle AP50 最高 `0.117647`，远低于 `0.60` gate。
- 降低 min exported points 到 5 只把 pre% 从 `3.0469%` 提到 `3.4740%`，oracle AP50 反而略低。

## Phase 2 Gate

Gate 要求:

```text
A3/A4 broad-support oracle AP50 >= 0.60
A3/A4 broad-support oracle AP25 >= 0.78
actual pre% >= 25%
best IoU >= C_hybrid + 0.08 = 0.355611
conflict <= C_hybrid unsup conflict
```

实际最好:

| criterion | best observed | pass |
|---|---:|---|
| A3/A4 oracle AP50 | 0.117647 | False |
| A3/A4 oracle AP25 | 0.558824 | False |
| actual pre% | 3.4740 | False |
| best IoU | 0.303492 | False |
| any Phase 2 gate pass | False | False |

因此:

```text
Phase 3 atom measurement bank: not started
Phase 4 global object solver: not started
Phase 5 posterior support method: not started
tune30/final: not started
```

## 主要 Insight

1. v14 的失败不是简单 evaluator 问题。Final scan 和 metric integrity 均通过，oracle/candidate/cross-support manifest 也正确区分 diagnostic-only。
2. Failure decomposition 证实 v13 的同一个结构性问题仍在: tiny primitive oracle 可以很高，broad primitive oracle 不够。C_surfel oracle 高但 support 只有 `4.2916%`；C_hybrid support `52.8088%` 但 oracle AP50 只有 `0.495495`。
3. 默认 A3/A4 atom split 更纯但更碎，AP/AP50 近零。Stream3D-on-A3/A4 support 高，说明 support subset 里有可评估对象信息，但 atom-as-object assignment 不行。
4. 生成 bank16 masks 是有效 density 修复: mask frames 从 `2/16` 到 `16/16`，unobserved surfel ratio 从 `0.145764` 降到 `0.007849`。但它没有自动改善 source-base atom，因为 carrier `src_mask_id` 仍来自旧 source masks。
5. `target_dominant` 是合理但仍不足的修复。它把 raw atom-known support 提到 `86-89%`，best IoU 提到 `0.303`，但 actual exported pre% 仍只有 `3%` 左右，oracle AP50 只有 `0.118`。这说明当前 posterior export / atom support materialization 仍过窄，而且 target-mask atom 仍不能形成完整 object。
6. 过宽修复两次触发导出规模 blocker，说明不能靠 min1/fringe/unknown 粗暴扩张 support；需要更强的 atom merging/object-level set packing，而不是输出阶段硬撑。
7. Phase 2 gate 不过时强行进入 Phase 3/4 会制造虚假的 solver 结果。当前最诚实的结论是 primitive/inference formulation 仍没站稳。

## 结论

Stream4D v14 在 probe5 上没有获得可报告的正式方法成功。最强 Phase 2 oracle AP50 是 default A0 的 `0.411765`，但它 actual pre% 只有 `2.2934%`，且不是 A3/A4 broad-support primitive。最有价值的修复是 bank16 + target-dominant atom base: raw atom-known support `88.8538%`，best IoU `0.302721`，但 oracle `0.068627 / 0.117647 / 0.558824`、actual pre% `3.0469%`，仍远低于 gate。

本轮最可靠的正结论是: 更多 2D mask frames 确实修复了 measurement density，target-base atom 能利用这些 masks 提高 raw support 和 per-GT best IoU。最可靠的负结论是: 当前 atom primitive/export 仍无法把 dense predicted mask observations 转成 broad object support；A3/A4 broad-support oracle 上界不够，不能启动 global solver method claim。

下一步若继续，应优先研究 atom-to-object 的 support materialization 和 object-level split/merge/set packing，而不是继续调 A0-A4 的局部 split 阈值。尤其要解决 raw atom support 高但 actual exported pre% 只有约 `3%` 的断裂。

## 审计材料

审计包:

- `stream4d_v14_probe5_code_review_packet.zip`
- SHA256: 见 `stream4d_v14_probe5_code_review_packet.sha256`
- filelist: `stream4d_v14_probe5_filelist.txt`
- git diff: `stream4d_v14_probe5_git_diff.patch`
- git status: `stream4d_v14_probe5_git_status.txt`
- zip test: `stream4d_v14_probe5_ziptest.log`

核心输出:

- `Stream3D/outputs/audit/v14_phase0/baseline_matrix_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v14_failure_decomposition/failure_decomposition_probe5.{json,md}`
- `Stream3D/outputs/audit/v14_atom_oracle/atom_oracle_matrix_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v14_phase2_summary/phase2_atom_repair_matrix_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v14_final/reportable_config_scan_v14_final.{json,csv,md}`
- `Stream3D/outputs/audit/v14_final/metric_integrity_v14_final.{json,md}`

复现脚本:

- `Stream3D/scripts/reproduce_v14_phase0_probe5.sh`
- `Stream3D/scripts/reproduce_v14_phase1_failure_decomposition_probe5.sh`
- `Stream3D/scripts/reproduce_v14_phase2_atoms_probe5.sh`
- `Stream3D/scripts/reproduce_v14_final_audit_probe5.sh`

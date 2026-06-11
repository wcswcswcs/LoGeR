# Stream4D v13 实验结果复盘

日期: 2026-06-09  
计划文档: `docs/stream4d_v13_object_explanation_reboot_experiment_plan_for_codex.md`  
执行日志: `docs/stream4d_v13_执行日志.md`  
结论先行: v13 完成了 video masklet density、candidate/oracle attribution、MDL object explanation、posterior support export、WTA repair、geometry diagnostic 和 unified evaluation 的 probe5 闭环，但没有得到可作为正式 method success 的结果。最强 method-result 是 M13c/M13d 的不同折中: M13c own AP/AP50/AP25 为 `0.224575 / 0.419119 / 0.781728`，M13d WTA own 为 `0.161109 / 0.427857 / 0.793144` 且 conflict 为 `0.0000%`。二者 S0/S1 cross-support 都远低于 gate，因此 tune30/final 未启动。

## 结果边界

- 所有 AP/AP50/AP25 来自 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt` 或统一评估汇总。
- Oracle 使用 GT selection，只能作为 diagnostic upper bound，不能进入 method table。
- Cross-support 行是 diagnostic-only，不写成方法成功。
- 没有把 `NA`、`nan`、空预测或 oracle 结果改写成有利数字。
- v13 所有 reportable method config 的 final scan 均无 GT prediction。

## v12 继承基准

| row | AP | AP50 | AP25 |
|---|---:|---:|---:|
| v12 Stream3D on S0 | 0.235730 | 0.414306 | 0.537786 |
| v12 Stream3D on S1 | 0.399213 | 0.597171 | 0.742535 |
| v12 best M5r3 own | 0.237230 | 0.455495 | 0.729781 |
| v12 M5r3 on S0 | 0.002042 | 0.005558 | 0.012047 |
| v12 M5r3 on S1 | 0.022346 | 0.046761 | 0.156118 |

v13 仍必须同时看 own 与 S0/S1；tiny support own AP 不能代表完整 reconstruction 成功。

## Phase A Code Audit

代码修复:

- `Stream3D/stream4d/export_scannet.py` lazy import `open3d`，避免 optional dependency 阻塞 pure-python 测试。
- 新增 `posterior_support` export mode，按 posterior core/fringe surfels 导出支持点，unknown/reject 不导出。
- M13d 修复中给 posterior support export 加入 WTA point assignment。
- 新增 video masklet、MDL object explanation、masklet candidate、candidate selection、geometry diagnostic、failure visual panel 工具。
- 新增 5 个测试文件，覆盖 measurement bank、posterior negative evidence、manifest/eval policy、open3d optional、GPU optional。

审计结果:

```text
py_compile: pass
unittest discover tests: Ran 27 tests in 1.416s ... OK
```

## Phase C Video Masklet Density

输出:

- `Stream3D/outputs/audit/v13_masklet_density/masklet_density_probe5.json`
- `Stream3D/outputs/audit/v13_masklet_density/masklet_density_probe5.csv`
- `Stream3D/outputs/audit/v13_masklet_density/masklet_density_probe5.md`
- `Stream3D/outputs/v13_masklet_measurements/C{1,2,3}/*/masklets.npz`

Aggregate:

| mode | semantic frames/surfel | obs/surfel | unobserved | masklets | frames/birth | agreement | neg outside |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 1.6151 | 1.6151 | 0.1458 | 0.00 | 0.0000 | 0.0000 | 0.0000 |
| C1 | 1.5619 | 1.5619 | 0.8832 | 247.80 | 14.4660 | 0.9925 | 0.0014 |
| C2 | 1.5619 | 1.5619 | 0.8832 | 247.80 | 14.4660 | 0.9925 | 0.0014 |
| C3 | 1.5619 | 1.5619 | 0.8832 | 247.80 | 14.4660 | 0.9925 | 0.0014 |

Per scene C3:

| scene | semantic frames/surfel | unobserved | masklets | frames/birth | compactness | area growth | agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 1.9053 | 0.8790 | 240 | 16.0000 | 0.6402 | 0.9685 | 0.9913 |
| scene0011_00 | 1.1758 | 0.8806 | 180 | 11.2500 | 0.6217 | 0.6292 | 0.9914 |
| scene0030_00 | 1.9818 | 0.8760 | 224 | 16.0000 | 0.6155 | 0.9992 | 0.9952 |
| scene0081_01 | 1.4385 | 0.8793 | 198 | 13.2000 | 0.5882 | 0.6924 | 0.9920 |
| scene0591_00 | 1.3080 | 0.9013 | 397 | 15.8800 | 0.7290 | 0.8971 | 0.9926 |

分析:

- C1/C2/C3 的 masklet 数量和轨迹一致，说明当前 gate 没有产生更细分的有效集合。
- Masklet 的 `frames/birth=14.4660` 证明 temporal extension 发生了，但它主要是在 birth surfels 内扩展，不是把未观测 surfels 变成可解释 object support。
- C3 unobserved `0.8832` 明显差于 C0 `0.1458`，所以 density reboot 没有解决 broad coverage。

## Phase B Candidate/Oracle Attribution

输出:

- `Stream3D/outputs/audit/v13_candidate_attribution/candidate_unsup_matrix_probe5.json`
- `Stream3D/outputs/audit/v13_candidate_attribution/candidate_oracle_matrix_probe5.json`
- `Stream3D/outputs/audit/v13_candidate_attribution/stream4d_v13_oracle_*_upper_bound.{json,csv,md}`

Unsupervised candidate matrix:

| row | AP | AP50 | AP25 | pre% | conflict | best IoU |
|---|---:|---:|---:|---:|---:|---:|
| C_mask unsup own | 0.058281 | 0.161357 | 0.343526 | 60.8842 | 6.0461 | 0.4658 |
| C_regionlet unsup own | 0.045679 | 0.122830 | 0.266596 | 18.5455 | 22.2760 | 0.6184 |
| C_surfel unsup own | 0.228316 | 0.460285 | 0.778069 | 4.2916 | 52.7632 | 0.6522 |
| C_masklet unsup own | 0.062802 | 0.267185 | 0.517357 | 2.1988 | 30.8936 | 0.5295 |
| C_hybrid unsup own | 0.023515 | 0.066350 | 0.133871 | 52.8088 | 15.2632 | 0.5182 |

Same-support Stream3D comparison:

| support | Stream3D AP/AP50/AP25 | method AP/AP50/AP25 | gap AP |
|---|---:|---:|---:|
| C_mask | 0.224237 / 0.401511 / 0.577226 | 0.058281 / 0.161357 / 0.343526 | -0.165957 |
| C_regionlet | 0.400714 / 0.568847 / 0.705466 | 0.045679 / 0.122830 / 0.266596 | -0.355035 |
| C_surfel | 0.358958 / 0.549096 / 0.740956 | 0.228316 / 0.460285 / 0.778069 | -0.130642 |
| C_masklet | 0.348380 / 0.520833 / 0.716146 | 0.062802 / 0.267185 / 0.517357 | -0.285578 |
| C_hybrid | 0.252842 / 0.418829 / 0.603208 | 0.023515 / 0.066350 / 0.133871 | -0.229327 |

Oracle upper bound:

| oracle | AP | AP50 | AP25 | pre% | union% | conflict | best IoU | method table |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C_mask oracle | 0.224691 | 0.453333 | 0.648889 | 60.8842 | 46.3060 | 2.3829 | 0.4613 | False |
| C_regionlet oracle | 0.338574 | 0.613208 | 0.829643 | 18.5455 | 7.2183 | 0.0965 | 0.6060 | False |
| C_surfel oracle | 0.395062 | 0.750000 | 0.993360 | 4.2916 | 3.7916 | 12.9326 | 0.6522 | False |
| C_masklet oracle | 0.183908 | 0.551724 | 0.926056 | 2.1168 | 1.4432 | 3.4172 | 0.5453 | False |
| C_hybrid oracle | 0.256256 | 0.495495 | 0.702512 | 52.8088 | 32.5679 | 1.6555 | 0.5107 | False |

分析:

- `C_surfel` 仍是最强 tiny-support primitive，own AP25 已到 `0.778069`，但 support 只有 `4.2916%`。
- `C_masklet oracle` 的 AP50/AP25 较高，但 support 只有 `2.1168%`，不能证明 broad reconstruction。
- `C_hybrid oracle` 是最相关的 broad-support upper bound，但 `0.256256 / 0.495495 / 0.702512` 不够强，且和 v12 C_hybrid 基本同一档。
- Same-support 下 Stream3D 通常明显强于 unsupervised candidates，说明问题不是 evaluator，而是 object assignment / candidate selection。

## Phase D/E Object MDL + Posterior Support

输出:

- `Stream3D/outputs/audit/v13_object_explanation_mdl/object_mdl_matrix_probe5.json`
- `Stream3D/outputs/v13_object_explanation_mdl/stream4d_v13_m13*_summary.{json,csv,md}`
- `Stream3D/outputs/audit/v13_object_explanation_mdl/reportable_config_scan_v13_mdl_probe5.{json,csv,md}`

Method-result matrix:

| row | AP | AP50 | AP25 | pre% | conflict | best IoU | method table |
|---|---:|---:|---:|---:|---:|---:|---|
| M13a MDL C3 posterior own | 0.145332 | 0.417796 | 0.698669 | 2.0784 | 26.4328 | 0.5875 | True |
| M13b strict posterior own | 0.159829 | 0.400813 | 0.678327 | 2.0401 | 26.2442 | 0.5963 | True |
| M13c full-mask ablation own | 0.224575 | 0.419119 | 0.781728 | 4.3855 | 55.8958 | 0.6576 | True |
| M13d posterior WTA repair own | 0.161109 | 0.427857 | 0.793144 | 2.0522 | 0.0000 | 0.5991 | True |

Cross-support:

| row | AP | AP50 | AP25 | pre% | conflict | best IoU |
|---|---:|---:|---:|---:|---:|---:|
| M13a on S0 | 0.000000 | 0.000000 | 0.000700 | 84.6744 | 26.4328 | 0.0121 |
| M13a on S1 | 0.000164 | 0.001192 | 0.022931 | 4.5145 | 26.4328 | 0.0629 |
| M13b on S0 | 0.000000 | 0.000000 | 0.000678 | 84.6744 | 26.2442 | 0.0121 |
| M13b on S1 | 0.000157 | 0.001136 | 0.033086 | 4.5145 | 26.2442 | 0.0629 |
| M13c on S0 | 0.002030 | 0.005451 | 0.010602 | 84.6744 | 55.8958 | 0.0318 |
| M13c on S1 | 0.022794 | 0.047972 | 0.149611 | 4.5145 | 55.8958 | 0.1784 |
| M13d on S0 | 0.000000 | 0.000000 | 0.000766 | 84.6744 | 0.0000 | 0.0119 |
| M13d on S1 | 0.000084 | 0.000377 | 0.023845 | 4.5145 | 0.0000 | 0.0616 |

Same-support Stream3D gap:

| row | Stream3D on same support | method own | method - Stream3D AP |
|---|---:|---:|---:|
| M13a | 0.335665 / 0.539506 / 0.709877 | 0.145332 / 0.417796 / 0.698669 | -0.190333 |
| M13b | 0.360494 / 0.580247 / 0.709877 | 0.159829 / 0.400813 / 0.678327 | -0.200665 |
| M13c | 0.338269 / 0.511111 / 0.694444 | 0.224575 / 0.419119 / 0.781728 | -0.113695 |
| M13d | 0.355830 / 0.580247 / 0.709877 | 0.161109 / 0.427857 / 0.793144 | -0.194721 |

修复记录:

- Blocker: posterior support M13a/M13b conflict 约 `26%`，full-mask M13c conflict `55.8958%`，且 cross-support 接近零。
- 修改: 给 posterior support export 加 WTA point assignment，生成 M13d。
- 修复结果: M13d conflict 降到 `0.0000%`，AP25 提升到 `0.793144`，但 AP `0.161109`、AP50 `0.427857`、S0/S1 仍失败。

分析:

- WTA 有效解决点级 overlap，不是无效修复。
- 但 WTA 后 support 仍只有 `2.0522%`，说明核心失败不是 overlap，而是可解释 object support 过小。
- M13c full-mask 能恢复到 v12 近似 support `4.3855%`，但 AP50 只有 `0.419119` 且 conflict 高，不能作为方法成功。
- Same-support Stream3D 仍明显强于 M13a/b/d，说明即使在同一 support 上，slot object assignment 也弱。

## Phase F Geometry Diagnostic

输出:

- `Stream3D/outputs/audit/v13_geometry_diagnostic/d4rt_sim3_residual_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v13_geometry_diagnostic/geometry_diagnostic_probe5.{json,md}`

Geometry rows:

| row | AP | AP50 | AP25 | on S0 | on S1 | residual med/p90 | spacing q50 | exported pts | conflict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.201139 | 0.344654 | 0.502268 | NA | NA | NA | NA | NA | NA |
| stream4d_v10_g1 | NA | NA | NA | 0.000000/0.000000/0.000000 | 0.000000/0.000000/0.000000 | NA | 0.0160 | 15.8000 | 14.1772 |
| stream4d_v10_g2 | 0.188825 | 0.364384 | 0.486341 | 0.000000/0.000000/0.000112 | 0.000000/0.000000/0.004523 | 0.0786/0.1350 | 0.0077 | 3065.2000 | 75.1744 |
| stream4d_v10_g3 | 0.173962 | 0.334164 | 0.461727 | 0.000000/0.000000/0.000112 | 0.000000/0.000000/0.004523 | 0.0786/0.1350 | 0.0077 | 3070.4000 | 74.1746 |
| stream4d_v10_g4 | 0.188825 | 0.364384 | 0.486341 | 0.000000/0.000000/0.000112 | 0.000000/0.000000/0.004523 | 0.0786/0.1350 | 0.0077 | 3065.2000 | 75.1744 |
| stream4d_v10_g5 | 0.173962 | 0.334164 | 0.461727 | 0.000000/0.000000/0.000112 | 0.000000/0.000000/0.004523 | 0.0786/0.1350 | 0.0077 | 3070.4000 | 74.1746 |

Sim3 residual diagnostic:

| metric | value |
|---|---:|
| num_windows | 5 |
| num_ok_windows | 5 |
| sim3_anchor_count_mean | 431.2 |
| sim3_scale_mean | 0.5601005412377911 |
| sim3_residual_median_mean | 0.46820795089862405 |
| sim3_residual_p90_mean | 0.8595805074130748 |
| sim3_residual_p95_mean | 1.0775159603094677 |
| visibility_mean_mean | 0.9998548150062561 |
| confidence_mean_mean | 0.9999313950538635 |
| uv_in01_rate_mean | 0.9858451843261719 |

分析:

- D4RT carrier 的 uv/visibility/confidence 仍高，作为 temporal/source evidence 可用。
- 但 direct D4RT geometry 或 Sim3 replacement 的 own/cross 指标不强，且 conflict 高。
- 新 residual diagnostic 使用 same-pixel anchors，残差 med/p90 `0.4682/0.8596`；geometry summary 中旧 v10 g2/g3 med/p90 为 `0.0786/0.1350`。两者定义不同，不能混为一个结论。
- v13 结论应是: D4RT 更适合作为 correspondence/identity evidence，不适合作为当前 pipeline 的直接 metric geometry 替换。

## Visualizations

输出:

- `Stream3D/outputs/audit/v13_visuals/v13_failure_panel_00.png` 到 `v13_failure_panel_19.png`
- 对应 JSON sidecar。
- `Stream3D/outputs/audit/v13_visuals/v13_failure_visuals_manifest.json`

说明:

- 共 20 个 PNG + 20 个 JSON + 1 个 manifest。
- 它们是 metric/failure panel，不是完整 3D overlay；复盘中只作为快速审计辅助。

## Gate 判断

| lane | gate | result | evidence |
|---|---|---|---|
| Phase A tests | py_compile + unittest | pass | 27 tests OK |
| Reportable scan | no GT prediction, manifest/eval policy pass | pass | 9 method configs, suspicious=0 |
| Metric integrity | phase0/evaluator integrity | pass | phase0_pass=True, evaluator hash equal |
| Phase C density | masklet should improve useful coverage | fail | C3 unobserved=0.8832 vs C0=0.1458 |
| Phase B broad oracle | broad-support upper bound competitive | fail | C_hybrid oracle AP=0.2563, AP50=0.4955 |
| M13 own | AP>=0.30, AP50>=0.55, AP25>=0.75 | fail | best AP M13c=0.2246; best AP50 M13d=0.4279 |
| M13 S0 | AP>=0.08, AP50>=0.18, AP25>=0.45 | fail | best M13c on S0=0.0020/0.0055/0.0106 |
| M13 S1 | AP>=0.18, AP50>=0.35, AP25>=0.60 | fail | best M13c on S1=0.0228/0.0480/0.1496 |
| Stream3D-on-M gap | method should be close to same-support Stream3D | fail | M13d AP gap=-0.1947 |
| Repair attempts | fix conflict and gate | partial/fail | WTA conflict 0, but AP/cross fail |
| tune30/final | only if probe5 gate passes | not started | gate failed |

## 主要 Insight

1. v13 reboot 闭环是可执行的，但没有产生正式方法成功。新增 masklet、MDL、posterior support 和 WTA repair 都能跑完统一评估，但 gate 全面失败。
2. Masklet temporal extension 没有转化为 broad object coverage。C3 frames/birth 很高，但 unobserved surfel ratio `0.8832`，说明只是延长了已 birth 的局部 evidence。
3. v13 的 oracle 仍暴露同一个上界分裂: tiny primitive 可以高 AP25，broad hybrid upper bound 不强。C_surfel oracle 高，但 tiny；C_hybrid oracle broad 但 AP50 不够。
4. WTA 是合理修复，但只修 overlap，不修 coverage。M13d conflict 从约 26% 降到 0%，AP25 达 `0.793144`，但 own AP/AP50 和 cross-support 仍失败。
5. Full-mask ablation 说明扩大 support 会带来 v12 类行为: M13c support 到 `4.3855%`，AP25 `0.781728`，但 conflict `55.8958%`、AP50 `0.419119`。
6. Same-support Stream3D 持续强于 M13，说明失败不只是 support 选择，也是 object assignment/set packing 不够好。
7. D4RT temporal carrier 质量仍可用，但 metric geometry replacement 不是当前有效突破口。

## 结论

Stream4D v13 在 probe5 上没有获得可报告的 SOTA 或超过 Stream3D 的方法结果。最强 own AP 是 M13c full-mask ablation `0.224575 / 0.419119 / 0.781728`，但它 conflict 高且 S0/S1 崩溃。最干净的修复是 M13d posterior WTA `0.161109 / 0.427857 / 0.793144`，conflict 为 `0.0000%`，但 support 只有 `2.0522%`，cross-support 仍接近零。

本轮最可靠的正结论是: v13 的 posterior support export 和 WTA repair 是可审计、可复现、方向合理的工程闭环；WTA 能真实消除点级冲突。最可靠的负结论是: 当前 masklet density 与 MDL set packing 仍没有把 tiny clean temporal evidence 扩展成完整 object reconstruction。下一步若继续，应优先增加真正覆盖未观测 surfels 的 semantic/video observations，或者引入更强的 object-level split-merge/MDL inference；不能把 v13 当前结果写成正式方法成功。

## 审计材料

审计包:

- `stream4d_v13_probe5_code_review_packet.zip`
- SHA256: 见 `stream4d_v13_probe5_code_review_packet.sha256`
- filelist: `stream4d_v13_probe5_filelist.txt`
- git diff: `stream4d_v13_probe5_git_diff.patch`
- git status: `stream4d_v13_probe5_git_status.txt`
- zip test: `stream4d_v13_probe5_ziptest.log`

审计通过:

```text
py_compile: pass
unittest discover tests: Ran 27 tests ... OK
reportable scan: num_reportable_method_configs=9, num_suspicious_configs=0, num_uses_gt_for_prediction=0
metric integrity: phase0_pass=True, evaluator_ap_core_equal_by_hash=True, gt_files_read_by_rescore=False
```

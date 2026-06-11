# Stream4D v22 实验结果复盘

日期: 2026-06-10
计划文档: `docs/stream4d_v22_rectification_plan.md`
执行日志: `docs/stream4d_v22_执行日志.md`
结果根目录: `Stream3D/outputs/audit`

结论先行: v22 完成了 Phase A code/test hygiene、Phase B self-stitch rectification、Phase C direct D4RT-on-ScanNet reconstruction benchmark，并补了一个真实 checkpoint 的 D5 persistent-ID 小规模 smoke。v22 没有得到可报告 method success，也没有生成 method table。关键新证据是: eval-Sim3 direct reconstruction 在 probe5 上很好，R4 F@10/completeness@20/depth delta1 为 `0.613669/0.856899/0.969155`，说明 D4RT sampled local geometry 在诊断对齐后并非完全坏；但 raw/self canonical 坐标极差，R0/R8 raw probe5 F@10 只有 `0.005726/0.005211`，scene0050 R2/R3 self-stitch F@10 也只有 `0.028903/0.029448`。因此按计划触发 Stop D: 继续修 scale/canonical/self-alignment，不启动 Phase F 或 semantic method table。未运行内容一律标为未运行/NA，不编造数据。

2026-06-11 继续执行更新: 按 Stop D 继续拆解 raw/self 失败原因，新增 camera-space direct metrics 和 R10/R11 `uv_pred + z` pinhole backprojection diagnostic。新证据显示 raw/self 不只是缺 ScanNet world pose/canonical anchoring: R0/R8 probe5 camera-space F@10 只有 `0.005737/0.004404`，而 R4 eval-Sim3 转回 camera-space 后为 `0.173169`。R10/R11 用 D4RT uv+z 回投 camera-space 后只有 `0.013578/0.010180`，是小幅正修复但远不够。因此下一步要查 D4RT `z` scale/shift、uv-z correspondence、valid/visibility filtering 和 OpenD4RT 输出坐标约定，不能只修 world pose 或只把 x/y 换成 pinhole 回投。

2026-06-11 再继续更新: 已补 z/uv/filter/depth-calibration diagnostic。GT-only linear depth calibration 能把 probe5 R15 camera F@10/F@20 提到 `0.244729/0.499092`，证明 D4RT `z` scale/shift 是强 blocker，但该结果使用 ScanNet depth 拟合，不能报告为 method。非 GT positive-z filter 只把 probe5 R11 camera F@10 从 `0.068136` 提到 `0.071781`，不足以过 gate。轴/符号 convention search 也没有救回 UV correspondence，probe5 R11 median reprojection error 仍 `368.200725px`。因此继续 Stop D，不启动 provider/AP 主线或 method table。

2026-06-11 第三次继续更新: 已按 OpenD4RT 官方 `infer_track_3d.py` 自查输出分支，确认旧 cache 混用了 target-frame `uv_pred = pred_local["uv_2d"]` 与 ref0/canonical `xyz_ref = pred_ref["xyz_3d"]`。本轮修改 `D4RTAdapter`/`CarrierBatch` 保存 `xyz_local = pred_local["xyz_3d"]`，并新增 local-vs-ref 真实 checkpoint diagnostic。Probe5 上 `xyz_local` 把 target reprojection median 从 `226.747204px` 降到 `30.753682px`，证明这是实质坐标分支问题；但 raw `xyz_local` camera F@10/F@20 仍只有 `0.014329/0.082448`，raw UVZ F@10/F@20 只有 `0.022713/0.107783`。eval-only calibrated UVZ 上界可到 `0.355244/0.693759`，但仍使用 ScanNet depth 拟合，不能作为 method。结论: 修正了诊断解释和 cache 字段，但 Stop D 继续，不启动 method table。

2026-06-11 第四次继续更新: 已检查 OpenD4RT training loss 的 `xyz_3d` 处理，确认训练配置使用 `normalize_by_mean_depth: true` 和 `sign_x_log1p_abs_x`。新增 `--xyz-transform-modes` diagnostic，验证 `raw` / `signed_log1p` / `signed_expm1`。Probe5 上 `xyz_local + signed_log1p` 是目前最强非 GT raw-camera 信号: xyz camera F@10/F@20 从 `0.014329/0.082448` 提升到 `0.063919/0.233917`，UVZ F@10/F@20 从 `0.022713/0.107783` 提升到 `0.080513/0.271661`，outlier@20 从 `0.837975` 降到 `0.652049`。但 reprojection median 恶化到 `129.863558px`，仍远低于 eval-only calibrated upper bound；`signed_expm1` 和简单 visibility/confidence filtering 都失败。因此这是正向 blocker 线索，不是 method success，Stop D 继续。

2026-06-11 第五次继续更新: 已把 `xyz_local + signed_log1p` 接入 v22 direct reconstruction benchmark，新增 R16-R19 diagnostic-only variants。Probe5 上 R19 (`xyz_local signed_log1p UV+Z`) 的 camera F@10/F@20 达到 `0.080797/0.272468`，outlier@20 降到 `0.535702`，比 R18 raw UV+Z 的 `0.022764/0.107910/0.799749` 明显更好。但 R16-R19 的 world F@10 全部仍为 `0.0`，说明没有非 GT world/canonical anchor 时仍不能形成 method geometry。新增测试后 final unittest 为 `65 tests OK, skipped 1`。

2026-06-11 第六次继续更新: 已新增 R20/R21 eval-only scene Sim3 upper-bound diagnostic。Probe5 上 R20 (`xyz_local raw + eval-Sim3`) 的 world F@10/F@20/comp@20 为 `0.408784/0.679166/0.650008`，证明 `xyz_local` 不是完全无结构，raw world F@10 为 `0.0` 的大头来自缺少非 GT world/canonical anchor。但 R20 仍低于旧 R4 `xyz_ref eval-Sim3` 的 `0.613669/0.868878/0.856899`；R21 (`signed_log1p + eval-Sim3`) 反而降到 `0.281703/0.533511/0.447521`。结论是: `signed_log1p` 主要是 raw camera-space 正信号，不适合直接作为 eval-Sim3 上界主线；下一步应恢复非 GT `pred_ref`/canonical/world anchoring，而不是只切到 `xyz_local` 或只追 log transform。final unittest 为 `66 tests OK, skipped 1`。

2026-06-11 第七次继续更新: 已新增 R22/R23 `xyz_ref0` ref0-pose/scale diagnostic。R22 只用窗口第 0 帧 ScanNet pose 映射 `xyz_ref0`，probe5 world F@10/F@20 只有 `0.108860/0.280340`；R23 固定 ref0 pose 后只用 ScanNet depth/pose anchors 拟合一个 eval-only scale，world F@10/F@20/comp@20 达到 `0.622259/0.902864/0.888330`，略高于旧 R4 full eval-Sim3 的 `0.613669/0.868878/0.856899`。这强烈说明 `xyz_ref` 更像 OpenD4RT ref0 坐标约定，关键缺口是非 GT ref0 pose/scale anchor，而不是任意 Sim3 或只换 `xyz_local`。但 R22/R23 使用 ScanNet pose，R23 还用 ScanNet depth/pose 拟合 scale，仍是 diagnostic-only，不是 method result。

2026-06-11 第八次继续更新: 已新增 R24/R25/R26 非 GT scale proxy diagnostic，尝试只用 D4RT cache 内部 `xyz_local` / `xyz_ref` 统计估计 scale，再接 ScanNet ref0 pose。R24/R25/R26 的 scale 不使用 ScanNet depth/pose anchors，但三者仍使用 ScanNet ref0 pose，因此仍是 diagnostic-only。Probe5 上 R24/R25/R26 world F@10/F@20 分别只有 `0.051524/0.186488`、`0.056787/0.185555`、`0.027983/0.116290`，远低于 R23 `0.622259/0.902864`。三种 proxy 估计出的 mean scale 基本贴近 1: `0.977360/0.976609/1.057880`，不能近似 R23 eval-only scale mean `0.744080`。结论: 简单 local/ref norm ratio 或 source-frame z ratio 不是足够的非 GT scale anchor，Stop D 继续。

2026-06-11 第九次继续更新: 已新增 R27 pose-trajectory scale diagnostic。R27 不使用 ScanNet depth 拟合 scale，而是用 D4RT 内部 `xyz_ref -> xyz_local` rigid translation norm 与 ScanNet pose trajectory baseline 的比值估计 scale；但它仍使用 ScanNet target poses，因此仍是 diagnostic-only。Probe5 上 R27 world F@10/F@20/comp@20 达到 `0.218665/0.556108/0.474452`，明显强于 R22 `0.108860/0.280340/0.254315` 和 R25 `0.056787/0.185555/0.149540`，raw depth delta1 也到 `0.650079`。但 R27 仍显著低于 R23 `0.622259/0.902864/0.888330`，且 scene0030 是强负例: trajectory scale `0.926332`，R23 scale `0.671717`，R27 F@10 只有 `0.017300`。结论: camera/pose trajectory consistency 是强于 norm proxy 的 scale 线索，但还不稳定，也不能作为 method。

2026-06-11 第十次继续更新: 已新增 v22.10 ref0 trajectory consistency diagnostic，用 per-frame CSV 直接比较 D4RT `xyz_ref -> xyz_local` rigid motion 与 ScanNet relative pose。结果排除了一个简单方向 bug: probe5 的 ref-to-target rotation error 中位都很小，约 `1.72..2.36°`，scene0030 也是 `1.930591°`；但 scene0030 的 trajectory ratio 随帧漂移严重，R27 median ratio `0.926332` 相对 R23 eval-only scale `0.656649` 的 absrel 为 `0.410697`，frame120 ratio 达 `1.716449`。候选统计显示原始 `ratio_median` 仍是 probe5 mean absrel 最低的简单 estimator: `0.150980`；low-residual / q25 / weighted median 都没有稳定优于 median。因此 v22.10 没有生成新 R28 方法结果，结论是: R27 失败不是低级 Sim3/SE3 方向反了，而是 D4RT ref/local translation magnitude 不够稳定，不能靠简单 quantile/filter 修成非 GT scale anchor。

2026-06-11 第十一次继续更新: 已新增 v22.11 ref/local trajectory anchor-policy sweep，验证 R27/v22.10 的 ratio 漂移是否来自 carrier selection、visibility/confidence threshold、source-frame subset 或 residual trimming。结论是否定的。Probe5 上 `vc05_trim90` 的 mean absrel vs R23 scale 从 default `0.150980` 小降到 `0.139010`，但 max absrel 仍 `0.415662`，scene0030 反而从 `0.410697` 变到 `0.415662`；scene0030 frame120 在所有 policy 下 ratio 都仍约 `1.686..1.763`，没有被 vc09、ref-source、target-source、nonref-source、trim80/90 修掉。因此漂移不是简单外点/低置信度/源帧选择问题，不能把 `trim90` 当作 R28 方法推进。

2026-06-11 第十二次继续更新: 已新增 v22.12 ref0 scale-convention diagnostic，直接比较 D4RT predicted `xyz_local/xyz_ref` 的 z 尺度、ScanNet target/source depth、R27 trajectory ratio 与 R23 eval-only ref0 scale。结果显示 `target_depth_over_local_z_median` 是目前最接近 R23 scale 的诊断线索: probe5 mean absrel `0.046781`，明显低于 R27 trajectory ratio 的 `0.150980`；scene0030 从 trajectory ratio absrel `0.410697` 降到 `0.060137`，frame120 从 `1.613954` 降到 `0.141389`。这强烈支持“缺 target-depth/mean-depth 尺度锚或 loss normalization 反归一化”的解释。但该线索使用 ScanNet depth，只能作为 upper-bound/attribution diagnostic，不能生成 method row。

2026-06-11 第十三次继续更新: 已新增 v22.13 ref0 intrinsics proxy diagnostic，按 OpenD4RT 官方 `extrinsics_from_queries/intrinsics_from_queries` 公式从 D4RT predicted `xyz_local/xyz_ref` + `uv` 估计 query-derived `fx/fy`，并与 ScanNet intrinsics / R23 eval-only scale 做 diagnostic 对照。结果是负证据: 最好的候选 `local_fx_over_scannet_fx` mean absrel vs R23 scale 仍为 `0.364440`，`local_fxy_over_scannet_fxy` 为 `0.428334`，明显差于 v22.12 `target_depth_over_local_z_median` 的 `0.046781`。scene0030 frame120 的 `local_fxy_over_scannet_fxy=1.005367`，而 R23 scale 是 `0.656649`，说明 intrinsics-from-queries 更多反映 scale-invariant 的 `x/z` 投影一致性，不能恢复 metric depth scale。本轮仍不新增 method row。

2026-06-11 第十四次继续更新: 已新增 v22.14 LoGeR geometry scale-proxy diagnostic，尝试用本仓库 LoGeR geometry backbone 的 local pointmap 作为不使用 ScanNet depth 的 pseudo-depth / pointmap scale proxy。LoGeR 真实 checkpoint inference 只用 GPU `0`，其余诊断和测试 CPU-only。结果是清晰负证据: 最好的非 GT 候选 `loger_z_over_d4rt_ref_z_median` mean absrel vs R23 scale 为 `0.730324`，`loger_z_over_d4rt_local_z_median` 为 `0.734930`，远差于 v22.12 GT-depth attribution 的 `0.046781`。GT 正控 `scannet_depth_over_d4rt_local_z_median` 仍然很强，mean absrel `0.044534`。这说明 LoGeR pointmap 自身不在可直接恢复 R23 的 metric scale 上，不能作为 method scale anchor。本轮仍不新增 method row。

2026-06-11 第十五次继续更新: 已新增 v22.15 OpenD4RT xyz loss scale-invariance diagnostic，直接审计训练 loss 的尺度性质。OpenD4RT source 显示 `xyz_3d` loss 对 pred 和 GT 分别按各自 mean abs-z normalize，再做 `sign(x)*log1p(abs(x))`，因此 uniform pred scale 在 loss space 中理论上不可见。真实 probe5 sweep 验证了这一点: 对 pred scale `0.25/0.5/1.0/2.0/4.0`，80 个 frame row 的 `loss_l1_signed_log_range_across_pred_scales_mean/max` 都是 `0.0`，而 `metric_l1_range_across_pred_scales_mean=4.750760`、`metric_z_absrel_range_across_pred_scales_mean=4.864648`。这说明 v22.12 的 `target_depth/local_z` 是缺失的 inference-time scale attribution，不是一个已由 `xyz_3d` head/loss 自带恢复的 metric scale。本轮仍是 diagnostic-only，不新增 method row。

2026-06-11 第十六次继续更新: 已新增 v22.16 target-scale observability diagnostic，直接测试 v22.12 的 GT-only `target_depth/local_z` scale label 是否能由 D4RT 内部可观测统计预测。Probe5 上 75 个 frame row、27 个 D4RT-internal features 的结果是弱信号但不可用: 对 `target_depth/local_z`，global median LOO mean absrel `0.160601`，最佳单特征 D4RT predictor `rigid_residual_median` 为 `0.129793`，但全特征线性 LOO 过拟合到 `0.466158`，且仍远差于 scene-oracle median `0.056326`。对 R23 eval scale，最佳 D4RT 单特征 `uv_x_std` mean absrel `0.157355`，只比 global median `0.179578` 小幅好；加入 ScanNet pose control 也没有改善。结论是 `target_depth/local_z` 在 scene 内相对稳定，但不能靠 visibility/confidence/uv spread/local-ref ratio/rigid residual 这些简单 D4RT 统计可靠恢复。本轮仍是 diagnostic-only，不新增 method row。

2026-06-11 第十七次继续更新: 已新增 v22.17 OpenD4RT scale metadata audit，直接检查“训练/推理是否已经保留了可反归一化 mean-depth / metric-scale side channel”。结果是负证据但很关键: OpenD4RT `train_effective.yaml` 确认 `normalize_by_mean_depth: true`，`d4rt_loss.py` 确认 pred/GT 分别调用 `_xyz_preprocess` 并各自按 mean abs-z 除尺度；模型输出 key 只有 `confidence/displacement/normal/uv_2d/visibility/xyz_3d`，没有 scale/depth head；官方 `infer_track_3d.py` 返回 `tracks_xyz_local/tracks_xyz_ref0/tracks_uv_norm/...`，没有 scale metadata；v22 local carrier cache 扫描 5 个 `carriers_window*.npz`，`cache_files_with_scale_like_keys=0`、`cache_files_with_depth_like_keys=0`。这说明当前没有“已落盘但未使用”的隐藏反归一化尺度可挖，下一步若要 method 化必须新增/学习/引入外部 scale anchor，而不是继续从现有 artifact 里找字段。本轮仍是 diagnostic-only，不新增 method row。

2026-06-11 第十八次继续更新: 已新增 v22.18 self-supervised scale-sensitivity diagnostic，检查只用现有 D4RT `xyz_local` / `uv` / normalized depth shape / depth rank consistency 是否能约束 uniform metric scale。Probe5 80 个 frame row 上，`uv_reprojection_median_px_range_mean=0.0`、`normalized_z_l1_range_mean=0.0`、`depth_rank_spearman_range_mean=0.0`，说明这些无 GT consistency 对整体尺度完全不敏感；同一 sweep 的 GT-depth positive control 明显敏感，`gt_depth_absrel_range_mean=4.864648`，`gt_depth_absrel_at_scale_1_mean=0.526143`，`gt_depth_absrel_min_mean=0.210736`。结论是: 仅靠当前 D4RT UV/relative-shape/depth-order 信号做 self-supervised target-depth consistency 不能恢复 metric scale，仍需要外部 metric anchor、显式 learned scale/depth head、或保留训练 normalization scale。本轮仍是 diagnostic-only，不新增 method row。

2026-06-11 第十九次继续更新: 已新增 v22.19 scale-anchor tolerance diagnostic，从 R23 `xyz_ref0 + ScanNet ref0 pose + eval-only scale` upper-bound 出发，只扰动 fitted scale，量化未来非 GT scale anchor 的精度需求。Probe5 上 oracle R23 F@10/comp@20/depth delta1 为 `0.622259/0.888330/0.968216`；10% under-scale 时 F@10 retention 只有 `0.514962`，10% over-scale 为 `0.746708`；25% scale error 时 F@10 retention 只有 `0.182785/0.294884`。结论是: scale anchor 误差对 point F-score 很敏感，v22.16 那种约 `0.13` mean AbsRel 的弱内部 predictor 很可能仍不够；后续 scale anchor 目标应逼近 `~0.05` mean AbsRel 量级或做 joint correction。本轮仍是 diagnostic-only，不新增 method row。

## 初始理解

v22 的核心不是直接追求新的 method table，而是先修复 v21.3 暴露出的可信度和诊断缺口:

1. Phase A 要确认 code review packet / provider / self-stitch / native D4RT tests 能在当前代码包独立运行。
2. Phase B 要修正 self-stitch 的 P0 问题: quantile-defined inlier ratio、diagnostic/provider matching 不一致、D5 warmstart 缺 persistent tube id、三窗口 Sim3 测试不足。
3. Phase C 要新增 D4RT-on-ScanNet direct reconstruction benchmark，直接回答 D4RT depth / point cloud / pose / track / stitching / coverage 的真实质量。
4. Phase D/E 必须建立在 A/B/C 可信基础上；如果 direct reconstruction 或 self-stitch gate 失败，按 Stop Rules 暂停语义 3D AP 主线。

## 当前状态

| item | status |
|---|---|
| Phase A code/test audit | 通过 |
| Phase B self-stitch rectification | 代码整改通过；cached diagnostic 已重跑 |
| Phase C direct reconstruction benchmark | 完成；Stop C 不触发；Stop D 触发 |
| Phase D provider replacement rerun | 未运行 full rerun；仅补 D5 persistent provider smoke |
| Phase E occupancy correction | 未运行 full repair；仅补 D5 persistent-ID real smoke |
| Phase F semantic method decision | 未运行；method table 未生成 |
| 2026-06-11 camera-space continuation | 完成；没有形成 method success |
| 2026-06-11 z/uv/depth continuation | 完成；GT-only 上界有效但 method 仍失败 |
| 2026-06-11 xyz_local/local-vs-ref continuation | 完成；发现坐标分支混用，但 raw method 仍失败 |
| 2026-06-11 loss-space xyz transform continuation | 完成；`signed_log1p` 是正向非 GT signal，但未过 method gate |
| 2026-06-11 signed_log1p direct branch | 完成；camera-space 改善，world/canonical 仍失败 |
| 2026-06-11 xyz_local eval-Sim3 upper bound | 完成；`xyz_local` 可被 GT Sim3 对齐到中等质量，但低于旧 R4 `xyz_ref` 上界 |
| 2026-06-11 xyz_ref0 ref0-pose/scale upper bound | 完成；ref0 pose + eval-only scale 达到 R23 F@10/F@20 `0.622259/0.902864`，但使用 ScanNet pose/depth，不能报告为 method |
| 2026-06-11 ref0 local/ref scale proxy | 完成；R24/R25/R26 简单非 GT scale proxy 失败，未逼近 R23 upper-bound |
| 2026-06-11 ref0 pose-trajectory scale | 完成；R27 明显优于 R22/R25，但仍用 ScanNet pose 且低于 R23 |
| 2026-06-11 ref0 trajectory consistency | 完成；排除简单方向 bug，没有找到稳定优于 R27 median 的候选 scale |
| 2026-06-11 ref0 trajectory anchor-policy sweep | 完成；外点/置信度/source-frame policy 不能解释 scene0030 ratio 漂移 |
| 2026-06-11 ref0 scale-convention diagnostic | 完成；GT target-depth / D4RT local-z 是强 scale attribution 线索，但不能作为 method |
| 2026-06-11 ref0 intrinsics proxy diagnostic | 完成；OpenD4RT query-derived intrinsics 不能近似 R23 metric scale，是负证据 |
| 2026-06-11 LoGeR geometry scale proxy diagnostic | 完成；LoGeR pointmap non-GT scale proxy 失败，GT target-depth 正控仍强 |
| 2026-06-11 OpenD4RT loss scale-invariance diagnostic | 完成；`xyz_3d` loss 对 uniform pred scale 不敏感，确认缺 inference-time metric scale anchor |
| 2026-06-11 target-scale observability diagnostic | 完成；D4RT 内部简单统计只有弱信号，不能可靠预测 `target_depth/local_z` scale |
| 2026-06-11 OpenD4RT scale metadata audit | 完成；当前训练/inference/cache 路径没有保留可用 mean-depth / metric-scale side channel |
| 2026-06-11 self-supervised scale-sensitivity diagnostic | 完成；现有 D4RT UV/relative-shape/depth-rank consistency 对 uniform metric scale 不敏感 |
| 2026-06-11 scale-anchor tolerance diagnostic | 完成；R23 oracle scale 偏 10%/25% 会显著降低 F@10 retention，弱 scale predictor 不足以支撑 method |

## Phase A: 代码包与核心依赖审计

新增/修改代码:

- `Stream3D/stream4d_native/sim3.py`
  - 新增公共 `fit_sim3_umeyama`，作为 packaged Sim3 single source of truth。
  - residual 改为 float64 直接计算，避免 float32 输出影响高精度测试。
- `Stream3D/geometry_provider/common.py`
  - 新增 provider 包内 `backproject_xy_world` 和 `fit_transform`。
- `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - 不再从 `tools.d4rt_geometry_diagnostic` 或 `tools.materialize_d4rt_aligned_geometry_for_stream3d` import helper。
- `Stream3D/tools/d4rt_geometry_diagnostic.py`
  - 复用 `stream4d_native.sim3.fit_sim3_umeyama`。
- `Stream3D/tools/materialize_d4rt_aligned_geometry_for_stream3d.py`
  - 保留旧 helper 名称作为 wrapper，但实现来源移到 packaged helper。

审计结果:

| item | result |
|---|---:|
| required core files present | true |
| missing_core_file_count | 0 |
| provider imports from diagnostic `tools.*` | 0 |
| sim3_helper_single_source_of_truth | true |
| py_compile | pass |
| `tests.test_v21_3_geometry_provider` | 5 tests OK |
| `tests.test_native_chunking_and_sim3` | 9 tests OK |
| full unittest discover | 59 tests OK |

Phase A 判定: 通过。一个初始失败是公共 Sim3 residual 使用 float32 路径导致高精度测试失败，已修复并重跑通过。

## Phase B: Self-Stitch 实现整改

新增/修改代码:

- `Stream3D/stream4d_native/self_stitch.py`
  - 统一 self-stitch residual diagnostics 和 overlap matching。
  - 新增 true inlier ratios: abs005 / abs010 / rel001 / rel002 / MAD。
  - matching 层级: persistent/stable id -> same source pixel -> mutual UV + cycle consistency。
- `Stream3D/stream4d_native/sim3.py`
  - `estimate_overlap_sim3` 不再把 p90 quantile residual 当作 inlier ratio。
- `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - provider self-stitch 改用统一 matching。
  - 新增 `persistent_tube_id` 读取和 match source diagnostics。
  - 支持 `all_window_union` / `best_confidence` / `lowest_residual` / `newest_window`。
- `Stream3D/stream4d/carrier_store.py`
  - `CarrierBatch` 支持写 persistent tube identity 字段。
- `Stream3D/tools/export_v21_3_occupancy_carrier_cache.py`
  - D5 exporter 增加 persistent tube id 传播与 retention 统计。
- `Stream3D/tools/native_geometry_diagnostics.py`
  - Phase C diagnostic 与 provider 使用同一 matching helper。

测试结果:

| test | result |
|---|---:|
| `tests.test_native_chunking_and_sim3` | 11 tests OK |
| `tests.test_v21_3_geometry_provider` | 6 tests OK |
| `tests.test_native_occupancy_and_builder` | 10 tests OK |
| full unittest discover | 63 tests OK |
| py_compile | pass |

Cached Phase C rerun after B fixes:

| metric | value |
|---|---:|
| num_pairs | 6 |
| alignment_fail_count | 0 |
| overlap_frame_count_mean | 16.0 |
| overlap_anchor_count_mean | 15891.666667 |
| self_sim3_inlier_ratio_abs005_mean | 0.395558 |
| self_sim3_inlier_ratio_abs010_mean | 0.884642 |
| self_sim3_inlier_ratio_rel001_mean | 0.275439 |
| self_sim3_inlier_ratio_rel002_mean | 0.790420 |
| self_sim3_inlier_ratio_mad_mean | 0.968174 |
| self_sim3_residual_p90_mean | 0.101950 |
| self_sim3_scale_std | 0.048063 |
| accumulated_scale_drift | 0.265904 |

Match source evidence from first pair:

| metric | value |
|---|---:|
| match_source_stable_id_count | 21168 |
| match_source_same_source_pixel_count | 0 |
| match_source_mutual_uv_count | 79 |
| stable_id_match_ratio | 0.996282 |
| mutual_uv_match_ratio | 0.003718 |
| cycle_consistency_pass_ratio | 0.918605 |
| appearance_consistency_available | false |

Phase B 判定: 代码可信度显著改善。此前 `inlier_ratio ~= 0.9` 的 quantile artifact 已去除；现在 abs010 true inlier mean 为 `0.884642`，abs005 只有 `0.395558`，说明 self-stitch pair residual 并非“天然通过”，尺度和阈值选择仍真实影响结论。appearance patch consistency 尚无输入，记录为 unavailable，没有伪造。

Artifact:

- `Stream3D/outputs/audit/v22_phaseB/native_geometry_diagnostics_v22_phaseB.*`

## Phase C: Direct D4RT-on-ScanNet Reconstruction

新增代码:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - 读取已有 D4RT carrier/tube cache。
  - 评估 raw/self/eval-Sim3 D4RT point cloud 到 ScanNet evaluation-only point cloud/depth 的距离。
  - 输出 `direct_reconstruction_summary.*`、`direct_reconstruction_scene_rows.*`、residual histogram PNG、F@10 bar plot。

执行中遇到的 blocker 与修复:

- 初始 all-variant scene0050 smoke 在 eval-Sim3 anchor 构建阶段过慢。
- 修复 runner: 增加 `--variants`、`--max-scenes`、`--max-windows-per-scene`、`--debug-progress`；eval-Sim3 fitting 先采样 anchors 再读 GT depth；self variants 先截断窗口再 stitch。
- 修复后所有 planned direct benchmark smoke/probe5 diagnostic 均完成。

### Scene0050 Direct Rows

R0/R4/R5:

| variant | F@10 | F@5 | comp@20 | Chamfer-L1 | outlier@20 | depth delta1 | depth AbsRel |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 raw single chunk | 0.040025 | 0.010371 | 0.155163 | 2.124020 | 0.907292 | 0.334201 | 0.372214 |
| R4 eval-only scene Sim3 | 0.774526 | 0.322281 | 0.944473 | 0.152623 | 0.005729 | 0.981522 | 0.041555 |
| R5 eval-only per-chunk Sim3 | 0.774526 | 0.322281 | 0.944473 | 0.152623 | 0.005729 | 0.981522 | 0.041555 |

R1/R2/R3:

| variant | F@10 | F@5 | comp@20 | Chamfer-L1 | outlier@20 | depth delta1 | depth AbsRel |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 sliding raw | 0.035874 | 0.008452 | 0.128601 | 2.428985 | 0.895667 | 0.218472 | 0.452097 |
| R2 self-Sim3 | 0.028903 | 0.007580 | 0.109396 | 2.866639 | 0.932667 | 0.172436 | 0.644672 |
| R3 scale-normalized self-Sim3 | 0.029448 | 0.006247 | 0.120370 | 2.869251 | 0.927667 | 0.169182 | 0.646745 |

R6/R7/R8/R9:

| variant | F@10 | F@5 | comp@20 | Chamfer-L1 | outlier@20 | depth delta1 | depth AbsRel |
|---|---:|---:|---:|---:|---:|---:|---:|
| R6 D5 warmstart raw | 0.036785 | 0.006062 | 0.156036 | 2.423119 | 0.893667 | 0.324388 | 0.377857 |
| R7 dense128 raw | 0.025877 | 0.006947 | 0.180592 | 2.116803 | 0.921875 | 0.338104 | 0.361807 |
| R8 D2r4 raw | 0.031051 | 0.007459 | 0.176440 | 2.107375 | 0.914583 | 0.370338 | 0.367863 |
| R9 D5 self-stitch | 0.037612 | 0.005571 | 0.143004 | 2.793762 | 0.917000 | 0.200434 | 0.482259 |

Scene0050 判定:

- eval-only Sim3 能把 single chunk D4RT point cloud 对齐到高质量: R4/R5 F@10 `0.774526`，comp@20 `0.944473`。
- raw coordinates 明显失败: R0 F@10 `0.040025`，outlier@20 `0.907292`。
- self-stitch 没有修复 raw 坐标: R2/R3 F@10 `0.028903/0.029448`，低于 R1 raw `0.035874`。
- D2r4/dense/D5 raw direct reconstruction 都没有接近 eval-Sim3 上界。

### Probe5 Eval-Sim3 Upper Diagnostic

Probe5 R4 eval-only scene Sim3 mean:

| variant | scenes | F@10 | F@5 | comp@20 | Chamfer-L1 | outlier@20 | depth delta1 | depth AbsRel | covered GT inst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R4 eval-only scene Sim3 | 5 | 0.613669 | 0.249488 | 0.856899 | 0.252397 | 0.112500 | 0.969155 | 0.045506 | 0.892631 |

Per scene:

| scene | F@10 | comp@20 | depth delta1 |
|---|---:|---:|---:|
| scene0050_00 | 0.659442 | 0.917692 | 0.991042 |
| scene0011_00 | 0.654141 | 0.927520 | 0.976721 |
| scene0030_00 | 0.660749 | 0.869323 | 0.971098 |
| scene0081_01 | 0.403453 | 0.787542 | 0.919913 |
| scene0591_00 | 0.690561 | 0.782418 | 0.987002 |

### Probe5 Raw Diagnostic

Probe5 R0/R8 raw mean:

| variant | scenes | F@10 | F@5 | comp@20 | Chamfer-L1 | outlier@20 | depth delta1 | depth AbsRel | covered GT inst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 raw single chunk | 5 | 0.005726 | 0.001442 | 0.028761 | 8.002120 | 0.987969 | 0.614228 | 0.241052 | 0.892631 |
| R8 D2r4 raw | 5 | 0.005211 | 0.000685 | 0.027247 | 7.962239 | 0.984375 | 0.596471 | 0.247281 | 0.891272 |

Per scene:

| scene | R0 F@10 | R0 comp@20 | R0 delta1 | R8 F@10 | R8 comp@20 | R8 delta1 |
|---|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.028630 | 0.143803 | 0.397574 | 0.026054 | 0.136235 | 0.284427 |
| scene0011_00 | 0.000000 | 0.000000 | 0.800607 | 0.000000 | 0.000000 | 0.800626 |
| scene0030_00 | 0.000000 | 0.000000 | 0.762180 | 0.000000 | 0.000000 | 0.757877 |
| scene0081_01 | 0.000000 | 0.000000 | 0.412338 | 0.000000 | 0.000000 | 0.430617 |
| scene0591_00 | 0.000000 | 0.000000 | 0.698440 | 0.000000 | 0.000000 | 0.708806 |

Phase C 判定:

- Stop C 不触发。按计划，如果 eval-Sim3 direct reconstruction 本身很差才停止在 geometry source；实际 R4 probe5 F@10 `0.613669`、comp@20 `0.856899`、depth delta1 `0.969155`，说明 D4RT sampled local geometry 在评估对齐后有较高上界。
- Stop D 触发。eval-Sim3 上界好，但 raw/self 不好: R0/R8 raw probe5 F@10 `0.005726/0.005211`，scene0050 R2/R3 self F@10 `0.028903/0.029448`。这符合计划中的 “eval-Sim3 direct good but raw/self bad -> continue scale/stitching, do not enter semantic method”。

Artifact:

- `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r0r4r5_smoke/*`
- `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r1r2r3/*`
- `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r6r7r8r9/*`
- `Stream3D/outputs/audit/v22_direct_reconstruction_probe5_r4_eval_scene/*`
- `Stream3D/outputs/audit/v22_direct_reconstruction_probe5_r0_r8_raw/*`

## D5 Persistent-ID Smoke

目的: v22 在代码层为 D5 exporter 加入 persistent tube identity，本小节只验证真实 checkpoint D5 path 能写出这些字段。该实验是 scene0050 小预算 smoke，不是 AP/coverage 方法结果。

设置:

| item | value |
|---|---|
| scene | scene0050_00 |
| frames | stride 10, max 48 |
| windows | 32 / 16, 2 windows |
| query budget | 512 per window |
| source points per round | 256 |
| checkpoint | `OpenD4RT_32CLIP_9Dataset_NoAUG` |

结果:

| metric | value |
|---|---:|
| status | ok |
| actual_source_query_count | 1024 |
| num_windows | 2 |
| num_carriers_saved | 1013 |
| warmstart_track_count | 353 |
| persistent_tube_retention_count | 6 |
| persistent_tube_retention_rate | 0.005923 |
| total_d4rt_time_sec | 13.120853 |

Window summary:

| window | carriers | persistent tubes | warmstarted carriers | queries | warmstart tracks | retention count | retention rate | uv_in01 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 507 | 507 | 0 | 512 | 0 | 0 | 0.000000 | 0.218997 |
| 1 | 506 | 506 | 6 | 512 | 353 | 6 | 0.011858 | 0.235425 |

NPZ 字段检查:

| window | persistent_tube_id | parent_tube_id | warmstart_source_chunk | warmstart_source_frame | is_warmstarted | src_frame_global | warmstarted |
|---:|---|---|---|---|---|---|---:|
| 0 | present | present | present | present | present | present | 0 |
| 1 | present | present | present | present | present | present | 6 |

Provider smoke:

| policy | variant | status | AP | pre% | hit | #pred | self pairs | stable-id matches | mutual-UV matches | abs010 inlier | residual p90 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_window_union | G2 | ok | NA | 0.000378 | 0.037554 | 1.0 | 1.0 | 31.0 | 76.0 | 0.504673 | 0.175374 |
| all_window_union | G6 | ok | NA | 0.000378 | 0.037554 | 1.0 | 1.0 | 31.0 | 76.0 | 0.504673 | 0.175374 |
| best_confidence | G2 | ok | NA | 0.000378 | 0.037554 | 1.0 | 1.0 | 31.0 | 76.0 | 0.504673 | 0.175374 |
| best_confidence | G6 | ok | NA | 0.000378 | 0.037554 | 1.0 | 1.0 | 31.0 | 76.0 | 0.504673 | 0.175374 |

判定:

- 正向: D5 real checkpoint path 现在确实能写 `persistent_tube_id` / `parent_tube_id` / warmstart source 字段；provider self-stitch 也确实读到了 stable/persistent IDs，`stable-id matches=31`。
- 负向: retention 非常低，window1 只保留 `6/506` 条 persistent carrier；provider AP 为 NA，因为小 cache 只形成 1 个 prediction，evaluator 对空类别均值返回 nan。这个 smoke 不能用作 method 或 AP 结果。
- overlap policy 在该小 cache 上没有区分度，因为最终 `source_windows_mean=1.0`，重复窗口选择空间不足。

Artifact:

- `Stream3D/outputs/stream4d_debug_v22_occupancy_d5_persistent_scene0050_smoke/scene0050_00/carriers_window*.npz`
- `Stream3D/outputs/audit/v22_occupancy_d5_persistent_scene0050_smoke/scene0050_00/summary.*`
- `Stream3D/outputs/audit/v22_d5_persistent_provider_smoke_allwin/*`
- `Stream3D/outputs/audit/v22_d5_persistent_provider_smoke_bestwin/*`

## 2026-06-11 Continuation: Camera-Space Diagnostic

目的: v22 Stop D 后继续拆解 raw/self direct reconstruction 失败。此前 world-space F@10 很低，但 depth median delta1 对 raw 还有 `0.6` 左右，因此需要确认失败是否只是缺 ScanNet world pose/canonical anchoring，还是 D4RT raw xyz 在每帧 camera-local space 内也不对。

代码修改:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - 新增 per-frame camera-space point metrics: `camera_space_fscore@10cm`、`camera_space_fscore@20cm`、`camera_space_chamfer_l1`、`camera_space_outlier_rate_20cm` 等。
  - 新增 `VariantSpec.point_mode`。
  - 新增 R10/R11 diagnostic-only variants:
    - R10: single-chunk raw cache，用 D4RT `uv_pred + z` 通过 ScanNet intrinsics 回投 camera-space points。
    - R11: D2r4 raw cache，用 D4RT `uv_pred + z` 通过 ScanNet intrinsics 回投 camera-space points。

Scene0050 raw/eval/D2r4:

| variant | world F@10 | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth median delta1 | depth LS delta1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 raw | 0.040025 | 0.001529 | 0.009415 | 4.537961 | 0.991667 | 0.334201 | 0.450543 |
| R4 eval-Sim3 | 0.774526 | 0.308946 | 0.765074 | 0.302274 | 0.088542 | 0.981522 | 0.982609 |
| R8 D2r4 raw | 0.031051 | 0.000508 | 0.008848 | 4.539411 | 0.991146 | 0.370338 | 0.448649 |

Scene0050 self-stitch:

| variant | world F@10 | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth median delta1 | depth LS delta1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 sliding raw | 0.035874 | 0.001025 | 0.009551 | 4.185458 | 0.992014 | 0.218472 | 0.520197 |
| R2 self-Sim3 | 0.028903 | 0.001025 | 0.006287 | 4.925231 | 0.994444 | 0.172436 | 0.599891 |
| R3 scale-normalized self-Sim3 | 0.029448 | 0.001025 | 0.006287 | 4.945916 | 0.994444 | 0.169182 | 0.609989 |

Probe5 raw/eval/D2r4:

| variant | world F@10 | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth median delta1 | depth LS delta1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 raw | 0.005726 | 0.005737 | 0.035425 | 3.187777 | 0.960469 | 0.614228 | 0.652708 |
| R4 eval-Sim3 | 0.613669 | 0.173169 | 0.490943 | 0.612816 | 0.443438 | 0.969155 | 0.971001 |
| R8 D2r4 raw | 0.005211 | 0.004404 | 0.033213 | 3.180623 | 0.962031 | 0.596471 | 0.661383 |

UV+Z backprojection:

| scope | variant | point mode | world F@10 | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth median delta1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| scene0050 | R10 single UV+Z | uvz_camera | 0.000000 | 0.020943 | 0.095642 | 2.464201 | 0.889063 | 0.334201 |
| scene0050 | R11 D2r4 UV+Z | uvz_camera | 0.000000 | 0.022487 | 0.091912 | 2.460551 | 0.891667 | 0.370338 |
| probe5 | R10 single UV+Z | uvz_camera | 0.000000 | 0.013578 | 0.060304 | 1.863811 | 0.934531 | 0.614228 |
| probe5 | R11 D2r4 UV+Z | uvz_camera | 0.000000 | 0.010180 | 0.060192 | 1.857475 | 0.936250 | 0.596471 |

Continuation 判定:

- Raw/self 失败不是单纯缺 world pose。若只是缺 ScanNet world pose，raw 点在 per-frame camera-space 应该接近 GT depth 点云；实际 R0/R8 probe5 camera F@10 只有 `0.005737/0.004404`，camera outlier@20 约 `0.960469/0.962031`。
- 新增 camera-space metric 有 sanity check。R4 eval-Sim3 转回 camera-space 后 camera F@10 为 `0.173169`、camera F@20 为 `0.490943`，明显高于 raw。
- Self-stitch 没有修复 local camera geometry。scene0050 R1/R2/R3 camera F@10 都约 `0.001025`，R2/R3 camera outlier@20 仍约 `0.994444`。
- UV+Z pinhole 回投只带来小幅改善。Probe5 R10/R11 camera F@10 `0.013578/0.010180`，比 R0/R8 raw 稍高，但远低于 R4 的 `0.173169`，且 camera outlier@20 仍超过 `0.93`。
- 直接结论: 不能把 v22 的 blocker 简化成“只差 canonical/world transform”或“只要用 uv+z 回投就好”。下一步要查 D4RT z 的 scale/shift、uv-z correspondence、valid/visibility 过滤策略，以及当前 cache 中 `xyz_ref` / `uv_pred` 是否按 OpenD4RT 预期坐标约定解释。

Artifact:

- `Stream3D/outputs/audit/v22_1_camera_space_scene0050_r0r4r8/*`
- `Stream3D/outputs/audit/v22_1_camera_space_scene0050_r1r2r3/*`
- `Stream3D/outputs/audit/v22_1_camera_space_probe5_r0r4r8/*`
- `Stream3D/outputs/audit/v22_1_uvz_camera_scene0050_r10r11/*`
- `Stream3D/outputs/audit/v22_1_uvz_camera_probe5_r10r11/*`

## 2026-06-11 Continuation: z/uv/filter/depth-calibration diagnostic

目的: 继续拆解 camera-space blocker，验证 D4RT `z` 是否只是 scale/shift 问题、正深度过滤是否能作为非 GT 修复、以及 `xyz`/`uv` 是否只是轴顺序或符号解释错误。

代码修改:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - 新增 `VariantSpec.depth_calibration`。
  - 新增 R12/R13/R14/R15 eval-only depth calibration variants。
  - 新增 `_fit_depth_calibration`，用 ScanNet depth 诊断性拟合 median scale 和 linear scale/shift。
  - 新增 `--min-pred-z` 非 GT positive-z filter。
  - 新增 `raw_uvz_*` 信号字段，包括 positive z rate、UV reprojection error、axis/sign best reprojection search。
  - 修复 `_instance_coverage` 每点重复 `np.unique(inst)` 的性能问题，指标定义不变。

### GT-only depth calibration upper-bound

scene0050:

| variant | calibration | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth raw delta1 | depth LS delta1 | linear scale | linear shift | raw uv reproj median px |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R10 single UV+Z | none | 0.061262 | 0.155613 | 2.324445 | 0.846409 | 0.212504 | 0.410729 | NA | NA | 649.548112 |
| R11 D2r4 UV+Z | none | 0.067654 | 0.159721 | 2.312221 | 0.848750 | 0.207035 | 0.469525 | NA | NA | 642.845727 |
| R13 single linear | eval-only linear | 0.185883 | 0.435729 | 0.602599 | 0.393403 | 0.439775 | 0.705642 | 0.201875 | 1.362088 | 649.548112 |
| R15 D2r4 linear | eval-only linear | 0.200641 | 0.443457 | 0.590846 | 0.387583 | 0.477873 | 0.713825 | 0.197679 | 1.372338 | 642.845727 |

Probe5 mean:

| variant | calibration | pred pts | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth raw delta1 | depth LS delta1 | linear scale | linear shift | raw uv reproj median px |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R10 single UV+Z | none | 26951.6 | 0.058913 | 0.164682 | 1.468482 | 0.800008 | 0.212910 | 0.641025 | NA | NA | 377.689976 |
| R11 D2r4 UV+Z | none | 43830.2 | 0.068136 | 0.181029 | 1.455479 | 0.802413 | 0.203974 | 0.655746 | NA | NA | 371.276760 |
| R12 single median | eval-only median | 26951.6 | 0.196149 | 0.427154 | 0.892196 | 0.519310 | 0.608358 | 0.641025 | 0.433842 | 0.786679 | 377.689976 |
| R13 single linear | eval-only linear | 26951.6 | 0.215022 | 0.477449 | 0.563770 | 0.397737 | 0.650033 | 0.700008 | 0.433842 | 0.786679 | 377.689976 |
| R14 D2r4 median | eval-only median | 43830.2 | 0.221848 | 0.450228 | 0.877473 | 0.522725 | 0.609444 | 0.655746 | 0.435936 | 0.786795 | 371.276760 |
| R15 D2r4 linear | eval-only linear | 43830.2 | 0.244729 | 0.499092 | 0.547921 | 0.397922 | 0.659358 | 0.704606 | 0.435936 | 0.786795 | 371.276760 |

Per-scene upper-bound variability:

| variant | scene | camera F@10 | camera F@20 | linear scale | linear shift | raw uv reproj median px |
|---|---|---:|---:|---:|---:|---:|
| R15 | scene0050_00 | 0.200641 | 0.443457 | 0.197679 | 1.372338 | 642.845727 |
| R15 | scene0011_00 | 0.331264 | 0.630486 | 0.477355 | 0.544455 | 368.040013 |
| R15 | scene0030_00 | 0.283783 | 0.565026 | 0.391967 | 1.132190 | 255.101827 |
| R15 | scene0081_01 | 0.144180 | 0.330075 | 0.679095 | 0.466488 | 428.298275 |
| R15 | scene0591_00 | 0.263775 | 0.526418 | 0.433586 | 0.418503 | 162.097957 |

判定:

- GT-only linear z calibration 是强上界修复: probe5 R15 camera F@10/F@20 到 `0.244729/0.499092`，显著高于 R11 no-calibration `0.068136/0.181029`。
- 该修复使用 ScanNet depth 拟合 z scale/shift，不能作为 method。它只证明 z scale/shift 是主因之一。
- linear scale/shift 跨 scene 差异很大，例如 R15 scale/shift 从 scene0050 `0.197679/1.372338` 到 scene0081 `0.679095/0.466488`，不支持一个固定全局常数修复。
- 即使用 GT linear calibration，R15 camera outlier@20 仍 `0.397922`，说明 z 线性校准也不是唯一问题。

### Non-GT positive-z filter

| scope | variant | pred pts | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | per-inst covered |
|---|---|---:|---:|---:|---:|---:|---:|
| scene0050 | R10 positive-z | 17985 | 0.077346 | 0.196634 | 1.274156 | 0.800362 | 0.541667 |
| scene0050 | R11 positive-z | 29364 | 0.085877 | 0.202954 | 1.262718 | 0.802104 | 0.583333 |
| probe5 | R10 positive-z | 23798.0 | 0.062131 | 0.172887 | 1.258419 | 0.790796 | 0.866835 |
| probe5 | R11 positive-z | 40103.0 | 0.071781 | 0.189676 | 1.245578 | 0.793083 | 0.883502 |

判定:

- 正深度过滤是小幅非 GT 修复，但幅度不够。Probe5 R11 camera F@10 只从 `0.068136` 到 `0.071781`。
- scene0050 提升更明显，但 support 被大量删掉，per-instance covered 只有 `0.541667/0.583333`。
- 因此 `z > 0` 过滤不能作为进入 Phase D/E 的 gate。

### Axis/sign reprojection convention diagnostic

| scope | variant | best convention | best median px | best p90 px | default median px | default p90 px |
|---|---|---|---:|---:|---:|---:|
| scene0050 | R10 | `+1,+0,+2` | 641.390188 | 5837.675830 | 649.548112 | 5845.670934 |
| scene0050 | R11 | `+1,+0,+2` | 627.465555 | 6346.574253 | 642.845727 | 6381.275413 |
| probe5 mean | R10 | mixed, mostly `+0,+1,+2` | 376.058392 | 1630.224503 | 377.689976 | 1631.823524 |
| probe5 mean | R11 | mixed, mostly `+0,+1,+2` | 368.200725 | 1731.522253 | 371.276760 | 1738.462485 |

判定:

- 轴/符号搜索不能把 UV reprojection error 降到可用范围。Probe5 R11 median 只从 `371.276760px` 到 `368.200725px`。
- scene0050 的 best convention 是 x/y swap，但 median 仍超过 `600px`，所以不是一个简单 axis/sign typo。
- 这把 blocker 收窄到 D4RT `z` 归一化/反变换、`uv` 与 `z/xyz` 是否来自同一坐标系、以及 carrier selection 的 uv-z consistency。

Artifacts:

- `Stream3D/outputs/audit/v22_2_depth_calibrated_scene0050_r10_r15/*`
- `Stream3D/outputs/audit/v22_2_depth_calibrated_probe5_r10_r15/*`
- `Stream3D/outputs/audit/v22_2_uvz_positive_z_scene0050_r10_r11/*`
- `Stream3D/outputs/audit/v22_2_uvz_positive_z_probe5_r10_r11/*`
- `Stream3D/outputs/audit/v22_2_reproj_convention_scene0050_r10_r11/*`
- `Stream3D/outputs/audit/v22_2_reproj_convention_probe5_r10_r11/*`

## v22.3: D4RT `xyz_local` vs `xyz_ref0`

目的: 继续拆解 Stop D 的 uv-z/canonical blocker。v22.2 发现 `uv_pred + z` 回投仍很差、轴/符号搜索也不能解释问题，因此本轮直接对照 OpenD4RT 官方 inference 输出，确认 `uv_pred` 应该和哪个 3D branch 配套。

代码修改:

- `Stream3D/stream4d/d4rt_adapter.py`
  - 从 `pred_local["xyz_3d"]` 保存 `xyz_local`。
  - valid mask 同时检查 `uv_pred`、`xyz_ref`、`xyz_local`。
- `Stream3D/stream4d/carrier_store.py`
  - `CarrierBatch` 新增可选 `xyz_local`，保存 cache 时写入。
- `Stream3D/stream4d/replay_memory.py`
  - 读取旧 cache 时兼容缺失 `xyz_local`。
- `Stream3D/tools/diagnose_v22_d4rt_local_vs_ref.py`
  - 新增真实 checkpoint diagnostic runner，对比 `xyz_ref0` 和 `xyz_local` 的 target reprojection、depth、camera-space point metrics、UV+Z raw/median/linear metrics。
  - camera metrics 按 frame 计算，避免跨 frame camera-space 点云池化。
  - 可写出带 `xyz_local` 的 diagnostic carrier cache。

关键审计发现:

- 本仓库旧 cache 中 `uv_pred` 来自 `pred_local["uv_2d"]`，但 `xyz_ref` 来自 `pred_ref["xyz_3d"]`。
- OpenD4RT 官方 `infer_track_3d.py` 同时保留 `tracks_xyz_local = pred_local["xyz_3d"]` 与 `tracks_xyz_ref0 = pred_ref["xyz_3d"]`。
- 因此 v22.2 R10/R11 用 `uv_pred + xyz_ref.z` 做 target-camera 回投时，确实存在坐标分支混用风险。

Probe5 mean:

| branch | target reproj median px | target reproj p90 px | raw depth delta1 | median depth delta1 | linear depth delta1 | xyz camera F@10 | xyz camera F@20 | xyz outlier@20 | raw UVZ F@10 | raw UVZ F@20 | linear UVZ F@10 | linear UVZ F@20 | source self UV median px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xyz_ref0` | 226.747204 | 438.900232 | 0.217184 | 0.810198 | 0.819182 | 0.016254 | 0.071730 | 0.874549 | 0.040446 | 0.129410 | 0.229843 | 0.562896 | 4.379759 |
| `xyz_local` | 30.753682 | 52.095698 | 0.197125 | 0.904115 | 0.903552 | 0.014329 | 0.082448 | 0.837975 | 0.022713 | 0.107783 | 0.355244 | 0.693759 | 4.379759 |

Probe5 `xyz_local` per-scene:

| scene | reproj median px | raw depth delta1 | linear depth delta1 | xyz F@10 | raw UVZ F@10 | linear UVZ F@10 | linear UVZ F@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 36.882055 | 0.019104 | 0.938281 | 0.000789 | 0.012700 | 0.602850 | 0.923622 |
| scene0011_00 | 19.019359 | 0.034646 | 0.902362 | 0.001315 | 0.001567 | 0.177168 | 0.617651 |
| scene0030_00 | 26.833669 | 0.071085 | 0.916229 | 0.010321 | 0.007297 | 0.504020 | 0.888708 |
| scene0081_01 | 32.554647 | 0.770416 | 0.858243 | 0.051153 | 0.080385 | 0.201751 | 0.504564 |
| scene0591_00 | 38.478680 | 0.090375 | 0.902645 | 0.008068 | 0.011618 | 0.290428 | 0.534251 |

Scene0050 standalone smoke:

| branch | target reproj median px | target reproj p90 px | raw depth delta1 | linear depth delta1 | xyz camera F@10 | xyz camera F@20 | raw UVZ F@10 | raw UVZ F@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xyz_ref0` | 395.720598 | 1017.236081 | 0.207089 | 0.778596 | 0.000524 | 0.003142 | 0.098752 | 0.240623 |
| `xyz_local` | 36.824630 | 68.256919 | 0.022706 | 0.930795 | 0.001000 | 0.035169 | 0.016902 | 0.081275 |

判定:

- `xyz_local` 明显修复 target UV/XYZ correspondence: probe5 reprojection median `226.747204px -> 30.753682px`，p90 `438.900232px -> 52.095698px`。
- 但 raw `xyz_local` 不是 method success。Raw xyz camera F@10/F@20 只有 `0.014329/0.082448`，raw UVZ F@10/F@20 只有 `0.022713/0.107783`，outlier@20 仍 `0.837975/0.801467`。
- Eval-only depth calibration 上界更强: `xyz_local` linear UVZ F@10/F@20 到 `0.355244/0.693759`，scene0050 到 `0.602850/0.923622`。但这一步使用 ScanNet depth 拟合，只能作为 diagnostic。
- Source self UV median 为 `4.379759px`，说明 query identity/UV 源点基本可追踪；source raw depth delta1 仍只有 `0.091690`，说明 D4RT depth/scale 在 source branch 也没有自然落到 ScanNet metric depth。
- 后续 direct/provider diagnostic 应显式区分 `xyz_local` 与 `xyz_ref0`。目前不能再把旧 `xyz_ref` 直接解释成 target-frame depth。

Artifacts:

- `Stream3D/outputs/audit/v22_3_local_vs_ref_scene0050/local_vs_ref_scene0050.*`
- `Stream3D/outputs/audit/v22_3_local_vs_ref_probe5/local_vs_ref_probe5.*`
- `Stream3D/outputs/stream4d_debug_v22_local_xyz_scene0050_r1/scene0050_00/carriers_window000.npz`
- `Stream3D/outputs/stream4d_debug_v22_local_xyz_probe5_r1/*/carriers_window000.npz`

## v22.4: OpenD4RT loss-space xyz transform diagnostic

目的: 继续拆解 `xyz_local` raw metric geometry 失败原因，验证 OpenD4RT loss-space 的 `sign(x) * log1p(abs(x))` 是否解释了部分输出尺度/分布问题。

新增代码:

- `Stream3D/tools/diagnose_v22_d4rt_local_vs_ref.py`
  - 新增 `--xyz-transform-modes`，支持 `raw`、`signed_log1p`、`signed_expm1`。
  - 输出新增 `xyz_transform` 字段。
  - 默认仍为 `raw`，避免改变旧 diagnostic 行为。

OpenD4RT source/config 发现:

- `Open-d4rt/src/losses/d4rt_loss.py` 的 `_xyz_preprocess` 会先按 mean depth normalize，再可选执行 `sign(x) * log1p(abs(x))`。
- `Open-d4rt/configs/train_effective.yaml` 中 `loss.xyz_3d.normalize_by_mean_depth=true`，`loss.xyz_3d.value_transform=sign_x_log1p_abs_x`。
- checkpoint 旁 `model.yaml` 不包含 loss 配置，所以本轮只做 hypothesis diagnostic，不把 transform 当作已确认 method postprocess。

Probe5 mean:

| branch | transform | n | reproj med | reproj p90 | raw d1 | linear d1 | xyz F@10 | xyz F@20 | outlier@20 | UVZ F@10 | UVZ F@20 | UVZ linear F@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xyz_local` | raw | 5 | 30.753682 | 52.095698 | 0.197125 | 0.903552 | 0.014329 | 0.082448 | 0.837975 | 0.022713 | 0.107783 | 0.355244 |
| `xyz_local` | signed_log1p | 5 | 129.863558 | 179.404753 | 0.262363 | 0.884994 | 0.063919 | 0.233917 | 0.652049 | 0.080513 | 0.271661 | 0.362463 |
| `xyz_local` | signed_expm1 | 5 | 126.714840 | 209.529425 | 0.000308 | 0.674063 | 0.000093 | 0.001296 | 0.996692 | 0.000204 | 0.002161 | 0.152485 |
| `xyz_ref0` | raw | 5 | 226.747204 | 438.900232 | 0.217184 | 0.819182 | 0.016254 | 0.071730 | 0.874549 | 0.040446 | 0.129410 | 0.229843 |
| `xyz_ref0` | signed_log1p | 5 | 284.717411 | 454.359420 | 0.241030 | 0.812735 | 0.043142 | 0.137393 | 0.796829 | 0.070060 | 0.253282 | 0.233035 |
| `xyz_ref0` | signed_expm1 | 5 | 185.013147 | 710.472078 | 0.003144 | 0.679512 | 0.000198 | 0.001237 | 0.997716 | 0.001695 | 0.008510 | 0.140930 |

Probe5 `xyz_local + signed_log1p` per-scene:

| scene | reproj med | raw d1 | xyz F@10 | xyz F@20 | UVZ F@10 | UVZ F@20 |
|---|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 148.381158 | 0.382807 | 0.047929 | 0.294691 | 0.148922 | 0.388211 |
| scene0011_00 | 106.745672 | 0.513386 | 0.110171 | 0.372452 | 0.140946 | 0.415392 |
| scene0030_00 | 119.040356 | 0.108049 | 0.081834 | 0.184626 | 0.044609 | 0.174926 |
| scene0081_01 | 142.523909 | 0.049307 | 0.019721 | 0.075887 | 0.020652 | 0.142802 |
| scene0591_00 | 132.626694 | 0.258266 | 0.059939 | 0.241927 | 0.047437 | 0.236974 |

Visibility/confidence sweep:

| setting | transform | anchors | reproj med | raw d1 | xyz F@10 | xyz F@20 | outlier@20 | UVZ F@10 | UVZ F@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vc05 | raw | 2776.000000 | 30.753682 | 0.197125 | 0.014329 | 0.082448 | 0.837975 | 0.022713 | 0.107783 |
| vc05 | signed_log1p | 2776.000000 | 129.863558 | 0.262363 | 0.063919 | 0.233917 | 0.652049 | 0.080513 | 0.271661 |
| vc07 | raw | 2278.400000 | 32.105646 | 0.192814 | 0.010022 | 0.065964 | 0.859145 | 0.018374 | 0.091269 |
| vc07 | signed_log1p | 2278.400000 | 134.293386 | 0.237951 | 0.046500 | 0.194865 | 0.678586 | 0.063260 | 0.242120 |
| vc09 | raw | 1455.000000 | 32.081337 | 0.175393 | 0.003971 | 0.039990 | 0.894544 | 0.010101 | 0.061590 |
| vc09 | signed_log1p | 1455.000000 | 136.735741 | 0.221335 | 0.040848 | 0.174127 | 0.704281 | 0.057031 | 0.224134 |

判定:

- `xyz_local + signed_log1p` 是 v22 目前最强的非 GT raw-camera 正向信号。Probe5 xyz F@10/F@20 从 `0.014329/0.082448` 到 `0.063919/0.233917`，UVZ F@10/F@20 从 `0.022713/0.107783` 到 `0.080513/0.271661`。
- 但它不是 method success。target reprojection median 从 `30.753682px` 恶化到 `129.863558px`，并且 F@10 仍远低于 GT-only calibrated upper bound。
- `signed_expm1` 是负修复。
- 简单 visibility/confidence filtering 也是负修复或不足修复: vc07/vc09 降低 anchors，也降低 `signed_log1p` F@20。
- 后续可把 `xyz_local + signed_log1p` 接入 direct/provider diagnostic branch，但必须保持 diagnostic-only，不能进入 method table。

Artifact:

- `Stream3D/outputs/audit/v22_4_xyz_transform_scene0050/xyz_transform_scene0050.*`
- `Stream3D/outputs/audit/v22_4_xyz_transform_probe5/xyz_transform_probe5.*`
- `Stream3D/outputs/audit/v22_4_xyz_transform_probe5_vc07/xyz_transform_probe5_vc07.*`
- `Stream3D/outputs/audit/v22_4_xyz_transform_probe5_vc09/xyz_transform_probe5_vc09.*`

## v22.5: `xyz_local + signed_log1p` direct branch

目的: 将 v22.4 找到的正向 signal 接入固定 direct reconstruction benchmark，确认它在同一套 world/camera/coverage 指标下是否足以推进 provider/AP 主线。

新增代码:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - `VariantSpec` 新增 `xyz_field` / `xyz_transform`。
  - 新增 `_transform_xyz_hypothesis` 和 `_apply_variant_xyz_to_windows`。
  - 新增 R16-R19。
- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 覆盖 `signed_log1p` 数学形式。
  - 覆盖从 NPZ 选择 `xyz_local` 并替换 window xyz。

Probe5 direct benchmark:

| variant | label | point mode | xyz transform | world F@10 | camera F@10 | camera F@20 | camera outlier@20 | reproj med | raw d1 | covered GT inst |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| R16 | D4RT xyz_local raw | xyz | raw | 0.000000 | 0.014329 | 0.082525 | 0.836184 | 30.753682 | 0.197125 | 0.937831 |
| R17 | D4RT xyz_local signed-log1p | xyz | signed_log1p | 0.000000 | 0.064293 | 0.234276 | 0.650592 | 129.863558 | 0.262363 | 0.937831 |
| R18 | D4RT xyz_local UV+Z camera backprojection | uvz_camera | raw | 0.000000 | 0.022764 | 0.107910 | 0.799749 | 30.753682 | 0.197125 | 0.937831 |
| R19 | D4RT xyz_local signed-log1p UV+Z camera backprojection | uvz_camera | signed_log1p | 0.000000 | 0.080797 | 0.272468 | 0.535702 | 129.863558 | 0.262363 | 0.937831 |

R19 per-scene:

| scene | camera F@10 | camera F@20 | camera outlier@20 | reproj med | raw d1 |
|---|---:|---:|---:|---:|---:|
| scene0050_00 | 0.149372 | 0.389861 | 0.357096 | 148.381158 | 0.382807 |
| scene0011_00 | 0.141079 | 0.416296 | 0.400661 | 106.745672 | 0.513386 |
| scene0030_00 | 0.045075 | 0.175641 | 0.712989 | 119.040356 | 0.108049 |
| scene0081_01 | 0.020589 | 0.143235 | 0.679641 | 142.523909 | 0.049307 |
| scene0591_00 | 0.047872 | 0.237309 | 0.528120 | 132.626694 | 0.258266 |

判定:

- Direct runner 复现并强化 v22.4: `signed_log1p` 是正向 camera-space signal。R17 相比 R16 的 camera F@10/F@20 从 `0.014329/0.082525` 到 `0.064293/0.234276`。
- `UV+Z + signed_log1p` 最强: R19 camera F@10/F@20 到 `0.080797/0.272468`，camera outlier@20 降到 `0.535702`。
- 但 world F@10 仍为 `0.0`。这说明该 branch 还没有解决非 GT world/canonical anchoring，不能进入 method table。
- reprojection median 仍是 `129.863558px`，说明 signed-log1p 改善 metric proximity 的同时破坏 raw UV/XYZ reprojection consistency。
- Stop D 继续。

Artifact:

- `Stream3D/outputs/audit/v22_5_direct_xyz_local_transform_probe5/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_5_direct_xyz_local_transform_probe5/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_5_direct_xyz_local_transform_probe5/direct_reconstruction_f10_by_variant.png`

## v22.6: `xyz_local` eval-Sim3 upper-bound

目的: v22.5 已证明 `xyz_local + signed_log1p` 有 raw camera-space 正信号，但 R16-R19 world F@10 全部为 `0.0`。本轮用 eval-only scene Sim3 测 `xyz_local` 上界，区分“缺非 GT world/canonical anchor”和“`xyz_local` 本身无可救药”。R20/R21 都使用 ScanNet depth/pose 拟合 Sim3，因此是 diagnostic-only / forbidden-for-method-table。

新增代码:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - 新增 R20: `D4RT xyz_local eval-only scene Sim3`。
  - 新增 R21: `D4RT xyz_local signed-log1p eval-only scene Sim3`。
- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 R20/R21 variant registry 测试。

Probe5 upper-bound:

| variant | label | transform | world F@10 | world F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | reproj med |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R4 old | D4RT eval-only scene Sim3 | old xyz_ref | 0.613669 | 0.868878 | 0.856899 | 0.887500 | 0.112500 | NA | NA | 0.971796 | NA |
| R20 | D4RT xyz_local eval-only scene Sim3 | raw | 0.408784 | 0.679166 | 0.650008 | 0.729472 | 0.270528 | 0.118600 | 0.360883 | 0.788298 | 30.753682 |
| R21 | D4RT xyz_local signed-log1p eval-only scene Sim3 | signed_log1p | 0.281703 | 0.533511 | 0.447521 | 0.711899 | 0.288101 | 0.083132 | 0.279111 | 0.678162 | 129.863558 |

R20/R21 per-scene:

| variant | scene | F@10 | F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | reproj med |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R20 | scene0050_00 | 0.348323 | 0.646027 | 0.515179 | 0.865972 | 0.134028 | 0.096239 | 0.356310 | 0.741122 | 36.882055 |
| R20 | scene0011_00 | 0.466501 | 0.763437 | 0.779927 | 0.747630 | 0.252370 | 0.077049 | 0.341575 | 0.848819 | 19.019359 |
| R20 | scene0030_00 | 0.555682 | 0.817010 | 0.813265 | 0.820790 | 0.179210 | 0.237734 | 0.605065 | 0.913605 | 26.833669 |
| R20 | scene0081_01 | 0.320899 | 0.570688 | 0.599693 | 0.544359 | 0.455641 | 0.057141 | 0.197598 | 0.556240 | 32.554647 |
| R20 | scene0591_00 | 0.352517 | 0.598668 | 0.541974 | 0.668608 | 0.331392 | 0.124838 | 0.303866 | 0.881705 | 38.478680 |
| R21 | scene0050_00 | 0.221050 | 0.511257 | 0.357854 | 0.894861 | 0.105139 | 0.093335 | 0.304126 | 0.596620 | 148.381158 |
| R21 | scene0011_00 | 0.316169 | 0.620123 | 0.557964 | 0.697867 | 0.302133 | 0.062856 | 0.259059 | 0.714961 | 106.745672 |
| R21 | scene0030_00 | 0.351252 | 0.576722 | 0.485485 | 0.710187 | 0.289813 | 0.128640 | 0.386270 | 0.808399 | 119.040356 |
| R21 | scene0081_01 | 0.287594 | 0.580192 | 0.571393 | 0.589266 | 0.410734 | 0.054087 | 0.215541 | 0.525424 | 142.523909 |
| R21 | scene0591_00 | 0.232450 | 0.379260 | 0.264909 | 0.667314 | 0.332686 | 0.076742 | 0.230561 | 0.745408 | 132.626694 |

判定:

- R20 证明 `xyz_local` 不是完全坏点云。它在 eval-Sim3 后达到 world F@10/F@20 `0.408784/0.679166`，说明 v22.5 raw world F@10 为 `0.0` 的大头确实来自缺少非 GT world/canonical anchoring。
- 但 R20 仍显著低于旧 R4: F@10 低 `0.204885`，comp@20 低 `0.206891`。这说明 `pred_ref`/canonical 分支仍携带更强的 eval-aligned geometry，不能简单切成 `xyz_local` 后推进 provider/AP。
- R21 是负向 upper-bound。`signed_log1p` 虽提升 raw camera-space，但在 eval-Sim3 上界中低于 R20；它更像 loss-space/camera-space clue，而不是直接替代 geometry 坐标。
- Stop D 继续。下一步应优先修非 GT `pred_ref`/canonical/world anchor，并用 `xyz_local` 检查 UV/target-frame consistency；暂不启动 provider/AP 或 method table。

Artifact:

- `Stream3D/outputs/audit/v22_6_eval_sim3_xyz_local_transform_probe5/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_6_eval_sim3_xyz_local_transform_probe5/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_6_eval_sim3_xyz_local_transform_probe5/direct_reconstruction_f10_by_variant.png`

## v22.7: `xyz_ref0` ref0-pose / scale upper-bound

目的: v22.6 说明 `xyz_ref`/canonical 分支仍有更高 eval-aligned 上界，但 raw world 完全失败。本轮按 OpenD4RT worldtrack/demo 代码线索新增两个 diagnostic:

- R22: 将 `xyz_ref0` 直接通过窗口第 0 帧 ScanNet pose 映射到 world。
- R23: 固定 ref0 pose，只用 ScanNet depth/pose anchors 拟合一个 eval-only scale，再映射到 world。

R22/R23 都使用 ScanNet pose；R23 还使用 ScanNet depth/pose anchors 拟合 scale。因此它们是 diagnostic-only / forbidden-for-method-table，不是 method result。本轮 CPU-only 运行，未占用 GPU。

代码修改:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - 新增 R22/R23 variant。
  - 新增 `_fit_ref0_pose_scale`，把 GT world anchors 反变换到 ref0 camera 坐标，只解一个正 scale。
  - 新增 `_spec_outputs_world`，让 R22/R23 的 depth/camera metric 按 world-space 输出处理。
  - scene row 新增 `ref0_pose_*` / `ref0_pose_scale_*` 诊断字段。
- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 R22/R23 registry/world-output 单测。

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 4 tests OK |

Probe5 mean comparison:

| variant | label | world F@10 | world F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | pred->GT med | pred->GT p90 | ref0 scale |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | D4RT single-chunk raw | 0.005726 | 0.016965 | 0.028761 | 0.012031 | 0.987969 | NA | NA | 0.205612 | 4.346847 | 6.385360 | NA |
| R4 | D4RT eval-only scene Sim3 | 0.613669 | 0.868878 | 0.856899 | 0.887500 | 0.112500 | NA | NA | 0.971796 | 0.074398 | 0.321411 | NA |
| R22 | xyz_ref0 + ScanNet ref0 pose | 0.108860 | 0.280340 | 0.254315 | 0.327160 | 0.672840 | 0.036096 | 0.143050 | 0.111773 | 0.530517 | 1.261981 | NA |
| R23 | xyz_ref0 + ScanNet ref0 pose + eval-only scale | 0.622259 | 0.902864 | 0.888330 | 0.920589 | 0.079411 | 0.492137 | 0.809117 | 0.966551 | 0.063193 | 0.176206 | 0.744080 |

R23 per-scene:

| scene | F@10 | F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | pred->GT med | pred->GT p90 | ref0 scale | residual p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.600588 | 0.965901 | 0.941033 | 0.992119 | 0.007881 | 0.500525 | 0.936842 | 0.975650 | 0.063385 | 0.138309 | 0.618160 | 0.316476 |
| scene0011_00 | 0.751893 | 0.968949 | 0.976211 | 0.961795 | 0.038205 | 0.519124 | 0.869600 | 0.978951 | 0.040801 | 0.117973 | 0.648489 | 0.181278 |
| scene0030_00 | 0.773478 | 0.939515 | 0.898917 | 0.983953 | 0.016047 | 0.693516 | 0.913275 | 0.960954 | 0.034564 | 0.111192 | 0.671717 | 0.272399 |
| scene0081_01 | 0.197514 | 0.721343 | 0.759940 | 0.686477 | 0.313523 | 0.082684 | 0.481774 | 0.929328 | 0.158568 | 0.439288 | 1.086632 | 0.802004 |
| scene0591_00 | 0.787823 | 0.918611 | 0.865551 | 0.978601 | 0.021399 | 0.664836 | 0.844096 | 0.987873 | 0.018648 | 0.074266 | 0.695402 | 0.214334 |

判定:

- R22 只用 ref0 pose 仍明显失败: F@10/F@20 `0.108860/0.280340`，raw depth delta1 `0.111773`。这说明 `xyz_ref0` 不是只缺 pose，还缺尺度。
- R23 固定 ref0 pose 后只拟合 eval-only scale，F@10/F@20/comp@20 达到 `0.622259/0.902864/0.888330`，略高于旧 R4 full eval-Sim3 的 `0.613669/0.868878/0.856899`。这说明 `xyz_ref` 的坐标系更接近 OpenD4RT ref0 convention，合适的 ref0 pose + scale 比自由 Sim3 更稳定。
- R23 camera-space 也明显变好: camera F@10/F@20 为 `0.492137/0.809117`，raw depth delta1 为 `0.966551`。这不是只在 world Chamfer 上碰巧有效。
- scale 不是常数: per-scene ref0 scale 从 `0.618160` 到 `1.086632`，scene0081 residual p90 `0.802004` 明显偏高。后续不能硬编码一个全局 scale。
- Stop D 继续。下一步应找非 GT ref0 pose/scale anchor，目标是逼近 R23，而不是继续只做任意 Sim3、只切 `xyz_local`、或只追 `signed_log1p`。

Artifact:

- `Stream3D/outputs/audit/v22_7_ref0_pose_scale_probe5/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_7_ref0_pose_scale_probe5/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_7_ref0_pose_scale_probe5/direct_reconstruction_f10_by_variant.png`

## v22.8: non-GT local/ref scale proxy diagnostic

目的: v22.7 的 R23 证明 `xyz_ref0 + ref0 pose + scale` 是强 upper-bound，但 R23 的 scale 使用 ScanNet depth/pose anchors。本轮尝试不用 ScanNet depth 拟合 scale，只从 D4RT cache 内部的 `xyz_local` / `xyz_ref` 统计估计 scale，再接 ScanNet ref0 pose 做 direct reconstruction diagnostic。

边界:

- R24/R25/R26 的 scale proxy 不使用 ScanNet depth/pose anchors。
- R24/R25/R26 仍使用 ScanNet ref0 pose，因此仍是 diagnostic-only / forbidden-for-method-table。
- 本轮 CPU-only 运行，未占用 GPU；用户当前 GPU 限制为只可用 0/1。

代码修改:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - 新增 R24: `local/ref median-norm scale`。
  - 新增 R25: `local/ref RMS-norm scale`。
  - 新增 R26: `source-frame z scale`。
  - 新增 `_estimate_ref0_local_scale`。
- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 R24/R25/R26 registry/world-output 单测。
  - 新增 synthetic scale estimator 单测。

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 6 tests OK |

Probe5 mean comparison:

| variant | scale proxy | world F@10 | world F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | pred->GT med | pred->GT p90 | estimated scale |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R23 upper-bound | GT depth/pose scale | 0.622259 | 0.902864 | 0.888330 | 0.920589 | 0.079411 | 0.492137 | 0.809117 | 0.966551 | 0.063193 | 0.176206 | 0.744080 |
| R24 | local/ref median-norm | 0.051524 | 0.186488 | 0.149141 | 0.283856 | 0.716144 | 0.025446 | 0.117937 | 0.197789 | 0.594960 | 1.484469 | 0.977360 |
| R25 | local/ref RMS-norm | 0.056787 | 0.185555 | 0.149540 | 0.282856 | 0.717144 | 0.030268 | 0.116029 | 0.203527 | 0.594858 | 1.458447 | 0.976609 |
| R26 | source-frame z | 0.027983 | 0.116290 | 0.081328 | 0.208961 | 0.791039 | 0.008791 | 0.052977 | 0.168462 | 0.754244 | 1.672021 | 1.057880 |

Per-scene estimated scale:

| scene | R24 scale | R24 F@10/F@20 | R25 scale | R25 F@10/F@20 | R26 scale | R26 F@10/F@20 |
|---|---:|---|---:|---|---:|---|
| scene0050_00 | 0.916684 | 0.010952 / 0.089711 | 0.947803 | 0.009401 / 0.064824 | 1.162987 | 0.000000 / 0.000000 |
| scene0011_00 | 1.053518 | 0.020654 / 0.076539 | 1.021793 | 0.020219 / 0.080914 | 1.017139 | 0.019581 / 0.082416 |
| scene0030_00 | 1.017340 | 0.010479 / 0.053080 | 1.017801 | 0.010556 / 0.053020 | 1.036978 | 0.010902 / 0.052890 |
| scene0081_01 | 0.867741 | 0.196239 / 0.583838 | 0.882035 | 0.220488 / 0.597902 | 1.072823 | 0.084079 / 0.312270 |
| scene0591_00 | 1.031516 | 0.019296 / 0.129270 | 1.013613 | 0.023269 / 0.131114 | 0.999472 | 0.025352 / 0.133875 |

判定:

- R24/R25/R26 都没有逼近 R23。最好的 R25 world F@10/F@20 只有 `0.056787/0.185555`，远低于 R23 `0.622259/0.902864`，也低于 R22 的 F@20 `0.280340`。
- 三种简单 scale proxy 都估出接近 `1.0` 的 scale: R24/R25/R26 mean 为 `0.977360/0.976609/1.057880`。这与 R23 eval-only mean scale `0.744080` 和 per-scene 范围 `0.618160..1.086632` 不匹配。
- scene0081 在 R24/R25 上相对最好，但不是整体解决；其它 4 个 scene 的 R24/R25 F@10 都在 `0.009401..0.023269` 附近。
- 结论: 非 GT scale anchor 不能用简单 `xyz_local` / `xyz_ref` norm ratio 或 source-frame z ratio 近似。下一步需要更强的 D4RT-native scale clue，例如模型 normalization 反变换、ref0/camera trajectory consistency、source-target reprojection+depth consistency、或 object/mask scale prior。
- Stop D 继续，不启动 provider/AP 或 method table。

Artifact:

- `Stream3D/outputs/audit/v22_8_ref0_local_scale_probe5/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_8_ref0_local_scale_probe5/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_8_ref0_local_scale_probe5/direct_reconstruction_f10_by_variant.png`

## v22.9: pose-trajectory scale diagnostic

目的: v22.8 证明简单 `xyz_local` / `xyz_ref` norm/z ratio 不能近似 R23 的 ref0 metric scale。本轮测试更强的 diagnostic 线索: 用 D4RT 内部 `xyz_ref -> xyz_local` carrier 对拟合无尺度 rigid transform，取其 translation norm；再用 ScanNet pose 中 ref0 到 target 的 camera-center baseline 除以该 D4RT translation norm，得到 trajectory scale。该 scale 不使用 ScanNet depth，但使用 ScanNet pose trajectory，因此仍是 diagnostic-only / forbidden-for-method-table。

代码修改:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - 新增 R27: `D4RT xyz_ref0 + ref0 pose + pose-trajectory scale`。
  - 新增 `_fit_rigid_no_scale`。
  - 新增 `_estimate_ref0_pose_trajectory_scale`。
- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 R27 registry/world-output 单测。
  - 新增 synthetic trajectory-scale 单测。

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 7 tests OK |

Probe5 mean comparison:

| variant | scale source | world F@10 | world F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | pred->GT med | pred->GT p90 | scale |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R22 | ref0 pose only | 0.108860 | 0.280340 | 0.254315 | 0.327160 | 0.672840 | 0.036096 | 0.143050 | 0.111773 | 0.530517 | 1.261981 | NA |
| R23 | eval-only depth/pose scale | 0.622259 | 0.902864 | 0.888330 | 0.920589 | 0.079411 | 0.492137 | 0.809117 | 0.966551 | 0.063193 | 0.176206 | 0.744080 |
| R25 | local/ref RMS-norm scale | 0.056787 | 0.185555 | 0.149540 | 0.282856 | 0.717144 | 0.030268 | 0.116029 | 0.203527 | 0.594858 | 1.458447 | 0.976609 |
| R27 | pose trajectory scale | 0.218665 | 0.556108 | 0.474452 | 0.701966 | 0.298034 | 0.122417 | 0.415634 | 0.650079 | 0.189421 | 0.517978 | 0.693379 |

R27 per-scene:

| scene | F@10 | F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | pred->GT med | pred->GT p90 | traj scale | scale mean/std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| scene0050_00 | 0.175587 | 0.670721 | 0.524977 | 0.928487 | 0.071513 | 0.134783 | 0.607808 | 0.852804 | 0.137554 | 0.192348 | 0.493855 | 0.467927 / 0.087287 |
| scene0011_00 | 0.550465 | 0.862890 | 0.834426 | 0.893365 | 0.106635 | 0.266822 | 0.642778 | 0.900787 | 0.055773 | 0.211131 | 0.572213 | 0.695970 / 0.226055 |
| scene0030_00 | 0.017300 | 0.129462 | 0.092278 | 0.216840 | 0.783160 | 0.012128 | 0.086723 | 0.118110 | 0.492710 | 1.223101 | 0.926332 | 0.946675 / 0.269850 |
| scene0081_01 | 0.234320 | 0.605460 | 0.540027 | 0.688938 | 0.311062 | 0.139921 | 0.416576 | 0.839753 | 0.102437 | 0.661338 | 0.889791 | 0.798835 / 0.220658 |
| scene0591_00 | 0.115653 | 0.512004 | 0.380550 | 0.782201 | 0.217799 | 0.058430 | 0.324285 | 0.538942 | 0.158632 | 0.301971 | 0.584703 | 0.596980 / 0.154907 |

判定:

- R27 是本轮第一个不用 ScanNet depth scale fitting、明显改善 ref0 direct reconstruction 的 diagnostic: F@10/F@20 从 R22 `0.108860/0.280340` 提到 `0.218665/0.556108`，raw depth delta1 从 `0.111773` 提到 `0.650079`。
- R27 明显强于 v22.8 的简单 local/ref scale proxy: R25 F@10/F@20 只有 `0.056787/0.185555`。
- 但 R27 仍远低于 R23 upper-bound `0.622259/0.902864`，且使用 ScanNet pose trajectory，不能作为 method。
- R27 的主要负例是 scene0030: trajectory scale `0.926332`，明显高于 R23 scale `0.671717`，F@10/F@20 只有 `0.017300/0.129462`。这说明 D4RT ref/local rigid translation baseline 并不总是稳定的 metric scale clue。
- 正向结论: camera/pose trajectory consistency 是比 norm/z proxy 更强的 scale 线索，后续可考虑用非 GT camera trajectory / VO / model-internal motion scale 逼近它。
- 负向结论: 仅凭 D4RT ref/local rigid translation + pose baseline 仍不够稳定，不能启动 provider/AP 或 method table。Stop D 继续。

Artifact:

- `Stream3D/outputs/audit/v22_9_ref0_pose_trajectory_scale_probe5/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_9_ref0_pose_trajectory_scale_probe5/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_9_ref0_pose_trajectory_scale_probe5/direct_reconstruction_f10_by_variant.png`

## v22.10: ref0 trajectory consistency diagnostic

目的: v22.9 的 R27 证明 pose trajectory scale 是强线索，但 scene0030 是明显负例。本轮不急着加 R28，而是把每个 target frame 的 D4RT ref-to-local rigid motion 与 ScanNet relative pose 拆开诊断，确认失败来自 transform 方向、rotation、translation direction，还是 translation magnitude / ratio 漂移。

代码修改:

- `Stream3D/tools/diagnose_v22_ref0_trajectory_scale.py`
  - 新增 per-frame ref0 trajectory diagnostic。
  - 输出 `ref0_trajectory_frame_rows.*`、`ref0_trajectory_window_rows.*`、`ref0_trajectory_scene_summary.*`、`ref0_trajectory_candidate_scale_errors.*`。
  - 每帧记录 D4RT rigid translation norm、ScanNet relative pose translation norm、ratio、rotation error、translation-direction error、rigid residual，以及与 R23 eval-only ref0 scale 的 absrel。
- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 rotation/translation direction helper 单测。

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 8 tests OK |
| full `unittest discover tests` with `envs/loger` | 71 tests OK, skipped 1 |

Probe5 scene summary:

| scene | R23 eval scale | median ratio | q25 | q75 | median absrel | rot err med | trans dir err med | rigid p90 med | frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.561127 | 0.493855 | 0.410124 | 0.552539 | 0.119887 | 1.717264 | 10.175581 | 0.047783 | 15 |
| scene0011_00 | 0.561406 | 0.572213 | 0.523307 | 0.847068 | 0.019250 | 2.363984 | 8.807973 | 0.061000 | 15 |
| scene0030_00 | 0.656649 | 0.926332 | 0.771188 | 1.042536 | 0.410697 | 1.930591 | 15.677520 | 0.029415 | 15 |
| scene0081_01 | 0.862760 | 0.889791 | 0.598418 | 0.932503 | 0.031330 | 2.315460 | 11.082123 | 0.066886 | 15 |
| scene0591_00 | 0.707645 | 0.584703 | 0.508000 | 0.623795 | 0.173734 | 1.902544 | 10.212947 | 0.039773 | 15 |

Candidate scale error vs R23 eval-only scale:

| candidate | mean absrel | median absrel | max absrel | scene count |
|---|---:|---:|---:|---:|
| ratio_median | 0.150980 | 0.119887 | 0.410697 | 5 |
| ratio_residual_weighted_median | 0.169472 | 0.173734 | 0.368032 | 5 |
| ratio_low_residual_median | 0.183642 | 0.151174 | 0.306391 | 5 |
| ratio_low_direction_median | 0.191472 | 0.141591 | 0.554731 | 5 |
| ratio_mean | 0.215588 | 0.166095 | 0.441677 | 5 |
| ratio_q25 | 0.219984 | 0.269107 | 0.306391 | 5 |
| ratio_q75 | 0.262226 | 0.118492 | 0.587662 | 5 |
| ratio_q10 | 0.279964 | 0.383812 | 0.457045 | 5 |
| ratio_min | 0.334162 | 0.445036 | 0.492834 | 5 |

scene0030 frame-level highlights:

| frame | ratio | absrel vs R23 scale | rot err | trans dir err | rigid p90 | D4RT trans | pose trans |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.607577 | 0.074730 | 0.132418 | 85.049825 | 0.026309 | 0.022755 | 0.013825 |
| 60 | 0.665212 | 0.013042 | 0.674470 | 8.785631 | 0.020088 | 0.289112 | 0.192321 |
| 80 | 0.926332 | 0.410697 | 1.475769 | 15.677520 | 0.023233 | 0.176657 | 0.163643 |
| 120 | 1.716449 | 1.613954 | 4.772149 | 12.395835 | 0.035809 | 0.236630 | 0.406164 |
| 150 | 0.776038 | 0.181816 | 6.395605 | 25.863996 | 0.047475 | 0.272615 | 0.211559 |

判定:

- v22.10 排除了一个简单实现错误: D4RT ref-to-local rigid rotation 与 ScanNet ref-to-target relative pose 的方向大体一致，probe5 median rotation error 约 `1.72..2.36°`，scene0030 也只有 `1.930591°`。
- scene0030 的失败更像 translation magnitude/ratio instability，而不是 rotation 反了。它在 frame60 ratio `0.665212` 几乎贴近 R23 eval scale `0.656649`，但 frame120 ratio 到 `1.716449`，直接把 median 推高到 `0.926332`。
- 简单替代 estimator 没有明确胜出。`ratio_median` 的 probe5 mean absrel `0.150980` 仍是所有候选里最低；`ratio_low_residual_median` 降低 max absrel 到 `0.306391`，但 mean absrel 变差到 `0.183642`，且 scene0081 明显受损。
- 因此本轮不新增 R28 benchmark，也不启动 provider/AP。下一步应查 D4RT ref/local translation magnitude 为什么随时间漂移，例如 model normalization、per-frame local scale、pose baseline saturation、或者 source-target correspondence 的 motion-dependent bias。

Artifact:

- `Stream3D/tools/diagnose_v22_ref0_trajectory_scale.py`
- `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_frame_rows.*`
- `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_window_rows.*`
- `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_scene_summary.*`
- `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_candidate_scale_errors.*`
- `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_consistency.md`

## v22.11: ref/local trajectory anchor-policy sweep

目的: v22.10 说明 scene0030 的主要问题是 D4RT ref/local translation magnitude 和 ScanNet pose baseline 的 ratio 随时间漂移。本轮继续排查这个漂移是否来自 carrier anchor policy: visibility/confidence 太宽、source-frame subset 混杂、或者 rigid residual 外点污染。

代码修改:

- `Stream3D/tools/diagnose_v22_ref0_trajectory_policy_sweep.py`
  - 新增 anchor-policy sweep diagnostic。
  - policies: `vc05_all`、`vc07_all`、`vc09_all`、`vc05_ref_source`、`vc05_target_source`、`vc05_nonref_source`、`vc05_trim90`、`vc05_trim80`。
  - 输出 per-frame / per-window / per-scene / per-policy error CSV+JSON+MD。
- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 source-policy mask 单测。

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 9 tests OK |

Probe5 policy error vs R23 eval-only scale:

| policy | mean absrel | median absrel | max absrel | mean frames | mean anchors |
|---|---:|---:|---:|---:|---:|
| vc05_trim90 | 0.139010 | 0.088908 | 0.415662 | 15.0 | 158.4 |
| vc05_trim80 | 0.144020 | 0.098948 | 0.422611 | 15.0 | 140.8 |
| vc05_all | 0.150980 | 0.119887 | 0.410697 | 15.0 | 176.2 |
| vc05_ref_source | 0.152608 | 0.097989 | 0.438942 | 9.6 | 12.6 |
| vc05_nonref_source | 0.153684 | 0.122569 | 0.408695 | 15.0 | 163.6 |
| vc09_all | 0.159319 | 0.106471 | 0.388400 | 15.0 | 93.8 |
| vc05_target_source | 0.164263 | 0.149934 | 0.390012 | 15.0 | 30.6 |
| vc07_all | 0.164732 | 0.111906 | 0.408655 | 15.0 | 145.2 |

scene0030 policy summary:

| policy | ratio median | absrel vs R23 scale | ratio std | anchors med | rot err med | trans dir err med | residual p90 med | frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vc05_all | 0.926332 | 0.410697 | 0.269850 | 310 | 1.930591 | 15.677520 | 0.029415 | 15 |
| vc07_all | 0.924992 | 0.408655 | 0.267216 | 267 | 1.938987 | 15.275346 | 0.028053 | 15 |
| vc09_all | 0.911691 | 0.388400 | 0.264265 | 136 | 1.968665 | 15.978524 | 0.025529 | 15 |
| vc05_ref_source | 0.944879 | 0.438942 | 0.287640 | 20 | 1.952456 | 18.166516 | 0.014459 | 15 |
| vc05_target_source | 0.912749 | 0.390012 | 0.272292 | 31 | 1.955479 | 16.902566 | 0.025078 | 15 |
| vc05_nonref_source | 0.925018 | 0.408695 | 0.268558 | 289 | 1.927312 | 15.531525 | 0.030274 | 15 |
| vc05_trim90 | 0.929592 | 0.415662 | 0.270684 | 279 | 1.943606 | 16.396445 | 0.024304 | 15 |
| vc05_trim80 | 0.934155 | 0.422611 | 0.278097 | 248 | 1.942271 | 16.687493 | 0.020790 | 15 |

scene0030 hard frame 120:

| policy | ratio | absrel vs R23 scale | D4RT trans | pose trans | anchors | residual p90 |
|---|---:|---:|---:|---:|---:|---:|
| vc05_all | 1.716449 | 1.613954 | 0.236630 | 0.406164 | 302 | 0.035809 |
| vc07_all | 1.714694 | 1.611280 | 0.236873 | 0.406164 | 256 | 0.034456 |
| vc09_all | 1.686167 | 1.567837 | 0.240880 | 0.406164 | 93 | 0.034764 |
| vc05_ref_source | 1.763000 | 1.684846 | 0.230382 | 0.406164 | 18 | 0.027978 |
| vc05_target_source | 1.701453 | 1.591117 | 0.238716 | 0.406164 | 25 | 0.026699 |
| vc05_nonref_source | 1.713780 | 1.609889 | 0.236999 | 0.406164 | 284 | 0.035751 |
| vc05_trim90 | 1.715394 | 1.612348 | 0.236776 | 0.406164 | 271 | 0.029741 |
| vc05_trim80 | 1.729790 | 1.634271 | 0.234805 | 0.406164 | 241 | 0.025416 |

判定:

- `trim90` 在 probe5 mean 上有小幅正信号: absrel `0.150980 -> 0.139010`，但它没有修 scene0030，且 max absrel 仍 `0.415662`。
- `vc09_all` / `target_source` 可以稍微降低 scene0030 absrel 到 `0.388400/0.390012`，但 probe5 mean 变差到 `0.159319/0.164263`，不是稳定修复。
- `ref_source` anchor 太稀，mean frame count 只有 `9.6`，且 scene0030 更差。
- scene0030 frame120 在所有 policy 下 ratio 都仍约 `1.686..1.763`。这说明 hard failure 不是由低置信度点、source-frame 混杂或 residual 外点造成，而是 D4RT ref/local translation magnitude 本身没有跟随 ScanNet pose baseline 增长。
- 因此本轮不新增 R28，不启动 provider/AP。下一步应转向 per-frame/per-window scale normalization 来源、D4RT model normalization / crop-depth convention，或检查 `xyz_ref` 与 `xyz_local` 的 translation 是否存在 motion-dependent compression。

Artifact:

- `Stream3D/tools/diagnose_v22_ref0_trajectory_policy_sweep.py`
- `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_frame_rows.*`
- `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_window_rows.*`
- `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_scene_summary.*`
- `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_errors.*`
- `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_sweep.md`

## v22.12: ref0 scale-convention diagnostic

目的: v22.10/v22.11 已排除 transform 方向、简单 anchor policy、confidence/source-frame/residual trimming。结合 OpenD4RT loss 中 `normalize_by_mean_depth=true`，本轮直接检查 predicted `xyz_local/xyz_ref` 的 z 尺度与 ScanNet target/source depth 是否能解释 R23 eval-only ref0 scale。该实验使用 ScanNet depth/pose 作 attribution，diagnostic-only。

代码修改:

- `Stream3D/tools/diagnose_v22_ref0_scale_convention.py`
  - 对每个 target frame 输出 D4RT ref->local rigid translation ratio、predicted local/ref z、ScanNet target/source depth、以及多个候选 scale。
  - 与 R23 eval-only ref0 scale 比较 absrel。
  - 输出 frame/window/scene/candidate-error CSV+JSON+MD。
- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 scale-convention helper 单测。

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 10 tests OK |
| full unittest discover | 73 tests OK, skipped 1 |

Probe5 candidate error vs R23 eval-only scale:

| candidate | mean absrel | median absrel | max absrel | mean candidate |
|---|---:|---:|---:|---:|
| target_depth_over_local_z_median | 0.046781 | 0.058547 | 0.082377 | 0.670878 |
| target_depth_over_local_z_mean | 0.054944 | 0.034633 | 0.161532 | 0.647083 |
| source_depth_over_ref_z_mean | 0.081876 | 0.067390 | 0.141542 | 0.656299 |
| source_depth_over_local_z_median | 0.091145 | 0.091139 | 0.169265 | 0.640300 |
| target_depth_over_ref_z_median | 0.113576 | 0.112007 | 0.207063 | 0.705249 |
| source_depth_over_ref_z_median | 0.128608 | 0.144181 | 0.207063 | 0.676243 |
| trajectory_scale_ratio | 0.150980 | 0.119887 | 0.410697 | 0.693379 |

Per-scene:

| scene | R23 eval scale | trajectory ratio | target/local z | source/ref z | target/ref z | source/local z | local/ref z | target/source depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.561127 | 0.493855 | 0.593979 | 0.677315 | 0.677315 | 0.586915 | 1.127166 | 1.012288 |
| scene0011_00 | 0.561406 | 0.572213 | 0.552688 | 0.584672 | 0.607600 | 0.533094 | 1.093874 | 0.978468 |
| scene0030_00 | 0.656649 | 0.926332 | 0.696137 | 0.751325 | 0.745886 | 0.721613 | 1.030412 | 0.980433 |
| scene0081_01 | 0.862760 | 0.889791 | 0.791689 | 0.721808 | 0.766125 | 0.716725 | 1.007092 | 0.972939 |
| scene0591_00 | 0.707645 | 0.584703 | 0.719900 | 0.646096 | 0.729319 | 0.643151 | 0.996841 | 1.120000 |

scene0030 hard-frame check:

| frame | trajectory ratio | traj absrel | target/local z | target/local absrel | source/ref z | source/ref absrel | target depth | local z | D4RT trans | pose trans |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0.926332 | 0.410697 | 0.704815 | 0.073352 | 0.769977 | 0.172586 | 2.366000 | 3.356908 | 0.176657 | 0.163643 |
| 120 | 1.716449 | 1.613954 | 0.749492 | 0.141389 | 0.748628 | 0.140074 | 2.531000 | 3.376956 | 0.236630 | 0.406164 |
| 150 | 0.776038 | 0.181816 | 0.695032 | 0.058454 | 0.743469 | 0.132217 | 2.792000 | 4.017079 | 0.272615 | 0.211559 |

判定:

- `target_depth_over_local_z_median` 明显强于 R27 trajectory ratio: mean absrel `0.046781` vs `0.150980`，max absrel `0.082377` vs `0.410697`。
- scene0030 的 hard failure 基本被 target/local-z scale 解释: scene median absrel `0.060137`，frame120 absrel `0.141389`，远低于 trajectory ratio 的 `1.613954`。
- `local_over_ref_z_median` 约 `1.0`，说明 `xyz_local` 与 `xyz_ref` 大多共享同一内部尺度；这解释了为什么 v22.8 简单 local/ref norm ratio 估不到 R23 scale。
- target/source depth ratio 也约 `1.0`，说明 scene0030 不是单纯 source 与 target 深度分布差异造成。
- 因此最强解释是: D4RT 输出存在 per-sample/per-frame depth scale ambiguity，训练 loss 的 mean-depth normalization 让绝对尺度弱约束；R23 的强效果来自补回了 target-depth/mean-depth 级别的尺度锚。
- 但本轮使用 ScanNet depth，因此仍不能进入 method table。下一步应尝试非 GT depth/scale proxy: 例如用 D4RT 自身预测的 local z 与 UV reprojection consistency、monocular depth prior、object/mask physical scale prior、或模型内部可恢复的 mean-depth proxy 去近似 `target_depth_over_local_z_median`。

Artifact:

- `Stream3D/tools/diagnose_v22_ref0_scale_convention.py`
- `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_frame_rows.*`
- `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_window_rows.*`
- `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_scene_summary.*`
- `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_candidate_errors.*`
- `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention.md`

## v22.13: ref0 intrinsics proxy diagnostic

目的: v22.12 指向 `target_depth_over_local_z_median`，但它使用 ScanNet depth，只能作为 upper-bound attribution。本轮检查 OpenD4RT 官方 `intrinsics_from_queries` 公式是否能从 D4RT predicted `xyz_local/xyz_ref` + `uv` 提供非 depth 的 metric scale proxy。该实验仍用 R23 eval-only scale 和 ScanNet intrinsics 做诊断对照，diagnostic-only。

代码修改:

- `Stream3D/tools/diagnose_v22_ref0_intrinsics_proxy.py`
  - 按 OpenD4RT `_estimate_intrinsics_params_from_predictions` 逻辑估计 per-frame `fx/fy`。
  - 同时输出 `local` / `ref` branch 的 focal geomean ratio、ScanNet-intrinsics reprojection error、query-derived intrinsics reprojection error。
  - 与 R23 eval-only ref0 scale 比较 absrel。
- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 query intrinsics estimator 的 uniform scale invariance 单测。

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 11 tests OK |

Probe5 candidate error vs R23 eval-only scale:

| candidate | mean absrel | median absrel | max absrel | mean candidate |
|---|---:|---:|---:|---:|
| local_fx_over_scannet_fx | 0.364440 | 0.378509 | 0.767552 | 0.883376 |
| local_fxy_over_scannet_fxy | 0.428334 | 0.504193 | 0.898481 | 0.926279 |
| local_fy_over_scannet_fy | 0.507383 | 0.517418 | 1.052878 | 0.975513 |
| ref_fxy_over_scannet_fxy | 0.566822 | 0.508222 | 0.855233 | 0.835982 |
| ref_fxy_over_local_fxy | 0.582795 | 0.423599 | 1.234153 | 0.914841 |

Per-scene:

| scene | R23 eval scale | ScanNet/local fxy | local/ScanNet fxy | local fxy | ScanNet fxy | local ScanNet reproj p90 | local pred-intr p90 | ref ScanNet reproj p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.561127 | 1.184775 | 0.844043 | 487.747422 | 577.870605 | 58.464093 | 7.347940 | 780.728892 |
| scene0011_00 | 0.561406 | 0.938246 | 1.065819 | 616.213695 | 578.159967 | 33.711921 | 10.507235 | 268.463152 |
| scene0030_00 | 0.656649 | 0.975672 | 1.024936 | 590.424947 | 576.060247 | 36.645796 | 7.086878 | 131.319775 |
| scene0081_01 | 0.862760 | 1.158068 | 0.863507 | 498.995252 | 577.870605 | 54.505246 | 10.439631 | 492.437685 |
| scene0591_00 | 0.707645 | 1.200353 | 0.833088 | 481.658274 | 578.159967 | 60.792617 | 6.002721 | 158.252875 |

scene0030 hard-frame check:

| frame | R23 scale | local/ScanNet fxy | local fx/ScanNet fx | ScanNet/local fxy | local ScanNet p90 | local pred-intr p90 | ref ScanNet p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0.656649 | 1.023584 | 0.915802 | 0.976959 | 33.771021 | 8.695748 | 128.015353 |
| 120 | 0.656649 | 1.005367 | 0.883582 | 0.994662 | 47.593972 | 28.959668 | 162.399082 |
| 150 | 0.656649 | 1.047867 | 0.936088 | 0.954319 | 40.536904 | 21.173376 | 361.735417 |

判定:

- OpenD4RT query-derived intrinsics 不能近似 R23 metric scale。最佳候选 `local_fx_over_scannet_fx` mean absrel `0.364440`，远差于 v22.12 `target_depth_over_local_z_median` 的 `0.046781`。
- scene0030 hard frame120 明确失败: `local_fxy_over_scannet_fxy=1.005367`，但 R23 scale 是 `0.656649`。
- `local_pred_intrinsics_reproj_error_px_p90` 通常只有 `6..29px`，明显低于 `ref` branch 的 ScanNet reprojection p90。这说明 `xyz_local` 的 `x/z` 投影关系内部较一致；但由于 focal estimation 对 uniform `xyz` scale 不敏感，它不能提供绝对 depth scale。
- 因此后续不应把 OpenD4RT `intrinsics_from_queries` 当成 metric-scale 修复主线；仍需寻找非 GT target-depth / mean-depth scale proxy，例如 monocular depth prior、模型 normalization 反变换、或 object/mask scale prior。

Artifact:

- `Stream3D/tools/diagnose_v22_ref0_intrinsics_proxy.py`
- `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_frame_rows.*`
- `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_window_rows.*`
- `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_scene_summary.*`
- `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_candidate_errors.*`
- `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy.md`

## v22.14: LoGeR geometry scale-proxy diagnostic

目的: v22.12 找到 `target_depth_over_local_z_median` 是最强 scale attribution 线索，但它使用 ScanNet depth，不能 method 化。本轮尝试用 LoGeR geometry backbone 的 local pointmap 作为不使用 ScanNet depth 的 pseudo-depth / pointmap scale proxy，并与 R23 eval-only ref0 scale 做对照。该实验 diagnostic-only。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- LoGeR inference 只使用 GPU `0`，显式 `CUDA_VISIBLE_DEVICES=0`。
- 诊断脚本和测试 CPU-only，显式 `CUDA_VISIBLE_DEVICES=""`。

代码修改:

- `Stream3D/tools/diagnose_v22_loger_scale_proxy.py`
  - 读取 LoGeR `.pt` 输出，在 D4RT target UV 位置采样 LoGeR local pointmap。
  - 输出非 GT scale candidates: `loger_z_over_d4rt_local_z_median`、`loger_z_over_d4rt_ref_z_median`、`loger_norm_over_d4rt_local_norm_median`、`loger_norm_over_d4rt_ref_norm_median`、`d4rt_local_z_over_loger_z_median`。
  - 输出 GT positive controls: `scannet_depth_over_loger_z_median`、`scannet_depth_over_d4rt_local_z_median`。
  - 所有结果与 R23 eval-only ref0 scale 比 absrel。
- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 LoGeR pointmap UV sampler 单测。
  - 新增 candidate rows 的 GT-control 标记单测。

LoGeR 4-frame inference:

| scene | output | status |
|---|---|---|
| scene0050_00 | `loger_scene0050_4f.pt` | ok |
| scene0011_00 | `loger_scene0011_4f.pt` | ok |
| scene0030_00 | `loger_scene0030_4f.pt` | ok |
| scene0081_01 | `loger_scene0081_4f.pt` | ok |
| scene0591_00 | `loger_scene0591_4f.pt` | ok |

验证:

| item | result |
|---|---|
| LoGeR scene0050 smoke | ok, `local_points=(4,378,504,3)` |
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 13 tests OK |

Probe5 candidate error vs R23 eval-only scale:

| candidate | uses ScanNet depth for proxy | mean absrel | median absrel | max absrel | mean candidate | scenes |
|---|---:|---:|---:|---:|---:|---:|
| loger_z_over_d4rt_ref_z_median | False | 0.730324 | 0.693623 | 0.884752 | 0.173463 | 5 |
| loger_norm_over_d4rt_ref_norm_median | False | 0.732677 | 0.692137 | 0.885836 | 0.171971 | 5 |
| loger_z_over_d4rt_local_z_median | False | 0.734930 | 0.691965 | 0.887910 | 0.170685 | 5 |
| loger_norm_over_d4rt_local_norm_median | False | 0.735012 | 0.690007 | 0.889936 | 0.170475 | 5 |
| d4rt_local_z_over_loger_z_median | False | 8.510500 | 8.955088 | 10.985711 | 6.388830 | 5 |
| scannet_depth_over_d4rt_local_z_median | True | 0.044534 | 0.035141 | 0.075514 | 0.681455 | 5 |
| scannet_depth_over_loger_z_median | True | 5.469452 | 4.859856 | 9.703976 | 4.507986 | 5 |

Per-scene:

| scene | R23 eval scale | loger/local z | loger/ref z | local/loger z | GT/local z | GT/loger z | frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.561127 | 0.179017 | 0.181363 | 5.586065 | 0.591627 | 3.288122 | 4 |
| scene0011_00 | 0.561406 | 0.151206 | 0.160306 | 6.613573 | 0.519012 | 3.426504 | 4 |
| scene0030_00 | 0.656649 | 0.202271 | 0.201182 | 4.943874 | 0.676458 | 3.343461 | 4 |
| scene0081_01 | 0.862760 | 0.096707 | 0.099431 | 10.340792 | 0.893079 | 9.234962 | 4 |
| scene0591_00 | 0.707645 | 0.224223 | 0.225030 | 4.459844 | 0.727100 | 3.246880 | 4 |

判定:

- LoGeR pointmap 不能直接作为 R23 metric scale proxy。最好的非 GT 候选 `loger_z_over_d4rt_ref_z_median` mean absrel `0.730324`，明显差于 v22.12 `target_depth_over_local_z_median` 的 `0.046781`。
- LoGeR/D4RT scale ratio 基本在 `0.17` 左右，而 R23 eval scale 是 `0.56..0.86`，不是一个小偏差。
- GT positive control `scannet_depth_over_d4rt_local_z_median` mean absrel `0.044534`，继续确认 “target depth / D4RT local z” 是真实 scale attribution 线索。
- `scannet_depth_over_loger_z_median` mean absrel `5.469452`，说明 LoGeR pointmap 自身也不是 ScanNet metric target-depth proxy。
- scene0081 是硬负例: `loger_z_over_d4rt_local_z_median=0.096707`，R23 scale `0.862760`，`GT/loger z=9.234962`。
- 本轮不生成 method row，不启动 Phase F。

Artifact:

- `Stream3D/tools/diagnose_v22_loger_scale_proxy.py`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_scene0050_smoke/*`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy.md`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_candidate_errors.*`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_scene_summary.*`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_frame_rows.*`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_window_rows.*`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_metadata.json`

## v22.15: OpenD4RT xyz loss scale-invariance diagnostic

目的: v22.12 找到 `target_depth_over_local_z_median` 是最强 scale attribution 线索，v22.13/v22.14 又排除了 query-derived intrinsics 和 LoGeR pointmap 这两条非 GT proxy。本轮直接检查 OpenD4RT 训练 loss 是否约束 metric scale。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮无 GPU inference；诊断和测试均 CPU-only，显式 `CUDA_VISIBLE_DEVICES=""`。

Source inspection:

- `Open-d4rt/src/losses/d4rt_loss.py`
  - `_xyz_preprocess` 对 pred/GT 分别按各自 mean abs-z normalize。
  - normalize 后再执行 `torch.sign(out) * torch.log1p(out.abs())`。
- `Open-d4rt/configs/train_effective.yaml`
  - `xyz_3d.normalize_by_mean_depth: true`
  - `xyz_3d.value_transform: sign_x_log1p_abs_x`
- `Open-d4rt/configs/model_effective.yaml`
  - heads 包括 `xyz_3d`、`uv_2d`、`visibility`、`displacement`、`normal`、`confidence`，未看到显式 metric scale / mean-depth head。
- `Open-d4rt/infer_track_3d.py`
  - inference 直接写 `pred_local["xyz_3d"]` / `pred_ref["xyz_3d"]`，没有看到 inverse-normalization metadata。

代码修改:

- 新增 `Stream3D/tools/diagnose_v22_loss_scale_invariance.py`
  - 从 `outputs/stream4d_debug_v22_local_xyz_probe5_r1` 读取 `xyz_local` / `uv_pred`。
  - 在 target UV 位置用 ScanNet depth + intrinsics 构造 GT camera-space XYZ。
  - 对 pred scale `0.25/0.5/1.0/2.0/4.0` 做 sweep。
  - 同时输出 metric-space L1 / z absrel 和 OpenD4RT loss-space L1。
  - Metadata 标记 `is_diagnostic_only=true`、`uses_scannet_depth_for_gt_xyz=true`、`forbidden_for_method_table=true`。
- 修改 `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 uniform pred scale 下 loss-space L1 不变的单测。
  - 新增 metric-space L1 会随 scale 改变的单测。

Probe5 diagnostic metadata:

| item | value |
|---|---:|
| frame rows | 80 |
| sweep rows | 400 |
| pred scales | `0.25,0.5,1.0,2.0,4.0` |
| loss signed-log range across pred scales mean | 0.000000 |
| loss signed-log range across pred scales max | 0.000000 |
| normalized loss range across pred scales mean | 0.000000 |
| normalized loss range across pred scales max | 0.000000 |
| metric L1 range across pred scales mean | 4.750760 |
| metric L1 range across pred scales max | 6.158282 |
| metric z absrel range across pred scales mean | 4.864648 |
| metric z absrel range across pred scales max | 6.728604 |

Scale sweep:

| pred scale | frames | loss L1 mean | normalized loss L1 mean | metric L1 mean | z absrel mean |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 80 | 0.034302 | 0.056277 | 0.606992 | 0.620367 |
| 0.50 | 80 | 0.034302 | 0.056277 | 0.253184 | 0.256145 |
| 1.00 | 80 | 0.034302 | 0.056277 | 0.520732 | 0.526143 |
| 2.00 | 80 | 0.034302 | 0.056277 | 1.995299 | 2.037693 |
| 4.00 | 80 | 0.034302 | 0.056277 | 4.967838 | 5.075384 |

Per-scene:

| scene | frames | anchors mean | loss range mean | metric range mean | metric best scale median | GT/pred mean-z median |
|---|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 16 | 255.187500 | 0.000000 | 4.963678 | 0.500000 | 0.578260 |
| scene0011_00 | 16 | 39.687500 | 0.000000 | 5.480789 | 0.500000 | 0.566638 |
| scene0030_00 | 16 | 285.750000 | 0.000000 | 4.918066 | 0.500000 | 0.677070 |
| scene0081_01 | 16 | 40.562500 | 0.000000 | 4.375589 | 1.000000 | 0.867373 |
| scene0591_00 | 16 | 170.125000 | 0.000000 | 4.015680 | 0.500000 | 0.727257 |

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 15 tests OK |
| full unittest discover | 78 tests OK, skipped 1 |

判定:

- OpenD4RT `xyz_3d` loss 对 uniform pred scale 不敏感。真实 probe5 上 loss-space 跨 pred-scale range 的 mean/max 都为 `0.0`。
- metric-space 同时明显随 scale 改变，说明这不是采样太少或计算退化。
- 因此 `xyz_3d` head/loss 本身不会给出唯一 metric scale；v22.12 的 `target_depth/local_z` 更像 inference-time 缺失的 scale attribution。
- 这解释了 v22.13 query-intrinsics 和 v22.14 LoGeR direct proxy 的负结果: 前者对 uniform scale 不敏感，后者没有独立 metric calibration。
- 本轮不生成 method row，不启动 Phase F。

Artifact:

- `Stream3D/tools/diagnose_v22_loss_scale_invariance.py`
- `Stream3D/tests/test_v22_direct_reconstruction.py`
- `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_invariance.md`
- `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_metadata.json`
- `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_frame_rows.*`
- `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_sweep_rows.*`
- `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_scene_summary.*`
- `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_by_pred_scale.*`

## v22.16: target-scale observability diagnostic

目的: v22.12 证明 `target_depth / D4RT local_z` 是当前最强 scale attribution，但它使用 ScanNet depth。v22.13/v22.14/v22.15 又排除了 query intrinsics、LoGeR pointmap 和 OpenD4RT loss 自带 metric scale。本轮继续测试: 这个 GT-only scale label 是否能由 D4RT 内部统计量在 leave-one-scene-out 设置下预测出来。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮无 GPU inference；诊断和测试均 CPU-only，显式 `CUDA_VISIBLE_DEVICES=""`。

代码修改:

- 新增 `Stream3D/tools/diagnose_v22_target_scale_observability.py`
  - 逐 frame 生成两个 label:
    - `target_depth_over_local_z_median`: ScanNet target depth / D4RT `xyz_local` abs-z median，使用 ScanNet depth。
    - `eval_ref0_depth_scale`: R23 eval-only ref0 scale，使用 ScanNet depth/pose。
  - 提取 27 个 D4RT-internal feature，包括 visibility/confidence、UV spread、pred local/ref z/norm、local/ref ratio、D4RT rigid translation/residual、source-frame spread。
  - 另加 2 个 ScanNet pose-control feature: `pose_translation_norm`、`trajectory_scale_ratio`，只作 diagnostic control。
  - 跑 global median LOO、all-feature linear LOO、D4RT+pose-control linear LOO、oracle scene median、以及所有单特征 LOO linear sweep。
  - 输出 frame/window/prediction/correlation/univariate/scene CSV+JSON+MD。
  - Metadata 标记 `is_diagnostic_only=true`、`forbidden_for_method_table=true`。
- 修改 `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 absrel summary、synthetic LOO linear predictor、Spearman tie handling 单测。

Probe5 metadata:

| item | value |
|---|---:|
| frame rows | 75 |
| window rows | 5 |
| D4RT internal features | 27 |
| pose diagnostic features | 29 |
| best D4RT univariate for target/local label | `rigid_residual_median` |
| best D4RT univariate target/local mean absrel | 0.129793 |
| best pose-control target/local mean absrel | 0.136832 |
| best D4RT univariate for R23 eval scale | `uv_x_std` |
| best D4RT univariate R23 eval mean absrel | 0.157355 |
| best pose-control R23 eval mean absrel | 0.192556 |

Predictor summary:

| predictor | label | mean absrel | median absrel | max absrel | uses pose feature |
|---|---|---:|---:|---:|---|
| global median LOO | eval ref0 depth scale | 0.179578 | 0.215069 | 0.294094 | False |
| linear LOO D4RT internal | eval ref0 depth scale | 0.316544 | 0.217751 | 1.212866 | False |
| linear LOO D4RT + pose diagnostic | eval ref0 depth scale | 0.374789 | 0.259574 | 1.793648 | True |
| oracle scene median | eval ref0 depth scale | 0.000000 | 0.000000 | 0.000000 | False |
| oracle scene median | target depth / local z | 0.056326 | 0.029889 | 0.451290 | False |
| global median LOO | target depth / local z | 0.160601 | 0.148082 | 0.452184 | False |
| linear LOO D4RT internal | target depth / local z | 0.466158 | 0.364515 | 2.336099 | False |
| linear LOO D4RT + pose diagnostic | target depth / local z | 0.619112 | 0.446028 | 2.756365 | True |

Best univariate LOO:

| label | feature | mean absrel | median absrel | max absrel | uses pose feature |
|---|---|---:|---:|---:|---|
| eval ref0 depth scale | `uv_x_std` | 0.157355 | 0.224958 | 0.318771 | False |
| eval ref0 depth scale | `pred_local_abs_z_median` | 0.158963 | 0.216945 | 0.346765 | False |
| eval ref0 depth scale | `local_over_ref_z_median` | 0.161262 | 0.206724 | 0.382397 | False |
| eval ref0 depth scale | `trajectory_scale_ratio` | 0.192556 | 0.223558 | 0.382371 | True |
| target depth / local z | `rigid_residual_median` | 0.129793 | 0.098340 | 0.403580 | False |
| target depth / local z | `local_over_ref_z_median` | 0.130900 | 0.094871 | 0.492785 | False |
| target depth / local z | `pred_ref_norm_median` | 0.139480 | 0.131980 | 0.425850 | False |
| target depth / local z | `trajectory_scale_ratio` | 0.136832 | 0.130683 | 0.459729 | True |

Per-scene:

| scene | frames | target/local scale med | target/local scale std | eval scale med | D4RT all-feature target absrel | pose-control target absrel | D4RT all-feature eval absrel |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | 15 | 0.552688 | 0.059716 | 0.561406 | 0.666638 | 0.638850 | 0.457999 |
| scene0030_00 | 15 | 0.696137 | 0.025571 | 0.656649 | 0.288845 | 0.400978 | 0.188945 |
| scene0050_00 | 15 | 0.593979 | 0.011217 | 0.561127 | 0.281819 | 0.888966 | 0.536035 |
| scene0081_01 | 15 | 0.791689 | 0.099616 | 0.862760 | 0.600985 | 0.704292 | 0.154847 |
| scene0591_00 | 15 | 0.719900 | 0.050971 | 0.707645 | 0.492503 | 0.462477 | 0.244892 |

Strongest correlations:

| label | feature | spearman | uses pose feature |
|---|---|---:|---|
| target depth / local z | `uv_y_std` | -0.507340 | False |
| eval ref0 depth scale | `local_over_ref_z_median` | -0.492555 | False |
| eval ref0 depth scale | `uv_bbox_area` | -0.479925 | False |
| eval ref0 depth scale | `confidence_mean` | -0.445530 | False |
| eval ref0 depth scale | `trajectory_scale_ratio` | 0.334467 | True |
| target depth / local z | `trajectory_scale_ratio` | 0.278065 | True |

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 18 tests OK |
| full unittest discover | 81 tests OK, skipped 1 |

判定:

- D4RT 内部统计有弱可观测信号，但没有形成可靠 scale anchor。最佳单特征对 `target_depth/local_z` 的 mean absrel 是 `0.129793`，好于 global median `0.160601`，但远差于 scene-oracle median `0.056326`，且 max absrel `0.403580`。
- 对 R23 eval scale，最佳 D4RT 单特征 `uv_x_std` mean absrel `0.157355`，只比 global median `0.179578` 小幅好，max absrel `0.318771`。
- 27 维 D4RT-internal linear LOO 明显过拟合: `target_depth/local_z` mean absrel `0.466158`，R23 eval scale mean absrel `0.316544`，都差于 global median。
- 加 ScanNet pose-control 后也没有改善，说明 R27 类 trajectory signal 在这个简单回归设置里不是稳定补充。
- 当前结论是: `target_depth/local_z` 在 scene 内相对稳定，但不能靠 visibility/confidence/uv spread/local-ref ratio/rigid residual 这些简单 D4RT 统计可靠恢复。
- 本轮不生成 method row，不启动 Phase F。

Artifact:

- `Stream3D/tools/diagnose_v22_target_scale_observability.py`
- `Stream3D/tests/test_v22_direct_reconstruction.py`
- `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability.md`
- `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_metadata.json`
- `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_frame_rows.*`
- `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_window_rows.*`
- `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_predictor_summary.*`
- `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_univariate_summary.*`
- `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_feature_correlations.*`
- `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_scene_summary.*`

## v22.17: OpenD4RT scale metadata audit

目的: v22.15 已确认 `xyz_3d` loss 对 uniform metric scale 不敏感，v22.16 又确认简单 D4RT 输出统计不能可靠预测 `target_depth/local_z`。本轮直接审计 OpenD4RT 训练配置、loss、模型 heads、官方 inference return、统一 schema 和当前 Stream3D carrier cache，确认是否存在一个已经保留但尚未使用的 mean-depth / metric-scale side channel。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮只读源码/config/cache；诊断和测试均 CPU-only，显式 `CUDA_VISIBLE_DEVICES=""`。

代码修改:

- 新增 `Stream3D/tools/diagnose_v22_opend4rt_scale_metadata.py`
  - 检查 `configs/train_effective.yaml` 是否启用 `loss.xyz_3d.normalize_by_mean_depth`。
  - 检查 `src/losses/d4rt_loss.py` 是否对 pred 和 GT 独立调用 `_xyz_preprocess` 并各自除以 mean abs-z。
  - 解析 `src/model/heads.py` 的输出 key，检查是否存在 scale/depth head。
  - 解析 `infer_track_3d.py` 的返回 key，检查 inference 是否返回 scale metadata。
  - 检查 `docs/data_schema.md` 是否声明 metric xyz，同时是否有 mean-depth / depth-scale / metric-scale 字段。
  - 扫描当前 v22 local carrier cache 的 `carriers_window*.npz` keys，检查是否有 scale-like 或 depth-like key。
  - 输出 source evidence / cache keys CSV+JSON+MD，metadata 标记 `diagnostic_only=true`、`method_result=false`。
- 修改 `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 scale-key detector、explicit scale head detector、independent loss normalization detector 单测。

Audit summary:

| item | value |
|---|---:|
| loss_config_enables_mean_depth_normalization | True |
| loss_normalizes_pred_and_gt_independently | True |
| model_has_explicit_scale_or_depth_head | False |
| inference_return_has_scale_metadata | False |
| schema_has_metric_xyz | True |
| schema_has_mean_depth_scale_field | False |
| cache_files_scanned | 5 |
| cache_files_with_scale_like_keys | 0 |
| cache_files_with_depth_like_keys | 0 |
| method_result | False |

Source evidence:

| check | result |
|---|---:|
| `train_effective.yaml` has `normalize_by_mean_depth: true` | True |
| `d4rt_loss.py` normalizes pred/GT independently | True |
| model output keys are only `confidence/displacement/normal/uv_2d/visibility/xyz_3d` | True |
| `infer_track_3d.py` return keys have no scale metadata | True |
| data schema declares metric xyz in meters | True |
| data schema has no mean-depth scale field | True |

Cache evidence:

| cache | scanned files | scale-like keys | depth-like keys |
|---|---:|---:|---:|
| `outputs/stream4d_debug_v22_local_xyz_probe5_r1` | 5 | 0 | 0 |

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 21 tests OK |
| full unittest discover | 84 tests OK, skipped 1 |

判定:

- 当前训练目标是 metric xyz，但启用的 `xyz_3d` loss 会在 loss space 内分别移除 pred 和 GT 的 mean-depth scale。
- 现有 OpenD4RT head 没有显式 scale/depth output；官方 inference return 也没有 scale metadata。
- 当前 Stream3D v22 carrier cache 也没有保存 mean-depth / depth-scale / metric-scale key。
- 因此没有“已落盘但未使用”的隐藏反归一化尺度可以直接挖出来；下一步如果继续做 method 修复，需要新增/学习/引入更强 scale anchor，例如训练/推理显式保留 scale、接入真实 metric pseudo-depth/VO，或做 self-supervised target-depth consistency。
- 本轮不生成 method row，不启动 Phase F。

Artifact:

- `Stream3D/tools/diagnose_v22_opend4rt_scale_metadata.py`
- `Stream3D/tests/test_v22_direct_reconstruction.py`
- `Stream3D/outputs/audit/v22_17_opend4rt_scale_metadata/opend4rt_scale_metadata_audit.md`
- `Stream3D/outputs/audit/v22_17_opend4rt_scale_metadata/opend4rt_scale_metadata_metadata.json`
- `Stream3D/outputs/audit/v22_17_opend4rt_scale_metadata/opend4rt_scale_metadata_source_evidence.*`
- `Stream3D/outputs/audit/v22_17_opend4rt_scale_metadata/opend4rt_scale_metadata_cache_keys.*`

## v22.18: self-supervised scale-sensitivity diagnostic

目的: v22.17 证明当前 inference/cache 没有可直接读出的 scale metadata。本轮继续检查一个自然后续方向: 只使用现有 D4RT `xyz_local` / `uv` / normalized depth shape / depth rank consistency，是否能对 uniform metric scale 产生非 GT 约束。GT depth 在本轮只作为 positive control，用来证明诊断本身能看见尺度变化。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮诊断和测试均 CPU-only，显式 `CUDA_VISIBLE_DEVICES=""`。

代码修改:

- 新增 `Stream3D/tools/diagnose_v22_self_supervised_scale_sensitivity.py`
  - 对 `outputs/stream4d_debug_v22_local_xyz_probe5_r1` 的 `xyz_local` 做 uniform scale sweep: `0.25/0.5/1.0/2.0/4.0`。
  - 统计 `uv_reprojection_median_px`、`normalized_z_l1`、`depth_rank_spearman` 和 `gt_depth_absrel` positive control。
  - 输出 frame/sweep/scene summary 的 CSV+JSON+MD，metadata 标记 `diagnostic_only=true`、`method_result=false`、`uses_scannet_depth_for_positive_control=true`。
- 修改 `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 synthetic scale-sweep 单测，确认 UV reprojection / normalized z / depth rank 对 uniform scale 不敏感，而 GT-depth AbsRel 能选出正确 scale。

Aggregate:

| item | value |
|---|---:|
| scene_count | 5 |
| frame_count | 80 |
| pred_scales | `0.25,0.5,1.0,2.0,4.0` |
| uv_reprojection_median_px_range_mean | 0.000000 |
| normalized_z_l1_range_mean | 0.000000 |
| depth_rank_spearman_range_mean | 0.000000 |
| gt_depth_absrel_range_mean | 4.864648 |
| gt_depth_absrel_min_mean | 0.210736 |
| gt_depth_absrel_at_scale_1_mean | 0.526143 |
| method_result | False |

Per-scene:

| scene | frames | uv range mean | normalized-z range mean | rank range mean | GT absrel range mean | GT absrel min mean | GT absrel scale1 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 16 | 0.000000 | 0.000000 | 0.000000 | 5.736074 | 0.162162 | 0.724559 |
| scene0011_00 | 16 | 0.000000 | 0.000000 | 0.000000 | 5.958859 | 0.134865 | 0.777020 |
| scene0030_00 | 16 | 0.000000 | 0.000000 | 0.000000 | 4.643735 | 0.266497 | 0.480311 |
| scene0081_01 | 16 | 0.000000 | 0.000000 | 0.000000 | 3.484639 | 0.193569 | 0.193569 |
| scene0591_00 | 16 | 0.000000 | 0.000000 | 0.000000 | 4.499931 | 0.296587 | 0.455255 |

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 22 tests OK |
| full unittest discover | 85 tests OK, skipped 1 |

判定:

- D4RT `uv` + `xyz_local` 的 pinhole reprojection consistency 对 uniform scale 完全不敏感，probe5 range mean/max 都是 `0.0`。
- normalized z shape 和 depth rank consistency 同样完全不敏感，probe5 range mean/max 都是 `0.0`。
- GT depth positive control 对同一 scale sweep 强敏感，`gt_depth_absrel_range_mean=4.864648`，说明诊断并非“看不见尺度”，而是当前无 GT consistency 本身没有尺度观测力。
- 因此“只用现有 D4RT UV/relative-shape/depth-order 做 self-supervised target-depth consistency”不能恢复 metric scale；仍需要外部 metric anchor、显式 learned scale/depth head、或训练/推理保留 normalization scale。
- 本轮不生成 method row，不启动 Phase F。

Artifact:

- `Stream3D/tools/diagnose_v22_self_supervised_scale_sensitivity.py`
- `Stream3D/tests/test_v22_direct_reconstruction.py`
- `Stream3D/outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5/self_supervised_scale_sensitivity.md`
- `Stream3D/outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5/self_supervised_scale_sensitivity_metadata.json`
- `Stream3D/outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5/self_supervised_scale_sensitivity_frame_rows.*`
- `Stream3D/outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5/self_supervised_scale_sensitivity_sweep_rows.*`
- `Stream3D/outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5/self_supervised_scale_sensitivity_scene_summary.*`

## v22.19: scale-anchor tolerance diagnostic

目的: v22.18 证明现有 self-supervised consistency 对 uniform scale 不敏感。本轮反过来从 R23 `xyz_ref0 + ScanNet ref0 pose + eval-only scale` upper-bound 出发，只扰动 fitted scale，量化一个未来非 GT scale anchor 需要多准确，才能保住 R23 的 point-cloud quality。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮诊断和测试均 CPU-only，显式 `CUDA_VISIBLE_DEVICES=""`。

代码修改:

- 新增 `Stream3D/tools/diagnose_v22_scale_anchor_tolerance.py`
  - 复用 v22 direct reconstruction benchmark 的 R23 point collection、ref0 pose scale fitting、point/depth metrics。
  - 对 R23 fitted scale 做 multiplier sweep: `0.5/0.75/0.9/1.0/1.1/1.25/1.5`。
  - 输出 scene rows / summary / metadata 的 CSV+JSON+MD。
  - metadata 标记 `diagnostic_only=true`、`method_result=false`、`uses_gt_depth_or_pose_for_oracle_scale=true`。
- 修改 `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 新增 scale-anchor tolerance helper 单测，确认 scale multiplier 不会 mutation 原始 fit。

Aggregate:

| item | value |
|---|---:|
| scene_count | 5 |
| scale_multipliers | `0.5,0.75,0.9,1.0,1.1,1.25,1.5` |
| oracle_fscore10_mean | 0.622259 |
| oracle_completeness20_mean | 0.888330 |
| oracle_depth_delta1_mean | 0.968216 |
| method_result | False |

Scale sweep:

| multiplier | rel scale error | F@10 mean | F@10 retention | comp@20 mean | outlier@20 mean | depth delta1 mean | depth AbsRel mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.500000 | 0.067882 | 0.109089 | 0.154312 | 0.826163 | 0.644346 | 0.227499 |
| 0.75 | 0.250000 | 0.113740 | 0.182785 | 0.333507 | 0.635708 | 0.872429 | 0.109861 |
| 0.9 | 0.100000 | 0.320440 | 0.514962 | 0.591724 | 0.259553 | 0.946160 | 0.067317 |
| 1.0 | 0.000000 | 0.622259 | 1.000000 | 0.888330 | 0.079411 | 0.968216 | 0.059158 |
| 1.1 | 0.100000 | 0.464646 | 0.746708 | 0.760754 | 0.285874 | 0.967814 | 0.064413 |
| 1.25 | 0.250000 | 0.183494 | 0.294884 | 0.502301 | 0.557986 | 0.960305 | 0.078098 |
| 1.5 | 0.500000 | 0.098092 | 0.157638 | 0.245881 | 0.697236 | 0.928020 | 0.097342 |

验证:

| item | result |
|---|---|
| py_compile | pass |
| `tests.test_v22_direct_reconstruction` | 23 tests OK |
| full unittest discover | 86 tests OK, skipped 1 |

判定:

- R23 upper-bound 对 scale anchor 误差非常敏感。10% under-scale 时 F@10 retention 只有 `0.514962`，10% over-scale 为 `0.746708`。
- 25% scale error 已基本破坏 R23 geometry: F@10 retention 只有 `0.182785/0.294884`，completeness@20 也降到 `0.333507/0.502301`。
- 这说明 v22.16 那种约 `0.13` mean AbsRel 的弱 D4RT-internal scale predictor 仍可能不够，尤其在 under-scale 或 hard-scene 上会把 point F-score 打掉很多。
- 下一步若要 method 化，scale anchor 目标不应只是“比 global median 略好”，而应尽量接近 R23/R12 GT target-depth attribution 的 `~0.05` mean AbsRel 量级，或通过 joint optimization 抵消 scale residual。
- 本轮不生成 method row，不启动 Phase F。

Artifact:

- `Stream3D/tools/diagnose_v22_scale_anchor_tolerance.py`
- `Stream3D/tests/test_v22_direct_reconstruction.py`
- `Stream3D/outputs/audit/v22_19_scale_anchor_tolerance_probe5/scale_anchor_tolerance.md`
- `Stream3D/outputs/audit/v22_19_scale_anchor_tolerance_probe5/scale_anchor_tolerance_metadata.json`
- `Stream3D/outputs/audit/v22_19_scale_anchor_tolerance_probe5/scale_anchor_tolerance_scene_rows.*`
- `Stream3D/outputs/audit/v22_19_scale_anchor_tolerance_probe5/scale_anchor_tolerance_summary.*`

## 是否启动 Phase D/E/F

| phase | status | reason |
|---|---|---|
| Phase D full provider replacement rerun | 未运行 full rerun | Phase C 已触发 Stop D；raw/self canonical alignment 失败。v22.7 证明 `xyz_ref0 + ref0 pose + eval-only scale` 可到 R23 F@10 `0.622259`，但 R22/R23 使用 ScanNet pose/depth，是 upper-bound 诊断。v22.8 证明简单 D4RT-internal local/ref scale proxy 不足；v22.9 R27 pose-trajectory scale 有明显改善但仍低于 R23 且使用 ScanNet target poses；v22.10 排除简单方向 bug；v22.11 排除简单 anchor-policy/trim 修复；v22.12 找到 GT-depth scale attribution 线索但仍非 method；v22.13 排除 OpenD4RT query-derived intrinsics 作为 metric-scale proxy；v22.14 排除 LoGeR pointmap 直接作为 metric target-depth proxy；v22.15 确认 OpenD4RT `xyz_3d` loss 本身对 uniform metric scale 不敏感；v22.16 证明简单 D4RT 内部统计只能弱预测 `target_depth/local_z`；v22.17 证明当前 OpenD4RT inference/cache 没有保存可直接复用的 mean-depth / metric-scale metadata；v22.18 证明现有 UV/relative-shape/depth-rank self-supervised consistency 对 uniform metric scale 不敏感；v22.19 进一步证明 scale anchor 误差 10%/25% 会显著降低 R23 F@10 retention，尚不能进入 provider/AP 主线。 |
| Phase E occupancy correction full repair | 未运行 full repair | v21.3 已有 D2r4/D5 occupancy/provider repair；v22 新证据表明直接 raw/self 坐标仍失败。仅验证 D5 persistent-ID real path、`xyz_local` diagnostic cache 和 loss-space transform hypothesis。 |
| Phase F method table | 未运行 | 无 frozen reportable method output；所有 eval-Sim3/direct/provider smoke 都是 diagnostic-only。 |

## Insight 与证据链

1. v22 修正了 self-stitch 指标语义。此前 `inlier_ratio` 接近 0.9 容易被误解；现在 abs010 true inlier mean 是 `0.884642`，但 abs005 只有 `0.395558`，说明是否“对齐好”强依赖尺度阈值，不能再用 quantile artifact 自证成功。

2. diagnostic 和 provider 现在共享同一套 matching helper。cached first pair 里 stable-id matches `21168`、mutual-UV matches `79`，provider D5 persistent smoke 里 stable-id matches `31`、mutual-UV matches `76`。这让后续差异更容易归因，而不是两个路径各自匹配。

3. direct reconstruction 改变了对 v21.3 的理解。v21.3 看到 eval-Sim3 provider AP 仍不够高，容易推断 D4RT local geometry/materialization 坏；v22 R4 probe5 F@10 `0.613669`、comp@20 `0.856899` 说明“D4RT sampled local geometry 经 GT Sim3 后能像一个不错的局部点云”。主要问题更集中在 raw/self canonical reference、scale/translation、cross-window metric alignment。

4. raw D4RT 坐标几乎没有可用重建质量。R0/R8 raw probe5 F@10 只有 `0.005726/0.005211`，4/5 scenes 的 F@10 为 `0.0`。这比单看 depth delta1 更严苛，因为点云整体坐标错位会让 Chamfer/F-score 直接崩掉。

5. self-stitch 当前没有把局部对齐转成全局坐标正确性。scene0050 R2/R3 F@10 `0.028903/0.029448`，低于 R1 raw `0.035874`；scale-normalized R3 没有改善。结合 Phase B `accumulated_scale_drift=0.265904`，说明 pairwise residual/inlier 不是唯一目标，还要解决 canonical reference anchoring。

6. dense/D2r4/D5 support 都没有自动修复 raw geometry。scene0050 R6/R7/R8/R9 F@10 都在 `0.025877..0.037612`，和 eval-Sim3 R4 的 `0.774526` 差距很大。D2r4 在 v21.3 eval-Sim3 + interior/outlier 下有 AP 提升，但 raw direct reconstruction 仍失败。

7. D5 persistent-ID 修复是必要但远不充分。真实 smoke 证明字段已落盘、provider 已读到 stable IDs；但 retention 只有 `6/506`，AP 为 NA，说明这只是 identity plumbing 的修复证据，不是 geometry 成功。

8. Stop D 是本轮最重要的实验决策。由于 eval-Sim3 direct 上界好、raw/self 极差，继续跑 Phase F semantic method table 会把 canonical alignment 的问题混进 object formation，不利于修复。下一轮应先让 raw/self direct reconstruction 指标有可解释提升。

9. 2026-06-11 camera-space continuation 进一步排除了“只差 world pose”的简单解释。R0/R8 probe5 camera F@10 只有 `0.005737/0.004404`，而 R4 eval-Sim3 转回 camera-space 后为 `0.173169`。这说明 raw D4RT xyz 在 per-frame camera-local point cloud 上也不匹配 GT depth 点云。

10. UV+Z pinhole 回投是一个小幅正修复，但不是解决方案。Probe5 R10/R11 camera F@10 为 `0.013578/0.010180`，比 R0/R8 raw 稍高，但仍远低于 R4 camera F@10 `0.173169`，且 camera outlier@20 仍约 `0.934531/0.936250`。

11. GT-only depth calibration 证明 `z` scale/shift 是真实 blocker。Probe5 R15 eval-only linear depth calibration 把 camera F@10/F@20 提到 `0.244729/0.499092`，比 R11 no-calibration `0.068136/0.181029` 强很多；但这一步用了 ScanNet depth 拟合，不能作为 method。

12. `z` 不是一个全局常数可修的问题。R15 per-scene linear scale/shift 差异很大: scene0050 `0.197679/1.372338`，scene0081 `0.679095/0.466488`，scene0591 `0.433586/0.418503`。后续若要 method 化，需要找到 D4RT-native 的归一化反变换或自监督尺度信号，而不是把 GT 拟合常数写进去。

13. 非 GT positive-z filter 只提供小幅改善。Probe5 R11 camera F@10 `0.068136 -> 0.071781`，仍远低于 GT-only linear upper-bound；scene0050 虽有更明显改善，但 per-instance coverage 降到 `0.583333`。

14. axis/sign convention typo 基本排除。Probe5 R11 最佳轴/符号搜索只把 median UV reprojection error 从 `371.276760px` 降到 `368.200725px`；scene0050 x/y swap 后仍超过 `600px`。这说明当前 `uv` 与 `xyz/z` 不一致不是简单的轴顺序或符号误读。

15. v22.3 找到了一个真实坐标分支问题。旧 cache 把 `pred_local["uv_2d"]` 和 `pred_ref["xyz_3d"]` 放在一起解释；OpenD4RT 官方同时输出 `tracks_xyz_local` 和 `tracks_xyz_ref0`。`xyz_local` 把 probe5 target reprojection median 从 `226.747204px` 降到 `30.753682px`，证明此前一部分 uv-z mismatch 来自 branch mismatch。

16. `xyz_local` 修复 correspondence，但没有修成 raw method。Probe5 raw `xyz_local` camera F@10/F@20 仍只有 `0.014329/0.082448`，raw UVZ F@10/F@20 只有 `0.022713/0.107783`。这说明 target branch 选对之后，D4RT metric depth/scale 或 per-window normalization 仍没有解决。

17. `xyz_local` 的 GT-only depth calibration 上界很强。Probe5 linear UVZ F@10/F@20 到 `0.355244/0.693759`，scene0050 到 `0.602850/0.923622`。这个上界比 v22.2 旧 R15 更说明“坐标分支 + depth scale/shift”是核心组合 blocker，但它仍依赖 ScanNet depth，不能进 method。

18. source self UV 可追踪，但 source depth 也有 scale 问题。Source self UV median 是 `4.379759px`，说明 source identity 并非完全坏；但 `xyz_local` source raw depth delta1 只有 `0.091690`，source linear depth delta1 也只有 `0.546804`。后续不应只在 target frame 上做表层过滤，还要查 D4RT 输出深度的归一化/单位来源。

19. OpenD4RT loss-space `signed_log1p` 是真实线索。训练配置对 `xyz_3d` 使用 `normalize_by_mean_depth` 和 `sign_x_log1p_abs_x`；v22.4 证明 `xyz_local + signed_log1p` 能把 probe5 raw camera F@10/F@20 从 `0.014329/0.082448` 提高到 `0.063919/0.233917`。

20. 但 `signed_log1p` 不是 method success。它把 target reprojection median 从 `30.753682px` 恶化到 `129.863558px`，且 F@10 仍远低于 GT-only calibrated upper bound。这更像 loss-space/canonical-scale 线索，不是可直接报告的后处理。

21. `signed_expm1` 基本排除。Probe5 `xyz_local + signed_expm1` 的 xyz F@10/F@20 只有 `0.000093/0.001296`，outlier@20 到 `0.996692`，说明简单把输出当 log-space 再反变换不成立。

22. 简单 visibility/confidence threshold 不是主修复。vc07/vc09 减少 anchors，但 `xyz_local + signed_log1p` 的 F@20 从 `0.233917` 降到 `0.194865/0.174127`，reprojection 也没有改善。

23. direct runner 接入后，`UV+Z + signed_log1p` 是当前最强非 GT camera-space branch。R19 camera F@10/F@20 为 `0.080797/0.272468`，outlier@20 为 `0.535702`，优于 R18 raw UV+Z 的 `0.022764/0.107910/0.799749`。

24. 但 world/canonical anchor 仍完全没解决。R16-R19 world F@10 全部是 `0.0`，说明 camera-space 变好不等于能落到 ScanNet world point universe；Phase D provider/AP 仍应后置。

25. v22.6 把 `xyz_local` 的上界单独测出来了。R20 eval-Sim3 F@10/F@20/comp@20 为 `0.408784/0.679166/0.650008`，说明 `xyz_local` 不是完全无结构；raw world F@10 为 `0.0` 的主因确实包含缺少非 GT world/canonical anchor。

26. 但 `xyz_local` 上界仍低于旧 `xyz_ref` 上界。R4 F@10/F@20/comp@20 是 `0.613669/0.868878/0.856899`，比 R20 高出 `0.204885/0.189712/0.206891`。这说明 `pred_ref`/canonical 分支不能被简单丢弃，后续要修的是 ref/canonical 到 ScanNet world 的自对齐，而不是只把主线切到 local。

27. `signed_log1p` 的作用域被收窄。它对 raw camera-space 是正向信号，但 R21 eval-Sim3 上界低于 R20: F@10/F@20 `0.281703/0.533511` vs `0.408784/0.679166`。因此它更适合作为 loss-space/scale clue，而不是直接替代点云坐标。

28. v22.7 进一步定位了 `xyz_ref` 的坐标约定。R22 只用 ScanNet ref0 pose 后 F@10/F@20 只有 `0.108860/0.280340`，说明问题不是“把 ref0 pose 乘上去”这么简单；但 R23 在同一 ref0 pose 上只补 eval-only scale 后达到 `0.622259/0.902864`，略高于 R4 full eval-Sim3。这说明 `xyz_ref` 更像 OpenD4RT ref0 coordinate，而不是任意 canonical blob。

29. scale 是当前最清楚的 ref/canonical blocker。R23 的 mean ref0 scale 是 `0.744080`，per-scene 从 `0.618160` 到 `1.086632`。后续 method 修复不能硬编码一个常数，而要找非 GT per-scene/per-window scale anchor。

30. R23 的提升不是只在 world metric 上碰巧成立。它的 camera-space F@10/F@20 达到 `0.492137/0.809117`，raw depth delta1 达到 `0.966551`，pred-to-GT median/p90 为 `0.063193/0.176206`。这说明 ref0-pose+scale 对齐同时修复了 world、camera、depth 三类 direct metric。

31. scene0081 是后续 scale/canonical 修复的重点负例。R23 在 scene0081 的 F@10 只有 `0.197514`，ref0 scale `1.086632`，residual p90 `0.802004`，明显差于其它 scene。下一轮如果只看 mean，容易掩盖这个场景的 ref0/scale failure。

32. v22.8 排除了一个最便宜的非 GT scale 近似。R24/R25/R26 只用 D4RT 内部 `xyz_local` / `xyz_ref` 统计估计 scale，但 probe5 F@10/F@20 只有 `0.051524/0.186488`、`0.056787/0.185555`、`0.027983/0.116290`，远低于 R23 `0.622259/0.902864`。

33. 简单 local/ref norm ratio 学不到 R23 的 metric scale。R24/R25/R26 mean scale 分别是 `0.977360/0.976609/1.057880`，基本贴近 1；R23 mean scale 是 `0.744080`，且 per-scene 从 `0.618160` 到 `1.086632`。这说明 D4RT 输出的 local/ref norm 本身大概率共享同一内部归一化，不足以恢复 ScanNet metric scale。

34. 下一步 scale 修复需要更强约束。候选方向应转向模型 normalization 反变换、ref0/camera trajectory consistency、source-target reprojection+depth consistency、或 object/mask scale prior；继续只拿 `xyz_local` / `xyz_ref` 做全局 norm/z ratio，预计不会逼近 R23。

35. v22.9 证明 pose trajectory baseline 是强 scale 线索。R27 不用 ScanNet depth scale fitting，只用 ScanNet pose trajectory 与 D4RT ref/local rigid translation ratio，就把 F@10/F@20 从 R22 `0.108860/0.280340` 提到 `0.218665/0.556108`，也显著强于 R25 `0.056787/0.185555`。

36. 但 trajectory scale 仍不稳定。R27 的 mean scale `0.693379` 接近 R23 mean `0.744080`，但 per-scene 误差会很大；scene0030 的 R27 scale `0.926332` 高于 R23 `0.671717`，导致 F@10 只有 `0.017300`。这说明 D4RT ref/local rigid translation baseline 可用作线索，但不能单独作为 reliable metric scale。

37. 现在最清楚的非 GT 方向不是 object/AP，而是替代 ScanNet pose trajectory。R27 的正信号提示: 如果后续能用非 GT VO / D4RT-internal camera motion / model normalization 近似 pose baseline，再和 ref/local rigid translation结合，可能比继续做点云 norm proxy 更接近 R23。

38. v22.10 排除了 R27 的低级方向错误解释。D4RT ref-to-local rigid rotation 和 ScanNet ref-to-target relative pose 的 median rotation error 在 probe5 上约 `1.72..2.36°`，scene0030 为 `1.930591°`；这不支持“transform 方向反了”作为主因。

39. scene0030 的核心坏点是 translation magnitude / ratio 随时间漂移。scene0030 frame60 ratio `0.665212` 几乎贴近 R23 eval scale `0.656649`，但 frame120 ratio 到 `1.716449`，把 median 拉到 `0.926332`。rigid residual p90 仍只有 `0.035809`，说明局部 rigid fit 好不等于 metric baseline 好。

40. 简单 quantile/filter 不是稳定修复。probe5 上 `ratio_median` 的 mean absrel vs R23 scale `0.150980` 仍最低；`ratio_low_residual_median` 虽把 max absrel 降到 `0.306391`，mean absrel 却升到 `0.183642`。因此不能仓促新增一个 R28 quantile estimator 当作进展。

41. v22.11 进一步排除了简单 anchor-policy 解释。`trim90` 把 probe5 mean absrel 从 `0.150980` 小降到 `0.139010`，但 scene0030 从 `0.410697` 变到 `0.415662`，hard frame120 仍是 ratio `1.715394`。这不是可报告修复。

42. source-frame subset 也不是答案。scene0030 的 `ref_source` / `target_source` / `nonref_source` ratio median 分别是 `0.944879/0.912749/0.925018`，仍远高于 R23 eval scale `0.656649`；`ref_source` 还非常稀，probe5 mean frame count 只有 `9.6`。

43. scene0030 frame120 是新的硬负例。所有 policy 下 frame120 ratio 都在 `1.686..1.763`，而 pose translation `0.406164` 明显大于 D4RT fitted translation `0.230382..0.240880`。后续应查 D4RT ref/local translation 的 motion-dependent compression 或 per-frame normalization，而不是继续调 carrier filter。

44. v22.12 找到目前最强的 scale attribution 线索。`target_depth_over_local_z_median` 对 R23 eval scale 的 probe5 mean absrel 只有 `0.046781`，明显优于 R27 trajectory ratio `0.150980`。这说明问题更像缺 target-depth / mean-depth scale anchor，而不是 ref/local motion ratio 本身不可解释。

45. scene0030 hard frame120 被 target/local-z scale 大幅解释。frame120 的 trajectory ratio absrel 是 `1.613954`，但 `target_depth_over_local_z_median` absrel 只有 `0.141389`。这把 blocker 从“D4RT motion fitting坏”进一步收窄到“raw D4RT 坐标缺少正确尺度反归一化”。

46. `xyz_local` 与 `xyz_ref` 很可能共享同一内部尺度。v22.12 的 local/ref z median 在 probe5 各 scene 约 `0.996841..1.127166`，不能恢复 R23 scale；这与 v22.8 的 local/ref norm proxy 失败一致。

47. v22.13 排除了 OpenD4RT query-derived intrinsics 作为 metric-scale proxy。最佳候选 `local_fx_over_scannet_fx` 的 mean absrel vs R23 scale 是 `0.364440`，`local_fxy_over_scannet_fxy` 是 `0.428334`，远差于 v22.12 `target_depth_over_local_z_median` 的 `0.046781`。

48. intrinsics proxy 更像 `x/z` 投影一致性诊断，而不是绝对尺度诊断。scene0030 frame120 的 `local_fxy_over_scannet_fxy=1.005367`，但 R23 eval scale 是 `0.656649`；同时 local pred-intrinsics reprojection p90 为 `28.959668px`，比 ref branch ScanNet p90 `162.399082px` 好。这说明 local branch projection relation 可解释，但 uniform `xyz` scale 仍不可见。

49. 后续不应把 `intrinsics_from_queries` 当成主要修复线。它能帮助确认 branch/camera convention，但不能替代 target-depth / mean-depth scale anchor。下一步应转向非 GT depth proxy、模型 normalization 反变换或物体/场景尺度先验。

50. v22.14 排除了 LoGeR pointmap 直接作为 metric target-depth scale proxy。最好的非 GT 候选 `loger_z_over_d4rt_ref_z_median` mean absrel vs R23 scale 是 `0.730324`，`loger_z_over_d4rt_local_z_median` 是 `0.734930`，比 v22.12 GT-depth attribution 的 `0.046781` 差一个量级。

51. GT positive control 再次确认 `target_depth / D4RT local_z` 是当前最强 attribution 线索。`scannet_depth_over_d4rt_local_z_median` mean absrel `0.044534`，与 v22.12 `target_depth_over_local_z_median=0.046781` 一致；这说明新脚本和采样流程本身没有把尺度关系破坏掉。

52. scene0081 是 pseudo-depth scale proxy 的硬负例。LoGeR/D4RT local z 比值只有 `0.096707`，R23 scale 是 `0.862760`，`GT/loger z=9.234962`。这说明简单 monocular/pointmap prior 需要额外 metric calibration，不能直接接进 method。

53. v22.15 确认 OpenD4RT `xyz_3d` loss 本身不约束 uniform metric scale。source 中 pred 和 GT 分别按各自 mean abs-z normalize，再做 signed-log transform；这个结构会把 uniform pred scale 完全抵消。

54. 真实 probe5 sweep 与理论一致。pred scale 从 `0.25` 到 `4.0` 时，80 个 frame row 的 `loss_l1_signed_log_range_across_pred_scales_mean/max` 都是 `0.0`，但 `metric_l1_range_across_pred_scales_mean=4.750760`、`metric_z_absrel_range_across_pred_scales_mean=4.864648`。

55. 因此 v22.12 的 `target_depth/local_z` 不是一个隐藏在 `xyz_3d` output 里已经可直接恢复的 scale head，而是缺失的 inference-time metric scale attribution。下一步必须找外部/自监督/先验 scale anchor，或改训练/推理让 mean-depth scale 可恢复。

56. v22.16 证明 D4RT 内部简单统计对 `target_depth/local_z` 只有弱可观测性。最佳单特征 `rigid_residual_median` mean absrel `0.129793`，比 global median `0.160601` 好一点，但远差于 scene-oracle median `0.056326`。

57. 多特征线性组合不是免费午餐。27 个 D4RT-internal features 的 LOO linear 对 `target_depth/local_z` mean absrel 反而到 `0.466158`，对 R23 eval scale 到 `0.316544`，说明小样本下简单回归会过拟合，不能当成 R28 修复方向。

58. ScanNet pose-control 也没有把 observability 救回来。加入 `pose_translation_norm` / `trajectory_scale_ratio` 后，对 `target_depth/local_z` mean absrel 为 `0.619112`，对 R23 eval scale 为 `0.374789`；这和 v22.10/v22.11 的结论一致，trajectory signal 有诊断价值但不稳定。

59. v22.17 确认当前 OpenD4RT path 没有隐藏的 scale metadata 可直接复用。训练配置启用 `normalize_by_mean_depth: true`，loss 对 pred/GT 各自 mean abs-z 归一化；但 model head 只有 `xyz_3d/uv_2d/visibility/displacement/normal/confidence`，没有 scale/depth head。

60. 官方 inference 和 Stream3D carrier cache 都没有保存反归一化尺度。`infer_track_3d.py` 返回 `tracks_xyz_local/tracks_xyz_ref0/tracks_uv_norm/tracks_visibility/tracks_confidence/...`，没有 scale metadata；v22 local probe5 cache 扫描 5 个 `carriers_window*.npz`，scale-like keys 和 depth-like keys 都是 `0`。

61. 因此下一步不能指望“把已有字段读出来”就恢复 metric scale。v22.17 把修复方向从 artifact mining 收窄到新增/学习/引入 scale anchor: 显式保留训练 normalization scale、真正 metric pseudo-depth/VO，或 self-supervised target-depth consistency。

62. v22.18 进一步证明，现有 D4RT 输出里那些最自然的 self-supervised consistency 也没有 uniform-scale 观测力。Probe5 上 `uv_reprojection_median_px_range_mean=0.0`、`normalized_z_l1_range_mean=0.0`、`depth_rank_spearman_range_mean=0.0`。

63. GT-depth positive control 对同一 scale sweep 很敏感，`gt_depth_absrel_range_mean=4.864648`，而 scale1 mean AbsRel 为 `0.526143`、best-scale mean AbsRel 为 `0.210736`。这说明诊断没有失效；失效的是“只靠 UV/relative-shape/rank consistency 恢复 metric scale”的假设。

64. 因此 self-supervised target-depth consistency 如果要继续做，必须引入额外 metric 信息或学习目标，例如外部 metric pseudo-depth/VO、明确的 scale/depth head、或训练时保留 normalization scale。把现有 D4RT UV + normalized shape/rank objective 直接接成 R28 不会解决 Stop D。

65. v22.19 量化了 scale anchor 的容错范围。R23 oracle F@10 是 `0.622259`；scale 低估 10% 时 F@10 retention 只有 `0.514962`，高估 10% 时也只有 `0.746708`。

66. 25% 级别的 scale error 会基本破坏 R23 upper-bound。scale multiplier `0.75/1.25` 的 F@10 retention 分别只有 `0.182785/0.294884`，completeness@20 也降到 `0.333507/0.502301`。

67. 因此 v22.16 的弱 scale predictor 不能只看 mean AbsRel “略好于 global median”。如果 scale error 仍在 10% 到 25% 区间，direct reconstruction 的 point F-score 会明显掉；下一步 scale anchor 应向 v22.12/R23 GT attribution 的 `~0.05` mean AbsRel 量级靠近，或者引入 joint correction 抵消 residual。

## 必答问题

1. v22 是否得到可报告方法成功: `False`。
2. Phase A 是否通过: `True`。
3. Phase B self-stitch rectification 是否完成: `True`。
4. 是否修复 quantile-defined inlier ratio: `True`。
5. diagnostic/provider matching 是否统一: `True`。
6. D5 persistent tube id 是否实现并真实 checkpoint smoke 验证: `True` for plumbing；`False` for method success。
7. Phase C direct reconstruction 是否完成: `True`。
8. Stop C 是否触发: `False`，R4 eval-Sim3 probe5 F@10/comp@20/delta1 为 `0.613669/0.856899/0.969155`。
9. Stop D 是否触发: `True`，R0/R8 raw probe5 F@10 只有 `0.005726/0.005211`，scene0050 R2/R3 self F@10 只有 `0.028903/0.029448`。
10. camera-space continuation 是否证明只差 world pose: `False`。R0/R8 probe5 camera F@10 只有 `0.005737/0.004404`。
11. UV+Z pinhole 回投是否足以修复 raw geometry: `False`。R10/R11 probe5 camera F@10 只有 `0.013578/0.010180`。
12. GT-only depth calibration 是否证明 z scale/shift 是 blocker: `True`。Probe5 R15 eval-only linear camera F@10/F@20 为 `0.244729/0.499092`。
13. GT-only depth calibration 是否可报告为 method: `False`。R12-R15 使用 ScanNet depth 拟合 z scale/shift，diagnostic-only / forbidden-for-method-table。
14. positive-z filter 是否足以修复 raw geometry: `False`。Probe5 R11 camera F@10 只到 `0.071781`。
15. axis/sign convention search 是否找到简单坐标解释 bug: `False`。Probe5 R11 best median reprojection error 仍 `368.200725px`。
16. 是否发现旧 cache 坐标分支混用: `True`。旧 `uv_pred` 来自 `pred_local`，旧 `xyz_ref` 来自 `pred_ref`；v22.3 已保存并诊断 `xyz_local`。
17. `xyz_local` 是否足以形成 method success: `False`。Probe5 raw `xyz_local` camera F@10/F@20 为 `0.014329/0.082448`，raw UVZ F@10/F@20 为 `0.022713/0.107783`。
18. `xyz_local` eval-only calibration 是否有上界: `True`。Probe5 linear UVZ F@10/F@20 为 `0.355244/0.693759`，但使用 ScanNet depth 拟合，不能作为 method。
19. OpenD4RT loss-space `signed_log1p` 是否是正向信号: `True`。Probe5 `xyz_local` raw camera F@10/F@20 从 `0.014329/0.082448` 提升到 `0.063919/0.233917`。
20. `signed_log1p` 是否足以形成 method success: `False`。Probe5 reprojection median 恶化到 `129.863558px`，F@10 仍低，不能进 method table。
21. visibility/confidence filtering 是否足以修复: `False`。vc07/vc09 的 `xyz_local + signed_log1p` F@20 降到 `0.194865/0.174127`。
22. v22.5 direct branch 是否改善 camera-space: `True`。R19 camera F@10/F@20 为 `0.080797/0.272468`，outlier@20 为 `0.535702`。
23. v22.5 direct branch 是否解决 world/canonical: `False`。R16-R19 world F@10 全部为 `0.0`。
24. `xyz_local` eval-Sim3 upper-bound 是否证明 local 点完全坏: `False`。R20 F@10/F@20/comp@20 为 `0.408784/0.679166/0.650008`，说明 local 点有中等上界。
25. R20 是否追上旧 R4 upper-bound: `False`。R4 F@10/F@20/comp@20 为 `0.613669/0.868878/0.856899`，仍显著更高。
26. R21 是否证明 `signed_log1p` 可作为 eval-Sim3 主坐标: `False`。R21 F@10/F@20 为 `0.281703/0.533511`，低于 R20。
27. R22 是否证明只需 ref0 pose 就能修 raw geometry: `False`。R22 F@10/F@20 只有 `0.108860/0.280340`，raw depth delta1 `0.111773`。
28. R23 是否证明 ref0 pose + scale 是强 upper-bound: `True`。R23 F@10/F@20/comp@20 为 `0.622259/0.902864/0.888330`，略高于 R4 full eval-Sim3。
29. R23 是否可报告为 method: `False`。R22/R23 使用 ScanNet pose；R23 还用 ScanNet depth/pose anchors 拟合 scale。
30. ref0 scale 是否可硬编码为单常数: `False`。R23 per-scene scale 从 `0.618160` 到 `1.086632`，scene0081 residual p90 到 `0.802004`。
31. R24/R25/R26 是否找到可用非 GT scale proxy: `False`。R24/R25/R26 F@10/F@20 只有 `0.051524/0.186488`、`0.056787/0.185555`、`0.027983/0.116290`，远低于 R23。
32. R24/R25/R26 是否可报告为 method: `False`。三者 scale 不用 ScanNet depth/pose anchors，但仍使用 ScanNet ref0 pose，只能作为 diagnostic。
33. R27 是否证明 pose trajectory scale 是强诊断线索: `True`。R27 F@10/F@20 为 `0.218665/0.556108`，明显高于 R22 和 R25。
34. R27 是否可报告为 method: `False`。R27 不用 ScanNet depth 拟合 scale，但使用 ScanNet target poses 计算 trajectory baseline。
35. R27 是否足以追上 R23: `False`。R27 F@10/F@20 仍低于 R23 `0.622259/0.902864`，scene0030 失败明显。
36. v22.10 是否证明 R27 失败来自 transform 方向反了: `False`。probe5 ref-to-target rotation error median 约 `1.72..2.36°`，scene0030 为 `1.930591°`。
37. v22.10 是否找到稳定优于 R27 median 的 scale estimator: `False`。`ratio_median` 的 mean absrel vs R23 eval scale 为 `0.150980`，仍低于 low-residual / weighted / quantile 候选。
38. v22.11 是否证明 carrier policy / residual trimming 能修 R27: `False`。最佳 mean policy `vc05_trim90` 只到 mean absrel `0.139010`，scene0030 仍 `0.415662`，frame120 仍 `1.715394`。
39. v22.11 是否值得新增 R28 benchmark row: `False`。policy sweep 没有稳定修复 scene0030，也没有形成 method 可用非 GT scale anchor。
40. v22.12 是否证明 target-depth / local-z scale 是强 attribution 线索: `True`。`target_depth_over_local_z_median` 的 mean absrel vs R23 eval scale 为 `0.046781`，scene0030 为 `0.060137`。
41. v22.12 是否可报告为 method: `False`。该 scale 使用 ScanNet target/source depth，只能作为 upper-bound/attribution diagnostic。
42. v22.13 是否证明 OpenD4RT query-derived intrinsics 能恢复 R23 metric scale: `False`。最佳候选 `local_fx_over_scannet_fx` mean absrel `0.364440`，明显差于 v22.12 target/local-z `0.046781`。
43. v22.13 是否可报告为 method: `False`。该诊断使用 ScanNet intrinsics 和 R23 eval-only scale 作为对照，且没有生成 reconstruction/provider/AP method row。
44. v22.14 是否证明 LoGeR pointmap 能恢复 R23 metric scale: `False`。最佳非 GT 候选 `loger_z_over_d4rt_ref_z_median` mean absrel `0.730324`，明显差于 v22.12 target/local-z `0.046781`。
45. v22.14 是否可报告为 method: `False`。LoGeR 只作为 pseudo-depth/pointmap scale diagnostic；结果没有生成 reconstruction/provider/AP method row，且 GT controls 只用于对照。
46. v22.15 是否证明 OpenD4RT `xyz_3d` loss 能约束 metric scale: `False`。pred/GT 分别 mean-depth normalize 后，uniform pred scale 在 loss-space 完全不可见；probe5 sweep 的 loss range across scales mean/max 都为 `0.0`。
47. v22.15 是否可报告为 method: `False`。该诊断用 ScanNet depth 构造 GT camera XYZ，只验证 loss 机制和 scale attribution，不生成 reconstruction/provider/AP method row。
48. v22.16 是否证明 D4RT 内部统计能可靠预测 `target_depth/local_z` scale: `False`。最佳单特征 mean absrel `0.129793` 仍远差于 scene-oracle `0.056326`，全特征 LOO 还过拟合到 `0.466158`。
49. v22.16 是否可报告为 method: `False`。该诊断使用 ScanNet depth/R23 eval scale 作 label，ScanNet pose 也只作为 control feature，没有生成 reconstruction/provider/AP method row。
50. v22.17 是否发现当前 OpenD4RT inference/cache 中已有可复用 scale metadata: `False`。model/inference/cache 都没有 mean-depth / metric-scale side channel；5 个 carrier cache 文件 scale-like/depth-like key 均为 `0`。
51. v22.17 是否可报告为 method: `False`。该轮是 source/config/cache audit，没有生成 reconstruction/provider/AP method row。
52. v22.18 是否证明现有 self-supervised consistency 能恢复 metric scale: `False`。UV reprojection / normalized z shape / depth rank consistency 对 uniform scale 的 range mean 都是 `0.0`。
53. v22.18 是否可报告为 method: `False`。该轮用 ScanNet depth 只作 positive control，没有生成 reconstruction/provider/AP method row。
54. v22.19 是否证明弱 scale predictor 已足够: `False`。R23 scale 偏 10% 时 F@10 retention 只有 `0.514962/0.746708`，偏 25% 时只有 `0.182785/0.294884`；v22.16 约 `0.13` mean AbsRel 的弱内部 predictor 仍可能明显伤害 point F-score。
55. v22.19 是否可报告为 method: `False`。该轮使用 R23 oracle scale 和 ScanNet ref0 pose/depth/pose anchors 做 tolerance diagnostic，没有生成 reconstruction/provider/AP method row。
56. 是否运行 Phase F 或生成 method table: `False`。
57. 是否有 GT/RGB-D/pose/mesh 泄漏到 method result: 本轮没有 method result。eval-Sim3 rows 明确是 diagnostic-only；R12-R15 depth calibration、v22.3/v22.4 median/linear UVZ calibration、R20/R21 scene Sim3、R22 ref0 pose、R23 ref0 pose + scale、R24/R25/R26 ref0 pose diagnostic、R27 pose trajectory diagnostic、v22.10 trajectory consistency diagnostic、v22.11 policy sweep diagnostic、v22.12 scale-convention diagnostic、v22.13 intrinsics proxy diagnostic、v22.14 LoGeR scale-proxy diagnostic、v22.15 loss scale-invariance diagnostic、v22.16 target-scale observability diagnostic、v22.17 OpenD4RT scale metadata audit、v22.18 self-supervised scale-sensitivity diagnostic、v22.19 scale-anchor tolerance diagnostic 都不能进入 method；direct reconstruction 使用 ScanNet GT 只作 evaluation/diagnostic。

## 后续修复方向

1. 优先把 direct reconstruction 作为下一轮 scale/canonical/local-coordinate 修复入口。目标不是先涨 AP，而是让 R0/R2/R3/R8/R10/R11/R16-R19 raw/self 的 world/camera F@10、completeness@20、outlier@20 接近 R23 upper-bound 的某个稳定比例。

2. 围绕 R23 找非 GT ref0 pose/scale anchor。当前证据不再支持“只差 world pose”这个单一解释: R22 只用 ref0 pose 仍失败，R23 只补 scale 就大幅超过 R4；v22.8 又证明简单 `xyz_local` / `xyz_ref` norm/z ratio 不能近似这个 scale。下一步应调查 `xyz_ref0` 如何用于 canonical/worldtrack、OpenD4RT demo 的 scale-only alignment 能否被非 GT 信号近似、是否存在模型内部归一化反变换，以及 `xyz_local` 和 `xyz_ref` 如何通过 reprojection/depth consistency 互相约束。

3. 优先查 `z` 的来源和反归一化。R12-R15 证明 GT-only linear scale/shift 有强上界，但跨 scene 的 scale/shift 差异很大，不能硬编码常数。下一步应追 OpenD4RT 输出里 depth/3D point 是否有内部 normalization、crop/resize、canonical scale、或 per-window reference scale。

4. 针对 depth/uv 质量做过滤和校准。R10/R11、v22.3 和 v22.4 说明用正确 branch 的 uv+z 仍不够；positive-z filter 不足；axis/sign search 也没有救回旧 branch；简单 visibility/confidence sweep 也是负修复。后续应优先找非 GT 的 per-window normalization、source-target consistency、mask/object scale prior，而不是继续只收紧阈值。

5. D5 persistent-ID 需要提高 retention。真实 smoke retention 只有 `6/506`；如果继续 D5，应先诊断 persistent matching 半径、overlap frame validity、warmstart accepted tubes 的 metric consistency，再扩到 probe5。

6. provider/AP 仍应作为后置 gate。v22.5 已经让 camera-space 有改善，v22.6 已经证明 local upper-bound 存在，v22.7 已经证明 ref0 pose + scale upper-bound 很强，v22.8 已经排除简单 local/ref scale proxy；但 raw/self world F@10 仍没有可报告改善。只有 direct raw/self world/canonical reconstruction 明显改善后，再跑 full provider replacement 和 semantic object formation，否则 AP 失败会继续混合 geometry 和 object formation 两类问题。

7. 不把 `signed_log1p` 当作唯一主线。它提升 raw camera-space，但 R21 eval-Sim3 upper-bound 低于 R20；后续应把它作为 scale/loss-space clue，而不是直接替代 geometry 坐标。

8. scene0081 要单列成 scale/canonical 负例。R23 在该 scene 仍只有 F@10 `0.197514`、residual p90 `0.802004`；如果下一步方法只在其它 4 个 scene 上改善，不能视为整体修复。

9. 不再优先尝试单纯 global norm/z scale proxy。R24/R25/R26 已经说明这类 proxy 容易估到接近 1 的内部尺度，却不恢复 ScanNet metric scale；后续应把 scale 与 source-target consistency、camera/ref0 trajectory、或模型 normalization 反变换绑定起来验证。

10. 把 R27 作为下一步非 GT 轨迹尺度目标。后续可以尝试用非 GT VO、D4RT 自身 ref/local motion consistency、或 OpenD4RT worldtrack/demo 中的 camera-motion 约定来替代 ScanNet pose baseline；但必须用 scene0030 作为硬负例检查稳定性。

11. v22.10 后不要优先做简单 R28 quantile/filter。下一步更应该查 translation magnitude 漂移的来源: D4RT `xyz_ref/xyz_local` 是否有 per-frame scale normalization，ref/local rigid translation 是否受 carrier subset / motion magnitude bias 影响，scene0030 frame120 这种 pose baseline 增大但 D4RT translation 不同步增大的情况是否普遍。

12. v22.11 后也不要优先做 anchor-filter R28。confidence/source-frame/residual trimming 已经基本排除，下一步应查模型输出的 per-frame scale convention: `normalize_by_mean_depth` 的反归一化是否缺失，`xyz_ref`/`xyz_local` 是否共享某个 clip-level/crop-level depth scale，或者 D4RT rigid translation 是否存在 motion-dependent compression。

13. v22.12 后下一步应追非 GT 版本的 `target_depth_over_local_z_median`。具体说，不要把 ScanNet depth scale 写进 method；应测试 D4RT 自身 predicted local-z consistency、monocular depth prior、mask/object scale prior、或 OpenD4RT 内部 camera/intrinsics head 是否能提供接近 target-depth 的 scale proxy。目标是逼近 v22.12 的 mean absrel `0.046781`，而不是继续调 R27 trajectory filter。

14. v22.13 后暂时下调 OpenD4RT intrinsics proxy 优先级。query-derived focal ratio 不能恢复 metric scale；它可作为 branch/camera convention sanity check，但下一步更应追 monocular/pseudo-depth target scale、训练 normalization 反归一化、或物体/房间尺度先验。

15. v22.14 后不要把 LoGeR local pointmap 当作直接 target-depth scale anchor。它可以继续作为相对几何/appearance-conditioned prior 的输入候选，但若要用于 scale，必须先有独立 metric calibration 或与 D4RT/scene/object constraints 联合估计；直接用 LoGeR/D4RT z 或 norm ratio 已经是负证据。

16. v22.15 后不要期待 trained `xyz_3d` head alone 自动恢复 metric scale。OpenD4RT loss 已证明对 uniform scale 不敏感；下一步应找外部/自监督 scale anchor，或修改训练/推理以输出或保留 mean-depth scale / normalization metadata。

17. v22.16 后不要优先做“简单 D4RT feature regression scale”。单特征只有弱改善，全特征 LOO 明显过拟合；下一步应转向更强约束: 训练/推理保留 mean-depth scale、引入真正 metric pseudo-depth/VO、或利用 target-depth consistency 的自监督目标，而不是继续堆 visibility/confidence/UV spread/local-ref ratio 的线性组合。

18. v22.17 后也不要继续从现有 OpenD4RT/cache 字段里找隐藏 scale。当前没有可读 side channel；下一步若走 mean-depth 反归一化，应改训练/推理接口显式输出或保留 scale，或者引入外部 metric scale source 并用 v22.12/v22.16 的 target/local label 作 diagnostic 对照。

19. v22.18 后不要把“现有 UV/relative-depth/rank consistency”直接当作自监督 metric-scale objective。它们已经被 uniform scale sweep 证明不敏感；下一步如果继续 self-supervised 线，应加入真正 metric anchor，例如 calibrated pseudo-depth/VO、learned scale head、或训练时显式输出 normalization scale。

20. v22.19 后 scale anchor 目标要按 direct F-score 容忍度来定，而不是只看 scale regression 的平均误差。10% 误差已经明显降低 R23 F@10 retention，25% 误差基本破坏 upper-bound；后续候选 anchor 应优先报告 scale AbsRel 分布、R23 perturbation 等价 retention、以及 scene0030/scene0081 hard-scene 表现。

## 最终验证

| item | result |
|---|---|
| final py_compile | pass |
| final unittest discover | 86 tests OK, skipped 1 |
| note on base conda unittest | failed because base conda lacks `torch`; canonical `envs/loger` full test passed |
| note on `CUDA_VISIBLE_DEVICES=0` unittest | failed 1 existing optional-GPU env test expecting empty/6/7; CPU-only rerun with `CUDA_VISIBLE_DEVICES=""` passed |
| Phase C direct benchmark artifacts | present |
| D5 persistent-ID smoke artifacts | present |
| camera-space continuation artifacts | present |
| z/uv/depth-calibration continuation artifacts | present |
| xyz_local/local-vs-ref continuation artifacts | present |
| loss-space xyz transform continuation artifacts | present |
| signed_log1p direct branch artifacts | present |
| xyz_local eval-Sim3 upper-bound artifacts | present |
| xyz_ref0 ref0-pose/scale upper-bound artifacts | present |
| ref0 local/ref scale proxy artifacts | present |
| ref0 pose-trajectory scale artifacts | present |
| ref0 trajectory consistency artifacts | present |
| ref0 trajectory policy sweep artifacts | present |
| ref0 scale-convention artifacts | present |
| ref0 intrinsics proxy artifacts | present |
| LoGeR scale-proxy artifacts | present |
| loss scale-invariance artifacts | present |
| target-scale observability artifacts | present |
| OpenD4RT scale metadata audit artifacts | present |
| self-supervised scale-sensitivity artifacts | present |
| scale-anchor tolerance artifacts | present |
| xyz_local cache spot check | present: scene0050 `(16,1024,3)`, probe5 scene0050 `(16,512,3)` |
| method table | 未生成 |
| Phase F | 未运行 |

## 审计材料

- Phase A/B:
  - `Stream3D/stream4d_native/self_stitch.py`
  - `Stream3D/stream4d_native/sim3.py`
  - `Stream3D/geometry_provider/common.py`
  - `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - `Stream3D/tools/native_geometry_diagnostics.py`
  - `Stream3D/tools/export_v21_3_occupancy_carrier_cache.py`
  - `Stream3D/outputs/audit/v22_phaseB/native_geometry_diagnostics_v22_phaseB.*`
- Phase C:
  - `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r0r4r5_smoke/*`
  - `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r1r2r3/*`
  - `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r6r7r8r9/*`
  - `Stream3D/outputs/audit/v22_direct_reconstruction_probe5_r4_eval_scene/*`
  - `Stream3D/outputs/audit/v22_direct_reconstruction_probe5_r0_r8_raw/*`
- D5 persistent smoke:
  - `Stream3D/outputs/stream4d_debug_v22_occupancy_d5_persistent_scene0050_smoke/scene0050_00/carriers_window*.npz`
  - `Stream3D/outputs/audit/v22_occupancy_d5_persistent_scene0050_smoke/scene0050_00/summary.*`
  - `Stream3D/outputs/audit/v22_d5_persistent_provider_smoke_allwin/*`
  - `Stream3D/outputs/audit/v22_d5_persistent_provider_smoke_bestwin/*`
- 2026-06-11 camera-space continuation:
  - `Stream3D/outputs/audit/v22_1_camera_space_scene0050_r0r4r8/*`
  - `Stream3D/outputs/audit/v22_1_camera_space_scene0050_r1r2r3/*`
  - `Stream3D/outputs/audit/v22_1_camera_space_probe5_r0r4r8/*`
  - `Stream3D/outputs/audit/v22_1_uvz_camera_scene0050_r10r11/*`
  - `Stream3D/outputs/audit/v22_1_uvz_camera_probe5_r10r11/*`
- 2026-06-11 z/uv/filter/depth-calibration continuation:
  - `Stream3D/outputs/audit/v22_2_depth_calibrated_scene0050_r10_r15/*`
  - `Stream3D/outputs/audit/v22_2_depth_calibrated_probe5_r10_r15/*`
  - `Stream3D/outputs/audit/v22_2_uvz_positive_z_scene0050_r10_r11/*`
  - `Stream3D/outputs/audit/v22_2_uvz_positive_z_probe5_r10_r11/*`
  - `Stream3D/outputs/audit/v22_2_reproj_convention_scene0050_r10_r11/*`
  - `Stream3D/outputs/audit/v22_2_reproj_convention_probe5_r10_r11/*`
- 2026-06-11 xyz_local/local-vs-ref continuation:
  - `Stream3D/tools/diagnose_v22_d4rt_local_vs_ref.py`
  - `Stream3D/stream4d/d4rt_adapter.py`
  - `Stream3D/stream4d/carrier_store.py`
  - `Stream3D/stream4d/replay_memory.py`
  - `Stream3D/outputs/audit/v22_3_local_vs_ref_scene0050/local_vs_ref_scene0050.*`
  - `Stream3D/outputs/audit/v22_3_local_vs_ref_probe5/local_vs_ref_probe5.*`
  - `Stream3D/outputs/stream4d_debug_v22_local_xyz_scene0050_r1/scene0050_00/carriers_window000.npz`
  - `Stream3D/outputs/stream4d_debug_v22_local_xyz_probe5_r1/*/carriers_window000.npz`
- 2026-06-11 loss-space xyz transform continuation:
  - `Stream3D/tools/diagnose_v22_d4rt_local_vs_ref.py`
  - `Stream3D/outputs/audit/v22_4_xyz_transform_scene0050/xyz_transform_scene0050.*`
  - `Stream3D/outputs/audit/v22_4_xyz_transform_probe5/xyz_transform_probe5.*`
  - `Stream3D/outputs/audit/v22_4_xyz_transform_probe5_vc07/xyz_transform_probe5_vc07.*`
  - `Stream3D/outputs/audit/v22_4_xyz_transform_probe5_vc09/xyz_transform_probe5_vc09.*`
- 2026-06-11 signed_log1p direct branch:
  - `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_5_direct_xyz_local_transform_probe5/direct_reconstruction_summary.*`
  - `Stream3D/outputs/audit/v22_5_direct_xyz_local_transform_probe5/direct_reconstruction_scene_rows.*`
  - `Stream3D/outputs/audit/v22_5_direct_xyz_local_transform_probe5/direct_reconstruction_f10_by_variant.png`
- 2026-06-11 xyz_local eval-Sim3 upper-bound:
  - `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_6_eval_sim3_xyz_local_transform_probe5/direct_reconstruction_summary.*`
  - `Stream3D/outputs/audit/v22_6_eval_sim3_xyz_local_transform_probe5/direct_reconstruction_scene_rows.*`
  - `Stream3D/outputs/audit/v22_6_eval_sim3_xyz_local_transform_probe5/direct_reconstruction_f10_by_variant.png`
- 2026-06-11 xyz_ref0 ref0-pose/scale upper-bound:
  - `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_7_ref0_pose_scale_probe5/direct_reconstruction_summary.*`
  - `Stream3D/outputs/audit/v22_7_ref0_pose_scale_probe5/direct_reconstruction_scene_rows.*`
  - `Stream3D/outputs/audit/v22_7_ref0_pose_scale_probe5/direct_reconstruction_f10_by_variant.png`
- 2026-06-11 ref0 local/ref scale proxy:
  - `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_8_ref0_local_scale_probe5/direct_reconstruction_summary.*`
  - `Stream3D/outputs/audit/v22_8_ref0_local_scale_probe5/direct_reconstruction_scene_rows.*`
  - `Stream3D/outputs/audit/v22_8_ref0_local_scale_probe5/direct_reconstruction_f10_by_variant.png`
- 2026-06-11 ref0 pose-trajectory scale:
  - `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_9_ref0_pose_trajectory_scale_probe5/direct_reconstruction_summary.*`
  - `Stream3D/outputs/audit/v22_9_ref0_pose_trajectory_scale_probe5/direct_reconstruction_scene_rows.*`
  - `Stream3D/outputs/audit/v22_9_ref0_pose_trajectory_scale_probe5/direct_reconstruction_f10_by_variant.png`
- 2026-06-11 ref0 trajectory consistency:
  - `Stream3D/tools/diagnose_v22_ref0_trajectory_scale.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_frame_rows.*`
  - `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_window_rows.*`
  - `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_scene_summary.*`
  - `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_candidate_scale_errors.*`
  - `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_consistency.md`
- 2026-06-11 ref0 trajectory anchor-policy sweep:
  - `Stream3D/tools/diagnose_v22_ref0_trajectory_policy_sweep.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_frame_rows.*`
  - `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_window_rows.*`
  - `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_scene_summary.*`
  - `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_errors.*`
  - `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_sweep.md`
- 2026-06-11 ref0 scale-convention diagnostic:
  - `Stream3D/tools/diagnose_v22_ref0_scale_convention.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_frame_rows.*`
  - `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_window_rows.*`
  - `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_scene_summary.*`
  - `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_candidate_errors.*`
  - `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention.md`
- 2026-06-11 ref0 intrinsics proxy diagnostic:
  - `Stream3D/tools/diagnose_v22_ref0_intrinsics_proxy.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_frame_rows.*`
  - `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_window_rows.*`
  - `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_scene_summary.*`
  - `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_candidate_errors.*`
  - `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy.md`
- 2026-06-11 LoGeR geometry scale-proxy diagnostic:
  - `Stream3D/tools/diagnose_v22_loger_scale_proxy.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_14_loger_scale_proxy_scene0050_smoke/*`
  - `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy.md`
  - `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_candidate_errors.*`
  - `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_scene_summary.*`
  - `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_frame_rows.*`
  - `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_window_rows.*`
  - `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_metadata.json`
- 2026-06-11 OpenD4RT xyz loss scale-invariance diagnostic:
  - `Stream3D/tools/diagnose_v22_loss_scale_invariance.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_invariance.md`
  - `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_metadata.json`
  - `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_frame_rows.*`
  - `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_sweep_rows.*`
  - `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_scene_summary.*`
  - `Stream3D/outputs/audit/v22_15_loss_scale_invariance_probe5/loss_scale_by_pred_scale.*`
- 2026-06-11 target-scale observability diagnostic:
  - `Stream3D/tools/diagnose_v22_target_scale_observability.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability.md`
  - `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_metadata.json`
  - `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_frame_rows.*`
  - `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_window_rows.*`
  - `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_predictor_summary.*`
  - `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_univariate_summary.*`
  - `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_predictions.*`
  - `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_feature_correlations.*`
  - `Stream3D/outputs/audit/v22_16_target_scale_observability_probe5/target_scale_observability_scene_summary.*`
- 2026-06-11 OpenD4RT scale metadata audit:
  - `Stream3D/tools/diagnose_v22_opend4rt_scale_metadata.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_17_opend4rt_scale_metadata/opend4rt_scale_metadata_audit.md`
  - `Stream3D/outputs/audit/v22_17_opend4rt_scale_metadata/opend4rt_scale_metadata_metadata.json`
  - `Stream3D/outputs/audit/v22_17_opend4rt_scale_metadata/opend4rt_scale_metadata_source_evidence.*`
  - `Stream3D/outputs/audit/v22_17_opend4rt_scale_metadata/opend4rt_scale_metadata_cache_keys.*`
- 2026-06-11 self-supervised scale-sensitivity diagnostic:
  - `Stream3D/tools/diagnose_v22_self_supervised_scale_sensitivity.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5/self_supervised_scale_sensitivity.md`
  - `Stream3D/outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5/self_supervised_scale_sensitivity_metadata.json`
  - `Stream3D/outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5/self_supervised_scale_sensitivity_frame_rows.*`
  - `Stream3D/outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5/self_supervised_scale_sensitivity_sweep_rows.*`
  - `Stream3D/outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5/self_supervised_scale_sensitivity_scene_summary.*`
- 2026-06-11 scale-anchor tolerance diagnostic:
  - `Stream3D/tools/diagnose_v22_scale_anchor_tolerance.py`
  - `Stream3D/tests/test_v22_direct_reconstruction.py`
  - `Stream3D/outputs/audit/v22_19_scale_anchor_tolerance_probe5/scale_anchor_tolerance.md`
  - `Stream3D/outputs/audit/v22_19_scale_anchor_tolerance_probe5/scale_anchor_tolerance_metadata.json`
  - `Stream3D/outputs/audit/v22_19_scale_anchor_tolerance_probe5/scale_anchor_tolerance_scene_rows.*`
  - `Stream3D/outputs/audit/v22_19_scale_anchor_tolerance_probe5/scale_anchor_tolerance_summary.*`

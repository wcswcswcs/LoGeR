# Stream4D v5 执行日志

日期：2026-06-08（Asia/Singapore）  
计划文件：`docs/stream4d_v5_deep_audit_and_parallel_experiment_plan_for_codex.md`  
工作根目录：`/mnt/data/users/chengshun.wang/pjs/LoGeR`  
Stream3D 根目录：`/mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D`  
Python：`/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`

本日志只记录真实执行过的命令、生成的文件、遇到的 blocker 和修复。未运行或未完成的计划项标注为 `not_run` 或 `blocked`。

## 0. 计划理解

v5 计划分为四条 lane：

```text
Phase A: Metric Safety and Code Hygiene
Phase B: D4RT Cache / Multi-window Infrastructure
Phase C: ScanNet Probe5 Algorithm Core
Phase D: Replica-Dynamic / Dynamic Replica
```

关键约束：

```text
1. 禁止把 oracle / GT-read diagnostic 当作 method result。
2. 禁止只报告 own recompute，不报告 fixed/inherit 对照。
3. 禁止把 scene0050 单点 96f/128f 当成多场景趋势。
4. 禁止在 Dynamic Replica 数据不可用时编造 IDF1/MOTA/4D IoU。
5. Probe5 未达到 AP>=0.32/AP50>=0.53/AP25>=0.70/#pred<=300 前，不启动 full ScanNet final。
```

## 1. 代码修改记录

### Phase A: manifest / oracle guard / metric audit

新增：

```text
Stream3D/tools/prediction_manifest.py
Stream3D/tools/scan_reportable_configs.py
```

修改：

```text
Stream3D/tools/oracle_candidate_upper_bound.py
Stream3D/evaluation/evaluate.py
Stream3D/tools/verify_stream4d_metric_integrity.py
Stream3D/stream4d/export_scannet.py
Stream3D/stream4d/reexport_scannet.py
Stream3D/tools/fuse_prediction_configs.py
Stream3D/tools/materialize_scannet_eval_subset.py
Stream3D/tools/evaluate_cross_prepoints.py
Stream3D/tools/export_mask_observation_bank.py
```

目的：

```text
1. 所有新写 prediction/TMP 的工具写 config_manifest.json。
2. oracle 工具写 output-config 时强制名称包含 oracle，并写 uses_gt=true / diagnostic-only manifest。
3. evaluator 默认拒绝 oracle-named prediction/TMP/output，除非显式 --allow-oracle-eval。
4. metric-integrity report 增加 manifest/oracle/suspicious 统计。
```

### Phase A: memory / densifier 修复

修改：

```text
Stream3D/stream4d/appearance_memory.py
Stream3D/stream4d/object_memory_v2.py
Stream3D/stream4d/reliable_densifier.py
Stream3D/tests/test_stream4d_protocol_fixes.py
```

目的：

```text
1. missing appearance 不再贡献正 similarity，避免 memory-v2 出现 missing-feature positive match。
2. reliable_densifier 增加 raw / after_quality_filter / selected / used_for_export 诊断字段。
3. 单元测试从 6 个扩展到 8 个。
```

### Phase B: D4RT cache / preflight

新增：

```text
Stream3D/tools/d4rt_preflight.py
Stream3D/stream4d/d4rt_worker.py
```

修改：

```text
Stream3D/stream4d/d4rt_adapter.py
Stream3D/stream4d/run_scannet.py
```

目的：

```text
1. D4RT preflight 计时 torch.load / build / load_state_dict / fake encode / fake decode。
2. 支持 checkpoint 复制到 local scratch。
3. carrier cache 写 per-window manifest。
4. D4RTAdapter 输出 last_infer_diagnostics。
```

遇到并修复的 blocker：

```text
第一次 2-frame smoke 暴露 NameError: num_queries is not defined。
修复文件：Stream3D/stream4d/d4rt_adapter.py
修复方式：用 target_count * num_carriers 记录 num_queries_per_decode。
验证：py_compile pass；2-frame smoke rerun pass。
```

### Phase C: observation/proposal summary 修复

修改：

```text
Stream3D/tools/export_mask_observation_bank.py
Stream3D/tools/export_local_proposal_bank.py
```

目的：

```text
同一个 output_config 跑多场景时，summary 文件名带 seq_name，避免 scene summary 互相覆盖。
新增保留 latest_summary 便于快速查看。
```

### Phase D: dynamic env checker

修改：

```text
Stream3D/tools/check_dynamic_replica_env.py
```

目的：

```text
增加 --root alias，兼容 v5 计划中的命令形式。
```

## 2. 代码验证

最终语法检查：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/prediction_manifest.py \
  tools/scan_reportable_configs.py \
  tools/d4rt_preflight.py \
  tools/oracle_candidate_upper_bound.py \
  tools/verify_stream4d_metric_integrity.py \
  tools/evaluate_cross_prepoints.py \
  tools/materialize_scannet_eval_subset.py \
  tools/fuse_prediction_configs.py \
  tools/export_mask_observation_bank.py \
  tools/export_local_proposal_bank.py \
  tools/check_dynamic_replica_env.py \
  stream4d/d4rt_adapter.py \
  stream4d/d4rt_worker.py \
  stream4d/run_scannet.py \
  stream4d/reexport_scannet.py \
  stream4d/export_scannet.py \
  stream4d/appearance_memory.py \
  stream4d/object_memory_v2.py \
  stream4d/reliable_densifier.py \
  evaluation/evaluate.py \
  tests/test_stream4d_protocol_fixes.py
```

结果：

```text
pass，无输出。
```

最终单元测试：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 8 tests in 0.002s
OK
```

## 3. Phase A 执行

### A1. 初始 scanner

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -u -m tools.scan_reportable_configs \
  --configs scannet_self_inherit_probe5,scannet_on_stream4d_32f_probe5,stream4d_32f_self_probe5,stream4d_v3_adapt_recompute_on_32f_probe5,stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5,stream4d_v4_1_probe5_oracle_egraph_minobs3 \
  --output outputs/audit/v5_reportable_config_scan_initial.md \
  2>&1 | tee logs/stream4d_v5_reportable_config_scan_initial.log
```

输出：

```text
outputs/audit/v5_reportable_config_scan_initial.md
outputs/audit/v5_reportable_config_scan_initial.json
outputs/audit/v5_reportable_config_scan_initial.csv
```

关键结果：

```text
num_configs=6
num_configs_missing_manifest=6
num_diagnostic_only_configs=1
num_oracle_configs=1
num_reportable_method_configs=0
num_suspicious_configs=6
num_uses_gt_and_method_result=0
```

### A2. 为历史非 oracle configs 写 retroactive manifest

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -u -m tools.scan_reportable_configs \
  --configs scannet_self_inherit_probe5,scannet_on_stream4d_32f_probe5,stream4d_32f_self_probe5,stream4d_v3_adapt_recompute_on_32f_probe5,stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5 \
  --retroactive-method-manifest \
  --require-manifest \
  --output outputs/audit/v5_reportable_config_scan_method.md \
  2>&1 | tee logs/stream4d_v5_reportable_config_scan_method.log
```

输出：

```text
outputs/audit/v5_reportable_config_scan_method.md
outputs/audit/v5_reportable_config_scan_method.json
outputs/audit/v5_reportable_config_scan_method.csv
```

关键结果：

```text
num_configs=5
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=5
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

注意：

```text
这些是 retroactive manifests，只用于给已有历史 artifact 加审计标签，不是重新证明历史 artifact 的生成过程。
```

### A3. oracle guard reject test

第一次手动测试使用了 `v5_bad_method_oracle_guard_test`，名称中包含 `oracle`，因此按新 guard 规则允许写出 diagnostic-only oracle artifact。该产物只作为 GT-read upper-bound diagnostic，不进入主表。

后续执行正确拒绝测试：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -u -m tools.oracle_candidate_upper_bound \
  ... \
  --output-config v5_bad_method_guardtest \
  ... \
  > outputs/audit/v5_oracle_guard_reject_test.log 2>&1
echo ORACLE_GUARD_REJECT_EXIT_STATUS=$? >> outputs/audit/v5_oracle_guard_reject_test.log
```

证据：

```text
outputs/audit/v5_oracle_guard_reject_test.log
```

关键输出：

```text
ValueError: --output-config for oracle_candidate_upper_bound must contain 'oracle'. This tool reads GT and any output prediction is diagnostic-only.
ORACLE_GUARD_REJECT_EXIT_STATUS=1
```

### A4. evaluator oracle reject test

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -m evaluation.evaluate \
  --pred_path data/prediction/v5_bad_method_oracle_guard_test_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --tmp_config v5_bad_method_oracle_guard_test \
  --no_class \
  > outputs/audit/v5_evaluator_oracle_reject_test.log 2>&1
echo EVALUATOR_ORACLE_REJECT_EXIT_STATUS=$? >> outputs/audit/v5_evaluator_oracle_reject_test.log
```

关键输出：

```text
ValueError: Refusing to evaluate oracle-named prediction/TMP/output without --allow-oracle-eval.
EVALUATOR_ORACLE_REJECT_EXIT_STATUS=1
```

### A5. metric integrity, reportable configs

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --configs scannet_self_inherit_probe5,scannet_on_stream4d_32f_probe5,stream4d_32f_self_probe5,stream4d_v3_adapt_recompute_on_32f_probe5,stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5 \
  --output outputs/audit/v5_metric_integrity_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v5_metric_integrity_probe5.log
```

输出：

```text
outputs/audit/v5_metric_integrity_probe5.md
outputs/audit/v5_metric_integrity_probe5.json
```

关键结果：

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

### A6. metric integrity, object_dict alignment sources

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -u -m tools.scan_reportable_configs \
  --configs scannet,stream4d_scannet_32f_ioc075_fixmem \
  --retroactive-method-manifest \
  --require-manifest \
  --output outputs/audit/v5_reportable_config_scan_alignment_sources.md

$PY -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --configs scannet,stream4d_scannet_32f_ioc075_fixmem \
  --output outputs/audit/v5_metric_integrity_probe5_alignment_sources.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v5_metric_integrity_probe5_alignment_sources.log
```

关键结果：

```text
scannet alignment: checked=5, mean_iou=1.0, min_iou=1.0, failed=0
stream4d_scannet_32f_ioc075_fixmem alignment: checked=5, mean_iou=1.0, min_iou=1.0, failed=0
phase0_pass=True
```

## 4. Phase B 执行

### B1. D4RT preflight, no local copy

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -u -m tools.d4rt_preflight \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --device cuda \
  --fake-frames 2 \
  --fake-queries 16 \
  --output outputs/audit/d4rt_preflight_v5_no_copy.md \
  2>&1 | tee logs/stream4d_v5_d4rt_preflight_no_copy.log
```

关键结果：

```text
ok=True
checkpoint_size_bytes=13950006682
seconds_torch_load=40.54081153869629
seconds_build_model=5.7353949546813965
seconds_load_state_dict=0.24279284477233887
seconds_fake_encode=1.075178623199463
seconds_fake_decode=0.22654008865356445
cuda_device_name=NVIDIA RTX A5000
```

### B2. D4RT preflight, local copy

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -u -m tools.d4rt_preflight \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --local-ckpt-copy /tmp/stream4d_v5_d4rt_ckpt/opend4rt.ckpt \
  --device cuda \
  --fake-frames 2 \
  --fake-queries 16 \
  --output outputs/audit/d4rt_preflight_v5_local_copy.md \
  2>&1 | tee logs/stream4d_v5_d4rt_preflight_local_copy.log
```

关键结果：

```text
ok=True
seconds_local_copy=7.6365907192230225
seconds_torch_load=5.030629634857178
seconds_build_model=5.640040397644043
seconds_load_state_dict=0.21878480911254883
seconds_fake_encode=0.211500883102417
seconds_fake_decode=0.07605218887329102
```

### B3. scene0011_00 2-frame smoke

第一次运行遇到 `NameError: num_queries is not defined`。修复 `stream4d/d4rt_adapter.py` 后 rerun：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -u -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt /tmp/stream4d_v5_d4rt_ckpt/opend4rt.ckpt \
  --seq-name scene0011_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 2 \
  --window-size 32 \
  --window-stride 16 \
  --max-points-per-mask 8 \
  --min-points-per-mask 2 \
  --query-chunk-size 512 \
  --rho-min 0.35 \
  --local-ioc-threshold 0.75 \
  --history-match-threshold 0.30 \
  --lost-tolerance-windows 3 \
  --memory-version old \
  --export-mode rgbd_eval \
  --export-nn-radius 0.08 \
  --output-config stream4d_v5_scene0011_2f_smoke_ioc075 \
  --debug-root outputs/stream4d_v5_cache_smoke_2f \
  2>&1 | tee logs/stream4d_v5_scene0011_2f_smoke_rerun.log
```

关键结果：

```text
window=1 carriers=256 props=30 objects=30 sec=0.66
done objects=30 points=127 hit_rate=0.5603 total_sec=1.03
```

证据：

```text
outputs/stream4d_v5_cache_smoke_2f/scene0011_00/carriers_window000_manifest.json
```

### B4. scene0011_00 32-frame smoke

命令同 B3，主要参数改为：

```text
--max-frames 32
--query-chunk-size 1024
--output-config stream4d_v5_scene0011_32f_smoke_ioc075
--debug-root outputs/stream4d_v5_cache_smoke_32f
```

关键结果：

```text
window=1 carriers=3104 props=208 objects=208 sec=18.31
done objects=208 points=5883 hit_rate=0.7143 total_sec=20.79
```

### B5. probe5 96f cache

场景：

```text
scene0050_00
scene0011_00
scene0030_00
scene0081_01
scene0591_00
```

命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CUDA_VISIBLE_DEVICES=<gpu> $PY -u -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt /tmp/stream4d_v5_d4rt_ckpt/opend4rt.ckpt \
  --seq-name <scene> \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 96 \
  --window-size 32 \
  --window-stride 16 \
  --max-points-per-mask 8 \
  --min-points-per-mask 2 \
  --query-chunk-size 1024 \
  --rho-min 0.35 \
  --local-ioc-threshold 0.75 \
  --history-match-threshold 0.30 \
  --lost-tolerance-windows 3 \
  --memory-version old \
  --export-mode rgbd_eval \
  --export-nn-radius 0.08 \
  --output-config stream4d_v5_96f_cache_<scene_short>_ioc075 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  2>&1 | tee logs/stream4d_v5_96f_<scene_short>.log
```

结果汇总：

```text
outputs/cache_audit/v5_cache_success_table.csv
outputs/cache_audit/v5_cache_success_summary.json
```

关键结果：

```text
success_96f=5
gate_96f_5of5=true
```

### B6. probe5 128f cache

场景：

```text
scene0050_00
scene0011_00
scene0030_00
```

命令模板同 B5，主要参数改为：

```text
--max-frames 128
--debug-root outputs/stream4d_v5_cache_128f_probe5
--output-config stream4d_v5_128f_cache_<scene_short>_ioc075
```

关键结果：

```text
success_128f=3
gate_128f_3of5=true
```

## 5. Phase C 执行

### C0. fixed baseline rerun

并行重跑以下 5 个 config：

```text
scannet_self_inherit_probe5
scannet_on_stream4d_32f_probe5
stream4d_32f_self_probe5
stream4d_v3_adapt_recompute_on_32f_probe5
stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
```

命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -u -m evaluation.evaluate \
  --pred_path data/prediction/<config>_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/v5_c0_<config>_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config <config> \
  --no_class \
  2>&1 | tee logs/stream4d_v5_c0_<config>_eval.log
```

实际输出：

```text
data/evaluation/scannet/v5_c0_scannet_self_inherit_probe5_class_agnostic.txt
data/evaluation/scannet/v5_c0_scannet_on_stream4d_32f_probe5_class_agnostic.txt
data/evaluation/scannet/v5_c0_stream4d_32f_self_probe5_class_agnostic.txt
data/evaluation/scannet/v5_c0_stream4d_v3_adapt_recompute_on_32f_probe5_class_agnostic.txt
data/evaluation/scannet/v5_c0_stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5_class_agnostic.txt
```

### C1. Mask Observation Bank, 96f probe5

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CONFIG=stream4d_v5_obs_bank_96f_probe5_ioc075
DEBUG=outputs/stream4d_v5_cache_96f_probe5
SUMMARY=outputs/mask_observation_bank_v5
for SCENE in scene0050_00 scene0011_00 scene0030_00 scene0081_01 scene0591_00; do
  $PY -u -m tools.export_mask_observation_bank \
    --debug-root "$DEBUG" \
    --seq-name "$SCENE" \
    --output-config "$CONFIG" \
    --backbone Cropformer \
    --min-coverage 0.0 \
    --top-k 0 \
    --export-nn-radius 0.05 \
    --min-points-per-mask 100 \
    --summary-root "$SUMMARY"
done 2>&1 | tee logs/stream4d_v5_obs_bank_96f_probe5_export.log
```

输出：

```text
outputs/mask_observation_bank_v5/stream4d_v5_obs_bank_96f_probe5_ioc075_<scene>_summary.json
outputs/mask_observation_bank_v5/stream4d_v5_obs_bank_96f_probe5_ioc075_aggregate.csv
data/prediction/stream4d_v5_obs_bank_96f_probe5_ioc075_class_agnostic/
data/TMP/stream4d_v5_obs_bank_96f_probe5_ioc075/
```

scanner：

```bash
$PY -u -m tools.scan_reportable_configs \
  --configs stream4d_v5_obs_bank_96f_probe5_ioc075 \
  --output outputs/audit/v5_obs_bank_config_scan.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v5_obs_bank_config_scan.log
```

evaluator：

```bash
$PY -u -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v5_obs_bank_96f_probe5_ioc075_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/v5_obs_bank_96f_probe5_ioc075_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v5_obs_bank_96f_probe5_ioc075 \
  --no_class \
  2>&1 | tee logs/stream4d_v5_obs_bank_96f_probe5_eval.log
```

metric integrity：

```bash
$PY -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --configs stream4d_v5_obs_bank_96f_probe5_ioc075 \
  --output outputs/audit/v5_metric_integrity_obs_bank_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v5_metric_integrity_obs_bank_probe5.log
```

### C2. Local Proposal Bank, 96f probe5, minimal object-formation attempt

这不是完整 v5 Boundary-Aware Proposal v2，只是用已有 `local_props` 做一个更接近 object proposal 层的真实尝试。

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CONFIG=stream4d_v5_localprop_96f_probe5_min2_bestframe_ioc075
DEBUG=outputs/stream4d_v5_cache_96f_probe5
SUMMARY=outputs/local_proposal_bank_v5
for SCENE in scene0050_00 scene0011_00 scene0030_00 scene0081_01 scene0591_00; do
  $PY -u -m tools.export_local_proposal_bank \
    --debug-root "$DEBUG" \
    --seq-name "$SCENE" \
    --output-config "$CONFIG" \
    --backbone Cropformer \
    --same-frame-policy best_per_frame \
    --min-observations 2 \
    --min-frames 2 \
    --export-nn-radius 0.05 \
    --export-mask-sample-stride 2 \
    --export-mask-max-pixels 12000 \
    --export-max-masks-per-object 5 \
    --export-mask-min-relative-coverage 0.0 \
    --export-min-points-per-object 100 \
    --export-score-mode observations \
    --summary-root "$SUMMARY"
done 2>&1 | tee logs/stream4d_v5_localprop_96f_probe5_export.log
```

scanner：

```bash
$PY -u -m tools.scan_reportable_configs \
  --configs stream4d_v5_localprop_96f_probe5_min2_bestframe_ioc075 \
  --output outputs/audit/v5_localprop_config_scan.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v5_localprop_config_scan.log
```

evaluator：

```bash
$PY -u -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v5_localprop_96f_probe5_min2_bestframe_ioc075_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/v5_localprop_96f_probe5_min2_bestframe_ioc075_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v5_localprop_96f_probe5_min2_bestframe_ioc075 \
  --no_class \
  2>&1 | tee logs/stream4d_v5_localprop_96f_probe5_eval.log
```

metric integrity：

```bash
$PY -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --configs stream4d_v5_localprop_96f_probe5_min2_bestframe_ioc075 \
  --output outputs/audit/v5_metric_integrity_localprop_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v5_metric_integrity_localprop_probe5.log
```

### C3. not_run / blocked 项

以下计划项本轮没有完整实现，不能编造成已完成：

```text
C-Exp2 Boundary-Aware Proposal v2: blocked，缺少完整 core/fringe/reject proposal 生成实现。
C-Exp3 Typed Evidence Graph v2: blocked，缺少 typed positive/complement/conflict/weak_bridge 边实现和实验矩阵。
C-Exp4 Split/Merge-Capable Object Memory: blocked，缺少 pending/split/quarantine lifecycle 实现和 96f 对照矩阵。
C-Exp5 oracle diagnostic: 只做 guard 测试和一次 diagnostic-only 误名允许测试，不进入主表。
Full ScanNet final: not_run，因为 probe5 gate 未通过。
```

## 6. Phase D 执行

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -u -m tools.check_dynamic_replica_env \
  --root data/dynamic-replica/v2 \
  --split valid \
  --output outputs/audit/dynamic_replica_env_v5.md \
  2>&1 | tee logs/stream4d_v5_dynamic_replica_env.log
```

输出：

```text
outputs/audit/dynamic_replica_env_v5.md
outputs/audit/dynamic_replica_env_v5.json
```

关键结果：

```text
root_exists=False
split_exists=False
usable_scenes=0
can_report_official_instance_tracking=False
can_report_d4rt_trajectory_metrics=False
```

因此 dynamic smoke / official tracking metrics 均未运行，原因是本地数据缺失。

## 7. 主要证据路径

```text
Phase A:
  Stream3D/outputs/audit/v5_reportable_config_scan_method.json
  Stream3D/outputs/audit/v5_metric_integrity_probe5.md
  Stream3D/outputs/audit/v5_metric_integrity_probe5_alignment_sources.md
  Stream3D/outputs/audit/v5_oracle_guard_reject_test.log
  Stream3D/outputs/audit/v5_evaluator_oracle_reject_test.log

Phase B:
  Stream3D/outputs/audit/d4rt_preflight_v5_no_copy.md
  Stream3D/outputs/audit/d4rt_preflight_v5_local_copy.md
  Stream3D/outputs/cache_audit/v5_cache_success_table.csv
  Stream3D/outputs/cache_audit/v5_cache_success_summary.json
  Stream3D/outputs/stream4d_v5_cache_96f_probe5/<scene>/carriers_window*_manifest.json
  Stream3D/outputs/stream4d_v5_cache_128f_probe5/<scene>/carriers_window*_manifest.json

Phase C:
  Stream3D/data/evaluation/scannet/v5_c0_*_class_agnostic.txt
  Stream3D/outputs/mask_observation_bank_v5/stream4d_v5_obs_bank_96f_probe5_ioc075_aggregate.csv
  Stream3D/data/evaluation/scannet/v5_obs_bank_96f_probe5_ioc075_class_agnostic.txt
  Stream3D/outputs/local_proposal_bank_v5/stream4d_v5_localprop_96f_probe5_min2_bestframe_ioc075_aggregate.csv
  Stream3D/data/evaluation/scannet/v5_localprop_96f_probe5_min2_bestframe_ioc075_class_agnostic.txt
  Stream3D/outputs/audit/v5_metric_integrity_obs_bank_probe5.md
  Stream3D/outputs/audit/v5_metric_integrity_localprop_probe5.md

Phase D:
  Stream3D/outputs/audit/dynamic_replica_env_v5.md
  Stream3D/outputs/audit/dynamic_replica_env_v5.json
```

## 8. 代码审计包

用户要求补充代码文件包用于审计。已生成：

```text
stream4d_v5_code_audit_packet_20260608.zip
stream4d_v5_code_audit_packet_20260608.sha256
stream4d_v5_code_audit_packet_20260608_zip_filelist.txt
code_audit_pack/stream4d_v5_code_audit_packet_20260608/
```

校验：

```text
sha256 file = stream4d_v5_code_audit_packet_20260608.sha256
zip filelist = stream4d_v5_code_audit_packet_20260608_zip_filelist.txt
zip entries = 34
zip size = 112K
```

说明：不把 zip 自身 sha256 固定写入包内日志，避免压缩包内容自引用导致 hash 随日志变化；最终校验值以同名 `.sha256` 文件为准。

包内包含：

```text
1. v5 计划文件、执行日志、实验结果复盘。
2. 本轮新增/修改的 Stream3D tools / stream4d / evaluation / tests 代码文件。
3. stream4d_v5_code_audit_packet_20260608_git_status.txt
4. stream4d_v5_code_audit_packet_20260608_git_diff.patch
5. stream4d_v5_code_audit_packet_20260608_filelist.txt
```

注意：

```text
该包用于代码审计，不包含大体积 data/cache/results。
实验证据路径仍以本执行日志第 7 节记录的本地 outputs/data/evaluation 文件为准。
```

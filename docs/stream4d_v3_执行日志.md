# Stream4D v3 执行日志

开始时间：2026-06-07 21:44:25 +08  
工作目录：`/mnt/data/users/chengshun.wang/pjs/LoGeR`  
计划文件：`docs/stream4d_v3_protocol_corrected_plan_for_codex.md`  
代码目录：`Stream3D`  
执行环境：`conda env: loger`

## 执行原则

- 不编造数据；只记录实际命令、实际落盘文件和实际输出。
- 主协议使用计划文件确认的 Stream3D-style cropped-TMP evaluator。
- fullmesh evaluator 仅作为额外诊断，不替换主协议。
- 重点执行 v3 P0/S0/S1：审计 evaluator 协议、实现 `pre_points` policy、比较 recompute 与 inherit。
- 若遇到 blocker，先按计划里的修复方向尝试；修复动作和结果同步写入复盘。

## 初始已知结果

原 Stream3D-Cropformer baseline：

```text
20.11 / 34.47 / 50.23
data/evaluation/scannet/scannet_class_agnostic.txt
```

Stream4D MVP：

```text
12.76 / 23.68 / 42.21
stream4d_scannet_32f_ioc075_fixmem
```

当前 best，recompute pre_points：

```text
20.37 / 35.52 / 55.06
stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one
```

## Phase 0：读取计划

读取命令：

```bash
sed -n '1,260p' docs/stream4d_v3_protocol_corrected_plan_for_codex.md
sed -n '260,520p' docs/stream4d_v3_protocol_corrected_plan_for_codex.md
sed -n '520,860p' docs/stream4d_v3_protocol_corrected_plan_for_codex.md
```

理解到的 P0 必做事项：

1. `rescore_scannet.py` 新增 `--pre-points-policy recompute|inherit|fixed_path`。
2. summary 必须记录 `pre_points_policy`、输入 pre_points 数、输出 pre_points 数、prediction union 数。
3. 新增 `tools/audit_stream3d_eval_protocol.py`。
4. 修复 `local_4d_filter.py` 中 set 与 weights zip 导致权重错配的问题，并增加测试。
5. 运行 S0 evaluator protocol audit。
6. 运行 S1 pre_points policy 实验。

继续读取计划后，补充理解到的后续要求：

```bash
sed -n '860,1220p' docs/stream4d_v3_protocol_corrected_plan_for_codex.md
sed -n '1220,1700p' docs/stream4d_v3_protocol_corrected_plan_for_codex.md
sed -n '1700,2100p' docs/stream4d_v3_protocol_corrected_plan_for_codex.md
```

后续阶段包括 tune/final split、quality score、reliable densification、memory-v2、D4RT Sim3 export、Replica-Dynamic。由于本轮 S1 证明 `inherit` 明显失败，按计划 9.1 的决策树，本轮没有继续做大规模 ratio grid 盲调，而是锁定协议审计、`pre_points` 对照、split 诊断，并把后续方向写入复盘。

## 代码修改记录

修改文件：

```text
Stream3D/stream4d/rescore_scannet.py
Stream3D/stream4d/local_4d_filter.py
Stream3D/stream4d/carrier_sampler.py
Stream3D/tools/audit_stream3d_eval_protocol.py
Stream3D/tools/make_scannet_stream4d_splits.py
Stream3D/tools/materialize_scannet_eval_subset.py
Stream3D/tests/__init__.py
Stream3D/tests/test_stream4d_protocol_fixes.py
```

具体修改：

- `rescore_scannet.py` 增加 `--pre-points-policy recompute|inherit|fixed_path`、`--fixed-pre-points-root`、`--fixed-pre-points-config`；summary 记录输入 pre_points 数、输出 pre_points 数、prediction union 数、集合是否相等、union 是否为 pre_points 子集。
- `local_4d_filter.py` 修复 carrier 权重错配：不再把无序 `support set` 与 `obs.weights` zip；改为直接按 `obs.carrier_ids` 与 `obs.weights` 的同序数组构造权重。
- `carrier_sampler.py` 修复潜在长度错配：当实际 mask 像素数少于请求采样数时，`src_frame/src_global/src_mask_id` 使用实际采样数量。
- `audit_stream3d_eval_protocol.py` 新增 evaluator 协议审计，输出 markdown/json/比例图。
- `make_scannet_stream4d_splits.py` 新增固定 seed tune/final split 生成工具。
- `materialize_scannet_eval_subset.py` 新增 split 子配置 materialize 工具，用软链接创建只包含 split 场景的 prediction/TMP 目录。
- `test_stream4d_protocol_fixes.py` 增加两个单元测试：carrier 权重顺序对齐、carrier sampler 实际采样数量一致。

验证命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n loger python -m py_compile stream4d/local_4d_filter.py stream4d/rescore_scannet.py stream4d/carrier_sampler.py tools/audit_stream3d_eval_protocol.py tests/test_stream4d_protocol_fixes.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
py_compile：通过
unittest：Ran 2 tests ... OK
```

环境修复记录：

- `conda` 不在默认 PATH，实际路径为 `/mnt/data/users/chengshun.wang/miniconda3/bin/conda`；后续命令使用绝对路径或 env 内 Python `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`。
- 计划默认原始 Stream3D 路径 `/mnt/data/orig_stream3d/Code_Stream3D` 无写权限，无法在那里解压；改为从 `/mnt/data/users/chengshun.wang/pjs/Code_Stream3D.zip` 解压到 `/mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D`。
- baseline `scannet` 的 TMP 在 `Stream3D/TMP/scannet`，Stream4D 后处理 TMP 在 `Stream3D/data/TMP/...`；audit 工具增加 fallback，同时检查两个位置。

## S0：Evaluator Protocol Audit

准备原始代码：

```bash
mkdir -p /mnt/data/users/chengshun.wang/pjs/orig_stream3d
unzip -q -o /mnt/data/users/chengshun.wang/pjs/Code_Stream3D.zip -d /mnt/data/users/chengshun.wang/pjs/orig_stream3d
```

执行命令：

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n loger python -m tools.audit_stream3d_eval_protocol \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --configs scannet,stream4d_scannet_32f_ioc075_fixmem,stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one \
  --seq-list splits/scannet.txt \
  --output outputs/audit/eval_protocol_audit.md \
  2>&1 | tee logs/stream4d_v3_s0_eval_protocol_audit.log
```

输出文件：

```text
Stream3D/outputs/audit/eval_protocol_audit.md
Stream3D/outputs/audit/eval_protocol_audit.json
Stream3D/outputs/audit/eval_protocol_audit_pre_points_ratio.png
Stream3D/outputs/audit/eval_protocol_audit_prediction_union_ratio.png
Stream3D/logs/stream4d_v3_s0_eval_protocol_audit.log
```

关键结果：

```text
original/current evaluate.py 均存在
original/current 均读取 *_pre_points.npy
AP core functions evaluate_matches / compute_averages 哈希一致：True
```

S0 汇总：

```text
scannet baseline: AP/AP50/AP25 = 20.11 / 34.47 / 50.23, mean pre_points ratio = 87.02%, mean #pred = 101.14
Stream4D MVP:     AP/AP50/AP25 = 12.76 / 23.68 / 42.21, mean pre_points ratio = 7.38%,  mean #pred = 185.02
Current best:     AP/AP50/AP25 = 20.37 / 35.52 / 55.06, mean pre_points ratio = 5.75%,  mean #pred = 15.20
```

## S1：pre_points policy 实验

基础输入配置：

```text
input_config = stream4d_scannet_32f_ioc075_fixmem
seq_list = splits/scannet.txt
backbone = Cropformer
```

运行的 4 个输出配置：

```text
stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute
stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit
stream4d_scannet_32f_ioc075_fixmem_v3_min250_area_one_recompute
stream4d_scannet_32f_ioc075_fixmem_v3_min250_area_one_inherit
```

adaptive top-k recompute：

```bash
CFG=stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n loger python -m stream4d.rescore_scannet \
  --seq-list splits/scannet.txt --backbone Cropformer \
  --input-config stream4d_scannet_32f_ioc075_fixmem --output-config "$CFG" \
  --score-mode one --select-mode mask_count \
  --filter-max-instances-ratio 0.14 --filter-min-instances 8 --filter-max-instances 18 \
  --pre-points-policy recompute --debug-root outputs/stream4d_rescore_v3 \
  2>&1 | tee logs/${CFG}_rescore.log
```

adaptive top-k inherit：同上，仅 `CFG` 改为 `..._inherit`，`--pre-points-policy inherit`。

min250 recompute：

```bash
CFG=stream4d_scannet_32f_ioc075_fixmem_v3_min250_area_one_recompute
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n loger python -m stream4d.rescore_scannet \
  --seq-list splits/scannet.txt --backbone Cropformer \
  --input-config stream4d_scannet_32f_ioc075_fixmem --output-config "$CFG" \
  --score-mode one --select-mode area \
  --filter-min-points-per-object 250 \
  --pre-points-policy recompute --debug-root outputs/stream4d_rescore_v3 \
  2>&1 | tee logs/${CFG}_rescore.log
```

min250 inherit：同上，仅 `CFG` 改为 `..._inherit`，`--pre-points-policy inherit`。

rescore summary 文件：

```text
Stream3D/outputs/stream4d_rescore_v3/*_summary.json
```

rescore 检查：

```text
4 个配置均为 312 scenes，errors = 0
adaptive recompute/inherit 平均保留实例数 = 15.1955
min250 recompute/inherit 平均保留实例数 = 10.4263
inherit 配置中 prediction union 均为 output pre_points 子集，且与 output pre_points 不相等
```

评价命令模板：

```bash
CFG=<output_config>
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/${CFG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet --no_class --tmp_config ${CFG} \
  --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
  2>&1 | tee logs/${CFG}_evaluate.log
```

S1 结果：

```text
adaptive recompute: 20.3718 / 35.5222 / 55.0649
adaptive inherit:   12.2851 / 23.3147 / 41.6773
min250 recompute:   18.9498 / 32.5721 / 53.0585
min250 inherit:      9.9167 / 18.9180 / 36.6473
```

S1 audit 命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.audit_stream3d_eval_protocol \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --configs scannet,stream4d_scannet_32f_ioc075_fixmem,stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one,stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute,stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit,stream4d_scannet_32f_ioc075_fixmem_v3_min250_area_one_recompute,stream4d_scannet_32f_ioc075_fixmem_v3_min250_area_one_inherit \
  --seq-list splits/scannet.txt \
  --output outputs/audit/eval_protocol_audit_s1.md \
  2>&1 | tee logs/stream4d_v3_s1_eval_protocol_audit.log
```

输出：

```text
Stream3D/outputs/audit/eval_protocol_audit_s1.md
Stream3D/outputs/audit/eval_protocol_audit_s1.json
Stream3D/outputs/audit/eval_protocol_audit_s1_pre_points_ratio.png
Stream3D/outputs/audit/eval_protocol_audit_s1_prediction_union_ratio.png
```

## Phase 1：tune/final split 诊断

生成 split：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.make_scannet_stream4d_splits \
  --input splits/scannet.txt \
  --tune-output splits/scannet_tune.txt \
  --final-output splits/scannet_final.txt \
  --seed 20260607 \
  2>&1 | tee logs/stream4d_v3_make_scannet_splits.log
```

结果：

```text
splits/scannet_tune.txt: 156 scenes
splits/scannet_final.txt: 156 scenes
```

materialize 命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
for SPLIT in tune final; do
  for CFG in scannet stream4d_scannet_32f_ioc075_fixmem stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit; do
    OUT=${CFG}_v3_${SPLIT}
    $PY -m tools.materialize_scannet_eval_subset --root . --config "$CFG" --seq-list splits/scannet_${SPLIT}.txt --output-config "$OUT"
  done
done 2>&1 | tee logs/stream4d_v3_materialize_tune_final.log
```

split evaluator 命令：对 materialize 得到的 8 个 `*_v3_tune/final` 配置，使用与 S1 相同的 `evaluation.evaluate` 模板执行。

split 诊断结果：

```text
scannet_v3_tune:                                                     20.8394 / 35.5954 / 50.8577
stream4d_scannet_32f_ioc075_fixmem_v3_tune:                          13.1056 / 24.2241 / 42.6161
adaptive recompute v3_tune:                                          20.5105 / 35.3847 / 55.1521
adaptive inherit v3_tune:                                            12.6566 / 23.6953 / 42.1901

scannet_v3_final:                                                    19.4294 / 33.3989 / 49.6361
stream4d_scannet_32f_ioc075_fixmem_v3_final:                         12.4298 / 23.1572 / 41.8428
adaptive recompute v3_final:                                         20.2401 / 35.6642 / 54.9907
adaptive inherit v3_final:                                           11.9313 / 22.9523 / 41.1922
```

注意：本轮没有完成计划中的完整 tune grid search。原因是 S1 已经触发计划 9.1 的分支：`recompute` 超 baseline 但 `inherit` 不超，继续大规模 ratio/grid 调参不能解决核心失败，后续应优先做 reliable densification 和 memory-v2。

结束时间：2026-06-07 22:10:39 +08

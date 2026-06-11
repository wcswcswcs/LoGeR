# ACL2 v47 Adaptive TTT Writing No-Chunk 执行日志

日期：2026-06-08（Asia/Singapore）  
目标：把 TTT writing 从“固定比例 tri replay / 手工 gamma”推进到真正 adaptive writing；禁止 chunk-id 参数和手工 tri replay 百分比；同时记录速度瓶颈。

结果根目录：

```text
results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk/
```

## 0. 背景定义

用户指出：

```text
adaptive TTT writing 不是 adaptive_quantile role assignment。
真正需要的是：TTT 写入策略本身自适应，不依赖手工置顶百分比进行 triple replay。
```

因此本轮定义：

```text
1. 不允许使用 C9/P0 的 absolute chunk-id maps。
2. 不允许用手工 positive/negative/neutral tri replay 百分比作为 role 选择。
3. 负写入强度不能由人工 gamma 表指定，必须由当前 chunk/layer 的风险分布自适应得到。
4. 需要同时关注速度；v46B 已定位 TTT rows 慢在 probe_ttt_write commit / hmc.build_probe_ttt_write_state(...)。
```

## 1. 代码修改

### 1.1 adaptive writer role mode

修改文件：

```text
loger/pipeline/ttt_write_controller.py
```

新增 role modes：

```text
adaptive_writer
adaptive_writer_fused
adaptive_otsu_writer
adaptive_otsu_fused
no_percentage
no_percentage_fused
```

核心行为：

```text
1. 使用当前 risk 分布做 Otsu-style 三段划分，生成 positive / neutral / negative masks。
2. 不调用 fixed top positive_frac / negative_frac cap。
3. external gamma 可以为 0；adaptive writer 自己根据 risk_gap、negative mass、token count 计算 gamma_eff。
4. fused path 将 positive / neutral / negative 写入合并为一次 replay，减少旧 tri replay 的重复计算。
```

审计字段：

```text
ttt_tri_replay_role_mode
ttt_tri_replay_role_source
ttt_tri_replay_role_uncapped_pos_mass
ttt_tri_replay_role_uncapped_neg_mass
ttt_tri_replay_adaptive_writer_thresholds
ttt_tri_replay_adaptive_writer_otsu_score
ttt_tri_replay_adaptive_writer_fused
ttt_tri_replay_adaptive_gamma
ttt_tri_replay_adaptive_neutral_lambda
ttt_tri_replay_adaptive_risk_gap
ttt_tri_replay_adaptive_token_count
```

### 1.2 CLI 接入

修改文件：

```text
run_pipeline_abc_v2.py
```

修改：

```text
--ttt_write_tri_replay_role_mode choices 增加 adaptive_writer* / no_percentage*。
```

### 1.3 新增 no-chunk launcher

新增文件：

```text
tools/run_v47_adaptive_ttt_writer_candidate.sh
```

rows：

```text
AW010_ADAPTIVE_TTT_ONLY
AW110_FRAME_ADAPTIVE_TTT
AW111_FRAME_ADAPTIVE_TTT_SWA
```

launcher 强制 no-chunk：

```text
READ_BETA_FRAME_CHUNKS=""
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS=""
TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
TTT_WRITE_COMMIT_EMA_CHUNKS=""
TTT_WRITE_NATIVE_MIX_CHUNKS=""
```

launcher 强制 no manual tri replay percentages：

```text
TTT_WRITE_GRADIENT_REVERSAL_GAMMA="0.0"
TTT_WRITE_GRADIENT_REVERSAL_NEGATIVE_FRAC="0.0"
TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="0.0"
TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="0.0"
TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="0.0"
TTT_WRITE_TRI_REPLAY_ROLE_MODE="adaptive_writer_fused"
```

每个 run 写出：

```text
v47_effective_config.yaml
adaptive_writer_audit.json
chunk_id_policy_audit.json
v47_reproduce_command.sh
```

### 1.4 新增 report

新增文件：

```text
tools/v47_adaptive_ttt_writer_report.py
```

输出：

```text
phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_registry.csv
phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_registry.json
phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_report.md
```

比较对象：

```text
C9/P0 ATE = 33.76294210291885
v46B fixed-percentage TTT 对应行：
  AW010 -> F010_ONLY_TTT
  AW110 -> F110_FRAME_ATTN_TTT
  AW111 -> F111_ALL_THREE
```

## 2. 语法验证

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile loger/pipeline/ttt_write_controller.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile run_pipeline_abc_v2.py loger/pipeline/ttt_write_controller.py
chmod +x tools/run_v47_adaptive_ttt_writer_candidate.sh
bash -n tools/run_v47_adaptive_ttt_writer_candidate.sh
chmod +x tools/v47_adaptive_ttt_writer_report.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/v47_adaptive_ttt_writer_report.py
```

结果：

```text
pass
```

## 3. 待执行 smoke

执行命令：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk \
V47_END_FRAME=96 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW010_ADAPTIVE_TTT_ONLY V47_SMOKE_AW010_96F
```

smoke 目标：

```text
1. 确认 role_mode=adaptive_writer_fused 被 run_pipeline 接受。
2. 确认 ttt_tri_replay_adaptive_writer_fused=True 落入 debug。
3. 确认 adaptive_gamma 非空。
4. 确认 timing_summary.json 中 probe_ttt_write_seconds 有值。
5. 确认没有 chunk-id maps 和手工 tri replay 百分比。
```

结果：

```text
DONE
frames = 96
adaptive_writer_fused_debug_count = 72
role_modes_seen = adaptive_writer_fused
role_sources_seen = adaptive_writer_otsu3
adaptive_gamma_mean = 0.001319117669481784
probe_ttt_write_seconds_mean = 3.450065791606903
chunk_total_seconds_mean = 32.96995633840561
```

注意：

```text
96F ATE 不是 full KITTI01 指标，不能和 C9/P0 full ATE 比较。
该 smoke 只证明 adaptive writer 路径、debug 和 timing 字段可用。
```

## 4. report 修复记录

第一次运行：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk
```

报错：

```text
ValueError: too many values to unpack (expected 2)
```

原因：

```text
tools.v18_true_action_report._load_kitti_gt 返回 gt_frames, gt_poses, gt_pos 三个值。
```

修复：

```text
tools/v47_adaptive_ttt_writer_report.py:
  _load_kitti_gt(...) 接收三返回值。
  只在 frames>=1000 时计算 delta_vs_C9_P0 / delta_vs_fixed_TTT，避免 96F smoke 被误写成 full ATE 结论。
```

后续又发现 report regex 对 01.log 的 raw string 写成了 `\\s`，导致 adaptive debug 抽取为空。

修复：

```text
tools/v47_adaptive_ttt_writer_report.py:
  将 debug regex 修正为 `\s`。
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/v47_adaptive_ttt_writer_report.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk
```

结果：

```text
Wrote v47 adaptive TTT writer report to
results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk/phase1_adaptive_writer/report_R1
```

## 5. risk source bug 与速度 blocker 诊断

### 5.1 初始 smoke 的解释边界

`V47_SMOKE_AW010_96F` 最初显示：

```text
probe_ttt_write_seconds_mean = 3.450065791606903
chunk_total_seconds_mean = 32.96995633840561
adaptive_gamma_mean = 0.001319117669481784
```

但后续审计发现：

```text
当 external gamma=0 且 risk source 请求 update_conflict_energy 时，
_build_gradient_reversal_risk_flat(...) 没有真正构造 update_conflict risk，
而是退回了 prior-like cheap risk。
```

因此该 smoke 只能说明 adaptive fused writer 路径可运行，不能说明 update-conflict risk 快。

### 5.2 修复 risk source gating

修改文件：

```text
loger/pipeline/ttt_write_controller.py
```

修改：

```text
当 tri_replay_role_mode 属于 adaptive_writer/no_percentage 系列时，
即使外部 gamma=0，也允许 requested risk_source 生效。
```

目的：

```text
让 adaptive writer 自己计算 gamma，而不是因为 external gamma=0 跳过风险构造。
```

验证命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile loger/pipeline/ttt_write_controller.py
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk \
V47_END_FRAME=96 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW010_ADAPTIVE_TTT_ONLY V47_RISKFIX_SMOKE_AW010_96F
```

结果：

```text
ttt_gradient_reversal_risk_source = update_conflict_energy
ttt_gradient_reversal_risk_source_applied = true
probe_ttt_write_seconds_mean = 69.28046560287476
chunk_total_seconds_mean = 105.64987802505493
```

结论：

```text
update_conflict_energy 的风险构造非常慢，单 chunk probe TTT write 接近 69s。
```

### 5.3 GPU/日志优化尝试

修改文件：

```text
loger/pipeline/ttt_write_controller.py
```

修改：

```text
1. _ttt_layer_w0_update_risk(...) 尽量保留 tensor 在 k.device 上，减少 CPU round-trip。
2. adaptive writer role 下不再把完整 risk vector clone 到 debug，只记录 count/statistics。
```

验证命令：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk \
V47_END_FRAME=96 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW010_ADAPTIVE_TTT_ONLY V47_GPUFIX_SMOKE_AW010_96F
```

结果：

```text
probe_ttt_write_seconds_mean = 75.27440929412842
chunk_total_seconds_mean = 113.34883093833923
```

结论：

```text
CPU copy/debug 不是主瓶颈。
真正拖慢的是 update_conflict_energy 本身的逐层/逐 token 风险计算。
```

### 5.4 d_tok cheap risk smoke

为确认“慢的是 update-conflict risk，不是 adaptive writer 本身”，切换为当前 chunk 已有的 D_tok 风险：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk \
V47_END_FRAME=96 \
V47_TTT_RISK_SOURCE=d_tok \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW010_ADAPTIVE_TTT_ONLY V47_DTOK_SMOKE_AW010_96F
```

结果：

```text
DONE
frames = 96
hmc_rows = 4
ttt_gradient_reversal_risk_source = d_tok
ttt_gradient_reversal_risk_source_applied = true
ttt_tri_replay_role_mode = adaptive_writer_fused
ttt_tri_replay_role_source = adaptive_writer_otsu3
probe_ttt_write_seconds_mean = 3.504811644554138
chunk_total_seconds_mean = 33.28179794549942
adaptive_gamma_mean = 0.0006464838952524588
```

速度对比：

| Run | risk source | chunks | probe_ttt_write mean | chunk total mean | 结论 |
|---|---|---:|---:|---:|---|
| `V47_SMOKE_AW010_96F` | fallback/prior-like | 4 | `3.450065791606903` | `32.96995633840561` | 快，但不是 update-conflict |
| `V47_RISKFIX_SMOKE_AW010_96F` | update_conflict_energy | 1 | `69.28046560287476` | `105.64987802505493` | 慢 |
| `V47_GPUFIX_SMOKE_AW010_96F` | update_conflict_energy optimized attempt | 1 | `75.27440929412842` | `113.34883093833923` | 仍慢 |
| `V47_DTOK_SMOKE_AW010_96F` | d_tok | 4 | `3.504811644554138` | `33.28179794549942` | 快 |

当前速度结论：

```text
adaptive writer fused 本身不是主要速度 blocker。
update_conflict_energy risk source 是主要拖累。
d_tok risk source 保留自适应写入定义，同时恢复到约 3.5s/chunk 的 probe TTT write 速度。
```

## 6. full KITTI01 d_tok adaptive writer 启动

因为 `d_tok` smoke 同时满足：

```text
1. no chunk-id policy
2. no manual tri replay percentage
3. adaptive_writer_fused 实际生效
4. probe_ttt_write_seconds 回到可接受量级
```

启动三条 full KITTI01：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk \
V47_TTT_RISK_SOURCE=d_tok \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW010_ADAPTIVE_TTT_ONLY V47_DTOK_AW010_ADAPTIVE_TTT_ONLY

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk \
V47_TTT_RISK_SOURCE=d_tok \
tools/run_v47_adaptive_ttt_writer_candidate.sh 1 AW110_FRAME_ADAPTIVE_TTT V47_DTOK_AW110_FRAME_ADAPTIVE_TTT

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk \
V47_TTT_RISK_SOURCE=d_tok \
tools/run_v47_adaptive_ttt_writer_candidate.sh 2 AW111_FRAME_ADAPTIVE_TTT_SWA V47_DTOK_AW111_FRAME_ADAPTIVE_TTT_SWA
```

当前状态：

```text
started at 2026-06-08 20:09:59 Asia/Singapore
done
```

完成状态：

| Run | Row | started | finished | hmc rows | frames | 状态 |
|---|---|---|---|---:|---:|---|
| `V47_DTOK_AW010_ADAPTIVE_TTT_ONLY` | `AW010_ADAPTIVE_TTT_ONLY` | `20:09:59` | `20:35:14` | `38` | `1101` | DONE |
| `V47_DTOK_AW110_FRAME_ADAPTIVE_TTT` | `AW110_FRAME_ADAPTIVE_TTT` | `20:09:59` | `20:35:04` | `38` | `1101` | DONE |
| `V47_DTOK_AW111_FRAME_ADAPTIVE_TTT_SWA` | `AW111_FRAME_ADAPTIVE_TTT_SWA` | `20:09:59` | `20:36:18` | `38` | `1101` | DONE |

## 7. final report

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/v47_adaptive_ttt_writer_report.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk
```

输出：

```text
results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk/phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_registry.csv
results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk/phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_registry.json
results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk/phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_report.md
```

为了避免同一 Row 下 smoke/aborted/full 混淆，report Markdown 增加了：

```text
Run
status
```

列。该修改只影响报告可读性，不改指标计算。

最终 full KITTI01 结果：

| Run | ATE_full | delta vs C9/P0 | fixed reference | delta vs fixed TTT | probe_ttt mean | chunk mean |
|---|---:|---:|---|---:|---:|---:|
| `V47_DTOK_AW010_ADAPTIVE_TTT_ONLY` | `41.38490315066118` | `+7.621961047742332` | `F010_ONLY_TTT=39.51910428415088` | `+1.8657988665103034` | `3.489066061220671` | `38.025220808229946` |
| `V47_DTOK_AW110_FRAME_ADAPTIVE_TTT` | `38.326470740397355` | `+4.563528637478505` | `F110_FRAME_ATTN_TTT=36.668910841145234` | `+1.6575598992521208` | `3.464312277342144` | `37.73894845184527` |
| `V47_DTOK_AW111_FRAME_ADAPTIVE_TTT_SWA` | `38.322144227956436` | `+4.5592021250375865` | `F111_ALL_THREE=36.65074045045878` | `+1.671403777497659` | `3.633767454247726` | `39.45773262099216` |

共同审计字段：

```text
frames = 1101
hmc_rows = 38
no_chunk_policy_pass = True
adaptive_writer_audit_pass = True
role_modes_seen = adaptive_writer_fused
role_sources_seen = adaptive_writer_otsu3
ttt_positive_mass_mean = 0.665234426134511
ttt_neutral_mass_mean = 0.20001342618151716
ttt_negative_mass_mean = 0.13475215238960167
adaptive_gamma_mean = 0.0005682417494857585
```

核心结论：

```text
1. v47 d_tok adaptive writer 是真正 no-chunk/no-manual-percentage adaptive writing。
2. 它解决了 update_conflict risk source 导致的速度问题。
3. 但 full KITTI01 性能显著低于 C9/P0，也低于 v46B fixed-percentage TTT 对应行。
4. 因此当前 adaptive TTT writing 不是可接受替代方案。
```

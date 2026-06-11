# ACL2 v46B 执行日志

日期：2026-06-08（Asia/Singapore）

计划文件：

```text
docs/ACL2_v46B_ComponentAttribution_FrameTTT_FrameSWA_Addendum.md
```

结果根目录：

```text
results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/
```

## 执行原则

```text
1. 本日志只记录真实执行过的命令、文件路径和状态。
2. 任何未运行或未落盘的结果都标记为 not_run / missing，不补写。
3. 所有 factorial rows 使用 clean no-chunk policy：
   READ_BETA_FRAME_CHUNKS=""
   TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
   TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
   TTT_WRITE_COMMIT_EMA_CHUNKS=""
4. 若 action debug 显示组件没有真的启用，row invalid，修 launcher/CLI/HMC 后重跑。
```

## 0. 计划与当前状态复核

读取计划：

```bash
sed -n '1,220p' docs/ACL2_v46B_ComponentAttribution_FrameTTT_FrameSWA_Addendum.md
sed -n '220,520p' docs/ACL2_v46B_ComponentAttribution_FrameTTT_FrameSWA_Addendum.md
```

确认：

```text
mandatory rows:
  F110_FRAME_ATTN_TTT
  F101_FRAME_ATTN_SWA

为完整 interaction 需要运行 8-row factorial:
  F000_NONE
  F100_ONLY_FRAME_ATTN
  F010_ONLY_TTT
  F001_ONLY_SWA
  F110_FRAME_ATTN_TTT
  F101_FRAME_ATTN_SWA
  F011_TTT_SWA
  F111_ALL_THREE
```

## 1. 新增代码

新增逐 row launcher：

```text
tools/run_v46b_factorial_candidate.sh
```

新增 report：

```text
tools/v46b_factorial_report.py
```

后续所有具体运行命令、stdout/stderr 状态、blocker 与修复会追加到本文件。

## 2. 语法与 dry-run 检查

语法检查：

```bash
chmod +x tools/run_v46b_factorial_candidate.sh
bash -n tools/run_v46b_factorial_candidate.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/v46b_factorial_report.py
```

结果：

```text
pass
```

第一次 report dry run：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v46b_factorial_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa
```

触发 blocker：

```text
TypeError: tuple indices must be integers or slices, not tuple
```

原因：

```text
tools.kitti_trajectory_diagnostics._load_kitti_gt 返回三元组：
(_gt_frames, gt_poses, gt_pos)
新 report 第一版错误地把它当成单个 gt_poses ndarray 使用。
```

修复：

```text
tools/v46b_factorial_report.py:
  _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
  _load_tum_prediction(path, gt_pos.shape[0])
```

修复后重新执行：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/v46b_factorial_report.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v46b_factorial_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa
```

结果：

```text
Wrote v46B report to results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1
```

dry-run 生成文件：

```text
phase2_component_interactions.csv
phase2_component_main_effects.csv
phase2_factorial_registry.csv
phase2_factorial_registry.json
phase2_frame_ttt_swa_summary.md
component_interaction_heatmap.png
segment_delta_stacked_bar.png
```

说明：

```text
这是 missing-state dry run；所有 rows 均为 missing_prediction/invalid。
该步骤只验证 report 不会崩、不编造 ATE。
```


## 3. Phase2 8-row factorial 启动

启动命令模板：

```bash
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa \
  tools/run_v46b_factorial_candidate.sh <GPU> <ROW> V46B_<ROW>
```

实际启动：

```bash
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa tools/run_v46b_factorial_candidate.sh 0 F000_NONE V46B_F000_NONE
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa tools/run_v46b_factorial_candidate.sh 1 F100_ONLY_FRAME_ATTN V46B_F100_ONLY_FRAME_ATTN
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa tools/run_v46b_factorial_candidate.sh 2 F010_ONLY_TTT V46B_F010_ONLY_TTT
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa tools/run_v46b_factorial_candidate.sh 3 F001_ONLY_SWA V46B_F001_ONLY_SWA
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa tools/run_v46b_factorial_candidate.sh 4 F110_FRAME_ATTN_TTT V46B_F110_FRAME_ATTN_TTT
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa tools/run_v46b_factorial_candidate.sh 5 F101_FRAME_ATTN_SWA V46B_F101_FRAME_ATTN_SWA
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa tools/run_v46b_factorial_candidate.sh 6 F011_TTT_SWA V46B_F011_TTT_SWA
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa tools/run_v46b_factorial_candidate.sh 7 F111_ALL_THREE V46B_F111_ALL_THREE
```

Launcher stdout/stderr：

```text
results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/launcher_logs/*.launcher.log
```

### 3.1 早退 blocker

轮询命令：

```bash
for d in results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/rollouts/V46B_*; do
  echo "---" ${d##*/}
  tail -5 "$d/run_status.txt" 2>/dev/null || echo no_status
done

ps -eo pid,ppid,stat,etime,cmd | rg 'V46B_|run_pipeline_abc_v2|run_attention_cue|run_v46b' || true

for d in results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/rollouts/V46B_*; do
  echo "---" ${d##*/}
  tail -40 "$d/01.log" 2>/dev/null || true
done
```

现象：

```text
8 个 rows 都只有 START，没有 DONE/FAIL。
没有 run_pipeline_abc_v2 / run_attention_cue / run_v46b 残留进程。
01.log 大多只写到：
  Collected 1101 images.
  Full-res images: skipped
F011/F111 只写到 RoPE warning。
```

判断：

```text
这不是有效实验结果，不能入指标表。
按 blocker 处理：先用同一 launcher 跑 64-frame debug row，确认是并发/IO/环境问题还是 launcher 配置问题。
debug row 不计入 v46B full-online ATE。
```

64-frame debug 命令：

```bash
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa \
V46B_END_FRAME=64 \
tools/run_v46b_factorial_candidate.sh 0 F000_NONE V46B_DEBUG_F000_64F
```

结果：

```text
[2026-06-08 17:39:37] DONE V46B_DEBUG_F000_64F
```

结论：

```text
同一 launcher 在 64-frame debug 下可完成并生成 01.txt。
早退更可能来自 8 路同时初始化的资源/IO/加载问题。
full-online 改为低并发批次重跑。
```

### 3.2 低并发 full-online 重跑

Batch 1:

```bash
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa \
  tools/run_v46b_factorial_candidate.sh 0 F000_NONE V46B_F000_NONE

V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa \
  tools/run_v46b_factorial_candidate.sh 1 F100_ONLY_FRAME_ATTN V46B_F100_ONLY_FRAME_ATTN
```

第一次 Batch 1 使用 `nohup ... &` 后仍早退：

```text
run_status 只有 START，无 DONE/FAIL。
ps 无残留 run_pipeline_abc_v2。
01.log 只写到 RoPE warning。
```

结合 64-frame 前台 debug 可完成，判断：

```text
这里更像当前执行环境在 shell 结束后清理后台进程组；
不是 v46B row 定义必然失败。
```

修复：

```text
不再使用 detached nohup 后台方式。
改用一个持久 shell session 内启动少量 jobs，并用 wait 保持进程组存活。
```

Batch 2:

```bash
V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa \
  tools/run_v46b_factorial_candidate.sh 2 F010_ONLY_TTT V46B_F010_ONLY_TTT

V46B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa \
  tools/run_v46b_factorial_candidate.sh 3 F001_ONLY_SWA V46B_F001_ONLY_SWA
```

### 3.3 中间报告与 Batch 3

时间：2026-06-08 18:08:05 +08

中间报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v46b_factorial_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa
```

结果：

```text
Wrote v46B report to results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1
```

中间审计：

```text
V46B_F000_NONE:
  DONE
  hmc_state_hash.jsonl rows = 38
  valid = true
  ATE = 41.750205
  action applied frame/ttt/swa = 0/0/0

V46B_F100_ONLY_FRAME_ATTN:
  DONE
  hmc_state_hash.jsonl rows = 38
  valid = true
  ATE = 38.593413
  action applied frame/ttt/swa = 1/0/0
```

未完成行：

```text
V46B_F010_ONLY_TTT:
  running
  hmc_state_hash.jsonl rows = 13 at check time
  action applied frame/ttt/swa = 0/13/0 in partial report

V46B_F001_ONLY_SWA:
  running
  hmc_state_hash.jsonl rows = 35 at check time
  action applied frame/ttt/swa = 0/0/34 in partial report
```

Batch 3 启动命令：

```bash
ROOT="results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa"

V46B_RESULT_ROOT="$ROOT" \
  tools/run_v46b_factorial_candidate.sh \
  0 F110_FRAME_ATTN_TTT V46B_F110_FRAME_ATTN_TTT \
  > "$ROOT/phase2_factorial/rollouts/V46B_F110_FRAME_ATTN_TTT.batch3_wait.log" 2>&1 &

V46B_RESULT_ROOT="$ROOT" \
  tools/run_v46b_factorial_candidate.sh \
  1 F101_FRAME_ATTN_SWA V46B_F101_FRAME_ATTN_SWA \
  > "$ROOT/phase2_factorial/rollouts/V46B_F101_FRAME_ATTN_SWA.batch3_wait.log" 2>&1 &

wait
```

说明：

```text
Batch 3 仍使用持久 shell session + wait，避免 B1 中 detached 后台进程早退。
F110/F101 是 v46B 文档强制要求补齐的 two-component rows。
```

Batch 4 启动命令：

```bash
ROOT="results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa"

V46B_RESULT_ROOT="$ROOT" \
  tools/run_v46b_factorial_candidate.sh \
  4 F011_TTT_SWA V46B_F011_TTT_SWA \
  > "$ROOT/phase2_factorial/rollouts/V46B_F011_TTT_SWA.batch4_wait.log" 2>&1 &

V46B_RESULT_ROOT="$ROOT" \
  tools/run_v46b_factorial_candidate.sh \
  5 F111_ALL_THREE V46B_F111_ALL_THREE \
  > "$ROOT/phase2_factorial/rollouts/V46B_F111_ALL_THREE.batch4_wait.log" 2>&1 &

wait
```

说明：

```text
F001_ONLY_SWA 已在 2026-06-08 18:09:19 +08 完成。
Batch 4 使用空闲 GPU4/GPU5 启动最后两个含 TTT 的组合行。
```

### 3.4 F001 中间审计

中间报告命令同 3.3。

结果：

```text
V46B_F001_ONLY_SWA:
  DONE
  hmc_state_hash.jsonl rows = 38
  row_valid = true
  no_chunk_policy_pass = true
  ATE = 41.73704103737674
  Rot = 5.320253824239548
  FinalErr = 4.0121577092299585
  action applied frame/ttt/swa = false/0/37
```

说明：

```text
F001 是 clean no-chunk 条件下 only-SWA positive isolated row。
该 row action debug 符合预期，可以进入最终 factorial 统计。
```

### 3.5 F101 中间审计

时间：2026-06-08 18:33:35 +08

完成行：

```text
V46B_F101_FRAME_ATTN_SWA:
  DONE
  hmc_state_hash.jsonl rows = 38
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v46b_factorial_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa
```

审计结果：

```text
row_valid = true
no_chunk_policy_pass = true
ATE = 38.592626794820525
Rot = 5.214515145577633
FinalErr = 2.8800223717942446
RPE_t/RPE_r = 92.392316 / 0.008647
action applied frame/ttt/swa = true/0/37
```

注意：

```text
第一次读取 F101 时，我把 report 生成和 CSV 读取并行执行，读到了旧 partial CSV，显示 missing_prediction。
随后串行重读确认 F101 为 valid。该并行读写只影响中间查看，不影响实验 artifact。
```

### 3.6 F010 中间审计

时间：2026-06-08 18:48:49 +08

完成行：

```text
V46B_F010_ONLY_TTT:
  DONE
  hmc_state_hash.jsonl rows = 38
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v46b_factorial_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa
```

审计结果：

```text
row_valid = true
no_chunk_policy_pass = true
ATE = 39.51910428415088
Rot = 6.485906780904279
FinalErr = 9.085096859131909
RPE_t/RPE_r = 92.334849 / 0.009391
action applied frame/ttt/swa = false/38/0
ttt_tri_replay_applied_layer_count_sum = 684
ttt positive/neutral/negative mass mean =
  0.3500000302903136 / 0.529985186126497 / 0.12001479024949827
```

## 4. 报告器补漏：RPE 解析

发现：

```text
v46B plan 要求记录 RPE_t / RPE_r。
报告器第一版沿用 v18 helper 计算 ATE/Rot/FinalErr，但没有把 KITTI benchmark 已落盘的 RPE 写入 registry。
```

已修复：

```text
文件：tools/v46b_factorial_report.py
修改：新增 _parse_rpe_metrics(run_dir)，从 results_sim3/results_rpe.txt 读取：
  RPE_t_full = t_err [%]
  RPE_r_full = r_err [deg/100m]
```

验证命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m py_compile tools/v46b_factorial_report.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v46b_factorial_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa
```

抽样验证：

```text
F000_NONE RPE_t/RPE_r = 92.396146 / 0.00839
F100_ONLY_FRAME_ATTN RPE_t/RPE_r = 92.393993 / 0.008656
F001_ONLY_SWA RPE_t/RPE_r = 92.394478 / 0.008383
```

审计意义：

```text
这是报告器 artifact 解析补漏，不改变 run、模型、evaluator 或指标计算。
RPE 来自同一 rollout 的 KITTI benchmark 落盘文件，不是重算或编造。
```

### 4.1 报告器补漏：SWA boundary 诊断接入口

发现：

```text
v46B plan 要求记录 SWA boundary_10f / boundary_20f。
hmc_state_hash 不直接输出该字段，但仓库已有离线工具 tools/v27_swa_boundary_diagnostics.py 可从 landed trajectories 计算 chunk-boundary 局部 ATE。
```

已修复：

```text
文件：tools/v46b_factorial_report.py
修改：新增 _read_boundary_lookup(...)。
如果 report_R1/swa_boundary/swa_boundary_summary.csv 存在，则按 run_dir 自动填入：
  swa_boundary_10f = mean_boundary_10f_ATE
  swa_boundary_20f = mean_boundary_20f_ATE
  swa_boundary_note = from v27_swa_boundary_diagnostics; reference=F000_NONE
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m py_compile tools/v46b_factorial_report.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v46b_factorial_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa
```

说明：

```text
当前尚未运行 boundary diagnostic，因为 F101/F011/F111 仍未完整落盘。
最终在所有 SWA rows 完成后执行 v27_swa_boundary_diagnostics.py，再重跑 v46B report。
```

### 4.2 速度瓶颈诊断：TTT probe write commit

触发原因：

```text
用户指出实验推进太慢，要求下一步必须发现拖累速度的原因。
```

运行中 profiling 命令：

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | rg 'V46B|run_pipeline_abc_v2|python'
nvidia-smi pmon -c 2
pidstat -p 3837667,3845304,3845305 -durh 1 5
iostat -xm 1 5

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY'
import json, pathlib, statistics
root=pathlib.Path('results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/rollouts')
rows=['V46B_F000_NONE','V46B_F100_ONLY_FRAME_ATTN','V46B_F001_ONLY_SWA','V46B_F101_FRAME_ATTN_SWA','V46B_F010_ONLY_TTT','V46B_F110_FRAME_ATTN_TTT','V46B_F011_TTT_SWA','V46B_F111_ALL_THREE']
keys=['pass1_probe_seconds','stage_b_seconds','stage_c_seconds','stage_d_seconds','pass2_control_seconds','chunk_total_seconds']
for name in rows:
    p=root/name/'timing_summary.json'
    print('---', name)
    if not p.exists():
        print('missing timing_summary')
        continue
    data=json.loads(p.read_text())
    chunks=data.get('chunks', [])
    print('chunks', len(chunks), 'model_load', data.get('model_load_seconds'))
    for k in keys:
        vals=[c.get(k,0.0) for c in chunks if isinstance(c.get(k,0.0),(int,float))]
        if vals:
            print(k, 'sum', round(sum(vals),3), 'mean', round(statistics.mean(vals),3), 'max', round(max(vals),3))
PY
```

关键发现：

```text
1. 非 TTT rows 的 wall time 约 24-26 分钟：
   F000_NONE: 17:41:07 -> 18:05:26
   F100_ONLY_FRAME_ATTN: 17:41:07 -> 18:05:48
   F001_ONLY_SWA: 17:44:33 -> 18:09:19
   F101_FRAME_ATTN_SWA: 18:07:53 -> 18:33:35

2. F010_ONLY_TTT wall time 约 64 分钟：
   17:44:33 -> 18:48:49

3. pidstat 显示 TTT rows 不是 I/O wait：
   iodelay=0，kB_rd/s=0，kB_wr/s≈0。
   iostat 采样中 %iowait≈0。

4. GPU 利用率不是瓶颈：
   nvidia-smi pmon 采样中三个 TTT 进程 SM 多数为 0-9%。

5. CPU 侧是主要活动：
   三个 TTT Python 进程 CPU 大约 110%-394%，并伴随大量 minor faults。

6. timing_summary.json 显示 TTT rows 的 chunk_total_seconds 大幅增大：
   F000 mean chunk_total_seconds ≈ 36.537s
   F100 mean chunk_total_seconds ≈ 37.057s
   F001 mean chunk_total_seconds ≈ 37.114s
   F101 mean chunk_total_seconds ≈ 38.298s
   F010 mean chunk_total_seconds ≈ 99.638s
   F110 partial mean chunk_total_seconds ≈ 105.354s
   F011 partial mean chunk_total_seconds ≈ 107.180s
   F111 partial mean chunk_total_seconds ≈ 100.246s

7. 已记录的 pass1/stageB/stageD/pass2 合计约 31-34 秒，不能解释 TTT rows 的 100 秒级 chunk total。
   对照代码路径，未分项计时的主要 TTT-only 区间是：
   hmc.build_probe_ttt_write_state(...)
```

因此当前速度瓶颈定位为：

```text
probe_ttt_write commit 阶段，尤其是 hmc.build_probe_ttt_write_state(...)。
不是磁盘 I/O，不是 GPU 满载，也不是 Stage B / Stage D 本身。
```

修复/审计改动：

```text
文件：run_pipeline_abc_v2.py
修改：在 chunk_timing 中新增 probe_ttt_write_seconds，并围绕 hmc.build_probe_ttt_write_state(...) 计时。
目的：后续 probe_ttt_write runs 能直接在 timing_summary.json 中记录该阶段耗时，不再只能用 chunk_total_seconds 减法推断。
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile run_pipeline_abc_v2.py
```

审计说明：

```text
该计时补丁不影响当前正在运行的三个 Python 进程，因为它们已加载旧代码。
它也不改变算法逻辑、模型参数、输出格式或 evaluator，只增加 future timing_summary 字段。
当前三个 run 的速度结论来自已经落盘的 timing_summary.json 和系统采样。
```

## 2026-06-08 Final 收口：8-row factorial 全部完成

### 最后三个 TTT rows 完成状态

检查命令：

```bash
for d in results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/rollouts/V46B_F*; do
  [ -d "$d" ] || continue
  base=$(basename "$d")
  case "$base" in *.log) continue;; esac
  echo "--- $base"
  cat "$d/run_status.txt"
done
```

关键输出：

```text
V46B_F110_FRAME_ATTN_TTT:
  START 2026-06-08 18:07:53
  DONE  2026-06-08 19:16:12

V46B_F111_ALL_THREE:
  START 2026-06-08 18:10:31
  DONE  2026-06-08 19:14:49

V46B_F011_TTT_SWA:
  START 2026-06-08 18:10:31
  DONE  2026-06-08 19:19:17
```

完整性：

```text
8/8 rows DONE
8/8 rows hmc_state_hash.jsonl = 38 rows
8/8 rows have 01.txt
8/8 rows have results_sim3/results_rpe.txt
```

### SWA boundary diagnostic

命令：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v27_swa_boundary_diagnostics.py \
  --reference-run "$ROOT/phase2_factorial/rollouts/V46B_F000_NONE" \
  --candidate-run "$ROOT/phase2_factorial/rollouts/V46B_F001_ONLY_SWA" \
  --candidate-run "$ROOT/phase2_factorial/rollouts/V46B_F101_FRAME_ATTN_SWA" \
  --candidate-run "$ROOT/phase2_factorial/rollouts/V46B_F011_TTT_SWA" \
  --candidate-run "$ROOT/phase2_factorial/rollouts/V46B_F111_ALL_THREE" \
  --out-dir "$ROOT/phase2_factorial/report_R1/swa_boundary" \
  --chunk-size 32 \
  --overlap 3
```

输出文件：

```text
results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1/swa_boundary/swa_boundary_summary.csv
results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1/swa_boundary/swa_boundary_summary.json
results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1/swa_boundary/swa_boundary_by_candidate_boundary.csv
```

### Final report 生成

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/v46b_factorial_report.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v46b_factorial_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa
```

输出：

```text
Wrote v46B report to results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1
```

最终 report artifacts：

```text
phase2_factorial_registry.csv
phase2_factorial_registry.json
phase2_component_main_effects.csv
phase2_component_interactions.csv
phase2_frame_ttt_swa_summary.md
component_interaction_heatmap.png
segment_delta_stacked_bar.png
swa_boundary/swa_boundary_summary.csv
swa_boundary/swa_boundary_by_candidate_boundary.csv
```

### Final validity check

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY'
import csv, pathlib, sys
root=pathlib.Path('results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1')
rows=list(csv.DictReader((root/'phase2_factorial_registry.csv').open()))
errors=[]
for r in rows:
    if r['row_valid'] != 'True': errors.append(f"invalid {r['row']}: {r['invalid_reason']}")
    if r['run_status_done'] != 'True': errors.append(f"not done {r['row']}")
    if r['no_chunk_policy_pass'] != 'True': errors.append(f"chunk audit fail {r['row']}")
    if r['hmc_rows'] != '38': errors.append(f"hmc rows {r['row']}={r['hmc_rows']}")
    if r['frames'] != '1101': errors.append(f"frames {r['row']}={r['frames']}")
for p in [
    root/'phase2_factorial_registry.csv',
    root/'phase2_factorial_registry.json',
    root/'phase2_component_main_effects.csv',
    root/'phase2_component_interactions.csv',
    root/'phase2_frame_ttt_swa_summary.md',
    root/'component_interaction_heatmap.png',
    root/'segment_delta_stacked_bar.png',
    root/'swa_boundary/swa_boundary_summary.csv',
    root/'swa_boundary/swa_boundary_by_candidate_boundary.csv',
]:
    if not p.exists() or p.stat().st_size == 0:
        errors.append(f"missing/empty {p}")
print('registry_rows', len(rows))
print('errors', errors)
if len(rows) != 8 or errors:
    sys.exit(1)
PY
```

输出：

```text
registry_rows 8
errors []
```

### Report 噪音修复

修复文件：

```text
tools/v46b_factorial_report.py
```

修复内容：

```text
1. READ_x_SWA interaction 的 boundary_note 改为在 boundary diagnostic 已存在时记录真实来源和 best_10f。
2. phase2_frame_ttt_swa_summary.md 的 Audit Notes 改为条件化说明：
   已有 boundary diagnostic 时写入来自 v27_swa_boundary_diagnostics；
   没有时才写 missing note。
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/v46b_factorial_report.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v46b_factorial_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa
```

审计说明：

```text
该修复只改变 report 文本字段，不改变任何 rollout、benchmark、ATE、RPE 或 action audit。
```

## 2026-06-08 Final 指标摘要

Final registry：

| Row | FRAME_ATTN | TTT | SWA | ATE_full | Rot_full | FinalErr_full | RPE_t | RPE_r | row_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `F000_NONE` | 0 | 0 | 0 | 41.750205342038015 | 5.354471107510558 | 3.923875981533656 | 92.396146 | 0.00839 | True |
| `F100_ONLY_FRAME_ATTN` | 1 | 0 | 0 | 38.593412804190564 | 5.250862613050851 | 2.982166037805976 | 92.393993 | 0.008656 | True |
| `F010_ONLY_TTT` | 0 | 1 | 0 | 39.51910428415088 | 6.485906780904279 | 9.085096859131909 | 92.334849 | 0.009391 | True |
| `F001_ONLY_SWA` | 0 | 0 | 1 | 41.73704103737674 | 5.320253824239548 | 4.0121577092299585 | 92.394478 | 0.008383 | True |
| `F110_FRAME_ATTN_TTT` | 1 | 1 | 0 | 36.668910841145234 | 6.463437289271693 | 14.699225179343587 | 92.33228 | 0.009781 | True |
| `F101_FRAME_ATTN_SWA` | 1 | 0 | 1 | 38.592626794820525 | 5.214515145577633 | 2.8800223717942446 | 92.392316 | 0.008647 | True |
| `F011_TTT_SWA` | 0 | 1 | 1 | 39.503549725643346 | 6.44974548970202 | 9.00026345482514 | 92.33275 | 0.009383 | True |
| `F111_ALL_THREE` | 1 | 1 | 1 | 36.65074045045878 | 6.418129083708555 | 14.614346164545774 | 92.330293 | 0.009768 | True |

Main effects：

| factor/combo | row | ATE_full | gain_vs_F000 |
|---|---|---:|---:|
| `FRAME_ATTN_only` | `F100_ONLY_FRAME_ATTN` | 38.593412804190564 | 3.1567925378474513 |
| `TTT_only` | `F010_ONLY_TTT` | 39.51910428415088 | 2.2311010578871375 |
| `SWA_only` | `F001_ONLY_SWA` | 41.73704103737674 | 0.013164304661273718 |
| `FRAME_ATTN_TTT` | `F110_FRAME_ATTN_TTT` | 36.668910841145234 | 5.081294500892781 |
| `FRAME_ATTN_SWA` | `F101_FRAME_ATTN_SWA` | 38.592626794820525 | 3.15757854721749 |
| `TTT_SWA` | `F011_TTT_SWA` | 39.503549725643346 | 2.246655616394669 |
| `ALL_THREE` | `F111_ALL_THREE` | 36.65074045045878 | 5.099464891579238 |

Interactions：

| interaction | classification | combo gain | margin |
|---|---|---:|---:|
| `READ_x_TTT` | `synergy` | 5.081294500892781 | 1.92450196304533 |
| `READ_x_SWA` | `swa_near_zero_under_read` | 3.15757854721749 | 0.0007860093700386983 |
| `TTT_x_SWA` | not classified | 2.246655616394669 | 0.015554558507531624 |
| `THREE_WAY` | not classified | 5.099464891579238 | 0.018170390686456983 vs best pair |

SWA boundary：

| candidate | boundary_count | mean 10f ATE | 10f delta vs H9 | mean 20f ATE | 20f delta vs H9 | gate |
|---|---:|---:|---:|---:|---:|---|
| `V46B_F001_ONLY_SWA` | 37 | 36.830794447749334 | -0.021091095523154024 | 36.93193481489409 | -0.019371327704718055 | False |
| `V46B_F101_FRAME_ATTN_SWA` | 37 | 33.25907636391737 | -3.5928091793551147 | 33.38934469468534 | -3.5619614479134682 | False |
| `V46B_F011_TTT_SWA` | 37 | 34.35453498575968 | -2.49735055751281 | 34.48241788958564 | -2.4688882530131693 | False |
| `V46B_F111_ALL_THREE` | 37 | 31.061558973199865 | -5.790326570072622 | 31.221660230588952 | -5.7296459120098575 | True |

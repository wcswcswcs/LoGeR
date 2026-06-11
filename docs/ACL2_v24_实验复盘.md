# ACL2 v24 实验复盘：SemanticPrior PathSpecific AllMemory Parallel

日期：2026-05-22（Asia/Singapore）  
计划文件：`docs/ACL2_v24_SemanticPrior_PathSpecific_AllMemory_Parallel_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v24_semanticprior_pathspecific_allmemory_parallel/`

本轮原则：只记录实际落盘结果；不把静态代码接线、自查脚本、smoke、short rollout、failed gate、coarse fallback fine-label 候选、或未启动矩阵写成 deployable online success。没有通过 Phase 0 implementation audit、Phase 2/3/4 durability gate、以及 no-GT selector gate 时，不启动 full online validation。

---

## 0. 当前结论

v24 已按当前 speed-gated Phase 2 策略执行到 stop rule，未达到 Target-25。

已完成并落盘：

```text
1. v24 专用 launcher / matrix / report / self-audit / passive-audit 脚本已实现。
2. Phase 0 smoke 完成 6/6。
3. Phase 0 no-op / pass-through parity 通过：
       K1_H9 / P0_01 / P0_02 / P0_03
       ATE_delta_vs_H9 = 0
       raw_trans_max_diff = 0
4. v24 implementation self-audit 当前 all_gate_pass = true。
5. Phase 1 passive semantic attribution 完成并通过审计。
6. Phase 2 h10 screen 完成：
       chunk10 h10 rows = 24/24
       failures = 0
7. Phase 2 h15 top confirmation 完成：
       selected h15 rows = 6/6
       plus early completed FRAMESEM h15 rows = 3
       h15 report rows = 9
       failures = 0
8. Phase 2 gate = fail。
```

最终边界：

```text
1. h10 full screen best ATE delta = -0.3175006931m。
2. h10 full screen best [200,300) delta = -1.9404135647m。
3. h15 top confirmation best ATE delta = -0.1383555865m。
4. h15 top confirmation best [200,300) delta = -1.0269153793m。
5. No Phase 2 candidate passed gate.
6. No Phase 3 pairwise combination was started.
7. No Phase 4 all-memory validation was started.
8. No no-GT selector was started.
9. No full online validation was launched.
10. No online Target-25 result was produced.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

---

## 1. 工程修改

新增：

```text
tools/run_v24_candidate_rollout.sh:
    v24 trusted short-rollout launcher
    supports Phase 0 smoke, Phase 1 passive debug, Phase 2 single-path,
    Phase 3 pairwise, Phase 4 all-memory, Phase 5 attribution candidates
    records fine_semantic_split_available / coarse fallback boundary
    moves stale/forced run dirs to .INVALID_RERUN_* before rerun
    uses warm-path checkpoint/snapshots when available

tools/run_v24_matrix.sh:
    phase0
    phase1
    phase2_initial
    phase2_expand
    phase3
    phase4
    phase5_attr

tools/v24_candidate_bank_report.py:
    aggregates v24 short-rollout candidate metrics
    overlays v24 Phase 2 / Phase 3 / Phase 4 gate fields

tools/v24_implementation_self_audit.py:
    static + dynamic Phase 0 hard gate
    writes required audit files:
        codex_self_check_report.md
        codex_self_check_summary.json
        codex_self_check_failures.jsonl
        semantic_role_alignment_audit.csv
        path_consumption_audit.csv
        noop_parity_metrics.csv

tools/v24_passive_semantic_audit.py:
    aggregates semantic group coverage and memory path consumption
    writes aggregate CSV/PNG passive attribution artifacts
    also reads Stage C predicted fine-label audit to separate fine-label coverage
    from runtime coarse-group memory-path policy

tools/run_v24_phase2_initial_queue.sh:
    dynamic GPU worker queue for Phase 2 initial matrix
    skips rows whose logs already contain DONE
    reruns stale/interrupted rows with FORCE=1
    avoids batch-level waiting where fast h10 rows leave GPUs idle behind slow h15 rows
```

修改：

```text
tools/run_attention_cue_experiment.sh:
    passes --output_video "" by default to disable main pipeline mp4 export
    this removes repeated results/pipeline_v2.mp4 writes from future short-rollout rows
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v24_candidate_bank_report.py \
    tools/v24_implementation_self_audit.py \
    tools/v24_passive_semantic_audit.py

bash -n tools/run_v24_candidate_rollout.sh
bash -n tools/run_v24_matrix.sh

PASS
```

重要边界：

```text
Stage C cache audit exposes predicted fine labels:
    building, fence, grass, road, sidewalk, sky, vegetation, wall

Current runtime semantic-memory path metrics are still coarse-group keyed.
Candidates with sky/vegetation wording are run only as coarse LOW_VALUE_STUFF fallback diagnostics
unless later code explicitly forwards fine labels into runtime policy.
They are not claimed as GT semantic policies or true fine-runtime-policy successes.
```

---

## 2. Phase 0：Implementation Audit / No-op Gate

输出：

```text
phase0_plumbing_report_R1/
implementation_audit/
rollouts/V24_P0_SMOKE_R1_*
```

矩阵：

```text
K1_H9
P0_01_SEMANTIC_ROLE_NOOP_IGNORED
P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED
P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY
FRAMESEM_02_LOWSTUFF_HIGHD_SKIP
ALLMEM_03_FRAME_GLOBAL_SWA_TTT_PATHSPEC

chunk = 10
horizon = 3
rows completed = 6/6
```

No-op parity：

| Candidate | ATE delta vs H9 | raw trans max diff |
|---|---:|---:|
| `K1_H9` | `0.0` | `0.0` |
| `P0_01_SEMANTIC_ROLE_NOOP_IGNORED` | `0.0` | `0.0` |
| `P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED` | `0.0` | `0.0` |
| `P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY` | `0.0` | `0.0` |

Path smoke evidence：

```text
FRAMESEM_02_LOWSTUFF_HIGHD_SKIP:
    context source skip applied
    empty source events = 0

ALLMEM_03_FRAME_GLOBAL_SWA_TTT_PATHSPEC:
    semantic_memory_paths = frame,global,swa,ttt
    diagnostic smoke only; not a performance success
```

Self-audit result：

| Gate | Result |
|---|---|
| hard_static_gate_pass | `true` |
| dynamic_smoke_gate_pass | `true` |
| all_gate_pass | `true` |

Blocker / fix:

```text
blocker:
    v24_implementation_self_audit.py initially read raw_trans_max_diff_vs_H9,
    while v24 candidate report writes raw_trans_max_diff.

fix:
    audit script now accepts raw_trans_max_diff as the landed report column.

result:
    dynamic_noop_parity_zero = true
```

Decision：

```text
Phase 0 gate = pass
Phase 1 passive attribution is allowed.
No deployable online result is claimed.
No selector/full online validation is allowed.
```

---

## 3. Phase 1：Passive Semantic Attribution

输出：

```text
phase1_passive_report_R1/
phase1_passive_semantic_audit/
rollouts/V24_P1_PASSIVE_R1_*
```

矩阵：

```text
K1_H9
PASSIVE_DEBUG_ONLY

chunks = 6, 10, 16
horizons = 10, 15
rows completed = 12/12
```

Passive audit summary：

| Metric | Value |
|---|---:|
| semantic memory metric runs | `18` |
| runtime coarse groups seen | `3` |
| coarse group diversity gate | `false` |
| Stage C predicted fine labels | `8` |
| fine label diversity gate | `true` |
| phase1 gate pass | `true` |
| context empty source event runs | `0` |

Fine-label boundary：

```text
fine label source:
    results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/
        semantic_audit/label_counts_by_chunk.csv

fine labels seen:
    building, fence, grass, road, sidewalk, sky, vegetation, wall

is_gt_semantic = false
runtime_fine_role_policy_available = false
```

Consumed-path counts：

```text
frame  = 8
global = 4
swa    = 4
ttt    = 4
```

Blocker / fix:

```text
blocker:
    Phase 1 audit initially failed because runtime semantic_memory_path_summary
    only reports coarse groups, and sequence-01 chunks observed only groups 0/3/4.

fix:
    tools/v24_passive_semantic_audit.py now reads the existing Stage C predicted
    fine-label audit and records it separately from runtime coarse-group metrics.
    The script explicitly marks fine_label_is_gt_semantic = false and
    runtime_fine_role_policy_available = false.

result:
    Phase 1 can distinguish "fine labels exist in predicted masklet cache" from
    "runtime policy is still coarse fallback".
```

Decision：

```text
Phase 1 gate = pass for passive attribution.
Phase 2 single-path initial matrix is allowed.

No GT semantic row exists in this local dataset.
No deployable online result is claimed.
No selector/full online validation is allowed.
```

---

## 4. Phase 2 速度自查与调度修复

用户指出：

```text
速度太慢了；现在跑一次时间太久，严重影响效率。
```

自查发现：

```text
1. 初始 Phase 2 调度是 6 row 一批，并等待整批全部结束后才启动下一批。
   h10 row 通常早于 h15 row 完成，导致部分 GPU 空等慢 h15。

2. 尝试验证 h15 轨迹是否可直接截出 h10 指标：
       FRAMESEM_01 / FRAMESEM_02 / FRAMESEM_03
       h10 轨迹 vs h15 前缀 prefix_exact = 0
   因此不能未经验证用 h15 截断替代 h10，否则有造假风险。

3. run_pipeline_abc_v2.py 默认 output_video=results/pipeline_v2.mp4。
   已完成早期 Phase 2 rows 的日志里出现 repeated:
       Saved video to results/pipeline_v2.mp4
   这是额外 I/O，后续 row 已关闭。
```

修复：

```text
1. 停止旧 batch controller。
   已完成的 9 row 保留。
   被中断的半截 row 由后续 FORCE rerun 移到 .INVALID_RERUN_*，不纳入报告。

2. 新增 dynamic queue runner:
       tools/run_v24_phase2_initial_queue.sh
   queue rows = 39
   skips DONE rows = 9
   原始配置使用 GPUs = 0..5。
   后续用户确认 GPU 0/1/2/3 可用，因此继续运行时限制为 GPUs = 0..3。

3. tools/run_attention_cue_experiment.sh 默认传入:
       --output_video ""
   后续 short-rollout 不再写主 pipeline mp4。
```

进一步自查：

```text
dynamic queue 只解决 GPU 空等和视频 I/O。
它没有解决真正主因：
    每个 candidate / horizon 都重新启动 Python，
    重新加载模型，
    并对相同 chunk/frame/input state 重复执行 Pass 1 probe 和 Stage B cue。

对 read_path_only + probe_native commit 的候选，Pass 1 probe 和 Stage B cue
不依赖 semantic role candidate，可以在相同 chunk/start/end/input-state 下复用。
之前没有复用，导致 Phase 2 read-path 候选时间被候选数量直接放大。
```

代码修复：

```text
run_pipeline_abc_v2.py:
    added --probe_cache_dir
    added --probe_cache_mode off/read/write/readwrite/refresh
    added --probe_cache_payload read_path_min/full
    added --probe_cache_require_hit
    writes timing_summary.json:
        model_load_seconds
        per-chunk pass1_probe_seconds
        per-chunk stage_b_seconds
        per-chunk stage_c_seconds
        per-chunk stage_d_seconds
        per-chunk pass2_control_seconds
        per-chunk chunk_total_seconds
        probe_cache_hit
        cue_cache_hit

tools/run_attention_cue_experiment.sh:
    forwards PROBE_CACHE_* args
    writes wall_time_summary.json for every run

tools/run_v24_candidate_rollout.sh:
    forwards V24_PROBE_CACHE_* to the Python runner

tools/run_v24_phase2_initial_queue.sh:
    supports V24_PHASE2_HORIZONS
    supports V24_PHASE2_CANDIDATES
    records row wall_seconds in queue logs

tools/v24_runtime_compare.py:
    compares trajectory reproducibility
    compares wall/runtime summaries
```

错误优化与回滚：

```text
first probe-cache attempt stored full ProbeOutput including TTT replay primitives.
This was too large:
    5 cached chunks ~= 52GB

Action:
    stopped the run
    deleted the full cache
    changed default cache payload to read_path_min

read_path_min keeps enough for read-path reuse:
    GeometryOutput
    native provisional state
    Stage B CueOutput
It drops:
    TTT replay primitives
    raw debug predictions

Boundary:
    read_path_min cache is not used for probe_ttt_write TTT commit rows.
    Those rows require real probe replay primitives and must rerun Pass 1.
```

复现与耗时验证：

```text
baseline row:
    V24_P2_INITIAL_R1_FRAMESEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT_chunk10_h10_globalgate_H9parent_SWKS3
    wall_seconds = 456

cachewrite R2:
    V24_SPEED_REPRO_CACHEWRITE_R2_FRAMESEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT_chunk10_h10_globalgate_H9parent_SWKS3
    wall_seconds = 488
    trajectory exact reproduction = true
    matched rows = 322/322
    max_translation_abs_diff = 0
    max_quaternion_abs_diff = 0
    probe_cache_hits = 0
    cache write size after full h10 ~= 21GB

cachehit R2:
    V24_SPEED_REPRO_CACHEHIT_R2_FRAMESEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT_chunk10_h10_globalgate_H9parent_SWKS3
    wall_seconds = 215
    trajectory exact reproduction = true
    matched rows = 322/322
    max_translation_abs_diff = 0
    max_quaternion_abs_diff = 0
    probe_cache_hits = 11/11
    wall speedup vs baseline = 2.1209302326x

Timing evidence:
    cachewrite R2 pass1_probe_seconds_sum = 127.6295084953
    cachehit R2 pass1_probe_seconds_sum   = 5.6741709709
    cachehit R2 chunk10 first chunk:
        probe_cache_hit = true
        cue_cache_hit = true
        pass1_probe_seconds = 0.56
        stage_b_seconds = 0.0
        chunk_total_seconds = 20.72
```

Phase 2 speed-gated rerun：

```text
The old exhaustive Phase 2 dynamic queue was stopped.
DONE rows before stop = 9.
Interrupted / partial rows are not counted.

Rerun settings:
    V24_GPUS="0 1 2 3"
    V24_PROBE_CACHE_MODE=readwrite
    V24_PROBE_CACHE_PAYLOAD=read_path_min
    V24_PHASE2_HORIZONS=10 for h10 screen first

h10 screen:
    matrix log dir = phase2_initial_R1
    h10 rows completed = 24/24
    h10 failures = 0
    queue completed at 2026-05-22T05:06:06+08:00

h15 confirmation:
    matrix log dir = phase2_h15_top_R1
    selected h15 rows completed = 6/6
    selected h15 failures = 0
    queue completed at 2026-05-22T05:28:38+08:00
```

Runtime observations:

```text
h10 queue rows with recorded wall_seconds:
    count = 18
    min wall_seconds = 218
    max wall_seconds = 510

h10 slowest rows:
    SWASEM_01_STRUCTURE_CACHE_KEEP              510s
    SWASEM_02_LOWSTUFF_HIGHD_CACHE_SOFTDROP    499s
    TTTSEM_04_SEMANTIC_PLUS_TTT_CONFLICT       485s
    SWASEM_03_SKY_PROTECT_VEG_HIGHD_DROP       475s
    TTTSEM_07_ROLE_SPECIFIC_LONG_SHORT         469s

h15 selected rows with recorded wall_seconds:
    count = 6
    min wall_seconds = 458
    max wall_seconds = 664

h15 slowest rows:
    SWASEM_05_OVERLAP_ONLY                     664s
    TTTSEM_01_STRUCTURE_POSITIVE_LONG          627s
    CHUNKSEM_01_STRUCTURE_KEEP                 482s
```

Speed interpretation:

```text
The cache optimization is real for read-path reuse:
    baseline FRAMESEM_01 h10 wall_seconds = 456
    cachehit FRAMESEM_01 h10 wall_seconds = 215
    exact trajectory reproduction = true
    speedup = 2.1209302326x

But this does not solve the full bottleneck:
    SWA / TTT probe_ttt_write rows still take about 450-510s for h10.
    selected h15 rows take about 458-664s.

Therefore the current optimization is partial.
It is not acceptable to resume full h15/all-chunk sweeping from this family.
Future speed work must split and cache candidate-independent TTT/SWA replay-input construction,
or run a persistent model worker instead of restarting Python/model load per row.
```

---

## 5. Phase 2 Single-Path Screen Result

输出：

```text
phase2_h10_screen_report_R1/
phase2_h15_top_report_R1/
matrix_logs/phase2_initial_R1/
matrix_logs/phase2_h15_top_R1/
```

h10 screen matrix:

```text
chunk = 10
horizon = 10
rows completed = 24/24
failures = 0
GPUs = 0,1,2,3
probe_cache_mode = readwrite
probe_cache_payload = read_path_min
```

h10 gate summary:

| Metric | Best |
|---|---:|
| Best h10 ATE delta vs H9 | `-0.3175006931` |
| Best h10 ATE candidates | `GLOBALSEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT`, `GLOBALSEM_02_LOWSTUFF_HIGHD_SKIP`, `FRAMEGLOBAL_02_GLOBAL_ONLY`, `CHUNKSEM_01_STRUCTURE_KEEP`, `CHUNKSEM_02_LOWSTUFF_HIGHD_SKIP`, `CHUNKSEM_03_PROTECT_SPECIAL_TOKENS` |
| Best h10 `[200,300)` delta vs H9 | `-1.9404135647` |
| Best h10 `[400,600)` delta vs H9 | `-0.3310072074` |
| Phase 2 pass candidates | `[]` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

h15 confirmation strategy:

```text
The h10 screen did not pass the Phase 2 local gate:
    required h10 [200,300) delta <= -3m
    observed best h10 [200,300) delta = -1.9404135647m

To avoid a full slow h15 sweep, only a top/representative subset was run:
    GLOBALSEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT
    GLOBALSEM_02_LOWSTUFF_HIGHD_SKIP
    FRAMEGLOBAL_02_GLOBAL_ONLY
    CHUNKSEM_01_STRUCTURE_KEEP
    SWASEM_05_OVERLAP_ONLY
    TTTSEM_01_STRUCTURE_POSITIVE_LONG

The h15 report also includes three earlier completed FRAMESEM h15 rows.
No stale START-only h15 logs are counted.
```

h15 top confirmation:

```text
selected h15 rows completed = 6/6
early completed FRAMESEM h15 rows = 3
report rows = 9
failures = 0
```

h15 gate summary:

| Metric | Best |
|---|---:|
| Best h15 ATE delta vs H9 | `-0.1383555865` |
| Best h15 ATE candidate | `SWASEM_05_OVERLAP_ONLY` |
| Best h15 `[200,300)` delta vs H9 | `-1.0269153793` |
| Best h15 `[400,600)` delta vs H9 | `+0.0644565686` |
| Phase 2 pass candidates | `[]` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Decision:

```text
Phase 2 gate = fail.

Required to enter Phase 3:
    h10 [200,300) delta <= -3m
    or h15 ATE delta <= -1.5m
    or h15 [200,300) delta <= -2.5m
    with [400,600) regression <= +1m
    and empty source events = 0
    and path metrics proving real source/write mass change.

Observed:
    best h10 [200,300) delta = -1.9404135647m
    best h15 ATE delta = -0.1383555865m
    best h15 [200,300) delta = -1.0269153793m

Therefore no candidate is allowed into Phase 3 pairwise combination.
No Phase 4 all-memory role controller is launched.
No durability attribution / selector / full online validation is launched.
```

Interpretation:

```text
v24 path-specific semantic single-path policies create only weak local changes on chunk10.
The strongest h10 disease-window improvement comes from global/chunk source behavior,
but it is below the Phase 2 gate and does not persist in the h15 top confirmation.

SWA overlap-only is the best h15 top-confirmation row, but its effect is small:
    h15 ATE delta = -0.1383555865m
    h15 [200,300) delta = -1.0269153793m

This supports stopping the current v24 semantic path-specific family here,
instead of spending more GPU time on full h15/chunk6/chunk16 sweeps.
```

---

## 6. GT Semantic Oracle Diagnostic Boundary

用户提出：

```text
能不能使用 GT semantic 信息，排除因为 video masklet 预测质量差的问题。
```

结论：

```text
可以，但只能作为 oracle semantic diagnostic / upper-bound audit。
不允许把 GT semantic row 计为 deployable online success。
不允许把 GT semantic row 用于 no-GT selector 训练或 full online 成功声明。
```

当前本地数据检查：

```text
/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/
    calib.txt
    image_2/
    image_3/
    times.txt

未发现 sequence 01 下有 labels / semantic / velodyne semantic label 目录。
因此当前标准 KITTI Odometry 本地副本没有可直接使用的 dense GT semantic。
```

如果后续提供 SemanticKITTI / KITTI-STEP / dense semantic GT：

```text
新增 Phase 1b GT-semantic oracle audit:
    GT semantic -> patch/token role projection
    predicted masklet role vs GT role IoU / confusion
    predicted-masklet short rollout vs GT-semantic oracle short rollout

边界：
    counts_as_online_ttt_write_success = false
    uses_gt_runtime_action = true
    selector_allowed = false
    full_online_validation_allowed = false
```

# ACL2 v36B 执行日志：NoOverblocking SemanticMemory Control Target30

日期：2026-05-24（Asia/Singapore）  
计划文件：`docs/ACL2_v36B_NoOverblocking_SemanticMemory_Control_Target30_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/`

本日志只记录实际执行过或复用的 landed artifact。每个 LoGeR 进程只绑定一张 GPU；并行来自多个独立 row 分配到不同 GPU。

---

## 0. 固定环境

工作目录：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
```

Python：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
```

用户确认可用 GPU：

```text
0,1,2,3 initially available
4,5,6,7 later reported temporarily available
```

GPU 使用原则：

```text
one LoGeR process per GPU
no multi-GPU single run
do not duplicate an active run directory
```

---

## 1. 新增 / 使用的 v36B 文件

新增或使用的 v36B 专用脚本：

```text
tools/run_v36b_snapshot_generation.sh
tools/run_v36b_h0c_action_smoke.sh
tools/v36b_h0c_action_smoke_report.py
tools/v36b_context_skip_summary.py
tools/v36b_h0b_attention_mass_feasibility_audit.py
tools/run_v36b_h1_h10.sh
tools/run_v36b_path_h10.sh
```

验证命令：

```bash
bash -n tools/run_v36b_snapshot_generation.sh
bash -n tools/run_v36b_h0c_action_smoke.sh
bash -n tools/run_v36b_h1_h10.sh
bash -n tools/run_v36b_path_h10.sh

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v36b_h0c_action_smoke_report.py \
  tools/v36b_context_skip_summary.py \
  tools/v36b_h0b_attention_mass_feasibility_audit.py
```

结果：

```text
PASS
```

---

## 2. Phase 0：Parent Snapshot 与 H0C

### 2.1 Parent snapshot generation

执行：

```bash
tools/run_v36b_snapshot_generation.sh
```

落盘 parent snapshots：

```text
phase0_parent_snapshots/state_snapshots/H9_V36B_R1/chunk_006_input.pt
phase0_parent_snapshots/state_snapshots/H9_V36B_R1/chunk_010_input.pt
phase0_parent_snapshots/state_snapshots/H9_V36B_R1/chunk_016_input.pt
phase0_parent_snapshots/state_snapshots/C9_V36B_R1/chunk_006_input.pt
phase0_parent_snapshots/state_snapshots/C9_V36B_R1/chunk_010_input.pt
phase0_parent_snapshots/state_snapshots/C9_V36B_R1/chunk_016_input.pt

phase0_parent_snapshots/merge_state_snapshots/H9_V36B_R1/chunk_006_input.pt
phase0_parent_snapshots/merge_state_snapshots/H9_V36B_R1/chunk_010_input.pt
phase0_parent_snapshots/merge_state_snapshots/H9_V36B_R1/chunk_016_input.pt
phase0_parent_snapshots/merge_state_snapshots/C9_V36B_R1/chunk_006_input.pt
phase0_parent_snapshots/merge_state_snapshots/C9_V36B_R1/chunk_010_input.pt
phase0_parent_snapshots/merge_state_snapshots/C9_V36B_R1/chunk_016_input.pt
```

边界：

```text
Full-sequence snapshot wrapper rows were stopped after required artifacts landed.
Those wrapper rows are not counted as DONE trajectory rows.
```

### 2.2 H0C action smoke

执行：

```bash
V36B_H0C_CHUNK=6 \
V36B_GPUS=2,3 \
V36B_MAX_PARALLEL=2 \
tools/run_v36b_h0c_action_smoke.sh
```

报告：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v36b_h0c_action_smoke_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0c_action_smoke/rollouts \
  --run-prefix V36B_H0C_SMOKE_R1_H9 \
  --chunk 6 \
  --horizon 3 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0c_action_smoke/report_R1
```

结果：

```text
h0c_action_smoke_gate_pass = true
rows_done = 7
missing_rows = 0
any_non_base_pair_distinguishable = true
any_source_skip_effect = true
context_empty_source_events_total = 0
```

H0C collapse observation：

```text
FG_RISK_00 / FG_SEM_01 / FG_SEM_02 / FG_SEM_03 collapse to same source-skip pattern.
FG_SEM_04 / FG_SEM_05 collapse to same source-skip pattern.
```

因此 H1 只跑代表项：

```text
V31_BASE_H9_REFERENCE
FG_RISK_00
FG_SEM_04
```

### 2.3 H0B attention-mass feasibility audit

触发原因：

```text
v36B allows short rollout without blocking on H0B, but attention-mass claims
still require explicit audit. User also asked to check VGGT4D-style region/source
attention skip carefully.
```

先读代码：

```text
third_party/VGGT4D/vggt4d/layers/attention.py
third_party/VGGT4D/vggt4d/models/aggregator.py
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py
loger/models/layers/attention.py
loger/models/pi3.py
```

执行：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v36b_h0b_attention_mass_feasibility_audit.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v36b_h0b_attention_mass_feasibility_audit.py \
  --repo-root . \
  --results-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0b_attention_mass_feasibility_audit_R1 \
  --sample-context-jsonl results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/rollouts/V36B_H1_H10_R2_H9_FG_RISK_00_chunk10_h10_globalgate_H9parent_SWKS3/context_skip_summary.jsonl \
  --sample-context-jsonl results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/rollouts/V36B_H1_H10_R2_H9_FG_SEM_04_chunk10_h10_globalgate_H9parent_SWKS3/context_skip_summary.jsonl \
  --sample-context-jsonl results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/rollouts/V36B_H1_H10_R2_C9_FG_RISK_00_chunk10_h10_globalgate_H9parent_SWKS3/context_skip_summary.jsonl \
  --sample-context-jsonl results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/rollouts/V36B_H1_H10_R2_C9_FG_SEM_04_chunk10_h10_globalgate_H9parent_SWKS3/context_skip_summary.jsonl \
  --sample-hmc-jsonl results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/rollouts/V36B_H1_H10_R2_H9_FG_RISK_00_chunk10_h10_globalgate_H9parent_SWKS3/hmc_state_hash.jsonl \
  --sample-hmc-jsonl results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/rollouts/V36B_H1_H10_R2_C9_FG_RISK_00_chunk10_h10_globalgate_H9parent_SWKS3/hmc_state_hash.jsonl
```

输出：

```text
phase0b_attention_mass_feasibility_audit_R1/
    h0b_artifact_field_audit.csv
    h0b_attention_mass_feasibility_summary.json
    h0b_attention_mass_feasibility_report.md
```

结果：

```text
source_removal_semantics_match_vggt4d = true
landed_attention_mass_available = false
sample_files_checked = 6
sample_records_checked = 198
sample_records_with_any_attention_mass = 0
```

代码审计结论：

```text
VGGT4D keeps all query rows and runs attention against a source-only/non-dynamic
K/V subset.

LoGeR compact_kv keeps all query rows and runs attention against K/V selected by
source_keep_mask. This matches the VGGT4D source-removal pattern at the K/V
compaction level.
```

边界：

```text
Current landed artifacts contain source skip counts / keep ratios, not q/k,
attention probabilities, or attention_mass_removed_before/after fields.

Therefore true attention-mass before/after cannot be reconstructed honestly from
landed v36B artifacts. No attention-mass causality claim is made.
```

Required future repair before claiming attention mass：

```text
1. instrument attention layer to compute/sample qk softmax mass before/after
   source removal;
2. write attention_mass_removed_before/after and retained/group mass fields into
   hmc trace;
3. rerun H0C or representative H1 source-skip rows.
```

### 2.4 H0B attention-mass instrumentation repair

触发原因：

```text
2.3 confirmed the old landed v36B artifacts cannot support attention-mass
claims. Because the repair path was clear, implement a default-off runtime
instrumentation path and rerun one representative compact_kv source-skip row.
```

代码修改：

```text
loger/models/layers/attention.py:
    _compact_kv_sdpa now optionally records sampled qk softmax mass over removed
    and retained K/V source columns before compaction.

loger/models/pi3.py:
    passes an attention_mass_stats collector through compact_kv attn_mask when
    context_source_skip_record_attention_mass is enabled, then writes the
    resulting scalar fields into the HMC trace record.

loger/pipeline/hybrid_memory_controller.py:
    adds default-off HMC config fields and aggregates attention-mass stats into
    hook_effect_summary.

run_pipeline_abc_v2.py:
    adds CLI args:
        --context_source_skip_record_attention_mass
        --context_source_skip_attention_mass_max_queries
    writes aggregated mass stats into context_skip_summary.jsonl.

tools/run_attention_cue_experiment.sh:
tools/run_v24_candidate_rollout.sh:
    forward the default-off attention-mass instrumentation env vars.

tools/v36b_context_skip_summary.py:
    reads attention mass fields when present.
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  loger/models/layers/attention.py \
  loger/models/pi3.py \
  loger/pipeline/hybrid_memory_controller.py \
  run_pipeline_abc_v2.py \
  tools/v36b_context_skip_summary.py

bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v24_candidate_rollout.sh
```

结果：

```text
PASS
```

Representative rerun：

```bash
ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR
RESULT_ROOT=results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30
SNAP_ROOT=$RESULT_ROOT/phase0_parent_snapshots
CHUNK=10
HORIZON=3
PARENT=H9
CANDIDATE=FG_RISK_00
SNAP=$(printf '%03d' "$CHUNK")
HMC=$ROOT/$SNAP_ROOT/state_snapshots/${PARENT}_V36B_R1/chunk_${SNAP}_input.pt
MERGE=$ROOT/$SNAP_ROOT/merge_state_snapshots/${PARENT}_V36B_R1/chunk_${SNAP}_input.pt

env \
  V24_ROOT="$RESULT_ROOT/phase0b_attention_mass_rerun_R1" \
  RUN_PREFIX="V36B_H0B_MASS_R1_${PARENT}" \
  LOAD_HMC_STATE_AT_CHUNK="$HMC" \
  LOAD_MERGE_STATE_AT_CHUNK="$MERGE" \
  CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
  CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
  FORCE=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$ROOT/tools/run_v24_candidate_rollout.sh" 0 "$CANDIDATE" "$CHUNK" "$HORIZON"
```

Run status：

```text
V36B_H0B_MASS_R1_H9_FG_RISK_00_chunk10_h3_globalgate_H9parent_SWKS3
START 2026-05-24 13:37:01
DONE  2026-05-24 13:40:28
```

Report command：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v36b_context_skip_summary.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0b_attention_mass_rerun_R1/rollouts \
  --run-prefix V36B_H0B_MASS_R1_H9 \
  --chunks 10 \
  --horizon 3 \
  --candidates FG_RISK_00 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0b_attention_mass_rerun_R1/report_R1
```

Aggregate helper command：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY'
import json, pathlib, statistics
run=pathlib.Path('results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0b_attention_mass_rerun_R1/rollouts/V36B_H0B_MASS_R1_H9_FG_RISK_00_chunk10_h3_globalgate_H9parent_SWKS3')
rows=[json.loads(l) for l in (run/'context_skip_summary.jsonl').read_text().splitlines() if l.strip()]
mass=[r for r in rows if r.get('attention_mass_available')]
summary={
 'run_dir': str(run),
 'rows_total': len(rows),
 'mass_rows': len(mass),
 'paths_with_mass': sorted(set(r.get('path') for r in mass)),
 'chunks_with_mass': [r.get('chunk_idx') for r in mass],
 'mean_removed_before': statistics.mean([float(r['mean_attention_mass_removed_before']) for r in mass]) if mass else None,
 'min_removed_before': min([float(r['mean_attention_mass_removed_before']) for r in mass]) if mass else None,
 'max_removed_before': max([float(r['mean_attention_mass_removed_before']) for r in mass]) if mass else None,
 'mean_removed_after': statistics.mean([float(r['mean_attention_mass_removed_after']) for r in mass]) if mass else None,
 'mean_retained_before': statistics.mean([float(r['mean_attention_mass_retained_before']) for r in mass]) if mass else None,
 'mean_retained_after': statistics.mean([float(r['mean_attention_mass_retained_after']) for r in mass]) if mass else None,
 'context_empty_source_events_total': sum(int(r.get('num_context_empty_source_events') or 0) for r in rows),
 'max_context_source_skip_tokens': max(int(r.get('max_context_source_skip_tokens') or 0) for r in rows),
 'attention_mass_status': 'sampled-qk-softmax-mass' if mass else 'missing',
}
out=pathlib.Path('results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0b_attention_mass_rerun_R1/report_R1/h0b_runtime_attention_mass_summary.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
```

Runtime H0B result：

```text
all_rows_done = true
attention_mass_removed_available = true
attention_mass_status = sampled-qk-softmax-mass
mass_rows = 4
paths_with_mass = [chunk_attention]
chunks_with_mass = [10,11,12,13]
mean_removed_before = 0.0549059110
min_removed_before = 0.0469467230
max_removed_before = 0.0641816035
mean_removed_after = 0.0
mean_retained_before = 0.9450940639
mean_retained_after = 1.0
context_empty_source_events_total = 0
max_context_source_skip_tokens = 2670
```

边界：

```text
This is sampled qk-softmax mass for one representative compact_kv source-skip
row, not a full H1/H2 matrix rerun and not a Target-30 trajectory result.
Frame-attention mass was not available in this row because that path used an
existing dense mask; compact_kv mass landed for chunk_attention.
```

---

## 3. H1 Frame/Global h10

### 3.1 R1 invalid

首次 R1 并发触发 CUDA OOM：

```text
V36B_H1_H10_R1_H9_FG_SEM_04_chunk10_h10_globalgate_H9parent_SWKS3
torch CUDA OOM
```

原因：

```text
original scheduler reused a GPU before the previous heavy row finished.
```

修复：

```text
tools/run_v36b_h1_h10.sh:
    changed to one worker queue per GPU.
```

R1 决策：

```text
invalid / partial; not used for gates.
```

### 3.2 R2 clean rerun

执行：

```bash
V36B_H1_RUN_PREFIX_BASE=V36B_H1_H10_R2 \
V36B_GPUS=0,1,2,3 \
V36B_MAX_PARALLEL=4 \
tools/run_v36b_h1_h10.sh
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates = V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_04
rows = 18/18
failures = 0
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/rollouts \
  --run-prefix V36B_H1_H10_R2_H9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_04 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/report_R2_H9 \
  --report-prefix h1_h10_H9 \
  --ate-threshold -1.5 --segment-threshold -3.0 --downstream-threshold 0.25

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/rollouts \
  --run-prefix V36B_H1_H10_R2_C9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_04 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/report_R2_C9 \
  --report-prefix h1_h10_C9 \
  --ate-threshold -1.5 --segment-threshold -3.0 --downstream-threshold 0.25
```

Context summary：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v36b_context_skip_summary.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/rollouts \
  --run-prefix V36B_H1_H10_R2_H9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_04 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/report_R2_H9_context

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v36b_context_skip_summary.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/rollouts \
  --run-prefix V36B_H1_H10_R2_C9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_04 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase1_h1_frame_global_h10/report_R2_C9_context
```

R2 result：

```text
H9 gate_pass = false
H9 best ATE delta = -0.0473357966m
H9 best [200,300) delta = -0.3418221710m

C9 gate_pass = false
C9 best ATE delta = -0.0694789609m
C9 best [200,300) delta = -0.2606290156m

context_empty_source_events_total = 0
attention_mass_status = attention-mass-unverified
```

Decision：

```text
No H1 h15 launched.
```

---

## 4. H2 SWA Local-Continuity

### 4.1 h10

执行：

```bash
V36B_PHASE_NAME=phase2_h2_swa_h10 \
V36B_RUN_PREFIX_BASE=V36B_H2_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_04_BOUNDARY_PROTECT,SWA_FINE_05_CACHE_LIFECYCLE \
V36B_GPUS=0,1,2,3 \
tools/run_v36b_path_h10.sh
```

报告：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/rollouts \
  --run-prefix V36B_H2_H10_R1_H9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_04_BOUNDARY_PROTECT,SWA_FINE_05_CACHE_LIFECYCLE \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/report_R1_H9 \
  --report-prefix h2_h10_H9 \
  --ate-threshold -1.5 --segment-threshold -3.0 --downstream-threshold 0.25

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/rollouts \
  --run-prefix V36B_H2_H10_R1_C9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_04_BOUNDARY_PROTECT,SWA_FINE_05_CACHE_LIFECYCLE \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/report_R1_C9 \
  --report-prefix h2_h10_C9 \
  --ate-threshold -1.5 --segment-threshold -3.0 --downstream-threshold 0.25
```

Boundary diagnostics：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v27_swa_boundary_diagnostics.py \
  --reference-run results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/rollouts/V36B_H2_H10_R1_H9_V31_BASE_H9_REFERENCE_chunk10_h10_globalgate_H9parent_SWKS3 \
  --candidate-run results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/rollouts/V36B_H2_H10_R1_H9_SWA_FINE_01_OVERLAP_STRUCTURE_KEEP_chunk10_h10_globalgate_H9parent_SWKS3 \
  --candidate-run results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/rollouts/V36B_H2_H10_R1_H9_SWA_FINE_04_BOUNDARY_PROTECT_chunk10_h10_globalgate_H9parent_SWKS3 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/boundary_R1_H9_chunk10

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v27_swa_boundary_diagnostics.py \
  --reference-run results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/rollouts/V36B_H2_H10_R1_C9_V31_BASE_H9_REFERENCE_chunk10_h10_globalgate_H9parent_SWKS3 \
  --candidate-run results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/rollouts/V36B_H2_H10_R1_C9_SWA_FINE_01_OVERLAP_STRUCTURE_KEEP_chunk10_h10_globalgate_H9parent_SWKS3 \
  --candidate-run results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/rollouts/V36B_H2_H10_R1_C9_SWA_FINE_04_BOUNDARY_PROTECT_chunk10_h10_globalgate_H9parent_SWKS3 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h10/boundary_R1_C9_chunk10
```

h10 result：

```text
H9 gate_pass = true
H9 best [200,300) delta = -3.7351609926m
H9 chunk10 candidates = SWA_FINE_01_OVERLAP_STRUCTURE_KEEP, SWA_FINE_04_BOUNDARY_PROTECT
H9 boundary 10f delta = +0.0723922636m
H9 boundary 20f delta = -0.0176117235m

C9 gate_pass = true
C9 best [200,300) delta = -3.4943516930m
C9 chunk10 candidates = SWA_FINE_01_OVERLAP_STRUCTURE_KEEP, SWA_FINE_04_BOUNDARY_PROTECT
C9 boundary 10f delta = +0.2221317699m
C9 boundary 20f delta = +0.1452799891m
```

### 4.2 h15

执行：

```bash
V36B_PHASE_NAME=phase2_h2_swa_h15 \
V36B_RUN_PREFIX_BASE=V36B_H2_H15_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_04_BOUNDARY_PROTECT \
V36B_CHUNKS=10 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=15 \
V36B_GPUS=0,1 \
tools/run_v36b_path_h10.sh
```

报告：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h15/rollouts \
  --run-prefix V36B_H2_H15_R1_H9 \
  --chunks 10 --horizon 15 \
  --candidates V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_04_BOUNDARY_PROTECT \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h15/report_R1_H9 \
  --report-prefix h2_h15_H9 \
  --ate-threshold -3.0 --segment-threshold -5.0 --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h15/rollouts \
  --run-prefix V36B_H2_H15_R1_C9 \
  --chunks 10 --horizon 15 \
  --candidates V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_04_BOUNDARY_PROTECT \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase2_h2_swa_h15/report_R1_C9 \
  --report-prefix h2_h15_C9 \
  --ate-threshold -3.0 --segment-threshold -5.0 --downstream-threshold 1.0
```

h15 result：

```text
H9 gate_pass = false; best [200,300) delta = -1.1991180979m
C9 gate_pass = false; best [200,300) delta = -0.8636036060m
```

Decision：

```text
H2 h15 fails. No H2 full online.
```

### 4.3 H2 washout attribution

触发原因：

```text
v36B plan 10.3 requires washout attribution when h10 is effective but h15 fails.
H2 SWA h10 passed on chunk10 for H9/C9, but h15 failed.
```

执行：

```bash
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
ROOT=results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30

for PARENT in H9 C9; do
  for CAND in SWA_FINE_01_OVERLAP_STRUCTURE_KEEP SWA_FINE_04_BOUNDARY_PROTECT; do
    $PY tools/v31_track_f_washout_attribution.py \
      --h10-run-dir "$ROOT/phase2_h2_swa_h10/rollouts/V36B_H2_H10_R1_${PARENT}_${CAND}_chunk10_h10_globalgate_H9parent_SWKS3" \
      --h15-run-dir "$ROOT/phase2_h2_swa_h15/rollouts/V36B_H2_H15_R1_${PARENT}_${CAND}_chunk10_h15_globalgate_H9parent_SWKS3" \
      --h10-report-csv "$ROOT/phase2_h2_swa_h10/report_R1_${PARENT}/h2_h10_${PARENT}_effects.csv" \
      --h15-report-csv "$ROOT/phase2_h2_swa_h15/report_R1_${PARENT}/h2_h15_${PARENT}_effects.csv" \
      --candidate "$CAND" \
      --chunk 10 \
      --h10-horizon 10 \
      --h15-horizon 15 \
      --out-dir "$ROOT/phase2_h2_swa_washout_R1/${PARENT}_${CAND}"
  done
done
```

输出：

```text
phase2_h2_swa_washout_R1/
    H9_SWA_FINE_01_OVERLAP_STRUCTURE_KEEP/
    H9_SWA_FINE_04_BOUNDARY_PROTECT/
    C9_SWA_FINE_01_OVERLAP_STRUCTURE_KEEP/
    C9_SWA_FINE_04_BOUNDARY_PROTECT/
```

结果摘要：

| Parent | Candidate | h10 `[200,300)` delta | h15 `[200,300)` delta | h15/h10 abs durability | TTT tail/h10 proxy | Frame-bias tail/h10 | SWA replace tail/h10 |
|---|---|---:|---:|---:|---:|---:|---:|
| `H9` | `SWA_FINE_01` | `-3.7351609926` | `-1.1991180979` | `0.3210351835` | `0.4499735800` | `0.0` | `0.0` |
| `H9` | `SWA_FINE_04` | `-3.7351609926` | `-1.1991180979` | `0.3210351835` | `0.4499735800` | `0.0` | `0.0` |
| `C9` | `SWA_FINE_01` | `-3.4943516930` | `-0.8636036060` | `0.2471427261` | `0.4492046627` | `0.0` | `0.0` |
| `C9` | `SWA_FINE_04` | `-3.4943516930` | `-0.8636036060` | `0.2471427261` | `0.4492046627` | `0.0` | `0.0` |

Boundary：

```text
Evidence level = proxy_only_no_tensor_state_snapshots.
No .pt tensor state snapshots were used for full tensor overwrite attribution.
The attribution is from landed hmc_state_hash.jsonl / hook summary proxies only.
```

Decision：

```text
H2 h10 local SWA effect washes out by h15.
The available proxy does not show continued SWA source-replace or frame-bias tail
movement; h15 tail still has TTT side-effect proxy mass around 0.45 of h10.
Do not continue H2 via full online; treat SWA as local/diagnostic only.
```

---

## 5. H3 TTT h10

执行：

```bash
V36B_PHASE_NAME=phase3_h3_ttt_h10 \
V36B_RUN_PREFIX_BASE=V36B_H3_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_RISK_01_CONFLICT_TRI \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=2,3 \
tools/run_v36b_path_h10.sh
```

报告：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase3_h3_ttt_h10/rollouts \
  --run-prefix V36B_H3_H10_R1_H9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_RISK_01_CONFLICT_TRI \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase3_h3_ttt_h10/report_R1_H9 \
  --report-prefix h3_h10_H9 \
  --ate-threshold -1.5 --segment-threshold -3.0 --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase3_h3_ttt_h10/rollouts \
  --run-prefix V36B_H3_H10_R1_C9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_RISK_01_CONFLICT_TRI \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase3_h3_ttt_h10/report_R1_C9 \
  --report-prefix h3_h10_C9 \
  --ate-threshold -1.5 --segment-threshold -3.0 --downstream-threshold 1.0
```

Result：

```text
H9 rows = 12/12, gate_pass = false
H9 best ATE delta = -0.0955755272m
H9 best [200,300) delta = -0.1236567417m

C9 rows = 12/12, gate_pass = false
C9 best ATE delta = -0.0962599959m
C9 best [200,300) delta = -0.1663533136m
```

Decision：

```text
No H3 h15.
```

---

## 6. H4 Semantic C23 Path Isolation

### 6.1 Initial h10

执行：

```bash
V36B_PHASE_NAME=phase4_h4_semc23_h10 \
V36B_RUN_PREFIX_BASE=V36B_H4_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,V31_A1B_SEM_Z_COARSE,V31_A5B_SEM_RESID_COARSE_L025,V31_B0_STATIC_RESCUE_EXISTING \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1 \
tools/run_v36b_path_h10.sh
```

报告：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_h10/rollouts \
  --run-prefix V36B_H4_H10_R1_H9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,V31_A1B_SEM_Z_COARSE,V31_A5B_SEM_RESID_COARSE_L025,V31_B0_STATIC_RESCUE_EXISTING \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_h10/report_R1_H9 \
  --report-prefix h4_h10_H9 \
  --ate-threshold -999 --segment-threshold -5.0 --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_h10/rollouts \
  --run-prefix V36B_H4_H10_R1_C9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,V31_A1B_SEM_Z_COARSE,V31_A5B_SEM_RESID_COARSE_L025,V31_B0_STATIC_RESCUE_EXISTING \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_h10/report_R1_C9 \
  --report-prefix h4_h10_C9 \
  --ate-threshold -999 --segment-threshold -5.0 --downstream-threshold 1.0
```

Result：

```text
H9 gate_pass = false; best [200,300) delta = -4.9916037591m
C9 gate_pass = false; best [200,300) delta = -4.9458168459m
```

### 6.2 Minimal beta525 repair

Reason：

```text
Initial H4 h10 missed the -5m gate by a very small margin.
Use prior landed v31 beta525 repair as a single conservative repair.
No threshold sweep was performed.
```

h10 execution on GPUs 4,5,6,7:

```bash
BETA_VALUE=5.25 \
V36B_PHASE_NAME=phase4_h4_semc23_repair_beta525_h10 \
V36B_RUN_PREFIX_BASE=V36B_H4_REPAIR_BETA525_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,V31_A1B_SEM_Z_COARSE \
V36B_CHUNKS=10 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=4,5,6,7 \
tools/run_v36b_path_h10.sh
```

h10 report:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_repair_beta525_h10/rollouts \
  --run-prefix V36B_H4_REPAIR_BETA525_H10_R1_H9 \
  --chunks 10 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,V31_A1B_SEM_Z_COARSE \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_repair_beta525_h10/report_R1_H9 \
  --report-prefix h4_repair_beta525_h10_H9 \
  --ate-threshold -999 --segment-threshold -5.0 --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_repair_beta525_h10/rollouts \
  --run-prefix V36B_H4_REPAIR_BETA525_H10_R1_C9 \
  --chunks 10 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,V31_A1B_SEM_Z_COARSE \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_repair_beta525_h10/report_R1_C9 \
  --report-prefix h4_repair_beta525_h10_C9 \
  --ate-threshold -999 --segment-threshold -5.0 --downstream-threshold 1.0
```

h10 result:

```text
H9 gate_pass = true; [200,300) delta = -6.0997514572m; [400,600) delta = +0.6826731549m
C9 gate_pass = true; [200,300) delta = -6.0435649622m; [400,600) delta = +0.6822210800m
```

h15 execution:

```bash
BETA_VALUE=5.25 \
V36B_PHASE_NAME=phase4_h4_semc23_repair_beta525_h15 \
V36B_RUN_PREFIX_BASE=V36B_H4_REPAIR_BETA525_H15_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,V31_A1B_SEM_Z_COARSE \
V36B_CHUNKS=10 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=15 \
V36B_GPUS=4,5,6,7 \
tools/run_v36b_path_h10.sh
```

h15 report:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_repair_beta525_h15/rollouts \
  --run-prefix V36B_H4_REPAIR_BETA525_H15_R1_H9 \
  --chunks 10 --horizon 15 \
  --candidates V31_BASE_H9_REFERENCE,V31_A1B_SEM_Z_COARSE \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_repair_beta525_h15/report_R1_H9 \
  --report-prefix h4_repair_beta525_h15_H9 \
  --ate-threshold -999 --segment-threshold -5.0 --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_repair_beta525_h15/rollouts \
  --run-prefix V36B_H4_REPAIR_BETA525_H15_R1_C9 \
  --chunks 10 --horizon 15 \
  --candidates V31_BASE_H9_REFERENCE,V31_A1B_SEM_Z_COARSE \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase4_h4_semc23_repair_beta525_h15/report_R1_C9 \
  --report-prefix h4_repair_beta525_h15_C9 \
  --ate-threshold -999 --segment-threshold -5.0 --downstream-threshold 1.0
```

h15 result:

```text
H9 gate_pass = true; [200,300) delta = -6.1462634267m; [400,600) delta = +0.2681140634m
C9 gate_pass = true; [200,300) delta = -6.1250511839m; [400,600) delta = +0.2516740832m
```

Decision：

```text
H4 beta525 repair passes h10 and h15 short gate.
H5 full online evidence is allowed.
```

---

## 7. H5 Full Online Evidence

Full-online exact landed row reused from v32:

```text
results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/h2_c9_combo/rollouts/V32_H2_01_C9_SEM_Z_COARSE_BETA525_ALL
results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/h3_repair/rollouts/V32_H3_01_C9_RESID_COARSE_L025_ALL
```

Reason for reuse:

```text
The SEM_Z landed row has:
    mode = hybrid
    read_cue_source = v31.sem_z_coarse.c23past
    beta_frame = 5.25
    beta_swa = 5.25
    hmc_commit_mode = probe_ttt_write
    hmc_write_score_source = stage_d_x_dg_inv_sqrt
    full sequence frames = 1101

The RESID landed row is the v32 conservative residual repair:
    read_cue_source = v31.sem_resid_coarse_l025.c23past
    mode = hybrid
    full sequence frames = 1101

These rows are reused as landed full-online evidence, not claimed as newly
launched v36B rows.
```

Initial SEM_Z-only report command:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v32_transfer_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30 \
  --runs C9_REF=/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v15_ttt_repro_causal_sandbox_target25/phase0_repro/V15_P0_A2_C9_REPEAT_no_state_save_SWKS3,V36B_H4_SEM_Z_COARSE_BETA525_ALL_REUSE_V32=/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/h2_c9_combo/rollouts/V32_H2_01_C9_SEM_Z_COARSE_BETA525_ALL \
  --reference-name C9_REF \
  --c9-reference-ate 33.7629421029 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase5_h5_full_online_reuse_v32_report \
  --target-ate 30.0
```

Follow-up report adding the landed residual repair:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v32_transfer_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30 \
  --runs C9_REF=/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v15_ttt_repro_causal_sandbox_target25/phase0_repro/V15_P0_A2_C9_REPEAT_no_state_save_SWKS3,V36B_H4_SEM_Z_COARSE_BETA525_ALL_REUSE_V32=/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/h2_c9_combo/rollouts/V32_H2_01_C9_SEM_Z_COARSE_BETA525_ALL,V36B_H4_RESID_COARSE_L025_ALL_REUSE_V32=/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/h3_repair/rollouts/V32_H3_01_C9_RESID_COARSE_L025_ALL \
  --reference-name C9_REF \
  --c9-reference-ate 33.7629421029 \
  --out-dir results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase5_h5_full_online_reuse_v32_report_R2_with_residual \
  --target-ate 30.0
```

Result:

```text
C9_REF ATE = 33.7629421029m
V36B_H4_SEM_Z_COARSE_BETA525_ALL_REUSE_V32 ATE = 34.5597307381m
V36B_H4_RESID_COARSE_L025_ALL_REUSE_V32 ATE = 34.3258261120m

SEM_Z delta vs C9 = +0.7967886352m
RESID delta vs C9 = +0.5628840091m
RESID [200,300) delta vs C9 = -0.6684692189m
RESID [400,600) delta vs C9 = +3.1115081606m

Target-30 pass = false
counts_as_deployable_online = false
```

---

### 7.1 H5 full path-isolation rerun per v36B 10.6

Reason：

```text
The reused v32 full rows prove that all-chunks semantic-z/residual hybrid rows
regress vs C9, but v36B plan 10.6 requires path isolation after C9 full-online
regression:
    disable conflicting paths,
    if semantic read-only still regresses, stop semantic deployment line.

Therefore launch new v36B full rows:
    C9_NO_SWA_BASE
    SEM_Z_NO_SWA
    RESID_NO_SWA
    SEM_Z_READONLY
    RESID_READONLY
```

GPU preflight：

```text
2026-05-24 13:49:11
GPU 0..7 memory used MiB / total MiB / util:
0, 1, 23028, 0
1, 1, 23028, 0
2, 1, 23028, 0
3, 4, 23028, 0
4, 4, 23028, 0
5, 4, 23028, 0
6, 1, 23028, 0
7, 4, 23028, 0
```

Launch command：

```bash
set -euo pipefail
ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR
cd "$ROOT"
RESULT_ROOT=results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30
BASE="$RESULT_ROOT/phase5_h5_path_isolation_full_R1/rollouts"
LOGDIR="$RESULT_ROOT/matrix_logs/phase5_h5_path_isolation_full_R1"
mkdir -p "$BASE" "$LOGDIR"

launch_base_no_swa() {
  local GPU="$1" NAME="$2"
  (
    cd "$ROOT"
    env \
      ATTN_CUE_BASE="$BASE" \
      KITTI_SEQ=01 \
      START_FRAME=0 \
      END_FRAME=10000 \
      RESET_EVERY=5 \
      READ_PATH=frame \
      FAST_CUE_EVAL=1 \
      ENABLE_SWA_WRITE_CONTROL=0 \
      ENABLE_SWA_OVERLAP_SOURCE_REPLACE=0 \
      ENABLE_SWA_OVERLAP_SOURCE_GATE=0 \
      ENABLE_SWA_OVERLAP_BIAS=0 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      tools/run_attention_cue_experiment.sh "$GPU" "$NAME" hybrid \
        acl2.gg.qq.low.g2_3.past_only.headmean.robustq \
        4.75 stage_d_x_dg_inv_sqrt
  ) 2>&1 | tee "$LOGDIR/${NAME}.log"
}

launch_semantic() {
  local GPU="$1" NAME="$2" MODE="$3" CUE="$4"
  (
    cd "$ROOT"
    env \
      ATTN_CUE_BASE="$BASE" \
      KITTI_SEQ=01 \
      START_FRAME=0 \
      END_FRAME=10000 \
      RESET_EVERY=5 \
      READ_PATH=frame \
      FAST_CUE_EVAL=1 \
      STAGE_C_MODE=reference \
      STAGE_C_CACHE_MODE=read \
      STAGE_C_CACHE_REQUIRE_HIT=1 \
      STAGE_C_CACHE_DIR=results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full \
      SEMANTIC_PRIOR_MODE=spg_v2 \
      SEMANTIC_ROLE_POLICY=fine_path_router_debug \
      SEMANTIC_MEMORY_PATHS=all \
      ENABLE_SWA_WRITE_CONTROL=0 \
      ENABLE_SWA_OVERLAP_SOURCE_REPLACE=0 \
      ENABLE_SWA_OVERLAP_SOURCE_GATE=0 \
      ENABLE_SWA_OVERLAP_BIAS=0 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      tools/run_attention_cue_experiment.sh "$GPU" "$NAME" "$MODE" "$CUE" \
        5.25 stage_d_x_dg_inv_sqrt
  ) 2>&1 | tee "$LOGDIR/${NAME}.log"
}

launch_base_no_swa 0 V36B_H5_R1_C9_NO_SWA_BASE &
launch_semantic 1 V36B_H5_R1_SEM_Z_NO_SWA hybrid v31.sem_z_coarse.c23past &
launch_semantic 2 V36B_H5_R1_RESID_NO_SWA hybrid v31.sem_resid_coarse_l025.c23past &
launch_semantic 3 V36B_H5_R1_SEM_Z_READONLY readonly v31.sem_z_coarse.c23past &
launch_semantic 4 V36B_H5_R1_RESID_READONLY readonly v31.sem_resid_coarse_l025.c23past &
wait
```

Rows：

```text
V36B_H5_R1_C9_NO_SWA_BASE:
    START 2026-05-24 13:49:11
    DONE  2026-05-24 14:11:42

V36B_H5_R1_RESID_NO_SWA:
    START 2026-05-24 13:49:11
    DONE  2026-05-24 14:14:40

V36B_H5_R1_RESID_READONLY:
    START 2026-05-24 13:49:11
    DONE  2026-05-24 14:14:34

V36B_H5_R1_SEM_Z_NO_SWA:
    START 2026-05-24 13:49:11
    DONE  2026-05-24 14:17:10

V36B_H5_R1_SEM_Z_READONLY:
    START 2026-05-24 13:49:11
    DONE  2026-05-24 14:15:10
```

Report command initial mistake：

```text
First report command missed --rollout-root. argparse rejected it and no report
was written. This was a reporting command issue only; rollout dirs were already
DONE. Corrected command below.
```

Corrected report command：

```bash
ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR
cd "$ROOT"
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
C9=$ROOT/results/kitti01_hmc_v2/acl2_v15_ttt_repro_causal_sandbox_target25/phase0_repro/V15_P0_A2_C9_REPEAT_no_state_save_SWKS3
BASE=$ROOT/results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase5_h5_path_isolation_full_R1/rollouts
OLDZ=$ROOT/results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/h2_c9_combo/rollouts/V32_H2_01_C9_SEM_Z_COARSE_BETA525_ALL
OLDR=$ROOT/results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/h3_repair/rollouts/V32_H3_01_C9_RESID_COARSE_L025_ALL
OUT=results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase5_h5_path_isolation_full_R1/report_R1
$PY tools/v32_transfer_report.py \
  --rollout-root . \
  --runs C9_REF=$C9,V36B_REUSE_SEM_Z_ALL=$OLDZ,V36B_REUSE_RESID_ALL=$OLDR,V36B_C9_NO_SWA_BASE=$BASE/V36B_H5_R1_C9_NO_SWA_BASE,V36B_SEM_Z_NO_SWA=$BASE/V36B_H5_R1_SEM_Z_NO_SWA,V36B_RESID_NO_SWA=$BASE/V36B_H5_R1_RESID_NO_SWA,V36B_SEM_Z_READONLY=$BASE/V36B_H5_R1_SEM_Z_READONLY,V36B_RESID_READONLY=$BASE/V36B_H5_R1_RESID_READONLY \
  --reference-name C9_REF \
  --out-dir "$OUT" \
  --target-ate 30.0
```

Report files：

```text
phase5_h5_path_isolation_full_R1/report_R1/v32_full_metrics.csv
phase5_h5_path_isolation_full_R1/report_R1/v32_full_metrics.json
phase5_h5_path_isolation_full_R1/report_R1/v32_full_summary.json
```

Display-command note：

```text
After the corrected report command succeeded, a display helper tried to cat
transfer_summary.json / transfer_metrics.csv. This script actually writes
v32_full_summary.json / v32_full_metrics.csv, so the display helper exited
non-zero after the report files had already landed. Metrics below are read from
the actual v32_full_* files.
```

Summary：

```text
rows = 8
done_rows = 8
target_ate = 30.0
reference_name = C9_REF
best_candidate = V36B_REUSE_RESID_ALL
best_candidate_ATE_full = 34.3258261120m
best_vs_c9_delta = +0.5628840091m
target30_pass = false
counts_as_deployable_online = false
```

Path-isolation full metrics：

| Run | Full ATE | `[200,300)` ATE | `[400,600)` ATE | Delta vs C9 |
|---|---:|---:|---:|---:|
| `C9_REF` | `33.7629421029` | `76.1021355543` | `41.8963642126` | `0.0` |
| `V36B_C9_NO_SWA_BASE` | `38.4008982950` | `74.4364805502` | `51.0119935782` | `+4.6379561921` |
| `V36B_SEM_Z_NO_SWA` | `38.6185780372` | `76.7424498908` | `51.5305074135` | `+4.8556359343` |
| `V36B_RESID_NO_SWA` | `38.2740049300` | `74.8701546766` | `50.8510186878` | `+4.5110628270` |
| `V36B_SEM_Z_READONLY` | `38.6598083906` | `76.8223138233` | `51.3057963914` | `+4.8968662877` |
| `V36B_RESID_READONLY` | `38.5202961329` | `75.0664994878` | `51.0993316597` | `+4.7573540299` |

Decision：

```text
H5 path-isolation full online = fail.
NoSWA does not rescue C9 compatibility.
Read-only/noTTT does not rescue C9 compatibility.
Per v36B 10.6, semantic deployment line should stop.
```

---

## 8. GPU 4-7 使用说明

在 H3/H4 已有 queue 运行时检查到 GPU 4/5/6/7 空闲，但没有直接抢跑同 prefix：

```text
tools/run_v24_candidate_rollout.sh will move an existing non-DONE run directory
to .INVALID_RERUN_* if the same RUN_NAME is launched again.
```

因此：

```text
Do not launch duplicate H3/H4 rows with the same prefix while old queues are active.
Use GPU 4/5/6/7 only for new non-conflicting beta525 repair phases.
```

实际使用：

```text
phase4_h4_semc23_repair_beta525_h10 used GPU 4,5,6,7.
phase4_h4_semc23_repair_beta525_h15 used GPU 4,5,6,7.
```

# ACL2 v43 执行日志：C9 Dechunk Attribution SemanticRead Target30

日期：2026-05-26（Asia/Singapore）

计划文件：

```text
docs/ACL2_v43_C9_Dechunk_Attribution_SemanticRead_Target30_Plan.md
```

主结果目录：

```text
results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/
```

原则：

```text
只记录实际执行指令和实际落盘结果。
不把 short/proxy/component audit 写成 deployable success。
每个 LoGeR 进程只绑定一张 GPU。
用户确认 GPU 0,1,2,3,4,5,6,7 可用。
```

---

## 2026-05-26 10:23 初始准备

阅读计划：

```bash
sed -n '1,260p' docs/ACL2_v43_C9_Dechunk_Attribution_SemanticRead_Target30_Plan.md
sed -n '261,620p' docs/ACL2_v43_C9_Dechunk_Attribution_SemanticRead_Target30_Plan.md
sed -n '621,980p' docs/ACL2_v43_C9_Dechunk_Attribution_SemanticRead_Target30_Plan.md
```

检查既有 v42 launcher / report / 语义 READ alias：

```bash
sed -n '1,220p' tools/run_v42_full_candidate.sh
sed -n '1,260p' tools/v42_full_online_report.py
sed -n '220,520p' tools/v42_full_online_report.py
sed -n '560,720p' tools/run_v24_candidate_rollout.sh
rg -n "SEM_RESID|sem_resid|SEM_Z|C23R|READ_A|F1|F2|F3|READ_BETA_FRAME_CHUNKS|TTT_WRITE_GRADIENT_REVERSAL_GAMMA|COMMIT_EMA|NATIVE_MIX|SWA_OVERLAP_SOURCE" \
    tools/run_v42_full_candidate.sh \
    tools/run_attention_cue_experiment.sh \
    tools/run_v24_candidate_rollout.sh \
    run_pipeline_abc_v2.py \
    loger/pipeline/hybrid_memory_controller.py
```

建立结果目录与 parent snapshot symlink：

```bash
mkdir -p results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30
ln -s ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots \
    results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase0_parent_snapshots
```

新增工具：

```text
tools/run_v43_full_candidate.sh
tools/v43_noop_gate_report.py
tools/v43_registry_summarize.py
```

验证：

```bash
bash -n tools/run_v43_full_candidate.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v43_noop_gate_report.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v43_registry_summarize.py
```

结果：

```text
PASS
```

---

## 2026-05-26 10:24 Phase 0 C9 locked repeat

启动：

```bash
LOGER_ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR \
bash tools/run_v43_full_candidate.sh \
    0 \
    V43_P0_F0_C9_LOCKED_REPEAT \
    P0_F0_C9_LOCKED_REPEAT
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/
    phase0_c9_repeat/rollouts/V43_P0_F0_C9_LOCKED_REPEAT/
```

状态：

```text
DONE at 2026-05-26 10:58:08
```

Phase 0 gate 汇总：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v43_noop_gate_report.py \
    --run-dir results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase0_c9_repeat/rollouts/V43_P0_F0_C9_LOCKED_REPEAT \
    --out-dir results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase0_c9_repeat/report_R1
```

精确 full metrics 汇总：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v42_full_online_report.py \
    --rollout-root results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30 \
    --runs F0=phase0_c9_repeat/rollouts/V43_P0_F0_C9_LOCKED_REPEAT \
    --reference-name F0 \
    --out-dir results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase0_c9_repeat/report_R1/full_metrics
```

结果：

```text
ATE = 33.76294210291885m
historical C9_P0_R2 = 33.7629421029m
abs_delta = 0.00000000001885m
hmc_rows = 38
effective_config_unexpected_diff_count = 0
stage_c_disabled = true
semantic_action_disabled = true
phase0_noop_gate_pass = true
```

---

## 2026-05-26 10:59 Phase 1 C9-flat dechunk full runs

启动：

```bash
set -euo pipefail
ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR
CANDS=(FLAT_01 FLAT_02 FLAT_03 FLAT_04)
GPUS=(0 1 2 3)
for i in "${!CANDS[@]}"; do
  cand="${CANDS[$i]}"
  gpu="${GPUS[$i]}"
  LOGER_ROOT="$ROOT" bash "$ROOT/tools/run_v43_full_candidate.sh" \
      "$gpu" "V43_P1_${cand}" "$cand" &
done
wait
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/
    phase1_flat/rollouts/
```

状态：

```text
RUNNING on GPU 0,1,2,3
Completed:
    FLAT_01 ATE = 35.2952180149m
    FLAT_02 ATE = 35.500496m
    FLAT_03 ATE = 35.360850m
    FLAT_04 ATE = 36.522942m
```

Phase 1 report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v42_full_online_report.py \
    --rollout-root results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30 \
    --runs F0=phase0_c9_repeat/rollouts/V43_P0_F0_C9_LOCKED_REPEAT,FLAT_01=phase1_flat/rollouts/V43_P1_FLAT_01,FLAT_02=phase1_flat/rollouts/V43_P1_FLAT_02,FLAT_03=phase1_flat/rollouts/V43_P1_FLAT_03,FLAT_04=phase1_flat/rollouts/V43_P1_FLAT_04 \
    --reference-name F0 \
    --out-dir results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase1_flat/report_R1

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v43_registry_summarize.py \
    --registry results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase1_flat/report_R1/full_online_registry.csv \
    --out-dir results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase1_flat/report_R1/v43 \
    --phase-name phase1_flat \
    --reference-name F0
```

Decision：

```text
best_flat = FLAT_01
best_flat_ATE = 35.29521801485317m
best_flat_delta_vs_C9 = +1.5322759119343203m
C9-flat acceptable = false
ATTR_02 global gamma = 0.003
ATTR_07 will reuse exact landed FLAT_01 evidence because ATTR_07 is defined
as same as Phase1 best flat.
```

---

## 2026-05-26 11:06 Phase 2 independent attribution rows

Phase 1 仍在运行。为充分利用 GPU 4-7，先启动不依赖 Phase1 best gamma / best flat 的 attribution rows。`ATTR_02` 与 `ATTR_07` 等 Phase1 结果出来后再跑。

启动：

```bash
set -euo pipefail
ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR
CANDS=(
  ATTR_01_C9_MINUS_READ_MAP_TO_FLAT
  ATTR_03_C9_MINUS_COMMIT_EMA
  ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE
  ATTR_05_C9_MINUS_TTT_TRI_REPLAY
)
GPUS=(4 5 6 7)
for i in "${!CANDS[@]}"; do
  cand="${CANDS[$i]}"
  gpu="${GPUS[$i]}"
  LOGER_ROOT="$ROOT" bash "$ROOT/tools/run_v43_full_candidate.sh" \
      "$gpu" "V43_P2_${cand}" "$cand" &
done
wait
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/
    phase2_attribution/rollouts/
```

状态：

```text
RUNNING on GPU 4,5,6,7
```

补跑 ATTR_06（FLAT_04 完成释放 GPU3 后启动）：

```bash
LOGER_ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR \
bash tools/run_v43_full_candidate.sh \
    3 \
    V43_P2_ATTR_06_C9_MINUS_NATIVE_MIX \
    ATTR_06_C9_MINUS_NATIVE_MIX
```

ATTR_02 launched after Phase 1 selected `0.003` global gamma:

```bash
LOGER_ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR \
V43_GLOBAL_TRI_GAMMA=0.003 \
bash tools/run_v43_full_candidate.sh \
    0 \
    V43_P2_ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT \
    ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT
```

Raw completed：

```text
V43_P2_ATTR_01_C9_MINUS_READ_MAP_TO_FLAT:
    DONE
    results_sim3 ATE = 33.789524m

V43_P2_ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT:
    DONE
    results_sim3 ATE = 34.733965m

V43_P2_ATTR_03_C9_MINUS_COMMIT_EMA:
    DONE
    results_sim3 ATE = 34.251324m

V43_P2_ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE:
    DONE
    results_sim3 ATE = 33.819190m

V43_P2_ATTR_05_C9_MINUS_TTT_TRI_REPLAY:
    DONE
    results_sim3 ATE = 36.209889m

V43_P2_ATTR_06_C9_MINUS_NATIVE_MIX:
    DONE
    results_sim3 ATE = 33.854642m
```

---

## 2026-05-26 13:48 Phase 2 attribution report

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v42_full_online_report.py \
    --rollout-root results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30 \
    --runs F0=phase0_c9_repeat/rollouts/V43_P0_F0_C9_LOCKED_REPEAT,ATTR_01=phase2_attribution/rollouts/V43_P2_ATTR_01_C9_MINUS_READ_MAP_TO_FLAT,ATTR_02=phase2_attribution/rollouts/V43_P2_ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT,ATTR_03=phase2_attribution/rollouts/V43_P2_ATTR_03_C9_MINUS_COMMIT_EMA,ATTR_04=phase2_attribution/rollouts/V43_P2_ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE,ATTR_05=phase2_attribution/rollouts/V43_P2_ATTR_05_C9_MINUS_TTT_TRI_REPLAY,ATTR_06=phase2_attribution/rollouts/V43_P2_ATTR_06_C9_MINUS_NATIVE_MIX,ATTR_07=phase1_flat/rollouts/V43_P1_FLAT_01 \
    --reference-name F0 \
    --out-dir results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase2_attribution/report_R1

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v43_registry_summarize.py \
    --registry results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase2_attribution/report_R1/full_online_registry.csv \
    --out-dir results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase2_attribution/report_R1/v43 \
    --phase-name phase2_attribution \
    --reference-name F0 \
    --component-ledger
```

结果：

```text
ATTR_01 delta = +0.0265805624m
ATTR_02 delta = +0.9710254058m
ATTR_03 delta = +0.4883854639m
ATTR_04 delta = +0.0562502451m
ATTR_05 delta = +2.4469526758m
ATTR_06 delta = +0.0917027499m
ATTR_07 delta = +1.5322759119m

largest_positive_component = ATTR_05_C9_MINUS_TTT_TRI_REPLAY
main conclusion = C9 depends strongly on TTT tri-replay and tri gamma chunk map.
```

输出：

```text
phase2_attribution/report_R1/full_online_registry.csv
phase2_attribution/report_R1/v43/phase2_attribution_report.md
phase2_attribution/report_R1/v43/component_contribution_ate.csv
phase2_attribution/report_R1/v43/component_contribution_segments.csv
phase2_attribution/report_R1/v43/component_contribution_rolling.csv
phase2_attribution/report_R1/v43/component_interaction_notes.md
phase2_attribution/report_R1/v43/component_ate_waterfall.png
phase2_attribution/report_R1/v43/component_segment_delta_bar.png
```

---

## 2026-05-26 11:42 Phase 3 C9-base semantic READ rows

Phase 1 FLAT rows 尚未全部汇总时，为充分利用 GPU 4-7 先启动 C9-base semantic READ rows。后续 Phase 1 证明 C9-flat 不可接受，因此这些 rows 成为官方 Phase 3 C9-base semantic READ matrix。

启动：

```bash
set -euo pipefail
ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR
CANDS=(
  SEM_READ_01_HIGH_INFLUENCE_ANOMALY
  SEM_READ_02_GUARDED_SKY_VEG_DYNAMIC
  SEM_READ_03_C23_RESID_READ_ONLY
  SEM_READ_04_ANOMALY_STATIC_RESCUE
)
GPUS=(4 5 6 7)
for i in "${!CANDS[@]}"; do
  cand="${CANDS[$i]}"
  gpu="${GPUS[$i]}"
  LOGER_ROOT="$ROOT" bash "$ROOT/tools/run_v43_full_candidate.sh" \
      "$gpu" "V43_P3_C9BASE_${cand}" "$cand" &
done
wait
```

Completed：

```text
V43_P3_C9BASE_SEM_READ_01_HIGH_INFLUENCE_ANOMALY:
    DONE
    ATE = 35.958086408625995m
    delta_vs_C9 = +2.195144305707146m

V43_P3_C9BASE_SEM_READ_02_GUARDED_SKY_VEG_DYNAMIC:
    DONE
    ATE = 54.4017022529366m
    delta_vs_C9 = +20.638760150017752m

V43_P3_C9BASE_SEM_READ_03_C23_RESID_READ_ONLY:
    DONE
    ATE = 33.487566750822836m
    delta_vs_C9 = -0.27537535209601316m

V43_P3_C9BASE_SEM_READ_04_ANOMALY_STATIC_RESCUE:
    DONE
    ATE = 53.93723362589884m
    delta_vs_C9 = +20.174291522979992m
```

---

## 2026-05-26 13:52 Phase 3 semantic READ report

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v42_full_online_report.py \
    --rollout-root results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30 \
    --runs F0=phase0_c9_repeat/rollouts/V43_P0_F0_C9_LOCKED_REPEAT,SEM_READ_01=phase3_semantic_read/rollouts/V43_P3_C9BASE_SEM_READ_01_HIGH_INFLUENCE_ANOMALY,SEM_READ_02=phase3_semantic_read/rollouts/V43_P3_C9BASE_SEM_READ_02_GUARDED_SKY_VEG_DYNAMIC,SEM_READ_03=phase3_semantic_read/rollouts/V43_P3_C9BASE_SEM_READ_03_C23_RESID_READ_ONLY,SEM_READ_04=phase3_semantic_read/rollouts/V43_P3_C9BASE_SEM_READ_04_ANOMALY_STATIC_RESCUE \
    --reference-name F0 \
    --out-dir results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase3_semantic_read/report_R1

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v43_registry_summarize.py \
    --registry results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase3_semantic_read/report_R1/full_online_registry.csv \
    --out-dir results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/phase3_semantic_read/report_R1/v43 \
    --phase-name phase3_semantic_read \
    --reference-name F0
```

结果：

```text
best_candidate = SEM_READ_03_C23_RESID_READ_ONLY
best_ATE = 33.487566750822836m
best_delta_vs_C9 = -0.27537535209601316m
minimum_progress_pass = false
target30_success = false

SEM_READ_01:
    ATE = 35.958086408625995m
    [200,300) delta = -3.7952646498m
    [400,600) delta = +2.1065748253m

SEM_READ_02:
    ATE = 54.4017022529366m

SEM_READ_04:
    ATE = 53.93723362589884m
```

Decision：

```text
Phase 3 semantic READ = fail minimum progress.
Phase 4 combinations not allowed:
    best improvement = 0.2753753521m < 0.3m
    best ATE = 33.4875667508m > 33.3m
Phase 5 cross-sequence not allowed:
    best ATE > 33.0m and improvement < 0.5m.
```

输出：

```text
phase3_semantic_read/report_R1/full_online_registry.csv
phase3_semantic_read/report_R1/v43/phase3_semantic_read_report.md
phase3_semantic_read/report_R1/v43/phase3_semantic_read_summary.json
```

---

## 2026-05-26 14:03 Final report / process audit

确认没有 v43 进程残留：

```bash
ps -eo pid,cmd | rg 'run_v43_full_candidate|run_attention_cue_experiment|run_pipeline_abc_v2.py|V43_' | rg -v rg || true
```

结果：

```text
No running v43 process.
```

Final reports written：

```text
results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/
    final_reports/v43_final_summary.md
    final_reports/v43_final_summary.json
```

Final decision：

```text
phase0_noop_gate_pass = true
phase1_c9_flat_acceptable = false
phase2_contribution_ledger = complete
phase3_best_semantic_read = SEM_READ_03_C23_RESID_READ_ONLY
phase3_best_semantic_read_ATE = 33.487566750822836m
phase3_best_semantic_read_delta_vs_C9 = -0.27537535209601316m
phase3_minimum_progress_pass = false
phase4_launched = false
phase5_launched = false
target30_success = false
deployable_online_success = false
best_deployable_online = C9_P0_R2
best_deployable_online_ate = 33.7629421029m
```

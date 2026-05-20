# ACL2 v18 实验复盘：TTT Write True Action Space Target25

日期：2026-05-20（Asia/Singapore）  
计划文件：`docs/ACL2_v18_TTT_Write_TrueActionSpace_Target25_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v18_ttt_write_true_actionspace_target25/`

本轮原则：只记录实际落盘结果；不把 sandbox oracle、short rollout、GT audit、instrumentation smoke、proxy artifact、或 failed gate 写成 deployable TTT success。没有通过 sandbox gate 时，不启动 no-GT selector，也不启动 full online validation。

术语说明：

```text
TTT write:
    test-time training 写入。模型在跑序列时，用当前窗口产生的信号更新未来窗口会读到的 fast-weight memory。

HMC:
    Hybrid Memory Controller。这里负责读写 fast weights，并记录每个 chunk 的状态 hash。

causal fork:
    从 full run 的某个 chunk 输入状态分叉，只跑未来短窗口。v16 已证明 HMC state + merge/gauge state + global chunk id 可以做到轨迹级一致。

h5 / h8 / h10:
    从候选 chunk 开始向后看的 5 / 8 / 10 个 chunk 短滚动窗口。v18 只允许这些短滚动作为 oracle diagnostic，不算 full online success。

cue:
    写入/读取 fast weights 时用的内部信号。本轮固定 read cue 为
    acl2.gg.qq.low.g2_3.past_only.headmean.robustq
    写入分数为 stage_d_x_dg_inv_sqrt，并在不同 candidate 中改变写入 action。
```

---

## 0. 工程与配置复盘

新增 / 修改：

```text
run_pipeline_abc_v2.py:
    新增 v18 true-action artifact 导出
    post_zp_delta_before_after.pt
    per_layer_branch_post_zp_delta.pt
    per_token_to_post_zp_contribution_summary.pt
    basis_vector_bank.pt
    basis_projection_coefficients.csv
    W_long_short_tensor_summary.pt
    W_short_apply_history.jsonl
    overlap_geometry_replay_target.pt
    overlap_geometry_replay_debug.jsonl
    window_scale_proxy.jsonl
    candidate_commit_manifest.csv
    支持 V18_TRUE_ACTION_TRACE_LAYERS / V18_TRUE_ACTION_TRACE_BRANCHES 控制 trace footprint

tools/run_v18_candidate_rollout.sh:
    v18 trusted short-rollout candidate launcher
    支持 h3/h5/h8/h10/h15
    固定 H9 parent + v16 causal fork snapshots
    candidate family:
        K1_H9
        PZBASIS_01..08
        AUXGEO_TRUE_01..06
        DLTRUE_01..08

tools/run_v18_matrix.sh:
    GPU worker scheduler
    phase2_pzbasis_initial
    phase3_auxgeo_initial
    phase4_dltrue_initial

tools/v18_artifact_audit.py:
    audit required true-action artifacts
    gate: required artifact coverage >= 90% and basis coefficients finite

tools/v18_true_action_report.py:
    aggregate h5/h8/h10/h15 candidate metrics
    recompute deltas against K1_H9 on the same frame intersection
    gate: h10/h15 ATE delta <= -3m, or [200,300) delta <= -5m with downstream proxy <= +1m
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    run_pipeline_abc_v2.py \
    tools/v18_true_action_report.py \
    tools/v18_artifact_audit.py

bash -n tools/run_v18_candidate_rollout.sh
bash -n tools/run_v18_matrix.sh

PASS
```

工程 blocker 与修复：

```text
blocker 1:
    第一条 PZBASIS instrumentation smoke 保存所有 layer/branch tensor，
    trace footprint 达到 9.3GB。

fix:
    按 v18 文档 memory-footprint blocker 方向，只保存 selected layers/branches。
    默认 V18_TRUE_ACTION_TRACE_LAYERS=0,6,12,last
    默认 V18_TRUE_ACTION_TRACE_BRANCHES=w0

post-fix footprint:
    AUXGEO smoke trace = 711MB
    DLTRUE smoke trace = 967MB

blocker 2:
    py_compile 第一次用裸 python 失败，因为 shell PATH 中没有 python。

fix:
    使用实验环境 Python:
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
    py_compile PASS
```

重要边界：

```text
所有 Phase 2/3/4 row 都是 trusted short-rollout oracle diagnostic。
diagnostic_only_short_rollout = true。
counts_as_ttt_write_success = false。
v18 trace artifacts are logging/audit artifacts only; they are not fed back as runtime GT/action inputs.
没有启动 selector。
没有启动 full online validation。
```

---

## 1. Phase 0 Boundary

v18 复用 v16 已通过的 locked boundary 与 causal fork parity。输入文件：

```text
results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25/registry_v16_phase0_boundary_R3.csv
results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25/phase1_causalfork/phase1_causalfork_gate_summary_R3.json
```

有效 boundary：

| Run | ATE | Rot | RPE_t | FinalErr | `[200,300)` | `[400,600)` | hmc rows | Counts as TTT write |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `H9_P0_R2` | `34.1257769401` | `6.5414` | `92.4053` | `6.189399` | `74.409927` | `44.353638` | `38` | `true` |
| `C9_P0_R2` | `33.7629421029` | `6.5259` | `92.3871` | `5.666384` | `76.102136` | `41.896364` | `38` | `true` |
| `WINGAM_P0_R3` | `34.1902782732` | `6.5666` | `92.4202` | `6.195236` | `75.576021` | `42.280485` | `38` | `true` |

固定解释：

```text
C9 locked repeat 仍是当前最好可计数 full online TTT write:
    ATE = 33.7629421029m

H9 仍作为 v18 candidate parent:
    H9 full ATE 比 C9 差，但 H9 在 [200,300) 病灶段更好。
```

NOGTPOSE_27 边界：

```text
v18 计划把 NOGTPOSE_27 作为 target-25 diagnostic upper bound。
本轮没有重跑 NOGTPOSE_27，也没有把它计为 TTT write success。
```

---

## 2. Phase 1 True-Action Instrumentation Gate

输出：

```text
phase1_instrumentation_audit/v18_artifact_audit.csv
phase1_instrumentation_audit/v18_artifact_audit.json
phase1_instrumentation_audit/v18_artifact_audit.md
phase1_instrumentation_audit/v18_artifact_audit_summary.json
true_action_traces/
```

Smoke runs：

| Run | Required files | Coverage | post-zp chunks | basis rows | coeff no NaN | Gate |
|---|---:|---:|---:|---:|---|---|
| `V18_SMOKE_PZBASIS_02_HARM_SUPPRESS_W0_chunk5_h3_globalgate_H9parent_SWKS3` | `11/11` | `1.0` | `4` | `216` | `true` | `pass` |
| `V18_SMOKE_AUXGEO_TRUE_01_OVERLAP_POINTMAP_V_W0_chunk5_h3_globalgate_H9parent_SWKS3` | `11/11` | `1.0` | `4` | `16` | `true` | `pass` |
| `V18_SMOKE_DLTRUE_02_SHORT_BODY_HARM_LONG_CONTINUITY_chunk5_h3_globalgate_H9parent_SWKS3` | `11/11` | `1.0` | `4` | `16` | `true` | `pass` |

Phase 1 conclusion：

```text
Phase 1 instrumentation gate = pass
all_runs_gate_pass = true
required v18 true-action artifacts are present and auditable
```

---

## 3. Phase 2 PZBASIS True Post-Zeropower Candidate Bank

输出：

```text
phase2_pzbasis_R1/
rollouts/V18_P2PZB_R1_*
matrix_logs/phase2_pzbasis_initial/
```

矩阵：

```text
candidates:
    K1_H9
    PZBASIS_01_CONTINUITY_BOOST_W0
    PZBASIS_02_HARM_SUPPRESS_W0
    PZBASIS_03_HARM_SUPPRESS_W0W2
    PZBASIS_04_SCALE_BASIS_SUPPRESS_W0
    PZBASIS_05_OVERLAP_BASIS_BOOST_W0
    PZBASIS_06_CONTINUITY_PLUS_HARM_W0

chunks: 6, 10
horizons: 5, 8, 10
rows launched: 42
rows completed: 42
failures: 0
```

Gate summary：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-0.391135` |
| Best h10/h15 ATE candidate | `PZBASIS_02_HARM_SUPPRESS_W0`, chunk `10`, h`10` |
| Best `[200,300)` delta vs H9 | `-0.457382` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Best by chunk / horizon：

| Chunk | Horizon | Best ATE candidate | ATE delta vs H9 | Best `[200,300)` candidate | `[200,300)` delta |
|---:|---:|---|---:|---|---:|
| `6` | `5` | `PZBASIS_01_CONTINUITY_BOOST_W0` | `+0.035566` | `PZBASIS_01_CONTINUITY_BOOST_W0` | `+0.045839` |
| `6` | `8` | `PZBASIS_01_CONTINUITY_BOOST_W0` | `+0.159636` | `PZBASIS_01_CONTINUITY_BOOST_W0` | `+0.135033` |
| `6` | `10` | `PZBASIS_01_CONTINUITY_BOOST_W0` | `+0.171948` | `PZBASIS_04_SCALE_BASIS_SUPPRESS_W0` | `+0.015049` |
| `10` | `5` | `PZBASIS_02_HARM_SUPPRESS_W0` | `-0.310478` | `PZBASIS_02_HARM_SUPPRESS_W0` | `-1.099855` |
| `10` | `8` | `PZBASIS_05_OVERLAP_BASIS_BOOST_W0` | `+0.215322` | `PZBASIS_05_OVERLAP_BASIS_BOOST_W0` | `+0.126597` |
| `10` | `10` | `PZBASIS_02_HARM_SUPPRESS_W0` | `-0.391135` | `PZBASIS_02_HARM_SUPPRESS_W0` | `-0.457382` |

Decision：

```text
Phase 2 PZBASIS gate = fail
best h10 ATE improvement = 0.391m < 1.0m signal line
no h15 extension launched
no selector/full online validation allowed
```

---

## 4. Phase 3 AUXGEO True Overlap-Geometry Candidate Bank

输出：

```text
phase3_auxgeo_R1/
rollouts/V18_P3AUX_R1_*
matrix_logs/phase3_auxgeo_initial/
```

矩阵：

```text
candidates:
    K1_H9
    AUXGEO_TRUE_01_OVERLAP_POINTMAP_V_W0
    AUXGEO_TRUE_02_OVERLAP_POINTMAP_KV_W0
    AUXGEO_TRUE_03_OVERLAP_SCALE_PROXY_W0
    AUXGEO_TRUE_04_OVERLAP_SCALE_PROXY_W0W2
    AUXGEO_TRUE_05_STRUCTURE_ONLY_OVERLAP_W0
    AUXGEO_TRUE_06_LOWD_STRUCTURE_OVERLAP_W0

chunks: 6, 10
horizons: 5, 8, 10
rows launched: 42
rows completed: 42
failures: 0
```

Gate summary：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-0.650435` |
| Best h10/h15 ATE candidate | `AUXGEO_TRUE_04_OVERLAP_SCALE_PROXY_W0W2`, chunk `6`, h`10` |
| Best `[200,300)` delta vs H9 | `-0.892010` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Best by chunk / horizon：

| Chunk | Horizon | Best ATE candidate | ATE delta vs H9 | Best `[200,300)` candidate | `[200,300)` delta |
|---:|---:|---|---:|---|---:|
| `6` | `5` | `AUXGEO_TRUE_03_OVERLAP_SCALE_PROXY_W0` | `+0.190151` | `AUXGEO_TRUE_01_OVERLAP_POINTMAP_V_W0` | `+0.196022` |
| `6` | `8` | `AUXGEO_TRUE_04_OVERLAP_SCALE_PROXY_W0W2` | `-0.232622` | `AUXGEO_TRUE_05_STRUCTURE_ONLY_OVERLAP_W0` | `-0.460328` |
| `6` | `10` | `AUXGEO_TRUE_04_OVERLAP_SCALE_PROXY_W0W2` | `-0.650435` | `AUXGEO_TRUE_04_OVERLAP_SCALE_PROXY_W0W2` | `-0.892010` |
| `10` | `5` | `AUXGEO_TRUE_06_LOWD_STRUCTURE_OVERLAP_W0` | `-0.382363` | `AUXGEO_TRUE_06_LOWD_STRUCTURE_OVERLAP_W0` | `-0.980379` |
| `10` | `8` | `AUXGEO_TRUE_02_OVERLAP_POINTMAP_KV_W0` | `+0.171535` | `AUXGEO_TRUE_02_OVERLAP_POINTMAP_KV_W0` | `+0.097669` |
| `10` | `10` | `AUXGEO_TRUE_04_OVERLAP_SCALE_PROXY_W0W2` | `-0.217712` | `AUXGEO_TRUE_04_OVERLAP_SCALE_PROXY_W0W2` | `-0.121281` |

Decision：

```text
Phase 3 AUXGEO gate = fail
best h10 ATE improvement = 0.650m < 1.0m signal line
no h15 extension launched
no semantic expansion launched
no selector/full online validation allowed
```

---

## 5. Phase 4 DLTRUE True Dual-Lifetime / Dual-Bank Candidate Bank

输出：

```text
phase4_dltrue_R1/
rollouts/V18_P4DL_R1_*
matrix_logs/phase4_dltrue_initial/
```

矩阵：

```text
candidates:
    K1_H9
    DLTRUE_01_SHORT_HARM_ONLY_W0
    DLTRUE_02_SHORT_BODY_HARM_LONG_CONTINUITY
    DLTRUE_03_SHORT_SCALE_CORRECTION_LONG_STRUCTURE
    DLTRUE_04_SHORT_OVERLAP_CORRECTION_LONG_NATIVE
    DLTRUE_05_SHORT_DECAY_FAST_K2
    DLTRUE_06_SHORT_DECAY_SLOW_K5
    DLTRUE_07_RESET_BOUND_SHORT_CLEAR
    DLTRUE_08_EXIT_WEAK_SHORT_HANDOFF

chunk: 10
horizons: 5, 8, 10
rows launched: 27
rows completed: 27
failures: 0
```

Gate summary：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-0.279586` |
| Best h10/h15 ATE candidate | `DLTRUE_01_SHORT_HARM_ONLY_W0`, chunk `10`, h`10` |
| Best `[200,300)` delta vs H9 | `-0.630857` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Best by horizon：

| Chunk | Horizon | Best ATE candidate | ATE delta vs H9 | Best `[200,300)` candidate | `[200,300)` delta |
|---:|---:|---|---:|---|---:|
| `10` | `5` | `DLTRUE_04_SHORT_OVERLAP_CORRECTION_LONG_NATIVE` | `-0.265998` | `DLTRUE_04_SHORT_OVERLAP_CORRECTION_LONG_NATIVE` | `-0.780976` |
| `10` | `8` | `DLTRUE_08_EXIT_WEAK_SHORT_HANDOFF` | `+0.245302` | `DLTRUE_02_SHORT_BODY_HARM_LONG_CONTINUITY` | `+0.140058` |
| `10` | `10` | `DLTRUE_01_SHORT_HARM_ONLY_W0` | `-0.279586` | `DLTRUE_01_SHORT_HARM_ONLY_W0` | `-0.630857` |

Decision：

```text
Phase 4 DLTRUE gate = fail
best h10 ATE improvement = 0.280m < 1.0m signal line
no h15 extension launched
no selector/full online validation allowed
```

---

## 6. Downstream Phase Decision

| Phase | Status | Reason |
|---|---|---|
| Phase 0 Boundary | pass | reused v16 locked H9/C9/WINGAM boundary and causal fork parity |
| Phase 1 True-action instrumentation | pass | 3 smoke runs, required artifacts `11/11`, finite coefficients |
| Phase 2 PZBASIS | fail | best h10 ATE delta `-0.391m`, below signal/gate |
| Phase 3 AUXGEO | fail | best h10 ATE delta `-0.650m`, below signal/gate |
| Phase 4 DLTRUE | fail | best h10 ATE delta `-0.280m`, below signal/gate |
| Phase 5 No-GT selector | not started | no family passed sandbox oracle gate |
| Phase 6 Full online validation | not started | forbidden by failed selector/full-run entry gate |

Boundary：

```text
No v18 short-rollout result counts as deployable TTT write success.
No GT-selected candidate is counted.
No no-GT selector was evaluated.
No full online validation was launched.
No online target-25 result was produced in v18.
Current best deployable online TTT write remains C9_P0_R2:
    ATE = 33.7629421029m
```

---

## 7. Final Decision

v18 的真实成功点：

```text
True-action instrumentation is now implemented and audited.
PZBASIS / AUXGEO / DLTRUE all produced required trace artifacts in smoke.
The v18 short-rollout matrix completed without failed runs:
    Phase 2: 42/42 complete
    Phase 3: 42/42 complete
    Phase 4: 27/27 complete
```

v18 的关键负结果：

```text
No true-action family reached the h10/h15 stage gate.
No family even crossed the 1m h10 ATE signal line:
    PZBASIS best h10 ATE delta = -0.391m
    AUXGEO best h10 ATE delta = -0.650m
    DLTRUE best h10 ATE delta = -0.280m

No family crossed the [200,300) 2m weak signal line:
    PZBASIS best h10 [200,300) delta = -0.457m
    AUXGEO best h10 [200,300) delta = -0.892m
    DLTRUE best h10 [200,300) delta = -0.631m
```

Interpretation：

```text
v18 made the real post-zeropower / overlap / dual-lifetime write state auditable,
then tested runtime HMC write controls that target those action surfaces:
    post-zeropower native-delta gating and suppression,
    overlap-geometry replay gating,
    dual-lifetime short/long write controls.

The measured effect is still weak and local.
This does not support the hypothesis that the current TTT fast-weight write-only action space
contains enough controllable drift/scale correction to reach Target-25.
```

Next required direction：

```text
Do not run selector/full online validation from v18 candidates.
Stop treating TTT write-only as the Target-25 mainline unless a new action family first passes the sandbox gate.
Use TTT write as a stabilizer/regularizer, and move Target-25 work toward explicit online trajectory-state / scale-state modules.
Any future full online target-25 validation must be preceded by:
    h10/h15 ATE delta <= -3m,
    or [200,300) delta <= -5m with downstream regression <= +1m,
    and a no-GT selector gate.
```

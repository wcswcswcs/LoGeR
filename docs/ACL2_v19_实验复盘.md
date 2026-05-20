# ACL2 v19 实验复盘：TTT Trajectory-State Write Reboot Target25

日期：2026-05-20 至 2026-05-21（Asia/Singapore）  
计划文件：`docs/ACL2_v19_TTT_TrajectoryState_Write_Reboot_Target25_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v19_ttt_trajectory_state_write_reboot_target25/`

本轮原则：只记录实际落盘结果；不把 offline audit、sandbox oracle、short rollout、GT audit、failed gate、或 h15 extension 写成 deployable TTT success。没有通过 sandbox gate 时，不启动 no-GT selector，也不启动 full online validation。

---

## 0. 工程与配置复盘

新增 / 修改：

```text
run_pipeline_abc_v2.py:
    新增 --ttt_write_scale_state_mode
    新增 --ttt_write_scale_state_proxy
    新增 --ttt_write_scale_state_carrier
    新增 --ttt_write_scale_state_alpha
    新增 --ttt_write_scale_state_branch_mask
    新增 --ttt_write_scale_state_chunks
    在 probe_ttt_write commit 前计算 no-GT scale-state payload
    将 v19 scale-state action 写入 TTT controller debug

loger/pipeline/ttt_write_controller.py:
    新增 v19 scale-state projection risk source
    支持 v19_scale_state / scale_state / online_scale_state / nogt_scale_state / trajectory_scale_state
    支持 carrier:
        all
        special_token
        structure_lowdg
        overlap_static_anchor
    使用 no-GT pose-step EMA scale proxy 调制 tri-replay risk

loger/pipeline/hybrid_memory_controller.py:
    将 v19 scale-state 参数透传到 TTTWriteController

tools/run_attention_cue_experiment.sh:
    透传 TTT_WRITE_SCALE_STATE_* 环境变量

tools/v19_scale_proxy_audit.py:
    B0 no-GT scale proxy offline audit

tools/run_v19_candidate_rollout.sh:
    v19 trusted short-rollout launcher
    固定 H9 parent + v16 causal fork snapshots
    支持 K1_H9 / SCALETTT_* / SCALECOMMIT_* candidates

tools/run_v19_matrix.sh:
    v19 GPU worker scheduler
    默认 GPU_LIST=0,1,2,3,4,5

tools/v19_candidate_bank_report.py:
    聚合 h5/h8/h10/h15 candidate metrics
    使用同帧 intersection 计算 delta

tools/v19_drift_state_autopsy.py:
    Track A landed-artifact drift-state autopsy

tools/v19_historical_basis_fit.py:
    Track D historical basis trace audit
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    run_pipeline_abc_v2.py \
    loger/pipeline/ttt_write_controller.py \
    loger/pipeline/hybrid_memory_controller.py \
    tools/v19_scale_proxy_audit.py \
    tools/v19_candidate_bank_report.py \
    tools/v19_drift_state_autopsy.py \
    tools/v19_historical_basis_fit.py

bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v19_candidate_rollout.sh
bash -n tools/run_v19_matrix.sh

PASS
```

工程 blocker 与修复：

```text
blocker 1:
    第一条 v19 action smoke 失败，原因是 argparse 的
    --ttt_write_gradient_reversal_risk_source choices 漏掉 v19_scale_state。

fix:
    在 run_pipeline_abc_v2.py 中补充合法 choices:
        v19_scale_state
        scale_state
        online_scale_state
        nogt_scale_state
        trajectory_scale_state
    重跑 py_compile PASS。
    同一 candidate smoke 后续成功落盘。

blocker 2:
    SCALETTT_02 chunk6 h3 smoke 成功运行但 ATE 从 baseline 8.9117m 变为 10.8901m。

fix / follow-up:
    按计划不把 smoke 负结果当终点。
    继续完整 B1/B2 sandbox matrix，并对 top candidate 做 h15 extension。

blocker 3:
    Track D historical basis 只有 3 个 landed true-action trace。

fix / decision:
    不编造 learned basis。
    记录为 insufficient landed true-action trace diversity。
```

调度优化：

```text
tools/run_v19_matrix.sh 默认 GPU_LIST 从 0,1,2,3 改为 0,1,2,3,4,5。
该修改只改变并行度，不改变模型参数、chunk、horizon、评分或落盘 metric。
```

---

## 1. Phase 0 Boundary

v19 复用 v16 已通过的 locked boundary 与 causal fork parity：

```text
results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25/registry_v16_phase0_boundary_R3.csv
results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25/phase1_causalfork/phase1_causalfork_gate_summary_R3.json
```

有效 full online boundary：

| Run | ATE | Rot | RPE_t | FinalErr | `[200,300)` | `[400,600)` | hmc rows | Counts as TTT write |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `H9_P0_R2` | `34.1257769401` | `6.5414` | `92.4053` | `6.189399` | `74.409927` | `44.353638` | `38` | `true` |
| `C9_P0_R2` | `33.7629421029` | `6.5259` | `92.3871` | `5.666384` | `76.102136` | `41.896364` | `38` | `true` |
| `WINGAM_P0_R3` | `34.1902782732` | `6.5666` | `92.4202` | `6.195236` | `75.576021` | `42.280485` | `38` | `true` |

固定解释：

```text
当前最好可计数 full online TTT write 仍是 C9_P0_R2:
    ATE = 33.7629421029m

v19 candidate parent 仍使用 H9:
    H9 full ATE 比 C9 差，
    但 H9 在 [200,300) 病灶段更好。
```

---

## 2. Track A Drift-State Autopsy

输出：

```text
trackA_drift_state_autopsy_R1/
trackA_drift_state_autopsy_R2_with_v19/
```

R1 使用 v18 landed artifacts；R2 加入 v19 B1/B2 结果。

R2 summary：

| Metric | Value |
|---|---:|
| row count | `148` |
| candidate row count | `145` |
| h10/h15 candidate rows | `49` |
| best h10/h15 ATE delta vs H9 | `-2.2244256984` |
| best h10/h15 `[200,300)` delta vs H9 | `-2.2990377600` |
| existing action gate-like pass | `false` |
| Spearman(action proxy, ATE delta) | `0.2696125659` |
| Spearman(action proxy, Rot) | `-0.2526051841` |
| Spearman(action proxy, FinalErr) | `-0.1345868265` |
| Spearman(`[200,300)` delta, `[400,600)` delta) | `0.2522657589` |

Interpretation：

```text
加入 v19 后，已有 action 的最好 h10/h15 signal 明显强于 v18，
但仍未达到 formal gate:
    h10/h15 ATE delta <= -3m
    or [200,300) delta <= -5m with downstream regression <= +1m

因此 Track A 支持继续探索 trajectory/scale-state action，
但不支持直接启动 selector/full online validation。
```

---

## 3. Track B0 Scale Proxy Offline Audit

输出：

```text
trackB0_scale_proxy_audit_R1/
```

该阶段只做 offline audit。GT 只作为离线 correlation target，不进入 runtime action。

Summary：

| Metric | Value |
|---|---:|
| trajectory artifacts seen | `114` |
| proxy rows | `1076` |
| failed artifacts | `0` |
| horizon | `10` |
| best abs Spearman | `0.3938189655` |
| best proxy | `scale_proxy_pose_step_median_log_ratio_ema` |
| best target | `segment_200_300_ate` |
| gate pass any abs >= 0.35 | `true` |
| uses GT runtime action | `false` |

Decision：

```text
B0 gate = pass
no-GT pose-step EMA scale proxy 有足够 offline correlation，
允许进入 B1/B2 scale-state TTT replay / commit modulation candidate bank。
```

---

## 4. Track B1/B2 Scale-State Candidate Bank

输出：

```text
trackB1B2_scale_state_R1/
rollouts/V19_P2SCALE_R1_*
matrix_logs/trackB1B2_scale_state_initial/
```

矩阵：

```text
candidates:
    K1_H9
    SCALETTT_01_SPECIAL_TOKEN_W0_A005
    SCALETTT_02_STRUCTURE_LOWDG_W0_A005
    SCALETTT_03_OVERLAP_STATIC_W0W2_A005
    SCALETTT_04_STRUCTURE_LOWDG_W0_A010
    SCALETTT_05_OVERLAP_STATIC_W0W2_A010
    SCALECOMMIT_01_PZBASIS_HARM_W0_G025
    SCALECOMMIT_02_AUXGEO_OVERLAP_W0_G025
    SCALECOMMIT_03_HIST_DELTA_W0_G025

chunks: 6, 10
horizons: 5, 8, 10
planned rows: 54
completed rows: 54
failures: 0

top-candidate h15 extension:
    K1_H9 chunk10 h15
    SCALECOMMIT_01_PZBASIS_HARM_W0_G025 chunk10 h15
completed rows: 2
failures: 0
```

Gate summary after h15 extension：

| Metric | Best |
|---|---|
| Status | `weak` |
| Best h10/h15 ATE delta vs H9 | `-2.2244256984` |
| Best h10/h15 ATE candidate | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025`, chunk `10`, h`10` |
| Best `[200,300)` delta vs H9 | `-2.2990377600` |
| Best `[200,300)` candidate | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025`, chunk `10`, h`10` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Best by chunk / horizon：

| Chunk | Horizon | Best ATE candidate | ATE delta vs H9 | Best `[200,300)` candidate | `[200,300)` delta | Downstream proxy delta |
|---:|---:|---|---:|---|---:|---:|
| `6` | `5` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `+0.819884` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `+0.869938` | `nan` |
| `6` | `8` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `+0.425668` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `+0.187137` | `-0.552407` |
| `6` | `10` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `-0.154291` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `-0.354012` | `-1.455725` |
| `10` | `5` | `SCALECOMMIT_02_AUXGEO_OVERLAP_W0_G025` | `-0.338711` | `SCALETTT_01_SPECIAL_TOKEN_W0_A005` | `-1.548288` | `-0.470291` |
| `10` | `8` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `+0.029687` | `SCALETTT_01_SPECIAL_TOKEN_W0_A005` | `+0.149101` | `+2.046526` |
| `10` | `10` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `-2.224426` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `-2.299038` | `-3.170471` |
| `10` | `15` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `-0.857679` | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `-1.633118` | `-2.143290` |

Important h15 rows：

| Candidate | Chunk | Horizon | ATE | ATE delta vs H9 | `[200,300)` delta | `[400,600)` delta | Done |
|---|---:|---:|---:|---:|---:|---:|---|
| `K1_H9` | `10` | `15` | `26.6392119621` | `-0.0264659731` | `-0.4321153091` | `-0.1165855816` | `true` |
| `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `10` | `15` | `25.8079991816` | `-0.8576787536` | `-1.6331179056` | `-2.1432898908` | `true` |

Interpretation：

```text
v19 找到当前最强 short-rollout TTT-write signal:
    SCALECOMMIT_01_PZBASIS_HARM_W0_G025
    chunk10 h10 ATE delta = -2.2244m
    [200,300) delta = -2.2990m
    [400,600) proxy delta = -3.1705m

这是明显好于 v18 best h10 ATE delta -0.6504m 的 weak signal。

但是：
    h10 仍未达到 -3m ATE gate；
    [200,300) 仍未达到 -5m gate；
    h15 extension 衰减到 -0.8577m。

因此 B1/B2 = weak signal, no stage pass。
```

---

## 5. Track D Historical Basis Fit

输出：

```text
trackD0_historical_basis_fit_R1/
```

Summary：

| Metric | Value |
|---|---:|
| trace count | `3` |
| basis row count | `3` |
| minimum diverse traces | `6` |
| enough diversity for PLS basis | `false` |
| candidate builder allowed | `false` |

Decision：

```text
Track D0 = blocked by insufficient landed true-action trace diversity.
没有生成 LEARNBASIS candidates。
没有编造 historical basis。
```

---

## 6. Downstream Phase Decision

| Phase / Track | Status | Reason |
|---|---|---|
| Phase 0 Boundary | pass | reused v16 locked H9/C9/WINGAM boundary and causal fork parity |
| Track A Drift Autopsy | pass diagnostic | confirms best existing/v19 action is weak, no gate-like pass |
| Track B0 Scale Proxy Audit | pass | best abs Spearman `0.3938` >= `0.35` |
| Track B1/B2 Scale-State Candidate Bank | weak, no stage pass | best h10 ATE delta `-2.2244m`; h15 decays to `-0.8577m` |
| Track D Historical Basis Fit | blocked | only `3` traces, below min `6`; no learned basis generated |
| No-GT Selector | not started | sandbox gate not met |
| Full Online Validation | not started | forbidden by failed stage gate |

Boundary：

```text
No v19 short-rollout result counts as deployable TTT write success.
No GT-selected candidate is counted.
No no-GT selector was evaluated.
No full online validation was launched.
No online target-25 result was produced in v19.
Current best deployable online TTT write remains C9_P0_R2:
    ATE = 33.7629421029m
```

---

## 7. Final Decision

v19 的真实成功点：

```text
B0 no-GT scale proxy audit passed:
    best proxy = scale_proxy_pose_step_median_log_ratio_ema
    Spearman vs segment_200_300_ate = 0.3938189655

v19 scale-state TTT write path implemented and audited:
    v19_scale_state_active=true appears in landed logs
    carrier / alpha / chunks / risk source are applied at runtime

B1/B2 candidate bank completed:
    planned matrix rows = 54/54 complete
    h15 extension rows = 2/2 complete
    failures = 0

Best v19 weak signal:
    SCALECOMMIT_01_PZBASIS_HARM_W0_G025
    chunk10 h10 ATE delta = -2.2244256984m
    [200,300) delta = -2.2990377600m
    downstream proxy [400,600) delta = -3.1704707038m
```

v19 的关键负结果：

```text
No candidate passed the formal sandbox gate.
h15 extension did not strengthen the signal:
    SCALECOMMIT_01 chunk10 h15 ATE delta = -0.8576787536m

Track D learned basis could not be started honestly:
    only 3 landed true-action traces exist
    min required diverse traces = 6
```

Interpretation：

```text
v19 partially validates the hypothesis that trajectory/scale-state signals matter:
    the best v19 candidate is much stronger than v18 best.

But the signal is not yet durable enough:
    h10 is weak-positive but below gate;
    h15 decays;
    target-25 remains unreached.

This suggests the current TTT fast-weight write path can nudge trajectory-state error,
but the current scale-state projection risk is not strong or stable enough to justify
selector or full online validation.
```

Next required direction：

```text
Do not run selector/full online validation from v19 candidates.

Recommended follow-up:
    1. Use SCALECOMMIT_01 as the new weak-positive anchor.
    2. Improve durability from h10 to h15:
        continuity-preserving basis,
        later-layer-only routing,
        reset-body-only routing,
        or W_short-style scale-state carrier.
    3. Collect more true-action traces before Track D learned-basis candidates:
        current trace_count = 3 < 6.
    4. Keep all future full online validation gated by:
        h10/h15 ATE delta <= -3m,
        or [200,300) delta <= -5m with downstream regression <= +1m,
        plus no-GT selector pass.
```

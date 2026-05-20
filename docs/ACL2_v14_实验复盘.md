# ACL2 v14 实验复盘：TTT Write Causal Sandbox Target25

日期：2026-05-19（Asia/Singapore）  
计划文件：`docs/ACL2_v14_TTT_Write_Causal_Sandbox_Target25_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v14_ttt_causal_sandbox_target25/`

本轮原则：只记录实际落盘结果；不把 failed / partial run 写成实验成功；不把 offline trajectory rewrite、GT audit、sandbox 工具接线写成 deployable TTT success。用户约束后所有 v14 run 只使用 GPU `0/1`，没有使用 GPU `4/5/6/7`。

---

## 0. v14 固定边界

计划要求 Phase 0 先复现：

```text
H9_REPEAT ATE target = 34.1258, tolerance <= 0.03m
C9_WEAK_FREEZE_C56 ATE target = 33.7629, tolerance <= 0.03m
online runs hmc rows = 38
NOGTPOSE_27 counts_as_ttt_write = false
```

Phase 0 未通过时，不允许进入 Phase 1 sandbox / Phase 2 candidate bank。

---

## 1. 工程接线

新增 / 修改：

```text
tools/run_attention_cue_experiment.sh:
    START_FRAME
    SAVE_HMC_STATES
    SAVE_HMC_STATE_CHUNKS
    LOAD_HMC_STATE_AT_CHUNK

run_pipeline_abc_v2.py:
    --save_hmc_state_chunks
    chunk_XXX_input.pt  # reset_every 后、真正进入 chunk 的 effective input state
    load_hmc_state_at_chunk 后将 state tensor tree 移回 args.device
    _save_hmc_state 先 deepcopy 再搬 CPU，避免保存副作用

tools/ttt_short_rollout_sandbox.py:
    landed sandbox/full window audit tool
    输出 short_rollout_metrics.csv / short_rollout_gt_audit.csv /
        short_rollout_proxy.jsonl / candidate_commit_debug.jsonl

tools/v14_prepare_effective_hmc_states.py:
    旧 snapshot 的 effective input state 衍生工具
```

接线验证：

```text
python -m py_compile run_pipeline_abc_v2.py tools/v14_prepare_effective_hmc_states.py tools/ttt_short_rollout_sandbox.py
bash -n tools/run_attention_cue_experiment.sh
PASS
```

说明：这些工具只完成了 v14 sandbox 必需的工程接口。由于 Phase 0 gate 失败，`tools/ttt_short_rollout_sandbox.py` 没有被用于启动 Phase 1 parity 结论。

---

## 2. Phase 0 Attempts

### 2.1 Attempt R1

Run：

```text
V14_P0_H9_REPEAT_SWKS3
V14_P0_C9_WEAK_FREEZE_C56_REPEAT_SWKS3
```

结果登记：

```text
results/kitti01_hmc_v2/acl2_v14_ttt_causal_sandbox_target25/registry_v14.csv
```

| Run | Status | ATE | Rot | hmc rows | Gate note |
|---|---|---:|---:|---:|---|
| `H9_REPEAT` | DONE | `34.2513` | `6.4833` | `38` | H9 drift `+0.1255m` > `0.03m`, fail |
| `C9_WEAK_FREEZE_C56_REPEAT` | DONE | `33.7629` | `6.5259` | `38` | C9 repeat pass |

R1 不能作为 Phase 0 pass，因为 H9 baseline 没有复现。

### 2.2 Attempt R2

Run：

```text
V14_P0_H9_REPEAT_R2_SWKS3
V14_P0_C9_WEAK_FREEZE_C56_REPEAT_R2_SWKS3
```

状态：

```text
both failed at hmc_rows = 29 / 38
reason = SAVE_HMC_STATES saved before/input/after for every chunk
observed state snapshot footprint before cleanup:
    H9_REPEAT_R2 ~= 78GB
    C9_REPEAT_R2 ~= 78GB
filesystem reached 100%
```

R2 没有完整 trajectory / benchmark，不计入实验结果。R2 失败后只清理了本轮生成的 failed state snapshot 目录，保留 run log 证据。

### 2.3 Attempt R3

修正：

```text
SAVE_HMC_STATE_CHUNKS = 5,10,16
```

Run：

| Run | GPU | Start | Done | hmc rows |
|---|---:|---|---|---:|
| `V14_P0_C9_WEAK_FREEZE_C56_REPEAT_R3_SWKS3` | `1` | `2026-05-19 07:16:45` | `07:49:32` | `38` |
| `V14_P0_H9_REPEAT_R3_SWKS3` | `0` | `2026-05-19 07:16:44` | `07:54:05` | `38` |

结果登记：

```text
results/kitti01_hmc_v2/acl2_v14_ttt_causal_sandbox_target25/registry_v14_final.csv
results/kitti01_hmc_v2/acl2_v14_ttt_causal_sandbox_target25/registry_v14_final.json
```

| Run | counts_as_ttt_write | ATE | Rot | RPE_t | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `H9_REPEAT_R3` | `true` | `34.2513` | `6.4833` | `92.4058` | `5.941` | `74.501` | `54.753` | `44.828` |
| `C9_WEAK_FREEZE_C56_REPEAT_R3` | `true` | `33.7629` | `6.5259` | `92.3871` | `5.666` | `76.102` | `56.006` | `41.896` |
| `NOGTPOSE_27_reset_global_clip35_body600_t105` | `false` | `22.3669` | n/a | n/a | `9.032` | `34.867` | `26.944` | `8.516` |

Dashboard：

```text
results/kitti01_hmc_v2/acl2_v14_ttt_causal_sandbox_target25/phase0_global_drift_dashboard/
```

No-GT diagnostic：

```text
results/kitti01_hmc_v2/acl2_v14_ttt_causal_sandbox_target25/phase0_nogt_pose_proxy/
```

State snapshots landed for future debugging only:

```text
results/kitti01_hmc_v2/acl2_v14_ttt_causal_sandbox_target25/phase1_sandbox/state_snapshots/
    H9_REPEAT_R3/chunk_{005,010,016}_{before,input,after}.pt
    C9_REPEAT_R3/chunk_{005,010,016}_{before,input,after}.pt
```

Total snapshot footprint after limiting chunks:

```text
18 files
~15GB
```

---

## 3. Phase 0 Gate Decision

Gate checks:

| Check | Required | Observed | Result |
|---|---:|---:|---|
| H9 repeat drift | `abs(ATE - 34.1258) <= 0.03` | `abs(34.2513 - 34.1258) = 0.1255` | fail |
| C9 repeat drift | `abs(ATE - 33.7629) <= 0.03` | `< 0.0001` | pass |
| H9 hmc rows | `38` | `38` | pass |
| C9 hmc rows | `38` | `38` | pass |
| online commit mode | `probe_ttt_write` | true for both R3 online runs | pass |
| NOGTPOSE_27 flag | `counts_as_ttt_write=false` | false | pass |
| NOGTPOSE_27 drift vs v10/v11 record | `<= 0.05m` | `abs(22.3669 - 22.4012) = 0.0343` | pass |

Phase 0 result:

```text
gate_pass = false
reason = H9_REPEAT_R3 did not reproduce the locked H9 baseline
```

This is not a target-25 result and not a TTT action-space conclusion. It is a boundary/reproducibility failure at Phase 0.

---

## 4. Downstream Phases

| Phase | Status | Reason |
|---|---|---|
| Phase 1 TTT Causal Sandbox | not started | Phase 0 H9 repeat failed |
| Phase 2 Candidate Commit Bank Oracle | not started | Phase 1 forbidden |
| Phase 3 No-GT Short-Horizon Selector | not started | no oracle upper bound |
| Phase 4 New TTT Action Families | not started | v14 plan forbids bypassing failed Phase 0/1 |
| Phase 5 Full Online Validation | not started | no candidate entered |
| Phase 6 Failure routing | stopped at Phase 0 | reproducibility issue must be resolved first |

Important boundary:

```text
C9_REPEAT_R3 is a deployable online repeat of the v13 C9 policy.
It is not a new v14 candidate.
No Phase 1 sandbox parity was run.
No candidate bank oracle table exists.
No no-GT selector was evaluated.
No online target-25 was produced.
```

---

## 5. Final Decision

v14 stops at Phase 0:

```text
No valid v14 Phase 0 pass.
No sandbox parity.
No candidate bank.
No new deployable TTT write candidate.
No online target-25.
```

Best completed online run in this v14 attempt:

```text
C9_WEAK_FREEZE_C56_REPEAT_R3
ATE / Rot = 33.7629 / 6.5259
counts_as_ttt_write = true
no GT runtime action
no offline trajectory rewrite
```

But it is a v13 repeat, not v14 progress:

```text
target-25 gap = 8.7629m
[200,300) = 76.102, worse than locked H9 baseline 74.410
```

Actionable next step before any v14 continuation:

```text
Reconcile why current H9 repeat lands at 34.2513 instead of locked 34.1258.
Until that baseline drift is resolved, v14 sandbox/candidate-bank results would be contaminated by engineering drift.
```

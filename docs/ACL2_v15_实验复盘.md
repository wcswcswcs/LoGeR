# ACL2 v15 实验复盘：TTT Write Repro CausalSandbox Target25

日期：2026-05-19（Asia/Singapore）  
计划文件：`docs/ACL2_v15_TTT_Write_Repro_CausalSandbox_Target25_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v15_ttt_repro_causal_sandbox_target25/`

本轮原则：只记录实际落盘结果；不把 offline trajectory rewrite、GT audit、partial run、sandbox smoke 写成 deployable TTT success。所有 v15 run 只使用 GPU `0/1/2/3`。

---

## 0. 工程与配置复盘

新增 / 修改：

```text
run_pipeline_abc_v2.py:
    --save_hmc_state_chunks
    --save_hmc_state_kinds
    --load_hmc_state_at_chunk 后移回目标 device
    _save_hmc_state deepcopy + cpu，避免保存副作用
    sliced run 输出 timestamp / merge window 尝试使用全局帧号

tools/run_attention_cue_experiment.sh:
    START_FRAME
    SAVE_HMC_STATES
    SAVE_HMC_STATE_CHUNKS
    SAVE_HMC_STATE_KINDS
    LOAD_HMC_STATE_AT_CHUNK

tools/ttt_short_rollout_sandbox.py:
    landed sandbox/full window audit
    added raw_pose_max_abs_diff_vs_full / raw_pose_max_trans_diff_vs_full

tools/v15_config_diff_report.py:
    config_diff_report.json / .md
```

验证：

```text
python -m py_compile run_pipeline_abc_v2.py tools/ttt_short_rollout_sandbox.py tools/v15_config_diff_report.py
bash -n tools/run_attention_cue_experiment.sh
PASS
```

配置 diff 输出：

```text
results/kitti01_hmc_v2/acl2_v15_ttt_repro_causal_sandbox_target25/phase0_config_diff/
```

关键复盘点：

```text
v13 locked H9 hmc_config.yaml: mp_alpha = 0.125
v14 H9 R3 hmc_config.yaml:     mp_alpha = 0.1
```

这解释了 v14 中 `C9_REPEAT_R3` 可复现而 `H9_REPEAT_R3` 漂移：C9 本来就是 `mp_alpha=0.1`，H9 locked reference 是 `0.125`。

---

## 1. Phase 0 Reproducibility

输出：

```text
registry_v15_phase0A.csv / .json
registry_v15_phase0BC.csv / .json
phase0_gate_summary.csv / .json
phase0_global_drift_dashboard_A/
phase0_global_drift_dashboard_BC/
```

### 1.1 P0-A no-state-save repeat

| Run | ATE | Rot | RPE_t | FinalErr | `[200,300)` | `[400,600)` | hmc rows | Drift vs target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `H9_P0_A` | `34.1258` | `6.5414` | `92.4053` | `6.189` | `74.410` | `44.354` | `38` | `-0.000023` |
| `C9_P0_A` | `33.7629` | `6.5259` | `92.3871` | `5.666` | `76.102` | `41.896` | `38` | `+0.000042` |
| `WINGAM_P0_A` | `34.1903` | `6.5666` | `92.4202` | `6.195` | `75.576` | `42.280` | `38` | `-0.000022` |

P0-A gate:

```text
H9 drift <= 0.03m: pass
C9 drift <= 0.03m: pass
WINGAM drift <= 0.03m: pass
hmc rows = 38 for all: pass
```

### 1.2 P0-B input-only state save and P0-C archived config replay

P0-B 只保存：

```text
chunk_005_input.pt
chunk_010_input.pt
chunk_016_input.pt
```

Snapshot footprint:

```text
6 files
~3.8GB
```

| Run | ATE | Rot | RPE_t | FinalErr | `[200,300)` | `[400,600)` | hmc rows | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `H9_P0_B` | `34.1258` | `6.5414` | `92.4053` | `6.189` | `74.410` | `44.354` | `38` | pass |
| `C9_P0_B` | `33.7629` | `6.5259` | `92.3871` | `5.666` | `76.102` | `41.896` | `38` | pass |
| `H9_P0_C` | `34.1258` | `6.5414` | `92.4053` | `6.189` | `74.410` | `44.354` | `38` | pass |

State-save side effect check:

```text
H9_P0_B - H9_P0_A ATE delta = 0.0000m <= 0.01m
C9_P0_B - C9_P0_A ATE delta = 0.0000m <= 0.01m
PASS
```

Phase 0 conclusion:

```text
Phase 0 gate = pass
v14 H9 drift root cause = H9 mp_alpha was run as 0.1 instead of locked 0.125
input-only HMC state snapshot path does not change full-run output
```

---

## 2. Phase 1 Causal Sandbox Parity

输出：

```text
phase1_sandbox/state_snapshots/
phase1_sandbox/native_rollouts/
phase1_sandbox/phase1_sandbox_gate_summary_R2.csv
phase1_sandbox/phase1_sandbox_gate_summary_R2.json
```

R1 发现审计 offset 假设错误：sliced run 的 `01.txt` timestamp 已经是全局帧号，因此不计入 gate。R2 使用 corrected audit。

### 2.1 HMC state hash parity

以 H9 chunk5 为例，full run 与 sandbox local chunks 的 HMC hashes 一致：

```text
full chunk5 input  = bad8b3a4fec6473c
sandbox chunk0 in  = bad8b3a4fec6473c
full chunk5 output = dc1e3035d10531e3
sandbox chunk0 out = dc1e3035d10531e3
full chunk6 output = be362523026829f6
sandbox chunk1 out = be362523026829f6
full chunk7 output = 1fe19658d20459a6
sandbox chunk2 out = 1fe19658d20459a6
```

说明：HMC fast-weight state load/commit 本身可以复现；失败发生在 trajectory / merge / pose gauge 层。

### 2.2 R2 native sandbox gate

| Run | Horizon | ATE diff vs full suffix | Raw pose max abs diff | Raw trans max diff | Gate |
|---|---:|---:|---:|---:|---|
| `H9_chunk5_h1_R2` | `1` | `-0.4392` | `9.3610` | `11.9086` | fail |
| `H9_chunk5_h3_R2` | `3` | `+0.1721` | `12.3442` | `13.8399` | fail |
| `H9_chunk5_h5_R2` | `5` | `+0.0091` | `14.2220` | `16.0356` | h5 ATE pass only |
| `H9_chunk10_h3_R2` | `3` | `-0.2435` | `22.2099` | `25.2264` | fail |
| `H9_chunk16_h3_R2` | `3` | `-0.3755` | `33.4826` | `37.5548` | fail |
| `C9_chunk5_h3_R2` | `3` | `+0.0210` | `12.5839` | `14.1061` | fail |
| `C9_chunk10_h3_R2` | `3` | `-0.2342` | `22.5613` | `25.5929` | fail |
| `C9_chunk16_h3_R2` | `3` | `-0.3570` | `33.8715` | `37.9433` | fail |

Phase 1 pass criteria:

```text
horizon=1 raw pose max diff <= 1e-5
horizon=3 ATE_suffix_delta <= 0.01m
horizon=5 ATE_suffix_delta <= 0.03m
H9/C9 all pass
```

Actual:

```text
h1 raw pose diff = 9.3610 >> 1e-5
h3 best ATE diff = 0.0210m > 0.01m
most h3 windows differ by 0.17m to 0.38m
Phase 1 gate = fail
```

Interpretation:

```text
Saved HMC state is sufficient to reproduce future HMC fast-weight hashes.
It is not sufficient to reproduce full-run trajectory suffix.
The missing state is likely trajectory/merge/pose-gauge state outside HybridMemoryState.
```

---

## 3. Downstream Phase Decision

| Phase | Status | Reason |
|---|---|---|
| Phase 2 Candidate Commit Bank Oracle | not started | Phase 1 sandbox parity failed |
| Phase 3 No-GT Selector | not started | no trusted oracle bank |
| Phase 4 New TTT Action Family full runs | not started | candidate/sandbox gate not established |
| Phase 5 Full Online Validation | not started | no candidate entered |

Boundary:

```text
No candidate bank result exists.
No GT-selected candidate is counted.
No no-GT selector was evaluated.
No new deployable online TTT write candidate was produced in v15.
No online target-25 was produced.
```

---

## 4. Final Decision

v15 成功修复了 v14 的 Phase 0 reproducibility issue：

```text
Best locked H9 repeats:
    H9_P0_A / H9_P0_B / H9_P0_C
    ATE / Rot = 34.1258 / 6.5414

Best deployable online repeat:
    C9_P0_A / C9_P0_B
    ATE / Rot = 33.7629 / 6.5259
    counts_as_ttt_write = true
```

但 v15 停在 Phase 1：

```text
Native short-rollout sandbox did not reproduce full-run trajectory suffix.
HMC fast-weight hash parity is not enough for trajectory-level parity.
Candidate bank / selector / action-family full runs are forbidden by the v15 stop rule.
```

下一步必须先扩展 sandbox state，而不是跑 candidate bank：

```text
save/load trajectory merge state or implement full-run internal fork candidate;
then rerun Phase 1 h1/h3/h5 parity;
only after Phase 1 passes can Phase 2 candidate commit bank oracle start.
```


# ACL2 v6 实验记录

日期：2026-05-07  
计划文件：`docs/ACL2_v6_Reproducibility_CachedSemanticPrior_TTTWrite_InitialPlan_v2.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v6_phase0_repro/`  
目标：先完成 Stage C/D 接入后的 reproducibility / cached semantic prior no-op gate，再进入语义 prior 与 TTT 写入实验。

说明：从 v6 开始，实验数据和结论统一记录在本文件；计划文件只保留实验设计与 gate。

---

## 1. Phase 0B：复现 `ACL2V5_SWKS3_03`

固定目标：

```text
reference = ACL2V5_SWKS3_03
expected KITTI01 ATE / Rot = 36.4153 / 6.6186
gate = |ATE - 36.4153| <= 0.03m, |Rot - 6.6186| <= 0.03deg
```

固定协议：

```text
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA_WRITE_KEEP_SCOPE = both_overlap
SWA_OVERLAP_SOURCE_REPLACE = source / kv / alpha 0.50 / last
RESET_EVERY = 5
```

| Run | Stage C/D 状态 | ATE RMSE | Rot RMSE | RPE t | RPE r | pose max diff vs B0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `ACL2V6_B0_SWKS3_reference_rerun` | off | `36.416102` | `6.612796` | `92.445197` | `0.008169` | reference | 通过 v5 historical best reproduction gate |

Trajectory diagnostics：

输出目录：

```text
results/kitti01_hmc_v2/acl2_v6_phase0_repro/trajectory_diagnostics_b0/
```

| Run | ATE RMSE | Final error | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ACL2V6_B0_SWKS3_reference_rerun` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `3.765` | `31.238916` |

结论：

1. v6 当前代码在 Stage C/D disabled 状态下成功复现 `ACL2V5_SWKS3_03`。
2. ATE 差值 `+0.0008m`，Rot 差值 `-0.0058deg`，均小于 gate 阈值。

---

## 2. Phase 0 工程接线

本轮补齐 v6 计划需要的最小 no-op / cache 接口：

- `run_pipeline_abc_v2.py`
  - 新增 `--stage_c_cache_dir`、`--stage_c_cache_mode`、`--stage_c_cache_require_hit`、`--stage_c_cache_validate`。
  - 新增 `--stage_c_inline_when_ignored`，默认 `0`，用于避免 Stage C 在 HMC ignore prior 的主进程中扰动 no-op parity。
  - 新增 `--semantic_prior_mode disabled/noop/pass_through/spg_v2`。
  - 新增 `--hmc_ignore_semantic_prior`。
  - `noop/pass_through` 会构造全 1 的 `PriorOutput`，用于 no-op / pass-through 安全性测试。
- `loger/pipeline/hybrid_memory_controller.py`
  - `prior_output=None` 时仍保留 HMC write-score override。
  - 当 semantic prior 被忽略且 `hmc_write_score_source=stage_d_x_*` 时，HMC 内部重建历史 v5 `stage_d` dyn-rank 基底，避免 base prior 缺失导致 write score 退化成全 0。
- `tools/run_attention_cue_experiment.sh`
  - 透传 `STAGE_C_MODE`、`STAGE_C_CACHE_*`、`STAGE_C_INLINE_WHEN_IGNORED`、`SEMANTIC_PRIOR_MODE`、`HMC_IGNORE_SEMANTIC_PRIOR`。

静态验证：

```text
python3 -m py_compile run_pipeline_abc_v2.py loger/pipeline/hybrid_memory_controller.py
bash -n tools/run_attention_cue_experiment.sh
```

结果：通过。

---

## 3. Phase 0 smoke：Stage C cache / Stage D no-op

固定：

```text
END_FRAME = 128
read = C23 past + pair/all beta 4.75
write = stage_d_x_dg_inv_sqrt
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA_WRITE_KEEP_SCOPE = both_overlap
SWA_OVERLAP_SOURCE_REPLACE = source / kv / alpha 0.50
RESET_EVERY = 5
```

| Run | Stage C cache | Stage D / HMC | ATE RMSE | Rot RMSE | RPE t | RPE r | pose max abs diff | cache | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `ACL2V6_SMOKE_B1_stageC_cache_ignored_e128` | `readwrite` compute/save | semantic disabled, HMC ignore | `1.354989` | `3.029616` | `95.055239` | `0.026470` | reference | wrote 5 chunks | Stage C cache 生成成功 |
| `ACL2V6_SMOKE_B2_stageC_cache_read_ignored_e128` | `read`, require hit | semantic disabled, HMC ignore | `1.355008` | `3.032874` | `95.051396` | `0.026456` | `0.00086` vs B1 | require-hit pass | cache load 成功 |
| `ACL2V6_SMOKE_B3_stageD_noop_ignored_e128` | `read`, require hit | `noop`, HMC ignore | `1.355008` | `3.032874` | `95.051396` | `0.026456` | `0.00086` vs B1 | require-hit pass | Stage D 构造但未消费，和 B2 完全一致 |
| `ACL2V6_SMOKE_B4_pass_through_consumed_e128` | `read`, require hit | `pass_through`, HMC consumed | `1.355008` | `3.032874` | `95.051396` | `0.026456` | `0.00086` vs B1 | require-hit pass | pass-through consumed 与 B2/B3 完全一致 |

补充：

- smoke 通过只代表短序列接口可用；full no-op parity 仍以 KITTI01 full B0/B1/B2 为准。
- 后续 full B1 证明 inline Stage C compute 即使被 HMC ignore，也会造成 full sequence 漂移，因此 no-op benchmark 不应在主 LoGeR/HMC 进程里 inline 跑 Stage C 前端。

---

## 4. Phase 0B full no-op parity：失败诊断与修复

### 4.1 失败诊断

第一轮 full B1 启用 Stage C cache compute/load 代码路径，但 HMC 忽略 semantic prior。结果未通过 B0 parity：

| Run | Stage C / semantic | ATE RMSE | Rot RMSE | RPE t | RPE r | pose max diff vs B0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `ACL2V6_B1_SWKS3_stageC_cache_ignored` | Stage C inline compute/save; semantic disabled; HMC ignore | `36.597859` | `6.704113` | `92.314059` | `0.008190` | `0.0353590000` | 失败；超过 `0.01m / 0.01deg` hard parity gate |
| `ACL2V6_B1F_SWKS3_stageC_ignored_noinline` | Stage C skipped when ignored; semantic disabled; HMC ignore | `36.551280` | `6.693533` | `92.445209` | `0.008217` | `0.0884090000` | 仍失败 |
| `ACL2V6_B2_SWKS3_stageC_cache_read_ignored` | Stage C cache read require-hit; semantic disabled; HMC ignore | `36.551280` | `6.693533` | `92.445209` | `0.008217` | `0.0884090000` | 与 B1F 完全一致；cache load 本身不是主要问题 |

定位：

1. `B1` 的 inline Stage C compute 会扰动后续 LoGeR/HMC 数值路径，因此不能作为 no-op parity 的安全方式。
2. `B1F/B2` 仍失败，说明第二个问题在 HMC write-score 路径：`hmc_ignore_semantic_prior=1` 后没有 `PriorOutput`，`stage_d_x_dg_inv_sqrt` 的 `stage_d` 基底退成全 1，再被 `_normalize01` 归一为全 0。
3. 修复：当 semantic prior 被忽略且 write score 是 `stage_d_x_*` 时，HMC 内部按 v5 的 dyn rank 逻辑重建 `stage_d` 基底，再乘对应 Dg/explicit dyn 项。

### 4.2 修复后结果

修复后重新跑 full B1/B2：

| Run | Stage C / semantic | ATE RMSE | Rot RMSE | RPE t | RPE r | pose max diff vs B0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `ACL2V6_B1G_SWKS3_stageC_ignored_noinline_fixed` | Stage C skipped when ignored; semantic disabled; HMC ignore; `stage_d` base rebuilt | `36.416102` | `6.612796` | `92.445197` | `0.008169` | `0.0000000000` | 通过；与 B0 bit-level trajectory 对齐 |
| `ACL2V6_B2G_SWKS3_stageC_cache_read_ignored_fixed` | Stage C cache read require-hit; semantic disabled; HMC ignore; `stage_d` base rebuilt | `36.416102` | `6.612796` | `92.445197` | `0.008169` | `0.0000000000` | 通过；cache-read no-op 与 B0 完全一致 |

运行时间：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `ACL2V6_B1G_SWKS3_stageC_ignored_noinline_fixed` | `2026-05-07 19:12:04` | `2026-05-07 19:35:05` | `23.0 min` |
| `ACL2V6_B2G_SWKS3_stageC_cache_read_ignored_fixed` | `2026-05-07 19:12:04` | `2026-05-07 19:38:31` | `26.5 min` |

Gate 结论：

1. `B0` 复现 v5 historical best 通过。
2. 修复后的 `B1G/B2G` 对 `B0` 的 ATE/Rot/RPE/trajectory 完全一致，Phase 0B hard no-op parity 通过。
3. Stage C cache require-hit read 在 semantic ignored / HMC ignore prior 时是安全 no-op。
4. Stage C inline compute 不应和 LoGeR/HMC full benchmark 混在同一主进程中作为 parity run；后续需要 cache 时，应先离线预计算或使用已存在 cache，再进入 HMC benchmark。

### 4.3 Stage D noop / pass-through full gate

为了完成 Phase 0B 的剩余 no-op diagnostic，继续跑：

- `B3G`：Stage C cache read require-hit + Stage D `noop`，但 HMC 忽略 semantic prior。
- `B4G`：Stage C cache read require-hit + Stage D `pass_through`，HMC 消费全 1 `PriorOutput`。

本轮额外修复：

- 当 `prior_output.A_tok` 是全 1 pass-through，且 `hmc_write_score_source=stage_d_x_*` 时，HMC 把它视为 no-op semantic prior，并重建 v5 `stage_d` 基底。
- 这避免 pass-through consumed 把 `stage_d_x_dg_inv_sqrt` 的 base prior 归一成全 0。

| Run | Stage C / semantic | ATE RMSE | Rot RMSE | RPE t | RPE r | pose max diff vs B0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `ACL2V6_B3G_SWKS3_stageD_noop_ignored_fixed` | cache read require-hit; Stage D `noop`; HMC ignore prior | `36.416102` | `6.612796` | `92.445197` | `0.008169` | `0.0000000000` | 通过；Stage D 构造但未消费是严格 no-op |
| `ACL2V6_B4G_SWKS3_pass_through_consumed_fixed` | cache read require-hit; Stage D `pass_through`; HMC consumed | `36.416102` | `6.612796` | `92.445197` | `0.008169` | `0.0000000000` | 通过；全 1 pass-through consumed 也是严格 no-op |

运行时间：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `ACL2V6_B3G_SWKS3_stageD_noop_ignored_fixed` | `2026-05-07 19:41:14` | `2026-05-07 20:05:06` | `23.9 min` |
| `ACL2V6_B4G_SWKS3_pass_through_consumed_fixed` | `2026-05-07 19:41:14` | `2026-05-07 20:04:49` | `23.6 min` |

Debug spot check：

| Run | chunk0 `hash_H_next` | `prior_hmc_write_score_mean` | `prior_hmc_write_corr_score_dyn` | `prior_ttt_write_mean` |
|---|---|---:|---:|---:|
| `B0` | `67b2b4e6ae318f0c` | `0.420922` | `-0.824594` | `1.000000` |
| `B3G` | `67b2b4e6ae318f0c` | `0.420922` | `-0.824594` | `1.000000` |
| `B4G` | `67b2b4e6ae318f0c` | `0.420922` | `-0.824594` | `1.000000` |

最终 Phase 0B 结论：

1. `B0/B1G/B2G/B3G/B4G` 的 ATE/Rot/RPE 与 trajectory txt 全部对齐。
2. Stage C cache 目录 `results/kitti01_hmc_v2/acl2_v6_stage_c_cache_full_swks3/` 已包含 `38` 个 chunk cache。
3. `pass_through consumed` 已可作为严格 no-op diagnostic；后续仍建议用 `hmc_ignore_semantic_prior=1` 作为 hard reproduction 保护，用 `pass_through` 做 write controller 消费路径调试。
4. Phase 0B gate 通过，可以进入 Phase 1 causal audit / Phase 2 passive semantic audit。

---

## 5. Phase 1：`[200,300)` causal audit

目标：

```text
核心病灶 = [200,300) worst 100f segment
问题 = 该段是否和 TTT/SWA memory state 有因果关系
```

本批固定使用 Phase 0B 通过后的 SWKS3 协议，并补一个 diagnostic：

```text
ACL2V6_A1_03_freeze_chunks456_SWKS3
TTT_FREEZE_CHUNKS = 4,5,6
含义：丢弃 chunks 4/5/6 的 TTT write commit，其它 memory state 保持正常
```

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `ACL2V6_A1_03_freeze_chunks456_SWKS3` | `2026-05-07 20:09:37` | `2026-05-07 20:31:28` | `21.9 min` |

Benchmark：

| Run | 角色 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | v6 reproduction reference | `36.4161` | `6.6128` | `92.4452` | `0.0082` | 当前 v6 reproducible baseline |
| `A1_TTTonly_TTEX` | TTT-native-mix reference | `36.5932` | `6.4327` | `92.4423` | `0.0078` | v5 TTT extrapolate reference |
| `A1_SWAactive_SWOVR` | SWA-active tiny reference | `36.5915` | `6.4307` | `92.4416` | `0.0078` | v5 tiny best reference |
| `A1_freeze456` | freeze chunks `4,5,6` diagnostic | `60.6379` | `8.7116` | `92.7028` | `0.0111` | 全局崩坏；不是候选 |
| `A1_semantic_noop` | B4G pass-through consumed | `36.4161` | `6.6128` | `92.4452` | `0.0082` | 与 B0 完全一致 |

Trajectory / lesion diagnostics：

输出目录：

```text
results/kitti01_hmc_v2/acl2_v6_phase1_200_300_audit/trajectory_diagnostics_a1/
results/kitti01_hmc_v2/acl2_v6_phase1_200_300_audit/phase1_audit_tables/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0_SWKS3` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `A1_TTTonly_TTEX` | `36.5932` | `5.120` | `78.815` | `78.750` | `57.753` | `3.596` | `31.219899` |
| `A1_SWAactive_SWOVR` | `36.5915` | `5.118` | `78.809` | `78.755` | `57.755` | `3.593` | `31.213845` |
| `A1_freeze456` | `60.6379` | `21.773` | `29.029` | `25.886` | `49.279` | `5.864` | `33.116085` |
| `A1_semantic_noop` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `3.765` | `31.238916` |

Chunk-level audit 表：

```text
results/kitti01_hmc_v2/acl2_v6_phase1_200_300_audit/phase1_audit_tables/per_chunk_memory.csv
results/kitti01_hmc_v2/acl2_v6_phase1_200_300_audit/phase1_audit_tables/per_chunk_cue.csv
results/kitti01_hmc_v2/acl2_v6_phase1_200_300_audit/phase1_audit_tables/per_chunk_write.csv
results/kitti01_hmc_v2/acl2_v6_phase1_200_300_audit/phase1_audit_tables/key_segments_200_300.csv
results/kitti01_hmc_v2/acl2_v6_phase1_200_300_audit/phase1_audit_tables/key_chunks_6_10.csv
```

关键 chunk 读数（B0 focus）：

| Chunk | Frame range | overlap with `[200,300)` | RMSE | B0 `D>0.5` mass | B0 TTT mean rel diff | 结论 |
|---:|---|---:|---:|---:|---:|---|
| 6 | `[174,206)` | `0.1875` | `53.799` | `0.1838` | `0.0270` | 病灶入口 |
| 7 | `[203,235)` | `1.0000` | `74.942` | `0.2082` | `0.0253` | high-error chunk |
| 8 | `[232,264)` | `1.0000` | `88.814` | `0.1954` | `0.0252` | worst chunk |
| 9 | `[261,293)` | `1.0000` | `76.800` | `0.1969` | `0.0248` | high-error chunk |
| 10 | `[290,322)` | `0.3125` | `37.576` | `0.1965` | `0.0345` | 病灶出口 |

### 5.1 Phase 1 结论

1. `B0_SWKS3` 与 `A1_semantic_noop` 完全一致，说明 Stage C/D pass-through no-op 在 full KITTI01 上仍然严格安全。
2. `[200,300)` 是稳定主病灶：B0、TTT-only、SWA-active、semantic no-op 的 worst 100f 都落在 `[200,300)`，ATE 约 `77.8-78.8m`。
3. `freeze chunks 4/5/6` 把 `[200,300)` 从 `77.831m` 大幅降到 `25.886m`，说明该病灶对 chunks 4/5/6 附近的 TTT/SWA state 有明确因果敏感性。
4. 但 freeze 同时让全局 ATE 从 `36.4161m` 崩到 `60.6379m`，FinalErr 从 `5.798m` 恶化到 `21.773m`，并把 worst segment 前移到 `[100,200)`。所以 freeze 只能作为 causal diagnostic，不能作为策略。
5. 机制判断：chunks 4/5/6 的 TTT 写入里确实包含会放大 `[200,300)` 病灶的成分，但这些 chunks 也提供后续全局尺度/轨迹连续性。下一步不能 hard freeze；应做更精细的 TTT 写入方向控制。

### 5.2 新 TTT 机制候选：gradient reversal / negative evidence replay

用户提出的新想法：

```text
对要压制的动态/低置信区域，不只是少写，而是做梯度反转。
```

当前理解：

- 之前的 soft prior / sparse / EMA 大多只是缩放 token learning-rate；
- TTT replay 后有 `zeropower_via_newtonschulz5` 和 fast-weight norm restoration，幅度信息会被部分折叠；
- 梯度反转更直接，它改变的是 harmful token 对 fast-weight update 的方向：

```text
G_commit = G_static - gamma * G_dynamic
```

建议新增实验族：

| Run | 机制 | Branch / layer | `gamma` | 目的 |
|---|---|---|---:|---|
| `ACL2V6_TTGR_01` | dynamic gradient reversal | `w0`, late layers | `0.05` | 最温和版本，避免 scale 崩 |
| `ACL2V6_TTGR_02` | dynamic gradient reversal | `w0`, late layers | `0.10` | 检查反向信号强度 |
| `ACL2V6_TTGR_03` | dynamic gradient reversal | `w0`, all layers | `0.05` | 对照 layer scope |
| `ACL2V6_TTGR_04` | dynamic gradient reversal | `w0+w2`, late layers | `0.05` | 测 hidden/gate 分支是否需要同步反向 |

安全边界：

- 第一批不碰 `w1`，因为 v5/v6 多次证明 `w1` value/output 分支对 scale/translation 极敏感；
- 不做 all-branch negative replay；
- `gamma` 从 `0.05/0.10` 起，不直接上强反转；
- 只在 Phase 1 记录后进入实现。

### 5.3 TTGR 工程实现与 smoke

已实现 `TTT gradient reversal / negative evidence replay`：

- `TTT_WRITE_GRADIENT_REVERSAL_MODE=low_prior|hard`；
- `TTT_WRITE_GRADIENT_REVERSAL_GAMMA`；
- `TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK`。

机制位置：

```text
A_tok -> normal branch token_prior -> eta mean-preserve
     -> selected branch signed token_prior
     -> fast_weight_replay_update / zeropower
```

也就是说，eta 仍然看正常 write prior；真正 replay 时，指定 branch 的低 prior / high-risk token 会变成小负 multiplier。`low_prior` 当前定义：

```text
risk   = (prior_max - prior) / (prior_max - prior_min)
signed = prior * (1 - risk) - gamma * risk
```

工程验证：

```text
python3 -m py_compile loger/pipeline/ttt_write_controller.py loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py
bash -n tools/run_attention_cue_experiment.sh
```

Short smoke：

| Run | END_FRAME | Layer | Branch | Gamma | Result |
|---|---:|---|---|---:|---|
| `ACL2V6_TTGR_SMOKE_w0late_g005_e128` | `128` | `late` | `w0` | `0.05` | 通过 |

smoke 路径：

```text
results/kitti01_hmc_v2/acl2_v6_ttt_gradient_reversal/ACL2V6_TTGR_SMOKE_w0late_g005_e128/
```

关键 debug：

| Field | Value |
|---|---:|
| `ttt_gradient_reversal_applied` | `True` |
| `ttt_gradient_reversal_active_branches` | `[0]` |
| `ttt_gradient_reversal_prior_min/max` | `0.875 / 1.125` |
| `ttt_gradient_reversal_risk_mean` | `0.5000` |
| `ttt_gradient_reversal_signed_mean` | `0.4957` |
| `ttt_gradient_reversal_signed_min` | `-0.0500` |
| `ttt_gradient_reversal_negative_mass` | `0.0530` |

smoke 只用于确认 hook 正确生效，不参与 full KITTI01 指标排名。

### 5.4 TTGR full KITTI01 小矩阵

固定协议：

```text
seq = KITTI01 full
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA = SWKS3-style: keep both_overlap + overlap source replace kv alpha 0.50
success target = KITTI01 ATE < 30m
```

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `ACL2V6_TTGR_01_w0late_g005_SWKS3` | `2026-05-07 20:45:05` | `21:09:24` | `24.3 min` |
| `ACL2V6_TTGR_02_w0late_g010_SWKS3` | `2026-05-07 20:45:05` | `21:08:15` | `23.2 min` |
| `ACL2V6_TTGR_03_w0all_g005_SWKS3` | `2026-05-07 20:45:05` | `21:07:56` | `22.9 min` |
| `ACL2V6_TTGR_04_w02late_g005_SWKS3` | `2026-05-07 20:45:05` | `21:08:28` | `23.4 min` |

Benchmark：

| Run | Branch / layer | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` reference | baseline | - | `36.4161` | `6.6128` | `92.4452` | `0.0082` | v6 reproduction baseline |
| `SWOVR_02` v5 tiny reference | source-V replacement | - | `36.5915` | `6.4307` | `92.4416` | `0.0078` | v5/v6 old tiny reference |
| `TTGR_01` | `w0`, late | `0.05` | `36.4901` | `6.1581` | `92.4403` | `0.0079` | Rot/Final/Yaw 强，但 ATE 不如 B0 |
| `TTGR_02` | `w0`, late | `0.10` | `36.5011` | `6.0961` | `92.4405` | `0.0079` | 更强 gamma 改 Rot，ATE 仍回退 |
| `TTGR_03` | `w0`, all | `0.05` | `36.4061` | `6.1780` | `92.4398` | `0.0079` | **v6 当前 ATE 新 best，但仍远离 <30m** |
| `TTGR_04` | `w0+w2`, late | `0.05` | `36.6753` | `6.0874` | `92.4428` | `0.0079` | Rot 最强之一，ATE 回退 |

Trajectory diagnostics：

输出目录：

```text
results/kitti01_hmc_v2/acl2_v6_ttt_gradient_reversal/trajectory_diagnostics_ttgr/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0_SWKS3` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `TTGR_01` | `36.4901` | `3.343` | `77.969` | `77.631` | `56.999` | `3.315` | `31.174857` |
| `TTGR_02` | `36.5011` | `2.236` | `77.806` | `77.553` | `56.929` | `3.224` | `31.183725` |
| `TTGR_03` | `36.4061` | `2.974` | `77.707` | `77.502` | `56.900` | `3.338` | `31.178672` |
| `TTGR_04` | `36.6753` | `2.322` | `77.728` | `77.053` | `56.663` | `3.195` | `31.200127` |

Gradient reversal debug 代表值：

| Run | Signed min | Signed mean | Negative mass | 说明 |
|---|---:|---:|---:|---|
| `TTGR_01/03` gamma `0.05` | `-0.0500` | `0.4957` | `0.0530` | 约 5.3% token 真正反向 |
| `TTGR_02` gamma `0.10` | `-0.1000` | `0.4707` | `0.0995` | 约 10.0% token 真正反向 |

TTGR 结论：

1. 梯度反转确实有机制信号：所有 TTGR 都显著改善 Rot / FinalErr / Yaw，且 `[200,300)` 主病灶从 `77.831m` 小幅降到 `77.50-77.63m`，`TTGR_04` 更降到 `77.053m`。
2. `TTGR_03 = w0 all, gamma 0.05` 是当前 v6 ATE best：`36.4061 / 6.1780`，比 `B0_SWKS3=36.4161` 好 `0.0100m`，比 v5 SWKS3 reference `36.4153` 也略好。
3. 但这个收益仍是厘米级，不是 `<30m` 所需的 `6m+` 跃迁。
4. late-only `w0` 更像姿态/endpoint regularizer；all-layer `w0` 才给 ATE 微弱正收益。
5. `w0+w2` late 让 Rot 和 `[200,300)` 最好，但 ATE 明显回退，说明 `w2` 仍容易牺牲全局 segment/scale。
6. 下一步如果继续 TTGR，不建议直接加大 gamma；优先做：
   - `w0 all gamma 0.025/0.075` 细查 ATE sweet spot；
   - `w0 all hard_low_prior` 小负样本比例控制；
   - 对 `[200,300)` chunks 做局部 TTGR diagnostic，而不是全局强反转。

---

## 6. 当前 v6 状态

已完成：

- `B0-SWKS3-reference-rerun` historical best reproduction。
- Stage C cache / semantic no-op / pass-through smoke。
- Full `B1/B2` no-op parity 失败诊断。
- 修复 `stage_d_x_*` 在 `hmc_ignore_semantic_prior=1` 下的 base prior 退化问题。
- 修复后的 full `B1G/B2G` hard no-op parity。
- Full `B3G/B4G` Stage D noop / pass-through parity。
- Phase 1 `[200,300)` causal audit 的 trajectory / chunk-level 初版。
- `TTGR` gradient reversal 写入机制实现与 short smoke。
- `TTGR_01-04` full KITTI01 小矩阵；当前 v6 ATE best 更新为 `TTGR_03 = 36.4061 / 6.1780`。

后续状态（第 7 节更新后）：

- Phase 2 passive semantic audit 已完成；cached Stage C/D ignored path 通过 no-op parity，但 semantic coverage gate 失败。
- TTGR follow-up 仍可作为 TTT 写入机制候选：围绕 `w0 all gamma=0.05` 做更温和/更精确的反转策略。
- Phase 3 semantic prior 写入实验暂停；除非重建 Stage C cache 并通过 semantic coverage gate，否则不进入 semantic TTT write matrix。

---

## 7. Phase 2：cached Stage C/D passive semantic audit

本节按计划执行 Stage C/D passive semantic audit：

```text
semantic_prior_mode = spg_v2
hmc_ignore_semantic_prior = 1
stage_c_cache_mode = read
stage_c_cache_require_hit = 1
stage_c_cache_validate = 1
stage_c_inline_when_ignored = 1
```

目的有两个：

1. 确认 cached Stage C/D + spg_v2 semantic prior 生成路径在 no-op / ignored 模式下不改变当前 B0/SWKS3 轨迹；
2. 审计 Stage C cache 的 masklet/semantic 覆盖是否足够进入 Phase 3 semantic TTT write matrix。

### 7.1 固定协议

```text
seq = KITTI01 full
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA = SWKS3-style: keep both_overlap + overlap source replace kv alpha 0.50
RESET_EVERY = 5
Stage C cache = results/kitti01_hmc_v2/acl2_v6_stage_c_cache_full_swks3/
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v6_phase2_passive_semantic/ACL2V6_P2_01_passive_semantic_cached_SWKS3/
```

运行记录：

| Run | Start | Done | Walltime | 备注 |
|---|---|---|---:|---|
| `ACL2V6_P2_01_passive_semantic_cached_SWKS3` | `2026-05-07 21:15:55` | `2026-05-07 21:40:16` | `24.4 min` | 38 chunks；cache read + spg_v2 prior 生成；HMC 忽略 semantic prior |

补充工具：

- 新增 `tools/semantic_cache_audit.py`，用于从 Stage C cache 生成 per-chunk semantic coverage CSV 和 summary。
- 审计输出：

```text
results/kitti01_hmc_v2/acl2_v6_phase2_passive_semantic/semantic_audit_tables/per_chunk_semantic.csv
results/kitti01_hmc_v2/acl2_v6_phase2_passive_semantic/semantic_audit_tables/key_chunks_200_300_semantic.csv
results/kitti01_hmc_v2/acl2_v6_phase2_passive_semantic/semantic_audit_tables/label_counts_by_chunk.csv
results/kitti01_hmc_v2/acl2_v6_phase2_passive_semantic/semantic_audit_tables/semantic_summary.md
```

### 7.2 No-op benchmark

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---|
| `B0_SWKS3` reference | `36.4161` | `6.6128` | `92.4452` | `0.0082` | v6 B0 reproduction baseline |
| `ACL2V6_P2_01` passive semantic | `36.4161` | `6.6128` | `92.4452` | `0.0082` | 与 B0 完全一致；cached Stage C/D ignored path 安全 |

说明：

- `hmc_state_hash.jsonl` 正常写出 `38` 个 chunk。
- 本 run 未生成 `prior_debug.jsonl`，但 `01.log` 中已打印每个 chunk 的 `semantic` prior debug；后续若需要 per-layer prior jsonl，需要单独补 debug dump hook。
- no-op trajectory 与 B0 完全一致，说明 Stage C cache read、spg_v2 prior 生成和 `hmc_ignore_semantic_prior=1` 没有污染 HMC commit / TTT / SWA state。

Trajectory diagnostics：

输出目录：

```text
results/kitti01_hmc_v2/acl2_v6_phase2_passive_semantic/trajectory_diagnostics/
```

| Run | ATE RMSE | FinalErr | 50f worst | 100f worst | 200f worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `P2_passive` | `36.4161` | `5.798` | `[200,250) = 78.2719` | `[200,300) = 77.8307` | `[200,400) = 57.1014` | `3.765` | `31.238916` |

### 7.3 Semantic cache coverage audit

Stage C cache 总览：

| Chunks | Chunks with masklets | Mean masklets / chunk | Mean coverage | Median coverage | `[200,300)` focus coverage | Max frame coverage | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| `38` | `1` | `0.03` | `0.0000` | `0.0000` | `0.0000` | `0.0029` | **FAIL** |

Semantic group mass：

| Group | Masklet count | Mean coverage | `[200,300)` focus coverage |
|---|---:|---:|---:|
| `STRUCTURE_ANCHOR` | `0` | `0.0000` | `0.0000` |
| `STATIC_THING` | `0` | `0.0000` | `0.0000` |
| `MOVABLE_THING` | `1` | `0.0000` | `0.0000` |
| `LOW_VALUE_STUFF` | `0` | `0.0000` | `0.0000` |
| `UNCERTAIN_REGION` | `0` | `0.0000` | `0.0000` |

`[200,300)` focus chunks：

| Chunk | Frame range | Focus overlap | Masklets | Coverage | Structure | Low stuff | Movable | Labels |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| `6` | `[174,206)` | `0.1875` | `0` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | none |
| `7` | `[203,235)` | `1.0000` | `0` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | none |
| `8` | `[232,264)` | `1.0000` | `0` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | none |
| `9` | `[261,293)` | `1.0000` | `0` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | none |
| `10` | `[290,322)` | `0.3125` | `0` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | none |

唯一非空 cache 出现在 chunk 12：

```text
chunk 12 = [348,380)
label = person
group = MOVABLE_THING
coverage_mean = 0.0002588
coverage_max = 0.0028739
```

### 7.4 Phase 2 结论

1. **No-op / ignored path 通过**：cached Stage C read + spg_v2 semantic prior 生成不会改变 SWKS3/B0 trajectory，`ACL2V6_P2_01 = 36.4161 / 6.6128`，与 B0 完全一致。
2. **Semantic coverage gate 失败**：38 个 chunk 中只有 1 个 chunk 有 1 条极小 `person` masklet；全局 mean coverage 约 `0`，`[200,300)` 病灶段 coverage 也是 `0`。
3. 当前 Stage C cache 不满足计划里的 Phase 2 标准：

```text
mask coverage >= 0.70 或至少 structure+lowstuff 覆盖充分
sky/tree/road/building/car/person 有合理分布
[200,300) semantic maps 可解释
```

4. 因此 **不启动 Phase 3 semantic TTT write 小矩阵**。如果在这个 cache 上直接跑 semantic write，实际只会退化成 geometry-only / almost-empty semantic prior，无法回答语义写入是否有效。
5. 当前计划在 Phase 2 gate 处停止。后续若要继续 semantic line，优先级不是调 TTT write，而是重新构建 Stage C frontend/cache，使 KITTI01 至少有 road/building/sky/tree/car/person 或 structure/lowstuff 的稳定覆盖。

### 7.5 当前 v6 状态更新

当前 v6 best 仍为：

```text
ACL2V6_TTGR_03_w0all_g005_SWKS3
ATE / Rot = 36.4061 / 6.1780
```

Phase 2 passive semantic audit 已完成，但 semantic-write gate 失败，按计划停止，不继续 Phase 3。

---

## 8. Stage C frontend prompt 修正与 masklet video 导出

动机：

- Phase 2 cache 几乎没有语义覆盖，审计发现旧 full cache 使用的是 indoor/default prompt：`stuff_prompts=floor,wall,ceiling`，结构 prompt 也是 `wall,floor,ceiling`。
- 这会导致 KITTI01 场景里 road/building/sky/tree 等语义基本无法被 video masklet frontend 覆盖。
- 用户要求在 pipeline v2 增加默认关闭的视频导出开关，用于直接查看 video masklet frontend 的输出。

### 8.1 工程改动

`run_pipeline_abc_v2.py`：

- 修正 KITTI 默认 prompt：
  - `thing_prompts`: `person,people,rider,bicycle,motorcycle,car,bus,truck,train,door,window,traffic sign,traffic light,pole,guardrail,barrier`
  - `stuff_prompts`: `road,sidewalk,building,wall,fence,railing,bridge,sky,tree,vegetation,grass,water,reflection,glass,cloud`
  - `sam31_structure_prompt_labels`: `road,sidewalk,building,wall,fence,bridge`
  - `sam31_text_track_labels` / `sam31_direct_text_prompt_labels`: `person,people,rider`
  - `sam31_nontext_object_prompt_budget=8`
  - `sam31_text_object_prompt_budget=2`
- 新增默认关闭的视频导出开关：
  - `--stage_c_save_video`
  - `--stage_c_video_path`
  - `--stage_c_video_fps`
  - `--stage_c_video_alpha`
- 开启 `--stage_c_save_video 1` 时，pipeline 会把每个 Stage C chunk 的 masklet overlay 写入同一个 mp4；非最后 chunk 会 trim tail overlap，避免重复帧。

`tools/run_attention_cue_experiment.sh`：

- 新增环境变量透传：
  - `STAGE_C_SAVE_VIDEO`
  - `STAGE_C_VIDEO_PATH`
  - `STAGE_C_VIDEO_FPS`
  - `STAGE_C_VIDEO_ALPHA`

验证：

```text
python3 -m py_compile run_pipeline_abc_v2.py
bash -n tools/run_attention_cue_experiment.sh
```

### 8.2 KITTI01 full video run

运行目的：只检查 Stage C video masklet frontend 输出；本 run 使用 `probe_only`，不作为 KITTI benchmark 指标。

输出：

```text
run id = ACL2V6_STAGEC_VIDEO_KITTI_PROMPTS_FULL
output dir = results/kitti01_hmc_v2/acl2_v6_stage_c_video/ACL2V6_STAGEC_VIDEO_KITTI_PROMPTS_FULL/
cache dir = results/kitti01_hmc_v2/acl2_v6_stage_c_cache_kitti_prompts_video_full/
video = results/kitti01_hmc_v2/acl2_v6_stage_c_video/ACL2V6_STAGEC_VIDEO_KITTI_PROMPTS_FULL/stage_c_masklets.mp4
```

运行记录：

```text
START = 2026-05-07 22:11:00
DONE  = 2026-05-07 22:47:05
```

视频校验：

| File | Size | Frames | FPS | Resolution |
|---|---:|---:|---:|---:|
| `stage_c_masklets.mp4` | `61M` | `1101` | `10.0` | `1240x376` |

Cache / prompt 校验：

| Item | Value |
|---|---:|
| cache chunks | `38` |
| `masklet.pt` files | `38` |
| chunks with semantic labels | `38` |
| total masklets | `99` |

Top labels：

| Label | Count |
|---|---:|
| `road` | `38` |
| `sky` | `33` |
| `tree` | `7` |
| `bridge` | `6` |
| `building` | `6` |
| `fence` | `5` |
| `grass` | `3` |
| `person` | `1` |

### 8.3 结论

1. `stuff_prompts` 已修正：新 cache 不再是旧的 `floor/wall/ceiling` indoor prompt，而是 KITTI road/sidewalk/building/sky/tree/vegetation 等 prompt。
2. Stage C video 导出开关已加入，默认关闭；开启后可以直接查看 frontend masklet overlay。
3. 新 full cache 的语义覆盖明显恢复：38 个 chunk 全部有 label，top label 以 `road/sky/tree/building/bridge` 为主，符合 KITTI 场景。
4. 视频已生成，可用于人工检查是否还有漏检/错分/覆盖过窄问题。

### 8.4 LSeg + SAM3.1 + YOLOE 三 chunk 预览

用户要求先停掉实验矩阵，重新校准 video masklet frontend，并用 `lseg + sam3/yoloe` 跑 KITTI01 预览。

本次使用的是 `run_video_masklet_front_end.py` 的 `VideoMaskletFrontend.from_config(...)` 路径，不是 `EfficientVideoMaskletFrontend`。其中 STUFF pass 通过 `stuff_backend=lseg` 接入 LSeg；脚本里 `efficientsam3_stuff_enable` 是历史命名，只表示启用 per-frame STUFF pass。

实现修正：

- `run_video_masklet_front_end.py`：修正 LSeg device 解析，`cuda` / `auto` 在单可见 GPU 时会归一到 `cuda:0`，避免 `torch.cuda.set_device(torch.device("cuda"))` 报错。

预览配置：

```text
input = /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2
frames = [0, 90)
chunk_size = 32
chunk_overlap = 3
sam_backend = sam31_multiplex
sam31_checkpoint = ckpts/SAM3/sam3.1_multiplex.pt
detector = yoloe
yoloe_model = yoloe-11l-seg.pt
stuff_backend = lseg
lseg_repo = /tmp/lsm_probe
lseg_checkpoint = ckpts/LSeg/demo_e200.ckpt
```

KITTI prompts：

```text
thing = person,people,rider,bicycle,motorcycle,car,bus,truck,train,door,window,traffic sign,traffic light,pole,guardrail,barrier
lseg stuff = road,sidewalk,building,wall,fence,railing,bridge,sky,tree,vegetation,grass,water,reflection,glass,cloud
lseg background = person,people,rider,bicycle,motorcycle,car,bus,truck,train,traffic sign,traffic light,pole,guardrail,barrier,other
sam31 structure = road,sidewalk,building,wall,fence,bridge
sam31 text/direct = person,people,rider
```

输出：

```text
dir = results/kitti01_hmc_v2/acl2_v6_stage_c_lseg_sam31_yoloe_preview/
video = results/kitti01_hmc_v2/acl2_v6_stage_c_lseg_sam31_yoloe_preview/kitti01_lseg_sam31_yoloe_3chunks_preview.mp4
pt = results/kitti01_hmc_v2/acl2_v6_stage_c_lseg_sam31_yoloe_preview/kitti01_lseg_sam31_yoloe_3chunks_preview.pt
log = results/kitti01_hmc_v2/acl2_v6_stage_c_lseg_sam31_yoloe_preview/run_retry_cuda0.log
```

视频校验：

| File | Size | Frames | FPS | Resolution |
|---|---:|---:|---:|---:|
| `kitti01_lseg_sam31_yoloe_3chunks_preview.mp4` | `9.1M` | `90` | `10.0` | `1240x376` |

最终 masklet summary：

| Item | Value |
|---|---:|
| total masklets | `8` |
| thing / structure / stuff | `1 / 2 / 5` |
| frame size | `376 x 1241` |
| Stage C runtime | `122.55s` |
| peak CUDA alloc/reserved | `9.65GiB / 10.15GiB` |

最终 labels：

| Label | Type | Visible |
|---|---|---:|
| `road` | `structure_tracked` | `74/90` |
| `fence` | `stuff_static` | `59/90` |
| `sky` | `stuff_static` | `90/90` |
| `tree` | `stuff_static` | `49/90` |
| `grass` | `stuff_static` | `29/90` |
| `guardrail` | `thing_tracked` | `14/90` |
| `bridge` | `stuff_static` | `29/90` |
| `bridge` | `structure_tracked` | `13/90` |

结论：

1. 这次预览确认 `VideoMaskletFrontend + SAM3.1 + YOLOE + LSeg` 路径可跑通。
2. LSeg STUFF 覆盖不再是空的，且输出包含 `road/sky/tree/grass/fence/bridge` 等 KITTI 场景标签。
3. 预览视频已生成，下一步应人工查看该 mp4，重点检查 road 是否过窄、tree/sky 边界是否错分、guardrail 是否误当 thing 传播。

### 8.5 Mask2Former Cityscapes Panoptic 三 chunk 预览

用户要求把语义源从 open-vocab prompt 换成 driving-scene panoptic model，先跑 KITTI01 前 3 个 chunk 预览。这里使用的是 `run_video_masklet_front_end.py` 的 `VideoMaskletFrontend.from_config(...)` 路径，不是 `EfficientVideoMaskletFrontend`。

本次新增/启用：

- `run_video_masklet_front_end.py` 新增 `stuff_backend=mask2former_cityscapes`。
- backend 使用 `facebook/mask2former-swin-large-cityscapes-panoptic`。
- Cityscapes `terrain` remap 成当前 pipeline 更常用的 `grass`。
- STUFF labels 固定为 driving-scene stuff：`road,sidewalk,building,wall,fence,sky,vegetation,terrain`。

预览配置：

```text
input = /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2
frames = [0, 90)
chunk_size = 32
chunk_overlap = 3
sam_backend = sam31_multiplex
sam31_checkpoint = ckpts/SAM3/sam3.1_multiplex.pt
detector = yoloe
yoloe_model = yoloe-11l-seg.pt
stuff_backend = mask2former_cityscapes
mask2former_model = facebook/mask2former-swin-large-cityscapes-panoptic
mask2former_labels = road,sidewalk,building,wall,fence,sky,vegetation,terrain
mask2former_label_map = terrain:grass
mask2former_confidence_threshold = 0.50
mask2former_min_area_ratio = 0.003
mask2former_max_area_ratio = 0.92
mask2former_morph_kernel = 3
mask2former_device = cuda:0
```

输出：

```text
dir = results/kitti01_hmc_v2/acl2_v6_stage_c_mask2former_cityscapes_preview/
video = results/kitti01_hmc_v2/acl2_v6_stage_c_mask2former_cityscapes_preview/kitti01_mask2former_cityscapes_3chunks_preview.mp4
pt = results/kitti01_hmc_v2/acl2_v6_stage_c_mask2former_cityscapes_preview/kitti01_mask2former_cityscapes_3chunks_preview.pt
log = results/kitti01_hmc_v2/acl2_v6_stage_c_mask2former_cityscapes_preview/run.log
```

视频校验：

| File | Size | Frames | FPS | Resolution |
|---|---:|---:|---:|---:|
| `kitti01_mask2former_cityscapes_3chunks_preview.mp4` | `7.4M` | `90` | `10.0` | `1240x376` |

运行资源：

| Item | Value |
|---|---:|
| Stage C runtime | `107.51s` |
| Throughput | `0.837 fps` |
| peak CUDA alloc/reserved | `9.26GiB / 9.74GiB` |
| output tensor size | `30M` |

最终 masklet summary：

| Item | Value |
|---|---:|
| total masklets | `9` |
| thing / structure / stuff | `1 / 2 / 6` |
| frame size | `376 x 1241` |

最终 labels：

| Label | Type | Visible | MeanQ | MeanArea |
|---|---|---:|---:|---:|
| `road` | `structure_tracked` | `74/90` | `1.000` | `0.21754` |
| `building` | `stuff_static` | `1/90` | `0.650` | `0.01294` |
| `fence` | `stuff_static` | `60/90` | `0.830` | `0.07081` |
| `grass` | `stuff_static` | `90/90` | `0.998` | `0.12695` |
| `road` | `stuff_static` | `90/90` | `1.000` | `0.24751` |
| `sky` | `stuff_static` | `90/90` | `1.000` | `0.09597` |
| `vegetation` | `stuff_static` | `90/90` | `1.000` | `0.46439` |
| `guardrail` | `thing_tracked` | `14/90` | `0.839` | `0.05960` |
| `bridge` | `structure_tracked` | `13/90` | `1.000` | `0.06487` |

结论：

1. `Mask2Former Cityscapes Panoptic` backend 已跑通，并成功替代 open-vocab stuff prompt 作为 driving-scene semantic source。
2. 相比 8.4 的 LSeg 预览，Cityscapes panoptic stuff 覆盖明显更稳定：`road/sky/vegetation/grass` 都达到 `90/90` 帧可见，`fence` 也达到 `60/90`。
3. 这个结果能解释之前 Stage C cache 语义覆盖弱的问题：open-vocab prompt / LSeg 对 KITTI driving stuff 的稳定性不够，而 Cityscapes panoptic 直接给出道路场景类别，覆盖更符合需求。
4. 仍需人工看视频确认：`road` 同时出现 `structure_tracked` 与 `stuff_static`，需要观察是否是合理重叠或后处理 dedup 还不够；`vegetation` 面积很大，也要检查是否吞掉路边结构。
5. 若人工预览确认边界和类别合理，下一步可以用 `mask2former_cityscapes` 重新生成 KITTI01 full Stage C cache，再继续 v6 semantic prior / TTT write 实验。

### 8.6 Mask2Former Cityscapes Panoptic KITTI01 full cache / video

用户确认三 chunk 预览效果可用后，按同一配置跑完整 KITTI01，并保存 full-sequence video masklet frontend 输出，供人工检查和后续 cache 构建使用。

固定配置：

```text
input = /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2
frames = full KITTI01, 1101 frames
chunk_size = 32
chunk_overlap = 3
sam_backend = sam31_multiplex
sam31_checkpoint = ckpts/SAM3/sam3.1_multiplex.pt
detector = yoloe
yoloe_model = yoloe-11l-seg.pt
stuff_backend = mask2former_cityscapes
mask2former_model = facebook/mask2former-swin-large-cityscapes-panoptic
mask2former_labels = road,sidewalk,building,wall,fence,sky,vegetation,terrain
mask2former_label_map = terrain:grass
mask2former_confidence_threshold = 0.50
mask2former_min_area_ratio = 0.003
mask2former_max_area_ratio = 0.92
mask2former_morph_kernel = 3
mask2former_device = cuda:0
```

输出：

```text
dir = results/kitti01_hmc_v2/acl2_v6_stage_c_mask2former_cityscapes_full/
video = results/kitti01_hmc_v2/acl2_v6_stage_c_mask2former_cityscapes_full/kitti01_mask2former_cityscapes_full.mp4
pt = results/kitti01_hmc_v2/acl2_v6_stage_c_mask2former_cityscapes_full/kitti01_mask2former_cityscapes_full.pt
log = results/kitti01_hmc_v2/acl2_v6_stage_c_mask2former_cityscapes_full/run.log
```

视频 / cache 校验：

| File | Size | Frames | FPS | Resolution |
|---|---:|---:|---:|---:|
| `kitti01_mask2former_cityscapes_full.mp4` | `74M` | `1101` | `10.0` | `1240x376` |
| `kitti01_mask2former_cityscapes_full.pt` | `294M` | n/a | n/a | sparse masklet tensor |

运行资源：

| Item | Value |
|---|---:|
| Stage C runtime | `411.21s` |
| Throughput | `2.677 fps` |
| peak CUDA alloc/reserved | `7.14GiB / 7.52GiB` |
| output tensor size | `294M` |

最终 masklet summary：

| Item | Value |
|---|---:|
| total masklets | `17` |
| thing / structure / stuff | `0 / 0 / 17` |
| frame size | `376 x 1241` |
| storage | sparse |

主要 label 覆盖：

| Label | Type | Visible | MeanQ | MeanArea |
|---|---|---:|---:|---:|
| `road` | `stuff_static` | `1101/1101` | `1.000` | `0.29194` |
| `sky` | `stuff_static` | `1101/1101` | `1.000` | `0.29990` |
| `vegetation` | `stuff_static` | `1101/1101` | `1.000` | `0.26914` |
| `grass` | `stuff_static` | `794/1101` | `0.931` | `0.09522` |
| `building` | `stuff_static` | `540/1101` | `0.988` | `0.04257` |
| `fence` | `stuff_static` | `480/1101` | `0.859` | `0.05974` |
| `wall` | `stuff_static` | `96/1101` | `0.777` | `0.07209` |
| `sidewalk` | `stuff_static` | `23/1101` | n/a | n/a |
| `bridge` | `stuff_static` | `1/1101` | n/a | n/a |

说明：

1. full KITTI01 video masklet frontend 输出已完成，覆盖比旧 Stage C cache 明显更合理，`road/sky/vegetation` 都达到全序列覆盖。
2. 当前 `.pt` 是 `run_video_masklet_front_end.py` 的 full-sequence frontend 输出，不是 `run_pipeline_abc_v2.py --stage_c_cache_dir` 直接 require-hit 的 per-chunk `masklet.pt` cache 目录。
3. 因此它已经可用于人工检查 full KITTI01 semantic masklet 质量；若要恢复 v6 Phase 3 cached semantic prior / TTT write 实验，下一步还需要用同一 Mask2Former Cityscapes 配置生成 pipeline-compatible per-chunk Stage C cache，或补转换脚本。
4. 计划进度更新：v6 reproducibility / no-op / passive audit 已完成；Phase 3 暂停的 blocker 是旧 Stage C cache 语义覆盖不足。本 full video/cache 解决了 frontend 质量验证的一半，下一步是把该配置接到 pipeline cache 生成路径，并重新跑 semantic coverage audit。

### 8.7 Mask2Former full sparse output 转 pipeline Stage C cache

为了继续执行 v6 plan，把 8.6 生成的 full-sequence sparse frontend 输出转换成 `run_pipeline_abc_v2.py` 可 `require-hit` 读取的 per-chunk Stage C cache。

转换工具：

```text
tools/convert_sparse_masklet_to_stage_c_cache.py
```

输入 / 输出：

```text
input_pt = results/kitti01_hmc_v2/acl2_v6_stage_c_mask2former_cityscapes_full/kitti01_mask2former_cityscapes_full.pt
cache_dir = results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/
chunk_size = 32
chunk_overlap = 3
```

转换结果：

| Item | Value |
|---|---:|
| chunks | `38` |
| masklet.pt files | `38` |
| cache size | `3.5G` |
| output schema | `masklet_v1` per chunk |
| conversion log | `results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/conversion.log` |
| cache index | `results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/cache_index.jsonl` |
| summary | `results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/conversion_summary.json` |

Semantic cache audit：

```text
audit_dir = results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/semantic_audit/
summary = results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/semantic_audit/semantic_summary.md
focus segment = [200,300)
```

覆盖率：

| Chunks | Chunks With Masklets | Mean Masklets / Chunk | Mean Coverage | Median Coverage | Focus Coverage | Max Frame Coverage | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| `38` | `38` | `6.63` | `0.9831` | `0.9870` | `0.9782` | `0.9998` | `PASS` |

Semantic group mass：

| Group | Masklet Count | Mean Coverage | Focus Coverage |
|---|---:|---:|---:|
| `STRUCTURE_ANCHOR` | `139` | `0.3459` | `0.3548` |
| `LOW_VALUE_STUFF` | `113` | `0.6375` | `0.6235` |
| `STATIC_THING` | `0` | `0.0000` | `0.0000` |
| `MOVABLE_THING` | `0` | `0.0000` | `0.0000` |
| `UNCERTAIN_REGION` | `0` | `0.0000` | `0.0000` |

Focus chunks：

| Chunk | Frame Range | Focus Overlap | Masklets | Coverage | Structure | Low Stuff | Labels |
|---:|---|---:|---:|---:|---:|---:|---|
| `6` | `[174,206)` | `0.1875` | `6` | `0.9421` | `0.3199` | `0.6224` | `road;fence;grass;sky;vegetation;wall` |
| `7` | `[203,235)` | `1.0000` | `7` | `0.9593` | `0.4103` | `0.5491` | `road;building;fence;sky;vegetation;wall;sidewalk` |
| `8` | `[232,264)` | `1.0000` | `6` | `0.9853` | `0.3561` | `0.6293` | `road;building;fence;grass;sky;vegetation` |
| `9` | `[261,293)` | `1.0000` | `7` | `0.9922` | `0.3205` | `0.6719` | `road;building;fence;grass;sky;vegetation;wall` |
| `10` | `[290,322)` | `0.3125` | `7` | `0.9921` | `0.3034` | `0.6888` | `road;building;fence;grass;sky;vegetation;wall` |

结论：

1. Mask2Former Cityscapes full sparse frontend 输出已经成功转成 pipeline v2 可读的 per-chunk Stage C cache。
2. 新 cache 的语义覆盖 gate 通过，解决了旧 Stage C cache “几乎没有语义覆盖”的 blocker。
3. 该 cache 目前以 stuff / structure 覆盖为主，`MOVABLE_THING=0`；这适合先做 `road/building/fence/sky/vegetation` 这类 driving-scene semantic prior，但暂时不能验证 movable thing 动态写入策略。
4. 下一步执行 v6 plan 的 Phase 2 pass-through：用 `STAGE_C_CACHE_REQUIRE_HIT=1` 读取该 cache，`SEMANTIC_PRIOR_MODE=spg_v2`，但 `HMC_IGNORE_SEMANTIC_PRIOR=1`，确认 Stage C/D 缓存回接不会破坏当前 SWKS3 / TTEX 主线轨迹。

## 9. Mask2Former cache 回接：Phase 2 no-op 与 Phase 3 semantic write 首批

本节使用 8.7 转出的 Mask2Former Cityscapes per-chunk Stage C cache：

```text
Stage C cache = results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/
cache mode = read
require hit = 1
validate = 1
semantic prior mode = spg_v2
```

固定主线协议：

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = frame pair/all
beta = 4.75
commit = probe_ttt_write
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
RESET_EVERY = 5
SWA = SWKS3-style
    ENABLE_SWA_WRITE_CONTROL = 1
    SWA_WRITE_KEEP_SCOPE = both_overlap
    SWA_WRITE_LAYER_MODE = last
    ENABLE_SWA_OVERLAP_SOURCE_REPLACE = 1
    mode = source
    target = kv
    alpha = 0.50
    layer_mode = last
```

说明：第一轮曾误跑了一个 passive run 和三条 Phase 3 run，缺少 `ENABLE_SWA_WRITE_CONTROL=1` 与 `SWA_WRITE_LAYER_MODE=last`，因此不进入主线比较。误跑的 passive diagnostic 为 `36.7625 / 6.3378`，已废弃。下表只记录修正后带完整 SWKS3-style SWA 设置的结果。

### 9.1 Phase 2 fixed passive no-op

运行：

```text
ACL2V6_P2M_02_passive_semantic_mask2former_SWKS3_fixed
HMC_IGNORE_SEMANTIC_PRIOR = 1
STAGE_C_INLINE_WHEN_IGNORED = 1
write score = stage_d_x_dg_inv_sqrt
```

结果：

| Run | Semantic 状态 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` reference | Stage C/D off | `36.416102` | `6.612796` | `92.445197` | `0.008169` | v6 reproduction baseline |
| `P2M_02` | Mask2Former cache read + spg_v2 generated, HMC ignored | `36.416102` | `6.612796` | `92.445197` | `0.008169` | 与 B0 完全一致；cache 回接 no-op 通过 |

Semantic cache audit：

```text
results/kitti01_hmc_v2/acl2_v6_phase2_mask2former_passive/semantic_audit_mask2former_p2m02/semantic_summary.md
```

| Chunks | Chunks With Masklets | Mean Masklets / Chunk | Mean Coverage | Median Coverage | Focus Coverage `[200,300)` | Gate |
|---:|---:|---:|---:|---:|---:|---|
| `38` | `38` | `6.63` | `0.9831` | `0.9870` | `0.9782` | `PASS` |

Semantic group mass：

| Group | Masklet Count | Mean Coverage | Focus Coverage |
|---|---:|---:|---:|
| `STRUCTURE_ANCHOR` | `139` | `0.3459` | `0.3548` |
| `LOW_VALUE_STUFF` | `113` | `0.6375` | `0.6235` |
| `STATIC_THING` | `0` | `0.0000` | `0.0000` |
| `MOVABLE_THING` | `0` | `0.0000` | `0.0000` |

结论：Mask2Former cache 回接已经通过 Phase 2 no-op / coverage gate。它解决了旧 Stage C cache 覆盖不足的问题，但该 Cityscapes panoptic cache 当前主要是 structure + low-value stuff，没有 movable/static thing 覆盖，因此 Phase 3 首批只能验证 driving stuff/structure prior。

### 9.2 Phase 3 fixed semantic write 首批

运行：

| Run | Semantic prior | Write score | 额外设置 |
|---|---|---|---|
| `ACL2V6_P3M_01b_semantic_prior_SWKS3_fixed` | consumed | `stage_d` | 直接使用 `spg_v2 A_tok` |
| `ACL2V6_P3M_02b_semantic_x_dg_SWKS3_fixed` | consumed | `stage_d_x_dg_inv_sqrt` | 试图把 semantic base 与 Dg write score 组合 |
| `ACL2V6_P3M_03b_lowstuff_medium_xdg_SWKS3_fixed` | consumed | `stage_d_x_dg_inv_sqrt` | `RHO_SEM=0.80`, `SPG_VALUE_DISTRACTOR=0.25` |

Global metrics：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | vs B0 ATE | vs `TTGR_03` ATE | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | `36.416102` | `6.612796` | `92.445197` | `0.008169` | reference | `+0.0100` | baseline |
| `TTGR_03` | `36.4061` | `6.1780` | `92.4398` | `0.0079` | `-0.0100` | reference | 当前 v6 ATE best |
| `P2M_02` | `36.416102` | `6.612796` | `92.445197` | `0.008169` | `0.0000` | `+0.0100` | no-op pass |
| `P3M_01b` | `36.473878` | `6.614096` | `92.444732` | `0.008159` | `+0.0578` | `+0.0678` | direct semantic prior 回退 |
| `P3M_02b` | `36.416102` | `6.612796` | `92.445197` | `0.008169` | `0.0000` | `+0.0100` | 与 B0 完全重合；当前 override 基本抹掉 semantic effect |
| `P3M_03b` | `36.416102` | `6.612796` | `92.445197` | `0.008169` | `0.0000` | `+0.0100` | 与 B0 完全重合；lowstuff stronger 设置未生效到 trajectory |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v6_phase3_semantic_write_mask2former/20_mask2former_phase2_phase3_diagnostics/
```

| Run | ATE RMSE | FinalErr | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `3.765` | `31.238916` |
| `TTGR_03` | `36.4061` | `2.974` | `29.770 / 77.707` | `30.404 / 77.502` | `30.486 / 56.900` | `3.338` | `31.178672` |
| `P2M_02` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `3.765` | `31.238916` |
| `P3M_01b` | `36.4739` | `5.800` | `29.770 / 78.342` | `30.420 / 77.872` | `30.431 / 57.133` | `3.772` | `31.233916` |
| `P3M_02b` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `3.765` | `31.238916` |
| `P3M_03b` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `3.765` | `31.238916` |

Debug 观察：

| Run | `hmc_write_score_source` | `hmc_write_override` | `prior_ttt_write_mean` | 结论 |
|---|---|---|---:|---|
| `P3M_01b` | `stage_d` | `False` | `1.0000` | direct `spg_v2` 进入 TTT write，但默认 low-stuff 抑制让 ATE 回退 |
| `P3M_02b` | `stage_d_x_dg_inv_sqrt` | `True` | `1.0000` | override 重建 v5 Dg write prior，semantic 影响被基本擦掉 |
| `P3M_03b` | `stage_d_x_dg_inv_sqrt` | `True` | `1.0000` | 同上，因此调 `RHO_SEM / LOWSTUFF` 不改变轨迹 |

### 9.3 首批结论与当前运行

1. Mask2Former Cityscapes cache 回接本身已经安全：`P2M_02` 与 `B0_SWKS3` 完全一致，且 semantic coverage gate 通过。
2. 默认 `spg_v2` direct semantic prior 没有带来收益：`P3M_01b = 36.4739 / 6.6141`，比 B0 回退 `0.0578m`，比当前 v6 best `TTGR_03 = 36.4061 / 6.1780` 差 `0.0678m`。
3. `stage_d_x_dg_inv_sqrt` 当前作为 override 使用时，会把 semantic base 基本擦掉，因此 `P3M_02b/P3M_03b` 与 B0 完全重合。这说明后续如果要做 “semantic x Dg”，需要新增明确的 semantic-aware write score，而不能复用旧 v5 override。
4. 由于 Mask2Former cache 的 `LOW_VALUE_STUFF` 覆盖占比很大，direct semantic 默认 `SPG_VALUE_DISTRACTOR=0.4` 可能过度抑制 sky/vegetation/horizon continuity。因此补了 4 条 low-stuff / semantic-strength 对照。

### 9.4 Phase 3 direct semantic low-stuff 强度对照

运行记录：

- `ACL2V6_P3M_04_lowstuff_soft070_SWKS3_fixed`：`2026-05-08 01:26:09 -> 01:52:54`
- `ACL2V6_P3M_05_lowstuff_mid055_SWKS3_fixed`：`2026-05-08 01:26:09 -> 01:52:18`
- `ACL2V6_P3M_06_rhosem030_default_SWKS3_fixed`：`2026-05-08 01:26:09 -> 01:51:50`
- `ACL2V6_P3M_07_rhosem000_geometry_only_SWKS3_fixed`：`2026-05-08 01:27:51 -> 01:54:15`

固定：

```text
HMC_IGNORE_SEMANTIC_PRIOR = 0
SEMANTIC_PRIOR_MODE = spg_v2
write score = stage_d
SWA = same SWKS3-style fixed protocol
```

Global metrics：

| Run | Semantic setting | ATE RMSE | Rot RMSE | RPE t | RPE r | vs B0 ATE | vs `TTGR_03` ATE | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `P3M_01b` | default: `rho=0.60`, lowstuff=`0.40` | `36.473878` | `6.614096` | `92.444732` | `0.008159` | `+0.0578` | `+0.0678` | direct semantic baseline，回退 |
| `P3M_04` | lowstuff=`0.70` | `36.473878` | `6.614096` | `92.444732` | `0.008159` | `+0.0578` | `+0.0678` | 与 P3M_01b 完全一致 |
| `P3M_05` | lowstuff=`0.55` | `36.473878` | `6.614096` | `92.444732` | `0.008159` | `+0.0578` | `+0.0678` | 与 P3M_01b 完全一致 |
| `P3M_06` | `rho_sem=0.30`, lowstuff=`0.40` | `36.473878` | `6.614096` | `92.444732` | `0.008159` | `+0.0578` | `+0.0678` | 与 P3M_01b 完全一致 |
| `P3M_07` | `rho_sem=0.00`, geometry-only Stage D | `36.473878` | `6.614096` | `92.444732` | `0.008159` | `+0.0578` | `+0.0678` | 与 P3M_01b 完全一致 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v6_phase3_semantic_write_mask2former/20_31_semantic_lowstuff_diagnostics/
```

| Run | ATE RMSE | FinalErr | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `3.765` | `31.238916` |
| `TTGR_03` | `36.4061` | `2.974` | `29.770 / 77.707` | `30.404 / 77.502` | `30.486 / 56.900` | `3.338` | `31.178672` |
| `P3M_01b` | `36.4739` | `5.800` | `29.770 / 78.342` | `30.420 / 77.872` | `30.431 / 57.133` | `3.772` | `31.233916` |
| `P3M_04` | `36.4739` | `5.800` | `29.770 / 78.342` | `30.420 / 77.872` | `30.431 / 57.133` | `3.772` | `31.233916` |
| `P3M_05` | `36.4739` | `5.800` | `29.770 / 78.342` | `30.420 / 77.872` | `30.431 / 57.133` | `3.772` | `31.233916` |
| `P3M_06` | `36.4739` | `5.800` | `29.770 / 78.342` | `30.420 / 77.872` | `30.431 / 57.133` | `3.772` | `31.233916` |
| `P3M_07` | `36.4739` | `5.800` | `29.770 / 78.342` | `30.420 / 77.872` | `30.431 / 57.133` | `3.772` | `31.233916` |

Debug 汇总：

| Run | `rho_sem` | `SPG_VALUE_DISTRACTOR` | `prior_ttt_write_mean` | `hmc_write_override` | 观察 |
|---|---:|---:|---:|---|---|
| `P3M_04` | `0.60` | `0.70` | `1.0000` | `False` | direct Stage D consumed，但 trajectory 与 P3M_01b 完全一致 |
| `P3M_05` | `0.60` | `0.55` | `1.0000` | `False` | 同上 |
| `P3M_06` | `0.30` | `0.40` | `1.0000` | `False` | 同上 |
| `P3M_07` | `0.00` | `0.40` | `1.0000` | `False` | geometry-only 与 semantic value variants 完全一致 |

9.4 结论：

1. `P3M_04/05/06/07` 全部没有达到 `KITTI01 ATE < 30m`，也没有超过 `B0` 或 `TTGR_03`。
2. lowstuff value 从 `0.40 -> 0.55 -> 0.70`、`rho_sem` 从 `0.60 -> 0.30 -> 0.00` 都不改变 trajectory，说明当前 direct Stage D write 的有效扰动主要来自 **geometry eligibility / chunk-scale path**，不是 semantic label value。
3. 因此，继续调 `SPG_VALUE_DISTRACTOR` 或 `RHO_SEM` 没有意义；当前 direct `spg_v2 A_tok` 不适合作 TTT write 主线。
4. 如果要继续语义写入，需要先做工程修正：把 `A_sem/A_semgeo` 或 semantic group map 作为明确的 HMC write score source 暴露出来，例如 `semantic_value`、`stage_d_x_sem`、`stage_d_x_sem_x_dg`、`lowstuff_veto`、`structure_boost`，并记录 per-group write mass。复用 `stage_d_x_dg_inv_sqrt` 旧 override 会擦掉 semantic effect。
5. 当前 v6 best 仍是 `TTGR_03 = 36.4061 / 6.1780`，Mask2Former semantic line 暂时只完成了 cache/no-op/audit gate，没有产生主指标收益。

### 9.5 Phase 3 semantic-aware HMC write source

9.4 之后做了工程修正：不再让旧 `stage_d_x_dg_inv_sqrt` override 擦掉 semantic prior，而是把 Mask2Former cache 产生的 patch-level semantic value 显式传入 HMC write source。

代码变化：

- `semantic_prior_generator.py`：`PriorOutput` 新增 `V_sem_patch_flat` / `R_mask_patch_flat`，把每个 patch 当前命中的 masklet semantic value 和 mask trust 显式导出。
- `hybrid_memory_controller.py`：`_phase_e_write_prior` 新增 `semantic_value_patch` 输入，并支持新的 write score：
  - `semantic_value` / `sem_value`
  - `sem_x_dg_inv_sqrt`
  - `stage_d_x_sem`
  - `stage_d_x_sem_x_dg_inv_sqrt`
- `run_pipeline_abc_v2.py`：CLI choices 和 debug jsonl 加入 semantic write 相关字段。

工程验证：

```text
python -m py_compile loger/pipeline/semantic_prior_generator.py loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py
ACL2V6_P3S_SMOKE2_semwrite_e128
```

说明：`SMOKE2` 在前 4 个完整 cache chunk 上通过 semantic write 路径；最后失败是因为 `END_FRAME=128` 产生了 `[116,128)` partial chunk，而 full cache 里对应的是 `[116,148)`。这不是语义路径错误。后续 full run 使用完整 KITTI01 chunk schedule，与 cache 对齐。

固定协议：

```text
seq = KITTI01 full
Stage C cache = results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/
read cue = C23 past_only
read = frame pair/all
beta = 4.75
commit = probe_ttt_write
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA = SWKS3-style fixed protocol
SEMANTIC_PRIOR_MODE = spg_v2
HMC_IGNORE_SEMANTIC_PRIOR = 0
```

运行记录：

- `ACL2V6_P3S_01_semvalue_SWKS3_fixed`：`2026-05-08 02:07:44 -> 02:32:21`
- `ACL2V6_P3S_02_stageDsem_SWKS3_fixed`：`2026-05-08 02:07:44 -> 02:32:39`
- `ACL2V6_P3S_03_stageDsemDg_SWKS3_fixed`：`2026-05-08 02:07:44 -> 02:32:53`

Global metrics：

| Run | Write score | ATE RMSE | Rot RMSE | RPE t | RPE r | vs B0 ATE | vs `TTGR_03` ATE | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | `stage_d_x_dg_inv_sqrt` | `36.416102` | `6.612796` | `92.445197` | `0.008169` | reference | `+0.0100` | v6 reproduction baseline |
| `TTGR_03` | gradient reversal | `36.4061` | `6.1780` | `92.4398` | `0.0079` | `-0.0100` | reference | 当前 v6 ATE best |
| `P3M_01b` | direct `spg_v2 A_tok` | `36.473878` | `6.614096` | `92.444732` | `0.008159` | `+0.0578` | `+0.0678` | 旧 direct semantic baseline |
| `P3S_01` | `semantic_value` | `36.4807` | `6.7457` | `92.4431` | `0.0083` | `+0.0646` | `+0.0746` | semantic value 明确生效，但 ATE/Rot 回退 |
| `P3S_02` | `stage_d_x_sem` | `36.5809` | `6.6017` | `92.4436` | `0.0081` | `+0.1648` | `+0.1748` | stage_d 与 semantic 相乘更差 |
| `P3S_03` | `stage_d_x_sem_x_dg_inv_sqrt` | `36.4869` | `6.6364` | `92.4444` | `0.0082` | `+0.0708` | `+0.0808` | 加 Dg 后仍未过 direct semantic |

Semantic write debug 均值：

| Run | Mean write score | Corr(score, semantic value) | Mean semantic value | q10 / q90 semantic value | Corr(score, dyn) | Corr(score, explicit dyn) |
|---|---:|---:|---:|---:|---:|---:|
| `P3S_01` | `0.6176` | `1.0000` | `0.6176` | `0.4000 / 1.0000` | `-0.6094` | `-0.5843` |
| `P3S_02` | `0.3554` | `0.8703` | `0.6176` | `0.4000 / 1.0000` | `-0.8566` | `-0.7811` |
| `P3S_03` | `0.3007` | `0.7823` | `0.6176` | `0.4000 / 1.0000` | `-0.7670` | `-0.6935` |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v6_phase3_semantic_write_mask2former/20_32_semantic_write_sources_diagnostics/
```

| Run | ATE RMSE | FinalErr | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `3.765` | `31.238916` |
| `TTGR_03` | `36.4061` | `2.974` | `29.770 / 77.707` | `30.404 / 77.502` | `30.486 / 56.900` | `3.338` | `31.178672` |
| `P3M_01b` | `36.4739` | `5.800` | `29.770 / 78.342` | `30.420 / 77.872` | `30.431 / 57.133` | `3.772` | `31.233916` |
| `P3S_01` | `36.4807` | `6.689` | `29.872 / 78.265` | `30.520 / 77.780` | `30.595 / 57.080` | `3.906` | `31.211744` |
| `P3S_02` | `36.5809` | `5.291` | `29.940 / 78.441` | `30.576 / 77.961` | `30.650 / 57.207` | `3.746` | `31.228230` |
| `P3S_03` | `36.4869` | `5.722` | `29.890 / 78.311` | `30.529 / 77.915` | `30.603 / 57.169` | `3.792` | `31.233005` |

9.5 结论：

1. semantic-aware HMC write source 已经真正接入：`P3S_01` 的 `Corr(score, semantic value)=1.0`，不再是旧 override 擦掉 semantic 的 no-op。
2. 但三条 semantic-aware write 都没有达到 `KITTI01 ATE < 30m`，也没有超过 `B0` / `TTGR_03`。当前 v6 best 仍是 `TTGR_03 = 36.4061 / 6.1780`。
3. `semantic_value` 单独使用时，ATE=`36.4807`，FinalErr/Yaw/Rot 都比 reference 差，说明当前 `structure=1.0 / lowstuff=0.4` 的 scalar value map 不是更好的 TTT 长期写入资格。
4. `stage_d_x_sem` 过度压低 write score，ATE 回退最大；`stage_d_x_sem_x_dg_inv_sqrt` 虽然比 `stage_d_x_sem` 好，但仍不如 direct semantic，也不如 B0。
5. 这批支持一个更具体的判断：Mask2Former Cityscapes cache 的语义覆盖已经可用，但用单个 patch-level semantic value 去控制 TTT write 太粗。后续若继续语义线，应转向 group-specific soft veto / structure boost / trust routing，而不是继续直接调 `semantic_value` 乘法。

### 9.6 Phase 3 semantic-aware lowstuff softening

9.5 的默认 semantic value 是：

```text
STRUCTURE_ANCHOR = 1.00
LOW_VALUE_STUFF = 0.40
```

考虑到 KITTI01 Mask2Former cache 主要由 road/building/fence 与 sky/vegetation/grass 构成，默认 `0.40` 可能把 horizon / vegetation continuity 压得太重。因此继续按 H3 测 low-value stuff softening。

固定协议同 9.5，只改 `SPG_VALUE_DISTRACTOR` 和 write score：

| Run | Write score | `SPG_VALUE_DISTRACTOR` | 目的 |
|---|---|---:|---|
| `ACL2V6_P3S_04_stageDsemDg_lowstuff070_SWKS3_fixed` | `stage_d_x_sem_x_dg_inv_sqrt` | `0.70` | lowstuff soft veto |
| `ACL2V6_P3S_05_stageDsemDg_lowstuff055_SWKS3_fixed` | `stage_d_x_sem_x_dg_inv_sqrt` | `0.55` | lowstuff medium veto |
| `ACL2V6_P3S_06_semvalue_lowstuff070_SWKS3_fixed` | `semantic_value` | `0.70` | direct semantic value 的 soft lowstuff 对照 |

运行记录：

- `P3S_04`：`2026-05-08 02:36:09 -> 03:01:18`
- `P3S_05`：`2026-05-08 02:36:09 -> 03:01:15`
- `P3S_06`：`2026-05-08 02:36:10 -> 03:01:23`

Global metrics：

| Run | Write score | Lowstuff value | ATE RMSE | Rot RMSE | RPE t | RPE r | vs B0 ATE | vs `TTGR_03` ATE | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | `stage_d_x_dg_inv_sqrt` | n/a | `36.416102` | `6.612796` | `92.445197` | `0.008169` | reference | `+0.0100` | baseline |
| `TTGR_03` | gradient reversal | n/a | `36.4061` | `6.1780` | `92.4398` | `0.0079` | `-0.0100` | reference | 当前 v6 best |
| `P3S_03` | `stage_d_x_sem_x_dg_inv_sqrt` | `0.40` | `36.4869` | `6.6364` | `92.4444` | `0.0082` | `+0.0708` | `+0.0808` | 默认 lowstuff 偏硬 |
| `P3S_04` | `stage_d_x_sem_x_dg_inv_sqrt` | `0.70` | `36.4459` | `6.6502` | `92.4451` | `0.0082` | `+0.0298` | `+0.0398` | 本批最好；softening 有信号但未过 gate |
| `P3S_05` | `stage_d_x_sem_x_dg_inv_sqrt` | `0.55` | `36.5251` | `6.6087` | `92.4434` | `0.0082` | `+0.1090` | `+0.1190` | medium 不如 soft |
| `P3S_06` | `semantic_value` | `0.70` | `36.5093` | `6.6818` | `92.4431` | `0.0082` | `+0.0932` | `+0.1032` | direct semantic soft 仍回退 |

Semantic write debug 均值：

| Run | Mean write score | Corr(score, semantic value) | Mean semantic value | q10 / q90 semantic value | Corr(score, dyn) | Corr(score, explicit dyn) |
|---|---:|---:|---:|---:|---:|---:|
| `P3S_04` | `0.3606` | `0.6639` | `0.8087` | `0.7000 / 1.0000` | `-0.8084` | `-0.7364` |
| `P3S_05` | `0.3306` | `0.7284` | `0.7131` | `0.5500 / 1.0000` | `-0.7931` | `-0.7197` |
| `P3S_06` | `0.8087` | `1.0000` | `0.8087` | `0.7000 / 1.0000` | `-0.6092` | `-0.5841` |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v6_phase3_semantic_write_mask2former/20_33_semantic_lowstuff_soft_diagnostics/
```

| Run | ATE RMSE | FinalErr | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `3.765` | `31.238916` |
| `TTGR_03` | `36.4061` | `2.974` | `29.770 / 77.707` | `30.404 / 77.502` | `30.486 / 56.900` | `3.338` | `31.178672` |
| `P3S_03` | `36.4869` | `5.722` | `29.890 / 78.311` | `30.529 / 77.915` | `30.603 / 57.169` | `3.792` | `31.233005` |
| `P3S_04` | `36.4460` | `5.946` | `29.820 / 78.255` | `30.460 / 77.838` | `30.530 / 57.111` | `3.805` | `31.234918` |
| `P3S_05` | `36.5251` | `5.619` | `29.937 / 78.434` | `30.575 / 77.986` | `30.660 / 57.229` | `3.779` | `31.219617` |
| `P3S_06` | `36.5093` | `6.462` | `29.957 / 78.319` | `30.594 / 77.915` | `30.668 / 57.180` | `3.841` | `31.218284` |

9.6 结论：

1. lowstuff softening 有弱信号：`P3S_04` 从默认 `P3S_03=36.4869` 改到 `36.4459`，说明默认把 sky/vegetation/grass 压到 `0.40` 可能过硬。
2. 但 `P3S_04` 仍没有超过 `B0=36.4161` 或 `TTGR_03=36.4061`，也没有接近最终目标 `KITTI01 ATE < 30m`。
3. `lowstuff=0.55` 反而比 `0.70` 差，direct `semantic_value lowstuff=0.70` 也差，说明 semantic value 的主要问题不是单纯 lowstuff 数值，而是它缺少更细的 group / layer / branch routing。
4. `[200,300)` worst segment 基本没被修掉：`P3S_04` 的 100f worst `77.838m` 只接近 B0，不如 `TTGR_03=77.502m`。
5. 当前判断：H3 的 “low-value stuff 不能 hard veto” 得到支持，但 lowstuff soft veto 不是主线突破。下一步如果继续语义线，应实现 structure-specific boost / trust-routing debug，或者回到 TTGR 机制与 semantic prior 组合，而不是继续扫 `SPG_VALUE_DISTRACTOR`。

### 9.7 Phase 3 semantic-aware write + TTGR combo

9.5/9.6 说明 semantic prior 单独写入没有超过 `B0` 或 `TTGR_03`，但 `lowstuff=0.70` 的 soft semantic value 有弱信号。因此本批把最好的 semantic-aware write 与当前最强 TTT gradient reversal 组合：

```text
seq = KITTI01 full
cue = C23 past_only
read = frame pair/all
beta = 4.75
SWA = SWKS3-style fixed protocol
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.05
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
```

候选：

| Run | Write score | Lowstuff value | TTGR | 目的 |
|---|---|---:|---|---|
| `ACL2V6_P3G_01_TTGR_stageDsemDg_lowstuff070_SWKS3` | `stage_d_x_sem_x_dg_inv_sqrt` | `0.70` | `w0 gamma=0.05` | best semantic write 与 TTGR 组合 |
| `ACL2V6_P3G_02_TTGR_stageDsemDg_lowstuff040_SWKS3` | `stage_d_x_sem_x_dg_inv_sqrt` | `0.40` | `w0 gamma=0.05` | 默认 lowstuff 对照 |
| `ACL2V6_P3G_03_TTGR_semvalue_lowstuff070_SWKS3` | `semantic_value` | `0.70` | `w0 gamma=0.05` | direct semantic value 与 TTGR 对照 |

Global metrics：

| Run | Write score | Lowstuff value | ATE RMSE | Rot RMSE | RPE t | RPE r | vs `TTGR_03` ATE | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | `stage_d_x_dg_inv_sqrt` | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | `+0.0100` | reproduction baseline |
| `TTGR_03` | gradient reversal | n/a | `36.4061` | `6.1780` | `92.4398` | `0.0079` | reference | previous v6 best |
| `P3S_04` | `stage_d_x_sem_x_dg_inv_sqrt` | `0.70` | `36.4459` | `6.6502` | `92.4451` | `0.0082` | `+0.0398` | best semantic-only write |
| `P3G_01` | `stage_d_x_sem_x_dg_inv_sqrt` | `0.70` | `36.4017` | `6.2289` | `92.4412` | `0.0079` | `-0.0044` | **当前 v6 ATE tiny best** |
| `P3G_02` | `stage_d_x_sem_x_dg_inv_sqrt` | `0.40` | `36.4988` | `6.2341` | `92.4404` | `0.0079` | `+0.0927` | 默认 lowstuff 太硬 |
| `P3G_03` | `semantic_value` | `0.70` | `36.6392` | `6.5265` | `92.4385` | `0.0081` | `+0.2331` | direct semantic value 不能直接当主 write |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v6_phase3_semantic_write_mask2former/20_34_semantic_ttgr_combo_diagnostics/
```

| Run | ATE RMSE | FinalErr | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `3.765` | `31.238916` |
| `TTGR_03` | `36.4061` | `2.974` | `29.770 / 77.707` | `30.404 / 77.502` | `30.486 / 56.900` | `3.338` | `31.178672` |
| `P3S_04` | `36.4460` | `5.946` | `29.820 / 78.255` | `30.460 / 77.838` | `30.530 / 57.111` | `3.805` | `31.234918` |
| `P3G_01` | `36.4017` | `3.509` | `29.790 / 77.856` | `30.429 / 77.568` | `30.504 / 56.942` | `3.390` | `31.184830` |
| `P3G_02` | `36.4988` | `3.625` | `29.902 / 78.100` | `30.545 / 77.807` | `30.644 / 57.104` | `3.380` | `31.184109` |
| `P3G_03` | `36.6392` | `6.314` | `30.083 / 78.454` | `30.711 / 78.016` | `30.900 / 57.241` | `3.679` | `31.151633` |

9.7 结论：

1. 当前 semantic write 相关的最好做法是 `P3G_01`：`stage_d * V_sem * sqrt(1-D_g)`，其中 `LOW_VALUE_STUFF=0.70`，再叠加 `w0` low-prior gradient reversal `gamma=0.05`。
2. `P3G_01 = 36.4017 / 6.2289` 成为 v6 当前 ATE tiny best，比 `TTGR_03=36.4061` 只好 `0.0044m`。这是平台内的小改进，不是 `<30m` 级突破。
3. 语义本身不是主信号：best semantic-only `P3S_04=36.4459` 仍差于 `B0/TTGR_03`。语义目前更像 TTGR 的轻量辅助权重，而不是独立替代 `stage_d/D_g`。
4. `lowstuff=0.70` 明显优于默认 `0.40`：`P3G_01` 过 `TTGR_03`，而 `P3G_02=36.4988` 大幅回退。这说明 sky/vegetation/grass 这类低价值 stuff 不能 hard veto，仍提供 horizon / scale / continuity 背景。
5. `semantic_value` 直接当 write score 失败：`P3G_03=36.6392`。当前 cache 主要是 structure + low-value stuff，没有 movable/static thing 覆盖；因此 semantic prior 只能做 soft modulation，不能直接决定 TTT 写入。
6. 下一步若继续语义线，不应继续扫单个 `LOW_VALUE_STUFF` 标量，而应做 branch/layer/group routing：例如 structure anchor boost 只作用 `w0` late layer，lowstuff 只做轻 veto，并结合 TTGR 的 low-prior 反向梯度。

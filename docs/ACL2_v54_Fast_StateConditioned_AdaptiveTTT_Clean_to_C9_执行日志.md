# ACL2 v54 Fast State-Conditioned Adaptive TTT Clean-to-C9 执行日志

日期: 2026-06-09  
计划文档: `docs/ACL2_v54_Fast_StateConditioned_AdaptiveTTT_Clean_to_C9_Plan.md`  
结果复盘: `docs/ACL2_v54_Fast_StateConditioned_AdaptiveTTT_Clean_to_C9_实验结果复盘.md`  
结果根目录: `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9`

## 执行原则

- 不使用 absolute chunk-id policy。
- 不使用手工 tri-replay positive/negative/neutral 百分比。
- 不编造缺失指标；缺失字段保留为 `NA` 或说明 artifact 缺失。
- 704F 只作为 promotion filter；未过 704F gate 不启动 full。
- 若遇到 blocker，优先按计划修复方向尝试：接线/审计、runtime、state variable/action-space，而不是绕过 gate。

## 2026-06-09 实现记录

### 代码修改

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `loger/pipeline/ttt_write_controller.py` | 新增 v54 state-energy role split：`safety=norm(p)*(1-risk)`，`danger=risk*(2-norm(p))`，阈值为 `median + k*MAD`，无百分比 fallback | 实现 M1/M2 的无手工百分比 role assignment |
| `loger/pipeline/ttt_write_controller.py` | 新增 causal state-energy gamma：`gamma_eff = gamma0 * g_energy * (1+g_risk)`；`E_t` 使用 per-layer/per-branch EMA，首次 bootstrap 后只用历史 EMA | 实现 M1，避免 chunk map |
| `loger/pipeline/ttt_write_controller.py` | 新增 `state_energy_directional_commit`：按 candidate/native 方向余弦和能量比计算 alpha，再混合 native/candidate commit | 实现 M2 directional commit guard |
| `run_pipeline_abc_v2.py` | 新增 `state_energy_directional_commit` / alias 到 CLI choices | 让 M2 launcher 能通过参数解析 |
| `tools/run_v54_state_conditioned_candidate.sh` | 新增 v54 wrapper，固定调用 v47 no-chunk runner，默认 risk source `ttt_residual_x_dg`，layer gamma `0/8/17:0.0075` | 复现实验命令，避免手写大量 env |
| `tools/v54_experiment_report.py` | 新增 v54 registry/report/failure routing/phase1 autopsy/code audit 生成脚本 | 统一统计 gate 和审计信息 |

### 静态检查

```bash
chmod +x tools/run_v54_state_conditioned_candidate.sh tools/v54_experiment_report.py
bash -n tools/run_v54_state_conditioned_candidate.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  loger/pipeline/ttt_write_controller.py \
  run_pipeline_abc_v2.py \
  tools/v54_experiment_report.py
```

结果:

- `bash -n`: pass
- `py_compile`: pass

备注:

- 直接执行 `tools/v54_experiment_report.py` 会走系统 python，缺 `matplotlib`。后续报告统一使用 `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`。

## Phase 0/1 初始报告

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v54_experiment_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9 \
  --out-dir results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial
```

输出:

- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/v54_code_audit.md`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/v54_phase1_teacher_student_autopsy.csv`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/v54_phase1_teacher_student_autopsy.md`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/teacher_student_post_zp_delta_timeline.png`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/teacher_student_layer_branch_heatmap.png`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/v54_704_report.md`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/v54_full_report.md`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/v54_failure_routing.md`

初始状态:

- v54 run rows = 0。
- H35 704F reference 已读取: `V53_PHASE7_SCREEN_H35_LAYERGAMMAFIX_RHO0075_704F`。
- H35 full reference 已读取: `V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075`。

## Phase 3 704F 运行计划

M1:

```bash
tools/run_v54_state_conditioned_candidate.sh \
  0 M1_704F V54_PHASE3_M1_STATE_ENERGY_MATCHED_704F
```

M2:

```bash
tools/run_v54_state_conditioned_candidate.sh \
  1 M2_704F V54_PHASE3_M2_STATE_ENERGY_DIRECTIONAL_COMMIT_704F
```

固定配置:

- `END_FRAME=704`
- `role_mode`: M1 = `adaptive_writer_state_energy_matched_split`；M2 = `adaptive_writer_state_energy_directional_commit_split`
- `risk_source=ttt_residual_x_dg`
- `layer_gammas=0:0.0075,8:0.0075,17:0.0075`
- `commit_filter_mode`: M1 = `none`；M2 = `state_energy_directional_commit`
- `stage_c_mode=none`
- `empty_cuda_cache_each_chunk=0`

## Phase 3 704F 实际执行

M1 command:

```bash
mkdir -p logs && set -o pipefail
tools/run_v54_state_conditioned_candidate.sh \
  0 M1_704F V54_PHASE3_M1_STATE_ENERGY_MATCHED_704F \
  2>&1 | tee logs/stream_v54_phase3_m1_state_energy_matched_704f.log
```

M2 command:

```bash
mkdir -p logs && set -o pipefail
tools/run_v54_state_conditioned_candidate.sh \
  1 M2_704F V54_PHASE3_M2_STATE_ENERGY_DIRECTIONAL_COMMIT_704F \
  2>&1 | tee logs/stream_v54_phase3_m2_state_energy_directional_commit_704f.log
```

完成时间:

- M1: `2026-06-09 15:09:26` DONE
- M2: `2026-06-09 15:11:13` DONE

关键输出目录:

- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/phase3_704_screen/rollouts/V54_PHASE3_M1_STATE_ENERGY_MATCHED_704F`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/phase3_704_screen/rollouts/V54_PHASE3_M2_STATE_ENERGY_DIRECTIONAL_COMMIT_704F`

接线观察:

- M1 `01.log` 中出现 `ttt_tri_replay_state_energy_gamma_mean`，证明 state-energy gamma 进入 controller。
- M2 `01.log` 中出现 `ttt_write_commit_filter_scale_mean`，证明 directional commit guard 进入 controller。
- 两个 run 的 `chunk_id_policy_audit.json` 与 report summary 均显示 no-chunk pass。
- 两个 run 的 manual percentage audit pass；role collapse rows 均为 0。

## Phase 3 report_final

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v54_experiment_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9 \
  --out-dir results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_final
```

输出:

- `report_final/v54_704_registry.csv`
- `report_final/v54_704_report.md`
- `report_final/v54_full_registry.csv`
- `report_final/v54_full_report.md`
- `report_final/v54_failure_routing.md`
- `report_final/v54_code_audit.md`
- `report_final/v54_state_energy_gamma_timeline.csv`
- `report_final/v54_commit_alpha_timeline.csv`
- `report_final/v54_state_energy_gamma_timeline.png`
- `report_final/teacher_student_commit_delta_timeline.png`

报告脚本修复:

- 初次 `report_final` 后发现 `v54_state_energy_gamma_timeline.csv` / `v54_commit_alpha_timeline.csv` 为空。
- 原因: state-energy/commit debug 在 `01.log` 中可见，但 `hmc_state_hash.jsonl` 未保留完整 layer debug。
- 修改 `tools/v54_experiment_report.py`: timeline extraction 同时解析 `hmc_state_hash.jsonl` 和 `01.log`。
- 重新 `py_compile` pass 后重跑 report，timeline CSV/PNG 已生成。
- 该修复只改变报告提取，不改变实验结果。

## 704F Gate 结果

来源: `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_final/v54_704_report.md`

| run | ATE704 | H35 delta | seg2 RMSE | rolling100 p90 | projected full min | chunk mean | TTT mean | no chunk | manual % | collapse | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| M1 state-energy | 40.095253 | +0.297006 | 30.466112 | 65.204727 | 25.764000 | 31.874800 | NA | True | True | 0 | False |
| M2 directional commit | 41.383880 | +1.585632 | 30.614087 | 67.420535 | 28.474667 | 34.943200 | NA | True | True | 0 | False |

Gate failure reason:

- M1: `ATE704_not_0.20m_better_than_H35`, `seg2_or_rolling100_not_0.50m_better_than_H35`
- M2: `projected_runtime_gt_28min`, `ATE704_not_0.20m_better_than_H35`, `seg2_or_rolling100_not_0.50m_better_than_H35`

Decision:

- M1/M2 均未过 704F promotion gate。
- 按计划未启动 full。
- `report_final/v54_full_report.md` 明确记录: `704F gate did not pass; full run was not allowed.`

备注:

- `probe_ttt_write_seconds_mean` 在 v54 registry 中为 `NA`，因为本次 run 未写出 `timing_summary.json`，report fallback 从 `01.log` 解析到 `pass2_control_seconds_mean`，但无法从 log 中无歧义恢复 `probe_ttt_write_seconds_mean`。未填充/替代该字段。

## 测试记录

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  loger/pipeline/ttt_write_controller.py \
  run_pipeline_abc_v2.py \
  tools/v54_experiment_report.py
```

结果: pass。

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果: fail，但失败来自 conda 环境中的 `site-packages/tests/*`，不是本 repo 测试；错误为 `ModuleNotFoundError: No module named 'pytest'`。

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s ./tests
```

结果: fail，`./tests` 不存在，`Start directory is not importable`。

本 repo 内未找到 LoGeR 根目录下的本地 `tests/`；`rg --files` 只发现 Stream3D/third_party 相关测试，和本轮 ACL2/HMC 修改无直接对应。

# ACL2 v56 H35 SemanticBoost NewTTTAction Fast 执行日志

日期: 2026-06-09
计划文档: `docs/ACL2_v56_H35_SemanticBoost_NewTTTAction_FastPlan.md`
结果复盘: `docs/ACL2_v56_H35_SemanticBoost_NewTTTAction_Fast_实验结果复盘.md`
工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR`
结果根目录: `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast`

## 执行边界

- 所有指标只来自落盘 artifact、`01.log`、`hmc_state_hash.jsonl`、`wall_time_summary.json` 或 evaluation 输出。
- 不使用 absolute chunk-id policy，不使用手工 tri replay percentage。
- Track A 使用 Stage C cache 时强制 `stage_c_cache_mode=read` 和 `stage_c_cache_require_hit=1`。
- 单条 full run runtime gate 为 wall time <= 28min；`probe_ttt_write_seconds_mean` 缺失时保持 unavailable。

## 代码与工具修改

| 文件 | 修改内容 |
|---|---|
| `loger/pipeline/ttt_write_controller.py` | 新增 v56 binary stable-anchor / risk-veto role modes，使用当前 chunk Otsu/median fallback 阈值，不使用 top percentage；记录 stable/risk/no-long-write mass。 |
| `tools/run_v56_h35_semantic_ttt_action_candidate.sh` | 新增 v56 统一 runner，覆盖 H35 repeat、Track A A1-A4、Track B B1-B4 和可选 combo。 |
| `tools/v56_experiment_report.py` | 新增 artifact-only 报告工具，生成 registry、failure report、diagnostic figures、执行日志和复盘。 |

验证命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile run_pipeline_abc_v2.py loger/pipeline/ttt_write_controller.py tools/v56_experiment_report.py
bash -n tools/run_v56_h35_semantic_ttt_action_candidate.sh
```

## 运行命令清单

| row | run | GPU | frames | status | wall min | ATE | 输出目录 |
|---|---|---:|---:|---|---:|---:|---|
| `H35_FULL_REPEAT` | `V56_PHASE0_H35_FULL_REPEAT` | 0 | 1101 | done | 25.283333 | 35.740897 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase0_h35_repeat/rollouts/V56_PHASE0_H35_FULL_REPEAT` |
| `A1_704F` | `V56_A1_SEM_C23_RESID_704F` | 1 | 704 | done | 18.150000 | 40.077571 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase1_track_a_704_screen/rollouts/V56_A1_SEM_C23_RESID_704F` |
| `A2_704F` | `V56_A2_HIGH_INFL_ANOM_READ_704F` | 2 | 704 | done | 17.800000 | 39.798248 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase1_track_a_704_screen/rollouts/V56_A2_HIGH_INFL_ANOM_READ_704F` |
| `A3_704F` | `V56_A3_ANOM_STATIC_RESCUE_704F` | 3 | 704 | done | 18.216667 | 39.798248 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase1_track_a_704_screen/rollouts/V56_A3_ANOM_STATIC_RESCUE_704F` |
| `A4_704F` | `V56_A4_SEM_C23_PLUS_ANOM_704F` | 4 | 704 | done | 17.816667 | 40.077571 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase1_track_a_704_screen/rollouts/V56_A4_SEM_C23_PLUS_ANOM_704F` |
| `A2_FULL` | `V56_A2_HIGH_INFL_ANOM_READ_FULL` | 1 | 1101 | done | 29.616667 | 35.740897 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase1_track_a_full/rollouts/V56_A2_HIGH_INFL_ANOM_READ_FULL` |
| `A3_FULL` | `V56_A3_ANOM_STATIC_RESCUE_FULL` | 2 | 1101 | done | 28.216667 | 35.740897 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase1_track_a_full/rollouts/V56_A3_ANOM_STATIC_RESCUE_FULL` |
| `A1_96F` | `V56_A1_SEM_C23_RESID_FIX1_96F` | 1 | 96 | done | 3.150000 | 1.334476 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase1_track_a_smoke/rollouts/V56_A1_SEM_C23_RESID_FIX1_96F` |
| `A2_96F` | `V56_A2_HIGH_INFL_ANOM_READ_FIX1_96F` | 2 | 96 | done | 2.900000 | 1.329108 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase1_track_a_smoke/rollouts/V56_A2_HIGH_INFL_ANOM_READ_FIX1_96F` |
| `A3_96F` | `V56_A3_ANOM_STATIC_RESCUE_FIX1_96F` | 3 | 96 | done | 3.066667 | 1.329108 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase1_track_a_smoke/rollouts/V56_A3_ANOM_STATIC_RESCUE_FIX1_96F` |
| `A4_96F` | `V56_A4_SEM_C23_PLUS_ANOM_FIX1_96F` | 4 | 96 | done | 3.000000 | 1.334476 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase1_track_a_smoke/rollouts/V56_A4_SEM_C23_PLUS_ANOM_FIX1_96F` |
| `B1_704F` | `V56_B1_BINARY_STABLE_ANCHOR_704F` | 0 | 704 | done | 18.283333 | 41.439231 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase2_track_b_704_screen/rollouts/V56_B1_BINARY_STABLE_ANCHOR_704F` |
| `B2_704F` | `V56_B2_RISK_VETO_COMMIT_704F` | 3 | 704 | done | 17.066667 | 41.727226 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase2_track_b_704_screen/rollouts/V56_B2_RISK_VETO_COMMIT_704F` |
| `B3_704F` | `V56_B3_TWO_LIFETIME_COMMIT_704F` | 4 | 704 | done | 16.833333 | 40.867985 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase2_track_b_704_screen/rollouts/V56_B3_TWO_LIFETIME_COMMIT_704F` |
| `B4_704F` | `V56_B4_PROJECTION_COMMIT_704F` | 5 | 704 | done | 17.033333 | 41.233692 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase2_track_b_704_screen/rollouts/V56_B4_PROJECTION_COMMIT_704F` |
| `B1_96F` | `V56_B1_BINARY_STABLE_ANCHOR_96F` | 0 | 96 | done | 2.683333 | 1.222306 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase2_track_b_smoke/rollouts/V56_B1_BINARY_STABLE_ANCHOR_96F` |
| `B2_96F` | `V56_B2_RISK_VETO_COMMIT_96F` | 3 | 96 | done | 2.666667 | 1.267612 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase2_track_b_smoke/rollouts/V56_B2_RISK_VETO_COMMIT_96F` |
| `B3_96F` | `V56_B3_TWO_LIFETIME_COMMIT_96F` | 4 | 96 | done | 2.733333 | 1.288025 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase2_track_b_smoke/rollouts/V56_B3_TWO_LIFETIME_COMMIT_96F` |
| `B4_96F` | `V56_B4_PROJECTION_COMMIT_FIX1_96F` | 5 | 96 | done | 2.616667 | 1.297223 | `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase2_track_b_smoke/rollouts/V56_B4_PROJECTION_COMMIT_FIX1_96F` |

复现单条 run 的模板:

```bash
tools/run_v56_h35_semantic_ttt_action_candidate.sh <GPU> <ROW> <RUN_NAME>
```

每个 run 目录都写有 `effective_config.yaml`、`adaptive_ttt_audit.json`、`chunk_id_policy_audit.json` 和 `reproduce_command.sh`。

## 报告生成

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v56_experiment_report.py
```

## Blocker 与修复记录

### Stage C cache short-tail miss

首次运行 `A1_96F/A2_96F/A3_96F/A4_96F` 使用 run 名 `V56_A*_96F`，全部失败。失败原因一致:

```text
Required Stage C cache miss for chunk 3:
results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/chunk_003_000087_000096/masklet.pt
```

审计发现 cache 中存在完整 chunk:

```text
results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/chunk_003_000087_000119/masklet.pt
```

修复: 修改 `run_pipeline_abc_v2.py`，新增 short-tail cache superset read。仅在 `stage_c_cache_validate=0` 时允许读取同 `chunk_idx/start` 且 `cached_end >= end` 的完整 cache，并用 `_slice_masklet_output()` 裁剪到当前短尾长度。没有关闭 `stage_c_cache_require_hit`，没有 inline Stage C fallback。报告工具 `tools/v56_experiment_report.py` 也补充了 exact/superset/missing cache artifact 检查。

修复后重跑:

```bash
tools/run_v56_h35_semantic_ttt_action_candidate.sh 1 A1_96F V56_A1_SEM_C23_RESID_FIX1_96F
tools/run_v56_h35_semantic_ttt_action_candidate.sh 2 A2_96F V56_A2_HIGH_INFL_ANOM_READ_FIX1_96F
tools/run_v56_h35_semantic_ttt_action_candidate.sh 3 A3_96F V56_A3_ANOM_STATIC_RESCUE_FIX1_96F
tools/run_v56_h35_semantic_ttt_action_candidate.sh 4 A4_96F V56_A4_SEM_C23_PLUS_ANOM_FIX1_96F
```

修复结果: 四条 FIX1 smoke 全部 `DONE`，Stage C cache artifact 检查均为 `hit_count=4/seen=4`、`superset_hit_count=1`、`missing_count=0`。

### B4 argparse enum blocker

首次运行:

```bash
tools/run_v56_h35_semantic_ttt_action_candidate.sh 5 B4_96F V56_B4_PROJECTION_COMMIT_96F
```

失败原因:

```text
argument --ttt_write_native_delta_gate_mode: invalid choice: 'orthogonal_suppress'
```

审计发现 `loger/pipeline/ttt_write_controller.py` 已实现 `orthogonal_suppress`，但 `run_pipeline_abc_v2.py` argparse choices 漏掉该枚举。

修复: 修改 `run_pipeline_abc_v2.py`，将 `orthogonal_suppress` 加入 `--ttt_write_native_delta_gate_mode` choices。未把 B4 改写为其他近似模式。

修复后重跑:

```bash
tools/run_v56_h35_semantic_ttt_action_candidate.sh 5 B4_96F V56_B4_PROJECTION_COMMIT_FIX1_96F
```

修复结果: B4 FIX1 96F `DONE`，后续 B4 704F 也 `DONE`。

### 报告工具修复

`tools/v56_experiment_report.py` 在执行中做过三处审计向修复:

- 覆盖 v53 旧 runtime gate，v56 使用 `wall_time_min<=28`、`chunk_total_seconds_mean<=42`、HMC rows/frame completeness；`probe_ttt_write_seconds_mean` 缺失保持 NA，不作为失败也不补造。
- 增加 Stage C cache artifact 检查，记录 exact/superset/missing hit counts。
- 从 `01.log` 的 commit debug dict 解析 B action role mass、threshold、native gate、dual-lifetime 字段，因为这些细粒度字段没有完整写入 `hmc_state_hash.jsonl`。

## Gate 决策

- Phase 0 H35 repeat 通过: ATE drift `0.0`，wall `25.283333min`，chunk mean `31.551579s`。
- Track A 704F: A1/A4 回退，停止；A2/A3 与 H35 704F 持平，按 borderline 规则各跑一条 full。
- Track A full: A2/A3 full ATE 都等于 H35 full，未达到 `<=35.2409` minimum progress，且 runtime 分别 `29.616667min` / `28.216667min`，均未通过 28min full runtime gate。
- Track B 704F: B1-B4 全部相对 H35 704F 回退，全部停止，不跑 Track B full。
- Combo 未运行: 计划要求 Track A 和 Track B 都有 full ATE `<=35.2409` 才组合；实际 Track A 无改善，Track B 无 full。
- Cross-sequence 未运行: 没有任何 KITTI01 full 达到 `<=35.2409`。

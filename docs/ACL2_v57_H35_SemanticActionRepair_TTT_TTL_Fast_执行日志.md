# ACL2 v57 H35 SemanticActionRepair TTT TTL Fast 执行日志

日期: 2026-06-09
计划文档: `docs/ACL2_v57_H35_SemanticActionRepair_TTT_TTL_FastPlan.md`
结果复盘: `docs/ACL2_v57_H35_SemanticActionRepair_TTT_TTL_Fast_实验结果复盘.md`
工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR`
结果根目录: `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast`
报告目录: `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/report_final`

## 执行边界

- 所有数据来自 landed artifacts、`01.log`、`hmc_state_hash.jsonl`、`wall_time_summary.json` 或 pose evaluation。
- 缺失字段保持 NA/unavailable；不把计划值、默认值或推测写成观测值。
- 语义路线必须先通过 action-realization smoke；若 S0 失败，不进入 semantic 704F/full。
- TTT 路线必须先完成 v56 behavioral audit，再跑 96F smoke 和 gate 允许的 704F/full。
- 不使用 absolute chunk-id policy，不使用 hand-specified positive/neutral/negative percentage。

## 代码与工具修改

| 文件 | 修改内容 | 审计理由 |
|---|---|---|
| `run_pipeline_abc_v2.py` | 将 context source skip 与 semantic role 相关 CLI 参数传入 `HybridMemoryController`，并把 v57 semantic prior/action debug 字段写入 trace。 | v56 A2/A3 参数到 HMC 的链路断开，导致 action inactive；修复后需要可审计地证明 source token 与 attention mass 被改变。 |
| `loger/pipeline/hybrid_memory_controller.py` | 增加 v57 semantic source-skip role policies，保留 group-level source role，避免 fine label 缺失时被 fallback 擦掉。 | 支持 S0/S1/S2 action-realization repair。 |
| `tools/run_v57_h35_semantic_action_repair_ttt_ttl_fast.sh` | 新增 v57 统一 runner，覆盖 semantic smoke/read、TTT smoke/704/full 和 combo。 | 让每条 run 带有 `effective_config.yaml`、audit JSON 与 reproduce command。 |
| `tools/v57_experiment_report.py` | 新增 artifact-only reporter，生成 Phase0 audit、registries、figures、执行日志和复盘；semantic action realization 接受 fine-label 或 group-level evidence。 | 避免手工摘数和虚构缺失指标，同时避免 SREAD02 这类只有 group evidence 的 active action 被误判 inactive。 |

验证命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile run_pipeline_abc_v2.py loger/pipeline/hybrid_memory_controller.py tools/v57_experiment_report.py
bash -n tools/run_v57_h35_semantic_action_repair_ttt_ttl_fast.sh
```

## Phase 0 审计产物

- semantic audit: `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/report_final/semantic_action_realization_audit.md`
- TTT audit: `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/report_final/ttt_action_regression_audit.md`

## Gate 停止记录

- Semantic 704F promoted rows: `SREAD03_704F`。
- Semantic full rows executed: `SREAD03_FULL`。
- TTT 704F promoted rows: `none`。
- TTT full rows executed: `none`。
- Combo rows executed: `none`。

## 运行命令清单

| row | run | GPU | frames | status | wall min | ATE | 输出目录 |
|---|---|---:|---:|---|---:|---:|---|
| `S0_96F` | `V57R1_S0_FORCED_SEMANTIC_SOURCE_SKIP_TRACE_96F` | 0 | 96 | done | 2.700000 | 1.279743 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase1_semantic_smoke/rollouts/V57R1_S0_FORCED_SEMANTIC_SOURCE_SKIP_TRACE_96F` |
| `S1_96F` | `V57R1_S1_SKY_LOWSTUFF_SOURCE_SKIP_TRACE_96F` | 1 | 96 | done | 2.750000 | 1.279743 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase1_semantic_smoke/rollouts/V57R1_S1_SKY_LOWSTUFF_SOURCE_SKIP_TRACE_96F` |
| `S2_96F` | `V57R1_S2_HIGH_D_SEMANTIC_SOURCE_SKIP_TRACE_96F` | 2 | 96 | done | 2.716667 | 1.279673 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase1_semantic_smoke/rollouts/V57R1_S2_HIGH_D_SEMANTIC_SOURCE_SKIP_TRACE_96F` |
| `S0_96F` | `V57_S0_FORCED_SEMANTIC_SOURCE_SKIP_96F` | 0 | 96 | done | 3.050000 | 1.279743 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase1_semantic_smoke/rollouts/V57_S0_FORCED_SEMANTIC_SOURCE_SKIP_96F` |
| `S1_96F` | `V57_S1_SKY_LOWSTUFF_SOURCE_SKIP_96F` | 1 | 96 | done | 3.150000 | 1.279743 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase1_semantic_smoke/rollouts/V57_S1_SKY_LOWSTUFF_SOURCE_SKIP_96F` |
| `S2_96F` | `V57_S2_HIGH_D_SEMANTIC_SOURCE_SKIP_96F` | 2 | 96 | done | 3.000000 | 1.279673 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase1_semantic_smoke/rollouts/V57_S2_HIGH_D_SEMANTIC_SOURCE_SKIP_96F` |
| `SREAD01_704F` | `V57_SREAD01_GENERAL_HIGH_INFLUENCE_ANOMALY_704F` | 0 | 704 | done | 18.266667 | 44.687484 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase2_semantic_read_704_screen/rollouts/V57_SREAD01_GENERAL_HIGH_INFLUENCE_ANOMALY_704F` |
| `SREAD02_704F` | `V57_SREAD02_SKY_LOWSTUFF_HIGH_INFLUENCE_704F` | 1 | 704 | done | 18.533333 | 44.712281 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase2_semantic_read_704_screen/rollouts/V57_SREAD02_SKY_LOWSTUFF_HIGH_INFLUENCE_704F` |
| `SREAD03_704F` | `V57_SREAD03_SEM_C23_RESIDUAL_ACTION_GUARD_704F` | 2 | 704 | done | 18.233333 | 38.209718 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase2_semantic_read_704_screen/rollouts/V57_SREAD03_SEM_C23_RESIDUAL_ACTION_GUARD_704F` |
| `SREAD04_704F` | `V57_SREAD04_ANOMALY_FILTER_STATIC_RESCUE_704F` | 3 | 704 | done | 18.866667 | 44.687484 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase2_semantic_read_704_screen/rollouts/V57_SREAD04_ANOMALY_FILTER_STATIC_RESCUE_704F` |
| `SREAD03_FULL` | `V57_SREAD03_SEM_C23_RESIDUAL_ACTION_GUARD_FULL` | 0 | 1101 | done | 27.950000 | 36.422824 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase3_semantic_read_full/rollouts/V57_SREAD03_SEM_C23_RESIDUAL_ACTION_GUARD_FULL` |
| `TTT01_704F` | `V57_TTT01_TWO_REPLAY_STATIC_NATIVE_704F` | 4 | 704 | done | 17.950000 | 41.476049 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase4_ttt_704_screen/rollouts/V57_TTT01_TWO_REPLAY_STATIC_NATIVE_704F` |
| `TTT02_704F` | `V57_TTT02_SHORT_RESIDUAL_TTL_704F` | 5 | 704 | done | 16.150000 | 41.781139 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase4_ttt_704_screen/rollouts/V57_TTT02_SHORT_RESIDUAL_TTL_704F` |
| `TTT03_704F` | `V57_TTT03_READ_CONDITIONED_RESTRICTED_NO_LONG_704F` | 4 | 704 | done | 18.150000 | 41.816010 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase4_ttt_704_screen/rollouts/V57_TTT03_READ_CONDITIONED_RESTRICTED_NO_LONG_704F` |
| `TTT01_96F` | `V57_TTT01_TWO_REPLAY_STATIC_NATIVE_96F` | 3 | 96 | done | 2.866667 | 1.227985 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase4_ttt_smoke/rollouts/V57_TTT01_TWO_REPLAY_STATIC_NATIVE_96F` |
| `TTT02_96F` | `V57_TTT02_SHORT_RESIDUAL_TTL_96F` | 4 | 96 | done | 2.766667 | 1.254267 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase4_ttt_smoke/rollouts/V57_TTT02_SHORT_RESIDUAL_TTL_96F` |
| `TTT03_96F` | `V57_TTT03_READ_CONDITIONED_RESTRICTED_NO_LONG_96F` | 5 | 96 | done | 2.750000 | 1.260858 | `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase4_ttt_smoke/rollouts/V57_TTT03_READ_CONDITIONED_RESTRICTED_NO_LONG_96F` |

复现单条 run 模板:

```bash
tools/run_v57_h35_semantic_action_repair_ttt_ttl_fast.sh <GPU> <ROW> <RUN_NAME>
```

每个 run 目录都包含 `effective_config.yaml`、`v57_effective_config.yaml`、`adaptive_ttt_audit.json`、`chunk_id_policy_audit.json`、`reproduce_command.sh`。

## 报告生成

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v57_experiment_report.py
```

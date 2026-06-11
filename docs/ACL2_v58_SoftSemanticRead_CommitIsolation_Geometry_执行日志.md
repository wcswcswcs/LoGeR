# ACL2 v58 SoftSemanticRead CommitIsolation Geometry 执行日志

日期: 2026-06-09
计划文档: `docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md`
结果根目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry`

## 代码/工具修改记录

- `loger/models/layers/attention.py`: 新增 `source_soft` SDPA descriptor，支持 soft attention bias mass 统计与 V-only effective value mass 统计。
- `loger/models/pi3.py`: context source control 新增 `v_only`、soft mass descriptor、`semantic_z_dg_soft_resid` 和 `random_same_mass_semantic_role_negative` mask；trace 记录 soft action 字段。
- `tools/run_v58_soft_semantic_read_commit_isolation.sh`: 新增 v58 候选 runner，所有候选使用 `HMC_COMMIT_MODE=probe_ttt_write`。
- `tools/v58_experiment_report.py`: 新增 artifact-only reporter，输出 registry、Phase0 autopsy、最终复盘。
- R1 efficiency repair: `loger/models/layers/attention.py` 的 V-only/no-bias source_soft path 改回 flash SDPA 执行实际 attention，保留采样统计。
- R4 soft repair: 不改语义 threshold，只用 `V58_R4_SOFT_RHO=0.65 V58_R4_SOFT_MIN_KEEP=0.4` 重跑 smoke；ratio 合格但 projected runtime 未过 gate。
- Reproduce repair: runner 和 R4 repair rollout 的 `reproduce_command.sh` 补写候选 layer/rho/min_keep，避免非默认 soft 参数丢失。

## 编译/静态检查

- `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile loger/models/pi3.py loger/models/layers/attention.py`
- `bash -n tools/run_v58_soft_semantic_read_commit_isolation.sh`

## Phase 0 命令

- `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v58_experiment_report.py --result-root /mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry --v57-root /mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast --out-dir /mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/report_final`

## 实验运行命令

- `R1_96F` `R1_SREAD03_V_ONLY_C1` status=`done` frames=`96` ATE=`1.045004`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58R1_R1_SREAD03_V_ONLY_C1_96F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58R1_R1_SREAD03_V_ONLY_C1_96F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" tools/run_v58_soft_semantic_read_commit_isolation.sh "0" "R1_96F" "V58R1_R1_SREAD03_V_ONLY_C1_96F"`
- `R4_96F` `R4_SEM_Z_DG_SOFT_RESID_C1` status=`done` frames=`96` ATE=`1.131055`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58R1_R4_SEM_Z_DG_SOFT_RESID_C1_96F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58R1_R4_SEM_Z_DG_SOFT_RESID_C1_96F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" V58_R4_LAYER_MODE="early" V58_R4_SOFT_RHO="0.65" V58_R4_SOFT_MIN_KEEP="0.4" tools/run_v58_soft_semantic_read_commit_isolation.sh "3" "R4_96F" "V58R1_R4_SEM_Z_DG_SOFT_RESID_C1_96F"`
- `N0_96F` `N0_RANDOM_SAME_MASS_SOFT_C1` status=`done` frames=`96` ATE=`1.032597`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58_N0_RANDOM_SAME_MASS_SOFT_C1_96F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58_N0_RANDOM_SAME_MASS_SOFT_C1_96F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" tools/run_v58_soft_semantic_read_commit_isolation.sh "4" "N0_96F" "V58_N0_RANDOM_SAME_MASS_SOFT_C1_96F"`
- `R1_96F` `R1_SREAD03_V_ONLY_C1` status=`done` frames=`96` ATE=`1.045305`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58_R1_SREAD03_V_ONLY_C1_96F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58_R1_SREAD03_V_ONLY_C1_96F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" tools/run_v58_soft_semantic_read_commit_isolation.sh "0" "R1_96F" "V58_R1_SREAD03_V_ONLY_C1_96F"`
- `R2_96F` `R2_SREAD03_BIAS_FLOOR_C1` status=`done` frames=`96` ATE=`1.102751`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58_R2_SREAD03_BIAS_FLOOR_C1_96F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58_R2_SREAD03_BIAS_FLOOR_C1_96F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" tools/run_v58_soft_semantic_read_commit_isolation.sh "1" "R2_96F" "V58_R2_SREAD03_BIAS_FLOOR_C1_96F"`
- `R3_96F` `R3_SREAD03_EARLY_ONLY_C1` status=`done` frames=`96` ATE=`1.096407`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58_R3_SREAD03_EARLY_ONLY_C1_96F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58_R3_SREAD03_EARLY_ONLY_C1_96F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" tools/run_v58_soft_semantic_read_commit_isolation.sh "2" "R3_96F" "V58_R3_SREAD03_EARLY_ONLY_C1_96F"`
- `R4_96F` `R4_SEM_Z_DG_SOFT_RESID_C1` status=`done` frames=`96` ATE=`1.100331`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58_R4_SEM_Z_DG_SOFT_RESID_C1_96F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase1_soft_read_smoke/rollouts/V58_R4_SEM_Z_DG_SOFT_RESID_C1_96F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" tools/run_v58_soft_semantic_read_commit_isolation.sh "3" "R4_96F" "V58_R4_SEM_Z_DG_SOFT_RESID_C1_96F"`
- `N0_704F` `N0_RANDOM_SAME_MASS_SOFT_C1` status=`done` frames=`704` ATE=`41.068937`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase2_soft_read_704_screen/rollouts/V58_N0_RANDOM_SAME_MASS_SOFT_C1_704F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase2_soft_read_704_screen/rollouts/V58_N0_RANDOM_SAME_MASS_SOFT_C1_704F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" tools/run_v58_soft_semantic_read_commit_isolation.sh "3" "N0_704F" "V58_N0_RANDOM_SAME_MASS_SOFT_C1_704F"`
- `R1_704F` `R1_SREAD03_V_ONLY_C1` status=`done` frames=`704` ATE=`40.820070`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase2_soft_read_704_screen/rollouts/V58_R1_SREAD03_V_ONLY_C1_704F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase2_soft_read_704_screen/rollouts/V58_R1_SREAD03_V_ONLY_C1_704F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" tools/run_v58_soft_semantic_read_commit_isolation.sh "0" "R1_704F" "V58_R1_SREAD03_V_ONLY_C1_704F"`
- `R2_704F` `R2_SREAD03_BIAS_FLOOR_C1` status=`done` frames=`704` ATE=`40.731486`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase2_soft_read_704_screen/rollouts/V58_R2_SREAD03_BIAS_FLOOR_C1_704F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase2_soft_read_704_screen/rollouts/V58_R2_SREAD03_BIAS_FLOOR_C1_704F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" tools/run_v58_soft_semantic_read_commit_isolation.sh "1" "R2_704F" "V58_R2_SREAD03_BIAS_FLOOR_C1_704F"`
- `R3_704F` `R3_SREAD03_EARLY_ONLY_C1` status=`done` frames=`704` ATE=`40.785790`
  - run_dir: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase2_soft_read_704_screen/rollouts/V58_R3_SREAD03_EARLY_ONLY_C1_704F`
  - reproduce: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/phase2_soft_read_704_screen/rollouts/V58_R3_SREAD03_EARLY_ONLY_C1_704F/reproduce_command.sh`
  - exact command: `V58_RESULT_ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry" V58_PLAN_NOTE="/mnt/data/users/chengshun.wang/pjs/LoGeR/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md" tools/run_v58_soft_semantic_read_commit_isolation.sh "2" "R3_704F" "V58_R3_SREAD03_EARLY_ONLY_C1_704F"`

## 复现说明

- 每个 rollout 目录内有 `effective_config.yaml`、`reproduce_command.sh`、`01.log`、`hmc_state_hash.jsonl`、`kitti_benchmark.log` 和 `wall_time_summary.json`。
- gate 和结论只使用已经落盘的 registry/trace；未运行的 Phase 不补数据。

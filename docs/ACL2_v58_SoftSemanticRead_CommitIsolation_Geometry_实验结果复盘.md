# ACL2 v58 SoftSemanticRead CommitIsolation Geometry 实验结果复盘

日期: 2026-06-09
计划文档: `docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md`
结果根目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry`

结论先行: 本报告只基于已落盘 artifact 生成；未运行、未写出或不可推断的字段保持 NA/unavailable。
H35 full baseline ATE: `35.74089695811434`；H35 704F baseline ATE: `39.79824772048563`。
Full minimum progress gate: ATE <= `35.2409`；semantic success gate: ATE <= `33.7409`；strong success gate: ATE <= `33.0`。
704F promotion gate: delta_vs_H35_704 <= `-0.5`；borderline diagnostic full gate: best delta <= `+0.2`。

## 代码/修复审计摘要

- `loger/models/layers/attention.py`: 新增 `source_soft` descriptor；R1 blocker 后把 V-only/no-bias path 的实际 attention 执行修回 flash SDPA，同时保留采样统计。
- `loger/models/pi3.py`: 新增 soft READ source control wiring，包括 `v_only`、soft bias、`semantic_z_dg_soft_resid` 和 `random_same_mass_semantic_role_negative`。
- `tools/run_v58_soft_semantic_read_commit_isolation.sh`: 新增 v58 runner，固定 C1 `probe_ttt_write` commit isolation，并在 reproduce 脚本中写入候选 layer/rho/min_keep。
- `tools/v58_experiment_report.py`: 新增 artifact-only reporter，输出 registry、Phase0 autopsy、执行日志与复盘；缺失字段保持 NA/unavailable。
- R4 repair 只调 soft action 参数 `V58_R4_SOFT_RHO=0.65`、`V58_R4_SOFT_MIN_KEEP=0.4`，没有调语义阈值；已有 R4 repair rollout 的 reproduce 脚本已补写这两个 override。

## Phase 0: SREAD03 autopsy

- timeline rows: `38`。
- SREAD03 704F artifact: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase2_semantic_read_704_screen/rollouts/V57_SREAD03_SEM_C23_RESIDUAL_ACTION_GUARD_704F`。
- SREAD03 full artifact: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/phase3_semantic_read_full/rollouts/V57_SREAD03_SEM_C23_RESIDUAL_ACTION_GUARD_FULL`。
- v57 SREAD03 是 hard compact K/V skip，registry/trace 中 affected-source attention mass after 为 0；这满足 v58 必须转 soft attenuation 的前提。
- SREAD01/SREAD04 per-token source_keep_mask 没有落盘，Jaccard 只能标为 unavailable；不能把 aggregate 相同伪写成 token mask 相同。

## Phase 1: 96F smoke

| run | row | candidate | ATE | stage_hit | groups | tokens | mass_before | mass_after | ratio | empty | commit | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V58R1_R1_SREAD03_V_ONLY_C1_96F | R1_96F | R1_SREAD03_V_ONLY_C1 | 1.045004 | 1.000000 | 3.000000 | 2334.500000 | 0.116683 | 0.058341 | 0.500000 | 0 | True | True |
| V58R1_R4_SEM_Z_DG_SOFT_RESID_C1_96F | R4_96F | R4_SEM_Z_DG_SOFT_RESID_C1 | 1.131055 | 1.000000 | 3.000000 | 3512.250000 | 0.180203 | 0.117728 | 0.653306 | 0 | True | False |
| V58_N0_RANDOM_SAME_MASS_SOFT_C1_96F | N0_96F | N0_RANDOM_SAME_MASS_SOFT_C1 | 1.032597 | 1.000000 | 3.000000 | 2334.500000 | 0.139334 | 0.075055 | 0.538668 | 0 | True | True |
| V58_R1_SREAD03_V_ONLY_C1_96F | R1_96F | R1_SREAD03_V_ONLY_C1 | 1.045305 | 1.000000 | 3.000000 | 2334.500000 | 0.116683 | 0.058342 | 0.500000 | 0 | True | False |
| V58_R2_SREAD03_BIAS_FLOOR_C1_96F | R2_96F | R2_SREAD03_BIAS_FLOOR_C1 | 1.102751 | 1.000000 | 3.000000 | 2334.500000 | 0.116162 | 0.062957 | 0.541971 | 0 | True | True |
| V58_R3_SREAD03_EARLY_ONLY_C1_96F | R3_96F | R3_SREAD03_EARLY_ONLY_C1 | 1.096407 | 1.000000 | 3.000000 | 2334.500000 | 0.124204 | 0.067465 | 0.543176 | 0 | True | True |
| V58_R4_SEM_Z_DG_SOFT_RESID_C1_96F | R4_96F | R4_SEM_Z_DG_SOFT_RESID_C1 | 1.100331 | 1.000000 | 3.000000 | 3512.250000 | 0.180047 | 0.131156 | 0.728456 | 0 | True | False |

Phase1 usable semantic smoke candidate exists: `True`。

Phase1 repair/blocker 记录:
- R1 V-only 初跑 runtime projected `31.033333` min，修复 V-only 无 bias fast SDPA 后 projected `27.866667` min，gate `True`。
- R4 sem-z 初跑 mass ratio `0.728456` 超过 0.7；按计划只调 soft rho/min_keep 到 `0.65`/`0.4`，ratio `0.653306`，但 projected `31.825000` min，仍因 runtime gate 停止。

## Phase 2: 704F screen

| run | row | candidate | ATE | dH35_704 | rolling100p90 | tokens | mass_ratio | static_rm | commit | promote |
|---|---|---|---|---|---|---|---|---|---|---|
| V58_N0_RANDOM_SAME_MASS_SOFT_C1_704F | N0_704F | N0_RANDOM_SAME_MASS_SOFT_C1 | 41.068937 | 1.270689 | 65.808350 | 2481.520000 | 0.535567 | 0.000000 | True | False |
| V58_R1_SREAD03_V_ONLY_C1_704F | R1_704F | R1_SREAD03_V_ONLY_C1 | 40.820070 | 1.021823 | 65.209733 | 2481.520000 | 0.500000 | 0.000000 | True | False |
| V58_R2_SREAD03_BIAS_FLOOR_C1_704F | R2_704F | R2_SREAD03_BIAS_FLOOR_C1 | 40.731486 | 0.933238 | 65.283255 | 2481.520000 | 0.538647 | 0.000000 | True | False |
| V58_R3_SREAD03_EARLY_ONLY_C1_704F | R3_704F | R3_SREAD03_EARLY_ONLY_C1 | 40.785790 | 0.987543 | 65.445749 | 2481.520000 | 0.539591 | 0.000000 | True | False |

Promoted to full: `none`。

## Phase 3: full KITTI01

| row | status |
|---|---|
| none | not run by gate |

Full 未运行原因: 704F 无候选过 promotion gate，best semantic 704F delta=`0.933238`，未达到 borderline diagnostic full 条件 `<= +0.2`。

## Gate summary

- best 704F semantic candidate: `R2_SREAD03_BIAS_FLOOR_C1` ATE `40.731486`，delta `0.933238`。
- borderline diagnostic full allowed: `False`。
- best full candidate: `none` ATE `NA`。
- semantic_min_progress=`False`, semantic_target=`False`。

## Insight 与证据链

- v58 的第一层证据是 action form：source 不再被 hard compact 到 mass_after=0；soft 候选必须在 trace 中保留 0.3-0.7 的 affected-source effective/attention mass。
- C1 commit isolation 通过 `hmc_commit_mode=probe_ttt_write`、`state_double_write_safe`、`probe_no_commit_hash_equal` 和 commit-source hash 字段审计；若这些字段缺失或失败，不把 full 结果算作语义成功。
- V-only 候选的 `mass_after` 是 effective value mass；同时 reporter 记录 `attention_mass_actual_after_mean` 和 `attention_mass_metrics_seen`，避免把 V attenuation 伪装成 attention-logit mass 改变。
- 若所有 full 均未达到 H35-0.5m，本轮结论应降级为 semantic READ 当前更适合 diagnostic，而不是继续扩大 hard skip/TTT 小扫。

## 必答问题

1. 语义 READ 是否已经从 hard skip 转成 soft attenuation: `yes`。
2. 是否真的保留了部分 source attention/effective mass: `yes`。
3. SREAD03 704F 改善来自哪些 source: 见 `phase0_sread03_autopsy/sread03_semantic_group_mass.csv`；v57 trace 主要能报告 group/label aggregate，不能恢复 token mask。
4. full 失败是否来自后段误激活，还是 commit side effect: 只有 full artifact 和 commit hash fields 同时存在时才能判定；缺失则保持 unavailable。
5. commit isolation 是否有效: `yes`。
6. soft semantic READ 是否相比 H35 带来 full ATE 改善: `not proven`。
7. 如果没有改善，语义路线应降级到 diagnostic/offline explanation，后续转向 TTT harmful update attribution、trajectory-state 或 merge-gauge controller。

## 审计材料

- registries: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/report_final`
- Phase0 autopsy: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/report_final/phase0_sread03_autopsy`
- figures: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/report_final/figures` and `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry/report_final/phase0_sread03_autopsy/figures`

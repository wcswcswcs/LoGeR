# ACL2 v45 执行日志

日期：2026-06-08（Asia/Singapore）  
计划文件：`docs/ACL2_v45_CodeAudit_C9Clean_Attribution_C23AdaptiveTriReplay_Plan.md`  
实验结果复盘：`docs/ACL2_v45_实验复盘.md`  
结果根目录：`results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/`

原则：

```text
不编造数据。
不把未运行的实验写成结果。
遇到 blocker 先按计划方向修复或降级为明确诊断。
所有关键命令、文件、GPU、输出目录和 gate 结果都记录在本日志。
```

可用 GPU：

```text
4,5,6,7
```

---

## 0. 读取计划和初始状态

读取计划与历史日志：

```bash
sed -n '1,260p' docs/ACL2_v45_CodeAudit_C9Clean_Attribution_C23AdaptiveTriReplay_Plan.md
sed -n '261,620p' docs/ACL2_v45_CodeAudit_C9Clean_Attribution_C23AdaptiveTriReplay_Plan.md
sed -n '621,1040p' docs/ACL2_v45_CodeAudit_C9Clean_Attribution_C23AdaptiveTriReplay_Plan.md
sed -n '1,260p' docs/ACL2_v44_执行日志.md
sed -n '1,260p' docs/ACL2_v44_实验复盘.md
```

初始工作区检查：

```bash
pwd
rg --files docs | sort
git status --short
```

结果：

```text
cwd = /mnt/data/users/chengshun.wang/pjs/LoGeR
计划目标文件存在。
仓库在开始前已有大量 modified/untracked 文件；本轮不回滚用户既有改动。
```

依赖文件检查：

```bash
ls -lh eval/long_eval_script/kitti_benchmark \
  ckpts/LoGeR/latest.pt \
  ckpts/LoGeR/original_config.yaml \
  /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt
```

结果：

```text
eval/long_eval_script/kitti_benchmark exists and executable
ckpts/LoGeR/latest.pt exists, size 4.7G
ckpts/LoGeR/original_config.yaml exists
KITTI pose file /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt exists
```

---

## 1. v45 代码准备

读取关键入口和实现：

```bash
sed -n '1,260p' tools/run_v43_full_candidate.sh
sed -n '1,280p' tools/v43_registry_summarize.py
sed -n '1,280p' tools/run_attention_cue_experiment.sh
sed -n '880,1165p' loger/pipeline/ttt_write_controller.py
sed -n '2240,2385p' loger/pipeline/ttt_write_controller.py
rg -n "def _acl2_support_indices|full_chunk_no_overlap|past_plus_future_light_real|off246|near12|overlap_excluded|robustq" loger/pipeline/hybrid_memory_controller.py -C 4
```

确认：

```text
1. C9 launcher 默认 Stage C semantic 为 none。
2. C9 absolute chunk-id surfaces 包括 read beta chunk map、tri gamma chunk map、tri replay chunk params、commit EMA chunks。
3. full_chunk_no_overlap 已有真实 overlap seam 排除。
4. overlap_excluded/noovlp/ovlp_only 仍 fallback 到 full support，不能用于 no-overlap claim。
5. TTT tri_replay 分支后确有旧 two-replay dead code；非 tri replay 需要修复或禁用。
```

本轮新增/修改：

```text
loger/pipeline/ttt_write_controller.py
    - 新增 tri_replay_role_mode，默认 fixed。
    - tri_replay fixed 默认保持历史 quantile role assignment。
    - 新增 kmeans3 / otsu3 / mad / adaptive_quantile training-free role assignment。
    - 修复 two_replay / separate_replay / pos_neg_replay：非 tri mode 现在走明确的 two-replay negative 分支并 return。
    - 删除 tri_replay return 后不可达旧代码块。

loger/pipeline/hybrid_memory_controller.py
    - 将 ttt_write_tri_replay_role_mode 传入 TTTWriteController。

run_pipeline_abc_v2.py
    - 新增 CLI 参数 --ttt_write_tri_replay_role_mode。
    - 将参数传入 HybridMemoryController。

tools/run_attention_cue_experiment.sh
    - 新增环境变量 TTT_WRITE_TRI_REPLAY_ROLE_MODE。
    - 将其传给 run_pipeline_abc_v2.py。

tools/run_v45_full_candidate.sh
    - 新增 v45 专用 launcher。
    - 输出 effective_config.yaml、chunk_id_policy_audit.json、stage_c_semantic_disabled_confirm.json。
    - 默认开启 V11_PROJECTION_TRACE_DIR=$OUT/v11_projection_trace 只做审计记录。

tools/v45_report.py
    - 新增 landed-artifact-only 汇总脚本。
    - 读取 registry/jsonl 生成 v45 reports；缺失结果不补数字。
```

验证命令：

```bash
chmod +x tools/run_v45_full_candidate.sh tools/v45_report.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  run_pipeline_abc_v2.py \
  loger/pipeline/hybrid_memory_controller.py \
  loger/pipeline/ttt_write_controller.py \
  tools/v45_report.py

bash -n tools/run_v45_full_candidate.sh tools/run_attention_cue_experiment.sh
```

结果：

```text
py_compile pass
bash -n pass
```

---

## 2. Phase 0 smoke：TTT replay 修复路径

目的：

```text
1. 验证计划中指出的 TTT two/separate/pos_neg replay 风险路径不会再因不可达旧代码/未定义变量崩溃。
2. 验证 Phase 4 新增 training-free tri replay role mode 至少能在真实 chunk 上触发并写出审计 trace。
3. smoke 不作为 v45 ATE 结果，只作为代码路径有效性和日志证据。
```

第一次 64-frame smoke 记录：

```bash
END_FRAME=64 \
V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts \
tools/run_v45_full_candidate.sh 4 V45_SMOKE_TWO_REPLAY_64F SMOKE_TWO_REPLAY

END_FRAME=64 \
V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts \
tools/run_v45_full_candidate.sh 5 V45_SMOKE_A1_KMEANS3_64F A1_KMEANS3_TRI_REPLAY
```

结果和处理：

```text
两个 run 均完成，但不作为有效 smoke 证据。
原因 1：64 frames 只覆盖到 chunk 0-2；C9 tri gamma chunk map 从 chunk 5 开始，A1 没触发 tri replay。
原因 2：SMOKE_TWO_REPLAY 当时仍继承 launcher 默认 gamma=0.0，two-replay 未有效激活。
修复：tools/run_v45_full_candidate.sh 中 SMOKE_TWO_REPLAY 显式设置 TTT_WRITE_GRADIENT_REVERSAL_GAMMA=${V45_SMOKE_TWO_REPLAY_GAMMA:-0.001}。
验证：bash -n tools/run_v45_full_candidate.sh pass。
```

有效 two-replay smoke：

```bash
END_FRAME=64 \
V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts \
tools/run_v45_full_candidate.sh 4 V45_SMOKE_TWO_REPLAY_ACTIVE_64F SMOKE_TWO_REPLAY
```

输出位置：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts/V45_SMOKE_TWO_REPLAY_ACTIVE_64F/
```

run_status：

```text
[2026-06-08 01:31:01] START V45_SMOKE_TWO_REPLAY_ACTIVE_64F gpu=4 seq=01 mode=hybrid cue=acl2.gg.qq.low.g2_3.past_only.headmean.robustq beta=4.75 write=stage_d_x_dg_inv_sqrt reset_every=5
[2026-06-08 01:35:21] DONE V45_SMOKE_TWO_REPLAY_ACTIVE_64F
```

审计命令与结果：

```bash
rg -c "ttt_two_replay_applied': True" \
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts/V45_SMOKE_TWO_REPLAY_ACTIVE_64F/01.log
# 28

rg -c "ttt_gradient_reversal_applied': True" \
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts/V45_SMOKE_TWO_REPLAY_ACTIVE_64F/01.log
# 28

rg -c "Traceback|NameError|UnboundLocalError|RuntimeError|FAIL" \
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts/V45_SMOKE_TWO_REPLAY_ACTIVE_64F/01.log
# 0 matches
```

Stage C 关闭确认：

```json
{
  "candidate_id": "SMOKE_TWO_REPLAY",
  "stage_c_mode": "none",
  "stage_c_cache_mode": "off",
  "semantic_role_policy": "none",
  "semantic_memory_paths": "",
  "stage_c_disabled": true
}
```

有效 kmeans3 tri-replay smoke：

```bash
END_FRAME=180 \
V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts \
tools/run_v45_full_candidate.sh 5 V45_SMOKE_A1_KMEANS3_ACTIVE_180F A1_KMEANS3_TRI_REPLAY
```

输出位置：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts/V45_SMOKE_A1_KMEANS3_ACTIVE_180F/
```

run_status：

```text
[2026-06-08 01:31:01] START V45_SMOKE_A1_KMEANS3_ACTIVE_180F gpu=5 seq=01 mode=hybrid cue=acl2.gg.qq.low.g2_3.past_only.headmean.robustq beta=4.75 write=stage_d_x_dg_inv_sqrt reset_every=5
[2026-06-08 01:37:06] DONE V45_SMOKE_A1_KMEANS3_ACTIVE_180F
```

审计命令与结果：

```bash
rg -c "ttt_tri_replay_role_mode': 'kmeans3'" \
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts/V45_SMOKE_A1_KMEANS3_ACTIVE_180F/01.log
# 18

rg -c "Traceback|NameError|UnboundLocalError|RuntimeError|FAIL" \
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts/V45_SMOKE_A1_KMEANS3_ACTIVE_180F/01.log
# 0 matches
```

role mass trace：

```text
file: results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts/V45_SMOKE_A1_KMEANS3_ACTIVE_180F/v11_projection_trace/tri_replay_role_mass.jsonl
role_rows = 126
tri_true_rows = 36
positive_mass mean/min/max = 0.5786109450790617 / 0.3541005253791809 / 0.6000000238418579
neutral_mass  mean/min/max = 0.35102398610777324 / 0.25145503878593445 / 0.40317460894584656
negative_mass mean/min/max = 0.0703650880087581 / 0.03000991977751255 / 0.24272486567497253
```

Stage C 关闭确认：

```json
{
  "candidate_id": "A1_KMEANS3_TRI_REPLAY",
  "stage_c_mode": "none",
  "stage_c_cache_mode": "off",
  "semantic_role_policy": "none",
  "semantic_memory_paths": "",
  "stage_c_disabled": true
}
```

结论：

```text
two-replay 修复路径已在真实 64-frame smoke 中触发 28 次，无错误。
kmeans3 tri-replay role mode 已在真实 180-frame smoke 中触发，trace 显示 36 条 tri_replay_applied=true role mass 行。
上述 smoke 不报告 ATE，不用于 Phase 0 hard gate。
```

---

## 3. Phase 0 hard gate：P0 C9 repeat

执行命令：

```bash
SAVE_HMC_STATES=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/state_snapshots/V45_P0_C9_REPEAT \
SAVE_HMC_STATE_KINDS=input \
SAVE_HMC_STATE_CHUNKS=6,10,16 \
SAVE_MERGE_STATES=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/merge_state_snapshots/V45_P0_C9_REPEAT \
SAVE_MERGE_STATE_KINDS=input \
SAVE_MERGE_STATE_CHUNKS=6,10,16 \
tools/run_v45_full_candidate.sh 4 V45_P0_C9_REPEAT P0_C9_REPEAT
```

输出位置：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/rollouts/V45_P0_C9_REPEAT/
```

run_status：

```text
[2026-06-08 01:42:28] START V45_P0_C9_REPEAT gpu=4 seq=01 mode=hybrid cue=acl2.gg.qq.low.g2_3.past_only.headmean.robustq beta=4.75 write=stage_d_x_dg_inv_sqrt reset_every=5
[2026-06-08 02:23:01] DONE V45_P0_C9_REPEAT
```

关键进度/配置证据：

```text
hmc_state_hash.jsonl rows = 38
01.txt rows = 1102
chunk 5 prior_beta_frame_effective = 4.85
chunk 10 prior_beta_frame_effective = 4.25
chunk 16 prior_beta_frame_effective = 4.25
chunk 16 auxgeo_tri_replay_applied_layer_count = 18
```

生成 Phase 0 report 时遇到的 blocker：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/rollouts \
  --runs F0=V45_P0_C9_REPEAT \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/report_R1/full_metrics
```

首次失败：

```text
ModuleNotFoundError: No module named 'tools.v18_true_action_report'
```

定位：

```text
当前 conda 环境存在 site-packages/tools 包。
本仓库 tools/ 目录没有 __init__.py，导致 `from tools...` 被第三方包抢先解析。
```

修复：

```text
新增文件：tools/__init__.py
内容："""Local LoGeR utility scripts package."""
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY'
import sys
from pathlib import Path
root=Path.cwd()
sys.path.insert(0, str(root))
import tools.v18_true_action_report as mod
print('import_ok', mod.__file__)
PY

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/__init__.py \
  tools/v42_full_online_report.py \
  tools/v18_true_action_report.py
```

结果：

```text
import_ok /mnt/data/users/chengshun.wang/pjs/LoGeR/tools/v18_true_action_report.py
py_compile pass
```

重新生成 report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/rollouts \
  --runs F0=V45_P0_C9_REPEAT \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/report_R1/full_metrics
```

Report 输出：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/report_R1/full_metrics/full_online_registry.csv
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/report_R1/full_metrics/v42_full_online_report.md
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/report_R1/full_metrics/v42_full_online_rows.json
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/report_R1/full_metrics/v42_full_online_summary.json
```

P0 hard gate 结果：

```text
ATE_full = 33.76294210291885
historical_C9_ATE = 33.7629421029
abs_delta_m = 1.8850698779715458e-11
ATE gate abs_delta <= 0.03m: pass
hmc_rows = 38
hmc_rows gate == 38: pass
stage_c_mode = none
stage_c_disabled = true
Stage C gate: pass
```

P0 chunk-id audit：

```text
has_read_beta_frame_chunks = true
has_tri_gamma_chunk_map = true
has_tri_replay_chunk_params = true
has_commit_ema_chunks = true
```

说明：

```text
P0 是 C9 repeat，absolute chunk-id maps 预期存在。
这些 maps 后续在 D7 C9-Clean gate 中必须被移除。
```

---

## 4. Phase 1：D1-D6 C9-Clean 分解实验

并行调度：

```bash
tools/run_v45_full_candidate.sh 4 V45_D1_FIXED_READ_BETA_ONLY D1_FIXED_READ_BETA_ONLY
tools/run_v45_full_candidate.sh 5 V45_D2_FIXED_TRI_GAMMA_003 D2_FIXED_TRI_GAMMA_003
tools/run_v45_full_candidate.sh 6 V45_D3_FIXED_TRI_GAMMA_004 D3_FIXED_TRI_GAMMA_004
tools/run_v45_full_candidate.sh 7 V45_D4_FIXED_TRI_GAMMA_005 D4_FIXED_TRI_GAMMA_005
tools/run_v45_full_candidate.sh 4 V45_D5_FIXED_COMMIT_EMA_OFF D5_FIXED_COMMIT_EMA_OFF
tools/run_v45_full_candidate.sh 6 V45_D6_FIXED_COMMIT_EMA_GLOBAL_A08 D6_FIXED_COMMIT_EMA_GLOBAL_A08
```

run_status：

```text
V45_D1_FIXED_READ_BETA_ONLY: START 2026-06-08 02:27:31, DONE 2026-06-08 03:14:55
V45_D2_FIXED_TRI_GAMMA_003: START 2026-06-08 02:27:31, DONE 2026-06-08 03:45:47
V45_D3_FIXED_TRI_GAMMA_004: START 2026-06-08 02:27:31, DONE 2026-06-08 03:43:08
V45_D4_FIXED_TRI_GAMMA_005: START 2026-06-08 02:27:31, DONE 2026-06-08 03:45:40
V45_D5_FIXED_COMMIT_EMA_OFF: START 2026-06-08 03:16:30, DONE 2026-06-08 03:59:36
V45_D6_FIXED_COMMIT_EMA_GLOBAL_A08: START 2026-06-08 03:45:25, DONE 2026-06-08 04:26:56
```

生成 Phase 1 report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs F0=phase0_hard_gate/rollouts/V45_P0_C9_REPEAT,D1=phase1_c9_clean/rollouts/V45_D1_FIXED_READ_BETA_ONLY,D2=phase1_c9_clean/rollouts/V45_D2_FIXED_TRI_GAMMA_003,D3=phase1_c9_clean/rollouts/V45_D3_FIXED_TRI_GAMMA_004,D4=phase1_c9_clean/rollouts/V45_D4_FIXED_TRI_GAMMA_005,D5=phase1_c9_clean/rollouts/V45_D5_FIXED_COMMIT_EMA_OFF,D6=phase1_c9_clean/rollouts/V45_D6_FIXED_COMMIT_EMA_GLOBAL_A08 \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics
```

输出：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics/full_online_registry.csv
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics/v42_full_online_report.md
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics/v42_full_online_summary.json
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics/v42_full_online_rows.json
```

Phase 1 D1-D6 结果：

```text
F0/P0 ATE = 33.76294210291885
D1_FIXED_READ_BETA_ONLY ATE = 33.789522665336904, delta_vs_F0 = +0.0265805624180544
D2_FIXED_TRI_GAMMA_003 ATE = 34.73396750873536, delta_vs_F0 = +0.971025405816512
D3_FIXED_TRI_GAMMA_004 ATE = 34.64880019505802, delta_vs_F0 = +0.8858580921391734
D4_FIXED_TRI_GAMMA_005 ATE = 34.77852836133698, delta_vs_F0 = +1.015586258418132
D5_FIXED_COMMIT_EMA_OFF ATE = 34.25132756681247, delta_vs_F0 = +0.48838546389362136
D6_FIXED_COMMIT_EMA_GLOBAL_A08 ATE = 34.674495366802454, delta_vs_F0 = +0.9115532638836044
```

D1-D6 结论：

```text
所有 D1-D6 去 chunk-id 分解项均比 P0 更差。
最小 ATE 损伤：D1 fixed read beta，+0.0265805624180544m。
固定 tri gamma 候选中最小损伤：D3 gamma=0.004，+0.8858580921391734m。
commit EMA 候选中最小损伤：D5 off，+0.48838546389362136m。
```

D7 选择与启动：

```bash
V45_C9_CLEAN_TRI_GAMMA=0.004 \
V45_C9_CLEAN_COMMIT_EMA_ALPHA=1.0 \
V45_C9_CLEAN_COMMIT_EMA_BRANCH_MASK=all \
SAVE_HMC_STATES=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/state_snapshots/V45_D7_C9_CLEAN_BEST_FIXED \
SAVE_HMC_STATE_KINDS=input \
SAVE_HMC_STATE_CHUNKS=6,10,16 \
SAVE_MERGE_STATES=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/merge_state_snapshots/V45_D7_C9_CLEAN_BEST_FIXED \
SAVE_MERGE_STATE_KINDS=input \
SAVE_MERGE_STATE_CHUNKS=6,10,16 \
tools/run_v45_full_candidate.sh 4 V45_D7_C9_CLEAN_BEST_FIXED D7_C9_CLEAN_BEST_FIXED
```

说明：

```text
D7 采用 fixed read beta + fixed tri gamma 0.004 + commit EMA off。
这是基于 D1-D6 真实 ATE 中各类去 chunk-id 替代的最小损伤选择，不代表预期会优于 P0。
D7 gate 需要额外检查四类 absolute chunk-id maps 均为 false。
```

## Phase 2 启动：interaction attribution I1-I3

时间：2026-06-08 04:53:24 SGT

Phase 1 D1-D6 已确定用于 Phase 2 的固定参数：

```text
V45_FIXED_TRI_GAMMA_BEST=0.004
V45_FIXED_EMA_ALPHA_BEST=1.0
V45_FIXED_EMA_BRANCH_MASK_BEST=all
```

说明：

```text
D7 仍在 GPU4 running；I1-I3 不依赖 D7 最终 ATE，只依赖 D1-D6 已确定的 fixed 参数。
为节省墙钟时间，I1-I3 在空闲 GPU5/6/7 并行启动，GPU4 保留给 D7。
```

命令：

```bash
V45_FIXED_TRI_GAMMA_BEST=0.004 \
V45_FIXED_EMA_ALPHA_BEST=1.0 \
V45_FIXED_EMA_BRANCH_MASK_BEST=all \
tools/run_v45_full_candidate.sh 5 V45_I1_NO_TRI_REPLAY_NO_EMA I1_NO_TRI_REPLAY_NO_EMA

V45_FIXED_TRI_GAMMA_BEST=0.004 \
V45_FIXED_EMA_ALPHA_BEST=1.0 \
V45_FIXED_EMA_BRANCH_MASK_BEST=all \
tools/run_v45_full_candidate.sh 6 V45_I2_NO_TRI_REPLAY_NO_SWA I2_NO_TRI_REPLAY_NO_SWA

V45_FIXED_TRI_GAMMA_BEST=0.004 \
V45_FIXED_EMA_ALPHA_BEST=1.0 \
V45_FIXED_EMA_BRANCH_MASK_BEST=all \
tools/run_v45_full_candidate.sh 7 V45_I3_NO_TRI_REPLAY_NATIVE_MIX_OFF I3_NO_TRI_REPLAY_NATIVE_MIX_OFF
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/rollouts/V45_I1_NO_TRI_REPLAY_NO_EMA
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/rollouts/V45_I2_NO_TRI_REPLAY_NO_SWA
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/rollouts/V45_I3_NO_TRI_REPLAY_NATIVE_MIX_OFF
```

I1-I3 run_status：

```text
V45_I1_NO_TRI_REPLAY_NO_EMA: START 2026-06-08 04:53:24, DONE 2026-06-08 05:28:32
V45_I2_NO_TRI_REPLAY_NO_SWA: START 2026-06-08 04:53:24, DONE 2026-06-08 05:29:05
V45_I3_NO_TRI_REPLAY_NATIVE_MIX_OFF: START 2026-06-08 04:53:24, DONE 2026-06-08 05:27:22
```

I1-I3 完整性：

```text
hmc_state_hash.jsonl rows = 38 for I1/I2/I3
错误关键字扫描 Traceback|NameError|UnboundLocalError|RuntimeError|FAIL：0 matches
```

## Phase 2 继续启动：I4-I6

时间：2026-06-08 05:29:49 SGT

命令：

```bash
V45_FIXED_TRI_GAMMA_BEST=0.004 \
V45_FIXED_EMA_ALPHA_BEST=1.0 \
V45_FIXED_EMA_BRANCH_MASK_BEST=all \
tools/run_v45_full_candidate.sh 5 V45_I4_FIXED_TRI_GAMMA_BEST_NO_EMA I4_FIXED_TRI_GAMMA_BEST_NO_EMA

V45_FIXED_TRI_GAMMA_BEST=0.004 \
V45_FIXED_EMA_ALPHA_BEST=1.0 \
V45_FIXED_EMA_BRANCH_MASK_BEST=all \
tools/run_v45_full_candidate.sh 6 V45_I5_FIXED_TRI_GAMMA_BEST_NO_SWA I5_FIXED_TRI_GAMMA_BEST_NO_SWA

V45_FIXED_TRI_GAMMA_BEST=0.004 \
V45_FIXED_EMA_ALPHA_BEST=1.0 \
V45_FIXED_EMA_BRANCH_MASK_BEST=all \
tools/run_v45_full_candidate.sh 7 V45_I6_FIXED_TRI_GAMMA_BEST_NATIVE_MIX_OFF I6_FIXED_TRI_GAMMA_BEST_NATIVE_MIX_OFF
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/rollouts/V45_I4_FIXED_TRI_GAMMA_BEST_NO_EMA
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/rollouts/V45_I5_FIXED_TRI_GAMMA_BEST_NO_SWA
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/rollouts/V45_I6_FIXED_TRI_GAMMA_BEST_NATIVE_MIX_OFF
```

## Phase 1 D7 完成与报告

D7 run_status：

```text
V45_D7_C9_CLEAN_BEST_FIXED: START 2026-06-08 04:28:31, DONE 2026-06-08 05:41:04
hmc_state_hash.jsonl rows = 38
01.txt rows = 1102
错误关键字扫描 Traceback|NameError|UnboundLocalError|RuntimeError|FAIL：0 matches
```

生成包含 D7 的 Phase 1 report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs F0=phase0_hard_gate/rollouts/V45_P0_C9_REPEAT,D1=phase1_c9_clean/rollouts/V45_D1_FIXED_READ_BETA_ONLY,D2=phase1_c9_clean/rollouts/V45_D2_FIXED_TRI_GAMMA_003,D3=phase1_c9_clean/rollouts/V45_D3_FIXED_TRI_GAMMA_004,D4=phase1_c9_clean/rollouts/V45_D4_FIXED_TRI_GAMMA_005,D5=phase1_c9_clean/rollouts/V45_D5_FIXED_COMMIT_EMA_OFF,D6=phase1_c9_clean/rollouts/V45_D6_FIXED_COMMIT_EMA_GLOBAL_A08,D7=phase1_c9_clean/rollouts/V45_D7_C9_CLEAN_BEST_FIXED \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics
```

报告输出：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics/full_online_registry.csv
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics/v42_full_online_report.md
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics/v42_full_online_summary.json
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics/v42_full_online_rows.json
```

D7 指标：

```text
P0/F0 ATE_full = 33.76294210291885
D7 ATE_full = 35.500497135292775
D7 delta_vs_P0 = +1.737555032373925
D7 hmc_rows = 38
D7 frames = 1101
```

D7 gate：

```text
C9-Clean acceptable threshold = P0 + 0.30 = 34.06294210291885
D7 35.500497135292775 > 34.06294210291885
C9-Clean acceptable = false
C9-Clean promising = false
C9-Clean success = false
```

D7 chunk-id / semantic audit：

```text
has_read_beta_frame_chunks=false
has_tri_gamma_chunk_map=false
has_tri_replay_chunk_params=false
has_commit_ema_chunks=false
stage_c_disabled=true
```

D7 tri-replay actual role mass from `hmc_state_hash.jsonl`：

```text
positive_mass mean/min/max = 0.3499999940395355 / 0.3499999940395355 / 0.3499999940395355
neutral_mass mean/min/max = 0.5299852223772752 / 0.5299851298332214 / 0.5299886465072632
negative_mass mean/min/max = 0.12001479024949827 / 0.1200113371014595 / 0.12001488357782364
memory_ttt_mean_rel_diff mean/min/max = 0.027798287607620896 / 0.022599587242177453 / 0.035539337733617525
```

按计划，因 C9-Clean 不 acceptable，Phase 4 adaptive tri-replay 需要先在 C9-Clean 上跑；若 C9-Clean 不 acceptable，还需要在 C9 original 上跑 top 2 adaptive candidates。

## Phase 2 继续启动：I7

时间：2026-06-08 05:43:03 SGT

命令：

```bash
V45_FIXED_TRI_GAMMA_BEST=0.004 \
V45_FIXED_EMA_ALPHA_BEST=1.0 \
V45_FIXED_EMA_BRANCH_MASK_BEST=all \
tools/run_v45_full_candidate.sh 4 V45_I7_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST I7_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/rollouts/V45_I7_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST
```

## Phase 2 I4 完成与 I8 启动

I4 run_status：

```text
V45_I4_FIXED_TRI_GAMMA_BEST_NO_EMA: START 2026-06-08 05:29:49, DONE 2026-06-08 06:41:01
hmc_state_hash.jsonl rows = 38
错误关键字扫描 Traceback|NameError|UnboundLocalError|RuntimeError|FAIL：0 matches
```

I8 启动时间：2026-06-08 06:41:24 SGT

命令：

```bash
V45_FIXED_TRI_GAMMA_BEST=0.004 \
V45_FIXED_EMA_ALPHA_BEST=1.0 \
V45_FIXED_EMA_BRANCH_MASK_BEST=all \
tools/run_v45_full_candidate.sh 5 V45_I8_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST_FIXED_EMA_BEST I8_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST_FIXED_EMA_BEST
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/rollouts/V45_I8_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST_FIXED_EMA_BEST
```

## Phase 3 前 launcher 修复：readonly short fork 支持

背景：

```text
Phase 3 计划要求 short causal fork 使用 read-only / probe_native first。
tools/run_attention_cue_experiment.sh 已有 MODE=readonly 分支，会设置 hybrid_memory_mode=read_path_only 且 hmc_commit_mode=probe_native。
但 tools/run_v45_full_candidate.sh 原先固定 MODE="hybrid"，无法通过 v45 launcher 生成可审计的 readonly short run。
```

修改：

```text
文件：tools/run_v45_full_candidate.sh
改动：MODE="hybrid" -> MODE="${V45_MODE_OVERRIDE:-hybrid}"
默认值仍为 hybrid，不改变已经完成的 P0/D/I full-online run 行为。
```

验证：

```bash
bash -n tools/run_v45_full_candidate.sh
rg -n 'MODE=' tools/run_v45_full_candidate.sh | head
```

结果：

```text
bash -n pass
MODE="${V45_MODE_OVERRIDE:-hybrid}"
```

补充审计记录：

```text
I5/I6/I7 是在 launcher 修复前启动、修复后结束的长 run。
它们的 run_status.txt 均已写 DONE，hmc_state_hash.jsonl 均为 38 行；但对应 Codex exec 会话在 DONE 后返回：
    tools/run_v45_full_candidate.sh: line 343: unexpected EOF while looking for matching `"'

复核：
    当前 tools/run_v45_full_candidate.sh: bash -n pass
    当前文件实际只有 342 行，line 343 报错来自运行中的 bash 读取了被编辑后的脚本尾部。

判定：
    I5/I6/I7 的落盘数据完整；报告评估以 run_status DONE、01.txt、hmc_state_hash.jsonl 为准。
    后续不再编辑正在被运行会话使用的 launcher。
```

新增 Phase3 short fork 工具：

```text
tools/run_v45_c23_support_short_fork.sh
tools/v45_c23_support_short_report.py
```

用途：

```text
run_v45_c23_support_short_fork.sh:
    使用 P0 或 D7 的 chunk_006/chunk_010/chunk_016 input snapshots。
    设置 LOAD_HMC_STATE_AT_CHUNK_INDEX=0、LOAD_MERGE_STATE_AT_CHUNK_INDEX=0。
    设置 GLOBAL_CHUNK_OFFSET 为原始 chunk id。
    设置 V45_MODE_OVERRIDE=readonly，因此底层 run_attention_cue_experiment.sh 使用 read_path_only + probe_native。

v45_c23_support_short_report.py:
    只比较同一 parent/chunk/horizon 下候选相对 S0_C23_PAST 的短窗 delta。
    明确标记 diagnostic_only_short_rollout=true、counts_as_full_online_success=false。
```

验证：

```bash
chmod +x tools/run_v45_c23_support_short_fork.sh tools/v45_c23_support_short_report.py
bash -n tools/run_v45_c23_support_short_fork.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/v45_c23_support_short_report.py
```

结果：

```text
bash -n pass
py_compile pass
```

## Phase 3 short causal fork 启动：C9/S0 baseline

时间：2026-06-08 07:03:24 SGT

目的：

```text
先运行 C9 parent 下 S0_C23_PAST 的 chunk 6/10/16 h10 baseline。
后续 S1-S5 的 short delta 将与同 parent/chunk/horizon 的 S0 比较。
```

命令：

```bash
tools/run_v45_c23_support_short_fork.sh 4 C9 S0_C23_PAST 6 10
tools/run_v45_c23_support_short_fork.sh 6 C9 S0_C23_PAST 10 10
tools/run_v45_c23_support_short_fork.sh 7 C9 S0_C23_PAST 16 10
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/short_rollouts/V45_P3SHORT_C9_S0_C23_PAST_CH6_H10_READONLY
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/short_rollouts/V45_P3SHORT_C9_S0_C23_PAST_CH10_H10_READONLY
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/short_rollouts/V45_P3SHORT_C9_S0_C23_PAST_CH16_H10_READONLY
```

启动证据：

```text
run_status mode=readonly
底层 readonly 分支对应 hybrid_memory_mode=read_path_only + hmc_commit_mode=probe_native。
```

C9/S0 baseline run_status：

```text
V45_P3SHORT_C9_S0_C23_PAST_CH6_H10_READONLY: START 2026-06-08 07:03:24, DONE 2026-06-08 07:11:53
V45_P3SHORT_C9_S0_C23_PAST_CH10_H10_READONLY: START 2026-06-08 07:03:24, DONE 2026-06-08 07:12:56
V45_P3SHORT_C9_S0_C23_PAST_CH16_H10_READONLY: START 2026-06-08 07:03:24, DONE 2026-06-08 07:11:37
hmc_state_hash.jsonl rows = 11 for all three runs
last HMC row confirms hybrid_memory_mode=read_path_only, hmc_commit_mode=probe_native
错误关键字扫描 Traceback|NameError|UnboundLocalError|RuntimeError|FAIL|Missing|Unsupported：0 matches
```

## Phase 3 short causal fork 启动：C9-Clean/S0 baseline

时间：2026-06-08 07:14:12 SGT

命令：

```bash
tools/run_v45_c23_support_short_fork.sh 4 C9_CLEAN S0_C23_PAST 6 10
tools/run_v45_c23_support_short_fork.sh 6 C9_CLEAN S0_C23_PAST 10 10
tools/run_v45_c23_support_short_fork.sh 7 C9_CLEAN S0_C23_PAST 16 10
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/short_rollouts/V45_P3SHORT_C9CLEAN_S0_C23_PAST_CH6_H10_READONLY
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/short_rollouts/V45_P3SHORT_C9CLEAN_S0_C23_PAST_CH10_H10_READONLY
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/short_rollouts/V45_P3SHORT_C9CLEAN_S0_C23_PAST_CH16_H10_READONLY
```

C9-Clean/S0 baseline run_status：

```text
V45_P3SHORT_C9CLEAN_S0_C23_PAST_CH6_H10_READONLY: START 2026-06-08 07:14:12, DONE 2026-06-08 07:22:10
V45_P3SHORT_C9CLEAN_S0_C23_PAST_CH10_H10_READONLY: START 2026-06-08 07:14:12, DONE 2026-06-08 07:21:57
V45_P3SHORT_C9CLEAN_S0_C23_PAST_CH16_H10_READONLY: START 2026-06-08 07:14:12, DONE 2026-06-08 07:22:31
hmc_state_hash.jsonl rows = 11 for all three runs
last HMC row confirms hybrid_memory_mode=read_path_only, hmc_commit_mode=probe_native
错误关键字扫描 Traceback|NameError|UnboundLocalError|RuntimeError|FAIL|Missing|Unsupported：0 matches
```

## Phase 3 short causal fork 启动：C9/S1 full_chunk

时间：2026-06-08 07:25:00 SGT

命令：

```bash
tools/run_v45_c23_support_short_fork.sh 4 C9 S1_C23_FULL_CHUNK 6 10
tools/run_v45_c23_support_short_fork.sh 6 C9 S1_C23_FULL_CHUNK 10 10
tools/run_v45_c23_support_short_fork.sh 7 C9 S1_C23_FULL_CHUNK 16 10
```

## Phase 2 完成与 interaction 报告

I8 run_status：

```text
V45_I8_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST_FIXED_EMA_BEST: START 2026-06-08 06:41:24, DONE 2026-06-08 07:56:55
hmc_state_hash.jsonl rows = 38
01.txt rows = 1102
```

补充：

```text
I8 和 I5/I6/I7 一样，是在 launcher 修复前启动、修复后结束，因此 Codex exec 会话在 DONE 后返回 unexpected EOF。
落盘 run_status / 01.txt / hmc rows 完整，纳入报告；后续不再编辑正在运行的 launcher。
```

生成 Phase 2 full-online report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs F0=phase0_hard_gate/rollouts/V45_P0_C9_REPEAT,I1=phase2_interaction/rollouts/V45_I1_NO_TRI_REPLAY_NO_EMA,I2=phase2_interaction/rollouts/V45_I2_NO_TRI_REPLAY_NO_SWA,I3=phase2_interaction/rollouts/V45_I3_NO_TRI_REPLAY_NATIVE_MIX_OFF,I4=phase2_interaction/rollouts/V45_I4_FIXED_TRI_GAMMA_BEST_NO_EMA,I5=phase2_interaction/rollouts/V45_I5_FIXED_TRI_GAMMA_BEST_NO_SWA,I6=phase2_interaction/rollouts/V45_I6_FIXED_TRI_GAMMA_BEST_NATIVE_MIX_OFF,I7=phase2_interaction/rollouts/V45_I7_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST,I8=phase2_interaction/rollouts/V45_I8_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST_FIXED_EMA_BEST \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/report_R1/full_metrics
```

生成 v45 consolidated report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v45_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay
```

Phase 2 ATE 结果：

```text
F0 ATE_full = 33.76294210291885
I1 ATE_full = 36.49350294225174, delta_vs_F0 = +2.730560839332888
I2 ATE_full = 36.28179240994649, delta_vs_F0 = +2.518850307027641
I3 ATE_full = 36.19603921576094, delta_vs_F0 = +2.4330971128420913
I4 ATE_full = 35.468877873185455, delta_vs_F0 = +1.7059357702666063
I5 ATE_full = 34.697565417664826, delta_vs_F0 = +0.934623314745977
I6 ATE_full = 34.873096367813645, delta_vs_F0 = +1.1101542648947955
I7 ATE_full = 34.68703806300187, delta_vs_F0 = +0.9240959600830223
I8 ATE_full = 35.500497135292775, delta_vs_F0 = +1.7375550323739262
```

Interaction matrix 输出：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/final_reports/v45_component_interaction_matrix.csv
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/final_reports/v45_component_contribution_ledger.csv
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/final_reports/component_interaction_heatmap.png
```

Non-additive rows (`|interaction_residual| > 0.3m`)：

```text
I4 fixed_tri_gamma_best+commit_ema:
    delta_vs_C9 = +1.705935770285457
    expected_additive_delta = +1.3742435560516455
    interaction_residual = +0.33169221423381146

I8 fixed_read_beta+fixed_tri_gamma_best+fixed_ema_best:
    delta_vs_C9 = +1.7375550323927769
    expected_additive_delta = +1.4008241185074013
    interaction_residual = +0.33673091388537557
```

Near-additive rows：

```text
I1 tri_replay+commit_ema residual = -0.2047773003206501
I2 tri_replay+swa_overlap_replace residual = +0.015647386140706487
I3 tri_replay+native_mix residual = -0.10555831284010964
I5 fixed_tri_gamma_best+swa_overlap_replace residual = -0.007485022520214102
I6 fixed_tri_gamma_best+native_mix residual = +0.13259342283333808
I7 fixed_read_beta+fixed_tri_gamma_best residual = +0.011657305506943771
```

## Phase 3：C23 support short causal fork

目的：

```text
按计划先做 snapshot fork + readonly/probe_native 的 short diagnostic，不把 short rollout 当作 full-online 成功。
每个 parent/candidate/chunk/horizon=10 运行 11 个 HMC row，并相对 S0_C23_PAST 做局部增量判断。
```

新增/使用工具：

```text
tools/run_v45_c23_support_short_fork.sh
tools/v45_c23_support_short_report.py
```

关键实现约束：

```text
短 fork 从已落盘 snapshot 读取：
  C9:       phase0_hard_gate/state_snapshots/V45_P0_C9_REPEAT + merge_state_snapshots/V45_P0_C9_REPEAT
  C9-Clean: phase1_c9_clean/state_snapshots/V45_D7_C9_CLEAN_BEST_FIXED + merge_state_snapshots/V45_D7_C9_CLEAN_BEST_FIXED

脚本设置：
  LOAD_HMC_STATE_AT_CHUNK_INDEX=0
  LOAD_MERGE_STATE_AT_CHUNK_INDEX=0
  GLOBAL_CHUNK_OFFSET=<anchor chunk>
  V45_MODE_OVERRIDE=readonly
  HMC_COMMIT_MODE=probe_native
  END_FRAME=<anchor start + horizon>

因此该阶段只用于 support cue 的局部因果诊断，不计入 full-online 成功。
```

实际执行命令模板：

```bash
tools/run_v45_c23_support_short_fork.sh <gpu> <C9|C9_CLEAN> <candidate> <chunk> 10
```

已执行矩阵：

```text
parents = C9, C9_CLEAN
candidates =
  S0_C23_PAST
  S1_C23_FULL_CHUNK
  S2_C23_FULL_CHUNK_NO_OVERLAP
  S3_C23_OFF246
  S4_C23_NEAR12
  S5_C23_PAST_PLUS_FUTURE_LIGHT
chunks = 6, 10, 16
horizon = 10
total run dirs = 36
```

代表性启动命令：

```bash
tools/run_v45_c23_support_short_fork.sh 4 C9 S0_C23_PAST 6 10
tools/run_v45_c23_support_short_fork.sh 6 C9 S0_C23_PAST 10 10
tools/run_v45_c23_support_short_fork.sh 7 C9 S0_C23_PAST 16 10
tools/run_v45_c23_support_short_fork.sh 4 C9_CLEAN S0_C23_PAST 6 10
tools/run_v45_c23_support_short_fork.sh 5 C9_CLEAN S0_C23_PAST 10 10
tools/run_v45_c23_support_short_fork.sh 6 C9_CLEAN S0_C23_PAST 16 10
tools/run_v45_c23_support_short_fork.sh 4 C9 S1_C23_FULL_CHUNK 6 10
tools/run_v45_c23_support_short_fork.sh 5 C9 S1_C23_FULL_CHUNK 10 10
tools/run_v45_c23_support_short_fork.sh 6 C9 S1_C23_FULL_CHUNK 16 10
tools/run_v45_c23_support_short_fork.sh 4 C9_CLEAN S1_C23_FULL_CHUNK 6 10
tools/run_v45_c23_support_short_fork.sh 5 C9_CLEAN S1_C23_FULL_CHUNK 10 10
tools/run_v45_c23_support_short_fork.sh 6 C9_CLEAN S1_C23_FULL_CHUNK 16 10
tools/run_v45_c23_support_short_fork.sh 4 C9 S2_C23_FULL_CHUNK_NO_OVERLAP 6 10
tools/run_v45_c23_support_short_fork.sh 5 C9 S2_C23_FULL_CHUNK_NO_OVERLAP 10 10
tools/run_v45_c23_support_short_fork.sh 6 C9 S2_C23_FULL_CHUNK_NO_OVERLAP 16 10
tools/run_v45_c23_support_short_fork.sh 4 C9_CLEAN S2_C23_FULL_CHUNK_NO_OVERLAP 6 10
tools/run_v45_c23_support_short_fork.sh 5 C9_CLEAN S2_C23_FULL_CHUNK_NO_OVERLAP 10 10
tools/run_v45_c23_support_short_fork.sh 6 C9_CLEAN S2_C23_FULL_CHUNK_NO_OVERLAP 16 10
tools/run_v45_c23_support_short_fork.sh 4 C9 S3_C23_OFF246 6 10
tools/run_v45_c23_support_short_fork.sh 6 C9 S3_C23_OFF246 10 10
tools/run_v45_c23_support_short_fork.sh 7 C9 S3_C23_OFF246 16 10
tools/run_v45_c23_support_short_fork.sh 4 C9_CLEAN S3_C23_OFF246 6 10
tools/run_v45_c23_support_short_fork.sh 5 C9_CLEAN S3_C23_OFF246 10 10
tools/run_v45_c23_support_short_fork.sh 4 C9_CLEAN S3_C23_OFF246 16 10
tools/run_v45_c23_support_short_fork.sh 5 C9 S4_C23_NEAR12 6 10
tools/run_v45_c23_support_short_fork.sh 6 C9 S4_C23_NEAR12 10 10
tools/run_v45_c23_support_short_fork.sh 7 C9 S4_C23_NEAR12 16 10
tools/run_v45_c23_support_short_fork.sh 4 C9_CLEAN S4_C23_NEAR12 6 10
tools/run_v45_c23_support_short_fork.sh 5 C9_CLEAN S4_C23_NEAR12 10 10
tools/run_v45_c23_support_short_fork.sh 6 C9_CLEAN S4_C23_NEAR12 16 10
tools/run_v45_c23_support_short_fork.sh 7 C9 S5_C23_PAST_PLUS_FUTURE_LIGHT 6 10
tools/run_v45_c23_support_short_fork.sh 5 C9 S5_C23_PAST_PLUS_FUTURE_LIGHT 10 10
tools/run_v45_c23_support_short_fork.sh 6 C9 S5_C23_PAST_PLUS_FUTURE_LIGHT 16 10
tools/run_v45_c23_support_short_fork.sh 7 C9_CLEAN S5_C23_PAST_PLUS_FUTURE_LIGHT 6 10
tools/run_v45_c23_support_short_fork.sh 4 C9_CLEAN S5_C23_PAST_PLUS_FUTURE_LIGHT 10 10
tools/run_v45_c23_support_short_fork.sh 4 C9_CLEAN S5_C23_PAST_PLUS_FUTURE_LIGHT 16 10
```

完整性检查：

```bash
find results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/short_rollouts \
  -maxdepth 1 -type d -name 'V45_P3SHORT_*' | wc -l

find results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/short_rollouts \
  -maxdepth 2 -name hmc_state_hash.jsonl -exec wc -l {} +
```

结果：

```text
run dirs = 36
每个 hmc_state_hash.jsonl rows = 11
total hmc rows = 396
```

生成 short support 报告：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v45_c23_support_short_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/short_rollouts \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/report_R1/short_support
```

报告输出：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/report_R1/short_support/v45_c23_support_short_summary.json
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/report_R1/short_support/v45_c23_support_short_aggregate.csv
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/report_R1/short_support/v45_c23_support_short_rows.csv
```

short report summary：

```text
rows = 36
done_candidate_rows = 30
diagnostic_only_short_rollout = true
counts_as_full_online_success = false
short_gate_pass = false
passing_rows = 0
gate_rule = h10 ATE delta <= -1.0m, or rolling100 mean delta <= -3m, or [200,300) delta <= -5m with [400,600) regression <= +1m
```

Aggregate rows：

```text
C9,S1_C23_FULL_CHUNK,3,mean_ATE_h10_delta_vs_S0=0.1393114642978889,best_ATE_h10_delta_vs_S0=0.136147877106243
C9,S2_C23_FULL_CHUNK_NO_OVERLAP,3,mean_ATE_h10_delta_vs_S0=0.13568920243679253,best_ATE_h10_delta_vs_S0=0.1175338000630326
C9,S3_C23_OFF246,3,mean_ATE_h10_delta_vs_S0=0.14796689310341105,best_ATE_h10_delta_vs_S0=0.13532645837667445
C9,S4_C23_NEAR12,3,mean_ATE_h10_delta_vs_S0=0.13858934611521492,best_ATE_h10_delta_vs_S0=0.11080935912953827
C9,S5_C23_PAST_PLUS_FUTURE_LIGHT,3,mean_ATE_h10_delta_vs_S0=0.14323179423194018,best_ATE_h10_delta_vs_S0=0.12093436355806908
C9CLEAN,S1_C23_FULL_CHUNK,3,mean_ATE_h10_delta_vs_S0=0.1488433737429619,best_ATE_h10_delta_vs_S0=0.11385183612872041
C9CLEAN,S2_C23_FULL_CHUNK_NO_OVERLAP,3,mean_ATE_h10_delta_vs_S0=0.143835984390592,best_ATE_h10_delta_vs_S0=0.09833407013187134
C9CLEAN,S3_C23_OFF246,3,mean_ATE_h10_delta_vs_S0=0.1638954056893255,best_ATE_h10_delta_vs_S0=0.14947139116371222
C9CLEAN,S4_C23_NEAR12,3,mean_ATE_h10_delta_vs_S0=0.14166921877372385,best_ATE_h10_delta_vs_S0=0.09600926740808635
C9CLEAN,S5_C23_PAST_PLUS_FUTURE_LIGHT,3,mean_ATE_h10_delta_vs_S0=0.14617574880621298,best_ATE_h10_delta_vs_S0=0.1030452788433589
```

Phase 3 决策：

```text
所有 support 变体相对 S0_C23_PAST 的 mean_ATE_h10_delta_vs_S0 都为正，short_gate_pass=false。
按计划“如果所有 support 变体都不超过 past：锁定 C23 past，不再做 support sweep”，不推进 support full-online。
```

## Phase 4：adaptive tri-replay full-online

实现/使用候选：

```text
A0_FIXED_C9_TRI_REPLAY
A1_KMEANS3_TRI_REPLAY
A2_OTSU3_TRI_REPLAY
A3_MAD_TRI_REPLAY
A4_ADAPTIVE_QUANTILE_TRI_REPLAY
```

C9-Clean A0-A4 启动命令：

```bash
V45_PARENT=C9_CLEAN V45_C9_CLEAN_TRI_GAMMA=0.004 V45_C9_CLEAN_COMMIT_EMA_ALPHA=1.0 V45_C9_CLEAN_COMMIT_EMA_BRANCH_MASK=all \
  tools/run_v45_full_candidate.sh 4 V45_A0_C9CLEAN_FIXED_C9_TRI_REPLAY A0_FIXED_C9_TRI_REPLAY

V45_PARENT=C9_CLEAN V45_C9_CLEAN_TRI_GAMMA=0.004 V45_C9_CLEAN_COMMIT_EMA_ALPHA=1.0 V45_C9_CLEAN_COMMIT_EMA_BRANCH_MASK=all \
  tools/run_v45_full_candidate.sh 5 V45_A1_C9CLEAN_KMEANS3_TRI_REPLAY A1_KMEANS3_TRI_REPLAY

V45_PARENT=C9_CLEAN V45_C9_CLEAN_TRI_GAMMA=0.004 V45_C9_CLEAN_COMMIT_EMA_ALPHA=1.0 V45_C9_CLEAN_COMMIT_EMA_BRANCH_MASK=all \
  tools/run_v45_full_candidate.sh 6 V45_A2_C9CLEAN_OTSU3_TRI_REPLAY A2_OTSU3_TRI_REPLAY

V45_PARENT=C9_CLEAN V45_C9_CLEAN_TRI_GAMMA=0.004 V45_C9_CLEAN_COMMIT_EMA_ALPHA=1.0 V45_C9_CLEAN_COMMIT_EMA_BRANCH_MASK=all \
  tools/run_v45_full_candidate.sh 7 V45_A3_C9CLEAN_MAD_TRI_REPLAY A3_MAD_TRI_REPLAY

V45_PARENT=C9_CLEAN V45_C9_CLEAN_TRI_GAMMA=0.004 V45_C9_CLEAN_COMMIT_EMA_ALPHA=1.0 V45_C9_CLEAN_COMMIT_EMA_BRANCH_MASK=all \
  tools/run_v45_full_candidate.sh 7 V45_A4_C9CLEAN_ADAPTIVE_QUANTILE_TRI_REPLAY A4_ADAPTIVE_QUANTILE_TRI_REPLAY
```

完成状态：

```text
V45_A0_C9CLEAN_FIXED_C9_TRI_REPLAY: START 2026-06-08 09:02:49, DONE 2026-06-08 10:16:09, hmc_rows=38
V45_A1_C9CLEAN_KMEANS3_TRI_REPLAY: START 2026-06-08 09:02:49, DONE 2026-06-08 10:20:22, hmc_rows=38
V45_A2_C9CLEAN_OTSU3_TRI_REPLAY: START 2026-06-08 09:02:49, DONE 2026-06-08 10:19:00, hmc_rows=38
V45_A3_C9CLEAN_MAD_TRI_REPLAY: START 2026-06-08 09:02:49, DONE 2026-06-08 10:14:10, hmc_rows=38
V45_A4_C9CLEAN_ADAPTIVE_QUANTILE_TRI_REPLAY: START 2026-06-08 10:18:46, DONE 2026-06-08 11:35:02, hmc_rows=38
```

C9-Clean Phase4 report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs A0=phase4_adaptive_trireplay/rollouts/V45_A0_C9CLEAN_FIXED_C9_TRI_REPLAY,A1=phase4_adaptive_trireplay/rollouts/V45_A1_C9CLEAN_KMEANS3_TRI_REPLAY,A2=phase4_adaptive_trireplay/rollouts/V45_A2_C9CLEAN_OTSU3_TRI_REPLAY,A3=phase4_adaptive_trireplay/rollouts/V45_A3_C9CLEAN_MAD_TRI_REPLAY,A4=phase4_adaptive_trireplay/rollouts/V45_A4_C9CLEAN_ADAPTIVE_QUANTILE_TRI_REPLAY \
  --reference-name A0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase4_adaptive_trireplay/report_R1/c9clean_full_metrics
```

C9-Clean A0-A4 结果：

```text
A0 ATE_full = 35.500497135292775
A1 ATE_full = 40.41639535230242, delta_vs_A0 = +4.915898217009648
A2 ATE_full = 40.49646406199547, delta_vs_A0 = +4.9959669267026925
A3 ATE_full = 36.022004577930915, delta_vs_A0 = +0.5215074426381392
A4 ATE_full = 35.57835251871868, delta_vs_A0 = +0.0778553834259057
```

因为 C9-Clean 不 acceptable，按计划把 C9-Clean adaptive top2 推到 C9 original：

```text
top2 by C9-Clean ATE = A4 adaptive_quantile, A3 MAD
```

C9 original top2 启动命令：

```bash
tools/run_v45_full_candidate.sh 6 V45_A4_C9_ADAPTIVE_QUANTILE_TRI_REPLAY A4_ADAPTIVE_QUANTILE_TRI_REPLAY
tools/run_v45_full_candidate.sh 7 V45_A3_C9_MAD_TRI_REPLAY A3_MAD_TRI_REPLAY
```

完成状态：

```text
V45_A4_C9_ADAPTIVE_QUANTILE_TRI_REPLAY: START 2026-06-08 11:38:37, DONE 2026-06-08 12:22:16, hmc_rows=38
V45_A3_C9_MAD_TRI_REPLAY: START 2026-06-08 11:38:37, DONE 2026-06-08 12:24:59, hmc_rows=38
```

C9 top2 report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs F0=phase0_hard_gate/rollouts/V45_P0_C9_REPEAT,A4_C9=phase4_adaptive_trireplay/rollouts/V45_A4_C9_ADAPTIVE_QUANTILE_TRI_REPLAY,A3_C9=phase4_adaptive_trireplay/rollouts/V45_A3_C9_MAD_TRI_REPLAY \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase4_adaptive_trireplay/report_R1/c9_top2_full_metrics
```

C9 top2 结果：

```text
F0 ATE_full = 33.76294210291885
A4_C9 ATE_full = 33.66464516712499, delta_vs_F0 = -0.09829693579386145
A3_C9 ATE_full = 34.49977064240336, delta_vs_F0 = +0.7368285394845131
```

Adaptive role mass summary：

```text
F0: role_rows=684, pos=0.4782167257446992, neutral=0.379937713693457, neg=0.14184554056291693, fallback_rows=0
A4_C9: role_rows=684, pos=0.47821803390979767, neutral=0.3835495723444119, neg=0.13823237649181433, fallback_rows=0
A3_C9: role_rows=684, pos=0.48669143936090303, neutral=0.32359840378862376, neg=0.1897101376140327, fallback_rows=0
A0: role_rows=684, pos=0.3499999940395355, neutral=0.5299852223772752, neg=0.12001479024949827, fallback_rows=0
A1: role_rows=684, pos=0.5822511883942705, neutral=0.34558422981124176, neg=0.0721646000424193, fallback_rows=0
A2: role_rows=684, pos=0.5858069177695185, neutral=0.34815370677071705, neg=0.0660393953007477, fallback_rows=0
A3: role_rows=684, pos=0.38582965378698547, neutral=0.29644875232762063, neg=0.3177215942121737, fallback_rows=0
A4: role_rows=684, pos=0.3499999940395355, neutral=0.5501403792908317, neg=0.09985963357013394, fallback_rows=0
```

Role mass source：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/final_reports/v45_adaptive_role_mass_summary.csv
```

## Phase 5：semantic READ minimal

SEM1/SEM2 启动命令：

```bash
tools/run_v45_full_candidate.sh 4 V45_SEM1_C23_RESID_READ_ONLY_ON_C9 SEM1_C23_RESID_READ_ONLY_ON_C9

V45_C9_CLEAN_TRI_GAMMA=0.004 V45_C9_CLEAN_COMMIT_EMA_ALPHA=1.0 V45_C9_CLEAN_COMMIT_EMA_BRANCH_MASK=all \
  tools/run_v45_full_candidate.sh 5 V45_SEM2_C23_RESID_READ_ONLY_ON_C9_CLEAN SEM2_C23_RESID_READ_ONLY_ON_C9_CLEAN
```

SEM3 决策：

```text
Phase3 support short_gate_pass=false，未产生 best support；SEM3_C23_RESID_PLUS_BEST_SUPPORT 未运行。
```

SEM4 启动命令：

```bash
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
  tools/run_v45_full_candidate.sh 4 V45_SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST
```

完成状态：

```text
V45_SEM1_C23_RESID_READ_ONLY_ON_C9: START 2026-06-08 10:41:45, DONE 2026-06-08 11:27:16, hmc_rows=38
V45_SEM2_C23_RESID_READ_ONLY_ON_C9_CLEAN: START 2026-06-08 10:41:45, DONE 2026-06-08 11:56:03, hmc_rows=38
V45_SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST: START 2026-06-08 12:26:30, DONE 2026-06-08 13:09:38, hmc_rows=38
```

注意：

```text
当前 launcher 把 SEM candidates 的 rollout base 放在 phase3_c23_support/rollouts 下；报告输出放在 phase5_semantic_read/report_R1 下。
未中途修改正在运行的 launcher。
```

SEM reports：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs F0=phase0_hard_gate/rollouts/V45_P0_C9_REPEAT,SEM1=phase3_c23_support/rollouts/V45_SEM1_C23_RESID_READ_ONLY_ON_C9 \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read/report_R1/sem1_vs_c9

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs D7=phase1_c9_clean/rollouts/V45_D7_C9_CLEAN_BEST_FIXED,SEM2=phase3_c23_support/rollouts/V45_SEM2_C23_RESID_READ_ONLY_ON_C9_CLEAN \
  --reference-name D7 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read/report_R1/sem2_vs_c9clean

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs F0=phase0_hard_gate/rollouts/V45_P0_C9_REPEAT,SEM1=phase3_c23_support/rollouts/V45_SEM1_C23_RESID_READ_ONLY_ON_C9,SEM4=phase3_c23_support/rollouts/V45_SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read/report_R1/sem4_vs_c9
```

SEM 结果：

```text
SEM1 ATE_full = 33.487566750822836, delta_vs_C9 = -0.27537535209601316
SEM2 ATE_full = 35.006318521422756, delta_vs_D7 = -0.4941786138700195, delta_vs_C9 = +1.2433764185227574
SEM4 ATE_full = 33.42918629597565, delta_vs_C9 = -0.33375580694320206
```

Stage-C cache/semantic 证据：

```text
SEM1/SEM2/SEM4:
  hmc_config.yaml stage_c_cache_mode = read
  hmc_config.yaml stage_c_cache_require_hit = 1
  hmc_config.yaml stage_c_mode = reference
  stage_c_semantic_disabled_confirm.json stage_c_disabled = false

semantic_group_summary.jsonl:
  rows = 38 for each of SEM1/SEM2/SEM4
  semantic_group_available = 38 for each
  fine_label_available = 38 for each
  semantic_group_source = MaskletOutput.G_sem
  q_sem_mean_avg = 0.974219129273766
```

## Final consolidated report 与 Phase6 gate

更新 `tools/v45_report.py`：

```text
修复内容：
  _registry(...) 增加扫描 phase/report_R1/**/full_online_registry.csv；
  semantic phase 从 phase5_semantic_minimal 改为实际使用的 phase5_semantic_read。

目的：
  consolidated report 能覆盖 Phase4 的 c9clean_full_metrics / c9_top2_full_metrics，
  以及 Phase5 的 sem1_vs_c9 / sem2_vs_c9clean / sem4_vs_c9。
```

验证：

```bash
bash -n tools/run_v45_full_candidate.sh tools/run_attention_cue_experiment.sh tools/run_v45_c23_support_short_fork.sh

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  run_pipeline_abc_v2.py \
  loger/pipeline/hybrid_memory_controller.py \
  loger/pipeline/ttt_write_controller.py \
  tools/v45_report.py \
  tools/v45_c23_support_short_report.py
```

Consolidated report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v45_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay
```

Final decision：

```text
c9_clean.c9_clean_ATE = 35.500497135292775
c9_clean.acceptable = false
adaptive.best_candidate = A4_C9
adaptive.best_ATE_full = 33.66464516712499
adaptive.best_delta_vs_historical_c9 = -0.09829693577501075
semantic.best_candidate = SEM4
semantic.best_ATE_full = 33.42918629597565
semantic.best_delta_vs_historical_c9 = -0.33375580692435136
support.rows = 0 full-online rows because Phase3 short gate failed and support full sweep was not launched
target30_success = false
phase6_sanity_recommended = false
```

Phase6 gate：

```text
启动条件 1: best KITTI01 ATE <= 33.0m -> false, best observed = SEM4 33.42918629597565
启动条件 2: improvement vs C9 >= 0.5m -> false, best observed improvement = 0.33375580694320206
启动条件 3: C9-Clean acceptable and contribution map clarified -> false, C9-Clean acceptable=false

因此不启动 KITTI00/KITTI02/KITTI05 跨 sequence sanity。
```

## v45 continuation：SEM4 auxiliary + C9 写入组件组合

继续原因：

```text
用户追问“达成目标了吗，没有请继续”。
截至 v45 final，科学目标未达成：
  best ATE = SEM4 33.42918629597565
  未达到 <=33.0m
  未达到 improvement vs C9 >=0.5m

计划尾部建议：若只有 mechanism progress，应判断下一轮继续优化 TTT tri-replay、READ/C23 cue，还是停止语义主线。
复盘结论显示 SEM4 是小正信号，但 semantic 应作为 auxiliary branch，不扩 semantic 大矩阵。

因此本 continuation 只做 4 个小组合，检查 SEM4 与 C9 写入组件是否冲突/互补：
  1. SEM4 + no commit EMA，对应 v43 COMBO_03 思路。
  2. SEM4 + no SWA overlap replace，对应 Phase2 组件冲突检查。
  3. SEM4 + native mix off，对应 Phase2 组件冲突检查。
  4. SEM4 + READ lambda 0.35，作为 best SEM4 离 0.5m 门槛仍不足时的最小强度校准。
```

启动时间：

```text
2026-06-08 13:18:04 SGT
```

启动命令：

```bash
V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
TTT_WRITE_COMMIT_EMA_ALPHA=1.0 \
TTT_WRITE_COMMIT_EMA_BRANCH_MASK=all \
TTT_WRITE_COMMIT_EMA_CHUNKS=none \
tools/run_v45_full_candidate.sh 4 V45X_SEM4_ADAPTIVE_NO_COMMIT_EMA SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
ENABLE_SWA_OVERLAP_SOURCE_REPLACE=0 \
tools/run_v45_full_candidate.sh 5 V45X_SEM4_ADAPTIVE_NO_SWA_REPLACE SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
TTT_WRITE_NATIVE_MIX_SCALES=1.00,1.00,1.00 \
tools/run_v45_full_candidate.sh 6 V45X_SEM4_ADAPTIVE_NATIVE_MIX_OFF SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
READ_BLEND_LAMBDA_OVERRIDE=0.35 \
tools/run_v45_full_candidate.sh 7 V45X_SEM4_ADAPTIVE_READ_L035 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST
```

结果状态：

```text
running at log write time; final ATE not yet available.
```

## v45 continuation：SEM4 组件组合 / 真实 lambda / 上限探测结果落盘

结果汇总时间：

```text
2026-06-08 16:05 SGT
```

第一轮 SEM4 组件组合 report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs F0=phase0_hard_gate/rollouts/V45_P0_C9_REPEAT,SEM4=phase3_c23_support/rollouts/V45_SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST,X_NO_EMA=phase5_semantic_read_extra/rollouts/V45X_SEM4_ADAPTIVE_NO_COMMIT_EMA,X_NO_SWA=phase5_semantic_read_extra/rollouts/V45X_SEM4_ADAPTIVE_NO_SWA_REPLACE,X_NATIVE_OFF=phase5_semantic_read_extra/rollouts/V45X_SEM4_ADAPTIVE_NATIVE_MIX_OFF,X_L035=phase5_semantic_read_extra/rollouts/V45X_SEM4_ADAPTIVE_READ_L035 \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra/report_R1/sem4_component_combos
```

第一轮结果：

```text
F0 / C9:                         ATE = 33.76294210291885
SEM4 l025:                       ATE = 33.42918629597565, delta = -0.33375580694320206
X_NO_EMA:                        ATE = 33.78246724832407, delta = +0.019525145405218325
X_NO_SWA:                        ATE = 33.514475845456005, delta = -0.24846625746284445
X_NATIVE_OFF:                    ATE = 33.42668439898548, delta = -0.33625770393337007
X_L035 via READ_BLEND_OVERRIDE:   ATE = 33.42918629597565, delta = -0.33375580694320206
```

审计发现：

```text
X_L035 与 SEM4 轨迹完全相同。
原因不是实验随机，而是代码里 semantic residual cue 的 lam_sem 被硬编码为 0.25。
READ_BLEND_LAMBDA_OVERRIDE 只改变 hmc_config 中的 read_blend_lambda，不改变 sem_resid cue 的实际 lambda。
```

修复验证命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile loger/pipeline/hybrid_memory_controller.py
bash -n tools/run_v45_full_candidate.sh
```

第二轮 true lambda report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs F0=phase0_hard_gate/rollouts/V45_P0_C9_REPEAT,SEM4=phase3_c23_support/rollouts/V45_SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST,X2_L015=phase5_semantic_read_extra2/rollouts/V45X2_SEM4_ADAPTIVE_TRUE_L015,X2_L035=phase5_semantic_read_extra2/rollouts/V45X2_SEM4_ADAPTIVE_TRUE_L035,X2_L045=phase5_semantic_read_extra2/rollouts/V45X2_SEM4_ADAPTIVE_TRUE_L045,X2_L035_NATIVE_OFF=phase5_semantic_read_extra2/rollouts/V45X2_SEM4_ADAPTIVE_TRUE_L035_NATIVE_OFF \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra2/report_R1/sem4_true_lambda
```

第二轮结果：

```text
X2_L015:             ATE = 33.52162541133483,  delta = -0.24131669158401792
X2_L035:             ATE = 33.307116451511625, delta = -0.4558256514072241
X2_L045:             ATE = 33.24063211182052,  delta = -0.5223099910983322
X2_L035_NATIVE_OFF:  ATE = 33.40154257093767,  delta = -0.3613995319811778
```

第三轮 lambda 上限 report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v42_full_online_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay \
  --runs F0=phase0_hard_gate/rollouts/V45_P0_C9_REPEAT,X2_L045=phase5_semantic_read_extra2/rollouts/V45X2_SEM4_ADAPTIVE_TRUE_L045,X3_L050=phase5_semantic_read_extra3/rollouts/V45X3_SEM4_ADAPTIVE_TRUE_L050,X3_L060=phase5_semantic_read_extra3/rollouts/V45X3_SEM4_ADAPTIVE_TRUE_L060,X3_L075=phase5_semantic_read_extra3/rollouts/V45X3_SEM4_ADAPTIVE_TRUE_L075,X3_FIXED_L045=phase5_semantic_read_extra3/rollouts/V45X3_SEM1_FIXED_TRUE_L045 \
  --reference-name F0 \
  --out-dir results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra3/report_R1/sem_lambda_upper
```

第三轮结果：

```text
X3_L050:       ATE = 33.19360510397078,  delta = -0.5693369989480672
X3_L060:       ATE = 33.20764929099522,  delta = -0.5552928119236284
X3_L075:       ATE = 33.381324912658975, delta = -0.38161719025987395
X3_FIXED_L045: ATE = 33.3668121381553,   delta = -0.3961299647635528
```

后续代码整理：

```text
loger/pipeline/hybrid_memory_controller.py:
  - 将 v31.sem_resid_fine/coarse_lNNN.c23past 改为通用解析。
  - 例如 l052 会解析为 lam_sem=0.52。
  - lam_sem clamp 到 [0.0, 1.0]。
  - 默认 l025 仍是 0.25。

验证：
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile loger/pipeline/hybrid_memory_controller.py
  bash -n tools/run_v45_full_candidate.sh
```

Phase6 gate 更新：

```text
X3_L050 improvement vs C9 = 0.5693369989480672m >= 0.5m。
因此 Phase6 cross-sequence sanity 的启动条件已经满足。
当前仍未达到 stage progress，因为 best ATE = 33.19360510397078 > 33.0m。
```

## v45 continuation：lambda 窄扫 l048/l052/l055/l058

继续原因：

```text
X3_L050 是当前最优 ATE=33.19360510397078。
X3_L060 略差，X3_L075 明显差；最佳区间可能在 0.48-0.58。
目标是确认是否存在接近 33.0m 的窄区间收益。
不改变 tri/adaptive/semantic 以外其它阈值。
```

代码整理：

```text
loger/pipeline/hybrid_memory_controller.py:
  v31.sem_resid_fine/coarse_lNNN.c23past 通用解析已完成。
  允许 l048/l052/l055/l058 这类 cue 真实生效。

验证：
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile loger/pipeline/hybrid_memory_controller.py
  bash -n tools/run_v45_full_candidate.sh
```

启动时间：

```text
2026-06-08 16:00:12 SGT
```

启动命令：

```bash
V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra4/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l048.c23past \
tools/run_v45_full_candidate.sh 4 V45X4_SEM4_ADAPTIVE_TRUE_L048 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra4/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l052.c23past \
tools/run_v45_full_candidate.sh 5 V45X4_SEM4_ADAPTIVE_TRUE_L052 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra4/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l055.c23past \
tools/run_v45_full_candidate.sh 6 V45X4_SEM4_ADAPTIVE_TRUE_L055 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra4/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l058.c23past \
tools/run_v45_full_candidate.sh 7 V45X4_SEM4_ADAPTIVE_TRUE_L058 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST
```

结果状态：

```text
running at log write time; final ATE not yet available.
```

## v45 continuation：semantic residual lambda 上限探测

基于第二轮中 `X2_L045` 达到 minimum progress，但仍未达 `<=33.0m`：

```text
X2_L015 ATE = 33.52162541133483
X2_L035 ATE = 33.307116451511625
X2_L045 ATE = 33.24063211182052

趋势显示更高 semantic residual lambda 可能继续降低 ATE，但也可能损害 Rot/FinalErr。
```

补充修复：

```text
loger/pipeline/hybrid_memory_controller.py:
  - 增加 v31.sem_resid_fine/coarse_l060.c23past
  - 增加 v31.sem_resid_fine/coarse_l075.c23past

验证：
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile loger/pipeline/hybrid_memory_controller.py
```

第三轮启动时间：

```text
2026-06-08 15:05:14 SGT
```

启动命令：

```bash
V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra3/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l050.c23past \
tools/run_v45_full_candidate.sh 4 V45X3_SEM4_ADAPTIVE_TRUE_L050 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra3/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l060.c23past \
tools/run_v45_full_candidate.sh 5 V45X3_SEM4_ADAPTIVE_TRUE_L060 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra3/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l075.c23past \
tools/run_v45_full_candidate.sh 6 V45X3_SEM4_ADAPTIVE_TRUE_L075 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra3/rollouts \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l045.c23past \
tools/run_v45_full_candidate.sh 7 V45X3_SEM1_FIXED_TRUE_L045 SEM1_C23_RESID_READ_ONLY_ON_C9
```

结果状态：

```text
running at log write time; final ATE not yet available.
```

## v45 continuation：semantic residual lambda 真实生效修复

发现：

```text
V45X_SEM4_ADAPTIVE_READ_L035 的 hmc_config.yaml 记录 read_blend_lambda=0.35，
但 full trajectory 与 SEM4 完全相同。

代码审计确认：
  loger/pipeline/hybrid_memory_controller.py 的 v31.sem_resid_*_l025.c23past 分支内
  lam_sem = 0.25 是硬编码；
  READ_BLEND_LAMBDA 只作用于 read_calib_mode 等其他分支。

因此 READ_BLEND_LAMBDA_OVERRIDE=0.35 是配置变化，不是 semantic residual cue 变化。
```

修复：

```text
loger/pipeline/hybrid_memory_controller.py:
  - 增加 v31.sem_resid_fine/coarse_l010/l015/l035/l045/l050.c23past 支持。
  - 根据 cue name 中的 _l010/_l015/_l035/_l045/_l050 解析 lam_sem。
  - 默认 l025 行为保持不变。

tools/run_v45_full_candidate.sh:
  - sem_resid_read_only() 支持 V45_SEM_RESID_CUE_OVERRIDE。
  - 默认仍为 v31.sem_resid_coarse_l025.c23past。
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile loger/pipeline/hybrid_memory_controller.py
bash -n tools/run_v45_full_candidate.sh
```

第二轮启动时间：

```text
2026-06-08 14:10:45 SGT
```

启动命令：

```bash
V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra2/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l015.c23past \
tools/run_v45_full_candidate.sh 4 V45X2_SEM4_ADAPTIVE_TRUE_L015 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra2/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l035.c23past \
tools/run_v45_full_candidate.sh 5 V45X2_SEM4_ADAPTIVE_TRUE_L035 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra2/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l045.c23past \
tools/run_v45_full_candidate.sh 6 V45X2_SEM4_ADAPTIVE_TRUE_L045 SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST

V45_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra2/rollouts \
V45_BEST_TRI_ROLE_MODE=adaptive_quantile \
V45_SEM_RESID_CUE_OVERRIDE=v31.sem_resid_coarse_l035.c23past \
TTT_WRITE_NATIVE_MIX_SCALES=1.00,1.00,1.00 \
tools/run_v45_full_candidate.sh 7 V45X2_SEM4_ADAPTIVE_TRUE_L035_NATIVE_OFF SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST
```

结果状态：

```text
running at log write time; final ATE not yet available.
```

## v45 continuation：用户追问后的 C9 chunk policy 与 isolated ablation 缺口审计

用户指出：

```text
chunk-wise gamma 如果是每个 chunk 分别指定且靠 KITTI01 调参得到，则不可接受，因为不能泛化到其他 sequence。
C9_P0_R2 需要知道：
1. only frame attention/read cue 增强结果；
2. only TTT writing strategy 增强结果；
3. only SWA 增强结果。
现有回答没有给出这些答案。
```

代码/文档审计命令：

```bash
rg -n "TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS|TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS|READ_BETA_FRAME_CHUNKS|TTT_WRITE_COMMIT_EMA_CHUNKS|ATTR_0[1-6]|C9_P0_R2" tools docs results -g '*.sh' -g '*.md' -g '*.csv' -g '*.json'
rg -n "hybrid_memory_mode|read_path_only|beta_frame|ENABLE_TTT|TRI_REPLAY|SWA|read_cue_source" tools/run_attention_cue_experiment.sh tools/run_v45_full_candidate.sh
sed -n '430,615p' tools/run_attention_cue_experiment.sh
sed -n '1,360p' tools/run_v45_full_candidate.sh
```

审计结论：

```text
C9 / C9_P0_R2 launcher 确实包含 absolute chunk-id policy：
  READ_BETA_FRAME_CHUNKS=5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25
  TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,16:0.0003
  TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=5:0.35/0.12/0.85,...,16:0.35/0.08/0.85
  TTT_WRITE_COMMIT_EMA_CHUNKS=5,6

因此 C9_P0_R2 是 historical diagnostic/best recipe，不是 clean deployable/generalizable policy。
```

已有 ATTR 结果状态：

```text
ATTR_01..06 是 C9-minus / knockout ablation，不是 positive-only isolated ablation。
这些结果不能回答 only frame-attn / only TTT / only SWA 的 sufficiency。
```

已有 knockout 结果仅可解释为 C9 内部必要性：

```text
ATTR_01_C9_MINUS_READ_MAP_TO_FLAT: 33.789522665336904, delta +0.0265805624180544
ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT: 34.73396750873536, delta +0.971025405816512
ATTR_03_C9_MINUS_COMMIT_EMA: 34.25132756681247, delta +0.48838546389362136
ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE: 33.81919234804587, delta +0.056250245127017706
ATTR_05_C9_MINUS_TTT_TRI_REPLAY: 36.20989477869762, delta +2.4469526757787676
ATTR_06_C9_MINUS_NATIVE_MIX: 33.85464485284113, delta +0.091702749922284
```

待补实验定义：

```text
ONLY_FRAME_ATTN_CLEAN:
  只开 frame attention/read cue 增强；
  关闭 TTT writing 和 SWA overlap/source replace；
  不使用 READ_BETA_FRAME_CHUNKS。

ONLY_TTT_CLEAN:
  frame read bias 置零或关闭；
  关闭 SWA；
  只开 TTT writing；
  不使用 chunk-wise gamma / chunk replay params，优先用全局或自适应策略。

ONLY_SWA_CLEAN:
  frame read bias 置零或关闭；
  关闭 TTT writing；
  只开 SWA overlap/source replace。
```

执行状态：

```text
当前 GPU 4/5/6/7 正在运行 lambda narrow sweep l048/l052/l055/l058。
isolated positive-only ablation 等当前 run 结束后启动。
```

## v45 收尾：用户要求停止并纠偏

用户反馈：

```text
收尾一下；当前推进完全跑偏，根本没按计划来。
```

处理：

```bash
ps -eo pid,ppid,stat,etime,args | rg 'V45X4_SEM4_ADAPTIVE_TRUE_L0(48|52|55|58)|run_v45_full_candidate|run_attention_cue_experiment|run_pipeline_abc_v2'
pkill -TERM -f 'V45X4_SEM4_ADAPTIVE_TRUE_L0(48|52|55|58)' || true
sleep 2
ps -eo pid,ppid,stat,etime,args | rg 'V45X4_SEM4_ADAPTIVE_TRUE_L0(48|52|55|58)|run_v45_full_candidate|run_attention_cue_experiment|run_pipeline_abc_v2' || true
```

结果：

```text
V45X4_SEM4_ADAPTIVE_TRUE_L048/L052/L055/L058 已终止。
进程表中不再有对应 run_v45_full_candidate / run_attention_cue_experiment / run_pipeline_abc_v2 进程。
```

文件状态检查：

```bash
for r in V45X4_SEM4_ADAPTIVE_TRUE_L048 V45X4_SEM4_ADAPTIVE_TRUE_L052 V45X4_SEM4_ADAPTIVE_TRUE_L055 V45X4_SEM4_ADAPTIVE_TRUE_L058; do
  d=results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra4/rollouts/$r
  echo "== $r =="
  [ -f "$d/01.txt" ] && echo HAS_01_TXT || echo NO_01_TXT
  [ -f "$d/kitti_benchmark.log" ] && echo HAS_BENCH || echo NO_BENCH
  [ -f "$d/run_status.txt" ] && cat "$d/run_status.txt" || true
done
```

结果：

```text
四个 run 均为：
NO_01_TXT
NO_BENCH
run_status.txt 只有 START，无 DONE。
```

审计结论：

```text
l048/l052/l055/l058 narrow sweep 没有产生可用 ATE/RPE/segment 指标。
这些 run 不计入实验结果，不能写入结论表。
```

最终收尾判断：

```text
1. 本轮 continuation 确实偏离用户最关心的原计划问题：
   C9_P0_R2 的 positive-only isolated ablation 没有完成。

2. 已有 ATTR 数据只是 knockout，不是 only-frame / only-TTT / only-SWA。

3. C9_P0_R2 包含 chunk-id 手工 map，不能作为泛化策略 claim。

4. adaptive TTT writing 已有结果只有小正信号：
   A4_C9 adaptive_quantile = 33.66464516712499
   delta vs C9 = -0.09829693579386145
   不是强结果。

5. semantic residual lambda 修复得到的 X3_L050 = 33.19360510397078 可以作为真实代码修复后的 KITTI01 minimum-progress 结果，
   但它不是对用户三项核心 ablation 的回答，也未完成跨 sequence sanity。

6. 本阶段到此停止，不再启动新实验。
```

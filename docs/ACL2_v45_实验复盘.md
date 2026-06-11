# ACL2 v45 实验复盘

日期：2026-06-08（Asia/Singapore）  
计划文件：`docs/ACL2_v45_CodeAudit_C9Clean_Attribution_C23AdaptiveTriReplay_Plan.md`  
执行日志：`docs/ACL2_v45_执行日志.md`  
结果根目录：`results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/`

## 当前状态

截至本文件初始化时：

```text
v45 代码准备已完成。
语法检查已通过。
尚未产生 v45 ATE 实验结果。
本复盘后续只记录真实运行得到的数据；没有运行的 phase 标注为 not_run。
```

## 已完成代码修复审计

| 文件 | 修改 | 目的 | 当前验证 |
|---|---|---|---|
| `loger/pipeline/ttt_write_controller.py` | 新增 `tri_replay_role_mode=fixed/kmeans3/otsu3/mad/adaptive_quantile` | 支持 Phase 4 training-free adaptive tri-replay，默认 fixed 保持 C9 行为 | `py_compile` pass |
| `loger/pipeline/ttt_write_controller.py` | 修复非 tri replay 分支并删除 tri return 后 dead code | 处理计划中的 two/separate/pos_neg replay 风险 | `py_compile` pass，待 smoke |
| `loger/pipeline/hybrid_memory_controller.py` | 传入 `ttt_write_tri_replay_role_mode` | 接通 run_pipeline 到 controller | `py_compile` pass |
| `run_pipeline_abc_v2.py` | 新增 CLI `--ttt_write_tri_replay_role_mode` | 允许 launcher 控制 adaptive role mode | `py_compile` pass |
| `tools/run_attention_cue_experiment.sh` | 新增 `TTT_WRITE_TRI_REPLAY_ROLE_MODE` | 允许实验脚本控制 role mode | `bash -n` pass |
| `tools/run_v45_full_candidate.sh` | 新增 v45 launcher | 隔离 v45 输出、记录 effective config 和 chunk-id audit | `bash -n` pass |
| `tools/v45_report.py` | 新增 landed-artifact 汇总 | 生成 v45 报告，不编造缺失结果 | `py_compile` pass |

## Phase 0 Smoke 复盘：TTT replay 风险路径

### 执行摘要

smoke 只用于验证代码路径，不作为 v45 ATE 结论。

| run | candidate | frames | GPU | 状态 | 关键证据 |
|---|---:|---:|---:|---|---|
| `V45_SMOKE_TWO_REPLAY_ACTIVE_64F` | `SMOKE_TWO_REPLAY` | 64 | 4 | DONE | `ttt_two_replay_applied=True` 28 次；`ttt_gradient_reversal_applied=True` 28 次 |
| `V45_SMOKE_A1_KMEANS3_ACTIVE_180F` | `A1_KMEANS3_TRI_REPLAY` | 180 | 5 | DONE | `ttt_tri_replay_role_mode='kmeans3'` 18 次；`tri_replay_role_mass.jsonl` 中 `tri_replay_applied=true` 36 行 |

错误扫描：

```text
V45_SMOKE_TWO_REPLAY_ACTIVE_64F: Traceback/NameError/UnboundLocalError/RuntimeError/FAIL = 0 matches
V45_SMOKE_A1_KMEANS3_ACTIVE_180F: Traceback/NameError/UnboundLocalError/RuntimeError/FAIL = 0 matches
```

Stage C 语义确认：

```text
两个有效 smoke 的 stage_c_semantic_disabled_confirm.json 均显示：
stage_c_mode = none
stage_c_cache_mode = off
semantic_role_policy = none
semantic_memory_paths = ""
stage_c_disabled = true
```

### 修复记录

第一次 64-frame smoke 暴露出一个 launcher 配置问题：

```text
SMOKE_TWO_REPLAY 虽然切到 two_replay mode，但仍继承早先默认 gamma=0.0。
结果：run 完成，但不能证明 two-replay negative update 实际生效。
```

已做修复：

```text
文件：tools/run_v45_full_candidate.sh
修改：SMOKE_TWO_REPLAY 显式设置 TTT_WRITE_GRADIENT_REVERSAL_GAMMA=${V45_SMOKE_TWO_REPLAY_GAMMA:-0.001}
验证：bash -n tools/run_v45_full_candidate.sh pass；随后有效 smoke 中 two-replay/gradient reversal 均出现 28 次 applied=True。
```

### kmeans3 role mass 证据

来源：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_smoke/rollouts/V45_SMOKE_A1_KMEANS3_ACTIVE_180F/v11_projection_trace/tri_replay_role_mass.jsonl
```

统计：

```text
role_rows = 126
tri_true_rows = 36
positive_mass mean/min/max = 0.5786109450790617 / 0.3541005253791809 / 0.6000000238418579
neutral_mass  mean/min/max = 0.35102398610777324 / 0.25145503878593445 / 0.40317460894584656
negative_mass mean/min/max = 0.0703650880087581 / 0.03000991977751255 / 0.24272486567497253
```

分析：

```text
1. `kmeans3` role mode 已接入真实 controller debug 路径，并非只停留在 CLI/config 层。
2. positive_mass 最大值 0.6000000238418579，negative_mass 最小值 0.03000991977751255，符合实现中为避免极端分配而设置的 mass cap/floor 行为。
3. smoke 不足以证明 ATE 收益；只能说明 Phase 4 adaptive tri replay 候选已经具备可运行和可审计基础。
```

### 当前结论

```text
计划中要求先处理的 TTT replay 风险路径已经完成代码修复与 smoke 验证。
后续可以进入 Phase 0 C9_REPEAT hard gate。
仍未产生 v45 full-online ATE；任何 C9-Clean / C23 / AdaptiveTriReplay 结论均保持 not_run。
```

## Phase 0 Hard Gate 复盘：C9 repeat

### 结果表

| 项 | 值 | Gate | 结论 |
|---|---:|---|---|
| `ATE_full` | `33.76294210291885` | `abs(ATE - 33.7629421029) <= 0.03m` | pass |
| `abs_delta_vs_historical_C9` | `1.8850698779715458e-11` | `<= 0.03m` | pass |
| `hmc_rows` | `38` | `== 38` | pass |
| `frames` | `1101` | 完整 KITTI01 估计帧 | pass |
| `stage_c_disabled` | `true` | Stage C semantic off | pass |

来源：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/report_R1/full_metrics/full_online_registry.csv
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/rollouts/V45_P0_C9_REPEAT/hmc_state_hash.jsonl
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/rollouts/V45_P0_C9_REPEAT/stage_c_semantic_disabled_confirm.json
```

### 配置证据链

P0 保留 C9 的 absolute chunk-id maps，这是 C9 repeat 的预期行为：

```text
has_read_beta_frame_chunks = true
has_tri_gamma_chunk_map = true
has_tri_replay_chunk_params = true
has_commit_ema_chunks = true
read_beta_frame_chunks = 5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25
tri_gamma_chunk_map = 5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,16:0.0003
commit_ema_chunks = 5,6
```

运行中抽样证据：

```text
chunk 5 prior_beta_frame_effective = 4.85
chunk 10 prior_beta_frame_effective = 4.25
chunk 16 prior_beta_frame_effective = 4.25
chunk 16 auxgeo_tri_replay_applied_layer_count = 18
```

Stage C 关闭证据：

```text
stage_c_mode = none
stage_c_cache_mode = off
semantic_role_policy = none
semantic_memory_paths = ""
stage_c_disabled = true
```

### Blocker 与修复审计

Phase 0 rollout 本身完成后，生成 report 时遇到 import blocker：

```text
ModuleNotFoundError: No module named 'tools.v18_true_action_report'
```

原因分析：

```text
conda 环境中存在 site-packages/tools 包。
本仓库 tools/ 没有 __init__.py，导致 `from tools.v18_true_action_report` 被第三方 tools 包遮蔽。
```

已做修复：

```text
新增 tools/__init__.py，将本仓库 tools 目录声明为本地 package。
验证 import 指向 /mnt/data/users/chengshun.wang/pjs/LoGeR/tools/v18_true_action_report.py。
验证 py_compile pass。
```

这项修复是环境/导入解析修复，不改变实验算法路径、模型参数或运行结果。

### 分析与结论

```text
1. P0 完整复现 C9：ATE 与历史 C9 基线差值只有 1.885e-11m，远小于 0.03m gate。
2. hmc_rows = 38，说明完整 38 个 online chunks 均落盘；不是短跑或不完整 run。
3. Stage C semantic 关闭，排除了 semantic read/cache 对 C9 repeat 的污染。
4. C9 的 absolute chunk-id maps 在 P0 中确实存在，并在 chunk 5/10/16 的运行证据中体现；这为 Phase 1 的 C9-Clean 去 chunk-id attribution 提供了可靠对照。
5. Phase 0 gate 通过，可以进入 Phase 1 D1-D6。
```

## 待实验 Gate

```text
Phase 0:
    P0_C9_REPEAT 必须满足 abs(ATE - 33.7629421029) <= 0.03m。done/pass
    hmc_rows 必须为 38。done/pass
    Stage C semantic 必须 off。done/pass

Phase 1:
    D1-D6 后选择 D7 C9-Clean 组合。
    D7 不允许含 read beta chunk map、tri gamma chunk map、tri replay chunk params、commit EMA chunks。

后续 Phase 2-6:
    仅基于前序真实结果推进。
```

## Phase 1 复盘：C9-Clean 去 chunk-id 分解

### D1-D6 结果

| 候选 | 改动 | ATE_full | Delta vs P0 | 结论 |
|---|---|---:|---:|---|
| `F0/P0` | C9 repeat reference | `33.76294210291885` | `0.0` | reference |
| `D1` | fixed read beta only | `33.789522665336904` | `+0.0265805624180544` | 最小损伤，但仍变差 |
| `D2` | fixed tri gamma `0.003` | `34.73396750873536` | `+0.971025405816512` | 明显变差 |
| `D3` | fixed tri gamma `0.004` | `34.64880019505802` | `+0.8858580921391734` | tri gamma 固定候选中最小损伤 |
| `D4` | fixed tri gamma `0.005` | `34.77852836133698` | `+1.015586258418132` | 明显变差 |
| `D5` | commit EMA off | `34.25132756681247` | `+0.48838546389362136` | EMA 替代中最小损伤 |
| `D6` | global commit EMA alpha `0.8` | `34.674495366802454` | `+0.9115532638836044` | 明显变差 |

来源：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics/full_online_registry.csv
```

所有 D1-D6 均为完整 run：

```text
D1-D6 hmc_rows 均为 38。
D1-D6 status 均为 done。
D1-D6 frames 均为 1101。
```

### Chunk-id audit 证据

分解项对应的 chunk-id 移除范围符合设计：

```text
D1: has_read_beta_frame_chunks=false；其它 C9 chunk maps 仍为 true。
D2/D3/D4: has_tri_gamma_chunk_map=false 且 has_tri_replay_chunk_params=false；read beta/EMA chunks 仍为 true。
D5/D6: has_commit_ema_chunks=false；read beta/tri chunk maps 仍为 true。
```

### 分析

```text
1. read beta chunk map 的作用最小：移除后 D1 仅比 P0 差 0.02658m，说明 read beta 的 absolute chunk-id tuning 对 C9 成绩贡献有限，但它仍不是无损。
2. tri gamma/chunk replay params 的作用最大：固定 gamma 0.003/0.004/0.005 均造成 +0.89m 到 +1.02m 级别退化，说明 C9 的 tri replay absolute chunk schedule 是主要收益来源之一。
3. commit EMA chunks 也有明显贡献：关闭 EMA 比 P0 差 +0.488m，global alpha 0.8 比 P0 差 +0.912m；在本组候选里 off 优于 global 0.8。
4. 因为所有去 chunk-id 替代项都劣于 P0，C9-Clean 的目标不应被解读为提升 C9，而是构造一个无 absolute chunk-id 的审计基线，供后续 C23/adaptive tri replay 判断泛化风险。
```

### D7 选择

基于真实数据，D7 采用每类替代中的最小 ATE 损伤组合：

```text
fixed read beta: D1 对应全局 4.75。
fixed tri gamma: D3，对应 gamma=0.004。
commit EMA: D5，对应 off；实现为 alpha=1.0, branch_mask=all, chunks=""。
```

D7 启动命令已写入执行日志。D7 当前状态：

```text
V45_D7_C9_CLEAN_BEST_FIXED: done
```

### D7 C9-Clean 结果

| 候选 | ATE_full | Delta vs P0 | hmc_rows | frames | Gate |
|---|---:|---:|---:|---:|---|
| `P0/F0` | `33.76294210291885` | `0.0` | `38` | `1101` | reference |
| `D7_C9_CLEAN_BEST_FIXED` | `35.500497135292775` | `+1.737555032373925` | `38` | `1101` | C9-Clean not acceptable |

Gate 判定：

```text
C9-Clean acceptable threshold = P0 + 0.30 = 34.06294210291885
D7 ATE_full = 35.500497135292775
D7 > threshold，因此 acceptable=false, promising=false, success=false。
```

D7 chunk-id / Stage C audit：

```text
has_read_beta_frame_chunks=false
has_tri_gamma_chunk_map=false
has_tri_replay_chunk_params=false
has_commit_ema_chunks=false
stage_c_disabled=true
```

D7 tri-replay role mass 证据：

```text
positive_mass mean/min/max = 0.3499999940395355 / 0.3499999940395355 / 0.3499999940395355
neutral_mass mean/min/max = 0.5299852223772752 / 0.5299851298332214 / 0.5299886465072632
negative_mass mean/min/max = 0.12001479024949827 / 0.1200113371014595 / 0.12001488357782364
memory_ttt_mean_rel_diff mean/min/max = 0.027798287607620896 / 0.022599587242177453 / 0.035539337733617525
```

分析：

```text
1. D7 的四类 absolute chunk-id policy 均已清空，因此 D7 退化不能归因于 launcher 没清干净。
2. D1 单独移除 read beta chunk map 仅 +0.02658m，但 D7 组合移除 read beta + tri gamma/chunk replay params + EMA chunks 后退化 +1.73756m，说明这些机制存在强交互，不是单组件 delta 的线性相加。
3. 固定 tri gamma 的三条 D2-D4 均接近 +0.9m 到 +1.0m 退化，D7 进一步退化到 +1.73756m，支持 Phase 2 必须做 interaction attribution，而不能只凭 leave-one-out 做贡献解释。
4. C9-Clean 未达到 acceptable，后续 Phase 4 adaptive tri-replay 需遵循计划：先跑 C9-Clean parent；同时因 C9-Clean 不 acceptable，C9 original parent 上还要跑 top 2 adaptive candidates。
```

### Phase3 前代码修复：readonly short fork

修复内容：

```text
tools/run_v45_full_candidate.sh:
    MODE="hybrid"
改为：
    MODE="${V45_MODE_OVERRIDE:-hybrid}"
```

合理性：

```text
Phase 3 计划明确要求 short causal fork 先用 read-only / probe_native。底层 tools/run_attention_cue_experiment.sh 已有 readonly 分支，真实设置为 hybrid_memory_mode=read_path_only 和 hmc_commit_mode=probe_native；v45 launcher 只是缺少把该模式暴露出来的入口。
本修改默认仍为 hybrid，因此不会改变 P0/D/I full-online 已完成 run，也不会影响后续 full-online 候选的默认行为。
```

验证：

```text
bash -n tools/run_v45_full_candidate.sh: pass
```

运行中编辑 launcher 的副作用：

```text
I5/I6/I7 在启动后、结束前，我修改了 tools/run_v45_full_candidate.sh 以暴露 V45_MODE_OVERRIDE。
这导致这些已经运行中的 bash 会话在 run_status DONE 后返回：
    unexpected EOF while looking for matching `"'

复核证据：
    I5/I6/I7 run_status.txt 均为 DONE。
    I5/I6/I7 hmc_state_hash.jsonl 均为 38 行。
    当前 tools/run_v45_full_candidate.sh 重新 bash -n 通过。
```

结论：

```text
这是实验编排层面的审计噪音，不是候选 run 内部 Traceback/RuntimeError，也不是模型计算失败。
后续不再编辑正在被运行会话使用的 launcher；若需要新增能力，优先新增独立脚本或等待相关 run 完成。
```

新增 Phase3 short fork 工具：

```text
tools/run_v45_c23_support_short_fork.sh
tools/v45_c23_support_short_report.py
```

这两个工具的目的不是制造 full-online 结论，而是让 Phase3 的 short causal fork 符合计划里的 snapshot fork + readonly/probe_native 条件，并在报告中强制标记 `counts_as_full_online_success=false`。

## Phase 2 复盘：C9 组件贡献与交互

### Full-online 结果

| 候选 | 组件组合 | ATE_full | Delta vs P0 |
|---|---|---:|---:|
| `F0/P0` | reference | `33.76294210291885` | `0.0` |
| `I1` | no tri replay + no EMA | `36.49350294225174` | `+2.730560839332888` |
| `I2` | no tri replay + no SWA replacement | `36.28179240994649` | `+2.518850307027641` |
| `I3` | no tri replay + native mix off | `36.19603921576094` | `+2.4330971128420913` |
| `I4` | fixed tri gamma best + no EMA | `35.468877873185455` | `+1.7059357702666063` |
| `I5` | fixed tri gamma best + no SWA replacement | `34.697565417664826` | `+0.934623314745977` |
| `I6` | fixed tri gamma best + native mix off | `34.873096367813645` | `+1.1101542648947955` |
| `I7` | fixed read beta + fixed tri gamma best | `34.68703806300187` | `+0.9240959600830223` |
| `I8` | fixed read beta + fixed tri gamma best + fixed EMA best | `35.500497135292775` | `+1.7375550323739262` |

来源：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/report_R1/full_metrics/full_online_registry.csv
```

所有 I1-I8 均为完整 run：

```text
hmc_rows = 38
frames = 1101
status = done
```

### 贡献账本

从 v43/v45 landed artifacts 汇总的单组件 delta：

```text
tri_replay = +2.4469526757787676
tri_gamma_chunk_map = +0.971025405816512
fixed_tri_gamma_best = +0.8858580921580241
commit_ema = +0.48838546389362136
fixed_ema_best = +0.48838546391247206
native_mix = +0.091702749922284
swa_overlap_replace = +0.056250245127017706
read_beta_map = +0.0265805624180544
fixed_read_beta = +0.026580562436905097
```

### Interaction residual

非线性残差超过 `0.3m` 的组合：

```text
I4 fixed_tri_gamma_best + commit_ema:
    observed delta = +1.705935770285457
    additive expected = +1.3742435560516455
    residual = +0.33169221423381146

I8 fixed_read_beta + fixed_tri_gamma_best + fixed_ema_best:
    observed delta = +1.7375550323927769
    additive expected = +1.4008241185074013
    residual = +0.33673091388537557
```

近似可加的组合：

```text
I2 tri_replay+swa residual = +0.015647386140706487
I3 tri_replay+native_mix residual = -0.10555831284010964
I5 fixed_tri_gamma+swa residual = -0.007485022520214102
I6 fixed_tri_gamma+native_mix residual = +0.13259342283333808
I7 fixed_read_beta+fixed_tri_gamma residual = +0.011657305506943771
```

分析结论：

```text
1. TTT tri-replay 仍是最大贡献项：去掉 tri replay 的 I1/I2/I3 都退化到 +2.43m 至 +2.73m 区间。
2. fixed tri gamma 无法替代 C9 chunk gamma map：I7 即使保留 C9 EMA 和 SWA/native mix，仍比 P0 差 +0.924m。
3. SWA overlap replacement 与 native mix 在当前账本里更接近 near-neutral additive 项；它们会影响 ATE，但不是 C9 主收益来源。
4. commit EMA 与 fixed tri gamma 有不可忽略交互：I4/I8 residual 均超过 +0.3m，说明不能简单用单组件贡献线性解释 C9-Clean 退化。
5. I8 与 D7 ATE 完全一致到当前记录精度，确认 D7 是 fixed read beta + fixed tri gamma + fixed EMA/off 的同构组合；这也再次说明 C9-Clean 不 acceptable。
```

## Phase 3 复盘：C23 support short diagnostic

### 修复/新增内容

新增两个工具以满足计划里的 snapshot fork + readonly/probe_native 要求：

```text
tools/run_v45_c23_support_short_fork.sh
tools/v45_c23_support_short_report.py
```

修改意图：

```text
1. run_v45_c23_support_short_fork.sh
   - 从 P0(C9) 或 D7(C9-Clean) 的 chunk_006/010/016 input snapshot 读取 HMC 与 merge 状态。
   - 设置 V45_MODE_OVERRIDE=readonly 与 HMC_COMMIT_MODE=probe_native，避免 short fork 被误解为完整 online commit 实验。
   - 设置 GLOBAL_CHUNK_OFFSET 和 END_FRAME，使每个 anchor chunk 只跑 horizon=10 的局部短窗。

2. v45_c23_support_short_report.py
   - 用同 parent、同 chunk、同 horizon 下的 S0_C23_PAST 作为 baseline。
   - 输出 rows/aggregate/summary 三类文件。
   - 在 summary 中强制标记 diagnostic_only_short_rollout=true、counts_as_full_online_success=false。
```

审计注意：

```text
这些 short rollout 不是 full-online 结果，不能用来声明最终 ATE 提升。
它们只用于判断 support 变体是否值得推进 full-online。
```

### 完整性

执行矩阵：

```text
parents = C9, C9-Clean
candidates = S0/S1/S2/S3/S4/S5
chunks = 6, 10, 16
horizon = 10
total runs = 36
```

完整性检查：

```text
run dirs = 36
每个 hmc_state_hash.jsonl rows = 11
total hmc rows = 396
report rows = 36
done_candidate_rows = 30
```

证据文件：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/report_R1/short_support/v45_c23_support_short_summary.json
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/report_R1/short_support/v45_c23_support_short_aggregate.csv
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/report_R1/short_support/v45_c23_support_short_rows.csv
```

### 结果

summary：

```text
diagnostic_only_short_rollout = true
counts_as_full_online_success = false
short_gate_pass = false
passing_rows = 0
gate_rule = h10 ATE delta <= -1.0m, or rolling100 mean delta <= -3m, or [200,300) delta <= -5m with [400,600) regression <= +1m
```

Aggregate：

| Parent | Candidate | done_chunks | mean ATE_h10 delta vs S0 | best ATE_h10 delta vs S0 | best [200,300) delta vs S0 | best rolling100 mean delta vs S0 | max [400,600) delta vs S0 |
|---|---|---:|---:|---:|---:|---:|---:|
| C9 | S1 full_chunk | 3 | `+0.1393114642978889` | `+0.136147877106243` | `+0.07118837574312664` | `+0.012625297824712334` | `+0.2500099962266056` |
| C9 | S2 full_chunk_no_overlap | 3 | `+0.13568920243679253` | `+0.1175338000630326` | `+0.03360504665862152` | `+0.01865220748944907` | `+0.2471259594397317` |
| C9 | S3 off246 | 3 | `+0.14796689310341105` | `+0.13532645837667445` | `-0.11008701245491181` | `+0.0016601706477743505` | `+0.275049112534532` |
| C9 | S4 near12 | 3 | `+0.13858934611521492` | `+0.11080935912953827` | `-0.06845655699881803` | `+0.03636823477632589` | `+0.2696046702885937` |
| C9 | S5 past_plus_future_light | 3 | `+0.14323179423194018` | `+0.12093436355806908` | `+0.04614967668996428` | `+0.022597078245407687` | `+0.26524938669834697` |
| C9-Clean | S1 full_chunk | 3 | `+0.1488433737429619` | `+0.11385183612872041` | `-0.008883916990548357` | `+0.049204191383687146` | `+0.2729005691729469` |
| C9-Clean | S2 full_chunk_no_overlap | 3 | `+0.143835984390592` | `+0.09833407013187134` | `+0.029315470582893965` | `+0.0335845614960526` | `+0.27771536968619515` |
| C9-Clean | S3 off246 | 3 | `+0.1638954056893255` | `+0.14947139116371222` | `-0.05153965239659897` | `+0.04053848143620087` | `+0.2873360584125386` |
| C9-Clean | S4 near12 | 3 | `+0.14166921877372385` | `+0.09600926740808635` | `-0.0544523475772678` | `+0.047569290008027565` | `+0.28306238620988466` |
| C9-Clean | S5 past_plus_future_light | 3 | `+0.14617574880621298` | `+0.1030452788433589` | `+0.08315574561327566` | `+0.03830415438956081` | `+0.2715002450536339` |

### 分析与结论

```text
1. 没有 support 变体在 short diagnostic 上通过 gate；passing_rows=0。
2. 所有候选的 mean_ATE_h10_delta_vs_S0 都为正，说明在三个 anchor chunk 的平均短窗上，S1-S5 都比 past_only 更差。
3. 个别候选在 [200,300) segment 有局部负 delta，例如 C9/S3=-0.110087、C9/S4=-0.068456、C9-Clean/S4=-0.054452，但幅度远未达到 -5m 的 segment gate，且 [400,600) regression 均为正。
4. C9-Clean 的 best ATE_h10 delta 数值看起来略小于 C9 若干项，但 mean 仍为正，并不能证明 support 能修复 C9-Clean 的 full-online 退化。
5. 按计划“如果所有 support 变体都不超过 past：锁定 C23 past，不再做 support sweep”，因此 Phase3 不推进 support full-online，SEM3 也暂不具备 best support 前提。
```

Insight：

```text
C23 read cue 的 past_only 不是任意选择；在这组 anchor short fork 里，加入 chunk 内 full/near/future-light support 反而带来平均短窗退化。
这暗示当前 C9 的读路径可能依赖 past-only 的因果偏置来稳定风险 cue；chunk 内 bidirectional support 虽然不是外部未来泄漏，但会改变支持集合分布，短窗上没有收益证据。
```

## Phase 4 复盘：adaptive tri-replay

### 修复/新增内容

本轮为 Phase4 新增并打通：

```text
loger/pipeline/ttt_write_controller.py:
  - 新增 tri_replay_role_mode。
  - 支持 fixed / kmeans3 / otsu3 / mad / adaptive_quantile。
  - 保留 fixed 作为默认值，避免改变 C9 baseline。
  - 修复 non-tri two_replay/separate_replay/pos_neg_replay 分支，移除 tri_replay return 之后的不可达 legacy block。

loger/pipeline/hybrid_memory_controller.py:
  - 透传 ttt_write_tri_replay_role_mode。

run_pipeline_abc_v2.py:
  - 新增 CLI 参数 --ttt_write_tri_replay_role_mode。
  - 输出 tri_replay_role_mass.jsonl 供审计 role mass。

tools/run_attention_cue_experiment.sh:
  - 透传 TTT_WRITE_TRI_REPLAY_ROLE_MODE。

tools/run_v45_full_candidate.sh:
  - 增加 A0-A4 candidate 映射。
```

补充修复：

```text
tools/v45_report.py:
  - _registry(...) 增加扫描 phase/report_R1/**/full_online_registry.csv。
  - semantic phase 从旧的 phase5_semantic_minimal 改为实际输出的 phase5_semantic_read。
```

验证：

```text
bash -n 通过：
  tools/run_v45_full_candidate.sh
  tools/run_attention_cue_experiment.sh
  tools/run_v45_c23_support_short_fork.sh

py_compile 通过：
  run_pipeline_abc_v2.py
  loger/pipeline/hybrid_memory_controller.py
  loger/pipeline/ttt_write_controller.py
  tools/v45_report.py
  tools/v45_c23_support_short_report.py
```

### C9-Clean full-online 结果

| Candidate | Role mode | ATE_full | Delta vs A0 |
|---|---|---:|---:|
| A0 | fixed | `35.500497135292775` | `0.0` |
| A1 | kmeans3 | `40.41639535230242` | `+4.915898217009648` |
| A2 | otsu3 | `40.49646406199547` | `+4.9959669267026925` |
| A3 | MAD | `36.022004577930915` | `+0.5215074426381392` |
| A4 | adaptive_quantile | `35.57835251871868` | `+0.0778553834259057` |

判断：

```text
Adaptive success 要求 full ATE <= fixed-tri baseline - 0.3m。
C9-Clean fixed baseline A0 = 35.500497135292775。
最佳 adaptive A4 = 35.57835251871868，比 A0 更差 +0.0778553834259057。
因此 C9-Clean adaptive tri 没有成功。
```

### C9 original top2 结果

因为 C9-Clean 不 acceptable，按计划把 C9-Clean top2 adaptive 候选 A4/A3 推到 C9 original：

| Candidate | Role mode | ATE_full | Delta vs C9/P0 |
|---|---|---:|---:|
| F0/P0 | C9 reference | `33.76294210291885` | `0.0` |
| A4_C9 | adaptive_quantile | `33.66464516712499` | `-0.09829693579386145` |
| A3_C9 | MAD | `34.49977064240336` | `+0.7368285394845131` |

判断：

```text
A4_C9 有小幅正信号，改善约 0.0983m，但未达到 0.5m minimum progress，也未达到 33.0m strong success。
A3_C9 明显退化，不应继续作为主线。
```

### Role mass 证据

来源：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/final_reports/v45_adaptive_role_mass_summary.csv
```

摘要：

| Candidate | pos mean | neutral mean | neg mean | fallback rows |
|---|---:|---:|---:|---:|
| A0 | `0.3499999940395355` | `0.5299852223772752` | `0.12001479024949827` | `0` |
| A1 | `0.5822511883942705` | `0.34558422981124176` | `0.0721646000424193` | `0` |
| A2 | `0.5858069177695185` | `0.34815370677071705` | `0.0660393953007477` | `0` |
| A3 | `0.38582965378698547` | `0.29644875232762063` | `0.3177215942121737` | `0` |
| A4 | `0.3499999940395355` | `0.5501403792908317` | `0.09985963357013394` | `0` |
| A4_C9 | `0.47821803390979767` | `0.3835495723444119` | `0.13823237649181433` | `0` |
| A3_C9 | `0.48669143936090303` | `0.32359840378862376` | `0.1897101376140327` | `0` |

分析：

```text
1. A1/A2 在 C9-Clean 上把 positive mass 推到约 0.58，同时 negative mass 降到约 0.07/0.066，ATE 退化超过 +4.9m。这说明“低风险 token 大量正向重放”会破坏当前 C9-Clean 的写入平衡。
2. A3 在 C9-Clean 上 negative mass 约 0.318，远高于 fixed 的 0.120，ATE 仍退化 +0.52m；在 C9 original 上也退化 +0.737m。MAD 的阈值策略可能过度扩大 negative replay。
3. A4 在 C9-Clean 上最接近 A0，但仍没有收益；在 C9 original 上有 -0.098m 小正信号。A4 的作用更像轻微平滑 role mass，而不是替代 C9 的核心 tri-replay/gamma 机制。
4. 所有 adaptive runs fallback_rows=0，说明结果不是 fallback 造成的假象；adaptive role assignment 确实生效。
```

Insight：

```text
自适应分组本身不是银弹。当前 C9 的收益更依赖 chunk gamma map 与 tri-replay 的具体配比，而不是单纯让 risk distribution 自己决定三类 token。
A4_C9 的小改善说明 role mass 仍有微调空间，但收益量级只有 0.1m 左右，远不足以打开 Target30 路线。
```

## Phase 5 复盘：semantic READ minimal

### 执行范围

运行：

```text
SEM1_C23_RESID_READ_ONLY_ON_C9
SEM2_C23_RESID_READ_ONLY_ON_C9_CLEAN
SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST
```

未运行：

```text
SEM3_C23_RESID_PLUS_BEST_SUPPORT
原因：Phase3 support short_gate_pass=false，没有 best support 候选。
```

审计注意：

```text
当前 launcher 将 SEM rollout 写入 phase3_c23_support/rollouts；
Phase5 report 输出到 phase5_semantic_read/report_R1。
这是目录归类问题，不影响 run 内容；执行日志记录了实际路径。
```

### Stage-C/cache 证据

SEM1/SEM2/SEM4 均满足：

```text
hmc_config.yaml:
  stage_c_cache_mode = read
  stage_c_cache_require_hit = 1
  stage_c_mode = reference

stage_c_semantic_disabled_confirm.json:
  stage_c_disabled = false

semantic_group_summary.jsonl:
  rows = 38
  semantic_group_available = 38
  fine_label_available = 38
  semantic_group_source = MaskletOutput.G_sem
  q_sem_mean_avg = 0.974219129273766
```

说明：

```text
没有单独的 numeric cache-hit-rate 字段输出；但 require_hit=1 且 run 成功、38/38 chunk 均有 semantic group，可作为 cache/read path 有效的证据链。
```

### Full-online 结果

| Candidate | Matched baseline | ATE_full | Delta vs matched baseline | Delta vs C9 |
|---|---|---:|---:|---:|
| SEM1 | C9/P0 | `33.487566750822836` | `-0.27537535209601316` | `-0.27537535207716246` |
| SEM2 | D7/C9-Clean | `35.006318521422756` | `-0.4941786138700195` | `+1.2433764185227574` |
| SEM4 | C9/P0 | `33.42918629597565` | `-0.33375580694320206` | `-0.33375580692435136` |

判断：

```text
Semantic minimum progress 要求 ATE <= 33.3m 或 matched improvement >= 0.5m。
SEM1: 33.4876 / improvement 0.2754m -> 不通过。
SEM2: 35.0063 / improvement 0.4942m -> 非常接近 0.5m，但仍未通过；且相对 C9 仍差 +1.2434m。
SEM4: 33.4292 / improvement 0.3338m -> 不通过。
Semantic stage success 要求 ATE <= 33.0m，三者均不通过。
```

分析：

```text
1. semantic READ 在 C9 上有稳定小正信号：SEM1 改善 0.275m，SEM4 改善 0.334m。
2. adaptive_quantile 与 semantic residual 的组合 SEM4 比 SEM1 额外改善约 0.058m，但总收益仍小。
3. SEM2 几乎把 D7/C9-Clean 拉回 0.494m，但 C9-Clean 自身退化过大，semantic READ 不足以修复 fixed clean 的机制缺口。
4. 语义方向应保留为 auxiliary read branch，不应作为主线扩展大矩阵。
```

Insight：

```text
semantic residual read 的收益方向是真实存在的，但更像稳定的小偏置修正，不是能覆盖 C9 tri/gamma 机制的主贡献项。
SEM4 是本轮最佳 full-online ATE，但仍只有 33.429m，距离 33.0m 与 Target30 都明显不足。
```

## Final 复盘与 Phase6 决策

最终 consolidated summary：

```text
c9_clean:
  D7/C9-Clean ATE = 35.500497135292775
  delta_vs_C9 = +1.7375550323927769
  acceptable = false

support:
  short_gate_pass = false
  full-online support rows = 0

adaptive:
  best_candidate = A4_C9
  best_ATE_full = 33.66464516712499
  best_delta_vs_C9 = -0.09829693577501075

semantic:
  best_candidate = SEM4
  best_ATE_full = 33.42918629597565
  best_delta_vs_C9 = -0.33375580692435136

target30_success = false
phase6_sanity_recommended = false
```

Phase6 gate 判断：

```text
ATE <= 33.0m:
  false，最佳 SEM4 = 33.42918629597565

improvement vs C9 >= 0.5m:
  false，最佳改善 = 0.33375580694320206

C9-Clean acceptable and contribution map clarified:
  false，C9-Clean acceptable=false
```

因此：

```text
不启动 KITTI00/KITTI02/KITTI05 跨 sequence sanity。
这不是放弃 blocker，而是计划定义的启动条件未满足。
```

总分析：

```text
1. v45 没有得到 Target30，也没有得到 <=33.0m 的强成功。
2. C9-Clean 去 chunk-id 化失败，主要退化来自 tri-replay/gamma/EMA 的组合交互；单纯 fixed read beta 或 fixed tri gamma 都不能复现 C9。
3. C23 support 扩展在 short diagnostic 上整体退化，past_only 应锁定。
4. adaptive tri-replay 有一条小正信号 A4_C9，但只有 0.098m，不值得单独开新主线。
5. semantic READ 是本轮最好的正信号，SEM4 达到 33.429m，但仍未过 minimum progress。后续若继续，应作为小增益 auxiliary 与更核心的 C9 tri/gamma 机制修复结合，而不是扩 semantic 大矩阵。
```

证据链：

```text
Phase1/2 full metrics:
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase1_c9_clean/report_R1/full_metrics/full_online_registry.csv
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase2_interaction/report_R1/full_metrics/full_online_registry.csv

Phase3 short support:
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/report_R1/short_support/v45_c23_support_short_summary.json
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase3_c23_support/report_R1/short_support/v45_c23_support_short_aggregate.csv

Phase4 adaptive:
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase4_adaptive_trireplay/report_R1/c9clean_full_metrics/full_online_registry.csv
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase4_adaptive_trireplay/report_R1/c9_top2_full_metrics/full_online_registry.csv
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/final_reports/v45_adaptive_role_mass_summary.csv

Phase5 semantic:
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read/report_R1/sem1_vs_c9/full_online_registry.csv
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read/report_R1/sem2_vs_c9clean/full_online_registry.csv
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read/report_R1/sem4_vs_c9/full_online_registry.csv

Final:
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/final_reports/v45_final_decision.json
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/final_reports/v45_final_decision.md
```

## v45 continuation：semantic residual lambda 修复与 minimum progress

### 为什么继续

用户追问“达成目标了吗，没有请继续”。截至原 v45 final：

```text
best = SEM4
ATE = 33.42918629597565
delta vs C9 = -0.33375580694320206
stage progress <=33.0: false
minimum progress >=0.5m improvement: false
```

因此继续按计划尾部建议，把 SEM4 作为 auxiliary read branch，检查它与 C9 写入组件的冲突，并尝试最小范围的 semantic residual 强度校准。

### 新增代码修改

修复 1：

```text
loger/pipeline/hybrid_memory_controller.py
  问题：v31.sem_resid_*_l025.c23past 分支里 lam_sem 被硬编码为 0.25。
  后果：READ_BLEND_LAMBDA_OVERRIDE=0.35 只改变配置记录，不改变实际 semantic residual cue。
  修改：支持 v31.sem_resid_fine/coarse_lNNN.c23past，并从 cue name 解析 lam_sem。
```

修复 2：

```text
tools/run_v45_full_candidate.sh
  sem_resid_read_only() 支持 V45_SEM_RESID_CUE_OVERRIDE。
  默认仍保持 v31.sem_resid_coarse_l025.c23past。
```

后续整理：

```text
loger/pipeline/hybrid_memory_controller.py
  将 l010/l015/l035/l045/l050/l060/l075 枚举改为通用 lNNN 解析。
  例如 l052 -> lam_sem=0.52。
  lam_sem clamp 到 [0.0, 1.0]。
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile loger/pipeline/hybrid_memory_controller.py
bash -n tools/run_v45_full_candidate.sh
```

### 第一轮：SEM4 组件组合

结果：

| Candidate | ATE_full | Delta vs C9 | 结论 |
|---|---:|---:|---|
| SEM4 l025 | `33.42918629597565` | `-0.33375580694320206` | 原 SEM4 正信号 |
| X_NO_EMA | `33.78246724832407` | `+0.019525145405218325` | 去掉 commit EMA 后退化 |
| X_NO_SWA | `33.514475845456005` | `-0.24846625746284445` | 去掉 SWA replace 后变差 |
| X_NATIVE_OFF | `33.42668439898548` | `-0.33625770393337007` | 只比 SEM4 好约 0.0025m，近似无实质收益 |
| X_L035 via READ_BLEND_OVERRIDE | `33.42918629597565` | `-0.33375580694320206` | 与 SEM4 完全相同，暴露 lam_sem 硬编码 blocker |

分析：

```text
1. no EMA 直接从 SEM4 的 33.4292 退到 33.7825，说明 C9 的 commit EMA 组件不能简单关闭。
2. no SWA 退到 33.5145，说明 SWA overlap source replace 对当前组合仍是正组件。
3. native mix off 只有 0.0025m 改善，不构成新方向。
4. READ_BLEND_OVERRIDE l035 完全无效，这是代码 blocker，不是科学结论。
```

### 第二轮：true semantic residual lambda

结果：

| Candidate | lam_sem | ATE_full | Delta vs C9 |
|---|---:|---:|---:|
| X2_L015 | `0.15` | `33.52162541133483` | `-0.24131669158401792` |
| X2_L035 | `0.35` | `33.307116451511625` | `-0.4558256514072241` |
| X2_L045 | `0.45` | `33.24063211182052` | `-0.5223099910983322` |
| X2_L035_NATIVE_OFF | `0.35` | `33.40154257093767` | `-0.3613995319811778` |

判断：

```text
X2_L045 首次达到 minimum progress：
improvement vs C9 = 0.5223099910983322m >= 0.5m。

但仍未达到 stage progress：
33.24063211182052 > 33.0m。
```

证据：

```text
hmc_state_hash / cue_quality 中记录：
prior_v31_semantic_residual_lambda = 0.45
prior_v31_semantic_recondition_mode = semantic_residual_l045
prior_read_cue_source = v31.sem_resid_coarse_l045.c23past
```

### 第三轮：lambda 上限探测

结果：

| Candidate | lam_sem | Role mode | ATE_full | Delta vs C9 | Rot_full | FinalErr_full |
|---|---:|---|---:|---:|---:|---:|
| X3_L050 | `0.50` | adaptive_quantile | `33.19360510397078` | `-0.5693369989480672` | `4.917204738307792` | `10.696424720755818` |
| X3_L060 | `0.60` | adaptive_quantile | `33.20764929099522` | `-0.5552928119236284` | `5.046232186528163` | `11.011514042863642` |
| X3_L075 | `0.75` | adaptive_quantile | `33.381324912658975` | `-0.38161719025987395` | `5.142147705709824` | `11.232977392775014` |
| X3_FIXED_L045 | `0.45` | fixed | `33.3668121381553` | `-0.3961299647635528` | `4.775581412255934` | `10.432782835268908` |

当前最佳：

```text
X3_L050:
ATE = 33.19360510397078
delta vs C9 = -0.5693369989480672
minimum_progress_pass = true
stage_progress_pass = false
target30_success = false
```

趋势分析：

```text
1. lambda 从 0.15 -> 0.35 -> 0.45 -> 0.50 持续降低 ATE。
2. lambda 0.60 已比 0.50 略差，0.75 明显退化，说明最佳区间大概率在 0.45-0.60 之间。
3. Rot_full / FinalErr_full 随 lambda 增大整体变差，说明 semantic residual 强度过高会换来姿态/终点误差风险。
4. fixed role 的 L045 明显差于 adaptive_quantile L045，说明本轮收益来自 semantic residual 与 adaptive tri replay 的组合，而不是单独 READ cue。
```

### 当前结论

可以写：

```text
v45 continuation 通过修复 semantic residual lambda 硬编码 blocker，将 KITTI01 best ATE 从 SEM4 的 33.42918629597565 降到 X3_L050 的 33.19360510397078，达成 minimum progress。
```

必须同时写：

```text
仍未达成 stage progress，因为 33.19360510397078 > 33.0。
仍未达成 Target30。
lambda 强度提升带来 Rot_full / FinalErr_full 增大风险，不能无脑继续增大。
```

不能写：

```text
v45 已经达成科学目标。
semantic READ 可以单独解决 C9 机制缺口。
跨 sequence 泛化已经验证。
```

### Phase6 gate 更新

原 final 阶段没有启动 Phase6，因为 best improvement 只有 0.3338m。现在：

```text
X3_L050 improvement vs C9 = 0.5693369989480672m >= 0.5m
```

因此 Phase6 启动条件已经满足。下一步必须用同一配置在 KITTI00/KITTI02/KITTI05 做 sanity，不允许为其他 sequence 调参。

### 证据路径

```text
SEM4 组件组合：
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra/report_R1/sem4_component_combos/full_online_registry.csv

true lambda：
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra2/report_R1/sem4_true_lambda/full_online_registry.csv

lambda 上限：
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase5_semantic_read_extra3/report_R1/sem_lambda_upper/full_online_registry.csv
```

## v45 continuation：C9 chunk policy、isolated ablation 缺口与 adaptive TTT writing 答复

### C9 chunk policy 是否可泛化

用户追问 chunk-wise gamma 是否是每个 chunk 单独指定。审计结论：

```text
是。C9 / C9_P0_R2 launcher 使用 absolute chunk-id policy。
```

具体包括：

```text
READ_BETA_FRAME_CHUNKS =
5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25

TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS =
5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,16:0.0003

TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS =
5:0.35/0.12/0.85,...,16:0.35/0.08/0.85

TTT_WRITE_COMMIT_EMA_CHUNKS =
5,6
```

因此：

```text
C9_P0_R2 是 historical diagnostic / locked KITTI01 recipe；
不能作为 clean generalized policy 直接 claim。
```

已有去 chunk-id 负证据：

```text
ATTR_07_C9_NO_CHUNK_ID_ALL ATE = 35.29521801485317
D7_C9_CLEAN_BEST_FIXED ATE = 35.500497135292775
```

说明 C9 的历史 best 强依赖 sequence/chunk-specific map。

### C9_P0_R2 组件 ablation 是否已经回答

没有完全回答。已有 `ATTR_01..06` 是 C9-minus / knockout ablation，不是 positive-only isolated ablation。

已有数据只能说明 C9 内部必要性：

| Row | ATE | Delta vs C9 | 解释边界 |
|---|---:|---:|---|
| `ATTR_01_C9_MINUS_READ_MAP_TO_FLAT` | `33.789522665336904` | `+0.0265805624180544` | read chunk map 在 C9 内近似中性；不能说明 only frame-attn/read cue |
| `ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT` | `34.73396750873536` | `+0.971025405816512` | tri chunk map 在 C9 内重要；但这是 chunk-specific |
| `ATTR_03_C9_MINUS_COMMIT_EMA` | `34.25132756681247` | `+0.48838546389362136` | commit EMA 在 C9 内中等重要 |
| `ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE` | `33.81919234804587` | `+0.056250245127017706` | SWA replace 在 C9 内 full ATE 贡献很小；不能说明 only SWA |
| `ATTR_05_C9_MINUS_TTT_TRI_REPLAY` | `36.20989477869762` | `+2.4469526757787676` | TTT tri replay 是 C9 内最大必要项；不能说明 only TTT |
| `ATTR_06_C9_MINUS_NATIVE_MIX` | `33.85464485284113` | `+0.091702749922284` | native mix 小贡献 |

因此用户要求的三个正向隔离问题：

```text
only frame attention/read cue
only TTT writing
only SWA
```

当前仍未由 existing results 回答，必须补跑 positive-only isolated ablation。

### Adaptive TTT writing 结果

当前已经完成的自适应 TTT writing 主要来自 Phase4 adaptive tri-replay。

在 C9-Clean parent 上：

| Candidate | Role mode | ATE | Delta vs historical C9 | 结论 |
|---|---|---:|---:|---|
| `A0` | fixed C9 tri replay | `35.500497135292775` | `+1.7375550323927769` | C9-clean 本身失败 |
| `A1` | kmeans3 | `40.41639535230242` | `+6.653453249402425` | 严重失败 |
| `A2` | otsu3 | `40.49646406199547` | `+6.733521959095469` | 严重失败 |
| `A3` | mad | `36.022004577930915` | `+2.259062475030916` | 仍失败 |
| `A4` | adaptive_quantile | `35.5783525187` | `+1.8154104158` | C9-clean 上不成立 |

因为 C9-Clean 不 acceptable，按计划把 top2 推回 original C9：

| Candidate | Role mode | ATE | Delta vs C9 | [200,300) | [400,600) |
|---|---|---:|---:|---:|---:|
| `F0` | original C9 | `33.76294210291885` | `0.0` | `76.10213555431245` | `41.896364212570404` |
| `A4_C9` | adaptive_quantile | `33.66464516712499` | `-0.09829693579386145` | `75.7619011308957` | `41.534127951641764` |
| `A3_C9` | mad | `34.49977064240336` | `+0.7368285394845131` | `76.94194932167122` | `44.767936577140894` |

结论：

```text
纯 adaptive TTT writing 目前最好是 A4_C9 adaptive_quantile。
它只比 C9 改善 0.0982969358m。
这是小正信号，不是可接受的强结果，也不是 Target30 方向的充分证据。
```

role mass 诊断：

```text
A4_C9:
positive_mass_mean = 0.47821803390979767
neutral_mass_mean  = 0.3835495723444119
negative_mass_mean = 0.13823237649181433
fallback_rows = 0
```

这说明 adaptive_quantile 实际生效了，不是 fallback；但效果幅度很小。

和 semantic residual READ 叠加时：

```text
X2_L045 adaptive_quantile ATE = 33.24063211182052
X3_FIXED_L045 fixed role ATE = 33.3668121381553
```

同为 semantic residual l045 时，adaptive_quantile 比 fixed role 好约：

```text
33.3668121381553 - 33.24063211182052 = 0.1261800263347816m
```

因此：

```text
adaptive TTT writing 与 semantic residual READ 有组合收益；
但单独 adaptive TTT writing 只有约 0.098m 小收益。
```

安全表述：

```text
自适应 TTT writing 当前不是失败到完全无信号，但也没有达到可称为解决方案的强度。
它可以作为 auxiliary，与 semantic residual READ 组合继续探索；
不能单独 claim 为 v45 主要成功。
```

## v45 收尾：本轮偏离计划的最终审计结论

用户要求收尾，并指出当前推进没有按原计划核心问题执行。这个批评成立。

### 哪些结果可以保留

可以保留的真实结果：

```text
1. C9_P0_R2 复现 reference：
   ATE = 33.76294210291885

2. 原 v45 best SEM4：
   ATE = 33.42918629597565
   delta vs C9 = -0.33375580694320206

3. semantic residual lambda 硬编码修复后：
   X3_L050 ATE = 33.19360510397078
   delta vs C9 = -0.5693369989480672
   达成 minimum progress，但未达成 stage progress / Target30。

4. adaptive TTT writing 单独结果：
   A4_C9 adaptive_quantile ATE = 33.66464516712499
   delta vs C9 = -0.09829693579386145
   只有小正信号。
```

这些数据都来自已完成 run 的 benchmark/report 文件。

### 哪些不能作为结论

不能写：

```text
1. 不能写 C9_P0_R2 是可泛化策略。
   它包含 chunk-id 手工 map：
   READ_BETA_FRAME_CHUNKS、TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS、
   TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS、TTT_WRITE_COMMIT_EMA_CHUNKS。

2. 不能写已经回答了 only frame-attn / only TTT / only SWA 的贡献。
   现有 ATTR_01..06 是 knockout ablation，不是 positive-only isolated ablation。

3. 不能写 adaptive TTT writing 已解决手工 triple replay 问题。
   A4_C9 只改善 0.0982969358m。

4. 不能写 X3_L050 已完成 v45 科学目标。
   它是 semantic residual READ + adaptive tri-replay 的组合结果；
   未跨 sequence 验证；
   也没有回答用户要求的三个组件独立贡献。

5. 不能使用 l048/l052/l055/l058 narrow sweep。
   这些 run 已按用户要求终止，没有 01.txt / kitti_benchmark.log / DONE。
```

### 本轮偏离在哪里

偏离点：

```text
1. 用户核心问题是 C9_P0_R2 的组件独立贡献与泛化性；
   我继续推进了 semantic residual lambda sweep，偏向局部 ATE 改善。

2. 用户要求判断 chunk-wise gamma 是否可泛化；
   我没有第一时间把 C9 chunk-id map 的不可泛化风险作为主结论。

3. 用户要求 C9_P0_R2 的 positive-only ablation：
   only frame-attn / only TTT / only SWA；
   但目前只整理了 knockout 数据，没有完成正向隔离实验。
```

### 当前最诚实答案

```text
v45 没有按用户计划完整达成。

C9_P0_R2 是 chunk-id-specific historical best，不是 clean deployable generalized policy。

已有 knockout 能说明 TTT tri-replay 在 C9 内必要性最大，
但不能说明 only TTT、only frame-attn、only SWA 的独立贡献。

adaptive TTT writing 有小正信号，但不足以替代 C9 手工 chunk map。

semantic residual lambda 修复得到 KITTI01 minimum progress，
但它不是用户追问的核心 ablation 答案。
```

### 已终止的未完成实验

按用户“收尾”要求终止：

```text
V45X4_SEM4_ADAPTIVE_TRUE_L048
V45X4_SEM4_ADAPTIVE_TRUE_L052
V45X4_SEM4_ADAPTIVE_TRUE_L055
V45X4_SEM4_ADAPTIVE_TRUE_L058
```

终止时状态：

```text
无 01.txt
无 kitti_benchmark.log
run_status.txt 只有 START，无 DONE
```

因此这些 run 不进入任何指标表。

### 如果未来重新开始，必须先做的最小实验

未来若继续，应该先做，不应再继续 lambda sweep：

```text
ONLY_FRAME_ATTN_CLEAN:
  read_path/frame attention only；
  no TTT writing；
  no SWA；
  no chunk-id read beta map。

ONLY_TTT_CLEAN:
  beta_frame=0 或关闭 read bias；
  no SWA；
  TTT writing only；
  no chunk-wise gamma / no chunk replay params；
  使用全局或真正自适应策略。

ONLY_SWA_CLEAN:
  beta_frame=0 或关闭 read bias；
  no TTT writing；
  SWA overlap/source replace only。
```

并且如果同时跑 C9-chunked only variants，必须标注为：

```text
只解释 historical C9 recipe，不用于泛化 claim。
```

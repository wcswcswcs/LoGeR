# ACL2 v15：TTT Write Reproducibility、Causal Sandbox 与 Target-25 行动空间重启实验计划

日期：2026-05-19  
对象：LoGeR / HMC Pipeline v2 / KITTI01 / TTT write policy  
目标：继续寻找真正可部署的 TTT 写入策略，禁止把 offline trajectory rewrite 或 GT runtime oracle 写成 TTT 成功。

---

## 0. 当前状态的独立判断

v14 没有产生新的 TTT 写入策略。它停在 Phase 0：`H9_REPEAT_R3` 没有复现 locked H9 baseline，`C9_WEAK_FREEZE_C56_REPEAT_R3` 虽然复现成功，但它只是 v13 旧策略 repeat，不是 v14 新候选。因此，v14 的主要信息不是“某个新 TTT action 失败”，而是：**当前实验系统的边界复现已经不稳定，不允许直接进入 sandbox / candidate bank / selector。**

当前 deployable online 边界如下：

```text
locked H9 reference:
    ATE / Rot = 34.1258 / 6.5414
    [200,300) = 74.410
    [400,600) = 44.354
    counts_as_ttt_write = true

v14 H9 repeat R3:
    ATE / Rot = 34.2513 / 6.4833
    [200,300) = 74.501
    [400,600) = 44.828
    drift vs locked H9 = +0.1255m
    gate = fail

v13 C9 repeat R3:
    ATE / Rot = 33.7629 / 6.5259
    [200,300) = 76.102
    [400,600) = 41.896
    counts_as_ttt_write = true
    but this is v13 repeat, not v14 progress

NOGTPOSE_27 repeat:
    ATE = 22.3669
    counts_as_ttt_write = false
    diagnostic only
```

新目标是：

```text
KITTI01 online deployable TTT write ATE <= 25.0m
```

当前最好的 deployable online TTT/HMC 结果如果按 v13 C9 算是 `33.7629m`，距离目标仍差：

$$
33.7629 - 25.0 = 8.7629m
$$

如果按 locked H9 算是 `34.1258m`，距离目标仍差：

$$
34.1258 - 25.0 = 9.1258m
$$

这不是继续微调 `gamma`、`neutral`、`read beta`、`commit EMA`、`post-zeropower mix` 能补上的距离。v14 暴露的问题更基础：**在没有复现 H9 的情况下，任何 candidate-bank / sandbox 结果都可能被工程漂移污染。**

---

## 1. 实验整体目标

v15 的整体目标不是继续做一个更细的 TTT 标量 sweep，而是重新建立一条可信、快速、可并行的 TTT write 研究链路。

本轮要回答五个核心问题。

第一，当前 locked H9 为什么在 v14 下不能复现？这是硬前置条件。只要 H9 drift 仍为 `0.1255m`，就不能判断后续任何 `0.1m` 到 `0.3m` 的 TTT 改进是否真实。

第二，HMC state snapshot / load / short rollout sandbox 是否能做到严格 parity？如果 sandbox 不能复现 full run suffix，candidate bank 和 short-horizon selector 都没有意义。

第三，TTT fast-weight candidate commit bank 是否存在足够上界？也就是说，在不改输出轨迹、不用后处理的前提下，是否存在某些可提交的 $W_{m+1}$ 候选能真正改善 `[200,300)`、`[400,600)` 和整体 ATE。

第四，如果 candidate bank 有 oracle 上界，是否能用 no-GT proxy 选出好 candidate？否则它仍只是 GT oracle，不是可部署 TTT 策略。

第五，如果 candidate bank 没有上界，则当前 TTT write action space 已经不足，需要升级到 post-zeropower residual basis、scale-aware auxiliary replay、dual-bank fast weights 或 MPC-style finite-difference sensitivity，而不是继续局部调参。

---

## 2. 本轮硬边界

### 2.1 可以算作 TTT write success 的结果

必须同时满足：

```text
1. online full LoGeR/HMC run；
2. output trajectory 来自模型 forward，不是后处理改写；
3. TTT commit 影响未来 HMC state；
4. runtime action 不使用 GT；
5. hmc rows = 38；
6. no failed chunks；
7. counts_as_ttt_write = true。
```

### 2.2 不能算作 TTT write success 的结果

以下都只能作为 diagnostic / oracle：

```text
offline trajectory rewrite
NOGTPOSE / pose-scale proxy output txt rewrite
GT runtime action
GT-selected candidate commit
sandbox logging-only smoke
partial run
hmc rows < 38
baseline reproduction failed 后继续跑出的结果
```

### 2.3 禁止继续扩大的方向

本轮不再继续扩大以下小修小补：

```text
gamma 0.0048 / 0.0052 级微扫
neutral 0.82 / 0.88 微扫
body/exit window 手工微调
read beta 4.80 / 4.90 微扫
chunk16 weak tri-replay 细扫
普通 post-zeropower mix/cap 细扫
simple dual-lifetime alpha/K 小矩阵
```

如果某个阶段需要这些变量，只能作为对照，不作为主实验族。

---

## 3. 核心假设

### H0：v14 的主要问题首先是 reproducibility，不是新 TTT action 失败

v14 的 `C9_REPEAT_R3` 能严格复现，而 `H9_REPEAT_R3` 不能复现。这说明问题不像全局随机性，更可能来自：

```text
H9 配置没有完全复原；
READ_BETA_FRAME_CHUNKS / body-exit-c16 policy 环境变量没有正确透传；
run script default 被近期修改；
保存 / 加载 HMC state 的新接口对某些路径有副作用；
H9 与 C9 经过了不同的 write-score / read-beta 分支；
历史 H9 artifact 和当前 H9 repeat 使用了不同代码路径。
```

H0 成立的标准是：修复后 H9 重复运行满足：

$$
|ATE_{H9\_repeat} - 34.1258| \le 0.03m
$$

并且：

```text
hmc rows = 38
commit mode = probe_ttt_write
trajectory no postprocess
state hash / config hash 可解释
```

若 H0 不成立，本轮不得进入 Phase 1。

---

### H1：Causal Sandbox 必须严格复现 full-run suffix，才能用于 candidate bank

如果从 chunk $m$ 保存的 HMC effective input state 开始短 rollout，且使用与 full run 相同的 native action，那么 short rollout 应该复现 full run 从 chunk $m$ 开始的 suffix 输出。

记 full run suffix 轨迹为 $T^{full}_{m:m+k}$，sandbox rollout 轨迹为 $T^{sb}_{m:m+k}$。定义：

$$
E_{pose}^{sb}(m,k)=\max_t \|T^{full}_t - T^{sb}_t\|_{\infty}
$$

H1 通过标准：

```text
E_pose_sb <= 1e-5 for tensor-level smoke，或
ATE_suffix_delta <= 0.01m for benchmark-level smoke。
```

如果 H1 不通过，candidate bank 的 short-horizon selector 不可信，必须先修 state save/load。

---

### H2：TTT candidate commit bank 可能存在比当前单策略更强的 fast-weight state

过去我们每个 chunk 只提交一个 $W_{m+1}$。但如果对同一个 chunk 构造多个候选：

$$
\mathcal{B}_m=\{W^{native}_{m+1}, W^{H9}_{m+1}, W^{C9}_{m+1}, W^{tri}_{m+1}, W^{pz}_{m+1}, W^{ema}_{m+1}, \dots\}
$$

可能存在某个候选 $W^*_{m+1}$，它对未来 1-3 个 chunk 的 scale / drift 更好，但此前被固定策略错过。

H2 通过标准分两层：

```text
oracle bank pass:
    oracle-selected candidate bank 在 sandbox short rollout 上
    对关键 future window 的 ATE 改善 >= 3m，且下游不崩。

full-run pass:
    oracle-derived top candidate 做 online full run 后，ATE <= 33.5m，
    或 [200,300) 改善 >= 5m 且 [400,600) 不恶化。
```

如果 oracle candidate bank 都没有上界，则当前候选族不足，必须扩展 action family。

---

### H3：如果 oracle bank 有上界，no-GT short-horizon selector 必须能预测好 candidate

可部署策略不能用 GT。no-GT selector 应用模型内部信号选择 candidate：

```text
overlap pointmap consistency
predicted step-length scale stability
TTT apply mismatch
TTT update conflict energy
memory state rel diff
frame attention high-D mass
SWA/overlap source consistency
```

对 candidate $c$，定义 no-GT score：

$$
S_{proxy}(c)=
w_s E_{scale}(c)+
w_o E_{overlap}(c)+
w_a E_{apply}(c)+
w_m E_{mem}(c)+
w_d E_{Dg}(c)
$$

H3 通过标准：

```text
Spearman(proxy score, oracle future ATE) >= 0.50
H9-or-better selection ratio >= 0.60
Top-1 selected candidate full run ATE <= current deployable best - 0.20m
```

如果 H3 失败，但 H2 成立，则问题是 selector，不是 TTT action。Codex 应优先扩展 proxy feature，而不是换 TTT action family。

---

### H4：如果 candidate bank 无上界，当前 TTT write action 必须升级为 fast-weight direction action

如果所有 candidate bank 都不能超过 H9/C9，说明简单 commit candidate 仍没有足够表达力。下一步不能继续 scalar gate，而要直接控制 fast-weight delta 的方向。

候选方向：

```text
post-zeropower residual basis routing
freeze-derived harmful direction projection
scale-aware auxiliary replay target
dual-bank TTT memory
finite-difference TTT sensitivity / MPC update
```

H4 通过标准：新的 fast-weight direction action 至少满足：

```text
KITTI01 ATE <= 33.0m
或 [200,300) 改善 >= 8m 且 [400,600) 不恶化
或 sandbox oracle 显示 future rollout gain >= 5m
```

若 H4 也失败，则 TTT write-only 不再作为 Target-25 主线，只作为 regularizer；但本轮仍必须先把 H0-H4 做完，不能凭直觉放弃。

---

## 4. Phase 0：Reproducibility Reconciliation

### 4.1 目标

解决 v14 的 H9 repeat drift。Phase 0 的目标不是刷指标，而是让实验边界重新可信。

### 4.2 必跑实验

#### P0-A：locked references repeat，不保存 state

```text
P0_A1_H9_REPEAT_no_state_save
P0_A2_C9_REPEAT_no_state_save
P0_A3_WINGAM_REPEAT_no_state_save
```

#### P0-B：minimal passive state save

只保存 `chunk_005_input.pt`、`chunk_010_input.pt`、`chunk_016_input.pt`，不保存 before/after，不影响控制路径。

```text
P0_B1_H9_REPEAT_input_only_save
P0_B2_C9_REPEAT_input_only_save
```

#### P0-C：archived config replay

直接从历史 H9 run 目录读取或复建：

```text
run_config.yaml
command.sh
env.json
software_commit.txt
hmc_config.yaml
```

运行：

```text
P0_C1_H9_ARCHIVED_CONFIG_REPLAY
```

### 4.3 必须记录的指标

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
[200,300)
[200,400)
[400,600)
hmc_rows
num_failed_chunks
commit_mode
counts_as_ttt_write
postprocess_flag
software_commit_hash
run_script_hash
resolved_cli_args.json
env_vars_effective.json
hmc_config.yaml
hmc_state_hash.jsonl
prior_debug_summary.json
read_beta_frame_chunks_effective
write_policy_effective
```

新增 `config_diff_report.json`，字段：

```text
field_name
old_value
new_value
source: cli/env/default/config
is_ttt_related
is_read_related
is_swa_related
is_eval_related
```

### 4.4 判断标准

Phase 0 通过条件：

```text
H9 repeat ATE drift <= 0.03m
C9 repeat ATE drift <= 0.03m
WINGAM repeat ATE drift <= 0.03m
hmc rows = 38 for all
no state-save path changes output beyond 0.01m
```

### 4.5 如果不满足，Codex 优先尝试方向

如果 H9 fail 但 C9 pass：

```text
1. diff H9 effective CLI/env against archived H9；
2. 检查 READ_BETA_FRAME_CHUNKS 是否实际生效；
3. 检查 body/exit/c16 read beta 是否被 default 覆盖；
4. 检查 TTT_WRITE_NATIVE_MIX_SCALES / TRI_REPLAY_ROLE 是否被正确传入；
5. 检查 fast_cue_eval、read_layer_mode、frame_bias_mode 是否一致；
6. 检查 write_score_source 是否从 stage_d_x_dg_inv_sqrt 退化；
7. 检查 HMC ignore semantic / pass-through 路径是否误触发；
8. 检查 evaluator 是否使用同一 trajectory txt。
```

如果 H9/C9 都 fail：

```text
1. 先停所有 full runs；
2. 回到 B0/SWKS3 historical reproduction；
3. 检查 script 修改、Python path、checkpoint/config、resolution/window/overlap/reset；
4. 检查 hmc_state save/load 是否有 side effect；
5. 只有 B0 和 H9 同时复现后才继续。
```

如果保存 state 后才 fail：

```text
1. _save_hmc_state 必须 deepcopy 后 detach+cpu；
2. 禁止保存 before/input/after 全量 state；
3. 只允许 selected chunks input-only；
4. snapshot 总量单 phase <= 20GB；
5. 保存逻辑必须在 commit 后不可改变 tensor alias。
```

---

## 5. Phase 1：Causal Sandbox Parity

### 5.1 目标

确认从保存的 HMC state 加载并短 rollout 能复现 full run suffix。

### 5.2 实验设计

对 H9 和 C9 分别保存：

```text
chunk 005 input state
chunk 010 input state
chunk 016 input state
```

从每个 state 启动 sandbox rollout：

```text
horizon = 1, 3, 5 chunks
candidate = native_same_as_full
```

实验：

```text
P1_A1_H9_chunk5_h1_native
P1_A2_H9_chunk5_h3_native
P1_A3_H9_chunk5_h5_native
P1_A4_H9_chunk10_h3_native
P1_A5_H9_chunk16_h3_native
P1_B1_C9_chunk5_h3_native
P1_B2_C9_chunk10_h3_native
P1_B3_C9_chunk16_h3_native
```

### 5.3 必须记录的指标

```text
short_rollout_metrics.csv
short_rollout_proxy.jsonl
state_load_debug.jsonl
candidate_commit_debug.jsonl
state_hash_before_load
state_hash_after_load
pose_tensor_max_diff_vs_full_suffix
pointmap_max_diff_vs_full_suffix
ATE_suffix_delta
Rot_suffix_delta
chunk_error_delta
walltime_sec
peak_gpu_mem
```

### 5.4 判断标准

Phase 1 通过条件：

```text
horizon=1 pose max diff <= 1e-5
horizon=3 ATE_suffix_delta <= 0.01m
horizon=5 ATE_suffix_delta <= 0.03m
H9/C9 都通过
```

### 5.5 如果不满足，Codex 优先尝试方向

```text
1. 检查保存的是 reset_every 后 effective input state，而不是 before/probe/control 中间态；
2. 检查 load 后 tensor tree 是否全部移动到正确 device；
3. 检查 random / dropout / compile mode 是否和 full run 一致；
4. 检查 SWA history 是否同时保存并恢复；
5. 检查 TTT w0/w1/w2 layer count、None slot、dtype 是否一致；
6. 检查 start_frame / chunk index / overlap trimming 是否对齐；
7. 如果 state load 仍不稳定，改为 full-run 内部 fork candidate，不落盘 state。
```

---

## 6. Phase 2：TTT Candidate Commit Bank Oracle

### 6.1 目标

回答一个根本问题：**TTT fast weights 里是否存在可提交的候选 state 能显著改善未来窗口？**

这一步允许使用 GT 作为 oracle 选择器，只用于测试上界；oracle 结果不算 deployable success。

### 6.2 Candidate bank 定义

对关键 chunk 生成候选：

```text
m in {5, 6, 10, 16}
```

每个 chunk 的候选 bank：

```text
B0_native:              原生 probe_ttt_write commit
B1_H9:                  H9 当前 commit policy
B2_C9_weak_freeze:       C9 weak freeze c56 policy
B3_WINGAM:               windowed tri-replay policy
B4_C16ROLE:              chunk16 role policy
B5_no_write:             当前 chunk TTT write discard, SWA/ref preserved
B6_commit_ema_090:       selected branch commit EMA alpha 0.90
B7_commit_ema_110:       selected branch commit EMA alpha 1.10
B8_pz_norm_mix025:       post-zeropower norm mix 0.25
B9_pz_w1_cap050:         w1 cap diagnostic candidate
B10_scale_aux_replay:    scale-aware replay target candidate
B11_overlap_aux_replay:  overlap feature replay target candidate
B12_dual_bank_short:     short-bank correction candidate
```

注意：candidate bank 是 TTT commit 候选，不修改输出轨迹，不做 offline pose rewrite。

### 6.3 Oracle rollout

从 saved state at chunk $m$ 出发，对每个 candidate 进行 short rollout：

```text
horizon = 3 chunks for quick gate
horizon = 5 chunks for passing candidates
```

oracle metric：

$$
J_{oracle}(c;m)=
ATE_{future}(c;m)+
\lambda_{down} \max(0, ATE_{downstream}(c;m)-ATE_{downstream}^{H9}(m))
$$

其中：

```text
future = m 到 m+horizon 的局部窗口
 downstream = 如果 m 在 5/6，则至少检查 [400,600) proxy 或后续窗口
```

### 6.4 必须记录的指标

```text
candidate_bank_registry.csv
candidate_commit_debug.jsonl
oracle_rollout_metrics.csv
candidate_vs_h9_delta.csv
candidate_state_diff.csv
candidate_layer_branch_norm.csv
candidate_role_mass.csv
candidate_selected_by_gt.json
```

每个 candidate 记录：

```text
chunk_id
candidate_id
action_family
branch_mask
layer_scope
state_diff_norm
update_cos_to_H9
update_cos_to_C9
future_ATE
future_Rot
future_scale_proxy
future_overlap_error
downstream_ATE_proxy
oracle_rank
```

### 6.5 判断标准

Phase 2 通过条件：

```text
至少一个 chunk 的 candidate bank oracle future ATE 改善 >= 3m；
或者组合 oracle rollout 可以让 [200,300) 改善 >= 5m 且 [400,600) 不恶化；
或者 top oracle candidate full online ATE <= 33.5m。
```

强通过：

```text
top oracle-derived full online ATE <= 32.5m
```

### 6.6 如果不满足，Codex 优先尝试方向

如果所有 candidate 都接近 H9：

```text
1. candidate bank 太窄，转 Phase 4 新 action family；
2. 增加 post-zeropower residual basis candidate；
3. 增加 freeze-derived harmful direction candidate；
4. 增加 scale-aware auxiliary replay candidate；
5. 增加 dual-bank W_long/W_short candidate。
```

如果 oracle 选出的 candidate 改善 `[200,300)` 但毁 `[400,600)`：

```text
1. candidate 缺 continuity preservation；
2. 加入 positive continuity term；
3. 禁止 hard freeze / full discard；
4. 使用 W_long + W_short 分离；
5. 加 downstream penalty 后重选。
```

如果 oracle 只有 GT 后验能选，no-GT proxy 无法区分：

```text
1. 不急着跑 full；
2. 进入 Phase 3 selector feature 扩展；
3. 加 overlap / scale / apply-mismatch / memory-diff proxy。
```

---

## 7. Phase 3：No-GT Short-Horizon Selector

### 7.1 目标

如果 Phase 2 证明 candidate bank 有 oracle 上界，则本阶段尝试不用 GT 选择 candidate。

### 7.2 Proxy feature

每个 candidate 的 proxy 特征：

```text
step_length_median_ratio
step_length_iqr
overlap_pointmap_residual
static_overlap_scale_proxy
SWA_history_consistency
TTT_apply_mismatch_mean
TTT_apply_mismatch_p90
update_conflict_energy_mean
memory_state_rel_diff
candidate_delta_norm
D_g_high_mass
frame_attention_entropy_shift
```

综合 proxy：

$$
S_{proxy}(c)=
0.30 E_{scale}(c)+
0.25 E_{overlap}(c)+
0.15 E_{apply}(c)+
0.15 E_{mem}(c)+
0.15 E_{conflict}(c)
$$

先不训练模型，用固定权重和 z-score normalization；如果 Spearman 不够，再尝试 logistic/ridge 轻量拟合，但只允许用非测试 run / offline candidate bank，不允许用同一 candidate full result过拟合。

### 7.3 必须记录的指标

```text
selector_feature_table.csv
selector_score_table.csv
selector_vs_oracle_scatter.png
spearman_proxy_oracle.json
chunk_choice_table.csv
selected_candidate_full_runs.csv
```

### 7.4 判断标准

通过条件：

```text
Spearman(proxy, oracle future ATE) >= 0.50
Top-1 selected candidate 在 60% chunk 上不差于 H9
selected policy full online ATE <= H9 - 0.20m
```

强通过：

```text
selected policy full online ATE <= 33.0m
且 [200,300) 改善 >= 3m
且 [400,600) 不恶化
```

### 7.5 如果不满足，Codex 优先尝试方向

```text
1. 如果 Spearman < 0.3，说明 proxy 不够，先不要 full run；
2. 加入 reset-window relative feature，而不是 raw feature；
3. 加入 future two-chunk cheap probe，不改 commit，只评估 no-GT residual；
4. 加入 per-axis scale/yaw proxy；
5. 如果 selector 始终失败但 oracle 强，通过 MPC-style online short rollout 直接选择，而非静态 proxy。
```

---

## 8. Phase 4：新 TTT Action Family

Phase 4 只在 Phase 2 candidate bank 无上界或上界不足时启动。目标是扩展 TTT write action，不再做标量门控。

### 8.1 Action Family A：post-zeropower residual basis routing

动机：pre-zeropower token prior 可能被 zeropower / norm restoration 折叠，导致 token-level prior 无法真正控制 fast-weight direction。

构造 post-zeropower delta：

$$
\Delta W^{post}_{m,l,r}
$$

收集候选 delta：

```text
H9 delta
C9 delta
WINGAM delta
freeze5 removed delta
freeze56 removed delta
PZ variants
```

做 basis decomposition：

$$
\Delta W = \sum_k a_k B_k + R
$$

实验候选：

```text
PZB_01: suppress harmful basis B_h by 0.25
PZB_02: suppress harmful basis B_h by 0.50
PZB_03: boost continuity basis B_c by 0.25
PZB_04: suppress B_h and boost B_c
PZB_05: apply only w0 selected layers
PZB_06: apply only chunks 5/6/16
```

通过条件：

```text
ATE <= 33.0m
或 [200,300) 改善 >= 5m 且 [400,600) 不恶化
```

如果失败：

```text
Codex 尝试按 layer/branch 单独 basis，而不是全局 basis；
若仍失败，说明 post-delta basis 不能捕获 scale-state。
```

---

### 8.2 Action Family B：scale-aware auxiliary replay target

动机：NOGTPOSE 说明 reset-window scale 是主误差方向，但后处理不能计为 TTT success。需要把 scale signal 接回 TTT replay objective。

不改输出轨迹，只改 TTT replay target。定义 static overlap / structure feature target：

$$
v'_i = v_i + \alpha \cdot s_m \cdot (v_i^{static} - v_i)
$$

其中 $s_m$ 是 no-GT window scale risk：

$$
s_m = \operatorname{clip}\left(
\frac{\operatorname{median}_{t \in window} \|\Delta \hat p_t\|}
{\operatorname{EMA}_{prev}\|\Delta \hat p_t\| + \epsilon} - 1,
- s_{max}, s_{max}
\right)
$$

候选：

```text
SCALE_REPLAY_01: alpha 0.025, structure-only
SCALE_REPLAY_02: alpha 0.050, structure-only
SCALE_REPLAY_03: alpha 0.025, overlap-static
SCALE_REPLAY_04: alpha 0.050, overlap-static
SCALE_REPLAY_05: scale risk only chunks 5/6/10/16
SCALE_REPLAY_06: scale risk + continuity penalty
```

通过标准：

```text
ATE <= 33.0m
or [200,400) improves >= 5m
and [400,600) not worse than H9 + 1m
```

如果失败：

```text
1. 检查 v'_i 是否过于接近 v_i，增大 alpha 前先看 update norm；
2. 检查 structure mask coverage；
3. 改用 apply-side auxiliary target；
4. 若 scale-risk 信号强但 TTT 无响应，说明 replay target 不能影响 pose-scale state。
```

---

### 8.3 Action Family C：dual-bank TTT memory

动机：chunk5/6 的 update 同时包含 harmful direction 和 useful continuity。单个 $W$ 中很难用一个 multiplier 分离。

最小实现：

```text
W_long: 只接收 continuity / static / low-conflict update
W_short: 接收 high-conflict correction，apply K chunks 后衰减
```

应用：

$$
W^{apply}_m = W^{long}_m + \alpha_m W^{short}_m
$$

提交：

$$
W^{long}_{m+1}=W^{long}_m + \Delta W^{pos/neu}_m
$$

$$
W^{short}_{m+1}=\rho W^{short}_m + \Delta W^{conflict}_m
$$

候选：

```text
DBANK_01: K=2, alpha=0.25, rho=0.5
DBANK_02: K=3, alpha=0.25, rho=0.5
DBANK_03: K=2, alpha=0.50, rho=0.3
DBANK_04: conflict only w0
DBANK_05: scale-risk gated short bank
DBANK_06: short bank only at chunks 5-12
```

通过标准：

```text
ATE <= 33.0m
or [200,300) improves >= 5m
and [400,600) not worse than H9
```

如果失败：

```text
1. 如果 FinalErr/Rot 好但 ATE 差，short bank 仍在做 orientation regularizer；
2. 如果 [200,300) 好但 [400,600) 崩，short lifetime 太长或 positive continuity 不够；
3. 改成 candidate-bank selector，而不是固定 short bank。
```

---

### 8.4 Action Family D：finite-difference TTT sensitivity / MPC update

动机：不要假设某个 cue 有用，直接测试 fast-weight perturbation 对未来 proxy 的敏感性。

对关键 delta direction $D_k$：

$$
W' = W + \epsilon D_k
$$

短 rollout 计算 no-GT proxy：

$$
\Delta S_k = S_{proxy}(W') - S_{proxy}(W)
$$

选择：

$$
D^* = \arg\min_k \Delta S_k
$$

候选 direction：

```text
native delta
tri-replay delta
post-zeropower harmful basis
continuity basis
scale-aware replay delta
w0 selected layer delta
w1 value-memory delta
```

通过标准：

```text
finite-difference selected direction full online ATE <= 33.0m
or sandbox future rollout gain >= 5m
```

如果失败：

```text
1. proxy 不敏感：换 proxy；
2. direction 不足：换 basis；
3. rollout 与 full 不一致：回 Phase 1 修 sandbox；
4. sensitivity 有效但 full 无效：加 downstream penalty / horizon。
```

---

## 9. 统一指标记录

每个 full run 必须记录：

```text
metrics_global.json
trajectory_diagnostics.json
per_frame_error.csv
per_chunk_error.csv
segment_50_100_200.csv
global_drift_dashboard.json
hmc_state_hash.jsonl
hmc_control_summary.jsonl
hmc_probe_summary.jsonl
ttt_write_debug.jsonl
ttt_layer_branch_update.csv
runtime_summary.json
resolved_cli_args.json
```

核心指标：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
[200,250)
[200,300)
[200,400)
[400,600)
50f_mean
100f_mean
200f_mean
```

TTT 指标：

```text
update_norm_w0/w1/w2
post_zeropower_delta_norm
norm_restore_ratio
update_cos_to_H9
update_cos_to_C9
update_cos_to_native
memory_state_rel_diff
tri_replay_pos_mass
tri_replay_neu_mass
tri_replay_neg_mass
candidate_delta_norm
candidate_delta_cosine
```

Sandbox 指标：

```text
suffix_pose_max_diff
suffix_ATE_delta
future_window_ATE
future_overlap_residual
future_scale_proxy
selector_proxy_score
oracle_rank
```

---

## 10. 必须可视化的内容

### 10.1 Global drift dashboard

每个晋级候选画：

```text
full XY trajectory
first half XY trajectory
second half XY trajectory
per-frame aligned translation error
sliding 50/100/200f ATE
Sim3 scale over time
cumulative yaw drift
per-axis x/y/z error
```

必须同时显示：

```text
H9
C9
candidate
NOGTPOSE diagnostic as dashed non-TTT reference
```

### 10.2 Candidate bank heatmap

```text
x-axis = candidate id
y-axis = chunk id
color = future ATE delta vs H9
outline = downstream regression
```

### 10.3 State/delta direction heatmap

```text
layer × branch update norm
layer × branch update cosine to H9
layer × branch update cosine to C9
layer × branch harmful-basis coefficient
layer × branch continuity-basis coefficient
```

### 10.4 Selector scatter

```text
proxy score vs oracle future ATE
proxy score vs full ATE delta
scale proxy vs [200,300) delta
overlap residual vs downstream [400,600) delta
```

### 10.5 Failure gallery

对 largest regression chunks 输出：

```text
RGB
D_g
semantic group
TTT conflict map
candidate chosen
candidate rejected
future error curve
```

---

## 11. 并行执行策略

为了加速，不再把所有事情串行等待。分四条并行线，但有 gate 约束。

### Track A：Reproducibility Gate

负责人：Codex-A  
资源：GPU 0/1，full run 2 并发  
任务：H9/C9/WINGAM repeat、config diff、state save side-effect 排查。

### Track B：Sandbox Engineering

负责人：Codex-B  
资源：CPU + 1 GPU smoke  
任务：state save/load parity、suffix rollout、snapshot footprint control。

Track B 可以在 Track A 跑 full 的同时写工具，但不能发布 Phase 1 结论，直到 Track A 通过。

### Track C：Offline Candidate Bank Tooling

负责人：Codex-C  
资源：CPU  
任务：候选定义、debug table、oracle selection report、visualization skeleton。

Track C 可先用已有 H9/C9 artifacts 做 dry-run，但不能启动 online candidate full。

### Track D：New Action Family Prototyping

负责人：Codex-D  
资源：短 smoke  
任务：post-zeropower basis、scale-aware replay、dual-bank、finite-diff hooks。

Track D 只做 `END_FRAME <= 180` smoke，不跑 full，直到 Phase 2 指出需要新 action family。

---

## 12. 停止规则

### 12.1 Repro 停止规则

如果连续三次 H9 repeat 都不能进入 `34.1258 ± 0.03m`，停止所有 TTT action full run，只做 config / code diff。

### 12.2 Sandbox 停止规则

如果 native sandbox 无法复现 full suffix，禁止 candidate bank。不得用不可信 sandbox 得出 action-space 结论。

### 12.3 Candidate bank 停止规则

如果 oracle bank 中没有 candidate 对 future window 改善超过 `3m`，停止 selector，转新 action family。

### 12.4 Selector 停止规则

如果 Spearman `<0.3`，停止 full selector run，先扩展 proxy features。

### 12.5 Action family 停止规则

同一 family 连续 4 条 full run：

```text
不超过 H9；
不改善 [200,300) >= 3m；
或改善 [200,300) 但 [400,600) 恶化 >= 3m；
```

则停止该 family。

---

## 13. 预期决策树

### 情况 A：H9 复现问题修复，candidate bank oracle 有明显上界

继续 Phase 3 no-GT selector。若 selector 成功，进入 full online validation；若 selector 不成功，扩展 proxy 或用 MPC-style short rollout selector。

### 情况 B：H9 复现修复，但 candidate bank oracle 无上界

说明现有 commit policy bank 不足。进入 Phase 4 新 TTT action family，优先 post-zeropower basis 和 scale-aware replay。

### 情况 C：new action family 仍无上界

结论：当前 TTT write-only 难以作为 Target-25 主线。此时 TTT 仍保留为 read/memory regularizer，但 Target-25 需要显式 online trajectory-state / scale-state 模块。该结论只能在 Phase 4 后得出，不能提前用 v14 Phase0 失败得出。

### 情况 D：NOGTPOSE 仍强，但 TTT action 接不进去

不要把 NOGTPOSE 伪装成 TTT success。单独开一个 online trajectory-state module 分支，并在报告中明确：

```text
TTT write = auxiliary regularizer
trajectory-state module = target-25 mainline
```

---

## 14. 本轮最小可执行清单

第一天只做这些：

```text
1. P0_A1/A2/A3 no-state repeat；
2. P0_B1/B2 input-only state save repeat；
3. config_diff_report for H9 locked vs v14 H9 R3；
4. sandbox native suffix smoke using C9 if H9 still unresolved；
5. candidate bank tooling dry-run，不启动 full candidate。
```

第二天条件分支：

```text
If H9 fixed:
    run Phase 1 sandbox parity and Phase 2 candidate bank oracle.

If H9 not fixed but C9 stable:
    isolate H9-specific readbeta/write config path.

If both unstable:
    stop all new action runs and rebuild reproduction from B0/SWKS3.
```

---

## 15. 最终判断标准

本轮真正成功不是跑出一个 `33.7m` repeat，而是满足以下任一：

```text
Strong success:
    deployable online TTT/HMC ATE <= 25.0m

Stage success:
    deployable online TTT/HMC ATE <= 32.5m
    and [200,300) improves >= 5m
    and [400,600) does not regress

Action-space success:
    oracle candidate bank or new TTT action family shows future rollout gain >= 5m
    and no-GT selector has Spearman >= 0.5

Diagnostic success:
    H9 reproducibility issue is resolved,
    sandbox parity is established,
    and we can decisively say whether TTT commit bank has or lacks upper bound.
```

如果只能得到：

```text
ATE 33.7m repeat
[200,300) worse than H9
no sandbox
no candidate bank
```

则这不是进展，只是旧结果复现。


# ACL2 v54：Fast State-Conditioned Adaptive TTT Clean-to-C9 实验计划

日期：2026-06-09  
对象：LoGeR / HMC Pipeline v2 / KITTI Odometry sequence 01  
第一目标：构造一个 **无 absolute chunk-id、无手工 tri-replay 百分比、runtime 小于 28 分钟** 的 clean adaptive TTT writing 策略，并尽量接近 `C9_P0_R2 = 33.7629421029m`。  
第二目标：只有第一目标达到 soft pass 后，才重新接入语义，探索语义是否能真实改进几何重建。  
本轮明确边界：**暂不考虑 SWA，暂不推进语义主线，暂不追求 ATE < 30m。**

---

## 0. 为什么需要 v54

v53 没有成功，但它给了一个很明确的结论：现在的 runtime 和审计边界已经基本干净，真正失败的是 adaptive TTT 写入的 action space。

v53 已经做到：

```text
1. SC-GammaSplit / SC-GammaCommit / sampled ConflictLite 都能在 28 分钟内完成 full KITTI01。
2. no-chunk policy audit 通过。
3. no-manual-percentage audit 通过。
4. role collapse 没有发生。
5. full runtime gate 通过。
```

但是性能失败：

```text
V53_FULL_A_SCGAMMASPLIT_AW110:
    ATE = 35.9128586669m
    delta vs C9 = +2.1499165639m

V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075:
    ATE = 35.7408969581m
    delta vs C9 = +1.9779548552m
```

这说明：

```text
runtime blocker 已经被修掉；
action space blocker 仍然存在。
```

v53 Phase7 还发现一个非常关键的工程事实：`adaptive_writer_sc_gamma_split` 的 sc-gamma 分支之前一直固定 `rho=0.005`，导致 layer gamma 实际没有进入强度计算。修复后，rho sweep 立即改变轨迹，H35 刷新了 v53 best。这说明：**之前部分“参数无效”的结论来自接线问题，而不是策略本身完全无信号。** 但 Phase8 再对 layer / branch / rho 做细化，没有突破 H35，说明当前 layer8+17 / branch0 / rho 空间已经接近平台。

因此 v54 不能继续做：

```text
role threshold 小扫；
rho 小扫；
path length proxy 小扫；
旧 statecommit 距离规则小扫；
384F 看起来接近 C9 就升级 full。
```

v54 要做的是：**用 C9 teacher trace 找到当前 adaptive writer 缺少的 state variable，然后设计 state-conditioned adaptive TTT writer。**

---

## 1. 本轮不可违反的约束

### 1.1 算法约束

本轮第一目标只允许使用：

```text
1. C9_P0_R2 的 frame-attention READ cue，保持不变。
2. TTT writing adaptive policy。
3. 不使用 SWA 作为优化主线。
4. 不使用 semantic policy 作为优化主线。
```

禁止：

```text
1. 禁止 absolute chunk-id policy。
   包括但不限于：
       chunks 5-9
       chunks 10-12
       chunk16
       commit_ema_chunks=5,6

2. 禁止手工指定 tri-replay role percentage。
   包括但不限于：
       positive_frac = 0.35
       negative_frac = 0.12
       neutral_lambda = 0.85 作为固定 replay 百分比或固定角色比例

3. 禁止训练 trigger / classifier / selector / role router。

4. 禁止用 GT ATE / GT pose / absolute frame id 作为 runtime action。

5. 禁止以 short-window ATE 作为成功结论。
```

允许：

```text
1. 使用全局固定常数，例如全局 clamp、全局 epsilon、全局 rho 上限。
2. 使用当前 chunk / 当前历史状态可计算的 no-GT state features。
3. 使用 C9 trace 做离线诊断，但不能在 runtime 中使用 C9 的 chunk id 表。
4. 使用 704F 作为 runtime / catastrophic filter，但不能把 704F 结果写成 full success。
```

### 1.2 效率约束

每条 full KITTI01 run 必须满足：

```text
wall time <= 28 min
chunk_total_seconds_mean <= 42s
probe_ttt_write_seconds_mean <= 8s
```

如果某条候选在前 5 个 chunk 后预测 full wall time 超过 28 分钟，Codex 必须立即停止该候选，不允许继续跑完后再报告失败。

---

## 2. 当前研究进展带来的启发

### 2.1 C9 的贡献已经比较清楚

v46B positive-only factorial 说明，在 clean no-chunk 条件下：

```text
ONLY_FRAME_ATTN gain ≈ 3.1568m
ONLY_TTT gain ≈ 2.2311m
ONLY_SWA gain ≈ 0.0132m
FRAME_ATTN + TTT gain ≈ 5.0813m
```

这意味着：

```text
READ 和 TTT 是主贡献；
SWA 不是当前 clean-to-C9 的主线；
READ + TTT 的交互是关键。
```

因此 v54 不再管 SWA。

### 2.2 adaptive TTT 的失败不是“没有自适应”

v47-v53 已经实现了多种 no-chunk / no-manual-percentage adaptive writer：

```text
v47 Otsu fused
v48 robust fused
v49 residual_x_dg fused
v50 robust split
v52 EnergyMatched split
v52 Cluster3D split
v53 SC-GammaSplit
v53 SC-GammaCommit
v53 sampled ConflictLite
v53 layergammafix H35
```

这些方法的共同边界是：

```text
它们能跑；
它们不依赖 chunk-id；
它们不使用手工 tri percentage；
它们多数 runtime 可接受；
但 full ATE 都停在 35.7--36.3m 平台，离 C9 还有约 2m。
```

所以失败不是因为“没有 adaptive 代码”，而是因为当前 adaptive writer 只在 token role / light gamma 上做局部修正，没有学到 C9 的长期写入行为。

### 2.3 C9 的核心可能是 state-conditioned timing，而不是 token split

C9 的手工配置包含：

```text
read_beta_frame_chunks:
    5-9: 4.85
    10-12: 4.25
    16: 4.25

ttt_write_gradient_reversal_chunk_gammas:
    5-9: 0.005
    10-12: 0.003
    16: 0.0003

ttt_write_tri_replay_chunk_params:
    5-12: 0.35/0.12/0.85
    16: 0.35/0.08/0.85

commit_ema_chunks:
    5,6
```

我们不能保留这些 absolute chunk-id，但必须理解它们在做什么。v53 的结果表明：简单用 risk / role mass / native delta 距离不能复现这种离散时序动作。当前更像是 **state aliasing**：一些 chunk 在局部可见特征上很像，但 C9 对它们采取不同动作。我们缺的是能够区分这些状态的 no-GT state features。

---

## 3. 本轮总体假设

v54 围绕三个假设设计。

### 假设 H1：当前 adaptive writer 缺少 state variable，而不是缺少更多 role threshold

如果 H1 成立，那么继续扫 role threshold、gamma clamp、rho、layer set 不会有明显收益。真正需要的是从 C9 trace 中找出：

```text
什么时候应该强写；
什么时候应该弱写；
什么时候应该 filtered commit；
什么时候应该主要保持 native continuity。
```

这些动作必须由 no-GT state features 决定，而不是 chunk id。

### 假设 H2：C9 的有效性体现在 post-zp update geometry

TTT 写入不是普通 token 加权。pre-update token role 经过 replay、zeropower、norm restoration 后，最终影响的是 fast-weight delta 的方向、范数和 branch/layer 分布。因此 adaptive writer 不能只匹配：

```text
positive mass
neutral mass
negative mass
```

还必须匹配：

```text
post-zp delta norm
candidate-vs-native delta cosine
branch/layer energy
commit delta magnitude
historical EMA state movement
```

### 假设 H3：704F 比 384F 更适合作为 full promotion filter，但仍不能替代 full

v53 证明 384F 可能接近 C9，但 full 会失败。H35 的 704F 与 full 方向一致，但幅度有限。因此 v54 使用 704F 作为 promotion filter，要求候选必须先在 704F 超过 H35/A-best，才允许 full。

---

## 4. Phase 0：代码与效率 hard gate

### 4.1 目标

在任何新算法实验前，先确认：

```text
1. C9 reference 可复现。
2. v53 H35 reference 可复现或可读取。
3. no-chunk audit 可用。
4. no-manual-percentage audit 可用。
5. runtime profiler 可用。
6. 自动 early-stop 可用。
```

### 4.2 必须记录的指标

每条候选都必须落盘：

```text
candidate_name
frames
hmc_rows
wall_time_min
projected_full_wall_min_after_5_chunks
chunk_total_seconds_mean
probe_ttt_write_seconds_mean
no_chunk_policy_pass
manual_percentage_audit_pass
role_collapse_count
tri_replay_role_mode
risk_source
split_debug_count
fused_debug_count
stage_c_disabled
swa_disabled_or_neutralized
```

### 4.3 Gate

```text
C9 repeat gate:
    abs(ATE - 33.7629421029) <= 0.03m

runtime gate:
    projected_full_wall_min_after_5_chunks <= 28
    chunk_total_seconds_mean <= 42s
    probe_ttt_write_seconds_mean <= 8s

clean gate:
    no_chunk_policy_pass = true
    manual_percentage_audit_pass = true
    role_collapse_count = 0
```

如果 Phase 0 不通过，Codex 不允许启动 704F 或 full；必须先修 launcher / CLI / HMC 接线。

---

## 5. Phase 1：C9 teacher behavior autopsy v2

### 5.1 目标

这一步不跑新 ATE。它只对比：

```text
C9 teacher
v53 H35 best clean adaptive
v50 robust split
v52 EnergyMatched
```

目的是找出 clean adaptive 与 C9 的真实差异。

### 5.2 需要导出的 per-chunk state features

每个 chunk 记录：

```text
D_g_mean
D_g_q90
D_g_mass_gt_05
D_g_entropy
read_source_risk_mass
stage_d_mean
stage_d_q10
write_score_mean
role_pos_mass
role_neu_mass
role_neg_mass
risk_mean
risk_q90
risk_spread_q90_q10
adaptive_gamma
adaptive_neutral_lambda
post_zp_delta_norm_total
post_zp_delta_norm_by_layer
post_zp_delta_norm_by_branch
candidate_native_delta_cosine
candidate_native_delta_norm_ratio
commit_delta_norm
native_delta_norm
commit_native_cosine
commit_ema_effective_alpha_if_any
state_norm_before
state_norm_after
state_norm_delta
pose_step_scale_proxy
pointmap_overlap_consistency_proxy
reset_phase_diagnostic_only
```

注意：`reset_phase_diagnostic_only` 可以用于离线解释，不允许作为 v54 runtime decision feature。

### 5.3 需要生成的图

```text
teacher_student_post_zp_delta_timeline.png
teacher_student_layer_branch_heatmap.png
teacher_student_gamma_timeline.png
teacher_student_commit_delta_timeline.png
teacher_student_candidate_native_cosine.png
risk_spread_vs_c9_gamma_scatter.png
seg0_seg1_seg2_error_timeline.png
```

### 5.4 判断标准

Phase 1 不是性能 gate，而是设计 gate。必须回答：

```text
1. C9 相比 H35/B1 的主要差异是 gamma timing，还是 commit behavior？
2. 差异集中在哪些 layer / branch？
3. H35 改善 seg1 但 seg2 失败，失败从哪个 chunk 开始？
4. 哪些 no-GT feature 在失败点前已经发生变化？
```

如果 Phase 1 无法生成这些字段，Codex 必须先补 instrumentation，不允许继续写新策略。

---

## 6. Phase 2：只实现两个新 adaptive TTT candidate

v54 不允许开大矩阵。只实现两个 candidate：M1 和 M2。它们都必须：

```text
保持 C9 READ cue 不变；
不使用 SWA；
不使用 semantic；
不使用 absolute chunk id；
不使用手工 tri replay percentage；
runtime <= 28min full projection。
```

---

### 6.1 Candidate M1：State-Energy Matched Split

#### 核心思想

v50/v52/v53 的 role split 已经能跑，但 post-zp energy 不像 C9。M1 不再只根据 role mass 设置 gamma，而是让当前 chunk 的 update energy 对齐历史 state 的稳定范围。

#### Runtime 输入

```text
risk vector r_i
write prior p_i
native update delta Δ_native
candidate split update delta Δ_candidate
EMA of previous accepted update energy E_ema
D_g distribution
stage_d distribution
```

#### Role assignment

Role 仍由自适应阈值决定，不用百分比：

```text
safety_i = norm(p_i) * (1 - norm(r_i))
danger_i = norm(r_i) * (1 + 1 - norm(p_i))

positive = safety_i > median(safety) + k_s * MAD(safety)
negative = danger_i > median(danger) + k_d * MAD(danger)
neutral = others
```

这里 $k_s$ 和 $k_d$ 是全局固定常数，不随 chunk 改变，也不是百分比。

#### Gamma / neutral lambda

定义：

$E_c = ||\Delta_{candidate}||$

$E_n = ||\Delta_{native}||$

$E_t = \operatorname{EMA}(E_n)$

$g_{energy}=\operatorname{clip}\left(\frac{E_t}{E_c+\epsilon}, g_{min}, g_{max}\right)$

$g_{risk}=\operatorname{clip}(\operatorname{MAD}(r) \cdot \operatorname{mass}_{neg}, 0, 1)$

最终：

$\gamma_{eff}=\gamma_0 \cdot g_{energy} \cdot (1 + g_{risk})$

neutral lambda：

$\lambda_{neu}=\operatorname{clip}\left(1 - \operatorname{mean}(r_{neutral}), \lambda_{min}, \lambda_{max}\right)$

关键要求：

```text
γ_eff 不能来自 chunk map；
positive/negative/neutral role 不能来自 fixed percentage；
E_t 来自 causal EMA，不看未来。
```

---

### 6.2 Candidate M2：State-Energy Matched Split + Directional Commit Guard

#### 核心思想

v53 的 SC-GammaCommit 只用 candidate/native 距离触发 commit filter，结果变差。M2 改成方向与能量双重 guard：不是 candidate 距 native 越近越好，而是 candidate 更新方向必须与 native continuity 不冲突，并且不能能量过冲。

#### Runtime 输入

同 M1，额外使用：

```text
cos(Δ_candidate, Δ_native)
energy ratio E_c / E_n
state movement norm
```

#### Commit mixing

定义：

$c = \cos(\Delta_{candidate}, \Delta_{native})$

$u = \frac{E_c}{E_n+\epsilon}$

方向 gate：

$a_{cos}=\operatorname{clip}\left(\frac{c-\tau_c}{1-\tau_c},0,1\right)$

能量 gate：

$a_{energy}=\operatorname{clip}\left(\frac{u_{max}-u}{u_{max}-1},0,1\right)$

最终 commit：

$\alpha_{commit}=\operatorname{clip}(a_{cos} \cdot a_{energy}, \alpha_{min}, 1)$

$W_{next}=\alpha_{commit} W_{candidate} + (1-\alpha_{commit}) W_{native}$

这里 $\tau_c$、$u_{max}$、$\alpha_{min}$ 是全局常数；不能按 chunk 改。

#### 预期

M2 不是为了压小所有更新，而是为了避免 adaptive candidate 在长期后段 drift 中把 useful continuity 方向抹掉。

---

## 7. Phase 3：704F screen，不再使用 384F 作为性能依据

### 7.1 为什么用 704F

v53 已证明：

```text
384F 接近 C9 不代表 full 成功；
H8/H14 384F 接近 C9，但 full 失败；
704F 与 full 方向更一致，虽然仍不能替代 full。
```

因此 v54 的升级规则是：**候选必须先过 704F，才允许 full。**

### 7.2 704F candidates

只跑：

```text
M1_704F
M2_704F
H35_704F_reference_if_missing
```

### 7.3 704F 记录指标

```text
ATE704
seg0_rmse
seg1_rmse
seg2_rmse
rolling50_mean / p90 / worst
rolling100_mean / p90 / worst
post_zp_delta_norm_timeline
candidate_native_delta_cosine_timeline
commit_alpha_timeline
runtime_wall_min
projected_full_wall_min
```

### 7.4 704F promotion gate

允许进入 full 必须满足全部：

```text
1. runtime projection <= 28min；
2. ATE704 <= H35_704F - 0.20m；
3. seg2_rmse <= H35_seg2 - 0.50m 或 rolling100_p90 <= H35_rolling100_p90 - 0.50m；
4. no_chunk_policy_pass = true；
5. manual_percentage_audit_pass = true；
6. role_collapse_count = 0。
```

如果 M1/M2 都不过，不允许 full。

---

## 8. Phase 4：最多两条 full KITTI01

### 8.1 Full candidates

最多运行：

```text
F1 = best of M1/M2 from 704F
F2 = second candidate only if it improves a different segment or has better runtime/energy stability
```

不再运行：

```text
extra rho sweep
extra role threshold sweep
extra path-length proxy
extra statecommit variant
```

### 8.2 Full 记录指标

```text
ATE_full
Rot_full
RPE_t
FinalErr
seg0_rmse
seg1_rmse
seg2_rmse
[200,300]
[400,600]
rolling50_mean / p90 / worst
rolling100_mean / p90 / worst
rolling200_mean / p90 / worst
wall_time_min
chunk_total_seconds_mean
probe_ttt_write_seconds_mean
role_mass_timeline
post_zp_delta_norm_timeline
commit_alpha_timeline
candidate_native_cosine_timeline
state_norm_delta_timeline
```

### 8.3 Full 判断标准

本轮第一目标不是 Target-30，而是 clean adaptive TTT 接近 C9。

```text
progress pass:
    ATE <= 35.30m

soft pass:
    ATE <= 34.60m

close-to-C9 pass:
    ATE <= 34.30m

excellent pass:
    ATE <= 34.06m  # C9 + 0.30m
```

如果 full ATE 仍然大于 `35.30m`，本轮判定：当前 M1/M2 仍未进入可用 clean adaptive TTT baseline。

---

## 9. 失败分流规则

### 9.1 如果 M1/M2 704F 都失败

判断：

```text
state-energy matching 仍不足以恢复 C9 timing。
```

Codex 下一步不要继续扫 k_s/k_d/gamma clamp，而应做：

```text
1. 检查 Phase 1 features 是否缺失关键 state variable；
2. 比较 C9 vs M1/M2 的 post-zp delta layer/branch heatmap；
3. 查哪个 layer/branch 的 energy mismatch 最大；
4. 只允许提出一个新的 action-space change，而不是参数 sweep。
```

### 9.2 如果 704F 过但 full 失败

判断：

```text
长期 drift 仍未捕捉。
```

Codex 必须生成：

```text
full_vs_704_failure_autopsy.md
failure_start_chunk.csv
seg_drift_timeline.png
post_704_state_drift_timeline.png
```

不得继续用 704F 当成功指标。

### 9.3 如果 M2 比 M1 更差

判断：

```text
directional commit guard 当前削弱了 useful continuity。
```

后续优先回到 M1，并只分析 commit_alpha 是否过低或过频繁。

### 9.4 如果 runtime 超过 28min

判断：

```text
效率 gate fail。
```

Codex 必须先优化：

```text
1. 减少 trace field；
2. 只在 selected layers 计算 delta cos；
3. 降低 post-zp full tensor logging 频率；
4. 禁用非必要 JSONL debug；
5. 确认 empty_cuda_cache_each_chunk = 0。
```

不允许继续跑 full。

---

## 10. 第二目标：语义何时重新进入

只有当第一目标至少达到：

```text
ATE <= 34.60m
```

才允许进入语义 Phase。

语义只允许从 READ / C23 进入，不允许直接进入 TTT writing 主线：

```text
allowed:
    semantic-conditioned C23 residual
    high-influence anomaly READ filtering
    semantic trust calibration for D_g interpretation

not allowed yet:
    semantic TTT scalar write
    semantic all-memory role matrix
    semantic SWA policy
    semantic-trigger learned selector
```

理由：历史实验已经说明，语义 TTT/SWA 全局控制很弱，而 semantic-conditioned C23 / READ residual 有局部信号但 full transfer 失败。必须先把 clean adaptive TTT baseline 稳住，再测试语义是否能叠加。

---

## 11. 本轮交付物

Codex 必须交付：

```text
v54_code_audit.md
v54_phase1_teacher_student_autopsy.csv
v54_phase1_teacher_student_autopsy.md
teacher_student_post_zp_delta_timeline.png
teacher_student_layer_branch_heatmap.png
teacher_student_gamma_timeline.png
teacher_student_commit_delta_timeline.png
v54_704_registry.csv
v54_704_report.md
v54_full_registry.csv
v54_full_report.md
v54_failure_routing.md
```

如果没有 full run，必须解释是哪条 704F gate 阻止 full。

---

## 12. 最终一句话

v54 的目标不是再试一个小参数，而是回答一个更根本的问题：

> C9 的 chunk-wise 手工配方能否被无 chunk-id、无手工 percentage、效率可接受的 state-conditioned adaptive TTT writer 替代？

如果 v54 仍然失败，并且 M1/M2 都停在 35.7m 以上，那么我们就可以比较有把握地说：当前 adaptive TTT action space 不足以复现 C9，下一步必须重新设计 TTT write action，而不是继续在 risk / gamma / rho 上做小修小补。

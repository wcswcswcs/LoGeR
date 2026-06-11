# ACL2 v55：C9 Schedule Autopsy + Fail-Forward Adaptive TTT Clean Plan

日期：2026-06-09  
对象：LoGeR / HMC Pipeline v2 / KITTI Odometry sequence 01  
当前参考：`C9_P0_R2`，full online ATE = `33.76294210291885m`  
当前 clean adaptive 参考：`H35`，full online ATE = `35.74089695811434m`  
本轮第一目标：**在不使用 absolute chunk-id policy、不使用手工 tri-replay percentage、不使用 SWA 主线、不引入语义的条件下，让 clean adaptive TTT writer 接近 C9。**  
本轮第二目标：**解决实验推进太慢的问题。Codex 不能因为 gate 未过就只停下；必须自动产出失败归因，并执行一条有依据的 fail-forward 修复。**

---

## 0. 本计划为什么重写

v54 没有成功，不只是算法失败，也是计划设计失败。v54 只实现并筛选了两个候选：

```text
M1 = State-Energy Matched Split
M2 = State-Energy + Directional Commit Guard
```

M1 的 704F ATE 是 `40.09525341863154m`，比 H35 704F 差 `+0.2970056981459063m`。M2 的 704F ATE 是 `41.38387973914448m`，比 H35 704F 差 `+1.5856320186588473m`，且 projected full runtime `28.474666666666668min` 超过 28min gate。因为 v54 gate 过硬，M1/M2 未过 704F gate 后没有 full run，也没有足够失败归因。

这说明两件事：

```text
1. M1/M2 当前设计没有突破 H35 平台。
2. 计划不能再写成 “gate fail -> stop”。
```

本轮 v55 的关键改动是：**实验必须 fail-forward**。如果候选失败，Codex 必须立即回答“为什么失败”，并根据失败类型启动一个有限、明确、低成本的修复，而不是把结果停在 fail 表格里。

---

## 1. 项目边界与禁止项

本轮只服务第一目标：clean adaptive TTT 接近 C9。语义和 SWA 不作为主线。

### 1.1 必须满足

所有可计入本轮 clean adaptive TTT candidate 的 full run 必须满足：

```text
1. no absolute chunk-id policy
2. no manually assigned tri-replay positive / neutral / negative percentage
3. no learned trigger / selector / classifier / role router
4. no per-sequence tuning
5. no semantic runtime action
6. no SWA mainline dependency
7. full KITTI01 runtime <= 28 minutes, or candidate is invalid for this round
```

### 1.2 禁止再次走偏

不允许：

```text
1. 继续大规模 role-threshold / gamma / rho 小扫；
2. 用 chunk id 或固定 chunk window 模拟 C9；
3. 用 [200,300) 或 KITTI01 特定窗口作为 runtime condition；
4. 把 384F / 704F 当 full success；
5. 继续用 fused replay 作为主线；
6. 继续把 semantic 写进 TTT 主线；
7. gate fail 后只写 “not promoted” 而不做失败归因。
```

---

## 2. 当前事实与核心判断

### 2.1 已经比较确定的事实

从 v46B clean factorial 和 v45/v52 attribution 已经知道：

```text
READ / frame-attention control 单独贡献大；
TTT write control 单独贡献大；
SWA 单独贡献几乎为 0；
READ + TTT 是最重要组合；
exact C9 的主要额外收益来自 TTT tri-replay、tri gamma schedule、commit EMA。
```

从 v47-v54 已经知道：

```text
1. fused adaptive replay 明显不够；
2. split replay 是必要条件；
3. residual_x_dg risk 比 d_tok 更合理，但仍不够；
4. 当前 SC-gamma / energy matching / directional commit guard 仍停在 H35 附近；
5. 704F 对 seg2 后段判断很弱，不能单独代表 full；
6. runtime blocker 基本解决，action-space blocker 仍然存在。
```

### 2.2 我的当前判断

C9 的强点不是简单的：

```text
把 token 分成 positive / neutral / negative 三类。
```

C9 更像是一个 **TTT write schedule**：

```text
在某些 memory state 下，增加或减少 TTT update energy；
在某些状态下启用 commit EMA；
在某些 branch/layer 上更强地改变 fast weights；
通过 READ + TTT 的相互配合，稳定后段 trajectory。
```

所以 v55 不再只问：

```text
adaptive split 怎么分组？
```

而是问：

```text
H35 / M1 与 C9 的差距到底来自 role mass、gamma energy、commit behavior、branch/layer direction，还是 reset-state timing？
```

---

## 3. 本轮核心假设

### 假设 H1：H35 离 C9 差距主要来自 seg2 / long-tail state，而不是前 384F

v53/v54 已经显示，部分候选在 384F 或 704F 看起来并不灾难，但 full 后仍然停在 `35.7m` 附近。H35 full 的 seg2 仍远差于 C9。本轮必须专门看：

```text
seg0: frames 000-384
seg1: frames 384-700
seg2: frames 700-end
rolling100 p90 / worst
final error
```

如果候选只改善 seg1，不改善 seg2，就不能继续小扫。

### 假设 H2：C9 的 chunk map 可能是 reset-state / memory-state 的粗代理

C9 的 chunk-specific 参数不能保留，但不能简单固定成全局常数。更合理的解释是：

```text
C9 的手工 chunk map 可能在近似表达某些 state：
    reset 后 state age；
    native/candidate update divergence；
    post-zp update energy；
    static anchor / high-D mass；
    branch/layer delta pattern；
    commit risk。
```

本轮要找到可替代 chunk-id 的 online state variable，而不是继续手写 chunk list。

### 假设 H3：现有 adaptive writer 只修了 role split，没有学到 commit schedule

M2 directional commit guard activation rate = `1.0`，commit scale mean 约 `0.284`，说明它几乎每个 chunk 都强行收缩 candidate，这不是选择性 commit policy。一个有效 commit policy 应该只在少数高风险 state 下生效，并且需要解释何时保留 candidate、何时拉回 native。

### 假设 H4：需要先做 autopsy，再设计新 action

如果不先找出 C9-H35 gap 的最大来源，继续命名 M3/M4 没有意义。本轮必须先做 trace autopsy，并且新 candidate 只能修 autopsy 指出的最大 gap。

---

## 4. 实验总流程

本轮最多允许：

```text
1 条 v54 salvage full diagnostic
1 次 teacher-student autopsy
2 条新 full candidates
1 条 emergency repair full candidate
```

也就是说，最多 4 条 full KITTI01。每条 full 必须 projected runtime <= 28min。任何阶段失败，都必须写 failure-routing artifact，而不是沉默停止。

---

## 5. Phase 0：v54 salvage 与效率 hard gate

### 5.1 目标

先补救 v54 的计划设计错误：M1 704F 只比 H35 差 `+0.297m`，属于 borderline。若 M1 full runtime 预计不超过 28min，应补跑 **M1 full diagnostic**，但不能把它写成 expected success。

### 5.2 实验

运行：

```text
S0_M1_FULL_DIAGNOSTIC
```

前提：

```text
projected full runtime <= 28min
no_chunk_policy_pass = true
manual_percentage_audit_pass = true
role_collapse_rows = 0
```

M2 不补 full，因为：

```text
704F 明显差；
projected full runtime > 28min；
commit guard activation rate = 1.0，已明显不健康。
```

### 5.3 记录指标

```text
full ATE / Rot / FinalErr
seg0 / seg1 / seg2 RMSE
rolling50 / rolling100 / rolling200 mean / p90 / worst
runtime wall min
chunk mean seconds
probe_ttt_write mean seconds
adaptive gamma timeline
role mass timeline
commit alpha timeline
branch/layer post-zp delta timeline
```

### 5.4 判断标准

```text
如果 S0_M1_FULL ATE <= H35 full + 0.30m:
    保留为 borderline action-space evidence。

如果 S0_M1_FULL ATE > H35 full + 0.30m:
    M1 确认为无效，不再扩展。
```

不管结果如何，都进入 Phase 1 autopsy。

---

## 6. Phase 1：C9-H35-M1 Trace Autopsy，不跑新算法

### 6.1 目标

这一阶段要把 “C9 为什么强” 从全局 ATE 拆成可操作差距。对比对象：

```text
Teacher: C9_P0_R2
Student A: H35, 当前 best clean adaptive
Student B: M1 full diagnostic，如果 Phase 0 补跑成功
Student C: v50 split residual_x_dg，可作为早期 split reference
```

### 6.2 必须输出的表

Codex 必须生成：

```text
phase1_trace_autopsy/
    c9_h35_m1_chunk_gap_table.csv
    c9_h35_m1_segment_gap_table.csv
    c9_h35_m1_layer_branch_delta_table.csv
    c9_h35_m1_role_mass_table.csv
    c9_h35_m1_commit_behavior_table.csv
    c9_h35_m1_state_variable_table.csv
    c9_h35_m1_autopsy_report.md
```

### 6.3 必须记录的变量

每个 chunk 记录：

```text
trajectory:
    chunk index only for analysis, not policy input
    frame range
    local chunk ATE proxy if available
    segment id: seg0 / seg1 / seg2
    rolling100 contribution

state variables:
    reset_age = chunk index since last reset
    reset_phase = chunk index modulo reset_every
    risk spread
    D_g mean / p80 / p90 / high-mass
    stage_d mean / low-mass
    residual_x_dg mean / p90
    native-candidate delta norm
    candidate-native cosine
    post-zp delta norm
    post-zp delta ratio to native
    static-anchor proxy mass

TTT write variables:
    positive role mass
    neutral role mass
    negative role mass
    effective gamma
    neutral lambda
    commit alpha / commit EMA effective strength
    branch w0 / w1 / w2 delta norm
    layer delta norm
    layer-branch heatmap values
```

### 6.4 必须可视化

```text
teacher_student_post_zp_delta_timeline.png
teacher_student_gamma_timeline.png
teacher_student_commit_alpha_timeline.png
teacher_student_role_mass_timeline.png
teacher_student_layer_branch_heatmap.png
segment_error_timeline.png
c9_minus_h35_gap_by_chunk.png
state_variable_vs_teacher_gamma_scatter.png
state_variable_vs_commit_alpha_scatter.png
```

### 6.5 必须回答的问题

Phase 1 report 必须明确回答：

```text
Q1: H35 和 C9 最大差距发生在 seg0 / seg1 / seg2 哪个阶段？
Q2: H35 的 role mass 是否接近 C9？
Q3: H35 的 post-zp delta energy 是否接近 C9？
Q4: H35 的 branch/layer delta pattern 是否接近 C9？
Q5: C9 的 commit EMA 实际影响是否只在少数 chunks 出现？
Q6: reset_age / state variables 是否能解释 C9 gamma / commit 行为？
Q7: M1 相比 H35 是修了最大 gap，还是修了无关变量？
```

### 6.6 Phase 1 决策

Phase 1 必须把失败类型归入下列之一：

```text
TYPE_A_ROLE_MASS_GAP:
    adaptive role split 与 C9 差距最大。

TYPE_B_GAMMA_ENERGY_GAP:
    role mass 接近，但 post-zp delta / gamma energy 差距最大。

TYPE_C_COMMIT_SCHEDULE_GAP:
    candidate write 本身不差，但 commit behavior 与 C9 差距最大。

TYPE_D_LAYER_BRANCH_ACTION_GAP:
    总能量接近，但 layer/branch delta pattern 错。

TYPE_E_SEG2_STATE_GAP:
    前中段接近，seg2 / tail 崩，说明缺长期 state rule。
```

只有完成这个分类，才能进入 Phase 2。

---

## 7. Phase 2：只实现两个 autopsy-driven candidates

Phase 2 不能提前指定复杂矩阵。候选由 Phase 1 failure type 决定。Codex 必须按照下面 routing 自动选择两个候选。

### 7.1 如果是 TYPE_A_ROLE_MASS_GAP

实现：

```text
Candidate A1: RoleSplitV3
```

定义：

```text
role score 不再只用 residual_x_dg。

positive evidence:
    high stage_d
    low D_g
    low residual_x_dg
    high static-anchor proxy

negative evidence:
    high D_g
    high residual_x_dg
    low stage_d
    high candidate-native conflict

neutral:
    neither positive nor negative
```

阈值：

```text
用当前 chunk 内 robust z-score / MAD 自适应；
不允许 top-k percentage；
不允许 fixed positive/negative fraction；
如果 role collapse，记录 collapse，不用固定 percentage 修补。
```

### 7.2 如果是 TYPE_B_GAMMA_ENERGY_GAP

实现：

```text
Candidate B1: TeacherEnvelopeGammaV2
```

定义：

```text
不是拟合 C9 chunk id，而是用当前 state 的 energy envelope 控制 gamma。

state_energy = robust EMA(post_zp_native_delta_norm)
risk_energy = residual_x_dg_p90 * D_g_high_mass
candidate_energy = predicted negative replay delta norm

gamma_eff = gamma_base * energy_ratio_guard
```

其中：

$$
energy\_ratio\_guard = \operatorname{clip}\left(\frac{target\_energy}{candidate\_energy + \epsilon}, g_{min}, g_{max}\right)
$$

`target_energy` 由 online EMA 和 reset-age warmup 共同决定，不能用 absolute chunk id。

### 7.3 如果是 TYPE_C_COMMIT_SCHEDULE_GAP

实现：

```text
Candidate C1: SelectiveCommitEMA
```

定义：

```text
不要像 M2 一样每个 chunk 都强行收缩。
commit EMA 只在以下条件同时成立时启用：
    candidate/native cosine 低；
    post-zp delta energy 超过 online EMA envelope；
    static-anchor mass 未 collapse；
    D_g high-mass 高于当前 reset block 的 robust median。
```

commit alpha 不是固定 chunk 表，而是：

$$
\alpha_{commit} = \operatorname{clip}\left(1 - r_{risk}, \alpha_{min}, 1\right)
$$

其中 $r_{risk}$ 来自当前 chunk 的 normalized commit risk。

### 7.4 如果是 TYPE_D_LAYER_BRANCH_ACTION_GAP

实现：

```text
Candidate D1: LayerBranchEnergyRouter
```

定义：

```text
不再全局同 rho。
根据每个 selected layer / branch 的 native delta energy 与 risk energy 分配 gamma。
```

候选只允许使用少量固定 layer group：

```text
layers = {0, 8, 17}
branch = w0 first
```

但强度由 state 决定，不由 chunk id 决定。

### 7.5 如果是 TYPE_E_SEG2_STATE_GAP

实现：

```text
Candidate E1: TailStateContinuityGuard
```

定义：

```text
如果当前 state 显示 long-tail continuity 风险高，降低 aggressive negative write，增强 neutral continuity。
```

输入变量：

```text
rolling state energy EMA
candidate/native cosine
neutral role risk mean
static-anchor mass
reset-age
post-zp delta overshoot
```

动作：

```text
neutral lambda 自适应提高；
negative gamma 自适应收缩；
commit EMA 只在 overshoot 时启用。
```

---

## 8. Phase 3：快速筛选，不再用硬 stop 浪费信息

### 8.1 96F smoke

每个候选先跑 96F：

```text
frames = 96
要求：
    no_chunk_policy_pass = true
    manual_percentage_audit_pass = true
    role_collapse_rate <= 0.05
    chunk_mean <= 42s
    probe_ttt_write_mean <= 8s
```

失败分流：

```text
如果 no_chunk 或 manual audit fail：修配置，不进入性能判断。
如果 runtime fail：先做 selected-layer / sampling / cache repair。
如果 role collapse：修 role z-score / threshold，不跑 704F。
```

### 8.2 704F screen

通过 96F 后跑 704F。判断不再是硬二元 gate。

```text
Promote to full:
    candidate_704F_ATE <= H35_704F + 0.10m

Borderline diagnostic full:
    H35_704F + 0.10m < candidate_704F_ATE <= H35_704F + 0.35m
    and projected full runtime <= 28min

Reject:
    candidate_704F_ATE > H35_704F + 0.35m
    or projected full runtime > 28min
```

这条规则是为避免 v54 再次出现 M1 borderline 但不 full 的情况。

### 8.3 必须记录

```text
704F ATE / Rot / FinalErr
seg0 / seg1 / rolling100 p90
projected full runtime
role mass
adaptive gamma
commit alpha
post-zp delta norm
candidate/native cosine
failure type tag
```

---

## 9. Phase 4：full KITTI01，最多两条主候选 + 一条 emergency repair

允许 full 的候选最多两条。如果两条都失败，但 failure report 显示同一 obvious bug 或 overly aggressive guard，可允许一条 emergency repair。

### 9.1 full 记录指标

```text
global:
    ATE
    Rot
    RPE_t
    RPE_r
    FinalErr
    runtime wall min
    chunk mean seconds
    probe TTT write seconds

segments:
    seg0 000-384 RMSE
    seg1 384-700 RMSE
    seg2 700-end RMSE
    [200,300]
    [400,600]

rolling:
    rolling50 mean / p90 / worst
    rolling100 mean / p90 / worst
    rolling200 mean / p90 / worst

TTT internals:
    role mass timeline
    gamma timeline
    neutral lambda timeline
    commit alpha timeline
    post-zp delta norm timeline
    branch/layer delta heatmap
    candidate/native cosine timeline

audit:
    no_chunk_policy_pass
    manual_percentage_audit_pass
    role_collapse_rate
    runtime_gate_pass
```

### 9.2 full 判断标准

```text
Progress pass:
    ATE <= 35.30m

Soft pass:
    ATE <= 34.60m

Close-to-C9 pass:
    ATE <= 34.30m

Excellent pass:
    ATE <= 34.06m

Fail:
    ATE > 35.30m
```

本轮不要求 ATE < 30m。当前必须先让 clean adaptive TTT 明显接近 C9。

---

## 10. Phase 5：如果仍失败，必须输出 action-space redesign conclusion

如果所有 full 候选都 `ATE > 35.30m`，Codex 不允许只写失败。必须输出：

```text
1. 当前 adaptive split action space 是否不足？
2. C9-H35 差距最大来源是否已经定位？
3. 是否需要改变 TTT action，而不是继续 role/gamma/commit 小修？
```

可能结论包括：

```text
A. 需要 branch/layer-specific replay，而不是只选 token role。
B. 需要 no-long-write / filtered commit，而不是 tri replay。
C. 需要 READ-TTT joint state controller，而不是 TTT-only writer。
D. 需要先回到 semantic / geometry read source，TTT clean 线暂时降级。
```

如果出现这种结论，下一轮不能继续命名 M4/M5 小修，必须重写 action space。

---

## 11. 效率计划

### 11.1 运行预算

```text
Phase 0 salvage full: <= 1 full
Phase 1 autopsy: 0 full
Phase 2 implementation: 0 full
Phase 3 smoke: <= 2 x 96F
Phase 3 screen: <= 2 x 704F
Phase 4 full: <= 2 full + 1 emergency repair
```

理论最大 full：4 条。实际目标：2-3 条。

### 11.2 并行策略

```text
1. autopsy/report 与 smoke 并行；
2. 两个 96F smoke 可并行；
3. 两个 704F screen 可并行；
4. full run 最多 2 条并行；
5. 每个 LoGeR 进程绑定一张 GPU；
6. 不允许同 GPU 多 full 并发；
7. 所有 full 必须 wall <= 28min。
```

### 11.3 效率失败处理

如果候选 runtime 超过 gate：

```text
1. 先检查是否写出 timing_summary.json；
2. 检查 probe_ttt_write_seconds_mean；
3. 若 probe TTT > 8s：启用 selected-layer / sampled-token / cached risk；
4. 若 chunk mean > 42s：关闭非必要 debug / 降低 tensor trace；
5. 若 wall > 28min：该候选不得 full promotion。
```

---

## 12. 本轮最终交付物

Codex 必须交付：

```text
v55_phase0_salvage_report.md
v55_phase1_c9_h35_m1_autopsy_report.md
v55_failure_type_summary.json
v55_candidate_design_decision.md
v55_smoke_registry.csv
v55_704f_registry.csv
v55_full_registry.csv
v55_runtime_audit.csv
v55_no_chunk_manual_percentage_audit.csv
v55_role_mass_timeline.csv
v55_gamma_timeline.csv
v55_commit_alpha_timeline.csv
v55_post_zp_delta_timeline.csv
v55_layer_branch_heatmap.png
v55_segment_error_timeline.png
v55_final_report.md
```

Final report 必须回答：

```text
1. v54 M1 full 是否应该被 704F gate 误杀？
2. C9-H35 最大差距是什么？
3. 新候选修的是不是最大差距？
4. 新候选是否满足 no chunk / no manual percentage / runtime gate？
5. 新候选是否接近 C9？
6. 如果不接近，下一步是修公式还是换 action space？
```

---

## 13. 一句话总结

v55 不是再试一个自适应 writer 名字，而是要把 C9 和 clean adaptive 之间的差距拆到 **chunk / segment / layer / branch / post-zp delta / commit behavior** 的层级。只有先知道最大差距在哪，才能设计真正有效的 no-chunk adaptive TTT。否则继续扫 threshold、rho、gamma 只是浪费时间。

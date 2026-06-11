# ACL2 v61：从 Clean H35 出发验证语义改善几何重建的实验计划

日期：2026-06-10
主基线：`v53 H35 full`，KITTI01 ATE = `35.7408969581m`
辅助参考：`C9_P0_R2 full`，KITTI01 ATE = `33.7629421029m`
计划定位：本轮不再把 C9 作为主要起点，因为 C9 含有 absolute chunk-id policy。本轮从 clean H35 出发，验证一个核心命题：

> **语义信息是否能通过 semantic residual READ cue 和 semantic TTT writing，真实改善 long-context geometric reconstruction？**

本轮特别强调：除了 full ATE，还必须记录 chunk-level scale consistency，因为前期 debug 已经提示 chunk 间尺度状态不一致可能是轨迹漂移的重要来源。语义不直接预测尺度，但语义可能通过 READ 和 TTT 写入影响哪些观测支配当前 chunk 的尺度估计，以及哪些信息被写入未来 chunk 的长期记忆。

---

## 0. 总体目标

本项目总目标仍然是：

> **Semantic-aware Memory Management for Long-context Geometric Reconstruction**

也就是让长视频几何模型在跨 chunk 推理时，利用语义信息辅助判断：

```text
what to read
what to write
what to preserve
what not to propagate
```

本轮 v61 的目标更具体：

1. 从 H35 clean baseline 出发，不使用 absolute chunk-id policy，不使用固定语义 action chunk，不使用 GT runtime semantic。
2. 继续探索 **semantic residual READ cue**，验证语义是否能重新解释内部几何不稳定 cue $D_i$，改善当前 chunk 对历史/source memory 的读取。
3. 继续探索 **semantic TTT writing**，验证语义是否能调制长期 fast-weight memory 写入，影响未来 chunk 的几何和尺度状态。
4. 在所有语义 READ / TTT 候选中强制记录 **chunk-level scale metrics**，验证语义是否不仅影响 ATE，也影响跨 chunk 尺度一致性。
5. 用 random same-mass、shuffled semantic、geometry-only 等负控制判断收益是否真的来自语义，而不是来自简单改变 attention mass 或 write mass。

本轮成功不要求一次到达 ATE 30，但必须能明确回答：

```text
语义是否在 clean H35 上有真实收益？
收益来自 READ，还是 TTT writing？
语义是否影响 chunk-level scale state？
如果语义失败，失败是 action 没生效、语义前端不稳定，还是当前 memory action 设计不对？
```

---

## 1. 本轮研究假设

### 1.1 假设 H1：semantic residual READ cue 能改善当前 chunk 的 source memory 读取

已有历史实验显示，semantic-conditioned C23 / semantic residual READ 是当前语义路线中最有正信号的一类。它的核心不是直接跳过某些语义类别，而是利用语义重新解释几何不稳定 cue：

```text
D_i 高并不总是同一种含义。
在动态物体上，D_i 高可能是 transient interference。
在 building / road / wall 上，D_i 高可能是 stable structure mismatch，需要 refinement。
在 sky / horizon 上，D_i 高可能是 context-only signal，不适合作 3D anchor，但也不能简单删除。
```

因此 Track A 需要验证：

> **语义作为 residual READ cue 是否能比 geometry-only cue 更好地控制当前 chunk 从历史/source tokens 中读取什么。**

### 1.2 假设 H2：semantic TTT writing 能影响未来 chunk 的长期记忆

READ 主要影响当前 chunk，但 TTT fast weights 会被未来 chunk 继承。因此语义如果要真正参与 memory management，必须继续探索它如何影响 TTT writing。

过去 semantic TTT 的失败不能简单解释成“语义不能写 TTT”。更可能是之前 action 太粗：

```text
semantic label -> positive / negative / no-long
```

这种规则无法区分 useful continuity 和 harmful update。v61 的 TTT 写入不做类别动作表，而是用语义 residual 地调制已有 clean TTT writer 的 write eligibility 和 write risk：

```text
geometry remains primary；
semantic only modifies value / risk；
mask quality controls trust。
```

### 1.3 假设 H3：语义可能通过 READ / TTT 影响 chunk-level scale consistency

NOGTPOSE 相关 debug 提示：KITTI01 的大误差里有明显 reset-window / chunk-level scale-state 成分。语义不能直接预测尺度，但可以帮助判断哪些观测适合作为尺度和几何 anchor。

因此 v61 不把 scale-anchor 作为完全独立支线，而是把 scale metrics 嵌入 READ / TTT 的评估：

```text
semantic residual READ:
    可能改变当前 chunk 的局部尺度估计。

semantic TTT writing:
    可能改变未来 chunk 继承的尺度状态。

semantic-geometric anchors:
    可能帮助选择可靠的 scale evidence。
```

本轮需要检验：

> **语义 READ / TTT 是否能降低 per-chunk scale residual、scale jump 或 step-length inconsistency。**

---

## 2. 基础记号与方法框架

设当前 chunk 为 $\mathcal{W}_m$，其中 token $i$ 有以下信号：

```text
D_i:
    geometry instability cue，来自 C23 / internal attention cue。

s_i^{stage}:
    原始几何写入资格。

e_i^{ttt}:
    TTT residual / update conflict proxy。

Q_i^{sem}:
    semantic masklet quality / temporal consistency。

L_i:
    semantic label or semantic group。

U_i^{src}:
    source attention usage，表示该 token 是否真的被 READ path 使用。

C_i^{geom}:
    geometry confidence。
```

Clean H35 中基础写入资格是：

$$
P_i = s_i^{stage}\sqrt{1-D_i}
$$

基础 TTT 写入风险是：

$$
R_i = e_i^{ttt}D_i
$$

v61 在此基础上加入语义锚点和瞬时风险。

### 2.1 Semantic anchor score

稳定语义-几何 anchor 分数定义为：

$$
S_i^{anchor}
=
Q_i^{sem}
\cdot
S_i^{static}
\cdot
(1-D_i)
\cdot
C_i^{geom}
\cdot
U_i^{src}
$$

其中：

```text
S_i^{static}:
    是否属于 stable structure semantic group。
    例如 road / ground / building / wall / fence / pole / traffic sign 等。

Q_i^{sem}:
    语义 masklet 质量和时序稳定性。

1-D_i:
    几何稳定性。

C_i^{geom}:
    几何置信度。

U_i^{src}:
    该 token 是否真的被模型作为 source memory 使用。
```

注意：一个 token 是 road 或 building 并不自动成为 anchor。它必须同时几何稳定、mask 质量高，并且确实被模型读取或写入。

### 2.2 Semantic transient risk score

瞬时语义风险定义为：

$$
S_i^{trans}
=
Q_i^{sem}
\cdot
S_i^{dynamic}
\cdot
D_i
\cdot
R_i
$$

其中：

```text
S_i^{dynamic}:
    是否属于 dynamic / transient / low-trust semantic group。
    例如 moving car / person / cyclist / unstable vegetation / reflection / bad mask region。

D_i:
    几何不稳定。

R_i:
    TTT residual risk。
```

同样，语义标签不会单独触发 action。只有当语义、几何 cue、TTT residual 同时支持时，才认为该 token 是 transient risk。

---

## 3. Track A：Semantic Residual READ Cue

### 3.1 目标

Track A 验证：

> **语义是否能作为 residual cue 改善 H35 的 READ path，从而改善当前 chunk 几何和尺度状态。**

READ path 的作用是控制当前 chunk 从历史/source tokens 中读取哪些信息。H35 已经有 geometry cue $D_i$，但 $D_i$ 的含义在不同语义区域中不同。因此 v61 用语义对 $D_i$ 做 residual reconditioning，而不是做 hard skip。

### 3.2 方法

基础 READ bias 是：

$$
b_i^{geo} = -\lambda_D D_i
$$

语义重解释后的不稳定 cue 是：

$$
D_i^{sem}
=
\operatorname{clip}
\left(
D_i
-
\alpha_A S_i^{anchor}
+
\alpha_T S_i^{trans},
0,
1
\right)
$$

最终 READ bias：

$$
b_i^{sem-read}
=
-\lambda_D D_i^{sem}
$$

等价地可以写成 residual bias：

$$
b_i^{sem-read}
=
-\lambda_D D_i
+
\lambda_A S_i^{anchor}
-
\lambda_T S_i^{trans}
$$

解释：

```text
stable semantic-geometric anchor:
    降低其 read risk，避免被 D_i 误伤。

transient semantic risk:
    提高其 read risk，但只做 soft reconditioning，不 hard delete。

context-only semantics:
    例如 sky / horizon，默认不作为 persistent anchor，但也不直接 hard suppress。
```

### 3.3 Track A 候选

```text
A0_H35_REPEAT
    H35 full or 704F repeat，确认 baseline 没漂。

A1_SEM_RESID_C23_READ
    复现/延续 semantic residual C23 READ。
    只改 READ，不改 TTT writing。

A2_SEM_CONDITIONED_DG_READ
    复刻 v31 semantic-conditioned D_g 思路：
    在同类语义内部重标定 D_i。
    只改 READ。

A3_SEM_ANCHOR_RESCUE_READ
    只降低 stable semantic-geometric anchor 的 read risk。
    不提高 transient risk。
    目标是验证“增强该记住的”是否优于“削弱可疑的”。

A4_SEM_TRANSIENT_RISK_READ
    只对 high D_i + high R_i + transient semantic 区域做 soft risk boost。
    不 hard skip。

A5_SEM_RESID_READ_C9_COMPAT
    在 C9 上复现 semantic residual READ 正信号。
    用作兼容性和上界参考，不作为 clean 成功。
```

### 3.4 Track A 负控制

```text
NA1_RANDOM_SAME_MASS_READ
    随机选择与 A1/A3 相同 source mass 的 token 做 residual READ。

NA2_SHUFFLED_SEMANTIC_READ
    打乱 semantic label 后构造 semantic residual READ。

NA3_GEOMETRY_ONLY_RESIDUAL_READ
    不使用语义，只使用 D_i / residual score 做同形 READ reconditioning。

NA4_SEMANTIC_ONLY_READ
    不使用 D_i，只使用 semantic label / group 做 READ。
    如果它效果差，说明语义必须和几何 cue 结合。
```

### 3.5 Track A 必须记录的指标

每个候选必须记录：

```text
ATE_704 / ATE_full
Rot
FinalErr
RPE_t / RPE_r
rolling50 mean / p90 / worst
rolling100 mean / p90 / worst
rolling200 mean / p90 / worst
[200,300) delta
[400,600) delta
seg0 / seg1 / seg2

source_attention_mass_before
source_attention_mass_after
semantic_anchor_source_mass
semantic_transient_source_mass
anchor_rescue_count
transient_boost_count
context_empty_source_events
READ layer action count

per-chunk overlap_sim3_scale
per-chunk log_scale_residual
per-chunk step_length_ratio
per-chunk scale_jump_vs_prev
semantic-anchor-weighted scale residual
geometry-only scale residual

runtime wall_min
chunk_mean_seconds
```

### 3.6 Track A 判断标准

704F promotion gate：

```text
delta_vs_H35_704 <= -0.50m
OR rolling100_p90_delta <= -2.0m
OR [200,300) delta <= -3.0m with [400,600) regression <= +1.0m
OR scale_residual_variance improves >= 10% with no ATE regression > +0.3m
```

同时要求：

```text
beats random/shuffled controls by >= 0.3m on ATE_704
OR beats controls by >= 1.0m on [200,300)
OR beats controls by >= 10% on scale residual variance

context_empty_source_events = 0
runtime_projected_full <= 28min
```

Full success gate：

```text
minimum semantic progress:
    ATE <= 35.2409m
    OR improvement vs H35 >= 0.5m

semantic target:
    ATE <= 33.7409m
    OR improvement vs H35 >= 2.0m
```

Mechanistic pass：

```text
If ATE improvement < 0.5m but scale_residual_variance improves >= 15%
and rolling100_p90 does not regress,
then Track A is marked as scale-mechanistic positive, not final method success.
```

---

## 4. Track B：Semantic TTT Writing

### 4.1 目标

Track B 验证：

> **语义是否能调制长期 TTT writing，并影响未来 chunk 的几何和尺度状态。**

这是 semantic-aware memory management 的关键，因为 READ 只影响当前 chunk，而 TTT fast weights 会影响未来 chunk。

### 4.2 方法

基础 clean TTT 写入资格：

$$
P_i = s_i^{stage}\sqrt{1-D_i}
$$

基础风险：

$$
R_i = e_i^{ttt}D_i
$$

语义调制写入资格：

$$
P_i^{sem}
=
P_i
\left(1+\eta_A S_i^{anchor}\right)
\left(1-\eta_T S_i^{trans}\right)
$$

语义调制写入风险：

$$
R_i^{sem}
=
R_i
+
\eta_R S_i^{trans}
-
\eta_S S_i^{anchor}
$$

然后进入 clean adaptive split replay：

```text
positive:
    high P_i^sem, low R_i^sem

neutral:
    medium confidence / uncertain evidence

negative:
    high R_i^sem, low P_i^sem
```

最终更新：

$$
\Delta W^{sem}
=
\Delta W_{pos}^{sem}
+
\lambda_{neu}\Delta W_{neu}^{sem}
-
\gamma_{neg}\Delta W_{neg}^{sem}
$$

### 4.3 Track B 关键原则

本轮 semantic TTT writing 禁止：

```text
semantic class alone -> positive / negative / no-long
broad no-long-write
hard freeze
fixed semantic action chunks
absolute chunk-id policy
manual positive/negative percentage
learned trigger
GT semantic runtime action
```

必须满足：

```text
geometry eligibility remains primary；
semantic only acts as residual value/risk modulation；
semantic action must be measurable in write mass and post-zp delta。
```

### 4.4 Track B 候选

```text
B0_H35_TTT_REPEAT
    H35 TTT baseline repeat。

B1_SEM_ANCHOR_WRITE_FLOOR
    stable semantic-geometric anchor 的 write eligibility 下限。
    不改 transient risk。
    目标：验证语义能否帮助 TTT 记住稳定结构。

B2_SEM_TRANSIENT_RISK_BOOST
    对 high D_i + high R_i + transient semantic token 提高 write risk。
    不 hard no-long。
    目标：验证语义能否降低 risky memory update。

B3_SEM_ANCHOR_PLUS_TRANSIENT_TTT
    B1 + B2。
    只有 B1 或 B2 有 704F 正信号后才跑。

B4_SEM_CONDITIONED_DG_TTT
    把 semantic-conditioned D_g 用于 TTT risk。
    继承 v31 思路，但从 READ 迁移到 TTT writing。

B5_SEM_RESIDUAL_TTL_REPAIR
    重新测试 short residual / TTL，但先保证 short_residual_norm 有效。
    如果 norm 低于阈值，不进入 704F。

B6_SEM_TTT_C9_COMPAT
    在 C9 parent / causal fork 上测试 semantic TTT writing compatibility。
    仅作为机制参考，不作为 clean success。
```

### 4.5 Track B 负控制

```text
NB1_RANDOM_SAME_MASS_TTT
    随机同等 write mass 的 TTT modulation。

NB2_SHUFFLED_SEMANTIC_TTT
    打乱 semantic label 后做同样 TTT modulation。

NB3_GEOMETRY_ONLY_TTT
    只用 D_i + R_i 做同形 TTT modulation。

NB4_SEMANTIC_ONLY_TTT
    只用 semantic group 做 TTT modulation。
    预期应弱于 semantic + geometry。
```

### 4.6 Track B 必须记录的指标

TTT action：

```text
positive_mass
neutral_mass
negative_mass
semantic_anchor_positive_mass
semantic_transient_negative_mass
semantic_write_floor_mass
semantic_risk_boost_mass
role_collapse_count
tri_replay_applied_count
semantic_ttt_action_count
```

TTT update：

```text
post_zp_delta_norm
post_zp_delta_norm_delta_vs_H35
branch_w0_delta_norm
branch_w1_delta_norm
branch_w2_delta_norm
layer_delta_norm_topk
candidate_native_cosine
update_conflict_energy if available
semantic_anchor_post_zp_contribution
semantic_transient_post_zp_contribution
```

未来尺度影响：

```text
future_chunk_log_scale_residual
future_scale_residual_variance
future_step_length_ratio
future_overlap_sim3_scale
future_rolling100_delta
future_[200,300)_delta
future_[400,600)_delta
```

Trajectory：

```text
ATE
Rot
FinalErr
rolling100 p90/worst
seg0 / seg1 / seg2
```

Runtime：

```text
wall_min
chunk_mean_seconds
probe_ttt_write_seconds_mean
```

### 4.7 Track B 判断标准

Smoke gate：

```text
semantic_ttt_action_count > 0
semantic_anchor_write_mass > 0 OR semantic_transient_risk_mass > 0
role_collapse_count = 0
post_zp_delta_norm_delta_vs_H35 is measurable
runtime_projected_full <= 28min
```

TTL 候选额外 gate：

```text
short_residual_norm > 1e-4
```

704F gate：

```text
delta_vs_H35_704 <= -0.50m
OR [200,300) delta <= -3.0m with [400,600) regression <= +1.0m
OR rolling100_p90_delta <= -2.0m
OR future_scale_residual_variance improves >= 10%
```

Full gate：

```text
minimum progress:
    ATE <= 35.2409m
    OR improvement vs H35 >= 0.5m

semantic TTT success:
    semantic TTT beats geometry-only TTT by >= 0.3m
    AND beats random/shuffled controls by >= 0.3m
```

Causal fork mechanistic pass：

```text
If changing only chunk k TTT write causes future chunks k+1...k+h
to reduce scale residual by >= 15% without increasing rolling100_p90 by > +0.5m,
then semantic TTT writing is considered future-memory-positive even if full ATE gain is small.
```

---

## 5. Track C：Scale-aware Analysis Embedded in A/B

Track C 不再是独立主线，而是嵌入 Track A/B 的分析模块。它回答：

> **semantic READ / semantic TTT 是否改变 chunk-level scale state？**

### 5.1 Scale residual 估计

对相邻 chunk overlap 中的点对 $\{x_i, y_i\}$，估计 Sim(3)：

$$
\hat{s}, \hat{R}, \hat{t}
=
\arg\min_{s,R,t}
\sum_i
w_i
\left\|
sRx_i + t - y_i
\right\|_2^2
$$

需要记录三种权重版本：

```text
all-overlap:
    w_i = 1

geometry-weighted:
    w_i = C_i^{geom}(1-D_i)

semantic-anchor-weighted:
    w_i = C_i^{geom}(1-D_i)Q_i^{sem}S_i^{static}(1-S_i^{trans})
```

scale residual：

$$
z_m = \log \hat{s}_m
$$

scale residual variance：

$$
V_{scale}
=
\operatorname{Var}_m(z_m)
$$

step-length ratio：

$$
\rho_m
=
\frac{
\operatorname{median}_{k \in \mathcal{W}_m}\|t_{k+1}-t_k\|
}{
\operatorname{median}_{k \in \text{global}}\|t_{k+1}-t_k\| + \epsilon
}
$$

### 5.2 Scale 相关指标

每条候选必须输出：

```text
per_chunk_scale_metrics.csv
    chunk_id
    frame_start
    frame_end
    overlap_sim3_scale_all
    overlap_sim3_scale_geo
    overlap_sim3_scale_sem
    log_scale_residual_all
    log_scale_residual_geo
    log_scale_residual_sem
    overlap_sim3_residual_all
    overlap_sim3_residual_geo
    overlap_sim3_residual_sem
    inlier_ratio_all
    inlier_ratio_geo
    inlier_ratio_sem
    step_length_median
    step_length_ratio
    pointmap_depth_scale_ratio
    semantic_anchor_ratio
    semantic_anchor_spatial_entropy
```

还需要输出：

```text
scale_residual_summary.csv
    variance_all
    variance_geo
    variance_sem
    mean_abs_log_scale_all
    mean_abs_log_scale_geo
    mean_abs_log_scale_sem
    corr_log_scale_with_rolling100
    corr_anchor_quality_with_scale_stability
```

### 5.3 Scale 判断标准

Scale mechanistic pass：

```text
semantic candidate reduces mean_abs_log_scale by >= 10%
OR reduces scale_residual_variance by >= 10%
OR improves corr(anchor_quality, scale_stability) by >= 0.1
```

但注意：

```text
Scale mechanistic pass 不等于 method success。
最终成功仍以 full ATE / local window / rolling metrics 为主。
```

---

## 6. 统一实验阶段

### Phase 0：代码、baseline、artifact 审计

Phase 0 目标是确保所有必要字段可用，不允许一边跑实验一边猜 action 是否生效。

必须审计：

```text
H35 landed artifact / repeat availability
C9 landed artifact availability
Stage C semantic cache hit rate
semantic token projection nonempty
D_i / C23 cue available
TTT residual risk available
source attention mass available
post-zp delta trace available
overlap Sim(3) point pairs available
per-chunk scale metric script available
```

输出：

```text
phase0_audit/
    h35_repeat_or_landed_audit.md
    semantic_cache_hit_audit.csv
    semantic_projection_audit.csv
    read_wiring_audit.md
    ttt_wiring_audit.md
    scale_metric_availability.md
    codex_self_check_report.md
```

Gate：

```text
H35 baseline available
semantic cache hit >= 0.95
semantic projection nonempty
D_i available
TTT residual risk available for Track B
scale metric script can run on H35
```

如果 Track B 的 TTT residual risk 缺失：

```text
Codex 必须先修 TTT residual logging；
Track A 可以继续；
Track B 不允许进入 704F。
```

---

### Phase 1：96F / 256F smoke

目的：验证 action 真实生效、速度可接受、scale metrics 可落盘。

并行运行：

```text
Track A smoke:
    A1 / A2 / A3 / A4
    NA1 / NA2 / NA3

Track B smoke:
    B1 / B2 / B4
    NB1 / NB2 / NB3

Scale smoke:
    对 H35 / A1 / B1 输出 per-chunk scale metrics。
```

Smoke gate：

```text
semantic action count > 0
source attention or write mass changed
context_empty_source_events = 0
role_collapse_count = 0 for TTT
scale metrics not empty
projected full runtime <= 28min
```

不满足时：

```text
action inactive:
    修 CLI -> HMC -> Pi3 / TTT controller 接线。

mass unchanged:
    检查是否被 normalization / override 擦掉。

scale metrics missing:
    先实现 per-chunk overlap Sim(3) / step-length metrics。
```

---

### Phase 2：704F screen

最多运行：

```text
Track A:
    3 semantic READ + 2 controls

Track B:
    3 semantic TTT + 2 controls

Optional:
    A+B combo 只在单轨有正信号后运行
```

704F 必须记录：

```text
trajectory metrics
READ action metrics
TTT action metrics
scale metrics
runtime metrics
controls comparison
```

704F promotion gate：

```text
delta_vs_H35_704 <= -0.50m
OR local window delta <= -3.0m with downstream regression <= +1.0m
OR rolling100_p90_delta <= -2.0m
OR scale_residual_variance improves >= 10% and ATE regression <= +0.3m
```

如果没有候选过 gate，但最佳候选满足：

```text
ATE regression <= +0.20m
AND scale_residual_variance improves >= 15%
AND beats controls on scale metrics
```

允许 1 条 full diagnostic，不能写成 method success。

---

### Phase 3：Causal fork for semantic TTT

Track B 必须做 causal fork，因为 TTT writing 主要影响未来 chunk，而不是当前输出。

选择 chunks：

```text
不要用固定 chunk id 作为 policy。
选择 H35 中 scale_residual 或 rolling100 error 最高的 top-K chunks 作为 diagnostic forks。
必须记录 selection_uses_ATE = false for runtime-style selection。
```

每个 fork 比较：

```text
Base:
    H35 TTT write

Semantic TTT:
    只改变 chunk k 的 TTT write

Future rollout:
    k+1 ... k+h
```

记录：

```text
current chunk delta
future chunk ATE delta
future scale residual delta
future rolling100 delta
future [200,300) / [400,600) delta
post-zp delta difference
```

判断：

```text
如果 current chunk 几乎不变，但 future scale residual 改善，
说明 semantic TTT writing 真实影响未来 memory。

如果 current/future 都不变，
说明 semantic write action 被擦掉或太弱。

如果 scale improves but ATE worse，
说明 scale correction 与 other geometry state 冲突，需要 merge/gauge controller。
```

---

### Phase 4：Full KITTI01

最多运行：

```text
F1:
    best Track A semantic READ

F2:
    best Track B semantic TTT

F3:
    best A+B combo only if A/B single tracks pass 704F

F4:
    best control if needed for causal claim
```

Full gate：

```text
frames = 1101
hmc_rows = 38
runtime <= 28min
stage_c_cache_hit >= 0.95 for semantic runs
```

Full success：

```text
minimum progress:
    ATE <= 35.2409m
    OR improvement vs H35 >= 0.5m

strong semantic progress:
    ATE <= 34.7409m
    OR improvement vs H35 >= 1.0m

semantic target:
    ATE <= 33.7409m
    OR improvement vs H35 >= 2.0m
```

Mechanistic success：

```text
Even if ATE gain < 0.5m,
if scale residual improves >= 15%
and rolling100_p90 does not regress,
candidate can be reported as scale-mechanistic positive, not full method success.
```

---

## 7. Ablation 设计

最终报告需要至少包含以下 ablation：

| Variant | READ cue | TTT writing | Semantic | Scale metrics | 目的 |
|---|---|---|---|---|---|
| H35 | geometry | clean adaptive | no | yes | clean baseline |
| READ-Geo | geometry residual | clean adaptive | no | yes | 几何 READ 对照 |
| READ-Sem | semantic residual READ | clean adaptive | yes | yes | 测 semantic READ |
| TTT-Geo | geometry READ | geometry-only TTT residual | no | yes | 几何 TTT 对照 |
| TTT-Sem | geometry READ | semantic TTT | yes | yes | 测 semantic TTT |
| READ+TTT-Sem | semantic READ | semantic TTT | yes | yes | 组合 |
| Random control | same mass | same mass | random | yes | 排除 mass effect |
| Shuffled control | shuffled label | shuffled label | corrupted | yes | 排除语义标签无关 |
| Semantic-only | semantic only | semantic only | yes | yes | 证明语义不能脱离几何 cue |

---

## 8. 可视化要求

每个进入 full 的候选必须生成：

```text
figures/
    semantic_read_residual_map_chunk10.png
    semantic_anchor_overlay_top_scale_chunk.png
    semantic_transient_overlay_top_scale_chunk.png
    source_attention_before_after_timeline.png
    ttt_role_mass_timeline.png
    semantic_anchor_write_mass_timeline.png
    post_zp_delta_norm_by_layer_branch.png
    log_scale_residual_timeline.png
    step_length_ratio_timeline.png
    scale_residual_vs_rolling100_error.png
    scale_metric_control_comparison_bar.png
    rolling100_delta_timeline.png
    segment_delta_bar.png
```

如果空间图缺数据：

```text
必须标注 no-data / unavailable。
不允许把缺失字段补 0。
```

---

## 9. Codex 失败分流规则

### 9.1 semantic READ 没有 action

表现：

```text
semantic_read_action_count = 0
source_attention_mass_before = 0
```

Codex 必须修：

```text
semantic projection
READ control parameter forwarding
source token flatten alignment
attention mass logging
Stage C cache short-tail hit
```

修好前不允许进入 704F。

---

### 9.2 semantic READ action 生效但 ATE 和 scale 都回退

表现：

```text
ATE delta > +0.5m
scale residual variance worse
```

Codex 必须尝试：

```text
降低 action strength；
改成 anchor rescue only；
关闭 transient risk boost；
保留 context-only sky/horizon；
检查 semantic mask flicker。
```

禁止继续 hard skip。

---

### 9.3 semantic TTT action 没有改变 post-zp delta

表现：

```text
semantic_write_mass > 0
但 post_zp_delta_norm_delta_vs_H35 = 0
```

Codex 必须检查：

```text
stage_d_x_dg normalization 是否擦掉 semantic floor；
TTT controller role mask 是否接收 P_i^sem / R_i^sem；
adaptive split replay 是否忽略 semantic fields；
post-zp logging 是否正确。
```

修好前不进入 704F/full。

---

### 9.4 semantic TTT 改变 post-zp delta 但未来 scale 不变

表现：

```text
post_zp_delta changed
future scale residual unchanged
```

Codex 必须输出：

```text
semantic update layer/branch 是否不是 scale-carrying branch；
anchor/transient tokens 是否和 scale residual chunks 不重叠；
candidate-native cosine 是否过低；
是否需要 branch/layer-specific semantic modulation。
```

---

### 9.5 semantic improves scale but hurts ATE

表现：

```text
scale residual improves
ATE / rolling100 worsens
```

Codex 不允许简单加大 scale action。必须分析：

```text
Rot 是否变差；
FinalErr 是否变差；
pointmap depth 是否扭曲；
merge/gauge state 是否和 TTT state 冲突；
是否只适合作 output-level scale controller 而非 TTT writing。
```

---

### 9.6 semantic 与 random/shuffled 一样好

表现：

```text
semantic candidate 和 random / shuffled control 差距 < 0.2m
且 scale metric 也接近
```

报告必须写：

```text
当前收益不能归因于 semantic meaning。
可能只是 attention/write mass perturbation。
```

后续转向 geometry-only baseline 或更强 semantic frontend。

---

## 10. 最终报告必须回答的问题

最终报告必须清楚回答：

```text
1. 从 clean H35 出发，semantic residual READ 是否有收益？
2. semantic residual READ 是否影响 current chunk scale metrics？
3. semantic TTT writing 是否真实改变 TTT write mass 和 post-zp update？
4. semantic TTT writing 是否影响 future chunks 的 scale residual？
5. semantic 是否优于 random / shuffled / geometry-only controls？
6. 语义收益主要来自 READ、TTT，还是 scale-state effect？
7. 如果失败，主要瓶颈是 semantic frontend、runtime action、TTT normalization，还是 scale-state controller 缺失？
```

---

## 11. 结论边界

如果 Track A 成功，可以说：

```text
Semantic residual READ cue helps reconstruction by reconditioning geometry-memory read cues.
```

如果 Track B 成功，可以说：

```text
Semantic cues can modulate long-term TTT writing beyond geometry-only risk.
```

如果 scale metrics 成功但 ATE 不成功，可以说：

```text
Semantic cues affect chunk-level scale state, but the current controller does not yet convert this into stable full-trajectory improvement.
```

如果全部失败，必须说：

```text
Starting from H35 clean baseline, current semantic READ/TTT mechanisms do not yet provide stable causal improvement.
The next bottleneck is either semantic frontend consistency or explicit scale-state / merge-gauge control.
```

---

## 12. 一句话总结

v61 的核心是：

> **从 clean H35 出发，不只看语义能不能降 ATE，还要验证 semantic residual READ 和 semantic TTT writing 是否能影响 chunk-level scale state；如果语义能改善 scale residual 或 future memory behavior，即使 full ATE 暂时不大，也能为 semantic-aware memory management 提供机制证据。**

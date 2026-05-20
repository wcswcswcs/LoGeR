# ACL2 v19：TTT 写入策略重启实验计划  
## From local fast-weight regularization to trajectory-state-aware TTT write

日期：2026-05-20  
目标数据集：KITTI Odometry Sequence 01  
当前 online TTT/HMC deployable best：`C9_P0_R2 = 33.7629421029m`  
当前健康 reference：`H9_P0_R2 = 34.1257769401m`，其中 `[200,300)` 比 C9 更好  
新目标：`KITTI01 ATE <= 25.0m`  
本计划性质：**下一阶段执行计划，不是结果报告**

---

## 0. 当前状态的独立判断

过去几轮已经证明，继续在普通 TTT 写入超参数上做小修小补不会接近 `25m`。当前最好的可计数 online TTT 写入配置仍是 `C9_P0_R2`，ATE 为 `33.7629421029m`。但它并不是健康的 target-25 候选，因为它的 `[200,300)` 为 `76.102136m`，比 `H9_P0_R2` 的 `74.409927m` 更差。也就是说，C9 更像是把后段误差与全局形状重新分配了一下，而不是修掉核心 drift 病灶。

v18 做了一件很重要的事情：它把真正的 action surface 做成了可审计对象，包括：

```text
post-zeropower delta / basis
overlap-geometry replay target
W_long / W_short dual-lifetime summary
candidate commit manifest
window scale proxy
```

但是，三类 true-action short rollout 都没有通过 h10/h15 sandbox gate：

```text
PZBASIS best h10 ATE delta = -0.391m
AUXGEO best h10 ATE delta = -0.650m
DLTRUE best h10 ATE delta = -0.280m

PZBASIS best h10 [200,300) delta = -0.457m
AUXGEO best h10 [200,300) delta = -0.892m
DLTRUE best h10 [200,300) delta = -0.631m
```

这说明当前这三类 action surface 的真实效果仍然弱且局部。它们不是 target-25 主路线。

但是，这并不等于“TTT 没价值”。LoGeR 的设计本来就是：TTT 负责 compressed global context / global consistency，SWA 负责 local smoothness / adjacent alignment。因此，如果 KITTI01 的主误差是 reset-window scale / trajectory-state drift，那么 TTT 仍然是合理的目标模块。问题在于：我们当前给 TTT 的动作还停留在 patch-token / branch / delta 的局部修改，而目标误差是 window-level 的 trajectory state。

所以 v19 的核心判断是：

> **我们不再继续寻找“哪些 patch 多写/少写”的小策略，而是验证 TTT 是否能接收、承载并传播一个 explicit trajectory-state / scale-state correction。**

换句话说，下一阶段不是继续问：

```text
这个 token prior 是否更好？
这个 gamma 是否更好？
这个 branch mask 是否更好？
这个 post-zp basis 是否更好？
```

而是问：

```text
TTT fast weights 是否能够通过写入一个 no-GT trajectory-state signal，改变未来窗口的 scale / drift？
```

如果不能，TTT write-only 就不应该再作为 target-25 主线。

---

## 1. 实验整体目标

v19 的整体目标是建立一个清晰的 yes/no 结论：

> **TTT fast-weight write 是否存在可部署的 trajectory-state-aware action space，能够把 KITTI01 online ATE 从 `33.76m` 级别推进到 `25m` 方向？**

为了回答这个问题，本计划分为四条并行但有依赖关系的主线：

```text
Track A: Drift-State Autopsy
    先确认当前 TTT action 到底是在修局部、修旋转，还是能动到 scale/drift。

Track B: Trajectory-State TTT Replay
    不再只写 patch feature，而是把 no-GT window scale / trajectory proxy 写入 TTT。

Track C: TTT MPC Candidate Commit Bank
    每个关键 chunk 生成多个 candidate fast-weight commit，用短窗口 proxy 选择 commit。

Track D: Causal Basis Learning
    从历史 run bank 学习哪些 post-zp delta direction 与 scale / drift 改善相关，再生成候选 commit。
```

这四条线的关系是：

```text
Track A 是必须做的诊断；
Track B / C / D 可以并行启动；
只有 B/C/D 中至少一条通过 sandbox gate，才允许 full KITTI01 online validation；
如果三条都失败，停止把 TTT write-only 作为 Target-25 主线。
```

---

## 2. 不再继续的方向

以下方向不再作为主线：

```text
gamma 微扫
neutral lambda 微扫
read beta 微扫
chunk16 weak regularizer 细扫
commit EMA alpha 细扫
post-zp scalar mix/cap 细扫
普通 negative-tail / low-prior TTGR 细扫
semantic scalar value 细扫
```

这些方向已经在 v7-v18 中充分暴露出平台化特征。它们可以改善 Rot、FinalErr 或某个短窗口，但不能稳定修 whole-scene drift。

停止理由是：

```text
1. 当前 online best 与 target-25 差距约 8.76m。
2. 这些方向的典型增益是 0.01m-0.6m。
3. h10/h15 持久性 gate 多数不过。
4. 即使局部 [200,300) 改善，也常伴随 [400,600) regression。
```

---

## 3. 核心假设

---

## H1：当前 TTT write 的失败不是 cue 缺失，而是 action 与目标误差错位

### 假设内容

当前 `update_conflict_energy`、PZBASIS、AUXGEO、DLTRUE 等 cue / action surface 有真实信号，但它们主要作用在 fast-weight local residual 上，没有显式建模 window-level scale / trajectory state。因此它们只能做 regularization，而不能推动 `ATE <= 25m`。

### 实验设计

不新增 full run。对已有 v8-v18 的所有可信结果做统一 autopsy：

```text
H9
C9
WINGAM
C16ROLE
PZBASIS best rows
AUXGEO best rows
DLTRUE best rows
BASIS proxy best rows
TTGRL_04
freeze5 / freeze56 diagnostic
NOGTPOSE_27 diagnostic
```

生成统一表：

```text
run_id
counts_as_ttt_write
ATE
Rot
FinalErr
[200,300)
[400,600)
Sim3Scale
YawRMSE
RPE_t
RPE_r
update_conflict_energy_mean
post_zp_delta_norm
PZ_basis_coeff
AUXGEO_overlap_score
DLTRUE_short_mass
body_window_score
exit_window_score
```

### 必须记录的指标

#### 全局指标

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
```

#### segment 指标

```text
ATE_50_mean
ATE_100_mean
ATE_200_mean
[0,100)
[100,200)
[200,300)
[200,400)
[300,400)
[400,500)
[400,600)
[600,800)
```

#### TTT action 指标

```text
per_chunk_update_conflict_energy
per_layer_branch_post_zp_delta_norm
per_layer_branch_post_zp_delta_cos_to_H9
per_layer_branch_post_zp_delta_cos_to_C9
PZ_basis_coefficients
W_short_apply_mass
AUXGEO_overlap_target_mass
```

### 可视化

必须生成：

```text
global_drift_dashboard/
    trajectory_xy_H9_C9_NOGTPOSE.png
    per_100f_ATE_curve.png
    per_reset_group_ATE_curve.png
    Sim3Scale_over_time.png
    Yaw_error_over_time.png
    [200,300)_vs_[400,600)_scatter.png

ttt_action_dashboard/
    chunk_x_layer_x_branch_delta_norm_heatmap.png
    update_conflict_energy_over_chunks.png
    post_zp_basis_coeff_over_chunks.png
    action_delta_vs_segment_delta_scatter.png
```

### 假设成立标准

H1 成立，如果满足：

```text
1. 当前 action signal 与 Rot / FinalErr 的相关性高于与 ATE / [200,300) 的相关性；
2. PZBASIS/AUXGEO/DLTRUE 的 h10/h15 ATE delta 均 < 1m；
3. [200,300) 改善与 [400,600) regression 呈明显 trade-off；
4. 没有一个已有 TTT action 能让 [200,300) 下降 >= 5m 且 [400,600) 回退 <= 1m。
```

如果 H1 不成立，即发现某个已有 action 与 `[200,300)` / ATE 有强相关但未被充分 sweep，则 Track B/C/D 的优先级下调，先扩展该 action。

---

## H2：TTT 需要显式写入 trajectory-state / scale-state，而不是只写 patch-feature residual

### 假设内容

LoGeR 的 TTT fast weights理论上用于维护 global consistency，但当前 TTT update target 仍主要是 hidden feature / KV target。若要修 target-25 级别的 drift，必须让 TTT 写入一个与 window scale / trajectory state 相关的 self-supervised signal。

### 核心思想

构造一个 no-GT scale-state signal：

$$
s_m = \log \frac{\bar{\ell}_{m}^{proxy}}{\operatorname{EMA}(\bar{\ell}^{proxy}) + \epsilon}
$$

其中 $\bar{\ell}_{m}^{proxy}$ 可以来自：

```text
overlap pointmap scale ratio
static-anchor depth/point norm ratio
predicted pose step length ratio
high-confidence road/building anchor displacement
window-local Sim3 scale proxy from prediction only
```

再把这个 $s_m$ 注入 TTT 写入，不通过后处理改轨迹，而是在 commit 时改变未来 fast weights：

$$
W_{m+1} = W_m + \Delta W_{feature} + \lambda_s \Delta W_{scale}
$$

其中 $\Delta W_{scale}$ 必须是 runtime no-GT 计算出来的 fast-weight update，不允许用 GT。

### 实验设计

#### B0：scale proxy offline audit

先不改变模型，只算每个 chunk 的 proxy：

```text
scale_proxy_overlap_pointmap
scale_proxy_static_anchor_norm
scale_proxy_pose_step_length
scale_proxy_confident_structure_depth
scale_proxy_median_fwd_motion
```

与真实 segment 误差比较：

```text
Spearman(proxy, [200,300) future error)
Spearman(proxy, per-reset scale drift)
Spearman(proxy, C9/H9 delta)
```

#### B1：scale-token replay

在 TTT replay 中选择一组 special / register / stable-anchor pooled tokens 作为 scale-state carrier：

```text
carrier tokens:
    option 1: register/special tokens
    option 2: lowest-D_g structure anchor patch tokens
    option 3: overlap static anchor pooled token
```

构造写入目标：

$$
v_i' = v_i + \alpha_s s_m \cdot d_i
$$

其中 $d_i$ 是从 stable token value direction 中构造的单位方向，不新增训练参数。

候选：

```text
SCALETTT_01:
    carrier = special tokens
    branch = w0
    alpha_s = 0.05

SCALETTT_02:
    carrier = structure low-D_g top 20%
    branch = w0
    alpha_s = 0.05

SCALETTT_03:
    carrier = overlap static anchor pooled
    branch = w0+w2
    alpha_s = 0.05

SCALETTT_04:
    same as SCALETTT_02
    alpha_s = 0.10

SCALETTT_05:
    same as SCALETTT_03
    alpha_s = 0.10
```

#### B2：scale-state commit modulation

不改 token target，而是用 scale proxy 控制 commit candidate：

$$
\Delta W_{commit} =
\Delta W_{native} +
g(s_m) \Delta W_{correction}
$$

其中：

$$
g(s_m) = \operatorname{clip}(\alpha |s_m|, 0, g_{max})
$$

候选：

```text
SCALECOMMIT_01:
    correction = PZBASIS harm-suppress W0
    gate = abs(scale_proxy)
    gmax = 0.25

SCALECOMMIT_02:
    correction = AUXGEO overlap-structure W0
    gate = abs(scale_proxy)
    gmax = 0.25

SCALECOMMIT_03:
    correction = C9-H9 historical delta basis
    gate = abs(scale_proxy)
    gmax = 0.25
```

### 必须记录的指标

```text
scale_proxy_value
scale_proxy_source
scale_proxy_reliability
carrier_token_count
carrier_token_semantic_group
carrier_token_Dg_mean
scale_delta_norm
scale_delta_cos_to_native
scale_delta_cos_to_C9_minus_H9
per_layer_branch_scale_delta_norm
per_layer_branch_scale_delta_cos
```

### 可视化

```text
scale_proxy_over_chunks.png
scale_proxy_vs_future_ATE.png
scale_proxy_vs_Sim3Scale.png
scale_delta_layer_branch_heatmap.png
carrier_token_maps_rgb_overlay/
```

### 假设成立标准

先在 sandbox h10/h15 通过以下任一条件：

```text
strong sandbox pass:
    h10/h15 ATE delta <= -3m

segment pass:
    [200,300) delta <= -5m
    and [400,600) regression <= +1m

scale-state pass:
    Sim3Scale proxy error improves >= 20%
    and ATE delta <= -1.5m
```

如果 B0 的 proxy 与 future error 相关性低于 `0.35`，Codex 不应启动 B1/B2 full matrix，而应先自动尝试新的 scale proxy：

```text
try:
    overlap static point distance ratio
    road/building point norm median
    per-reset pose-step median ratio
    horizon-line / sky-road boundary scale proxy if semantic cache exists
```

如果 B1/B2 全部 h10/h15 ATE delta > `-1m`，停止 scale-token replay，转 Track C / D。

---

## H3：TTT candidate commit 需要 MPC-style selection，而不是单一 commit

### 假设内容

一个 chunk 只能生成一个 `W_{m+1}` 太弱。正确做法应该是：为关键 chunk 生成多个 TTT commit candidate，并用短窗口 no-GT consistency proxy 选择未来更稳定的 commit。

### 实验设计

基于 v16 可信 causal fork，重新做 candidate bank，但 candidate 不再是 v18 的弱 family，而是来自 Track B / D 的强 family。

每个关键 chunk 生成：

```text
K1_H9 native parent
K2_C9-style weak freeze / mp_alpha variant
K3_scale_token_replay
K4_scale_commit_modulation
K5_historical_C9_minus_H9_delta
K6_overlap_geometry_aux_true
K7_dual_bank_scale_short
K8_post_zp_basis_learned
```

关键 chunks：

```text
chunk 5
chunk 6
chunk 10
chunk 16
chunk 20
```

horizons：

```text
h5
h8
h10
h15
```

### No-GT selector proxy

selector 不许用 GT。候选 proxy 包括：

```text
overlap pointmap residual
future h1/h2 self-consistency
TTT apply mismatch
scale_proxy_stability
read attention entropy stability
SWA overlap consistency
per-reset local scale variance
```

组合 selector score：

$$
S_{sel} =
w_1 E_{overlap}
+ w_2 E_{apply}
+ w_3 E_{scale}
+ w_4 E_{entropy}
+ w_5 E_{boundary}
$$

### 必须记录的指标

```text
candidate_id
parent_run
chunk_id
horizon
candidate_type
runtime_GT_used
selector_score
oracle_ATE
oracle_[200,300)
oracle_[400,600)
proxy_components
selected_by_proxy
rank_oracle
rank_proxy
```

### 可视化

```text
candidate_bank_oracle_heatmap.png
proxy_vs_oracle_scatter.png
selected_candidate_timeline.png
candidate_family_delta_boxplot.png
```

### 假设成立标准

Candidate bank oracle pass：

```text
1. At least one family has h10/h15 ATE delta <= -3m;
or
2. At least one family has [200,300) delta <= -5m with [400,600) regression <= +1m.
```

No-GT selector pass：

```text
1. Spearman(selector_score, oracle_ATE) <= -0.50
   because lower score means better candidate;
2. top-1 selected candidate is H9-or-better in >= 60% chunks;
3. selected-candidate h10/h15 average ATE delta <= -2m;
4. no selected candidate causes [400,600) regression > +2m.
```

如果 oracle bank 不过，不启动 selector。  
如果 selector 不过，不启动 full online validation。  
如果 selector correlation 低但 oracle 有强 family，Codex 应自动尝试 proxy ablation：

```text
remove one proxy component at a time
try rank-normalized proxy
try per-reset-group normalized proxy
try body/exit separate weights
try h1-to-h3 consistency rather than h5 proxy
```

---

## H4：当前 post-zp basis 太弱，因为 basis 是人工定义的；需要从历史 run bank 学 causal basis

### 假设内容

v18 的 PZBASIS 是 true-action，但 basis 设计仍然过于手工。真正的 drift-correction basis 应该从历史 runs 的 fast-weight deltas 与 trajectory deltas 中估计。

### 实验设计

建立 historical run bank：

```text
H9
C9
WINGAM
C16ROLE
TTGRL_04
P3G_01
PZ_05
freeze5
freeze56
NOGTPOSE diagnostic boundaries
best/worst v17/v18 candidates
```

对每个 run 抽取 per-layer/branch post-zp delta summary：

```text
D_{run,l,b} = vec(Delta W_{run,l,b})
```

构造 segment improvement target：

$$
y_{run} =
\Delta ATE_{[200,300)} -
\lambda \max(0, \Delta ATE_{[400,600)})
$$

学习一个低维 basis：

$$
B_k = \sum_{run} a_{k,run} D_{run}
$$

目标不是训练神经网络，而是做 PCA / ridge / partial least squares 级别的线性子空间审计：

```text
PCA basis
PLS basis toward [200,300) improvement
PLS basis with downstream penalty
C9-H9 contrast basis
freeze5-harmful contrast basis with continuity penalty
```

候选：

```text
LEARNBASIS_01:
    basis = C9 - H9
    branch = w0
    scale = 0.10

LEARNBASIS_02:
    basis = PLS_body_downstream_safe_1
    branch = w0
    scale = 0.10

LEARNBASIS_03:
    basis = PLS_body_downstream_safe_1
    branch = w0+w2
    scale = 0.10

LEARNBASIS_04:
    basis = freeze5_harmful_projected_orthogonal_to_C9
    branch = w0
    scale = 0.05
```

### 必须记录的指标

```text
basis_id
basis_source_runs
basis_layer_branch_norm
basis_cos_to_H9_delta
basis_cos_to_C9_delta
basis_cos_to_freeze5_removed
basis_cos_to_PZBASIS02
candidate_delta_norm
candidate_delta_cos_to_native
```

### 可视化

```text
basis_pca_scatter_by_run.png
basis_cosine_matrix.png
basis_layer_branch_heatmap.png
candidate_effect_by_basis.png
```

### 假设成立标准

Sandbox pass：

```text
h10/h15 ATE delta <= -3m
or
[200,300) delta <= -5m and [400,600) regression <= +1m
```

如果 learned basis only improves h5 but decays at h10/h15，Codex 应尝试：

```text
add continuity-preserving basis
restrict to later layers
restrict to w0 only
apply basis only at reset-body chunks
convert basis to W_short rather than W_long
```

如果 all learned basis fail below `-1m` h10/h15 ATE delta，stop Track D.

---

## H5：如果 TTT 必须修 trajectory-state，可能需要 dual-bank memory with explicit lifecycle

### 假设内容

当前 single fast-weight bank 既承担 long-term continuity，又承担 short-term correction，因此任何 strong correction 都会污染 downstream。需要明确分离：

```text
W_long:
    structure / continuity / global anchor

W_short:
    local correction / body-window drift mitigation
    decays across K chunks
```

### 实验设计

不是复用 v18 弱 DLTRUE，而是用 Track B/D 生成的 trajectory-state-aware delta 作为 W_short。

公式：

$$
W_{apply,m} = W_{long,m} + \rho_m W_{short,m}
$$

$$
W_{long,m+1} = W_{long,m} + \Delta W_{pos} + \lambda_{neu}\Delta W_{neu}
$$

$$
W_{short,m+1} = \delta W_{short,m} + \Delta W_{scale/correction}
$$

其中：

```text
rho_m: short apply scale
delta: short decay
Delta W_scale/correction: from Track B or Track D
```

候选：

```text
DBANK_01:
    W_short = SCALETTT_02 delta
    rho = 0.25
    decay = 0.50
    chunks = 5-12

DBANK_02:
    W_short = LEARNBASIS_02 delta
    rho = 0.25
    decay = 0.50
    chunks = 5-12

DBANK_03:
    W_long = H9 native continuity
    W_short = C9-H9 contrast
    rho = 0.15
    decay = 0.70
    chunks = 5-16

DBANK_04:
    W_short only apply to branch w0
    rho = 0.25
    decay = 0.50
```

### 必须记录的指标

```text
W_long_norm
W_short_norm
W_short_to_W_long_ratio
W_short_decay_history
W_short_apply_chunks
W_short_cos_to_native
W_short_cos_to_scale_basis
long_delta_norm
short_delta_norm
```

### 可视化

```text
W_short_norm_over_chunks.png
W_long_short_ratio_over_chunks.png
dual_bank_segment_effect.png
```

### 假设成立标准

```text
h10/h15 sandbox:
    ATE delta <= -3m
or:
    [200,300) delta <= -5m
    [400,600) regression <= +1m

full online:
    ATE <= 32.5 for weak pass
    ATE <= 30.0 for stage pass
    ATE <= 25.0 for final success
```

如果 W_short improves `[200,300)` but worsens `[400,600)`，Codex 应尝试：

```text
increase decay
reduce rho
apply W_short only to body chunks
protect W_long structure tokens
force W_short not to commit after exit chunks
```

If W_short has no h10/h15 effect, stop dual-bank.

---

## 4. 执行顺序与并行调度

### 4.1 总体调度

```text
Day 1 AM:
    Track A drift-state autopsy
    Track B0 scale proxy audit
    Track D historical basis extraction

Day 1 PM:
    Track B sandbox candidates
    Track D sandbox candidates
    Track C candidate bank manifest construction

Day 2 AM:
    Candidate bank oracle h10/h15 for B/D candidates
    If oracle passes, run no-GT selector offline

Day 2 PM:
    Only if selector passes, launch full KITTI01 online validation
```

### 4.2 并行策略

#### Offline / audit jobs

可以并行，不占 GPU 或少量 GPU：

```text
A1 drift dashboard
B0 scale proxy audit
D0 basis extraction
C0 manifest generation
```

#### Sandbox jobs

建议 GPU 0-5，最多 6 并发：

```text
short rollout h10/h15
one candidate per GPU
no full state snapshot unless needed
save only selected layers/branches
```

#### Full online jobs

只有 selector 通过才运行：

```text
max 4 并发
必须保存 full dashboard
必须保留 HMC hash
必须标注 counts_as_ttt_write
```

---

## 5. Full validation gate

任何 full online candidate 必须满足：

```text
counts_as_ttt_write = true
uses_gt_runtime = false
offline_postprocess = false
commit_mode = probe_ttt_write
full KITTI01 run complete
hmc rows = 38
```

通过标准分三层：

```text
weak progress:
    ATE <= 32.5
    and [200,300) <= 70
    and [400,600) <= 44.5

stage progress:
    ATE <= 30.0
    and [200,300) <= 60
    and [400,600) <= 40

final target:
    ATE <= 25.0
    and no segment regression > +3m vs H9
```

如果 candidate 只满足：

```text
ATE improved but [200,300) worsens
```

则不作为 target-25 主线，只作为 regularizer / drift redistribution result。

---

## 6. 失败自动分流规则

为了让 Codex 能并行推进，同时避免继续在无效空间里耗时，以下规则必须硬执行。

### 6.1 Track B scale-state TTT replay 失败

如果：

```text
scale proxy Spearman with future ATE < 0.35
```

Codex 自动尝试新的 proxy：

```text
overlap pointmap scale ratio
pose step median ratio
static anchor depth norm ratio
road/building semantic scale ratio
SWA overlap source scale ratio
```

如果所有 proxy 仍低于 `0.35`：

```text
停止 scale-token replay；
转 Track C candidate selector 或 Track D learned basis。
```

### 6.2 Candidate bank oracle 失败

如果：

```text
no family h10/h15 ATE delta <= -3m
and no family [200,300) delta <= -5m
```

不运行 selector/full。Codex 自动尝试：

```text
extend horizon to h15 if h10 near gate;
try chunk 5/6/10/16/20 separately;
try W_short instead of W_long;
try learned basis rather than hand basis.
```

如果二次 candidate bank 仍失败：

```text
TTT write-only 降级为 regularizer。
```

### 6.3 Selector 失败

如果：

```text
Spearman(selector_score, oracle_ATE) > -0.5
or selected candidate better-than-H9 ratio < 0.6
```

Codex 自动尝试：

```text
proxy component ablation
rank normalization
body/exit separate selector weights
shorter h3 proxy vs h10 target calibration
```

如果仍失败：

```text
不跑 full online。
```

### 6.4 Full online 失败

如果：

```text
full ATE > 33.0
or [200,300) worsens by > 1m
or [400,600) worsens by > 2m
```

该 family 停止。Codex 应回到 sandbox，不能继续 full sweep。

### 6.5 三条强 action family 全失败

如果以下都失败：

```text
Trajectory-state TTT replay
MPC candidate commit
Historical causal basis routing
Dual-bank trajectory-state memory
```

则给出明确结论：

```text
TTT write-only does not contain enough controllable drift-correction action under current LoGeR interface.
TTT remains a stabilizer / regularizer.
Target-25 mainline must move to explicit online trajectory-state / scale-state module.
```

---

## 7. 需要新增或修复的代码任务

### 7.1 Scale proxy audit

新增：

```text
tools/v19_scale_proxy_audit.py
```

输出：

```text
scale_proxy_by_chunk.csv
scale_proxy_correlation_summary.csv
scale_proxy_debug_maps/
```

### 7.2 Trajectory-state TTT replay

修改：

```text
loger/pipeline/ttt_write_controller.py
loger/pipeline/hybrid_memory_controller.py
run_pipeline_abc_v2.py
tools/run_attention_cue_experiment.sh
```

新增参数：

```text
--ttt_write_scale_state_mode
--ttt_write_scale_state_proxy
--ttt_write_scale_state_carrier
--ttt_write_scale_state_alpha
--ttt_write_scale_state_branch_mask
--ttt_write_scale_state_chunks
```

### 7.3 Candidate commit bank

新增 / 扩展：

```text
tools/run_v19_candidate_rollout.sh
tools/v19_candidate_bank_report.py
tools/v19_selector_audit.py
```

### 7.4 Historical basis learning

新增：

```text
tools/v19_historical_basis_fit.py
tools/v19_basis_candidate_builder.py
```

### 7.5 Dual-bank true memory

扩展：

```text
loger/pipeline/ttt_write_controller.py
loger/pipeline/hybrid_memory_controller.py
```

必须明确区分：

```text
W_long
W_short
W_apply
W_commit
```

并落盘：

```text
W_long_short_summary.jsonl
W_short_apply_history.jsonl
```

---

## 8. 最终判断标准

v19 不是为了得到一个小幅 best，而是为了回答：

```text
TTT fast weights 是否能通过 trajectory-state-aware write 接近 target-25？
```

如果有希望，应看到：

```text
sandbox h10/h15 ATE delta >= 3m 级别；
full online ATE 从 33.76 明显进入 32m 或以下；
[200,300) 不再停留在 74-76m；
[400,600) 不发生系统性 regression。
```

如果只能看到：

```text
h5 局部改善；
Rot / FinalErr 改善；
[200,300) 不动；
[400,600) regression；
ATE 仍在 33.7-34.1m；
```

则说明 TTT write-only 继续卡在 regularizer 层级，不足以作为 target-25 主线。

---

## 9. 一句话执行摘要

**下一阶段不要再调 TTT scalar gate。先用 scale/trajectory-state proxy 明确注入 TTT write，建立 candidate commit bank，并用 h10/h15 sandbox gate 决定是否值得 full online。若 trajectory-state-aware TTT 仍没有 3m 级别上界，TTT write-only 应正式降级为 regularizer。**

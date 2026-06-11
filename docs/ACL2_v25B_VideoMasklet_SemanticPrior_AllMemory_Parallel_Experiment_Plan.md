# ACL2 v25B：Video Masklet Semantic Prior Generator All-Memory 并行实验计划

日期：2026-05-22  
对象：LoGeR / HMC Pipeline v2 / Video Masklet Front-end / Semantic Prior Generator / Frame Attention / Global Attention / SWA / TTT  
主开发集：KITTI Odometry Sequence 01  
当前 deployable online TTT write best：`C9_P0_R2 = 33.7629421029m`  
目标：`KITTI01 ATE <= 25m`  
本计划定位：在短期无法获得 KITTI01 对齐的 dense 2D GT semantic 时，改用 **Video Masklet Front-end 的预测输出** 作为 Stage C 语义来源，继续验证 Semantic Prior Generator 是否能成为所有 memory path 的有效 role controller。

---

## 0. 本计划为什么从 GT semantic 改回 video masklet 输出

上一版 v25 计划原本希望使用 GT semantic 做 upper-bound 诊断。执行后发现，本地 KITTI Odometry sequence 01 没有能覆盖目标 frames 的 GT 2D semantic。已支持扫描的来源包括 KITTI semantic benchmark、KITTI-STEP、KITTI-360、SemanticKITTI 点云投影等，但本地 dense hit 和 projection hit 都为 0。因此 v25 在 Phase 0 GT semantic hard gate 处合法停止，没有启动 passive attribution、single-path、pairwise/all-memory、selector 或 full online。这个结论说明：当前不能把 predicted semantic fallback 伪装成 GT semantic，也不能继续等待 GT 阻塞实验推进。

因此，本轮 v25B 改为使用 **Video Masklet Front-end 输出**。这不是 GT 上界实验，而是 deployable semantic-memory 实验。所有 semantic 输入来自 Stage C / video masklet cache，必须标记为：

```text
uses_gt_semantic = false
uses_video_masklet_semantic = true
semantic_source = video_masklet_frontend_cache
```

如果 cache 是由 Video Masklet Front-end 根据当前视频离线生成、并在 HMC 实验中以 require-hit 方式读取，那么它可以作为 deployable pipeline 的一部分。它不是后处理 trajectory rewrite，不是 GT oracle，也不是数据集专用标签。

本轮的核心问题不再是“GT semantic 上界有多强”，而是：

> **在真实可部署的 predicted video masklet 条件下，Semantic Prior Generator 能不能把语义变成所有 memory path 的有效 role signal，并产生 h15 级别的持久 trajectory 改善？**

---

## 1. 实验整体目标

本轮不继续粗暴扩大 v24 的 coarse semantic all-memory 矩阵，也不把 Video Masklet 输出简单当作一个 `S_tok` 标量。v24 已经证明，工程通路基本可靠：no-op / pass-through parity 通过，frame/global/SWA/TTT 都可以消费 semantic role；但 predicted/coarse path-specific semantic role 的信号非常弱，Phase 2 最好的 h10 ATE delta 只有约 $-0.3175m$，最好的 h10 $[200,300)$ delta 约 $-1.9404m$，h15 更弱，没有 candidate 进入 pairwise / all-memory / selector / full online。

v25B 的目标是把 Semantic Prior Generator 从“语义分数生成器”升级为 **video-masklet-driven memory role router**。具体要回答七个问题。

### 1.1 问题 A：Video masklet 输出本身是否可靠到足以控制 memory？

在没有 GT semantic 的情况下，首先要判断 Stage C 输出是否可用。Video masklet 输出至少要满足：

```text
1. chunk/frame 对齐正确；
2. coverage 足够高；
3. semantic group / fine label 不为空；
4. masklet temporal continuity 可接受；
5. quality / trust score 能区分可靠和不可靠 masklet；
6. no-op / pass-through 不扰动 HMC trajectory。
```

如果这些基础条件不满足，后面的 memory-control 实验没有意义。Codex 应先修 Stage C cache / masklet projection / semantic group mapping，而不是跑候选矩阵。

### 1.2 问题 B：Video masklet semantic 应该在不同 memory path 中扮演什么角色？

LoGeR 的 memory path 功能不同：

```text
frame attention:
    当前 chunk 内跨帧读 source token，用于当前 read-path reasoning。

global/chunk attention:
    更高层的 chunk/global source context，用于全局几何整合。

SWA:
    相邻 chunk 的 local K/V source cache，用于 overlap continuity 和 local alignment。

TTT:
    压缩的 fast-weight global memory，用于长期 coordinate / scale / trajectory consistency。
```

同一个语义区域不能在四条路径里使用相同动作。例如 sky 在 TTT 中不应作为 strong positive long write，但在 frame/global/SWA 中可能提供 horizon / scale continuity；vegetation 在 high-D 时可能是 risky source，但在 low-D 远景中也可能帮助 continuity；road/building/wall/fence 更像 positive structure，但如果 TTT conflict 很高，也不能无条件长期写。

所以本轮要定义四套 path-specific role：

```text
R_frame  -> frame/global K/V source role
R_global -> chunk/global source role
R_swa    -> SWA cache / overlap source role
R_ttt    -> TTT positive / neutral / negative write role
```

### 1.3 问题 C：语义是否必须和 D_g / TTT conflict / scale-state risk 条件组合？

v24 失败的重要原因可能是 semantic role 太直接。语义标签本身不是 trajectory drift 的直接因果变量；它必须与几何和 TTT-native signal 组合。

本轮不再只测：

```text
sky skip
vegetation skip
structure keep
lowstuff highD skip
```

而是测条件策略：

```text
structure + low D_g + low TTT conflict:
    positive long memory

structure + high D_g or high TTT conflict:
    read source may keep, but long TTT write should be guarded

sky + low D_g:
    neutral keep, preserve horizon / scale continuity

sky + high D_g + high uncertainty:
    weak source skip or short negative, not hard long negative

vegetation + high D_g + high TTT conflict:
    source skip and/or short negative

low semantic trust:
    fallback to geometry / D_g / conflict, reduce semantic authority
```

### 1.4 问题 D：Video masklet quality 是否能作为 trust / routing，而不是直接 write score？

Semantic Prior Generator v2 的原则是职责分离：geometry 决定 eligibility，semantic 决定 value，mask quality 决定 trust/routing，chunk budget 主要由 geometry 决定。本轮必须继续遵守这个原则。

mask quality 不应该直接表示“区域不值得写”。更合理的是：

```text
high-quality masklet:
    semantic branch authority high

low-quality masklet:
    reduce semantic authority, fallback to geometry / D_g / TTT conflict

uncovered region:
    geometry-only prior, no semantic hard decision
```

### 1.5 问题 E：语义 role 能否让 h10 修正持久到 h15？

v20-v22 的核心模式是：h10 有局部 signal，但 h15 衰减。v24 更糟糕，h10 本身就不够强。v25B 的成功标准不能只看短期 h10，而必须看 durability：

$$
D_{dur} = \frac{|\Delta ATE_{h15}|}{|\Delta ATE_{h10}| + \epsilon}
$$

如果 h10 强但 h15 弱，说明语义只是 read/source filter，没有变成 memory commit。下一步必须转向 skip-aware TTT/SWA/global commit 或 lifecycle policy，而不是继续调阈值。

### 1.6 问题 F：Video masklet semantic 是否能够解释已有强信号？

已有强信号包括：

```text
C23 D_g
compact_kv / source skip
scale-state commit
update_conflict_energy
h10/h15 overwrite attribution
```

v25B 需要先做 attribution：这些强信号落在哪些 semantic fine labels / groups 上？

如果强信号主要落在 vegetation high-D，那么 semantic role 应该帮助 vegetation 条件 skip。  
如果强信号主要落在 road boundary 或 building edge，那么简单 lowstuff skip 就不是主因。  
如果强信号与语义无关，说明语义不是 Target-25 的主杠杆，只能作为辅助。

### 1.7 问题 G：不同数据集/序列可以诊断，但不能调参打榜

本轮允许记录不同数据集或不同序列的 semantic failure mode，但不允许为某个数据集单独调语义阈值、label table 或 gamma。所有策略必须使用同一套规则：

```text
same semantic taxonomy
same D_g / conflict / trust conditions
same role mapping
same promotion gates
```

如果 KITTI01 失败，不允许临时把 sky 阈值、vegetation 阈值、road 权重改成 KITTI01 专用。我们要研究机制，不是在数据集上打榜。

---

## 2. 实验总原则与硬边界

### 2.1 Stage C 必须使用离线 cache + require-hit

Video Masklet Front-end 计算慢，且在过去 v6 中已经发现 inline Stage C compute 即使被 HMC ignore，也可能扰动 full-sequence parity。因此本轮必须采用两阶段执行：

```text
Step 1: 离线生成 Stage C video masklet cache
Step 2: HMC 实验只读取 cache，require-hit，不 inline compute
```

运行规则：

```text
stage_c_cache_mode = read
stage_c_cache_require_hit = true
stage_c_inline_when_ignored = false
if cache missing:
    stop run
    do not fallback to empty/noop/predicted alternative
```

如果要重建 cache，必须在独立的 cache-build job 中完成，不能在 benchmark run 中临时 compute。

### 2.2 no-op / pass-through 必须是 hard gate

任何 semantic memory 实验前，必须先通过：

```text
semantic loaded but HMC ignored -> exact H9 parity
semantic pass-through consumed -> exact H9 parity
semantic debug-only all-memory -> exact H9 parity
```

通过标准：

```text
ATE_delta_vs_H9 = 0
raw_trans_max_diff = 0
hmc_state_hash unchanged where expected
context_empty_source_events = 0
no stale run directory contamination
```

如果 no-op 失败，Codex 不能继续跑矩阵，必须先排查参数接线、cache lookup、prior pass-through、write score override。

### 2.3 Video masklet results 可以算 deployable，但 short-rollout 不能

如果语义来自 video masklet frontend 的预测 cache，且不使用 GT、不使用 offline trajectory rewrite、不使用 benchmark feedback，那么 full online run 可以计为 deployable candidate。

但 short-rollout / causal fork 仍然只是 diagnostic：

```text
short_rollout = diagnostic_only
full_online = deployable validation
```

### 2.4 先 single-path，后 pairwise，再 all-memory

不允许一开始全路径同时控制。顺序必须是：

```text
Phase 2: single-path
    frame-only
    global-only
    swa-only
    ttt-only

Phase 3: pairwise
    只组合 single-path 过 gate 或接近 gate 的路径

Phase 4: all-memory
    只在 pairwise 不互相抵消时启动
```

### 2.5 不满足 gate 时，不跑 full online

本轮 full online 很贵，且目标差距仍约 $8.76m$。必须先用 trusted causal fork short-rollout 过滤。只有 candidate 满足 durability gate 后才允许 full online。

---

## 3. 关键输入与输出定义

### 3.1 Video Masklet 输出

Stage C 输出记为：

```text
MaskletOutput:
    M_mask[J,T,H,W]
    V_mask[J,T]
    Q_mask[J,T]
    L_sem[J]
    G_sem[J]
    W_sem[J]
    source_type[J]
    birth_frame[J]
    debug
```

本轮必须额外从 Stage C cache 或 Semantic Prior Generator 中导出：

```text
fine_label_patch[t,h,w]
coarse_group_patch[t,h,w]
masklet_id_patch[t,h,w]
mask_trust_patch[t,h,w]
semantic_trust_patch[t,h,w]
mask_coverage_patch[t,h,w]
```

如果 fine label 不存在，只能标记为 coarse fallback，不得声称 sky/vegetation/fence 等 fine policy 已验证。

### 3.2 Semantic Prior Generator v25B 输出

Semantic Prior Generator 不只输出 `A_tok`，还要输出 path-specific role tensors：

```text
R_frame_tok   # frame/global K/V source role
R_global_tok  # global/chunk source role
R_swa_tok     # SWA cache/source role
R_ttt_tok     # TTT write role
V_sem_tok     # semantic value
Q_sem_tok     # semantic trust
G_sem_tok     # semantic group
L_sem_tok     # optional fine label id
```

每条 role tensor 的值不应该只是语义类别，而是最终 memory role：

```text
0 = fallback / geometry-only
1 = positive_keep_or_write
2 = neutral_keep
3 = conditional_skip
4 = short_negative
5 = protect_neutral
```

### 3.3 Memory path 行为定义

不同 path 的 role 含义不同：

```text
Frame / Global source:
    positive_keep -> K/V source always keep
    neutral_keep -> source keep unless highD and high conflict
    conditional_skip -> skip source when highD or high conflict
    short_negative -> skip source; do not remove query
    protect_neutral -> keep source even if D_g is moderately high

SWA cache/source:
    positive_keep -> keep in previous/current overlap cache
    neutral_keep -> keep partial source for continuity
    conditional_skip -> downweight source in cache
    short_negative -> remove or weak-replace from source cache
    protect_neutral -> preserve overlap continuity

TTT write:
    positive_keep -> positive long write
    neutral_keep -> reduced but persistent write
    conditional_skip -> no long write; maybe short contribution
    short_negative -> short negative / non-persistent correction
    protect_neutral -> keep neutral, never negative
```

---

## 4. 核心假设与实验设计

## H0：Video masklet cache 与 semantic role plumbing 是可信的

### 假设

如果 Stage C cache 和 Semantic Prior Generator 接线正确，那么加载语义但不消费、pass-through 消费、debug-only all-memory 都应与 H9 reference 完全一致。Video masklet cache 不应引入 hidden side effect。

### 实验设计

固定：

```text
parent = H9 causal fork snapshots
read cue = C23 past
read path = current locked frame pair/all protocol
commit = probe_ttt_write
stage_c_cache_mode = read
stage_c_cache_require_hit = true
```

运行：

```text
P0_00_H9_REFERENCE
P0_01_VM_CACHE_LOADED_HMC_IGNORE
P0_02_VM_PASS_THROUGH_CONSUMED
P0_03_VM_DEBUG_ONLY_FRAME_GLOBAL_SWA_TTT
P0_04_VM_ROLE_NONEMPTY_SMOKE
P0_05_VM_COMPACT_KV_SMOKE
```

### 必须记录指标

```text
repro:
    ATE_delta_vs_H9
    raw_trans_max_diff
    hmc_hash_diff_count
    pose_txt_sha
    run_config_sha

cache:
    stage_c_cache_hit_rate
    chunks_with_masklets
    mean_masklets_per_chunk
    mean_coverage
    focus_coverage_200_300
    fine_label_available
    coarse_group_available

path:
    frame_role_consumed
    global_role_consumed
    swa_role_consumed
    ttt_role_consumed
    context_source_skip_requested
    compact_kv_requested
    compact_kv_applied_count
    empty_source_events

anti-contamination:
    stale_run_dir_moved
    no_inline_stage_c_compute
    no_gt_semantic
    no_predicted_fallback_alternative
```

### 成立标准

H0 通过：

```text
ATE_delta_vs_H9 = 0
raw_trans_max_diff = 0
stage_c_cache_hit_rate = 1.0
chunks_with_masklets >= 95% of chunks
mean_coverage >= 0.80
focus_coverage_200_300 >= 0.80
all requested path flags are true
empty_source_events = 0
```

### 不满足时 Codex 自动处理

```text
if cache_hit_rate < 1.0:
    rebuild offline Stage C cache with exact global frame ids; do not run HMC matrix.

if mean_coverage < 0.80:
    run cache-quality repair track: increase discovery stride density, enable stuff prompts, inspect prompts/cache layout.

if no-op parity fails:
    inspect HMC prior override, pass-through all-ones path, stage_d base reconstruction, stale run dir contamination.

if compact_kv requested but applied_count = 0:
    inspect run_attention_cue_experiment.sh CLI forwarding and pi3.py source mask path.

if fine_label unavailable:
    allow coarse fallback only, but mark fine policy as blocked; do not claim sky-specific experiments.
```

---

## H1：Video masklet semantic quality 决定语义是否有上界

### 假设

如果 video masklet 输出覆盖不足、边界碎片化、semantic group 过粗或 trust score 不可靠，则语义 memory control 不可能产生强信号。相反，如果 masklet quality 足够高但 memory control 仍弱，说明问题在 role/action，而非前端预测。

### 实验设计

先做 passive attribution，不改变模型行为，只记录 semantic 与已有强信号的关系。

分析对象：

```text
chunks = 5,6,10,16
horizon = h10,h15
signals:
    D_g
    update_conflict_energy
    scale_state_risk
    compact_kv skip mask
    C_anchor / uncertainty
    trajectory segment delta
```

对每个 semantic group / fine label 统计：

```text
coverage
D_g distribution
conflict distribution
scale-risk distribution
source skip overlap
TTT write mass overlap
h10/h15 overwrite attribution
```

### 必须记录指标

```text
per_group_coverage.csv
per_label_coverage.csv
per_group_dg_stats.csv
per_group_conflict_stats.csv
per_group_scale_risk_stats.csv
per_group_mask_trust_stats.csv
per_group_source_keep_ratio.csv
per_group_ttt_role_mass.csv
per_group_swa_cache_mass.csv
semantic_vs_dg_iou.csv
semantic_vs_conflict_iou.csv
semantic_vs_scale_risk_iou.csv
semantic_attribution_dashboard.md
```

核心相关性：

$$
\rho_{label,risk} = Spearman(coverage(label, chunk), segment\_error(chunk))
$$

$$
IoU(label, highD) = \frac{|label \cap highD|}{|label \cup highD|}
$$

### 成立标准

H1 通过的含义不是性能通过，而是 attribution 有解释力。至少满足一项：

```text
1. 某些 semantic labels 与 high-D/high-conflict/high-scale-risk 有稳定高 overlap；
2. disease chunks 中某些 semantic group 的 role mass 显著异常；
3. video masklet trust 能解释语义失败区域；
4. h10->h15 overwrite 主要集中在特定 semantic roles。
```

如果全部不成立，说明当前 video masklet semantic 与 drift 病灶弱相关，后续语义矩阵应缩小，不应大规模扩展。

### 不满足时 Codex 自动处理

```text
if semantic coverage looks high but labels weakly correlate with D_g/conflict/scale risk:
    switch to conditional semantic role only; do not run standalone semantic policies.

if fine labels are too noisy:
    collapse to robust groups: structure / lowstuff / movable / uncertain.

if coarse groups mix contradictory labels:
    create temporary diagnostic split from fine labels if available, otherwise tag as blocked.

if mask trust unreliable:
    reduce semantic authority and fallback to geometry-only branch.
```

---

## H2：Single-path semantic role 必须先产生强局部信号

### 假设

如果语义真的能帮助 memory control，那么至少有一条 memory path 在 single-path 条件下应产生可见 h10 或 h15 信号。否则 all-memory 组合大概率只是互相抵消。

### 实验设计

固定 H9 parent，分别测试：

```text
FRAME_ONLY:
    frame attention K/V source role

GLOBAL_ONLY:
    global/chunk attention source role

SWA_ONLY:
    SWA overlap/current/previous source cache role

TTT_ONLY:
    TTT positive/neutral/negative write role
```

每条 path 测三类 semantic role policy：

```text
Policy A: structure_keep
    structure + lowD -> keep / positive
    lowstuff -> neutral keep
    highD lowstuff -> conditional skip

Policy B: lowstuff_conditional_skip
    lowstuff + highD -> source skip / short negative
    lowstuff + lowD -> neutral keep
    structure -> keep

Policy C: conflict_conditioned_role
    semantic role must also satisfy update_conflict_energy or scale_state_risk
```

总矩阵：

```text
4 paths × 3 policies × chunks {6,10,16} × horizons {h10,h15}
```

优先只跑 h10，h10 接近 gate 才补 h15。

### 必须记录指标

```text
trajectory:
    ATE_delta_h10
    ATE_delta_h15
    [200,300]_delta_h10
    [200,300]_delta_h15
    [400,600]_delta_h10
    [400,600]_delta_h15
    Rot_delta
    FinalErr_delta
    YawRMSE_delta
    Sim3Scale_delta

path behavior:
    source_keep_ratio_by_role
    source_skip_mass_by_role
    compact_kv_applied_count
    SWA_cache_keep_mass_by_role
    SWA_cache_replace_mass_by_role
    TTT_positive_mass_by_role
    TTT_neutral_mass_by_role
    TTT_negative_mass_by_role
    post_zp_update_norm_by_role
    update_conflict_energy_by_role
```

### 成立标准

Single-path 候选通过至少满足一项：

```text
h10 [200,300) delta <= -3m
or h15 ATE delta <= -1.5m
or h15 [200,300) delta <= -2.5m
```

并且：

```text
[400,600) regression <= +1m
empty_source_events = 0
path consumption is non-empty
```

### 不满足时 Codex 自动处理

```text
if all FRAME_ONLY policies fail:
    try true compact_kv rather than bias mode if not already used; reduce hard skip to soft skip.

if GLOBAL_ONLY fails but FRAME_ONLY works:
    do not combine global path; keep global source native.

if SWA_ONLY improves h10 but hurts h15:
    switch to SWA cache lifecycle / partial keep, not hard source removal.

if TTT_ONLY fails:
    do not keep semantic as standalone write score; require D_g/conflict/scale-risk gating.

if all paths fail h10:
    stop semantic role matrix and return to attribution; likely semantic is not the primary causal variable.
```

---

## H3：Fine-label runtime policy 比 coarse group 更可能有效

### 假设

v24 的 coarse groups 太粗，`LOW_VALUE_STUFF` 混合 sky、vegetation、grass、sidewalk 等不同功能区域，导致语义 role 无法精准控制 memory。Video masklet output 如果有 fine labels，应优先验证 fine-label role。

### 实验设计

如果 H0/H1 确认 fine labels 可用，运行 fine-label policy：

```text
sky:
    frame/global: partial neutral keep; highD weak skip
    SWA: partial keep for horizon continuity
    TTT: neutral, never strong negative

vegetation/grass/tree:
    lowD: neutral keep
    highD + conflict: source skip / short negative

road/sidewalk:
    lowD: structure positive / SWA keep
    high conflict: avoid long TTT write

building/wall/fence:
    positive structure if lowD and low conflict

movable thing:
    source skip and short negative if highD

unknown/uncertain:
    fallback to geometry-only
```

与 coarse policy 对照：

```text
coarse_policy_A vs fine_policy_A
coarse_policy_B vs fine_policy_B
```

### 必须记录指标

```text
fine_label_coverage
fine_label_path_role_mass
fine_label_Dg_conflict_joint_stats
fine_vs_coarse_delta_report
sky_vegetation_road_building_ablation.csv
```

### 成立标准

Fine-label policy 成立：

```text
fine version improves h10 [200,300) by >= 1m over coarse version
or fine version improves h15 ATE by >= 0.75m over coarse version
and no downstream regression > +1m
```

### 不满足时 Codex 自动处理

```text
if fine labels unavailable:
    mark H3 blocked; do not run fake sky-specific policies.

if fine labels too sparse/noisy:
    create robust fine-supergroups: sky_horizon / vegetation / road_ground / structure_vertical / movable.

if sky-specific skip hurts:
    change sky to protect_neutral and only skip sky+highD+uncertainty.

if vegetation skip helps h10 but hurts h15:
    route vegetation correction to short-lifetime memory only.
```

---

## H4：All-memory semantic role 必须是 path-specific，不是同一 role 同时套所有 path

### 假设

v23/v24 的 all-memory 失败可能是因为同一 coarse role 同时作用到 frame/global/SWA/TTT，造成互相抵消。正确做法是 path-specific：frame/global 更偏 source read，SWA 更偏 local continuity，TTT 更偏 long write。

### 实验设计

只组合 H2/H3 中通过或接近 gate 的 path。组合顺序：

```text
Pairwise 1:
    FRAME + TTT

Pairwise 2:
    FRAME + SWA

Pairwise 3:
    FRAME + GLOBAL

Pairwise 4:
    SWA + TTT

All-memory:
    FRAME + GLOBAL + SWA + TTT
```

但每个 path 使用自己的 role：

```text
R_frame != R_swa != R_ttt
```

### 必须记录指标

```text
pairwise_interaction_table.csv
all_memory_path_mass.csv
path_conflict_matrix.csv
role_overlap_matrix.csv
h10_h15_durability_table.csv
```

path conflict 定义：

$$
Conflict(P_a, P_b) = \Delta ATE(P_a + P_b) - \min(\Delta ATE(P_a), \Delta ATE(P_b))
$$

如果 $Conflict > 0$，说明组合比单 path 更差，存在互相抵消或路径冲突。

### 成立标准

Pairwise 通过：

```text
pairwise h10 [200,300) delta <= best single-path - 1m
or pairwise h15 ATE delta <= best single-path - 0.75m
```

All-memory 通过：

```text
h10/h15 ATE delta <= -3m
or h10/h15 [200,300) delta <= -5m
and [400,600) regression <= +1m
and durability_ratio >= 0.45
```

### 不满足时 Codex 自动处理

```text
if pairwise worsens relative to both single paths:
    do not run all-memory with that pair; inspect path_conflict_matrix.

if FRAME+TTT works but FRAME+SWA fails:
    freeze SWA semantic role to native.

if all-memory h10 works but h15 fails:
    start durability attribution and lifecycle policy, not more semantic thresholds.

if all-memory worsens [400,600):
    add continuity protection role for SWA/global; do not increase skip strength.
```

---

## H5：Semantic role 的核心价值可能是让已有 source-skip / scale-commit 信号持久化

### 假设

v20/v21/v22 已经出现过强短期 signal，例如 context source skip + scale commit 可以在 h10 明显压低 $[200,300)$，但 h15 衰减。语义可能不是独立创造 signal，而是帮助这些已有 strong signal 更持久。

### 实验设计

选择已有 strong short signal 作为 base：

```text
base_A = compact_kv D_g q80 hard / structure rescue
base_B = scale-state commit / SCALECOMMIT-like action
base_C = skip-aware TTT commit
```

加入 video masklet semantic role：

```text
base_A + semantic structure protect
base_A + lowstuff conditional skip
base_B + semantic positive long structure
base_B + semantic short negative highD lowstuff
base_C + semantic lifecycle split
```

重点不是单独 semantic，而是 **semantic-as-persistence-router**。

### 必须记录指标

```text
h10_delta
h15_delta
durability_ratio
h10_to_h15_overwrite_ratio_HMC
h10_to_h15_overwrite_ratio_merge
TTT_state_move_after_h10
SWA_cache_refresh_after_h10
global_source_refresh_after_h10
per_role_persistence_mass
```

### 成立标准

H5 通过：

```text
base h10 signal is preserved or improved
and h15 delta improves by >= 1.5m over base
and durability_ratio increases to >= 0.45
and [400,600) regression <= +1m
```

### 不满足时 Codex 自动处理

```text
if semantic improves h10 but not h15:
    inspect overwrite source: TTT, SWA, global, or merge/gauge.

if TTT overwrite dominates:
    implement semantic-conditioned TTT long/short lifecycle.

if SWA overwrite dominates:
    implement semantic-conditioned SWA cache keep/protect.

if global source overwrite dominates:
    implement semantic-conditioned global source protect.

if merge/gauge dominates:
    semantic memory is insufficient; escalate to trajectory-state / scale-state module.
```

---

## H6：Video masklet trust 应决定 semantic authority

### 假设

Predicted video masklets can be wrong. If mask trust is ignored, semantic roles may corrupt memory. The correct policy is trust-weighted routing:

```text
high trust:
    semantic role has authority

medium trust:
    semantic role only conditions D_g/conflict, no hard action

low trust:
    fallback to geometry-only or TTT-native cue
```

### 实验设计

对通过 H2/H3 的 candidates 做 trust ablation：

```text
trust_mode = ignore_trust
trust_mode = hard_filter_low_trust
trust_mode = soft_trust_weight
trust_mode = fallback_geometry_low_trust
```

### 必须记录指标

```text
mask_trust_distribution
per_role_trust_mean
low_trust_action_mass
fallback_geometry_mass
semantic_authority_mass
trajectory_delta_by_trust_quantile
```

### 成立标准

Trust routing 成立：

```text
soft_trust or fallback improves h15 ATE by >= 0.5m over ignore_trust
or reduces [400,600) regression by >= 1m
without losing h10 [200,300) improvement by more than 0.5m
```

### 不满足时 Codex 自动处理

```text
if trust has no effect:
    check Q_mask projection and role tensors; it may not be consumed.

if hard_filter hurts:
    switch to soft trust, preserve neutral keep.

if low-trust masks dominate coverage:
    repair Stage C frontend/cache before more memory-control experiments.
```

---

## 5. 加速与并行执行策略

### 5.1 四条 Codex 并行 Track

本轮不串行等待所有实验。分成四个并行 track：

```text
Track A: Stage C cache / semantic role implementation audit
    目标：保证 cache、role、path consumption 正确。

Track B: Passive attribution + label taxonomy diagnosis
    目标：判断 video masklet semantic 是否与 D_g/conflict/scale risk 有关系。

Track C: Single-path semantic role screen
    目标：快速筛出 frame/global/SWA/TTT 哪条 path 有信号。

Track D: Strong-signal persistence combination
    目标：把 semantic role 接到 compact_kv / scale-commit / skip-aware TTT 上，看能否提升 h15 durability。
```

Track A 是 hard gate；B/C/D 可以在 A 通过后并行。

### 5.2 Speed-gated matrix

每个候选先跑：

```text
chunk10 h10
```

如果接近 gate，再跑：

```text
chunk10 h15
chunk6 h15
chunk16 h15
```

只有 h15 仍有信号，才进入 pairwise/all-memory。不要一开始全组合。

### 5.3 Run scheduling

建议调度：

```text
short rollout:
    6 GPU 并发可用，但每 GPU 只跑一个 worker。

full KITTI01 online:
    4 GPU 并发上限，避免 host RAM/swap 风险。

Stage C cache build:
    单独队列，不与 full HMC 混跑。
```

### 5.4 Stale-run 防护

每条 run 必须写入：

```text
run_config.yaml
run_config_sha
semantic_cache_sha
candidate_id
parent_id
source_commit_hash
DONE marker
```

如果目录已存在但 config sha 不匹配：

```text
move to .INVALID_RERUN_<timestamp>
rerun clean
```

---

## 6. 必须输出的可视化

### 6.1 Semantic Memory Dashboard

每个 top candidate 必须输出：

```text
RGB frame
video masklet overlay
semantic group map
fine label map if available
D_g map
update_conflict_energy map
scale_state_risk map
R_frame map
R_global map
R_swa map
R_ttt map
source skip mask
TTT positive/neutral/negative map
```

重点可视化 frames：

```text
chunks 6-10 around [200,300)
chunk16 diagnostic window
largest gain frame
largest regression frame
```

### 6.2 Path Mass Heatmap

输出：

```text
semantic group × memory path source/write mass
semantic label × memory path source/write mass
role × chunk index
role × horizon endpoint
```

### 6.3 Durability Attribution Plot

输出：

```text
h10 delta vs h15 delta scatter
h10->h15 overwrite ratio by memory path
TTT state move after h10
SWA cache refresh after h10
global source refresh after h10
merge/gauge move after h10
```

### 6.4 Trajectory Dashboard

输出：

```text
XY trajectory: H9, candidate h10, candidate h15
per-100f ATE curve
[200,300) focus curve
[400,600) downstream curve
Sim3 scale over time
Yaw error over time
FinalErr comparison
```

---

## 7. Promotion Gates

### Gate 0：implementation

```text
no-op parity exact
cache hit rate = 1.0
mean coverage >= 0.80
focus coverage >= 0.80
all requested path flags true
empty source events = 0
```

### Gate 1：single-path screen

```text
h10 [200,300) delta <= -3m
or h15 ATE delta <= -1.5m
or h15 [200,300) delta <= -2.5m

and [400,600) regression <= +1m
```

### Gate 2：pairwise/all-memory

```text
h10/h15 ATE delta <= -3m
or h10/h15 [200,300) delta <= -5m

and durability_ratio >= 0.45
and [400,600) regression <= +1m
```

### Gate 3：selector entry

```text
At least 2 candidate rows satisfy Gate 2
and proxy correlation between no-GT score and short-rollout ATE <= -0.35 or rank recall >= 0.60
```

### Gate 4：full online validation

```text
selector-selected candidate expected full ATE <= C9 - 1m
or short rollout predicts [200,300) improvement <= -5m with no downstream regression
```

### Final success

```text
KITTI01 full online ATE <= 25m
counts_as_deployable_online_success = true
uses_gt_semantic = false
no offline trajectory rewrite
no GT runtime action
```

---

## 8. 本轮停止规则

立即停止某一 family：

```text
1. 4 条同族 h10 run 均未达到 -1m ATE 或 -2m [200,300)；
2. h10 有信号但 h15 durability ratio < 0.15 且连续 3 条如此；
3. [400,600) regression > +2m 且无法用 continuity protect 修复；
4. path consumption debug 显示语义 role 没有实际改变 source/write mass；
5. no-op / pass-through parity 被破坏。
```

切换方向：

```text
if semantic single-path weak:
    move to semantic attribution + taxonomy repair, stop matrix expansion.

if frame/global source works but TTT write fails:
    keep semantic as source policy, do not force TTT semantic write.

if TTT semantic write works locally but not h15:
    test lifecycle / W_long-W_short, not more thresholds.

if all semantic paths fail:
    downgrade semantic to diagnostic and return main Target-25 search to trajectory-state / scale-state module.
```

---

## 9. 第一批具体执行清单

### Batch 0：cache and no-op audit

```text
P0_00_H9_REFERENCE
P0_01_VM_CACHE_LOADED_HMC_IGNORE
P0_02_VM_PASS_THROUGH_CONSUMED
P0_03_VM_DEBUG_ONLY_ALL_MEMORY
P0_04_VM_ROLE_NONEMPTY_SMOKE
P0_05_VM_COMPACT_KV_SMOKE
```

### Batch 1：passive attribution

```text
ATTR_01_CHUNK6
ATTR_02_CHUNK10
ATTR_03_CHUNK16
ATTR_04_DISEASE_WINDOW_200_300
```

### Batch 2：single-path h10 screen

```text
FRAME_A_STRUCTURE_KEEP
FRAME_B_LOWSTUFF_COND_SKIP
FRAME_C_CONFLICT_CONDITIONED

GLOBAL_A_STRUCTURE_KEEP
GLOBAL_B_LOWSTUFF_COND_SKIP
GLOBAL_C_CONFLICT_CONDITIONED

SWA_A_STRUCTURE_KEEP
SWA_B_LOWSTUFF_COND_SKIP
SWA_C_CONFLICT_CONDITIONED

TTT_A_STRUCTURE_POSITIVE
TTT_B_LOWSTUFF_SHORTNEG
TTT_C_CONFLICT_CONDITIONED
```

### Batch 3：h15 confirmation

Only top candidates from Batch 2.

### Batch 4：semantic + strong-signal persistence

```text
COMPACTKV_SEM_STRUCTURE_PROTECT
COMPACTKV_SEM_LOWSTUFF_COND_SKIP
SCALECOMMIT_SEM_STRUCTURE_LONG
SCALECOMMIT_SEM_LOWSTUFF_SHORTNEG
SKIPAWARE_TTT_SEM_LIFECYCLE
```

### Batch 5：pairwise / all-memory

Only if Batch 2/3/4 pass gates.

---

## 10. 本轮预期结论形式

本轮结束时必须明确给出以下判断：

```text
1. Video masklet semantic quality 是否足够？
2. 哪条 memory path 最适合 semantic role？
3. fine label 是否必要，还是 coarse group 足够？
4. semantic 是否必须依赖 D_g/conflict/scale-risk 条件？
5. semantic 是否能让 h10 correction 持久到 h15？
6. semantic 是否值得进入 no-GT selector / full online？
7. 如果不值得，失败是 semantic 前端问题、role 设计问题，还是 semantic 本身不是 Target-25 主因？
```

如果 video masklet semantic 仍没有产生强 h10 或 h15 signal，结论应明确写成：

```text
Semantic Prior Generator all-memory path is deployable and correctly connected,
but current video-masklet semantic role does not have enough causal power for Target-25.
Semantic should be kept as auxiliary diagnostic / weak source policy, not main target-25 driver.
```

如果语义产生 h10 强信号但 h15 弱，则结论应写成：

```text
semantic source/read correction is real,
but persistence is missing;
next step must be skip-aware memory commit / lifecycle, not more semantic threshold sweep.
```

如果 h15 也强，则启动 selector / full online validation。

---

## 11. 最终一句话

v25B 不再等待 GT 2D semantic。它以 Video Masklet Front-end 的真实预测 cache 为语义来源，重点验证 **Semantic Prior Generator 能否作为 frame/global/SWA/TTT 的 path-specific memory role controller**。本轮成功的标准不是“语义被接上”，因为这已经基本完成；本轮真正要证明的是：video masklet semantic 能否和 D_g、TTT conflict、scale-state risk 结合，产生 h15 级别的 durable memory correction。如果不能，就要把 semantic 降级为辅助诊断，而不是继续扩大 all-memory 语义矩阵。

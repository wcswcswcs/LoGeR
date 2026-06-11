# ACL2 v28：Semantic Prior Generator Token-Exact Risk-Conditioned All-Memory 并行实验计划

日期：2026-05-22  
对象：LoGeR / Pipeline v2 / HMC / Video Masklet Frontend / Semantic Prior Generator  
主目标：优先探索语义对所有记忆路径的作用，包括 frame attention、global/chunk attention、SWA、TTT。  
重要边界：本计划允许诊断不同数据集或不同序列的 failure mode，但不允许为了某个数据集单独调参。我们不是在 KITTI 或其他数据集上打榜，而是在寻找可解释、可迁移的 memory role 机制。

---

## 0. 当前判断与本轮计划的定位

v25B、v26、v27 已经把很多工程不确定性排除了。Video masklet cache 能够命中，coverage 高；Semantic Prior Generator 能够把语义 role 送进 frame/global/SWA/TTT；no-op、pass-through、debug-only smoke 不再扰动 HMC 输出；path consumption 也能被记录。也就是说，当前失败不再主要是“语义没有接上”。

当前最好的 deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

当前目标是：

```text
KITTI01 ATE <= 25m
```

所以当前差距约为：

$$
33.7629421029 - 25.0 = 8.7629421029m
$$

v27 的结果说明，当前 semantic causal role router 已经能接入，但 Phase 2 h10 screen 的普通候选最好只得到很弱的改善；修复 SWA hook 后，SWA path 在 $[200,300)$ 有局部改善，但 boundary 10f / 20f 指标回退。这说明现在的问题不是“语义完全没用”，而是“语义 role 还没有足够因果力，并且不同 memory path 的作用被混在一起了”。

本轮 v28 的核心目标不是继续扩大 coarse semantic / all-memory 矩阵，也不是继续扫 `LOW_VALUE_STUFF`、`STRUCTURE_ANCHOR` 的阈值。v28 要把 Semantic Prior Generator 升级为：

```text
fine label
+ token-exact D_g
+ token-exact mask trust
+ token-exact TTT conflict
+ token-exact scale-state risk
+ memory-path-specific role
```

最终输出不是一个单一 `A_tok`，而是四条 memory role stream：

```text
R_frame_tok  -> frame attention K/V source role
R_global_tok -> global/chunk attention K/V source role
R_swa_tok    -> SWA local cache / overlap source role
R_ttt_tok    -> TTT positive / neutral / negative write role
```

本轮要回答的不是“语义能不能控制所有 memory”这个过于粗的问题，而是：

> 在每条 memory path 中，语义如何和几何、TTT conflict、scale-state risk 共同决定 token 的 memory role？

---

## 1. 实验整体目标

本轮实验围绕五个科学问题展开。

### 1.1 问题 A：语义失败是因为语义本身弱，还是因为 risk 没有 token-exact？

v26/v27 里的 conflict / scale risk 不是严格 token-exact。v27 明确记录当前 conflict/scale risk 是 provenance-tagged broadcast 条件，而不是 token-level 条件。因此，即使语义 fine label 已经进入 runtime，也不能说已经验证了真正的：

```text
fine label + token-level conflict + token-level scale-risk router
```

v28 的第一目标是把 risk 从 chunk-level 或 broadcast-level 变成 token-aligned runtime streams：

```text
D_g_tok
Q_mask_tok
C_ttt_conflict_tok
S_scale_risk_tok
L_fine_tok
```

如果 token-exact risk-conditioned router 仍然没有强信号，才可以更有信心地说：当前语义不是 Target-25 的主因果变量。

### 1.2 问题 B：不同 memory path 是否需要完全不同的语义角色？

frame/global attention、SWA 和 TTT 的功能不同。

frame/global attention 关心的是当前 chunk 从哪些 K/V source token 读取上下文。SWA 关心的是相邻 chunk 的 local source topology 和 boundary continuity。TTT 关心的是长期 compressed fast-weight memory，是否把某些 token 的更新写进未来状态。

因此，不能把同一个语义 role 同时复制到所有 path。v28 必须把每条 path 的动作分开验证，再验证组合。

### 1.3 问题 C：语义能否作为 source/read filter，而不是 TTT write score？

过去很多 semantic TTT scalar prior 都很弱。相反，context source skip 和 compact K/V 曾经产生过更强 h10 短期信号。因此语义可能更适合先做 read/source role，而不是直接做 TTT write score。

本轮会优先区分：

```text
semantic for source routing
semantic for local SWA cache
semantic for long TTT write
semantic for short negative correction
```

### 1.4 问题 D：SWA 的语义 role 是否应该以 boundary/local continuity 为主？

v27 中 SWA hook 修复后，$[200,300)$ 有局部改善，但 boundary 10f / 20f 指标回退。说明 SWA 不能简单复用 TTT 的 semantic negative/positive 逻辑。SWA 是 local memory，语义 role 需要以 boundary、overlap、adjacent alignment 为主。

### 1.5 问题 E：语义能否跨数据集保持相同规则？

本轮可以诊断 KITTI01、KITTI00/02/05 或其他数据集的差异，但不能为某个数据集单独调阈值。所有规则必须固定：同一 taxonomy、同一 role rule、同一 gate、同一 default quantile。跨数据集只用于判断 failure mode，不用于打榜和过拟合。

---

## 2. 总原则与硬边界

### 2.1 不允许把 short-rollout 写成 deployable online success

所有 h10/h15 short rollout 都只是 causal diagnostic。只有 full online run 且不使用 GT runtime action、不使用 offline postprocess、不使用 no-GT pose rewrite，才可以计入 deployable online result。

### 2.2 不允许因为某个数据集失败就调专属参数

允许做：

```text
KITTI01 failure attribution
KITTI05 boundary diagnostic
KITTI02 scale-risk diagnostic
indoor/outdoor semantic distribution diagnostic
```

不允许做：

```text
KITTI01 专用 sky threshold
KITTI02 专用 vegetation role
KITTI05 专用 gamma
某个 sequence 专用 chunk window
```

如果某个数据集表现不同，只记录差异并分析原因，不在本轮调整规则。

### 2.3 所有候选必须先 single-path，再 pairwise，再 all-memory

不能一开始就 all-memory。执行顺序固定为：

```text
Phase 0: implementation hard gate
Phase 1: passive token-exact semantic-risk attribution
Phase 2: single-path short rollout
Phase 3: pairwise memory-path combination
Phase 4: all-memory semantic role router
Phase 5: durability / washout attribution
Phase 6: no-GT selector
Phase 7: full online validation
```

Phase 2 不过 gate，不启动 Phase 3。Phase 3 不过 gate，不启动 Phase 4。Phase 4 不过 gate，不启动 selector/full online。

### 2.4 如果结果只改善 h10，但 h15 弱，不能继续扫阈值

这类结果说明 correction 没有持久化。应该立刻做 washout attribution，而不是继续扫：

```text
highD quantile
semantic positive scale
negative scale
SWA skip threshold
TTT gamma
```

---

## 3. 需要新增或确认的核心张量

本轮最重要的工程前提是 token-exact runtime streams。每个 token 必须有以下字段：

```text
L_fine_tok             fine semantic label id
G_sem_tok              coarse semantic group id
Q_mask_tok             masklet trust / semantic trust
D_g_tok                ACL2 C23 past dynamic/read-risk cue
C_ttt_conflict_tok     token-level TTT update conflict energy
S_scale_risk_tok       token-level scale-state / trajectory-state risk
R_frame_tok            frame attention source role
R_global_tok           global/chunk attention source role
R_swa_tok              SWA cache/source role
R_ttt_tok              TTT write role
```

其中 `C_ttt_conflict_tok` 和 `S_scale_risk_tok` 是本轮关键。它们不能只是 chunk-level broadcast。如果暂时只能 broadcast，则必须标记：

```text
risk_level = broadcast
not_token_exact = true
candidate_allowed = false, except diagnostic rows
```

### 3.1 TTT conflict token stream

推荐从 TTT replay 的 token contribution 中构造：

$$
C_{conflict,i} = 1 - \cos(J_i, \Delta W_{static})
$$

其中 $J_i$ 是 token $i$ 对某个 TTT branch/layer 的 update contribution，$\Delta W_{static}$ 可以由低 $D_g$、高 trust、结构类 token 的 aggregate update 近似。如果现在还没有完整 token contribution，可以先做 selected layer/branch 的 token-level summary，而不是全层全 branch 一次性保存。

### 3.2 Scale-state token stream

`S_scale_risk_tok` 不应该只是 window-level scale proxy broadcast。可以先构造 token-to-window-risk proxy：

$$
S_{scale,i} = \operatorname{norm01}(U_i \cdot R_{window})
$$

其中 $U_i$ 是 token 的 effective write/update magnitude，$R_{window}$ 是当前 window 的 no-GT scale-risk proxy。后续再升级为 token update 对 scale proxy 的 finite-difference sensitivity。

### 3.3 Role 类型定义

每条 path 使用同一套 role id，但解释不同：

```text
0 = IGNORE_OR_FALLBACK
1 = SOURCE_KEEP / POSITIVE_LONG
2 = NEUTRAL_KEEP
3 = SOURCE_SKIP / NEGATIVE_SHORT
4 = PROTECT_NEUTRAL
5 = BLOCK_LONG_WRITE
```

在不同 path 中解释为：

```text
frame/global:
    1 = keep K/V source
    2 = weak keep
    3 = source skip / compact out
    4 = protected source, never skip

SWA:
    1 = keep overlap/local source
    2 = weak keep for continuity
    3 = skip high-risk previous source
    4 = protected boundary anchor

TTT:
    1 = positive long write
    2 = neutral write
    3 = short negative / non-persistent correction
    4 = protected neutral, not negative
    5 = block long write but keep read source
```

---

## 4. 核心假设与实验设计

## H0：token-exact semantic-risk role router 工程正确

### 假设

如果 token alignment、fine label projection、risk stream、role stream 或 path consumption 有任何一个出错，后续性能结果都不可信。v28 必须先通过 hard gate。

### 实验设计

运行以下 smoke：

```text
P0_00_H9_REFERENCE
P0_01_SEM_LOADED_IGNORED
P0_02_SEM_PASS_THROUGH_CONSUMED
P0_03_TOKEN_RISK_DEBUG_ONLY
P0_04_FRAME_SOURCE_SMOKE
P0_05_GLOBAL_SOURCE_SMOKE
P0_06_SWA_SOURCE_SMOKE
P0_07_TTT_WRITE_SMOKE
P0_08_ALL_PATH_DEBUG_ONLY
```

### 必须记录

```text
no-op:
    ATE_delta_vs_H9
    raw_translation_max_diff
    raw_pose_max_diff
    hmc_state_hash_match
    trajectory_txt_hash_match

semantic token:
    L_fine_tok_nonempty
    G_sem_tok_nonempty
    Q_mask_tok_nonempty
    D_g_tok_nonempty
    C_ttt_conflict_tok_nonempty
    S_scale_risk_tok_nonempty

role stream:
    R_frame_nonempty
    R_global_nonempty
    R_swa_nonempty
    R_ttt_nonempty
    per_role_count
    per_label_role_count

path consumption:
    frame_consumed
    global_consumed
    swa_consumed
    ttt_consumed
    context_empty_source_events
    source_keep_ratio
    source_skip_count

alignment:
    frame_id_match
    global_frame_start_end_match
    token_grid_shape_match
    masklet_to_patch_projection_valid
    ignore_label_fraction
```

### 成立标准

H0 通过必须满足：

```text
1. P0_01/P0_02/P0_03 对 H9 的 raw pose diff = 0 或 <= 1e-6。
2. 所有 token stream 非空。
3. 所有 role stream 非空。
4. 每条 path 的 consumed flag 为 true。
5. compact/source skip 没有 empty source event。
6. fine label runtime policy 可用，不允许 coarse-only fallback 伪装成 fine policy。
7. C_ttt_conflict_tok 和 S_scale_risk_tok 必须标记 token_exact=true；如果不是，则只允许 diagnostic，不允许 candidate gate。
```

### 如果不满足，Codex 先做什么

```text
if fine label missing:
    修复 VideoMasklet -> SemanticPriorGenerator 的 L_fine_tok 投影。
    检查 masklet fine_label id 是否稳定，不允许 masklet index 伪装成 label id。

if risk stream is broadcast:
    先实现 token-level risk summary。
    如果成本太高，只保存 selected TTT layers/branches，但不能把 broadcast 当 token-exact。

if path consumed false:
    检查 run_pipeline_abc_v2.py 参数透传、tools launcher 环境变量、HMC control_prior 重建顺序。

if source skip empty events:
    保护 special tokens，加入 minimum source keep floor，禁用全 source skip。

if no-op drift:
    停止所有 candidate，先修 no-op；不允许继续跑 Phase 2。
```

---

## H1：语义 fine label 单独不是主信号，但 fine label + token-exact risk 可能是有效 role router

### 假设

v25B/v26/v27 显示 semantic-only 很弱。语义标签本身只回答“这个区域是什么”，不能回答“它的 memory update 是否会造成 trajectory drift”。如果语义有用，它应该在和 $D_g$、TTT conflict、scale risk 结合后才变强。

### 实验设计

对同一组 path 做三类对照：

```text
semantic-only:
    fine label + mask trust only

risk-only:
    D_g + C_ttt_conflict + S_scale_risk only, no semantic label

semantic-risk:
    fine label + D_g + C_ttt_conflict + S_scale_risk + Q_mask
```

先只在 chunk10 h10 做 screen，不直接 h15/full。

候选：

```text
FRAME_SEM_ONLY
FRAME_RISK_ONLY
FRAME_SEM_RISK

GLOBAL_SEM_ONLY
GLOBAL_RISK_ONLY
GLOBAL_SEM_RISK

SWA_SEM_ONLY
SWA_RISK_ONLY
SWA_SEM_RISK

TTT_SEM_ONLY
TTT_RISK_ONLY
TTT_SEM_RISK
```

### 必须记录

```text
trajectory:
    h10 ATE_delta
    h10 [200,300)_delta
    h10 [400,600)_delta
    h10 Rot_delta
    h10 FinalErr_delta

semantic attribution:
    per_label_coverage
    per_label_D_g_mean/p90
    per_label_conflict_mean/p90
    per_label_scale_risk_mean/p90
    per_label_role_mass_by_path

memory action:
    source_keep_ratio_by_label
    source_skip_mass_by_label
    SWA_cache_keep_mass_by_label
    TTT_positive_mass_by_label
    TTT_neutral_mass_by_label
    TTT_negative_mass_by_label
    post_zp_update_norm_by_label
    update_conflict_energy_after_control
```

### 成立标准

H1 成立需要：

```text
semantic-risk 至少比 semantic-only 强 1.0m h10 ATE delta，或强 2.0m [200,300) delta。

且 semantic-risk 不能主要通过 [400,600) regression 换取 [200,300) 改善：
    [400,600)_delta <= +1.0m
```

进入 h15 条件：

```text
h10 ATE_delta <= -1.5m
or h10 [200,300)_delta <= -3.0m
```

如果 H1 不成立：

```text
如果 semantic-only、risk-only、semantic-risk 都弱：
    语义不是当前 path 的有效控制变量，停止该 path 的语义实验。

如果 risk-only 强、semantic-risk 不强：
    语义只做 attribution，不进入 control。

如果 semantic-risk 强但 h15 弱：
    转 Phase 5 washout attribution，不扫阈值。
```

---

## H2：frame/global 的语义 role 应主要控制 K/V source，而不是 TTT 写入

### 假设

VGGT4D 的启发是保留 query，不让高风险区域作为 K/V source。对 LoGeR 来说，frame/global attention 的语义 role 应该用于 source routing，而不是长期写入。语义高风险区域如果被当成 source，会污染当前 read；但它不一定应该进入 TTT negative。

### 实验设计

只测 frame/global source：

```text
FG_01_STRUCTURE_KEEP_LOW_RISK
FG_02_LOWSTUFF_HIGHD_SOFT_SKIP
FG_03_VEGETATION_HIGHD_SKIP
FG_04_SKY_NEUTRAL_NOT_NEGATIVE
FG_05_MOVABLE_HIGHD_SKIP
FG_06_FULL_DECISION_TREE
```

source skip 实现优先使用 compact K/V：

```text
query tokens = full length
K/V source = compacted by keep mask
special tokens = always kept
source keep floor = 0.75 initially
```

### 必须记录

```text
frame/global:
    source_keep_ratio
    skipped_token_count
    protected_token_count
    empty_source_events
    attention_entropy_before/after
    attention_mass_to_highD_before/after
    attention_mass_to_structure_before/after
    attention_mass_to_sky/vegetation_before/after

trajectory:
    h10/h15 ATE_delta
    h10/h15 [200,300)_delta
    h10/h15 [400,600)_delta
    Rot/FinalErr/Yaw delta
```

### 成立标准

frame/global source role 通过 h10 gate：

```text
h10 [200,300)_delta <= -3m
or h10 ATE_delta <= -1.5m
```

进入 h15 gate 后，必须满足：

```text
h15 ATE_delta <= -1.5m
or h15 [200,300)_delta <= -3m

and [400,600)_delta <= +1m
and source_keep_ratio >= 0.75
and empty_source_events = 0
```

如果 h10 强但 h15 弱：

```text
Codex 自动检查：
    1. 被 skip 的 source 是否在后续 chunks 又进入 SWA/TTT/global memory。
    2. source skip 是否只影响 read，没有影响 commit。
    3. 是否需要 skip-aware TTT/SWA commit，而不是再扫 threshold。
```

---

## H3：SWA 的语义 role 必须以 boundary/local continuity 为主

### 假设

SWA 是 local lossless memory，不是长期 global memory。语义 source skip 如果破坏 overlap anchor，就可能改善 $[200,300)$ 却伤 boundary。v27 的 SWA hook 修复后已经出现这种现象。因此 SWA 语义 role 必须围绕 boundary metric 优化。

### 实验设计

只测 SWA path：

```text
SWA_01_STRUCTURE_OVERLAP_KEEP
SWA_02_ROAD_WALL_FENCE_KEEP
SWA_03_LOWSTUFF_HIGHD_SOFT_SKIP
SWA_04_VEGETATION_HIGHD_WEAK_SKIP
SWA_05_SKY_LOWD_PARTIAL_KEEP
SWA_06_MOVABLE_HIGHD_SKIP
SWA_07_BOUNDARY_PROTECTED_TREE
```

动作不是简单 source skip，而是：

```text
source keep / weak keep / protect / skip
previous-source only vs previous+current source
overlap source protection
boundary anchor protection
```

### 必须记录

```text
SWA metrics:
    boundary_10f_ATE
    boundary_20f_ATE
    chunk_boundary_pose_jump
    overlap_pointmap_residual
    previous_source_keep_ratio
    current_source_keep_ratio
    semantic_group_source_mass
    fine_label_source_mass
    SWA_cache_update_norm
    SWA_attention_mass_to_skipped_labels

trajectory:
    h10/h15 ATE_delta
    h10/h15 [200,300)_delta
    h10/h15 [400,600)_delta
```

### 成立标准

SWA candidate 不能只看 ATE。必须满足：

```text
h10 [200,300)_delta <= -3m
or h10 ATE_delta <= -1.5m
```

同时 boundary 不得明显变差：

```text
boundary_10f_delta <= +0.25m
boundary_20f_delta <= +0.25m
chunk_boundary_pose_jump_delta <= +0.25m
```

如果 ATE 有改善但 boundary 回退：

```text
Codex 自动尝试：
    1. 把 hard skip 改成 soft skip。
    2. 加 structure/road/wall/fence overlap protect。
    3. 只作用 previous-source high-risk，不作用 current-source。
    4. 降低 source skip 范围，保留 top support anchors。
```

如果 boundary 仍然回退，则停止该 SWA semantic family。

---

## H4：TTT 的语义 role 应该是 positive / neutral / short-negative，而不是 scalar write score

### 假设

TTT 是长期 compressed memory。语义不应该直接变成一个写入分数，而应该帮助决定 token 的生命周期：

```text
positive long:
    low-risk structure anchor

neutral keep:
    horizon / sky low-risk / stable background

short negative:
    high-risk semantic + high D_g + high conflict

block long write:
    high conflict structure that may still be useful for read but不适合写入长期 memory
```

### 实验设计

只测 TTT path：

```text
TTT_01_STRUCTURE_LOW_RISK_POS_LONG
TTT_02_LOWSTUFF_LOW_RISK_NEUTRAL
TTT_03_LOWSTUFF_HIGHD_HIGHCONFLICT_SHORTNEG
TTT_04_STRUCTURE_HIGHCONFLICT_BLOCK_LONG
TTT_05_VEGETATION_HIGHD_CONFLICT_SHORTNEG
TTT_06_SKY_LOW_RISK_PROTECT_NEUTRAL
TTT_07_FULL_DECISION_TREE
```

每个候选都记录 positive/neutral/negative/block 的 mass，不能只记录最终 `A_tok`。

### 必须记录

```text
TTT role mass:
    positive_mass
    neutral_mass
    negative_mass
    block_long_mass
    protected_neutral_mass

per label:
    positive_mass_by_label
    neutral_mass_by_label
    negative_mass_by_label
    block_mass_by_label

fast-weight:
    branch0_update_norm
    branch1_update_norm
    branch2_update_norm
    post_zp_update_norm
    update_cosine_to_H9
    update_conflict_energy_before/after
    scale_risk_before/after

trajectory:
    h10/h15 ATE_delta
    h10/h15 [200,300)_delta
    h10/h15 [400,600)_delta
    FinalErr/Yaw/Sim3Scale delta
```

### 成立标准

TTT candidate 进入 h15 条件：

```text
h10 ATE_delta <= -1.5m
or h10 [200,300)_delta <= -3m
```

TTT candidate 进入 pairwise 条件：

```text
h15 ATE_delta <= -1.5m
or h15 [200,300)_delta <= -3m

and [400,600)_delta <= +1m
and negative_mass is not zero
and positive_mass is not zero
```

如果 TTT semantic-only 弱，而 risk-only 强：

```text
语义降级为 attribution / role explanation，不作为 TTT write controller 主条件。
```

如果 short negative 改善 h10 但 h15 弱：

```text
Codex 自动做 washout attribution：
    1. h10 endpoint state vs h15 endpoint state 的 TTT tail update。
    2. negative token role 是否被后续 chunk 改写。
    3. block_long 是否需要持续到 reset-window end。
```

---

## H5：all-memory 组合只有在单 path 有强信号时才有意义

### 假设

如果 single-path 都没有强信号，all-memory 组合大概率只是互相抵消或产生小波动。v23/v24/v25B 已经反复显示粗 all-memory 弱。因此 v28 只允许通过 gate 的 path 进入组合。

### 实验设计

进入 Phase 3 的组合只包括 Phase 2 通过或接近通过的 path。组合顺序：

```text
Pairwise:
    frame + TTT
    frame + SWA
    global + TTT
    SWA + TTT

All-memory:
    frame + global + SWA + TTT
```

组合策略不是简单并集，而是 role reconciliation：

```text
如果 frame wants skip but SWA wants protect:
    SWA overlap protect wins for boundary source.

如果 TTT wants negative but frame wants source keep:
    allow read source, block long TTT write.

如果 semantic lowstuff lowD:
    neutral keep, not negative.

如果 highD + high conflict + low trust:
    short negative or skip depending on path.
```

### 必须记录

```text
pairwise_conflict_matrix:
    label
    frame_role
    global_role
    swa_role
    ttt_role
    conflict_count
    resolved_role

role override counts:
    frame_overridden_by_swa
    ttt_negative_blocked_by_protect
    source_skip_downgraded_to_weak_keep

performance:
    h10/h15 ATE_delta
    h10/h15 segment deltas
    boundary metrics
    update metrics
```

### 成立标准

pairwise 通过：

```text
h15 ATE_delta <= -2m
or h15 [200,300)_delta <= -4m

and [400,600)_delta <= +1m
and boundary metrics do not regress > +0.25m
```

all-memory 通过：

```text
h15 ATE_delta <= -3m
or h15 [200,300)_delta <= -5m

and durability_ratio >= 0.45
and [400,600)_delta <= +1m
```

如果 pairwise 比 single-path 弱：

```text
停止 all-memory。
Codex 自动生成 role conflict heatmap，并选择冲突最大的 path 做回退。
```

---

## H6：跨数据集只做诊断，不做调参

### 假设

如果 semantic role router 是有效机制，它不应只依赖 KITTI01 专用阈值。不同数据集可以暴露不同 failure mode，但规则应固定。

### 实验设计

只对通过 h15 gate 的 1-2 个 candidate 做跨数据集 short-rollout diagnostic。候选规则完全固定，不调整阈值。

建议数据：

```text
KITTI00 / KITTI02 / KITTI05:
    driving domain diagnostic

如果有非 KITTI 数据：
    只做 semantic coverage / role distribution / failure mode audit
```

### 记录指标

```text
per dataset:
    semantic_label_distribution
    D_g_distribution_by_label
    conflict_distribution_by_label
    scale_risk_distribution_by_label
    role_mass_by_path
    h10/h15 ATE_delta
    segment delta
    boundary delta
```

### 成立标准

不要求所有数据都 ATE 最优，但要求：

```text
1. candidate 不出现系统性崩坏；
2. role distribution 合理，不出现某个数据集所有 source 被 skip；
3. 同一规则在至少两个序列上有同向 h10/h15 信号；
4. 如果某数据集失败，只记录 failure mode，不为其单独调参。
```

---

## 5. 可视化要求

每个通过 H0 的候选都必须生成以下可视化。没有可视化的结果不能进入下一阶段。

### 5.1 Token map grid

每个关键 chunk 输出：

```text
RGB
fine semantic label
D_g
Q_mask
C_ttt_conflict
S_scale_risk
R_frame
R_global
R_swa
R_ttt
source skip mask
TTT positive / neutral / negative / block map
```

重点 chunk：

```text
chunk6
chunk10
chunk16
```

如果候选涉及 h10/h15，必须可视化 h10 endpoint 和 h15 endpoint 的同类 map。

### 5.2 Per-label role dashboard

输出柱状图：

```text
label coverage
label mean D_g
label mean conflict
label mean scale risk
label source keep ratio
label TTT positive mass
label TTT negative mass
label h10/h15 contribution
```

### 5.3 Memory path interaction matrix

对 pairwise/all-memory 候选，输出：

```text
frame role vs TTT role conflict matrix
global role vs SWA role conflict matrix
SWA protect vs frame skip conflict matrix
semantic label vs final resolved role matrix
```

### 5.4 Trajectory and segment dashboard

输出：

```text
trajectory XY overlay
per-frame translation error
per-100f ATE curve
[200,300), [200,400), [400,600) bar plot
h10 vs h15 delta plot
durability ratio plot
Sim3 scale over time
Yaw error over time
```

### 5.5 SWA boundary dashboard

只要候选涉及 SWA，必须输出：

```text
boundary_10f_ATE bar
boundary_20f_ATE bar
chunk_boundary_pose_jump
SWA previous/current source keep ratio
overlap pointmap residual
semantic source mass in overlap
```

### 5.6 Washout attribution dashboard

如果 h10 强 h15 弱，必须输出：

```text
base -> h10 state delta
h10 -> h15 tail update delta
TTT tail update norm
SWA cache replacement mass
global source role changes
merge/gauge state movement
role mass changed after h10
```

---

## 6. 加速与并行执行策略

### 6.1 并行 track

Codex 可以并行执行四条 track，但每条都必须先过 H0。

```text
Track A: token-exact risk stream implementation and audit
Track B: frame/global K/V source router
Track C: SWA boundary-aware semantic router
Track D: TTT positive/neutral/short-negative semantic router
```

Track A 是硬前置。如果 Track A 没过，B/C/D 只能跑 diagnostic，不能进入 gate。

### 6.2 短滚动优先

所有候选先跑：

```text
chunk10 h10
```

通过 gate 后再跑：

```text
chunk10 h15
chunk6 h15
chunk16 h15
```

只有 h15 或 durability gate 过，才允许 pairwise / all-memory。

### 6.3 资源控制

建议：

```text
short-rollout: 6 并发以内，根据显存和 RAM 动态限制
full online: 2-4 并发，不做 8 并发
long cross-dataset: 2 并发
```

所有 launcher 必须：

```text
跳过 DONE rows
污染目录自动 .INVALID_RERUN_*
写 run_config.yaml
写 candidate_manifest.json
写 gate_summary.json
不允许 stale JSONL 混入新结果
```

---

## 7. Codex 自动失败分流规则

### 7.1 H0 失败

```text
if no-op drift:
    stop all candidates
    compare H9 direct pose
    inspect pass-through PriorOutput
    inspect write score base prior reconstruction

if fine labels missing:
    inspect VideoMaskletOutput.L_sem
    inspect stable fine-label id mapping
    inspect chunk global frame indexing

if risk token stream missing:
    implement selected-layer token conflict summary
    do not run candidate matrix

if path consumed false:
    inspect CLI forwarding and HMC control_prior rebuilding
```

### 7.2 Phase 2 h10 弱

```text
if semantic-only weak and risk-only weak:
    stop this memory path semantic control

if risk-only strong but semantic-risk weak:
    semantic is hurting; use semantic only for attribution

if semantic-risk stronger than semantic-only but still below gate:
    inspect per-label maps; try only top two causal labels, not full taxonomy
```

### 7.3 h10 强但 h15 弱

```text
run washout attribution immediately
try skip-aware commit or lifecycle only if attribution points to a specific memory path
if washout path unclear:
    do not sweep thresholds
```

### 7.4 SWA boundary 回退

```text
switch hard skip -> soft skip
protect structure overlap tokens
protect road/wall/fence boundary anchors
apply previous-source-only skip
if boundary still regresses:
    stop SWA semantic family
```

### 7.5 all-memory 组合弱

```text
compute role conflict matrix
remove the path with largest destructive override
if pairwise weaker than both single paths:
    stop all-memory for that family
```

### 7.6 跨数据集失败

```text
if one dataset fails:
    record label/risk distribution shift
    do not tune threshold

if all datasets fail:
    semantic is not primary control signal; demote to attribution/regularizer

if only one dataset succeeds:
    mark as dataset-specific diagnostic, not general mechanism
```

---

## 8. 最终成功标准

### 8.1 阶段成功

v28 阶段成功不要求直接到 ATE 25，但必须达到至少一个：

```text
h15 ATE_delta <= -3m
or h15 [200,300)_delta <= -5m with [400,600)_delta <= +1m
or full online ATE improves current deployable best by >= 1m without [200,300) regression
```

### 8.2 Target-25 成功

最终成功必须是：

```text
full online deployable run
no GT runtime action
no offline postprocess
no trajectory rewrite
KITTI01 ATE <= 25m
```

### 8.3 失败判定

如果以下情况同时成立：

```text
1. token-exact risk stream 已经实现；
2. path-specific semantic router 已经过 single-path 和 pairwise gate；
3. no candidate h15 ATE_delta <= -1.5m；
4. no candidate [200,300)_delta <= -3m；
5. cross-dataset diagnostic 没有同向信号；
```

则应得出结论：

```text
Semantic Prior Generator 不是 Target-25 主杠杆。
语义降级为 attribution / auxiliary regularizer。
主线回到 trajectory-state / scale-state / TTT-native causal action。
```

---

## 9. 本轮推荐最小执行清单

第一批只做 20 条以内，避免再次扩大成低价值矩阵。

```text
Phase 0:
    9 条 smoke / audit

Phase 1:
    passive token-exact attribution on H9 chunks 6,10,16

Phase 2 h10:
    FRAME_SEM_ONLY / RISK_ONLY / SEM_RISK
    GLOBAL_SEM_ONLY / RISK_ONLY / SEM_RISK
    SWA_SEM_ONLY / RISK_ONLY / SEM_RISK
    TTT_SEM_ONLY / RISK_ONLY / SEM_RISK

If any pass:
    run h15 for passing rows only

If none pass:
    stop. Do not run pairwise/all-memory.
```

预期最有价值的候选不是 semantic-only，而是：

```text
FRAME_SEM_RISK
SWA_SEM_RISK with boundary protection
TTT_SEM_RISK with positive/neutral/short-negative roles
```

本轮最关键的判断不是“哪条 ATE 最低”，而是：

> token-exact risk-conditioned semantic role 是否比 semantic-only 和 risk-only 都更有因果力。

如果答案是否定的，后续就不要继续把语义放在主线位置。

---

## 10. 一句话总结

v28 的目标不是继续证明语义可以接入；这一点已经基本成立。v28 要验证的是：**语义能否在 token-exact risk 条件下，成为 frame/global/SWA/TTT 的因果 memory role router。** 如果不能，语义应从 Target-25 主线降级为辅助诊断；如果能，则继续做 pairwise / all-memory / selector / full online。 

# ACL2 v38: Training-Free Semantic Memory Control, Durability, and Target-30 Plan

日期：2026-05-24  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / frame attention / global attention / SWA / TTT  
目标：在 **training-free** 边界内，把语义信息从“标签规则”升级为 **memory-path-aware 的风险与锚点控制机制**，优先验证能否把当前 deployable best `C9_P0_R2 = 33.7629421029m` 推进到 `ATE <= 30m`。

---

## 0. 项目目标与硬边界

本项目不是训练一个新模型，不是在 KITTI01 上拟合一个触发器，也不是用 oracle label 选择 chunk。项目目标是：

> 在不训练新模型、不训练 trigger / selector / classifier / role router、不使用 GT runtime、不针对单一数据集调参的前提下，从 LoGeR 内部 cue、VideoMasklet 语义、SWA/TTT memory 状态中构造 deterministic memory-control policy，提升长序列重建质量。

本轮必须遵守：

```text
禁止训练 trigger / selector / classifier / role router
禁止用 oracle rollout label 拟合规则
禁止使用 absolute chunk id 作为策略条件
禁止为 KITTI01 或任何单个数据集单独调 threshold / gamma / label table
禁止把 short rollout / fixed-window / diagnostic / GT 或 projected semantic audit 写成 deployable success
禁止只用 [200,300) 作为策略目标；它只能作为 KITTI01 stress diagnostic
```

成功标准分三层：

```text
Stage success:
    full online ATE <= 32m

Target-30 success:
    full online ATE <= 30m

Mechanism success:
    h15 或 full online 中，语义 memory policy 产生可持久改进，且不是牺牲 downstream continuity 换来的局部改善。
```

当前 deployable best：

```text
C9_P0_R2
ATE = 33.7629421029m
Target-30 gap = 3.7629421029m
```

---

## 1. 本轮重新定义问题

v37 证明，当前 semantic action family 已经能够进入模型路径，但仍没有产生 deployable result：

```text
Track 0 action/influence audit: pass
Track 1 frame/global source surgery h10: fail
Track 2 SWA h10: local diagnostic pass
Track 2 SWA h15: fail
Track 3 TTT h10: scale-state diagnostic pass
Track 3 TTT h15: fail
Track 4 semantic C23 path isolation h10: fail
Track 5 full online: not launched
```

这说明现在不能再继续问：

```text
sky 要不要 skip？
vegetation 要不要 negative？
structure 要不要 positive？
```

这些问题太粗。真正应该问：

> 哪些语义风险区域在 LoGeR 的真实计算中承载了高影响力？它们是在 frame/global read source、SWA local cache、还是 TTT long memory 中造成问题？相反，哪些语义区域是 static scale anchors，应该被保护并长期写入？

因此 v38 的核心问题是：

```text
1. 语义风险区域是否真的承载 high influence？
2. high influence 风险区域应该在哪条 memory path 上被抑制？
3. stable semantic anchors 是否应该在 frame/global、SWA、TTT 中被共同保护？
4. h10 有效但 h15 失效，是被哪条 memory path 或 merge/gauge state 洗掉？
5. 能否用固定、training-free、scene-agnostic 的规则进入 full online Target-30？
```

---

## 2. 核心科学假设

### H1：过去语义失败，不等价于“天空/植被/动态物体不干扰重建”

之前很多实验失败，可能是因为 action 过弱、path 错、hook 被压扁、或只测了单 masklet，而不是因为语义风险假设错。VGGT4D 的启发不是“语义标签直接写规则”，而是 **风险区域不应作为 early-stage source token 影响几何推理**。

在 LoGeR 中，这个思想必须拆成三类 action：

```text
frame/global:
    控制 K/V source，不删 Query。

SWA:
    控制 local overlap / previous-source cache，优先保护 boundary continuity。

TTT:
    控制 long-memory write lifecycle，区分 positive long、neutral、short negative、no-long-write。
```

### H2：进入 Target-30 的关键不是单纯 suppress dynamic，而是 suppress risky source + preserve static scale anchors

如果只删除动态/低价值区域，很容易造成局部改善但 downstream 回退。更可能的成功机制是：

```text
risk source suppression
+
static scale-anchor protection
+
path-specific memory lifecycle
```

也就是说，同一 semantic masklet 可能在不同 path 中有不同角色：

```text
stable road/building/wall/fence:
    frame/global source keep
    SWA overlap anchor protect
    TTT positive long write

high-D vegetation/dynamic/lowtrust:
    frame/global source attenuation or skip
    SWA non-overlap skip or V attenuation
    TTT short negative or no-long-write

sky / horizon / far background:
    not positive long TTT
    frame/global partial keep if low risk
    SWA boundary keep if it helps local continuity
```

### H3：SWA 的目标不是修固定区间，而是 scene-agnostic local continuity

SWA 不能以 `[200,300)` 作为主目标。SWA 是 local memory，评价必须看 all-boundary 和 rolling-window health。

SWA 成功必须满足：

```text
1. rolling high-error windows 改善；
2. boundary_10f / boundary_20f 不系统性回退；
3. overlap residual 不回退；
4. downstream rolling windows 不崩；
5. 不依赖固定 chunk id 或 KITTI01-specific segment。
```

### H4：TTT 语义写入不能是 semantic scalar，必须是 lifecycle policy

TTT 是 compressed global memory。语义对 TTT 的作用应是：

```text
stable structure -> positive long
risky dynamic/vegetation/shadow -> short negative or no-long-write
neutral sky/background -> neutral, not positive long, not hard negative
```

如果一个 correction 只 h10 有效、h15 失效，优先做 washout attribution，而不是继续加大负写强度。

### H5：semantic-conditioned C23 的 local signal 必须以 residual 方式进入，并与 C9 做 path isolation

v31/v34 说明 semantic-conditioned C23 有局部信号，但 all-chunks full 和 C9 interaction 失败。下一步不再做 learned trigger，也不使用 fixed chunk。只允许 deterministic residual：

$$
D_{final} = D_{base} + \lambda (D_{sem} - D_{base})
$$

其中 $\lambda$ 是固定或由能量归一得到的 training-free 系数，不从数据集拟合。

---

## 3. 总体实验结构

v38 分为 6 个并行 Track。Track 0 是 instrumentation / influence atlas，不阻塞所有实验，但会控制解释可信度。Track 1-4 是四条可并行 action family。Track 5 是组合和 full online。

```text
Track 0: Semantic Influence Atlas v2
Track 1: VGGT4D-style frame/global source surgery with static rescue
Track 2: SWA local-continuity semantic source policy
Track 3: TTT static-anchor positive long + risky short-negative
Track 4: semantic-conditioned C23 residual path isolation
Track 5: minimal combination + full online Target-30 validation
```

所有 Track 默认使用：

```text
training-free deterministic policy
no learned trigger
no learned selector
no absolute chunk id
no GT semantic at runtime
VideoMasklet Stage-C cache as runtime semantic source
SemanticKITTI projected 3D only as offline trust/audit evidence
```

---

## 4. Track 0：Semantic Influence Atlas v2

### 4.1 目标

Track 0 不再只是检查 hook 是否能跑，而是建立每个 semantic group / fine label 在每条 memory path 上的 **实际影响力图谱**。

它要回答：

```text
1. 哪些语义区域真的承载 frame/global attention mass？
2. 哪些语义区域真的进入 SWA overlap / non-overlap source cache？
3. 哪些语义区域真的贡献 TTT post-zp update delta？
4. 不同 semantic policy 产生的 action 是否真的不同？
5. 被 skip / attenuate 的 source 是否原本真的被模型使用？
```

### 4.2 实验设计

Codex 生成一组 no-trajectory 或 h3 smoke，覆盖：

```text
A. synthetic stress:
    all_patch_skip
    center_box_skip
    random_20pct_skip
    left_half_skip
    all_dynamic_role
    all_static_role

B. semantic group stress:
    dynamic_highD
    vegetation_highD
    sky_highD
    lowtrust_highD
    structure_lowD
    structure_lowD_lowConflict

C. path stress:
    frame only
    global only
    frame+global
    SWA only
    TTT only
```

### 4.3 必须记录的指标

每个 row 必须落盘：

```text
action_tensor_summary.csv
policy_jaccard_matrix.csv
per_label_action_mass.csv
per_masklet_action_mass.csv
frame_source_keep_ratio_by_label.csv
global_source_keep_ratio_by_label.csv
swa_overlap_keep_ratio_by_label.csv
swa_nonoverlap_keep_ratio_by_label.csv
ttt_role_mass_by_label.csv
attention_mass_removed_before_after.csv
source_attention_mass_by_label.csv
swa_source_attention_mass_by_label.csv
ttt_post_zp_update_norm_by_label.csv
protected_token_count.csv
context_empty_source_events.csv
```

### 4.4 判断标准

Track 0 不是性能 gate，但要给每条 action family 标注可信度：

```text
Reachability pass:
    action count / keep ratio / TTT role mass 有非零变化
    context_empty_source_events = 0

Influence pass:
    removed source attention mass before >= 0.03
    or TTT post-zp update norm changed >= 3%
    or SWA source mass changed >= 3%

Distinctness pass:
    Jaccard(policy_A, policy_B) <= 0.90
    or per-label keep/mass difference >= 0.05
```

如果 reachability 失败，Codex 必须先修 hook。  
如果 reachability 通过但 influence 低，Codex 不停止实验，但必须把该 family 改成 high-influence selection，而不是继续低影响力 mask。

### 4.5 必须可视化

```text
semantic_group_memory_path_heatmap.png
action_jaccard_heatmap.png
source_attention_mass_removed_bar.png
swa_overlap_nonoverlap_keep_bar.png
ttt_role_mass_by_label_bar.png
influence_atlas_by_chunk.html or png grid
```

---

## 5. Track 1：VGGT4D-style frame/global source surgery with static rescue

### 5.1 目标

验证“风险区域不作为 early read source”是否能提升重建，但不能破坏 static anchors。这里优先作用 frame/global K/V source，不动 TTT long write。

### 5.2 核心原则

```text
Query 保留；
K/V source 控制；
structure low-risk source 保护；
semantic high-risk source skip；
先 frame/global，不先动 TTT；
优先 compact_kv，attention bias 只做对照。
```

### 5.3 候选族

```text
FG_01_DYNAMIC_HIGHD_SKIP:
    dynamic thing + highD -> source skip

FG_02_VEGETATION_HIGHD_SKIP:
    vegetation/tree/grass + highD + mask trust ok -> source skip

FG_03_LOWTRUST_HIGHD_SKIP:
    lowtrust or inconsistent masklet + highD -> source skip

FG_04_STRUCTURE_RESCUE:
    structure lowD -> source protect
    highD risky stuff -> skip

FG_05_RISK_SKIP_STATIC_RESCUE:
    dynamic/vegetation/lowtrust highD skip
    road/building/wall/fence lowD protect

FG_06_COMPACT_KV_TRUE:
    same as FG_05 but true compact_kv source compaction

FG_07_BIAS_ONLY_CONTROL:
    same masks but attention bias only, to compare with compact_kv
```

### 5.4 记录指标

除 ATE/Rot/RPE 外，必须记录：

```text
frame_source_keep_ratio
global_source_keep_ratio
attention_mass_removed_before/after
attention_mass_to_structure_before/after
attention_mass_to_risky_stuff_before/after
context_empty_source_events
protected_structure_mass
rolling_50f/100f/200f ATE mean/p90/worst
all-boundary 10f/20f mean/p90/worst
segment diagnostics only as stress, not success criterion
```

### 5.5 成立标准

h10 entry：

```text
ATE delta <= -1.5m
or rolling_100f_worst_delta <= -3m
or any high-error rolling window delta <= -5m
```

h15 durability：

```text
h15 ATE delta <= -2m
or h15 rolling_100f_worst_delta <= -4m
and boundary_10f_p90_delta <= +0.25m
and boundary_20f_p90_delta <= +0.25m
and downstream rolling_200f_worst_delta <= +1m
```

如果 FG track h10 失败但 influence atlas 显示 removed attention mass 很低，Codex 自动改成 high-attention-risk source selection。  
如果 compact_kv 失败但 bias-only 有效，Codex 检查 compact path 的 source index / protected tokens。  
如果 FG 改善 rolling windows 但 boundary 恶化，进入 SWA continuity repair，不直接 full。

---

## 6. Track 2：SWA local-continuity semantic source policy

### 6.1 目标

SWA 不是修固定区间的工具，而是 local continuity memory。本 track 评估 semantic control 是否能改善 adjacent chunk alignment，同时不破坏 boundary。

### 6.2 候选族

```text
SWA_01_NONOVERLAP_RISK_REMOVE:
    high-risk dynamic/vegetation/lowtrust only in non-overlap source -> remove
    overlap source preserved

SWA_02_OVERLAP_K_KEEP_V_ATTEN:
    overlap high-risk tokens keep K, attenuate V

SWA_03_STRUCTURE_OVERLAP_PROTECT:
    road/building/wall/fence lowD overlap tokens protected

SWA_04_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_PROTECT:
    dynamic highD non-overlap remove
    structure overlap protect

SWA_05_SKY_HORIZON_NEUTRAL:
    sky/horizon tokens never positive long; in SWA overlap, keep K, attenuate V only if highD/highConflict

SWA_06_SOURCE_TOPOLOGY_CONTROL:
    preserve source topology by limiting removal ratio per frame/semantic group
```

### 6.3 记录指标

```text
boundary_10f_mean/p90/worst
boundary_20f_mean/p90/worst
chunk_boundary_pose_jump_mean/p90
overlap_pointmap_residual_mean/p90
swa_overlap_keep_ratio_by_label
swa_nonoverlap_keep_ratio_by_label
swa_k_keep_ratio
swa_v_attenuation_ratio
swa_source_attention_mass_by_label
rolling_50f/100f/200f ATE mean/p90/worst
downstream_regression_max
```

### 6.4 成立标准

SWA h10 pass 不能只看单一 segment。必须：

```text
rolling_100f_worst_delta <= -3m
or h10 ATE delta <= -1.5m
```

同时：

```text
boundary_10f_p90_delta <= +0.25m
boundary_20f_p90_delta <= +0.25m
overlap_residual_p90_delta <= +0.25m
downstream_200f_worst_delta <= +1m
```

h15 pass：

```text
h15 ATE delta <= -2m
or h15 rolling_100f_worst_delta <= -4m
and same boundary/downstream safety constraints
```

如果 SWA 改善一个局部 high-error window 但 boundary p90 回退，Codex 自动尝试：

```text
K preserve / V attenuate
non-overlap-only removal
overlap structure anchor protection
per-frame max removal cap
```

如果所有 SWA rows 轨迹完全相同，Codex 必须回到 Track 0 修 SWA action identity，不继续 rollout。

---

## 7. Track 3：TTT static-anchor long write + risky short-negative

### 7.1 目标

验证语义是否能帮助 TTT 区分长期 scale anchors 与短期风险源。TTT 不再使用 semantic scalar，不再简单少写 dynamic，而是显式生命周期控制。

### 7.2 候选族

```text
TTT_01_STRUCTURE_LONG:
    structure lowD lowConflict lowScaleRisk -> positive long

TTT_02_STRUCTURE_LONG_DYNAMIC_NOLONG:
    structure lowD -> positive long
    dynamic highD -> no-long-write

TTT_03_VEGETATION_CONDITIONAL_SHORTNEG:
    vegetation highD highConflict -> short negative
    vegetation lowD -> neutral

TTT_04_LOWTRUST_SHORTNEG:
    lowtrust highD highConflict -> short negative

TTT_05_SKY_NEUTRAL:
    sky lowD -> neutral
    sky highD -> no-positive-long, weak source attenuation only

TTT_06_FULL_LIFECYCLE_POLICY:
    structure anchor positive long
    dynamic/vegetation/lowtrust high-risk short negative or no-long-write
    sky/far background neutral
```

### 7.3 记录指标

```text
ttt_positive_long_mass_by_label
ttt_short_negative_mass_by_label
ttt_no_long_write_mass_by_label
per_branch_update_norm w0/w1/w2
post_zp_delta_norm_by_branch
update_conflict_energy_before_after
scale_state_proxy_before_after
hmc_state_rel_diff
rolling_50f/100f/200f metrics
h10/h15 durability ratio
[400,600) or downstream rolling regression
```

### 7.4 成立标准

```text
h10 ATE delta <= -1.5m
or rolling_100f_worst_delta <= -3m
```

h15：

```text
h15 ATE delta <= -2m
or h15 rolling_100f_worst_delta <= -4m
and downstream_200f_worst_delta <= +1m
and TTT update norm not abnormal
```

如果 TTT h10 weak：Codex 检查 TTT role mass 是否真的改变 post-zp delta。  
如果 TTT h10 strong / h15 weak：Codex 保存 tensor-state snapshots，做 washout attribution，而不是继续调 gamma。  
如果 positive long structure 伤 ATE：Codex 检查 anchor quality，加入 lowConflict / lowScaleRisk / high attention mass 条件。  
如果 short negative 伤 downstream：Codex 减少 long write impact，只做 no-long-write 而不是 negative replay。

---

## 8. Track 4：semantic-conditioned C23 residual path isolation

### 8.1 目标

语义最好信号曾来自 C23 / D_g reconditioning，而不是直接 memory action。本 track 只做 training-free residual，不训练 trigger，不使用 fixed chunk id。

### 8.2 候选族

设：

$$
D_{res} = D_{sem} - D_{base}
$$

使用固定 residual：

$$
D_{final} = D_{base} + \lambda D_{res}
$$

$\lambda$ 只允许固定值或能量归一值，不允许拟合：

$$
\lambda = \min\left(1, \frac{\rho \cdot RMS(D_{base})}{RMS(D_{res})+\epsilon}\right)
$$

其中 $\rho$ 取固定值，比如 `0.25` 或 `0.5`。

候选：

```text
C23R_01_READ_ONLY_RESID:
    semantic residual only affects frame/global read cue

C23R_02_NO_TTT:
    semantic residual never enters TTT write score

C23R_03_NO_SWA:
    semantic residual never controls SWA source

C23R_04_FRAMEGLOBAL_COMPACT_ONLY:
    semantic residual only used for frame/global compact_kv source surgery

C23R_05_STATIC_RESCUE_RESID:
    semantic residual + structure lowD rescue

C23R_06_C9_COMPAT_READ_ONLY:
    C9 base, semantic residual read-only, no SWA/TTT semantic control
```

### 8.3 记录指标

```text
D_base statistics
D_sem statistics
D_res RMS / p90 / sign distribution
semantic-conditioned z by label
frame/global attention mass changes
SWA action flags disabled/enabled
TTT write score changes disabled/enabled
rolling windows metrics
full-online compatibility with C9 if h15 passes
```

### 8.4 成立标准

h10：

```text
ATE delta <= -1.5m
or rolling_100f_worst_delta <= -3m
or high-error window delta <= -5m
```

h15：

```text
h15 ATE delta <= -2m
or h15 rolling_100f_worst_delta <= -4m
and [400,600)-style downstream regression <= +1m
```

Full online only if h15 pass and no path conflict.

If residual all-chunks fails but read-only works, Codex must isolate C9 components:

```text
C9 without SWA semantic effect
C9 without TTT semantic write effect
C9 read-only semantic residual
C9 semantic residual with original TTT write
```

No learned trigger is allowed.

---

## 9. Track 5：minimal combination and full online

### 9.1 Entry conditions

Track 5 starts only if at least one of Track 1-4 passes h15 durability.

### 9.2 Combination philosophy

Do not combine all successful rows blindly. Combine by memory role:

```text
read source surgery from Track 1
+
SWA boundary-safe local policy from Track 2
+
TTT static-anchor/short-negative from Track 3
+
semantic C23 residual from Track 4 only if it does not conflict
```

### 9.3 Full online candidates

At most 4 full rows first:

```text
FULL_01 = best single h15 candidate
FULL_02 = best read/source + SWA-safe combination
FULL_03 = best read/source + TTT lifecycle combination
FULL_04 = best all-compatible minimal combination
```

### 9.4 Full online metrics

```text
ATE
Rot
RPE_t / RPE_r
FinalErr
YawRMSE
Sim3Scale
rolling_50f / 100f / 200f mean / p90 / worst
all-boundary_10f / 20f mean / p90 / worst
overlap residual mean / p90
per-reset block ATE
per-path action summary
runtime state hash / HMC rows
```

### 9.5 Promotion

```text
Full useful:
    ATE <= 32m
    no severe boundary/downstream regression

Target-30:
    ATE <= 30m
    no GT runtime
    no learned trigger/selector
    no offline trajectory rewrite
```

If full row improves local high-error windows but ATE remains >33m, classify as diagnostic only.

---

## 10. Cross-sequence and no dataset-specific tuning

After any candidate reaches full ATE <= 32m on KITTI01, run diagnostic cross-seq:

```text
KITTI00 / KITTI02 / KITTI05
same fixed semantic policy
same thresholds
same lambda/rho
same memory path rules
```

Allowed:

```text
diagnose failure mode differences
report semantic distribution shifts
report action mass differences
report rolling-window differences
```

Forbidden:

```text
per-sequence chunk id
per-sequence label table
per-sequence gamma/beta/threshold
per-dataset sky/vegetation tuning
```

If cross-seq fails, do not tune per dataset. Instead classify the mechanism as KITTI01-specific diagnostic unless a scene-agnostic reason is found.

---

## 11. Visualization package

Every h10/h15 candidate that reaches a local gate must output:

```text
RGB + semantic overlay
D_g base heatmap
D_sem / D_res heatmap if applicable
source_attention_mass heatmap
frame/global source keep/drop map
SWA overlap/non-overlap keep/drop map
TTT positive/neutral/short-negative/no-write map
static anchor map
risk source map
rolling-window ATE curve
all-boundary error curve
overlap residual plot
h10 vs h15 durability curve
trajectory overlay for H9/C9/candidate on full or short rollout
```

If spatial overlays are missing, Codex must not fabricate them. It should write:

```text
spatial_overlay_unavailable = true
missing_tensors = [...]
```

and continue performance rollout only if action reachability is already proven.

---

## 12. Codex parallel execution plan

### Codex A：Track 0 Influence Atlas

```text
Run synthetic + semantic stress.
Generate action/influence reports.
If action equivalent, repair role projection.
If influence low, switch candidate selection to high-attention/high-update tokens.
```

### Codex B：Track 1 Frame/Global

```text
Run FG_01-FG_07 h10 on chunks 6/10/16 from H9 and C9 parent where available.
If h10 weak, check attention mass removed.
If mass low, rerun high-influence risk selection.
If compact_kv differs from bias, debug path.
```

### Codex C：Track 2 SWA

```text
Run SWA_01-SWA_06 h10.
Immediately compute all-boundary and overlap metrics.
If local improves but boundary regresses, try K preserve / V attenuation / non-overlap-only.
If all SWA rows identical, stop and fix hook identity.
```

### Codex D：Track 3 TTT

```text
Run TTT_01-TTT_06 h10.
Check post-zp delta and branch update changes.
If no actual update change, fix TTT role mapping.
If h10 strong and h15 weak, save tensor states for washout.
```

### Codex E：Track 4 C23 Residual

```text
Run C23R_01-C23R_06 h10.
If semantic residual helps only read path, keep it isolated.
If it hurts C9, run component isolation no-SWA/no-TTT.
```

### Codex F：Full online coordinator

```text
Only launch full rows from h15-qualified candidates.
Use max 4 GPUs for full KITTI01.
If no h15 candidate, do not launch full.
Generate final decision table.
```

---

## 13. Failure routing

### Case 1：action hook not real

```text
Symptoms:
    source keep ratio unchanged
    role mass unchanged
    post-zp delta unchanged

Action:
    fix hook / projection / model control path
    rerun Track 0 only
```

### Case 2：action real but attention/update influence low

```text
Symptoms:
    keep/drop counts changed
    removed attention mass < 0.03
    TTT update norm change < 3%

Action:
    rerun with high-influence selection:
        high source attention mass
        high update norm
        high D_g/conflict/scale-risk
```

### Case 3：frame/global source skip weak

```text
Action:
    test compact_kv vs bias
    add static structure rescue
    reduce aggressive skip to high-risk only
    if still weak, demote frame/global semantic source surgery
```

### Case 4：SWA local improves but boundary regresses

```text
Action:
    protect overlap anchors
    keep K / attenuate V
    apply skip only to non-overlap source
    cap removal ratio per frame
```

### Case 5：TTT semantic write weak

```text
Action:
    check post-zp delta changed
    require lowConflict/lowScaleRisk for positive long
    change negative to no-long-write if short negative hurts downstream
    if still weak, semantic is not TTT primary signal; use TTT-native cue instead
```

### Case 6：h10 strong but h15 weak

```text
Action:
    save tensor-state snapshots
    measure TTT/SWA/global/merge movement from h10 to h15
    decide whether washout comes from TTT tail update, SWA refresh, global source update, or merge/gauge
    do not continue threshold sweep
```

### Case 7：full online worse than C9

```text
Action:
    isolate component conflict:
        read only
        no SWA
        no TTT
        C9 write unchanged
    if all fail, semantic family is diagnostic-only under current architecture
```

---

## 14. Final decision criteria

At the end of v38, write one of four conclusions.

### Conclusion A：semantic memory control is deployable

```text
At least one full online row <= 32m,
preferably <= 30m,
without learned trigger / dataset tuning / postprocess.
```

### Conclusion B：semantic helps read/source but not durable memory

```text
h10 strong, h15/full weak,
washout attribution identifies memory overwrite path.
```

### Conclusion C：semantic hooks are real but causal effect weak

```text
action/influence real,
all h10 candidates weak,
no path reaches local gate.
```

### Conclusion D：implementation still suspect

```text
action identity or influence mass cannot be trusted,
hook repair remains priority.
```

If Conclusion C is reached after action realism is proven, Semantic Prior Generator should be demoted from Target-30 mainline to diagnostic/trust calibration, while Target-30 mainline should move to:

```text
explicit online trajectory-state / scale-state
C9-native lifecycle repair
TTT-native causal action
merge/gauge-aware correction
```

---

## 15. One-line summary

v38 的核心不是继续写语义规则，而是验证：

> 天空、植被、阴影、动态物体这些语义风险区域，是否真的以高影响力进入 LoGeR 的 frame/global/SWA/TTT 计算路径；如果它们有影响力，应该在哪条 memory path 上被 source-skip、local-protect、short-negative 或 positive-long 处理，才能形成 h15/full-online 的持久 trajectory improvement。

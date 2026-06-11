# ACL2 v25：GT Semantic Prior Generator All-Memory 诊断与并行实验计划

日期：2026-05-22  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / GT semantic labels  
主开发目标：判断 **GT 语义信息是否能作为所有 memory path 的可靠 role signal**，并探索它是否能帮助 TTT/SWA/frame/global memory 形成更持久的轨迹修正。  
重要边界：本轮使用 GT semantic 是为了诊断语义上界，不是为了在某个数据集上打榜，也不允许根据某个序列专门调参。

---

## 0. 为什么要做这一轮

v23/v24 已经说明，Semantic Prior Generator 的工程通路基本打通了。语义 role 可以进入 frame attention、global/chunk attention、SWA、TTT，no-op/pass-through 也能保持严格不扰动。但是结果仍然很弱：v24 最好的 h10 ATE delta 只有约 $-0.32m$，最好的 h10 $[200,300)$ delta 约 $-1.94m$，h15 改善更弱，没有 candidate 通过进入 pairwise/all-memory/selector/full online 的 gate。

这说明当前失败不太像“语义没有接上”，而更像下面三个问题之一：

1. **语义预测不够准**：Mask2Former/Stage C predicted semantic 的 label、边界或时间对齐不够好，导致 Semantic Prior Generator 的输入本身有噪声。
2. **语义 role 定义不对**：即便语义准确，当前 coarse role 仍然不能直接解释轨迹 drift。
3. **语义不是主因果变量**：trajectory / scale drift 主要由 geometry/TTT state 决定，语义只能作为辅助条件。

因此，本轮必须使用 GT semantic 信息做上界诊断。如果 GT semantic 仍然不能产生显著 h10/h15 改善，那就说明问题不是预测语义不准，而是语义本身在当前 memory controller 中没有足够因果力。反过来，如果 GT semantic 能产生强信号，那下一步才值得回头研究 predicted semantic 如何逼近这个上界。

---

## 1. 本轮整体目标

本轮目标不是把某个数据集调到最好，而是回答五个科学问题。

### 1.1 问题 A：GT 语义能不能产生明显强于 predicted semantic 的 memory-control 信号？

v24 的 predicted/coarse semantic signal 很弱。GT 语义应该先作为 upper bound：

```text
如果 GT semantic 仍然弱：
    语义路线本身不是当前 Target-25 主杠杆。

如果 GT semantic 明显强：
    说明之前卡点至少部分来自 semantic prediction / projection / role mapping。
```

### 1.2 问题 B：语义应当控制哪一条 memory path？

LoGeR 的 memory 是混合结构：

```text
frame/global attention:
    当前 chunk 内部或跨帧 source token 的读取路径。

SWA:
    相邻 chunk 的局部 K/V source/cache，负责 local continuity。

TTT:
    压缩的 fast-weight global memory，负责长期 coordinate / scale / trajectory consistency。
```

同一个语义类别在不同路径中不应使用同一个动作。例如，sky 可能不适合作 TTT positive long write，但它可能对 horizon / scale continuity 有帮助，不能在 frame/global/SWA 中全部 hard skip。

### 1.3 问题 C：GT 语义是否能帮助已有强信号变得持久？

v20-v22 的共同模式是：

```text
h10 有信号；
h15 衰减；
full online 不启动。
```

所以本轮不只看短期 h10，还要看 GT semantic 是否能让 correction 持久到 h15。重点不是“短期能否压低 $[200,300)$”，而是：

```text
h10 -> h15 的 durability 是否提升。
```

### 1.4 问题 D：语义 role 是否必须条件化于 D_g / TTT conflict / scale-state risk？

本轮不能只测：

```text
sky skip
vegetation skip
structure keep
```

这些过于粗。更重要的是条件组合：

```text
sky + lowD:
    neutral keep

sky + highD + high uncertainty:
    weak skip or short negative

road/building + lowD + low TTT conflict:
    positive long memory

road/building + high TTT conflict:
    keep as read source but avoid long write

vegetation + highD + high conflict:
    source skip or short negative
```

### 1.5 问题 E：不同数据库/序列的问题是否不同，但不能针对数据集调参？

本轮允许诊断不同数据集或序列的 failure mode，例如：

```text
KITTI01:
    可能以 long-range scale / trajectory drift 为主。

City-like driving sequences:
    sky / road / building 的比例可能很高。

Indoor datasets:
    wall / floor / furniture 可能更重要。
```

但不允许做：

```text
KITTI01 专用阈值
某个 sequence 专用 label table
某个 dataset 专用 gamma
```

所有策略必须使用固定规则，并在不同数据集上只做诊断，不做 per-dataset tuning。

---

## 2. 实验总原则

### 2.1 使用 GT semantic，但不把它写成 deployable success

GT semantic 是 upper-bound diagnostic。所有使用 GT semantic 的结果必须标记：

```text
uses_gt_semantic = true
counts_as_deployable_online_success = false
```

除非目标场景本身就允许实时 GT semantic，这通常不是我们的 deployable setting。

### 2.2 不允许 predicted fallback

为了避免混淆，本轮 GT semantic 实验必须禁止 predicted semantic fallback：

```text
if GT semantic cache missing:
    stop run
    mark as invalid
    do not fallback to Mask2Former / predicted cache
```

### 2.3 先 short-rollout，再 full online

本轮所有 candidate 先跑 trusted causal fork short rollout：

```text
chunks = 6, 10, 16
horizon = h10, h15
```

只有通过 gate 的 candidate 才允许进入 full online validation。

### 2.4 先 single-path，再 pairwise，再 all-memory

不能一开始就 all-memory。顺序必须是：

```text
single path:
    frame / global / SWA / TTT 分开

pairwise:
    只组合过 gate 或接近 gate 的 path

all-memory:
    只在 pairwise 不互相抵消时启动
```

### 2.5 不根据数据集调参

本轮可以用多个数据集做诊断，但参数必须固定：

```text
same semantic taxonomy
same role rules
same D_g/conflict/scale-risk conditions
same gates
```

如果某个数据集失败，只记录 failure mode，不为该数据集单独调阈值。

---

## 3. 核心假设

## H1：如果语义预测噪声是主要瓶颈，GT semantic 应显著强于 predicted semantic

### 假设

v24 失败可能是因为 predicted semantic 边界、label、时间对齐或 coarse fallback 不够准。如果 H1 成立，GT semantic 应该至少在 h10/h15 short rollout 上产生明显更强改善。

### 实验设计

建立三组 semantic 输入：

```text
S_pred:
    v24 predicted/coarse semantic role

S_gt_coarse:
    GT fine labels 映射到 coarse group

S_gt_fine:
    GT fine labels 直接进入 runtime role policy
```

固定 parent：

```text
parent = H9 causal fork snapshots
cue = C23 past
read path = current locked frame pair/all protocol
commit = probe_ttt_write
```

跑以下最小对照：

```text
GT0_NOOP:
    GT semantic loaded but HMC ignores it

GT1_COARSE_FRAME_ONLY
GT2_FINE_FRAME_ONLY

GT3_COARSE_TTT_ONLY
GT4_FINE_TTT_ONLY

GT5_COARSE_SWA_ONLY
GT6_FINE_SWA_ONLY
```

### 必须记录指标

```text
global:
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

semantic:
    per_label_coverage
    per_label_Dg_mean
    per_label_Dg_p90
    per_label_conflict_mean
    per_label_source_keep_ratio
    per_label_source_skip_mass
    per_label_TTT_positive_mass
    per_label_TTT_neutral_mass
    per_label_TTT_negative_mass
    unknown_or_ignore_coverage

implementation:
    GT_cache_hit_rate
    frame_id_alignment_check
    resize_crop_alignment_check
    no_predicted_fallback
    path_consumption_flags
```

### 假设成立标准

H1 通过条件：

```text
GT fine semantic 至少满足以下之一：
    h10 [200,300) delta <= -3m
    或 h15 ATE delta <= -1.5m
    或 h15 [200,300) delta <= -2.5m

并且：
    [400,600) regression <= +1m
    GT no-op parity = exact
```

H1 不成立条件：

```text
GT fine semantic 与 predicted/coarse 语义差异很小；
所有 GT semantic single-path candidate h10 ATE delta > -0.5m；
所有 GT semantic single-path h10 [200,300) delta > -2m。
```

如果 H1 不成立，Codex 应停止 predicted semantic 改进路线，转向 geometry/TTT/scale-state 主线；语义只保留为 visualization / diagnostics。

---

## H2：语义必须 path-specific；同一 semantic role 同时控制所有 memory 会互相抵消

### 假设

frame/global/SWA/TTT 的职责不同，语义 role 也应不同。如果同一规则同时作用于所有 path，会出现：

```text
frame/global source clean 了；
但 SWA local continuity 被破坏；
或者 TTT long memory 被弱化；
最终 h15 回退。
```

### 实验设计

定义 path-specific role：

```text
R_frame:
    控制 frame attention K/V source。

R_global:
    控制 global/chunk attention K/V source。

R_swa:
    控制 SWA previous/current cache source。

R_ttt:
    控制 TTT positive / neutral / negative write。
```

先跑 single-path：

```text
F1_GT_FINE_FRAME_SOURCE
G1_GT_FINE_GLOBAL_SOURCE
S1_GT_FINE_SWA_CACHE
T1_GT_FINE_TTT_WRITE
```

再根据 single-path 结果启动 pairwise：

```text
FG = frame + global
FS = frame + SWA
FT = frame + TTT
GS = global + SWA
GT = global + TTT
ST = SWA + TTT
```

最后才 all-memory：

```text
ALL_ROLE_A:
    best frame + best global + best SWA + best TTT

ALL_ROLE_B:
    same but TTT uses neutral-preserving version

ALL_ROLE_C:
    same but SWA uses conservative cache keep
```

### 角色定义初版

#### Frame/global source role

```text
KEEP_SOURCE:
    road, building, wall, fence, sidewalk
    if D_g low or conflict low

PARTIAL_KEEP:
    sky lowD
    vegetation lowD
    grass lowD

SKIP_SOURCE:
    movable highD
    vegetation highD
    sky highD + high_uncertainty
    unknown highD

PROTECT_SOURCE:
    structure lowD near overlap
```

#### SWA cache role

```text
KEEP_CACHE:
    road, building, wall, fence in overlap
    lowD structure in previous/current cache

PARTIAL_CACHE:
    sky lowD
    vegetation lowD

DROP_OR_WEAKEN_CACHE:
    movable highD
    vegetation highD
    low-trust boundary highD

PROTECT_OVERLAP:
    stable road/building/wall/fence across overlap
```

#### TTT write role

```text
POSITIVE_LONG:
    structure lowD low_conflict high_confidence

NEUTRAL_LONG:
    sky lowD
    vegetation lowD
    road far/background low conflict

NEGATIVE_SHORT:
    movable highD
    vegetation highD high_conflict
    sky highD high_uncertainty
    unknown highD high_conflict

PROTECT_NEUTRAL:
    horizon-like sky
    road boundary if low conflict
    far structure with low D_g
```

### 必须记录指标

除 H1 的通用指标，还要记录 path interaction：

```text
path_effect_table.csv:
    path
    h10_ATE_delta
    h15_ATE_delta
    h10_[200,300]_delta
    h15_[200,300]_delta
    h15_[400,600]_delta
    durability_ratio

path_interaction_table.csv:
    path_a
    path_b
    additive_prediction
    observed_combination_delta
    interaction_gain
    destructive_interference_flag

semantic_path_mass.csv:
    label
    R_frame
    R_global
    R_swa
    R_ttt
    coverage
    source_keep_mass
    cache_keep_mass
    ttt_write_mass
```

### 假设成立标准

H2 成立：

```text
至少一个 path-specific single-path candidate 强于 all-memory coarse v24；
且 pairwise combination 的 observed gain 不低于 best single-path 的 70%；
且没有 [400,600) regression > +1m。
```

如果所有 pairwise 都明显弱于 single-path：

```text
说明 memory paths 在相互抵消；
停止 all-memory；
转向单 path + persistence/handoff。
```

---

## H3：GT 语义只有和 D_g / TTT conflict / scale-state risk 条件化后才有足够因果力

### 假设

语义本身不是 trajectory drift 的直接因果变量。它必须和已有强信号结合：

```text
D_g:
    read/source harmful-support cue

update_conflict_energy:
    TTT-native write conflict cue

scale_state_risk:
    reset-window trajectory/scale risk cue

geometry confidence / uncertainty:
    fallback trust cue
```

### 实验设计

构造四类 GT semantic policy：

```text
P_sem_only:
    only GT semantic label decides role

P_sem_dg:
    semantic role conditioned on D_g

P_sem_conflict:
    semantic role conditioned on update_conflict_energy

P_sem_scale:
    semantic role conditioned on scale_state_risk

P_sem_all:
    semantic role conditioned on D_g + conflict + scale_state_risk
```

候选：

```text
COND_01_FRAME_SEM_ONLY
COND_02_FRAME_SEM_DG
COND_03_FRAME_SEM_DG_CONFLICT

COND_04_TTT_SEM_ONLY
COND_05_TTT_SEM_DG
COND_06_TTT_SEM_DG_CONFLICT
COND_07_TTT_SEM_DG_CONFLICT_SCALE

COND_08_ALLMEM_SEM_DG_CONFLICT_SCALE
```

### 公式定义

定义 semantic fine label prior：

$$
L_i \in \{\text{sky}, \text{vegetation}, \text{road}, \text{building}, \text{wall}, \text{fence}, \text{movable}, \text{unknown}\}
$$

定义 risk：

$$
R_i = \alpha D_i + \beta C^{ttt}_i + \gamma S^{scale}_i + \delta U_i
$$

其中：

```text
D_i:
    C23 D_g

C_i^{ttt}:
    update_conflict_energy normalized

S_i^{scale}:
    no-GT scale-state risk proxy

U_i:
    geometry uncertainty / low confidence
```

role assignment：

$$
R^{role}_i =
f(L_i, D_i, C^{ttt}_i, S^{scale}_i, U_i)
$$

本轮不优化 $\alpha,\beta,\gamma,\delta$，先固定：

```text
D_g primary for frame/global
TTT conflict primary for TTT
scale_state primary for persistence/handoff
semantic label as role prior
```

### 必须记录指标

```text
per_label_risk_distribution.csv:
    label
    D_g_mean/p90
    conflict_mean/p90
    scale_risk_mean/p90
    uncertainty_mean/p90

conditional_role_counts.csv:
    label
    lowD_lowConflict_count
    highD_lowConflict_count
    lowD_highConflict_count
    highD_highConflict_count
    role_assigned

risk_to_error_correlation.csv:
    risk_type
    Spearman_with_h10_ATE_delta
    Spearman_with_h15_ATE_delta
    Spearman_with_[200,300]_delta
```

### 成立标准

H3 成立：

```text
P_sem_dg 或 P_sem_dg_conflict 显著强于 P_sem_only；
至少满足：
    h10 [200,300) delta improvement over sem_only >= 1.5m
    或 h15 ATE delta improvement over sem_only >= 1.0m
```

如果 semantic-only 和 conditional semantic 都弱：

```text
说明 GT semantic 不是当前 memory correction 的主因果变量；
Codex 应停止扩展 semantic matrix，转向 TTT/scale-state action。
```

---

## H4：GT 语义可以解释不同数据集/序列的 failure mode，但不能用于 per-dataset tuning

### 假设

不同数据集的语义结构不同，memory failure 可能不同；但我们需要泛化机制，不需要针对某个数据集打榜。

### 实验设计

使用固定策略候选，只做诊断：

```text
Datasets / sequences:
    KITTI01
    KITTI00
    KITTI02
    KITTI05
    optional indoor sequence with GT semantics if available
    optional city sequence with dense semantic GT if available
```

每个数据集只跑：

```text
GT_NOOP
best single-path semantic policy
best conditional semantic policy
best all-memory semantic policy if gate passed
```

不允许为每个数据集修改：

```text
thresholds
semantic label weights
D_g/conflict quantile
gamma
memory path set
```

### 必须记录指标

```text
dataset_semantic_distribution.csv:
    dataset
    label_coverage
    structure_ratio
    sky_ratio
    vegetation_ratio
    road_ratio
    movable_ratio
    unknown_ratio

dataset_failure_profile.csv:
    dataset
    baseline_ATE
    candidate_ATE_delta
    worst_segment
    segment_semantic_distribution
    scale_drift_proxy
    yaw_drift_proxy

generalization_table.csv:
    candidate
    mean_delta
    median_delta
    worst_regression
    num_sequences_improved
    num_sequences_regressed
```

### 成立标准

H4 成立：

```text
同一固定 policy 在多个序列上改善方向一致；
没有任何序列发生 > 5% ATE regression；
failure profile 能解释某类语义结构和 drift pattern 的关系。
```

如果某个数据集表现不同：

```text
记录为 dataset-specific diagnosis；
不针对该数据集调参；
只在机制层面解释为什么失败。
```

---

## 4. 实验阶段设计

---

# Phase 0：GT Semantic Implementation Hard Gate

## 目标

在所有主实验之前，先保证 GT semantic cache 和 runtime role policy 可信。任何 semantic 实验如果没有通过 Phase 0，结果都不可信。

## 必须实现

### 0.1 GT semantic cache loader

Codex 需要实现或检查：

```text
GTSemanticProvider:
    input:
        sequence id
        frame id / timestamp
        RGB image resolution
        crop/resize metadata
    output:
        fine_label_map[t,H,W]
        coarse_group_map[t,H,W]
        ignore_mask[t,H,W]
        quality/trust map if available
```

禁止：

```text
fallback to predicted masklet cache
fallback to coarse predicted group
silent missing labels
silent frame-id mismatch
```

### 0.2 Label mapping

建立统一映射：

```text
fine labels:
    road
    sidewalk
    building
    wall
    fence
    pole
    traffic sign/light
    vegetation
    terrain
    sky
    person/rider
    car/truck/bus/train/motorcycle/bicycle
    unknown/ignore

coarse groups:
    STRUCTURE_ANCHOR
    LOW_VALUE_STUFF
    MOVABLE_THING
    STATIC_THING
    UNCERTAIN_REGION
    IGNORE
```

注意：coarse group 只能做 summary，runtime policy 必须能访问 fine label。

### 0.3 Token projection

GT labels 必须投影到 LoGeR patch tokens：

```text
L_gt_patch
L_gt_tok
G_gt_tok
ignore_gt_tok
label_confidence_tok
```

对每个 patch token，记录：

```text
majority_label
majority_fraction
top2_label
top2_fraction
label_entropy
ignore_fraction
```

如果 `majority_fraction < 0.55`，该 token 标记为 boundary/mixed，不允许直接 hard role。

## Phase 0 记录指标

```text
GT_cache_hit_rate
GT_frame_id_match_rate
GT_resolution_match
resize_crop_alignment_error
ignore_pixel_ratio
unknown_token_ratio
mixed_token_ratio
per_label_token_count
per_label_pixel_count
no_predicted_fallback_flag
semantic_role_noop_ATE_delta
semantic_role_noop_raw_trans_diff
path_consumption_flags
```

## Phase 0 通过标准

必须全部满足：

```text
GT_cache_hit_rate = 1.0 for selected frames
no_predicted_fallback_flag = true
unknown_token_ratio <= 0.20
mixed_token_ratio <= 0.30
semantic_role_noop_ATE_delta = 0
semantic_role_noop_raw_trans_diff = 0
all path consumption smoke has non-empty debug when enabled
no stale run directory contamination
```

## 若不满足，Codex 自动尝试

```text
if GT_cache_hit_rate < 1:
    修 frame-id / timestamp mapping，不跑 candidate。

if unknown_token_ratio > 0.20:
    输出 label coverage report；
    如果 GT 本身缺失，停止该 dataset；
    不 fallback predicted semantic。

if mixed_token_ratio > 0.30:
    改进 projection:
        use area-weighted majority
        separate boundary tokens
        disallow hard skip on mixed tokens

if no-op parity fails:
    检查 semantic prior 是否被 HMC 消费；
    检查 pass-through 是否改变 write score normalization；
    修复后重新跑 Phase 0。
```

---

# Phase 1：Passive GT Semantic Attribution

## 目标

先不控制模型，观察 GT semantic 与已有强信号的关系。回答：

```text
哪些 semantic labels 落在 D_g high 区？
哪些 labels 的 TTT conflict 高？
哪些 labels 与 [200,300) / [400,600) failure 有关？
GT labels 是否能解释 v20-v22 的 h10 strong / h15 weak？
```

## 实验设计

选择 parent states：

```text
H9 parent
C9 parent
best v22 short signal parent if available
```

chunks：

```text
6, 10, 16
```

horizons：

```text
h10, h15
```

只做 passive logging，不修改 memory。

## 必须记录指标

```text
per_label_attribution.csv:
    label
    coverage
    D_g_mean
    D_g_p90
    update_conflict_mean
    update_conflict_p90
    scale_risk_mean
    scale_risk_p90
    source_attention_mass
    SWA_cache_mass
    TTT_write_mass_native
    TTT_post_zp_delta_norm
    h10_error_contribution_proxy
    h15_error_contribution_proxy

per_segment_semantic.csv:
    segment
    label_distribution
    D_g_distribution
    conflict_distribution
    scale_risk_distribution
    baseline_error

label_path_correlation.csv:
    label
    frame_source_corr_with_error
    global_source_corr_with_error
    swa_cache_corr_with_error
    ttt_write_corr_with_error
```

## 必须可视化

```text
1. RGB + GT fine label overlay
2. RGB + GT coarse group overlay
3. D_g heatmap
4. update_conflict_energy heatmap
5. scale_state_risk heatmap
6. per-label D_g/conflict boxplot
7. per-label TTT update norm bar chart
8. [200,300) segment semantic histogram
9. [400,600) segment semantic histogram
```

## 判断标准

Phase 1 成功不是看 ATE，而是看 attribution 是否清楚：

```text
Pass if:
    per-label coverage 和 memory-path mass 都非空；
    至少能识别 2-3 个 high-risk label-condition groups；
    D_g/conflict/scale-risk 和 semantic label 的关系可解释。

Fail if:
    GT semantic 与所有 risk cue 无明显关系；
    per-label role 几乎不改变 source/write mass；
    fine label runtime projection 实际仍退化成 coarse group。
```

若失败，Codex 自动尝试：

```text
1. 检查 fine label projection 是否真的进入 runtime；
2. 检查 GT label 是否与 RGB frame 对齐；
3. 把 token label 从 majority 改为 soft distribution；
4. 若仍无语义-risk 关系，停止语义主线，保留 semantic visualization。
```

---

# Phase 2：Single-Path GT Semantic Role Screening

## 目标

分别测试 GT semantic 在每条 memory path 上的作用，不允许一开始 all-memory。

## 2.1 Frame/global source path

### 假设

GT semantic 主要能帮助 read/source path：让高风险语义区域不作为 K/V source，但保留 query token。

### 候选

```text
FRAME_GT_01_STRUCTURE_KEEP_LOWSTUFF_PARTIAL
FRAME_GT_02_LOWSTUFF_HIGHD_SKIP
FRAME_GT_03_SKY_NEUTRAL_VEG_HIGHD_SKIP
FRAME_GT_04_MOVABLE_HIGHD_SKIP
FRAME_GT_05_STRUCTURE_PROTECT_HIGHD_SOFT

GLOBAL_GT_01_STRUCTURE_KEEP_LOWSTUFF_PARTIAL
GLOBAL_GT_02_LOWSTUFF_HIGHD_SKIP
GLOBAL_GT_03_SKY_NEUTRAL_VEG_HIGHD_SKIP
GLOBAL_GT_04_MOVABLE_HIGHD_SKIP
GLOBAL_GT_05_STRUCTURE_PROTECT_HIGHD_SOFT
```

### 记录指标

```text
context_source_keep_ratio
context_source_skip_tokens
num_empty_source_events
per_label_source_keep_ratio
per_label_source_skip_mass
attention_mass_to_skipped_before_after
h10/h15 trajectory deltas
```

### 成立标准

```text
h10 [200,300) delta <= -3m
or h15 ATE delta <= -1.5m

and:
    [400,600) regression <= +1m
    empty_source_events = 0
```

## 2.2 SWA semantic cache path

### 假设

GT semantic 在 SWA 中主要影响 adjacent continuity，不应当硬删 sky/road 等可能提供局部尺度的信息。

### 候选

```text
SWA_GT_01_STRUCTURE_OVERLAP_KEEP
SWA_GT_02_MOVABLE_HIGHD_DROP
SWA_GT_03_SKY_PARTIAL_KEEP_VEG_HIGHD_DROP
SWA_GT_04_MIXED_BOUNDARY_NEUTRAL
SWA_GT_05_STRUCTURE_PROTECT_LOWSTUFF_PARTIAL
```

### 记录指标

```text
SWA_cache_label_mass_before
SWA_cache_label_mass_after
SWA_previous_source_keep_ratio
SWA_current_source_keep_ratio
overlap_pointmap_residual
boundary_10f_ATE
boundary_20f_ATE
h10/h15 deltas
```

### 成立标准

```text
boundary_10f_ATE improves
and h15 ATE delta <= -1m
and [400,600) regression <= +1m
```

## 2.3 TTT semantic write path

### 假设

GT semantic 能帮助 TTT 区分长期 positive、长期 neutral、短期 negative，但不能独立决定；必须与 D_g/conflict/scale risk 结合。

### 候选

```text
TTT_GT_01_STRUCTURE_LOWCONFLICT_POSITIVE
TTT_GT_02_LOWSTUFF_LOWCONFLICT_NEUTRAL
TTT_GT_03_LOWSTUFF_HIGHD_CONFLICT_SHORTNEG
TTT_GT_04_SKY_NEUTRAL_NO_NEG
TTT_GT_05_MOVABLE_HIGHD_SHORTNEG
TTT_GT_06_STRUCTURE_POSITIVE_MOVABLE_SHORTNEG
TTT_GT_07_SEM_DG_CONFLICT_SCALE_ALL_CONDITIONAL
```

### 记录指标

```text
per_label_TTT_positive_mass
per_label_TTT_neutral_mass
per_label_TTT_negative_mass
per_label_post_zp_delta_norm
per_label_update_conflict_energy
TTT_write_score_mean_by_label
TTT_memory_state_diff_by_chunk
h10/h15 deltas
```

### 成立标准

```text
h10 ATE delta <= -1.5m
or h15 ATE delta <= -1.5m
or h15 [200,300) delta <= -2.5m

and:
    no [400,600) regression > +1m
```

## Phase 2 若不满足，Codex 自动尝试

```text
if frame/global has h10 signal but h15 weak:
    add skip-aware TTT commit or SWA cache preservation.

if SWA improves boundary but hurts ATE:
    reduce hard drop, keep sky/road partial, protect overlap structure.

if TTT semantic write has no signal:
    inspect per-label TTT positive/negative mass;
    if role mass too small, adjust role assignment, not gamma;
    if role mass large but no trajectory effect, semantic is not causal for TTT.

if all single-path weak:
    stop pairwise/all-memory;
    run deeper attribution instead.
```

---

# Phase 3：Pairwise Semantic Memory Role Combination

## 目标

只组合 Phase 2 中有信号的路径。组合目标不是“全都打开”，而是验证互补或冲突。

## 实验设计

如果 Phase 2 有通过或接近通过的 paths，跑：

```text
PAIR_GT_FRAME_GLOBAL
PAIR_GT_FRAME_SWA
PAIR_GT_FRAME_TTT
PAIR_GT_GLOBAL_TTT
PAIR_GT_SWA_TTT
PAIR_GT_FRAME_SWA_TTT
```

每个 pairwise 都必须保留 path attribution，不能只输出最终 delta。

## 记录指标

```text
pairwise_interaction.csv:
    path_a
    path_b
    single_a_delta
    single_b_delta
    expected_additive_delta
    observed_delta
    interaction_gain
    destructive_flag

path_mass_after_combination.csv:
    label
    path
    role
    mass_single
    mass_combined
```

## 判断标准

```text
Pass:
    observed_delta <= min(single_a_delta, single_b_delta) - 0.5m
    or durability_ratio improves by >= 0.2
    and [400,600) regression <= +1m

Fail:
    combination weaker than both singles
    or destructive_flag true for main segment
```

若失败，Codex 自动尝试：

```text
1. Identify destructive path using pairwise_interaction.csv.
2. Disable the destructive path.
3. Try sequential policy:
       source skip first, TTT write second
   instead of simultaneous control.
4. If all pairs destructive, stop all-memory combination.
```

---

# Phase 4：All-Memory GT Semantic Role Validation

## 目标

只有 Phase 3 证明 paths 互补时才启动。目标是形成真正的 all-memory semantic role policy。

## 候选

```text
ALL_GT_01_BEST_PAIR_PLUS_TTT_NEUTRAL
ALL_GT_02_FRAME_GLOBAL_SWA_WITH_TTT_POSITIVE_ONLY
ALL_GT_03_FRAME_GLOBAL_SKIP_WITH_TTT_CONDITIONAL_NEGATIVE
ALL_GT_04_FULL_ROLE_WITH_SKY_PROTECT_NEUTRAL
ALL_GT_05_FULL_ROLE_WITH_STRUCTURE_LONG_AND_LOWSTUFF_SHORT
```

## 记录指标

```text
all_memory_role_summary.csv
pathwise_mass_summary.csv
state_overwrite_summary.csv
durability_h10_h15.csv
trajectory_segment_deltas.csv
```

## 判断标准

进入 no-GT selector 或 full online diagnostic 的条件：

```text
h10 [200,300) delta <= -5m
or h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m

and:
    durability_ratio >= 0.45
    [400,600) regression <= +1m
    no empty source events
    no semantic path mass collapse
```

如果 all-memory h10 强但 h15 弱：

```text
转 Phase 5 durability attribution。
```

如果 all-memory h10 也弱：

```text
停止 semantic all-memory 主线；
只保留 best single-path diagnostic。
```

---

# Phase 5：Durability Attribution

## 目标

如果 GT semantic 产生 h10 信号但 h15 消失，必须定位 correction 被哪条路径覆盖。

## 实验设计

对 best candidate 保存 h10/h15 endpoint states：

```text
base state
candidate h10 state
candidate h15 state
```

比较：

```text
HMC/TTT state movement
SWA cache semantic mass movement
global source buffer movement
merge/gauge state movement
```

## 记录指标

```text
state_overwrite_ratio.csv:
    path
    base_to_h10_norm
    h10_to_h15_norm
    overwrite_ratio

semantic_mass_decay.csv:
    label
    path
    h10_mass
    h15_mass
    decay_ratio

correction_persistence.csv:
    correction_type
    h10_gain
    h15_gain
    durability_ratio
```

## 判断标准

```text
if TTT overwrite ratio > 0.7:
    correction 被后续 TTT write 洗掉。
    尝试 skip-aware TTT commit / W_long W_short。

if SWA cache decay > 0.7:
    correction 被 SWA local cache refresh 洗掉。
    尝试 semantic SWA cache preserve。

if merge/gauge movement > 0.5:
    correction 与 trajectory merge/gauge state 不一致。
    转 trajectory-state module 或 scale-state conditioned write。

if none:
    检查 evaluation/frame intersection 和 role projection。
```

---

# Phase 6：Cross-Dataset / Cross-Sequence Diagnosis without Tuning

## 目标

诊断不同数据集/序列是否有不同 failure profile，但不做 per-dataset tuning。

## 实验设计

使用固定候选：

```text
GT_NOOP
best_single_path
best_pairwise
best_all_memory_if_any
```

固定参数跑：

```text
KITTI01
KITTI00
KITTI02
KITTI05
optional dataset with dense GT semantic labels
```

## 记录指标

```text
sequence_metrics.csv:
    sequence
    ATE
    Rot
    FinalErr
    [200,300]_or_worst100
    [400,600]_or_worst200
    semantic_distribution
    label-risk_correlation

generalization_summary.csv:
    candidate
    num_sequences_improved
    num_sequences_regressed
    mean_delta
    worst_regression
```

## 判断标准

```text
Pass:
    fixed policy improves at least 2/3 diagnostic sequences
    and no sequence regression > 5%
    and failure profiles are explainable by semantic/geometry conditions.

Fail:
    policy only works on KITTI01
    or requires per-sequence thresholds.
```

若失败：

```text
不调参。
输出 dataset diagnosis。
把语义策略降级为 dataset-specific diagnostic tool。
```

---

## 5. 加速与并行执行策略

## 5.1 并行 tracks

Codex 可并行启动 5 条 track：

```text
Track A:
    GT semantic cache / projection / no-op hard gate

Track B:
    passive semantic attribution

Track C:
    frame/global source single-path

Track D:
    SWA cache single-path

Track E:
    TTT write single-path
```

依赖关系：

```text
Track A must pass before B/C/D/E.

Track C/D/E can run in parallel after Track A.

Phase 3 pairwise depends on Phase 2 pass/near-pass.

Phase 4 all-memory depends on Phase 3 pass.

Phase 5 attribution depends on h10 strong + h15 weak.

Phase 6 cross-seq depends on at least one stable candidate.
```

## 5.2 Resource strategy

```text
short rollout:
    6 GPU parallel allowed if RAM safe.

full KITTI01:
    max 4 GPU parallel.

cross-sequence long full:
    max 2 GPU parallel.

Stage C GT semantic cache:
    precompute offline.
    require-hit mode in all candidate runs.
```

## 5.3 Run invalidation

每个 launcher 必须：

```text
move stale run dir to .INVALID_RERUN_*
write run_config.yaml
write semantic_gt_config.yaml
write path_consumption_summary.jsonl
write no_predicted_fallback flag
```

禁止复用污染目录。

---

## 6. 必须可视化

每个通过 Phase 0 的 candidate 至少输出：

```text
1. RGB + GT fine label overlay
2. RGB + GT coarse group overlay
3. D_g heatmap
4. update_conflict_energy heatmap
5. scale_state_risk heatmap
6. frame/global K/V source keep map
7. SWA cache keep/drop map
8. TTT positive/neutral/negative role map
9. semantic role by path map
10. h10 vs h15 segment error waterfall
11. per-label memory mass bar chart
12. per-path contribution bar chart
13. trajectory overlay:
        H9
        C9
        candidate
        GT
14. segment trajectory view:
        [200,300)
        [400,600)
15. dataset semantic distribution histogram for cross-seq runs
```

---

## 7. Final promotion rules

## 7.1 Diagnostic success

```text
GT semantic explains why predicted/coarse semantic failed;
or identifies one semantic label-condition group that strongly affects memory.
```

## 7.2 Short-rollout success

```text
h10 [200,300) delta <= -5m
or h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m

and:
    durability_ratio >= 0.45
    [400,600) regression <= +1m
```

## 7.3 Full online diagnostic success

Because GT semantic is diagnostic, not deployable:

```text
counts_as_gt_semantic_diagnostic = true
counts_as_deployable_online_success = false
```

Full online GT semantic candidate is meaningful if:

```text
ATE <= 30m
or ATE improves over C9 by >= 3m
or [200,300) improves by >= 10m without [400,600) regression > +2m
```

## 7.4 Target-25 upper-bound success

GT semantic route shows target-level upper bound if:

```text
KITTI01 ATE <= 25m
```

If this happens, next phase should translate GT semantic to predicted / online semantic.

## 7.5 Semantic route failure

Semantic route is considered insufficient if all are true:

```text
GT fine semantic single-path weak;
GT fine semantic conditional policies weak;
pairwise/all-memory destructive or weak;
no h15 durable candidate;
no full GT diagnostic improvement > 3m.
```

Then:

```text
Semantic Prior Generator remains diagnostic / auxiliary;
Target-25 mainline shifts back to trajectory-state / scale-state online module.
```

---

## 8. Codex failure-action table

| Failure condition | Codex should try next |
|---|---|
| GT cache missing | Fix GTSemanticProvider; do not fallback predicted |
| GT labels misaligned | Fix frame-id / resize / crop projection |
| Fine labels unavailable at runtime | Implement fine label runtime projection before running semantic candidates |
| No-op parity fails | Inspect pass-through and write-score normalization |
| Single-path weak | Run passive attribution; refine label-condition groups |
| Frame/global h10 strong but h15 weak | Add TTT/SWA persistence, not threshold sweep |
| SWA improves boundary but hurts ATE | Reduce hard drop; protect overlap structure; preserve sky lowD |
| TTT semantic mass too small | Adjust role assignment, not gamma |
| TTT semantic mass large but no trajectory effect | Semantic not causative for TTT; stop TTT semantic matrix |
| Pairwise destructive | Disable destructive path; test sequential source-then-write policy |
| All-memory weak | Stop all-memory; keep best single path |
| GT semantic strong but predicted weak | Diagnose semantic predictor; build predictor improvement plan |
| GT semantic weak | Stop semantic mainline; return to TTT/scale-state action |
| Cross-sequence regression | Do not tune per dataset; record failure profile |

---

## 9. 本轮预期结论格式

Codex 每批结束后必须用下面格式汇报：

```text
1. What was tested?
2. Did GT semantic actually enter runtime path?
3. Did it improve h10?
4. Did it persist to h15?
5. Which labels changed memory mass?
6. Which memory path was responsible?
7. Did [400,600) regress?
8. Does this justify pairwise/all-memory?
9. If not, what exact branch should run next?
```

---

## 10. 一句话总结

本轮不是继续验证 predicted semantic 是否有小收益，而是用 GT semantic 做上界实验：

> **如果 GT semantic 都不能产生 durable all-memory correction，那问题就不是语义预测质量，而是语义本身不是 Target-25 的主因果变量；如果 GT semantic 有强信号，我们再回头解决 predicted semantic 如何逼近 GT。**

# ACL2 v40：Training-Free Quality-Gated Semantic-Geometry Memory Controller Target30 实验计划

日期：2026-05-25  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / VideoMasklet / frame-global attention / SWA / TTT  
目标：把语义从“全局写入先验”改造成 **质量下降时才介入的 training-free 诊断与记忆控制器**，并验证它是否能把当前可部署 best `C9_P0_R2 = 33.7629421029m` 推进到 `ATE <= 30m`。

---

## 0. 项目边界与禁止事项

本项目的目标不是训练一个新模块，也不是在 KITTI01 上学习一个触发器。当前路线必须保持 **training-free**：所有运行时策略只能使用 LoGeR 内部 cue、VideoMasklet 输出、几何一致性、appearance anomaly、HMC/SWA/TTT memory trace 和无 GT self-consistency 指标。

本轮明确禁止：

```text
禁止训练 trigger / selector / classifier / role router
禁止用 oracle label 拟合规则
禁止用 absolute chunk id 或固定 segment 作为 deployable 条件
禁止针对 KITTI01 或任何单一数据集调 threshold / gamma / label table
禁止使用 GT semantic 作为 runtime action
禁止把 short rollout / fixed-window / diagnostic row 写成 deployable success
禁止把 [200,300) 当作 SWA 或全局策略设计目标
```

`[200,300)` 只作为 KITTI01 当前已知 stress window，用于诊断和复盘；真正的主 gate 必须是 scene-agnostic 的 rolling-window、all-boundary、overlap consistency、trajectory-state health 和 full-online ATE。

---

## 1. 本轮为什么要改方向

v39 已经证明 semantic / appearance action 能到达模型路径：Track 0 完成 `150/150`，hook reachability、action distinguishability 和 influence nontriviality 都通过，最大 influence mass 为 `0.2376947701`，最大 skipped-source influence mass 为 `0.1286811680`。同时，RGB/Lab audit 确实发现 sky appearance variation，例如 `sky_lab_delta_p90 = 2.3016548456`，但由于没有落盘 per-label spatial source-attention / SWA / TTT contribution maps，所以 sky causality 仍然不能成立。

后续 Track 1-4 都没有过 gate。Frame/global source surgery 几乎无效，best ATE delta 只有约 `-0.0695m`；SWA 有局部信号，best ATE delta 约 `-0.7081m`、best rolling100 delta 约 `-1.3354m`、best `[200,300)` delta 约 `-2.39m`；TTT lifecycle 很弱，best ATE delta 约 `-0.1086m`；semantic C23 / appearance anomaly 的局部 `[200,300)` 改善接近 `-4.97m`，但 ATE、rolling100 和 downstream `[400,600)` gate 都失败。没有 h15，没有 full online，没有 Target-30。

因此，本轮不再继续问：

```text
sky 要不要 skip？
vegetation 要不要 skip？
dynamic 要不要 negative？
structure 要不要 positive？
```

而是改问：

```text
当前 chunk / adjacent chunk 是否出现无 GT 的重建质量下降信号？
如果质量没有下降，语义是否应该保持 passive？
如果质量下降，语义和几何/appearance/memory trace 能否定位原因？
定位后应该操作 frame/global、SWA、TTT，还是 TTT commit/reset？
这个操作是否能改善 rolling-window 和 full online，而不是只改善一个固定 stress segment？
```

---

## 2. 总体目标

本轮目标是建立一个 **Quality-Gated Semantic-Geometry Memory Controller**。

它的运行逻辑是：

```text
默认：
    沿用当前可部署 best C9 / H9 协议，不主动启用 semantic memory action。

Step 1: Chunk health monitor
    用无 GT 几何、appearance、attention、SWA、TTT 指标判断当前 chunk 或 adjacent chunk 是否可能出现质量下降。

Step 2: Semantic-geometry diagnosis
    如果 health gate 触发，再用 semantic label、masklet trust、appearance anomaly、D_g、TTT conflict、scale-state proxy 和 source influence 定位异常来源。

Step 3: Failure-mode routing
    根据 failure mode 选择只操作对应 memory path：
        frame/global read source
        SWA local cache / overlap source
        TTT write / commit
        selected TTT soft rollback diagnostic

Step 4: Training-free action
    所有 action 都是 deterministic rule / robust statistic / Pareto guard，不训练任何 trigger 或 selector。

Step 5: Promotion
    只有 h10/h15 同时健康，并且 rolling-window / all-boundary / downstream 不恶化，才允许 full online。
```

本轮阶段性目标：

```text
Stage success:
    full online ATE <= 32m

Target success:
    full online ATE <= 30m

Diagnostic success:
    明确指出 quality drop 是 read source、SWA cache、TTT write，还是 global trajectory-state / merge-gauge 问题。
```

---

## 3. 核心假设

### H1：语义不是全局 prior，而是质量下降时的诊断过滤器

如果当前 chunk health 正常，semantic action 很容易误伤正常 source / memory。语义应该默认 passive，只有当无 GT health metrics 显示质量下降风险时才介入。

判断标准：

```text
如果 quality-gated semantic policy 比 always-on semantic policy 更稳，
并且在 rolling-window / boundary / downstream 上更少 regression，
H1 成立。
```

### H2：天空颜色变化、植被闪烁、阴影和动态物体只有在高影响力时才是因果风险

appearance anomaly 本身不是 action。只有当它与 `D_g`、source attention mass、SWA cache mass 或 TTT update contribution 重叠时，才应该触发 memory action。

判断标准：

```text
如果 sky/vegetation/shadow anomaly 高但 influence mass 低，action 无效或有害；
如果 anomaly + high influence + high D_g 同时成立，source surgery 或 filtered write 有效，
H2 成立。
```

### H3：SWA 的目标是 local continuity，不是修固定 segment

SWA 是 adjacent-chunk local memory。SWA action 必须用 all-boundary、overlap consistency 和 rolling-window health 评价，而不能以 `[200,300)` 为主目标。

判断标准：

```text
SWA candidate 只有在以下条件下晋级：
    rolling window p90/worst 改善；
    boundary_10f / boundary_20f mean,p90,worst 不回退；
    overlap residual 不回退；
    downstream regression 不超过阈值。
```

### H4：TTT 语义操作应是生命周期控制，而不是语义 scalar

TTT 是 compressed global memory。语义进入 TTT 时只能作为 lifecycle routing 的上下文：stable structure 可能 positive long，高风险 dynamic/vegetation/shadow 可能 no-long-write 或 short negative，sky/low-risk background 多数是 neutral。

判断标准：

```text
如果 TTT positive/negative semantic scalar 弱，但 lifecycle routing 能提升 h15/full，H4 成立。
如果所有 TTT semantic lifecycle 仍弱，语义从 TTT 主控制降级为 trust/context，回到 TTT-native cue。
```

### H5：TTT reset / rollback 只能是 emergency diagnostic，不是语义规则

只有当 read/SWA/TTT/scale 多个 health metrics 同时异常，并且 no-commit probe 显示 commit 会恶化 self-consistency，才允许尝试 filtered commit 或 soft rollback。Hard reset 只作 diagnostic，不作默认 deployable strategy。

判断标准：

```text
soft rollback 必须改善 rolling-window 与 h15，且不破坏 downstream；
hard reset 即使改善局部，也不能算 deployable success。
```

---

## 4. Chunk Health Monitor：无 GT 质量下降诊断

每个 chunk 和 adjacent transition 都要计算一个 health vector：

$$
H_m = \{H^{read}_m, H^{swa}_m, H^{ttt}_m, H^{geom}_m, H^{app}_m\}
$$

### 4.1 Read health

记录：

```text
D_g mean / p90 / high-mass
semantic-conditioned D_g z-score mass
source attention mass to high-D regions
source attention mass to dynamic / vegetation / sky / shadow / lowtrust regions
static anchor source mass
appearance-anomaly weighted source mass
context empty source events
```

触发候选：

```text
read_contamination_flag =
    highD_source_mass is high
    and high_influence_semantic_anomaly_mass is high
    and static_anchor_source_mass drops
```

### 4.2 SWA health

记录：

```text
boundary_10f mean / p90 / worst proxy
boundary_20f mean / p90 / worst proxy
chunk_boundary_pose_jump
SWA overlap source keep ratio
SWA non-overlap source keep ratio
SWA K/V source mass by semantic group
SWA cache update norm
SWA source topology change
```

触发候选：

```text
swa_contamination_flag =
    boundary_proxy abnormal
    or overlap_residual abnormal
    or risky semantic source mass dominates SWA cache
```

### 4.3 TTT health

记录：

```text
update_conflict_energy mean / p90
post-zp delta norm by layer / branch
TTT apply mismatch
static anchor positive write mass
risky semantic write mass
scale-state proxy
native-vs-controlled commit delta norm
```

触发候选：

```text
ttt_write_risk_flag =
    update_conflict_energy spike
    or post-zp delta spike
    or scale-state proxy spike
    or static_anchor_write_mass collapse
```

### 4.4 Geometry / point-cloud health

记录：

```text
pointmap confidence mean / p10
world/local point residual between neighboring support frames
overlap pointmap residual
valid point coverage
projection / depth-order inconsistency proxy
Sim3 scale-step proxy
```

### 4.5 Appearance / semantic health

记录：

```text
Lab delta mean / p90 by semantic label
RGB histogram drift by masklet
feature drift by masklet
masklet temporal IoU
masklet label flip rate
masklet fragmentation
VideoMasklet trust Q_mask
SemanticKITTI sparse projection support/agreement when available, diagnostic only
```

Important：

```text
appearance anomaly 只作为风险证据；
如果没有 high influence mass，不触发 memory action。
```

---

## 5. Failure Mode Taxonomy 与对应 action

### Mode A：LOCAL_READ_CONTAMINATION

条件：

```text
read health abnormal
source attention mass on risky semantic/appearance regions high
SWA/TTT health not severely abnormal
```

Action：

```text
frame/global:
    query 保留
    risky K/V source soft attenuation or compact K/V skip
    stable structure source rescue

SWA:
    no action or only protect anchors

TTT:
    no action
```

判断：

```text
如果 read-only source surgery 能改善 rolling100 / h10，但 TTT/SWA 不动时不伤 downstream，Mode A 成立。
```

### Mode B：LOCAL_SWA_CONTAMINATION

条件：

```text
SWA boundary / overlap health abnormal
risky semantic source mass high in SWA cache
read health may or may not be abnormal
TTT health not severely abnormal
```

Action：

```text
SWA:
    non-overlap risky source skip
    overlap K preserve / V attenuation
    structure overlap anchor protect
    avoid hard remove on overlap unless boundary guard passes

frame/global:
    optional weak attenuation

TTT:
    no action unless write risk also high
```

主评价不是固定 segment，而是：

```text
boundary_10f / boundary_20f mean,p90,worst
overlap residual
rolling_50f / 100f / 200f p90,worst
downstream regression
```

### Mode C：TTT_WRITE_RISK

条件：

```text
TTT update_conflict or scale-state proxy abnormal
risky semantic regions contribute to TTT update
static anchor write mass collapses
```

Action：

```text
TTT:
    structure lowD lowConflict lowScaleRisk -> positive long
    dynamic/vegetation/shadow highD highConflict -> no-long-write or short negative
    sky high anomaly -> neutral/no-positive-long, not strong negative
    filtered commit if commit health guard fails
```

### Mode D：GLOBAL_STATE_FAILURE / EMERGENCY_COMMIT_CONTROL

条件：

```text
read, SWA, TTT, scale-state health all abnormal
paired no-commit probe predicts commit deterioration
static anchor mass collapses
```

Action ladder：

```text
Level 0: no action
Level 1: read source attenuation
Level 2: no-long-write / filtered commit
Level 3: soft rollback selected TTT branch/layer
Level 4: hard reset diagnostic only
```

Soft rollback formula：

$$
W_{m+1}^{safe} = (1-\rho) W_{m+1}^{candidate} + \rho W_m^{last\_good}
$$

where $\rho$ is fixed by robust health severity, not learned from data.

### Mode E：APPEARANCE_ONLY_NONCAUSAL

条件：

```text
appearance anomaly high
but source influence / SWA mass / TTT contribution low
```

Action：

```text
no memory action
only log visualization
```

---

## 6. 实验设计

### Phase 0：Health instrumentation 与 no-op gate

目标：确保 health metrics 本身不会扰动 pipeline，并且所有指标都能落盘。

Run：

```text
P0_00_C9_REFERENCE
P0_01_HEALTH_LOGGING_ONLY
P0_02_SEMANTIC_PASSIVE_ONLY
P0_03_APPEARANCE_AUDIT_ONLY
```

必须记录：

```text
ATE / Rot / RPE / FinalErr
raw pose max diff vs C9 reference
health jsonl rows count
source influence rows count
SWA health rows count
TTT health rows count
appearance/masklet rows count
context_empty_source_events
```

通过标准：

```text
raw_pose_max_diff = 0 or <= numerical tolerance
ATE_delta_vs_C9 = 0 within exact no-op tolerance
all required health streams non-empty
no context_empty_source_events
```

失败分流：

```text
如果 no-op 漂移：Codex 优先修 logging side effect。
如果某 health stream 缺失：只修 instrumentation，不启动 Phase 1。
如果 per-label/per-masklet influence 缺失：Phase 1 可跑 aggregate，但必须标记 explainability_limited。
```

---

### Phase 1：Full passive Health Atlas

目标：不做任何 memory action，先在 C9/H9 full or trusted snapshots 上建立全序列 health timeline，找出质量下降风险是否能由 no-GT health metrics 表达。

输入：

```text
parents = H9, C9
chunks = all available chunks if full passive possible; otherwise reset starts + chunks 6/10/16/adjacent chunks
mode = logging only
```

输出文件：

```text
health_atlas/chunk_health_timeline.csv
health_atlas/read_health_by_chunk.csv
health_atlas/swa_health_by_boundary.csv
health_atlas/ttt_health_by_chunk.csv
health_atlas/geometry_health_by_chunk.csv
health_atlas/appearance_health_by_semantic.csv
health_atlas/memory_path_influence_by_semantic.csv
health_atlas/health_flag_summary.json
```

必须可视化：

```text
1. chunk health timeline heatmap
2. read/SWA/TTT health separate timelines
3. semantic appearance anomaly overlay for top health-drop chunks
4. source attention mass by semantic group
5. SWA overlap/non-overlap source mass by semantic group
6. TTT update norm by semantic group and branch/layer
7. rolling-window ATE diagnostic overlay, only for offline analysis
```

判断标准：

```text
健康指标必须能区分 high-risk chunks / transitions：
    top health flags should include known stress regions as diagnostic,
    but must not depend on fixed chunk id or fixed segment.

如果 health flags are uniformly high:
    thresholds / normalization are useless; Codex must switch to reset-block robust normalization.

如果 health flags never fire:
    add influence-weighted anomaly metrics, not lower thresholds blindly.
```

---

### Phase 2：Quality-gated single-path interventions

目标：只在 health gate 触发的 chunks / adjacent transitions 上执行 memory action；不再 all-chunk semantic action。

General rule：

```text
if no health flag:
    use baseline C9 memory policy
else:
    route by failure mode
```

#### 2A：Frame/global read source intervention

Candidates：

```text
READ_A1_HIGH_INFLUENCE_ANOMALY_V_ATTEN
READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT
READ_A3_DYNAMIC_VEG_SHADOW_HIGHD_SKIP_STRUCT_RESCUE
READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH
READ_A5_STATIC_ANCHOR_RESCUE_ONLY
```

记录指标：

```text
source_keep_ratio
source_attention_mass_removed_before/after
static_anchor_attention_mass_before/after
D_g distribution before/after
rolling_50f/100f/200f delta
h10/h15 ATE delta
```

通过标准：

```text
h10 or h15 ATE delta <= -1.5m
or rolling100 p90/worst improves >= 2m
and downstream regression <= +1m
and no static_anchor_mass collapse
```

失败分流：

```text
如果 removed attention mass < 0.03：改为 high-influence source selection。
如果 static anchor mass drops：加 structure rescue。
如果 read action有效但 downstream坏：转 residual/attenuation，不做 hard compact。
```

#### 2B：SWA local-continuity intervention

Candidates：

```text
SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP
SWA_B2_OVERLAP_K_PRESERVE_V_ATTEN_RISKY
SWA_B3_STRUCTURE_OVERLAP_ANCHOR_PROTECT
SWA_B4_SKY_HORIZON_NEUTRAL_K_KEEP_V_ATTEN_IF_ANOMALOUS
SWA_B5_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_OVERLAP_PROTECT
```

主指标：

```text
boundary_10f_mean/p90/worst
boundary_20f_mean/p90/worst
overlap_pointmap_residual_mean/p90
chunk_boundary_pose_jump
rolling_50f/100f/200f mean/p90/worst
SWA K/V source mass by semantic group
overlap vs non-overlap keep ratio
```

通过标准：

```text
boundary_10f_p90_delta <= +0.25m
boundary_20f_p90_delta <= +0.25m
overlap_residual_delta <= +0.25m
and rolling100 p90/worst improves >= 2m
or h15 ATE delta <= -1.5m
```

失败分流：

```text
如果局部 stress window 改善但 boundary 回退：
    改成 K preserve / V attenuation 或 non-overlap-only。
如果 overlap anchor 被误删：
    加 structure/road/wall/fence anchor protect。
如果所有 SWA action h10 有效但 h15 衰减：
    做 SWA cache persistence / source refresh attribution，而不是继续调语义阈值。
```

#### 2C：TTT write / commit intervention

Candidates：

```text
TTT_C1_STRUCTURE_LOW_RISK_POSITIVE_LONG
TTT_C2_DYNAMIC_HIGH_RISK_NO_LONG_WRITE
TTT_C3_VEGETATION_HIGH_D_CONFLICT_SHORT_NEGATIVE
TTT_C4_SKY_ANOMALY_NO_POSITIVE_LONG_NEUTRAL
TTT_C5_COMBINED_LIFECYCLE
TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL
```

记录指标：

```text
TTT update_conflict_energy before/after
post-zp delta norm by layer/branch
positive/neutral/short-negative/no-long-write mass
static anchor write mass
risky semantic write mass
scale-state proxy
native-vs-controlled commit delta
h10/h15 ATE and rolling-window delta
```

通过标准：

```text
h15 ATE delta <= -1.5m
or rolling100 p90/worst improves >= 2m
and [400,600)-style downstream regression <= +1m
and post-zp delta norm not spiking above robust p95
```

失败分流：

```text
如果 TTT semantic action仍弱：语义从 TTT 主控制降级为 trust/context，转 TTT-native update_conflict/scale-state。
如果 short negative有效但 h15衰减：加 filtered commit 或 positive long anchor。
如果 filtered commit伤 downstream：保留 read/SWA action，不动 TTT commit。
```

#### 2D：Emergency TTT commit / soft rollback diagnostic

只在 severe global-state health flag 下运行。

Candidates：

```text
RESET_D1_FILTERED_COMMIT_BRANCH0_ONLY
RESET_D2_SOFT_ROLLBACK_W0_LOW_RHO
RESET_D3_SOFT_ROLLBACK_SELECTED_LAYERS
RESET_D4_HARD_RESET_DIAGNOSTIC_ONLY
```

通过标准：

```text
soft rollback candidate must improve h15 ATE <= -2m
and rolling100/200 p90 not regress
and full online later must not rely on fixed chunk id.
```

Hard reset 只作机制诊断，不计入 deployable success。

---

### Phase 3：Failure-mode-specific minimal combination

目标：只组合 Phase 2 中作用路径不同且互不冲突的 action。

允许组合：

```text
READ contamination + TTT write risk:
    read source attenuation + filtered/no-long-write TTT

SWA local contamination + static anchor:
    SWA overlap anchor protect + TTT structure positive long

Appearance anomaly + read contamination:
    appearance-weighted read source attenuation + static rescue
```

禁止组合：

```text
两个 action 都大幅修改同一 SWA overlap source
两个 action 都强改 TTT branch0 commit
semantic-conditioned C23 replacement + TTT write risk without isolation
```

记录：

```text
component action masks
component deltas
combined delta
interaction term:
    I = Delta_combined - Delta_A - Delta_B
```

判断：

```text
如果 I << 0 且 combined better，说明互补。
如果 I > 0 或 combined 回退，说明冲突，回到 path isolation。
```

---

### Phase 4：Full online validation

只有满足以下条件才启动 full online：

```text
1. candidate 不依赖 fixed chunk id / fixed segment；
2. h15 ATE delta <= -1.5m 或 rolling100 p90/worst 改善 >= 2m；
3. all-boundary health 不回退；
4. downstream rolling-window regression <= +1m；
5. 同一 rule 在 H9 和 C9 parent 上方向一致，或明确是 C9-compatible。
```

Full online runs：

```text
FULL_01_C9_BASE_REPEAT
FULL_02_QUALITY_GATED_READ_ONLY
FULL_03_QUALITY_GATED_SWA_ONLY
FULL_04_QUALITY_GATED_TTT_ONLY
FULL_05_MINIMAL_COMBINED
```

Full online 成功标准：

```text
stage success:
    ATE <= 32m

Target-30 success:
    ATE <= 30m

safety:
    rolling100 p90/worst not worse than C9
    boundary_10f/20f p90 not worse than C9
    no catastrophic [400,600)-style downstream regression
```

---

### Phase 5：Cross-sequence diagnostic, not tuning

目标：诊断规则是否过拟合 KITTI01。

Datasets / sequences：

```text
KITTI00 / KITTI02 / KITTI05 if available
or other long driving sequences with same pipeline inputs
```

规则：

```text
不允许改 threshold / label table / gamma。
只记录 failure mode。
```

指标：

```text
health flag frequency
failure mode distribution
semantic anomaly distribution
rolling-window p90/worst delta
full ATE if full runs are affordable
```

判断：

```text
如果规则只在 KITTI01 chunk-specific stress window 有效：降级为 diagnostic。
如果多序列都能减少 high-risk rolling-window / boundary regression：保留为 general memory policy。
```

---

## 7. Codex 并行执行安排

### Codex A：Health instrumentation and atlas

负责：

```text
Phase 0 no-op
Phase 1 passive health atlas
per-label/per-masklet influence map landing
appearance anomaly overlay
```

如果失败：

```text
先修 logging / instrumentation；不启动 action matrix。
```

### Codex B：Frame/global read source action

负责：

```text
Phase 2A candidates
source attention mass before/after
static rescue audit
```

如果失败：

```text
如果 source mass removed 低，转 high-influence selection。
如果 static anchor 被删，加入 rescue。
如果 compact_kv 太强，改 V-only attenuation。
```

### Codex C：SWA local-continuity action

负责：

```text
Phase 2B candidates
all-boundary / overlap / rolling-window metrics
K preserve / V attenuation variants
```

如果失败：

```text
boundary 回退 -> non-overlap-only 或 V attenuation。
全部弱 -> SWA semantic 降级为 diagnostic。
```

### Codex D：TTT lifecycle / commit action

负责：

```text
Phase 2C / 2D
post-zp / update_conflict / scale-state / filtered commit
```

如果失败：

```text
semantic weak -> TTT-native control。
rollback hurts downstream -> only read/SWA action.
```

### Codex E：Combination and full online

负责：

```text
Phase 3 interaction term
Phase 4 full online launch only when gate passes
```

如果失败：

```text
combination conflict -> path isolation。
full online fails despite h15 pass -> washout attribution.
```

---

## 8. 必须落盘的 CSV / JSONL / PT 文件

```text
health_atlas/chunk_health_timeline.csv
health_atlas/read_health_by_chunk.csv
health_atlas/swa_health_by_boundary.csv
health_atlas/ttt_health_by_chunk.csv
health_atlas/geometry_health_by_chunk.csv
health_atlas/appearance_health_by_semantic.csv
health_atlas/memory_path_influence_by_semantic.csv

rollouts/*/source_attention_mass_before_after.csv
rollouts/*/swa_boundary_health.csv
rollouts/*/overlap_residual_summary.csv
rollouts/*/ttt_update_health.csv
rollouts/*/post_zp_delta_by_layer_branch.csv
rollouts/*/semantic_action_mask_summary.csv
rollouts/*/failure_mode_decision.json
rollouts/*/candidate_vs_parent_delta.csv

final_reports/health_timeline_report.md
final_reports/path_action_report.md
final_reports/durability_report.md
final_reports/full_online_report.md
final_reports/failure_routing_summary.md
```

---

## 9. 必须可视化

每个 top health-drop chunk / transition：

```text
1. RGB strip
2. VideoMasklet semantic overlay
3. appearance anomaly heatmap
4. D_g heatmap
5. source attention mass heatmap
6. SWA overlap/non-overlap source map
7. TTT update_conflict / post-zp delta map
8. static anchor mass map
9. selected action mask map
10. before/after source keep map
```

全序列：

```text
1. chunk health timeline heatmap
2. failure mode timeline
3. boundary_10f/20f mean,p90,worst timeline
4. rolling 50f/100f/200f mean,p90,worst curves
5. h10/h15 durability waterfall
6. full trajectory overlay for C9 vs candidate if full launched
```

---

## 10. 最终决策规则

### 10.1 保留语义主线的条件

```text
至少一个 quality-gated semantic-geometry policy 满足：
    h15 ATE delta <= -1.5m
    or rolling100 p90/worst improves >= 2m
并且 full online 不比 C9 差。
```

### 10.2 进入 Target-30 主线的条件

```text
full online ATE <= 32m:
    进入下一轮 refinement。

full online ATE <= 30m:
    Target-30 achieved。
```

### 10.3 降级语义的条件

如果以下同时成立：

```text
1. health instrumentation pass；
2. high-influence semantic/appearance regions identified；
3. frame/global/SWA/TTT quality-gated actions 均无 h15 或 rolling-window 上界；
4. C9-compatible full rows 仍 >33m；
```

则结论为：

```text
Semantic Prior Generator 在当前 VideoMasklet + training-free rule 形式下不是 Target-30 主杠杆；
语义保留为 diagnostics / trust calibration / visualization；
主线转回 TTT-native trajectory-state / scale-state / merge-gauge controller。
```

---

## 11. 一句话总结

v40 的核心不是继续试“天空跳过”或“植被少写”，而是：

> **默认不动 memory；先用无 GT health metrics 判断当前 chunk / adjacent chunk 是否真的质量下降；如果下降，再用语义、appearance、几何和 memory trace 定位是哪条 memory path 被污染；最后只对那条 path 做 training-free、可审计、scene-agnostic 的最小 intervention。**


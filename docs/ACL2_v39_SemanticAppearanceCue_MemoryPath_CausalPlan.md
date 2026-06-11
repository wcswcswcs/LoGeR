# ACL2 v39：Training-Free Semantic + Appearance Anomaly Memory Control 实验计划

日期：2026-05-24  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / frame attention / global attention / SWA / TTT  
主目标：在 **training-free** 边界内，重新验证“天空、植被、阴影、动态物体等语义/外观异常区域会干扰重建”这一假设，并把语义信息真正转化为可部署的 memory control，而不是继续做低效语义规则矩阵。  
当前 deployable best：`C9_P0_R2 = 33.7629421029m`  
阶段目标：先进入 `KITTI01 ATE <= 30m`，同时保证策略不是 KITTI01 固定窗口调参。  

---

## 0. 项目边界：防止再次走偏

本项目是 **training-free memory control**，不是训练 trigger、selector、classifier 或 learned router。v39 所有候选必须满足以下硬边界：

```text
禁止训练 trigger / selector / classifier / role router。
禁止使用 oracle label 拟合规则。
禁止使用 absolute chunk id 作为策略条件。
禁止针对 KITTI01 或任何单一数据集调参。
禁止把 short rollout / fixed-window / diagnostic 写成 deployable success。
禁止把 SemanticKITTI sparse projection 当作 runtime dense GT semantic。
禁止把 [200,300) 作为策略目标；它只能作为 KITTI01 的一个 stress diagnostic。
```

v39 允许使用：

```text
LoGeR 内部 cue：D_g / C23 / TTT conflict / scale-state proxy / attention mass。
VideoMasklet 前端输出：fine label、coarse group、mask quality、temporal consistency。
SemanticKITTI sparse projection：只作为 offline trust calibration / audit，不作为 runtime GT action。
Runtime no-GT self-consistency：overlap residual、boundary proxy、attention/source mass、TTT update health、SWA cache health。
```

v39 不再问：

```text
sky 要不要 skip？
vegetation 要不要 negative？
road 要不要 positive？
```

v39 改问：

```text
某个语义区域在当前 cue 条件下是否是高影响力风险源？
它在 frame/global/SWA/TTT 哪条 memory path 中造成影响？
应当 source skip、SWA protect、TTT no-long-write、short negative，还是 neutral context？
这种修正能否从 h10 持久到 h15/full online？
```

---

## 1. 对 v38 结果的重新理解

v38 的真正结论不是“语义无用”，也不是“天空/植被/动态区域干扰假设失败”。v38 说明：

```text
Track 0 action/influence audit 通过，说明 action path 可达。
Track 1 frame/global source surgery h10 失败，说明当前 source surgery 没打中强读路径。
Track 2 SWA 有局部信号，但没过 h10 gate，也没有 h15 continuation。
Track 3 TTT h10 弱，说明当前 TTT semantic action 仍不是主力。
Track 4 semantic C23 residual 有局部改善，但远低于 continuation gate。
Track 5 full online 未启动。
```

v38 中最值得注意的信号不是 overall ATE，而是：

```text
Track2 best [200,300) delta ≈ -2.39m / -2.36m。
Track4 best [200,300) delta ≈ -1.28m / -1.30m。
```

这说明局部病灶确实对 semantic / source / memory intervention 有反应，但反应幅度不足、没有持久性，也没有转化为 full online。用户观察到 `[200,300)` 附近天空颜色有显著变化，这很可能是一个重要线索，但它必须被验证，而不是直接写成规则。v39 将它定义为 **Semantic-Appearance Anomaly Hypothesis**。

---

## 2. 核心假设

### H1：天空/植被/阴影/动态物体不是天然“坏语义”，只有在外观异常或几何异常时才是高风险源

#### 假设

同一个语义类别在不同条件下作用不同：

```text
sky + low anomaly + low D_g：可能是 horizon / scale context，应 neutral keep。
sky + strong color/exposure change + high D_g：可能污染 frame/global source，应 weak source skip。
vegetation + low D_g：可能是远景 continuity，应 neutral keep。
vegetation + high D_g + high TTT conflict：可能是 risk source，应 source skip 或 short negative。
road/building/wall/fence + low D_g + low conflict：可能是 static scale anchor，应 keep / positive long。
road/building + high conflict：可能 read 可用，但不应 long TTT write。
```

因此，不应该做纯语义规则，而应该做：

$$
R_{path}(i)=f(sem_i, A_i^{app}, D_i, C_i^{ttt}, S_i^{scale}, Q_i, path)
$$

其中 $A_i^{app}$ 是 semantic-conditioned appearance anomaly，$D_i$ 是 C23/D_g，$C_i^{ttt}$ 是 TTT update conflict，$S_i^{scale}$ 是 scale-state risk，$Q_i$ 是 masklet trust。

#### 验证标准

H1 成立需要同时满足：

```text
1. 在高误差 rolling windows 中，sky/vegetation/shadow/dynamic 等区域的 appearance anomaly 或 D_g/conflict 显著高于同类语义的稳定窗口。
2. 这些区域在 frame/global/SWA/TTT 至少一条 path 中有非零 high influence：source attention mass、SWA cache mass 或 TTT update contribution。
3. 对这些 high-risk 区域做 path-specific intervention 后，h10/h15 rolling-window error 改善，并且 all-boundary/overlap health 不显著回退。
```

如果天空颜色变化明显，但 sky token 的 source attention mass / SWA mass / TTT update contribution 很低，则天空变化只是视觉相关，不是当前 LoGeR error 的主因。

---

### H2：VGGT4D-style source skip 在 LoGeR 中应优先作用于 frame/global K/V source，而不是直接作用于 TTT write

#### 假设

VGGT4D 的启发不是“用 mask 调一个 score”，而是让风险区域不再作为 early inference 的 source context。LoGeR 中最接近的路径是：

```text
query token 保留；
K/V source token 根据 semantic + anomaly + D_g + influence 被 skip 或 attenuate；
特殊 token / stable structure anchor 始终保护；
先验证 frame/global，再谨慎进入 SWA 和 TTT。
```

#### 验证标准

H2 成立需要：

```text
frame/global K/V source surgery 在 h10 或 h15 上达到：
    ATE delta <= -1.5m
    或 rolling100 worst/p90 delta <= -3m
    或 stress-window delta <= -5m
同时：
    source_empty_events = 0
    static_anchor_attention_mass 不下降超过 10%
    all-boundary p90 不回退超过 +0.25m
```

如果 frame/global source skip 只改变 source count，但被 skip token 的原始 attention mass 很低，则该策略不是主杠杆，Codex 应自动转向 high-influence selection，而不是继续扫语义类别。

---

### H3：SWA 的语义策略必须以 local continuity 为目标，而不是以某个固定区间为目标

#### 假设

SWA 是 local / lossless memory。SWA 语义控制的核心不是修 `[200,300)`，而是改善所有 chunk boundary 的 local alignment。任何只改善一个局部窗口、但破坏 boundary continuity 的策略都不能部署。

#### 验证标准

SWA 候选必须同时满足：

```text
all-boundary boundary_10f_mean/p90/worst 不显著回退；
boundary_20f_mean/p90/worst 不显著回退；
overlap_pointmap_residual 不显著回退；
rolling50/100/200 p90 或 worst 改善；
downstream max regression <= +1m；
不依赖固定 chunk id 或固定 dataset segment。
```

如果 SWA 改善某个 stress window 但 boundary 回退，则候选只能作为 diagnostic，Codex 应改为：

```text
K preserve / V attenuation；
non-overlap-only skip；
structure overlap anchor protect；
soft attenuation 而不是 hard remove。
```

---

### H4：TTT 语义写入不应再用 scalar prior，而应区分 static long anchor 与 risky short negative

#### 假设

TTT 是 compressed global memory，主要与长期 coordinate / scale 有关。语义如果要进入 TTT，应该做 lifecycle routing：

```text
stable structure + lowD + low conflict + low scale risk -> positive long write；
dynamic / vegetation / shadow / lowtrust + highD + high conflict -> short negative or no-long-write；
sky / far background + low risk -> neutral context，不做 positive long，也不做 strong negative。
```

#### 验证标准

TTT 候选必须达到：

```text
h10 ATE delta <= -1.5m 或 h10 rolling100 delta <= -3m；
h15 durability ratio >= 0.45；
h15 ATE delta <= -2m 或 h15 rolling100 delta <= -3m；
[400,600) 或 downstream rolling windows 回退 <= +1m；
post-zp update norm 不异常，branch/layer update 没有爆炸。
```

如果 TTT semantic action 仍弱，Codex 应将语义从 TTT 主控制降级为 trust/context，只保留 TTT-native cue 和 scale-state cue。

---

### H5：语义最有可能通过“重解释 cue”产生作用，而不是直接替代 cue

#### 假设

`D_g` 的意义依赖语义上下文。相同的高 `D_g` 对 sky、road、building、vegetation、dynamic thing 含义不同。因此应构造 semantic-conditioned cue：

$$
z_D(i)=\frac{D_g(i)-\mu_{sem(i)}(D_g)}{\sigma_{sem(i)}(D_g)+\epsilon}
$$

同理可构造：

$$
z_C(i)=\frac{C_{ttt}(i)-\mu_{sem(i)}(C_{ttt})}{\sigma_{sem(i)}(C_{ttt})+\epsilon}
$$

$$
z_A(i)=\frac{A_{app}(i)-\mu_{sem(i)}(A_{app})}{\sigma_{sem(i)}(A_{app})+\epsilon}
$$

最终只做 residual injection：

$$
D_{final}=D_{base}+\lambda(D_{semrisk}-D_{base})
$$

不再全量替换 `D_g`。

#### 验证标准

semantic-conditioned cue 成立需要：

```text
read-only / no-TTT / no-SWA isolation 中出现 h10/h15 稳定改善；
C9-compatible path 下不回退；
full online 不低于 C9；
若 full online 未过，必须能解释 washout path。
```

---

## 3. Phase 0：Semantic-Appearance Influence Atlas

### 目标

不先跑新策略，而是先解释用户观察的天空颜色变化、植被/阴影/动态区域干扰是否真进入 LoGeR 的高影响力路径。

### 必须计算的图和表

对所有 chunks，尤其是 rolling100 top-worst windows、largest-gain/largest-regression windows，落盘：

```text
per_frame_rgb_luma_stats.csv
per_semantic_label_lab_delta.csv
per_masklet_lab_delta.csv
per_masklet_temporal_iou.csv
per_masklet_label_stability.csv
per_semantic_Dg_stats.csv
per_semantic_ttt_conflict_stats.csv
per_semantic_scale_risk_stats.csv
per_semantic_source_attention_mass.csv
per_semantic_swa_cache_mass.csv
per_semantic_ttt_update_contribution.csv
semantic_appearance_influence_atlas.csv
```

Appearance anomaly 推荐用 Lab 颜色空间和 feature drift 双线计算：

$$
A_{Lab}(m,t)=\|\mu_{Lab}(m,t)-\operatorname{EMA}_{past}(\mu_{Lab}(m,t))\|_2
$$

$$
A_{semZ}(i)=\frac{A_i-\operatorname{median}_{j:sem(j)=sem(i)}A_j}{\operatorname{MAD}_{j:sem(j)=sem(i)}A_j+\epsilon}
$$

### 必须可视化

每个 stress window 至少输出：

```text
RGB frame strip
semantic mask overlay
sky/vegetation/shadow/dynamic masks
D_g heatmap
semantic-conditioned appearance anomaly heatmap
source attention mass heatmap
SWA overlap/non-overlap source mass map
TTT update contribution map
combined risk map
trajectory error over time
```

### 判断标准

如果天空颜色变化是真因，需要看到：

```text
sky/sky-boundary masklet 在 stress window 中 appearance anomaly 显著上升；
sky 区域同时有非零或较高 source attention / SWA mass / TTT contribution；
sky high-anomaly 区域和 D_g / C23 residual / scale-risk 有交集；
后续 source surgery 对这些高影响力 sky tokens 有效。
```

如果只看到颜色变化，但 influence mass 低，则不要围绕 sky 继续调参。

---

## 4. Phase 1：Group-level VGGT4D-style frame/global source surgery

### 目标

验证风险语义区域作为 K/V source 是否真的污染 read。此阶段不碰 TTT 写入，不碰 SWA cache。

### 候选组

```text
FG_01_DYNAMIC_HIGHD_SKIP:
    dynamic thing + highD/high appearance anomaly -> K/V skip

FG_02_VEGETATION_HIGHD_SKIP:
    vegetation/tree/grass + highD/high conflict -> K/V skip

FG_03_SKY_APPANOM_WEAK_SKIP:
    sky + high appearance anomaly + highD -> soft source attenuation, not hard skip

FG_04_LOWTRUST_APPANOM_SKIP:
    low mask trust + high appearance anomaly + highD -> K/V skip

FG_05_RISK_SKIP_STRUCTURE_RESCUE:
    skip dynamic/vegetation/lowtrust high-risk
    protect road/building/wall/fence lowD lowConflict anchors

FG_06_SHADOW_PROXY_SKIP:
    low-light / shadow-like appearance anomaly + highD -> source attenuation
```

其中 shadow 没有可靠 semantic label 时，不用假标签，而用 appearance proxy：局部亮度突变、低饱和/暗区变化、高 uncertainty、高 D_g 交集。

### 记录指标

```text
source_keep_ratio_by_semantic
source_attention_mass_removed_before/after
static_anchor_attention_mass_before/after
attention_mass_to_highD_before/after
context_empty_source_events
per-layer read effect
ATE_h10/h15
rolling50/100/200 mean/p90/worst
all-boundary 10f/20f mean/p90/worst
downstream max regression
```

### 成立标准

进入 h15 的条件：

```text
h10 ATE delta <= -1.5m
或 h10 rolling100 p90/worst delta <= -3m
或 stress-window delta <= -5m
且 boundary p90 regression <= +0.25m
且 static_anchor_attention_mass 下降 <= 10%
```

如果失败方向明确：

```text
skip token attention mass 低 -> 改为 high-influence selection；
sky skip 伤 boundary/scale -> sky 改为 neutral horizon context；
vegetation skip 无效 -> 只对 vegetation highD+highConflict 生效；
structure protect 太弱 -> 加强 static anchor rescue；
```

---

## 5. Phase 2：SWA local-continuity semantic policy

### 目标

SWA 不以 `[200,300)` 为主目标，而以 scene-agnostic local continuity 为主目标。该阶段验证语义/外观风险是否能改善 local memory 而不伤 chunk boundary。

### 候选

```text
SWA_01_NONOVERLAP_DYNAMIC_REMOVE:
    dynamic/highD only in non-overlap source removed；overlap protected。

SWA_02_OVERLAP_K_KEEP_V_ATTEN_DYNAMIC:
    dynamic/highD in overlap keeps K but attenuates V。

SWA_03_STRUCTURE_OVERLAP_PROTECT:
    road/building/wall/fence lowD lowConflict protected as overlap anchors。

SWA_04_SKY_HORIZON_NEUTRAL:
    sky low anomaly kept; sky high anomaly V-atten only, no K remove。

SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP:
    vegetation highD highConflict skip only outside overlap。

SWA_06_COMBINED_LOCAL_TOPOLOGY:
    non-overlap risky skip + overlap structure protect + K preserve/V atten for uncertain areas。
```

### 记录指标

```text
boundary_10f_mean/p90/worst
boundary_20f_mean/p90/worst
chunk_boundary_pose_jump_mean/p90
overlap_pointmap_residual_mean/p90
SWA source mass by semantic group
SWA K_keep / V_keep / V_attenuation ratio
overlap vs non-overlap keep ratio
rolling50/100/200 mean/p90/worst
h10/h15 ATE and segment deltas
```

### 成立标准

SWA 候选通过必须满足：

```text
all-boundary 10f/20f p90 regression <= +0.25m；
overlap residual 不回退；
rolling100 p90/worst 改善；
downstream max regression <= +1m；
h15 durability ratio >= 0.45。
```

如果局部窗口改善但 boundary 回退，Codex 自动尝试：

```text
hard remove -> K keep + V atten；
overlap remove -> non-overlap-only remove；
semantic skip -> source attention high-risk only；
add structure overlap protect。
```

---

## 6. Phase 3：TTT static scale-anchor + risky short-negative

### 目标

TTT 不再使用语义 scalar。TTT 只做两类事：

```text
1. 稳定结构作为 long positive anchor；
2. 高风险语义/外观异常作为 short negative 或 no-long-write。
```

### 候选

```text
TTT_01_STRUCTURE_LONG_ANCHOR:
    road/building/wall/fence + lowD + lowConflict + lowScaleRisk + highTrust -> positive long。

TTT_02_DYNAMIC_NO_LONG_WRITE:
    dynamic thing + highD/highConflict -> no-long-write。

TTT_03_VEG_SHORT_NEGATIVE:
    vegetation/tree/grass + highD + highConflict -> short negative；lowD vegetation neutral。

TTT_04_SKY_NEUTRAL_NO_LONG:
    sky never positive long；high anomaly sky no-long-write; low anomaly sky neutral。

TTT_05_COMBINED_LIFECYCLE:
    structure long anchor + dynamic/veg high-risk short negative + sky neutral。

TTT_06_SHADOW_LOWTRUST_NO_LONG:
    shadow/low-light proxy + low trust + highD -> no-long-write。
```

### 记录指标

```text
TTT role mass by semantic label
positive/negative contribution norm
post-zp delta norm by branch/layer
update_conflict_energy before/after
scale-risk before/after
h10/h15 ATE and rolling metrics
[400,600) and downstream regression
TTT branch w0/w1/w2 update norm
memory_state_rel_diff
```

### 成立标准

```text
h10 ATE delta <= -1.5m 或 rolling100 delta <= -3m；
h15 durability ratio >= 0.45；
h15 ATE delta <= -2m 或 h15 rolling100 delta <= -3m；
[400,600) regression <= +1m；
post-zp update norm 不异常。
```

如果失败：

```text
TTT semantic long anchor 无效 -> 语义不再主控 TTT，只保留 trust/context；
short negative 伤 continuity -> 改为 no-long-write，不做 negative；
w0 有信号 w1/w2 伤 -> 限制 branch=w0；
h10 有效 h15 弱 -> 做 tensor-state washout attribution。
```

---

## 7. Phase 4：Semantic-conditioned C23 residual path isolation

### 目标

验证语义是否应主要用于重新解释 C23/D_g，而不是直接做 source/write action。

### 候选

```text
C23R_01_READ_ONLY_RESID:
    D_final = D_base + lambda(D_semrisk - D_base)，只作用 read。

C23R_02_NO_TTT:
    semantic residual 不进入 TTT write。

C23R_03_NO_SWA:
    semantic residual 不控制 SWA。

C23R_04_FRAMEGLOBAL_ONLY:
    semantic residual 只控制 frame/global read source。

C23R_05_APPANOM_SEM_Z:
    semantic z + appearance anomaly z + D_g residual。

C23R_06_STATIC_RESCUE:
    semantic residual + structure lowD protection。
```

推荐 residual 能量归一：

$$
D_{final}=D_{base}+\lambda(D_{semrisk}-D_{base})
$$

$$
\lambda = \min\left(1,\frac{\rho \cdot RMS(D_{base})}{RMS(D_{semrisk}-D_{base})+\epsilon}\right)
$$

其中 $\rho$ 是固定设计常数，不从数据集训练。

### 记录指标

```text
D_base / D_semrisk / D_final map statistics
semantic z map by label
appearance anomaly z by label
read attention mass shift
frame/global source keep ratio
SWA/TTT unaffected confirmation for isolation rows
h10/h15 ATE and rolling metrics
full online only after h15 pass
```

### 成立标准

```text
read-only / no-TTT / no-SWA isolation 中至少一条 h15 过 gate；
C9-compatible path 不回退；
full online 若启动，必须优于 C9 至少 1m 才继续该族。
```

---

## 8. Phase 5：组合与 full online 最小验证

### 启动条件

只有以下情况才允许 full online：

```text
某候选 h15 ATE delta <= -2m；
或 h15 rolling100 delta <= -3m；
或 h15 stress-window delta <= -5m 且 boundary/downstream 健康；
且 h15 durability_ratio >= 0.45。
```

### 最小组合策略

最多组合 2 个 path，避免互相抵消：

```text
COMBO_01 = frame/global source surgery + SWA local protect
COMBO_02 = frame/global source surgery + TTT static anchor
COMBO_03 = C23 residual read-only + SWA local protect
COMBO_04 = C23 residual read-only + TTT static anchor
```

### Full online 成功标准

```text
stage success:
    KITTI01 full ATE <= 32m

Target-30 success:
    KITTI01 full ATE <= 30m

minimum continuation:
    improve C9 by >= 1m
    and rolling100 p90/worst improves
    and no downstream max regression > +1m
```

如果 full online 不过，但 h15 很强：

```text
做 washout attribution；
定位是 TTT tail、SWA refresh、global source、merge/gauge 哪个覆盖 correction；
不继续调语义阈值。
```

---

## 9. 并行执行安排

### Codex A：Influence / Appearance Atlas

负责：

```text
生成 semantic-appearance influence atlas。
输出 RGB/semantic/D_g/appearance/source-attention/SWA/TTT overlays。
判断天空颜色变化是否是真因。
```

失败分流：

```text
如果 sky anomaly 高但 influence mass 低：停止 sky 主线。
如果 vegetation/dynamic/road-boundary influence 更高：转向对应语义组。
```

### Codex B：Frame/Global Source Surgery

负责 Phase 1 候选。  
并行跑 H9/C9 parent chunks `6,10,16` 的 h10，过 gate 才 h15。

失败分流：

```text
attention mass removed 低 -> high-influence selection；
structure attention 被误删 -> static rescue；
hard skip 回退 -> soft attenuation。
```

### Codex C：SWA Local Continuity

负责 Phase 2 候选。  
必须输出 boundary/overlap metrics，不得只看 segment ATE。

失败分流：

```text
boundary 回退 -> K preserve/V atten；
overlap 回退 -> non-overlap-only skip；
local 改善但 h15 衰减 -> SWA cache persistence / washout attribution。
```

### Codex D：TTT Lifecycle

负责 Phase 3 候选。  
只做 static long anchor / risky short negative，不做 semantic scalar。

失败分流：

```text
TTT 弱 -> 降级为 trust/context；
short negative 伤 continuity -> 改 no-long-write；
branch 非 w0 伤 -> 限 branch=w0。
```

### Codex E：Semantic-conditioned C23 Residual

负责 Phase 4 候选。

失败分流：

```text
read-only 无信号 -> 停止 C23 semantic residual；
no-TTT 有效但 hybrid 无效 -> TTT write 冲突，semantic 只用于 read；
no-SWA 有效但 full 无效 -> SWA source 冲突，semantic 不控 SWA。
```

### Codex F：Aggregation / Full Online Gate

负责所有 report、gate、可视化、full online 最小组合。

失败分流：

```text
没有 h15-qualified candidate -> 不启动 full online；
full online 弱 -> washout attribution；
所有语义 path 都弱 -> semantic 降级，Target-30 转 trajectory-state / scale-state。
```

---

## 10. 必须记录与可视化

### 10.1 统一 CSV/JSONL

每个候选必须记录：

```text
run_config.yaml
semantic_policy_summary.json
semantic_appearance_influence_atlas.csv
source_attention_mass_removed.csv
swa_boundary_metrics.csv
ttt_update_contribution.csv
rolling_window_metrics.csv
segment_metrics.csv
hmc_state_hash.jsonl
memory_state_delta_summary.csv
```

### 10.2 必须可视化

```text
RGB frame strip
semantic overlay
sky/vegetation/shadow/dynamic masks
D_g heatmap
appearance anomaly heatmap
source attention heatmap
SWA overlap/non-overlap source map
TTT contribution map
risk/action overlay
rolling ATE curve
boundary error curve
h10->h15 durability curve
```

固定窗口：

```text
全序列 rolling100 top-5 worst windows；
all-boundary top-5 regression windows；
chunk6/10/16 diagnostic windows；
不只固定 [200,300)。
```

---

## 11. 最终停止规则

如果以下条件同时满足：

```text
1. action/influence audit 通过；
2. frame/global source surgery 无 h15-qualified candidate；
3. SWA local policy 无 boundary-healthy h15 candidate；
4. TTT static/negative lifecycle 无 h15 candidate；
5. semantic-conditioned C23 residual 无 C9-compatible h15/full candidate；
```

则结论应更新为：

```text
当前 VideoMasklet/SemanticPrior 路线不是进入 ATE 30 的主杠杆。
语义保留为 diagnostic / trust calibration / weak regularizer。
Target-30 主线转向 explicit online trajectory-state / scale-state / gauge-aware module。
```

如果任一候选 full online 达到：

```text
ATE <= 32m
```

则进入 cross-sequence diagnostic，但不允许针对数据集调参。

如果 full online 达到：

```text
ATE <= 30m
```

则阶段成功。

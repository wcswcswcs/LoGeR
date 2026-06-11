# ACL2 v37：Training-Free Semantic Influence 与 Memory Surgery Target-30 实验计划

日期：2026-05-24  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / Video Masklet / ACL2 attention cue  
主目标：在 **training-free** 前提下，用语义信息增强重建质量，优先推动 KITTI01 online deployable ATE 进入 $30m$。  
当前可计数 deployable online best：`C9_P0_R2 = 33.7629421029m`。  
阶段目标：先达到 $ATE \le 32m$，再冲击 $ATE \le 30m$。  
硬边界：本计划不训练任何 trigger、selector、router、classifier，不使用 GT runtime action，不针对 KITTI01 或任何数据集调参。

---

## 0. 项目目标与硬约束

本项目的目标不是在 KITTI01 上用监督信号训练一个策略，也不是通过 fixed chunk、GT oracle 或后处理把 ATE 刷低。目标是构造一个 **training-free memory control policy**：在 LoGeR 推理时，仅使用模型内部 cue、Video Masklet 输出、稀疏 SemanticKITTI 投影锚点、无 GT 自一致性指标和 HMC/TTT/SWA memory trace，决定 frame attention、global attention、SWA 和 TTT 中哪些 token 应该作为 source、local anchor、long memory 或 short negative evidence。

本轮必须严格遵守以下边界：

```text
禁止训练 trigger / selector / role router / classifier。
禁止用 oracle label 拟合规则。
禁止使用 absolute chunk id 作为 runtime 条件。
禁止针对 KITTI01 或某个数据集专门调 semantic threshold / beta / gamma。
禁止把 short rollout、fixed-window diagnostic、GT projection audit、offline trajectory rewrite 写成 deployable success。
允许诊断不同数据集或序列的 failure mode，但固定同一套规则，不做 per-dataset tuning。
```

这轮实验不再继续做“语义规则矩阵”式小修小补。之前的失败已经说明：`sky skip / vegetation skip / structure keep / lowstuff highD negative` 这类 label-only rule 太弱，无法解释 trajectory drift。新的核心问题是：

> **语义风险区域是否真的以高能量进入 LoGeR 的 frame/global/SWA/TTT 计算路径；如果进入了，它们应该被作为 source、local anchor、long write、short negative 还是 no-long-write？**

换句话说，Semantic Prior Generator 不应再被看成“语义分数生成器”，而应被看成 **training-free semantic influence router**。它要做的是在每条 memory path 中，根据 semantic label、masklet trust、$D_g$、TTT conflict、scale risk 和实际 attention/write energy，决定 deterministic memory action。

---

## 1. v36B 结果的独立判断

v36B 修正了 v36 的过度阻塞：缺少 attention mass 日志不再阻止 short rollout。H0C action distinguishability smoke 通过，说明 source skip / SWA gate / TTT action 已经能进入真实 hook；H0B 也完成了代表性 attention-mass instrumentation repair，记录到 skipped source 在 skip 前平均 attention mass 约 $0.0549$，skip 后为 $0$。这说明 **至少在代表 row 中，被移除的 source token 不是完全无 attention 的死 token**。

但 v36B 仍然没有产生可部署结果：H1 frame/global source-skip 的 18 条 h10 R2 都没有过 gate；H2 SWA h10 曾过 short gate，但 h15 失败，并且 washout attribution 只显示 proxy 级证据；H3 TTT semantic static/negative 24/24 完成但未过 gate；H4 semantic C23 path isolation 在 beta 5.25 修复后 h10/h15 通过 local diagnostic，但复用 v32 full online 和新增 path-isolation full rows 都失败，best path-isolation full row `RESID_NO_SWA` 反而达到 `38.2740m`，远差于 C9。最终没有任何 v36B row 进入 Target-30，当前 best 仍是 `C9_P0_R2 = 33.7629m`。

我的结论是：

```text
1. v36B 证明 hook 大体可达，不应再用“hook 可能完全没生效”解释所有失败。
2. v36B 也证明当前 semantic action 仍太粗，无法形成 full online trajectory correction。
3. v36B 的最大问题不是缺少某个解释性字段，而是缺少“semantic influence 是否足够高、action 是否作用于真正高影响 token”的判定。
4. 不能继续对所有 sky/vegetation/structure 做统一规则；必须改成 influence-ranked semantic memory surgery。
```

因此 v37 的核心不是继续扫 semantic threshold，而是先建立 **Semantic Influence Atlas**，找出每条 memory path 里真正有影响力的 semantic-risk source/write energy，再做有限、可解释、training-free 的 memory surgery。

---

## 2. 总体假设

本轮围绕六个假设推进。每个假设都必须被实验直接验证，而不是通过间接 ATE 猜测。

### H1：前几轮 semantic policy 失败，可能是因为它们没有作用到高影响 token

同一个 semantic mask 可能覆盖很多 token，但这些 token 未必被 attention 读取、未必进入 SWA 有效 cache、未必贡献 TTT fast-weight update。如果 semantic skip 主要删掉低 attention mass token，轨迹自然不动。因此本轮第一件事是构造每个 semantic group / masklet / token 在各 memory path 的真实 influence 分布。

定义某个 path $p$、semantic group $g$ 的 influence mass：

$$
I_{p,g}=\frac{\sum_{i \in g} M_{p,i}}{\sum_i M_{p,i}+\epsilon}
$$

其中 $M_{p,i}$ 不是语义面积，而是 path-specific 影响量：

```text
frame/global:
    source attention mass 或 sampled qk-softmax source mass
SWA:
    previous/current source cache mass、overlap source mass、boundary source mass
TTT:
    pre-zp contribution norm、post-zp delta attribution、update_conflict_energy mass
```

H1 成立的标准不是 ATE，而是：动态物体、植被、阴影/低置信、high-D lowstuff、low-trust masklets 等风险区域在至少一条 memory path 上有非平凡 influence mass，例如 $I_{p,g} \ge 0.05$；否则这一类语义区域即使看起来大，也不是当前模型真正使用的 source/write。

### H2：VGGT4D-style source skip 应优先作用在 frame/global K/V source，而不是先作用 TTT 写入

VGGT4D 的关键启发是 training-free 地从 global attention 中挖 motion cues，并在 early-stage inference 中抑制动态 image tokens，使动态区域不再作为上下文 source 传播。LoGeR 不同于 VGGT，它有 frame/global attention、SWA local memory 和 TTT compressed memory；因此不能照抄，但 source-token filtering 的原则应优先在 frame/global K/V source 上验证。

H2 不是问“sky 要不要跳过”，而是问：

```text
high-influence dynamic / vegetation / lowtrust / shadow-like risk source
是否应该从 frame/global K/V source 中移除或衰减，
同时保护 low-D structure anchors？
```

### H3：SWA 的语义控制必须以 local continuity 为核心，不能照搬 frame/global 或 TTT

SWA 是相邻 chunk 的 local memory。前几轮已经多次出现 `[200,300)` 改善但 boundary_10f / boundary_20f 回退的现象。SWA 语义策略如果 hard remove overlap source，很可能破坏 adjacent alignment。因此 SWA 的动作应区分 non-overlap 与 overlap：

```text
non-overlap high-risk source:
    可以 remove 或 strong attenuation。

overlap source:
    优先 keep K，attenuate V；或保护 stable structure / boundary anchor。
```

H3 成立的标准是同时满足 segment improvement 和 boundary health，不允许只看 `[200,300)`。

### H4：TTT 中语义的主要价值不是 scalar prior，而是区分 static long anchor 与 short negative

TTT 是 compressed global memory，负责长期 coordinate / scale。语义不能简单成为 `semantic_value * write_score`。更合理的是：

```text
stable structure + low D_g + low conflict + low scale risk:
    positive long write。

movable / vegetation / lowtrust + high D_g + high conflict:
    short negative 或 no-long-write。

sky / low-D vegetation / far background:
    neutral context，不做 positive long，也不做 strong negative。
```

H4 的关键在于 long/short lifecycle，而不是更强 negative gamma。

### H5：semantic-conditioned C23 的局部信号是真实的，但它不能直接全序列常开

v31-v34 已经显示 semantic-conditioned C23 在 chunk10 h10/h15 有强 local diagnostic signal，但 full online 和 C9 组合失败。这个方向不应被丢掉，但必须改成 conservative residual 和 path isolation，而不是全量替换 $D_g$ 或训练 trigger。

本轮只允许 training-free residual：

$$
D_{final}=D_{base}+\lambda(D_{sem}-D_{base})
$$

其中 $\lambda$ 是固定 conservative energy ratio，不由 KITTI 拟合：

$$
\lambda = \min\left(1,\frac{\rho \cdot \operatorname{RMS}(D_{base})}{\operatorname{RMS}(D_{sem}-D_{base})+\epsilon}\right)
$$

默认 $\rho=0.25$，只作为固定设计常数。

### H6：进入 ATE 30 的可能路径不是“删坏区域”单线，而是“删高影响风险 source + 保护分布式 static scale anchors + 防止 SWA boundary 破坏 + TTT long/short 分离”

当前 deployable best 是 $33.7629m$，要进 $30m$ 还需要约 $3.7629m$。单个 masklet intervention 或单一路径微调很难达到这个量级。因此本轮必须并行验证三个机制是否互补：

```text
1. frame/global source surgery 减少动态/植被/阴影/低信任 high-influence source 污染；
2. SWA local continuity policy 保留 overlap anchors，避免 boundary 回退；
3. TTT static-anchor positive long write + risky short negative 提供长期 trajectory support。
```

---

## 3. 实验结构总览

v37 分为六个并行 track，所有 track 都遵守同一个 speed-gated 机制：先做 hook/action/influence audit，再做 h10/h15 short rollout，最后只有真正强候选才进入 full online。

```text
Track 0: Semantic Influence Atlas 与 action realism audit
Track 1: VGGT4D-style frame/global source surgery
Track 2: SWA local-continuity semantic source policy
Track 3: TTT static-anchor / short-negative write policy
Track 4: Semantic-conditioned C23 residual path-isolation
Track 5: Minimal full-online combination and failure attribution
```

Track 之间可以并行，但不能绕过必要边界。注意：缺少解释性日志不应阻止所有 track；只有 safety/hook gate 失败才阻止对应 path。

---

## 4. Track 0：Semantic Influence Atlas 与 action realism audit

Track 0 的目标是防止再次出现“策略名字不同，实际 action 相同”或“skip 了无用 token”的情况。它不是性能实验，而是所有后续结果的解释基础。

### 4.1 输入与固定范围

固定 parent 和候选窗口：

```text
parents: H9, C9
chunks: 6, 10, 16
horizons for audit: h3 or h5 smoke only
semantic source: VideoMasklet cache + optional SemanticKITTI sparse projection trust
base cue: C23 past, C9 protocol as deployable reference
```

### 4.2 必须落盘的 action tensors

Codex 必须为每个 policy 保存以下张量或 summary，不允许只写 role count：

```text
frame_source_keep_mask
frame_source_skip_mask
global_source_keep_mask
global_source_skip_mask
swa_nonoverlap_keep_mask
swa_overlap_keep_mask
swa_k_keep_mask
swa_v_keep_mask
ttt_positive_long_mask
ttt_neutral_mask
ttt_short_negative_mask
ttt_no_long_write_mask
```

同时保存：

```text
per_label_action_mass.csv
per_masklet_action_mass.csv
policy_jaccard_matrix.csv
action_keep_ratio_by_path.csv
protected_token_count.csv
context_empty_source_events.csv
```

### 4.3 必须落盘的 influence fields

Track 0 要记录的不只是 token count，而是真正影响计算的量：

```text
frame/global:
    attention_mass_removed_before
    attention_mass_removed_after
    retained_source_attention_mass
    skipped_source_attention_mass
    source_token_count_before/after

SWA:
    previous_source_mass_removed
    overlap_source_mass_removed
    boundary_source_mass_removed
    K_keep_ratio
    V_keep_ratio
    cache_update_norm_by_semantic_group

TTT:
    pre_zp_contribution_norm_by_role
    post_zp_delta_norm_by_role
    branch0/1/2_update_norm_by_role
    update_conflict_energy_by_role
    no_long_write_mass
```

如果某个字段暂时不能完整实现，Codex 不能让全局停止；应将该字段标记为 `explainability_missing`，并允许对应 path 继续做 minimal rollout，只是结论标注为 `source-count supported, attention-mass not proven`。

### 4.4 判断标准

Track 0 通过不是看 ATE，而是看 action 真实性：

```text
H0A hook reachability:
    frame/global source_skip_tokens > 0
    context_empty_source_events = 0
    SWA source gate applied > 0 for SWA policy
    TTT post-zp/action-delta changed for TTT policy

H0B action distinguishability:
    at least one intended semantic-risk policy pair has Jaccard <= 0.85
    or keep_ratio difference >= 0.05
    or TTT role mass difference >= 0.05

H0C influence nontriviality:
    skipped or modified semantic-risk tokens have path influence mass >= 0.03 for smoke
    and at least one candidate group has influence mass >= 0.05
```

如果 H0A 失败，对应 path 停止并修 hook。  
如果 H0B 失败，停止 rollout，Codex 修 role projection / protected token / fallback。  
如果 H0C 失败，不停止所有实验，但后续候选必须改用 high-influence selection，而不是 semantic label selection。

### 4.5 可视化

每个 audit chunk 输出：

```text
RGB frame
semantic fine label overlay
masklet trust overlay
D_g map
TTT conflict map
scale-risk map
frame/global source attention mass map
SWA overlap/source mass map
TTT update norm map
actual skipped source overlay
actual positive_long / short_negative TTT overlay
```

还要输出两类图：

```text
semantic_group x memory_path influence heatmap
semantic_group x action_role mass heatmap
```

---

## 5. Track 1：VGGT4D-style frame/global source surgery

Track 1 直接验证“天空、植被、阴影、动态物体等风险区域作为 K/V source 会干扰重建”的假设，但不是 label-only skip，而是 high-influence semantic-risk source surgery。

### 5.1 设计原则

```text
Query tokens 保留，不改变输出 token shape。
只控制 K/V source。
优先 frame/global read path，不先进入 TTT write。
动态/植被/低信任/阴影-like high-risk source 进行 skip 或 attenuation。
stable structure anchors 需要保护，不能被 high-D rule 误删。
```

其中 shadow 不一定有显式 semantic label，可用以下 proxy 定义：

```text
low mask trust
high D_g
high local appearance inconsistency
low semantic confidence
high TTT conflict but non-thing label
```

### 5.2 候选设计

所有候选先在 H9 parent 和 C9 parent 的 chunk10 h10/h15 上做 short rollout。若 C9 parent 上完全无效，则不进入 full online。

```text
FG_01_DYNAMIC_HIGHD_SKIP:
    MOVABLE_THING + highD + high influence source -> compact K/V source skip

FG_02_VEGETATION_HIGHD_SKIP:
    vegetation/grass/tree + highD + high influence source -> compact K/V source skip

FG_03_LOWTRUST_RISK_SKIP:
    low trust masklets + highD or high conflict -> compact K/V source skip

FG_04_RISK_SKIP_STRUCTURE_RESCUE:
    risk skip from FG_01/02/03
    plus structure lowD lowConflict source protect

FG_05_DYNAMIC_VEG_LOWTRUST_UNION:
    union of dynamic highD, vegetation highD, lowtrust highD
    structure lowD protected

FG_06_SOURCE_ATTENTION_TOP_RISK:
    select top semantic-risk groups by source attention mass, not by area
```

### 5.3 记录指标

```text
ATE / Rot / RPE_t / RPE_r
[200,300), [200,400), [400,600)
boundary_10f / boundary_20f
attention_mass_removed_before/after
static_anchor_attention_mass_before/after
source_keep_ratio_by_semantic_group
context_empty_source_events
special/protected token count
D_g mean and p90 on skipped vs retained source
```

### 5.4 成立标准

Track 1 的 h10 进入标准：

```text
h10 [200,300) delta <= -5m
and boundary_10f_delta <= +0.25m
and [400,600) proxy regression <= +1m
```

或：

```text
h10 ATE delta <= -1.5m
and static_anchor_attention_mass_after >= 0.95 * before
```

h15 durability 标准：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
and durability_ratio >= 0.45
and [400,600) regression <= +1m
```

如果 Track 1 只 h10 有效、h15 失败，Codex 自动转 Track 5 washout attribution，不允许继续同类 threshold sweep。

---

## 6. Track 2：SWA local-continuity semantic source policy

SWA 不能照搬 VGGT4D-style hard source skip。SWA 的核心任务是 local continuity，因此本 track 只验证语义对 SWA cache/source topology 的作用。

### 6.1 设计原则

```text
non-overlap high-risk source 可以 remove。
overlap source 默认保护；除非非常高风险，只能 V attenuation，不能 K/V 同删。
road/building/wall/fence boundary anchor 保护。
sky/vegetation 在 overlap 中先保留 K，必要时弱化 V。
```

### 6.2 候选设计

```text
SWA_01_NONOVERLAP_RISK_REMOVE:
    dynamic/vegetation/lowtrust high-risk only in non-overlap source remove

SWA_02_OVERLAP_KEEPK_ATTENV:
    overlap high-risk source keep K, attenuate V

SWA_03_STRUCTURE_OVERLAP_PROTECT:
    structure lowD lowConflict in overlap always keep K/V

SWA_04_RISK_NONOVERLAP_PLUS_STRUCT_PROTECT:
    non-overlap risk remove + overlap structure protect

SWA_05_SKY_HORIZON_NEUTRAL:
    sky lowD in overlap protected as neutral K source, V attenuate only if highD highScaleRisk
```

### 6.3 必须记录指标

```text
ATE / segment metrics
boundary_10f_delta
boundary_20f_delta
chunk_boundary_pose_jump_delta
SWA previous-source count
SWA overlap source count
K keep ratio / V keep ratio
semantic group source mass in overlap/non-overlap
overlap pointmap residual proxy
SWA cache update norm by semantic group
```

### 6.4 成立标准

SWA candidate 晋级必须同时满足：

```text
[200,300) delta <= -3m or h10 ATE delta <= -1.0m
boundary_10f_delta <= +0.25m
boundary_20f_delta <= +0.25m
[400,600) regression <= +1m
```

如果 `[200,300)` 改善但 boundary 回退，Codex 必须自动改成：

```text
K preserve / V attenuation
non-overlap only skip
overlap structure protect
sky/road boundary neutral keep
```

如果多种 SWA action 产生完全相同 trajectory delta，Codex 必须停止 SWA rollout，回到 Track 0 检查 SWA role identity 是否被 shared source/gate 压扁。

---

## 7. Track 3：TTT static-anchor long write 与 risky short negative

Track 3 不再测试 semantic scalar。TTT 只接收语义组织后的 long/short evidence。

### 7.1 基本角色

```text
positive_long:
    road/building/wall/fence/ground-like structure
    + high masklet trust
    + low D_g
    + low update_conflict_energy
    + low scale risk
    + nontrivial TTT update contribution

short_negative:
    movable thing / vegetation / lowtrust / shadow-like uncertain
    + high D_g
    + high update_conflict_energy or high scale risk

neutral:
    sky lowD, vegetation lowD, far background, horizon context

no_long_write:
    high conflict structure, lowtrust masklets, uncertain semantic regions
```

### 7.2 候选设计

```text
TTT_01_STRUCTURE_POSITIVE_LONG:
    positive_long only, no negative

TTT_02_RISK_NO_LONG_WRITE:
    high-risk semantic/cue regions excluded from long write

TTT_03_DYNAMIC_SHORT_NEG:
    movable highD highConflict -> short negative

TTT_04_VEG_SHORT_NEG_NEUTRAL_LOW_D:
    vegetation highD highConflict -> short negative
    vegetation lowD -> neutral

TTT_05_ANCHOR_POS_PLUS_RISK_SHORTNEG:
    positive_long structure anchors + short_negative dynamic/veg/lowtrust

TTT_06_ANCHOR_POS_TTT_NATIVE_CONFLICT_GUARD:
    positive_long only if low post-zp conflict and update direction aligns with native stable direction
```

### 7.3 写入方式

TTT action 默认只作用 branch `w0`，因为历史上 `w0` 最稳定。若 `w0` 完全无信号，再试 `w0+w2`，不先动 `w1`。

写入不使用 learned weight。所有 token role 用 deterministic rule 产生，write energy 使用 absolute budget，不允许 mean-preserving normalization 抹掉整体低价值 chunk 的绝对意义。

记号：

$$
G_{commit}=G_{pos}+\lambda_{neu}G_{neu}-\gamma_{short}G_{neg}
$$

其中 $\lambda_{neu}$ 和 $\gamma_{short}$ 取固定小值或按当前 chunk robust energy 归一，不从 KITTI 拟合。

### 7.4 记录指标

```text
ATE / segment metrics
TTT branch0/1/2 update norm
post_zp_delta_norm by role
positive_long mass
short_negative mass
no_long_write mass
update_conflict_energy before/after
cosine to native stable update
memory_state_rel_diff
h10/h15 durability ratio
```

### 7.5 成立标准

TTT candidate 进入 h15/full 的条件：

```text
h10 ATE delta <= -1.5m
or h10 [200,300) delta <= -3m
```

h15 成功条件：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
and [400,600) regression <= +1m
and TTT update norm not abnormal
```

如果所有 TTT semantic candidates 都弱，结论不是“语义假设失败”，而是：semantic 不应直接控制 TTT long write，应回到 TTT-native cue / scale-state 作为 TTT 主控制，语义只做 trust/context。

---

## 8. Track 4：Semantic-conditioned C23 residual path isolation

Track 4 处理 v31-v34 的关键发现：semantic-conditioned C23 有 strong local signal，但 full online 和 C9 组合失败。本轮不训练 trigger、不 fixed chunk、不全量替换 C23。只做 deterministic residual 和 path isolation。

### 8.1 候选设计

```text
SEM_C23_01_READ_ONLY_RESID:
    D_final = D_base + lambda(D_sem - D_base)
    only frame/global read path consumes D_final

SEM_C23_02_NO_TTT:
    semantic residual allowed in read/source skip
    TTT write score still uses D_base

SEM_C23_03_NO_SWA:
    semantic residual does not control SWA source/cache

SEM_C23_04_FRAMEGLOBAL_COMPACT_ONLY:
    semantic residual only affects frame/global compact_kv source skip
    no attention-bias replacement

SEM_C23_05_STATIC_RESCUE_RESID:
    semantic residual plus structure lowD lowConflict source protection
```

### 8.2 记录指标

```text
D_base RMS
D_sem RMS
D_sem - D_base RMS
lambda effective value
D_final mass / p90 / per semantic group
attention mass to highD source
attention mass to static anchors
TTT write score unchanged check for no-TTT rows
SWA source unchanged check for no-SWA rows
```

### 8.3 成立标准

Track 4 不能只看 H9 parent。必须 H9 和 C9 parent 都过 short gate，才允许 full online。

```text
C9 parent h10 [200,300) delta <= -5m
or C9 parent h15 ATE delta <= -3m
and [400,600) regression <= +1m
```

Full online gate：

```text
C9 + semantic residual full ATE <= 32m:
    stage success, continue combination.

C9 + semantic residual full ATE <= 30m:
    Target-30 success.

C9 + semantic residual full ATE > 33m:
    semantic C23 deployment line fails; keep as diagnostic only.
```

---

## 9. Track 5：Minimal full-online combination and washout attribution

Only candidates satisfying h15 criteria from Tracks 1-4 can enter Track 5. No full online is allowed for weak h10-only signals.

### 9.1 Combination principle

Combine only non-conflicting mechanisms:

```text
frame/global risk source skip + SWA overlap anchor protect
frame/global source skip + TTT structure positive long
semantic C23 read-only residual + TTT native C9 unchanged
```

Do not combine mechanisms that both operate on the same path without isolation evidence.

### 9.2 Full online candidate families

```text
FULL_01_FG_ONLY:
    strongest frame/global candidate only

FULL_02_SWA_ONLY:
    strongest SWA boundary-healthy candidate only

FULL_03_TTT_ONLY:
    strongest TTT static-anchor/short-negative candidate only

FULL_04_FG_PLUS_SWA:
    frame/global source skip + SWA boundary protect

FULL_05_FG_PLUS_TTT:
    frame/global risk skip + TTT static anchors

FULL_06_READ_RESID_ONLY:
    semantic C23 residual read-only

FULL_07_MINIMAL_COMBO:
    best non-conflicting minimal combination
```

### 9.3 Full online metrics

```text
ATE
Rot
RPE_t / RPE_r
FinalErr
[200,300), [200,400), [400,600)
50f / 100f / 200f mean/worst
YawRMSE
Sim3Scale
boundary_10f / boundary_20f
HMC state movement
TTT update norm by branch
SWA cache health
semantic action mass over chunks
```

### 9.4 Success standards

```text
Stage success:
    full ATE <= 32m
    and [200,300) improves vs C9
    and [400,600) regression <= +1m

Target success:
    full ATE <= 30m
    no GT runtime action
    no training
    no offline trajectory rewrite
    no absolute chunk id rule

Reject:
    full ATE > 33m
    or [400,600) regression > +2m
    or SWA boundary regression > +0.5m
```

---

## 10. 并行执行安排

为了加快实验，Codex 按职责并行推进，但每个 LoGeR process 只绑定一张 GPU。建议并行方式如下：

```text
Codex A: Track 0 action / influence audit and instrumentation repair
Codex B: Track 1 frame/global source surgery
Codex C: Track 2 SWA local-continuity semantic policy
Codex D: Track 3 TTT static-anchor / short-negative write
Codex E: Track 4 semantic C23 residual path isolation
Codex F: Track 5 reports, washout attribution, and full-online queue
```

资源建议：

```text
Short rollout:
    up to 4 concurrent jobs if host RAM safe.

Full online:
    at most 2-4 concurrent jobs depending on memory footprint.

Do not use uncontrolled 8-way full concurrency.
```

Every Codex worker must write a machine-readable status file:

```text
status.json:
    phase
    row_id
    status = pending/running/done/fail/blocked
    gate_pass
    reason_if_blocked
    next_recommended_action
```

---

## 11. 失败自动分流规则

### 11.1 Hook/action 不真实

如果 synthetic action 不能改变 source/write/cache summary：

```text
Stop corresponding path.
Codex fixes hook / role projection / protected-token fallback.
Do not run trajectory rows.
```

### 11.2 Action indistinguishable

如果 semantic-only、risk-only、semantic-risk 或多个 action 的 Jaccard $>0.95$：

```text
Stop rollout for that family.
Check role tensor collapse.
Check coarse group fallback.
Check protected/special token override.
Check compact_kv fallback path.
```

### 11.3 Attention/source influence too low

如果 skipped semantic group influence mass $<0.03$：

```text
Do not claim semantic failure.
Switch candidate selection from label-based to influence-ranked.
Select top source-attention or TTT-update semantic-risk groups.
```

### 11.4 Frame/global source skip improves h10 but hurts h15

```text
Run washout attribution.
Check whether later TTT/SWA/global source update overwrites correction.
Try static anchor protection and residual D_g injection.
Do not continue threshold sweep.
```

### 11.5 SWA improves segment but boundary regresses

```text
Do not promote.
Switch to K preserve / V attenuation.
Restrict removal to non-overlap.
Protect road/building/wall/fence overlap anchors.
```

### 11.6 TTT semantic weak

```text
Do not keep adding semantic scalar variants.
Switch TTT main control back to TTT-native update_conflict / scale-state.
Use semantic only for trust and static-anchor candidate filtering.
```

### 11.7 Full online fails despite strong h15

```text
Run local-to-full transfer audit.
Compare parent state vs full state at target chunks.
Check whether action is enabled in wrong chunks.
Check C9 component conflict: read beta, TTT tri-replay, SWA replacement, mp_alpha.
Try path isolation before any new full run.
```

### 11.8 All semantic tracks fail

If Tracks 1-4 all fail with valid hooks and influence diagnostics:

```text
Semantic Prior Generator is demoted from Target-30 mainline.
Keep semantic as diagnostic / trust calibration / visualization.
Target-30 mainline returns to TTT-native / scale-state / read-cue search.
```

---

## 12. 不针对数据集调参的执行规范

本计划允许在 KITTI00/01/02/05 或其他数据集上做 failure-mode diagnosis，但不允许 dataset-specific tuning。具体要求：

```text
Allowed:
    compare semantic influence atlas across sequences;
    compare which memory path fails across datasets;
    record label distribution and masklet trust distribution;
    run same fixed policy on multiple sequences.

Forbidden:
    KITTI01-specific chunk id rule;
    sequence-specific semantic threshold;
    dataset-specific gamma/beta;
    label value table tuned for one dataset;
    trigger trained from oracle labels.
```

如果某个 policy 只在 KITTI01 chunk10 有效，它只能作为 diagnostic，不能进入 deployable claim。

---

## 13. 必须生成的报告与可视化

每个 phase 必须生成 Markdown 和 CSV/JSON summary。最低要求：

```text
phase0_action_influence_report.md
semantic_influence_atlas.csv
semantic_group_memory_path_heatmap.png
action_jaccard_matrix.csv
action_jaccard_heatmap.png
track1_frameglobal_report.md
track2_swa_report.md
track3_ttt_report.md
track4_semantic_c23_report.md
track5_full_online_report.md
failure_routing_summary.md
```

必须可视化：

```text
RGB + semantic label overlay
VideoMasklet trust overlay
D_g overlay
TTT conflict overlay
scale-risk overlay
source attention mass overlay
actual skipped source overlay
SWA overlap keep/drop overlay
TTT positive/negative/no-write overlay
h10 -> h15 durability curve
[200,300) and [400,600) segment ATE bar chart
full trajectory overlay: C9 vs candidate vs GT
```

---

## 14. 最终决策逻辑

本轮的价值不只是能不能直接进 $30m$，而是要对“语义风险区域干扰重建”这个假设给出清晰判断。

如果 Track 0 证明 semantic-risk groups 有高 influence，且 Track 1/2/3 至少一条 path 有 h15 级别有效候选，那么继续推进 semantic memory control。

如果 semantic action hook 真实、influence nontrivial，但所有 path 都无 h10/h15 信号，则说明当前 VideoMasklet/SPG 形式下，语义不是 Target-30 主杠杆。

如果 frame/global source skip 有效但 TTT/SWA 失败，则语义应定位为 read/source control，而不是 long-memory write prior。

如果 static anchors 有效，则下一阶段围绕 distributed static scale-anchor persistence 设计，而不是继续 dynamic suppression。

如果 SWA boundary 总是被破坏，则语义不应直接 hard skip SWA overlap source，SWA 只保留 conservative K/V attenuation policy。

最终 full online 成功标准：

```text
C9 + training-free semantic memory policy
ATE <= 30m
no learned trigger/selector/router
no GT runtime action
no fixed chunk id
no offline trajectory rewrite
```

---

## 15. 一句话总结

v37 的核心不是继续赌某个语义规则，而是：

> **先确认语义风险区域是否真的以高影响力进入 LoGeR 的 frame/global/SWA/TTT 计算路径，再分别做 training-free 的 source surgery、local-continuity protection 和 static-anchor long memory。只有证明这些 action 有 h15 持久收益，才允许进入 full online。**


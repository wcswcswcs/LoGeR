# ACL2 v36：Training-Free Semantic Memory Control / Target-30 实验计划

日期：2026-05-24  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / Video Masklet / SemanticKITTI sparse anchor  
当前 deployable best：`C9_P0_R2 = 33.7629421029m`  
阶段目标：先进入 `KITTI01 ATE <= 30m`  
核心约束：**training-free、no GT runtime、no dataset-specific tuning、no learned trigger / selector / router**

---

## 0. 本轮计划要纠正什么

前几轮语义实验虽然把 `VideoMasklet -> SemanticPrior -> frame/global/SWA/TTT` 的工程链路逐步接通，也验证了 no-op、fine label、SemanticKITTI sparse projection、action distinguishability、semantic-conditioned C23 等能力，但仍然没有得到 deployable online Target-30 结果。现在不能再继续做“语义规则矩阵”的小修小补，例如：

```text
sky skip
vegetation skip
structure keep
lowstuff highD negative
semantic-only / risk-only / semantic-risk
frame / global / swa / ttt 全组合
```

这些实验已经暴露出几个本质问题：

```text
1. 语义类别不是 trajectory drift 的直接因果变量。
2. 不同名字的 semantic policy 可能最后产生相同或近似相同的 action。
3. 单个 masklet 的 intervention 太弱，不能代表语义整体潜力。
4. SWA 局部改善经常伴随 boundary regression。
5. semantic-conditioned C23 有局部强信号，但 full online 迁移失败。
6. 之前提出 learned trigger / trained router 是错误方向，不符合 training-free 项目目标。
```

因此，本轮的目标不是继续试更多人工语义规则，而是验证一个更本质的假设：

> **天空、植被、阴影、动态物体等区域确实可能干扰重建，但它们必须通过“真实 source / cache / write action”作用到 LoGeR 的 frame/global/SWA/TTT memory；语义本身不能直接决定 action，必须与 `D_g`、TTT conflict、scale-risk、masklet trust 和 memory path 类型共同决定。**

本轮把 Semantic Prior Generator 从“语义写入分数生成器”重定义为：

```text
training-free semantic-cue memory role router
```

它的职责不是训练一个触发器，也不是拟合一个选择器，而是用固定、可解释、可审计的规则，把语义风险区域和已有 cue 结合起来，控制 LoGeR 中不同 memory path 的真实计算。

---

## 1. 项目硬边界：禁止再次走偏

### 1.1 允许的输入

运行时只能使用：

```text
LoGeR 内部 attention / QK / TTT / SWA trace
Video Masklet 前端输出
SemanticKITTI sparse 3D projection 作为 trust calibration / diagnostic anchor
Dynamic cue / D_g / C23 / TTT conflict / scale-risk proxy
无 GT 的 overlap / boundary / consistency 指标
```

### 1.2 禁止的做法

本轮明确禁止：

```text
1. 训练 trigger / classifier / selector / role router。
2. 用 oracle label 拟合规则。
3. 用 absolute chunk id 作为策略条件。
4. 为 KITTI01 或任何单一数据集调阈值、gamma、label value。
5. 把 fixed chunk diagnostic 写成 deployable success。
6. 把 short rollout / sandbox oracle / GT sparse projection diagnostic 写成 full online success。
7. 把后处理轨迹修正写成 TTT/SWA/frame/global memory improvement。
```

### 1.3 本轮成功必须是 deployable online

最终进入 Target-30 的结果必须满足：

```text
full online run
no GT runtime action
no offline trajectory rewrite
training-free deterministic policy
counts_as_deployable_online_success = true
```

---

## 2. 当前事实与本轮判断

### 2.1 当前 deployable best

当前最好可部署 online result 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

目标是：

```text
KITTI01 ATE <= 30m
```

因此还差：

$$
33.7629421029 - 30.0 = 3.7629421029m
$$

这个差距不是 `0.1m ~ 0.3m` 的语义阈值微扫可以解决的。

### 2.2 语义路线已有证据

已有证据可以分成四类：

```text
工程接线：
    v25B-v28 已经证明 video masklet / fine label / semantic role / token risk 可以进入 runtime。

稀疏 3D anchor：
    v29C 已经补齐 SemanticKITTI/KITTI Odometry sequence 01 数据，生成 sparse 3D->2D projection，用于校准 masklet trust。

masklet causal bank：
    单 masklet / top road / diverse masklet intervention 信号偏弱，不能证明语义直接 action 有上界。

semantic-conditioned C23：
    v31-v34 证明语义重新解释 C23 / D_g 有强局部信号，但 full online 和 C9 兼容性仍失败。
```

### 2.3 本轮核心判断

语义不能直接写成：

```text
semantic label -> action
```

而应该写成：

$$
action = f(semantic, D_g, C_{ttt}, S_{scale}, Q_{mask}, path)
$$

其中 $f$ 必须是固定的 training-free rule，不是训练出来的模型。

更重要的是，本轮必须先证明 action 真的进入了 LoGeR 的真实计算路径：

```text
frame/global:
    K/V source token 是否真的被 skip / keep / compact？

SWA:
    previous-source cache / overlap source / K/V 是否真的改变？

TTT:
    positive / neutral / short-negative / no-long-write 是否真的改变 post-zp update？
```

如果 action 没有真实变化，则任何 ATE 结果都不能说明语义假设失败。

---

## 3. 本轮整体目标

本轮总目标是：

> **验证“语义风险区域 + LoGeR 内部 cue”是否能形成真实、可部署、training-free 的 memory control，并把当前 C9 best 从 `33.76m` 推向 `<=30m`。**

为此，本轮要回答六个科学问题。

### 3.1 问题 A：之前语义实验失败，是否因为 action hook 没有真实生效？

如果不同 semantic policies 最后生成了近似相同的 action，或者 skip 掉的是模型本来就不 attend 的 token，那么过去的负结果不能否定语义假设。

### 3.2 问题 B：VGGT4D-style source skip 在 LoGeR 的 frame/global attention 上是否成立？

VGGT4D 的启发不是“调 attention bias 后结束”，而是让动态区域不再作为 early inference 的 image token source。LoGeR 中最接近的是：

```text
保留 query；控制 K/V source。
```

本轮优先在 frame/global attention 上做 group-level source skip，而不是先动 TTT。

### 3.3 问题 C：SWA 的语义控制是否必须保护 local boundary？

SWA 是 local lossless memory。它的目标是 adjacent alignment，不是 long-term writing。因此语义 skip 不能简单复用 TTT 的规则。SWA 需要单独验证：

```text
哪些 source 是 local alignment anchor？
哪些 source 可以从 non-overlap cache 删除？
哪些 source 只能 attenuate V，不能删除 K？
```

### 3.4 问题 D：TTT 语义写入是否应从“semantic scalar”改成“static long anchor + short negative”？

TTT 是 compressed global memory。语义在 TTT 中最可能的作用不是所有 lowstuff 少写，而是：

```text
stable structure -> positive long write
high-risk dynamic/vegetation/shadow -> short negative or no-long-write
sky / lowD vegetation -> neutral context，不做 long positive，也不做 strong negative
```

### 3.5 问题 E：semantic-conditioned C23 的局部信号能否以 residual 方式、read-only 方式、path-isolated 方式兼容 C9？

v31-v34 表明 semantic-conditioned C23 有局部强信号，但 full online / C9 叠加失败。下一步不是训练 trigger，而是拆解冲突源，验证语义 residual 是否能只作用于 read/source，而不污染 TTT/SWA。

### 3.6 问题 F：如果所有真实 action 都失败，语义是否应该从主线降级？

如果在 action hook 真实有效、group-level source skip 有效测试、SWA/TTT path-specific 设计都失败后，语义仍然不能进入 Target-30，那么 Semantic Prior Generator 应降级为：

```text
diagnostic / trust calibration / visualization / weak regularizer
```

Target-30 主线应转回：

```text
TTT-native causal action
trajectory-state / scale-state module
read cue / source skip 非语义路线
```

---

## 4. 核心假设

---

## H0：过去的 semantic negative result 可能来自 action collapse 或 hook bug

### 假设

过去 `SEM_ONLY / RISK_ONLY / SEM_RISK` 等实验名不同，但底层 action 可能高度重合，或者 action 没有真实改变 frame/global/SWA/TTT 计算。因此在继续跑 full/rollout 前，必须先证明 action 真实且可区分。

### 实验设计

构造 synthetic mask stress test，不依赖 video masklet 预测：

```text
SYN_01_all_patch_skip
SYN_02_center_box_skip
SYN_03_random_20pct_skip
SYN_04_left_half_skip
SYN_05_all_dynamic_role
SYN_06_all_static_role
```

分别作用于：

```text
frame source
chunk/global source
SWA source/cache
TTT write role
```

再对真实 semantic policies 做 action distinguishability：

```text
semantic-only
risk-only
semantic-risk
static-anchor
dynamic-risk
vegetation-highD
sky-highD
structure-lowD
```

### 必须记录的指标

```text
frame_source_keep_mask
frame_source_skip_mask
global_source_keep_mask
global_source_skip_mask
swa_k_keep_mask
swa_v_keep_mask
swa_cache_keep_mask
ttt_positive_mask
ttt_neutral_mask
ttt_short_negative_mask
ttt_no_long_write_mask
```

对每组 action 计算：

```text
Jaccard(action_A, action_B)
source_keep_ratio
source_skip_ratio
per-label action mass
attention_mass_removed_before
attention_mass_removed_after
SWA_overlap_source_mass_removed
TTT_update_norm_changed
post_zp_delta_norm_changed
context_empty_source_events
```

### 成立标准

H0 通过条件：

```text
1. synthetic all/source skip 必须显著改变对应 path 的 source/write summary；
2. context_empty_source_events = 0；
3. 不同 semantic policies 的 Jaccard <= 0.85，或 keep ratio 差 >= 0.05；
4. 被 skip 的 source 原本 attention mass >= 0.05，否则说明 skip 的不是关键 source；
5. TTT semantic action 必须让 per-role update norm 或 post-zp delta norm 改变 >= 5%。
```

如果 H0 不通过：

```text
禁止继续 rollout；Codex 必须先修 hook / projection / fallback / protected-token 逻辑。
```

---

## H1：VGGT4D-style group-level source skip 应优先在 frame/global attention 验证

### 假设

动态物体、high-D 植被、阴影/低可信区域会作为 frame/global attention 的 K/V source 干扰当前 pose / geometry reasoning。正确操作不是删除 query，也不是只改 logits，而是训练-free 地控制 K/V source。

### 实验设计

固定 parent：

```text
primary parent = C9_P0_R2
secondary parent = H9_P0_R2
chunks = 6, 10, 16
horizon = h10, h15
```

先跑 read-only / source-only，不动 TTT write：

```text
FG_SKIP_01_DYNAMIC_HIGHD
    dynamic thing + highD -> source skip

FG_SKIP_02_VEGETATION_HIGHD
    vegetation/tree + highD -> source skip

FG_SKIP_03_LOWTRUST_HIGHD
    low-trust masklet + highD -> source skip

FG_SKIP_04_DYNAMIC_VEG_HIGHD
    dynamic thing OR vegetation highD -> source skip

FG_SKIP_05_DYNAMIC_VEG_SKIP_STRUCTURE_PROTECT
    dynamic/vegetation highD source skip
    structure lowD source protect

FG_SKIP_06_COMPACT_KV_TRUE
    same as 05，但使用 true compact_kv，不使用 bias approximation

FG_SKIP_07_SHADOW_LOWCONF_HIGHD
    shadow/low-light/reflection proxy = low mask trust + highD + low confidence -> source attenuation
```

每个候选都要测试：

```text
frame only
global/chunk only
frame + global
```

### 关键实现约束

```text
1. Query tokens 保持完整。
2. Special/protected tokens 永远保留。
3. K/V source 被 skip/compact。
4. early/read layers 优先。
5. 不允许 TTT write score 跟随 semantic skip 改变。
```

### 必须记录的指标

轨迹指标：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
[200,300)
[200,400)
[400,600)
YawRMSE
Sim3Scale
```

source 指标：

```text
num_context_source_skip_applied
mean_context_source_keep_ratio
max_context_source_skip_tokens
num_context_empty_source_events
attention_mass_to_skipped_source_before
attention_mass_to_skipped_source_after
attention_mass_to_structure_source_before/after
attention_mass_to_dynamic_source_before/after
```

semantic/cue 指标：

```text
per-label skip ratio
per-label attention removed
D_g mean/p90 for skipped and kept tokens
masklet trust mean for skipped and kept tokens
SemanticKITTI sparse-anchor agreement for skipped/kept masklets
```

### 成立标准

短滚动 gate：

```text
h10 [200,300) delta <= -5m
or h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
```

健康 gate：

```text
[400,600) regression <= +1m
boundary_10f_delta <= +0.25m
boundary_20f_delta <= +0.25m
context_empty_source_events = 0
structure attention mass not reduced by more than 10%
```

如果 H1 不通过：

```text
1. 如果 attention_mass_removed < 0.05：说明 skip 没打到关键 source，Codex 改 layer/scope/source type。
2. 如果 boundary regression：改为 K保留/V衰减，或 non-overlap-only。
3. 如果 structure mass 被误删：加入 structure rescue/protect。
4. 如果 compact_kv 与 bias 结果不同很大：优先保留 compact_kv 路线，bias 只作为 diagnostic。
```

---

## H2：SWA 语义控制必须围绕 local continuity，而不是简单跳过风险区域

### 假设

SWA 是局部无损记忆，负责相邻 chunk 的高精度对齐。语义风险区域可能污染 SWA cache，但 hard skip 也会破坏 boundary alignment。因此 SWA 应使用更保守的 local-continuity policy。

### 实验设计

只在 H0 证明 SWA action 真实可区分后启动。

候选：

```text
SWA_01_NONOVERLAP_RISK_REMOVE
    dynamic/vegetation/highD 只在 non-overlap source 中 remove；overlap source 全保留。

SWA_02_OVERLAP_K_KEEP_V_ATTEN
    overlap 中 high-risk source 保留 K，衰减 V。

SWA_03_STRUCTURE_OVERLAP_PROTECT
    road/building/wall/fence lowD 在 overlap 中强制保护。

SWA_04_SKY_ROAD_BOUNDARY_NEUTRAL
    sky/road boundary 作为 neutral local context，禁止 hard skip。

SWA_05_RISK_REMOVE_STRUCTURE_PROTECT
    non-overlap risk remove + overlap structure protect。

SWA_06_DYNAMIC_ONLY_NONOVERLAP
    只移除 non-overlap dynamic thing，不处理 vegetation/sky。
```

### 必须记录的指标

```text
SWA previous-source K count
SWA previous-source V count
SWA overlap source mass
SWA non-overlap source mass
SWA semantic group mass by source type
SWA cache keep/drop/attenuate ratio
boundary_10f_ATE
boundary_20f_ATE
chunk_boundary_pose_jump
[200,300)
[400,600)
```

### 成立标准

SWA candidate 只有同时满足下面条件才可晋级：

```text
h10 [200,300) delta <= -3m
or h15 ATE delta <= -2m
```

且：

```text
boundary_10f_delta <= +0.25m
boundary_20f_delta <= +0.25m
chunk_boundary_pose_jump_delta <= +0.25m
[400,600) regression <= +1m
```

如果 SWA 改善 `[200,300)` 但 boundary 回退：

```text
禁止进入 full；Codex 自动改为 K keep / V attenuation / non-overlap only / structure anchor protect。
```

---

## H3：TTT 语义写入应拆成 static long anchor 与 short negative，不再使用 semantic scalar

### 假设

TTT 是 compressed global memory，用来维护 global coordinate / scale。语义在 TTT 中最有价值的不是“低价值类别少写”，而是区分：

```text
stable structure anchor -> positive long write
risky dynamic/vegetation/shadow -> short negative or no-long-write
sky/lowD vegetation -> neutral context，不做 positive long，也不做 strong negative
```

### 实验设计

固定只动 TTT，frame/global/SWA 不启用语义 source skip。先只作用 `branch=w0`，因为历史上 `w0` 是最稳的入口控制分支。

候选：

```text
TTT_ANCHOR_01_STRUCTURE_LOW_D
    structure semantic AND lowD -> positive long

TTT_ANCHOR_02_STRUCTURE_LOWD_LOWCONFLICT
    structure AND lowD AND low update_conflict -> positive long

TTT_ANCHOR_03_STRUCTURE_LOWD_LOWCONFLICT_LOWSCALE
    structure AND lowD AND low conflict AND low scale-risk -> positive long

TTT_NEG_01_DYNAMIC_HIGHD_CONFLICT
    dynamic thing AND highD AND high conflict -> short negative / no-long-write

TTT_NEG_02_VEGETATION_HIGHD_CONFLICT
    vegetation AND highD AND high conflict -> short negative / no-long-write

TTT_MIX_01_STATIC_LONG_RISK_SHORT
    positive long = structure lowD lowConflict lowScale
    short negative = dynamic/vegetation highD highConflict
    neutral = sky lowD / lowstuff lowD

TTT_MIX_02_STATIC_LONG_RISK_NOLONG
    same as MIX_01，但 high-risk 只 no-long-write，不做 negative replay。
```

### 关键公式

对每个 token $i$，先做 semantic-conditioned robust normalization：

$$
z_D(i)=\frac{D_g(i)-\mu_{sem(i)}(D_g)}{\sigma_{sem(i)}(D_g)+\epsilon}
$$

$$
z_C(i)=\frac{C_{ttt}(i)-\mu_{sem(i)}(C_{ttt})}{\sigma_{sem(i)}(C_{ttt})+\epsilon}
$$

$$
z_S(i)=\frac{S_{scale}(i)-\mu_{sem(i)}(S_{scale})}{\sigma_{sem(i)}(S_{scale})+\epsilon}
$$

然后定义：

```text
positive_long:
    structure-like semantic
    AND z_D <= q30
    AND z_C <= q30
    AND trust high

short_negative:
    dynamic / vegetation / lowtrust / shadow-like semantic
    AND z_D >= q80
    AND z_C >= q80

neutral:
    sky / vegetation / lowstuff
    AND not short_negative
```

这里的 `q30/q80` 是固定的 robust quantile rule，不从 KITTI01 拟合；如果换数据集仍使用相同规则。

### 必须记录的指标

```text
TTT role mass: positive / neutral / short_negative / no_long_write
per-label TTT role mass
per-branch update norm w0/w1/w2
post-zp delta norm by branch/layer
update_conflict_energy before/after
scale-risk proxy before/after
TTT write score mean/p90
TTT long/short contribution norm
[200,300)
[400,600)
h10/h15 durability ratio
```

### 成立标准

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
```

且：

```text
[400,600) regression <= +1m
TTT update norm not > 2x baseline
Rot regression <= +0.15deg
```

如果 TTT semantic write 仍弱：

```text
1. 若 positive_long mass 太小：放宽 structure anchor selection，但保持 low conflict。
2. 若 negative hurt downstream：改 short_negative -> no-long-write。
3. 若 TTT action 不改变 post-zp delta：查 branch/layer hook 或 zeropower/norm restore 折叠。
4. 若所有 TTT semantic 候选低于 gate：TTT semantic 降级，继续用 TTT-native conflict/scale-state 主线。
```

---

## H4：semantic-conditioned C23 应以 residual read-only / path-isolated 方式验证，而不是训练 trigger 或全路径替换

### 假设

v31-v34 证明 semantic-conditioned C23 有局部强信号，但 full online/C9 叠加失败。原因可能是：

```text
1. semantic D_g 替换太强；
2. semantic cue 同时进入 TTT/SWA，破坏 C9 已有平衡；
3. semantic cue 应只用于 read/source，不应进入 write；
4. full online 常开导致低风险 chunk 也被过度干预。
```

本轮不用训练 trigger，而是用 deterministic residual injection 和 path isolation。

### 实验设计

定义：

$$
D_{final}=D_{base}+\lambda(D_{sem}-D_{base})
$$

其中：

$$
\lambda = \min\left(1, \frac{\rho \cdot RMS(D_{base})}{RMS(D_{sem}-D_{base})+\epsilon}\right)
$$

固定：

```text
rho = 0.25
```

这是固定设计常数，不针对 KITTI 调参。

候选：

```text
SEM_C23_01_READ_ONLY_RESID
    semantic residual 只用于 frame/global read bias。

SEM_C23_02_NO_TTT_WRITE
    semantic residual 不进入 TTT write score。

SEM_C23_03_NO_SWA
    semantic residual 不控制 SWA source/cache。

SEM_C23_04_FRAME_GLOBAL_COMPACT_ONLY
    semantic residual 只用于 frame/global compact_kv source skip。

SEM_C23_05_STRUCTURE_RESCUE_RESID
    semantic residual + structure lowD source protect。

SEM_C23_06_C9_READ_ONLY_COMPAT
    在 C9 parent 下只改 read，不动 C9 TTT/SWA。
```

### 必须记录的指标

```text
D_base mean/std/p90
D_sem mean/std/p90
D_final mean/std/p90
RMS(D_sem-D_base)
lambda actual
attention mass shift
source keep ratio
TTT write score difference vs C9
SWA source mass difference vs C9
ATE / Rot / RPE / FinalErr
[200,300) / [400,600)
h10/h15 durability
```

### 成立标准

短滚动 gate：

```text
h15 [200,300) delta <= -5m
and [400,600) regression <= +1m
```

full online gate：

```text
C9 + semantic residual ATE <= 32m -> stage success
C9 + semantic residual ATE <= 30m -> Target-30 success
```

如果 H4 失败：

```text
1. 如果 read-only 好但 C9 full 差：拆 C9 component conflict。
2. 如果 no-TTT 好于 full path：semantic cue 禁止进入 TTT write。
3. 如果 no-SWA 好于 full path：semantic cue 禁止控制 SWA overlap。
4. 如果 all residual weak：semantic-conditioned C23 降级为 diagnostic。
```

---

## H5：有效语义策略必须组合“risk suppression”和“static scale-anchor construction”

### 假设

进入 ATE 30 的关键可能不是只 suppress dynamic，而是构建稳定的 distributed static scale anchor set。语义应该帮助 LoGeR 在多个 memory path 中持续保留这些 anchor：

```text
frame/global:
    static anchor 作为 K/V source keep

SWA:
    static anchor 在 overlap/local cache 中 protect

TTT:
    static anchor 作为 positive long write
```

### 实验设计

构造 static anchor：

```text
semantic in {road, building, wall, fence, stable ground}
AND trust high
AND z_D low
AND z_C low
AND z_S low
AND source attention mass high
```

候选：

```text
ANCHOR_01_FRAME_GLOBAL_ONLY
ANCHOR_02_SWA_ONLY
ANCHOR_03_TTT_ONLY
ANCHOR_04_FRAME_GLOBAL_SWA
ANCHOR_05_FRAME_GLOBAL_TTT
ANCHOR_06_FRAME_GLOBAL_SWA_TTT
ANCHOR_07_STATIC_ANCHOR_PLUS_RISK_SUPPRESS
```

`ANCHOR_07` 同时启用：

```text
static anchor protect / positive long
high-risk dynamic/vegetation source skip or no-long-write
```

### 必须记录的指标

```text
static_anchor_token_count
static_anchor_attention_mass
static_anchor_SWA_overlap_mass
static_anchor_TTT_positive_mass
static_anchor temporal persistence
anchor dropped by later chunks ratio
[200,300)
[400,600)
Sim3 scale drift
yaw drift
FinalErr
```

### 成立标准

```text
h15 ATE delta <= -3m
or full online ATE <= 32m
```

且：

```text
static_anchor_mass must not collapse after h10
[400,600) regression <= +1m
SWA boundary regression <= +0.25m
```

如果 H5 失败：

```text
1. 如果 anchor mass 太少：检查 semantic trust / projected 3D support 过严。
2. 如果 anchor mass 足够但无 trajectory effect：说明 semantic static anchor 不是主杠杆。
3. 如果 h10 好 h15 弱：做 anchor persistence / washout attribution。
```

---

## 5. 并行执行计划

本轮分成六条 Codex 并行工作线。任何一条线都必须遵守 training-free 约束。

---

### Codex-A：Action Realism / Hook Audit

目标：证明 semantic action 真正改变底层计算。

任务：

```text
1. 实现 synthetic mask stress test。
2. 输出所有 path 的 action tensors。
3. 计算 action Jaccard / keep ratio / attention mass removed。
4. 验证 frame/global/SWA/TTT hooks 的实际 effect。
```

输出文件：

```text
action_realism/action_tensor_summary.csv
action_realism/action_jaccard_matrix.csv
action_realism/source_attention_mass_removed.csv
action_realism/swa_cache_effect_summary.csv
action_realism/ttt_update_effect_summary.csv
action_realism/hook_audit_summary.json
action_realism/hook_audit_report.md
```

Gate：H0 通过后，其它 track 才能把结果当成有效机制证据。

失败自动处理：

```text
if synthetic mask no effect:
    fix hook before rollout
if policy action equivalent:
    fix role projection before rollout
if attention mass removed too small:
    change layer/scope/source target before rollout
```

---

### Codex-B：Frame/Global VGGT4D-style Source Skip

目标：验证 dynamic/vegetation/shadow/high-risk 区域是否作为 frame/global K/V source 干扰重建。

任务：

```text
1. 实现 group-level K/V source skip / compact_kv。
2. 跑 FG_SKIP_01 到 FG_SKIP_07。
3. 分 frame-only / global-only / frame+global。
4. 对 C9 和 H9 parent 都做 h10/h15 short rollout。
```

输出文件：

```text
fg_source_skip/candidate_metrics.csv
fg_source_skip/context_skip_summary.jsonl
fg_source_skip/attention_mass_by_label.csv
fg_source_skip/source_keep_by_label.csv
fg_source_skip/trajectory_segments.csv
fg_source_skip/gate_summary.json
```

晋级条件：满足 H1 short/h15 gate。

失败自动处理：

```text
if source skip doesn't change attention mass:
    switch from bias to compact_kv or target earlier layers
if boundary regression:
    protect overlap / structure anchors
if structure mass removed:
    add structure lowD rescue
```

---

### Codex-C：SWA Local-Continuity Semantic Control

目标：验证 SWA semantic role 是否能改善 disease window，同时不破坏 adjacent boundary。

任务：

```text
1. 实现 non-overlap-only remove。
2. 实现 overlap K keep / V attenuation。
3. 实现 structure overlap protect。
4. 跑 SWA_01 到 SWA_06。
```

输出文件：

```text
swa_semantic/swa_source_mass_by_label.csv
swa_semantic/swa_kv_keep_ratio.csv
swa_semantic/boundary_10f_20f.csv
swa_semantic/chunk_boundary_pose_jump.csv
swa_semantic/trajectory_segments.csv
swa_semantic/gate_summary.json
```

晋级条件：满足 H2 gate。

失败自动处理：

```text
if [200,300) improves but boundary regresses:
    switch to K keep / V attenuation
    or non-overlap-only skip
if anchor/remove rows identical:
    inspect underlying K/V cache mask and fallback source
```

---

### Codex-D：TTT Static Anchor / Short Negative

目标：验证语义在 TTT 中是否应作为 positive long / neutral / short negative，而不是 scalar write prior。

任务：

```text
1. 实现 TTT_ANCHOR_01-03。
2. 实现 TTT_NEG_01-02。
3. 实现 TTT_MIX_01-02。
4. 只在 w0 先测，必要时再分 layer。
```

输出文件：

```text
ttt_semantic/ttt_role_mass_by_label.csv
ttt_semantic/per_branch_update_norm.csv
ttt_semantic/post_zp_delta_norm.csv
ttt_semantic/update_conflict_before_after.csv
ttt_semantic/scale_risk_before_after.csv
ttt_semantic/trajectory_segments.csv
ttt_semantic/gate_summary.json
```

晋级条件：满足 H3 gate。

失败自动处理：

```text
if TTT role mass changes but post-zp delta unchanged:
    inspect zeropower/norm restore folding
if negative hurts downstream:
    change short_negative to no_long_write
if anchor too sparse:
    loosen trust but keep low conflict condition
```

---

### Codex-E：Semantic-conditioned C23 Residual / Path Isolation

目标：保留 v31 semantic C23 局部强信号，但避免全路径替换和 C9 冲突。

任务：

```text
1. 实现 fixed residual D_final = D_base + lambda(D_sem-D_base)。
2. lambda 使用 energy-normalized fixed rho，不训练。
3. 跑 SEM_C23_01-06。
4. 对每个候选记录是否进入 TTT/SWA/write path。
```

输出文件：

```text
sem_c23_residual/d_map_stats.csv
sem_c23_residual/lambda_actual.csv
sem_c23_residual/path_diff_vs_C9.csv
sem_c23_residual/attention_mass_shift.csv
sem_c23_residual/swa_diff_vs_C9.csv
sem_c23_residual/ttt_write_diff_vs_C9.csv
sem_c23_residual/trajectory_segments.csv
sem_c23_residual/gate_summary.json
```

晋级条件：满足 H4 gate。

失败自动处理：

```text
if read-only helps but full-path hurts:
    disable semantic in TTT or SWA according to path diff
if residual too weak:
    inspect RMS normalization, not tune to KITTI
if all residual weak:
    semantic C23 downgraded to diagnostic
```

---

### Codex-F：Distributed Static Scale-Anchor Persistence

目标：验证语义是否能通过分布式 static anchors 形成长期、稳定的 scale / trajectory correction。

任务：

```text
1. 构造 static anchor set。
2. 跑 ANCHOR_01-07。
3. 对 h10 -> h15 做 persistence / washout attribution。
4. 如果有 h15 gate，再 full online。
```

输出文件：

```text
static_anchor/anchor_token_count.csv
static_anchor/anchor_attention_mass.csv
static_anchor/anchor_swa_overlap_mass.csv
static_anchor/anchor_ttt_positive_mass.csv
static_anchor/anchor_persistence_h10_h15.csv
static_anchor/washout_attribution.csv
static_anchor/trajectory_segments.csv
static_anchor/gate_summary.json
```

晋级条件：满足 H5 gate。

失败自动处理：

```text
if h10 good but h15 weak:
    identify whether TTT/SWA/global/merge washes out correction
if anchors not used by attention:
    route through frame/global source keep first
if anchors hurt [400,600):
    restrict TTT long write, keep source only
```

---

## 6. 统一记录指标

所有候选必须至少记录以下指标。

### 6.1 轨迹指标

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
[0,100)
[100,200)
[200,300)
[200,400)
[400,600)
50f / 100f / 200f mean and worst
```

### 6.2 Action realism 指标

```text
source_keep_ratio
source_skip_ratio
action_jaccard
attention_mass_removed
attention_mass_to_structure
attention_mass_to_dynamic
context_empty_source_events
protected_token_count
fallback_source_count
```

### 6.3 Semantic / masklet quality 指标

```text
fine label distribution
coarse group distribution
masklet temporal IoU
label flip rate
fragmentation score
Q_mask mean/p90
SemanticKITTI sparse support ratio
SemanticKITTI agreement ratio
masklet trust score
```

### 6.4 SWA 指标

```text
SWA K keep ratio
SWA V keep/attenuate ratio
SWA overlap source mass
SWA non-overlap source mass
boundary_10f_ATE
boundary_20f_ATE
chunk_boundary_pose_jump
SWA cache movement h10->h15
```

### 6.5 TTT 指标

```text
TTT positive / neutral / short_negative / no_long_write mass
per-label TTT role mass
branch w0/w1/w2 update norm
post-zp delta norm
norm restore ratio
update_conflict_energy before/after
scale-risk before/after
TTT state movement h10->h15
```

### 6.6 Durability 指标

```text
h10_delta
h15_delta
durability_ratio = abs(h15_delta / (h10_delta + eps))
[400,600) regression
HMC state movement
TTT state movement
SWA cache movement
merge/gauge movement
```

---

## 7. 必须可视化的内容

### 7.1 Per-frame overlay

每个通过 h10 gate 的候选必须输出：

```text
RGB
semantic fine label overlay
VideoMasklet trust overlay
SemanticKITTI sparse projection overlay
D_g map
z_D / z_C / z_S maps
source skip mask
static anchor mask
short negative mask
SWA overlap protect mask
TTT positive/negative mask
```

### 7.2 Action audit dashboard

```text
action Jaccard heatmap
per-label keep/skip bar chart
attention mass removed by label
SWA K/V keep by label
TTT role mass by label
```

### 7.3 Trajectory dashboard

```text
GT vs baseline vs candidate trajectory
per-100f ATE curve
[200,300) zoom plot
[400,600) zoom plot
Sim3 scale over time
Yaw drift over time
boundary error plot
```

### 7.4 Durability / washout dashboard

```text
h10 vs h15 effect decay
TTT/SWA/global/merge state movement
anchor persistence curve
role mass over future chunks
```

---

## 8. Promotion / Stop Rules

### 8.1 Hard no-run stop

如果以下任一条件成立，不允许跑 rollout：

```text
action Jaccard > 0.95 and keep_ratio_diff < 0.02
context_empty_source_events > 0
synthetic mask stress no real effect
semantic cache miss or stale run contamination
no-op parity fails
```

### 8.2 h10 entry gate

候选进入 h15 confirmation 需要满足：

```text
h10 ATE delta <= -1.5m
or h10 [200,300) delta <= -3m
```

且：

```text
[400,600) regression <= +1m
boundary_10f_delta <= +0.25m if SWA involved
```

### 8.3 h15 durability gate

候选进入 full online 需要满足：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
```

且：

```text
durability_ratio >= 0.45
[400,600) regression <= +1m
SWA boundary safe
TTT update norm safe
```

### 8.4 Full online gate

```text
C9 + candidate ATE <= 32m:
    stage success, continue family

C9 + candidate ATE <= 30m:
    Target-30 success

C9 + candidate ATE > 33m:
    candidate family is diagnostic only unless it has strong segment value
```

---

## 9. 不针对数据集调参的执行规则

允许：

```text
用 KITTI01 诊断 failure mode。
用 KITTI00/02/05 或其他数据集验证同一规则是否行为一致。
使用每个 sequence 内部的 robust statistics，如 median/MAD/quantile。
```

禁止：

```text
KITTI01 专用 label table。
KITTI01 专用 chunk list。
KITTI01 专用 threshold。
训练任何 trigger/selector/router。
使用 sequence id 或 absolute chunk id。
```

所有规则必须写成：

```text
same semantic taxonomy
same robust quantile rule
same memory path rule
same gate
```

不同数据集只用于诊断，不用于调参。

---

## 10. 预期结论分支

### 分支 A：frame/global source skip 有强信号

如果 H1 通过，说明 VGGT4D-style source skip 适合 LoGeR read/source path。下一步重点是把它和 C9 兼容，并做 full online。

### 分支 B：static anchor 有强 h15/full 信号

如果 H5 通过，说明语义主杠杆不是 suppress dynamic，而是构建 durable static scale-anchor。后续主线转向 anchor persistence。

### 分支 C：SWA 改善局部但伤 boundary

如果 SWA 一直出现这个模式，SWA 语义控制只允许 non-overlap / V attenuation，不允许 hard source remove。

### 分支 D：TTT semantic write 弱

如果 H3 失败，语义不再直接控制 TTT write；TTT 继续走 TTT-native conflict / scale-state cue，语义只用于 static anchor / trust。

### 分支 E：全部真实 action 都无上界

如果 H0 已通过且 H1-H5 全失败，则必须停止把 Semantic Prior Generator 作为 Target-30 主线。语义保留为：

```text
trust calibration
visualization
failure diagnosis
weak auxiliary regularizer
```

Target-30 主线转向：

```text
trajectory-state / scale-state module
TTT-native causal action
non-semantic source skip / read cue refinement
```

---

## 11. 本轮最小执行顺序

为了加速，Codex 应按以下顺序并行执行。

```text
Day 1 / Batch 0:
    Codex-A synthetic mask stress + action realism audit
    Codex-B frame/global source skip smoke
    Codex-C SWA K/V distinguishability smoke
    Codex-D TTT role mass / post-zp smoke

Day 1 / Batch 1:
    只对通过 action realism 的 candidates 跑 h10。
    不通过的 family 直接修 hook，不跑 trajectory。

Day 2 / Batch 2:
    对 h10 gate 通过的 candidates 跑 h15。
    同时做 durability/washout attribution。

Day 2 / Batch 3:
    对 h15 gate 通过的 candidates 跑 C9 full online。
    不满足 h15 gate 的 family 停止，不做 full。
```

---

## 12. 一句话总结

本轮的核心不是继续问：

```text
sky 要不要跳过？
vegetation 要不要少写？
road 要不要保留？
```

而是要回答：

> **语义风险区域是否真实进入了 LoGeR 的 frame/global/SWA/TTT 计算路径？如果进入了，哪一种 path-specific action 能把局部干扰变成可持久的 trajectory improvement？**

只有这个问题回答清楚，Semantic Prior Generator 才可能从诊断模块变成真正推动 Target-30 的 memory controller。

# ACL2 v22 实验计划：Durable Context-Source Skip、Semantic All-Memory Role 与 TTT 持久写入策略

日期：2026-05-21  
对象：LoGeR / HMC Pipeline v2 / KITTI01 / ACL2 C23 读控制 / TTT 写入与多记忆 source policy  
当前线上可计数 best：`C9_P0_R2`, `ATE = 33.7629421029m`  
新目标：`KITTI01 ATE <= 25.0m`  
当前 gap：

$$
33.7629421029 - 25.0 = 8.7629421029m
$$

本计划的核心立场是：v21 的 strongest diagnostic 已经证明 source skip + scale-state commit 能在 h10 短窗口显著压低病灶段，但 h15 几乎完全衰减。因此下一轮不能继续只扫阈值、beta、gamma、keep ratio；必须围绕 **durability** 设计实验，把局部 read/source correction 变成跨 h15 仍存在的 memory-state correction。

---

## 1. 当前结果的独立判断

v21 没有产生新的 deployable online TTT write result。所有 v21 rows 都是 trusted short-rollout diagnostic，不计入线上成功；没有启动 no-GT selector，也没有 full online validation。当前可计数线上 best 仍然是 `C9_P0_R2 = 33.7629421029m`。

v21 的真实进展有四点：

第一，`full_chunk_no_overlap` 已经变成真正 no-overlap，不再是 fallback。这消除了 v20 里 support 实现不可信的问题。

第二，`compact_kv` path 已接通。它不再只是对 attention logit 加大负 bias，而是真的让 Query 保持完整、Key/Value source 经过 source_keep_mask compact。这个实现更接近 VGGT4D 的核心思想：不是删除 query token，而是让高风险 token 不再作为上下文 source。

第三，Stage C exact coarse semantic group 已经能投影到 HMC/model control。也就是说，v20 的语义 blocker 从“只有 `S_tok` scalar”推进到“有 coarse semantic group role”。不过 fine sky / vegetation split 仍不可用，不能把结果解释成精确 sky-skip 已经验证。

第四，最强 diagnostic 是 `TTTSSP_02`，它在 chunk10 h10 上达到：

$$
\Delta ATE_{h10} = -2.6117631916m
$$

$$
\Delta E_{200:300,h10} = -5.0388073200m
$$

但同一候选 h15 ATE delta 只有：

$$
\Delta ATE_{h15} = -0.1846540636m
$$

定义持久性比例：

$$
D_{dur} = \frac{|\Delta ATE_{h15}|}{|\Delta ATE_{h10}| + \epsilon}
$$

则：

$$
D_{dur} \approx \frac{0.184654}{2.611763} \approx 0.071
$$

这个值远低于本计划要求的 `0.45`。所以 v21 的结论不是“找到了好 TTT 写入策略”，而是：**source skip 与 semantic/static rescue 能降低短期病灶污染，但 correction 没有被写成持久 trajectory / memory state。**

---

## 2. 问题本质：不是 cue 不够，而是 read/source correction 与 memory commit 脱节

过去几轮已经反复证明：`D_g = acl2.gg.qq.low.g2_3.past_only.headmean.robustq` 是有用的 read cue；`update_conflict_energy` 是有用的 TTT-native cue；scale-state proxy 有弱但真实的方向性；VGGT4D-style source skip 也有 h10 局部信号。现在的问题不是完全没有 cue，而是这些 cue 只在短窗口改变了当前 read/source 行为，却没有稳定改变后续记忆。

当前模型里至少有四类 memory/source：

```text
frame attention source
chunk/global attention source
SWA local source/history
TTT fast-weight write
```

如果只在 frame attention 里 skip high-D source，但 TTT/SWA/global memory 仍然把同一批区域以 native 方式写入或保留，那么 h10 的改善会在 h15 被后续 chunk 的 memory update 冲掉。这正是 v21 看到的现象。

因此，v22 的核心假设是：

> 要让 h10 correction 延续到 h15，必须让 context-source skip、semantic role、TTT write、SWA cache commit 和 global/chunk source policy 使用同一套 memory role，而不是把 skip 只作为 read-path filter。

---

## 3. 实验总目标

v22 不追求继续把 h10 delta 从 `-2.61m` 微调到 `-2.70m`。v22 的目标是建立一个能进入 full online validation 的 durable candidate。具体目标分三层。

### 3.1 诊断目标

解释 v21 strongest candidate 为什么 h10 有效、h15 衰减：

```text
h10 correction 是被后续 TTT write 覆盖？
h10 correction 是被 SWA cache 刷新覆盖？
h10 correction 是 merge/gauge 层短期收益？
semantic/static rescue 是否只保护了 read source，而没有保护 memory source？
```

### 3.2 机制目标

把 source skip 从 read filter 升级成 memory role policy：

```text
frame/global attention:
    high-risk source 不作为 context K/V

SWA:
    high-risk previous source 不进入或弱进入 history/cache

TTT:
    high-risk source 不进入长期 W_long；必要时作为 weak negative 或 short-lived correction

semantic:
    structure anchor 保护；low-value stuff 条件式 neutral/skip；不做无条件 sky/tree hard drop
```

### 3.3 指标目标

至少产生一个 short-rollout candidate 满足：

$$
\Delta ATE_{h10/h15} \le -3.0m
$$

或：

$$
\Delta E_{200:300,h10/h15} \le -5.0m
$$

且：

$$
\Delta E_{400:600} \le +1.0m
$$

同时：

$$
D_{dur} \ge 0.45
$$

只有满足这些条件，才允许进入 no-GT selector 和 full online validation。

---

## 4. 固定基线与不可混淆边界

### 4.1 固定线上边界

```text
H9_P0_R2:
    ATE = 34.1257769401m
    [200,300) = 74.409927m
    [400,600) = 44.353638m
    role = candidate parent / disease-window reference

C9_P0_R2:
    ATE = 33.7629421029m
    [200,300) = 76.102136m
    [400,600) = 41.896364m
    role = current deployable online TTT-write best by ATE

TTTSSP_02:
    role = v21 strongest short-rollout diagnostic
    not deployable online success
```

### 4.2 不允许混淆的结果类型

```text
Full online deployable:
    可计入 TTT/write success

Trusted short rollout:
    只能作为 diagnostic / oracle bank

GT-selected candidate:
    不计入 deployable success

No-GT selector:
    只有在 h10/h15 durability gate 通过后才能启动

Offline trajectory rewrite / postprocess:
    只能作为 target-direction diagnostic，不能计入 TTT success
```

---

## 5. 核心假设与实验设计

---

# H1：`past_only -> full_chunk` 不是主杠杆，但 true no-overlap support 仍需结合 skip 复验

## 假设

单独把 `C23 past_only` 改成 `full_chunk` 或 `full_chunk_no_overlap` 不足以打开 Target-25，因为 support variants 在 v21 Phase A 中 best h10/h15 ATE delta 只有约 `-0.436m`。但 full_chunk / no-overlap 可能在 source-skip 或 semantic rescue 之后发挥作用，因为 skip 后 future support 不再被 high-risk source 污染。

## 实验设计

第一步不做 full online，只在 trusted short rollout 中比较：

```text
SUP_LOCKED:
    C23 past_only

SUP_FULL_TRUE:
    C23 full_chunk_true

SUP_NO_OVERLAP:
    C23 full_chunk_no_overlap

SUP_PAST_NEAR_FUTURE12:
    C23 past_plus_near_future12
```

每个 support 只和两个 strong read/source setting 组合：

```text
setting A:
    compact_kv = off

setting B:
    compact_kv = frame_early_dg_q80_hard

setting C:
    compact_kv = frame_early_structure_rescue_dg_q80
```

优先 chunks / horizons：

```text
chunks = 6, 10, 16
horizons = h10, h15
```

## 必须记录指标

```text
support_count_mean
future_ratio_mean
weighted_future_mass
overlap_seam_support
D_g_mass
compact_keep_ratio
num_empty_source_events
ATE_delta_h10
ATE_delta_h15
E_200_300_delta_h10
E_200_300_delta_h15
E_400_600_delta_h10
E_400_600_delta_h15
durability_ratio
```

## 假设成立标准

`full_chunk` 或 no-overlap 只有在满足以下条件时才可进入后续：

```text
h15 ATE delta 至少比 past_only 好 1.0m
或者 h15 [200,300) delta 至少比 past_only 好 2.0m
且 [400,600) regression <= +1.0m
且 compact keep ratio 没有低于 0.70
```

## 不满足条件时 Codex 先尝试

如果 `full_chunk_true` 和 `full_chunk_no_overlap` 都不过 gate：

```text
1. 不再扩大 full_chunk support 矩阵；
2. 回退 C23 past_only 作为 locked cue；
3. 只保留 full_chunk_no_overlap 作为 semantic/global support diagnostic；
4. Codex 转去 H2/H3，不允许继续 support 微扫。
```

如果 `full_chunk_no_overlap` 实现出现异常：

```text
1. 先打印每帧 support index list；
2. 验证 overlap seam support 是否为 0；
3. 验证 future mass 是否符合预期；
4. 不准用 fallback full_chunk 结果冒充 no-overlap。
```

---

# H2：VGGT4D-style source skip 必须从 bias 近似升级为 durable compact-K/V memory policy

## 背景判断

VGGT4D 的启发不是“删掉动态 query”，而是动态区域不应作为 early-stage context source。LoGeR v21 已经实现 compact_kv，Query 保持完整，K/V source 被 compact。v21 smoke 中没有 empty source event，说明工程路径可用。

v20/v21 已证明 source skip 有 h10 局部收益，但 h15 衰减。H2 要验证：如果把 compact K/V 的 source-role 同步到 memory commit，是否能获得 durability。

## 假设

`read-only source skip` 只能产生短期 h10 收益；`skip-aware memory commit` 才能让 h15 保留 correction。

## 实验分组

### H2-A：读路径 compact_kv 对照

```text
KVC_READ_01:
    frame early dg_q80 compact_kv hard

KVC_READ_02:
    frame early dg_q90 compact_kv hard

KVC_READ_03:
    chunk early dg_q80 compact_kv hard

KVC_READ_04:
    frame+chunk early dg_q80 compact_kv hard
```

这些只验证 read/source skip，不改变 TTT/SWA commit。

### H2-B：skip-aware TTT write

把 source_keep_mask 映射到 TTT write role：

```text
source kept & structure/lowD:
    positive write

source skipped & highD:
    neutral or weak negative

source skipped & semantic lowstuff:
    neutral by default, not hard negative
```

候选：

```text
KVC_TTT_01:
    read compact_kv + TTT skip tokens neutral

KVC_TTT_02:
    read compact_kv + TTT skipped highD weak negative

KVC_TTT_03:
    read compact_kv + TTT structure kept boost

KVC_TTT_04:
    read compact_kv + TTT source_keep_mask gates stage_d_x_dg_inv_sqrt
```

### H2-C：skip-aware SWA/global source persistence

虽然本轮重点仍是 TTT，但 v21 的 h15 decay 可能来自 SWA/global source refresh。因此需要最小化测试：

```text
KVC_MEM_01:
    read compact_kv + SWA history excludes skipped highD source

KVC_MEM_02:
    read compact_kv + SWA history downweights skipped highD source

KVC_MEM_03:
    read compact_kv + global/chunk source skip in early global blocks

KVC_MEM_04:
    read compact_kv + TTT skip-aware + SWA skipped source downweight
```

## 必须记录指标

```text
frame_keep_ratio
chunk_keep_ratio
global_keep_ratio
swa_keep_ratio
num_empty_source_events
special_token_kept_ratio
structure_kept_ratio
lowstuff_skipped_ratio
TTT_positive_mass
TTT_neutral_mass
TTT_negative_mass
TTT_update_norm_w0/w1/w2
SWA_history_token_count
SWA_history_skipped_mass
ATE_delta_h10/h15
[200,300]_delta_h10/h15
[400,600]_delta_h10/h15
durability_ratio
```

## 通过标准

H2 通过要求：

```text
h15 ATE delta <= -1.5m
且 h15 [200,300) delta <= -3.0m
且 durability_ratio >= 0.35
```

进入 selector/full 前的强 gate：

```text
h10/h15 ATE delta <= -3.0m
或 h10/h15 [200,300) delta <= -5.0m
且 [400,600) regression <= +1.0m
且 durability_ratio >= 0.45
```

## 不满足条件时 Codex 先尝试

如果 h10 有收益但 h15 衰减：

```text
1. 先运行 state attribution：比较 h10->h15 的 TTT state diff、SWA history diff、global source summary；
2. 若 TTT state 覆盖掉 correction，转 KVC_TTT_*；
3. 若 SWA history 覆盖掉 correction，转 KVC_MEM_*；
4. 若 merge/gauge 指标变化最大，标记为 trajectory-state issue，不继续 source skip 微扫。
```

如果 h10 也没有收益：

```text
1. 降低 hard skip 阈值强度，测试 q70/q80/q90 中是否 q80 最优；
2. 加 structure rescue，避免 over-skip；
3. 若仍无 h10 作用，停止该 skip mask，转 semantic role。
```

如果出现 empty source event：

```text
1. 强制 special tokens keep；
2. 对每个 query 保证 min_source_count；
3. fallback to bias mode 只用于 smoke，不计入正式 compact_kv 结论。
```

---

# H3：语义必须做 all-memory role，不应只作为 TTT scalar prior

## 假设

语义不是一个统一的 `S_tok` 标量，而是 memory-role assignment。不同语义区域应在不同 memory 中承担不同角色：

```text
structure anchor:
    作为 source 和 long-term write positive

low-value stuff:
    默认 neutral；highD 时可 source-skip 或 weak negative

sky:
    不应无条件 hard skip；可作为 horizon/scale neutral carrier

vegetation/tree:
    highD 时 skip/negative；lowD 时 neutral

movable thing:
    source-skip 和 short-lived memory，不进 long-term TTT
```

当前 v21 只有 coarse 5-group，fine sky/vegetation split 还不可用。因此 H3 分两步。

## H3-A：coarse semantic group role

使用已接通的 coarse groups：

```text
STRUCTURE_ANCHOR
LOW_VALUE_STUFF
MOVABLE_THING
STATIC_THING
UNCERTAIN_REGION
```

候选：

```text
SEM_ROLE_01_structure_rescue:
    structure always source-keep; highD structure 不 hard skip，只降权

SEM_ROLE_02_lowstuff_highD_skip:
    lowstuff ∩ highD compact source skip

SEM_ROLE_03_lowstuff_all_skip_diagnostic:
    lowstuff 全 skip，只做 diagnostic，预期可能伤 scale

SEM_ROLE_04_structure_positive_ttt:
    structure ∩ lowD -> positive TTT write
    lowstuff ∩ highD -> neutral / weak negative

SEM_ROLE_05_all_memory_role:
    frame/global/SWA source policy + TTT positive/neutral/negative role 同步
```

## H3-B：fine semantic split instrumentation

如果 fine label 已在 MaskletOutput 中存在但没有进入 HMC：

```text
1. 将 L_sem fine label 映射到 per-token L_sem_tok；
2. 新增 exact masks: sky, vegetation, road, building, wall, fence, car/person if available；
3. 输出 semantic_group_summary.jsonl 与 fine_semantic_summary.jsonl；
4. 不允许用 S_tok <= 0.45 冒充 sky/vegetation。
```

候选：

```text
FINE_SEM_01_sky_neutral:
    sky 不作为 positive，不作为 negative，source keep ratio >= 0.5

FINE_SEM_02_sky_highD_skip:
    sky ∩ highD source skip，sky ∩ lowD keep neutral

FINE_SEM_03_vegetation_highD_skip:
    vegetation ∩ highD source skip / weak negative

FINE_SEM_04_structure_lowD_write:
    road/building/wall/fence ∩ lowD positive TTT write

FINE_SEM_05_semantic_all_memory:
    fine semantic role 同步 frame/global/SWA/TTT
```

## 记录指标

```text
per_group_coverage
per_group_keep_ratio
per_group_skip_ratio
per_group_TTT_positive_mass
per_group_TTT_negative_mass
per_group_SWA_history_mass
per_group_attention_source_mass
per_group_update_norm_w0/w1/w2
per_group_h10_delta
per_group_h15_delta
sky_keep_ratio
vegetation_keep_ratio
structure_rescue_ratio
```

## 通过标准

```text
SEM_ROLE candidate 通过：
    h15 ATE delta <= -1.5m
    或 h15 [200,300) delta <= -3m
    且 [400,600) regression <= +1m

fine semantic 通过：
    相比 coarse semantic，h15 durability_ratio 提升 >= 0.15
    且没有出现 sky/vegetation hard skip 导致的 scale collapse
```

## 不满足条件时 Codex 先尝试

如果 lowstuff skip 伤 h15：

```text
1. 将 lowstuff 从 negative 改为 neutral；
2. 只对 lowstuff ∩ highD ∩ high_unc 执行 skip；
3. 强制 sky lowD source keep；
4. 对 vegetation/tree 单独拆分后再判断。
```

如果 structure rescue 导致 h10 收益下降：

```text
1. structure highD 不强 rescue，只 rescue structure lowD；
2. 保护比例从 1.0 降到 0.5；
3. 只在 frame/global read 中 rescue，不进入 TTT positive。
```

如果 semantic group coverage 异常：

```text
1. 先跑 passive semantic overlay；
2. 检查 chunk10 / chunk6 / chunk16 每帧 group coverage；
3. 如果某组 coverage < 5%，不要基于该组做 full candidate。
```

---

# H4：source skip 的 h15 衰减来自 memory overwrite，需要做 skip-aware commit 或生命周期分离

## 假设

`TTTSSP_02` 的 h10 强而 h15 弱，是因为后续 chunks 的 TTT/SWA/global write 覆盖了 h10 correction。解决方式不是更强 skip，而是让 skip-aware correction 具有合适生命周期。

## 实验设计

### H4-A：h10-to-h15 state attribution

固定 `TTTSSP_02` 和 H9 parent，记录 h10/h15 中间 state：

```text
state at chunk10 input
state at chunk10 output
state at h10 endpoint
state at h15 endpoint
```

比较：

```text
TTT W_long diff
TTT W_short / transient diff if any
SWA history diff
frame/global source summary diff
merge/gauge cursor diff
```

计算：

$$
R_{overwrite}^{TTT} = \frac{\|W_{h15} - W_{h10}\|}{\|W_{h10} - W_{base}\| + \epsilon}
$$

$$
R_{overwrite}^{SWA} = \frac{\|H^{SWA}_{h15} - H^{SWA}_{h10}\|}{\|H^{SWA}_{h10} - H^{SWA}_{base}\| + \epsilon}
$$

### H4-B：skip-aware write-probe TTT commit

```text
TTT_DUR_01:
    read compact_kv only

TTT_DUR_02:
    read compact_kv + TTT write cache generated under skip mask

TTT_DUR_03:
    read compact_kv + native read output + skip-aware TTT replay only

TTT_DUR_04:
    read compact_kv + post-zp skip basis routing
```

### H4-C：W_long / W_short memory lifecycle

```text
TTT_LIFE_01:
    skipped highD correction -> W_short, decay K=2
    structure lowD -> W_long

TTT_LIFE_02:
    skipped highD correction -> W_short, decay K=4
    structure lowD -> W_long

TTT_LIFE_03:
    lowstuff highD -> W_short only
    source kept structure -> W_long

TTT_LIFE_04:
    scale-state correction -> W_long
    highD skip correction -> W_short
```

## 指标

```text
ATE_delta_h10/h15
[200,300]_delta_h10/h15
[400,600]_delta_h10/h15
durability_ratio
TTT_overwrite_ratio
SWA_overwrite_ratio
W_long_norm
W_short_norm
W_short_decay_curve
TTT_update_cos_to_H9
SWA_history_keep_ratio
```

## 成立标准

H4 通过要求：

```text
TTT_DUR or TTT_LIFE candidate:
    h15 ATE delta <= -2.0m
    and durability_ratio >= 0.45
    and [400,600) regression <= +1m
```

强通过：

```text
h15 [200,300) delta <= -5m
或 full online selector candidate predicted ATE <= 31.5m
```

## 不满足条件时 Codex 先尝试

如果 h15 correction 被 TTT overwrite：

```text
1. 降低后续 chunks 的 highD TTT write；
2. 加 structure positive continuity；
3. 尝试 W_long/W_short 分离；
4. 若仍失败，转 explicit trajectory-state module。
```

如果 h15 correction 被 SWA overwrite：

```text
1. skip-aware SWA cache commit；
2. 保留 structure source，弱化 lowstuff highD source；
3. 对 SWA 只做 cache write，不动 read；
4. 若仍失败，SWA 作为辅助，不再主导。
```

如果 merge/gauge state 主导：

```text
1. 不再把该问题归因于 TTT write；
2. 进入 online trajectory-state / scale-state 模块分支；
3. TTT write 只保留 regularizer。
```

---

# H5：如果 TTT / source skip 都不能产生 h15 durability，就必须引入显式 online trajectory-state / scale-state

## 假设

Target-25 主杠杆可能不是 memory-source filtering，而是显式的 online scale/trajectory state。NOGTPOSE 类结果表明 no-GT trajectory-state proxy 可以达到 target 区间，但目前是后处理，不可计入 TTT success。H5 要验证是否能把该方向在线化，不做后处理。

## 实验设计

只有在 H2-H4 都无法通过 h15 durability gate 时启动。

### H5-A：online scale-state module smoke

```text
输入：
    current predicted pose increments
    reset-window step length EMA
    overlap geometry scale proxy
    TTT scale-state risk

输出：
    window-level scale multiplier s_m

作用点：
    before trajectory merge / gauge commit
    not offline rewrite
```

### H5-B：TTT-conditioned scale state

```text
SCALE_ONLINE_01:
    s_m from pose-step EMA only

SCALE_ONLINE_02:
    s_m from pose-step EMA + TTT conflict energy

SCALE_ONLINE_03:
    s_m from pose-step EMA + semantic structure coverage

SCALE_ONLINE_04:
    s_m from pose-step EMA + context skip keep ratio
```

## 指标

```text
full online ATE
Rot
FinalErr
[200,300]
[400,600]
Sim3Scale
per-reset local scale
scale_multiplier_history
scale_update_smoothness
RPE_t/r
```

## 成立标准

```text
Stage success:
    full online ATE <= 31.5m
    and [200,300) <= 60m
    and [400,600) does not regress > +2m

Target success:
    full online ATE <= 25m
```

## 不满足条件时 Codex 先尝试

如果 online scale-state 过强导致 Rot/FinalErr 崩：

```text
1. 降低 scale update clamp；
2. 只在 reset-window boundary update；
3. 对 body/exit 采用不同 smoothing；
4. 加 structure coverage gate。
```

如果 scale-state 改善后段但不改善 [200,300)：

```text
1. body window 使用 separate scale accumulator；
2. 与 compact_kv skip 的 h10 correction 结合；
3. 对 chunks 6-10 单独打开 short-lived scale correction。
```

---

## 6. 并行执行与加速策略

v22 采用 sandbox-first，不允许 full-first。

### 6.1 Codex 并行 tracks

```text
Track A: Support / full_chunk / no-overlap
    负责人：Codex-A
    先做 support audit + h10/h15 short rollout

Track B: compact_kv durable source skip
    负责人：Codex-B
    做 read-only compact_kv、skip-aware TTT、skip-aware SWA/global

Track C: semantic all-memory role
    负责人：Codex-C
    做 coarse group role；若 possible 补 fine sky/vegetation split

Track D: h10->h15 durability attribution
    负责人：Codex-D
    记录 TTT/SWA/merge/gauge state overwrite ratio

Track E: online trajectory-state fallback
    负责人：Codex-E
    只有 H2-H4 failure gate 触发后启动
```

### 6.2 执行顺序

第一批只跑：

```text
A: support variants × compact off/on
B: compact_kv read-only strongest 4 rows
C: semantic group passive audit + SEM_ROLE_01/02
D: TTTSSP_02 h10/h15 state attribution
```

第二批根据第一批 gate 决定：

```text
如果 B h10 有效但 h15 弱：
    跑 KVC_TTT_* 和 KVC_MEM_*

如果 C semantic role 有效：
    跑 SEM_ROLE_04/05

如果 D 显示 TTT overwrite：
    跑 TTT_DUR_* / TTT_LIFE_*

如果 D 显示 merge/gauge 主导：
    跑 SCALE_ONLINE smoke
```

### 6.3 禁止项

```text
禁止继续单独扫:
    beta 4.70/4.75/4.80
    q78/q80/q82
    gamma 0.0048/0.0050/0.0052
    lowstuff scalar 0.65/0.70/0.75

除非这些微调依附于已经通过 h15 durability gate 的机制。
```

---

## 7. 统一记录格式

每个 candidate 必须输出：

```text
candidate_manifest.json
run_config.yaml
context_skip_summary.jsonl
semantic_group_summary.jsonl
support_index_summary.csv
hmc_state_hash.jsonl
ttt_write_debug.jsonl
swa_history_debug.jsonl
trajectory_diagnostics.json
candidate_vs_H9_delta_by_horizon.csv
true_action_gate_summary.json
```

必要字段：

```text
candidate_id
parent_run
chunk
horizon
support_mode
context_source_skip_impl
context_source_skip_mask
semantic_role_mode
memory_scope
ATE
Rot
FinalErr
RPE_t
RPE_r
E_200_300
E_400_600
ATE_delta_vs_H9
E_200_300_delta_vs_H9
E_400_600_delta_vs_H9
keep_ratio_frame
keep_ratio_global
keep_ratio_swa
semantic_structure_mass
semantic_lowstuff_mass
TTT_positive_mass
TTT_negative_mass
TTT_neutral_mass
durability_ratio
selector_allowed
full_online_allowed
counts_as_online_ttt_write_success
```

---

## 8. 必须可视化

### 8.1 h10/h15 durability dashboard

每个 candidate 画：

```text
ATE_delta_h10 vs ATE_delta_h15 scatter
[200,300]_delta_h10 vs [200,300]_delta_h15 scatter
durability_ratio bar plot
[400,600] regression bar plot
```

### 8.2 Context skip map

对 chunk10、chunk6、chunk16 输出：

```text
RGB
D_g
compact source keep mask
skipped source mask
structure rescue mask
semantic lowstuff mask
semantic structure mask
final K/V source map
```

### 8.3 Memory role overlay

```text
TTT positive / neutral / negative token map
SWA kept / downweighted source map
frame/global source mass map
per-group role heatmap
```

### 8.4 State overwrite dashboard

```text
TTT overwrite ratio over chunks
SWA overwrite ratio over chunks
merge/gauge diff over chunks
W_long/W_short norm over chunks
```

---

## 9. Promotion 与停止规则

### 9.1 进入 no-GT selector 的条件

必须满足：

```text
h10/h15 ATE delta <= -3m
或 h10/h15 [200,300) delta <= -5m
且 [400,600) regression <= +1m
且 durability_ratio >= 0.45
```

### 9.2 进入 full online 的条件

必须满足：

```text
no-GT selector Spearman(proxy, ATE_delta) >= 0.45
selector chosen candidates h15 ATE delta <= -2m on at least 2 chunks
no empty source event
no semantic coverage anomaly
```

### 9.3 停止某一族的条件

```text
同一 action family 连续 6 条 short rollout：
    h10 ATE delta > -1m
    或 h15 ATE delta > -0.5m
则停止该族。

同一 action family 出现：
    [400,600) regression > +2m in 3 rows
则停止该族或加入 continuity protection。

如果 H2/H3/H4 全部无法达到 durability_ratio >= 0.25：
    TTT / source skip 降级为 regularizer；启动 H5 online trajectory-state module。
```

---

## 10. 本轮最重要的预期结论

v22 最重要的不是立刻出 Target-25，而是回答：

```text
1. source skip 是否只能短期读修正？
2. 语义 role 是否能把 source skip 变成 memory write policy？
3. TTT/SWA/global 哪一类 memory 覆盖了 h10 correction？
4. 是否存在 h15 durable candidate？
5. 如果不存在，是否应把 Target-25 主线转向 explicit online trajectory-state / scale-state？
```

如果 v22 仍然只得到 h10 强、h15 弱的结果，那么结论应当非常明确：

> 当前 memory-source filtering 和 TTT write interface 不足以产生 Target-25 所需的 persistent trajectory correction；TTT 应保留为 regularizer，主线转向显式 online trajectory-state / scale-state module。


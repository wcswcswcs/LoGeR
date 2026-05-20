# ACL2 v20 实验计划：TTT 写入重启、VGGT4D-style Context Skip、full_chunk support 与语义全记忆策略

日期：2026-05-20 / 2026-05-21 之后  
对象：LoGeR / HMC Pipeline v2 / KITTI01 / ACL2 attention cue / Stage C-D semantic prior / TTT-SWA-frame/global memory control  
当前目标：KITTI01 online deployable ATE $\le 25.0m$  
当前最好可计数 online TTT/HMC：`C9_P0_R2 = 33.7629421029m`，但 `[200,300)` 更差，不是健康的 Target-25 方向。  
当前主要 locked read cue：`acl2.gg.qq.low.g2_3.past_only.headmean.robustq`，简称 `C23 past`。

---

## 0. 本轮实验前的独立判断

v19 没有找到好的 TTT 写入策略。它的价值在于确认了一个重要边界：**trajectory / scale-state 信号确实比纯 token-level dynamic prior 更接近目标，但当前将 scale-state proxy 接入 TTT write 的方式仍然不够强、不够持久**。

当前 v19 最强短滚动 TTT-write 信号是：

```text
SCALECOMMIT_01_PZBASIS_HARM_W0_G025
chunk10 h10 ATE delta vs H9 = -2.2244m
chunk10 h10 [200,300) delta vs H9 = -2.2990m
chunk10 h10 [400,600) proxy delta = -3.1705m
```

但它没有通过正式 gate：

```text
required h10/h15 ATE delta <= -3m
or [200,300) delta <= -5m with downstream regression <= +1m

actual h15 ATE delta = -0.8577m
actual h15 [200,300) delta = -1.6331m
```

因此，这不是一个可进入 no-GT selector 或 full online validation 的候选。它只能作为 weak-positive anchor。

我的判断是：

```text
TTT cue / scale-state cue 有信号，方向没有死。
但当前 TTT write action 仍然太像局部 fast-weight regularization。
它能 nudging trajectory-state error，却不能稳定控制 whole-scene drift。
```

这解释了为什么进度显得慢：从 v8/v9 到 v19，很多实验都在 `33.7m-34.2m` 区间内震荡，而目标是 `25m`。当前 gap 仍然约为：

$$
33.7629421029 - 25.0 = 8.7629421029m
$$

这个 gap 不是继续微扫 `gamma`、`neutral`、`carrier alpha` 或 `chunk10 h10/h15` 能补上的。下一轮必须从 action interface 上升级：不只问“TTT 写多写少”，而是问“哪些 token / semantic / memory source 应该作为上下文来源，哪些应该跳过计算，哪些应该写入长期 memory，哪些只能短期存在”。

---

## 1. 本轮整体目标

v20 不再只围绕 TTT scalar write 微调。整体目标是并行验证四个更根本的问题：

1. **C23 past 的 support 选择是否限制了 read cue**  
   验证 `past_only` 是否应改成 `full_chunk`，尤其是在 `pair/all` read intervention 下。过去 `full` 和 `past_only` 的比较主要发生在 earlier read-layer 或不同协议下；现在必须在当前主协议下重新验证。

2. **VGGT4D-style context-source skip 是否优于 attention bias**  
   当前 LoGeR 主要是通过 attention bias 操作 attention map。VGGT4D 的更关键思想不是“加 bias”，而是在早期层让 dynamic / unreliable patch 不再作为 K/V 上下文来源。LoGeR 应该验证：对 `D_g` 或 semantic mask 标出的区域，保留 Query，但从 frame/global attention 的 K/V source 中移除，是否比 pair bias 更有效。

3. **语义能否作为所有 memory/source 的写入与读取策略**  
   不再只把 semantic prior 当 TTT scalar multiplier。语义应参与四类记忆/上下文路径：

   ```text
   frame attention: 当前 chunk 内每帧 token 的 K/V source eligibility
   global attention: 当前 chunk 全局 token 的 K/V source eligibility
   SWA: previous/current overlap source cache eligibility
   TTT: long-term fast-weight write eligibility / positive-neutral-negative replay
   ```

4. **TTT 是否仍有可部署的 Target-25 action space**  
   保留 `SCALECOMMIT_01` 作为 weak-positive TTT anchor，但不直接进入 full validation。必须先通过 h10/h15 sandbox gate；如果失败，Codex 应自动转向 context-skip / semantic-memory / trajectory-state module，而不是继续微调 TTT scalar。

---

## 2. 关键原则：不要误读 VGGT4D

VGGT4D 的方法不能被简化成“完全删除动态 token”。更准确地说，它做的是 **early-stage context-source removal**：动态 patch 仍然作为 Query 存在，仍然会经过 embedding、QKV 投影和后续网络；但在早期 attention 中，它不再作为其他 token 可读取的 Key/Value 来源。

LoGeR 里的对应实现也应采用这个保守原则：

$$
O_i^l = \operatorname{Attn}(Q_i^l, K_{\Omega_l}^l, V_{\Omega_l}^l)
$$

其中：

$$
\Omega_l = \{j \mid M_j^l = 0\}
$$

$M_j^l=1$ 表示 token $j$ 在第 $l$ 层被判定为不应作为上下文来源。Query 不删除，special token 不删除，至少第一轮不删除 register / camera / role token。

这个设计与 attention bias 的区别是：

```text
attention bias:
    仍然让高风险 token 存在于 K/V 集合中，只是降低 logits。

context-source skip:
    直接从 K/V source 集合中移除高风险 token，防止它们成为上下文提供者。
```

VGGT4D 还提示两个边界：

```text
1. full-mask all layers 是危险的；早期层选择性 masking 才是合理干预。
2. dynamic / semantic mask 需要先变成 patch-level mask，并且 special tokens 应默认保留。
```

因此 LoGeR 的 v20 skip 计划必须从 **early / selected layers / source K-V only / soft fallback** 开始，而不是全层 hard mask。

---

## 3. 实验总协议与固定边界

### 3.1 固定线上基线

每个阶段都必须记录以下参考：

```text
H9_P0_R2:
    ATE = 34.1257769401
    Rot = 6.5414
    [200,300) = 74.409927
    [400,600) = 44.353638
    counts_as_ttt_write = true

C9_P0_R2:
    ATE = 33.7629421029
    Rot = 6.5259
    [200,300) = 76.102136
    [400,600) = 41.896364
    counts_as_ttt_write = true

WINGAM_P0_R3:
    ATE = 34.1902782732
    Rot = 6.5666
    [200,300) = 75.576021
    [400,600) = 42.280485
    counts_as_ttt_write = true
```

解释：

```text
C9 是当前可计数 online ATE best，但 [200,300) 更差。
H9 是更健康的 candidate parent，因为 [200,300) 更好。
WINGAM 是 update_conflict_energy + tri-replay historical anchor。
```

### 3.2 固定 pipeline 协议

除特别说明外，默认：

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read mode = frame attention pair/all
beta = 4.75
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
SWA = current SWKS3 fixed protocol
RESET_EVERY = 5
```

### 3.3 实验类型边界

每条 run 必须标注：

```text
counts_as_online_ttt_write_success: true / false
diagnostic_only_short_rollout: true / false
uses_gt_runtime_action: true / false
uses_offline_postprocess: true / false
uses_semantic_cache: true / false
uses_context_skip: true / false
```

只有满足以下条件的 run 才能被计为可部署 online 成功：

```text
full KITTI01 online run
no GT runtime action
no offline trajectory rewrite
commit affects future HMC state
state/cfg hash valid
```

---

## 4. 核心假设与实验设计

---

# H1：`past_only` 可能不是当前 `C23` 的最优 support，应重新验证 `full_chunk`

## 4.1 假设

`C23 past` 是当前 locked cue，但它可能是一个历史阶段的最优选择。进入 `pair/all` read intervention、VGGT4D-style context-source skip 和 semantic memory policy 后，support 需求可能改变。

特别是：

```text
past_only:
    causal / streaming 更干净，但可能缺少当前 chunk 内未来帧提供的静态 manifold。

full_chunk:
    使用当前 chunk 内所有其它帧作为 support，可能更适合 LoGeR chunk-wise bidirectional reasoning；
    但也可能引入 future leakage 或把短时动态一致性误认为静态。
```

本假设认为：`full_chunk` 不应被历史结果直接否定，必须在当前 pair/all + SWKS3 + TTT write protocol 下重测。

## 4.2 需要实现的 support variants

```text
C23_past:
    existing acl2.gg.qq.low.g2_3.past_only.headmean.robustq

C23_full_chunk:
    support = all frames in current chunk except target frame

C23_full_chunk_no_overlap:
    full_chunk support but exclude external overlap seam frames if defined

C23_past_plus_future_light:
    support score = 0.75 * past + 0.25 * future-within-chunk

C23_bidirectional_near:
    near12 or near24 symmetric support, already partly exists but must rerun under current protocol
```

如果代码里 `full` 已经等价于 full_chunk，则 Codex 需要先做 support-index audit：

```text
for each frame t:
    print support frame ids for past_only
    print support frame ids for full
    print support frame ids for full_chunk
    print support frame ids for near12 / near24
```

如果 `full` 与 `full_chunk` 不等价，保留两个名字。若等价，统一命名为 `full_chunk` 并在 config 里写明。

## 4.3 第一批实验

短滚动优先，不直接 full。固定 parent 为 H9，chunks 6 和 10，horizons 5/10/15：

```text
S1_00: C23_past baseline
S1_01: C23_full_chunk
S1_02: C23_full_chunk_no_overlap
S1_03: C23_past_plus_future_light
S1_04: C23_bidirectional_near24
```

通过 short rollout 后，最多 2 条进入 full KITTI01。

## 4.4 必须记录的指标

```text
support_indices.jsonl
D_g_mean / D_g_p90 / D_g_mass_gt_0.5
D_g_temporal_entropy
D_g_olddyn_corr
anchor_collision
semantic_group_D_g_mean
attention_mass_to_high_D_before_after
ATE_h5 / ATE_h10 / ATE_h15
[200,300)_h5/h10/h15
[400,600)_proxy_h10/h15
Rot_h10/h15
FinalErr_h10/h15
```

Full run 还要记录：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
ATE_50_mean/worst
ATE_100_mean/worst
ATE_200_mean/worst
per-reset-group ATE
```

## 4.5 成立标准

H1 通过条件：

```text
strong:
    full_chunk full online ATE <= C9 - 0.5m
    and [200,300) <= H9 [200,300) - 3m
    and [400,600) regression <= +1m

stage:
    short rollout h10/h15 ATE delta <= -3m
    or [200,300) delta <= -5m with downstream regression <= +1m

weak:
    full_chunk does not beat C9 but improves [200,300) by >= 2m and does not hurt overall ATE by > 0.2m
```

如果 `full_chunk` 没有短滚动信号：

```text
Codex should stop full_chunk full runs;
try past_plus_future_light and near24 only;
then return support to C23_past baseline.
```

---

# H2：VGGT4D-style context-source skip 可能比 attention bias 更适合 LoGeR read path

## 5.1 假设

当前 pair/all attention bias 已经非常强，但它仍然让 high-risk tokens 作为 K/V source 存在。VGGT4D 的核心启发是：只在早期层，把 dynamic tokens 从 K/V source 集合中拿掉，让所有 Query 只能从 static source 聚合信息。

LoGeR 也可能受益于这种机制，尤其是在 frame/global attention 这类 read path 中。

## 5.2 需要实现的 intervention types

### Type A：Frame-attention K/V source skip

```text
module = frame attention
query = all tokens kept
K/V = remove source tokens with M_src=1
special tokens = keep
layer scope = early / early_mid / all_diagnostic
```

### Type B：Global/chunk attention K/V source skip

```text
module = global/chunk attention if hook exists
query = all tokens kept
K/V = remove source tokens with M_src=1
special tokens = keep
layer scope = g2_3_neighborhood / early global / all diagnostic
```

### Type C：Bias + K/V skip hybrid

```text
low risk tokens:
    keep normal pair/all bias
high risk tokens:
    K/V skip
```

### Type D：Soft source skip / top-k keep fallback

如果 hard K/V skip 导致 empty source 或 OOD：

$$
K'_j = g_j K_j, \quad V'_j = g_j V_j
$$

其中：

$$
g_j = 1 - \rho M_j
$$

但第一版优先验证 hard K/V source removal，因为它更接近 VGGT4D。

## 5.3 Mask source variants

```text
Dg_high:
    M_src = 1[D_g > q_80]

Dg_high_strict:
    M_src = 1[D_g > q_90]

semantic_lowstuff:
    sky / vegetation / grass / water / reflection

semantic_conditional:
    LOW_VALUE_STUFF ∩ high_Dg

semantic_movable:
    person / car / bus / truck / bicycle if present

semantic_sky_only:
    sky only, diagnostic

semantic_sky_highD:
    sky ∩ high_Dg, safer than sky_only
```

注意：sky / vegetation 不得第一轮 hard 全删。v6/v7 已经显示 low-value stuff hard downweight 很容易伤 ATE，因此 semantic skip 必须分为 unconditional 与 conditional 两种。

## 5.4 第一批实验矩阵

短滚动阶段：

```text
KVS_00: pair/all bias baseline, C23 past
KVS_01: frame early K/V skip, mask=Dg_high q80
KVS_02: frame early K/V skip, mask=Dg_high q90
KVS_03: frame early K/V skip, mask=LOW_VALUE_STUFF ∩ highD
KVS_04: frame early K/V skip, mask=sky_only
KVS_05: frame early K/V skip, mask=sky ∩ highD
KVS_06: frame early K/V skip + pair/all bias, mask=Dg_high q90
KVS_07: global/chunk early K/V skip, mask=Dg_high q90
KVS_08: frame+global K/V skip, mask=LOW_VALUE_STUFF ∩ highD
```

如果 `KVS_04 sky_only` 伤害严重，Codex 不得继续扩展 unconditional lowstuff skip；转向 conditional skip。

## 5.5 必须记录的指标

```text
num_query_tokens
num_source_tokens_before
num_source_tokens_after
source_keep_ratio
per_frame_source_keep_ratio
per_semantic_group_keep_ratio
special_token_kept_count
empty_source_events
attention_entropy_before_after
attention_mass_to_skipped_tokens_before
attention_mass_to_highD_before_after
hidden_norm_before_after
pose_delta_vs_baseline
ATE_h5/h10/h15
[200,300)_h5/h10/h15
[400,600)_proxy
```

## 5.6 可视化

必须生成：

```text
RGB
D_g map
semantic group map
M_src skip map
attention mass before/after to skipped source
source keep ratio by frame
source keep ratio by semantic group
trajectory h10/h15 overlay
per-chunk error curve
```

## 5.7 成立标准

H2 通过条件：

```text
strong:
    K/V skip full online ATE <= 32.5m
    or short h10/h15 ATE delta <= -4m and [400,600) regression <= +1m

stage:
    h10/h15 ATE delta <= -3m
    or [200,300) delta <= -5m with [400,600) regression <= +1m

weak:
    improves Rot/FinalErr but not ATE; keep only as auxiliary if ATE regression <= 0.2m
```

失败分流：

```text
if hard K/V skip hurts ATE by > 1m:
    try soft source gate rho=0.25/0.50, keep special, restrict to layers 0-2.

if sky_only fails but sky_highD helps:
    semantic skip must be conditional, not unconditional.

if frame K/V skip works but global K/V skip fails:
    keep skip only in frame attention, not global attention.

if K/V skip only improves Rot:
    combine with TTT scale-state action only after support gate passes; do not full-expand.
```

---

# H3：语义应该被定义为 memory-source role，不是单一 write scalar

## 6.1 假设

语义不是简单的 `V_sem` scalar。更合理的是把语义分配到 memory roles：

```text
STRUCTURE_ANCHOR:
    road / building / wall / fence / lane / sidewalk
    role = positive source / positive write anchor

LOW_VALUE_STUFF:
    sky / vegetation / grass / water / reflection
    role = neutral or conditional skip, not hard negative by default

MOVABLE_THING:
    car / person / bicycle / bus / truck
    role = dynamic source skip / weak or negative TTT write if highD

UNCERTAIN:
    low trust mask / fragmented / low confidence
    role = fallback to geometry-only, not semantic hard rule
```

## 6.2 Memory-path roles

### Frame attention memory-source role

```text
semantic structure:
    keep as K/V source

semantic lowstuff:
    keep if lowD; skip or soft gate if highD

semantic movable:
    skip as K/V source in early layers
```

### Global attention memory-source role

```text
structure:
    keep

sky:
    optional skip only in early global layers; never all layers initially

vegetation:
    conditional skip if highD or high uncertainty
```

### SWA source/cache role

本轮不以 SWA 为主，但语义策略要记录，不一定改动。可先只做 audit：

```text
previous-source semantic group mass
SWA attention mass to sky / vegetation / structure / movable
boundary error vs source semantic composition
```

只有 semantic source audit 显示 SWA source mass 与 `[200,300)` 或 boundary error 高相关，才启动 SWA modification。

### TTT write role

```text
positive:
    structure ∩ lowD ∩ high confidence

neutral:
    sky / vegetation / grass ∩ lowD

negative:
    highD ∩ lowstuff
    highD ∩ movable
    high uncertainty ∩ semantic low trust
```

TTT tri-replay role：

$$
G_{commit}=G_{pos}+\lambda_{neu}G_{neu}-\gamma G_{neg}
$$

## 6.3 第一批语义实验

```text
SEM_00: semantic noop cache repeat
SEM_01: frame K/V skip, sky_only
SEM_02: frame K/V skip, sky_highD
SEM_03: frame K/V skip, lowstuff_highD
SEM_04: frame K/V skip, movable_highD
SEM_05: TTT tri-replay semantic pos/neu/neg, no K/V skip
SEM_06: frame K/V skip lowstuff_highD + TTT semantic pos/neu/neg
SEM_07: global K/V skip lowstuff_highD only
SEM_08: all memory audit only, no modification
```

注意：`SEM_01 sky_only` 是风险验证，不得直接作为大矩阵扩展基础。

## 6.4 必须记录的指标

```text
semantic_coverage_total
semantic_coverage_by_group
semantic_group_Dg_mean
semantic_group_unc_mean
semantic_group_write_score_mean
semantic_group_TTT_update_norm_w0/w1/w2
semantic_group_negative_mass
semantic_group_attention_source_keep_ratio
semantic_group_attention_mass_before_after
semantic_group_SWA_source_mass
semantic_group_frame_count
```

Trajectory：

```text
ATE / Rot / RPE / FinalErr / YawRMSE / Sim3Scale
[200,300), [200,400), [400,600)
per-reset-group ATE
per-semantic-group regression map
```

## 6.5 成立标准

语义策略通过标准：

```text
strong:
    full online ATE <= 32.5m
    and [200,300) improves >= 5m
    and no [400,600) regression > +1m

stage:
    short h10/h15 ATE delta <= -3m
    or [200,300) delta <= -5m with downstream regression <= +1m

semantic role useful:
    semantic group contribution explains at least one successful h10/h15 candidate,
    and semantic policy beats Dg-only policy by >= 0.5m in h10/h15.
```

失败分流：

```text
if all semantic skip hurts ATE:
    keep semantic only as audit / visualization;
    do not use semantic as hard mask.

if sky_highD helps but sky_only hurts:
    semantic policy must be conditional on Dg/unc.

if structure boost hurts:
    structure semantic labels are not reliable write anchors; fallback to geometry eligibility.

if movable coverage is near zero:
    do not draw conclusion about movable policy; collect better Stage C cache or skip movable experiments.
```

---

# H4：TTT scale-state weak signal 应与 context-source skip / semantic roles 组合，而不是独立微扫

## 7.1 假设

v19 的 `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` 是 weak-positive，因为它在 chunk10 h10 有 `-2.224m` ATE delta，但 h15 衰减到 `-0.858m`。这说明它可能修了短期 scale-state，却没有保护长程 continuity。

如果 frame/global context-source skip 或 semantic role 能减少错误源进入 read/context，那么 TTT scale-state action 的 h15 durability 可能增强。

## 7.2 第一批组合

只在 short rollout 运行：

```text
TTTSS_00: H9 baseline
TTTSS_01: SCALECOMMIT_01 only
TTTSS_02: K/V skip Dg_high q90 only
TTTSS_03: SCALECOMMIT_01 + K/V skip Dg_high q90
TTTSS_04: SCALECOMMIT_01 + K/V skip lowstuff_highD
TTTSS_05: SCALECOMMIT_01 + semantic pos/neutral/negative TTT role
TTTSS_06: SCALECOMMIT_01 + context skip + semantic TTT role
```

Chunks：

```text
chunk6, chunk10, chunk16
horizons = h5, h10, h15
```

## 7.3 成立标准

```text
stage pass:
    h10/h15 ATE delta <= -3m
    or [200,300) delta <= -5m with downstream regression <= +1m

strong pass:
    h15 ATE delta <= -3m
    and h10 -> h15 decay ratio >= 0.65

selector allowed:
    at least 2 candidate families pass stage gate
    and no-GT proxy rank correlation >= 0.45 on candidate bank
```

失败分流：

```text
if SCALECOMMIT_01 + skip worsens h15:
    skip and scale-state action compete; decouple them and only keep the better one.

if h10 signal remains but h15 decays:
    move to W_short / lifecycle carrier, not more gamma.

if no family crosses -3m h10/h15:
    do not run full online; redirect to read/support/semantic skip full only if those pass independent gates.
```

---

# H5：如果语义和 skip 都失败，TTT write-only 应降级为 regularizer

## 8.1 假设

Target-25 的主杠杆可能不是 TTT write-only，而是 online trajectory-state / scale-state module。NOGT pose proxy 已证明 reset-window scale correction 有强上界，但那是 offline diagnostic，不计作 TTT success。

v20 不应直接跳到后处理，但必须设置硬边界：如果 TTT/skip/semantic 都无法给 h10/h15 产生足够上界，应停止把 Target-25 压在 TTT write-only 上。

## 8.2 停止标准

如果以下三类都失败：

```text
1. full_chunk / support variants fail short gate;
2. context-source skip fail short gate;
3. semantic memory roles fail short gate;
4. SCALECOMMIT_01 combination h15 remains < -1.5m improvement or decays strongly;
```

则：

```text
TTT write-only = auxiliary regularizer
Target-25 mainline = explicit online trajectory-state / scale-state module
```

Codex 后续应并行实现：

```text
online reset-window scale-state estimator
no-GT pose-step EMA module inside pipeline
scale-state feedback to camera/merge state before evaluation output
cross-seq sanity on KITTI00/02/05
```

但这必须明确标记为 trajectory-state module，不伪装成 TTT 写入。

---

## 9. 并行执行计划

为了加速，本轮分 4 个并行 worker track。所有 track 都先 short rollout，只有过 gate 才 full。

### Track 1：Support variants

负责人任务：

```text
1. audit support indices for past/full/full_chunk/near.
2. implement full_chunk if missing.
3. run S1 short rollout matrix.
4. generate support dashboard.
```

自动失败分流：

```text
if support indices equal unexpectedly:
    fix parser / naming before running.

if full_chunk h10/h15 no improvement:
    try past_plus_future_light.

if all support variants fail:
    return to C23_past and stop Track 1 full runs.
```

### Track 2：VGGT4D-style K/V source skip

负责人任务：

```text
1. implement source K/V skip hook for frame attention.
2. smoke with END_FRAME=128, verify source_keep_ratio and no empty source.
3. implement layer scope early / early_mid / all diagnostic.
4. run KVS short rollout matrix.
```

自动失败分流：

```text
if hard skip OOD / ATE huge regression:
    implement soft source gate.

if frame skip works:
    test global skip separately.

if global skip fails:
    do not combine frame+global in full.
```

### Track 3：Semantic memory roles

负责人任务：

```text
1. verify Mask2Former semantic cache coverage and group labels.
2. produce semantic role maps: structure, lowstuff, movable, uncertain.
3. run semantic skip/TTT short matrix.
4. produce per-group memory contribution dashboard.
```

自动失败分流：

```text
if semantic coverage missing:
    stop semantic control; only audit.

if sky_only fails:
    disable unconditional lowstuff skip.

if semantic TTT scalar fails:
    keep semantic as role-conditioned source mask, not scalar write score.
```

### Track 4：TTT scale-state anchor combination

负责人任务：

```text
1. rerun SCALECOMMIT_01 h10/h15 repeat if needed.
2. run TTTSS combination matrix only in short rollout.
3. measure h10/h15 decay ratio.
4. decide selector/full validation eligibility.
```

自动失败分流：

```text
if h15 decay ratio < 0.4:
    try W_short lifecycle, not more gamma.

if combination worsens vs SCALECOMMIT_01:
    isolate skip and scale-state; do not combine.

if no candidate meets stage gate:
    full online validation forbidden.
```

---

## 10. 全局指标与日志格式

每条 run 必须写入：

```text
run_registry.csv
run_config.yaml
hmc_state_hash.jsonl
trajectory_metrics.json
segment_metrics_50_100_200.csv
memory_control_debug.jsonl
support_indices.jsonl
source_skip_debug.jsonl
semantic_memory_debug.jsonl
ttt_write_debug.jsonl
```

### 10.1 Global metrics

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
ATE_50_mean / worst
ATE_100_mean / worst
ATE_200_mean / worst
[200,300)
[200,400)
[400,600)
per-reset-group ATE
```

### 10.2 Context skip metrics

```text
source_keep_ratio
source_keep_ratio_by_frame
source_keep_ratio_by_semantic_group
num_empty_source_events
special_token_keep_ratio
attention_entropy_delta
attention_mass_to_skipped_source_before
attention_mass_to_skipped_source_after
hidden_norm_delta
```

### 10.3 Semantic metrics

```text
semantic_coverage_total
semantic_coverage_by_group
Dg_mean_by_group
unc_mean_by_group
write_score_mean_by_group
TTT_update_norm_by_group_branch
negative_mass_by_group
neutral_mass_by_group
positive_mass_by_group
SWA_source_mass_by_group
frame/global K/V keep ratio by group
```

### 10.4 TTT scale-state metrics

```text
candidate_id
chunk_id
horizon
ATE_delta_vs_H9
[200,300)_delta_vs_H9
[400,600)_delta_vs_H9
h10_to_h15_decay_ratio
PZBASIS_delta_norm
post_zp_delta_cosine
branch_update_norm_w0/w1/w2
scale_proxy_value
scale_proxy_rank
```

---

## 11. 必须可视化的图

### 11.1 Support / cue dashboard

```text
RGB
D_g_past
D_g_full_chunk
D_g_diff = full_chunk - past
support frame ids per t
D_g mass over time
D_g by semantic group
```

### 11.2 VGGT4D-style skip dashboard

```text
RGB
D_g / semantic masks
M_src skip map
source_keep_ratio per frame
source_keep_ratio per semantic group
attention mass to skipped source before/after
hidden norm before/after
trajectory short rollout overlay
```

### 11.3 Semantic memory dashboard

```text
semantic group map
structure / sky / vegetation / movable masks
per-group write score heatmap
per-group attention source contribution
per-group TTT update norm
per-group SWA source mass
```

### 11.4 Global drift dashboard

```text
trajectory XY overlay
per-frame translation error curve
per-100f segment ATE bar chart
per-reset-group ATE
Sim3 scale over time
Yaw drift over time
[200,300) / [400,600) comparison table
```

---

## 12. Promotion gates

### 12.1 Short rollout gate

允许进入 full online 的最低条件：

```text
h10/h15 ATE delta <= -3m
or [200,300) delta <= -5m with [400,600) regression <= +1m
```

如果是 semantic/context skip 候选，还必须满足：

```text
empty_source_events = 0
source_keep_ratio >= 0.30
special_token_keep_ratio = 1.0
no catastrophic hidden_norm spike
```

### 12.2 Full online gate

```text
strong success:
    KITTI01 ATE <= 30m
    and [200,300) improves >= 10m vs H9
    and [400,600) regression <= +1m

Target-25 success:
    KITTI01 ATE <= 25m
    and no offline postprocess
    and no GT runtime action
    and counts_as_online_success = true

useful progress:
    ATE <= 32.5m
    or ATE <= C9 - 1m with [200,300) not worse than H9
```

### 12.3 Stop rules

```text
if same family 4 short runs fail gate:
    stop family.

if same family 2 full runs fail to beat C9 and worsen [200,300):
    stop family.

if intervention only improves Rot/FinalErr but ATE and [200,300) do not improve:
    keep diagnostic; no more full runs.

if unconditional semantic skip fails:
    switch to conditional semantic skip only.

if K/V skip hard mask fails:
    try soft source gate once; if still fails, stop context skip family.
```

---

## 13. 第一轮实际运行顺序

### Batch A：support-index audit + support short rollout

```text
A0 support index audit, no model full
A1 C23_past h10/h15 baseline
A2 C23_full_chunk h10/h15
A3 C23_full_chunk_no_overlap h10/h15
A4 C23_past_plus_future_light h10/h15
A5 C23_near24 h10/h15
```

### Batch B：K/V skip smoke + short rollout

```text
B0 END_FRAME=128 hard K/V skip Dg q90 smoke
B1 Dg q90 frame early skip h10/h15
B2 Dg q80 frame early skip h10/h15
B3 lowstuff_highD frame early skip h10/h15
B4 sky_only frame early skip h10 diagnostic
B5 sky_highD frame early skip h10/h15
B6 Dg q90 frame early skip + pair/all bias h10/h15
```

### Batch C：semantic all-memory role audit

```text
C0 semantic cache coverage / group audit
C1 semantic noop repeat
C2 semantic role maps export only
C3 frame skip lowstuff_highD
C4 TTT semantic pos/neutral/negative
C5 frame skip + TTT semantic role
C6 global skip lowstuff_highD diagnostic
```

### Batch D：TTT scale-state combination

```text
D0 repeat SCALECOMMIT_01 chunk10 h10/h15 if needed
D1 SCALECOMMIT_01 + Dg K/V skip q90
D2 SCALECOMMIT_01 + lowstuff_highD skip
D3 SCALECOMMIT_01 + semantic TTT pos/neu/neg
D4 SCALECOMMIT_01 + context skip + semantic role
```

Only candidates that pass h10/h15 gate may enter full KITTI01.

---

## 14. 本轮预期的真正结论

本轮不是为了再从 `33.76m` 刷到 `33.70m`。真正要回答的是：

```text
1. C23 support 是否应该从 past_only 改为 full_chunk？
2. LoGeR 是否像 VGGT4D 一样更适合 source K/V removal，而不是 attention bias？
3. 语义能否作为 memory-source role，而不是 TTT scalar？
4. TTT scale-state weak signal 是否能通过 read/source/semantic policy 变得 h15 durable？
5. 如果这些都不能产生 h10/h15 上界，Target-25 是否应正式转向 online trajectory-state / scale-state module？
```

如果答案是“都不能”，那不是失败，而是一个重要边界：

```text
TTT write-only and memory-source filtering are insufficient for Target-25.
Target-25 requires explicit online trajectory-state / scale-state correction.
```

但在得出这个结论前，本轮必须把 `full_chunk`、VGGT4D-style K/V skip、semantic all-memory role 这三条关键未验证分支跑干净。

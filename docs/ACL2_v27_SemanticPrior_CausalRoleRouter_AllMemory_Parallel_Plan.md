# ACL2 v27：SemanticPrior Causal Role Router All-Memory 并行实验计划

日期：2026-05-22  
对象：LoGeR / HMC Pipeline v2 / VideoMasklet Semantic Prior Generator / Frame-Global-SWA-TTT all-memory control  
主目标：把 Semantic Prior Generator 从“语义标签驱动的简单 source/write gate”升级为 **fine-label + path-specific + risk-conditioned memory role router**。  
开发集：KITTI01 causal fork / short rollout 为主。  
诊断集：KITTI00 / KITTI02 / KITTI05 或其他可用序列只用于 failure-mode 诊断，不允许对单个数据集调参。  
当前 deployable online TTT best：`C9_P0_R2 = 33.7629421029m`。  
最终目标：KITTI01 online ATE `<= 25m`。  

---

## 0. 本轮为什么要换计划

v23-v26 已经说明，Semantic Prior Generator 的工程通路基本不是当前 blocker。

当前已经成立的事实是：

```text
1. VideoMasklet cache hit / coverage / focus coverage 都可用。
2. Stage C cache + Stage D semantic no-op 不扰动 HMC / trajectory。
3. fine labels 已经 runtime-visible。
4. R_frame_tok / R_global_tok / R_swa_tok / R_ttt_tok 四条 role stream 均非空。
5. frame / global / SWA / TTT 都能消费 semantic role。
6. no-op / pass-through / debug-only smoke 可以严格对齐 reference。
```

但是当前语义策略仍然没有过性能 gate。v26 的关键边界是：

```text
original h10 best ATE delta = -0.2713m
original h10 best [200,300) delta = -1.2687m
risk-coupled h10 best ATE delta = -1.9229m
risk-coupled h10 best [200,300) delta = -1.8379m
risk-coupled h15 best ATE delta = -1.0253m
risk-coupled h15 best [200,300) delta = -1.6843m
Phase 2 gate = fail
No pairwise / all-memory / selector / full online validation
```

这说明：

```text
语义已经接上了；
语义也有解释力；
但语义本身还不是强因果 action。
```

因此 v27 不再问：

```text
某个语义类要不要 skip？
某个粗粒度 group 要不要写？
所有 memory path 要不要一起使用同一套 semantic rule？
```

而是问：

```text
在每条 memory path 中，某个 fine semantic label 在什么 geometry / TTT conflict / scale-risk 条件下，应该扮演什么 memory role？
```

这就是本轮的核心转向。

---

## 1. 实验整体目标

本轮实验有四个整体目标。

### 1.1 目标 A：建立真正的 semantic memory role router

Semantic Prior Generator 不再只输出一个 `A_tok` 或一个 coarse role，而是输出四条 path-specific role stream：

```text
R_frame_tok   -> frame attention K/V source role
R_global_tok  -> global/chunk attention K/V source role
R_swa_tok     -> SWA local cache / overlap source role
R_ttt_tok     -> TTT positive / neutral / negative write role
```

这些 role 必须同时条件化于：

```text
fine semantic label
D_g read/source risk
Q_mask / mask trust
update_conflict_energy
scale-state risk
memory path type
```

语义只提供类别先验，不单独决定最终 memory action。

---

### 1.2 目标 B：判断 semantic 是否能辅助已有强信号，而不是单独开新矩阵

历史上更强的短期信号来自：

```text
C23 D_g
compact_kv source skip
scale-state commit
update_conflict_energy
skip-aware TTT / SWA / global source
```

v26 说明 semantic-only 很弱，risk-coupled 变强。因此 v27 的核心不是让 semantic 替代这些信号，而是让 semantic 去解释和路由这些信号。

本轮要回答：

```text
1. 哪些 fine labels 与 high D_g / high conflict / scale-risk 同时出现？
2. 哪些 fine labels 是 positive continuity source？
3. 哪些 fine labels 是 short negative evidence？
4. 哪些 fine labels 不应该写入 TTT，但应该保留为 frame/global/SWA context？
```

---

### 1.3 目标 C：把 h10 信号转成 h15 / full-run durability

之前多轮都出现类似模式：

```text
h10 有局部改善；
h15 衰减；
full online 不允许启动。
```

v26 的 risk-coupled h15 有一定改善，但不够强。本轮必须把 durability 当成主目标，而不是只看 h10。

定义 durability ratio：

$$
R_{dur}^{ATE} = \frac{|\Delta ATE_{h15}|}{|\Delta ATE_{h10}| + \epsilon}
$$

$$
R_{dur}^{200} = \frac{|\Delta E_{[200,300),h15}|}{|\Delta E_{[200,300),h10}| + \epsilon}
$$

如果一个 candidate 只有 h10 强、h15 弱，则它不是 memory policy，只是短期 read/source filter。

---

### 1.4 目标 D：跨数据集只诊断，不调参

可以使用 KITTI00 / KITTI02 / KITTI05 或其他可用数据集做诊断，目的是回答：

```text
同一套 semantic role 在不同场景中失败在哪里？
某些 label 是否只在 KITTI01 有高风险？
某些 path 是否跨数据集一致有效？
```

但不允许：

```text
为 KITTI01 单独调 threshold；
为 KITTI02 单独改 sky rule；
为某个 sequence 单独改 gamma；
为某个数据集单独改 semantic value table。
```

所有策略必须使用同一套规则。跨数据集只做 failure-mode diagnostics，不做 benchmark tuning。

---

## 2. 本轮原则

### 2.1 不做小修小补

本轮不继续做下面这些低收益实验：

```text
coarse group threshold sweep
LOW_VALUE_STUFF value 小扫
semantic positive_scale / negative_scale 小扫
all-memory 直接叠加
h10 only 强 candidate 直接 full online
```

除非某个 candidate 先通过 h10/h15 gate，否则不允许进入更大矩阵。

---

### 2.2 先 attribution，后 intervention

语义如果没有解释力，就不应该直接控制 memory。

本轮每条 candidate 之前必须先输出：

```text
fine label coverage
fine label x D_g
fine label x update_conflict_energy
fine label x scale-state risk
fine label x memory path action
fine label x h10/h15 error delta
```

只有当 semantic role 能解释已有强信号，才允许进入 action candidate。

---

### 2.3 单 path 过 gate 才能 pairwise，pairwise 过 gate 才能 all-memory

严格顺序：

```text
Phase 0: implementation/no-op/path audit
Phase 1: passive semantic-risk attribution
Phase 2: single-path semantic-risk role candidates
Phase 3: pairwise path combinations
Phase 4: all-memory durable role policy
Phase 5: no-GT selector diagnostic
Phase 6: full online validation
```

如果 Phase 2 没有 candidate 过 gate，则 Phase 3 / 4 / full online 全部停止。

---

## 3. 核心假设

---

## H0：v26 工程通路虽然通过，但 v27 需要确认 token-aligned risk condition 真正接入 runtime

### 假设

v26 已经把 fine label runtime-visible，并且四条 role stream 非空。但 v26 自己也记录了一个重要限制：`update_conflict_energy` 和 scale-state risk 还不是 token-aligned runtime input。因此 v27 的第一目标是确认这两个风险信号真的进入 semantic role router，而不是只作为 write-side diagnostic coupling。

### 实验设计

Codex 先实现并自查：

```text
L_sem_tok              fine semantic label token stream
D_g_tok                read/source risk
Q_mask_tok             mask trust
U_conflict_tok         token-aligned update_conflict_energy or branch/layer aggregated proxy projected to token
S_scale_tok            token-aligned scale-state risk or chunk/path-level scale risk broadcast with provenance tag
R_frame_tok            frame source role
R_global_tok           global source role
R_swa_tok              SWA cache/source role
R_ttt_tok              TTT write role
```

风险信号允许有不同级别：

```text
token_exact:
    true token-level risk

patch_projected:
    patch-level risk projected to token

chunk_broadcast:
    chunk-level risk broadcast to tokens

unavailable:
    not available, candidate invalid if it requires this risk
```

每个 run 必须记录 provenance：

```text
condition_signal_conflict_level = token_exact / patch_projected / chunk_broadcast / unavailable
condition_signal_scale_level    = token_exact / patch_projected / chunk_broadcast / unavailable
```

### 必须记录

```text
semantic_role_router_audit.json
semantic_role_alignment_audit.csv
per_token_condition_summary.csv
path_consumption_audit.csv
noop_parity_metrics.csv
context_skip_summary.jsonl
semantic_role_summary.jsonl
semantic_memory_path_summary.jsonl
```

### H0 通过标准

```text
1. no-op / pass-through direct pose diff = 0；
2. R_frame/R_global/R_swa/R_ttt 均非空；
3. frame/global/SWA/TTT consumption flags 全为 true；
4. fine label id 与 cache fine label 对齐；
5. risk condition provenance 不为 unavailable；
6. source skip empty events = 0；
7. stale run dir / old JSONL contamination = 0。
```

如果 H0 不通过，不允许启动 Phase 1 candidate。

### 如果 H0 不通过，Codex 先尝试

```text
1. 如果 fine labels 为空：检查 Stage C cache -> L_sem -> token projection。
2. 如果 role stream 为空：检查 PriorOutput pass-through 和 HMC control prior forwarding。
3. 如果 conflict/scale risk unavailable：先实现 chunk_broadcast fallback，并明确标记 provenance。
4. 如果 no-op 漂移：检查 pass-through prior 是否改变 stage_d base，禁止继续跑性能矩阵。
5. 如果 source skip empty source events > 0：加 protected tokens / minimum keep source fallback。
```

---

## H1：fine semantic label 有解释力，但只有和 D_g / conflict / scale-risk 组合后才有控制力

### 假设

单独语义很弱，但语义能解释已有强风险信号的空间分布。也就是说，fine label 可以告诉我们某类风险 token 应该如何处理。

例如：

```text
sky + low D_g:
    neutral horizon context

sky + high D_g + high scale risk:
    weak frame/global source skip, not TTT negative

vegetation + high D_g + high conflict:
    frame/global source skip + TTT short negative

road/building/wall/fence + low D_g + low conflict:
    positive long memory

road/building + high conflict:
    read source may be useful, but TTT long write should be blocked
```

### Phase 1 passive attribution

对 locked parent snapshots 跑 passive，不改变模型输出。重点 chunk：

```text
chunk 6
chunk 10
chunk 16
```

horizon：

```text
h10
h15
```

### 必须记录

每个 fine label `l`，每条 path `p`，每个风险 bin `b`：

```text
coverage(l)
D_g_mean(l), D_g_p90(l)
Q_mask_mean(l)
conflict_mean(l), conflict_p90(l)
scale_risk_mean(l), scale_risk_p90(l)
frame_source_keep_ratio(l)
global_source_keep_ratio(l)
swa_cache_keep_mass(l)
ttt_pos_mass(l)
ttt_neutral_mass(l)
ttt_neg_mass(l)
post_zp_update_norm(l)
post_zp_update_conflict(l)
```

按 segment 记录：

```text
ATE_delta_h10
ATE_delta_h15
[200,300]_delta_h10
[200,300]_delta_h15
[400,600]_delta_h10
[400,600]_delta_h15
FinalErr_delta
YawRMSE_delta
Sim3Scale_delta
```

构造条件 lift：

$$
Lift(l, r, p) = E[\Delta E \mid label=l, risk=r, path=p] - E[\Delta E \mid risk=r, path=p]
$$

其中 $\Delta E$ 可以是 h10/h15 的 ATE delta 或 $[200,300)$ delta。

### H1 成立标准

H1 通过至少满足两条：

```text
1. 某些 fine labels 在同一 coarse group 内 D_g / conflict / scale-risk 分布明显不同；
2. semantic+risk 的预测力强于 risk-only，Spearman 或 AUROC 提升 >= 0.10；
3. fine label 能解释 v20/v22/v26 中强 h10 candidate 的主要 token mass；
4. semantic label 的 conditional lift 在至少一个 path 上达到 >= 0.5m；
5. passive attribution 能指出至少一个明确角色，例如 vegetation highD conflict -> source skip / short negative。
```

如果 H1 不成立，则语义不作为主控制信号，降级为 debug/visualization；后续只做已有强信号的 path control。

### 如果 H1 不成立，Codex 先尝试

```text
1. 检查 fine label taxonomy 是否过粗或错映射。
2. 将 labels 合并为 functional groups：horizon_context / planar_structure / vertical_structure / vegetation / movable / uncertain。
3. 如果 semantic 与任何 risk 都无条件相关，停止 semantic action，改做 risk-only controller。
4. 如果 semantic 只在某个 dataset 有相关性，只记录 domain diagnostic，不调数据集参数。
```

---

## H2：frame/global source 语义角色应控制 K/V source，而不是控制 query 或 TTT write

### 假设

VGGT4D 给的启发是：不删除 query token，而是在早期层让高风险动态区域不再作为 K/V source。LoGeR 中 frame/global attention 的语义 role 应该主要控制 source token，而不是直接影响输出 token。

### 实验设计

固定 parent：H9 causal fork。  
固定 read cue：C23 past。  
固定 source skip impl：`compact_kv`。  
先做 single-path，不同时改 SWA/TTT。

候选：

```text
FG_RISK_00:
    D_g / conflict / scale-risk only, no semantic

FG_SEM_01:
    structure lowD keep
    lowstuff highD skip
    sky lowD neutral keep

FG_SEM_02:
    vegetation highD skip
    sky always neutral
    structure lowD keep

FG_SEM_03:
    road/building/wall/fence lowD keep
    road/building high conflict protected from TTT but kept for read source

FG_SEM_04:
    semantic+risk source skip
    skip only if semantic risk AND D_g high AND conflict or scale-risk high

FG_SEM_05:
    soft skip instead of hard skip
    source keep weight = 0.5 for risky lowstuff / vegetation
```

### 必须记录

```text
frame_source_keep_ratio_by_label
global_source_keep_ratio_by_label
num_context_source_skip_applied
max_context_source_skip_tokens
num_context_empty_source_events
attention_mass_to_skipped_sources_before_after
attention_entropy_before_after
protected_token_mass
h10/h15 segment metrics
```

### H2 成立标准

进入 Phase 3 的条件：

```text
h10 [200,300) delta <= -3m
or h10 ATE delta <= -1.5m
and h15 ATE regression <= +0.5m
and [400,600) regression <= +1m
and empty source events = 0
```

强通过条件：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
and durability ratio >= 0.45
and [400,600) regression <= +1m
```

### 如果 H2 不通过，Codex 先尝试

```text
1. 如果 keep ratio 几乎没变：检查 R_frame/R_global 是否真正被 compact_kv 消费。
2. 如果 h10 有改善 h15 消失：转 Phase 5 durability attribution，不继续阈值扫。
3. 如果 [200,300) 改善但 [400,600) 崩：加入 structure/horizon protected source。
4. 如果 sky skip 伤 ATE：改为 sky neutral partial keep，禁止 sky hard skip。
5. 如果 all semantic rows 都弱于 risk-only：frame/global 语义降级，只保留 risk-only compact_kv。
```

---

## H3：SWA semantic role 不是“少写语义低价值区域”，而是保护 overlap local source topology

### 假设

SWA 是 local lossless memory，它的目标是 adjacent alignment。语义在 SWA 中的角色不应照搬 TTT。

SWA 应该保留：

```text
stable road / wall / fence / building overlap source
horizon/sky lowD if useful for local continuity
```

SWA 应该减弱：

```text
movable highD
vegetation highD
uncertain boundary high conflict
```

### 实验设计

固定 frame/global 和 TTT 不变，只控制 SWA cache/source。

候选：

```text
SWA_SEM_01_STRUCTURE_OVERLAP_KEEP
SWA_SEM_02_LOWSTUFF_HIGHD_WEAK_SKIP
SWA_SEM_03_VEGETATION_HIGHD_SKIP_SKY_NEUTRAL
SWA_SEM_04_ROAD_WALL_FENCE_LOCAL_ANCHOR
SWA_SEM_05_RISK_ONLY_SWA_CACHE_CONTROL
```

### 必须记录

```text
swa_cache_keep_mass_by_label
swa_previous_source_mass_by_label
swa_current_source_mass_by_label
swa_overlap_source_mass_by_label
swa_kv_replace_mass_by_label
boundary_10f_ATE
boundary_20f_ATE
overlap_pointmap_residual
chunk_boundary_pose_jump
h10/h15 segment metrics
```

### H3 成立标准

SWA semantic role 不一定必须降低 full ATE，但至少要满足：

```text
h10/h15 boundary metric 改善 >= 10%
or h15 ATE delta <= -1.5m
or h15 [200,300) delta <= -3m
and [400,600) regression <= +1m
```

### 如果 H3 不通过，Codex 先尝试

```text
1. 如果 SWA cache mass 没变：检查 SWA role stream 是否被写入 cache path。
2. 如果 boundary metrics 没落盘：先实现 boundary diagnostics，不跑候选。
3. 如果 SWA skip 伤 h15：把 lowstuff/sky 改为 neutral keep，只 skip movable/vegetation highD。
4. 如果 SWA-only 弱但 frame/global 强：停止 SWA semantic 单 path，等 pairwise 阶段只做 compatibility check。
```

---

## H4：TTT semantic role 必须和 conflict / scale-state risk 共同决定 positive / neutral / negative replay

### 假设

语义本身不能决定 TTT write。TTT 的目标是长期 compressed global memory，必须看 token update 是否和 TTT conflict / scale drift 相关。

TTT role 应定义为：

```text
positive_long:
    structure label
    low D_g
    low update_conflict_energy
    low scale-state risk
    high mask trust

neutral_keep:
    sky lowD
    vegetation lowD
    road/sidewalk low conflict
    horizon/context-like stuff

negative_short:
    movable highD high conflict
    vegetation highD high conflict
    lowstuff highD high scale-risk

block_long_write:
    structure label but high conflict or high scale-risk
```

### 实验设计

固定 frame/global/SWA 不动，只控制 TTT write role。

候选：

```text
TTT_ROLE_00_RISK_ONLY
    update_conflict + scale-risk only

TTT_ROLE_01_STRUCTURE_LOWCONFLICT_POS
    structure lowD low conflict -> positive long

TTT_ROLE_02_LOWSTUFF_HIGHD_CONFLICT_SHORTNEG
    lowstuff highD high conflict -> negative short
    sky lowD -> neutral

TTT_ROLE_03_VEGETATION_CONDITIONAL_NEG
    vegetation highD high conflict -> negative short
    vegetation lowD -> neutral

TTT_ROLE_04_BLOCK_HIGHCONFLICT_STRUCTURE_LONGWRITE
    structure high conflict -> no long write, but no negative

TTT_ROLE_05_FULL_ROLE_TREE
    combine positive_long / neutral_keep / negative_short / block_long_write
```

### 必须记录

```text
TTT_pos_mass_by_label
TTT_neutral_mass_by_label
TTT_neg_mass_by_label
TTT_block_long_mass_by_label
update_conflict_energy_by_role
scale_risk_by_role
branch0_update_norm_by_role
post_zp_delta_norm_by_role
post_zp_delta_cosine_to_H9
h10/h15 ATE and segment metrics
h10->h15 state overwrite ratio
```

### H4 成立标准

单 path TTT semantic candidate 通过条件：

```text
h10 ATE delta <= -1.5m
or h10 [200,300) delta <= -3m
and h15 ATE regression <= +0.5m
and [400,600) regression <= +1m
```

强通过条件：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
and durability ratio >= 0.45
and TTT_pos/neg role mass is non-trivial
```

### 如果 H4 不通过，Codex 先尝试

```text
1. 如果 role mass 太小：降低 condition threshold，但不得按数据集调参，只用 quantile rules。
2. 如果 negative_short 伤 h15：改为 block_long_write，不做 negative replay。
3. 如果 positive_long 伤 [200,300)：说明 structure 中有 high-conflict tokens，必须加 conflict gate。
4. 如果 risk-only 明显强于 semantic+risk：语义只保留为解释标签，不参与 TTT action。
5. 如果 h10 强 h15 弱：进入 durability attribution，检查 TTT tail update 是否覆盖 correction。
```

---

## H5：all-memory 只有在 single-path 互补时才有意义，否则会互相抵消

### 假设

v23/v24/v25B/v26 都说明，直接 all-memory 很容易失败。all-memory 只有在单 path 结果有明确互补性时才值得做。

### 实验设计

只组合通过 Phase 2 gate 的 path。

组合类型：

```text
PAIR_FG_TTT:
    frame/global source role + TTT role

PAIR_FG_SWA:
    frame/global source role + SWA role

PAIR_SWA_TTT:
    SWA local source role + TTT role

ALLMEM_ROLE_TREE:
    frame/global + SWA + TTT, but each path has different role rule
```

### 必须记录

```text
component_id
component_single_path_delta
pairwise_delta
interaction_gain
```

定义 interaction gain：

$$
Gain_{int} = \Delta E_{pair} - \min(\Delta E_A, \Delta E_B)
$$

如果 $Gain_{int}<0$ 且 pairwise worse，则说明两个 path 抵消。

### H5 成立标准

```text
pairwise h15 ATE delta <= -3m
or pairwise h15 [200,300) delta <= -5m
and durability ratio >= 0.45
and [400,600) regression <= +1m
```

如果 pairwise 都不成立，不允许 all-memory。

### 如果 H5 不通过，Codex 先尝试

```text
1. 如果 pairwise 比单 path 差：分析 interaction_gain，不继续 all-memory。
2. 如果 frame/global + TTT 冲突：frame/global 只保留 source skip，TTT 只保留 block_long_write，不做 negative replay。
3. 如果 SWA + TTT 冲突：SWA 保留 local continuity，TTT 降低 negative_short 或转 neutral。
4. 如果 all-memory h10 强 h15 弱：只做 durability attribution，不跑 full。
```

---

## H6：跨数据集诊断要固定策略，不能为数据集调参

### 假设

如果 semantic role 是可靠机制，它应在不同序列上表现出可解释的 failure mode，而不是只能靠特定数据集调参成立。

### 实验设计

只拿通过 KITTI01 h15 gate 的 1-2 个 candidate 跑诊断集。固定同一套参数。

建议诊断集：

```text
KITTI00
KITTI02
KITTI05
可选：KITTI-360 / STEP / 其他 driving sequence，如果已有 compatible pipeline
```

不允许每个数据集改：

```text
label thresholds
D_g quantile
conflict quantile
gamma
semantic value table
path weights
```

### 必须记录

```text
ATE / Rot / RPE / FinalErr
segment metrics
semantic label coverage distribution
D_g distribution by label
conflict distribution by label
scale-risk distribution by label
path source keep ratio by label
failure mode summary
```

### H6 成立标准

跨数据集不要求全部 ATE 最优，但要求：

```text
1. 不出现 >5% catastrophic regression；
2. 至少两个诊断集上有同向 improvement 或可解释的稳定 role behavior；
3. semantic role 的 label-risk 分布解释 failure mode；
4. 不需要 per-dataset threshold change 才能运行。
```

如果跨数据集失败：

```text
记录 failure mode；
不针对单个数据集调参；
回到 role design 层做通用规则修改。
```

---

## 4. 并行执行计划

本轮为了加速，拆成 5 个 Codex track，并行推进。

---

### Track A：Implementation / Audit Track

目标：先保证所有 role stream 和 risk condition 都真的接入。

任务：

```text
A1. 实现 token-aligned or provenance-tagged conflict / scale risk input。
A2. 更新 semantic_prior_generator 输出 R_frame/R_global/R_swa/R_ttt。
A3. 更新 HMC control prior，确保四条 path 都能消费对应 role。
A4. 输出 semantic_role_router_audit.json。
A5. 跑 Phase 0 no-op / pass-through。
```

停止条件：H0 不过，其他 track 不允许跑性能候选。

失败自动分流：见 H0。

---

### Track B：Passive Attribution Track

目标：确认 fine label 是否能解释已有 risk / memory behavior。

任务：

```text
B1. 对 chunk6/10/16 跑 passive h10/h15 attribution。
B2. 生成 label x risk x path dashboard。
B3. 比较 semantic-only / risk-only / semantic+risk 的预测力。
B4. 输出 top harmful semantic-risk bins 和 top continuity bins。
```

停止条件：H1 不成立，则 semantic 降级，不继续 action matrix。

---

### Track C：Single Path Candidate Track

目标：分别验证 frame/global、SWA、TTT 三条 path。

并行任务：

```text
C1. Frame/global source candidates FG_SEM_*。
C2. SWA cache/source candidates SWA_SEM_*。
C3. TTT role candidates TTT_ROLE_*。
```

每个 task 都先跑：

```text
chunk10 h10
```

只有接近 gate 才补：

```text
chunk10 h15
chunk6 h10/h15
chunk16 h10/h15
```

停止条件：单 path 没过 gate，不启动 pairwise。

---

### Track D：Durability / Washout Track

目标：如果 h10 强 h15 弱，定位 correction 被哪条 memory path 洗掉。

任务：

```text
D1. 保存 h10 endpoint state 和 h15 endpoint state。
D2. 比较 TTT / SWA / global / merge-gauge 的 state movement。
D3. 计算 h10->h15 overwrite ratio。
D4. 给出 repair action recommendation：
    skip-aware TTT commit
    SWA cache protect
    global source protect
    W_long/W_short lifecycle
    merge/gauge not memory issue
```

停止条件：没有 h10 强候选，不启动 D track。

---

### Track E：Combination / Cross-dataset Diagnostic Track

目标：只对过 gate 候选做 pairwise/all-memory 和跨数据集诊断。

任务：

```text
E1. Pairwise combination。
E2. All-memory role tree。
E3. No-GT selector。
E4. Full online validation。
E5. Cross-dataset diagnostic。
```

启动条件：H2/H3/H4 至少有一个 path 通过 h15 gate。

停止条件：pairwise 未过 gate，不启动 all-memory；selector 未过 gate，不启动 full。

---

## 5. Candidate 命名规范

建议统一命名：

```text
V27_P0_*
V27_ATTR_*
V27_FG_*
V27_SWA_*
V27_TTT_*
V27_PAIR_*
V27_ALLMEM_*
V27_XSEQ_*
```

示例：

```text
V27_FG_04_SEM_RISK_COMPACT_chunk10_h10
V27_TTT_05_FULL_ROLE_TREE_chunk10_h15
V27_PAIR_FG_TTT_02_chunk10_h15
```

每个 run 必须写入：

```text
run_metadata.yaml
candidate_config.yaml
semantic_role_router_audit.json
semantic_memory_path_summary.jsonl
context_skip_summary.jsonl
candidate_metrics.csv
```

---

## 6. 指标与可视化

---

### 6.1 轨迹指标

每条 candidate 必须记录：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
ATE_50
ATE_100
ATE_200
[200,300)
[200,400)
[400,600)
chunk6-10 local metrics
chunk16 local metrics
```

---

### 6.2 Semantic attribution 指标

```text
per_label_coverage
per_label_visibility
per_label_Q_mask
per_label_D_g_mean
per_label_D_g_p90
per_label_conflict_mean
per_label_scale_risk_mean
per_label_path_role_counts
per_label_path_action_mass
```

---

### 6.3 Memory path 指标

Frame/global：

```text
source_keep_ratio_by_label
source_skip_mass_by_label
attention_mass_to_kept_sources
attention_mass_to_skipped_sources
attention_entropy_before_after
empty_source_events
protected_token_ratio
```

SWA：

```text
swa_cache_keep_mass_by_label
swa_previous_source_mass_by_label
swa_overlap_source_mass_by_label
swa_kv_replace_mass_by_label
boundary_10f_ATE
boundary_20f_ATE
overlap_pointmap_residual
chunk_boundary_pose_jump
```

TTT：

```text
TTT_pos_mass_by_label
TTT_neutral_mass_by_label
TTT_neg_mass_by_label
TTT_block_long_mass_by_label
update_conflict_energy_by_role
scale_risk_by_role
branch0_update_norm_by_role
post_zp_delta_norm_by_role
post_zp_delta_cosine
h10_to_h15_state_overwrite_ratio
```

---

### 6.4 必须生成的图

每个阶段至少生成以下图：

```text
1. semantic_role_dashboard.png
   fine label x path x role mass heatmap

2. label_risk_distribution.png
   fine label 的 D_g / conflict / scale risk distribution

3. memory_path_action_heatmap.png
   frame/global/SWA/TTT 的 keep/skip/write mass

4. h10_h15_delta_waterfall.png
   h10 -> h15 error delta 和 washout

5. semantic_overlay_grid/
   RGB
   fine label map
   D_g
   update_conflict_energy
   scale-risk
   R_frame / R_global / R_swa / R_ttt

6. source_skip_attention_before_after.png
   attention mass and entropy before/after source role

7. ttt_role_update_heatmap.png
   layer x branch x semantic role update norm

8. cross_dataset_diagnostic_dashboard.png
   only if Phase 6 starts
```

---

## 7. Gate 与停止规则

### 7.1 Phase 0 implementation gate

必须全部满足：

```text
no-op direct pose diff = 0
role streams non-empty
path consumption flags true
risk condition provenance valid
empty source events = 0
cache hit rate = 1.0
no stale run contamination
```

---

### 7.2 Phase 1 attribution gate

至少满足两条：

```text
semantic+risk 比 risk-only 的预测力提升 >= 0.10
至少一个 fine label conditional lift >= 0.5m
能解释已有强 h10 signal 的主要 token mass
输出明确 path-specific role hypothesis
```

---

### 7.3 Phase 2 single-path gate

进入 h15 confirmation：

```text
h10 ATE delta <= -1.5m
or h10 [200,300) delta <= -3m
```

进入 pairwise：

```text
h15 ATE delta <= -1.5m
or h15 [200,300) delta <= -3m
and [400,600) regression <= +1m
```

强候选：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
and durability_ratio >= 0.45
and [400,600) regression <= +1m
```

---

### 7.4 Pairwise / all-memory gate

```text
pairwise h15 ATE delta <= -3m
or pairwise h15 [200,300) delta <= -5m
and durability_ratio >= 0.45
and [400,600) regression <= +1m
```

---

### 7.5 Full online gate

只允许满足以下条件的 candidate full online：

```text
short rollout h15 gate passed
no-GT selector passed if selector is involved
no path interaction conflict
no dataset-specific parameters
```

Full online 通过标准：

```text
ATE <= 32.5m as stage success
ATE <= 30m as strong progress
ATE <= 25m as final target
and [200,300) does not regress vs H9 by >1m
and [400,600) regression <= +1m
```

---

## 8. Codex 失败自动分流表

| 失败现象 | 不要做什么 | Codex 下一步先尝试什么 |
|---|---|---|
| no-op 不对齐 | 不跑 candidate | 检查 pass-through prior 是否改变 stage_d base；检查 stale JSONL；检查 HMC ignore flag |
| fine label 空 | 不跑 coarse fallback 冒充 fine | 修 L_sem_tok projection；检查 Stage C cache fine labels |
| role stream 非空但 source keep ratio 不变 | 不跑 full | 检查 R_frame/R_global 是否被 compact_kv 消费 |
| semantic-only 全弱 | 不扩大 all-memory | 做 semantic+risk attribution；若仍弱，semantic 降级 |
| h10 强 h15 弱 | 不调阈值 | 做 state washout attribution；检查 TTT/SWA/global/merge 谁覆盖 correction |
| [200,300) 改善但 [400,600) 崩 | 不直接 full | 加 continuity protection；降低 negative action；保护 structure/horizon |
| sky skip 伤 ATE | 不继续 hard sky skip | sky 改 neutral/partial keep；只在 highD + high risk 时 weak skip |
| vegetation skip 伤 h15 | 不默认 vegetation negative | vegetation lowD neutral，highD+conflict 才 short negative |
| structure positive 伤 ATE | 不继续强写 structure | 加 low conflict / low scale-risk gate；high conflict structure block long write |
| SWA semantic 弱 | 不继续 SWA 大矩阵 | 检查 boundary metrics；若无效，SWA semantic 降级为 compatibility check |
| TTT semantic 弱于 risk-only | 不继续 semantic TTT | TTT 用 risk-only，semantic 只做 explainability |
| pairwise 比 single path 差 | 不启动 all-memory | 做 interaction_gain 分析，找冲突 path |
| 跨数据集失败 | 不调数据集阈值 | 记录 failure mode，回到通用 role 设计 |

---

## 9. 资源与加速策略

```text
1. Phase 0 / Phase 1 以 CPU-light audit + h3 smoke 为主。
2. Phase 2 每个 track 先只跑 chunk10 h10。
3. 只有 h10 接近 gate 才补 h15 / chunk6 / chunk16。
4. 只有 h15 过 gate 才启动 pairwise。
5. 只有 pairwise 过 gate 才启动 all-memory。
6. full online 只给 1-2 个最终候选。
7. 所有 Stage C / VideoMasklet 必须使用 offline cache + require-hit，不允许 inline compute 干扰 parity。
8. 默认 4 GPU 并发；host RAM 高风险时自动降到 2-3。
9. 每个 run 必须独立目录；stale dir 必须移动到 .INVALID_RERUN_*。
```

---

## 10. 预期结论形态

本轮不会把结果强行写成成功。最终可能出现三种结论。

### 10.1 结论 A：semantic+risk role 成功

如果 semantic+risk candidate 过 h15 gate，并且 full online 有明显提升，则 Semantic Prior Generator 成为 all-memory role controller 主线。

### 10.2 结论 B：semantic 只能辅助解释

如果 semantic+risk attribution 成立，但 action 弱于 risk-only，则语义保留为 explainability / reliability module，不再作为主控制信号。

### 10.3 结论 C：semantic 路线暂时降级

如果 fine label + risk 仍无法产生 h10/h15 strong signal，则说明 video-masklet semantic 不是 Target-25 主因果变量。后续主线转回：

```text
D_g / compact_kv / scale-state / update_conflict_energy
explicit online trajectory-state / scale-state module
```

语义只用于可视化、诊断和 safety guard。

---

## 11. 最终一句话

v27 的核心不是继续问“哪个语义类该跳过”，而是建立一个 **语义 + 几何风险 + TTT conflict + scale-state risk** 共同决定 memory role 的系统。如果这个系统仍然不能产生 h15 / full-run durable improvement，就应明确把 Semantic Prior Generator 从 Target-25 主线降级为辅助诊断模块。

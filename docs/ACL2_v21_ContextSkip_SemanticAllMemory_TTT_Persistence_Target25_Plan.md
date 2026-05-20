# ACL2 v21：Context Skip、Full-Chunk Support、Semantic All-Memory 与 TTT Persistence 实验计划

日期：2026-05-21  
对象：LoGeR / HMC Pipeline v2 / KITTI01 Target-25  
当前 deployable online TTT reference：`C9_P0_R2 = 33.7629421029m`  
当前可信 parent / sandbox reference：`H9_P0_R2 = 34.1257769401m`  
当前目标：`KITTI01 ATE <= 25m`  
当前 gap：

$$
33.7629421029 - 25.0 = 8.7629421029m
$$

本计划的核心原则是：**不再把 short rollout、sandbox oracle、GT audit、proxy semantic mask、instrumentation smoke 或 failed gate 写成 deployable online success。所有 full online validation 必须由 h10/h15 sandbox gate 授权。**

---

## 0. 本轮 v20 结果的独立分析

v20 的工程工作是有价值的：context-source-skip 参数已经接入，`S_tok` 可以从 HMC 传到模型侧，`full_chunk / full_chunk_no_overlap / past_plus_future_light` 等 support alias 已经接入，short-rollout launcher 和 report 工具也能完整落盘。矩阵完整性也不错，Batch A/B/D 的 rollout 目录、benchmark log 和 `01.txt` 都完整，没有 OOM 或 RuntimeError。

但从科学结果看，v20 仍然没有找到好的 TTT 写入策略，也没有产生 deployable online Target-25 候选。所有 v20 row 都是 trusted short-rollout diagnostic，`counts_as_online_ttt_write_success=false`，没有启动 no-GT selector，也没有启动 full online validation。

当前最好可计数 full online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
counts_as_ttt_write = true
```

而 v20 最强 diagnostic 是：

```text
TTTSS_03B_SCALECOMMIT_DGQ80_HARD
chunk = 10
horizon = h10
ATE delta vs H9 = -2.5220781811781947m
[200,300) delta vs H9 = -4.852495000715884m
[400,600) proxy delta vs H9 = -3.3403945846268996m
```

这个结果很接近短滚动 gate，但仍然没有过 gate：

$$
-2.522 > -3.0
$$

$$
-4.852 > -5.0
$$

更重要的是，它在 h15 上几乎完全衰减：

```text
TTTSS_03B chunk10 h15 ATE delta = -0.019716631966346654m
TTTSS_03B chunk10 h15 [200,300) delta = -2.154195732316907m
```

这说明当前最强动作不是 durable trajectory correction，而是 h10 局部 stabilizer。它能改善局部病灶和后段 proxy，但不能把 correction 写成持续到 h15 的 memory/state。

---

## 1. 对 v20 三条新增方向的判断

### 1.1 `past_only -> full_chunk`：目前证据不支持直接替换

Batch A 已经测试了 support variants。结果显示：support window variants 没有超过原始 `C23 past_only` cue，最好的 h10/h15 ATE delta 只有 `-0.43593138334250625m`，最好的 `[200,300)` delta 只有 `-0.693624840597316m`，远低于 gate。

同时，support audit 暴露了一个重要实现问题：

```text
full_chunk_no_overlap falls back to full_chunk because HMC cue builder has no external overlap metadata.
past_plus_future_light 当前 support count 仍是 31，它不是严格的 sparse temporal support，而是 weighted centroid 变体。
```

因此不能得出“full_chunk 一定没用”的最终结论，但可以得出一个更准确的结论：**当前 v20 版本里的 full_chunk/full_chunk_no_overlap/past_plus_future_light 没有提供足够强的短滚动证据，不能替代 C23 past_only。**

下一轮如果继续验证 full_chunk，重点不应是再跑同一个 alias，而是先修正 support 语义：

```text
1. full_chunk: 当前 chunk 内所有非自身帧，允许 future。
2. full_chunk_no_overlap: 真正排除 head/tail overlap seam，而不是 fallback。
3. past_plus_future_light: 真实加权聚合，而不是只在 summary 中标记。
4. causal_past_plus_near_future: 只加入 near future 1-2 帧，避免 full future 污染。
```

### 1.2 VGGT4D-style skip：有局部信号，但当前 attention-bias 近似不够持久

v20 实现的 context source skip 是受 VGGT4D 启发的 source-token 屏蔽：不删除 query token，不改变输出 token 形状，只屏蔽部分 source token。hard mode 使用 `-1e4` source-column bias，soft mode 使用 log keep-ratio bias。这个设计方向是对的，但它目前仍是 **attention-bias 近似**，不是完全等价于 VGGT4D 代码里的 K/V source compaction。

Batch B hard skip 的 best 局部信号是：

```text
KVS_01_FRAME_EARLY_DG_Q80_HARD
chunk10 h10 ATE delta = -0.7335244625430164m
chunk10 h10 [200,300) delta = -3.440533198161674m
chunk10 h10 [400,600) delta = -0.5762800137900612m
```

这说明 context source skip 确实能减少 disease window 的局部污染。但 h15 退化非常明显：

```text
KVS_01 chunk10 h15 ATE delta = +0.8351645148756539m
KVS_01 chunk10 h15 [200,300) delta = -0.9395844189062785m
KVS_01 chunk10 h15 [400,600) delta = +0.9104590883865811m
```

soft fallback 没有解决 h15 decay：

```text
KVS_09_FRAME_EARLY_DG_Q90_SOFT_R025
chunk10 h10 ATE delta = -0.5686290820475755m
chunk10 h10 [200,300) delta = -2.7600784604837614m
chunk10 h15 ATE delta = +1.1078553778640448m
```

所以当前结论是：**VGGT4D-style source skip 是有局部 read/source filtering 信号的，但当前 hard/soft bias 版本没有形成持久 memory correction。**

这也解释了为什么 `Scale + Context Skip` 的 h10 信号接近 gate，但 h15 collapse。source skip 很可能只修当前 controlled read 或短期 context，而没有把修正写入长期 TTT/SWA/global memory state。

### 1.3 语义 all-memory：当前还没有真正验证，只有 proxy blocker follow-up

v20 的 semantic memory 还没有完成真正的语义分组控制。当前 HMC control path 暴露到模型侧的是 `S_tok` semantic scalar，不是离散 semantic group id。Batch C 只做了一个 blocker follow-up：

```text
KVS_03_FRAME_EARLY_LOWSTUFF_HIGHD_HARD
mask = S_tok <= 0.45 and highD
```

它不是 exact sky / vegetation / movable / road / building group routing。结果如下：

```text
KVS_03 chunk10 h10 ATE delta = -0.5677100090959719m
KVS_03 chunk10 h10 [200,300) delta = -2.7635027280600184m
KVS_03 chunk10 h15 ATE delta = +1.1454205086597433m
KVS_03 chunk10 h15 [200,300) delta = +0.0421825365730939m
```

这说明 `lowstuff + highD` proxy 有局部 read filtering 信号，但不能说明“跳过天空/植被”已经被验证。真正语义 all-memory 还需要把 exact semantic group id 从 `MaskletOutput.G_sem` / Semantic Prior Generator 显式传入 HMC 和模型控制路径。

---

## 2. 问题本质：当前不是缺一个更细 gamma，而是缺 durable memory-state action

v20 最强 h10 信号来自：

```text
Scale commit + Dg q80 hard context skip
```

它说明 `scale-state` 和 `source-skip` 可以互补：scale commit 触碰 TTT trajectory-state，context skip 减少 high-D source pollution。h10 中 `[200,300)` delta 接近 `-5m`，下游 `[400,600)` 也没有 regression。

但是 h15 几乎没有 ATE 改善，说明修正没有稳定写进后续 memory/state。这是核心瓶颈。

换句话说，当前系统能做到：

```text
短期减少有害 source 影响；
短期调整 scale-aware TTT commit；
短期压低 [200,300) disease segment；
```

但做不到：

```text
把这个修正变成 h15 / full-sequence 持久轨迹状态；
在不牺牲 downstream continuity 的情况下持续降低 ATE；
让 TTT/SWA/global memory 都遵循同一个 semantic / dynamic role policy。
```

因此下一阶段不能只是继续微调：

```text
Dg q80/q85/q90
soft rho 0.2/0.3
scale gamma 0.020/0.025/0.030
```

这些只会继续得到 h10 local improvement 和 h15 decay。下一阶段必须验证更本质的东西：**source skip 是否需要从 read-only intervention 变成 memory-write intervention；semantic role 是否必须作用到所有 memory source；full_chunk support 是否能提供更稳定的 static manifold。**

---

## 3. 下一阶段总目标

v21 的目标不是直接刷一个新的 tiny best，而是回答四个机制问题：

1. **C23 support 是否还有空间**：`past_only` 是否真的最优，还是当前 `full_chunk` 实现不够准确？
2. **VGGT4D-style skip 是否应从 attention bias 近似升级为 true K/V source compaction**：hard bias 有局部效果但 h15 decay，是否 true K/V compaction 或 commit-aware skip 能增强持久性？
3. **语义是否能扩展为 all-memory role policy**：不是用 `S_tok` scalar，而是用 exact semantic group 控制 frame/global/SWA/TTT 的 source 和 write role。
4. **局部 h10 修正如何变成 h15 durable correction**：source skip 不能只修 read path，还要进入 TTT/SWA/global memory commit 或 trajectory-state carrier。

最终成功标准仍然是：

```text
KITTI01 full online deployable ATE <= 25m
```

阶段性 sandbox gate 为：

```text
h10/h15 ATE delta <= -3m
or [200,300) delta <= -5m with [400,600) regression <= +1m
```

并新增 durability gate：

$$
Durability = \frac{|\Delta ATE_{h15}|}{|\Delta ATE_{h10}| + \epsilon}
$$

```text
Durability >= 0.45 才允许进入 no-GT selector。
Durability < 0.20 视为短期局部修补，不允许 full online。
```

---

## 4. Phase 0：边界复现与 instrumentation gate

### 4.1 目标

在 v21 开始前，确认所有新 hook 不改变 H9/C9/WINGAM 边界，并确认 v20 新增的 context skip / semantic control / support alias 在 no-op 模式下严格安全。

### 4.2 实验

固定复现：

```text
H9_P0_R2
C9_P0_R2
WINGAM_P0_R3
```

新增 no-op smoke：

```text
N0_CONTEXT_SKIP_OFF
N1_CONTEXT_SKIP_MASK_EMPTY
N2_TRUE_KV_COMPACTION_OFF
N3_SEM_GROUP_PASS_THROUGH
N4_SUPPORT_ALIAS_PAST_ONLY_EXACT
```

### 4.3 必须记录

```text
ATE / Rot / RPE_t / RPE_r
FinalErr
[200,300), [200,400), [400,600]
hmc_state_hash.jsonl
context_skip_summary.jsonl
semantic_group_summary.jsonl
support_index_summary.json
run_config.yaml
```

### 4.4 通过标准

```text
|ATE - reference| <= 0.03m
|Rot - reference| <= 0.03deg
hmc rows = 38
no-op mode 下 num_context_source_skip_applied = 0
semantic group pass-through 不改变 prior_hmc_write_score_mean
support past_only indices 与 v20 audit 完全一致
```

若不通过，Codex 优先尝试：

```text
1. diff run_config.yaml，检查 beta / mp_alpha / commit mode / reset_every；
2. diff context_skip_summary，确认没有误触发 source skip；
3. diff semantic_group_summary，确认 no-op group mask 全 False；
4. diff support_index_summary，确认 support alias 没改变 C23 past。
```

---

## 5. Phase A：Support 重新验证，不再重复 v20 alias

### 5.1 假设

`past_only` 的优势可能不是因为未来帧无用，而是因为当前 full support 实现把 overlap seam / future inconsistent support 混入了 centroid。真正的 `full_chunk_no_overlap` 或 `past_plus_near_future` 可能提供更稳的 static manifold。

### 5.2 实验设计

只在修正 support metadata 后运行。候选：

```text
S0_C23_PAST_LOCKED
S1_C23_FULL_CHUNK_TRUE
S2_C23_FULL_CHUNK_NO_OVERLAP_TRUE
S3_C23_PAST_PLUS_NEAR_FUTURE12
S4_C23_PAST_PLUS_FUTURE_LIGHT_REAL
S5_C23_PAST_PLUS_STATIC_FUTURE_ONLY
```

定义：

```text
FULL_CHUNK_TRUE:
    当前 chunk 内所有非自身帧。

FULL_CHUNK_NO_OVERLAP_TRUE:
    排除 head/tail overlap seam frames。
    必须使用真实 overlap metadata，不允许 fallback。

PAST_PLUS_NEAR_FUTURE12:
    support = past frames + future frames with offset <= 2。

PAST_PLUS_FUTURE_LIGHT_REAL:
    centroid = 0.75 * past_centroid + 0.25 * future_centroid。
    如果某侧 support 为空，要单独记录，不得隐式退回 full。

PAST_PLUS_STATIC_FUTURE_ONLY:
    future support 只取 low-D_g / high-confidence / structure-like frames。
```

### 5.3 运行顺序

先做 support audit，不跑模型：

```text
support_index_summary.csv
support_index_by_frame.jsonl
support_future_mass.csv
support_overlap_exclusion_check.csv
```

再跑 h10/h15 sandbox：

```text
chunks = 6, 10, 16
horizons = h10, h15
parent = H9 causal fork snapshots
```

只有通过 sandbox gate 才进入 full online。

### 5.4 指标

```text
support_count_mean / min / max
future_support_ratio
overlap_removed_count
cue_D_mean
cue_D_p90
Mean D>0.5
fragmentation
anchor_collision
ATE_delta_h10/h15
[200,300]_delta_h10/h15
[400,600]_delta_h10/h15
Durability
```

### 5.5 成立标准

支持 `full_chunk` 替代 `past_only` 的条件：

```text
1. h10/h15 ATE delta 比 S0_C23_PAST_LOCKED 至少好 1.0m；
2. [200,300) h10/h15 至少一个 horizon 好 2.5m；
3. h15 不出现 ATE regression；
4. Durability >= 0.45；
5. support audit 证明 no_overlap 真的排除了 overlap seam。
```

如果 full_chunk 仍失败，Codex 不再重复 full_chunk 微扫，转向：

```text
1. per-head C23 past support；
2. static-future-only support；
3. support reliability gate，而不是 support window 变体。
```

---

## 6. Phase B：VGGT4D-style true K/V source compaction

### 6.1 假设

v20 的 context source skip 使用 source-column attention bias 近似跳过 source token。VGGT4D 的更强做法是保留所有 Query，但在早期层只让 Query 从非动态 K/V 读取。这个 true K/V compaction 可能比 `-1e4` bias 更干净，也能减少 high-D source 的 Value 聚合。

### 6.2 实现要求

新增 true compaction mode：

```text
CONTEXT_SOURCE_SKIP_IMPL = bias | compact_kv
```

`compact_kv` 必须满足：

```text
Query rows: all tokens kept
Key source: only keep non-skipped tokens
Value source: only keep non-skipped tokens
special tokens: always kept
empty source event: fallback to protected static tokens or disable skip for this layer
output shape: unchanged
```

对 frame attention：

```text
mask shape = per frame [B*S, N]
```

对 global attention：

```text
mask shape = full chunk [B, S*N]
```

### 6.3 实验矩阵

第一批只测早期层，避免 full-layer OOD：

```text
KVC_01_FRAME_EARLY_DG_Q80_COMPACT
KVC_02_FRAME_EARLY_DG_Q90_COMPACT
KVC_03_FRAME_EARLY_LOWSTUFF_HIGHD_COMPACT
KVC_04_GLOBAL_EARLY_DG_Q80_COMPACT
KVC_05_FRAME_GLOBAL_EARLY_DG_Q80_COMPACT
KVC_06_FRAME_EARLY_DG_Q80_BIAS_REPEAT
```

第二批只在第一批过局部 gate 时测：

```text
KVC_07_FRAME_EARLY_MID_DG_Q80_COMPACT
KVC_08_FRAME_EARLY_DG_Q80_COMPACT_WITH_STATIC_RESCUE
KVC_09_FRAME_EARLY_DG_Q80_COMPACT_COMMIT_AWARE
```

`COMMIT_AWARE` 定义：source skip 不是只影响 controlled read pass，而是额外跑一个 `write-probe-skip` cache，用于构造 TTT commit candidate。该 candidate 仍然不直接使用 controlled side effect，而是显式 replay：

```text
output = controlled read with skip
commit = probe/native base + explicit TTT write from skip-aware write cache
```

### 6.4 指标

```text
skip_impl
skip_scope
skip_layer_mode
skip_mask_source
mean_keep_ratio
p10/p50/p90_keep_ratio
num_empty_source_events
attention_mass_to_skipped_before
attention_mass_to_skipped_after
ATE_delta_h10/h15
[200,300]_delta_h10/h15
[400,600]_delta_h10/h15
Durability
TTT_commit_delta_norm_if_commit_aware
```

### 6.5 成立标准

true K/V compaction 通过条件：

```text
1. compact_kv h10 [200,300) delta 优于 bias repeat 至少 1.0m；
2. compact_kv h15 ATE 不 regression；
3. Durability >= 0.45；
4. empty_source_events = 0 或有明确 fallback；
5. mean_keep_ratio 不低于 0.45，避免过度 OOD。
```

如果 compact_kv h10 有信号但 h15 collapse，Codex 自动尝试：

```text
1. commit-aware skip-probe；
2. static rescue mask；
3. layers early only，不进 middle；
4. q80 -> q85，减少 skip mass；
5. only frame attention，不叠 global。
```

如果 compact_kv 完全不如 bias，Codex 回退到 bias path，并优先做 semantic exact group role。

---

## 7. Phase C：Exact semantic group id 接入

### 7.1 假设

语义不是一个 scalar value，而是 memory role assignment。当前 `S_tok` scalar 只能表达 low-value 程度，不能区分 sky / vegetation / road / building / movable / uncertain，因此无法真正验证“跳过天空”或“结构区域强写”。

### 7.2 必须先完成的工程

从 Stage C / D 到 HMC / model 传递：

```text
G_sem_tok: [L_tok] int semantic group id
L_sem_tok: [L_tok] optional fine label id
Q_sem_tok: [L_tok] semantic confidence / trust
M_sem_group_patch: [num_groups, T, Htok, Wtok] optional group masks
```

语义组至少包含：

```text
STRUCTURE_ANCHOR
LOW_VALUE_STUFF_SKY
LOW_VALUE_STUFF_VEGETATION
LOW_VALUE_STUFF_OTHER
MOVABLE_THING
STATIC_THING
UNCERTAIN_REGION
BACKGROUND_UNLABELED
```

如果当前 Mask2Former 或 cache 不能区分 tree/vegetation/sky，必须明确标记为 unavailable，不能继续用 `S_tok <= 0.45` 冒充 exact semantic role。

### 7.3 No-op gate

```text
SEM_GROUP_PASS_THROUGH
SEM_GROUP_ALL_KEEP
SEM_GROUP_DEBUG_ONLY
```

要求：

```text
ATE diff <= 0.01m
hmc hash unchanged
semantic group mass logged
```

### 7.4 语义 coverage audit

必须输出：

```text
semantic_group_coverage_by_chunk.csv
semantic_group_coverage_by_segment.csv
semantic_group_Dg_joint_mass.csv
semantic_group_confidence.csv
semantic_group_quality.csv
semantic_group_frame_grid.png
```

特别关注：

```text
sky coverage
vegetation coverage
road/building/wall/fence coverage
movable coverage
[200,300) focus chunks semantic distribution
chunks 10-16 semantic distribution
```

---

## 8. Phase D：Semantic all-memory role policy

### 8.1 总体假设

语义应控制所有 memory/source，而不是只控制 TTT write scalar。不同 memory 的 role 不同：

```text
Frame attention:
    谁能作为 early K/V source。

Global attention:
    谁能进入 global chunk context source。

SWA:
    谁能作为 previous/current local-memory K/V source。

TTT:
    谁作为 positive / neutral / negative write evidence。
```

### 8.2 语义角色规则 v1

不要 hard skip 所有天空。第一版语义规则如下：

```text
road/building/wall/fence/structure low-D:
    positive source / positive TTT write

sky low-D:
    neutral source, weak keep
    不作为强 positive，也不作为 negative

sky high-D or high-unc:
    weak source skip, weak negative TTT evidence

vegetation/tree/grass low-D:
    neutral source

vegetation/tree/grass high-D:
    conditional source skip, conditional weak negative

movable thing high-D:
    source skip, negative or no-write

uncertain region:
    no-write or weak negative
```

### 8.3 实验矩阵

#### D1：Frame attention semantic source skip

```text
SEMFA_01_SKY_HIGHD_FRAME_EARLY_COMPACT
SEMFA_02_VEG_HIGHD_FRAME_EARLY_COMPACT
SEMFA_03_MOVABLE_FRAME_EARLY_COMPACT
SEMFA_04_LOWSTUFF_HIGHD_FRAME_EARLY_COMPACT
SEMFA_05_STRUCTURE_RESCUE_DGQ80_FRAME_EARLY_COMPACT
```

#### D2：Global attention semantic source skip

```text
SEMGG_01_SKY_HIGHD_GLOBAL_EARLY_COMPACT
SEMGG_02_LOWSTUFF_HIGHD_GLOBAL_EARLY_COMPACT
SEMGG_03_STRUCTURE_RESCUE_GLOBAL_EARLY_COMPACT
```

#### D3：SWA semantic source policy

```text
SEMSWA_01_PREV_LOWSTUFF_HIGHD_WEAK_SKIP
SEMSWA_02_PREV_STRUCTURE_KEEP_LOWSTUFF_CONDITIONAL_SKIP
SEMSWA_03_TAIL_OVERLAP_SKY_NEUTRAL_VEG_HIGHD_SKIP
```

SWA 只允许作为 short-rollout diagnostic，先不做 full online。

#### D4：TTT semantic positive / neutral / negative write

```text
SEMTTT_01_POS_STRUCTURE_NEU_LOWSTUFF_NEG_HIGHD
SEMTTT_02_POS_STRUCTURE_NEG_SKY_HIGHD_ONLY
SEMTTT_03_POS_STRUCTURE_NEG_VEG_HIGHD_ONLY
SEMTTT_04_POS_STRUCTURE_NEG_MOVABLE_ONLY
SEMTTT_05_STAGE_D_X_SEMROLE_X_DG_INV_SQRT
```

TTT replay objective：

$$
G_{commit} = G_{pos} + \lambda_{neu}G_{neu} - \gamma G_{neg}
$$

其中：

```text
G_pos = structure ∩ lowD ∩ high semantic trust
G_neu = sky/vegetation lowD or unlabeled stable region
G_neg = highD ∩ semantic-risk group ∩ sufficient trust
```

### 8.4 指标

除全局轨迹指标外，必须记录：

```text
per_group_source_skip_mass
per_group_keep_ratio
per_group_attention_mass_before_after
per_group_ttt_write_score_mean
per_group_positive_mass
per_group_neutral_mass
per_group_negative_mass
per_group_update_norm_by_branch_layer
per_group_update_cosine_to_native
per_group_update_conflict_energy
semantic_group_h10_h15_effect_decay
```

### 8.5 成立标准

Semantic all-memory policy 通过条件：

```text
1. exact semantic group id available，不接受 S_tok scalar proxy；
2. 至少一个 semantic role policy 达到 h10/h15 ATE delta <= -3m；
3. 或 [200,300) delta <= -5m with [400,600) regression <= +1m；
4. h15 Durability >= 0.45；
5. 不允许 sky hard-skip 导致 [400,600) 或 Sim3Scale 明显恶化。
```

如果 sky skip 失败，Codex 自动转向：

```text
sky neutral keep + highD weak skip;
vegetation highD conditional skip;
structure rescue;
semantic only TTT role，不碰 frame/global source。
```

如果 all semantic policies 都只 h10 有效 h15 collapse，则说明语义是 read/local stabilizer，不是 persistence mechanism；下一步转向 commit-aware skip-probe。

---

## 9. Phase E：把 h10 局部修正变成 h15 durable memory correction

### 9.1 假设

v20 最强结果 `TTTSS_03B` 接近 h10 gate，但 h15 collapse。根因可能是 context skip 只作用于短期 read/source，没有改变后续 memory commit。需要将 source skip 产生的 clean context 变成 TTT/SWA/global memory 的 commit candidate。

### 9.2 实验方向

#### E1：skip-aware write-probe TTT commit

```text
SKIPTTT_01_DGQ80_FRAME_EARLY_SKIP_PROBE_WRITE
SKIPTTT_02_DGQ80_FRAME_GLOBAL_EARLY_SKIP_PROBE_WRITE
SKIPTTT_03_SEM_LOWSTUFF_HIGHD_SKIP_PROBE_WRITE
SKIPTTT_04_STRUCTURE_RESCUE_SKIP_PROBE_WRITE
```

协议：

```text
Pass 1 native probe: collect native output and native write cache.
Pass 1b skip-write probe: same H_m, apply source skip only to collect write cache, output discarded.
Pass 2 controlled read: source skip affects current output.
Commit: probe/native base + explicit TTT write from skip-write cache.
```

这不是 controlled commit。它仍然保持 HMC 的安全隔离，但新增一个专门为 write 设计的 clean-context probe。

#### E2：skip-aware SWA cache commit

```text
SKIPSWA_01_PREV_SOURCE_SKIP_WRITE_CACHE
SKIPSWA_02_STRUCTURE_KEEP_LOWSTUFF_SKIP_CACHE
SKIPSWA_03_DGQ80_TAIL_OVERLAP_CACHE_SKIP
```

只在 sandbox 中验证。

#### E3：scale-commit + skip persistence

```text
TTTSSP_01_SCALECOMMIT_DGQ80_SKIP_WRITEPROBE
TTTSSP_02_SCALECOMMIT_DGQ80_SKIP_WRITEPROBE_SEM_STRUCTURE_RESCUE
TTTSSP_03_SCALECOMMIT_DGQ80_SKIP_WRITEPROBE_LONGSHORT
```

### 9.3 指标

```text
h10/h15 ATE delta
h10/h15 [200,300) delta
h10/h15 [400,600) delta
Durability
TTT write cache delta norm native vs skip-probe
post-zp delta cosine native vs skip-probe
SWA history K/V diff
commit state hash
scale proxy before/after
```

### 9.4 成立标准

```text
h10 ATE delta <= -3m 或 [200,300) delta <= -5m；
h15 ATE delta 保留至少 h10 的 45%；
[400,600) regression <= +1m；
commit debug 显示 skip-probe write cache 确实不同于 native cache。
```

如果 skip-probe write cache 与 native cache 差异很小，Codex 不继续 full，转向：

```text
post-zp delta routing by semantic/source-skip basis;
scale-state explicit commit;
trajectory-state module。
```

---

## 10. Phase F：No-GT selector 与 full online validation

只有候选通过 sandbox gate 才启动。

### 10.1 Selector 输入

```text
short rollout proxy:
    no-GT scale proxy
    overlap pointmap consistency
    attention source keep ratio
    semantic group skip mass
    TTT apply mismatch
    TTT update conflict energy
    SWA source consistency
```

### 10.2 Selector gate

```text
Spearman(proxy, h10 ATE delta) >= 0.45
Spearman(proxy, h15 ATE delta) >= 0.35
selected candidate not worse than H9 in >= 70% chunks
```

### 10.3 Full online gate

```text
Stage target:
    ATE <= 32m
    [200,300) <= 68m
    no [400,600) regression over C9 by more than 1m

Intermediate target:
    ATE <= 30m

Final target:
    ATE <= 25m
```

如果 full online fails but sandbox passed，Codex 自动：

```text
1. 检查 selector 是否选择了错误 candidate；
2. 检查 h15 -> full effect decay；
3. 加 reset-window continuity protection；
4. 若连续 3 个 selector full 失败，则停止该 family。
```

---

## 11. 必须可视化的内容

每个 track 都必须生成 dashboard。

### 11.1 Support dashboard

```text
support indices by frame
future / past support mass
cue D_g map: past vs full_chunk vs no_overlap
D_diff map
fragmentation / anchor collision map
support error scatter
```

### 11.2 Context skip dashboard

```text
RGB
D_g
semantic group overlay
skip mask
keep ratio map
attention mass to skipped source before/after
source K/V count per layer
empty source event map
h10 vs h15 effect decay plot
```

### 11.3 Semantic all-memory dashboard

```text
semantic group coverage by chunk
semantic group coverage in [200,300)
per-group D_g distribution
per-group write mass
per-group skip mass
per-group update norm by TTT branch/layer
per-group attention source mass before/after
sky/vegetation/structure failure gallery
```

### 11.4 Trajectory dashboard

```text
XY trajectory: H9 / C9 / candidate / GT
per-100f ATE curve
[200,300), [200,400), [400,600] bar chart
Sim3 scale over time
Yaw drift over time
reset-group boundary error
h10/h15 effect decay
```

---

## 12. 并行执行计划

为了加速，分成五个 Codex track 并行推进。

### Track A：support 修正与 sandbox

负责人目标：

```text
实现 true full_chunk_no_overlap metadata。
修正 past_plus_future_light 真实加权。
跑 Phase A support audit + h10/h15 sandbox。
```

如果 `full_chunk` 失败：

```text
停止 full_chunk 微扫；转 per-head C23 past 或 static-future-only。
```

### Track B：true K/V compaction

负责人目标：

```text
实现 compact_kv source skip。
比较 bias vs compact。
先 frame early，再 global early。
```

如果 h10 有效 h15 decay：

```text
转 commit-aware skip-probe。
```

如果 compact 不如 bias：

```text
保留 bias path，集中 semantic exact role。
```

### Track C：semantic exact group id

负责人目标：

```text
把 G_sem_tok / L_sem_tok / Q_sem_tok 从 Stage C/D 传到 HMC/model。
完成 no-op gate 和 coverage audit。
```

如果 semantic group coverage 不足：

```text
修 Stage C cache 或 Mask2Former label mapping；不跑 semantic all-memory full。
```

### Track D：semantic all-memory role

负责人目标：

```text
Frame/global/SWA/TTT 四类 memory/source role policy。
先 sandbox，不 full-first。
```

如果 sky hard skip 失败：

```text
改成 sky neutral + highD weak skip。
```

如果 movable coverage 为空：

```text
暂不做 movable role，集中 structure/sky/vegetation。
```

### Track E：durability / persistence

负责人目标：

```text
实现 skip-aware write-probe commit。
组合 scalecommit + context skip。
测试 h15 durability。
```

如果 skip-probe write cache 与 native cache 差异很小：

```text
转 post-zp source-skip basis 或 explicit trajectory-state module。
```

---

## 13. 最终停止规则

如果出现以下任一情况，停止对应 family：

```text
1. 同一 family 4 条 h10/h15 sandbox 都无法达到 h10 ATE delta <= -1m；
2. h10 有效但连续 3 条 h15 Durability < 0.20；
3. [200,300) 改善但 [400,600) regression > +2m；
4. semantic group id 不可用，禁止继续 semantic scalar proxy 冒充 exact semantic；
5. compact_kv empty source events > 5% frames/layers；
6. full_chunk_no_overlap 仍 fallback 到 full_chunk。
```

如果 Phase A-E 全部失败，则下一阶段结论应明确改为：

```text
TTT write-only / context-source filtering / semantic memory role 在当前 action interface 下不足以支撑 Target-25。
TTT 和 context skip 保留为 stabilizer / regularizer。
Target-25 主线转向显式 online trajectory-state / scale-state module，
但继续使用 C23/semantic/skip 作为该模块的 observation cue。
```

---

## 14. 本计划的核心判断

v20 的真正信息不是“又失败了”，而是：

```text
1. C23 past_only 仍然是最稳 support，目前没有被 full_chunk 替代；
2. VGGT4D-style source skip 有真实 h10 局部信号，但 h15 不持久；
3. 语义 all-memory 还没有被真正验证，因为 exact semantic group id 未接入；
4. Scalecommit + Dg-q80 source skip 是当前最接近 gate 的 diagnostic，但仍缺 durability；
5. 下一阶段的关键不是再调阈值，而是让 context skip / semantic role 进入 memory commit，使局部修正成为持久 correction。
```

因此 v21 的主线应是：**support 只做修正验证，真正重点放在 true K/V source compaction、exact semantic role、以及 skip-aware memory commit。**

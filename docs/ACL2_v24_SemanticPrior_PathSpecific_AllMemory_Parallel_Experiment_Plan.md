# ACL2 v24：Semantic Prior Generator 优先的 All-Memory Path-Specific Role 实验计划

日期：2026-05-21  
目标数据集：KITTI Odometry Sequence 01  
当前最好可计数 online TTT write：`C9_P0_R2`, `ATE = 33.7629421029m`  
当前目标：`KITTI01 ATE <= 25.0m`  
当前 gap：

$$
33.7629421029 - 25.0 = 8.7629421029m
$$

本计划的核心任务是：**优先探索 Semantic Prior Generator 对所有 memory path 的控制方式**，包括：

```text
frame attention
global / chunk attention
SWA local memory
TTT fast-weight memory
```

但本计划不再把语义当成一个简单 scalar prior，也不再盲目把同一个语义 mask 同时套到所有 memory。v23 已经证明：工程接线基本成功，但粗粒度 all-memory semantic role 没有带来有效、持久的轨迹改善。因此 v24 的核心改变是：

> 从 “semantic scalar / coarse role” 升级为 “path-specific semantic memory role controller”。

---

## 0. 对 v23 结果的独立判断

v23 的结果需要分成两层看。

### 0.1 工程层面：通过

v23 完成了 Semantic Prior Generator 的 all-memory 接线，新增了：

```text
V_sem_tok
R_sem_tok
R_sem_patch_flat
semantic_role_policy
semantic_memory_paths
semantic_role_summary.jsonl
semantic_memory_path_summary.jsonl
```

并且 implementation self-audit、dynamic smoke、no-op / pass-through parity 都通过：

```text
hard_static_gate_pass = true
dynamic_smoke_gate_pass = true
all_gate_pass = true
ATE_delta_vs_H9 = 0
raw_trans_max_diff = 0
```

这说明当前失败不是因为语义完全没接上，也不是 no-op 漂移造成的。

### 0.2 科学层面：没有成功

v23 完成了：

```text
Phase 2 single-path matrix: 36/36 rows
Phase 3 all-memory matrix: 36/36 rows
```

但结果很弱：

```text
Phase 2 best:
FRAME_SEM_01 chunk16 h15 ATE delta = -1.0299m

Phase 3 best:
ALLSEM_04 chunk16 h15 ATE delta = -0.9515m

Phase 3 best [200,300) delta:
+0.8707m
```

也就是说，语义 role 对后段 chunk16 有一点弱稳定作用，但没有改善核心 `[200,300)` 病灶，反而让它更差。因此：

```text
No no-GT selector
No full online validation
No online Target-25
```

当前最好 deployable online 仍是 `C9_P0_R2 = 33.7629421029m`。

### 0.3 当前最核心问题

v23 的失败不是“语义没用”，而是：

> 当前语义 role 太粗，而且没有按 memory path 分职责。

具体说：

```text
frame/global attention:
    语义更像 K/V source eligibility

SWA:
    语义更像 local overlap/cache topology policy

TTT:
    语义更像 positive / neutral / negative write evidence

budget:
    不应该由语义直接决定，而应主要由几何稳定性、TTT conflict、trajectory-state risk 决定
```

因此，v24 不再做 “all-memory 同一 role policy” 的大矩阵，而是按 path 拆开职责，先找每条 memory path 中语义真正能起作用的位置，再做组合。

---

## 1. 实验整体目标

v24 的整体目标是建立一套 **Semantic Prior Generator driven all-memory role policy**，让语义不只是输出一个 `A_tok` 或 `S_tok`，而是为不同 memory path 输出不同角色。

我们要回答五个核心问题。

### 1.1 问题 A：语义 role 是否真的与 token / patch 对齐？

如果 `R_sem_tok`、`V_sem_tok`、masklet cache、global frame index、patch token index 有任何错位，后续实验都会变成噪声。因此 v24 第一阶段必须让 Codex 自查并证明：

```text
semantic group projection 正确；
role token 对齐正确；
Stage C cache 命中正确；
path-specific control 真正消费 role；
no-op / pass-through 不扰动结果。
```

### 1.2 问题 B：不同 memory path 是否需要不同语义职责？

v23 的粗粒度 all-memory 策略失败，说明同一个 role 不能同时解释所有 memory path。v24 要验证：

```text
frame/global source skip 是否有局部 read/source 改善；
SWA cache role 是否能提高 h10->h15 durability；
TTT semantic write 是否能提供 long-memory positive/negative evidence；
不同 path 组合是否互补，还是互相抵消。
```

### 1.3 问题 C：语义必须和 D_g / TTT conflict / geometry trust 条件化吗？

语义类别本身不是漂移的直接因果变量。例如：

```text
sky:
    可能不是好的 3D anchor，但可能提供 horizon / scale continuity

vegetation:
    可能是非刚体风险，也可能是稳定远景背景

road/building/fence/wall:
    可能是结构 anchor，但如果 D_g 高或 TTT conflict 高，也可能是错误 source
```

所以 v24 要把语义 role 改成条件式：

$$
role_i = f(G_{sem,i}, D_i, C^{ttt}_i, Q_i, U_i)
$$

其中：

```text
G_sem: semantic group
D: ACL2 internal attention risk
C_ttt: TTT update conflict
Q: mask trust / semantic trust
U: uncertainty / geometry risk
```

### 1.4 问题 D：h10 有效但 h15 失效，到底是谁覆盖了修正？

v20-v22 多次出现：

```text
h10 [200,300) 改善明显；
h15 ATE / [200,300) 效果衰减或反向；
full online 不允许启动。
```

v24 必须追踪修正被谁覆盖：

```text
TTT 后续写入覆盖？
SWA cache refresh 覆盖？
global/chunk source update 覆盖？
merge/gauge state 覆盖？
semantic role 在后续 chunk 中消失？
```

### 1.5 问题 E：什么时候才允许启动 no-GT selector / full online？

v24 不允许因为某个 short-rollout 局部好就直接 full。必须满足 durability gate：

$$
DR = \frac{\max(0, -\Delta ATE_{h15})}{\max(\epsilon, -\Delta ATE_{h10})}
$$

其中 $DR$ 是 durability ratio。只有满足：

```text
h10/h15 ATE delta 足够大
或 h10/h15 [200,300) delta 足够大
且 [400,600) 不崩
且 DR >= 0.45
```

才允许进入 no-GT selector。selector 再过 gate 后，才允许 full online validation。

---

## 2. 总体实验原则

v24 不再把 full KITTI01 当作默认验证方式。所有实验按以下顺序推进：

```text
implementation audit
    -> passive semantic attribution
    -> single-path short rollout
    -> pairwise path combination
    -> all-memory path-specific role
    -> durability attribution
    -> no-GT selector
    -> full online validation
```

每一步都有 hard gate。如果某一步失败，Codex 必须根据 fallback 规则自动转向，而不是继续扫同类阈值。

---

## 3. 固定边界与主参考

### 3.1 Deployable online reference

当前最好可计数 online TTT write：

```text
C9_P0_R2
ATE = 33.7629421029m
Rot = 6.5259
[200,300) = 76.102136m
[400,600) = 41.896364m
counts_as_ttt_write = true
```

重要说明：C9 全局 ATE 最低，但 `[200,300)` 比 H9 更差，所以它不是健康 target path，只是 deployable online best。

### 3.2 H9 causal-fork parent

v24 short-rollout 仍建议以 H9 parent 为主：

```text
H9_P0_R2
ATE = 34.1257769401m
[200,300) = 74.409927m
[400,600) = 44.353638m
```

原因：

```text
H9 在 [200,300) 病灶段比 C9 更好；
v16 之后 causal fork parity 已经可信；
H9 更适合作为病灶改善的 parent。
```

### 3.3 Diagnostic upper bound

`NOGTPOSE_27` 只能作为 diagnostic upper bound，不计入 TTT / semantic success：

```text
NOGTPOSE_27
ATE ≈ 22.4m
counts_as_ttt_write = false
```

它说明 target-25 的 trajectory-state / scale-state 方向存在，但不等于 TTT 或 semantic memory 已经成功。

---

## 4. 语义角色定义

v24 使用 path-specific role，不再只使用一个全局 `R_sem_tok` 策略。

### 4.1 基础语义组

至少需要这些 coarse / fine group：

```text
STRUCTURE_ANCHOR:
    road, building, wall, fence, pole-like static structure

LOW_VALUE_STUFF:
    sky, vegetation, grass, water, reflection, screen

MOVABLE_THING:
    car, person, rider, bicycle, motorcycle, bus, truck

STATIC_THING:
    parked vehicle, traffic sign, large static object, barrier

UNCERTAIN_REGION:
    low mask trust, ambiguous label, fragmented mask, uncovered region
```

如果 fine label 当前不可用，Codex 应优先补：

```text
sky
vegetation
road
building
wall/fence
movable thing
unknown/uncertain
```

不要继续把 sky、vegetation、grass 全部混成一个不可解释的 `LOW_VALUE_STUFF`。

### 4.2 Path-specific roles

对于每个 token $i$，Semantic Prior Generator 应输出至少四条 role stream：

```text
R_frame_i
R_global_i
R_swa_i
R_ttt_i
```

每条 stream 的 role 含义不同。

#### Frame / global attention role

```text
SOURCE_KEEP:
    可以作为 K/V source

SOURCE_SKIP:
    不应作为 K/V source

SOURCE_SOFT_KEEP:
    以较低权重保留

SOURCE_PROTECT:
    即使 D_g 高也保护，避免误删 horizon / structure
```

#### SWA role

```text
CACHE_KEEP:
    可以进入 previous/current SWA cache

CACHE_SOFT_KEEP:
    低权重进入 cache

CACHE_DROP:
    不进入 cache 或被替换

CACHE_PROTECT:
    保护 overlap continuity source
```

#### TTT role

```text
POSITIVE_LONG:
    正向写入长期 fast weights

NEUTRAL_KEEP:
    保留连续性，但不作为强 anchor

NEGATIVE_SHORT:
    短期负证据或弱反向，不能长期污染

PROTECT_NEUTRAL:
    不强写、不强删，保护 scale / horizon / continuity
```

---

## 5. Phase 0：Codex implementation audit hard gate

### 5.1 目标

在任何性能实验前，Codex 必须自查代码实现是否可信。该阶段目标不是提升 ATE，而是证明 semantic role 控制链路真的生效。

### 5.2 Codex 必须检查的代码路径

必须检查：

```text
run_pipeline_abc_v2.py
loger/pipeline/semantic_prior_generator.py
loger/pipeline/hybrid_memory_controller.py
loger/models/pi3.py
loger/models/layers/attention.py
loger/pipeline/ttt_write_controller.py
tools/run_attention_cue_experiment.sh
tools/run_v23/v24 candidate launcher
```

### 5.3 必须检查的 bug 类型

#### 参数传递 bug

Codex 必须确认：

```text
SEMANTIC_ROLE_* 环境变量进入 Python CLI；
semantic_memory_paths 被正确解析；
readonly / hybrid / short-rollout 都收到相同 semantic role args；
context_source_skip_impl 在 readonly 和 hybrid 下都被转发；
stage_c_cache_dir / require_hit / validate 生效。
```

#### Stage C cache bug

必须确认：

```text
causal fork 使用 global frame start/end；
chunk_006_000174_000206 这类 global chunk key 能命中；
不会退回到 chunk_006_000000_000032；
cache miss 时 require_hit 会 fail，不会 silent fallback；
inline Stage C compute 不参与 parity run。
```

#### semantic projection bug

必须确认：

```text
masklet group -> patch map -> token map 对齐；
R_sem_patch_flat 与 PatchMeta 对齐；
R_sem_tok 长度等于 token count；
special tokens role 合法；
semantic group counts 非空；
fine label 没有被全部 fallback。
```

#### path consumption bug

必须确认四条 path 都真的消费 role：

```text
frame/global:
    source keep ratio 有变化；
    context_skip_summary 有 applied rows；
    empty source events = 0。

SWA:
    semantic group cache mass 有变化；
    cache keep/drop summary 非空。

TTT:
    per-role write mass 非空；
    branch update norm 有变化；
    positive / neutral / negative role mass 非空。

debug-only/no-op:
    只记录，不改变 trajectory。
```

#### stale run bug

必须确认：

```text
forced rerun 会移动旧目录到 .INVALID_RERUN_*；
report 排除 INVALID 目录；
jsonl 不混旧 run；
run_config hash 与 expected 一致；
没有 partial DONE 被算入结果。
```

### 5.4 Phase 0 必须落盘

```text
implementation_audit/codex_self_check_report.md
implementation_audit/codex_self_check_summary.json
implementation_audit/codex_self_check_failures.jsonl
implementation_audit/semantic_role_alignment_audit.csv
implementation_audit/path_consumption_audit.csv
implementation_audit/noop_parity_metrics.csv
```

### 5.5 Phase 0 通过标准

Phase 0 通过条件：

```text
hard_static_gate_pass = true
dynamic_smoke_gate_pass = true
all_gate_pass = true

no-op ATE delta = 0
raw_trans_max_diff = 0
semantic role counts non-empty
semantic memory path metrics non-empty
context source skip applied when requested
empty source events = 0
no stale run contamination
```

如果 Phase 0 不通过，Codex 不允许启动任何 Phase 1-6 性能实验。

### 5.6 Phase 0 失败时 Codex 自动尝试方向

如果 semantic role counts 为空：

```text
检查 Stage C cache 命中；
检查 G_sem / L_sem 是否从 MaskletOutput 进入 SPG；
检查 coarse group taxonomy；
检查 R_sem_patch_flat 是否全 FALLBACK；
补 fine label mapping 或 fallback group mapping。
```

如果 path metrics 为空：

```text
检查 semantic_memory_paths 解析；
检查 HMC prior 是否带 R_sem_tok；
检查 pi3.py / attention.py 是否消费 source skip；
检查 TTTWriteController 是否接收 semantic role score；
加 smoke log，不跑 full。
```

如果 no-op 漂移：

```text
关闭所有 semantic consumed path；
只保留 debug-only；
逐 path 开启，定位漂移源；
确认 pass-through role 不改变 write score；
确认 inline Stage C 不参与 benchmark。
```

---

## 6. Phase 1：Passive semantic all-memory attribution

### 6.1 目标

Phase 1 不改变模型输出，只被动统计：

```text
不同 semantic group 在每条 memory path 中的覆盖、风险、写入、source、cache、冲突分布。
```

目标是先回答：

> 哪些语义组真的参与了 memory？它们在 h10/h15 的错误变化中扮演什么角色？

### 6.2 实验设计

固定 parent：

```text
H9_P0_R2
chunks = 6, 10, 16
horizons = h10, h15
mode = debug-only / no-op
```

对每个 chunk/horizon 记录语义 group 与 memory path 的 passive metrics，不做 source skip、不做 TTT write 改动、不改 SWA cache。

### 6.3 必须记录的指标

#### semantic coverage

```text
group_coverage
group_patch_count
group_token_count
mask_trust_mean
mask_trust_p10/p50/p90
semantic_entropy
fallback_ratio
fine_label_distribution
```

#### 与几何 / attention cue 的关系

```text
D_g_mean_by_group
D_g_p90_by_group
uncertainty_mean_by_group
confidence_mean_by_group
C_anchor_mean_by_group
old_dyn_overlap_by_group
```

#### 与 TTT-native cue 的关系

```text
update_conflict_energy_mean_by_group
update_conflict_energy_p90_by_group
post_zp_update_norm_by_group
TTT_positive_mass_by_group
TTT_neutral_mass_by_group
TTT_negative_mass_by_group
branch_w0/w1/w2_update_norm_by_group
```

#### 与 memory source 的关系

```text
frame_source_mass_by_group
global_source_mass_by_group
SWA_previous_source_mass_by_group
SWA_current_source_mass_by_group
SWA_cache_keep_mass_by_group
```

#### 与 error 的关系

```text
h10_ATE_delta
h15_ATE_delta
h10_[200,300]_delta
h15_[200,300]_delta
h10_[400,600]_delta
h15_[400,600]_delta
```

### 6.4 必须可视化

每个重点 chunk 至少输出：

```text
RGB
semantic group map
fine label map
D_g map
update_conflict_energy map
frame/global source mass map
SWA cache mass map
TTT role mass map
[200,300) / [400,600) trajectory error plot
```

还要输出：

```text
semantic_group_vs_Dg_scatter.png
semantic_group_vs_ttt_conflict_scatter.png
semantic_group_memory_path_heatmap.png
semantic_group_error_correlation.png
```

### 6.5 Phase 1 通过标准

Phase 1 不要求 ATE 提升，但必须产生可解释 attribution：

```text
1. 至少 4 个主要 semantic group 有非空覆盖；
2. D_g / update_conflict / source mass by group 非空；
3. 至少一个 group 在某条 memory path 上有显著 role difference；
4. 能识别 v23 all-memory 失败是否来自：
       role 粗；
       path 相互抵消；
       semantic taxonomy 粗；
       role 没有实际改变 memory mass。
```

如果 Phase 1 不能解释任何差异，Codex 必须暂停性能矩阵，先修 semantic taxonomy / logging。

---

## 7. Phase 2：Single-path semantic role experiments

Phase 2 是 v24 的第一个性能阶段，但仍然只跑 short-rollout，不跑 full online。

### 7.1 总规则

固定：

```text
parent = H9_P0_R2
chunks = 6, 10, 16
horizons = h10, h15
read cue = C23 past
SWA/TTT locked base = H9 protocol
```

每个 path 单独测试，不做 all-memory 组合。

### 7.2 Track A：Frame/global semantic K/V source policy

#### 假设

语义在 frame/global attention 中最适合控制 K/V source，而不是直接控制 TTT write。

#### 角色规则

```text
STRUCTURE_ANCHOR + lowD:
    SOURCE_KEEP

LOW_VALUE_STUFF + lowD:
    SOURCE_SOFT_KEEP

LOW_VALUE_STUFF + highD:
    SOURCE_SKIP or SOURCE_SOFT_SKIP

MOVABLE_THING + highD:
    SOURCE_SKIP

UNCERTAIN_REGION:
    SOURCE_SOFT_SKIP unless protected
```

#### 候选

```text
FRAMESEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT
FRAMESEM_02_LOWSTUFF_HIGHD_SKIP
FRAMESEM_03_SKY_NEUTRAL_VEGETATION_HIGHD_SKIP
GLOBALSEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT
GLOBALSEM_02_LOWSTUFF_HIGHD_SKIP
FRAMEGLOBAL_01_FRAME_ONLY
FRAMEGLOBAL_02_GLOBAL_ONLY
FRAMEGLOBAL_03_FRAME_AND_GLOBAL
```

#### 必须记录

```text
source_keep_ratio_by_group
source_skip_ratio_by_group
mean_context_source_keep_ratio
num_context_source_skip_applied
num_context_empty_source_events
attention_mass_to_group_before_after
h10/h15 trajectory deltas
```

### 7.3 Track B：SWA semantic cache policy

#### 假设

SWA 是 local memory，不应该和 TTT 使用同一语义策略。它的语义 role 应该保护 overlap continuity，同时减少高风险 previous-source 污染。

#### 角色规则

```text
STRUCTURE_ANCHOR + lowD:
    CACHE_KEEP

LOW_VALUE_STUFF + lowD:
    CACHE_SOFT_KEEP

LOW_VALUE_STUFF + highD:
    CACHE_SOFT_DROP

MOVABLE_THING + highD:
    CACHE_DROP

sky_lowD:
    CACHE_PROTECT or CACHE_SOFT_KEEP

vegetation_highD:
    CACHE_SOFT_DROP
```

#### 候选

```text
SWASEM_01_STRUCTURE_CACHE_KEEP
SWASEM_02_LOWSTUFF_HIGHD_CACHE_SOFTDROP
SWASEM_03_SKY_PROTECT_VEG_HIGHD_DROP
SWASEM_04_PREVIOUS_SOURCE_ONLY
SWASEM_05_OVERLAP_ONLY
SWASEM_06_CURRENT_AND_PREVIOUS_COMPARE
```

#### 必须记录

```text
SWA_cache_keep_mass_by_group
SWA_cache_drop_mass_by_group
SWA_previous_source_attention_mass_by_group
SWA_current_source_attention_mass_by_group
overlap_pointmap_residual
boundary_10f_ATE
boundary_20f_ATE
chunk_boundary_pose_jump
h10/h15 trajectory deltas
```

### 7.4 Track C：TTT semantic write policy

#### 假设

TTT 是 long memory，语义应该辅助 positive / neutral / negative evidence，而不能单独决定写入强度。

#### 角色规则

```text
positive_long =
    STRUCTURE_ANCHOR
    AND lowD
    AND low update_conflict
    AND high confidence

neutral_keep =
    LOW_VALUE_STUFF
    AND lowD
    AND low uncertainty

negative_short =
    highD
    AND high update_conflict
    AND (MOVABLE_THING or LOW_VALUE_STUFF or low trust)

protect_neutral =
    sky lowD
    or horizon-like region
    or stable far structure
```

#### 候选

```text
TTTSEM_01_STRUCTURE_POSITIVE_LONG
TTTSEM_02_LOWSTUFF_HIGHD_NEGATIVE_SHORT
TTTSEM_03_SKY_NEUTRAL_PROTECT
TTTSEM_04_SEMANTIC_PLUS_TTT_CONFLICT
TTTSEM_05_SEMANTIC_PLUS_DG_PLUS_CONFLICT
TTTSEM_06_ROLE_SPECIFIC_BRANCH_W0
TTTSEM_07_ROLE_SPECIFIC_LONG_SHORT
```

#### 必须记录

```text
TTT_positive_mass_by_group
TTT_neutral_mass_by_group
TTT_negative_mass_by_group
TTT_protect_mass_by_group
branch_w0/w1/w2_update_norm_by_role
post_zp_delta_norm_by_role
update_conflict_energy_by_role
memory_state_rel_diff
h10/h15 trajectory deltas
```

### 7.5 Track D：Global/chunk attention semantic source policy

如果 global/chunk path 和 frame path 在代码上不同，需要单独测：

```text
CHUNKSEM_01_STRUCTURE_KEEP
CHUNKSEM_02_LOWSTUFF_HIGHD_SKIP
CHUNKSEM_03_PROTECT_SPECIAL_TOKENS
```

记录：

```text
global_source_keep_ratio
chunk_attention_source_mass_by_group
special_token_source_mass
global_context_update_norm
```

### 7.6 Phase 2 单 path 通过标准

单 path 候选要进入 Phase 3，必须满足至少一个：

```text
h10 [200,300) delta <= -3m
或 h15 ATE delta <= -1.5m
或 h15 [200,300) delta <= -2.5m
```

并且必须满足：

```text
[400,600) regression <= +1m
empty source events = 0
no-op parity 不破坏
memory path metrics 证明该 path 确实改变了 source/write mass
```

如果某个 path 所有候选都不满足，Codex 不允许继续微调该 path 阈值。必须根据 fallback 转向：

```text
无 source mass 变化:
    修 path consumption bug

source mass 有变化但 trajectory 无变化:
    该 path 暂停，转其它 path

h10 有效 h15 失效:
    转 durability attribution

[200,300) 改善但 [400,600) 崩:
    加 continuity protection / protect_neutral，不继续扩大 skip
```

---

## 8. Phase 3：Pairwise semantic path combination

### 8.1 目标

v23 的 all-memory 直接组合失败。v24 先做 pairwise 组合，检查 path 之间是否互补或抵消。

### 8.2 允许组合的前提

只有 Phase 2 过 gate 的 single-path 候选才能进入组合。

例如：

```text
FRAMESEM_best + TTTSEM_best
FRAMESEM_best + SWASEM_best
SWASEM_best + TTTSEM_best
GLOBALSEM_best + TTTSEM_best
FRAMESEM_best + GLOBALSEM_best
```

不允许把未过 gate 的所有 path 强行 all-memory。

### 8.3 必须记录 path cancellation matrix

对每个组合，记录：

```text
single_path_delta_A
single_path_delta_B
pair_delta
interaction = pair_delta - min(single_path_delta_A, single_path_delta_B)
```

对于 ATE delta，若 `interaction > 0`，说明组合比单 path 更差。

### 8.4 Phase 3 通过标准

进入 Phase 4 的组合必须满足：

```text
h10/h15 ATE delta <= -3m
或 h10/h15 [200,300) delta <= -5m
```

并且：

```text
[400,600) regression <= +1m
durability_ratio >= 0.35
pair interaction 不显著为正
```

如果 pairwise 全失败，不允许进入 all-memory；必须回到 Phase 2 或更新 role taxonomy。

---

## 9. Phase 4：Path-specific all-memory semantic role controller

### 9.1 目标

只有当 single-path / pairwise 有证据时，才构建真正 all-memory role controller。

v24 all-memory 不再用同一个 role 同时控制所有 path，而是：

```text
R_frame_i  = f_frame(G_sem, D, Q, U)
R_global_i = f_global(G_sem, D, Q, U)
R_swa_i    = f_swa(G_sem, D, Q, U)
R_ttt_i    = f_ttt(G_sem, D, C_ttt, Q, U)
```

### 9.2 建议公式

#### Frame/global source keep

$$
K^{frame}_i =
\operatorname{clip}
\left(
1 - \rho_D D_i - \rho_U U_i - \rho_S S^{risk}_i + \rho_P P^{protect}_i,
K_{min},
1
\right)
$$

其中：

```text
D_i: D_g risk
U_i: uncertainty
S_risk: semantic source risk
P_protect: structure / horizon protect score
```

#### SWA cache keep

$$
K^{swa}_i =
\operatorname{clip}
\left(
1 - \rho_D D_i - \rho_{sem} S^{cache-risk}_i + \rho_{ov} O_i,
K_{min},
1
\right)
$$

其中 $O_i$ 是 overlap continuity protect。

#### TTT positive/neutral/negative mass

$$
P^{ttt}_i =
\mathbf{1}[\text{structure}] \cdot (1-D_i) \cdot (1-C^{ttt}_i)
$$

$$
N^{ttt}_i =
\mathbf{1}[\text{semantic-risk}] \cdot D_i \cdot C^{ttt}_i
$$

$$
Z^{ttt}_i =
\mathbf{1}[\text{neutral}] \cdot (1-U_i)
$$

最终 TTT write 可以写为：

$$
G_{commit}
=
G_{pos}
+
\lambda_{neu}G_{neu}
-
\gamma_{neg}G_{neg}
$$

### 9.3 All-memory 候选

```text
ALLMEM_01_FRAME_TTT_PATHSPEC
ALLMEM_02_FRAME_SWA_TTT_PATHSPEC
ALLMEM_03_FRAME_GLOBAL_SWA_TTT_PATHSPEC
ALLMEM_04_SKY_NEUTRAL_STRUCTURE_LONG
ALLMEM_05_LOWSTUFF_HIGHD_SHORTNEG
ALLMEM_06_CONFLICT_GATED_SEMANTIC
```

### 9.4 Phase 4 通过标准

all-memory 候选必须满足：

```text
h10/h15 ATE delta <= -3m
或 h10/h15 [200,300) delta <= -5m
```

并且：

```text
durability_ratio >= 0.45
[400,600) regression <= +1m
semantic role coverage 非空
source/write mass 变化可解释
```

如果 all-memory 候选只 h10 强、h15 弱，则不启动 selector，进入 Phase 5 durability attribution。

---

## 10. Phase 5：Durability attribution

### 10.1 目标

如果出现 h10 有效但 h15 衰减，Phase 5 必须回答：

```text
修正是被哪个 memory path 覆盖掉的？
```

### 10.2 必须保存状态

对 base 与 candidate，在 h10 endpoint 和 h15 endpoint 保存：

```text
HMC fast-weight state
TTT post-zp delta summary
SWA cache state
global/chunk source state
merge/gauge state
semantic role mass summary
```

### 10.3 overwrite ratio

定义：

$$
OR_{ttt} =
\frac{
\|W_{h15}^{cand} - W_{h10}^{cand}\|
}{
\|W_{h10}^{cand} - W_{h10}^{base}\| + \epsilon
}
$$

类似定义：

```text
OR_swa
OR_global
OR_merge
```

如果某个 $OR$ 很高，说明该 path 在 h10->h15 期间覆盖了修正。

### 10.4 判断与 Codex fallback

如果 TTT overwrite 高：

```text
尝试 skip-aware TTT commit；
尝试 W_long / W_short；
尝试只保护 POSITIVE_LONG，不保护 NEGATIVE_SHORT。
```

如果 SWA overwrite 高：

```text
尝试 semantic SWA cache protect；
限制 highD lowstuff 进入 previous-source cache；
保护 structure / horizon overlap cache。
```

如果 global overwrite 高：

```text
尝试 global source compact；
延长 frame/global source role 到 h15；
保护 special tokens。
```

如果 merge/gauge overwrite 高：

```text
不要继续 memory threshold sweep；
转 online trajectory-state / scale-state module。
```

### 10.5 Phase 5 输出

必须输出：

```text
durability_attribution.csv
overwrite_ratio_by_path.csv
h10_to_h15_state_diff_heatmap.png
semantic_role_mass_h10_h15.png
trajectory_h10_h15_error_curve.png
```

---

## 11. Phase 6：No-GT selector

只有 Phase 4/5 出现 durable candidate 后，才启动 selector。

### 11.1 Selector 输入

```text
semantic role metrics
D_g metrics
TTT conflict metrics
source keep ratios
SWA cache mass
TTT role mass
no-GT scale proxy
short rollout proxy metrics
```

### 11.2 Selector gate

selector 必须满足：

```text
Spearman(proxy_score, h15_ATE_delta) <= -0.45
Top-3 recall of true improving candidates >= 0.67
selected candidate does not violate [400,600) regression <= +1m
```

如果 selector 不过 gate，不跑 full online。

---

## 12. Phase 7：Full online validation

只有 selector gate 通过，才跑 full KITTI01。

### 12.1 Full online success criteria

阶段成功：

```text
ATE <= 32.5m
且 [200,300) <= 70m
且 [400,600) 不高于 H9 + 1m
```

强成功：

```text
ATE <= 30m
```

最终目标：

```text
ATE <= 25m
```

如果 full online 只改善 Rot / FinalErr，而 ATE 不降，不算主线成功。

---

## 13. 加速与并行执行方案

### 13.1 并行 track

Codex 可以并行推进以下 track：

```text
Track A: implementation audit + taxonomy fine label
Track B: passive semantic attribution
Track C: frame/global source policy
Track D: SWA cache policy
Track E: TTT semantic write policy
Track F: visualization/report aggregation
```

### 13.2 推荐调度

```text
Phase 0:
    1 GPU smoke + static audit

Phase 1:
    3 chunks x 2 horizons debug-only，可 6 并发

Phase 2:
    每条 path 先 chunk10 h10/h15
    过初筛后补 chunk6/chunk16
    最多 6 并发 short-rollout

Phase 3:
    pairwise 只跑 top-2 path combinations
    不超过 12 rows

Phase 4:
    all-memory 只跑 4-6 个候选
    不做 36-row 粗矩阵

Full online:
    最多 2-4 并发
    不使用 8 并发，避免 host RAM / IO 风险
```

### 13.3 不满足条件时的自动尝试方向

#### 情况 1：semantic group 太粗

症状：

```text
LOW_VALUE_STUFF 覆盖过大；
sky / vegetation / grass 混在一起；
role 无法解释效果。
```

Codex 应尝试：

```text
拆 fine labels；
至少拆 sky / vegetation / road / building / wall_fence / movable / unknown；
重跑 Phase 1 attribution，不跑 full。
```

#### 情况 2：single-path 全失败

症状：

```text
source/write mass 变化存在，但 h10/h15 没有改善。
```

Codex 应尝试：

```text
不要继续阈值微扫；
改成 conditional role:
    semantic AND highD
    semantic AND update_conflict
    semantic AND uncertainty
```

#### 情况 3：h10 强，h15 弱

Codex 应尝试：

```text
durability attribution；
找 overwrite path；
若 TTT overwrite 高，试 W_long/W_short；
若 SWA overwrite 高，试 semantic cache protect；
若 global overwrite 高，试 global source persistence；
若 merge overwrite 高，转 trajectory-state module。
```

#### 情况 4：all-memory 比 single-path 差

Codex 应尝试：

```text
构建 path cancellation matrix；
回到 pairwise 组合；
不要继续 all-memory 阈值微扫。
```

#### 情况 5：implementation audit 反复失败

Codex 应尝试：

```text
缩到 END_FRAME=128 smoke；
只开 one path；
确认 run_config hash；
确认 role_counts 非空；
确认 summary jsonl 写入；
确认 invalid dirs 被排除；
然后再恢复 short-rollout。
```

---

## 14. v24 停止规则

如果下面条件同时成立：

```text
1. Phase 2 所有 single-path 候选不过 gate；
2. Phase 3 pairwise 组合不过 gate；
3. Phase 4 all-memory 不产生 h15 durable signal；
4. Phase 5 attribution 显示修正主要被 merge/gauge 或 trajectory state 覆盖；
```

则结论应更新为：

> Semantic Prior Generator 可以作为 memory regularizer / diagnostic，但不足以作为 Target-25 主线。Target-25 主线应转向 explicit online trajectory-state / scale-state module，TTT/SWA/semantic memory 作为辅助 regularizer。

如果 Phase 2 或 Phase 4 出现 durable candidate，则进入 selector/full online，不再重复旧的 coarse role matrix。

---

## 15. 最终判断

v24 的重点不是再跑一个更大的 semantic all-memory 矩阵，而是：

```text
先确认语义 role 的 token 对齐和 path consumption；
再按 memory path 分职责测试；
再做 pairwise 组合；
最后才做 path-specific all-memory role controller。
```

核心科学问题是：

> 语义到底在每条 memory path 中应该扮演什么角色？

而不是：

> sky / vegetation / road 要不要统一写多一点或少一点？

一句话总结：

> v23 已经证明 Semantic Prior Generator 可以接入所有 memory；v24 要证明语义是否能成为 path-specific memory role controller。如果仍然失败，就应把语义降级为 regularizer / diagnostic，把 Target-25 主线转向显式 trajectory-state correction。
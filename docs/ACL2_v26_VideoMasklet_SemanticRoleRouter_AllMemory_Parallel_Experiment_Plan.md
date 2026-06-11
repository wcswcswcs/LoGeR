
# ACL2 v26 实验计划：VideoMasklet Semantic Prior 的 Path-Specific All-Memory Role Router

日期：2026-05-22  
目标系统：LoGeR / HMC Pipeline v2 / Video Masklet Front-end / Semantic Prior Generator  
主开发集：KITTI Odometry 01  
目标指标：KITTI01 deployable online ATE $\le 25m$  
当前 deployable online best：`C9_P0_R2, ATE = 33.7629421029m`  
本轮核心优先级：**优先探索语义对所有 memory path 的写入 / 读取 / 缓存策略，即把 Semantic Prior Generator 从粗粒度 prior score 升级成 path-specific semantic memory role router。**

---

## 0. 当前判断：为什么要重启 v26

v25B 已经排除了几个重要工程疑点：

```text
1. Video-masklet cache hit rate = 1.0。
2. chunks_with_masklets_ratio = 1.0。
3. mean_coverage ≈ 0.9817。
4. focus_coverage_200_300 ≈ 0.9782。
5. no-op / pass-through / debug-only parity 通过。
6. frame / global / SWA / TTT 四条 memory path 都有 consumption evidence。
7. context empty source event runs = 0。
```

但 v25B 的性能信号很弱：

```text
best h10 ATE delta = -0.3175m
best h10 [200,300) delta = -1.9404m
best h15 ATE delta = -0.0736m
best h15 [200,300) delta = -0.1255m

No Phase 2 candidate passed gate.
No pairwise combination.
No all-memory validation.
No selector.
No full online validation.
No online Target-25 result.
```

这说明失败不再能简单归因于：

```text
Stage C cache 没命中；
masklet coverage 不够；
semantic prior 没接上；
no-op 路径污染；
frame/global/SWA/TTT 没有消费语义。
```

更可能的问题是：

```text
1. runtime semantic policy 仍然太粗，只是 coarse-group keyed；
2. fine labels 虽然在 cache audit 里存在，但没有真正成为 runtime policy；
3. 语义没有和 D_g / TTT conflict / scale-state risk 绑定；
4. 同一套语义规则直接作用到 frame/global/SWA/TTT，容易互相抵消；
5. 当前语义 role 只是弱 regularizer，不是 trajectory-drift causal controller。
```

因此 v26 的核心问题不是“再扫几个语义阈值”，而是重新定义 Semantic Prior Generator 的职责：

> Semantic Prior Generator 不应该只输出一个 `A_tok` 或一个 coarse semantic role；它应该输出 path-specific memory roles：当前 token 在 frame/global attention、SWA local memory、TTT fast-weight memory 中分别应当作为 source、anchor、neutral context、short negative evidence，还是 ignore region。

---

## 1. 本轮整体目标

本轮不是为了在 KITTI01 上打榜，也不是为了对某个数据集调参。  
本轮的目标是建立一个可解释、可迁移的语义 memory role 机制。

### 1.1 科学目标

本轮要回答五个问题。

**问题 A：fine semantic label 是否比 coarse group 有真实价值？**

v25B 的 runtime 语义策略仍是 coarse-group keyed。虽然 cache 中有：

```text
building
fence
grass
road
sidewalk
sky
vegetation
wall
```

但当前 runtime policy 不能真正区分这些 fine labels。本轮必须让 fine label 真正进入 runtime role policy，再判断语义失败到底是因为 coarse grouping 太粗，还是语义本身不是主因果变量。

**问题 B：语义应当如何在不同 memory path 中扮演不同角色？**

LoGeR 的 memory path 不是同一种东西：

```text
frame/global attention:
    控制当前 chunk 从哪些 source token 读取上下文。

SWA:
    控制相邻 chunk 的 local K/V source 和 overlap continuity。

TTT:
    控制长期 compressed fast-weight memory。
```

同一个区域在不同 path 中不应有同一动作。例如：

```text
sky:
    frame/global 中可能要弱 source；
    SWA 中可能要保留 horizon continuity；
    TTT 中一般不应 positive long write；
    但也不应默认 negative write。

road/building:
    frame/global 中可作为稳定 source；
    SWA 中可作为 overlap anchor；
    TTT 中在 low D_g + low conflict 条件下可作为 positive long memory。
```

**问题 C：语义是否必须条件化于 D_g、TTT conflict 和 scale-state risk？**

当前目标错误更像：

```text
reset-window scale drift
trajectory-state drift
h10 correction 到 h15 被 wash out
```

这不是单独的 semantic label 能决定的。因此本轮必须验证：

```text
semantic label + D_g
semantic label + update_conflict_energy
semantic label + scale-state risk
semantic label + mask trust
```

是否比单独 semantic role 更强。

**问题 D：语义是否能把短期 source correction 变成 durable memory correction？**

历史最强的短期信号来自 source skip + scale commit，而不是纯语义。v20 / v22 都显示：

```text
h10 可以压低 [200,300)
h15 会衰减
```

v26 要检验语义是否能解释并保护这类短期 correction，让它在 h15 仍有效。

**问题 E：不同数据集的问题是否不同，但不做数据集调参？**

本轮可以诊断不同数据集 / 序列上的 failure mode，例如：

```text
KITTI01:
    可能是 reset-window scale drift。

KITTI00 / 02 / 05:
    可能有不同语义分布和不同运动结构。

其他 city / indoor 数据:
    sky / road / building / wall / floor 比例不同。
```

但所有策略必须固定，不允许：

```text
为 KITTI01 单独调 threshold；
为某个 sequence 单独换 label value；
为某个数据集单独换 gamma；
为某个失败 segment 手工写 rule。
```

数据集差异只用于诊断，不用于打榜调参。

---

## 2. 总体原则

### 2.1 不再用 coarse group 直接代表 runtime fine policy

coarse group 仍可用于 summary，但 runtime policy 必须能访问 fine label：

```text
L_fine_tok:
    sky / vegetation / grass / road / sidewalk / building / fence / wall / movable / unknown ...

G_coarse_tok:
    STRUCTURE_ANCHOR / LOW_VALUE_STUFF / MOVABLE_THING / STATIC_THING / UNCERTAIN_REGION
```

如果只实现了 coarse group，则不能声称验证了：

```text
skip sky
skip vegetation
keep road
keep building
```

这类 fine semantic hypothesis。

### 2.2 Semantic Prior Generator 输出四条 path-specific role stream

本轮要求 Stage D 输出：

```text
R_frame_tok
R_global_tok
R_swa_tok
R_ttt_tok
```

而不是一个共享的 `R_sem_tok`。

四条 path 的语义角色可以不同：

```text
R_frame:
    controls frame-attention K/V source role

R_global:
    controls global/chunk-attention K/V source role

R_swa:
    controls SWA previous/current source cache role

R_ttt:
    controls TTT positive / neutral / negative write role
```

### 2.3 语义只提供类别先验，最终角色必须由条件组合决定

本轮不允许类似下面这种单因子 hard rule：

```text
sky -> skip
vegetation -> negative
road -> positive
building -> positive
```

正确形式应当是条件式：

```text
sky + low D_g:
    neutral keep

sky + high D_g + high uncertainty:
    weak source skip, not TTT negative

vegetation + high D_g + high conflict:
    source skip or short negative

road/building + low D_g + low conflict:
    positive long memory

road/building + high conflict:
    keep as read source may be ok, but avoid long TTT write
```

### 2.4 先 short rollout，再 full online

所有 candidate 必须先通过 trusted causal fork short rollout：

```text
chunks = 6, 10, 16
horizons = h10, h15
```

只有通过 gate 的 candidate 才允许进入：

```text
pairwise combination
all-memory validation
no-GT selector
full online validation
```

### 2.5 不对数据集调参

跨数据集时固定：

```text
semantic taxonomy
D_g threshold / quantile
conflict threshold / quantile
scale-risk threshold
memory path role table
write strength
skip strength
selector rule
```

如果某个数据集失败，只记录：

```text
semantic distribution
failure segment
memory path contribution
drift type
```

不为它单独调参。

---

## 3. 语义角色定义

本轮建议把每个 token 的 role 定义为六种。

```text
KEEP_SOURCE:
    可以作为 K/V source。

WEAK_SOURCE:
    可以作为 source，但降低权重。

SKIP_SOURCE:
    不作为 K/V source，但 query token 保留。

POSITIVE_LONG:
    可以写入长期 TTT memory。

NEUTRAL_CONTEXT:
    对 horizon / scale / context 有用，但不应强写长期 memory。

NEGATIVE_SHORT:
    只作为短期负证据或短期 correction，不进入长期 positive memory。

IGNORE:
    低信任 / 未知 / 无效区域，不由 semantic 分支控制，fallback 到 geometry。
```

### 3.1 Fine label 初始角色先验

这不是最终规则，只是 role prior。最终角色还要结合 $D_g$、conflict、scale risk、mask trust。

| Fine label | Frame/global role prior | SWA role prior | TTT role prior |
|---|---|---|---|
| road | KEEP_SOURCE if lowD | overlap anchor | POSITIVE_LONG if lowD + low conflict |
| sidewalk | WEAK/KEEP_SOURCE | local continuity | NEUTRAL or POSITIVE if lowD |
| building | KEEP_SOURCE | overlap anchor | POSITIVE_LONG if lowD + low conflict |
| wall/fence | KEEP_SOURCE | local anchor | POSITIVE_LONG or NEUTRAL |
| sky | WEAK_SOURCE or NEUTRAL | partial keep for horizon | NEUTRAL_CONTEXT, not POSITIVE_LONG |
| vegetation/grass | WEAK_SOURCE; skip only if highD | weak local context | NEUTRAL or NEGATIVE_SHORT if highD + high conflict |
| movable thing | SKIP_SOURCE if highD | drop from previous cache if highD | NEGATIVE_SHORT or IGNORE |
| unknown/ignore | fallback to geometry | fallback to geometry | fallback to geometry |

### 3.2 Conditional role formula

对每个 token $i$，定义：

```text
L_i = fine semantic label
D_i = D_g token score
C_i = TTT update_conflict_energy
S_i = scale-state risk
Q_i = masklet trust
```

语义角色不是函数 $R_i = f(L_i)$，而应是：

$$
R_i^{path} = f_{path}(L_i, D_i, C_i, S_i, Q_i)
$$

其中 $path \in \{frame, global, swa, ttt\}$。

例如 TTT positive long role：

$$
R_{ttt,i} = POSITIVE\_LONG
$$

当且仅当：

$$
L_i \in STRUCTURE
$$

且：

$$
D_i < q_D^{low}, \quad C_i < q_C^{low}, \quad Q_i > q_Q^{min}
$$

TTT negative short role：

$$
R_{ttt,i} = NEGATIVE\_SHORT
$$

当且仅当：

$$
D_i > q_D^{high}
$$

且：

$$
C_i > q_C^{high}
$$

且语义属于：

$$
L_i \in \{vegetation, grass, movable, uncertain\}
$$

sky 默认不进入 negative short，除非：

$$
D_i > q_D^{high}, \quad C_i > q_C^{high}, \quad S_i > q_S^{high}
$$

---

## 4. 核心假设与实验设计

---

# H0：fine-label runtime role 接线必须先通过 hard gate

## 假设

v25B 的弱结果可能部分来自 coarse runtime policy。若 fine label 没有进入 runtime policy，后续所有“sky / vegetation / road / building”实验都不合法。

## 实验设计

Codex 先实现或自查：

```text
1. VideoMasklet cache 中的 fine labels 被读入。
2. 每个 patch token 有 L_fine_tok。
3. 每个 patch token 有 G_coarse_tok。
4. Stage D 输出 R_frame_tok / R_global_tok / R_swa_tok / R_ttt_tok。
5. HMC / model path 分别消费四条 role stream。
6. no-op / pass-through / debug-only 不改变 H9 trajectory。
```

需要跑：

```text
P0_00_H9_REFERENCE
P0_01_FINE_LABEL_LOADED_BUT_IGNORED
P0_02_FINE_ROLE_PASS_THROUGH_CONSUMED
P0_03_FINE_ROLE_DEBUG_ONLY_ALL_PATHS
P0_04_FRAME_SOURCE_SMOKE
P0_05_SWA_CACHE_SMOKE
P0_06_TTT_ROLE_SMOKE
```

## 必须记录

```text
implementation_audit/codex_self_check_report.md
implementation_audit/codex_self_check_summary.json
implementation_audit/codex_self_check_failures.jsonl

fine_label_token_projection.csv
fine_label_coverage_by_chunk.csv
fine_label_coverage_by_path.csv
path_consumption_summary.jsonl
semantic_role_summary.jsonl
semantic_memory_path_summary.jsonl
noop_parity_metrics.csv
context_skip_summary.jsonl
swa_semantic_cache_summary.jsonl
ttt_semantic_write_summary.jsonl
```

## 通过标准

H0 必须全部通过：

```text
cache_hit_rate >= 0.98
chunks_with_masklets_ratio >= 0.95
fine_label_count >= 6
runtime_fine_role_policy_available = true
R_frame/R_global/R_swa/R_ttt all non-empty
path_consumption_flags all true
no-op ATE delta = 0
raw_trans_max_diff = 0
context_empty_source_events = 0
no stale run directory contamination
```

## 不满足时 Codex 自动尝试

如果 `runtime_fine_role_policy_available=false`：

```text
Codex 必须先实现 L_fine_tok -> role stream projection。
不要继续跑 Phase 1/2。
```

如果 no-op parity 失败：

```text
Codex 比较 run_config / HMC config / prior score mean / HMC state hash。
优先排查 pass-through prior 是否改变 stage_d_x_* base prior。
```

如果 path consumption 为空：

```text
Codex 检查 run_pipeline_abc_v2.py CLI 透传；
检查 hybrid_memory_controller.py 是否构造 PriorOutput；
检查 pi3.py 是否真正接收 model hmc_control；
检查 summary JSONL 是否写入。
```

---

# H1：fine label semantic attribution 能解释为什么 coarse group 弱

## 假设

coarse group 把不同语义混在一起，使 memory action 被稀释。例如 `LOW_VALUE_STUFF` 可能同时包含 sky、grass、vegetation、sidewalk；它们对 trajectory / scale 的作用不同。

## 实验设计

不先改模型性能，先做 passive attribution。固定 H9 parent 和 v20/v22/v25B 历史强弱候选，统计：

```text
K1_H9
C9
v20 TTTSS_03B scale+skip
v22 TTT_LIFE_04
v25B best h10/h15 semantic rows
```

将 fine label 映射到：

```text
D_g
update_conflict_energy
scale-state risk
source skip mass
SWA cache mass
TTT positive/neutral/negative mass
h10/h15 ATE delta
h10/h15 [200,300) delta
```

## 必须记录

```text
passive_attribution/per_label_memory_mass.csv
passive_attribution/per_label_dg_conflict_scale.csv
passive_attribution/per_label_path_action.csv
passive_attribution/per_label_segment_error_corr.csv
passive_attribution/per_chunk_label_distribution.csv
passive_attribution/label_condition_correlation_summary.csv
```

## 必须可视化

```text
RGB + fine label overlay
D_g overlay
update_conflict_energy overlay
scale-state risk overlay
per-label D_g distribution
per-label conflict distribution
per-label source keep / skip bar chart
per-label TTT role mass bar chart
per-label h10 vs h15 waterfall
```

## 通过标准

H1 通过不是以 ATE 为准，而是以解释力为准。满足任一即可：

```text
1. 某些 fine label condition 与 h10/h15 delta 有明确相关性；
2. coarse group 内部 labels 的 D_g/conflict/scale-risk 分布显著不同；
3. v20/v22 强 h10 correction 主要集中在少数 fine label + risk condition 上；
4. 可以解释 v25B coarse semantic 为何弱。
```

## 不满足时 Codex 自动尝试

如果 fine label 与任何风险/误差都无关：

```text
Codex 将语义降级为 auxiliary diagnostic；
下一轮不要继续 semantic matrix，转到 D_g / conflict / scale-state 主导。
```

如果 coarse group 内差异明显但 runtime 还无法细分：

```text
Codex 优先修 runtime fine label policy。
```

---

# H2：frame/global 语义 source role 必须是 fine-label + D_g 条件式，而不是 coarse hard skip

## 假设

frame/global attention 的语义作用主要在 source K/V 选择。语义可以帮助决定哪些 source token 不该被读，但不能直接删除 query，也不能全层 hard skip。

## 实验设计

固定：

```text
parent = H9 causal fork
chunks = 6,10,16
horizon = h10
mode = readonly / read-path source control
query = kept
source K/V = controlled
```

候选：

```text
FG_FINE_01_STRUCTURE_KEEP:
    road/building/wall/fence lowD -> KEEP_SOURCE
    lowstuff highD -> WEAK_SOURCE
    sky lowD -> WEAK_SOURCE
    movable highD -> SKIP_SOURCE

FG_FINE_02_LOWSTUFF_HIGHD_SKIP:
    sky/vegetation/grass highD -> SKIP_SOURCE
    sky/vegetation/grass lowD -> WEAK_SOURCE
    structure lowD -> KEEP_SOURCE

FG_FINE_03_SKY_NEUTRAL:
    sky always WEAK_SOURCE, never SKIP_SOURCE
    vegetation highD can SKIP_SOURCE

FG_FINE_04_STRUCTURE_RESCUE:
    structure positive source protected even if D_g moderate
    but not if conflict very high

FG_FINE_05_CONFLICT_CONDITIONED:
    source skip only if semantic_risk AND D_g high AND conflict high
```

## 必须记录

```text
frame_source_keep_ratio_by_label
global_source_keep_ratio_by_label
num_context_source_skip_applied
max_context_source_skip_tokens
num_context_empty_source_events
attention_mass_to_highD_before_after
attention_mass_to_label_before_after
h10/h15 trajectory metrics
```

## 通过标准

进入 h15 top confirmation：

```text
h10 [200,300) delta <= -3m
or h10 ATE delta <= -1.5m
and [400,600) regression <= +1m
and empty_source_events = 0
```

进入 pairwise：

```text
h15 ATE delta <= -1.5m
or h15 [200,300) delta <= -2.5m
and durability_ratio >= 0.25
```

## 不满足时 Codex 自动尝试

如果 source keep ratio 变化很小：

```text
Codex 检查 threshold 是否没有实际选中 token；
输出 keep/drop map；
不要直接加大 hard skip，先修 role selection。
```

如果 h10 有效但 h15 失效：

```text
Codex 转 Phase 5 durability attribution；
检查后续 TTT/SWA/global source 是否 wash out correction。
```

如果 sky skip 使 ATE 变差：

```text
sky 改为 NEUTRAL_CONTEXT；
只允许 sky highD + high conflict + high scale-risk 做 weak skip。
```

---

# H3：SWA 的语义策略应服务 overlap/local continuity，而不是长期写入价值

## 假设

SWA 是 local memory。语义在 SWA 中的主要作用是维护相邻 chunk 的 overlap continuity，不是决定长期 global memory。

## 实验设计

候选：

```text
SWA_FINE_01_OVERLAP_STRUCTURE_KEEP:
    road/building/wall/fence in overlap -> keep
    movable highD in previous source -> drop

SWA_FINE_02_SKY_PARTIAL_KEEP:
    sky lowD in overlap -> keep partially
    sky highD -> weak source

SWA_FINE_03_VEGETATION_CONDITIONAL:
    vegetation lowD -> weak keep
    vegetation highD + conflict high -> drop

SWA_FINE_04_BOUNDARY_PROTECT:
    road/sidewalk/building boundary -> keep if low conflict
    high conflict boundary -> fallback to D_g

SWA_FINE_05_CACHE_LIFECYCLE:
    semantic-risk highD source enters W_short-like SWA cache for only one handoff
```

## 必须记录

```text
swa_cache_keep_mass_by_label
swa_previous_source_mass_by_label
swa_current_source_mass_by_label
swa_overlap_keep_ratio
swa_cache_lifetime_by_label
boundary_10f_ATE
boundary_20f_ATE
overlap_pointmap_residual
chunk_boundary_pose_jump
h10/h15 segment metrics
```

## 通过标准

```text
h10 [200,300) delta <= -3m
or boundary_10f_ATE improves by >= 1m
```

但进入 pairwise 还要求：

```text
h15 ATE delta <= -1m
or h15 [200,300) delta <= -2m
and [400,600) regression <= +1m
```

## 不满足时 Codex 自动尝试

如果 boundary metrics 改善但 ATE 不变：

```text
保留为 SWA auxiliary，不进入 all-memory 主线。
```

如果 h10 改善但 h15 回退：

```text
尝试 semantic cache lifecycle；
区分 previous-source keep 与 current-source keep。
```

如果 overlap structure keep 也弱：

```text
停止 SWA semantic matrix；
转向 frame/global source 或 TTT write。
```

---

# H4：TTT 语义写入必须绑定 TTT-native conflict 与 scale-state risk

## 假设

语义单独决定 TTT write 太弱。TTT 的语义 role 必须和 TTT-native cue 结合。

## 实验设计

固定：

```text
write_score_base = stage_d_x_dg_inv_sqrt
risk_base = update_conflict_energy
parent = H9
chunks = 6,10,16
h10 screen first
```

候选：

```text
TTT_FINE_01_STRUCTURE_POSITIVE:
    positive_long = structure lowD lowConflict
    neutral = lowstuff lowD
    negative_short = highD highConflict non-structure

TTT_FINE_02_SKY_NEUTRAL:
    sky always neutral, never positive/negative
    vegetation highD highConflict -> negative_short

TTT_FINE_03_SCALE_CONDITIONED:
    negative_short only if highD + highConflict + highScaleRisk
    positive_long only if structure + lowD + lowScaleRisk

TTT_FINE_04_LOWSTUFF_HIGHD_SHORT:
    lowstuff highD -> short negative
    lowstuff lowD -> neutral
    structure lowD -> positive

TTT_FINE_05_STRUCTURE_PROTECT:
    structure tokens protected from negative replay unless conflict is extreme
```

## 公式

基础三路 replay：

$$
G_{commit} = G_{pos} + \lambda_{neu}G_{neu} - \gamma G_{neg}
$$

其中：

$$
G_{pos} = \sum_{i \in \mathcal{P}} J_i
$$

$$
G_{neu} = \sum_{i \in \mathcal{N}} J_i
$$

$$
G_{neg} = \sum_{i \in \mathcal{R}} J_i
$$

候选集合由 fine semantic 与风险条件共同决定：

$$
\mathcal{P} = \{i: L_i \in STRUCTURE,\ D_i < q_D^{low},\ C_i < q_C^{low}\}
$$

$$
\mathcal{R} = \{i: D_i > q_D^{high},\ C_i > q_C^{high},\ S_i > q_S^{high},\ L_i \notin PROTECTED\}
$$

## 必须记录

```text
ttt_positive_mass_by_label
ttt_neutral_mass_by_label
ttt_negative_mass_by_label
ttt_update_norm_by_label
ttt_update_conflict_by_label
post_zp_delta_norm_by_label
branch0_update_norm_by_label
memory_state_rel_diff
h10/h15 trajectory metrics
```

## 通过标准

进入 h15：

```text
h10 ATE delta <= -1.5m
or h10 [200,300) delta <= -3m
```

进入 pairwise/all-memory：

```text
h15 ATE delta <= -1.5m
or h15 [200,300) delta <= -2.5m
and [400,600) regression <= +1m
```

## 不满足时 Codex 自动尝试

如果 semantic TTT role mass 很低：

```text
Codex 检查 token projection / role mapping；
输出 per-label TTT mass heatmap。
```

如果 role mass 高但 trajectory 无变化：

```text
说明 TTT semantic write 不是主因；
转为 frame/global/SWA source control。
```

如果 role mass 高且 Rot/FinalErr 改善但 ATE 不动：

```text
标记为 regularizer；
不要继续扩大 TTT semantic scalar。
```

---

# H5：只有 single-path 有强信号，才允许 pairwise / all-memory

## 假设

v23/v24/v25B 的 all-memory 弱结果可能来自路径之间互相抵消。必须先找到 single-path 强信号，再组合。

## 实验设计

只允许组合通过 gate 或 near-pass 的路径：

```text
FRAME + TTT
FRAME + SWA
GLOBAL + TTT
SWA + TTT
FRAME + GLOBAL
FRAME + GLOBAL + SWA + TTT
```

组合原则：

```text
不同 path 使用不同 R_path；
不共享同一个 R_sem_tok；
sky / vegetation 不允许在所有 path 同时 hard skip；
TTT negative_short 不自动等于 frame/global skip。
```

## 必须记录

```text
single_path_delta.csv
pairwise_delta.csv
interaction_gain = pair_delta - best(single_delta)
path_conflict_table.csv
role_mass_by_path_and_label.csv
```

## 通过标准

pairwise 通过：

```text
interaction_gain <= -0.5m
or h15 ATE delta <= -2m
or h15 [200,300) delta <= -3m
and [400,600) regression <= +1m
```

all-memory 通过：

```text
h10 [200,300) delta <= -5m
or h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m

and durability_ratio >= 0.45
and [400,600) regression <= +1m
```

## 不满足时 Codex 自动尝试

如果 pairwise 比 single-path 差：

```text
Codex 生成 path_conflict_table；
找出哪个 label/path 的 role 冲突；
不要直接继续 all-memory。
```

如果 all-memory h10 强 h15 弱：

```text
转 H6 durability attribution。
```

---

# H6：h10 强但 h15 弱时，必须定位 correction 被哪条 memory path 洗掉

## 假设

如果语义或 source-skip 在 h10 有效但 h15 弱，说明 correction 被后续 memory 更新覆盖。v22 state attribution 已显示 h10 到 h15 的 HMC/TTT state movement 很大，可能会 wash out local correction。

## 实验设计

对 h10 强、h15 弱候选保存状态：

```text
base state
candidate h10 endpoint
candidate h15 endpoint
TTT state
SWA cache state
global/chunk source summary
merge/gauge state
```

比较：

$$
overwrite\_ratio = \frac{\|S_{h15} - S_{h10}\|}{\|S_{h10} - S_{base}\| + \epsilon}
$$

按 path 拆：

```text
TTT overwrite ratio
SWA overwrite ratio
global source overwrite ratio
merge/gauge overwrite ratio
```

## 必须记录

```text
state_attribution/h10_h15_state_norms.csv
state_attribution/path_overwrite_ratio.csv
state_attribution/label_role_mass_h10_h15.csv
state_attribution/memory_path_washout_summary.md
```

## 通过标准

H6 通过是诊断成功，不是性能成功。满足任一：

```text
1. 明确发现 TTT washout 是主因；
2. 明确发现 SWA cache refresh 是主因；
3. 明确发现 global/chunk source 后续覆盖是主因；
4. 明确发现 merge/gauge state movement 是主因。
```

## 不满足时 Codex 自动尝试

如果无法归因：

```text
Codex 增加更细的 state hash / norm：
    per layer TTT
    per branch TTT
    per label SWA source mass
    per frame global source mass
```

如果 TTT washout 是主因：

```text
尝试 skip-aware TTT commit / W_long-W_short lifecycle。
```

如果 SWA washout 是主因：

```text
尝试 semantic SWA cache lifecycle。
```

如果 merge/gauge 是主因：

```text
语义 memory route 可能不是 Target-25 主线；
转 trajectory-state module。
```

---

# H7：跨数据集只做诊断，不做调参

## 假设

同一 semantic role policy 在不同数据库上可能表现不同，这反映场景结构差异，不应变成 per-dataset tuning。

## 实验设计

只有当 KITTI01 short-rollout 出现 stable candidate 时，才跑跨序列 / 跨数据诊断。

固定 policy：

```text
best semantic role policy from v26
same thresholds
same role table
same memory paths
same write strength
```

诊断数据：

```text
KITTI00
KITTI02
KITTI05
optional KITTI-360 / city sequence if available
optional indoor sequence if available
```

不允许修改：

```text
D_g quantile
semantic role table
label weights
skip threshold
write gamma
memory path set
```

## 必须记录

```text
dataset_semantic_distribution.csv
dataset_failure_profile.csv
dataset_segment_error.csv
generalization_table.csv
fixed_policy_config.yaml
```

## 成立标准

跨数据诊断成功不是要求每个数据集都提升，而是要求：

```text
1. 固定 policy 的改善 / 回退方向可解释；
2. 不出现 >5% ATE regression 的灾难；
3. 能用 semantic distribution + drift profile 解释失败；
4. 不做任何 per-dataset tuning。
```

---

## 5. 实验阶段与并行计划

---

## Phase 0：Codex implementation hard gate

### 目标

先保证 fine-label path-specific semantic role 真的进入 runtime。Phase 0 不通过，不允许任何性能实验。

### 并行任务

```text
Codex-A:
    实现 / 检查 fine label token projection。

Codex-B:
    实现 / 检查 R_frame/R_global/R_swa/R_ttt 四路输出。

Codex-C:
    检查 HMC / model path consumption。

Codex-D:
    运行 no-op / pass-through / smoke。

Codex-E:
    生成 implementation audit report。
```

### 产物

```text
phase0_implementation_audit/
phase0_smoke_rollouts/
phase0_noop_report/
phase0_path_consumption_report/
```

### Gate

必须满足 H0 全部条件。

---

## Phase 1：Passive fine-label semantic attribution

### 目标

不跑大矩阵，先看语义是否解释已有强弱信号。

### 并行任务

```text
Codex-A:
    per-label coverage / D_g / conflict / scale-risk audit。

Codex-B:
    v20/v22/v25B strong/weak rows 的 label attribution。

Codex-C:
    per-memory-path semantic mass audit。

Codex-D:
    visual dashboard。
```

### 产物

```text
phase1_passive_attribution/
phase1_visual_dashboard/
phase1_label_condition_report.md
```

### Gate

若 H1 无解释力，停止大规模语义矩阵，把语义降级为辅助诊断。

---

## Phase 2：Single-path semantic role screen

### 目标

frame/global/SWA/TTT 分开验证。禁止一开始 all-memory。

### 并行 tracks

```text
Track A:
    frame/global fine semantic source role

Track B:
    SWA semantic cache / overlap role

Track C:
    TTT semantic positive/neutral/negative write role

Track D:
    semantic + D_g/conflict/scale-risk conditional role
```

每个 track 先跑：

```text
chunk10 h10
```

通过或 near-pass 后再跑：

```text
chunk6 h10/h15
chunk10 h15
chunk16 h10/h15
```

### Gate

进入 Phase 3 的条件：

```text
h10 [200,300) delta <= -3m
or h15 ATE delta <= -1.5m
or h15 [200,300) delta <= -2.5m

and [400,600) regression <= +1m
and path metrics prove actual source/write mass changed
```

如果没有任何 path 过 gate，不启动 Phase 3。

---

## Phase 3：Pairwise memory-role combination

### 目标

只组合 Phase 2 过 gate 或 near-pass 的 path，验证交互是否为正。

### 组合

```text
FRAME + TTT
FRAME + SWA
GLOBAL + TTT
SWA + TTT
FRAME + GLOBAL
```

### Gate

进入 Phase 4 的条件：

```text
interaction_gain <= -0.5m
or h15 ATE delta <= -2m
or h15 [200,300) delta <= -3m

and [400,600) regression <= +1m
```

---

## Phase 4：All-memory role controller

### 目标

最终验证 path-specific semantic role 是否能同时控制：

```text
frame attention
global attention
SWA
TTT
```

但每个 path 使用不同 role stream，不共享一个 coarse role。

### Gate

进入 Phase 5 的条件：

```text
h10 [200,300) delta <= -5m
or h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m

and durability_ratio >= 0.45
and [400,600) regression <= +1m
```

---

## Phase 5：Durability attribution

### 目标

当 h10 强但 h15 弱时，找出 washout source。

### 产物

```text
phase5_state_attribution/
phase5_washout_report.md
```

### 后续分流

```text
TTT washout:
    skip-aware TTT commit / W_long-W_short

SWA washout:
    semantic SWA cache lifecycle

global source washout:
    global source persistence / protected structure source

merge/gauge washout:
    target-25 主线转 explicit trajectory-state module
```

---

## Phase 6：Selector / full online validation

### 启动条件

只有 Phase 4 通过，或者 Phase 5 找到明确 durable correction action，才允许进入。

### Full online gate

```text
diagnostic improvement:
    ATE <= 32.5m
    or ATE improves over C9 by >= 1.5m

strong success:
    ATE <= 30m
    and no [400,600) regression > +2m

final target:
    ATE <= 25m
```

---

## Phase 7：Cross-dataset diagnosis

### 启动条件

至少一个 fixed policy 在 KITTI01 full 或 h15 gate 上表现稳定。

### 原则

```text
固定 policy
固定 thresholds
固定 role table
不为数据集调参
只做 failure profile
```

---

## 6. 必须记录的指标

### 6.1 Trajectory metrics

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
[200,300)
[200,400)
[400,600)
chunk6-10 error
chunk16 error
```

### 6.2 Semantic metrics

```text
per_label_coverage
per_label_visible_ratio
per_label_mask_quality
per_label_Dg_mean
per_label_Dg_p90
per_label_confidence_mean
per_label_conflict_mean
per_label_scale_risk_mean
per_label_unknown_ratio
```

### 6.3 Path metrics

```text
frame_source_keep_ratio_by_label
global_source_keep_ratio_by_label
swa_cache_keep_ratio_by_label
swa_previous_source_mass_by_label
swa_current_source_mass_by_label
ttt_positive_mass_by_label
ttt_neutral_mass_by_label
ttt_negative_mass_by_label
post_zp_update_norm_by_label
memory_state_rel_diff_by_path
```

### 6.4 Durability metrics

```text
h10_delta
h15_delta
durability_ratio = h15_delta / (h10_delta + eps)
overwrite_ratio_HMC
overwrite_ratio_TTT
overwrite_ratio_SWA
overwrite_ratio_global
overwrite_ratio_merge
```

### 6.5 Resource metrics

```text
wall_seconds
gpu_id
max_gpu_mem
host_ram_available_min
cache_hit_rate
probe_cache_mode
probe_cache_payload
run_status
failure_reason
```

---

## 7. 必须可视化

每个进入 h15 或 Phase 3 的 candidate 必须输出：

```text
1. RGB + fine label overlay
2. RGB + coarse group overlay
3. D_g heatmap
4. update_conflict_energy heatmap
5. scale-state risk heatmap
6. frame K/V source keep/drop map
7. global source keep/drop map
8. SWA cache keep/drop map
9. TTT positive/neutral/negative map
10. per-label path role map
11. per-label memory mass bar chart
12. h10 vs h15 segment error waterfall
13. trajectory overlay:
        H9 / C9 / candidate / GT
14. segment trajectory view:
        [200,300)
        [400,600)
15. state overwrite ratio plot
16. semantic role confusion table:
        fine label -> actual role by path
```

---

## 8. 加速与并行执行策略

### 8.1 调度原则

```text
short rollout:
    6 GPUs 并行，若 host RAM safe。

full KITTI01:
    最多 4 GPUs 并行。

cross-sequence full:
    最多 2 GPUs 并行。

Stage C:
    只读 offline video-masklet cache。
    不允许 inline compute 干扰 HMC benchmark。
```

### 8.2 Speed-gated execution

```text
1. Phase 0 不通过，不跑任何 candidate。
2. Phase 1 无解释力，不跑 Phase 2 full matrix。
3. Phase 2 h10 不过 gate，只跑少量 h15 top confirmation。
4. Phase 2 无 candidate，禁止 Phase 3。
5. Phase 3 无正交互，禁止 Phase 4。
6. Phase 4 无 durability，禁止 selector/full online。
```

### 8.3 Codex 并行 worker 设计

```text
Worker 1:
    implementation audit / no-op / path consumption

Worker 2:
    passive attribution / visualization

Worker 3:
    frame/global source role candidates

Worker 4:
    SWA semantic role candidates

Worker 5:
    TTT semantic role candidates

Worker 6:
    report aggregation / gate decision / failure routing
```

### 8.4 Stale run 防护

所有 launcher 必须：

```text
move stale run dir to .INVALID_RERUN_*
write run_config.yaml
write semantic_role_config.yaml
write path_consumption_summary.jsonl
write candidate_family
write diagnostic_only_short_rollout flag
write counts_as_online_ttt_write_success flag
```

---

## 9. 失败自动分流表

| 失败现象 | 解释 | Codex 下一步 |
|---|---|---|
| no-op 不对齐 | 语义接入污染 baseline | 停止实验，比较 HMC config / prior score / state hash |
| fine labels 有 cache 但 runtime 不可用 | 还在 coarse fallback | 实现 `L_fine_tok` runtime projection |
| path consumption 为空 | 参数没传到 model/HMC | 检查 CLI、HMC prior、Pi3 control |
| source keep ratio 基本不变 | role 没真正选中 token | 输出 keep/drop map，修 threshold/projection |
| h10 无强信号 | semantic 单独不是主因 | 转 D_g/conflict/scale-conditioned role |
| h10 强 h15 弱 | correction 被 wash out | 启动 Phase 5 durability attribution |
| `[200,300)` 改善但 `[400,600)` 崩 | continuity 被破坏 | 加 neutral protection / SWA cache lifecycle |
| sky skip 伤 ATE | sky 提供 horizon/scale | sky 改为 neutral/weak source，不做 negative |
| vegetation skip 伤 ATE | vegetation 也可能提供远景 continuity | 只在 highD + highConflict 时 skip |
| TTT semantic mass 高但无轨迹变化 | TTT semantic write 不是主杠杆 | 降级 TTT semantic，转 frame/global/SWA source |
| all-memory 比 single-path 差 | path role 互相抵消 | 输出 path_conflict_table，回退 pairwise |
| 跨数据集某类失败 | 场景分布不同 | 记录 failure profile，不调参 |

---

## 10. 最终决策规则

### 10.1 继续 semantic 主线的条件

满足以下任一：

```text
1. fine-label semantic 在 h10 或 h15 上明显强于 v25B coarse；
2. 发现某个 label-condition 对 memory path 有明确因果作用；
3. frame/global/SWA/TTT 至少一条 path 过 Phase 2 gate；
4. pairwise 或 all-memory 产生 durable improvement。
```

### 10.2 降级 semantic 的条件

如果满足：

```text
Phase 0 pass
Phase 1 attribution 无解释力
Phase 2 single-path 全部弱
Phase 3 不允许启动
```

则结论应写为：

> VideoMasklet semantic 在当前形式下不是 Target-25 主杠杆。语义保留为 diagnostic / weak auxiliary，主线回到 D_g / TTT conflict / scale-state / explicit trajectory-state。

### 10.3 进入 full online 的条件

必须满足：

```text
Phase 4 或 Phase 5 candidate:
    h15 ATE delta <= -3m
    or h15 [200,300) delta <= -5m
    or h10 strong + durability_ratio >= 0.45

and:
    [400,600) regression <= +1m
    path metrics show real role effect
    no source empty events
    no stale run contamination
```

---

## 11. 本轮预期结果与风险

### 11.1 最理想结果

```text
fine-label + conditional role 找到一个 path 或 pairwise candidate：
    h10 [200,300) <= -5m
    h15 ATE <= -3m
    durability_ratio >= 0.45

之后进入 full online，ATE 至少低于 32.5m。
```

### 11.2 中等结果

```text
fine-label 能解释 v25B coarse 为什么弱；
某些 labels 对 source/write 有明确贡献；
但没有 full candidate。
```

这仍然是科学进展，可以决定语义是否继续作为辅助。

### 11.3 负结果

```text
fine-label runtime role 接通；
passive attribution 也没有解释力；
single-path 全部弱。
```

则应停止继续扩大 semantic all-memory 矩阵。

---

## 12. 一句话总结

v26 的核心不是“再试一批 semantic 阈值”，而是：

> 用 fine-label VideoMasklet 语义把 Semantic Prior Generator 升级成 path-specific memory role router，并验证语义在 frame/global/SWA/TTT 中是否能和 $D_g$、TTT conflict、scale-state risk 一起形成 durable memory correction；如果不能，就把语义从 Target-25 主线降级为辅助诊断。

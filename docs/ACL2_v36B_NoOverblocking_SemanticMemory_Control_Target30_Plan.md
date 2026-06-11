# ACL2 v36B：Training-Free Semantic Memory Control 修正版计划

日期：2026-05-24  
目标：纠正 v36 的过度阻塞设计，让 Codex 可以继续有效探索，同时保留必要的工程自查。  
核心约束：training-free；不训练 trigger、selector、router；不使用 GT runtime action；不针对 KITTI01 或任何数据集调参；不把 diagnostic、short rollout、fixed chunk、oracle 结果写成 deployable success。

---

## 0. 为什么必须改 v36 计划

v36 的失败不是因为语义假设被否定，而是因为计划把一个 **解释性 instrumentation 指标** 写成了 **阻止所有后续实验的 hard gate**。

v36 已经证明了几件重要事情：

```text
frame/global synthetic source skip 能进入真实 source skip hook；
SWA synthetic remove 能进入 SWA source gate summary；
TTT synthetic negative 能产生真实 post-zp/action-delta tensor trace；
context empty source events = 0；
```

但因为没有落盘：

```text
attention_mass_removed_before
attention_mass_removed_after
```

原计划让 H0 fail，并禁止 H1-H5 启动。这在工程审计上很严格，但对当前科研目标不合理。我们真正需要的是先验证：

```text
语义风险区域的 source/write/cache action 是否会影响重建质量；
哪些 memory path 对语义最敏感；
哪些 action 是局部有效但不持久；
哪些 action 能向 Target-30 推进。
```

因此 v36B 将 H0 拆成两类：

```text
H0A: hook reachability gate，必须通过，决定能不能跑 short rollout。
H0B: attention-mass explainability gate，强烈建议实现，但不阻止 H1-H3。
```

也就是说，缺少 attention mass before/after 时，不能声称“被跳过的是高 attention source”；但只要 hook 确认真实生效，就允许进入 small short-rollout causality matrix。

---

## 1. 项目目标与边界

本项目不是训练语义模型，不是在 KITTI01 上拟合 trigger，也不是打榜。目标是：

> 在 training-free 条件下，把视频语义、LoGeR internal cue、SWA/TTT memory trace 结合成可解释的 memory control policy，提升 long-context reconstruction。

### 1.1 不能做的事

以下内容禁止进入主线：

```text
训练 trigger / classifier / selector / role router；
用 short-rollout oracle label 拟合规则；
使用 absolute chunk id；
为 KITTI01 单独调 semantic label value、threshold、gamma；
把 GT diagnostic / short rollout / fixed chunk / oracle 写成 deployable success；
在 action hook 未真实生效时解释 ATE 结果。
```

### 1.2 可以做的事

```text
使用 video masklet semantic 作为 noisy observation；
使用 SemanticKITTI sparse projection 做 trust calibration；
使用 deterministic cue conjunction 形成 semantic risk/action；
使用 no-commit paired probe 做无 GT self-consistency 检查；
使用 short rollout 作为 diagnostic；
固定规则跨序列诊断 failure mode，但不调数据集专属参数。
```

---

## 2. 当前事实与本轮新策略

当前 deployable online best 仍是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

Target-30 gap 是：

$$
33.7629421029 - 30.0 = 3.7629421029m
$$

v31/v34 说明 semantic-conditioned C23 有局部强信号，但不能 full transfer。v20-v22 说明 context source skip / compact K/V 有 h10 局部信号，但 h15 durability 弱。v27/v28/v30 说明语义 action 经常被压扁，SWA boundary 容易回退。v36 说明 synthetic source/TTT hooks 部分真实，但 attention mass explainability 还缺。

因此 v36B 不再扩大语义规则矩阵，而是并行验证四个实质问题：

```text
1. 语义风险区域是否真实影响 frame/global K/V source；
2. 语义风险区域在 SWA 中应如何保护 local continuity；
3. TTT 是否更适合使用 semantic static anchors，而不是 semantic scalar；
4. semantic-conditioned C23 的局部收益能否用 residual/read-only path 兼容 C9。
```

---

## 3. H0：分层工程自查，不再过度阻塞

### H0A：Hook Reachability Gate

这是继续实验的最低要求。必须满足：

```text
frame/global synthetic source skip:
    max_context_source_skip_tokens > 0
    context_empty_source_events = 0

SWA synthetic remove:
    num_source_gate_applied > 0
    source gate summary 非空

TTT synthetic negative:
    post_zp_action_delta_over_native_max > 0.05
    或 memory_ttt_mean_rel_diff > 0.005

no-op/pass-through:
    pose max diff = 0 或在既定 no-op tolerance 内
```

如果 H0A 不通过，不允许进入 rollout；Codex 必须先修 hook、参数透传或 role projection。

### H0B：Attention Mass Explainability Gate

这是解释性 gate，不阻止 H1-H3。Codex 应并行实现：

```text
attention_mass_removed_before
attention_mass_removed_after
attention_mass_retained_before
attention_mass_to_structure_before/after
attention_mass_to_dynamic_before/after
attention_mass_to_semantic_group_before/after
```

如果 H0B 不通过，则所有 H1/H2 的 source skip 结果必须标记为：

```text
source-count/action-summary supported
attention-mass causality not yet proven
```

但不再阻止 short rollout。

### H0C：Action Distinguishability Gate

对每个 semantic policy 记录：

```text
frame_source_keep_mask
global_source_keep_mask
swa_k_keep_mask
swa_v_keep_mask
swa_cache_keep_mask
ttt_positive_mask
ttt_neutral_mask
ttt_short_negative_mask
ttt_no_long_write_mask
```

通过标准：

```text
Jaccard(policy_A, policy_B) <= 0.95
或 source_keep_ratio 差 >= 0.02
或 TTT role mass 差 >= 0.02
```

若不同 policy 实际 action 等价，则 Codex 不得跑 rollout，必须检查 role projection、protected token、fallback、semantic threshold collapse。

---

## 4. H1：VGGT4D-style frame/global source skip

### 4.1 假设

动态物体、植被、阴影、低信任区域会污染 frame/global attention 的 K/V source。它们不一定应该被删除 query，但不应该作为 early/read-stage context source。这个假设与 VGGT4D 的 training-free source suppression 思路一致，但在 LoGeR 中要分 path 验证。

### 4.2 实验设计

固定：

```text
parent = H9 and C9 short-rollout snapshots
chunks = 6,10,16
horizon = h10 first, then h15 if gate passes
read cue = C23 past / current locked protocol
commit = probe_ttt_write
```

候选：

```text
FG_SKIP_01_DYNAMIC_HIGHD:
    movable/dynamic thing AND high D_g -> source skip

FG_SKIP_02_VEGETATION_HIGHD:
    vegetation/tree/grass AND high D_g -> source skip

FG_SKIP_03_LOWSTUFF_HIGHD_SKY_PROTECT:
    low-value stuff AND high D_g -> source skip
    but sky low scale-risk is protected as neutral context

FG_SKIP_04_DYNAMIC_VEGETATION_HIGHD:
    dynamic thing OR vegetation highD -> source skip

FG_SKIP_05_RISK_SKIP_STRUCTURE_PROTECT:
    dynamic/vegetation/lowtrust highD skip
    structure lowD protected as source

FG_SKIP_06_TRUE_COMPACT_KV:
    same as FG_SKIP_05 but true compact_kv, not bias approximation
```

### 4.3 需要记录的指标

```text
ATE_delta_h10/h15
[200,300)_delta
[400,600)_delta
boundary_10f_delta
boundary_20f_delta
context_empty_source_events
source_keep_ratio
source_skip_token_count
attention_mass_removed_before/after if available
attention_mass_to_static_anchor_before/after
attention_mass_to_highD_before/after
semantic_group_removed_mass
structure_protected_mass
```

### 4.4 成立标准

H1 进入 h15 条件：

```text
h10 ATE delta <= -1.5m
或 h10 [200,300) delta <= -3m
且 boundary_10f_delta <= +0.25m
且 boundary_20f_delta <= +0.25m
```

H1 strong pass：

```text
h15 ATE delta <= -3m
或 h15 [200,300) delta <= -5m
且 [400,600) regression <= +1m
```

失败分流：

```text
如果 source_keep_ratio 变化但 ATE 不变：检查被 skip source 是否有 attention mass。
如果 h10 有效但 h15 退化：进入 persistence / washout attribution。
如果 boundary 退化：不要继续 hard skip，改成 soft attenuation 或 structure/boundary protect。
如果 compact_kv 不如 bias：检查 compact path 是否改变 K/V shape、special tokens、protected tokens。
```

---

## 5. H2：SWA local-continuity semantic control

### 5.1 假设

SWA 是 local lossless memory，不能照搬 frame/global 或 TTT 的语义规则。SWA 中的语义风险区域可能短期污染 overlap source，但 hard remove 会破坏 adjacent chunk boundary。正确策略应区分 overlap / non-overlap、K / V、structure anchor / dynamic risk。

### 5.2 实验设计

候选：

```text
SWA_01_NONOVERLAP_DYNAMIC_REMOVE:
    dynamic/movable highD 只在 non-overlap previous source remove
    overlap source 全保护

SWA_02_OVERLAP_KEEPK_ATTENV:
    overlap risky source keep K, attenuate V

SWA_03_STRUCTURE_OVERLAP_PROTECT:
    road/building/wall/fence lowD 在 overlap 中强保护

SWA_04_SKY_ROAD_BOUNDARY_NEUTRAL:
    sky/road boundary lowD 保留 K，V 只在 highD/highconflict 时弱化

SWA_05_DYNAMIC_REMOVE_STRUCTURE_PROTECT:
    dynamic highD non-overlap remove + structure overlap protect
```

### 5.3 指标

```text
ATE_delta_h10/h15
[200,300)_delta
[400,600)_delta
boundary_10f_delta
boundary_20f_delta
chunk_boundary_pose_jump_delta
overlap_source_keep_ratio
nonoverlap_source_keep_ratio
K_keep_ratio
V_attenuation_mass
semantic_group_overlap_mass
SWA cache update norm
```

### 5.4 成立标准

```text
h10 [200,300) delta <= -3m
且 boundary_10f_delta <= +0.25m
且 boundary_20f_delta <= +0.25m
```

h15 strong pass：

```text
h15 ATE delta <= -3m
或 h15 [200,300) delta <= -5m
且 [400,600) regression <= +1m
```

失败分流：

```text
如果 [200,300) 改善但 boundary 退化：改成 K preserve / V attenuation。
如果 anchor/remove action 结果相同：先修 SWA source gate，让 K/V action 可区分。
如果 non-overlap remove 无效：说明 SWA 主要需要 overlap topology，不继续扩大 remove matrix。
如果 SWA 始终只局部有效：SWA 作为 local read辅助，不进入 long memory 主线。
```

---

## 6. H3：TTT semantic static-anchor 与 short-negative

### 6.1 假设

TTT 不适合使用粗 semantic scalar。它应使用语义辅助构造两类 evidence：

```text
positive long:
    stable structure / ground / building / wall / fence / road
    low D_g
    low TTT conflict
    low scale risk
    high masklet trust

short negative / no long write:
    dynamic thing / vegetation / shadow / lowtrust
    high D_g
    high TTT conflict or high scale risk
```

### 6.2 实验设计

候选：

```text
TTT_ANCHOR_01_STRUCTURE_LOWD_POSLONG
TTT_ANCHOR_02_STRUCTURE_LOWD_LOWCONFLICT_POSLONG
TTT_ANCHOR_03_STRUCTURE_LOWD_LOWCONFLICT_LOWSCALE_POSLONG
TTT_NEG_01_DYNAMIC_HIGHD_NO_LONG_WRITE
TTT_NEG_02_DYNAMIC_HIGHD_HIGHCONFLICT_SHORTNEG
TTT_NEG_03_VEGETATION_HIGHD_HIGHCONFLICT_SHORTNEG
TTT_MIX_01_ANCHOR_POS_DYNAMIC_SHORTNEG_SKY_NEUTRAL
TTT_MIX_02_ANCHOR_POS_RISK_NOWRITE_SKY_VEG_NEUTRAL
```

### 6.3 指标

```text
ATE_delta_h10/h15
[200,300)_delta
[400,600)_delta
TTT positive mass
TTT short negative mass
TTT no-long-write mass
per-branch update norm
post-zp delta norm
update_conflict_energy before/after
scale_state_proxy before/after
memory_ttt_rel_diff
Sim3Scale proxy
```

### 6.4 成立标准

```text
h10 ATE delta <= -1.5m
或 h10 [200,300) delta <= -3m
```

进入 full online 前必须：

```text
h15 ATE delta <= -3m
或 h15 [200,300) delta <= -5m
且 [400,600) regression <= +1m
且 TTT update norm 没有异常 spike
```

失败分流：

```text
如果 positive long 有效但 h15 弱：做 anchor persistence / W_long protection。
如果 short negative 有效但 [400,600) 回退：改成 no-long-write，不做 negative。
如果 TTT semantic 全部弱：TTT 主线回到 TTT-native cue / scale-state，语义只做 trust context。
```

---

## 7. H4：semantic-conditioned C23 residual path isolation

### 7.1 假设

语义最有希望的作用不是直接写 memory，而是重解释 C23 / D_g。v31/v34 已证明 semantic-conditioned C23 有 strong local signal，但 full online transfer 失败。需要拆清它在哪条 path 与 C9 冲突。

### 7.2 实验设计

不训练 trigger，不用 fixed chunk。使用 residual injection：

$$
D_{final}=D_{base}+\lambda(D_{sem}-D_{base})
$$

其中 $\lambda$ 为固定保守值或能量归一，不从数据拟合。

候选：

```text
SEM_C23_READ_ONLY:
    semantic residual 只用于 frame/global read

SEM_C23_NO_TTT:
    semantic residual 不进入 TTT write score

SEM_C23_NO_SWA:
    semantic residual 不控制 SWA source

SEM_C23_STRUCT_RESCUE:
    semantic residual + structure lowD protection

SEM_C23_SOURCE_SKIP:
    semantic residual 只用于 K/V source skip，不用于 attention bias
```

### 7.3 指标

```text
ATE_delta_h10/h15/full
[200,300)_delta
[400,600)_delta
read attention mass to highD
read attention mass to static anchors
SWA boundary metrics
TTT update conflict
semantic residual RMS
D_final distribution by semantic group
D_final vs D_base correlation
```

### 7.4 成立标准

```text
short pass:
    h10/h15 [200,300) <= -5m
    and [400,600) regression <= +1m

full stage pass:
    C9 + policy ATE <= 32m

Target pass:
    C9 + policy ATE <= 30m
```

失败分流：

```text
如果 read-only 有效、hybrid 失败：semantic cue 与 TTT/SWA 写入冲突，限制为 read-only。
如果 no-SWA 有效：SWA source path 是冲突源，禁用语义控制 SWA。
如果 no-TTT 有效：TTT write 是冲突源，语义只做 read/source cue。
如果所有 path isolation 都失败：semantic-conditioned C23 降级为 diagnostic。
```

---

## 8. H5：组合与 full online 验证

只有 H1-H4 中至少一个满足 h15 strong pass，才允许 full online。

### 8.1 组合原则

不允许盲目 all-memory 叠加。组合只能使用已经通过 path health gate 的组件。

候选组合示例：

```text
Combo A:
    frame/global VGGT-style source skip
    + TTT static anchor positive long

Combo B:
    frame/global source skip
    + SWA overlap structure protect

Combo C:
    semantic-conditioned C23 read-only residual
    + TTT static anchor positive long

Combo D:
    source skip dynamic/vegetation risk
    + semantic-conditioned C23 structure rescue
```

### 8.2 full online 指标

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
[200,300)
[200,400)
[400,600)
50f/100f/200f mean and worst
boundary_10f/20f
HMC state movement
TTT update norm
SWA cache health
```

### 8.3 full online 成立标准

```text
Stage success:
    ATE <= 32m
    且 [400,600) 不比 C9 回退 > +1m

Target success:
    ATE <= 30m
    no GT runtime action
    no offline trajectory rewrite
    no learned trigger/selector
```

---

## 9. 并行执行安排

### Codex A：H0 instrumentation / action realism

负责：

```text
H0A hook reachability
H0B attention mass explainability
H0C action distinguishability
synthetic stress test
```

若 H0A fail，立即修 hook，不跑 trajectory。

### Codex B：Frame/global source skip

负责 H1：

```text
FG_SKIP_01..06
chunk 6/10/16 h10
h10 gate 后 h15
```

### Codex C：SWA local continuity

负责 H2：

```text
SWA_01..05
boundary diagnostics
K/V split audit
```

### Codex D：TTT semantic anchor/negative

负责 H3：

```text
TTT_ANCHOR_*
TTT_NEG_*
TTT_MIX_*
```

### Codex E：semantic-conditioned C23 path isolation

负责 H4：

```text
SEM_C23_READ_ONLY
SEM_C23_NO_TTT
SEM_C23_NO_SWA
SEM_C23_STRUCT_RESCUE
SEM_C23_SOURCE_SKIP
```

### Codex F：report + full online gate

负责：

```text
统一 h10/h15 report
path health gate
full online candidate registry
Target-30 report
```

---

## 10. 失败时自动分流

### 10.1 action hook fail

```text
停止 rollout；
修 role tensor projection / CLI forwarding / model hook / source mask fallback。
```

### 10.2 attention mass 未实现

```text
不阻止 h10 rollout；
但所有 source skip 结果标记为 attention-mass-unverified；
Codex A 继续并行实现该 instrumentation。
```

### 10.3 h10 有效，h15 失败

```text
做 washout attribution；
判断是 TTT tail update、SWA cache refresh、global source update、merge/gauge 覆盖；
转 persistence mechanism，不继续扫 threshold。
```

### 10.4 SWA boundary 回退

```text
禁用 hard remove；
改 K preserve / V attenuation；
只在 non-overlap source skip；
保护 structure overlap anchors。
```

### 10.5 TTT semantic weak

```text
停止 semantic scalar / semantic coarse role TTT；
转 static-anchor positive long 或回到 TTT-native update_conflict/scale-state。
```

### 10.6 C9 full online 回退

```text
做 path isolation；
禁用冲突 path；
若 semantic read-only 仍回退，则停止 semantic deployment line。
```

---

## 11. 本计划的本质判断

v36B 的核心不是“降低 gate 标准”，而是区分：

```text
工程 hook 是否真实生效；
被跳过 source 是否有 attention mass；
semantic action 是否能改变 trajectory；
semantic correction 是否能持久；
semantic 是否和 C9 兼容。
```

缺 attention mass instrumentation 不应该让整个研究停下。它只应该限制我们对 source skip 机制的解释力度，而不应该阻止 h10/h15 causal exploration。

最终，如果 H1-H4 都失败，我们才能比较有把握地说：

```text
在当前 VideoMasklet/SPG 形式下，语义不是进入 ATE 30 的主杠杆；
语义应降级为 trust calibration / failure diagnosis / weak regularizer；
Target-30 主线应转向 trajectory-state / scale-state / TTT-native action。
```

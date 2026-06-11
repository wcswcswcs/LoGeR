# ACL2 v32：Semantic-Cue Local-to-Full Transfer 与 Target-30 并行实验计划

日期：2026-05-23  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / Semantic-conditioned C23 cue  
当前目标：先进入 KITTI01 ATE $30m$，同时解释为什么 v31 的 h10/h15 强局部语义 cue 没有转化为 full online 收益。  
重要边界：本计划允许使用 KITTI01 诊断病灶，但不允许为 KITTI01 或任何单个数据集专门调阈值、label value、chunk id 或 gamma。所有可部署策略必须由 runtime 信号触发，而不是由固定序列编号触发。

---

## 0. 背景与本轮必须改变的思路

v31 首次证明了一个重要现象：**语义不是完全没用**。`semantic-conditioned C23 / D_g reconditioning` 在短滚动中能够形成强局部修正。尤其是 `V31_A1B_SEM_Z_COARSE` 在修复 `_coarse` bug 后，经过 beta `5.25` 的最小修复，h10 和 h15 都能明显降低 `[200,300)` 病灶：

```text
h10 [200,300) delta = -5.9984m
h15 [200,300) delta = -6.1266m
h15 [400,600) delta = +0.2170m
```

但是它进入 full online 后完全没有转化为 Target-30：

```text
H9_BASE full ATE = 36.8192m
SEM_Z_COARSE_BETA525 full ATE = 36.6906m
full ATE delta = -0.1286m
[200,300) delta = -0.1846m
[400,600) delta = -1.7121m
```

同时，当前可计数 deployable online best 仍然是：

```text
C9_P0_R2 ATE = 33.7629m
```

如果以 Target-30 为阶段目标，仍差：

$$
33.7629 - 30.0 = 3.7629m
$$

因此下一步不能继续做普通的语义规则矩阵，也不能只做 masklet-level 单点因果干预。v31 的结果说明：**语义进入 cue 计算比语义直接进入 action 更有潜力**；但短滚动有效、full online 失败，说明我们真正需要解释的是：

> 为什么在同一个 chunk 的 causal fork 里，semantic-conditioned C23 可以强修 `[200,300)`；但从 frame 0 full online 运行时，这个修正几乎消失？

本计划的核心目标就是把这个 “local-to-full transfer gap” 拆开。只有解释这个 gap，才知道 Semantic Prior Generator 下一步应该做什么。

---

## 1. 总体目标

本阶段不再以“多试几个 sky/road/vegetation 规则”为目标，而是回答五个更根本的问题。

### 1.1 问题 A：v31 的局部强信号是真 cue，还是 sandbox parent-state 特化？

v31 的 h10/h15 强信号发生在 causal fork parent state 下。full online 从 frame 0 运行时，早期 chunks 会持续改变 HMC state、frame bias、SWA cache、TTT fast weights、merge/gauge state。因此局部效果可能依赖特定 parent state，无法从 full start 转移。

本问题要通过 **local-to-full counterfactual audit** 判断：

```text
同一 candidate 在以下条件中是否一致：
1. causal fork from H9 chunk10
2. full online from frame0, candidate active only at chunk10 window
3. full online from frame0, candidate active all chunks
4. full online from frame0, candidate active by runtime trigger
```

如果只有 causal fork 有效，则说明当前 semantic cue 是 parent-state local correction，不是 deployable full-run mechanism。

### 1.2 问题 B：semantic-conditioned C23 是否应该作为 “cue redefinition”，而不是 action rule？

之前语义多用于 memory action，例如 source skip、SWA cache、TTT positive/negative。v31 说明更有力的方向是直接改变 `D_g` 的定义：在每个 semantic group 内做 z-score / residual normalization，让 C23 读到的 query inconsistency 更符合类别分布。

因此本阶段必须将 Semantic Prior Generator 的角色改为：

```text
Semantic Prior Generator = cue reconditioner + memory role router
```

而不是仅仅输出 `A_tok` 或 semantic action mask。

### 1.3 问题 C：semantic-conditioned C23 能否叠加到当前 best C9，而不是只和 H9 read-only base 比？

v31 full online 只验证了 H9 base 与 SEM_Z_COARSE_BETA525；但当前 deployable best 是 `C9_P0_R2 = 33.7629m`。如果 semantic cue 只比 H9_BASE 好 `0.1286m`，却不能叠加 C9，则它没有进入 Target-30 的价值。

必须验证：

```text
C9 + semantic-conditioned C23
```

而不是只验证：

```text
H9 + semantic-conditioned C23
```

### 1.4 问题 D：full online 失败是否来自“所有 chunks 都激活”导致 cue drift / over-application？

h10/h15 repair 只是在局部 chunk10 附近有效。full online 如果把 semantic-conditioned cue 用在所有 chunks，可能会在早期 chunk、reset boundary、low-risk chunk 中引入错误干预。要验证 runtime activation 是否应当由异常信号触发，而不是全序列常开。

候选触发信号包括：

```text
semantic_z_high_mass
D_g semantic residual p90
frame-bias norm spike
TTT conflict spike
scale-risk spike
SWA boundary risk
local overlap drift proxy
```

### 1.5 问题 E：语义是否应该先服务 static scale-anchor，而不是 dynamic suppression？

长期看，Target-30 的缺口更像 trajectory / scale drift，而不只是动态区域污染。语义的潜力可能不是“跳过坏区域”，而是帮 C23/D_g 找出 **可跨 chunk 持久传播的 static scale anchors**：

```text
structure semantic
+ high mask trust
+ low D_g semantic z
+ low TTT conflict
+ low scale risk
+ high attention source mass
```

如果 static anchor path 能稳定改善 h15/full，而 dynamic skip path 只能 h10 有效，则后续 Semantic Prior Generator 应以 anchor construction 为主。

---

## 2. 固定基线与不可混淆边界

### 2.1 固定参考

所有实验必须同时报告：

```text
H9_BASE:
    v31 full-online H9 read-only/base reference if using v31 full setup

C9_P0_R2:
    current best deployable online TTT write
    ATE = 33.7629421029m

V31_SEM_Z_COARSE_BETA525:
    h10/h15 local gate-passing semantic cue repair
    full online ATE = 36.6905744722m
```

### 2.2 计数规则

```text
short h10/h15:
    diagnostic only

causal fork:
    diagnostic / oracle only

full online with no GT runtime action and no offline trajectory rewrite:
    can count as deployable candidate

fixed chunk-id activation:
    diagnostic only unless later replaced by runtime trigger

SemanticKITTI projected 3D semantic:
    sparse diagnostic / trust calibration only, not dense GT

video-masklet semantic:
    deployable candidate source but noisy
```

### 2.3 不允许的数据集调参

不允许：

```text
KITTI01 专用 threshold
KITTI01 专用 chunk id policy
某个 sequence 专用 semantic label value
某个 sequence 专用 beta/gamma
```

允许：

```text
用 KITTI01 诊断 failure mode
用同一套 runtime trigger 在 KITTI00/02/05 做诊断
记录不同数据集的 semantic/cue 分布差异
```

---

## 3. 核心假设与实验设计

---

## H1：v31 local-to-full gap 来自 full-run state drift，而不是 cue 本身无效

### 假设

`SEM_Z_COARSE_BETA525` 在 chunk10 的 h10/h15 causal fork 中有效，但 full online 失败，是因为 full-run 前序 chunks 改变了 chunk10 进入状态。也就是说，cue 本身有局部因果力，但 full run 中 parent state 不一致。

### 实验设计

固定 candidate：

```text
SEM_Z_COARSE_BETA525
```

对比四种运行方式：

```text
H1-00:
    H9 causal fork chunk10 h15
    candidate active from chunk10
    already exists as v31 short rollout reference

H1-01:
    full online from frame0
    candidate active only on chunks 10-12
    diagnostic only, chunk-specific

H1-02:
    full online from frame0
    candidate active on chunks 6-12
    diagnostic only, broader local window

H1-03:
    full online from frame0
    candidate active all chunks
    v31 full R1 reference

H1-04:
    full online from frame0
    candidate active by runtime trigger:
        activate if semantic_z_high_mass > EMA + k * MAD
        or C23 semantic residual p90 > EMA + k * MAD
```

其中 H1-01 / H1-02 是诊断，不允许作为最终 deployable 策略；H1-04 是可部署方向，因为它由 runtime 信号触发，不含固定 chunk id。

### 必须记录指标

```text
trajectory:
    Full ATE
    Rot
    RPE_t/r
    FinalErr
    [200,300)
    [200,400)
    [400,600)
    50f/100f/200f mean ATE
    YawRMSE
    Sim3 scale

cue/state:
    per_chunk semantic_z_high_mass
    per_chunk D_g_original_mean/p90
    per_chunk D_g_sem_z_mean/p90
    Jaccard(highD_original, highD_sem_z)
    frame_bias_norm
    frame_bias_highD_mass
    TTT memory rel diff
    SWA cache source keep/drop mass
    merge/gauge cursor diff
    hmc_state_hash consistency

transfer:
    chunk10 entry-state distance full vs causal-fork parent
    chunk10 D_g map diff full vs causal-fork
    chunk10 semantic_z map diff full vs causal-fork
```

### 可视化

```text
1. H9 causal fork vs full online chunk10 D_g map
2. original C23 vs sem_z C23 map for chunks 6/10/16
3. per-chunk semantic_z_high_mass curve
4. per-chunk frame_bias_norm curve
5. trajectory overlay:
       H9
       C9
       SEM_Z all chunks
       SEM_Z chunk10-only diagnostic
       SEM_Z runtime-trigger
6. segment waterfall:
       [0,100), [100,200), [200,300), [300,400), [400,600)
7. state distance timeline:
       TTT / SWA / merge-gauge
```

### 成立标准

H1 成立，如果满足：

```text
chunk10-only full online diagnostic recovers at least 50% of causal-fork [200,300) gain
or chunk10 entry-state / cue maps differ strongly between full online and causal-fork parent
```

更具体：

```text
H1 pass:
    H1-01 [200,300) delta <= -3m
    and H1-03 [200,300) delta > -1m
```

这说明 full online 常开失败来自 activation / state drift，而不是 local cue 无效。

H1 不成立，如果：

```text
H1-01 / H1-02 / H1-04 都没有明显恢复 local gain
```

此时 semantic cue reconditioning 降级为 sandbox diagnostic，不再作为 full-run主线。

---

## H2：semantic-conditioned C23 必须在 C9 best protocol 上验证

### 假设

C9 是当前 deployable best，但它和 H9 的 error 分布不同：C9 全局 ATE 更低，`[400,600)` 更好，但 `[200,300)` 更差。semantic-conditioned C23 可能与 C9 互补，也可能冲突。必须验证。

### 实验设计

固定 C9 protocol：

```text
C9_P0_R2 locked config
```

运行：

```text
H2-00:
    C9 repeat sanity

H2-01:
    C9 + SEM_Z_COARSE_BETA525 all chunks

H2-02:
    C9 + SEM_Z_COARSE_BETA525 runtime-trigger

H2-03:
    C9 + SEM_Z_COARSE_BETA525 only on high semantic_z anomaly chunks
    diagnostic if chunk list fixed, deployable if trigger-derived

H2-04:
    C9 + SEM_Z_FINE beta selected from v31 original best
    diagnostic comparison
```

### 必须记录指标

除 H1 指标外，增加：

```text
interaction:
    C9_delta_vs_H9
    SEM_delta_vs_H9
    C9+SEM_delta_vs_C9
    interaction_gain = observed(C9+SEM) - expected_additive

error redistribution:
    [200,300) change
    [400,600) change
    FinalErr change
    Yaw/scale change
```

### 成立标准

H2 通过：

```text
C9 + semantic cue improves C9 full ATE by >= 1.0m
or C9 + semantic cue improves [200,300) by >= 5m while [400,600) regression <= +1m
```

若只改善 H9 而不能改善 C9，则 semantic cue 只是弱 regularizer，不是 Target-30 主线。

---

## H3：semantic-conditioned C23 需要 EMA / sequence-stable normalization，而不是 per-window local z

### 假设

v31 local fork 有效，但 full online 失败，可能是因为 semantic z normalization 在 full sequence 中不稳定。单个 chunk 的 semantic 分布不同，导致 z-score 触发不一致。需要比较：

```text
per-chunk z
reset-window EMA z
sequence-EMA z
semantic-class robust quantile z
```

### 实验设计

构造四种 reconditioning：

```text
H3-01: per_chunk_sem_z
    current v31 behavior

H3-02: reset_block_ema_sem_z
    each reset block has EMA mean/std per semantic group

H3-03: sequence_ema_sem_z
    online EMA across chunks, no future frames

H3-04: robust_quantile_sem_z
    per semantic group robust q05/q95 normalization

H3-05: semantic_residual_to_static_anchor
    D_g residual relative to static semantic anchor set
```

所有候选先跑 h10/h15 chunk10，再只把过 gate 的进入 full online。

### 必须记录指标

```text
normalization:
    per_label mean/std/q05/q50/q95 of D_g
    z map saturation ratio
    z fallback ratio
    semantic group sample count
    low-support semantic group rate

stability:
    z map Jaccard across adjacent chunks
    high-risk semantic mass over time
    correlation(z_score, [200,300) improvement)
```

### 成立标准

H3 通过：

```text
EMA / robust normalization keeps h10/h15 local gain and improves full online transfer by >= 0.5m over per-chunk z
```

如果所有 normalization variants h10 有效但 full 都失败，说明问题不只是 z normalization，而是 memory persistence / state interaction。

---

## H4：semantic-conditioned C23 的真正价值是重新定义 read cue，而不是直接控制 TTT/SWA write

### 假设

过去语义直接控制 TTT/SWA/frame action 的效果弱；v31 说明语义重定义 `D_g` 更有潜力。因此下一步应系统比较：

```text
semantic used in cue computation
vs
semantic used in action mask
vs
semantic used in both
```

### 实验设计

固定 base：

```text
C23 past + pair/all read
probe_ttt_write
```

候选：

```text
H4-01:
    original C23 cue + original action
    baseline

H4-02:
    semantic-conditioned C23 cue only
    no semantic memory action

H4-03:
    original C23 cue + semantic memory action only
    previous v28/v30 style

H4-04:
    semantic-conditioned C23 cue + semantic memory action

H4-05:
    semantic-conditioned C23 cue + static anchor protection
```

### 必须记录指标

```text
read path:
    frame attention mass to high-D original
    frame attention mass to high-D semantic-z
    read bias norm
    entropy before/after
    source mass removed/protected

write path:
    TTT write role mass
    TTT update norm
    SWA cache mass
    global source mass
```

### 成立标准

H4 成立，如果：

```text
H4-02 >> H4-03
```

即 semantic as cue reconditioner 明显强于 semantic as action mask。若 H4-04 比 H4-02 不增益，停止 semantic all-memory action，保留 semantic cue-only。

---

## H5：Target-30 需要 distributed static scale-anchor，而不是单 masklet intervention

### 假设

v29C/v30 的单 masklet causal bank 很弱，不代表语义无用，而是单 masklet 太局部。语义的真正潜力是构造分布式 static scale anchors。

### 实验设计

构造 anchor set：

```text
Anchor candidates =
    semantic in {road, building, wall, fence, stable structure}
    AND masklet_trust high
    AND D_g_sem_z low
    AND TTT_conflict low
    AND scale_risk low
    AND frame/global source attention mass high
```

候选：

```text
H5-01:
    semantic structure only anchor

H5-02:
    structure + low semantic-z D_g

H5-03:
    structure + low semantic-z D_g + low conflict

H5-04:
    structure + low semantic-z D_g + low conflict + low scale-risk

H5-05:
    H5-04 + source keep in frame/global

H5-06:
    H5-04 + source keep + TTT positive long

H5-07:
    H5-04 + source keep + SWA overlap protect + TTT positive long
```

### 必须记录指标

```text
anchor:
    anchor coverage
    anchor temporal persistence
    anchor semantic label distribution
    anchor D_g/conflict/scale-risk distributions
    anchor source attention mass
    anchor TTT write mass
    anchor SWA cache mass

trajectory:
    h10/h15 ATE
    [200,300)
    [400,600)
    boundary_10f/20f
    Sim3 scale
    FinalErr
```

### 成立标准

H5 通过：

```text
h15 ATE delta <= -2m
or h15 [200,300) delta <= -5m
with [400,600) regression <= +1m
```

如果 distributed anchors 有信号，进入 full online runtime-trigger version。若无信号，停止 static-anchor line。

---

## H6：semantic risk negative 应该是短期 lifecycle，而不是 long-memory negative

### 假设

高风险语义区域可能短期污染 read/source，但长期 hard negative 会伤 continuity。应做短期 negative，而非长期负写。

### 实验设计

定义：

```text
RiskNegative =
    high semantic-z D_g
    AND high conflict or high scale-risk
    AND semantic in {movable, vegetation, uncertain, lowtrust}
```

动作：

```text
frame/global:
    source attenuation or source skip

SWA:
    non-overlap skip
    overlap soft value attenuation, preserve key

TTT:
    no long write
    optional short negative with TTL
```

候选：

```text
H6-01: frame/global risk source attenuation only
H6-02: H6-01 + TTT no-long-write
H6-03: H6-01 + TTT short-negative TTL=1
H6-04: H6-01 + TTT short-negative TTL=3
H6-05: H6-01 + SWA non-overlap skip, overlap preserve-key attenuate-value
```

### 必须记录指标

```text
negative mass
negative semantic distribution
short-negative TTL state
TTT long write removed mass
SWA overlap key/value mass
boundary regression
h10/h15 decay
```

### 成立标准

H6 通过：

```text
h10 [200,300) delta <= -5m
and h15 durability_ratio >= 0.45
and boundary_10f_delta <= +0.25m
```

If h10 strong but h15 weak, do washout attribution instead of threshold sweep.

---

## H7：short h10/h15 success must be explained before full online scaling

### 假设

如果 semantic cue works in short rollout but not full online, the missing mechanism is not “one more beta”; it is state transfer / memory persistence.

### 实验设计

For any candidate passing h10/h15:

```text
1. save HMC/TTT/SWA/merge states at candidate activation start
2. save same states at h10 endpoint
3. save same states at h15 endpoint
4. compare base vs candidate state movement
5. compare h10->h15 tail overwrite ratio
```

Metrics:

```text
TTT_all_overwrite_ratio
TTT_branch0_overwrite_ratio
SWA_cache_overwrite_ratio
frame_bias_tail_ratio
merge_gauge_overwrite_ratio
semantic_z_map_tail_drift
```

### 成立标准

Before full online:

```text
h15 durability gate pass
and overwrite ratio <= 0.5 for the memory state responsible for correction
or explicit persistence mechanism is present
```

If full online fails despite h15 pass, run full-vs-fork transfer audit before more full attempts.

---

## 4. 并行执行计划

为了加速，Codex 分成 6 条并行 track。

### Codex A：Local-to-Full Transfer Audit

任务：

```text
run H1-01 / H1-02 / H1-03 / H1-04
generate local_to_full_transfer_report.csv
generate cue_state_diff_dashboard
```

如果 H1-01 也失败：

```text
mark semantic cue as fork-specific
stop full online semantic cue expansion
```

如果 H1-01 成功但 H1-03 失败：

```text
implement runtime trigger and EMA normalization
do not use all-chunk activation
```

### Codex B：C9 Combination Track

任务：

```text
run C9 + SEM_Z_COARSE_BETA525 h10/h15 and full if h15 passes
compare interaction against C9 and H9
```

如果 C9+SEM 不改善 C9 by >= 1m：

```text
do not promote semantic cue to deployable Target-30 line
```

### Codex C：Semantic C23 Normalization Track

任务：

```text
implement reset-block EMA z
implement sequence-EMA z
implement robust quantile semantic z
compare against per-chunk z
```

If all full online fail:

```text
semantic cue reconditioning remains local diagnostic
switch to static-anchor / trajectory-state track
```

### Codex D：Distributed Static Scale Anchor Track

任务：

```text
construct anchor sets H5-01..H5-07
run h10/h15
record anchor persistence and memory mass
```

If no h15 signal:

```text
stop structure-anchor matrix
do not tune label values
```

### Codex E：Risk Short-Negative Lifecycle Track

任务：

```text
construct H6 candidates
measure h10/h15/boundary
focus on lifecycle not threshold
```

If SWA boundary regresses:

```text
preserve key
attenuate value
restrict to non-overlap source
```

### Codex F：Washout / Persistence Attribution

任务：

```text
for every h15-passing candidate:
    save state snapshots
    compute h10->h15 overwrite ratio
    classify washout path
```

If TTT tail washout:

```text
try positive long anchor persistence
```

If SWA washout:

```text
try source cache persistence
```

If merge/gauge washout:

```text
semantic memory path not sufficient; route to trajectory-state module
```

---

## 5. 必须记录的统一指标

每个 run，无论 short 或 full，都必须写：

```text
run_config.yaml
semantic_recondition_config.yaml
hmc_state_hash.jsonl
semantic_group_summary.jsonl
semantic_role_summary.jsonl
semantic_memory_path_summary.jsonl
context_skip_summary.jsonl
ttt_write_debug.jsonl
trajectory_diagnostics.csv
segment_error_50_100_200.csv
```

统一 performance fields：

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
50f_mean
100f_mean
200f_mean
```

统一 cue fields：

```text
D_g_original_mean/p90
D_g_sem_z_mean/p90
semantic_z_high_mass
z_saturation_ratio
z_fallback_ratio
highD_jaccard_original_vs_semz
per_label_Dg_mean/p90
per_label_conflict_mean/p90
per_label_scale_risk_mean/p90
```

统一 memory fields：

```text
frame_source_keep_ratio
global_source_keep_ratio
SWA_cache_keep_ratio
TTT_positive_mass
TTT_neutral_mass
TTT_negative_mass
TTT_no_write_mass
TTT_update_norm
SWA_boundary_10f_delta
SWA_boundary_20f_delta
merge_gauge_diff
```

---

## 6. 必须可视化

每个 candidate group 至少生成：

```text
1. RGB + semantic fine/coarse overlay
2. original C23 D_g heatmap
3. semantic-conditioned D_g heatmap
4. D_g difference map
5. highD original vs highD semantic-z mask overlay
6. semantic label distribution for highD semantic-z
7. per-chunk semantic_z_high_mass curve
8. per-chunk frame_bias_norm curve
9. source keep/drop maps for frame/global/SWA
10. TTT positive/neutral/negative maps
11. trajectory overlay:
        H9
        C9
        candidate
        GT
12. segment waterfall:
        [0,100), [100,200), [200,300), [300,400), [400,600)
13. h10 vs h15 durability plot
14. full-vs-fork cue/state difference plot
```

---

## 7. 成功标准与停止规则

### 7.1 h10 entry gate

```text
ATE delta <= -1.5m
or [200,300) delta <= -3m
```

### 7.2 h15 durability gate

```text
ATE delta <= -3m
or [200,300) delta <= -5m

and:
    durability_ratio >= 0.45
    [400,600) regression <= +1m
```

### 7.3 full online Target-30 gate

```text
KITTI01 full ATE <= 30m
no GT runtime action
no offline trajectory rewrite
no fixed chunk-id deployment rule
```

### 7.4 C9 interaction gate

```text
candidate_on_C9 improves C9 ATE by >= 1m
or improves [200,300) by >= 5m with [400,600) regression <= +1m
```

### 7.5 Semantic line continuation gate

继续 semantic mainline 需要至少满足：

```text
one of:
    semantic-conditioned cue improves full online by >= 1m
    semantic anchor h15 passes durability gate
    semantic runtime trigger transfers local h15 effect to full online
```

否则：

```text
Semantic Prior Generator remains diagnostic / regularizer.
Target-30 mainline should switch to explicit trajectory-state / scale-state / merge-gauge-aware correction.
```

---

## 8. Codex 失败自动分流

### 8.1 如果 action 等价

现象：

```text
SEM_ONLY / RISK_ONLY / SEM_RISK produces same masks or same trajectory deltas.
```

Codex 应做：

```text
1. compare actual action tensors;
2. check protected token fallback;
3. check source keep ratio;
4. check whether role projection collapsed to same highD mask;
5. do not run rollout until action Jaccard <= 0.85.
```

### 8.2 如果 local h10/h15 有效但 full online 失败

Codex 应做：

```text
1. run H1 local-to-full audit;
2. compare chunk10 entry state;
3. compare cue maps full vs fork;
4. test runtime trigger instead of all-chunk activation;
5. test C9 interaction before more H9 full runs.
```

### 8.3 如果 semantic-conditioned C23 fails in C9

Codex 应做：

```text
1. stop semantic cue deployment line;
2. keep semantic cue as diagnostic;
3. test static anchor only if H5 has h15 signal;
4. return Target-30 line to trajectory-state / scale-state.
```

### 8.4 如果 SWA improves local segment but boundary regresses

Codex 应做：

```text
1. preserve K, attenuate V only;
2. restrict skip to non-overlap source;
3. protect road/wall/fence/sky-horizon overlap anchors;
4. if boundary still regresses, stop SWA semantic action.
```

### 8.5 如果 h10 strong but h15 weak

Codex 应做：

```text
1. run state washout attribution;
2. if TTT tail washout: add positive long anchor persistence;
3. if SWA washout: add source cache persistence;
4. if merge/gauge washout: semantic memory cannot solve alone.
```

### 8.6 如果 no candidate reaches h10 gate

Codex 应做：

```text
1. do not launch h15 / full;
2. do not expand semantic rules;
3. test semantic-conditioned cue only if not yet tested;
4. otherwise downgrade semantic line.
```

---

## 9. Cross-dataset diagnostic without tuning

Cross-dataset runs are allowed only for explanation, not tuning.

Allowed:

```text
run fixed candidate on KITTI00/02/05
measure semantic_z distribution
measure D_g semantic residual distribution
compare failure mode
```

Forbidden:

```text
change beta for KITTI02
change sky role for KITTI05
change semantic label table for one sequence
choose chunk ids per dataset
```

Decision:

```text
If a candidate only works on KITTI01 chunk10, it is diagnostic only.
If the same runtime trigger works across sequences, it can be promoted.
```

---

## 10. 最终判断路径

本阶段结束时必须给出三类结论之一：

### Case A：Semantic cue transfer succeeds

```text
full online <= 30m
or C9+semantic improves by >= 1m and moves toward 30m
```

Then:

```text
promote semantic-conditioned C23 to mainline.
```

### Case B：Semantic local cue works but does not transfer

```text
h10/h15 pass
full online fail
C9 interaction fail
```

Then:

```text
semantic cue remains local diagnostic;
investigate trajectory-state / merge-gauge persistence.
```

### Case C：Semantic anchor/risk also weak

```text
H5/H6 no h15 signal
```

Then:

```text
Semantic Prior Generator is not Target-30 mainline.
Keep it for visualization, trust calibration, and weak regularization.
Move mainline to explicit trajectory-state / scale-state / TTT-native causal actions.
```

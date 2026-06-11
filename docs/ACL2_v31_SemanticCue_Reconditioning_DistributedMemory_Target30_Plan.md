# ACL2 v31：Semantic Cue Reconditioning 与 Distributed Memory Role 实验计划

日期：2026-05-23  
对象：LoGeR / HMC Pipeline v2 / VideoMasklet / SemanticKITTI sparse projection / Semantic Prior Generator  
近阶段目标：先把 KITTI01 deployable online ATE 从当前 $33.76m$ 平台推进到 $30m$ 以下  
长期目标：验证语义是否能成为重建质量提升的主因果变量，而不是只作为可视化或弱 regularizer  
本轮原则：不继续扩大粗语义规则矩阵；不把 h10 short rollout、oracle、GT/projection diagnostic、partial run 写成 deployable online success；不针对单个数据集调参，只做跨数据集/跨序列 failure mode 诊断。

---

## 0. 当前状态与本轮核心转向

截至 v30，当前最好可计数 deployable online TTT write 仍是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

目标先定为：

```text
KITTI01 ATE <= 30m
```

因此当前还差约：

$$
33.7629421029 - 30.0 = 3.7629421029m
$$

v30 做了正确的一步：不再手写简单语义规则，而是先做 action distinguishability 和 diverse masklet causal bank。工程上，v30 已经完成：

```text
1. diverse masklet selection；
2. action distinguishability audit；
3. clean h10 causal bank R2；
4. context empty source events = 0；
5. SWA boundary diagnostic。
```

但 v30 的关键负结果也非常明确：

```text
best h10 ATE delta       = -0.3883615963m
best h10 [200,300) delta = -2.4341499157m
best TTT ATE delta       = -0.1717134765m
SWA boundary 10f delta   = +0.4789576412m
SWA boundary 20f delta   = +0.3681139080m
```

这说明当前的 masklet-level direct intervention 没有 Target-30 级别的上界。尤其是 SWA rows 再次出现不同 masklet / anchor / remove action 轨迹效果几乎相同的现象，说明 SWA 语义 action 可能被 shared hook/source behavior 主导，而不是由具体 semantic identity 主导。

因此，v31 不再继续问：

```text
sky 要不要 skip？
vegetation 要不要 negative？
road 要不要 positive？
```

而是改问：

```text
语义如何重新解释已有 cue？
语义如何让 D_g / TTT conflict / scale risk 在不同类别内变得可比？
语义如何形成分布式 static scale anchors，而不是单个 masklet 的局部 action？
语义如何让 h10 局部修正持久到 h15 / full online？
```

本轮最重要的概念转向是：

> Semantic Prior Generator 不再是 semantic rule generator，而是 cue-conditioned memory role router。

换句话说，语义不是主动作，语义是解释 cue 的上下文。最终 action 必须由以下信息共同决定：

```text
fine semantic label
masklet trust
D_g read/source risk
TTT update_conflict_energy
scale-state risk
memory path type
attention/source usage
SWA boundary role
TTT write/update role
```

---

## 1. 本轮整体目标

v31 不是继续做“语义规则小矩阵”，而是要回答四个根本问题。

### 1.1 目标 A：语义是否应该先进入 cue 的定义，而不是只进入 action？

过去的流程是：

```text
C23 D_g cue -> semantic role action
```

但如果不同语义类别天然具有不同的 $D_g$ 分布，那么直接使用全局阈值会把 sky、road、building、vegetation 放到同一尺度上比较，这是不合理的。

本轮要验证：语义是否应先用于 **cue reconditioning**。

例如对每个 token $i$：

$$
z_D(i)=\frac{D_g(i)-\mu_{label(i)}(D_g)}{\sigma_{label(i)}(D_g)+\epsilon}
$$

$$
z_C(i)=\frac{C_{ttt}(i)-\mu_{label(i)}(C_{ttt})}{\sigma_{label(i)}(C_{ttt})+\epsilon}
$$

$$
z_S(i)=\frac{S_{scale}(i)-\mu_{label(i)}(S_{scale})}{\sigma_{label(i)}(S_{scale})+\epsilon}
$$

这不是问“$D_g$ 高不高”，而是问“在同类语义区域里，这个 token 是否异常高”。

### 1.2 目标 B：语义是否能帮助构造分布式 static scale-anchor set？

过去很多策略偏向 suppress 动态/不稳定区域。但如果目标是进入 ATE 30，核心可能不是“少写坏东西”，而是“稳定地写入足够多的长期 scale anchors”。

本轮要验证下面的大胆假设：

> 语义真正的价值不是 dynamic suppression，而是帮助发现长期可传播的 static scale anchors。

候选 anchor 不应只由 semantic 决定，而应由：

```text
structure semantic label
+ high masklet trust
+ low z_D
+ low z_C
+ low z_S
+ high source attention mass
+ high temporal/masklet consistency
+ projected 3D sparse anchor agreement
```

共同定义。

### 1.3 目标 C：语义是否能定义 short negative，而不是 long negative？

同一 semantic label 中既有短期有害区域，也有长期有用 context。之前 hard freeze / hard skip 多次显示：删掉有害方向的同时也会删掉 continuity。v7 freeze5 / freeze56 说明有害信息和有用尺度连续性混在一起。因此语义风险区域不应默认进入 long negative，而应优先进入 short negative / transient correction。

本轮定义：

```text
positive long = static scale anchor
neutral context = horizon / far background / low risk stuff
short negative = semantic-risk + high z_D + high z_C or high z_S
no long write = ambiguous or low trust
```

### 1.4 目标 D：如果语义仍然没有上界，应尽快降级为诊断，不再占主线

本轮必须设置硬停止规则。如果 semantic-conditioned cue、distributed anchors、short negative、diverse causal bank 都不能产生 h10/h15 级别强信号，则结论应是：

```text
Semantic Prior Generator 当前只能作为 diagnostic / trust calibration / weak regularizer；
Target-30 主线应转回 read cue / scale-state / trajectory-state / TTT-native causal action。
```

---

## 2. 不针对数据集调参的约束

本轮允许诊断不同数据集 / 序列的问题，但不允许为了某个数据集打榜而调参。

允许：

```text
1. 在 KITTI01 上诊断 failure mode；
2. 在 KITTI00/02/05 上检查同一规则是否出现相同模式；
3. 对不同数据集报告 semantic/cue 分布差异；
4. 用固定参数比较 failure mode。
```

不允许：

```text
1. KITTI01 专用 sky threshold；
2. KITTI01 专用 chunk id policy；
3. 某个 sequence 专用 gamma；
4. 某个 dataset 专用 semantic label table；
5. 根据 KITTI01 h10 结果手动调 label action 再宣称泛化。
```

Role router 的输入可以包含：

```text
semantic label
masklet trust
D_g / z_D
TTT conflict / z_C
scale risk / z_S
source attention mass
SWA overlap membership
TTT update norm
```

但不应包含：

```text
sequence id
absolute chunk id
dataset name
KITTI01-specific threshold table
```

---

## 3. 总体实验结构

v31 分成 7 个并行 Track，但所有 Track 都遵循 speed gate：不过 gate 就不跑 h15 / pairwise / full。

```text
Track 0: Action realism and distinguishability audit
Track A: Semantic-conditioned C23 / D_g reconditioning
Track B: Distributed static scale-anchor memory
Track C: Semantic-risk short negative lifecycle
Track D: Diverse masklet causal bank v2
Track E: Causal role router learning from masklet bank
Track F: Durability / washout attribution
Track G: Cross-sequence diagnostic without tuning
```

核心顺序：

```text
先确保 action 真不同；
再验证 semantic-conditioned cue 和 distributed anchors；
再看 short negative；
再用 diverse causal bank 找 oracle upper bound；
oracle 有上界才训练 role router；
h10 过 gate 才 h15；
h15 过 gate 才 no-GT selector / full online。
```

---

## 4. Track 0：Action Realism and Distinguishability Audit

### 4.1 假设

v28 / v30 多次出现“不同策略产生相同 trajectory delta”的问题。v31 必须先检查策略是否真的产生不同 action，而不是只看 run name。

### 4.2 实验设计

对所有候选 policy，在不跑 full rollout 前先导出实际 action tensors：

```text
R_frame_tok
R_global_tok
R_swa_tok
R_ttt_tok
frame_source_keep_mask
global_source_keep_mask
swa_cache_keep_mask
ttt_positive_mask
ttt_neutral_mask
ttt_negative_mask
ttt_no_write_mask
```

比较以下 pair：

```text
semantic-only vs risk-only
semantic-only vs semantic-risk
risk-only vs semantic-risk
anchor policy vs negative policy
SWA anchor vs SWA remove
C23 original vs semantic-conditioned C23
```

### 4.3 必须记录指标

```text
Jaccard(action_A, action_B)
source_keep_ratio_A / B
source_keep_ratio_delta
TTT positive/neutral/negative mass delta
attention mass removed
attention mass kept
SWA overlap source mass removed
SWA boundary source mass removed
TTT update norm removed
post-zp update norm changed
per-label action mass
per-masklet action mass
```

### 4.4 可视化

```text
action_mask_overlay_frame_global.png
action_mask_overlay_swa.png
ttt_role_overlay.png
semantic_label_overlay.png
D_g_overlay.png
z_D / z_C / z_S heatmap
action_jaccard_matrix.png
per_label_action_barplot.png
```

### 4.5 成立标准

候选允许进入 rollout 的最低条件：

```text
same-path Jaccard <= 0.85
or source_keep_ratio_delta >= 0.05
or TTT role mass delta >= 0.05
or removed attention mass delta >= 0.03
```

如果不满足，Codex 必须先修：

```text
role projection collapse
threshold collapse
protected token override
SWA special fallback
source mask not consumed
write role not applied
```

不允许继续跑名字不同但 action 等价的 rollout。

---

## 5. Track A：Semantic-conditioned C23 / D_g Reconditioning

### 5.1 假设

C23 是历史上最强主 cue 之一，但它目前不区分 semantic context。语义可能应该先用于重新定义 C23 / $D_g$，而不是只用于控制 memory action。

### 5.2 实验设计

以 C23 为基础，构造 5 类 semantic-conditioned read cue。

#### A0：baseline

```text
C23_past_original = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
```

#### A1：semantic z-score normalization

对 $D_g$ 做 label-conditioned normalization：

$$
z_D(i)=\frac{D_g(i)-\mu_{label(i)}(D_g)}{\sigma_{label(i)}(D_g)+\epsilon}
$$

再映射回 $[0,1]$：

$$
D_{g,z}(i)=\operatorname{clip}(\sigma(a z_D(i)+b),0,1)
$$

初始不调 $a,b$，使用固定 $a=1,b=0$。

#### A2：same-role support centroid

C23 的 support centroid 不再用所有过去帧 token，而只用同类 role：

$$
D_{g,role}(i)=1-\cos(q_i, \operatorname{centroid}_{j \in support, role(j)=role(i)} q_j)
$$

fallback：如果同类 support 数量不足，退回 original C23。

#### A3：static-anchor support centroid

只用 static scale-anchor candidates 作为 support：

```text
support = structure semantic
          AND trust high
          AND low z_D
          AND low z_C
          AND low z_S
```

#### A4：semantic-risk excluded support

支持集排除：

```text
low trust masklets
high z_D + high z_C tokens
movable / unstable semantic-risk tokens
```

#### A5：C23 original + semantic residual

保持原 C23，但加入语义归一化残差：

$$
D_{mix}=\operatorname{clip}(D_g + \lambda(D_{g,z}-D_g),0,1)
$$

第一轮固定 $\lambda=0.25$，不做大扫。

### 5.3 运行矩阵

先跑 h10 short rollout，不直接 full：

```text
A0 baseline C23 original
A1 semantic_z_D
A2 same_role_support
A3 static_anchor_support
A4 semantic_risk_excluded_support
A5 residual_mix_lambda025
```

优先 chunks：

```text
chunk 6
chunk 10
chunk 16
```

如果 h10 过 gate，再跑 h15。

### 5.4 必须记录指标

```text
D_g distribution by fine label
z_D distribution by fine label
support count by label/role
fallback ratio
same-role support hit ratio
static-anchor support hit ratio
D_g vs D_g_sem correlation
D_g_sem high-mass ratio
attention mass to high-D before/after
ATE_delta_h10/h15
[200,300]_delta_h10/h15
[400,600]_delta_h10/h15
boundary_10f/20f if SWA affected
```

### 5.5 可视化

```text
D_g_original_map.png
D_g_semantic_z_map.png
D_g_same_role_map.png
D_g_static_anchor_map.png
D_diff_sem_minus_original.png
per_label_D_distribution.png
support_centroid_coverage_heatmap.png
trajectory_h10_overlay.png
segment_delta_barplot.png
```

### 5.6 成立标准

Track A 通过 h10 条件：

```text
h10 ATE delta <= -1.5m
or h10 [200,300) delta <= -3m
```

进入 h15 条件：

```text
h10 gate pass
and [400,600) regression <= +1m
```

强成立条件：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
and durability_ratio >= 0.45
```

如果 A1/A2/A3 全弱，则语义不适合改 C23 cue，后续停止 semantic-conditioned C23。

---

## 6. Track B：Distributed Static Scale-Anchor Memory

### 6.1 假设

进入 ATE 30 的关键可能不是继续 suppress dynamic，而是建立一组跨 chunk 稳定传播的 static scale anchors。

### 6.2 Anchor 定义

定义 masklet/token anchor score：

$$
A_{anchor}=T_{mask}\cdot V_{sem}\cdot (1-D_{g,z})\cdot (1-C_{z})\cdot (1-S_{z})\cdot M_{attn}
$$

其中：

```text
T_mask: masklet trust
V_sem: semantic value, structure/fence/wall/road/building higher
D_g,z: semantic-conditioned read risk
C_z: semantic-conditioned TTT conflict
S_z: semantic-conditioned scale risk
M_attn: source attention mass or support usage
```

### 6.3 动作

对 high anchor tokens：

```text
frame/global:
    force source keep / protect source

SWA:
    protect overlap/local cache source

TTT:
    positive long write
```

对 neutral background：

```text
frame/global:
    partial keep

SWA:
    keep if boundary / overlap useful

TTT:
    neutral, not positive long
```

### 6.4 实验矩阵

```text
B0 baseline no semantic anchor
B1 semantic structure only
B2 structure + trust
B3 structure + trust + low z_D
B4 structure + trust + low z_D + low z_C
B5 structure + trust + low z_D + low z_C + low z_S
B6 B5 + attention source mass high
```

先 single-path：

```text
frame/global source protect
SWA cache protect
TTT positive long
```

再 pairwise：

```text
FG source protect
FG + SWA protect
FG + TTT positive
FG + SWA + TTT
```

### 6.5 必须记录指标

```text
anchor_token_count
anchor_masklet_count
anchor_coverage_by_label
anchor_overlap_membership_ratio
anchor_attention_mass
anchor_TTT_update_norm
anchor_SWA_source_mass
positive_long_write_mass
ATE_delta_h10/h15
[200,300]_delta_h10/h15
[400,600]_delta_h10/h15
h10_to_h15 durability ratio
```

### 6.6 成立标准

Track B 通过条件：

```text
h10 ATE delta <= -1.5m
or h10 [200,300) delta <= -3m
```

强通过条件：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
and [400,600) regression <= +1m
and durability_ratio >= 0.45
```

如果 B2 强而 B5 弱，说明 semantic structure 本身有效，不需要复杂 risk gating。  
如果 B5/B6 强而 B1/B2 弱，说明语义必须结合 cue。  
如果全部弱，static semantic anchor 不是主杠杆。

---

## 7. Track C：Semantic-risk Short Negative Lifecycle

### 7.1 假设

高风险语义区域不应做 long negative，而应做 short negative 或 no-long-write。高风险定义必须结合 semantic + cue。

### 7.2 Negative 定义

$$
R_{neg}=T_{mask}\cdot \mathbf{1}[z_D>\tau_D]\cdot \mathbf{1}[z_C>\tau_C \;\text{or}\; z_S>\tau_S]\cdot V_{risk}(label)
$$

初始固定：

```text
z_D > 1.0
z_C > 1.0 or z_S > 1.0
V_risk high for movable / vegetation / lowtrust / uncertain
```

### 7.3 动作

```text
frame/global:
    source skip or source attenuation

SWA:
    non-overlap source skip
    overlap source soft attenuation only
    key preserve, value attenuation optional

TTT:
    short negative or no long write
    never hard long negative in first round
```

### 7.4 实验矩阵

```text
C1 highD only
C2 highD + semantic risk
C3 highD + high conflict
C4 highD + high scale risk
C5 highD + semantic risk + high conflict
C6 highD + semantic risk + high scale risk
C7 highD + semantic risk + high conflict + high scale risk
```

每个先测：

```text
frame/global only
SWA only
TTT only
```

如果单 path 有 h10 信号，再组合。

### 7.5 必须记录指标

```text
negative_mask_mass
negative_label_distribution
negative_attention_mass
negative_SWA_overlap_ratio
negative_TTT_update_norm
short_negative_lifetime
h10/h15 ATE delta
[200,300] delta
[400,600] delta
boundary_10f/20f delta
```

### 7.6 成立标准

```text
h10 [200,300) delta <= -3m
or h10 ATE delta <= -1.5m
```

但必须同时满足：

```text
SWA boundary_10f_delta <= +0.25m
SWA boundary_20f_delta <= +0.25m
[400,600) regression <= +1m
```

如果 h10 强 h15 弱，进入 Track F durability attribution，不继续调阈值。

---

## 8. Track D：Diverse Masklet Causal Bank v2

### 8.1 目标

v29C 主要测 top road，v30 扩了 diverse bank 但仍在 chunk10 wave1。v31 要进一步确保 causal bank 覆盖真正可能有害/有用的 masklets，而不是只覆盖 high-support road。

### 8.2 Masklet selection

每个 chunk 选 24 个 masklets，来自以下 bucket：

```text
road / ground high support
building / wall / fence high support
vegetation / grass high D_g
sky / horizon high attention mass
movable thing if present
low-trust masklets
high TTT conflict masklets
high scale-risk masklets
high attention source mass masklets
SWA overlap boundary masklets
masklet-3D disagreement masklets
```

优先 chunks：

```text
chunk 6
chunk 10
chunk 16
```

### 8.3 Interventions

每个 masklet 不再全 path 全动作爆炸式跑，而是按 bucket 选择 path：

```text
source-heavy masklets:
    frame/global source keep vs skip

SWA-overlap masklets:
    keep anchor vs soft value attenuation vs remove non-overlap only

TTT-conflict masklets:
    positive long vs no long write vs short negative

static-anchor candidates:
    source protect + TTT positive long
```

### 8.4 Metrics

```text
masklet_id
fine_label
coarse_group
projected_3d_support_ratio
projected_3d_agreement
masklet_trust
D_g_mean/p90
z_D_mean/p90
C_ttt_mean/p90
z_C_mean/p90
S_scale_mean/p90
z_S_mean/p90
source_attention_mass
SWA_overlap_membership
TTT_update_norm
h5/h10/h15 ATE delta
[200,300] delta
[400,600] delta
boundary deltas for SWA
```

### 8.5 Gate

Masklet oracle 上界成立条件：

```text
h10 ATE delta <= -3m
or h15 ATE delta <= -3m
or h10/h15 [200,300) delta <= -5m
```

如果 diverse bank v2 仍不过，则语义 masklet causal upper bound 不成立，停止训练 role router。

---

## 9. Track E：Causal Role Router Learning

### 9.1 前置条件

只有 Track D oracle 过 gate，才训练 router。

### 9.2 输入特征

```text
fine label
coarse group
masklet trust
projected 3D support/agreement
mask temporal IoU
mask fragmentation
mask area
D_g mean/p90
z_D mean/p90
TTT conflict mean/p90
z_C mean/p90
scale risk mean/p90
z_S mean/p90
source attention mass
SWA overlap membership
TTT update norm
boundary proximity
```

### 9.3 输出

```text
R_frame: keep / partial / skip
R_global: keep / partial / skip
R_swa: protect / partial / weaken / remove_nonoverlap
R_ttt: positive_long / neutral / no_long_write / short_negative
confidence per path
```

### 9.4 模型限制

第一版只允许可解释模型：

```text
decision tree
rule list
logistic regression
small random forest with max_depth <= 4
```

不允许直接训练大型黑盒模型。

### 9.5 Generalization check

```text
train chunk10 -> test chunk6/chunk16
train chunk6+10 -> test chunk16
train KITTI01 chunk subset -> diagnostic run on KITTI00/02/05 fixed parameters
```

不允许使用 sequence id / absolute chunk id 作为特征。

### 9.6 通过标准

```text
test h10 ATE delta <= -1.5m
or test h10 [200,300) delta <= -3m
```

进入 full 前必须：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
and [400,600) regression <= +1m
and durability_ratio >= 0.45
```

---

## 10. Track F：Durability / Washout Attribution

### 10.1 目标

如果 h10 强而 h15 弱，要判断 correction 被哪条 state 洗掉：

```text
TTT tail update
SWA cache refresh
global/chunk source refresh
merge/gauge state
```

### 10.2 记录

```text
HMC state movement h10->h15
TTT state movement h10->h15
SWA cache movement h10->h15
global/chunk source mass movement
merge/gauge movement
role mass over time
anchor persistence over time
negative mass over time
```

### 10.3 判断

如果 TTT washout：

```text
尝试 positive long commit / anchor persistence / no-long-write for negative
```

如果 SWA washout：

```text
尝试 source cache persistence / overlap anchor protection / key preserve value attenuation
```

如果 merge/gauge washout：

```text
语义 memory path 可能不是主解法；转 trajectory-state / scale-state module
```

---

## 11. Track G：Cross-sequence / Cross-dataset Diagnostic

### 11.1 目标

检查同一套 semantic-cue rule 是否只在 KITTI01 有效。

### 11.2 数据

先诊断：

```text
KITTI00
KITTI02
KITTI05
```

如果有其他数据集，不为它们调参数，只输出 failure profile。

### 11.3 记录

```text
semantic label distribution
D_g distribution by label
z_D distribution by label
conflict distribution by label
scale-risk distribution by label
masklet trust distribution
anchor coverage
negative mass
ATE / Rot / FinalErr / segment errors
```

### 11.4 不允许

```text
不允许 KITTI00 专用 rule
不允许 KITTI02 专用 threshold
不允许 sequence-specific chunk id
```

如果规则只在 KITTI01 chunk10 有效，标记为 diagnostic，不进入 deployable path。

---

## 12. Codex 并行执行计划

### Codex A：Action audit and SWA hook differentiability

任务：

```text
1. 计算所有候选 action mask Jaccard；
2. 检查 SWA anchor/remove 是否产生不同 hook effect；
3. 检查 source keep ratio、attention mass removed、boundary source mass；
4. 如果 action 等价，停止 rollout并修 hook。
```

失败自动方向：

```text
if Jaccard > 0.95:
    check role projection collapse;
    check protected/special token fallback;
    check SWA source mask consumed;
    check role stream overwritten by default gate;
    do not run rollout.
```

### Codex B：Semantic-conditioned C23

任务：

```text
实现 A1-A5 cue variants；
跑 chunk6/10/16 h10；
生成 D_g distribution and maps；
只允许 h10 gate pass 后进入 h15。
```

失败自动方向：

```text
if same-role support fallback ratio > 0.5:
    switch to coarse role support;
if static-anchor support count too low:
    relax trust or conflict cutoff once;
if A1-A5 all weak:
    stop semantic-conditioned C23.
```

### Codex C：Static scale-anchor memory

任务：

```text
实现 B1-B6 anchor definitions；
单 path frame/global/SWA/TTT；
记录 anchor persistence 和 write mass。
```

失败自动方向：

```text
if structure-only works but risk-gated fails:
    simplify anchor rule;
if risk-gated works but structure-only fails:
    keep semantic as context, not direct rule;
if all weak:
    stop static-anchor semantic track.
```

### Codex D：Short negative lifecycle

任务：

```text
实现 C1-C7；
测试 frame/global/SWA/TTT single path；
SWA 必须记录 boundary metrics。
```

失败自动方向：

```text
if SWA boundary regression > +0.25m:
    switch hard skip to value attenuation;
    preserve key;
    apply only non-overlap source;
if TTT negative hurts ATE:
    use no-long-write instead of short negative;
if h10 strong h15 weak:
    send to Track F washout.
```

### Codex E：Diverse masklet bank v2

任务：

```text
按 bucket 选 chunk6/10/16 masklets；
生成 causal bank；
算 per-masklet causal effect；
输出 top positive/negative masklets。
```

失败自动方向：

```text
if only road selected:
    enforce semantic diversity quota;
if projected support sparse for sky:
    use video-masklet trust-only bucket, but mark no 3D support;
if no oracle upper bound:
    stop learned router.
```

### Codex F：Router learner

任务：

```text
oracle 过 gate 后训练 simple router；
做 cross-chunk validation；
不进入 full until h15 gate pass。
```

失败自动方向：

```text
if train works but test fails:
    remove absolute chunk features;
    increase non-semantic features;
    reduce label-specific thresholds;
if still fails:
    semantic route downgraded.
```

### Codex G：Durability / washout attribution

任务：

```text
分析 h10 -> h15 washout；
输出 state movement；
定位 TTT/SWA/global/merge。
```

失败自动方向：

```text
if TTT washout:
    anchor positive long commit;
if SWA washout:
    cache persistence;
if merge/gauge washout:
    semantic memory path is not sufficient;
    recommend trajectory-state module.
```

---

## 13. 统一指标与可视化要求

每个 run 必须记录：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
[200,300]
[200,400]
[400,600]
h5/h10/h15 deltas
boundary_10f
boundary_20f
chunk_boundary_pose_jump
```

Semantic/cue 指标：

```text
fine label coverage
coarse group coverage
masklet trust
projected 3D support/agreement
D_g mean/p90 by label
z_D mean/p90 by label
TTT conflict mean/p90 by label
z_C mean/p90 by label
scale risk mean/p90 by label
z_S mean/p90 by label
source attention mass by label
SWA source mass by label
TTT role mass by label
```

Action 指标：

```text
action Jaccard
source_keep_ratio
attention_mass_removed
SWA_overlap_source_mass_removed
TTT_positive_mass
TTT_neutral_mass
TTT_negative_mass
TTT_no_write_mass
post-zp_update_norm_change
```

可视化：

```text
semantic_overlay.png
projected_3d_overlay.png
masklet_trust_overlay.png
D_g_original_map.png
D_g_sem_map.png
z_D/z_C/z_S maps
action_mask_overlay_frame_global.png
action_mask_overlay_swa.png
ttt_role_overlay.png
source_attention_removed_heatmap.png
SWA_boundary_overlay.png
trajectory_overlay_h10_h15.png
segment_delta_barplot.png
per_label_cue_distribution.png
per_label_action_mass_barplot.png
h10_to_h15_durability_plot.png
```

---

## 14. Promotion Gates

### 14.1 h10 entry gate

候选进入 h15 的条件：

```text
h10 ATE delta <= -1.5m
or h10 [200,300) delta <= -3m
```

且：

```text
[400,600) regression <= +1m
context empty source events = 0
no failed chunks
```

### 14.2 h15 durability gate

候选进入 no-GT selector / full online 前必须满足：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
```

且：

```text
[400,600) regression <= +1m
durability_ratio >= 0.45
SWA boundary regression <= +0.25m if SWA path involved
```

### 14.3 Target-30 full gate

Full online success：

```text
KITTI01 full ATE <= 30m
counts_as_deployable_online = true
no GT runtime action
no offline trajectory rewrite
no selector using GT
```

### 14.4 Semantic mainline stop rule

如果以下全部成立：

```text
Track A no strong signal;
Track B no strong signal;
Track C no strong signal;
Track D diverse masklet oracle no upper bound;
no h15 durable candidate;
```

则结论为：

```text
Semantic Prior Generator 当前不能作为 Target-30 主线；
语义降级为 diagnostic/trust/weak regularizer；
主线转回 read cue / scale-state / trajectory-state / TTT-native causal action。
```

---

## 15. 最终预期

v31 要给出一个清晰结论，而不是继续产生大量小负/小正结果。

可能结论 A：semantic-conditioned C23 有强信号。  
则语义优先进入 cue definition。

可能结论 B：static scale anchors 有强 h15 信号。  
则语义的主价值是 durable anchor construction。

可能结论 C：short negative 有 h10 但无 h15。  
则语义只适合 transient suppression，需要 lifecycle/persistence 机制。

可能结论 D：diverse masklet oracle 仍无上界。  
则语义不是 Target-30 主因果变量，停止 semantic mainline。

本轮最重要的成功不是跑出某个小数值，而是判断：

> 语义是否能从“类别标签”变成“cue-conditioned memory role”。


# ACL2 v30: Semantic-Cue Causal Memory Bank and Role Router Experiment Plan

日期：2026-05-23  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / Video Masklet / SemanticKITTI sparse projection  
主开发集：KITTI Odometry Sequence 01  
阶段目标：先突破当前 `33.76m` 平台，进入 `KITTI01 ATE <= 30m`  
长期目标：`KITTI01 ATE <= 25m`  
当前 best deployable online TTT write：

```text
C9_P0_R2
ATE = 33.7629421029m
```

---

## 0. 本轮计划的核心变化

前几轮 Semantic Prior Generator 的做法，本质上还是：

```text
semantic label -> semantic role rule -> frame/global/SWA/TTT memory action
```

这条路已经被 v25B/v26/v27/v28/v29C 多轮实验证明不够强。语义 cache、fine label、path-specific role、token-exact conflict、token-level scale proxy、SemanticKITTI sparse projection、masklet-3D alignment 都已经不同程度接上，但没有任何 semantic candidate 进入 h15、selector 或 full online validation。v29C 进一步证明，projected 3D semantic 可以校准 VideoMasklet trust，但 top projected-support road masklet 的 causal bank 仍然没有过 gate。

因此 v30 不再继续扩大 “sky skip / vegetation skip / structure keep / lowstuff highD negative” 这类规则矩阵。本轮重新定义 Semantic Prior Generator：

```text
Semantic Prior Generator
    = semantic-conditioned cue calibrator
    + masklet-level causal memory role router
    + durable static scale-anchor memory builder
```

也就是说，语义不再直接决定 memory action。语义先帮助解释已有 cue，然后参与决定：

```text
1. D_g 在某个语义区域里是否异常？
2. TTT conflict 在某个语义区域里是否异常？
3. scale-risk 在某个语义区域里是否异常？
4. 这个 masklet 是否可信、连续、与 sparse 3D semantic anchor 一致？
5. 这个 masklet 在 frame/global/SWA/TTT 中到底应该作为 source、anchor、positive long、short negative，还是 neutral context？
```

本轮的核心思想是：

> 先测 masklet 在不同 memory path 中的真实因果作用，再让 Semantic Prior Generator 学会这些作用。

---

## 1. 当前证据与问题定位

### 1.1 已经确认的事实

当前可计数 deployable online TTT write best 仍然是：

```text
C9_P0_R2 = 33.7629421029m
```

阶段目标是：

```text
KITTI01 ATE <= 30m
```

当前差距是：

$$
33.7629421029 - 30.0 = 3.7629421029m
$$

长期目标 `25m` 的差距是：

$$
33.7629421029 - 25.0 = 8.7629421029m
$$

v29C 已经补齐 KITTI Odometry + SemanticKITTI sequence 01 数据，生成 sparse projected 2D semantic anchors。Phase 0 data gate 通过，projection hit rate 为 `757 / 757`。Sparse projection 平均 pixel coverage 约 `0.0341`，所以它是稀疏 anchor，不是 dense 2D GT。Masklet-3D alignment gate 通过，supportable masklet 支持率约 `0.9610`，structure/ground agreement ratio 约 `0.8353`。

v29C 也完成了 offline action distinguishability 和最小 runtime masklet-level h10 causal bank。但最好的 h10 ATE 改善约 `-0.3935m`，最好的 `[200,300)` 改善约 `-2.3655m`，没有达到 oracle gate：

```text
h10 ATE delta <= -3m
or h10 [200,300) delta <= -5m
```

因此没有启动 h15、learned router、no-GT selector 或 full online validation。

### 1.2 当前真正的问题

当前失败不能再简单归因于：

```text
semantic cache 没有命中
semantic 没接进 memory path
fine label 不可见
SemanticKITTI projection 数据缺失
VideoMasklet 完全不可用
```

这些都已经被不同阶段排查过。当前真正的问题是：

```text
semantic label 本身不是 trajectory drift 的直接因果变量。
```

语义告诉我们区域是什么：

```text
road / building / wall / fence / sky / vegetation / grass / moving car
```

但 memory controller 真正需要知道的是：

```text
这个 token 作为 frame/global K/V source 会不会污染 read？
这个 token 进入 SWA cache 会不会破坏 overlap continuity？
这个 token 写进 TTT fast weights 会不会破坏 scale / trajectory state？
这个 token 的 correction 到 h15 会不会被后续 memory 洗掉？
```

因此，下一轮必须从 “语义规则” 转向 “语义条件下的 cue calibration 和 causal role discovery”。

---

## 2. 本轮整体目标

v30 的目标不是再做一个更大的语义组合矩阵，而是回答五个更根本的问题。

### 2.1 目标 A：语义能否提升 C23 / D_g 这个主 read cue 本身？

历史上最大的进展之一来自 `C23 past` attention cue。它已经证明 LoGeR 内部 global query inconsistency 是强 read signal。过去我们把语义作为 action prior，但没有让语义改变 cue 本身。

本轮要验证：

> C23 的 support 和 normalization 是否应该按 semantic role 条件化？

例如，对 road token，不应把 support centroid 和 sky / vegetation / movable token 混在一起；对 building token，应优先和过去帧中的 structure anchor 比较；对 sky，则不应把 high-D 直接解释为 dynamic harmful。

### 2.2 目标 B：语义是否能帮助构建 durable static scale anchors？

过去过多关注 “跳过动态 / 跳过天空 / 跳过植被”。但进入 ATE 30 更可能需要长期稳定的 scale anchors。

本轮要验证：

> 语义 + D_g + TTT conflict + scale-risk + masklet trust 能否找到一组稳定 static scale-anchor masklets，并让它们在 frame/global/SWA/TTT 中持续发挥作用？

### 2.3 目标 C：语义是否只适合做短期 negative，而不适合做 long write？

v29C 显示 top road masklet 在 global/SWA source path 有弱信号，但 TTT positive/negative 几乎没信号。过去多轮也说明 TTT write 的主因可能不是语义类别，而是 fast-weight conflict / scale-state direction。

本轮要验证：

> 语义 negative 是否只应作为短期 source/filter/lifecycle signal，而不是长期 TTT negative write？

### 2.4 目标 D：masklet-level oracle 是否存在语义上界？

如果我们对每个 masklet、每条 path、每种 action 做 causal intervention，理论上能不能过 gate？

如果 masklet-level oracle 都过不了：

```text
h10 ATE delta <= -3m
or h15 ATE delta <= -3m
or [200,300) delta <= -5m
```

那么语义主线应降级为诊断 / trust calibration / weak regularizer。

如果 oracle 有上界，则说明当前规则没有学会，值得继续训练 / 归纳 role router。

### 2.5 目标 E：跨数据集只诊断，不调参

本轮允许诊断 KITTI00/01/02/05 或其他序列的 failure mode，但不允许为任何单个数据集单独调 threshold / label value / gamma。

所有规则必须固定，允许记录：

```text
哪类语义在不同数据集中更容易成为 drift source；
哪类 masklet trust 更低；
哪类 memory path 更容易失效；
```

但不能做：

```text
KITTI01 专用 sky threshold；
KITTI02 专用 vegetation value；
某 sequence 专用 chunk id 策略；
```

---

## 3. 核心假设

---

## H1：语义应该先用于校准 C23 / D_g，而不是直接决定 memory action

### 假设

当前 `D_g` 是 global query low-sim cue，但它把不同 semantic 类别放在同一个分布里解释。实际上，不同类别的 query inconsistency 分布不同：sky 天然可能不稳定，road/building 的 high-D 更可能代表几何失败，vegetation 的 high-D 更可能代表动态 / 非刚体 / 边界不稳定。

如果 H1 成立，则 semantic-conditioned C23 应比原始 C23 提供更干净的 read cue 或 source role。

### 实验设计

构造语义条件化 cue：

#### H1-A：semantic z-normalized D_g

对每个 token $i$，按照 semantic fine label 或 semantic role 分组，计算：

$$
z_D(i)=\frac{D_g(i)-\mu_{sem(i)}(D_g)}{\sigma_{sem(i)}(D_g)+\epsilon}
$$

然后用 $z_D$ 替代原始 $D_g$ 做 read/source role。

#### H1-B：same-role support centroid

原始 C23 cue：

$$
D_g(i)=1-\cos(q_i, \operatorname{centroid}_{j \in support} q_j)
$$

改为：

$$
D_{g,sem}(i)=1-\cos(q_i, \operatorname{centroid}_{j \in support,\ role(j)=role(i)} q_j)
$$

如果 same-role support 数量不足，则 fallback 到原始 C23 past。

#### H1-C：static-anchor support centroid

只使用过去帧中满足下列条件的 token 作为 support：

```text
structure semantic
masklet trust high
D_g low
TTT conflict low
scale-risk low
```

也就是：

$$
D_{g,anchor}(i)=1-\cos(q_i, \operatorname{centroid}_{j \in \mathcal{A}_{static}} q_j)
$$

其中 $\mathcal{A}_{static}$ 是 static scale-anchor support set。

### 实验矩阵

固定 parent：H9 causal fork snapshots。  
优先 chunks：`6, 10, 16`。  
优先 horizons：`h10`，通过 gate 后再 h15。

```text
H1_00_ORIG_C23
H1_01_SEM_Z_DG
H1_02_SAME_COARSE_ROLE_SUPPORT
H1_03_SAME_FINE_LABEL_SUPPORT
H1_04_STATIC_ANCHOR_SUPPORT
H1_05_STATIC_ANCHOR_SUPPORT_PLUS_ZDG
```

### 必须记录指标

```text
cue distribution:
    D_g mean / p90 / mass>0.5
    z_D mean / p90 per semantic label
    support_count per token
    same-role support fallback ratio
    static-anchor support ratio
    semantic group entropy of support

trajectory:
    ATE_delta_h10 / h15
    [200,300]_delta_h10 / h15
    [400,600]_delta_h10 / h15
    Rot_delta
    FinalErr_delta
    YawRMSE_delta
    Sim3 scale

memory / source:
    frame attention mass to high-D before/after
    global attention mass to high-D before/after
    source keep ratio
    context empty source events
```

### 假设成立标准

H1 成立条件：

```text
1. 任一 semantic-conditioned cue 在 h10 满足：
       ATE delta <= -1.5m
       or [200,300) delta <= -3m

2. 且 boundary / downstream 不明显恶化：
       [400,600) regression <= +1m
       context empty source events = 0

3. h15 confirmation 后满足：
       h15 ATE delta <= -1.5m
       or h15 [200,300) delta <= -3m
```

如果 H1 完全失败：

```text
停止 semantic-conditioned cue 线；
语义不再参与 C23 cue 本身，只保留在 action / trust 侧。
```

---

## H2：进入 ATE 30 的核心不是负样本 suppression，而是 static scale-anchor persistence

### 假设

过去的 source skip / scale commit / SWA/TTT 修正大多是 h10 有效、h15 衰减。原因可能是没有建立持续存在的 static scale-anchor set。语义最有价值的作用不是定义哪些区域要删，而是定义哪些 masklet 可以作为长期 scale anchors。

### 实验设计

构造 static scale-anchor set：

```text
A_sem = semantic in {road, building, wall, fence, sidewalk, stable terrain}
A_trust = masklet trust >= tau_trust
A_geom = D_g low
A_ttt = TTT conflict low
A_scale = scale-risk low
A_attn = frame/global source attention mass high
```

最终 anchor：

$$
A_{static}=A_{sem}\land A_{trust}\land A_{geom}\land A_{ttt}\land A_{scale}
$$

动作：

```text
frame/global:
    always keep as K/V source

SWA:
    protect as overlap/local source if present

TTT:
    positive long write
    no gradient reversal
    no short negative
```

### 实验矩阵

```text
H2_00_NO_ANCHOR_BASE
H2_01_SEM_STRUCTURE_ONLY
H2_02_SEM_STRUCTURE_PLUS_TRUST
H2_03_SEM_STRUCTURE_PLUS_TRUST_PLUS_LOWD
H2_04_SEM_STRUCTURE_PLUS_TRUST_PLUS_LOWD_PLUS_LOWCONFLICT
H2_05_SEM_STRUCTURE_PLUS_TRUST_PLUS_LOWD_PLUS_LOWCONFLICT_PLUS_LOWSCALE
H2_06_H2_05_PLUS_ATTENTION_MASS_TOPK
```

### 必须记录指标

```text
anchor set:
    anchor token ratio
    anchor masklet count
    anchor semantic label distribution
    projected 3D support ratio
    masklet temporal IoU
    masklet label stability
    source attention mass covered by anchors

memory:
    TTT positive long mass on anchors
    TTT post-zp update norm from anchors
    SWA protected source mass
    frame/global source keep mass
    h10->h15 anchor persistence ratio

trajectory:
    ATE h10 / h15
    [200,300] h10 / h15
    [400,600] h10 / h15
    boundary_10f / boundary_20f
    FinalErr / Yaw / Sim3 scale
```

### 假设成立标准

H2 通过条件：

```text
1. h10 ATE delta <= -2m
   or h10 [200,300) delta <= -4m

2. h15 durability_ratio >= 0.45，其中：

   durability_ratio = |delta_h15| / (|delta_h10| + epsilon)

3. [400,600) regression <= +1m

4. SWA boundary_10f_delta <= +0.25m
   and boundary_20f_delta <= +0.25m
```

如果 H2 中 semantic-only anchor 弱，而 H2_05 / H2_06 强：

```text
证明语义必须和 D_g / conflict / scale-risk 结合。
```

如果全部弱：

```text
static semantic anchor 不是主杠杆；转向 H3/H4。
```

---

## H3：风险语义 negative 只能短期存在，不能写进长期 TTT memory

### 假设

高风险语义区域不是一律无用。它们可能短期有用，但不适合长期 memory。因此 action 应该区分：

```text
source skip
SWA local attenuation
TTT short negative
TTT no long write
```

而不是统一 hard skip 或 long negative。

### 实验设计

定义 risk semantic set：

```text
R_sem = semantic in {movable, vegetation, grass, uncertain, low-trust masklet}
R_cue = z_D high OR TTT conflict high OR scale-risk high
R = R_sem AND R_cue
```

测试三类 action：

```text
source-only negative:
    frame/global source skip
    no TTT negative

short-memory negative:
    frame/global source skip
    SWA non-overlap skip or soft attenuation
    TTT short negative only for K chunks

long-negative ablation:
    frame/global source skip
    TTT long negative
```

### 实验矩阵

```text
H3_01_RISK_SOURCE_SKIP_ONLY
H3_02_RISK_SOURCE_SKIP_PLUS_SWA_NONOVERLAP_ATTEN
H3_03_RISK_SOURCE_SKIP_PLUS_TTT_SHORTNEG_K1
H3_04_RISK_SOURCE_SKIP_PLUS_TTT_SHORTNEG_K3
H3_05_RISK_LONG_NEGATIVE_ABLATION
H3_06_RISK_NO_LONG_WRITE_ONLY
```

### 必须记录指标

```text
risk set:
    risk token ratio
    risk semantic distribution
    risk trust distribution
    overlap vs non-overlap ratio
    attention mass removed

TTT:
    short negative mass
    long negative mass
    no-write mass
    W_short decay curve
    W_long update norm

SWA:
    non-overlap skip mass
    overlap attenuation mass
    boundary_10f / boundary_20f

trajectory:
    h10/h15 ATE
    h10/h15 [200,300]
    h10/h15 [400,600]
    durability ratio
```

### 假设成立标准

H3 成立条件：

```text
1. short negative 或 no-long-write 明显优于 long negative；
2. h10 [200,300) delta <= -3m 或 h10 ATE <= -1.5m；
3. h15 不被洗掉：durability_ratio >= 0.35；
4. [400,600) regression <= +1m。
```

如果 long negative 最好：

```text
说明某些 semantic-risk token 确实应长期反向；进入 branch/layer attribution。
```

如果所有 negative 都弱：

```text
停止语义 negative 主线，转 H2 static anchor 或 H4 causal bank。
```

---

## H4：masklet-level causal oracle 决定语义路线是否值得继续作为主线

### 假设

当前人工 role 规则可能没学会，但单个 masklet 在某些 path/action 上仍可能有显著因果作用。必须先测 oracle 上界。

### 实验设计

选择 diverse top masklets，而不是只选 top road。

每个 chunk 选择最多 16 个 masklet，覆盖：

```text
road high projected support
building / wall / fence high projected support
vegetation high-D
grass / terrain high conflict
sky / horizon high attention mass
movable thing if present
low-trust masklets
high TTT conflict masklets
high scale-risk masklets
overlap-boundary masklets
large-area masklets
small fragmented masklets
```

优先 chunks：

```text
chunk6
chunk10
chunk16
```

对每个 masklet $j$、path $p$、action $a$ 做 short rollout。

### Action set

```text
frame:
    source_keep
    soft_skip
    hard_skip

global:
    source_keep
    soft_skip
    hard_skip

SWA:
    cache_keep
    nonoverlap_skip
    overlap_protect
    value_atten_only

TTT:
    positive_long
    neutral_keep
    short_negative_K1
    short_negative_K3
    no_long_write
```

### Causal effect 定义

对于越低越好的 metric $M$，定义 improvement：

$$
E_{j,p,a}^{h}(M)=M_{base}^{h}-M_{intervention(j,p,a)}^{h}
$$

如果 $E>0$，表示该 intervention 改善了 metric。

对主指标分别记录：

$$
E_{j,p,a}^{h}(ATE)
$$

$$
E_{j,p,a}^{h}([200,300])
$$

$$
E_{j,p,a}^{h}([400,600])
$$

### 必须记录指标

```text
masklet identity:
    masklet_id
    fine_label
    coarse_group
    projected_3d_support_ratio
    projected_3d_agreement
    video_label_stability
    temporal_iou
    fragmentation
    area_mean
    birth/death frame

cue stats:
    D_g mean / p90
    z_D mean / p90
    TTT conflict mean / p90
    scale-risk mean / p90
    mask trust score
    source attention mass
    overlap membership

intervention:
    path
    action
    source_keep_ratio
    action mask Jaccard to base
    attention mass removed
    SWA cache mass removed/protected
    TTT role mass

trajectory:
    h5 / h10 / h15 ATE
    h5 / h10 / h15 [200,300]
    h5 / h10 / h15 [400,600]
    boundary_10f / boundary_20f
    FinalErr
    YawRMSE
    Sim3 scale
```

### Hypothesis pass criteria

H4 oracle passes if at least one masklet/path/action satisfies:

```text
h10 ATE delta <= -3m
or h15 ATE delta <= -3m
or h10 [200,300) delta <= -5m
or h15 [200,300) delta <= -5m
```

and:

```text
[400,600) regression <= +1m
SWA boundary regression <= +0.25m if path = SWA
context empty source events = 0
```

If H4 fails:

```text
Semantic Prior Generator is not a Target-30 mainline.
Semantic is downgraded to diagnostic / trust calibration / weak regularizer.
No learned semantic role router should be trained.
```

If H4 passes:

```text
Train or derive a simple causal role router from the causal bank.
```

---

## H5：从 causal bank 学 role router，而不是人工写语义规则

### 假设

如果 masklet-level oracle 有上界，当前失败是因为人工规则没学会。此时应该从 causal bank 学一个简单可解释的 role router。

### Feature set

每个 masklet/path 样本的输入特征：

```text
semantic fine label
semantic coarse group
projected_3d_support_ratio
projected_3d_agreement
video_label_stability
masklet temporal IoU
fragmentation
area_mean
D_g mean / p90
z_D mean / p90
TTT conflict mean / p90
scale-risk mean / p90
source attention mass
SWA overlap membership
TTT update norm
boundary proximity
memory path id
```

### Output labels

对每个 memory path 预测 action：

```text
R_frame:
    keep_source / soft_skip / hard_skip

R_global:
    keep_source / soft_skip / hard_skip

R_swa:
    cache_keep / overlap_protect / value_atten / nonoverlap_skip

R_ttt:
    positive_long / neutral / short_negative / no_long_write
```

### Model class

首选简单可解释模型：

```text
decision tree depth <= 4
rule list
logistic regression one-vs-rest
small random forest as diagnostic only
```

不允许使用复杂不可解释模型作为第一版。

### Generalization test

不允许只在 chunk10 学、chunk10 测。必须做 cross-chunk validation：

```text
train chunk6 -> test chunk10 / chunk16
train chunk10 -> test chunk6 / chunk16
train chunk6+10 -> test chunk16
```

不允许输入 absolute chunk id 或 sequence id。

### 成立标准

H5 通过条件：

```text
1. test chunk h10 ATE delta <= -1.5m
   or test chunk h10 [200,300) delta <= -3m

2. h15 durability_ratio >= 0.35

3. learned rules use non-dataset-specific features:
       semantic label
       D_g
       conflict
       scale-risk
       trust
       attention/source mass

4. no per-dataset threshold tuning.
```

If H5 fails despite H4 oracle pass:

```text
Semantic has oracle value but current features/rules do not generalize.
Codex should add attention mass / overlap / lifecycle features before full online.
```

---

## 4. Execution Plan and Parallel Codex Tracks

### Track A：Action distinguishability and role logic audit

目标：确认不同 semantic / risk / semantic-risk strategies 真的产生不同 action。

Codex A must output：

```text
action_tensor_summary.csv
action_jaccard_matrix.csv
per_path_source_keep_ratio.csv
per_label_role_mass.csv
attention_mass_removed.csv
swa_cache_removed_mass.csv
ttt_role_mass.csv
```

Hard gate：

```text
For any two candidates A and B:
    Jaccard(action_A, action_B) <= 0.85
    or source_keep_ratio difference >= 0.05
    or TTT role mass difference >= 0.05
```

If gate fails：

```text
Do not run rollout.
Codex must inspect:
    role projection collapse
    protected token fallback
    threshold collapse
    coarse-label overwrite
    source skip not applied
    SWA role identity ignored
```

---

### Track B：Semantic-conditioned C23 cue

目标：判断语义是否应该先改 cue，而不是直接改 action。

Codex B implements and tests：

```text
SEM_CUE_00_ORIG_C23
SEM_CUE_01_ZDG_BY_FINE_LABEL
SEM_CUE_02_ZDG_BY_ROLE
SEM_CUE_03_SAME_FINE_SUPPORT
SEM_CUE_04_STATIC_ANCHOR_SUPPORT
SEM_CUE_05_STATIC_ANCHOR_SUPPORT_PLUS_ZDG
```

Speed rule：

```text
Run h10 on chunks 6/10/16 first.
Only candidates with h10 ATE delta <= -1.5m or [200,300] <= -3m enter h15.
```

If all fail：

```text
Semantic-conditioned cue line stops.
```

---

### Track C：Static scale-anchor memory

目标：寻找能持久存在的 semantic scale anchors。

Codex C tests：

```text
ANCHOR_01_SEM_STRUCTURE_ONLY
ANCHOR_02_STRUCTURE_TRUST
ANCHOR_03_STRUCTURE_TRUST_LOWD
ANCHOR_04_STRUCTURE_TRUST_LOWD_LOWCONFLICT
ANCHOR_05_STRUCTURE_TRUST_LOWD_LOWCONFLICT_LOWSCALE
ANCHOR_06_STRUCTURE_TRUST_LOWD_LOWCONFLICT_LOWSCALE_ATTNTOPK
```

Actions：

```text
frame/global source always keep
SWA overlap protect
TTT positive long write
```

Gate：

```text
h10 ATE delta <= -2m
or h10 [200,300] delta <= -4m

and h15 durability_ratio >= 0.45
and [400,600] regression <= +1m
and boundary regression <= +0.25m
```

If semantic-only weak but cue-conditioned strong：

```text
Keep combined anchor policy.
```

If all weak：

```text
Static semantic anchor is not main lever.
```

---

### Track D：Risk semantic short-negative lifecycle

目标：验证 risky semantic regions 是否应短期处理，而不是长期写入。

Codex D tests：

```text
RISK_01_SOURCE_SKIP_ONLY
RISK_02_SOURCE_SKIP_SWA_NONOVERLAP_ATTEN
RISK_03_SOURCE_SKIP_TTT_SHORTNEG_K1
RISK_04_SOURCE_SKIP_TTT_SHORTNEG_K3
RISK_05_LONG_NEGATIVE_ABLATION
RISK_06_NO_LONG_WRITE_ONLY
```

Gate：

```text
short-negative/no-long-write better than long-negative;
h10 [200,300] delta <= -3m or h10 ATE <= -1.5m;
h15 durability_ratio >= 0.35;
[400,600] regression <= +1m.
```

If long-negative best：

```text
Run branch/layer attribution before full online.
```

If all weak：

```text
Stop semantic negative line.
```

---

### Track E：Diverse masklet causal bank

目标：建立真正的 semantic causal memory bank。

Codex E selects masklets from chunks：

```text
6, 10, 16
```

Selection must include：

```text
road high projected support
building/wall/fence high projected support
vegetation high-D
grass / terrain high conflict
sky/horizon high attention mass
movable thing if present
low-trust masklets
high TTT conflict masklets
high scale-risk masklets
overlap-boundary masklets
```

Action budget：

```text
max 16 masklets per chunk
max 4 memory paths
max 3 actions per path for first wave
```

Gate：

```text
h10 ATE delta <= -3m
or h15 ATE delta <= -3m
or h10 [200,300] <= -5m
or h15 [200,300] <= -5m
```

If no oracle upper bound：

```text
Semantic mainline is downgraded.
```

If oracle upper bound exists：

```text
Start Track F role router learner.
```

---

### Track F：Causal role router learner

Only starts if Track E passes.

Model：

```text
decision tree / rule list / logistic regression
```

Cross-chunk validation：

```text
train chunk6 -> test chunk10/16
train chunk10 -> test chunk6/16
train chunk6+10 -> test chunk16
```

Gate：

```text
test h10 ATE delta <= -1.5m
or test h10 [200,300] <= -3m

and no [400,600] regression > +1m
and no SWA boundary regression > +0.25m
```

If overfits：

```text
Add non-semantic features:
    attention source mass
    overlap membership
    TTT update norm
    conflict / scale-risk

Do not add sequence id or chunk id.
```

---

### Track G：Durability / washout attribution

Triggered when h10 strong but h15 weak.

Codex G records：

```text
HMC state movement
TTT state movement
SWA cache movement
global source state movement
merge/gauge movement
role mass over time
anchor persistence ratio
correction washout ratio
```

If washout is TTT：

```text
Try positive long anchor commit or W_long protection.
```

If washout is SWA：

```text
Try source cache persistence / overlap anchor protection.
```

If washout is merge/gauge：

```text
Semantic memory path is not sufficient; escalate to trajectory-state module.
```

---

## 5. Required Visualizations

Each phase must produce visual artifacts, not only CSV.

### 5.1 Semantic-conditioned cue maps

For selected frames：

```text
RGB
original D_g
z_D by semantic
same-role support count
static-anchor support count
semantic fine label overlay
VideoMasklet masklet id overlay
projected 3D sparse semantic overlay
```

### 5.2 Masklet causal bank visual dashboard

For each selected masklet：

```text
RGB with masklet boundary
semantic label
projected 3D support pixels
D_g heatmap inside masklet
TTT conflict heatmap inside masklet
scale-risk heatmap inside masklet
action applied
h5/h10/h15 trajectory delta summary
```

### 5.3 Memory path action visualization

```text
frame source keep/drop overlay
global source keep/drop overlay
SWA cache keep/drop/protect overlay
TTT positive/neutral/negative/no-write overlay
attention mass removed map
SWA boundary regression map
```

### 5.4 Durability visualization

```text
h5/h10/h15 metric curves
anchor persistence over time
role mass over time
TTT/SWA/global/merge state movement over time
[200,300] vs [400,600] tradeoff plot
```

### 5.5 Cross-dataset diagnostic visualization

With fixed rules only：

```text
semantic label distribution per dataset
masklet trust distribution per dataset
D_g/conflict/scale-risk distribution per semantic label
failure segment overlays
no threshold tuning per dataset
```

---

## 6. Metrics to Record in Every Candidate

### 6.1 Trajectory metrics

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
[200,300]
[200,400]
[400,600]
chunk6-10 error
chunk16 error
```

### 6.2 Memory health metrics

```text
frame source keep ratio
global source keep ratio
SWA cache keep ratio
SWA overlap protected mass
TTT positive mass
TTT neutral mass
TTT short negative mass
TTT no-write mass
TTT post-zp update norm
TTT conflict mean / p90
scale-risk mean / p90
context empty source events
```

### 6.3 Semantic and masklet metrics

```text
fine label
coarse group
masklet id
area mean
temporal IoU
fragmentation
birth/death frame
label stability entropy
projected 3D support ratio
projected 3D agreement ratio
mask trust score
VideoMasklet confidence
```

### 6.4 Action distinguishability metrics

```text
action Jaccard matrix
source_keep_ratio difference
role_mass difference
attention_mass_removed
SWA_cache_mass_removed
TTT_update_norm_removed
```

### 6.5 Durability metrics

```text
durability_ratio_h15_h10
h10_to_h15_washout_ratio
TTT_state_movement_ratio
SWA_state_movement_ratio
global_source_movement_ratio
merge_gauge_movement_ratio
anchor_persistence_ratio
```

---

## 7. Failure Routing for Codex

### 7.1 If action tensors are equivalent

Symptoms：

```text
SEM_ONLY / RISK_ONLY / SEM_RISK produce identical trajectory delta.
action Jaccard > 0.95.
```

Codex actions：

```text
1. inspect role projection collapse;
2. inspect protected token fallback;
3. inspect threshold collapse;
4. inspect coarse label overwrite;
5. inspect SWA role identity ignored;
6. do not run rollout until action tensors differ.
```

### 7.2 If semantic-conditioned C23 fails

Codex actions：

```text
1. report support fallback ratio;
2. check same-role support count;
3. inspect semantic support contamination;
4. stop semantic-conditioned cue line if no h10 signal;
5. continue H2/H4 instead.
```

### 7.3 If static anchors are weak

Codex actions：

```text
1. inspect anchor coverage;
2. inspect anchor attention mass;
3. inspect anchor projected 3D agreement;
4. add attention-mass top-k condition;
5. if still weak, stop static anchor line.
```

### 7.4 If risk negative improves h10 but hurts h15

Codex actions：

```text
1. reduce long negative;
2. convert to short negative;
3. protect overlap anchors in SWA;
4. add W_long / W_short attribution;
5. do not run full until durability improves.
```

### 7.5 If SWA improves [200,300] but boundary regresses

Codex actions：

```text
1. switch hard skip to soft value attenuation;
2. preserve key, attenuate value;
3. apply only to non-overlap source;
4. protect road/wall/fence overlap anchors;
5. require boundary_10f/20f <= +0.25m before promotion.
```

### 7.6 If masklet oracle has no upper bound

Codex actions：

```text
1. downgrade Semantic Prior Generator mainline;
2. keep semantic only as diagnostics/trust calibration;
3. redirect to TTT-native / scale-state / read-cue source-skip lines;
4. do not train role router.
```

### 7.7 If oracle works but router does not generalize

Codex actions：

```text
1. add attention mass feature;
2. add overlap membership feature;
3. add update norm feature;
4. reduce label-specific rules;
5. check cross-chunk generalization;
6. do not use sequence id or absolute chunk id.
```

---

## 8. Cross-Dataset Diagnostics Without Tuning

After any candidate passes h15 gate on KITTI01, test the same candidate on：

```text
KITTI00
KITTI02
KITTI05
```

Rules：

```text
same semantic taxonomy
same thresholds
same role router
same gamma / beta / lifecycle
no per-dataset modifications
```

Record only diagnostic differences：

```text
which semantic groups dominate failures
which memory path fails
whether masklet trust degrades
whether projected 3D support differs
whether source skip causes boundary regression
```

No dataset-specific tuning is allowed.

---

## 9. Promotion Rules

### 9.1 Enter h15

A h10 candidate enters h15 if：

```text
h10 ATE delta <= -1.5m
or h10 [200,300] delta <= -3m
```

and:

```text
[400,600] regression <= +1m
context empty source events = 0
SWA boundary regression <= +0.25m if path includes SWA
```

### 9.2 Enter no-GT selector

A h15 candidate enters selector if：

```text
h15 ATE delta <= -3m
or h15 [200,300] delta <= -5m
or durability_ratio >= 0.45 with h10 [200,300] <= -5m
```

### 9.3 Enter full online

A candidate enters full online if：

```text
selector proxy correlation is positive and stable
and selected action differs from base action
and predicted full ATE target <= 31m
```

### 9.4 Target-30 success

A deployable full online candidate succeeds if：

```text
KITTI01 ATE <= 30m
counts_as_online_success = true
uses_no_GT_runtime = true
no postprocess trajectory rewrite
no sequence-specific tuning
```

---

## 10. Expected Outcomes and Decisions

### Outcome A：semantic-conditioned C23 or static anchors produce strong durable signal

Decision：

```text
Promote Semantic Prior Generator as active mainline.
Start role router integration and h15/full validation.
```

### Outcome B：masklet oracle has upper bound but hand rules fail

Decision：

```text
Train simple causal role router from bank.
Do not continue manual semantic rules.
```

### Outcome C：masklet oracle has no upper bound

Decision：

```text
Semantic is not Target-30 mainline.
Use semantic only for visualization, trust calibration, failure diagnosis.
Return mainline to TTT-native / scale-state / read cue interventions.
```

### Outcome D：h10 strong but h15 weak

Decision：

```text
Focus on durability and washout.
Do not add more semantic categories.
```

### Outcome E：cross-dataset diagnostics show semantic labels differ but fixed rules remain stable

Decision：

```text
Keep rule as general prior.
Do not tune dataset-specific thresholds.
```

---

## 11. Minimal First 48-Hour Execution

To accelerate, the first two days should not run a full matrix. Run only:

```text
Day 1:
    Track A action distinguishability audit
    Track B semantic-conditioned C23 h10 for chunks 6/10/16
    Track E diverse masklet causal bank wave 1 for chunk10

Day 2:
    Track C static scale-anchor h10 for chunks 6/10/16
    Track D risk short-negative h10 for chunks 6/10/16
    Track G washout attribution only for any h10-strong candidate
```

Stop early if:

```text
action tensors equivalent;
no candidate achieves h10 ATE <= -1.5m or [200,300] <= -3m;
masklet oracle has no upper bound;
SWA boundary regression dominates.
```

Do not run pairwise / all-memory / full online until the above gates pass.

---

## 12. Final Summary

The next step is not another semantic rule sweep. The next step is to test whether semantic masklets have causal memory roles.

The plan therefore changes the question from:

```text
Should sky / road / vegetation be skipped or written?
```

to:

```text
For this specific masklet, in this specific memory path, under this cue condition,
which action improves future trajectory?
```

If this causal bank shows a real upper bound, Semantic Prior Generator becomes a learned role router. If not, semantic should be downgraded to diagnostic support, and the main Target-30 path should return to read-cue / TTT-native / trajectory-state interventions.

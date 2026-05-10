# ACL2 v7：TTT 写入优先的正负证据、语义先验与 `[200,300)` 病灶因果实验计划

日期：2026-05-07  
对象：LoGeR / HMC Pipeline v2 / KITTI01  
当前固定 read cue：`acl2.gg.qq.low.g2_3.past_only.headmean.robustq`  
当前固定 read path：frame-attention `pair/all`，`beta=4.75`  
当前固定 commit protocol：`probe_ttt_write`  
本轮暂不探索 SWA 变体：SWA 保持当前可复现的 `ACL2V5_SWKS3_03` 协议，不新增 SWA gate / replace / keep-scope sweep。

---

## 0. 本轮计划的定位

v4/v5/v6 已经把 read cue 主线收敛到 `C23 past`，并且 `pair/all` frame-attention read intervention 是当前最大收益来源。v6 又证明 Stage C/D 语义模块可以通过 cache / no-op / pass-through parity gate 接回 HMC，且 `[200,300)` 是当前 KITTI01 的核心病灶。现在继续泛扫 read cue 或 SWA gate 的边际价值不高，本轮应把 **TTT 写入机制** 放到最高优先级。

本轮计划不把 TTT 写入理解成“把某些 token 写多一点或写少一点”这么简单。当前实验已经反复显示，单纯少写 dynamic / high-risk token 通常会改善 rotation、final error 或 yaw，但 ATE 和 `[200,300)` 主病灶基本不动。因此，本轮 TTT 写入要升级为三类证据的显式建模：

```text
positive evidence:
    应该写入长期 TTT memory 的 token / group / update direction

neutral evidence:
    当前 chunk 有用，但不应强长期传播的 token / group

negative evidence:
    应该被反向抵消、遗忘或从长期 fast weights 中排除的 token / group / update direction
```

形式上，本轮的核心实验对象不再只是：

$$
G_{commit}=\sum_i a_i G_i
$$

而是：

$$
G_{commit}=G_{pos}+\lambda_{neu}G_{neu}-\gamma G_{neg}
$$

其中 $G_i$ 表示 token 对 TTT fast weights 的 replay update 贡献，$G_{pos}$ 是正证据写入，$G_{neu}$ 是中性保留，$G_{neg}$ 是负证据反向写入或反向抵消。

---

## 1. 当前事实与设计约束

### 1.1 当前固定 baseline

本轮所有实验都从已经通过复现 gate 的 `ACL2V6_B0_SWKS3_reference_rerun` 出发。固定协议为：

```text
cue = C23 past_only
read = frame pair/all
beta = 4.75
write score = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA = SWKS3-style fixed protocol
RESET_EVERY = 5
```

当前可复现 baseline：

```text
B0_SWKS3:
ATE / Rot = 36.4161 / 6.6128
FinalErr = 5.798
[200,300) = 77.831
YawRMSE = 3.765
```

当前 v6 tiny best：

```text
P3G_01:
ATE / Rot = 36.4017 / 6.2289
FinalErr = 3.509
[200,300) = 77.568
YawRMSE = 3.390
```

但 `P3G_01` 只比 `TTGR_03` 好 `0.0044m`，不能被视为语义突破；它只是说明 semantic soft modulation 和 TTGR 不冲突，且在平台内有极小正收益。

### 1.2 本轮不变项

为了让 TTT 写入实验可归因，本轮固定如下内容：

```text
read cue:        C23 past_only
read path:       frame-attention pair/all
read beta:       4.75
commit:          probe_ttt_write
reset:           RESET_EVERY=5
SWA:             fixed SWKS3 protocol, no new SWA exploration
Stage C:         cached require-hit only, no inline compute during benchmark
Stage D:         semantic prior only through explicit TTT write source or group routing
```

如果任何 run 没有通过 B0 parity 或写入路径 debug 缺失，不进入指标比较。

### 1.3 为什么暂不探索 SWA，但保留 SWKS3 固定协议

这里的“暂不探索 SWA”不是把 SWA 全关，而是不再新增 SWA 写入策略变量。原因是当前最好可复现系统已经包含 SWKS3 风格 SWA active protocol；如果为了 TTT 实验把 SWA 突然关掉，会改变整体 memory state 与 current best 的上下文，导致 TTT 结论难以转回当前主线。

因此本轮采用：

```text
SWA as fixed background protocol:
    keep both_overlap
    overlap source replacement = source / kv / alpha 0.50 / last

TTT as active experimental variable:
    write score
    positive/negative group
    branch
    layer
    direction
    chunk trigger
    semantic group routing
```

如需隔离纯 TTT effect，只在少数 diagnostic run 中用 `TTEX_01` 作为 TTT-only reference，不作为主开发 baseline。

---

## 2. 本轮整体目标

本轮有五个整体目标。

### 目标 A：把 `[200,300)` 病灶从“定位”推进到“可归因”

v6 已经证明 `[200,300)` 是稳定 worst segment，`freeze chunks 4,5,6` 可以把该段从约 `77.8m` 降到约 `25.9m`，但会把全局 ATE 崩到 `60m+`。这说明 chunks 4/5/6 的 memory state 里同时有有害方向和有用连续性。下一步必须定位：

```text
哪个 chunk 触发？
哪个 TTT branch 触发？
哪个 TTT layer 触发？
哪些 token group 贡献有害方向？
哪些 token group 维持全局 continuity？
```

### 目标 B：把 TTT write 从 scalar gate 升级为 positive / neutral / negative evidence replay

之前的 `stage_d*sqrt(1-D_g)`、explicit dyn veto、sparse write、focal static、post-commit projection 等大多是单 scalar 或 post-hoc filter。它们证明了 TTT 写入有弱信号，但不能解决主病灶。本轮要显式验证：

```text
少写 high-risk token 是否不够？
对 high-risk token 做负方向 replay 是否更有效？
负方向是否必须限制在特定 branch / layer / group？
```

### 目标 C：让语义 prior 进入 TTT 写入，但不能继续做单 scalar semantic value

Semantic Prior Generator v2 的设计原则是职责分离：几何负责 eligibility，语义负责 value，mask quality 负责 trust / routing，chunk budget 不应和语义混成一个量。本轮要真正利用这个思想，而不是只把 `V_sem` 乘到全局 write score 上。

具体目标是验证：

```text
structure semantic 是否适合作 positive write anchor？
sky / vegetation / grass 是否只能 soft neutral，而不能 hard veto？
low-value stuff 是否只有在 D_g / uncertainty 同时高时才进入 negative replay？
语义 group 是否应该作用于不同 branch / layer？
```

### 目标 D：系统探索所有已找到 cue 对 TTT 写入的作用，但先做小矩阵

本轮会用到的 cue 包括：

```text
D_g = C23 past global query low-sim cue
stage_d = 当前 safe write baseline score
explicit_dyn = Stage B explicit dynamic cue
old_dyn = old_dyn / dyn family diagnostic
C_anchor / G_write_geo = 几何 anchor / eligibility
C_unc / C_occ = uncertainty / occlusion risk
semantic groups = structure / low-value stuff / movable / uncertain
TTT residual = update-needed / memory mismatch diagnostic
```

但不会把所有 cue 做全组合。每个阶段只回答一个假设，避免不可归因的大矩阵。

### 目标 E：明确晋级标准，避免把 tiny best 误判成突破

当前所有 `36.4x` 结果都离最终 `KITTI01 ATE < 30m` 很远。本轮将同时定义相对成功和最终成功：

```text
relative success:
    比 B0 / TTGR_03 / P3G_01 有稳定可解释改进

segment success:
    明显降低 [200,300)，且不把错误转移到 [100,200) 或 [300,400)

final success:
    KITTI01 ATE < 30m
```

---

## 3. 核心假设

本轮围绕九个假设组织实验。

---

## H1：`[200,300)` 的主因是 chunks 4/5/6 的 TTT fast-weight 有害方向，而不是全局均匀误差

### 假设内容

`freeze chunks 4,5,6` 能大幅降低 `[200,300)`，说明这个病灶对前序 chunk 的 memory state 敏感。但 hard freeze 同时破坏全局 trajectory，说明 chunks 4/5/6 中既有有害方向也有有用 continuity。本假设认为：

> `[200,300)` 的主病灶来自 chunks 4/5/6 中某些 TTT branch / layer / token group 的有害方向，而不是所有 TTT 写入都应该被抑制。

### 实验设计

第一批只做 lesion / state replacement，不引入语义新策略。目的是定位 causal scope。

#### 组 H1-A：chunk-level lesion

固定 B0 协议，分别做：

```text
freeze chunk 4 only
freeze chunk 5 only
freeze chunk 6 only
freeze chunks 4+5
freeze chunks 5+6
freeze chunks 4+6
freeze chunks 4+5+6 reference repeat
```

其中 freeze 表示丢弃该 chunk 的 TTT write commit，但其它 chunk 正常。

#### 组 H1-B：branch-level lesion

对 H1-A 定位出的主 chunk，测试：

```text
w0 native / freeze only
w1 native / freeze only
w2 native / freeze only
w0+w2 native / freeze
w0+w1+w2 native / freeze
```

如果当前代码不支持 per-branch chunk freeze，需要在 HMC commit 阶段增加：

```text
TTT_FREEZE_CHUNK_BRANCH_POLICY = "4:w0;5:w0;6:w0"
```

#### 组 H1-C：layer-level lesion

对主 branch 继续拆：

```text
early TTT layers only
middle TTT layers only
late TTT layers only
all except early
all except late
single high-risk layer if trace shows spike
```

### 必须记录的指标

每个 run 必须输出：

```text
overall ATE / Rot / RPE_t / RPE_r
FinalErr
YawRMSE
Sim3Scale
[100,200)
[200,250)
[200,300)
[200,400)
[300,400)
50f mean / worst
100f mean / worst
200f mean / worst
```

并新增或确认已有：

```text
per_chunk_ttt_update_norm.csv
per_chunk_ttt_memory_reldiff.csv
per_chunk_branch_update_norm.csv
per_chunk_layer_update_norm.csv
per_chunk_write_prior_stats.csv
per_chunk_Dg_stats.csv
per_chunk_semantic_group_coverage.csv
```

### 可视化

必须生成：

```text
lesion_waterfall_200_300.png
    x-axis = lesion policy
    y-axis = [200,300) ATE and overall ATE

chunk_error_timeline.png
    B0 vs lesion candidates

branch_layer_update_heatmap_chunk456.png
    layer × branch update norm / memory rel diff

trajectory_xy_focus_200_300.png
    GT / B0 / lesion candidate
```

### 假设成立标准

H1 强成立：

```text
1. 某个单 chunk 或 chunk pair 让 [200,300) 降低 >= 15m；
2. overall ATE 不超过 45m，或虽然超过 45m 但局部 effect 明显可归因；
3. branch/layer lesion 能把 effect 进一步收敛到某个 branch 或 layer group；
4. 错误没有完全转移到 [100,200) 或 [300,400)。
```

H1 部分成立：

```text
[200,300) 对多个 branch/layer 都敏感，说明是累积 state basin 问题，而非单 branch 污染。
```

H1 不成立：

```text
单 chunk / branch / layer lesion 都不能接近 freeze456 的效果，说明 freeze456 的收益来自更粗的 state reset / alignment artifact，而不是可拆分写入方向。
```

---

## H2：TTT 写入需要正负方向，而不是只做写入强度缩放

### 假设内容

之前的 soft prior / sparse / focal / explicit veto 大多只改变 token multiplier 的幅度或排名。由于 TTT replay 后存在 fast-weight normalization / Muon-like update，幅度缩放可能无法真正移除 harmful direction。TTGR 已经证明低 prior token 的负方向 replay 有机制信号，但当前负样本定义太粗。本假设认为：

> 对高风险 token 做小比例负方向 replay，比单纯少写更可能清理 TTT 中的姿态 / endpoint 污染；但负 replay 必须严格限制在正确的 branch、layer 和 token group。

### 实验设计

本阶段不引入语义 group，只用现有 `D_g / stage_d / explicit_dyn / old_dyn / C_unc` 定义 negative sets。

#### 组 H2-A：TTGR fine sweep

围绕当前 `TTGR_03 = w0 all gamma 0.05`：

```text
TTGR_FINE_01: w0 all, gamma=0.025, negative=low_prior
TTGR_FINE_02: w0 all, gamma=0.050, negative=low_prior repeat
TTGR_FINE_03: w0 all, gamma=0.075, negative=low_prior
TTGR_FINE_04: w0 late, gamma=0.050, negative=low_prior repeat
TTGR_FINE_05: w0 middle+late, gamma=0.050, negative=low_prior
```

#### 组 H2-B：hard negative mass control

当前 `low_prior` 约 5.3% token 负向。需要显式控制负样本比例：

```text
TTGR_HARD_01: w0 all, gamma=0.05, hard negative bottom 2.5%
TTGR_HARD_02: w0 all, gamma=0.05, hard negative bottom 5.0%
TTGR_HARD_03: w0 all, gamma=0.05, hard negative bottom 7.5%
TTGR_HARD_04: w0 all, gamma=0.075, hard negative bottom 5.0%
```

#### 组 H2-C：cue-defined negative sets

用风险 cue 定义 negative，而不是只按低 prior：

```text
NEG_DG_HIGH:
    D_g > q90(D_g)

NEG_EXP_HIGH:
    explicit_dyn > q90(explicit_dyn)

NEG_DG_EXP_INTER:
    D_g high AND explicit_dyn high

NEG_DG_UNC_INTER:
    D_g high AND C_unc high

NEG_OLD_DG_DISAGREE:
    old_dyn high AND D_g low, diagnostic only
```

每个 negative set 先只跑：

```text
branch = w0
layer = all
gamma = 0.05
positive score = stage_d * sqrt(1-D_g)
```

### 公式定义

一般写成：

$$
R_i^{neg}=\mathbf{1}[i \in \mathcal{N}_{neg}]
$$

$$
S_i^{pos}=S_{stageD,i}\sqrt{1-D_{g,i}}
$$

$$
M_i=S_i^{pos}-\gamma R_i^{neg}
$$

TTT replay 中使用 $M_i$ 作为 signed multiplier。

### 必须记录的指标

除了标准 trajectory，还必须记录：

```text
ttt_negative_mass
ttt_negative_multiplier_min / p10 / mean
ttt_positive_mass
ttt_neutral_mass
negative_mass_by_chunk
negative_mass_by_branch
negative_mass_by_layer
negative_mass_in_chunks_4_5_6
negative_mass_in_frames_200_300
```

TTT update 方向指标：

```text
update_norm_pos
update_norm_neg
update_norm_total
cosine(pos_update, native_update)
cosine(neg_update, native_update)
cosine(pos_update, neg_update)
memory_reldiff_after_commit
```

### 可视化

```text
negative_mask_gallery_chunk456/
    RGB
    D_g
    explicit_dyn
    old_dyn
    C_unc
    negative_mask
    positive_mask
    semantic overlay if available

signed_multiplier_histogram.png
branch_layer_signed_update_heatmap.png
segment_ATE_vs_negative_mass_scatter.png
```

### 假设成立标准

H2 强成立：

```text
1. ATE <= 36.20，或 [200,300) 下降 >= 5m 且 overall ATE 不回退超过 0.10m；
2. Rot <= 6.20 且 FinalErr <= 4.0；
3. negative mass 不超过 10%，且主要集中在 chunks 4/5/6 或 high-risk frames；
4. 对比 positive-only baseline，有明确额外收益。
```

H2 弱成立：

```text
ATE <= P3G_01 + 0.03m，且 Rot / FinalErr / Yaw 至少两项优于 P3G_01。
```

H2 不成立：

```text
所有 negative replay 只改善 Rot / FinalErr，却不改善 ATE 或 [200,300)。
```

---

## H3：所有 cue 都应先按写入角色分类，而不是直接相乘

### 假设内容

现在手头有很多 cue，但它们语义不同：

```text
D_g:
    read-path harmful support cue

explicit_dyn / old_dyn:
    geometry inconsistency / dynamic residual cue

C_anchor / G_write_geo:
    geometry write eligibility / anchor cue

C_unc / C_occ:
    risk / trust cue

semantic groups:
    value / role cue

TTT residual:
    update-needed diagnostic cue
```

本假设认为：

> Cue 组合必须先映射到 positive、neutral、negative 三类写入角色；直接把所有 cue 相乘或相加，会把不同错误空间混在一起，导致 old/explicit routing 那样的回退。

### 实验设计

本阶段先不使用 negative replay，只验证 positive score 是否更好。目的是找到稳定的正写入资格。

#### 正证据 score 候选

```text
POS_00: stage_d
POS_01: stage_d * sqrt(1-D_g)
POS_02: G_write_geo
POS_03: C_anchor
POS_04: stage_d * sqrt(1-D_g) * C_anchor
POS_05: stage_d * sqrt(1-D_g) * sqrt(1-C_unc)
POS_06: stage_d * sqrt(1-D_g) * sqrt(1-explicit_dyn)
POS_07: stage_d * sqrt(1-D_g) * sqrt(1-D_old)
POS_08: stage_d * consensus_static
```

其中：

$$
consensus\_static_i=(1-D_{g,i})(1-E_i)(1-C_{unc,i})
$$

或温和版本：

$$
consensus\_static_i=\sqrt{1-D_{g,i}}\sqrt{1-E_i}\sqrt{1-C_{unc,i}}
$$

#### 固定策略

```text
branch = w0
layer = all
negative replay = off
WRITE_ALPHA = 0.125
read = C23 pair/all beta 4.75
SWA = fixed SWKS3
```

### 必须记录的指标

```text
write_score_mean / p10 / p50 / p90
corr(score, D_g)
corr(score, explicit_dyn)
corr(score, old_dyn)
corr(score, C_anchor)
corr(score, C_unc)
score_mass_by_chunk
score_mass_in_chunks_4_5_6
score_mass_by_semantic_group if Stage C cache available
```

### 可视化

```text
write_score_map_gallery/
    RGB
    stage_d
    D_g
    explicit_dyn
    C_anchor
    C_unc
    candidate_score
    candidate_score - baseline_score

score_correlation_matrix.png
positive_score_vs_segment_delta_scatter.png
```

### 假设成立标准

H3 成立：

```text
某个 positive score 相比 POS_01 至少满足：
    ATE 改善 >= 0.05m
    或 [200,300) 改善 >= 2m
    且 Rot / FinalErr 不恶化。
```

H3 不成立：

```text
所有组合 score 都不如 stage_d*sqrt(1-D_g)，说明正写入资格已基本触顶，下一步应转向 negative direction / semantic group routing。
```

---

## H4：语义不是单 scalar value，而是 group-specific TTT 写入角色

### 假设内容

v6 语义实验已经表明：

```text
semantic_value direct write 失败；
stage_d * V_sem * sqrt(1-D_g) 只有 tiny gain；
LOW_VALUE_STUFF=0.70 明显优于 0.40；
sky / vegetation / grass 不能 hard veto。
```

本假设认为：

> 语义 prior 的正确用法不是一个全局 `V_sem` scalar，而是把不同 semantic group 分配为 positive / neutral / conditional negative，并作用到不同 branch / layer。

### 需要的代码修改

当前 `SemanticPriorGenerator` 已经计算 `v_sem`、`A_patch_flat`、`A_tok`、`Elig_pix` 和 `B_chunk_geo`，但 group-specific TTT 写入还需要额外输出 patch-level semantic group maps。

建议扩展 `PriorOutput`：

```text
S_group_patch: [num_patch, num_groups]
S_structure_patch: [num_patch]
S_lowstuff_patch: [num_patch]
S_movable_patch: [num_patch]
S_uncertain_patch: [num_patch]
S_sky_patch: [num_patch] optional
S_vegetation_patch: [num_patch] optional
S_road_patch: [num_patch] optional
S_building_patch: [num_patch] optional
mask_trust_patch: [num_patch]
semantic_coverage_patch: [num_patch]
```

如果不想立刻改 dataclass，可先把这些放入 `PriorOutput.debug`，但 HMC write source 需要能读取它们。

同时新增 HMC write source：

```text
stage_d_x_sem_x_dg_inv_sqrt
structure_boost_x_dg_inv_sqrt
lowstuff_conditional_veto
semantic_posneg
semantic_structure_pos_lowstuff_highD_neg
```

### 语义分组角色

本轮先用 Mask2Former cache 中已经有覆盖的 groups：

```text
STRUCTURE_ANCHOR:
    road / building / fence / wall 等
    默认 positive candidate

LOW_VALUE_STUFF:
    sky / vegetation / grass 等
    默认 neutral / soft low value
    只有 high D_g 或 high uncertainty 时进入 conditional negative

MOVABLE_THING:
    当前 cache 覆盖为 0，暂不作为主实验
    后续若接入 car/person masklets 再测

UNCERTAIN_REGION:
    trust low，默认 neutral or weak negative
```

### 实验设计

#### 组 H4-A：semantic positive-only

```text
SEM_POS_01:
    S = stage_d * sqrt(1-D_g) * (1 + 0.10 * I_structure)

SEM_POS_02:
    S = stage_d * sqrt(1-D_g) * (1 + 0.25 * I_structure)

SEM_POS_03:
    S = stage_d * sqrt(1-D_g) * (1 + 0.10 * I_structure) * sqrt(1-C_unc)
```

#### 组 H4-B：lowstuff soft neutral / veto

```text
SEM_LS_01:
    lowstuff value = 0.85

SEM_LS_02:
    lowstuff value = 0.70

SEM_LS_03:
    lowstuff value = 0.55

SEM_LS_04:
    sky neutral, vegetation value = 0.70

SEM_LS_05:
    sky value = 0.85, vegetation value = 0.60
```

不要再试 `LOW_VALUE_STUFF=0.40` 作为主线，因为它已经显示过度抑制。

#### 组 H4-C：semantic conditional negative replay

```text
SEM_NEG_01:
    positive = structure AND low D_g
    negative = lowstuff AND high D_g
    branch = w0 all
    gamma = 0.05

SEM_NEG_02:
    positive = structure AND low D_g
    negative = vegetation/grass AND high D_g
    branch = w0 all
    gamma = 0.05

SEM_NEG_03:
    positive = structure AND low D_g
    negative = sky AND high D_g
    branch = w0 all
    gamma = 0.025

SEM_NEG_04:
    positive = structure AND low D_g AND low C_unc
    negative = lowstuff AND high D_g AND high C_unc
    branch = w0 all
    gamma = 0.05
```

注意：`sky` 单独负向要非常谨慎，初始 `gamma=0.025`，因为天空 / horizon 可能参与 scale / yaw continuity。

### 公式

语义 positive mask：

$$
P_i^{struct}=I_i^{structure}(1-D_{g,i})(1-C_{unc,i})
$$

语义 conditional negative mask：

$$
R_i^{lowstuff}=I_i^{lowstuff}D_{g,i}C_{unc,i}
$$

最终 signed multiplier：

$$
M_i=S_{stageD,i}\sqrt{1-D_{g,i}}(1+\alpha P_i^{struct})-\gamma R_i^{lowstuff}
$$

### 必须记录的指标

按 semantic group 记录：

```text
coverage_by_group
write_score_mean_by_group
positive_mass_by_group
negative_mass_by_group
update_norm_by_group
update_cosine_to_native_by_group
branch0_update_norm_by_group
branch1_update_norm_by_group
branch2_update_norm_by_group
chunks_4_5_6_group_mass
frames_200_300_group_mass
```

### 可视化

```text
semantic_write_gallery_chunk456/
    RGB
    semantic group overlay
    D_g
    C_unc
    positive mask
    negative mask
    signed multiplier
    write score

semantic_group_update_heatmap.png
    rows = semantic groups
    columns = branch/layer
    values = update norm / negative mass

sky_vegetation_case_study.md
    对 sky/tree/vegetation 被抑制前后的轨迹与 cue map 对比
```

### 假设成立标准

H4 强成立：

```text
1. SEM_NEG 或 SEM_POS 使 ATE <= 36.20，或 [200,300) 下降 >= 5m；
2. sky / vegetation 不出现大面积 hard negative；
3. structure group 的 positive update cosine 与 native/static update 同向；
4. lowstuff high-D 的 negative update 主要集中在 chunks 4/5/6 或 high-error frames；
5. 不把错误转移到 [100,200)。
```

H4 弱成立：

```text
ATE <= P3G_01 + 0.03m，且 Rot / FinalErr / Yaw 至少两项更好。
```

H4 不成立：

```text
semantic group routing 仍只有 tiny change，且不能解释 [200,300)。此时语义暂时降级为 diagnostic，不作为 TTT write 主线。
```

---

## H5：TTT branch 和 layer 决定 negative replay 是否安全

### 假设内容

v6 TTGR 显示 `w0 all gamma=0.05` 是 ATE tiny best，`w0+w2 late` rotation 和 `[200,300)` 更好但 ATE 回退。这说明 branch / layer scope 对正负 replay 非常敏感。

本假设认为：

> negative replay 不是所有 branch/layer 都安全。`w0` 更适合全层或较宽层范围，`w2` 可能只适合局部晚层，`w1` 初期不应直接参与。

### 实验设计

只对 H2/H4 的 top 2 signed policy 做 branch/layer sweep。

#### Branch sweep

```text
BR_01: w0 only
BR_02: w2 only
BR_03: w0+w2
BR_04: w1 diagnostic, gamma=0.025 only
BR_05: w0+w1+w2 diagnostic only if previous safe
```

#### Layer sweep

```text
LY_01: all
LY_02: late
LY_03: middle+late
LY_04: early only diagnostic
LY_05: no early
LY_06: lesion-identified layers from H1
```

### 必须记录的指标

```text
branch_update_norm
branch_signed_update_norm
branch_negative_mass
layer_update_norm
layer_negative_mass
layer_memory_reldiff
branch_layer_cosine_to_B0
branch_layer_cosine_to_TTGR03
```

### 可视化

```text
branch_layer_policy_matrix.md
branch_layer_ATE_heatmap.png
branch_layer_Rot_heatmap.png
branch_layer_segment200300_heatmap.png
branch_layer_update_norm_heatmap.png
```

### 假设成立标准

H5 成立：

```text
存在明确 branch/layer sweet spot，使 top policy 相比 all-layer w0：
    ATE 改善 >= 0.05m
    或 [200,300) 改善 >= 3m
    且 Rot/FinalErr 不恶化。
```

H5 不成立：

```text
所有 branch/layer 只形成 Rot/FinalErr trade-off，不影响 ATE 和 [200,300)。
```

---

## H6：localized / risk-triggered TTT write policy 比全局策略更可能修 `[200,300)`

### 假设内容

`[200,300)` 病灶高度集中，全局 TTGR 或全局 semantic routing 只能带来 tiny best。因此本假设认为：

> TTT 写入策略应该由 chunk risk 触发，只在高风险 chunk / layer / branch 上使用负 replay 或 semantic routing；其它 chunk 保持 B0 / current best，以保留全局 continuity。

### 实验设计

#### 组 H6-A：manual localized diagnostic

先基于 H1 定位结果做手工局部策略：

```text
LOC_01: TTGR only at chunk 4
LOC_02: TTGR only at chunk 5
LOC_03: TTGR only at chunk 6
LOC_04: TTGR only at chunks 4+5
LOC_05: TTGR only at chunks 5+6
LOC_06: TTGR only at chunks 4+5+6
```

每个 run 固定使用 H2/H4 top policy 的 branch/layer/gamma。

#### 组 H6-B：automatic risk-triggered policy

定义 chunk risk：

$$
R_m=w_1\,q90(D_g)_m+w_2\,q90(C_{unc})_m+w_3\,\Delta W_m^{norm}+w_4\,NegMass_m+w_5\,SemRisk_m
$$

其中：

```text
D_g: read cue risk
C_unc: uncertainty risk
Delta W norm: TTT update norm spike
NegMass: negative replay mass from passive signed policy
SemRisk: lowstuff high-D coverage or semantic uncertainty
```

第一版权重不要学习，先用 normalized z-score mean：

$$
R_m=\frac{1}{K}\sum_k z_{m,k}
$$

触发规则：

```text
trigger if R_m > median(R) + 1.0 * MAD(R)
```

或：

```text
trigger top-3 risk chunks
```

### 必须记录的指标

```text
chunk_risk_score.csv
triggered_chunks
trigger_reason_breakdown
triggered_negative_mass
triggered_update_norm
triggered_semantic_group_mass
```

### 可视化

```text
chunk_risk_timeline.png
triggered_chunk_gallery.md
manual_vs_auto_trigger_table.md
```

### 假设成立标准

H6 强成立：

```text
automatic trigger 选中 chunks 4/5/6 或其邻域，且：
    [200,300) 下降 >= 5m
    overall ATE 不回退超过 0.10m
    [100,200) 不成为新的 worst segment。
```

H6 弱成立：

```text
manual trigger 有效，但 automatic trigger 还不稳定。此时继续改 risk score，不直接进 cross-seq。
```

H6 不成立：

```text
manual 和 automatic trigger 都不能影响 [200,300)，说明病灶可能不是可局部触发的写入污染。
```

---

## H7：absolute write budget 与 mean-preserving token prior 需要重新区分

### 假设内容

旧写入控制常用 mean-preserving prior，使 token prior 的绝对均值被部分消掉。Semantic Prior Generator v2 明确指出应保留 token prior 绝对意义，并把 token-level gate 与 chunk-level budget 分离。本假设认为：

> 如果整段 chunk 语义/几何都低价值，TTT 应真的少写；但如果只是局部 high-risk，则保持总写入量并重排 token ranking 更安全。

### 实验设计

只对 H2/H4 top policy 做三种 normalization / budget 对照。

#### 模式 A：relative-only

$$
\gamma_i^{rel}=\frac{\eta_iS_i}{\operatorname{Mean}_j(\eta_jS_j)+\epsilon}
$$

#### 模式 B：absolute token gate

$$
\gamma_i^{abs}=\frac{\eta_iS_i}{\operatorname{Mean}_j(\eta_j)+\epsilon}
$$

#### 模式 C：absolute token gate + chunk budget

$$
B_m=\operatorname{Mean}_i(S_i)
$$

$$
\lambda_m=\lambda_{min}+(\lambda_{max}-\lambda_{min})B_m
$$

$$
\Delta W_m=\lambda_m\sum_i\gamma_i^{abs}J_i
$$

### 必须记录的指标

```text
mean_token_prior
chunk_budget_B
lambda_chunk
effective_update_norm
relative_update_norm_vs_B0
per_chunk_write_magnitude
```

### 假设成立标准

H7 成立：

```text
absolute / budget 模式明显改善 [200,300) 或 ATE，且不损害 continuity。
```

H7 不成立：

```text
absolute budget 总是损害 ATE，说明 LoGeR TTT 需要稳定总写入量，只能改 direction / ranking。
```

---

## H8：semantic cache / Stage D 必须继续保持 no-op reproducibility

### 假设内容

v6 已经证明 Stage C inline compute 会扰动 full-sequence parity，cache require-hit 是必要策略。本轮语义实验更多，必须确保每个 semantic run 的变化来自 semantic prior，而不是 Stage C 前端重算或 HMC no-op 破坏。

### 实验设计

每批 semantic TTT 之前都要跑或复用：

```text
SEM_NOOP_00:
    Stage C cache read require-hit
    semantic_prior_mode = noop
    hmc_ignore_semantic_prior = 1

SEM_PASS_00:
    Stage C cache read require-hit
    semantic_prior_mode = pass_through
    HMC consumed but no-op
```

与 B0 比较：

```text
ATE diff <= 0.01m
Rot diff <= 0.01deg
pose max diff = 0 if possible
hash_H_next identical or expected
```

### 假设成立标准

H8 是 hard gate：不通过则停止所有 semantic TTT run。

---

## H9：跨序列验证只给真正候选，不给 tiny best

### 假设内容

当前在 KITTI01 上已经有很多 `0.005m-0.05m` tiny movement。它们不值得立刻跑跨序列。只有两类候选值得 cross-seq：

```text
1. KITTI01 ATE 有 >=0.20m 实质改进；
2. [200,300) 有 >=5m segment 改进且 overall 不崩。
```

### 实验设计

通过 gate 后跑：

```text
KITTI00 full
KITTI02 full
KITTI05 full
```

对照：

```text
B0_SWKS3
TTGR_03
P3G_01
new candidate
```

### 假设成立标准

候选晋级：

```text
1. 00/02/05 平均 ATE 不差于 B0；
2. 没有任一序列 ATE regression > 3%；
3. KITTI02 不出现明显退化；
4. Rot / FinalErr 至少不显著恶化。
```

---

## 4. 必须记录的指标总表

### 4.1 全局 trajectory 指标

每个 full run 必须记录：

```text
ATE_RMSE
Rot_RMSE
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
ATE_50_mean / ATE_50_worst
ATE_100_mean / ATE_100_worst
ATE_200_mean / ATE_200_worst
```

### 4.2 重点 segment 指标

必须显式记录：

```text
ATE_[0,100)
ATE_[100,200)
ATE_[200,250)
ATE_[200,300)
ATE_[200,400)
ATE_[300,400)
ATE_[400,600)
```

如果某个 run full ATE 改善，但 `[200,300)` 不动，需要标记为：

```text
global tiny improvement, not disease-fixing
```

### 4.3 TTT 写入指标

每个 chunk / layer / branch 记录：

```text
update_norm
semantic_update_norm
native_update_norm
candidate_update_norm
memory_reldiff
cosine_to_native
cosine_to_B0
cosine_pos_neg
positive_mass
neutral_mass
negative_mass
signed_multiplier_mean / min / p10 / p50 / p90
```

### 4.4 Cue / semantic group 指标

```text
D_g_mean / q90 / mass_gt_0.5
explicit_dyn_mean / q90 / mass_gt_0.5
old_dyn_mean / q90 / mass_gt_0.5
C_unc_mean / q90
C_anchor_mean / q90
G_write_geo_mean
semantic_coverage
structure_coverage
lowstuff_coverage
sky_coverage
vegetation_coverage
road_coverage
building_coverage
mask_trust_mean
```

### 4.5 Group-level update attribution

如果启用语义：

```text
update_norm_by_group
negative_mass_by_group
positive_mass_by_group
write_score_mean_by_group
cosine_to_native_by_group
branch0_update_by_group
branch2_update_by_group
chunks456_group_update
frames200300_group_update
```

---

## 5. 必须输出的可视化

本轮不能只看表格。每个候选至少输出以下可视化。

### 5.1 `[200,300)` 病灶 dashboard

```text
focus_200_300_dashboard.md
```

包含：

```text
trajectory XY: GT / B0 / candidate
per-frame position error
per-chunk ATE bars
[200,300) zoom-in
Sim3 scale comparison
Yaw over time
```

### 5.2 TTT branch-layer heatmap

```text
branch_layer_update_heatmap.png
branch_layer_negative_mass_heatmap.png
branch_layer_cosine_heatmap.png
```

横轴为 branch，纵轴为 layer，分别显示 update norm、negative mass、cosine to native。

### 5.3 Signed multiplier gallery

对 chunks 4/5/6 和 frames `[200,300)` 的代表帧输出：

```text
RGB
D_g
explicit_dyn
C_unc
semantic overlay
positive mask
negative mask
signed multiplier
write score
```

### 5.4 Semantic group attribution

```text
semantic_group_write_dashboard.md
```

包含：

```text
group coverage bar
group write score bar
group positive / negative mass
group update norm
group effect in chunks 4/5/6
group effect in [200,300)
```

### 5.5 Lesion waterfall

```text
lesion_waterfall.png
```

同时显示：

```text
overall ATE
[200,300) ATE
FinalErr
YawRMSE
```

---

## 6. 实验执行顺序

本轮按下面顺序执行，不能跳过 hard gate。

---

## Phase 0：Reproducibility 与 semantic cache no-op hard gate

### 目标

确保接下来所有 TTT 写入实验都从可复现 baseline 开始。

### Runs

```text
V7_P0_B0_repeat_SWKS3
V7_P0_semantic_noop_cache_read
V7_P0_pass_through_consumed
```

### 通过标准

```text
B0 repeat:
    |ATE - 36.4161| <= 0.03m
    |Rot - 6.6128| <= 0.03deg

semantic noop:
    |ATE - B0_ATE| <= 0.01m
    |Rot - B0_Rot| <= 0.01deg
    pose max diff = 0 or documented tiny diff
```

不通过则先修工程，不进入 Phase 1。

---

## Phase 1：`[200,300)` causal lesion refinement

### 目标

把 freeze456 的粗因果效果拆到 chunk / branch / layer。

### 第一批 runs

```text
V7_H1_freeze4
V7_H1_freeze5
V7_H1_freeze6
V7_H1_freeze45
V7_H1_freeze56
V7_H1_freeze46
V7_H1_freeze456_repeat
```

### 第二批 runs

基于第一批结果选择最有影响的 chunk，做：

```text
V7_H1_branch_w0
V7_H1_branch_w1_diag
V7_H1_branch_w2
V7_H1_branch_w0w2
```

### 第三批 runs

基于 branch 结果做 layer：

```text
V7_H1_layer_early
V7_H1_layer_middle
V7_H1_layer_late
V7_H1_layer_middle_late
V7_H1_layer_noearly
```

### 决策

如果定位出 causal scope，则 Phase 2/5 的 TTGR 只优先作用该 scope。如果定位不清，则 Phase 2 继续使用 `w0 all` 作为默认。

---

## Phase 2：TTGR fine / hard negative / cue negative 小矩阵

### 目标

确认“负方向写入”是否比单纯少写更有效，并确定 negative set。

### Runs

```text
V7_H2_TTGR_w0all_g0025_lowprior
V7_H2_TTGR_w0all_g0050_lowprior_repeat
V7_H2_TTGR_w0all_g0075_lowprior
V7_H2_TTGR_w0midlate_g0050_lowprior

V7_H2_HARD_w0all_g0050_bot025
V7_H2_HARD_w0all_g0050_bot050
V7_H2_HARD_w0all_g0050_bot075

V7_H2_NEG_DG_high
V7_H2_NEG_EXP_high
V7_H2_NEG_DG_EXP_inter
V7_H2_NEG_DG_UNC_inter
```

### 晋级标准

进入 Phase 5/6 的候选最多 2 个，必须满足：

```text
ATE <= 36.35
或 [200,300) 下降 >= 3m 且 overall 不回退超过 0.10m
或 Rot/FinalErr/Yaw 三项显著优于 P3G_01 且 ATE 不差于 P3G_01 + 0.03m
```

---

## Phase 3：positive write score 与 cue 组合角色验证

### 目标

验证所有现有 cue 是否能构造更好的正写入资格。

### Runs

```text
V7_H3_POS_stageD_repeat
V7_H3_POS_stageD_dg_sqrt_repeat
V7_H3_POS_GwriteGeo
V7_H3_POS_Canchor
V7_H3_POS_stageD_dg_Canchor
V7_H3_POS_stageD_dg_unc
V7_H3_POS_stageD_dg_exp
V7_H3_POS_stageD_consensus_static
```

### 晋级标准

```text
ATE <= 36.35
或 [200,300) 下降 >= 2m
或作为 Phase 4 semantic positive base 明显优于 stageD_dg_sqrt。
```

如果全部失败，后续不再扩大 positive-only cue combination，把重心放到 negative / semantic group routing。

---

## Phase 4：semantic group-specific TTT write

### 目标

把语义从 scalar value 改为 group-specific write role。

### 工程前置

必须先导出 patch-level semantic group maps。若未实现，Phase 4 不跑 full。

### Runs

```text
V7_H4_SEM_structure_boost_a010
V7_H4_SEM_structure_boost_a025
V7_H4_SEM_structure_unc_a010

V7_H4_SEM_lowstuff_085
V7_H4_SEM_lowstuff_070_repeat
V7_H4_SEM_sky_neutral_veg070
V7_H4_SEM_sky085_veg060

V7_H4_SEM_NEG_lowstuff_highD_g005
V7_H4_SEM_NEG_vegetation_highD_g005
V7_H4_SEM_NEG_sky_highD_g0025
V7_H4_SEM_POSNEG_structure_lowD_lowstuff_highD
```

### 晋级标准

```text
strong:
    ATE <= 36.20
    或 [200,300) 下降 >= 5m

weak:
    ATE <= P3G_01 + 0.03m
    且 Rot / FinalErr / Yaw 至少两项改善
```

如果语义只带来 `<=0.01m` tiny movement，停止继续扫 `LOW_VALUE_STUFF` 标量，转向 group attribution / cache quality。

---

## Phase 5：branch / layer policy for top signed candidate

### 目标

确定负 replay / semantic posneg 的安全 branch/layer。

### Runs

对 Phase 2/4 top candidate 做：

```text
V7_H5_BR_w0_all
V7_H5_BR_w0_late
V7_H5_BR_w0_middle_late
V7_H5_BR_w2_late
V7_H5_BR_w0w2_late
V7_H5_BR_w1_diag_g0025

V7_H5_LAYER_early
V7_H5_LAYER_middle
V7_H5_LAYER_late
V7_H5_LAYER_noearly
V7_H5_LAYER_lesion_scope
```

### 晋级标准

```text
ATE 比 top candidate 再改善 >= 0.05m
或 [200,300) 再下降 >= 3m
且不损害 FinalErr / Yaw。
```

---

## Phase 6：localized / risk-triggered TTT write

### 目标

用局部触发策略修 `[200,300)`，避免全局 negative replay 伤 continuity。

### Runs

```text
V7_H6_LOC_chunk4
V7_H6_LOC_chunk5
V7_H6_LOC_chunk6
V7_H6_LOC_chunk45
V7_H6_LOC_chunk56
V7_H6_LOC_chunk456

V7_H6_AUTO_top3risk
V7_H6_AUTO_mad1
V7_H6_AUTO_mad15
```

### 晋级标准

```text
[200,300) 下降 >= 5m
且 overall ATE 不回退超过 0.10m
且 [100,200) 没有成为新 worst。
```

如果 manual 有效但 auto 无效，下一轮只优化 risk score；不要手动 chunk 策略进主线。

---

## Phase 7：absolute budget / normalization 对照

### 目标

确认 TTT 需要稳定总写入量，还是 chunk-level budget 可以帮助。

### Runs

对 top 1 candidate 做：

```text
V7_H7_relative_only
V7_H7_absolute_token_gate
V7_H7_absolute_token_gate_chunk_budget
V7_H7_budget_lambda_0p7_1p2
V7_H7_budget_lambda_0p5_1p5
```

### 晋级标准

```text
budget 版本 ATE / [200,300) 明显优于 relative-only，且 update norm 不出现异常尖峰。
```

---

## Phase 8：cross-sequence sanity

### 触发条件

只有满足以下任一条件才跑：

```text
KITTI01 ATE <= 36.20
或 [200,300) 下降 >= 5m 且 overall ATE 不回退
或 relative best 对 P3G_01 改善 >= 0.20m
```

### Runs

```text
KITTI00 full
KITTI02 full
KITTI05 full
```

对照：

```text
B0_SWKS3
TTGR_03
P3G_01
new candidate
```

---

## 7. 资源与调度策略

### 7.1 并发

```text
KITTI01 full:
    4-6 并发可用；优先 4 并发保证复现稳定。

semantic cache / Mask2Former:
    只允许离线预计算；benchmark 中 require-hit read。

KITTI00/02 long full:
    2 并发优先，避免 host RAM / swap 干扰。
```

### 7.2 每批最大 run 数

```text
Phase 1 lesion:
    7 + selected branch/layer，不超过 16 full runs

Phase 2 TTGR:
    第一批不超过 12 full runs

Phase 3 positive score:
    不超过 8 full runs

Phase 4 semantic:
    每轮不超过 12 full runs，必须先看 passive / no-op

Phase 5 branch/layer:
    只对 top 2 candidate，最多 12 full runs

Phase 6 localized:
    先 manual 6 runs，再 auto 3 runs
```

---

## 8. 结果汇报模板

每一批实验报告必须包含以下结构。

```text
1. 本批目标
2. 固定协议
3. 变量矩阵
4. benchmark table
5. trajectory diagnostics
6. [200,300) focus table
7. TTT update attribution
8. semantic group attribution if applicable
9. 可视化索引
10. 是否通过假设 gate
11. 下一步决策
```

### 8.1 最小 benchmark table 字段

```text
Run
Policy
Branch
Layer
Gamma
Negative source
Semantic mode
ATE
Rot
RPE_t
RPE_r
FinalErr
[200,300)
[200,400)
YawRMSE
Sim3Scale
Conclusion
```

### 8.2 最小 TTT attribution table 字段

```text
Run
Chunk
Branch
LayerGroup
UpdateNorm
MemoryRelDiff
PositiveMass
NegativeMass
NegativeMin
CosPosNative
CosNegNative
CosPosNeg
SemanticGroupTop
```

---

## 9. 停止规则

为了避免继续在平台里小调，本轮设置停止规则。

### 9.1 停止某类 negative replay

如果同一 negative source 的 3 个 gamma / mass 都满足：

```text
ATE >= B0 + 0.10m
且 [200,300) 改善 < 1m
```

则停止该 negative source。

### 9.2 停止 semantic scalar sweep

如果 `LOW_VALUE_STUFF` 或 `V_sem` 标量 sweep 只产生 `<=0.02m` 波动，停止标量 sweep，只保留 group routing。

### 9.3 停止 positive-only cue combination

如果 Phase 3 所有正证据 score 都不超过 `stage_d*sqrt(1-D_g)`，停止继续乘更多 cue。

### 9.4 停止全局 TTT policy

如果全局 signed replay 无法让 `[200,300)` 下降超过 `2m`，但 localized policy 有信号，则停止全局策略，转向 risk-trigger。

---

## 10. 我对第一批最推荐的实际执行顺序

为了尽快得到有用结论，我建议第一周先跑下面这些。

### Day 1：Phase 0 + H1 chunk lesion

```text
V7_P0_B0_repeat_SWKS3
V7_P0_semantic_noop_cache_read
V7_H1_freeze4
V7_H1_freeze5
V7_H1_freeze6
V7_H1_freeze45
V7_H1_freeze56
V7_H1_freeze456_repeat
```

目标：确认病灶入口到底是不是单 chunk / chunk pair。

### Day 2：H2 TTGR fine + hard negative

```text
V7_H2_TTGR_w0all_g0025_lowprior
V7_H2_TTGR_w0all_g0050_lowprior_repeat
V7_H2_TTGR_w0all_g0075_lowprior
V7_H2_HARD_w0all_g0050_bot025
V7_H2_HARD_w0all_g0050_bot050
V7_H2_HARD_w0all_g0050_bot075
```

目标：确认当前 TTGR tiny best 是否有真实 sweet spot。

### Day 3：H2 cue-defined negative + H3 positive cue

```text
V7_H2_NEG_DG_high
V7_H2_NEG_EXP_high
V7_H2_NEG_DG_EXP_inter
V7_H2_NEG_DG_UNC_inter

V7_H3_POS_stageD_dg_Canchor
V7_H3_POS_stageD_dg_unc
V7_H3_POS_stageD_consensus_static
```

目标：判断负样本是否必须是 cue consensus，正写入是否能由 anchor / uncertainty 改善。

### Day 4：semantic group instrumentation + passive audit

不先跑 full 指标，先确认输出：

```text
semantic group patch maps
semantic coverage by chunk
structure/lowstuff/sky/vegetation maps
chunks 4/5/6 group coverage
[200,300) group coverage
```

目标：确保 Phase 4 的 semantic TTT 不是盲跑。

### Day 5：H4 semantic posneg 第一批

```text
V7_H4_SEM_structure_boost_a010
V7_H4_SEM_lowstuff_070_repeat
V7_H4_SEM_NEG_lowstuff_highD_g005
V7_H4_SEM_NEG_vegetation_highD_g005
V7_H4_SEM_NEG_sky_highD_g0025
V7_H4_SEM_POSNEG_structure_lowD_lowstuff_highD
```

目标：验证语义是否能提供真正的 TTT positive / negative role。

---

## 11. 最终判断逻辑

本轮结束后，应能给出以下四种结论之一。

### 结论类型 A：TTT negative replay 找到主病灶解法

条件：

```text
ATE <= 36.20 或 [200,300) 降 >= 5m，且整体不崩。
```

下一步：branch/layer refine + cross-seq。

### 结论类型 B：语义 group routing 是关键

条件：

```text
semantic posneg 明显优于 non-semantic TTGR，且 group attribution 可解释。
```

下一步：扩展 semantic cache 类别，加入 car/person/movable thing，做 cross-seq。

### 结论类型 C：TTT 只做姿态 regularizer，不修 ATE 主病灶

条件：

```text
所有 TTT 写入策略都只改善 Rot/FinalErr/Yaw，不动 [200,300)。
```

下一步：回到 read cue / per-head / model internal failure diagnosis；TTT 写入只保留 balanced variant。

### 结论类型 D：病灶不是 token-level write policy 可解

条件：

```text
lesion 有效，但任何 branch/layer/token/group policy 都不能复现 lesion 的局部收益。
```

下一步：考虑更大机制修改，例如 TTT replay objective、state reset alignment、chunk merge / Sim3 failure、或者把 long-term TTT memory 分成 static 与 transient 两套状态。

---

## 12. 本轮一句话总结

本轮要验证的核心不是“哪些 token 少写”，而是：

> **哪些 token / semantic group / cue consensus 应该产生正向长期记忆，哪些 token 只能短期参与当前 chunk，哪些 token 应该以负方向抵消或被局部触发式遗忘。**

只有把 TTT 写入从 scalar gate 升级到 positive / neutral / negative evidence replay，并把 `[200,300)` 病灶作为硬诊断目标，才有可能判断语义 prior 是否真的适合 TTT 长期写入。

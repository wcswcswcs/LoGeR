# ACL2 v33：Semantic Cue Local-to-Full Transfer 与 C9-Compatible Memory Strategy 实验计划

日期：2026-05-23  
目标系统：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / ACL2 semantic-conditioned C23 cue  
主开发集：KITTI01  
阶段目标：先进入 KITTI01 ATE $30m$，不再做语义规则小修小补。  
重要边界：fixed chunk-id activation、short rollout、oracle trigger、GT / offline rewrite 都只能算 diagnostic，不能算 deployable online success。

---

## 0. 本轮为什么必须换思路

v31 证明了一个重要事实：语义不是完全没用。`semantic-conditioned C23 / D_g reconditioning` 在 chunk10 的 h10/h15 short rollout 中能明显压低 $[200,300)$，并且 `SEM_Z_COARSE_BETA525` 在 h15 上也通过了局部 durability gate。

但是 v32 证明了另一个更关键的事实：这个局部信号没有成功转成可部署 full online 策略。fixed chunk10-12 activation 在 full online 中能改善 H9，但它使用了 KITTI01 chunk id，不能部署；runtime triggers 没有复现 fixed-window 效果；更严重的是，把 semantic-conditioned cue 叠到当前最好 C9 协议上，所有结果都比 C9 差。

因此，本轮的核心判断是：

> 当前不是继续扩大 semantic rule matrix，也不是继续扫 beta / threshold。真正问题是 **local-to-full transfer** 和 **C9 compatibility**。语义 reconditioning 有局部因果力，但我们还没有找到可部署触发器，也没有让它和当前最好 TTT/SWA 协议兼容。

本轮计划的重点不是“再找一种 sky / vegetation / road rule”，而是回答：

1. fixed-window 语义信号为什么有效？
2. runtime trigger 为什么抓不住？
3. C9 为什么和 semantic-conditioned C23 冲突？
4. 语义应当进入 cue 计算、read path、anchor memory、还是 trajectory-state trigger？
5. 如果语义路线要进入 ATE $30m$，需要怎样从局部 $[200,300)$ correction 变成全序列 trajectory correction？

---

## 1. 当前实验事实与边界

当前可部署 online best 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

阶段目标是：

```text
KITTI01 ATE <= 30m
```

所以当前差距是：

$$
33.7629421029 - 30.0 = 3.7629421029m
$$

v32 的关键边界：

```text
H1_FIXED_CH10_12:
    full ATE = 36.0801185755m
    delta vs H9 = -0.7390457418m
    [200,300) delta vs H9 = -1.2521234262m
    diagnostic only, because fixed chunk-id activation is not deployable

runtime triggers:
    semantic_z_high_mass delta vs H9 = +0.4207011488m
    semantic_d_mean_ema_mad delta vs H9 = +0.0552074110m
    semantic_d_q90_ema_mad delta vs H9 = +0.3083686367m

C9 interaction:
    C9 reference ATE = 33.7629421029m
    best C9 + semantic ATE = 34.2952287438m
    best delta vs C9 = +0.5322866409m

residual repair:
    C9_RESID_COARSE_ALL ATE = 34.3258261120m
    delta vs C9 = +0.5628840091m
```

这说明：

```text
1. 语义 cue reconditioning 有局部/full diagnostic signal；
2. 当前 runtime trigger 不能部署；
3. 当前语义 cue 不兼容 C9；
4. residual repair 不能解决 C9 冲突；
5. 继续 full-online threshold sweep 没意义。
```

---

## 2. 本轮整体目标

本轮的目标不是直接生成更多语义候选，而是建立一个能通向 ATE $30m$ 的因果路线。

本轮必须回答五个问题。

### 2.1 问题 A：local semantic cue 为什么不能 transfer 到 full online？

v31 的 h10/h15 局部效果很强，但 v32 full online 效果很弱。我们必须拆清楚这个 transfer gap 来自哪里：

```text
1. full online 早期 chunks 改变了 chunk10 parent state；
2. semantic z cue 只适合 chunk10-12，不适合全序列常开；
3. runtime trigger 看错了统计量；
4. C9 的 TTT/SWA state 与 semantic cue 冲突；
5. merge/gauge state 或 reset-boundary state 覆盖了局部修正。
```

### 2.2 问题 B：runtime trigger 应该预测什么？

v32 的 runtime triggers 用的是：

```text
semantic_z_high_mass
semantic_d_mean_ema_mad
semantic_d_q90_ema_mad
```

这些都失败了。说明 trigger 不应只看 semantic cue 强弱，而应预测：

> 如果在这个 chunk / reset-group 激活 semantic-conditioned C23，它是否能改善未来 h10/h15 trajectory，而不会伤害 downstream continuity？

因此 trigger 必须从 “high semantic mass detector” 升级成 “semantic cue value predictor”。

### 2.3 问题 C：C9 为什么和 semantic cue 冲突？

C9 已经有：

```text
read beta chunks
TTT tri-replay
update_conflict_energy risk
SWKS3-style SWA
mp_alpha = 0.1
```

semantic-conditioned C23 可能和这些机制冲突：

```text
1. semantic cue 改变了 read path，但 C9 的 TTT/SWA write 已按旧 read state 调好；
2. semantic cue 修 [200,300)，但把 [400,600) continuity 破坏；
3. C9 的 tri-replay 已经在 body/exit windows 处理了一部分风险，semantic cue 又重复干预；
4. semantic cue 和 SWA keep-scope / overlap replacement 的 source topology 冲突。
```

本轮必须定位冲突组件，而不是直接宣布语义不兼容 C9。

### 2.4 问题 D：语义应该进入 cue，而不是直接进入 action

v31 相比 v29C/v30 的关键变化是：语义改变了 `D_g` / C23 cue 的解释方式，而不是直接做 masklet action。这个方向比语义 action matrix 更强。

因此，本轮优先考虑：

```text
semantic-conditioned C23
semantic-conditioned support centroid
semantic-conditioned robust normalization
semantic-conditioned static anchor support
```

而不是继续：

```text
sky skip
vegetation negative
structure keep
```

### 2.5 问题 E：如何避免数据集调参？

本轮允许诊断不同数据集或序列上的 failure mode，但不允许为 KITTI01 单独调规则。

禁止：

```text
hard-code chunk10-12 as deployable trigger
KITTI01-specific threshold
KITTI01-specific label value
KITTI01-specific gamma
sequence id feature
absolute chunk id feature
```

允许：

```text
reset-relative position
current chunk memory state statistics
D_g distribution
semantic-conditioned D residual
TTT conflict
scale risk
SWA boundary risk
attention source mass
```

---

## 3. 核心假设

---

## H1：v31/v32 的 semantic cue 不是无效，而是需要 state-conditioned activation

### 假设

`semantic-conditioned C23` 在特定 state/window 下有效。它不适合全序列常开，也不能靠简单 high-mass trigger 激活。真正 trigger 应该由 memory-state、cue-residual、TTT conflict、SWA boundary risk 共同决定。

### 实验设计

对 H9 和 C9 分别做 reset-group fixed-window short/full diagnostic，用同一套 candidate：

```text
SEM_Z_COARSE_BETA525
SEM_Z_FINE
SEM_RESID_COARSE_L025
ORIG_C23
```

窗口不使用 absolute KITTI01 chunk id 作为 deployable trigger，但用于 oracle 标签生成：

```text
reset group 0: chunks 0-4
reset group 1: chunks 5-9
reset group 2: chunks 10-14
reset group 3: chunks 15-19
reset group 4: chunks 20-24
reset group 5: chunks 25-29
reset group 6: chunks 30-34
```

每个 reset group 只跑 short rollout h10/h15，不先跑 full。

### 记录指标

每个窗口记录：

```text
ATE_h10
ATE_h15
[200,300]_h10
[200,300]_h15
[400,600]_h10
[400,600]_h15
FinalErr
YawRMSE
Sim3Scale
D_orig_mean / q90
D_sem_mean / q90
DeltaD = mean(|D_sem - D_orig|)
DeltaD on structure
DeltaD on lowstuff
semantic_z_high_mass
TTT update_conflict_energy mean / q90
scale_state_risk mean / q90
SWA boundary_10f / boundary_20f
frame/global source_keep_ratio
SWA source_keep_ratio
HMC state movement norm
merge/gauge state movement norm
```

定义局部收益：

$$
G_{win}^{h} = ATE_{base}^{h} - ATE_{semantic}^{h}
$$

定义下游损伤：

$$
R_{down}^{h} = Err_{semantic}^{[400,600]} - Err_{base}^{[400,600]}
$$

### 假设成立标准

H1 通过条件：

```text
1. 至少一个 reset-relative window 在 H9 或 C9 parent 上有：
       h15 ATE delta <= -2m
       或 h15 [200,300) delta <= -4m
       且 [400,600) regression <= +1m

2. 这个 window 的特征不是 absolute chunk id，而可以由 no-GT statistics 区分：
       cue residual / TTT conflict / scale risk / boundary risk 之一明显高于邻近 windows

3. 如果只在 KITTI01 absolute chunk10-12 有效、reset-relative feature 不可区分，
   则该效果只能算 diagnostic，不进入 deployable track。
```

---

## H2：失败的 runtime trigger 是因为 trigger 预测目标错误

### 假设

v32 trigger 预测的是 “semantic cue 是否强”，不是 “semantic cue 是否会改善未来 trajectory”。因此它选错窗口。

### 实验设计

用 H1 fixed-window short rollout 生成 oracle labels：

```text
label positive:
    G_win_h10 >= 1.5m
    or [200,300] improvement >= 3m
    and downstream regression <= +1m

label negative:
    ATE regression >= +0.5m
    or [400,600] regression >= +1m
```

用 no-GT features 训练 / 拟合轻量 trigger：

```text
features:
    semantic_z_high_mass
    semantic_D_mean
    semantic_D_q90
    DeltaD_mean
    DeltaD_structure
    DeltaD_lowstuff
    D_orig_entropy
    D_sem_entropy
    TTT conflict mean/q90
    scale risk mean/q90
    SWA boundary risk
    source attention mass removed
    reset-relative chunk index within reset block
    but not absolute sequence id / absolute chunk id
```

先用简单规则和可解释模型：

```text
decision tree depth <= 3
logistic regression
rule list
```

不要直接训练复杂模型。

### 必须记录

```text
trigger_precision
trigger_recall
trigger_F1
false_positive_windows
false_negative_windows
selected_windows_per_sequence
mean_gain_selected
mean_regression_selected
feature_importance
rule_list.json
```

### 假设成立标准

```text
1. On held-out windows within KITTI01:
       precision >= 0.6
       selected-window mean ATE delta <= -1m
       selected-window [400,600) regression <= +1m

2. Cross-sequence diagnostic, fixed rule:
       no per-sequence threshold tuning
       does not select every reset boundary
       does not select only one absolute chunk id
```

如果 H2 不成立，停止 trigger learning，不允许 full online trigger validation。

---

## H3：semantic-conditioned C23 与 C9 的冲突来自某个 C9 subcomponent，而不是整体不可兼容

### 假设

C9 + semantic cue 失败不一定说明语义不兼容。可能是 C9 的某个 subcomponent 和 semantic cue 冲突，例如：

```text
C9 read beta chunk map
TTT tri-replay body/exit gamma
SWA keep_scope / overlap source replacement
mp_alpha = 0.1
semantic z full/all-chunk activation
```

### 实验设计

在 trusted short rollout / selected windows 上做 component deconstruction，不先 full。

配置族：

```text
Base-H9 + semantic z
Base-C9 + semantic z

C9 minus read-beta chunks + semantic z
C9 minus TTT tri-replay + semantic z
C9 minus SWA replacement + semantic z
C9 with semantic z read-only only
C9 with semantic z frame/global only
C9 with semantic z no TTT/SWA semantic changes
C9 with semantic z fixed window diagnostic
```

### 记录指标

```text
ATE_h10 / h15
[200,300]
[400,600]
Rot
FinalErr
SWA boundary_10f / boundary_20f
TTT update norm
SWA source mass
frame/global attention mass to high-D
activation window
```

### 成立标准

H3 通过条件：

```text
1. 找到一个 C9 subcomponent ablation 使 semantic z 不再回退：
       h15 ATE delta vs C9 parent <= -1m
       or [200,300] delta <= -3m
       with [400,600] regression <= +1m

2. 若所有 C9 component ablation 都失败：
       semantic-conditioned C23 cannot stack with C9 current protocol
       语义路线不再以 C9 combination 为主。
```

---

## H4：semantic-conditioned C23 的正确形式不是 all-chunk replacement，而是 residual / anchor-conditioned reweighting

### 假设

`SEM_Z_COARSE_BETA525` full online 失败，是因为它把 semantic-conditioned C23 作为全局 replacement。更合理的是只在语义显著改变 cue 解释的地方加 residual：

$$
D_{final} = D_{orig} + \lambda \cdot M_{gate} \cdot (D_{sem} - D_{orig})
$$

其中 $M_{gate}$ 由 semantic trust、cue residual、TTT conflict、scale risk 控制。

### 实验设计

候选：

```text
R1: D_orig only
R2: D_sem replacement
R3: D_orig + 0.25 * (D_sem - D_orig)
R4: D_orig + 0.50 * (D_sem - D_orig)
R5: D_orig + 0.25 * gate_structure * (D_sem - D_orig)
R6: D_orig + 0.25 * gate_risk * (D_sem - D_orig)
R7: D_orig + 0.25 * gate_scale_conflict * (D_sem - D_orig)
```

其中：

```text
gate_structure = structure label AND high masklet trust
gate_risk = high DeltaD AND high D_orig or high conflict
gate_scale_conflict = high scale risk AND high TTT conflict
```

### 成立标准

```text
h10 entry:
    ATE delta <= -1.5m
    or [200,300] delta <= -3m

h15 durability:
    ATE delta <= -2m
    or [200,300] delta <= -4m
    [400,600] regression <= +1m

full online:
    must beat C9 by >= 1m before being called serious candidate
```

如果 residual/anchor-conditioned versions 仍然只给 $<0.5m$ h10 ATE gain，停止 semantic reconditioning action family。

---

## H5：进入 ATE 30 需要 distributed static scale-anchor memory，而不是单窗口 suppress

### 假设

Target-30 需要约 $3.76m$ full ATE improvement。单 chunk / 单 masklet / 单 trigger 不够。语义的更大潜力是构建跨 chunk 的 static scale anchors：

```text
road/building/wall/fence
+ high trust
+ low D_g
+ low TTT conflict
+ low scale risk
+ high source attention mass
+ temporal persistence
```

这些 anchors 应该：

```text
frame/global:
    always source keep

SWA:
    protect as overlap/local alignment anchors

TTT:
    positive long write or no negative-write

C23 support:
    used as support centroid for same semantic anchor role
```

### 实验设计

构建 distributed anchor set：

```text
A0: no semantic anchor
A1: semantic structure only
A2: semantic structure + masklet trust
A3: semantic structure + trust + low D_g
A4: semantic structure + trust + low D_g + low conflict
A5: semantic structure + trust + low D_g + low conflict + low scale risk
A6: A5 + high source attention mass
```

动作：

```text
frame/global:
    source keep / protect

SWA:
    overlap anchor protect, not hard skip

TTT:
    positive long only for A5/A6

C23:
    support centroid restricted / weighted by anchor set
```

### 记录指标

```text
anchor coverage
anchor temporal persistence
anchor source attention mass
anchor SWA overlap mass
anchor TTT positive mass
ATE h10/h15
[200,300], [400,600]
SWA boundary metrics
Sim3Scale
FinalErr
YawRMSE
```

### 成立标准

```text
1. A5/A6 must beat A1/A2:
       showing geometry/TTT/scale conditioning is necessary.

2. h15 ATE delta <= -2m
       or h15 [200,300] delta <= -4m
       with [400,600] regression <= +1m.

3. Full validation only if h15 passes and full activation is not absolute chunk-id based.
```

---

## H6：如果语义仍无法通过 causal gates，应降级为 diagnostic，Target-30 主线回到 trajectory-state / scale-state

### 假设

如果 H1-H5 都没有给出 h15 / full-transfer 信号，那么语义不是 Target-30 主因果变量。它最多用于：

```text
debug visualization
masklet trust calibration
failure mode diagnosis
weak regularization
```

而不是主线。

### 判定标准

如果满足以下任一组条件，则停止 semantic mainline：

```text
1. H1 无任何 reset-relative window 有 h15 >= 2m improvement；
2. H2 trigger precision < 0.5 或 selected windows 平均回退；
3. H3 无 C9-compatible semantic variant；
4. H4 residual reconditioning h10 ATE gain < 0.5m；
5. H5 distributed anchor h15 ATE gain < 1m。
```

停止后主线切换为：

```text
explicit online trajectory-state module
scale-state memory module
merge/gauge-aware correction
TTT/SWA lifecycle independent of semantic labels
```

---

## 4. 并行执行设计

本轮分 6 条 Codex track 并行，避免一条线失败拖慢整体。

---

## Track A：Local-to-full transfer audit

### 目标

解释 v31/v32 的 local-to-full gap。

### 任务

```text
1. 对 H9 和 C9 parent 分别跑 reset-group semantic z short rollout。
2. 生成每个 reset group 的 semantic cue benefit label。
3. 对比 fixed chunk10-12、runtime trigger、all-chunk activation 的 cue maps 和 state stats。
```

### 输出

```text
trackA_transfer_windows.csv
trackA_transfer_features.csv
trackA_transfer_heatmap.png
trackA_state_drift_by_window.png
trackA_decision.md
```

### 不满足条件时 Codex 处理

```text
如果 h10/h15 都没有任何 window 有 >1m ATE gain：
    停止 semantic trigger line，转 H5 static anchor 或 H6 fallback。

如果只有 absolute chunk10-12 有效：
    标记为 diagnostic only，不允许 full trigger。
```

---

## Track B：No-GT semantic cue trigger learner

### 目标

把 fixed-window local gain 变成 deployable runtime trigger。

### 任务

```text
1. 用 Track A 生成 labels。
2. 提取 no-GT features。
3. 拟合 decision tree / rule list / logistic regression。
4. 在 held-out reset groups 测 precision/recall。
5. 只在 trigger 过 gate 时跑 full online。
```

### 输出

```text
trackB_trigger_train.csv
trackB_trigger_rules.json
trackB_trigger_eval.csv
trackB_selected_windows.jsonl
trackB_activation_timeline.png
```

### 失败分流

```text
如果 trigger 只学到 absolute chunk index：
    删除该 feature，重训；若仍失败，停止 trigger line。

如果 trigger false positive 多：
    加 downstream risk feature: SWA boundary risk / scale risk / C9 state movement。

如果 trigger 不选任何窗口：
    说明 semantic local gain不可部署，停止 trigger line。
```

---

## Track C：C9 compatibility decomposition

### 目标

找出 semantic cue 与 C9 冲突的组件。

### 任务

```text
1. C9 full protocol 分解为 read-beta / TTT tri-replay / SWA / mp_alpha。
2. 在 selected windows 上做 ablation + semantic z。
3. 判断哪一部分导致 [400,600) regression 或 ATE regression。
```

### 输出

```text
trackC_c9_component_matrix.csv
trackC_segment_delta_by_component.csv
trackC_c9_conflict_heatmap.png
trackC_decision.md
```

### 失败分流

```text
如果去掉某个 C9 component 后 semantic z 有效：
    该 component 进入 compatibility repair track。

如果所有 ablation 都失败：
    semantic z 不再和 C9 组合，改 H5 static anchor 或 H6 fallback。
```

---

## Track D：Residual / gated semantic-conditioned C23

### 目标

避免 all-chunk replacement 过强，改成 cue residual。

### 任务

```text
1. 实现 D_final = D_orig + lambda * gate * (D_sem - D_orig)。
2. gate 分为 structure / risk / scale-conflict。
3. 先 h10/h15，再 full。
```

### 输出

```text
trackD_residual_cue_matrix.csv
trackD_deltaD_maps/
trackD_Dorig_Dsem_Dfinal_gallery/
trackD_decision.md
```

### 失败分流

```text
如果 D_sem 和 D_orig 的 DeltaD 很小：
    说明 semantic z 对 cue 无实质影响，停止。

如果 h10 有效 h15 失败：
    转 washout attribution。

如果 full 回退但 h15 强：
    转 Track B trigger / Track C compatibility。
```

---

## Track E：Distributed static scale-anchor memory

### 目标

用语义找长期 scale anchors，而不是只找动态负样本。

### 任务

```text
1. 构建 A0-A6 anchor sets。
2. 在 C23 support、frame/global source、SWA cache、TTT positive long write 中使用 anchor。
3. 记录 anchor persistence 和 trajectory metrics。
```

### 输出

```text
trackE_anchor_set_stats.csv
trackE_anchor_rollout_metrics.csv
trackE_anchor_maps/
trackE_anchor_persistence_timeline.png
trackE_decision.md
```

### 失败分流

```text
如果 semantic-only anchor 弱，但 lowD/conflict/scale conditioned anchor 强：
    保留 cue-conditioned anchor line。

如果所有 anchor 弱：
    说明语义不适合做 scale-anchor，转 H6 fallback。
```

---

## Track F：Durability / washout attribution

### 目标

解释 h10 有效但 h15/full 失败。

### 任务

对所有 h10 strong candidates 记录：

```text
h10 endpoint state
h15 endpoint state
TTT state movement
SWA cache movement
global source movement
merge/gauge movement
role mass over time
```

### 输出

```text
trackF_washout_attribution.csv
trackF_state_movement_plot.png
trackF_role_mass_timeline.png
trackF_decision.md
```

### 失败分流

```text
如果 TTT tail update washout:
    尝试 positive long commit 或 W_long/W_short。

如果 SWA refresh washout:
    尝试 source cache persistence / overlap anchor protection。

如果 merge/gauge washout:
    semantic memory path 不是主因，转 explicit trajectory-state correction。
```

---

## 5. 统一记录指标

每个候选必须记录：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
[0,200)
[200,300)
[200,400)
[400,600)
50f mean / worst
100f mean / worst
200f mean / worst
```

Cue / semantic 指标：

```text
D_orig mean/q90
D_sem mean/q90
D_final mean/q90
DeltaD mean/q90
DeltaD by semantic label
semantic_z_high_mass
masklet trust mean
projected 3D support ratio
semantic fine-label distribution
```

Memory path 指标：

```text
frame source keep ratio
global source keep ratio
SWA source keep ratio
SWA boundary_10f / boundary_20f
SWA overlap mass
TTT positive / neutral / negative mass
TTT update norm
TTT conflict mean/q90
scale risk mean/q90
HMC state movement
merge/gauge state movement
```

Trigger 指标：

```text
selected windows
trigger precision / recall / F1
false positives
false negatives
selected-window average gain
selected-window downstream regression
```

---

## 6. 必须可视化

每个通过 h10 的候选必须生成：

```text
1. trajectory XY plot:
       H9 / C9 / candidate / GT

2. segment error timeline:
       per-frame or per-100f ATE

3. D cue gallery:
       RGB
       D_orig
       D_sem
       D_final
       DeltaD

4. semantic overlay:
       fine label map
       masklet trust
       projected 3D sparse support

5. activation timeline:
       chunks selected by trigger
       reset-group view
       semantic cue active flag

6. memory state plot:
       HMC / TTT / SWA / merge movement over chunks

7. SWA boundary plot:
       boundary_10f / boundary_20f before/after
```

---

## 7. Promotion / Stop Rules

### h10 entry gate

```text
ATE delta <= -1.5m
or [200,300) delta <= -3m
```

### h15 durability gate

```text
ATE delta <= -2m
or [200,300) delta <= -4m
[400,600) regression <= +1m
```

### full online gate

```text
must beat C9 by >= 1m
or ATE <= 32.0m
```

### Target-30 gate

```text
KITTI01 full ATE <= 30.0m
no GT runtime action
no offline trajectory rewrite
no absolute chunk-id trigger
```

### semantic-mainline stop gate

Stop Semantic Prior Generator as Target-30 mainline if:

```text
1. no H1 reset-relative window has h15 ATE <= -2m;
2. no H2 trigger reaches precision >= 0.6 and selected mean gain <= -1m;
3. no H3 C9-compatible variant improves C9;
4. no H4 residual cue gives h10 ATE <= -1.5m;
5. no H5 static anchor gives h15 ATE <= -2m.
```

If stopped, keep semantics for:

```text
masklet trust calibration
failure visualization
static/dynamic explanation
weak regularization only
```

and move Target-30 mainline to:

```text
online trajectory-state / scale-state module
merge/gauge-aware correction
TTT/SWA lifecycle independent of semantic labels
```

---

## 8. 数据集诊断原则

允许：

```text
1. 在 KITTI00/01/02/05 上诊断不同 failure mode。
2. 记录不同数据集的 semantic distribution / D_g distribution / scale-risk pattern。
3. 判断某个机制是否只在某类场景有效。
```

禁止：

```text
1. 为单个序列调 threshold。
2. 为 KITTI01 硬编码 chunk id。
3. 使用 sequence id 作为 trigger feature。
4. 为某个数据集单独改 label value / gamma / beta。
```

如果跨序列失败：

```text
1. 先判断是 semantic distribution shift、D_g distribution shift，还是 memory-state shift；
2. 不调数据集专属参数；
3. 若规则不泛化，则降级为 diagnostic。
```

---

## 9. 实际运行顺序

### Batch 0：代码与 action realism audit

```text
B0-1: verify D_orig / D_sem / D_final maps differ
B0-2: verify runtime trigger features are logged before action
B0-3: verify C9 component toggles actually change config
B0-4: verify no absolute chunk id in deployable trigger
```

### Batch 1：H1 transfer audit

```text
H9 reset-groups short rollout
C9 reset-groups short rollout
semantic z / residual / orig C23 variants
```

### Batch 2：H2 trigger learner

```text
fit rule-based trigger
evaluate held-out reset groups
run full only if gate passes
```

### Batch 3：H3 C9 compatibility

```text
C9 component ablation + semantic z
identify conflict source
```

### Batch 4：H4 residual semantic C23

```text
D_orig + lambda * gate * (D_sem - D_orig)
h10/h15 validation
```

### Batch 5：H5 static anchor memory

```text
A0-A6 distributed anchor sets
support + source + SWA + TTT positive long
```

### Batch 6：full online validation

Only if one candidate passes h15 and trigger/compatibility gates:

```text
full KITTI01
then diagnostic KITTI00/02/05 without tuning
```

---

## 10. 本轮预期结果与解释

### 情况 A：semantic-conditioned C23 有 reset-relative trigger

解释：语义信号有效，但需要 state-conditioned activation。进入 full online trigger validation。

### 情况 B：只有 fixed chunk10-12 有效

解释：当前 semantic signal 是 KITTI01 diagnostic，不可部署。停止 trigger line。

### 情况 C：semantic z 与 C9 某一 component 冲突

解释：语义不是无用，而是需要与 C9 protocol 分职责。修 compatibility。

### 情况 D：semantic residual cue 有效

解释：replacement 过强，residual/gated cue 是正确形式。进入 full validation。

### 情况 E：static anchor 有效

解释：语义的主价值是 long-memory anchor，而不是 dynamic suppression。进入 anchor mainline。

### 情况 F：所有方向都弱

解释：语义不能作为 Target-30 主线，降级为 diagnostic。

---

## 11. 最终判断标准

本轮真正成功不是 h10 局部改善，而是：

```text
1. 找到非 absolute-chunk-id 的 runtime trigger；
2. 或找到 C9-compatible semantic cue;
3. 或找到 distributed static anchor memory;
4. 并最终在 full online 中至少达到 ATE <= 32m，
   下一阶段再冲 ATE <= 30m。
```

如果本轮仍然没有 full online $<=32m$ 或至少 beat C9 by $1m$ 的候选，就不应继续语义主线大矩阵。

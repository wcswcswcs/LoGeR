# ACL2 v34：Semantic Cue Runtime Value Trigger、C9 Compatibility 与 Target-30 实验计划

日期：2026-05-23  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / C23 semantic-conditioned cue  
目标：把 v31-v33 中已经出现的 **chunk10 局部强语义 cue 信号** 转化为 **可部署 full online 策略**，并首先冲击 KITTI01 ATE $\le 30m$。

---

## 0. 本轮计划的核心判断

v31-v33 已经说明，继续做语义类别规则矩阵已经不是主线。现在的事实链是：

```text
v31:
    semantic-conditioned C23 / D_g 在 chunk10 short rollout 中产生强局部信号，
    beta525 repair 甚至通过 h15 durability gate，
    但 full online 失败。

v32:
    fixed chunk10-12 full online 比 all-chunks 更好，
    说明语义 cue 不能全序列常开；
    runtime triggers 没有恢复 fixed-window 效果；
    C9 + semantic 全部比 C9 差。

v33:
    chunk10 的 semantic cue 信号在 H9 parent 与 C9 parent 下都能复现，
    说明它不是 H9-only 偶然；
    但是只有 chunk10 稳定，chunk5 弱，chunk16 回退；
    reset-group parent snapshot 覆盖不完整，不能训练 honest no-GT trigger。
```

因此，本轮不再问：

```text
sky 是否跳过？
vegetation 是否负写？
road 是否保留？
```

而要问：

> **semantic-conditioned C23 什么时候是有价值的？它为什么在 chunk10 有效？如何在不使用 absolute chunk id 的情况下触发它，并让它与 C9 兼容？**

本轮的主任务不是设计更多语义规则，而是把 `semantic cue strength detector` 升级成 `semantic cue value predictor`。

---

## 1. 实验整体目标

### 1.1 目标 A：补全 reset-group parent snapshots，解决 v33 的 trigger blocker

v33 的最大 blocker 是 oracle labels 不完整。可用 parent snapshots 只有 `[5,10,16]`，共同命中 expected reset starts 只有 `[5,10]`。如果用这些点训练 trigger，模型会退化成 absolute chunk-id predictor。这不允许。

本轮必须补齐：

```text
expected reset starts = [0, 5, 10, 15, 20, 25, 30]
parents = H9, C9
state = HMC + merge/gauge + global chunk offset
horizons = h10, h15
```

只有完整 reset-group oracle 存在，才能进入 trigger 学习。

### 1.2 目标 B：把 trigger label 定义为“激活 semantic cue 的未来收益”，而不是 cue 自身强弱

v32 已经证明，`semantic_z_high_mass`、`D_mean EMA-MAD`、`D_q90 EMA-MAD` 这类强度 trigger 都失败。它们在预测：

```text
这个 chunk 的 semantic cue 是否强？
```

但真正要预测的是：

```text
如果在这个 chunk 激活 semantic-conditioned C23，未来 h10/h15 trajectory 是否会改善？
```

因此，本轮把每个 reset chunk 的 oracle label 定义为：

$$
Y_m^{h} = ATE_{base,m}^{h} - ATE_{sem,m}^{h}
$$

以及：

$$
Y_{m,[200,300]}^{h} = Err_{base,m,[200,300]}^{h} - Err_{sem,m,[200,300]}^{h}
$$

这里 $h \in \{10,15\}$。只有 $Y$ 足够大，才认为该 chunk 应该激活 semantic cue。

### 1.3 目标 C：解释 C9 compatibility，而不是继续盲目叠加

C9 是当前 best deployable online TTT write，ATE 为 `33.7629421029m`。v32 已经显示所有 `C9 + semantic` full online 都更差，其中 best C9+semantic 仍比 C9 差 `+0.5323m`，并且 fixed chunk10-12 虽改善 `[200,300)`，却让 `[400,600)` 回退 `+3.1541m`。

所以本轮必须拆清楚：

```text
semantic cue 与 C9 的哪个机制冲突？

可能冲突源：
    1. C9 read beta chunk map
    2. TTT tri-replay / update_conflict_energy
    3. SWA keep-scope / overlap source replacement
    4. mp_alpha / stage_d_x_dg_inv_sqrt write score
    5. semantic cue全序列常开导致低风险 chunk 过度干预
```

如果不拆 C9，继续全量叠加 semantic cue 没意义。

### 1.4 目标 D：从 semantic cue replacement 改为 residual correction

v31/v32 里的语义 cue更多是替换或强改写 D_g。下一步应改成更保守的残差形式：

$$
D_{final} = D_{orig} + \lambda \cdot G_{value} \cdot (D_{sem} - D_{orig})
$$

其中：

```text
D_orig:
    原 C23 past D_g
D_sem:
    semantic-conditioned C23 / D_g
G_value:
    no-GT runtime value trigger，表示当前 chunk 是否值得使用 semantic residual
lambda:
    残差强度，不再全量替换
```

这个设计的本质是：

```text
semantic cue 只在它有价值时纠正 D_orig，
不在全序列强行重写 D_g。
```

### 1.5 目标 E：如果 semantic cue 仍不能 transfer，则明确降级

本轮目标是 Target-30，不是继续证明“有局部信号”。如果补全 oracle、训练 trigger、拆解 C9、做 residual correction 后仍无法过 gate，则 Semantic Prior Generator 不再作为 Target-30 主线，而降级为：

```text
1. failure diagnosis
2. cue visualization
3. weak regularizer
4. masklet trust calibration
```

主线应回到 explicit online trajectory-state / scale-state module。

---

## 2. 固定边界与不可违反的规则

### 2.1 当前 deployable baseline

本轮所有 full online 结果必须和以下边界对比：

```text
C9_P0_R2
ATE = 33.7629421029m
counts_as_ttt_write = true
no GT runtime action
no offline trajectory rewrite
```

Target-30 gap：

$$
33.7629421029 - 30.0 = 3.7629421029m
$$

### 2.2 不允许的数据集调参

可以用 KITTI01、KITTI00、KITTI02、KITTI05 诊断不同 failure mode，但不允许：

```text
KITTI01 专用阈值
某个 sequence 专用 chunk id
某个 dataset 专用 label table
某个 sequence 专用 gamma / beta
```

允许使用：

```text
reset-relative index
runtime cue statistics
D_g distribution
semantic z statistics
TTT conflict
scale-state proxy
masklet trust
SWA boundary health
```

### 2.3 fixed chunk 只用于 oracle，不可部署

固定 `[10,11,12]` activation 可以作为 diagnostic，不得作为 deployable success。

### 2.4 short rollout 不是 full online success

h10/h15 只能作为 causal diagnostic。必须通过 selector / trigger gate 后才允许 full online。

---

## 3. 核心假设

## H1：v33 的 chunk10 local signal 是真实 semantic cue value，但其分布不是所有 reset groups 通用

### 假设

semantic-conditioned C23 对某些 reset groups 有强因果作用，但不是全局常开策略。它需要被 runtime value trigger 选择性激活。

### 实验设计

补齐 H9/C9 parent snapshots 后，对每个 expected reset start 运行：

```text
parent ∈ {H9, C9}
chunk ∈ {0,5,10,15,20,25,30}
horizon ∈ {h10,h15}
candidates:
    base C23
    SEM_Z_COARSE_BETA525
    SEM_Z_FINE
    SEM_RESID_COARSE_L025
    ORIG_C23
```

记录每个 candidate 的：

```text
ATE delta vs base
[200,300) delta
[400,600) delta
Rot delta
FinalErr delta
SWA boundary 10f/20f delta
active chunk id
reset-relative position
parent type H9/C9
```

### 判断标准

H1 成立条件：

```text
1. 至少两个 reset groups 上 SEM_Z_COARSE 或 SEM_Z_FINE 有正收益；
2. chunk10 仍然是强正例，但不是唯一正例；
3. 至少一个非 chunk10 正例满足：
       h15 ATE delta <= -1.5m
       或 h15 [200,300) delta <= -3m
4. [400,600) regression <= +1m。
```

如果只有 chunk10 一个正例，H1 仍然是 diagnostic-only，不允许训练 deployable trigger。

---

## H2：runtime trigger 应预测 semantic cue value，而不是 semantic cue strength

### 假设

v32 trigger 失败，是因为它们预测的是 cue strength，而不是 future value。value predictor 应该用 no-GT 特征预测 $Y_m^h$。

### 特征设计

每个 chunk $m$ 生成一个 feature vector：

```text
semantic cue features:
    mean(D_sem)
    q90(D_sem)
    mean(|D_sem - D_orig|)
    q90(|D_sem - D_orig|)
    semantic_z_high_mass
    per-label z outlier mass

C23 / attention features:
    mean(D_orig)
    q90(D_orig)
    high-D mass
    anchor collision
    fragmentation
    source attention mass to high-D

masklet trust features:
    mean Q_mask
    label flip rate
    temporal IoU
    projected 3D agreement if available
    masklet fragmentation

TTT / memory features:
    update_conflict_energy mean / q90
    TTT update norm spike
    memory state rel diff
    branch0 update norm

SWA / boundary features:
    overlap source mass
    boundary_10f_proxy
    boundary_20f_proxy
    source keep ratio

scale / trajectory proxy:
    no-GT pose-step EMA scale proxy
    local yaw jump proxy
    local translation drift proxy

reset-relative features:
    reset_phase = chunk_id mod RESET_EVERY
    distance_from_reset_start
    distance_to_reset_end
```

禁止使用：

```text
absolute chunk id
absolute frame id
sequence id
KITTI01-specific hand label
```

### 模型

只允许先用简单可解释模型：

```text
rule list
small decision tree depth <= 3
logistic regression
isotonic threshold over one or two features
```

不先上复杂模型，避免过拟合。

### 判断标准

trigger 必须通过：

```text
1. Leave-one-reset-group-out validation：
       held-out positive recall >= 0.5
       false positive rate <= 0.35

2. 不使用 absolute chunk id 时仍然能找到 chunk10-like 正例。

3. 在 H9 与 C9 parent 上都不过度偏向某一种 parent。

4. 触发后的 h15 average ATE delta <= -1.5m，
   或 h15 [200,300) delta <= -3m，
   且 [400,600) regression <= +1m。
```

如果 trigger 只能记住 chunk10，则失败。

---

## H3：semantic-conditioned C23 与 C9 的冲突来自某个 C9 component，而非语义 cue 本身无效

### 假设

v32 中 `C9 + semantic` 全部更差，不代表 semantic cue 无效；可能是 C9 的某个 memory component 与 semantic cue 冲突。

### 实验设计

在 short rollout 下拆 C9 component。父状态使用 C9 parent snapshots。

Candidate families：

```text
C9_FULL:
    full C9 parent / full C9 protocol

C9_NO_TTT_TRI:
    disable TTT tri-replay modification，保留 C9 read/SWA

C9_NO_SWA_REPLACE:
    disable SWA overlap replacement，保留 C9 read/TTT

C9_READ_ONLY_COMPAT:
    use C9 read beta map，但 commit = probe_native

C9_TTT_ONLY_COMPAT:
    use C9 TTT write path，但 no semantic read change

C9_SEM_READ_ONLY:
    apply semantic residual only to frame/global read, no TTT/SWA semantic action

C9_SEM_SWA_ONLY:
    apply semantic residual only to SWA source, no frame/global/TTT semantic action

C9_SEM_TTT_ONLY:
    apply semantic-conditioned write only, no frame/global read semantic change
```

对每个 family 测：

```text
semantic mode:
    none
    SEM_Z_COARSE_BETA525 replacement
    SEM_RESID_COARSE_L025
    D_final = D_orig + lambda*(D_sem-D_orig), lambda=0.25/0.50
```

### 判断标准

H3 成立条件：

```text
1. 至少一个 C9 component ablation 下，semantic residual h15 ATE delta <= -1.5m；
2. 该 component ablation 不导致 [400,600) regression > +1m；
3. 能解释 v32 的 C9+semantic full failure，例如：
       semantic + SWA replacement 冲突，
       或 semantic + TTT tri-replay 冲突。
```

如果所有 C9 ablations 下 semantic cue都无效，则语义 cue与 C9不兼容，应退出 C9 stacking。

---

## H4：semantic cue 应以 residual 方式修正 D_g，而非替换 D_g

### 假设

v31/v32 replacement 过强，full online 失败。残差式 reconditioning 可以保留原 C23 的稳定性，只在 value trigger 认为有用时修正。

### 实验设计

定义：

$$
D_{final} = D_{orig} + \lambda \cdot G_m \cdot (D_{sem} - D_{orig})
$$

其中 $G_m$ 是 trigger 输出，取值 $0$ 或 $[0,1]$。

测试：

```text
lambda = 0.25, 0.50, 0.75
G_m = oracle fixed-window label   # diagnostic only
G_m = learned no-GT trigger       # deployable candidate
scope = frame/global read only
scope = frame/global + SWA
scope = frame/global + TTT write
```

### 判断标准

残差策略必须满足：

```text
1. short h15 ATE delta <= -2m or [200,300) delta <= -4m；
2. [400,600) regression <= +1m；
3. full online vs C9 improves by >= 1m，
   或 full online ATE <= 32.5m 进入下一轮；
4. 不使用 fixed absolute chunk id。
```

---

## H5：语义真正有用的方式是构建 distributed static scale anchors，而不是 dynamic suppression

### 假设

Target-30 需要稳定长期 anchor。语义应帮助寻找跨 chunk 持久的 static scale-anchor set，而不只是跳过动态/低价值区域。

### Anchor 定义

候选 anchor token 满足：

```text
semantic ∈ {road, building, wall, fence, stable structure}
masklet trust high
D_g low
TTT conflict low
scale-risk low
source attention mass high
projected 3D semantic agreement high if available
```

定义 anchor score：

$$
A_{anchor}(i)=T_{mask}(i) \cdot (1-D_g(i)) \cdot (1-C_{conflict}(i)) \cdot (1-S_{scale}(i)) \cdot V_{sem}(i)
$$

其中 $V_{sem}$ 不直接决定写入，只作为语义 value。

### 动作

```text
frame/global:
    source keep / protect from skip

SWA:
    overlap/local cache protect

TTT:
    positive long write，not negative
```

### 判断标准

H5 成立条件：

```text
1. h15 ATE delta <= -2m；
2. h15 [400,600) 不回退；
3. 相比 dynamic suppression，anchor strategy 更能改善 FinalErr / Sim3 scale；
4. full online 相对 C9 改善 >= 1m，或进入 32.5m 以下。
```

如果 anchor 只改善局部 h10 不改善 h15，则说明 persistence 仍未解决。

---

## H6：如果 semantic local signal 仍然不能 full transfer，Target-30 主线必须转向 trajectory-state / scale-state

### 假设

如果 v31-v33 的 local signal 无法通过 value trigger / residual cue / anchor memory 转成 full online，那么语义只能作为 diagnostic，不能作为 Target-30 主线。

### 停止标准

如果出现任一情况，本轮停止语义主线扩展：

```text
1. reset-group oracle 补全后，semantic cue 只有 chunk10 一个正例；
2. trigger 无法跨 reset-group 泛化；
3. C9 component ablation 下没有任何 semantic residual 正信号；
4. residual D_final full online 不改善 C9；
5. static anchor h15 gate 不通过；
6. 所有 full online 候选仍 > 33m。
```

转向：

```text
explicit online trajectory-state module
explicit scale-state module
merge/gauge-aware correction
C9-native lifecycle / risk-state repair
```

---

## 4. 必须记录的指标

### 4.1 轨迹主指标

每个 h10/h15 与 full run 都记录：

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
50f / 100f / 200f mean and worst
```

### 4.2 local-to-full transfer 指标

```text
local_h10_delta
local_h15_delta
full_delta
transfer_ratio = full_delta / h15_delta
activation_chunks
activation_precision
activation_recall
false_positive_chunks
false_negative_chunks
```

如果 $|h15\_delta|$ 很大但 $|full\_delta|$ 很小，必须进入 washout analysis。

### 4.3 trigger 指标

```text
oracle_label_h10
oracle_label_h15
predicted_trigger_score
trigger_threshold
trigger_decision
TP / FP / TN / FN
leave-one-reset-group-out metrics
feature_importance
uses_absolute_chunk_id flag
```

### 4.4 cue map 指标

```text
mean(D_orig)
mean(D_sem)
mean(abs(D_sem-D_orig))
q90(abs(D_sem-D_orig))
semantic_z_high_mass
per-label D_orig mean/q90
per-label D_sem mean/q90
per-label residual mass
```

### 4.5 memory state 指标

```text
TTT update norm by layer/branch
TTT conflict mean/q90
SWA source keep ratio
SWA boundary source mass
frame/global attention mass to high-D
merge/gauge state movement
HMC state rel diff
```

### 4.6 C9 compatibility 指标

```text
C9 component active flags
C9 read beta map
TTT tri-replay role mass
SWA replacement mass
mp_alpha
semantic residual strength
component interaction delta
```

---

## 5. 必须可视化

### 5.1 Activation timeline

画出全序列：

```text
chunk index
reset group
semantic trigger score
oracle label
activation decision
[200,300) / [400,600) contribution
```

必须能看出 trigger 是否只是 chunk10 detector。

### 5.2 Cue map dashboard

对 chunk5 / 10 / 16 和其他 reset starts 输出：

```text
RGB
D_orig
D_sem
D_sem - D_orig
semantic label map
masklet trust
TTT conflict
scale-risk
trigger score map
```

### 5.3 C9 compatibility heatmap

矩阵：

```text
rows = C9 components / ablations
cols = semantic modes
cell = h15 ATE delta / [200,300) delta / [400,600) delta
```

### 5.4 Transfer dashboard

对每个 candidate：

```text
h10 delta
h15 delta
full delta
transfer ratio
```

如果 h10/h15 很强但 full 失败，要可视化 state drift。

### 5.5 Anchor persistence dashboard

记录 static anchor：

```text
anchor mass over chunks
anchor semantic distribution
anchor source attention mass
SWA overlap anchor survival
TTT positive write mass
h10 -> h15 survival ratio
```

---

## 6. 并行执行计划

### Codex A：Snapshot Coverage Builder

任务：

```text
1. 补齐 H9/C9 parent snapshots for reset starts [0,5,10,15,20,25,30]
2. 确保 HMC + merge/gauge + global_chunk_offset 齐全
3. 跑 h1/h3 parity smoke
4. 生成 snapshot_coverage_report.md
```

通过标准：

```text
complete_expected_reset_coverage = true
H9/C9 parent available for all reset starts
h3 raw pose diff = 0
HMC hash mismatch = 0
```

失败自动分流：

```text
if missing HMC snapshot:
    rerun boundary with SAVE_HMC_STATE_CHUNKS reset starts
if missing merge snapshot:
    rerun with SAVE_MERGE_STATE_CHUNKS reset starts
if disk > limit:
    save only input state + merge cursor, not full before/after states
if parity fails:
    inspect global_chunk_offset / timestamp mapping / merge cursor
```

### Codex B：Reset Oracle Bank

任务：

```text
1. 对每个 reset start 跑 H9/C9 parent h10/h15
2. 候选包括 base, sem_z_coarse, sem_z_fine, sem_resid
3. 生成 oracle_labels.csv
```

通过标准：

```text
rows complete = expected_reset_starts * parents * candidates * horizons
no failed rows
at least two non-absolute positive examples OR semantic line is diagnostic only
```

失败自动分流：

```text
if only chunk10 positive:
    stop trigger learner; go to C9 component decomposition and static anchor only
if candidates fail on all reset groups:
    semantic cue reconditioning not mainline
```

### Codex C：No-GT Value Trigger Learner

任务：

```text
1. 从 oracle bank 提取 no-GT features
2. 训练 rule list / shallow tree / logistic trigger
3. 做 leave-one-reset-group-out validation
4. 输出 trigger config
```

通过标准：

```text
held-out positive recall >= 0.5
false positive rate <= 0.35
no absolute chunk id used
no sequence id used
```

失败自动分流：

```text
if trigger uses reset_phase only:
    remove reset_phase and retrain
if trigger overfits chunk10:
    require additional positives or stop
if false positives cause [400,600) regression:
    add downstream-risk feature and rerun validation
```

### Codex D：C9 Compatibility Decomposition

任务：

```text
1. 在 C9 parent 上逐个关闭 / 保留 C9 components
2. 测 semantic residual 与每个 component 的 interaction
3. 生成 compatibility heatmap
```

通过标准：

```text
找到至少一个 C9-compatible semantic scope：
    h15 ATE delta <= -1.5m
    or [200,300) delta <= -3m
    [400,600) regression <= +1m
```

失败自动分流：

```text
if semantic conflicts with SWA replacement:
    try frame/global-only semantic residual
if semantic conflicts with TTT tri-replay:
    disable semantic TTT write, keep read only
if semantic only works without C9:
    treat as alternative branch, not C9 additive branch
```

### Codex E：Residual D_final Implementer

任务：

```text
1. 实现 D_final = D_orig + lambda * G * (D_sem - D_orig)
2. 支持 oracle G 与 learned trigger G
3. 支持 frame/global only, +SWA, +TTT scopes
```

通过标准：

```text
oracle residual must improve h15 and not regress [400,600)
learned residual must reproduce at least 50% oracle gain
```

失败自动分流：

```text
if oracle residual fails:
    semantic replacement/residual line stops
if learned fails but oracle passes:
    improve trigger features, not cue math
```

### Codex F：Distributed Static Anchor Track

任务：

```text
1. 构建 static scale-anchor score
2. 保护 anchor in frame/global/SWA
3. positive long write in TTT
4. 监控 h10/h15 persistence
```

通过标准：

```text
h15 ATE delta <= -2m
or h15 FinalErr/Sim3Scale substantial improvement
[400,600) no regression
```

失败自动分流：

```text
if anchor improves h10 but not h15:
    add persistence / anchor handoff
if anchor hurts SWA boundary:
    protect only non-boundary or use SWA partial keep
if anchor TTT write weak:
    keep anchor for read/SWA only, drop TTT positive
```

---

## 7. Full online validation gate

只有同时满足以下条件，才允许 full online：

```text
1. short h15 gate:
       ATE delta <= -2m
       or [200,300) delta <= -4m

2. C9 compatibility gate:
       C9 parent h15 not worse than C9 base
       [400,600) regression <= +1m

3. trigger gate:
       deployable trigger does not use absolute chunk id
       held-out reset group validation passes

4. action audit:
       semantic residual actually changes D_g / source masks
       no no-op action collapse
```

Full online promotion：

```text
weak success:
    full ATE <= 32.5m

stage success:
    full ATE <= 31.0m

Target-30 success:
    full ATE <= 30.0m
```

如果 full online improvement is < 1m vs C9，且没有 segment-level large gain，则不继续微调。

---

## 8. 本轮停止规则

停止 semantic cue deployment line if：

```text
1. reset oracle bank cannot produce non-absolute positives;
2. value trigger cannot generalize;
3. C9 compatibility decomposition finds no compatible scope;
4. oracle residual fails;
5. static anchor line fails h15;
6. full online remains > 33m after all above.
```

一旦停止，结论写成：

```text
semantic cue reconditioning is a local diagnostic / weak regularizer;
Target-30 mainline should return to explicit trajectory-state / scale-state / merge-gauge correction.
```

---

## 9. 预期输出目录

```text
results/kitti01_hmc_v2/acl2_v34_semanticcue_value_trigger_c9compat_target30/
    phase0_snapshot_coverage/
    phase1_reset_oracle_bank/
    phase2_value_trigger/
    phase3_c9_decomposition/
    phase4_residual_dfinal/
    phase5_static_anchor/
    phase6_full_online/
    dashboards/
```

每个 phase 必须输出：

```text
summary.json
summary.csv
report.md
invalid_rows.jsonl
config_snapshot.yaml
```

---

## 10. 一句话总结

v34 不再继续“语义规则矩阵”。本轮只回答一个核心问题：

> **semantic-conditioned C23 的局部强信号，能不能通过非绝对 chunk-id 的 value trigger 和 C9-compatible residual cue 转成 full online Target-30 收益？**

如果不能，语义路线应降级为 diagnostic，Target-30 主线回到 trajectory-state / scale-state。

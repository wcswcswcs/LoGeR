# ACL2 v35B：Training-Free Semantic Cue Self-Consistency、C9 Compatibility 与 Target-30 实验计划

日期：2026-05-24  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / ACL2 C23 read cue / C9 best online protocol  
当前 deployable online best：`C9_P0_R2`, KITTI01 ATE `33.7629421029m`  
阶段目标：先进入 KITTI01 ATE `<= 30m`  
核心约束：**training-free，不训练 trigger，不训练 selector，不训练 role router，不用 GT runtime，不针对 KITTI01 或任何单一数据集调参。**

---

## 0. 先纠偏：本项目到底在做什么

本项目不是在 KITTI01 上学习一个触发器，也不是为了打榜去拟合某个 sequence 的 chunk id。当前 pipeline 的目标是：

> 在不训练新模型、不使用 GT runtime、不做数据集专用调参的前提下，从 LoGeR 内部 cue、hybrid memory 状态、Video Masklet 语义先验和无 GT self-consistency 指标中，构造可解释、可部署、可审计的 memory control policy，以改善 long-context 3D reconstruction。

因此，从本计划开始，下面这些做法明确禁止进入主线：

```text
1. 训练 trigger / classifier / decision tree / random forest / logistic regression。
2. 用 reset oracle label 拟合选择规则。
3. 使用 absolute chunk id、KITTI01-specific chunk window、sequence id 作为 runtime policy 输入。
4. 为 KITTI01 单独调 semantic label value、threshold、gamma、beta。
5. 把 fixed chunk diagnostic、short rollout oracle、GT projection diagnostic、offline trajectory rewrite 写成 deployable online success。
6. 在 h10/h15 gate 没过时强行跑 full online。
```

允许的 training-free 策略只有三类：

```text
1. Deterministic robust-stat rule:
   使用 chunk / reset-block 内部统计，例如 median、MAD、quantile、energy normalization。

2. Paired no-commit probe:
   从同一个 committed HMC state 出发，对比 original cue 与 semantic-conditioned cue 的无 GT consistency 指标。

3. Predefined path-specific memory role:
   使用语义 + D_g + TTT conflict + scale-risk + masklet trust 的固定 conjunction rule，且规则不依赖数据集 id。
```

---

## 1. 为什么 v31-v34 说明必须换范式

最近几轮实验给出的信息非常清楚。

v31 证明 `semantic-conditioned C23 / D_g reconditioning` 有真实局部信号。`SEM_Z_COARSE_BETA525` 在 h10/h15 short rollout 上显著压低 `[200,300)`，h15 `[200,300)` 可以达到约 `-6m` 级别。但 full online 只得到 `36.6906m`，远没有达到 Target-30。

v32 进一步证明，fixed chunk10-12 full online 能相对 H9 改善约 `0.739m`，但这是 fixed chunk id diagnostic，不可部署；同时 tested runtime triggers 失败，`C9 + semantic` 全部差于 C9，说明语义 cue 与当前最强 C9 protocol 存在冲突。

v33 证明，semantic cue 在 C9 parent 下的 chunk10 short diagnostic 仍然成立：局部信号不是 H9 偶然状态造成的。但 reset-group snapshot coverage 不完整，无法诚实训练或验证 trigger。

v34 补齐 H9/C9 reset-group parent snapshots，并完成完整 H1 reset oracle。结果显示 h15 positive 出现在 reset chunks `0` 和 `10`，H9/C9 都成立。但 no-GT runtime value trigger 在 leave-one-reset-group-out 上完全失败：held-out positive recall 为 `0.0`，false positive 很高。这说明 fixed/reset-window local oracle 是真实的，但当前“cue strength trigger”不能部署。

这里的关键不是“trigger 再训练一下就好”，而是：**训练 trigger 本身不符合项目目标。** 我们要改成 training-free paired-probe rule。

当前最重要的科学判断是：

> 语义 cue 最有价值的地方不是直接写 memory，也不是单独决定 sky/road/vegetation 的动作，而是重新解释 C23 / D_g。接下来的问题是：如何在不训练、不用 GT、不用 chunk id 的条件下，判断什么时候 semantic-conditioned C23 应该被启用，并且如何让它兼容当前 C9 best protocol。

---

## 2. 总体目标

本轮目标不是继续扩大 semantic rule matrix，而是建立一个真正 training-free 的 semantic cue activation / memory role protocol。

本轮要回答六个问题。

### 2.1 问题 A：semantic-conditioned C23 的局部收益能否用无 GT paired-probe 指标识别？

过去失败的 trigger 在预测“semantic cue 强不强”。这不是正确目标。真正需要判断的是：

```text
从当前 committed HMC state 出发，如果我临时打开 semantic-conditioned C23，
无 GT consistency 是否变好，且不会伤 SWA boundary / TTT state / static anchor source？
```

如果 paired-probe 指标能识别 chunk0 / chunk10 这类 positive windows，那么 semantic cue 才可能 deployable。

### 2.2 问题 B：semantic-conditioned C23 应该 residual 注入，而不是替换原 C23 吗？

直接替换 `D_g` 可能太强。更合理的是保留原始 C23，再加入能量归一的 semantic residual：

$$
D_{final}=\operatorname{clip}(D_{base}+\lambda G_{probe}(D_{sem}-D_{base}),0,1)
$$

其中 $G_{probe}$ 是 paired-probe 的 deterministic gate，不是 trained trigger。

### 2.3 问题 C：C9 哪个 component 与 semantic-conditioned C23 冲突？

v32 证明 `C9 + semantic` 全部比 C9 差，但没有拆清冲突源。C9 里包含 read beta map、TTT tri-replay、SWA overlap replacement、`mp_alpha=0.1` 等机制。下一步必须拆开：

```text
是 read beta map 冲突？
是 TTT tri-replay 冲突？
是 SWA overlap source replacement 冲突？
是 semantic cue 全局常开导致低风险窗口过度干预？
```

### 2.4 问题 D：chunk0 positive 和 chunk10 positive 是否是不同机制？

v34 完整 oracle 显示 h15 positive chunks 是 `0` 和 `10`。二者很可能机制不同：

```text
chunk0:
    更像 sequence initialization / reset-start scale-state setup。

chunk10:
    更像 [200,300) disease-window read rescue。
```

training-free policy 不应该用一个静态 trigger 混合解释这两个机制，而应该用不同 self-consistency guard 区分。

### 2.5 问题 E：语义是否应构造 distributed static scale anchors，而不只是 dynamic suppression？

过去太关注 `skip bad tokens`。但要从 `33.76m` 进入 `30m`，更可能需要长期稳定的 scale anchors。语义的作用可能是：

```text
road / building / wall / fence + high trust + low D_g + low conflict + low scale-risk
    -> frame/global source keep
    -> SWA overlap/local anchor
    -> TTT positive long memory
```

这个方向必须与 semantic-conditioned C23 并行验证。

### 2.6 问题 F：如果 training-free semantic paired-probe 仍不能进入 ATE 30，是否应降级语义主线？

本轮必须有明确停止标准。若 paired-probe、C9 compatibility、residual injection、static anchor 四条线都不能让 full online 到 `<=32m`，就不应继续把 Semantic Prior Generator 当 Target-30 主线。它应降级为 diagnostic / trust calibration / weak regularizer。

---

## 3. 核心假设

### H1：paired no-commit probe 比 cue-strength trigger 更符合 training-free 目标

#### 假设

不训练 trigger，也不用 oracle label。对每个 candidate chunk，从同一个 committed state 出发执行两个 probe-only forward：

```text
Probe A: original C23 / D_base
Probe B: semantic-conditioned C23 / D_sem
```

如果 semantic-conditioned probe 在多个无 GT consistency 指标上呈 Pareto 改善，就启用 semantic residual；否则保持原 C23。

#### Training-free decision rule

定义无 GT probe 指标集合：

```text
M_overlap:
    overlap pointmap consistency / overlap local residual

M_boundary:
    boundary_10f_proxy / boundary_20f_proxy / chunk boundary pose jump proxy

M_attn:
    attention mass to high-D source, attention mass to static anchors

M_ttt:
    TTT update_conflict_energy, TTT update norm, branch0 conflict mass

M_swa:
    SWA source keep health, context empty events, overlap source mass

M_scale:
    pose-step scale consistency proxy, reset-block step-length EMA residual
```

设 $\Delta M_k = M_k(B)-M_k(A)$，每个指标方向固定为“越低越好”或“越高越好”。用 robust block statistics 判断改善：

$$
Improve_k = 1[\Delta M_k < -\tau_k]
$$

$$
Regress_k = 1[\Delta M_k > \rho_k]
$$

其中 $\tau_k,\rho_k$ 由当前 reset-block 的 median/MAD 或固定相对比例得到，不从数据集拟合。

启用 semantic residual 的 deterministic rule：

```text
G_probe = 1 if:
    improve_count >= 3
    and hard_regress_count = 0
    and SWA_boundary_risk = false
    and static_anchor_mass_drop <= allowed_drop
else:
    G_probe = 0
```

#### 必须记录的指标

```text
probe_pair_id
parent = H9 / C9
chunk_id_global
reset_phase
D_base_mean / D_sem_mean / D_delta_rms
M_overlap_A / M_overlap_B / delta
M_boundary_A / M_boundary_B / delta
M_attn_highD_A / B / delta
M_attn_static_anchor_A / B / delta
M_ttt_conflict_A / B / delta
M_swa_health_A / B / delta
M_scale_A / B / delta
improve_count
hard_regress_count
G_probe
no_commit_state_hash_before
no_commit_state_hash_after_A
no_commit_state_hash_after_B
```

#### 假设成立标准

H1 通过条件：

```text
1. paired-probe 不改变 committed HMC state；hash before/after exact。
2. 对 v34 oracle positive chunks 0/10，G_probe 能命中至少 3/4 parent-horizon cases。
3. 对 v34 negative reset chunks，false positive <= 25%。
4. 不使用 absolute chunk id，不读取 GT，不读取 oracle outcome。
```

如果 H1 不成立，不允许进入 full online semantic activation。Codex 必须转入 probe metric repair，而不是训练 trigger。

---

### H2：semantic-conditioned C23 应以 residual injection 方式进入，而不是全量替换 D_g

#### 假设

全量替换 `D_base` 为 `D_sem` 会扰动低风险 chunk 和 C9 状态。保守 residual injection 更可能兼容 full online：

$$
D_{final}=\operatorname{clip}(D_{base}+\lambda G_{probe}\Delta D,0,1)
$$

其中：

$$
\Delta D = D_{sem}-D_{base}
$$

$\lambda$ 不训练，使用能量归一：

$$
\lambda = \min\left(\lambda_{max},\frac{\rho\cdot RMS(D_{base})}{RMS(\Delta D)+\epsilon}\right)
$$

固定候选：

```text
rho = 0.25
lambda_max = 0.50
```

这不是 tuning sweep，而是保守注入策略。只有如果 Phase H1/H2 有明确强信号，才允许试一个 `rho=0.35` 的预注册 ablation。

#### 实验设计

比较四个方案：

```text
D0_ORIG:
    original C23, no semantic.

D1_SEM_REPLACE:
    semantic-conditioned C23 full replacement, diagnostic only.

D2_SEM_RESID_FIXED:
    D_base + 0.25 * (D_sem-D_base), always on, diagnostic.

D3_SEM_RESID_PROBE:
    energy-normalized residual, gated by paired-probe G_probe.
```

先跑 causal fork h10/h15，不直接 full。

#### 必须记录

```text
D map distribution:
    mean / std / p10 / p50 / p90 / mass > 0.5

semantic residual:
    delta_rms
    delta_positive_mass
    delta_negative_mass
    per semantic label delta mean

trajectory:
    ATE
    Rot
    FinalErr
    [200,300)
    [400,600)
    boundary_10f / 20f
    h10 / h15 durability ratio
```

#### 通过标准

```text
h10 gate:
    ATE delta <= -1.5m
    or [200,300) delta <= -3m
    and [400,600) regression <= +1m
    and boundary regression <= +0.25m

h15 gate:
    ATE delta <= -3m
    or [200,300) delta <= -5m
    and durability_ratio >= 0.45
    and [400,600) regression <= +1m
```

如果 `D1_SEM_REPLACE` 有局部效果但 `D3_SEM_RESID_PROBE` 无效，说明 probe guard 或 residual injection 太保守；Codex 应先检查 G_probe 命中率和 action magnitude，不允许转向 training trigger。

---

### H3：C9 compatibility 需要 component isolation，而不是直接 C9+semantic

#### 假设

C9 是当前最好 deployable online protocol，但它不是单一机制。Semantic cue 与 C9 的冲突可能只来自某个 component。如果直接跑 C9+semantic 全量，无法定位原因。

#### 实验设计

建立 C9 decomposition sandbox。所有 row 都使用同一个 semantic residual policy，分别打开/关闭 C9 components。

```text
C9_FULL:
    C9 reference。

C9_READ_ONLY:
    C9 read beta map retained, TTT tri-replay disabled, SWA replacement disabled。

C9_TTT_ONLY:
    C9 TTT tri-replay retained, read beta map reverted to H9, SWA replacement disabled。

C9_SWA_ONLY:
    C9 SWA overlap replacement retained, read beta map reverted to H9, TTT tri-replay disabled。

C9_NO_TTT:
    C9 without TTT tri-replay + semantic residual。

C9_NO_SWA:
    C9 without SWA replacement + semantic residual。

C9_READ_SEM_ONLY:
    semantic residual affects read path only; TTT/SWA write/read role unchanged。

C9_TTT_BLOCK_SEM:
    semantic residual affects frame/global read but explicitly blocked from TTT write score。
```

#### 必须记录

```text
ATE / Rot / RPE / FinalErr
[200,300), [400,600), [200,400)
C9 component flags
semantic cue active mass
TTT tri-replay role mass
SWA source replacement mass
frame/global attention mass to high-D and static anchors
state movement: HMC / TTT / SWA / merge if available
```

#### 判断标准

```text
如果 C9_READ_SEM_ONLY 比 C9 好，但 C9_FULL+SEM 差：
    semantic cue 与 TTT/SWA commit 冲突；后续只允许 read-only semantic residual。

如果 C9_NO_SWA+SEM 好，而 C9_FULL+SEM 差：
    semantic cue 与 SWA overlap replacement 冲突；需要 SWA protection。

如果 C9_NO_TTT+SEM 好，而 C9_FULL+SEM 差：
    semantic cue 与 TTT tri-replay 冲突；禁止 semantic residual 进入 TTT write。

如果所有 C9 decompositions 都差：
    semantic cue 与 C9 state 不兼容；只保留 H9 diagnostic，不作为 deployable Target-30 主线。
```

H3 通过的最低条件：至少一个 C9-compatible semantic row 在 h15 上满足：

```text
ATE delta vs C9 <= -1.5m
or [200,300) delta <= -3m with [400,600) regression <= +1m
```

---

### H4：chunk0 和 chunk10 是两个机制，不能用同一个 rule

#### 假设

v34 reset oracle 中 positive chunks 是 `0` 和 `10`。它们可能分别代表：

```text
chunk0:
    reset / initialization / scale-state setup。

chunk10:
    disease-window read rescue。
```

二者应使用不同 training-free guard，而不是统一 trigger。

#### 实验设计

建立两个 deterministic mode：

```text
Mode INIT_SCALE:
    只允许在 reset-block start 或 sequence early phase 进行 probe；
    activation 由 scale consistency / overlap consistency / static anchor source mass 决定；
    不使用 absolute sequence chunk id。

Mode READ_RESCUE:
    允许在任意 reset block 内进行 probe；
    activation 由 attention mass to highD decrease、overlap consistency、TTT conflict non-regression 决定。
```

注意：`reset-block start` 是 LoGeR 运行机制，不是 KITTI01 chunk id。它可以用，但不能写成 `chunk0` 特例。

#### 必须记录

```text
mode = INIT_SCALE / READ_RESCUE
reset_phase
probe metrics per mode
activation decision
oracle-positive diagnostic label only after run, not used at runtime
h10/h15 delta
```

#### 通过标准

```text
INIT_SCALE:
    在 reset-start positives 上命中，且不在普通 reset-start negatives 上大量误触发。

READ_RESCUE:
    在 chunk10-like disease-window positives 上命中，且 [400,600) regression <= +1m。

任何 mode 不能使用 absolute chunk id。
```

如果 mode-specific rules 都失败，semantic-conditioned C23 的部署路线降级。

---

### H5：distributed static scale-anchor 比单 masklet action 更可能带来 durable full improvement

#### 假设

单 masklet causal bank 弱，不代表分布式 static anchor 弱。要进入 Target-30，语义可能需要帮助建立跨 chunk 的 stable anchor set，而不是单点干预。

#### Static anchor 定义

构造 anchor eligibility：

```text
semantic in {road, building, wall, fence, stable structure}
masklet trust high
D_g semantic-z low
TTT conflict low
scale-risk low
temporal coverage stable
source attention mass high
```

令：

$$
A_{anchor}(i)=1[sem(i)\in S_{static}]\cdot 1[z_D(i)<q_D]\cdot 1[z_C(i)<q_C]\cdot 1[z_S(i)<q_S]\cdot Trust(i)
$$

其中 $q_D,q_C,q_S$ 使用 per-chunk robust quantile，不做数据集 tuning。

#### Actions

```text
frame/global:
    anchor tokens always source keep。

SWA:
    anchor tokens protected in overlap/local cache。

TTT:
    anchor tokens positive long write only if TTT conflict low。
```

#### 对照

```text
A0: no anchor protection
A1: semantic structure only
A2: structure + low z_D
A3: structure + low z_D + low TTT conflict
A4: structure + low z_D + low conflict + low scale risk + high trust
```

#### 通过标准

```text
h15 ATE delta <= -2m
or h15 [200,300) delta <= -4m
and [400,600) regression <= +1m
and SWA boundary regression <= +0.25m
```

如果 A4 明显优于 A1/A2，说明语义必须结合 cue。  
如果 A1 已经好，说明 semantic structure 本身强。  
如果全弱，static-anchor semantic route 降级。

---

## 4. 并行执行计划

本轮分成五个 Codex track 并行推进。任何 track 不能越过自己的 gate。

### Track 0：Mainline boundary 与 forbidden-feature audit

任务：

```text
1. 复现 C9_P0_R2 或加载可信 boundary。
2. 检查所有 runtime semantic policy 不含 trained trigger、learned selector、absolute chunk id。
3. 检查 semantic-conditioned C23 / paired probe 不 commit state。
4. 检查 paired probes 的 A/B state hash 不改变 committed state。
```

输出：

```text
phase0_training_free_audit.md
phase0_training_free_summary.json
forbidden_feature_scan.csv
noop_parity.csv
```

必须通过才能进入其他 track。

---

### Track A：Paired-probe value rule

任务：实现 original vs semantic-conditioned paired probe，输出 self-consistency metrics 和 deterministic gate。

候选 chunks：

```text
reset starts: 0,5,10,15,20,25,30
available H9/C9 parents
```

不训练，不拟合。只运行 fixed rule。

输出：

```text
paired_probe_metrics.csv
paired_probe_decisions.csv
probe_state_hash_audit.csv
probe_metric_delta_heatmap.png
```

如果 H1 gate fail，Codex 应先修 self-consistency metric 或 action magnitude，不允许训练 trigger。

---

### Track B：Semantic residual injection

任务：比较 original / replacement / fixed residual / paired-probe gated residual。

输出：

```text
semantic_residual_rollout_metrics.csv
D_base_D_sem_delta_stats.csv
semantic_residual_maps/
```

如果 semantic replacement strong 而 residual weak，Codex 应检查 energy normalization 和 $G_probe$，不允许扩大 threshold sweep。

---

### Track C：C9 compatibility decomposition

任务：拆解 C9 components 与 semantic residual 的冲突源。

输出：

```text
c9_component_matrix.csv
c9_component_deltas_by_segment.csv
c9_component_state_movement.csv
c9_compatibility_report.md
```

如果定位到冲突 component，Codex 自动切入对应 repair：

```text
TTT conflict:
    semantic read-only, block TTT write。

SWA conflict:
    semantic does not touch overlap SWA source, or key-preserve/value-attenuate。

read beta conflict:
    semantic residual only under paired-probe gate, no all-chunk activation。
```

---

### Track D：Mode-specific deterministic rules

任务：拆分 INIT_SCALE 与 READ_RESCUE 两个机制。

输出：

```text
mode_specific_probe_metrics.csv
init_scale_decisions.csv
read_rescue_decisions.csv
mode_oracle_audit_after_run.csv
```

如果两个 mode 都无法区分 positive / negative windows，semantic trigger route 降级。

---

### Track E：Distributed static scale-anchor

任务：用 semantic + cue 构造 distributed anchors，并测试其 path-specific memory effect。

输出：

```text
anchor_set_stats.csv
anchor_source_keep_summary.csv
anchor_swa_cache_summary.csv
anchor_ttt_write_summary.csv
anchor_rollout_metrics.csv
anchor_maps/
```

如果 anchor source mass 很低或与 original C23 attention mass 无关，Codex 应停止 anchor route，避免空跑 full。

---

## 5. 必须记录的统一指标

每个 run 必须记录：

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
50f_mean / 100f_mean / 200f_mean
boundary_10f_proxy
boundary_20f_proxy
chunk_boundary_pose_jump_proxy
```

每个 semantic run 必须额外记录：

```text
D_base_mean / p90 / mass_gt_0.5
D_sem_mean / p90 / mass_gt_0.5
D_delta_rms
semantic label mass
masklet trust mass
static_anchor_mass
highD_source_attention_mass
static_anchor_source_attention_mass
SWA overlap source mass
TTT update_conflict_energy
TTT role mass
context_empty_source_events
```

每个 paired probe 必须记录：

```text
probe_A_metrics
probe_B_metrics
metric_delta
improve_count
regress_count
hard_guard_flags
G_probe
state_hash_before
state_hash_after_A
state_hash_after_B
```

---

## 6. 可视化要求

每个通过 h10 gate 的候选必须生成以下可视化：

```text
1. D_base / D_sem / D_delta map overlay。
2. semantic fine label map + masklet trust overlay。
3. static anchor map。
4. high-risk negative map。
5. attention mass before/after semantic residual。
6. SWA overlap source keep/drop map。
7. TTT conflict map / role mass heatmap。
8. per-chunk ATE curve。
9. [200,300) and [400,600) segment error bar chart。
10. h10 -> h15 durability plot。
```

如果没有可视化，不允许 promotion 到 full online。

---

## 7. Promotion Gates

### Gate 0：Training-free hard gate

```text
No trained trigger.
No learned selector.
No oracle label fitting.
No absolute chunk id.
No dataset-specific threshold.
No GT runtime action.
No offline trajectory rewrite.
```

Fail 则该 row invalid。

### Gate 1：Paired probe gate

```text
state hash no-commit exact
oracle-positive diagnostic recall >= 75% after run
false positive <= 25%
no absolute chunk id
```

Oracle labels 只用于事后评估，不能用于 rule 构造。

### Gate 2：Short rollout h10/h15 gate

```text
h10:
    ATE delta <= -1.5m
    or [200,300) delta <= -3m
    with [400,600) regression <= +1m

h15:
    ATE delta <= -3m
    or [200,300) delta <= -5m
    durability_ratio >= 0.45
    [400,600) regression <= +1m
    SWA boundary regression <= +0.25m
```

### Gate 3：C9 compatibility gate

```text
candidate vs C9:
    h15 ATE delta <= -1.5m
    or h15 [200,300) delta <= -3m
    and [400,600) regression <= +1m
```

### Gate 4：Full online gate

```text
Stage success:
    full online ATE <= 32m
    no segment catastrophic regression

Target-30 success:
    full online ATE <= 30m
    deployable runtime only
    no GT / no offline rewrite / no trained trigger
```

---

## 8. Codex 失败自动分流

### 情况 A：paired-probe metrics 无法识别 oracle positive

Codex 不得训练 trigger。应尝试：

```text
1. 增加 no-GT self-consistency metrics，如 overlap residual、scale-step consistency、attention mass。
2. 检查 semantic residual magnitude 是否过小。
3. 检查 D_sem 是否与 D_base 太接近。
4. 检查 state hash，确保 probe 没 commit 或污染。
5. 若仍失败，停止 runtime activation route。
```

### 情况 B：semantic replacement 有效，residual injection 无效

Codex 应尝试：

```text
1. 检查 energy normalization 是否过强。
2. 使用 fixed residual 0.25 / 0.35 作为预注册 ablation。
3. 检查 semantic z map 是否被 clip 到无效。
4. 不允许改成 trained scale。
```

### 情况 C：C9 + semantic 仍比 C9 差

Codex 应拆 component：

```text
1. semantic read-only。
2. block semantic from TTT write。
3. block semantic from SWA overlap source。
4. keep semantic only in frame/global read。
5. 找出冲突源后只修对应 path。
```

### 情况 D：SWA 局部改善但 boundary 回退

Codex 应改成：

```text
1. key preserve / value attenuation。
2. overlap anchor protection。
3. semantic skip only on non-overlap source。
4. 禁止 hard remove SWA overlap anchors。
```

### 情况 E：h10 强但 h15 弱

Codex 应做 washout attribution：

```text
1. TTT state movement。
2. SWA cache movement。
3. global source movement。
4. merge/gauge movement。
5. 判断 correction 被哪条 path 洗掉。
```

如果无法定位 washout，则不允许 full online。

### 情况 F：所有 semantic-conditioned routes 失败

Codex 应执行降级决策：

```text
Semantic Prior Generator -> diagnostic / trust calibration / weak regularizer。
Target-30 主线转向 explicit trajectory-state / scale-state 或 TTT-native cue action。
```

---

## 9. 资源与并行策略

```text
1. Phase 0 / paired probe 可以 4-6 并发，但每进程绑定单 GPU。
2. full online 默认最多 4 并发，避免 host RAM / IO 崩。
3. short rollout 先 h10，只有过 gate 才 h15。
4. h15 过 gate 才 full online。
5. 所有 stale run dirs 必须 rename 到 .INVALID_*，不能混 JSONL。
6. 每个 run 必须写 run_config.yaml 和 boundary flags。
```

---

## 10. 本轮最终决策逻辑

本轮结束后必须给出下面四种结论之一：

### 结论 1：training-free semantic cue 成功进入 Target-30

条件：

```text
C9 + deterministic semantic policy full online ATE <= 30m。
```

### 结论 2：semantic cue 有 stage-level value，但未达 Target-30

条件：

```text
full online ATE <= 32m but >30m。
```

下一步继续围绕 deployable policy 做 cross-seq diagnostic。

### 结论 3：semantic cue 只有 local diagnostic value

条件：

```text
h10/h15 strong but full online fail。
```

下一步只保留为 failure diagnosis，不继续主线。

### 结论 4：semantic cue 路线降级

条件：

```text
paired-probe fail
or C9 compatibility fail
or no h15 gate pass。
```

Semantic Prior Generator 不再作为 Target-30 主线，只作为 masklet trust / visualization / auxiliary regularizer。


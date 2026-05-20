# ACL2 v10：面向 KITTI01 ATE 25 的 TTT Drift Action Space 与并行验证实验计划

版本：2026-05-18  
主序列：KITTI Odometry Sequence 01  
新目标：`KITTI01 ATE RMSE <= 25.0m`  
当前参考 best：`V9_H9_READBETA2_03_body485_exit425_c16_425_SWKS3 = 34.1258 / 6.5414`  
当前差距：`34.1258 - 25.0 = 9.1258m`

---

## 0. 这份计划的定位

这份计划不是继续在当前 `gamma / neutral / read_beta` 附近做小修小补。当前实验已经说明，`update_conflict_energy + tri_replay`、`chunk16 weak tri-replay`、`exit/c16 read beta cooling` 都有真实信号，但它们已经进入 `34.1m` 平台。新目标是 `ATE <= 25m`，这要求至少再下降 `9.1m`，不是靠 `0.002m`、`0.03m` 级局部调参能完成的。

因此 v10 的目标是重新验证一个更本质的问题：

> **TTT 写入控制到底有没有足够 action space 修复 whole-scene drift？如果有，必须找到能作用于 trajectory drift direction 的写入方式；如果没有，应尽快从 TTT write-only 转向 pose-scale / read-side / memory-lifetime 结构。**

本计划把下一阶段拆成三个并行方向：

```text
Lane A: TTT true projection oracle
    验证 TTT fast-weight update 是否存在可被控制的 trajectory-drift 方向。

Lane B: no-GT window drift policy
    如果 oracle 有上界，构造不依赖 GT 的 drift proxy 和自动 action policy。

Lane C: non-TTT drift fallback
    如果 TTT oracle 没有足够上界，转向 read-side support / pose-scale correction / chunk alignment，而不是继续写入标量细扫。
```

同时，本计划要求每个实验都必须用 **global drift dashboard** 判断是否真的减轻 whole-scene drift，而不是只优化 chunk 内或某个局部 segment。

---

## 1. 当前实验结果的独立判断

### 1.1 已经有进展，但进展越来越慢

从早期到现在，进展大致分成三次大跳：

```text
LoGeR native / old HMC baseline: 约 41.75m
frame-attention C23 pair/all:    约 36.68m
TTT tri-replay / update_conflict: 约 34.19m
v8/v9 微调平台:                 约 34.13m
```

这说明项目不是没有进展。真正的大信号来自两处：

1. `C23 past` 的 frame-attention read cue，尤其是 pair/all read intervention；
2. `update_conflict_energy + tri_replay`，说明 TTT 自身 cue 对长期轨迹平衡有一定作用。

但是最近几轮的进度非常慢：

```text
C16ROLE_01:      34.1583
READBETA_01:     34.1282
H9_READBETA2_03: 34.1258
```

从 v8 best 到 v9 best 只提升 `0.0325m`，H9 相比 H8 只提升 `0.0024m`。这说明当前可用的 action space 已经非常窄。

### 1.2 当前 best 不是好的 TTT 写入策略

当前 best `H9_READBETA2_03` 的组成是：

```text
write-side:
    chunks 5-9 gamma 0.005, role 0.35/0.12/0.85
    chunks 10-12 gamma 0.003, role 0.35/0.12/0.85
    chunk16 gamma 0.0003, role 0.35/0.08/0.85
read-side:
    body 5-9 beta = 4.85
    exit 10-12 beta = 4.25
    c16 beta = 4.25
    base beta = 4.75
```

这已经不是纯 TTT 写入策略，而是 **TTT windowed tri-replay + read-side beta cooling** 的混合结果。它的收益很小但分布健康：`[200,300)`、`[200,400)`、`[400,600)` 和 FinalErr 都略好于 `C16ROLE_01`。不过这仍然不是目标级别的漂移修复。

当前 best：

```text
ATE = 34.1258
[200,300) = 74.410
[200,400) = 54.651
[400,600) = 44.354
RPE_t = 92.4053
```

这里最关键的是：`RPE_t` 基本没有动。也就是说，当前策略没有真正改善局部相对平移结构；它主要是在 Sim3 对齐后的全局形状、窗口间误差分配和 read/write coupling 上做微调。

### 1.3 当前改善更像 reset-window 级别平衡，不是 whole-scene drift correction

`update_conflict_energy + tri_replay` 从 `36.4m` 推到 `34.2m`，这不是纯 chunk 内优化；它确实改变了 reset-window 级别的长期轨迹平衡。但 v8/v9 后续实验显示：

```text
1. body/exit gamma 已经收口；
2. neutral_lambda 已经收口；
3. chunk16 gamma 只有极弱窗口有用；
4. read beta 只能带来 0.03m 级收益；
5. oracle-like scalar role/gamma 没打开新空间；
6. projection oracle 尚未真正接入，不能伪装成已验证。
```

所以当前卡住的不是“哪个 chunk 的 gamma 还没扫干净”，而是：

> **我们仍然没有能直接作用于 trajectory drift direction 的 TTT action。**

### 1.4 离目标有多远

新目标是：

```text
KITTI01 ATE <= 25.0m
```

当前 best 是：

```text
34.1258m
```

差距：

$$
Gap_{25}=34.1258-25.0=9.1258m
$$

相对当前误差比例：

$$
RelativeGap=\frac{9.1258}{34.1258}\approx 26.7\%
$$

这不是小幅优化目标，而是要求下一阶段找到新的机制上界。v10 不应再用 `34.10` 当主目标，而要先判断是否存在能把轨迹推到 `<=30m`、甚至接近 `25m` 的 action family。

---

## 2. 核心问题：到底卡在哪里？

### 2.1 卡在 action space，而不只是 cue

现在已有的 `update_conflict_energy` 是有用 cue。它能把 TTT 写入从 `36m` 平台推到 `34m` 平台。但当前 action 基本还是：

```text
chunk 是否启用 tri-replay
negative gamma 多大
positive / negative / neutral fraction 多大
某个 chunk 的 read beta 多大
某个 layer/head 是否加权
```

这些都是低维标量动作。它们可以调局部平衡，但很难直接修正 trajectory drift direction。

当前应该验证的是：

> **TTT fast-weight update 中是否存在一个可投影、可分离、可控制的 drift-correction subspace？**

如果存在，就要做 projection / direction-aware replay。  
如果不存在，就要承认 TTT write-only 不是通往 ATE 25 的主路线。

### 2.2 卡在局部指标和全局漂移指标之间的错配

当前很多候选改善：

```text
Rot
FinalErr
YawRMSE
[200,300)
```

但同时损伤：

```text
[400,600)
整体 ATE
Sim3 scale
long-window continuity
```

这说明我们面对的不是单点污染，而是 **drift-state redistribution**。一个 action 修了 body window，可能伤 exit / handoff / post-window。反过来，exit cooling 可能改善后段，却不能大幅修 body。

因此 v10 需要把评估单位从单 chunk 改成 reset-window 和 downstream window。

### 2.3 卡在没有真正的 oracle 上界

v9 计划里原本要做 `GT-aligned window drift projection oracle`，但实际没有接入有效 full run：当前 online TTT controller 没有 token update 到 pose residual projection 的 action path；已有 `update_conflict_energy` 只是相对 chunk aggregate fast-weight update 的 projection，不等同于 GT drift projection oracle。

所以现在我们还不知道：

```text
TTT write action 如果有真正的 drift projection，上界能到多少？
是 33m、30m、还是 25m？
```

这个问题必须先回答，否则继续做 no-GT proxy 没意义。

---

## 3. v10 总目标与阶段目标

### 3.1 最终目标

```text
Final target:
    KITTI01 ATE RMSE <= 25.0m
```

### 3.2 阶段目标

为了避免一直追最终目标导致决策混乱，v10 设三层 gate：

```text
Stage Gate A:
    ATE <= 33.5m
    且 [200,300) <= 72m
    且 [400,600) <= 44.5m

Stage Gate B:
    ATE <= 31.5m
    且 [200,300) <= 65m
    且 [400,600) <= 43.5m

Final Gate:
    ATE <= 25.0m
    且 [200,300) <= 50m
    且 [400,600) 不明显回退
    且 KITTI00/02/05 sanity 不崩
```

如果某路线连 Stage Gate A 都无法接近，就不应继续大规模细扫。

---

## 4. 实验总原则

### 4.1 不再做纯标量微扫

以下实验族暂停：

```text
body gamma 0.0048 / 0.0052
exit gamma 0.0028 / 0.0032
neutral 0.83 / 0.87
c16 gamma 0.00025 / 0.00035
read beta 4.80 / 4.90 单独细扫
```

除非某个新机制通过 oracle gate，否则这些微扫不允许继续占用 full-run 资源。

### 4.2 每个新机制必须先做 offline / smoke / oracle

Full KITTI01 run 成本约 30 分钟，不能让 Codex 直接把每个想法都跑 full。新机制的进入顺序固定为：

```text
1. 静态检查 / py_compile / bash -n
2. END_FRAME=128 smoke，验证 action 非 no-op
3. offline dashboard，确认指标被正确记录
4. oracle 或 proxy 判别 gate
5. top 2-4 个候选进入 full KITTI01
```

### 4.3 每个 full run 必须报告 whole-scene drift，而不是单 ATE

任何 full run 必须记录：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
[200,300)
[200,400)
[400,600)
ATE_50_mean / worst
ATE_100_mean / worst
ATE_200_mean / worst
YawRMSE
Sim3Scale
per-axis RMSE: x/y/z
reset-window error: 0-4, 5-9, 10-14, 15-19, 20-24, ...
```

### 4.4 并行验证，但严格 stopping rule

每个 batch 最多 4 个 full run 并行。超过 4 并发容易 OOM 或污染可复现性。每个实验族最多两批 full run；如果两批内没有超过当前 best 至少 `0.20m`，则该族停止。

---

## 5. 核心假设与实验设计

---

## H1：当前平台是 action space 不足，而不是 cue 不足

### 假设

`update_conflict_energy` 已经有足够诊断价值，但当前 `gamma / neutral / role / beta` action 无法直接作用于 trajectory drift direction，因此只能得到 `34.1m` 平台。

### 实验设计

不新增 full run，先整理已有 v7-v9 数据，构建 action-space saturation report。

需要汇总的 run：

```text
B0_SWKS3
WINGAM_03_repeat
C16ROLE_01
READBETA_01
H9_READBETA2_03
AUTO_WIN_05/06
POSTREG_11
C16FINE_03
H6_ORACLE_03/04
```

### 记录指标

```text
run_id
family
action_type
ATE
Rot
RPE_t
FinalErr
[200,300)
[400,600)
Sim3Scale
improvement_vs_prev_best
best_metric_type
```

并计算：

$$
Gain_{ATE}=ATE_{baseline}-ATE_{candidate}
$$

$$
Tradeoff_{post}=E_{400,600}^{candidate}-E_{400,600}^{baseline}
$$

$$
DiseaseGain=E_{200,300}^{baseline}-E_{200,300}^{candidate}
$$

### 成立标准

如果满足：

```text
1. 最近 10 个同族变体的 best gain < 0.05m；
2. 不同标量 action 的收益集中在 Rot/FinalErr 或局部 segment；
3. [200,300) 和 [400,600) 呈现明显 trade-off；
4. RPE_t 基本不变；
```

则 H1 成立，后续停止 scalar 微扫，转入 H2/H3。

### 不满足时 Codex 尝试方向

如果发现某个 action family 仍有 `>=0.20m` 的单族收益趋势，Codex 可以继续做一批局部验证，但必须限制为：

```text
最多 4 条 full run；
每条必须解释 why not scalar plateau；
如果 best < current_best - 0.20m，才允许第二批。
```

---

## H2：必须先建立 true TTT projection oracle，判断 TTT write-side 上界

### 假设

如果 TTT 写入能通向 ATE 25，那么在 oracle 条件下，应该存在一个 fast-weight update projection action，能显著降低 whole-scene drift。这里的 oracle 不是 `update_conflict_energy`，而是使用 GT trajectory residual 构建 window drift direction，再把 TTT token/layer/branch update 投影到这个方向上。

### 工程目标

实现一个真正的 oracle projection action path：

```text
per-token / per-layer / per-branch update contribution
        ↓
window-level GT trajectory residual direction
        ↓
project update contribution into helpful / harmful components
        ↓
commit only helpful component, suppress or reverse harmful component
```

### 必须新增 instrumentation

Codex 先实现以下落盘，不跑 full 控制：

```text
per_token_update_group.pt
per_layer_branch_update_matrix.pt
tri_replay_role_mass.jsonl
window_pose_residual_gt.json
window_drift_direction_gt.pt
update_to_drift_projection.csv
```

最小字段：

```text
chunk_id
layer_id
branch_id
head_id
token_group
role: pos / neg / neutral
update_norm
update_cos_to_native
update_cos_to_window_drift
projection_helpful_energy
projection_harmful_energy
projection_orthogonal_energy
```

定义：

$$
J_{i,l,b}=\text{token }i\text{ 对 layer }l, branch b\text{ 的 pre-zeropower update}
$$

$$
d_w=\text{window-level GT drift direction surrogate}
$$

$$
P_{help}(i,l,b)=\max\left(0,\cos(J_{i,l,b}, d_w)\right)\|J_{i,l,b}\|
$$

$$
P_{harm}(i,l,b)=\max\left(0,-\cos(J_{i,l,b}, d_w)\right)\|J_{i,l,b}\|
$$

注意：这里 $d_w$ 不是 fast-weight 空间的天然向量，因此实现时可以使用一个可计算 surrogate：

```text
候选 1: token update 对 next-window pose increment disagreement 的有限差分影响
候选 2: update group leave-one-out 后的 window ATE delta
候选 3: fitted linear surrogate 从 update feature -> window residual
```

### Oracle 控制实验

只有 smoke 通过后，进入 4 条 full：

```text
ORACLEPROJ_01:
    scope = chunks 5-12
    action = suppress harmful projection only
    strength = 0.50

ORACLEPROJ_02:
    scope = chunks 5-12 + chunk16
    action = suppress harmful + keep helpful
    strength = 0.50

ORACLEPROJ_03:
    scope = chunks 5-12
    action = reverse harmful projection
    strength = 0.25

ORACLEPROJ_04:
    scope = chunks 5-12 + chunk16
    action = helpful boost + harmful suppress
    helpful boost = 1.10
    harmful suppress = 0.50
```

### 记录指标

除标准 trajectory 指标外，必须记录：

```text
oracle_projection_energy_total
oracle_helpful_energy_committed
oracle_harmful_energy_removed
oracle_harmful_energy_reversed
projection_cos_by_layer_branch
projection_energy_by_window
per-window ATE delta
```

### 假设成立标准

H2 强成立：

```text
ATE <= 31.5
或 [200,300) <= 60 且 [400,600) <= 44.5
```

H2 弱成立：

```text
ATE <= 33.5
且 [200,300) <= 72
且 [400,600) 不超过 current best + 1.0m
```

H2 不成立：

```text
oracle projection best ATE > 33.8
或只改善 Rot / FinalErr
或 [200,300) 与 [400,600) 仍强 trade-off
```

### 不满足时 Codex 尝试方向

如果 oracle 接线困难：

```text
Fallback A: leave-one-layer-branch replay oracle
    对每个 layer/branch/group 做 leave-one-out replay，估计 helpful/harmful。

Fallback B: update-feature linear surrogate oracle
    用已有 70 runs 拟合 update feature -> ATE / [200,300) / [400,600)，选 top projection。

Fallback C: stop TTT projection
    如果 A/B 都不能给出 >1m oracle gain，则停止 TTT write-only，转 H5 pose-scale/read-side。
```

---

## H3：如果 oracle 有上界，no-GT policy 必须从 window-drift proxy 预测 action，而不是 raw conflict peak

### 假设

`update_conflict_energy` 的 raw peak 会选到 reset-like / posture chunks，不能直接决定 body/exit/action。有效 no-GT policy 应该预测：

```text
local disease gain
minus downstream cost
```

而不是只看当前 chunk conflict。

### 实验设计

先不跑 full，训练或拟合一个 lightweight action scorer：

$$
Score(m)=\hat G_{local}(m)-\lambda \hat C_{downstream}(m)
$$

其中：

$$
\hat G_{local}=f_{local}(x_m)
$$

$$
\hat C_{downstream}=f_{downstream}(x_m)
$$

features $x_m$ 包括：

```text
update_conflict_energy mean/q90/max
w0 delta norm spike
layer12/head0 conflict mass
tri-replay neg/neutral/pos mass
memory rel diff
pass1/pass2 pose increment disagreement
read attention entropy / high-D mass
Sim3 scale proxy shift
reset-relative chunk index
previous window correction residual
```

### 离线验证

用已有 completed runs 做 leave-one-run-family-out 验证：

```text
train: v7-v8-v9 existing runs except one family
valid: held-out family
```

必须输出：

```text
surrogate_spearman_global
surrogate_top5_recall
local_gain_prediction_corr
post_cost_prediction_corr
action_rank_table.csv
```

### Full 候选

只有离线 gate 通过，才跑 top 4：

```text
NOGT_01: predicted best conservative
NOGT_02: predicted best aggressive
NOGT_03: predicted best post-safe
NOGT_04: predicted best disease-focused
```

### 成立标准

离线 gate：

```text
Spearman(score, ATE) >= 0.75
Top-5 recall >= 0.60
post_cost_prediction_corr >= 0.50
```

Full gate：

```text
ATE <= 33.5
或相比 current best 至少改善 0.30m
且 [400,600) 不恶化超过 1.0m
```

### 不满足时 Codex 尝试方向

如果 no-GT proxy 不过：

```text
1. 不允许直接跑更多 full no-GT variants。
2. Codex 改做 feature audit，输出哪个 feature 误导 action。
3. 若 reset-relative false positive 仍高，加入 reset-group normalization：

   x'_m = (x_m - mean_reset_group) / (std_reset_group + eps)

4. 若仍不通过，转 H5，不再把 TTT auto-window 作为主线。
```

---

## H4：如果 TTT projection 成立，必须做 long/short lifetime，而不是单一长期 fast weight

### 假设

当前 TTT action 之所以卡住，是因为同一个 update 同时包含：

```text
long-term continuity
short-term correction
drift-harmful component
```

把它们都写入同一个长期 $W$，会产生 body/post trade-off。需要把 fast weights 拆成长期和短期 overlay。

### 设计

定义：

$$
W_{apply}=W_{long}+\alpha_t W_{short}
$$

$$
W_{commit}=W_{long}
$$

其中：

```text
W_long 接受 helpful / continuity update；
W_short 接受短期 correction 或 uncertain update；
W_short 在 K 个 chunk 后 decay；
W_short 不进入长期 commit。
```

### 实验矩阵

只在 H2 oracle 成立后做：

```text
LIFE_01:
    W_short = harmful projection correction
    K = 1
    alpha_t = 0.50

LIFE_02:
    W_short = harmful projection correction
    K = 2
    alpha_t = 0.50

LIFE_03:
    W_short = neutral uncertain update
    K = 2
    alpha_t = 0.25

LIFE_04:
    W_long = helpful only
    W_short = all correction residual
    K = 3
    alpha_t = 0.25
```

### 记录指标

```text
W_long_update_norm
W_short_update_norm
W_short_decay_curve
apply_short_contribution_norm
commit_long_only_hash
per-window short contribution to pose proxy
```

### 成立标准

```text
ATE <= projection oracle best + 0.20m
且 [200,300) 不回退
且 [400,600) 明显优于 pure projection
```

### 不满足时 Codex 尝试方向

如果 dual lifetime 失败：

```text
1. 检查 W_short 是否被 apply path 实际消费；
2. 如果消费过弱，提高 alpha_t，不改 commit；
3. 如果消费过强导致 [400,600) 退化，缩短 K；
4. 如果所有 K/alpha 都失败，说明短期 overlay 不是 LoGeR 当前 TTT 的有效控制面，停止。
```

---

## H5：如果 TTT oracle 不足，必须转向 pose-scale / read-side drift，而不是继续 TTT write

### 假设

若 H2 oracle 都不能给出 `<=33.5m` 或显著降低 `[200,300)`，说明 TTT write-side action 对目标 25 没有足够上界。此时应转向：

```text
1. read-side support selection / per-head cue；
2. reset-window pose-scale correction；
3. chunk overlap alignment / local Sim3 handoff；
4. trajectory state correction outside TTT fast weights。
```

### Pose-scale oracle

先做一个不改变模型的 diagnostic oracle：

```text
POSEORACLE_01:
    per reset-group Sim3 alignment to GT

POSEORACLE_02:
    per body/exit/handoff window Sim3 alignment to GT

POSEORACLE_03:
    only correct scale, not rotation

POSEORACLE_04:
    only correct yaw/translation drift, not scale
```

目的不是作为方法，而是判断目标 25 的误差主要来自：

```text
scale drift
yaw drift
translation shape
local odometry
```

### no-GT pose-scale proxy

如果 oracle 显示 pose-scale correction 能接近 25，Codex 做 no-GT proxy：

```text
overlap pointmap consistency
SWA overlap pose discontinuity
read attention entropy jump
TTT apply mismatch jump
local scale proxy from chunk pointmap depth
```

### 成立标准

Pose-scale oracle 强成立：

```text
ATE <= 25.0
```

Pose-scale oracle 弱成立：

```text
ATE <= 30.0
且能明确归因为 scale/yaw/translation 某一类。
```

如果 pose oracle 也不能接近 25，则说明目标 25 可能需要 read cue / geometry output 本身更强，而非 memory controller。

### 不满足时 Codex 尝试方向

```text
如果 scale-only oracle 有效：
    Codex 实现 reset-window scale proxy correction。

如果 yaw-only oracle 有效：
    Codex 实现 overlap yaw consistency correction。

如果 Sim3 oracle 有效但单项无效：
    Codex 实现 lightweight window alignment module，不再约束为 TTT write。

如果所有 pose oracle 都无效：
    转 read cue / feature extraction，优先 per-head / support selection。
```

---

## H6：read-side 不能继续 beta 微扫，必须转 support / per-head / conditional policy

### 假设

H8/H9 已经证明 exit/body read beta cooling 只有 `0.03m` 级收益。继续调 beta 不足以到 25。read-side 若要继续，应改变 cue 或 support，而不是调 scalar beta。

### 实验设计

只在 H2/H5 决策后执行。候选：

```text
READSUP_01:
    C23 past_only -> reset-relative past_only
    对 body/exit 使用不同 support window

READSUP_02:
    C23 per-head top2 / top4
    只在 pair/all read path，不改 TTT write

READSUP_03:
    body 用 C23 past，exit 用 future diagnostic 或 deep static rescue

READSUP_04:
    beta 由 no-GT window risk 自动调度，而不是手工 chunks 5-9/10-12/16
```

### 记录指标

```text
read_beta_by_chunk
cue_mass_by_chunk
attention_entropy_before_after
attention_mass_to_highD
support_indices_by_frame
per-head cue agreement
```

### 成立标准

```text
ATE <= 33.5
或相比 current best 改善 >= 0.30m
且不是单纯 Rot trade-off
```

### 不满足时 Codex 尝试方向

```text
若 per-head 失败：回到 headmean，不继续 per-head。
若 support selection 有效但过拟合 KITTI01：立即跑 KITTI00/02/05 sanity。
若 conditional beta 有效但不稳定：用 surrogate 降维成 2-3 个 discrete modes。
```

---

## 6. 统一指标记录规范

### 6.1 Global metrics

每个 full run 必须输出：

```text
ATE_RMSE
Rot_RMSE
RPE_t
RPE_r
FinalErr
ATE_50_mean
ATE_50_worst
ATE_100_mean
ATE_100_worst
ATE_200_mean
ATE_200_worst
YawRMSE
Sim3Scale
GapTo25 = ATE_RMSE - 25.0
```

### 6.2 Segment metrics

固定记录：

```text
[0,100)
[100,200)
[200,300)
[200,400)
[300,400)
[400,500)
[400,600)
[500,600)
```

reset-group：

```text
chunks 0-4
chunks 5-9
chunks 10-14
chunks 15-19
chunks 20-24
chunks 25-29
chunks 30-34
chunks 35-37
```

### 6.3 Drift decomposition

新增：

```text
axis_rmse_x
axis_rmse_y
axis_rmse_z
cumulative_translation_drift
cumulative_yaw_drift
scale_proxy_by_window
pose_increment_disagreement
reset_boundary_pose_jump
```

### 6.4 TTT internals

```text
chunk_id
layer_id
branch_id
head_id
update_conflict_energy_mean/q90/max
update_norm_native
update_norm_candidate
update_cos_to_native
update_cos_to_window_drift
pos_mass
neg_mass
neutral_mass
harmful_projection_energy
helpful_projection_energy
orthogonal_projection_energy
memory_rel_diff
zeropower_distortion_ratio
apply_mismatch_mean/q90
```

### 6.5 Read-side internals

```text
read_beta_effective_by_chunk
cue_mass_gt_0.5
attention_entropy_before_after
attention_mass_to_highD
attention_mass_to_anchor
support_pattern
per_head_agreement
```

---

## 7. 必须可视化的内容

### 7.1 Global drift dashboard

每个晋级候选必须画：

```text
trajectory XY full
trajectory XY zoom [200,400]
trajectory XY zoom [400,600]
per-frame error curve
per-100f ATE bar
per-reset-window ATE bar
Sim3 scale over time
yaw drift over time
axis-wise error over time
```

### 7.2 TTT action dashboard

```text
chunk × layer × branch update_conflict_energy heatmap
chunk × layer × branch projection_harmful_energy heatmap
chunk × layer × branch projection_helpful_energy heatmap
role mass timeline: pos / neg / neutral
gamma / read beta schedule timeline
memory_rel_diff timeline
zeropower distortion timeline
```

### 7.3 Trade-off dashboard

必须比较：

```text
current best H9_READBETA2_03
C16ROLE_01
WINGAM_03_repeat
new candidate
```

图：

```text
x-axis: [200,300) improvement
 y-axis: [400,600) cost
point color: ATE
point size: FinalErr
```

### 7.4 Oracle/proxy dashboard

```text
predicted score vs actual ATE scatter
predicted local gain vs actual [200,300) gain
predicted downstream cost vs actual [400,600) cost
Top-k action recall plot
```

---

## 8. 并行执行计划

### Batch 0：整理与验证，不跑 full

目标：确认 current best、指标、dashboard、registry 全部统一。

Codex tasks：

```text
C0-1: 生成 action_space_saturation_report.md
C0-2: 生成 v9_global_drift_dashboard_current_best/
C0-3: 检查 H9_READBETA2_03 是否所有 metrics 齐全
C0-4: 检查 full-run script 4 并发安全配置
```

Gate：

```text
如果 H9 best 的 trajectory diagnostics 缺失，先补 diagnostics，不跑新 full。
```

---

### Batch 1：True projection oracle instrumentation

不跑 full，先实现与 smoke。

Codex tasks：

```text
C1-1: 在 ttt_write_controller.py 中导出 per-token pre-zeropower update group。
C1-2: 在 hybrid_memory_controller.py 中记录 tri_replay role mass by layer/branch/head。
C1-3: 新增 tools/v10_ttt_projection_audit.py。
C1-4: END_FRAME=128 smoke，确认 action 非 no-op 且 debug 完整。
```

Gate：

```text
per-token update group present
projection_energy 不全为 0
smoke candidate 与 baseline pose diff 非 0
无 NaN / no OOM
```

不满足时：

```text
Codex fallback to leave-one-layer-branch replay audit。
```

---

### Batch 2：Projection oracle full, 4 并发

候选：

```text
ORACLEPROJ_01
ORACLEPROJ_02
ORACLEPROJ_03
ORACLEPROJ_04
```

Gate：

```text
如果 best ATE <= 33.5 或 [200,300) <= 72，进入 Batch 3。
如果 best ATE > 33.8，停止 TTT projection 主线，进入 Batch 5 pose-scale oracle。
```

---

### Batch 3：No-GT proxy policy

仅在 Batch 2 通过时执行。

Codex tasks：

```text
C3-1: fit local/downstream action surrogate
C3-2: generate top 20 actions offline
C3-3: select 4 actions by diversity and predicted score
```

Full candidates：

```text
NOGT_01..04
```

Gate：

```text
ATE <= 33.5 或 improvement >= 0.30m
否则停止 no-GT proxy。
```

---

### Batch 4：Dual-lifetime TTT

仅在 Batch 2/3 证明 projection 有上界时执行。

候选：

```text
LIFE_01..04
```

Gate：

```text
比 projection best 的 [400,600) 更好，且 ATE 不差超过 0.20m。
```

---

### Batch 5：Pose-scale oracle fallback

如果 TTT projection 上界不足，立即执行。

候选：

```text
POSEORACLE_01 per-reset Sim3
POSEORACLE_02 per-body/exit/handoff Sim3
POSEORACLE_03 scale-only
POSEORACLE_04 yaw/translation-only
```

Gate：

```text
如果任何 oracle <= 25m：
    说明目标 25 是 trajectory-state / pose-scale 问题，进入 no-GT pose proxy。

如果 oracle 只到 30-33m：
    说明需要 read + pose 联合，不再 TTT-only。

如果 oracle 也 >33m：
    说明当前 geometry output / read cue 本身不足，转 per-head/read support。
```

---

### Batch 6：Read-side support / per-head conditional policy

仅在 TTT/pose 决策后执行。

候选最多 4 条：

```text
READSUP_01
READSUP_02
READSUP_03
READSUP_04
```

Gate：

```text
ATE <= 33.5 或 improvement >= 0.30m
否则停止 read beta/support 小扫。
```

---

### Batch 7：Cross-sequence sanity

只有 KITTI01 达到：

```text
ATE <= 33.5
或单机制 improvement >= 0.50m
```

才跑：

```text
KITTI00 full
KITTI02 full
KITTI05 full
```

候选：

```text
current best H9
new best v10
baseline C16ROLE_01
```

Gate：

```text
平均 ATE 不差于 H9
且无单序列 regression > 5%
```

---

## 9. 停止规则

### 9.1 标量 action 停止

如果一个实验族连续 4 条 full run：

```text
best improvement < 0.05m
```

则停止该族。

### 9.2 oracle 停止

如果 oracle best：

```text
ATE > 33.8
且 [200,300) > 72
```

则停止该 oracle family。

### 9.3 read-side 停止

如果 read-side 新机制只能：

```text
改善 Rot / FinalErr
但 ATE 不改善 >=0.20m
```

则停止。

### 9.4 TTT write-only 停止

如果 true TTT projection oracle 不成立，则停止 TTT write-only 作为 target-25 主线，保留 TTT 仅作 auxiliary regularizer。

---

## 10. Codex 并行工作清单

Codex 可以并行开 6 个任务，但 full run 只允许 4 并发。

### Codex Task A：projection instrumentation

```text
files:
    loger/pipeline/ttt_write_controller.py
    loger/pipeline/hybrid_memory_controller.py
outputs:
    per_token_update_group.pt
    projection_debug.jsonl
```

失败时尝试：

```text
leave-one-layer-branch replay audit
```

### Codex Task B：dashboard 工具

```text
tools/v10_global_drift_dashboard.py
tools/v10_projection_dashboard.py
```

必须输出所有图，不允许只输出表格。

### Codex Task C：surrogate policy

```text
fit local/downstream predictors
rank candidate actions
produce top-20 + diversity top-4
```

失败时尝试：

```text
reset-group normalized features
ridge / random forest / monotonic scoring
```

### Codex Task D：pose oracle

```text
per-reset Sim3
per-window Sim3
scale-only
yaw/translation-only
```

失败时尝试：

```text
Umeyama robust trim
window overlap-only alignment
```

### Codex Task E：runner and registry

```text
unified run_registry.csv
config hash
software commit
GPU / walltime / OOM records
```

### Codex Task F：report automation

```text
每批自动生成:
    batch_summary.md
    promotion_decision.md
    failed_family_reason.md
```

---

## 11. 最终决策树

### 情况 A：TTT projection oracle 达到 <=31.5m

结论：TTT write-side 仍是正确主线。

下一步：

```text
1. no-GT proxy
2. dual-lifetime
3. cross-sequence
4. 继续向 25m 推进
```

### 情况 B：TTT oracle 只能到 33-34m

结论：TTT write-side 是辅助 regularizer，不是 target-25 主线。

下一步：

```text
1. pose-scale oracle
2. read-side support/per-head
3. TTT 仅保留 C16ROLE/H9 style auxiliary
```

### 情况 C：pose-scale oracle 能到 <=25m

结论：目标 25 主要是 trajectory state / scale-yaw correction 问题。

下一步：

```text
1. no-GT overlap scale/yaw proxy
2. reset-window alignment module
3. TTT write 只作为 input to proxy
```

### 情况 D：pose-scale oracle 也不能到 <=30m

结论：当前 geometry output 或 read cue 本身不足。

下一步：

```text
1. per-head read cue
2. true attention-map statistics
3. support selection
4. 必要时回到 geometry backbone / model output correction
```

---

## 12. 本计划的关键结论

v10 的核心不是继续追逐 `34.1258 -> 34.10`。新目标是 `25m`，当前 gap 超过 `9m`。因此必须先问：

```text
TTT write-side 有没有 25m 级别的 action 上界？
```

如果 true projection oracle 不能打开上界，就应该尽快停止 TTT write-only，把它降级为辅助 regularizer。真正的主线要转向 window trajectory state、pose-scale correction、read support/per-head 或其他能直接影响 whole-scene drift 的 action。


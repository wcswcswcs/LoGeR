# ACL2 v13：TTT 写入因果动作空间重启实验计划（Target-25）

日期：2026-05-19  
对象：LoGeR / HMC Pipeline v2 / KITTI01  
当前在线 TTT/HMC 最好结果：`H9_REPEAT = 34.1258 / 6.5414`  
诊断型离线 no-GT pose proxy：`NOGTPOSE_27 = 22.4012`，不计入 TTT 写入成功  
新目标：KITTI01 ATE RMSE $\le 25.0m$

---

## 0. 这份计划的立场

这一轮不再把目标定义成“再找一个更好的 gamma / beta / neutral / chunk scalar”。v8 到 v12 已经反复证明，当前这类标量写入动作已经平台化：它们可以改善 rotation、final error、yaw 或某些局部窗口，但无法把在线 TTT/HMC 从 `34.1m` 推向 `25m`。

本计划的核心判断是：

> 当前失败不是因为实验数量不够，而是因为 TTT 写入动作空间没有直接作用到真正的错误变量：reset-window 级别的 scale / trajectory-state drift。

因此 v13 的整体目标是：

1. 严格保持“不做离线轨迹后处理”的边界；
2. 继续探索 TTT 写入，但不再探索单一标量写入；
3. 先验证 TTT fast weights 是否存在足够强的可控 drift-correction subspace；
4. 若存在，就把它做成可部署的 online candidate-commit / MPC-style TTT 写入策略；
5. 若不存在，就明确把 TTT 写入降级为 regularizer，并把 Target-25 主线转向 online trajectory-state module，而不是继续耗在 TTT write-only 上。

---

## 1. 当前实验结果的独立解读

### 1.1 已经不是“没有任何进展”

从历史上看，系统确实有过两次实质跃迁：

```text
frame-attention pair/all read intervention:
    38m 级 -> 36m 级

TTT-native update_conflict_energy + tri-replay:
    36m 级 -> 34m 级
```

这说明 LoGeR 内部 cue 与 TTT-native cue 都是真信号。`update_conflict_energy` 不是噪声；tri-replay 也不是完全无效。问题在于，v8-v12 后续动作没有继续打开新空间。

### 1.2 v12 的实际结论

v12 记录中的关键边界是：

```text
Best online TTT/HMC baseline:
    H9_REPEAT = 34.1258 / 6.5414
    counts_as_ttt_write = true

Best offline no-GT pose diagnostic:
    NOGTPOSE_27 = 22.4012
    counts_as_ttt_write = false

Best v11 projection oracle:
    ORACLE_TTT_01 = 34.8647
    uses_gt_runtime_action = true
    deployable_success = false
```

v12 Phase 1 projection autopsy显示：

```text
Spearman helpful vs scale improvement = 0.0128
Spearman helpful vs segment ATE improvement = -0.2781
normal+inverse oracle worse than H9 = true
projection action coordinate failed = true
```

这意味着当前的 projection coordinate 并不能预测真实 scale improvement，也不能预测 segment ATE improvement。

v12 Phase 2A offline selector 也失败：

```text
candidate_count = 9
chunk_selected_H9_or_better_ratio = 0.2500
selected_oracle_with_worse_global_ATE_count = 10
Spearman proxy score vs ATE = 0.2833
gate_pass = false
```

也就是说，当前 no-GT selector 不能可靠选择更好的 TTT candidate。

v12 Phase 3 auxiliary TTT full runs 全部未过：

```text
H9_REPEAT = 34.1258
best AUX = AUX_TTT_03 = 34.2658
best AUX 比 H9 差 0.1400m
best AUX [200,300) = 74.911，比 H9 74.410 更差
```

v12 Phase 4 dual-lifetime TTT 也失败：

```text
best DLTTT ATE = 35.9173
best DLTTT [200,300) = 77.941，比 H9 差 3.531m
best DLTTT [400,600) = 45.933，比 H9 差 1.580m
```

因此 v12 的最终结论是合理的：

```text
No new deployable online TTT write candidate.
No online target-25.
Best deployable online TTT/HMC remains H9_REPEAT.
```

### 1.3 但不能把 NOGTPOSE_27 当作 TTT 成功

`NOGTPOSE_27 = 22.4012` 很重要，但它是离线 no-GT trajectory-state proxy，不是 online TTT 写入。它证明的是：

> Target-25 所需的误差方向主要存在于 reset-window scale / trajectory-state，而不是普通 token-level dynamic write gate。

它不证明：

```text
当前 TTT 写入已经找到了解法。
```

### 1.4 当前最根本的问题

现在卡住的不是 cue 本身，而是 TTT action space。

当前 TTT 写入 action 大多仍是：

```text
token prior
risk scalar
gamma
neutral lambda
branch/layer mask
chunk window
projection helpfulness scalar
short delta apply scale
```

这些动作只能间接影响 fast weights，再间接影响未来 pose / pointmap / scale。目标错误却是 window-level scale drift：

```text
[200,300) remains ~74m
[200,400) remains ~54m
[400,600) remains ~44m
```

因此现在必须验证一个更高层的问题：

> TTT fast weights 是否能通过候选 commit / finite-difference / rollout scoring 直接控制未来窗口 scale drift？

如果答案是否定的，继续 TTT write-only 就不合理。

---

## 2. 本阶段总目标

v13 的目标不是直接刷一个 `34.0m`，而是做一个 hard decision：

### 2.1 科学目标

回答下面四个问题：

1. TTT fast-weight commit 中是否存在足够强的 drift-correction action？
2. 如果存在，能否通过 no-GT window consistency score 在线选择正确 TTT candidate？
3. 如果不能选择，是 selector 特征不够，还是 candidate bank 本身没有上界？
4. 如果 TTT write-only 不足，是否应正式把 Target-25 主线转向 online pose/trajectory-state module？

### 2.2 工程目标

让 Codex 可以并行推进，而不是串行等待一个大矩阵：

```text
Workstream A: candidate commit bank + short rollout evaluator
Workstream B: post-zeropower norm-preserving delta router
Workstream C: finite-difference TTT sensitivity audit
Workstream D: no-GT window consistency selector
Workstream E: dashboard / registry / stopping-rule automation
```

### 2.3 关键原则

v13 不允许把以下内容写成 TTT success：

```text
offline trajectory rewrite
GT runtime action
postprocess scale correction
只改输出 txt 不改未来 memory state
```

v13 可计入 TTT 写入成功的唯一条件是：

```text
online HMC full run
TTT commit 影响未来 state
输出轨迹由 LoGeR/HMC forward 产生
no offline trajectory rewrite
no GT runtime action
```

---

## 3. 核心假设

## H1：当前 TTT scalar action 已经平台化

### 假设

`gamma / neutral / chunk scalar / read beta / auxiliary replay / dual lifetime scalar` 这一类低维标量动作无法把 KITTI01 ATE 从 `34.1m` 推向 `25m`。

### 已有证据

v12 中 12 条在线 TTT full runs：

```text
AUX_TTT_01-06
DLTTT_01-06
```

没有任何一条超过 H9；没有任何一条把 `[200,300)` 或 `[400,600)` 同时改善到可接受范围。

### 本阶段实验

不再继续做同族微扫。只保留 `H9_REPEAT` 作为 baseline。

### 成立标准

H1 默认成立，除非新 action family 明确满足：

```text
ATE <= 33.5
或 [200,300) <= 70 且 [400,600) <= 44.354
```

如果只是：

```text
34.12 -> 34.08
Rot / FinalErr 改善
[200,300) 不变或变差
```

则仍视为平台内震荡。

---

## H2：TTT candidate commit bank 比单一写入公式更有可能打开上界

### 假设

每个 chunk 不应该只生成一个 `W_{m+1}`。应该生成多个候选 fast-weight commit，让系统根据未来 1-2 个 chunk 的 no-GT consistency 选择真正提交的候选。

形式上，对 chunk $m$ 生成候选：

$$
\mathcal{B}_m = \{W_{m+1}^{(0)}, W_{m+1}^{(1)}, \dots, W_{m+1}^{(K-1)}\}
$$

然后选择：

$$
W_{m+1}^{*}=\arg\min_{W \in \mathcal{B}_m} S_{nogt}(W; m+1:m+L)
$$

这里 $S_{nogt}$ 不是 GT ATE，而是未来短窗口的一组内部一致性 proxy。

### 候选 bank 第一版

候选不追求多，而要互相正交，覆盖当前已知主要 action：

```text
C0: H9 default commit
C1: WINGAM-style tri-replay
C2: C16ROLE-style c16 weak tri-replay
C3: softer body gamma
C4: stronger body gamma
C5: exit gamma reduced
C6: exit gamma removed
C7: native-mix protected commit
C8: post-zeropower norm-preserving delta mix
C9: weak freeze-like continuity-preserving chunk5/6 candidate
```

第一轮只在重点 chunks 上开 bank：

```text
chunks 5-12
chunk 16
```

非重点 chunk 仍使用 H9 default。

### no-GT selector 候选分数

每个 candidate commit 后，短 rollout $L=1$ 或 $2$ 个 chunk，计算：

```text
overlap pointmap residual
pose-increment scale consistency
frame-to-frame translation median consistency
TTT apply mismatch
C23 high-D attention mass after read
SWA/overlap source consistency if available
window local Sim3-free scale proxy
```

定义：

$$
S_{nogt} =
\alpha E_{overlap}
+ \beta E_{scale}
+ \gamma E_{apply}
+ \delta E_{posejump}
+ \eta E_{readmass}
$$

所有项先在候选内部 robust normalize。

### 必须记录

每个 chunk、每个 candidate 记录：

```text
candidate_id
chunk_id
active_window
chosen_by_oracle_gt
chosen_by_nogt_proxy
short_rollout_score
ATE_future_100_if_offline_eval_available
E_overlap
E_scale
E_apply
E_posejump
E_readmass
commit_state_hash
candidate_delta_norm_by_layer_branch
candidate_delta_cos_to_H9
```

### 判断标准

第一阶段先做 oracle selector 上界，不作为 deployable success：

```text
oracle-bank strong pass:
    ATE <= 30.0
    or [200,300) <= 60 and ATE <= 32.5

oracle-bank weak pass:
    ATE <= 33.0
    or [200,300) improves >= 5m and [400,600) not worse than H9

oracle-bank fail:
    best oracle-bank ATE > 33.5
```

如果 oracle-bank fail，说明 candidate bank 里的 TTT actions 本身上界不足，不应继续训练/调 selector。

如果 oracle-bank pass，但 no-GT selector fail，说明 candidate actions 有上界，selector 需要改。

---

## H3：现有 projection coordinate 失败，不代表所有 TTT projection 都失败

### 假设

v12 的 projection coordinate failed，是因为 projection target / coordinate 不对，而不是因为 fast-weight state 完全没有可控 drift subspace。

现有失败点包括：

```text
Spearman helpful vs scale improvement ≈ 0.0128
Spearman helpful vs segment ATE improvement ≈ -0.2781
normal+inverse oracle worse than H9
```

这说明当前 projection helpfulness 没有对齐真实 scale improvement。

### 新实验

改成 **finite-difference sensitivity**，不再相信某个解析 projection 坐标。

对某个 chunk $m$ 和 layer/branch group $g$，定义原始 TTT delta：

$$
\Delta W_{m,g}=W_{m+1,g}-W_{m,g}
$$

测试：

$$
W_{m+1,g}^{+}=W_{m,g}+ (1+\epsilon)\Delta W_{m,g}
$$

$$
W_{m+1,g}^{-}=W_{m,g}+ (1-\epsilon)\Delta W_{m,g}
$$

或者对候选方向 $d_g$：

$$
W_{m+1,g}^{\pm}=W_{m+1,g}\pm \epsilon d_g
$$

然后 rollout 未来 1-2 个 chunk，比较 no-GT proxy 和离线评估诊断。

### 首批 group

```text
chunk 5: w0, w1, w2, all layers
chunk 6: w0, w1, w2, all layers
chunks 5-9: w0 layer groups early/mid/late
chunk 16: w0 all, w1 all
```

### 指标

```text
sensitivity_scale = (E_scale_plus - E_scale_minus) / (2 eps)
sensitivity_overlap
sensitivity_apply
sensitivity_proxy_total
sensitivity_future_ATE_if_diagnostic
```

### 判断标准

如果某些 group 满足：

```text
finite-diff proxy sensitivity 与 future segment ATE delta Spearman >= 0.5
且存在 perturbation 让 [200,300) 或 [400,600) 明显下降
```

则说明 TTT fast weights 有可控 subspace，可以继续做 projection action。

如果所有 group sensitivity 都无法解释 future error：

```text
Spearman < 0.2
且最佳 perturbation ATE 不优于 H9
```

则停止 TTT projection line。

---

## H4：TTT 写入控制必须从 pre-zeropower token prior 转到 post-zeropower delta routing

### 假设

LoGeR 的 TTT replay 中存在 Muon / zeropower / norm restoration。大量 token prior 改动在 pre-zeropower 阶段可能被方向归一化与 norm restoration 折叠掉，导致“少写动态”只改变小幅方向，不改变真正的 fast-weight drift state。

因此需要直接控制 post-zeropower delta：

$$
\Delta W^{native}_{g}
$$

$$
\Delta W^{candidate}_{g}
$$

而不是只控制 token multiplier。

### 实验动作

#### A. Norm-preserving direction mix

保持 native update norm，只替换方向：

$$
\Delta W^{mix}_{g}
=
\|\Delta W^{native}_{g}\|
\cdot
\frac{(1-\lambda)\Delta W^{native}_{g}+\lambda\Delta W^{candidate}_{g}}
{\|(1-\lambda)\Delta W^{native}_{g}+\lambda\Delta W^{candidate}_{g}\|+\epsilon}
$$

#### B. Norm-capped candidate direction

$$
\Delta W^{cap}_{g}
=
\min\left(1, c\frac{\|\Delta W^{native}_{g}\|}{\|\Delta W^{candidate}_{g}\|+\epsilon}\right)
\Delta W^{candidate}_{g}
$$

#### C. Orthogonal harmful component suppression

给定 continuity direction $d_{cont}$：

$$
\Delta W = \Delta W_{\parallel} + \Delta W_{\perp}
$$

$$
\Delta W_{\parallel}=\operatorname{proj}_{d_{cont}}(\Delta W)
$$

$$
\Delta W_{new}=\Delta W_{\parallel}+\rho\Delta W_{\perp}
$$

其中 $\rho < 1$。

### 首批 run

```text
PZ_01: w0 all, lambda=0.25, norm-preserving
PZ_02: w0 all, lambda=0.50, norm-preserving
PZ_03: w0 body window only, lambda=0.25
PZ_04: w0 exit window only, lambda=0.25
PZ_05: w1 norm-capped candidate, cap=0.5
PZ_06: w0 orthogonal suppression rho=0.5
```

### 判断标准

强通过：

```text
ATE <= 33.5
or [200,300) improves >= 5m and [400,600) not worse
```

弱通过：

```text
ATE <= 34.0
and [200,300), [400,600), FinalErr 至少两项优于 H9
```

失败：

```text
只改善 Rot/FinalErr
ATE >= 34.1
[200,300) 不改善或变差
```

---

## H5：如果 TTT write-only 上界不足，Target-25 应转向 online trajectory-state module，而不是继续 TTT 微扫

### 假设

如果 candidate-bank oracle、finite-difference sensitivity、post-zeropower routing 都不能把在线 ATE 推到 `33.5m` 以下，说明当前 TTT write-only 的可控空间不足以支持 Target-25。

这并不等于 TTT 没用，而是它更适合作：

```text
read correction support
rotation / endpoint / local regularizer
short-window trajectory stabilization
```

而 Target-25 主线需要一个显式 online pose/trajectory-state module。

### 判断标准

如果满足以下任一条件，则触发降级：

```text
1. candidate-bank oracle best ATE > 33.5
2. finite-diff best perturbation ATE > 33.5
3. post-zeropower routing 6 条 full run 均未优于 H9
4. no-GT selector selected-run ATE > H9 + 0.1
```

降级后，TTT 不停止，但角色改变：

```text
TTT write = auxiliary regularizer
Target-25 mainline = online trajectory-state / scale-state module
```

---

## 4. 实验阶段设计

## Phase 0：边界锁定与资源调度

### 目标

确保 v13 不是建立在漂移的 baseline 上。

### 必跑

```text
P0_01: H9_REPEAT rerun
P0_02: WINGAM_03 rerun
P0_03: NOGTPOSE_27 diagnostic repeat, not counted as TTT
```

### Gate

```text
H9_REPEAT ATE drift <= 0.03m
WINGAM_03 ATE drift <= 0.03m
NOGTPOSE_27 ATE drift <= 0.05m and counts_as_ttt_write=false
```

不通过则停止，先修复评估 / config / cache。

### 资源

```text
KITTI01 full online HMC:
    GPU 0-3 并发，最多 4 个 full run

short rollout / finite-diff:
    可用 GPU 0-7，但每个 worker 限制 host RAM 与 compile threads

TorchInductor:
    TORCHINDUCTOR_COMPILE_THREADS=1
```

---

## Phase 1：Candidate Commit Bank Oracle

### 目标

先回答 candidate action 有没有上界，不直接训练 selector。

### 实验设计

对重点 chunks 生成 commit candidates：

```text
chunks 5,6,7,8,9,10,11,12,16
```

每个 chunk 生成 candidate bank：

```text
C0 H9 default
C1 WINGAM body/exit
C2 C16ROLE
C3 softer body
C4 stronger body
C5 exit reduced
C6 native protected
C7 post-zeropower norm-preserving
C8 weak freeze-like candidate
C9 no semantic / no negative correction candidate
```

先跑 oracle-bank：使用 GT 只用于选择 best candidate，目的是测上界，不计入 deployable。

### 必须记录

```text
candidate_bank_registry.csv
candidate_choice_oracle.jsonl
candidate_delta_layer_branch.csv
candidate_future_segment_metrics.csv
candidate_state_hash.jsonl
```

字段：

```text
chunk_id
candidate_id
candidate_name
ATE_if_selected_locally
future_100f_ATE
future_200f_ATE
[200,300)
[400,600)
FinalErr
YawRMSE
Sim3Scale
state_hash
```

### 可视化

```text
candidate_regret_heatmap.png
oracle_choice_timeline.png
per_candidate_trajectory_overlay.png
window_segment_delta_waterfall.png
```

### 判断标准

```text
oracle strong pass:
    ATE <= 30.0

oracle useful pass:
    ATE <= 32.5
    or [200,300) <= 60 and [400,600) <= 44.354

oracle weak pass:
    ATE <= 33.5

fail:
    ATE > 33.5
```

### 不满足条件时 Codex 方向

如果 fail：

```text
Codex-A:
    扩充 candidate bank，不调 gamma，而是加入 post-zeropower direction actions。

Codex-B:
    检查 candidate commits 是否真的进入未来 state，比较 state hash 与 per-layer delta。

Codex-C:
    如果 bank 已经覆盖主要 action，但 oracle 仍 fail，自动跳到 Phase 3 finite-diff。
```

---

## Phase 2：No-GT Selector for Candidate Bank

只有 Phase 1 至少 weak pass 才启动。

### 目标

把 oracle candidate choice 转成可部署 no-GT selection。

### selector 输入

```text
E_overlap
E_scale_proxy
E_apply_mismatch
E_posejump
E_readmass
update_conflict_energy_mean
update_conflict_energy_p90
TTT_delta_norm
TTT_delta_cos_to_H9
neutral_mass
negative_mass
D_g_mass
semantic structure/lowstuff mass if available
```

### selector 形式

第一版不用学习模型，使用 robust rank aggregation：

$$
S(c)=\sum_k w_k \operatorname{rank}_k(c)
$$

候选选择：

$$
c^*=\arg\min_c S(c)
$$

第二版允许一个小型 leave-one-window ridge/logistic selector，但必须报告是否过拟合。

### 指标

```text
oracle_agreement_rate
candidate_regret_mean
candidate_regret_p90
selected_H9_ratio
selected_worse_than_H9_count
Spearman(proxy_score, future_ATE)
```

### Gate

```text
selector strong pass:
    selected online ATE <= 32.5
    and selected_oracle_regret_mean <= 0.2m

selector weak pass:
    selected online ATE <= 33.5
    and no segment regression > 2m vs H9

fail:
    selected online ATE > H9
    or selected_worse_than_H9_count high
```

### 不满足条件时 Codex 方向

```text
If Spearman < 0.3:
    Codex-D 增加 no-GT features，不跑 full；先做 offline selector scatter。

If oracle agreement low but top-2 contains oracle often:
    Codex-E 改成 top-2 candidate rollout with longer horizon L=2。

If selector picks H9 too often:
    Codex-F 调整 proxy scale normalization，避免保守 collapse。
```

---

## Phase 3：Finite-Difference TTT Sensitivity

Phase 1 fail 或 Phase 2 fail 都可以并行启动 Phase 3。

### 目标

不再猜哪个 cue 对，而是直接测 fast-weight perturbation 对未来 window proxy / ATE 的影响。

### 实验设计

对 group $g$ 做 $+$ / $-$ perturbation：

```text
G1 chunk5 w0 all
G2 chunk6 w0 all
G3 chunks5-9 w0 early
G4 chunks5-9 w0 mid
G5 chunks5-9 w0 late
G6 chunk16 w0 all
G7 chunk16 w1 all
G8 exit10-12 w0 all
```

扰动：

```text
epsilon = 0.05, 0.10
方向 = native delta / tri-replay delta / conflict delta
```

### 记录

```text
finite_diff_group.csv
finite_diff_rollout_proxy.csv
finite_diff_future_eval.csv
```

字段：

```text
group_id
chunk_scope
layer_scope
branch
perturb_direction
epsilon
E_scale_plus
E_scale_minus
E_overlap_plus
E_overlap_minus
ATE_future_plus
ATE_future_minus
sensitivity_proxy
sensitivity_ATE
```

### 可视化

```text
sensitivity_heatmap_layer_branch.png
proxy_vs_ATE_sensitivity_scatter.png
best_perturbation_trajectory_overlay.png
```

### 判断标准

```text
pass:
    Spearman(sensitivity_proxy, sensitivity_ATE) >= 0.5
    and best perturbation improves ATE >= 0.5m or [200,300) >= 3m

fail:
    Spearman < 0.2
    and no perturbation improves H9 meaningfully
```

### 不满足条件时 Codex 方向

```text
If all sensitivity near zero:
    Codex-G 检查 perturbation 是否被 norm restore 抹掉，改用 post-zeropower injection。

If proxy sensitivity differs from ATE sensitivity:
    Codex-H 替换 no-GT proxy，增加 scale-window median / overlap pointmap / posejump 权重。

If ATE sensitivity exists but proxy cannot predict:
    Codex-I 暂停 selector，走 oracle upper-bound branch。
```

---

## Phase 4：Post-Zeropower Delta Routing

### 目标

验证直接控制 fast-weight delta direction 是否比 pre-zeropower token prior 更有效。

### 实验组

```text
PZ_01: w0 all norm-preserving mix lambda=0.25
PZ_02: w0 all norm-preserving mix lambda=0.50
PZ_03: w0 body-only norm-preserving lambda=0.25
PZ_04: w0 exit-only norm-preserving lambda=0.25
PZ_05: w1 norm-capped cap=0.5
PZ_06: w0 orthogonal suppression rho=0.5
PZ_07: layer12/head0 targeted if hook verified
PZ_08: high-sensitivity group from Phase 3
```

### 记录

```text
post_zeropower_delta_norm.csv
post_zeropower_delta_cosine.csv
norm_restore_ratio.csv
per_layer_branch_update_heatmap.csv
```

### 判断标准

```text
strong pass:
    ATE <= 33.0

useful pass:
    ATE <= 33.5
    or [200,300) improves >= 5m without [400,600) regression

fail:
    no run beats H9 by >= 0.3m
```

### 不满足条件时 Codex 方向

```text
If PZ all fail:
    Codex-J 停止 post-zeropower routing，回到 candidate bank upper bound。

If only Rot/FinalErr improve:
    Codex-K 标记为 regularizer，不进入 target-25 mainline。

If one branch helps but ATE still high:
    Codex-L combine with candidate selector, not gamma sweep。
```

---

## Phase 5：Global Drift Dashboard 必须成为主指标

每个 full run 必须输出以下 dashboard。没有 dashboard 不允许进入结论。

### 轨迹指标

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
```

### window drift 指标

```text
per_reset_window_ATE
per_reset_window_scale_proxy
per_reset_window_yaw_drift
per_reset_window_translation_norm_median
chunk_boundary_pose_jump
body_window_error
exit_window_error
handoff_window_error
```

### TTT state 指标

```text
per_chunk_memory_rel_diff
per_layer_branch_delta_norm
per_layer_branch_delta_cos_to_H9
per_layer_branch_delta_cos_to_native
post_zeropower_delta_norm
norm_restore_ratio
role_mass_pos/neu/neg
candidate_selected_id
```

### 可视化

```text
trajectory_xy_overlay.png
per_100f_segment_bar.png
per_reset_window_ATE_curve.png
scale_proxy_over_time.png
yaw_drift_over_time.png
candidate_choice_timeline.png
layer_branch_delta_heatmap.png
proxy_vs_future_error_scatter.png
```

---

## Phase 6：Cross-sequence sanity

只有满足以下条件才跑 cross-seq：

```text
KITTI01 ATE <= 33.0
or [200,300) improves >= 8m and overall ATE <= 34.0
or selector/action family shows clear oracle upper bound <= 32.5
```

序列：

```text
KITTI00 full
KITTI02 full
KITTI05 full
```

必须对比：

```text
H9_REPEAT
WINGAM_03
NOGTPOSE_27 diagnostic only, if needed
best v13 online candidate
```

通过标准：

```text
mean ATE over 00/02/05 improves over H9 by >= 1.0m
no sequence regression > 5%
KITTI02 must not regress catastrophically
```

---

## 5. 并行执行安排

### Codex-A：Candidate bank implementation

任务：

```text
实现多个 TTT candidate commit 的生成、存储、hash、选择。
保证 candidate commit 影响未来 HMC state。
```

优先 smoke：

```text
END_FRAME=180
candidate_count >= 4
不同 candidate state_hash 不同
selected candidate 写入下一 chunk
```

如果失败：先修 state plumbing，不跑 full。

### Codex-B：Short rollout evaluator

任务：

```text
对每个 candidate 运行未来 1-2 chunk probe，计算 no-GT proxy。
```

必须输出：

```text
candidate_proxy.jsonl
candidate_rollout_state_hash.jsonl
```

### Codex-C：Finite-difference perturbation

任务：

```text
实现 layer/branch group 的 +/- epsilon perturbation。
```

必须防止 perturbation 被 norm restore 立即抹掉；需要记录 perturbation after restore 的实际范数。

### Codex-D：Post-zeropower delta routing

任务：

```text
在 zeropower 之后、commit 之前直接修改 delta direction / norm。
```

必须记录：

```text
pre_delta_norm
post_delta_norm
cos_pre_post
norm_restore_ratio
```

### Codex-E：Dashboard automation

任务：

```text
每条 full run 自动生成 global drift dashboard。
不生成 dashboard 的 run 不进入 summary。
```

---

## 6. 总停止规则

为了避免继续慢速平台震荡，v13 使用硬停止规则。

### 停止 TTT scalar 微扫

任何同族 scalar 微扫，如果连续 4 条 full run 满足：

```text
ATE > current_best - 0.05
且 [200,300) 没有改善 >= 2m
```

立即停止。

### 停止 candidate bank

如果 oracle bank best：

```text
ATE > 33.5
```

停止 no-GT selector，不再训练/调 selector。

### 停止 projection / finite-diff

如果 sensitivity 与 future ATE correlation：

```text
Spearman < 0.2
```

且最佳 perturbation 不优于 H9，则停止该 projection family。

### 停止 post-zeropower routing

如果 8 条 PZ full run 无一满足：

```text
ATE <= 33.5
or [200,300) improves >= 5m
```

停止该方向。

### 触发路线降级

如果 Phase 1-4 都失败，则给出明确结论：

```text
TTT write-only 当前动作空间不足以支持 target-25。
TTT 保留为 regularizer。
Target-25 主线转向 online trajectory-state / scale-state module。
```

这不是放弃 TTT，而是承认 TTT write-only 不是主杠杆。

---

## 7. 预期输出目录

```text
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/
    phase0_repeats/
    phase1_candidate_bank_oracle/
    phase2_nogt_selector/
    phase3_finite_diff_sensitivity/
    phase4_post_zeropower_routing/
    phase5_global_drift_dashboard/
    registry_v13.csv
    decision_log.md
```

---

## 8. 最终成功标准

### Final success

```text
KITTI01 ATE <= 25.0
counts_as_ttt_write = true
no offline trajectory rewrite
no GT runtime action
```

### Strong TTT progress

```text
KITTI01 ATE <= 30.0
or [200,300) <= 55 and ATE <= 32.5
```

### Useful TTT progress

```text
KITTI01 ATE <= 33.0
or [200,300) improves >= 8m and [400,600) not worse
```

### Regularizer only

```text
Rot / FinalErr improve
but ATE remains >= 34.0
or [200,300) remains >= 70
```

---

## 9. 一句话总结

v13 要验证的不是“再调一个更好的 TTT prior”，而是：

> TTT fast weights 是否存在可部署的、能控制 reset-window scale / trajectory-state drift 的 commit action。如果存在，用 candidate bank + short rollout selector 找出来；如果不存在，停止 TTT write-only 主线，把 Target-25 转给 online trajectory-state module。

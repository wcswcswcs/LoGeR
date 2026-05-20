# ACL2 v18：TTT Write True Action Space / Target-25 实验计划

日期：2026-05-20  
对象：LoGeR / HMC Pipeline v2 / KITTI01 / TTT write policy  
主目标：在线、可部署、不使用 GT runtime action、不做离线 trajectory rewrite 的前提下，继续寻找能接近或达到 KITTI01 ATE $\le 25m$ 的 TTT 写入策略。

---

## 0. 当前结论与本计划定位

v17 的结果说明，当前没有新的可部署 TTT 写入策略。当前最好可计数线上配置仍是：

```text
C9 locked repeat
ATE = 33.7629421029m
Rot = 6.5259
[200,300) = 76.102136m
[400,600) = 41.896364m
counts_as_ttt_write = true
```

H9 仍是更适合做 causal-fork parent 的基准，因为它的关键 body 病灶段更好：

```text
H9 ATE = 34.1257769401m
H9 [200,300) = 74.409927m
C9 [200,300) = 76.102136m
```

v17 做了 174 条 diagnostic short-rollout row：

```text
Phase1 horizon expansion: 84 rows
Phase2 basis proxy:       36 rows
Phase3 AUXGEO proxy:      36 rows
Phase4 dual-bank proxy:   18 rows
```

但没有任何候选达到进入 selector 或 full validation 的门槛：

```text
best h8/h10 ATE delta = -0.860542m
best h8/h10 [200,300) delta = -1.858631m
required h8/h10 ATE delta <= -3m
or required h8/h10 [200,300) delta <= -5m without downstream regression
```

因此，本计划不再把目标写成“继续调 gamma / neutral / weak freeze / proxy gate”。这些已经被 v8-v17 的结果证明基本进入平台。v18 的核心任务是验证一个更根本的问题：

> TTT fast weights 中是否存在一个可部署的、可选择的、能修正 reset-window scale / trajectory drift 的真实 action space？

如果答案是肯定的，v18 要把它从 sandbox oracle 推进到 no-GT selector，再推进到 full online validation。  
如果答案是否定的，就必须停止把 TTT write-only 当作 Target-25 主线，转为 “TTT 稳定器 + 显式 online trajectory-state / scale-state 模块”。

---

## 1. 实验整体目标

本阶段不是小修小补，而是重新定义 TTT 写入探索的 action space。

### 1.1 总目标

在固定 read 主线和 HMC commit 协议下，寻找一种 TTT 写入策略，使其满足：

```text
final target:
    KITTI01 full online ATE <= 25.0m
    counts_as_ttt_write = true
    no GT runtime action
    no offline trajectory rewrite
    no postprocess trajectory replacement
```

阶段性目标分三层：

```text
stage target 1:
    sandbox h10/h15 candidate shows ATE delta <= -3m
    or [200,300) delta <= -5m without [400,600) regression > +1m

stage target 2:
    full online KITTI01 ATE <= 32.5m
    and [200,300) improves >= 5m over H9
    and [400,600) does not regress > +1m

stage target 3:
    full online KITTI01 ATE <= 30.0m
    and no severe reset-window trade-off

final target:
    full online KITTI01 ATE <= 25.0m
```

### 1.2 本计划要回答的核心问题

本计划围绕六个科学问题展开。

第一，v17 的失败是否只是因为候选太弱，还是 TTT fast weights 本身缺少可控 drift-correction subspace？

第二，v17 的 residual basis / AUXGEO / dual-bank 都是 proxy 级实现。真正 post-zeropower tensor basis routing、真正 overlap-geometry auxiliary replay、真正 W_long / W_short dual bank 是否仍有上界？

第三，短窗口 h5 上出现的局部 body-window 信号为何无法延展到 h8/h10？是因为 action 生命周期太短，还是因为有害方向和 continuity 方向没有分解？

第四，当前 no-GT proxy 是否能预测 candidate commit 的未来收益？如果不能，应该先改善 proxy，而不是直接 full online。

第五，TTT 写入是否必须接入 reset-window scale / trajectory-state，而不是只控制 token update prior？

第六，如果 TTT write-only 的真实上界仍然不足，应如何快速切换到 TTT-assisted online trajectory-state module，而不是继续消耗 full-run 小矩阵？

---

## 2. 固定边界、成功定义与禁止事项

### 2.1 固定基线

v18 所有实验必须显式记录以下基线：

```text
H9 locked parent:
    ATE = 34.1257769401m
    Rot = 6.5414
    [200,300) = 74.409927m
    [400,600) = 44.353638m

C9 locked online best:
    ATE = 33.7629421029m
    Rot = 6.5259
    [200,300) = 76.102136m
    [400,600) = 41.896364m

NOGTPOSE_27 diagnostic:
    ATE ~= 22.4m
    counts_as_ttt_write = false
    offline / trajectory-state diagnostic only
```

H9 是 candidate-bank parent 之一；C9 是当前可计数 online best；NOGTPOSE 只作为 target-25 上界诊断，不允许写成 TTT success。

### 2.2 什么算 TTT write success

一个结果只有同时满足以下条件，才可以计入 deployable TTT write candidate：

```text
1. full KITTI01 online run completed;
2. hmc_commit_mode = probe_ttt_write;
3. TTT state is actually committed and affects future chunks;
4. no GT runtime action;
5. no offline trajectory rewrite;
6. no postprocess replacement of trajectory txt;
7. counts_as_ttt_write = true;
8. run has full trajectory diagnostics and HMC rows = 38.
```

Sandbox / oracle / proxy / partial run 只能作为 diagnostic，不计入 deployable TTT success。

### 2.3 禁止继续做的低价值实验

除非用于 smoke 或 ablation，不再启动下面这些 full-run 小矩阵：

```text
gamma 0.0048 / 0.0052 微扫
neutral 0.82 / 0.88 微扫
read beta 4.80 / 4.90 微扫
chunk16 scalar gate 微扫
commit EMA 0.90 / 1.10 full 微扫
post-zp mix/cap 小矩阵
单纯 sparse ratio 小扫
semantic scalar multiplier 小扫
```

这些方向已经在 v8-v17 证明只能产生局部 regularization 或误差重分配，不足以支撑 Target-25。

---

## 3. 核心假设

### H1：v17 失败的主要原因是 action proxy 不够真实，而不是 TTT action 必然无上界

v17 Phase2 是 proxy-only BASIS，没有实现 true historical post-zeropower tensor basis projection，也没有产出 `post_zp_delta_before_after.pt`。Phase3 AUXGEO 没有稳定导出 semantic structure-token cosine / per-group target fields。Phase4 dual-bank 也没有 full W_long / W_short tensor cosine。

假设 H1 认为：

> 如果把 proxy action 升级为真实 tensor action，TTT 可能仍然存在可控上界。

验证方式是实现 true action instrumentation，并先用 sandbox oracle 测上界。

H1 成立标准：

```text
true post-zp basis / true auxgeo / true dual-bank 任一族在 h10/h15 上达到：
    ATE delta <= -3m
    or [200,300) delta <= -5m with [400,600) regression <= +1m
```

H1 不成立标准：

```text
三类 true action 全部无法超过：
    h10/h15 ATE delta <= -1m
    and [200,300) delta <= -2m
```

如果 H1 不成立，TTT write-only 应降级为 regularizer，Target-25 主线转入 online trajectory-state / scale-state module。

---

### H2：当前短窗口局部收益衰减，是因为 action 没有建模 reset-window 生命周期

v17 中 `BASIS_02_PROXY_HARM_W0_EMA090` 在 h5 局部 `[200,300)` 接近 `-4.857m`，但 h8/h10 没过 gate，且有 downstream regression。这说明某些 action 能短期修 body-window，但无法维持到整个 reset-window。

假设 H2 认为：

> TTT correction 需要 lifecycle：某些 correction 应覆盖 body window，某些 correction 应在 exit window 衰减，某些 continuity direction 必须进入 long memory。

验证方式是把候选按生命周期拆为：

```text
W_long: continuity / structure / scale-stable update
W_body: body-window correction, active for chunks 5-9 equivalent range
W_exit: weaker handoff correction, active for chunks 10-12 equivalent range
```

对应 commit：

$$
W_{apply}(m)=W_{long}(m)+a_b(m)W_{body}(m)+a_e(m)W_{exit}(m)
$$

$$
W_{commit}(m+1)=W_{long}(m+1)
$$

其中 $a_b(m)$ 和 $a_e(m)$ 由 reset-window position 和 no-GT risk proxy 决定。

H2 成立标准：

```text
lifecycle candidate 在 h10/h15 上相对 H9:
    ATE delta <= -3m
    or [200,300) delta <= -5m
且 [400,600) regression <= +1m
```

---

### H3：TTT write-only 必须接入 window-scale / trajectory-state proxy，单靠 token risk 不够

NOGTPOSE_27 的诊断意义是：target-25 的误差方向存在于 reset-window scale / trajectory-state。它不是 TTT success，但它证明了主误差不是普通 dynamic-token write gate 能直接解释。

假设 H3 认为：

> 一个有效 TTT 写入策略必须使用 no-GT window-scale / trajectory-state proxy 来决定 candidate commit，而不是只看 token-level prior。

定义窗口级 proxy：

$$
S_{scale}(m)=\left|\log \frac{\bar s_{pred}(m)}{\operatorname{EMA}(\bar s_{pred})}\right|
$$

$$
S_{overlap}(m)=\operatorname{Mean}_p\|X_{tail,m}(p)-X_{head,m+1}(p)\|_2
$$

$$
S_{yaw}(m)=\|\Delta \psi_m-\operatorname{EMA}(\Delta \psi)\|
$$

综合 proxy：

$$
R_{window}(m)=w_sS_{scale}(m)+w_oS_{overlap}(m)+w_yS_{yaw}(m)+w_tS_{ttt}(m)
$$

其中 $S_{ttt}$ 可以来自 `update_conflict_energy`、post-zp delta flip、role mass imbalance 等。

H3 成立标准：

```text
no-GT selector 在 offline/sandbox audit 中：
    Spearman(proxy_score, future_ATE_delta) >= 0.50
    top-1 candidate hit rate >= 0.50
    selected candidate h10/h15 ATE delta <= -2m for at least two fork points
```

若 proxy 低于该标准，不允许 full online selector validation，Codex 应继续改 proxy，而不是跑 full。

---

### H4：真正有效的 TTT action 应该作用于 post-zeropower delta / fast-weight state，而不只是 pre-zeropower token prior

LoGeR TTT 的 pre-token update 经过 zeropower / norm restoration。许多 token prior 改动可能在矩阵正交化和范数恢复后被折叠，导致 full-run 只出现 Rot / FinalErr trade-off。

假设 H4 认为：

> 如果要控制 drift，需要直接作用在 post-zeropower branch/layer delta 的方向与范数上。

定义每个 chunk、layer、branch 的 post-zp delta：

$$
\Delta W_{m,l,b}^{zp}=W_{m+1,l,b}^{candidate}-W_{m,l,b}
$$

构造 basis：

```text
B_continuity: 与 H9 / C9 稳定 continuity 方向一致
B_body_harm:  与 body-window drift 相关的 harmful residual
B_scale:      与 no-GT scale proxy 相关的 residual
B_overlap:    与 overlap geometry mismatch 相关的 residual
```

候选 action：

$$
\Delta W' = \Delta W - \alpha_h P_{B_{harm}}(\Delta W) + \alpha_c P_{B_{cont}}(\Delta W)
$$

H4 成立标准：

```text
post-zp true basis candidate 在 sandbox h10/h15：
    ATE delta <= -3m
    or [200,300) delta <= -5m without downstream regression
```

---

### H5：如果 TTT true action 仍无上界，则 Target-25 不应继续由 TTT write-only 承担

这是阶段性停止假设。若 H1-H4 的 true action families 都不能打开上界，则说明 TTT write-only 在当前 LoGeR/HMC 接口下更适合作为稳定器，而不是 Target-25 主线。

H5 触发条件：

```text
true post-zp basis fail
true auxgeo overlap replay fail
true lifecycle / dual-bank fail
candidate bank oracle fail
no-GT selector cannot reach Spearman >= 0.50
```

触发后策略：

```text
TTT remains:
    regularizer / stabilizer / continuity helper
Target-25 mainline moves to:
    explicit online trajectory-state / scale-state module
But the module must be online and no-GT, not offline postprocess.
```

---

## 4. 实验阶段设计

## Phase 0：边界复现与工具一致性

### 目标

确认 H9/C9/WINGAM/NOGTPOSE 边界仍然稳定，确保 causal fork 和 v17 plus-one horizon evaluation 仍可复用。

### 实验

运行或复用 locked artifact：

```text
P0_H9_repeat
P0_C9_repeat
P0_WINGAM_repeat
P0_NOGTPOSE_diagnostic_only
```

### 必须记录

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
[200,300)
[200,400)
[400,600)
HMC rows
hmc_state_hash
merge/gauge hash
counts_as_ttt_write
uses_gt_runtime_action
uses_offline_trajectory_rewrite
```

### 通过标准

```text
H9 ATE drift <= 0.03m
C9 ATE drift <= 0.03m
HMC rows = 38
Causal fork parity rows from v16/v17 remain pass
NOGTPOSE flagged diagnostic_only / counts_as_ttt_write=false
```

若失败，Codex 应先执行：

```text
1. compare hmc_config.yaml against locked reference;
2. compare mp_alpha, read beta, reset_every, SWA/TTT flags;
3. compare trajectory txt hash and HMC state hash;
4. stop all candidate full runs until boundary restored.
```

---

## Phase 1：True Action Instrumentation Gate

### 目标

v17 Phase2/3/4 多数是 proxy 实现。v18 先要求真实 action instrumentation 通过，再允许 candidate bank。

### 需要落盘的 artifact

```text
post_zp_delta_before_after.pt
per_layer_branch_post_zp_delta.pt
per_token_to_post_zp_contribution_summary.pt
basis_vector_bank.pt
basis_projection_coefficients.csv
W_long_short_tensor_summary.pt
W_short_apply_history.jsonl
overlap_geometry_replay_target.pt
overlap_geometry_replay_debug.jsonl
window_scale_proxy.jsonl
candidate_commit_manifest.csv
```

### smoke tests

```text
SMOKE_POSTZP_TRUE_BASIS_e160
SMOKE_AUXGEO_TRUE_REPLAY_e160
SMOKE_DUALBANK_TRUE_TENSOR_e160
SMOKE_SELECTOR_PROXY_e160
```

### 通过标准

```text
required artifacts present >= 90%
post_zp_delta_before_after.pt present
W_long/W_short tensors present for dual-bank smoke
overlap_geometry target present for auxgeo smoke
no NaN in basis projection coefficients
smoke trajectory completes
```

若不通过，Codex 应自动尝试：

```text
- if post_zp delta missing:
    instrument ttt_write_controller after zeropower and after norm restoration;
    add branch/layer id and chunk id to debug rows.

- if W_long/W_short tensors unavailable:
    export complete tensor summaries, not only scalar proxy rows;
    add cosine(W_short, W_long) and norm ratio logging.

- if AUXGEO semantic fields unavailable:
    fall back to pure geometry overlap target first;
    do not fabricate semantic cosine fields.

- if memory footprint too high:
    save only selected chunks 5/6/10/16 and only selected layers/branches first;
    disable candidate merge snapshots by default.
```

---

## Phase 2：True Post-Zeropower Residual Basis Candidate Bank

### 目标

验证真正 post-zp residual basis routing 是否能产生 durable h10/h15 drift correction。

### 候选族

固定 parent：

```text
H9 parent for body-window health
C9 parent for full-ATE best contrast
```

fork chunks：

```text
chunk 5
chunk 6
chunk 10
chunk 16
```

horizons：

```text
h5, h8, h10, h15
```

候选：

```text
PZBASIS_01_continuity_boost_w0
PZBASIS_02_harm_suppress_w0
PZBASIS_03_harm_suppress_w0w2
PZBASIS_04_scale_basis_suppress_w0
PZBASIS_05_overlap_basis_boost_w0
PZBASIS_06_continuity_plus_harm_w0
PZBASIS_07_lowrank_basis_rank4
PZBASIS_08_lowrank_basis_rank8
```

### 公式

给定 post-zp delta $\Delta W$ 和 basis $B_k$，定义投影：

$$
P_{B_k}(\Delta W)=\frac{\langle \Delta W,B_k\rangle}{\|B_k\|^2+\epsilon}B_k
$$

候选更新：

$$
\Delta W' = \Delta W + \sum_k c_k P_{B_k}(\Delta W)
$$

其中 $c_k$ 可以是 continuity boost 的正数，也可以是 harmful suppress 的负数。

### 必须记录

```text
candidate_id
parent_id
fork_chunk
horizon
ATE_delta_vs_H9_suffix
ATE_delta_vs_parent_suffix
[200,300)_delta
[400,600)_delta
FinalErr_delta
Sim3Scale_delta
basis_coefficients
post_zp_delta_norm_before
post_zp_delta_norm_after
cos_delta_to_continuity_basis
cos_delta_to_harm_basis
cos_delta_to_scale_basis
cos_delta_to_overlap_basis
```

### 通过标准

强通过：

```text
h10 or h15 ATE delta <= -3m
and [400,600) regression <= +1m
```

局部通过：

```text
h10 or h15 [200,300) delta <= -5m
and [400,600) regression <= +1m
```

失败：

```text
best h10/h15 ATE delta > -1m
or all candidates with [200,300) improvement cause [400,600) regression > +2m
```

若失败，Codex 应尝试：

```text
1. change basis construction from chunk-local to reset-window accumulated;
2. use low-rank SVD basis from freeze5/freeze56 removed directions;
3. separate branch w0 and w2; do not mix w1 unless diagnostic shows w1 is necessary;
4. expand horizon to h20 only if h15 shows at least -1.5m signal;
5. if no signal remains, stop PZBASIS family.
```

---

## Phase 3：True Overlap-Geometry Auxiliary Replay

### 目标

验证把 overlap geometry consistency 直接作为 TTT auxiliary replay target 是否能比 token risk / semantic scalar 更接近 scale/drift。

### 动机

当前 AUXGEO proxy 只产生弱信号，且 semantic structure-token fields 不稳定。v18 先做 pure geometry overlap target，再逐步加入 semantic。

### 候选族

```text
AUXGEO_TRUE_01_overlap_pointmap_v_w0
AUXGEO_TRUE_02_overlap_pointmap_kv_w0
AUXGEO_TRUE_03_overlap_scale_proxy_w0
AUXGEO_TRUE_04_overlap_scale_proxy_w0w2
AUXGEO_TRUE_05_structure_only_overlap_w0
AUXGEO_TRUE_06_lowD_structure_overlap_w0
```

### auxiliary loss / replay target

对 overlap 区域构造 pseudo target：

$$
T_{geo}(i)=v_i + \lambda_g \cdot \phi(E_{overlap}(i))
$$

其中 $E_{overlap}(i)$ 是 overlap pointmap residual 或 scale proxy residual，$\phi$ 是投影到 value feature space 的轻量映射或 existing feature-center residual。

TTT replay 分解为：

$$
G_{commit}=G_{main}+\lambda_{aux}G_{aux}
$$

### 必须记录

```text
overlap_pointmap_residual_before
overlap_pointmap_residual_after
overlap_scale_proxy_before
overlap_scale_proxy_after
aux_replay_norm
aux_replay_cos_to_native
aux_replay_cos_to_post_zp_delta
per_group_structure_coverage
per_group_lowD_structure_coverage
horizon metrics h5/h8/h10/h15
```

### 通过标准

```text
h10/h15 ATE delta <= -2m
or overlap pointmap residual improves >= 10% and full candidate bank [200,300) delta <= -5m
```

如果只改善 Rot / FinalErr，但 ATE 与 `[200,300)` 不改善，则停止 AUXGEO_TRUE，不再加 semantic scalar。

若失败，Codex 应尝试：

```text
- if geometry target too weak:
    increase lambda_aux only in w0 and late TTT layers;
- if downstream regression appears:
    add continuity basis protection from Phase2;
- if semantic fields missing:
    keep pure geometry target; do not block on segmentation;
- if pure geometry works:
    then add semantic structure filter as a second-stage ablation.
```

---

## Phase 4：True Dual-Bank TTT Memory

### 目标

验证 TTT 是否需要显式分离 long memory 和 short correction，而不是把所有 correction 写进同一个 fast-weight state。

### 设计

定义：

$$
W_{apply}(m)=W_{long}(m)+\alpha_s(m)W_{short}(m)
$$

$$
W_{commit}(m+1)=W_{long}(m+1)
$$

短期记忆衰减：

$$
W_{short}(m+1)=\rho_s W_{short}(m)+\Delta W_{short}(m)
$$

长期记忆：

$$
W_{long}(m+1)=W_{long}(m)+\Delta W_{long}(m)
$$

### 候选族

```text
DLTRUE_01_short_harm_only_w0
DLTRUE_02_short_body_harm_long_continuity
DLTRUE_03_short_scale_correction_long_structure
DLTRUE_04_short_overlap_correction_long_native
DLTRUE_05_short_decay_fast_K2
DLTRUE_06_short_decay_slow_K5
DLTRUE_07_reset_bound_short_clear
DLTRUE_08_exit_weak_short_handoff
```

### 必须记录

```text
W_long_norm
W_short_norm
W_short_to_long_norm_ratio
cos(W_short, W_long)
short_apply_scale_by_chunk
short_decay_by_chunk
long_delta_norm_by_layer_branch
short_delta_norm_by_layer_branch
[200,300)_delta
[400,600)_delta
h10/h15 metrics
```

### 通过标准

```text
h10/h15 [200,300) delta <= -5m
and [400,600) regression <= +1m
```

或者：

```text
full online ATE <= 32.5m after selector validation
```

若失败，Codex 应尝试：

```text
- if W_short hurts downstream:
    reduce apply scale, increase decay, restrict to w0;
- if W_short has no effect:
    increase active lifetime within reset group, but clear at reset boundary;
- if W_short and W_long cosine high:
    basis split is not separating roles; return to Phase2 basis construction;
- if W_short improves [200,300) but hurts [400,600):
    add W_long continuity protection and exit handoff decay.
```

---

## Phase 5：No-GT Short-Horizon Selector

### 目标

只有当 Phase2/3/4 任一族过 oracle gate，才启动 no-GT selector。Selector 的目标是在 runtime 不使用 GT 的情况下选择 candidate commit。

### 候选输入

来自 Phase2/3/4 的 top candidates：

```text
K candidate fast-weight commits per fork chunk
parent state
short rollout h5/h8/h10/h15
no-GT proxy features
```

### proxy features

```text
overlap_pointmap_residual
local scale ratio stability
reset-window scale drift proxy
yaw increment stability
TTT apply mismatch
post-zp delta norm spike
role mass imbalance
D_g high mass
structure lowD mass
```

### selector score

$$
Score(c)=w_oE_{overlap}(c)+w_sE_{scale}(c)+w_yE_{yaw}(c)+w_aE_{apply}(c)+w_rE_{role}(c)
$$

选择：

$$
c^*=\arg\min_c Score(c)
$$

### offline audit

在已有 oracle table 上评估：

```text
Spearman(score, oracle_ATE_delta)
Top1 hit rate
Top2 hit rate
selected vs H9 mean delta
selected vs best oracle regret
```

### 通过标准

```text
Spearman >= 0.50
Top1 hit rate >= 0.50
selected candidate h10/h15 ATE delta <= -2m in at least 2 fork settings
selector regret <= 1m
```

如果不通过，Codex 应尝试：

```text
- remove features with wrong correlation;
- fit separate body and exit selectors;
- add reset-position feature;
- add downstream penalty term for [400,600) proxy;
- do not launch full online validation until selector passes.
```

---

## Phase 6：Full Online Validation

### 目标

只验证 selector 通过后的少数候选，避免 full-run 浪费。

### 候选数量限制

每批最多：

```text
2 deployable candidates + H9 repeat + C9 repeat
```

### 必须记录

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
[0,200)
[200,300)
[200,400)
[400,600)
50f_mean/worst
100f_mean/worst
200f_mean/worst
YawRMSE
Sim3Scale
per-reset scale
per-reset ATE
selector choices by chunk
candidate selected count
candidate rejected count
HMC rows
state hash
```

### 成功标准

最终成功：

```text
ATE <= 25.0m
counts_as_ttt_write = true
no GT runtime action
no offline trajectory rewrite
```

强进展：

```text
ATE <= 30.0m
and [200,300) improves >= 15m
and [400,600) does not regress > +1m
```

阶段进展：

```text
ATE <= 32.5m
or [200,300) improves >= 8m with ATE <= 33.5m
```

如果 full online candidate 只比 C9 好小于 `0.3m`，且 `[200,300)` 不改善，则不算进展，停止该 family。

---

## 5. 全局指标与可视化要求

每个 phase 都必须输出统一 dashboard，避免只看 ATE。

### 5.1 Trajectory metrics

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
per-axis RMSE x/y/z
per-frame translation error
per-frame yaw error
```

### 5.2 Segment metrics

```text
[0,100), [100,200), [200,300), [300,400), [400,500), [500,600)
[200,400)
[400,600)
50f mean / worst
100f mean / worst
200f mean / worst
reset-group ATE
body-window ATE
exit-window ATE
handoff-window ATE
```

### 5.3 Drift-state metrics

```text
per-reset Sim3 scale
local step-length scale ratio
cumulative yaw drift
cumulative translation drift
chunk-to-chunk pose jump
overlap pointmap residual
overlap pose discontinuity
```

### 5.4 TTT write metrics

```text
per-layer branch delta norm
post-zp delta norm before/after
norm restoration ratio
cosine to native delta
cosine to continuity basis
cosine to harmful basis
cosine to scale basis
role mass: positive / neutral / negative
W_long norm
W_short norm
W_short / W_long ratio
candidate selected count
candidate oracle regret
```

### 5.5 可视化

必须生成：

```text
trajectory_xy_overlay.png
per_frame_error_curve.png
segment_ate_bar.png
reset_group_scale_curve.png
chunk_window_drift_heatmap.png
candidate_horizon_delta_heatmap.png
basis_coefficient_heatmap.png
post_zp_layer_branch_heatmap.png
overlap_residual_map_grid.png
selector_score_vs_oracle_delta_scatter.png
W_long_short_norm_curve.png
```

可视化要求：

```text
1. 所有图必须同时显示 H9、C9、candidate。
2. 所有 candidate heatmap 必须标注 chunk、horizon、parent。
3. 所有 full online 图必须标出 [200,300) 和 [400,600) 区间。
4. 所有 selector 图必须显示 chosen candidate 和 oracle best candidate 的差距。
```

---

## 6. 并行执行策略

为了加速，不再顺序等待一条路线完整失败。Codex 可以并行启动以下工作流。

### Track A：Instrumentation

目标：Phase1 artifact coverage。

Codex tasks：

```text
A1. implement post_zp_delta_before_after.pt export
A2. implement per-layer/branch post-zp tensor summaries
A3. implement W_long/W_short tensor summaries
A4. implement overlap geometry target export
A5. implement window scale proxy export
```

如果某个 artifact 缺失，优先修工具，不跑 full。

### Track B：Sandbox candidate bank

目标：Phase2 true post-zp basis candidate。

Codex tasks：

```text
B1. generate PZBASIS candidate manifest
B2. run chunks 6/10 first, h5/h8/h10
B3. if h10 signal >= -1m, extend to h15
B4. if no h10 signal, stop family
```

### Track C：AUXGEO true replay

目标：Phase3 overlap-geometry replay。

Codex tasks：

```text
C1. pure geometry target first
C2. add structure semantic only after pure geometry shows signal
C3. restrict branch to w0 first
C4. if downstream regression, add continuity basis from Phase2
```

### Track D：Dual-bank true memory

目标：Phase4 true W_long/W_short。

Codex tasks：

```text
D1. implement true W_short apply / decay / clear
D2. test chunk10 h5/h8/h10 first
D3. compare W_short/W_long cosine
D4. if W_short hurts [400,600), add exit decay
```

### Track E：Selector proxy

目标：Phase5 no-GT selector。

Codex tasks：

```text
E1. build offline selector dataset from all candidate tables
E2. compute proxy correlations
E3. fit body selector and exit selector separately
E4. stop if Spearman < 0.5
```

### Track F：Full online validation

目标：只跑通过 gate 的候选。

Codex tasks：

```text
F1. run at most 2 candidates per batch
F2. include H9 and C9 repeat only if boundary has not been checked in same week
F3. generate global drift dashboard automatically
F4. do not accept candidate without counts_as_ttt_write=true
```

---

## 7. 失败分流规则

为了避免继续慢速无效实验，每个失败条件都有明确分流。

### 情况 1：h5 有信号，h10/h15 没信号

解释：短期局部 regularizer，不是 drift correction。

Codex 尝试方向：

```text
1. add lifecycle / longer active window;
2. add continuity protection;
3. if still decays, stop this candidate family.
```

### 情况 2：改善 `[200,300)`，但 `[400,600)` 回退

解释：删除或压制了 continuity direction。

Codex 尝试方向：

```text
1. split continuity basis and harmful basis;
2. add W_long continuity boost;
3. reduce negative coefficient;
4. add exit handoff correction;
5. if regression remains > +2m, stop candidate.
```

### 情况 3：Rot / FinalErr 改善，但 ATE 与 segments 不改善

解释：姿态 regularizer，不是 trajectory drift correction。

Codex 尝试方向：

```text
1. mark as regularizer;
2. do not full-expand;
3. only combine if main candidate already reduces ATE.
```

### 情况 4：selector proxy correlation 低

解释：proxy 不能选 candidate。

Codex 尝试方向：

```text
1. remove bad proxy features;
2. add overlap pointmap residual;
3. add per-reset scale proxy;
4. separate body/exit selector;
5. if Spearman still < 0.5, no full selector run.
```

### 情况 5：三类 true TTT action 都无上界

解释：TTT write-only 不足以成为 Target-25 主线。

Codex 尝试方向：

```text
1. keep best TTT as stabilizer;
2. build explicit online trajectory-state / scale-state module;
3. use TTT cues as features for the state module;
4. do not call offline postprocess a TTT success.
```

---

## 8. 预期决策点

### Decision D1：TTT true action 是否还有上界

在 Phase2/3/4 完成后判断。

```text
pass:
    at least one family reaches h10/h15 ATE delta <= -3m
    or [200,300) delta <= -5m without downstream regression

fail:
    all families best h10/h15 ATE delta > -1m
    and [200,300) delta > -2m
```

### Decision D2：是否允许 no-GT selector

只有 D1 pass 且 proxy audit pass 才允许。

```text
pass:
    Spearman >= 0.50
    Top1 hit rate >= 0.50

fail:
    no full online validation
```

### Decision D3：是否保留 TTT write-only 主线

在 Phase6 full online 后判断。

```text
keep TTT write-only mainline:
    full ATE <= 32.5
    or [200,300) improves >= 8m with ATE <= 33.5

reduce to regularizer:
    no candidate beats C9 by >= 0.5m
    and no candidate reduces [200,300) by >= 5m
```

### Decision D4：Target-25 是否需要 trajectory-state module

若 TTT write-only reduce to regularizer，则启动新主线：

```text
online scale-state / trajectory-state module
TTT cues used as features
no offline rewrite
no GT runtime
```

---

## 9. 最终备注

v17 的失败不是没有价值。它告诉我们：

```text
1. h3/h5 局部收益会衰减；
2. proxy action 不足以证明 true action；
3. dual-bank proxy 太弱；
4. overlap-geometry proxy 太弱；
5. 当前 TTT write-only 在旧接口下不是 Target-25 主线。
```

v18 的目标是把这个结论推进一步：不是继续证明旧接口失败，而是验证真实 fast-weight state action 是否存在。如果真实 action 也没有上界，就应该停止把 Target-25 压在 TTT write-only 上，转为更直接的 online trajectory-state / scale-state 模块。


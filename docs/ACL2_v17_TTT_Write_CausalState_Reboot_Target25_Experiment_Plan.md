# ACL2 v17：TTT Write Causal-State Reboot 与 Target-25 并行实验计划

日期：2026-05-19  
对象：LoGeR / HMC Pipeline v2 / KITTI01  
主目标：继续寻找真正有效的 TTT 写入策略，但停止低维标量微扫。  
当前在线 deployable baseline：

```text
H9_REPEAT / H9_P0_R2
ATE / Rot = 34.1258 / 6.5414
[200,300) = 74.410
[400,600) = 44.354
counts_as_ttt_write = true
```

当前非候选诊断上界：

```text
NOGTPOSE_27
ATE = 22.4 左右
counts_as_ttt_write = false
offline trajectory-state / scale-state diagnostic only
```

新目标：

```text
KITTI01 online deployable TTT-write ATE <= 25.0m
```

这份计划的出发点是：v16 已经修复了 v15 的 causal fork blocker，Phase 1 trajectory-level causal fork parity 已经通过；但是 Candidate Commit Bank 的现有 action family 只产生弱局部信号，尚不足以进入 no-GT selector 或 full online validation。下一阶段不能继续做 `gamma / neutral / read beta / weak freeze / post-zeropower mix` 的小修小补，而要重新定义 TTT 写入 action space。

---

## 1. 当前实验结果的独立判断

### 1.1 这轮 v16 的真正进展

v16 不是 TTT 写入策略突破，但它解决了一个非常重要的实验基础设施问题：**trusted short-horizon causal bank 终于可用了**。

v15 失败在：保存的 HMC state 能复现 fast-weight hash，但不能复现 full-run trajectory suffix。v16 加入 merge/gauge cursor save/load、raw prediction buffer summary、`global_chunk_offset` 后，Phase 1 已经能做到：

```text
H9 chunk5 h1/h3/h5: ATE diff = 0, raw pose diff = 0, HMC hash mismatch = 0
H9 chunk10 h3: pass
H9 chunk16 h3: pass
C9 chunk5/10/16 h3: pass
```

这意味着后续可以不再每个想法都跑 full KITTI01，而是在可信的 forked state 上做候选 fast-weight commit 的短 horizon 验证。这个工具价值很大，因为它把实验从“每条 full run 30 分钟”推进到“先用 sandbox 筛掉绝大多数 action”。

### 1.2 这轮 v16 没有找到好的 TTT 写入策略

Phase 2 Candidate Commit Bank 只产生弱局部信号。当前最好的短 horizon 结果是：

```text
K11_ORTHO_SUPPRESS_W0 at chunk10:
    h3 ATE delta vs H9 = -0.507m
    [200,300) eval-intersection delta = -1.029m

K21_ORTHO_SUPPRESS_ALL at chunk10:
    [200,300) eval-intersection delta = -1.766m
```

这些改善是真信号，但远低于 stage gate：

```text
h3/full-equivalent improvement >= 1.0m
or [200,300) improvement >= 5m without downstream regression
```

因此它不足以启动 no-GT selector，更不该进入 full online validation。更严重的是，chunk6 routing variants 非常不稳定，出现 h3 ATE 回退超过 7m 的情况。这说明当前 action family 对病灶入口处的 continuity 很脆弱。

### 1.3 scale-state diagnostic 也没有把信号接回 TTT

R7/R10 的 H5 follow-up 很有信息量：

```text
GT scale projection into TTT tri-replay risk:
    chunk6 / chunk10 均没有打开上界，甚至使 h3 ATE 变差。

explicit online scale-state module:
    chunk10 h3 改善约 0.13m
    [200,300) 改善约 0.40m
    chunk6 基本不变或略差
```

这说明：  
第一，window scale / trajectory-state 仍是关键误差方向；  
第二，当前把 scale residual 塞进 TTT tri-replay 的方式并不对；  
第三，显式 online scale-state 模块有一点点局部效果，但还远不够，也不能算 TTT write success。

### 1.4 当前离 Target-25 还差多远

以 H9 为主 deployable online baseline：

$$
Gap_{H9}=34.1258-25.0=9.1258m
$$

以 C9 这个 ATE 更低但 `[200,300)` 更差的在线策略为参考：

$$
Gap_{C9}=33.7629-25.0=8.7629m
$$

而 v16 sandbox candidate 的最好短 horizon 局部改善只有 $0.5m$ 到 $1.8m$ 量级。这和目标所需的 $8.8m$ 到 $9.1m$ 不是同一量级。下一阶段不能再以“比 H9 好 0.05m”为成功，必须要求 candidate 能在 short horizon 或 window horizon 中表现出 $3m$ 到 $5m$ 级别的真实上界，否则不值得 full run。

---

## 2. 问题本质：不是 cue 不够，而是 TTT action space 不够

### 2.1 过去的有效进展在哪里

过去的大进展主要有两段：

```text
1. Frame attention read cue:
   C23 past + pair/all，把系统从 38m 级推进到 36m 级。

2. TTT-native cue:
   update_conflict_energy + tri-replay，把系统从 36m 级推进到 34m 级。
```

这说明 TTT 内部 cue 不是没用。`update_conflict_energy` 确实暴露了一部分 TTT fast-weight update conflict。但是从 v8 到 v16，围绕这个 cue 的 action 基本都是：

```text
select chunks
set gamma
set neutral
set positive / negative fraction
commit EMA
native mix
weak freeze
orthogonal suppress
post-zeropower mix / cap
scale residual risk
```

这些动作多数仍然是低维 scalar routing。它们可以做 local regularization，也可以改 rotation、endpoint 或局部窗口，但不能直接解决 reset-window scale / trajectory-state drift。

### 2.2 为什么 Candidate Bank 没有打开上界

现有 Candidate Bank 的候选大多是对已有 fast-weight delta 做局部混合、抑制或替换。它们没有生成真正新的 memory hypothesis，也没有明确区分：

```text
continuity direction:
    保证后续 [400,600) 不崩的方向

drift-correction direction:
    能压低 [200,300) / [200,400) 的方向

harmful overwrite direction:
    会导致 reset-window scale / yaw / translation drift 的方向
```

因此它们经常出现两种失败：

```text
1. 修 [200,300)，但 [400,600) 崩；
2. 改 FinalErr / Rot，但 overall ATE 或 [200,300) 基本不动。
```

这不是某个候选参数的问题，而是 action 没有显式表示“方向分解”和“生命周期分离”。

### 2.3 为什么 scale-state 不能直接塞进当前 TTT replay

LoGeR 的 TTT fast weights是压缩 global memory，理论上和 scale/drift 有关；但当前 `f_W(k) -> v` 的 replay目标是 hidden feature-space target，不是 trajectory-state target。把 GT scale residual 或 online scale residual直接变成 token risk，再乘进 tri-replay，很可能发生两件事：

```text
1. zeropower / norm restoration 把幅度信息折叠；
2. hidden feature update 与 pose-scale residual 的因果方向不对齐。
```

所以 v16 的 scale projection失败并不说明 TTT 与 scale无关，而说明 **当前 TTT 写入接口没有把 scale-state 作为显式目标**。

---

## 3. 下一阶段总目标

下一阶段的目标不是立即 full-run 追 `25m`，而是建立一个新的判断链：

```text
Step A:
    trusted causal fork 已可用，先用它做短 horizon action upper-bound。

Step B:
    验证 TTT fast weights 中是否存在可控的 drift-correction subspace。

Step C:
    若存在，用 no-GT short-horizon selector 选择 candidate commit。

Step D:
    只有 selector 通过，才做 full KITTI01 online validation。

Step E:
    若不存在，明确把 TTT write-only 降级为 regularizer，同时转向 explicit online trajectory-state / scale-state module。
```

最终成功必须是：

```text
online HMC / TTT-write candidate
no GT runtime action
no offline trajectory rewrite
KITTI01 ATE <= 25.0m
```

中间强 gate 是：

```text
candidate-bank oracle:
    horizon h5/h8 improvement >= 3m
    or [200,300) improvement >= 5m
    and [400,600)-proxy does not regress

online validation:
    ATE <= 32.5 for first promoted family
    then ATE <= 30
    then ATE <= 25
```

---

## 4. 核心假设

### H1：现有 h3 sandbox 太短，无法判断 whole-window drift-correction potential

v16 的 Candidate Bank 主要使用 h3 short rollout。h3 对局部 state fork 很有用，但 Target-25 需要修的是 reset-window / trajectory-state drift。h3 的微小改善可能不足以反映 h8/h10 的长期效果，也可能把局部改善误判为全局无效。

假设：

```text
h5/h8/h10 horizon 能更好区分真正 drift-correction action 和局部 regularizer。
```

成立标准：

```text
存在 candidate 在 h8 或 h10 上：
    ATE delta <= -2.5m
    或 [200,300) / body-window delta <= -5m
    且 downstream proxy 不恶化超过 1m
```

若不成立：

```text
停止扩展同类 candidate bank；转向更强 action family。
```

### H2：TTT fast-weight delta 中存在可分离的 continuity / correction / harmful 子空间

freeze5/freeze56 早期诊断证明 chunks 5/6 可以大幅改变 `[200,300)`，但 hard freeze 会毁掉后续窗口。这说明同一个 TTT commit 中混有 positive continuity 和 harmful drift direction。

假设：

```text
post-zeropower fast-weight delta 可以通过 residual basis / SVD / contrastive state 差分分解出有用方向。
```

成立标准：

```text
构造的 basis candidate 在 sandbox oracle 中达到：
    h5/h8 ATE delta <= -2m
    or [200,300) delta <= -5m
    and no large downstream proxy regression
```

若不成立：

```text
TTT post-delta basis routing 不再作为主线；转向 auxiliary objective 或 explicit trajectory-state。
```

### H3：当前 TTT replay objective 缺少 geometry/scale target，需要加入 overlap-geometry auxiliary replay

现有 TTT update 的目标基本是 hidden feature association。Target-25 的主误差是 reset-window scale / trajectory-state，因此需要一个更贴近 geometry consistency 的 replay target。

假设：

```text
用 overlap pointmap scale consistency / structure-token pseudo target 做 auxiliary replay，比继续 token prior/risk 更能影响 drift。
```

成立标准：

```text
AUX-GEO candidate 在 sandbox h5/h8 中：
    [200,300) 或 body-window delta <= -4m
    或 full-equivalent score 明显优于 K11/K21
```

若不成立：

```text
不要继续加权 overlap pseudo replay；改为直接引入 trajectory-state module。
```

### H4：candidate commit selector 必须先有 oracle 上界，否则 no-GT selector 没意义

v16 Phase 2 没有 stage pass，因此 Phase 3 no-GT selector没有启动是正确的。Selector不是 magic，它只能在 candidate bank 里选；如果 candidate bank本身没有上界，selector 只会把噪声工程化。

成立标准：

```text
candidate bank oracle 至少满足：
    best h8/h10 ATE delta <= -3m
    or best [200,300) delta <= -5m
```

若不成立：

```text
Codex 不允许启动 no-GT selector full runs。
继续扩展 action family 或停止 TTT write-only target-25 主线。
```

### H5：若 TTT action family 连续失败，应把 TTT 降级为 regularizer，而不是继续小修小补

成立标准：

```text
三类强 action family：
    residual basis routing
    overlap-geometry auxiliary replay
    dual-bank / long-short TTT
若均无法在 sandbox oracle 超过 -3m h8 gain，
则 TTT write-only 不再是 Target-25 主线。
```

---

## 5. 实验阶段设计

## Phase 0：锁定可信边界与并行资源规则

### 目标

继续保护 v16 已经修好的 causal fork，不让后续结果被配置漂移污染。

### 固定边界

每次新 action family 首批必须包含：

```text
H9 reference:
    ATE 34.1258 / Rot 6.5414
    [200,300) 74.410
    [400,600) 44.354

C9 reference:
    ATE 33.7629 / Rot 6.5259
    [200,300) 76.102
    [400,600) 41.896

WINGAM reference:
    ATE 34.1903 / Rot 6.5666
    [200,300) 75.576
    [400,600) 42.280
```

Phase 0 不需要每次重跑 full，但必须检查：

```text
run_config.yaml hash
hmc_config.yaml
mp_alpha
reset_every
read_beta_frame_chunks
TTT_WRITE_* env
SWA fixed protocol
global_chunk_offset
HMC/merge snapshot hash
```

### 通过标准

```text
所有 boundary config hash 与 v16 locked reference 一致；
sandbox native fork h3 parity 仍为 0 diff；
若任一 hash 不一致，禁止启动 candidate action。
```

### Codex 自动分流

如果 Phase 0 失败：

```text
1. 先跑 config_diff_report；
2. 对比 H9/C9/WINGAM env；
3. 修复 launcher；
4. rerun boundary only；
5. 不允许跑 candidate bank。
```

---

## Phase 1：从 h3 升级到 h8/h10 trusted rollout

### 目标

判断 v16 Candidate Bank 的弱信号是否只是因为 horizon 太短。

### 运行方式

在 causal fork 中使用相同 HMC/merge snapshot，新增 horizon：

```text
horizon = 5
horizon = 8
horizon = 10
```

优先 chunks：

```text
chunk5
chunk6
chunk10
chunk16
```

优先 candidates：

```text
K1_H9
K11_ORTHO_SUPPRESS_W0
K21_ORTHO_SUPPRESS_ALL
K13_ORTHO_RHO025_W0
K14_TTGR_ZERO_ORTHO_W0
K25/K26 online-scale-state diagnostic, but marked non-TTT success
```

### 必须记录

```text
short_rollout_metrics_h5_h8_h10.csv
candidate_vs_H9_delta_by_horizon.csv
candidate_window_segment_intersection.csv
candidate_downstream_proxy.csv
hmc_state_hash.jsonl
merge_state_hash.jsonl
```

字段：

```text
candidate_id
chunk_id
horizon
ATE_horizon
ATE_delta_vs_H9
Rot_horizon
FinalErr_horizon
intersection_[200,300]_ATE
intersection_[200,300]_delta
intersection_[400,600]_proxy
raw_pose_max_abs_diff
raw_trans_max_diff
hmc_hash_mismatch
merge_hash_mismatch
```

### 可视化

```text
horizon_gain_heatmap:
    x = horizon
    y = candidate
    value = ATE delta vs H9

segment_gain_heatmap:
    x = candidate
    y = [200,300], [200,400], [400,600]
    value = delta vs H9

trajectory_suffix_overlay:
    H9 vs candidate for h8/h10 windows
```

### 成立标准

```text
pass:
    exists candidate with h8 or h10 ATE delta <= -2.5m
    or [200,300) delta <= -5m
    and downstream proxy regression <= +1m

weak:
    exists candidate with h8/h10 delta <= -1.5m
    and no downstream regression

fail:
    all candidates h8/h10 delta > -1.5m
    or any [200,300) improvement is paired with large downstream regression
```

### 不满足条件时 Codex 尝试方向

如果 h3 有改善但 h8/h10 消失：

```text
Codex should test whether candidate is local regularizer:
    compare apply mismatch over horizon
    compare scale proxy over horizon
    log candidate effect decay curve
Then stop this candidate family.
```

如果 h8/h10 有信号：

```text
Codex should expand same family locally:
    rho / gamma small grid
    branch w0 vs all
    chunk10 plus chunk16 handoff
but only in sandbox, no full run.
```

---

## Phase 2：Contrastive fast-weight residual basis routing

### 目标

不再直接用 scalar risk，而是从已知状态差分中构造 fast-weight residual basis，试图分离 continuity / correction / harmful directions。

### 输入状态

使用历史状态作为 contrastive anchors：

```text
H9:
    current best online TTT baseline

C9:
    ATE lower but [200,300) worse, [400,600) better

WINGAM:
    windowed tri-replay reference

freeze5 / freeze56 diagnostics:
    strong [200,300) improvement but downstream collapse

K11/K21:
    v16 weak local positive candidates
```

### 构造 basis

对每个 chunk、layer、branch，取 post-zeropower delta：

$$
\Delta W_{run,m,l,b}=W_{after,m,l,b}^{run}-W_{before,m,l,b}^{run}
$$

构造：

$$
B_{body}=\Delta W_{freeze56}-\Delta W_{H9}
$$

$$
B_{continuity}=\Delta W_{C9}-\Delta W_{H9}
$$

$$
B_{wingam}=\Delta W_{WINGAM}-\Delta W_{H9}
$$

$$
B_{local}=\Delta W_{K11}-\Delta W_{H9}
$$

然后对 candidate delta 做投影：

$$
\Delta W'=\Delta W-\alpha P_{harm}(\Delta W)+\beta P_{corr}(\Delta W)+\lambda P_{cont}(\Delta W)
$$

其中：

$$
P_B(\Delta W)=\frac{\langle \Delta W,B\rangle}{\|B\|^2+\epsilon}B
$$

### Candidate family

```text
BASIS_01:
    subtract harmful basis from chunk5/chunk6 w0 only

BASIS_02:
    subtract harmful basis, add continuity basis

BASIS_03:
    project out orthogonal harmful residual only

BASIS_04:
    apply to layers with high update_conflict_energy only

BASIS_05:
    apply only on post-zeropower delta before norm restoration

BASIS_06:
    branch-separated: w0 harmful removal, w2 continuity preserve
```

### 必须记录

```text
basis_projection_summary.csv
basis_norm_by_layer_branch.csv
candidate_basis_coefficients.jsonl
post_zp_delta_before_after.pt
norm_restore_ratio_before_after.csv
short_rollout_h5_h8_h10.csv
```

字段：

```text
chunk_id
layer_id
branch_id
basis_name
coeff_alpha
coeff_beta
delta_norm_before
delta_norm_after
cos_to_H9
cos_to_C9
cos_to_freeze56
cos_to_WINGAM
horizon_ATE_delta
segment_delta_[200,300]
downstream_delta
```

### 可视化

```text
layer_branch_basis_cosine_heatmap
basis_coeff_over_chunks
delta_norm_before_after_heatmap
trajectory_suffix_overlay_h8
[200,300]_vs_[400,600]_tradeoff_plot
```

### 成立标准

```text
strong:
    h8/h10 ATE delta <= -3m
    or [200,300) delta <= -6m
    and downstream regression <= +1m

stage:
    h8/h10 ATE delta <= -2m
    and no downstream regression

fail:
    no candidate exceeds -1.5m h8/h10
    or [200,300) gain always causes [400,600) regression > +2m
```

### 不满足条件时 Codex 尝试方向

如果 all basis candidates fail：

```text
Codex should verify:
    Are basis tensors aligned by same layer/branch/chunk?
    Are deltas pre- or post-norm restoration?
    Are freeze deltas comparable to non-freeze deltas?
If implementation is correct, stop residual basis routing and move to Phase 3.
```

如果 harmful removal helps but downstream regresses:

```text
Codex should try:
    add continuity basis
    reduce alpha
    apply only selected layers
    create long/short dual bank from harmful basis
```

---

## Phase 3：Overlap-geometry auxiliary TTT replay objective

### 目标

用几何一致性构造更直接的 TTT replay target，而不是继续只对 hidden-feature replay 做 token risk。

### 动机

v12 的 auxiliary replay失败，因为实际实现是 frame-static feature-center proxy，不是真正的 scale-risk-gated structure replay。下一阶段要实现真实的 overlap geometry replay objective。

### 设计

对当前 chunk 与 previous overlap，构造 structure token 集合：

```text
structure tokens:
    low D_g
    high C_anchor or high stage_d
    semantic group = road/building/wall/fence if available
    confidence high
```

构造 geometry target：

```text
overlap pointmap scale consistency
local relative pose consistency
same-surface patch feature target
```

最小形式：

$$
v_i^{geo}=v_i+\rho \cdot \operatorname{stopgrad}(v_{overlap\_static}-v_i)
$$

或者在 K/V replay 层：

$$
L_{TTT}=L(f_W(k_i),v_i)+\lambda_{geo}L(f_W(k_i^{geo}),v_i^{geo})
$$

### Candidate family

```text
AUXGEO_01:
    overlap static pseudo-V, w0 only

AUXGEO_02:
    overlap static pseudo-KV, w0 only

AUXGEO_03:
    structure tokens only, w0+w2

AUXGEO_04:
    body window chunks 5-9 only

AUXGEO_05:
    chunk10/16 handoff only

AUXGEO_06:
    scale-stable structure replay with low D_g + high anchor
```

### 必须记录

```text
overlap_geometry_replay_debug.jsonl
pseudo_target_norm.csv
pseudo_target_cos_to_native.csv
structure_token_count.csv
per_group_update_norm.csv
short_rollout_h5_h8_h10.csv
```

字段：

```text
chunk_id
num_structure_tokens
num_overlap_tokens
pseudo_target_norm
pseudo_target_cosine_to_native_v
pseudo_target_cosine_to_original_v
branch_update_norm
layer_update_norm
horizon_ATE_delta
segment_[200,300]_delta
downstream_delta
```

### 可视化

```text
structure_token_overlay
overlap_static_source_map
pseudo_target_norm_heatmap
candidate_vs_H9_suffix_trajectory
scale_proxy_over_horizon
```

### 成立标准

```text
strong:
    [200,300) delta <= -5m without downstream regression
    or h10 ATE delta <= -3m

stage:
    h8/h10 ATE delta <= -2m
    and proxy scale drift improves

fail:
    best candidate < -1.5m or all improvements are Rot/FinalErr only
```

### 不满足条件时 Codex 尝试方向

如果 AUXGEO worsens ATE but improves Rot:

```text
Codex should reduce lambda_geo and branch scope to w0 only.
```

如果 AUXGEO improves chunk10 but not chunk6:

```text
Codex should split body vs exit:
    body uses weak lambda
    exit uses stronger structure replay
```

如果 no structure token has stable target:

```text
Codex should fallback to Phase 4 dual-bank memory.
```

---

## Phase 4：Dual-bank TTT memory with long/short lifecycle

### 目标

把当前 TTT fast weights拆成：

```text
W_long:
    只承载长期 scale / structure / continuity

W_short:
    承载短期 correction / dynamic adaptation
    apply for K chunks
    不直接进入长期 memory
```

这不是 one-hop subtract。one-hop subtract失败的原因是 transient delta 不是纯 harmful correction；dual-bank 要在 commit 时先分解角色，再决定生命周期。

### 最小实现

当前 apply 使用：

$$
W_{apply}=W_{long}+\mu_m W_{short}
$$

commit 使用：

$$
W_{long}^{m+1}=W_{long}^{m}+\Delta W_{pos}+\lambda_{neu}\Delta W_{neu}
$$

$$
W_{short}^{m+1}=\tau W_{short}^{m}+\Delta W_{corr}
$$

其中：

```text
positive:
    low conflict, structure, high anchor

neutral:
    lowstuff but low D_g, continuity support

correction:
    high update_conflict_energy, high apply mismatch, not structure
```

### Candidate family

```text
DLBANK_01:
    W_short from high-conflict tokens, K decay tau=0.5

DLBANK_02:
    W_short from harmful basis residual, tau=0.5

DLBANK_03:
    W_long only structure/lowD, W_short all other update

DLBANK_04:
    chunk5-9 body uses short correction, chunk10-12 exit drains W_short

DLBANK_05:
    W_short apply only to next 2 chunks, no long commit

DLBANK_06:
    W_short branch w0 only, W_long w0+w2
```

### 必须记录

```text
dual_bank_state_hash.jsonl
W_long_norm.csv
W_short_norm.csv
short_to_long_cosine.csv
short_decay_curve.csv
apply_W_short_mass.csv
short_rollout_h5_h8_h10.csv
```

字段：

```text
chunk_id
layer_id
branch_id
W_long_norm
W_short_norm
cos_Wshort_Wlong
short_apply_scale
short_decay_tau
pos_mass
neutral_mass
correction_mass
horizon_ATE_delta
segment_delta
downstream_delta
```

### 可视化

```text
W_long/W_short norm over chunks
short memory decay curve
role mass over reset group
trajectory suffix H9 vs dual-bank
```

### 成立标准

```text
strong:
    h10 ATE delta <= -3m
    and [200,300) delta <= -5m
    and downstream regression <= +1m

stage:
    h8/h10 ATE delta <= -2m
    and W_short decays without causing post-window collapse

fail:
    W_short improves local segment but worsens downstream by >2m
    or W_short has no measurable effect
```

### 不满足条件时 Codex 尝试方向

If W_short helps [200,300) but hurts downstream:

```text
reduce apply_scale
increase decay
move correction to exit drain window
preserve W_long structure tokens
```

If W_short has no effect:

```text
check whether TTT apply path actually uses W_short;
increase apply_scale in smoke only;
inspect apply mismatch change.
```

---

## Phase 5：No-GT candidate selector only after oracle upper bound

### 前置条件

不得启动 no-GT selector，除非 Phase 2/3/4 至少一个 family 满足：

```text
h8/h10 ATE delta <= -3m
or [200,300) delta <= -5m without downstream regression
```

### Selector candidate proxies

如果满足前置条件，才构建 no-GT selector：

```text
overlap pointmap residual
relative step-length median drift
TTT apply mismatch
SWA/TTT consistency
high-conflict mass trend
structure-token replay residual
W_short norm decay
chunk-to-chunk pose jump proxy
```

### Selector 训练 / 校准

只能用 candidate bank oracle table 离线拟合，不用 GT runtime：

$$
score(c)=\sum_k w_k p_k(c)
$$

要报告：

```text
Spearman(score, oracle ATE)
Top-1 oracle regret
Top-3 recall
bad-choice rate
```

### 成立标准

```text
selector Spearman >= 0.6
Top-1 regret <= 0.5m
bad-choice rate <= 20%
```

若不满足：

```text
不跑 full online selector；
Codex 改进 proxy 或回到 action family。
```

---

## Phase 6：Full online validation

### 前置条件

只有通过 Phase 5 selector gate 的 candidate才进入 full run。

### Full run 候选上限

每个 family首轮最多：

```text
2 full KITTI01 runs
```

不得无上界全铺。

### 必须记录

```text
kitti_benchmark.log
trajectory_diagnostics.json
global_drift_dashboard/
per_frame_error.csv
per_chunk_error.csv
hmc_state_hash.jsonl
candidate_decision_log.jsonl
```

### 成功标准

```text
Stage success:
    ATE <= 32.5
    [200,300) <= 68
    [400,600) not worse than H9 by > 1m

Strong success:
    ATE <= 30.0
    [200,300) <= 55

Final success:
    ATE <= 25.0
    no GT runtime
    no offline trajectory rewrite
    counts_as_ttt_write = true
```

### 失败自动分流

If ATE improves but [200,300) worsens:

```text
Codex should classify as downstream-reallocation strategy.
Do not continue full runs.
Return to basis/dual-bank to preserve body-window correction.
```

If [200,300) improves but [400,600) worsens:

```text
Codex should classify as continuity-breaking strategy.
Try W_long/W_short separation or continuity basis add-back.
```

If neither improves:

```text
Stop that family.
```

---

## 6. 并行执行方案

为了加快进度，下一阶段采用 sandbox-first 并行，不再 full-first。

### 并行组 A：Horizon 扩展

```text
Owner: Codex-A
Task:
    Run h5/h8/h10 for K11, K21, K13, K14 on chunk10 and chunk6.
Deliver:
    horizon_gain_heatmap.csv
    stop/pass decision
```

### 并行组 B：Residual basis routing

```text
Owner: Codex-B
Task:
    Extract post-zp delta basis from H9/C9/WINGAM/freeze diagnostics.
    Run BASIS_01-06 in sandbox only.
Deliver:
    basis_cosine_heatmap
    h8/h10 oracle table
```

### 并行组 C：Overlap-geometry auxiliary replay

```text
Owner: Codex-C
Task:
    Implement true structure-token overlap pseudo replay.
    Run AUXGEO_01-06 in sandbox.
Deliver:
    pseudo_target_debug
    h8/h10 oracle table
```

### 并行组 D：Dual-bank memory

```text
Owner: Codex-D
Task:
    Implement W_long/W_short smoke.
    Verify W_short affects apply path.
    Run DLBANK_01-03 sandbox.
Deliver:
    dual_bank_debug
    short rollout table
```

### 并行组 E：Metrics and dashboard

```text
Owner: Codex-E
Task:
    Build unified v17 dashboard.
    No model changes.
Deliver:
    all candidate tables, plots, selector preconditions.
```

### 并行资源规则

```text
Sandbox:
    May run high parallelism if GPU/CPU safe.
    Prefer 6-8 short rollout jobs.

Full KITTI01:
    Only after gate.
    Max 2-4 concurrent jobs.
    Never run 8 full jobs due RAM pressure.

Long-sequence cross-seq:
    Not before KITTI01 strong candidate.
```

---

## 7. 最终停止规则

### Stop scalar sweeping

永久停止以下方向作为主线：

```text
gamma micro sweep
neutral lambda micro sweep
read beta micro sweep
chunk16 tiny gamma sweep
commit EMA 0.9/1.1 sweep
post-zeropower mix/cap scalar sweep
simple weak freeze variants
```

除非它们是某个新 action family 的 smoke，不得作为独立 full run。

### Stop TTT write-only target-25 if all strong families fail

若下面三类都失败：

```text
residual basis routing
overlap-geometry auxiliary replay
dual-bank memory
```

并且没有任何候选在 h8/h10 达到：

```text
ATE delta <= -2m
or [200,300) delta <= -5m
```

则结论应更新为：

```text
TTT write-only is not the Target-25 mainline under current architecture.
TTT remains a regularizer.
Target-25 should move to explicit online trajectory-state / scale-state module,
while keeping TTT write for stability.
```

这个结论不是放弃 TTT，而是停止把 target-25 完全压在 TTT 写入上。

---

## 8. 本阶段预期产物

```text
acl2_v17_horizon_expansion_report.md
acl2_v17_basis_routing_report.md
acl2_v17_auxgeo_replay_report.md
acl2_v17_dualbank_report.md
acl2_v17_candidate_bank_oracle_table.csv
acl2_v17_selector_precondition_report.md
acl2_v17_full_validation_registry.csv
acl2_v17_next_stage_decision.md
```

`acl2_v17_next_stage_decision.md` 必须回答：

```text
1. TTT candidate bank 是否存在 target-25 级别上界？
2. 哪个 action family 最有希望？
3. 是否允许启动 no-GT selector？
4. 是否允许启动 full online validation？
5. 如果不允许，下一步是继续 TTT 强 action，还是转 explicit trajectory-state？
```

---

## 9. 一句话总结

v16 的价值在于：**trusted causal fork 终于可用**。但 v16 同时说明：**当前 fast-weight action candidates 只有弱局部信号，不足以支持 Target-25，也不足以启动 selector/full validation**。v17 的核心不是继续小修小补，而是用 sandbox 并行验证更强的 TTT action space：residual basis routing、overlap-geometry auxiliary replay、dual-bank TTT memory。如果这些仍然没有上界，就应明确把 TTT write-only 从 Target-25 主线降级为 regularizer，并转向显式 online trajectory-state / scale-state 模块。

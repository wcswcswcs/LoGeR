# ACL2 v14：TTT Write Causal Sandbox 与 Target-25 结构性实验计划

日期：2026-05-18 之后  
对象：LoGeR / HMC Pipeline v2 / KITTI01 / TTT write policy  
当前 deployable online best：`H9_REPEAT = 34.1258 / 6.5414`  
当前 v13 deployable online tiny best：`C9_WEAK_FREEZE_C56 = 33.7629 / 6.5259`  
最终目标：`KITTI01 ATE <= 25.0m`  
本计划约束：**不把 offline trajectory rewrite、GT runtime oracle、postprocess 结果计入 TTT write success。**

---

## 0. 本轮实验的独立判断

v13 没有找到令人满意的 TTT 写入策略。`C9_WEAK_FREEZE_C56` 是一个真实 online TTT/HMC 改进点，它把 ATE 从 `H9_REPEAT = 34.1258m` 降到 `33.7629m`，改善约 `0.3629m`。但是它没有达到 v13 的 useful gate，也远没有接近 Target-25；更重要的是，它把核心病灶段 `[200,300)` 从 `74.410m` 恶化到 `76.102m`，只是把后段 `[400,600)` 从 `44.354m` 改善到 `41.896m`。

因此，`C9` 不是“好的 TTT 写入策略”，而是一个 **error redistribution policy**：它缓解了后段 drift 或 endpoint-like drift，但加重了 body 病灶窗口。它说明 TTT 写入确实还能影响长期轨迹状态，但当前 action 仍然不能同时控制 body window 和 downstream window。

v13 的 finite-difference commit-EMA sensitivity 也没有打开空间。chunk5/chunk6 的 `w0` commit-EMA alpha `0.90/1.10` 四条 full run 全部没有超过 `H9_REPEAT`，也没有让 `[200,300)` 出现 `>=3m` 的实质改善。Post-zeropower delta routing 同样失败：最好 `PZ_04_EXIT_MIX025 = 34.7590m`，明显差于 H9；`PZ_05_W1_CAP050` 虽然把 `[200,300)` 压到 `68.615m`，但 ATE 崩到 `38.2047m`，`[400,600)` 恶化到 `56.087m`。

这说明当前问题不再是“还差一个更好的 gamma / neutral / branch mask”。当前真正卡住的是：

> **我们还没有找到一个 TTT fast-weight action，可以把 window-level scale / trajectory-state drift 作为目标来控制，同时保留局部 continuity。**

所以 v14 的目标不是继续微调 `C9`、`H9`、`PZ`，而是建立一个 **TTT causal sandbox**：先用短 horizon、可复现、可并行的 counterfactual commit 实验判断 TTT fast weights 中是否存在可控的 drift-correction subspace；如果存在，再把它转成 deployable online policy；如果不存在，就及时停止这一类 TTT write-only 搜索，转向更大的 TTT action 设计，而不是继续扫标量。

---

## 1. 当前阶段的问题本质

### 1.1 已经证明的事实

到目前为止，系统已经证明了三类事实。

第一，frame-attention read cue 是有效的。`C23 past + pair/all` 是目前最大的 read-side 跃迁来源，它比早期 TTT write-score 调整大得多。这个事实说明 `D_g` 更像 read-path harmful-support cue，而不是天然的 TTT write eligibility。

第二，TTT write scalar policy 已经平台化。`stage_d * sqrt(1-D_g)`、sparse write、explicit-dyn veto、semantic scalar、localized TTGR、tri-replay、body/exit gamma、read-beta cooling、candidate policy bank、finite-difference alpha、post-zeropower routing都只在 `34m` 平台附近移动，无法逼近 `25m`。

第三，Target-25 的主要杠杆不是普通 token 写入强弱，而是 reset-window scale / trajectory-state。`NOGTPOSE_27` 这个 offline no-GT trajectory-state proxy 能到 `22.4012m`，但它不是 online TTT write，也不是 deployable TTT success。它的意义是证明：当前 `34m` 平台里存在很强的 window-level scale/trajectory-state误差；如果能在 online pipeline 中控制这个状态，就有可能接近 Target-25。

### 1.2 当前 TTT write action 为什么不够

当前 TTT write controller 的大多数动作是低维门控：

```text
token prior
branch mask
layer mask
gamma
neutral lambda
positive / negative fraction
native mix
commit EMA
post-zeropower cap / mix
```

这些动作在 hidden fast-weight update 空间里调权重，但最终指标是 camera trajectory 的 window-level scale / drift。中间有三层间接映射：

```text
TTT replay token -> fast-weight delta -> future hidden apply -> pose / pointmap trajectory
```

只调 token prior，很容易被以下机制折叠掉：

```text
Muon / zeropower 方向归一化
norm restoration
branch interaction
SWA / frame-attention read correction
Sim3 global alignment
reset_every=5 lifecycle
```

因此，当前 TTT write 不是没有信号，而是表达能力不够直接。它能做 regularization，却不能直接消除 whole-scene drift。

### 1.3 新目标与差距

当前 deployable online HMC/TTT best 是：

```text
H9_REPEAT:
    ATE = 34.1258m

v13 tiny deployable best:
    C9_WEAK_FREEZE_C56:
    ATE = 33.7629m
```

新目标是：

```text
Target:
    KITTI01 ATE <= 25.0m
```

差距为：

$$
Gap_{H9}=34.1258-25.0=9.1258m
$$

$$
Gap_{C9}=33.7629-25.0=8.7629m
$$

这个 gap 不是 `0.01m` 到 `0.05m` 的微扫能解决的。因此 v14 必须把实验目标改成：

> **验证新的 TTT action family 是否存在至少 1m 到 3m 级别的可部署空间。**

如果一个方向连续 4 条 full run 只能带来 `<0.1m` 的收益，立即停止。

---

## 2. v14 总体目标

v14 的目标不是直接刷一个更小 ATE，而是建立一个清晰的决策链：

```text
Step 1: TTT fast weights 是否真的能因果影响未来 window drift？
Step 2: 如果能，哪些 layer / branch / chunk / role 是有效 action scope？
Step 3: 是否存在不依赖 GT 的短 horizon selector，可以选择更好的 TTT commit state？
Step 4: 选出的 TTT commit 是否能在 full online KITTI01 上超过 H9/C9，并同时改善 body 与 downstream window？
Step 5: 如果不能，是否应停止 TTT write-only scalar/controller 线，转入更大的 TTT objective / trajectory-state 模块？
```

v14 的核心实验对象不是后处理轨迹，而是 **真实 online HMC commit state**。所有成功候选必须满足：

```text
counts_as_ttt_write = true
output_from_online_hmc = true
no GT runtime action
no offline trajectory rewrite
commit affects future HMC state
hmc rows = 38
```

---

## 3. 核心假设

## H1：当前 TTT write 平台的根因是 action space 不够，而不是 cue 不够

### 假设内容

当前 `update_conflict_energy`、`D_g`、semantic prior 和 old dynamic cues 都有局部信号，但它们被用于低维 scalar gate 后，无法直接修 window-level drift。因此继续寻找更好的 scalar score 不会接近 Target-25。

### 实验要验证什么

如果 H1 成立，应该观察到：

```text
1. scalar / finite-diff / post-zeropower 小矩阵不超过 H9 或 C9；
2. 局部改善某个 segment 时，另一个 segment 系统性恶化；
3. TTT delta 的 norm/cosine 有变化，但 future window scale proxy 没有可靠改善；
4. candidate full-run 的 RPE 基本不动，说明局部里程计没有被修，只是全局形状重新分配。
```

### 判断标准

H1 成立后，停止这些实验族：

```text
gamma 微扫
neutral lambda 微扫
single chunk weak freeze 微扫
commit EMA alpha 微扫
post-zeropower scalar mix 微扫
read beta 小范围微扫
```

并进入 candidate commit bank / short rollout sensitivity。

---

## H2：TTT 仍可能有用，但必须通过 candidate commit bank 而不是单一路径 commit

### 假设内容

当前每个 chunk 只生成一个 `W_{m+1}`。但实际可能有多个 plausible TTT commit states：

```text
native / H9 default
tri-replay body/exit
weak-freeze C56
post-zeropower filtered
positive-only
neutral-preserved
scale-aware replay
```

单一固定规则很难在 body window 和 downstream window 同时最优。更合理的是：每个关键 chunk 生成多个 candidate fast-weight commit，然后用 no-GT short-horizon consistency score 选择最终提交。

### 关键区别

这不是 postprocess，也不是改输出轨迹。candidate selector 只做：

```text
选择下一个 committed TTT state H_{m+1}
```

它必须发生在未来 chunk forward 之前。

### 判断标准

先做 oracle upper-bound：用 GT 只在离线分析中判断每个 chunk 的 best candidate 是否存在。如果 oracle bank 都不能让 full ATE 或 local rollout显著优于 H9/C9，则 candidate-bank action space 不足，不能再做 no-GT selector。

---

## H3：短 horizon rollout 比全序列 full run 更适合作 TTT action search

### 假设内容

每条 full run 约 25-35 分钟，且很多策略只在一个 window 内有作用。全序列 brute-force 太慢。应该先把关键 chunk 的 committed state 保存下来，对 candidate commit 做 `K=2/3/5` chunk short rollout。

### 目标

短 rollout 应回答：

```text
某个 TTT commit candidate 是否会改善未来 2-5 个 chunk 的：
    overlap pointmap consistency
    pose increment scale proxy
    TTT apply mismatch
    D_g-weighted high-risk region error proxy
    GT audit segment ATE
```

只有 short rollout 通过的 candidate 才进入 full run。

### 判断标准

short rollout selector 必须满足：

```text
Spearman(proxy_score, GT_future_ATE) <= -0.45
Top-1 candidate regret <= 0.25m local future ATE
Top-2 recall of oracle-best >= 0.60
```

若不满足，不能启动 no-GT selector full。

---

## H4：TTT write success 必须同时改善 body window 和 downstream window

### 假设内容

`C9` 改善 ATE 是因为 downstream `[400,600)` 改善，但 body `[200,300)` 变差。`PZ_05` 改善 `[200,300)`，但 downstream 和 ATE 崩。真正好的策略应该同时降低两个窗口，或者至少不明显牺牲其中任何一个。

### 关键 segment gate

候选必须同时报告：

```text
[200,300)
[200,400)
[400,600)
[0,100)
[100,200)
[600,800)
FinalErr
YawRMSE
Sim3Scale
```

成功分层：

```text
Stage weak success:
    ATE <= 33.0
    and [200,300) <= 73.0
    and [400,600) <= 43.0

Stage useful success:
    ATE <= 32.0
    and [200,300) improves >= 5m vs H9
    and [400,600) does not regress vs H9

Target-path success:
    ATE <= 30.0
    and [200,400) <= 45.0
    and [400,600) <= 35.0

Final target:
    ATE <= 25.0
```

如果某个策略只改善一个窗口但另一个窗口明显恶化，标记为 trade-off diagnostic，不晋级。

---

## H5：如果 TTT candidate bank oracle 没有上界，必须换 TTT action，而不是继续调参

### 假设内容

如果多个真实 online TTT candidate commits 在 oracle/offline local rollout 下都不能显著改善 future window，那么问题不是 selector 不好，而是 TTT action family 没有上界。

### 触发条件

满足任一条件，停止该 action family：

```text
1. oracle candidate bank best full ATE > 33.5；
2. oracle short rollout 没有 candidate 能让 [200,300) 改善 >= 5m；
3. no-GT selector Top-1 regret > 0.5m；
4. 4 条 full run 连续未超过 H9；
5. 任何 candidate 的 best 改善主要来自 Rot/FinalErr，但 ATE 和 segments 不改善。
```

停止后 Codex 应自动尝试下一类 action family，而不是继续在同类 action 中细扫。

---

## 4. 实验阶段总览

v14 分为六个并行/串行阶段。

```text
Phase 0: 边界复现与数据锁定
Phase 1: TTT Causal Sandbox 与 state save/load
Phase 2: Candidate Commit Bank Oracle
Phase 3: No-GT Short-Horizon Selector
Phase 4: New TTT Action Families
Phase 5: Full Online Validation 与 Cross-Sequence Gate
Phase 6: Failure Routing / Codex 自动尝试方向
```

---

# Phase 0：边界复现与实验锁定

## 目标

确保当前对照全部可复现，避免后续改善来自工程漂移。

## 必跑 run

```text
P0-01 H9_REPEAT
P0-02 C9_WEAK_FREEZE_C56 repeat
P0-03 PZ_05_W1_CAP050 repeat only if needed for segment diagnostic
P0-04 NOGTPOSE_27 diagnostic repeat, flagged non-TTT
```

## 必须记录

```text
metrics_global.json
trajectory_diagnostics.json
segment_100f.csv
segment_200f.csv
hmc_state_hash.jsonl
run_config.yaml
counts_as_ttt_write flag
no_postprocess flag
uses_gt_runtime_action flag
runtime_summary.json
```

## 通过标准

```text
H9 repeat |ATE - 34.1258| <= 0.03m
C9 repeat |ATE - 33.7629| <= 0.03m
NOGTPOSE_27 flagged counts_as_ttt_write=false
all online runs hmc rows = 38
```

如果 Phase 0 不通过，不进入 Phase 1。

---

# Phase 1：TTT Causal Sandbox

## 目标

建立一个可以从任意 chunk 的 committed HMC state 开始，快速测试候选 TTT commit 的 sandbox。它用于代替全序列暴力跑参。

## 关键要求

Sandbox 必须能做到：

```text
1. 保存 chunk m 之前的 committed HybridMemoryState；
2. 从相同 state 重放 probe/control；
3. 生成多个 candidate TTT commit；
4. 只 rollout 未来 K 个 chunk；
5. 记录 no-GT proxy 与 GT audit；
6. sandbox candidate 在 full run 中可复现。
```

## 工程任务

### 1. State snapshot

每个 full baseline run 保存：

```text
state_before_chunk_{m}.pt
probe_cache_chunk_{m}.pt
control_summary_chunk_{m}.json
write_cache_chunk_{m}.pt
swa_history_chunk_{m}.pt
```

保存对象至少包含：

```text
TTT w0/w1/w2
SWA history
frame attention read config
D_g / cue maps
TTT replay primitives
commit metadata
```

### 2. Short rollout runner

新增工具：

```text
tools/ttt_short_rollout_sandbox.py
```

输入：

```text
--state_snapshot_dir
--start_chunk
--horizon_chunks 2/3/5
--candidate_policy name
--output_dir
```

输出：

```text
short_rollout_metrics.csv
short_rollout_gt_audit.csv
short_rollout_proxy.jsonl
candidate_commit_debug.jsonl
```

### 3. Sandbox correctness check

对 H9 candidate 做 sandbox rollout，必须与 H9 full run 中相同 chunk range 的 metrics 对齐。

## 指标

### GT audit 指标

这些只用于研究，不允许进入 deploy selector：

```text
future_window_ATE
future_window_Rot
future_window_scale_error
future_segment_[200,300)_proxy_if_overlap
future_chunk_RMSE
```

### no-GT proxy 指标

这些可以用于 selector：

```text
overlap_pointmap_residual
pose_increment_scale_ratio
step_length_median_ratio
TTT_apply_mismatch_mean
TTT_apply_mismatch_p90
frame_attention_highD_mass
D_g_weighted_confidence_drop
SWA_overlap_consistency
local_pointmap_seam_error
```

## 通过标准

Phase 1 通过条件：

```text
1. H9 sandbox replay 与 H9 full 对应 horizon ATE 差 <= 0.05m；
2. candidate commit 的 state hash 正确变化；
3. short rollout runtime <= full runtime 的 35%；
4. no-GT proxy 和 GT audit 都成功落盘；
5. no offline trajectory rewrite。
```

如果 Phase 1 失败，Codex 优先修 state snapshot / replay，不允许直接继续 full-run 大矩阵。

---

# Phase 2：Candidate Commit Bank Oracle

## 目标

先验证 TTT action space 有没有上界。这里的 oracle 只用于离线评估 candidate bank 的 potential，不作为 deploy result。

## Candidate families

固定 read side：

```text
C23 past + pair/all
beta policy = H9 setting
SWA = SWKS3 fixed
RESET_EVERY = 5
```

关键 chunk：

```text
body chunks: 5,6,7,8,9
exit chunks: 10,11,12
handoff chunks: 15,16
post chunks: 20,30,35 only diagnostic
```

每个关键 chunk 生成候选 commit：

```text
B0: H9 default commit
B1: C9 weak-freeze C56 style commit
B2: WINGAM body/exit tri-replay
B3: C16ROLE role commit
B4: PZ orthogonal suppress w0
B5: PZ w1 cap, diagnostic only
B6: native-protected commit
B7: positive-only commit
B8: neutral-preserved commit
B9: conflict-energy top-layer-only commit
B10: post-zeropower delta projected-to-native commit
```

## Oracle evaluation

对每个 start chunk $m$ 和 candidate $c$，rollout future horizon $K$：

```text
K = 2, 3, 5 chunks
```

计算：

$$
Regret(c,m,K)=ATE_{future}(c,m,K)-ATE_{future}(c^*_{oracle},m,K)
$$

$$
Gain(c,m,K)=ATE_{future}(H9,m,K)-ATE_{future}(c,m,K)
$$

## 必须记录

```text
candidate_bank_registry.csv
candidate_by_chunk_metrics.csv
candidate_oracle_table.csv
candidate_segment_effect.csv
candidate_state_hash.jsonl
candidate_ttt_delta_summary.jsonl
```

每个 candidate 记录：

```text
chunk_id
candidate_id
ATE_future_K2/K3/K5
Rot_future
scale_proxy
[200,300) contribution if applicable
[400,600) contribution if applicable
TTT_delta_norm_by_layer_branch
TTT_delta_cos_to_H9
TTT_delta_cos_to_native
post_zeropower_norm_ratio
role_mass_pos_neu_neg
```

## Phase 2 判断标准

### Strong upper-bound pass

```text
oracle bank full reconstructed estimate <= 31.5
or local horizon shows [200,300) improvement >= 8m and [400,600) no regression
```

### Useful upper-bound pass

```text
oracle bank estimate <= 33.0
or at least 3 key chunks have future K=5 gain >= 2m
```

### Fail

```text
best oracle bank estimate > 33.5
and no key chunk future K=5 gain >= 2m
```

如果 Phase 2 fail，不进入 no-GT selector；Codex 转到 Phase 4 action-family expansion。

---

# Phase 3：No-GT Short-Horizon Selector

## 目标

如果 Candidate Bank Oracle 有上界，训练或构造一个不使用 GT 的 selector，选择每个 chunk 的 TTT commit candidate。

## Selector 输入特征

```text
overlap_pointmap_residual_mean/p90
step_length_median_ratio
pose_increment_scale_ratio
TTT_apply_mismatch_mean/p90
TTT_delta_norm_w0/w1/w2
TTT_delta_cos_to_native
post_zeropower_norm_ratio
D_g mass / p90
semantic group mass if cache available
frame attention highD mass
SWA seam residual
```

## Selector 形式

第一版不用训练模型，先做规则和线性打分：

$$
Score(c,m)=
-w_1 R_{overlap}(c,m)
-w_2 R_{scale}(c,m)
-w_3 R_{apply}(c,m)
-w_4 R_{seam}(c,m)
-w_5 R_{instability}(c,m)
$$

选择：

$$
c_m = \arg\max_c Score(c,m)
$$

注意：公式中 `Score` 只使用预测输出和内部状态，不使用 GT。

## Selector audit

在 sandbox 中对比：

```text
oracle choice
selector choice
H9 default choice
C9 fixed policy
```

## 通过标准

```text
Spearman(proxy score, -future ATE) >= 0.45
Top-1 selector within 0.25m of oracle on >=50% key chunks
Top-2 contains oracle on >=70% key chunks
selector does not select candidates that improve one segment but catastrophically regress another
```

如果不通过，Codex 优先尝试：

```text
1. 加入 longer horizon K=5 proxy；
2. 加入 reset-group normalized scale feature；
3. 去掉 misleading feature；
4. 分 body / exit / handoff 训练不同 selector；
5. 如果仍不通过，停止 selector full。
```

---

# Phase 4：新的 TTT Action Families

如果 Phase 2 的 candidate bank 上界不足，说明当前候选 family 不够。此时不继续 selector，而是扩展 action。

## 4.1 Action Family A：post-zeropower residual basis routing

### 动机

pre-zeropower token prior 可能被 Muon / zeropower / norm restoration 折叠掉。直接控制 post-zeropower fast-weight delta 可能更有效。

### 实验

按 layer / branch 构建 residual basis：

```text
basis_native
basis_H9
basis_C9
basis_freeze5_removed
basis_conflict_energy
basis_positive_structure
basis_negative_highD
```

组合：

$$
\Delta W_{new}=\Delta W_{H9}
+ a \cdot P_{corr}(\Delta W_{candidate})
- b \cdot P_{harm}(\Delta W_{candidate})
$$

其中 $P_{corr}$ 和 $P_{harm}$ 先由 offline oracle audit 决定，不直接用于 deploy。

### 小矩阵

```text
PZB_01 w0 only, body 5-9, a=0.10, b=0.05
PZB_02 w0 only, body 5-9, a=0.20, b=0.05
PZB_03 w0+w2, body 5-9, a=0.10, b=0.05
PZB_04 w1 protected, w0 routed, exit 10-12 weak
```

### 停止标准

如果 4 条 full run 都满足：

```text
ATE > H9
or [200,300) improves but [400,600) regresses > 2m
```

停止该 family。

---

## 4.2 Action Family B：TTT replay objective with overlap geometry target

### 动机

v12 的 auxiliary replay 失败，是因为 proxy 过粗。新的 objective 必须直接关联跨 chunk 几何一致性，而不是普通 feature center。

### 定义

构造 overlap pseudo target：

```text
static overlap tokens = low D_g + high confidence + low apply mismatch
pseudo_v = consistency-aligned value from previous/current overlap
```

新的 replay loss：

$$
L_{TTT}=L_{kv}+eta_{geo} L_{overlap}+\beta_{scale} L_{scale-proxy}
$$

其中：

$$
L_{overlap}=\|f_W(k_{overlap})-v_{pseudo}\|^2
$$

$L_{scale-proxy}$ 使用 no-GT 的 step-length / overlap pointmap scale consistency，不用 GT。

### 小矩阵

```text
AUXG_01 beta_geo=0.05, beta_scale=0.00
AUXG_02 beta_geo=0.10, beta_scale=0.00
AUXG_03 beta_geo=0.05, beta_scale=0.05
AUXG_04 beta_geo=0.10, beta_scale=0.05
```

### 通过标准

```text
ATE <= 33.5
or [200,300) improves >= 3m and [400,600) no regression
```

若只是改善 Rot / FinalErr，停止。

---

## 4.3 Action Family C：dual-bank TTT memory with protected continuity

### 动机

之前 dual-lifetime 失败，是因为 short delta 混有 useful continuity。新的版本必须显式保护 continuity：

```text
W_long: positive continuity + stable geometry
W_short: high-risk correction / transient adaptation
```

### 更新

$$
W_{apply}=W_{long}+\alpha_s W_{short}
$$

$$
W_{commit}=W_{long}+\lambda_s \operatorname{Project}_{safe}(W_{short})
$$

其中 `Project_safe` 只允许与 continuity proxy 同向的短期更新进入长期 memory。

### 小矩阵

```text
DLB_01 K=2, alpha_s=0.25, lambda_s=0.00
DLB_02 K=2, alpha_s=0.25, lambda_s=0.25 safe projection
DLB_03 K=3, alpha_s=0.25, lambda_s=0.25 safe projection
DLB_04 K=2, alpha_s=0.50, lambda_s=0.25 safe projection
```

### 停止标准

若 `[200,300)` 与 `[400,600)` 同时比 H9 差，立即停止，不再扫 K/alpha。

---

## 4.4 Action Family D：TTT as selector, not generator

### 动机

如果直接改 delta 不稳定，可以让 TTT 只选择已有安全状态：

```text
candidate states = {H9, C9, WINGAM, native-protected, positive-only}
selector = TTT internal no-GT proxy
commit = selected state
```

这保留 TTT write 的 deployability，但避免对 fast-weight delta 做连续 interpolation。

### 小矩阵

```text
SEL_01 selector from overlap residual only
SEL_02 selector from overlap + step-length scale
SEL_03 selector from overlap + apply mismatch + delta norm
SEL_04 body/exit separate selectors
```

### 通过标准

```text
full online ATE <= 33.0
and [200,300) <= 73.0
and [400,600) <= 43.0
```

---

# Phase 5：Full Online Validation

## 进入条件

只有以下 candidate 进入 full online：

```text
1. sandbox oracle 或 no-GT selector 通过；
2. short rollout K=5 显示 future ATE gain >= 1.5m；
3. no-GT proxy 不只是 Rot / FinalErr proxy；
4. candidate 没有违反 online TTT boundary。
```

## 必跑 full run

```text
H9_REPEAT
C9_WEAK_FREEZE_C56
best candidate family A
best candidate family B
best candidate family C
best selector policy
```

## 记录指标

### Global metrics

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
```

### Segment metrics

```text
ATE_50_mean / worst
ATE_100_mean / worst
ATE_200_mean / worst
[0,100)
[100,200)
[200,300)
[200,400)
[400,600)
[600,800)
per-reset-group ATE
```

### TTT metrics

```text
per-layer branch update norm
post-zeropower delta norm
post-zeropower delta cosine to H9/native
norm_restore_ratio
memory_state_rel_diff
role_mass_pos/neu/neg
candidate_selected_per_chunk
selector_score_per_chunk
apply_mismatch_per_chunk
```

### Boundary flags

```text
counts_as_ttt_write
no_postprocess
uses_gt_runtime_action
output_from_online_hmc
hmc rows
state_changed_count
```

## Full pass gates

```text
Weak pass:
    ATE <= 33.0
    and [200,300) <= 73.0
    and [400,600) <= 43.0

Useful pass:
    ATE <= 32.0
    and [200,300) improves >= 5m vs H9
    and [400,600) not worse than H9

Target-path pass:
    ATE <= 30.0

Final success:
    ATE <= 25.0
```

---

# Phase 6：可视化与 Dashboard

每个 full candidate 都必须生成以下可视化。没有这些图，不允许进入下一轮决策。

## 6.1 Global drift dashboard

```text
trajectory XY: GT / H9 / C9 / candidate
per-frame aligned translation error
cumulative yaw proxy
Sim3 scale over time
step-length ratio over time
per-reset-group ATE bar
```

## 6.2 Segment waterfall

显示每个 candidate 相对 H9 的 segment delta：

```text
Delta [0,100)
Delta [100,200)
Delta [200,300)
Delta [200,400)
Delta [400,600)
Delta FinalErr
```

颜色规则：

```text
green = improvement
red = regression
blue outline = candidate selected chunk
```

## 6.3 TTT layer/branch heatmap

```text
layer x branch update norm
layer x branch delta cosine to H9
layer x branch post-zeropower norm ratio
layer x branch action mask
```

## 6.4 Candidate selector debug

```text
chunk id
candidate selected
oracle best candidate
proxy score
future GT audit ATE
selector regret
segment affected
```

## 6.5 Failure gallery

对 largest regression chunks 输出：

```text
RGB frames
D_g map
TTT conflict energy
semantic group map if available
candidate state selected
H9 vs candidate pose local overlay
overlap pointmap residual map
```

---

# 7. 并行执行策略

## 7.1 资源策略

```text
Full KITTI01 online runs:
    GPU 0/1/2/3，4 并发

Short rollout sandbox:
    可以 4-8 并发，优先 CPU/light GPU，但必须监控 RAM

Visualization / offline audit:
    CPU 并行，不占 GPU
```

## 7.2 Codex 并行任务分配

### Codex Track A：sandbox 和 state snapshot

负责：

```text
state save/load
short rollout runner
H9 sandbox parity
candidate commit bank executor
```

不满足条件时：

```text
如果 H9 sandbox 不能复现 full chunk metrics：
    先修 state snapshot，不跑任何新 policy。

如果 state hash 不稳定：
    增加 deterministic seed / disable inline cache mutation。
```

### Codex Track B：candidate action families

负责：

```text
post-zeropower residual basis routing
overlap geometry auxiliary replay
dual-bank TTT memory
selector-only candidate commit
```

不满足条件时：

```text
如果 family 连续 4 条 full 不过 H9：
    停止该 family，切下一 family。

如果只改善 Rot/FinalErr：
    标记为 regularizer，不继续扫。
```

### Codex Track C：no-GT proxy / selector

负责：

```text
proxy feature extraction
oracle vs selector audit
Spearman / regret / top-k recall
body/exit/handoff separate selector
```

不满足条件时：

```text
如果 Spearman < 0.3：
    不启动 full selector。
    尝试 reset-relative feature、longer horizon、删除 misleading feature。

如果 Top-1 regret > 0.5m：
    改为 Top-2 conservative selector 或 stop。
```

### Codex Track D：dashboard 和失败归因

负责：

```text
global drift dashboard
segment waterfall
layer-branch heatmap
failure gallery
```

不满足条件时：

```text
如果 candidate best 不能解释 segment trade-off：
    不进入 cross-seq，不扩大矩阵。
```

---

# 8. 下一轮最小执行清单

第一批不要超过 12 个 full-equivalent jobs。

## Batch 1：Phase 0 + Sandbox

```text
P0-01 H9_REPEAT
P0-02 C9_REPEAT
S1-01 H9 sandbox parity chunk5 K=3
S1-02 H9 sandbox parity chunk10 K=3
S1-03 H9 sandbox parity chunk16 K=3
```

## Batch 2：Candidate Bank Oracle short rollout

```text
CB-01 chunk5 candidates B0-B10 K=5
CB-02 chunk6 candidates B0-B10 K=5
CB-03 chunk10 candidates B0-B10 K=5
CB-04 chunk16 candidates B0-B10 K=5
```

## Batch 3：Action family smoke

```text
PZB_01 post-zeropower basis routing, short rollout only
AUXG_01 overlap geometry replay, short rollout only
DLB_01 dual-bank, short rollout only
SEL_01 selector-only, short rollout only
```

## Batch 4：Full online only if gates pass

最多启动 4 条 full：

```text
F-01 best candidate bank selector
F-02 best post-zeropower basis
F-03 best overlap replay objective
F-04 best dual-bank memory
```

---

# 9. 失败时的自动尝试方向

为了避免进度卡死，Codex 在失败时按下面规则自动切换。

## Case A：candidate bank oracle 没有上界

条件：

```text
best oracle estimate > 33.5
and no future K=5 gain >= 2m
```

尝试方向：

```text
1. 扩展 candidate family 到 post-zeropower residual basis；
2. 加入 overlap geometry auxiliary replay；
3. 加入 dual-bank TTT；
4. 如果三者都失败，停止 TTT write-only as target-25 mainline。
```

## Case B：oracle 有上界，但 no-GT selector 失败

条件：

```text
oracle best <= 33.0
but selector Spearman < 0.45 or Top-1 regret high
```

尝试方向：

```text
1. 增加 K=5 horizon feature；
2. 分 body / exit / handoff 训练不同 selector；
3. 加入 reset-relative scale proxy；
4. 加入 overlap pointmap seam feature；
5. 如果仍失败，保留 oracle diagnostic，不跑 full selector。
```

## Case C：short rollout 成功，full run 失败

条件：

```text
short K=3/K=5 gain >= 2m
but full ATE 不超过 H9/C9
```

尝试方向：

```text
1. 检查 horizon 是否太短；
2. 增加 downstream stability penalty；
3. 对 candidate action 加 [400,600) regression guard；
4. 降低 action magnitude；
5. 如果仍失败，说明 local proxy 不代表 whole-scene drift。
```

## Case D：只改善 `[200,300)`，但后段崩

条件：

```text
[200,300) improves >= 5m
but [400,600) regresses >= 3m
```

尝试方向：

```text
1. 增加 continuity preservation term；
2. 保护 w1 / value branch；
3. 对 exit chunks 加 positive/neutral handoff；
4. 禁止 hard freeze / hard cap；
5. 若仍失败，标记为 body-window diagnostic。
```

## Case E：只改善后段，body 崩

条件：

```text
[400,600) improves >= 2m
but [200,300) regresses >= 1m
```

尝试方向：

```text
1. 降低 body chunks weak-freeze intensity；
2. 增加 body window body-protection；
3. 使用 candidate selector 只在 exit/handoff chunk 生效；
4. 将该策略标记为 downstream regularizer，不作为主 TTT write policy。
```

---

# 10. 最终决策规则

v14 结束时必须做出明确判断。

## 情况 1：TTT action space 有效

满足：

```text
online ATE <= 32.0
or [200,300) improves >= 5m and [400,600) no regression
```

下一步：

```text
扩展 candidate selector
跑 KITTI00/02/05 sanity
继续向 Target-25 推进
```

## 情况 2：TTT action space 只有 regularizer 价值

满足：

```text
online best 在 33.5-34.0
且无法同时改善 body/downstream
```

下一步：

```text
TTT write 保留为 regularizer
主线转为 online trajectory-state / scale-state module
TTT 只提供 cue 或 regularization
```

## 情况 3：TTT action space 无上界

满足：

```text
candidate bank oracle fail
finite-diff fail
post-zeropower fail
auxiliary replay fail
```

下一步：

```text
停止 TTT write-only target-25 主线
保留 TTT-native cue dashboard
从 read-side / pose-scale / online state module 重新建主线
```

---

## 11. 本计划的核心结论

v14 的核心不是继续问：

```text
gamma 是否应该是 0.0048 还是 0.0052？
chunk16 是否应该是 0.00025 还是 0.00030？
```

而是问：

> **TTT fast weights 里是否存在一个可部署、可选择、可验证的 candidate commit action，能控制 reset-window scale / drift，同时不破坏 downstream continuity？**

如果答案是 yes，v14 应该能给出至少 `1m` 级别的在线改进。  
如果答案是 no，就必须承认当前 TTT write-only controller 只能作为 local/regularizing mechanism，Target-25 的主解法需要更显式的 online trajectory-state / scale-state action。

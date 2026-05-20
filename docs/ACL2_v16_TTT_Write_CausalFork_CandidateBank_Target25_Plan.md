# ACL2 v16：TTT Write Causal Fork、Candidate Commit Bank 与 Target-25 实验计划

日期：2026-05-19  
对象：LoGeR / HMC Pipeline v2 / KITTI01 / TTT 写入策略  
当前线上目标：`KITTI01 ATE <= 25.0m`  
当前可信 online TTT/HMC baseline：

```text
H9_REPEAT / H9_P0_A / H9_P0_B / H9_P0_C
ATE / Rot = 34.1258 / 6.5414
[200,300) = 74.410
[400,600) = 44.354
counts_as_ttt_write = true
```

当前 deployable online repeat 里 ATE 最低但不健康的候选：

```text
C9_WEAK_FREEZE_C56
ATE / Rot = 33.7629 / 6.5259
[200,300) = 76.102
[400,600) = 41.896
counts_as_ttt_write = true
```

当前 diagnostic-only no-GT trajectory proxy：

```text
NOGTPOSE_27
ATE = 22.4012 或 22.3669 级别
counts_as_ttt_write = false
output_from_online_hmc = false
no_postprocess_flag = false
```

本计划的第一原则：**不把 offline trajectory rewrite、GT oracle、sandbox smoke、trajectory 后处理、或 failed parity run 写成 TTT write success。** 只有 online HMC full run、TTT commit 真实影响未来 state、且没有 GT runtime action / postprocess 的候选，才计入 TTT 写入策略。

---

## 1. 当前实验结果的独立判断

v15 修复了 v14 的第一层边界问题。v14 中 `H9_REPEAT_R3` 漂到 `34.2513`，后来确认根因是 locked H9 的 `mp_alpha` 应为 `0.125`，而 v14 H9 R3 误用了 `0.1`。v15 的 Phase 0 已经把这个问题修好：`H9_P0_A / H9_P0_B / H9_P0_C` 都复现到 `34.1258 / 6.5414`，`C9_P0_A / C9_P0_B` 也复现到 `33.7629 / 6.5259`。同时 input-only state save 不改变 full-run 输出，说明现在基础 reproduction gate 已经恢复。

但是 v15 没有进入真正的 TTT candidate bank。Phase 1 native sandbox 失败：HMC fast-weight hashes 可以复现，但 trajectory suffix 不能复现。最典型的是 `H9_chunk5_h1_R2` 的 raw pose max abs diff 达到 `9.3610`，远大于 `1e-5` gate；horizon=3 也多处差 `0.17m-0.38m`，只有个别 horizon=5 ATE suffix 差异看似接近，但 raw pose / gauge 明显不对。这说明目前保存的 `HybridMemoryState` 足以重建 TTT/SWA fast-weight hash，却不足以重建完整 full-run trajectory suffix。缺失状态很可能在 trajectory merge、window alignment、pose gauge、global timestamp/window scheduling、或 Pi3 merge transform state 里。

因此，这一轮不能被解读为“TTT candidate bank 失败”，也不能被解读为“新 TTT action family 没有上界”。更准确的结论是：**v15 把 Phase 0 修好了，但 Phase 1 sandbox parity 没过；所以 Phase 2 candidate bank、Phase 3 no-GT selector、Phase 4 new action families 都没有合法启动。**

这也解释了为什么进度感觉很慢：过去几轮已经证明普通 scalar action space 平台化，而真正能加速搜索的 sandbox / candidate bank 又因为 trajectory-level parity 没通过而不能用。现在的核心瓶颈不是多跑几条 full run，而是缺少一个可信的 short-horizon causal fork。没有这个工具，每个新 TTT 写入策略都要 full KITTI01 跑一次，且很容易在 `33.7m-34.2m` 平台内震荡。

---

## 2. 问题本质

### 2.1 当前不是缺少一个更好的 gamma，而是缺少正确的 TTT action interface

过去已经试过大量低维写入动作：

```text
token prior
gamma
neutral lambda
branch / layer mask
commit EMA
native mix
weak freeze
post-zeropower cap / mix
short transient overlay
projection risk scalar
read beta cooling
```

它们能把系统从 `36m` 级推进到 `34m` 级，说明 TTT 写入确实有信号；但它们没有把 `[200,300)` 从 `74m` 级压下来，也没有接近 `ATE <= 25m`。这说明这类动作主要在做局部 regularization、rotation / endpoint 平衡、后段误差重分配，而不是直接控制 window-level scale / trajectory-state drift。

LoGeR 的 TTT fast weights 按设计承担 compressed global memory，用于 anchor global coordinate frame 并防止 scale drift；SWA 则承担 uncompressed local context 和 adjacent alignment。因此，目标错误本质上是 TTT 应该能影响的。但当前控制方式只是通过 token write prior 间接影响 fast weights，再间接影响未来 pose / pointmap，这条链太长、太弱。

### 2.2 Target-25 的真正杠杆是 reset-window scale / trajectory-state

v10 的 pose-scale oracle 已经说明：如果允许对 reset-window 做 Sim3 或 scale-only correction，ATE 可以降到远低于 25m 的范围；`NOGTPOSE_27` 作为 offline no-GT trajectory-state proxy 也能到 `22.4m` 级别。虽然它不能算 TTT 写入成功，但它揭示了关键事实：**当前 34m 平台的主误差包含强烈的 reset-window scale / trajectory-state 成分。**

因此下一阶段必须回答：TTT fast weights 里是否存在可部署的 action，能在线改变这个 scale / trajectory-state，而不是只在输出轨迹后处理时修正它。

### 2.3 v15 的新瓶颈是 trajectory-level fork state，不是 HMC fast-weight state

v15 Phase 1 已经证明，保存 / 加载 HMC fast weights 后，HMC hash 可以一致；但 short rollout 的 raw pose 与 full suffix 不一致。这说明 `W0/W1/W2/history` 不是完整 causal state。要做 candidate bank，必须捕捉或复现以下状态：

```text
Pi3 / LoGeR internal window merge state
Sim3 / SE3 alignment transform reused across reset block
external chunk global frame index / timestamp state
raw prediction merge buffer
pose gauge / trajectory origin state
any reset-block local transform cache
SWA history already present in HMC state, but merge gauge may be outside it
```

如果不解决这个问题，candidate bank oracle 评出来的 suffix ATE 不是 full-run causal effect，后续 no-GT selector 也会基于错误的 target 学习。

---

## 3. 本阶段总体目标

本阶段目标不是继续把 `34.1258` 微调到 `34.0`，而是建立一条能真正加速 TTT 写入探索的因果实验链：

```text
可信 full-run boundary
    -> trajectory-level causal fork parity
    -> candidate commit bank oracle upper bound
    -> no-GT short-horizon selector
    -> full online validation
    -> cross-sequence sanity
```

只有这条链建立后，才能高效回答：

```text
TTT fast weights 是否还有 target-25 的可控上界？
如果有，哪类 candidate commit action 能接近它？
如果没有，TTT write 是否只能作为 regularizer，target-25 是否必须交给 explicit online trajectory-state module？
```

本阶段不允许继续无目标地全序列扫低维标量。每个实验必须服务于以下三类问题之一：

1. **Causal-state correctness**：短 rollout 是否等价于 full-run suffix；
2. **TTT-action upper bound**：候选 fast-weight commit 是否有足够上界；
3. **Deployable selection**：不使用 GT 的 proxy 能否选择正确 commit。

---

## 4. 核心假设

### H1：当前 sandbox 失败是因为缺少 trajectory merge / pose gauge state，而不是 HMC fast weights 错误

v15 已经显示 HMC hashes 可以复现，但 trajectory suffix 不复现。本假设认为，缺失的是 HMC 外部的 trajectory-level state。

如果 H1 成立，补齐 merge/gauge state 或使用 in-process fork 后，native sandbox 应该能通过：

```text
horizon=1 raw pose max diff <= 1e-5
horizon=3 ATE_suffix_delta <= 0.01m
horizon=5 ATE_suffix_delta <= 0.03m
H9 / C9 / WINGAM all pass
```

如果补齐这些状态后仍失败，则说明模型 forward 内还有 hidden nondeterminism 或未保存的 tensor cache，需要继续 instrumentation，而不是启动 candidate bank。

### H2：TTT candidate commit bank 有可能找到比单一路径写入更好的 fast-weight state

当前每个 chunk 只提交一个 `W_{m+1}`。但历史 freeze、C9、H9、WINGAM、PZ cap 等实验说明，不同 commit 会在 `[200,300)` 与 `[400,600)` 之间产生不同 trade-off。Candidate bank 假设认为：同一 chunk 可以生成多个合法 TTT commit candidates，其中某些 candidate 在短 horizon 上能更好地控制未来 scale / drift。

候选集合不是继续扫一个标量，而是覆盖不同 action family：

```text
native/probe commit
H9 current commit
C9 weak-freeze style commit
WINGAM tri-replay commit
post-zeropower residual cap/mix commit
commit EMA / native mix commit
overlap pseudo replay commit
scale-aware auxiliary replay commit
dual-bank short/long memory commit
```

H2 的上界成立标准不是最终 `25m`，而是 candidate oracle 至少能显著超过 H9，并显示 `[200,300)` 或 reset-window scale 有因果改善。

### H3：如果 candidate oracle 有上界，则 no-GT short-horizon proxy 可以学习或选择其中一部分

`NOGTPOSE_27` 说明不使用 GT 的 trajectory statistics 能强烈指向 target-25，但它现在是 offline trajectory rewrite。H3 不是要做后处理，而是用 no-GT proxy 在 commit bank 中选择 `W_{m+1}`。

可用 proxy 包括：

```text
future 1-3 chunks predicted step-length median stability
overlap pointmap scale consistency
chunk-to-chunk raw pose increment continuity
TTT apply mismatch reduction
SWA overlap feature / geometry consistency
read cue D_g mass stability
TTT update_conflict_energy trend
memory_state_rel_diff boundedness
```

如果 proxy 与 oracle choice 相关性不足，则 selector 不能启动 full online；Codex 应自动尝试新的 proxy，而不是跑更多 full validation。

### H4：如果 candidate bank oracle 无上界，则当前 TTT write-only action space 不足

如果即使用 oracle 在候选 bank 中选择，仍无法超过 H9 或无法让 `[200,300)` 明显下降，那么说明当前候选动作不足。此时继续调 `gamma / neutral / read beta / commit EMA` 没意义，应切换到更强动作 family：

```text
post-zeropower residual basis routing
TTT auxiliary objective using overlap geometry consistency
dual-bank TTT memory with explicit W_long / W_short
scale-state-conditioned TTT write
online trajectory-state module as separate path
```

### H5：如果 TTT 仍无法接入 scale-state，Target-25 主线不能继续押在 TTT write-only 上

这不是放弃 TTT，而是角色重定义。TTT 可以继续作为 read / memory regularizer；但若 TTT fast-weight commit 无法承载 window-scale correction，Target-25 必须引入 explicit online trajectory-state / scale-state action，并清楚标注它不是 TTT write success。

---

## 5. Phase 0：重新锁定边界与 artifact hygiene

### 5.1 实验目的

确保后续所有 sandbox / candidate bank 都基于可信 boundary，不再发生 v14 H9 drift、全量 snapshot 爆盘、或 candidate 与 diagnostic 混淆。

### 5.2 固定边界 run

每次进入新 action-family 前必须至少保留这些边界：

```text
H9 locked online baseline
C9 weak-freeze online repeat
WINGAM online repeat
NOGTPOSE_27 diagnostic-only repeat, optional
```

### 5.3 必须记录

每个 boundary run 写入：

```text
metrics_global.json
trajectory_diagnostics.json
per_100f_segment.csv
per_200f_segment.csv
global_drift_dashboard.csv
hmc_state_hash.jsonl
hmc_config.yaml
run_config.yaml
software_commit.txt
artifact_manifest.json
```

`artifact_manifest.json` 必须包含：

```text
run_id
counts_as_ttt_write
uses_gt_runtime_action
uses_offline_trajectory_rewrite
output_from_online_hmc
hmc_rows
commit_mode
mp_alpha
read_beta_frame
write_score_source
reset_every
state_snapshot_kinds
state_snapshot_chunks
snapshot_size_gb
```

### 5.4 成立标准

```text
H9 ATE drift <= 0.03m
C9 ATE drift <= 0.03m
WINGAM ATE drift <= 0.03m
hmc rows = 38
NOGTPOSE counts_as_ttt_write = false
input-only state save changes ATE <= 0.01m
snapshot footprint <= 20GB per boundary batch
```

### 5.5 不满足时 Codex 的尝试方向

如果 H9 漂移：

```text
1. diff hmc_config.yaml against locked H9 archive;
2. compare mp_alpha, WRITE_ALPHA, read beta, reset_every, SWA flags;
3. rerun with archived config exactly;
4. dump config_diff_report.json;
5. do not proceed to sandbox.
```

如果 state save 改变输出：

```text
1. switch to input-only snapshots;
2. ensure deepcopy + CPU move after effective input state;
3. disable before/after all-chunk snapshot;
4. re-run H9_P0_A/B parity.
```

如果 artifact footprint 超过 20GB：

```text
1. save only chunks 5,10,16 initially;
2. save only input state unless debugging requires before/after;
3. compress metadata JSONL separately;
4. never save all chunks × before/input/after by default.
```

---

## 6. Phase 1：Trajectory-Level Causal Fork Parity

### 6.1 实验目的

建立一个短 horizon causal fork，使得从 chunk `m` 的 saved state 启动，跑 `h=1/3/5` 个 chunks 后，输出与 full-run suffix 完全或近似一致。没有这个 parity，candidate bank 不能启动。

### 6.2 两条并行实现路线

#### 路线 A：保存并恢复完整 trajectory merge / pose gauge state

Codex 应检查 LoGeR / Pi3 merge 相关状态，包括但不限于：

```text
window_raw_predictions already merged or buffered
_window_start / _window_end
global frame timestamp mapping
reset-block transform cache
Sim3 / SE3 transform reused within reset block
previous raw pose gauge
output trajectory origin / alignment cache
external chunk merge accumulators
```

需要新增：

```text
--save_merge_state_chunks
--load_merge_state_at_chunk
merge_state_snapshot.pt
merge_state_hash.jsonl
raw_prediction_buffer_summary.json
```

#### 路线 B：in-process fork，不做外部 sliced run

如果恢复完整 merge state 太复杂，应实现更可靠的 in-process fork：

```text
在 full run 进行到目标 chunk m 时：
    deepcopy current model/HMC/merge state
    对 candidate A 做短 horizon rollout
    恢复 deepcopy state
    对 candidate B 做短 horizon rollout
    恢复原状态
    继续 full run主路径
```

这避免 sliced run 重新构造全局 gauge 的问题。第一版可以只支持 H9 chunk5 horizon=1/3，优先验证 raw pose parity。

### 6.3 Native sandbox parity run

固定候选：

```text
H9 chunk5 h1/h3/h5
H9 chunk10 h3
H9 chunk16 h3
C9 chunk5 h3
C9 chunk10 h3
C9 chunk16 h3
WINGAM chunk5 h3
```

每个 run 输出：

```text
short_rollout_metrics.csv
short_rollout_raw_pose_diff.csv
short_rollout_gt_audit.csv
short_rollout_proxy.jsonl
hmc_hash_parity.jsonl
merge_state_hash_parity.jsonl
raw_pose_txt_sandbox.txt
raw_pose_txt_full_suffix.txt
```

### 6.4 必须记录的指标

```text
ATE_suffix_delta
Rot_suffix_delta
raw_pose_max_abs_diff_vs_full
raw_pose_max_trans_diff_vs_full
raw_pose_mean_trans_diff_vs_full
hmc_input_hash_equal
hmc_output_hash_equal
merge_state_hash_equal
num_frames_matched
timestamp_mapping_equal
window_start_end_equal
```

### 6.5 判断标准

Phase 1 通过条件：

```text
horizon=1 raw_pose_max_abs_diff_vs_full <= 1e-5
horizon=1 raw_pose_max_trans_diff_vs_full <= 1e-5
horizon=3 ATE_suffix_delta <= 0.01m for all required H9/C9 runs
horizon=5 ATE_suffix_delta <= 0.03m for H9 chunk5
hmc hash parity pass
merge/gauge state parity pass or in-process fork path pass
```

### 6.6 不满足时 Codex 的尝试方向

如果 HMC hash pass 但 raw pose diff 巨大：

```text
1. inspect merge state / pose gauge / timestamp mapping;
2. compare raw per-window predictions before merge;
3. run in-process fork prototype to bypass external slicing;
4. if in-process fork passes, deprecate external sliced sandbox for candidate bank.
```

如果 h1 raw pose pass 但 h3 ATE fails：

```text
1. check state commit after chunk0 inside sandbox;
2. compare HMC output hash at local chunk1/2 with full chunk m+1/m+2;
3. check reset_every scheduling and global chunk index;
4. check SWA history handoff indexing.
```

如果 timestamp mapping mismatch：

```text
1. enforce global frame indices in trajectory txt;
2. output both local and global timestamps;
3. never rely on positional row matching without timestamp audit.
```

---

## 7. Phase 2：Candidate Commit Bank Oracle

### 7.1 实验目的

验证 TTT commit action 本身是否还有足够上界。此阶段允许 GT 只用于离线 oracle 选择，不计入 deployable success；目标是判断是否值得做 no-GT selector。

### 7.2 候选 bank 定义

在每个关键 chunk 生成多个 candidate `W_{m+1}`，并对未来 `h=3/5` chunks 做 trusted short rollout。

关键 chunks：

```text
5, 6, 9, 10, 12, 16
```

第一版候选：

```text
K0 native/probe commit
K1 H9 current commit
K2 C9 weak-freeze style commit
K3 WINGAM tri-replay commit
K4 chunk-local TTGR mild
K5 post-zeropower cap w0
K6 post-zeropower cap w1
K7 commit EMA 0.90
K8 native mix protected
K9 overlap auxiliary replay
K10 dual-bank short correction, if implemented
```

每个 candidate 必须有 manifest：

```text
candidate_id
parent_run_id
chunk_id
action_family
branch_mask
layer_mask
gamma / neutral / alpha / cap
post_zeropower_flag
counts_as_deployable_if_selected
uses_gt_runtime_action = false for generation
```

### 7.3 Oracle selection 指标

GT oracle 只用于评估候选 bank 上界。对每个 candidate 计算：

```text
future_h1_ATE
future_h3_ATE
future_h5_ATE
future_h3_[200,300)_segment_if_applicable
future_h3_[400,600)_segment_if_applicable
future_h3_scale_proxy_gt
future_h3_yaw_error
future_h3_final_error
```

同时计算健康约束：

```text
future_continuity_penalty
[400,600) regression
raw pose discontinuity
HMC memory rel diff
candidate delta norm
```

Oracle score：

$$
S_{oracle} = ATE_{h3} + 0.3 \cdot Penalty_{400:600} + 0.2 \cdot Jump_{boundary}
$$

其中 `Penalty_400:600` 只在候选使 downstream window 明显恶化时计入。

### 7.4 判断标准

H2 strong pass：

```text
oracle-selected candidate full-equivalent / trusted rollout indicates ATE <= 32.5m potential
or [200,300) decreases >= 8m while [400,600) regression <= 2m
```

H2 stage pass：

```text
oracle-selected candidate improves H9 by >= 1.0m estimated full ATE
or [200,300) decreases >= 5m while [400,600) does not worsen
```

H2 fail：

```text
best oracle candidate improves H9 by < 0.5m
and [200,300) improves < 3m
```

### 7.5 不满足时 Codex 的尝试方向

如果 no candidate beats H9 in trusted rollout：

```text
1. expand action family, not scalar grid;
2. add post-zeropower residual basis routing;
3. add overlap-geometry auxiliary replay objective;
4. add dual-bank W_long/W_short;
5. add candidate that changes TTT apply gating but keeps commit safe;
6. rerun only sandbox oracle, not full KITTI01.
```

如果 candidate improves `[200,300)` but hurts `[400,600)`：

```text
1. add continuity-preserving positive component;
2. split candidate into body correction + exit continuity;
3. make candidate lifecycle short for body correction and long for structure anchor;
4. evaluate dual-lifetime bank.
```

If oracle improves only Rot / FinalErr：

```text
1. mark action family as regularizer;
2. stop using it for target-25 mainline;
3. keep only if combined with a drift-correction candidate.
```

---

## 8. Phase 3：No-GT Short-Horizon Selector

### 8.1 实验目的

如果 oracle bank 有上界，则构造不使用 GT 的 selector，选择每个 chunk 的 TTT candidate commit。

### 8.2 候选 no-GT proxy

必须至少包含四类 proxy。

#### A. Trajectory-state proxy

```text
predicted step-length median stability
reset-window scale ratio
chunk-to-chunk translation increment smoothness
yaw increment smoothness
body / exit handoff scale discontinuity
```

#### B. Geometry-overlap proxy

```text
overlap pointmap residual
head/tail overlap relative scale
static structure overlap consistency
road/building/fence/wall consistency if semantic cache available
```

#### C. TTT memory proxy

```text
TTT apply mismatch
update_conflict_energy trend
candidate update norm
candidate delta cosine to native
memory_state_rel_diff
post-zeropower norm restoration ratio
```

#### D. Read/cue proxy

```text
D_g high-mass shift
D_g p90 stability
attention mass to high-D tokens
confidence / uncertainty trend
semantic lowstuff high-D mass, optional
```

### 8.3 Selector score

第一版使用手工 weighted score，不训练模型：

$$
S_{nogt}(k) = w_s S_{scale}(k) + w_o S_{overlap}(k) + w_t S_{ttt}(k) + w_j S_{jump}(k)
$$

每个 term 必须 z-score 到同一 run/chunk 的 candidate set 内：

$$
\tilde S_x(k) = \frac{S_x(k)-\mu_x}{\sigma_x+\epsilon}
$$

初始权重：

```text
w_s = 0.40
w_o = 0.30
w_t = 0.20
w_j = 0.10
```

### 8.4 离线 selector audit

在不跑 full online 前，先用 oracle table 做离线评估：

```text
candidate_count
chunk_selected_H9_or_better_ratio
selected_oracle_with_worse_global_ATE_count
Spearman(proxy_score, oracle_ATE)
Top1 selector regret
Top3 oracle recall
```

Selector pass：

```text
Spearman(proxy_score, oracle_ATE) <= -0.50  # score lower is better
Top3 oracle recall >= 0.70
selected H9-or-better ratio >= 0.70
no selected candidate with severe [400,600) regression
```

如果 pass，再启动 online full selector run。

### 8.5 不满足时 Codex 的尝试方向

如果 proxy correlation poor：

```text
1. remove noisy term one-by-one;
2. test scale-only proxy;
3. test overlap-only proxy;
4. test TTT-only proxy;
5. fit tiny ridge/logistic selector from existing oracle table, but report as learned selector diagnostic;
6. add more oracle candidates only if candidate diversity insufficient.
```

如果 selector selects candidates that improve short horizon but hurt full run：

```text
1. increase downstream continuity penalty;
2. extend horizon from h3 to h5;
3. add exit-window proxy;
4. use conservative fallback to H9 when proxy confidence is low.
```

---

## 9. Phase 4：New TTT Action Families Only If Bank Upper Bound Exists

### 9.1 Post-zeropower residual basis routing

当前 pre-zeropower token prior 可能被 Muon / zeropower / norm restoration 折叠掉。新的动作应直接控制 zeropower 后的 fast-weight delta：

$$
\Delta W = \Delta W_{native} + \sum_b \alpha_b B_b
$$

其中 basis 可以来自：

```text
native delta
H9-H0 delta
C9-H9 delta
freeze5 removed direction
update_conflict negative direction
structure-positive direction
```

第一批 candidate：

```text
PZB_01 native + 0.1*(C9-H9)
PZB_02 native + 0.2*(C9-H9)
PZB_03 native + 0.1*(freeze5_removed projected to continuity-safe subspace)
PZB_04 native - 0.05*(high_conflict_direction)
```

记录：

```text
basis_norm
basis_cos_to_native
basis_cos_to_freeze5_removed
post_delta_norm
norm_restore_ratio
future_h3_ATE
[200,300) / [400,600)
```

### 9.2 Overlap-geometry auxiliary TTT replay objective

不是再用 token prior，而是给 TTT replay 增加目标：写入后在 overlap/static structure 上更一致。

伪目标：

$$
L_{aux} = \| f_W(k_{overlap}) - v_{overlap}^{stable} \|^2
$$

其中 `v_overlap_stable` 可由 static overlap centroid、structure semantic group、或 previous/next overlap agreement 构造。

候选：

```text
AUXG_01 overlap static centroid weak
AUXG_02 road/building structure centroid weak
AUXG_03 high-confidence low-D overlap centroid
AUXG_04 apply-mismatch suppression only on conflict layers
```

### 9.3 Dual-bank TTT memory

把 TTT memory 分成：

```text
W_long: only positive / structure / continuity update
W_short: dynamic / conflict correction, applied for K chunks but not committed long term
```

最小实现：

```text
W_apply = W_long + alpha_short * W_short
W_commit = W_long
W_short decays with K = 1/2/3 chunks
```

候选：

```text
DB_01 K=1 alpha=0.25
DB_02 K=2 alpha=0.25
DB_03 K=2 alpha=0.50
DB_04 K=3 alpha=0.25
```

通过标准：

```text
[200,300) improves >= 5m
[400,600) regression <= 2m
ATE improves H9 by >= 1m
```

### 9.4 不满足时 Codex 的尝试方向

如果 all new families fail oracle gate：

```text
1. stop TTT write-only target-25 mainline;
2. keep best TTT as regularizer;
3. start online trajectory-state / scale-state module plan;
4. do not continue low-dimensional TTT scalar sweep.
```

---

## 10. Phase 5：Full Online Validation

### 10.1 进入条件

只有满足下面之一，才跑 full KITTI01 online：

```text
candidate bank oracle stage pass
no-GT selector offline audit pass
new action family sandbox h3/h5 pass
```

禁止：

```text
sandbox parity fail 时跑 full
oracle upper bound fail 时跑 selector full
proxy correlation fail 时跑 selector full
```

### 10.2 Full run 必须记录

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
[0,100), [100,200), [200,300), [300,400), [400,600)
50f / 100f / 200f mean and worst
chunk-level RMSE
reset-group RMSE
hmc state hash
candidate selection log per chunk
selected candidate family
selector score terms
```

### 10.3 Full success gates

Strong success：

```text
ATE <= 25.0m
counts_as_ttt_write = true
no GT runtime action
no offline trajectory rewrite
[200,300) <= 45m
[400,600) <= 40m
```

Stage success：

```text
ATE <= 30.0m
[200,300) improves >= 10m vs H9
[400,600) does not worsen vs H9 by > 3m
```

Weak success：

```text
ATE <= 33.0m
or ATE improves H9 by >= 1.0m and [200,300) improves >= 5m
```

Fail / stop：

```text
ATE > H9 + 0.05m
or only Rot/FinalErr improves
or [200,300) worsens while ATE improves by error transfer
or [400,600) worsens by > 5m
```

---

## 11. Phase 6：Cross-sequence sanity

只有 KITTI01 通过 weak success 以上才跑：

```text
KITTI00 full
KITTI02 full
KITTI05 full
```

对照：

```text
H9 locked
C9 weak freeze
candidate v16
NOGTPOSE diagnostic, optional but not counted
```

通过标准：

```text
average ATE improves H9 by >= 5%
no sequence regression > 5%
KITTI02 not significantly worse
```

如果 KITTI01 提升但 KITTI02 崩，则标记为 KITTI01-specific window overfit，不晋级。

---

## 12. 必须可视化的内容

### 12.1 Sandbox parity dashboard

输出：

```text
full suffix raw pose XY
sandbox raw pose XY
translation residual over local time
rotation residual over local time
HMC hash equality timeline
merge state hash equality timeline
```

目的：一眼判断差异来自 HMC 还是 merge/gauge。

### 12.2 Candidate bank heatmap

行：chunk id。列：candidate id。颜色：future h3 ATE 或 proxy score。

必须同时输出：

```text
candidate ATE heatmap
candidate [200,300) heatmap
candidate [400,600) heatmap
candidate selector score heatmap
candidate chosen map
```

### 12.3 Global drift dashboard

每个 full candidate 必须画：

```text
trajectory XY full / first half / second half
per-frame aligned translation error
sliding 50/100/200-frame ATE
reset-group ATE
Sim3 scale over time proxy
cumulative yaw drift
axis-wise x/y/z error
```

### 12.4 TTT memory dashboard

```text
per-layer × branch update norm
per-layer × branch candidate delta cosine to native
post-zeropower norm restoration ratio
memory_state_rel_diff over chunks
selected candidate family over chunks
update_conflict_energy over chunks
```

### 12.5 Error transfer dashboard

必须展示：

```text
[200,300) delta vs H9
[400,600) delta vs H9
FinalErr delta vs H9
Yaw delta vs H9
ATE delta vs H9
```

如果 `[200,300)` 降、`[400,600)` 升，要标红为 error transfer。

---

## 13. 并行执行与加速策略

### 13.1 Codex 并行任务拆分

Codex A：repro / config / artifact hygiene

```text
Phase 0 repeat
config diff
snapshot size audit
artifact manifest
```

Codex B：trajectory-level sandbox parity

```text
merge state capture
in-process fork prototype
raw pose parity audit
```

Codex C：candidate bank generator

```text
generate candidates after sandbox parity
candidate manifest
fast-weight delta save
```

Codex D：oracle / selector audit

```text
oracle table
proxy score terms
selector offline audit
correlation reports
```

Codex E：visualization dashboard

```text
global drift dashboard
sandbox parity plots
candidate heatmaps
error transfer dashboard
```

### 13.2 GPU / CPU policy

```text
Full KITTI01 online validation: max 4 concurrent unless host RAM audit says safe.
Sandbox short rollout: may run 6-8 concurrent if state load footprint is bounded.
No full run launched before Phase gate passes.
No all-chunk before/input/after snapshots.
```

### 13.3 Stop rules to avoid slow progress

```text
If Phase 1 sandbox parity fails after both merge-state and in-process fork attempts:
    stop candidate bank; work only on state correctness.

If candidate bank oracle improves H9 by <0.5m:
    stop no-GT selector; expand action family in sandbox only.

If no-GT selector offline Spearman is weak:
    do not run full selector; improve proxy terms.

If 4 full runs from the same action family fail to beat H9:
    stop that family.

If a candidate improves ATE but worsens [200,300) by >1m:
    mark as error transfer, not target-25 progress.

If a candidate improves only Rot / FinalErr:
    keep as regularizer, not mainline.
```

---

## 14. 本计划的最终决策逻辑

本阶段有三种可能结局。

### 结局 A：TTT candidate bank oracle 有强上界

如果 oracle 可以显著超过 H9，并且 no-GT selector 能较好选择候选，则继续把 TTT write 作为 Target-25 主线。下一步做 selector online full run 和 cross-sequence sanity。

### 结局 B：Oracle 有上界，但 no-GT selector 失败

说明 TTT action 有潜力，但缺少可部署选择信号。下一步重点转向 proxy design，而不是 action design。可考虑小模型 selector，但必须标注为 learned selector diagnostic。

### 结局 C：Oracle 都没有上界

说明当前 TTT fast-weight action family 不能支撑 Target-25。此时不再继续 TTT write-only target-25 主线。TTT 保留为 regularizer，同时启动 explicit online trajectory-state / scale-state module；但必须清楚标注它不是 TTT write success。

---

## 15. 当前最重要的短句结论

**v15 的重点不是 TTT action 失败，而是 causal sandbox 尚未达到 trajectory-level parity。下一步的最优动作不是继续跑 full-run 写入小矩阵，而是先让 short-horizon fork 等价于 full-run suffix。只有这样，candidate commit bank 才能真正加速 TTT 写入策略搜索。**


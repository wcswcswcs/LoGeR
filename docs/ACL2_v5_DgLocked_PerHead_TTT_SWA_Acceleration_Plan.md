# ACL2 v5：D_g_locked 解释、Per-head 挖掘、TTT/SWA Memory Policy 与 Pipeline 加速实验计划

日期：2026-05-05  
对象：LoGeR / HMC Pipeline v2 / ACL2 Attention Cue Library  
主开发集：KITTI Odometry Sequence 01  
跨序列 sanity：KITTI00 / KITTI02 / KITTI05  
当前主线：

```text
D_g_locked = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
beta = 3.75
read intervention = frame-attention early-layer pair bias
commit = probe_ttt_write
write score = stage_d
```

当前对照候选：

```text
D_g_ate_shadow = acl2.gg.qq.low.g2_4.past_only.headmean.robustq
beta = 4.00
read intervention = frame-attention early-layer pair bias
commit = probe_ttt_write
write score = stage_d
```

---

## 0. 当前阶段判断

v4 已经完成了一个关键收敛：read cue 主线不再是笼统的“早中层窗口”，而是可以锁到：

```text
g2_3.past_only 的 global query-query low-similarity cue
```

它的价值不是只在 KITTI01 上刷新一个小数点，而是它同时满足了三个条件：

```text
1. 在 KITTI01 上接近 ATE best，同时 rotation / final error / yaw 更稳；
2. 在 KITTI00 / KITTI02 / KITTI05 full cross-sequence 平均上优于 g3.full 和 g2_4.past_only；
3. 与 probe_ttt_write 仍然存在稳定互补，不只是 read-only trick。
```

但是，当前阶段仍然不能说 cue 已经解释清楚，也不能说 memory policy 已经最优。现有最强结果仍然主要是：

```text
single internal-attention read cue
+
frame-attention pair-bias read intervention
+
probe_ttt_write safe TTT write protocol
```

所以 v5 的核心不是继续无序扩大 cue 名字，而是围绕一个已经锁定的强 cue 做三件事：

```text
1. 解释 D_g_locked 为什么有效；
2. 挖掘 per-head 维度，判断 headmean 是否稀释了更强信号；
3. 以 D_g_locked 为固定 read cue，系统探索 TTT write policy 和 SWA local-memory policy。
```

同时，现在 pipeline 必须加速。v4 已经暴露了资源规律：KITTI01 full run 4 并发是安全甜点位；8 并发会把 host RAM 压到危险区；KITTI00 / KITTI02 这类长序列 full run 默认应使用 2 并发。因此 v5 不只是实验计划，也必须包含工程调度、缓存、passive audit 和结果复用机制。

---

## 1. 实验整体目标

v5 的目标分成四个层级。

### 1.1 目标 A：解释 `D_g_locked` 的作用机制

当前 `D_g_locked` 是：

```text
acl2.gg.qq.low.g2_3.past_only.headmean.robustq
```

但它仍然是一个复合对象：

```text
global stack layers g2_3
query-query low similarity
past_only support
headmean aggregation
robustq normalization
frame-attention pair-bias intervention
probe_ttt_write commit
```

v4 已经证明它强，但还没有解释清楚：

```text
1. 它 suppress 的 token 到底分布在哪里？
2. 它和 old_dyn / explicit_dyn 的 disagreement 为什么不能简单 fusion？
3. 它为什么比 g2_4.past_only 更跨序列稳定？
4. 它是主要修 global drift、yaw drift、scale drift，还是 chunk boundary？
5. 它的有效性来自少数 heads，还是 headmean 的整体 consensus？
```

v5 第一部分必须把这些问题变成可量化、可视化、可审计的解释实验。

### 1.2 目标 B：补 per-head attention cue mining

当前主线仍然是 `headmean`。这有两个风险：

```text
1. 如果动态 / harmful-support 信号只藏在少数 heads，headmean 会稀释它；
2. 如果某些 heads 很噪，headmean 会把 false positive 平均进主 cue。
```

因此 v5 要专门验证：

```text
per-head / top-k-head / head-agreement / head-var / head-entropy 是否能超过 headmean。
```

per-head mining 不能回到全空间乱扫，而应围绕已经锁定的结构：

```text
layers = g2_3, optional shadow g2_4
support = past_only
statistic = qq low-similarity
normalization = robustq
read intervention = frame-attention pair bias
```

### 1.3 目标 C：探索 TTT write policy

目前 `probe_ttt_write` 使用 `stage_d` write score，并且在 C23/C3 上提供稳定同 beta ATE gain。v4 的结论支持继续使用 safe write protocol，但不代表 write policy 已最优。

v5 要回答：

```text
1. branch0 仍然是唯一有效写入分支吗？
2. 不同 TTT layer 的 write 是否应不同强度？
3. D_g_locked 能否转成 write eligibility，而不只是 read suppress cue？
4. sparse / exact-preserve / branch-specific write 是否能进一步改善 cross-sequence stability？
5. safe write 的收益是否来自更干净的 memory update，还是只是固定 branch0 的强度效果？
```

### 1.4 目标 D：继续探索 SWA local-memory policy

SWA 不是 TTT fast weights，它没有“parametric write”的意义。但在 LoGeR 中，SWA 是相邻 chunk 的 lossless local memory highway。当前 chunk 的 token 会成为下一 chunk SWA 的 previous-source context。因此可以把 SWA policy 定义为：

```text
哪些 token 被写入 / 保留为 SWA local memory source？
哪些 previous-source token 在下一 chunk 被弱化读取？
哪些 local-memory token 需要附带 reliability metadata？
```

v4 的 SWA-only previous-source gate 没有通过 performance gate，但这不等于 SWA line 结束。它只说明：

```text
单独用 D_g_locked 去 gate SWA previous-source，不能替代 frame-attention pair bias。
```

v5 对 SWA 的目标应该改成：

```text
不再期待 SWA-only 大幅降全局 ATE；
而是验证 SWA policy 是否能改善 chunk boundary、overlap seam、local alignment、FinalErr 或 rotation。
```

### 1.5 目标 E：加速 pipeline

v5 实验会引入 per-head cache、TTT branch/layer sweep、SWA local metrics 和 cross-sequence sanity。如果不加速，会被 full run 数量拖垮。

因此必须同时建立：

```text
1. run registry 和 config hash；
2. passive cue cache；
3. shared probe / cue extraction reuse；
4. resource-aware scheduler；
5. offline visualization pipeline；
6. standard metric aggregator；
7. strict promotion gate，减少 full run 浪费。
```

---

## 2. 固定主线与固定协议

### 2.1 主线 cue

v5 所有主要实验使用：

```text
D_g_locked = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
beta_locked = 3.75
```

`D_g_ate_shadow` 只作为对照：

```text
D_g_ate_shadow = acl2.gg.qq.low.g2_4.past_only.headmean.robustq
beta_shadow = 4.00
```

### 2.2 固定 read intervention

除非阶段明确写了 intervention ablation，否则固定：

```text
read path = frame attention early layers
bias mode = pair
normalization = robustq
support = past_only
commit = probe_ttt_write for hybrid
```

### 2.3 固定 HMC commit 协议

所有主实验只允许两种主评价：

```text
read-only:
    output = controlled forward
    commit = probe_native

hybrid-safe:
    output = controlled forward
    commit = probe_ttt_write
```

`controlled commit` 只做 diagnostic，不作为主结论。

### 2.4 固定 baseline

每批实验必须报告这些 baseline：

| Baseline | 用途 | 当前参考 |
|---|---|---:|
| `native LoGeR` | no-control anchor | `41.7502 / 8.9928` |
| `old_dyn_addclip` | 旧 Phase F best | `39.3103 / 9.7097` |
| `gg.qq.middle.low + probe_ttt_write` | ACL v1 anchor | `38.9714 / 9.2084` |
| `C3 = g3.full + probe_ttt_write` | ACL2 previous anchor | `38.4298 / 8.9846` |
| `C23 full + probe_ttt_write` | v4 full-support Pareto | `38.3847 / 8.7583` |
| `C24 full + probe_ttt_write` | v4 full-support ATE | `38.3805 / 8.8707` |
| `C23 past + probe_ttt_write` | v4 locked mainline | `38.3706 / 8.6694` |
| `C24 past + probe_ttt_write` | v4 KITTI01 ATE shadow | `38.3566 / 8.7660` |

---

## 3. 核心假设

v5 的实验必须围绕假设组织，而不是围绕 run 名字组织。

---

## H1：`D_g_locked` 是 harmful-support cue，不是传统 dynamic-object mask

### 假设内容

`D_g_locked` 的语义不是“哪里是动态物体”，而是：

```text
哪些 patch token 不适合作为当前 frame-attention read path 的 support。
```

这解释了 v4 中两个现象：

```text
1. old_dyn / explicit_dyn routing 失败，因为它们把 D_g 拉向几何 residual false positive；
2. AnchorCollide 不能 hard reject，因为 attention read support 和 TTT write anchor 不是同一个概念。
```

### 实验设计

对以下 cue 做 passive + controlled audit：

```text
D_g_locked = C23 past
D_g_shadow = C24 past
D_g3 = C3 full
D_old = old_dyn_addclip
D_exp = explicit_dyn_only
C_anchor
confidence
uncertainty
future_only orientation cue
```

对每个 chunk 记录：

```text
D_g-only region
old_dyn-only region
agreement region
anchor-collision region
high-confidence high-D region
low-confidence high-D region
```

定义：

$$
Q_{g,old}^{11}=\mathbf{1}[D_g>\tau_g]\mathbf{1}[D_{old}>\tau_o]
$$

$$
Q_{g,old}^{10}=\mathbf{1}[D_g>\tau_g]\mathbf{1}[D_{old}\le\tau_o]
$$

$$
Q_{g,old}^{01}=\mathbf{1}[D_g\le\tau_g]\mathbf{1}[D_{old}>\tau_o]
$$

默认 $\tau_g=0.5, \tau_o=0.5$。

### 必须记录的指标

```text
mass_Q11
mass_Q10
mass_Q01
mass_Q00
AnchorCollide_Q10
AnchorCollide_Q01
ConfMean_Q10
ConfMean_Q01
UncMean_Q10
UncMean_Q01
ZoneMass_Q10_upper/middle/lower
ZoneMass_Q01_upper/middle/lower
chunk_delta_ATE_vs_C3
chunk_delta_ATE_vs_C23_full
chunk_delta_Rot_vs_C3
chunk_delta_FinalErr_vs_C3
```

### 可视化

必须输出：

```text
cue_dashboard/D_g_locked_vs_old_dyn/
    RGB
    D_g_locked
    D_old
    D_exp
    C_anchor
    confidence
    Q11/Q10/Q01 overlay
    attention suppression overlay
```

固定帧集合：

```text
1. 每 100 帧采样 1 帧；
2. largest-gain chunks 每个采样 2 帧；
3. largest-regression chunks 每个采样 2 帧；
4. KITTI02 中 C24 regression chunks 额外采样。
```

### 假设成立标准

H1 成立需要满足：

```text
1. D_g-only 区域能解释 C23/C24 相比 old_dyn 的主要 gain；
2. old_dyn-only 区域与 regression 或 quality warning 有正相关；
3. AnchorCollide 与 chunk delta ATE 不呈稳定正相关；
4. D_g 与 attention mass shift 有一致关系：high-D source 被 pair bias 后 attention mass 明显下降；
5. cue map 显示 D_g 不是简单的 moving object mask，而是包含 road/horizon/static support-risk 等区域。
```

量化 gate：

```text
Corr(old_dyn-only mass, delta ATE) >= 0.25 表示 old_dyn-only 可能是风险区域；
Corr(D_g-only mass, improvement vs C3) <= -0.25 表示 D_g-only 与改善有关；
AnchorCollide 不能作为 reject，除非 Corr(AnchorCollide, delta ATE) >= 0.35 且跨序列稳定。
```

---

## H2：`headmean` 可能稀释少数强 heads，per-head mining 可能继续提升 D_g

### 假设内容

当前 `D_g_locked` 使用 headmean。若 harmful-support 信号集中在少数 attention heads，则：

```text
head_topk / head_max / best_single_head 可能超过 headmean；
head_agreement 可能提高 reliability；
head_var / head_entropy 可能定位不稳定区域。
```

### 实验设计

只在以下结构上做 per-head，不回到全空间：

```text
layers = g2_3
support = past_only
statistic = qq low-similarity
normalization = robustq
```

同时保留 shadow：

```text
layers = g2_4
support = past_only
```

设第 $h$ 个 head 的 low-sim cue 为 $D_h$。构造以下 head aggregation：

#### Headmean baseline

$$
D_{mean}=\frac{1}{H}\sum_{h=1}^{H}D_h
$$

#### Head max

$$
D_{max}=\max_h D_h
$$

#### Head top-k mean

$$
D_{topk}=\frac{1}{k}\sum_{h\in TopK(D_h)}D_h
$$

建议：

```text
k = 2, 4
```

#### Head agreement

$$
A_{head}=\frac{1}{H}\sum_h \mathbf{1}[D_h>\tau_h]
$$

构造 agreement-gated cue：

$$
D_{agree}=D_{mean}\cdot \operatorname{clip}\left(\frac{A_{head}}{a_0},0,1\right)
$$

建议：

```text
a0 = 0.25, 0.50
```

#### Head variance

$$
D_{var}=\operatorname{Var}_h(D_h)
$$

可用作 reliability penalty：

$$
D_{varpen}=D_{mean}\cdot(1-\alpha\,\operatorname{norm}(D_{var}))
$$

#### Best single head

对每个 head 先做 passive audit，再选 top heads 跑 read-only：

```text
best_single_h0, best_single_h1, ..., best_single_hN
```

### 实验流程

#### Phase P1：per-head passive cache

对 KITTI01 全序列保存：

```text
D_h for all heads
D_mean
D_max
D_topk2
D_topk4
D_agree_a025
D_agree_a050
D_varpen_a025
D_varpen_a050
```

保存格式：

```text
cache/acl2_v5_perhead/kitti01/chunk_XXXX.npz
```

字段：

```text
D_heads: [H,T,Htok,Wtok]
D_mean
D_max
D_topk2
D_topk4
D_agreement
D_var
frame_ids
chunk_id
support_indices
layer_ids
head_ids
config_hash
```

#### Phase P2：passive quality ranking

不跑模型，先计算：

```text
mean_D
mass_D_gt_0.5
mass_D_gt_0.8
fragmentation
anchor_collision
old_dyn_corr
explicit_corr
confidence_corr
uncertainty_corr
temporal_consistency
zone_mass
head_agreement
head_entropy
```

过滤 gate：

```text
0.08 <= mass_D_gt_0.5 <= 0.35
coverage >= 0.95
fragmentation <= 0.20
old_dyn_corr <= 0.35
not all mass concentrated in upper_sky_horizon
```

这里 `AnchorCollide` 不作为 hard reject，只记录。

#### Phase P3：KITTI01 read-only gate

只允许 top 8 个 per-head candidates 进入 read-only：

```text
commit = probe_native
beta = 3.25, 3.50, 3.75, 4.00
read path = frame-attention pair bias
```

如果候选是 `D_max` 或 high-mass candidate，先从较低 beta 开始：

```text
beta = 2.75, 3.25, 3.75
```

#### Phase P4：hybrid-safe gate

只允许 read-only 通过者进入：

```text
commit = probe_ttt_write
write score = stage_d
beta = best read-only beta and beta_locked
```

#### Phase P5：cross-sequence sanity

最多 3 个 per-head 候选进入：

```text
KITTI00 full
KITTI02 full
KITTI05 full
```

与以下 baseline 比：

```text
C3 full
C23 past headmean
C24 past headmean
```

### 必须记录的指标

除 standard trajectory metrics 外，per-head 必须记录：

```text
head_id
head_rank_by_mass
head_rank_by_quality
head_rank_by_read_only_ATE
head_agreement_mean
head_agreement_p90
head_entropy_mean
head_var_mean
best_head_id_per_chunk
head_stability_across_chunks
head_stability_across_sequences
```

### 可视化

必须输出：

```text
per_head_dashboard/
    head_grid_maps_chunk_XXXX.png
    head_mass_histogram.png
    head_agreement_map.png
    head_var_map.png
    head_topk_vs_headmean_diff.png
    best_head_frequency_over_chunks.png
    per_head_quality_error_scatter.png
```

每张 head grid 至少显示：

```text
RGB
D_head_0 ... D_head_H
D_mean
D_topk2
D_topk4
D_agree
D_headmean - D_topk2
```

### 假设成立标准

H2 强成立：

```text
1. 某个 per-head aggregation 在 KITTI01 hybrid ATE <= 38.25；
2. 或者 ATE <= 38.35 且 Rot / FinalErr 明显优于 C23 past；
3. cross-sequence avg ATE 优于 C23 past >= 0.20m；
4. 没有任何序列 ATE regression > 3%。
```

H2 弱成立：

```text
1. per-head candidate 不明显降 ATE，但 Rot 改善 >= 0.10deg；
2. 或 FinalErr 改善 >= 0.30m；
3. 且 cross-sequence 不崩。
```

H2 不成立：

```text
headmean 仍然 Pareto 最优；
per-head topk/max 明显增加 false positive；
best_single_head 只在 KITTI01 有效，跨序列退化。
```

如果 H2 不成立，后续所有 write policy 均继续使用 `headmean C23 past`。

---

## H3：`D_g_locked` 的 read cue 不能直接等价为 TTT write cue，需要构造 write eligibility

### 假设内容

`D_g_locked` 是 read suppress cue，表示 harmful support。TTT 写入需要的是 static memory eligibility。两者相关，但不等价。

因此直接把 `D_g` 用作 write suppress 可能不稳定。更合理的是构造：

$$
E_g=1-D_g
$$

并加入 geometry / reliability / static rescue：

$$
E_{write}=\operatorname{clip}\left((1-D_g)^\gamma \cdot R_{static}\cdot R_{conf},0,1\right)
$$

第一版可以使用：

```text
E_write_g = 1 - D_g_locked
E_write_soft = sqrt(1 - D_g_locked)
E_write_hard = (1 - D_g_locked)^2
```

### 实验设计

固定 read cue：

```text
read cue = D_g_locked
read beta = 3.75
read intervention = frame-attention pair bias
```

只改变 TTT write policy。

写入分数候选：

```text
W0: stage_d current baseline
W1: E_write_g = 1 - D_g_locked
W2: E_write_soft = sqrt(1 - D_g_locked)
W3: E_write_hard = (1 - D_g_locked)^2
W4: residual_reliability diagnostic
W5: stage_d * E_write_g
W6: stage_d * E_write_soft
W7: stage_d with D_g high-token exact-preserve
```

### 运行流程

#### Phase T1：same-read write-score sweep

全部使用：

```text
mode = hybrid-safe
commit = probe_ttt_write
read cue = C23 past
beta = 3.75
```

运行：

| Run | Write score | 目的 |
|---|---|---|
| T1-00 | `stage_d` | baseline |
| T1-01 | `1-D_g` | D_g 转写入资格 |
| T1-02 | `sqrt(1-D_g)` | 更温和资格 |
| T1-03 | `(1-D_g)^2` | 更强保护 high-D |
| T1-04 | `stage_d * (1-D_g)` | stage_d 与 D_g eligibility 组合 |
| T1-05 | `stage_d * sqrt(1-D_g)` | 温和组合 |
| T1-06 | `residual_reliability` | 旧 reference diagnostic |

#### Phase T2：write strength sweep

只对 T1 通过者扫 write gain：

```text
write_strength = 0.50, 0.75, 1.00, 1.25
```

不要同时扫 read beta。read beta 固定 3.75。

#### Phase T3：read beta / write policy decoupling

对 T1/T2 最优两个候选补：

```text
read beta = 3.25, 3.75, 4.25
write_strength = best
```

目的：确认新的 write policy 不是只适配单个 read beta。

### 必须记录的指标

TTT write run 除 standard metrics 外，必须记录：

```text
write_score_source
write_strength
write_score_mean
write_score_p10/p50/p90
write_score_mass_lt_0.3
write_score_mass_gt_0.7
branch0_update_norm
branch1_update_norm
branch2_update_norm
layer_update_norm_mean
layer_update_norm_p90
update_cosine_to_native
update_cosine_to_stage_d
memory_state_rel_diff
memory_state_hash_before
memory_state_hash_after_probe
memory_state_hash_after_commit
write_gain_vs_read_only
```

定义：

$$
\Delta_{write}=ATE_{read-only}(D_g,\beta)-ATE_{hybrid}(D_g,\beta,W)
$$

$$
UpdateCos=\frac{\langle \Delta W_{candidate},\Delta W_{stage\_d}\rangle}{\|\Delta W_{candidate}\|\|\Delta W_{stage\_d}\|+\epsilon}
$$

$$
MemDiff=\frac{\|W_{after}-W_{probe}\|_F}{\|W_{probe}\|_F+\epsilon}
$$

### 可视化

必须输出：

```text
ttt_write_dashboard/
    write_score_map_vs_Dg.png
    per_layer_update_norm_heatmap.png
    per_branch_update_norm_heatmap.png
    update_cosine_to_stage_d_heatmap.png
    memory_diff_over_chunks.png
    write_gain_curve.png
    chunk_delta_vs_write_score_mass.png
```

### 假设成立标准

H3 成立：

```text
1. 某个 E_write candidate 优于 stage_d baseline；
2. KITTI01 hybrid ATE <= 38.30，或 cross-sequence avg ATE 优于 stage_d >= 0.20m；
3. write_gain 同 beta 仍为正；
4. update_cosine_to_stage_d 不接近 1，说明不是等价写入；
5. Rot / FinalErr 不明显恶化。
```

H3 部分成立：

```text
ATE 与 stage_d 接近，但 Rot / FinalErr / cross-seq 更稳；
或 branch/layer diagnostic 清楚揭示哪些 write 成分有害。
```

H3 不成立：

```text
stage_d 仍最优；
D_g-based write eligibility 导致 ATE 回退或 cross-seq 退化；
说明 D_g 目前只适合 read cue，不适合直接转 write cue。
```

---

## H4：TTT write policy 的有效性可能是 branch-specific / layer-specific

### 假设内容

当前 `probe_ttt_write` 主要继承 Phase D/F 中 branch0 write 的成功经验。但 LoGeR 的 TTT fast-weight模块有多个分支。不同分支和不同 TTT 层可能承担不同功能：

```text
branch0: 可能更接近 gate / alignment / memory access；
branch1: 可能更接近 output projection；
branch2: 可能更接近 content；
early TTT layers: 可能影响局部表示稳定；
mid/late TTT layers: 可能影响 global trajectory / pose。
```

因此需要验证：

```text
是否只有 branch0 应被控制？
是否某些 layer 的 write 应 exact-preserve？
是否 late layers 控制带来 rotation / endpoint 改善？
```

### 实验设计

固定：

```text
read cue = D_g_locked
read beta = 3.75
commit = probe_ttt_write
write score baseline = stage_d
```

#### Branch matrix

| Run | Branch policy |
|---|---|
| B0 | branch0 only current baseline |
| B1 | branch1 only |
| B2 | branch2 only |
| B01 | branch0 + branch1 |
| B02 | branch0 + branch2 |
| B12 | branch1 + branch2 |
| B012 | all branches |

#### Layer matrix

把 TTT layers 按 block index 划分：

```text
early_ttt
mid_ttt
late_ttt
all_ttt
```

第一轮只测：

```text
branch0 × early/mid/late/all
branch0+2 × early/mid/late/all
```

#### Sparse / exact-preserve matrix

定义 high-risk token：

$$
M_{risk}=\mathbf{1}[D_g>\tau]
$$

对应 write suppress：

$$
E_{sparse}=1-M_{risk}\cdot s
$$

参数：

```text
tau = 0.5, 0.7
s = 0.5, 1.0
s=1.0 表示 high-D exact-preserve / no-write
```

### 运行顺序

#### Phase T4A：branch-only smoke full

只跑 KITTI01 hybrid：

```text
B0, B1, B2, B02, B012
```

如果 B1/B2 完全崩，则不再扩大。

#### Phase T4B：layer-specific write

只对 B0 和 B02：

```text
early only
mid only
late only
all
```

#### Phase T4C：sparse exact-preserve

只在 best branch/layer 上试：

```text
tau = 0.5, 0.7
s = 0.5, 1.0
```

### 必须记录的指标

```text
branch_policy
layer_policy
sparse_tau
sparse_strength
branch_update_norm
layer_update_norm
branch_layer_update_norm_matrix
update_cosine_to_B0
memory_diff
write_gain
per_chunk_delta_ATE
per_chunk_delta_Rot
FinalErr
YawRMSE
cross_seq_avg
```

### 可视化

```text
ttt_branch_layer_dashboard/
    branch_layer_update_norm_heatmap.png
    branch_layer_update_cosine_heatmap.png
    branch_policy_beta_curve.png
    layer_policy_trajectory_overlay.png
    sparse_mask_vs_Dg_overlay.png
```

### 假设成立标准

强成立：

```text
1. 某个 branch/layer policy 在 KITTI01 ATE <= 38.25；
2. 或 cross-seq average ATE 优于 B0 baseline >= 0.25m；
3. 没有任何 full sequence regression > 3%；
4. update pattern 可解释，不是只靠 update norm 缩小。
```

弱成立：

```text
1. ATE 接近 baseline，但 Rot / FinalErr 明显更好；
2. 或揭示某些 branch/layer 明确有害，可用于缩小未来 search。
```

不成立：

```text
B0 branch0 all-layers stage_d 仍最优；
其它 branch/layer 要么退化，要么只在 KITTI01 小幅波动。
```

---

## H5：SWA policy 不应作为全局 ATE read 替代，而应作为 local-memory write/read-cache policy 评价

### 假设内容

LoGeR 中 TTT 是 compressed global memory，SWA 是 lossless local memory。SWA 的目标更接近：

```text
相邻 chunk 的局部高保真对齐；
chunk boundary 的几何连续；
overlap / seam 的局部一致性。
```

因此 SWA policy 的成功标准不能只看 global ATE。v4 的 SWA-only ATE 接近 no-control，不代表 SWA 没价值；它可能只是无法单独替代 frame-attention pair bias。

v5 对 SWA 的假设是：

```text
SWA policy 作为 FA pair bias 的辅助 local-memory policy，可能改善 boundary metrics、FinalErr、rotation 或 local seam，而不是单独降 ATE。
```

### SWA policy 的定义

在 v5 中，将 SWA policy 分成三类。

#### SWA-cache metadata write

当前 chunk 的 token 被存入下一 chunk SWA local memory时，同时保存 reliability：

$$
R_i^{swa}=1-D_g(i)
$$

cache 中保存：

```text
H_i
R_i_swa
D_i
frame_id
chunk_id
layer_id
```

#### SWA previous-source value gate

下一 chunk 读取 previous source 时：

$$
V_j'=g_j^{src}V_j
$$

$$
g_j^{src}=\operatorname{clip}(1-\rho D_j,g_{min},1)
$$

#### SWA cross-chunk attention bias

不直接改 value，而是对 previous-source attention logit 加 bias：

$$
\ell_{ij}'=\ell_{ij}-\beta_{swa}D_j
$$

这比 value gate 更接近 attention routing 控制，可能更安全。

### 实验设计

固定主 read cue baseline：

```text
FA-BIAS baseline:
D_g_locked + frame-attention pair bias + probe_ttt_write
beta = 3.75
```

SWA 不再先跑大规模 SWA-only。只做小矩阵：

#### Group S0：SWA hook correctness

```text
read path = native/no-control
SWA gate = identity
```

必须和 native / baseline byte-level 或 metric-level parity。

#### Group S1：SWA-only diagnostic

只保留少量：

```text
SWA previous-source value gate:
    rho = 0.05, g_min = 0.85
    rho = 0.10, g_min = 0.85

SWA previous-source attention bias:
    beta_swa = 0.25, 0.50
```

目的不是晋级，而是记录 boundary/local 指标。

#### Group S2：FA bias + SWA value gate

```text
FA bias beta = 3.75
SWA rho = 0.025, 0.05, 0.10
SWA g_min = 0.85, 0.90
commit = probe_ttt_write
```

#### Group S3：FA bias + SWA attention bias

```text
FA bias beta = 3.75
SWA beta_swa = 0.10, 0.25, 0.50
commit = probe_ttt_write
```

#### Group S4：SWA layer selection

只对通过 S2/S3 的候选测：

```text
first_swa_only
first_two_swa
all_swa
```

### 必须记录的指标

标准 trajectory：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
ATE_50
ATE_100
ATE_200
YawRMSE
Sim3Scale
```

SWA local-memory 指标：

```text
swa_gate_calls
swa_gate_mean
swa_gate_p10/p50/p90
swa_gate_min/max
swa_source_highD_mass
swa_source_anchor_mass
cross_chunk_attention_mass_before
cross_chunk_attention_mass_after
attention_mass_to_highD_previous_before
aattention_mass_to_highD_previous_after
attention_mass_to_anchor_previous_before
attention_mass_to_anchor_previous_after
swa_value_norm_before
swa_value_norm_after
swa_residual_norm_before
swa_residual_norm_after
```

Boundary / seam 指标：

```text
chunk_boundary_pose_jump
boundary_10f_ATE
boundary_20f_ATE
overlap_pointmap_residual
overlap_pose_discontinuity
pre_boundary_yaw_jump
post_boundary_yaw_jump
pointmap_seam_error
```

定义 boundary pose jump：

$$
E_{bdry}(m)=\left\|\log\left(T_{m,end}^{-1}T_{m+1,start}\right)\right\|
$$

定义 overlap pointmap residual：

$$
E_{overlap}(m)=\operatorname{Mean}_{p}\left\|X_{m,k}(p)-X_{m+1,k}(p)\right\|_2
$$

### 可视化

```text
swa_policy_dashboard/
    boundary_error_over_chunks.png
    overlap_pointmap_residual_over_chunks.png
    cross_chunk_attention_mass_before_after.png
    swa_gate_map_previous_source.png
    highD_previous_source_attention_overlay.png
    pointmap_seam_visualization.png
    trajectory_boundary_markers.png
```

### 假设成立标准

SWA policy 强成立：

```text
1. FA bias + SWA policy hybrid ATE <= D_g_locked baseline ATE；
2. boundary_10f_ATE 改善 >= 3%；
3. overlap_pointmap_residual 改善 >= 5%；
4. FinalErr 或 Rot 至少一项改善；
5. cross-seq 不退化。
```

SWA policy 弱成立：

```text
1. global ATE 回退 <= 0.08m；
2. boundary/local 指标明显改善；
3. 可作为 local-alignment variant 保留。
```

SWA policy 不成立：

```text
1. global ATE 回退 > 0.15m；
2. boundary/local 指标无改善；
3. 或只改变 SWA norm，不改变实际 seam / boundary error。
```

注意：SWA-only 不通过不直接终止 SWA line；只有 FA bias + SWA 也不改善 local 指标，才停止。

---

## H6：pipeline 加速必须在不改变结果的前提下完成

### 假设内容

v5 的实验数量会比 v4 更多。如果继续每个 run 独立生成 cue、独立 probe、独立可视化，资源会不可控。

加速假设是：

```text
通过 cue cache、shared probe、result registry、resource-aware scheduler 和 offline visualization，可以把候选吞吐提升 30%-50%，同时保持 metric parity。
```

### 加速设计

#### A. Run registry 与 config hash

每个 run 生成唯一 hash：

```text
hash = SHA1(
    model_commit
    dataset
    sequence
    frame_range
    cue_config
    read_config
    write_config
    commit_config
    seed
)
```

保存：

```text
runs/registry.csv
runs/{hash}/run_config.yaml
runs/{hash}/metrics_global.json
runs/{hash}/runtime_summary.json
runs/{hash}/status.json
```

如果 hash 已存在且 status=success，则跳过。

#### B. Passive cue cache

对固定 `D_g_locked` 和 per-head candidates：

```text
先生成 cue cache，再复用到多个 read/write/intervention runs。
```

cache 分层：

```text
cache/geometry_probe/
cache/acl2_qk_patchvec/
cache/acl2_perhead/
cache/cue_bank/
cache/visualization_inputs/
```

每个 cache 带：

```text
source_model_hash
sequence_id
chunk_size
overlap
layer_ids
head_ids
support_pattern
normalization
shape
checksum
```

#### C. Shared probe pass

对同一 dataset / sequence / base model / HMC state：

```text
Pass 1 probe/native forward 可复用；
cue extraction 可复用；
只有 Pass 2 controlled forward 需要按 candidate 跑。
```

如果当前 runner 不支持完全复用 Pass 1 state，至少复用：

```text
q/k patchvec
D_g cue maps
per-head maps
geometry confidence / anchors
old_dyn / explicit maps
```

#### D. CPU/GPU 优化

当前 ACL2 部分慢点来自 CPU 上做 q/k layer-window/support 统计。v5 优化目标：

```text
1. support centroid 预计算；
2. per-head q/k similarity 在 GPU batch 化；
3. cache fp16/bfloat16 存储，metric 计算用 fp32；
4. 避免重复 einsum；
5. visualization 从主 run 中分离，离线生成。
```

#### E. 资源调度

固定规则：

```text
KITTI01 full:
    max_concurrency = 4

KITTI00/02 long full:
    max_concurrency = 2

per-head passive cache:
    根据 host RAM 自动分批；默认 2-4

visualization:
    不占 GPU 主队列，单独 CPU/offline 队列
```

如果 host available RAM < 64GB：

```text
不启动新 full run；
等待当前 run 完成。
```

如果 available RAM < 24GB：

```text
主动暂停 queue，避免 swap / OOM。
```

#### F. 两阶段 gate

v5 不使用 short slice 作为 promotion 依据，但可以用于工程 smoke。

```text
smoke END_FRAME=128:
    只验证 hook / cache / shape / correctness

passive full audit:
    过滤明显坏 cue

KITTI01 read-only:
    判断 read cue / intervention 是否值得 hybrid

KITTI01 hybrid:
    判断是否过主集 gate

cross-sequence:
    判断是否 promote
```

### 加速指标

每个 run 保存：

```text
walltime_sec
queue_wait_sec
probe_time_sec
cue_time_sec
controlled_forward_time_sec
evaluation_time_sec
visualization_time_sec
peak_gpu_mem_gb
peak_host_ram_gb
host_ram_available_min_gb
gpu_util_mean
gpu_util_p90
cpu_util_mean
io_read_gb
io_write_gb
cache_hit_rate
num_cache_loaded
num_cache_written
```

### 加速成功标准

强成功：

```text
1. KITTI01 full candidate 平均 walltime 降低 >= 30%；
2. passive/per-head cue sweep walltime 降低 >= 50%；
3. no-cache vs cache metric ATE 差异 <= 0.005m；
4. no-cache vs cache Rot 差异 <= 0.005deg；
5. 0 OOM / 0 swap incident；
6. registry skip rate >= 20%。
```

弱成功：

```text
walltime 降低 15%-30%，且稳定性明显改善。
```

失败：

```text
cache 导致 metric drift；
或 runtime 降低不明显；
或资源调度仍频繁 OOM。
```

---

## 4. 统一记录指标

v5 所有实验，不管是 per-head、TTT、SWA 还是 acceleration，都必须统一保存下列文件。

### 4.1 Global trajectory metrics

```text
metrics_global.json
```

字段：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
ATE_50
ATE_100
ATE_200
YawRMSE
Sim3Scale
num_frames
num_chunks
sequence_id
candidate_name
```

### 4.2 Per-frame / per-chunk error

```text
per_frame_error.csv
per_chunk_error.csv
```

字段：

```text
frame_id
chunk_id
translation_error
rotation_error
yaw_error
aligned_error

chunk_id
frame_start
frame_end
chunk_ate
chunk_rot
chunk_final_error
chunk_yaw_rmse
chunk_scale_proxy
worst_frame_id
worst_frame_error
```

### 4.3 Cue quality metrics

```text
cue_quality_summary.json
cue_quality_per_chunk.jsonl
```

字段：

```text
mean_D
std_D
p10_D
p50_D
p90_D
mass_D_gt_0.5
mass_D_gt_0.8
coverage
fragmentation
anchor_collision
confidence_corr
old_dyn_corr
explicit_corr
C_anchor_corr
uncertainty_corr
zone_upper_mass
zone_middle_mass
zone_lower_mass
zone_center_mass
zone_border_mass
temporal_consistency
support_concentration
```

Fragmentation：

$$
Frag(D)=\frac{1}{THW}\sum_{t,h,w}\mathbf{1}\left[|D_{t,h,w}-D_{t,h,w+1}|>\tau_f\right]+\mathbf{1}\left[|D_{t,h,w}-D_{t,h+1,w}|>\tau_f\right]
$$

Temporal consistency：

$$
TempCons(D)=1-\frac{1}{T-1}\sum_{t=1}^{T-1}\operatorname{Mean}_{h,w}|D_{t+1,h,w}-D_{t,h,w}|
$$

### 4.4 Read effect metrics

```text
read_effect_summary.jsonl
```

字段：

```text
chunk_id
layer_id
head_id
read_path
bias_mode
beta
mean_abs_bias
p95_abs_bias
suppressed_token_ratio
attn_shift_l1
attn_shift_l2
attn_mass_to_highD_before
attn_mass_to_highD_after
attn_mass_to_anchor_before
attn_mass_to_anchor_after
same_frame_attention_mass
cross_frame_attention_mass
support_concentration
```

关键量：

$$
\Delta Attn_{highD}=Mass_{after}(D>\tau)-Mass_{before}(D>\tau)
$$

正常 suppress 应满足：

```text
Delta_Attn_highD < 0
```

但 anchor attention 不应大幅下降：

```text
Mass_after(anchor) >= Mass_before(anchor) - tolerance
```

### 4.5 TTT write metrics

```text
ttt_write_summary.jsonl
memory_delta_summary.jsonl
hmc_state_hash.jsonl
```

字段：

```text
chunk_id
layer_id
branch_id
commit_mode
write_score_source
write_strength
update_norm_native
update_norm_candidate
update_norm_stage_d
update_cosine_to_native
update_cosine_to_stage_d
memory_state_rel_diff
memory_state_hash_before
memory_state_hash_after_probe
memory_state_hash_after_commit
branch0_update_norm
branch1_update_norm
branch2_update_norm
```

### 4.6 SWA metrics

```text
swa_policy_summary.jsonl
boundary_metrics.jsonl
```

字段：

```text
chunk_id
swa_layer_id
swa_policy
rho
beta_swa
g_min
source_type
swa_gate_mean
swa_gate_p10
swa_gate_p50
swa_gate_p90
cross_chunk_attention_mass_before
cross_chunk_attention_mass_after
attention_mass_to_highD_previous_before
attention_mass_to_highD_previous_after
attention_mass_to_anchor_previous_before
attention_mass_to_anchor_previous_after
swa_residual_norm_before
swa_residual_norm_after
chunk_boundary_pose_jump
boundary_10f_ATE
boundary_20f_ATE
overlap_pointmap_residual
pointmap_seam_error
```

### 4.7 Runtime / acceleration metrics

```text
runtime_summary.json
resource_trace.csv
```

字段：

```text
walltime_sec
probe_time_sec
cue_time_sec
controlled_forward_time_sec
eval_time_sec
visualization_time_sec
peak_gpu_mem_gb
peak_host_ram_gb
host_ram_available_min_gb
gpu_util_mean
gpu_util_p90
cpu_util_mean
io_read_gb
io_write_gb
cache_hit_rate
queue_concurrency
num_failed_chunks
```

---

## 5. 必须可视化内容

每个晋级候选必须输出可视化。v5 不接受只有 ATE 表格的晋级。

### 5.1 D_g interpretation dashboard

```text
visuals/dg_interpretation/
    RGB
    D_g_locked
    D_g_shadow
    D_g3
    old_dyn_addclip
    explicit_dyn_only
    C_anchor
    confidence
    uncertainty
    D_g-only / old-only / agreement overlay
    attention mass before/after overlay
```

### 5.2 Per-head dashboard

```text
visuals/per_head/
    head_grid_all_heads.png
    headmean_vs_topk2.png
    headmean_vs_topk4.png
    head_agreement_map.png
    head_variance_map.png
    best_head_frequency.png
    per_head_quality_error_scatter.png
```

### 5.3 TTT write dashboard

```text
visuals/ttt_write/
    write_score_maps.png
    branch_update_norm_heatmap.png
    layer_update_norm_heatmap.png
    branch_layer_update_norm_heatmap.png
    update_cosine_to_stage_d.png
    memory_diff_over_chunks.png
    write_gain_curve.png
```

### 5.4 SWA policy dashboard

```text
visuals/swa_policy/
    chunk_boundary_pose_jump_curve.png
    boundary_10f_ATE_curve.png
    overlap_pointmap_residual_curve.png
    cross_chunk_attention_mass_before_after.png
    highD_previous_source_attention_overlay.png
    swa_gate_map.png
    pointmap_seam_visualization.png
```

### 5.5 Trajectory comparison

每个晋级候选输出：

```text
GT trajectory
native LoGeR
C3 g3.full
C23 past locked
C24 past shadow
candidate
```

至少三张：

```text
full XY
first half XY
second half XY
```

并标记：

```text
worst chunks
largest gain chunks
largest regression chunks
chunk boundary positions
```

### 5.6 Error-over-time

```text
per-frame translation error
per-frame rotation error
sliding 50f ATE
sliding 100f ATE
sliding 200f ATE
cumulative yaw drift proxy
FinalErr over chunk index
```

### 5.7 Resource dashboard

```text
visuals/resources/
    walltime_by_stage.png
    host_ram_over_time.png
    gpu_mem_over_time.png
    gpu_util_over_time.png
    cache_hit_rate_by_candidate.png
    throughput_runs_per_hour.png
```

---

## 6. 阶段化执行计划

---

## Stage 0：工程锁定、cache parity 与 baseline freeze

### 目标

确保 v5 的加速和 cache 不改变结果。

### 运行列表

| Run | Candidate | Mode | Commit | 目的 |
|---|---|---|---|---|
| S0-01 | `C23 past` no-cache | hybrid | `probe_ttt_write` | locked baseline repeat |
| S0-02 | `C23 past` cache | hybrid | `probe_ttt_write` | cache parity |
| S0-03 | `C24 past` no-cache | hybrid | `probe_ttt_write` | shadow repeat |
| S0-04 | `C24 past` cache | hybrid | `probe_ttt_write` | cache parity |
| S0-05 | `C23 past` read-only | read-only | `probe_native` | read baseline |
| S0-06 | `C23 past` smoke END_FRAME=128 | smoke | n/a | hook/cue shape |

### 通过标准

```text
cache vs no-cache ATE diff <= 0.005m
cache vs no-cache Rot diff <= 0.005deg
C23 past repeat equals 38.3706 / 8.6694 within 0.03m / 0.03deg
C24 past repeat equals 38.3566 / 8.7660 within 0.03m / 0.03deg
no failed chunks
host RAM available min > 64GB for KITTI01 4-concurrency
```

如果 S0 不通过，不进入 per-head 或 write policy。

---

## Stage 1：解释 `D_g_locked`

### 目标

回答 `D_g_locked` 到底 suppress 什么、和 old_dyn/explicit 有何不同、为什么 C23 比 C24 更稳。

### 实验内容

不新增 controlled full run，复用 v4 结果和 cache，做 passive join：

```text
C23 past
C24 past
C23 full
C24 full
C3 full
old_dyn_addclip
explicit_dyn_only
future_only diagnostic
C_anchor
confidence
uncertainty
```

### 输出文件

```text
stage1_dg_interpretation/
    cue_quality_join.csv
    quadrant_metrics.csv
    quality_error_corr.csv
    chunk_gain_regression_table.csv
    cue_dashboard/
```

### 判断标准

Stage 1 完成条件：

```text
1. 能定位 C23 vs C24 的主要差异 chunk；
2. 能解释 old_dyn routing 为什么失败：old-only region 与 regression/uncertainty/zone bias 有关；
3. 能输出至少 20 个 key frame dashboard；
4. 能为 per-head 和 write policy 指出重点 chunk。
```

---

## Stage 2：per-head cue mining

### 目标

判断 headmean 是否最优。

### 实验矩阵

#### Passive candidates

```text
C23_past_headmean
C23_past_headmax
C23_past_topk2
C23_past_topk4
C23_past_head_agree_a025
C23_past_head_agree_a050
C23_past_varpen_a025
C23_past_varpen_a050
C23_past_best_single_head_top4
C24_past_topk2
C24_past_head_agree_a050
```

#### Read-only gate

只跑 top 8：

```text
beta = 3.25, 3.75, 4.00
commit = probe_native
```

#### Hybrid gate

只跑 top 3：

```text
beta = best read-only beta and 3.75
commit = probe_ttt_write
write score = stage_d
```

#### Cross-sequence

只跑 top 1-2：

```text
KITTI00 full
KITTI02 full
KITTI05 full
```

### 晋级标准

```text
KITTI01 strong:
    hybrid ATE <= 38.25

KITTI01 balanced:
    hybrid ATE <= 38.35 and Rot <= 8.65

Cross-seq promote:
    avg ATE improves over C23 past >= 0.20m
    no sequence regression > 3%
```

如果没有 per-head 晋级，headmean 保持主线。

---

## Stage 3：TTT write-score policy

### 目标

固定 read cue，探索更合适的 TTT write score。

### 实验矩阵

固定：

```text
read cue = C23 past
read beta = 3.75
commit = probe_ttt_write
```

写入分数：

```text
stage_d
1-D_g
sqrt(1-D_g)
(1-D_g)^2
stage_d * (1-D_g)
stage_d * sqrt(1-D_g)
residual_reliability diagnostic
```

第一轮 KITTI01 full hybrid，第二轮对 top 2 做 cross-sequence。

### 晋级标准

```text
Strong:
    KITTI01 ATE <= 38.25
    or cross-seq avg ATE improves >= 0.25m

Balanced:
    ATE close to baseline within 0.03m
    and Rot or FinalErr improves clearly
```

---

## Stage 4：TTT branch / layer / sparse policy

### 目标

验证 write 是否 branch-specific 或 layer-specific。

### 实验矩阵

#### Branch policy

```text
branch0 baseline
branch1 only
branch2 only
branch0+1
branch0+2
all branches
```

#### Layer policy

```text
branch0 early only
branch0 mid only
branch0 late only
branch0 all
branch0+2 early/mid/late/all
```

#### Sparse policy

```text
tau = 0.5, 0.7
s = 0.5, 1.0
```

### Gate

先 KITTI01，只有满足下面任一条件进 cross-seq：

```text
ATE <= 38.30
or FinalErr improves >= 0.40m with ATE regression <= 0.03m
or Rot improves >= 0.10deg with ATE regression <= 0.03m
```

---

## Stage 5：SWA local-memory policy

### 目标

继续探索 SWA，但换评价标准：看 local / boundary，而不是期待 SWA-only 全局 ATE 降到 38.x。

### 实验矩阵

#### Correctness

```text
SWA identity gate
SWA metadata cache no-op
```

#### SWA-only diagnostic

```text
previous-source value gate rho=0.05 gmin=0.85
previous-source value gate rho=0.10 gmin=0.85
previous-source attention bias beta_swa=0.25
previous-source attention bias beta_swa=0.50
```

#### FA + SWA combination

```text
FA pair beta=3.75 + SWA value gate rho=0.025 gmin=0.90
FA pair beta=3.75 + SWA value gate rho=0.05  gmin=0.90
FA pair beta=3.75 + SWA value gate rho=0.05  gmin=0.85
FA pair beta=3.75 + SWA attn bias beta_swa=0.10
FA pair beta=3.75 + SWA attn bias beta_swa=0.25
```

### Gate

```text
SWA-only:
    不作为主线晋级，只记录 boundary/local

FA+SWA success:
    global ATE <= 38.3706 + 0.08
    and boundary_10f_ATE improves >= 3%
    or overlap_pointmap_residual improves >= 5%
    or FinalErr improves >= 0.30m
```

如果 FA+SWA 没有任何 local 指标改善，SWA policy 暂停。

---

---

## Stage 6：Final promotion and reporting

### 目标

把所有候选分为：

```text
promoted mainline
shadow candidate
diagnostic only
rejected
blocked by engineering
```

### Promotion rules

候选必须满足：

```text
1. KITTI01 不弱于 C23 past mainline；
2. cross-sequence average 优于 C23 past 或至少不退化；
3. ATE / Rot / FinalErr / boundary 至少两类指标有同步改善；
4. 机制可解释，有 cue map 和 write/SWA diagnostics；
5. 资源开销可接受。
```

强 promotion：

```text
KITTI01 ATE <= 38.25
or cross-seq avg ATE improves >= 0.25m
and no sequence regression > 3%
```

弱 promotion：

```text
ATE 持平，但 Rot / FinalErr / boundary 明显改善，作为 balanced variant。
```

---

## 7. 推荐首批运行顺序

首批不应太大。我建议按下面顺序执行。

### Batch 1：工程和解释，不跑新 full 大矩阵

```text
1. S0 cache parity: C23 past / C24 past
2. D_g interpretation passive join
3. support index / overlap_excluded correctness smoke
4. resource dashboard baseline
```

### Batch 2：per-head passive + read-only top candidates

```text
1. Generate C23 past per-head cache
2. Passive rank per-head candidates
3. Read-only top 8 on KITTI01
4. Hybrid top 3 on KITTI01
```

### Batch 3：TTT write-score小矩阵

```text
1. stage_d baseline repeat
2. 1-D_g
3. sqrt(1-D_g)
4. stage_d*(1-D_g)
5. stage_d*sqrt(1-D_g)
```

### Batch 4：SWA policy minimal组合

```text
1. SWA identity correctness
2. SWA-only diagnostic r0.05
3. FA bias + SWA value r0.025 / r0.05
4. FA bias + SWA attention bias beta_swa 0.10 / 0.25
```

### Batch 5：cross-sequence sanity

进入 cross-seq 的最多 4 个：

```text
C23 past locked baseline
best per-head candidate
best TTT write candidate
best SWA local variant, only if local metrics pass
```

---

## 8. 预期决策树

### 情况 A：per-head 成功

如果 per-head top-k/head-agreement 明显超过 headmean：

```text
D_g_locked 更新为 per-head candidate；
后续 TTT/SWA policy 全部用新 D_g；
C23 headmean 变成 baseline。
```

### 情况 B：per-head 不成功，TTT write 成功

```text
保留 C23 past headmean read cue；
更新 write policy；
进入 branch/layer sparse write。
```

### 情况 C：per-head 和 TTT write 都不成功，SWA local 成功

```text
主线仍是 C23 past + stage_d；
SWA local policy 作为 balanced/local-alignment variant；
重点报告 boundary/local 改善。
```

### 情况 D：三条都不成功

```text
v5 结论为：C23 past headmean + stage_d + FA pair bias 已接近当前 internal-attention pipeline 的稳定上限；
下一步需要新 cue source 或学习式 reliability model。
```

---

## 9. 最终交付物

v5 完成后应交付：

```text
1. ACL2_v5_experiment_report.md
2. metrics_summary_all_runs.csv
3. cross_sequence_summary.csv
4. per_head_audit_summary.csv
5. ttt_write_policy_summary.csv
6. swa_policy_summary.csv
7. resource_acceleration_summary.csv
8. visuals/
9. cache_manifest.json
10. promoted_candidate_registry.yaml
```

报告必须回答：

```text
D_g_locked 是否被 per-head 替代？
TTT write policy 是否超过 stage_d？
SWA local-memory policy 是否有 local/boundary 价值？
pipeline 加速是否可靠？
当前 mainline 是什么？
哪些方向停止，哪些保留？
```

---

## 10. 当前推荐主线

在 v5 开始前，当前推荐主线保持：

```text
D_g_locked:
    acl2.gg.qq.low.g2_3.past_only.headmean.robustq

beta:
    3.75

read intervention:
    frame-attention early-layer pair bias

commit:
    probe_ttt_write

write score:
    stage_d
```

当前 shadow：

```text
D_g_ate_shadow:
    acl2.gg.qq.low.g2_4.past_only.headmean.robustq
    beta = 4.00
```

v5 的原则是：

```text
先解释，再 per-head；
先固定 read cue，再探索 TTT write；
SWA 继续探索，但用 local-memory / boundary 指标评价；
所有新实验必须进入加速后的 registry/cache/scheduler。
```

一句话总结：

> v5 不再是无边界 cue search，而是以 `C23 past_only` 为锁定主线，系统回答“headmean 是否最优、TTT 怎样写更安全、SWA local memory 是否能补边界、pipeline 怎样跑得更快”这四个问题。

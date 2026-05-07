# ACL2 v4：早中层 Global Query Cue 因果确认、质量重标定与组合验证准备实验计划

日期：2026-05-05  
对象：LoGeR / HMC Pipeline v2 / ACL2 attention cue library  
主开发集：KITTI Odometry Sequence 01  
当前最强候选：

```text
ATE-oriented candidate:
acl2.gg.qq.low.g2_4.full.headmean.robustq + probe_ttt_write
beta = 4.00
ATE / Rot = 38.3805 / 8.8707
FinalErr = 4.503
YawRMSE = 5.401

Balanced / Pareto candidate:
acl2.gg.qq.low.g2_3.full.headmean.robustq + probe_ttt_write
beta = 3.75
ATE / Rot = 38.3847 / 8.7583
FinalErr = 3.677
YawRMSE = 5.291

Previous ACL2 anchor:
acl2.gg.qq.low.g3.full.headmean.robustq + probe_ttt_write
beta = 3.75
ATE / Rot = 38.4298 / 8.9846
FinalErr = 4.692
YawRMSE = 5.477
```

---

## 0. 本计划先回答一个关键判断

当前实验已经不是最初的 `g3.full` 单点发现阶段了，但严格说，**当前仍然主要处在 single-cue mining / single-cue causal validation 阶段**。

目前最强结果都是：

```text
一个 global query-query low-sim attention cue
+
frame-attention pair bias read control
+
probe_ttt_write safe memory protocol
```

其中 `probe_ttt_write` 是 memory commit / write protocol，不是 cue 组合。也就是说，下面这些真正的 cue 组合还没有系统验证：

```text
D_g + old_dyn_addclip
D_g + explicit_dyn_only
D_g + deep static rescue
D_g + C_anchor protection
D_g + reliability routing
D_g + residual / SWA intervention path
```

因此，下一阶段不能直接宣布 “cue combination 已经成功”。更准确的阶段目标应该是：

> 先把 `g2_3.full` / `g2_4.full` 这两个早中层 global query cue 的因果关系、稳定性、support 依赖、layer 贡献和质量指标解释清楚；然后锁定一个 `D_g_locked`，再进入小规模、可归因的 cue combination validation。

---

## 1. 实验整体目标

本阶段的目标不是盲目把 KITTI01 ATE 从 `38.38m` 再刷低一点，而是建立一个可以支撑后续工作的可靠结论链。

本计划要回答六个问题。

### 1.1 问题 A：`g2_3` / `g2_4` 是否真的稳定超过 `g3`？

`g2_4.full` 当前 ATE best 是 `38.3805m`，比 `g3.full` 的 `38.4298m` 好 `0.0493m`。这个提升是健康的，但并不大。`g2_3.full` 的 ATE 是 `38.3847m`，只比 `g2_4.full` 差 `0.0042m`，却在 rotation、final error、yaw 上明显更好。

所以不能只看单个最小 ATE。必须确认：

```text
1. g2_3 / g2_4 的结果是否可复现；
2. 它们是否在同 beta read-only / hybrid-safe 对照下仍然成立；
3. 它们的优势是否体现在 trajectory diagnostics，而不是单个 ATE 偶然波动；
4. g2_3 是否应该作为主线候选，g2_4 是否只作为 ATE-oriented 对照。
```

### 1.2 问题 B：safe write 的增益在新窗口上是否仍然稳定？

在 `g3.full` 上，同 beta 比较已经证明 `probe_ttt_write` 稳定提供约 `0.23m ~ 0.31m` 的 ATE gain。对 `g2_3` 和 `g2_4`，目前还没有完整同 beta read-only / hybrid-safe 曲线。

必须计算：

$$
\Delta_{write}(\beta)=ATE_{read}(\beta)-ATE_{hybrid}(\beta)
$$

如果 `g2_3/g2_4` 的提升主要来自 read cue 本身，而 safe write 的收益仍然稳定，那么它们可以继承当前 HMC v2 主协议。如果 safe write 在这些窗口上收益变小或反向，说明新的 cue 与 branch0 write 的相互作用已经改变，后续不能直接沿用旧 write protocol。

### 1.3 问题 C：`g4` 是有用补充，还是 rotation / endpoint 损伤来源？

`g2_3.full` 和 `g2_4.full` 的差异非常关键：

```text
g2_4 比 g2_3 ATE 好 0.0042m，几乎可以忽略；
g2_3 比 g2_4 Rot 好 0.1124deg；
g2_3 比 g2_4 FinalErr 好 0.826m；
g2_3 比 g2_4 YawRMSE 好 0.110。
```

这暗示 `g4` 可能带来一点全局 ATE 收益，但也可能引入 orientation / endpoint 代价。下一步需要拆解：

```text
g2 单层贡献是什么？
g3 单层贡献是什么？
g4 单层贡献是什么？
g2_3 为什么 balanced？
g2_4 为什么 ATE 略低但 final/yaw 变差？
是否存在 g2_3 + soft g4 的中间点？
```

### 1.4 问题 D：`full support` 对 `g2_3/g2_4` 是否仍然最优？

目前 `g3` 的 support sweep 说明 `full` 稳定强于 `off246/near12/near24`，但这个结论不能直接推广到 `g2_3/g2_4`。窗口 cue 可能对 temporal support 更敏感。

因此必须比较：

```text
full
near12
near24
off246
past_only
future_only
overlap_excluded
```

并判断：

```text
1. full 是否确实最强；
2. future support 是否贡献过大；
3. past_only 是否接近 full，从而更适合 causal / streaming；
4. overlap_excluded 是否降低边界污染；
5. support pattern 是否改变 rotation / final error trade-off。
```

### 1.5 问题 E：现有 cue quality 指标是否需要重标定？

当前 `g2_3/g2_4` 的 anchor collision 很高：

```text
g2_3.full anchor collision = 0.205
g2_4.full anchor collision = 0.199
```

按旧直觉，这应该很危险；但它们实际效果最好。这说明 `AnchorCollide` 不能再作为 hard reject gate。需要重新解释：

```text
1. C_anchor 本身是否不够准；
2. 高 anchor collision 是否只是 attention read suppression 与 write anchor 概念不一致；
3. 被 suppress 的 anchor-like 区域是否其实是 harmful support；
4. 质量指标哪些能预测 ATE / Rot / FinalErr，哪些只能做 diagnostic。
```

### 1.6 问题 F：什么时候才进入真正 cue combination？

只有当 `D_g_locked` 确认后，才进入组合验证。组合验证的目标不是把 cue 相加，而是验证不同信号是否互补：

```text
D_g: LoGeR 内部 global query manifold inconsistency
D_old: 几何 residual / old_dyn addclip
D_exp: explicit_dyn_only
R_static: deep high-sim / low-entropy / high-support static rescue
```

组合必须有 component attribution，否则只是盲调。

---

## 2. 核心假设

本阶段所有实验围绕下面的假设设计。每个假设都有对应的验证实验、记录指标和通过标准。

---

## H1：`g2_3.full` 与 `g2_4.full` 是真实强 cue，不是单次运行偶然

### 假设内容

`g2_3.full` 与 `g2_4.full` 的提升来自早中层 global query inconsistency 的真实信号，而不是一次 beta、一次 run 或一次评估波动。

### 实验设计

对三个 anchor 候选做 repeat：

```text
C3 = acl2.gg.qq.low.g3.full.headmean.robustq
C23 = acl2.gg.qq.low.g2_3.full.headmean.robustq
C24 = acl2.gg.qq.low.g2_4.full.headmean.robustq
```

重复运行：

```text
C3  hybrid beta=3.75, repeat 1 次
C23 hybrid beta=3.75, repeat 2 次
C24 hybrid beta=4.00, repeat 2 次
```

如果 pipeline 完全确定性，则 repeat 应 byte-level 或 metric-level 一致；如果存在轻微非确定性，则记录均值与标准差。

### 必须记录的指标

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
per_chunk_error.csv
run_config.yaml
hmc_state_hash.jsonl
runtime_summary.json
```

### 假设成立标准

H1 通过条件：

```text
1. C23 repeat ATE 标准差 <= 0.03m；
2. C24 repeat ATE 标准差 <= 0.03m；
3. C23/C24 的 mean ATE 均优于 C3 mean ATE 至少 0.03m；
4. C23 的 Rot / FinalErr / YawRMSE 优势在 repeat 后仍然存在；
5. 无 HMC commit hash 异常，无 failed chunks。
```

如果 `C24` repeat 后优势消失，而 `C23` 稳定，则主线切到 `C23`。如果两者都不稳定，回退到 `C3` 作为保守 anchor，并优先排查工程与运行差异。

---

## H2：`g2_3.full` 是当前更合理的主候选，`g2_4.full` 是 ATE-oriented 对照

### 假设内容

`g2_4.full` 的 ATE 最低，但优势只有 `0.0042m`，不能构成稳定主线依据。`g2_3.full` 在 rotation、final error、yaw 上明显更好，更适合作为后续组合和 intervention path 的 base cue。

### 实验设计

补齐 `g2_3.full` 的 beta sweep，并与 `g2_4.full` 做同 beta 对照。

建议 beta：

```text
C23 read-only:
3.25, 3.50, 3.75, 4.00, 4.25

C23 hybrid:
3.25, 3.50, 3.75, 4.00, 4.25

C24 read-only:
3.50, 3.75, 4.00, 4.25

C24 hybrid:
3.50, 3.75, 4.00, 4.25
```

每个 beta 计算：

$$
\Delta_{write}(\beta)=ATE_{read}(\beta)-ATE_{hybrid}(\beta)
$$

$$
\Delta_{rot}(\beta)=Rot_{candidate}(\beta)-Rot_{C3,hybrid,b3.75}
$$

$$
\Delta_{final}(\beta)=FinalErr_{candidate}(\beta)-FinalErr_{C3,hybrid,b3.75}
$$

### 必须记录的指标

除了全局 metrics，还要记录：

```text
write_gain_curve.csv
beta_curve_metrics.csv
trajectory_diagnostics.json
per_chunk_delta_vs_C3.csv
per_chunk_delta_vs_C23.csv
```

字段包含：

```text
beta
mode
commit_mode
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
DeltaWrite
DeltaATE_vs_C3
DeltaRot_vs_C3
DeltaFinal_vs_C3
```

### 假设成立标准

如果满足以下条件，则确认 `C23` 为主候选：

```text
1. C23 最佳 ATE 与 C24 最佳 ATE 差距 <= 0.05m；
2. C23 最佳 Rot 比 C24 最佳 Rot 好 >= 0.08deg；
3. C23 FinalErr 比 C24 好 >= 0.40m；
4. C23 YawRMSE 比 C24 好 >= 0.05；
5. C23 的 write_gain(beta) 在至少 3 个 beta 点为正。
```

如果 `C24` 在 repeat 和 beta sweep 后 ATE 优势扩大到 `>=0.10m`，且 FinalErr / Rot 不再明显差，则 `C24` 可升级为主线。否则将：

```text
D_g_balanced = C23
D_g_ate      = C24
D_g_anchor   = C3
```

---

## H3：`g4` 是一个需要软权重的 trade-off layer，而不是越多越好

### 假设内容

`g2_4` 相比 `g2_3` 引入的 `g4` 可能提供轻微 ATE 收益，但同时带来 rotation / endpoint 代价。直接平均 `g2:g4` 不是最优，可能存在更好的 soft-g4 权重。

### 实验设计

分三组：单层、窗口、加权窗口。

#### 组 A：单层对照

```text
g2.full
g3.full
g4.full
```

#### 组 B：窗口对照

```text
g1_3.full
g2_3.full
g3_4.full
g2_4.full
g2_5.full
```

#### 组 C：soft-g4 加权窗口

第一版采用 map-level interpolation，避免马上改 layer-statistic 聚合代码。

定义：

$$
D_{23+\lambda4}=\operatorname{clip}\left((1-\lambda)D_{23}+\lambda D_4,0,1\right)
$$

其中：

```text
lambda = 0.10, 0.25, 0.50, 0.75
```

如果 map-level interpolation 有信号，再实现 statistic-level weighted layer aggregation：

$$
q_{win}=\frac{\sum_l w_l q_l}{\sum_l w_l}
$$

然后重新计算 low-similarity cue。

### 运行协议

第一轮只做 read-only：

```text
mode = read-only
commit = probe_native
beta = best beta from H2 for C23, plus beta=3.75 as anchor
```

通过 gate 后再做 hybrid：

```text
mode = hybrid-safe
commit = probe_ttt_write
beta = same beta as read-only best
```

### 必须记录的指标

```text
layer_window_metrics.csv
weighted_g4_metrics.csv
per_chunk_delta_vs_C23.csv
cue_quality_summary.json
cue_quality_per_chunk.jsonl
```

额外记录：

```text
D23_mass
D4_mass
D23_D4_corr
D23_only_mass
D4_only_mass
D23_and_D4_mass
AnchorCollide_D23_only
AnchorCollide_D4_only
```

定义：

$$
Q_{23,4}^{11}=\mathbf{1}[D_{23}>\tau]\mathbf{1}[D_4>\tau]
$$

$$
Q_{23,4}^{10}=\mathbf{1}[D_{23}>\tau]\mathbf{1}[D_4\le\tau]
$$

$$
Q_{23,4}^{01}=\mathbf{1}[D_{23}\le\tau]\mathbf{1}[D_4>\tau]
$$

默认 $\tau=0.5$。

### 假设成立标准

H3 成立的标准是：

```text
1. g4 单层或 g3_4 窗口显示 ATE 信号，但 Rot/FinalErr 明显差于 g2_3；
2. g2_3 + soft g4 存在一个 lambda，使 ATE 接近或优于 g2_4，同时 Rot/FinalErr 接近 g2_3；
3. D4-only 区域集中解释 g2_4 的 ATE gain 或 rotation loss；
4. weighted candidate 的 improvement 不是单纯 mass shift。
```

promotion 标准：

```text
strong:
ATE <= 38.35 且 Rot <= 8.80 且 FinalErr <= 4.00

balanced:
ATE <= 38.40 且 Rot <= 8.75 且 FinalErr <= 3.80
```

如果所有 soft-g4 都不如 `C23`，后续组合 base 使用 `C23`，`C24` 只保留为 ATE diagnostic。

---

## H4：`full support` 在 `g2_3/g2_4` 上未必最优，必须重新验证

### 假设内容

`g3.full` 的 support sweep 不能替代 `g2_3/g2_4` 的 support sweep。窗口 cue 可能受 temporal support 影响更大，尤其是 future support、overlap support 和 local support。

### 实验设计

对两个候选分别扫 support：

```text
D_g_balanced = g2_3
D_g_ate      = g2_4
```

support 候选：

```text
full
off246
near12
near24
past_only
future_only
overlap_excluded
```

第一轮固定 beta：

```text
g2_3: beta = best from H2, default 3.75
g2_4: beta = best from H2, default 4.00
```

先跑 read-only。只有 read-only 满足 gate 的 support 才进入 hybrid-safe。

### 必须记录的指标

```text
support_sweep_metrics.csv
support_cue_quality.csv
support_temporal_profile.jsonl
support_attention_effect.jsonl
```

关键字段：

```text
support_name
support_frame_count_mean
support_frame_count_p10/p90
mean_temporal_span
past_mass
future_mass
overlap_mass
same_chunk_support_mass
D_mass
AnchorCollide
Frag
TempCons
SupportConc
ATE
Rot
FinalErr
YawRMSE
```

定义 support concentration：

$$
SupportConc=\frac{1}{T}\sum_t \max_s w_{t,s}
$$

其中 $w_{t,s}$ 是 support aggregation 中第 $s$ 个支持帧的归一化权重。如果当前实现只做均值，则可以先用候选 support 数量和有效帧分布近似。

### 假设成立标准

H4 成立的标准不是某个 support 一定超过 full，而是能清楚回答 support 依赖：

```text
1. 如果 full 仍最好，确认 g2_3/g2_4 需要全 chunk temporal context；
2. 如果 past_only 接近 full，说明 cue 更适合 streaming/causal 版本；
3. 如果 future_only 明显优于 past_only，要警惕它依赖非因果 chunk 内未来信息；
4. 如果 overlap_excluded 改善 FinalErr 或 boundary metrics，说明 overlap 附近可能有污染；
5. 如果 near12/near24 改善 Rot 但 ATE 略回退，可保留为 balanced variant。
```

support 晋级标准：

```text
read-only ATE <= corresponding full support + 0.05m
且 Rot 或 FinalErr 明显更好

或

read-only ATE 优于 full support >= 0.05m
```

hybrid 晋级标准：

```text
hybrid ATE <= 38.35
或 hybrid ATE <= 38.43 且 Rot/FinalErr 明显优于 C23/C24 full
```

---

## H5：当前 cue quality 指标不能作为 hard ranking，需要重标定

### 假设内容

`AnchorCollide`、`Frag`、`Corr old_dyn` 等指标有 diagnostic value，但目前不能直接预测 ATE。尤其 `g2_3/g2_4` 的高 anchor collision 与强指标表现冲突，说明旧质量解释需要重标定。

### 实验设计

对所有候选导出 passive cue maps：

```text
C3
C23
C24
best soft-g4 candidate
best support variant
old_dyn_addclip
explicit_dyn_only
deep high-sim rescue cue
C_anchor
confidence map
```

统一输出 chunk-level 和 frame-level cue quality，并与 error delta 对齐。

### 必须记录的指标

#### cue 统计

```text
mean_D
p50_D
p90_D
mass_D_gt_0.5
mass_D_gt_0.8
coverage
fragmentation
anchor_collision
confidence_correlation
old_dyn_correlation
explicit_correlation
C_anchor_correlation
image_zone_mass
TempCons
```

#### image-zone 分布

将 patch 网格按图像区域分为：

```text
upper_sky_horizon
middle_structure
lower_road
left_border
right_border
center
```

记录：

```text
ZoneMass_upper
ZoneMass_middle
ZoneMass_lower
ZoneMass_left
ZoneMass_right
ZoneMass_center
```

#### 与误差的相关性

对每个候选和每个 chunk 计算：

$$
\Delta ATE_c = ATE_{candidate,c}-ATE_{baseline,c}
$$

$$
\Delta Rot_c = Rot_{candidate,c}-Rot_{baseline,c}
$$

$$
\Delta Final_c = FinalErr_{candidate,c}-FinalErr_{baseline,c}
$$

然后计算：

```text
Corr(mean_D, DeltaATE)
Corr(AnchorCollide, DeltaATE)
Corr(Frag, DeltaATE)
Corr(ZoneMass_lower, DeltaATE)
Corr(ZoneMass_upper, DeltaRot)
Corr(D4_only_mass, DeltaFinal)
```

baseline 用：

```text
primary baseline: C3 hybrid b3.75
secondary baseline: C23 hybrid best
```

### 可视化要求

必须输出：

```text
quality_error_scatter/
    meanD_vs_deltaATE.png
    anchorCollision_vs_deltaATE.png
    frag_vs_deltaATE.png
    zoneLower_vs_deltaATE.png
    D4only_vs_deltaFinal.png

cue_quality_heatmap.md
```

每张 scatter 要标出：

```text
largest improvement chunks
largest regression chunks
worst native chunks
boundary chunks
```

### 假设成立标准

H5 成立的实用标准：

```text
1. 找到至少一个 quality 指标与 DeltaATE 或 DeltaRot 有稳定相关性，|corr| >= 0.30；
2. 证明 AnchorCollide 不能作为 hard reject，因为高 AnchorCollide 候选仍可改善 ATE/Rot；
3. 能解释 g2_3 与 g2_4 的差异至少落在哪些 chunk / zone / disagreement 区域；
4. 能给 Stage 7 cue combination 提供 routing 或 rescue 的依据。
```

如果没有任何 quality 指标能解释性能差异，则后续组合必须更保守，只允许极少数候选进入 full controlled。

---

## H6：真正的 cue combination 尚未开始，必须在 `D_g_locked` 后做小矩阵 routing

### 假设内容

当前结果仍然是单一 global query cue。`D_g` 与 `old_dyn / explicit_dyn` 来源不同，可能互补，但简单 add/clip 可能扩大 false positive。更合理的方式是 routing。

### 前置条件

只有当 Stage 1-5 锁定 `D_g_locked` 后，才能进入组合。

`D_g_locked` 选择规则见第 8 节。

### cue bank

组合阶段使用以下标准 cue：

```text
D_g        = locked global query cue
D_old      = old_dyn_addclip
D_exp      = explicit_dyn_only
R_static   = deep high-sim / low entropy / high support rescue
R_anchor   = C_anchor
R_conf     = confidence / low uncertainty
```

所有 cue 必须投影到同一 patch grid：

$$
D \in [0,1]^{T\times H_{tok}\times W_{tok}}
$$

### 组合族 A：`D_g + D_old` routing

定义 quadrant：

$$
Q_{11}=\mathbf{1}[D_g>\tau_g]\mathbf{1}[D_{old}>\tau_o]
$$

$$
Q_{10}=\mathbf{1}[D_g>\tau_g]\mathbf{1}[D_{old}\le\tau_o]
$$

$$
Q_{01}=\mathbf{1}[D_g\le\tau_g]\mathbf{1}[D_{old}>\tau_o]
$$

其中默认：

```text
tau_g = 0.5
tau_o = 0.5
```

routing cue：

$$
D_{route}=Q_{11}\max(D_g,D_{old})+Q_{10}\lambda_gD_g+Q_{01}\lambda_oD_{old}
$$

参数：

```text
lambda_g = 0.75, 1.00
lambda_o = 0.25, 0.50, 0.75
```

设计解释：

```text
agreement 区域强 suppress；
D_g-only 区域保留较高权重，因为 D_g 当前更强；
D_old-only 区域默认降权，因为 old_dyn 可能包含几何 residual false positive。
```

### 组合族 B：`D_g + D_exp` routing

使用 `explicit_dyn_only` 替代 `old_dyn`：

$$
D_{route\_exp}=Q_{11}^{g,e}\max(D_g,D_{exp})+Q_{10}^{g,e}\lambda_gD_g+Q_{01}^{g,e}\lambda_eD_{exp}
$$

参数：

```text
lambda_g = 0.75, 1.00
lambda_e = 0.50, 0.75, 1.00
```

理由：`explicit_dyn_only` 的 ATE 虽不如当前 attention cue，但 rotation 比 old_dyn 系更好，可能作为 cleaner geometry cue。

### 组合族 C：static rescue

对主 suppress cue 加 static rescue：

$$
D_{rescue}=\operatorname{clip}\left(D_{main}(1-\alpha R_{static}),0,1\right)
$$

或 anchor-gated rescue：

$$
D_{rescue\_anchor}=\operatorname{clip}\left(D_{main}(1-\alpha R_{static}R_{anchor}),0,1\right)
$$

参数：

```text
alpha = 0.25, 0.50, 0.75
```

### 实验流程

先 passive audit，再 read-only，再 hybrid。

#### Passive audit 通过条件

```text
0.08 <= mean(D>0.5) <= 0.35
coverage >= 0.95
Frag <= 0.20
Q11_mass >= 0.02
Q10_mass <= 0.30
Q01_mass <= 0.30
AnchorCollide 不作为 hard reject，但必须记录并解释
```

#### Read-only full gate

```text
mode = read-only
commit = probe_native
beta = beta_locked - 0.25, beta_locked, beta_locked + 0.25
```

read-only 晋级条件：

```text
ATE <= D_g_locked read-only best - 0.05m
或 Rot 改善 >= 0.15deg 且 ATE 回退 <= 0.05m
或 FinalErr 改善 >= 0.30m 且 ATE 回退 <= 0.05m
```

#### Hybrid full gate

```text
mode = hybrid-safe
commit = probe_ttt_write
beta = best read-only beta
```

hybrid 强成功：

```text
ATE <= 38.25
Rot <= 8.90
FinalErr <= 4.20
```

hybrid 弱成功：

```text
ATE <= 38.35
且 Rot/FinalErr 明显优于 D_g_locked
```

如果 ATE 只比 `38.38` 好 `0.02m`，但没有 Rot/FinalErr 改善，不算实质成功。

### component attribution

任何通过 hybrid gate 的组合必须补：

```text
D_g only, same beta
aux cue only, same beta
simple add/clip, same beta
routing without rescue, same beta
routing with rescue, same beta
mass-matched D_g only
```

组合被认定成功的条件：

```text
1. 优于所有单分量；
2. 优于 simple add/clip；
3. 不是单纯 mass 下降造成；
4. trajectory / segment / final error 至少两类同步改善；
5. write_gain 同 beta 仍为正。
```

---

## H7：intervention path 仍是未验证变量，应在 `D_g_locked` 后小规模测试

### 假设内容

当前使用方式是：

```text
cue -> frame-attention pair bias
```

还没有系统验证：

```text
cue -> frame-attention residual gate
cue -> SWA source/residual gate
cue -> chunk/global residual gate
```

由于 LoGeR 中 SWA 是相邻 chunk 的 lossless local memory，TTT 是长程 compressed global memory，SWA gate 更可能影响 boundary / local alignment，而不一定大幅降低全局 ATE。

### 前置条件

只用 `D_g_locked` 做 intervention path，不再用旧 `g3` 大规模测试。

### 实验组

#### 组 A：Frame-attention bias baseline

```text
FA-BIAS:
path = frame_attention
mode = pair bias
cue = D_g_locked
beta = beta_locked
commit = probe_native / probe_ttt_write
```

#### 组 B：Frame-attention residual gate

定义：

$$
H_i^{out}=H_i^{in}+g_i\Delta_i^{FA}
$$

$$
g_i=\operatorname{clip}(1-\rho D_i,g_{min},1)
$$

参数：

```text
rho = 0.10, 0.20, 0.30
g_min = 0.50, 0.70
```

#### 组 C：SWA previous-source soft gate

source-side SWA gate：

$$
\Delta_i^{SWA}=\sum_j a_{ij}^{SWA}g_j^{src}V_j
$$

$$
g_j^{src}=\operatorname{clip}(1-\rho D_j,g_{min},1)
$$

只对 previous chunk source 启用：

```text
source = previous_only
layer = first_swa_only
rho = 0.05, 0.10, 0.20
g_min = 0.70, 0.85
```

#### 组 D：FA bias + SWA soft gate

只有 SWA 单独不崩时才组合：

```text
FA bias beta = beta_locked 或 beta_locked - 0.25
SWA rho = best safe rho from group C
commit = probe_ttt_write
```

### 需要额外记录的 local / boundary 指标

SWA gate 必须记录：

```text
chunk_boundary_pose_jump
boundary_10f_ATE
boundary_20f_ATE
overlap_pointmap_residual
overlap_pose_discontinuity
cross_chunk_attention_mass
attention_mass_to_highD_previous_source
attention_mass_to_anchor_previous_source
swa_residual_norm_before
swa_residual_norm_after
swa_gate_mean
swa_gate_p10/p50/p90
```

定义一个边界 pose jump：

$$
E_{bdry}(m)=\left\|\log\left(T_{m,end}^{-1}T_{m+1,start}\right)\right\|
$$

如果有 overlap frame，则记录 overlap pointmap residual：

$$
E_{overlap}(m)=\operatorname{Mean}_p\left\|X_{m,k}(p)-X_{m+1,k}(p)\right\|_2
$$

### 判断标准

intervention path 成功不只看 ATE。

FA residual gate 成功：

```text
ATE <= D_g_locked read-only + 0.05m
且 Rot 或 FinalErr 改善明显
```

SWA gate 成功：

```text
boundary_10f_ATE 改善 >= 3%
或 overlap_pointmap_residual 改善 >= 5%
且 global ATE 回退 <= 0.08m
```

FA bias + SWA gate 成功：

```text
hybrid ATE <= D_g_locked hybrid best
且 boundary / FinalErr 至少一项明显更好
```

如果 SWA gate 只改善 local 指标但 ATE 略回退，可以保留为 balanced/local-alignment branch，不作为主 ATE best。

---

## H8：必须做小规模 cross-sequence sanity，防止 KITTI01 过拟合

### 假设内容

KITTI01 是开发集，已经进行了大量 beta、layer、support 调参。当前 `g2_3` 与 `g2_4` 的 ATE 差距只有厘米级，继续只看 KITTI01 容易过拟合。

### 进入条件

满足以下任一条件的候选进入 cross-sequence：

```text
1. KITTI01 hybrid ATE <= 38.35；
2. KITTI01 hybrid ATE <= 38.43 且 Rot/FinalErr 明显优于 current best；
3. 作为 anchor baseline 必须进入：C3, C23, C24。
```

最多进入 5 个候选：

```text
C3 anchor
g2_3 best
g2_4 best
best weighted g4
best combination or rescue candidate
```

### 数据集

```text
KITTI00
KITTI02
KITTI05 或 KITTI08
```

### 评价指标

```text
per-sequence ATE
per-sequence Rot
per-sequence FinalErr
per-sequence ATE_50/100/200
average ATE
average Rot
relative degradation count
```

### 判断标准

通过 sanity：

```text
1. 平均 ATE 优于 C3 anchor；
2. 没有任何序列 ATE 恶化超过 5%；
3. 平均 Rot 不恶化超过 3%；
4. 至少一个长序列显示稳定 gain；
5. 如果 g2_3 比 g2_4 更稳，则主线优先 g2_3。
```

如果候选只在 KITTI01 提升，在其他序列系统性变差，则标记为：

```text
KITTI01-specific, not promoted
```

---

## 3. 固定实验协议

### 3.1 HMC commit 协议

所有主实验只允许两种 commit：

```text
read-only:
    output = controlled forward
    commit = probe_native

hybrid-safe:
    output = controlled forward
    commit = probe_ttt_write
```

`controlled commit` 只作为 diagnostic，不作为主结论。原因是此前 HMC v2 已经证明 controlled read forward 的 TTT side effect 不能直接提交到未来 memory。

### 3.2 read path 固定

除非明确做 intervention path ablation，否则固定：

```text
read path = frame attention early layers
bias mode = pair
normalization = robustq
head aggregation = headmean
```

### 3.3 不引入 external motion cue

本阶段不做 RAFT / GMFlow / external optical flow。所有 cue 来自 LoGeR 内部 attention / existing geometry cue / memory trace。

### 3.4 Baseline 必须随批次报告

每批 summary 表必须包含：

| Baseline | 作用 | 当前参考值 |
|---|---|---:|
| `native LoGeR` | no-control anchor | `41.7502 / 8.9928` |
| `old_dyn_addclip` | 旧 Phase F best | `39.3103 / 9.7097` |
| `gg.qq.middle.low + probe_ttt_write` | ACL v1 anchor | `38.9714 / 9.2084` |
| `C3 = g3.full + probe_ttt_write` | ACL2 previous anchor | `38.4298 / 8.9846` |
| `C24 = g2_4.full + probe_ttt_write` | current ATE best | `38.3805 / 8.8707` |
| `C23 = g2_3.full + probe_ttt_write` | current Pareto candidate | `38.3847 / 8.7583` |

---

## 4. 必须记录的指标

本阶段不能只记录 ATE/Rot。每个 run 必须保存下面几类指标。

### 4.1 全局 trajectory 指标

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

### 4.2 per-frame / per-chunk 指标

输出：

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

用于判断：

```text
1. 改善是否集中在少数 chunk；
2. 是否新增 worst chunk；
3. g2_3 的 endpoint 优势来自哪些 chunk；
4. g2_4 的 ATE 优势是否以某些 late chunks 为主。
```

### 4.3 cue quality 指标

每个 cue 必须记录：

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
zone_upper_mass
zone_middle_mass
zone_lower_mass
zone_center_mass
zone_border_mass
temporal_consistency
support_concentration
```

fragmentation 定义：

$$
Frag(D)=\frac{1}{THW}\sum_{t,h,w}\mathbf{1}\left[|D_{t,h,w}-D_{t,h,w+1}|>\tau_f\right]+\mathbf{1}\left[|D_{t,h,w}-D_{t,h+1,w}|>\tau_f\right]
$$

temporal consistency 定义：

$$
TempCons(D)=1-\frac{1}{T-1}\sum_{t=1}^{T-1}\operatorname{Mean}_{h,w}|D_{t+1,h,w}-D_{t,h,w}|
$$

### 4.4 read effect 指标

每个 controlled run 保存：

```text
read_effect_summary.jsonl
```

字段：

```text
chunk_id
layer_id
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

关键差值：

$$
\Delta Attn_{highD}=Mass_{after}(D>\tau)-Mass_{before}(D>\tau)
$$

正常 read suppression 应满足：

```text
attn_mass_to_highD_after < attn_mass_to_highD_before
```

但不应显著降低 anchor attention：

```text
attn_mass_to_anchor_after 不应大幅小于 before
```

### 4.5 TTT / memory 指标

hybrid run 必须保存：

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
update_norm_native
update_norm_probe_write
update_cosine_to_native
memory_state_rel_diff
memory_state_hash_before
memory_state_hash_after_probe
memory_state_hash_after_commit
branch0_update_norm
branch1_update_norm
branch2_update_norm
```

定义：

$$
UpdateCos=\frac{\langle \Delta W_{candidate},\Delta W_{native}\rangle}{\|\Delta W_{candidate}\|\|\Delta W_{native}\|+\epsilon}
$$

$$
MemDiff=\frac{\|W_{after}-W_{probe}\|_F}{\|W_{probe}\|_F+\epsilon}
$$

### 4.6 boundary / local 指标

用于 SWA 或 support ablation，也建议常规记录：

```text
chunk_boundary_pose_jump
boundary_10f_ATE
boundary_20f_ATE
overlap_pointmap_residual
overlap_pose_discontinuity
```

---

## 5. 必须可视化的内容

每个晋级候选必须生成可视化，不允许只有表格。

### 5.1 Beta curve

对 `C23/C24/C3` 输出：

```text
beta vs ATE
beta vs Rot
beta vs FinalErr
beta vs YawRMSE
beta vs write_gain
```

read-only 和 hybrid-safe 用同一张图展示。

### 5.2 Layer/window heatmap

横轴：layer/window 候选  
纵轴：metrics

```text
ATE
Rot
FinalErr
YawRMSE
mean_D
AnchorCollide
Frag
```

必须能直观看到：

```text
g2, g3, g4, g2_3, g2_4 的 trade-off
```

### 5.3 Support heatmap

横轴：support pattern  
纵轴：candidate (`C23`, `C24`)  
格子：ATE / Rot / FinalErr / support_concentration

### 5.4 Cue map dashboard

固定帧集合：

```text
每 100 帧采样 1 帧
每个 worst chunk 采样 2 帧
每个 largest-gain chunk 采样 2 帧
每个 largest-regression chunk 采样 2 帧
```

每张 dashboard 包含：

```text
RGB
D_g3
D_g2_3
D_g2_4
D_g4
old_dyn_addclip
explicit_dyn_only
C_anchor
confidence
candidate suppression overlay
```

### 5.5 g2_3 vs g2_4 差异图

必须输出：

```text
D_g2_3
D_g2_4
D_g2_4 - D_g2_3
D4-only region
D23-only region
anchor collision overlay
```

重点看：

```text
g2_4 多 suppress 的区域是否对应 rotation/final error 变差。
```

### 5.6 Trajectory comparison

每个晋级候选输出：

```text
GT trajectory
native LoGeR
old_dyn_addclip
C3 g3.full
C23 g2_3.full
C24 g2_4.full
candidate
```

至少三张图：

```text
full trajectory XY
first half trajectory XY
second half trajectory XY
```

标记：

```text
worst chunks
largest improvement chunks
largest regression chunks
```

### 5.7 Error-over-time

输出：

```text
per-frame translation error
per-frame rotation error
sliding 50f ATE
sliding 100f ATE
sliding 200f ATE
cumulative yaw drift proxy
```

### 5.8 Quality-error scatter

输出：

```text
meanD_vs_deltaATE
AnchorCollide_vs_deltaATE
Frag_vs_deltaATE
D4onlyMass_vs_deltaFinal
ZoneLowerMass_vs_deltaATE
ZoneUpperMass_vs_deltaRot
```

### 5.9 Combination quadrant maps

进入组合阶段后必须输出：

```text
Q11: D_g high, aux high
Q10: D_g high, aux low
Q01: D_g low, aux high
Q00: both low
static rescue overlay
anchor collision overlay
```

---

## 6. 实验阶段与运行顺序

---

## Stage 0：工程锁定与 baseline repeat

### 目标

确认当前代码、runner、hook、metric parser 没有变化，避免后续小幅改进被工程波动污染。

### 运行列表

| Run ID | Cue | Mode | Commit | Beta | 目的 |
|---|---|---|---|---:|---|
| V4_S0_01 | none | native | no-control | n/a | native LoGeR anchor |
| V4_S0_02 | `g3.full` | hybrid | `probe_ttt_write` | 3.75 | previous ACL2 anchor |
| V4_S0_03 | `g2_3.full` | hybrid | `probe_ttt_write` | 3.75 | Pareto candidate repeat |
| V4_S0_04 | `g2_3.full` | hybrid | `probe_ttt_write` | 3.75 | repeat |
| V4_S0_05 | `g2_4.full` | hybrid | `probe_ttt_write` | 4.00 | ATE candidate repeat |
| V4_S0_06 | `g2_4.full` | hybrid | `probe_ttt_write` | 4.00 | repeat |

### 通过标准

```text
C3 repeat ATE within 0.03m of 38.4298
C23 repeat ATE within 0.03m of 38.3847
C24 repeat ATE within 0.03m of 38.3805
no failed chunks
HMC identity / state hash 正常
```

如果 S0 不通过，不进入后续实验。

---

## Stage 1：`g2_3/g2_4` 同 beta read-only / hybrid-safe 因果确认

### 目标

拆清 read correction 与 safe write 的贡献，判断 `g2_3/g2_4` 是否继承 `g3` 上的 stable write gain。

### 运行列表

| Candidate | Mode | Commit | Beta |
|---|---|---|---:|
| `g2_3.full` | read-only | `probe_native` | 3.25 |
| `g2_3.full` | hybrid | `probe_ttt_write` | 3.25 |
| `g2_3.full` | read-only | `probe_native` | 3.50 |
| `g2_3.full` | hybrid | `probe_ttt_write` | 3.50 |
| `g2_3.full` | read-only | `probe_native` | 3.75 |
| `g2_3.full` | hybrid | `probe_ttt_write` | 3.75 |
| `g2_3.full` | read-only | `probe_native` | 4.00 |
| `g2_3.full` | hybrid | `probe_ttt_write` | 4.00 |
| `g2_3.full` | read-only | `probe_native` | 4.25 |
| `g2_3.full` | hybrid | `probe_ttt_write` | 4.25 |
| `g2_4.full` | read-only | `probe_native` | 3.50 |
| `g2_4.full` | hybrid | `probe_ttt_write` | 3.50 |
| `g2_4.full` | read-only | `probe_native` | 3.75 |
| `g2_4.full` | hybrid | `probe_ttt_write` | 3.75 |
| `g2_4.full` | read-only | `probe_native` | 4.00 |
| `g2_4.full` | hybrid | `probe_ttt_write` | 4.00 |
| `g2_4.full` | read-only | `probe_native` | 4.25 |
| `g2_4.full` | hybrid | `probe_ttt_write` | 4.25 |

### 输出

```text
stage1_beta_write_gain.md
stage1_beta_write_gain.csv
stage1_beta_curves/
```

### 判断

如果 `g2_3` 在 beta sweep 后达到：

```text
ATE <= 38.38
Rot <= 8.80
FinalErr <= 4.00
```

则 `g2_3` 正式作为主候选。

如果 `g2_4` ATE 优势扩大到：

```text
C24_best_ATE <= C23_best_ATE - 0.10m
且 Rot / FinalErr 不明显更差
```

则 `g2_4` 升级为主候选。否则 `g2_4` 保留为 ATE-oriented 对照。

---

## Stage 2：layer decomposition 与 soft-g4 实验

### 目标

解释 `g2_3` 与 `g2_4` 的差异，确定是否存在更好的 weighted window。

### 第一轮 read-only

| Run group | Cue | Beta | Mode |
|---|---|---:|---|
| single | `g2.full` | `beta_C23_best` | read-only |
| single | `g3.full` | `beta_C23_best` | read-only |
| single | `g4.full` | `beta_C23_best` | read-only |
| window | `g1_3.full` | `beta_C23_best` | read-only |
| window | `g2_3.full` | `beta_C23_best` | read-only |
| window | `g3_4.full` | `beta_C23_best` | read-only |
| window | `g2_4.full` | `beta_C23_best` | read-only |
| window | `g2_5.full` | `beta_C23_best` | read-only |
| weighted | `g2_3 + 0.10*g4` | `beta_C23_best` | read-only |
| weighted | `g2_3 + 0.25*g4` | `beta_C23_best` | read-only |
| weighted | `g2_3 + 0.50*g4` | `beta_C23_best` | read-only |
| weighted | `g2_3 + 0.75*g4` | `beta_C23_best` | read-only |

### 第二轮 hybrid

只让 read-only 通过 gate 的候选进入 hybrid：

```text
read-only ATE <= C23_read_best + 0.05m
或 Rot/FinalErr 明显更好且 ATE 回退 <= 0.08m
```

### 输出

```text
stage2_layer_window_table.md
stage2_weighted_g4_table.md
stage2_layer_heatmap.png
stage2_g4_disagreement_dashboard.md
```

### 判断

如果 weighted-g4 找到候选：

```text
ATE <= 38.35
Rot <= 8.80
FinalErr <= 4.00
```

则 `D_g_locked` 可能改为 weighted-g4。否则保留 `C23` 或 `C24`。

---

## Stage 3：support sweep for `g2_3/g2_4`

### 目标

验证 support pattern 是否影响早中层 cue 的效果与泛化。

### 第一轮 read-only

| Candidate | Support |
|---|---|
| `g2_3` | `full` |
| `g2_3` | `off246` |
| `g2_3` | `near12` |
| `g2_3` | `near24` |
| `g2_3` | `past_only` |
| `g2_3` | `future_only` |
| `g2_3` | `overlap_excluded` |
| `g2_4` | `full` |
| `g2_4` | `off246` |
| `g2_4` | `near12` |
| `g2_4` | `near24` |
| `g2_4` | `past_only` |
| `g2_4` | `future_only` |
| `g2_4` | `overlap_excluded` |

beta 使用 Stage 1 的 best beta。

### 第二轮 hybrid

只对满足 gate 的 support 跑 hybrid。

### 输出

```text
stage3_support_sweep.md
stage3_support_heatmap.png
stage3_support_temporal_profile.csv
```

### 判断

如果 `past_only` 接近 full：

```text
ATE gap <= 0.05m
Rot/FinalErr 不差
```

则保留 `past_only` 为 causal-friendly variant。

如果 `overlap_excluded` 改善 boundary / final：

```text
FinalErr 改善 >= 0.30m
或 boundary_10f_ATE 改善 >= 3%
```

则后续 SWA / residual gate 优先围绕 overlap / boundary 做。

---

## Stage 4：cue quality 重标定与 failure gallery

### 目标

解释为什么高 anchor collision 的 `g2_3/g2_4` 反而有效，并为后续 routing 提供依据。

### 输入候选

```text
C3
C23
C24
best weighted-g4 candidate
best support candidate
old_dyn_addclip
explicit_dyn_only
deep high-sim candidate
C_anchor
confidence
```

### 输出文件

```text
stage4_cue_quality_registry.csv
stage4_quality_error_correlation.md
stage4_failure_gallery.md
visuals/stage4_cue_dashboard/
visuals/stage4_scatter/
```

### 必须回答的问题

```text
1. C23 比 C24 少 suppress 了哪些区域？
2. C24 多出来的 suppress 是否集中在 late / boundary / horizon / road？
3. AnchorCollide 高的区域是否真的是长期 static anchor，还是 read-path harmful support？
4. 哪些质量指标和 DeltaATE / DeltaRot / DeltaFinal 有相关性？
5. old_dyn-only 区域和 D_g-only 区域分别长什么样？
```

### 判断

如果 Stage 4 能解释 C23/C24 差异，并识别出 old_dyn false positive 或 D_g-only hidden interference，进入 Stage 5 combination。否则先不要组合，继续 per-head / attention-map 挖掘。

---

## Stage 5：锁定 `D_g_locked`

### 目标

给后续所有组合、residual/SWA gate、cross-sequence 提供唯一主 base cue，避免每条线使用不同 base 导致归因混乱。

### 决策表

候选包括：

```text
C23 best beta/support
C24 best beta/support
best weighted-g4
best support variant
C3 anchor
```

综合评分不直接替代判断，但用于排序：

$$
Score = ATE + 0.20\cdot Rot + 0.05\cdot FinalErr + 0.10\cdot YawRMSE
$$

其中每项可以先做相对归一化，避免单位影响过大。最终由表格和可视化共同决定。

### 锁定标准

主线优先级：

```text
1. 如果某候选 ATE 明显最好，差距 >= 0.10m，且 Rot/FinalErr 不差，则锁定它；
2. 如果 ATE 差距 <= 0.05m，则优先 Rot/FinalErr/Yaw 更好的候选；
3. 如果某候选跨 support / beta 更稳定，优先稳定候选；
4. 如果 C23 与 C24 难以取舍，则：
   D_g_locked = C23
   D_g_ate_shadow = C24
```

预期最可能结果：

```text
D_g_locked = g2_3.full 或 g2_3 + soft-g4
D_g_ate_shadow = g2_4.full
```

---

## Stage 6：小规模 cue combination validation

### 目标

正式验证 cue 组合，而不是继续单 cue mining。

### 候选组合

第一批最多 8 个：

```text
D_g + D_old routing, lambda_o=0.25
D_g + D_old routing, lambda_o=0.50
D_g + D_exp routing, lambda_e=0.50
D_g + D_exp routing, lambda_e=0.75
D_g + R_static rescue, alpha=0.25
D_g + R_static rescue, alpha=0.50
D_route_old + R_static rescue
D_route_exp + R_static rescue
```

### 实验流程

1. passive audit；
2. read-only full at beta locked；
3. read-only local beta if promising；
4. hybrid same beta；
5. component attribution。

### 输出

```text
stage6_combination_passive_audit.md
stage6_combination_readonly_table.md
stage6_combination_hybrid_table.md
stage6_component_attribution.md
visuals/stage6_quadrant_maps/
```

### 判断

组合成功强标准：

```text
hybrid ATE <= 38.25
Rot <= 8.90
FinalErr <= 4.20
并且优于所有单分量
```

组合成功弱标准：

```text
hybrid ATE <= 38.35
且 Rot 或 FinalErr 明显优于 D_g_locked
```

如果所有组合都不超过 `D_g_locked`，结论不是失败，而是：

```text
当前最强仍是 single internal attention cue；下一步应转 per-head / attention-map / layer-evolution，而不是继续 routing 大矩阵。
```

---

## Stage 7：intervention path 小矩阵

### 目标

验证 `D_g_locked` 是否只适合 attention routing，还是也适合 residual reliability / SWA local-memory gating。

### 运行矩阵

| Run group | Path | Parameters | Commit |
|---|---|---|---|
| FA-BIAS | frame attention pair bias | beta locked | probe_native / probe_ttt_write |
| FA-RG | frame attention residual gate | rho=0.10/0.20, g_min=0.70 | probe_native |
| SWA-SRC | previous-source SWA gate | rho=0.05/0.10/0.20, g_min=0.85/0.70 | probe_native |
| FA+SWA | FA bias + best SWA gate | beta locked, rho best | probe_ttt_write |

### 判断

不要只用 ATE 判断 SWA gate。SWA gate 的成功条件包括：

```text
boundary_10f_ATE 改善 >= 3%
overlap_pointmap_residual 改善 >= 5%
FinalErr 改善 >= 0.30m
且 ATE 回退 <= 0.08m
```

如果 SWA gate 改善 local / boundary，但 ATE 不降，保留为 balanced branch。

---

## Stage 8：cross-sequence sanity

### 目标

防止 KITTI01 过拟合。

### 候选

最多 5 个：

```text
C3 anchor
D_g_locked
D_g_ate_shadow
best combination
best intervention-path variant
```

### 序列

```text
KITTI00
KITTI02
KITTI05 或 KITTI08
```

### 判断

通过：

```text
平均 ATE 优于 C3 anchor
没有序列 ATE 恶化超过 5%
平均 Rot 不恶化超过 3%
```

不通过：

```text
只在 KITTI01 提升，其它序列系统性恶化
```

---

## 7. 输出目录规范

建议：

```text
results/kitti01_hmc_v2/acl2_v4_earlymid_validation/
```

每个 run 目录必须包含：

```text
run_config.yaml
metrics_global.json
kitti_benchmark.log
01.txt
per_frame_error.csv
per_chunk_error.csv
cue_quality_summary.json
cue_quality_per_chunk.jsonl
read_effect_summary.jsonl
ttt_write_summary.jsonl
memory_delta_summary.jsonl
hmc_state_hash.jsonl
runtime_summary.json
visual_index.md
```

总表：

```text
acl2_v4_run_registry.csv
acl2_v4_promotion_table.md
acl2_v4_decision_log.md
acl2_v4_failure_gallery.md
```

`run_registry.csv` 字段：

```text
run_id
stage
cue_name
cue_family
layer_window
support
head_agg
normalization
mode
commit_mode
beta
rho
g_min
write_score_source
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
mean_D
mass_D_gt_0.5
AnchorCollide
Frag
TempCons
old_dyn_corr
explicit_corr
C_anchor_corr
write_gain
promotion_status
notes
```

---

## 8. 最终 promotion / stopping rules

### 8.1 single-cue promotion

单 cue 被认为真正优于当前 best，需要满足：

```text
强提升：
ATE <= 38.30
且 Rot <= 8.90
且 FinalErr <= 4.50

或 balanced 提升：
ATE <= 38.40
且 Rot <= 8.75
且 FinalErr <= 3.80
```

若只是：

```text
ATE 改善 < 0.05m
且 Rot/FinalErr 没有明显改善
```

则不视为实质主线切换。

### 8.2 combination promotion

组合成功必须满足：

```text
1. 优于 D_g_locked 或形成明确 Pareto improvement；
2. component attribution 证明不是单分量或 beta 偶然；
3. cue quality 没有异常恶化；
4. trajectory / segment metrics 同步改善；
5. hybrid-safe write 不破坏 memory safety；
6. 至少通过一个 cross-sequence sanity。
```

### 8.3 stopping rules

停止某个实验族的条件：

```text
1. 同族 6 个 full run 都不能超过 D_g_locked；
2. 所有提升都来自 mass 大幅下降但 segment metrics 不改善；
3. hybrid 总是比 read-only 差，说明 write-incompatible；
4. quality audit 显示严重 fragmentation 或 failure chunks 集中恶化；
5. cross-sequence 明显退化。
```

---

## 9. 建议的实际运行顺序

### Batch A：repeat 与同 beta 因果确认

```text
A1: C23 hybrid b3.75 repeat x2
A2: C24 hybrid b4.00 repeat x2
A3: C3 hybrid b3.75 repeat x1
A4: C23 read-only/hybrid beta 3.25/3.50/3.75/4.00/4.25
A5: C24 read-only/hybrid beta 3.50/3.75/4.00/4.25
```

### Batch B：layer / soft-g4

```text
B1: g2/g3/g4 full read-only
B2: g1_3/g2_3/g3_4/g2_4/g2_5 full read-only
B3: g2_3 + lambda*g4, lambda=0.10/0.25/0.50/0.75 read-only
B4: promising candidates hybrid-safe
```

### Batch C：support sweep

```text
C1: g2_3 support sweep read-only
C2: g2_4 support sweep read-only
C3: passing support candidates hybrid-safe
```

### Batch D：quality recalibration

```text
D1: export cue bank for C3/C23/C24/best variants/old_dyn/explicit/C_anchor
D2: quality-error correlation
D3: cue dashboard and failure gallery
```

### Batch E：lock D_g

```text
E1: build decision table
E2: choose D_g_locked and D_g_ate_shadow
```

### Batch F：cue combination

```text
F1: passive audit D_g + D_old / D_exp / R_static
F2: top candidates read-only
F3: passing candidates hybrid-safe
F4: component attribution
```

### Batch G：intervention path

```text
G1: FA residual gate small matrix
G2: SWA previous-source gate small matrix
G3: FA bias + best SWA gate
```

### Batch H：cross-sequence sanity

```text
H1: C3 / D_g_locked / D_g_ate_shadow on KITTI00/02/05 or 08
H2: best combination if any
H3: best intervention variant if any
```

---

## 10. 预期结果与解释方式

### 情况 A：`g2_3` 复现稳定，且 beta sweep 找到更好点

解释：`g2_3` 是当前主线。`g2_4` 的 ATE best 是局部边界提升，不足以压过 `g2_3` 的 rotation/final 优势。后续以 `g2_3` 做组合和 intervention path。

### 情况 B：`g2_4` 复现后 ATE 优势扩大

解释：`g4` 贡献确实重要。继续做 soft-g4 和 support sweep，寻找是否能保留 ATE 同时修复 final/yaw。

### 情况 C：weighted-g4 明显优于两者

解释：layer window 不应该硬平均。后续 cue library 应支持 layer-weighted aggregation，而不是只支持 singleton/window mean。

### 情况 D：support sweep 中 past_only 接近 full

解释：该 cue 可能具备更强 causal-friendly 潜力。后续跨序列和 streaming 设置优先测试 past_only variant。

### 情况 E：组合 cue 没有超过 `D_g_locked`

解释：目前最强信息仍来自 internal global query cue。下一步应回到 per-head、真实 attention-map、layer evolution，而不是继续手工 routing。

### 情况 F：SWA gate 改善 boundary 但不降 ATE

解释：SWA gate 是 local alignment auxiliary，不是 ATE 主线。可作为 balanced/local branch 保留，后续与 write policy 分开处理。

---

## 11. 本阶段结束时必须交付

```text
acl2_v4_experiment_report.md
acl2_v4_run_registry.csv
acl2_v4_cue_quality_recalibration.md
acl2_v4_Dg_locked_decision.md
acl2_v4_combination_validation_report.md
acl2_v4_next_stage_decision.md
```

`acl2_v4_next_stage_decision.md` 必须明确回答：

```text
1. 当前主线是 C23、C24、weighted-g4，还是回退 C3？
2. safe write 在新主线上的同 beta gain 是否稳定？
3. g4 是有效补充还是 trade-off 来源？
4. support pattern 是否改变主结论？
5. AnchorCollide / Frag 等质量指标应如何使用？
6. 是否存在通过 gate 的 cue combination？
7. 是否进入 TTT write policy 大阶段？
8. 是否需要优先转 per-head / attention-map / layer evolution？
9. cross-sequence sanity 是否支持当前主线？
```

---

## 12. 一句话执行结论

下一阶段不应该马上把 `38.3805m` 当作稳定新主线，也不应该直接开始大规模 cue fusion。正确路线是：

> 先复现并拆清 `g2_3/g2_4` 的 read/write/beta 因果关系，再解释 `g4` 的 ATE-rotation-endpoint trade-off，随后补 support sweep 和 cue quality 重标定；只有在锁定 `D_g_locked` 后，才进入小规模 routing / static rescue / residual-SWA intervention / cross-sequence sanity。


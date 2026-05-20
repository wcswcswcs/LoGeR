# ACL2 v8：TTT-native Cue、窗口化 Tri-Replay 与全局漂移修正实验计划

日期：2026-05-11  
对象：LoGeR / HMC Pipeline v2 / TTT write policy  
主开发集：KITTI Odometry Sequence 01  
当前固定 read/SWA 背景：

```text
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame-attention pair bias
read layers = all
beta = 4.75
commit = probe_ttt_write
base write = stage_d * sqrt(1 - D_g)
WRITE_ALPHA = 0.125
SWA = SWKS3-style fixed protocol
RESET_EVERY = 5
```

当前 TTT 写入最好完整结果：

```text
run = V7_TRIREPLAY_WINGAM_03_bodyg005_exitg0030_neu085_SWKS3
ATE / Rot = 34.1903 / 6.5666
RPE t / r = 92.4202 / 0.0083

TTT self cue = update_conflict_energy
objective = tri replay
branch = w0
chunks 5-9:
    positive_frac = 0.35
    negative_frac = 0.12
    neutral_lambda = 0.85
    negative_gamma = 0.005
chunks 10-12:
    positive_frac = 0.35
    negative_frac = 0.12
    neutral_lambda = 0.85
    negative_gamma = 0.003
```

当前最终目标仍然是：

```text
KITTI01 ATE RMSE < 30m
```

当前阶段的核心判断是：`update_conflict_energy + tri replay` 已经证明 TTT-native cue 有价值，但它仍不是最终的 TTT 写入策略。`WINGAM_03` 的改进更像 reset-window 级别的长期轨迹平衡，而不是已经真正修掉全场景漂移。下一阶段不能继续在 body/exit gamma 上做微扫，而要把手工窗口、手工 gamma、全层统一 routing 升级为可解释、可自动触发、可跨序列验证的 TTT write policy。

---

## 1. 实验整体目标

本轮 v8 的目标不是继续把 `34.1903` 靠 gamma 微调刷低一点，而是建立一套真正面向 **全局漂移修正** 的 TTT 写入机制。具体来说，本轮要完成五件事。

第一，确认 `WINGAM_03` 是可复现的当前主 baseline，并把它和 `B0_SWKS3`、`TTGRL_04`、`WINMAP_04`、`WINNEU_08` 放进同一个诊断体系。现在每个新实验都必须解释自己到底改善了什么：是降低 `[200,300)`，降低 `[400,600)`，稳定 Sim3 scale，减少 yaw drift，还是只是在局部 chunk 内优化。

第二，建立 **global drift dashboard**。之前大部分表格以 overall ATE、Rot、`[200,300)` 为核心，但 `WINGAM_03` 已经说明：最佳 overall ATE 并不是靠把 `[200,300)` 单段压到最低获得的，而是靠更好的 reset-window / long-trajectory balance。因此必须把 per-axis RMSE、cumulative translation drift、yaw drift、Sim3 scale、reset-group boundary error 和 per-reset-group ATE 纳入主表。

第三，把手工窗口策略变成 **自动触发策略**。当前最强配置依赖人工指定：chunks `5-9` 使用较强 negative gamma，chunks `10-12` 使用较弱 negative gamma。这个策略如果不能从 TTT 自己的内部指标自动触发，就很可能只是 KITTI01-specific tuning。v8 必须回答：`update_conflict_energy`、fast-weight delta spike、direction flip、neutral inertia、D_g mass、scale/yaw proxy 等指标能否自动发现 body window 和 exit window。

第四，把 TTT-native cue 从 chunk-level 进一步推进到 **layer/head/token direction**。目前 `update_conflict_energy` 已经比 semantic scalar、explicit dyn、residual reliability 更有效，但仍主要以统一 branch `w0`、统一 layer/head 的方式使用。v8 需要验证 layer/head routed tri-replay 是否能进一步改善长期轨迹，尤其是已经观察到风险集中在特定 layer/head 的情况下。

第五，建立进入跨序列验证的标准。任何只在 KITTI01 上靠手工 window、手工 gamma、手工 reset group 提升的策略，都不能直接称为新的 TTT 写入机制。v8 的候选必须先在 KITTI01 达到明确 gate，再跑 KITTI00 / KITTI02 / KITTI05 的 sanity check。

---

## 2. 当前证据与约束

### 2.1 已经成立的事实

当前最强 read cue 仍是 `C23 past`。它来自 LoGeR global decoder 第 `g2_3` 层 query 特征，使用 past-only support，产生的 `D_g` 在 read path 上作为 frame-attention pair bias 打到 all layers。v5/v6/v7 的结果都说明，这条 read path 是当前最大收益来源，TTT 写入策略应暂时固定在它之上，不再同时改 read cue。

TTT 写入侧已经从最初的 scalar gate 进入结构化 replay。单纯少写动态区域、explicit dyn veto、semantic scalar、sparse write、one-hop transient 都没有解决主问题；真正把结果推进到 `34m` 平台的是 TTT 自己的 `update_conflict_energy` 加 tri replay。

`freeze5/freeze56` 证明 chunks 5/6 中确实存在控制 `[200,300)` 的 fast-weight state。`freeze56` 能把 `[200,300)` 打到目标线附近，但 overall ATE 崩坏，说明 fast weight 中混合了 harmful direction 和必要 continuity，不能 hard remove。

`WINGAM_03` 的当前优势不来自 `[200,300)` 单段最低。它的 `[200,300)` 并不是所有候选中最低，但 overall ATE、Rot、axis RMSE 和全局 scale 更好。这意味着 TTT 写入策略不能再只围绕单段病灶设计，而要把 reset-window 和 whole-trajectory drift 同时纳入目标。

`RESET_EVERY=5` 是当前稳定性约束。reset=10 或 no-reset 在 strong tri-replay 下会严重崩坏，因此不能把“让 fast weights 活得更久”当作解决长期漂移的捷径。长期一致性必须靠写入内容、方向和生命周期，而不是简单延长记忆。

### 2.2 现在还没有成立的事实

还没有证明 `WINGAM_03` 可跨序列泛化。它可能只是 KITTI01 上针对 chunks `5-12` 的病灶特化策略。

还没有证明手工 chunks `5-9` / `10-12` 可以由 TTT self cue 自动发现。没有自动触发，策略就不能成为 general TTT write policy。

还没有证明 `update_conflict_energy` 已经是最优 TTT-native cue。它目前只是比 residual / semantic / D_g 更有效，还需要进一步与 direction alignment、pre-update prediction error、apply mismatch、fast-weight delta flip 等 TTT 内部量组合。

还没有证明 tri-replay 的 positive / neutral / negative roles 是 layer/head-independent 的。统一 `pos_frac=0.35`、`neg_frac=0.12`、`neutral_lambda=0.85` 可能只是平均后可用，真正有害方向可能集中在少数 layer/head。

---

## 3. 统一 baseline、候选与停止规则

### 3.1 必须保留的 baseline

每一批实验都必须至少包含下面四个 reference：

```text
B0_SWKS3:
    ATE / Rot = 36.4161 / 6.6128
    [200,300) = 77.831
    说明：v7 reproducible baseline

TTGRL_04:
    ATE / Rot = 36.2957 / 6.6182
    [200,300) = 77.644
    说明：chunk5 / w0 / all / gamma=0.025 localized negative replay reference

WINMAP_04:
    ATE / Rot = 34.3421 / 6.5767
    [200,300) = 75.440
    说明：split-window gamma reference，body=0.005, exit=0.0035, neutral=1.0

WINGAM_03:
    ATE / Rot = 34.1903 / 6.5666
    [200,300) = 75.576
    说明：current v7 best，body=0.005, exit=0.0030, neutral=0.85
```

### 3.2 新候选的分层成功标准

本轮不把 `34.18 -> 34.12` 这种微小变化视为实质成功，除非它同时改善 global drift dashboard。成功标准分三层。

**Diagnostic success**：候选解释了某个机制，即使 ATE 没有提升。例如：自动 trigger 能复现 chunks `5-9` / `10-12`，layer/head risk 能解释 WINGAM 的改进来源，或者某个 proxy 与 per-window ATE delta 有明显相关性。

**Relative success**：候选超过 `WINGAM_03` 的主指标或全局漂移指标：

```text
ATE <= 34.00
或 ATE <= 34.20 且同时满足以下至少两项：
    Rot <= 6.50
    [200,300) <= 74.50
    [200,400) <= 54.80
    FinalErr <= 5.80
    reset-group drift 明显优于 WINGAM_03
```

**Strong success**：候选进入下一阶段主线：

```text
ATE <= 33.50
且 [200,300) <= 74.00
且 [400,600) 不高于 WINGAM_03
且 100f / 200f mean ATE 同步下降
```

**Final success**：

```text
KITTI01 ATE < 30m
并且至少一个 cross-sequence sanity 不崩
```

### 3.3 停止规则

如果某个实验族连续 6 个 full run 都只在 `34.1-34.5m` 之间波动，且 global drift dashboard 没有新解释，则停止该实验族。

如果某个策略只改善 Rot / FinalErr，不改善 `[200,300)`、`[200,400)`、reset-group drift 或 axis RMSE，则只保留为 regularizer，不进入主线。

如果某个策略依赖固定 chunk id 且无法被自动 trigger 复现，则不能进入 cross-sequence sanity。

---

## 4. H1：`WINGAM_03` 是稳定主 baseline，且需要 global drift dashboard 而不是单段判断

### 4.1 假设

`WINGAM_03` 的收益不是单次运行偶然，也不是只由 `[200,300)` 单段下降构成；它的真实价值在于更好的 reset-window / long-trajectory balance。因此下一阶段应该把 `WINGAM_03` 固定为 TTT write 主 baseline，并建立全局漂移诊断仪表盘。

### 4.2 实验设计

先复现 `WINGAM_03` 一次。如果之前 run 是完全 deterministic，一次 repeat 足够；如果复现差异超过 `0.03m`，需要重复两次并记录均值/方差。

同时对下面四个 run 输出统一 dashboard：

```text
B0_SWKS3
TTGRL_04
WINMAP_04
WINNEU_08
WINGAM_03
```

如果计算资源允许，加上 `freeze5` 和 `freeze56` 作为 diagnostic-only 参考，但它们不参与 strategy ranking。

### 4.3 必须记录的指标

全局指标：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
50f mean / worst
100f mean / worst
200f mean / worst
```

关键 segment：

```text
[100,200)
[200,250)
[200,300)
[200,400)
[300,400)
[400,500)
[400,600)
[600,800)
```

per-reset-group 指标，假设 reset group 以 5 chunks 为单位：

```text
reset_group_id
chunk_range
frame_range
ATE_group
Rot_group
FinalErr_group
Yaw_group
Scale_group
BoundaryJump_to_next_group
```

axis-wise drift：

```text
RMSE_x
RMSE_y
RMSE_z
Cumulative_x_drift
Cumulative_y_drift
Cumulative_z_drift
```

TTT 写入诊断：

```text
per_chunk_update_conflict_mean
per_chunk_update_conflict_p90
per_chunk_positive_mass
per_chunk_neutral_mass
per_chunk_negative_mass
per_chunk_neutral_lambda_effective
per_chunk_negative_gamma_effective
per_layer_update_norm_w0/w1/w2
per_layer_update_conflict_w0
per_head_update_conflict_w0
memory_state_rel_diff
```

### 4.4 必须可视化

输出目录建议：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/global_drift_dashboard/
```

必须生成：

```text
trajectory_xy_overlay.png
trajectory_xz_overlay.png
per_frame_translation_error.png
per_100f_ATE_bar.png
per_200f_ATE_bar.png
cumulative_xyz_drift.png
cumulative_yaw_drift.png
sim3_scale_over_time.png
reset_group_ATE_heatmap.png
reset_group_boundary_jump_heatmap.png
chunk_error_vs_conflict_energy.png
layer_head_conflict_heatmap_chunk5_12.png
tri_replay_role_mass_over_time.png
```

### 4.5 假设成立标准

H1 成立需要满足：

```text
1. WINGAM_03 repeat ATE 与原结果差距 <= 0.03m；
2. WINGAM_03 在 overall ATE、Rot、至少一个 long-window mean 上优于 WINMAP_04；
3. Dashboard 能显示 WINGAM_03 的优势主要来自 reset-window/global drift balance，而不只是 [200,300)；
4. per-reset-group 指标能定位 WINGAM_03 改善最大的 reset group；
5. 输出完整 dashboard，并能用于后续候选统一比较。
```

如果 H1 不成立，即 WINGAM repeat 不稳定，则先停止所有新策略，排查 runner、cache、randomness、state hash 和 benchmark 路径。

---

## 5. H2：手工窗口 `5-9 / 10-12` 可以由 TTT self cue 自动触发

### 5.1 假设

当前最强策略不应该依赖人工指定 chunks `5-9` 和 `10-12`。如果 `update_conflict_energy` 真的是 TTT-native harmful-write cue，那么它应该能从 TTT 内部 trace 自动识别：哪些 chunks 属于 body correction window，哪些 chunks 属于 exit / handoff continuity window。

### 5.2 自动触发定义

定义每个 chunk 的 TTT body risk：

$$
R_m^{body} =
w_1 z(E_{m,w0}^{conflict,mean}) +
w_2 z(E_{m,w0}^{conflict,p90}) +
w_3 z(1 - \cos(\Delta W_m^{w0}, \Delta W_{m-1}^{w0})) +
w_4 z(\|\Delta W_m^{w0}\|) +
w_5 z(M_m^{D_g>0.5})
$$

其中 $z(\cdot)$ 是在当前 sequence 或当前 reset group 内的 robust z-score。

定义 exit / handoff risk：

$$
R_m^{exit} =
\mathbf{1}[R_{m-k}^{body} > \tau_{body}] \cdot
\left(
    a_1 z(E_{m,w0}^{conflict,mean}) +
    a_2 z(N_m^{neutral}) +
    a_3 z(B_m^{boundary})
\right)
$$

其中 $N_m^{neutral}$ 是 tri-replay neutral role mass 或 neutral inertia，$B_m^{boundary}$ 是 chunk-to-chunk pose jump / overlap consistency proxy。第一版可以没有 $B_m^{boundary}$，先用 TTT-only 指标。

自动策略定义：

```text
body chunks:
    top-k or threshold chunks by R_body within reset group

exit chunks:
    chunks immediately after body window where R_exit remains high but R_body begins to decay
```

### 5.3 实验矩阵

固定 base：

```text
read = C23 past pair/all beta 4.75
SWA = SWKS3 fixed
base write = stage_d * sqrt(1-D_g)
tri replay cue = update_conflict_energy
branch = w0
pos_frac = 0.35
neg_frac = 0.12
neutral_lambda = 0.85
```

候选：

```text
AUTO_WIN_01:
    body = top 5 chunks by R_body among all chunks
    exit = next 3 chunks after body window
    body_gamma = 0.005
    exit_gamma = 0.003

AUTO_WIN_02:
    body = threshold R_body > tau_body within each reset group
    exit = hysteresis tail until R_exit < tau_exit
    body_gamma = 0.005
    exit_gamma = 0.003

AUTO_WIN_03:
    body = top chunks by layer12/head0 conflict only
    exit = next 3 chunks
    body_gamma = 0.005
    exit_gamma = 0.003

AUTO_WIN_04:
    body = top chunks by combined R_body
    exit = no explicit exit window
    body_gamma = 0.005
    exit_gamma = 0

AUTO_WIN_05:
    body = manual 5-9
    exit = auto-triggered
    body_gamma = 0.005
    exit_gamma = 0.003

AUTO_WIN_06:
    body = auto-triggered
    exit = manual 10-12
    body_gamma = 0.005
    exit_gamma = 0.003
```

`AUTO_WIN_05/06` 用来区分 body 触发和 exit 触发哪个更难。

### 5.4 必须记录的指标

除了所有全局指标，还要记录自动触发质量：

```text
selected_body_chunks
selected_exit_chunks
manual_body_chunks = [5,6,7,8,9]
manual_exit_chunks = [10,11,12]
body_precision
body_recall
body_f1
exit_precision
exit_recall
exit_f1
R_body_per_chunk
R_exit_per_chunk
trigger_thresholds
```

其中：

$$
Precision_{body}=\frac{|B_{auto} \cap B_{manual}|}{|B_{auto}|+\epsilon}
$$

$$
Recall_{body}=\frac{|B_{auto} \cap B_{manual}|}{|B_{manual}|+\epsilon}
$$

### 5.5 可视化

```text
R_body_curve_with_selected_chunks.png
R_exit_curve_with_selected_chunks.png
manual_vs_auto_window_timeline.png
chunk_error_vs_R_body_scatter.png
chunk_error_delta_vs_R_body_scatter.png
reset_group_trigger_heatmap.png
```

### 5.6 假设成立标准

H2 的 diagnostic success：

```text
body_f1 >= 0.60
且 auto-selected chunks 覆盖 chunk5 或 chunk6 至少一个关键入口
```

H2 的 policy success：

```text
AUTO candidate ATE <= 34.40
且不使用固定 chunk ids
且 [200,300) <= 76.0
且 [400,600) 不高于 WINGAM_03 + 0.5m
```

H2 的 strong success：

```text
AUTO candidate ATE <= 34.00
或 ATE 接近 WINGAM_03 且 cross-sequence 上更稳
```

如果所有 auto trigger 都无法接近 `WINGAM_03`，说明当前风险指标尚不足以替代人工窗口。下一步不是继续调 threshold，而是先做 H3/H4 的 layer/head 和 direction attribution。

---

## 6. H3：Layer/head routed tri-replay 比全层统一 tri-replay 更适合 TTT 写入

### 6.1 假设

`update_conflict_energy` 的有效风险不是均匀分布在所有 TTT layer/head 上。已有诊断显示部分风险集中在特定 layer/head，因此统一 `w0/all-layer` 的 tri-replay 可能同时压到有害方向和有用 continuity。按 layer/head routing 应能减少不必要的 negative replay，同时保留长期轨迹修正。

### 6.2 路由定义

定义 layer/head risk：

$$
R_{m,l,h}^{conflict} = z(E_{m,l,h}^{conflict,mean})
$$

定义 routed gamma：

$$
\gamma_{m,l,h} = \gamma_m \cdot
\operatorname{clip}(1 + \alpha R_{m,l,h}^{conflict}, g_{min}, g_{max})
$$

第一版不要直接连续化所有 layer/head，而采用少数离散 routing：

```text
high-risk routed head/layer:
    gamma = body_gamma * scale_high

medium-risk layer/head:
    gamma = body_gamma

low-risk layer/head:
    gamma = body_gamma * scale_low 或不做 negative replay
```

### 6.3 实验矩阵

固定窗口先使用手工 `WINGAM_03` 窗口，避免同时改变 window trigger 和 layer routing：

```text
body chunks = 5-9
exit chunks = 10-12
body_gamma = 0.005
exit_gamma = 0.003
neutral_lambda = 0.85
pos_frac = 0.35
neg_frac = 0.12
branch = w0
```

候选：

```text
LH_01_layer12_head0_boost:
    layer12/head0 gamma *= 1.50
    other layers keep WINGAM_03

LH_02_layer12_head0_only:
    only layer12/head0 uses negative replay
    other layers keep positive+neutral only

LH_03_layer12_plus_layer5_9:
    layer12/head0 gamma *= 1.50
    layer5/layer9 gamma *= 0.75
    other layers keep WINGAM_03

LH_04_lowrisk_suppress:
    high-risk layers use WINGAM gamma
    low-risk layers no negative replay

LH_05_continuous_routed_gamma:
    gamma_lh = gamma * clip(1 + 0.5 * z(conflict_lh), 0.5, 1.5)

LH_06_exit_layer_routing:
    body window keep WINGAM
    exit window only high-risk layer/head gets gamma=0.003
    low-risk exit layers neutral only
```

### 6.4 必须记录的指标

```text
per_layer_head_gamma_effective.csv
per_layer_head_pos_mass.csv
per_layer_head_neu_mass.csv
per_layer_head_neg_mass.csv
per_layer_head_update_norm_before_after.csv
per_layer_head_update_cosine_to_WINGAM.csv
per_layer_head_conflict_energy.csv
per_layer_head_memory_diff.csv
```

核心方向指标：

$$
C_{l,h}^{candidate} = \cos(\Delta W_{l,h}^{candidate}, \Delta W_{l,h}^{WINGAM})
$$

$$
N_{l,h}^{ratio} = \frac{\|\Delta W_{l,h}^{candidate}\|}{\|\Delta W_{l,h}^{WINGAM}\|+\epsilon}
$$

### 6.5 可视化

```text
layer_head_conflict_heatmap_B0.png
layer_head_conflict_heatmap_WINGAM.png
layer_head_gamma_heatmap_candidate.png
layer_head_update_norm_ratio.png
layer_head_update_cosine_to_WINGAM.png
layer_head_contribution_waterfall.png
```

### 6.6 假设成立标准

H3 的 weak success：

```text
ATE <= 34.20
且 Rot <= WINGAM_03 Rot
且 layer/head routing 后 low-risk layer update norm 没有异常放大
```

H3 的 strong success：

```text
ATE <= 33.80
或 [200,300) <= 74.50 且 [400,600) 不恶化
```

如果 layer/head routing 只改善 Rot，不改善 ATE 或 long-window drift，则保留为 diagnostic，但不继续微扫。

---

## 7. H4：`update_conflict_energy` 还不够，必须加入 direction alignment 来区分 harmful direction 与 useful continuity

### 7.1 假设

`update_conflict_energy` 能识别 TTT replay 内部的风险能量，但它未必区分“有害方向”和“必要但强变化的 continuity direction”。因此，下一步需要用 direction alignment 将 high-energy token 分成三类：

```text
positive continuity:
    高写入影响，但方向与稳定/长期轨迹一致

negative conflict:
    高写入影响，且方向与稳定/长期轨迹相冲突

neutral adaptation:
    有写入影响，但方向既不明显有害也不明显正向
```

### 7.2 被动 attribution 先行

固定比较：

```text
B0_SWKS3
freeze5
freeze56
TTGRL_04
WINMAP_04
WINGAM_03
```

对 chunk5、chunk6、chunks 7-12 记录 fast-weight delta。

定义 freeze-removed direction：

$$
\Delta W_{freeze5\_removed}^{(l,h,r)} = \Delta W_{B0}^{(l,h,r)} - \Delta W_{freeze5}^{(l,h,r)}
$$

定义候选修正方向：

$$
\Delta W_{corr}^{(l,h,r)} = \Delta W_{candidate}^{(l,h,r)} - \Delta W_{B0}^{(l,h,r)}
$$

计算：

$$
A_{freeze}^{(l,h,r)} = \cos(\Delta W_{corr}^{(l,h,r)}, \Delta W_{freeze5\_removed}^{(l,h,r)})
$$

还要定义 continuity direction，例如 B0 中后段稳定 write 或 WINGAM 中保留的 positive direction：

$$
A_{cont}^{(l,h,r)} = \cos(J_i^{(l,h,r)}, \Delta W_{cont}^{(l,h,r)})
$$

### 7.3 新 risk score

在 token 级定义：

$$
R_i^{neg} = E_i^{conflict} \cdot \operatorname{clip}(1 - A_i^{cont}, 0, 1)
$$

或：

$$
R_i^{neg} = E_i^{conflict} \cdot \operatorname{clip}(A_i^{freeze}, 0, 1)
$$

其中 $A_i^{freeze}$ 表示 token update contribution 是否对齐 freeze 删除方向。

positive score：

$$
R_i^{pos} = (1 - E_i^{conflict}) \cdot \operatorname{clip}(A_i^{cont}, 0, 1)
$$

neutral score：

$$
R_i^{neu} = 1 - \max(R_i^{pos}, R_i^{neg})
$$

### 7.4 实验矩阵

固定窗口使用 WINGAM 手工 window：

```text
body chunks = 5-9
exit chunks = 10-12
body_gamma = 0.005
exit_gamma = 0.003
neutral_lambda = 0.85
branch = w0
```

候选：

```text
DIR_01_energy_baseline:
    negative risk = update_conflict_energy
    same as WINGAM_03 repeat

DIR_02_energy_x_not_continuity:
    negative risk = E_conflict * (1 - A_cont)
    positive risk = low E_conflict * A_cont

DIR_03_freeze_alignment:
    negative risk = E_conflict * positive_cos_to_freeze_removed
    positive risk = low E_conflict * A_cont

DIR_04_dual_alignment:
    negative risk = E_conflict * positive_cos_to_freeze_removed * (1 - A_cont)

DIR_05_exit_continuity_protect:
    body risk = DIR_04
    exit risk = E_conflict only when A_cont is low
```

### 7.5 必须记录的指标

```text
per_token_E_conflict.pt
per_token_A_cont.pt
per_token_A_freeze.pt
per_token_R_pos.pt
per_token_R_neu.pt
per_token_R_neg.pt
role_overlap_with_semantic.csv
role_overlap_with_Dg.csv
role_overlap_with_stage_d.csv
role_mass_by_chunk_layer_head.csv
```

### 7.6 可视化

```text
chunk5_rgb_overlay_R_neg.png
chunk5_rgb_overlay_R_pos.png
chunk5_R_neg_vs_Dg.png
chunk5_R_neg_vs_semantic_group.png
freeze_alignment_heatmap_layer_head.png
continuity_alignment_heatmap_layer_head.png
role_mass_timeline.png
```

### 7.7 假设成立标准

H4 成立的 diagnostic 标准：

```text
1. DIR risk 能解释 freeze5/freeze56 删除方向；
2. DIR risk 与 WINGAM 改善 chunk 的 update delta 有正相关；
3. DIR risk 的 semantic / Dg overlay 不只是简单复制 D_g 或 lowstuff。
```

H4 的 policy success：

```text
ATE <= 34.00
或 [200,300) <= 74.00 且 [400,600) 不恶化
```

如果 DIR 系列比 WINGAM 差，但 attribution 显示解释力强，保留为下一轮 replay structure 的 cue，不直接晋级。

---

## 8. H5：TTT 写入需要 window-level trajectory surrogate，而不是只看 chunk 内 replay loss

### 8.1 假设

当前 tri-replay 仍只根据 chunk 内 TTT replay/update cue 决定写入角色，但最终目标是长期 trajectory state。一个 chunk 的写入是否有害，不能只看当前 chunk replay，而要看它对后续 reset window 的局部轨迹、scale、yaw 和 boundary consistency 的影响。

因此需要一个 cheap window-level trajectory surrogate，让策略选择不完全依赖 full KITTI benchmark。

### 8.2 surrogate 定义

候选 proxy：

```text
P1: chunk-to-chunk pose jump
P2: overlap pointmap consistency
P3: local Sim3 scale proxy
P4: yaw jump proxy
P5: TTT apply mismatch on next chunk
P6: memory_state_rel_diff spike
P7: update_conflict_energy propagation into next chunk
```

定义综合 downstream risk：

$$
Q_{m \rightarrow m+K} =
q_1 z(P1) + q_2 z(P2) + q_3 z(P3) + q_4 z(P4) + q_5 z(P5)
$$

在离线评估阶段，计算 proxy 与真实 per-window ATE delta 的相关性：

$$
\rho_Q = Corr(Q_{m \rightarrow m+K}, \Delta ATE_{window})
$$

### 8.3 实验设计

第一批不直接控制模型，只做 correlation audit。对已有 run：

```text
B0
TTGRL_04
WINMAP_04
WINNEU_08
WINGAM_03
freeze5
freeze56
```

计算每个 chunk / reset group 的 proxy 和真实 error delta。

第二批做 cheap state selection。对 chunks `5-12`，每个 chunk 只比较两个候选 policy：

```text
candidate A = WINGAM-style tri replay
candidate B = baseline stage_d * sqrt(1-D_g)
```

使用 surrogate 选择当前 chunk 的 commit，然后跑 full KITTI01。

候选：

```text
SURR_01:
    select by overlap pointmap consistency

SURR_02:
    select by yaw/scale proxy

SURR_03:
    select by combined Q

SURR_04:
    select by combined Q but only inside auto-triggered body/exit windows
```

### 8.4 必须记录的指标

```text
proxy_values_per_chunk.csv
proxy_values_per_reset_group.csv
proxy_vs_true_delta_corr.csv
selection_decision_per_chunk.csv
selected_policy_timeline.csv
```

### 8.5 可视化

```text
proxy_vs_true_delta_scatter.png
selected_policy_timeline.png
proxy_curve_with_error_curve.png
reset_group_proxy_heatmap.png
```

### 8.6 假设成立标准

Diagnostic success：

```text
至少一个 proxy 与真实 per-window ATE delta 的 Pearson 或 Spearman 相关 >= 0.40
```

Policy success：

```text
SURR candidate ATE <= 34.20
且不依赖固定 chunk ids
```

Strong success：

```text
SURR candidate ATE <= 33.50
或 cross-sequence 比手工 WINGAM 更稳
```

如果所有 proxy 与真实误差无关，则停止 surrogate selection，转向纯 TTT-native layer/head attribution。

---

## 9. H6：双生命周期 fast weights 不能简单 short/long 分离，但可以做 conflict residual 的 gated long retention

### 9.1 假设

早期 dual-lifetime 实验显示纯 short residual 或简单减少 long commit 会回退，说明 update-conflict tri-replay 的收益必须部分进入 long fast weights。但这不等于所有 conflict residual 都应该长期保留。更合理的是：

```text
positive continuity -> long memory
neutral continuity  -> long memory with lambda < 1
negative conflict   -> short or weak long, depending on direction alignment
```

### 9.2 改进结构

定义：

$$
\Delta W_{long} = G_{pos} + \lambda_{neu}G_{neu} - \gamma_{long}G_{neg}^{aligned}
$$

$$
\Delta W_{short} = -\gamma_{short}G_{neg}^{unaligned}
$$

apply 时：

$$
W_{apply} = W_{long} + \alpha_t W_{short}
$$

commit 时：

$$
W_{commit} = W_{long}
$$

其中 $G_{neg}^{aligned}$ 是对长期轨迹仍有益或不伤 continuity 的 conflict residual，$G_{neg}^{unaligned}$ 是只应短期起作用的 correction。

### 9.3 实验矩阵

只在 H4 的 direction alignment 有解释力后启动。固定：

```text
body chunks = 5-9
exit chunks = 10-12
risk cue = best from H4
branch = w0
```

候选：

```text
DLR_01:
    gamma_long = 0.003
    gamma_short = 0.005
    K = until reset end
    alpha_t = 1.0

DLR_02:
    gamma_long = 0.005
    gamma_short = 0.003
    K = until reset end
    alpha_t = 0.5

DLR_03:
    body long = 0.005, short = 0
    exit long = 0.003, short = 0.003

DLR_04:
    long retention determined by A_cont
    short residual determined by A_freeze
```

### 9.4 必须记录的指标

```text
long_delta_norm
short_delta_norm
long_short_cosine
short_lifetime_remaining
apply_state_hash
commit_state_hash
memory_rel_diff_long
memory_rel_diff_short
```

### 9.5 假设成立标准

H6 成立需要：

```text
1. DLR candidate 不劣于 WINGAM_03；
2. [200,300) 或 [200,400) 明显优于 WINGAM_03；
3. [400,600) 不恶化；
4. long/short delta 可解释，不是所有 conflict 都被丢到 short。
```

如果 H6 候选仍然回退，则说明当前 LoGeR TTT fast weights 不适合 explicit dual-bank 改造，应回到 single-bank routed tri-replay。

---

## 10. H7：跨序列 sanity 必须验证 `update_conflict_energy + tri-replay` 是否泛化

### 10.1 假设

如果 `update_conflict_energy + tri replay` 是真实 TTT write mechanism，它应该在其他 KITTI sequence 上至少不崩，并且在长序列上带来一定 ATE / Rot / drift 改善。如果它只在 KITTI01 chunks `5-12` 有效，则它只是局部病灶特化策略。

### 10.2 候选选择

只让满足下面条件的 candidate 进入 cross-sequence：

```text
1. KITTI01 ATE <= 34.20；
2. 或 KITTI01 ATE <= 34.40 且使用 automatic window，不含固定 chunk ids；
3. 或 global drift dashboard 明显优于 WINGAM_03。
```

最小 cross-sequence 集合：

```text
B0_SWKS3
WINGAM_03 manual
best_auto_window_candidate
best_layer_head_candidate
```

测试序列：

```text
KITTI00 full
KITTI02 full
KITTI05 full
```

### 10.3 必须记录的指标

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
per_100f_mean/worst
per_200f_mean/worst
reset_group_drift
selected_auto_windows
```

### 10.4 假设成立标准

Cross-sequence weak success：

```text
candidate 在 00/02/05 平均 ATE 优于 B0_SWKS3
且没有任何一条序列 ATE regression > 5%
```

Cross-sequence strong success：

```text
candidate 在 00/02/05 平均 ATE 优于 B0_SWKS3 至少 1.0m
且 Rot 或 long-window mean 同步改善
```

如果 manual WINGAM 在 cross-sequence 上崩，而 auto-window 候选稳定，则优先发展 auto-window，即使 KITTI01 ATE 略高。

---

## 11. 需要新增或整理的工程模块

### 11.1 TTT cue cache

新增一个被动审计 cache：

```text
results/.../ttt_cue_cache/{run_id}/chunk_{m:03d}.pt
```

每个 chunk 保存：

```text
D_g
stage_d
update_conflict_energy per token / layer / head
pre_update_error
apply_mismatch
update_norm w0/w1/w2
update_direction_cosine_to_prev
update_direction_flip
pos/neu/neg role assignment
semantic group overlay if available
```

### 11.2 run registry

每个 full run 都必须写：

```text
run_config.yaml
run_hash.json
state_hash.jsonl
metrics_global.json
metrics_segments.csv
metrics_reset_groups.csv
metrics_ttt_roles.csv
metrics_layer_head.csv
```

### 11.3 dashboard generator

新增或扩展：

```text
tools/ttt_global_drift_dashboard.py
```

输入 run dirs，输出统一图表。该工具必须支持：

```text
--runs B0 TTGRL04 WINMAP04 WINGAM03 candidate
--segments 100,200
--reset-every 5
--output dashboard_dir
```

### 11.4 cheap window runner

为了避免每个候选 full run 30-60 分钟，建议新增：

```text
tools/run_ttt_window_candidate.py
```

功能：

```text
1. 从指定 chunk state snapshot 开始；
2. 只跑 chunks 5-14 或指定 reset group；
3. 输出 local proxy 和局部 trajectory diagnostics；
4. 只让通过 gate 的候选进入 full KITTI01。
```

该 runner 不能替代最终 full benchmark，但可以作为 Phase H2/H3/H4 的低成本筛选。

---

## 12. 执行顺序与资源预算

### Batch A：复现与 dashboard

```text
A1: WINGAM_03 repeat
A2: B0 / TTGRL_04 / WINMAP_04 / WINNEU_08 / WINGAM_03 dashboard
```

目标：确认当前 best 稳定，并建立之后所有实验的统一诊断面板。

通过后才进入 Batch B。

### Batch B：自动 window trigger，被动审计

```text
B1: compute R_body / R_exit for B0, WINMAP, WINGAM
B2: compare auto-selected windows vs manual 5-9 / 10-12
B3: run AUTO_WIN_01-04 full or window-gated first
```

目标：判断手工窗口能否自动化。

### Batch C：layer/head routed tri-replay

```text
C1: LH_01, LH_02, LH_03, LH_04
C2: 若 C1 有信号，跑 LH_05/LH_06
```

目标：判断是否能用 layer/head routing 替代全层统一 gamma。

### Batch D：direction alignment risk

```text
D1: passive freeze direction attribution
D2: DIR_02 / DIR_03 / DIR_04 / DIR_05
```

目标：判断 harmful direction 与 useful continuity 是否可区分。

### Batch E：window-level surrogate

```text
E1: proxy vs true delta correlation audit
E2: SURR_01-04
```

目标：把 TTT 写入策略从 chunk 内 cue 升级到 downstream consistency cue。

### Batch F：dual-lifetime gated retention

只在 H4 或 H5 有强解释力后启动。

```text
F1: DLR_01-04
```

目标：验证 single-bank routed tri-replay 是否需要升级成 gated long/short memory。

### Batch G：cross-sequence sanity

只对最多 2 个候选执行：

```text
G1: KITTI00 full
G2: KITTI02 full
G3: KITTI05 full
```

候选必须来自 Batch B/C/D/E 中通过 gate 的策略。

---

## 13. 每个实验必须输出的记录表

### 13.1 global_metrics.csv

字段：

```text
run_id
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
ATE_50_mean
ATE_50_worst
ATE_100_mean
ATE_100_worst
ATE_200_mean
ATE_200_worst
```

### 13.2 segment_metrics.csv

字段：

```text
run_id
segment_start
segment_end
ATE_segment
Rot_segment
Yaw_segment
Scale_segment
```

### 13.3 reset_group_metrics.csv

字段：

```text
run_id
reset_group_id
chunk_start
chunk_end
frame_start
frame_end
ATE_group
FinalErr_group
Yaw_group
Scale_group
BoundaryJump_prev
BoundaryJump_next
```

### 13.4 ttt_role_metrics.csv

字段：

```text
run_id
chunk_id
role_policy
pos_frac
neg_frac
neutral_lambda
neg_gamma
pos_mass
neu_mass
neg_mass
pos_update_norm
neu_update_norm
neg_update_norm
pos_neu_cosine
pos_neg_cosine
neu_neg_cosine
```

### 13.5 layer_head_metrics.csv

字段：

```text
run_id
chunk_id
layer_id
head_id
branch
conflict_mean
conflict_p90
energy_mean
update_norm
update_cosine_to_prev
update_flip
gamma_effective
pos_mass
neu_mass
neg_mass
```

### 13.6 trigger_metrics.csv

字段：

```text
run_id
chunk_id
R_body
R_exit
selected_as_body
selected_as_exit
manual_body
manual_exit
trigger_reason
threshold_body
threshold_exit
```

### 13.7 proxy_metrics.csv

字段：

```text
run_id
chunk_id
P_pose_jump
P_overlap_pointmap
P_scale_proxy
P_yaw_proxy
P_apply_mismatch
Q_combined
true_delta_ATE_window
true_delta_Yaw_window
```

---

## 14. 每个实验必须输出的可视化

### 14.1 主 dashboard

```text
trajectory_xy_overlay.png
trajectory_xz_overlay.png
per_frame_translation_error.png
per_frame_yaw_error.png
sim3_scale_over_time.png
cumulative_xyz_drift.png
segment_100f_bar.png
segment_200f_bar.png
reset_group_heatmap.png
```

### 14.2 TTT cue dashboard

```text
update_conflict_energy_timeline.png
layer_head_conflict_heatmap_chunks5_12.png
role_mass_timeline.png
gamma_effective_timeline.png
update_norm_by_branch_layer.png
update_cosine_to_prev_by_branch_layer.png
```

### 14.3 window trigger dashboard

```text
R_body_R_exit_timeline.png
manual_vs_auto_windows.png
trigger_reason_timeline.png
chunk_error_vs_R_body.png
chunk_error_vs_R_exit.png
```

### 14.4 direction attribution dashboard

```text
freeze_removed_direction_cosine.png
candidate_correction_vs_freeze_direction.png
continuity_alignment_heatmap.png
negative_role_overlay_chunk5.png
positive_role_overlay_chunk5.png
neutral_role_overlay_chunk5.png
```

### 14.5 semantic overlay optional

如果 semantic cache 可用，额外输出：

```text
semantic_group_vs_R_neg.png
semantic_group_vs_R_pos.png
semantic_group_role_mass_by_chunk.csv
sky_vegetation_road_building_role_maps.png
```

语义只用于解释 TTT-native cue，不作为本轮主控制变量，除非 H4 显示 semantic group 能解释 harmful direction。

---

## 15. 预期结果与决策树

### 情况 A：auto window 接近或超过 WINGAM

如果 `AUTO_WIN_*` 能在不使用固定 chunk id 的情况下达到：

```text
ATE <= 34.40
```

则下一阶段主线切到：

```text
update_conflict_energy + auto-window tri-replay
```

随后优先做 cross-sequence sanity，而不是继续 layer/head routing。

### 情况 B：layer/head routing 超过 WINGAM

如果 `LH_*` 达到：

```text
ATE <= 34.00
```

则说明 uniform all-layer tri-replay 确实过粗。下一阶段把 H2 的 auto-window 和 H3 的 layer/head routing 结合：

```text
auto window + layer/head routed gamma
```

### 情况 C：direction alignment 降低 `[200,300)`

如果 `DIR_*` 让：

```text
[200,300) <= 74.00
且 [400,600) 不恶化
```

则说明主病灶需要 harmful direction attribution，而不是单纯能量分数。下一阶段优先做 dual-lifetime gated retention。

### 情况 D：所有策略都卡在 34m 平台

如果 Batch B/C/D/E 都不能让 ATE 低于 `34.0` 或明显改善 `[200,300)`，则不要继续在 TTT write policy 上微扫。应回到 read path 或模型级 trajectory cue：

```text
1. per-head frame attention cue
2. deeper TTT apply/readback diagnostic
3. cross-chunk trajectory surrogate from geometry output
4. possible external/global alignment diagnostic only for analysis
```

---

## 16. 本轮最重要的实验哲学

v8 的核心不是“再调一个 gamma”，而是把 TTT 写入从手工策略推进到可解释机制：

```text
旧问题：
    哪些 token 少写一点？

新问题：
    哪些 token 是 positive continuity？
    哪些 token 是 harmful fast-weight direction？
    哪些 token 是 neutral trajectory evidence？
    哪些 layer/head 才真正携带 harmful direction？
    哪些 chunks 应自动进入 body / exit write window？
    当前写入是否改善了 whole-scene drift，而不只是局部 chunk？
```

只有当这些问题回答清楚，TTT 写入策略才有可能从 KITTI01 手工调参，变成可泛化的 LoGeR memory control policy。


# ACL2 v9：TTT Window-Drift Projection 与非标量写入策略实验计划

日期：2026-05-18  
对象：LoGeR / HMC Pipeline v2 / TTT-native write policy  
主开发集：KITTI Odometry Sequence 01  
当前固定 read 主线：`acl2.gg.qq.low.g2_3.past_only.headmean.robustq` + frame-attention `pair/all` + beta `4.75`  
当前固定 SWA 背景：`SWKS3-style fixed protocol`，本计划不再新增 SWA 策略变量  
当前 TTT 主线：`update_conflict_energy + tri_replay`  
当前 v8 best：

```text
V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3
ATE / Rot = 34.1583 / 6.5327
RPE t / r = 92.4059 / 0.0082
[200,300) = 74.522
```

最终目标仍然是：

```text
KITTI01 ATE RMSE < 30m
```

这意味着当前 best 距离最终目标仍差约：

$$
34.1583 - 30.0 = 4.1583m
$$

---

## 0. 当前阶段的独立判断

本阶段不能再被定义成“继续调 TTT 写入强度”。v7/v8 已经证明，`update_conflict_energy + tri_replay` 是目前最有效的 TTT-native 写入方向，但 v8 后半段也证明，继续调 `gamma / neg_frac / neutral / chunk16 role / layer12 scalar / commit filter / post-window scalar` 已经进入平台。

当前实验真正暴露的问题是：

```text
现有 TTT controller 主要仍在按 token/chunk 的局部 risk 调整写入，
但 KITTI01 的主失败模式更像 reset-window 内的累积漂移状态。
```

因此，下一阶段的目标不是把 `34.1583` 再微调到 `34.12`，而是验证一个更本质的假设：

> TTT 写入策略要从“局部 token risk 的标量门控”升级到“按 window-level drift direction 对 fast-weight update residual 做方向投影与成分分解”。

换句话说，我们要从：

$$
G_{commit}=G_{pos}+\lambda G_{neu}-\gamma G_{neg}
$$

升级为：

$$
G_{commit}=G_{cont}+G_{corr}-G_{harm}
$$

其中三个分量不是简单由 token risk 高低决定，而是由它们对窗口漂移方向的投影决定：

```text
G_cont: 对长期连续性有用的 fast-weight update 成分
G_corr: 能修正当前 reset-window 漂移方向的成分
G_harm: 会加剧 window drift 的成分
```

---

## 1. 这轮实验已经说明了什么

### 1.1 TTT write-side 确实有进展

从 v6/v7 的 `B0_SWKS3 = 36.4161` 到 v8 的 `C16ROLE_01 = 34.1583`，TTT 写入方向已经带来约：

$$
36.4161 - 34.1583 = 2.2578m
$$

的 ATE 改善。这不是没有进展，也不是随机微扰。`update_conflict_energy` 从最初的 diagnostic cue 发展成了有效的 TTT-native risk source；`tri_replay` 也比单纯 `stage_d * sqrt(1-D_g)`、sparse write、semantic scalar、old/expl dyn veto 更有机制意义。

但是这类进展主要来自：

```text
chunks 5-9 body window 的 weak negative replay
chunks 10-12 exit window 的更弱 negative replay
chunk16 的极弱 handoff / drift correction
```

它改善的是 reset-window 内的轨迹平衡，不是彻底解决 whole-scene drift。

### 1.2 当前 best 仍不是满意的 TTT 写入策略

当前 best `C16ROLE_01` 只比 `C16FINE_03` 好 `0.0044m`，属于 tiny gain。`[200,300)` 仍然高达 `74.522m`，而且 overall ATE 最优点和 `[200,300)` 最低点并不一致：`C16ROLE_03` / `C16FINE_03` 的 `[200,300)` 更低，但 overall ATE 不如 `C16ROLE_01`。

这说明：

```text
局部病灶最低 != overall trajectory 最优
```

也说明当前策略仍在分配误差，而不是消除漂移。

### 1.3 auto-window 第一批失败暴露了 cue 的语义错位

naive auto-window 会选择 `0,20,30,35` 这类 reset-like 或 globally recurring chunk，而不是病灶 body/exit window。它们常常改善 Rot / Yaw / FinalErr，却牺牲 `[200,300)` 和 overall ATE。

这说明当前 chunk-level `update_conflict_energy` 的全局峰值更像：

```text
姿态 / endpoint regularizer 信号
```

而不是：

```text
病灶 body / exit 触发器
```

因此，auto selector 不能直接用 raw global peak。它必须做 reset-relative normalization，并且要引入 downstream drift cost。

### 1.4 chunk16 是后段 handoff / scale correction，不是强 body window

chunk16 的有效 gamma 是 `1e-4` 到 `4e-4` 量级，远小于 body `0.005` 和 exit `0.003`。这说明 chunk16 不是新的强 negative body window，而是后段 weak handoff / scale-drift correction。

chunk16 能把 `[200,300)` 从 `75.576` 降到 `74.4x`，但常常让 `[400,600)` 从 `42.280` 恶化到 `44.x`。这暴露了新的 trade-off：

```text
修主病灶局部平台，会破坏后段连续性。
```

### 1.5 head-route 当前不能下机制结论

v8 中 per-head routing 结果与 reference 完全重合，并且 debug 中没有出现 expected routed 字段。这说明它大概率是 hook / routing path 没有真正生效，而不是“head routing 失败”。

因此，后续不能基于这些 no-op full runs 否定 per-head route。正确做法是先修 hook，做 no-op / non-no-op smoke，再允许 full run。

### 1.6 layer12 scalar routing 不够

layer12 的 `update_conflict_energy` 确实高，但单独加强 layer12 gamma 没超过当前 best。原因可能不是 layer12 cue 错，而是 action 错：

```text
当前 action 只是加大某层 gamma，仍是 scalar gate；
它没有区分该层内部哪些 update residual 是 drift-correcting，哪些是 continuity-preserving，哪些是 harmful。
```

这意味着下一步不应该继续做 layer12 标量微扫，而要做 direction-projection routing。

### 1.7 oracle window drift 也没有打开空间

H6 的 oracle 说明，body / exit / handoff 的 drift-state 是更正确的诊断单位，但已有 oracle full runs 没有超过 `C16ROLE_01`。这不是说明 window-level drift 方向没用，而是说明当前 action space 仍然太弱：它只是把 gamma / role fraction 做成 oracle 选择，仍然没有改变 replay residual 的成分定义。

因此下一步必须先验证更强的 upper-bound：

```text
如果用 GT drift direction 只做 update residual projection，能否超过 C16ROLE_01？
```

如果这个 oracle projection 也不行，才应停止 TTT write-side 扩展，转向 read-side / pose-scale failure。

---

## 2. 本计划的整体目标

本计划要回答四个核心问题。

### 2.1 问题 A：当前 TTT 写入是否只是局部 regularizer？

判断标准不是看 `34.1583` 是否又下降一点，而是看新策略是否同时改善：

```text
overall ATE
[200,300)
[200,400)
[400,600)
FinalErr
YawRMSE
Sim3 scale drift
RPE t/r
reset-window drift slope
```

如果候选只改善 Rot / FinalErr，却让 `[400,600)` 或 overall ATE 回退，它仍然是 regularizer，不是 drift solution。

### 2.2 问题 B：TTT-native cue 是否能自动触发 body / exit / handoff？

手工窗口 `5-9 / 10-12 / 16` 不能作为最终策略。必须从 TTT-native signals 中自动发现：

```text
body window: 需要较强 drift correction 的 reset-window 前半段
exit window: 需要弱 negative correction / neutral handoff 的窗口后半段
handoff chunk: 影响后段 scale / orientation 的微弱修正点
```

### 2.3 问题 C：当前 action space 是否已经到头？

如果 oracle projection 能超过 `C16ROLE_01`，说明当前瓶颈是 no-GT drift direction estimator；继续做 TTT write-side 有意义。  
如果 oracle projection 不能超过 `C16ROLE_01`，说明 TTT write-side 当前 observable / action space 已经接近上限；应转向 read-side 或 pose-scale correction。

### 2.4 问题 D：如何加快实验？

每一批都必须拆成：

```text
offline diagnostics / no-GT proxy first
smoke correctness second
small full-run parallel batch third
strict stopping gate fourth
```

不能再让 Codex 无条件跑 20 条 full KITTI01。每个阶段要给出不满足条件时的下一步尝试方向，让 Codex 自动推进或停止。

---

## 3. 固定 baseline 与冻结变量

### 3.1 固定 baseline

所有实验必须保留这些对照：

| 名称 | 配置 | 指标 |
|---|---|---:|
| `B0_SWKS3` | v6/v7 reproducible baseline | `36.4161 / 6.6128` |
| `WINGAM_03_repeat` | windowed tri-replay v7/v8 reference | `34.1903 / 6.5666` |
| `C16FINE_03` | chunk16 gamma `0.00030` | `34.1627 / 6.4986` |
| `C16ROLE_01` | current v8 best | `34.1583 / 6.5327` |

### 3.2 固定 read / SWA / reset

除非某阶段明确转向 read-side，所有 TTT write 实验固定：

```text
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
write source = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
SWA = SWKS3-style fixed protocol
RESET_EVERY = 5
commit = probe_ttt_write
```

不得修改 reset 机制。`RESET_EVERY=5` 是当前 LoGeR/HMC 对齐前提，不作为 v9 探索变量。

### 3.3 不再继续的方向

以下方向停止，除非计划中明确作为对照：

```text
body gamma 0.0048 / 0.0052 微扫
exit gamma 0.0028 / 0.0032 微扫
chunk16 gamma / role 微扫
chunk17-20 scalar positive recovery
layer12 scalar gamma 微扫
native_delta_gate / commit_filter 小扫
naive auto-window raw peak selector
head-route full run，直到 hook 修复并有 non-noop smoke
```

---

## 4. 核心假设与实验设计

---

# H1：当前平台来自 action space 错误，而不是 cue 完全错误

## 假设

`update_conflict_energy` 是有用的 TTT-native cue，但当前使用方式仍然太粗：它只控制 `gamma / neg_frac / neutral`，没有按 drift direction 分解 fast-weight update residual。

如果 H1 成立，应看到：

```text
1. C16ROLE_01 附近的 scalar variants 只在 34.16m 平台内微扰；
2. layer12 scalar routing 不能过 best；
3. commit filter / native delta gate 只改善 Rot / FinalErr，不改善 ATE；
4. projection-routed oracle 如果有效，会明显超过 scalar variants。
```

## 实验设计

H1 不再跑新的 full 模型，先做已有实验的离线归因。

输入 run：

```text
WINGAM_03_repeat
C16FINE_03
C16ROLE_01
C16ROLE_03
POSTWIN_01
POSTWIN_04
CFILTER_03
H6_ORACLE_03
H6_ORACLE_04
```

输出每个 reset-window 的 drift-state 表：

```text
window_id
chunk_ids
ATE_window
ATE_100_[200,300]
ATE_200_[200,400]
ATE_200_[400,600]
FinalErr
YawRMSE
Sim3Scale
scale_delta_vs_reference
x_drift_slope
z_drift_slope
yaw_drift_slope
update_conflict_mean
update_conflict_q90
tri_pos_mass
tri_neg_mass
tri_neutral_mass
commit_delta_norm_w0
native_control_update_cos_w0
memory_rel_diff_w0
```

定义窗口漂移向量：

$$
d_m = p_{m,end}^{aligned} - p_{m,start}^{aligned}
$$

定义相对 reference 的窗口 drift delta：

$$
\Delta d_m = d_m^{candidate} - d_m^{reference}
$$

定义 local gain 与 downstream cost：

$$
G_{local}=ATE_{200,300}^{reference}-ATE_{200,300}^{candidate}
$$

$$
C_{down}=ATE_{400,600}^{candidate}-ATE_{400,600}^{reference}
$$

## 必须记录的指标

```text
window_drift_state_raw.csv
window_drift_state_summary.csv
run_pair_tradeoff.csv
feature_correlation_to_local_gain.csv
feature_correlation_to_downstream_cost.csv
```

## 可视化

```text
window_drift_vector_plot_xy.png
window_drift_vector_plot_xz.png
local_gain_vs_downstream_cost_scatter.png
conflict_energy_vs_local_gain.png
conflict_energy_vs_downstream_cost.png
chunk16_gamma_vs_[200,300]_[400,600]_pareto.png
```

## 假设成立标准

H1 成立条件：

```text
1. scalar variants 的 local gain 与 downstream cost 存在稳定 trade-off；
2. update_conflict_energy 单独不能预测 final ATE，但能解释一部分 local gain；
3. downstream cost 需要额外的 scale/yaw/window drift features 才能解释；
4. C16ROLE_01 处于 Pareto 前沿，但不是 [200,300) 最低点。
```

如果 H1 不成立，也就是现有 diagnostics 无法解释任何 trade-off，则 Codex 应先补 debug，不允许继续 full run：

```text
Codex fallback:
    - 检查每个 run 是否有 per-window trajectory diagnostics；
    - 补齐 hmc_probe_summary / hook_effect_summary 的 chunk ids；
    - 补齐 tri_pos/neg/neutral mass by chunk；
    - 补齐 commit_delta_norm_w0 by chunk；
    - 重新生成 window_drift_state_raw.csv。
```

---

# H2：oracle projection 是 TTT write-side 是否还有空间的关键 upper bound

## 假设

如果 TTT 写入侧仍有大幅空间，那么使用 GT 轨迹仅构造 drift direction，不使用误差大小，应该可以把 fast-weight update residual 分成更正确的 `corr / cont / harm` 三类，并超过 `C16ROLE_01`。

## 核心思想

对每个 token/layer/head/branch 的 update contribution $J_i$，计算它对窗口 drift direction 的投影符号。

先定义窗口 drift error direction：

$$
e_m = \frac{d_m^{candidate} - d_m^{gt}}{\|d_m^{candidate} - d_m^{gt}\| + \epsilon}
$$

如果某个 update residual 对当前 apply / pose proxy 的一阶影响方向与 $e_m$ 同向，则它可能加剧 drift；如果反向，则它可能纠正 drift。

形式上先用可实现的 proxy：

$$
P_i = \langle \phi(J_i), e_m \rangle
$$

其中 $\phi(J_i)$ 是 token update 对 pose / apply / hidden residual 的可观测投影代理。第一版可以用：

```text
TTT apply output delta projected to pose-token / camera-token channel
或者 chunk-level probe/control pose increment disagreement 的线性方向
```

分组：

```text
P_i > tau_pos:
    harmful, enters G_harm

P_i < -tau_neg:
    corrective, enters G_corr

otherwise:
    continuity / neutral, enters G_cont
```

提交目标：

$$
G_{commit}=G_{cont}+\lambda_c G_{corr}-\gamma_h G_{harm}
$$

## 实验设计

第一批只跑 2 条 full KITTI01，作为 upper-bound。

| Run | 机制 | 目的 |
|---|---|---|
| `V9_ORACLEPROJ_01` | chunks `5-12` projection-routed tri replay | 验证 body/exit window 内 direction projection 是否能超过 C16ROLE |
| `V9_ORACLEPROJ_02` | chunks `5-12 + 16` projection-routed long gate | 验证 handoff chunk 是否能用 projection 解决 `[400,600)` cost |

固定：

```text
base = C16ROLE_01
risk = update_conflict_energy
branch = w0
SWA = SWKS3 fixed
RESET_EVERY = 5
```

建议初始参数：

```text
lambda_cont = 0.85
gamma_harm_body = 0.005
gamma_harm_exit = 0.003
gamma_harm_c16 = 0.0003
corr_boost = 1.05
projection_tau = q70(|P_i|)
```

## 必须记录的指标

```text
projection_group_mass_by_chunk.csv
projection_group_mass_by_layer.csv
projection_group_mass_by_head.csv
harmful_projection_mean.csv
corrective_projection_mean.csv
window_drift_state_comparison.csv
```

字段：

```text
chunk_id
layer_id
head_id
pos_mass
neutral_mass
harm_mass
corr_mass
harm_update_norm
corr_update_norm
cont_update_norm
cos_harm_to_native
cos_corr_to_native
window_drift_projection_before
window_drift_projection_after
```

## 可视化

```text
projection_group_mass_heatmap_chunk_layer.png
projection_sign_map_overlay_chunk5_8.png
oracle_projection_trajectory_xy.png
oracle_projection_error_over_time.png
oracle_projection_pareto_local_vs_downstream.png
```

## 成立标准

强成立：

```text
ATE <= 33.80
且 [200,300) <= 73.5
且 [400,600) <= 44.0
```

弱成立：

```text
ATE < 34.1583 - 0.15
或 [200,300) 下降 >= 1.5m 且 [400,600) 不恶化超过 0.5m
```

不成立：

```text
两条 oracle projection 都不能超过 34.1583
或只改善 Rot/FinalErr，不改善 ATE
或 [400,600) 明显恶化
```

## 不成立时 Codex 的尝试方向

如果 H2 不成立，Codex 不应继续 full run，而应自动转到：

```text
1. 检查 phi(J_i) 是否真的非零，输出 projection_debug_nonzero_rate；
2. 检查 projection grouping 是否改变了 replay multiplier，输出 changed_token_mass；
3. 若 grouping 是 no-op，修 hook；
4. 若 grouping 生效但无收益，停止 TTT write-side action 扩展，进入 H8 read/pose-scale 分支；
5. 同时保留 C16ROLE_01 为 TTT write best。
```

---

# H3：no-GT drift proxy 只能在 oracle projection 成功后进入控制

## 假设

如果 oracle projection 成功，说明 direction-projection action space 有价值。下一步要用 no-GT features 近似 drift direction，而不是继续使用 GT。

## no-GT feature 候选

```text
pass1/pass2 pose increment disagreement
probe vs controlled camera-token delta
TTT apply mismatch trend
SWA overlap source residual
window-level update_conflict_energy drift
memory_ttt_w0 relative diff slope
Sim3 scale proxy from overlap alignment
chunk-to-chunk yaw increment mismatch
```

定义 no-GT drift direction estimator：

$$
\hat e_m = \operatorname{Normalize}\left(\sum_k a_k f_{m,k}\right)
$$

初始不训练深模型，只做线性组合与 rank scoring。

## 离线验证

用已有 runs 作为 labeled dataset：

```text
WINGAM_03
C16FINE_03
C16ROLE_01-07
POSTREG_04/09/11/14
POSTWIN_01-04
CFILTER_01-04
H6_ORACLE_01-04
```

label：

```text
local_gain = [200,300] improvement
post_cost = [400,600] degradation
overall_gain = ATE improvement
safe = overall_gain > 0 and post_cost < 0.5
```

## 指标

```text
AUC_safe_vs_unsafe
Spearman(proxy_score, overall_gain)
Spearman(proxy_local, local_gain)
Spearman(proxy_cost, post_cost)
TopK precision for best 5 configs
```

## 判定标准

no-GT proxy 通过离线 gate：

```text
AUC_safe_vs_unsafe >= 0.75
Spearman(proxy_score, overall_gain) >= 0.45
Top5 precision >= 0.60
```

如果不过：

```text
Codex fallback:
    - 不运行 no-GT projection full；
    - 增加 feature: apply_mismatch_slope, window_hidden_norm_slope；
    - 尝试使用 reset-window relative normalization；
    - 若仍不过，转 H8 read/pose-scale。
```

## full run 设计

只有离线通过后，跑 4 条 full：

| Run | drift direction source | window | 目的 |
|---|---|---|---|
| `V9_PROXYPROJ_01` | pass1/pass2 pose disagreement | 5-12 | 最直接 no-GT 方向 |
| `V9_PROXYPROJ_02` | TTT apply mismatch slope | 5-12 | TTT-native readback 方向 |
| `V9_PROXYPROJ_03` | combined proxy linear score | 5-12 | 综合 proxy |
| `V9_PROXYPROJ_04` | combined proxy + chunk16 handoff | 5-12 + 16 | 测 handoff |

## full run promotion gate

```text
ATE <= 33.80
或 ATE <= 34.00 且 [200,300) <= 73.8 且 [400,600) <= 44.0
```

---

# H4：auto-window selector 必须 reset-relative 与 downstream-aware

## 假设

raw `update_conflict_energy` 会选择 reset-like false positives，因此 auto-window 必须做：

```text
reset-group relative normalization
body/exit role separation
downstream cost prediction
```

## 新 auto-window score

定义 reset group $r$ 内 chunk $m$ 的标准化 risk：

$$
R_m^{rel}=\frac{R_m-\operatorname{Median}_{j\in r}(R_j)}{\operatorname{MAD}_{j\in r}(R_j)+\epsilon}
$$

定义 body score：

$$
S_m^{body}=R_m^{rel}+a\cdot SlopeErr_m+b\cdot DgMass_m-c\cdot ResetPeak_m
$$

定义 exit score：

$$
S_m^{exit}=ExitShape_m+d\cdot NeutralMass_m-e\cdot PostCostProxy_m
$$

定义 handoff score：

$$
S_m^{handoff}=ScaleShiftProxy_m+YawShiftProxy_m-PostCostProxy_m
$$

其中 `ResetPeak` 是用于惩罚 `0,20,30,35` 这类周期性峰值的特征。

## 离线评价

用手工窗口作为 weak label：

```text
body = 5-9
exit = 10-12
handoff = 16
```

但不要硬追 F1，必须同时看 performance proxy：

```text
body_F1
exit_F1
handoff_hit
reset_false_positive_count
predicted_post_cost
```

通过标准：

```text
body_F1 >= 0.60
exit_F1 >= 0.50
reset_false_positive_count <= 1
handoff includes 16 or 15/16
```

## full run 小矩阵

通过后最多 4 条：

| Run | body | exit | handoff | 目的 |
|---|---|---|---|---|
| `V9_AUTOWIN2_01` | auto | manual 10-12 | manual 16 | 测 body auto |
| `V9_AUTOWIN2_02` | manual 5-9 | auto | manual 16 | 测 exit auto |
| `V9_AUTOWIN2_03` | manual 5-9 | manual 10-12 | auto | 测 handoff auto |
| `V9_AUTOWIN2_04` | auto | auto | auto | 全自动 |

通过标准：

```text
ATE <= 34.10
且没有 [400,600) > 44.8
```

如果不通过：

```text
Codex fallback:
    - 输出 auto-selected windows 与 manual windows 的差异表；
    - 标记 false-positive chunks 的 top features；
    - 不再跑 auto full，回到 manual + projection oracle；
    - 或进入 H8 read/pose-scale。
```

---

# H5：trajectory-aware run selection 用于减少 full run 浪费

## 目标

v8 已经出现大量“Rot/FinalErr 好但 ATE 不过”的 run。下一阶段必须用 cheap surrogate 先过滤。

## surrogate 定义

定义：

$$
S_{traj}=ATE_{coarse}+a\max(0,ATE_{400,600}-ATE_{400,600}^{ref})+b\cdot YawRMSE+c\cdot |Scale-Scale_{ref}|
$$

其中 `ATE_coarse` 可以来自：

```text
short full subset
window diagnostics
prefix + selected window evaluation
```

第一版不用于最终结论，只用于 full-run 排队。

## 离线拟合

用已有 runs 拟合参数 $a,b,c$，目标是 rank correlation：

```text
Spearman(S_traj, full_ATE) >= 0.60
TopK recall for best-5 configs >= 0.60
```

如果达不到，surrogate 只做可视化，不做 gate。

## Codex 任务

```text
tools/fit_trajectory_surrogate.py
输入：所有 trajectory_diagnostics_*/*.csv
输出：surrogate_fit_report.md, surrogate_coefficients.json
```

如果 fitting 失败：

```text
Codex fallback:
    - 改用 rule-based hard filter：
      reject if [400,600) worsens > 1.0m
      reject if FinalErr worsens > 0.5m
      reject if Sim3 scale deviates > 0.25
```

---

# H6：head-route hook 必须先修，不允许继续 no-op full run

## 假设

per-head route 可能仍然有价值，但当前 v8 的 head-route full run 是 no-op，不能用于机制判断。

## 工程 smoke

Codex 必须先实现并验证：

```text
TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES 生效路径
tri_replay + chunk_gamma_map + head_routes 同时开启时可改变 multiplier
hmc_state_hash.jsonl 输出 ttt_head_routed_* 字段
changed_head_token_mass > 0
```

最小 smoke：

```text
END_FRAME=96
head route = layer12:head0
gamma exaggerated = 0.05
expected: trajectory 或 write debug 必须不同于 no-route reference
```

通过标准：

```text
debug shows route parsed
changed_head_token_mass > 0.01
update_norm_w0 differs from reference
commit hash differs from reference
```

如果 smoke 不通过：

```text
Codex fallback:
    - 不跑 full；
    - 检查 route parse 是否到 HMC；
    - 检查 HMC 是否传到 TTTWriteController；
    - 检查 TTTWriteController 是否按 head dimension 分配 multiplier；
    - 检查 debug 是否记录 per-head changed mass。
```

full run 只有 smoke 通过后才允许：

| Run | head route | 目的 |
|---|---|---|
| `V9_HEADROUTE_01` | layer12/head0 stronger negative | 测 high-risk head |
| `V9_HEADROUTE_02` | layer12/head0 negative + other heads neutral | 保 continuity |
| `V9_HEADROUTE_03` | layer12/head0 + layer5/head0 weak | 组合 |

promotion gate：

```text
ATE <= 34.05
或 ATE <= 34.15 且 Rot <= 6.35 且 [400,600) 不恶化
```

---

# H7：cross-sequence sanity 只给真正通过 KITTI01 gate 的候选

## 进入条件

候选满足任一条件才进入 cross-seq：

```text
ATE <= 33.80
或 ATE <= 34.00 且 [200,300) / [400,600) 同时改善
或 oracle projection 证明 action space 有效，需要测试是否只过拟合 KITTI01
```

## 序列

```text
KITTI00 full
KITTI02 full
KITTI05 full
```

## 对照

```text
B0_SWKS3
WINGAM_03
C16ROLE_01
new candidate
```

## 指标

```text
per-sequence ATE
per-sequence Rot
per-sequence FinalErr
per-sequence [200,300)-like worst 100f
per-sequence worst 200f
average ATE
average Rot
regression_count
```

## 通过标准

```text
average ATE improves over C16ROLE_01-equivalent baseline
no sequence ATE regression > 5%
no sequence Rot regression > 5%
at least one non-KITTI01 long sequence improves by >= 0.5m
```

如果不过：

```text
标记为 KITTI01-specific；
不继续调该策略；
回到 cue/action design。
```

---

# H8：如果 oracle projection 失败，转向 read-side / pose-scale 本质问题

## 触发条件

满足任一条件即触发 H8：

```text
oracle projection 不能超过 C16ROLE_01
no-GT proxy 离线判别不过
auto-window reset-relative 仍不能复现 manual body/exit/handoff
head-route 修复后仍无收益
所有候选连续 12 条 full run 都不能超过 34.10
```

## 判断

这说明当前 TTT write-side 已经接近可观察 action space 的上限。剩余 `4.16m` 不是继续写入策略微调能解决的，需要重新看：

```text
read-side window-conditioned beta / support selection
pose-scale drift proxy
frame-attention cue per-head / attention-map
Sim3 scale / merge alignment failure
SWA window-level source policy
```

## H8 第一批方向

### H8-A read-side window-conditioned beta

把 `C23 past pair/all beta=4.75` 改成窗口条件：

```text
body window 5-9: beta 4.75 or 5.0
exit window 10-12: beta 4.25 or 4.5
handoff chunk16: beta 4.25
```

目的：验证 current drift 是否更像 read-path over/under-suppression，而不是 write-path。

### H8-B support-conditioned read cue

测试：

```text
body window: past_only
exit window: near12 or full
handoff: future diagnostic only offline
```

### H8-C pose-scale proxy reranking

不改 model，先对 chunk-level trajectory merge / scale 做 diagnostic reranking。只作为机制验证，不作为最终策略。

### H8-D return to SWA window-level source policy

只在 window-level，而不是 token rho：

```text
body window: keep both_overlap
exit window: reduce tail source
handoff: structure-only source keep
```

H8 的 full run 数量第一批最多 6 条，且必须并行。

---

## 5. 并行与加速策略

### 5.1 三层运行队列

每一轮必须分成三类任务并行：

```text
Queue A: offline diagnostics / tool building / CSV aggregation
Queue B: smoke correctness / no-op / short END_FRAME
Queue C: full KITTI01 candidates
```

Codex 优先同时推进 A/B，不等待 full run 结束。

### 5.2 full run 并发

根据现有记录：

```text
KITTI01 full: 6-7 并发通常可用，每条约 30-37 min
8 并发可能造成 host RAM 危险，不作为默认
KITTI00/02 long sequence: 默认 2-3 并发
```

### 5.3 每批 full run 数量上限

```text
oracle projection: <= 2 full runs
proxy projection: <= 4 full runs
auto-window: <= 4 full runs
head-route after smoke: <= 3 full runs
read-side fallback: <= 6 full runs
```

任何阶段不满足 gate，不允许继续同族微扫。

### 5.4 run registry 必须统一

每个 run 必须写入：

```text
run_id
family
hypothesis
config_hash
base_reference
gpu_id
start_time
end_time
walltime
ATE
Rot
RPE_t
RPE_r
FinalErr
[200,300]
[200,400]
[400,600]
YawRMSE
Sim3Scale
promotion_status
failure_reason
next_action
```

---

## 6. 必须记录的指标

### 6.1 全局轨迹指标

```text
ATE RMSE
Rot RMSE
RPE t
RPE r
FinalErr
YawRMSE
Sim3Scale
50f mean / worst
100f mean / worst
200f mean / worst
```

### 6.2 病灶窗口指标

```text
[100,200)
[200,250)
[200,300)
[200,400)
[300,400)
[400,500)
[400,600)
chunk5-8 cumulative error slope
chunk9-12 recovery slope
chunk16 handoff effect
```

### 6.3 TTT-native 写入指标

```text
update_conflict_energy_mean / q90
positive_mass
negative_mass
neutral_mass
commit_delta_norm_w0/w1/w2
native_control_update_cos_w0/w1/w2
memory_rel_diff_w0/w1/w2
per-layer update norm
per-head update norm
per-layer/head conflict energy
changed_token_mass
changed_head_token_mass
```

### 6.4 projection 指标

```text
projection_pos_mass
projection_neutral_mass
projection_harm_mass
corr_update_norm
harm_update_norm
cont_update_norm
cos_corr_to_native
cos_harm_to_native
projection_nonzero_rate
projection_changed_commit_norm
```

### 6.5 auto-window 指标

```text
body_F1_vs_manual
exit_F1_vs_manual
handoff_hit
reset_false_positive_count
selected_chunks
selected_reason_top_features
body_score
exit_score
handoff_score
```

### 6.6 工程正确性指标

```text
no-op hash match
commit hash changed when expected
head route parsed
head route changed mass
projection mode parsed
projection group nonzero
num_failed_chunks
runtime
peak GPU memory
peak host memory
```

---

## 7. 必须可视化的内容

### 7.1 global drift dashboard

每个晋级候选必须输出：

```text
full XY trajectory
first half trajectory
second half trajectory
per-frame aligned error
sliding 100f ATE
sliding 200f ATE
Sim3 scale over time
yaw error over time
chunk cumulative drift vector
```

### 7.2 window-level trade-off plots

```text
[200,300] vs [400,600] scatter
ATE vs [200,300] scatter
ATE vs [400,600] scatter
Rot vs ATE scatter
FinalErr vs ATE scatter
Sim3Scale vs ATE scatter
```

### 7.3 TTT cue heatmap

```text
chunk × layer update_conflict_energy
chunk × head update_conflict_energy
chunk × layer positive/neutral/negative mass
chunk × layer commit_delta_norm
chunk × layer native_control_cos
```

### 7.4 projection visualization

```text
chunk5-12 projection sign map
projection group mass by layer/head
corr/harm/cont contribution heatmap
oracle vs proxy direction cosine over chunks
```

### 7.5 auto-window visualization

```text
manual vs auto window timeline
auto score timeline by chunk
reset false positive markers
body/exit/handoff labels
```

---

## 8. 第一轮具体执行顺序

### Batch A：离线诊断，不跑 full

```text
A1: build window_drift_state_raw.csv for all v8 runs
A2: fit local_gain / downstream_cost feature correlations
A3: generate global drift dashboard for WINGAM / C16FINE / C16ROLE / POSTWIN / H6_ORACLE
A4: output decision: proceed to H2 oracle projection or fix diagnostics
```

并行任务：Codex 可以立即实现 `tools/ttt_window_drift_state_audit.py` 和 `tools/fit_trajectory_surrogate.py`。

### Batch B：projection oracle smoke

```text
B1: END_FRAME=128 projection oracle smoke, exaggerated effect
B2: verify projection_group_mass nonzero
B3: verify commit hash changes
B4: verify no-op when projection mode disabled
```

不通过则修 hook，不跑 full。

### Batch C：oracle projection full，最多 2 条

```text
C1: V9_ORACLEPROJ_01 chunks 5-12
C2: V9_ORACLEPROJ_02 chunks 5-12 + 16
```

如果两条都不过，直接跳 H8，不再做 no-GT proxy full。

### Batch D：no-GT proxy 离线判别

仅当 Batch C 通过：

```text
D1: build proxy dataset
D2: fit proxy direction / safe-vs-unsafe classifier
D3: validate AUC / Spearman / TopK
```

### Batch E：proxy projection full，最多 4 条

仅当 Batch D 通过：

```text
E1: pass1/pass2 pose disagreement proxy
E2: TTT apply mismatch proxy
E3: combined proxy
E4: combined proxy + chunk16 handoff
```

### Batch F：auto-window selector v2

与 D/E 可并行做离线，不依赖 proxy full：

```text
F1: reset-relative auto-window score
F2: body/exit/handoff offline F1
F3: only if offline pass -> 4 full runs
```

### Batch G：head-route hook repair

与所有阶段并行，但只做 smoke：

```text
G1: fix head-route hook
G2: END_FRAME non-noop smoke
G3: if pass -> 3 full candidates
```

### Batch H：fallback read / pose-scale

如果 C/D/E/F/G 都失败或没有超过 `34.00`：

```text
H1: read-side window-conditioned beta small matrix
H2: support-conditioned read cue small matrix
H3: pose-scale diagnostic reranking
H4: SWA window-level source policy small matrix
```

---

## 9. promotion / stopping rules

### 9.1 强成功

```text
ATE <= 33.50
且 [200,300) <= 73.0
且 [400,600) <= 44.0
```

### 9.2 弱成功

```text
ATE <= 34.00
且同时满足以下至少两项：
    [200,300) <= 74.0
    [400,600) <= 44.0
    Rot <= 6.35
    FinalErr <= 6.00
```

### 9.3 diagnostic success

```text
没有过 ATE best，
但清楚证明某个 action space 不成立，
并能减少未来 full-run 搜索空间。
```

### 9.4 停止规则

同一 family 满足任一条件立即停止：

```text
4 条 full run 都不能超过 C16ROLE_01
或 best 只改善 < 0.02m 且没有其它指标改善
或只改善 Rot/FinalErr，ATE/[400,600) 系统性回退
或 smoke 显示 hook no-op
或 oracle upper-bound 都不过 best
```

---

## 10. Codex 自动推进规则

### 10.1 如果离线诊断缺字段

Codex 应先补数据，不跑 full：

```text
补 per-run trajectory diagnostics
补 hmc debug jsonl parser
补 tri mass by chunk/layer/head
补 memory delta summary
```

### 10.2 如果 projection oracle smoke 是 no-op

Codex 自动检查：

```text
CLI arg 是否接入 run_pipeline_abc_v2.py
HMC 是否接收到 projection mode
TTTWriteController 是否收到 projection grouping
multiplier 是否改变
zeropower 前后的 update norm 是否改变
commit state hash 是否改变
```

### 10.3 如果 oracle projection full 失败

Codex 不再做 proxy full，转 H8：

```text
read-side window beta
pose-scale proxy
SWA window source
```

### 10.4 如果 oracle projection 成功但 proxy 离线失败

Codex 尝试增强 no-GT features：

```text
pass1/pass2 pose disagreement
TTT apply mismatch
SWA overlap residual
hidden norm slope
window cumulative delta norm
```

最多两轮 feature 增强；仍不过则停止 proxy full。

### 10.5 如果 auto-window offline 不过

Codex 输出 false-positive explanation，不跑 full：

```text
which chunks selected incorrectly
which features caused selection
whether reset-relative normalization failed
```

### 10.6 如果 head-route hook no-op

Codex 只修工程，不跑 full。修完必须有：

```text
changed_head_token_mass > 0
commit hash differs
short-run metric differs
```

### 10.7 如果所有 TTT write-side action space 都不过

Codex 自动进入 H8 fallback，不再提议：

```text
gamma micro sweep
neutral micro sweep
chunk16 role micro sweep
layer12 scalar sweep
```

---

## 11. 预计交付物

```text
ACL2_v9_window_drift_experiment_report.md
v9_run_registry.csv
v9_window_drift_state_audit.md
v9_projection_oracle_report.md
v9_proxy_drift_direction_report.md
v9_auto_window_v2_report.md
v9_head_route_hook_report.md
v9_next_stage_decision.md
```

`v9_next_stage_decision.md` 必须明确回答：

```text
1. TTT write-side 是否还有可用 action space？
2. oracle projection 是否打开了超过 34m 的空间？
3. no-GT proxy 是否足够支撑 full run？
4. auto-window 是否能替代 manual 5-9 / 10-12 / 16？
5. 是否停止 TTT write-side，转 read / pose-scale / SWA window source？
```

---

## 12. 最终判断

v8 不是没有进展。`update_conflict_energy + tri_replay` 已经把 TTT 写入从 `36.4m` 推到 `34.16m`，这是一条真实路线。

但 v8 后半段也已经清楚表明：继续围绕 chunk16、exit gamma、layer12 scalar、post-window scalar 做微调，收益非常低。当前最大问题不是少了一个更好的 gamma，而是 TTT controller 的状态变量错了：它还没有显式建模 reset-window drift state，也没有按 drift direction 分解 fast-weight residual。

下一阶段只有两个真正有价值的答案：

```text
A. oracle projection 成功：
    说明 TTT write-side 仍有空间，下一步做 no-GT drift proxy。

B. oracle projection 失败：
    说明当前 TTT write-side action space 接近上限，必须转 read-side / pose-scale / SWA window-level source。
```

这比继续把 `34.1583` 微调成 `34.12` 更重要。

# ACL2 v3：从单一 Cue 挖掘到 Cue 组合验证的可执行实验计划

日期：2026-05-05  
对象：LoGeR / HMC Pipeline v2 / KITTI01 为主开发集  
当前主线：LoGeR 内部 global-attention cue，尤其是 `acl2.gg.qq.low.g3.full.headmean.robustq`  
当前最强单 cue 结果：

```text
read-only probe_native:
acl2.gg.qq.low.g3.full.headmean.robustq
ATE / Rot = 39.1170 / 8.8160, beta=1.0

hybrid-safe probe_ttt_write:
acl2.gg.qq.low.g3.full.headmean.robustq
ATE / Rot = 38.4298 / 8.9846, beta=3.75
```

---

## 0. 先回答当前问题：现在是不是还主要在挖单一 cue？

是的。严格说，当前阶段仍然主要是在 **挖掘和验证单一 read cue**，还没有系统进入 **cue 组合验证**。

当前最强配置：

```text
acl2.gg.qq.low.g3.full.headmean.robustq + probe_ttt_write
```

这里包含两部分：

1. `acl2.gg.qq.low.g3.full.headmean.robustq` 是一个单一 attention read cue；
2. `probe_ttt_write` 是 commit-safe 的长期记忆写入协议，不应被算作 cue 组合。

也就是说，当前 best 本质上是：

```text
single internal-attention read cue + safe TTT write protocol
```

而不是：

```text
old_dyn + explicit_dyn + global query cue + deep static rescue + routing fusion
```

之前确实做过一些 fusion 或 mix，例如：

```text
old_dyn_addclip
old_dyn_soft_or
old_dyn_calibrated_soft_or
mix.old_route_gg_smd
mix.exp_add_fa_decay
mix.old_agree_fa_decay
```

但这些实验有两个限制：

1. 它们主要围绕旧 `old_dyn` / 早期 global cue / frame-attention cue 做组合，不是围绕当前最强 `acl2.g3.full` 做系统组合；
2. 它们大多是 add / clip / agree 这类局部规则，不是完整的 routing、static rescue、cue reliability、component attribution 实验。

因此，当前最准确的判断是：

> ACL2 v2 已经证明 `g3` 附近的 global query-query low-similarity 是强单 cue，但还没有系统验证 cue 组合是否能进一步突破 `38.43m` hybrid-safe 平台。

本计划的目标就是把实验从 **单 cue mining** 推进到 **有假设、有归因、有审计的 cue combination validation**。

---

## 1. 实验整体目标

本阶段的总目标不是继续盲目刷 ATE，而是回答下面四个科学问题：

### 问题 1：当前 best 的提升到底来自哪里？

当前 best 是：

```text
acl2.g3.full read cue + probe_ttt_write, beta=3.75
ATE / Rot = 38.4298 / 8.9846
```

但现在还没有完全拆清楚：

```text
改进来自更强 read suppression？
改进来自 safe TTT write？
改进来自 beta 变大？
改进来自 g3 层本身？
改进来自 full temporal support？
```

所以第一阶段必须先把 `g3.full` 的因果链拆清楚。

### 问题 2：`g3.full` 是最优单 cue，还是只是当前最强被测试配置？

`g3.full` 是一个复合配置：

```text
global stack layer g3
query-query low similarity
full temporal support
head-mean aggregation
robustq normalization
frame-attention read control
```

目前还没有完全解耦：

```text
layer index
layer window
support pattern
head aggregation
normalization
beta strength
```

因此在进入 cue 组合前，需要完成一轮最小必要的单 cue结构确认，避免在错误的 base cue 上做组合。

### 问题 3：`g3` attention cue 和旧几何 cue 是否互补？

当前 `g3.full` 与 `old_dyn` 的相关性很低：

```text
Corr old_dyn = 0.086
```

这说明两者可能捕捉的是不同信号。低相关性可能是好事，也可能是坏事：

```text
好事：两者互补，组合后可以进一步提升；
坏事：其中一个 cue 在大量区域是 false positive，组合后会过度 suppress。
```

因此下一阶段真正要验证的是：

> global query inconsistency 是否可以和 geometry inconsistency 形成互补，而不是互相污染。

### 问题 4：deep/global/static cue 是否应该作为 dynamic suppress，还是 static rescue？

深层 low-sim 作为 dynamic cue 明显失败：

```text
g13 / g17 / g13_15 / g12_17 low-sim dynamic suppress 都不推荐
```

但这不等于深层无用。更合理的假设是：

> early/mid global query inconsistency 适合发现当前 read path 的干扰区域；deep high-sim / low entropy / high support 更适合保护 static anchor，作为“不要 suppress”的 rescue cue。

因此本阶段必须把 deep cue 从 “dynamic suppress candidate” 改成 “static rescue candidate” 来验证。

---

## 2. 当前证据总结与独立判断

### 2.1 已经成立的结论

第一，非比例 layer window 的价值已经被证明。旧 `gg.qq.middle.low.robustq` read-only 为：

```text
39.6811 / 9.2540
```

而非比例 singleton `g3.full` read-only 达到：

```text
39.1170 / 8.8160
```

这说明按比例切 `shallow / middle / deep` 会漏掉 LoGeR 内部真正有效的 attention signal。

第二，`g3.full` 不是单点 beta 偶然。hybrid-safe beta sweep 中：

```text
beta = 3.25: 38.4754 / 8.9599
beta = 3.50: 38.4539 / 8.9772
beta = 3.75: 38.4298 / 8.9846
beta = 4.00: 38.4765 / 9.0092
```

`3.25-4.00` 均处于强平台，说明该 cue 具有稳定有效区间。

第三，当前 ATE 提升主要不像是局部 RPE 改善，而更像是全局 drift / scale / yaw trajectory shape 的改善。因为 RPE t / RPE r 在不同 beta 下变化很小，而 ATE、final error、50/100/200-frame segment mean ATE 同步改善。

第四，`probe_ttt_write` 仍然是必要协议。LoGeR 的设计本身就是 TTT 负责长程压缩记忆、SWA 负责局部高保真邻接对齐；read-path control 如果直接把 controlled forward 的 TTT side effect 提交到未来 memory，就会污染未来状态。因此后续所有 read cue 或 combination cue 都必须在 commit-safe 协议下判断。

### 2.2 尚未成立的结论

第一，不能说 `g3` 是最终最优层。因为目前 `g3.full` 和 `g2_6.off246` 的比较同时改变了 layer window 和 support pattern。必须补：

```text
g3.full vs g3.off246
g2_6.full vs g2_6.off246
g3.near12 vs g3.full
g3.past vs g3.future
```

第二，不能说 hybrid 比 read-only 一定带来 `0.6872m` 的纯 write gain。因为 read-only 当前只在 beta=1.0 下报告，而 hybrid best 在 beta=3.75。严格的写入收益应当同 beta 计算：

$$
\Delta_{write}(\beta) = ATE_{read}(\beta) - ATE_{hybrid}(\beta)
$$

例如 beta=1.0 时：

```text
read-only: 39.1170 / 8.8160
hybrid:    38.8598 / 8.7889
```

所以 beta=1.0 下 safe write 的真实增益约为：

$$
\Delta_{write}(1.0)=39.1170-38.8598=0.2572m
$$

而不是 `0.6872m`。

第三，不能说 cue quality 指标已经足够解释 ATE。`g3.full` 的 mass、anchor collision、fragmentation 并不是所有候选中最漂亮的，却是最强 ATE cue。这说明当前质量指标只能作为 reject gate，还不能作为 ranking predictor。

第四，不能说 cue 组合已经被验证。当前 best 是强单 cue，旧 fusion 不是围绕当前 best 设计的 systematic combination。

---

## 3. 本阶段核心假设

本阶段不再以“多跑一些 cue 名字”为目标，而是围绕以下假设组织实验。

### H1：`g3.full` 的有效性来自 early-mid global query manifold inconsistency

定义 `D_g` 为 `g3.full` 产生的 dynamic read cue。该假设认为：

> LoGeR 在早中层 global stack 中已经形成了一个相对稳定的静态几何 query manifold；偏离这个 manifold 的 patch token 更可能是当前 read path 中的干扰源。

如果 H1 成立，应观察到：

1. `g2/g3/g4` 附近的 cue 显著强于深层 `g13/g17` 的 low-sim cue；
2. `g3` 在多个 support pattern 下仍然有信号；
3. `g3` 的提升不是只依赖某一个 beta 尖峰；
4. `D_g` 高的 chunk 与 ATE/yaw drift 改善具有相关性。

### H2：`g3` attention cue 与 `old_dyn / explicit_dyn` 存在互补

定义：

```text
D_g    = g3 global query low-sim cue
D_old  = old_dyn_addclip cue
D_exp  = explicit_dyn_only cue
```

该假设认为：

> `D_g` 捕捉的是 LoGeR 内部 representation / attention manifold 的不一致，`D_old` / `D_exp` 捕捉的是几何重建一致性残差。它们来源不同，因此可以互补。

如果 H2 成立，应观察到：

1. `D_g` 与 `D_old` 的 disagreement 区域不是随机噪声，而集中在特定失败 chunk 或特定图像区域；
2. `D_g` high 且 `D_old` low 的区域，单独用 old_dyn 会漏掉部分干扰；
3. `D_old` high 且 `D_g` low 的区域，存在 old_dyn false positive，需要 routing 降权；
4. routing fusion 优于简单 add/clip。

### H3：深层 high-sim cue 更适合作为 static rescue，而不是 dynamic suppress

定义 `R_deep` 为深层 high-sim / low-entropy / high-support 的 static reliability cue。该假设认为：

> 深层 representation 更像稳定结构或 pose/global context 的后验，不适合直接找 dynamic token，但适合保护长期 static anchor，减少 over-suppression。

如果 H3 成立，应观察到：

1. `D_g` 加 deep static rescue 后，rotation 或 final error 改善；
2. ATE 不明显回退；
3. anchor collision 降低；
4. 被 rescue 的区域主要是 road/building/horizon/long-range static structure，而不是 moving objects。

### H4：cue combination 的收益必须在 read-only 和 hybrid-safe 两个层面都成立

该假设是实验协议假设：

> 一个 cue 组合如果只在 hybrid-safe 下提升，但 read-only 不提升，可能是 write protocol 或 beta 偶然；如果只在 read-only 提升，但 hybrid-safe 不提升，可能不适合长期 memory commit。

因此每个晋级组合必须同时报告：

$$
ATE_{read}(\beta),\quad ATE_{hybrid}(\beta),\quad \Delta_{write}(\beta)
$$

其中：

$$
\Delta_{write}(\beta)=ATE_{read}(\beta)-ATE_{hybrid}(\beta)
$$

### H5：组合 cue 更容易过拟合 KITTI01，因此必须有 cross-sequence sanity

当前开发集是 KITTI01，属于 open-loop、长距离、对 drift 敏感的序列，非常适合开发 cue。但如果不断在 KITTI01 上微调 routing 参数，组合 cue 很容易过拟合。因此：

> 只有在 KITTI01 上通过 strict gate 的少量组合，才进入 KITTI00/02/05 或 08 的 sanity check。

---

## 4. 实验总原则

### 4.1 先因果隔离，再做组合

不能直接从当前 best 开始做大规模 fusion。必须先回答：

```text
当前 best 是否可复现？
read beta alone 能否达到类似结果？
safe write 的同 beta 贡献是多少？
g3 强是 layer 强，还是 support 强？
```

只有这些问题清楚后，cue 组合结果才有归因意义。

### 4.2 `probe_ttt_write` 是 memory protocol，不是 cue 组合

后续报告中必须把三类变量分开记录：

```text
read cue: D_read
read strength: beta
commit/write protocol: controlled / probe_native / probe_ttt_write
write score source: stage_d / residual_reliability / etc.
```

不要把 `D_read + probe_ttt_write` 称为 cue fusion。

### 4.3 组合优先做 routing，不优先做简单 add/clip

简单相加的问题是：

```text
D_final = clip(D_a + D_b)
```

会把 false positive 也加起来，容易扩大 suppress 区域。更合理的是先做 quadrant routing：

```text
两个 cue 都高：强 suppress
D_g 高 D_old 低：保留但降权，看是否是 internal-only hidden interference
D_old 高 D_g 低：怀疑 old_dyn false positive，降权
两个都低：不 suppress
```

### 4.4 每个组合都必须有 component attribution

一个组合如果有效，必须回答：

```text
是 D_g 起作用？
是 D_old 起作用？
是 D_exp 起作用？
是 R_static rescue 起作用？
还是只是 beta/mass 改变造成的？
```

所以每个组合至少要配套以下消融：

```text
main cue only
aux cue only
simple add/clip
routing fusion
routing without rescue
routing with rescue
```

### 4.5 质量审计先于 full controlled run

短序列、局部 slice、甚至 full passive cue quality 都不能完全预测 ATE，但它们可以有效过滤明显坏 cue。任何组合进入 full controlled 前，必须先记录：

```text
mass
coverage
anchor collision
fragmentation
old_dyn / g3 / explicit agreement
image-zone distribution
temporal consistency
```

---

## 5. 数据集与固定协议

### 5.1 主开发集

主开发集继续使用：

```text
KITTI Odometry Sequence 01
```

原因是它是 open-loop、长距离、对 drift 和 scale/yaw 累积误差敏感，最适合检测 LoGeR 长期 memory 和 read correction 的变化。

### 5.2 Sanity check 集

在 KITTI01 上通过 strict gate 的候选，进入：

```text
KITTI00
KITTI02
KITTI05 或 KITTI08
```

选择原则：

```text
KITTI00 / 02: 长序列，含 loop 或大范围运动，用于泛化检查
KITTI05 / 08: 结构和运动模式不同，用于防止只适配 KITTI01
```

### 5.3 固定 HMC 协议

后续所有 cue 实验必须至少包含两种模式：

```text
read-only:
    output = controlled forward
    commit = probe_native

hybrid-safe:
    output = controlled forward
    commit = probe_ttt_write
```

关键 commit 对照只在少数 beta 点跑：

```text
controlled
probe_native
probe_ttt_write
```

原因：已有实验已经证明 controlled commit 会污染未来 TTT memory，后续不再把 controlled commit 当主评价协议。

### 5.4 固定 read path

当前主 read path 固定为：

```text
frame attention early layers
bias mode = pair
normalization = robustq
```

除非某个阶段明确做 read-path ablation，否则不同时改变 read path 和 cue 结构。

### 5.5 固定 baseline

每一批实验必须报告以下 baseline：

| 名称 | 说明 | 当前指标 |
|---|---|---:|
| `native LoGeR` | no control | `41.7502 / 8.9928` |
| `old_dyn_addclip` read | 旧 Phase F best read cue | `39.3103 / 9.7097` |
| `gg.qq.middle.low` hybrid | 旧 ACL v1 best | `38.9714 / 9.2084` |
| `acl2.g3.full` read-only | 当前 ACL2 best read-only | `39.1170 / 8.8160` |
| `acl2.g3.full` hybrid | 当前 ACL2 best hybrid-safe | `38.4298 / 8.9846` |

---

## 6. 必须记录的指标

本阶段所有 run 都必须记录四类指标：trajectory 指标、cue quality 指标、read effect 指标、TTT/write/memory 指标。

### 6.1 Trajectory 主指标

每个 full run 必须记录：

| 指标 | 记号 | 用途 |
|---|---|---|
| ATE RMSE | `ATE` | 主指标，衡量全局轨迹误差 |
| Rotation RMSE | `Rot` | 检查是否用旋转损伤换 ATE |
| RPE translation | `RPE_t` | 局部相对平移误差 |
| RPE rotation | `RPE_r` | 局部相对旋转误差 |
| final aligned error | `FinalErr` | 检查 endpoint 是否恶化 |
| 50-frame mean ATE | `ATE_50` | 短窗口误差 |
| 100-frame mean ATE | `ATE_100` | 中窗口误差 |
| 200-frame mean ATE | `ATE_200` | 长窗口误差 |
| yaw RMSE | `YawRMSE` | 判断 yaw drift 是否改善 |
| Sim(3) scale | `Sim3Scale` | 检查 scale drift 与 Umeyama 对齐变化 |

### 6.2 Per-chunk / per-frame 误差指标

每个 run 需要输出：

```text
per_chunk_error.csv
per_frame_error.csv
```

字段：

```text
chunk_id
frame_start
frame_end
chunk_ate_rmse
chunk_rot_rmse
chunk_final_error
chunk_yaw_error
chunk_scale_proxy
worst_frame_id
worst_frame_error
```

这些指标用于判断：

```text
组合 cue 是全局改善，还是只改善少数 chunk？
是否在某些 chunk 产生灾难性 over-suppression？
worst chunks 是否从旧 best 迁移到新位置？
```

### 6.3 Cue quality 指标

每个 cue 或 cue 组合都必须记录：

| 指标 | 记号 | 解释 |
|---|---|---|
| mean mass | `mean(D)` | cue 平均强度 |
| high mass ratio | `mean(D>0.5)` | 被强 suppress 区域比例 |
| p90 mass | `p90(D)` | 高强度尾部 |
| coverage | `coverage(D>tau)` | 每个 chunk 是否有足够覆盖 |
| fragmentation | `Frag` | cue 是否碎片化 |
| anchor collision | `AnchorCollide` | cue 是否误伤 high-anchor 区域 |
| confidence correlation | `Corr(D,Conf)` | cue 是否偏向高/低置信区域 |
| old_dyn correlation | `Corr(D,D_old)` | 与 old_dyn 的一致性 |
| explicit correlation | `Corr(D,D_exp)` | 与 explicit cue 的一致性 |
| g3 correlation | `Corr(D,D_g)` | 与 g3 cue 的一致性 |
| image-zone distribution | `ZoneMass` | cue 是否集中在 sky/road/horizon/lower image |
| temporal consistency | `TempCons` | 相邻帧 cue 是否稳定 |
| support concentration | `SupportConc` | support 是否过度集中在少数帧 |
| quadrant mass | `Q00/Q01/Q10/Q11` | 组合 cue 分歧区域比例 |

其中 fragmentation 建议定义为 patch-grid 上的边界密度：

$$
Frag(D)=\frac{1}{THW}\sum_{t,h,w}\mathbf{1}\left[|D_{t,h,w}-D_{t,h,w+1}|>\tau_f\right]+\mathbf{1}\left[|D_{t,h,w}-D_{t,h+1,w}|>\tau_f\right]
$$

temporal consistency 可定义为：

$$
TempCons(D)=1-\frac{1}{T-1}\sum_{t=1}^{T-1}\operatorname{Mean}_{h,w}|D_{t+1,h,w}-D_{t,h,w}|
$$

如果 patch token 不同帧之间没有像素对齐，也可以先用 same-pixel 近似，作为 diagnostic 而非强 gate。

### 6.4 Cue combination quadrant 指标

对于两个 cue `D_a` 和 `D_b`，定义：

$$
Q_{11}=\mathbf{1}[D_a>\tau_a]\mathbf{1}[D_b>\tau_b]
$$

$$
Q_{10}=\mathbf{1}[D_a>\tau_a]\mathbf{1}[D_b\le\tau_b]
$$

$$
Q_{01}=\mathbf{1}[D_a\le\tau_a]\mathbf{1}[D_b>\tau_b]
$$

$$
Q_{00}=\mathbf{1}[D_a\le\tau_a]\mathbf{1}[D_b\le\tau_b]
$$

每个组合必须记录：

```text
mass_Q11
mass_Q10
mass_Q01
mass_Q00
ATE_error_mean_on_Q11_chunks
ATE_error_mean_on_Q10_chunks
ATE_error_mean_on_Q01_chunks
AnchorCollide_Q10
AnchorCollide_Q01
```

这些指标用于判断：

```text
组合是否真的利用了 disagreement？
D_old-only 区域是否更像 false positive？
D_g-only 区域是否包含 hidden interference？
```

### 6.5 Read effect 指标

每个 controlled run 必须记录：

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
num_tokens_suppressed
suppressed_token_ratio
attn_shift_l1
attn_shift_l2
attn_mass_to_highD_before
attn_mass_to_highD_after
attn_mass_to_anchor_before
attn_mass_to_anchor_after
```

关键量：

$$
AttnShift=\operatorname{Mean}|A_{controlled}-A_{native}|
$$

以及：

$$
\Delta Attn_{highD}=Mass_{after}(D>\tau)-Mass_{before}(D>\tau)
$$

如果 cue 正常工作，应看到 high-D 区域的 attention mass 减少，而 anchor 区域不应显著减少。

### 6.6 TTT / write / memory 指标

每个 hybrid run 必须记录：

```text
ttt_write_summary.jsonl
hmc_state_hash.jsonl
memory_delta_summary.jsonl
```

字段：

```text
chunk_id
layer_id
branch_id
head_id
commit_mode
write_score_source
update_norm_native
update_norm_controlled
update_norm_probe_write
update_cosine_to_native
update_cosine_to_old_best
memory_state_rel_diff
memory_state_hash_before
memory_state_hash_after_probe
memory_state_hash_after_commit
prior_ttt_write_present
```

关键指标：

$$
UpdateCos=\frac{\langle \Delta W_{candidate},\Delta W_{native}\rangle}{\|\Delta W_{candidate}\|\|\Delta W_{native}\|+\epsilon}
$$

$$
MemDiff=\frac{\|W_{after}-W_{probe}\|_F}{\|W_{probe}\|_F+\epsilon}
$$

这些指标用于判断：

```text
组合 cue 是否只是改变 read output？
是否同时改变了未来 memory？
branch0 write 是否过强？
是否出现 controlled commit 污染？
```

### 6.7 运行稳定性与工程正确性指标

每批实验都必须记录：

```text
run_config.yaml
software_commit.txt
hmc_correctness_summary.json
identity_hook_check.json
runtime_summary.json
```

字段：

```text
pass1_pass2_pose_diff_max
pass1_pass2_point_diff_max
probe_no_commit_hash_equal
state_double_write_safe
max_identity_bias
num_chunks
num_failed_chunks
walltime_sec
peak_gpu_mem_gb
```

---

## 7. 必须可视化的内容

每个晋级候选必须输出可视化，不只看 ATE 表。

### 7.1 Cue map dashboard

对固定帧集合和 worst chunks 输出：

```text
RGB
D_g3
D_old
D_exp
R_deep_static
D_combined
suppression overlay
anchor map
confidence map
```

固定帧建议：

```text
每 100 帧采样 1 帧
每个 worst chunk 采样 2 帧
每个 largest-gain chunk 采样 2 帧
每个 largest-regression chunk 采样 2 帧
```

每张图要同时显示：

```text
cue heatmap
cue > 0.5 binary mask
anchor collision overlay
```

### 7.2 Disagreement quadrant map

对于组合 cue，必须可视化：

```text
Q11: D_g high, D_old high
Q10: D_g high, D_old low
Q01: D_g low,  D_old high
Q00: both low
```

颜色固定：

```text
Q11: red
Q10: orange
Q01: blue
Q00: transparent
static rescue: green outline
anchor collision: magenta outline
```

这个图是判断 routing 是否有意义的核心。

### 7.3 Trajectory comparison

每个晋级候选必须输出：

```text
GT trajectory
native LoGeR
old_dyn_addclip
acl2.g3.full current best
candidate
```

并生成三张 trajectory 图：

```text
full trajectory XY
first half trajectory XY
second half trajectory XY
```

同时标记：

```text
worst chunk positions
largest improvement chunk positions
largest regression chunk positions
```

### 7.4 Error-over-time 曲线

必须输出：

```text
per-frame aligned translation error
per-frame rotation error
cumulative yaw drift proxy
sliding 50/100/200-frame ATE
```

用来判断候选是：

```text
前半段改善？
后半段改善？
只改善 endpoint？
是否在某个 chunk 后突然崩？
```

### 7.5 Cue-quality vs error scatter

对每个 chunk 输出 scatter：

```text
mean(D_combined) vs chunk_ATE_delta
AnchorCollide vs chunk_ATE_delta
Frag vs chunk_ATE_delta
Q10_mass vs chunk_ATE_delta
Q01_mass vs chunk_ATE_delta
R_static_rescue_mass vs Rot_delta
```

其中：

$$
chunk\_ATE\_delta = chunk\_ATE_{candidate} - chunk\_ATE_{baseline}
$$

baseline 采用当前 best `acl2.g3.full hybrid b3.75` 或 read-only 对应 beta。

### 7.6 TTT/write diagnostic dashboard

hybrid run 必须可视化：

```text
per-layer update norm heatmap
per-branch update norm heatmap
update cosine to native heatmap
memory relative diff over chunks
write_gain(beta) curve
```

尤其要看 branch0 是否仍然主导有效改进，或者组合 cue 是否改变了其它 branch 的行为。

---

## 8. 实验阶段设计

本阶段分为 8 个阶段。前两个阶段仍是 single-cue 因果隔离和结构补全；从 Stage 3 开始才进入真正 cue 组合验证。

---

## Stage 0：Baseline repeat 与工程锁定

### 目标

确认当前 best 与所有 baseline 可复现，避免后续组合结果建立在不稳定代码上。

### 要跑的实验

| Run | Cue | Mode | Commit | Beta | 目的 |
|---|---|---|---|---:|---|
| S0-01 | none | native | controlled/no-control | n/a | 复现 LoGeR native |
| S0-02 | old_dyn_addclip | read/hybrid | probe_ttt_write | 1.25 | 复现旧 Phase F best |
| S0-03 | gg.qq.middle.low | hybrid | probe_ttt_write | 2.50 | 复现 ACL v1 best |
| S0-04 | acl2.g3.full | read-only | probe_native | 1.00 | 复现 ACL2 read-only best |
| S0-05 | acl2.g3.full | hybrid | probe_ttt_write | 3.75 | 复现 ACL2 hybrid best |
| S0-06 | acl2.g3.full | hybrid | probe_ttt_write | 3.75 | repeat，检查确定性 |

### 必须记录

```text
metrics_global.json
per_chunk_error.csv
cue_quality_summary.json
hmc_state_hash.jsonl
runtime_summary.json
```

### 判断标准

S0 通过条件：

```text
S0-04 ATE 与 39.1170 差异 <= 0.03m
S0-05/S0-06 ATE 与 38.4298 差异 <= 0.03m
identity / correctness hook 全部通过
无 failed chunks
```

如果 S0 不通过，不进入后续阶段。

---

## Stage 1：`g3.full` read/write 因果隔离

### 目标

拆清楚当前 best 中 read beta 与 safe write 的贡献。这个阶段仍然不是 cue 组合，而是为后续组合提供干净基线。

### 实验设计

固定 cue：

```text
D_g = acl2.gg.qq.low.g3.full.headmean.robustq
```

跑同一组 beta 的 read-only 与 hybrid-safe：

```text
beta = 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.75, 4.0
```

| Run group | Mode | Commit | Beta set |
|---|---|---|---|
| S1-R | read-only | probe_native | 1.0 to 4.0 |
| S1-H | hybrid-safe | probe_ttt_write | 1.0 to 4.0 |

另外在关键 beta 上跑 commit 对照：

```text
beta = 1.0, 2.5, 3.75
commit = controlled / probe_native / probe_ttt_write
```

### 记录的派生量

对每个 beta 计算：

$$
\Delta_{read}(\beta)=ATE_{native}-ATE_{read}(\beta)
$$

$$
\Delta_{write}(\beta)=ATE_{read}(\beta)-ATE_{hybrid}(\beta)
$$

$$
\Delta_{rot}(\beta)=Rot_{candidate}(\beta)-Rot_{baseline}
$$

### 判断标准

H4 成立的标准：

1. 在至少 3 个 beta 点上，`hybrid-safe` 优于同 beta 的 `read-only`；
2. `controlled commit` 在关键 beta 上不优于 `probe_native / probe_ttt_write`，或出现明显 memory contamination；
3. 最佳区间不是单点尖峰，至少存在宽度不小于 `0.5` 的 beta 平台；
4. `write_gain(beta)` 曲线可解释，不应出现只在一个 beta 上巨大跳变。

输出图：

```text
beta vs ATE: read-only and hybrid
beta vs Rot: read-only and hybrid
beta vs write_gain
beta vs final error
```

---

## Stage 2：`g3` 邻域单 cue 结构补全

### 目标

回答：`g3.full` 强，到底是 `g3` 层强，还是 `full support` 强，或者两者组合才强。

这个阶段仍然属于 single-cue completion，不是组合验证。

### Layer/window 候选

```text
g2
g3
g4
g2_3
g3_4
g2_4
g3_5
g2_6
```

### Support 候选

```text
full
off246
near12
near24
past_only
future_only
overlap_excluded
```

### 实验流程

先 passive audit，不做 controlled full：

```text
all layer/window × support candidates
```

通过 passive audit 的候选进入 read-only full。初始 gate：

```text
coverage >= 0.95
0.05 <= mean(D>0.5) <= 0.35
AnchorCollide <= 0.14
Frag <= 0.20
```

read-only full 只跑 top 12 个候选。

### 必须补的关键对照

无论 passive 排名如何，以下对照必须跑：

```text
g3.full
g3.off246
g3.near12
g3.near24
g3.past_only
g3.future_only
g2_6.full
g2_6.off246
g2_4.full
g3_4.full
```

### 判断标准

H1 成立的标准：

1. `g2/g3/g4` 邻域显著强于 deep low-sim dynamic cue；
2. `g3` 在至少两个 support pattern 下 read-only ATE 优于 `gg.qq.middle.low`；
3. 如果只有 `g3.full` 强而 `g3.off246/near12/past/future` 全弱，则说明 `full support` 是关键，后续组合要谨慎；
4. 如果 `g2_4.full` 或 `g3_4.full` 超过 `g3.full`，后续组合 base cue 改为该 window。

晋级标准：

```text
read-only ATE <= 39.20
Rot <= 9.10
或 ATE <= 39.35 且 Rot <= 8.80
```

---

## Stage 3：建立可组合 cue bank

### 目标

把所有可进入组合验证的 cue 固定成标准接口，避免每个组合临时改代码导致不可比。

### Cue bank 定义

第一版 cue bank 包含：

```text
D_g       : best single global query cue，初始为 g3.full
D_g_alt   : Stage 2 中第二强 global query cue
D_old     : old_dyn_addclip
D_exp     : explicit_dyn_only
D_imp     : implicit_dyn_only，仅 diagnostic
R_deep    : deep high-sim static rescue cue
R_anchor  : C_anchor / geometry anchor cue
R_conf    : confidence / low uncertainty reliability cue
```

所有 cue 必须映射到 patch-level：

$$
D \in [0,1]^{T\times H_{tok}\times W_{tok}}
$$

并使用相同 normalization：

```text
robustq
per-chunk calibration
clamp to [0,1]
```

### 统一保存格式

每个 chunk 保存：

```text
cue_bank/chunk_XXXX.npz
```

字段：

```text
D_g
D_g_alt
D_old
D_exp
D_imp
R_deep
R_anchor
R_conf
patch_meta
frame_ids
chunk_id
```

### 判断标准

Cue bank 通过条件：

1. 每个 cue 的 shape、frame order、patch order 完全一致；
2. 每个 cue 的 mean/mass/coverage 在 passive audit 中可复现；
3. `D_g` 复现 Stage 0/1 的 controlled 行为；
4. `D_old` 复现 old_dyn_addclip 行为；
5. `R_deep` 不直接作为 dynamic cue 跑主线，只进入 rescue 验证。

---

## Stage 4：Cue 组合的 passive audit

### 目标

先不跑 controlled full，只构造组合 cue 并审计其 mass、fragmentation、anchor collision、agreement/disagreement。这个阶段用于过滤明显危险的组合。

### 组合族 A：Agreement routing

定义两个主 cue：

```text
D_g
D_old
```

定义 threshold：

```text
tau_g = 0.5
tau_old = 0.5
```

构造 quadrant：

$$
Q_{11}=\mathbf{1}[D_g>\tau_g]\mathbf{1}[D_{old}>\tau_{old}]
$$

$$
Q_{10}=\mathbf{1}[D_g>\tau_g]\mathbf{1}[D_{old}\le\tau_{old}]
$$

$$
Q_{01}=\mathbf{1}[D_g\le\tau_g]\mathbf{1}[D_{old}>\tau_{old}]
$$

候选 routing：

$$
D_{route}=Q_{11}\max(D_g,D_{old})+Q_{10}\lambda_gD_g+Q_{01}\lambda_{old}D_{old}
$$

参数小矩阵：

```text
lambda_g   = 0.75, 1.00
lambda_old = 0.25, 0.50, 0.75
```

设计意图：

```text
D_g-only 区域保留较高权重，因为 g3 是当前最强；
D_old-only 区域默认降权，因为 old_dyn 可能包含几何残差 false positive；
agreement 区域强 suppress。
```

### 组合族 B：Explicit geometry rescue / replacement

用 `D_exp` 替代 `D_old` 或参与 old-only 区域：

$$
D_{route\_exp}=Q_{11}^{g,exp}\max(D_g,D_{exp})+Q_{10}^{g,exp}\lambda_gD_g+Q_{01}^{g,exp}\lambda_{exp}D_{exp}
$$

参数：

```text
lambda_g = 0.75, 1.00
lambda_exp = 0.50, 0.75, 1.00
```

原因：`explicit_dyn_only` ATE 稳定且 rotation 比 old_dyn 更好，适合作为几何分解 cue。

### 组合族 C：Deep static rescue

以 `D_g` 或 `D_route` 作为主 suppress cue，用 `R_deep` 做保护：

$$
D_{rescue}=\operatorname{clip}\left(D_{main}\cdot(1-\alpha R_{deep}),0,1\right)
$$

参数：

```text
alpha = 0.25, 0.50, 0.75
```

也可以只在 anchor 区域启用：

$$
D_{rescue\_anchor}=D_{main}\cdot\left(1-\alpha R_{deep}R_{anchor}\right)
$$

### 组合族 D：Reliability-weighted routing

构造 chunk-level 或 local reliability：

$$
R_{old}=\operatorname{clip}\left((1-Frag_{old}/q_f)(1-AnchorCollide_{old}/q_a),0,1\right)
$$

$$
R_g=\operatorname{clip}\left((1-Frag_g/q_f)(1-AnchorCollide_g/q_a),0,1\right)
$$

然后：

$$
D_{rel}=\frac{R_gD_g+R_{old}D_{old}}{R_g+R_{old}+\epsilon}
$$

第一版先用 chunk-level reliability，避免局部 reliability 噪声过大。

### Passive audit 通过标准

组合进入 read-only full 的条件：

```text
coverage >= 0.95
0.08 <= mean(D>0.5) <= 0.35
AnchorCollide <= max(0.13, AnchorCollide_g3 + 0.03)
Frag <= 0.18
Q11_mass 不小于 0.02，说明组合确实有 agreement 区域
Q10/Q01 任一不超过 0.25，避免单侧 disagreement 过大
```

如果某组合主要靠大幅降低 mass 获得看似更安全的质量指标，需要标记为 `mass-shifted`，后续 read-only 不能直接和原 cue 比较，必须使用相同 target mass 校准。

---

## Stage 5：Cue 组合 read-only full 验证

### 目标

验证组合 cue 是否在不写入未来 TTT side effect 的情况下改善当前 output。

### 固定协议

```text
mode = read-only
commit = probe_native
read path = frame attention early layers
bias = pair
normalization = robustq
```

### 候选数量控制

从 Stage 4 中最多选 12 个组合进入 read-only full：

```text
Agreement routing: top 4
Explicit routing: top 3
Deep rescue: top 3
Reliability routing: top 2
```

每个候选先跑 beta：

```text
beta = 1.0, 2.0, 3.0
```

如果任一 beta 接近或超过当前 read-only best，再局部细扫：

```text
beta = beta_best - 0.25, beta_best, beta_best + 0.25, beta_best + 0.50
```

### Read-only 晋级标准

组合 cue 必须满足至少一个 strong gate：

```text
Gate A: ATE <= 38.95，且 Rot <= 9.10
Gate B: ATE 比 g3.full read-only 同 beta 改善 >= 0.15m，且 Rot 恶化 <= 0.15deg
Gate C: Rot 比 g3.full read-only 改善 >= 0.25deg，且 ATE 不差于 g3.full read-only + 0.05m
```

此外必须满足：

```text
FinalErr 不比 g3.full read-only 差超过 0.50m
ATE_50/100/200 至少两个窗口优于 baseline
RPE_t/RPE_r 不出现明显恶化
```

### 必须做的 component attribution

任何通过 read-only gate 的组合，都必须补：

```text
D_g only, same beta
D_old or D_exp only, same beta
simple add/clip, same beta
routing without rescue, same beta
routing with rescue, same beta
```

判断组合真实有效的标准：

```text
routing/rescue 版本优于所有单分量；
不是简单 mass 降低造成；
不是 beta 偶然；
不是只改善一个局部 slice。
```

---

## Stage 6：Cue 组合 hybrid-safe 验证

### 目标

只让 read-only 通过 gate 的组合进入 hybrid-safe，验证它们是否适合长期 TTT memory commit。

### 固定协议

```text
mode = hybrid-safe
commit = probe_ttt_write
write score source = stage_d initially
```

### 同 beta 对照

每个候选必须跑：

```text
read-only beta_best
hybrid beta_best
```

必要时再跑：

```text
beta_best - 0.25
beta_best + 0.25
```

计算：

$$
\Delta_{write}(\beta)=ATE_{read}(\beta)-ATE_{hybrid}(\beta)
$$

### Hybrid 晋级标准

组合 cue 在 hybrid-safe 下要超过当前 best：

```text
Current best hybrid: 38.4298 / 8.9846
```

强晋级标准：

```text
ATE <= 38.25
Rot <= 9.05
FinalErr <= 4.90
```

弱晋级标准：

```text
ATE <= 38.35
Rot <= 8.95
FinalErr <= 5.10
```

如果 ATE 只改善小于 `0.05m`，不算实质突破，除非 rotation 或 final error 有明显改善：

```text
Rot 改善 >= 0.20deg
或 FinalErr 改善 >= 0.40m
```

### 写入安全标准

hybrid 候选必须满足：

```text
probe_ttt_write hash 正常
controlled commit 对照不作为主结果
memory_state_rel_diff 不出现异常尖峰
branch0 update norm 不超过当前 best 的 1.25x
update cosine to native 不低于当前 best - 0.10，除非 ATE 明显提升
```

如果某组合 hybrid 比 read-only 变差，标记为：

```text
read-useful but write-incompatible
```

后续可以作为 read cue 保留，但不进入 TTT write policy 主线。

---

## Stage 7：Static rescue 专项诊断

### 目标

验证 H3：deep cue 是否适合作为 static rescue。

### 候选 rescue cue

```text
R_deep_g13_high
R_deep_g17_high
R_deep_g13_17_high
R_deep_low_entropy
R_deep_high_support
R_anchor
R_deep * R_anchor
```

### 主 suppress cue

```text
D_main = D_g
D_main = best routing from Stage 5
```

### 公式

基础 rescue：

$$
D_{final}=D_{main}(1-\alpha R_{deep})
$$

anchor-gated rescue：

$$
D_{final}=D_{main}(1-\alpha R_{deep}R_{anchor})
$$

confidence-gated rescue：

$$
D_{final}=D_{main}(1-\alpha R_{deep}R_{conf})
$$

### 参数

```text
alpha = 0.25, 0.50, 0.75
```

### 判断标准

Static rescue 成立的标准：

```text
Rot 改善 >= 0.15deg
或 FinalErr 改善 >= 0.30m
同时 ATE 回退 <= 0.08m
AnchorCollide 降低 >= 10%
被 rescue 区域不是 moving object 主导
```

如果 rescue 主要降低了 suppress mass，但没有改善 Rot/FinalErr，则不保留。

---

## Stage 8：Cross-sequence sanity check

### 目标

验证组合 cue 是否只是 KITTI01 过拟合。

### 进入条件

只有满足以下条件的候选进入跨序列：

```text
KITTI01 hybrid ATE <= 38.35
或 KITTI01 hybrid ATE <= 38.45 且 Rot/FinalErr 明显优于 current best
```

最多进入 3 个候选：

```text
current best g3.full
best routing combination
best rescue combination
```

### 数据集

```text
KITTI00
KITTI02
KITTI05 或 KITTI08
```

### 评价方式

不要求每个序列都绝对最优，但要求：

```text
平均 ATE 优于 g3.full baseline
没有任何序列 ATE 恶化超过 5%
Rot 平均不恶化超过 3%
至少一个长序列显示稳定 gain
```

如果某组合只在 KITTI01 提升，其他序列明显恶化，则标记为：

```text
KITTI01-specific, not promoted
```

---

## 9. 实验输出目录规范

建议目录：

```text
results/kitti01_hmc_v2/acl2_v3_cue_combination/
```

每个 run 目录必须包含：

```text
run_config.yaml
metrics_global.json
kitti_benchmark.log
01.txt
per_chunk_error.csv
per_frame_error.csv
cue_quality_summary.json
cue_quality_per_chunk.jsonl
cue_component_quadrants.jsonl
read_effect_summary.jsonl
ttt_write_summary.jsonl
memory_delta_summary.jsonl
hmc_state_hash.jsonl
runtime_summary.json
visual_index.md
```

可视化目录：

```text
visuals/
    cue_grid/
    quadrant_maps/
    trajectory/
    error_curves/
    scatter/
    ttt_write/
    failure_gallery/
```

汇总表：

```text
acl2_v3_run_registry.csv
acl2_v3_promotion_table.md
acl2_v3_failure_cases.md
```

`run_registry.csv` 至少包含：

```text
run_id
cue_family
cue_name
components
mode
commit_mode
beta
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
mean_mass
high_mass
coverage
anchor_collision
frag
corr_old_dyn
corr_explicit
corr_g3
Q11_mass
Q10_mass
Q01_mass
rescue_mass
promotion_status
notes
```

---

## 10. 最终判断标准

### 10.1 单 cue 补全成功标准

如果 Stage 1-2 完成后发现：

```text
某个 g3 邻域单 cue read-only <= 39.00
或 hybrid <= 38.35
```

则说明 single-cue mining 仍未完成，组合实验应以这个新单 cue 作为 base。

### 10.2 组合 cue 成功标准

组合 cue 被认为真正成功，必须同时满足：

```text
1. 超过 current best 或在 Rot/FinalErr 上形成明确 Pareto improvement；
2. component attribution 证明不是单分量或 beta 偶然；
3. cue quality 没有明显更坏；
4. trajectory / segment metrics 同步改善；
5. hybrid-safe 不破坏 commit safety；
6. 至少通过一个 cross-sequence sanity check。
```

强成功标准：

```text
KITTI01 hybrid ATE <= 38.25
Rot <= 9.05
FinalErr <= 4.90
cross-sequence average 不退化
```

弱成功标准：

```text
KITTI01 hybrid ATE <= 38.35
且 Rot 或 FinalErr 明显优于 current best
```

### 10.3 停止标准

如果以下情况出现，应停止当前组合族：

```text
同族 6 个以上 full run 都不能超过 g3.full read-only/hybrid；
所有提升都来自 mass 大幅下降但 segment metrics 不改善；
AnchorCollide 或 Frag 系统性恶化；
hybrid-safe 总是比 read-only 差，说明 write-incompatible；
跨序列 sanity 明显退化。
```

---

## 11. 建议的第一批运行顺序

### Batch A：因果隔离

```text
A1: g3.full read-only beta 1.0/1.5/2.0/2.5/3.0/3.5/3.75/4.0
A2: g3.full hybrid    beta 1.0/1.5/2.0/2.5/3.0/3.5/3.75/4.0
A3: commit mode controlled/probe_native/probe_ttt_write at beta 1.0/2.5/3.75
```

### Batch B：single-cue 邻域补全

```text
B1: g3 support sweep full/off246/near12/near24/past/future/overlap_excluded
B2: g2/g3/g4/g2_3/g3_4/g2_4/g3_5/g2_6 full support
B3: g2_6 full vs off246 direct compare
```

### Batch C：cue bank passive audit

```text
C1: export D_g, D_old, D_exp, R_deep, R_anchor, R_conf
C2: compute cue quality and quadrant tables
C3: generate cue grid and quadrant maps
```

### Batch D：组合 passive audit

```text
D1: agreement routing D_g + D_old
D2: agreement routing D_g + D_exp
D3: D_g with deep static rescue
D4: D_route with deep static rescue
D5: reliability-weighted D_g / D_old
```

### Batch E：组合 read-only full

```text
E1: top 12 passive candidates, beta 1.0/2.0/3.0
E2: local beta sweep for candidates passing read-only gate
E3: component attribution for passing candidates
```

### Batch F：组合 hybrid-safe full

```text
F1: passing read-only candidates, same beta hybrid-safe
F2: local beta sweep around best beta
F3: TTT/write diagnostics and memory safety audit
```

### Batch G：cross-sequence sanity

```text
G1: current best g3.full on KITTI00/02/05 or 08
G2: best routing combination on same sequences
G3: best rescue combination on same sequences
```

---

## 12. 预期可能出现的结果与解释

### 情况 A：组合 cue 明显超过 current best

如果某组合达到：

```text
ATE <= 38.25
Rot <= 9.05
```

并通过 attribution，则说明：

> global query inconsistency 与 geometry residual / static rescue 真正互补，下一阶段可以进入 TTT write policy 的 branch/layer-specific 优化。

### 情况 B：组合 read-only 提升，但 hybrid 不提升

说明该组合适合当前 output correction，但不适合长期 memory commit。保留为 read cue，但不要直接进入 write policy。

### 情况 C：组合 hybrid 提升，但 read-only 不提升

优先怀疑：

```text
beta confound
write score confound
commit side effect
mass shift
```

必须补同 beta read-only 和 component attribution，不能直接认定 cue combination 成功。

### 情况 D：deep rescue 改善 Rot/FinalErr，但 ATE 小幅回退

这可能是有价值的 Pareto candidate。保留为 balanced branch，尤其适合后续 TTT write policy 或 endpoint-sensitive evaluation。

### 情况 E：所有组合都不超过 g3.full

这不是失败，而是说明：

> 当前最强方向仍是 single attention cue mining，需要继续做 per-head、真实 attention-map statistics、layer evolution，而不是复杂 fusion。

---

## 13. 最终交付物

本阶段结束后必须交付四个文件：

```text
acl2_v3_experiment_report.md
acl2_v3_run_registry.csv
acl2_v3_cue_combination_dashboard.md
acl2_v3_next_stage_decision.md
```

其中 `acl2_v3_next_stage_decision.md` 必须明确回答：

```text
1. 当前是否仍是 single-cue 最强？
2. 是否存在通过 gate 的 cue combination？
3. 最优组合的有效分量是什么？
4. 是否进入 TTT write policy 阶段？
5. 如果不进入，下一步是 per-head / attention-map / layer-evolution 哪一条？
```

---

## 14. 本计划的核心结论

当前阶段确实还主要是在挖单一 cue。`acl2.gg.qq.low.g3.full.headmean.robustq` 证明了 global query inconsistency 是真信号，但它还不是 cue 组合验证的终点。

下一阶段应该先完成：

```text
g3 因果隔离
邻域 layer/support 单 cue 补全
cue bank 标准化
```

然后再进入：

```text
D_g + D_old / D_exp routing
D_g + deep static rescue
reliability-weighted routing
read-only full validation
hybrid-safe validation
cross-sequence sanity
```

只有当组合 cue 同时通过 read-only、hybrid-safe、component attribution、cue quality、trajectory visualization 和 cross-sequence sanity，才能说：

> Attention Cue Library v2 不只是找到了一个强单 cue，而是真正找到了可组合、可解释、可推广的 LoGeR internal cue policy。

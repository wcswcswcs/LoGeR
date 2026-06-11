# ACL2 v52：C9 去 chunk-id、组件归因、自适应 TTT 写入与语义几何闭环计划

日期：2026-06-08  
对象：LoGeR / Pipeline v2 / HMC / KITTI Odometry Sequence 01  
本轮优先级：**先回答 C9 的机制问题和 adaptive TTT 写入问题，再推进语义几何目标。**

---

## 0. 项目边界与本轮硬要求

本项目的目标不是在某个数据集上靠手工规则打榜，而是构造一个 **training-free、online、可迁移、可解释的 LoGeR memory-control pipeline**。LoGeR 将长视频分成 chunk 处理，并使用两类记忆：

- **READ / frame-global attention path**：当前 chunk 推理时，从哪些 token / source frame 读取信息。
- **SWA，Sliding Window Attention**：短期局部记忆，负责相邻 chunk 的连续和 overlap 对齐。
- **TTT，Test-Time Training fast weights**：长期压缩记忆，负责全局坐标系、尺度和长程轨迹稳定。

本轮不再把目标定义为马上把 KITTI01 ATE 推到 30m 以下。本轮 top priority 是回答三个更基础的问题：

1. **C9_P0_R2 的组件贡献到底是什么？** 需要知道 READ、TTT、SWA 各自贡献多少，READ+TTT、READ+SWA 的交互是什么。
2. **C9 中所有 absolute chunk-id policy 必须移除或标记为 diagnostic-only。** 不允许再使用 `chunk5-9`、`chunk10-12`、`chunk16` 这种人工指定 chunk 的正式策略。
3. **TTT writing 的 tri replay 不允许再用手工指定 positive / neutral / negative percentage。** 必须使用自适应算法替代固定 `positive=0.35`、`negative=0.12`、`neutral=0.85`。本轮 adaptive TTT 的目标是接近 C9_P0_R2，不要求马上超过 C9，也不要求马上 ATE < 30m。

同时保留第二目标：**思考并验证语义信息如何真实改善几何重建**。但语义目标不能再用大规模 all-memory semantic matrix 盲跑，必须基于已经发现的代码问题和历史证据，缩小成可验证的机制。

### 0.1 禁止项

本轮禁止下面这些做法：

```text
禁止训练 trigger / selector / classifier / learned router
禁止用 oracle label 拟合规则
禁止使用 absolute chunk id 作为 runtime 条件
禁止针对 KITTI01 或其它 sequence 单独调参
禁止把 short rollout / fixed-window diagnostic 写成 deployable success
禁止继续扩大 semantic all-memory 大矩阵
禁止继续用 fixed positive / negative percentage 作为 adaptive TTT 的核心
```

### 0.2 接受小幅 ATE 回退，但必须有解释

本轮允许 no-chunk / adaptive TTT 比 C9 小幅回退。可接受目标为：

```text
C9_P0_R2: 33.7629421029m
Adaptive close pass: ATE <= 34.30m
Adaptive soft pass:  ATE <= 34.60m
Fail:                ATE > 34.60m
```

如果 adaptive TTT 仍无法接近 C9，必须回答为什么：是 role masks 错、risk proxy 错、split replay 语义丢失、gamma/lambda 能量不匹配，还是 C9 的收益主要来自 chunk-id schedule / commit EMA / native mix 交互。

### 0.3 效率硬门槛

KITTI01 full rollout 不能再被当作默认筛选工具。只要单条 full run wall time 达到几十分钟级别，就必须先处理效率问题，再继续扩大实验矩阵。

硬要求：

```text
1. 每个 full run 必须落盘 timing_summary.json 和 wall_time_summary.json。
2. 每批 full run 后必须运行 tools/v52_runtime_profile_report.py。
3. 任何新 full candidate 必须先通过 96F smoke + runtime projection。
4. projected full wall time > 30min 或 chunk_total_mean > 30s 时，不允许进入 full，除非该 run 是唯一必要的审计复现。
5. full run 不允许作为阈值网格搜索手段；只能用于已经通过机制、runtime、slice/global sanity gate 的少量候选。
```

v52 当前 DONE R2/no-SWA 的本地 timing 显示单条 full 约 24.5-26.6min，chunk mean 37-40s；如果实际调度环境出现 61-62min wall time，则直接视为效率 blocker。无论采用哪个数字，当前执行策略都太重，后续必须先降低 full run 频率并记录阶段耗时。

---

## 1. 本轮代码审计结论与必须修复的问题

### 1.1 当前代码包可审计，但不是完整独立复现包

`acl2_v45_v51_code_audit_packet_20260608_2257.zip` 包含了本轮相关的主入口、HMC、TTT controller、semantic prior、VideoMasklet、v45-v50 launcher/report 和关键结果目录。独立 `py_compile` 检查通过，shell launcher `bash -n` 也通过。它足够做 v45-v50 代码审计和结果解释。

但它不是完整复现包，还缺少若干 supporting files，例如：

```text
run_pipeline_abc.py
run_geometry_backbone_inference.py
inference_dynamic_cue_extractor.py
loger/utils/rotation.py
tools/kitti_trajectory_diagnostics.py
eval/long_eval_script/kitti_benchmark
完整 checkpoint/config/data layout 说明
```

这些缺失不会推翻 v45-v50 已落盘结果，但会影响新的编程 AI 从零复现实验。因此 Codex 必须在 v52 输出一个补全版 code audit pack，至少包括上述 supporting files 或明确说明它们来自原仓库路径。

### 1.2 必须修复的代码问题

#### 问题 A：`past_plus_future_light_real` 在 C23/QQ 路径里疑似实现错误

`_acl2_support_indices` 接受 `past_plus_future_light_real`，但在 `_global_acl2_centroid_metric` 的 per-frame `qq/qk` 分支里，weighted past/future 逻辑没有覆盖 `_real` alias。结果是：

```text
acl2.gg.qq.low.g2_3.past_plus_future_light_real.headmean.robustq
```

很可能退化成 full support，而不是计划中的 `past 0.75 + future 0.25`。这会影响 v45 Phase 3 中 `past_plus_future_light_real` 的解释。

修复要求：

```text
1. qq / qk / kk 三个分支都必须统一支持 past_plus_future_light_real。
2. 修复后必须通过 support-index unit audit：对每个 frame 输出 past_count / future_count / weight_sum。
3. 修复前的 S5_PAST_PLUS_FUTURE_LIGHT 结果只能标记为 suspect，不得用它否定 light future support。
```

#### 问题 B：adaptive writer 的 debug 字段容易误读

`tri_replay` 路径里仍可能写 `ttt_two_replay_applied=True`，这会污染报告语义。后续报告不得用 `two_replay_applied` 判断 tri replay 是否生效，必须优先看：

```text
ttt_tri_replay_applied
ttt_tri_replay_role_mode
ttt_tri_replay_pos_mass / neu_mass / neg_mass
adaptive_writer_split_debug_count / fused_debug_count
```

#### 问题 C：v45 的 `adaptive_quantile` 不满足当前硬要求

v45 的 `adaptive_quantile` 仍固定 positive fraction 为 `0.35`，只自适应 negative fraction。因此它不能算真正替代手工 percentage。v47-v50 的 `adaptive_writer_*` 系列才是当前应审计的 adaptive TTT 方向。

#### 问题 D：fused adaptive writer 与 split adaptive writer 语义不同

v47-v49 的 fused writer 把 positive / neutral / negative 合成一次 replay，v50 的 split writer 恢复正/中/负三分支 replay。v50 明显优于 v49，说明 **split replay 的分支语义是核心，不是实现细节**。后续 adaptive TTT 必须优先保留 split replay，不再把 fused 作为主线。

---

## 2. 目前实验结果总结

### 2.1 C9_P0_R2 当前状态

当前历史 best deployable online TTT-write 是：

```text
C9_P0_R2 ATE = 33.7629421029m
```

但 C9 不是语义策略。它的 Stage C semantic 是关闭的，核心是：

```text
READ cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
write score = stage_d * sqrt(1 - D_g)
TTT write = tri_replay
risk = update_conflict_energy
branch = w0
manual positive/negative/neutral = 0.35 / 0.12 / 0.85
chunk-specific gamma / commit EMA / read beta map
```

因此 C9 不能作为“语义帮助 memory management”的成功证据。它只是当前 HMC / TTT-write baseline。

### 2.2 v46B：READ / TTT / SWA positive-only factorial

v46B 做了 clean no-chunk 8-row factorial：

```text
F000_NONE
F100_ONLY_FRAME_ATTN
F010_ONLY_TTT
F001_ONLY_SWA
F110_FRAME_ATTN_TTT
F101_FRAME_ATTN_SWA
F011_TTT_SWA
F111_ALL_THREE
```

关键结果：

| Row | 含义 | ATE | Gain vs F000 |
|---|---|---:|---:|
| F000 | 无 READ / TTT / SWA | 41.7502 | 0.0000 |
| F100 | 只开 READ / frame attention | 38.5934 | 3.1568 |
| F010 | 只开 TTT tri replay | 39.5191 | 2.2311 |
| F001 | 只开 SWA | 41.7370 | 0.0132 |
| F110 | READ + TTT | 36.6689 | 5.0813 |
| F101 | READ + SWA | 38.5926 | 3.1576 |
| F011 | TTT + SWA | 39.5035 | 2.2467 |
| F111 | READ + TTT + SWA | 36.6507 | 5.0995 |

解释：

```text
READ 单独贡献很大。
TTT 单独贡献也大。
SWA 单独几乎没有贡献。
READ + TTT 有明显 synergy，约 +1.92m。
READ + SWA 几乎没有 synergy。
TTT + SWA synergy 很小。
```

这回答了“READ/TTT/SWA 谁重要”的一半：在 clean no-chunk factorial 下，最重要的是 READ 和 TTT，SWA 很小。

但要注意：v46B 是 clean no-chunk factorial，不是 exact C9_P0_R2。它关闭了 C9 中的 chunk-specific gamma / commit EMA 等手工配方。因此它解释的是基础组件贡献和交互，不是 exact C9 的全部贡献。

### 2.3 v45：C9 exact / C9-Clean / leave-one-out attribution

v45 已确认 C9 精确复现：

```text
C9 repeat ATE = 33.76294210291885m
```

同时 v45 发现简单 dechunk fixed-value 会明显退化。C9-Clean / D7 大约为：

```text
D7 C9-Clean ATE = 35.5005m
Delta vs C9 = +1.7376m
```

v45 的 leave-one-out / contribution ledger 给出：

```text
remove TTT tri-replay      -> +2.4469m
remove tri gamma chunk map -> +0.9710m
remove commit EMA          -> +0.4884m
remove native mix          -> +0.0917m
remove SWA overlap replace -> +0.0563m
remove read beta map       -> +0.0266m
```

解释：

```text
C9 的 exact full ATE 优势主要来自 TTT tri-replay、chunk gamma map 和 commit EMA。
READ beta chunk map 本身贡献很小。
SWA overlap replacement 在 full ATE 上贡献也很小。
```

这说明：如果必须去掉 chunk-wise 手工参数，就不能简单把 C9 的 chunk maps 设成固定值后期待性能不变。需要用 adaptive TTT writing 重新补回 TTT tri-replay / gamma / EMA 的有效行为。

### 2.4 v47-v50：adaptive TTT writing 当前失败在哪里

v47-v50 的共同硬约束：

```text
no chunk-wise policy = pass
manual tri replay positive / negative / neutral fractions = 0
external gamma = 0
role mode = adaptive_writer_* 系列
```

结果：

| 版本 | 方法 | Best AW111 ATE | Delta vs C9 | 结论 |
|---|---|---:|---:|---|
| v47 | Otsu fused, d_tok risk | 38.3221 | +4.5592 | positive mass 过大，gamma 太小 |
| v48 | robust fused, d_tok risk | 38.3395 | +4.5765 | mass/gamma 修正但轨迹仍差 |
| v49 | robust fused, residual_x_dg risk | 38.1722 | +4.4093 | risk proxy 小幅更好，仍失败 |
| v50 | robust split, residual_x_dg risk | 35.9853 | +2.2224 | split replay 恢复大量性能，但仍离 C9 很远 |

核心解释：

```text
1. fused single replay 丢失了 fixed tri replay 的正/中/负三分支更新语义。
2. split replay 是必要条件；v50 比 v49 明显更好。
3. 但 residual_x_dg risk + robust thresholds 仍不能复现 C9 的 update_conflict_energy 行为。
4. adaptive gamma / neutral lambda 虽然自适应，但没有匹配 C9 teacher 的 post-zp delta norm / branch contribution。
5. v50 还没有恢复 C9 的 commit EMA / chunk gamma / native mix 交互，剩余差距约 2.22m。
```

所以当前 adaptive TTT 失败不是因为“没有接上”，而是因为 adaptive role / risk / delta-energy 还没有学到 C9 真正有效的写入行为。

### 2.5 语义方向的真实状态

历史语义实验说明：

```text
语义工程通路已经基本可用：VideoMasklet cache、fine labels、role streams、path consumption、action influence audit 都逐步跑通。
语义直接控制 TTT/SWA/all-memory 很弱。
语义最有希望的方向是 READ / C23 reconditioning，即用语义解释 D_g，而不是直接写 TTT。
```

关键事实：

```text
v31 semantic-conditioned C23 short rollout 有强局部信号，但 full online 失败。
v40/v41 health-gated READ 有局部强信号，但 v42 full-online health-selected READ 失败。
v45 semantic residual READ 最好 X3_L050 到 33.1936m，比 C9 改善 0.5693m，是当前最好的语义辅助 full-online 增益。
```

因此目标 2 的正确方向不是“语义全局控制所有 memory”，而是：

```text
语义作为 cue context：解释 C23/D_g。
语义作为 source guard：只处理 high-D + high-source-influence + semantic/appearance anomaly 的 READ source。
语义作为 TTT 约束：只在 TTT conflict/scale risk 明确异常时辅助 no-long-write / short-negative，而不是直接 semantic scalar write。
```

---

## 3. 本轮总目标

本轮分为两个目标，但优先级不同。

### 3.1 目标 1：得到一版 no-chunk adaptive TTT 算法，性能接近 C9

固定：

```text
frame attention read cue 沿用 C9_P0_R2 不变：
acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path / frame-attn READ control 保留 C9 主线形式
```

硬约束：

```text
1. 不允许 absolute chunk-id policy。
2. 不允许手工指定 tri replay positive / negative / neutral percentage。
3. adaptive writer 必须使用当前 chunk/layer/token 的 online risk / update statistics 自适应决定角色和强度。
4. 算法必须 full-online 运行，不能是 short oracle。
```

目标：

```text
Close pass: ATE <= 34.30m
Soft pass:  ATE <= 34.60m
Fail:       ATE > 34.60m
```

### 3.2 目标 2：语义真实帮助几何重建，长期目标 KITTI01 ATE 30m

本轮目标 2 不再大范围探索，只做最小验证：

```text
1. 在目标 1 的 clean adaptive baseline 上，复测语义 residual READ。
2. 如果语义仍能提供 >=0.3m full-online gain，则保留为下一阶段 Target-30 主线。
3. 如果语义 full-online gain 消失，说明 v45 的语义增益依赖 C9 exact / chunk-map 交互，需要回到语义机制归因。
```

---

## 4. Phase 0：代码修复与硬审计

### 4.1 目标

先修复会影响实验解释的代码问题，保证后续结果不会被错误 alias / debug / 包缺失污染。

### 4.2 必做项

Codex 必须完成：

```text
1. 修复 past_plus_future_light_real 在 qq/qk/kk 中的 weighted support。
2. report 中不再用 ttt_two_replay_applied 判断 tri replay。
3. 确认 adaptive_writer_robust_split 的 debug 字段完整落盘。
4. 补充 code audit packet 缺失 supporting files。
5. 对 run_pipeline_abc_v2.py、hybrid_memory_controller.py、ttt_write_controller.py、launcher、report 做 py_compile / bash -n。
```

### 4.3 输出

```text
phase0_code_audit/bugfix_report.md
phase0_code_audit/support_alias_unit_audit.csv
phase0_code_audit/adaptive_writer_debug_field_audit.json
phase0_code_audit/code_packet_completeness_audit.md
```

### 4.4 通过标准

```text
py_compile failures = 0
bash -n failures = 0
past_plus_future_light_real support weights pass for qq/qk/kk
tri replay debug fields unambiguous
```

如果 Phase 0 不通过，不允许启动新的 full run。

---

## 5. Phase 1：C9 贡献归因最终确认

### 5.1 目标

把 C9 的贡献拆清楚，回答：READ、TTT、SWA 以及 READ+TTT / READ+SWA 到底贡献多少。

### 5.2 已有证据整合

Codex 不需要重复所有 v46B / v45 run，但必须重新生成一份统一 attribution report，包含：

```text
1. v46B positive-only factorial：F000/F100/F010/F001/F110/F101/F011/F111。
2. v45 exact C9 leave-one-out contribution ledger。
3. 二者差异解释：clean no-chunk factorial vs exact C9 knockout。
```

### 5.3 如果缺项则补跑

必须确认以下 rows 都存在且 valid：

```text
F000_NONE
F100_ONLY_FRAME_ATTN
F010_ONLY_TTT
F001_ONLY_SWA
F110_FRAME_ATTN_TTT
F101_FRAME_ATTN_SWA
F011_TTT_SWA
F111_ALL_THREE
```

如果任何 row 缺失或 invalid，必须补跑。

### 5.4 记录指标

对每个 row 记录：

```text
ATE / Rot / FinalErr / RPE_t / RPE_r
[200,300) / [400,600)
rolling50 / rolling100 / rolling200 mean/p90/worst
hmc_rows / frames
frame_attn_read_control_active
ttt_tri_replay_applied_count
tri role mass
SWA overlap replace applied count
no_chunk_policy_pass
```

### 5.5 输出

```text
phase1_c9_attribution/c9_component_attribution_report.md
phase1_c9_attribution/positive_only_factorial_table.csv
phase1_c9_attribution/exact_c9_knockout_table.csv
phase1_c9_attribution/component_main_effects.png
phase1_c9_attribution/interaction_heatmap.png
```

### 5.6 判断标准

本 Phase 不要求 ATE 下降，要求把问题回答清楚：

```text
必须给出 READ-only, TTT-only, SWA-only, READ+TTT, READ+SWA, TTT+SWA, all-three 的 ATE 和 gain。
必须给出 exact C9 移除 TTT tri-replay / tri gamma / commit EMA / read beta / SWA / native mix 的退化。
必须明确说明：哪些是 clean no-chunk factorial，哪些是 exact C9 knockout。
```

---

## 6. Phase 2：自适应 TTT 失败机制审计

### 6.1 目标

不再直接猜新公式。先理解 v50 adaptive split 为什么还差 C9 2.22m。

### 6.2 对比对象

```text
Teacher 1: C9_P0_R2
Teacher 2: v46B F111 fixed-percentage no-chunk all-three
Student 1: v50 AW111 adaptive_writer_robust_split
Student 2: v50 AW010 adaptive TTT-only
```

### 6.3 审计字段

对每个 chunk / layer / branch 记录：

```text
positive_mass / neutral_mass / negative_mass
positive_delta_norm_mean
neutral_delta_norm_mean
negative_delta_norm_mean
post_zp_delta_norm
branch w0/w1/w2 contribution norm
adaptive_gamma
adaptive_neutral_lambda
risk mean / p90 / std
prior mean / std
role threshold debug
probe_ttt_write_seconds
```

### 6.4 关键比较

计算：

$$
R_{neg/pos} = \frac{\|\Delta_{neg}\|}{\|\Delta_{pos}\| + \epsilon}
$$

$$
R_{neu/pos} = \frac{\|\Delta_{neu}\|}{\|\Delta_{pos}\| + \epsilon}
$$

$$
R_{postzp} = \frac{\|\Delta W_{postzp}^{student}\|}{\|\Delta W_{postzp}^{teacher}\| + \epsilon}
$$

目标是判断 student 是：

```text
role mask 选错
negative energy 太弱/太强
neutral contribution 错
post-zp delta norm 不匹配
branch contribution 不匹配
risk proxy 与 teacher conflict 不一致
```

### 6.5 输出

```text
phase2_adaptive_failure_audit/teacher_student_role_mass_timeline.png
phase2_adaptive_failure_audit/delta_norm_ratio_by_layer.csv
phase2_adaptive_failure_audit/post_zp_delta_ratio_by_chunk.png
phase2_adaptive_failure_audit/adaptive_failure_autopsy.md
```

### 6.6 通过标准

Phase 2 的成功不是 ATE，而是诊断清楚：

```text
必须定位 v50 主要差距来自 role assignment、risk proxy、gamma/lambda energy、split replay、EMA/native mix 还是 interaction。
如果无法定位，不允许 Phase 3 继续猜新 adaptive formula。
```

---

## 7. Phase 3：Adaptive TTT v2 候选

### 7.1 总体原则

本 Phase 只测试少量自适应算法。所有候选必须满足：

```text
READ cue 沿用 C9。
no chunk-wise policy pass。
manual positive/negative/neutral percentage = 0。
external gamma = 0 或仅作为安全 clamp 的全局常数，不作为主强度。
role assignment 自适应。
split replay 优先。
```

### 7.2 候选 A：ConflictLite adaptive split

动机：v47-v50 使用 d_tok / residual_x_dg risk 仍不能复现 C9，而 C9 原始有效风险来自 update_conflict_energy。直接 full update_conflict 太慢，因此实现 selected-layer GPU conflict proxy。

定义：

```text
risk_source = conflict_lite_selected_layers
selected_layers = early/mid/late representative layers
branch = w0 first
role_mode = adaptive_writer_conflictlite_split
```

Role assignment：

```text
positive = low conflict + high write prior + low D_g
negative = high conflict + high D_g + low/static prior disagreement
neutral = remaining
```

不得指定 positive/negative percentage。阈值来自当前 chunk/layer risk distribution 的 robust modes / MAD / mixture separation。

### 7.3 候选 B：Energy-matched adaptive split

动机：v50 有 role mass，但可能 post-zp delta energy 不像 C9。该候选不追求 fixed percentage，而是让 negative / neutral 能量相对 positive 自平衡。

定义：

```text
先自适应生成 role masks。
计算 Δpos, Δneu, Δneg。
根据当前 layer/chunk 内的 Δnorm 自动解 gamma/lambda。
```

核心公式：

$$
\gamma_{eff} = \operatorname{clip}\left(
\frac{\operatorname{RMS}(\Delta_{pos})}{\operatorname{RMS}(\Delta_{neg}) + \epsilon}
\cdot s_{risk},\ \gamma_{min},\ \gamma_{max}
\right)
$$

其中 $s_{risk}$ 来自当前 negative group 的 risk separation：

$$
s_{risk}=\operatorname{clip}(\mu_{neg}(risk)-\mu_{pos}(risk), 0, 1)
$$

这不是手工 percentage，而是用当前 chunk 的更新能量决定反向强度。

### 7.4 候选 C：3D feature-cluster adaptive split

动机：之前 kmeans3 只在 1D risk 上聚类，容易把大 plateau 分错。新版本在三维特征上聚类：

```text
x_i = [risk_i, 1 - prior_i, update_norm_i]
```

聚类后按 cluster mean 解释：

```text
positive cluster = low risk + high prior + stable update
negative cluster = high risk + low prior + high update
neutral cluster = rest
```

仍然不指定百分比。

### 7.5 候选 D：Adaptive split + adaptive commit EMA

动机：C9 的 commit EMA 对 exact C9 有中等贡献，但它是 chunk-id specific。此候选用 state delta / write instability 自适应 EMA，非 chunk-id。

定义：

```text
if post_zp_delta_norm spike 或 HMC state delta spike:
    commit_alpha = smaller
else:
    commit_alpha = 1.0
```

EMA 不得使用 chunk id。它只在 Phase 3A/3B 有至少 soft pass 后启动。

### 7.6 运行矩阵

先跑 96F smoke 和 runtime projection；只有通过效率 gate 后才允许跑 full KITTI01。full 只跑少量：

```text
A1: FRAME + AdaptiveTTT_ConflictLite + SWA
B1: FRAME + AdaptiveTTT_EnergyMatched + SWA
C1: FRAME + AdaptiveTTT_3DCluster + SWA
Best_noSWA diagnostic: FRAME + best adaptive TTT only
D1: best adaptive + adaptive EMA, only if A/B/C <= 34.60
```

### 7.7 记录指标

```text
ATE / Rot / FinalErr / RPE
segment [200,300), [400,600)
rolling50/100/200 mean/p90/worst
role mass by chunk/layer
role thresholds / cluster centers
gamma_eff / neutral_lambda_eff by chunk/layer
post_zp_delta_norm by branch/layer
probe_ttt_write_seconds_mean
wall_seconds
chunk_total_seconds_mean/max
pass1_probe_seconds_mean
stage_b_seconds_mean
pass2_control_seconds_mean
save_outputs_seconds
runtime_gate_pass
no_chunk_policy_pass
manual_percentage_audit_pass
```

### 7.8 成功标准

```text
Hard pass: ATE <= 34.30m
Soft pass: ATE <= 34.60m
Fail: ATE > 34.60m
```

如果全部 fail，但 behavior audit 显示某个候选接近 teacher delta norm，则保留该机制继续优化；否则停止 adaptive TTT 公式猜测。

---

## 8. Phase 4：语义如何真实改进几何重建

### 8.1 目标

目标 2 不在本轮大规模展开，但必须明确方向。历史实验表明：

```text
语义直接控制 all-memory / TTT write 很弱。
语义作为 READ / C23 reconditioning 有局部和 full-online 小增益。
语义当前最好的 full-online 辅助是 semantic residual READ，v45 X3_L050 到 33.1936m。
```

因此下一步语义应以 **READ cue context / high-influence source guard** 为主，而不是 all-memory semantic scalar。

### 8.2 哪些失败路线可能因为代码问题仍有希望

#### 路线 A：semantic write score

早期 v6 曾经出现 `stage_d_x_dg_inv_sqrt` override 擦掉 semantic prior 的问题。因此早期 “semantic write 很弱” 不能完全当作最终科学结论。后续 v23-v28 已经修了很多接线，但 TTT semantic write 仍弱。因此该方向不作为主线，只保留为 TTT no-long-write / short-negative diagnostic。

#### 路线 B：SWA semantic control

v27 发现 SWA launcher READ_PATH 没同步，修复后有局部信号但 boundary 退化。说明 SWA 不是完全没希望，但它应该围绕 all-boundary / overlap consistency，而不是固定病灶窗口。暂不作为 v52 主线。

#### 路线 C：semantic-conditioned C23

v31 曾经有 coarse label bug，修复后有强局部信号。v45 X3_L050 有 full-online 小幅收益。因此这是语义最值得保留的方向。

### 8.3 本轮最小语义复测

只在 Phase 3 adaptive TTT hard/soft pass 后做，否则语义会和不稳定 adaptive TTT 混在一起解释不清。

候选：

```text
S1: best adaptive baseline + semantic residual C23 lambda 0.50 read-only
S2: best adaptive baseline + high-influence anomaly READ filtering
S3: best adaptive baseline + semantic residual C23 + static structure rescue
```

不跑 all-memory semantic matrix。

### 8.4 记录指标

除了 ATE，还必须记录：

```text
semantic source influence mass
source attention before/after
D_g map overlap with semantic labels
appearance anomaly overlap
static anchor source mass
semantic residual D_g RMS
stage_c cache hit / fine label coverage
```

### 8.5 成功标准

```text
Semantic useful if:
    improvement over adaptive baseline >= 0.3m
    or ATE <= 33.0m

Semantic Target-30 candidate if:
    ATE <= 32.0m
    and no [400,600) regression > +1m
```

如果语义复测不能稳定带来 >=0.3m，则语义降级为 diagnostic / cue visualization，不进入下一轮主线。

---

## 9. Phase 5：跨 sequence sanity

跨 sequence 不是本轮主目标，但必须规划清楚。

启动条件：

```text
KITTI01 no-chunk adaptive TTT ATE <= 34.30m
or semantic/adaptive combo improves C9 by >= 0.5m
```

运行：

```text
KITTI00
KITTI02
KITTI05
```

规则：

```text
同一套参数，不允许 per-sequence retune。
只做 sanity，不为其它 sequence 调参。
记录是否 catastrophic regression。
```

指标：

```text
ATE / Rot / FinalErr
rolling window p90/worst
hmc_rows / frames
no_chunk_policy_pass
manual_percentage_audit_pass
```

通过标准：

```text
No catastrophic regression:
    ATE 不超过对应 baseline + 10%
    或至少不出现明显 trajectory collapse / hmc failure。
```

---

## 10. Codex 并行执行安排

### Codex A：代码修复与审计

```text
修 past_plus_future_light_real alias bug。
修 tri debug report ambiguity。
补齐 code audit packet missing files。
生成 phase0_code_audit。
```

### Codex B：C9 attribution 汇总

```text
合并 v45/v46B attribution。
如果 8-row factorial 缺失则补跑。
生成最终 component attribution report。
```

### Codex C：Adaptive TTT behavior audit

```text
比较 C9 / F111 fixed / v50 adaptive。
输出 role mass、delta norm、post-zp、branch contribution。
定位 v50 失败原因。
```

### Codex D：Adaptive TTT v2 实现

```text
实现 conflict_lite_selected_layers。
实现 energy_matched adaptive split。
实现 3D feature-cluster adaptive split。
实现 optional adaptive commit EMA。
```

### Codex E：Full KITTI01 adaptive rows

```text
先运行 tools/v52_runtime_profile_report.py 汇总已有 full/smoke timing。
只允许通过 runtime projection 的 A/B/C 主候选和必要 no-SWA diagnostic 进入 full。
每个 run 单 GPU。
落盘 full metrics、timing breakdown 和 audit。
```

### Codex F：Semantic minimal复测

```text
只在 adaptive baseline 达到 soft pass 后启动。
复测 semantic residual READ / high-influence source guard。
```

---

## 11. 失败分流

### 11.1 如果 C9 attribution 报告仍无法回答贡献

Codex 必须补跑缺失 factorial 或 exact knockout row，不得进入 adaptive phase。

### 11.2 如果 adaptive behavior audit 无法定位 v50 失败

停止公式猜测，补充 true-action trace：

```text
selected layers / w0 branch / post-zp delta
teacher-student per-layer delta norm
role source token overlap
```

### 11.3 如果 conflict_lite 太慢

只保留 selected layers / branch w0；如果仍然 > 8s/chunk，则降级为 diagnostic，不作为 full path。

### 11.3b 如果 full rollout 太慢

如果 full wall time > 30min、chunk_total_mean > 30s，或用户观测到单条 full 约 61-62min：

```text
1. 暂停新增 full candidate。
2. 先输出 runtime profile，拆分 pass1 / Stage B / pass2 / probe TTT / save outputs。
3. 优先处理可验证的工程开销，例如不必要的 per-chunk empty_cache、重复 full rerun、缺少 start/end slice 入口、未落盘 DONE 导致重跑。
4. 后续候选必须先用 smoke/slice/global sanity gate 筛选；full 只做最终确认。
5. 不得用“多 GPU 空闲”为理由启动无机制解释的 full 网格。
```

### 11.4 如果 adaptive TTT 全部 > 34.60m

结论必须写明：

```text
当前无 chunk / 无手工 percentage adaptive TTT 不能复现 C9。
C9 的性能依赖 fixed percentage / chunk gamma / EMA 交互。
```

下一步只能做 C9 teacher behavior distillation 或重新考虑 TTT write action space，不允许继续盲扫 risk threshold。

### 11.5 如果语义 full-online 仍没有 >=0.3m gain

语义降级为：

```text
diagnostic / visualization / cue context
```

不再做 all-memory semantic matrix。

---

## 12. 本轮交付物

```text
ACL2_v52_CodeAudit_Report.md
phase0_code_audit/*
phase1_c9_attribution/*
phase2_adaptive_failure_audit/*
phase3_adaptive_ttt_v2/*
phase4_semantic_minimal/*
phase5_cross_sequence_sanity/*
ACL2_v52_Final_Report.md
```

最终报告必须回答：

```text
1. C9 的 READ / TTT / SWA 贡献各是多少？
2. C9 exact 里哪些 chunk-id 组件贡献最大？
3. 去 chunk-id 后损失多少？
4. adaptive TTT 是否已经替代手工 percentage？
5. adaptive TTT 与 C9 差距是多少？
6. adaptive TTT 差距来自 role、risk、gamma/lambda、split replay、EMA 还是 interaction？
7. 语义是否还能在 adaptive/clean baseline 上提供 full-online 增益？
8. 是否满足跨 sequence sanity 启动条件？
```

---

## 13. 本轮成功定义

本轮不以 ATE < 30m 作为唯一成功定义。

### 必须成功项

```text
C9 contribution attribution 完整交付。
no-chunk adaptive TTT 算法完整实现并 full-online 验证。
manual percentage audit pass。
no chunk-id audit pass。
```

### 性能成功项

```text
Adaptive hard pass: ATE <= 34.30m
Adaptive soft pass: ATE <= 34.60m
Semantic useful pass: semantic over adaptive >= 0.3m
```

### 失败也要有结论

如果性能失败，必须明确定位失败原因，而不是继续开新矩阵。

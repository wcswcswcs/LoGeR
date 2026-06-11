# ACL2 v57：H35 基线上的 Semantic Action 修复与新 TTT Action Space 快速实验计划

日期：2026-06-09  
目标基线：`v53 H35 full`，KITTI01 ATE = `35.74089695811434m`  
当前定位：先不管 SWA，不再继续优化 clean adaptive tri-replay 小参数；本轮必须回答两个更直接的问题：

1. **目标一：语义能不能真实改善几何重建？** 以 H35 为基线，语义相关方法至少带来 `2m` ATE 收益才算目标成功，即 full KITTI01 ATE 需要达到：

   $$
   ATE \le 35.7409 - 2.0 = 33.7409m
   $$

   如果只改善 `0.5m`，只能算最低有效进展，不能算目标成功。

2. **目标二：新的 TTT writing action 能不能替代当前失败的 adaptive tri-replay 路线？** 不再局限于三分支 replay。可以尝试二分 replay、短期残差、no-long-write、TTL、projection commit 等新动作。以 H35 为基线，至少带来 `1m` ATE 收益才算目标成功，即：

   $$
   ATE \le 35.7409 - 1.0 = 34.7409m
   $$

本轮仍然保持项目硬边界：training-free，不训练 trigger / selector / classifier / role router；不使用 GT runtime action；不允许 absolute chunk-id policy；不允许 hand-specified positive / neutral / negative percentage；KITTI01 full run 必须控制在 `28min` 内。

---

## 1. 为什么 v56 失败不能简单写成“语义无用 / 新 TTT action 无用”

v56 的负结果很明确：Semantic Track A 没有改善 H35，New TTT Action Track B 也没有改善 H35。H35 repeat full 与 landed H35 完全一致，说明基线稳定；Track A 的 A2/A3 full 结果与 H35 完全相同，Track B 的 B1/B2/B3/B4 在 704F 全部明显回退。因此 v56 不能写成方法成功。

但 v56 的失败有两个不同性质，不能混为一谈。

第一，**语义 Track A 更像 action 没有实际打到 source path，而不是证明语义本身无效。** A2/A3 full 结果与 H35 完全相同，同时 `context_source_skip_applied_count=0`、`affected_source_token_count_max=0.0`。这说明 high-influence anomaly READ filtering / static rescue 在当前路径没有选中任何 source tokens。Stage C cache 已经命中，但 semantic action 没有变成实际 READ source filtering。换句话说，v56 的语义失败首先是 **semantic-to-action 实现或选择策略失败**，而不是语义假设被科学否定。

第二，**New TTT Action Track B 的机制确实触发了，但动作设计在中长窗口不稳定。** B1/B2/B3/B4 的 debug 显示机制有触发，例如 B1 有 stable / no-long mass，B2 有 risk / no-long mass，B3 有 dual-lifetime override，B4 有 native-delta gate。但 704F 全部回退，说明这些动作不是 smoke-only bug；它们确实改变了 TTT 写入，但改变方向不对。

因此 v57 不能继续做“大语义矩阵”，也不能继续写几个新的 TTT action 名字盲跑。v57 必须先做到：**语义 action 真的选中高影响力 source token；TTT 新 action 必须从 freeze / TTL / dynamic residual 的历史 insight 出发，而不是继续粗暴 no-long-write。**

---

## 2. 当前历史进展给我们的约束和启发

v46B 的 clean no-chunk factorial 已经说明，READ / frame attention 和 TTT 都是主贡献，而 SWA 单独几乎没有 full ATE 贡献；READ+TTT 是最重要组合。因此 v57 继续暂停 SWA，把精力集中在 READ 和 TTT。历史上，ONLY_FRAME_ATTN 带来约 `3.16m` gain，ONLY_TTT 带来约 `2.23m` gain，READ+TTT 组合 gain 达到约 `5.08m`，SWA 单独只有约 `0.01m` gain。这说明如果要在 H35 上获得实质进展，最可能的入口仍然是 READ 和 TTT，而不是 SWA。

v50 证明 split replay 明显优于 fused replay，但仍没接近 C9；v52/v53/v55 进一步说明，clean adaptive tri-replay 已经进入平台。H35 是目前最好的 clean adaptive baseline，但仍是 `35.7409m`，距离 C9 `33.7629m` 约 `1.98m`。继续在 layer/rho/asym/branch 上小扫收益很低。

v7 freeze 诊断给出一个非常重要的启发：`freeze5` 可以把 `[200,300)` 从 `77.831m` 降到 `41.899m`，说明 chunk5 的 TTT commit 中确实有有害方向；但全局 ATE 变成 `38.9727m`，后段 `[400,600)` 明显恶化，说明同一个 commit 里也有必要的 continuity / scale 信息。`freeze56` 更极端，局部病灶更好但全局崩坏。因此 TTT 不能简单 freeze，也不能大面积 no-long-write。更合理的写入模型是：**有些动态 / 异常残差可以短期帮助下一跳，但不应无限进入长期 fast weights。** 这就是 v57 重新考虑 TTL / short residual 的原因。

v31/v40/v41 也说明语义最强的证据不是“semantic label -> TTT write”，而是 READ / C23 cue reconditioning 和高影响力异常 source filtering。v31 的 semantic-conditioned C23 在 short rollout 中对局部病灶有强信号；v40/v41 的 READ 路径也比 SWA/TTT 更强。v56 的 A2/A3 没有 action token，因此不能否定这条线。

---

## 3. v57 总体实验目标

v57 分两个主目标并行推进，但二者互不混淆。

**目标一：Semantic Action Repair and READ Geometry Boost。** 本目标不再问“语义是否全局帮助所有 memory”，而是问：语义能不能真正选中高影响力异常 source token，并通过 READ path 改善 H35 full ATE。它必须先通过 action-realization gate：source tokens 不能是 0，affected source tokens 不能是 0，attention/source mass 必须确实被改变。如果 action 仍为 0，本轮不允许把 full ATE 持平写成语义失败，必须先修代码。

**目标二：New TTT Action Space。** 本目标不再继续 adaptive tri-replay 小修小补，而是测试新的二分 / TTL / dynamic residual action。目标不是立刻到 C9，而是先相对 H35 获得至少 `1m` full ATE 收益。核心假设是：H35 的 TTT action 仍然太像“当前 chunk 内 risk 分组”，而不是“短期残差 vs 长期 anchor 生命周期管理”。

这两个目标都必须满足效率约束：full KITTI01 run wall time 不超过 `28min`。96F smoke 用于接线，704F screen 用于筛选；full run 总数严格控制。v57 不是大矩阵实验。

---

## 4. Phase 0：v56 负结果修复性审计，不跑新 full

Phase 0 的目标是避免重复 v56 的两类浪费：语义 full 等于 H35 但 action 没有选中 token；TTT action 机制触发但不知道为什么中长窗口崩。

### 4.1 Semantic action audit

Codex 必须从 v56 A2/A3/A1/A4 的 landed artifacts 生成一份 `semantic_action_realization_audit.md`。这个审计不看 ATE 作为第一证据，而看 action 是否真实进入模型路径。必须记录：

```text
stage_c_cache_hit_rate
stage_c_cache_superset_hit_count
semantic_label_count_mean
semantic_group_count_mean
source_tokens_selected
context_source_skip_applied_count
affected_source_token_count_max
affected_source_token_count_mean
attention_mass_removed_before_mean
attention_mass_removed_after_mean
per_label_source_mass
per_label_affected_token_mass
static_anchor_protected_mass
context_empty_source_events
```

判定标准如下：

```text
semantic_action_realized = true if:
    stage_c_cache_hit_rate == 1.0
    semantic_label_count_mean is available and > 0
    affected_source_token_count_mean > 0
    attention_mass_removed_before_mean >= 0.02
    context_empty_source_events == 0
```

如果 `affected_source_token_count_mean == 0` 或 `semantic_label_count_mean == NA`，Codex 不得启动新的 704F/full。必须进入自动修复：检查 Stage C cache 到 `MaskletOutput.L_sem / G_sem`，再到 `PriorOutput`，再到 HMC prior，再到 `context_source_skip_mask` / compact K/V 的完整链路。v56 A2/A3 的持平结果只能标记为 `semantic_action_inactive`，不能标记为 `semantic_not_helpful`。

### 4.2 TTT action behavioral audit

Codex 必须从 B1/B2/B3/B4 的 704F artifacts 生成 `ttt_action_regression_audit.md`。必须记录：

```text
stable_anchor_token_mass_mean
risk_token_mass_mean
no_long_write_token_mass_mean
short_residual_norm_mean
long_residual_norm_mean
post_zp_delta_norm_mean
post_zp_delta_norm_p90
candidate_native_delta_cos_mean
candidate_native_delta_cos_p10
layer_branch_delta_norm_table
segment_000_384_rmse
segment_384_700_rmse
rolling100_p90
```

审计要回答三个问题：

1. B1/B2 是否因为 no-long-write 覆盖太宽而破坏 continuity？
2. B3/B4 是否因为 commit/delta 操作过强而造成 post-zp energy 失衡？
3. 回退主要发生在前段、中段，还是后段？

如果 B1 的 no-long mass 大于 `0.3` 且 ATE 回退，下一轮不允许再使用 broad no-long-write；必须把 no-long 限制到 high-risk ∩ high-influence 的交集。若 post-zp delta norm 明显低于 H35 native envelope，则说明动作过度抑制；若明显高于 H35，则说明动作过度放大。

---

## 5. Phase 1：Semantic READ action repair，先证明语义真的作用于 source

Phase 1 只跑 96F / H3 smoke，不看 full ATE。目标是修复 v56 A2/A3 的 source-token=0 问题。

### 5.1 三条强制 action-realization smoke

Codex 运行下面三条 96F smoke：

```text
S0_FORCED_SEMANTIC_SOURCE_SKIP
    使用 semantic mask 中任意非空 group，强制构造 source skip。
    目的：验证 semantic token projection -> source skip hook 是通的。

S1_SKY_OR_LOWSTUFF_SOURCE_SKIP
    使用 sky / lowstuff / vegetation 等语义 group，但不要求 appearance anomaly。
    目的：验证真实 VideoMasklet label 可以投影到 source token。

S2_HIGH_D_SEMANTIC_SOURCE_SKIP
    只在 semantic group 内选择 D_g 高的 source token。
    目的：验证语义 + 几何交集不会再次变成空集。
```

这些 smoke 的 gate 是：

```text
affected_source_token_count_mean > 0
attention_mass_removed_before_mean >= 0.02
attention_mass_removed_after_mean == 0 or decreased by >= 80%
context_empty_source_events == 0
stage_c_cache_hit_rate == 1.0
```

如果 S0 都失败，是代码路径 bug，不允许进入 Phase 2。  
如果 S0 成功但 S1/S2 失败，是 VideoMasklet label/token projection 或语义 group 选择过窄，需要修 label mapping / group taxonomy。  
如果 S1 成功但 S2 失败，是 D_g 阈值过严，需要改为 robust top-k-by-mass，而不是固定阈值。

---

## 6. Phase 2：Semantic READ 704F screen，最多四个候选

只有 Phase 1 action-realization 通过后，才进入 Phase 2。Phase 2 不再跑 A1-A4 那种语义 residual 大矩阵，只跑 4 个候选：

```text
SREAD_01_GENERAL_HIGH_INFLUENCE_ANOMALY
    语义只作为 trust/context；实际 source 选择由 high source influence + high D_g + appearance anomaly 决定。

SREAD_02_SKY_LOWSTUFF_HIGH_INFLUENCE
    只在 sky / lowstuff / vegetation 中选择 high-influence source。
    若 sky 没有 source mass，不会行动。

SREAD_03_SEM_C23_RESIDUAL_WITH_ACTION_GUARD
    使用 semantic-conditioned C23 residual，但只有当 residual map 影响到真实 source tokens 时才启用。

SREAD_04_ANOMALY_FILTER_PLUS_STATIC_RESCUE
    在过滤 high-influence anomaly source 的同时保护 road / building / wall / fence / stable structure source。
```

Phase 2 记录指标：

```text
ATE_704F
Rot_704F
FinalErr_704F
segment_000_384_rmse
segment_384_700_rmse
rolling50_mean/p90/worst
rolling100_mean/p90/worst
affected_source_token_count_mean
attention_mass_removed_before/after
static_anchor_protected_mass
per_label_action_mass
wall_min
chunk_mean_seconds
```

Promotion gate：

```text
full_candidate if:
    ATE_704F <= H35_704F - 0.7m
    or rolling100_p90 <= H35_rolling100_p90 - 3m
    or candidate is within H35_704F + 0.20m but has strong action evidence and no segment regression > +0.5m
```

v57 不允许因为 704F 与 H35 完全相同而进入 full；这通常意味着 action inactive。

---

## 7. Phase 3：Semantic READ full，最多两条

Phase 3 最多跑两条 full KITTI01：Phase 2 最好的一个 general anomaly 候选和一个 semantic-specific 候选。如果 SREAD_04 明显强于 SREAD_01，则替换掉 semantic-specific 候选。

成功标准：

```text
minimum semantic progress:
    ATE <= 35.2409m
    or improvement_vs_H35 >= 0.5m

target semantic success:
    ATE <= 33.7409m
    or improvement_vs_H35 >= 2.0m

hard fail:
    ATE >= H35 + 0.3m
    or runtime > 28min
    or affected_source_token_count_mean == 0
```

如果 Phase 3 结果有 improvement 但小于 `0.5m`，不能称为语义目标成功，只能保留为 weak diagnostic。若达到 `>=0.5m`，可以进入后续组合；若达到 `>=2m`，目标一成功。

---

## 8. Phase 4：新 TTT action，不再做 broad no-long 或三分支小扫

Phase 4 只在 Phase 0 TTT action audit 完成后启动。它探索新的 TTT writing action space，目标是相对 H35 获得至少 `1m` full ATE 收益。SWA 仍然关闭，不引入语义。

### 8.1 新动作 TTT_01：Two-Replay Static Long + Native Residual

旧 tri-replay 是 positive / neutral / negative 三类。TTT_01 改成二分思想：

```text
long-static replay:
    只用 stable-anchor tokens 生成长期 update。

native/full replay:
    保留普通 native update 作为 continuity carrier。
```

最终 commit 不是 hard no-long，而是：

$$
W_{m+1}=W_{native}+\lambda_{state}(W_{static}-W_{native})
$$

其中 $\lambda_{state}$ 由当前 chunk 的 risk spread、static-anchor confidence、post-zp energy envelope 决定，不使用 chunk id，不使用固定 percentage。

这个动作的目的不是删除所有 non-static token，而是在不破坏 continuity 的前提下把长期 fast weights 轻微拉向 stable anchor。

### 8.2 新动作 TTT_02：Short Residual TTL

TTT_02 直接利用 freeze5 的 insight：有些异常 / dynamic residual 短期有用，但不应长期累积。

本动作把 TTT update 拆成：

$$
\Delta W_{full}=\Delta W_{static}+\Delta W_{short}
$$

其中 $\Delta W_{short}$ 只允许活一跳。运行时保存上一轮 short residual，在下一次 commit 时扣除或衰减：

$$
W_{m+1}=W_m+\Delta W_{static,m}+\eta_m\Delta W_{short,m}-\rho_m\Delta W_{short,m-1}
$$

$\eta_m$ 和 $\rho_m$ 都由 state energy 自动决定，不使用固定 chunk id 和手工比例。初始实现只作用于 `w0`、layer8+17，因为 H35 的 clean adaptive 信号主要来自这两个层组。

TTT_02 的核心假设是：**不要 broad no-long-write，而是允许 risky residual 短期帮助下一 chunk，但阻止它继续叠进长期 fast weights。**

### 8.3 新动作 TTT_03：Read-Conditioned No-Long Write

v56 B1/B2 的 no-long 太宽，导致 704F 大幅回退。TTT_03 只对已经被 READ/attention 证明为 high-influence anomaly 的 token 做 no-long-write。选择条件是：

```text
D_g high
AND source influence high
AND stage_d low or residual risk high
```

这不是语义 action，不用 Stage C；它只把 READ influence 用作写入风险证据。与 B1/B2 的区别是：no-long 不再覆盖大量 stable / neutral token。

### 8.4 Phase 4 screen 与 full gate

先跑 96F smoke 验证：

```text
post_zp_delta_norm_nonzero
short_residual_norm_nonzero
static_long_delta_norm_nonzero
context_empty_source_events == 0
runtime projection <= 28min
```

再跑 704F screen。进入 full 的 gate：

```text
ATE_704F <= H35_704F - 0.7m
or candidate within H35_704F + 0.20m with clear segment/rolling improvement and no energy collapse
```

最多两条 TTT full。

成功标准：

```text
minimum TTT progress:
    ATE <= 35.2409m
    or improvement_vs_H35 >= 0.5m

target TTT success:
    ATE <= 34.7409m
    or improvement_vs_H35 >= 1.0m

hard fail:
    ATE >= H35 + 0.3m
    or runtime > 28min
    or post_zp_delta energy collapse / explosion
```

---

## 9. Phase 5：组合，仅在单目标达标后启动

v57 不默认组合 Semantic READ 和 New TTT action。只有当以下任一条件成立才组合：

```text
Semantic full improvement_vs_H35 >= 0.5m
or TTT full improvement_vs_H35 >= 0.5m
```

组合最多一条：

```text
COMBO_01 = best_semantic_READ + best_new_TTT_action
```

如果二者作用区域冲突，例如 semantic READ 过滤的 source 正好被 TTT TTL 当作 short residual carrier，则不组合，先做 action overlap audit。

组合目标：

```text
combo progress:
    ATE <= min(best_semantic, best_ttt) - 0.3m

combo target:
    ATE <= 33.7409m for semantic target
    or ATE <= 34.7409m for TTT target
```

---

## 10. 必须落盘的指标和可视化

每条 704F/full run 必须记录：

```text
ATE
Rot
FinalErr
RPE_t / RPE_r
segment RMSE: 000-384, 384-700, 700-end
rolling50 mean/p90/worst
rolling100 mean/p90/worst
rolling200 mean/p90/worst
wall_min
chunk_total_seconds_mean
probe_ttt_write_seconds_mean
hmc_rows
frames
no_chunk_policy_audit
manual_percentage_audit
```

Semantic READ 必须额外记录：

```text
stage_c_cache_hit_rate
stage_c_cache_superset_hit_count
semantic_label_count_mean
semantic_group_count_mean
affected_source_token_count_mean/max
attention_mass_removed_before/after
per_label_source_mass
per_label_action_mass
static_anchor_protected_mass
context_empty_source_events
source_keep_ratio_by_layer
```

TTT action 必须额外记录：

```text
post_zp_delta_norm_mean/p90
candidate_native_delta_cos_mean/p10
branch_layer_delta_norm_table
static_long_delta_norm
short_residual_norm
previous_short_residual_subtracted_norm
no_long_write_token_mass
energy_collapse_flag
energy_explosion_flag
```

必须生成图：

```text
semantic_action_overlay_grid.png
    RGB / semantic mask / D_g / source attention mass / affected source mask

semantic_action_mass_by_label.png
    每个语义类别的 source mass 与 affected mass

ttt_delta_energy_timeline.png
    每个 chunk 的 post-zp delta / static delta / short residual

ttt_layer_branch_heatmap.png
    layer x branch 的 update energy

segment_error_comparison.png
    H35 vs candidate 的 segment RMSE

rolling100_timeline.png
    H35 vs candidate 的 rolling100 误差曲线
```

---

## 11. Codex 失败自动分流

### 11.1 Semantic action inactive

如果任何 semantic candidate full/704F 与 H35 完全相同，且 `affected_source_token_count_mean=0`，Codex 必须停止 ATE run，进入 code repair：

```text
check MaskletOutput.L_sem / G_sem nonempty
check token projection from masks to patch grid
check PriorOutput fields nonempty
check HMC control_prior carries semantic fields
check pi3 context_source_skip_mask receives nonempty mask
check compact_kv actually consumes mask
```

修复后重跑 96F smoke，不允许直接 full。

### 11.2 Semantic action active but regresses

如果 action active 但 704F ATE 回退超过 `+0.3m`，Codex 必须输出：

```text
which labels were affected
whether static anchors were suppressed
whether attention_mass_removed came mostly from sky/vegetation/road/building
whether source_keep_ratio too low
```

然后只允许一种 repair：加入 static rescue 或收窄到 high-influence anomaly。不能继续语义大矩阵。

### 11.3 TTT no-long too broad

如果 `no_long_write_token_mass_mean > 0.25` 且 ATE 回退，Codex 必须把 no-long 限制到：

```text
high D_g AND high influence AND low stage_d
```

不得扩大 no-long。

### 11.4 TTT energy collapse / explosion

如果 candidate 的 post-zp delta norm 相比 H35 小于 `0.5x` 或大于 `1.8x`，Codex 不允许继续 full。必须修 energy matching 或 residual scaling。

### 11.5 Runtime blocker

如果 projected full runtime 超过 `28min`，Codex 必须先做效率修复：

```text
disable dense overlays outside smoke
reduce per-layer trace to selected layers 8/17
turn off video export
avoid CPU transfer in residual risk
write aggregate jsonl instead of full tensors
```

效率修复后先跑 96F runtime smoke，再决定是否继续 704F/full。

---

## 12. 本轮结束时必须回答的问题

v57 结束后，报告必须明确回答：

1. v56 的 semantic failure 是否主要因为 action inactive？如果是，修复后是否能让语义选中真实 source tokens？
2. 语义 READ 在 H35 上是否能带来至少 `0.5m` / `2m` full ATE 收益？
3. 新 TTT action 中，TTL / short residual 是否优于 broad no-long-write？
4. 是否有任何新 TTT action 相对 H35 带来至少 `1m` full ATE 收益？
5. 如果两个目标都失败，是因为 action space 不足，还是因为 H35 的主要误差不来自 READ/TTT 写入？
6. 后续是否应该继续语义，还是转向更底层的 trajectory-state / merge-gauge / pose-scale controller？

---

## 13. 最终判定标准

本轮不以“是否达到 C9”作为唯一标准。因为当前基线是 H35，本轮目标是让语义和新 TTT action 先在 H35 上产生实质性增益。

```text
Goal 1 semantic success:
    full ATE <= 33.7409m
    or improvement_vs_H35 >= 2.0m

Goal 1 minimum progress:
    full ATE <= 35.2409m
    or improvement_vs_H35 >= 0.5m

Goal 2 TTT success:
    full ATE <= 34.7409m
    or improvement_vs_H35 >= 1.0m

Goal 2 minimum progress:
    full ATE <= 35.2409m
    or improvement_vs_H35 >= 0.5m

Hard failure:
    no semantic action realized after repair
    and no TTT candidate improves H35 by >=0.5m
    or all valid candidates exceed 28min runtime
```

如果只出现小于 `0.5m` 的改善，不写成成功；如果语义仍然 action inactive，不写成语义科学失败，只写成代码/映射失败；如果 TTT TTL 仍回退，则可以比较有把握地说，当前 clean H35 的剩余误差不是靠简单 TTT write action 能修掉的。

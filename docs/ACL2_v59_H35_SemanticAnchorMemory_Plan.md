# ACL2 v59 H35 Semantic Anchor Memory：用语义“增强该记住的”，而不是继续“删掉可疑的”

日期：2026-06-09  
主基线：`v53 H35 full`，KITTI01 ATE = `35.7408969581m`  
本轮目标：推进“语义如何真实帮助几何重建”这个目标，而不是继续证明语义能不能接线。  
重要边界：本轮不探索 SWA，不探索新的 TTT action 大矩阵，不做语义 hard skip，不做 learned trigger，不使用 absolute chunk id。

---

## 0. 为什么 v58 之后必须换思路

v58 给出的结果非常清楚：我们已经把语义 READ 从 hard skip 改成了 soft attenuation，并且 commit isolation 也通过了审计，但所有 704F 语义候选仍然回退。最好的语义候选 `R2_SREAD03_BIAS_FLOOR_C1` 在 704F 上仍然比 H35 差：

```text
H35 704F ATE = 39.7982477205
R2 704F ATE  = 40.731486
delta        = +0.933238
```

随机同等 mass 的负控制也回退：

```text
N0 random same mass 704F ATE = 41.068937
delta vs H35_704 = +1.270689
```

这说明两个事实：

第一，v58 的语义动作已经不是 v56 那种 inactive 状态。  
第二，“减少模型读取某些 source token”这件事本身大概率伤害 LoGeR 几何，哪怕是 soft attenuation。

因此，本轮必须停止继续问：

```text
怎样让模型少看语义可疑区域？
```

转而问：

```text
怎样让模型更多依赖语义和几何都可信的稳定锚点？
```

换句话说，本轮的核心转向是：

> **不要先忘掉可疑区域；先增强该记住的稳定几何锚点。**

这更符合当前代码和历史实验。LoGeR 的 READ 和 TTT 都需要稳定的几何支撑。如果我们直接削弱天空、植被、低值区域，很容易同时破坏相机姿态、地平线、远景连续性或局部几何上下文。v57/v58 都在说明这一点。

---

## 1. 本轮整体目标

本轮目标是在 H35 clean baseline 上验证：

> **语义能否通过识别稳定的几何锚点，增强 READ 和 TTT 对这些区域的依赖，从而改善 KITTI01 full-online ATE。**

成功标准分三层：

```text
最低有效进展：
    full KITTI01 ATE <= 35.2409m
    或相对 H35 改善 >= 0.5m

目标成功：
    full KITTI01 ATE <= 33.7409m
    或相对 H35 改善 >= 2.0m

强成功：
    full KITTI01 ATE <= 33.0m
```

本轮不是 Target-30 主攻轮。  
本轮要回答更基础的问题：

```text
语义能不能帮助 LoGeR 更好地记住稳定几何？
```

如果连这个都不成立，那么“语义帮助 memory management”的性能叙事必须降级为 diagnostic / explainability，而不是继续扩大语义规则矩阵。

---

## 2. 本轮核心假设

### 2.1 假设 H1：语义当前最有用的形式不是 suppress risky source，而是 promote stable anchor

历史上 hard skip、soft attenuation、broad no-long-write 都表现不好。这说明“删掉可疑区域”很容易误删有用几何信息。

本轮假设：

```text
road / building / wall / fence / pole / ground 等稳定结构，
如果同时满足低 D_g、高 stage_d、高 source attention、跨帧稳定，
就应该被视作 geometry anchor。
```

这些 anchor 不一定需要大幅改变所有 token，只需要在 READ 或 TTT 写入中被轻微保护 / 增强。

---

### 2.2 假设 H2：语义不能单独定义 anchor，必须和几何 cue 共同定义

一个 token 只是 `building` 或 `road` 并不自动值得长期记住。  
它还必须满足：

```text
D_g 低；
stage_d 高；
source attention 非零；
masklet 稳定；
几何 confidence 高；
不是边缘破碎区域；
空间覆盖足够分散。
```

因此本轮不使用：

```text
semantic_label -> action
```

而使用：

```text
semantic_static_label
AND geometry_stable
AND source_used
AND spatially_diverse
    -> anchor candidate
```

---

### 2.3 假设 H3：正向 anchor 增强应该比负向 source attenuation 更安全

v58 证明弱化 source 会伤几何。本轮反过来：不碰非 anchor source，只增强 anchor 的相对影响。

动作将分为两类：

```text
READ anchor boost:
    当前 chunk 推理时，让模型稍微更相信稳定 anchor source。

TTT anchor write floor:
    当前 chunk 写入未来 memory 时，稳定 anchor 至少保留一定写入资格。
```

注意：本轮不再做 broad no-long-write。  
“忘掉该忘的”在本轮只以保守方式表达：

```text
非 anchor 不额外增强；
高风险 transient 不进入 anchor boost；
不主动大规模删除。
```

也就是说，本轮先解决“该记住什么”。  
如果“该记住什么”成立，再谈“该忘掉什么”。

---

### 2.4 假设 H4：如果 anchor 增强也不带来收益，语义目前不应作为 H35 性能主线

如果 stable semantic anchor 在 704F 和 full 上都没有收益，并且随机同等 mass 的 anchor control 表现相近，那么说明当前 video-masklet 语义对 H35 的几何性能帮助不足。

这时不能继续做语义小修小补，应转向：

```text
TTT harmful update attribution；
trajectory-state / merge-gauge controller；
或语义只作为 offline diagnosis。
```

---

## 3. 相关历史证据如何影响本轮设计

### 3.1 为什么不再 hard skip

v57 已经证明语义 hard skip 能真实作用到 READ source，但大多数候选严重回退。v58 则进一步证明 soft attenuation 也没有改善 H35。  
所以本轮禁止把 attention mass 直接打到 0，也不再把“少看可疑区域”作为主动作。

### 3.2 为什么不再做 broad no-long-write

v57 的 TTT01 no-long mass 约 `0.54`，704F 明显回退。TTT03 把 no-long 缩到约 `0.13` 仍然回退。  
这说明当前我们还不知道哪些 TTT update 真的该忘，不能靠语义/风险粗略 no-long。

### 3.3 为什么聚焦 READ 和 TTT，不管 SWA

v46B clean no-chunk factorial 已经说明：

```text
READ / frame attention 单独有明显贡献；
TTT 单独有明显贡献；
SWA 单独 full ATE 贡献几乎为 0；
READ + TTT 是最重要组合。
```

因此本轮只做 READ anchor boost 和 TTT anchor write floor。SWA 暂停。

### 3.4 为什么要保留 H35 commit，不做复杂 controlled commit

v58 已经做过 commit isolation。它没有救 soft attenuation。  
本轮采用更简单的原则：

```text
READ anchor boost 默认不改变 TTT 写入；
TTT anchor write floor 只改变写入资格，不改变 READ source topology；
如果 READ 候选进入 full，则同时测试 H35-native commit isolation 与 controlled commit。
```

这能避免把 READ 修正副作用误写入未来 memory。

---

## 4. 本轮方法：Semantic Geometry Anchor Bank

本轮核心新模块不是新的语义规则表，而是一个 **Semantic Geometry Anchor Bank**。它不是长期存储模块，只是在每个 chunk 中临时计算哪些 token 是稳定 anchor。

对每个 patch token $i$ 计算一个 anchor score：

$$
A_i = S_{\text{sem}}(i) \cdot S_{\text{geo}}(i) \cdot S_{\text{attn}}(i) \cdot S_{\text{mask}}(i) \cdot S_{\text{div}}(i)
$$

各项含义如下。

### 4.1 语义稳定性 $S_{\text{sem}}$

静态结构类别给高分：

```text
road
sidewalk
building
wall
fence
pole
ground
traffic sign / traffic light if stable
```

上下文类给中性：

```text
sky
vegetation
grass
terrain
```

动态类和低信任类不给 anchor boost：

```text
car
person
cyclist
moving object
unknown unstable masklet
```

这不是写入规则，只是 anchor 资格的一部分。

---

### 4.2 几何稳定性 $S_{\text{geo}}$

几何稳定性由 H35 已有 cue 计算：

```text
D_g low
stage_d high
confidence high
C_stat high if available
C_unc low
```

推荐实现：

$$
S_{\text{geo}}(i) =
\sqrt{1 - D_g(i)}
\cdot \operatorname{norm01}(stage\_d(i))
\cdot \operatorname{norm01}(conf(i))
$$

如果某些字段不存在，必须落盘 `field_missing=true`，不能伪造为 0 或 1。

---

### 4.3 Source 使用度 $S_{\text{attn}}$

Anchor 必须是真的被模型读取过的 source，不能只是语义上看起来稳定。

记录：

```text
source_attention_mass_before
source_attention_rank
source_keep_count
```

如果某个静态结构区域 source mass 很低，它不应成为 READ boost 目标，但仍可作为 TTT anchor diagnostic。

---

### 4.4 Masklet 信任度 $S_{\text{mask}}$

使用 VideoMasklet 输出质量：

```text
masklet temporal visibility
masklet area stability
masklet label stability
masklet confidence
```

如果有 SemanticKITTI sparse projection audit，可以作为 offline trust calibration，但不能作为 runtime GT action。

---

### 4.5 空间覆盖 $S_{\text{div}}$

不能让 anchor 全部集中在一个局部区域。  
本轮必须记录 spatial coverage：

```text
image quadrant coverage
token-grid entropy
vertical/horizontal spread
near/far proxy if available
```

如果 anchor 全在道路下半部或天空边界，不能让它主导 TTT write。

---

## 5. 本轮候选设计

本轮最多允许 5 个主候选和 2 个负控制。

### 5.1 A1：READ Anchor Boost

目的：验证增强稳定 source 是否比弱化可疑 source 更好。

动作：

```text
不删除任何 source token；
不降低非 anchor source；
只对 anchor source 增加轻微 attention bias；
目标 anchor source mass 增加 10%-30%。
```

实现要求：

```text
context_empty_source_events = 0
non_anchor_mass_after >= 0.90 * non_anchor_mass_before
anchor_mass_after > anchor_mass_before
static_anchor_removed_ratio = 0
```

这条主要回答：

```text
稳定语义几何 anchor 是否能帮助当前 READ 几何推理？
```

---

### 5.2 A2：READ Anchor Boost + Commit Isolation

和 A1 相同，但未来 memory commit 使用 H35 native / clean adaptive commit，不提交 controlled READ 造成的 side effect。

目的：

```text
判断 READ anchor boost 是否只应该影响当前输出，而不应该写入未来 memory。
```

如果 A2 比 A1 更好，说明 controlled READ side effect 仍会污染后续 memory。

---

### 5.3 A3：TTT Anchor Write Floor

目的：验证语义 anchor 是否能帮助 TTT 记住稳定几何。

动作：

```text
不改变 READ；
不改变非 anchor token；
只给 anchor token 增加一个写入资格下限；
不使用手工 top percentage；
anchor mass 完全由 anchor score 决定。
```

形式：

$$
W_i^{final} = \max(W_i^{H35}, \lambda_{anchor} \cdot A_i)
$$

其中 $\lambda_{anchor}$ 不是手工 chunk 参数，而由 anchor coverage 自适应：

$$
\lambda_{anchor} =
\operatorname{clip}
\left(
\frac{\operatorname{mean}(W^{H35})}{\operatorname{mean}(A)+\epsilon},
0.2,
1.0
\right)
$$

这个式子的意思是：anchor floor 的整体能量不超过 H35 原有写入能量尺度，避免把语义写入放大成主导。

必须记录：

```text
anchor_write_floor_applied_count
anchor_write_mass
non_anchor_write_mass_delta
post_zp_delta_norm_delta_vs_H35
branch/layer delta norm
```

---

### 5.4 A4：READ Anchor Boost + TTT Anchor Write Floor

目的：验证“当前更相信 anchor + 未来更记住 anchor”是否互补。

动作：

```text
A1 + A3
```

但必须限制：

```text
如果 A1 或 A3 单独 704F 都没有任何正信号，A4 不运行。
```

避免无意义组合。

---

### 5.5 A5：Semantic-conditioned D_g Anchor Rescue

目的：重测 v31 思路，但只用于 anchor rescue，不再用于全局 suppression。

动作：

```text
在同一语义类别内部重新解释 D_g；
只对 low-D static anchor 降低风险；
不提高任何 token 的 risk；
不产生 source skip；
不进入 no-long-write。
```

也就是说，它只做：

```text
让稳定结构更不容易被 H35/D_g 误伤。
```

而不是：

```text
让高风险语义被更强抑制。
```

---

## 6. 负控制设计

### 6.1 N1：Random Same-Mass Anchor Boost

随机选取与 A1 相同数量 / 相同 attention mass 的 token 做 anchor boost。

目的：

```text
证明不是“随便增强一批 source”都能改善。
```

如果 N1 和 A1 一样好，说明语义 anchor 没有因果力。

---

### 6.2 N2：Shuffled Semantic Label Anchor

保留几何分数，打乱 semantic label，再生成 anchor。

目的：

```text
判断语义标签本身是否有帮助，还是几何 D_g/stage_d 已经足够。
```

如果 N2 和 A1/A3 一样好，说明语义没有额外贡献，只是几何 anchor 在起作用。

---

## 7. Phase 0：代码与 action audit

### 7.1 目标

本阶段不跑 full，只确认 anchor bank 和动作真实生效。

必须回答：

```text
anchor bank 是否非空？
anchor bank 是否主要由稳定结构组成？
READ boost 是否真的提高 anchor source mass？
TTT anchor floor 是否真的改变 write mass / post-zp delta？
负控制是否可运行？
```

### 7.2 必须输出

```text
phase0_anchor_audit/
    anchor_bank_by_chunk.csv
    anchor_semantic_group_distribution.csv
    anchor_spatial_coverage.csv
    anchor_dg_stage_d_histogram.csv
    read_anchor_boost_smoke.csv
    ttt_anchor_write_floor_smoke.csv
    random_control_smoke.csv
    shuffled_semantic_control_smoke.csv
    anchor_action_realization_report.md
    figures/
        anchor_semantic_bar.png
        anchor_spatial_coverage_heatmap.png
        anchor_dg_stage_d_histogram.png
        read_anchor_mass_before_after.png
        ttt_anchor_write_mass_timeline.png
```

### 7.3 Phase 0 gate

Phase 0 通过条件：

```text
anchor_token_ratio_mean in [0.03, 0.30]
anchor_spatial_entropy >= 0.35
static_semantic_ratio >= 0.60
dynamic_semantic_ratio <= 0.10
read_anchor_mass_after / before >= 1.10 for READ candidates
non_anchor_mass_after / before >= 0.90
ttt_anchor_write_mass_delta > 0 for TTT candidates
context_empty_source_events = 0
runtime_projected_full <= 28min
```

如果 anchor_token_ratio 太低，Codex 必须先修 semantic projection 或放宽几何 anchor eligibility。  
如果 anchor_token_ratio 太高，Codex 必须先加入 spatial diversity / source-attention requirement。  
如果 READ mass 没变化，修 READ bias wiring。  
如果 TTT write mass 没变化，检查 H35 adaptive writer 是否把 semantic anchor floor 归一化擦掉。

---

## 8. Phase 1：96F smoke

### 8.1 候选

```text
A1_READ_ANCHOR_BOOST
A2_READ_ANCHOR_BOOST_COMMIT_ISO
A3_TTT_ANCHOR_WRITE_FLOOR
A5_SEM_DG_ANCHOR_RESCUE
N1_RANDOM_SAME_MASS_ANCHOR_BOOST
N2_SHUFFLED_SEMANTIC_ANCHOR
```

A4 组合不在 smoke 阶段跑，除非 A1/A3 都通过。

### 8.2 必须记录

```text
ATE_96F
runtime
stage_c_hit_rate
anchor_token_ratio
anchor_source_mass_before
anchor_source_mass_after
non_anchor_source_mass_before
non_anchor_source_mass_after
anchor_write_mass
non_anchor_write_mass_delta
context_empty_source_events
commit_protocol
no_chunk_id_policy
no_gt_runtime_semantic
```

96F 只用于验证机制和速度，不用于性能结论。

---

## 9. Phase 2：704F screen

### 9.1 运行数量限制

最多运行：

```text
A1
A2
A3
A5
N1
N2
```

如果 A1 和 A3 都在 704F 有正信号，再运行：

```text
A4
```

否则 A4 不跑。

### 9.2 704F 记录指标

全局：

```text
ATE_704
delta_vs_H35_704
Rot_704
FinalErr_704
rolling50 mean / p90 / worst
rolling100 mean / p90 / worst
rolling200 mean / p90 / worst
runtime
```

语义/anchor：

```text
anchor_token_ratio_mean
anchor_semantic_distribution
anchor_spatial_entropy
anchor_source_mass_before_mean
anchor_source_mass_after_mean
anchor_mass_gain_ratio
non_anchor_mass_change_ratio
random_control_delta
shuffled_control_delta
```

TTT：

```text
anchor_write_mass
non_anchor_write_mass_delta
post_zp_delta_norm_delta_vs_H35
branch_w0/w1/w2_delta_norm
layer_delta_norm_topk
```

### 9.3 704F promotion gate

进入 full 必须满足：

```text
delta_vs_H35_704 <= -0.50m
OR rolling100_p90_delta <= -2.0m
```

同时：

```text
N1 random control 不得达到同等改善；
N2 shuffled semantic 不得达到同等改善；
[384,704) segment 不回退超过 +0.5m；
non_anchor_mass_change_ratio >= 0.90；
runtime_projected_full <= 28min；
context_empty_source_events = 0。
```

如果语义候选仅比负控制好不到 0.2m，不允许写成语义成功。

允许 1 条 borderline full diagnostic：

```text
best semantic delta_vs_H35_704 <= +0.20m
AND semantic better than both controls by >= 0.5m
AND action metrics healthy
```

---

## 10. Phase 3：full KITTI01

### 10.1 运行数量限制

最多 3 条 full：

```text
F1 = best semantic anchor candidate
F2 = second best if mechanism differs
F3 = best negative control if needed for causal claim
```

如果没有 704F 正信号，不跑 full。

### 10.2 full 指标

主指标：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3 scale
wall_min
chunk_mean_seconds
```

分段：

```text
seg0 [0,384)
seg1 [384,704)
seg2 [704,end)
[200,300)
[400,600)
rolling50 mean / p90 / worst
rolling100 mean / p90 / worst
rolling200 mean / p90 / worst
```

anchor/action：

```text
anchor_source_mass_gain_timeline
non_anchor_mass_change_timeline
anchor_write_mass_timeline
post_zp_delta_norm_timeline
anchor_semantic_distribution_by_chunk
anchor_spatial_entropy_by_chunk
context_empty_source_events
```

### 10.3 full success gate

最低有效进展：

```text
ATE <= 35.2409m
or improvement vs H35 >= 0.5m
```

目标成功：

```text
ATE <= 33.7409m
or improvement vs H35 >= 2.0m
```

强成功：

```text
ATE <= 33.0m
```

安全要求：

```text
seg2 [704,end) regression <= +0.5m
[400,600) regression <= +0.8m
rolling100_p90 regression <= +0.5m
FinalErr regression <= +1.0m
runtime <= 28min
```

如果 full ATE 下降但 Rot 或 FinalErr 明显崩，只算 diagnostic，不算成功。

---

## 11. 失败后 Codex 自动分流

### 11.1 如果 anchor bank 为空或太少

行动：

```text
检查 Stage C semantic cache；
检查 L_sem / G_sem token projection；
检查 chunk short-tail cache；
检查 semantic label mapping；
降低几何 eligibility 的 hard AND，改为 multiplicative score top adaptive threshold。
```

不允许直接 full。

---

### 11.2 如果 anchor 太多或集中

行动：

```text
加入 spatial diversity selection；
按 grid cell 保留 top anchor；
降低重复 road/ground token dominance；
保留 building/wall/fence 等垂直结构覆盖。
```

不允许用 fixed chunk id 或 per-sequence rule。

---

### 11.3 如果 READ boost 没有改变 anchor mass

行动：

```text
检查 attention bias injection；
检查 READ layer scope；
检查 source token flatten alignment；
检查 protected/special token collision；
输出 before/after attention mass。
```

---

### 11.4 如果 TTT anchor floor 没有改变 write mass

行动：

```text
检查 hmc_write_score_source 是否覆盖 semantic floor；
检查 adaptive writer normalization 是否擦掉 anchor floor；
检查 post-zp delta 是否对 anchor floor 不敏感；
若被归一化擦掉，改为 hard eligibility floor，而不是 scalar multiplier。
```

这条特别重要，因为 v6 曾出现 semantic prior 被 stage_d_x_dg override 擦掉的历史问题。

---

### 11.5 如果 704F 负控制和语义候选同样好

行动：

```text
停止声称语义有因果力。
转为 geometry-only anchor baseline。
语义只作为解释字段保留。
```

---

### 11.6 如果 704F 好但 full 坏

行动：

```text
做 tail autopsy；
检查 seg2 是否退化；
检查 anchor boost 是否后段误激活；
检查 commit side effect；
尝试 A2 commit isolation 或 tail-safe anchor confidence guard。
```

不允许重开语义矩阵。

---

### 11.7 如果全部失败

如果 A1/A2/A3/A5 全部失败，并且负控制无显著区别，则本轮结论必须写成：

```text
当前 VideoMasklet 语义不能在 H35 上提供可验证的性能增益。
语义可保留为 diagnostic / visualization / trust calibration。
性能主线应转向 TTT harmful update attribution 或 trajectory-state controller。
```

---

## 12. 可视化要求

进入 full 的候选必须生成：

```text
figures/
    anchor_overlay_chunk10.png
    anchor_overlay_chunk6.png
    anchor_overlay_chunk16.png
    anchor_semantic_distribution.png
    anchor_spatial_coverage_heatmap.png
    read_anchor_mass_timeline.png
    non_anchor_mass_change_timeline.png
    ttt_anchor_write_mass_timeline.png
    post_zp_delta_norm_timeline.png
    rolling100_delta_timeline.png
    segment_delta_bar.png
    negative_control_comparison_bar.png
```

如果没有空间图，不允许用 0 补。必须标注：

```text
no-data / unavailable
```

---

## 13. 效率要求

本轮必须控制实验规模：

```text
Phase 0:
    不跑 full。

Phase 1:
    最多 6 条 96F。

Phase 2:
    最多 7 条 704F。

Phase 3:
    最多 3 条 full。
```

每条 full：

```text
wall time <= 28min
```

如果预计超过 28min：

```text
关闭 dense overlay；
关闭非必要 tensor trace；
只保留 scalar mass；
只对 full candidate 后处理可视化；
Stage C 必须 cache read，不允许 inline compute。
```

---

## 14. 本轮最终报告必须回答的问题

最终报告必须用普通语言回答：

```text
1. 语义 anchor bank 是否真实存在？
2. 它主要由哪些语义类别组成？
3. 它是否具有足够空间覆盖？
4. READ anchor boost 是否真的增加了稳定 anchor source mass？
5. TTT anchor write floor 是否真的改变了 anchor write contribution？
6. 负控制是否排除了“随便增强 source 也有效”的可能？
7. 语义 anchor 是否带来 704F 或 full ATE 改善？
8. 如果失败，是因为 anchor 选错、action 没生效、还是语义本身没有额外信息？
```

---

## 15. 本轮结论边界

如果成功，可以说：

```text
Semantic cues help geometry by selecting stable semantic-geometric anchors
that LoGeR should trust and remember more.
```

不能说：

```text
语义已经解决 TTT 写入；
语义能控制所有 memory；
语义已经决定哪些区域该忘掉；
sky/vegetation/dynamic 已经被证明是主要污染源。
```

如果失败，必须说清楚：

```text
semantic action works；
hard skip fails；
soft attenuation fails；
anchor promotion also fails；
therefore current VideoMasklet semantic does not provide performance-critical geometry information on H35.
```

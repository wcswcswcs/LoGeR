# ACL2 v58 语义帮助几何重建：从 Hard Skip 转向 Soft Read + Commit Isolation 的实验计划

日期：2026-06-09  
基线：`v53 H35 full`，KITTI01 ATE = `35.7408969581m`  
当前目标：不再证明“语义能接上”，而是证明“语义能以正确形式改善几何重建”。

---

## 0. 本轮必须先说清楚的判断

v57 以后，我们不能再把语义失败简单归因成“Stage C 没接上”或者“语义没有生效”。v57 已经把这个问题推进了一步：语义 action-realization 修复后，semantic mask 确实能进入 READ source filtering，确实能影响 source token 和 attention mass。问题变成了另一个更本质的问题：

**现在的语义动作太硬、太宽、太粗。**

它把一批语义/几何上可疑的 source token 从 READ path 里直接删掉，导致 attention mass 从约 `0.118` 直接变成 `0`。这说明语义可以改变模型计算，但也说明 hard source skip 容易把模型仍然需要的几何上下文一起删掉。

因此，本轮不再继续做：

```text
sky hard skip
vegetation hard skip
lowstuff hard skip
semantic all-memory role matrix
broad no-long-write TTT
semantic scalar TTT write
```

本轮只围绕一个核心问题推进：

> **语义如何在不破坏几何上下文的前提下，帮助 LoGeR 少相信异常 source，并且不把 controlled read 的副作用写入未来 memory？**

这句话里有两个关键转向：

第一，语义不再做 hard skip，而是做 **soft attenuation**。  
第二，语义 READ 不再默认提交 controlled memory，而是优先采用 **commit isolation**：当前 chunk 可以用 semantic READ 修正输出，但未来 TTT/SWA/HMC state 尽量沿用 H35 native / clean adaptive state，避免局部 read 干预污染长期轨迹。

这个思路来自历史结论：Pipeline v2 早期已经证明 read-path control 有真实信号，但 controlled read 的 TTT side effect 不能直接 commit；正确做法是把当前输出修正和未来 memory commit 解耦。现在语义 READ 也必须回到这个原则上。

---

## 1. 本轮整体目标

本轮只有一个主目标：

> **在 H35 = 35.7408969581m 的 clean baseline 上，验证语义是否能通过 soft READ source control + commit isolation 带来真实 full-online ATE 改善。**

成功分为三层：

```text
最低有效进展：
    full KITTI01 ATE <= 35.2409m
    或者相对 H35 改善 >= 0.5m

语义目标成功：
    full KITTI01 ATE <= 33.7409m
    或者相对 H35 改善 >= 2.0m

强成功：
    full KITTI01 ATE <= 33.0m
```

注意：本轮不是要把所有语义记忆策略一起做完，也不是要立即解决 Target-30。目标是先建立一条可信叙事：

```text
语义不是全局 prior；
语义不是 hard skip；
语义是 READ source 可信度调制器；
并且它的副作用不应该写进未来 TTT memory。
```

如果这一条在 full-online 上成立，后面再考虑把语义用于更复杂的 TTT 写入或长期状态管理。

---

## 2. 本轮围绕哪些假设设计实验

### 2.1 假设 H1：v57 的 broad semantic skip 失败，不代表语义无用，而代表 hard skip 太粗暴

v57 里 S0/S1/S2 smoke 证明语义确实能选中 source token，并且把 attention mass 直接清零。随后 SREAD01 / SREAD02 / SREAD04 在 704F 上严重回退。这说明 broad hard skip 不是正确动作。

本轮要验证：

```text
如果保留 K/V source topology，只弱化 value 或 attention 权重，
是否能保留 SREAD03 的局部收益，同时避免 full tail 回退？
```

这对应三个动作层级：

```text
hard compact skip:
    直接删除 K/V source，v57 已经证明风险很大。

V-only attenuation:
    保留 K，弱化 V。
    模型仍能匹配 source 结构，但不完全相信其 value。

soft attention bias:
    不删除 source，只降低其 attention mass。
    让 mass_after / mass_before 保持在 0.3-0.7 之间。
```

本轮不再允许 `mass_after = 0` 的 full candidate 进入主结果，除非它只是 diagnostic。

---

### 2.2 假设 H2：SREAD03 的 704F 改善是真的，但 full 失败来自作用域过宽或后段误激活

v57 中唯一值得保留的是 SREAD03：704F 相对 H35_704 改善约 `-1.59m`，但 full 相对 H35 反而差约 `+0.68m`。这说明 SREAD03 可能在前中段修对了一些东西，但在后段伤了轨迹。

本轮要先拆 SREAD03：

```text
它在哪些 chunk 生效？
它删/弱化的是哪些 semantic group？
它主要作用在 high-D source、sky/lowstuff、vegetation，还是 general high-influence anomaly？
它在 704F 之后是否仍然激活？
它在后段是否误伤 static anchors？
```

如果 SREAD03 的后段误激活明显，本轮不训练 trigger，而是使用 deterministic guard：

```text
只在 source_influence_mass 高、D_g anomaly 高、static_anchor_loss 低的 chunk 上启用；
否则 semantic passive only。
```

这里不是学习触发器，也不是 absolute chunk id。它只是固定的 no-GT safety guard。

---

### 2.3 假设 H3：semantic READ 必须和 memory commit 解耦，否则局部 read 修正会污染后续 HMC/TTT state

LoGeR pipeline v2 的关键经验是：read control 和 memory commit 是两件事。当前 chunk 的 controlled read 可以改善输出，但 controlled forward 内部产生的 TTT/SWA/HMC side effect 未必适合写给未来 chunk。

所以本轮所有 semantic READ 候选都必须至少比较两种 commit 协议：

```text
Protocol C0: controlled commit
    当前 semantic read 后的 controlled memory 被提交给未来。
    这是风险最高的形式。

Protocol C1: native / H35 commit isolation
    当前 chunk 用 semantic read 产生输出；
    但未来 memory commit 仍使用 H35 native / clean adaptive commit。
    也就是说，semantic 只影响当前输出，不影响未来 TTT state。

Protocol C2: filtered commit diagnostic
    如果 C1 成功但某些局部 correction 消失，再考虑只提交极小的 safe delta。
    C2 不是首选，只有 C1 通过后才允许。
```

本轮优先验证 C1。  
如果 C1 显著优于 C0，说明此前 full 失败主要是 controlled read side effect 污染 future memory，而不是语义本身无效。

---

### 2.4 假设 H4：语义应该解释 D_g，而不是替代 D_g

历史 v31 的 semantic-conditioned C23 / D_g reconditioning 给过强局部信号。这说明语义更像是 D_g 的解释上下文，而不是独立 action。

因此本轮保留一条 `semantic-conditioned D_g` 路线，但动作要变软：

```text
原来的风险：
    semantic_z 直接产生强 read bias，full 失败。

本轮修正：
    semantic_z 只参与 soft attenuation 的权重；
    不直接替换 D_g；
    不进入 TTT write；
    不做 all-memory action。
```

简化公式：

$$
R_i = z_{\text{semD}}(i) \cdot M_{\text{source}}(i) \cdot (1 - A_{\text{static}}(i))
$$

其中：

- $z_{\text{semD}}(i)$ 表示同类语义内部的 D_g 异常程度；
- $M_{\text{source}}(i)$ 表示该 token 是否真的被 READ source attention 使用；
- $A_{\text{static}}(i)$ 表示该 token 是否属于静态 anchor / 结构保护区域。

最终不是 hard delete，而是：

$$
w_i = 1 - \rho \cdot \operatorname{clip}(R_i, 0, 1)
$$

并且要求：

$$
0.3 \leq \frac{\text{mass_after}}{\text{mass_before}} \leq 0.7
$$

---

## 3. 本轮不再尝试什么

为避免再次发散，本轮明确停止以下方向：

```text
1. 不做 semantic all-memory 大矩阵。
2. 不做 broad sky / vegetation / lowstuff hard skip。
3. 不做 broad no-long-write TTT。
4. 不做新的 TTT action full run。
5. 不做 learned trigger / selector / role router。
6. 不做 absolute chunk-id policy。
7. 不把 [200,300) 当 runtime 条件。
8. 不把 704F 改善写成 full-online 成功。
```

TTT 目标二在 v57 中只是负向推进：new TTT actions 确实触发，但 704F 全部回退。当前 TTT 不应继续小扫。语义路线要先在 READ path 上建立 full-online 改善，再谈 TTT。

---

## 4. 实验结构总览

本轮分成五个阶段。

```text
Phase 0:
    修复和审计 SREAD03 action trace。
    必须解释 v57 中 SREAD03 为什么 704F 改善但 full 失败。

Phase 1:
    实现 soft READ action，而不是 hard compact skip。
    先用 96F smoke 检查 action realization、mass retention、static protection。

Phase 2:
    704F screen。
    只筛选 soft READ + commit isolation 候选。

Phase 3:
    full KITTI01。
    最多跑 3 条 full，全部 runtime <= 28min。

Phase 4:
    full 后解释与分流。
    如果成功，固化语义 READ 方法；
    如果失败，判断语义是否只能做 diagnostic，还是需要进入更底层 pose/scale state controller。
```

---

## 5. Phase 0：SREAD03 autopsy，先解释现象

### 5.1 目的

Phase 0 不跑新模型，只读取 v57 已落盘 artifact，并补必要 trace。它必须回答：

```text
SREAD03 为什么 704F 好？
SREAD03 为什么 full 坏？
SREAD01/SREAD04 是否实际 source mask 等价？
static rescue 是否真的改变了 source_keep_mask？
```

### 5.2 必须产出文件

Codex 必须输出：

```text
phase0_sread03_autopsy/
    sread03_chunk_activation_timeline.csv
    sread03_segment_metrics.csv
    sread03_source_mass_timeline.csv
    sread03_semantic_group_mass.csv
    sread03_static_anchor_overlap.csv
    sread01_vs_sread04_action_jaccard.csv
    sread03_full_tail_failure_report.md
    figures/
        sread03_activation_timeline.png
        sread03_mass_before_after_timeline.png
        sread03_segment_delta_bar.png
        sread01_sread04_mask_jaccard_heatmap.png
```

### 5.3 必须记录指标

每个 chunk 记录：

```text
chunk_id
frame_start
frame_end
source_tokens_affected
source_attention_mass_before
source_attention_mass_after
mass_retention_ratio
semantic_group_affected_mass
high_D_mass
static_anchor_mass
protected_static_anchor_mass
context_empty_source_events
READ layer count
commit protocol
```

每个 segment 记录：

```text
ATE segment RMSE
rolling50 mean / p90 / worst
rolling100 mean / p90 / worst
rolling200 mean / p90 / worst
[200,300) diagnostic delta
[400,600) diagnostic delta
[704,end) tail delta
```

### 5.4 判断标准

Phase 0 不是性能 gate，而是解释 gate。

如果发现：

```text
SREAD03 的 full 失败主要来自 704F 后仍持续激活；
```

下一阶段必须加入 tail-safe guard。

如果发现：

```text
SREAD03 704F 改善主要来自 hard deletion，mass_after = 0；
```

下一阶段必须使用 soft attenuation，禁止 hard compact skip。

如果发现：

```text
SREAD01 和 SREAD04 的 source_keep_mask Jaccard > 0.98；
```

则 static rescue 实验不能算科学负结果，必须先修 source-mask algebra，再重新测试 static rescue。

如果发现：

```text
SREAD03 主要处理的 token source_attention_mass 很低；
```

说明 704F 改善可能来自统计副作用而不是语义 source filtering，后续必须降级。

---

## 6. Phase 1：Soft READ action 实现与 96F smoke

### 6.1 设计原则

语义 READ 控制从 hard skip 改成三种 soft action：

```text
R1: V-only attenuation
    保留 K，弱化 V。
    用于保留 source topology / matching。

R2: Soft attention bias with mass floor
    不 compact K/V，只降低 attention logits。
    目标让 mass_after / mass_before 保持在 0.3-0.7。

R3: Early-layer-only attenuation
    只在早期 READ layers 使用 soft attenuation。
    避免深层全局 context 被破坏。
```

所有候选都必须使用 commit isolation 版本 C1：

```text
semantic read controls current output
future memory commit = H35 native / clean adaptive commit
semantic does not enter TTT write
semantic does not enter SWA
```

如果代码中无法直接实现 output-only/native commit，必须先实现或明确证明现有 `probe_native / probe_ttt_write` commit path 等价。不能模糊地跑。

---

### 6.2 候选定义

本轮只允许 4 个主候选：

```text
R1_SREAD03_V_ONLY_C1
    使用 SREAD03 的 source selection。
    保留 K，V 乘以 soft weight。
    commit isolation。

R2_SREAD03_BIAS_FLOOR_C1
    使用 SREAD03 的 source selection。
    不删 K/V，只加 soft attention bias。
    mass_after/mass_before 目标 0.5。
    commit isolation。

R3_SREAD03_EARLY_ONLY_C1
    SREAD03 soft attenuation 只作用 early READ layers。
    commit isolation。

R4_SEM_Z_DG_SOFT_RESID_C1
    semantic-conditioned D_g 只作为 soft attenuation risk，
    不替代 D_g，不进入 TTT。
    commit isolation。
```

可以有 1 个负控制：

```text
N0_RANDOM_SAME_MASS_SOFT_C1
    随机选同等 source mass 做 soft attenuation。
    用于证明不是“随便弱化一点 source 都有效”。
```

### 6.3 96F smoke gate

每个候选先跑 96F，只检查动作与速度，不看 full ATE。

必须通过：

```text
stage_c_cache_hit_rate = 1.0
semantic_group_count > 0
affected_source_token_count_mean > 100
source_attention_mass_before_mean >= 0.03
0.3 <= source_attention_mass_after / before <= 0.7
context_empty_source_events = 0
static_anchor_removed_ratio <= 0.10
commit_isolation_hash_check = pass
wall_time_projected_full <= 28min
```

如果某候选 `mass_after = 0`，直接标为 hard-skip violation，不进入 704F。  
如果某候选 `affected_source_token_count = 0`，直接标为 action inactive，Codex 必须修 wiring，而不是继续跑。  
如果 commit isolation hash check 失败，说明 semantic read side effect 仍被提交，必须修 commit path。

---

## 7. Phase 2：704F screen

### 7.1 目的

704F 不是最终成功，但它比 96F 更能发现 full-tail 风险。Phase 2 只筛选软动作和 commit isolation是否值得 full。

### 7.2 运行数量

最多允许：

```text
4 个主候选 + 1 个负控制 = 5 条 704F
```

可以并行，但每个进程只绑定一张 GPU。

### 7.3 704F 记录指标

每条 704F 必须记录：

```text
ATE_704
delta_vs_H35_704
Rot_704
FinalErr_704
rolling50 mean / p90 / worst
rolling100 mean / p90 / worst
rolling200 mean / p90 / worst
source_mass_before_mean
source_mass_after_mean
mass_retention_ratio_mean
affected_source_token_count_mean
static_anchor_removed_ratio
per_semantic_group_affected_mass
context_empty_source_events
commit_isolation_hash_check
wall_min
chunk_mean_seconds
```

### 7.4 704F promotion gate

候选进入 full 必须满足：

```text
delta_vs_H35_704 <= -0.50m
OR rolling100_p90_delta <= -2.0m
```

同时必须满足：

```text
mass_retention_ratio in [0.3, 0.7]
static_anchor_removed_ratio <= 0.10
context_empty_source_events = 0
commit_isolation_hash_check = pass
projected_full_runtime <= 28min
random_same_mass control 不应同等改善
```

如果所有候选都没过，允许一个 borderline full diagnostic：

```text
best_704_delta <= +0.20m
and runtime pass
and action metrics healthy
```

这条规则是为了避免 v54 那种 borderline 候选被过硬 gate 误杀。

---

## 8. Phase 3：full KITTI01

### 8.1 运行数量

最多跑 3 条 full：

```text
F1 = best Phase2 semantic soft candidate
F2 = second best if mechanism different
F3 = borderline diagnostic only if needed
```

不跑 TTT combo，不跑 SWA combo，不跑 all-memory combo。

### 8.2 full 必须记录指标

全局：

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
frames
hmc_rows
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

READ action：

```text
affected_source_token_count_mean / max
source_attention_mass_before_mean
source_attention_mass_after_mean
mass_retention_ratio_mean
static_anchor_removed_ratio
semantic_group_affected_mass
high_D_affected_mass
per-layer action_count
context_empty_source_events
```

Memory isolation：

```text
commit_protocol
native_commit_hash_match_rate
TTT post_zp_delta_norm_delta_vs_H35
HMC state hash drift vs H35
merge/gauge hash drift vs H35
```

### 8.3 full success gate

最低有效进展：

```text
ATE <= 35.2409m
or improvement vs H35 >= 0.5m
```

语义目标成功：

```text
ATE <= 33.7409m
or improvement vs H35 >= 2.0m
```

安全性要求：

```text
[704,end) tail regression <= +0.5m
[400,600) regression <= +0.8m
rolling100_p90 not worse than H35 by more than +0.5m
context_empty_source_events = 0
runtime <= 28min
```

如果 full ATE 改善但 Rot / FinalErr 明显崩，候选只能标为 diagnostic，不算成功。

---

## 9. Phase 4：失败后自动分流

### 9.1 如果 action inactive

表现：

```text
affected_source_token_count = 0
source_attention_mass_before = 0
```

Codex 不允许跑 full，必须修：

```text
semantic prior projection
source mask flatten / token alignment
READ layer control wiring
candidate alias -> HMC -> pi3 参数转发
```

修复后重跑 96F smoke。

---

### 9.2 如果 soft attenuation 仍然像 hard skip

表现：

```text
mass_after / mass_before < 0.2
```

Codex 必须调 soft action 实现，不允许调语义阈值。优先：

```text
改 attention bias floor
改 V attenuation clamp
改 layer scope
```

---

### 9.3 如果 704F 改善但 full 变差

表现：

```text
delta_vs_H35_704 < 0
full delta_vs_H35 > 0
```

Codex 必须输出 full-tail autopsy：

```text
seg0/seg1/seg2 deltas
activation after 704F 是否过强
source mass tail 是否过高
commit isolation 是否失败
```

然后只允许尝试：

```text
tail-safe guard
or commit isolation repair
```

不允许重新开语义大矩阵。

---

### 9.4 如果 commit isolation 失败

表现：

```text
native_commit_hash_match_rate < 0.99
or HMC state drift vs H35 unexpected
```

这说明 semantic read side effect 被提交到了未来。Codex 必须修 two-pass commit protocol：

```text
当前输出可以用 controlled read；
未来 commit 必须来自 H35/native probe state；
semantic prior 不进入 TTT write controller；
semantic prior 不进入 SWA cache write。
```

不修好不允许 full。

---

### 9.5 如果 soft semantic READ full 仍然没有任何改善

表现：

```text
best full ATE > 35.7409m - 0.2m
```

则判断：

```text
semantic READ 在 H35 上目前只能做 diagnostic，不能支撑性能主线。
```

下一步不再继续 READ 语义小修，转向：

```text
TTT harmful update attribution
trajectory-state / merge-gauge controller
or semantic as offline explanation only
```

---

## 10. 可视化要求

每个进入 full 的候选必须生成：

```text
figures/
    rgb_semantic_overlay_chunk10.png
    sread03_soft_action_overlay_chunk10.png
    source_mass_before_after_timeline.png
    mass_retention_ratio_timeline.png
    semantic_group_affected_mass_bar.png
    static_anchor_removed_ratio_timeline.png
    rolling100_delta_timeline.png
    segment_delta_bar.png
    commit_isolation_hash_timeline.png
```

如果图中缺失数据，不允许补 0；必须标注：

```text
no-data / unavailable
```

---

## 11. 效率要求

本轮必须控制实验数量。

```text
Phase 0:
    不跑新 full。

Phase 1:
    最多 5 条 96F smoke。

Phase 2:
    最多 5 条 704F。

Phase 3:
    最多 3 条 full。

每条 full:
    wall time <= 28min。
```

如果 projected full runtime 超过 28min：

```text
先修效率；
不得直接 full。
```

效率修复优先级：

```text
1. 不在 full run 中生成 dense overlays。
2. Stage C 只读 cache，不 inline compute。
3. 关闭非必要 JSONL dense tensor dump。
4. 只对进入 full 的候选补可视化。
5. 避免 CPU fallback 的 attention/source mass 计算。
```

---

## 12. 本轮最终报告必须回答的问题

最终报告必须用普通语言回答：

```text
1. 语义 READ 是否已经从 hard skip 转成 soft attenuation？
2. 是否真的保留了部分 source attention mass？
3. SREAD03 704F 改善来自哪些 source？
4. full 失败是否来自后段误激活，还是来自 commit side effect？
5. commit isolation 是否有效？
6. soft semantic READ 是否相比 H35 带来 full ATE 改善？
7. 如果没有改善，语义路线应降级到什么角色？
```

---

## 13. 本轮预期结论边界

如果成功，本轮可以声称：

```text
Semantic cues help LoGeR geometry by softly down-weighting high-risk READ sources under commit-isolated inference.
```

不能声称：

```text
语义已经解决 TTT 写入。
语义已经控制所有 memory。
语义已经证明 sky / vegetation 本身是因果污染源。
```

如果失败，本轮也必须给出清楚结论：

```text
semantic action is real;
hard skip fails;
soft attenuation + commit isolation still fails;
therefore semantic READ is currently diagnostic rather than performance-critical on H35.
```

这比继续大矩阵更有价值，因为它能明确决定语义路线是否继续作为性能主线。
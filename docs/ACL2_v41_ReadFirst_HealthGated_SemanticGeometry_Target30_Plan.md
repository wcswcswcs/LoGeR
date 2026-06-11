# ACL2 v41：Read-First Health-Gated Semantic-Geometry Memory Control Target-30 实验计划

日期：2026-05-25  
对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / VideoMasklet / KITTI01 主开发诊断  
当前 deployable best：`C9_P0_R2`, ATE `33.7629421029m`  
阶段目标：先进入 `ATE <= 33.0m`，再冲击 `ATE <= 30.0m`

---

## 0. 本轮必须拉回的项目目标

本项目是 **training-free** 的 LoGeR memory control 工作。目标不是训练一个新模型，也不是在 KITTI01 上拟合一个触发器，而是在不引入训练、不使用 runtime GT、不针对某个数据集调参的前提下，用 LoGeR 内部几何 cue、attention cue、memory trace、VideoMasklet 语义/外观信息，构造可解释、可部署的 memory management policy。

本轮不再做大规模语义规则矩阵，也不再试图让语义同时控制 frame attention、global attention、SWA、TTT。前几轮已经说明：语义和外观异常能进入模型路径，READ/SWA/TTT 都能被操作，但多数 action 只产生短期局部信号，不能稳定转成 full-online ATE 改善。因此，本轮必须把目标收窄到一个能产生切实进展的问题：

> **能否用语义 + 几何 + 外观异常，在无 GT、training-free 的条件下判断哪些 chunk 或相邻 chunk 处于重建质量下降风险，并只在这些 bad chunks 上用 READ path 过滤高影响力异常 source，从而让 C9 full-online ATE 产生可见改善？**

这里的关键词是：

```text
quality-gated
read-first
training-free
no learned trigger
no dataset-specific tuning
no fixed chunk id
no global semantic prior
```

---

## 1. 为什么本轮必须 Read-first

v40 的结果已经给出非常明确的优先级。READ path 是当前最强信号：

```text
Phase 2A READ h10:
    best ATE delta = -1.3698298799m
    best rolling100 delta = -3.4811567463m
    best [200,300) delta = -6.3477371145m / -6.3241080599m
```

相比之下：

```text
Phase 2B SWA h10:
    best ATE delta = -0.7081471064m
    best rolling100 delta = -1.3354202480m

Phase 2C TTT h10:
    best ATE delta = -0.2368777135m
    best rolling100 delta = -1.5074228818m
```

这说明下一步不能平均用力。READ path 已经有接近 continuation gate 的 h10 信号，SWA 和 TTT 目前仍偏弱。TTT reset 更不应该作为主线，因为 v40 没有 severe global-state health flag，且历史 freeze/rollback 类实验已经证明 hard reset 容易把局部病灶变好但让后段连续性崩坏。

因此，本轮主线是：

```text
先把 READ_A2 / READ_A4 这条局部强信号解释清楚；
再做 h10 -> h15 验证；
最后只把 h15 过 gate 的 READ-only candidate 推到 C9 full online。
```

---

## 2. 本轮禁止事项

为了防止再次走偏，本轮硬性禁止：

```text
禁止训练 trigger / selector / classifier / role router
禁止用 oracle label 拟合规则
禁止使用 absolute chunk id 或 fixed [200,300) 作为 runtime 条件
禁止针对 KITTI01 或任何单一数据集调 threshold / label value / gamma
禁止把 short rollout / fixed-window / diagnostic 结果写成 deployable success
禁止在没有 h15-qualified candidate 时启动 full online
禁止在没有 severe TTT health flag 时 reset TTT
禁止重新铺大规模 semantic × memory path 矩阵
```

`[200,300)` 只作为 KITTI01 的 stress diagnostic，不是策略目标。所有 runtime 策略必须依赖 scene-agnostic 的 health metrics 和 memory trace，而不是固定时间段。

---

## 3. 本轮核心假设

### H1：bad chunks 可以由无 GT health metrics 诊断出来

本轮首先要验证，不使用 ATE/GT、不使用训练触发器，只用当前模型内部信号，是否能够识别哪些 chunk 或相邻 chunk 正在经历重建质量下降。

这些 health metrics 包括：

```text
READ health:
    high-D source attention mass
    semantic-conditioned D_g anomaly mass
    appearance anomaly source mass
    static anchor source mass drop

SWA health:
    boundary / overlap proxy
    risky semantic source mass in overlap
    static overlap anchor drop

TTT health:
    update_conflict_energy spike
    post-zp delta spike
    scale-state proxy spike
    static anchor write mass collapse

Appearance / semantic health:
    sky Lab / feature drift
    vegetation flicker
    shadow or illumination anomaly
    masklet label instability
    masklet temporal IoU drop
```

成立标准不是训练 accuracy，而是诊断可用性：

```text
Top health-risk chunks must overlap high-error rolling windows in offline analysis.
Health detector must not mark almost all chunks as bad.
Health detector must not rely on absolute chunk id.
H_read / H_swa / H_ttt should be distinguishable; all three cannot always fire together.
```

---

### H2：READ_A2 / READ_A4 的收益来自高影响力异常 source，而不是随机 bias 改动

v40 最强 READ ATE candidate 是 `READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT`，最强局部 segment candidate 是 `READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH`。本轮要解释它们到底在处理什么：

```text
READ_A2:
    general high-influence anomaly source filtering

READ_A4:
    sky appearance anomaly source attenuation, but only if source mass is high
```

如果 `READ_A4` 的有效 token 主要落在 sky 且 source attention mass 高，那么用户观察到的天空颜色变化很可能是因果线索。如果 `READ_A2` 明显强于 `READ_A4`，说明问题不是 sky 专属，而是 general high-influence appearance/geometric anomaly。

---

### H3：语义应先服务 READ path，而不是直接控制所有 memory

如果当前 bad chunk 主要表现为 high-risk source 被 frame/global attention 读取，则优先操作 READ path：

```text
query 保留
K/V source compact 或 V attenuation
static structure source rescue
不动 SWA
不动 TTT
不改 TTT commit
```

只有当 health detector 显示 SWA 或 TTT 本身异常时，才允许进入对应 memory path。

---

### H4：h10 局部有效必须通过 h15 durability 才能 full online

本轮不允许只因为 h10 `[200,300)` 好就启动 full online。h10 只是初筛。进入 full online 前必须证明：

```text
h15 仍保持 meaningful improvement；
[400,600) 或 downstream rolling windows 不明显回退；
SWA boundary / overlap consistency 不明显变差；
C9 parent 下仍有效。
```

---

### H5：如果 READ h15 失败，先做 washout attribution，而不是立刻试 SWA/TTT 大矩阵

如果 READ h10 有效、h15 失效，说明短期 read correction 被后续 memory 洗掉。此时应定位 washout 来源：

```text
被 SWA cache refresh 洗掉？
被 TTT tail update 洗掉？
被 global/chunk source update 洗掉？
被 merge/gauge state 覆盖？
```

只有定位后，才允许最小补充 SWA / TTT action，例如 SWA cache persistence 或 TTT no-long-write。不要直接 reset TTT。

---

## 4. 本轮实验总体结构

本轮分为六个阶段。每个阶段都有明确的允许继续条件。

```text
Phase 0:
    锁定边界与 no-op，确认 v40 artifacts 可复用。

Phase 1:
    从已有 v40 health atlas 生成 chunk health detector 报告。

Phase 2:
    解释 READ_A2 / READ_A4 的空间因果来源。

Phase 3:
    只在 health-selected bad chunks 上跑 3 个 READ-only h10/h15 candidates。

Phase 4:
    只有 h15 candidate 过 gate，才跑 C9/H9 full online 最小验证。

Phase 5:
    若 h10 有效但 h15 或 full 失败，做 washout / C9 compatibility attribution。

Phase 6:
    仅在 health 指向 SWA/TTT 时启动最小 SWA/TTT补救，不做大矩阵。
```

---

## 5. Phase 0：边界锁定与 no-op 复查

### 目标

确认本轮只在 v40 可信基础上推进，不重新引入工程漂移。

### 输入

```text
C9_P0_R2 full online:
    ATE = 33.7629421029m

v40 Phase 0 no-op:
    max_abs_ATE_delta_vs_noop_reference = 0
    max_raw_pose_abs_diff_vs_noop_reference = 0
    required health streams nonempty = true
```

### 要求 Codex 检查

```text
1. v40 Phase 0 no-op report 是否存在。
2. v40 Phase 1 health atlas 是否存在。
3. v40 READ Phase 2A report 是否存在。
4. parent snapshots for H9/C9 chunks 6,10,16 是否存在。
5. 没有 stale run directory 混入新实验。
6. 当前 run_prefix 不与历史 RUN_NAME 冲突。
```

### 通过标准

```text
no-op drift = 0
parent snapshots available
health atlas available
READ report available
```

若不通过，Codex 不允许启动新 rollout，必须先修复 artifact / path / stale-run 问题。

---

## 6. Phase 1：Training-free Chunk Health Detector

### 目标

先回答一个基本问题：**现在是否能用语义 + 几何 + 外观 + memory trace 判断出哪些 chunk 或相邻 chunk 有病灶？**

### 方法

不训练模型，不拟合 oracle label。只用 robust statistics 构造 scene-agnostic health scores。

对每个 chunk $m$ 定义四个健康风险分数。

#### 6.1 READ health

$$
H_{\text{read}}(m)
=
R_{\text{highD-src}}(m)
+
R_{\text{semD-anom}}(m)
+
R_{\text{app-src}}(m)
+
R_{\text{anchor-drop}}(m)
$$

各项含义：

```text
R_highD-src:
    high D_g source attention mass

R_semD-anom:
    semantic-conditioned D_g anomaly mass

R_app-src:
    appearance anomaly mass weighted by source influence

R_anchor-drop:
    static structure source mass drop
```

#### 6.2 SWA health

$$
H_{\text{swa}}(m)
=
R_{\text{boundary}}(m)
+
R_{\text{overlap}}(m)
+
R_{\text{risky-overlap-src}}(m)
+
R_{\text{static-overlap-drop}}(m)
$$

SWA health 不使用固定 `[200,300)`。它看的是所有 chunk boundary / overlap 的 local continuity proxy。

#### 6.3 TTT health

$$
H_{\text{ttt}}(m)
=
R_{\text{conflict}}(m)
+
R_{\text{post-zp-spike}}(m)
+
R_{\text{scale-proxy}}(m)
+
R_{\text{static-write-drop}}(m)
$$

#### 6.4 Appearance / semantic health

$$
H_{\text{app}}(m)
=
R_{\text{sky-drift}}(m)
+
R_{\text{veg-flicker}}(m)
+
R_{\text{shadow-illum}}(m)
+
R_{\text{masklet-instability}}(m)
$$

### 归一化规则

所有 $R$ 使用 robust rank / robust z-score，不使用 dataset-specific threshold：

$$
z(x_m)=\frac{x_m-\operatorname{median}(x)}
{\operatorname{MAD}(x)+\epsilon}
$$

也可以同时输出 percentile rank：

$$
r(x_m)=\operatorname{rank}_{percentile}(x_m)
$$

Runtime gate 不直接用固定 segment id。Phase 1 只是 offline diagnostic。

### 必须记录的指标

```text
chunk_idx
parent = H9 / C9
H_read
H_swa
H_ttt
H_app
all component R terms
rolling50/100/200 ATE offline diagnostic
stress-window membership diagnostic
top-k health rank
health type = read / swa / ttt / app / mixed
```

### 必须输出的文件

```text
phase1_health_detector/chunk_health_table.csv
phase1_health_detector/rolling_window_health_alignment.csv
phase1_health_detector/health_component_by_chunk.csv
phase1_health_detector/health_vs_rolling_ate_scatter.png
phase1_health_detector/chunk_health_timeline.png
phase1_health_detector/bad_chunk_report.md
```

### 判断标准

Phase 1 通过必须满足：

```text
1. H_read / H_swa / H_ttt / H_app 至少有一种能区分 chunk，而不是全常数。
2. Top-3 health-risk chunks 覆盖至少一个 top rolling100 bad window。
3. [200,300) 对应的 chunk 在 offline diagnostic 中应被某类 health 指标标为 high risk；若没有，必须解释是哪类 health 缺失。
4. 高风险 chunk 占比不能超过 35%，否则不是 selector，而是全局开启。
5. 不能使用 absolute chunk id 或 fixed segment id。
```

### 若不满足，Codex 自动分流

```text
如果 H_read 无法区分：
    补 source attention / affected source mask / high-D source influence instrumentation。

如果 H_app 很高但 H_read 低：
    记录 appearance-only anomaly，不允许 memory action。

如果 H_swa 全高：
    检查 boundary proxy 是否错误归一化。

如果 H_ttt 全低：
    不允许 reset / filtered commit，继续 READ-only路线。
```

---

## 7. Phase 2：READ_A2 / READ_A4 机制拆解

### 目标

解释 v40 最强 READ 信号到底来自哪里。不能继续只说“READ_A4 有效”，必须知道它处理的 token 是什么。

### 待比较候选

```text
READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT
READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH
READ_A5_STATIC_ANCHOR_RESCUE_ONLY
```

### 必须落盘的空间证据

对每个 parent / chunk / candidate 输出：

```text
RGB frame strip
semantic mask overlay
sky mask
vegetation mask
road/building/wall/fence mask
appearance anomaly heatmap
D_g map
source attention mass map before action
source attention mass map after action
candidate affected source mask
static anchor source map
high-D source map
high-influence anomaly map
```

### 必须记录的表格指标

```text
per_label_source_mass_before
per_label_source_mass_after
per_label_removed_source_mass
per_label_affected_token_count
per_label_highD_mass
per_label_appearance_anomaly_mass
per_label_static_anchor_mass
Jaccard(affected_mask, sky_mask)
Jaccard(affected_mask, high_influence_anomaly)
Jaccard(affected_mask, static_anchor)
removed_attention_mass_total
removed_attention_mass_sky
removed_attention_mass_vegetation
removed_attention_mass_dynamic
removed_attention_mass_structure
static_anchor_mass_loss
```

### 输出文件

```text
phase2_read_mechanism/read_a2_a4_attribution.csv
phase2_read_mechanism/per_label_removed_source_mass.csv
phase2_read_mechanism/action_mask_overlap.csv
phase2_read_mechanism/read_a4_sky_causality_report.md
phase2_read_mechanism/overlays/chunkXXX_parentYYY_candidateZZZ_*.png
```

### 判断标准

Phase 2 不看 ATE gate，而看机制是否清楚。

#### 情况 A：sky 是主因

若满足：

```text
READ_A4 affected source mass 中 sky 占比 >= 40%
sky removed attention mass >= 0.05
sky high appearance anomaly mass 高于 non-sky 至少 1.5x
READ_A4 改善主要发生在 sky-affected frames / chunks
```

则进入 sky-specific READ candidate。

#### 情况 B：general high-influence anomaly 是主因

若满足：

```text
READ_A2 affected source mass 与 high-influence anomaly Jaccard 高
READ_A2 removed attention mass 明显高于 sky-only
READ_A2 best ATE 强于 READ_A4
```

则进入 general anomaly READ candidate。

#### 情况 C：static rescue 是必要条件

若满足：

```text
READ_A2 suppress 后 static anchor mass loss 明显；
READ_A5 或 R3 rescue 能减少 downstream regression；
```

则必须将 static rescue 纳入最终 READ candidate。

#### 情况 D：机制不清楚

若满足：

```text
affected mask 与 sky / anomaly / high-D / structure 都不重叠；
removed attention mass 很低；
```

则不允许继续 READ full online，先修 instrumentation 或停止语义 READ 路线。

---

## 8. Phase 3：最小 READ-only h10/h15 候选

### 目标

用最少候选，把语义帮助记忆管理落到真实 ATE 改善上。

### 候选

只跑 3 个主候选 + 2 个负控制。

```text
R1_READ_HIGH_INFLUENCE_ANOMALY:
    高影响力 appearance/geometric anomaly source -> K/V compact 或 V attenuation

R2_READ_SKY_APP_ANOMALY:
    sky + appearance anomaly + source mass high -> weak attenuation

R3_READ_ANOMALY_PLUS_STATIC_RESCUE:
    R1 + road/building/wall/fence lowD static anchor protection

R4_NEG_CONTROL_SKY_NO_SOURCE_MASS:
    sky anomaly without source mass condition

R5_NEG_CONTROL_STATIC_RESCUE_ONLY:
    static anchor rescue only
```

### Chunk 选择

不能使用固定 chunk id。使用 Phase 1 health detector 选：

```text
bad_chunks = top health-risk chunks, max 3 chunks
```

同时为了和历史可比，本轮可以在报告里注明这些 chunks 是否覆盖 KITTI01 `[200,300)` stress window，但不能把该覆盖作为 runtime 条件。

### 运行矩阵

```text
parents = H9, C9
horizon = h10, h15
chunks = Phase1 health-selected bad chunks only
candidates = R1, R2, R3, R4, R5
```

如果 bad_chunks 数量为 3：

```text
2 parents × 3 chunks × 5 candidates × 2 horizons = 60 rows
```

若资源紧张：

```text
先 h10：
    2 parents × 3 chunks × 5 = 30 rows

只对 h10 过 gate 的 candidate/chunk 跑 h15。
```

### 必须记录的指标

```text
ATE delta vs same parent/chunk/horizon base
rolling50/100/200 delta
stress-window delta diagnostic
downstream [400,600) delta diagnostic
boundary_10f / boundary_20f proxy
overlap residual proxy
source attention mass removed
static anchor mass preserved
context_empty_source_events
affected token keep ratio
per-label affected mass
```

### h10 初筛 gate

满足任一：

```text
h10 ATE delta <= -1.5m
h10 rolling100 delta <= -3.0m
h10 stress-window delta <= -5.0m with downstream regression <= +1.0m
```

且必须：

```text
context_empty_source_events = 0
static_anchor_mass_loss <= 0.15 unless candidate is static-rescue control
```

### h15 durability gate

满足任一：

```text
h15 ATE delta <= -1.5m
h15 rolling100 delta <= -3.0m
h15 stress-window delta <= -5.0m
```

并且：

```text
[400,600) regression <= +1.0m
boundary_10f / boundary_20f not worse than +0.25m
durability_ratio >= 0.45
```

其中：

$$
durability\_ratio =
\frac{|\Delta_{h15}|}{|\Delta_{h10}|+\epsilon}
$$

---

## 9. Phase 4：最小 full online 验证

### 启动条件

只有 Phase 3 h15 过 gate 才允许启动。

### Full online rows

最多 2 条主行 + 1 条安全对照：

```text
F1_C9_READ_HEALTHGATED_BEST:
    C9 + best Phase 3 READ candidate
    semantic/geometry health-gated
    READ-only
    no SWA semantic action
    no TTT semantic action

F2_H9_READ_HEALTHGATED_BEST:
    H9 + best Phase 3 READ candidate
    用于机制对照，不作为 deployable best 主结果

F3_C9_READ_HEALTHGATED_READONLY_NO_TTT_WRITE_EFFECT:
    如果 F1 失败，用于确认是否与 C9 TTT write / SWA 交互冲突
```

### Full online 成功标准

```text
Minimum useful progress:
    C9 full ATE <= 33.0m

Strong progress:
    C9 full ATE <= 32.0m

Target success:
    C9 full ATE <= 30.0m
```

还必须满足：

```text
no offline trajectory rewrite
no GT runtime action
no learned trigger
hmc rows complete
no context empty source events
no severe downstream [400,600) regression > +1.0m
rolling100 / rolling200 p90 not worse
boundary metrics not systematically worse
```

---

## 10. Phase 5：失败后的归因与分流

### 10.1 如果 health detector 找不到 bad chunk

说明目前 health metrics 不能做病灶诊断。Codex 必须：

```text
补 pointmap consistency / overlap residual / source attention maps；
重新计算 H_read/H_swa/H_ttt/H_app；
不允许继续跑 memory action。
```

### 10.2 如果 Phase 2 证明 sky 不是 high-influence source

停止 sky-specific 主线，转向：

```text
general high-influence anomaly source filtering。
```

### 10.3 如果 READ h10 无效

不要动 SWA/TTT。先判断：

```text
action mask 是否作用到 high-influence source？
removed attention mass 是否足够？
static anchor 是否被误删？
```

若这些都正常但 ATE 不动，说明问题不是 read-source contamination，本轮 READ路线停止。

### 10.4 如果 READ h10 有效但 h15 失败

启动 washout attribution，不跑 full：

```text
比较 h10 endpoint 和 h15 endpoint：
    SWA cache state movement
    TTT state movement
    global/chunk source movement
    merge/gauge movement
    static anchor mass over time
    risky source mass re-entry over time
```

根据结果分流：

```text
SWA washout:
    只试 SWA cache persistence / overlap anchor protect。

TTT washout:
    只试 no-long-write for affected risky source / filtered commit。
    不 reset。

merge/gauge washout:
    semantic READ 不再作为主线，转 trajectory-state / scale-state。
```

### 10.5 如果 READ h15 有效但 C9 full 失败

做 C9 compatibility isolation：

```text
C9 without semantic influence on TTT write
C9 without semantic influence on SWA
C9 read-only semantic residual
C9 with original read beta map but READ health-gated source compact
```

目的不是调参，而是确定冲突 path。

### 10.6 如果 C9 full 进到 33m 以内

立即冻结策略，做最小 sanity：

```text
repeat C9 candidate
KITTI00/02/05 diagnostic sanity
no per-dataset tuning
```

---

## 11. 并行执行安排

### Codex A：Health detector and reports

负责：

```text
从 v40 health atlas 生成 chunk_health_table.csv
生成 health_vs_rolling_ate_scatter.png
生成 bad_chunk_report.md
```

不得启动 rollout。先回答：

```text
能否无 GT 识别 bad chunks？
H_read / H_swa / H_ttt 哪个主导？
```

### Codex B：READ_A2 / READ_A4 attribution

负责：

```text
补 source attention mass spatial maps
补 READ_A4 affected mask overlay
补 sky / anomaly / structure overlap analysis
```

如果缺 instrumentation，先补默认关闭的 logging，再跑 h3/h5 小窗口，不允许直接 full。

### Codex C：READ-only h10 queue

负责：

```text
按 Phase 1 bad chunks 启动 R1-R5 h10
只跑 detector-selected chunks
每个 LoGeR 进程绑定一张 GPU
不与已有 RUN_NAME 冲突
```

### Codex D：READ h15 continuation

只在 C 的 h10 过 gate 后启动。

### Codex E：Full online

只在 D 的 h15 过 gate 后启动。最多 2-3 条。

### Codex F：Washout attribution

如果 h10 强 h15 弱，F 启动；如果 h10 不强，F 不启动。

---

## 12. 本轮预期结果与判断

本轮必须产生比 v40 更清楚的结论。允许三种结果。

### 结果 A：READ health-gated full online improves C9

这是理想结果。

```text
C9 ATE <= 33.0m:
    语义帮助 memory management 的叙事成立：
    语义 + 几何 + 外观异常能识别 bad chunks，
    并在 READ path 上过滤 high-influence anomaly source。

C9 ATE <= 30.0m:
    Target-30 achieved.
```

### 结果 B：READ h10/h15 strong but full online fails

这说明局部 read correction 真实，但 full transfer 或 C9 compatibility 有问题。下一步只做 path isolation / washout，不再做 semantic rule matrix。

### 结果 C：READ h10 仍不够强

说明语义/外观 health-gated READ 不是短期突破口。此时不要转大矩阵，直接进入：

```text
trajectory-state / scale-state / merge-gauge controller
```

语义保留为 diagnostic / trust calibration。

---

## 13. 本轮最重要的交付物

必须交付以下 artifacts：

```text
1. bad_chunk_report.md
2. chunk_health_table.csv
3. health_vs_rolling_ate_scatter.png
4. READ_A2_A4_attribution_report.md
5. per_label_removed_source_mass.csv
6. source_attention_before_after_overlays/
7. read_h10_candidate_report.md
8. read_h15_candidate_report.md if h10 passes
9. full_online_report.md if h15 passes
10. failure_routing_summary.md
```

如果没有这些解释性 artifacts，即使某个 run ATE 下降，也不允许 claim 机制成功。

---

## 14. 一句话总结

本轮计划的核心不是继续扩大语义控制矩阵，而是：

> **先证明我们能用语义 + 几何 + 外观异常识别 bad chunks，再只在这些 chunks 上用 READ path 过滤高影响力异常 source，并最终在 C9 full online 上产生真实 ATE 改善。**

这才是把“语义能帮助记忆管理”落地为可检验结果的最快路径。

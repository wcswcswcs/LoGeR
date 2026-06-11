# ACL2 v42：C9 基础上的 Health-Gated Semantic READ 控制计划

日期：2026-05-26  
目标对象：LoGeR / HMC Pipeline v2 / Semantic Prior Generator / Video Masklet / KITTI01  
当前可部署最好结果：`C9_P0_R2 = 33.7629421029m`  
阶段目标：先进入 `ATE < 33m`，再冲 `ATE < 30m`  
项目边界：training-free，no GT runtime，no learned trigger，no dataset-specific tuning

---

## 0. 这轮计划要解决什么问题

当前当务之急不是继续扩大 `sky / vegetation / dynamic / structure` 语义规则矩阵，也不是继续让语义全局参与 frame attention、SWA、TTT。过去几轮已经反复说明：语义通路能接上，hook 也能触达模型，但全局语义干预不能稳定转成 full-online ATE 改善。v41 给出了一个更清楚的方向：**先用语义、几何、外观和 source influence 判断哪里可能坏，再只在这些坏 chunk 上做 READ path 的高影响力异常 source 过滤**。

本轮 v42 的核心目标是把这个思路落成一个最小可部署闭环：

```text
C9 current best
    ↓
probe 当前 chunk 的 health
    ↓
如果 chunk health 正常：完全不启用语义干预
    ↓
如果 chunk health 异常：只在 READ path 过滤高影响力异常 source
    ↓
验证 full online C9 ATE 是否从 33.7629m 推进到 33m 内，进一步冲 30m
```

这里的“语义帮助记忆管理”不是指语义永远参与所有 memory，而是指：

```text
语义帮助判断异常区域是什么；
几何和 attention 判断异常区域是否真的影响当前 read；
health gate 判断当前 chunk 是否真的需要干预；
READ memory 只在必要 chunk 上过滤高影响力异常 source。
```

这轮禁止再走偏成训练 trigger 或 dataset-specific 规则。

---

## 1. 当前事实与本轮判断

v41 已经给出三个关键事实。

第一，health detector 在不使用 ATE、不使用固定 chunk id 的情况下选出了 `chunk10`：

```text
selected_bad_chunks = [10]
selection_uses_ATE = false
selection_uses_fixed_chunk_or_segment = false
```

这说明“语义 + 几何 + 外观 + source influence”作为 bad chunk 诊断器有初步可行性。

第二，READ path 的短期干预在被选中的 chunk 上有强局部效果：

```text
h10 best ATE delta       = -0.5994474373m
h10 best rolling100 delta = -3.3097470786m
h10 best [200,300) delta = -8.5850808794m
```

第三，h15 中局部改善仍然存在，但整体 ATE 改善很小：

```text
C9 h15 best ATE delta       = -0.1484353180m
C9 h15 best [200,300) delta = -4.8163673672m
```

我的独立判断是：v41 不是证明语义路线失败，而是说明 **READ path 是目前唯一足够强的语义落点**。SWA 和 TTT 在当前证据下都不应该继续作为第一主线。v42 要先让 READ path 形成 full-online ATE 改善，再考虑是否需要更轻量的 memory barrier。

---

## 2. 本轮不可违反的项目边界

本轮必须严格保持 training-free。任何 Codex 实现或实验若违反以下条款，结果一律标记为 invalid。

```text
禁止训练 trigger / selector / classifier / role router。
禁止用 oracle ATE label 拟合规则。
禁止使用 absolute chunk id 作为 runtime 条件。
禁止针对 KITTI01 或任意单一数据集调阈值、label value、gamma。
禁止把 [200,300) 当作 runtime 条件。
禁止使用 GT semantic 作为 runtime action。
禁止把 short rollout / fixed-window diagnostic 写成 deployable success。
禁止在没有 severe TTT health flag 时 reset TTT。
```

允许的内容：

```text
允许使用 VideoMasklet 语义输出。
允许使用 SemanticKITTI sparse projection 作为离线 trust/audit，不作为 runtime action。
允许用当前 chunk probe 产生的 no-GT health metrics 决定是否启用 READ filtering。
允许使用固定、可解释、training-free 的 robust statistic / Pareto guard rule。
允许在不同数据集上诊断 failure mode，但不允许 per-dataset tuning。
```

---

## 3. 本轮整体假设

### H1：C9 的剩余大误差里，至少有一部分来自 bad chunk 的 READ source contamination

LoGeR 的 READ path 可以理解成当前 chunk 在做几何推理时“参考哪些 source token”。如果天空颜色突变、植被闪烁、阴影、动态物体或其他异常区域被模型高强度读取，它们可能污染当前 chunk 的几何判断。

本假设不是说“sky 一定有害”，而是说：

```text
只有当某个语义/外观异常区域同时具有 high D_g、high source influence、high attention mass 时，它才是 READ contamination 候选。
```

因此 action 应该是：

```text
query 保留；
只控制 K/V source；
只在 health detector 选中的 bad chunk 上启用；
正常 chunk 不启用语义干预。
```

### H2：v41 的 h10 改善没有转成 full online，是因为 action scope 还没有形成最小可部署闭环

v41 已经说明 h10 上局部改善很强，但 h15 overall ATE 只有小幅改善。这里不应该理解成“语义必须一直有效”，而应该理解成：

```text
READ correction 对 bad chunk 有效；
但我们还没有验证 full online 中这个局部 correction 是否足够安全、是否会被后续正常 chunk 稀释、是否需要短期 low-trust episode scope。
```

所以本轮不再用 h15 overall ATE 作为唯一 gate，而是把 h15 改成 safety check：局部坏窗口改善是否仍存在，下游窗口和 boundary 是否不崩。

### H3：为了让 ATE 真正进入 33m 内，必须直接跑少量 C9 full-online，而不是无限 short rollout

过去多轮实验过于保守，常常因为 h10/h15 某个 gate 差一点而阻止 full online。v41 的 h10/h15 局部信号已经足够强，本轮应该更务实：先跑少量 C9 full-online READ-only variants，验证是否有真实 ATE 改善。

本轮不追求大矩阵，只跑少量最有解释力的候选。

### H4：如果 READ-only full online 有改善但不足以进入 33m，需要添加“memory barrier”，而不是直接 reset TTT

如果 READ filtering 能改善 C9，但幅度不够，下一步才考虑对同一批高影响力异常 source 做轻量 memory barrier：

```text
TTT no-long-write：这些 risky source 不进入长期 TTT 写入；
SWA non-overlap V attenuation：只在非 overlap source 上弱化，不破坏 boundary；
不做 hard reset；不做 all-memory semantic scalar。
```

---

## 4. 实验设计总览

本轮分为五个阶段。每个阶段都必须落盘指标和可视化，不允许只看最终 ATE。

```text
Phase 0：C9 baseline 与 health detector no-op audit
Phase 1：full-sequence chunk health detector 与 bad chunk selection
Phase 2：READ mechanism audit，解释 READ 到底动了什么
Phase 3：C9-centered READ-only full-online 最小验证
Phase 4：如果 READ-only 有收益但不够，再加最小 memory barrier
Phase 5：总结与 go / no-go 决策
```

其中 Phase 3 是本轮最关键阶段，必须产出 full-online 结果。Phase 0-2 只作为必要的解释和保护，不得无限延长。

---

## 5. Phase 0：C9 baseline 与 no-op audit

### 5.1 目标

确认 v42 新增 health logging、READ filtering hooks、visualization logging 在 no-op 状态下不会扰动 C9。这个阶段不能太重，最多跑一条 C9 no-op repeat 和若干短 smoke。

### 5.2 实验

运行：

```text
V42_P0_C9_REFERENCE_REPEAT
V42_P0_HEALTH_LOGGING_ONLY
V42_P0_READ_HOOK_NOOP
```

固定：

```text
base = C9_P0_R2 config
semantic runtime = VideoMasklet cache only
no read action
no SWA action
no TTT action
```

### 5.3 必须记录

```text
ATE / Rot / RPE_t / RPE_r
FinalErr
[200,300) / [400,600) stress diagnostics
rolling50 / rolling100 / rolling200 mean / p90 / worst
chunk_boundary_10f / 20f mean / p90 / worst
hmc rows = 38
raw pose max diff vs C9 reference
health stream nonempty flags
context_empty_source_events
```

### 5.4 通过标准

```text
|ATE - 33.7629421029| <= 0.03m
raw_pose_max_diff = 0 or numerically negligible under existing reproducibility tolerance
hmc rows = 38
health logs nonempty
no context empty source events
```

如果 Phase 0 不过，Codex 必须停止 full run，先修 no-op；不允许把后续结果解释为策略效果。

---

## 6. Phase 1：full-sequence chunk health detector

### 6.1 目标

建立一个 training-free、no-GT、scene-agnostic 的 chunk health detector。它的任务不是预测 ATE，而是判断当前 chunk / adjacent chunk 是否出现 READ source contamination 风险。

这个 detector 不允许使用固定 chunk id，也不允许使用 `[200,300)`。它只能使用当前或过去 probe 产生的内部健康指标。

### 6.2 Health metrics

对每个 chunk $m$，计算以下指标。

#### READ health

$$
H_{read}(m)=
Z(M_{highD\_src}(m))
+Z(M_{app\_src}(m))
+Z(M_{semz\_src}(m))
+Z(M_{source\_influence}(m))
-Z(M_{static\_anchor}(m))
$$

其中：

```text
M_highD_src：高 D_g source attention mass
M_app_src：外观异常 source mass，例如 Lab/feature drift 落在高 source influence 区域
M_semz_src：semantic-conditioned D_g anomaly mass
M_source_influence：被模型实际读取的 source influence mass
M_static_anchor：road/building/wall/fence 等稳定结构 source mass
Z：reset-block 或 trailing-history robust z-score，不使用 GT
```

#### SWA health

$$
H_{swa}(m)=
Z(E_{boundary10}(m))
+Z(E_{boundary20}(m))
+Z(E_{overlap}(m))
+Z(M_{risky\_overlap}(m))
-Z(M_{static\_overlap\_anchor}(m))
$$

本轮 SWA 只做 passive health，不作为主 action。

#### TTT health

$$
H_{ttt}(m)=
Z(C_{ttt\_conflict}(m))
+Z(\Delta W_{postzp}(m))
+Z(S_{scale}(m))
-Z(M_{static\_write\_anchor}(m))
$$

本轮 TTT 只作为 passive health，除非 Phase 4 进入 memory barrier。

#### Appearance-semantic health

$$
H_{app}(m)=
Z(A_{sky}(m))
+Z(A_{veg}(m))
+Z(A_{shadow}(m))
+Z(A_{masklet\_instability}(m))
$$

其中 appearance anomaly 不直接触发 action，必须与 source influence / D_g 共同满足。

### 6.3 Bad chunk 判定规则

本轮采用固定 deterministic consensus rule：

```text
bad_read_chunk(m) = true 如果：
    H_read(m) 位于当前 reset-block / trailing history 的 top-risk 区域
    且至少两个 READ 证据项同时异常：
        highD source mass high
        appearance anomaly source mass high
        semantic-z source anomaly high
        source influence mass high
    且 static anchor mass 不处于极高保护状态
```

不得使用：

```text
absolute chunk id
trajectory ATE
[200,300) segment label
manual chunk list
```

### 6.4 必须落盘

```text
phase1_health/chunk_health_table.csv
phase1_health/chunk_health_flags.jsonl
phase1_health/health_component_by_chunk.csv
phase1_health/selected_bad_chunks.json
phase1_health/health_vs_rolling_window_diagnostic.csv
phase1_health/chunk_health_timeline.png
phase1_health/health_component_stackplot.png
phase1_health/bad_chunk_report.md
```

### 6.5 判断标准

Phase 1 的目标不是直接证明策略成功，而是确认 detector 没有明显失控。

通过标准：

```text
selected_bad_chunk_ratio <= 0.20 for full sequence
selected_bad_chunk_ratio > 0
selected_bad_chunks not all reset-start chunks
selection_uses_ATE = false
selection_uses_fixed_chunk_or_segment = false
READ health 是主要触发源时，H_swa/H_ttt 不应无差别全高
```

诊断标准：

```text
post-hoc rolling-window alignment 可以看，但不能用于 runtime action。
如果 selected chunks 覆盖了高 rolling100 / rolling200 error windows，说明 detector 有诊断价值。
如果 detector 完全错过所有 high-error windows，READ health 公式需要修，但不能用 ATE 来拟合阈值。
```

---

## 7. Phase 2：READ mechanism audit

### 7.1 目标

解释 v41 READ 信号到底来自哪里。本阶段必须回答三个问题：

```text
Q1：READ filtering 是不是主要动了 sky appearance anomaly？
Q2：READ filtering 是不是主要动了 general high-influence anomaly？
Q3：READ filtering 是否误伤了 static structure anchors？
```

这一步是为了避免继续盲试 `sky skip` 或 `vegetation skip`。

### 7.2 可视化与统计

对每个 selected bad chunk 输出以下 overlay：

```text
RGB frames
VideoMasklet fine/coarse semantic map
sky mask / vegetation mask / road-building-wall-fence mask
appearance anomaly map
D_g map
semantic-conditioned z_D map
source attention mass map
source influence mass map
READ affected source mask
before/after attention mass map
static anchor rescue map
high-influence anomaly mask
```

必须同时输出 token-level / masklet-level 表：

```text
per_label_source_mass_before_after.csv
per_label_affected_source_mass.csv
per_masklet_source_mass_before_after.csv
per_masklet_appearance_anomaly.csv
per_masklet_Dg_q90.csv
per_masklet_influence_mass.csv
per_masklet_static_anchor_flag.csv
```

### 7.3 机制判定

判定规则如下：

```text
如果 sky affected mass 占 READ affected source mass >= 0.50，
且 sky source mass before high，
且 sky D_g / appearance anomaly high，
则 sky hypothesis = supported。

如果 high-influence anomaly affected mass 高，
但 semantic label 分散，
则 general anomaly hypothesis = supported。

如果 static anchor affected mass 高，
则 current READ candidate 可能误伤结构，需要 static rescue。

如果 affected source 原本 attention/influence mass 很低，
则 READ action 可能是无效 mask，不能进入 full。
```

### 7.4 通过标准

Phase 2 不要求 ATE 改善，但必须提供可解释的 action 证据。若缺少 spatial maps，Codex 不能停止实验，但必须把机制解释标记为 `incomplete_explainability`，同时继续 Phase 3 full-online 小矩阵。

---

## 8. Phase 3：C9-centered READ-only full-online 最小验证

### 8.1 目标

本阶段是 v42 的核心。目标是在 C9 基础上跑最少量 full online，把语义帮助 READ memory management 的叙事落到实际 ATE 改善。

### 8.2 候选设计

只跑 4 个 C9 full-online 候选和 1 个 H9 对照。不得扩展大矩阵。

#### F0：C9 reference repeat / no-op

```text
base = C9_P0_R2
semantic passive only
no action
```

#### F1：C9 + R1 high-influence anomaly READ filtering

```text
action:
    selected bad chunks only
    query keep
    frame/global K/V source filtering or compact_kv
    target = high-influence anomaly source
    no SWA action
    no TTT action
```

R1 不依赖 sky label。它只依赖：

```text
source influence high
appearance anomaly high or semantic-z anomaly high
D_g high
masklet trust not low enough to discard diagnosis
```

#### F2：C9 + R2 sky appearance anomaly READ filtering

```text
action:
    selected bad chunks only
    sky or sky-like masklets only
    require appearance anomaly high
    require source influence mass high
    weak source attenuation, not hard remove
    no SWA action
    no TTT action
```

F2 用来验证用户观察的 sky hypothesis，但不把 sky 作为默认因果源。

#### F3：C9 + R3 high-influence anomaly + static structure rescue

```text
action:
    selected bad chunks only
    attenuate high-influence anomaly source
    protect static structure source:
        road/building/wall/fence lowD lowconflict
    no SWA action
    no TTT action
```

F3 用来验证是否“只 suppress 不够，必须同时 rescue static anchors”。

#### F4：C9 + R3 episode-scope one-step follow-through

```text
action:
    F3 on selected bad chunk
    if same masklet/semantic region persists into next chunk
       and READ health remains elevated but below severe threshold:
        keep weak attenuation for one additional chunk
    otherwise stop action
```

这不是“语义一直有效”。它只是 bad-event episode 的短期 follow-through，用来避免同一个异常 source 在下一 chunk 立刻重新污染 READ。

#### F5：H9 + best READ candidate diagnostic

```text
purpose:
    判断 READ policy 是否只和 C9 不兼容，还是对 H9/C9 都有效。
```

### 8.3 必须记录的指标

全局指标：

```text
ATE / Rot / RPE_t / RPE_r
FinalErr
Yaw RMSE
Sim3 scale
```

分段与 rolling 指标：

```text
rolling50 mean / p90 / worst
rolling100 mean / p90 / worst
rolling200 mean / p90 / worst
[200,300) stress diagnostic
[400,600) downstream diagnostic
selected bad chunk local window metrics
```

memory health：

```text
selected_bad_chunks
action_active_chunks
action_active_ratio
per-chunk READ health before action
per-chunk READ health after action if available
source influence removed
static anchor source preserved
context_empty_source_events
```

SWA safety：

```text
boundary_10f mean / p90 / worst
boundary_20f mean / p90 / worst
overlap residual mean / p90
```

TTT safety：

```text
TTT update_conflict_energy timeline
post-zp delta norm timeline
static anchor write mass timeline
```

### 8.4 full-online 判断标准

#### 最低有效进展

```text
C9 + candidate ATE <= 33.3m
and no downstream [400,600) regression > +1m
and no boundary_10f/20f p90 regression > +0.25m
```

#### 阶段成功

```text
C9 + candidate ATE <= 33.0m
```

#### 强成功

```text
C9 + candidate ATE <= 32.0m
```

#### Target success

```text
C9 + candidate ATE <= 30.0m
```

#### 失败判定

```text
If all F1-F4 have ATE >= C9 - 0.3m improvement:
    READ-only health-gated semantic policy is not enough.

If F2 close to F1/F3:
    sky appearance anomaly is likely causal.

If F1/F3 strong and F2 weak:
    general high-influence anomaly is the better explanation.

If F3 > F1 by >= 0.3m:
    static rescue is necessary.

If F4 > F3 by >= 0.3m without downstream/boundary harm:
    one-step episode follow-through is useful.

If F4 hurts downstream:
    do not use quarantine; keep current-chunk-only action.
```

---

## 9. Phase 4：如果 READ-only 有收益但不够，添加最小 memory barrier

### 9.1 启动条件

只有在 Phase 3 满足以下条件时才启动 Phase 4：

```text
best READ-only full candidate improves C9 by >= 0.3m
but ATE still > 33.0m
```

如果 READ-only 没有任何 full ATE 收益，不启动 Phase 4。

### 9.2 设计原则

Phase 4 不是 all-memory semantic control，也不是 TTT reset。它只针对 Phase 3 中已经被 READ filtering 证明有害的同一批 source 做最小 memory barrier。

### 9.3 候选

#### M1：READ best + TTT no-long-write for filtered source

```text
对被 READ filtering 的同一批 high-influence anomaly tokens：
    当前 chunk output 可以使用 controlled forward；
    但这些 token 不参与长期 TTT positive write。
```

#### M2：READ best + TTT neutralize risky write

```text
高风险 source 不做 negative，只设 neutral / no-positive-long。
适用于 sky / low-value stuff 等可能带有 horizon/scale context 的区域。
```

#### M3：READ best + SWA non-overlap V attenuation

```text
只在 non-overlap source 上做 V attenuation；
overlap 区域 K preserve，不 hard remove。
```

#### M4：READ best + filtered commit diagnostic

```text
只在 selected bad chunk 上对 risky source 的 TTT contribution 做 filtered commit；
不 reset TTT；
不 hard freeze；
仅作为 deployable candidate 的前置诊断。
```

### 9.4 判断标准

```text
M candidate 必须比 READ-only best 再改善 >= 0.3m ATE，
且 [400,600) regression <= +1m，
且 boundary p90 不回退。
```

如果所有 M 候选都失败，说明当前语义路线应保持 READ-only，不要强行操作 SWA/TTT。

---

## 10. Phase 5：TTT reset / rollback 只作为诊断，不作为主线

### 10.1 启动条件

只有同时满足以下条件才允许 diagnostic soft rollback：

```text
READ-only 和 memory barrier 都无法改善 C9；
H_ttt 明显高；
TTT conflict spike；
scale-state proxy spike；
static anchor write mass collapse；
```

### 10.2 允许动作

```text
soft rollback selected TTT branch/layer with rho = 0.1 or 0.2
filtered commit vs native commit comparison
no hard reset as deployable strategy
```

### 10.3 禁止动作

```text
hard reset TTT as main candidate
freeze chunk as deployable strategy
semantic -> reset direct rule
```

---

## 11. Codex 并行执行安排

### Codex A：Phase 1 health detector 与 bad chunk report

负责：

```text
生成 full-sequence chunk_health_table.csv
生成 selected_bad_chunks.json
确认 selection_uses_ATE=false
确认 selection_uses_fixed_chunk_or_segment=false
输出 health timeline / component stackplot
```

如果 detector 选出过多 chunk：

```text
不要调 KITTI01 chunk id；
改为 consensus rule，更严格要求多个 independent health signals 同时异常。
```

如果 detector 选不出任何 chunk：

```text
检查 source influence / appearance anomaly / D_g 是否为空；
检查 health metric normalization；
不要直接放宽到固定 chunk。
```

### Codex B：Phase 2 READ mechanism visualization

负责：

```text
输出 READ affected maps
输出 sky/general anomaly/static rescue attribution
确认 affected source 原本有非零 attention/influence mass
```

如果缺 spatial maps：

```text
补 instrumentation；
但不阻塞 Phase 3 full run，机制解释标记为 incomplete。
```

### Codex C：Phase 3 C9 full-online READ-only candidates

负责：

```text
F0-F5 full online 调度
每个 LoGeR 进程只绑定一张 GPU
优先 GPU 0-5 并行，最多 6 并发；如 host RAM 压力大，降到 4 并发
```

如果 F1-F4 均失败：

```text
立即停止 READ-only expansion；
进入 failure analysis，不再加 sky/vegetation variants。
```

### Codex D：Phase 4 memory barrier

只在 Phase 3 有 full ATE 收益时启动。

如果启动，负责 M1-M4 最小矩阵。

### Codex E：Final report

必须输出：

```text
v42_final_summary.md
v42_final_summary.json
full_online_registry.csv
health_detector_summary.md
read_mechanism_report.md
trajectory_diagnostics/
```

---

## 12. 必须可视化

本轮可视化不是装饰，是判断机制是否成立的证据。

### 12.1 Health timeline

```text
x-axis: chunk id
curves:
    H_read
    H_swa
    H_ttt
    H_app
    action_active
    rolling100 ATE post-hoc diagnostic
```

### 12.2 READ mechanism overlay

每个 selected bad chunk 至少 3 帧：

```text
RGB
semantic mask
sky mask
appearance anomaly
D_g
source influence mass
READ affected source
static anchor rescue
before/after source attention mass
```

### 12.3 Full trajectory plot

```text
C9 baseline trajectory
candidate trajectory
GT trajectory
error-over-time curve
rolling100 ATE curve
rolling200 ATE curve
```

### 12.4 Boundary / downstream safety chart

```text
boundary10/20 mean/p90/worst bar chart
[200,300) / [400,600) stress diagnostic bar chart
rolling50/100/200 worst-window bar chart
```

---

## 13. 预期结果与决策

### Case A：C9 + READ health-gated policy 进入 33m 内

这是本轮最低阶段成功。下一步继续优化 READ policy 与 static rescue，争取进入 32m / 30m。

叙事成立：

```text
语义 + 几何 + 外观 health detector 能发现坏 chunk；
READ path 对高影响力异常 source 的局部过滤能改善 full online trajectory；
语义帮助 memory management 的第一个可部署闭环成立。
```

### Case B：READ full 有改善，但没有进入 33m

启动 Phase 4 memory barrier。目标是把 READ 改善写成更稳定的 memory effect，但只对已证明有害的 source 做 no-long-write / neutralize，不 reset TTT。

### Case C：READ h10/h15 很强，但 full online 仍失败

检查：

```text
health detector 在 full online 是否选中了相同 bad chunks；
action 是否实际启用；
C9 read beta / SWA / TTT 是否抵消 READ action；
是否 action scope 太窄或太宽。
```

不要继续扩语义矩阵。

### Case D：READ full 完全无收益

结论：当前语义/几何 READ filtering 只能作为 diagnostic，不足以推动 Target-30。下一步转向 trajectory-state / scale-state / merge-gauge controller，不再把 semantic all-memory 当主线。

---

## 14. 本轮最终成功标准

### 14.1 最低可接受进展

```text
C9 full ATE <= 33.3m
or C9 full ATE improvement >= 0.5m
```

### 14.2 阶段成功

```text
C9 full ATE <= 33.0m
```

### 14.3 强成功

```text
C9 full ATE <= 32.0m
```

### 14.4 Target-30 成功

```text
C9 full ATE <= 30.0m
```

任何 short rollout、fixed-window、diagnostic、proxy、GT audit 都不能替代 full online success。

---

## 15. 最终说明

本轮计划的目标不是“再试几个语义规则”，而是用最少量实验完成一个关键闭环：

```text
诊断 bad chunk
    → READ-only 高影响力异常 source filtering
    → C9 full online ATE 改善
```

如果这个闭环成立，就有了语义帮助 memory management 的第一个扎实证据。  
如果不成立，就应停止继续扩大语义 all-memory 矩阵，把语义降级为 diagnostic/trust calibration，把 Target-30 主线转向 trajectory-state / scale-state / merge-gauge。

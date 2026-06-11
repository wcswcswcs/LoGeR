# ACL2 v56：以 H35 Clean 为基线的语义几何增益与新 TTT 写入动作实验计划

日期：2026-06-09  
当前主基线：`v53 H35 full`，KITTI01 ATE = `35.74089695811434m`  
当前历史强基线：`C9_P0_R2`，KITTI01 ATE = `33.76294210291885m`  
本轮优先级：先在 H35 clean 版本上做出可解释、可运行、可推进的增益；不是继续追求一轮内直接到 ATE 30。

---

## 0. 本轮必须遵守的边界

这轮不能再回到之前几条已经反复消耗时间的路线。Codex 在实现时必须把下面这些作为 hard rule：

```text
1. 不允许 absolute chunk-id policy。
   禁止使用 chunk5-9、chunk10-12、chunk16 之类的手工 chunk 列表。

2. 不允许手工指定 tri replay percentage。
   禁止 positive_frac=0.35、negative_frac=0.12 这类人工 top percentage。

3. 本轮先不管 SWA。
   所有主候选必须关闭 SWA 主控制；SWA 只能保持 H35 baseline 所需的默认状态，不能作为新增变量。

4. 不训练 trigger / selector / role router。
   所有规则必须是 training-free、当前 chunk 内自适应、或基于已存在 LoGeR 内部状态。

5. 不允许用 GT ATE、固定窗口、固定 segment 作为 runtime 条件。
   `[200,300)` 只可以用于诊断，不可以用于选择或激活。

6. 单条 KITTI01 full run 必须不超过 28 分钟。
   如果预计超过 28 分钟，Codex 必须先进入 efficiency repair，而不是继续跑 full。

7. 失败时不能只 stop。
   必须输出失败类型、最可能瓶颈、下一步自动尝试方向。
```

---

## 1. 本轮整体目标

本轮有两个目标，顺序非常明确。

### 目标 1：在 H35 clean 版本上验证语义是否能真实帮助几何重建

H35 clean 是当前最好的无 chunk-wise、无手工百分比 clean adaptive TTT 基线。它的 ATE 是：

```text
H35 full ATE = 35.74089695811434m
```

本轮语义目标不是证明 “语义能接入”，这件事历史上已经多次证明；也不是再跑 all-memory 语义矩阵。目标是回答：

> 在 H35 clean 基线之上，语义是否能通过 READ / C23 / 高影响力异常 source 过滤，真实改善几何轨迹？

成功标准：

```text
目标 1 成功：
    H35 + semantic full ATE <= 33.7409m
    即相对 H35 至少改善 2.0m。

目标 1 强信号：
    H35 + semantic full ATE <= 34.7409m
    即至少改善 1.0m。

目标 1 最低有效进展：
    H35 + semantic full ATE <= 35.2409m
    即至少改善 0.5m。
```

如果所有 semantic 候选都不能相对 H35 改善至少 0.5m，这一轮必须承认：在当前实现形式下，语义仍只能作为诊断信号，不能作为 clean baseline 的主增益模块。

### 目标 2：探索新的 TTT 写入动作，而不是继续修 tri replay

过去的 adaptive TTT 尝试主要还是围绕三分支 replay：positive / neutral / negative。即使已经去掉手工百分比，v50/v52/v53/v54/v55 都没有接近 C9。下一步要承认：

> 当前三分支 adaptive tri replay 的 action space 可能不够，或者表达方式不对。

因此本轮目标 2 不再继续扫 tri split 的阈值、gamma、layer、rho，而是测试新的 TTT action：二分写入、长期/短期分离、风险 no-long-write、投影式 commit。目标是相对 H35 至少改善 1m。

成功标准：

```text
目标 2 成功：
    H35 + new TTT writing full ATE <= 34.7409m
    即相对 H35 至少改善 1.0m。

目标 2 最低有效进展：
    H35 + new TTT writing full ATE <= 35.2409m
    即至少改善 0.5m。

目标 2 失败：
    所有新 TTT action full ATE > 35.2409m。
```

---

## 2. 为什么要这样改：从历史实验得到的核心启发

### 2.1 C9 的强处已经比较清楚，但 clean 复现很难

v46B 的 clean no-chunk factorial 已经说明，READ / frame attention 和 TTT 是主要贡献，SWA 单独几乎没有 full ATE 贡献。clean factorial 中，`ONLY_FRAME_ATTN` 相对空控制提升约 `3.16m`，`ONLY_TTT` 提升约 `2.23m`，`ONLY_SWA` 只有约 `0.01m`，`FRAME_ATTN+TTT` 组合提升约 `5.08m`。

这说明：

```text
READ 有用。
TTT 有用。
READ+TTT 的组合尤其重要。
SWA 不是当前最优先主线。
```

但是 v53/v54/v55 也说明，想用无 chunk-id、无手工百分比的 adaptive TTT 复现 C9，非常困难。H35 仍在 `35.74m` 左右，v55 E1/E2 甚至比 H35 更差。因此继续围绕 tri replay 小修小补，收益很可能很低。

### 2.2 语义直接控制所有 memory 基本不成立，但语义帮助 READ/C23 仍有希望

历史语义实验显示：语义 all-memory 矩阵、语义直接 TTT scalar、SWA 语义控制，多数都弱。v25B/v26/v27/v28 已经把语义接线、fine label、token risk、path consumption 做得越来越完整，但结果仍没有产生 full-online 成功。

但是，语义在 READ / C23 上有更强证据：

```text
v31 semantic-conditioned C23:
    short h10/h15 上 [200,300) 可改善约 5-6m，说明语义能重解释 D_g。

v40/v41 health-gated READ:
    READ path 在 short rollout 上有明显局部收益，说明语义/外观/几何更可能帮助 READ source filtering。

v43/v45 semantic READ residual:
    在 C9 上有小幅 full-online 改善，说明语义 READ residual 不是完全无效。
```

因此本轮语义不能再做 all-memory，而要只做：

```text
semantic-conditioned C23 / D_g reconditioning
high-influence anomaly READ filtering
static anchor rescue
```

### 2.3 之前失败但可能因为代码问题仍值得小规模复查的方向

这轮不会全面重开旧矩阵，但有几个方向曾经被实现问题污染，值得以最小形式复查：

```text
1. semantic-conditioned C23 coarse/fine：
   v31 曾经有 coarse cue 检测字段 bug，后续修复后局部信号强。
   现在可以在 H35 clean 上最小复测。

2. semantic residual lambda：
   早期有 lambda 硬编码问题；v45 后修复。
   因此 H35 上应只复测 1-2 个固定 lambda，而不是大扫。

3. stage_d_x_sem_x_dg 类 semantic write：
   v6 曾经发现 stage_d_x_dg override 会擦掉 semantic effect。
   但本轮不把它作为主线，因为直接 semantic TTT write 历史上仍弱。

4. two-replay / no-long-write：
   旧 two_replay smoke 曾有 gamma 接线问题，且没有作为现代 clean no-chunk action 系统评估。
   这次可以作为新的 TTT action 正式测试。
```

### 2.4 已经多次证明不该继续主攻的方向

```text
1. 不再做 semantic all-memory 大矩阵。
2. 不再做 SWA 语义主线。
3. 不再做 fused adaptive TTT。
4. 不再继续小扫 tri replay 的 rho / gamma / layer / threshold。
5. 不再使用 health detector 选择 absolute chunk 行为作为主策略。
6. 不再使用 learned trigger。
```

---

## 3. 本轮实验结构总览

本轮只允许两个主方向并行推进：

```text
Track A：Semantic-on-H35
    目标：在 H35 clean 基线上验证语义是否能带来 >=2m 改善。
    操作路径：READ / C23 / high-influence anomaly。
    不动 SWA，不重写 TTT 主策略。

Track B：New TTT Action-on-H35
    目标：探索非三分支 replay 的 TTT writing action。
    操作路径：two-replay、no-long-write、long/short 分离、projection commit。
    不使用 chunk id，不使用手工 percentage。
```

Track A 和 Track B 先分别独立跑。只有当两者之一在 full online 上达到最低有效进展，才允许做一次组合；否则不组合。

---

## 4. Phase 0：基线、效率和审计 hard gate

Phase 0 的目标是确认本轮不是在漂移基线上继续浪费时间。

### 4.1 必须复现 H35

Codex 必须先跑或复用当前可信 H35 artifact，并写出：

```text
H35 full ATE
H35 full Rot
H35 FinalErr
H35 rolling50 / rolling100 / rolling200 mean/p90/worst
H35 segment 000-384 / 384-700 / 700-end RMSE
H35 wall time
H35 chunk_total_seconds_mean
H35 probe_ttt_write_seconds_mean 或缺失说明
```

如果 H35 repeat 和 landed H35 差异超过 `0.05m`，不能继续 Track A/B，必须先修 baseline drift。

### 4.2 Runtime gate

每条 full KITTI01 run 必须满足：

```text
wall_time_minutes <= 28
chunk_total_seconds_mean <= 42s
one process per GPU
no multi-GPU per single rollout
```

如果缺少 `timing_summary.json`，不能直接判 fail，也不能补造 `probe_ttt_write_seconds_mean`。Codex 应使用：

```text
wall_time_summary.json
01.log START/DONE timestamp
hmc rows count
```

作为 fallback runtime audit，并明确标注 `probe_ttt_write_seconds_mean = unavailable`。

### 4.3 No-chunk / no-manual audit

所有 Track B 新 TTT action 必须通过：

```text
absolute_chunk_id_policy_audit.pass = true
manual_percentage_audit.pass = true
```

禁止出现：

```text
READ_BETA_FRAME_CHUNKS 非空
TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS 非空
TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS 非空
TTT_WRITE_COMMIT_EMA_CHUNKS 非空
positive_frac / negative_frac 用于 role selection
```

Track A semantic READ 可以使用固定全局 lambda / beta，但不能使用 chunk list。

### 4.4 Stage C / semantic runtime boundary

Track A 使用 VideoMasklet / Stage C cache 时，必须：

```text
stage_c_cache_mode = read
stage_c_cache_require_hit = true
禁止 inline Stage C compute
禁止 predicted fallback 冒充 GT
禁止 GT semantic runtime action
```

如果 cache miss，直接停 semantic candidate，修 cache，不允许 fallback。

---

## 5. Phase 1：Track A — 语义在 H35 上的最小有效验证

Track A 的核心假设是：

> H35 clean 的 TTT 写入虽然不如 C9，但 READ / C23 仍然是主要几何控制入口；语义最可能通过重新解释 D_g 和过滤高影响力异常 source 来帮助几何，而不是直接写 TTT。

### 5.1 Track A 候选设计

本轮只跑四个 semantic 候选，不允许扩矩阵。

#### A1：Semantic-conditioned C23 residual

作用：只改变 READ cue，不改变 TTT write rule。

形式：

$$
D_{final} = D_{base} + \lambda (D_{sem} - D_{base})
$$

其中 $D_{sem}$ 是同语义类别内部重标定后的 C23 / D_g。它的意义是：同一个 `D_g` 在 sky、road、building、vegetation 内部的异常程度不同，不能全局统一解释。

固定设置：

```text
lambda = 0.50
path = READ only
TTT write = H35 baseline
SWA = no new control
```

#### A2：High-influence appearance anomaly READ filtering

作用：不关心具体语义类别，只过滤模型真正读取的异常 source。

一个 token 只有同时满足下面条件才被弱化：

```text
source attention / influence mass 高
D_g 高
appearance anomaly 高 或 semantic low-trust 高
不是 static structure anchor
```

这避免 `sky -> skip` 这类粗规则。天空如果只是颜色变化但模型没有读它，就不处理；如果天空颜色变化且被大量读取，才处理。

#### A3：A2 + static structure rescue

作用：防止只 suppress 异常 source 时误伤结构区域。保留 road / building / wall / fence / stable ground 这类低 D_g、低 anomaly、较高 source mass 的结构 anchor。

#### A4：Semantic C23 residual + high-influence anomaly filtering

作用：组合 A1 和 A2，但仍然只在 READ path 生效，不进入 TTT/SWA。

### 5.2 Track A 执行方式

Track A 不再跑 h10/h15 short 大矩阵，直接采用 fast screen + full 少量候选。

```text
Step A0: 96F smoke
    目的：确认 Stage C cache hit、semantic maps loaded、action mask 非空、runtime pass。

Step A1: 704F screen
    四个候选并行跑 704F。
    如果候选在 704F 上比 H35 704F 改善 >= 0.5m，进入 full。
    如果候选比 H35 704F 差 <= 0.25m 且局部 rolling100 明显改善，允许 1 条 borderline full。
    如果候选比 H35 704F 差 > 0.25m，停止该候选。

Step A2: full online
    最多跑 2 条 Track A full。
```

### 5.3 Track A 必须记录的指标

每条 Track A run 必须记录：

```text
full ATE / Rot / FinalErr / RPE_t / RPE_r
segment RMSE: 000-384, 384-700, 700-end
rolling50 / rolling100 / rolling200 mean/p90/worst
[200,300) diagnostic only
[400,600) downstream diagnostic
wall time / chunk mean / hmc rows / frames
stage_c cache hit rate
semantic label coverage
source influence mass before/after
affected source token count
static anchor protected token count
context_empty_source_events
```

并且必须输出可视化：

```text
semantic_label_overlay.png
D_g_base_vs_D_sem_vs_D_final.png
source_influence_mass_map.png
affected_source_mask_overlay.png
static_anchor_rescue_overlay.png
rolling100_error_timeline.png
segment_error_bar.png
```

### 5.4 Track A 成功/失败判定

```text
Track A success:
    full ATE <= 33.7409m

Track A strong signal:
    full ATE <= 34.7409m

Track A minimum progress:
    full ATE <= 35.2409m

Track A fail:
    no full candidate <= 35.2409m
```

如果 A1 有效而 A2 无效，说明语义主要在重解释 C23 / D_g。  
如果 A2 有效而 A1 无效，说明语义/外观主要通过 high-influence source filtering 起作用。  
如果 A3 明显强于 A2，说明 static rescue 是必要组成。  
如果 A4 比 A1/A2 都差，说明两种 READ 操作互相冲突，不再组合。

---

## 6. Phase 2：Track B — 新 TTT 写入动作，不再继续 tri replay 小修

Track B 的核心假设是：

> H35 clean adaptive TTT 不接近 C9，不是因为 tri replay 的阈值没扫够，而是当前 TTT action space 仍然错误。需要测试二分写入、long/short 分离、no-long-write、投影式 commit 等更符合 TTT 记忆机制的动作。

### 6.1 候选 B1：Binary Stable-Anchor Replay

B1 不再把 token 分成 positive / neutral / negative 三类，而是只分成：

```text
stable-anchor tokens
non-anchor tokens
```

stable-anchor 由当前 chunk 内的 robust score 决定，不使用 top percentage：

$$
S_i = \operatorname{norm01}(P_i) \cdot (1 - \operatorname{norm01}(R_i)) \cdot (1 - \operatorname{norm01}(D_i))
$$

其中：

```text
P_i = stage_d / write eligibility prior
R_i = fast risk proxy，例如 residual_x_dg 或 selected cheap conflict proxy
D_i = C23 D_g risk
```

分组规则：

```text
使用 Otsu / robust valley threshold 在 S_i 上分两组。
不允许使用 top-k percentage。
```

写入动作：

```text
stable-anchor replay -> long commit
non-anchor tokens -> native / no-long-write
不做 negative replay
```

这个设计检验一个核心假设：也许 TTT 不需要 negative branch，当前最缺的是只把稳定 anchor 写进 long memory。

### 6.2 候选 B2：Risk-Veto Commit

B2 不重新发明 positive replay，而是在 commit 阶段把高风险部分从 long memory 中排除。

分组：

$$
U_i = \operatorname{norm01}(R_i) \cdot (1 - P_i + \epsilon) \cdot (D_i + \epsilon)
$$

用 robust threshold 分出：

```text
risk tokens
non-risk tokens
```

写入动作：

```text
non-risk tokens 正常参与 commit
risk tokens 可以参与当前输出，但不进入 long commit
不做 hard reset
不做 negative replay
```

这个设计检验：C9 的收益是否来自“不要把坏 token 写进长期 TTT”，而不是来自负写入。

### 6.3 候选 B3：Two-Lifetime Commit

B3 将 TTT 更新分成两个生命周期：

```text
long update:
    stable / low-risk update，会提交给未来 chunks。

short update:
    risky / high-change update，只允许服务当前 chunk 或极短窗口，不长期保存。
```

实现近似：

```text
Pass 2 controlled output 使用 candidate update。
Commit 给下一 chunk 的 fast weights 使用 long-filtered update。
```

这个设计来自 freeze5 的启发：chunk5 的 commit 中同时有 harmful component 和 useful continuity，hard freeze 会同时删掉两者。Two-Lifetime 试图保留有用 continuity，同时不把 risky part 长期写入。

### 6.4 候选 B4：Projection Commit

B4 不做 token 三分组，而是在 fast-weight delta 层面处理：

```text
计算 native delta 方向。
计算 candidate delta 方向。
如果 candidate delta 中与 native / stable-anchor delta 正交或冲突的部分过大，削弱该部分。
```

形式上：

$$
\Delta W_{commit} = \Delta W_{parallel} + \alpha \cdot \operatorname{Proj}_{stable}(\Delta W_{candidate})
$$

或：

$$
\Delta W_{commit} = \Delta W_{candidate} - \beta \cdot \operatorname{Proj}_{conflict}(\Delta W_{candidate})
$$

其中 $\alpha, \beta$ 由当前 chunk 的 delta norm / cosine 自适应得到，不使用 chunk id。

这个设计检验：TTT 的问题是否主要在 update direction，而不是 token role。

### 6.5 Track B 执行方式

```text
Step B0: 96F smoke
    四个候选必须证明 no-chunk / no-manual-percentage / runtime pass。

Step B1: 704F screen
    四个候选并行跑 704F。
    进入 full 的条件：
        ATE <= H35_704F - 0.5m
        或 ATE <= H35_704F + 0.25m 且 rolling100 / segment 不恶化。

Step B2: full online
    最多跑 2 条 Track B full。
```

### 6.6 Track B 必须记录的指标

除了 full trajectory 指标外，TTT 写入必须记录：

```text
role / group mass by chunk
threshold by chunk
fallback / collapse count
post-zp delta norm by layer/branch
candidate-vs-native delta cosine
commit delta norm
long update norm
short update norm
no-long-write token mass
risk token mass
stable anchor token mass
runtime per chunk
```

必须输出可视化：

```text
role_mass_timeline.png
threshold_timeline.png
post_zp_delta_norm_by_chunk.png
branch_layer_delta_heatmap.png
candidate_native_cosine_timeline.png
long_short_update_energy_timeline.png
segment_error_timeline.png
```

### 6.7 Track B 成功/失败判定

```text
Track B success:
    full ATE <= 34.7409m

Track B minimum progress:
    full ATE <= 35.2409m

Track B fail:
    no full candidate <= 35.2409m
```

如果 B1 有效，说明 stable-anchor-only long write 是主线。  
如果 B2 有效，说明 no-long-write / risk veto 是主线。  
如果 B3 有效，说明 lifecycle 是关键，后续可以专门做 long/short memory。  
如果 B4 有效，说明 TTT 的核心是 delta direction，而不是 token role。

---

## 7. Phase 3：只在有证据时做最小组合

如果 Track A 和 Track B 都没有达到最低有效进展，不做组合。

如果其中一个达到 minimum progress，而另一个失败，也不做组合，先总结机制。

只有当：

```text
Track A 有 full ATE <= 35.2409m
且 Track B 有 full ATE <= 35.2409m
```

才允许一个组合候选：

```text
Best Semantic READ + Best New TTT Action
```

组合必须满足：

```text
no chunk id
no manual percentage
no SWA main control
runtime <= 28min
```

组合成功标准：

```text
combo full ATE <= min(best_A, best_B) - 0.3m
```

如果组合不如单项，记录为 conflict，不继续组合矩阵。

---

## 8. Phase 4：跨 sequence sanity，只有成功后才做

跨 sequence 不是本轮主目标。只有当 KITTI01 上出现：

```text
full ATE <= 35.2409m
```

才允许跑 KITTI00/02/05 sanity。

规则：

```text
同一配置直接跑。
不允许 per-sequence 调参。
不允许修改阈值。
不允许按 sequence 选择不同策略。
```

记录：

```text
ATE / Rot / FinalErr
relative delta vs H35-equivalent baseline if available
runtime
failure mode
```

跨 sequence 的目的不是打榜，而是检查是否 catastrophic regression。

---

## 9. Codex 执行与失败分流

### 9.1 如果 Phase 0 baseline drift

如果 H35 repeat 不在 `0.05m` 内，Codex 必须：

```text
1. 停止所有 Track A/B。
2. 比对 effective config。
3. 检查 no-chunk / manual percentage audit。
4. 检查 Stage C 是否误开。
5. 修复后重跑 H35 repeat。
```

### 9.2 如果 semantic candidate cache miss

Codex 必须：

```text
1. 不允许 fallback。
2. 生成 cache_miss_report.md。
3. 检查 chunk frame range / global frame id / Stage C cache path。
4. 修 cache 后只重跑 affected smoke。
```

### 9.3 如果 Track A 704F 全失败

Codex 不允许继续 full。必须输出：

```text
semantic_failure_report.md
包含：
    D_sem 与 D_base 差异是否太小；
    affected source mass 是否太低；
    static rescue 是否误伤；
    Stage C label 是否过粗；
    是否应退回 semantic-only diagnostic。
```

### 9.4 如果 Track B 704F 全失败

Codex 不允许再扫 threshold。必须输出：

```text
ttt_action_failure_report.md
包含：
    role collapse 是否发生；
    stable/risk mass 是否极端；
    post-zp delta energy 是否偏离 H35/C9；
    candidate/native cosine 是否异常；
    runtime 是否超限；
    哪个新 action class 最有可能继续。
```

### 9.5 如果某个候选 borderline

如果候选比 H35 704F 差不超过 `0.25m`，并且 runtime pass，可以允许一条 full diagnostic。避免 v54 那种 borderline 被过早挡掉。

### 9.6 如果 full run 超过 28 分钟

Codex 必须：

```text
1. 停止该 candidate 后续 full。
2. 找出最慢 chunk。
3. 标记是 Stage C、TTT risk、post-zp trace、还是 logging 造成。
4. 默认优先关闭非必要 trace，而不是修改模型逻辑。
```

---

## 10. 本轮最终报告必须回答的问题

最终报告不允许只写 “fail”。必须回答：

```text
1. H35 clean baseline 是否稳定？
2. 语义是否能在 H35 上带来 >=2m / >=1m / >=0.5m 收益？
3. 语义收益来自 C23 reconditioning、high-influence filtering，还是 static rescue？
4. 新 TTT action 是否能带来 >=1m / >=0.5m 收益？
5. 哪种新 TTT action 最有希望：binary anchor、risk veto、two-lifetime、projection commit？
6. full run 是否全部 <=28min？
7. 如果失败，失败是 action space 不够、risk proxy 不对、commit 方向不对，还是 runtime blocker？
8. 下一步应该继续语义、继续新 TTT，还是转向 C9 teacher trace distillation？
```

---

## 11. 本轮最小运行清单

为了控制时间，本轮最大 full run 数量如下：

```text
Phase 0:
    H35 repeat: 1 full

Track A:
    4 x 704F screen
    max 2 x full

Track B:
    4 x 704F screen
    max 2 x full

Combination:
    max 1 x full only if both tracks minimum progress
```

总 full upper bound：

```text
1 + 2 + 2 + 1 = 6 full runs
```

如果 704F screen 全失败，则 full 数量只有 1 条 H35 repeat。这样可以避免再次把一轮实验时间消耗在无效大矩阵上。

---

## 12. 最终判断

本轮不再追求把 clean adaptive TTT 直接逼近 C9，也不再继续修补 tri replay。H35 已经显示 clean 版本在 `35.74m` 平台附近很难继续靠小修推进。因此本轮必须回答两个更本质的问题：

```text
1. 语义是否能在 H35 上真实改善几何？
   成功线：至少 2m，最低有效进展 0.5m。

2. 新 TTT write action 是否比 tri replay adaptive split 更有潜力？
   成功线：至少 1m，最低有效进展 0.5m。
```

如果这两个问题都失败，项目应停止围绕语义 all-memory 和 adaptive tri replay 继续消耗时间，转向新的 TTT action-space 或 C9 teacher behavior 的更强形式化建模。

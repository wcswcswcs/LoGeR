# ACL2 v43：C9 去 chunk-id 化、贡献归因与最小语义 READ 推进 Target-30 实验计划

日期：2026-05-26  
对象：LoGeR / Pipeline v2 / C9_P0_R2 / Semantic Prior Generator / READ path  
当前可部署 best：`C9_P0_R2 = 33.7629421029m`  
阶段目标：先把 KITTI01 full-online ATE 推进到 `33.0m` 以内，再冲 `30.0m` 门槛  
最终目标：在不训练新模型、不用 GT runtime、不使用 absolute chunk-id policy、不针对某个数据集调参的前提下，证明语义信息可以帮助 LoGeR 的记忆管理并带来真实 full-online ATE 改善。

---

## 0. 这轮计划为什么必须重写

上一轮 v42 的最大教训不是“语义完全没用”，而是两个更基本的问题暴露出来了。

第一，当前历史 best `C9_P0_R2` 本身并不是一个干净的机制策略。它的 full ATE 是 `33.7629421029m`，比 `H9_P0_R2 = 34.1257769401m` 好约 `0.3628m`，并且明显改善了 `[400,600)`，但它在 `[200,300)` 反而比 H9 更差。也就是说，C9 是当前 best deployable online 配方，但它更像一个有效的经验组合，而不是一个已经解释清楚、可迁移、可泛化的理论策略。

第二，v42 已经证明“health-selected READ full-online”当前不可用。v42 修复了 C9 no-op 复现问题，F0 R2 精确复现 C9；但 Phase 1 选择出的 chunks 是 `[7,9,12,14,16,17,19]`，并且 F1-F4 全部比 C9 差，best READ candidate `F1` 的 ATE 是 `34.7539112804m`，比 C9 差 `+0.9909691774m`。这说明当前 health detector 不能直接替代 chunk-id，也不能作为部署策略继续扩展。

因此，本轮不能继续做下面这些事：

```text
1. 继续按 absolute chunk id 微调。
2. 继续把 chunk-specific 规则换成尚不可靠的 health-driven chunk 规则。
3. 继续大规模语义 all-memory 矩阵。
4. 继续没有贡献归因地把 C9 当作黑箱 best。
5. 继续只看 short rollout，不产生 full-online 结果。
```

本轮必须做三件更本质的事情：

```text
1. 去掉 C9 里的 absolute chunk-id policy，用固定全局值替代。
2. 建立 C9 组件贡献账本，弄清楚到底哪些设计贡献了 ATE 下降。
3. 在 C9 或 C9-flat 基础上只做极少量 full-online 语义 READ 候选，验证语义能否真实推进 full ATE。
```

这轮不是继续证明“语义可能有用”，而是必须回答：

> 在去掉 chunk-id 设计债之后，语义 + 几何是否还能在 full-online C9 系统上带来实际 ATE 改善？

---

## 1. 项目边界与硬约束

本项目是 **training-free memory control**，不是在 KITTI01 上训练选择器，也不是用人为 chunk-id 打榜。因此 v43 必须遵守下面的硬约束。

### 1.1 禁止项

```text
禁止训练 trigger / selector / classifier / role router。
禁止用 oracle label 拟合规则。
禁止使用 absolute chunk id 作为 runtime 条件。
禁止使用 [200,300) 或任何固定 segment 作为 runtime 条件。
禁止针对 KITTI01 或任何单一数据集单独调 label table / threshold / gamma。
禁止把 short rollout / diagnostic / fixed-window result 写成 deployable success。
禁止在没有 severe TTT health evidence 时做 hard reset TTT。
```

### 1.2 允许项

```text
允许在 KITTI01 上做开发和诊断。
允许用 KITTI01 full-online ATE 判断阶段推进。
允许用同一套固定策略在其他 sequence 上做 sanity check。
允许做 component ablation 和 contribution accounting。
允许使用 VideoMasklet / SemanticKITTI sparse projection 做 offline audit 和 trust calibration。
允许使用语义、D_g、appearance anomaly、source influence 共同生成 token-level READ filtering action。
```

### 1.3 本轮成功必须是 full-online

这轮的最低有效推进不再是 h10/h15 diagnostic，而是 full-online。

```text
最低有效进展：
    C9-derived full ATE <= 33.3m
    或者相对 C9 full ATE 改善 >= 0.5m

阶段成功：
    C9-derived full ATE <= 33.0m

强成功：
    C9-derived full ATE <= 32.0m

Target-30 成功：
    C9-derived full ATE <= 30.0m
```

---

## 2. 核心假设

## H1：C9 的 absolute chunk-id 参数是设计债，可以用固定值替代

### 假设内容

C9 中 `read_beta_frame_chunks`、chunk-specific tri-replay gamma、`commit_ema_chunks=5,6` 等规则带有明显 KITTI01 chunk-id 痕迹。它们可以解释一部分历史 best，但不适合做最终策略。因为这些值相差不是很大，第一步应该用固定值替代，而不是换成尚不可靠的 health-driven chunk selector。

本轮要验证：

```text
C9-flat 是否能保留 C9 的大部分收益？
如果不能，退化主要来自哪个 component？
```

### 需要验证的 C9 chunk-id 设计债

```text
read_beta_frame_chunks = 5-9:4.85, 10-12/16:4.25
tri-replay gamma chunks = 5-9:0.005, 10-12:0.003, chunk16:0.0003
commit_ema_chunks = 5,6
```

这些都必须在 v43 里被替换成 fixed global settings 或者 ablated 掉。

---

## H2：C9 的真实贡献必须拆清楚，否则不能继续在它上面叠语义

### 假设内容

C9 比 H9 只改善约 `0.36m`，但它包含 read beta、TTT tri-replay、native mix、commit EMA、SWA overlap replacement 等多个设计。当前还不知道每个设计贡献了多少，也不知道哪些 component 在互相抵消。

因此本轮必须建立 contribution ledger。

对每个 component $k$，定义 leave-one-out 贡献：

$$
\Delta^-_k = ATE(C9_{-k}) - ATE(C9)
$$

如果 $\Delta^-_k > 0$，说明去掉该 component 会变差，该 component 对 C9 有正贡献。  
如果 $\Delta^-_k < 0$，说明去掉该 component 反而更好，该 component 是设计债或与其它机制冲突。

同时，对于 flat 版本，定义 fixed-value 代价：

$$
\Delta_{flat} = ATE(C9_{flat}) - ATE(C9)
$$

目标不是证明 C9 完美，而是要知道：

```text
哪个组件必须保留？
哪个组件可以去掉？
哪个组件只在 chunk-id map 下有效？
哪个组件会阻碍语义 READ 加入？
```

---

## H3：语义短期最现实的落点是 READ path，不是 SWA / TTT

### 假设内容

v40/v41 已经显示，READ path 是当前语义 + 几何最强的短期信号；SWA 和 TTT 都弱很多。v42 的 health-selected full-online READ 失败，并不意味着 READ 方向错了，而是说明当前 selected chunks 和 action 触发方式不对。

本轮不继续让语义全局控制所有 memory，也不再优先动 SWA/TTT。语义只进入 READ path，并且只使用 fixed, token-level, sequence-agnostic rule：

```text
high influence anomaly source -> READ source attenuation / compact K/V
sky/vegetation/dynamic/shadow 只有在 high-D + high influence + appearance anomaly 条件下才处理
structure low-D source 必须 rescue
```

这里没有 chunk-id，也没有训练 trigger。

---

## H4：TTT 写入当前只是弱 regularizer，不能继续围绕 chunk-local TTT 微扫

### 假设内容

TTT 写入的历史结果显示，freeze/reset 能强烈改变局部病灶，但很容易破坏全局连续性；localized TTGR / tri-replay / scalar gamma 多数是局部 regularization。C9 的 tri-replay 是当前 best 的一部分，但还没证明它是可迁移策略。

本轮 TTT 只做两件事：

```text
1. 贡献归因：看 tri-replay / native mix / commit EMA 到底贡献多少。
2. 固定值去 chunk-id 化：不再按 chunk 微调。
```

除非贡献归因显示 TTT 是主要正贡献，否则本轮不继续设计新的 TTT 语义写入策略。

---

## H5：其他 sequence 只做 sanity，不做调参

### 假设内容

当前最紧急目标是 KITTI01 进入 ATE 30m 门槛。其它 sequence 验证是必要的，但优先级低于先在 01 上拿到真实 full-online 改善。

本轮只在以下条件满足时启动 cross-sequence sanity：

```text
C9-derived full ATE <= 33.0m
或相对 C9 full ATE 改善 >= 0.5m
```

并且 cross-sequence 不允许改任何参数。

---

## 3. Phase 0：C9 no-op 与 effective config 锁定

### 目标

确保所有后续 full-online 结果都和真实 `C9_P0_R2` 可比，避免 v42 F0 R1 那种 baseline drift 再次发生。

### 实验

运行：

```text
V43_P0_F0_C9_LOCKED_REPEAT
```

要求：

```text
ATE = 33.7629421029m ± 0.03m
hmc_rows = 38
effective_config_diff against C9_P0_R2 = empty or expected logging-only
Stage C disabled unless semantic candidate explicitly needs it
outside semantic read candidate, prior_output = None
```

### 必须落盘

```text
phase0_c9_repeat/kitti_benchmark.log
phase0_c9_repeat/results_sim3/results_ate.txt
phase0_c9_repeat/hmc_state_hash.jsonl
phase0_c9_repeat/effective_config.yaml
phase0_c9_repeat/effective_config_diff_vs_C9.json
phase0_c9_repeat/noop_gate_summary.json
```

### 通过标准

```text
abs(ATE - 33.7629421029) <= 0.03m
hmc_rows = 38
no unexpected Stage C / semantic / chunk gate activation
```

如果失败，Codex 必须停止所有后续 full run，只允许做 config diff 修复。

---

## 4. Phase 1：C9 去 chunk-id 化 fixed-value candidates

### 目标

把 C9 中的 absolute chunk-id 规则替换成固定值，建立可迁移的 C9-flat baseline。这个阶段不是为了马上超过 C9，而是为了回答：去 chunk-id 化的代价是多少。

### 固定原则

```text
不允许 read_beta_frame_chunks。
不允许 tri-replay gamma chunks。
不允许 commit_ema_chunks。
不允许 semantic_action_active_chunks。
不允许 absolute chunk id gating。
```

### 候选设计

#### FLAT_01：minimal flat, 保留 C9 主要结构但移除 chunk maps

```text
read beta frame = 4.75 global
read beta SWA = 4.75 global
tri-replay gamma = 0.003 global all chunks
pos / neg / neutral = 0.35 / 0.12 / 0.85
branch = w0
native_mix_scales = 1.10,1.00,1.00
commit EMA disabled
SWA overlap source replacement = C9 same
```

#### FLAT_02：balanced flat

```text
read beta frame = 4.75 global
read beta SWA = 4.75 global
tri-replay gamma = 0.004 global all chunks
pos / neg / neutral = 0.35 / 0.12 / 0.85
branch = w0
native_mix_scales = 1.10,1.00,1.00
commit EMA disabled
SWA overlap source replacement = C9 same
```

#### FLAT_03：conservative write flat

```text
read beta frame = 4.75 global
read beta SWA = 4.75 global
tri-replay gamma = 0.002 global all chunks
pos / neg / neutral = 0.35 / 0.12 / 0.85
branch = w0
native_mix_scales = 1.10,1.00,1.00
commit EMA disabled
SWA overlap source replacement = C9 same
```

#### FLAT_04：read-only C9 flat diagnostic

```text
read beta frame = 4.75 global
read beta SWA = 4.75 global
TTT tri-replay disabled
commit EMA disabled
native_mix_scales = 1.00,1.00,1.00
SWA overlap source replacement = C9 same
```

### 记录指标

每条 full online 必须记录：

```text
ATE / Rot / RPE_t / RPE_r / FinalErr
[200,300), [200,400), [400,600)
rolling50 / rolling100 / rolling200 mean / p90 / worst
boundary_10f / boundary_20f mean / p90 / worst
Sim3 scale
Yaw RMSE
hmc rows
semantic disabled flag
chunk-map usage audit
```

### 判断标准

```text
C9-flat acceptable:
    best FLAT ATE <= C9 + 0.30m

C9-flat promising:
    best FLAT ATE <= C9 + 0.10m

C9-flat breakthrough:
    best FLAT ATE < C9
```

如果所有 FLAT 都比 C9 差超过 `0.5m`，则说明 C9 的 chunk-id maps 不是无害细节，而是主要贡献来源或重要 interaction，此时必须进入 Phase 2 contribution accounting，不允许继续调 flat gamma。

---

## 5. Phase 2：C9 组件贡献账本

### 目标

弄清楚 C9 到底靠什么从 H9 推到 `33.7629m`，并找出哪些 component 是必须保留、哪些是设计债。

### Component 定义

```text
R = C23 READ control / frame-global read beta
W = TTT tri-replay update_conflict_energy
M = native_mix + commit EMA / commit smoothing
S = SWA overlap source replacement
B = chunk-id beta/gamma/EMA map
```

其中 B 是要被移除的设计债，不作为最终策略组件。

### Leave-one-out full runs

以 locked C9 为中心，运行：

```text
ATTR_01_C9_MINUS_READ_MAP_TO_FLAT
    把 read_beta_frame_chunks 替换为 beta=4.75 global。

ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT
    把 tri-replay gamma chunks 替换为 global gamma=0.003/0.004 中 Phase1 best。

ATTR_03_C9_MINUS_COMMIT_EMA
    关闭 commit_ema_chunks。

ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE
    关闭 SWA overlap source replacement。

ATTR_05_C9_MINUS_TTT_TRI_REPLAY
    关闭 tri-replay，只保留 read + SWA + native mix。

ATTR_06_C9_MINUS_NATIVE_MIX
    native_mix_scales = 1.00,1.00,1.00。

ATTR_07_C9_NO_CHUNK_ID_ALL
    同 Phase1 best flat，用于与其它 ablation 统一对比。
```

### 可选 add-one full runs

如果资源允许，做 4 条 add-one：

```text
ADD_01_BASE_PLUS_READ
ADD_02_BASE_PLUS_SWA
ADD_03_BASE_PLUS_TTT_TRI_GLOBAL
ADD_04_BASE_PLUS_READ_PLUS_SWA
```

其中 BASE 是 H9-like fixed protocol，不使用 chunk map。

### 贡献计算

对每个 component 计算：

$$
\Delta^-_k = ATE(C9_{-k}) - ATE(C9)
$$

对 segment 也计算：

$$
\Delta^-_{k,seg} = Err_{seg}(C9_{-k}) - Err_{seg}(C9)
$$

记录成：

```text
component_contribution_ate.csv
component_contribution_segments.csv
component_contribution_rolling.csv
component_interaction_notes.md
```

### 判断标准

```text
major positive component:
    removing it worsens ATE by >= 0.5m

moderate positive component:
    removing it worsens ATE by 0.2m to 0.5m

neutral component:
    removing it changes ATE by < 0.2m

harmful component:
    removing it improves ATE by >= 0.2m
```

### 可视化

必须生成：

```text
component_ate_waterfall.png
component_segment_delta_bar.png
component_rolling100_delta_heatmap.png
component_boundary_delta_plot.png
```

### 分流规则

如果 `R` 是最大正贡献：Phase 3 优先做语义 READ。  
如果 `S` 是最大正贡献：语义/SWA 要重新规划，但本轮不急于 full matrix。  
如果 `W` 是最大正贡献：TTT tri-replay 是 C9 核心，需要做 fixed-value TTT 但不能回 chunk-id。  
如果 `B` 是主要贡献：说明 C9 过拟合 chunk-id，必须把 C9 退回 diagnostic baseline，不再把它当最终策略框架。  
如果没有单个组件贡献超过 `0.3m`：说明 C9 是高阶 interaction，需要做 small factorial，而不是继续单组件调参。

---

## 6. Phase 3：最小语义 READ full-online 候选

### 目标

在 C9 或 best C9-flat 基础上，只测试少数几个最有希望的 semantic READ 候选，争取拿到真实 full-online ATE 改善。

本阶段不再用 health-selected chunk list，因为 v42 已经证明当前 health-selected chunks 导致 full-online 回退。语义 READ action 必须是 token-level / source-level deterministic rule，在全序列中根据 token 条件生效，而不是根据 chunk id 生效。

### 运行基线

优先选择：

```text
BASE = best of {C9 locked, best C9-flat}
```

如果 C9-flat 不比 C9 差超过 `0.3m`，优先在 C9-flat 上测试语义，因为它没有 chunk-id 设计债。  
如果 C9-flat 明显差，则先在 C9 locked 上测试语义，但最终必须回到 dechunked 版本。

### 候选

#### SEM_READ_01：high-influence anomaly READ filtering

```text
action path = frame/global READ only
query tokens = kept
K/V source = compact or attenuate risky source
condition = high D_g + high source influence + appearance/trust anomaly
semantic = used as context, not direct action
static structure = protected
SWA = unchanged
TTT = unchanged
```

#### SEM_READ_02：sky/vegetation/dynamic guarded filtering

```text
action path = frame/global READ only
query tokens = kept
K/V source = attenuate only if:
    semantic in {sky, vegetation, dynamic, low-trust stuff}
    AND high D_g
    AND high source influence
    AND high appearance anomaly or low mask trust
structure anchors protected
SWA = unchanged
TTT = unchanged
```

#### SEM_READ_03：semantic-conditioned C23 residual, read-only

不替换原始 `D_g`，只加 residual：

$$
D_{final}=D_{base}+\lambda(D_{sem}-D_{base})
$$

其中 $\lambda$ 使用固定保守值或能量归一，不由数据集训练得到：

$$
\lambda = \min\left(1,\frac{\rho \cdot RMS(D_{base})}{RMS(D_{sem}-D_{base})+\epsilon}\right)
$$

本轮固定 $\rho=0.25$。

#### SEM_READ_04：anomaly filtering + static rescue

```text
SEM_READ_01
+
road/building/wall/fence low-D high-trust source protection
+
static source mass floor
```

### full-online gate

每个候选直接跑 full online。因为 v41/v42 已经说明 short rollout 与 full 转换不稳定，本阶段必须直接验证 full ATE。

### 记录指标

除全局指标外，每个语义候选必须记录：

```text
source_skip_applied_by_chunk
source_keep_ratio_by_chunk
skipped_source_attention_mass
retained_static_anchor_mass
affected_semantic_group_mass
sky/vegetation/dynamic affected mass
context_empty_source_events
Stage C cache hit rate
semantic action active ratio
```

### 判断标准

```text
minimum progress:
    ATE <= 33.3m or improvement >= 0.5m vs BASE

stage success:
    ATE <= 33.0m

strong success:
    ATE <= 32.0m

Target-30:
    ATE <= 30.0m

safety:
    [400,600) regression <= +1m
    boundary_10f/20f p90 not worse by > +0.3m
    context_empty_source_events = 0
```

如果所有 SEM_READ candidates 都比 BASE 差，则不继续扩展 semantic READ matrix。必须回到 component attribution 或 trajectory-state/scale-state。

---

## 7. Phase 4：最小组合与二次 full 验证

### 触发条件

只有在 Phase 3 至少一个候选满足：

```text
ATE improvement >= 0.3m
or ATE <= 33.3m
```

才允许启动 Phase 4。

### 组合候选

```text
COMBO_01 = best C9-flat + best SEM_READ
COMBO_02 = best C9 locked + best SEM_READ, but no TTT semantic write
COMBO_03 = best SEM_READ + no commit EMA
COMBO_04 = best SEM_READ + fixed global tri gamma selected from Phase1
```

### 目的

确认语义 READ 与 C9 的哪个部分冲突或互补。

### 记录

```text
combo_vs_component_table.csv
combo_ate_waterfall.png
full_trajectory_overlay.png
rolling_window_overlay.png
```

---

## 8. Phase 5：跨 sequence sanity，只诊断不调参

### 触发条件

只有当 KITTI01 满足：

```text
ATE <= 33.0m
or improvement >= 0.5m vs C9
```

才运行。

### 序列

```text
KITTI00
KITTI02
KITTI05
```

### 规则

```text
完全相同参数。
不调 threshold。
不改 semantic label table。
不改 beta / gamma。
不改 reset / chunk id。
```

### 记录

```text
cross_sequence_ate_table.csv
cross_sequence_rot_table.csv
cross_sequence_failure_mode_report.md
```

### 判断

这一步不是为了打榜，而是为了判断：

```text
01 上的改进是否看起来像机制，而不是序列特化。
```

如果其它序列严重回退，仍然可以继续先优化 01，但必须把结果标记为开发集特化，不允许写成通用策略。

---

## 9. Codex 并行执行安排

### Codex A：C9 no-op 与 dechunk full runs

负责 Phase 0 / Phase 1。

输出：

```text
phase0_noop_gate_summary.json
phase1_flat_registry.csv
phase1_flat_comparison.md
```

如果 no-op drift，立即停止所有其它 full run。

### Codex B：component contribution ledger

负责 Phase 2 leave-one-out / add-one。

输出：

```text
component_contribution_ate.csv
component_contribution_segments.csv
component_ate_waterfall.png
component_interaction_notes.md
```

如果发现某 component label 无效或参数未真正改变，必须标记 invalid，不允许计入贡献。

### Codex C：semantic READ candidate full runs

负责 Phase 3 full-online 语义候选。

输出：

```text
semantic_read_full_registry.csv
semantic_action_summary_by_chunk.csv
semantic_read_failure_or_success_report.md
```

如果 Stage C cache miss 或 semantic action inactive，row invalid。

### Codex D：visualization and action audit

负责所有 spatial / influence 可视化。

输出：

```text
RGB + semantic + D_g + source influence overlays
source_skip_before_after_mass.png
semantic_group_action_heatmap.png
rolling_window_ATE_overlay.png
```

### Codex E：final decision report

负责整合所有阶段，生成最终决策。

输出：

```text
v43_final_summary.md
v43_final_summary.json
```

---

## 10. 失败自动分流

### 情况 A：C9-flat 明显差于 C9

如果：

```text
best FLAT ATE > C9 + 0.5m
```

说明 chunk-id maps 是 C9 的重要贡献或强 interaction。Codex 不能继续调 fixed gamma，而应：

```text
1. 运行 Phase 2 contribution attribution。
2. 标记 C9 为 diagnostic best，而不是机制-clean best。
3. 优先寻找无 chunk-id 的 READ/semantic full improvement。
```

### 情况 B：component attribution 显示某组件无贡献

如果 removing component improves ATE：

```text
移除该组件形成 CLEAN baseline。
不要继续把它放入后续 semantic candidate。
```

### 情况 C：semantic READ 全部回退

如果 SEM_READ_01-04 全部比 BASE 差：

```text
不扩大 semantic READ 矩阵。
检查 action audit：
    skip 是否动到 high-attention source？
    static anchor 是否被误伤？
    Stage C semantic 是否低信任？
然后决定语义降级为 diagnostic / trust calibration。
```

### 情况 D：semantic READ 局部改善但 full ATE 差

如果 `[200,300)` 或 rolling100 改善，但 full ATE 回退：

```text
不要继续固定窗口调参。
检查 downstream regression / boundary / [400,600)。
如果后段回退，优先考虑 read-only action 是否过强，而不是动 TTT reset。
```

### 情况 E：C9 + semantic 改善 >= 0.5m 但未进 33

如果：

```text
ATE improvement >= 0.5m but ATE > 33.0m
```

说明语义有真实贡献，但还不够。Codex 可以尝试 Phase 4 minimal combination，不允许新开大矩阵。

### 情况 F：达到 <=33.0m

进入 cross-sequence sanity，但不调参。

---

## 11. 本轮最终决策标准

本轮结束后必须给出四个明确结论。

### 11.1 C9-cleanliness conclusion

```text
C9 的 absolute chunk-id 设计去掉后还能保留多少性能？
C9-flat 是否可作为新 baseline？
```

### 11.2 Contribution conclusion

```text
C9 的 ATE 改善主要来自 READ / SWA / TTT / native_mix / commit_EMA 中哪几项？
是否存在有害组件？
```

### 11.3 Semantic contribution conclusion

```text
语义 READ 是否在 full online 上带来 >=0.5m 改善？
如果没有，失败原因是 action 无效、误伤 anchor、还是与 C9 冲突？
```

### 11.4 Target progress conclusion

```text
是否进入 ATE <= 33.0m？
是否接近 ATE <= 30.0m？
如果没有，下一步是否应该停止 semantic READ 主线，转向 trajectory-state / scale-state / merge-gauge controller？
```

---

## 12. 一句话总结

v43 的目标不是再做一次语义探索，而是要把 C9 从一个 chunk-id-heavy 的经验配方拆成可解释组件，并在去 chunk-id 后验证最小语义 READ full-online 改善。

> 如果这轮能让 C9 从 `33.7629m` 进入 `33m` 以内，就说明语义 + 几何 READ filtering 终于产生了真实 deployable 进展。  
> 如果不能，就要承认当前语义路线主要是 diagnostic，Target-30 主线需要转向 trajectory-state / scale-state / merge-gauge。  

# ACL2 v46B 补充计划：增加 FRAME_ATTN+TTT 与 FRAME_ATTN+SWA 组件归因实验

日期：2026-06-08  
适用计划：`ACL2_v46_C9_NoChunk_ComponentAttribution_AdaptiveTriReplay_Plan.md`  
目的：补齐 C9 组件贡献归因中最关键的两条组合实验，避免只测单组件而无法解释 READ 与 TTT / SWA 的交互。

---

## 1. 为什么必须增加这两个实验

此前的 `ONLY_FRAME_ATTN`、`ONLY_TTT`、`ONLY_SWA` 只能回答每条 memory path 单独启用时有没有用，但不能回答：

```text
READ / frame attention 控制是否需要 TTT write 才能发挥作用？
READ / frame attention 控制是否和 SWA local memory 互补？
TTT 的收益是否依赖 READ cue 先把当前 chunk 的 read path 修干净？
SWA 的收益是否只有在 READ source 被过滤后才显现？
```

因此必须增加：

```text
FRAME_ATTN+TTT
FRAME_ATTN+SWA
```

这两条不是为了刷 ATE，而是为了回答 C9 中最重要的组件归因问题。

---

## 2. 术语边界

### 2.1 FRAME_ATTN / READ 控制

这里指当前 chunk 的 read path 控制，包括：

```text
C23 / D_g read cue
frame/global attention read bias 或等价 source filtering
read_beta 全局固定值
```

它不包含 TTT tri-replay，也不包含 SWA overlap source replacement。

### 2.2 TTT 控制

这里指 TTT write strategy，包括：

```text
stage_d_x_dg_inv_sqrt write score
tri-replay role assignment
tri-replay gamma / positive / neutral / negative action
commit EMA if candidate explicitly includes it
native mix if candidate explicitly includes it
```

注意：`ONLY_TTT` 中允许使用 C23 / D_g 作为 TTT write risk / eligibility 信号，但禁止使用 C23 / D_g 去控制 frame attention read path。这样才能区分：

```text
D_g 作为 read cue 的贡献
vs
D_g 作为 TTT write cue 的贡献
```

### 2.3 SWA 控制

这里指 SWA local memory / overlap source replacement，包括：

```text
SWA overlap source replacement
SWA K/V source target
SWA alpha
SWA keep scope
```

`ONLY_SWA` 允许使用 D_g 选择 SWA source，但禁止启用 frame attention read bias 和 TTT tri-replay。

---

## 3. 新增实验 rows

新增两条 full-online rows：

```text
ATTR_PLUS_01_FRAME_ATTN_TTT
ATTR_PLUS_02_FRAME_ATTN_SWA
```

它们应与已有 `ONLY_FRAME_ATTN`、`ONLY_TTT`、`ONLY_SWA` 使用同一套 no-chunk fixed baseline。

---

## 4. 实验配置定义

### 4.1 `ONLY_FRAME_ATTN`

```text
READ / frame attention control = ON
TTT tri-replay = OFF
TTT write score = stage_d 或 native safe baseline
SWA overlap source replacement = OFF
commit EMA = OFF
native mix = OFF unless part of immutable base
absolute chunk-id policy = OFF
```

目的：测 READ / frame attention 控制本身对 full-online ATE 的贡献。

---

### 4.2 `ONLY_TTT`

```text
READ / frame attention control = OFF
TTT tri-replay = ON
TTT write score = stage_d_x_dg_inv_sqrt
SWA overlap source replacement = OFF
commit EMA = candidate-defined
native mix = candidate-defined
absolute chunk-id policy = OFF
```

目的：测 TTT write strategy 本身对 full-online ATE 的贡献。

关键注意：这里可以计算 D_g，但不能把 D_g 施加到 frame attention read bias。否则它会混入 READ 贡献。

---

### 4.3 `ONLY_SWA`

```text
READ / frame attention control = OFF
TTT tri-replay = OFF
TTT write score = stage_d 或 native safe baseline
SWA overlap source replacement = ON
commit EMA = OFF
native mix = OFF unless part of immutable base
absolute chunk-id policy = OFF
```

目的：测 SWA local memory / overlap replacement 本身对 full-online ATE 的贡献。

---

### 4.4 新增：`FRAME_ATTN+TTT`

```text
READ / frame attention control = ON
TTT tri-replay = ON
TTT write score = stage_d_x_dg_inv_sqrt
SWA overlap source replacement = OFF
commit EMA = candidate-defined
native mix = candidate-defined
absolute chunk-id policy = OFF
```

目的：回答 READ 与 TTT 是否互补。

如果：

```text
FRAME_ATTN+TTT 明显好于 ONLY_FRAME_ATTN 和 ONLY_TTT
```

则说明 READ 先修当前 chunk 的 source / attention，TTT 再安全写入未来 memory，两者存在协同。

如果：

```text
FRAME_ATTN+TTT ≈ ONLY_TTT
```

则说明 READ 对 C9 的贡献很小，C9 主要是 TTT write 配方。

如果：

```text
FRAME_ATTN+TTT 比 ONLY_TTT 更差
```

则说明 READ 与 TTT 在当前配置下冲突，后续不能简单叠加 semantic READ。

---

### 4.5 新增：`FRAME_ATTN+SWA`

```text
READ / frame attention control = ON
TTT tri-replay = OFF
TTT write score = stage_d 或 native safe baseline
SWA overlap source replacement = ON
commit EMA = OFF
native mix = OFF unless part of immutable base
absolute chunk-id policy = OFF
```

目的：回答 READ 与 SWA 是否互补。

如果：

```text
FRAME_ATTN+SWA 明显好于 ONLY_FRAME_ATTN 和 ONLY_SWA
```

则说明 READ filtering 与 SWA local continuity 可以组合。

如果：

```text
FRAME_ATTN+SWA ≈ ONLY_FRAME_ATTN
```

则 SWA overlap replacement 对 full ATE 贡献很小。

如果：

```text
FRAME_ATTN+SWA 比 ONLY_FRAME_ATTN 更差
```

则说明 SWA 在当前 C9-derived setting 下可能伤害 boundary / overlap continuity。

---

## 5. 建议保留但非用户新增要求的 rows

为了完整计算 interaction，建议仍保留：

```text
TTT+SWA
FRAME_ATTN+TTT+SWA
NONE / BASE
```

否则无法完整估计三因素 interaction。

最小完整 factorial 表为：

| Row | FRAME_ATTN | TTT | SWA | 目的 |
|---|---:|---:|---:|---|
| `F000_NONE` | 0 | 0 | 0 | 干净基础 |
| `F100_ONLY_FRAME_ATTN` | 1 | 0 | 0 | READ 单独贡献 |
| `F010_ONLY_TTT` | 0 | 1 | 0 | TTT 单独贡献 |
| `F001_ONLY_SWA` | 0 | 0 | 1 | SWA 单独贡献 |
| `F110_FRAME_ATTN_TTT` | 1 | 1 | 0 | READ × TTT 互补 |
| `F101_FRAME_ATTN_SWA` | 1 | 0 | 1 | READ × SWA 互补 |
| `F011_TTT_SWA` | 0 | 1 | 1 | TTT × SWA 互补 |
| `F111_ALL_THREE` | 1 | 1 | 1 | 三者组合 |

用户要求新增的是 `F110` 和 `F101`；为了归因完整性，`F011` 与 `F111` 应保留。

---

## 6. 记录指标

每条 row 必须记录 full-online 指标：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
[200,300)
[400,600)
rolling50_mean / p90 / worst
rolling100_mean / p90 / worst
rolling200_mean / p90 / worst
hmc_rows
frames
```

同时必须记录 action/debug 指标：

```text
frame_attn_read_control_active
read_beta_effective_mean
read_beta_policy
D_g_mean / D_g_p90
TTT tri_replay_applied_count
TTT positive_mass / neutral_mass / negative_mass
TTT update_conflict_energy mean / p90
SWA overlap_replace_applied_count
SWA source_keep_ratio
SWA boundary_10f / boundary_20f
absolute_chunk_id_policy_audit
```

---

## 7. 判断标准

### 7.1 READ 与 TTT 互补判断

定义：

$$
Gain(X)=ATE(F000)-ATE(X)
$$

如果：

$$
Gain(F110) > \max(Gain(F100), Gain(F010)) + 0.20
$$

则认为 READ 与 TTT 有互补。

如果：

$$
|Gain(F110)-Gain(F010)| < 0.10
$$

则认为 READ 在 TTT 存在时贡献近似为零。

如果：

$$
Gain(F110) < Gain(F010)-0.20
$$

则认为 READ 与 TTT 冲突。

---

### 7.2 READ 与 SWA 互补判断

如果：

$$
Gain(F101) > \max(Gain(F100), Gain(F001)) + 0.20
$$

则认为 READ 与 SWA 有互补。

如果 `F101` 的 ATE 改善但 boundary / rolling worst 恶化，不能直接晋级，必须标为 local trade-off。

---

### 7.3 组件归因输出

本补充实验完成后，final report 必须用自然语言回答：

```text
1. FRAME_ATTN 单独有没有贡献？
2. TTT 单独有没有贡献？
3. SWA 单独有没有贡献？
4. FRAME_ATTN+TTT 是否明显强于二者单独？
5. FRAME_ATTN+SWA 是否明显强于二者单独？
6. C9 的主要收益到底来自 READ、TTT、SWA，还是 interaction？
```

---

## 8. Codex 执行要求

Codex 必须把这两个新增 row 加入 v46 Phase 2。输出：

```text
phase2_factorial_registry.csv
phase2_component_main_effects.csv
phase2_component_interactions.csv
phase2_frame_ttt_swa_summary.md
component_interaction_heatmap.png
segment_delta_stacked_bar.png
```

如果 `FRAME_ATTN+TTT` 或 `FRAME_ATTN+SWA` 的 action debug 显示某个组件没有真的启用，则该 row invalid，必须修 launcher / CLI / HMC 接线后重跑。

---

## 9. 最终补充结论

`FRAME_ATTN+TTT` 和 `FRAME_ATTN+SWA` 是 C9 组件贡献归因中不可缺的两条组合实验。

只测 `ONLY_FRAME_ATTN / ONLY_TTT / ONLY_SWA` 不够，因为 C9 的实际收益很可能来自组件交互，而不是单个路径独立贡献。

因此本补充计划将这两条 row 提升为 v46 Phase 2 的 mandatory rows。

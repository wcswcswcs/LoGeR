# ACL2 v5 补充：explicit dyn cue 约束 TTT 写入实验计划

日期：2026-05-06

目标：验证用户提出的粗略想法：

```text
结合 explicit dyn cue，让 TTT 少写入动态区域，看看是否能帮助 KITTI01 ATE 继续下降。
```

最终成功标准仍按用户更新后的硬指标：

```text
KITTI01 ATE RMSE < 30m
```

---

## 1. 当前上下文

v5 到目前为止有两个重要结论：

1. TTT writing policy 的 D_g / focal-style 变体只能带来小幅相对提升，最好为 `38.2658m`，远未达 `<30m`。
2. 非 TTT writing 的 read intervention 扩层到 `pair/all` 后出现明显跃迁，当前最好为：

```text
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = frame-attention pair/all
write score = stage_d*sqrt(1-D_g)
beta = 4.75
KITTI01 = 36.6803 / 6.3855
```

因此，本补充实验不再单独扩大 TTT 写入矩阵，而是在当前最强 read path 上测试 explicit dyn write veto 是否还能提供额外收益。

---

## 2. 核心假设

### H-exp-write

`explicit_dyn_only` 更接近显式动态物体区域。若 TTT 写入把这些区域写入 fast weights，会污染后续 chunk 的几何记忆。因此：

```text
在 TTT write score 中显式降低 explicit dyn 高响应 token 的写入权重，可能改善 ATE / Rot / drift。
```

与 v4/v5 早期失败的 old_dyn/explicit routing 不同，本实验只把 explicit dyn 用作 write-side veto，不把 read cue 拉向 explicit dyn。

---

## 3. 固定协议

主协议：

```text
sequence = KITTI01 full
mode = hybrid
commit = probe_ttt_write
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
FAST_CUE_EVAL = 1
WRITE_ALPHA = 0.125
```

主 beta：

```text
beta = 4.75
```

因为当前 pair/all C23 beta sweep 的最好点在 `4.75`。

---

## 4. Write policy 候选

### Baseline

| ID | Write score | 用途 |
|---|---|---|
| B0 | `stage_d*sqrt(1-D_g)` | 当前 pair/all best write |
| B1 | `stage_d` | 判断 D_g write eligibility 是否仍必要 |

### explicit dyn veto

设：

```text
E = explicit_dyn_only
D = D_g_locked
S = stage_d base write score
```

候选：

| ID | Write score | 定义 | 直觉 |
|---|---|---|---|
| E1 | `stage_d_x_exp_inv` | `S * (1-E)` | 直接 veto explicit dynamic |
| E2 | `stage_d_x_exp_inv_sqrt` | `S * sqrt(1-E)` | 温和 veto，避免过度关写 |
| E3 | `stage_d_x_exp_inv_sq` | `S * (1-E)^2` | 更强 focal-style dynamic veto |
| E4 | `stage_d_x_dg_exp_inv_sqrt` | `S * sqrt(1-D) * sqrt(1-E)` | 同时要求 D_g 与 explicit 都认为可写 |
| E5 | `stage_d_x_union_dyn_inv` | `S * (1-max(D,E))` | 最强 union veto，只保留双静态区域 |

---

## 5. 最小验证矩阵

第一批只跑 5 个 full run，避免把失败方向放大：

| Run | Cue | Beta | Read layer | Write score | Gate |
|---|---|---:|---|---|---|
| EXPV-01 | C23 past | 4.75 | all | `stage_d_x_exp_inv` | ATE 是否低于 `36.6803` |
| EXPV-02 | C23 past | 4.75 | all | `stage_d_x_exp_inv_sqrt` | 温和 veto gate |
| EXPV-03 | C23 past | 4.75 | all | `stage_d_x_exp_inv_sq` | 强 veto gate |
| EXPV-04 | C23 past | 4.75 | all | `stage_d_x_dg_exp_inv_sqrt` | D_g + explicit 双约束 |
| EXPV-05 | C23 past | 4.75 | all | `stage_d_x_union_dyn_inv` | 最强 veto diagnostic |

若第一批没有任何结果低于 pair/all baseline `36.6803m`，则 explicit dyn write-veto 方向判为未通过 gate，不继续扩大。

若有任一结果优于 baseline，再补：

```text
beta = 4.25 / 5.25
WRITE_ALPHA = 0.10 / 0.15
```

---

## 6. 记录要求

所有结果只写入：

```text
docs/ACL2_v5_实验记录.md
```

记录内容必须包括：

1. benchmark ATE / Rot / RPE；
2. write mean / dynamic mass / corr diagnostics；
3. trajectory diagnostics；
4. 是否达到 `<30m`；
5. 如果失败，明确说明是 explicit dyn 写入 veto 失败，而不是 explicit dyn read cue 本身失败。

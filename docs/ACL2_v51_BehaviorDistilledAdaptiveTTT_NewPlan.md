# ACL2 v51 Behavior-Distilled Adaptive TTT 新计划

日期：2026-06-08（Asia/Singapore）

## 0. 为什么需要新计划

截至 v50，用户硬约束中的工程部分已经能满足：

```text
1. no chunk-wise hand-tuned params: pass
2. no manual tri replay percentage: pass
3. avoid CPU update_conflict slow path: pass
```

但性能目标仍未满足：

```text
C9/P0 = 33.76294210291885
acceptable threshold = 34.06294210291885
v50 best = 35.985306009701524
delta vs C9/P0 = +2.2223639067826753
```

因此，继续在当前 `risk proxy + role threshold + gamma formula` 上盲调不合理。

## 1. 已知事实

| Stage | Best config | ATE | vs C9/P0 | 结论 |
|---|---|---:|---:|---|
| v47 Otsu fused | `AW111 adaptive_writer_fused` | `38.322144227956436` | `+4.5592021250375865` | positive mass 崩坏，gamma 太小 |
| v48 robust fused d_tok | `AW111 adaptive_writer_robust_fused` | `38.339491488384624` | `+4.576549385465775` | role/gamma 修正但无收益 |
| v49 robust fused residual_x_dg | `AW111 adaptive_writer_robust_fused` | `38.17221175936185` | `+4.409269656443001` | risk proxy 小幅改善 |
| v50 robust split residual_x_dg | `AW111 adaptive_writer_robust_split` | `35.985306009701524` | `+2.2223639067826753` | split replay 是大瓶颈，但仍不够 |

关键 insight：

```text
1. Adaptive TTT only 仍很弱：v50 AW010 = 38.68991051176215。
2. Frame read + adaptive split TTT + SWA 才到 35.9853。
3. v50 已优于 fixed-percentage F111 约 0.665m，但距离 C9/P0 仍差 2.22m。
4. 剩余差距大概率来自 read/write/EMA 的交互，而不是 TTT writing 单点。
```

## 2. 新目标

目标仍不变：

```text
no chunk-wise hand tuned params
no manual tri replay percentage
no CPU-heavy update_conflict path
ATE <= 34.06294210291885 on KITTI01 full-online
```

但 v51 不再直接 full-run 猜新公式。
v51 先做 teacher/student 行为归因：

```text
teacher = C9/P0 或 fixed F111/C9 components 的 landed artifacts
student = adaptive no-chunk writer
目标 = 找出 C9 有效行为中哪些统计可以无 chunk-id 地复现
```

## 3. Phase A：写入行为审计，不跑新算法

产物：

```text
tools/acl2_ttt_write_behavior_audit.py
results/.../v51_behavior_audit/
docs/ACL2_v51_BehaviorDistilledAdaptiveTTT_执行日志.md
docs/ACL2_v51_BehaviorDistilledAdaptiveTTT_实验复盘.md
```

输入 runs：

```text
C9/P0 reference
v46B F111 fixed-percentage TTT
v50 AW111 adaptive split
v50 AW010 adaptive TTT only
```

审计字段：

```text
per chunk:
  ATE prefix / rolling error proxy if available
  frame read enabled / SWA enabled / TTT enabled

per layer:
  pos_mass / neu_mass / neg_mass
  pos_delta_norm_mean
  neu_delta_norm_mean
  neg_delta_norm_mean
  gamma
  neutral_lambda
  branch active mask
  update delta norm by branch

aggregates:
  teacher/student delta_norm ratio
  teacher/student gamma-like strength
  teacher/student neutral contribution ratio
  role mass correlation
```

Gate：

```text
如果 v50 和 teacher 的 delta norm / branch contribution 差异无法定位，
不允许继续猜新 formula。
```

## 4. Phase B：无 chunk-id adaptive policy distillation

只在 Phase A 找到差异后实现。

候选策略：

```text
B1 delta-norm matched adaptive split:
  gamma 不直接来自 risk_gap，而是匹配 teacher-like negative delta / positive delta ratio。
  仍不使用 chunk-id。
  仍不使用手工 percentage。

B2 neutral-balance adaptive split:
  neutral_lambda 匹配 teacher-like neutral contribution ratio。
  role masks 仍由 robust risk/prior 自适应产生。

B3 read/write coupled controller:
  beta/read confidence 与 TTT gamma 由同一个 online risk summary 决定。
  禁止 absolute chunk map。

B4 EMA-free or adaptive-EMA audit:
  不恢复 C9 的 commit_ema_chunks=5,6。
  只允许基于 online stability 的 adaptive EMA。
```

禁止项：

```text
禁止 chunk-id maps。
禁止固定 pos/neg percentage。
禁止把 update_conflict_energy 作为 full path 主风险源。
禁止只因 96F ATE 好就推进 full。
```

## 5. Phase C：short gate，避免 full-run 浪费

Anchor chunks：

```text
chunks 5, 10, 16, 24
horizon = 10 or 20
parents = C9/P0 snapshots and v50 snapshots
mode = readonly / probe_native where possible
```

Gate：

```text
candidate must beat v50 AW111 short baseline by >= 0.5m mean,
or show >= 1.0m improvement in at least 2/4 anchors without >0.5m regression in others.
```

只有通过 short gate 才跑 full KITTI01。

## 6. Phase D：full KITTI01 acceptance

Full gate：

```text
ATE <= 34.06294210291885
hmc_rows = 38
frames = 1101
no_chunk_policy_pass = true
adaptive_writer_audit_pass = true
manual tri replay fractions = 0
role mode not fixed/quantile
probe_ttt_write_seconds_mean <= 8s/chunk preferred
```

如果 ATE <= 34.0629：

```text
再跑 KITTI00/KITTI02/KITTI05 sanity，不允许 sequence-specific retune。
```

## 7. Phase E：失败退出条件

停止条件：

```text
1. behavior audit 证明 C9 的主要收益来自 chunk-id schedule 且无法用 online summary 解释；
2. 三个 distillation candidates 都未通过 short gate；
3. full KITTI01 最好仍 > 35.0。
```

若触发停止条件，结论必须写：

```text
当前无 chunk / 无手工 percentage adaptive TTT 不能复现 C9/P0。
C9/P0 不能作为 clean deployable generalized policy claim。
```

## 8. 当前建议

下一步先做 Phase A，不再直接开新 full run。

理由：

```text
v50 已经证明 split replay 是必要组件；
但剩余 2.22m 差距不是继续调 risk/gamma 能可靠解决的。
必须先知道 teacher 到底做了什么，再蒸馏成无 chunk-id 的 online policy。
```

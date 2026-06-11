# ACL2 v49 Residual-Proxy Adaptive TTT No-Chunk 实验复盘

日期：2026-06-08（Asia/Singapore）  
执行日志：`docs/ACL2_v49_ResidualProxyAdaptiveTTT_NoChunk_执行日志.md`  
结果根目录：`results/kitti01_hmc_v2/acl2_v49_residual_proxy_adaptive_ttt_nochunk/`

本复盘只记录真实运行得到的数据。未完成 full run 标注为 `running`，没有落盘指标不补写。

## 当前状态

```text
v49 已实现设备侧 residual risk proxy。
语法检查已通过。
96F smoke 已完成，速度可接受，审计通过。
三条 full KITTI01 run 已完成。
相比 v48 有小幅改善，但仍未达到 C9/P0 33.7±0.3 目标。
```

## 修改审计

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| `loger/pipeline/ttt_write_controller.py` | `_ttt_layer_residual_risk` 去掉 `.cpu()`，在原设备上计算 residual risk | 让 `ttt_residual_x_dg` 成为可测试的 fast conflict-like risk proxy | `py_compile` pass |

## 方法定义

```text
不使用 chunk-wise 参数：
READ_BETA_FRAME_CHUNKS=""
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS=""
TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
TTT_WRITE_COMMIT_EMA_CHUNKS=""

不使用手工 tri replay percentage：
TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC=0.0
TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC=0.0
TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA=0.0

自适应写入：
TTT_WRITE_TRI_REPLAY_ROLE_MODE=adaptive_writer_robust_fused
TTT_WRITE_GRADIENT_REVERSAL_GAMMA=0.0
TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE=ttt_residual_x_dg
```

## 96F smoke 结果

| Run | frames | full | ATE | chunk sec mean | probe TTT sec mean | adaptive gamma mean | role mass | audit |
|---|---:|---|---:|---:|---:|---:|---|---|
| `V49_SMOKE_RESXDG_AW111_96F` | `96` | `False` | `1.2930573692955973` | `33.79590791463852` | `3.828352212905884` | `0.004105180576314322` | `0.26440375132693184 / 0.6014257089959251 / 0.13417054071194595` | `True/True` |

分析：

```text
1. 设备侧 residual_x_dg 比 v48 d_tok 慢：
   3.828352212905884s/chunk vs 2.360940396785736s/chunk。

2. 但它没有回到 update_conflict_energy 的 69-75s/chunk 慢路径。

3. role mass 没有出现 v47 positive mass≈0.665 的崩坏：
   positive≈0.264, neutral≈0.601, negative≈0.134。

4. 96F ATE 不作为最终性能结论，只证明该路径可运行、可审计、速度可接受。
```

## Full KITTI01 结果

| Run | Row | ATE_full | Delta vs C9/P0 | Fixed ref | Delta vs fixed | hmc_rows | chunk sec mean | probe TTT sec mean | adaptive gamma mean |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| `V49_FULL_RESXDG_AW010` | `AW010_ADAPTIVE_TTT_ONLY` | `41.23581999927922` | `+7.472877896360373` | `F010_ONLY_TTT` | `+1.7167157151283448` | `38` | `38.249441171947275` | `3.8385791025663676` | `0.004582297524106444` |
| `V49_FULL_RESXDG_AW110` | `AW110_FRAME_ADAPTIVE_TTT` | `38.1827531698907` | `+4.419811066971853` | `F110_FRAME_ATTN_TTT` | `+1.5138423287454685` | `38` | `40.13206444915972` | `3.991430169657657` | `0.004582297524106444` |
| `V49_FULL_RESXDG_AW111` | `AW111_FRAME_ADAPTIVE_TTT_SWA` | `38.17221175936185` | `+4.409269656443001` | `F111_ALL_THREE` | `+1.5214713089030738` | `38` | `38.03651075614126` | `3.8098207398464807` | `0.004582297524106444` |

role mass：

```text
positive_mass_mean = 0.26737220689915775
neutral_mass_mean  = 0.5875416106125066
negative_mass_mean = 0.14508618282600802
role_modes_seen    = adaptive_writer_robust_fused
role_sources_seen  = adaptive_writer_robust
```

审计：

```text
no_chunk_policy_pass=True
adaptive_writer_audit_pass=True
full_kitti01=True
hmc_rows=38
```

## Gate 判断

目标：

```text
C9/P0 = 33.76294210291885
acceptable threshold = 34.06294210291885
```

判断：

```text
V49 best = V49_FULL_RESXDG_AW111
ATE = 38.17221175936185
delta vs C9/P0 = +4.409269656443001
delta vs acceptable threshold = +4.109269656443001
success = false
```

## 与 v48 对比

| Row | v48 robust d_tok | v49 residual_x_dg | 改善 |
|---|---:|---:|---:|
| `AW010` | `41.438038230648004` | `41.23581999927922` | `0.2022182313687825` |
| `AW110` | `38.34992971436906` | `38.1827531698907` | `0.16717654447835937` |
| `AW111` | `38.339491488384624` | `38.17221175936185` | `0.167279729022775` |

速度对比：

```text
v48 AW111 probe_ttt_write_seconds_mean = 2.3837527789567647
v49 AW111 probe_ttt_write_seconds_mean = 3.8098207398464807

v49 比 v48 慢约 1.426s/chunk，但远低于 update_conflict_energy 的 69-75s/chunk。
```

## 当前安全结论

```text
v49 证明 residual_x_dg fast proxy 比纯 d_tok 有稳定但很小的收益。
它仍没有恢复到 fixed-percentage TTT，更没有接近 C9/P0。
不能 claim adaptive TTT writing 已经解决。
```

## 分析

```text
1. residual_x_dg 的小幅收益说明 risk proxy 的方向是对的：
   比纯 d_tok 更接近 fixed/update-conflict 语义。

2. 但收益只有约 0.167m，远小于当前差距：
   AW111 仍比 fixed-percentage F111 差 1.521m，比 C9/P0 差 4.409m。

3. 因此当前主要缺口不只是 risk source。
   fused single replay 可能丢失了原 tri replay 分支更新差异；
   fixed TTT 的正/中/负分支并非只靠 role mask/gamma 可复现。

4. v49 仍满足用户三条工程约束：
   无 chunk-wise 手工参数；
   无手工 tri replay percentage；
   没有回到 CPU update_conflict 慢路径。
```

## Insight

```text
“更像 conflict 的风险源”能带来一点改善，但不是瓶颈主因。
下一步应尝试 adaptive split/branch semantics：
保持 role 和 gamma 自适应，仍不允许手工 percentage，
但不要把正/中/负三类完全压成 fused single replay。

同时必须控制速度，不能退回 69-75s/chunk 的 CPU update_conflict_energy。
```

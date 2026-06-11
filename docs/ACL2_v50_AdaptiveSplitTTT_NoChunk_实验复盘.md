# ACL2 v50 Adaptive Split TTT No-Chunk 实验复盘

日期：2026-06-08（Asia/Singapore）  
执行日志：`docs/ACL2_v50_AdaptiveSplitTTT_NoChunk_执行日志.md`  
结果根目录：`results/kitti01_hmc_v2/acl2_v50_adaptive_split_ttt_nochunk/`

本复盘只记录真实运行得到的数据。未完成 full run 标注为 `running`，没有落盘指标不补写。

## 当前状态

```text
v50 adaptive split TTT 代码已实现。
语法检查已通过。
96F smoke 已完成，split 路径实际触发，审计通过。
full KITTI01 已完成。
v50 split 明显优于 v49 fused，并且优于 v46B fixed-percentage F111；
但仍未达到 C9/P0 33.7±0.3 目标。
```

## 修改审计

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| `loger/pipeline/ttt_write_controller.py` | 新增 `adaptive_writer_robust_split`，用自适应 gamma/lambda 走正/中/负 split replay | 验证 fused single replay 是否丢失 fixed tri replay 的分支语义 | `py_compile` pass |
| `run_pipeline_abc_v2.py` | CLI choices 接入 split modes | 允许 launcher 选择新策略 | `py_compile` pass |
| `tools/run_v47_adaptive_ttt_writer_candidate.sh` | audit definition 区分 fused/split | 复现和审计时能看出真实 replay 语义 | `bash -n` pass |
| `tools/v47_adaptive_ttt_writer_report.py` | 增加 `adaptive_writer_split_debug_count` | 报告 split 路径触发次数 | `py_compile` pass |

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
TTT_WRITE_TRI_REPLAY_ROLE_MODE=adaptive_writer_robust_split
TTT_WRITE_GRADIENT_REVERSAL_GAMMA=0.0
TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE=ttt_residual_x_dg
```

## 96F smoke 结果

| Run | frames | full | ATE | chunk sec mean | probe TTT sec mean | adaptive gamma | adaptive neutral lambda | role mass | split count | audit |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| `V50_SMOKE_SPLIT_RESXDG_AW111_96F` | `96` | `False` | `1.3508655062568091` | `33.429367423057556` | `4.572635114192963` | `0.004124691038871081` | `0.9224307462573051` | `0.26438067025608486 / 0.6012868881225586 / 0.13433243913782966` | `72` | `True/True` |

分析：

```text
1. split 路径实际触发：adaptive_writer_split_debug_count=72，fused_debug_count=0。
2. 速度比 v49 fused 慢，但没有回到 CPU update_conflict 慢路径。
3. role mass 与 v49 接近，说明本轮主要变量是 replay 语义，不是 role 选择。
4. 96F ATE 不作为最终性能结论。
```

## Full KITTI01 状态

已完成：

| Run | Row | ATE_full | Delta vs C9/P0 | Fixed ref | Delta vs fixed | hmc_rows | probe TTT sec mean |
|---|---|---:|---:|---|---:|---:|---:|
| `V50_FULL_SPLIT_RESXDG_AW010` | `AW010_ADAPTIVE_TTT_ONLY` | `38.68991051176215` | `+4.926968408843301` | `F010_ONLY_TTT` | `-0.8291937723887273` | `38` | `5.185564298378794` |
| `V50_FULL_SPLIT_RESXDG_AW110` | `AW110_FRAME_ADAPTIVE_TTT` | `36.001552976948304` | `+2.2386108740294546` | `F110_FRAME_ATTN_TTT` | `-0.66735786419693` | `38` | `5.086444083013032` |
| `V50_FULL_SPLIT_RESXDG_AW111` | `AW111_FRAME_ADAPTIVE_TTT_SWA` | `35.985306009701524` | `+2.2223639067826753` | `F111_ALL_THREE` | `-0.6654344407572523` | `38` | `5.3981262633675025` |

role mass：

```text
positive_mass_mean = 0.2674835180722133
neutral_mass_mean  = 0.5871237028411954
negative_mass_mean = 0.14539277882516735
adaptive_gamma_mean = 0.0046066328231653755
adaptive_neutral_lambda_mean = 0.9124036325000183
role_modes_seen = adaptive_writer_robust_split
role_sources_seen = adaptive_writer_robust
```

审计：

```text
no_chunk_policy_pass=True
adaptive_writer_audit_pass=True
adaptive_writer_fused_debug_count=0
adaptive_writer_split_debug_count=684
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
V50 best = V50_FULL_SPLIT_RESXDG_AW111
ATE = 35.985306009701524
delta vs C9/P0 = +2.2223639067826753
delta vs acceptable threshold = +1.922363906782674
success = false
```

## 与 v49 / fixed 对比

| Row | v49 fused | v50 split | v50 improvement | fixed ref delta |
|---|---:|---:|---:|---:|
| `AW010` | `41.23581999927922` | `38.68991051176215` | `2.5459094875170675` | `-0.8291937723887273` |
| `AW110` | `38.1827531698907` | `36.001552976948304` | `2.181200192942396` | `-0.66735786419693` |
| `AW111` | `38.17221175936185` | `35.985306009701524` | `2.186905749660326` | `-0.6654344407572523` |

## 分析

```text
1. fused single replay 是重要瓶颈。
   仅把 fused 改成 split，同时保持同样 residual_x_dg risk 和 robust role，
   AW111 改善 2.1869m。

2. v50 已经不是“adaptive TTT 一点没改进”：
   相比 v46B fixed-percentage F111，v50 AW111 好 0.6654m。

3. 但 v50 仍远低于 C9/P0：
   说明 C9 的剩余收益不是只来自 TTT writing 内部三分支语义。
   还包含 C9 的 chunk-specific read beta / gamma schedule / commit EMA 交互，或其它 read/write 耦合。

4. v50 速度可接受但变慢：
   AW111 probe TTT write mean = 5.398s/chunk；
   v49 fused = 3.810s/chunk；
   仍远低于 update_conflict_energy 69-75s/chunk。

5. 当前路线不能继续盲调 risk/gamma。
   必须进入新的计划：行为蒸馏和交互归因，而不是继续 surface-level adaptive TTT-only。
```

## 当前安全结论

```text
v50 是当前 ACL2 adaptive TTT 的最强结果：
无 chunk-wise 手工参数；
无手工 tri replay percentage；
不走 CPU update_conflict 慢路径；
优于 fixed-percentage F111；
但没有达到 C9/P0 33.7±0.3。

因此本阶段收尾为 partial_success / target_fail。
```

不能写：

```text
不能写 adaptive TTT 已达到 C9/P0。
不能写目标完成。
不能继续把 AW010 adaptive TTT only 说成接近原版 LoGeR/C9；它仍然是 38.69。
不能用 v50 的 partial improvement 替代用户要求的 33.7±0.3 gate。
```

## 当前安全结论

```text
v50 目前只证明 adaptive split TTT 可运行、可审计、速度可接受。
它是否比 v49 fused 更接近 C9/P0，需要 full KITTI01。
```

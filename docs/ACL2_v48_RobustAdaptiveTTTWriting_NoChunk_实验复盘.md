# ACL2 v48 Robust Adaptive TTT Writing No-Chunk 实验复盘

日期：2026-06-08（Asia/Singapore）  
执行日志：`docs/ACL2_v48_RobustAdaptiveTTTWriting_NoChunk_执行日志.md`  
结果根目录：`results/kitti01_hmc_v2/acl2_v48_robust_adaptive_ttt_writer_nochunk/`

本复盘只记录真实运行得到的数据。没有运行完成的实验标注为 `not_run`，没有落盘指标不补写。

## 当前状态

```text
v48 robust adaptive TTT writing 代码已实现。
语法检查已通过。
96F smoke 已完成并通过审计。
三条 full KITTI01 run 已完成。
工程审计通过，但 ATE gate 失败。
```

## 修改审计

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| `loger/pipeline/ttt_write_controller.py` | 新增 `adaptive_writer_robust_fused` 等 role modes | 修正 v47 positive mass 过高和 adaptive gamma 过小的问题 | `py_compile` pass |
| `run_pipeline_abc_v2.py` | CLI choices 接入 robust role modes | 允许 launcher 使用新策略 | `py_compile` pass |
| `tools/run_v47_adaptive_ttt_writer_candidate.sh` | 更新 adaptive writer audit 文案 | 让 robust/Otsu 两类自适应策略落盘说明都准确 | `bash -n` pass |

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
TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE=d_tok
```

## 预期验证项

```text
1. no_chunk_policy_pass=True
2. adaptive_writer_audit_pass=True
3. role_modes_seen 包含 adaptive_writer_robust_fused
4. manual positive/negative/neutral frac 均为 0
5. probe_ttt_write_seconds 不回到 update_conflict_energy 的 69-75s/chunk
6. full KITTI01 ATE 需要接近 C9/P0 33.76294210291885；<=34.06294210291885 才可接受
```

## 96F smoke 结果

| Run | frames | full | ATE | chunk sec mean | probe TTT sec mean | adaptive gamma mean | audit |
|---|---:|---|---:|---:|---:|---:|---|
| `V48_SMOKE_ROBUST_AW111_96F` | `96` | `False` | `1.292421` | `31.455756` | `2.360940` | `0.006728` | `True/True` |

role mass 抽样：

```text
pos ≈ 0.2709-0.2762
neutral ≈ 0.5708-0.5866
neg ≈ 0.1397-0.1584
```

分析：

```text
1. 相比 v47 d_tok Otsu 的 positive_mass_mean≈0.6652，robust writer 把 positive mass 降到约 0.27。
2. neutral mass 回到约 0.58，接近 v46B fixed TTT 的 0.529985。
3. adaptive gamma 提升到约 0.006728，不再像 v47 那样只有约 0.000568。
4. probe TTT write mean=2.360940s，没有出现 update_conflict_energy 的 69-75s/chunk 速度灾难。
5. 96F ATE 只用于 smoke，不作为 full KITTI01 结论。
```

## Full KITTI01 结果

| Run | Row | ATE_full | Delta vs C9/P0 | Fixed ref | Delta vs fixed | hmc_rows | chunk sec mean | probe TTT sec mean | adaptive gamma mean |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| `V48_FULL_ROBUST_AW010` | `AW010_ADAPTIVE_TTT_ONLY` | `41.438038230648004` | `+7.675096127729155` | `F010_ONLY_TTT` | `+1.918933946497127` | `38` | `35.85741560082687` | `2.393188953399658` | `0.007298373690757312` |
| `V48_FULL_ROBUST_AW110` | `AW110_FRAME_ADAPTIVE_TTT` | `38.34992971436906` | `+4.586987611450212` | `F110_FRAME_ATTN_TTT` | `+1.6810188732238274` | `38` | `38.174459426026594` | `2.56031732810171` | `0.007298373690757312` |
| `V48_FULL_ROBUST_AW111` | `AW111_FRAME_ADAPTIVE_TTT_SWA` | `38.339491488384624` | `+4.576549385465775` | `F111_ALL_THREE` | `+1.6887510379258472` | `38` | `35.22462919511293` | `2.3837527789567647` | `0.007298373690757312` |

role mass：

```text
positive_mass_mean = 0.2762368164564434
neutral_mass_mean  = 0.5678295477440483
negative_mass_mean = 0.15593364089727402
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
V48 best = V48_FULL_ROBUST_AW111
ATE = 38.339491488384624
delta vs C9/P0 = +4.576549385465775
delta vs acceptable threshold = +4.276549385465775
success = false
```

## 分析

```text
1. v48 确实修复了 v47 的两个明显工程问题：
   positive mass 从约 0.665 降到约 0.276；
   adaptive gamma 从约 0.000568 提高到约 0.007298。

2. v48 没有回退到 update_conflict_energy 的 CPU 慢路径：
   probe TTT write mean 约 2.38-2.56s/chunk，
   明显低于 v47 update_conflict_energy 的 69-75s/chunk。

3. 但 ATE 没有恢复，AW111 仍为 38.33949。
   这说明失败不只是 role mass 或 gamma 标度问题。

4. 当前最可疑的科学/工程缺口是 risk proxy：
   d_tok 虽快，但它没有捕捉 fixed C9 / update-conflict 写入中真正有用的 token-level conflict 语义。

5. fused single replay 也可能丢失了旧 tri replay 的正/中/负分支更新差异。
   但在恢复准确率前，不能回到完全 CPU update_conflict_energy。
```

## Insight

```text
v48 给出了一个清晰负结论：
“无 chunk + 无手工 percentage + 快速 fused adaptive writer”本身不够。
如果 risk source 只用 d_tok，哪怕 role mass 和 gamma 看起来合理，仍会比 fixed-percentage TTT 差约 1.69m，比 C9/P0 差约 4.58m。
下一步应优先寻找 GPU/设备侧的 conflict-like risk proxy，而不是继续调 role mass 阈值。
```

# ACL2 v48 Robust Adaptive TTT Writing No-Chunk 执行日志

日期：2026-06-08（Asia/Singapore）

目标：

```text
1. 不允许 chunk-wise 手工指定参数。
2. TTT writing tri replay 不得手工指定 positive / neutral / negative percentage，必须自适应。
3. 算法必须提速，不能把主要计算甩到 CPU。
4. 目标逼近 C9/P0 ATE = 33.76294210291885，误差 0.3m 可接受。
```

结果根目录：

```text
results/kitti01_hmc_v2/acl2_v48_robust_adaptive_ttt_writer_nochunk/
```

## 1. v47 失败审计

v47 `adaptive_writer_fused` 的关键失败证据：

```text
AW111 frame + adaptive TTT + SWA:
ATE = 38.322144227956436
delta vs C9/P0 = +4.5592021250375865
delta vs v46B F111 fixed TTT = +1.671403777497659
```

审计字段：

```text
role_mode = adaptive_writer_fused
role_source = adaptive_writer_otsu3
positive_mass_mean ≈ 0.6652
neutral_mass_mean ≈ 0.2000
negative_mass_mean ≈ 0.1348
adaptive_gamma_mean ≈ 0.000568
```

分析：

```text
1. d_tok Otsu 把大量低风险 plateau 分到 positive role，positive mass 过高。
2. adaptive gamma 除以 sqrt(token_count)，在 35k-40k token/chunk 下过小。
3. update_conflict_energy risk source 太慢，probe TTT write 约 69-75s/chunk；d_tok 约 3.5s/chunk。
```

## 2. 代码修改

文件：

```text
loger/pipeline/ttt_write_controller.py
run_pipeline_abc_v2.py
tools/run_v47_adaptive_ttt_writer_candidate.sh
```

修改：

```text
1. 新增 role modes:
   adaptive_writer_robust
   adaptive_writer_robust_fused
   robust_adaptive_writer
   robust_adaptive_writer_fused
   no_percentage_robust
   no_percentage_robust_fused

2. robust role assignment:
   safety = normalized_write_prior * (1 - risk)
   danger = risk * (1 + (1 - normalized_write_prior))
   positive / negative role 由 safety/danger 的 median/MAD 或 mean/std 自适应阈值决定。
   不使用 top percentage。

3. robust adaptive gamma:
   gamma = risk_gap * negative_mass * prior_std
   不使用 chunk-id gamma map。
   不使用外部手工 gamma。

4. 保持 fused single replay，避免旧 tri replay 多次 replay 的速度开销。
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  loger/pipeline/ttt_write_controller.py \
  run_pipeline_abc_v2.py \
  tools/v47_adaptive_ttt_writer_report.py

bash -n tools/run_v47_adaptive_ttt_writer_candidate.sh
```

结果：

```text
pass
```

## 3. 96F smoke

命令：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v48_robust_adaptive_ttt_writer_nochunk \
V47_END_FRAME=96 \
V47_TTT_RISK_SOURCE=d_tok \
V47_TTT_ROLE_MODE=adaptive_writer_robust_fused \
tools/run_v47_adaptive_ttt_writer_candidate.sh \
  0 AW111_FRAME_ADAPTIVE_TTT_SWA V48_SMOKE_ROBUST_AW111_96F
```

完成：

```text
START 2026-06-08 20:47:32 Asia/Singapore
DONE  2026-06-08 20:50:01 Asia/Singapore
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v48_robust_adaptive_ttt_writer_nochunk
```

smoke report：

```text
Run = V48_SMOKE_ROBUST_AW111_96F
Row = AW111_FRAME_ADAPTIVE_TTT_SWA
frames = 96
full = False
ATE = 1.292421
chunk_sec_mean = 31.455756
probe_ttt_sec_mean = 2.360940
adaptive_gamma_mean = 0.006728
valid audit = True/True
```

role mass 抽样（前 4 个 hmc rows）：

```text
pos ≈ 0.2709-0.2762
neutral ≈ 0.5708-0.5866
neg ≈ 0.1397-0.1584
```

结论：

```text
1. robust adaptive writer 修正了 v47 Otsu/d_tok 的 positive mass 过高问题。
2. adaptive gamma 从 v47 的约 0.000568 提升到 smoke 的约 0.006728。
3. probe TTT write 没有回退到 update_conflict_energy 的 69-75s/chunk。
4. 96F ATE 不是 full KITTI01 结果，不能用于最终性能结论。
```

## 4. Full KITTI01 启动

命令：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v48_robust_adaptive_ttt_writer_nochunk \
V47_TTT_RISK_SOURCE=d_tok \
V47_TTT_ROLE_MODE=adaptive_writer_robust_fused \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW010_ADAPTIVE_TTT_ONLY V48_FULL_ROBUST_AW010

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v48_robust_adaptive_ttt_writer_nochunk \
V47_TTT_RISK_SOURCE=d_tok \
V47_TTT_ROLE_MODE=adaptive_writer_robust_fused \
tools/run_v47_adaptive_ttt_writer_candidate.sh 1 AW110_FRAME_ADAPTIVE_TTT V48_FULL_ROBUST_AW110

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v48_robust_adaptive_ttt_writer_nochunk \
V47_TTT_RISK_SOURCE=d_tok \
V47_TTT_ROLE_MODE=adaptive_writer_robust_fused \
tools/run_v47_adaptive_ttt_writer_candidate.sh 2 AW111_FRAME_ADAPTIVE_TTT_SWA V48_FULL_ROBUST_AW111
```

启动时间：

```text
2026-06-08 20:51:15 Asia/Singapore
```

完成状态：

```text
V48_FULL_ROBUST_AW111 DONE: 2026-06-08 21:14:41 Asia/Singapore
V48_FULL_ROBUST_AW010 DONE: 2026-06-08 21:15:05 Asia/Singapore
V48_FULL_ROBUST_AW110 DONE: 2026-06-08 21:16:43 Asia/Singapore
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v48_robust_adaptive_ttt_writer_nochunk
```

报告文件：

```text
results/kitti01_hmc_v2/acl2_v48_robust_adaptive_ttt_writer_nochunk/phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_registry.csv
results/kitti01_hmc_v2/acl2_v48_robust_adaptive_ttt_writer_nochunk/phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_report.md
```

关键结果：

| Run | Row | ATE_full | Delta vs C9/P0 | Fixed ref | Delta vs fixed | hmc_rows | chunk sec mean | probe TTT sec mean | pos/neutral/neg mass | adaptive gamma |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|
| `V48_FULL_ROBUST_AW010` | `AW010_ADAPTIVE_TTT_ONLY` | `41.438038230648004` | `+7.675096127729155` | `F010_ONLY_TTT` | `+1.918933946497127` | `38` | `35.85741560082687` | `2.393188953399658` | `0.2762368164564434 / 0.5678295477440483 / 0.15593364089727402` | `0.007298373690757312` |
| `V48_FULL_ROBUST_AW110` | `AW110_FRAME_ADAPTIVE_TTT` | `38.34992971436906` | `+4.586987611450212` | `F110_FRAME_ATTN_TTT` | `+1.6810188732238274` | `38` | `38.174459426026594` | `2.56031732810171` | `0.2762368164564434 / 0.5678295477440483 / 0.15593364089727402` | `0.007298373690757312` |
| `V48_FULL_ROBUST_AW111` | `AW111_FRAME_ADAPTIVE_TTT_SWA` | `38.339491488384624` | `+4.576549385465775` | `F111_ALL_THREE` | `+1.6887510379258472` | `38` | `35.22462919511293` | `2.3837527789567647` | `0.2762368164564434 / 0.5678295477440483 / 0.15593364089727402` | `0.007298373690757312` |

审计：

```text
no_chunk_policy_pass=True
adaptive_writer_audit_pass=True
role_modes_seen=adaptive_writer_robust_fused
role_sources_seen=adaptive_writer_robust
```

结论：

```text
v48 通过无 chunk-wise 手工参数、无手工 tri replay percentage、速度不回退三项工程审计。
但 full KITTI01 ATE 没有逼近 C9/P0。
最佳 V48_FULL_ROBUST_AW111 = 38.339491488384624，距离 acceptable gate 34.06294210291885 仍差 4.276549385465775m。
```

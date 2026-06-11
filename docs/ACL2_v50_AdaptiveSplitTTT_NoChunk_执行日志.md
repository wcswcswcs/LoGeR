# ACL2 v50 Adaptive Split TTT No-Chunk 执行日志

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
results/kitti01_hmc_v2/acl2_v50_adaptive_split_ttt_nochunk/
```

## 1. v49 负结论

v49 `ttt_residual_x_dg + adaptive_writer_robust_fused`：

```text
V49_FULL_RESXDG_AW111:
ATE = 38.17221175936185
delta vs C9/P0 = +4.409269656443001
delta vs F111 fixed-percentage TTT = +1.5214713089030738
probe_ttt_write_seconds_mean = 3.8098207398464807
```

分析：

```text
residual_x_dg 比 d_tok 有小幅收益，但远不足以恢复 fixed tri replay。
下一步验证 fused single replay 是否丢失正/中/负三分支更新语义。
```

## 2. 代码修改

文件：

```text
loger/pipeline/ttt_write_controller.py
run_pipeline_abc_v2.py
tools/run_v47_adaptive_ttt_writer_candidate.sh
tools/v47_adaptive_ttt_writer_report.py
```

修改：

```text
1. 新增 role mode:
   adaptive_writer_robust_split
   robust_adaptive_writer_split
   no_percentage_robust_split

2. split mode 仍使用 adaptive_writer_robust role assignment：
   safety = normalized_write_prior * (1 - risk)
   danger = risk * (1 + (1 - normalized_write_prior))
   阈值来自 median/MAD 或 mean/std，不使用 top percentage。

3. split mode 不使用手工 gamma / neutral lambda：
   gamma = risk_gap * negative_mass * prior_std
   neutral_lambda = 1 - neutral_risk_mean
   二者均由当前 chunk/layer 的 risk 和 role mask 自适应计算。

4. split mode 不走 fused single replay，而是分别 replay positive / neutral / negative 三类，
   再用 adaptive gamma/lambda 合成分支更新。
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
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v50_adaptive_split_ttt_nochunk \
V47_END_FRAME=96 \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_robust_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh \
  0 AW111_FRAME_ADAPTIVE_TTT_SWA V50_SMOKE_SPLIT_RESXDG_AW111_96F
```

完成：

```text
START 2026-06-08 21:57:25 Asia/Singapore
DONE  2026-06-08 22:00:02 Asia/Singapore
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v50_adaptive_split_ttt_nochunk
```

smoke report：

```text
Run = V50_SMOKE_SPLIT_RESXDG_AW111_96F
Row = AW111_FRAME_ADAPTIVE_TTT_SWA
frames = 96
full = False
ATE = 1.3508655062568091
chunk_sec_mean = 33.429367423057556
probe_ttt_sec_mean = 4.572635114192963
adaptive_gamma_mean = 0.004124691038871081
adaptive_neutral_lambda_mean = 0.9224307462573051
role mass = 0.26438067025608486 / 0.6012868881225586 / 0.13433243913782966
adaptive_writer_fused_debug_count = 0
adaptive_writer_split_debug_count = 72
no_chunk_policy_pass=True
adaptive_writer_audit_pass=True
```

结论：

```text
split 路径实际触发，速度可接受：
4.57s/chunk 比 v49 fused 的 3.81s/chunk 慢，但远低于 update_conflict_energy 的 69-75s/chunk。
96F ATE 不作为 full KITTI01 结论。
```

## 4. Full KITTI01 启动

命令：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v50_adaptive_split_ttt_nochunk \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_robust_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW010_ADAPTIVE_TTT_ONLY V50_FULL_SPLIT_RESXDG_AW010

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v50_adaptive_split_ttt_nochunk \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_robust_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh 1 AW110_FRAME_ADAPTIVE_TTT V50_FULL_SPLIT_RESXDG_AW110

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v50_adaptive_split_ttt_nochunk \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_robust_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh 2 AW111_FRAME_ADAPTIVE_TTT_SWA V50_FULL_SPLIT_RESXDG_AW111
```

启动时间：

```text
2026-06-08 22:01:39 Asia/Singapore
```

当前状态：

```text
V50_FULL_SPLIT_RESXDG_AW110 DONE: 2026-06-08 22:27:44 Asia/Singapore
V50_FULL_SPLIT_RESXDG_AW010 DONE: 2026-06-08 22:27:52 Asia/Singapore
V50_FULL_SPLIT_RESXDG_AW111 DONE: 2026-06-08 22:29:07 Asia/Singapore
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v50_adaptive_split_ttt_nochunk
```

报告文件：

```text
results/kitti01_hmc_v2/acl2_v50_adaptive_split_ttt_nochunk/phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_registry.csv
results/kitti01_hmc_v2/acl2_v50_adaptive_split_ttt_nochunk/phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_report.md
```

关键结果：

| Run | Row | ATE_full | Delta vs C9/P0 | Fixed ref | Delta vs fixed | hmc_rows | chunk sec mean | probe TTT sec mean | role mass | adaptive gamma |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|
| `V50_FULL_SPLIT_RESXDG_AW010` | `AW010_ADAPTIVE_TTT_ONLY` | `38.68991051176215` | `+4.926968408843301` | `F010_ONLY_TTT` | `-0.8291937723887273` | `38` | `39.61076065741087` | `5.185564298378794` | `0.2674835180722133 / 0.5871237028411954 / 0.14539277882516735` | `0.0046066328231653755` |
| `V50_FULL_SPLIT_RESXDG_AW110` | `AW110_FRAME_ADAPTIVE_TTT` | `36.001552976948304` | `+2.2386108740294546` | `F110_FRAME_ATTN_TTT` | `-0.66735786419693` | `38` | `39.34245571337248` | `5.086444083013032` | `0.2674835180722133 / 0.5871237028411954 / 0.14539277882516735` | `0.0046066328231653755` |
| `V50_FULL_SPLIT_RESXDG_AW111` | `AW111_FRAME_ADAPTIVE_TTT_SWA` | `35.985306009701524` | `+2.2223639067826753` | `F111_ALL_THREE` | `-0.6654344407572523` | `38` | `41.31188210060722` | `5.3981262633675025` | `0.2674835180722133 / 0.5871237028411954 / 0.14539277882516735` | `0.0046066328231653755` |

审计：

```text
no_chunk_policy_pass=True
adaptive_writer_audit_pass=True
role_modes_seen=adaptive_writer_robust_split
role_sources_seen=adaptive_writer_robust
adaptive_writer_fused_debug_count=0
adaptive_writer_split_debug_count=684
```

结论：

```text
v50 split 明显优于 v49 fused：
AW111: 38.17221175936185 -> 35.985306009701524，改善 2.186905749660326m。

v50 split 也优于 v46B fixed-percentage F111：
delta_vs_fixed_percentage_TTT = -0.6654344407572523m。

但仍明显没有达到用户目标：
C9/P0 = 33.76294210291885
acceptable threshold = 34.06294210291885
V50 best = 35.985306009701524
delta vs acceptable threshold = +1.922363906782674m
```

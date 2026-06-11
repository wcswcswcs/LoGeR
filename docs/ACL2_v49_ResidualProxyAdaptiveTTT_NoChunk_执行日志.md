# ACL2 v49 Residual-Proxy Adaptive TTT No-Chunk 执行日志

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
results/kitti01_hmc_v2/acl2_v49_residual_proxy_adaptive_ttt_nochunk/
```

## 1. v48 负结论

v48 `d_tok + adaptive_writer_robust_fused` 通过工程审计但性能失败：

```text
V48_FULL_ROBUST_AW111:
ATE = 38.339491488384624
delta vs C9/P0 = +4.576549385465775
delta vs F111 fixed-percentage TTT = +1.6887510379258472

probe_ttt_write_seconds_mean = 2.3837527789567647
role mass = 0.2762368164564434 / 0.5678295477440483 / 0.15593364089727402
adaptive_gamma_mean = 0.007298373690757312
```

分析：

```text
role mass 和 gamma 已修正，但 ATE 没回来。
下一步不继续调 role 阈值，而是换更像 conflict/update 语义的 fast risk proxy。
```

## 2. 代码修改

文件：

```text
loger/pipeline/ttt_write_controller.py
```

修改：

```text
_ttt_layer_residual_risk 不再把 apply_output_raw/v 拉到 CPU。
改为在原设备上计算 residual risk，并在 per_tok.device 上分配输出。
```

修改目的：

```text
ttt_residual_x_dg 之前虽然是 conflict-like proxy，但 residual risk 构建里存在 CPU transfer。
本轮目标是验证设备侧 residual × d_tok 是否能比纯 d_tok 更接近 fixed/update-conflict 写入语义，
同时避免 update_conflict_energy 的 69-75s/chunk 慢路径。
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
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v49_residual_proxy_adaptive_ttt_nochunk \
V47_END_FRAME=96 \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_robust_fused \
tools/run_v47_adaptive_ttt_writer_candidate.sh \
  0 AW111_FRAME_ADAPTIVE_TTT_SWA V49_SMOKE_RESXDG_AW111_96F
```

完成：

```text
START 2026-06-08 21:21:13 Asia/Singapore
DONE  2026-06-08 21:23:52 Asia/Singapore
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v49_residual_proxy_adaptive_ttt_nochunk
```

smoke report：

```text
Run = V49_SMOKE_RESXDG_AW111_96F
Row = AW111_FRAME_ADAPTIVE_TTT_SWA
frames = 96
full = False
ATE = 1.2930573692955973
chunk_sec_mean = 33.79590791463852
probe_ttt_sec_mean = 3.828352212905884
adaptive_gamma_mean = 0.004105180576314322
role mass = 0.26440375132693184 / 0.6014257089959251 / 0.13417054071194595
no_chunk_policy_pass=True
adaptive_writer_audit_pass=True
```

结论：

```text
v49 smoke 比 v48 d_tok smoke 慢一些：
3.828352212905884s/chunk vs 2.360940396785736s/chunk。

但仍远低于 update_conflict_energy 的 69-75s/chunk。
role mass 没有崩，值得进入 full KITTI01。
96F ATE 不作为 full KITTI01 结论。
```

## 4. Full KITTI01 启动

命令：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v49_residual_proxy_adaptive_ttt_nochunk \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_robust_fused \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW010_ADAPTIVE_TTT_ONLY V49_FULL_RESXDG_AW010

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v49_residual_proxy_adaptive_ttt_nochunk \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_robust_fused \
tools/run_v47_adaptive_ttt_writer_candidate.sh 1 AW110_FRAME_ADAPTIVE_TTT V49_FULL_RESXDG_AW110

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v49_residual_proxy_adaptive_ttt_nochunk \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_robust_fused \
tools/run_v47_adaptive_ttt_writer_candidate.sh 2 AW111_FRAME_ADAPTIVE_TTT_SWA V49_FULL_RESXDG_AW111
```

启动时间：

```text
2026-06-08 21:24:26 Asia/Singapore
```

当前状态：

```text
V49_FULL_RESXDG_AW010 DONE: 2026-06-08 21:49:49 Asia/Singapore
V49_FULL_RESXDG_AW111 DONE: 2026-06-08 21:49:41 Asia/Singapore
V49_FULL_RESXDG_AW110 DONE: 2026-06-08 21:51:11 Asia/Singapore
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v49_residual_proxy_adaptive_ttt_nochunk
```

报告文件：

```text
results/kitti01_hmc_v2/acl2_v49_residual_proxy_adaptive_ttt_nochunk/phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_registry.csv
results/kitti01_hmc_v2/acl2_v49_residual_proxy_adaptive_ttt_nochunk/phase1_adaptive_writer/report_R1/v47_adaptive_ttt_writer_report.md
```

关键结果：

| Run | Row | ATE_full | Delta vs C9/P0 | Fixed ref | Delta vs fixed | hmc_rows | chunk sec mean | probe TTT sec mean | role mass | adaptive gamma |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|
| `V49_FULL_RESXDG_AW010` | `AW010_ADAPTIVE_TTT_ONLY` | `41.23581999927922` | `+7.472877896360373` | `F010_ONLY_TTT` | `+1.7167157151283448` | `38` | `38.249441171947275` | `3.8385791025663676` | `0.26737220689915775 / 0.5875416106125066 / 0.14508618282600802` | `0.004582297524106444` |
| `V49_FULL_RESXDG_AW110` | `AW110_FRAME_ADAPTIVE_TTT` | `38.1827531698907` | `+4.419811066971853` | `F110_FRAME_ATTN_TTT` | `+1.5138423287454685` | `38` | `40.13206444915972` | `3.991430169657657` | `0.26737220689915775 / 0.5875416106125066 / 0.14508618282600802` | `0.004582297524106444` |
| `V49_FULL_RESXDG_AW111` | `AW111_FRAME_ADAPTIVE_TTT_SWA` | `38.17221175936185` | `+4.409269656443001` | `F111_ALL_THREE` | `+1.5214713089030738` | `38` | `38.03651075614126` | `3.8098207398464807` | `0.26737220689915775 / 0.5875416106125066 / 0.14508618282600802` | `0.004582297524106444` |

结论：

```text
v49 比 v48 robust d_tok 有小幅改善：
AW111: 38.339491488384624 -> 38.17221175936185，改善约 0.167279729022775m。

但仍明显失败：
delta vs C9/P0 = +4.409269656443001
delta vs acceptable threshold 34.06294210291885 = +4.109269656443001
```

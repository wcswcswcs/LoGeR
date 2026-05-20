# ACL2 v9 实验复盘：WindowDrift Projection Action Space

日期：2026-05-18  
计划文件：`docs/ACL2_v9_TTT_WindowDrift_Projection_ActionSpace_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v9_window_drift_projection/`

本轮原则：只记录实际落盘结果；没有实现或没有跑通的机制不写成实验结论。

---

## 0. v8 继承基线

v8 最优参考：

| Run | ATE RMSE | Rot RMSE | FinalErr | `[200,300)` | `[400,600)` | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | `34.1903` | `6.5666` | `6.195` | `75.576` | `42.280` | v7/v8 repeat reference |
| `C16ROLE_01` | `34.1583` | `6.5327` | `6.249` | `74.521` | `44.369` | v8 best |

v8 结论已经显示：write-side scalar action space（gamma / neutral / role / chunk16 scalar gate）进入平台，剩余主要问题是窗口级 drift-state 和 read/scale balance。

---

## 1. Batch A：offline window-drift audit 与 surrogate

新增工具：

```text
tools/v9_window_drift_state_audit.py
tools/fit_trajectory_surrogate.py
```

输出：

```text
results/kitti01_hmc_v2/acl2_v9_window_drift_projection/batch_a_window_drift/
results/kitti01_hmc_v2/acl2_v9_window_drift_projection/batch_a_surrogate/
```

关键读数：

| 项目 | 结果 |
|---|---|
| offline audit runs | `70` |
| surrogate usable runs | `70` |
| surrogate Spearman(score, ATE) | `0.960` |
| surrogate Top-5 recall | `0.800` |

Batch A 机制结论：

1. body window 的 `rmse_chunk_rmse_m` / `mean_drift_norm_m` 仍强预测 `[200,300)` 局部病灶。
2. c16 / handoff window 指标强相关 `[400,600)` 下游代价。
3. 现有 `hmc_state_hash.jsonl` 缺少可直接执行 v9 projection oracle 所需的 per-token/per-layer projection groups 与 tri-replay mass debug，因此不能把 projection 机制伪装成已经验证。

---

## 2. Projection oracle 接线复查

输出：

```text
results/kitti01_hmc_v2/acl2_v9_window_drift_projection/v9_projection_oracle_report.md
```

结论：

1. 没有启动有效的 `V9_ORACLEPROJ_*` full run。
2. 当前 online TTT controller 没有接入 GT-aligned window drift direction，也没有 token update 到 pose residual projection 的 action path。
3. 现有 `update_conflict_energy` 是相对 chunk aggregate fast-weight update 的 projection，不等同于 v9 计划里的 GT drift projection oracle。
4. 按计划 gate：projection oracle 不成立，因此不跑 no-GT projection 近似矩阵，转入 H8 read-side fallback。

---

## 3. H8 read-side window beta full runs

工程补充：

```text
run_pipeline_abc_v2.py
    新增 --read_beta_frame_chunks

tools/run_attention_cue_experiment.sh
    新增 READ_BETA_FRAME_CHUNKS 环境变量透传
```

说明：首轮 `02-06` 因显存并发过高 OOM，已保留失败目录；随后按用户指定 GPU `0-3` 约束补跑为 `02R-06R`。以下表格只使用完成的有效 full run。

固定主线继承 `C16ROLE_01`：

```text
chunks 5-9:  gamma 0.005, role 0.35/0.12/0.85
chunks 10-12: gamma 0.003, role 0.35/0.12/0.85
chunk16: gamma 0.0003, role 0.35/0.08/0.85
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
default beta_frame = 4.75 unless run overrides
```

Global metrics 与 trajectory diagnostics：

| Run | Read beta policy | ATE RMSE | Rot RMSE | RPE t | RPE r | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `C16ROLE_01` | v8 best | `34.1583` | `6.5327` | `92.4059` | `0.0082` | `6.249` | `74.521` | `54.740` | `44.369` | reference |
| `READBETA_01` | exit `10-12=4.25`, c16 `4.25`, base `4.75` | `34.1282` | `6.5368` | `92.4055` | `0.0082` | `6.177` | `74.442` | `54.673` | `44.335` | **v9 best** |
| `READBETA_02R` | exit `10-12=4.50`, c16 `4.25`, base `4.75` | `34.1565` | `6.5340` | `92.4057` | `0.0082` | `6.232` | `74.515` | `54.731` | `44.371` | 略过 v8 ATE，但未过 01 |
| `READBETA_03R` | body `5-9=5.00`, exit `10-12=4.50`, c16 `4.25` | `34.1634` | `6.5420` | `92.4061` | `0.0082` | `6.266` | `74.508` | `54.726` | `44.408` | 未过 v8 best |
| `READBETA_04R` | body `5-9=5.00`, base `4.75` | `34.1653` | `6.5404` | `92.4063` | `0.0082` | `6.283` | `74.514` | `54.735` | `44.406` | 未过 v8 best |
| `READBETA_05R` | global beta `4.50` | `34.1824` | `6.5367` | `92.4041` | `0.0082` | `6.350` | `74.504` | `54.717` | `44.497` | 全局降 beta 不成立 |
| `READBETA_06R` | global beta `5.00` | `34.1550` | `6.5101` | `92.4076` | `0.0082` | `6.190` | `74.599` | `54.787` | `44.300` | Rot/后段好，但主病灶差 |

H8 诊断目录：

```text
results/kitti01_hmc_v2/acl2_v9_window_drift_projection/trajectory_diagnostics_h8_readbeta/
results/kitti01_hmc_v2/acl2_v9_window_drift_projection/v9_run_registry.csv
```

---

## 4. H8 阶段 best（后续 H9 已刷新）

```text
V9_H8_READBETA_01_exit425_c16_425_SWKS3
ATE / Rot = 34.1282 / 6.5368
RPE t / r = 92.4055 / 0.0082
FinalErr = 6.177
[200,300) = 74.442
[200,400) = 54.673
[400,600) = 44.335
```

相对 `C16ROLE_01`：

| Metric | C16ROLE_01 | READBETA_01 | 改善 |
|---|---:|---:|---:|
| ATE RMSE | `34.1583` | `34.1282` | `+0.0301m` |
| FinalErr | `6.249` | `6.177` | `+0.0714m` |
| `[200,300)` | `74.521` | `74.442` | `+0.0796m` |
| `[200,400)` | `54.740` | `54.673` | `+0.0671m` |
| `[400,600)` | `44.369` | `44.335` | `+0.0338m` |
| Rot RMSE | `6.5327` | `6.5368` | `-0.0041deg` |

结论：

1. v9 没有达到 strong target `<30m`，也没有达到 weak target `ATE <= 34.00`。
2. 但 H8 read-side fallback 形成了新的 relative best：`READBETA_01=34.1282`，相比 v8 best 改善 `0.0301m`。
3. `READBETA_01` 的收益不是单纯 Rot trade-off：`[200,300)`、`[200,400)`、`[400,600)` 和 FinalErr 都相对 `C16ROLE_01` 改善。
4. 全局 beta 对照不解释这个收益：`global 4.50` 明显回退，`global 5.00` 虽 Rot 和 `[400,600)` 更好，但 `[200,300)` 变差且 ATE 未过 `READBETA_01`。
5. 机制判断：当前最有效的新 action 不是继续改 write-side scalar，而是在 exit/c16 read-side 降低 beta，减轻 post/body handoff 的 read-side过强 coupling，同时保留 body 主窗口的写入结构。

---

## 5. H8 后续建议（已执行为 H9）

优先围绕 `READBETA_01` 做很小的 read-side follow-up，而不是回到 write-side scalar：

| Candidate | 目的 |
|---|---|
| exit `10-12=4.25`, c16 `4.50` | 验证 c16 beta 是否需要和 exit 同步降低 |
| exit `10-12=4.35`, c16 `4.25` | 收窄 `4.25/4.50` 之间的 exit sweet spot |
| exit `10-12=4.25`, c16 `4.25`, body `5-9=4.85` | 测 body read 轻微降温是否保留收益 |
| exit `10-12=4.25` only | 拆分 c16 read beta 是否必要 |

晋级标准建议保持：

```text
ATE < 34.1282
且 [400,600) <= 44.335 或 FinalErr/Yaw 有明确补偿
```

---

## 6. H9 read-side follow-up：围绕 READBETA_01 收窄

本节按 H8 后续建议继续推进，使用用户指定 GPU `0-3` 并线验证。

固定主线仍继承 `C16ROLE_01` 的 write-side 结构：

```text
chunks 5-9:   gamma 0.005, role 0.35/0.12/0.85
chunks 10-12: gamma 0.003, role 0.35/0.12/0.85
chunk16:      gamma 0.0003, role 0.35/0.08/0.85
default read beta_frame = 4.75
```

运行记录：

| Run | GPU | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V9_H9_READBETA2_01_exit425_c16_450_SWKS3` | `0` | `2026-05-18 10:41:20` | `11:14:23` | `33.1 min` |
| `V9_H9_READBETA2_02_exit435_c16_425_SWKS3` | `1` | `2026-05-18 10:41:20` | `11:12:51` | `31.5 min` |
| `V9_H9_READBETA2_03_body485_exit425_c16_425_SWKS3` | `2` | `2026-05-18 10:41:20` | `11:15:09` | `33.8 min` |
| `V9_H9_READBETA2_04_exit425_only_SWKS3` | `3` | `2026-05-18 10:41:20` | `11:13:40` | `32.3 min` |

Global metrics 与 trajectory diagnostics：

| Run | Read beta policy | ATE RMSE | Rot RMSE | RPE t | RPE r | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `C16ROLE_01` | v8 best | `34.1583` | `6.5327` | `92.4059` | `0.0082` | `6.249` | `74.521` | `54.740` | `44.369` | reference |
| `READBETA_01` | exit `10-12=4.25`, c16 `4.25`, base `4.75` | `34.1282` | `6.5368` | `92.4055` | `0.0082` | `6.177` | `74.442` | `54.673` | `44.335` | H8 best |
| `H9_READBETA2_01` | exit `10-12=4.25`, c16 `4.50`, base `4.75` | `34.1287` | `6.5371` | `92.4055` | `0.0082` | `6.177` | `74.442` | `54.673` | `44.337` | 与 H8 best 近似，未过 |
| `H9_READBETA2_02` | exit `10-12=4.35`, c16 `4.25`, base `4.75` | `34.1316` | `6.5353` | `92.4056` | `0.0082` | `6.208` | `74.460` | `54.685` | `44.334` | exit `4.35` 未过 |
| `H9_READBETA2_03` | body `5-9=4.85`, exit `10-12=4.25`, c16 `4.25`, base `4.75` | `34.1258` | `6.5414` | `92.4053` | `0.0082` | `6.189` | `74.410` | `54.651` | `44.354` | **当前 v9 best** |
| `H9_READBETA2_04` | exit `10-12=4.25` only, base `4.75` | `34.1288` | `6.5372` | `92.4055` | `0.0082` | `6.177` | `74.442` | `54.673` | `44.337` | c16 read beta 影响很小 |

诊断目录：

```text
results/kitti01_hmc_v2/acl2_v9_window_drift_projection/trajectory_diagnostics_h9_readbeta2/
results/kitti01_hmc_v2/acl2_v9_window_drift_projection/v9_run_registry.csv
```

H9 结论：

1. 本批仍没有达到 `<30m`，也没有达到 weak target `ATE <= 34.00`。
2. `H9_READBETA2_03` 小幅刷新 v9 best：`34.1258 / 6.5414`。相比 H8 best `READBETA_01=34.1282`，ATE 只改善 `0.0024m`。
3. `H9_READBETA2_03` 的收益主要来自 body/read 轻微降温后主病灶继续下降：`[200,300)` 从 `74.442` 降到 `74.410`，`[200,400)` 从 `54.673` 降到 `54.651`。
4. 代价是姿态和后段略回退：Rot 从 `6.5368` 回到 `6.5414`，`[400,600)` 从 `44.335` 到 `44.354`。但 `[400,600)` 仍优于 v8 best gate `C16ROLE_01=44.369`。
5. `H9_READBETA2_01` 与 `H9_READBETA2_04` 几乎重合，说明 c16 read beta 是否为 `4.25/4.50/default` 不是当前主杠杆；关键更像 exit `10-12=4.25` 加 body `5-9` 的轻微 read cooling。
6. `exit=4.35` 未过 `exit=4.25`，说明 exit read beta sweet spot 仍偏向 `4.25`，不应继续向 `4.50` 方向扫。

---

## 7. 当前 v9 best

```text
V9_H9_READBETA2_03_body485_exit425_c16_425_SWKS3
ATE / Rot = 34.1258 / 6.5414
RPE t / r = 92.4053 / 0.0082
FinalErr = 6.189
[200,300) = 74.410
[200,400) = 54.651
[400,600) = 44.354
```

相对 v8 best `C16ROLE_01`：

| Metric | C16ROLE_01 | H9_READBETA2_03 | 改善 |
|---|---:|---:|---:|
| ATE RMSE | `34.1583` | `34.1258` | `+0.0325m` |
| FinalErr | `6.249` | `6.189` | `+0.0594m` |
| `[200,300)` | `74.521` | `74.410` | `+0.1115m` |
| `[200,400)` | `54.740` | `54.651` | `+0.0890m` |
| `[400,600)` | `44.369` | `44.354` | `+0.0151m` |
| Rot RMSE | `6.5327` | `6.5414` | `-0.0087deg` |

当前判断：

1. v9 read-side window beta 路线能稳定给出 `34.13m` 附近的小幅相对收益，但仍远离 `<30m`。
2. H8/H9 的新增收益已经进入非常窄的平台：四条 H9 中三条在 `34.128-34.132`，best 只进一步改善 `0.0024m`。
3. 若继续尝试，应避免大矩阵扫。下一轮只适合做 2-3 条非常小的 read-side局部验证，例如 body `5-9=4.80/4.90` 与 exit `10-12=4.20/4.25` 的组合；若仍不超过 `34.10`，应按 v9 计划转向 read-side support selection / pose-scale failure，而不是继续 beta 标量微调。

---

## 8. H10 read-side micro：body / exit beta cliff 复查

本节继续执行第 7 节建议，围绕 H9 best 做极窄 read beta 微调。目标是确认 `body=4.85, exit=4.25` 是否还有邻域收益，或是否已经是窄峰。

运行记录：

| Run | GPU | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V9_H10_READBETA3_01_body480_exit425_c16_425_SWKS3` | `0` | `2026-05-18 11:18:49` | `11:52:17` | `33.5 min` |
| `V9_H10_READBETA3_02_body490_exit425_c16_425_SWKS3` | `1` | `2026-05-18 11:18:49` | `11:52:24` | `33.6 min` |
| `V9_H10_READBETA3_03_body485_exit420_c16_425_SWKS3` | `2` | `2026-05-18 11:18:49` | `11:51:02` | `32.2 min` |
| `V9_H10_READBETA3_04_body485_exit430_c16_425_SWKS3` | `3` | `2026-05-18 11:18:49` | `11:50:56` | `32.1 min` |

Global metrics 与 trajectory diagnostics：

| Run | Read beta policy | ATE RMSE | Rot RMSE | RPE t | RPE r | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `H9_READBETA2_03` | body `5-9=4.85`, exit `10-12=4.25`, c16 `4.25` | `34.1258` | `6.5414` | `92.4053` | `0.0082` | `6.189` | `74.410` | `54.651` | `44.354` | current best |
| `H10_READBETA3_01` | body `5-9=4.80`, exit `10-12=4.25`, c16 `4.25` | `35.5362` | `8.3385` | `92.3630` | `0.0083` | `4.344` | `70.621` | `52.400` | `47.294` | 局部病灶降，但全局崩 |
| `H10_READBETA3_02` | body `5-9=4.90`, exit `10-12=4.25`, c16 `4.25` | `35.5225` | `8.3338` | `92.3630` | `0.0083` | `4.357` | `70.604` | `52.385` | `47.280` | 同类回退 |
| `H10_READBETA3_03` | body `5-9=4.85`, exit `10-12=4.20`, c16 `4.25` | `35.5511` | `8.3346` | `92.3631` | `0.0083` | `4.335` | `70.638` | `52.413` | `47.320` | exit 过低导致回退 |
| `H10_READBETA3_04` | body `5-9=4.85`, exit `10-12=4.30`, c16 `4.25` | `35.5701` | `8.3390` | `92.3634` | `0.0083` | `4.323` | `70.666` | `52.437` | `47.352` | exit 4.30 也回退 |

诊断目录：

```text
results/kitti01_hmc_v2/acl2_v9_window_drift_projection/trajectory_diagnostics_h10_readbeta3/
```

H10 结论：

1. 本批没有达到 `<30m`，没有达到 `ATE <= 34.00`，也没有超过当前 v9 best `H9_READBETA2_03=34.1258`。
2. H10 明确暴露 read beta cliff：body/exit read cooling 稍微偏离 H9 best 后，`[200,300)` 可以大幅下降到 `~70.6`，但 ATE 回退到 `35.5m`、Rot 回退到 `8.33deg`、`[400,600)` 恶化到 `47.3m`。
3. 这说明 H9 best 附近不是还有宽阔可扫平台，而是一个很窄的 global balance 点；继续围绕 body/exit beta 做标量微扫预期收益很低。
4. `FinalErr` 下降到 `4.32-4.36` 不能作为晋级理由，因为它伴随 overall ATE 与后段 drift 明显恶化。
5. 当前 v9 best 仍保持：

```text
V9_H9_READBETA2_03_body485_exit425_c16_425_SWKS3
ATE / Rot = 34.1258 / 6.5414
RPE t / r = 92.4053 / 0.0082
[200,300) = 74.410
[400,600) = 44.354
```

下一步判断：

```text
不继续 read beta scalar 微扫。
v9 若继续，应按原计划转向 read-side support selection / pose-scale failure：
    1. window-conditioned support selection，而不是只调 beta；
    2. Sim3 scale drift no-GT proxy；
    3. pose increment consistency / segment-level reranking；
    4. SWA overlap source replacement 的 window-level gate。
```

---

## 9. H11 read-side support selection 首批

本节固定 H9 best 的 read beta map，不再改 beta，只测试 read cue 的 support 稀疏化 / 校准是否能改善 window-level read-side failure。

固定：

```text
read beta:
    chunks 5-9   = 4.85
    chunks 10-12 = 4.25
    chunk16      = 4.25
    default      = 4.75
write-side:
    same as C16ROLE_01 / H9 best
```

运行记录：

| Run | GPU | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V9_H11_RDSUPPORT_01_topk008_SWKS3` | `0` | `2026-05-18 11:56:21` | `12:29:25` | `33.1 min` |
| `V9_H11_RDSUPPORT_02_topk012_SWKS3` | `1` | `2026-05-18 11:56:21` | `12:30:05` | `33.7 min` |
| `V9_H11_RDSUPPORT_03_topk016_SWKS3` | `2` | `2026-05-18 11:56:21` | `12:30:03` | `33.7 min` |
| `V9_H11_RDSUPPORT_04_calib_m08_blend1_SWKS3` | `3` | `2026-05-18 11:56:21` | `12:29:36` | `33.3 min` |

Global metrics 与 trajectory diagnostics：

| Run | Read support policy | ATE RMSE | Rot RMSE | RPE t | RPE r | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `H9_READBETA2_03` | no support postprocess | `34.1258` | `6.5414` | `92.4053` | `0.0082` | `6.189` | `74.410` | `54.651` | `44.354` | current best |
| `H11_RDSUPPORT_01` | `READ_TOPK_FRAC=0.08` | `37.6698` | `8.5703` | `92.3668` | `0.0080` | `3.684` | `71.618` | `53.551` | `50.729` | 强回退 |
| `H11_RDSUPPORT_02` | `READ_TOPK_FRAC=0.12` | `37.3460` | `8.5992` | `92.3677` | `0.0081` | `3.640` | `71.662` | `53.517` | `50.207` | 强回退 |
| `H11_RDSUPPORT_03` | `READ_TOPK_FRAC=0.16` | `37.0339` | `8.5314` | `92.3655` | `0.0081` | `3.626` | `71.487` | `53.343` | `49.854` | 本批 ATE best，但远未过 |
| `H11_RDSUPPORT_04` | `per_frame_quantile`, target `0.08`, blend `1.0` | `37.4950` | `8.5972` | `92.3684` | `0.0081` | `3.603` | `71.523` | `53.481` | `50.323` | 强回退 |

诊断目录：

```text
results/kitti01_hmc_v2/acl2_v9_window_drift_projection/trajectory_diagnostics_h11_readsupport/
```

H11 结论：

1. 本批没有达到 `<30m`，没有达到 `ATE <= 34.00`，也没有超过当前 v9 best `H9_READBETA2_03=34.1258`。
2. read support 稀疏化会显著降低局部病灶段和 endpoint：`[200,300)` 降到 `~71.5`，FinalErr 降到 `3.60-3.68`。
3. 但代价非常大：overall ATE 回退到 `37.03-37.67`，Rot 回退到 `8.53-8.60`，`[400,600)` 恶化到 `49.85-50.73`。
4. 机制判断：当前 cue 的 high-response support 并不是可直接保留的“安全 support”。topk / hard calibration 相当于把 read field 变成 endpoint/local regularizer，破坏全局 scale/orientation balance。
5. 因此不继续做 read topk / per-frame quantile support 稀疏化矩阵。v9 当前所有已接好的 read/write scalar 与 support-selection knob 都没有打开 `<34.0` 空间。

截至 H11，当前 v9 best 仍保持：

```text
V9_H9_READBETA2_03_body485_exit425_c16_425_SWKS3
ATE / Rot = 34.1258 / 6.5414
```

下一步不应继续 full-run 参数微扫。若继续，应先做离线诊断或新机制接线：

```text
1. Sim3 scale drift no-GT proxy / read-side scale controller；
2. pose increment consistency loss 或 segment-level reranking；
3. SWA overlap source replacement 的 window-level gate；
4. 若回到 projection action space，必须先实现真实 per-token update-direction projection debug/action path。
```

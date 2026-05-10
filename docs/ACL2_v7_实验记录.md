# ACL2 v7 实验记录

日期：2026-05-08  
计划文件：`docs/ACL2_v7_TTT_Write_PosNeg_Semantic_Causal_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v7_ttt_posneg/`  
目标：优先探索 TTT 写入的 positive / neutral / negative evidence 策略，并把 `[200,300)` 病灶从粗 freeze 诊断推进到 chunk / branch / layer 可归因。

固定主线，除非单节特别说明：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read = frame pair/all
beta = 4.75
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
RESET_EVERY = 5
SWA = SWKS3-style fixed protocol
    ENABLE_SWA_WRITE_CONTROL = 1
    SWA_WRITE_KEEP_SCOPE = both_overlap
    SWA_WRITE_LAYER_MODE = last
    ENABLE_SWA_OVERLAP_SOURCE_REPLACE = 1
    mode = source
    target = kv
    alpha = 0.50
    layer_mode = last
Stage C cache for semantic no-op = results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/
```

当前参考：

| Reference | ATE RMSE | Rot RMSE | FinalErr | `[200,300)` | Yaw RMSE | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | `36.4161` | `6.6128` | `5.798` | `77.831` | `3.765` | v6 reproducible baseline |
| `TTGR_03` | `36.4061` | `6.1780` | `2.974` | `77.502` | `3.338` | v6 pre-semantic TTGR best |
| `P3G_01` | `36.4017` | `6.2289` | `3.509` | `77.568` | `3.390` | v6 semantic + TTGR tiny ATE best |

成功标准分层：

```text
final success:
    KITTI01 ATE < 30m

relative success:
    ATE 明显超过 P3G_01，或 [200,300) 下降 >= 3-5m 且 overall ATE 不崩

diagnostic success:
    明确定位 chunk / branch / layer / semantic group 的因果贡献
```

---

## 1. Phase 0 / H1 第一批：B0/no-op 与单 chunk freeze

本批目的：

1. 复查 v7 当前代码仍能复现 `B0_SWKS3`。
2. 复查 Mask2Former Stage C cache require-hit + semantic ignored 是 no-op。
3. 按 v7 H1 开始拆 `[200,300)` 病灶：先测单 chunk freeze `4` / `5`。

运行记录：

| Run | Start | Done | Walltime |
|---|---|---:|
| `V7_P0_B0_repeat_SWKS3` | `2026-05-08 04:34:49` | `04:58:29` | `23.7 min` |
| `V7_P0_semantic_noop_cache_read` | `2026-05-08 04:34:49` | `05:01:06` | `26.3 min` |
| `V7_H1_freeze4_SWKS3` | `2026-05-08 04:34:49` | `04:58:13` | `23.4 min` |
| `V7_H1_freeze5_SWKS3` | `2026-05-08 04:34:49` | `04:58:19` | `23.5 min` |

Global metrics：

| Run | 变量 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `V7_P0_B0_repeat_SWKS3` | Stage C off | `36.416102` | `6.612796` | `92.445197` | `0.008169` | 复现 B0，通过 |
| `V7_P0_semantic_noop_cache_read` | Stage C cache read + `spg_v2`, HMC ignore | `36.416102` | `6.612796` | `92.445197` | `0.008169` | 与 B0 完全一致，no-op 通过 |
| `V7_H1_freeze4_SWKS3` | `TTT_FREEZE_CHUNKS=4` | `36.582589` | `6.624986` | `92.447238` | `0.008149` | 全局轻微回退 |
| `V7_H1_freeze5_SWKS3` | `TTT_FREEZE_CHUNKS=5` | `38.972747` | `6.657010` | `92.554963` | `0.008692` | 全局明显回退，错误转移 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_h1a_batch1/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 100f `[400,500)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `56.073` | `47.825` | `3.765` | `31.238916` |
| `P0_noop` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `56.073` | `47.825` | `3.765` | `31.238916` |
| `freeze4` | `36.5826` | `5.887` | `77.971` | `77.376` | `56.826` | `57.210` | `48.762` | `3.785` | `31.255077` |
| `freeze5` | `38.9727` | `12.352` | `34.316` | `41.899` | `37.176` | `72.964` | `61.875` | `3.984` | `31.984218` |

Focus chunk diagnostics：

| Run | chunk6 `[174,206)` | chunk7 `[203,235)` | chunk8 `[232,264)` | chunk9 `[261,293)` | chunk10 `[290,322)` |
|---|---:|---:|---:|---:|---:|
| `B0` | `53.799` | `74.942` | `88.814` | `76.800` | `37.576` |
| `freeze4` | `53.896` | `74.663` | `88.343` | `76.225` | `36.762` |
| `freeze5` | `10.837` | `28.378` | `50.540` | `49.062` | `19.492` |

结论：

1. v7 当前代码和固定协议仍严格复现 B0：`36.416102 / 6.612796`。
2. Mask2Former Stage C cache read + `spg_v2` + HMC ignore 是严格 no-op；可继续作为 semantic cache 回接保护线。
3. `freeze4` 对 `[200,300)` 只有小幅改善，但全局 ATE / FinalErr / Yaw 均略回退，不是策略。
4. `freeze5` 是强 causal signal：它把 `[200,300)` 从 `77.831m` 降到 `41.899m`，chunk6-10 error 也大幅下降；但它把错误转移到 `[400,600)`，全局 ATE 崩到 `38.9727m`，FinalErr 到 `12.352m`。
5. 机制判断：chunk5 的 TTT commit 中确实有导致 `[200,300)` 病灶的有害方向，但同一 commit 也承担了后续 `[400,600)` 的连续性/尺度信息。后续不能 hard freeze；需要 branch/layer/token 级地保留 positive continuity，同时只反转或削弱 negative evidence。

下一步：

- 继续 H1 chunk lesion 组合：`freeze6`、`freeze45`、`freeze56`、`freeze46`。
- 若组合仍显示 chunk5 是关键入口，则补 branch/layer 级 freeze 或 localized TTGR，只处理 chunk5 的 `w0` / late/all layer 负证据。

---

## 2. H1 第二批：chunk freeze 组合定位

本批继续只做 causal diagnostic，不作为候选策略。变量：

```text
TTT_FREEZE_CHUNKS = 6 / 4,5 / 5,6 / 4,6
```

运行记录：

| Run | Start | Done | Walltime |
|---|---|---:|
| `V7_H1_freeze6_SWKS3` | `2026-05-08 05:02:17` | `05:25:57` | `23.7 min` |
| `V7_H1_freeze45_SWKS3` | `2026-05-08 05:02:17` | `05:25:30` | `23.2 min` |
| `V7_H1_freeze56_SWKS3` | `2026-05-08 05:02:17` | `05:26:02` | `23.8 min` |
| `V7_H1_freeze46_SWKS3` | `2026-05-08 05:02:17` | `05:25:09` | `22.9 min` |

Global metrics：

| Run | Freeze chunks | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | none | `36.4161` | `6.6128` | `92.4452` | `0.0082` | reference |
| `freeze4` | `4` | `36.5826` | `6.6250` | `92.4472` | `0.0081` | 小幅局部改善，全局回退 |
| `freeze5` | `5` | `38.9727` | `6.6570` | `92.5550` | `0.0087` | 强 causal，但全局转移错误 |
| `V7_H1_freeze6_SWKS3` | `6` | `43.6320` | `7.7391` | `92.5671` | `0.0100` | chunk6 hard freeze 明显破坏入口连续性 |
| `V7_H1_freeze45_SWKS3` | `4,5` | `39.1159` | `6.6144` | `92.5572` | `0.0087` | 接近 freeze5，chunk4 不是主因 |
| `V7_H1_freeze56_SWKS3` | `5,6` | `60.3998` | `8.7004` | `92.7018` | `0.0111` | 病灶被压低最多，但全局严重崩坏 |
| `V7_H1_freeze46_SWKS3` | `4,6` | `43.6134` | `7.6520` | `92.5687` | `0.0099` | 与 freeze6 类似，chunk6 是危险入口 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_h1a_all/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 100f `[400,500)` | 200f `[400,600)` | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `56.073` | `47.825` | `31.238916` |
| `freeze4` | `36.5826` | `5.887` | `77.971` | `77.376` | `56.826` | `57.210` | `48.762` | `31.255077` |
| `freeze5` | `38.9727` | `12.352` | `34.316` | `41.899` | `37.176` | `72.964` | `61.875` | `31.984218` |
| `freeze6` | `43.6320` | `14.181` | `40.695` | `42.283` | `39.820` | `77.738` | `65.957` | `32.153800` |
| `freeze45` | `39.1159` | `12.315` | `33.511` | `41.029` | `36.943` | `73.330` | `62.131` | `32.001789` |
| `freeze56` | `60.3998` | `21.704` | `29.015` | `26.102` | `49.120` | `101.426` | `85.960` | `33.107264` |
| `freeze46` | `43.6134` | `14.052` | `40.097` | `41.682` | `39.578` | `77.951` | `66.219` | `32.171026` |

Focus chunk diagnostics：

| Run | chunk6 `[174,206)` | chunk7 `[203,235)` | chunk8 `[232,264)` | chunk9 `[261,293)` | chunk10 `[290,322)` |
|---|---:|---:|---:|---:|---:|
| `B0` | `53.799` | `74.942` | `88.814` | `76.800` | `37.576` |
| `freeze4` | `53.896` | `74.663` | `88.343` | `76.225` | `36.762` |
| `freeze5` | `10.837` | `28.378` | `50.540` | `49.062` | `19.492` |
| `freeze6` | `97.614` | `24.009` | `44.864` | `44.769` | `19.651` |
| `freeze45` | `10.634` | `27.527` | `49.649` | `48.116` | `18.796` |
| `freeze56` | `56.257` | `26.471` | `20.027` | `23.091` | `34.621` |
| `freeze46` | `97.044` | `23.338` | `44.174` | `44.247` | `18.883` |

H1 结论：

1. `freeze5` 与 `freeze45` 几乎同类，说明 chunk4 不是 `[200,300)` 主病灶的必要原因；chunk5 是强因果入口。
2. `freeze6` 与 `freeze46` 几乎同类，并且 chunk6 自身 error 被打到 `~97m`，说明 chunk6 commit / continuity 对病灶入口至关重要，不能 hard freeze。
3. `freeze56` 把 `[200,300)` 压到 `26.102m`，首次低于最终目标线，但代价是整体 ATE `60.3998m`、后段 `[400,600)` `85.960m`。这证明 chunks 5/6 确实包含能决定病灶的 TTT state，但 hard freeze 把必要的尺度/连续性也一起删掉。
4. 当前定位：有害方向主要在 chunk5 写入后影响 chunk7-9；chunk6 同时承担入口连续性和错误传播，不能用 freeze 类策略处理。
5. 下一步应从 hard freeze 改成 **localized negative evidence**：保留 chunk5/6 的正向 TTT continuity，只对低 prior / high dynamic token 在指定 branch/layer 做小幅反向或削弱。

下一步策略：

```text
优先实现 chunk-local TTGR：
    target chunks = 5 或 5,6
    branch = w0
    layer = all / late
    gamma = 0.025 / 0.05

若 chunk-local TTGR 仍只有厘米级收益：
    做 pos/neg token decomposition，按 per-frame static topk 保留 positive，
    对 bottom dynamic token 做小负 replay，而不是用同一 signed prior 连续插值。
```

---

## 3. H2：chunk-local TTGR 第一批

H1 显示 chunk5 是强因果入口，但 hard freeze 会把后段连续性一起删掉。因此新增 chunk-local TTGR 开关：

```text
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = comma-separated chunk list
empty = old behavior, all chunks
```

工程验证：

- `run_pipeline_abc_v2.py` 新增 `--ttt_write_gradient_reversal_chunks`。
- `tools/run_attention_cue_experiment.sh` 新增 `TTT_WRITE_GRADIENT_REVERSAL_CHUNKS` 透传。
- `V7_TTGR_CHUNK_SMOKE_c0_g005_e128` 通过；debug 确认 chunk0 `TTGR on`，后续 chunk `mode=none`。

固定协议仍为 SWKS3-style。第一批矩阵：

| Run | Chunks | Layer | Branch | Gamma | 目的 |
|---|---|---|---|---:|---|
| `V7_TTGRL_01_c5_w0all_g005_SWKS3` | `5` | all | `w0` | `0.05` | 直接把 v6 global TTGR 强度局部化到 chunk5 |
| `V7_TTGRL_02_c5_w0late_g005_SWKS3` | `5` | late | `w0` | `0.05` | 测 late-only 是否更像 endpoint regularizer |
| `V7_TTGRL_03_c56_w0all_g005_SWKS3` | `5,6` | all | `w0` | `0.05` | 测 chunk5+6 joint negative 是否对应 freeze56 |
| `V7_TTGRL_04_c5_w0all_g0025_SWKS3` | `5` | all | `w0` | `0.025` | 更温和，避免 chunk5 useful continuity 被反向过度 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---:|
| `V7_TTGRL_01_c5_w0all_g005_SWKS3` | `2026-05-08 05:32:37` | `05:56:18` | `23.7 min` |
| `V7_TTGRL_02_c5_w0late_g005_SWKS3` | `2026-05-08 05:32:37` | `05:55:36` | `23.0 min` |
| `V7_TTGRL_03_c56_w0all_g005_SWKS3` | `2026-05-08 05:32:37` | `05:55:41` | `23.1 min` |
| `V7_TTGRL_04_c5_w0all_g0025_SWKS3` | `2026-05-08 05:32:37` | `05:56:05` | `23.5 min` |

Global metrics：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | vs `P3G_01` ATE | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | `36.4161` | `6.6128` | `92.4452` | `0.0082` | `+0.0144` | baseline |
| `P3G_01` | `36.4017` | `6.2289` | n/a | n/a | reference | previous v6/v7 tiny best |
| `TTGRL_01` | `36.4202` | `6.6838` | `92.4444` | `0.0082` | `+0.0185` | chunk5 gamma `0.05` 过强，回退 |
| `TTGRL_02` | `36.4400` | `6.6647` | `92.4445` | `0.0082` | `+0.0383` | late-only 不成立 |
| `TTGRL_03` | `36.5023` | `6.6627` | `92.4452` | `0.0082` | `+0.1006` | chunk5+6 同时反转过强 |
| `TTGRL_04` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | `-0.1060` | **当前 v7 ATE best** |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_ttgr_localized/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 100f `[400,500)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `56.073` | `47.825` | `3.765` | `31.238916` |
| `P3G_01` | `36.4017` | `3.509` | `77.856` | `77.568` | `56.942` | `56.886` | `48.433` | `3.390` | `31.184830` |
| `TTGRL_01` | `36.4202` | `5.706` | `77.988` | `77.678` | `57.011` | `56.175` | `47.868` | n/a | `31.232786` |
| `TTGRL_02` | `36.4400` | `5.642` | `78.016` | `77.743` | `57.048` | `56.095` | `47.906` | n/a | `31.233398` |
| `TTGRL_03` | `36.5023` | `5.764` | `78.047` | `77.786` | `57.083` | `56.233` | `48.050` | n/a | `31.239250` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `55.863` | `47.582` | n/a | `31.219364` |

Focus chunk diagnostics：

| Run | chunk6 `[174,206)` | chunk7 `[203,235)` | chunk8 `[232,264)` | chunk9 `[261,293)` | chunk10 `[290,322)` |
|---|---:|---:|---:|---:|---:|
| `B0` | `53.799` | `74.942` | `88.814` | `76.800` | `37.576` |
| `P3G_01` | `53.560` | `74.557` | `88.331` | `76.862` | `37.595` |
| `TTGRL_01` | `53.700` | `74.652` | `88.567` | `76.809` | `37.824` |
| `TTGRL_02` | `53.565` | `74.650` | `88.666` | `76.914` | `37.928` |
| `TTGRL_03` | `53.835` | `74.728` | `88.574` | `77.044` | `37.991` |
| `TTGRL_04` | `53.381` | `74.567` | `88.555` | `76.806` | `37.878` |

结论：

1. `TTGRL_04` 是当前 v7 ATE best：`36.2957 / 6.6182`，比 `P3G_01=36.4017` 改善 `0.1060m`。这已经不是单纯厘米级抖动，但仍远高于最终目标 `<30m`。
2. 局部 TTGR 的最佳点不是 `gamma=0.05`，而是更温和的 `gamma=0.025`。这与 H1 一致：chunk5 同时有 harmful direction 和 useful continuity，反转必须非常轻。
3. `TTGRL_04` 没有像 freeze 那样大幅压低 `[200,300)`；它主要改善整体长段平衡：`[400,600)` 从 B0 `47.825` 降到 `47.582`，FinalErr 从 `5.798` 降到 `5.387`，Sim3 scale 也更接近旧平台。
4. `chunk5+6 gamma=0.05` 失败，说明 chunk6 不能被同等强度反转；chunk6 更像 continuity carrier，而不是主要 negative replay target。
5. 下一步继续围绕 chunk5 / w0 / all layer 做 gamma fine sweep，并尝试把 chunk6 只给极弱或不处理。

下一批：

```text
TTGRL fine sweep:
    chunks=5, branch=w0, layer=all
    gamma = 0.0125 / 0.020 / 0.030 / 0.0375

如果 gamma sweet spot 稳定在 0.02-0.03：
    再和 semantic lowstuff=0.70 做组合，验证 P3G semantic soft prior 是否还能叠加。
```

---

## 4. H2：chunk5 localized TTGR gamma fine sweep

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
branch = w0
layer = all
mode = low_prior
```

本批只扫 gamma，目标是确认 `TTGRL_04 gamma=0.025` 是否为稳定 sweet spot。

运行记录：

| Run | Gamma | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V7_TTGRL_05_c5_w0all_g00125_SWKS3` | `0.0125` | `2026-05-08 05:58:27` | `06:20:34` | `22.1 min` |
| `V7_TTGRL_06_c5_w0all_g0020_SWKS3` | `0.0200` | `2026-05-08 05:58:27` | `06:20:06` | `21.7 min` |
| `V7_TTGRL_07_c5_w0all_g0030_SWKS3` | `0.0300` | `2026-05-08 05:58:27` | `06:20:42` | `22.3 min` |
| `V7_TTGRL_08_c5_w0all_g00375_SWKS3` | `0.0375` | `2026-05-08 05:58:27` | `06:20:31` | `22.1 min` |

Global metrics：

| Run | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | `0.0250` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRL_05` | `0.0125` | `36.4296` | `6.6000` | `92.4442` | `0.0082` | 太弱，回退 |
| `TTGRL_06` | `0.0200` | `36.3372` | `6.6273` | `92.4435` | `0.0082` | 第二好，但未过 0.025 |
| `TTGRL_07` | `0.0300` | `36.4374` | `6.6196` | `92.4444` | `0.0082` | 过强，回退 |
| `TTGRL_08` | `0.0375` | `36.4436` | `6.6143` | `92.4451` | `0.0082` | 继续回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_ttgr_localized_fine/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 100f `[400,500)` | 200f `[400,600)` | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `56.073` | `47.825` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `55.863` | `47.582` | `31.219364` |
| `TTGRL_05` | `36.4296` | `5.545` | `78.087` | `77.857` | `57.126` | `56.022` | `47.823` | `31.229547` |
| `TTGRL_06` | `36.3372` | `6.036` | `77.995` | `77.738` | `57.032` | `55.906` | `47.703` | `31.223769` |
| `TTGRL_07` | `36.4374` | `5.532` | `78.121` | `77.804` | `57.096` | `56.028` | `47.843` | `31.235960` |
| `TTGRL_08` | `36.4436` | `5.541` | `78.147` | `77.844` | `57.116` | `55.993` | `47.817` | `31.238752` |

结论：

1. `gamma=0.025` 仍是当前最好点；`0.020` 有同向信号但不够强，`0.030/0.0375` 开始明显回退。
2. 这个 sweet spot 很窄，说明 chunk5 负证据不是越强越好；它更像微小方向校正，而不是强动态 veto。
3. `TTGRL_04` 的提升主要来自长段平衡和后段误差降低，而不是把 `[200,300)` 大幅压下去。它与 freeze5/56 的机制不同：不是删除 chunk5 贡献，而是在保留连续性的同时轻微改变 `w0` 更新方向。
4. 继续纯 gamma 扫没有意义。下一步应改变负证据构造方式：把 positive token 保持为原 write prior，只让最低置信的一小部分 token 进入小负 replay，而不是对所有 token 做连续 signed interpolation。

下一步：

```text
新增 TTGR negative-tail 模式：
    positive tokens: 保留原 prior
    negative tokens: bottom risk fraction 进入 -gamma
    neutral tokens: 保持原 prior 或 1.0

首批只作用 chunk5 / w0 / all layer：
    neg_frac = 0.03 / 0.05 / 0.08
    gamma = 0.025
```

---

## 5. H2：chunk5 negative-tail TTGR

动机：

`TTGRL_04` 证明 chunk5 / w0 / all-layer 的轻量 gradient reversal 有正信号，但 `low_prior` 会把一整段低置信区域连续压成 signed prior。为了验证是否只需要反转最坏的一小撮 token，本批新增 `negative_tail`：

```text
risk = normalize(prior_max - prior)
tail tokens = highest-risk bottom fraction
tail tokens write multiplier = -gamma
other tokens keep original positive prior
```

固定协议：

```text
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_MODE = negative_tail
branch = w0
layer = all
SWA = SWKS3-style fixed protocol
```

工程验证：

- `TTT_WRITE_GRADIENT_REVERSAL_NEGATIVE_FRAC` 已接到 `run_pipeline_abc_v2.py` / `hybrid_memory_controller.py` / `ttt_write_controller.py` / `tools/run_attention_cue_experiment.sh`。
- smoke `V7_TTGRTAIL_SMOKE_c0_frac005_e128` 通过，debug 确认 `negative_mass ~= 0.05`，只在指定 chunk 开启。

运行记录：

| Run | Gamma | Negative frac | Start | Done | Walltime |
|---|---:|---:|---|---|---:|
| `V7_TTGRTAIL_01_c5_frac003_g0025_SWKS3` | `0.025` | `0.03` | `2026-05-08 06:26:43` | `06:50:12` | `23.5 min` |
| `V7_TTGRTAIL_02_c5_frac005_g0025_SWKS3` | `0.025` | `0.05` | `2026-05-08 06:26:43` | `06:49:51` | `23.1 min` |
| `V7_TTGRTAIL_03_c5_frac008_g0025_SWKS3` | `0.025` | `0.08` | `2026-05-08 06:26:43` | `06:50:05` | `23.4 min` |
| `V7_TTGRTAIL_04_c5_frac005_g0020_SWKS3` | `0.020` | `0.05` | `2026-05-08 06:26:43` | `06:49:41` | `23.0 min` |

Global metrics：

| Run | Gamma | Negative frac | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | `0.025` | continuous low-prior | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRTAIL_01` | `0.025` | `0.03` | `36.5181` | `6.6436` | `92.4443` | `0.0082` | 回退 |
| `TTGRTAIL_02` | `0.025` | `0.05` | `36.5189` | `6.6600` | `92.4436` | `0.0082` | 回退 |
| `TTGRTAIL_03` | `0.025` | `0.08` | `36.4294` | `6.6522` | `92.4441` | `0.0082` | 本批最好，但未过 B0/TTGRL_04 |
| `TTGRTAIL_04` | `0.020` | `0.05` | `36.4810` | `6.6649` | `92.4448` | `0.0082` | 回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_ttgr_negative_tail/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRTAIL_01` | `36.5181` | `5.629` | `78.256` | `77.860` | `57.150` | `48.103` | `3.796` | `31.227881` |
| `TTGRTAIL_02` | `36.5189` | `5.607` | `78.229` | `77.861` | `57.137` | `48.124` | `3.807` | `31.221238` |
| `TTGRTAIL_03` | `36.4294` | `5.763` | `78.059` | `77.813` | `57.095` | `47.820` | `3.798` | `31.233986` |
| `TTGRTAIL_04` | `36.4810` | `5.622` | `78.259` | `77.941` | `57.189` | `47.904` | `3.798` | `31.240160` |

结论：

1. `negative_tail` 没有达到 `KITTI01 ATE < 30m`，也没有超过 `TTGRL_04`。
2. 只反转最高风险的一小撮 token 不如连续 `low_prior` 软反转；best `TTGRTAIL_03=36.4294` 仍比 B0 差 `+0.0133m`，比 `TTGRL_04` 差 `+0.1337m`。
3. 这批没有明显压低 `[200,300)`：best `TTGRTAIL_03` 的 `[200,300)=77.813`，只比 B0 好 `0.018m`，远不如 freeze diagnostic。
4. `TTGRL_04` 的优势来自更宽的低置信方向微调，而不是少量 hard negative token。说明 TTT harmful update 不是少数 outlier token，而是 chunk5 里一片弱风险区域对 `w0` 更新方向的累积偏置。
5. 继续扫 `negative_frac` 意义不大。下一步回到当前最优 `chunk5 / w0 / all / low_prior gamma=0.025`，尝试与 Mask2Former semantic soft prior 组合，验证语义能否帮助区分 chunk5 内的 useful continuity 与 harmful direction。

---

## 6. H2/H3：localized TTGR + Mask2Former semantic soft prior

动机：

v6 的 `P3G_01` 说明 semantic soft modulation 与 TTGR 不冲突，但它使用的是全局 `w0 gamma=0.05`。v7 当前最好点已经变成 chunk5-only `gamma=0.025`。本批测试：语义 soft prior 能否在 chunk5 localized TTGR 上继续区分 useful continuity 与 harmful direction。

固定：

```text
Stage C cache = results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full/
SEMANTIC_PRIOR_MODE = spg_v2
HMC_IGNORE_SEMANTIC_PRIOR = 0
write score = stage_d_x_sem_x_dg_inv_sqrt
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
branch = w0
layer = all
SWA = SWKS3-style fixed protocol
```

候选：

| Run | Gamma | Lowstuff value | Structure value | 目的 |
|---|---:|---:|---:|---|
| `V7_P3GL_01_c5g0025_semDg_low070_SWKS3` | `0.025` | `0.70` | `1.00` | 当前 best TTGRL_04 + v6 best semantic soft prior |
| `V7_P3GL_02_c5g0020_semDg_low070_SWKS3` | `0.020` | `0.70` | `1.00` | semantic 后 gamma sweet spot 是否变弱 |
| `V7_P3GL_03_c5g0025_semDg_low085_SWKS3` | `0.025` | `0.85` | `1.00` | lowstuff 更接近 neutral |
| `V7_P3GL_04_c5g0025_semDg_low070_struct115_SWKS3` | `0.025` | `0.70` | `1.15` | structure anchor boost 对照 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_P3GL_01_c5g0025_semDg_low070_SWKS3` | `2026-05-08 06:54:00` | `07:19:36` | `25.6 min` |
| `V7_P3GL_02_c5g0020_semDg_low070_SWKS3` | `2026-05-08 06:54:00` | `07:19:42` | `25.7 min` |
| `V7_P3GL_03_c5g0025_semDg_low085_SWKS3` | `2026-05-08 06:54:00` | `07:18:46` | `24.8 min` |
| `V7_P3GL_04_c5g0025_semDg_low070_struct115_SWKS3` | `2026-05-08 06:54:00` | `07:19:04` | `25.1 min` |

Global metrics：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `P3G_01` | `36.4017` | `6.2289` | `92.4412` | `0.0079` | v6 semantic + TTGR reference |
| `P3GL_01` | `36.4903` | `6.6349` | `92.4457` | `0.0082` | semantic + gamma 0.025 回退 |
| `P3GL_02` | `36.3598` | `6.6469` | `92.4437` | `0.0082` | 本批最好；过 B0/P3G，但未过 TTGRL_04 |
| `P3GL_03` | `36.3750` | `6.6515` | `92.4436` | `0.0082` | lowstuff 0.85 接近，但未过 |
| `P3GL_04` | `36.4903` | `6.6349` | `92.4457` | `0.0082` | 与 P3GL_01 完全重合；structure boost 未改变有效路径 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_semantic_local_ttgr/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `P3G_01` | `36.4017` | `3.509` | `77.856` | `77.568` | `56.942` | `48.433` | `3.390` | `31.184830` |
| `P3GL_01` | `36.4903` | `5.487` | `78.180` | `77.855` | `57.126` | `47.976` | `3.785` | `31.247067` |
| `P3GL_02` | `36.3598` | `6.137` | `77.946` | `77.655` | `56.992` | `47.894` | `3.798` | `31.220307` |
| `P3GL_03` | `36.3750` | `5.907` | `77.977` | `77.718` | `57.035` | `47.772` | `3.815` | `31.226401` |
| `P3GL_04` | `36.4903` | `5.487` | `78.180` | `77.855` | `57.126` | `47.976` | `3.785` | `31.247067` |

Chunk5 semantic write debug：

| Run | Mean write score | Mean sem value | Sem q10/q90 | Corr(score, sem) | Corr(score, D_g) |
|---|---:|---:|---:|---:|---:|
| `P3GL_01` | `0.3470` | `0.8102` | `0.70 / 1.00` | `0.5465` | `-0.7623` |
| `P3GL_02` | `0.3470` | `0.8102` | `0.70 / 1.00` | `0.5465` | `-0.7623` |
| `P3GL_03` | `0.3793` | `0.9051` | `0.85 / 1.00` | `0.4479` | `-0.7874` |
| `P3GL_04` | `0.3470` | `0.8102` | `0.70 / 1.00` | `0.5465` | `-0.7623` |

结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `TTGRL_04=36.2957`。
2. `P3GL_02=36.3598` 说明 semantic soft prior 与 localized TTGR 仍有弱正信号：它超过 B0 和 P3G_01，但低于纯 geometry TTGRL_04。
3. 语义加入后，最佳 gamma 从 `0.025` 往 `0.020` 变弱，说明 semantic value 已经承担了一部分 suppression；继续把 gamma 保持在 `0.025` 会过强。
4. `lowstuff=0.85` 没有超过 `0.70`，说明把 sky/vegetation/grass 完全接近 neutral 不够；它们需要 soft 保留，但不能完全等同 structure。
5. `structure=1.15` 与默认 structure 完全重合，说明当前 HMC write source / normalization 路径对 structure boost 不敏感，不能靠单独抬高 structure scalar 解决。
6. 结论更新：语义目前仍是辅助项，不是主 TTT 策略。当前 v7 最好仍是 `TTGRL_04`。下一步从 chunk-local gamma 改为 **chunk-specific gamma map**：chunk5 保持 `0.025`，chunk4/chunk6 只给极弱反转，验证邻接 state 是否能保留 chunk5 收益并减少后段漂移。

---

## 7. H2：chunk-specific gamma map

动机：

H1/H2 已经收敛出两个事实：

- chunk5 是主要 negative replay target；
- chunk6 是 continuity carrier，不能同强度反转。

因此新增 `TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS`，允许：

```text
4:0.0125,5:0.025,6:0.005
```

这种 chunk-specific gamma map，而不是把多个 chunk 绑在同一个 gamma 上。

工程验证：

- `run_pipeline_abc_v2.py` 新增 `--ttt_write_gradient_reversal_chunk_gammas`；
- `tools/run_attention_cue_experiment.sh` 新增 `TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS`；
- smoke `V7_TTGRMAP_SMOKE_c0c1_e96` 通过，日志确认 chunk0 gamma `0.025`、chunk1 gamma `0.005`、后续 chunk off。

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
branch = w0
layer = all
SWA = SWKS3-style fixed protocol
```

候选：

| Run | Chunk gamma map | 目的 |
|---|---|---|
| `V7_TTGRMAP_01_c4g00125_c5g0025_SWKS3` | `4:0.0125,5:0.025` | 给 chunk4 极弱预处理，保留 chunk5 best |
| `V7_TTGRMAP_02_c5g0025_c6g0005_SWKS3` | `5:0.025,6:0.005` | chunk6 极弱校正 |
| `V7_TTGRMAP_03_c4g00125_c5g0025_c6g0005_SWKS3` | `4:0.0125,5:0.025,6:0.005` | 邻接两端都轻触 |
| `V7_TTGRMAP_04_c5g0025_c6g0010_SWKS3` | `5:0.025,6:0.010` | chunk6 稍强一点的风险测试 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TTGRMAP_01_c4g00125_c5g0025_SWKS3` | `2026-05-08 07:25:25` | `07:49:04` | `23.7 min` |
| `V7_TTGRMAP_02_c5g0025_c6g0005_SWKS3` | `2026-05-08 07:25:25` | `07:48:53` | `23.5 min` |
| `V7_TTGRMAP_03_c4g00125_c5g0025_c6g0005_SWKS3` | `2026-05-08 07:25:25` | `07:48:50` | `23.4 min` |
| `V7_TTGRMAP_04_c5g0025_c6g0010_SWKS3` | `2026-05-08 07:25:25` | `07:48:46` | `23.4 min` |

Global metrics：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRMAP_01` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | 与 TTGRL_04 完全重合；chunk4 weak 无可见 effect |
| `TTGRMAP_02` | `36.3467` | `6.6245` | `92.4435` | `0.0082` | chunk6 gamma 0.005 已回退 |
| `TTGRMAP_03` | `36.3467` | `6.6245` | `92.4435` | `0.0082` | 与 MAP_02 重合；chunk4 仍无 effect |
| `TTGRMAP_04` | `36.4127` | `6.6161` | `92.4426` | `0.0082` | chunk6 gamma 0.010 基本退回 B0 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_ttgr_chunk_gamma_map/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRMAP_01` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRMAP_02` | `36.3467` | `5.484` | `77.855` | `77.685` | `57.008` | `47.662` | `3.778` | `31.223038` |
| `TTGRMAP_03` | `36.3467` | `5.484` | `77.855` | `77.685` | `57.008` | `47.662` | `3.778` | `31.223038` |
| `TTGRMAP_04` | `36.4127` | `5.625` | `77.930` | `77.728` | `57.056` | `47.940` | `3.769` | `31.210547` |

结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04`。
2. chunk4 的 `gamma=0.0125` 对最终 trajectory 没有可见影响：`TTGRMAP_01` 与 `TTGRL_04` 完全重合。
3. chunk6 即使只给 `gamma=0.005` 也会回退，说明 chunk6 的 TTT state 主要是 continuity carrier，不适合作 negative replay。
4. `TTGRL_04` 的机制进一步收窄为：**只动 chunk5，branch w0，all-layer，gamma=0.025**。
5. 下一步不再碰 chunk4/6，也不继续扫 chunk gamma。转向 layer split：同样只动 chunk5/w0/gamma0.025，查明 all-layer 收益是否来自某个 layer 段，还是必须全层轻微转向。

---

## 8. H2：chunk5 / w0 / gamma0.025 的 layer split

动机：

`TTGRL_04` 已把当前最优机制收窄到：

```text
只动 chunk5
branch = w0
mode = low_prior gradient reversal
gamma = 0.025
layer = all
```

本批固定其它协议，只改变 TTGR 作用的 TTT layer 范围，检查 `all-layer` 收益是否其实来自某个层段。

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
SWA = SWKS3-style fixed protocol
semantic = disabled
```

候选：

| Run | Layer mode | 目的 |
|---|---|---|
| `V7_TTGRLY_01_c5_w0early_g0025_SWKS3` | early | 检查早层是否承担主要 geometry correction |
| `V7_TTGRLY_02_c5_w0middle_g0025_SWKS3` | middle | 检查中层是否是 chunk5 harmful direction 来源 |
| `V7_TTGRLY_03_c5_w0late_g0025_SWKS3` | late | 检查 late 层是否单独保留 TTGR 收益 |
| `V7_TTGRLY_04_c5_w0single9_g0025_SWKS3` | single layer 9 | 查一个 middle/late 边界单层点 |

Global metrics：

| Run | Layer mode | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | none | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | all | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRLY_01` | early | `36.4714` | `6.6475` | `92.4456` | `0.0082` | 回退 |
| `TTGRLY_02` | middle | `36.5301` | `6.6325` | `92.4442` | `0.0082` | 回退更明显 |
| `TTGRLY_03` | late | `36.3733` | `6.6053` | `92.4431` | `0.0081` | layer split best；保留部分信号，但未过 all-layer |
| `TTGRLY_04` | single 9 | `36.4181` | `6.6398` | `92.4449` | `0.0082` | 接近 B0，未过 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_ttgr_layer_split/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRLY_01` | `36.4714` | `5.411` | `78.377` | `77.862` | `57.129` | `47.793` | `3.796` | `31.251335` |
| `TTGRLY_02` | `36.5301` | `5.923` | `78.541` | `77.931` | `57.175` | `48.103` | `3.784` | `31.229942` |
| `TTGRLY_03` | `36.3733` | `5.643` | `77.892` | `77.639` | `56.983` | `47.844` | `3.764` | `31.216933` |
| `TTGRLY_04` | `36.4181` | `5.772` | `78.276` | `77.794` | `57.073` | `47.802` | `3.792` | `31.235247` |

结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `TTGRL_04`。
2. late-only 是 layer split 里最接近的点：`TTGRLY_03 = 36.3733 / 6.6053`，它保留了 `[200,300)` 的局部收益，但全局 ATE 和 `[400,600)` 不如 all-layer。
3. early / middle / single9 都没有独立成为有效点，说明 `TTGRL_04` 不是某个单独 layer 段的收益，而是 chunk5 的 `w0` 更新方向需要全层小幅转向。
4. 继续做 layer split 不值得。下一步主动尝试 branch-specific gamma：保留已知有效的 `w0 gamma=0.025`，只给 `w2` 极小 gamma，测试能否在不破坏 ATE 的情况下改善 Rot / Yaw / endpoint。

---

## 9. H2：branch-specific gradient reversal gamma

动机：

layer split 后，当前最佳仍是 `TTGRL_04`：

```text
chunk5 only
branch = w0
layer = all
gamma = 0.025
```

但此前 `w0+w2` late 强反转能改善 Rot / endpoint，却会牺牲 ATE。因此本批新增 branch-specific gamma，让 `w0` 保持当前 best，`w2` 只给极小 negative evidence，测试能否在不破坏 ATE 的情况下改善局部病灶或 rotation。

工程补充：

- 新增 `TTT_WRITE_GRADIENT_REVERSAL_BRANCH_GAMMAS` / `--ttt_write_gradient_reversal_branch_gammas`；
- 格式示例：`0:0.025,2:0.005`；
- 默认关闭，不影响旧 `TTT_WRITE_GRADIENT_REVERSAL_GAMMA + BRANCH_MASK` 行为；
- smoke `V7_TTGRBG_SMOKE2_c0_w0g0025_w2g0005_e96` 通过，debug 确认 `active_branches=[0,2]`，`active_branch_gammas={'0':0.025,'2':0.005}`。

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
PRIOR_BRANCH_MASK = 0,2
SWA = SWKS3-style fixed protocol
semantic = disabled
```

候选：

| Run | Branch gamma map | 目的 |
|---|---|---|
| `V7_TTGRBG_01_c5_w0g0025_w2g00025_SWKS3` | `0:0.025,2:0.0025` | 给 w2 极轻 negative evidence |
| `V7_TTGRBG_02_c5_w0g0025_w2g0005_SWKS3` | `0:0.025,2:0.005` | w2 小负证据 |
| `V7_TTGRBG_03_c5_w0g0020_w2g0005_SWKS3` | `0:0.020,2:0.005` | 降低 w0，给 w2 留空间 |
| `V7_TTGRBG_04_c5_w0g0025_w2g0010_SWKS3` | `0:0.025,2:0.010` | w2 稍强风险测试 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TTGRBG_01_c5_w0g0025_w2g00025_SWKS3` | `2026-05-08 08:27:50` | `08:51:06` | `23.3 min` |
| `V7_TTGRBG_02_c5_w0g0025_w2g0005_SWKS3` | `2026-05-08 08:27:50` | `08:50:43` | `22.9 min` |
| `V7_TTGRBG_03_c5_w0g0020_w2g0005_SWKS3` | `2026-05-08 08:27:50` | `08:50:59` | `23.2 min` |
| `V7_TTGRBG_04_c5_w0g0025_w2g0010_SWKS3` | `2026-05-08 08:27:50` | `08:50:48` | `23.0 min` |

Global metrics：

| Run | Branch gamma map | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | none | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | `0:0.025` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRBG_01` | `0:0.025,2:0.0025` | `36.5192` | `6.6108` | `92.4437` | `0.0081` | 回退 |
| `TTGRBG_02` | `0:0.025,2:0.005` | `36.3704` | `6.6642` | `92.4431` | `0.0082` | 本批 ATE best，但未过 TTGRL_04 |
| `TTGRBG_03` | `0:0.020,2:0.005` | `36.4157` | `6.6897` | `92.4429` | `0.0082` | 接近 B0，未过 |
| `TTGRBG_04` | `0:0.025,2:0.010` | `36.4202` | `6.6131` | `92.4434` | `0.0081` | 接近 B0，未过 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_ttgr_branch_gamma/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRBG_01` | `36.5192` | `5.477` | `78.061` | `77.783` | `57.092` | `48.125` | `3.759` | `31.231609` |
| `TTGRBG_02` | `36.3704` | `5.786` | `77.869` | `77.663` | `57.009` | `47.784` | `3.814` | `31.220836` |
| `TTGRBG_03` | `36.4157` | `6.023` | `77.969` | `77.611` | `56.966` | `47.959` | `3.848` | `31.222146` |
| `TTGRBG_04` | `36.4202` | `5.572` | `78.047` | `77.863` | `57.136` | `47.725` | `3.774` | `31.233564` |

结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04`。
2. `w2` 极小 gamma 并不是可靠补充。`TTGRBG_02` 是本批最好，但 ATE 仍比 `TTGRL_04` 差 `0.0747m`。
3. `w2` 确实能进一步压一部分 `[200,300)`：`TTGRBG_03` 的 `[200,300)=77.611` 好于 `TTGRL_04=77.644`，但它恶化 `[400,600)` 和 FinalErr，导致全局 ATE 回到 B0 附近。
4. 机制判断：`w2` 不是完全无效，而是负证据残留的后段代价太高。当前主线仍应保持 **chunk5 / w0 / all-layer / gamma0.025**。
5. 下一步主动探索 post-replay / commit-level 的 TTT 修正幅度，而不是继续给其它 branch 负证据。先用已有 `ttt_semantic_write_scale_chunks` 对 chunk5 semantic-vs-native delta 做局部缩放，检查 `TTGRL_04` 是不是 post-replay correction 幅度刚好或仍可外推。

---

## 10. H2：chunk5 TTGR post-replay delta scale

动机：

branch-specific gamma 没有超过 `TTGRL_04`。`w2` 能稍微压 `[200,300)`，但后段代价明显。因此本批不再扩大 negative branch，而是检查 `TTGRL_04` 的 post-replay correction 幅度是否偏强或偏弱。

使用已有 `TTT_SEMANTIC_WRITE_SCALE_CHUNKS`：

```text
W_commit = W_native + scale * (W_controlled - W_native)
```

这里只对 chunk5 生效。`scale=1.0` 等价于 `TTGRL_04`。

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
SWA = SWKS3-style fixed protocol
semantic = disabled
```

候选：

| Run | Chunk5 delta scale | 目的 |
|---|---:|---|
| `V7_TTGRSCALE_01_c5_w0g0025_scale075_SWKS3` | `0.75` | 检查 TTGR correction 是否过强 |
| `V7_TTGRSCALE_02_c5_w0g0025_scale050_SWKS3` | `0.50` | 更强拉回 native |
| `V7_TTGRSCALE_03_c5_w0g0025_scale125_SWKS3` | `1.25` | 检查 correction 是否偏保守 |
| `V7_TTGRSCALE_04_c5_w0g0025_scale150_SWKS3` | `1.50` | 强外推风险测试 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TTGRSCALE_01_c5_w0g0025_scale075_SWKS3` | `2026-05-08 08:54:04` | `09:16:13` | `22.2 min` |
| `V7_TTGRSCALE_02_c5_w0g0025_scale050_SWKS3` | `2026-05-08 08:54:04` | `09:16:21` | `22.3 min` |
| `V7_TTGRSCALE_03_c5_w0g0025_scale125_SWKS3` | `2026-05-08 08:54:04` | `09:16:06` | `22.0 min` |
| `V7_TTGRSCALE_04_c5_w0g0025_scale150_SWKS3` | `2026-05-08 08:54:04` | `09:16:46` | `22.7 min` |

Global metrics：

| Run | Scale | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | `1.00` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRSCALE_01` | `0.75` | `36.4746` | `6.6312` | `92.4444` | `0.0082` | 拉回 native 回退 |
| `TTGRSCALE_02` | `0.50` | `36.4969` | `6.6551` | `92.4436` | `0.0082` | 更强拉回更差 |
| `TTGRSCALE_03` | `1.25` | `36.3905` | `6.6160` | `92.4428` | `0.0082` | 本批最好，但未过 TTGRL_04 |
| `TTGRSCALE_04` | `1.50` | `36.4698` | `6.6031` | `92.4433` | `0.0081` | 外推过强回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_ttgr_scale/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRSCALE_01` | `36.4746` | `5.616` | `78.255` | `77.969` | `57.212` | `47.826` | `3.787` | `31.236956` |
| `TTGRSCALE_02` | `36.4969` | `5.967` | `78.193` | `77.747` | `57.065` | `48.171` | `3.809` | `31.215799` |
| `TTGRSCALE_03` | `36.3905` | `5.668` | `78.044` | `77.779` | `57.076` | `47.813` | `3.773` | `31.214484` |
| `TTGRSCALE_04` | `36.4698` | `5.514` | `78.150` | `77.907` | `57.170` | `47.919` | `3.754` | `31.217915` |

结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04`。
2. `scale < 1` 明显回退，说明 `TTGRL_04` 不是简单 over-update。
3. `scale=1.25` 有弱信号但仍差于 `TTGRL_04`；`scale=1.50` 又回退，说明 post-replay correction 幅度已经接近平台 sweet spot。
4. 继续做 post-replay scalar scale 没意义。当前证据指向：TTGR 的方向有用，但 scalar 幅度/branch/layer/chunk 都已进入平台。
5. 下一步切到 H3 positive evidence：保持 chunk5 `w0 gamma=0.025` 负证据不变，改正证据 write score，测试是否能让 positive/negative 角色更清楚，而不是继续在同一个 `stage_d_x_dg_inv_sqrt` 上微调。

---

## 11. H3：positive write score + localized TTGR

动机：

前面几批已经说明：

- chunk / branch / layer / gamma 的 scalar 微调基本收敛到 `TTGRL_04`；
- branch-specific `w2` 和 post-replay scale 都没能突破；
- 因此按计划切到 positive evidence：保持当前最有效的 negative replay 不变，只换正证据写入资格。

固定 negative replay：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
SWA = SWKS3-style fixed protocol
semantic = disabled
```

候选：

| Run | Positive write score | 目的 |
|---|---|---|
| `V7_TTGRPOS_01_stageD_c5w0g0025_SWKS3` | `stage_d` | 原始 safe write score |
| `V7_TTGRPOS_02_stageD_dgHighSqrt_c5w0g0025_SWKS3` | `stage_d * sqrt(1 - high(D_g))` | 只压高 D_g，减少对中低风险 token 的惩罚 |
| `V7_TTGRPOS_03_stageD_dgExpInterSqrt_c5w0g0025_SWKS3` | `stage_d * sqrt(1 - D_g * explicit_dyn)` | 只压 D_g 与 explicit dyn 交集 |
| `V7_TTGRPOS_04_stageD_unionDyn_c5w0g0025_SWKS3` | `stage_d * (1 - max(D_g, explicit_dyn))` | 更保守的 union dyn veto |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TTGRPOS_01_stageD_c5w0g0025_SWKS3` | `2026-05-08 09:19:25` | `09:42:54` | `23.5 min` |
| `V7_TTGRPOS_02_stageD_dgHighSqrt_c5w0g0025_SWKS3` | `2026-05-08 09:19:25` | `09:42:37` | `23.2 min` |
| `V7_TTGRPOS_03_stageD_dgExpInterSqrt_c5w0g0025_SWKS3` | `2026-05-08 09:19:25` | `09:43:32` | `24.1 min` |
| `V7_TTGRPOS_04_stageD_unionDyn_c5w0g0025_SWKS3` | `2026-05-08 09:19:25` | `09:42:04` | `22.7 min` |

Global metrics：

| Run | Positive write score | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | `stage_d_x_dg_inv_sqrt` | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | `stage_d_x_dg_inv_sqrt` + TTGR | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRPOS_01` | `stage_d` + TTGR | `36.5513` | `6.6935` | `92.4452` | `0.0082` | 明显回退 |
| `TTGRPOS_02` | `stage_d_x_dg_high_inv_sqrt` + TTGR | `36.4168` | `6.6103` | `92.4438` | `0.0081` | 最接近，但基本退回 B0 |
| `TTGRPOS_03` | `stage_d_x_dg_exp_inter_inv_sqrt` + TTGR | `36.5246` | `6.6971` | `92.4441` | `0.0082` | 回退 |
| `TTGRPOS_04` | `stage_d_x_union_dyn_inv` + TTGR | `36.4237` | `6.6432` | `92.4430` | `0.0082` | `[200,300)` 局部略好，但全局未过 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_posneg/trajectory_diagnostics_ttgr_positive_score/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRPOS_01` | `36.5513` | `6.157` | `78.443` | `77.891` | `57.154` | `48.162` | `3.845` | `31.237343` |
| `TTGRPOS_02` | `36.4168` | `5.235` | `77.998` | `77.720` | `57.039` | `47.882` | `3.755` | `31.225088` |
| `TTGRPOS_03` | `36.5246` | `5.867` | `78.304` | `77.891` | `57.152` | `48.033` | `3.840` | `31.227310` |
| `TTGRPOS_04` | `36.4237` | `5.513` | `78.069` | `77.623` | `56.963` | `47.961` | `3.804` | `31.213718` |

结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04`。
2. `stage_d` 原始正证据不够，叠加 TTGR 后比 B0 明显回退；说明 `D_g` 的 soft static eligibility 仍是必要正证据。
3. `stage_d_x_dg_high_inv_sqrt` 几乎退回 B0，说明只压高 D_g 不足以保留 `TTGRL_04` 的收益；中低风险区域的连续调制也重要。
4. `stage_d_x_union_dyn_inv` 对 `[200,300)` 略有局部收益，但把 `[400,600)` 和 Yaw 拉坏，仍不是主线。
5. 当前机制收敛：在现有 TTT replay / single fast-weight state 结构下，最好的仍是 `stage_d_x_dg_inv_sqrt + chunk5 w0 all low-prior TTGR gamma=0.025`。
6. 至此，plan 内的核心 TTT 写入小矩阵和主动扩展（branch gamma、post-replay scale、positive score）都没有找到 `<30m` 或明显新主线。后续若继续追最终成功，不能再扩大 scalar/gamma 小扫，应进入结构性方向：把 TTT fast weight 分成 long-term static 与 one-hop transient，或实现真正的 positive/neutral/negative 双 replay objective，而不是继续改单个 `A_tok`。

---

## 12. 当前 v7 状态

当前 v7 best：

```text
V7_TTGRL_04_c5_w0all_g0025_SWKS3
ATE / Rot = 36.295748 / 6.618212
RPE t / r = 92.442778 / 0.008160
FinalErr = 5.387
[200,300) = 77.644
YawRMSE = 3.769
```

相对 `B0_SWKS3 = 36.416102 / 6.612796`：

```text
ATE 改善 = 0.120354m
Rot 基本持平略回退
[200,300) 改善 = 0.187m
```

但相对最终目标：

```text
KITTI01 ATE < 30m
当前还差约 6.30m
```

已完成：

- v7 B0 reproduction / semantic no-op。
- H1 chunk freeze / chunk pair freeze causal audit。
- H2 localized TTGR：chunk5/w0/all/gamma0.025 是当前最佳。
- H2 gamma fine sweep。
- H2 hard negative tail。
- H2 semantic local TTGR。
- H2 chunk-specific gamma map。
- H2 layer split。
- H2 branch-specific gamma。
- H2 post-replay chunk5 delta scale。
- H3 positive score + localized TTGR。

当前停止判断：

1. `chunk5 / w0 / all / gamma0.025` 是现有 single-state TTT 写入空间里的 best point。
2. `w2`、chunk6、hard negative subset、semantic scalar、post-replay scalar scale、positive write score 替换都没有过这个点。
3. 继续在同一类 scalar prior / gamma / branch / layer 上细扫，预期只会得到 `36.3-36.6m` 平台内波动。
4. 下一轮若继续，应优先做结构性 TTT：
   - long-term static fast weight + transient one-hop fast weight；
   - two-replay objective：`G_commit = G_pos + lambda_neu G_neu - gamma G_neg`，而不是单 signed prior；
   - chunk5 causal target 专用的 commit-after-use 机制，让 negative evidence 修 `[200,300)` 但不长期污染 `[400,600)`。

---

## 13. 结构性 TTT：one-hop transient negative evidence

用户补充判断：

```text
应该换结构性 TTT 策略，比如双 fast-weight 生命周期、two-replay objective，
或者 chunk5 “用后提交/短期负证据”机制。
```

本批先做最小结构 hook：仍然在 chunk5 里使用当前 best 的 `w0` negative evidence，但把这部分 negative correction 视为 transient delta。当前 chunk 正常使用 controlled replay；提交给下一 chunk 时，从上一轮 fast weight 里扣掉一部分 transient delta：

```text
W_controlled = normal TTT replay with chunk5 TTGR
W_ref        = same replay without TTGR
delta_neg    = W_controlled - W_ref

current chunk uses:     W_controlled
next chunk initializes: W_commit - subtract_scale * delta_neg
```

固定协议：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE = one_hop_delta
TTT_WRITE_TRANSIENT_DELTA_BRANCH_MASK = 0
SWA = SWKS3-style fixed protocol
semantic = disabled
```

工程验证：

- `TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE=one_hop_delta` 已接入 `ttt_write_controller.py`、`hybrid_memory_controller.py`、`run_pipeline_abc_v2.py` 和 `tools/run_attention_cue_experiment.sh`。
- smoke `V7_TTGRST_SMOKE_c5_w0g0025_onehop_e224` 通过：
  - chunk5 debug 显示 `ttt_gradient_reversal_transient_applied=True`；
  - chunk6 debug 显示 `ttt_transient_delta_prev_subtract_applied=True`。

候选：

| Run | Gamma | Subtract scale | 目的 |
|---|---:|---:|---|
| `V7_TTGRST_01_c5_w0g0025_onehop_s100_SWKS3` | `0.025` | `1.00` | 完整扣除上一 chunk 的 negative transient delta |
| `V7_TTGRST_02_c5_w0g0025_onehop_s050_SWKS3` | `0.025` | `0.50` | 半扣除，避免删除 useful continuity |
| `V7_TTGRST_03_c5_w0g0025_onehop_s025_SWKS3` | `0.025` | `0.25` | 温和 one-hop forget |
| `V7_TTGRST_04_c5_w0g0020_onehop_s100_SWKS3` | `0.020` | `1.00` | 降低 gamma 后完整扣除 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TTGRST_01_c5_w0g0025_onehop_s100_SWKS3` | `2026-05-08 10:21:54` | `10:45:53` | `24.0 min` |
| `V7_TTGRST_02_c5_w0g0025_onehop_s050_SWKS3` | `2026-05-08 10:21:54` | `10:44:13` | `22.3 min` |
| `V7_TTGRST_03_c5_w0g0025_onehop_s025_SWKS3` | `2026-05-08 10:21:54` | `10:44:37` | `22.7 min` |
| `V7_TTGRST_04_c5_w0g0020_onehop_s100_SWKS3` | `2026-05-08 10:21:54` | `10:44:58` | `23.1 min` |

Global metrics：

| Run | Gamma | Subtract scale | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | `0.025` | `0.00` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRST_01` | `0.025` | `1.00` | `36.4587` | `6.6166` | `92.4436` | `0.0082` | 完整扣除回退 |
| `TTGRST_02` | `0.025` | `0.50` | `36.4671` | `6.6495` | `92.4435` | `0.0082` | 半扣除仍回退 |
| `TTGRST_03` | `0.025` | `0.25` | `36.3705` | `6.5914` | `92.4434` | `0.0082` | 本批最好；Rot 好，但 ATE 未过 TTGRL_04 |
| `TTGRST_04` | `0.020` | `1.00` | `36.4089` | `6.6238` | `92.4444` | `0.0082` | 接近 B0，未过 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_structural/trajectory_diagnostics_ttgr_transient_onehop/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRST_01` | `36.4587` | `5.447` | `78.191` | `77.883` | `57.141` | `47.913` | `3.765` | `31.224139` |
| `TTGRST_02` | `36.4671` | `5.767` | `78.191` | `77.890` | `57.165` | `47.977` | `3.809` | `31.224284` |
| `TTGRST_03` | `36.3705` | `5.775` | `78.077` | `77.793` | `57.077` | `47.779` | `3.750` | `31.223313` |
| `TTGRST_04` | `36.4089` | `5.759` | `78.148` | `77.877` | `57.144` | `47.788` | `3.786` | `31.230766` |

结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04`。
2. one-hop transient hook 生效，但直接扣除 TTGR delta 会吃掉 chunk5 里仍然有用的连续性；`subtract_scale=1.0/0.5` 都明显回退。
3. `subtract_scale=0.25` 是本批最好：ATE `36.3705`、Rot `6.5914`，说明“短期负证据用后遗忘”有结构信号，但当前 transient delta 太粗。
4. 该策略没有压低 `[200,300)` 主病灶，反而比 `TTGRL_04` 差；说明 harmful state 不是完整 `W_controlled-W_ref`，里面混有 useful correction。
5. 下一步不应继续扫 subtract scale。更合理的是挖掘 **TTT 自己的 cue**：不要只用外部 `D_g/semantic` 来定义 risk，而是让 TTT replay 自己告诉我们哪些 token / branch / layer 对 fast-weight 更新是有害的。

---

## 14. 下一步预案：TTT self cue / update-internal risk

用户提出：

```text
是不是应该挖掘 TTT 自己的 cue，或许效果会更好。
```

当前判断：应该。原因是 v7 已经证明外部 cue 的 scalar 用法基本收敛：

- `D_g` 适合作 read 和 soft write eligibility，但继续换正证据 score 没有超过 `TTGRL_04`；
- semantic Mask2Former cache 覆盖已过 gate，但 semantic scalar 只能做弱辅助；
- chunk5 TTGR 的方向有用，但 risk 仍由外部 write prior 间接定义；
- one-hop transient 失败说明 `W_controlled-W_ref` 里混有正负信息，需要 TTT 内部 cue 拆分。

TTT self cue 的候选不再问 “patch 看起来像不像动态”，而是问：

```text
这个 token / layer / branch 在 TTT replay 里是否真的造成不稳定更新？
```

优先实现顺序：

1. `ttt_residual_risk`：用 replay cache 中 `||y-v||` 的 per-token residual 作为 negative evidence risk。它比外部 `D_g` 更接近 TTT loss 本身。
2. `ttt_residual_x_dg`：只在 TTT residual 高且 `D_g` 也高的 token 上反转，避免 residual 把正常几何难点误杀。
3. `ttt_delta_conflict`：比较 native replay delta 与 controlled replay delta 的方向，负证据只作用在与 native continuity 冲突的 token / branch / layer。
4. `two-replay objective`：显式构造 `G_pos + lambda_neu G_neu - gamma G_neg`，而不是把所有东西揉进一个 signed prior。

第一批建议从最小可控实现开始：

| Run | Risk source | Chunk | Branch | Gamma | 目的 |
|---|---|---|---|---:|---|
| `V7_TTSELF_01` | `ttt_residual` | `5` | `w0` | `0.025` | 用 TTT 自己 residual 替代外部 prior risk |
| `V7_TTSELF_02` | `ttt_residual` | `5` | `w0` | `0.015` | 更温和，避免 residual risk 过强 |
| `V7_TTSELF_03` | `ttt_residual_x_dg` | `5` | `w0` | `0.025` | TTT residual 与 Dg consensus |
| `V7_TTSELF_04` | `ttt_residual_x_dg` | `5` | `w0` | `0.015` | 温和 consensus |

Promotion gate：

- 若超过 `TTGRL_04 = 36.2957 / 6.6182`，保留为新的 TTT 写入主线；
- 若 `[200,300)` 下降明显但 ATE 回退，继续做 long-term / transient 拆分；
- 若全部回退，说明 `||y-v||` residual 仍太粗，下一步转向 branch-level update conflict cue。

---

## 15. TTT self cue 第一批：replay residual risk

本节把 14 节的第一批 TTT self cue 做成实际 hook。核心变化是：TTGR 的 risk 不再只能来自外部 write prior / `D_g`，也可以来自 TTT replay 自己的 residual。

实现方式：

```text
TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE = ttt_residual / ttt_residual_x_dg
```

其中：

```text
ttt_residual      = normalize(||apply_output_raw - v|| / ||v||)
ttt_residual_x_dg = normalize(ttt_residual * D_g)
```

含义：

- `ttt_residual`：哪个 token 在当前 layer 的 TTT replay 里自身 residual 更大，就更像 negative evidence 候选；
- `ttt_residual_x_dg`：只在 TTT residual 高、外部 `D_g` 也高的地方提高 negative risk；
- 仍固定只动 chunk5 / branch `w0` / all-layer，避免把已经定位好的因果结构放宽。

工程验证：

- `TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE` 已接入 `ttt_write_controller.py`、`hybrid_memory_controller.py`、`run_pipeline_abc_v2.py` 和 `tools/run_attention_cue_experiment.sh`。
- smoke `V7_TTSELF_SMOKE_c5_residual_g0025_e224` 通过，chunk5 debug 确认：
  - `ttt_gradient_reversal_risk_source = ttt_residual`
  - `ttt_gradient_reversal_risk_source_effective = ttt_residual`
  - `ttt_gradient_reversal_applied = true`
  - `ttt_gradient_reversal_active_branches = [0]`

固定协议：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
SWA = SWKS3-style fixed protocol
semantic = disabled
```

候选：

| Run | Risk source | Gamma | 目的 |
|---|---|---:|---|
| `V7_TTSELF_01_c5_w0_residual_g0025_SWKS3` | `ttt_residual` | `0.025` | 用 TTT residual 替代外部 prior risk |
| `V7_TTSELF_02_c5_w0_residual_g0015_SWKS3` | `ttt_residual` | `0.015` | 更温和 residual risk |
| `V7_TTSELF_03_c5_w0_residualDg_g0025_SWKS3` | `ttt_residual_x_dg` | `0.025` | residual 与 Dg consensus |
| `V7_TTSELF_04_c5_w0_residualDg_g0015_SWKS3` | `ttt_residual_x_dg` | `0.015` | 温和 consensus |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TTSELF_01_c5_w0_residual_g0025_SWKS3` | `2026-05-08 11:00:01` | `11:23:40` | `23.7 min` |
| `V7_TTSELF_02_c5_w0_residual_g0015_SWKS3` | `2026-05-08 11:00:01` | `11:25:37` | `25.6 min` |
| `V7_TTSELF_03_c5_w0_residualDg_g0025_SWKS3` | `2026-05-08 11:00:01` | `11:24:06` | `24.1 min` |
| `V7_TTSELF_04_c5_w0_residualDg_g0015_SWKS3` | `2026-05-08 11:00:01` | `11:24:33` | `24.5 min` |

Global metrics：

| Run | Risk source | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | external prior | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | external low-prior | `0.025` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTSELF_01` | `ttt_residual` | `0.025` | `36.4292` | `6.6280` | `92.4438` | `0.0081` | gamma 偏强，回退 |
| `TTSELF_02` | `ttt_residual` | `0.015` | `36.3667` | `6.6580` | `92.4422` | `0.0082` | 本批最好；超过 B0，但未过 TTGRL_04 |
| `TTSELF_03` | `ttt_residual_x_dg` | `0.025` | `36.4175` | `6.6338` | `92.4419` | `0.0082` | 接近 B0，未过 |
| `TTSELF_04` | `ttt_residual_x_dg` | `0.015` | `36.4549` | `6.6523` | `92.4440` | `0.0082` | 回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_selfcue/trajectory_diagnostics_ttt_selfcue/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTSELF_01` | `36.4292` | `5.757` | `78.083` | `77.721` | `57.044` | `47.989` | `3.788` | `31.220634` |
| `TTSELF_02` | `36.3667` | `5.709` | `78.013` | `77.656` | `56.989` | `47.800` | `3.820` | `31.208212` |
| `TTSELF_03` | `36.4175` | `5.977` | `78.261` | `77.939` | `57.177` | `47.845` | `3.784` | `31.205619` |
| `TTSELF_04` | `36.4549` | `5.899` | `78.357` | `78.047` | `57.255` | `47.832` | `3.801` | `31.225280` |

Focus chunk diagnostics：

| Run | chunk6 `[174,206)` | chunk7 `[203,235)` | chunk8 `[232,264)` | chunk9 `[261,293)` | chunk10 `[290,322)` |
|---|---:|---:|---:|---:|---:|
| `B0` | `53.799` | `74.942` | `88.814` | `76.800` | `37.576` |
| `TTGRL_04` | `53.381` | `74.567` | `88.555` | `76.806` | `37.878` |
| `TTSELF_01` | `53.640` | `74.690` | `88.740` | `76.745` | `37.796` |
| `TTSELF_02` | `53.550` | `74.607` | `88.678` | `76.697` | `37.750` |
| `TTSELF_03` | `53.492` | `74.920` | `88.877` | `77.106` | `37.890` |
| `TTSELF_04` | `53.544` | `74.971` | `89.038` | `77.219` | `37.975` |

15 节结论：

1. TTT self cue 方向是有信号的：`TTSELF_02 = 36.3667 / 6.6580` 明确超过 B0，并且 `[200,300)=77.656` 已经接近 `TTGRL_04=77.644`。
2. 但它没有超过当前 best `TTGRL_04=36.2957`，主要输在 `[400,600)` 和全局 ATE 平衡；说明 replay residual 本身能定位 chunk5 的部分 harmful direction，但还不如外部 low-prior 的宽区域软反转稳定。
3. `ttt_residual_x_dg` 没有帮助，说明 TTT residual 与 `D_g` 的 harmful 空间不是简单交集；强行 consensus 反而压掉了有用 correction。
4. debug 中 residual risk 的 true negative mass 很小，实际更像 broad rescaling，而不是强负证据。下一步不应直接加大 gamma，而应测试 residual-tail / update-conflict：
   - residual-tail：只把 TTT residual 最高的一小段 token 反转；
   - update-conflict：比较 native replay 与 controlled replay 的 update direction，负证据只给和 continuity 冲突的方向。
5. 当前 v7 best 仍保持：

```text
V7_TTGRL_04_c5_w0all_g0025_SWKS3
ATE / Rot = 36.2957 / 6.6182
```

---

## 21. 结构性 TTT：two-replay positive / negative objective

根据 `docs/TTT_Write_Controller_Design_v1.md`，本批不再把所有 token 放进同一个 signed prior，而是实现双 replay：

```text
positive replay:
    使用正常正证据 prior，得到 W_pos

negative replay:
    使用 risk prior 单独 replay，得到 W_neg

commit:
    W_commit = W_pos - gamma * (W_neg - W_old)
```

目的：把 positive / negative evidence 的更新方向分开，避免单个连续 signed prior 同时承担“该写”和“该反写”两种语义。

工程验证：

- 新增 `TTT_WRITE_GRADIENT_REVERSAL_MODE=two_replay / separate_replay / pos_neg_replay`。
- smoke `V7_TT2R_SMOKE_c5_w0_g0025_e224` 通过。
- debug 确认 `ttt_two_replay_applied=True`、active branch `[0]`，并记录 `ttt_two_replay_w0_neg_delta_norm_mean`。

固定协议：

```text
SWA = SWKS3-style fixed protocol
TTT_WRITE_GRADIENT_REVERSAL_MODE = two_replay
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
PRIOR_LAYER_MODE = all
```

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TT2R_01_c5_w0_g0010_SWKS3` | `2026-05-08 15:03:04` | `15:25:25` | `22.4 min` |
| `V7_TT2R_02_c5_w0_g0025_SWKS3` | `2026-05-08 15:03:05` | `15:25:36` | `22.5 min` |
| `V7_TT2R_03_c5_w0_g0050_SWKS3` | `2026-05-08 15:03:05` | `15:25:10` | `22.1 min` |
| `V7_TT2R_04_c5_w0_conflictEnergy_g0015_SWKS3` | `2026-05-08 15:03:05` | `15:27:57` | `24.9 min` |

Global metrics：

| Run | Risk source | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | baseline | - | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | continuous low-prior TTGR | `0.025` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TT2R_01` | prior risk | `0.010` | `36.5117` | `6.6463` | `92.4459` | `0.0082` | 回退 |
| `TT2R_02` | prior risk | `0.025` | `36.5046` | `6.6219` | `92.4450` | `0.0082` | 回退 |
| `TT2R_03` | prior risk | `0.050` | `36.5073` | `6.6099` | `92.4441` | `0.0082` | 回退 |
| `TT2R_04` | `ttt_w0_conflict_energy` | `0.015` | `36.4446` | `6.6166` | `92.4450` | `0.0082` | 本批最好，但未过 B0 / TTGRL_04 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_two_replay/trajectory_diagnostics_two_replay/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `3.769` | `31.219364` |
| `TT2R_01` | `36.5117` | `6.020` | `78.417` | `78.010` | `57.239` | `3.801` | `31.245635` |
| `TT2R_02` | `36.5046` | `5.614` | `78.349` | `77.989` | `57.216` | `3.775` | `31.241065` |
| `TT2R_03` | `36.5074` | `5.698` | `78.309` | `77.963` | `57.214` | `3.761` | `31.223585` |
| `TT2R_04` | `36.4446` | `5.783` | `78.269` | `77.900` | `57.152` | `3.771` | `31.234399` |

21 节结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04=36.2957`。
2. 当前 two-replay 实现比 continuous low-prior TTGR 更差：prior-risk 三条全部回到 `36.50m` 平台，说明 “先算负 replay 再扣 delta” 这版会破坏 chunk5 的 useful continuity。
3. `ttt_w0_conflict_energy` 作为 negative replay risk source 比纯 prior risk 更好，但 `36.4446m` 仍不如 B0，更不如 TTGRL_04。
4. 机制判断：negative evidence 不能被简单当成独立 replay delta 扣掉；当前 LoGeR TTT 更新中，负证据需要以更软的方式改变 pre-zeropower 方向，而不是形成单独 fast-weight delta 后硬扣。
5. two-replay objective 不作为下一阶段主线。按设计文档下一步转向 **special/register token write policy**：patch token 被 suppress 时，special token 不能继续默认全 1 写入。

---

## 22. TTT special token write policy 启动

`docs/TTT_Write_Controller_Design_v1.md` 指出一个此前没有系统测试的绕写路径：

```text
patch token 被 D_g / prior suppress
但 register / role token 仍保持 prior = 1.0
动态上下文可能通过 special token 写进 TTT fast weights
```

工程改动：

- `TTTWriteController` 新增默认关闭的 special-token policy：
  - `TTT_WRITE_SPECIAL_TOKEN_POLICY=patch_q10 / patch_q25 / patch_mean / frame_q25 ...`
  - `TTT_WRITE_SPECIAL_TOKEN_FLOOR`
  - `TTT_WRITE_SPECIAL_TOKEN_CEILING`
- 策略只作用在 full token layout，patch-only replay diagnostic 会自动跳过。
- CLI / `tools/run_attention_cue_experiment.sh` 已接线。

验证：

```text
python -m py_compile loger/pipeline/ttt_write_controller.py loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py
bash -n tools/run_attention_cue_experiment.sh
```

Short smoke：

| Run | Policy | Floor | END_FRAME | 结论 |
|---|---|---:|---:|---|
| `V7_TTSPTOK_SMOKE_c5_patchq25_floor085_e96` | `patch_q25` | `0.85` | `96` | 通过；special prior 从 `1.0` 拉到 patch q25 `0.9375` |

首批 full matrix 已启动：

| Run | Base | Special policy | Floor | 目的 |
|---|---|---|---:|---|
| `V7_TTSPTOK_01_c5w0g0025_patchq25_floor085_SWKS3` | `TTGRL_04` | `patch_q25` | `0.85` | 温和抑制 special token |
| `V7_TTSPTOK_02_c5w0g0025_patchq10_floor085_SWKS3` | `TTGRL_04` | `patch_q10` | `0.85` | 更强按低 patch prior 约束 special |
| `V7_TTSPTOK_03_c5w0g0025_frameq25_floor085_SWKS3` | `TTGRL_04` | `frame_q25` | `0.85` | 每帧自适应 special prior |
| `V7_TTSPTOK_04_c5w0g0025_patchq25_floor090_SWKS3` | `TTGRL_04` | `patch_q25` | `0.90` | 更保守版本，避免误伤全局 continuity |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TTSPTOK_01_c5w0g0025_patchq25_floor085_SWKS3` | `2026-05-08 15:27:30` | `15:50:05` | `22.6 min` |
| `V7_TTSPTOK_02_c5w0g0025_patchq10_floor085_SWKS3` | `2026-05-08 15:27:30` | `15:50:34` | `23.1 min` |
| `V7_TTSPTOK_03_c5w0g0025_frameq25_floor085_SWKS3` | `2026-05-08 15:27:30` | `15:50:45` | `23.3 min` |
| `V7_TTSPTOK_04_c5w0g0025_patchq25_floor090_SWKS3` | `2026-05-08 15:27:30` | `15:51:04` | `23.6 min` |

Global metrics：

| Run | Special policy | Floor | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | none | - | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | none | - | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTSPTOK_01` | `patch_q25` | `0.85` | `36.4711` | `6.6117` | `92.4431` | `0.0082` | 回退 |
| `TTSPTOK_02` | `patch_q10` | `0.85` | `36.5421` | `6.6724` | `92.4447` | `0.0082` | 更强 special suppression 更差 |
| `TTSPTOK_03` | `frame_q25` | `0.85` | `36.4884` | `6.5955` | `92.4436` | `0.0081` | Rot 略好但 ATE 回退 |
| `TTSPTOK_04` | `patch_q25` | `0.90` | `36.4711` | `6.6117` | `92.4431` | `0.0082` | 与 floor `0.85` 等价 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_special_token/trajectory_diagnostics_special_token/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `3.769` | `31.219364` |
| `TTSPTOK_01` | `36.4711` | `5.413` | `78.300` | `77.989` | `57.220` | `3.766` | `31.224258` |
| `TTSPTOK_02` | `36.5421` | `6.054` | `78.881` | `78.383` | `57.469` | `3.813` | `31.234964` |
| `TTSPTOK_03` | `36.4884` | `5.547` | `78.285` | `77.906` | `57.170` | `3.753` | `31.227899` |
| `TTSPTOK_04` | `36.4711` | `5.413` | `78.300` | `77.989` | `57.220` | `3.766` | `31.224258` |

22 节结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04` 或 B0。
2. special-token policy 确实生效：smoke 与 full log 中都能看到 register/role token 从 `1.0` 被拉到 patch q25 / per-frame q25，例如 `0.9375` 左右。
3. 但这不是当前主瓶颈。只要压 special token，`[200,300)` 和 overall ATE 都回退；`patch_q10` 更强 suppression 回退最明显。
4. `frame_q25` 能稍微改善 Rot / Yaw，但 ATE 仍回退，说明 special token 更像 continuity / context carrier，而不是 harmful dynamic bypass。
5. 至此，按 `TTT_Write_Controller_Design_v1` 的 token-wise / two-replay / block-gain / special-token 主要候选都没有超过 `TTGRL_04`。下一步需要真正结构性 fast-weight lifecycle，而不是继续调同一个 commit state 的 token prior。

当前 v7 best 仍保持：

```text
V7_TTGRL_04_c5_w0all_g0025_SWKS3
ATE / Rot = 36.2957 / 6.6182
```

---

## 30. update-conflict per-head routing 与全序列弱剂量 tri-replay

用户要求继续探索直到 `KITTI01 ATE < 30m`。本节接 29 节的 `update_conflict_energy` 诊断，先尝试把最高 conflict 的 `layer12/head0` 作为 per-head TTT 自有 cue；随后尝试全序列极弱 `update_conflict_energy` tri-replay，验证 TTT cue 是否只应局限在 chunk5。

### 30.1 工程补充：per-head routing

新增：

```text
TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES
--ttt_write_gradient_reversal_head_routes
```

格式示例：

```text
12:0
12:0+1;13:0
```

语义：

- 只在指定 layer 的指定 fast-weight head/slice 上使用 controlled tri-replay；
- 未选中的 head 保留 base replay；
- 若同时设置全层 base gamma，则未选中 head 保留 base tri-replay，选中 head 使用 layer/head override gamma。

Smoke：

```text
V7_TRIREPLAY_HEAD_SMOKE_c5_l12h0_g004_e192
V7_TRIREPLAY_HEAD_SMOKE_base002_l12h0_g004_e192
```

验证：

| Check | Value |
|---|---:|
| `ttt_head_routed_w0_applied` | `True` |
| `ttt_head_routed_w0_heads` | `[0]` |
| `ttt_head_routed_w0_head_count` | `2` |
| base+head smoke `ttt_head_routed_w0_base_gamma` | `0.02` |

### 30.2 per-head routing full 结果

固定：

```text
seq = KITTI01 full
SWA = SWKS3-style fixed protocol
TTT_WRITE_GRADIENT_REVERSAL_MODE = tri_replay
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
risk = update_conflict_energy
positive_frac = 0.35
negative_frac = 0.12
neutral_lambda = 1.0
branch = w0
```

| Run | 设置 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `TRIREPLAY_12` | chunk5 all-layer base `gamma=0.030` | `35.8712` | `6.6142` | `92.4389` | `0.0082` | reference |
| `HEAD_01` | layer12/head0 only `gamma=0.040` | `36.4672` | `6.6439` | `92.4437` | `0.0082` | head-only 不够 |
| `HEAD_02` | layer12/head0 only `gamma=0.060` | `36.4325` | `6.6708` | `92.4455` | `0.0082` | head-only 仍回退 |
| `HEAD_03` | base `0.020` + layer12/head0 `0.040` | `35.9413` | `6.6662` | `92.4403` | `0.0082` | 接近但未过 reference |
| `HEAD_04` | base `0.030` + layer12/head0 `0.050` | `35.9809` | `6.6364` | `92.4398` | `0.0082` | 未过 reference |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_head_routed/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `TRIREPLAY_12` | `35.8712` | `5.573` | `76.863` | `56.433` | `3.793` | `31.177167` |
| `HEAD_01` | `36.4672` | `5.580` | `77.776` | `57.079` | `3.790` | `31.227705` |
| `HEAD_02` | `36.4325` | `5.653` | `77.876` | `57.137` | `3.809` | `31.242247` |
| `HEAD_03` | `35.9413` | `6.178` | `76.892` | `56.468` | `3.849` | `31.181895` |
| `HEAD_04` | `35.9809` | `6.042` | `77.017` | `56.552` | `3.825` | `31.185864` |

结论：

1. `layer12/head0` 是有诊断价值的 TTT 自有 cue 位置，但不能单独替代全层 tri-replay。
2. best head route `HEAD_03=35.9413` 仍差于 `TRIREPLAY_12=35.8712`。
3. TRIREPLAY 的收益不是单 head outlier，而是多层/多 head 的弱 conflict field；per-head routing 可用于诊断，不是当前主线。

### 30.3 全序列弱剂量 update-conflict tri-replay

动机：29 节和 30.2 都围绕 chunk5。为验证 `update_conflict_energy` 是否可以作为全序列 TTT 自有 cue，本批暂时取消 chunk gate，用极弱 gamma 跑全 KITTI01。

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = tri_replay
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = empty  # all chunks
risk = update_conflict_energy
positive_frac = 0.35
negative_frac = 0.12
neutral_lambda = 1.0
branch = w0
```

| Run | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `TRIREPLAY_12` | chunk5 `0.030` | `35.8712` | `6.6142` | `92.4389` | `0.0082` | old best |
| `GLOBAL_01` | all chunks `0.005` | `35.1293` | `7.9762` | `92.3989` | `0.0096` | **当前 ATE 新 best，但 Rot/Final/Yaw 代价大** |
| `GLOBAL_02` | all chunks `0.010` | `35.1810` | `7.9628` | `92.4002` | `0.0096` | ATE 强但不如 0.005 |
| `GLOBAL_03` | all chunks `0.015` | `35.2476` | `7.9152` | `92.4007` | `0.0095` | ATE 回退 |
| `GLOBAL_04` | all chunks `0.020` | `35.2237` | `7.9680` | `92.3990` | `0.0096` | ATE 回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_global_conflict/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.644` | `56.968` | `3.769` | `31.219364` |
| `TRIREPLAY_12` | `35.8712` | `5.573` | `76.863` | `56.433` | `3.793` | `31.177167` |
| `GLOBAL_01` | `35.1293` | `19.287` | `74.915` | `55.125` | `5.130` | `30.719503` |
| `GLOBAL_02` | `35.1810` | `19.143` | `75.016` | `55.193` | `5.106` | `30.734930` |
| `GLOBAL_03` | `35.2476` | `18.646` | `75.037` | `55.212` | `5.075` | `30.739188` |
| `GLOBAL_04` | `35.2237` | `18.882` | `75.018` | `55.196` | `5.109` | `30.718616` |

结论：

1. 这是 v7 目前最大的 ATE 改进信号：`GLOBAL_01=35.1293`，比 `TRIREPLAY_12` 好 `0.7419m`，比 `B0` 好 `1.2868m`。
2. `[200,300)` 也从 `76.863` 进一步降到 `74.915`，说明它不是单纯指标噪声，而是确实压低了主病灶。
3. 但代价也很明确：FinalErr 从 `5.573` 恶化到 `19.287`，Yaw 从 `3.793` 恶化到 `5.130`，Rot 从 `6.6142` 恶化到 `7.9762`。它是强 ATE / 弱姿态的 trade-off，不可直接晋级为主线。
4. 更重要的是，全序列 tri-replay 每个 full run 接近 `1h`，用户明确认为不可接受。因此不再用全序列 tri-replay 做细扫。

取消的长 run：

| Run | 状态 | 原因 |
|---|---|---|
| `GLOBAL_05` | killed | refine run 过长 |
| `GLOBAL_06` | killed | refine run 过长 |
| `GLOBAL_07` | killed | refine run 过长 |
| `GLOBAL_08` | killed | refine run 过长 |

后续约束：

```text
不再做 1h/run 的全序列 tri-replay 矩阵。
GLOBAL_01 只作为强信号 reference。
下一步改为少量 chunk-gated conflict replay 或 short-run lesion gate；
只有短跑明显超过 GLOBAL_01 / TRIREPLAY_12 的趋势，才跑 full KITTI01。
```

---

## 25. 最新结构性 TTT 复盘：two-replay / short negative TTL

说明：本节是 `2026-05-09` 最新追加结果，放在文件末尾便于查看。freeze 仍只作为 causal diagnostic，不作为策略。

### 25.1 先说好消息

1. `two_replay` 工程路径已经跑通，debug 确认正向 replay 和负向 replay 是分开的，不是旧的 signed prior 小改。
2. `short negative TTL` 也跑完整 KITTI01，没有出现 strict dual-lifetime smoke 那种卡住问题。
3. 机制结论很明确：chunk5 的 negative evidence 不能 one-hop 后立刻移除，它至少需要活到当前 `RESET_EVERY=5` 周期结束；过早撤掉会把 `TTGRL_04` 的收益吃掉。

### 25.2 strict dual-lifetime 状态

尝试了 strict dual fast-weight lifetime smoke：

| Run | 结果 |
|---|---|
| `V7_TTDL_TRUE_SMOKE_c5_w0_g0025_ttl4_long050_e224` | 异常变慢 / 无稳定进展，手动停止 |
| `V7_TTDL_TRUE_SMOKE2_c5_w0_g0025_ttl4_long050_e177` | 仍然异常慢 / 不稳定，手动停止 |

结论：strict dual-lifetime 方向仍值得做，但当前实现路径不稳定，本轮不计 full 指标。

### 25.3 two-replay objective

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = two_replay
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
SWA = SWKS3-style fixed protocol
```

| Run | Negative risk source | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | baseline | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | low-prior signed replay | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `V7_TT2R_01` | `prior` | `36.5046` | `6.6219` | `92.4450` | `0.0082` | 回退 |
| `V7_TT2R_02` | `update_conflict` | `36.4593` | `6.6373` | `92.4449` | `0.0082` | 回退 |
| `V7_TT2R_03` | `update_conflict_energy` | `36.4560` | `6.5800` | `92.4434` | `0.0081` | two-replay best，但未过 |
| `V7_TT2R_04` | `ttt_residual_x_dg` | `36.5463` | `6.6213` | `92.4437` | `0.0082` | 回退 |

诊断目录：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_two_replay_structural/trajectory_diagnostics_two_replay/
```

结论：two-replay 是可用机制，但第一版 `all positive + risk negative` 不如 `TTGRL_04` 的 continuous low-prior direction shift。harmful update 更像 chunk5 `w0` 大面积弱偏置，而不是一个可以单独扣掉的强 `G_neg`。

### 25.4 short negative TTL

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE = ttl_delta
TTT_WRITE_TRANSIENT_DELTA_SUBTRACT_SCALE = 1.0
TTT_WRITE_TRANSIENT_DELTA_BRANCH_MASK = 0
```

| Run | TTL | ATE RMSE | Rot RMSE | `[200,300)` | `[400,600)` | FinalErr | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | `36.4161` | `6.6128` | `77.831` | `47.825` | `5.798` | baseline |
| `TTGRL_04` | persistent within reset | `36.2957` | `6.6182` | `77.644` | `47.582` | `5.387` | current v7 best |
| `V7_TTGRTTL_01` | `1` | `36.4587` | `6.6166` | `77.883` | `47.913` | `5.447` | 太早扣除，回退 |
| `V7_TTGRTTL_02` | `2` | `36.4256` | `6.6056` | `77.881` | `47.832` | `5.679` | 接近 B0，但未过 |
| `V7_TTGRTTL_03` | `3` | `36.3475` | `6.6616` | `77.817` | `47.584` | `5.914` | TTL best，但未过 TTGRL_04 |
| `V7_TTGRTTL_04` | `4` | `36.2957` | `6.6182` | `77.644` | `47.582` | `5.387` | 与 TTGRL_04 完全重合 |

诊断目录：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_short_negative_ttl/trajectory_diagnostics_short_negative_ttl/
```

25.4 结论：

1. short negative TTL 没有超过 `TTGRL_04`。
2. TTL 越短，越接近把 chunk5 negative evidence 从长期状态中过早拿掉，ATE / `[200,300)` 都回退。
3. TTL=4 与 `TTGRL_04` 完全重合，说明当前 best 需要 chunk5 negative correction 在整个 reset window 内持续存在。
4. 这条结果反过来给了结构方向：不是 “用一下就删掉”，而是要 **在一个 reset 周期内维持 short negative state，同时不要让它跨 reset 污染长期 state**。当前 `RESET_EVERY=5` 已经天然做到了一部分，所以 simple TTL 没带来额外收益。

### 25.5 当前判断

当前 v7 best 不变：

```text
V7_TTGRL_04_c5_w0all_g0025_SWKS3
ATE / Rot = 36.2957 / 6.6182
```

但结构实验带来了明确边界：

1. freeze 不是解药，只是诊断；
2. one-hop short negative 不够，chunk5 negative direction 必须覆盖到 reset window 末尾；
3. first-version two-replay 不够，需要更明确的三组目标：

```text
G_commit = G_pos + lambda_neu * G_neu - gamma * G_neg
```

其中 `G_pos` 只取 high-continuity / static token，`G_neg` 只取 update-conflict high-risk token，`G_neu` 保留普通 continuity。下一步如果继续结构性 TTT，应优先实现这个三组 replay objective，而不是继续扫 TTL / scalar gamma。

---

## 36. 最新阅读索引（2026-05-10 04:13）

完整阶段总结见本文件第 35 节：

```text
35. v7 TTT 写入探索阶段总结（截至 2026-05-10 04:13）
```

当前最强完整结果：

```text
V7_TRIREPLAY_WINGAM_03_bodyg005_exitg0030_neu085_SWKS3
ATE / Rot = 34.1903 / 6.5666
RPE t / r = 92.4202 / 0.0083
```

一句话结论：

```text
v7 的有效突破来自 TTT 自有 update_conflict_energy cue + 三组 replay objective。
当前最好策略不是 freeze，也不是单 chunk 局部反转，而是在 chunks 5-9 保持较强负证据、
chunks 10-12 使用更弱负证据，并把 neutral continuity 缩到 0.85。
```

未完成 / 不计入：

```text
WINGAMF_01-04 在用户要求总结后中止，只有 partial chunk 输出，没有完整 KITTI01 benchmark。
```

---

## 35. v7 TTT 写入探索阶段总结（截至 2026-05-10 04:13）

本节是给当前 v7 结果的人工可读总结，避免只靠 run id / 缩写定位。未完成的 `WINGAMF_01-04` follow-up 已在用户要求总结后中止，不计入结论。

### 35.1 固定主线到底做了什么

除特别说明外，v7 的主线都固定在同一个读路径和 SWA 路径上：

```text
读控制：
    cue = LoGeR global decoder 第 2-3 层 query 特征，past-only support
    D_g 高 = 当前 patch 不像过去稳定结构，更像动态/不稳定区域
    frame attention bias = pair bias
    read layers = all layers
    beta = 4.75

基础 TTT 正向写入：
    base write score = stage_d * sqrt(1 - D_g)
    stage_d = HMC 原始几何/稳定性写入资格分数，越高越适合写进 TTT fast weight
    sqrt(1 - D_g) = 对高动态 / 高不稳定 patch 做温和降权
    WRITE_ALPHA = 0.125
    只让 branch w0 使用这个动态写入 prior，w1/w2 默认保持 native 更新

SWA：
    开启 SWA write control
    只保留 head/tail overlap 相关历史（both_overlap）
    只作用最后一个 SWA layer
    对上一段 overlap source 的 K/V 做 source replacement
    replacement mode = source, target = K/V, alpha = 0.50

TTT 生命周期：
    RESET_EVERY = 5
```

这里的重点是：v7 不是单独改 attention read，也不是单独改 SWA；真正探索的是 **TTT 写入时哪些 token / branch / chunk 的 replay update 应该正向保留、弱保留、或者反向扣掉**。

### 35.2 已做过的主要实验族

| 实验族 | 具体做法 | 最好结果 | 主要 insight |
|---|---|---:|---|
| B0 / no-op 复现 | 关闭 semantic write，只用 v6/SWKS3 固定协议；另用 Mask2Former cache 生成 semantic prior 但 HMC 忽略，确认不改变轨迹 | `36.4161 / 6.6128` | v7 代码和 cache 回接是安全的，后续变化来自 TTT 策略本身 |
| chunk freeze 诊断 | 不提交指定 chunk 的 TTT fast-weight 更新，只看因果影响 | `freeze56` 把 `[200,300)` 压到 `26.102`，但 overall ATE 崩到 `60.3998` | freeze 只能证明 chunk5/6 对病灶有因果性，不能当解药；它会把后段连续性和尺度一起删掉 |
| localized gradient reversal | 只在 chunk5，branch w0，所有 TTT layers，对低写入资格 token 给小负 multiplier，而不是只少写 | `TTGRL_04 = 36.2957 / 6.6182` | chunk5 有 harmful update direction；最有效的是很轻的 `gamma=0.025`，太强或扩到 chunk6 都回退 |
| negative-tail | 只反转最高风险的一小撮 token，其它 token 保持正向写入 | 未过 B0 / TTGRL_04 | harmful update 不是少数 outlier token，而是一片弱风险区域的累积偏置 |
| semantic write | 用 Mask2Former Cityscapes panoptic cache 产生 road/sky/vegetation/fence/building 等 semantic value，直接控制 TTT write | semantic-only 最好仍差于 B0；semantic+TTGR `P3G_01=36.4017` | semantic cache 覆盖已解决，但单个 patch-level semantic value 太粗；语义目前只能做弱辅助，不能替代 TTT 自身 cue |
| TTT residual self cue | 用 TTT replay 内部 residual，例如 `||y-v||`，做负证据或 block write gain | best 约 `36.37-36.41` | residual 比外部 semantic 更贴近 TTT，但仍不够准；容易用局部病灶换后段连续性 |
| update-conflict cue | 用 TTT 自身 fast-weight update conflict / energy 作为风险分数，判断哪些 token 的 replay update 和稳定更新方向冲突 | 进入 `35.x -> 34.x` 平台 | 这是目前最有用的 TTT 自有 cue，比 semantic scalar 和 residual-tail 更接近 harmful write |
| two-replay objective | 分开做正向 replay 和负向 replay，再合成 fast weight | 第一版未过 TTGRL_04 | 两组 replay 跑通，但 `all positive + risk negative` 太粗；需要三组 objective |
| tri-replay objective | 把 token 分成 positive / neutral / negative 三组：低风险 token 正向写，高风险 token 负向扣，中间 token 作为 neutral continuity 弱保留 | 当前 best `WINGAM_03 = 34.1903 / 6.5666` | 这是目前 v7 最大进展；关键不是强负证据，而是 positive / neutral / negative 的长期平衡 |
| reset 生命周期对照 | 在 strong tri-replay 配置上测试 `RESET_EVERY=10` 和不 reset | reset10: `63.3871`，reset0: `651.5409` | 简单延长 TTT fast-weight 生命周期会严重污染；长期轨迹优化不能靠“不 reset”，必须在短 reset 周期内控制写入内容 |

### 35.3 当前最好配置：具体怎么做

当前最好是：

```text
run = V7_TRIREPLAY_WINGAM_03_bodyg005_exitg0030_neu085_SWKS3
ATE / Rot = 34.1903 / 6.5666
RPE t / r = 92.4202 / 0.0083
```

具体策略如下：

```text
读路径：
    使用 C23 past-only attention cue 产生 D_g
    frame attention pair bias 打到所有 read layers
    beta = 4.75

SWA：
    使用 SWKS3-style 固定协议
    只保留 overlap 相关 SWA history
    在最后一个 SWA layer 对上一段 overlap source 的 K/V 做 alpha=0.50 的 source replacement

TTT 正向基础写入：
    先用 stage_d 作为几何/稳定性写入资格
    再乘 sqrt(1 - D_g)，温和压低动态/不稳定 patch
    只作用 branch w0，w1/w2 不做同样的动态 prior

TTT 自有风险 cue：
    使用 update_conflict_energy
    它不是外部 semantic，也不是 attention D_g
    它来自 TTT replay/update 本身，用来衡量 token 对 fast-weight update 的冲突/风险能量

tri-replay：
    对 chunks 5-9:
        positive_frac = 0.35
        negative_frac = 0.12
        neutral_lambda = 0.85
        negative gamma = 0.005
    对 chunks 10-12:
        positive_frac = 0.35
        negative_frac = 0.12
        neutral_lambda = 0.85
        negative gamma = 0.003
    其它 chunks:
        不做 tri-replay negative correction

含义：
    lowest-risk 35% token:
        正向 replay，作为长期几何/连续性证据写入
    highest-risk 12% token:
        小负 replay，从 w0 update 里扣掉
    中间 token:
        不是删掉，而是作为 neutral continuity，以 0.85 倍保留
```

这比早期 `TTGRL_04` 的“chunk5 单点轻反转”更结构化：它把病灶入口、主体和出口窗口分开处理，让 chunks `5-9` 保持较强纠偏，chunks `10-12` 用更弱的 negative gamma 保护后续轨迹连续性。

### 35.4 效果对比

| 配置 | ATE RMSE | Rot RMSE | FinalErr | `[200,300)` | `[200,400)` | 结论 |
|---|---:|---:|---:|---:|---:|---|
| B0/SWKS3 baseline | `36.4161` | `6.6128` | `5.798` | `77.831` | `57.101` | v7 复现基线 |
| localized chunk5 TTGR | `36.2957` | `6.6182` | `5.387` | `77.644` | `56.968` | 证明 chunk5/w0 轻负证据有效，但提升有限 |
| first strong tri-replay window `5-12` | `34.3660` | `6.6336` | `6.150` | `75.497` | `55.405` | 第一次大幅跳到 34m 平台 |
| split-window gamma `5-9` / `10-12` | `34.3421` | `6.5767` | `6.145` | `75.440` | `55.376` | 出口段 gamma 需要更弱 |
| neutral continuity shrink | `34.2680` | `6.6070` | `5.826` | `75.512` | `55.391` | neutral token 不能全量写，应收缩到 0.85 |
| 当前 best `WINGAM_03` | `34.1903` | `6.5666` | `6.195` | `75.576` | `55.428` | ATE / Rot 当前最好，但 `[200,300)` 不是最低 |

关键观察：

1. 当前 best 并不是靠把 `[200,300)` 单段压到最低获得的；它的 `[200,300)=75.576` 略差于 `WINMAP_04=75.440`。
2. 当前 best 更像是长期轨迹平衡更好：overall ATE、Rot、x/z 轴 RMSE 和全局 Sim3 scale 一起改善。
3. 这符合用户提醒：TTT 写入策略关键不只是 chunk 内局部优化，更要优化长期 trajectory state。

### 35.5 目前最重要的 insight

1. **freeze 是诊断，不是解法。**  
   `freeze56` 能把 `[200,300)` 打到 `26m`，但 overall ATE 变成 `60m+`。这说明 chunks 5/6 里有病灶因果信息，也有必需的尺度/连续性信息，不能 hard remove。

2. **TTT 自己的 cue 比外部 semantic 更有价值。**  
   Mask2Former 解决了 semantic 覆盖问题，但 semantic scalar write 仍弱。真正把 ATE 从 `36.3` 推到 `34.2` 的，是 `update_conflict_energy` 这种 TTT update-internal cue。

3. **负证据需要窗口化，不是全局化。**  
   chunk5 是入口，但只动 chunk5 只能到 `36.2957`。把 update-conflict tri-replay 扩到 chunks `5-12` 后进入 `34m` 平台。有效窗口大约覆盖病灶入口、主体和出口，而不是全序列。

4. **exit window 要更温和。**  
   chunks `10-12` 负责把病灶段带回长期轨迹连续性。它们需要比 chunks `5-9` 更弱的 negative gamma：`0.003` 比 `0.0035/0.004` 更好。

5. **neutral continuity 是关键变量。**  
   中间风险 token 不能当作 fully positive 写入，也不能删掉。`neutral_lambda=0.85` 是当前最好点，说明中等风险 token 应该弱保留，作为连续性证据。

6. **不能靠延长 reset 获得长期优化。**  
   `RESET_EVERY=10` 和不 reset 都严重崩坏。长期轨迹优化必须靠“写入内容/方向”对，而不是让 fast weight 活更久。

7. **当前仍没有达到最终目标。**  
   `34.1903m` 距离 `KITTI01 ATE < 30m` 还差约 `4.19m`。继续在 body/exit gamma 上细扫收益会很低，下一步应该改 replay 结构或加入更直接的长期 trajectory cue。

### 35.6 WINGAMF follow-up 状态

在 `WINGAM_03` 之后曾启动一批很窄的 exit gamma follow-up：

```text
WINGAMF_01: exit gamma 0.0025
WINGAMF_02: exit gamma 0.00275
WINGAMF_03: exit gamma 0.00325
WINGAMF_04: exit gamma 0.0030 + exit neutral 0.80
```

用户随后要求总结当前实验，因此该批在未完成时中止；截至停止时只跑到约 `18-21 / 38` chunks，没有完整 benchmark，不进入结果比较。

### 35.7 后续建议

不建议继续做大规模 full-sequence gamma 微扫。下一步更值得做：

1. **把 `update_conflict_energy` 正式变成 per-layer / per-head TTT self cue。**  
   现在它只作为 token risk 使用，后续应该统计哪些 TTT layer/head 的 conflict 真正对应长期 ATE 改善。

2. **做低成本筛选。**  
   full KITTI01 tri-replay 单 run 约 `30min+`，不适合盲扫。可以先用关键窗口 / cached update diagnostics 筛选，再跑 full。

3. **引入长期 trajectory cue。**  
   当前 tri-replay 仍是 chunk-local update cue。要继续接近 `<30m`，需要把后续窗口的轨迹一致性、scale drift 或 segment-level residual 反馈到 TTT commit 决策里。

4. **明确 long / short 写入角色。**  
   不是简单 dual-lifetime，也不是不 reset；而是：
   - short negative: 在 reset window 内修病灶；
   - long positive / neutral: 只保留经过 conflict 筛选的连续性更新；
   - exit chunks: 更弱负证据，负责回到长期轨迹。

## 34. WINNEU_08 附近：gamma-neutral joint micro sweep

33.6 后的当前 best 是：

```text
V7_TRIREPLAY_WINNEU_08_body_exit_neu085_SWKS3
body chunks 5-9:  gamma=0.0050, neutral=0.85
exit chunks 10-12: gamma=0.0035, neutral=0.85
ATE / Rot = 34.2680 / 6.6070
```

本批只做很窄的 gamma-neu 联合微调，不再扩大 all-chunk tri-replay sweep。固定：

```text
TTT self cue = update_conflict_energy
objective = tri replay
branch = w0
pos_frac = 0.35
neg_frac = 0.12
neutral_lambda = 0.85
SWA = SWKS3-style fixed protocol
RESET_EVERY = 5
```

候选：

| Run | Body gamma `5-9` | Exit gamma `10-12` | 目的 |
|---|---:|---:|---|
| `V7_TRIREPLAY_WINGAM_01` | `0.0045` | `0.0035` | body 稍弱 |
| `V7_TRIREPLAY_WINGAM_02` | `0.0055` | `0.0035` | body 稍强 |
| `V7_TRIREPLAY_WINGAM_03` | `0.0050` | `0.0030` | exit 稍弱 |
| `V7_TRIREPLAY_WINGAM_04` | `0.0050` | `0.0040` | exit 稍强 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TRIREPLAY_WINGAM_01_bodyg0045_exitg0035_neu085_SWKS3` | `2026-05-10 03:12:55` | `03:47:37` | `34.7 min` |
| `V7_TRIREPLAY_WINGAM_02_bodyg0055_exitg0035_neu085_SWKS3` | `2026-05-10 03:12:55` | `03:43:49` | `30.9 min` |
| `V7_TRIREPLAY_WINGAM_03_bodyg005_exitg0030_neu085_SWKS3` | `2026-05-10 03:12:55` | `03:43:54` | `31.0 min` |
| `V7_TRIREPLAY_WINGAM_04_bodyg005_exitg0040_neu085_SWKS3` | `2026-05-10 03:12:55` | `03:43:53` | `31.0 min` |

Global metrics：

| Run | Body gamma | Exit gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `WINNEU_08` | `0.0050` | `0.0035` | `34.2680` | `6.6070` | `92.4188` | `0.0083` | previous best |
| `WINGAM_01` | `0.0045` | `0.0035` | `34.3553` | `6.6007` | `92.4210` | `0.0083` | body gamma 变弱后 ATE 回退 |
| `WINGAM_02` | `0.0055` | `0.0035` | `34.3523` | `6.6016` | `92.4212` | `0.0083` | body gamma 变强也回退 |
| `WINGAM_03` | `0.0050` | `0.0030` | `34.1903` | `6.5666` | `92.4202` | `0.0083` | **当前 v7 best** |
| `WINGAM_04` | `0.0050` | `0.0040` | `34.2364` | `6.5639` | `92.4208` | `0.0083` | Rot 最好，但 ATE 未过 WINGAM_03 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_wingam/
```

| Run | ATE RMSE | FinalErr | 100f `[200,300)` | 200f `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `WINNEU_08` | `34.2680` | `5.826` | `75.512` | `55.391` | `3.812` | `30.975154` |
| `WINGAM_01` | `34.3553` | `6.044` | `75.707` | `55.538` | `3.796` | `31.000133` |
| `WINGAM_02` | `34.3523` | `5.930` | `75.756` | `55.574` | `3.807` | `31.003221` |
| `WINGAM_03` | `34.1903` | `6.195` | `75.576` | `55.428` | `3.774` | `30.994373` |
| `WINGAM_04` | `34.2364` | `5.896` | `75.613` | `55.459` | `3.778` | `30.998016` |

34 结论：

1. `WINGAM_03` 刷新当前 v7 best：`34.1903 / 6.5666`，相比 `WINNEU_08` 改善 `0.0778m / 0.0404deg`，相比 `B0` 改善 `2.2258m / 0.0462deg`。
2. body chunks `5-9` 的 gamma sweet spot 仍在 `0.0050`：`0.0045` 和 `0.0055` 都回退，说明 body window 的负证据强度已经很窄。
3. exit chunks `10-12` 的 gamma 应比 `WINNEU_08` 再弱一点：`0.0030` 最好，`0.0040` 虽然 Rot 更好但 ATE 略差。
4. 这次提升更像长期轨迹平衡改善，而不是只修 `[200,300)`：`[200,300)` 从 `75.512` 回退到 `75.576`，但整体 ATE、Rot、axis RMSE 和 200f 长段仍更好。
5. 距离最终目标 `KITTI01 ATE < 30m` 仍差约 `4.19m`。继续在 body/exit gamma 上做大矩阵不划算；如果继续，应以 `WINGAM_03` 为 base，优先改 replay 结构或引入更强的长期 trajectory cue，而不是继续全序列标量微扫。

当前 v7 best 更新为：

```text
V7_TRIREPLAY_WINGAM_03_bodyg005_exitg0030_neu085_SWKS3
ATE / Rot = 34.1903 / 6.5666
RPE t / r = 92.4202 / 0.0083
```

---

## 33. WINMAP split-window 后续：exit role routing 收口

本节接续 32 节的长期轨迹导向探索。用户指出 all-chunk tri-replay 单 run 接近 1 小时不能接受，因此后续只保留低成本窗口实验，不再跑全序列 tri-replay 大矩阵。

当前局部最强参考：

```text
V7_TRIREPLAY_WINMAP_04_c5to9g005_c10to12g0035_SWKS3
chunks 5-9 gamma = 0.005
chunks 10-12 gamma = 0.0035
tri replay = pos 0.35 / neg 0.12 / neutral 1.0
negative risk = update_conflict_energy
branch = w0
```

### 33.1 当前 best

| Run | Gamma map | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | none | `36.4161` | `6.6128` | `92.4452` | `0.0082` | v7 reproduction baseline |
| `TTGRL_04` | chunk5 `0.025` low-prior TTGR | `36.2957` | `6.6182` | `92.4428` | `0.0082` | pre-trireplay best |
| `WIN_09` | chunks `5-12:0.005` | `34.3660` | `6.6336` | `92.4235` | `0.0083` | first large tri-replay jump |
| `WINMAP_04` | `5-9:0.005, 10-12:0.0035` | `34.3421` | `6.5767` | `92.4225` | `0.0083` | **current v7 best** |

Trajectory diagnostics:

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `WIN_09` | `34.3660` | `6.150` | `75.497` | `55.405` | `3.835` | `31.023986` |
| `WINMAP_04` | `34.3421` | `6.145` | `75.440` | `55.376` | `3.780` | `31.010462` |

结论：

1. `WINMAP_04` 是当前 v7 最好结果，ATE 从 B0 的 `36.4161` 降到 `34.3421`，已经是 v7 里最实质的进展。
2. 它仍远高于最终目标 `KITTI01 ATE < 30m`，还差约 `4.34m`。
3. split-window 的有效信息是：chunks `5-9` 需要较强的 update-conflict negative correction，chunks `10-12` 是 exit / handoff 区域，应该用更弱 gamma。

### 33.2 exit role routing 对照

为了确认 exit chunks `10-12` 是否应该改变 pos/neg token 配比，而不是只降低 gamma，本批新增 chunk-specific tri-replay role routing：

```text
base:
    chunks 5-9  gamma=0.005
    chunks 10-12 gamma=0.0035
    global pos/neg/neutral = 0.35 / 0.12 / 1.0

WINROLE_01:
    chunks 10-12 pos/neg/neutral = 0.35 / 0.08 / 1.0

WINROLE_02:
    chunks 10-12 pos/neg/neutral = 0.45 / 0.08 / 1.0
```

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TRIREPLAY_WINROLE_01_c5to9_base_c10to12_neg008_SWKS3` | `2026-05-10 00:50:59` | `2026-05-10 01:23:55` | `33.0 min` |
| `V7_TRIREPLAY_WINROLE_02_c5to9_base_c10to12_pos045neg008_SWKS3` | `2026-05-10 00:50:59` | `2026-05-10 01:22:22` | `31.4 min` |

Global metrics:

| Run | Exit role params | ATE RMSE | Rot RMSE | RPE t | RPE r | vs `WINMAP_04` | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `WINMAP_04` | pos `0.35`, neg `0.12`, neu `1.0` | `34.3421` | `6.5767` | `92.4225` | `0.0083` | reference | current best |
| `WINROLE_01` | pos `0.35`, neg `0.08`, neu `1.0` | `34.4151` | `6.6067` | `92.4233` | `0.0083` | `+0.0730` | 减少 exit negative fraction 回退 |
| `WINROLE_02` | pos `0.45`, neg `0.08`, neu `1.0` | `34.3670` | `6.5488` | `92.4228` | `0.0083` | `+0.0249` | Rot/Yaw 略好，但 ATE 未过 |

Trajectory diagnostics:

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_winrole/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINMAP_04` | `34.3421` | `6.145` | `75.440` | `55.376` | `3.780` | `31.010462` |
| `WINROLE_01` | `34.4151` | `6.173` | `75.581` | `55.501` | `3.797` | `31.021263` |
| `WINROLE_02` | `34.3670` | `6.096` | `75.542` | `55.469` | `3.758` | `31.017244` |

33.2 结论：

1. exit chunks `10-12` 改 pos/neg token 比例没有超过 `WINMAP_04`。
2. `WINROLE_02` 的 Rot/Yaw/FinalErr 稍好，但 ATE、`[200,300)`、`[200,400)` 都比 `WINMAP_04` 略差，因此不能晋级。
3. 这说明 exit 区域的关键不是 “少选 negative token / 多选 positive token”，而是 **整体 negative correction 强度要弱**。目前最好的表达仍是 `chunks 10-12 gamma=0.0035`。
4. 后续不继续扫 exit role fraction。若继续 TTT，应换机制，而不是在 `pos/neg/neutral` 比例上细扫。

### 33.3 当前下一步判断

当前最可信的 TTT 策略形状是：

```text
TTT self cue = update_conflict_energy
objective = tri replay
branch = w0
window:
    chunks 5-9  : gamma 0.005
    chunks 10-12: gamma 0.0035
positive / negative / neutral = 0.35 / 0.12 / 1.0
reset = 5
SWA = SWKS3 fixed
```

下一步不建议：

1. 不跑 all-chunk tri-replay，单 run 太长且用户已明确不能接受；
2. 不继续扫 uniform gamma；
3. 不继续扫 exit pos/neg fraction；
4. 不把 freeze 当策略，只把它作为因果诊断。

更值得继续的方向：

1. **TTT 自有 cue 诊断**：把 `update_conflict_energy` 做 per-layer / per-head / per-chunk 统计，找出为什么 chunks `5-9` 有效而 `10-12` 需要弱化。
2. **窗口级长期轨迹目标**：当前所有写入仍只看 chunk 内 replay，下一步应引入跨 chunk surrogate，例如对 chunk `5-12` 的 commit 加 downstream consistency score，而不是只调 replay token。
3. **双 fast-weight 生命周期**：保留 `w0` tri-replay correction 在当前 reset 周期内生效，但把长期 continuity 和 short negative correction 分开维护，避免同一个 fast-weight 同时承担局部修病灶和长期轨迹连续性。

### 33.4 neutral continuity 窄矩阵

WINROLE 说明 exit chunks `10-12` 不适合简单改 pos/neg token 比例。下一步改 `neutral_lambda`，测试中性 replay continuity 是否应该在 body 或 exit 窗口收缩/放大。

固定：

```text
base = WINMAP_04
chunks 5-9 gamma = 0.005
chunks 10-12 gamma = 0.0035
global pos / neg / neutral = 0.35 / 0.12 / 1.0
```

候选：

| Run | Window | Chunk params | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| `WINMAP_04` | reference | all neutral `1.0` | `34.3421` | `6.5767` | `92.4225` | `0.0083` | previous best |
| `V7_TRIREPLAY_WINNEU_01` | exit `10-12` | pos/neg/neutral `0.35/0.12/0.85` | `34.2739` | `6.6084` | `92.4224` | `0.0083` | **current v7 best** |
| `V7_TRIREPLAY_WINNEU_02` | exit `10-12` | pos/neg/neutral `0.35/0.12/1.15` | `34.3953` | `6.5460` | `92.4230` | `0.0083` | Rot 好，但 ATE 回退 |
| `V7_TRIREPLAY_WINNEU_03` | body `5-9` | pos/neg/neutral `0.35/0.12/0.85` | `34.2835` | `6.5605` | `92.4203` | `0.0082` | 也过 WINMAP_04，但略差于 WINNEU_01 |
| `V7_TRIREPLAY_WINNEU_04` | body `5-9` | pos/neg/neutral `0.35/0.12/1.15` | `34.5908` | `6.6068` | `92.4256` | `0.0083` | body neutral 放大失败 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TRIREPLAY_WINNEU_01_exit_neu085_SWKS3` | `2026-05-10 01:28:50` | `2026-05-10 01:59:07` | `30.3 min` |
| `V7_TRIREPLAY_WINNEU_02_exit_neu115_SWKS3` | `2026-05-10 01:28:50` | `2026-05-10 02:00:37` | `31.8 min` |
| `V7_TRIREPLAY_WINNEU_03_body_neu085_SWKS3` | `2026-05-10 01:28:50` | `2026-05-10 02:01:20` | `32.5 min` |
| `V7_TRIREPLAY_WINNEU_04_body_neu115_SWKS3` | `2026-05-10 01:28:50` | `2026-05-10 01:59:00` | `30.2 min` |

Trajectory diagnostics:

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_winneu/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINMAP_04` | `34.3421` | `6.145` | `75.440` | `55.376` | `3.780` | `31.010462` |
| `WINNEU_01` | `34.2739` | `5.835` | `75.300` | `55.238` | `3.805` | `31.012096` |
| `WINNEU_02` | `34.3953` | `6.118` | `75.581` | `55.498` | `3.753` | `31.015436` |
| `WINNEU_03` | `34.2835` | `6.157` | `75.704` | `55.559` | `3.771` | `30.990214` |
| `WINNEU_04` | `34.5908` | `6.390` | `75.643` | `55.535` | `3.814` | `31.041463` |

33.4 结论：

1. `WINNEU_01` 刷新当前 v7 best：`34.2739 / 6.6084`。相比 `WINMAP_04`，ATE 改善 `0.0681m`，`[200,300)` 改善 `0.139m`，`[200,400)` 改善 `0.138m`，FinalErr 改善 `0.310m`。
2. `WINNEU_03` 也超过 `WINMAP_04`，说明 neutral continuity 收缩是有效方向；但 body 收缩主要改善 RPE/Rot，局部 `[200,300)` 不如 exit 收缩。
3. `neutral_lambda=1.15` 在 body 或 exit 都不适合：放大中性 replay 会回退 ATE。
4. 新机制判断：长期轨迹收益不是来自更强 negative，也不是来自更多 positive，而是 **在 exit handoff 处减少中性 replay 的惯性**。也就是说，chunks `10-12` 需要保留弱 negative correction，但同时不能让 neutral full-continuity replay 过度盖住前面 chunks `5-9` 的修正方向。

当前 v7 best 更新为：

```text
V7_TRIREPLAY_WINNEU_01_exit_neu085_SWKS3
ATE / Rot = 34.2739 / 6.6084
```

### 33.5 neutral continuity fine sweep

围绕 `WINNEU_01` 做 narrow sweep，确认 exit neutral 的 sweet spot，并测试 body+exit 同时收缩是否能叠加。

候选：

| Run | Window | Chunk params | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| `WINNEU_01` | exit `10-12` | neutral `0.85` | `34.2739` | `6.6084` | `92.4224` | `0.0083` | previous best |
| `V7_TRIREPLAY_WINNEU_05` | exit `10-12` | neutral `0.75` | `34.3137` | `6.6148` | `92.4224` | `0.0083` | exit neutral 过低，ATE 回退 |
| `V7_TRIREPLAY_WINNEU_06` | exit `10-12` | neutral `0.80` | `34.3170` | `6.5979` | `92.4225` | `0.0083` | 未过 0.85 |
| `V7_TRIREPLAY_WINNEU_07` | exit `10-12` | neutral `0.90` | `34.3124` | `6.5816` | `92.4212` | `0.0083` | Rot/RPE 好，但 ATE 未过 |
| `V7_TRIREPLAY_WINNEU_08` | body `5-9` + exit `10-12` | neutral `0.85` | `34.2680` | `6.6070` | `92.4188` | `0.0083` | **current v7 best** |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TRIREPLAY_WINNEU_05_exit_neu075_SWKS3` | `2026-05-10 02:03:29` | `2026-05-10 02:36:21` | `32.9 min` |
| `V7_TRIREPLAY_WINNEU_06_exit_neu080_SWKS3` | `2026-05-10 02:03:29` | `2026-05-10 02:35:33` | `32.1 min` |
| `V7_TRIREPLAY_WINNEU_07_exit_neu090_SWKS3` | `2026-05-10 02:03:29` | `2026-05-10 02:33:54` | `30.4 min` |
| `V7_TRIREPLAY_WINNEU_08_body_exit_neu085_SWKS3` | `2026-05-10 02:03:29` | `2026-05-10 02:33:30` | `30.0 min` |

Trajectory diagnostics:

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_winneu_fine/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINNEU_01` | `34.2739` | `5.835` | `75.300` | `55.238` | `3.805` | `31.012096` |
| `WINNEU_05` | `34.3137` | `6.125` | `75.318` | `55.244` | `3.808` | `31.009551` |
| `WINNEU_06` | `34.3170` | `6.297` | `75.382` | `55.296` | `3.800` | `31.013713` |
| `WINNEU_07` | `34.3124` | `6.060` | `75.314` | `55.270` | `3.785` | `30.995565` |
| `WINNEU_08` | `34.2680` | `5.826` | `75.512` | `55.391` | `3.812` | `30.975154` |

33.5 结论：

1. exit-only neutral sweet spot 确认在 `0.85` 附近；`0.75/0.80/0.90` 都没有超过 `WINNEU_01`。
2. body+exit 同时 `0.85` 刷新 overall ATE 到 `34.2680`，但 `[200,300)` 从 `75.300` 回退到 `75.512`。它不是更好地修病灶段，而是改善了全局长段平衡 / 后段 RPE。
3. 这支持用户关于 “TTT 写入策略要优化长期轨迹” 的判断：body neutral 收缩牺牲了一点局部病灶，却改善 overall ATE 和 RPE。
4. 下一步不再动 exit neutral；固定 exit `0.85`，只在 body `5-9` 上围绕 `0.85` 做微调，找全局 ATE 和病灶段之间的折中。

当前 v7 best 更新为：

```text
V7_TRIREPLAY_WINNEU_08_body_exit_neu085_SWKS3
ATE / Rot = 34.2680 / 6.6070
```

### 33.6 body neutral fine sweep

固定 exit `10-12` neutral `0.85`，只调 body `5-9` neutral，验证 `WINNEU_08` 是否已经是 body neutral 的最优点。

| Run | Body neutral | Exit neutral | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `WINNEU_08` | `0.85` | `0.85` | `34.2680` | `6.6070` | `92.4188` | `0.0083` | current best |
| `V7_TRIREPLAY_WINNEU_09` | `0.75` | `0.85` | `34.2869` | `6.5509` | `92.4194` | `0.0082` | Rot/FinalErr 好，但 ATE 未过 |
| `V7_TRIREPLAY_WINNEU_10` | `0.80` | `0.85` | `34.3296` | `6.5443` | `92.4198` | `0.0082` | 回退 |
| `V7_TRIREPLAY_WINNEU_11` | `0.90` | `0.85` | `34.3492` | `6.5767` | `92.4202` | `0.0082` | 回退 |
| `V7_TRIREPLAY_WINNEU_12` | `0.95` | `0.85` | `34.3143` | `6.6055` | `92.4226` | `0.0083` | 回退 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TRIREPLAY_WINNEU_09_body075_exit085_SWKS3` | `2026-05-10 02:38:07` | `2026-05-10 03:08:51` | `30.7 min` |
| `V7_TRIREPLAY_WINNEU_10_body080_exit085_SWKS3` | `2026-05-10 02:38:07` | `2026-05-10 03:11:03` | `32.9 min` |
| `V7_TRIREPLAY_WINNEU_11_body090_exit085_SWKS3` | `2026-05-10 02:38:07` | `2026-05-10 03:09:52` | `31.8 min` |
| `V7_TRIREPLAY_WINNEU_12_body095_exit085_SWKS3` | `2026-05-10 02:38:07` | `2026-05-10 03:08:47` | `30.7 min` |

Trajectory diagnostics:

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_winneu_body_fine/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINNEU_08` | `34.2680` | `5.826` | `75.512` | `55.391` | `3.812` | `30.975154` |
| `WINNEU_09` | `34.2869` | `5.587` | `75.771` | `55.595` | `3.742` | `30.985809` |
| `WINNEU_10` | `34.3296` | `5.547` | `75.693` | `55.541` | `3.752` | `30.992989` |
| `WINNEU_11` | `34.3492` | `5.707` | `75.610` | `55.488` | `3.789` | `30.993652` |
| `WINNEU_12` | `34.3143` | `5.988` | `75.593` | `55.453` | `3.804` | `31.015603` |

33.6 结论：

1. body neutral `0.85` 仍是当前 ATE 最优；`0.75/0.80/0.90/0.95` 都没有超过 `WINNEU_08`。
2. body neutral 降到 `0.75/0.80` 能改善 Rot / FinalErr，但会明显损 `[200,300)` 和 ATE。这是典型 local orientation 与 global trajectory 的 trade-off。
3. neutral continuity 线目前收口：`body=0.85, exit=0.85` 是当前最稳组合。
4. 下一步若继续，只值得做 `WINNEU_08` 附近的 gamma-neu 联合微调；单独继续扫 neutral 没意义。

---

## 32. WIN_09 之后：长期轨迹导向的 TTT 写入探索

说明：本节是 `2026-05-09/10` 的追加实验。用户指出 TTT 写入策略不能只优化 chunk 内局部效果，更要优化长期轨迹；同时 `RESET_EVERY=0/10` 已经证明直接延长 fast-weight 生命周期会崩坏。因此这里从 `WIN_09` 出发，测试更细的长期/出口段写入结构。

当前进入本节前的 best：

```text
WIN_09 = chunks 5-12, tri_replay, update_conflict_energy, w0, gamma 0.005
ATE / Rot = 34.3660 / 6.6336
```

### 32.1 long residual whitelist / update-conflict commit gate

假设：

```text
WIN_09 的 update-conflict residual 不能直接延长生命周期；
但也许可以只让低 conflict 的 residual 进入长期 commit，
高 conflict 的 residual 在 commit 时往 native / old fast weight 拉回。
```

固定：

```text
base = WIN_09
TTT_WRITE_COMMIT_FILTER_MODE = native_to_candidate_by_risk
TTT_WRITE_COMMIT_FILTER_RISK_SOURCE = update_conflict_energy
TTT_WRITE_COMMIT_FILTER_CHUNKS = 5,6,7,8,9,10,11,12
TTT_WRITE_COMMIT_FILTER_BRANCH_MASK = 0
```

结果：

| Run | Commit filter | Scale mean approx | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `WIN_09` | none | n/a | `34.3660` | `6.6336` | `92.4235` | `0.0083` | reference |
| `V7_LONGWL_01` | mean, gain `-0.25`, min `0.75` | `0.982` | `34.4894` | `6.5516` | `92.4225` | `0.0083` | Rot/RPE 好，但 ATE 回退 |
| `V7_LONGWL_02` | q90, gain `-0.25`, min `0.75` | `0.958` | `34.3852` | `6.5958` | `92.4224` | `0.0083` | 接近 WIN_09，但未过 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_long_whitelist/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `WIN_09` | `34.3660` | `6.150` | `75.497` | `55.405` | `3.835` | `31.023986` |
| `LONGWL_01` | `34.4894` | `5.771` | `75.657` | `55.552` | `3.764` | `31.014366` |
| `LONGWL_02` | `34.3852` | `6.258` | `75.541` | `55.452` | `3.804` | `31.006972` |

结论：

1. long whitelist 没有达到 `<30m`，也没有超过 `WIN_09`。
2. q90 filter 接近 WIN_09，但仍回退 `0.019m`；mean filter 回退更明显。
3. 这说明 WIN_09 的有效长期 residual 不是简单按 layer conflict mean/q90 就能筛出来的。过度往 native 拉回会损失主 RMSE 收益。
4. 该方向不继续扩矩阵；如果未来继续做 long gate，需要更精确地拆 positive continuity 与 negative conflict，而不是 commit 后统一缩放。

### 32.2 post-window positive/neutral replay 诊断

假设：

```text
5-12 负责病灶 conflict correction；
13-20 可能需要 positive/neutral replay 补偿长期 continuity。
```

实现方式：

```text
chunks 5-12: gamma = 0.005
chunks 13-20: gamma = 0.0001
mode = tri_replay
```

`13-20` 的极小 gamma 只用于触发 tri-replay 分组，近似 positive/neutral-only 后窗；它不是主线，因为单 run 时间明显变长。

| Run | Active chunks | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `WIN_09` | `5-12`, gamma `0.005` | `34.3660` | `6.6336` | `92.4235` | `0.0083` | reference |
| `V7_TRIREPLAY_WINPOS_01` | `5-12:0.005`, `13-20:0.0001` | `34.5670` | `6.4005` | `92.4076` | `0.0082` | 局部病灶更好，但全局 ATE 回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_winpos/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `WIN_09` | `34.3660` | `6.150` | `75.497` | `55.405` | `42.831` | `3.835` | `31.023986` |
| `WINPOS_01` | `34.5670` | `7.060` | `74.745` | `54.945` | `45.736` | `3.741` | `30.822223` |

结论：

1. post-window positive/neutral replay 没有超过 WIN_09。
2. 它把 `[200,300)` 从 `75.497` 降到 `74.745`，局部病灶确实更好；但 `[400,600)` 从 `42.831` 退到 `45.736`，FinalErr 也变差。
3. 这说明后窗 replay 可以改变长期 trajectory balance，但 13-20 过宽、代价太高；不能用作主线。
4. 该结果支持用户判断：TTT 写入确实需要长期轨迹视角，但后段补偿必须更精确，不能把一整段后窗都纳入 tri-replay。

### 32.3 WIN_09 gamma 窄扫

固定 chunks `5-12`，只测试 `gamma=0.004/0.006`。

| Run | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `WIN_09` | `0.0050` | `34.3660` | `6.6336` | `92.4235` | `0.0083` | reference |
| `V7_TRIREPLAY_WIN_13` | `0.0040` | `34.4883` | `6.6058` | `92.4241` | `0.0083` | 太弱，ATE 回退 |
| `V7_TRIREPLAY_WIN_14` | `0.0060` | `34.4543` | `6.5954` | `92.4237` | `0.0083` | 稍强也回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_win09_gamma_fine/
```

结论：

1. 全窗口统一 gamma 的 sweet spot 基本确认是 `0.005`。
2. `0.004/0.006` 都会损失 ATE；继续全窗口 gamma 细扫意义不大。

### 32.4 split-window gamma：主体强、出口弱

新的结构假设：

```text
chunks 5-9 = 病灶主体 / conflict correction，应保持 gamma 0.005；
chunks 10-12 = 出口 / 长期交接，应降低 gamma，减少后段 trajectory 代价。
```

第一批：

| Run | Chunk gamma map | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `WIN_09` | `5-12:0.005` | `34.3660` | `6.6336` | `92.4235` | `0.0083` | reference |
| `V7_TRIREPLAY_WINMAP_01` | `5-9:0.005, 10-12:0.003` | `34.3440` | `6.5635` | `92.4227` | `0.0083` | 新 best；ATE/Rot/FinalErr 同时好 |
| `V7_TRIREPLAY_WINMAP_02` | `5-9:0.005, 10-12:0.007` | `34.3534` | `6.5910` | `92.4221` | `0.0083` | 也过 WIN_09，但不如 0.003 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_winmap_split/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `WIN_09` | `34.3660` | `6.150` | `75.497` | `55.405` | `3.835` | `31.023986` |
| `WINMAP_01` | `34.3440` | `5.645` | `75.431` | `55.366` | `3.762` | `31.016851` |
| `WINMAP_02` | `34.3534` | `5.966` | `75.398` | `55.338` | `3.800` | `31.007894` |

结论：

1. `WINMAP_01` 刷新 v7 best：`34.3440 / 6.5635`。
2. 这比 long whitelist 更有价值：它不是在 commit 后统一拉回 native，而是按 trajectory role 把窗口拆成主体 correction 与出口交接。
3. `10-12` 降到 `0.003` 同时改善 ATE、Rot、FinalErr、Yaw，说明出口段确实不应该和病灶主体同强度。

### 32.5 split-window gamma 细化

继续围绕出口段 `0.003` 做细扫：

| Run | Chunk gamma map | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `WINMAP_01` | `5-9:0.005, 10-12:0.0030` | `34.3440` | `6.5635` | `92.4227` | `0.0083` | previous best |
| `V7_TRIREPLAY_WINMAP_03` | `5-9:0.005, 10-12:0.0025` | `34.3804` | `6.6028` | `92.4234` | `0.0083` | 出口太弱，回退 |
| `V7_TRIREPLAY_WINMAP_04` | `5-9:0.005, 10-12:0.0035` | `34.3421` | `6.5767` | `92.4225` | `0.0083` | **当前 v7 tiny best** |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_winmap_fine/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `WINMAP_01` | `34.3440` | `5.645` | `75.431` | `55.366` | `3.762` | `31.016851` |
| `WINMAP_03` | `34.3804` | `6.355` | `75.528` | `55.434` | `3.803` | `31.021114` |
| `WINMAP_04` | `34.3421` | `6.145` | `75.440` | `55.376` | `3.780` | `31.010462` |

再调 chunk5 入口强度：

| Run | Chunk gamma map | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `WINMAP_04` | `5-9:0.005, 10-12:0.0035` | `34.3421` | `6.5767` | `92.4225` | `0.0083` | reference / current best |
| `V7_TRIREPLAY_WINMAP_05` | `5:0.0035, 6-9:0.005, 10-12:0.0035` | `34.4262` | `6.5754` | `92.4219` | `0.0083` | chunk5 过弱，ATE 回退 |
| `V7_TRIREPLAY_WINMAP_06` | `5:0.0065, 6-9:0.005, 10-12:0.0035` | `34.4082` | `6.6046` | `92.4224` | `0.0083` | chunk5 过强，也回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_winmap_entry/
```

32.5 结论：

1. 当前 v7 best 更新为：

```text
V7_TRIREPLAY_WINMAP_04_c5to9g005_c10to12g0035_SWKS3
ATE / Rot = 34.3421 / 6.5767
```

2. 这比 `WIN_09` 只好 `0.0239m`，但方向明确：入口/主体/出口的 gamma 应该分层，而不是整个 window 一个标量。
3. chunk5 入口保持 `0.005` 最好；降到 `0.0035` 或升到 `0.0065` 都回退。
4. 出口 `10-12` 降到 `0.003-0.0035` 比 `0.005` 更好，说明长期轨迹优化主要发生在病灶出口/交接段，不是简单加强/减弱整个窗口。
5. 当前仍远高于最终目标 `<30m`，但本节把 best 从 `34.3660` 推到 `34.3421`，并明确了下一步结构方向：继续做 **window role routing**，例如按 chunk role 选择不同 `pos_frac / neg_frac / neutral_lambda`，而不是继续只扫 gamma。

---

## 31. chunk-window update-conflict tri-replay

用户指出全序列 tri-replay 单 run 接近 1 小时，不能作为常规探索方式。因此本节停止全序列细扫，把 `GLOBAL_01` 只作为强信号 reference，改成只在少数关键 chunk window 启用 `update_conflict_energy` tri-replay。

固定协议：

```text
seq = KITTI01 full
SWA = SWKS3-style fixed protocol
TTT_WRITE_GRADIENT_REVERSAL_MODE = tri_replay
risk = update_conflict_energy
branch = w0
positive_frac = 0.35
negative_frac = 0.12
neutral_lambda = 1.0
```

第一批 window：

| Run | Active chunks | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | none | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | chunk5 low-prior | `0.025` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | old localized TTGR best |
| `TRIREPLAY_12` | `5` | `0.030` | `35.8712` | `6.6142` | `92.4389` | `0.0082` | old tri-replay best |
| `GLOBAL_01` | all chunks | `0.005` | `35.1293` | `7.9762` | `92.3989` | `0.0096` | ATE 强，但 Rot/Final/Yaw 代价大 |
| `WIN_01` | `5,6,7,8,9,10` | `0.005` | `34.7706` | `6.5694` | `92.4260` | `0.0083` | **当前 v7 best；强 ATE 且 Rot 没崩** |
| `WIN_02` | `5,7,8,9` | `0.005` | `34.8857` | `6.5944` | `92.4180` | `0.0084` | 去掉 6/10 后略差 |
| `WIN_03` | `5,8,9,15,16` | `0.005` | `35.1249` | `6.5148` | `92.4191` | `0.0083` | 加后段 chunk 不如 5-10 window |
| `WIN_04` | `5,7,8,9` | `0.010` | `34.9453` | `6.6030` | `92.4176` | `0.0084` | gamma 更强未过 WIN_02 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_window_conflict/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.644` | `56.968` | `3.769` | `31.219364` |
| `TRIREPLAY_12` | `35.8712` | `5.573` | `76.863` | `56.433` | `3.793` | `31.177167` |
| `GLOBAL_01` | `35.1293` | `19.287` | `74.915` | `55.125` | `5.130` | `30.719503` |
| `WIN_01` | `34.7706` | `6.440` | `76.238` | `56.043` | `3.786` | `31.050356` |
| `WIN_02` | `34.8857` | `5.888` | `77.924` | `57.376` | `3.737` | `30.987162` |
| `WIN_03` | `35.1249` | `6.661` | `76.026` | `55.956` | `3.760` | `30.958956` |
| `WIN_04` | `34.9453` | `5.663` | `77.992` | `57.421` | `3.749` | `30.985023` |

31.1 结论：

1. `WIN_01 = chunks 5-10, gamma=0.005` 是当前 v7 最强结果：`34.7706 / 6.5694`，首次把 v7 推到 `34m` 段。
2. 相比 all-chunk `GLOBAL_01`，`WIN_01` 的 ATE 更好，Rot/Yaw/FinalErr 也明显更安全；说明全序列长期负证据不是必要条件，关键是 `[200,300)` 附近的局部窗口。
3. `WIN_02` 去掉 chunk6/10 后略差，说明 chunk6/10 在弱 tri-replay 下不是纯 continuity carrier，也提供有用的 conflict correction。
4. `WIN_03` 加入后段 chunk15/16 没有超过 `WIN_01`，说明当前收益仍主要来自 chunks 5-10，而不是任意后段 conflict chunk。
5. 下一步继续窄扫 `WIN_01` 周围，但保持普通 full run 成本，不再回到全序列 1h/run 矩阵。

正在跑的第二批：

| Run | Active chunks | Gamma | 目的 |
|---|---|---:|---|
| `WIN_05` | `5,6,7,8,9,10` | `0.0025` | 检查 WIN_01 是否 gamma 偏强 |
| `WIN_06` | `5,6,7,8,9,10` | `0.0075` | 检查 WIN_01 是否可继续外推 |
| `WIN_07` | `5,6,7,8,9` | `0.005` | 去掉 chunk10，验证窗口右边界 |
| `WIN_08` | `5,6,7,8,9,10,11` | `0.005` | 加 chunk11，验证窗口右边界 |

### 31.2 window refine 第二批

第二批结果：

| Run | Active chunks | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `WIN_01` | `5,6,7,8,9,10` | `0.005` | `34.7706` | `6.5694` | `92.4260` | `0.0083` | previous window best |
| `WIN_05` | `5,6,7,8,9,10` | `0.0025` | `34.8815` | `6.5244` | `92.4258` | `0.0083` | gamma 太弱，ATE 回退但 Rot 更好 |
| `WIN_06` | `5,6,7,8,9,10` | `0.0075` | `34.8247` | `6.5838` | `92.4250` | `0.0083` | gamma 稍强也回退 |
| `WIN_07` | `5,6,7,8,9` | `0.005` | `35.1681` | `6.5785` | `92.4293` | `0.0083` | 去掉 chunk10 后明显回退 |
| `WIN_08` | `5,6,7,8,9,10,11` | `0.005` | `34.4942` | `6.6067` | `92.4230` | `0.0083` | **当前 v7 best** |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_window_refine/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `TRIREPLAY_12` | `35.8712` | `5.573` | `76.863` | `56.433` | `3.793` | `31.177167` |
| `GLOBAL_01` | `35.1293` | `19.287` | `74.915` | `55.125` | `5.130` | `30.719503` |
| `WIN_01` | `34.7706` | `6.440` | `76.238` | `56.043` | `3.786` | `31.050356` |
| `WIN_05` | `34.8815` | `6.351` | `76.410` | `56.180` | `3.748` | `31.051191` |
| `WIN_06` | `34.8247` | `6.832` | `76.337` | `56.138` | `3.784` | `31.038451` |
| `WIN_07` | `35.1681` | `5.739` | `76.697` | `56.407` | `3.763` | `31.090331` |
| `WIN_08` | `34.4942` | `6.025` | `75.662` | `55.567` | `3.810` | `31.016633` |

31.2 结论：

1. `WIN_08 = chunks 5-11, gamma=0.005` 继续刷新 v7 best：`34.4942 / 6.6067`。
2. `gamma=0.005` 是当前窗口强度 sweet spot：`0.0025` 和 `0.0075` 都不如它。
3. 去掉 chunk10 明显回退，加入 chunk11 明显改善，说明有效窗口右边界至少到 chunk11。
4. `WIN_08` 比 all-chunk `GLOBAL_01` ATE 更好，并且 Rot/Yaw/FinalErr 代价小得多；这证明 “局部窗口 TTT 自有 conflict cue” 比全序列弱剂量更合理。
5. 当前距离最终目标 `KITTI01 ATE < 30m` 仍差 `4.49m`，但已经从 v7 初始 `36.4161` 改善 `1.9219m`。

第三批边界检查已启动：

| Run | Active chunks | Gamma | 目的 |
|---|---|---:|---|
| `WIN_09` | `5,6,7,8,9,10,11,12` | `0.005` | 继续右扩到 chunk12 |
| `WIN_10` | `5,6,7,8,9,10,11,12,13` | `0.005` | 继续右扩到 chunk13 |
| `WIN_11` | `4,5,6,7,8,9,10,11` | `0.005` | 左侧加入 chunk4 |
| `WIN_12` | `6,7,8,9,10,11` | `0.005` | 去掉 chunk5，验证 chunk5 是否仍必要 |

---

### 31.3 window boundary 第三批

第三批结果：

| Run | Active chunks | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `WIN_08` | `5,6,7,8,9,10,11` | `0.005` | `34.4942` | `6.6067` | `92.4230` | `0.0083` | previous window best |
| `WIN_09` | `5,6,7,8,9,10,11,12` | `0.005` | `34.3660` | `6.6336` | `92.4235` | `0.0083` | **当前 v7 best** |
| `WIN_10` | `5,6,7,8,9,10,11,12,13` | `0.005` | `34.5919` | `6.6009` | `92.4242` | `0.0083` | 右扩到 13 回退 |
| `WIN_11` | `4,5,6,7,8,9,10,11` | `0.005` | `34.4942` | `6.6067` | `92.4230` | `0.0083` | 与 WIN_08 完全重合，chunk4 无 effect |
| `WIN_12` | `6,7,8,9,10,11` | `0.005` | `34.4550` | `6.5835` | `92.4189` | `0.0083` | 去掉 chunk5 仍接近，但未过 WIN_09 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_window_boundary/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `WIN_08` | `34.4942` | `6.025` | `75.662` | `55.567` | `3.810` | `31.016633` |
| `WIN_09` | `34.3660` | `6.150` | `75.497` | `55.405` | `3.835` | `31.023986` |
| `WIN_10` | `34.5919` | `5.895` | `75.725` | `55.563` | `3.808` | `31.029063` |
| `WIN_11` | `34.4942` | `6.025` | `75.662` | `55.567` | `3.810` | `31.016633` |
| `WIN_12` | `34.4550` | `6.328` | `76.576` | `56.266` | `3.767` | `30.984402` |

31.3 结论：

1. `WIN_09 = chunks 5-12, gamma=0.005` 再次刷新 v7 best：`34.3660 / 6.6336`。
2. 有效窗口不是单 chunk，也不是全序列，而是覆盖 `[200,300)` 病灶入口、主体和出口附近的一段短窗口。相比 `B0`，`WIN_09` 的 ATE 改善 `2.0501m`，`[200,300)` 改善 `2.334m`。
3. 右扩到 chunk13 回退，说明 negative/conflict correction 的长期传播不能无限延伸；超过窗口后开始伤害后续 continuity。
4. 左加 chunk4 完全无 effect，进一步确认 chunk4 不是关键写入入口。
5. 去掉 chunk5 后 `WIN_12=34.4550` 仍接近，但 `[200,300)` 明显差于 `WIN_09`。这说明 chunk5 不再是唯一关键点，但它仍帮助病灶主段降低。
6. 用户指出的长期轨迹问题是正确的：当前 `WIN_09` 虽然强于局部 TTGR，但仍是按 chunk 内 update-conflict cue 改 commit，没有显式区分短期纠偏和长期轨迹连续性。下一步不继续盲目扩 window，转向 **dual-lifetime / short-negative-long-positive**：

```text
short state:
    使用 WIN_09 tri-replay negative/conflict residual，服务后续 1-2 个 chunk

long state:
    只提交 reference / positive continuity fast weight，或只保留少量 negative residual
```

目标是保留 `WIN_09` 对 `[200,300)` 的收益，同时减少 FinalErr / Yaw / 后段 continuity 代价。

---

### 31.4 dual-lifetime short negative / long positive 对照

用户指出 TTT 写入策略关键不只在于 chunk 内优化，更要优化长期轨迹。因此本批不再扩 window，而是在 `WIN_09` 基础上测试短期负证据与长期 fast weight commit 分离：

```text
active chunks = 5,6,7,8,9,10,11,12
risk = update_conflict_energy
tri replay = pos 0.35 / neg 0.12 / neutral 1.0
branch = w0
mode = dual_lifetime
```

含义：

- short state：当前 tri-replay 的 conflict residual，可在后续若干 chunk 的读取态中短期生效；
- long state：只提交 reference / positive continuity fast weight，或只保留一小部分 conflict residual；
- 目标：保留 `WIN_09` 的 `[200,300)` 收益，同时减少长期轨迹代价。

Smoke：

| Run | END_FRAME | 设置 | 结论 |
|---|---:|---|---|
| `V7_TRIDUAL_SMOKE_c0_g0005_ttl1_long000_e96` | `96` | chunk0, `ttl=1`, `long_scale=0.0` | 通过；debug 确认 `ttt_gradient_reversal_transient_applied=True`，并存储 `w0` short residual |

Full 结果：

| Run | Gamma | TTL | Long scale | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `WIN_09` | `0.005` | persistent | `1.00` | `34.3660` | `6.6336` | `92.4235` | `0.0083` | current v7 best |
| `TRIDUAL_01` | `0.005` | `1` | `0.00` | `35.8884` | `6.6254` | `92.4320` | `0.0083` | 纯 short residual，主收益消失 |
| `TRIDUAL_02` | `0.005` | `2` | `0.00` | `35.6902` | `6.6409` | `92.4302` | `0.0083` | TTL 加长略好，但仍回退 |
| `TRIDUAL_03` | `0.005` | `1` | `0.25` | `35.2160` | `6.6318` | `92.4240` | `0.0083` | 本批最好，但仍未过 WIN_09 |
| `TRIDUAL_04` | `0.0075` | `1` | `0.00` | `35.8894` | `6.6270` | `92.4319` | `0.0083` | stronger gamma 不能补偿 pure-short 回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_tridual/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `WIN_09` | `34.3660` | `6.150` | `75.497` | `55.405` | `3.835` | `31.023986` |
| `TRIDUAL_01` | `35.8884` | `5.802` | `79.277` | `58.025` | `3.747` | `31.123519` |
| `TRIDUAL_02` | `35.6902` | `5.578` | `78.916` | `57.782` | `3.752` | `31.104317` |
| `TRIDUAL_03` | `35.2160` | `5.937` | `78.286` | `57.383` | `3.767` | `31.042992` |
| `TRIDUAL_04` | `35.8894` | `5.826` | `79.291` | `58.032` | `3.747` | `31.123632` |

31.4 结论：

1. dual-lifetime 机制跑通，但没有超过 `WIN_09`。
2. 纯 short residual (`long_scale=0`) 会显著回退，尤其 `[200,300)` 从 `WIN_09=75.497` 退到 `79.277/78.916`。这说明 update-conflict tri-replay 的收益不能只用于短期读取；它必须至少部分进入长期 fast weight。
3. `long_scale=0.25` 明显好于 pure-short，ATE 到 `35.2160`，但仍低于 `WIN_09=34.3660`。当前 best 需要较强的长期 residual，而不是完全 long/short 分离。
4. 这批支持用户的长期轨迹判断：真正问题不是“负证据用一下就删”，而是要学习 **哪些 conflict residual 可以长期保留**。简单把 negative/conflict residual 从 long commit 中拿掉会破坏主收益。
5. 下一步不继续 pure dual-lifetime；更合理的是从 `WIN_09` 出发，做 reset/lifetime 对照和 TTT self cue 的长期保留门控。

### 31.5 WIN_09 reset / lifetime 对照

用户要求补充 `不 reset TTT` 和 `reset=10` 作为对照。本批固定 `WIN_09` 的 TTT 写入策略，只改变 fast-weight reset 生命周期：

```text
base = WIN_09
active chunks = 5,6,7,8,9,10,11,12
risk = update_conflict_energy
tri replay = pos 0.35 / neg 0.12 / neutral 1.0
branch = w0
SWA = SWKS3-style fixed protocol
```

运行记录：

| Run | RESET_EVERY | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V7_TRIREPLAY_WIN09_RESET0_c5to12_g0005_SWKS3` | `0` | `2026-05-09 17:39:11` | `18:12:30` | `33.3 min` |
| `V7_TRIREPLAY_WIN09_RESET10_c5to12_g0005_SWKS3` | `10` | `2026-05-09 17:39:11` | `18:08:54` | `29.7 min` |

Global metrics：

| Run | RESET_EVERY | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | `5` | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `WIN_09` | `5` | `34.3660` | `6.6336` | `92.4235` | `0.0083` | current v7 best |
| `RESET10` | `10` | `63.3871` | `8.5089` | `92.9686` | `0.0146` | 长生命周期明显污染 |
| `RESET0` | `0` | `651.5409` | `48.1025` | `94.8415` | `0.0609` | 不 reset 完全崩坏 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_win09_reset_compare/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `WIN_09` | `34.3660` | `6.150` | `75.497` | `55.405` | `3.835` | `31.023986` |
| `RESET10` | `63.3871` | `73.996` | `98.274` | `77.546` | `5.906` | `38.042164` |
| `RESET0` | `651.5409` | `942.865` | `981.476` | `834.573` | `37.430` | `69.398450` |

31.5 结论：

1. `RESET_EVERY=5` 仍是当前唯一安全生命周期。`WIN_09` 的收益不是因为 TTT fast weight 应该更长期保存，而是因为在一个短 reset 周期内对 chunks `5-12` 做了合适的 update-conflict tri-replay。
2. `RESET_EVERY=10` 已经严重污染：ATE 从 `34.3660` 崩到 `63.3871`，Sim3 scale 从 `31.02` 漂到 `38.04`，FinalErr 到 `73.996m`。
3. `RESET_EVERY=0` 完全不可用：ATE `651.5409`，FinalErr `942.865m`，Yaw `37.430deg`。这直接否定了“简单不 reset 就能获得长期轨迹优化”的路线。
4. 用户关于长期轨迹优化的判断仍然成立，但实现方式不能是延长同一个 fast-weight 生命周期；需要在默认 `reset=5` 的安全边界内做长期信息筛选，或把长期记忆和短期 conflict correction 分成两个不同状态。
5. 下一步应从 `WIN_09` 出发做 **long residual whitelist / long-term continuity gate**：保留能支撑长期轨迹的 positive / neutral residual，只让 update-conflict negative residual 在 reset 周期内短期生效，不能让未经筛选的 TTT state 跨更长生命周期累积。

## 29. update-conflict layer routing / layer boost

本节继续围绕当前最强三组 replay 线：

```text
base reference = V7_TRIREPLAY_12_c5_w0_conflictEnergy_g0030_pos035_neg012_neu100_SWKS3
ATE / Rot = 35.8712 / 6.6142
mechanism = chunk5, w0, tri_replay, risk=update_conflict_energy
positive_frac = 0.35
negative_frac = 0.12
neutral_lambda = 1.0
```

上一轮 `update_conflict_energy` audit 显示 chunk5 的冲突能量主要集中在 layer12/head0，因此本节测试两种用法：

1. **layer-only routing**：只在指定 TTT layer 上启用 tri-replay，其它层保持正常 replay。
2. **layer boost**：保留全层 base gamma，再对 layer12 做 conflict-cue gamma override。

### 29.1 工程变更

新增：

```text
TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS
--ttt_write_gradient_reversal_layer_gammas
```

语义：

- 当全局 `TTT_WRITE_GRADIENT_REVERSAL_GAMMA=0` 时，`LAYER:GAMMA` 表示只在列出的层启用 tri-replay；
- 当全局 gamma 大于 0 时，未列出的层使用全局 gamma，列出的层使用 layer override gamma；
- smoke 已验证：
  - layer-only：只有 layer12 启用；
  - boost：layer11/13 使用 base gamma `0.03`，layer12 使用 override `0.04`。

### 29.2 Layer-only routing 结果

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = tri_replay
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.0
TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE = update_conflict_energy
positive_frac = 0.35
negative_frac = 0.12
neutral_lambda = 1.0
```

| Run | Layer gamma map | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `TRIREPLAY_12` | all-layer base `0.030` | `35.8712` | `6.6142` | n/a | n/a | current best reference |
| `V7_TRIREPLAY_LR_01` | `12:0.030` | `36.6141` | `6.5047` | `92.4570` | `0.0082` | layer12-only 回退 |
| `V7_TRIREPLAY_LR_02` | `12:0.040` | `36.5623` | `6.5357` | `92.4563` | `0.0082` | layer12-only 略好于 0.03，但仍回退 |
| `V7_TRIREPLAY_LR_03` | `12:0.030,5:0.015,9:0.010` | `36.7493` | `6.4996` | `92.4583` | `0.0082` | 加 layer5/9 回退更大 |
| `V7_TRIREPLAY_LR_04` | `12:0.030,5:0.030,9:0.030` | `36.6495` | `6.4999` | `92.4568` | `0.0082` | 多层 hard routing 不成立 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_layer_routed/
```

关键段：

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `TRIREPLAY_12` | `35.8712` | `5.573` | `76.863` | `56.433` | `3.793` | `31.177167` |
| `LR_01` | `36.6141` | `5.383` | `78.106` | `57.235` | `3.600` | `31.370997` |
| `LR_02` | `36.5623` | `5.524` | `77.995` | `57.150` | `3.635` | `31.363610` |
| `LR_03` | `36.7493` | `5.413` | `78.074` | `57.215` | `3.590` | `31.377299` |
| `LR_04` | `36.6495` | `5.248` | `77.938` | `57.126` | `3.591` | `31.362599` |

结论：

1. `update_conflict_energy` 的 layer concentration 是有诊断价值的，但不能把 tri-replay 收缩成 layer-only。
2. TRIREPLAY 的收益依赖全层小幅正/中/负 replay 平衡；只打 layer12 会让 `[200,300)` 和全局 ATE 都回退。
3. layer-only 主要改善 Rot/Yaw/FinalErr，像姿态/endpoint regularizer，不是 ATE 主突破。

### 29.3 Layer12 boost 结果

修正 layer-gamma 语义后继续测试：未列出 layer 使用全局 base gamma，layer12 使用更高 override gamma。

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = tri_replay
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE = update_conflict_energy
positive_frac = 0.35
negative_frac = 0.12
neutral_lambda = 1.0
```

| Run | Base gamma | Layer gamma map | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| `TRIREPLAY_12` | `0.030` | none | `35.8712` | `6.6142` | n/a | n/a | current best reference |
| `V7_TRIREPLAY_BOOST_01` | `0.030` | `12:0.040` | `36.0822` | `6.5297` | `92.4504` | `0.0083` | boost 有效但未过 reference |
| `V7_TRIREPLAY_BOOST_02` | `0.030` | `12:0.050` | `36.0589` | `6.4810` | `92.4520` | `0.0082` | Rot 更好，ATE 仍回退 |
| `V7_TRIREPLAY_BOOST_03` | `0.030` | `5:0.020,9:0.020,12:0.040` | `36.0234` | `6.5496` | `92.4507` | `0.0083` | 本批第二好；降低 layer5/9 有一点帮助 |
| `V7_TRIREPLAY_BOOST_04` | `0.020` | `12:0.040` | `35.9908` | `6.4824` | `92.4505` | `0.0082` | 本批最好，但未过 `TRIREPLAY_12` |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_layer_boost/
```

关键段：

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `TRIREPLAY_12` | `35.8712` | `5.573` | `76.863` | `56.433` | `3.793` | `31.177167` |
| `BOOST_01` | `36.0822` | `5.301` | `77.278` | `56.631` | `3.649` | `31.302889` |
| `BOOST_02` | `36.0589` | `5.226` | `77.363` | `56.681` | `3.623` | `31.321189` |
| `BOOST_03` | `36.0234` | `5.577` | `77.318` | `56.652` | `3.675` | `31.307934` |
| `BOOST_04` | `35.9908` | `5.488` | `77.234` | `56.597` | `3.619` | `31.305691` |

29.3 结论：

1. layer12 boost 比 layer-only 明显更合理：所有 boost run 都比 B0 强，最好 `BOOST_04=35.9908`。
2. 但 boost 仍没有超过 `TRIREPLAY_12=35.8712`。`TRIREPLAY_12` 对 `[200,300)` 仍最好：`76.863m`；BOOST_04 退到 `77.234m`。
3. `BOOST_04` 的规律有价值：base gamma 降到 `0.020`、只在 layer12 提高到 `0.040`，比全层 `0.030 + layer12 boost` 更稳。这说明 update-conflict cue 可以用于 layer-weighted shaping，但不能替代全层 tri-replay。
4. 当前 v7 best 保持不变：

```text
V7_TRIREPLAY_12_c5_w0_conflictEnergy_g0030_pos035_neg012_neu100_SWKS3
ATE / Rot = 35.8712 / 6.6142
```

下一步不建议继续做 layer scalar 小扫。更有价值的方向是把 `update_conflict_energy` 正式变成 per-head replay routing：优先验证 layer12/head0 的局部冲突是否能作为 `G_neg`，同时保留其它 head/layer 的 `G_pos/G_neu`，而不是整层一起加权。

---

## 28. 当前最新状态索引

最新完成的是第 27 节：

```text
TRIREPLAY_06 窄扫与 update_conflict_energy TTT 自有 cue 诊断
```

当前 v7 best 已更新为：

```text
V7_TRIREPLAY_12_c5_w0_conflictEnergy_g0030_pos035_neg012_neu100_SWKS3
ATE / Rot = 35.8712 / 6.6142
```

核心结论：

1. `TRIREPLAY_12` 比 `TRIREPLAY_06=35.8961 / 6.5946` 进一步改善 ATE `0.0249m`，但 Rot 略回退。
2. `update_conflict_energy` 已正式作为 TTT 自有 cue 输出 per-layer / per-head 诊断。
3. chunk5 的 self-cue 风险高度集中在 `layer 12 / head 0`：

```text
layer12 risk_mean = 0.020680
layer12 head0 risk_mean = 0.028733
layer12 head0 energy_mean = 137.657547
```

4. 下一步不应继续做全层统一 `gamma / neg_frac` 小扫，而应做 **layer/head routed tri-replay**：

```text
layer12/head0:
    stronger update_conflict_energy negative replay

layer5/layer9:
    weak negative replay 或 diagnostic routing

其它 low-risk layers/heads:
    保留 positive + neutral continuity
```

---

## 27. TRIREPLAY_06 窄扫与 update-conflict TTT 自有 cue 诊断

本节继续围绕 26 节的当前结构性 best：

```text
V7_TRIREPLAY_06_c5_w0_conflictEnergy_pos035_neg015_neu100_SWKS3
ATE / Rot = 35.8961 / 6.5946
```

用户要求继续窄扫：

```text
neg_frac = 0.12 / 0.18 / 0.20
gamma    = 0.020 / 0.030
risk source = update_conflict_energy
```

并把 `update_conflict_energy` 正式作为 TTT 自有 cue 做 per-layer / per-head 诊断。

### 27.1 工程补充

`ttt_write_controller.py`：

- `update_conflict_energy` cue 现在输出 per-layer 统计：
  - `energy_mean / energy_p90`
  - `risk_mean / risk_p90`
  - `cos_mean`
  - `negative_cos_mass`
- 同时输出 per-head 统计：
  - head-level `risk_mean / energy_mean / cos_mean`
  - top-risk head index 与对应 risk / energy / cos

新增诊断工具：

```text
tools/ttt_update_conflict_audit.py
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/update_conflict_cue_audit_gamma_neg_sweep/
```

输出文件：

```text
update_conflict_layers_raw.csv
update_conflict_heads_raw.csv
update_conflict_layers_summary.csv
update_conflict_heads_summary.csv
update_conflict_summary.md
```

验证：

```text
python -m py_compile loger/pipeline/ttt_write_controller.py loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py
python -m py_compile tools/ttt_update_conflict_audit.py
bash -n tools/run_attention_cue_experiment.sh
```

结果：通过。

### 27.2 固定协议

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
PRIOR_BRANCH_MASK = 0
PRIOR_LAYER_MODE = all
TTT_WRITE_GRADIENT_REVERSAL_MODE = tri_replay
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE = update_conflict_energy
TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC = 0.35
TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA = 1.00
RESET_EVERY = 5
SWA = SWKS3-style fixed protocol
```

SWA fixed protocol：

```text
ENABLE_SWA_WRITE_CONTROL = 1
SWA_WRITE_KEEP_SCOPE = both_overlap
SWA_WRITE_LAYER_MODE = last
SWA_WRITE_SCORE_SOURCE = read
ENABLE_SWA_OVERLAP_SOURCE_REPLACE = 1
SWA_OVERLAP_SOURCE_REPLACE_MODE = source
SWA_OVERLAP_SOURCE_REPLACE_TARGET = kv
SWA_OVERLAP_SOURCE_REPLACE_ALPHA = 0.50
SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE = last
```

### 27.3 运行记录

6 个 full KITTI01 run 并发完成，GPU 使用 `0/1/2/3/5/6`。

| Run | Gamma | Neg frac | Start | Done | Walltime |
|---|---:|---:|---|---|---:|
| `V7_TRIREPLAY_09_c5_w0_conflictEnergy_g0020_pos035_neg012_neu100_SWKS3` | `0.020` | `0.12` | `2026-05-09 07:37:52` | `08:03:14` | `25.4 min` |
| `V7_TRIREPLAY_10_c5_w0_conflictEnergy_g0020_pos035_neg018_neu100_SWKS3` | `0.020` | `0.18` | `2026-05-09 07:37:52` | `08:03:58` | `26.1 min` |
| `V7_TRIREPLAY_11_c5_w0_conflictEnergy_g0020_pos035_neg020_neu100_SWKS3` | `0.020` | `0.20` | `2026-05-09 07:37:52` | `08:04:16` | `26.4 min` |
| `V7_TRIREPLAY_12_c5_w0_conflictEnergy_g0030_pos035_neg012_neu100_SWKS3` | `0.030` | `0.12` | `2026-05-09 07:37:52` | `08:04:07` | `26.3 min` |
| `V7_TRIREPLAY_13_c5_w0_conflictEnergy_g0030_pos035_neg018_neu100_SWKS3` | `0.030` | `0.18` | `2026-05-09 07:37:52` | `08:05:09` | `27.3 min` |
| `V7_TRIREPLAY_14_c5_w0_conflictEnergy_g0030_pos035_neg020_neu100_SWKS3` | `0.030` | `0.20` | `2026-05-09 07:37:52` | `08:03:14` | `25.4 min` |

### 27.4 Global metrics

| Run | Gamma | Neg frac | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | n/a | n/a | `36.2957` | `6.6182` | `92.4428` | `0.0082` | old localized TTGR best |
| `TRIREPLAY_06` | `0.025` | `0.15` | `35.8961` | `6.5946` | `92.4381` | `0.0082` | previous tri-replay best |
| `TRIREPLAY_09` | `0.020` | `0.12` | `35.9098` | `6.6212` | `92.4392` | `0.0082` | 接近 TRIREPLAY_06，但未过 |
| `TRIREPLAY_10` | `0.020` | `0.18` | `35.9294` | `6.5712` | `92.4394` | `0.0082` | Rot 较好，ATE 回退 |
| `TRIREPLAY_11` | `0.020` | `0.20` | `35.9090` | `6.6204` | `92.4391` | `0.0082` | 接近但未过 |
| `TRIREPLAY_12` | `0.030` | `0.12` | `35.8712` | `6.6142` | `92.4389` | `0.0082` | **当前 v7 新 best** |
| `TRIREPLAY_13` | `0.030` | `0.18` | `35.9488` | `6.6331` | `92.4392` | `0.0082` | neg 过多，回退 |
| `TRIREPLAY_14` | `0.030` | `0.20` | `35.9882` | `6.5816` | `92.4393` | `0.0082` | Rot 较好但 ATE 明显回退 |

相对改善：

```text
TRIREPLAY_12 vs TRIREPLAY_06:
    ATE: -0.0249m
    Rot: +0.0196deg

TRIREPLAY_12 vs TTGRL_04:
    ATE: -0.4245m
    Rot: -0.0041deg

TRIREPLAY_12 vs B0:
    ATE: -0.5449m
```

仍未达到最终目标：

```text
KITTI01 ATE < 30m
当前 best = 35.8712m
```

### 27.5 Trajectory diagnostics

输出目录：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_tri_replay_gamma_neg_sweep/
```

| Run | ATE RMSE | FinalErr | 50f worst | 100f `[200,300)` | 200f `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `[200,250)=78.272` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `[200,250)=77.914` | `77.644` | `56.968` | `3.769` | `31.219364` |
| `TRIREPLAY_06` | `35.8961` | `5.479` | `[250,300)=77.003` | `76.925` | `56.482` | `3.773` | `31.165145` |
| `TRIREPLAY_09` | `35.9098` | `6.174` | `[250,300)=76.892` | `76.845` | `56.417` | `3.809` | `31.176359` |
| `TRIREPLAY_10` | `35.9294` | `5.858` | `[250,300)=77.075` | `77.002` | `56.529` | `3.769` | `31.180984` |
| `TRIREPLAY_11` | `35.9090` | `5.750` | `[250,300)=77.175` | `77.028` | `56.554` | `3.808` | `31.173823` |
| `TRIREPLAY_12` | `35.8712` | `5.573` | `[250,300)=76.967` | `76.863` | `56.433` | `3.793` | `31.177167` |
| `TRIREPLAY_13` | `35.9488` | `5.978` | `[250,300)=77.030` | `76.990` | `56.536` | `3.805` | `31.171230` |
| `TRIREPLAY_14` | `35.9882` | `5.515` | `[250,300)=77.200` | `77.132` | `56.631` | `3.769` | `31.179485` |

Focus run `TRIREPLAY_12` chunk diagnostics：

| Rank | Chunk | Frame range | RMSE | 结论 |
|---:|---:|---|---:|---|
| 1 | `8` | `[232,264)` | `87.7440` | 仍是主病灶核心 |
| 2 | `9` | `[261,293)` | `75.5270` | 高误差延续 |
| 3 | `7` | `[203,235)` | `73.4274` | 病灶前半段 |
| 4 | `15` | `[435,467)` | `60.0390` | 后段次病灶 |
| 5 | `16` | `[464,496)` | `59.5291` | 后段次病灶 |
| 6 | `6` | `[174,206)` | `52.7968` | 入口仍有误差 |
| 10 | `10` | `[290,322)` | `39.3231` | 病灶出口 |

### 27.6 update-conflict-energy self-cue audit

本批将 `update_conflict_energy` 正式作为 TTT 自有 cue 输出并汇总。审计聚焦 chunk5。

审计输出：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/update_conflict_cue_audit_gamma_neg_sweep/update_conflict_summary.md
```

记录数：

| Item | Count |
|---|---:|
| Layer records | `108` |
| Head records | `216` |

由于这 6 个 run 使用相同输入 state 和相同 risk source，cue 本身不随 `gamma / neg_frac` 改变；不同 run 的 per-layer / per-head cue 数值一致，这是预期结果。

Top layers by risk：

| Layer | Risk mean | Energy mean | Cos mean | Negative cos mass | 解释 |
|---:|---:|---:|---:|---:|---|
| `12` | `0.020680` | `95.897720` | `0.604961` | `0.000000` | chunk5 最强 conflict-energy 层 |
| `5` | `0.006077` | `20.799225` | `0.601344` | `0.001314` | 次强风险层 |
| `9` | `0.005017` | `17.790094` | `0.663874` | `0.000000` | 中后层弱风险 |
| `1` | `0.004298` | `36.740490` | `0.766374` | `0.000360` | 能量高但方向较同向 |
| `11` | `0.004095` | `10.760206` | `0.678233` | `0.001848` | 弱风险层 |

Top heads by risk：

| Layer | Head | Risk mean | Energy mean | Cos mean | 解释 |
|---:|---:|---:|---:|---:|---|
| `12` | `0` | `0.028733` | `137.657547` | `0.697410` | 最强风险 head |
| `12` | `1` | `0.012626` | `54.137917` | `0.512513` | 同层第二 head |
| `5` | `0` | `0.011799` | `39.632828` | `0.680653` | layer5 主风险 head |
| `9` | `1` | `0.009454` | `33.237492` | `0.644911` | layer9 主风险 head |
| `1` | `0` | `0.007298` | `68.660889` | `0.835582` | 高能量但冲突弱 |

### 27.7 结论

1. 本节当前 v7 best 更新为：

```text
V7_TRIREPLAY_12_c5_w0_conflictEnergy_g0030_pos035_neg012_neu100_SWKS3
ATE / Rot = 35.8712 / 6.6142
```

2. `TRIREPLAY_12` 比 `TRIREPLAY_06` 只好 `0.0249m`，但它确认了局部 sweet spot：`gamma=0.030` 可以更强一点，前提是 `neg_frac` 收窄到 `0.12`。
3. `neg_frac=0.18/0.20` 在两个 gamma 下都回退，说明 negative set 不能继续扩大；有害方向更像少量高 conflict-energy layer/head，而不是全 chunk 大面积负样本。
4. `update_conflict_energy` 已经不仅是一个 write source，而是可审计的 TTT 自有 cue。chunk5 风险高度集中在 **layer 12，尤其 head 0**。
5. 当前最有价值的下一步不是继续扫 scalar `gamma / neg_frac`，而是把三组 replay 做成 **layer/head routed objective**：

```text
low-risk layers:
    保持 positive + neutral continuity，少用或不用 negative replay

layer 12 / head 0:
    使用较强 update_conflict_energy negative replay

layer 5 / layer 9:
    使用较弱 negative replay 或只做 diagnostic routing
```

6. 这也解释了为什么全层统一 tri-replay 只带来 `35.9m` 平台：真正 harmful cue 已经很集中，但当前控制仍按全层/全 head 同一规则执行，容易把 continuity layer 一起扰动。

### 27.8 下一步计划

优先级：

1. 实现 layer-routed tri-replay：
   - `layer12 only`
   - `layer12 + layer5 + layer9`
   - high-risk layer 用 `gamma=0.030, neg_frac=0.12`
   - 其它层退回 `TTGRL_04` 或 neutral continuity。
2. 若 layer-routed 有正信号，再实现 head-routed tri-replay：
   - layer12 head0 使用 stronger negative replay；
   - layer12 head1 弱化或只保留 neutral；
   - 其它 layer/head 保持 continuity。
3. 若 layer/head routing 仍卡在 `35.8-36.0m`，说明 TTT write 方向已经被 chunk5 local objective 榨得差不多，需要重新看 `[200,300)` 的 read-side / pose-scale failure，而不是继续做 scalar TTT 写入小扫。

---

## 26. 结构性 TTT：三组 replay objective

### 26.1 动机与实现

上一节确认 two-replay 和 short negative TTL 都没有超过 `TTGRL_04`，但也给出一个明确方向：不要再把所有 token 放在同一个 signed prior 里连续插值，而是把 replay update 拆成三组：

```text
positive  = high-continuity / low-risk token
neutral   = middle-risk token
negative  = high-risk token

W_commit = W_old
         + (W_pos - W_old)
         + lambda_neu * (W_neu - W_old)
         - gamma * (W_neg - W_old)
```

本轮新增：

- `TTT_WRITE_GRADIENT_REVERSAL_MODE=tri_replay`
- `TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC`
- `TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC`
- `TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA`

固定结构：

```text
target chunk = 5
branch = w0
layer = all
gamma = 0.025
SWA = SWKS3-style fixed protocol
positive write score = stage_d_x_dg_inv_sqrt
```

smoke：

| Run | END_FRAME | Risk source | Pos / Neu / Neg | 结论 |
|---|---:|---|---|---|
| `V7_TRIREPLAY_SMOKE_c5_w0_pos035_neg015_neu100_e224` | `224` | `prior` | `0.35 / 0.50 / 0.15` | 通过；chunk5 debug 显示 `ttt_tri_replay_applied=True`、active branch `[0]` |

说明：freeze 仍然只作为 causal diagnostic，本节不是 freeze 类策略；它保留 chunk5 commit，只改 replay objective 的正/中/负证据组成。

### 26.2 第一批：prior risk vs update-conflict-energy

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TRIREPLAY_01_c5_w0_prior_pos035_neg015_neu100_SWKS3` | `2026-05-09 06:05:22` | `06:29:32` | `24.2 min` |
| `V7_TRIREPLAY_02_c5_w0_prior_pos025_neg015_neu100_SWKS3` | `2026-05-09 06:05:22` | `06:29:05` | `23.7 min` |
| `V7_TRIREPLAY_03_c5_w0_prior_pos035_neg015_neu075_SWKS3` | `2026-05-09 06:05:22` | `06:28:45` | `23.4 min` |
| `V7_TRIREPLAY_04_c5_w0_conflictEnergy_pos035_neg010_neu100_SWKS3` | `2026-05-09 06:05:22` | `06:30:00` | `24.6 min` |

Global metrics：

| Run | Risk source | Pos | Neg | Neu lambda | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | baseline | - | - | - | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | continuous low-prior | - | - | - | `36.2957` | `6.6182` | `92.4428` | `0.0082` | previous v7 best |
| `TRIREPLAY_01` | `prior` | `0.35` | `0.15` | `1.00` | `36.4193` | `6.6420` | `92.4466` | `0.0082` | 回退 |
| `TRIREPLAY_02` | `prior` | `0.25` | `0.15` | `1.00` | `36.3890` | `6.6705` | `92.4464` | `0.0082` | 未过 TTGRL |
| `TRIREPLAY_03` | `prior` | `0.35` | `0.15` | `0.75` | `36.3702` | `6.6734` | `92.4442` | `0.0082` | prior-risk best，但未过 |
| `TRIREPLAY_04` | `update_conflict_energy` | `0.35` | `0.10` | `1.00` | `35.9632` | `6.6376` | `92.4403` | `0.0082` | **显著新 best** |

第一批结论：

1. 三组 replay 的收益主要来自 **TTT 自身的 update-conflict-energy cue**，不是来自原来的 prior risk。
2. `TRIREPLAY_04` 首次把 v7 ATE 明显拉到 `36m` 以下：相比 `TTGRL_04` 改善 `0.3326m`，相比 B0 改善 `0.4529m`。
3. 这说明用户提出的 “挖掘 TTT 自己的 cue” 是对的：TTT update 内部的 conflict/energy 比外部 Dg-prior 更适合作 negative replay 分组。

### 26.3 第二批：update-conflict-energy 参数小矩阵

固定：

```text
risk source = update_conflict_energy
target chunk = 5
branch = w0
layer = all
gamma = 0.025
```

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TRIREPLAY_05_c5_w0_conflictEnergy_pos035_neg005_neu100_SWKS3` | `2026-05-09 06:32:26` | `06:57:08` | `24.7 min` |
| `V7_TRIREPLAY_06_c5_w0_conflictEnergy_pos035_neg015_neu100_SWKS3` | `2026-05-09 06:32:26` | `06:57:27` | `25.0 min` |
| `V7_TRIREPLAY_07_c5_w0_conflictEnergy_pos025_neg010_neu100_SWKS3` | `2026-05-09 06:32:26` | `06:57:08` | `24.7 min` |
| `V7_TRIREPLAY_08_c5_w0_conflictEnergy_pos035_neg010_neu075_SWKS3` | `2026-05-09 06:32:26` | `06:57:08` | `24.7 min` |

Global metrics：

| Run | Pos | Neg | Neu lambda | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `TRIREPLAY_04` | `0.35` | `0.10` | `1.00` | `35.9632` | `6.6376` | `92.4403` | `0.0082` | 第一批 best |
| `TRIREPLAY_05` | `0.35` | `0.05` | `1.00` | `36.0316` | `6.6681` | `92.4400` | `0.0082` | neg 太少，回退 |
| `TRIREPLAY_06` | `0.35` | `0.15` | `1.00` | `35.8961` | `6.5946` | `92.4381` | `0.0082` | **当前 v7 best** |
| `TRIREPLAY_07` | `0.25` | `0.10` | `1.00` | `36.0466` | `6.6552` | `92.4408` | `0.0082` | pos 太少，回退 |
| `TRIREPLAY_08` | `0.35` | `0.10` | `0.75` | `36.0219` | `6.6347` | `92.4388` | `0.0082` | neutral 衰减回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_tri_replay_objective/trajectory_diagnostics_tri_replay_refine/
```

| Run | ATE RMSE | FinalErr | 50f mean / worst | 100f mean / worst | 200f mean / worst | `[200,300)` | `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `77.831` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `29.632 / 77.914` | `30.275 / 77.644` | `30.314 / 56.968` | `77.644` | `47.582` | `3.769` | `31.219364` |
| `TRIREPLAY_04` | `35.9632` | `6.148` | `29.333 / 76.849` | `29.976 / 76.837` | `29.943 / 56.433` | `76.837` | `47.102` | `3.838` | `31.185695` |
| `TRIREPLAY_05` | `36.0316` | `6.152` | `29.420 / 77.129` | `30.062 / 76.998` | `30.026 / 56.537` | `76.998` | n/a | `3.847` | `31.180274` |
| `TRIREPLAY_06` | `35.8961` | `5.479` | `29.267 / 77.003` | `29.909 / 76.925` | `29.910 / 56.482` | `76.925` | `46.857` | `3.773` | `31.165145` |
| `TRIREPLAY_07` | `36.0466` | `6.009` | `29.452 / 77.107` | `30.095 / 76.993` | `30.064 / 56.539` | `76.993` | n/a | `3.837` | `31.191433` |
| `TRIREPLAY_08` | `36.0219` | `5.388` | `29.387 / 77.362` | `30.026 / 77.345` | `30.048 / 56.769` | `77.345` | n/a | `3.816` | `31.175059` |

### 26.4 结论

1. 本节找到当前 v7 最强结构性 TTT 策略：

```text
V7_TRIREPLAY_06_c5_w0_conflictEnergy_pos035_neg015_neu100_SWKS3
ATE / Rot = 35.8961 / 6.5946
```

2. 相比 `TTGRL_04 = 36.2957 / 6.6182`：

```text
ATE 改善 = 0.3997m
Rot 改善 = 0.0236deg
[200,300) 改善 = 0.719m
[400,600) 改善 = 0.725m
```

3. 相比 `B0_SWKS3 = 36.4161 / 6.6128`：

```text
ATE 改善 = 0.5200m
[200,300) 改善 = 0.906m
```

4. 这不是 freeze 型解法，也不是把 chunk5 state 删除；它保留 chunk5 commit，并把 TTT replay objective 显式拆成 positive / neutral / negative 三组。
5. 最关键的机制发现：`update_conflict_energy` 比原来的 Dg/prior risk 更能识别 harmful negative replay。prior-risk 三组 replay 全部卡在 `36.37-36.42m`，而 conflict-energy 直接进入 `35.9m`。
6. `neg_frac=0.15` 优于 `0.10/0.05`，`pos_frac=0.35` 优于 `0.25`，`neutral_lambda=1.0` 优于 `0.75`。也就是说当前 best 不是少写 neutral，而是保留中性连续性，同时对足够多的 high-conflict token 做负证据。
7. 仍未达到最终目标 `KITTI01 ATE < 30m`，但这是 v7 目前最强的结构性突破，已经明显跳出 `36.3m` 平台。

下一步建议：

```text
围绕 TRIREPLAY_06 做更窄的结构追击：
    1. risk source = update_conflict_energy
    2. chunk = 5
    3. branch = w0
    4. pos_frac = 0.35
    5. neg_frac = 0.12 / 0.18 / 0.20
    6. gamma = 0.020 / 0.030

同时考虑把 update_conflict_energy 作为 TTT 自有 cue 输出成 per-layer / per-head diagnostic，
因为它已经比外部 semantic / Dg scalar 更接近真正 harmful write source。
```

---

## 18. 结构性 TTT：two-replay 与 short negative lifetime

前面 freeze 只作为诊断使用，不能当策略。它证明 chunk5 / chunk6 对 `[200,300)` 有强因果关系，但 hard freeze 会把后段连续性一起删掉。因此本节进入真正结构性 TTT：把 short negative evidence 和 long positive continuity 分开，而不是继续只调单个 scalar prior / gamma。

本轮先尝试两条结构路径：

1. **strict dual fast-weight lifetime**：long state 只提交 positive continuity，short negative delta 只在后续若干 chunk 临时 apply；
2. **two-replay objective / short negative TTL**：正向 replay 与负向 replay 分离，或让 chunk5 负证据先短期生效，再按 TTL 从长期 fast weight 中扣除。

### 18.1 strict dual-lifetime smoke

实现侧新增了 strict dual fast-weight 相关 hook：

```text
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE = dual_lifetime / dual_fast_weight
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE
```

但两次 short smoke 都在 chunk5 附近异常变慢 / 停滞：

| Run | 结果 | 处理 |
|---|---|---|
| `V7_TTDL_TRUE_SMOKE_c5_w0_g0025_ttl4_long050_e224` | 长时间无 `hmc_state_hash.jsonl` 更新 | 手动停止，不计结果 |
| `V7_TTDL_TRUE_SMOKE2_c5_w0_g0025_ttl4_long050_e177` | 仍然异常慢 / 不稳定 | 手动停止，不计结果 |

结论：

1. strict dual-lifetime 是正确方向，但当前实现路径还不稳定，不能作为 full KITTI01 结果。
2. 后续如果继续 dual fast-weight，需要先把 long-only probe 与 short-apply 路径重构到更轻、更可验证的实现，而不是继续直接跑 full。

### 18.2 two-replay objective

`two_replay` 机制已跑通：正向 replay 使用原 write prior，负向 replay 使用 risk prior，然后按 active branch 做：

```text
W_commit = W_pos - gamma * (W_neg - W_old)
```

smoke：

```text
V7_TT2R_SMOKE_c5_w0_g0025_e177
```

debug 确认：

```text
ttt_two_replay_applied = True
active_branches = [0]
mode = two_replay
chunk5 gate on
gamma = 0.025
```

固定 full 协议：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = two_replay
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
SWA = SWKS3-style fixed protocol
semantic = disabled
```

Global metrics：

| Run | Negative risk source | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | baseline | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | low-prior signed replay | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `V7_TT2R_01_c5_w0_prior_g0025_SWKS3` | `prior` | `36.5046` | `6.6219` | `92.4450` | `0.0082` | 回退 |
| `V7_TT2R_02_c5_w0_updateConflict_g0025_SWKS3` | `update_conflict` | `36.4593` | `6.6373` | `92.4449` | `0.0082` | 回退 |
| `V7_TT2R_03_c5_w0_conflictEnergy_g0025_SWKS3` | `update_conflict_energy` | `36.4560` | `6.5800` | `92.4434` | `0.0081` | two-replay best，但未过 |
| `V7_TT2R_04_c5_w0_residualDg_g0025_SWKS3` | `ttt_residual_x_dg` | `36.5463` | `6.6213` | `92.4437` | `0.0082` | 回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_two_replay_structural/trajectory_diagnostics_two_replay/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `3.769` | `31.219364` |
| `TT2R_01` | `36.5046` | `5.614` | `78.349` | `77.989` | `57.216` | `3.775` | `31.241065` |
| `TT2R_02` | `36.4593` | `5.737` | `78.291` | `77.879` | `57.135` | `3.796` | `31.233062` |
| `TT2R_03` | `36.4560` | `5.305` | `78.232` | `77.852` | `57.126` | `3.737` | `31.221104` |
| `TT2R_04` | `36.5463` | `5.736` | `78.389` | `77.947` | `57.198` | `3.776` | `31.220430` |

18.2 结论：

1. two-replay 工程路径成立，但首批没有达到 `<30m`，也没有超过 `TTGRL_04`。
2. `update_conflict_energy` 是 two-replay 内部最好风险源，Rot / FinalErr 有一点信号，但 `[200,300)` 和 overall ATE 都不如原来的 continuous low-prior TTGR。
3. 这说明当前 harmful update 不是一个可以被单独 `G_neg` 轻松剥离的独立 replay；原来的 signed low-prior 更像是在整体转动 chunk5 的 `w0` update direction。

### 18.3 short negative TTL

为避免 strict dual-lifetime 的不稳定，本批采用更稳的 transient delta 机制：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE = ttl_delta
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK = 0
TTT_WRITE_TRANSIENT_DELTA_SUBTRACT_SCALE = 1.0
TTT_WRITE_TRANSIENT_DELTA_BRANCH_MASK = 0
```

含义：

- chunk5 的 negative TTGR 先正常提交，让它能影响后续 chunk；
- 之后按 `TTL` 在指定 chunk commit 时从长期 fast weight 里扣除这份 negative delta；
- 这是 “用后提交 / 短期负证据” 的稳定近似，不是 freeze。

运行记录：

| Run | TTL | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V7_TTGRTTL_01_c5_w0g0025_ttl1_sub100_SWKS3` | `1` | `2026-05-09 05:26:50` | `05:49:45` | `22.9 min` |
| `V7_TTGRTTL_02_c5_w0g0025_ttl2_sub100_SWKS3` | `2` | `2026-05-09 05:26:50` | `05:49:37` | `22.8 min` |
| `V7_TTGRTTL_03_c5_w0g0025_ttl3_sub100_SWKS3` | `3` | `2026-05-09 05:26:50` | `05:50:47` | `24.0 min` |
| `V7_TTGRTTL_04_c5_w0g0025_ttl4_sub100_SWKS3` | `4` | `2026-05-09 05:26:50` | `05:49:52` | `23.0 min` |

Global metrics：

| Run | TTL | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | persistent within reset | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRTTL_01` | `1` | `36.4587` | `6.6166` | `92.4436` | `0.0082` | 太早扣除，回退 |
| `TTGRTTL_02` | `2` | `36.4256` | `6.6056` | `92.4446` | `0.0082` | 接近 B0，但未过 |
| `TTGRTTL_03` | `3` | `36.3475` | `6.6616` | `92.4444` | `0.0082` | TTL best，但未过 TTGRL_04 |
| `TTGRTTL_04` | `4` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | 与 TTGRL_04 完全重合 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_short_negative_ttl/trajectory_diagnostics_short_negative_ttl/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRTTL_01` | `36.4587` | `5.447` | `78.191` | `77.883` | `57.141` | `47.913` | `3.765` | `31.224139` |
| `TTGRTTL_02` | `36.4256` | `5.679` | `78.160` | `77.881` | `57.141` | `47.832` | `3.768` | `31.235610` |
| `TTGRTTL_03` | `36.3475` | `5.914` | `78.064` | `77.817` | `57.097` | `47.584` | `3.810` | `31.235191` |
| `TTGRTTL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |

18.3 结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04`。
2. TTL 机制提供了清楚的 lifetime 结论：chunk5 negative evidence 不能太早从长期 fast weight 中移除；TTL=1/2 基本吃掉收益，TTL=3 只保留一部分，TTL=4 与 `TTGRL_04` 完全重合。
3. 因为 `RESET_EVERY=5`，TTL=4 等价于让 chunk5 的 negative correction 活到当前 reset 周期结束。这说明当前 best 依赖一个短 reset 周期内的持续 `w0` update direction shift，而不是 one-hop negative evidence。
4. 这也解释了为什么 freeze 不能当解药：有效的不是删掉 chunk5，而是在 chunk5 之后的整个 reset window 内保留一小段方向校正。

### 18.4 当前判断

截至本节，真正结构性 TTT 的第一轮结果是：

| Mechanism | 是否跑通 | 是否超过 `TTGRL_04` | 主要结论 |
|---|---|---|---|
| strict dual fast-weight lifetime | smoke 不稳定 | 未计入 | 方向仍有价值，但实现需要重构 |
| two-replay objective | 跑通 | 否 | 独立 `G_neg` 没有比 signed low-prior 更好 |
| short negative TTL | 跑通 | 否 | chunk5 负证据需要活到 reset 结束，过早撤掉会回退 |

当前 v7 best 仍保持：

```text
V7_TTGRL_04_c5_w0all_g0025_SWKS3
ATE / Rot = 36.2957 / 6.6182
```

下一步不应再扩大 TTL / scalar 小扫。更可能有价值的方向：

1. 修复 strict dual-lifetime 的实现，让 long positive 与 short negative 真正成为两个可独立 apply / commit 的 state；
2. 改 two-replay objective 的构造，不再是 `all positive + risk negative`，而是显式三组：

```text
G_commit = G_pos + lambda_neu * G_neu - gamma * G_neg
```

其中 `G_pos` 只来自 high static / high continuity token，`G_neg` 来自 update-conflict high-risk token，`G_neu` 保留普通 continuity；
3. 继续挖 TTT 自己的 cue，但重点从 residual magnitude 转向 **native-vs-controlled update conflict direction**，因为 residual / Dg / semantic scalar 都没能稳定分出 harmful update。

## 24. 结构性 TTT：chunk6/7 positive continuity scale

用户明确提醒：

```text
freeze 只能当诊断用，不能当解药。
```

因此本批不再把 `freeze` 当候选策略，而是在当前 best `TTGRL_04` 基础上做一个更温和的结构诊断：

- chunk5 仍保留轻量 negative evidence：`w0 / all-layer / low_prior TTGR / gamma=0.025`；
- chunk6/7 不 freeze；
- 只用已有 post-replay scale hook 调整 chunk6/7 的 positive continuity commit 幅度；
- 目标是看能否保留 chunk5 负证据收益，同时避免 hard freeze 的后段错误转移。

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
SWA = SWKS3-style fixed protocol
semantic = disabled
```

候选：

| Run | Positive continuity scale | 目的 |
|---|---|---|
| `V7_TTPOSCONT_01_c5ttgr_c6scale110_SWKS3` | chunk6 `1.10` | chunk6 正向 continuity 轻微外推 |
| `V7_TTPOSCONT_02_c5ttgr_c6scale125_SWKS3` | chunk6 `1.25` | chunk6 更强外推 |
| `V7_TTPOSCONT_03_c5ttgr_c6scale090_SWKS3` | chunk6 `0.90` | 检查 chunk6 commit 是否过强 |
| `V7_TTPOSCONT_04_c5ttgr_c6c7scale110_SWKS3` | chunk6 `1.10`, chunk7 `1.10` | 连续两个 chunk 温和正向外推 |

运行说明：

- 4 条 full KITTI01 run 都正常写出完整 `01.txt` 和 38 个 HMC chunk debug；
- 本批 wrapper 在中断后没有自动落 `results_sim3/`，已用绝对路径手动重跑 `kitti_benchmark`；
- trajectory diagnostics 输出：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_pos_continuity/trajectory_diagnostics_pos_continuity/
```

Global metrics：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTPOSCONT_01` | `36.3745` | `6.6380` | `92.4447` | `0.0082` | chunk6 `1.10`，回退 |
| `TTPOSCONT_02` | `36.3654` | `6.6332` | `92.4444` | `0.0082` | chunk6 `1.25`，略好于 1.10 但未过 best |
| `TTPOSCONT_03` | `36.4348` | `6.6331` | `92.4431` | `0.0082` | chunk6 `0.90`，回退 |
| `TTPOSCONT_04` | `36.3025` | `6.5930` | `92.4428` | `0.0081` | 本批 best，Rot 好于 TTGRL_04，但 ATE 仍略差 |

Trajectory diagnostics：

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `POSCONT_01` | `36.3745` | `5.404` | `78.104` | `77.806` | `57.084` | `47.637` | `3.780` | `31.233953` |
| `POSCONT_02` | `36.3654` | `5.635` | `78.037` | `77.791` | `57.074` | `47.647` | `3.782` | `31.233839` |
| `POSCONT_03` | `36.4348` | `5.585` | `78.081` | `77.788` | `57.088` | `47.933` | `3.788` | `31.220205` |
| `POSCONT_04` | `36.3025` | `5.684` | `77.945` | `77.679` | `56.999` | `47.636` | `3.752` | `31.214513` |

Focus chunk diagnostics：

| Run | c6 `[174,206)` | c7 `[203,235)` | c8 `[232,264)` | c9 `[261,293)` | c10 `[290,322)` | c15 `[435,467)` | c16 `[464,496)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `53.799` | `74.942` | `88.814` | `76.800` | `37.576` | `61.503` | `60.853` |
| `TTGRL_04` | `53.381` | `74.567` | `88.555` | `76.806` | `37.878` | `61.269` | `60.647` |
| `POSCONT_01` | `53.536` | `74.743` | `88.769` | `76.910` | `37.946` | `61.290` | `60.671` |
| `POSCONT_02` | `53.444` | `74.663` | `88.747` | `76.963` | `38.022` | `61.223` | `60.612` |
| `POSCONT_03` | `53.563` | `74.743` | `88.709` | `76.917` | `37.960` | `61.645` | `61.074` |
| `POSCONT_04` | `53.438` | `74.588` | `88.597` | `76.843` | `37.907` | `61.327` | `60.736` |

24 节结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04=36.2957`。
2. `POSCONT_04=36.3025 / 6.5930` 是本批最好，ATE 只比 `TTGRL_04` 差 `0.0068m`，Rot 略好。这说明温和 positive continuity scale 不会像 hard freeze 那样把后段打崩。
3. 但 `POSCONT_04` 的 `[200,300)=77.679`、`[400,600)=47.636` 都仍略差于 `TTGRL_04`，所以它不是新主线。
4. chunk6 单独 scale 无论 `0.90/1.10/1.25` 都不能超过 TTGRL，说明 chunk6 continuity 不是简单幅度问题。
5. 这批进一步确认：**freeze 只能用于定位，不能用于策略**；温和正向 scale 虽然安全得多，但仍只是 scalar post-replay 调参，没有真正分离 short harmful evidence 与 long useful continuity。
6. 下一步如果继续结构性 TTT，应实现真正的双生命周期或 two-replay objective：
   - long-term fast weight 只接收 positive / continuity update；
   - short-term fast weight 接收 chunk5 negative evidence，只服务后续少数 chunk；
   - commit 时不要把 short negative evidence 永久写进 long state。

---

## 23. 结构性 TTT：chunk5 use-after / downstream commit freeze

用户要求继续尝试更结构化的 TTT 机制。13/19/20 节已经证明 `chunk5` 的 one-hop / TTL transient delta 不能简单扣除，因为 `W_controlled - W_ref` 里混有 useful continuity。本节先做一个不新增大代码的结构诊断：

```text
保留当前 best:
    chunk5 / w0 / all-layer / low-prior TTGR gamma=0.025

额外冻结后续 chunk 的 TTT commit:
    freeze chunk6 / chunk7 / chunk8 / chunk6+7
```

直觉：

- 如果 chunk5 负证据本身有效，但后续某个 chunk 的 commit 把污染重新放大，那么冻结对应下游 commit 应该能保留 `[200,300)` 改善且不严重伤全局；
- 如果全部大幅回退，则说明后续 chunk commit 不是纯污染，而是必要的尺度/连续性载体。

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
SWA = SWKS3-style fixed protocol
semantic = disabled
```

运行记录：

| Run | Extra freeze | Start | Done | Walltime |
|---|---|---|---|---:|
| `V7_TTUSE_01_c5ttgr_freeze6_SWKS3` | `6` | `2026-05-09 02:24:41` | `02:47:47` | `23.1 min` |
| `V7_TTUSE_02_c5ttgr_freeze7_SWKS3` | `7` | `2026-05-09 02:24:41` | `02:47:40` | `23.0 min` |
| `V7_TTUSE_03_c5ttgr_freeze8_SWKS3` | `8` | `2026-05-09 02:24:41` | `02:47:16` | `22.6 min` |
| `V7_TTUSE_04_c5ttgr_freeze67_SWKS3` | `6,7` | `2026-05-09 02:24:41` | `02:47:54` | `23.2 min` |

Global metrics：

| Run | Extra freeze | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | none | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | none | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTUSE_01` | `6` | `43.5659` | `7.7448` | `92.5668` | `0.0101` | `[200,300)` 大幅改善，但全局崩 |
| `TTUSE_02` | `7` | `50.4076` | `8.3403` | `92.5831` | `0.0105` | 病灶/后段都崩 |
| `TTUSE_03` | `8` | `51.7114` | `6.7863` | `92.5682` | `0.0082` | 病灶明显更差 |
| `TTUSE_04` | `6,7` | `70.1707` | `9.5185` | `92.7294` | `0.0117` | 严重崩坏 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_use_after/trajectory_diagnostics_use_after/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 100f `[400,500)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `56.073` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `55.863` | `47.582` | `3.769` | `31.219364` |
| `TTUSE_01` | `43.5659` | `13.817` | `40.633` | `42.309` | `39.803` | `77.512` | `65.785` | `4.869` | `32.154545` |
| `TTUSE_02` | `50.4076` | `15.922` | `103.921` | `78.492` | `64.005` | `84.589` | `71.807` | `5.406` | `32.393610` |
| `TTUSE_03` | `51.7114` | `11.671` | `117.660` | `96.516` | `75.187` | `85.893` | `73.136` | `3.984` | `32.383344` |
| `TTUSE_04` | `70.1707` | `23.457` | `76.430` | `56.804` | `68.796` | `114.127` | `96.928` | `6.511` | `33.555449` |

Focus chunk diagnostics：

| Run | chunk6 `[174,206)` | chunk7 `[203,235)` | chunk8 `[232,264)` | chunk9 `[261,293)` | chunk10 `[290,322)` | chunk15 `[435,467)` | chunk16 `[464,496)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `53.799` | `74.942` | `88.814` | `76.800` | `37.576` | `61.503` | `60.853` |
| `TTGRL_04` | `53.381` | `74.567` | `88.555` | `76.806` | `37.878` | `61.269` | `60.647` |
| `TTUSE_01` | `97.541` | `23.954` | `44.793` | `44.919` | `19.809` | `83.241` | `80.501` |
| `TTUSE_02` | `102.422` | `121.967` | `43.137` | `39.578` | `21.785` | `90.370` | `87.091` |
| `TTUSE_03` | `88.548` | `113.573` | `125.345` | `33.921` | `15.472` | `91.727` | `88.207` |
| `TTUSE_04` | `155.987` | `77.316` | `21.773` | `23.277` | `48.285` | `120.126` | `113.992` |

23 节结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04`；所有 downstream freeze 都严重回退。
2. `freeze6` 是最有诊断价值的一条：它把 `[200,300)` 从 `77.644` 降到 `42.309`，chunk7-10 也明显变好，但 chunk6 自身炸到 `97.541`，后段 `[400,600)` 从 `47.582` 恶化到 `65.785`。这说明 chunk6 commit 既传播了会放大 `[200,300)` 的状态，也承载了后段连续性/尺度。
3. `freeze7/8` 不仅不能作为 use-after 机制，还会让病灶或后段更坏；尤其 `freeze8` 把 `[200,300)` 拉到 `96.516`。
4. `freeze6+7` 虽然让部分 focus chunks 变低，但整体 ATE `70.1707`，证明后续 commit 不能粗暴丢弃。
5. 结构判断更新：**commit-after-use 不能用 hard freeze 实现**。下一步如果继续结构性 TTT，需要真正的双 fast-weight 生命周期：
   - long-term fast weight 保留 chunk6/7 的 continuity；
   - short-term/transient fast weight 承接 chunk5 negative evidence，只影响 `[200,300)` 附近；
   - apply path 可以读 `W_long + W_short`，commit path 只把可信 static/continuity 分量写回 `W_long`。

当前 v7 best 仍保持：

```text
V7_TTGRL_04_c5_w0all_g0025_SWKS3
ATE / Rot = 36.2957 / 6.6182
```

---

## 16. TTT self cue 第二批：residual-tail negative evidence

15 节的 `ttt_residual` broad risk 有弱正信号，但仍未超过 `TTGRL_04`。本批进一步测试一个更硬的假设：

```text
只把 TTT replay residual 最高的一小段 token 当作 negative evidence，
其它 token 保持原 positive prior。
```

这对应 `negative_tail` 模式，但 risk source 换成 TTT 自己的 residual：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = negative_tail
TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE = ttt_residual / ttt_residual_x_dg
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
SWA = SWKS3-style fixed protocol
semantic = disabled
```

候选：

| Run | Risk source | Negative frac | Gamma | 目的 |
|---|---|---:|---:|---|
| `V7_TTSELFTAIL_01_c5_residual_frac003_g0025_SWKS3` | `ttt_residual` | `0.03` | `0.025` | residual top 3% hard negative |
| `V7_TTSELFTAIL_02_c5_residual_frac005_g0025_SWKS3` | `ttt_residual` | `0.05` | `0.025` | residual top 5% hard negative |
| `V7_TTSELFTAIL_03_c5_residualDg_frac003_g0025_SWKS3` | `ttt_residual_x_dg` | `0.03` | `0.025` | residual 与 Dg consensus top 3% |
| `V7_TTSELFTAIL_04_c5_residual_frac005_g0015_SWKS3` | `ttt_residual` | `0.05` | `0.015` | 更温和 top 5% |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TTSELFTAIL_01_c5_residual_frac003_g0025_SWKS3` | `2026-05-08 11:30:29` | `11:53:46` | `23.3 min` |
| `V7_TTSELFTAIL_02_c5_residual_frac005_g0025_SWKS3` | `2026-05-08 11:30:29` | `11:53:32` | `23.1 min` |
| `V7_TTSELFTAIL_03_c5_residualDg_frac003_g0025_SWKS3` | `2026-05-08 11:30:29` | `11:54:04` | `23.6 min` |
| `V7_TTSELFTAIL_04_c5_residual_frac005_g0015_SWKS3` | `2026-05-08 11:30:29` | `11:55:05` | `24.6 min` |

Global metrics：

| Run | Risk source | Negative frac | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | baseline | n/a | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | external low-prior broad | n/a | `0.025` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTSELF_02` | `ttt_residual` broad | n/a | `0.015` | `36.3667` | `6.6580` | `92.4422` | `0.0082` | residual broad reference |
| `TTSELFTAIL_01` | `ttt_residual` | `0.03` | `0.025` | `36.4797` | `6.6355` | `92.4429` | `0.0082` | 回退 |
| `TTSELFTAIL_02` | `ttt_residual` | `0.05` | `0.025` | `36.5261` | `6.6505` | `92.4451` | `0.0082` | 回退更明显 |
| `TTSELFTAIL_03` | `ttt_residual_x_dg` | `0.03` | `0.025` | `36.4656` | `6.6485` | `92.4457` | `0.0082` | 本批最好，但仍未过 B0 |
| `TTSELFTAIL_04` | `ttt_residual` | `0.05` | `0.015` | `36.5411` | `6.6307` | `92.4456` | `0.0082` | 温和 gamma 仍回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_selfcue_tail/trajectory_diagnostics_ttt_selfcue_tail/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTSELF_02` | `36.3667` | `5.709` | `78.013` | `77.656` | `56.989` | `47.800` | `3.820` | `31.208212` |
| `TTSELFTAIL_01` | `36.4797` | `5.782` | `78.484` | `78.051` | `57.280` | `47.960` | `3.789` | `31.213718` |
| `TTSELFTAIL_02` | `36.5261` | `5.417` | `78.584` | `78.100` | `57.297` | `47.914` | `3.795` | `31.241445` |
| `TTSELFTAIL_03` | `36.4656` | `5.682` | `78.387` | `77.990` | `57.216` | `47.818` | `3.795` | `31.246247` |
| `TTSELFTAIL_04` | `36.5411` | `5.490` | `78.670` | `78.171` | `57.343` | `47.937` | `3.781` | `31.241079` |

16 节结论：

1. residual-tail 全部失败，没有达到 `<30m`，也没有超过 `B0` / `TTSELF_02` / `TTGRL_04`。
2. 这说明 TTT replay residual 最高的一小撮 token 不是当前 harmful write 的主因；它更像“困难 token”或“高 residual token”，不等于应该反向写入的 token。
3. broad `ttt_residual` 反而比 residual-tail 好，说明如果 TTT self cue 有用，它不是少数 outlier，而是 chunk5 里较宽区域的 update-structure 偏置。
4. `ttt_residual_x_dg` 仍不成立，进一步说明 TTT residual 与外部 `D_g` 不是简单相乘关系。
5. 结合 `docs/TTT_Write_Controller_Design_v1.md`，下一步不应继续扫 `negative_frac/gamma`。更合理的方向是把 token-wise `Gamma_write` 与 block-level `Lambda_write` 分开：
   - token 侧保留当前 best 的 broad low-prior TTGR；
   - block/layer/head 侧用 TTT self cue 控制整体 write gain；
   - 或实现 update-conflict cue，直接比较 native / controlled update direction，而不是用 residual 大小当 harmful risk。

当前 v7 best 仍保持：

```text
V7_TTGRL_04_c5_w0all_g0025_SWKS3
ATE / Rot = 36.2957 / 6.6182
```

---

## 19. 结构性 TTT 补充：one-hop transient negative re-run

13 节已经验证过直接 one-hop 扣除 transient delta 的方向。本节在后续 TTL hook 接入后，重新跑一组同类 one-hop transient negative，用于确认代码路径与诊断结论是否稳定。

机制仍然是：

```text
chunk5:
    W_controlled = chunk5 TTGR controlled replay
    W_ref        = same replay without TTGR
    delta_neg    = W_controlled - W_ref

chunk6 init:
    W_next = W_commit - subtract_scale * delta_neg
```

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE = one_hop_delta
TTT_WRITE_TRANSIENT_DELTA_BRANCH_MASK = 0
SWA = SWKS3-style fixed protocol
semantic = disabled
```

smoke：

```text
V7_TTGRTRANS_SMOKE_c5_g0025_sub100_e224
```

验证结果：

- chunk5 存入 `w0` transient delta，`18` 个 layer，mean norm 约 `0.015613`。
- chunk6 读取上一轮 transient delta 并扣除 `18` 个 tensor。
- chunk7 不再携带 transient metadata，确认是 one-hop。

Global metrics：

| Run | Gamma | Subtract scale | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | `0.025` | `0.00` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRTRANS_01` | `0.025` | `1.00` | `36.4587` | `6.6166` | `92.4436` | `0.0082` | 完整扣除回退 |
| `TTGRTRANS_02` | `0.025` | `0.50` | `36.4671` | `6.6495` | `92.4435` | `0.0082` | 半扣除仍回退 |
| `TTGRTRANS_03` | `0.025` | `0.75` | `36.4489` | `6.5933` | `92.4434` | `0.0081` | Rot 稍好，但 ATE 回退 |
| `TTGRTRANS_04` | `0.020` | `1.00` | `36.4089` | `6.6238` | `92.4444` | `0.0082` | 接近 B0，未过 best |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_transient_ttl/trajectory_diagnostics_ttt_transient_ttl/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRTRANS_01` | `36.4587` | `5.447` | `78.191` | `77.883` | `57.141` | `47.913` | `3.765` | `31.224139` |
| `TTGRTRANS_02` | `36.4671` | `5.767` | `78.191` | `77.890` | `57.165` | `47.977` | `3.809` | `31.224284` |
| `TTGRTRANS_03` | `36.4489` | `5.640` | `78.168` | `77.852` | `57.128` | `47.988` | `3.754` | `31.217354` |
| `TTGRTRANS_04` | `36.4089` | `5.759` | `78.148` | `77.877` | `57.144` | `47.788` | `3.786` | `31.230766` |

19 节结论：

1. one-hop transient negative 没有达到 `<30m`，也没有超过 `TTGRL_04`。
2. 直接在下一 chunk 扣掉 `W_controlled-W_ref` 会损伤 useful continuity；`subtract_scale=1.00/0.50/0.75` 都没有成为新主线。
3. 降低 gamma 到 `0.020` 后更接近 B0，但也失去 `TTGRL_04` 的全局 ATE 收益。
4. 这进一步说明 `W_controlled-W_ref` 不是纯 harmful delta，里面混有正向几何/尺度连续性。后续若做 transient，必须控制生命周期，而不是简单 one-hop subtract。

---

## 20. 结构性 TTT：transient delta TTL 生命周期

19 节的 one-hop 删除太早，因此本节测试 transient negative delta 的生命周期。新增：

```text
TTT_WRITE_TRANSIENT_DELTA_TTL
```

语义：

- chunk5 生成 transient delta；
- 后续 chunk 只携带 metadata，不立即扣除；
- TTL 归零时再扣除 transient delta；
- 用来测试 negative evidence 应该影响几个 chunk 后再从长期 fast weight 中移除。

smoke：

```text
V7_TTGRTTL_SMOKE_c5_g0025_ttl4_sub100_e322
```

TTL=4 的 debug 序列：

| Chunk | 事件 |
|---:|---|
| `5` | 存入 transient delta，`ttl_out=4` |
| `6` | carry，`prev_ttl=4 -> ttl_out=3` |
| `7` | carry，`prev_ttl=3 -> ttl_out=2` |
| `8` | carry，`prev_ttl=2 -> ttl_out=1` |
| `9` | `prev_ttl=1`，扣除 transient delta，`18` 个 tensor |

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE = one_hop_delta
TTT_WRITE_TRANSIENT_DELTA_BRANCH_MASK = 0
SWA = SWKS3-style fixed protocol
semantic = disabled
```

Global metrics：

| Run | TTL | Subtract scale | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | no explicit TTL | `0.00` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTGRTTL_01` | `3` | `1.00` | `36.3475` | `6.6616` | `92.4444` | `0.0082` | TTL 太短，回退 |
| `TTGRTTL_02` | `4` | `1.00` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | 与 TTGRL_04 完全一致 |
| `TTGRTTL_03` | `4` | `0.50` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | 与 TTGRL_04 完全一致 |
| `TTGRTTL_04` | `5` | `1.00` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | 与 TTGRL_04 完全一致 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_transient_ttl/trajectory_diagnostics_ttt_transient_ttl/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRTTL_01` | `36.3475` | `5.914` | `78.064` | `77.817` | `57.097` | `47.584` | `3.810` | `31.235191` |
| `TTGRTTL_02` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRTTL_03` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTGRTTL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |

20 节结论：

1. TTL 实验没有达到 `<30m`，也没有超过 `TTGRL_04`。
2. `TTL=3` 会过早移除 chunk5 negative evidence，ATE 从 `36.2957` 回退到 `36.3475`。
3. `TTL=4/5` 与 `TTGRL_04` 完全重合。原因是固定 `RESET_EVERY=5` 后，chunk5 到 chunk9 已经构成自然 fast-weight lifetime group；在 reset 边界之前扣除或不扣除，对最终 trajectory 没有可见区别。
4. 这反过来解释了当前 best：`TTGRL_04` 本质上已经是 **chunk5 negative evidence 在一个 reset group 内短期保留**，而不是长期记忆策略。
5. 下一步不应继续扫 TTL；应补 chunk6 positive continuity 或真正 two-replay objective。chunk6 是 H1 里明确的 continuity carrier，值得测试“chunk5 负证据 + chunk6 正证据增强”是否能保留 TTGRL 的收益并减少后段代价。

---

## 18. H4：TTT 自身 update-conflict cue

用户补充了 `docs/TTT_Write_Controller_Design_v1.md` 中的 TTT cue 思路。这里先落实一条最直接的 TTT self cue：

```text
对每个 replay token，估计它对当前 TTT layer 的 w0 update 方向；
再和该 layer 的 aggregate update 方向比较。
```

直觉：

- 如果某个 token 的 update 方向和整体 update 同向，它更像 positive evidence；
- 如果它和整体方向夹角大，或接近反向，它可能是会污染 fast weight 的 negative / conflicting evidence；
- 这比外部 `D_g` 更靠近 TTT 写入机制本身。

实现新增：

- `TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE=ttt_w0_conflict / ttt_w0_anti / ttt_w0_energy / ttt_w0_conflict_energy`。
- 对 `w0` 估计每个 token 的 raw contribution：

```text
gate = k @ w0_old
hidden = k @ w2_old
dhidden = v @ w1_old.T
dgate = silu_backward(dhidden * hidden, gate)
token_update_i ~= outer(k_i, dgate_i)
aggregate_update = sum_i token_update_i
cos_i = cosine(token_update_i, aggregate_update)
```

风险定义：

| Risk source | 定义 | 含义 |
|---|---|---|
| `ttt_w0_conflict` | `(1 - cos_i) / 2` | 不同向就提高风险 |
| `ttt_w0_anti` | `max(-cos_i, 0)` | 只惩罚真正反向 token |
| `ttt_w0_energy` | normalized token update energy | 只看 update 强度 |
| `ttt_w0_conflict_energy` | conflict × energy | 同时要求冲突和有能量 |

smoke：

```text
V7_TTCONFLICT_SMOKE2_c5_w0_conflict_g0015_e224
```

smoke 通过。chunk5 active layer debug 显示：

| Field | Value |
|---|---:|
| mean conflict risk | `0.3624` |
| mean update cosine | `0.5672` |
| negative cosine mass | `0.0066` |
| signed min | `-0.0150` |
| negative signed mass | `0.0014` |

说明：真正反向的 token 很少；`conflict` 更多是在抓“不是强同向”的 token，而不是大量显式 anti update。

固定协议：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
SWA = SWKS3-style fixed protocol
semantic = disabled
```

运行记录：

| Run | Start | Done | Walltime |
|---|---|---:|
| `V7_TTCONFLICT_01_c5_w0_conflict_g0015_SWKS3` | `2026-05-08 13:03:16` | `13:26:59` | `23.7 min` |
| `V7_TTCONFLICT_02_c5_w0_conflict_g0025_SWKS3` | `2026-05-08 13:03:16` | `13:26:28` | `23.2 min` |
| `V7_TTCONFLICT_03_c5_w0_anti_g0025_SWKS3` | `2026-05-08 13:03:17` | `13:26:07` | `22.8 min` |
| `V7_TTCONFLICT_04_c5_w0_conflictEnergy_g0015_SWKS3` | `2026-05-08 13:03:17` | `13:27:56` | `24.7 min` |

Global metrics：

| Run | Risk source | Gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | baseline | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | external low-prior TTGR | `0.025` | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTCONFLICT_01` | `ttt_w0_conflict` | `0.015` | `36.3896` | `6.6240` | `92.4440` | `0.0082` | 过 B0，但未过 TTGRL |
| `TTCONFLICT_02` | `ttt_w0_conflict` | `0.025` | `36.5258` | `6.6461` | `92.4441` | `0.0082` | conflict 过强回退 |
| `TTCONFLICT_03` | `ttt_w0_anti` | `0.025` | `36.3743` | `6.6366` | `92.4442` | `0.0082` | anti 太稀疏，未过 TTGRL |
| `TTCONFLICT_04` | `ttt_w0_conflict_energy` | `0.015` | `36.3556` | `6.6309` | `92.4431` | `0.0082` | 本批最好，但仍未过 TTGRL |

Update-conflict debug 均值：

| Run | Risk mean | Risk p90 | Cos mean | Negative cos mass | Signed mean | Signed min | Negative mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `TTCONFLICT_01` | `0.3624` | `0.5877` | `0.5672` | `0.0066` | `0.6338` | `-0.0150` | `0.0014` |
| `TTCONFLICT_02` | `0.3624` | `0.5877` | `0.5672` | `0.0066` | `0.6288` | `-0.0250` | `0.0023` |
| `TTCONFLICT_03` | `0.0025` | `0.0000` | `0.5672` | `0.0066` | `0.9948` | `-0.0250` | `0.0000` |
| `TTCONFLICT_04` | `0.0730` | `0.1692` | `0.5672` | `0.0066` | `0.9259` | `-0.0150` | `0.0001` |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_update_conflict/trajectory_diagnostics_ttt_update_conflict/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `TTCONFLICT_01` | `36.3896` | `5.549` | `78.101` | `77.823` | `57.107` | `47.751` | `3.774` | `31.230034` |
| `TTCONFLICT_02` | `36.5258` | `5.633` | `78.312` | `77.936` | `57.187` | `48.038` | `3.800` | `31.234034` |
| `TTCONFLICT_03` | `36.3743` | `5.980` | `78.148` | `77.769` | `57.056` | `47.804` | `3.793` | `31.225577` |
| `TTCONFLICT_04` | `36.3556` | `5.650` | `78.106` | `77.743` | `57.036` | `47.695` | `3.783` | `31.215106` |

18 节结论：

1. TTT 自身 update-conflict cue 有效，但没有超过当前 best `TTGRL_04=36.2957`。
2. `conflict_energy` 是本批最好：`36.3556 / 6.6309`，超过 B0 但仍比 TTGRL_04 差 `0.0599m`。
3. 真正 anti-cosine token 极少，`negative cosine mass` 只有约 `0.0066`，所以 `ttt_w0_anti` 信号太稀疏；TTT harmful direction 更像大面积弱冲突，而不是少数强反向 token。
4. update-conflict 更适合做 diagnostic / secondary block gain，不适合作单独 negative replay selector。
5. 至此，TTT cue 本身也仍卡在 `36.3m` 平台。下一步进入结构性 TTT：**chunk5 短期负证据 / one-hop transient delta**。目标是让 chunk5 negative evidence 先影响下一段，但在下一次 commit 时从长期 fast weight 里扣掉，避免负证据长期传播。

---

## 17. TTT self cue 第三批：residual-driven block Lambda write gain

`docs/TTT_Write_Controller_Design_v1.md` 强调 TTT write controller 里应区分两件事：

```text
Gamma_write: token-wise contribution weighting，决定哪些 token 主导 pre-zeropower update direction
Lambda_write: block/layer/branch-level write gain，决定 post-replay correction 最终写多少
```

前两批 self cue 主要还在 token 侧做 negative evidence。16 节的 residual-tail 失败后，本批改测 block-level `Lambda_write`：token 侧保留当前 best 的 chunk5/w0/all `low_prior` TTGR `gamma=0.025`，commit 侧只在 chunk5 用 TTT residual risk 对 `w0` 做 layer-wise write gain。

实现/验证：

- `ttt_write_controller.py` 的 commit filter risk source 新增 `ttt_residual` / `ttt_residual_x_dg`。
- `run_pipeline_abc_v2.py` / `tools/run_attention_cue_experiment.sh` 新增 `TTT_WRITE_COMMIT_FILTER_CHUNKS`，本批只让 commit filter 作用于 chunk5。
- smoke `V7_TTSELFLAMBDA_SMOKE_c5_resmean_e224` 通过；chunk5 debug 确认：
  - `ttt_gradient_reversal_applied=True`
  - `ttt_write_commit_filter_applied=True`
  - `ttt_write_commit_filter_risk_source=ttt_residual`
  - `ttt_write_commit_filter_scale_mean ~= 0.95`

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
TTT_WRITE_COMMIT_FILTER_CHUNKS = 5
TTT_WRITE_COMMIT_FILTER_BRANCH_MASK = 0
SWA = SWKS3-style fixed protocol
semantic = disabled
```

候选：

| Run | Commit risk | Stat | Scale formula | 目的 |
|---|---|---|---|---|
| `V7_TTSELFLAMBDA_01_c5_resmean_decay_mild_SWKS3` | `ttt_residual` | mean | `clip(1.05 - 0.20*risk, 0.85, 1.05)` | 温和 layer-wise damping |
| `V7_TTSELFLAMBDA_02_c5_resq90_decay_mild_SWKS3` | `ttt_residual` | q90 | `clip(1.10 - 0.20*risk, 0.85, 1.05)` | 用高分位风险控制 layer |
| `V7_TTSELFLAMBDA_03_c5_resmean_decay_stronger_SWKS3` | `ttt_residual` | mean | `clip(1.00 - 0.15*risk, 0.85, 1.00)` | 更强 damping |
| `V7_TTSELFLAMBDA_04_c5_resDgmean_decay_mild_SWKS3` | `ttt_residual_x_dg` | mean | `clip(1.05 - 0.20*risk, 0.85, 1.05)` | residual 与 Dg consensus 的 block gain |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V7_TTSELFLAMBDA_01_c5_resmean_decay_mild_SWKS3` | `2026-05-08 12:14:52` | `12:37:38` | `22.8 min` |
| `V7_TTSELFLAMBDA_02_c5_resq90_decay_mild_SWKS3` | `2026-05-08 12:14:52` | `12:38:50` | `24.0 min` |
| `V7_TTSELFLAMBDA_03_c5_resmean_decay_stronger_SWKS3` | `2026-05-08 12:14:52` | `12:37:48` | `22.9 min` |
| `V7_TTSELFLAMBDA_04_c5_resDgmean_decay_mild_SWKS3` | `2026-05-08 12:14:52` | `12:37:08` | `22.3 min` |

Global metrics：

| Run | Commit risk / stat | Scale mean / min / max | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | baseline | n/a | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | token TTGR only | n/a | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `TTSELFLAMBDA_01` | residual / mean | `0.951 / 0.908 / 1.017` | `36.4135` | `6.6667` | `92.4434` | `0.0082` | 接近 B0，但未过 |
| `TTSELFLAMBDA_02` | residual / q90 | `0.974 / 0.932 / 1.045` | `36.3775` | `6.6586` | `92.4443` | `0.0082` | 本批 best，但未过 TTGRL_04 |
| `TTSELFLAMBDA_03` | residual / mean stronger | `0.926 / 0.894 / 0.975` | `36.3881` | `6.6373` | `92.4436` | `0.0082` | stronger damping 没有更好 |
| `TTSELFLAMBDA_04` | residual×Dg / mean | `1.024 / 1.011 / 1.042` | `36.4015` | `6.6224` | `92.4428` | `0.0082` | 局部病灶略好，但后段回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_self_lambda/trajectory_diagnostics_ttt_self_lambda/
```

| Run | ATE RMSE | FinalErr | 50f `[200,250)` | 100f `[200,300)` | 200f `[200,400)` | 200f `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `5.798` | `78.272` | `77.831` | `57.101` | `47.825` | `3.765` | `31.238916` |
| `TTGRL_04` | `36.2957` | `5.387` | `77.914` | `77.644` | `56.968` | `47.582` | `3.769` | `31.219364` |
| `LAMBDA_01` | `36.4135` | `5.846` | `78.100` | `77.796` | `57.094` | `47.867` | `3.816` | `31.219615` |
| `LAMBDA_02` | `36.3775` | `5.839` | `78.170` | `77.834` | `57.108` | `47.654` | `3.814` | `31.233938` |
| `LAMBDA_03` | `36.3881` | `5.651` | `78.067` | `77.821` | `57.105` | `47.775` | `3.789` | `31.221120` |
| `LAMBDA_04` | `36.4015` | `5.841` | `77.985` | `77.687` | `57.017` | `47.972` | `3.773` | `31.211226` |

17 节结论：

1. 本批没有达到 `<30m`，也没有超过 `TTGRL_04=36.2957`。
2. residual-driven `Lambda_write` 比 residual-tail 稳定，说明 TTT self cue 更适合做 broad / layer-wise block gain，而不是挑少量 hard negative token。
3. 但 block gain 仍没有成为新主线：best `TTSELFLAMBDA_02=36.3775` 只超过 B0，明显落后 `TTGRL_04`。
4. `residual×Dg` 能把 `[200,300)` 拉到 `77.687`，接近 `TTGRL_04=77.644`，但 `[400,600)` 回退到 `47.972`，说明它仍在用后段 continuity 换局部病灶。
5. 结合 15-17 节，`||y-v||` residual 不是足够准确的 harmful cue。下一步如果继续挖 TTT 自己的 cue，应转向 **update-conflict**：
   - 比较 native replay 与 controlled replay 的 fast-weight delta；
   - 只惩罚与 native continuity 方向冲突的分量；
   - 或实现真正的 two-replay objective，把 positive / neutral / negative 分开，而不是继续用 residual 大小控制同一个 replay。

当前 v7 best 仍保持：

```text
V7_TTGRL_04_c5_w0all_g0025_SWKS3
ATE / Rot = 36.2957 / 6.6182
```

---

## 25. 最新结构性 TTT 复盘：two-replay / short negative TTL

说明：本节是 `2026-05-09` 最新追加结果，放在文件末尾便于查看。freeze 仍只作为 causal diagnostic，不作为策略。

### 25.1 先说好消息

1. `two_replay` 工程路径已经跑通，debug 确认正向 replay 和负向 replay 是分开的，不是旧的 signed prior 小改。
2. `short negative TTL` 也跑完整 KITTI01，没有出现 strict dual-lifetime smoke 那种卡住问题。
3. 机制结论很明确：chunk5 的 negative evidence 不能 one-hop 后立刻移除，它至少需要活到当前 `RESET_EVERY=5` 周期结束；过早撤掉会把 `TTGRL_04` 的收益吃掉。

### 25.2 strict dual-lifetime 状态

| Run | 结果 |
|---|---|
| `V7_TTDL_TRUE_SMOKE_c5_w0_g0025_ttl4_long050_e224` | 异常变慢 / 无稳定进展，手动停止 |
| `V7_TTDL_TRUE_SMOKE2_c5_w0_g0025_ttl4_long050_e177` | 仍然异常慢 / 不稳定，手动停止 |

结论：strict dual-lifetime 方向仍值得做，但当前实现路径不稳定，本轮不计 full 指标。

### 25.3 two-replay objective

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = two_replay
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
SWA = SWKS3-style fixed protocol
```

| Run | Negative risk source | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | baseline | `36.4161` | `6.6128` | `92.4452` | `0.0082` | baseline |
| `TTGRL_04` | low-prior signed replay | `36.2957` | `6.6182` | `92.4428` | `0.0082` | current v7 best |
| `V7_TT2R_01` | `prior` | `36.5046` | `6.6219` | `92.4450` | `0.0082` | 回退 |
| `V7_TT2R_02` | `update_conflict` | `36.4593` | `6.6373` | `92.4449` | `0.0082` | 回退 |
| `V7_TT2R_03` | `update_conflict_energy` | `36.4560` | `6.5800` | `92.4434` | `0.0081` | two-replay best，但未过 |
| `V7_TT2R_04` | `ttt_residual_x_dg` | `36.5463` | `6.6213` | `92.4437` | `0.0082` | 回退 |

诊断目录：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_two_replay_structural/trajectory_diagnostics_two_replay/
```

结论：two-replay 是可用机制，但第一版 `all positive + risk negative` 不如 `TTGRL_04` 的 continuous low-prior direction shift。harmful update 更像 chunk5 `w0` 大面积弱偏置，而不是一个可以单独扣掉的强 `G_neg`。

### 25.4 short negative TTL

固定：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = low_prior
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS = 5
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK = 0
TTT_WRITE_GRADIENT_REVERSAL_GAMMA = 0.025
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE = ttl_delta
TTT_WRITE_TRANSIENT_DELTA_SUBTRACT_SCALE = 1.0
TTT_WRITE_TRANSIENT_DELTA_BRANCH_MASK = 0
```

| Run | TTL | ATE RMSE | Rot RMSE | `[200,300)` | `[400,600)` | FinalErr | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0_SWKS3` | n/a | `36.4161` | `6.6128` | `77.831` | `47.825` | `5.798` | baseline |
| `TTGRL_04` | persistent within reset | `36.2957` | `6.6182` | `77.644` | `47.582` | `5.387` | current v7 best |
| `V7_TTGRTTL_01` | `1` | `36.4587` | `6.6166` | `77.883` | `47.913` | `5.447` | 太早扣除，回退 |
| `V7_TTGRTTL_02` | `2` | `36.4256` | `6.6056` | `77.881` | `47.832` | `5.679` | 接近 B0，但未过 |
| `V7_TTGRTTL_03` | `3` | `36.3475` | `6.6616` | `77.817` | `47.584` | `5.914` | TTL best，但未过 TTGRL_04 |
| `V7_TTGRTTL_04` | `4` | `36.2957` | `6.6182` | `77.644` | `47.582` | `5.387` | 与 TTGRL_04 完全重合 |

诊断目录：

```text
results/kitti01_hmc_v2/acl2_v7_ttt_short_negative_ttl/trajectory_diagnostics_short_negative_ttl/
```

25.4 结论：

1. short negative TTL 没有超过 `TTGRL_04`。
2. TTL 越短，越接近把 chunk5 negative evidence 从长期状态中过早拿掉，ATE / `[200,300)` 都回退。
3. TTL=4 与 `TTGRL_04` 完全重合，说明当前 best 需要 chunk5 negative correction 在整个 reset window 内持续存在。
4. 这条结果反过来给了结构方向：不是 “用一下就删掉”，而是要 **在一个 reset 周期内维持 short negative state，同时不要让它跨 reset 污染长期 state**。当前 `RESET_EVERY=5` 已经天然做到了一部分，所以 simple TTL 没带来额外收益。

### 25.5 当前判断

当前 v7 best 不变：

```text
V7_TTGRL_04_c5_w0all_g0025_SWKS3
ATE / Rot = 36.2957 / 6.6182
```

但结构实验带来了明确边界：

1. freeze 不是解药，只是诊断；
2. one-hop short negative 不够，chunk5 negative direction 必须覆盖到 reset window 末尾；
3. first-version two-replay 不够，需要更明确的三组目标：

```text
G_commit = G_pos + lambda_neu * G_neu - gamma * G_neg
```

其中 `G_pos` 只取 high-continuity / static token，`G_neg` 只取 update-conflict high-risk token，`G_neu` 保留普通 continuity。下一步如果继续结构性 TTT，应优先实现这个三组 replay objective，而不是继续扫 TTL / scalar gamma。

---

## 37. 最新总结入口（2026-05-10 04:13）

完整阶段总结已写在第 35 节：`v7 TTT 写入探索阶段总结（截至 2026-05-10 04:13）`。

当前已完成 full KITTI01 的最好结果：

```text
V7_TRIREPLAY_WINGAM_03_bodyg005_exitg0030_neu085_SWKS3
ATE / Rot = 34.1903 / 6.5666
```

核心结论：

```text
有效进展来自 TTT 自有 update_conflict_energy cue + 三组 replay objective。
freeze 只作为因果诊断，不是解法。
当前最好策略是在 chunks 5-9 使用较强负证据、chunks 10-12 使用更弱负证据，
并把 neutral continuity 以 0.85 保留，而不是全量写入。
```

`WINGAMF_01-04` 在用户要求总结后中止，只有 partial chunk 输出，没有完整 benchmark，不参与结论。

# ACL2 v8 实验记录

日期：2026-05-12  
计划文件：`docs/ACL2_v8_TTT_Native_Cue_Windowed_TriReplay_GlobalDrift_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/`  
目标：从 v7 `WINGAM_03` 出发，验证 windowed tri-replay 的可复现性，并测试 TTT-native cue 是否能自动触发 body / exit window。

固定主线，除非单节特别说明：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read = frame pair/all
beta = 4.75
write score = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
RESET_EVERY = 5
SWA = SWKS3-style fixed protocol
TTT objective = tri_replay
risk = update_conflict_energy
branch = w0
positive_frac = 0.35
negative_frac = 0.12
```

当前 v7 reference：

| Reference | ATE RMSE | Rot RMSE | FinalErr | `[200,300)` | 结论 |
|---|---:|---:|---:|---:|---|
| `B0_SWKS3` | `36.4161` | `6.6128` | `5.798` | `77.831` | v7 reproducible baseline |
| `WINGAM_03` | `34.1903` | `6.5666` | `6.195` | `75.576` | v7 best full KITTI01 |

---

## 1. H1 / Batch A：WINGAM_03 repeat 与 global drift dashboard

固定 `WINGAM_03` 配置：

```text
body chunks 5-9:
    gamma = 0.005
    pos / neg / neutral = 0.35 / 0.12 / 0.85
exit chunks 10-12:
    gamma = 0.0030
    pos / neg / neutral = 0.35 / 0.12 / 0.85
```

运行记录：

| Run | GPU | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V8_A1_WINGAM_03_repeat_SWKS3` | `0` | `2026-05-12 07:52:39` | `08:22:10` | `29.5 min` |

Global metrics：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---|
| `WINGAM_03` | `34.1903` | `6.5666` | `92.4202` | `0.0083` | v7 best reference |
| `V8_A1_WINGAM_03_repeat_SWKS3` | `34.190276` | `6.566600` | `92.420235` | `0.008255` | strict repeat pass |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_a1_repeat/
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/global_drift_dashboard_h1_final/
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/global_drift_dashboard_h1_final_v8/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `3.774` | `30.994373` |

H1 结论：

1. `WINGAM_03` 在 v8 主目录下严格复现，ATE 与原 v7 结果对齐到 `34.190276m`。
2. H1 promotion gate 通过：repeat 偏差远小于计划阈值 `0.03m`。
3. `global drift dashboard` 已生成，可作为 v8 后续所有窗口策略的固定对照。

---

## 2. H2：TTT-native cue auto-window 第一批

使用 `GLOBAL_01` all-chunk `update_conflict_energy` 诊断作为自动窗口来源，新增工具：

```text
tools/ttt_update_conflict_audit.py
tools/ttt_auto_window_selector.py
tools/ttt_global_drift_dashboard.py
```

auto selector 输出：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/update_conflict_audit_global01_allchunks/
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/auto_window_selector_global01/
```

自动窗口摘要：

| Strategy | Body chunks | Exit chunks | Body F1 vs manual `5-9` | Exit F1 vs manual `10-12` |
|---|---|---|---:|---:|
| `AUTO_WIN_01` | `0,5,20,30,35` | `36,37` | `0.200` | `0.000` |
| `AUTO_WIN_02` | `0,1,5,9,10,13,14,15,20,23,24,25,28,29,30,31,33,35,36` | `37` | `0.167` | `0.000` |
| `AUTO_WIN_03` | `0,5,10,20,30` | `31,32,33` | `0.200` | `0.000` |
| `AUTO_WIN_04` | `0,5,20,30,35` | none | `0.200` | n/a |
| `AUTO_WIN_05` | `5,6,7,8,9` | `20,30,35` | `1.000` | `0.000` |
| `AUTO_WIN_06` | `0,5,20,30,35` | `10,11,12` | `0.200` | `1.000` |

Full run 记录：

| Run | GPU | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V8_AUTO_WIN_01_top5Rbody_tail3_SWKS3` | `1` | `2026-05-12 08:23:30` | `08:55:05` | `31.6 min` |
| `V8_AUTO_WIN_02_thresholdRbody_tail_SWKS3` | `2` | `2026-05-12 08:23:30` | `09:07:04` | `43.6 min` |
| `V8_AUTO_WIN_03_l12h0_tail3_SWKS3` | `3` | `2026-05-12 08:23:30` | `08:55:30` | `32.0 min` |
| `V8_AUTO_WIN_04_top5Rbody_noexit_SWKS3` | `4` | `2026-05-12 08:23:30` | `08:52:51` | `29.4 min` |
| `V8_AUTO_WIN_05_manualbody_autoexit_SWKS3` | `5` | `2026-05-12 08:23:30` | `08:54:31` | `31.0 min` |
| `V8_AUTO_WIN_06_autobody_manualexit_SWKS3` | `6` | `2026-05-12 08:23:30` | `08:53:58` | `30.5 min` |

Global metrics：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | `34.1903` | `6.5666` | `92.4202` | `0.0083` | reference |
| `AUTO_WIN_01` | `36.5787` | `6.1073` | `92.4286` | `0.0076` | ATE 回退，Rot/endpoint 好 |
| `AUTO_WIN_02` | `36.0854` | `7.2079` | `92.4198` | `0.0087` | 宽窗口成本高且姿态崩 |
| `AUTO_WIN_03` | `36.0454` | `7.9771` | `92.4322` | `0.0095` | layer12/head0 auto window 不成立 |
| `AUTO_WIN_04` | `36.5790` | `6.0646` | `92.4286` | `0.0076` | 与 AUTO_WIN_01 近似，同类回退 |
| `AUTO_WIN_05` | `35.3640` | `5.9553` | `92.4211` | `0.0077` | 本批 ATE best，但未过 reference |
| `AUTO_WIN_06` | `35.7853` | `6.0668` | `92.4230` | `0.0076` | manual exit 有帮助但 body 自动失败 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_auto_windows/
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/global_drift_dashboard_auto_windows/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `42.280` | `3.774` | `30.994373` |
| `AUTO_WIN_01` | `36.5787` | `3.141` | `78.369` | `57.497` | `47.307` | `3.378` | `31.093914` |
| `AUTO_WIN_02` | `36.0854` | `13.116` | `77.519` | `56.870` | `46.434` | `4.426` | `30.977612` |
| `AUTO_WIN_03` | `36.0454` | `15.093` | `77.867` | `57.066` | `45.734` | `5.081` | `31.113532` |
| `AUTO_WIN_04` | `36.5790` | `2.586` | `78.369` | `57.497` | `47.307` | `3.331` | `31.093888` |
| `AUTO_WIN_05` | `35.3640` | `2.499` | `77.029` | `56.653` | `45.071` | `3.236` | `31.015044` |
| `AUTO_WIN_06` | `35.7853` | `2.763` | `77.185` | `56.506` | `45.644` | `3.350` | `31.031701` |

H2 结论：

1. H2 第一批没有达到 `<30m`，也没有超过 `WINGAM_03_repeat=34.1903`。
2. 当前 naive auto-window 不能替代人工窗口。它过度选择 reset-like / globally recurring chunks，例如 `0,20,30,35`，导致 body window F1 低。
3. `AUTO_WIN_05` 说明 manual body `5-9` 仍是必要主结构；但自动 exit 选成 `20,30,35` 后，ATE 从 `34.1903` 回退到 `35.3640`。
4. `AUTO_WIN_06` 说明 manual exit `10-12` 有帮助，但自动 body 仍失败，ATE 回退到 `35.7853`。
5. 有趣的是，多条 auto-window 明显改善 Rot / Yaw / FinalErr，却牺牲 `[200,300)` 与 overall ATE。这说明当前 TTT-native cue 的全局峰值更像姿态/endpoint regularizer，不是病灶 body/exit 触发器。
6. 下一步不应直接进入 H3 layer/head routing；应先修正 auto-window selector，去掉 reset-periodic false positives，并把 chunk-level cue 改成相对 reset-group / downstream-drift aware 的 score。

当前 v8 best 仍保持：

```text
V8_A1_WINGAM_03_repeat_SWKS3
ATE / Rot = 34.190276 / 6.566600
```

---

## 4. H2C：post-region weak tri-replay 诊断

H2 / H2B 说明手工 body/exit window 仍难以被 naive auto selector 替代，且 `WINGAM_03` 附近的 body / exit gamma 与 neutral 细扫已基本收口。本批转向 post-region weak tri-replay：在 `WINGAM_03` 的 chunks `5-12` 基础上，给后段候选 chunk 极弱 `update_conflict_energy` tri-replay，测试是否能改善长期 drift / secondary disease window。

固定：

```text
base = WINGAM_03
mode = tri_replay
risk = update_conflict_energy
branch = w0
body chunks 5-9: gamma = 0.005
exit chunks 10-12: gamma = 0.003
pos / neg / neutral = 0.35 / 0.12 / 0.85
RESET_EVERY = 5
SWA = SWKS3-style fixed protocol
```

第一批 post-region：

| Run | Extra chunks | Extra gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | none | n/a | `34.1903` | `6.5666` | `92.4202` | `0.0083` | reference |
| `POSTREG_01` | `20,30,35` | `0.00025` | `34.4678` | `5.9934` | `92.4132` | `0.0077` | 姿态/endpoint 好，但 ATE 回退 |
| `POSTREG_02` | `20,30,35` | `0.00050` | `34.4509` | `5.9543` | `92.4133` | `0.0077` | 同类回退 |
| `POSTREG_03` | `20,30,35` | `0.00100` | `34.4258` | `5.9894` | `92.4140` | `0.0077` | ATE 仍未过 reference |
| `POSTREG_04` | `15,16` | `0.00050` | `34.1962` | `6.4377` | `92.4113` | `0.0081` | 接近 reference，Rot 明显更好 |
| `POSTREG_05` | `15,16,20` | `0.00050` | `34.3059` | `6.4101` | `92.4081` | `0.0081` | 加 chunk20 后 ATE 回退 |
| `POSTREG_06` | `13,14,15,16` | `0.00025` | `34.4694` | `6.4352` | `92.4116` | `0.0082` | 过宽后窗回退 |
| `POSTREG_07` | `13,14,15,16` | `0.00050` | `34.4472` | `6.4322` | `92.4110` | `0.0082` | 仍回退 |

第二批围绕 chunks `15/16` 收口：

| Run | Extra chunks | Extra gamma map | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| `POSTREG_08` | `15,16` | `15:0.00025,16:0.00025` | `34.2154` | `6.4603` | `92.4105` | `0.0082` | 未过 |
| `POSTREG_09` | `15,16` | `15:0.00075,16:0.00075` | `34.1908` | `6.4337` | `92.4107` | `0.0082` | 几乎追平 reference，但未过 |
| `POSTREG_10` | `15` | `15:0.00050` | `34.2518` | `6.5013` | `92.4198` | `0.0082` | chunk15 alone 回退 |
| `POSTREG_11` | `16` | `16:0.00050` | `34.2017` | `6.4478` | `92.4038` | `0.0081` | chunk16 alone 有信号 |
| `POSTREG_12` | `15,16` | `15:0.00025,16:0.00050` | `34.2071` | `6.4769` | `92.4107` | `0.0082` | 未过 |
| `POSTREG_13` | `15,16` | `15:0.00050,16:0.00025` | `34.2037` | `6.4620` | `92.4108` | `0.0082` | 未过 |
| `POSTREG_14` | `15,16` | `15:0.00050,16:0.00050`, exit `0.00275` | `34.1958` | `6.4853` | `92.4106` | `0.0082` | 未过 reference |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_postreg/
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_postreg15/
```

关键 segment：

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `3.774` | `30.994373` |
| `POSTREG_04` | `34.1962` | `6.088` | `75.022` | `55.070` | `3.732` | `30.878330` |
| `POSTREG_09` | `34.1908` | `6.401` | `75.044` | `55.088` | `3.740` | `30.876584` |
| `POSTREG_11` | `34.2017` | `5.726` | `74.459` | `54.705` | `3.744` | `30.786295` |
| `POSTREG_14` | `34.1958` | `6.450` | `75.019` | `55.082` | `3.781` | `30.873597` |

H2C 结论：

1. 本批没有达到 `<30m`，也没有严格超过 `WINGAM_03_repeat=34.1903`。
2. 但 post-region weak tri-replay 是有效诊断方向。`POSTREG_04/09/11/14` 都把 `[200,300)` 从 `75.576` 降到 `75.0` 或 `74.46` 附近，同时 Rot 明显好于 `WINGAM_03`。
3. `20,30,35` 这类 auto-window false positive chunks 更像姿态/endpoint regularizer，能改善 Rot / Yaw / FinalErr，但整体 ATE 明显回退。
4. chunks `15/16` 比 chunks `20/30/35` 更接近真实长期 drift 修正；其中 chunk16 alone 的信号最干净：`POSTREG_11` 把 `[200,300)` 降到 `74.459`，但 ATE 仍差 `+0.0114m`。
5. 下一步围绕 chunk16 单点做更窄 gamma sweep，而不是继续加宽 post window。

---

## 5. H2D：chunk16 weak tri-replay fine sweep

基于 H2C，固定 `WINGAM_03` body/exit，额外只给 chunk16 极弱 tri-replay：

```text
base chunks 5-9:  gamma = 0.005
base chunks 10-12: gamma = 0.003
extra chunk = 16
extra pos / neg / neutral = 0.35 / 0.12 / 0.85
```

运行记录：

| Run | GPU | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V8_C16FINE_01_c16g00010_SWKS3` | `1` | `2026-05-12 11:25:47` | `12:00:20` | `34.6 min` |
| `V8_C16FINE_02_c16g00020_SWKS3` | `2` | `2026-05-12 11:25:47` | `12:00:52` | `35.1 min` |
| `V8_C16FINE_03_c16g00030_SWKS3` | `3` | `2026-05-12 11:25:47` | `12:02:42` | `36.9 min` |
| `V8_C16FINE_04_c16g00040_SWKS3` | `4` | `2026-05-12 11:25:47` | `11:59:28` | `33.7 min` |
| `V8_C16FINE_05_c16g00060_SWKS3` | `5` | `2026-05-12 11:25:47` | `12:00:23` | `34.6 min` |
| `V8_C16FINE_06_exitg00275_c16g00025_SWKS3` | `6` | `2026-05-12 11:25:47` | `12:00:25` | `34.6 min` |
| `V8_C16FINE_07_exitg0025_c16g00025_SWKS3` | `7` | `2026-05-12 11:25:47` | `12:00:20` | `34.6 min` |

Global metrics：

| Run | Chunk16 gamma | Exit gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | n/a | `0.0030` | `34.1903` | `6.5666` | `92.4202` | `0.0083` | previous best |
| `C16FINE_01` | `0.00010` | `0.0030` | `34.1787` | `6.5190` | `92.4046` | `0.0082` | 过 reference |
| `C16FINE_02` | `0.00020` | `0.0030` | `34.1811` | `6.5193` | `92.4047` | `0.0082` | 接近 |
| `C16FINE_03` | `0.00030` | `0.0030` | `34.1627` | `6.4986` | `92.4046` | `0.0082` | **当前 v8 best** |
| `C16FINE_04` | `0.00040` | `0.0030` | `34.2119` | `6.4158` | `92.4032` | `0.0081` | Rot 最好，但 ATE 回退 |
| `C16FINE_05` | `0.00060` | `0.0030` | `34.1970` | `6.5257` | `92.4040` | `0.0082` | 未过 best |
| `C16FINE_06` | `0.00025` | `0.00275` | `34.2112` | `6.4876` | `92.4034` | `0.0082` | exit gamma 联动未过 |
| `C16FINE_07` | `0.00025` | `0.00250` | `34.2385` | `6.4567` | `92.4038` | `0.0081` | Rot 好但 ATE 回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_c16fine/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `3.774` | `30.994373` |
| `C16FINE_01` | `34.1787` | `6.041` | `74.454` | `54.699` | `3.807` | `30.795558` |
| `C16FINE_02` | `34.1811` | `6.231` | `74.476` | `54.713` | `3.807` | `30.796737` |
| `C16FINE_03` | `34.1627` | `6.218` | `74.445` | `54.693` | `3.791` | `30.794618` |
| `C16FINE_04` | `34.2119` | `5.768` | `74.460` | `54.708` | `3.717` | `30.780640` |
| `C16FINE_05` | `34.1970` | `6.364` | `74.460` | `54.705` | `3.815` | `30.788924` |
| `C16FINE_06` | `34.2112` | `6.351` | `74.451` | `54.716` | `3.770` | `30.776585` |
| `C16FINE_07` | `34.2385` | `6.016` | `74.489` | `54.744` | `3.752` | `30.782204` |

Focus chunk diagnostics for `C16FINE_03`：

| Rank | Chunk | Frame range | RMSE |
|---:|---:|---|---:|
| 1 | `8` | `[232,264)` | `83.489` |
| 2 | `9` | `[261,293)` | `77.317` |
| 3 | `7` | `[203,235)` | `68.655` |
| 4 | `16` | `[464,496)` | `58.883` |
| 5 | `15` | `[435,467)` | `58.562` |
| 6 | `6` | `[174,206)` | `48.306` |
| 10 | `10` | `[290,322)` | `39.952` |

H2D 结论：

1. `C16FINE_03` 刷新 v8 best：`34.1627 / 6.4986`。相比 `WINGAM_03_repeat`，ATE 改善 `0.0276m`，Rot 改善 `0.0680deg`。
2. 更重要的是它通过 v8 relative local gate：`[200,300)=74.445 <= 74.50`，`[200,400)=54.693` 也明显优于 `WINGAM_03_repeat=55.428`。
3. 但它仍没有达到 strong success：ATE 仍高于 `34.00`，FinalErr `6.218` 略差于 `WINGAM_03_repeat=6.195`，且距离 `<30m` 仍约 `4.16m`。
4. chunk16 的有效 gamma 极弱，量级在 `1e-4` 到 `4e-4`。这说明 post-region correction 不是新的强 body window，而更像长期 drift / scale handoff 的微弱校正。
5. lowering exit gamma 与 chunk16 联动没有帮助：`C16FINE_06/07` 都回退，说明当前 body/exit 主窗口仍应保持 `0.005 / 0.003`。
6. 下一步围绕新 best 做更窄 chunk16 sweep，并测试 chunk17 邻接或 chunk16 neutral role，确认该收益是 chunk16 单点 drift correction，还是后段 weak replay window 的入口。

当前 v8 best 更新为：

```text
V8_C16FINE_03_c16g00030_SWKS3
ATE / Rot = 34.1627 / 6.4986
[200,300) = 74.445
```

---

## 6. H2E：chunk16 / chunk17 micro follow-up

H2D 显示 chunk16 极弱 tri-replay 有相对成功信号，但 `gamma=0.0003` 只是粗网格点。本批继续围绕 chunk16 做更窄 gamma，并测试 chunk17 是否可以作为后段邻接 handoff。

固定：

```text
base = C16FINE_03 / WINGAM_03 + chunk16 weak tri-replay
chunks 5-9: gamma = 0.005
chunks 10-12: gamma = 0.003
chunk16 baseline gamma = 0.0003
pos / neg / neutral = 0.35 / 0.12 / 0.85
```

运行记录：

| Run | GPU | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V8_C16FINE2_08_c16g00025_SWKS3` | `1` | `2026-05-12 13:13:39` | `13:49:16` | `35.6 min` |
| `V8_C16FINE2_09_c16g000275_SWKS3` | `2` | `2026-05-12 13:13:38` | `13:47:30` | `33.9 min` |
| `V8_C16FINE2_10_c16g000325_SWKS3` | `3` | `2026-05-12 13:13:39` | `13:49:23` | `35.7 min` |
| `V8_C16FINE2_11_c16g00035_SWKS3` | `4` | `2026-05-12 13:13:39` | `13:48:51` | `35.2 min` |
| `V8_C16FINE2_12_c16g0003_c17g0001_SWKS3` | `5` | `2026-05-12 13:13:39` | `13:50:22` | `36.7 min` |
| `V8_C16FINE2_13_c16g0003_c17g0002_SWKS3` | `6` | `2026-05-12 13:13:39` | `13:51:22` | `37.7 min` |
| `V8_C16FINE2_14_c16g0003_c16neu100_SWKS3` | `7` | `2026-05-12 13:13:39` | `13:48:41` | `35.0 min` |

Global metrics：

| Run | Setting | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | reference | `34.1903` | `6.5666` | `92.4202` | `0.0083` | old reference |
| `C16FINE_03` | c16 `0.00030` | `34.1627` | `6.4986` | `92.4046` | `0.0082` | current best |
| `C16FINE2_08` | c16 `0.00025` | `34.2045` | `6.4493` | `92.4038` | `0.0081` | gamma 稍弱，ATE 回退 |
| `C16FINE2_09` | c16 `0.000275` | `34.1818` | `6.4561` | `92.4041` | `0.0081` | 接近，但未过 |
| `C16FINE2_10` | c16 `0.000325` | `34.2165` | `6.4771` | `92.4035` | `0.0082` | 回退 |
| `C16FINE2_11` | c16 `0.00035` | `34.1822` | `6.4930` | `92.4044` | `0.0082` | 接近，但未过 |
| `C16FINE2_12` | c16 `0.00030`, c17 `0.00010` | `34.2139` | `6.5085` | `92.3998` | `0.0082` | 局部病灶更好，但 ATE 回退 |
| `C16FINE2_13` | c16 `0.00030`, c17 `0.00020` | `34.2201` | `6.4554` | `92.3993` | `0.0082` | 同类回退 |
| `C16FINE2_14` | c16 `0.00030`, c16 neutral `1.00` | `34.1919` | `6.4217` | `92.4039` | `0.0081` | Final/Yaw/Rot 好，但 ATE 未过 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_c16fine2/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `3.774` | `30.994373` |
| `C16FINE_03` | `34.1627` | `6.218` | `74.445` | `54.693` | `3.791` | `30.794618` |
| `C16FINE2_08` | `34.2045` | `5.742` | `74.479` | `54.718` | `3.748` | `30.788131` |
| `C16FINE2_09` | `34.1818` | `5.806` | `74.467` | `54.709` | `3.757` | `30.791573` |
| `C16FINE2_10` | `34.2165` | `6.238` | `74.464` | `54.711` | `3.778` | `30.781336` |
| `C16FINE2_11` | `34.1822` | `6.325` | `74.462` | `54.706` | `3.782` | `30.790076` |
| `C16FINE2_12` | `34.2139` | `6.169` | `74.121` | `54.492` | `3.823` | `30.734756` |
| `C16FINE2_13` | `34.2201` | `6.013` | `74.117` | `54.492` | `3.772` | `30.727964` |
| `C16FINE2_14` | `34.1919` | `5.636` | `74.457` | `54.703` | `3.729` | `30.787828` |

H2E 结论：

1. 本批没有达到 `<30m`，也没有超过 `C16FINE_03=34.1627`。
2. chunk16 gamma sweet spot 仍以 `0.00030` 最稳；`0.000275/0.00035` 接近但未过，`0.00025/0.000325` 明显回退。
3. chunk17 有很强局部病灶信号：`C16FINE2_12/13` 把 `[200,300)` 压到 `~74.12`、`[200,400)` 压到 `54.49`，但 overall ATE 回退到 `34.21+`。这说明 chunk17 weak replay 更像局部病灶/scale correction，会牺牲全局平衡。
4. `C16FINE2_14` 说明 chunk16 的 neutral role 可以作为 Rot / FinalErr regularizer：FinalErr 从 `6.218` 降到 `5.636`、Rot 从 `6.4986` 降到 `6.4217`，但 ATE 回到 `34.1919`。
5. 下一步不继续单纯扫 chunk16 gamma。更合理的是做 chunk16 role routing：固定 `gamma=0.0003`，调 c16 的 `pos/neg/neutral`，验证它到底是 weak negative correction、neutral continuity handoff，还是二者混合。

当前 v8 best 仍保持：

```text
V8_C16FINE_03_c16g00030_SWKS3
ATE / Rot = 34.1627 / 6.4986
```

---

## 7. H2F：chunk16 role routing

H2E 显示 chunk16 `gamma=0.0003` 附近已经收口，但 `neutral=1.0` 能改善 FinalErr / Rot，`neg_frac` 过强可能伤害整体平衡。因此本批固定 chunk16 gamma，只调 chunk16 自己的 `pos / neg / neutral`。

固定：

```text
chunks 5-9: gamma = 0.005, pos/neg/neutral = 0.35/0.12/0.85
chunks 10-12: gamma = 0.003, pos/neg/neutral = 0.35/0.12/0.85
chunk16: gamma = 0.0003
risk = update_conflict_energy
branch = w0
```

运行记录：

| Run | GPU | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3` | `1` | `2026-05-12 13:53:55` | `14:26:54` | `33.0 min` |
| `V8_C16ROLE_02_c16_pos035_neg010_neu085_SWKS3` | `2` | `2026-05-12 13:53:55` | `14:26:39` | `32.7 min` |
| `V8_C16ROLE_03_c16_pos035_neg014_neu085_SWKS3` | `3` | `2026-05-12 13:53:55` | `14:27:58` | `34.1 min` |
| `V8_C16ROLE_04_c16_pos045_neg012_neu085_SWKS3` | `4` | `2026-05-12 13:53:55` | `14:29:38` | `35.7 min` |
| `V8_C16ROLE_05_c16_pos025_neg012_neu085_SWKS3` | `5` | `2026-05-12 13:53:55` | `14:27:04` | `33.2 min` |
| `V8_C16ROLE_06_c16_pos035_neg012_neu075_SWKS3` | `6` | `2026-05-12 13:53:55` | `14:28:03` | `34.1 min` |
| `V8_C16ROLE_07_c16_pos035_neg012_neu095_SWKS3` | `7` | `2026-05-12 13:53:55` | `14:27:11` | `33.3 min` |

Global metrics：

| Run | Chunk16 role | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `C16FINE_03` | `0.35/0.12/0.85` | `34.1627` | `6.4986` | `92.4046` | `0.0082` | previous best |
| `C16ROLE_01` | `0.35/0.08/0.85` | `34.1583` | `6.5327` | `92.4059` | `0.0082` | **当前 v8 best，ATE tiny gain** |
| `C16ROLE_02` | `0.35/0.10/0.85` | `34.1979` | `6.5012` | `92.4045` | `0.0082` | 回退 |
| `C16ROLE_03` | `0.35/0.14/0.85` | `34.2004` | `6.4398` | `92.4034` | `0.0081` | Rot/Final 好，但 ATE 回退 |
| `C16ROLE_04` | `0.45/0.12/0.85` | `34.1792` | `6.4440` | `92.4045` | `0.0081` | pos 增强未过 |
| `C16ROLE_05` | `0.25/0.12/0.85` | `34.1745` | `6.4930` | `92.4050` | `0.0082` | pos 降低未过 |
| `C16ROLE_06` | `0.35/0.12/0.75` | `34.1883` | `6.5080` | `92.4044` | `0.0082` | neutral 过低未过 |
| `C16ROLE_07` | `0.35/0.12/0.95` | `34.1598` | `6.5042` | `92.4052` | `0.0082` | 接近 best，但未过 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_c16role/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `3.774` | `30.994373` |
| `C16FINE_03` | `34.1627` | `6.218` | `74.445` | `54.693` | `3.791` | `30.794618` |
| `C16ROLE_01` | `34.1583` | `6.249` | `74.522` | `54.740` | `3.803` | `30.809795` |
| `C16ROLE_02` | `34.1979` | `6.402` | `74.500` | `54.732` | `3.791` | `30.791308` |
| `C16ROLE_03` | `34.2004` | `5.796` | `74.442` | `54.695` | `3.737` | `30.782325` |
| `C16ROLE_04` | `34.1792` | `5.979` | `74.520` | `54.745` | `3.741` | `30.796179` |
| `C16ROLE_05` | `34.1745` | `6.165` | `74.548` | `54.762` | `3.773` | `30.801907` |
| `C16ROLE_06` | `34.1883` | `6.218` | `74.472` | `54.713` | `3.798` | `30.791159` |
| `C16ROLE_07` | `34.1598` | `6.300` | `74.482` | `54.717` | `3.799` | `30.799346` |

H2F 结论：

1. `C16ROLE_01` 刷新当前 v8 best：`34.1583 / 6.5327`，但只比 `C16FINE_03` 改善 `0.0044m`，属于 tiny gain。
2. chunk16 的 `neg_frac=0.08` 略优于 `0.12`，说明 chunk16 不是强 negative body window，而是后段 weak handoff / drift correction；负样本过多会伤整体平衡。
3. `C16ROLE_03` 和 `C16ROLE_04` 改善 Rot / FinalErr / Yaw，但 ATE 未过，说明 chunk16 role 可以当姿态 regularizer，却不是继续冲 `<30m` 的主杠杆。
4. `[200,300)` 最低仍来自更强局部 correction：`C16ROLE_03=74.442` 或 `C16FINE_03=74.445`，而当前 ATE best `C16ROLE_01` 的 `[200,300)=74.522` 略差。这再次说明 overall ATE 最优和局部病灶最低不是同一个点。
5. chunk16 线已经进入 `34.16m` 平台内微扰，继续调 c16 role/gamma 预期收益很低。下一步应回到计划的 H3/H4：layer/head routed tri-replay 或 direction-alignment risk，而不是继续围绕 post-region chunk16 微扫。

当前 v8 best 更新为：

```text
V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3
ATE / Rot = 34.1583 / 6.5327
[200,300) = 74.522
```

---

## 3. H2B：reset-aware auto selector 复查与 WINGAMF 窄扫补完

用户要求继续探索，并说明 GPU `1-7` 可用。本节先修正 H2 第一批暴露的问题：naive auto-window 会选择 reset-like / globally recurring chunks，而不是病灶窗口。

工程补充：

```text
tools/ttt_auto_window_selector.py
    新增 reset phase residual score
    新增 local contiguous window selector
    新增 AUTO2_WIN_* 输出
```

selector v2 输出目录：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/auto_window_selector_global01_v2/
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/auto_window_selector_global01_v2_local5_12/
```

结论：reset-aware selector 虽然去掉了一部分 reset artifact，但仍不能可靠复原 `5-9 / 10-12`；连续窗口偏向 `8-12` 或 `9-13`。因此本节不把 auto selector 作为 full-run 主线，而是补完此前 `WINGAMF` 中止留下的 current-best 附近窄扫。

固定：

```text
base = WINGAM_03
mode = tri_replay
risk = update_conflict_energy
branch = w0
chunks = 5-12
body chunks 5-9 baseline gamma = 0.005
exit chunks 10-12 baseline gamma = 0.003
pos / neg / neutral baseline = 0.35 / 0.12 / 0.85
SWA = SWKS3-style fixed protocol
RESET_EVERY = 5
```

运行记录：

| Run | GPU | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V8_WINGAMF_01_bodyg005_exitg0025_neu085_SWKS3` | `1` | `2026-05-12 09:29:07` | `10:02:50` | `33.7 min` |
| `V8_WINGAMF_02_bodyg005_exitg00275_neu085_SWKS3` | `2` | `2026-05-12 09:29:07` | `10:01:10` | `32.1 min` |
| `V8_WINGAMF_03_bodyg005_exitg00325_neu085_SWKS3` | `3` | `2026-05-12 09:29:07` | `09:59:55` | `30.8 min` |
| `V8_WINGAMF_04_bodyg005_exitg0030_exitneu080_SWKS3` | `4` | `2026-05-12 09:29:07` | `10:00:51` | `31.7 min` |
| `V8_WINGAMF_05_bodyg00475_exitg0030_neu085_SWKS3` | `5` | `2026-05-12 09:29:07` | `10:02:15` | `33.1 min` |
| `V8_WINGAMF_06_bodyg00525_exitg0030_neu085_SWKS3` | `6` | `2026-05-12 09:29:07` | `10:02:23` | `33.3 min` |
| `V8_WINGAMF_07_bodyg005_exitg0030_exitneu090_SWKS3` | `7` | `2026-05-12 09:29:07` | `10:03:43` | `34.6 min` |

Global metrics：

| Run | Body gamma | Exit gamma | Exit neutral | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | `0.0050` | `0.0030` | `0.85` | `34.1903` | `6.5666` | `92.4202` | `0.0083` | current v8 reference |
| `WINGAMF_01` | `0.0050` | `0.0025` | `0.85` | `34.2360` | `6.5493` | `92.4200` | `0.0082` | Rot 好，但 ATE 未过 |
| `WINGAMF_02` | `0.0050` | `0.00275` | `0.85` | `34.2025` | `6.5811` | `92.4200` | `0.0083` | 本批 ATE best，仍未过 reference |
| `WINGAMF_03` | `0.0050` | `0.00325` | `0.85` | `34.2484` | `6.5972` | `92.4199` | `0.0083` | exit gamma 稍强回退 |
| `WINGAMF_04` | `0.0050` | `0.0030` | `0.80` | `34.2689` | `6.5470` | `92.4209` | `0.0082` | exit neutral 过低，ATE 回退 |
| `WINGAMF_05` | `0.00475` | `0.0030` | `0.85` | `34.3131` | `6.5727` | `92.4200` | `0.0082` | body gamma 稍弱回退 |
| `WINGAMF_06` | `0.00525` | `0.0030` | `0.85` | `34.3765` | `6.5734` | `92.4205` | `0.0083` | body gamma 稍强回退 |
| `WINGAMF_07` | `0.0050` | `0.0030` | `0.90` | `34.2801` | `6.5802` | `92.4202` | `0.0083` | exit neutral 过高也回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_wingamf/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `3.774` | `30.994373` |
| `WINGAMF_01` | `34.2360` | `5.981` | `75.610` | `55.467` | `3.758` | `30.989156` |
| `WINGAMF_02` | `34.2025` | `6.012` | `75.574` | `55.437` | `3.783` | `30.989698` |
| `WINGAMF_03` | `34.2484` | `5.873` | `75.572` | `55.435` | `3.796` | `30.988395` |
| `WINGAMF_04` | `34.2689` | `5.891` | `75.646` | `55.493` | `3.753` | `30.996754` |
| `WINGAMF_05` | `34.3131` | `5.767` | `75.652` | `55.495` | `3.786` | `30.992243` |
| `WINGAMF_06` | `34.3765` | `5.918` | `75.759` | `55.596` | `3.791` | `30.996653` |
| `WINGAMF_07` | `34.2801` | `6.154` | `75.671` | `55.529` | `3.786` | `30.990322` |

H2B 结论：

1. 本批没有达到 `<30m`，也没有超过 `WINGAM_03_repeat=34.1903`。
2. `WINGAMF_02=34.2025 / 6.5811` 非常接近 current best，只差 `0.0122m`，但仍不能晋级。
3. exit gamma 的 ATE sweet spot 仍在 `0.0030` 附近；`0.0025/0.00275/0.00325` 都略差。`0.0025` 能改善 Rot / Yaw / FinalErr，但损失 ATE。
4. body gamma 的 sweet spot 仍是 `0.0050`；`0.00475` 和 `0.00525` 都回退，说明 body `5-9` 的强度已经非常窄。
5. exit neutral 的 sweet spot 仍是 `0.85`；`0.80` 和 `0.90` 都回退。
6. auto-window 方向当前仍只能作为诊断，不能替代人工 body/exit。下一步若继续，需要换机制，而不是继续围绕 `body gamma / exit gamma / neutral` 做标量细扫。

当前 v8 best 仍保持：

```text
V8_A1_WINGAM_03_repeat_SWKS3
ATE / Rot = 34.190276 / 6.566600
```

---

## 4. H3：后段姿态 regularizer / chunk15-16 极弱 tri-replay

H2B 说明 `body/exit gamma/neutral` 已经是窄峰。H2 第一批的 auto-window 虽然 ATE 回退，但 Rot / Yaw / FinalErr 明显改善，因此本节尝试把这类“姿态 regularizer”以极弱 gamma 叠加到 `WINGAM_03` 上。

固定 base：

```text
body chunks 5-9:  gamma = 0.005
exit chunks 10-12: gamma = 0.003
pos / neg / neutral = 0.35 / 0.12 / 0.85
risk = update_conflict_energy
branch = w0
SWA = SWKS3-style fixed protocol
```

第一批候选：

| Run | Extra chunks | Extra gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | none | n/a | `34.1903` | `6.5666` | `92.4202` | `0.0083` | reference |
| `POSTREG_01` | `20,30,35` | `0.00025` | `34.4678` | `5.9934` | `92.4132` | `0.0077` | Rot/endpoint 好，但 ATE 回退大 |
| `POSTREG_02` | `20,30,35` | `0.00050` | `34.4509` | `5.9543` | `92.4133` | `0.0077` | 同类，仍回退 |
| `POSTREG_03` | `20,30,35` | `0.00100` | `34.4258` | `5.9894` | `92.4140` | `0.0077` | ATE 仍回退 |
| `POSTREG_04` | `15,16` | `0.00050` | `34.1962` | `6.4377` | `92.4113` | `0.0081` | **强信号，几乎过 best** |
| `POSTREG_05` | `15,16,20` | `0.00050` | `34.3059` | `6.4101` | `92.4081` | `0.0081` | 加 chunk20 后 ATE 回退 |
| `POSTREG_06` | `13-16` | `0.00025` | `34.4694` | `6.4352` | `92.4116` | `0.0082` | 后窗过宽回退 |
| `POSTREG_07` | `13-16` | `0.00050` | `34.4472` | `6.4322` | `92.4110` | `0.0082` | 后窗过宽回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_postreg/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `42.280` | `3.774` | `30.994373` |
| `POSTREG_04` | `34.1962` | `6.088` | `75.022` | `55.070` | `43.658` | `3.732` | `30.878330` |
| `POSTREG_05` | `34.3059` | `6.703` | `75.013` | `55.075` | `44.234` | `3.758` | `30.841625` |

H3 第一批结论：

1. `20/30/35` 这类 reset-like posture chunks 可以显著改善 Rot / Yaw / FinalErr，但 ATE 明显回退，不能作为主线。
2. `chunk15/16` 是强信号：`POSTREG_04` 几乎追平 current best，同时明显改善 Rot、`[200,300)` 和 `[200,400)`。
3. `POSTREG_04` 没有晋级的主要原因是 `[400,600)` 从 `42.280` 回退到 `43.658`。这说明 chunk15/16 是病灶修正与后段连续性的 trade-off 旋钮。
4. 下一步只拆 `chunk15` / `chunk16`，寻找更弱剂量能否保留局部病灶收益而减少后段代价。

### 4.1 chunk15/16 分解

| Run | Extra chunks | Extra gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | none | n/a | `34.1903` | `6.5666` | `92.4202` | `0.0083` | reference |
| `POSTREG_08` | `15,16` | `0.00025` | `34.2154` | `6.4603` | `92.4105` | `0.0082` | 剂量偏弱，未过 |
| `POSTREG_09` | `15,16` | `0.00075` | `34.1908` | `6.4337` | `92.4107` | `0.0082` | ATE 几乎持平，Rot 明显更好 |
| `POSTREG_10` | `15` | `0.00050` | `34.2518` | `6.5013` | `92.4198` | `0.0082` | chunk15 单独不是主因 |
| `POSTREG_11` | `16` | `0.00050` | `34.2017` | `6.4478` | `92.4038` | `0.0081` | chunk16 是主要杠杆 |
| `POSTREG_12` | `15:0.00025,16:0.00050` | mixed | `34.2071` | `6.4769` | `92.4107` | `0.0082` | 未过 |
| `POSTREG_13` | `15:0.00050,16:0.00025` | mixed | `34.2037` | `6.4620` | `92.4108` | `0.0082` | 未过 |
| `POSTREG_14` | `exit gamma 0.00275 + 15,16` | `0.00050` | `34.1958` | `6.4853` | `92.4106` | `0.0082` | 接近但未过 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_postreg15/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `42.280` | `3.774` | `30.994373` |
| `POSTREG_09` | `34.1908` | `6.401` | `75.044` | `55.088` | `43.650` | `3.740` | `30.876584` |
| `POSTREG_11` | `34.2017` | `5.726` | `74.458` | `54.705` | `44.685` | `3.744` | `30.786295` |
| `POSTREG_14` | `34.1958` | `6.450` | `75.019` | `55.082` | `43.641` | `3.781` | `30.873597` |

4.1 结论：

1. `POSTREG_09` 只比 current best 差 `0.0005m`，但 Rot 从 `6.5666` 改到 `6.4337`，说明 chunk15/16 极弱 regularizer 是真实可用方向。
2. `chunk16` 是主要杠杆：`POSTREG_11` 把 `[200,300)` 压到 `74.458`、`[200,400)` 压到 `54.705`，但 `[400,600)` 恶化到 `44.685`。
3. 这明确暴露了新的 trade-off：chunk16 可以显著修主病灶，但会牺牲后段连续性。
4. 下一步应只围绕 chunk16 做 `0.0001-0.0006` 细扫，并测试与 exit gamma 的组合。

---

## 5. H3：chunk16 fine sweep

固定 base 仍为 `WINGAM_03`，只对 chunk16 添加极弱 `update_conflict_energy` tri-replay。目标：保留 `POSTREG_11` 的局部病灶收益，同时减少 `[400,600)` 后段代价。

运行记录：

| Run | Extra setting | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | none | `34.1903` | `6.5666` | `92.4202` | `0.0083` | old v8 best |
| `C16FINE_01` | `c16 gamma 0.00010` | `34.1787` | `6.5190` | `92.4046` | `0.0082` | 过 old best |
| `C16FINE_02` | `c16 gamma 0.00020` | `34.1811` | `6.5193` | `92.4047` | `0.0082` | 过 old best，但不如 0.00010/0.00030 |
| `C16FINE_03` | `c16 gamma 0.00030` | `34.1627` | `6.4986` | `92.4046` | `0.0082` | **当前 v8 new best** |
| `C16FINE_04` | `c16 gamma 0.00040` | `34.2119` | `6.4158` | `92.4032` | `0.0081` | Rot 最好，但 ATE 回退 |
| `C16FINE_05` | `c16 gamma 0.00060` | `34.1970` | `6.5257` | `92.4040` | `0.0082` | 过强后回退 |
| `C16FINE_06` | `exit gamma 0.00275 + c16 0.00025` | `34.2112` | `6.4876` | `92.4034` | `0.0082` | exit gamma 组合未过 |
| `C16FINE_07` | `exit gamma 0.00250 + c16 0.00025` | `34.2385` | `6.4567` | `92.4038` | `0.0081` | exit gamma 更弱也未过 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_c16fine/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `42.280` | `3.774` | `30.994373` |
| `C16FINE_01` | `34.1787` | `6.041` | `74.454` | `54.699` | `44.551` | `3.807` | `30.795558` |
| `C16FINE_02` | `34.1811` | `6.231` | `74.476` | `54.713` | `44.543` | `3.807` | `30.796737` |
| `C16FINE_03` | `34.1627` | `6.218` | `74.445` | `54.693` | `44.549` | `3.791` | `30.794618` |
| `C16FINE_04` | `34.2119` | `5.768` | `74.460` | `54.708` | `44.766` | `3.717` | `30.780640` |
| `C16FINE_05` | `34.1970` | `6.364` | `74.460` | `54.705` | `44.658` | `3.815` | `30.788924` |
| `C16FINE_06` | `34.2112` | `6.351` | `74.451` | `54.716` | `44.775` | `3.770` | `30.776585` |
| `C16FINE_07` | `34.2385` | `6.016` | `74.489` | `54.744` | `44.816` | `3.752` | `30.782204` |

H3 结论：

1. `C16FINE_03` 刷新当前 v8 best：`34.1627 / 6.4986`。相比 `WINGAM_03_repeat=34.1903 / 6.5666`，ATE 改善 `0.0276m`，Rot 改善 `0.0680deg`。
2. 这不是单纯姿态 trade-off：`[200,300)` 从 `75.576` 降到 `74.445`，`[200,400)` 从 `55.428` 降到 `54.693`。
3. 代价仍然存在：`[400,600)` 从 `42.280` 回退到 `44.549`，说明 chunk16 的后段连续性副作用没有完全解决。
4. chunk16 gamma 的 ATE sweet spot 在 `0.00030` 附近；`0.00010/0.00020` 有正信号，`0.00040/0.00060` 开始偏向姿态改善但 ATE 回退。
5. 降低 exit gamma 到 `0.00275/0.00250` 不能抵消 chunk16 的后段代价，反而 ATE 回退。
6. 下一步如果继续，应围绕 `C16FINE_03` 做两条：
   - 尝试只在 chunk16 的 late/layer12/head0 做 regularizer，减少 `[400,600)` 代价；
   - 或在 chunks `17-18` 做极弱 positive/neutral recovery，专门修 chunk16 之后的 handoff。

当前 v8 best 更新为：

```text
V8_C16FINE_03_c16g00030_SWKS3
ATE / Rot = 34.162704 / 6.498617
RPE t / r = 92.404601 / 0.008161
```

---

## 8. H3：head-routed tri-replay 接线复查

动机：

v7/v8 的 `update_conflict_energy` audit 都提示 chunk5 的风险集中在 `layer12/head0`。因此本批尝试在当前 v8 best 附近启用 `TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES`，验证 per-head routed tri-replay 是否能在不扰动全层 continuity 的情况下继续降低 ATE。

固定主线：

```text
objective = tri_replay
risk = update_conflict_energy
branch = w0
pos_frac / neg_frac / neutral = 0.35 / 0.12 / 0.85
SWA = SWKS3-style fixed protocol
RESET_EVERY = 5
```

主参考：

```text
C16ROLE_01:
    chunks 5-9  gamma=0.005
    chunks 10-12 gamma=0.003
    chunk16 gamma=0.0003
    chunk16 role = pos0.35 / neg0.08 / neutral0.85
```

运行记录：

| Run | Head routes | Base | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| `C16ROLE_01` | none | current best | `34.1583` | `6.5327` | `92.4059` | `0.0082` | reference |
| `V8_LHROUTE_01_c16role_l12h0_SWKS3` | `12:0` | C16ROLE | `34.1583` | `6.5327` | `92.4059` | `0.0082` | 与 reference 完全重合 |
| `V8_LHROUTE_02_c16role_l12h1_SWKS3` | `12:1` | C16ROLE | `34.1583` | `6.5327` | `92.4059` | `0.0082` | 与 reference 完全重合 |
| `V8_LHROUTE_03_c16role_l5h0_l9h1_l12h0_SWKS3` | `5:0;9:1;12:0` | C16ROLE | `34.1583` | `6.5327` | `92.4059` | `0.0082` | 与 reference 完全重合 |
| `V8_LHROUTE_04_c16role_l5h0_l12h0_SWKS3` | `5:0;12:0` | C16ROLE | `34.1583` | `6.5327` | `92.4059` | `0.0082` | 与 reference 完全重合 |
| `V8_LHROUTE_05_c16role_l9h1_l12h0_SWKS3` | `9:1;12:0` | C16ROLE | `34.1583` | `6.5327` | `92.4059` | `0.0082` | 与 reference 完全重合 |
| `WINGAM_03` | none | WINGAM | `34.1903` | `6.5666` | `92.4202` | `0.0083` | reference |
| `V8_LHROUTE_06_wingam_l12h0_SWKS3` | `12:0` | WINGAM | `34.1903` | `6.5666` | `92.4202` | `0.0083` | 与 WINGAM 完全重合 |

诊断：

```text
log dir = results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/_logs_lhroute_20260512_143409/
```

关键发现：

1. `hmc_config.yaml` 中能看到 `ttt_write_gradient_reversal_head_routes` 已经被写入。
2. 但对应 `hmc_state_hash.jsonl` 中没有出现 `ttt_head_routed_*` debug 字段。
3. 所有 head-route run 与各自 reference 完全重合，说明本批不是“head routing 策略失败”，而是当前 head-route hook 在 chunk-gamma-map / tri-replay 路径下没有实际生效，或生效路径没有被 debug/commit 捕获。

结论：

1. H3 的 per-head routing 不能按本批结果下机制结论，因为实验实际上是 no-op。
2. 当前 v8 best 仍保持：

```text
V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3
ATE / Rot = 34.1583 / 6.5327
```

3. 下一步不继续扫 `HEAD_ROUTES`。先改用已经在 v7 中验证过会生效的 layer-gamma routing，做一小批 H3 layer-level routing；若 layer routing 有正信号，再回头修 head-route hook。

---

## 9. H3：layer12 gamma routing 小矩阵

8 节发现 `HEAD_ROUTES` 在当前 chunk-gamma-map / tri-replay 路径下是 no-op。因此本批改用已经验证会进入配置的 layer-level gamma override，先在 `WINGAM_03` 主窗口上测试 layer12 的 high-conflict cue 是否能作为有效 routing。

固定：

```text
base = WINGAM_03
chunks 5-9  gamma = 0.005
chunks 10-12 gamma = 0.003
tri replay = pos 0.35 / neg 0.12 / neutral 0.85
risk = update_conflict_energy
branch = w0
SWA = SWKS3-style fixed protocol
RESET_EVERY = 5
```

候选：

| Run | Layer gamma override | 目的 |
|---|---|---|
| `V8_LGROUTE_01_wingam_l12g0025_SWKS3` | `12:0.0025` | layer12 比 exit baseline 稍弱 |
| `V8_LGROUTE_02_wingam_l12g0035_SWKS3` | `12:0.0035` | layer12 接近 exit/body 中间强度 |
| `V8_LGROUTE_03_wingam_l12g0045_SWKS3` | `12:0.0045` | layer12 接近 body 强度 |
| `V8_LGROUTE_04_wingam_l12g0060_SWKS3` | `12:0.0060` | layer12 强 boost 风险测试 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V8_LGROUTE_01_wingam_l12g0025_SWKS3` | `2026-05-12 15:16` | `15:48:28` | `~32.5 min` |
| `V8_LGROUTE_02_wingam_l12g0035_SWKS3` | `2026-05-12 15:16` | `15:48:26` | `~32.4 min` |
| `V8_LGROUTE_03_wingam_l12g0045_SWKS3` | `2026-05-12 15:16` | `15:47:07` | `~31.1 min` |
| `V8_LGROUTE_04_wingam_l12g0060_SWKS3` | `2026-05-12 15:16` | `15:46:27` | `~30.7 min` |

Global metrics：

| Run | Layer12 gamma | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `WINGAM_03` | base chunk map | `34.1903` | `6.5666` | `92.4202` | `0.0083` | WINGAM reference |
| `C16ROLE_01` | base + chunk16 role | `34.1583` | `6.5327` | `92.4059` | `0.0082` | current v8 best |
| `LGROUTE_01` | `0.0025` | `34.2701` | `6.6014` | `92.4206` | `0.0083` | 本批 ATE best，但未过 WINGAM/C16ROLE |
| `LGROUTE_02` | `0.0035` | `34.3436` | `6.5526` | `92.4211` | `0.0082` | 回退 |
| `LGROUTE_03` | `0.0045` | `34.2702` | `6.5842` | `92.4205` | `0.0083` | 接近 LGROUTE_01，但未过 |
| `LGROUTE_04` | `0.0060` | `34.3825` | `6.5684` | `92.4210` | `0.0083` | 过强，回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_lgroute/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `WINGAM_03` | `34.1903` | `6.195` | `75.576` | `55.428` | `3.774` | `30.994373` |
| `C16ROLE_01` | `34.1583` | `6.249` | `74.522` | `54.740` | `3.803` | `30.809795` |
| `LGROUTE_01` | `34.2701` | `6.108` | `75.652` | `55.496` | `3.797` | `30.994731` |
| `LGROUTE_02` | `34.3436` | `5.709` | `75.780` | `55.591` | `3.761` | `31.003499` |
| `LGROUTE_03` | `34.2702` | `5.879` | `75.597` | `55.464` | `3.796` | `30.995120` |
| `LGROUTE_04` | `34.3825` | `6.059` | `75.787` | `55.610` | `3.784` | `31.000488` |

9 节结论：

1. 本批没有达到 `<30m`，也没有超过当前 v8 best `C16ROLE_01=34.1583 / 6.5327`。
2. `layer12` gamma override 确实进入配置，但作为单层 scalar routing 没有带来收益。最好的 `LGROUTE_01=34.2701` 仍比 `WINGAM_03` 差 `0.0798m`，比 `C16ROLE_01` 差 `0.1118m`。
3. `layer12` 加强到 `0.0035/0.0045/0.0060` 都没有改善 `[200,300)`；说明 v8 当前收益不是靠单个 high-conflict layer 的强弱，而是 chunks `5-12` 多层弱 conflict field 与 chunk16 post-region weak correction 的组合。
4. `LGROUTE_02` 有较好的 FinalErr / Yaw，但 ATE 和病灶段明显回退，不能晋级。
5. H3 layer/head scalar routing 收口：head-route 当前 no-op，layer12 scalar routing 有效但不优。下一步应转向 H4 的 direction-alignment / trajectory-aware cue，而不是继续做 layer12 标量微扫。

当前 v8 best 仍保持：

```text
V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3
ATE / Rot = 34.1583 / 6.5327
RPE t / r = 92.4059 / 0.0082
[200,300) = 74.522
```

---

## 10. H4：direction-alignment / commit handoff 首批

本节接 9 节结论继续推进 v8 plan。9 节已显示 head-route 当前是 no-op、layer12 scalar routing 不优，因此 H4 先测试两个已有结构 hook：

1. `native_delta_gate`：把 controlled TTT correction 与 native continuity delta 做 direction alignment / cap。
2. `commit_filter`：只在 chunk16 handoff 处按 `update_conflict_energy` 缩放 long commit，尝试缓解 chunk16 修病灶但伤后段连续性的副作用。

固定 base：

```text
base = C16ROLE_01
chunks 5-9   gamma = 0.005
chunks 10-12 gamma = 0.003
chunk16      gamma = 0.0003
chunk16 role = pos0.35 / neg0.08 / neu0.85
risk = update_conflict_energy
branch = w0
SWA = SWKS3-style fixed protocol
RESET_EVERY = 5
```

当前参考：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | `[200,300)` | `[200,400)` | `[400,600)` | FinalErr | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | `34.1903` | `6.5666` | `92.4202` | `0.0083` | `75.576` | `55.428` | `42.280` | `6.195` | v7/v8 handoff reference |
| `C16ROLE_01` | `34.1583` | `6.5327` | `92.4059` | `0.0082` | `74.522` | `54.740` | `44.369` | `6.249` | current v8 best |

### 10.1 native-delta direction gate

运行记录：

| Run | Gate mode | Params | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| `V8_H4_DGATE_01R` | `cosine_soft` | `min_cos=0.25, fallback=0.50` | `34.4300` | `6.4854` | `92.4091` | `0.0082` | Rot 改善，但 ATE 明显回退 |
| `V8_H4_DGATE_02R` | `cosine_soft` | `min_cos=0.50, fallback=0.50` | `34.6002` | `6.5129` | `92.4113` | `0.0082` | gate 更强，ATE 更差 |
| `V8_H4_DGATE_03R` | `cosine_cap` | `min_cos=0.25, fallback=0.50, cap=0.75` | `34.9699` | `6.5507` | `92.4207` | `0.0082` | 大幅回退 |
| `V8_H4_DGATE_04R` | `cap` | `cap=0.75` | `34.9699` | `6.5507` | `92.4207` | `0.0082` | 与 cosine_cap 重合式回退 |

诊断：

- 本批不是 no-op。debug 中可见 `ttt_write_native_delta_gate_applied=True`，例如 `cosine_soft` 的全层 gate scale mean 约 `0.93`。
- 但 native delta gate 会把 tri-replay 的主 ATE 收益一起压掉。最好的 `DGATE_01R=34.4300` 仍比 `C16ROLE_01=34.1583` 差 `0.2717m`。
- 结论：direction alignment gate 适合当 Rot regularizer 线索，不适合作当前主 TTT write policy。

### 10.2 chunk16 commit-filter handoff

固定只作用 chunk16：

```text
TTT_WRITE_COMMIT_FILTER_CHUNKS = 16
TTT_WRITE_COMMIT_FILTER_BRANCH_MASK = 0
TTT_WRITE_COMMIT_FILTER_RISK_SOURCE = update_conflict_energy
```

Global metrics：

| Run | Mode | Stat / scale | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| `C16ROLE_01` | none | reference | `34.1583` | `6.5327` | `92.4059` | `0.0082` | current v8 best |
| `CFILTER_01` | `old_decay_by_risk` | mean, `1.0 - 0.25*risk`, min `0.75` | `34.1926` | `6.5326` | `92.4050` | `0.0082` | 回退到 WINGAM 附近 |
| `CFILTER_02` | `old_decay_by_risk` | q90, `1.0 - 0.25*risk`, min `0.75` | `34.1805` | `6.4927` | `92.4051` | `0.0082` | Rot 最好，但 ATE 未过 |
| `CFILTER_03` | `native_to_candidate_by_risk` | q90, `1.0 - 0.25*risk`, min `0.75` | `34.1672` | `6.5284` | `92.4056` | `0.0082` | 本批 ATE best，但仍未过 C16ROLE |
| `CFILTER_04` | `old_decay_by_risk` | q90, `1.0 - 0.15*risk`, min `0.85` | `34.1709` | `6.5030` | `92.4051` | `0.0082` | 接近，但未过 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_h4_cfilter/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `WINGAM_03` | `34.1903` | `6.195` | `75.576` | `55.428` | `42.280` | `3.774` | `30.994373` |
| `C16ROLE_01` | `34.1583` | `6.249` | `74.521` | `54.740` | `44.369` | `3.803` | `30.809795` |
| `CFILTER_02` | `34.1805` | `6.001` | `74.535` | `54.751` | `44.450` | `3.773` | `30.803459` |
| `CFILTER_03` | `34.1672` | `6.336` | `74.531` | `54.747` | `44.389` | `3.808` | `30.808110` |

10 节结论：

1. v8 plan 还没有最终完成，因为 H4 只完成了首批 direction-alignment / commit-filter 对照，还没形成强新策略；但现有 H4 两条主 hook 已有第一轮结果。
2. `native_delta_gate` 明确回退，不继续作为主线。
3. chunk16 commit-filter 有弱 regularizer 信号：`CFILTER_02` 改善 Rot / FinalErr / Yaw，`CFILTER_03` ATE 接近 current best，但二者都没有超过 `C16ROLE_01`。
4. commit-filter 没有解决 chunk16 的核心 trade-off：`[200,300)` 仍接近 `74.53`，但 `[400,600)` 仍在 `44.39-44.45`，没有回到 `WINGAM_03=42.280`。
5. 当前 v8 best 保持：

```text
V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3
ATE / Rot = 34.1583 / 6.5327
RPE t / r = 92.4059 / 0.0082
[200,300) = 74.522
```

下一步建议：不继续扫 DGATE / commit-filter scalar。若继续 H4，应转向更明确的 trajectory-aware objective，例如在 chunk16 后的 `[400,600)` recovery 上做显式 post-window positive/neutral recovery，或修复 head-route hook 后做真正 per-head route，而不是再做全层 scalar gate。

---

## 11. H4：post-window positive / neutral handoff 首批

接第 10 节。`native_delta_gate` 和 chunk16 commit filter 都没有超过当前 best，但它们说明 chunk16 附近确实存在 Rot / FinalErr 可调空间。本批继续保留当前 v8 best 的主结构，只在 chunk16 之后加入极弱 post-window positive / neutral handoff，测试是否能修复 `C16ROLE_01` 的后段 drift。

当前 reference：

```text
V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3
ATE / Rot = 34.1583 / 6.5327
[200,300) = 74.522
[400,600) = 44.369
```

固定主线：

```text
TTT_WRITE_GRADIENT_REVERSAL_MODE = tri_replay
TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE = update_conflict_energy
branch = w0
chunks 5-9   gamma = 0.005, role = 0.35 / 0.12 / 0.85
chunks 10-12 gamma = 0.003, role = 0.35 / 0.12 / 0.85
chunk 16     gamma = 0.0003, role = 0.35 / 0.08 / 0.85
SWA = SWKS3-style fixed protocol
RESET_EVERY = 5
```

新增候选：

| Run | Added post-window policy | 目的 |
|---|---|---|
| `V8_H4_POSTWIN_01_c17c18_pos045_neg002_neu090_SWKS3` | chunks `17,18`: gamma `0.0001`, role `0.45/0.02/0.90` | 极弱 positive / neutral handoff |
| `V8_H4_POSTWIN_02_c17c18_g0003_pos035_neg006_neu085_SWKS3` | chunks `17,18`: gamma `0.0003`, role `0.35/0.06/0.85` | 更接近 chunk16 的弱 conflict correction |
| `V8_H4_POSTWIN_03_c18to20_pos045_neg002_neu090_SWKS3` | chunks `18-20`: gamma `0.0001`, role `0.45/0.02/0.90` | 更靠后的 positive / neutral recovery |
| `V8_H4_POSTWIN_04_c16pos045_c17c18_recover_SWKS3` | chunk16 role `0.45/0.05/0.90` + chunks `17,18` recovery | 同时让 chunk16 更 positive，并加后窗恢复 |

运行记录：

| Run | Done | 备注 |
|---|---|---|
| `POSTWIN_01` | `2026-05-12` | full KITTI01 complete |
| `POSTWIN_02` | `2026-05-12` | full KITTI01 complete |
| `POSTWIN_03` | `2026-05-12` | full KITTI01 complete |
| `POSTWIN_04` | `2026-05-12` | full KITTI01 complete |

Global metrics：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---|
| `C16ROLE_01` | `34.1583` | `6.5327` | `92.4059` | `0.0082` | current v8 best |
| `POSTWIN_01` | `34.1864` | `6.4239` | `92.3974` | `0.0081` | Rot/RPE 明显好，但 ATE 回退 |
| `POSTWIN_02` | `34.2275` | `6.4987` | `92.3976` | `0.0082` | ATE/Rot 均未过 best |
| `POSTWIN_03` | `34.2195` | `6.4031` | `92.4003` | `0.0080` | Rot 最好，但 ATE 回退 |
| `POSTWIN_04` | `34.1799` | `6.4311` | `92.3986` | `0.0081` | 本批 ATE best，但未过 C16ROLE_01 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_h4_postwin/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | `[400,500)` | `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C16ROLE_01` | `34.1583` | `6.249` | `74.521` | `54.740` | `53.020` | `44.369` | `3.803` | `30.809795` |
| `POSTWIN_01` | `34.1864` | `5.637` | `73.937` | `54.379` | `54.948` | `45.506` | `3.747` | `30.706786` |
| `POSTWIN_02` | `34.2275` | `6.122` | `73.973` | `54.403` | `54.921` | `45.592` | `3.821` | `30.706184` |
| `POSTWIN_03` | `34.2195` | `6.058` | `74.251` | `54.579` | `54.155` | `45.210` | `3.757` | `30.742524` |
| `POSTWIN_04` | `34.1799` | `6.089` | `74.040` | `54.444` | `54.674` | `45.357` | `3.763` | `30.720168` |

11 节结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过当前 v8 best `C16ROLE_01=34.1583 / 6.5327`。
2. post-window positive / neutral handoff 有明确姿态收益：`POSTWIN_01/03/04` 把 Rot 降到 `6.40-6.43`，RPE t 也降到 `92.397-92.400`。
3. 但这个收益是用后段 ATE 换来的：`[400,600)` 从 `C16ROLE_01=44.369` 恶化到 `45.210-45.592`。
4. 同时 `[200,300)` 局部病灶反而更低，best `POSTWIN_01=73.937`，说明后窗 handoff 会改变全局 Sim3 / trajectory balance，而不是单纯修后段。
5. 机制判断：chunk16 后窗 positive / neutral replay 更像 orientation / endpoint regularizer；它不能恢复 `C16ROLE_01` 的后段 drift，反而把 `[400,600)` 推坏。
6. 因此不继续做 post-window scalar 微扫。下一阶段应转向更直接的 global-drift / trajectory-aware selection，而不是继续在 chunks 17-20 加弱 replay。

当前 v8 best 保持不变：

```text
V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3
ATE / Rot = 34.1583 / 6.5327
RPE t / r = 92.4059 / 0.0082
```

---

## 12. 新探索计划：trajectory-aware TTT 写入，不再只靠 chunk-local replay cue

截至第 11 节，v8 已经系统验证了几条自然后续：

| 方向 | 结果 | 判断 |
|---|---|---|
| chunk16 role routing | `C16ROLE_01=34.1583 / 6.5327` | 当前 best，说明后段 chunk16 有用 |
| layer/head route 复查 | 未形成有效 route | 当前 head-route 不是主线 |
| native-delta gate | best `34.4300 / 6.4854` | Rot 好但 ATE 大幅回退 |
| chunk16 commit filter | best `34.1672 / 6.5284` | 接近但未过 best |
| post-window recovery | best `34.1799 / 6.4311` | 姿态收益明显，ATE 未过 best |

当前卡点不是 TTT cue 不够强，而是 **chunk-local cue 与长期 trajectory objective 不一致**：

```text
- 加强局部 / 后窗 correction 可以压 [200,300) 或 Rot；
- 但经常把 [400,600) 或 global ATE 推坏；
- C16ROLE_01 的成功来自一个很窄的 balance，而不是单项指标最优。
```

下一阶段不建议继续做：

```text
1. 不继续全序列 tri-replay；成本高且姿态代价大。
2. 不继续 post-window 17-20 scalar 微扫；已确认是 Rot/ATE trade-off。
3. 不继续 native-delta gate / commit-filter 小扫；信号弱，无法过 best。
4. 不使用 freeze 作为策略；仍只保留为 causal diagnostic。
```

建议的新 H5 路线：

### H5-A. trajectory-aware run selection / cheap surrogate

目标：先离线统计已有 runs 的 segment trade-off，找到能预测 final ATE 的 surrogate，而不是盲跑 full KITTI01。

候选 surrogate：

```text
score = ATE_global
      + a * max(0, segment_400_600 - reference_400_600)
      + b * yaw_rmse
      + c * |sim3_scale - target_scale|
```

用途：

```text
- 用已有 WINGAM / C16ROLE / CFILTER / POSTWIN runs 拟合哪些局部指标对应 global ATE；
- 后续新配置先跑 short/full-diagnostics subset，只晋级少量 full KITTI01。
```

### H5-B. chunk16 selective long residual gate

当前 chunk16 有用，但后窗 replay 失败。下一步应只在 chunk16 内做更细的 commit selection：

```text
base = C16ROLE_01
only chunk16:
    keep positive / neutral residual if it improves native direction alignment
    suppress residual if it increases downstream [400,600) drift proxy
```

首批 full 候选最多 4 条：

| Run | Chunk16 policy | 目的 |
|---|---|---|
| `V8_H5_C16LONG_01` | chunk16 old_decay q90, very soft min `0.90` | 比第 10 节更温和的 long gate |
| `V8_H5_C16LONG_02` | chunk16 native blend q90, min `0.90` | 避免把 C16 useful correction 全拉回 |
| `V8_H5_C16LONG_03` | chunk16 only neutral `0.80`, gamma unchanged | 测是否只需降低 chunk16 neutral inertia |
| `V8_H5_C16LONG_04` | chunk16 pos `0.45`, neg `0.08`, neutral `0.80` | 更 positive 但压 neutral |

### H5-C. downstream-aware exit window split

post-window 说明 chunks 17-20 不适合统一 positive recovery，但也改善了 Rot。下一步不是继续加后窗，而是拆成：

```text
chunk16: 纠偏
chunk17: very weak handoff or none
chunk18-20: no tri-replay，保留 native continuity
```

优先测试只加 chunk17，而不是 17-18 / 18-20：

| Run | Policy | 目的 |
|---|---|---|
| `V8_H5_HANDOFF_01` | chunk17 only gamma `0.0001`, role `0.45/0.02/0.90` | 最小 handoff |
| `V8_H5_HANDOFF_02` | chunk17 only gamma `0.0003`, role `0.35/0.06/0.85` | 稍强 conflict handoff |

晋级标准：

```text
必须同时满足：
    ATE < 34.1583
    [400,600) <= 44.37 或 FinalErr/Yaw 有显著补偿
否则不替换 C16ROLE_01。
```

---

## 13. drift mode 诊断：突发 chunk 漂移 vs 缓慢累积漂移

用户问题：

```text
诊断一下是突然的 chunk 间的漂移厉害，还是整体慢慢积累的漂移。
```

诊断目录：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/drift_mode_diagnosis_v8_current/
```

输出：

```text
summary.csv
top_jumps.csv
chunk_trends.csv
```

诊断口径：

```text
chunk-boundary sudden drift:
    相邻 frame 的 aligned residual vector jump = ||e_t - e_{t-1}||
    重点看 chunk boundary / reset boundary 是否贡献主要 residual variation

gradual drift:
    看每个 chunk 内 start_error -> end_error 的趋势
    以及 [200,300), [400,600) 等长窗口是否持续高误差
```

关键统计：

| Run | ATE | Max residual jump | Top jump frame | Boundary? | Mean boundary jump | Mean non-boundary jump | Boundary TV share | Reset boundary TV share | `[200,300)` | `[400,600)` |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `B0` | `36.4161` | `11.1268` | `696` | yes | `3.4187` | `0.5941` | `0.1669` | `0.0180` | `77.831` | `47.825` |
| `WINGAM_03` | `34.1903` | `10.9149` | `696` | yes | `3.5164` | `0.5867` | `0.1726` | `0.0190` | `75.576` | `42.280` |
| `C16FINE_03` | `34.1627` | `10.8794` | `696` | yes | `3.5533` | `0.5847` | `0.1746` | `0.0180` | `74.445` | `44.549` |
| `C16ROLE_01` | `34.1583` | `10.9315` | `696` | yes | `3.5483` | `0.5840` | `0.1746` | `0.0182` | `74.521` | `44.369` |
| `POSTWIN_01` | `34.1864` | `11.0810` | `696` | yes | `3.5099` | `0.5808` | `0.1738` | `0.0182` | `73.937` | `45.506` |
| `POSTWIN_04` | `34.1799` | `10.9267` | `696` | yes | `3.4871` | `0.5803` | `0.1730` | `0.0182` | `74.040` | `45.357` |

结论：

1. chunk boundary 的突发 jump 是真实存在的：boundary jump 均值约 `3.5m`，non-boundary 均值约 `0.58m`，单次事件强度约 `6x`。
2. 但它不是 overall ATE 的主因：chunk boundary 只贡献总 residual-vector variation 的约 `17%`，reset boundary 只有约 `1.8-1.9%`。
3. 主病灶更像 **窗口级缓慢积累 / 高误差平台**：chunk5-8 误差在 chunk 内持续爬升，然后 chunk8-10 维持高位再回落。
4. 典型趋势：

```text
WINGAM_03:
    chunk5  [145,173): 18.60 -> 34.44, +15.84
    chunk6  [174,202): 38.42 -> 58.09, +19.67
    chunk7  [203,231): 57.54 -> 76.40, +18.86
    chunk8  [232,260): high plateau
    chunk9  [261,289): 89.56 -> 64.48, -25.08
    chunk10 [290,318): 61.89 -> 18.13, -43.77

C16ROLE_01:
    chunk6  [174,202): 38.74 -> 57.88, +19.14
    chunk7  [203,231): 57.32 -> 75.62, +18.31
    chunk8  [232,260): high plateau
    chunk9  [261,289): 88.25 -> 62.88, -25.37
    chunk10 [290,318): 60.29 -> 16.38, -43.91
```

机制判断：

```text
不是“某一个 chunk 边界突然炸掉”主导；
而是 chunks 5-8 逐步把 residual 推上高平台，chunks 9-10 再回落。
chunk boundary jump 是局部不连续症状，但 current TTT 写入主问题是窗口级漂移状态没有被显式建模。
```

因此后续策略不能只做 boundary smoothing / reset smoothing。真正需要的是 window-level drift-state controller：在 chunks 5-12 内追踪累积漂移方向、neutral inertia、exit handoff，而不是继续只调单 chunk token prior。

---

## 14. H5：chunk16 long gate 与 chunk17 handoff 补跑

本节补完第 12 节 H5-B / H5-C 预案。base 为当前 v8 best：

```text
V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3
ATE / Rot = 34.1583 / 6.5327
```

固定主线：

```text
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
SWA = SWKS3-style fixed protocol
TTT mode = tri_replay
risk = update_conflict_energy
branch = w0
chunks 5-9:  gamma 0.005, role 0.35/0.12/0.85
chunks 10-12: gamma 0.003, role 0.35/0.12/0.85
chunk16 base: gamma 0.0003, role 0.35/0.08/0.85
```

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V8_H5_C16LONG_01_c16_olddecay_q90_vsoft_SWKS3` | `2026-05-12 21:40:24` | `22:13:09` | `32.8 min` |
| `V8_H5_C16LONG_02_c16_native_q90_vsoft_SWKS3` | `2026-05-12 21:40:24` | `22:13:20` | `32.9 min` |
| `V8_H5_C16LONG_03_c16_neu080_SWKS3` | `2026-05-12 21:40:24` | `22:12:49` | `32.4 min` |
| `V8_H5_C16LONG_04_c16_pos045_neg008_neu080_SWKS3` | `2026-05-12 21:40:24` | `22:14:04` | `33.7 min` |
| `V8_H5_HANDOFF_01_c17g0001_pos045_neg002_neu090_SWKS3` | `2026-05-12 21:40:24` | `22:14:35` | `34.2 min` |
| `V8_H5_HANDOFF_02_c17g0003_pos035_neg006_neu085_SWKS3` | `2026-05-12 21:40:24` | `22:13:05` | `32.7 min` |

Global metrics：

| Run | Policy | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `C16ROLE_01` | reference | `34.1583` | `6.5327` | `92.4059` | `0.0082` | current v8 best |
| `H5_C16LONG_01` | c16 old_decay q90, base `1.0`, gain `0.10`, min `0.90` | `34.2067` | `6.4909` | `92.4047` | `0.0082` | Rot/FinalErr 好，ATE 回退 |
| `H5_C16LONG_02` | c16 native blend q90, gain `-0.10`, min `0.90` | `34.1863` | `6.5175` | `92.4051` | `0.0082` | 接近但未过 |
| `H5_C16LONG_03` | c16 neutral `0.80` | `34.1881` | `6.5013` | `92.4050` | `0.0082` | Rot 好，ATE 回退 |
| `H5_C16LONG_04` | c16 pos `0.45`, neg `0.08`, neutral `0.80` | `34.1722` | `6.4946` | `92.4057` | `0.0082` | 本批 ATE best，但未过 |
| `H5_HANDOFF_01` | c17 gamma `0.0001`, role `0.45/0.02/0.90` | `34.1731` | `6.4863` | `92.3997` | `0.0082` | RPE/Rot 好，ATE 未过 |
| `H5_HANDOFF_02` | c17 gamma `0.0003`, role `0.35/0.06/0.85` | `34.2022` | `6.4599` | `92.3992` | `0.0082` | Rot 最好，ATE 回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_h5/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C16ROLE_01` | `34.1583` | `6.249` | `74.521` | `54.740` | `44.369` | `3.803` | `30.809795` |
| `H5_C16LONG_01` | `34.2067` | `5.942` | `74.534` | `54.753` | `44.541` | `3.776` | `30.798253` |
| `H5_C16LONG_02` | `34.1863` | `6.108` | `74.513` | `54.737` | `44.491` | `3.803` | `30.801948` |
| `H5_C16LONG_03` | `34.1881` | `6.050` | `74.517` | `54.740` | `44.494` | `3.786` | `30.800747` |
| `H5_C16LONG_04` | `34.1722` | `6.291` | `74.554` | `54.764` | `44.408` | `3.791` | `30.807591` |
| `H5_HANDOFF_01` | `34.1731` | `6.364` | `74.116` | `54.490` | `45.185` | `3.805` | `30.735263` |
| `H5_HANDOFF_02` | `34.2022` | `6.154` | `74.085` | `54.472` | `45.394` | `3.768` | `30.723597` |

14 节结论：

1. H5-B / H5-C 没有达到 `<30m`，也没有超过 `C16ROLE_01=34.1583`。
2. `H5_C16LONG_04=34.1722` 是本批 ATE best，但仍比 `C16ROLE_01` 差 `0.0139m`。它说明 chunk16 更 positive + lower neutral 有弱信号，但不足以晋级。
3. very soft commit gate 能改善 Rot / FinalErr，但会损 ATE 和 `[400,600)`；说明 chunk16 useful correction 不能靠 post-commit scalar gate 简单筛出。
4. chunk17 handoff 两条都显著降低 `[200,300)` 到 `74.1m` 附近，但 `[400,600)` 从 `44.369` 恶化到 `45.185/45.394`，属于局部病灶换后段 drift，不是成功策略。
5. 至此，v8 plan 的 full-run H5 候选已经补完；当前 best 保持：

```text
V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3
ATE / Rot = 34.1583 / 6.5327
```

---

## 15. H6 新探索计划：从 token / chunk 写入转向 window-level drift state

H1-H5 的共同边界：

```text
1. update_conflict_energy + tri_replay 是有效主线；
2. fixed chunks 5-12 + chunk16 weak correction 已经接近当前局部最优；
3. chunk16 / chunk17 的局部 tweak 只能在 ATE、Rot、[200,300)、[400,600) 之间重新分配误差；
4. drift 诊断显示主问题不是单个 boundary jump，而是 chunks 5-8 的累积高误差平台。
```

因此下一轮不应继续做：

```text
- chunk16 / chunk17 scalar micro sweep
- post-window 17-20 positive recovery
- layer/head scalar routing
- boundary/reset smoothing
```

新的核心假设：

```text
当前 TTT write controller 只看 chunk 内 token/update risk；
但 failure 是 window-level drift state：
    chunks 5-8 累积漂移
    chunks 9-12 回落 / handoff
    chunk16 影响后段 scale / orientation trade-off

所以 controller 需要一个 reset-window 内的 drift-state signal，
而不是继续只给单个 chunk 设置 gamma / neutral。
```

### H6-A. 离线 drift-state attribution

先不跑 full model，做诊断工具：

```text
tools/ttt_window_drift_state_audit.py
```

输入：

```text
trajectory diagnostics:
    per_frame_errors.csv
    chunk_errors.csv

TTT debug:
    hmc_state_hash.jsonl
    hmc_probe_summary.jsonl
    hook_effect_summary.jsonl
```

输出每个 chunk 的：

```text
drift velocity:
    residual vector start -> end
    dominant axis x/z
    within-chunk slope

TTT write state:
    update_conflict_energy mean/q90
    tri-replay positive/neutral/negative mass
    commit delta norm
    native-vs-controlled alignment

downstream cost:
    correlation with [200,300), [200,400), [400,600), FinalErr
```

目标：

```text
找出哪些 TTT write signals 预测的是局部病灶下降，
哪些预测的是后段 [400,600) drift 恶化。
```

### H6-B. oracle upper-bound diagnostic，不作为正式策略

用 GT-aligned residual direction 做 2-3 条 oracle full run，只回答一个问题：

```text
如果知道 window drift direction，TTT write 是否还有足够空间低于 34m / 接近 30m？
```

候选：

| Run | 机制 | 目的 |
|---|---|---|
| `V8_H6_ORACLE_01` | chunks 5-12 的 gamma / neutral 按 residual slope sign 分段 | 测 window-level drift-state 是否有上界收益 |
| `V8_H6_ORACLE_02` | chunk16 correction 只在不恶化 oracle `[400,600)` proxy 时保留 | 测 chunk16 是否可被正确 gate |

若 oracle 也不能明显过 `34.1583`，说明当前 TTT write surface 已接近上限，应转向 read-side / pose-scale failure。若 oracle 大幅改善，再做 no-GT self cue 近似。

### H6-C. no-GT drift-state cue

不使用 GT，构造 deployment-safe proxy：

```text
drift_state_proxy =
    overlap source replacement residual
    + native-vs-controlled pose increment disagreement
    + reset-window cumulative TTT delta direction
```

第一批正式候选只做 4 条：

| Run | 机制 | 目的 |
|---|---|---|
| `V8_H6_DRIFT_01` | chunks 5-12：若 drift proxy 持续同向累积，则 neutral_lambda 从 `0.85` 降到 `0.75` | 直接抑制窗口级惯性 |
| `V8_H6_DRIFT_02` | chunks 9-12：drift proxy 回落时提前降低 gamma 到 `0.0025` | 改 exit handoff，不碰 chunk16 |
| `V8_H6_DRIFT_03` | chunk16：只有 drift proxy 与 `[400,600)` safe direction 一致时启用 `0.0003` | 防止 chunk16 用局部收益换后段 drift |
| `V8_H6_DRIFT_04` | combine `DRIFT_02 + DRIFT_03` | 测 exit handoff 与 chunk16 gate 是否可叠加 |

晋级标准：

```text
primary:
    ATE < 34.1583

strong relative:
    ATE <= 34.00
    且 [400,600) <= 44.37

diagnostic:
    即使 ATE 未过，也必须证明 drift proxy 能预测
        [200,300) 改善 vs [400,600) 恶化
```

H6 的关键不是再找一个更细的 chunk16 标量，而是让 TTT write policy 第一次显式感知 “本 reset window 正在往哪个方向漂”。这与 13 节 drift 诊断一致：当前失败是慢慢积累的窗口漂移，不能只靠 chunk boundary 或单点 handoff 修掉。

---

## 16. H6-A/H6-B 实际执行：window drift-state attribution 与 oracle upper-bound

执行时间：`2026-05-12 22:47-23:28 +08`

本节接第 15 节 H6 计划。目标不是再扫 chunk16 scalar，而是先回答两个问题：

```text
1. 当前失败到底是不是 reset-window 内慢漂移，而不是突然 chunk jump？
2. 如果 oracle 知道 drift window 的方向/角色，现有 TTT write surface 有没有明显低于 34.1583 的空间？
```

### 16.1 H6-A 离线 attribution 工具

新增工具：

```text
tools/ttt_window_drift_state_audit.py
```

输入：

```text
trajectory diagnostics:
    per_frame_errors.csv
    chunk_errors.csv
    segment_errors.csv
    summary.json

per-run HMC/TTT no-GT state:
    hmc_state_hash.jsonl
```

输出：

```text
chunk_attribution.csv
window_summary.csv
feature_correlations.csv
window_feature_correlations.csv
delta_vs_reference.csv
h6_window_drift_state_audit.md
```

已跑三组 audit：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/h6_window_drift_state_audit_h5/
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/h6_window_drift_state_audit_c16role/
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/h6_window_drift_state_audit_oracle/
```

H6-A 关键读数：

| Audit | Window | Target | Strongest signal | 结论 |
|---|---|---|---|---|
| H5 | body `5-9` | `[200,300)` | `rmse_chunk_rmse_m r=0.998`, `sum_drift_vec_norm_m r=0.993` | body window drift 强预测主病灶 |
| C16ROLE | body `5-9` | `[200,300)` | `rmse_chunk_rmse_m r=0.999`, `sum_drift_vec_norm_m r=0.997` | 当前 best 仍是 body 慢漂移问题 |
| H5 | handoff `17-18` | `[400,600)` | `rmse_chunk_rmse_m r=0.997`, `mean_chunk_rmse_m r=0.990` | 后段代价由 handoff / post window 决定 |
| C16ROLE | handoff `17-18` | `[400,600)` | `rmse_chunk_rmse_m r=0.999`, `mean_delta_error_m r=-0.992` | chunk16 之后的 handoff 是下游 drift gate |

解释：

1. 第 13/15 节的判断成立：主失败不是单个 chunk boundary jump，而是 reset-window 内误差方向持续累积。
2. body window `5-9` 控制 `[200,300)`；exit / c16 / handoff 控制 `[400,600)` 和 endpoint。
3. 当前 `hmc_state_hash.jsonl` 里的单 chunk no-GT scalar 对全局 ATE 的直接相关很弱；有用信号必须先聚合成 reset-window state，再看 body / exit / handoff 的角色。

### 16.2 H6-B oracle upper-bound full run

固定 base：

```text
base = C16ROLE_01
tri replay = update_conflict_energy
branch = w0
positive_frac = 0.35
negative_frac = 0.12
neutral_lambda = 0.85 unless chunk params override
chunks = 5-12 + 16
SWA = SWKS3-style fixed protocol
RESET_EVERY = 5
```

候选：

| Run | 机制 | 目的 |
|---|---|---|
| `V8_H6_ORACLE_01_c5to12_neu075_SWKS3` | chunks `5-12` neutral `0.75` | oracle 假设整个 body+exit 有持续惯性，整体压 neutral |
| `V8_H6_ORACLE_02_c5to8_neu075_SWKS3` | chunks `5-8` neutral `0.75` | 只压病灶入口/body 前半段 |
| `V8_H6_ORACLE_03_c9to12g0025_SWKS3` | chunks `9-12` gamma `0.0025` | 只让 exit handoff 更弱 |
| `V8_H6_ORACLE_04_c5to8neu075_c9to12g0025_SWKS3` | `ORACLE_02 + ORACLE_03` | body 惯性压制 + exit 弱化叠加 |

运行记录：

| Run | Start | Done | Walltime |
|---|---|---|---:|
| `V8_H6_ORACLE_01_c5to12_neu075_SWKS3` | `2026-05-12 22:47:01` | `23:21:08` | `34.1 min` |
| `V8_H6_ORACLE_02_c5to8_neu075_SWKS3` | `2026-05-12 22:47:01` | `23:19:46` | `32.8 min` |
| `V8_H6_ORACLE_03_c9to12g0025_SWKS3` | `2026-05-12 22:47:01` | `23:19:12` | `32.2 min` |
| `V8_H6_ORACLE_04_c5to8neu075_c9to12g0025_SWKS3` | `2026-05-12 22:47:01` | `23:18:25` | `31.4 min` |

Global metrics：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---|
| `WINGAM_03_repeat` | `34.1903` | `6.5666` | `92.4202` | `0.0083` | v7/v8 repeat reference |
| `C16ROLE_01` | `34.1583` | `6.5327` | `92.4059` | `0.0082` | current v8 best |
| `H6_ORACLE_01` | `34.2388` | `6.4765` | `92.4041` | `0.0082` | Rot/Final 好，但 ATE 回退 |
| `H6_ORACLE_02` | `34.2898` | `6.5286` | `92.4054` | `0.0082` | 回退 |
| `H6_ORACLE_03` | `34.2352` | `6.4828` | `92.4046` | `0.0082` | oracle best ATE，但未过 C16ROLE |
| `H6_ORACLE_04` | `34.2518` | `6.5009` | `92.4050` | `0.0082` | FinalErr 最好，但 ATE 回退 |

Trajectory diagnostics：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/trajectory_diagnostics_h6_oracle/
```

| Run | ATE RMSE | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `WINGAM_03_repeat` | `34.1903` | `6.195` | `75.576` | `55.428` | `42.280` | `3.774` | `30.994373` |
| `C16ROLE_01` | `34.1583` | `6.249` | `74.521` | `54.740` | `44.369` | `3.803` | `30.809795` |
| `H6_ORACLE_01` | `34.2388` | `5.874` | `74.672` | `54.859` | `44.675` | `3.757` | `30.790094` |
| `H6_ORACLE_02` | `34.2898` | `6.351` | `74.819` | `54.980` | `44.649` | `3.801` | `30.804489` |
| `H6_ORACLE_03` | `34.2352` | `5.948` | `74.533` | `54.770` | `44.686` | `3.771` | `30.793799` |
| `H6_ORACLE_04` | `34.2518` | `5.806` | `74.764` | `54.931` | `44.551` | `3.779` | `30.806060` |

Window audit：

```text
results/kitti01_hmc_v2/acl2_v8_ttt_windowed_trireplay/h6_window_drift_state_audit_oracle/
```

### 16.3 H6 结论

1. H6-A attribution 成立：body / exit / handoff 的 drift-state 是比单 chunk boundary 更正确的诊断单位。
2. H6-B oracle upper-bound 没有打开新空间：四条 oracle full run 都没有超过 `C16ROLE_01=34.1583`。
3. oracle 主要改善 Rot / FinalErr，而不是 ATE。最好 ATE 的 `H6_ORACLE_03=34.2352` 仍比 C16ROLE 差 `0.0768m`；最好 FinalErr 的 `H6_ORACLE_04=5.806` 也用 ATE 回退换来。
4. `H6_ORACLE_03` 证明 “exit gamma 更弱” 是合理方向，但这个方向在 C16ROLE 附近已经被压到平台；继续把 `9-12` gamma 从 `0.003` 往 `0.0025` 调，只会变成 Rot/endpoint regularizer，不是 ATE 主杠杆。
5. 因为 oracle 都没过 best，不应继续直接跑 H6-C 的四条 no-GT full 矩阵。否则只是把失败的 oracle 近似成更噪的 proxy。

当前 v8 best 保持：

```text
V8_C16ROLE_01_c16_pos035_neg008_neu085_SWKS3
ATE / Rot = 34.1583 / 6.5327
```

---

## 17. H6 后的深度判断：下一步 H7 计划

H6 给出的重要信息不是 “window drift proxy 失败”，而是：

```text
window-level drift-state 是正确诊断单位；
但现有 TTT write action 仍只是调 gamma / neutral / role fraction，
这个 action space 对 ATE 已接近平台。
```

因此下一步不建议再做：

1. `body/exit gamma` 细扫；
2. `neutral_lambda 0.75/0.80/0.90` 细扫；
3. chunk16 role / commit-filter / post-window handoff 的标量微调；
4. H6-C 四条 no-GT proxy full run 的直接近似版。

### H7-A. no-GT drift proxy 先做判别，不直接控制

先把 H6-C 从 “直接改写入策略” 改成 “离线判别器”：

```text
input no-GT features:
    reset-window cumulative TTT delta
    pass1/pass2 pose increment disagreement
    SWA overlap replacement score / residual
    update_conflict_energy window summary
    memory_ttt_w0 mean / q90 relative diff

target labels from completed runs:
    body improves [200,300) but hurts [400,600)
    exit improves FinalErr/Rot but hurts ATE
    c16/handoff safe vs unsafe
```

promotion gate：

```text
proxy 必须能在已有 C16ROLE / H5 / H6 oracle runs 上解释：
    local gain vs downstream cost
否则不允许进入 full run。
```

### H7-B. 换 action space：drift-state 不是调 scalar，而是改 replay residual composition

当前 `tri_replay` 的 action 仍然是：

```text
pos_frac / neg_frac / neutral_lambda / gamma
```

H6 表明这些标量已经卡住。下一步应实现 window-conditioned residual composition：

```text
body window 5-9:
    保持 update_conflict_energy negative replay
    但只对与 window drift direction 同向的 residual 做 negative
    与 drift correction 反向的 residual 归入 neutral/positive

exit window 10-12:
    不再统一降低 gamma
    改成只保留能减少 body accumulated drift projection 的 neutral residual

c16/handoff:
    只允许降低 [400,600) drift projection 的 correction 进入 long commit
```

也就是从：

```text
按 token risk 分组
```

升级为：

```text
按 token update direction 与 window drift direction 的投影分组
```

### H7-C. oracle projection upper-bound

在写 no-GT proxy 前，先做一个更强 oracle：

```text
drift projection oracle:
    使用 GT residual vector 只定义方向，不直接用误差大小；
    token / layer update delta 若投影到 correction direction，则进 positive/neutral；
    若投影到加剧 drift direction，则进 negative。
```

候选只跑 2 条 full：

| Run | 机制 | 目的 |
|---|---|---|
| `V8_H7_ORACLEPROJ_01` | chunks `5-12` projection-routed tri replay | 测 projection action space 是否过 C16ROLE |
| `V8_H7_ORACLEPROJ_02` | chunks `5-12 + 16/17/18` projection-routed long gate | 测下游 handoff 是否能被方向投影安全控制 |

promotion gate：

```text
如果 oracle projection 不能过 34.1583：
    停止 TTT write-side action 扩展，转 read-side / pose-scale failure。

如果能过：
    再做 no-GT proxy 近似，优先从 pass1/pass2 pose disagreement 和 SWA residual 构造 drift direction。
```

### H7-D. 若 H7 oracle projection 仍失败，转向 read / pose-scale

H6-B 已经暗示：当前 write-side scalar action 对 ATE 接近上限。若 H7-C 仍不能打开空间，应停止 TTT write 继续微调，转向：

```text
1. read-side window-conditioned beta / support selection；
2. Sim3 scale drift 的 no-GT proxy；
3. pose increment consistency loss / segment-level reranking；
4. SWA overlap source replacement 的 window-level gate。
```

当前阶段判断：

```text
TTT write-side 已从 36.4161 推到 34.1583；
剩余到 <30m 的 4.16m 差距，不太可能靠 gamma/neutral/chunk16 标量拿到。
下一步必须验证 “update direction projection against window drift” 这个新 action space。
```

# ACL2 v46B 实验结果复盘

日期：2026-06-08（Asia/Singapore）

计划文件：

```text
docs/ACL2_v46B_ComponentAttribution_FrameTTT_FrameSWA_Addendum.md
```

执行日志：

```text
docs/ACL2_v46B_执行日志.md
```

结果根目录：

```text
results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/
```

## 当前状态

```text
v46B clean no-chunk 8-row factorial 已完成。
8/8 rows 均为 full-online DONE，hmc_rows=38，frames=1101，row_valid=True。
最终 report、SWA boundary diagnostic、interaction plots 已生成。
本复盘只记录真实运行得到的数据；missing/not_run 不补写。
```

## 本轮必须回答的问题

```text
1. FRAME_ATTN 单独有没有贡献？
2. TTT 单独有没有贡献？
3. SWA 单独有没有贡献？
4. FRAME_ATTN+TTT 是否明显强于二者单独？
5. FRAME_ATTN+SWA 是否明显强于二者单独？
6. C9 的主要收益到底来自 READ、TTT、SWA，还是 interaction？
```

## 已完成代码修改审计

| 文件 | 修改 | 目的 | 当前验证 |
|---|---|---|---|
| `tools/run_v46b_factorial_candidate.sh` | 新增 v46B clean no-chunk 8-row factorial launcher | 将 FRAME_ATTN / TTT / SWA 以三因子形式独立打开/关闭，并写 component/chunk-id audit | `bash -n` pass；8/8 rows action audit pass |
| `tools/v46b_factorial_report.py` | 新增 v46B landed-artifact report；修复 RPE 解析；修复 boundary note 条件化说明 | 生成 registry、main effects、interactions、summary 和 plots，不补写缺失值 | `py_compile` pass；final report pass |
| `run_pipeline_abc_v2.py` | 新增 `probe_ttt_write_seconds` 计时字段 | 定位 TTT rows 速度瓶颈到 `hmc.build_probe_ttt_write_state(...)` | `py_compile` pass；只影响 future timing，不改变当前 rollout |

## 关键设计约束

```text
1. C9_P0_R2 的 chunk-id map 不用于 v46B rows。
2. TTT_OFF 使用 HMC_COMMIT_MODE=probe_native，避免 hidden TTT writing 混入。
3. TTT_ON 使用 HMC_COMMIT_MODE=probe_ttt_write + tri_replay + global gamma/fractions。
4. FRAME_ATTN_OFF 使用 READ_PATH=none，但仍允许计算 D_g 作为 TTT/SWA 内部风险信号。
5. SWA_ON 只启用 SWA overlap source replacement，不自动启用 read path 或 TTT。
6. commit EMA 与 native mix 在 clean attribution 中默认关闭，避免重新引入 C9 chunk-specific 配方。
```

## 结果表

最终结果从以下文件生成：

```text
results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1/phase2_factorial_registry.csv
```

## Blocker / 修复记录

### B0：report dry-run GT loader 接口错误

现象：

```text
第一次运行 tools/v46b_factorial_report.py 时触发：
TypeError: tuple indices must be integers or slices, not tuple
```

原因：

```text
新 report 第一版误以为 _load_kitti_gt 返回 gt_poses。
实际该 helper 返回 (_gt_frames, gt_poses, gt_pos)。
同时 _load_tum_prediction 需要传入 n_gt。
```

修复：

```text
文件：tools/v46b_factorial_report.py
修改：
  _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
  frames, raw_poses, _ = _load_tum_prediction(path, gt_pos.shape[0])
```

审计意义：

```text
这是报告器接口修复，不改变模型、launcher、实验策略或 evaluator 算法。
修复后需重新 dry-run，确认 missing-state report 能正常生成且不补写指标。
```

修复验证：

```text
missing-state dry-run 已通过。
生成 registry/main-effects/interactions/summary/plots。
所有 rows 正确显示 missing_prediction，不含伪造 ATE。
```

后续如发生新 blocker，将记录：

```text
blocker 现象
影响范围
按计划采取的修复
修复涉及文件
验证命令
是否需要重跑
```

### B1：8-row 并发启动后全部早退

现象：

```text
8 个 v46B rows 均只有 START，没有 DONE/FAIL。
ps 中没有残留 run_pipeline_abc_v2 / run_attention_cue / run_v46b 进程。
01.log 大多停在：
  Collected 1101 images.
  Full-res images: skipped
```

结论：

```text
这些启动不产生任何 ATE 结果，不能进入实验指标表。
```

按计划修复方向：

```text
先跑同 launcher 的 64-frame debug row，确认是并发/IO/环境问题还是 launcher 配置问题；
若 debug 失败，修 launcher/CLI/HMC 接线；
若 debug 通过，则改为低并发/逐 row full-online 重跑。
```

64-frame debug 结果：

```text
V46B_DEBUG_F000_64F DONE。
```

解释：

```text
同一 launcher 可完成短跑并生成 trajectory，因此 B1 不是 row 定义必然错误。
后续 full-online 改为低并发批次重跑。
debug row 不进入 v46B 指标表。
```

补充现象：

```text
低并发 Batch 1 若用 nohup 后台启动，仍只有 START、无 DONE/FAIL。
由于 64-frame 前台 debug 可完成，判断这是执行环境后台进程生命周期问题。
```

修复：

```text
full-online 改为持久 shell session + wait，不再用 detached background。
```

## 中间进度：F000/F100 已完成，TTT/SWA 与组合行进行中

时间：2026-06-08 18:08:05 +08

已完成并通过 action audit 的 rows：

| Row | 状态 | ATE | hmc rows | frame | TTT | SWA | 审计结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `F000_NONE` | done | `41.750205` | 38 | 0 | 0 | 0 | valid |
| `F100_ONLY_FRAME_ATTN` | done | `38.593413` | 38 | 1 | 0 | 0 | valid |
| `F010_ONLY_TTT` | done | `39.519104` | 38 | 0 | 38 | 0 | valid |
| `F001_ONLY_SWA` | done | `41.737041` | 38 | 0 | 0 | 37 | valid |
| `F101_FRAME_ATTN_SWA` | done | `38.592627` | 38 | 1 | 0 | 37 | valid |

初步解释：

```text
F100 相比 F000 的 gain = 41.750205 - 38.593413 = 3.156792m。
这是 clean no-chunk 条件下 frame-attn/read cue alone 的第一条正向隔离证据。
F010 相比 F000 的 gain = 41.750205 - 39.519104 = 2.231101m。
这是 clean no-chunk 条件下 TTT writing alone 的正向隔离证据，但它伴随 Rot/FinalErr 变差：
Rot = 6.485906780904279，FinalErr = 9.085096859131909。
F001 相比 F000 的 gain = 41.750205 - 41.737041 = 0.013164m。
这是 clean no-chunk 条件下 SWA alone 的正向隔离证据；目前看 only-SWA 对 full ATE 近似中性。
F101 相比 F000 的 gain = 41.750205 - 38.592627 = 3.157579m。
F101 相比 F100 只改善约 0.000786m，说明 SWA 加到 frame-attn/read cue 上几乎没有额外 full-ATE 收益。
它仍只是 8-row factorial 的部分结果；不能在 F010/F110/F011/F111 完成前写最终 TTT 交互结论。
```

进行中 rows：

| Row | 状态 | partial hmc rows | partial action audit |
|---|---|---:|---|
| `F110_FRAME_ATTN_TTT` | running | not_reported_yet | pending |
| `F011_TTT_SWA` | running | not_reported_yet | pending |
| `F111_ALL_THREE` | running | not_reported_yet | pending |

审计说明：

```text
1. F010 partial report 中 TTT applied count 与已落盘 hmc rows 一致，说明 TTT path 已接通。
2. F001 partial report 中 SWA applied count > 0，说明 SWA path 已接通。
3. F110/F101 刚启动，待完整 run 后必须检查 action debug：
   - F110 必须 frame=true 且 TTT applied>0 且 SWA=0。
   - F101 必须 frame=true 且 SWA applied>0 且 TTT=0。
4. 若任一强制 row action debug 不匹配，按 v46B plan 判为 invalid，先修 launcher/CLI/HMC 后重跑，不能用错误 row 算贡献。
```

## 报告器修复：RPE 从 KITTI benchmark artifact 读取

问题：

```text
v46B plan 要求记录 RPE_t / RPE_r。
报告器第一版没有读取已经落盘的 results_sim3/results_rpe.txt，因此 RPE 列为 missing。
```

修复：

```text
tools/v46b_factorial_report.py 新增 _parse_rpe_metrics(run_dir)。
RPE_t_full / RPE_r_full 现在直接来自每个 rollout 的 results_sim3/results_rpe.txt。
```

验证：

| Row | RPE_t_full | RPE_r_full |
|---|---:|---:|
| `F000_NONE` | `92.396146` | `0.00839` |
| `F100_ONLY_FRAME_ATTN` | `92.393993` | `0.008656` |
| `F001_ONLY_SWA` | `92.394478` | `0.008383` |

审计说明：

```text
这是报告器补漏，不影响实验运行和 ATE/Rot/FinalErr。
RPE 数据来自真实 KITTI benchmark 输出，不是复盘阶段编造。
```

## 中间复盘：实验速度瓶颈定位

用户指出实验推进太慢后，本轮对正在运行的 TTT rows 做了系统层与 run 内 timing 诊断。

### 证据

已完成 rows 的 wall time：

| Row | 开始 | 结束 | 约耗时 | TTT |
|---|---|---|---:|---:|
| `F000_NONE` | 17:41:07 | 18:05:26 | 24.3 min | off |
| `F100_ONLY_FRAME_ATTN` | 17:41:07 | 18:05:48 | 24.7 min | off |
| `F001_ONLY_SWA` | 17:44:33 | 18:09:19 | 24.8 min | off |
| `F101_FRAME_ATTN_SWA` | 18:07:53 | 18:33:35 | 25.7 min | off |
| `F010_ONLY_TTT` | 17:44:33 | 18:48:49 | 64.3 min | on |

`timing_summary.json` 聚合：

| Row | chunks | mean chunk total | mean pass1 | mean Stage B | mean Stage D | mean pass2 |
|---|---:|---:|---:|---:|---:|---:|
| `F000_NONE` | 38 | 36.537s | 12.190s | 9.035s | 1.108s | 12.395s |
| `F100_ONLY_FRAME_ATTN` | 38 | 37.057s | 12.192s | 9.289s | 1.107s | 12.798s |
| `F001_ONLY_SWA` | 38 | 37.114s | 12.237s | 9.333s | 1.144s | 12.619s |
| `F101_FRAME_ATTN_SWA` | 38 | 38.298s | 12.643s | 9.629s | 1.120s | 12.886s |
| `F010_ONLY_TTT` | 38 | 99.638s | 12.051s | 8.946s | 1.124s | 10.714s |
| `F110_FRAME_ATTN_TTT` | 31 partial | 105.354s | 13.110s | 8.961s | 1.124s | 11.289s |
| `F011_TTT_SWA` | 29 partial | 107.180s | 13.627s | 9.299s | 1.136s | 12.120s |
| `F111_ALL_THREE` | 31 partial | 100.246s | 12.108s | 8.814s | 1.111s | 10.857s |

系统采样：

```text
pidstat:
  TTT Python 进程 CPU 约 110%-394%。
  iodelay=0。
  kB_rd/s≈0，kB_wr/s≈0。

iostat:
  %iowait≈0。

nvidia-smi pmon:
  TTT 进程 GPU SM 多数为 0-9%。
```

### 结论

速度拖慢的主因不是磁盘 I/O，也不是 GPU 算力打满，而是 TTT 写入路径中的 CPU/调度侧开销。

更精确地说：

```text
已记录的 pass1 + Stage B + Stage D + pass2 合计约 31-34 秒。
TTT rows 的 chunk_total_seconds 是 100 秒级。
差出来的 65-75 秒主要落在未单独计时的 probe_ttt_write commit 区间：
    hmc.build_probe_ttt_write_state(...)
```

因此当前瓶颈定位为：

```text
probe_ttt_write commit / hmc.build_probe_ttt_write_state(...)
```

### 已做修复

修改：

```text
run_pipeline_abc_v2.py
```

新增：

```text
chunk_timing["probe_ttt_write_seconds"]
```

并围绕：

```text
hmc.build_probe_ttt_write_state(...)
```

记录耗时，写入后续 run 的 `timing_summary.json`。

验证：

```text
py_compile pass
```

审计说明：

```text
该修改只增加 timing 字段，不改变模型、算法分支、evaluator 或当前正在运行的结果。
当前三条 active run 已加载旧代码，因此最终 v46B 指标不受该补丁影响。
后续若要继续优化速度，应先用这个字段确认 build_probe_ttt_write_state 内部耗时，再拆分到 layer loop / tensor clone / state hash / debug serialization 等更细粒度。
```

## Final Phase2 收口：8-row clean factorial 完成

### 完整性检查

最终 report 重新生成于：

```text
results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1/
```

已确认：

```text
registry_rows = 8
errors = []
8/8 row_valid=True
8/8 run_status_done=True
8/8 no_chunk_policy_pass=True
8/8 hmc_rows=38
8/8 frames=1101
```

关键 report artifacts：

```text
phase2_factorial_registry.csv
phase2_factorial_registry.json
phase2_component_main_effects.csv
phase2_component_interactions.csv
phase2_frame_ttt_swa_summary.md
component_interaction_heatmap.png
segment_delta_stacked_bar.png
swa_boundary/swa_boundary_summary.csv
swa_boundary/swa_boundary_by_candidate_boundary.csv
```

### Final registry

| Row | FRAME_ATTN | TTT | SWA | ATE_full | Rot_full | FinalErr_full | RPE_t | RPE_r | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `F000_NONE` | 0 | 0 | 0 | 41.750205342038015 | 5.354471107510558 | 3.923875981533656 | 92.396146 | 0.00839 | True |
| `F100_ONLY_FRAME_ATTN` | 1 | 0 | 0 | 38.593412804190564 | 5.250862613050851 | 2.982166037805976 | 92.393993 | 0.008656 | True |
| `F010_ONLY_TTT` | 0 | 1 | 0 | 39.51910428415088 | 6.485906780904279 | 9.085096859131909 | 92.334849 | 0.009391 | True |
| `F001_ONLY_SWA` | 0 | 0 | 1 | 41.73704103737674 | 5.320253824239548 | 4.0121577092299585 | 92.394478 | 0.008383 | True |
| `F110_FRAME_ATTN_TTT` | 1 | 1 | 0 | 36.668910841145234 | 6.463437289271693 | 14.699225179343587 | 92.33228 | 0.009781 | True |
| `F101_FRAME_ATTN_SWA` | 1 | 0 | 1 | 38.592626794820525 | 5.214515145577633 | 2.8800223717942446 | 92.392316 | 0.008647 | True |
| `F011_TTT_SWA` | 0 | 1 | 1 | 39.503549725643346 | 6.44974548970202 | 9.00026345482514 | 92.33275 | 0.009383 | True |
| `F111_ALL_THREE` | 1 | 1 | 1 | 36.65074045045878 | 6.418129083708555 | 14.614346164545774 | 92.330293 | 0.009768 | True |

### Main effects

Gain 定义为：

```text
gain_vs_F000 = F000_NONE ATE - candidate ATE
```

| Factor / combo | Row | ATE_full | gain_vs_F000 |
|---|---|---:|---:|
| `FRAME_ATTN_only` | `F100_ONLY_FRAME_ATTN` | 38.593412804190564 | 3.1567925378474513 |
| `TTT_only` | `F010_ONLY_TTT` | 39.51910428415088 | 2.2311010578871375 |
| `SWA_only` | `F001_ONLY_SWA` | 41.73704103737674 | 0.013164304661273718 |
| `FRAME_ATTN_TTT` | `F110_FRAME_ATTN_TTT` | 36.668910841145234 | 5.081294500892781 |
| `FRAME_ATTN_SWA` | `F101_FRAME_ATTN_SWA` | 38.592626794820525 | 3.15757854721749 |
| `TTT_SWA` | `F011_TTT_SWA` | 39.503549725643346 | 2.246655616394669 |
| `ALL_THREE` | `F111_ALL_THREE` | 36.65074045045878 | 5.099464891579238 |

### Interaction 结果

| Interaction | classification | combo gain | margin |
|---|---|---:|---:|
| `READ_x_TTT` | `synergy` | 5.081294500892781 | 1.92450196304533 |
| `READ_x_SWA` | `swa_near_zero_under_read` | 3.15757854721749 | 0.0007860093700386983 |
| `TTT_x_SWA` | not classified | 2.246655616394669 | 0.015554558507531624 |
| `THREE_WAY` | not classified | 5.099464891579238 | 0.018170390686456983 vs best pair |

解释：

```text
1. FRAME_ATTN-only 是最强单组件，单独改善 3.1568m。
2. TTT-only 也有明显正贡献，单独改善 2.2311m，但它显著恶化 Rot_full 和 FinalErr_full。
3. SWA-only 几乎没有 full ATE 贡献，只有 0.0132m。
4. FRAME_ATTN+TTT 有真实 synergy，组合 gain=5.0813m，比最强单组件多 1.9245m。
5. FRAME_ATTN+SWA 几乎等于 FRAME_ATTN-only，SWA 在 READ 下近似零贡献。
6. ALL_THREE 是最佳行，ATE=36.65074045045878，但只比 FRAME_ATTN+TTT 好 0.01817m。
```

### SWA boundary diagnostic

SWA boundary 诊断命令使用 `v27_swa_boundary_diagnostics.py`，reference 为 `F000_NONE`。

| candidate | boundary_count | mean 10f ATE | 10f delta vs H9 | mean 20f ATE | 20f delta vs H9 | gate |
|---|---:|---:|---:|---:|---:|---|
| `V46B_F001_ONLY_SWA` | 37 | 36.830794447749334 | -0.021091095523154024 | 36.93193481489409 | -0.019371327704718055 | False |
| `V46B_F101_FRAME_ATTN_SWA` | 37 | 33.25907636391737 | -3.5928091793551147 | 33.38934469468534 | -3.5619614479134682 | False |
| `V46B_F011_TTT_SWA` | 37 | 34.35453498575968 | -2.49735055751281 | 34.48241788958564 | -2.4688882530131693 | False |
| `V46B_F111_ALL_THREE` | 37 | 31.061558973199865 | -5.790326570072622 | 31.221660230588952 | -5.7296459120098575 | True |

分析：

```text
SWA-only 在 boundary windows 几乎没有收益；
但当 SWA 与 READ/TTT 组合时，boundary-local ATE 明显降低。
这解释了为什么 full ATE 中 SWA 是 near-zero：它像局部边界稳定器，不是全程主贡献项。
```

### 回答用户最初三问

#### 1. 只对 frame attn / read cue 做增强结果如何？

```text
F100_ONLY_FRAME_ATTN:
ATE = 38.593412804190564
gain vs clean baseline F000 = +3.1567925378474513m
```

结论：

```text
frame attention / read cue 是最强单组件。
它不是 C9 chunk-id map 的泛化证明，因为 v46B 使用的是 clean fixed beta=4.75、无 chunk-id maps。
但在 clean factorial 中，它确实贡献最大。
```

#### 2. 只对 TTT 做增强结果如何？

```text
F010_ONLY_TTT:
ATE = 39.51910428415088
gain vs F000 = +2.2311010578871375m
ttt_tri_replay_applied_count = 38
ttt_tri_replay_applied_layer_count_sum = 684
```

结论：

```text
TTT-only 有明确正贡献，但副作用也明显：
Rot_full 从 F000 的 5.354471107510558 升到 6.485906780904279；
FinalErr_full 从 3.923875981533656 升到 9.085096859131909。
所以 TTT 是强组件，但单独使用会带来姿态/终点误差风险。
```

#### 3. 只对 SWA 做增强结果如何？

```text
F001_ONLY_SWA:
ATE = 41.73704103737674
gain vs F000 = +0.013164304661273718m
swa_overlap_replace_applied_count = 37
```

结论：

```text
SWA-only 对 full ATE 几乎没有贡献。
它更像局部边界稳定器，只有在与 READ/TTT 组合时 boundary window 指标变好。
```

### 最终结论

v46B 完成了用户要求的 positive-only isolated ablation：

```text
only FRAME_ATTN: +3.1568m
only TTT:        +2.2311m
only SWA:        +0.0132m
READ x TTT:      strong synergy, +5.0813m combo gain
READ x SWA:      near-zero beyond READ
best row:        F111_ALL_THREE, ATE=36.65074045045878
```

和 C9/P0 对比必须诚实写：

```text
C9/P0_R2 historical reference ATE = 33.76294210291885。
v46B best clean no-chunk row F111 ATE = 36.65074045045878。
因此 v46B attribution rows 解释了 clean 组件贡献，但没有复现 C9 historical best。
```

安全结论：

```text
1. C9 的历史强结果不能归因于 SWA-only。
2. clean setting 下，READ 是最强单组件，TTT 是第二强单组件。
3. READ+TTT 的 interaction 是最重要组合收益。
4. C9/P0_R2 的额外优势仍来自 chunk-specific schedule / EMA / gamma 等 historical recipe，不是 v46B clean factorial 能复现的泛化策略。
```

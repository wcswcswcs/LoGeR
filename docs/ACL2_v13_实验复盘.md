# ACL2 v13 实验复盘：TTT Write Causal ActionSpace Reboot Target25

日期：2026-05-19（Asia/Singapore）  
计划文件：`docs/ACL2_v13_TTT_Write_Causal_ActionSpace_Reboot_Target25_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/`

本轮原则：只记录实际落盘结果；不把 offline trajectory rewrite、GT runtime action、postprocess txt 改写写成 deployable TTT success。用户约束后所有继续运行只使用 GPU `0/1/2/3`。

---

## 0. Phase 0 边界锁定

输出：

```text
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/registry_v13.csv
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/phase0_nogt_pose_proxy/
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/phase5_global_drift_dashboard/phase0_repeats/
```

| Run | counts_as_ttt_write | ATE | Rot | FinalErr | `[200,300)` | `[400,600)` | hmc rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `H9_REPEAT` | `true` | `34.1258` | `6.5414` | `6.189` | `74.410` | `44.354` | `38` |
| `WINGAM_REPEAT` | `true` | `34.1903` | `6.5666` | `6.195` | `75.576` | `42.280` | `38` |
| `NOGTPOSE_27_reset_global_clip35_body600_t105` | `false` | `22.4012` | n/a | `8.753` | `35.187` | `8.587` | n/a |

Gate:

```text
H9 drift vs v11/v12 reference: < 0.0001m <= 0.03m
WINGAM drift vs v11 reference: < 0.0001m <= 0.03m
NOGTPOSE_27 drift vs v10/v11 record: < 0.0001m <= 0.05m
NOGTPOSE_27 counts_as_ttt_write = false
PASS
```

说明：Phase 0 两个 shell session 因本轮中途修改 launch script，在 Python run 已 `DONE` 后出现旧 bash 进程读取脚本的语法错误；当前 `bash -n tools/run_attention_cue_experiment.sh` 已通过，且两个 run 均有 38 条 HMC row、`kitti_benchmark.log` 和 trajectory 输出。本复盘按已落盘 run artifact 计数，不把 shell 收尾错误当成实验数据。

---

## 1. 工程接线

新增/修改：

```text
loger/pipeline/ttt_write_controller.py:
    native_delta_gate_mode += orthogonal_suppress
    update_native_mix_chunks
    commit_ema_chunks
    post-zeropower debug fields

run_pipeline_abc_v2.py:
    --ttt_write_native_mix_chunks
    --ttt_write_commit_ema_chunks
    post_zeropower_delta_norm.csv / cosine / norm_restore_ratio / heatmap logging

tools/run_attention_cue_experiment.sh:
    TTT_WRITE_NATIVE_MIX_CHUNKS
    TTT_WRITE_COMMIT_EMA_CHUNKS
```

Smoke：

| Run | Scope | ATE / Rot | Artifact check |
|---|---|---:|---|
| `V13_PZ_SMOKE_orthogonal_e180_SWKS3` | `END_FRAME=180` | `4.0160 / 3.7452` | `post_zeropower_delta_norm.csv` 379 lines |

PZ_06 full run later产生：

```text
post_zeropower_delta_norm.csv: 2053 lines
post_zeropower_delta_cosine.csv
norm_restore_ratio.csv
per_layer_branch_update_heatmap.csv
```

---

## 2. Phase 1 Candidate Policy Bank

计划中的 per-chunk GT oracle candidate selector / short rollout scorer 没有在本轮实现；本轮实际完成的是 **online full-run policy bank**：每条 candidate 都是真实 full online HMC，commit 影响未来 state，无 GT runtime selector，无后处理。没有把它伪写成 per-chunk oracle。

输出：

```text
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/phase1_candidate_bank_oracle/candidate_bank_registry.csv
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/phase5_global_drift_dashboard/phase1_candidate_bank_oracle/
```

| Run | ATE | Rot | FinalErr | `[200,300)` | `[400,600)` | hmc rows |
|---|---:|---:|---:|---:|---:|---:|
| `C9_WEAK_FREEZE_C56` | `33.7629` | `6.5259` | `5.666` | `76.102` | `41.896` | `38` |
| `H9_REPEAT` | `34.1258` | `6.5414` | `6.189` | `74.410` | `44.354` | `38` |
| `WINGAM_REPEAT` | `34.1903` | `6.5666` | `6.195` | `75.576` | `42.280` | `38` |
| `C5_EXIT_REDUCED` | `34.2151` | `6.4608` | `5.694` | `74.416` | `44.769` | `38` |
| `C3_SOFT_BODY` | `34.2405` | `6.5841` | `6.399` | `74.415` | `44.870` | `38` |
| `C2_C16ROLE` | `34.3011` | `6.4750` | `5.998` | `74.666` | `44.859` | `38` |
| `C4_STRONG_BODY` | `34.3187` | `6.5646` | `6.547` | `74.610` | `44.936` | `38` |
| `C7_NATIVE_PROTECTED` | `34.8994` | `6.5505` | `5.525` | `76.109` | `44.992` | `38` |
| `C6_EXIT_REMOVED` | `35.0455` | `6.6036` | `5.976` | `76.727` | `44.280` | `38` |
| `C8_PZ_NORM_MIX025` | `35.5372` | `6.5842` | `5.718` | `76.744` | `46.413` | `38` |

Gate:

```text
best deployable online v13 ATE = 33.7629 (C9_WEAK_FREEZE_C56)
improvement vs H9 ATE = 0.3629m
but best > 33.5
and [200,300) worsens by 1.692m
Phase 1 weak/oracle gate = fail
Phase 2 no-GT selector = not started
```

结论：C9 是真实 online TTT write improvement，但不是 target-25 candidate，也不是 v13 useful progress gate：ATE 仍远高于 33/30/25，且关键 `[200,300)` 继续恶化。

---

## 3. Phase 3 Finite-Difference Sensitivity

因为 Phase 1 gate fail，启动有限差分 full runs。实际运行了 chunk5/chunk6 w0 commit-EMA alpha `0.90/1.10` 四条 full；没有实现完整 G1-G8 矩阵，也没有伪造 no-GT proxy Spearman。

输出：

```text
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/phase3_finite_diff_sensitivity/finite_diff_registry.csv
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/phase3_finite_diff_sensitivity/finite_diff_future_eval.csv
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/phase5_global_drift_dashboard/phase3_finite_diff_sensitivity/
```

| Run | ATE | Rot | FinalErr | `[200,300)` | `[400,600)` |
|---|---:|---:|---:|---:|---:|
| `H9_REPEAT` | `34.1258` | `6.5414` | `6.189` | `74.410` | `44.354` |
| `FD_G2_CH6_ALPHA090` | `34.1581` | `6.5168` | `5.848` | `74.578` | `44.428` |
| `FD_G1_CH5_ALPHA090` | `34.2594` | `6.5072` | `5.629` | `74.705` | `44.723` |
| `FD_G1_CH5_ALPHA110` | `34.2999` | `6.5436` | `6.052` | `74.255` | `45.041` |
| `FD_G2_CH6_ALPHA110` | `34.3453` | `6.5073` | `5.971` | `74.388` | `45.185` |

Gate:

```text
best finite-diff ATE = 34.1581
no perturbation beats H9
no perturbation improves [200,300) by >= 3m
no no-GT proxy Spearman claimed (proxy scorer not implemented)
Phase 3 = fail
```

---

## 4. Phase 4 Post-Zeropower Delta Routing

`PZ_01` 由 Phase 1 的 `C8_PZ_NORM_MIX025` 覆盖；`PZ_02-PZ_06` 在 Phase 4 full run 中执行。`PZ_07` 未启动，因为 targeted layer/head hook 未验证；`PZ_08` 未启动，因为 Phase 3 没有 high-sensitivity group。

输出：

```text
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/phase4_post_zeropower_routing/post_zeropower_registry.csv
results/kitti01_hmc_v2/acl2_v13_ttt_causal_actionspace_reboot/phase5_global_drift_dashboard/phase4_post_zeropower_routing/
```

| Run | ATE | Rot | FinalErr | `[200,300)` | `[400,600)` |
|---|---:|---:|---:|---:|---:|
| `H9_REPEAT` | `34.1258` | `6.5414` | `6.189` | `74.410` | `44.354` |
| `PZ_04_EXIT_MIX025` | `34.7590` | `6.5976` | `6.102` | `76.209` | `44.100` |
| `PZ_02_NORM_MIX050` | `34.9279` | `6.5352` | `5.486` | `75.986` | `45.337` |
| `PZ_06_W0_ORTHO_RHO050` | `35.0135` | `6.5485` | `6.276` | `75.946` | `45.369` |
| `PZ_03_BODY_MIX025` | `35.0621` | `6.6006` | `5.930` | `75.299` | `46.726` |
| `PZ_01_C8_NORM_MIX025` | `35.5372` | `6.5842` | `5.718` | `76.744` | `46.413` |
| `PZ_05_W1_CAP050` | `38.2047` | `6.5533` | `6.522` | `68.615` | `56.087` |

Gate:

```text
strong pass ATE <= 33.0: fail
useful pass ATE <= 33.5: fail
[200,300) improves >= 5m without [400,600) regression: fail
best PZ ATE = 34.7590
PZ_05 improves [200,300) but catastrophically regresses [400,600) and ATE
Phase 4 = fail
```

---

## 5. Final Decision

v13 的有效新增 deployable online result：

```text
Best deployable online TTT/HMC by ATE:
    C9_WEAK_FREEZE_C56
    ATE / Rot = 33.7629 / 6.5259
    FinalErr = 5.666
    [200,300) = 76.102
    [400,600) = 41.896
    counts_as_ttt_write = true
    no GT runtime action
    no offline postprocess
```

但它不是 target-25 / strong / useful progress：

```text
ATE > 33.5
[200,300) worse than H9 by 1.692m
target-25 gap = 8.7629m
```

停止/跳过规则：

```text
Phase 1 policy-bank best ATE > 33.5 => Phase 2 selector not started
Phase 3 finite-diff best did not beat H9 => finite-diff line fail
Phase 4 PZ runs did not beat H9 and did not satisfy segment gate => PZ line fail
Phase 6 cross-seq not started:
    no KITTI01 ATE <= 33.0
    no [200,300) improvement >= 8m with ATE <= 34.0
    no oracle/policy upper bound <= 32.5
```

最终结论：

```text
v13 found a small deployable online TTT improvement (C9, 33.7629),
but did not produce online target-25,
did not establish a controllable reset-window drift-correction action space,
and did not justify no-GT candidate selector or cross-seq.

TTT write should remain useful as a local/regularizing mechanism.
Target-25 mainline should move to explicit online trajectory-state / scale-state module.
```


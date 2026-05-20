# ACL2 v20 实验复盘：TTT ContextSkip SemanticMemory Target25

日期：2026-05-21（Asia/Singapore）  
计划文件：`docs/ACL2_v20_TTT_ContextSkip_SemanticMemory_Target25_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v20_ttt_contextskip_semanticmemory_target25/`

本轮原则：只记录实际落盘结果；不把 short rollout、sandbox oracle、GT audit、instrumentation smoke、proxy semantic mask、或 failed gate 写成 deployable online success。没有通过 sandbox gate 时，不启动 no-GT selector，也不启动 full online validation。

术语说明：

```text
TTT:
    Test-time training，测试时训练。这里指模型在跑序列时，用当前窗口产生的内部信号更新未来窗口会读到的 fast-weight memory。

HMC:
    Hybrid Memory Controller，混合记忆控制器。它负责在每个 chunk 里读写 fast weights，并记录每个 chunk 的状态 hash。

causal fork:
    从 full run 的某个 chunk 输入状态分叉，只跑未来短窗口。v16 已证明 HMC state + merge/gauge state + global chunk id 能复现 full-run 轨迹 suffix。

h10 / h15:
    从候选 chunk 开始向后看的 10 / 15 个 chunk 短滚动窗口。v20 只允许这些短滚动作为 diagnostic oracle，不算 full online success。

context source skip:
    受 VGGT4D 启发，在 attention 中只屏蔽一部分 source token，不删除 query token、不改变张量形状。
    hard mode 用 -1e4 attention bias 近似“跳过这些 source token”。
    soft mode 用较小权重衰减 source token。

semantic memory:
    计划目标是按语义类别区分 token，例如天空、植被、动态物体等。
    本轮代码中没有暴露精确 semantic group id，只能用已有 semantic scalar 近似 low-stuff + highD mask。
```

---

## 0. 工程与配置复盘

新增 / 修改：

```text
run_pipeline_abc_v2.py:
    新增 context-source-skip CLI 参数
    透传 context skip scope/mode/mask/layer/soft-rho 配置到 HMC
    支持 v20 候选的 run_config 边界字段

loger/models/pi3.py:
    新增 context source skip attention bias 构造
    支持 frame/chunk attention source-token 屏蔽
    保留 query rows、special/protected tokens，避免破坏输出形状
    hard skip 使用 -1e4 source-column bias
    soft skip 使用 log keep-ratio bias

loger/pipeline/hybrid_memory_controller.py:
    HybridMemoryControlPrior 增加 S_tok
    D_tok / P_ref / S_tok 传入模型 HMC control
    新增 support alias:
        full_chunk
        full_chunk_no_overlap
        past_plus_future_light
    新增 context source skip hook summary:
        num_context_source_skip_applied
        mean_context_source_keep_ratio
        max_context_source_skip_tokens
        num_context_empty_source_events

loger/pipeline/ttt_write_controller.py:
    补充 v20 scale-commit / context-skip 组合候选需要的 action 分支

tools/run_attention_cue_experiment.sh:
    透传 CONTEXT_SOURCE_SKIP_* 环境变量

tools/run_v20_candidate_rollout.sh:
    v20 trusted short-rollout launcher
    固定使用 v16 H9 causal-fork parent snapshots
    每个 run 写入 diagnostic_only_short_rollout / counts_as_online_ttt_write_success 等边界字段

tools/run_v20_matrix.sh:
    v20 matrix scheduler
    支持 Batch A support、Batch B context skip、Batch D scale/skip 组合

tools/v20_support_audit.py:
    support index coverage audit

tools/v20_candidate_bank_report.py:
    基于 v18 report 聚合 v20 h10/h15 metrics
    同 frame intersection 重新计算 delta vs K1_H9
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    run_pipeline_abc_v2.py \
    loger/pipeline/hybrid_memory_controller.py \
    loger/models/pi3.py \
    tools/v20_support_audit.py \
    tools/v20_candidate_bank_report.py

bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v20_candidate_rollout.sh
bash -n tools/run_v20_matrix.sh

PASS
```

矩阵落盘完整性：

| Prefix | rollout dirs | `kitti_benchmark.log` | `01.txt` |
|---|---:|---:|---:|
| `V20_A_SUPPORT_R1` | `20` | `20` | `20` |
| `V20_B_SMOKE_R1` | `1` | `1` | `1` |
| `V20_B_KVSKIP_R1` | `24` | `24` | `24` |
| `V20_B_KVSOFT_R1` | `4` | `4` | `4` |
| `V20_D_SCALE_R1` | `2` | `2` | `2` |
| `V20_D_COMBO_R1` | `4` | `4` | `4` |
| `V20_D_COMBO_R2` | `2` | `2` | `2` |

日志检查：

```text
matrix_logs 中未发现 Traceback / ERROR / RuntimeError / CUDA out of memory / FAIL
```

重要边界：

```text
所有 v20 row 都是 trusted short-rollout oracle diagnostic。
diagnostic_only_short_rollout = true。
counts_as_online_ttt_write_success = false。
没有启动 selector。
没有启动 full online validation。
```

---

## 1. Phase 0 Boundary

v20 复用 v16 已通过的 locked boundary 与 causal fork parity。当前最好可计数 full online TTT write 没有变化：

```text
C9_P0_R2
ATE = 33.7629421029m
counts_as_ttt_write = true
```

v20 candidate parent 仍使用 H9：

```text
H9_P0_R2
ATE = 34.1257769401m

原因：
H9 full ATE 比 C9 差，但 H9 在 [200,300) 病灶段更好，且 v16 causal fork snapshots 已完成轨迹级 parity。
```

---

## 2. Batch A Support Window Audit

输出：

```text
batchA_support_audit/support_indices.jsonl
batchA_support_audit/support_index_summary.csv
batchA_support_audit/support_index_summary.json
batchA_support_audit/support_index_audit.md
```

Support audit：

| Support | Rows | Count min | Count max | Count mean | Frames with future | Frames with past | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| `past_only` | `64` | `0` | `31` | `15.5` | `0` | `62` |  |
| `full` | `64` | `31` | `31` | `31.0` | `62` | `62` |  |
| `full_chunk` | `64` | `31` | `31` | `31.0` | `62` | `62` | explicit alias of full support inside current chunk |
| `full_chunk_no_overlap` | `64` | `31` | `31` | `31.0` | `62` | `62` | falls back to full_chunk because HMC cue builder has no external overlap metadata |
| `past_plus_future_light` | `64` | `31` | `31` | `31.0` | `62` | `62` | weighted centroid: 0.75 past + 0.25 future within current chunk |
| `near12` | `64` | `2` | `4` | `3.8125` | `62` | `62` |  |
| `near24` | `64` | `2` | `4` | `3.625` | `60` | `60` |  |

Batch A candidate result：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-0.43593138334250625` |
| Best h10/h15 candidate | `S1_00_C23_PAST`, chunk `6`, h`15` |
| Best `[200,300)` delta vs H9 | `-0.693624840597316` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Interpretation：

```text
Support window variants did not beat the original C23 past-only cue.
full_chunk_no_overlap is not a real no-overlap implementation yet; it falls back to full_chunk due to missing overlap seam metadata.
No selector/full online validation allowed.
```

---

## 3. Batch B Context Source Skip

Context source skip smoke run：

```text
V20_B_SMOKE_R1_KVS_02_FRAME_EARLY_DG_Q90_HARD_chunk5_h3_globalgate_H9parent_SWKS3
```

Smoke evidence from hook summary：

| Field | Value |
|---|---:|
| `implemented_paths` includes `context_source_skip` | `true` |
| `frame_attention.num_context_source_skip_applied` | `6` |
| `frame_attention.mean_context_source_keep_ratio` | `0.9666650295257568` |
| `frame_attention.max_context_source_skip_tokens` | `4013` |
| `frame_attention.num_context_empty_source_events` | `0` |
| `frame_attention.max_abs_bias` | `10000.0` |

说明：

```text
context source skip 确实进入了 frame attention。
hard skip 通过 -1e4 source-column bias 屏蔽 source token。
没有出现 empty source event。
```

### 3.1 Hard Skip Matrix

输出：

```text
batchB_kvskip_report/true_action_gate_summary.json
batchB_kvskip_report/candidate_vs_H9_delta_by_horizon.csv
```

Gate summary：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-0.7522421638232899` |
| Best h10/h15 candidate | `KVS_01_FRAME_EARLY_DG_Q80_HARD`, chunk `6`, h`15` |
| Best `[200,300)` delta vs H9 | `-3.440533198161674` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Key rows：

| Candidate | Chunk | Horizon | ATE delta vs H9 | `[200,300)` delta | `[400,600)` proxy delta |
|---|---:|---:|---:|---:|---:|
| `KVS_01_FRAME_EARLY_DG_Q80_HARD` | `10` | `10` | `-0.7335244625430164` | `-3.440533198161674` | `-0.5762800137900612` |
| `KVS_01_FRAME_EARLY_DG_Q80_HARD` | `10` | `15` | `+0.8351645148756539` | `-0.9395844189062785` | `+0.9104590883865811` |
| `KVS_02_FRAME_EARLY_DG_Q90_HARD` | `10` | `10` | `-0.538021836172657` | `-2.5890963720909994` | `-0.43005738097362567` |
| `KVS_02_FRAME_EARLY_DG_Q90_HARD` | `10` | `15` | `+1.0802412918537563` | `+0.07454002610523958` | `+1.1171722864053137` |
| `KVS_03_FRAME_EARLY_LOWSTUFF_HIGHD_HARD` | `10` | `10` | `-0.5677100090959719` | `-2.7635027280600184` | `-0.4530947279865707` |
| `KVS_03_FRAME_EARLY_LOWSTUFF_HIGHD_HARD` | `10` | `15` | `+1.1454205086597433` | `+0.0421825365730939` | `+1.1811508941388489` |
| `KVS_07_CHUNK_EARLY_DG_Q90_HARD` | `10` | `10` | `-0.6026635303569243` | `-2.8143117596215745` | `-0.5262743203936289` |
| `KVS_07_CHUNK_EARLY_DG_Q90_HARD` | `10` | `15` | `+0.9496881527439776` | `-0.31131165601840394` | `+0.9694172279831186` |

Interpretation：

```text
Context source skip has a real local effect, especially chunk10 h10 disease segment.
But h15 decay/regression is severe.
Hard skip did not pass gate.
```

### 3.2 Soft Fallback

按计划处理 hard skip h15 decay：尝试 soft fallback。

输出：

```text
batchB_kvsoft_report/true_action_gate_summary.json
batchB_kvsoft_report/candidate_vs_H9_delta_by_horizon.csv
```

Gate summary：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-0.5686290820475755` |
| Best h10/h15 candidate | `KVS_09_FRAME_EARLY_DG_Q90_SOFT_R025`, chunk `10`, h`10` |
| Best `[200,300)` delta vs H9 | `-2.7600784604837614` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Key rows：

| Candidate | Chunk | Horizon | ATE delta vs H9 | `[200,300)` delta | `[400,600)` proxy delta |
|---|---:|---:|---:|---:|---:|
| `KVS_09_FRAME_EARLY_DG_Q90_SOFT_R025` | `10` | `10` | `-0.5686290820475755` | `-2.7600784604837614` | `-0.4535027317865037` |
| `KVS_09_FRAME_EARLY_DG_Q90_SOFT_R025` | `10` | `15` | `+1.1078553778640448` | `-0.013684404456739685` | `+1.1485295950241436` |

Interpretation：

```text
Soft fallback did not fix h15 decay.
No selector/full online validation allowed.
```

### 3.3 Semantic Memory Limitation

计划中的 exact semantic memory role 需要按语义组区分 token，例如 sky / vegetation / movable / background。当前 HMC control path 暴露到模型侧的是 `S_tok` semantic scalar，而不是离散 semantic group id。

本轮已经做的 blocker 尝试：

```text
实现了 KVS_03_FRAME_EARLY_LOWSTUFF_HIGHD_HARD：
    mask = S_tok <= 0.45 and highD

这只是 low-stuff + highD 的 proxy，不是 exact sky/vegetation/movable group role。
```

结果：

```text
KVS_03 chunk10 h10:
    ATE delta = -0.5677100090959719
    [200,300) delta = -2.7635027280600184

KVS_03 chunk10 h15:
    ATE delta = +1.1454205086597433
    [200,300) delta = +0.0421825365730939
```

审计边界：

```text
不能把 KVS_03 写成完成了 exact semantic role validation。
它只是 semantic scalar proxy blocker follow-up。
下一步如要完成 Batch C，必须把 semantic group id 从 prior generator / MaskletOutput.G_sem 显式传入 HybridMemoryControlPrior 和模型 hmc_control。
```

---

## 4. Batch D Scale Anchor and Scale + Skip Combo

### 4.1 Scale Anchor Repeat

输出：

```text
batchD_scale_anchor_report/true_action_gate_summary.json
batchD_scale_anchor_report/candidate_vs_H9_delta_by_horizon.csv
```

Gate summary：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-2.0956269826087137` |
| Best h10/h15 candidate | `SCALECOMMIT_01_PZBASIS_HARM_W0_G025`, chunk `10`, h`10` |
| Best `[200,300)` delta vs H9 | `-2.009094806127834` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Rows：

| Candidate | Chunk | Horizon | ATE delta vs H9 | `[200,300)` delta | `[400,600)` proxy delta |
|---|---:|---:|---:|---:|---:|
| `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `10` | `10` | `-2.0956269826087137` | `-2.009094806127834` | `-3.0570166429246513` |
| `SCALECOMMIT_01_PZBASIS_HARM_W0_G025` | `10` | `15` | `-0.7803554203914125` | `-1.3449914511564174` | `-2.0575265300386114` |

Interpretation：

```text
v19 weak scale anchor is confirmed.
h10 improves strongly, but h15 effect decays.
```

### 4.2 Scale + Context Skip

输出：

```text
batchD_scale_skip_combo_report/true_action_gate_summary.json
batchD_scale_skip_combo_report/candidate_vs_H9_delta_by_horizon.csv
```

Gate summary：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-2.5220781811781947` |
| Best h10/h15 candidate | `TTTSS_03B_SCALECOMMIT_DGQ80_HARD`, chunk `10`, h`10` |
| Best `[200,300)` delta vs H9 | `-4.852495000715884` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Rows：

| Candidate | Chunk | Horizon | ATE delta vs H9 | `[200,300)` delta | `[400,600)` proxy delta |
|---|---:|---:|---:|---:|---:|
| `TTTSS_03_SCALECOMMIT_DGQ90_HARD` | `10` | `10` | `-2.4139452867187075` | `-4.083810301884981` | `-3.311969836045698` |
| `TTTSS_03_SCALECOMMIT_DGQ90_HARD` | `10` | `15` | `+0.17593481154431956` | `-1.2364986996678624` | `-0.8745118907478364` |
| `TTTSS_03B_SCALECOMMIT_DGQ80_HARD` | `10` | `10` | `-2.5220781811781947` | `-4.852495000715884` | `-3.3403945846268996` |
| `TTTSS_03B_SCALECOMMIT_DGQ80_HARD` | `10` | `15` | `-0.019716631966346654` | `-2.154195732316907` | `-0.9860045124731158` |

Gate distance：

```text
TTTSS_03B h10 ATE gate:
    observed = -2.5220781811781947
    required <= -3.0
    short by about 0.478m

TTTSS_03B [200,300) gate:
    observed = -4.852495000715884
    required <= -5.0
    short by about 0.148m

Downstream [400,600) proxy:
    observed = -3.3403945846268996
    no downstream regression

But h15:
    ATE delta = -0.019716631966346654
    effect is not durable.
```

Interpretation：

```text
This is the best v20 diagnostic signal.
It nearly reaches the h10 disease-segment gate but fails the exact gate and collapses by h15.
No selector/full online validation allowed.
```

### 4.3 Scale + Skip + Pair

按 blocker follow-up 继续尝试 pair cue combination。

输出：

```text
batchD_scale_skip_pair_combo_report/true_action_gate_summary.json
batchD_scale_skip_pair_combo_report/candidate_vs_H9_delta_by_horizon.csv
```

Gate summary：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-1.949811004360761` |
| Best h10/h15 candidate | `TTTSS_03C_SCALECOMMIT_DGQ80_HARD_PLUS_PAIR`, chunk `10`, h`10` |
| Best `[200,300)` delta vs H9 | `-1.6401779420848968` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Rows：

| Candidate | Chunk | Horizon | ATE delta vs H9 | `[200,300)` delta | `[400,600)` proxy delta |
|---|---:|---:|---:|---:|---:|
| `TTTSS_03C_SCALECOMMIT_DGQ80_HARD_PLUS_PAIR` | `10` | `10` | `-1.949811004360761` | `-1.6401779420848968` | `-2.934747629107825` |
| `TTTSS_03C_SCALECOMMIT_DGQ80_HARD_PLUS_PAIR` | `10` | `15` | `-0.6382339938224746` | `-0.8799295564872267` | `-1.939034357289227` |

Interpretation：

```text
Pair cue did not help the strongest scale+skip candidate.
It improved h15 ATE relative to TTTSS_03B but hurt the disease-segment h10 gain badly.
```

---

## 5. Downstream Phase Decision

| Phase | Status | Reason |
|---|---|---|
| Phase 0 Boundary | pass | reused v16 locked H9/C9/WINGAM boundary and causal fork parity |
| Batch A support | fail | support variants did not beat C23 past-only cue; best h10/h15 ATE delta `-0.436m` |
| Batch B context source skip | fail / weak local signal | best h10 disease delta `-3.441m`, but h15 regressed |
| Batch B soft fallback | fail | soft skip did not fix h15 decay |
| Batch C semantic role | partial blocker follow-up only | exact semantic group id unavailable; proxy KVS_03 failed |
| Batch D scale anchor | weak | h10 ATE `-2.096m`, h15 decays to `-0.780m` |
| Batch D scale + skip | closest but fail | best h10 ATE `-2.522m`, `[200,300)` `-4.852m`; both miss gate; h15 decays |
| No-GT selector | not started | no sandbox oracle gate pass |
| Full online validation | not started | forbidden by failed gate |

Boundary：

```text
No v20 short-rollout result counts as deployable online TTT success.
No GT-selected candidate is counted.
No no-GT selector was evaluated.
No full online validation was launched.
No online target-25 result was produced in v20.
Current best deployable online TTT write remains:
    C9_P0_R2
    ATE = 33.7629421029m
```

---

## 6. Final Decision

v20 的好消息：

```text
Context source skip is implemented and verified in frame attention.
It gives real local disease-segment improvement without downstream [400,600) regression in h10.

Best v20 diagnostic:
    TTTSS_03B_SCALECOMMIT_DGQ80_HARD
    chunk10 h10 ATE delta vs H9 = -2.5220781811781947m
    [200,300) delta vs H9 = -4.852495000715884m
    [400,600) proxy delta vs H9 = -3.3403945846268996m
```

v20 的关键负结果：

```text
The best h10 result still misses both formal gates:
    h10/h15 ATE delta required <= -3m
    observed best h10 = -2.5220781811781947m

    [200,300) delta required <= -5m
    observed best h10 = -4.852495000715884m

The effect is not durable:
    TTTSS_03B chunk10 h15 ATE delta = -0.019716631966346654m
```

Interpretation：

```text
VGGT4D-style source-token skipping is useful as a local stabilizer/filter.
Scale-commit plus Dg-q80 hard context skip is the strongest current short-window action.
However, it does not yet create a persistent trajectory correction over h15.

This means v20 does not justify a no-GT selector or a full online Target-25 validation.
The next direction should focus on durability:
    expose exact semantic group ids for semantic memory routing,
    add overlap seam metadata for true no-overlap support,
    or combine context skip with an explicit trajectory/scale-state module that persists beyond h10.
```


# Pipeline v2 / HMC Phase A-F 实验汇报报告

日期：2026-05-04  
数据集：KITTI Odometry sequence 01  
主结果目录：`results/kitti01_hmc_v2/`  
主运行脚本：`run_pipeline_abc_v2.py`  
指标：KITTI benchmark Sim(3) alignment 下的 ATE RMSE、Rot RMSE、RPE t、RPE r

---

## 0. 总览结论

Pipeline v2 的核心工作，是把原先较难解释的 write-control / memory-control 实验，重构成一个可审计的 two-pass Hybrid Memory Controller pipeline。整个 A-F 阶段可以概括为：

1. Phase A 证明新 HMC v2 runner 在无控制、identity hook、TTT-only write 下可以精确复现旧结果，工程正确性成立。
2. Phase B 用 dashboard 检查 Pass 1 是否能暴露可解释的 memory / cue 信号，发现 write prior 比 read cue 更可靠，read cue 需要谨慎验证。
3. Phase C 前半段发现 frame-attention read-path control 有真实信号，但常规 controlled commit 会污染未来 TTT memory；Phase C v5 通过 commit isolation 把 read-only ATE 从 `41.0733 m` 推到 `39.7820 m`。
4. Phase D 将 commit-safe read correction 与 branch0 TTT write 组合，最佳 D5-07 达到 `39.4903 m`，证明 read correction 和 branch0 write 互补。
5. Phase E 尝试修复 D5 的 rotation / endpoint 损伤，测试 reference protection、Gram modulation、residual write、sparse write 和 pair search；最佳 E5-resrel-b125 为 `39.4881 m`，只比 D5-07 改善 `0.0022 m`，说明 hand-rule co-design 已进入平台期。
6. Phase F 转向新 cue source discovery，系统拆解 old_dyn、测试 global attention cue、per-frame attention cue、update-needed write cue；最佳结果为 `F1_11_olddyn_addclip_b125 = 39.3103 m`，但仍未突破 `<39.0 m` gate。true-flow 仍 blocked，因为当前实现没有真正 RAFT/GMFlow residual。

总体上，Pipeline v2 从原始 LoGeR native `41.7502 m` 推进到 Phase F 最好 `39.3103 m`，总 ATE 改善 `2.4399 m`。最重要的科学结论不是某个 beta sweep，而是：

- read-path control 有效；
- controlled read 的 TTT side effect 不能直接 commit；
- 正确 commit 模式是 controlled output + probe/native memory commit，再加安全 branch0 write；
- 当前 hand-designed cue family 在 KITTI 01 上约进入 `39.3-39.5 m` 平台，需要真正外部 motion source 或学习式 reliability gate 才可能继续大幅突破。

---

## 1. 整体思路与实验 Motivation

### 1.1 为什么要做 Pipeline v2

在 Pipeline v2 之前，我们已经有了一批 GSL-WC / TTT write-control 相关实验，最好有效结果是 BL01：只对 TTT branch0 做 dynamic score 的 mean-preserving write prior，KITTI 01 ATE 达到 `41.3665 m`。这个结果证明了 memory write control 有价值，但它也暴露了几个问题：

1. 旧 pipeline 很难区分“当前 chunk 输出被修正”和“未来 memory 被改写”这两件事。
2. 很多控制信号会同时影响 read path 和 TTT update，导致实验结果可解释性不足。
3. 短序列、prefix、slice 上的提升经常不能转化为 full KITTI 01 提升。
4. 一些控制虽然改善 rotation，却显著伤害 ATE；一些控制改善 ATE，却伤害 rotation / endpoint。

所以 Pipeline v2 的 motivation 是：把 long-context HMC 的实验拆成可审计的因果链路。我们需要能回答：

- 一个 cue 到底是修正了当前输出，还是污染了未来 memory？
- read-path correction 和 write-path adaptation 是否真的互补？
- TTT state 是否应该继承 controlled forward 的 side effect？
- 短程 cue 是否具有 full-sequence 稳定性？
- 当前瓶颈是 controller 强度、commit protocol，还是 cue source 本身？

### 1.2 为什么采用 two-pass HMC protocol

Pipeline v2 的核心设计是 two-pass：

1. Pass 1 是 probe/native forward，用于读取当前 HMC state、生成 cue、记录 memory trace，并产生可安全提交的 native/probe memory。
2. Pass 2 是 controlled forward，用于对当前 chunk output 施加 read-path control。
3. commit 阶段单独决定下一 chunk 的 memory 来自哪里：controlled state、probe/native state，或者 probe/native cache 上再执行安全 TTT write。

这个拆分的动机是：long-sequence reconstruction 的当前输出和未来 memory 是两个不同问题。read-path control 可以帮助当前 chunk，但它在 controlled forward 内部产生的 fast-weight update 未必适合写入未来。Phase C v5 的结果最终证明，这个设计是整个 Pipeline v2 最关键的突破。

### 1.3 为什么先做 correctness，再做 cue search

HMC pipeline 涉及 frame attention、SWA read、TTT apply、chunk/global attention、TTT fast weights、state commit 等多个 hook 点。任何一个 identity hook 如果不能保持 no-op，后续所有非 identity 控制都会变得不可信。因此实验顺序必须是：

1. Phase A：证明新 runner 和 identity hooks 不改变 LoGeR / LoGeR* / BL01。
2. Phase B：先看 probe trace 是否有可解释信号。
3. Phase C：只允许 single read-path control，避免多个变量耦合。
4. Phase C v5：在 read cue 已有信号后，专门隔离 commit side effect。
5. Phase D：只在 commit-safe 规则下组合 read + write。
6. Phase E：发现 ATE/rotation trade-off 后，再做 protection / sparse / residual write。
7. Phase F：当 hand-rule co-design 进入平台期后，再寻找真正新 cue source。

这条路线看起来长，但它避免了“跑出一个数，却不知道为什么”的问题。

### 1.4 实验假设的演化

Pipeline v2 的核心假设经历了几次变化：

| 阶段 | 当时假设 | 实验后修正 |
|---|---|---|
| Phase A | 新 HMC runner 必须严格复现旧结果 | 成立，后续实验可信 |
| Phase B | Pass 1 可能暴露有用 read/write 信号 | write prior 更可解释，read cue 较噪 |
| Phase C v1-v4 | 找一个更好的 read cue 即可推进 | read cue 有信号，但短程 gate 会 false positive |
| Phase C v5 | RFR-100 可能被 controlled commit side effect 限制 | 成立，commit isolation 将 ATE 推到 `39.7820 m` |
| Phase D | commit-safe read 可以和 branch0 write 互补 | 成立，D5-07 达到 `39.4903 m` |
| Phase E | protection / residual / sparse write 可修复 rotation 并继续降 ATE | 只部分成立，ATE 进入平台 |
| Phase F | 新 cue source 可突破 old_dyn family | 未完全成立，old_dyn_addclip 仍最好 |

### 1.5 评价标准与停止逻辑

我们不只看单个 ATE 数字，还同时看：

- ATE RMSE：主指标；
- Rot RMSE：是否用 orientation 损伤换 ATE；
- RPE t / RPE r：局部相对误差；
- final aligned error：endpoint 是否恶化；
- 50/100/200-frame mean ATE：局部窗口是否真的改善；
- cue quality：dynamic mass、coverage、fragmentation、anchor collision、confidence correlation；
- commit hash：下一 chunk memory 是否按预期提交；
- identity hook max bias：工程路径是否 no-op。

停止逻辑也逐渐严格化：

- 64-frame / 256-frame win 不能直接晋级 full；
- stateful slice win 也不能直接晋级 full；
- cue 必须先过 full-run quality audit；
- 如果多个 hand-rule 只在 `39.45-39.55 m` 附近震荡，就停止继续扫；
- 如果没有真正 RAFT/GMFlow residual，不把 proxy flow 当 Phase F true-flow candidate。

---

## 2. 关键里程碑表

| 阶段 | 代表实验 | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | 结论 |
|---|---|---:|---:|---:|---:|---|
| Phase A | A3 HMC v2 LoGeR native | 41.7502 | 8.9928 | 92.3961 | 0.0084 | 新 runner 精确复现 LoGeR |
| Phase A | A5 TTT branch0 BL01 | 41.3665 | 8.9490 | 92.3947 | 0.0083 | 新 runner 精确复现旧 BL01 |
| Phase C v1 | RFR-100 controlled read | 41.0733 | 9.0158 | 92.3818 | 0.0085 | read-path 有效，但未过最初 gate |
| Phase C v5 | CM02 probe_native commit | 39.7820 | 9.7417 | 92.3953 | 0.0096 | commit isolation 产生关键突破 |
| Phase D v5 | D5-07 probe_ttt_write beta=1.25 | 39.4903 | 9.8299 | 92.3937 | 0.0097 | read + branch0 write 互补 |
| Phase E | E5-resrel-b125 | 39.4881 | 9.7984 | 92.3950 | 0.0097 | tiny numerical best，平台期 |
| Phase F | F1_11 old_dyn_addclip b=1.25 | 39.3103 | 9.7097 | 92.3912 | 0.0097 | 当前 Phase F 最好 |
| Phase F | F7_06 addclip + residual_reliability write | 39.3149 | 9.7586 | 92.3914 | 0.0097 | 追平但未超过 F1_11 |

---

## 3. Phase A：HMC v2 正确性与复现 Gate

### 3.1 实验目的

Phase A 的目标不是追求新指标，而是证明 `run_pipeline_abc_v2.py` 的 two-pass HMC pipeline 没有破坏原始模型行为。只有满足这些 gate，后续 read/write hook 结果才可信。

验证内容包括：

- two-pass no-control 中 Pass 1 / Pass 2 输出完全一致；
- probe pass 不提交 memory；
- identity hook 在 frame attention、SWA read、TTT apply、chunk/global attention 上完全 no-op；
- HMC v2 native 可复现 LoGeR；
- HMC v2 native 可复现 LoGeR*；
- HMC v2 TTT-only 可复现旧 BL01 branch0 write result。

### 3.2 工程改动

新增/完善日志文件：

- `hmc_config.yaml`
- `hmc_state_hash.jsonl`
- `hmc_probe_summary.jsonl`
- `hmc_control_summary.jsonl`
- `hmc_hook_identity_check.json`

`hmc_state_hash.jsonl` 记录：

- `hash_H_m_before_probe`
- `hash_H_m_after_probe`
- `hash_H_m_before_pass2`
- `hash_H_m_after_commit`
- `hash_H_next`

同时记录 Pass 1 / Pass 2 的 pose、pointmap、confidence 差异。

### 3.3 关键结果

| Run | 目的 | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | 结论 |
|---|---|---:|---:|---:|---:|---|
| A0/A1 unity64 | 64-frame two-pass no-control smoke | n/a | n/a | n/a | n/a | 3/3 chunks probe-safe，double-write-safe，Pass1/2 diff=0 |
| G2 identity hooks | full LoGeR identity hook parity | 41.7502 | 8.9928 | 92.3961 | 0.0084 | 与 A3 no-control 完全一致 |
| A3 native LoGeR | HMC v2 no-control LoGeR | 41.7502 | 8.9928 | 92.3961 | 0.0084 | 精确复现 |
| A4 native LoGeR* | HMC v2 no-control LoGeR* | 47.9793 | 5.8502 | 90.7286 | 0.0075 | 精确复现 |
| A5 TTT branch0 BL01 | HMC v2 TTT-only branch0 | 41.3665 | 8.9490 | 92.3947 | 0.0083 | 精确复现旧 BL01 |

### 3.4 诊断结果

G2 identity hook full run：

- `probe_no_commit_hash_equal`: 38 / 38 chunks true；
- `state_double_write_safe`: 38 / 38 chunks true；
- max Pass1/Pass2 pose translation diff: `0.0 m`；
- max Pass1/Pass2 pose matrix abs diff: `0.0`；
- output trajectory 与 A3 no-control byte-identical；
- frame attention、SWA read、TTT apply、chunk attention hook coverage 全部存在，max bias 全为 `0.0`。

### 3.5 Phase A 结论

Phase A 全部 correctness gates 通过。HMC v2 runner 是可信的，后续非 identity read/write hook 可以基于它继续实验。

---

## 4. Phase B：Probe Trace Dashboard 与 Signal Gate

### 4.1 实验目的

Phase B 用 Pass 1 生成 dashboard，检查 HMC memory / read cue / write cue 是否有可解释信号。它不是模型选择阶段，没有 ATE gate，而是为 Phase C 的 read-path 实验选择方向。

### 4.2 工程改动

新增 `tools/hmc_phase_b_dashboard.py`，输出：

- dashboard panels；
- `phase_b_config.json`；
- `phase_b_trace_summary.jsonl`；
- `phase_b_chunk_summary.csv`；
- `phase_b_trace_availability.json`。

### 4.3 Dashboard 覆盖片段

| Dashboard | 目的 | Chunks | Visualized frames |
|---|---|---:|---:|
| `HMC_B_dashboard_segments_0_200` | old MP hurt segment | 7 | 8 |
| `HMC_B_dashboard_segments_300_500_400_600` | win/hurt comparison | 21 | 16 |
| `HMC_B_dashboard_segments_800_1000` | late drift / hurt segment | 35 | 8 |

### 4.4 Chunk-level 统计

| Segment group | Mean `C_dyn` | Mean `C_dyn` p90 | Mean `C_unc` | Mean `C_anchor` | Mean `G_write` | Mean `attn_dyn` | Mean `ttt_update_proxy` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[0,200)` | 0.2846 | 0.7866 | 0.6359 | 0.2237 | 0.5522 | 0.4996 | 0.3726 |
| `[300,600)` sampled | 0.4310 | 0.8086 | 0.7715 | 0.1405 | 0.4520 | 0.4994 | 0.3483 |
| `[800,1000)` | 0.4347 | 0.8098 | 0.7866 | 0.1372 | 0.4472 | 0.4994 | 0.3513 |

### 4.5 观察结论

- `G_write` 和最终 `A_prior` 比较可解释：偏向 road / lower-image stable structure，降低 sky / horizon / uncertain region 写入。
- `C_dyn` 不是干净的 moving-object detector，经常在 sky / horizon / background structure 上也较高。
- `attn_dyn` 接近常数均值 `0.499`，不能直接作为强 controller source。
- read-side signal 噪声较大，因此 Phase C 只允许从 single-path read control 开始，不应直接组合多个 controller。

### 4.6 Phase B 结论

Dashboard 生成 PASS；Phase C readiness 是谨慎通过：工程 hook 已准备好，但 read cue 信号质量不足，需要单路径验证。

---

## 5. Phase C：Read-Path Single-Path Validation 到 Commit-Safe 突破

Phase C 是 Pipeline v2 中最长的一段，包含 v1 到 v5。它的核心问题逐步从“哪个 read cue 有效”转向“read control 的 side effect 应该如何 commit”。

---

### 5.1 Phase C v1：Frame Attention 与 TTT Apply 单路径验证

#### 目的

在 Phase A/B 通过后，测试单一路径 read control 能否改善 KITTI 01。此阶段只测试单路径，不做 read/write 组合。

#### 工程改动

新增：

- `--read_layer_mode all|early|middle|late|single`
- `--read_single_layer`
- frame-attention layer schedule
- TTT-apply residual/gate summary

#### Full KITTI 01 结果

| Run | Control | Strength | Layer | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r | Delta vs A3 | Delta vs BL01 | 结论 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| A3 native | none | n/a | n/a | 41.7502 | 8.9928 | 92.3961 | 0.0084 | 0.0000 | +0.3837 | reference |
| BL01 TTT write | branch0 dyn MP-01 | n/a | all | 41.3665 | 8.9490 | 92.3947 | 0.0083 | -0.3837 | 0.0000 | previous best |
| RFR-025 | frame attention | beta=0.25 | early | 41.1524 | 9.1499 | 92.3839 | 0.0085 | -0.5978 | -0.2141 | useful |
| RFR-050 | frame attention | beta=0.50 | early | 41.1323 | 9.1113 | 92.3808 | 0.0086 | -0.6179 | -0.2342 | useful |
| RFR-100 | frame attention | beta=1.00 | early | 41.0733 | 9.0158 | 92.3818 | 0.0085 | -0.6769 | -0.2932 | best Phase C v1 |
| RFR-125 | frame attention | beta=1.25 | early | 41.2165 | 9.0406 | 92.3866 | 0.0086 | -0.5337 | -0.1500 | over-strength |
| RFR-150 | frame attention | beta=1.50 | early | 41.2125 | 9.0046 | 92.3853 | 0.0086 | -0.5377 | -0.1540 | worse |
| RFR-200 | frame attention | beta=2.00 | early | 41.5707 | 8.9167 | 92.3880 | 0.0085 | -0.1795 | +0.2042 | too strong |
| R-TTTA-010 | TTT apply | rho=0.10 | all | 42.6125 | 8.6284 | 92.4558 | 0.0081 | +0.8623 | +1.2460 | ATE bad |
| R-TTTA-020 | TTT apply | rho=0.20 | all | 43.5418 | 8.0282 | 92.5090 | 0.0083 | +1.7916 | +2.1753 | bad |
| R-TTTA-030 | TTT apply | rho=0.30 | all | 44.8062 | 7.7155 | 92.5640 | 0.0091 | +3.0560 | +3.4397 | fail |

#### 结论

- frame-attention read control 有真实 ATE 信号；
- RFR-100 是 v1 最佳，较 A3 改善 `0.6769 m`，较 BL01 改善 `0.2932 m`；
- 但 RFR-100 未达到最初 `<41.0 m` gate；
- TTT-apply control 虽改善 rotation，但严重损害 ATE，不安全。

---

### 5.2 Phase C v2 FineGate：短程 false positive 暴露

#### 目的

围绕 RFR-100 继续细化 read cue、bias 方向、top-k dynamic mass、static/reference protection 与 early-layer schedule。

#### 新增 knobs

- `--read_cue_source dyn|dyn_reliable|internal_attn|key_cosine_avg|key_cosine_shallow|key_cosine_deep`
- `--frame_bias_mode pair|protected_pair|key|query`
- `--read_topk_frac`
- `--read_protect_static`
- `--read_layer_mode early_quarter|early_half`

#### 64-frame smoke

| Run | Cue | Bias | Layer | Strength | ATE RMSE (m) | Rot RMSE (deg) | 结论 |
|---|---|---|---|---:|---:|---:|---|
| F64-01 | `dyn` | pair | early | beta=1.0 | 1.0058 | 2.5517 | RFR smoke |
| F64-02 | `dyn` | key | early | beta=0.5 | 0.9548 | 2.6367 | useful |
| F64-04 | `dyn_reliable` | protected_pair | early | beta=1.0 | 0.9122 | 2.7108 | best 64-frame |
| F64-07 | `internal_attn` top10 | protected_pair | early_quarter | beta=1.0 | 0.9233 | 2.6854 | useful smoke |

#### 256-frame gate

| Run | Cue | Bias | Layer | Strength | ATE RMSE (m) | Rot RMSE (deg) | 结论 |
|---|---|---|---|---:|---:|---:|---|
| F256-01 | `dyn` | pair | early | beta=1.0 | 3.8666 | 4.2789 | RFR reference |
| F256-02 | `dyn_reliable` | protected_pair | early | beta=1.0 | 3.7941 | 3.7843 | best 256 |
| F256-07 | `dyn` | key | early | beta=0.5 | 3.8467 | 4.1613 | second best |

#### Full KITTI 01

| Run | Cue | Bias | Layer | Strength | ATE RMSE (m) | Rot RMSE (deg) | Delta vs RFR-100 | 结论 |
|---|---|---|---|---:|---:|---:|---:|---|
| FC2-01 | `dyn` | pair | early | beta=1.0 | 41.0733 | 9.0158 | 0.0000 | reference |
| FC2-02 | `dyn_reliable` | protected_pair | early | beta=1.0 | 41.4679 | 9.0387 | +0.3946 | failed full |
| FC2-03 | `dyn` | key | early | beta=0.5 | 41.2003 | 9.1682 | +0.1270 | worse than RFR |
| FC2-04 | `internal_attn` top10 | protected_pair | early_quarter | beta=1.0 | 41.8235 | 9.0160 | +0.7502 | failed |

#### 结论

短序列和 256-frame 的胜利不可靠。`dyn_reliable` 在 64/256 上最好，但 full sequence 反而差。Phase C v2 没有通过，不能进入 Phase D。

---

### 5.3 Phase C v3 SignalGate：新 cue 与 stateful slice gate

#### 目的

停止围绕旧 `C_dyn` family 微调，测试 Gram / entropy / flow proxy 等新 read cue，并加入 stateful slice gate。

#### 工程改动

新增 cue：

- `old_dyn`
- `gram_lite`
- `gram4d`
- `entropy`
- `flow`
- `flow_sem_veto`
- `random`
- `inverted_dyn`

新增：

- `--read_path frame|swa|chunk|ttt_apply`
- `--flow_model`
- `--flow_pair_stride`
- `--flow_fb_thr`
- `--flow_residual_thr`
- `--gram_layer_groups`
- stateful slice 保存/加载 HMC state
- `cue_quality_per_chunk.jsonl`
- `cue_quality_summary.json`
- `hook_effect_summary.jsonl`
- `hmc_correctness_summary.json`

重要限制：`flow_sem_veto` 是 patch-match / Stage-B reprojection proxy，不是真 RAFT/GMFlow。

#### Short gate

| Run | Cue | Bias | Frames | ATE RMSE (m) | Rot RMSE (deg) | 结论 |
|---|---|---|---:|---:|---:|---|
| RFR-100 | old `C_dyn` | pair | 64 | 1.0058 | 2.5517 | reference |
| C31B | `gram_lite` | key | 64 | 0.9796 | 2.7205 | useful |
| C31C | `gram4d` | key | 64 | 0.9386 | 2.6826 | strong smoke |
| C31F | `flow_sem_veto` proxy | key | 64 | 0.9163 | 2.7308 | best smoke |
| RFR-100 | old `C_dyn` | pair | 256 | 3.8666 | 4.2789 | reference |
| C31F | `flow_sem_veto` proxy | key | 256 | 3.8000 | 3.9007 | best 256 |

#### Stateful slice gate for `flow_sem_veto`

| Chunk | Frames | No-control ATE (m) | Candidate ATE (m) | Delta (m) | Improved |
|---:|---|---:|---:|---:|---|
| 0 | `[0,128)` | 1.0204 | 1.0490 | +0.0286 | no |
| 5 | `[145,273)` | 41.0849 | 41.0653 | -0.0196 | yes |
| 10 | `[290,418)` | 29.6698 | 29.4409 | -0.2289 | yes |
| 15 | `[435,563)` | 16.6805 | 16.6076 | -0.0729 | yes |
| 20 | `[580,708)` | 34.2306 | 34.0583 | -0.1723 | yes |
| 25 | `[725,853)` | 35.9329 | 35.8356 | -0.0973 | yes |
| 30 | `[870,998)` | 4.8915 | 4.8408 | -0.0507 | yes |

#### Full KITTI 01

| Run | Cue | Path | Bias | Layer | Strength | ATE RMSE (m) | Rot RMSE (deg) | Delta vs RFR-100 | 结论 |
|---|---|---|---|---|---:|---:|---:|---:|---|
| RFR-100 | old `C_dyn` | frame | pair | early | 1.0 | 41.0733 | 9.0158 | 0.0000 | reference |
| FC3-02 | `flow_sem_veto` proxy | frame | key | early | 1.0 | 41.4333 | 9.1252 | +0.3600 | failed full |

#### 结论

即使 64-frame、256-frame、6/7 stateful slices 都过，full sequence 仍然失败。`flow_sem_veto` mean dynamic mass 只有 `0.010`，太稀疏。Phase C v3 证明 slice gate 也会 false positive，需要 full-run cue quality audit。

---

### 5.4 Phase C v4 Global-Safe Cue Gate

#### 目的

在跑 full candidate 前先做全序列 cue quality audit，防止 v3 这种短程 / slice false positive。

#### 新增 cue 和工具

新增：

- `flow_proxy`
- `flow_sem_veto_proxy`
- `flow_proxy_calib`
- `flow_sem_veto_calib`
- `old_dyn_plus_flow_proxy`
- `old_dyn_switch_flow_proxy`

新增质量指标：

- mean dynamic mass；
- chunk coverage；
- anchor collision；
- fragmentation；
- Corr(D,Conf)；
- old_dyn IoU / coverage / recall；
- fallback rate。

#### Full-run stateful probe cue audit

| Run | Cue | Target / Blend | Mean mass `D>0.5` | Coverage | Anchor collision | Fragmentation | Corr(D,Conf) | IoU old_dyn | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| P-A | `old_dyn` | native | 0.4650 | 1.0000 | 0.0328 | 0.0141 | 0.0004 | 1.0000 | reference |
| P-B | `flow_proxy_calib` | q=0.06/tau=0.05 | 0.0601 | 1.0000 | 0.1060 | 0.2363 | 0.2071 | 0.1241 | fragmentation fail |
| P-C | `old_dyn_plus_flow_proxy` | lambda=0.50 | 0.4434 | 1.0000 | 0.0412 | 0.0180 | 0.0520 | 0.8537 | too close to old_dyn |
| P-D | `old_dyn_switch_flow_proxy` | q=0.06 | 0.1362 | 0.5789 | 0.0642 | 0.1673 | 0.2745 | 0.3136 | coverage/fragmentation fail |
| P-E | `flow_proxy_calib` | q=0.10/tau=0.10 | 0.0993 | 1.0000 | 0.1149 | 0.1774 | 0.2063 | 0.1857 | still fragmented |
| P-F | `old_dyn_switch_flow_proxy` | q=0.10/tau=0.10 | 0.1463 | 0.6053 | 0.0600 | 0.1638 | 0.2949 | 0.3548 | fails |

#### 256-frame controlled gate

| Run | Cue | ATE RMSE (m) | Rot RMSE (deg) | Mean mass | Fragmentation | Delta vs RFR-256 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| G256-00 | `old_dyn` | 3.8666 | 4.2789 | 0.3009 | 0.0246 | 0.0000 | reference |
| G256-01 | `flow_proxy_calib q=0.06` | 3.7706 | 4.0398 | 0.0603 | 0.1709 | -0.0960 | best 256, fails audit |
| G256-02 | `flow_proxy_calib q=0.06` | 3.8251 | 4.1699 | 0.0603 | 0.1705 | -0.0415 | fails audit |
| G256-04 | `old_dyn+0.50 flow_proxy` | 3.8497 | 4.1199 | 0.2234 | 0.0365 | -0.0169 | tiny gain |
| G256-06 | `flow_proxy_calib q=0.10` | 3.8616 | 4.1451 | 0.0992 | 0.1307 | -0.0050 | no useful gain |

#### 结论

没有新 cue 同时满足 full-run cue quality 和 meaningful 256-frame gain。Phase C v4 不跑 full candidates 是正确的。proxy flow 有短程信号，但不是稳定全序列 cue。

---

### 5.5 Phase C v5 Commit-Safe Read-Path

#### 目的

重新审视 RFR-100：它是不是 cue 不够强，还是 controlled read 的 TTT side effect 污染未来 memory？Phase C v5 重点测试 commit mode。

#### 工程改动

新增：

- `--hmc_commit_mode controlled|probe_native|split_ttt_native`
- TTT state relative diff logging
- commit-state hash diagnostics

三种 commit：

- `controlled`: 提交 Pass-2 controlled state；
- `probe_native`: 输出用 Pass-2 controlled geometry，但提交 Pass-1 native/probe state；
- `split_ttt_native`: 输出用 controlled geometry，但 TTT fast weights 用 native/probe。当前数值等价于 `probe_native`。

#### Correctness reproduction

| Run | Model | Commit mode | ATE RMSE (m) | Rot RMSE (deg) | HMC correctness | 结论 |
|---|---|---|---:|---:|---|---|
| C50A | LoGeR | controlled / no-control | 41.7502 | 8.9928 | 38/38 safe | exact A3 |
| C50B | LoGeR* | controlled / no-control | 47.9793 | 5.8502 | 18/18 safe | exact A4 |

#### Commit-mode isolation

| Run | Commit mode | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r | Delta vs CM01 | Mean TTT side effect | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| CM01 | controlled | 41.0733 | 9.0158 | 92.3818 | 0.0085 | 0.0000 | 0.018061 | RFR-100 reproduced |
| CM02 | probe_native | 39.7820 | 9.7417 | 92.3953 | 0.0096 | -1.2913 | 0.018118 | strong pass |
| CM03 | split_ttt_native | 39.7820 | 9.7417 | 92.3953 | 0.0096 | -1.2913 | 0.018118 | same as CM02 |
| CM02R | probe_native repeat | 39.7820 | 9.7417 | 92.3953 | 0.0096 | -1.2913 | 0.018118 | deterministic |

Trajectory diagnostics：

| Run | Aligned ATE (m) | Sim(3) scale | Final error (m) | 200-frame mean ATE (m) |
|---|---:|---:|---:|---:|
| RFR100 controlled | 41.0733 | 30.659544 | 2.907 | 36.868 |
| RFR100 probe_native | 39.7820 | 30.762480 | 5.589 | 35.299 |
| RFR100 split_ttt_native | 39.7820 | 30.762480 | 5.589 | 35.299 |

#### 结论

Phase C v5 是 Pipeline v2 的第一个关键突破：

- RFR-100 read signal 本身很强；
- 问题在于 controlled read forward 产生的 TTT state 不适合提交到未来 memory；
- 只改当前 output、不污染未来 memory 后，ATE 从 `41.0733 m` 降到 `39.7820 m`；
- 但 trade-off 是 rotation 和 final error 变差。

Phase C v5 通过 `<40.0 m` main-candidate precondition，因此 Phase D 可以启动，但必须保留 commit isolation。

---

## 6. Phase D：Commit-Safe Read + Branch0 Write

### 6.1 实验目的

Phase D 目标是把 Phase C v5 的 read-path correction 与 BL01 branch0 TTT write 组合，同时不重新引入 controlled read memory contamination。

关键设计：

- 当前输出：Pass-2 controlled geometry；
- 未来 memory：Pass-1 native/probe TTT cache；
- TTT write：在 probe cache 上显式执行 branch0 controlled write；
- commit mode：`probe_ttt_write`。

### 6.2 工程改动

新增：

- `--hmc_commit_mode probe_ttt_write`
- hybrid mode trace 中同时报告 `ttt_update` 和 active read hook path；
- smoke check 确认 `hash_H_next` 不等于 pure controlled，也不等于 pure probe_native，且 `prior_ttt_write_present=true`。

### 6.3 Full KITTI 01 结果

| Run | Mode | Commit | Beta | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r | Delta vs BL01 | Delta vs CM02 | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| BL01 TTT write | TTT branch0 | controlled | n/a | 41.3665 | 8.9490 | 92.3947 | 0.0083 | 0.0000 | +1.5845 | write baseline |
| D5-02 | hybrid | controlled | 1.00 | 41.0221 | 9.0510 | 92.3842 | 0.0085 | -0.3444 | +1.2401 | naive commit contaminated |
| D5-03 | hybrid | probe_native | 1.00 | 39.7820 | 9.7417 | 92.3953 | 0.0096 | -1.5845 | 0.0000 | equals read-only |
| D5-04 | hybrid | probe_ttt_write | 1.00 | 39.5127 | 9.7345 | 92.3937 | 0.0096 | -1.8538 | -0.2693 | strong combo |
| D5-05 | hybrid | probe_ttt_write | 0.75 | 39.5861 | 9.6165 | 92.3932 | 0.0094 | -1.7804 | -0.1959 | more rotation-safe |
| D5-06 | hybrid | probe_ttt_write | 0.50 | 39.7739 | 9.4839 | 92.3934 | 0.0092 | -1.5926 | -0.0081 | little combo gain |
| D5-07 | hybrid | probe_ttt_write | 1.25 | 39.4903 | 9.8299 | 92.3937 | 0.0097 | -1.8762 | -0.2917 | best ATE |
| D5-08 | hybrid | probe_ttt_write | 1.50 | 39.5279 | 9.9083 | 92.3940 | 0.0098 | -1.8386 | -0.2541 | too much rotation damage |
| BE-H | hybrid | probe_ttt_write | bias-energy norm | 39.5251 | 9.7667 | 92.3940 | 0.0097 | -1.8414 | -0.2569 | worse than fixed beta |

### 6.4 Trajectory diagnostics

| Run | Aligned ATE (m) | Final error (m) | 50-frame mean ATE (m) | 100-frame mean ATE (m) | 200-frame mean ATE (m) |
|---|---:|---:|---:|---:|---:|
| BL01 | 41.3665 | 3.590 | 36.704 | 37.170 | 37.177 |
| ReadOnly_CM02 | 39.7820 | 5.589 | 34.887 | 35.380 | 35.299 |
| D5_b100 | 39.5127 | 6.058 | 34.599 | 35.094 | 35.042 |
| D5_b125 | 39.4903 | 6.311 | 34.590 | 35.081 | 35.043 |

Worst D5-b125 chunks：

| Chunk | Frames | RMSE (m) | Worst frame |
|---:|---|---:|---:|
| 8 | `[232,264)` | 87.73 | 255 |
| 9 | `[261,293)` | 74.47 | 264 |
| 7 | `[203,235)` | 70.92 | 228 |
| 15 | `[435,467)` | 65.06 | 463 |
| 16 | `[464,496)` | 63.54 | 464 |

### 6.5 Phase D 结论

- `probe_ttt_write` 是正确组合模式；
- D5-07 为当时最佳 ATE：`39.4903 m`；
- 相比 BL01 改善 `1.8762 m`；
- 相比 Phase C v5 read-only CM02 改善 `0.2917 m`；
- D5-04 更平衡，ATE `39.5127 m`，rotation `9.7345 deg`；
- ATE 提升伴随 rotation / final error 变差，下一阶段需要 reference / rotation protection；
- 未达到 `<38.0 m` main-candidate bar。

---

## 7. Phase E：Read/Write Co-Design 与 Rotation-Safe 尝试

### 7.1 实验目的

Phase E 目标是保留 Phase D 的 ATE gain，同时降低 rotation 和 endpoint 损伤。实验覆盖：

- read-side reference protection；
- read cue modulation；
- update-needed TTT residual write；
- sparse exact-preserve write；
- read/write pair search。

所有有效 full runs 继续使用 Phase D 的 commit rule：controlled Pass-2 geometry output + `probe_ttt_write` commit。

### 7.2 工程改动

新增：

- `--read_protection_mode none|overlap|anchor|high_anchor|static|reset|ref|attention|attn|combined_light|combined_strong`
- `--read_ref_strength`
- `--read_overlap_frames`
- `--read_reset_frames`
- `--read_attention_q`
- `--hmc_write_score_source stage_d|old_dyn|ttt_residual|residual_reliability|alignment_confidence`
- `--hmc_write_sparse_ratio`
- `--hmc_write_sparse_mode`
- `old_dyn_gram_lite_agree`
- `old_dyn_gram4d_agree`
- `old_dyn_key_static_rescue`

### 7.3 Full KITTI 01 结果

| Run | 类别 | Read / Write variant | Beta | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| E0-D5-04R | repeat | D5 old_dyn + BL01 write | 1.00 | 39.5127 | 9.7345 | 92.3937 | 0.0096 | exact repeat |
| E0-D5-07R | repeat | D5 old_dyn + BL01 write | 1.25 | 39.4903 | 9.8299 | 92.3937 | 0.0097 | exact repeat |
| E1-overlap | read protection | overlap protected read + BL01 | 1.00 | 40.5264 | 9.2121 | 92.3874 | 0.0086 | rotation improves, ATE fails |
| E1-anchor | read protection | anchor protected read + BL01 | 1.00 | 39.5291 | 9.7454 | 92.3936 | 0.0096 | near D5-04 |
| E1-combined-light | read protection | overlap+anchor+attention + BL01 | 1.00 | 41.4207 | 8.9605 | 92.3946 | 0.0084 | rotation-safe, ATE too bad |
| E3-residual-reliability | update-needed write | old_dyn + residual x reliability | 1.00 | 39.5209 | 9.7076 | 92.3949 | 0.0096 | slight rotation help |
| E4-oldDyn-sparse95 | sparse write | old_dyn + oldDyn sparse95 | 1.00 | 39.5307 | 9.6327 | 92.3934 | 0.0095 | rotation help, ATE worse |
| E4-resrel-sparse95 | sparse write | old_dyn + residual sparse95 | 1.00 | 39.6552 | 9.4967 | 92.3947 | 0.0093 | balanced-ish, ATE too high |
| E5-anchor-sparse95 | pair search | anchor read + oldDyn sparse95 | 1.00 | 39.5111 | 9.6396 | 92.3935 | 0.0095 | best balanced-ish |
| E5-anchor-resrel | pair search | anchor read + residual write | 1.00 | 39.7182 | 9.7850 | 92.3958 | 0.0096 | fails |
| E5-resrel-b125 | pair search | old_dyn + residual write | 1.25 | 39.4881 | 9.7984 | 92.3950 | 0.0097 | tiny new best |
| E5-oldDyn-sparse95-b125 | pair search | old_dyn + oldDyn sparse95 | 1.25 | 39.5445 | 9.7486 | 92.3937 | 0.0097 | worse ATE |
| E2-oldDyn-GramLite | modulation | oldDyn x Gram-lite + BL01 | 1.00 | 39.8441 | 9.6976 | 92.3944 | 0.0095 | fails |
| E2-oldDyn-Gram4D | modulation | oldDyn x Gram4D + BL01 | 1.00 | 39.8048 | 9.6467 | 92.3927 | 0.0094 | fails |
| E2-key-static-rescue | modulation | key/static rescue + BL01 | 1.00 | 39.5127 | 9.7345 | 92.3937 | 0.0096 | identity-equivalent |

### 7.4 Trajectory diagnostics

| Run | Aligned ATE (m) | Final error (m) | 50-frame mean ATE (m) | 100-frame mean ATE (m) | 200-frame mean ATE (m) |
|---|---:|---:|---:|---:|---:|
| BL01 | 41.3665 | 3.590 | 36.704 | 37.170 | 37.177 |
| D5_b100 | 39.5127 | 6.058 | 34.599 | 35.094 | 35.042 |
| D5_b125 | 39.4903 | 6.311 | 34.590 | 35.081 | 35.043 |
| E5_resrel_b125 | 39.4881 | 6.309 | 34.597 | 35.088 | 35.037 |
| E5_anchor_sparse_b100 | 39.5111 | 5.967 | 34.578 | 35.072 | 34.996 |

### 7.5 Phase E 结论

- E5-resrel-b125 达到 `39.4881 m`，只比 D5-07 好 `0.0022 m`，是数值 best，但不是实质突破。
- reference / overlap protection 能改善 rotation，但 ATE 损失明显。
- sparse write 改善 rotation / final error，但不能改善 ATE。
- Gram-lite / Gram4D modulation 没有帮助。
- true-flow 仍 blocked，因为当前只有 patch-match / Stage-B proxy flow。
- Phase E stop condition hit：没有达到 `<39.0 m`，也没有达到 `<38.0 m` main candidate bar。

---

## 8. Phase F：New Cue Source 与 Training-Free Reliability

### 8.1 实验目的

Phase F 的目标不是继续修补 old_dyn hand-rule，而是回答：

- old_dyn 的有效性来自 explicit geometry residual、implicit attention cue，还是 fusion？
- global attention / Gram / QK 是否能成为独立 read cue？
- per-frame attention cue 是否能替代 old_dyn？
- update-needed write cue 是否比 read-side dynamic suppression 更好？
- true optical flow / epipolar residual 是否能成为新 external motion cue？

### 8.2 Phase F 总体完成情况

结果目录：`results/kitti01_hmc_v2/phaseF_newcue_trainingfree/`

截至 2026-05-04：

- 共 42 个 Phase F result dirs；
- 每个目录都有 `01.txt` 和 `kitti_benchmark.log`；
- 最近一次中断没有留下未完成 runnable gap；
- GPU 0-3 检查时为空闲；
- F4 true-flow 仍 blocked，原因是缺少真正 RAFT/GMFlow residual。

### 8.3 F0：Correctness / Baseline Gate

| Run | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r | 结论 |
|---|---:|---:|---:|---:|---|
| F0A no-control LoGeR | 41.7502 | 8.9928 | 92.3961 | 0.0084 | exact A3 |
| F0B no-control LoGeR* | 47.9793 | 5.8502 | 90.7286 | 0.0075 | exact A4 |
| F0C identity hooks | 41.7502 | 8.9928 | 92.3961 | 0.0084 | exact identity |
| F0D D5-07 repro | 39.4903 | 9.8299 | 92.3937 | 0.0097 | exact D5 reproduction |
| F0E E5-resrel-b125 repro | 39.4881 | 9.7984 | 92.3950 | 0.0097 | exact E5 reproduction |

结论：Phase F 新代码没有破坏 HMC baseline。

### 8.4 F1：old_dyn decomposition and fusion

| Run | Read cue / Write | Beta | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| F1_01 | explicit_dyn_only + BL01 | 1.00 | 39.4191 | 9.5794 | 92.3956 | 0.0094 | explicit 分支很强 |
| F1_02 | implicit_dyn_only + BL01 | 1.00 | 39.8990 | 9.5933 | 92.3963 | 0.0095 | 有信号但弱 |
| F1_03 | old_dyn_calibrated_soft_or + BL01 | 1.00 | 39.5127 | 9.7345 | 92.3937 | 0.0096 | D5 b1.0 reference |
| F1_04 | explicit_dyn_only + BL01 | 1.25 | 39.4214 | 9.6600 | 92.3966 | 0.0096 | 稳定 |
| F1_05 | implicit_dyn_only + BL01 | 1.25 | 39.8656 | 9.6406 | 92.3962 | 0.0095 | 不够 |
| F1_06 | old_dyn_calibrated_soft_or + BL01 | 1.25 | 39.4903 | 9.8299 | 92.3937 | 0.0097 | D5 b1.25 reference |
| F1_07 | old_dyn_max + BL01 | 1.25 | 39.5613 | 9.7416 | 92.3924 | 0.0096 | 不如 calibrated |
| F1_08 | old_dyn_soft_or + BL01 | 1.25 | 39.4147 | 9.8046 | 92.3907 | 0.0097 | 强 |
| F1_09 | implicit_dyn_only + residual_reliability | 1.25 | 39.9830 | 9.6336 | 92.3990 | 0.0095 | fails |
| F1_10 | old_dyn_avg + BL01 | 1.25 | 39.6868 | 9.7195 | 92.3939 | 0.0096 | fails |
| F1_11 | old_dyn_addclip + BL01 | 1.25 | 39.3103 | 9.7097 | 92.3912 | 0.0097 | Phase F best |

F1 结论：

- explicit 分支比 implicit 分支更接近主有效信号；
- fusion form 很重要；
- `old_dyn_addclip` 是当前最强 read cue；
- `implicit_dyn_only` 不能单独替代 old_dyn。

### 8.5 F2：Global / Gram / QK cue

| Run | Cue / Mode | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| F2_01 | dyn4d_patch read-only | 40.2555 | 9.5389 | 92.4008 | 0.0094 | fails |
| F2_02 | dyn4d_patch hybrid | 39.9985 | 9.5368 | 92.3987 | 0.0094 | 接近 40，但不够 |
| F2_03 | qk_var read-only | 40.2060 | 9.5394 | 92.4011 | 0.0094 | fails |
| F2_04 | gram4d read-only | 41.1249 | 9.2537 | 92.3952 | 0.0087 | rotation 好，ATE 差 |
| F2_05 | gram4d hybrid | 40.8677 | 9.2598 | 92.3947 | 0.0087 | fails |
| F2_06 | qqkk_disagree read-only | 40.8638 | 9.4104 | 92.3970 | 0.0089 | fails |

F2 结论：

- global/Gram/QK cue 没有成为 primary read cue；
- 部分 Gram cue 改善 rotation，但 ATE 明显不够；
- 不能进入 F8 主候选。

### 8.6 F3：Per-frame attention cue

| Run | Cue / Mode | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| F3_01 | manual_implicit read-only | 40.1474 | 9.5945 | 92.3976 | 0.0095 | fails |
| F3_02 | manual_implicit hybrid | 39.8990 | 9.5933 | 92.3963 | 0.0095 | weak |
| F3_03 | key_cosine_avg read-only | 39.7820 | 9.7417 | 92.3953 | 0.0096 | 有信号 |
| F3_04 | key_cosine_shallow read-only | 39.7820 | 9.7417 | 92.3953 | 0.0096 | 有信号 |
| F3_05 | key_cosine_deep read-only | 39.7820 | 9.7417 | 92.3953 | 0.0096 | 有信号 |
| F3_06 | entropy key read-only | 41.0887 | 9.9225 | 92.3972 | 0.0097 | fails |
| F3_07 | shallow_deep_disagree read-only | 41.7316 | 8.9914 | 92.3960 | 0.0084 | 接近 no-control |
| F3_08 | key_cosine_avg hybrid | 39.8103 | 9.7032 | 92.3945 | 0.0096 | safe write 后变差 |
| F3_08 query | query_cosine_avg read-only | 41.7316 | 8.9914 | 92.3960 | 0.0084 | no-control-like |
| F3_09 | query_cosine_shallow read-only | 41.7316 | 8.9914 | 92.3960 | 0.0084 | no-control-like |
| F3_10 | query_cosine_deep read-only | 41.7316 | 8.9914 | 92.3960 | 0.0084 | no-control-like |

F3 结论：

- key-cosine cue 有真实 read signal，但强度只到 `39.7820 m`；
- query cosine 和 shallow/deep disagreement 基本没有主 cue 价值；
- entropy 不可靠；
- key-cosine + safe write 没有超过 read-only，更没有超过 old_dyn family。

### 8.7 F4：True optical flow / epipolar residual

F4 未正式运行合法 full candidate。

原因：

- 当前代码中 `flow` / `flow_sem_veto` 仍是 patch-match / Stage-B reprojection proxy；
- Phase F plan 明确要求 RAFT 或 GMFlow 级别 frozen true optical flow residual；
- `flow_model=patch_match` 不能作为 Phase F full candidate；
- 因此 true-flow 分支是 blocked，不是实验失败。

### 8.8 F5 / F6：Semantic reliability 与 promotion gate

Phase F 中没有形成独立 F5 主候选。F6 promotion gate 的实际作用是根据 F1-F3/F7 结果决定是否进入 F8：

- F1 `old_dyn_addclip` 成为当前最好 read cue；
- F2 global cue 未通过；
- F3 key-cosine 有信号但不够；
- F4 true-flow blocked；
- F7 write cue 没有带来 `<39.0 m` 级别提升；
- 因此 F8 大规模 pair search 没有触发。

### 8.9 F7：Update-needed write from probe cache

| Run | Read cue | Write cue | Sparse | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r | 结论 |
|---|---|---|---|---:|---:|---:|---:|---|
| F7_02 | old_dyn | TTT residual | dense | 39.8884 | 9.8843 | 92.3957 | 0.0098 | fails |
| F7_04 | old_dyn | alignment_confidence | dense | 39.7311 | 9.6980 | 92.3930 | 0.0096 | 不够 |
| F7_05 | old_dyn | residual_reliability | sparse95 | 39.7822 | 9.5632 | 92.3943 | 0.0094 | rotation 好，ATE 差 |
| F7_06 | old_dyn_addclip | residual_reliability | dense | 39.3149 | 9.7586 | 92.3914 | 0.0097 | 追平 F1_11，未超过 |
| F7_07 | old_dyn_addclip | residual_reliability | sparse95 | 39.3864 | 9.7821 | 92.3940 | 0.0097 | sparse loses ATE |

F7 结论：

- update-needed write 没有产生实质突破；
- dense residual_reliability 能追平 `old_dyn_addclip`，但不是新 best；
- sparse exact-preserve 有一定 rotation-side 价值，但损失 ATE；
- F7 没有达到 `<39.0 m` gate。

### 8.10 Phase F 总结表

| 类别 | 最好 run | ATE RMSE (m) | Rot RMSE (deg) | 结论 |
|---|---|---:|---:|---|
| Baseline reproduction | F0E E5 repro | 39.4881 | 9.7984 | Phase E exact repeat |
| Explicit dyn | F1_01 / F1_04 | 39.4191 / 39.4214 | 9.5794 / 9.6600 | 很强，rotation 更好 |
| Implicit dyn | F1_05 | 39.8656 | 9.6406 | 有信号但不够 |
| old_dyn fusion | F1_11 old_dyn_addclip | 39.3103 | 9.7097 | 当前最好 |
| Global cue | F2_02 dyn4d hybrid | 39.9985 | 9.5368 | 不够 |
| Per-frame key-cosine | F3_03/04/05 | 39.7820 | 9.7417 | 有信号，不能替代 old_dyn |
| Update-needed write | F7_06 | 39.3149 | 9.7586 | 追平但没突破 |
| Sparse write | F7_07 | 39.3864 | 9.7821 | ATE 变差 |
| True flow | n/a | n/a | n/a | blocked |

### 8.11 Phase F 结论

- Phase F 最好结果为 `F1_11_olddyn_addclip_b125 = 39.3103 m`；
- old_dyn family 仍然是当前最有效的 cue family；
- explicit 分支很强，说明有效信号主要不是纯 implicit attention；
- global / Gram / QK cue 没有成为独立主 cue；
- per-frame key-cosine 有信号但不够强；
- residual_reliability write 只能追平，不能突破；
- true-flow 是最重要的未完成分支，但需要真正 RAFT/GMFlow residual；
- 当前手工 training-free cue search 进入 `39.3-39.5 m` 平台。

---

## 9. 跨阶段核心发现

### 9.1 read-path control 是有效方向

Phase C v1 已经证明 frame-attention read control 相比 A3 native 和 BL01 有明显 ATE 改善。Phase C v5 进一步证明，在正确 commit isolation 下，这个 read signal 可以达到 `39.7820 m`。

### 9.2 commit isolation 是 Pipeline v2 的关键突破

同样的 RFR-100 read control：

| Commit mode | ATE RMSE (m) | 解释 |
|---|---:|---|
| controlled | 41.0733 | controlled read side effect 被写入未来 memory |
| probe_native | 39.7820 | 当前输出受控，未来 memory 不污染 |
| split_ttt_native | 39.7820 | 与 probe_native 等价 |

这说明 read correction 可以用来修当前输出，但不能让 controlled forward 的 TTT state 直接变成下一 chunk memory。

### 9.3 branch0 write 与 read correction 互补

Phase D `probe_ttt_write` 把 CM02 的 `39.7820 m` 进一步推到 D5-07 的 `39.4903 m`。这说明 branch0 TTT write 不是完全应该丢弃，而是必须在 probe/native cache 上安全执行。

### 9.4 rotation / endpoint 是主要 trade-off

从 BL01 到 CM02/D5/E5，ATE 下降，但 rotation 和 final error 变差：

| Run | ATE (m) | Rot (deg) | Final error (m) |
|---|---:|---:|---:|
| BL01 | 41.3665 | 8.9490 | 3.590 |
| CM02 | 39.7820 | 9.7417 | 5.589 |
| D5_b125 | 39.4903 | 9.8299 | 6.311 |
| E5_resrel_b125 | 39.4881 | 9.7984 | 6.309 |
| E5_anchor_sparse_b100 | 39.5111 | 9.6396 | 5.967 |

Phase E 的 protection/sparse write 可以缓解一点 rotation/final，但代价是 ATE 回退。

### 9.5 短序列 gate 不可靠

Phase C v2/v3/v4 多次证明：

- 64-frame win 不代表 full win；
- 256-frame win 不代表 full win；
- stateful slice 6/7 improved 仍可能 full fail；
- cue quality 必须全序列审计，包括 mass、coverage、fragmentation、anchor collision。

### 9.6 当前瓶颈不是 beta，而是 cue source

D/E/F 多次 sweep 后，结果集中在 `39.3-39.5 m`。继续 old_dyn hand-rule modulation 收益有限。下一步需要：

- 真正 external optical-flow / epipolar residual；
- 更强 learned/training-free reliability gate；
- 或跨序列验证后重新设计 memory/update protocol。

---

## 10. 当前可用 cue 与实验结论

| Cue / Write | 最好 ATE / Rot | 状态 | 结论 |
|---|---:|---|---|
| `old_dyn_addclip` read | 39.3103 / 9.7097 | primary | 当前最好 |
| `old_dyn_addclip + residual_reliability` write | 39.3149 / 9.7586 | secondary | 追平但未超越 |
| `explicit_dyn_only` | 39.4191 / 9.5794 | strong diagnostic | 强，rotation 较好 |
| `old_dyn_soft_or` | 39.4147 / 9.8046 | strong | 不如 addclip |
| `old_dyn_calibrated_soft_or` | 39.4903 / 9.8299 | baseline | D5 reference |
| `residual_reliability` write | 39.4881 / 9.7984 | weak improvement | Phase E tiny best |
| `key_cosine_avg/shallow/deep` | 39.7820 / 9.7417 | diagnostic | 有信号，不够强 |
| `alignment_confidence` write | 39.7311 / 9.6980 | diagnostic | 不够 |
| `implicit_dyn_only` | 39.8656 / 9.6406 | weak | 不能独立替代 |
| `dyn4d_patch` hybrid | 39.9985 / 9.5368 | weak | 接近 40，但不够 |
| `gram4d` | 40.8677 / 9.2598 | rejected | rotation 好但 ATE 差 |
| `entropy` | 41.0887 / 9.9225 | rejected | 不可靠 |
| `query_cosine_*` | 41.7316 / 8.9914 | rejected | 接近 no-control |
| true RAFT/GMFlow flow | n/a | blocked | 尚未实现 |

---

## 11. 最终结论与下一步建议

### 11.1 最终结论

Pipeline v2 从 Phase A 到 Phase F 完成了一个完整的、可审计的 HMC 实验链路：

- 工程正确性：通过；
- identity hook correctness：通过；
- TTT-only write reproduction：通过；
- read-path signal：确认有效；
- commit isolation：确认是关键；
- read + safe branch0 write：确认互补；
- rotation-safe co-design：部分有效，但没有主指标突破；
- new cue search：old_dyn family 仍最强，其他 cue 未超过；
- true-flow：仍未实现，因此不能评价。

当前最强结果：

| Rank | Run | ATE RMSE (m) | Rot RMSE (deg) | 说明 |
|---:|---|---:|---:|---|
| 1 | F1_11 old_dyn_addclip b=1.25 | 39.3103 | 9.7097 | 当前最好 |
| 2 | F7_06 addclip + residual_reliability | 39.3149 | 9.7586 | 几乎追平 |
| 3 | F7_07 addclip + sparse95 residual | 39.3864 | 9.7821 | sparse 损失 ATE |
| 4 | F1_08 old_dyn_soft_or b=1.25 | 39.4147 | 9.8046 | 强 fusion |
| 5 | F1_01 explicit b=1.0 | 39.4191 | 9.5794 | explicit strong |

### 11.2 下一步建议

优先级最高：

1. 实现真正 RAFT/GMFlow optical-flow / epipolar residual cache，重启 F4。
2. 做 full-run cue quality audit，再允许 true-flow candidate 进入 controlled full run。
3. 不再继续 old_dyn hand-rule 小幅 sweep，除非有新的外部 cue 或 learned reliability gate。

可保留候选：

- `F1_11 old_dyn_addclip_b125` 作为当前最佳；
- `F7_06 addclip_resrel_b125` 作为 write-side residual_reliability 对照；
- `F1_01 explicit_dyn_only_b100` 作为 rotation 更好的强诊断 cue；
- `E5_anchor_sparse95` 作为 balanced diagnostic，不作为主 best。

停止建议：

- 暂停 Gram/QK/entropy/query-cosine 主线；
- 暂停 proxy flow 手工阈值 sweep；
- 暂停单纯 sparse write sweep；
- 暂停不带新 cue source 的 F8 大矩阵组合。

---

## 12. 主要参考文件

- `docs/exp.md`
- `docs/exp.log`
- `docs/pipeline2/Pipelinev2_Experiment_Plan_Typora.md`
- `docs/pipeline2/KITTI01_Pipelinev2_PhaseC_v5_CommitSafe_ReadPath_Experiment_Plan_Typora.md`
- `docs/pipeline2/KITTI01_Pipelinev2_PhaseE_ReadWrite_CoDesign_RotationSafe_Plan_Typora.md`
- `docs/pipeline2/KITTI01_Pipelinev2_PhaseF_NewCueSource_TrainingFreeReliability_Plan_Typora.md`

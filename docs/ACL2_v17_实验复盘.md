# ACL2 v17 实验复盘：TTT Write Causal-State Reboot Target25

日期：2026-05-19/20（Asia/Singapore，实验跨午夜落盘）  
计划文件：`docs/ACL2_v17_TTT_Write_CausalState_Reboot_Target25_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v17_ttt_write_causalstate_reboot_target25/`

本轮原则：只记录实际落盘结果；不把 sandbox oracle、proxy family、GT/offline diagnostic、partial run、failed run、或未通过 selector 的候选写成 deployable TTT success。

---

## 0. 工程与执行复盘

本轮新增 / 修改用于 v17 的部分：

```text
tools/run_v17_candidate_rollout.sh:
    v17 short-rollout launcher
    horizon 使用 plus-one local chunk 运行，report 再 crop 回目标 h5/h8/h10
    SAVE_CANDIDATE_MERGE_STATES 默认 0，避免每个 candidate 保存大 merge .pt
    added Phase1 / BASIS / AUXGEO / DLBANK candidate ids

tools/v17_horizon_report.py:
    h5/h8/h10 metrics aggregation
    eval-frame crop for plus-one rollout
    gate summary / heatmap csv / markdown report

tools/v17_effect_decay_report.py:
    h3-to-h5/h8/h10 effect decay diagnostic

tools/v17_basis_proxy_report.py:
    basis proxy audit artifacts

tools/v17_auxgeo_proxy_report.py:
    AUXGEO proxy audit artifacts
    unavailable runtime fields are marked unavailable / NaN, not fabricated

tools/v17_dual_bank_report.py:
    dual-bank proxy audit artifacts

run_pipeline_abc_v2.py:
    exported landed write-debug summaries for AUXGEO and dual-lifetime proxy audits
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    run_pipeline_abc_v2.py \
    tools/v17_horizon_report.py \
    tools/v17_effect_decay_report.py \
    tools/v17_basis_proxy_report.py \
    tools/v17_auxgeo_proxy_report.py \
    tools/v17_dual_bank_report.py \
    tools/v17_phase0_audit.py

bash -n tools/run_v17_candidate_rollout.sh tools/run_attention_cue_experiment.sh

PASS
```

重要 blocker 与修复：

```text
Phase1 R2 smoke:
    initially omitted the plus-one future overlap chunk.
    Result was invalid for trajectory-level comparison.

Fix:
    restored plus-one END_FRAME in launcher.
    v17_horizon_report crops metrics back to requested h5/h8/h10 window.

Phase3 R1 / R2 resource blocker:
    candidate merge snapshots filled disk because each candidate saved large merge .pt files.
    removed redundant v17 candidate merge_state_snapshots (~489GB reclaimed).
    changed launcher default SAVE_CANDIDATE_MERGE_STATES=0.

Phase3 R2 OOM blocker:
    naive 8-way scheduler reused busy GPUs and caused CUDA OOM.

Fix:
    stopped the bad scheduler.
    reran non-DONE AUXGEO rows with a GPU-locked repair scheduler.
    one GPU carried at most one rollout at a time.
    final AUXGEO matrix reached 36/36 DONE.
```

---

## 1. Phase 0 Boundary

输出：

```text
phase0_boundary/v17_phase0_gate_summary.csv
phase0_boundary/v17_phase0_boundary_audit.md
```

Gate：

```text
config_gate_pass = True
metric_gate_pass = True
snapshot_gate_pass = True
phase1_v16_parity_gate_pass = True
phase1_v16_parity_rows = 8
phase1_v16_parity_failures = 0
phase0_gate_pass = True
```

结论：

```text
Phase 0 gate = pass
v17 使用 v16 locked boundary artifacts，没有重新把 config-mismatch run 计入候选。
```

---

## 2. Phase 1 Horizon Expansion

输出：

```text
phase1_horizon_R3_evalcrop/
phase1_horizon_R3_evalcrop/effect_decay/
```

纠偏说明：

```text
R2 smoke without plus-one local chunk is invalid and not counted.
R3 eval-crop is the corrected result.
```

规模：

```text
84 diagnostic short-rollout rows
chunks = 5 / 6 / 10 / 16
horizons = h5 / h8 / h10
```

Gate summary：

| Best metric | Candidate | Chunk | Horizon | Delta vs H9 |
|---|---|---:|---:|---:|
| h8/h10 ATE | `K14_TTGR_ZERO_ORTHO_W0` | `10` | `10` | `-0.759139` |
| h8/h10 `[200,300)` | `K14_TTGR_ZERO_ORTHO_W0` | `10` | `10` | `-1.083022` |
| any-horizon `[200,300)` | `K21_ORTHO_SUPPRESS_ALL` | `10` | `5` | `-1.168375` |

Decision：

```text
Phase 1 gate = fail
selector_allowed = False
full_online_validation_allowed = False
```

Interpretation：

```text
v16 h3-local gains did not grow into durable h8/h10 drift correction.
Effect-decay report classifies these candidates as local regularizers.
```

---

## 3. Phase 2 Residual Basis Routing Proxy

输出：

```text
phase2_basis_proxy_R1/
phase2_basis_proxy_R1/audit/
```

边界：

```text
This was proxy-only BASIS_01-06.
It did not implement true historical post-zeropower tensor basis projection.
post_zp_delta_before_after.pt was not produced.
```

规模：

```text
36 diagnostic short-rollout rows
chunks = 6 / 10
horizons = h5 / h8 / h10
```

Best rows：

| Best metric | Candidate | Chunk | Horizon | ATE delta | `[200,300)` delta | `[400,600)` delta |
|---|---|---:|---:|---:|---:|---:|
| h8/h10 ATE | `BASIS_02_PROXY_HARM_W0_EMA090` | `6` | `8` | `-0.860542` | `-1.787323` | `+2.570160` |
| h8/h10 `[200,300)` | `BASIS_02_PROXY_HARM_W0_EMA090` | `10` | `10` | not best | `-1.858631` | below gate |
| h5 local `[200,300)` | `BASIS_02_PROXY_HARM_W0_EMA090` | `10` | `5` | `-1.362376` | `-4.857467` | `+0.196896` |

Decision：

```text
Phase 2 gate = fail
No selector/full validation allowed.
```

Interpretation：

```text
The h5 local body-window signal came close to -5m but did not satisfy h8/h10 gate.
The best h8 ATE row also had downstream proxy regression > +1m.
Basis proxy remains diagnostic and is not a deployable TTT write result.
```

---

## 4. Phase 3 AUXGEO Proxy Replay

输出：

```text
phase3_auxgeo_proxy_R2/
phase3_auxgeo_proxy_R2/audit/
```

无效 / 修复记录：

```text
R1 was invalid/partial due disk-full.
R2 initial scheduler caused OOM by reusing busy GPUs.
GPU-locked repair completed the intended matrix.
Final status = 36/36 DONE.
```

规模：

```text
36 diagnostic short-rollout rows
chunks = 6 / 10
horizons = h5 / h8 / h10
```

Gate summary：

| Best metric | Candidate | Chunk | Horizon | Delta vs H9 |
|---|---|---:|---:|---:|
| h8/h10 ATE | `AUXGEO_03_PROXY_STRUCT_KV_W0W2` | `6` | `10` | `-0.560961` |
| h8/h10 `[200,300)` | `AUXGEO_06_PROXY_STATIC_TOPK_KV_W0` | `6` | `10` | `-0.835052` |
| any-horizon `[200,300)` | `AUXGEO_06_PROXY_STATIC_TOPK_KV_W0` | `10` | `5` | `-1.084050` |

Decision：

```text
Phase 3 gate = fail
No selector/full validation allowed.
```

Interpretation：

```text
Overlap pseudo replay produced only weak local movement.
Runtime did not export stable semantic structure-token cosine/per-group target fields;
audit marks these fields unavailable rather than inventing values.
```

---

## 5. Phase 4 Dual-Bank Proxy

输出：

```text
phase4_dualbank_R1/
phase4_dualbank_R1/audit/
```

规模：

```text
18 diagnostic short-rollout rows
chunks = 6 / 10
horizons = h5 / h8 / h10
Final status = 18/18 DONE
```

Gate summary：

| Best metric | Candidate | Chunk | Horizon | Delta vs H9 |
|---|---|---:|---:|---:|
| h8/h10 ATE | `DLBANK_03_STRUCTURE_LONG_SHORT_REST` | `10` | `10` | `-0.121780` |
| h8/h10 `[200,300)` | `DLBANK_03_STRUCTURE_LONG_SHORT_REST` | `10` | `10` | `-0.350366` |
| any-horizon `[200,300)` | `DLBANK_03_STRUCTURE_LONG_SHORT_REST` | `10` | `5` | `-0.780976` |

Decision：

```text
Phase 4 gate = fail
No selector/full validation allowed.
```

Interpretation：

```text
The landed transient/dual-lifetime proxy had measurable but very small effect.
It did not demonstrate a useful W_short correction path.
Full W_long/W_short tensor cosine is unavailable unless runtime exports full short/long tensors.
```

---

## 6. Unified Candidate Bank And Selector Decision

输出：

```text
acl2_v17_candidate_bank_oracle_table.csv
acl2_v17_selector_precondition_report.md
acl2_v17_next_stage_decision.md
acl2_v17_full_validation_registry.csv
```

Unified rows：

| Phase | Rows |
|---|---:|
| Phase1 horizon expansion | `84` |
| Phase2 basis proxy | `36` |
| Phase3 AUXGEO proxy | `36` |
| Phase4 dual-bank proxy | `18` |
| Total | `174` |

Best observed across v17：

| Metric | Phase | Candidate | Chunk | Horizon | Delta |
|---|---|---|---:|---:|---:|
| h8/h10 ATE | Phase2 basis proxy | `BASIS_02_PROXY_HARM_W0_EMA090` | `6` | `8` | `-0.860542` |
| h8/h10 `[200,300)` | Phase2 basis proxy | `BASIS_02_PROXY_HARM_W0_EMA090` | `10` | `10` | `-1.858631` |
| any-horizon `[200,300)` | Phase2 basis proxy | `BASIS_02_PROXY_HARM_W0_EMA090` | `10` | `5` | `-4.857467` |

Selector/full precondition：

```text
Required:
    h8/h10 ATE delta <= -3m
    or h8/h10 [200,300) delta <= -5m without downstream regression

Actual:
    best h8/h10 ATE delta = -0.860542m
    best h8/h10 [200,300) delta = -1.858631m

selector_allowed = False
full_online_validation_allowed = False
```

Full validation registry：

```text
Phase6 full online validation = not_started
num_full_runs = 0
```

---

## 7. 当前最好配置与 Pipeline 解释

本节给没有项目背景的读者一个可审计解释：现在“最好”必须按用途分开说，不能把短窗口沙盒诊断结果说成线上成功。

### 7.1 当前最好的可计数线上配置

如果问题是：

```text
现在已经完整跑完 KITTI 01，并且 counts_as_ttt_write=true 的最好线上 TTT 写入配置是哪一个？
```

答案是：

```text
C9 locked repeat
代表 run: C9_P0_R2 / C9_P0_A / C9_P0_B
ATE = 33.7629421029m
Rot = 6.5259
RPE_t = 92.3871
FinalErr = 5.666384m
[200,300) = 76.102136m
[400,600) = 41.896364m
```

这里的 ATE 是 Absolute Trajectory Error，即“整条相机轨迹和真值轨迹对齐后的平均位置误差”，越低越好。Rot 是旋转误差相关指标。RPE_t 是 Relative Pose Error translation，即相邻片段的平移相对误差。

为什么是 C9：

```text
C9 的全序列 ATE 低于 H9:
    C9 ATE = 33.7629421029m
    H9 ATE = 34.1257769401m

C9 是完整在线 run，不是 sandbox / oracle / partial rollout。
C9 没有使用 GT runtime action。
C9 counts_as_ttt_write = true。
```

所以从“已经落盘、可计数、全序列线上 TTT 写入”的角度，C9 是当前最好配置。

### 7.2 为什么 v17 仍然以 H9 作为候选实验父配置

v17 的 candidate bank 没有直接以 C9 作为唯一父配置，而是继续大量使用 H9 causal-fork parent，这是因为 H9 在 body window 上更强，尤其是 v16/v17 重点观察的 `[200,300)` 区间：

```text
H9 [200,300) = 74.409927m
C9 [200,300) = 76.102136m
```

也就是说：

```text
C9 是当前全序列 ATE 最好的可计数线上配置。
H9 是更适合做 causal fork / candidate bank 的父配置之一，因为它在关键 body window 上更好。
```

这两个结论不冲突。C9 是“当前最好可交付线上结果”；H9 是“当前最好沙盒父轨迹之一”。

### 7.3 C9 配置到底做了什么

C9 的核心是一个在线 Test-Time Training 写入配置。Test-Time Training，简称 TTT，意思是在测试序列运行过程中，不改主模型 checkpoint，而是根据当前序列临时更新一小部分 fast weights。这里的 fast weights 可以理解为“只在当前测试过程中生效的短期适应记忆”。

C9 使用的主要模块：

```text
Hybrid Memory Controller:
    项目里的混合记忆控制器。
    它决定当前帧应该怎么读历史记忆、怎么写入临时 fast weights、怎么控制 attention cache。

LoGeR geometry pipeline:
    基础几何网络。
    它从图片序列预测每一帧的相机位姿、点云/深度相关表示、置信度和局部几何。

Semantic Prior Generator v2:
    语义先验生成器。
    它给不同区域分配写入优先级，例如结构区域更可信，动态物体区域更不可信。

Attention cue:
    从模型内部 attention 中构造出来的提示信号。
    它告诉系统哪些 token/区域可能是在错误读取旧记忆，或者正在被动态/不稳定区域干扰。

TTT write:
    在线临时写入 fast weights。
    它不是离线改轨迹，也不是用真值重写结果。
```

C9 的关键配置：

```text
hybrid_memory_mode = hybrid
hmc_commit_mode = probe_ttt_write
read_cue_source = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
hmc_write_score_source = stage_d_x_dg_inv_sqrt
read_layer_mode = all
beta_frame = 4.75
beta_swa = 4.75
read_beta_frame_chunks = 5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25
mp_alpha = 0.1
reset_every = 5
chunk_size = 32
chunk_overlap = 3
window_size = 32
overlap_size = 3
semantic_prior_mode = spg_v2
dyn_fusion_mode = calibrated_soft_or
ttt_write_gradient_reversal_mode = tri_replay
ttt_write_gradient_reversal_risk_source = update_conflict_energy
ttt_write_gradient_reversal_branch_mask = 0
```

其中 `mp_alpha` 是 C9 和 H9 的关键差异：

```text
C9 mp_alpha = 0.1
H9 mp_alpha = 0.125
```

可以粗略理解为：`mp_alpha` 控制记忆写入/记忆先验参与最终 fast-weight 更新的强度。C9 使用更小的 `0.1`，在当前完整 KITTI 01 序列上全局 ATE 更好。

### 7.4 C9 用了哪个 cue，以及 cue 怎么构造

C9 的读记忆 cue 是：

```text
acl2.gg.qq.low.g2_3.past_only.headmean.robustq
```

逐段解释：

```text
acl2:
    本项目 ACL2 系列实验的 cue family 名称。

gg.qq:
    使用 query-query attention 关系。
    在 attention 里，query 可以理解为“当前 token 想找什么信息”。
    query-query 相似度就是比较不同 token 的查询模式是否一致。

low:
    使用低相似 / 低支持区域。
    这些区域常常对应不稳定、动态、遮挡、或者容易错误读旧记忆的位置。

g2_3:
    使用中间层 group 2 到 group 3 的 attention 统计。
    它不是只看最浅层纹理，也不是只看最高层输出。

past_only:
    只和过去帧比较，不看未来帧。
    这保证 cue 是在线因果的，不偷看未来。

headmean:
    对多个 attention head 求平均。
    attention head 可以理解为模型内部不同观察角度。

robustq:
    使用 robust quantile 归一化。
    它用更稳健的分位数尺度，而不是容易被极端值带偏的普通最大最小归一化。
```

因此，这个 cue 的直觉是：

```text
如果当前区域和过去记忆的 attention 查询模式不稳定、不一致、低支持，
就降低它从旧记忆里强读的机会，或降低它作为安全写入样本的权重。
```

C9 的写入 cue 是：

```text
hmc_write_score_source = stage_d_x_dg_inv_sqrt
```

按代码命名和实验配置，它表示把 `stage_d` 和 `D_g` 组合成写入分数：

```text
stage_d:
    语义/几何先验给出的“这个 token 适不适合写入记忆”的分数。
    结构背景通常更可靠，动态物体通常更危险。

D_g:
    由动态/几何/attention cue 融合出的风险或不稳定性提示。

x_dg_inv_sqrt:
    用 inverse-square-root 形式调节 D_g。
    直觉是：风险越高，写入越谨慎；风险越低，越允许作为稳定样本参与写入。
```

### 7.5 整个 pipeline 从图片到结果的流程

完整流程可以按 9 步理解。

1. 读取 KITTI 01 图片序列  
   系统读取连续驾驶视频帧。目标是估计每一帧相机在三维空间中的位置和朝向。

2. 切成重叠窗口  
   C9 使用：

```text
window_size / chunk_size = 32 frames
overlap_size / chunk_overlap = 3 frames
```

意思是每次处理 32 帧，相邻窗口共享 3 帧。共享帧用于后面把局部轨迹平滑接起来。

3. 基础 LoGeR 几何预测  
   LoGeR 模型对每个窗口预测：

```text
相机位姿
局部点云 / 深度相关表示
置信度
attention 内部统计
```

4. 构造 attention cue  
   使用 `acl2.gg.qq.low.g2_3.past_only.headmean.robustq` 从模型内部 attention 得到在线因果 cue。这个 cue 不用真值，也不看未来帧。

5. 构造语义/几何写入先验  
   `spg_v2` 根据 cue、几何和语义规则给 token 分级。C9 中的默认倾向是：

```text
结构区域权重高
背景区域次高
不确定区域较低
可移动物体区域最低
```

这样做的目的很简单：不要让车、人、遮挡、反光等不稳定区域过度污染记忆。

6. 控制读取历史记忆  
   Hybrid Memory Controller 用 `read_cue_source` 控制当前窗口怎么读历史记忆。

```text
beta_frame = 4.75
beta_swa = 4.75
```

`beta` 可以理解为 cue 的放大强度。部分 chunk 有单独 override：

```text
chunks 5-9:  beta = 4.85
chunks 10-12: beta = 4.25
chunk 16:     beta = 4.25
```

这些 chunk 是根据先前实验发现的漂移敏感区间设置的。

7. 控制 attention cache / K-V 记忆  
   K/V 是 key/value 的缩写，来自 Transformer attention。简单说：

```text
query: 当前 token 想找什么
key: 历史 token 能被什么查询匹配到
value: 匹配成功后实际读出的信息
```

项目里的 SWA 在这里指 attention history/cache 相关路径，可以理解为一段保存历史 key/value 的短期记忆。C9 使用 overlap source replacement：

```text
enable_swa_overlap_source_replace = 1
swa_overlap_source_replace_alpha = 0.5
swa_overlap_source_replace_target = kv
swa_overlap_source_replace_layer_mode = last
```

意思是：在重叠帧附近，用受控比例替换最后层的 key/value 历史来源，让窗口衔接更稳定。

8. 执行 TTT fast-weight 写入  
   C9 的 commit mode 是：

```text
probe_ttt_write
```

直觉是先用 probe/native pass 看看当前窗口的几何和风险，再决定怎么写入 fast weights。

写入时使用：

```text
ttt_write_gradient_reversal_mode = tri_replay
ttt_write_gradient_reversal_risk_source = update_conflict_energy
```

`update_conflict_energy` 是模型更新自身产生的冲突风险信号。`tri_replay` 把 token 分成三类：

```text
positive / continuity tokens:
    低风险、连续、适合强化的 token。

negative / harmful tokens:
    高风险、冲突强、可能污染记忆的 token。

neutral tokens:
    中间区域，不完全丢弃，但降低影响。
```

C9 的 tri-replay 默认比例是：

```text
positive fraction = 0.35
negative fraction = 0.12
neutral lambda = 0.85
```

并且只在 branch mask `0` 上施加很小的 chunk-specific gamma：

```text
chunks 5-9:   gamma = 0.005
chunks 10-12: gamma = 0.003
chunk 16:     gamma = 0.0003
```

这说明 C9 不是大幅重写模型，而是非常小心地做在线 fast-weight 调整。

9. 合并窗口轨迹并评分  
   每个窗口都有局部相机轨迹。merge/gauge 逻辑用重叠帧把窗口拼成完整 `01.txt` 轨迹，然后 KITTI benchmark 计算 ATE、RPE、Rot 等指标。

### 7.6 v16/v17 为什么还要做 causal fork

v15 发现一个问题：只保存 HMC fast-weight state 虽然能复现 fast-weight hash，但不能复现完整轨迹后缀。缺的是 merge/gauge 轨迹状态。

v16 修复后，causal fork 会同时保存/恢复：

```text
HMC fast-weight state
merge/gauge state
global chunk id
```

这样短窗口 sandbox rollout 才能和完整 full run 的轨迹后缀精确对齐。v16 Phase 1 已经证明：

```text
required H9/C9 windows raw pose diff = 0.0
ATE suffix diff = 0.0
HMC hash mismatch = 0
```

所以 v17 的 candidate bank 是可信的短窗口因果诊断平台。

### 7.7 v17 里最强但不可计数的局部信号

v17 最强 diagnostic row 是：

```text
BASIS_02_PROXY_HARM_W0_EMA090
chunk = 10
horizon = 5
[200,300) delta vs H9 = -4.857467m
ATE delta vs H9 = -1.362376m
```

这是好消息，因为它说明“残差基路由 / harmful-token suppression”方向确实能在短窗口局部压低 body-window 误差。

但它不能算 deployable success，原因是：

```text
它是 proxy-only sandbox diagnostic。
它没有通过 h8/h10 持久性 gate。
它没有进入 no-GT selector。
它没有启动 full online validation。
它不是完整 KITTI 01 online result。
```

v17 的最好持久性结果仍然远低于进入 full validation 的门槛：

```text
best h8/h10 ATE delta = -0.860542m
best h8/h10 [200,300) delta = -1.858631m

required:
    h8/h10 ATE delta <= -3m
    or h8/h10 [200,300) delta <= -5m without downstream regression
```

因此，v17 没有产生新的可计数 Target-25 线上配置。

### 7.8 给审计用的一句话结论

```text
当前最好可计数线上 TTT 写入配置是 C9 locked repeat，ATE = 33.7629421029m。
v17 最强新信号是 BASIS_02_PROXY_HARM_W0_EMA090 的短窗口局部 body-window 改善，
但它没有通过持久性 gate，不能算 deployable TTT success。
截至本轮复盘，没有任何新配置达到 Target-25。
```

---

## 8. Final Decision

v17 的好消息：

```text
v16 causal fork boundary remained trusted.
h5/h8/h10 evaluation path is now corrected with plus-one merge context.
Resource blockers were fixed without counting failed/partial rows.
Unified v17 candidate-bank oracle has 174 landed diagnostic rows.
```

v17 的目标结果：

```text
Target-25 was not reached.
No deployable online TTT-write candidate with ATE <= 25 was produced.
No no-GT selector was authorized.
No full online validation was launched.
```

根据 v17 stop rule：

```text
Residual basis proxy failed.
Overlap-geometry auxiliary replay proxy failed.
Dual-bank proxy failed.
No candidate reached h8/h10 ATE delta <= -2m or [200,300) delta <= -5m.
```

结论：

```text
TTT write-only is not the Target-25 mainline under the current action interface.
TTT remains useful as a regularizer / stabilizer.
Target-25 should move to explicit online trajectory-state / scale-state modules,
while keeping TTT write for stability.
```

Boundary：

```text
No v17 Phase 1/2/3/4 row counts as deployable TTT write success.
No GT-selected candidate is counted as online success.
No offline trajectory rewrite is counted.
No full online target-25 validation was launched.
No online target-25 result was produced in v17.
```

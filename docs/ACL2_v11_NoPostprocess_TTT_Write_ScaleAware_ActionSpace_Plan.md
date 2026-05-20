# ACL2 v11：禁止后处理的 TTT 写入重定位与 Target-25 并行实验计划

日期：2026-05-18  
对象：LoGeR / HMC Pipeline v2 / KITTI01  
新目标：**KITTI01 ATE RMSE <= 25.0m**  
当前在线 LoGeR/HMC 主参考：`H9_READBETA2_03 = 34.1258 / 6.5414`  
当前离线 no-GT 后处理参考：`NOGTPOSE_27 = 22.4012 / 3.5515`  
本计划的硬约束：**后处理轨迹修正不能计入 TTT 写入策略成功。**

---

## 0. 本计划的核心判断

这轮 v10 结果非常有价值，但它没有完成我们真正想要的事情。

`NOGTPOSE_27_reset_global_clip35_body600_t105` 达到 `22.4012m`，已经过 `target-25`。但是它是 **offline no-GT trajectory-state proxy**，不是 online LoGeR full run，也不是 TTT write result。它使用预测轨迹自身的 reset-window step-length 统计做 scale correction，GT 只参与最终评价。这个结果只能说明 **target-25 在轨迹状态 / window scale 这个方向上有足够上界**，不能说明 TTT 写入策略已经成功。

因此本计划把 v10 的结果重新定义为：

```text
TTT write-side scalar policy:
    没有成功，已经平台化。

Offline no-GT pose proxy:
    证明 target-25 的主误差不是普通 token-level dynamic contamination，
    而是 reset-window / window-local scale / trajectory-state 错误。

下一阶段:
    禁止把离线 trajectory postprocess 当作 TTT 成果；
    必须把 pose-scale / drift-state 信号接回 TTT write action，
    或者证明 TTT write-side 无法承担 target-25 主线。
```

---

## 1. 实验整体目标

本阶段不是继续把 `34.1258m` 微调到 `34.10m`。本阶段要回答一个更根本的问题：

> **TTT fast-weight write 是否能在线地产生与 no-GT pose-scale proxy 类似的 window-level scale / drift correction？**

如果能，本阶段目标是把 TTT 写入从 scalar token gate 升级为 **scale-aware / window-drift-aware memory update**。

如果不能，本阶段必须明确给出停止结论：TTT write-side 只能作为 auxiliary regularizer，target-25 主线必须转向 online trajectory-state / pose-scale 模块，或者 read/SWA/pose-alignment action。

本阶段总目标拆成五个问题：

1. **严格边界问题**：哪些结果可以算 TTT write，哪些只能算 postprocess diagnostic？
2. **上界问题**：如果把 scale / window drift 方向接入 TTT write，是否有足够上界接近 target-25？
3. **代理问题**：不使用 GT 时，哪些 no-GT drift / scale proxy 能预测错误窗口？
4. **动作问题**：TTT write action 如何从 `gamma / neutral / role fraction` 升级为 direction-aware residual composition？
5. **工程加速问题**：如何让 Codex 并行推进，不满足 gate 时自动切换方向，而不是等待人工判断？

---

## 2. 明确禁止：不能把后处理当成 TTT 写入

### 2.1 结果分类

后续所有结果必须先分类：

| 类型 | 允许计入 TTT write success? | 说明 |
|---|---:|---|
| Online HMC full run，TTT commit 改变未来 state，输出轨迹由 LoGeR forward 自然产生 | yes | 真正 TTT 写入策略 |
| Online HMC full run，read path / TTT write / SWA 固定，最后改写输出轨迹 | no | 后处理，不算 TTT |
| Offline no-GT trajectory proxy，例如 reset-window scale correction | no | 可作为 diagnostic / upper bound，不算 TTT |
| 使用 GT 拟合 window transform | no | oracle upper bound，只用于判断上界 |
| 使用 GT 定义 TTT projection direction 并在线 full run | oracle-only | 可以判断 TTT action 上界，但不能算部署结果 |
| 使用 no-GT proxy 选择 TTT commit / replay action | yes | 如果输出轨迹来自 online HMC full run，可计入 |

### 2.2 硬 gate

任何 run 想被记录为 “TTT write candidate”，必须满足：

```text
1. output trajectory comes from run_pipeline_abc_v2.py online full run;
2. no external trajectory file rewrite after model forward;
3. no per-window Sim3 / scale / yaw correction applied after output;
4. no GT used to compute runtime action;
5. TTT state hash / commit debug must show write policy actually affected W_{m+1};
6. hmc_commit_mode must remain probe_ttt_write or explicitly documented safe variant.
```

如果不满足，统一归入：

```text
diagnostic / postprocess / oracle
```

而不是 TTT write success。

---

## 3. 当前数据的独立分析

### 3.1 scalar action space 已经饱和

v10 Batch 0 扫描了 `88` 个 landed runs。当前在线 best 是：

```text
H9_READBETA2_03:
ATE = 34.1258
Rot = 6.5414
FinalErr = 6.189
[200,300) = 74.410
[400,600) = 44.354
```

多个 action family 已经平台：

```text
read_beta_scalar:
    best = 34.1258
    top-5 span = 0.0058

chunk16_scalar:
    best = 34.1583
    已平台

commit_or_delta_gate:
    best = 34.1672
    未过 best

handoff_scalar:
    best = 34.1722
    未过 best

window_scalar_oracle:
    best = 34.2352
    未打开空间

read_support:
    best = 37.0339
    明显失败
```

结论：

> **继续扫 `gamma / neutral / read beta / chunk16 scalar / handoff scalar` 不可能带来从 `34.13m` 到 `25m` 的 `9.13m` 跨越。**

这些 action 主要在局部窗口、姿态、endpoint 或 segment trade-off 里重新分配误差，不是真正消除 whole-scene drift。

### 3.2 真正的 TTT projection oracle 没有接通

v10 Batch 1 明确显示，计划中的 true projection oracle 所需关键信息缺失：

```text
per_token_update_group.pt: missing
per_layer_branch_update_matrix.pt: missing
tri_replay_role_mass.jsonl: missing
window_pose_residual_gt.json: missing
window_drift_direction_gt.pt: missing
update_to_drift_projection.csv: missing
update_cos_to_window_drift debug: missing
projection_helpful_energy debug: missing
source-level window_drift_direction hook: missing
source-level ttt_update_conflict_energy: present
```

这意味着：

> **TTT write projection action 并没有被真正测试。**

当前的 `update_conflict_energy` 只是 fast-weight aggregate conflict，不等于 **token update 到 window drift direction 的 projection**。所以我们不能说 “TTT projection 失败”，只能说 “TTT projection 没接通”。

### 3.3 pose-scale oracle 说明主误差是 window-local scale / trajectory-state

v10 Batch 5-A 显示：

```text
baseline_global_sim3:
ATE = 34.1258

POSEORACLE_01_per_reset_sim3:
ATE = 4.7974

POSEORACLE_03_per_reset_scale_only:
ATE = 5.5678
```

`scale_only` 已经能到 `5.5678m`，远低于 target-25。这说明当前 `34m` 平台不是因为几何完全错了，而是因为 **reset-window / window-local scale / trajectory-state 错误巨大**。

这非常关键：

```text
如果只在 TTT token 写入强度上微调，
但不直接感知或控制 window scale state，
就很难接近 target-25。
```

### 3.4 NOGTPOSE_27 是强诊断，不是 TTT 成果

`NOGTPOSE_27` 结果：

```text
ATE = 22.4012
Yaw = 3.5515
FinalErr = 8.753
[200,300) = 35.187
[200,400) = 26.983
[400,600) = 8.587
```

相比 `H9_READBETA2_03`：

```text
ATE:        34.1258 -> 22.4012
[200,300): 74.410  -> 35.187
[200,400): 54.651  -> 26.983
[400,600): 44.354  -> 8.587
```

这说明 no-GT scale proxy 有非常强的轨迹修复能力。但代价也很明显：

```text
FinalErr: 6.189 -> 8.753
worst chunk 转移到 [0,100)
```

因此它证明的是：

```text
Target-25 有可达上界；
主误差集中在 window trajectory-state / scale；
但当前实现是离线轨迹后处理，不满足 TTT write 要求。
```

---

## 4. 问题本质

### 4.1 不是缺一个更好的 dynamic mask

早期假设是动态区域污染 TTT memory，所以要少写、反向写、或者让低价值语义少写。实验已经反复说明这不够：

```text
D_g / explicit_dyn / semantic lowstuff / sparse write:
    多数只能改善 Rot / FinalErr / Yaw，
    不能大幅降低 ATE。

TTGR / tri-replay:
    能从 36m 推到 34m，
    但卡在 34m 平台。

NOGTPOSE:
    直接校正 window scale 后可以到 22m。
```

这说明主误差不是纯 token dynamic contamination，而是：

> **TTT / HMC 在 reset-window 层面形成了错误的 trajectory-scale state。**

### 4.2 不是缺少 TTT cue，而是 cue 没有接到正确 action

`update_conflict_energy` 是有用的 TTT-native cue。它帮助把结果从 `36m` 推到 `34m`。但当前 action 是：

```text
pos_frac
neg_frac
neutral_lambda
gamma
chunk window
read beta
```

这些都是低维 scalar action。它们没有直接回答：

```text
这个 TTT update residual 是否会加剧 window scale drift？
这个 TTT update residual 是否会纠正 body accumulated drift？
这个 TTT update residual 是否维持 downstream continuity？
```

因此现在卡住的是：

```text
cue/action mismatch
```

而不是单纯：

```text
cue 不够好
```

### 4.3 LoGeR TTT 的理论角色和当前问题正好对齐

LoGeR 论文中，TTT 被设计为 long-term compressed global memory，用来保存 coarse geometry 和 scale，SWA 则作为 adjacent chunk 的 lossless local highway。也就是说，**scale drift 本应是 TTT 该负责的方向**。如果 TTT write-side 无法修 scale，问题不是 TTT 不相关，而是当前 write objective 没把 scale-state 作为优化对象。

### 4.4 当前 TTT write 没有显式 scale objective

当前 TTT write 主要通过这些方式间接影响 memory：

```text
token prior
branch mask
layer mask
positive / neutral / negative replay
update_conflict_energy
read beta coupling
```

但没有直接优化：

```text
reset-window step scale
overlap pointmap scale consistency
pose increment scale consistency
future window scale stability
```

因此它只能局部调 representation，不能系统性修全场景 scale drift。

---

## 5. 本阶段核心假设

后续实验必须围绕下面的假设设计。

---

## H1：TTT write-side 只有在接入 scale / drift projection action 后，才可能接近 target-25

### 假设内容

当前 scalar tri-replay action 不足以修 whole-scene drift。必须把 TTT update residual 分解到 window scale / drift direction 上：

$$
J_{i,l,r}
=
J^{corr}_{i,l,r}
+
J^{cont}_{i,l,r}
+
J^{harm}_{i,l,r}
$$

其中：

- $J^{corr}$：有利于纠正 window drift / scale 的方向；
- $J^{cont}$：维持 downstream continuity 的方向；
- $J^{harm}$：加剧 window drift 的方向。

新的 TTT commit 应该是：

$$
G_{commit}
=
G_{corr}
+
\lambda_{cont}G_{cont}
-
\gamma_{harm}G_{harm}
$$

而不是：

$$
G_{commit}
=
G_{pos}
+
\lambda G_{neu}
-
\gamma G_{neg}
$$

后者只按 risk 分组，不按 drift direction 分组。

### 验证思路

先做 oracle projection，但要求 action 作用在 TTT write 上，而不是后处理轨迹。

通过标准：

```text
TTT projection oracle full run:
    ATE <= 31.0
    或 [200,300) <= 55 且 [400,600) 不恶化
```

如果 oracle projection 都达不到，说明 TTT write-side action 上界不足，应停止把 TTT write 当 target-25 主线。

---

## H2：no-GT scale proxy 可以作为 TTT commit 选择器，而不是作为后处理

### 假设内容

`NOGTPOSE_27` 证明 reset-window step-length proxy 很强，但它目前是离线轨迹后处理。我们要把它改成在线 TTT commit 选择器：

```text
给定当前 chunk m:
    生成多个候选 TTT commit W_{m+1}^{(k)}
    用 no-GT scale / drift proxy 估计哪个 commit 更可能稳定下一窗口
    只提交被选中的 W_{m+1}^{(k)}
```

注意：这里输出轨迹仍来自 LoGeR forward，不做后处理。proxy 只用来选择 memory state。

### 关键区别

错误做法：

```text
run LoGeR -> 得到 trajectory -> offline scale correction -> 评价
```

正确做法：

```text
run LoGeR probe/control -> 生成 candidate TTT commits
-> no-GT proxy 选择 W_{m+1}
-> 下一个 chunk 用被选中的 TTT memory forward
-> 输出 trajectory 不再修改
```

### 通过标准

```text
KITTI01:
    ATE <= 32.0 作为第一阶段可接受；
    ATE <= 30.0 作为强通过；
    ATE <= 25.0 作为目标通过。

同时:
    [200,300) 明显下降；
    [400,600) 不比 H9 best 恶化超过 2m；
    FinalErr 不比 H9 best 恶化超过 2m。
```

---

## H3：window scale error 不是单一 body 问题，而是 body / exit / handoff 的状态机问题

### 假设内容

v8/v9 已经显示：

```text
body window:
    决定 [200,300) 病灶

exit / c16 / handoff:
    决定 [400,600) 和 downstream continuity
```

所以策略不能只问 “chunk 5-9 gamma 多大”，而要建一个 window state：

$$
z_m =
[
r^{scale}_m,
r^{yaw}_m,
r^{ttt}_m,
r^{overlap}_m,
r^{read}_m
]
$$

并根据状态决定写入角色：

```text
body:
    stronger correction
exit:
    weaker correction / continuity preserve
handoff:
    scale state transfer / no over-correction
```

### 通过标准

自动状态机应至少复现人工窗口的角色分配：

```text
body window recall >= 0.8 on chunks 5-9
exit window recall >= 0.67 on chunks 10-12
false positive reset-periodic chunks <= 2
```

更重要的是 full-run：

```text
ATE <= C16ROLE_01 或 H9_READBETA2_03
且不是只改善 Rot/FinalErr。
```

---

## H4：如果 scale oracle 很强但 TTT projection oracle 弱，说明需要显式 pose-scale module

### 假设内容

如果：

```text
pose-scale oracle ATE << 25
但 TTT projection oracle ATE > 32
```

那么 TTT write-side 不是 target-25 主 action。此时正确方向不是继续硬凑 TTT，而是设计一个 online pose-scale / reset-window alignment module。

这不是放弃 TTT，而是重新分工：

```text
TTT:
    保持 global memory / read regularizer

pose-scale module:
    显式修 window trajectory scale / drift-state

SWA/read:
    保持 local alignment / current output correction
```

---

## 6. 实验分阶段设计

---

# Phase 0：复现与边界确认

## 目标

锁定所有 reference，并确保 v10 的 postprocess 结果不会被误记为 TTT 写入成果。

## 实验

### P0-1：reference repeat

运行：

```text
H9_READBETA2_03 repeat
C16ROLE_01 repeat
WINGAM_03 repeat
```

记录：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
[200,300)
[200,400)
[400,600)
YawRMSE
Sim3Scale
hmc_state_hash
runtime
```

### P0-2：NOGTPOSE reproduction as diagnostic only

重跑：

```text
NOGTPOSE_27
NOGTPOSE_21
NOGTPOSE_13
```

但输出目录必须标注：

```text
diagnostic_only = true
counts_as_ttt_write = false
```

## 成立标准

```text
H9 repeat ATE within 0.03m
NOGTPOSE_27 repeat ATE within 0.05m
diagnostic flags present in result_registry.csv
```

不满足时：

```text
Codex action:
    1. 检查 trajectory input file 是否一致；
    2. 检查 Sim3 evaluator path；
    3. 检查 NOGTPOSE 参数是否写入 registry；
    4. 不允许进入 Phase 1。
```

---

# Phase 1：补齐 TTT projection oracle instrumentation

## 目标

先把 v10 Batch 1 缺失的 projection path 真正接通。不直接跑 full 矩阵。

## 必须新增落盘

对每个 chunk、layer、branch、token 组，落盘：

```text
per_token_update_group.pt
per_layer_branch_update_matrix.pt
tri_replay_role_mass.jsonl
window_pose_residual_gt.json
window_drift_direction_gt.pt
update_to_drift_projection.csv
update_cos_to_window_drift.jsonl
projection_helpful_energy.jsonl
ttt_update_conflict_energy.jsonl
```

## 核心定义

### 1. Window drift direction

oracle 版本：

$$
d^{gt}_w =
\frac{
\operatorname{vec}(T^{gt}_{w} - T^{pred}_{w})
}{
\|\operatorname{vec}(T^{gt}_{w} - T^{pred}_{w})\| + \epsilon
}
$$

不要求公式完全按 SE(3) 实现，工程上可先使用：

```text
translation residual direction
scale residual scalar
yaw residual scalar
```

分别落盘。

### 2. TTT update projection

对 token update contribution $J_{i,l,r}$，定义：

$$
p^{drift}_{i,l,r}
=
\frac{
\langle \operatorname{vec}(J_{i,l,r}), d^{ttt}_{w,l,r} \rangle
}{
\|\operatorname{vec}(J_{i,l,r})\| \|d^{ttt}_{w,l,r}\| + \epsilon
}
$$

其中 $d^{ttt}_{w,l,r}$ 是通过 finite difference / aggregate delta 映射得到的 TTT-space drift direction。

第一版如果无法得到精确 $d^{ttt}$，可先用 candidate delta 近似：

$$
d^{ttt}_{w,l,r}
=
\operatorname{vec}(\Delta W^{oracle\_good}_{l,r} - \Delta W^{baseline}_{l,r})
$$

## Smoke 流程

只跑 `END_FRAME=160`：

```text
P1_SMOKE_projection_trace
```

检查：

```text
all required files exist
nonzero projection energy
nonzero role mass
chunk5 / chunks5-12 have valid entries
no NaN / inf
```

## 成立标准

```text
1. required files coverage = 100%
2. projection energy nonzero for w0 at chunks 5-12
3. update_conflict_energy 与 projection_energy 不是完全相同字段
4. role_mass sums to approximately 1 per selected chunk
5. smoke full trajectory unchanged when only logging is enabled
```

不满足时：

```text
Codex action:
    A. 如果缺 per_token_update_group:
        从 TTTWriteController replay cache 中保存 token index / frame / patch / role group。
    B. 如果缺 per_layer_branch_update_matrix:
        在 pre-zeropower 和 post-zeropower 两处都保存 ΔW。
    C. 如果 window_drift_direction_gt 缺:
        从 prediction + GT trajectory 脚本先生成 offline json，再由 HMC 只读入 diagnostic。
    D. 如果 projection energy 全零:
        检查 flatten shape / branch mapping / device cast。
    E. 如果 logging 改变 output:
        关闭任何 write action，只保留 detached CPU logging。
```

Phase 1 不通过，不允许声称 projection oracle 失败。

---

# Phase 2：真正的 TTT projection oracle full run

## 目标

回答：**如果 GT 只用来定义 projection direction，TTT write action 是否有足够上界？**

这不是部署策略，但它是 TTT write-side 的上界测试。

## 固定协议

```text
seq = KITTI01 full
base = H9_READBETA2_03 protocol
read cue = C23 past
read path = frame pair/all
commit = probe_ttt_write
SWA = fixed SWKS3 background, no new SWA variables
RESET_EVERY = 5
no trajectory postprocess
```

## 实验候选

### ORACLE_TTT_01：body projection-routed tri replay

```text
chunks = 5-12
branch = w0
layer = all
role = projection routed
gamma_harm = 0.005
lambda_cont = 0.85
```

角色定义：

```text
if projection aggravates drift:
    negative
elif projection corrects drift:
    positive
else:
    neutral
```

### ORACLE_TTT_02：body + handoff projection

```text
chunks = 5-12, 16
chunk16 gamma_harm = 0.0003
```

### ORACLE_TTT_03：scale-only projection

只使用 scale residual direction，不使用 translation/yaw：

```text
projection target = scale
chunks = reset groups covering 5-16
```

### ORACLE_TTT_04：layer/branch restricted projection

根据 Phase 1 heatmap，只在 projection energy top layers / top branches 作用：

```text
top_layers = top 25% projection helpful energy
branch = w0 unless w1/w2 helpful energy dominates
```

## 必须记录

```text
metrics_global.json
trajectory_diagnostics.json
per_segment_error.csv
per_reset_group_error.csv
update_to_drift_projection.csv
projection_role_mass.jsonl
projection_energy_by_layer_branch.csv
ttt_state_hash.jsonl
no_postprocess_flag.json
```

## 成立标准

强通过：

```text
ATE <= 30.0
且 [200,300) <= 55
且 [400,600) <= 44.354
```

弱通过：

```text
ATE <= 32.0
或 [200,300) <= 60 且 overall ATE <= 34.0
```

失败：

```text
ATE > 33.5
或只改善 Rot / FinalErr 而 ATE 无明显下降
或 [400,600) 明显恶化
```

不满足时：

```text
Codex action:
    1. 如果 projection oracle ATE > 33.5:
        停止 TTT projection 主线，转 Phase 6 pose-scale module。
    2. 如果 [200,300) 降但 [400,600) 崩:
        进入 Phase 4 dual-lifetime，不做更多 gamma sweep。
    3. 如果 only Rot improves:
        标记为 regularizer，不进入 target-25 主线。
    4. 如果 oracle action no-op:
        回 Phase 1 修 hook，不算失败。
```

---

# Phase 3：no-GT TTT commit candidate selection

## 目标

把 no-GT pose-scale proxy 从后处理改成 TTT commit 选择器。

## 思路

每个 chunk 不只生成一个 TTT commit，而是生成少量候选：

$$
\mathcal{W}_{m+1}
=
\{W^{base}_{m+1}, W^{body}_{m+1}, W^{exit}_{m+1}, W^{scale}_{m+1}\}
$$

用 no-GT proxy 评分：

$$
Score_k
=
\alpha |r^{step}_k|
+
\beta |r^{overlap}_k|
+
\gamma E^{apply}_k
+
\delta J^{pose}_k
$$

选择：

$$
W_{m+1}^{commit}
=
\arg\min_{k} Score_k
$$

注意：输出轨迹不后处理。proxy 只决定 future memory。

## no-GT proxy 定义

### 1. step-length consistency

$$
r^{step}_m =
\log
\frac{
\operatorname{median}_{t \in W_m}\|\hat p_t - \hat p_{t-1}\|
}{
\operatorname{EMA}_{j<m}\operatorname{median}_{t \in W_j}\|\hat p_t - \hat p_{t-1}\| + \epsilon
}
$$

### 2. overlap pointmap scale proxy

$$
r^{ov}_m =
\log
\operatorname{median}_{p \in \Omega_{ov}}
\frac{
\|x^{prev}_{p}\|
}{
\|x^{cur}_{p}\|+\epsilon
}
$$

### 3. TTT apply mismatch

$$
E^{apply}_{m}
=
\operatorname{mean}_{i}
\frac{
\|f_{W_m}(q_i)-v_i\|
}{
\|v_i\|+\epsilon
}
$$

### 4. pose increment jump

$$
J^{pose}_m =
\left\|
\log(\hat T_{m,start}^{-1}\hat T_{m,end})
-
\operatorname{EMA}_{j<m}\log(\hat T_{j,start}^{-1}\hat T_{j,end})
\right\|
$$

## Candidate commits

第一批只做 4 个 candidate：

```text
K0 = current H9/C16ROLE style baseline
K1 = WINGAM_03 style body/exit tri-replay
K2 = scale-aware weak negative
K3 = conservative continuity preserve
```

## Full runs

### SELECT_TTT_01：chunk-local selection

只在 chunks 5-16 启用 candidate selection。

### SELECT_TTT_02：reset-group selection

每个 reset group 只在 group start / middle / exit 选择一次。

### SELECT_TTT_03：selection + projection-risk

候选 score 加入 Phase 1 的 no-GT update_conflict / projection proxy。

### SELECT_TTT_04：selection with confidence fallback

如果 proxy disagreement 大，则 fallback 到 baseline commit。

## 成立标准

强通过：

```text
ATE <= 30.0
```

弱通过：

```text
ATE <= 32.5
且 [200,300) <= 65
且 [400,600) 不恶化
```

diagnostic 通过：

```text
selector 在 chunks 5-12 的选择与 oracle beneficial candidates 一致率 >= 0.7
```

不满足时：

```text
Codex action:
    1. 如果 selection ATE 与 baseline 几乎相同:
        检查候选 W 是否真的不同，保存 candidate_state_diff。
    2. 如果 selection 过度选 aggressive 导致后段崩:
        加入 downstream penalty 和 confidence fallback。
    3. 如果 proxy 无法区分 candidates:
        回 Phase 1 增加 projection / overlap scale feature。
    4. 如果 selection 有明显 segment 改善但 ATE 不过:
        进入 Phase 4 dual-lifetime。
```

---

# Phase 4：dual-lifetime TTT memory，不再单一长期 commit

## 目标

解决 freeze / TTGR 的核心矛盾：

```text
某些 update 对当前/下一窗口有用，
但长期保留会造成 downstream drift。
```

因此将 TTT memory 拆成：

```text
W_long:
    长期 static / scale memory

W_short:
    短期 correction / dynamic adaptation
```

应用时：

$$
W^{apply}_{m}
=
W^{long}_{m}
+
\rho_m W^{short}_{m}
$$

提交时：

$$
W^{long}_{m+1}
=
W^{long}_{m}
+
G_{corr}
+
\lambda_{cont}G_{cont}
$$

$$
W^{short}_{m+1}
=
\kappa W^{short}_{m}
-
\gamma G_{harm}
$$

其中 $0 \le \kappa < 1$ 是短期寿命衰减。

## 实验候选

### DLTTT_01：harmful projection short-lived

```text
G_harm -> W_short
K = 1 reset group
W_long only receives corr/cont
```

### DLTTT_02：body short / exit long

```text
chunks 5-9:
    high conflict residual enters W_short

chunks 10-12:
    continuity residual enters W_long
```

### DLTTT_03：scale-state short correction

只把 scale-proxy harmful residual 放入短期：

```text
harm = projection onto scale aggravation direction
```

### DLTTT_04：fallback-preserve version

如果 proxy uncertainty 高：

```text
W_short = 0
W_long = baseline
```

## 成立标准

```text
ATE <= 30.0: target pass
ATE <= 32.0: strong structural pass
[200,300) <= 60 且 [400,600) <= 44.354: mechanism pass
```

不满足时：

```text
Codex action:
    1. 如果 W_short 过强:
        reduce rho / increase decay.
    2. 如果 W_short 没作用:
        check apply hook and state diff.
    3. 如果局部改善、后段恶化:
        shorter lifetime K=1 or exit-only neutral.
    4. 如果完全无效:
        TTT long/short split 不作为主线，转 Phase 6。
```

---

# Phase 5：把 no-GT pose proxy 接入 online pipeline，但标记为 trajectory-state module

## 目标

如果 TTT projection / selection 不能接近 target-25，但 no-GT pose proxy 明确有效，则把它工程化为 online trajectory-state module，而不是伪装成 TTT 写入。

这一步不是 TTT write success，但它可能是最终系统必须有的模块。

## 设计

在线模块只允许使用：

```text
predicted poses
overlap pointmaps
reset-window step statistics
chunk-local confidence
HMC no-GT memory diagnostics
```

禁止使用：

```text
GT
evaluation residual
post-hoc full-sequence future statistics
```

## Online policy

### POSEMOD_01：causal reset-window scale state

只使用过去和当前 reset group：

$$
\hat s_m =
\operatorname{clip}
\left(
\frac{
\operatorname{EMA}_{past}\operatorname{median}\|\Delta p\|
}{
\operatorname{median}_{current}\|\Delta p\|+\epsilon
},
s_{min},s_{max}
\right)
$$

### POSEMOD_02：overlap pointmap scale state

利用当前/上一 chunk overlap pointmap norm ratio：

$$
\hat s_m =
\operatorname{median}_{p \in \Omega_{ov}}
\frac{
\|x^{prev}_{p}\|
}{
\|x^{cur}_{p}\|+\epsilon
}
$$

### POSEMOD_03：hybrid scale state

$$
\hat s_m =
\alpha \hat s^{step}_m
+
(1-\alpha)\hat s^{ov}_m
$$

### POSEMOD_04：scale state feeds TTT write

不是直接改输出，而是：

```text
scale_state high:
    reduce harmful projection commit
scale_state low:
    preserve continuity / increase positive scale-support update
```

## 成立标准

如果作为 formal trajectory-state module：

```text
KITTI01 ATE <= 25.0
FinalErr <= 9.0
[0,100) 不成为新灾区
cross-seq sanity 不崩
```

如果作为 TTT assist：

```text
TTT full output ATE <= 32.0
且 no offline trajectory rewrite
```

---

# Phase 6：cross-sequence sanity

## 触发条件

只有以下候选进入跨序列：

```text
1. online TTT write candidate ATE <= 32.0
2. online pose-state module ATE <= 25.0
3. oracle projection 显示 TTT action 有强上界
```

## 数据集

```text
KITTI00 full
KITTI02 full
KITTI05 full
KITTI08 full optional
```

## 必须比较

```text
H9_READBETA2_03 equivalent baseline
WINGAM_03 / C16ROLE style baseline
best online TTT candidate
best pose-state candidate
NOGTPOSE diagnostic only
```

## 成立标准

TTT candidate 泛化通过：

```text
平均 ATE 相比 baseline 改善 >= 1.0m
无单序列 regression > 5%
KITTI02 不明显恶化
```

Pose-state candidate 泛化通过：

```text
KITTI01 <= 25
KITTI00/02/05 平均改善 >= 20%
FinalErr regression 不超过 25%
```

---

## 7. 必须记录的指标

### 7.1 全局指标

```text
ATE_RMSE
Rot_RMSE
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
```

### 7.2 segment 指标

必须记录：

```text
[0,100)
[100,200)
[200,300)
[200,400)
[300,400)
[400,500)
[400,600)
[600,800)
```

以及：

```text
50f_mean
50f_worst
100f_mean
100f_worst
200f_mean
200f_worst
```

### 7.3 reset-window drift 指标

```text
reset_group_id
group_start_frame
group_end_frame
group_ATE
group_scale_proxy_step
group_scale_proxy_overlap
group_yaw_proxy
group_pose_jump
group_drift_norm
group_downstream_cost
```

### 7.4 TTT write 指标

```text
chunk_id
layer_id
branch_id
update_norm
update_cos_to_baseline
update_cos_to_projection_direction
projection_helpful_energy
projection_harmful_energy
positive_mass
neutral_mass
negative_mass
continuity_mass
write_score_mean
write_score_p10_p50_p90
memory_rel_diff
candidate_state_diff
selected_candidate_id
```

### 7.5 no-GT proxy 指标

```text
step_length_median
step_length_ema
step_length_log_ratio
overlap_scale_ratio
apply_mismatch
pose_increment_jump
proxy_score
proxy_confidence
selector_margin
fallback_triggered
```

---

## 8. 必须可视化

### 8.1 Global drift dashboard

每个重要 run 输出：

```text
trajectory XY
trajectory XZ
per-frame translation error
per-frame yaw error
per-frame scale proxy
per-reset group ATE
per-segment ATE bars
```

### 8.2 TTT projection dashboard

```text
chunk x layer x branch projection helpful energy
chunk x layer x branch harmful energy
role mass over chunks
selected candidate over chunks
update cosine to baseline
update cosine to drift direction
```

### 8.3 Pose-scale proxy dashboard

```text
step_length median by reset group
overlap scale ratio by chunk
predicted scale correction by chunk
proxy confidence by chunk
proxy score vs segment ATE
NOGTPOSE diagnostic vs online output comparison
```

### 8.4 Failure gallery

必须包含：

```text
best TTT candidate
best pose proxy candidate
H9 baseline
NOGTPOSE_27 diagnostic
```

对比：

```text
[0,100)
[200,300)
[400,600)
final 100 frames
```

---

## 9. 并行执行安排

为了加速，分成 5 条 Codex track 并行。

### Track A：Instrumentation

目标：

```text
补齐 true projection oracle 所需所有落盘。
```

任务：

```text
A1. per_token_update_group.pt
A2. per_layer_branch_update_matrix.pt
A3. role_mass jsonl
A4. drift_direction_gt diagnostic reader
A5. projection_energy debug
```

通过后交给 Track B/C。

### Track B：Oracle TTT projection

目标：

```text
验证 TTT write action upper bound。
```

依赖 Track A。

先跑 smoke，再跑 4 条 full。

如果失败：

```text
立即停止 TTT projection 主线，不继续微扫。
```

### Track C：No-GT selector

目标：

```text
把 no-GT pose proxy 改为 commit selector，而不是 postprocess。
```

可部分独立于 Track A。

先做 candidate state diff smoke，再跑 `SELECT_TTT_01-04`。

### Track D：Dual-lifetime TTT

目标：

```text
验证长期/短期 fast-weight 分离是否能解决局部-后段 trade-off。
```

依赖 Track A 的基本 role mass，不一定依赖 GT projection。

### Track E：Online pose-state module

目标：

```text
如果 TTT oracle 弱，快速验证部署级 pose-scale module。
```

不计入 TTT write success，但可作为 target-25 系统主线。

---

## 10. 停止规则

### 10.1 停止 TTT scalar 微调

已经停止：

```text
gamma fine sweep
neutral_lambda sweep
read_beta scalar sweep
chunk16 scalar
handoff scalar
window scalar oracle
read support scalar
```

除非某个新 action family 先通过 Phase 2/3 gate，否则不再开启。

### 10.2 停止 TTT write 主线的条件

如果满足任一：

```text
1. true TTT projection oracle ATE > 33.5
2. no-GT TTT selector ATE > 33.0 且 segment 无显著改善
3. dual-lifetime TTT ATE > 33.0 且 [200,300) 不下降
```

则 TTT write-side 降级为 auxiliary regularizer。

### 10.3 进入 target-25 主线的条件

只有满足：

```text
online output ATE <= 25.0
no postprocess
repeat within 0.05m
cross-seq sanity 不崩
```

才算正式达成。

---

## 11. 第一批具体 run 列表

### Batch 1：Instrumentation smoke

```text
V11_P1_SMOKE_projection_trace_e160
V11_P1_SMOKE_projection_trace_full_logging_noop
```

### Batch 2：TTT projection oracle

```text
V11_ORACLE_TTT_01_body_proj_w0_all
V11_ORACLE_TTT_02_body_handoff_proj
V11_ORACLE_TTT_03_scale_only_proj
V11_ORACLE_TTT_04_top_layer_branch_proj
```

### Batch 3：no-GT TTT commit selector

```text
V11_SELECT_TTT_01_chunk_local
V11_SELECT_TTT_02_reset_group
V11_SELECT_TTT_03_with_projection_risk
V11_SELECT_TTT_04_confidence_fallback
```

### Batch 4：dual-lifetime TTT

```text
V11_DLTTT_01_harm_short
V11_DLTTT_02_body_short_exit_long
V11_DLTTT_03_scale_short
V11_DLTTT_04_fallback_preserve
```

### Batch 5：online pose-state module diagnostic

```text
V11_POSEMOD_01_step_scale_causal
V11_POSEMOD_02_overlap_scale
V11_POSEMOD_03_hybrid_scale
V11_POSEMOD_04_scale_state_feeds_ttt
```

---

## 12. 最终判断标准

### 成功

```text
Online LoGeR/HMC output ATE <= 25.0
No trajectory postprocess
Cross-sequence sanity pass
```

### 结构性成功

```text
Online TTT write candidate ATE <= 30.0
or online TTT candidate [200,300) <= 55 with no downstream collapse
```

### 诊断成功

```text
TTT projection oracle proves TTT write has / does not have target-25 upper bound.
```

### 失败但有价值

```text
Pose-scale module succeeds but TTT write fails:
    target-25 系统方向转向 pose-state module；
    TTT write 保留为 regularizer。
```

---

## 13. 当前我对路线的判断

当前最重要的事实是：

```text
NOGTPOSE_27 达到 22.4012m，
但它是后处理。
```

这不能被当成 TTT 成果，但它暴露了真正问题：

> **LoGeR 当前 34m 平台的主误差是 reset-window scale / trajectory-state，而不是单纯动态 token 写入污染。**

因此下一阶段不是继续“更好地少写动态”，而是要把 TTT 写入改造成：

```text
scale-aware memory update
window-drift-aware residual composition
long/short lifetime separated fast weights
or commit candidate selection by no-GT scale proxy
```

如果这些仍然失败，就应该诚实地把 TTT write-side 降级为辅助 regularizer，转向 online pose-scale / trajectory-state 模块。这个结论不是失败，而是把问题从错误的 action space 中解放出来。

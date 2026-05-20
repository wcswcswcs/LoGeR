# ACL2 v12：面向 KITTI01 ATE 25 的 TTT 写入机制重构实验计划

日期：2026-05-18  
对象：LoGeR / HMC Pipeline v2 / KITTI01  
当前 online TTT/HMC best：`H9_READBETA2_03_repeat = 34.1258 / 6.5414`  
当前 diagnostic-only best：`NOGTPOSE_27 = 22.4012`，不计入 TTT write success  
新目标：`KITTI01 ATE <= 25.0m`  
本计划原则：**不把后处理当作 TTT 写入；不继续做普通 gamma / neutral / beta 微扫；优先验证 TTT 写入是否存在能修 whole-scene drift 的新 action space。**

---

## 0. 当前结论先写清楚

v11 的结果必须非常严格地区分三类东西。

第一类是 **online HMC / TTT write candidate**。目前最好仍然是：

```text
H9_READBETA2_03_repeat
ATE / Rot = 34.1258 / 6.5414
[200,300) = 74.410
[400,600) = 44.354
counts_as_ttt_write = true
no_postprocess = true
```

这是当前真正可计入 TTT/HMC online pipeline 的最好结果。

第二类是 **offline no-GT pose proxy**。目前最好是：

```text
NOGTPOSE_27_reset_global_clip35_body600_t105
ATE / Yaw = 22.4012 / 3.5515
[200,300) = 35.187
[400,600) = 8.587
counts_as_ttt_write = false
```

它通过了 `ATE <= 25m`，但它是离线轨迹状态修正，不是 LoGeR online full run，不是 TTT 写入成功。它只能证明 target-25 的主要误差方向存在于 window-level pose/scale state，而不能直接计入 TTT write。

第三类是 **GT oracle-only TTT write upper bound**。v11 已经修好了 projection instrumentation，并且真的跑了 online full oracle TTT write action；但是结果没有超过 H9：

```text
H9_REPEAT:       34.1258 / 6.5414
ORACLE_TTT_01:   34.8647 / 6.6735
ORACLE_TTT_02:   34.8690 / 6.6004
ORACLE_TTT_03:   34.9331 / 6.5492
ORACLE_TTT_04:   36.3971 / 6.6635
ORACLE_TTT_05_INV: 35.5911 / 6.6745
ORACLE_TTT_06_INV: 35.7524 / 6.6115
```

这说明：**当前这种 scale-projection-routed token risk action，不仅不是 deployable TTT write，而且连 oracle upper bound 都不够。**

因此 v12 的起点不是“继续修一个没接通的 projection hook”。v11 R4 已经接通，Phase 2 full oracle 已经失败。v12 的核心问题是：

> 当前 TTT 写入 action space 本身不对。它仍然在用 token risk / gamma / role mass 间接影响 fast weights，而真正主误差是 reset-window 级别的 scale / trajectory-state drift。

---

## 1. 实验整体目标

v12 不再以“某个 run 小幅超过 34.1258”为主要目标。新的实验目标分四层。

### 1.1 目标 A：判断 TTT 写入是否还有 target-25 主线价值

目前 online TTT/HMC best 是 `34.1258m`，目标是 `25.0m`，差距为：

$$
\Delta_{target}=34.1258-25.0=9.1258m
$$

如果一个实验族只能带来 `0.02m-0.10m` 的提升，就算它超过当前 best，也不能作为 target-25 主线。v12 必须优先验证是否存在 **能够至少打开 2m 以上空间** 的 TTT 写入机制。

### 1.2 目标 B：把 TTT 写入动作从 scalar gate 升级为 window-level decision

v7-v11 已经测试过大量标量动作：

```text
gamma
negative fraction
neutral lambda
chunk body / exit window
chunk16 weak tri-replay
read beta cooling
projection-routed token risk
```

这些动作的问题是都没有直接控制 window-level drift state。v12 的写入策略必须至少具备以下能力之一：

```text
1. 对同一 chunk 生成多个 candidate TTT commit，并根据 no-GT window consistency 选择 W_{m+1}；
2. 在 TTT replay 中加入 overlap / pose-scale consistency auxiliary objective；
3. 把 fast weights 分成 long-memory 与 short-correction 两个生命周期；
4. 根据未来 1-2 个 chunk 的 cheap probe consistency 做 MPC-style write selection；
5. 让 TTT 写入显式服务 window-level scale proxy，而不是只服务 token hidden-value reconstruction。
```

### 1.3 目标 C：禁止把离线轨迹后处理计入 TTT 写入成功

`NOGTPOSE_27` 是很强的 diagnostic：它说明 window-local scale correction 可以把 KITTI01 ATE 从 `34.1258m` 拉到 `22.4012m`。但是它不能被写成 TTT 写入成功。

v12 的所有 candidate 必须满足：

```text
output_from_online_hmc = true
no_postprocess = true
counts_as_ttt_write = true
committed HMC/TTT state changes future chunks
```

如果某个方法只改写最终 `01.txt` trajectory，不进入 HMC future state，一律标为 diagnostic-only。

### 1.4 目标 D：加快实验，允许 Codex 并行推进，但每条线都有停止规则

v12 必须减少“跑完才知道没意义”的 full run。每条实验线都要有：

```text
smoke gate
artifact gate
offline replay gate
short full gate
full KITTI01 gate
停止条件
下一尝试方向
```

Codex 可以并行做工程和短实验，但不能绕过 gate 直接跑大矩阵。

---

## 2. 当前失败的本质分析

### 2.1 不是 cue 完全失败，而是 cue 与 action 不匹配

`update_conflict_energy` 是有价值的 TTT-native cue。它帮助系统从 `36m` 级进入 `34m` 级。但 v8-v11 后的结果说明，它作为 scalar risk source 已经平台化。

`update_conflict_energy` 能告诉我们：

```text
哪些 chunk / layer / branch 的 fast-weight update 与当前 aggregate update 冲突；
哪些窗口存在 TTT 更新方向异常；
哪些 chunks 更像 reset-window drift 的触发点。
```

但当前 action 只是：

```text
对高 risk token 增加 negative replay；
对部分 chunk 设置 body / exit gamma；
对某些 layer/head 加权；
用 GT scale residual 构造 token risk。
```

这些 action 并没有真正把 fast-weight update 投影到 **pose-scale correction direction**。它们只是改了 hidden-space TTT replay 的 token weighting。

### 2.2 projection oracle 失败说明当前 projection action 不具备 target-25 上界

v11 的 projection action 不是 logging-only，它已经进入真实 `probe_ttt_write` commit；但 `ORACLE_TTT_01-06` 全部失败，最好 `34.8647m` 也比 H9 的 `34.1258m` 差。

这说明：

```text
当前 v11_gt_scale_projection risk source
+
tri-replay token role action
+
w0 branch update
```

这套组合即使使用 GT runtime projection direction，也没有提供足够的 TTT write upper bound。

这不是“TTT 一定没用”。更精确的判断是：

> 当前把 GT scale residual 转成 token risk，再通过 tri-replay 改 fast weights 的方式不对。它没有把 scale correction 写进 TTT memory，只是在 token-level hidden update 上做了一个间接扰动。

### 2.3 NOGTPOSE 的成功说明主误差是 window trajectory-state

`NOGTPOSE_27` 把：

```text
ATE:        34.1258 -> 22.4012
[200,300): 74.410  -> 35.187
[400,600): 44.354  -> 8.587
```

这不是通过更好的 dynamic mask，也不是通过更多/更少 TTT token 写入做到的，而是通过 reset-window step-length / scale proxy 修正轨迹状态做到的。

因此主误差更像：

```text
reset-window local scale state 错；
body / exit / handoff 的 trajectory-state transition 错；
chunk 内 geometry 可能已经相对合理，但跨 chunk 累积 scale/yaw/translation drift 错。
```

### 2.4 TTT 写入如果要继续作为主线，必须直接服务 window state

LoGeR 的 TTT 理论上负责 compressed global memory 和 scale drift prevention；SWA 负责 adjacent local continuity。问题是当前 TTT write controller 修改的是 hidden-space memory replay，而没有显式 window-state objective。

v12 的新假设是：

> 好的 TTT 写入策略不是“少写动态 token”，也不是“按 risk 反向几个 token”，而是“选择或构造一个能让未来 1-2 个 chunk 的 no-GT window consistency 更好的 fast-weight state”。

这就要求 TTT 写入至少引入候选选择、辅助一致性目标、或者短/长期 memory 分离。

---

## 3. v12 核心假设

### H1：当前 TTT scalar action space 已经饱和

假设内容：

```text
gamma / neutral / role / read beta / projection risk scalar
这类低维 action 不可能把 ATE 从 34.1 拉到 25。
```

证据预期：

```text
同族 4-6 个 full run 都在 34.0-34.5 之间；
[200,300) 仍高于 70；
RPE_t 基本不动；
主要收益来自 FinalErr / Rot / Yaw 小幅调节。
```

v12 不再继续验证 H1，而把它作为停止普通微扫的前提。

### H2：TTT candidate commit selection 比单一手写策略更可能打开空间

假设内容：

同一个 chunk 的 fast-weight commit 存在多个候选状态：

```text
W_native
W_h9_style
W_trireplay_body
W_trireplay_exit
W_scale_target
W_overlap_aux
W_short_lifetime
```

单个手写策略很难全程正确，但一个 no-GT proxy 可能能在每个 reset-window 里选择更好的 `W_{m+1}`。

核心形式：

$$
W_{m+1}^{selected}=\arg\min_{W_{m+1}^{(k)}} J_{proxy}(W_{m+1}^{(k)})
$$

其中：

$$
J_{proxy}=\lambda_s J_{scale}+\lambda_o J_{overlap}+\lambda_y J_{yaw}+\lambda_a J_{apply}+\lambda_r J_{risk}
$$

如果 H2 成立，candidate selector 应该能明显超过 H9，且不是离线改轨迹，而是改变后续 chunk 的 TTT state。

### H3：TTT 写入需要辅助 window consistency objective，而不是只重构 hidden value `v`

LoGeR TTT 默认目标近似是让 fast-weight network 学会：

$$
f_W(k_i) \approx v_i
$$

但 target-25 的误差不只是 hidden-value reconstruction，而是跨 chunk scale / pose consistency。因此需要引入不依赖 GT 的辅助目标，例如 overlap consistency：

$$
L_{overlap}=\operatorname{Mean}_{i\in O}\|f_W(q_i^{tail})-f_W(q_i^{head})\|_2^2
$$

或者 window-scale consistency proxy：

$$
L_{scale}=\left|\log s_{step}(m)-\operatorname{EMA}_{m'}\log s_{step}(m')\right|
$$

这里不是要直接修改 final pose，而是让 TTT commit 后的 fast-weight state 更倾向生成跨 chunk 一致的 hidden readback。

### H4：TTT fast weights 需要 long / short lifecycle 分离

v7-v11 反复显示：hard freeze 能局部修 `[200,300)`，但会毁掉 `[400,600)`；negative replay 能微调但无法强修病灶。这说明 fast-weight delta 中混有：

```text
长期有用 continuity direction；
短期有用 correction direction；
长期有害 drift direction。
```

假设 TTT memory 应拆成：

$$
W_{apply}=W_{long}+\alpha_s W_{short}
$$

$$
W_{commit}=W_{long}
$$

`W_short` 只在接下来 `K` 个 chunk 中 apply，不进入长期 state。若 H4 成立，它应该降低 `[200,300)` 或 `[200,400)`，同时不显著恶化 `[400,600)`。

### H5：如果 H2-H4 都失败，TTT write-only 应降级为 auxiliary regularizer

这是一个必须写入计划的硬判断。

如果以下条件成立：

```text
TTT candidate selector 不能超过 H9；
TTT auxiliary objective 不能超过 H9；
TTT long/short memory 不能超过 H9；
TTT oracle / no-GT proxy 都无法达到 <=32m；
```

那么 target-25 不应继续押注 TTT write-only。TTT 保留为 auxiliary regularizer，主线转向 online trajectory-state / pose-scale module。但这一步必须在上述实验完成后再下结论。

---

## 4. 固定边界与 baseline

### 4.1 固定 online baseline

每一批必须包含或引用：

```text
H9_READBETA2_03_repeat
ATE / Rot = 34.1258 / 6.5414
FinalErr = 6.189
[200,300) = 74.410
[200,400) = 54.651
[400,600) = 44.354
```

### 4.2 固定 diagnostic baseline

```text
NOGTPOSE_27
ATE = 22.4012
[200,300) = 35.187
[400,600) = 8.587
counts_as_ttt_write = false
```

它只能作为 target-25 可达性参考，不能计入 TTT success。

### 4.3 固定 correctness 条件

每个 TTT write candidate 必须满足：

```text
online_full_run = true
output_from_online_hmc = true
no_postprocess = true
hmc_commit_mode = probe_ttt_write
state_changed_count = 38
counts_as_ttt_write = true
```

任一不满足，candidate 降级为 diagnostic。

### 4.4 固定评估指标

每个 full run 必须记录：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
ATE_50_mean / worst
ATE_100_mean / worst
ATE_200_mean / worst
[0,100)
[100,200)
[200,300)
[200,400)
[300,400)
[400,600)
[600,end)
per_chunk_error.csv
per_frame_error.csv
```

每个 TTT run 额外记录：

```text
per_chunk_selected_candidate
per_chunk_proxy_score
per_chunk_proxy_components
per_layer_branch_update_norm
per_layer_branch_update_cosine_native
per_layer_branch_update_to_drift_projection
tri_replay_role_mass
fast_weight_memory_rel_diff
TTT apply mismatch
TTT pre-update prediction error
TTT update_conflict_energy
```

---

## 5. Phase 0：v12 工程边界与不可混淆项

### 5.1 目标

确认 v12 实验不会把后处理、GT oracle、diagnostic-only proxy 写成 TTT success。

### 5.2 要做的工作

Codex 需要先检查或新增：

```text
v12_result_registry.csv
fields:
    run_id
    output_from_online_hmc
    no_postprocess
    counts_as_ttt_write
    uses_gt_runtime_action
    uses_offline_trajectory_rewrite
    hmc_commit_mode
    state_changed_count
    candidate_family
    gate_status
```

每个 run 结束后自动写入 registry。

### 5.3 通过标准

```text
H9_REPEAT counts_as_ttt_write = true
NOGTPOSE_27 counts_as_ttt_write = false
ORACLE_TTT_* uses_gt_runtime_action = true
ORACLE_TTT_* deployable_success = false
```

### 5.4 不通过时 Codex 的尝试方向

如果 registry 字段缺失：

```text
先补 registry，不跑 full。
```

如果 H9 repeat 与参考差异大于 `0.03m`：

```text
暂停所有新实验；
检查 run config、SWA config、read beta、reset_every、hmc_commit_mode、state hash。
```

---

## 6. Phase 1：TTT write action autopsy，弄清楚 oracle 为什么失败

### 6.1 目标

不是再跑一个 TTT 策略，而是回答：

> 为什么使用 GT scale residual 的 projection-routed TTT action 反而比 H9 差？

如果这个问题不回答，继续换 proxy 没意义。

### 6.2 核心分析对象

固定以下 run：

```text
H9_REPEAT
ORACLE_TTT_01
ORACLE_TTT_02
ORACLE_TTT_03
ORACLE_TTT_05_INV
NOGTPOSE_27 diagnostic trajectory
```

### 6.3 必须计算的量

#### A. projection action 与真实 output scale 变化的关系

对每个 active chunk $m$，记录：

```text
risk_mean
risk_p90
projection_helpful_energy
projection_harmful_energy
projection_role_mass
actual_delta_step_scale_after_action
actual_delta_yaw_after_action
actual_delta_segment_ATE
```

计算相关性：

$$
\rho_{scale}=\operatorname{Spearman}(projection\_helpful\_energy, -\Delta J_{scale})
$$

$$
\rho_{ate}=\operatorname{Spearman}(projection\_helpful\_energy, -\Delta ATE_{segment})
$$

如果 `projection_helpful_energy` 与实际 scale/ATE 变化无关或反向，说明 current projection feature 不是有效 action coordinate。

#### B. candidate fast-weight delta 是否进入有效 branch/layer

记录：

```text
layer_id
branch_id
native_update_norm
candidate_update_norm
update_cos_to_native
update_cos_to_window_drift
candidate_minus_native_norm
```

画 heatmap：

```text
layer × branch update norm
layer × branch cosine to drift
layer × branch actual effect proxy
```

#### C. H9 与 ORACLE_TTT 的 trajectory damage source

必须比较：

```text
[200,300)
[200,400)
[400,600)
FinalErr
YawRMSE
Sim3Scale
per-reset scale proxy curve
```

### 6.4 假设成立标准

H1.1：projection action coordinate 失败，如果满足任一：

```text
projection_helpful_energy 与 actual scale/ATE improvement Spearman <= 0.2；
ORACLE_TTT normal 与 inverse 都比 H9 差；
active chunk 上风险能量变大但 segment ATE 不降；
```

H1.2：branch/layer作用空间错，如果满足：

```text
只有 w0 动作有效但 actual scale 变化不相关；
或高 projection energy 集中在 layer/head 但 action 没有改变对应 fast-weight apply output；
```

### 6.5 不满足条件时的 Codex 尝试方向

如果 projection energy 与 actual scale improvement 相关，但 full ATE 仍差：

```text
进入 Phase 2 candidate selection；
说明 projection cue 有用，但单一 action policy 不好。
```

如果 projection energy 完全不相关：

```text
停止 v11_gt_scale_projection risk source；
转 Phase 3 overlap/scale auxiliary objective。
```

如果相关性只在某些 layer/branch 上成立：

```text
Codex 实现 layer/branch-specific projection action；
只在相关 layer/branch 上进入 Phase 2 小矩阵。
```

---

## 7. Phase 2：TTT candidate commit bank + no-GT selector

### 7.1 目标

把 TTT 写入从“一个手写策略”改成“每个 chunk / reset-window 在多个候选 TTT commit 中选择”。

这不是后处理，因为最终改变的是：

$$
H_{m+1}=(W_{m+1}, history_{m+1})
$$

而不是改写输出 trajectory。

### 7.2 候选 fast-weight state

对每个 target chunk 或 reset-window，生成候选：

```text
C0: native / H9 default
C1: tri-replay WINGAM-style
C2: C16ROLE-style
C3: ORACLE_TTT-style projection risk without GT, if Phase1 allowed
C4: overlap auxiliary TTT candidate, if Phase3 implementation ready
C5: long/short lifetime candidate, if Phase4 ready
C6: conservative candidate: native write + H9 read beta
```

候选的核心不是多跑完整模型，而是在同一 probe cache 上复用 replay，生成多个 `W_{m+1}^{(k)}`。

### 7.3 no-GT selector 目标函数

候选选择不使用 GT。定义：

$$
J_{proxy}^{(k)}=\lambda_s J_{scale}^{(k)}+\lambda_o J_{overlap}^{(k)}+\lambda_y J_{yaw}^{(k)}+\lambda_a J_{apply}^{(k)}+\lambda_r J_{risk}^{(k)}
$$

其中：

#### Scale proxy

用预测 pose increment 的 reset-window step-length 统计：

$$
J_{scale}=\left|\log \operatorname{Median}_{t\in W}\|\Delta p_t\|_2-\operatorname{EMA}(\log \|\Delta p\|_2)\right|
$$

#### Overlap consistency proxy

$$
J_{overlap}=\operatorname{Mean}_{p\in O}\|X_{tail}(p)-X_{head}(p)\|_2
$$

#### Yaw continuity proxy

$$
J_{yaw}=\left|\Delta yaw_m-\operatorname{Median}_{m'\in \mathcal{N}}\Delta yaw_{m'}\right|
$$

#### TTT apply mismatch proxy

$$
J_{apply}=\operatorname{Mean}_{i}\frac{\|f_W(q_i)-v_i\|_2}{\|v_i\|_2+\epsilon}
$$

#### Risk penalty

$$
J_{risk}=\operatorname{Mean}_{l,b}\operatorname{ReLU}(\|\Delta W_{l,b}\|-\tau_{l,b})
$$

### 7.4 实验顺序

#### Phase 2A：offline selector audit，不跑 full

用已有 H9 / C16ROLE / WINGAM / ORACLE_TTT trajectories 和 debug，离线评估 selector 是否能排出更好的候选。

通过标准：

```text
selector 在已有候选上选择 H9 或优于 H9 的候选比例 >= 70%；
selector 不应选择 ORACLE_TTT_01 这类 ATE 明显更差的候选；
selector score 与 ATE Spearman >= 0.5；
```

如果不通过，不允许跑 full。

#### Phase 2B：short horizon smoke

使用 `END_FRAME=256` 或固定 chunks `5-12`，测试 candidate selection 能否真实改变 committed state。

必须检查：

```text
selected_candidate_id != always C0
state hash changes according to selected candidate
no postprocess
runtime <= 2x H9 short run
```

#### Phase 2C：KITTI01 full 小矩阵

只跑 4 个 full run：

```text
SELECT_TTT_01: selector weights balanced
SELECT_TTT_02: scale-heavy selector
SELECT_TTT_03: overlap-heavy selector
SELECT_TTT_04: conservative selector, only switch when proxy margin > delta
```

### 7.5 成功标准

强成功：

```text
ATE <= 30.0
[200,300) <= 55
[400,600) <= 44.354
```

阶段成功：

```text
ATE <= 32.5
或 ATE <= 33.0 且 [200,300) <= 65
```

弱成功：

```text
ATE < 34.1258 - 0.30
且 [200,300), [400,600), FinalErr 至少两项不恶化
```

### 7.6 不满足条件时 Codex 的尝试方向

如果 selector 总是选择 C0：

```text
说明 proxy 过保守；降低 switch margin，增强 scale/overlap weight；先不要 full，再跑 offline audit。
```

如果 selector 选择非 C0 但 ATE 变差：

```text
检查 proxy 与 ATE 的 rank correlation；
若 correlation < 0.5，停止 selector full，回 Phase 1/3 改 proxy。
```

如果 `[200,300)` 改善但 `[400,600)` 恶化：

```text
进入 Phase 4 long/short lifetime；
说明 candidate 改变了 body drift，但破坏后段 continuity。
```

如果 selector 过 `32.5m`：

```text
立刻 repeat；
再跑 KITTI00/02/05 sanity；
同时做 component ablation。
```

---

## 8. Phase 3：TTT overlap / scale auxiliary replay objective

### 8.1 目标

当前 TTT 默认 replay 学的是 hidden value memory：

$$
f_W(k_i) \rightarrow v_i
$$

v12 要测试：如果把 TTT replay objective 改成兼顾 overlap / scale consistency，是否比 token risk gate 更有效。

### 8.2 机制 A：overlap consistency replay

在 probe cache 中选 overlap static tokens：

```text
tail overlap from previous/current boundary
head overlap of next chunk or current head
D_g low
semantic structure or high confidence
```

构建辅助 key/value 对：

$$
(k_i^{tail}, v_i^{head})
$$

或让 apply output 对齐：

$$
L_{aux}=\operatorname{Mean}_{i\in O}\|f_W(k_i^{tail})-v_i^{head}\|_2^2
$$

在实现上可以先不显式求新的 loss，只做 pseudo-replay：把 overlap consistency pair 插入 TTT replay token set，并赋予较小权重。

### 8.3 机制 B：scale-stable structure replay

选择结构 token：

```text
D_g low
stage_d high
semantic structure if available
old_dyn low
confidence high
```

在 reset-window 中对这些 token 增加正向写入，但只在 window scale proxy 异常时启用。

形式：

$$
S_{structure}=stage\_d\cdot \sqrt{1-D_g}\cdot R_{scale\_risk}
$$

其中：

$$
R_{scale\_risk}=\operatorname{clip}\left(\frac{|\log s_{step}-\operatorname{EMA}(\log s)|}{\tau_s},0,1\right)
$$

### 8.4 机制 C：apply mismatch suppression

如果某些 token 的 TTT apply 与 observation value mismatch 很高：

$$
E_{apply,i}=\frac{\|f_W(q_i)-v_i\|_2}{\|v_i\|_2+\epsilon}
$$

且它们同时是 high-D 或 high-unc，则将其从 long memory write 中弱化，但不要反向；反向只给 direction-conflict 明确的 token。

### 8.5 实验矩阵

第一批只跑 6 条：

```text
AUX_TTT_01: overlap pseudo-replay, weight 0.05
AUX_TTT_02: overlap pseudo-replay, weight 0.10
AUX_TTT_03: scale-stable structure replay, weight 0.05
AUX_TTT_04: scale-stable structure replay, weight 0.10
AUX_TTT_05: apply mismatch suppression, mild
AUX_TTT_06: overlap + scale-stable replay, conservative
```

### 8.6 记录指标

除主指标外必须记录：

```text
overlap_replay_token_count
overlap_replay_weight_mean
structure_replay_mass
apply_mismatch_mean_before / after
window_scale_proxy_before / after
per-layer update norm change
per-branch update norm change
```

### 8.7 成功标准

强成功：

```text
ATE <= 32.5
且 [200,300) <= 65
且 [400,600) <= 44.354
```

弱成功：

```text
ATE <= 33.8
且比 H9 改善 >= 0.30m
且 100f/200f mean 至少一个明显改善
```

### 8.8 不满足条件时 Codex 的尝试方向

如果 overlap replay 改善 boundary 但不改善 ATE：

```text
将它作为 Phase 2 selector 的 candidate，不再单独调权重。
```

如果 scale-stable replay 改善 ATE 但 FinalErr 恶化：

```text
加入 exit/handoff conservative switch；
或者将该 candidate 放入 Phase 4 short/long memory。
```

如果所有 AUX_TTT 都失败：

```text
说明只改 replay token/target仍不够；转 Phase 4 dual-lifetime 或 Phase 5 decision gate。
```

---

## 9. Phase 4：dual-lifetime TTT memory，不再把所有 update 写入长期 fast weights

### 9.1 目标

解决 freeze / TTGR / post-window 实验反复出现的 trade-off：

```text
局部 body 病灶改善 -> 后段 continuity 恶化
后段 continuity 保留 -> [200,300) 降不下来
```

### 9.2 机制

维护两套 fast-weight delta：

```text
W_long: 长期 commit，进入未来 reset-window
W_short: 短期 correction，只 apply K 个 chunk，不长期 commit
```

形式：

$$
W_{apply,m}=W_{long,m}+\alpha_m W_{short,m}
$$

$$
W_{commit,m+1}=W_{long,m+1}
$$

短期 memory 来源：

```text
high update_conflict_energy
high apply mismatch
high D_g but overlap-relevant
scale-risk body window correction
```

长期 memory 来源：

```text
structure low-D tokens
stage_d high tokens
low apply mismatch
stable overlap consistency tokens
```

### 9.3 最小实现

Codex 不需要一次性重构完整模型。先实现 commit-time overlay：

```text
保存 W_short_delta_m
在未来 K 个 chunks 的 TTT apply 前临时加到 W_long
commit 时不把 W_short 合并进 W_long
K 到期后 decay 或删除
```

### 9.4 实验矩阵

```text
DLTTT_01: K=1, alpha=0.25, short = high conflict residual
DLTTT_02: K=2, alpha=0.25
DLTTT_03: K=2, alpha=0.50
DLTTT_04: K=3, alpha=0.25
DLTTT_05: K=2, short only body chunks 5-9
DLTTT_06: K=2, short body + conservative exit
```

### 9.5 记录指标

```text
W_short_norm_by_chunk
W_short_decay_curve
W_long_update_norm
W_short_apply_effect_norm
apply_mismatch_with_short / without_short
[200,300)
[400,600)
per-reset scale proxy
```

### 9.6 成功标准

强成功：

```text
ATE <= 30.0
```

阶段成功：

```text
ATE <= 32.5
或 [200,300) <= 60 且 [400,600) <= 44.354
```

弱成功：

```text
ATE <= 33.8
且 [200,300) 改善 >= 3m
且 [400,600) 回退 <= 1m
```

### 9.7 不满足条件时 Codex 的尝试方向

如果 W_short 改善 `[200,300)` 但伤 `[400,600)`：

```text
降低 K 或 alpha；
把 short 只作用在 apply，不作用 update；
把 short 限制到 layer/branch with high sensitivity。
```

如果 W_short 完全无效：

```text
回 Phase 2 candidate selector；
说明 short correction 不在当前 TTT apply path 可控。
```

如果 W_short 让 Rot/FinalErr 变好但 ATE 不动：

```text
标记为 auxiliary regularizer；
不再当 target-25 主线。
```

---

## 10. Phase 5：TTT-sensitive finite-difference / one-step MPC

### 10.1 目标

v11 的 projection action 是先构造 risk，再希望它影响 scale。但它没有验证 fast-weight update 对 future pose 的敏感性。Phase 5 直接测：

> 对某个 layer/branch 的 TTT delta 做小扰动，下一 chunk 的 no-GT scale / overlap / apply mismatch 会如何变化？

### 10.2 方法

对当前 committed state $W_m$，生成小扰动候选：

$$
W_m^{(k)}=W_m+\epsilon \Delta W_k
$$

其中 $\Delta W_k$ 可以是：

```text
native update direction
negative conflict direction
structure positive direction
overlap auxiliary direction
random orthogonal control
```

对每个候选只 probe 下一 chunk或下一两个 chunk，不完整跑全序列，计算：

$$
\Delta J_{proxy}^{(k)}=J_{proxy}(W_m^{(k)})-J_{proxy}(W_m)
$$

由此得到 sensitivity：

$$
S_k=-\frac{\Delta J_{proxy}^{(k)}}{\epsilon}
$$

### 10.3 使用方式

如果某些 $\Delta W_k$ 对 no-GT proxy 有明显负梯度，则 Phase 2 candidate bank 增加该候选。

### 10.4 成功标准

```text
至少一个 non-random TTT direction 的 proxy sensitivity 显著强于 random control；
该 direction 在 short full run 中能改善对应 window proxy；
```

### 10.5 不满足条件时 Codex 的尝试方向

如果所有 TTT direction sensitivity 都弱：

```text
TTT write path 可能无法直接驱动 target-25 的 scale state；
进入 Phase 6 decision gate。
```

如果 random direction 也能同等改善：

```text
proxy 不可靠；
先修 proxy，不跑 full。
```

---

## 11. Phase 6：是否继续 TTT write 主线的硬决策

### 11.1 决策目标

v12 必须避免无限循环。完成 Phase 2-5 后，必须给出结论：

```text
TTT write 是否仍是 target-25 主线？
还是只作为 auxiliary regularizer？
```

### 11.2 保留 TTT write 主线的条件

满足任一：

```text
1. 任一 online TTT candidate ATE <= 30.0；
2. 任一 online TTT candidate ATE <= 32.5，且 [200,300) <= 65；
3. candidate selector / auxiliary / dual-lifetime 中至少一个方向连续两批超过 H9 >= 0.5m；
4. finite-difference sensitivity 找到稳定可控 direction，且 short/full 都验证。
```

### 11.3 降级 TTT write 的条件

满足任一：

```text
1. Phase 2 selector 失败；
2. Phase 3 auxiliary objective 失败；
3. Phase 4 dual-lifetime 失败；
4. Phase 5 sensitivity 显示 TTT write 对 window scale proxy 不敏感；
5. 连续 12 个 online TTT full run 没有超过 H9 by 0.3m。
```

如果降级，结论必须写成：

> TTT write 仍可作为 Rot / FinalErr / local regularizer，但 target-25 主线需要 online pose-state / trajectory-state module。不要再把后处理 proxy 伪装成 TTT write。

---

## 12. 并行执行计划

### Track A：工程与 registry

负责人：Codex A  
任务：

```text
1. v12_result_registry.csv 自动写入；
2. no-postprocess / counts_as_ttt_write flags；
3. run_config.yaml 与 HMC state hash；
4. dashboards 统一生成。
```

停止规则：

```text
如果 registry 不完整，不允许任何 full run 晋级。
```

### Track B：Phase 1 autopsy

负责人：Codex B  
任务：

```text
1. 读取 H9 / ORACLE_TTT traces；
2. 计算 projection energy 与 actual scale/ATE delta 的相关性；
3. 输出 layer × branch heatmap；
4. 生成 failure report。
```

如果相关性低，Codex B 同时给 Phase 3 proxy 需要修正的建议。

### Track C：Candidate commit bank

负责人：Codex C  
任务：

```text
1. 在 ttt_write_controller 内生成 multiple candidate W_{m+1};
2. 实现 selector dry-run；
3. 先 END_FRAME=256 smoke；
4. 通过后跑 SELECT_TTT_01-04。
```

不满足条件时：

```text
如果生成候选太慢，先只做 chunks 5-12；
如果候选状态 hash 未变，先修 action path；
如果 selector 总选 C0，降低 margin 并做 offline audit。
```

### Track D：Auxiliary replay objective

负责人：Codex D  
任务：

```text
1. 实现 overlap pseudo-replay；
2. 实现 scale-stable structure replay；
3. 实现 apply mismatch suppression；
4. 分别跑 smoke 和 AUX_TTT_01-06。
```

不满足条件时：

```text
如果 token 数过多导致慢，先限制到 overlap static top-k；
如果 full ATE 回退，保留该候选给 Phase 2 selector，不继续单独扫权重。
```

### Track E：Dual-lifetime TTT

负责人：Codex E  
任务：

```text
1. 实现 W_short overlay；
2. K=1/2 smoke；
3. 跑 DLTTT_01-06；
4. 输出 short/long norm dashboard。
```

不满足条件时：

```text
如果 W_short 破坏 HMC state parity，先只做 diagnostic apply，不进入 full；
如果 W_short full run 只改善 Rot，不改善 ATE，停止该族。
```

### Track F：MPC sensitivity

负责人：Codex F  
任务：

```text
1. 实现 one-step probe candidate；
2. 计算 proxy sensitivity；
3. 输出 random-control 对照；
4. 只把有 sensitivity 的 direction 交给 Track C。
```

---

## 13. 可视化与报告要求

每个阶段必须输出以下图表。

### 13.1 Global drift dashboard

```text
trajectory_xy_full.png
trajectory_xy_0_300.png
trajectory_xy_300_end.png
per_frame_error.png
sliding_50_100_200_ate.png
sim3_scale_over_time.png
cumulative_yaw_drift.png
axis_rmse_x_y_z.png
reset_group_error.png
```

### 13.2 TTT action dashboard

```text
layer_branch_update_norm_heatmap.png
layer_branch_update_cos_native_heatmap.png
update_to_drift_projection_heatmap.png
projection_helpful_vs_harmful_energy.png
tri_replay_role_mass_over_chunks.png
candidate_selection_timeline.png
W_long_short_norm_timeline.png
```

### 13.3 Window-state dashboard

```text
step_length_median_by_reset.png
scale_proxy_before_after.png
overlap_residual_by_chunk.png
window_proxy_score_by_candidate.png
selector_choice_by_chunk.png
```

### 13.4 Failure attribution plots

如果候选失败，必须输出：

```text
candidate_minus_H9_per_chunk_error.png
candidate_minus_H9_segment_bar.png
proxy_score_vs_actual_delta_scatter.png
selected_candidate_vs_oracle_best_table.csv
```

---

## 14. v12 第一批建议运行顺序

### Batch 0：registry / repeat / smoke

```text
B0-01: H9 repeat, registry check
B0-02: NOGTPOSE diagnostic registry check
B0-03: ORACLE_TTT result classification check
```

### Batch 1：projection action autopsy

```text
B1-01: H9 vs ORACLE_TTT_01/02/03/05/06 trace analysis
B1-02: projection energy correlation report
B1-03: layer/branch action heatmap
```

不跑 full。

### Batch 2：candidate bank offline + smoke

```text
B2-01: offline selector audit on existing candidates
B2-02: END_FRAME=256 candidate bank smoke
B2-03: selected state hash / commit check
```

只有通过 offline selector gate 才跑 full。

### Batch 3：candidate selector full

```text
SELECT_TTT_01: balanced proxy
SELECT_TTT_02: scale-heavy proxy
SELECT_TTT_03: overlap-heavy proxy
SELECT_TTT_04: conservative margin selector
```

### Batch 4：auxiliary replay objective

```text
AUX_TTT_01-06
```

可与 Batch 3 并行，但要先完成 smoke。

### Batch 5：dual-lifetime

```text
DLTTT_01-06
```

只在 Batch 3 或 4 出现局部改善但后段恶化时优先运行。

### Batch 6：cross-sequence sanity

只有 candidate 满足：

```text
ATE <= 32.5
or ATE <= 33.0 and [200,300) <= 65
```

才跑：

```text
KITTI00
KITTI02
KITTI05
```

---

## 15. 总结

v12 的核心不是继续寻找一个更细的 `gamma`。当前数据已经说明，普通 TTT scalar write action 不能把 KITTI01 从 `34.1258m` 推到 `25m`。`NOGTPOSE_27` 证明 target-25 的主要杠杆是 window-level scale / trajectory-state，但它目前是后处理 diagnostic。v11 的 projection-routed TTT oracle 又证明，简单把 scale residual 转成 token risk 并走 tri-replay，并没有 target-25 上界。

因此下一步必须验证新的 TTT write action space：

```text
candidate commit selection
auxiliary overlap / scale consistency replay
dual-lifetime TTT memory
finite-difference / MPC-style sensitivity
```

如果这些都失败，项目应明确把 TTT write 降级为 auxiliary regularizer，把 target-25 主线转向 online pose-state / trajectory-state module。这个结论不是放弃 TTT，而是避免继续把一个不具备直接 scale-control action 的写入路径当成主优化杠杆。

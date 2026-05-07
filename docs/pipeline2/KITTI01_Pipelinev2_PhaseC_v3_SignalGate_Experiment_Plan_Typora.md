# KITTI 01 / HMC Phase C v3：Read-Path Signal Gate 细化实验方案

> 目标：在新的 **Hybrid Memory Controller（HMC）two-pass pipeline** 上，重新细化 Phase C 的 read-path signal gate。Phase C v3 不再围绕当前 `C_dyn` family 继续扫 beta / protection / top-k，而是先重新确认 HMC pipeline correctness，再把 Phase C 拆成 **code parity gate → cue-source quality gate → stateful slice gate → read-path injection gate → full-sequence acceptance gate**。只有当 read-path-only controller 在 full KITTI 01 上稳定超过 RFR-100，并跨过 `<= 40.8 m` 的 Phase C gate，才允许进入 Phase D 和 TTT branch0 write 组合。

---

## 0. 当前结论与 Phase C v3 的必要性

Phase C v1 和 v2 已经说明一件事：read-path control 不是完全没有信号，但当前信号太弱、太不稳定，不能直接进入组合实验。

Phase C v1 中，`frame attention + dyn cue + pair bias + early layers + beta=1.0`，也就是 `RFR-100`，在 full KITTI 01 上达到：

$$
ATE_{RFR100}=41.0733\text{ m}
$$

它比 HMC v2 native baseline 的 $41.7502\text{ m}$ 和旧的 TTT branch0 write baseline `BL01` 的 $41.3665\text{ m}$ 都更好，但它仍然没有通过 `<41.0 m` 的 debug gate。更重要的是，beta 继续增大后 ATE 反而恶化，说明失败不是因为控制强度不足，而是 signal / layer / path 的可靠性不够。

Phase C v2 做了更细的 FineGate：`dyn_reliable`、`protected_pair`、`top-k dynamic mass cap`、`internal_attn`、`key-only bias`、`early_quarter` 等。短序列和 256-frame 中有一些看似有希望的结果，例如 `dyn_reliable + protected_pair` 在 256-frame 上比 RFR reference 更好；但 full KITTI 01 上它退化到 $41.4679\text{ m}$。这说明 **64-frame / 256-frame prefix gain 不能作为 full-sequence model selection 的充分依据**。

因此 Phase C v3 的核心不是继续问：

$$
\text{beta 要不要更大？top-k 要不要更小？protection 要不要更强？}
$$

而是要问：

$$
\text{当前 cue 是否真的能在长序列全局轨迹上区分 harmful dynamic token 和 useful geometry token？}
$$

以及：

$$
\text{这个 cue 应该注入 frame attention、SWA read，还是 chunk-wise bidirectional attention？}
$$

Phase C v3 先不组合 TTT write。它只解决 read-path-only controller 是否真的值得保留的问题。

---

## 1. Phase C v3 的中心假设

### 1.1 假设 H0：HMC 新 pipeline 的 no-control / identity-hook 必须严格复现 LoGeR 和 LoGeR\*

Phase C v3 必须建立在新 HMC pipeline 自己的正确性上，不能借用旧 `geometry_eval_mode`、旧 native controller、旧 TTT-only route 或旧 merge 逻辑来产出正式 baseline。LoGeR 原始模型是 hybrid memory 架构，每个 block 的顺序是：

$$
\text{per-frame attention}
\rightarrow
\text{SWA}
\rightarrow
\text{TTT apply/update}
\rightarrow
\text{chunk-wise bidirectional attention}
$$

HMC pipeline 需要真实走这一套 two-pass / control-hook 框架，即使 control 是 identity，也必须通过同样的 HMC orchestration、state handoff、reset、SWA、TTT、chunk attention hooks。

如果 HMC no-control 本身无法复现 LoGeR 和 LoGeR\* 在 KITTI 01 上的指标，Phase C v3 的所有 signal 结论都不可信。

---

### 1.2 假设 H1：当前 `C_dyn` family 已经达到上限，Phase C v3 需要新的 motion cue source

Phase C v2 表明，在当前 `C_dyn` family 上做 reliability filtering、top-k mass cap、protected pair、key-only bias 等，都没有稳定超过 RFR-100。`internal_attn` 的现有实现也没有成为可靠 motion cue。

这并不说明 read-path control 无效，而是说明：

$$
D_{dyn}^{old}
$$

还不够可靠。它可能经常把道路边缘、护栏、建筑边界、远处低纹理结构、强视差区域当成 dynamic / risk，从而损伤 translation-useful token。

Phase C v3 需要把 motion cue source 拆开，先在 probe-only 阶段验证 cue 本身，再进入 attention injection。

---

### 1.3 假设 H2：短 prefix 选择不可靠，必须引入 stateful slice gate

Phase C v2 中，`F256-02` 是 256-frame 最好候选，但 full 退化最明显。这说明 prefix 评估没有覆盖 long-memory state 的失败模式。Phase C v3 需要引入一种比 full run 便宜、但比 prefix 更接近真实长序列的 gate。

核心做法是：先用 HMC no-control full run 保存每个 chunk 的 committed hybrid memory state：

$$
\mathcal H_m=
\{W_m^{TTT}, H_m^{SWA}, R_m^{ref}, \text{reset metadata}\}
$$

然后从多个中后段 chunk 的真实 $\mathcal H_m$ 分叉，运行局部 controlled continuation。这样可以测试 candidate 在长期 memory 已经累积后的行为，而不是只看序列开头。

---

### 1.4 假设 H3：read-path control 需要按路径归因，而不是默认 frame attention-only

RFR-100 的 frame-attention control 是目前最好的 read-path-only result，但 LoGeR 真正跨帧几何推理还包括 SWA 和 chunk-wise bidirectional attention。Frame attention 是每帧内部空间特征提取，它未必是动态物体污染全局轨迹的主要路径。

Phase C v3 要分别回答：

$$
\text{motion cue 注入 frame attention 是否有效？}
$$

$$
\text{motion cue 注入 SWA previous-key read 是否有效？}
$$

$$
\text{motion cue 注入 chunk-wise bidirectional attention 是否有效？}
$$

只有单路径通过 gate，才允许组合。

---

## 2. Phase C v3 总体流程

Phase C v3 分成五个连续阶段。每个阶段都有明确的停止条件。

```text
C3-0  HMC correctness gate
   ↓
C3-1  Probe-only cue source quality gate
   ↓
C3-2  Stateful slice gate
   ↓
C3-3  Single-path read injection gate
   ↓
C3-4  Full KITTI 01 acceptance gate
```

Phase C v3 明确禁止下面两件事：

1. 不允许在没有重新通过 HMC correctness gate 的情况下开始 full model selection。
2. 不允许只凭 64-frame 或 256-frame prefix win 推 full candidate；必须先通过 stateful slice gate。

---

## 3. C3-0：HMC correctness gate

### 3.1 实验目的

C3-0 的目的不是提升 ATE，而是证明新的 HMC pipeline 是 LoGeR 的真实超集，而不是一条绕回旧逻辑的捷径。所有 Phase C v3 实验都必须从这里开始。

HMC no-control 应满足：

$$
\text{HMC no-control} \approx \text{LoGeR native}
$$

identity hooks 应满足：

$$
\text{HMC identity hooks} \approx \text{HMC no-control}
$$

LoGeR\* 也必须复现，因为 LoGeR\* 的 SE(3) overlap alignment / reset behavior 是另一个非常敏感的 pipeline correctness check。

---

### 3.2 必跑实验

#### C3-0A：LoGeR HMC no-control full reproduction

配置要求：

```text
checkpoint: ckpts/LoGeR/latest.pt
config: ckpts/LoGeR/original_config.yaml
hybrid_memory_mode: hmc_no_control
stage_c_mode: none
chunk_size: 32
chunk_overlap: 3
window_size: 32
overlap_size: 3
reset_every: 5
geometry_edge_rtol: 0.0
```

这一次不能使用旧 `geometry_eval_mode` 作为正式输出路径。允许用它作为外部对照，但正式 result 必须来自 HMC pipeline：

```text
Pass 1 probe disabled or identity
Pass 2 controlled pass with identity controls
HMC state commit
HMC merge / trajectory export
```

判断标准：

$$
|ATE_{HMC\_LoGeR}-41.7502| < 0.15\text{ m}
$$

更严格的理想标准：

$$
|ATE_{HMC\_LoGeR}-41.7502| < 0.05\text{ m}
$$

轨迹 sanity check：

$$
\operatorname{mean}\|t^{HMC}_i-t^{native}_i\|_2 < 0.05\text{ m}
$$

$$
\max_i\|t^{HMC}_i-t^{native}_i\|_2 < 0.15\text{ m}
$$

如果这一步失败，不进入 C3-1。

---

#### C3-0B：LoGeR\* HMC no-control full reproduction

配置要求：

```text
checkpoint: ckpts/LoGeR_star/latest.pt
config: ckpts/LoGeR_star/original_config.yaml
hybrid_memory_mode: hmc_no_control
stage_c_mode: none
chunk_size: 64
chunk_overlap: 3
window_size: 64
overlap_size: 3
reset_every: 5
se3: true
geometry_edge_rtol: 0.0
```

判断标准：

$$
|ATE_{HMC\_LoGeR^*}-47.9793| < 0.15\text{ m}
$$

更严格的理想标准：

$$
|ATE_{HMC\_LoGeR^*}-47.9793| < 0.05\text{ m}
$$

LoGeR\* 的 rotation 更好但 ATE 更差，这个特征也应该复现：

$$
RotRMSE_{HMC\_LoGeR^*} \approx 5.85^\circ
$$

如果 LoGeR\* 复现失败，需要先检查 SE(3) overlap alignment、reset-block merge、HMC state commit 是否真的走了同一条逻辑。

---

#### C3-0C：HMC identity-hook parity

在 LoGeR checkpoint 上跑 full KITTI 01，开启所有 read-path hooks，但每个 hook 都设置为 identity：

```text
frame_attention_hook: identity
swa_read_hook: identity
ttt_apply_hook: identity
chunk_attention_hook: identity
ttt_write_hook: identity / unity
```

判断标准：

$$
|ATE_{identity}-ATE_{hmc\_no\_control}| < 0.05\text{ m}
$$

并且每个 hook 的 debug 里必须记录：

```text
hook_enabled = true
hook_identity = true
num_hook_calls > 0
mean_abs_bias = 0
max_abs_bias = 0
```

这一步专门防止“hook 没跑，所以看起来 parity 很好”的假通过。

---

#### C3-0D：sign and effect sanity check

在 64-frame 上做两个非模型选择的 sanity check：

1. random patch bias；
2. inverted dynamic cue bias。

它们不需要好，但必须让输出变化。判断标准：

$$
ATE_{random} \neq ATE_{identity}
$$

且 debug 中：

```text
mean_abs_bias > 0
attention_prob_delta > 0
```

如果 random / inverted cue 完全不改变输出，说明 read-path hook 实际没有生效。

---

## 4. C3-1：Probe-only cue source quality gate

### 4.1 实验目的

C3-1 只跑 probe，不注入控制。它的目标是筛掉不可靠的 motion cue source，避免再出现 64/256-frame 看起来好、full 退化的情况。

Phase C v2 说明当前 `C_dyn` family 已经接近上限，`internal_attn` 的现有实现也不可靠。因此 C3-1 必须引入新的 cue source，并先用 dashboard / statistics 验证。

---

### 4.2 需要比较的 cue source

#### Cue A：旧 `C_dyn` reference

这是 RFR-100 的 reference cue。

$$
D^{old}_i = C_{dyn,i}
$$

它作为对照，不再作为主要创新方向。

---

#### Cue B：Gram-lite Q/K motion cue

参考 VGGT4D 的思想：普通 QK attention 会混合动态、语义和纹理，Gram similarity 更能放大 motion-induced distribution discrepancy。LoGeR GeometryOutput 里已经有若干 Q/K patch vectors 或 layer-wise Q/K features 的接口，Phase C v3 应尽量利用这些内部信号。

对每个 token $i$，在支持帧窗口 $W(t)$ 中计算 layer-group Gram similarity。可先定义：

$$
S^{KK}_{g}(t,i)=
\operatorname{Mean}_{s\in W(t),l\in g}
\cos(K_{l,t,i}, K_{l,s,i})
$$

$$
S^{QQ}_{g}(t,i)=
\operatorname{Mean}_{s\in W(t),l\in g}
\cos(Q_{l,t,i}, Q_{l,s,i})
$$

$$
V^{QK}_{g}(t,i)=
\operatorname{Var}_{s\in W(t),l\in g}
\cos(Q_{l,t,i}, K_{l,s,i})
$$

先做一个简化版：

$$
D^{gram}_i=
\operatorname{Norm}
\left[
(1-S^{KK}_{shallow,i})
\cdot
(1-S^{QQ}_{middle,i})
\cdot
S^{QQ}_{deep,i}
\right]
$$

再做 VGGT4D-style 版本：

$$
w_{shallow}=(1-S^{KK}_{shallow})\cdot V^{QK}_{shallow}
$$

$$
w_{middle}=1-S^{QQ}_{middle}
$$

$$
w_{deep}=(1-V^{QQ}_{deep})\cdot S^{QQ}_{deep}
$$

$$
D^{gram4d}=\operatorname{Norm}(w_{shallow}\cdot w_{middle}\cdot w_{deep})
$$

判断重点不是它是否像分割 mask，而是它是否比旧 `C_dyn` 更少误伤 road / building / guardrail，同时对 moving object / inconsistent object 有更高响应。

---

#### Cue C：attention entropy / dispersion cue

MUT3R 的核心观察是动态区域会产生更 dispersed / unstable 的 attention behavior。Phase C v2 的 `internal_attn` 可能太粗，只用了平均 attention；C3-1 要改成 entropy 和 temporal variance。

对某层某 token attention row $A_{i,:}$，定义：

$$
H_i=-\sum_j A_{i,j}\log(A_{i,j}+\epsilon)
$$

再在多层和支持帧上聚合：

$$
D^{ent}_i=
\operatorname{Norm}
\left(
\operatorname{Mean}_{l\in L_{early}}H_{l,i}
+
\operatorname{Var}_{s\in W(t)}H_{s,i}
\right)
$$

这个 cue 只进入 C3-1 dashboard，不直接进入 full run。只有它通过 quality gate，才进入 C3-2。

---

#### Cue D：flow-epipolar / static-flow residual cue

这是 Phase C v3 最重要的新外部几何 cue。当前 `C_dyn` 和 LoGeR pointmap residual 容易把相机自运动 / parallax 误判为 dynamic。Flow-epipolar cue 通过比较 observed flow 和 static scene predicted flow 来区分真实 moving object。

对当前帧 $t$ 的像素 $u$，Pass 1 给出世界点：

$$
X_t(u)=T_{w\leftarrow c,t}P_{cam,t}(u)
$$

投影到相邻帧 $s$：

$$
\tilde u_{t\rightarrow s}=\pi(T_{c_s\leftarrow w}X_t(u))
$$

用光流模型得到 observed correspondence：

$$
u_{obs}=u+F_{t\rightarrow s}^{obs}(u)
$$

定义 residual：

$$
r_{flow}(t,s,u)=
\frac{\|\tilde u_{t\rightarrow s}-u_{obs}\|_2}{\sqrt{H^2+W^2}}
$$

动态 cue：

$$
D^{flow}(t,u)=
\sigma\left(\frac{r_{flow}(t,s,u)-\tau_f}{\sigma_f}\right)
\cdot
(1-C_{occ}(t,u))
\cdot
Conf(t,u)
$$

为了避免光流模型自身错误，加入 forward-backward consistency：

$$
FB(t,u)=
\|F_{t\rightarrow s}^{obs}(u)+F_{s\rightarrow t}^{obs}(u+F_{t\rightarrow s}^{obs}(u))\|_2
$$

最终：

$$
D^{flow\_safe}=D^{flow}\cdot
\mathbf{1}[FB<\tau_{fb}]
$$

如果暂时不能引入 RAFT / GMFlow，可以先实现 **sparse flow proxy**：用现有 keypoint / patch matching 或 LoGeR attention correspondence 估计 observed patch displacement。但如果条件允许，Phase C v3 推荐直接上一个强 optical-flow 模块，因为当前问题已经不是 scalar tuning 能解决的。

---

#### Cue E：semantic-vetoed motion cue

Stage C 的语义结果当前不能主导 controller，但可以做 veto / protection。KITTI prompt 应该使用：

```text
THING:
car, van, truck, bus, trailer, train,
person, pedestrian, rider, cyclist,
bicycle, motorcycle

STRUCTURE:
road, lane marking, sidewalk, building, building facade,
guardrail, fence, barrier, pole, traffic sign, traffic light,
bridge

LOW_VALUE_STUFF:
sky, tree, vegetation, grass, bush
```

语义只用于保护：

$$
D^{sem\_veto}_i=D_i\cdot(1-P^{structure}_i)
$$

其中 $P^{structure}_i=1$ 表示可靠 road / building / sidewalk / guardrail。对车辆不要直接压低，因为 KITTI 中 parked cars 可能是有用 landmark。只有当 `movable semantic` 与 high motion cue 同时成立时，才允许它被视为 dynamic：

$$
D_i^{movable}=D_i\cdot P_i^{movable}\cdot \mathbf 1[D_i>\tau_D]
$$

---

### 4.3 Probe-only dashboard

每个 cue source 都必须输出以下 dashboard：

```text
RGB overlay
D_tok heatmap
D_tok top-k mask
road/building/sky/car qualitative panels
per-frame dynamic mass
per-chunk dynamic mass
D_tok temporal variance
D_tok overlap-frame consistency
D_tok vs C_unc / C_occ correlation
D_tok vs Conf correlation
reference-token protected ratio
```

同时记录数值：

$$
\rho_D(t)=\frac{1}{N}\sum_i\mathbf 1[D_{t,i}>\tau_D]
$$

$$
contrast_D=Q_{90}(D)-Q_{10}(D)
$$

$$
TV_D=\frac{1}{T-1}\sum_t\|D_{t+1}-D_t\|_1
$$

$$
Corr(D,C_{unc}),\quad Corr(D,C_{occ}),\quad Corr(D,Conf)
$$

---

### 4.4 Cue quality 判断标准

一个 cue source 进入 C3-2 的最低条件：

1. dynamic mass 不能过大：

$$
0.03 < \operatorname{Mean}_t\rho_D(t) < 0.25
$$

2. temporal flicker 不能比旧 `C_dyn` 更高：

$$
TV_D \leq 1.2\cdot TV_{C_{dyn}}
$$

3. 不能大面积覆盖结构区域。如果使用 semantic protection dashboard，可靠 road/building/sidewalk/guardrail 上的 dynamic mass 应满足：

$$
\rho_D^{structure}<0.10
$$

4. 不能和 uncertainty 完全同义：

$$
Corr(D,C_{unc})<0.75
$$

5. 不能和 confidence 完全反相关：

$$
Corr(D,1-Conf)<0.75
$$

如果一个 cue 只是低 confidence detector，它不适合作为 Phase C motion cue。

进入 C3-2 的 cue 上限为 3 个，优先级建议：

```text
1. flow_safe
2. gram4d
3. entropy_dispersion
4. old C_dyn reference
```

---

## 5. C3-2：Stateful slice gate

### 5.1 实验目的

C3-2 用来解决 Phase C v2 的核心失败：prefix / 256-frame 结果无法预测 full KITTI 01。它不是旧的 prefix run，而是从 full-sequence no-control run 的真实 memory state 分叉。

C3-2 的输入是 C3-0A 保存的 HMC no-control committed states：

$$
\mathcal H_m=
\{W_m^{TTT},H_m^{SWA},R_m^{ref},\text{history},\text{reset metadata}\}
$$

从多个位置启动 controlled continuation：

```text
slice starts: chunk 0, 5, 10, 15, 20, 25, 30
or frame starts approximately: 0, 160, 320, 480, 640, 800, 960
slice length: 128 or 192 frames
```

每个 slice 都同时跑：

```text
no-control continuation
candidate controlled continuation
```

两者必须从同一个 $\mathcal H_m$ 开始。

---

### 5.2 为什么这个 gate 比 256-frame prefix 更可靠

普通 prefix 只测试序列开头。KITTI 01 的 drift / reset / SWA / TTT memory 状态在中后段已经不同，controller 对早期有效不代表对中后段有效。

Stateful slice gate 测的是：

$$
\text{candidate 在真实 long-memory state 下是否仍然有效}
$$

它可以快速暴露 Phase C v2 中 `F256-02 → full fail` 这类问题。

---

### 5.3 Slice 评价方式

每个 slice 需要记录两类指标。

第一类是 local Sim(3)-aligned segment ATE：

$$
ATE_{slice}^{Sim3}
$$

第二类是 continuation consistency，不允许候选只靠局部 Sim(3) 好看但破坏全局轨迹。具体记录：

$$
\Delta t_{end}=\|t^{cand}_{end}-t^{ref}_{end}\|_2
$$

$$
\Delta R_{end}=\angle(R^{cand}_{end}(R^{ref}_{end})^{-1})
$$

其中 ref 是 no-control continuation。

对于每个 slice：

$$
\Delta ATE_s=ATE_s^{cand}-ATE_s^{nocontrol}
$$

通过标准：

```text
至少 5 / 7 个 slices 的 ΔATE_s < 0
任一 slice 的 ΔATE_s 不能 > +0.15 m
end translation drift 不能系统性大于 no-control
```

如果一个 candidate 只在开头好，在中后段多数 slice 坏，不进入 full run。

---

## 6. C3-3：Single-path read injection gate

### 6.1 实验目的

C3-3 只使用通过 C3-1 / C3-2 的 cue source，分别注入不同 read path。禁止多个 read path 一起开。

这样做是为了回答：

$$
\text{motion cue 应该作用在哪条 LoGeR read path？}
$$

而不是一开始就把 frame / SWA / chunk attention 全打开。

---

### 6.2 Frame attention gate

Frame attention 是每帧内部 spatial feature extraction。C3 v1 的 RFR-100 已经证明它有一定信号，但上限不够。

Phase C v3 只保留两个 frame-attention bias mode：

#### Mode F-key：VGGT4D-style key suppression

$$
B_{qk}^{frame}=\beta_f\log(1-D_k+\epsilon)
$$

它只压 dynamic key，不做 pairwise query-key 矩阵。

#### Mode F-pair：MUT3R-style static-query-to-dynamic-key suppression

$$
B_{qk}^{frame}=\beta_f
\log(1-(1-D_q)D_k+\epsilon)
$$

C3 v2 的 `pair` 是 reference，C3 v3 只在新 cue source 上测试。

Layer schedule 不再大扫，只用：

```text
early: reference schedule
early_quarter: conservative schedule
```

强度：

```text
beta = 0.75, 1.0
```

不再扫 1.25 / 1.5 / 2.0，因为 Phase C v1 已证明过强会坏。

---

### 6.3 SWA previous-key read gate

SWA 是 LoGeR 的 adjacent local memory highway。它可能比 frame attention 更接近跨 chunk drift 的来源。SWA gate 只对 previous chunk keys 生效，不压 current chunk token。

定义 previous-token gate：

$$
P^{SWA}_{prev,k}=1-D^{prev}_k
$$

参考 token 永远保护：

$$
P^{SWA}_{prev,k}=1, \quad k\in P_{ref}
$$

SWA attention bias：

$$
B_k^{SWA}=\beta_{swa}\log(P^{SWA}_{prev,k}+\epsilon)
$$

SWA gate 的前置 dashboard 必须显示：

```text
previous-key dynamic mass
protected previous-key ratio
overlap frame protected ratio
attention probability change on previous keys
```

如果 SWA gate 明显压了 overlap/reference tokens，直接判失败。

SWA gate 的 first-run 强度：

```text
beta_swa = 0.5, 1.0
```

---

### 6.4 Chunk-wise bidirectional attention gate

Chunk-wise bi-attn 是当前 chunk 内跨帧几何推理最强的路径，也最危险。C3 v3 只允许在 cue 质量过关后做极轻量测试。

两个 bias mode：

#### Mode C-key

$$
B_{qk}^{chunk}=\beta_c\log(1-D_k+\epsilon)
$$

#### Mode C-pair

$$
B_{qk}^{chunk}=\beta_c
\log(1-(1-D_q)D_k+\epsilon)
$$

Layer schedule：

```text
early_quarter only
```

强度：

```text
beta_c = 0.25, 0.5
```

不允许上来用 beta=1.0，因为 chunk-wise attention 对当前 pose / pointmap 影响更直接。

---

### 6.5 TTT-apply read gate 暂停作为主线

Phase C v1 已经说明 TTT apply read gate 不安全：小 rho 就让 ATE 大幅恶化，同时 rotation 变好。这说明它很可能改变 scale / translation continuity。Phase C v3 不再把 TTT apply 作为主线，只保留一个 rescue check：

```text
rho = 0.02
min_gate = 0.95
layer_mode = early_quarter
```

通过标准很严格：

$$
ATE < 41.0733\text{ m}
$$

否则完全停止 TTT-apply read gate。

---

## 7. C3-4：Full KITTI 01 acceptance gate

### 7.1 Full run 候选进入条件

一个 candidate 必须同时满足：

1. C3-1 cue quality 通过；
2. C3-2 stateful slices 通过；
3. C3-3 单路径 injection 在 slices 中不破坏 continuation；
4. 不是旧 `C_dyn` family 的简单 beta/protection 变体。

每次最多只推 4 个 full candidates，避免低信息量 full sweep。

---

### 7.2 Full run 评价标准

Phase C v3 的 reference：

$$
ATE_{RFR100}=41.0733\text{ m}
$$

最低通过标准：

$$
ATE < 41.0\text{ m}
$$

Phase C v3 的正式通过标准：

$$
ATE \leq 40.8\text{ m}
$$

强通过标准：

$$
ATE \leq 40.5\text{ m}
$$

如果达到：

$$
ATE \leq 40.0\text{ m}
$$

则允许进入 Phase D read-path + TTT branch0 write combination。

同时要求 RPE 不得恶化明显：

$$
RPE_t^{cand}\leq RPE_t^{RFR100}+0.05\%
$$

$$
RPE_r^{cand}\leq RPE_r^{RFR100}+0.0005
$$

rotation 可以略差，但如果 ATE 改善主要来自某种评估偶然性，RPE / segment continuation 会暴露问题。

---

## 8. Phase C v3 推荐实验矩阵

### 8.1 C3-0 correctness runs

| Run ID | Model | Mode | Purpose | Pass criterion |
|---|---|---|---|---|
| C3-0A | LoGeR | HMC no-control | 新 pipeline 复现 LoGeR | $|ATE-41.7502|<0.15$ |
| C3-0B | LoGeR\* | HMC no-control + SE(3) | 新 pipeline 复现 LoGeR\* | $|ATE-47.9793|<0.15$ |
| C3-0C | LoGeR | identity hooks full | 确认 hooks 不改变输出 | $|ATE-ATE_{nocontrol}|<0.05$ |
| C3-0D | LoGeR | random / inverted cue 64-frame | 确认 hooks 非 identity 时真的生效 | output changes, bias nonzero |

---

### 8.2 C3-1 cue source probe runs

| Run ID | Cue source | Output | Main question |
|---|---|---|---|
| C3-1A | old `C_dyn` | dashboard only | reference |
| C3-1B | Gram-lite simple | dashboard only | Q/K Gram 是否比旧 `C_dyn` 更少误伤结构 |
| C3-1C | VGGT4D-style Gram | dashboard only | shallow/mid/deep 组合是否更稳 |
| C3-1D | entropy / dispersion | dashboard only | attention entropy 是否更像 motion cue |
| C3-1E | flow-epipolar residual | dashboard only | 是否能区分 ego-motion parallax 与 moving object |
| C3-1F | flow + semantic structure veto | dashboard only | 是否能保护 road/building/guardrail |

只允许最多 3 个 cue 进入 C3-2。

---

### 8.3 C3-2 stateful slice runs

每个通过 C3-1 的 cue 先在以下 setting 上做 slice：

| Run ID | Path | Bias | Layer | Strength |
|---|---|---|---|---|
| SLC-F1 | frame | key | early | beta=1.0 |
| SLC-F2 | frame | pair | early | beta=1.0 |
| SLC-SWA1 | SWA | previous-key | SWA layers | beta=0.5 |
| SLC-C1 | chunk bi-attn | key | early_quarter | beta=0.25 |

通过标准：

```text
≥5/7 slices improve
no slice worse by more than +0.15 m
continuation end drift not systematically worse
```

---

### 8.4 C3-3 full candidate runs

只推 C3-2 通过的候选。建议第一批 full candidates 最多 4 个：

| Run ID | Cue | Path | Bias | Strength | Purpose |
|---|---|---|---|---|---|
| FC3-01 | best Gram cue | frame | key | beta=1.0 | 替代旧 RFR-100 的 internal cue |
| FC3-02 | best flow cue | frame | key or pair | beta=0.75/1.0 | 检查 flow motion 是否强于 `C_dyn` |
| FC3-03 | best flow/Gram cue | SWA | prev-key | beta=0.5 | 检查跨 chunk local memory 是否是瓶颈 |
| FC3-04 | best cue | chunk bi-attn | key | beta=0.25 | 检查当前 chunk 跨帧推理是否是瓶颈 |

Full pass：

$$
ATE \leq 40.8\text{ m}
$$

如果没有任何 run 低于 $41.0\text{ m}$，Phase C v3 判定 read-path-only 仍未找到有效信号，不进入 Phase D。

---

## 9. 需要新增的实现与日志

### 9.1 必须新增的 CLI / config

```text
--read_cue_source old_dyn|gram_lite|gram4d|entropy|flow|flow_sem_veto
--flow_model none|raft|gmflow|patch_match
--flow_pair_stride 1|2
--flow_fb_thr
--flow_residual_thr
--gram_layer_groups shallow,middle,deep
--stateful_slice_mode
--stateful_slice_starts
--stateful_slice_len
--save_hmc_states
--load_hmc_state_at_chunk
--read_path frame|swa|chunk|ttt_apply
--chunk_bias_mode key|pair
--swa_bias_mode prev_key
```

---

### 9.2 必须新增的 debug 文件

每个 run 至少保存：

```text
hmc_correctness_summary.json
cue_quality_summary.json
cue_quality_per_chunk.jsonl
hook_effect_summary.jsonl
stateful_slice_summary.csv
attention_delta_summary.jsonl
trajectory_01.txt
kitti_benchmark.log
```

Cue dashboard 保存到：

```text
results/kitti01_hmc_v2/phaseC_v3/cue_dashboards/{run_id}/
```

每个 selected chunk 保存：

```text
rgb_overlay.jpg
D_tok_heatmap.jpg
D_tok_topk.jpg
C_dyn_reference.jpg
C_unc.jpg
C_occ.jpg
confidence.jpg
semantic_protection_overlay.jpg  # if available
```

---

### 9.3 Hook effect summary

每个 hook 必须记录：

```text
num_calls
layer_id
path_type
mean_abs_bias
max_abs_bias
attention_prob_delta_mean
attention_prob_delta_max
protected_token_bias_mean
dynamic_token_bias_mean
reference_token_bias_mean
```

对于 SWA：

```text
prev_key_bias_mean
overlap_key_bias_mean
current_key_bias_mean
protected_prev_key_count
```

如果 protected / reference token 的 bias 不是接近 0，说明 protection 逻辑有 bug。

---

## 10. 决策树

### 10.1 如果 C3-0 失败

停止所有模型实验。优先修：

```text
HMC state commit
reset-block merge
SE(3) alignment
identity hook implementation
SWA state handoff
```

不能用旧 pipeline 结果替代。

---

### 10.2 如果 C3-1 所有新 cue 都失败

说明当前 probe 信号仍然不足。此时不应该继续 Phase C read-path injection，而应该转向：

1. 更强外部 motion module；
2. 更可靠 KITTI semantic / object motion prior；
3. 或轻量学习式 reliability gate。

---

### 10.3 如果 C3-1 有 cue 通过，但 C3-2 slice 失败

说明 cue 看起来合理，但对 long-memory state 不稳定。此时不推 full run，先分析失败 slice：

```text
which chunk starts fail
whether reset boundary fails
whether SWA overlap fails
whether dynamic mass spikes
whether road/building protection fails
```

可能需要 chunk-level reliability gate：

$$
g_m=\mathbf 1[contrast_D>\tau_c]\cdot\mathbf 1[\rho_D<\tau_D]\cdot\mathbf 1[\rho_{structure\_dyn}<\tau_s]
$$

然后：

$$
B_{qk}=g_m\cdot B_{qk}^{raw}
$$

---

### 10.4 如果 C3-2 通过，但 full 仍失败

说明 slice gate 还不够。需要检查 full run 的 failure segment，并扩大 stateful slice starts，特别覆盖：

```text
reset boundary before / after
dynamic-heavy region
long straight low-texture road
end-of-sequence accumulated drift
```

如果再次失败，Phase C read-path-only 路线进入低优先级。

---

### 10.5 如果 full 达到 `<=40.8 m`

Phase C v3 通过。此时才允许进入 Phase D：

```text
best read-path controller + TTT branch0 dynamic MP-01
best read-path controller + TTT branch0 sparse preserve
best read-path controller + semantic structure protection
```

Phase D 的目标不再是 `<41.0 m`，而是：

$$
ATE < 40.0\text{ m}
$$

如果 Phase D 达不到 $40.0\text{ m}$，说明 read-path 与 TTT write 没有形成强互补。

---

## 11. Phase C v3 的预期结果与风险

最理想的情况是 `flow_sem_veto` 或 `gram4d` cue 在 C3-1 / C3-2 都明显优于 old `C_dyn`，并且在 frame 或 SWA read gate 上 full 达到：

$$
ATE \leq 40.8\text{ m}
$$

这说明当前瓶颈确实是 motion cue source，而不是 HMC read-path control 结构。

次理想情况是 cue 在 slice 上有用，但 full 仍略差。这说明需要 chunk-level reliability gate，不应继续只用固定 beta。

最坏情况是所有新 cue 在 probe dashboard 或 stateful slice 上都不稳定。这时应该停止 Phase C read-path-only，承认当前 training-free internal/geometry cue 对 KITTI 01 的 read-path control 上限有限，转向更强的外部 motion / semantic / learned reliability。

---

## 12. Phase C v3 最终成功标准

Phase C v3 的成功不是“比 BL01 好”，也不是“比 LoGeR native 好”。Phase C v3 必须超过 RFR-100，并且跨过明确 gate。

最低继续投入标准：

$$
ATE < 41.0\text{ m}
$$

Phase C 正式通过标准：

$$
ATE \leq 40.8\text{ m}
$$

强通过标准：

$$
ATE \leq 40.5\text{ m}
$$

进入 Phase D 标准：

$$
ATE \leq 40.8\text{ m}
$$

进入主模型候选标准：

$$
ATE < 38.0\text{ m}
$$

如果 Phase C v3 仍然停留在 $41.0\sim41.2\text{ m}$，就不应该继续把 read-path-only 当作主线，而应该把它作为辅助信号，与 TTT branch0 sparse write、semantic protection 或 learned reliability gate 组合时再验证。

---

## 13. 一句话总结

Phase C v3 的核心不是继续调 RFR-100，而是重新验证：

$$
\text{新的 motion cue 是否足够可靠？}
$$

$$
\text{它应该控制 LoGeR 的哪条 read path？}
$$

$$
\text{它在真实 long-memory state 下是否仍然有效？}
$$

只有这三个问题都通过，Phase C 才值得进入 Phase D。否则继续围绕当前 `C_dyn` family 做 beta / top-k / protection sweep，只会重复 Phase C v2 的失败模式。

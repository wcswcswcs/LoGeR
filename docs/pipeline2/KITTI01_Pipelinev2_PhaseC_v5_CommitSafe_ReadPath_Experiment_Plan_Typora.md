# KITTI 01 HMC Phase C v5：Commit-Safe Read-Path 与 True Motion Rescue 实验方案

> 本文档是 Phase C v4 失败后的下一步实验方案。它不再继续围绕 patch-match / reprojection flow proxy 做更多阈值扫参，而是把 Phase C 的问题重新拆成两个更具体的假设：
>
> 1. 当前 read-path cue 可能并不是完全无效，而是 **controlled read path 影响了后续 memory commit**，导致短窗和 slice 有收益、full sequence 变差。
> 2. 当前 flow proxy 的短窗收益来自局部尖锐残差，但它不是 full-sequence safe cue；若继续使用 motion cue，必须引入 **true external flow residual** 或把 flow 只作为 old-dyn regularizer 的 rescue / veto，而不是直接替换 old-dyn。
>
> Phase C v5 的目标不是继续证明“某个 cue 在 64/256 帧上好”，而是验证：**read-path 控制是否可以在不污染 LoGeR hybrid memory 的情况下稳定改善 full KITTI 01**。

---

## 0. 当前结论与失败原因分析

Phase C v4 的执行结果非常有价值，因为它排除了一个很有诱惑力但错误的方向：继续手调 patch-match / reprojection flow proxy。v4 中，`flow_proxy_calib q=0.06` 的 dynamic mass 几乎精确达到目标，mean mass 为 $0.0601$，chunk coverage 为 $1.0000$，但 fragmentation 达到 $0.2363$，远高于 global-safe threshold；`q=0.10/tau=0.10` 的 mass 更安全，fragmentation 仍有 $0.1774$，短窗收益基本消失。256-frame 中，`flow_proxy_calib q=0.06, beta=0.75` 虽然把 ATE 从 RFR reference 的 $3.8666$ m 降到 $3.7706$ m，但它没有通过 full-run cue audit，因此没有资格进入 full candidate。v4 的正确结论是：**proxy flow 有局部信号，但它不是全序列安全的 motion cue**。

更重要的是，v4 暴露了一个新的结构性问题。Phase C 从 v2 到 v4 连续出现同一类 false positive：某个 cue 在 64-frame、256-frame 或 stateful slice 上有明显收益，但一旦放到 full KITTI 01，就无法超过 RFR-100 reference。v3 中 `flow_sem_veto` 通过了 $6/7$ stateful slice，却 full ATE 到 $41.4333$ m；v4 中更严格的 cue audit 直接拦下了 flow proxy full candidate。这个现象说明，**局部 continuation win 不等价于 full-sequence trajectory win**。

现在的 read-path best 仍然是 RFR-100：

$$
ATE_{RFR100}=41.0733\text{ m}
$$

它没有过 $<41.0$ m 或 $\le 40.8$ m gate，但它比 A3 HMC native 的 $41.7502$ m、旧 TTT branch0 write 的 $41.3665$ m 都更强。因此，RFR-100 仍然是 read-path control 的唯一有效 reference。Phase C v5 不应该丢掉 RFR-100，而应该回答：

> RFR-100 为什么只能到 $41.0733$ m？是因为 read cue 不够强，还是因为 read-path 控制同时改变了后续 memory 写入，带来了长期副作用？

这是 Phase C v5 的核心问题。

---

## 1. Phase C v5 的核心假设

Phase C v5 围绕四个假设设计实验。

### 假设 H1：RFR-100 是 dense read regularizer，不是动态物体 mask

`old_dyn` 的 full-run probe mass 约为 $0.4650$，它远高于 v4 中为 object-style dynamic cue 设定的 mass band。可是它仍然是当前最好的 read-path full candidate。这说明 `old_dyn` 很可能不是在精确检测 moving object，而是在给 frame attention 提供一种 dense、smooth、sequence-stable 的 regularization field。

因此，v5 不再强迫所有 read cue 都满足 object-dynamic style 的 low mass。我们把 cue 分成两类：

$$
D_{reg}: \text{dense read regularizer}
$$

$$
D_{motion}: \text{sparse true motion / dynamic object cue}
$$

`old_dyn` 属于 $D_{reg}$，可以 dense；flow residual 属于 $D_{motion}$，必须满足 motion cue quality gate。两者不能混用同一个质量标准。

### 假设 H2：当前 full failure 可能来自 controlled memory commit，而不只是 controlled output

LoGeR 的 block 顺序是：

```text
frame attention → SWA → TTT apply/update → chunk-wise bidirectional attention
```

Phase C 的 frame-attention control 虽然被称为 read-path control，但它发生在 TTT update 之前。如果 Pass 2 用 controlled tokens 继续执行 TTT update 并提交 memory，那么 read-path control 不只是改变当前 chunk output，也可能改变：

$$
W_m \rightarrow W_{m+1}
$$

以及 SWA/local state 的后续传播。也就是说，一个 cue 可以在当前 chunk 改善 pose，却把 altered representation 写入后续 hybrid memory，从而在 full sequence 中产生长期副作用。

因此，Phase C v5 必须引入 **commit-mode isolation**：

```text
controlled-output + controlled-memory commit
controlled-output + native/probe memory commit
```

二者的差异可以判断 read-path signal 本身是否有价值，以及 failure 是否来自 memory contamination。

### 假设 H3：flow proxy 不应再作为 full candidate，除非替换成 true external flow residual

v4 已经证明，patch-match / Stage-B reprojection proxy 经校准后要么 fragmented，要么失去短窗收益。继续扫 `q/tau/lambda` 只会重复 v3/v4 的 false positive。

因此，Phase C v5 中：

```text
patch_match / reprojection flow proxy 只能作为 negative control 或 rescue ablation
不能作为新的 full candidate 主 cue
```

若要继续使用 optical-flow style motion cue，必须实现 true RAFT / GMFlow / 其他可靠 external optical flow，并通过 full-run probe-only quality gate。

### 假设 H4：candidate selection 应按 bias effect 和 memory side effect，而不是只按 cue mass

v4 已经说明，dynamic mass 本身不足以判断安全性。真正注入模型的是 attention logit bias：

$$
B_{qk}=\beta \log(1-(1-D_q)D_k+\epsilon)
$$

或 key-only bias：

$$
B_k=\beta \log(1-D_k+\epsilon)
$$

因此，Phase C v5 要记录并约束：

$$
E_B=\operatorname{Mean}_{hooks}(|B|)
$$

以及 reference / anchor tokens 被施加的 bias energy。一个 cue 即使 mass 合适，如果它把大量 bias 加在 road/building/overlap/reference tokens 上，也不应该进入 full run。

---

## 2. 必须先做的新 HMC pipeline 正确性验证

Phase C v5 继续要求所有结果来自 **新 HMC two-pass pipeline**。正式 baseline 和 correctness gate 不能通过旧 `geometry_eval_mode`、旧 `run_pipeline_abc.py`、旧 TTT-only controller 或任何绕过 HMC hooks 的捷径生成。

### 2.1 LoGeR no-control HMC full reproduction

用新 HMC pipeline 运行 LoGeR no-control full KITTI 01。这里的 no-control 不是旧 native 路径，而是完整 two-pass HMC：

```text
Pass 1: probe
Pass 2: no-control / identity-control
Commit: HMC native commit
```

要求：

$$
|ATE_{HMC\ no\ control}^{LoGeR}-41.7502| < 0.15\text{ m}
$$

同时要求：

```text
probe_no_commit_hash_equal_all = true
state_double_write_safe_all = true
pass1/pass2 pose translation diff = 0 or numerical epsilon
all enabled hooks report zero bias under identity mode
```

如果这个 gate 不通过，Phase C v5 停止，不能进入任何 model experiment。

### 2.2 LoGeR* no-control HMC full reproduction

LoGeR* 必须也在新 HMC pipeline 中复现。设置应使用 LoGeR* 的 SE(3) overlap alignment 配置：

```text
--se3
--chunk_size 64
--window_size 64
--chunk_overlap 3
--overlap_size 3
--reset_every 5
```

要求：

$$
|ATE_{HMC\ no\ control}^{LoGeR*}-47.9793| < 0.20\text{ m}
$$

LoGeR* 的复现不是为了调参，而是为了确认新 HMC pipeline 没有破坏 LoGeR 原始 overlap-based feedforward alignment / reset behavior。如果 LoGeR 可以复现但 LoGeR* 不能复现，说明 HMC state / reset / alignment 逻辑仍有问题。

### 2.3 RFR-100 reference reproduction

在 HMC Phase C v5 代码状态下，复现当前 read-path reference：

```text
read_cue_source = old_dyn
read_path = frame
frame_bias_mode = pair
read_layer_mode = early
beta = 1.0
commit_mode = controlled_memory
```

要求：

$$
|ATE_{RFR100}^{rerun}-41.0733| < 0.15\text{ m}
$$

如果 RFR-100 reference 不能复现，说明 v5 代码状态与之前 Phase C 不可比，必须先修工程。

---

## 3. C5-1：Commit-Mode Isolation Gate

### 3.1 目的

这个 gate 是 v5 最重要的新实验。它要回答：

> RFR-100 没有过 $<41.0$ m，是因为 read cue 本身上限低，还是因为 controlled read path 改写了后续 hybrid memory，造成长期副作用？

### 3.2 新增 commit mode

建议在 HMC controlled pass 中增加：

```text
--hmc_commit_mode controlled
--hmc_commit_mode probe_native
--hmc_commit_mode split_ttt_native
```

三种模式定义如下。

#### controlled

这是当前默认逻辑。Pass 2 controlled forward 产生的 geometry 和 memory 都被提交：

$$
Y_m=Y_m^{ctrl}, \qquad \mathcal H_{m+1}=\mathcal H_{m+1}^{ctrl}
$$

#### probe_native

输出使用 Pass 2 controlled geometry，但 memory 使用 Pass 1 native provisional state：

$$
Y_m=Y_m^{ctrl}, \qquad \mathcal H_{m+1}=\mathcal H_{m+1}^{probe}
$$

这表示：read-path control 只修当前 chunk output，不让 altered controlled tokens 写入未来 memory。

#### split_ttt_native

输出使用 controlled geometry；SWA/local state 可以按 controlled 或 native 另行配置，但 TTT fast weights 使用 probe/native commit：

$$
W_{m+1}^{TTT}=W_{m+1}^{probe}
$$

这个模式用于判断长期副作用主要来自 TTT fast weights 还是其他 local memory。

### 3.3 实验矩阵

先只对 RFR-100 做 full KITTI 01，不引入新 cue。因为 RFR-100 是唯一已知有效的 read-path signal。

| Run | Cue | Bias | Layer | Beta | Commit mode | 目的 |
|---|---|---|---|---:|---|---|
| CM-00 | none | none | none | 0 | HMC no-control | reproduction reference |
| CM-01 | old_dyn | pair | early | 1.0 | controlled | reproduce RFR-100 |
| CM-02 | old_dyn | pair | early | 1.0 | probe_native | isolate output-only read control |
| CM-03 | old_dyn | pair | early | 1.0 | split_ttt_native | isolate TTT commit side effect |

### 3.4 判断标准

如果 CM-02 明显优于 CM-01，例如：

$$
ATE_{CM02} < 41.0\text{ m}
$$

或者更强：

$$
ATE_{CM02} \le 40.8\text{ m}
$$

则说明 read-path cue 是有效的，但 controlled memory commit 有副作用。Phase C 可以通过，但通过的是一种更准确的策略：

```text
read-path controlled output + native/probe memory commit
```

此时下一步应该进入 Phase D，组合 read-path output control 与独立的 TTT branch0 write controller，而不是让 read-path modified tokens 自动写入 TTT memory。

如果 CM-01、CM-02、CM-03 都停在 $41.0$ m 以上，说明 RFR-100 的瓶颈主要在 output-level read cue 自身，不是 memory commit side effect。此时不能继续围绕 RFR-100 做小参数扫参。

如果 CM-03 优于 CM-01，但 CM-02 和 CM-03 差距不大，说明主要副作用来自 TTT fast weights。此时 HMC 需要显式支持：

```text
read path controlled, TTT update native or separately controlled
```

如果 CM-02 反而比 CM-01 差，说明 RFR-100 需要 controlled memory propagation 才能发挥作用；此时应放弃 probe-native commit，把重点转向 cue source / bias scheduling。

---

## 4. C5-2：Bias-Energy Normalized RFR Gate

### 4.1 目的

RFR-100 的 beta sweep 显示 beta=1.0 最好，beta=1.25/1.5 开始变差，beta=2.0 明显过强。这说明问题不是“控制强度不够”，而是不同 chunk 的 actual bias effect 不稳定。

Phase C v5 不再只固定 $\beta$，而是按实际 attention bias energy 做归一化。定义候选 cue $D_m$ 在 chunk $m$ 的 raw bias：

$$
B_m(\beta=1)=\log(1-(1-D_q)D_k+\epsilon)
$$

其 bias energy：

$$
E_m=\operatorname{Mean}(|B_m|)
$$

以 RFR-100 full probe 中的 median energy 作为 reference：

$$
E_{ref}=\operatorname{Median}_m(E_m^{RFR100})
$$

则每个 chunk 的 beta 使用：

$$
\beta_m=\operatorname{clip}\left(\beta_0\frac{E_{ref}}{E_m+\epsilon},\beta_{min},\beta_{max}\right)
$$

默认：

$$
\beta_0=1.0,\quad \beta_{min}=0.5,\quad \beta_{max}=1.5
$$

这个策略的目标是让每个 chunk 的 perturbation budget 接近 reference，而不是让某些 chunk 过强、某些 chunk 过弱。

### 4.2 实验矩阵

| Run | Cue | Bias | Layer | Beta policy | Commit mode | 目的 |
|---|---|---|---|---|---|---|
| BE-01 | old_dyn | pair | early | fixed beta=1.0 | controlled | RFR-100 reference |
| BE-02 | old_dyn | pair | early | bias-energy normalized | controlled | test energy stabilization |
| BE-03 | old_dyn | pair | early | bias-energy normalized | probe_native | combine with commit isolation |
| BE-04 | old_dyn | pair | early_quarter | bias-energy normalized | controlled | reduce layer perturbation |

### 4.3 判断标准

256-frame 只作为 sanity，不再作为 promotion gate。BE candidates 进入 full 的条件是 full-probe bias dashboard 满足：

```text
reference_bias_energy_fraction < 0.02
anchor_bias_energy_fraction < 0.08
chunk_boundary_bias_jump_p90 < reference + 20%
beta_clip_fraction < 0.20
```

Full KITTI 01 判断：

$$
ATE < 41.0\text{ m}
$$

才算 Phase C debug pass；

$$
ATE \le 40.8\text{ m}
$$

才算 Phase C 正式 pass；

$$
ATE \le 40.5\text{ m}
$$

才算 strong pass。

如果 BE-02/03 只把 ATE 从 $41.0733$ m 改到 $41.0$ m 附近但不过 $40.8$ m，则不继续做 BE v2 扫参。

---

## 5. C5-3：True External Flow Residual as Rescue, Not Replacement

### 5.1 目的

v4 的 flow proxy 失败，不代表 optical flow residual 方向完全错误；它只说明 patch-match / reprojection proxy 不够稳定。Phase C v5 若继续使用 flow，必须实现 true external optical flow，并且 flow 只作为 old_dyn 的 rescue / veto，而不是直接替换 RFR-100。

### 5.2 true flow residual 定义

对相邻帧或 stride-$s$ 帧对 $(t,s)$，用 RAFT / GMFlow 得到 observed flow：

$$
F^{obs}_{t\rightarrow s}(u)
$$

用 LoGeR pointmap 和 pose 得到 rigid flow。当前帧像素 $u$ 的世界点为：

$$
X_t(u)=T_{w\leftarrow c,t}P_t(u)
$$

投影到支持帧：

$$
\tilde u_{t\rightarrow s}=\pi(T_{c\leftarrow w,s}X_t(u))
$$

刚体流为：

$$
F^{rigid}_{t\rightarrow s}(u)=\tilde u_{t\rightarrow s}-u
$$

flow residual：

$$
r_{flow}(u)=\frac{\|F^{obs}_{t\rightarrow s}(u)-F^{rigid}_{t\rightarrow s}(u)\|_2}{\tau_0+\|F^{obs}_{t\rightarrow s}(u)\|_2+\|F^{rigid}_{t\rightarrow s}(u)\|_2}
$$

有效性 mask：

$$
V_{flow}=\mathbf 1[FB(u)<\tau_{fb}]\cdot \mathbf 1[Conf(u)>\tau_c]\cdot \mathbf 1[\text{not depth edge}]
$$

最终 motion cue：

$$
D_{flow}=\operatorname{RobustNorm}(r_{flow}\cdot V_{flow})
$$

它必须经过 spatial closing / median smoothing / temporal hysteresis，不能直接用 speckle map。

### 5.3 flow 只用于 rescue / veto

我们从 RFR-100 的 dense regularizer $D_{old}$ 出发。true flow 用来做两件事。

#### Static rescue

如果 old_dyn 高，但 true flow 低，且 anchor/confidence 高，则说明 old_dyn 可能误伤了静态结构：

$$
S_{static}=\mathbf 1[D_{old}>\tau_o]\cdot \mathbf 1[D_{flow}<\tau_f]\cdot \mathbf 1[C_{anchor}>\tau_a]\cdot \mathbf 1[Conf>\tau_c]
$$

然后降低这些区域的 read suppression：

$$
D_{ctrl}=D_{old}(1-\lambda_s S_{static})
$$

#### Motion reinforcement

如果 true flow 高并且 flow validity 高，则增强这些区域：

$$
M_{motion}=\operatorname{Smooth}(\mathbf 1[D_{flow}>\tau_f]\cdot V_{flow})
$$

$$
D_{ctrl}=\operatorname{clip}(D_{old}(1-\lambda_s S_{static})+\lambda_m M_{motion},0,1)
$$

默认从保守参数开始：

$$
\lambda_s=0.25,\quad \lambda_m=0.10
$$

### 5.4 true flow probe gate

true flow cue 作为 $D_{motion}$ 必须满足：

```text
chunk coverage >= 0.95
mean motion mass in [0.03, 0.25]
fragmentation <= 0.08 after smoothing
anchor collision <= 0.06
abs Corr(D_flow, Conf) <= 0.25
valid flow ratio >= 0.70
```

如果 true flow 本身不满足这些 gate，则不能进入 full run；最多用于 visualization 和 failure analysis。

### 5.5 实验矩阵

| Run | Cue | Bias | Layer | Beta policy | Commit mode | 目的 |
|---|---|---|---|---|---|---|
| TF-00 | true_flow standalone | key | early | energy-normalized | probe_native | diagnostic only |
| TF-01 | old_dyn + true_flow static rescue | pair | early | energy-normalized | controlled | reduce false old_dyn suppression |
| TF-02 | old_dyn + true_flow static rescue | pair | early | energy-normalized | probe_native | rescue + no memory side effect |
| TF-03 | old_dyn + true_flow rescue + motion reinforce | pair | early | energy-normalized | probe_native | full rescue/motion hybrid |

TF-00 不作为主 candidate，除非它 full-run cue audit 非常干净，并且 256/global slice 明显优于 RFR-100。

---

## 6. C5-4：Memory Side-Effect Dashboard

Phase C v5 必须为每个 non-identity read-path run 记录 memory side effect。否则无法判断 full failure 是 output failure 还是 memory propagation failure。

对每个 chunk $m$，记录：

$$
\Delta W_m^{(r)}=\frac{\|W_{m+1,ctrl}^{(r)}-W_{m+1,probe}^{(r)}\|_F}{\|W_{m+1,probe}^{(r)}\|_F+\epsilon}
$$

以及：

```text
TTT branch0/1/2 fast-weight diff norm
history KV diff norm if applicable
SWA state diff norm if available
chunk boundary pose delta
bias energy on reference/overlap tokens
bias energy on high-anchor tokens
bias energy p90 per chunk
beta clip fraction
commit mode
```

### 判断标准

如果某个 read-path candidate 的 current-chunk ATE / slice ATE 有改善，但：

```text
TTT memory diff p90 is large
chunk boundary pose delta p90 is large
reference bias energy is non-negligible
```

那么它不能直接进入 full controlled-memory run。它必须先以 `probe_native` 或 `split_ttt_native` commit mode 验证。

---

## 7. C5-5：Global Continuity Slice Gate

v3 的 stateful slice gate 已经被证明过于宽松。因此 v5 的 slice gate 不再只看 local Sim(3) segment ATE。

每个 candidate 在 full 前必须通过以下 global-continuity slice test：

```text
slice starts: chunks 0, 5, 10, 15, 20, 25, 30
slice length: 128 frames
load state: from no-control full HMC saved state
commit mode: same as candidate
```

对每个 slice 记录四个指标：

1. local Sim(3) ATE delta；
2. global-fixed ATE delta；
3. endpoint drift delta；
4. path-length ratio delta。

定义：

$$
\Delta ATE_s^{global}=ATE_s^{global}(candidate)-ATE_s^{global}(reference)
$$

$$
\Delta d_{end}=\|p_{end}^{cand}-p_{end}^{gt}\|_2-\|p_{end}^{ref}-p_{end}^{gt}\|_2
$$

$$
\Delta r_{path}=\left|\frac{L_{path}^{cand}}{L_{path}^{gt}}-1\right|-\left|\frac{L_{path}^{ref}}{L_{path}^{gt}}-1\right|
$$

通过标准：

```text
local Sim(3) improved in >= 5/7 slices
global-fixed ATE improved or neutral in >= 5/7 slices
no slice has global-fixed delta > +0.10 m
endpoint drift delta mean <= 0
path-length ratio delta mean <= 0
```

如果 local Sim(3) 改善但 global-fixed ATE 或 endpoint drift 变差，说明这个 candidate 是局部对齐假阳性，不能进入 full run。

---

## 8. Full KITTI 01 candidate policy

Phase C v5 最多允许三个 full candidate。超过三个还没过 gate，就停止 Phase C read-path-only tuning。

推荐 full candidate 顺序如下。

### FC5-01：RFR-100 with probe-native commit

这是最重要的实验。它直接检验 commit-mode isolation：

```text
cue = old_dyn
bias = pair
layer = early
beta = 1.0
commit = probe_native
```

如果它过 $<41.0$ 或 $\le 40.8$，说明 RFR read signal 本身有效，之前瓶颈可能来自 controlled memory commit。

### FC5-02：RFR-100 with bias-energy normalization

```text
cue = old_dyn
bias = pair
layer = early
beta = energy-normalized
commit = controlled or probe_native, based on CM result
```

如果 commit-mode isolation 没有明显收益，但 bias-energy normalization 有收益，则说明 RFR-100 的主要问题是 chunk-level perturbation 不均衡。

### FC5-03：old_dyn + true_flow static rescue

只有 true flow cue 通过 probe quality gate 时才运行。

```text
cue = old_dyn rescued by true_flow
bias = pair
layer = early
beta = energy-normalized
commit = best commit mode from CM gate
```

如果 true flow 没有实现或没有通过 gate，FC5-03 取消，不用 proxy flow 代替。

---

## 9. Phase C v5 的成功标准和停止条件

Phase C v5 采用更清晰的分级标准。

### Debug pass

$$
ATE < 41.0\text{ m}
$$

这表示 read-path-only 终于超过 RFR-100 的心理门槛，但仍不算主模型成功。

### Formal Phase C pass

$$
ATE \le 40.8\text{ m}
$$

这是进入 Phase D 组合实验的最低标准。

### Strong Phase C pass

$$
ATE \le 40.5\text{ m}
$$

如果达到这个标准，read-path controller 可以作为 HMC 主组件之一进入 Phase D。

### Stop condition

如果下面任一情况成立，则停止 Phase C read-path-only：

```text
FC5-01/02/03 全部 >= 41.0 m
或 probe-native commit 没有改善 RFR-100
或 true flow cue 不能通过 full-run probe gate
或 bias-energy normalization 只带来 <0.05 m 改善
```

停止后不再写 Phase C v6 继续扫 read cue。下一步应转向：

```text
Phase D: read-path reference + TTT branch0 write combination
或 learned reliability gate
或 explicit semantic/flow supervised motion module
```

---

## 10. 预期结果与决策树

Phase C v5 最希望看到的是：

$$
ATE_{RFR100, probe\_native} < ATE_{RFR100, controlled}
$$

如果成立，说明 read-path control 的当前输出收益被 memory side effect 抵消。此时 HMC pipeline 应明确拆成：

```text
current output control: read path controlled
future memory control: separately decided by TTT/SWA memory controller
```

这会直接改变 Phase D 的设计。

如果 probe-native commit 没有收益，而 bias-energy normalization 有收益，说明 read cue 本身可用，但控制强度在不同 chunk 间不稳定。Phase D 可以沿用 energy-normalized read path。

如果 true flow rescue 有收益，说明 old_dyn 的 dense regularizer 里确实包含误伤静态结构的问题，true flow 可以作为 correction signal。

如果三者都没有收益，结论应当非常明确：

> Phase C read-path-only has reached its ceiling on KITTI 01. The next improvement cannot come from finer read cue hand-tuning. We should stop Phase C and move to memory-write combination or learned reliability.

---

## 11. 本阶段不再做的事情

Phase C v5 明确不做以下事情：

1. 不继续用 patch-match / reprojection flow proxy 做 full candidate。
2. 不继续围绕 `q/tau/lambda` 对 flow proxy 扫参。
3. 不继续用 64-frame 或 256-frame ATE 作为 promotion 主标准。
4. 不把 local Sim(3) slice improvement 当作 full-sequence safety 证据。
5. 不在没有 commit-mode isolation 的情况下组合 Phase D。
6. 不把 $41.0$ m 附近的微小改善称为主模型成功。

---

## 12. 最终总结

Phase C v5 的关键变化是：不再把问题理解成“还没找到更好的 dynamic cue”，而是先验证 **read-path control 的长期 memory side effect**。RFR-100 已经证明 old_dyn frame-attention early control 有真实信号，但它卡在 $41.0733$ m。v5 的第一优先级是通过 commit-mode isolation 判断这个上限是否来自 controlled memory commit。

如果 `probe_native` commit 能让 RFR-100 过 $<41.0$ 或 $\le 40.8$，Phase C 仍然有希望，并且下一步应转向 read-output control 与 memory-write control 的解耦组合。如果它不能改善，则 Phase C read-path-only 基本到顶，继续手调 cue 不值得。

因此，Phase C v5 是最后一个 read-path-only refinement 阶段。它要么给出一个可进入 Phase D 的 clean read-path component，要么明确宣告 Phase C 结束。

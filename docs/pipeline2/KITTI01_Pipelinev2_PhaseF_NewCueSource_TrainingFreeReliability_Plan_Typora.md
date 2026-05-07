# KITTI 01 / HMC Phase F：New Cue Source and Training-Free Reliability Estimation v2

> 版本：v2  
> 目标序列：KITTI Odometry 01  
> 当前强基线：D5 / E5 commit-safe HMC family  
> 方法约束：**training-free**。不得训练 LoGeR，不得训练 reliability predictor，不得用 KITTI GT 拟合 gate / threshold / routing 参数。所有 cue、reliability estimate、routing、read bias 和 write policy 都必须来自冻结模型输出、冻结外部模块、几何一致性、attention / QK 统计、probe-cache 统计和确定性规则。

---

## 0. v2 相比 v1 的关键更新

Phase F v1 已经把方向从 “Reliability Learning” 修正为 “Training-Free Reliability Estimation”。但是 v1 还有一个容易误解的地方：虽然它写了 true flow、Gram/QK、semantic、TTT residual 等模块，但没有足够明确地区分 **真正的新 cue source** 和 **old_dyn 的 modulation / protection**。Phase E 的结果已经告诉我们，如果继续围绕 `old_dyn × something` 做手工修补，很可能只会在 $39.48\sim39.55\,m$ 附近抖动，而不是突破当前 plateau。

因此 v2 做了四个关键调整。

第一，Phase F 的第一组实验不再直接跳到 true flow 或 Gram，而是先做 **F1：old_dyn decomposition and implicit-cue isolation**。之前有效的 `old_dyn` 并不是纯显式几何动态 cue，而是 Stage-B `C_dyn` family，其中包含显式几何 residual 分支和你手工设计的隐式 attention / key-cosine 动态分支。D5 / E5 的 best 都用的是 fused old_dyn，但我们还没有回答：真正起作用的是显式分支、隐式分支，还是 fusion 形式。v2 把这件事放到最前面。

第二，v2 明确把 cue source 分成三条 read-cue 线：

```text
Global / chunk-wise / cross-frame attention cue
Per-frame attention cue
True external optical-flow / epipolar residual cue
```

其中 global attention cue 对应 VGGT4D 式 Gram/QK mining；per-frame attention cue 对应 MUT3R 式 self-attention / key-cosine / entropy / dispersion mining。它们都必须能独立生成 `D_new`，不能只作为 old_dyn 的乘法调制项。

第三，v2 明确规定什么才算 “new cue”。一个 cue 必须满足：

$$
D_{new}=f(\text{source})
$$

且 $D_{new}$ 能在 read-path-only 或 probe_ttt_write 协议下单独运行，才算新 read cue。下面这些不算真正的新 cue：

```text
old_dyn × Gram agreement
old_dyn + overlap protection
old_dyn + anchor protection
old_dyn + flow rescue
old_dyn + semantic structure floor
```

它们只能算 modulation / protection / rescue。

第四，v2 仍然保持 training-free。这里的 reliability estimator 不是学习模型，而是 deterministic probe-only audit：mass、coverage、fragmentation、anchor collision、confidence correlation、temporal stability、old_dyn overlap、worst-chunk response 等统计量只用于判断 cue 是否可靠，不训练任何 gate。

---

## 1. 之前实验已经给出的核心结论

### 1.1 HMC pipeline 的正确性已经建立，但 Phase F 必须继续守住 correctness gate

早期 Gate 0 / Gate 1 已经证明当前工程链路能复现 LoGeR 和 LoGeR* 在 KITTI 01 上的指标：LoGeR native 是 $41.7502\,m$，LoGeR* native 是 $47.9793\,m$，与论文表格数值非常接近。unity replay 也与 native 保持足够接近，说明基础 TTT replay 和 reset-block-aware merge 是可信的。

后续 HMC v2 又进一步验证了 two-pass no-control、identity hook、controlled / probe_native / split_ttt_native commit mode 的正确性。Phase C v5 中 no-control LoGeR / LoGeR* 在新 HMC pipeline 上精确复现，identity hooks 的 frame attention、SWA read、TTT apply、chunk attention 都保持 zero-bias，不改变输出。

Phase F 会新增更多 cue exporter、true-flow cache、Q/K/Gram statistics、probe-only audit 和 read/write pair selector。任何一个新组件都有可能无意中改变图像顺序、chunk state、TTT commit 或 hook behavior。因此 Phase F 仍然必须从 correctness gate 开始。

Phase F 的第一原则是：

$$
\text{A new cue source is invalid unless HMC no-control still reproduces LoGeR / LoGeR*.}
$$

---

### 1.2 TTT-only write control 给出了弱信号，但不是主模型

早期 GSL-WC 证明，单纯做 TTT write control 只能得到非常小的收益。geometry-first suppressive gating 最好只到 $41.5765\,m$，eta-mean-preserving 后也没有突破。score decomposition 发现 `-rank(C_dyn)` 是最有用的写入 score，branch/layer selective 又发现它只在 TTT branch 0 上有效，得到 BL01：

$$
ATE_{BL01}=41.3665\,m.
$$

这个结果说明 TTT write 并非无效，但它不能单独支撑主模型。更重要的是，branch 1 / branch 2 或 all-branch 写入不安全，TTT 三个 branch 必须区别对待。

---

### 1.3 Read-path correction 是第一次带来实质收益的机制

Phase C 发现 `old_dyn` 用在 early per-frame attention 的 pair bias 上，可以改善当前 chunk output。这个配置被记作 RFR-100：

```text
read cue      : old_dyn = Stage-B C_dyn family
read path     : per-frame attention
bias mode     : pair
layer schedule: early layers
beta          : 1.0
```

它不是写入控制，而是在当前 chunk 的 forward 里调整 attention logits，让静态 token 少 attend 到动态或不稳定 token。pair bias 的形式是：

$$
B_{ij}=\beta\log\left(1-(1-D_i)D_j+\epsilon\right),
$$

其中 $D_i$ 是 query token 的 dynamic score，$D_j$ 是 key token 的 dynamic score。这个公式主要压制：

$$
\text{static query}\rightarrow \text{dynamic key}.
$$

RFR-100 controlled commit 从 native 的 $41.7502\,m$ 改善到 $41.0733\,m$，说明 read-path signal 真实有效。但真正的突破来自 commit isolation。

---

### 1.4 Commit isolation 是 Phase C/D 的核心发现

Phase C v5 证明，同样使用 RFR-100 controlled output，如果直接提交 Pass-2 controlled forward 产生的 TTT state，full KITTI 01 是：

$$
ATE_{controlled}=41.0733\,m.
$$

如果当前输出仍然使用 Pass-2 controlled geometry，但下一 chunk 的 TTT memory 改用 Pass-1 native/probe state，则得到：

$$
ATE_{probe\_native}=39.7820\,m.
$$

这说明：

$$
\boxed{\text{read-path correction is useful, but controlled TTT commit has long-horizon side effects.}}
$$

原因来自 LoGeR 的 block 顺序：

```text
per-frame attention → SWA → TTT apply/update → chunk-wise bidirectional attention
```

Pass-2 中被 read-control 改过的 token 表示会进入后续 TTT update。如果直接提交 Pass-2 的 $W_{m+1}^{controlled}$，就把 read perturbation 写进未来 memory。Phase C v5 说明这会显著伤害 full-sequence trajectory。

Phase D v5 进一步证明，最佳 commit 模式是 `probe_ttt_write`：当前输出使用 Pass-2 controlled geometry，未来 memory 使用 Pass-1 probe cache，再在 probe cache 上做 branch0 controlled write。

D5 最好结果是：

$$
ATE_{D5-07}=39.4903\,m.
$$

对应配置：

```text
read : old_dyn, frame-attention pair, early layers, beta=1.25
write: probe-cache BL01 branch0 dynamic MP-01
commit: probe_ttt_write
```

---

### 1.5 Phase E 证明 hand-rule co-design 已经进入平台期

Phase E 试图保住 D5 的 ATE gain，同时修复 rotation / final endpoint 损伤。它测试了 read-side protection、Gram modulation、residual-reliability write、sparse exact-preserve write 和小规模 read/write pair search。

最好数值是：

$$
ATE_{E5-resrel-b125}=39.4881\,m.
$$

这只比 D5-07 的 $39.4903\,m$ 好 $0.0022\,m$，属于 deterministic tiny best，而不是模型意义上的突破。最 balanced 的候选是 `E5-anchor-sparse95`：

$$
ATE=39.5111\,m,\quad Rot=9.6396^\circ,\quad FinalErr=5.967\,m.
$$

Phase E 的负结果更加重要。Gram-lite / Gram4D modulation 没有赢；overlap / combined reference protection 能改善 rotation 但牺牲 ATE；sparse write 更稳定但不赢 ATE；true-flow 仍然 blocked，之前 flow 结果只是 patch-match / Stage-B proxy，不是 RAFT / GMFlow true flow。

因此 Phase E 的结论是：

$$
\boxed{\text{old\_dyn read + probe-cache branch0 write family has plateaued around }39.49\,m.}
$$

Phase F 必须从 “old_dyn hand-rule protection” 转向真正的新 cue source discovery。

---

## 2. Phase F 的实验目的

Phase F 的目标不是再证明 read-path control 有效，也不是再证明 commit isolation 有效。这些已经成立。Phase F 要回答的是：

$$
\textbf{Can we find a better training-free cue source than old\_dyn, and can we decide when to trust it without learning?}
$$

具体有五个问题。

第一，当前最有效的 `old_dyn` 到底来自哪里？它是显式几何 residual 有效，还是你手工设计的 implicit dyn cue 有效，还是二者 fusion 才有效？如果不先回答这个问题，Phase F 所谓 “new cue source” 会缺少基准。

第二，global / chunk-wise / cross-frame attention 里是否存在比 old_dyn 更好的 motion cue？这对应 VGGT4D 的思路：不要只看普通 QK attention，而是用 QQ / KK / QK Gram statistics 在 shallow / middle / deep layer 上构造独立 dynamic saliency。

第三，per-frame attention 里是否存在比 old_dyn 更好的 motion / instability cue？这对应 MUT3R 的思路：动态区域在 self-attention 中表现出 attention dispersion、temporal instability、layer disagreement；但 Phase C v2 的粗糙 `internal_attn` 失败了，所以 Phase F 必须重新设计 per-frame cue，而不是复用粗糙 mean attention。

第四，true optical flow / epipolar residual 是否能提供 old_dyn 缺失的外部 motion evidence？之前 proxy flow 的负结果不能代表 true flow。

第五，write cue 是否应该从 update-needed 出发，而不是沿用 read-side dynamic suppression？Phase D/E 说明 branch0 write 有用，但 `-rank(C_dyn)` 仍然很粗。Phase F 要继续探索 probe-cache TTT residual、alignment confidence、sparse exact-preserve 等 training-free write cues。

---

## 3. Phase F 的 cue taxonomy

Phase F 必须先把 cue 角色说清楚。一个 cue 可能适合 read path，但不适合 write path；也可能只适合作为 reference protection 或 reliability veto。v2 使用下面的 taxonomy。

### 3.1 Read cue：控制当前 chunk 的 forward output

Read cue 记为：

$$
D_{read}\in[0,1]^{T\times H_{tok}\times W_{tok}}.
$$

它作用在 Pass-2 controlled forward 的 attention bias 上，主要目标是改善当前 chunk 的 pose / pointmap / confidence。当前已经验证有效的 read path 是 early per-frame attention pair bias：

$$
A_{ij}^{ctrl}=\operatorname{Softmax}\left(\frac{Q_iK_j^\top}{\sqrt d}+B_{ij}(D_{read})\right).
$$

Read cue 可以来自 old_dyn、global attention、per-frame attention、true flow 或 semantic-aided reliability。但是一个 read cue 是否有效，必须在 commit-safe protocol 下验证，而不能只看 64 / 256-frame prefix。

### 3.2 Write cue：控制未来 memory 的 probe-cache safe write

Write cue 记为：

$$
S_{write}\in\mathbb{R}^{L_{tok}}.
$$

它不一定等于 $D_{read}$。Write cue 回答的是：

> 当前 probe cache 中哪些 token 值得写入未来 TTT branch0 memory？

更合理的形式是：

$$
S_{write,i}=U_{memory,i}\cdot R_{obs,i}\cdot (1-P_{ref,i}),
$$

其中 $U_{memory}$ 表示当前 TTT memory 解释不好，需要更新；$R_{obs}$ 表示当前 observation 可靠；$P_{ref}$ 表示 reference / overlap / global-stability token，需要保护。

### 3.3 Reliability / reference cue：决定是否信任 read/write cue

Reliability cue 不是学习出来的 gate，而是 deterministic statistics：

```text
mass / coverage / fragmentation / anchor collision / confidence correlation
old_dyn overlap / temporal stability / boundary concentration / worst-chunk response
```

它决定一个新 cue 是：

```text
primary read cue
secondary modulator
write-side reliability term
diagnostic-only cue
```

---

## 4. 什么才算 Phase F 的“新 cue”

Phase F 必须避免“看起来找新 cue，实际上仍然是 old_dyn modulation”。v2 定义如下。

一个真正的新 read cue 必须能独立生成：

$$
D_{new}=f(\text{source}),
$$

并且能在以下协议下 standalone 运行：

```text
read cue      : D_new
read path     : early per-frame attention 或明确指定的新 read path
commit        : probe_native 或 probe_ttt_write
old_dyn usage : disabled, except as baseline comparison
```

下面这些不算新 cue：

```text
old_dyn × Gram agreement
old_dyn + flow rescue
old_dyn + semantic protection
old_dyn with overlap protection
old_dyn with high-anchor protection
```

它们可以作为 secondary experiments，但不能作为 Phase F 的主线。

Phase F 必须先跑 standalone，再跑 combination。推荐顺序是：

```text
D_new standalone
D_new + probe_ttt_write
old_dyn + D_new agreement/rescue
old_dyn vs D_new disagreement diagnostic
```

如果 $D_{new}$ standalone 完全失败，它可以作为 modulation / diagnostic，但不能被宣称为新的主 read cue。

---

## 5. Phase F 的总体 HMC 协议

所有有效 Phase F full runs 都必须使用新的 HMC commit-safe protocol。

对 chunk $m$，输入已提交 memory $H_m$。

Pass 1 native probe：

$$
(Y_m^{probe}, Cache_m^{probe}, H_{m+1}^{probe})=\operatorname{LoGeR}(X_m,H_m;\text{native probe}).
$$

Cue extraction：

$$
D_{read,m},\quad R_{cue,m},\quad S_{write,m},\quad P_{ref,m}
$$

Pass 2 controlled read：

$$
(Y_m^{ctrl},H_{m+1}^{ctrl})=\operatorname{LoGeR}(X_m,H_m;D_{read,m}).
$$

Final output：

$$
Y_m^{final}=Y_m^{ctrl}.
$$

Final memory commit：

$$
H_{m+1}^{final}=\operatorname{SafeWrite}(H_m,Cache_m^{probe},S_{write,m}).
$$

这意味着：

$$
\boxed{\text{controlled read output, probe-cache safe write commit}.}
$$

Phase F 正式 candidate 不允许用 `controlled commit`。`controlled commit` 可以作为 diagnostic，但不能作为 model-selection result。

---

## 6. F0：Correctness and Baseline Gate

### 6.1 实验目的

F0 的目标是确保 Phase F 的新增模块没有破坏 HMC baseline。Phase F 会新增 true-flow cache、global / per-frame attention cue exporter、implicit-cue isolation、cue audit logger、training-free reliability estimator 和 read/write pair selector。这些都会增加工程风险。

### 6.2 实验设计

F0 包含四个 full KITTI 01 correctness experiments。

第一个是 LoGeR HMC no-control。它必须使用 Phase F 新脚本、新 HMC two-pass pipeline、新 hook path，但所有 control 关闭。结果应复现：

$$
ATE=41.7502\,m,\quad Rot=8.9928^\circ.
$$

第二个是 LoGeR* HMC no-control。它必须复现：

$$
ATE=47.9793\,m,\quad Rot=5.8502^\circ.
$$

第三个是 identity hook full run。frame attention、SWA read、TTT apply、chunk attention hooks 都启用，但所有 bias / gate 为 identity。结果必须与 no-control 一致。

第四个是 D5/E5 baseline reproduction。它复现当前 best family：

```text
read : old_dyn, frame-attention pair, early layers
write: probe-cache branch0 write
commit: probe_ttt_write
```

至少要复现：

$$
ATE_{D5-07}\approx39.4903\,m
$$

和：

$$
ATE_{E5-resrel-b125}\approx39.4881\,m.
$$

### 6.3 判断标准

F0 必须满足下面条件才允许进入 F1。

$$
|ATE_{LoGeR}^{PhaseF}-41.7502|<0.1\,m.
$$

$$
|ATE_{LoGeR*}^{PhaseF}-47.9793|<0.1\,m.
$$

identity hook 与 no-control 的 pose translation diff max 应接近 $0$，允许极小数值误差，但不能出现系统性轨迹漂移。

D5 / E5 reproduction 至少应在 $\pm0.05\,m$ 内，否则说明 Phase F 改动影响了 read/write protocol，后续结果不可信。

---

## 7. F1：old_dyn decomposition and implicit-cue isolation

### 7.1 实验目的

F1 是 v2 新增的第一核心实验。当前最有效的 read cue 被称为 `old_dyn`，但它并不是单一来源。它来自 Stage-B `C_dyn` family，包含至少两类信号：

```text
explicit geometry dynamic cue
implicit attention / key-cosine dynamic cue
```

之前 D5 / E5 里有效的 `old_dyn` 使用了 calibrated fusion，比如：

```text
dyn_fusion_mode = calibrated_soft_or
implicit_weight = 0.50
implicit_gate_floor = 0.25
```

这意味着你之前手工设计的 implicit dyn_cue 已经作为 old_dyn 的一部分参与了有效实验，但我们还没有把它单独拿出来验证。F1 要回答：

> 当前 $39.49\,m$ family 的有效性到底来自 explicit cue、implicit cue，还是 fusion？

这个问题非常关键。如果 implicit-only 已经很强，Phase F 应该重点围绕 global/per-frame internal attention cue 展开。如果 explicit-only 接近 fused old_dyn，Phase F 应更多依赖 true flow / geometry consistency。如果 fusion 明显最好，就说明两个分支互补，需要重新设计 fusion，而不是简单替换。

### 7.2 cue 定义

记显式几何动态分支为：

$$
D_{exp}.
$$

它主要来自 LoGeR pointmap / pose / confidence 的几何一致性违背，例如静态一致性差、遮挡调整后的 dynamic residual 高等。

记隐式动态分支为：

$$
D_{impl}.
$$

它来自 Stage-A / Stage-B 已有的 attention / key-cosine / frame-attn response，例如：

```text
attn_dynamic_patch
frame_attn_key_cosine_avg
frame_attn_cosine_avg
manual implicit dyn_cue
```

当前 fused old_dyn 可抽象为：

$$
D_{old}=\operatorname{Fuse}(D_{exp},D_{impl}).
$$

Phase F 需要显式导出以下 read cue source：

```text
explicit_dyn_only
implicit_dyn_only
old_dyn_calibrated_soft_or
old_dyn_max
old_dyn_soft_or
old_dyn_avg
old_dyn_addclip
```

### 7.3 实验设计

F1 分两步。第一步只做 probe-only audit，不做控制。第二步在当前 best HMC protocol 下做 full KITTI 01 controlled runs。

#### F1-A：probe-only decomposition audit

对 full KITTI 01 只跑 Pass 1 native probe，保存每个 chunk 的：

```text
D_exp mass / coverage / fragmentation
D_impl mass / coverage / fragmentation
D_old mass / coverage / fragmentation
IoU(D_exp high, D_impl high)
D_exp vs D_impl correlation
D_impl vs confidence correlation
D_impl vs C_anchor collision
D_impl response on D5 worst chunks
```

D5 / E5 的 worst chunks 包括：

```text
frames 203–293
frames 435–496
```

这些区域必须单独输出 dashboard。

#### F1-B：standalone controlled full runs

固定 HMC protocol：

```text
commit mode : probe_ttt_write
read path   : early per-frame attention
bias mode   : pair
beta        : 1.00 and 1.25
write path  : branch0 BL01 write first, residual_reliability second
stage C     : none
```

优先跑下面 full KITTI 01：

| Run | Read cue | Write cue | Beta | 目的 |
|---|---|---|---:|---|
| F1-01 | explicit_dyn_only | BL01 branch0 | 1.00 | 测显式几何分支是否独立有效 |
| F1-02 | implicit_dyn_only | BL01 branch0 | 1.00 | 测手工隐式 dyn_cue 是否独立有效 |
| F1-03 | old_dyn_calibrated_soft_or | BL01 branch0 | 1.00 | D5-04 reference |
| F1-04 | explicit_dyn_only | BL01 branch0 | 1.25 | 测显式分支在 best beta 下是否稳定 |
| F1-05 | implicit_dyn_only | BL01 branch0 | 1.25 | 测隐式分支是否能达到 D5-07 |
| F1-06 | old_dyn_calibrated_soft_or | BL01 branch0 | 1.25 | D5-07 reference |
| F1-07 | old_dyn_max | BL01 branch0 | 1.25 | 判断 fusion form 是否重要 |
| F1-08 | old_dyn_soft_or | BL01 branch0 | 1.25 | 判断 soft-or 是否优于 calibrated |
| F1-09 | implicit_dyn_only | residual_reliability | 1.25 | 判断隐式 cue 更适合 read 还是 write-pair |

如果预算紧，先跑 F1-01 到 F1-06。F1-07 到 F1-09 根据前六个结果决定。

### 7.4 判断标准

F1 的 primary comparison 是 D5 / E5 plateau：

$$
ATE_{D5-07}=39.4903\,m,
$$

$$
ATE_{E5-resrel-b125}=39.4881\,m.
$$

如果 `implicit_dyn_only` 达到：

$$
ATE<39.6\,m
$$

则说明你手工设计的 implicit dyn_cue 已经接近主 cue，需要进入 F3 per-frame/internal attention expansion。

如果 `implicit_dyn_only` 明显优于 explicit，例如好 $>0.3\,m$，则 Phase F 主线转向 attention-internal cue mining。

如果 `explicit_dyn_only` 与 fused old_dyn 几乎相同，而 implicit 很弱，则说明 old_dyn 的有效性主要来自几何 residual，下一步应优先 true flow / epipolar residual。

如果 fused old_dyn 明显优于 explicit-only 和 implicit-only，则说明 fusion 是关键，下一步应设计更可靠的 fusion estimator，而不是替换 cue。

如果任何 F1 run 达到：

$$
ATE<39.0\,m,
$$

则直接进入 Phase F candidate pool。否则 F1 的主要产出是决定后续 cue search 的方向。

---

## 8. F2：Global / chunk-wise / cross-frame attention cue discovery

### 8.1 实验目的

F2 对应 “从 global attention 里找 cue”。这里的 global 不是泛指所有 attention，而是 LoGeR 中跨帧 / chunk-level / global QK 交互产生的内部 motion signal。VGGT4D 的关键发现是：pretrained geometry transformer 的 global attention layers 已经隐含 dynamic cues，但普通 QK attention 混入纹理和语义噪声，所以要用 QQ / KK / QK Gram similarity 放大 motion-induced distribution discrepancy。

Phase E 里的 Gram 失败不等于 F2 失败，因为 Phase E 只是做了：

```text
oldDyn × Gram agreement
```

这不是独立 cue。F2 要构造真正独立的：

$$
D_{global}=f(Q,K,\text{layer groups},\text{temporal window}).
$$

### 8.2 可用原料

当前 GeometryOutput 已经有若干可能的 global / cross-frame 原料：

```text
dyn4d_patch
dyn4d_qq_mean_patch
dyn4d_qk_var_patch
dyn4d_kk_mean_patch
global_q_raw_patchvec
global_k_raw_patchvec
global_q_raw_patchvec_layers
global_k_raw_patchvec_layers
dyn4d_global_layer_ids
```

如果已有 `dyn4d_patch` 已经接近 VGGT4D-style cue，它必须作为 standalone read cue 验证，而不是只用作 old_dyn modulation。

### 8.3 cue 候选

F2 定义以下候选。

#### F2-G1：dyn4d_patch standalone

$$
D_{global}^{dyn4d}=\operatorname{Norm}(\texttt{dyn4d\_patch}).
$$

这是最低成本的 global-attn read cue。

#### F2-G2：QK variance cue

$$
D_{global}^{qkvar}=\operatorname{Norm}(\texttt{dyn4d\_qk\_var\_patch}).
$$

它测试 temporal QK variance 是否能捕捉 moving / unstable token。

#### F2-G3：QQ / KK disagreement cue

$$
D_{global}^{qqkk}=\operatorname{Norm}\left((1-S^{QQ})\cdot(1-S^{KK})\right).
$$

它测试 token self-similarity 不稳定是否比 QK variance 更有用。

#### F2-G4：VGGT4D-style Gram cue

若可以导出 layer-group Q/K，则构造：

$$
w_{shallow}=(1-S^{KK}_{shallow})\odot V^{QK}_{shallow},
$$

$$
w_{middle}=1-S^{QQ}_{middle},
$$

$$
w_{deep}=(1-V^{QQ}_{deep})\odot S^{QQ}_{deep},
$$

$$
D_{global}^{gram}=\operatorname{Norm}(w_{shallow}\odot w_{middle}\odot w_{deep}).
$$

这里 layer groups 不必完全复制 VGGT4D，因为 LoGeR 的层结构不同。初始可用：

```text
shallow: earliest global / chunk attention group
middle : middle global / chunk attention group
deep   : late global / chunk attention group
```

F2 必须记录每组 layer 的 contribution map，避免组合后不可解释。

#### F2-G5：chunk-attention dispersion cue

如果 HMC hook 能拿到 chunk-wise bidirectional attention trace，则计算：

$$
H_i=-\sum_j A_{ij}\log(A_{ij}+\epsilon),
$$

$$
V_i=\operatorname{Var}_{s\in W(t)}A_{i,\cdot}^{t\rightarrow s},
$$

$$
D_{global}^{chunk}=\operatorname{Norm}(aH_i+bV_i+cD_{layer-disagree}).
$$

这个 cue 更接近 LoGeR chunk 内几何推理路径。

### 8.4 实验设计

F2 分三步。

第一步是 full-run probe-only audit。所有 F2-G candidates 都只提取，不控制，输出 full-sequence cue quality。

第二步是 standalone read-only `probe_native` test。若 cue 通过 probe audit，则使用：

```text
read cue  : D_global candidate
read path : early per-frame attention first
bias      : key and pair both test, but full candidates最多两个
commit    : probe_native
write     : none
```

为什么先控制 per-frame attention？因为目前唯一稳定有效的 read intervention point 是 early per-frame attention。即使 cue 来自 global attention，它也可以作为 per-frame attention 的 dynamic map。

第三步是 `probe_ttt_write` test。若 standalone read-only 接近或超过 CM02，则组合 branch0 safe write。

建议优先实验：

| Run | Cue | Bias | Commit | 目的 |
|---|---|---|---|---|
| F2-01 | dyn4d_patch | pair | probe_native | 最低成本 global standalone |
| F2-02 | dyn4d_patch | pair | probe_ttt_write | 与 safe write 组合 |
| F2-03 | qk_var | pair | probe_native | 测 QK variance |
| F2-04 | gram_vggt4d_style | pair | probe_native | 测独立 Gram cue |
| F2-05 | gram_vggt4d_style | pair | probe_ttt_write | 若 F2-04 接近 CM02 |
| F2-06 | chunk_dispersion | pair | probe_native | 若 chunk trace 可用 |

### 8.5 判断标准

F2 的 standalone read-only 必须对比：

$$
ATE_{CM02}=39.7820\,m.
$$

如果某个 global cue read-only 不能达到：

$$
ATE<40.0\,m,
$$

它不能作为 primary read cue，但可以保留为 auxiliary reliability cue。

如果某个 global cue 在 probe_ttt_write 下达到：

$$
ATE<39.0\,m,
$$

则它成为 Phase F main candidate。

如果 F2-G candidates 都失败，但 F1 显示 implicit dyn cue 有效，则说明当前 global implementation 仍不够好，Phase F 应转向 per-frame attention cue，而不是继续 Gram modulation。

---

## 9. F3：Per-frame attention cue discovery

### 9.1 实验目的

F3 对应 “从 per-frame attention 里找 cue”。这是目前最有现实根据的方向，因为有效 read intervention point 本身就是 per-frame attention。MUT3R 的启发是，frozen transformer 的 self-attention map 会暴露 dynamic / unstable regions，动态区域往往 attention 更分散、更不稳定，静态区域 attention 更集中、更一致。

Phase C v2 的 `internal_attn` / `internal_attn top10` 失败，不能说明 per-frame attention cue 方向失败。它只能说明当时的 per-frame-attn-derived cue 太粗。F3 要重新设计更细的 per-frame cue，并把你手工设计的 implicit dyn_cue 作为 standalone family。

### 9.2 可用原料

当前 GeometryOutput 已经有：

```text
attn_dynamic_patch
frame_attn_cosine_shallow
frame_attn_cosine_deep
frame_attn_cosine_avg
frame_attn_key_cosine_l0
frame_attn_key_cosine_l4
frame_attn_key_cosine_shallow
frame_attn_key_cosine_deep
frame_attn_key_cosine_avg
frame_attn_cosine_query_layers
frame_attn_cosine_key_layers
frame_attn_cosine_layer_ids
```

F3 还可以从 HMC hook 直接导出 selected frame-attention layers 的 attention entropy、key exposure 和 layer disagreement。

### 9.3 cue 候选

#### F3-P1：manual implicit dyn cue

这是你之前手工设计的隐式 dynamic cue。Phase F 必须把它从 old_dyn fusion 中独立出来，作为：

$$
D_{frame}^{manual}=D_{impl}.
$$

#### F3-P2：key-cosine cue

从 key-cosine response 构造：

$$
D_{frame}^{keycos}=\operatorname{Norm}(1-\operatorname{CosSim}(K_t,K_{support})).
$$

分别测试 shallow、deep、avg：

```text
frame_attn_key_cosine_shallow
frame_attn_key_cosine_deep
frame_attn_key_cosine_avg
```

#### F3-P3：query-cosine cue

类似地测试 query-side response：

```text
frame_attn_cosine_shallow
frame_attn_cosine_deep
frame_attn_cosine_avg
```

#### F3-P4：attention entropy cue

对 frame self-attention：

$$
H_i=-\sum_j A_{ij}\log(A_{ij}+\epsilon).
$$

高 entropy 表示 query 分散、不稳定；但也可能只是低纹理或 sky，因此必须配合 reliability audit。

#### F3-P5：key exposure / received-attention cue

$$
E_j=\sum_i A_{ij}.
$$

长期高 exposure、低 dynamic 的 token 可能是 reference；短时 spike 的 token 可能是 transient。这里可构造：

$$
D_{frame}^{exposure}=\operatorname{Norm}(\text{short-spike}(E_j)-\text{stable-exposure}(E_j)).
$$

#### F3-P6：shallow-deep disagreement cue

$$
D_{frame}^{sd}=\operatorname{Norm}\left(|D_{shallow}-D_{deep}|\right).
$$

如果 shallow 高、deep 低，可能是 semantic/motion saliency；如果 deep 也高，可能是 geometry-critical structure 或 persistent ambiguity。

### 9.4 实验设计

F3 也分 probe audit、standalone read、probe_ttt_write 三步。

优先 full candidates 不超过 8 个。建议顺序：

| Run | Cue | Bias | Commit | 目的 |
|---|---|---|---|---|
| F3-01 | manual_implicit_dyn | pair | probe_native | 验证手工 implicit cue read-only |
| F3-02 | manual_implicit_dyn | pair | probe_ttt_write | 验证 implicit cue + safe write |
| F3-03 | key_cosine_avg | pair | probe_native | 验证 key-cosine avg |
| F3-04 | key_cosine_shallow | pair | probe_native | 验证 shallow key instability |
| F3-05 | key_cosine_deep | pair | probe_native | 验证 deep key instability |
| F3-06 | entropy | key | probe_native | 测 entropy 是否可用 |
| F3-07 | shallow_deep_disagree | pair | probe_native | 测 layer disagreement |
| F3-08 | best F3 cue | pair | probe_ttt_write | 组合 branch0 safe write |

如果 F1 已经显示 manual implicit cue 强，F3-01 / F3-02 必须优先。

### 9.5 判断标准

F3 的 primary gate 是：

$$
ATE_{probe\_native}<39.7820\,m
$$

或：

$$
ATE_{probe\_ttt\_write}<39.4903\,m.
$$

如果 manual implicit cue 达到 D5/E5 水平，则说明你之前的隐式 cue 是当前系统中最值得继承的 internal signal。

如果 key-cosine / entropy / disagreement cue 在 read-only 上能低于 $39.5\,m$，说明 per-frame attention cue 可以替代 old_dyn，进入 Phase F candidate pool。

如果所有 F3 cue 都不能超过 old_dyn，但它们能显著降低 rotation / final error，则它们可以作为 reference / protection estimator，而不是主 read cue。

---

## 10. F4：True optical-flow / epipolar residual cue

### 10.1 实验目的

F4 是真正的 external motion cue。之前的 `flow_sem_veto`、`flow_proxy_calib` 不是 RAFT / GMFlow 等真实 optical flow，而是 Stage-B reprojection / patch-match proxy。它们在 64 / 256-frame 和 stateful slice 上有信号，但 full sequence 失败，原因包括 mass 过稀疏、fragmentation 高、high-confidence sparse residual selector 偏差。

F4 的目标是验证：冻结 RAFT / GMFlow 级别的 true flow 是否能提供 old_dyn 缺失的 motion-specific evidence。

### 10.2 cue 计算

对于帧 $t$ 到支持帧 $s$，冻结 flow 网络输出：

$$
F_{obs}(t\rightarrow s,u).
$$

LoGeR probe 几何给出当前点 $X_t(u)$ 和相机 pose，于是静态刚体解释下的 expected flow 是：

$$
F_{rigid}(t\rightarrow s,u)=\pi(T_s^{-1}T_tX_t(u))-u.
$$

定义 flow residual：

$$
r_{flow}(t,s,u)=\frac{\|F_{obs}(t\rightarrow s,u)-F_{rigid}(t\rightarrow s,u)\|_2}{\epsilon+\|F_{obs}(t\rightarrow s,u)\|_2}.
$$

再结合 forward-backward consistency：

$$
M_{fb}(t,s,u)=\mathbf{1}\left[\|F_{t\rightarrow s}(u)+F_{s\rightarrow t}(u+F_{t\rightarrow s}(u))\|<\tau_{fb}\right].
$$

最终：

$$
D_{flow}(t,u)=\operatorname{RobustAgg}_{s\in\mathcal{N}(t)}\left[r_{flow}(t,s,u)\cdot M_{fb}\cdot M_{conf}\cdot(1-C_{occ})\right].
$$

这里 $M_{conf}$ 可以来自 flow confidence、LoGeR confidence 和有效投影 mask。

### 10.3 实验设计

F4 必须先实现 frozen flow cache，并明确记录模型版本：

```text
flow_model = RAFT or GMFlow
flow_pair_stride = 1 / 2 / 4
flow_fb_thr
flow_conf_thr
```

不能再用 `flow_model=patch_match` 作为 Phase F full candidate。

实验顺序：

| Run | Cue | Bias | Commit | 目的 |
|---|---|---|---|---|
| F4-01 | true_flow_residual | key | probe_native | standalone flow read-only |
| F4-02 | true_flow_residual | pair | probe_native | 测 pair bias 是否更好 |
| F4-03 | true_flow_residual | pair | probe_ttt_write | standalone flow + safe write |
| F4-04 | old_dyn ∧ true_flow agreement | pair | probe_ttt_write | 只增强二者一致区域 |
| F4-05 | old_dyn with true_flow static rescue | pair | probe_ttt_write | flow 只修 old_dyn 误伤 |
| F4-06 | true_flow with semantic structure protect | pair | probe_ttt_write | 防止 road/building 误伤 |

### 10.4 判断标准

F4 的 standalone cue 必须先过 full-run probe audit。dynamic mass 不能像 v3 那样低到 $0.010$，fragmentation 不能像 v4 proxy flow 那样高。

如果 F4-01 / F4-02 read-only 不能接近：

$$
ATE<40.0\,m,
$$

则 true flow 不作为 primary read cue，但仍可作为 old_dyn 的 static rescue / motion agreement。

如果 F4-03 到 F4-06 任一达到：

$$
ATE<39.0\,m,
$$

则 true flow 成为 Phase F main candidate。

如果 true flow 仍然表现为 short-window 好、full-run 失败，则说明 KITTI 01 的 full trajectory problem 不是单纯 moving-object mask 问题，后续不应再扩展 flow 阈值 sweep。

---

## 11. F5：Semantic and structure-aware training-free reliability cue

### 11.1 实验目的

F5 不把 semantic 当作主 dynamic cue，而是作为 structure/reference reliability。Phase E 中 rotation-safe protection 已经证明，保护 overlap / anchor / reference 能改善 rotation，但会牺牲 ATE。F5 的目标不是重复这些保护，而是更精准地区分：

```text
road / building / sidewalk / guardrail: orientation / endpoint sensitive structure
moving car / person / cyclist: possible dynamic distractor
sky / vegetation: low-value but not necessarily harmful
```

### 11.2 KITTI-specific semantic reference

Stage C 的 thing / stuff 还不够稳，thing 会漏检，stuff 边界粗糙。因此 semantic 只能作为 weak reliability cue。Prompt bank 应面向 KITTI：

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

### 11.3 用法

对 structure token，不直接强行提高写入，而是降低其 read suppression：

$$
D_{read}^{sem}(i)=D_{read}(i)(1-\rho_{struct}P_{struct}(i)).
$$

对 movable thing，只在 motion cue 也高时才压制：

$$
D_{read}^{mov}(i)=D_{read}(i)\cdot \mathbf{1}[P_{mov}(i)\land D_{read}(i)>\tau].
$$

对 sky / vegetation，不作为强动态，但可以降低 write priority：

$$
S_{write}(i)\leftarrow S_{write}(i)(1-\rho_{low}P_{low}(i)).
$$

### 11.4 判断标准

F5 不要求 standalone 赢 old_dyn。它只要在已有 best cue 上修复 rotation / final error 即可。通过标准是：

$$
ATE\le 39.6\,m,
$$

同时：

$$
Rot<9.5^\circ,
$$

$$
FinalErr<5.5\,m.
$$

如果 semantic protection 仍然导致 ATE 大幅损失，则语义保留为 visualization / debug，不进入主方法。

---

## 12. F6：Training-free reliability estimator and cue promotion gate

### 12.1 实验目的

F6 是 Phase F 的 gate，不是学习模块。它决定一个 cue 能不能进入 full controlled candidate。

过去 Phase C v3 / v4 最大的问题是 short-sequence 和 stateful slice 会产生 false positive。`flow_sem_veto` 在 64 / 256-frame 和 6/7 stateful slices 上都不错，但 full KITTI 01 失败。F6 要防止这种情况重演。

### 12.2 cue audit statistics

每个 cue 都必须记录：

```text
mean dynamic mass
per-chunk coverage
fragmentation
anchor collision
confidence correlation
C_anchor collision
old_dyn IoU / recall / precision
temporal stability
support-frame consistency
boundary concentration
worst-chunk response
reference-frame suppression
```

其中：

$$
\text{mass}=\operatorname{Mean}(D>0.5),
$$

$$
\text{fragmentation}=\frac{\#\text{connected components}}{\text{dynamic area}+\epsilon},
$$

$$
\text{anchor collision}=\operatorname{Mean}(D\cdot C_{anchor}).
$$

还要记录 cue 在 D5 worst chunks：

```text
frames 203–293
frames 435–496
```

上的响应。一个 cue 如果只在 easy chunks 上激活，而在 worst chunks 无响应，不应作为 primary cue。

### 12.3 training-free promotion rules

F6 的 promotion 不依赖 GT ATE 训练，只用 cue statistics 给出初筛。

一个 cue 可以进入 controlled full candidate，需要满足：

```text
chunk coverage high
fragmentation not extreme
anchor collision controlled
not overly correlated with confidence
not near-empty in long middle chunks
not suppressing reference / overlap tokens excessively
```

具体数值可以根据 full-run cue 分布设成宽松 band，但不得用 ATE 最优值来拟合阈值。推荐采用 reference-based sanity：old_dyn 的 mass / fragmentation / coverage 作为参考，不要求新 cue 完全相同，但极端 sparse 或极端 fragmented 必须 reject。

### 12.4 cue role assignment

F6 最终给每个 cue 分配角色：

```text
primary_read
secondary_modulator
write_reliability
reference_protection
diagnostic_only
rejected
```

只有 `primary_read` 可以进入 F8 read/write pair search。`secondary_modulator` 只能和 old_dyn 或 promoted cue 做 agreement / rescue。`diagnostic_only` 不允许跑 full model candidate。

---

## 13. F7：Update-needed write from probe cache

### 13.1 实验目的

F7 是 write-side 新 cue。Phase D/E 已经证明 branch0 write 与 read correction 互补，但当前 write cue 仍然主要来自 dynamic suppression。Phase F 要测试真正的 update-needed write。

### 13.2 TTT residual cue

对于 probe cache 中的 TTT branch $r$：

$$
\hat v_i^{(r)}=f_{W_m}^{(r)}(k_i),
$$

$$
e_i^{(r)}=\frac{\|\hat v_i^{(r)}-v_i\|_2}{\|v_i\|_2+\epsilon}.
$$

如果 $e_i$ 高，表示当前 memory 对 token $i$ 解释不好。但高 residual 可能来自 dynamic / occlusion / low confidence，所以要乘 reliability：

$$
S_{write,i}=\operatorname{rank}(e_i)\cdot R_i\cdot(1-P_{ref,i}).
$$

### 13.3 alignment confidence cue

借鉴 TTT3R，memory-observation alignment 可以作为 learning-rate confidence：

$$
\beta_i=\sigma\left(\operatorname{Agg}(Q_{memory}K_{obs}^\top)\right).
$$

在 LoGeR 中可以用 probe cache 的 q/k 或 TTT apply residual近似。alignment high 不一定代表要强写；它可能代表当前 observation 与 memory 对齐好，写入安全。可设计：

$$
S_{write,i}=\operatorname{rank}(e_i)\cdot \beta_i\cdot R_i.
$$

### 13.4 sparse exact-preserve route

借鉴 MeMix，dense gate 会让每个 token 都写一点；sparse route 允许未选 token exact preserve。对 branch0：

$$
M_i=\mathbf{1}[S_{write,i}\in\operatorname{TopK}],
$$

$$
p_i=\frac{M_i}{\frac{\sum_j\eta_jM_j}{\sum_j\eta_j+\epsilon}+\epsilon}.
$$

第一版只测试：

```text
branch0 only
sparse ratio = 0.95 / 0.90 / 0.85
branch1/2 unity
special/reference protected
```

### 13.5 实验设计

F7 在固定 read cue 下测试 write cue。起点用两个 read cue：

```text
old_dyn reference read
best promoted new read cue from F1-F4
```

候选写入：

| Run | Read cue | Write cue | Sparse | 目的 |
|---|---|---|---|---|
| F7-01 | old_dyn | BL01 old_dyn branch0 | dense MP | D5 reference |
| F7-02 | old_dyn | TTT residual | dense MP | 测 residual write |
| F7-03 | old_dyn | residual × reliability | dense MP | 测 reliability write |
| F7-04 | old_dyn | alignment confidence | dense MP | 测 alignment write |
| F7-05 | old_dyn | residual × reliability | sparse95 | 测 exact preserve |
| F7-06 | best new read | residual × reliability | dense MP | 新 read + update-needed write |
| F7-07 | best new read | residual × reliability | sparse95 | 新 read + sparse update-needed |

### 13.6 判断标准

F7 必须对比：

$$
ATE_{E5-resrel-b125}=39.4881\,m.
$$

如果 update-needed write 只改善 $<0.05\,m$，则它只是 weak modulator。

如果它达到：

$$
ATE<39.0\,m,
$$

并且 rotation / final error 不恶化，则进入 Phase F candidate pool。

如果 sparse route 降低 rotation / final error 但 ATE 不赢，可作为 balanced candidate，但不能作为 best model。

---

## 14. F8：Read/write pair search with strict budget

### 14.1 实验目的

F8 不是大矩阵 sweep，而是把 F1–F7 里被 promotion gate 认可的少数 cue 成对组合。Phase F 的重点是 “new cue source”，不是无休止组合。

### 14.2 pair search 原则

只允许进入 F8 的 read cue：

```text
old_dyn reference
implicit_dyn_only if F1 passes
best global-attn cue if F2 passes
best per-frame-attn cue if F3 passes
true_flow cue if F4 passes
```

只允许进入 F8 的 write cue：

```text
BL01 branch0 old_dyn write
residual_reliability
alignment confidence
sparse residual route
semantic/reference-protected sparse route
```

### 14.3 实验预算

F8 最多跑 8 个 full KITTI 01 candidates。推荐模板：

| Run | Read cue | Write cue | Commit | 目的 |
|---|---|---|---|---|
| F8-01 | old_dyn | BL01 | probe_ttt_write | D5/E5 reference |
| F8-02 | implicit_dyn | best write | probe_ttt_write | 若 F1 implicit pass |
| F8-03 | global_gram | best write | probe_ttt_write | 若 F2 global pass |
| F8-04 | perframe_attn | best write | probe_ttt_write | 若 F3 per-frame pass |
| F8-05 | true_flow | best write | probe_ttt_write | 若 F4 true-flow pass |
| F8-06 | old_dyn + best_newcue rescue | best write | probe_ttt_write | 测 modulator |
| F8-07 | best_read | sparse residual route | probe_ttt_write | 测 exact-preserve |
| F8-08 | best_read | semantic/reference protected write | probe_ttt_write | 测 balanced candidate |

### 14.4 判断标准

Phase F 只有达到下面标准才算成功。

最小有效进展：

$$
ATE<39.0\,m.
$$

主模型候选：

$$
ATE<38.0\,m.
$$

强结果：

$$
ATE<35.0\,m.
$$

不允许用严重 rotation / endpoint 损伤换 ATE。若某个结果只有 $ATE$ 略低，但：

$$
Rot>10.0^\circ
$$

或：

$$
FinalErr>7.0\,m,
$$

则只能算 diagnostic，不算 main candidate。

---

## 15. F9：Worst-chunk and failure attribution dashboard

### 15.1 实验目的

当前 D5/E5 的 worst chunks 集中在：

```text
Chunk 7–9: frames 203–293
Chunk 15–16: frames 435–496
```

如果一个新 cue 在全局平均上看起来不错，但对这些 worst chunks 没有改善，就很难突破 D5/E5 plateau。F9 要为所有 promoted cue 输出同一套可视化。

### 15.2 dashboard 内容

每个 promoted cue 对 worst chunks 输出：

```text
RGB frames
old_dyn map
explicit_dyn map
implicit_dyn map
new cue map
cue disagreement map
C_anchor / confidence / C_occ / C_unc maps
frame-attention bias energy
TTT branch0 write prior
TTT residual / alignment confidence
semantic reference overlay if enabled
local trajectory error
chunk-boundary continuity
```

### 15.3 使用方式

F9 不是模型选择本身，而是失败定位。若 full result 不过关，但 dashboard 明确显示问题来自 reference suppression、flow fragmentation 或 semantic false positive，则可以设计一个 single rescue run。否则不允许继续无限制 sweep。

---

## 16. Stop conditions

Phase F 必须有明确停止条件，避免继续在 old_dyn family 附近小幅震荡。

满足任一条件即停止当前分支：

1. F1 证明 implicit / explicit / fused 都无法解释 D5 以外的新收益，且 F2–F4 无新 cue 过 $39.0\,m$。
2. true flow standalone、global Gram standalone、per-frame attention standalone 都未能在 probe_ttt_write 下超过 $39.49\,m$。
3. update-needed write 只带来 $<0.05\,m$ 的提升，且不能改善 rotation / final error。
4. 所有 promoted candidates 都停留在 $39.45\sim39.55\,m$。
5. 任何分支需要依赖 KITTI GT 调 threshold 才有效，则该分支不符合 training-free 约束，必须停止。

如果 Phase F 结束时最好仍然约为：

$$
ATE\approx39.5\,m,
$$

那么结论应是：

> HMC training-free hand-designed cue search 在 KITTI 01 上已经达到约 $39.5\,m$ 的稳定平台。若要继续大幅提升，需要引入更强的冻结外部 motion source、更多序列验证，或改变 evaluation / inference protocol，而不是继续 old_dyn 微调。

---

## 17. Phase F 预期结果与决策树

Phase F 预期可能出现四种结果。

第一种结果是 F1 发现 implicit_dyn_only 已经接近或超过 fused old_dyn。如果发生这种情况，说明你之前手工设计的隐式 dyn_cue 是核心有效信号。后续应集中在 per-frame / global internal cue refinement，而不是 true flow。

第二种结果是 true flow standalone 过关。如果 RAFT / GMFlow residual 能进入 $<39.0\,m$，说明 old_dyn 的主要缺陷是 motion evidence 不独立，Phase F 应转向 flow + structure reliability。

第三种结果是 global Gram/QK standalone 过关。如果独立 Gram cue 过关，说明 VGGT4D-style internal motion mining 能迁移到 LoGeR/HMC，后续应重点完善 layer-group selection 和 early-stage masking。

第四种结果是所有新 cue standalone 都不如 old_dyn。那说明 old_dyn 的 full-sequence stability 很难替代。此时只能把新 cue 作为 diagnostic / protection，而不应继续主 cue search。

---

## 18. 最终交付物

Phase F 结束时必须交付：

```text
1. F0 correctness report
2. explicit / implicit / fused old_dyn decomposition report
3. global-attn cue audit report
4. per-frame-attn cue audit report
5. true-flow cue audit report, if flow implemented
6. semantic structure reliability report
7. update-needed write report
8. read/write pair search result table
9. worst-chunk dashboard
10. final recommendation: promoted cue / rejected cue / stop condition
```

每个 full run 必须保存：

```text
01.txt trajectory
kitti_benchmark.log
hmc_correctness_summary.json
cue_quality_summary.json
cue_quality_per_chunk.jsonl
hook_effect_summary.jsonl
write_prior_summary.jsonl
trajectory_diagnostics.json
wandb run id
```

---

## 19. 一句话总结

Phase F v2 的核心不是继续修补 `old_dyn`，而是先拆清楚 `old_dyn` 的显式/隐式贡献，再分别从 global attention、per-frame attention、true optical flow 和 probe-cache memory residual 中寻找真正独立的新 cue source。所有新 cue 必须在 training-free、commit-safe HMC protocol 下通过 full-run probe audit 和 controlled full validation；只有能突破 $39.0\,m$，并且不以严重 rotation / endpoint 损伤换 ATE 的 cue，才有资格进入主模型候选。

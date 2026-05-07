# LoGeR Attention Cue Library v2：内部 Attention Cue 深挖与实验计划

版本：v2.0  
目标阶段：Cue discovery before TTT write policy  
核心约束：不引入 external motion model；所有主 cue 必须来自 LoGeR 内部 attention、token、memory trace 或现有几何输出。  
主评价协议：HMC two-pass，read-only 使用 `probe_native` commit，hybrid 使用 `probe_ttt_write` commit。  
公式格式：Typora-friendly，全部使用 `$...$` 或 `$$...$$`。

---

## 0. 当前判断与结论

我同意“cue 还没有挖干净”这个判断，而且这不是补几个 beta sweep 的问题，而是现有 Attention Cue Library 的覆盖维度还不完整。

目前 `gg.qq.middle.low.robustq` 的意义很大：它证明 **global query inconsistency** 是 LoGeR 内部真实存在的有效信号，并且这个信号可以突破旧 `old_dyn_addclip` 的平台。但它不等于最终最优 cue，因为它只固定了一个相对粗的组合：

```text
global attention source + query-query similarity + middle proportional window + low-similarity direction + robust quantile normalization
```

这里至少还有五个没有充分回答的问题：

1. `middle` 是否真的是最优层段，还是只是当前比例划分碰巧命中？
2. `qq` 是否真的是最优 latent basis，还是少数 heads / q-k / real attention-map 统计里还有更强信号？
3. `low similarity` 是否应该作为唯一 dynamic score，还是应该和 shallow saliency、deep prior、static rescue 组合？
4. 当前 cue 是更适合 read suppression，还是也适合 write eligibility？
5. 当前成功来自 attention cue 本身，还是来自它和 `probe_ttt_write` commit-safe 协议的组合？

因此下一阶段不应该直接进入大规模 TTT write policy，也不应该只围绕 `gg.qq.middle.low.robustq` 做细 beta sweep。更合理的路线是建立 **Attention Cue Library v2**：先把 LoGeR 内部可观测的 attention / token / memory 信号系统导出、质量审计、可视化和分层筛选，然后只让通过质量审计的候选进入 full read-only 与 hybrid-safe 实验。

本计划的核心结论是：

> `gg.qq.middle.low.robustq` 是新主线的 seed，不是终点。下一阶段的目标不是“再找一个名字更复杂的 cue”，而是建立一个可审计的 cue library，并把每个 cue 明确归类为 `D_read`、`S_rescue`、`R_cue` 或 `W_hint`。

---

## 1. 从 VGGT4D 与 MUT3R 迁移到 LoGeR 的设计原则

### 1.1 VGGT4D 给我们的原则

VGGT4D 的关键思想可以压缩成四点。

第一，它不直接相信普通 attention map。普通 query-key attention 容易混合语义、纹理和运动响应，因此动态区域在标准 attention 里可能并不干净。VGGT4D 使用同分布向量之间的 Gram similarity，例如 query-query 或 key-key self-similarity，来放大由运动造成的分布差异。

第二，它不是简单按网络深度比例划分 shallow / middle / deep。它观察到不同层段有不同功能：浅层更像 semantic saliency，中层更像 motion variability，深层更像 spatial prior / outlier suppression。因此 layer windows 必须从经验现象出发，而不是只做比例切分。

第三，它不是只取一个 factor。VGGT4D 的动态图来自多个 layer group 的组合：浅层 factor 提供 foreground / semantic saliency，中层 factor 提供 motion instability，深层 factor 提供空间先验和 outlier suppression。也就是说，product 或 routing 是它的重要组成部分。

第四，它只在 early-stage 做 selective masking，而不是全层 hard masking。过强或过深的 masking 会让模型偏离预训练分布。迁移到 LoGeR 时，这一点对应：read intervention 要保持 soft、早期、direction-aware，并且 future memory commit 必须隔离 controlled read 的副作用。

### 1.2 MUT3R 给我们的原则

MUT3R 的关键思想也可以压缩成四点。

第一，冻结的 3D transformer 内部 attention 已经有 motionness signal。这个 signal 不一定直接等于物体运动 mask，但它可以表示 token 在跨时间/跨状态交互中的不稳定性。

第二，motionness 不应该只看单层或单头。MUT3R 从多层 self-attention response 聚合出 patch-level dynamic score。迁移到 LoGeR 时，我们必须做 per-layer、per-window、per-head、per-support audit。

第三，gating 要 direction-aware。不同 attention 方向里应该抑制 query 还是 key 并不相同：如果目标是阻止动态 token 被静态 query 读取，应该更像 key suppression；如果目标是阻止不稳定 token 从 memory 读取错误上下文，应该更像 query suppression。

第四，layer evolution 本身是信号。浅层 activation 碎、局部、语义强；中层开始聚合；深层更几何一致。动态干扰如何跨层传播，可能比某一层的绝对数值更有诊断价值。

### 1.3 LoGeR 与 VGGT4D/MUT3R 的差别

LoGeR 不是普通 offline VGGT，也不是 CUT3R 式单状态 recurrent decoder。LoGeR 是 chunk-wise 长序列架构，包含：

```text
frame attention -> SWA local memory -> TTT global memory apply/update -> chunk/global bidirectional attention
```

因此 LoGeR 的 cue discovery 必须考虑三种语义不同的信号：

1. 当前 chunk 内部 attention 是否发现了不稳定 token；
2. SWA / overlap 是否发现了跨 chunk 局部对齐不稳定；
3. TTT apply/update 是否显示某些 token 正在污染长期 memory。

这意味着下一阶段不能只做 `gg.qq.middle.low` 的局部变体，而应该把所有 LoGeR 内部可用 trace 组织成一个 library，并区分其用途。

---

## 2. 下一阶段的核心目标

### 2.1 不是直接刷 ATE，而是建立可审计 cue library

下一阶段的直接目标不是立刻找一个 `38.8 m` 的 run，而是建立一个能长期复用的 Attention Cue Library v2。这个 library 应满足：

1. 每个 cue 的来源、层段、head 聚合、support 策略、归一化、方向都可复现；
2. 每个 cue 在 full sequence 上都有质量审计，而不是只看 64/256 frame；
3. 每个 cue 都明确归类为 read dynamic、static rescue、reliability gate 或 write hint；
4. 只有通过质量审计的 cue 才进入 read-only full run；
5. 只有 read-only 有明确增益的 cue 才进入 hybrid `probe_ttt_write`。

### 2.2 四类 cue 输出，而不是一个 scalar

Attention Cue Library v2 不应该只输出一个 `D_dyn`。建议统一输出四类 map：

| 输出 | 语义 | 主要用途 |
|---|---|---|
| `D_read` | 当前 read path 中疑似应抑制的动态/不稳定 token | frame/global attention read bias |
| `S_rescue` | 当前不应被抑制的静态可靠 token | static protection / rescue |
| `R_cue` | 当前 cue 本身可信度 | routing / fallback |
| `W_hint` | memory-side update-needed 或 write-risk hint | 第二优先级 TTT write policy |

其中最重要的原则是：

$$
D_{read} \neq 1 - S_{rescue}
$$

有些 token 不是明显 dynamic，但也不适合作为长期 anchor；有些 token 看起来不稳定，但 deep/global context 可能认为它是可救的静态结构。把 dynamic suppression 和 static rescue 分开，是下一阶段避免 ATE/rotation trade-off 的关键。

---

## 3. 统一数据结构与命名规范

### 3.1 Cue tensor 基本形状

对当前 chunk，设：

- $T$：chunk 内帧数；
- $P$：每帧 patch token 数；
- $L$：LoGeR 可观测 global/chunk stack 层数；
- $H$：attention head 数；
- $d_h$：每个 head 的维度。

每个 patch cue 最终都应投影为：

$$
D \in [0,1]^{T \times P}
$$

如果是 per-head cue，则中间形状为：

$$
D^{head} \in [0,1]^{L_w \times H \times T \times P}
$$

其中 $L_w$ 是某个 layer window 的层数。

### 3.2 Cue ID 命名

建议所有候选使用统一名字：

```text
acl2.<source>.<basis>.<stat>.<layerwin>.<support>.<headagg>.<norm>.<role>
```

示例：

```text
acl2.gg.qq.low.g13_15.off6.headmean.robustq.Dread
acl2.gg.qq.var.v4d_mid.off246.headtop2.robustq.Dread
acl2.gg.attn.entropy.g0_5.full.headmean.robustq.Dread
acl2.gg.qq.high.g13_17.full.headagree.robustq.Srescue
acl2.cam.patch.mass.g13_17.full.headmean.robustq.Rcue
acl2.ttt.applyerr.g12_17.native.tokenmean.robustq.Whint
```

命名中必须保留以下字段：

| 字段 | 示例 | 说明 |
|---|---|---|
| `source` | `gg`, `fa`, `swa`, `ttt`, `cam`, `reg` | cue 来源 |
| `basis` | `qq`, `kk`, `qk`, `attn`, `applyerr` | 统计对象 |
| `stat` | `low`, `high`, `mean`, `var`, `entropy`, `drift` | 统计方式 |
| `layerwin` | `g0`, `g3_5`, `v4d_mid`, `slide3_g8` | 层窗口 |
| `support` | `full`, `off246`, `past`, `future`, `noovlp` | temporal support |
| `headagg` | `headmean`, `headmax`, `headtop2`, `headvar`, `headagree` | head 聚合 |
| `norm` | `robustq`, `zsigmoid`, `rank`, `otsu` | 归一化 |
| `role` | `Dread`, `Srescue`, `Rcue`, `Whint` | 用途 |

### 3.3 每个 cue 必须落盘的 metadata

每个 cue 目录必须包含：

```text
cue_manifest.json
cue_quality_summary.json
cue_quality_per_chunk.csv
cue_quality_per_frame.csv
cue_tensor_stats.npz
visual_index.json
```

`cue_manifest.json` 至少包含：

```json
{
  "cue_id": "acl2.gg.qq.low.g13_15.off246.headtop2.robustq.Dread",
  "source": "gg",
  "basis": "qq",
  "stat": "low",
  "layers": [13, 14, 15],
  "support": [-6, -4, -2, 2, 4, 6],
  "head_aggregation": "top2",
  "normalization": "robustq",
  "role": "D_read",
  "created_from_run": "...",
  "commit_protocol": "probe_only_for_cue_cache",
  "notes": "..."
}
```

---

## 4. 归一化与基础公式

### 4.1 robust quantile normalization

所有 raw score 进入控制前必须做 robust normalization。对某个 raw map $X$：

$$
Z_{robust}(X) = \operatorname{clip}\left(\frac{X - q_{lo}(X)}{q_{hi}(X)-q_{lo}(X)+\epsilon}, 0, 1\right)
$$

默认：

$$
q_{lo}=0.05, \qquad q_{hi}=0.95
$$

对低相似度作为 dynamic 的 cue：

$$
D_{low}(X) = 1 - Z_{robust}(X)
$$

对高相似度作为 static rescue 的 cue：

$$
S_{high}(X) = Z_{robust}(X)
$$

### 4.2 temporal support 集合

对目标帧 $t$，定义支持帧集合：

$$
\mathcal S(t) = \{s \mid s \neq t, s \in [0,T-1]\}
$$

下一阶段必须比较以下 support：

| support ID | 定义 | 目的 |
|---|---|---|
| `full` | chunk 内所有其他帧 | 最大上下文 |
| `off246` | $t+\{-6,-4,-2,2,4,6\}$，越界裁剪 | VGGT4D-style 固定 offset |
| `near12` | $t+\{-2,-1,1,2\}$ | 局部一致性 |
| `far612` | $t+\{-12,-8,-6,6,8,12\}$ | 远距离一致性 |
| `past` | $s<t$ | causal read/write 分析 |
| `future` | $s>t$ | offline cue 上界 |
| `noovlp` | 排除 overlap frames | 防止 overlap token 过拟合 |
| `ovlp_only` | 只看 overlap / boundary frames | 跨 chunk 边界诊断 |

固定 offset 支持集定义为：

$$
\mathcal S_{off246}(t)=\{t+\delta \mid \delta \in \{-6,-4,-2,2,4,6\}, 0 \leq t+\delta < T\}
$$

### 4.3 query-query / key-key Gram similarity

设第 $l$ 层、第 $h$ 个 head、第 $t$ 帧、第 $p$ 个 patch 的 query/key 为：

$$
q_{l,h,t,p} \in \mathbb R^{d_h}, \qquad k_{l,h,t,p} \in \mathbb R^{d_h}
$$

对目标 token $(t,p)$ 和支持帧 $s$，定义跨帧 token similarity：

$$
G^{qq}_{l,h,t,s}(p)=\operatorname{Agg}_{p' \in \mathcal P_s}\; \cos(q_{l,h,t,p}, q_{l,h,s,p'})
$$

其中 $\operatorname{Agg}$ 可取：

```text
mean, max, topk_mean, percentile90, percentile10
```

默认先使用 `topk_mean`，因为它比全局 mean 更接近“是否能在支持帧找到稳定 counterpart”：

$$
G^{qq,topk}_{l,h,t,s}(p)=\frac{1}{K}\sum_{p' \in \operatorname{TopK}_{p'} \cos(q_{l,h,t,p},q_{l,h,s,p'})}\cos(q_{l,h,t,p},q_{l,h,s,p'})
$$

对支持帧和层窗口聚合：

$$
\bar G^{qq}_{\mathcal L, h, t}(p)=\operatorname{Mean}_{l \in \mathcal L, s \in \mathcal S(t)}G^{qq}_{l,h,t,s}(p)
$$

动态 low-sim cue：

$$
D^{qq-low}_{\mathcal L,h,t}(p)=1-Z_{robust}\left(\bar G^{qq}_{\mathcal L,h,t}(p)\right)
$$

同理可定义 $G^{kk}$ 与 $G^{qk}$。

---

## 5. Cue family A：非比例 layer windows

### 5.1 为什么必须做

当前 `gg.qq.middle.low` 的 `middle` 由比例规则得到，例如 `[13,15,17,19,21,23]` 这样的层组。但 VGGT4D 的浅/中/深不是按比例取出来的，而是基于观察到的功能层段：浅层语义显著性、中层运动变化、深层空间先验。因此 LoGeR 也必须做非比例 layer search。

### 5.2 LoGeR ratio-mapped VGGT4D windows

设 VGGT 原始层数为 $L_v$，LoGeR 可用 global/chunk stack 层数为 $L_g$。将 VGGT 层 $l_v$ 映射到 LoGeR 层 $l_g$：

$$
l_g = \operatorname{round}\left(\frac{l_v-1}{L_v-1}(L_g-1)\right)
$$

若取 $L_v=24$，$L_g=18$，则 VGGT4D-style 层段大致映射为：

| VGGT4D 层段 | 功能 | LoGeR ratio-mapped 候选 |
|---|---|---|
| Layer 1 | shallow semantic saliency | `g0` |
| Layers 4-8 | middle motion variability | `g2:g6` |
| Layers 18-22 | deep spatial prior | `g13:g17` |
| Layers 19-20 | deep variance prior | `g13:g15` 或 `g14:g15` |

同时保留用户提出的 LoGeR 原生窗口：

```text
g0, g3, g8, g13, g17
win[2:6], win[12:17], win[13:15]
```

### 5.3 必扫 layer windows

第一轮只做 audit，不直接 full control。候选如下：

| 组别 | 候选 | 说明 |
|---|---|---|
| singleton | `g0` 到 `g17` 全部 | 找真正峰值层 |
| sparse singleton | `g0`, `g3`, `g8`, `g13`, `g17` | 用户指定关键层 |
| sliding length 2 | `g0_1`, `g1_2`, ..., `g16_17` | 局部双层稳定性 |
| sliding length 3 | `g0_2`, `g1_3`, ..., `g15_17` | 局部三层 |
| sliding length 5 | `g0_4`, `g2_6`, `g4_8`, `g8_12`, `g12_16`, `g13_17` | 中大窗口 |
| VGGT4D mapped | `v4d_shallow=g0`, `v4d_mid=g2_6`, `v4d_deep=g13_17`, `v4d_deepvar=g13_15` | exact/ratiomapped |
| current reference | 当前 `middle` 定义 | 与 `gg.qq.middle.low.robustq` 对照 |

### 5.4 每个窗口必须记录的 layer profile

对每个 candidate 不只记录最终 cue map，还要记录 layer profile：

$$
\mu_l=\operatorname{Mean}_{t,p}D_l(t,p)
$$

$$
\sigma_l=\operatorname{Std}_{t,p}D_l(t,p)
$$

$$
\rho_l=\operatorname{Corr}\left(D_l, D_{old\_dyn}\right)
$$

$$
\alpha_l=\operatorname{Corr}\left(D_l, C_{anchor}\right)
$$

必须可视化：

```text
layer index -> mean dynamic mass
layer index -> fragmentation
layer index -> Corr(D, old_dyn)
layer index -> Corr(D, C_anchor)
layer index -> ATE if promoted
```

---

## 6. Cue family B：严格 VGGT4D-style factor 与 product

### 6.1 目标

这一组不是为了直接相信 product，而是为了分解：到底是 shallow factor、中层 factor、deep factor，还是它们的组合对 KITTI01 有贡献。

用户指定必须补：

```text
v4d.mean1
v4d.var1
v4d.mean2
v4d.mean3
v4d.var3
v4d.product
```

其中下标含义建议定义为：

| factor | LoGeR window | 初始解释 |
|---|---|---|
| `mean1` | shallow `g0` | 语义/前景显著性 |
| `var1` | shallow `g0` 或 `g0_2` | 浅层不稳定 / 纹理显著性 |
| `mean2` | middle `g2_6`，另测 `g8`, `g12_17`, current middle | 中层 query/query low-sim 或 motion inconsistency |
| `mean3` | deep `g13_17` | deep spatial prior / static rescue |
| `var3` | deep `g13_15` 或 `g14_15` | deep outlier / boundary instability |

### 6.2 factor 定义

对任意 layer window $\mathcal L_i$，先计算 Gram mean：

$$
M_i(t,p)=\operatorname{Mean}_{l \in \mathcal L_i, h, s \in \mathcal S(t)}G^{qq}_{l,h,t,s}(p)
$$

计算 Gram variance：

$$
V_i(t,p)=\operatorname{Var}_{l \in \mathcal L_i, h, s \in \mathcal S(t)}G^{qq}_{l,h,t,s}(p)
$$

定义六个 raw factor：

$$
F_{mean1}=Z_{robust}(M_1)
$$

$$
F_{var1}=Z_{robust}(V_1)
$$

$$
F_{mean2}^{low}=1-Z_{robust}(M_2)
$$

$$
F_{mean3}^{high}=Z_{robust}(M_3)
$$

$$
F_{var3}^{low}=1-Z_{robust}(V_3)
$$

其中 `mean2` 默认用 low direction，因为当前最强 seed 是 `gg.qq.middle.low`。

### 6.3 product 族，而不是单一 product

严格 VGGT4D-style product 应该视为一个 product family。第一轮至少跑下面五个 product：

| product ID | 公式 | 目的 |
|---|---|---|
| `v4d.product.m1_m2low_m3` | $F_{mean1} \cdot F_{mean2}^{low} \cdot F_{mean3}^{high}$ | canonical 三段组合 |
| `v4d.product.v1_m2low_m3` | $F_{var1} \cdot F_{mean2}^{low} \cdot F_{mean3}^{high}$ | 浅层 variance 是否优于 shallow mean |
| `v4d.product.m2low_m3` | $F_{mean2}^{low} \cdot F_{mean3}^{high}$ | 去掉浅层语义，测试是否更适合 KITTI |
| `v4d.product.m1_m2low_m3_v3low` | $F_{mean1} \cdot F_{mean2}^{low} \cdot F_{mean3}^{high} \cdot F_{var3}^{low}$ | deep variance 作为 outlier veto |
| `v4d.product.m2low_rescue` | $F_{mean2}^{low} \cdot (1-\lambda_s S_{deep})$ | 加 static rescue 的 dynamic read map |

其中：

$$
S_{deep}=F_{mean3}^{high}\cdot F_{var3}^{low}
$$

product 后再次 robust normalization：

$$
D_{product}=Z_{robust}\left(\prod_i F_i^{\alpha_i}\right)
$$

默认 $\alpha_i=1$。若 product 过稀疏，再测试 $\alpha_i=0.5$ 的 softened product。

### 6.4 必须记录 factor contribution

对每个 product，必须记录它与各 factor 的关系：

$$
\operatorname{Corr}(D_{product}, F_{mean1}),\quad
\operatorname{Corr}(D_{product}, F_{mean2}^{low}),\quad
\operatorname{Corr}(D_{product}, S_{deep})
$$

如果 product 成功，但几乎完全由 $F_{mean2}^{low}$ 决定，则说明 `gg.qq.middle.low` 已经捕获主要成分；如果 product 失败但某个 factor 单独成功，则说明乘法过强或方向错。

---

## 7. Cue family C：per-head cue

### 7.1 为什么必须做

目前很多 q/k cue 是 head-average 后的 patch vector similarity。这样会隐藏一个重要现象：动态信号可能只存在于少数 heads 中，而 head-average 会把它稀释。

因此下一阶段必须保留 per-head Q/K/V 或 per-head attention statistics。

### 7.2 需要新增的 hook

在 global/chunk attention 中导出：

```text
q: [chunk, layer, frame, head, token, d_head]
k: [chunk, layer, frame, head, token, d_head]
v: [chunk, layer, frame, head, token, d_head] optional
attn_summary: optional compressed attention statistics
```

注意：不要求一开始保存全 attention matrix。先保存 Q/K per-head 就能做大部分 Gram cue。

### 7.3 per-head map 定义

对每个 head 独立计算：

$$
D_h(t,p)=1-Z_{robust}\left(\bar G^{qq}_{h,t}(p)\right)
$$

然后比较以下 head 聚合方式。

#### head mean

$$
D_{mean}(t,p)=\frac{1}{H}\sum_{h=1}^H D_h(t,p)
$$

#### head max

$$
D_{max}(t,p)=\max_h D_h(t,p)
$$

#### head top-k

$$
D_{topk}(t,p)=\frac{1}{K}\sum_{h \in \operatorname{TopK}_h D_h(t,p)}D_h(t,p)
$$

建议 $K \in \{1,2,4\}$。

#### head variance

$$
D_{varhead}(t,p)=\operatorname{Var}_{h}(D_h(t,p))
$$

这个 cue 不是直接 dynamic，而是 head disagreement；优先作为 `R_cue` 或 diagnostic。

#### head agreement

给定阈值 $\tau_h$：

$$
A_{agree}(t,p)=\frac{1}{H}\sum_h \mathbf 1[D_h(t,p)>\tau_h]
$$

agreement 高说明多个 heads 同意该 token 不稳定；agreement 低但 max 高说明只有少数 heads 激活，可能是高价值稀疏 cue，也可能是假阳性。

#### head entropy

定义每个 token 的 head distribution：

$$
\pi_h(t,p)=\frac{D_h(t,p)+\epsilon}{\sum_{h'}D_{h'}(t,p)+H\epsilon}
$$

$$
H_{head}(t,p)=-\frac{1}{\log H}\sum_h \pi_h(t,p)\log(\pi_h(t,p)+\epsilon)
$$

低 entropy 表示少数 heads 主导，高 entropy 表示多 heads 均匀响应。它本身更适合作为 `R_cue`。

### 7.4 per-head audit 必须回答的问题

1. 哪些 heads 在 full sequence 上最稳定？
2. 最强 heads 是否集中在某些层段？
3. `head_max` 是否比 `head_mean` 更强？
4. `head_topk` 是否能提升 ATE，同时不增加 fragmentation？
5. head disagreement 是否对应 rotation improvement 或 endpoint damage？
6. top heads 在不同 chunk 是否一致，还是只在局部片段偶然激活？

---

## 8. Cue family D：真实 attention-map 统计

### 8.1 为什么必须做

Q/K centroid 或 Gram similarity 是省显存的 proxy，但它不等于真实 token-to-token attention map。VGGT4D 的 spatial std / temporal aggregation 本质上是在 attention 或 Gram matrix 的 token-to-token 维度上做统计。LoGeR 当前如果只看 q/k 向量相似度，可能漏掉以下信号：

1. 一个 query 是否分散地 attend 到多个不一致区域；
2. 一个 token 是否主要 attend 到同帧、近帧、远帧或 overlap；
3. attention 的空间重心和方差是否异常；
4. dynamic token 是否是被很多 static queries 读取的 harmful key。

### 8.2 内存策略

不建议一开始全量保存 attention matrix：

$$
A \in \mathbb R^{L \times H \times TP \times TP}
$$

这会非常大。建议采用 compressed attention-map audit：

1. 只对 selected layers 保存：`g0`, `g3`, `g8`, `g13`, `g17`, `g2_6`, `g13_17`；
2. 只对 selected heads 保存：先保存 head mean 和 top heads；
3. 只保存 row-wise statistics，不保存完整矩阵；
4. 对少量 diagnostic chunks 保存完整 attention submatrix，用于可视化。

### 8.3 attention entropy

对 attention row $A_{i,j}$：

$$
H_{attn}(i)=-\frac{1}{\log N}\sum_{j=1}^{N}A_{i,j}\log(A_{i,j}+\epsilon)
$$

高 entropy 可能表示 query 不确定或分散；低 entropy 可能表示稳定匹配，也可能表示错误过度集中。因此它必须与 spatial std、support consistency 一起看。

### 8.4 attention spatial std

每个 key token $j$ 有空间坐标 $x_j=(u_j,v_j)$。定义 query $i$ 的 attention spatial centroid：

$$
\mu_i=\sum_j A_{i,j}x_j
$$

attention spatial variance：

$$
\sigma_i^2=\sum_j A_{i,j}\|x_j-\mu_i\|_2^2
$$

归一化后得到：

$$
D_{spatialstd}(i)=Z_{robust}(\sigma_i)
$$

对帧内/跨帧分别计算：

$$
\sigma_{same}(i),\quad \sigma_{cross}(i),\quad \sigma_{support}(i)
$$

### 8.5 temporal attention mass

定义 query token $i=(t,p)$ attend 到支持帧 $s$ 的 mass：

$$
M_{t \rightarrow s}(i)=\sum_{j \in \mathcal P_s}A_{i,j}
$$

必须记录：

```text
same-frame mass
near-frame mass
far-frame mass
past mass
future mass
overlap mass
```

动态/不稳定 token 可能表现为：

- 过高 same-frame mass，缺乏跨帧支持；
- 跨帧 mass 分散但没有稳定峰；
- past/future asymmetry 很强；
- 对 overlap/history 的 attention 异常低。

### 8.6 key harmfulness

不仅 query row 有意义，key column 也有意义。一个 token 如果被很多 static queries attend 到，可能是有价值 static anchor；如果被很多不稳定 queries attend 到，可能是污染源。

定义 key received mass：

$$
K_{recv}(j)=\sum_i A_{i,j}
$$

定义 static-query 到 key 的 mass：

$$
K_{static}(j)=\sum_i (1-D_i)A_{i,j}
$$

如果某个 key 自己 dynamic 高，但 $K_{static}$ 高，则它是 harmful key，应适合 key suppression：

$$
D_{harmkey}(j)=D_j \cdot Z_{robust}(K_{static}(j))
$$

这类 cue 可能比单纯 `D_j` 更适合 read-path key bias。

---

## 9. Cue family E：temporal support 选择

### 9.1 目标

当前 `gg.qq.middle.low` 使用 chunk 内其他帧 centroid 或全 support。VGGT4D 使用固定 temporal window，例如 source frames with stride 2。LoGeR 必须比较 support 策略，因为 KITTI01 的长直线/高速运动会让不同 temporal offset 的稳定性差异很大。

### 9.2 support ablation matrix

每个强候选至少比较以下 support：

| support | 控制变量 | 预期用途 |
|---|---|---|
| `full` | 所有非当前帧 | 上界/平均稳定 |
| `off246` | 固定 offset | VGGT4D-style |
| `near12` | 邻近帧 | 局部对齐 |
| `far612` | 远帧 | 长程一致性 |
| `past` | 只过去 | causal 写入语义 |
| `future` | 只未来 | offline upper bound |
| `noovlp` | 排除 overlap | 防止 overlap artifact |
| `ovlp_only` | 只 overlap | boundary diagnostic |

### 9.3 support asymmetry 指标

定义 past/future cue：

$$
D_{past}(t,p),\quad D_{future}(t,p)
$$

support asymmetry：

$$
A_{pf}(t,p)=|D_{past}(t,p)-D_{future}(t,p)|
$$

如果 $A_{pf}$ 高，说明该 token 的不稳定判断依赖时间方向。它可能是遮挡边界、刚进入/刚离开视野的物体，或者 pose/localization 错误。此类 cue 不一定适合 hard dynamic suppression，更适合 `R_cue` 或 `C_occ` 类似角色。

---

## 10. Cue family F：camera/register/special token cue

### 10.1 为什么必须做

LoGeR 中 patch tokens 不是唯一携带 pose/rigidity 信息的 token。register token、role/PE tokens、camera/global tokens 可能对相机位姿和 chunk-level geometry 更敏感。VGGT4D 也讨论了 camera-token attention 的局限：camera token 不足以单独生成精确 dynamic mask，但它仍然可能作为 frame-level reliability 或 veto signal。

因此下一阶段不要把 camera/register cue 当作 primary dynamic mask，而应优先把它们作为：

```text
frame-level reliability
chunk-level veto
static rescue confidence
rotation-risk diagnostic
```

### 10.2 special-to-patch attention

若存在 special token $c$ 到 patch token $p$ 的 attention：

$$
A_{c \rightarrow p}^{l,h,t}
$$

定义 special attention saliency：

$$
C_{sp}(t,p)=\operatorname{Mean}_{l,h,c}A_{c \rightarrow p}^{l,h,t}
$$

可构造：

$$
D_{cam-low}(t,p)=1-Z_{robust}(C_{sp}(t,p))
$$

但不建议直接用作 dynamic cue。更好的用法是判断现有 dynamic cue 是否与 pose token 的关注区域冲突。

### 10.3 patch-to-special attention

patch token 是否强依赖 special token：

$$
A_{p \rightarrow c}^{l,h,t}
$$

定义：

$$
R_{pose}(t,p)=Z_{robust}\left(\operatorname{Mean}_{l,h,c}A_{p \rightarrow c}^{l,h,t}\right)
$$

如果一个 token 的 $D_{read}$ 高但 $R_{pose}$ 也高，它可能是 pose-sensitive 区域，不宜直接 hard suppress，需要 static rescue 或 lower beta。

### 10.4 frame-level reliability

定义 frame-level special conflict：

$$
F_{conflict}(t)=\operatorname{Mean}_{p}\left(D_{read}(t,p)\cdot R_{pose}(t,p)\right)
$$

如果 $F_{conflict}(t)$ 高，说明当前 frame 的 dynamic suppression 可能伤 pose。该 frame 应触发：

```text
read beta decay
static rescue stronger
write budget reduction
```

---

## 11. Cue family G：layer evolution cue

### 11.1 为什么必须做

只看某一层的相似度会漏掉“动态干扰在网络里传播”的信息。MUT3R 的 layer-wise PCA 分析说明，浅层到深层的演化本身能反映动态干扰是否被网络吸收、放大或消解。

### 11.2 q/k drift

定义相邻层 query drift：

$$
\Delta q_l(t,p)=1-\cos(q_{l,t,p}, q_{l+1,t,p})
$$

如果维度或投影不同，则先对向量做 LayerNorm 或使用对应 layer 的 patch embedding $h_l$：

$$
\Delta h_l(t,p)=1-\cos(h_{l,t,p}, h_{l+1,t,p})
$$

跨窗口 drift：

$$
\Delta q_{a:b}(t,p)=1-\cos\left(\operatorname{Mean}_{l \in a}q_l(t,p),\operatorname{Mean}_{l \in b}q_l(t,p)\right)
$$

候选：

```text
q_drift.g0_3
q_drift.g3_8
q_drift.g8_13
q_drift.g13_17
shallow_high_deep_low
middle_unstable_deep_stable
deep_stability_gain
```

### 11.3 cross-regime patterns

定义 shallow saliency：

$$
S_1=F_{mean1}
$$

middle instability：

$$
D_2=F_{mean2}^{low}
$$

deep static prior：

$$
S_3=F_{mean3}^{high}\cdot F_{var3}^{low}
$$

构造四种模式：

| 模式 | 公式 | 解释 | 建议用途 |
|---|---|---|---|
| shallow high, middle low, deep low | $S_1D_2(1-S_3)$ | 可能是真动态或噪声 | `D_read` |
| shallow high, middle low, deep high | $S_1D_2S_3$ | 可能是动态物体边界，也可能可救 | `D_read` with rescue |
| shallow low, middle low, deep low | $(1-S_1)D_2(1-S_3)$ | 几何不稳定但非语义前景 | `R_cue` / uncertainty |
| middle low, deep high | $D_2S_3$ | 当前 `gg.qq` 可能的一种核心模式 | read candidate |

---

## 12. Cue family H：static rescue cue

### 12.1 为什么必须做

当前 best ATE 已经过 `<39`，下一瓶颈很可能是 rotation、endpoint 或局部漂移。单纯 dynamic suppression 容易误伤 pose-sensitive static anchors，尤其是道路边缘、建筑边界、远景结构、horizon 附近稳定区域。

因此必须构造显式的 static rescue cue。

### 12.2 static rescue 定义

建议从 deep high-similarity、low entropy、高 support agreement 生成：

$$
S_{deep}=Z_{robust}(M_3)\cdot (1-Z_{robust}(V_3))
$$

如果有 attention map entropy：

$$
S_{ent}=1-Z_{robust}(H_{attn})
$$

如果有 support coverage：

$$
S_{sup}=Z_{robust}(\operatorname{Coverage}_{support})
$$

最终：

$$
S_{rescue}=Z_{robust}\left(S_{deep}^{\alpha}S_{ent}^{\beta}S_{sup}^{\gamma}\right)
$$

默认：

$$
\alpha=1,\quad \beta=0.5,\quad \gamma=0.5
$$

### 12.3 rescue 接入 read cue

不要把 rescue 直接当作 negative dynamic。建议用 protection 形式：

$$
D_{read}^{rescued}=D_{read}\cdot (1-\lambda_{rescue}S_{rescue})
$$

其中：

$$
\lambda_{rescue} \in \{0.25,0.5,0.75\}
$$

另一个选择是只保护 top rescue：

$$
D_{read}^{rescued}(t,p)=
\begin{cases}
\min(D_{read}(t,p), d_{cap}), & S_{rescue}(t,p)>\tau_s \\
D_{read}(t,p), & \text{otherwise}
\end{cases}
$$

默认：

$$
\tau_s=0.8,\quad d_{cap}=0.2
$$

### 12.4 rescue 必须记录的指标

1. rescue mass：$\operatorname{Mean}(S_{rescue}>0.5)$；
2. rescue 与 `C_anchor` 相关性；
3. rescue 与 `old_dyn` collision；
4. 被 rescue 保护的 token 中，原本 $D_{read}$ 高的比例；
5. rescue 后 dynamic mass 降低多少；
6. read-only ATE 是否上升，Rot 是否下降；
7. endpoint 是否改善。

---

## 13. Cue family I：TTT/SWA memory-side cue

### 13.1 定位

这一类不是第一优先级 read cue，但必须在 library 中落盘，因为它对第二优先级 TTT write policy 很有价值。

现有 `residual_reliability` 仍然较粗。下一阶段需要按 TTT layer、branch、token type 记录更细的 memory-side signal。

### 13.2 TTT apply residual

设 TTT apply 输出为：

$$
o_i=f_W(q_i)
$$

定义 apply residual magnitude：

$$
R^{apply}_{l,i}=\frac{\|o_{l,i}\|_2}{\|h_{l,i}\|_2+\epsilon}
$$

如果某些 token 在 TTT apply 中被强烈改变，但后续 geometry error 变差，它们可能是 read-risk token。

### 13.3 TTT update prediction error

TTT update 用 key/value 构造 self-supervised loss。定义：

$$
E^{pred}_{l,b,i}=\frac{\|f_{W_l^{b}}(k_{l,i})-v_{l,i}\|_2}{\|v_{l,i}\|_2+\epsilon}
$$

其中 $b$ 是 TTT branch。高 prediction error 表示当前 token 对 memory 来说难以解释，可能需要弱写或作为 update-needed cue。

### 13.4 update magnitude

对 token contribution $J_{l,b,i}$：

$$
U_{l,b,i}=\|J_{l,b,i}\|_F
$$

按 branch 聚合：

$$
U_i^{branch0},\quad U_i^{branch1},\quad U_i^{branch2}
$$

必须记录：

```text
branch0 update magnitude map
branch1 update magnitude map
branch2 update magnitude map
branch disagreement map
patch token vs special token update ratio
```

### 13.5 SWA history attention cue

若 SWA trace 可用，记录当前 token attend 到 previous chunk tokens 的 mass：

$$
M_{swa-prev}(i)=\sum_{j \in C_{m-1}}A^{swa}_{i,j}
$$

以及 overlap mass：

$$
M_{swa-ovlp}(i)=\sum_{j \in O_{m-1,m}}A^{swa}_{i,j}
$$

低 history mass 可能表示局部对齐弱，高 history mass 但 geometry residual 高可能表示错误对齐。

---

## 14. Cue family J：routing fusion，而不是 add/clip

### 14.1 为什么 simple add/clip 不够

`old_dyn_addclip` 和 `gg.qq.middle.low` 都强，但它们的错误类型不同。直接相加会把两者的 false positive 也相加，容易增加 fragmentation 或 anchor collision。

更合理的 fusion 是 routing：

```text
当 geometry cue 可信时，用 geometry cue；
当 geometry cue 不可信但 attention cue 稳定时，用 attention cue；
当两者都高时，强 suppress；
当 static rescue 高时，保护；
当两者冲突时，降低 beta 或退回 explicit_dyn。
```

### 14.2 soft routing 公式

设：

- $D_g$：geometry dynamic cue，例如 `old_dyn_addclip` 或 `explicit_dyn_only`；
- $D_a$：attention dynamic cue，例如 `gg.qq.middle.low`；
- $S$：static rescue；
- $R_g$：geometry cue reliability；
- $R_a$：attention cue reliability。

定义 attention route weight：

$$
w_a=\frac{R_a}{R_a+R_g+\epsilon}
$$

geometry route weight：

$$
w_g=\frac{R_g}{R_a+R_g+\epsilon}
$$

基础融合：

$$
D_{route}=w_gD_g+w_aD_a
$$

agreement boost：

$$
D_{agree}=D_{route}+\lambda_{agree}D_gD_a
$$

static rescue：

$$
D_{final}=\operatorname{clip}\left(D_{agree}(1-\lambda_s S),0,1\right)
$$

默认：

$$
\lambda_{agree}\in\{0.25,0.5\},\quad \lambda_s\in\{0.25,0.5,0.75\}
$$

### 14.3 reliability 定义

geometry reliability 可由低 fragmentation、低 uncertainty、与 anchor 不冲突生成：

$$
R_g=(1-C_{unc})\cdot(1-C_{occ})\cdot(1-C_{anchor}\cdot D_g)
$$

attention reliability 可由 head agreement、support consistency、low temporal flicker 生成：

$$
R_a=A_{headagree}\cdot(1-F_{support\_asym})\cdot(1-F_{flicker})
$$

如果没有所有项，第一版可使用：

$$
R_a=A_{headagree}\cdot S_{sup}
$$

### 14.4 hard routing diagnostic

为了理解错误模式，也保留 hard routing 诊断：

| 区域 | 条件 | 处理 |
|---|---|---|
| agree dynamic | $D_g>\tau_g$ 且 $D_a>\tau_a$ | strong suppress |
| geometry only | $D_g>\tau_g$ 且 $D_a\leq\tau_a$ | use explicit/old with lower beta |
| attention only | $D_g\leq\tau_g$ 且 $D_a>\tau_a$ | weak suppress, log as implicit |
| rescued | $S>\tau_s$ | cap suppression |
| uncertain conflict | $D_g$ high, $D_a$ high, $R_a$ low | no promotion without visual audit |

---

## 15. 实验总协议

### 15.1 固定 HMC 协议

所有 cue discovery 必须固定以下协议：

```text
Cue cache / audit:
    Pass 1 native/probe only
    no controlled read
    no controlled commit

Read-only evaluation:
    output = controlled forward
    commit = probe_native

Hybrid-safe evaluation:
    output = controlled forward
    commit = probe_ttt_write
```

禁止用 `controlled commit` 作为 cue 是否有效的主判断，因为它会把 controlled read 的 TTT side effect 写入未来 memory。

### 15.2 分阶段实验

下一阶段命名为 `ACL2`，分为七个阶段：

```text
ACL2-0: hook correctness and cache format
ACL2-1: full-run cue audit without control
ACL2-2: layer/window/support/head candidate pruning
ACL2-3: read-only full evaluation for promoted single cues
ACL2-4: VGGT4D-style factor/product evaluation
ACL2-5: static rescue and routing fusion
ACL2-6: hybrid-safe probe_ttt_write evaluation
ACL2-7: failure analysis and cross-sequence sanity check
```

---

## 16. ACL2-0：工程 hook 与 correctness

### 16.1 目标

确认新增 per-head Q/K、attention-map stats、special-token trace、TTT/SWA memory-side trace 不改变模型输出。

### 16.2 必须实现的 hooks

| hook | 形状 | 是否必须 | 用途 |
|---|---|---|---|
| global/chunk q per-head | `[layer, frame, head, token, d_head]` | 必须 | qq/qk/kk Gram |
| global/chunk k per-head | `[layer, frame, head, token, d_head]` | 必须 | qq/qk/kk Gram |
| attention row stats | `[layer, head, frame, token, stats]` | 必须 | entropy/spatial std/mass |
| attention submatrix sample | selected chunks only | 可选但建议 | visualization |
| special token q/k/attn | varies | 必须 | pose/register cue |
| TTT apply residual | `[layer, token]` | 必须 | memory-side diagnostic |
| TTT update prediction error | `[layer, branch, token]` | 必须 | write hint |
| SWA history mass | `[swa_layer, token]` | 建议 | boundary/local memory cue |

### 16.3 correctness gates

必须复现：

1. no-control LoGeR native；
2. identity hooks；
3. current best read-only/hybrid reference；
4. cue cache enabled but no control 时输出完全一致。

记录：

```text
max_pose_diff
max_pointmap_diff
max_confidence_diff
state_hash_before_probe
state_hash_after_probe
state_hash_after_commit
hook_identity_max_bias
cache_write_success_rate
```

Gate：

```text
max_pose_diff == 0 or < 1e-8
identity hook max bias == 0
probe_no_commit_hash_equal == true for all chunks
no-control ATE exactly reproduced within numerical tolerance
```

---

## 17. ACL2-1：full-run cue audit without control

### 17.1 目标

在不施加任何控制的情况下，对所有 candidate cue 做 full sequence 质量审计。这个阶段不看 controlled ATE，只筛掉明显不稳定、过稀疏、过碎、过度撞 anchor 的 cue。

### 17.2 候选生成范围

第一轮 audit 生成以下候选：

```text
A. layer windows:
   singleton / sliding / VGGT4D-mapped / current-middle

B. basis:
   qq / kk / qk / qqkk_disagree

C. stat:
   low / high / mean / var / layer_drift

D. support:
   full / off246 / near12 / far612 / past / future / noovlp / ovlp_only

E. head aggregation:
   headmean / headmax / headtop1 / headtop2 / headtop4 / headvar / headagree / headentropy

F. attention-map stats:
   row_entropy / spatial_std / temporal_mass / harmful_key / cross_frame_concentration

G. special/memory:
   cam_patch_mass / reg_patch_mass / ttt_applyerr / ttt_prederr / swa_prevmass
```

### 17.3 cue quality metrics

每个 cue 必须记录以下 full-run 指标。

#### dynamic mass

$$
M_{dyn}=\operatorname{Mean}_{t,p}\mathbf 1[D(t,p)>0.5]
$$

同时记录：

```text
mean(D)
p50(D)
p75(D)
p90(D)
p95(D)
```

#### coverage

按 chunk 定义是否有效：

$$
\operatorname{valid}(m)=\mathbf 1[M_{dyn}^{m} > \tau_{min}]
$$

$$
\operatorname{Coverage}=\frac{1}{M}\sum_m \operatorname{valid}(m)
$$

默认：

$$
\tau_{min}=0.01
$$

#### fragmentation

对每帧 threshold 后的 connected components 数 $N_{cc}(t)$ 和 dynamic area $A_{dyn}(t)$：

$$
F_{frag}=\operatorname{Mean}_t\frac{N_{cc}(t)}{A_{dyn}(t)+\epsilon}
$$

可以用 patch-grid connected components。无需精确像素级。

#### anchor collision

$$
C_{anchor-collide}=\operatorname{Mean}_{t,p}\left[D(t,p)\cdot C_{anchor}(t,p)\right]
$$

如果 dynamic cue 经常压到 high anchor 区域，rotation 和 endpoint 很容易变差。

#### confidence correlation

$$
\rho_{conf}=\operatorname{Corr}(D, Conf_{geo})
$$

通常希望 dynamic risk 不只是低 confidence 的翻版。

#### old/explicit cue relationship

记录：

$$
\operatorname{Corr}(D, D_{old})
$$

$$
\operatorname{IoU}(D>0.5, D_{old}>0.5)
$$

$$
\operatorname{Corr}(D, D_{explicit})
$$

这能判断新 cue 是提供新信息，还是只是 old_dyn 的平滑版。

#### temporal flicker

$$
F_{flicker}=\operatorname{Mean}_{t,p}|D(t,p)-D(t-1,p)|
$$

需要注意 KITTI 中 camera motion 强，same-pixel flicker 不是完美指标，但仍可作为 proxy。

#### support asymmetry

$$
F_{asym}=\operatorname{Mean}_{t,p}|D_{past}(t,p)-D_{future}(t,p)|
$$

#### spatial vertical distribution

把图像分成上中下三个 y-bin：

```text
top / middle / bottom
```

记录 dynamic mass 在三个区域的比例。KITTI 上如果 cue 过多集中在 sky/top，可能是 false positive；如果过多集中在 bottom road，可能伤 pose/scale anchor。

### 17.4 audit promotion gate

候选进入 read-only full evaluation 前，建议满足：

```text
Coverage >= 0.90
0.05 <= mean(D>0.5) <= 0.60
Fragmentation <= 0.08  或者显著低于 proxy-flow failed case
Anchor collision <= 0.15
std(D) >= 0.05
不是纯 confidence map: |Corr(D, Conf)| <= 0.60
不是纯 old_dyn copy: Corr(D, old_dyn) <= 0.95，除非目标是 old_dyn refinement
```

这不是绝对物理阈值，而是防止短序列 false positive 的第一道门。

---

## 18. ACL2-2：层、head、support 的候选剪枝

### 18.1 目标

从 ACL2-1 的大批 cue 中找出最值得 full controlled run 的少数候选。

### 18.2 分层剪枝流程

先按 family 内排序，不跨 family 直接比较。

#### Layer window 排序

每个 basis/stat/support/headagg 固定后，按 layer window 排序：

```text
quality_score = + coverage
              - fragmentation
              - anchor_collision
              + std(D)
              + novelty_score
```

其中：

$$
novelty=1-|\operatorname{Corr}(D,D_{old})|
$$

不要求 novelty 越高越好；如果 ATE 目标强，仍可保留与 old_dyn 高相关的 refinement。但至少要知道它是不是新信息。

#### Head 聚合排序

对同一 layer/support/stat，比较：

```text
headmean
headmax
headtop1
headtop2
headtop4
headagree
headvar
```

判断是否存在少数 heads 显著优于平均。

#### Support 排序

对同一 layer/head/stat，比较：

```text
full
off246
near12
far612
past
future
noovlp
```

如果 `future` 明显强于 `past`，说明该 cue 可能依赖 offline 信息，不适合作长期 write，但仍可用于 read-only。若 `past` 与 `future` 都强，则更适合未来写入控制。

### 18.3 输出 top candidates

每个 family 最多保留：

```text
layer-window family: top 8
v4d factor/product family: top 8
per-head family: top 8
attention-map family: top 6
support asymmetry family: top 4
special/register family: top 4
layer-evolution family: top 6
static rescue family: top 6
memory-side family: top 6 as write hints only
```

---

## 19. ACL2-3：single cue read-only full evaluation

### 19.1 目标

只测试通过 audit 的 single cue，使用统一 read intervention 和 `probe_native` commit，判断它是否真能改善当前 output。

### 19.2 固定 read intervention

默认使用 key suppression，因为当前目标是阻止 static/pose query 读取不稳定 dynamic keys：

$$
B_{i,j}=-\beta D_{read}(j)
$$

若使用 pair mode：

$$
B_{i,j}=-\beta (1-D_{read}(i))D_{read}(j)
$$

若使用 query mode：

$$
B_{i,j}=-\beta D_{read}(i)
$$

第一轮建议固定 key mode，避免 read direction 和 cue source 混淆。只有 key mode 通过后，再比较 pair/query。

### 19.3 beta sweep

对每个 promoted cue，先粗扫：

```text
beta = 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00
```

若最佳在边界，扩展：

```text
beta = 3.50, 4.00
```

若最佳在 2.0-3.0，细扫：

```text
beta = best-0.50, best-0.25, best, best+0.25, best+0.50
```

### 19.4 read-only performance metrics

必须记录：

```text
ATE RMSE
Rot RMSE
RPE t
RPE r
Final aligned error
50-frame mean ATE
100-frame mean ATE
200-frame mean ATE
per-chunk RMSE
worst-10 chunks
Sim(3) scale
trajectory length ratio
```

### 19.5 read-only promotion gate

相对 reference：

```text
Reference 1: no-control native
Reference 2: old_dyn_addclip read/hybrid
Reference 3: gg.qq.middle.low read-only
```

晋级 hybrid 的条件建议为：

```text
ATE <= gg.qq.middle.low read-only - 0.10
或
ATE <= gg.qq.middle.low read-only + 0.05 且 Rot 改善 >= 0.25 deg
或
ATE <= old_dyn_addclip read + 0.05 且 endpoint/final error 明显改善
```

不要只看 0.01m 的微小波动。

---

## 20. ACL2-4：VGGT4D-style factor/product evaluation

### 20.1 目标

验证用户指出的核心问题：当前 `middle` 信号只是 seed，还没有做 VGGT4D-style factor composition。

### 20.2 factor full evaluation 顺序

先 single factor，再 product：

```text
1. v4d.mean1
2. v4d.var1
3. v4d.mean2.low
4. v4d.mean3.high as S_rescue
5. v4d.var3.low as S_rescue confidence
6. v4d.product variants
```

### 20.3 factor 解释记录

每个 factor 需要回答：

| 问题 | 记录方式 |
|---|---|
| 是否改善 ATE？ | read-only full run |
| 是否改善 Rot？ | Rot RMSE / RPE r |
| 是否只是 old_dyn copy？ | Corr / IoU |
| 是否过度集中在 sky/road？ | vertical distribution |
| 是否碎片化？ | fragmentation |
| 是否可做 rescue？ | Corr with C_anchor, Rot improvement |

### 20.4 product 失败也有价值

如果 product 不如 single `mean2.low`，不要立即丢弃。需要看失败原因：

1. product 过稀疏：mass 太低；
2. shallow factor 引入语义噪声；
3. deep factor 方向错；
4. deep rescue 把真正 dynamic 保护了；
5. support 策略与 factor 不匹配。

对应修复：

```text
softened product: factor exponent 0.5
remove shallow factor
replace deep high by deep low-var only
support-specific product
gated product instead of multiplication
```

---

## 21. ACL2-5：static rescue 与 routing fusion

### 21.1 目标

在 single cue 找到强信号后，测试是否能降低 rotation/endpoint trade-off。重点不是继续 add/clip，而是 routing。

### 21.2 rescue first

对当前最强 `D_read`，测试：

```text
D_read only
D_read * (1 - 0.25 S_rescue)
D_read * (1 - 0.50 S_rescue)
D_read * (1 - 0.75 S_rescue)
D_read with top-rescue cap
```

必须同时看：

```text
ATE
Rot
Final error
worst chunk RMSE
trajectory endpoint
cue mass reduction
anchor collision reduction
```

### 21.3 routing fusion matrix

第一轮 routing 只允许少量组合：

| ID | $D_g$ | $D_a$ | $S$ | 目的 |
|---|---|---|---|---|
| `route.old_gg_rescue` | `old_dyn_addclip` | best attention cue | best static rescue | 主融合 |
| `route.exp_gg_rescue` | `explicit_dyn_only` | best attention cue | best static rescue | rotation-friendly |
| `route.old_v4d_rescue` | `old_dyn_addclip` | best v4d product | best static rescue | VGGT4D-style |
| `route.gg_factor_rescue` | none | best attention factor | best static rescue | pure attention |
| `route.old_attn_agree` | `old_dyn_addclip` | best attention cue | none | agreement only |

### 21.4 routing gate

Routing 候选进入 hybrid 前必须 read-only 达到：

```text
ATE <= best_single_attention_read + 0.05
且 Rot 不比 best_single_attention_read 差超过 0.20 deg
或
ATE 与 best_single_attention_read 持平但 Final error 改善 >= 0.3m
```

---

## 22. ACL2-6：hybrid-safe `probe_ttt_write` evaluation

### 22.1 目标

只把 ACL2-3/4/5 中 read-only 通过的 top candidates 放入 `probe_ttt_write`。不要在 cue 没稳定前做大量 TTT write policy sweep。

### 22.2 hybrid 固定策略

第一轮固定已有安全写入方式：

```text
commit mode = probe_ttt_write
write branch = branch0
write prior = current safe BL01 / existing branch0 policy
read cue = candidate D_read
```

只扫 read beta 附近，不同时扫 write beta，避免混淆。

### 22.3 hybrid beta

如果 read-only 最佳 beta 为 $\beta^*$，hybrid 只跑：

$$
\beta \in \{\beta^*-0.25,\beta^*,\beta^*+0.25\}
$$

如果结果非常敏感，再补：

$$
\beta \in \{\beta^*-0.5,\beta^*+0.5\}
$$

### 22.4 hybrid 记录

除了 performance metrics，还必须记录：

```text
TTT state relative diff vs probe_native
TTT update norm per branch
write prior mass
commit hash
branch0 update magnitude histogram
per-layer update norm
per-token update suppression ratio
read cue mass vs write update mass correlation
```

### 22.5 hybrid promotion gate

与当前 best hybrid 比较：

```text
ATE improvement >= 0.05m
或
ATE 不差于 best + 0.03m 且 Rot 改善 >= 0.30deg
或
ATE 不差于 best + 0.03m 且 Final error 改善 >= 0.5m
```

微小数值变化必须 repeat 一次确认。

---

## 23. ACL2-7：失败分析与跨序列 sanity

### 23.1 为什么必须做

Pipeline v2 已经证明短序列 gate 会 false positive。即便 full KITTI01 有提升，也可能是对单序列调参。因此最后必须做 failure analysis 和至少小规模跨序列 sanity。

### 23.2 failure gallery

对每个 top candidate，输出：

```text
best-improved chunks
worst-regressed chunks
high dynamic mass chunks
high anchor collision chunks
high rescue conflict chunks
high TTT update conflict chunks
```

每个 chunk 保存：

```text
RGB frame grid
D_read overlay
S_rescue overlay
old_dyn overlay
explicit_dyn overlay
attention factor panels
trajectory local segment
per-frame pose error curve
per-frame rot error curve
```

### 23.3 跨序列 sanity

至少选择：

```text
KITTI 01 full: 主调试序列
KITTI 03 or 04: 短 open sequence sanity
KITTI 05 or 06: loop / moderate dynamics sanity
KITTI 08 or 10: longer drift sanity
```

如果计算资源有限，先跑 read-only，不跑 hybrid。

跨序列通过标准：

```text
没有在其他序列出现 >5% ATE regression
Rot 不出现系统性恶化
cue mass/fragmentation 分布不崩
```

---

## 24. 可视化规范

### 24.1 全局 dashboard

每个候选都要生成一个 dashboard HTML 或 PNG grid，至少包含：

1. trajectory comparison；
2. per-frame ATE curve；
3. per-frame rotation error curve；
4. dynamic mass timeline；
5. anchor collision timeline；
6. cue fragmentation timeline；
7. support asymmetry timeline；
8. TTT update norm timeline；
9. top failure chunks thumbnails。

### 24.2 cue overlay grid

对固定 frames 输出：

```text
RGB
old_dyn
explicit_dyn
gg.qq.middle.low reference
candidate D_read
candidate S_rescue
D_read after rescue/routing
C_anchor
confidence
pose error
```

建议固定片段：

```text
frames 0-200
frames 200-320
frames 435-496
frames 800-1000
worst chunks from current best
new worst chunks from candidate
```

### 24.3 layer/head panel

必须为 per-head/per-layer cue 生成：

```text
layer x metric heatmap
head x metric heatmap
layer-head dynamic mass matrix
layer-head anchor collision matrix
layer-head fragmentation matrix
selected head cue overlays
```

### 24.4 temporal support panel

对同一 cue 比较：

```text
full
off246
near12
far612
past
future
noovlp
```

每个 panel 保持同一 color scale，避免视觉误判。

### 24.5 attention-map diagnostic panel

对少量 chunks 保存真实 attention map 子矩阵：

```text
query frame -> key frame mass matrix
attention entropy map
attention spatial std map
harmful key map
same-frame mass map
cross-frame mass map
```

### 24.6 routing panel

对 routing fusion 必须显示四象限：

```text
old_dyn high / attention high
old_dyn high / attention low
old_dyn low / attention high
old_dyn low / attention low
```

并叠加 static rescue 区域。这个图非常关键，可以直接判断 routing 是否只是 add/clip 的伪装。

---

## 25. 结果表规范

### 25.1 performance summary

`acl2_performance_summary.csv` 字段：

```text
run_id
cue_id
role
read_mode
commit_mode
beta
ATE_RMSE
Rot_RMSE
RPE_t
RPE_r
Final_error
ATE_50_mean
ATE_100_mean
ATE_200_mean
Sim3_scale
worst_chunk_id
worst_chunk_rmse
repeat_of
notes
```

### 25.2 cue quality summary

`acl2_cue_quality_summary.csv` 字段：

```text
cue_id
family
source
basis
stat
layers
support
headagg
norm
role
mean_D
mass_gt_05
coverage
fragmentation
anchor_collision
corr_conf
corr_old_dyn
corr_explicit_dyn
iou_old_dyn
temporal_flicker
support_asymmetry
vertical_top_mass
vertical_mid_mass
vertical_bottom_mass
promotion_status
promotion_reason
```

### 25.3 head/layer audit summary

`acl2_head_layer_summary.csv` 字段：

```text
cue_id
layer
head
mean_D
std_D
mass_gt_05
fragmentation
anchor_collision
corr_old_dyn
corr_anchor
corr_conf
head_rank_global
head_rank_chunk_stability
```

### 25.4 TTT/SWA memory summary

`acl2_memory_trace_summary.csv` 字段：

```text
run_id
chunk_id
layer
branch
apply_residual_mean
apply_residual_p90
pred_error_mean
pred_error_p90
update_norm_mean
update_norm_p90
write_prior_mean
write_prior_p90
state_relative_diff
swa_prev_mass_mean
swa_overlap_mass_mean
```

---

## 26. 第一批建议运行顺序

### 26.1 Batch 0：correctness

```text
ACL2_0A_no_control_cache_off
ACL2_0B_no_control_cache_on_qk
ACL2_0C_identity_hooks_cache_on
ACL2_0D_current_best_reproduce
```

### 26.2 Batch 1：full-run audit cache

只跑 native/probe，不控制：

```text
ACL2_1A_layer_singleton_qq_low_full_headmean
ACL2_1B_layer_sliding_qq_low_full_headmean
ACL2_1C_v4d_mapped_factors_off246_headmean
ACL2_1D_perhead_qq_low_selected_windows
ACL2_1E_attention_map_stats_selected_windows
ACL2_1F_special_token_stats
ACL2_1G_ttt_swa_memory_stats
```

### 26.3 Batch 2：layer/window promoted read-only

优先测试：

```text
gg.qq.low.g0.full.headmean.robustq
gg.qq.low.g3.full.headmean.robustq
gg.qq.low.g8.full.headmean.robustq
gg.qq.low.g13.full.headmean.robustq
gg.qq.low.g17.full.headmean.robustq
gg.qq.low.g2_6.off246.headmean.robustq
gg.qq.low.g12_17.off246.headmean.robustq
gg.qq.low.g13_15.off246.headmean.robustq
```

### 26.4 Batch 3：per-head promoted read-only

从 audit 选 top heads 后跑：

```text
gg.qq.low.bestwin.off246.headmax.robustq
gg.qq.low.bestwin.off246.headtop1.robustq
gg.qq.low.bestwin.off246.headtop2.robustq
gg.qq.low.bestwin.off246.headtop4.robustq
gg.qq.low.bestwin.off246.headagree.robustq
```

### 26.5 Batch 4：VGGT4D factors

```text
v4d.mean1
v4d.var1
v4d.mean2.low
v4d.mean3.high_as_rescue
v4d.var3.low_as_rescue_conf
v4d.product.m1_m2low_m3
v4d.product.v1_m2low_m3
v4d.product.m2low_m3
v4d.product.m1_m2low_m3_v3low
```

### 26.6 Batch 5：attention-map stats

```text
attn.row_entropy.bestwin
attn.spatial_std.bestwin
attn.temporal_crossmass.bestwin
attn.same_frame_mass.bestwin
attn.harmful_key.bestwin
attn.cross_frame_concentration.bestwin
```

### 26.7 Batch 6：rescue/routing

```text
bestD + deep_rescue_025
bestD + deep_rescue_050
bestD + deep_rescue_075
old_dyn_addclip route bestD with rescue
explicit_dyn route bestD with rescue
v4d_product route old_dyn with rescue
```

### 26.8 Batch 7：hybrid-safe

只对 top 3 read-only candidates：

```text
top1_read + probe_ttt_write beta around best
top2_read + probe_ttt_write beta around best
top3_read + probe_ttt_write beta around best
best_routing + probe_ttt_write
best_rescue + probe_ttt_write
```

---

## 27. 关键停止规则

为了避免再次进入无效大矩阵，设置以下 stop conditions。

### 27.1 停止某个 cue family

如果某个 family 满足以下任一条件，暂停：

```text
top 5 read-only ATE 都差于 gg.qq.middle.low read-only 0.30m 以上
且没有 Rot/endpoint 明显改善
```

或：

```text
audit 中 mass/fragmentation/coverage 全部不达标
```

或：

```text
cue 与 no-control 几乎等价，std(D) 太低
```

### 27.2 停止 product sweep

如果 product 全部失败，但 single factor 成功，则停止 product，转向 routing。

### 27.3 停止 hybrid sweep

如果 read-only 没有明确优于 reference，不进入 hybrid。

如果 hybrid 不优于 read-only，说明当前 cue 更适合 read correction，不要硬做 write policy。

---

## 28. 下一阶段的预期发现

### 28.1 可能出现的结果 A：非比例窗口明显优于 current middle

这会证明当前比例 `middle` 不是最优，下一步固定 best window，做 per-head 和 support 深挖。

### 28.2 可能出现的结果 B：per-head top-k 明显优于 headmean

这说明动态信号藏在少数 heads，未来 write policy 也应该考虑 head-specific，而不是 token-only。

### 28.3 可能出现的结果 C：v4d product 不如 mean2.low

这不代表 VGGT4D 思路错，而说明 KITTI01/LoGeR 中浅层语义或深层 prior 方向不同，应采用 routing/rescue，而不是乘法。

### 28.4 可能出现的结果 D：static rescue 改善 Rot/endpoint 但略损 ATE

这非常有价值，说明下一步 TTT write policy 可以用 rescue 保护 pose-sensitive anchors，read path 保持 stronger dynamic suppression。

### 28.5 可能出现的结果 E：attention-map stats 不如 Gram proxy

这说明 Gram proxy 已经抓住主要信号，真实 attention map 可以降级为 diagnostic，不需要大量保存。

### 28.6 可能出现的结果 F：TTT memory-side cue 与 read cue 弱相关

这意味着 read cue 和 write cue 必须分开。后续第二优先级 TTT write policy 应以 `W_hint` 为主，而不是直接复用 `D_read`。

---

## 29. 最终交付物

本阶段完成后应产生以下交付物：

```text
1. Attention Cue Library v2 implementation
2. full-run cue cache for all audited candidates
3. cue_quality_summary.csv
4. performance_summary.csv
5. layer/head/support audit dashboard
6. VGGT4D-style factor/product ablation report
7. static rescue/routing fusion report
8. top candidates for TTT write policy phase
```

最终报告必须明确回答：

1. LoGeR 的最优 internal-attention cue 来自哪几个层段？
2. 是否存在少数 heads 主导的 dynamic signal？
3. 真实 attention-map stats 是否提供额外价值？
4. VGGT4D-style product 在 LoGeR 上是否成立？哪个 factor 真正有贡献？
5. temporal support 应该用 full、fixed offset、past/future 还是 no-overlap？
6. special/register token 是否能预测 pose/rotation risk？
7. static rescue 是否能缓解 rotation/endpoint trade-off？
8. routing 是否优于 add/clip？
9. 哪些 cue 适合 read path，哪些 cue 只适合作 write policy？
10. 是否已经有足够稳定的 cue 进入第二优先级 TTT write policy？

---

## 30. 最短可执行版本

如果资源有限，建议先跑这个最短版本：

```text
Step 1: 导出 per-head q/k for selected layers
        g0, g3, g8, g13, g17, g2_6, g12_17, g13_15

Step 2: 对 qq.low 做 full-run audit
        support = full, off246, past, future, noovlp
        headagg = headmean, headmax, headtop2, headagree

Step 3: 跑 read-only full
        top 8 candidates only
        beta = 1.5, 2.0, 2.5, 3.0

Step 4: 跑 VGGT4D factors
        mean1, var1, mean2.low, mean3.high, var3.low, product

Step 5: 构造 static rescue
        D_best * (1 - lambda S_rescue)
        lambda = 0.25, 0.5, 0.75

Step 6: top 3 进入 probe_ttt_write
```

这套最短版本已经覆盖用户指出的最核心缺口：非比例 layer windows、VGGT4D-style factors、per-head cue、temporal support、static rescue 和 safe hybrid。

---

## 31. 总结

下一阶段的主线应正式命名为：

```text
Attention Cue Library v2
```

它的目标不是替换 `gg.qq.middle.low.robustq`，而是把它作为 seed，系统回答：

```text
最优层段是什么？
信号是否藏在少数 heads？
Gram factor 哪个真正有效？
真实 attention map 有没有额外价值？
temporal support 怎么选？
special/camera token 是否能做 reliability？
static rescue 能否修复 rotation/endpoint？
routing 是否优于 add/clip？
```

我的最终判断是：

> 你提出的十个方向都成立，其中优先级最高的是 **非比例 layer windows + VGGT4D-style factors + per-head audit + temporal support**。这四项会决定 `gg.qq.middle.low.robustq` 到底是偶然命中、局部最优，还是一个可以推广的核心机制。随后再做 static rescue 和 routing fusion，最后才把 top cue 放入 `probe_ttt_write`。只有这样，第二优先级 TTT write policy 才不会建立在未挖干净的 read cue 上。

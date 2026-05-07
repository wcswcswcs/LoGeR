# LoGeR 内部 Attention Cue Library 实验计划 v1

日期：2026-05-04  
项目阶段：Pipeline v2 / HMC 后续阶段  
当前优先级：**不引入 external motion cue，不接 RAFT / GMFlow / 新分割模型；第一优先级是重新挖掘 LoGeR 内部 attention cue，建立可审计 cue 库；第二优先级才是 TTT write policy。**

---

## 0. 本计划的核心判断

本阶段不继续把主要精力放在 external optical flow / epipolar residual 上，因为它会引入新模型、新误差源和新的工程依赖。现在更应该充分利用 LoGeR 自身已经产生的中间信号：frame attention、chunk/global attention、q/k token representation、TTT residual、SWA/TTT hook trace。目标不是马上设计一个新的单一 `best cue`，而是建立一套 **Attention Cue Library**，让每一种内部 cue 都有明确的来源、公式、可视化、质量审计、read-only 评价和 hybrid-safe 评价。

本计划的出发点是：

1. Pipeline v2 已经证明 frame-attention read-path correction 是有效方向，但当前 cue 没有被充分挖掘。
2. `old_dyn_addclip` 是当前最强结果，但它仍然是 geometry explicit cue 和已有 implicit cue 的简单融合，已经进入 `39.3-39.5 m` 平台。
3. VGGT4D 和 MUT3R 的共同思想不是“直接复制某个 attention map”，而是：预训练几何 Transformer 内部已经隐式编码动态/不稳定区域，需要用 layer-aware、direction-aware、distribution-aware 的方式把它挖出来，再只在早期 read path 做温和控制。
4. 对 LoGeR 来说，cue discovery 必须和 commit protocol 解耦。所有 read cue 的主评价都应该使用 `controlled output + probe/native commit`，不能用 contaminated controlled commit 判断 cue 好坏。

因此，本阶段的目标定义为：

> 在不引入外部模型的前提下，系统性挖掘 LoGeR 内部 attention / token / memory 信号，建立一个可复用、可审计、可视化完整的 cue library，并通过 commit-safe HMC 协议判断哪些 cue 适合作为 $D_{read}$，哪些 cue 只适合作为 reliability gate，哪些 cue 可以进入后续 TTT write policy。

---

## 1. 从 VGGT4D 和 MUT3R 提取的设计原则

### 1.1 VGGT4D 给 LoGeR 的启发

VGGT4D 的关键观察是：VGGT 的 global attention layers 已经隐含了动态线索，但直接看标准 attention map 会混入大量语义和纹理噪声。因此它不直接使用普通 cross-attention，而是用 Gram similarity 在同分布的 query-query、key-key、query-key 表征中放大动态造成的分布差异。VGGT4D 还强调 layer-wise cue 的互补性：浅层更偏语义显著性，中层更能表现 motion instability，深层更像空间先验或 outlier suppressor。

迁移到 LoGeR 时，不能只实现一个 `gram4d` 标量然后扫 beta。更合理的做法是把 VGGT4D 的思想拆成三个问题：

1. **哪些层给 semantic saliency？** 这类 cue 很可能会把车、人、树、路牌都点亮，不等于动态，但可以作为候选区域或 saliency prior。
2. **哪些层给 motion / geometry instability？** 这类 cue 更接近我们要的 $D_{read}$。
3. **哪些深层信号能抑制 false positive？** 深层 cue 不一定适合作为主动态图，但很适合做 reliability / veto / rescue。

VGGT4D 还有一个非常重要的控制原则：**不要全层 hard masking**。它的 ablation 显示 full masking 可能比 no mask 更差，而只在 early-stage layers suppress dynamic tokens 更合适。因此 LoGeR 的 attention cue 实验也应该优先控制 frame attention 的 early layers，不能一开始就把 cue 同时塞进 frame attention、chunk attention、TTT apply 和 TTT write。

### 1.2 MUT3R 给 LoGeR 的启发

MUT3R 的关键观察是：在 recurrent / stateful reconstruction 中，动态区域会污染 image tokens 和 state/memory tokens 的交互；模型内部 attention maps 可以生成 coarse motionness prior；然后用 attention-level gating 在早期层抑制动态内容，避免 motion interference 继续传播到深层几何表征。

迁移到 LoGeR 时，MUT3R 的启发主要有三点：

第一，motion cue 不必来自外部监督。LoGeR 的 frame attention、global attention 和 TTT residual 都是模型自己产生的内部诊断信号。

第二，attention gating 应该 direction-aware。不同路径的控制含义不同：

- frame self-attention 中，核心是避免静态 query 过度读取 dynamic key；
- state/memory update 中，核心是避免不稳定 image tokens 写入 memory；
- state/memory read 中，核心是避免 unstable query 从历史 memory 中获得过强、不匹配的修正。

第三，早期层抑制比后期层抑制更安全。早期层更容易携带 appearance / semantic / dynamic disturbance；深层已经形成较强几何结构，过度干预可能把模型推到 out-of-distribution 状态。

### 1.3 两篇工作的共同设计原则

两篇工作的共同原则可以总结为：

1. **internal-only**：优先从 frozen backbone 内部读信号，不额外训练、不引入外部模型。
2. **layer-aware**：浅层、中层、深层的 cue 含义不同，不能简单平均。
3. **direction-aware**：query-side、key-side、pairwise bias 的语义不同，不能混用。
4. **soft control**：优先用 soft bias / soft gate，不做全层 hard mask。
5. **early intervention**：先在早期 read path 抑制动态干扰，不直接重写深层几何和 memory。
6. **cue 与控制解耦**：先审计 cue 的质量，再把 cue 接入 HMC；不要靠最终 ATE 反推 cue 是否合理。
7. **mask / cue 不是最终目标**：最终目标是改善静态几何、pose 和长期 memory，一张 visually sharp 的动态图如果伤害 pose，也不能晋级。

---

## 2. LoGeR 当前可用接口与约束

### 2.1 LoGeR memory 结构对 cue 实验的约束

LoGeR 使用 hybrid memory：TTT fast weights 作为 compressed global memory，SWA 作为 lossless local memory。TTT 的 apply step 用历史 fast weights 修正当前 token，update step 把当前 chunk 信息写入下一步 fast weights。LoGeR block 内部包含 frame attention、SWA、TTT apply/update、chunk-wise bidirectional attention。这个结构决定了我们不能把“当前 read correction”和“未来 memory commit”混在一起。

HMC v2 已经证明：同一个 read cue 如果直接提交 controlled forward 的 TTT side effect，会显著污染未来 memory；正确评估 read cue 时，应使用 controlled output，但提交 probe/native memory。后续所有 attention cue 的主评价都必须遵循这个规则。

### 2.2 当前代码已经暴露的内部信号

当前 `GeometryOutput` 已经包含多类 attention / global cue 字段：

- `frame_attention_prior: [T,T]`
- `attn_dynamic_patch: [T,H_tok,W_tok]`
- `dyn4d_patch`
- `dyn4d_qq_mean_patch`
- `dyn4d_qk_var_patch`
- `dyn4d_kk_mean_patch`
- `global_q_raw_patchvec`
- `global_k_raw_patchvec`
- `global_q_raw_patchvec_layers`
- `global_k_raw_patchvec_layers`
- `dyn4d_global_layer_ids`
- `frame_attn_cosine_shallow`
- `frame_attn_cosine_deep`
- `frame_attn_cosine_avg`
- `frame_attn_key_cosine_l0`
- `frame_attn_key_cosine_l4`
- `frame_attn_key_cosine_shallow`
- `frame_attn_key_cosine_deep`
- `frame_attn_key_cosine_avg`
- `frame_attn_cosine_query_layers`
- `frame_attn_cosine_key_layers`
- `frame_attn_cosine_layer_ids`

当前 `DynamicCueExtractor` 已经把 explicit geometry cue 和 implicit attention cue 做融合，支持：

- `explicit`
- `implicit`
- `max`
- `soft_or`
- `avg`
- `addclip`
- `calibrated_soft_or`

当前 HMC 的 `read_cue_source` 已支持很多候选，包括 `dyn4d_patch`、`gram4d`、`qk_var`、`key_cosine_avg`、`query_cosine_avg`、`shallow_deep_disagree` 等。但 Phase F 结果说明，已有这些 cue 的 naive 使用没有超过 `old_dyn_addclip`。因此下一步不是简单再扫一遍已有 cue，而是要把 cue 提取、归一化、分层聚合、质量审计、可视化和 HMC 控制全部标准化，形成 cue library。

### 2.3 当前已知实验事实

当前最重要的参考结果如下：

| 结果 | ATE RMSE | 作用 |
|---|---:|---|
| LoGeR native | 41.7502 m | no-control reference |
| RFR-100 controlled commit | 41.0733 m | read cue 有效但 commit 污染 |
| RFR-100 probe_native commit | 39.7820 m | commit-safe read-only reference |
| D5-07 probe_ttt_write | 39.4903 m | read + safe branch0 write |
| F1_11 `old_dyn_addclip b=1.25` | 39.3103 m | 当前最好 |
| F1_01 `explicit_dyn_only b=1.0` | 39.4191 m | explicit geometry 分支很强，rotation 更好 |
| F3 `key_cosine_* read-only` | 39.7820 m | attention key cosine 有信号，但不够强 |
| F2 `dyn4d_patch / gram4d / qk_var` | 约 40.0-41.1 m | global/Gram naive 用法不够 |

这些结果意味着：

- attention cue 有信号，但现有抽取方式还不够；
- global/Gram cue 不能直接作为 primary read cue，需要更细 layer-aware 设计；
- key-cosine 有 read signal，值得继续拆解；
- attention cue 更可能作为 old_dyn 的 reliability / rescue / veto，而不一定直接替代 old_dyn；
- read cue 的评价必须始终使用 commit-safe protocol。

---

## 3. 总体实验目标

本阶段的目标不是直接刷一个 `<39.0 m`，而是建立一套能持续服务后续研究的 cue 库。最终希望得到三类输出：

1. **Primary read cue 候选**：可以直接作为 $D_{read}$ 接入 frame-attention read control，read-only 或 hybrid 指标明显优于现有 attention cue。
2. **Reliability cue 候选**：单独作为 $D_{read}$ 不强，但能判断 old_dyn 什么时候可信、什么时候 false positive。
3. **Static rescue / anchor cue 候选**：能识别不该被 suppress 的静态 anchor，用于降低 rotation / endpoint 损伤。

因此本阶段不是只找一张动态图，而是明确拆出：

$$
D_{read}(t,i): \text{当前 read path 应该抑制的 token}
$$

$$
R_{cue}(t,i): \text{当前 cue 本身的可信度或 agreement}
$$

$$
S_{static}(t,i): \text{应该保护的静态 anchor / reference token}
$$

$$
E_{write}(t,i): \text{后续 TTT write policy 可能使用的写入资格}
$$

本计划第一阶段只把 $D_{read}$、$R_{cue}$、$S_{static}$ 做扎实；$E_{write}$ 只记录，不作为主控制目标。

---

## 4. Attention Cue Library 的数据模型

### 4.1 CueCard

每一个 cue 都必须有一张 `CueCard`，记录它从哪里来、怎么计算、怎么归一化、预期含义是什么、失败模式是什么。

建议新增：

```python
@dataclass
class AttentionCueCard:
    name: str
    family: str
    source_path: str
    tensor_shape: str
    layer_group: str
    direction: str
    raw_formula: str
    normalization: str
    calibration: str
    expected_semantics: str
    expected_failure_modes: list[str]
    control_eligible: bool
    diagnostic_only: bool
```

每个 cue 在落盘时保存：

```text
cue_bank/
  chunk_0000/
    meta.json
    raw_maps.pt
    calibrated_maps.pt
    quality.json
    visualizations/
      frame_000_rgb_overlay.png
      frame_000_cue_grid.png
```

### 4.2 CueBank

`CueBank` 是一个 full-run 级别的集合，包含所有 chunk、所有 cue 的统计和可视化索引。

建议新增：

```python
@dataclass
class AttentionCueBank:
    sequence_id: str
    model_id: str
    chunk_size: int
    overlap: int
    patch_grid: tuple[int, int]
    cards: dict[str, AttentionCueCard]
    chunk_records: list[dict]
```

落盘结构建议：

```text
results/kitti01_hmc_v2/attention_cue_library_v1/
  config.yaml
  cue_cards.json
  cue_quality_per_chunk.jsonl
  cue_quality_summary.csv
  cue_correlation_summary.csv
  cue_bank_index.json
  visualizations/
    timeline_mass.png
    timeline_fragmentation.png
    cue_corr_heatmap.png
    cue_gallery_good_bad/
```

### 4.3 Cue 命名规范

命名必须能看出来源、层、方向、归一化方式和用途。建议格式：

```text
<family>.<source>.<direction>.<layer_group>.<stat>.<calib>
```

例子：

```text
fa.key.shallow.high.robustq
fa.key.mid.var.robustq
fa.key.shallow_deep.decay.robustq
gg.qk.mid.var.robustq
gg.vggt4d.smd.product.robustq
gg.deep.static_rescue.robustq
mix.old_addclip.fa_key_decay.lambda05
mix.old_veto.gg_static_rescue.lambda05
```

---

## 5. 统一归一化与校准协议

不同 cue 的数值尺度完全不同。如果不统一校准，很容易把“尺度大”误判成“cue 强”。因此所有 raw cue 都必须先经过统一归一化。

### 5.1 Per-frame robust quantile normalization

对每帧 cue map $X_t(i)$，定义：

$$
Q_t^{lo}=\operatorname{Quantile}_{i}(X_t(i), q_{lo})
$$

$$
Q_t^{hi}=\operatorname{Quantile}_{i}(X_t(i), q_{hi})
$$

$$
\mathcal N_{q}(X_t(i))=
\operatorname{clip}\left(
\frac{X_t(i)-Q_t^{lo}}{Q_t^{hi}-Q_t^{lo}+\epsilon},0,1
\right)
$$

默认：

$$
q_{lo}=0.50,\qquad q_{hi}=0.95
$$

该归一化适合把每帧 top dynamic/salient 区域拉开，但会丢掉全局绝对强度。因此需要同时记录 raw mean / raw std / raw quantile，不能只保存 normalized map。

### 5.2 Target-mass calibration

部分 attention cue 天然过稀疏或过稠密。为了让不同 cue 的控制能量可比，建议再提供一种 target-mass 校准：

$$
\tau_t = \operatorname{Quantile}_i(X_t(i), 1-m_{target})
$$

$$
\mathcal C_{m,s}(X_t(i)) = \sigma\left(\frac{X_t(i)-\tau_t}{s}\right)
$$

其中 $m_{target}$ 是希望 $D>0.5$ 的 token 比例，$s$ 是 soft threshold temperature。第一轮建议只测：

$$
m_{target}\in\{0.10,0.25,0.40\},\qquad s\in\{0.05,0.10\}
$$

注意：target-mass calibration 只用于控制实验；cue quality audit 必须同时看 uncalibrated 和 calibrated 两种统计。

### 5.3 Sign test

很多 internal cue 的方向不一定明确。例如 key-cosine 高值可能代表 salient/dynamic，也可能代表 stable/static support。因此每个新 cue 初次进入库时必须测试三种方向：

$$
D^{high}=\mathcal N_q(X)
$$

$$
D^{low}=\mathcal N_q(1-X)
$$

$$
D^{abs}=\mathcal N_q(|X-\operatorname{Median}(X)|)
$$

只有通过质量审计和 read-only full-run 的方向，才允许进入后续组合。

### 5.4 Temporal smoothing 只作为 ablation

由于 KITTI 01 是自车运动场景，same-pixel temporal smoothing 可能错误平滑 static background 的视差变化。因此 temporal smoothing 不是默认，只作为 ablation：

$$
\bar D_t = \lambda D_t + (1-\lambda)\bar D_{t-1}
$$

只测试：

$$
\lambda\in\{0.7,0.9\}
$$

如果 smoothing 降低 fragmentation 但伤害 ATE，则不保留。

---

## 6. Cue families 与具体公式

本节是 cue library 的核心。每个 family 都要先生成 raw map，再做统一归一化、质量审计、可视化，然后才允许接入 HMC。

### 6.1 Family A：Frame-Attention Response Cues，MUT3R-inspired

LoGeR 的 frame attention 是当前 Phase C/F 中最有效的 read-control 接入点。因此第一类 cue 直接从 frame attention 的 query/key token 表征中提取。

设第 $l$ 层、第 $h$ 个 head、第 $t$ 帧、第 $i$ 个 patch token 的 normalized query/key 为：

$$
\hat q_{l,h,t,i}=\frac{q_{l,h,t,i}}{\|q_{l,h,t,i}\|_2+\epsilon}
$$

$$
\hat k_{l,h,t,i}=\frac{k_{l,h,t,i}}{\|k_{l,h,t,i}\|_2+\epsilon}
$$

#### A1. Key response cue

当前代码已有近似实现，核心是 key token 与 frame 内 query centroid 的 cosine：

$$
r^K_{l,t,i}=\frac{1}{H}\sum_h
\left\langle
\hat k_{l,h,t,i},
\frac{1}{P}\sum_{j=1}^{P}\hat q_{l,h,t,j}
\right\rangle
$$

生成候选：

$$
D^{K-high}_{g,t,i}=\mathcal N_q\left(\operatorname{Mean}_{l\in\mathcal L_g} r^K_{l,t,i}\right)
$$

$$
D^{K-low}_{g,t,i}=\mathcal N_q\left(1-\operatorname{Mean}_{l\in\mathcal L_g} r^K_{l,t,i}\right)
$$

层组：

```text
g = l0, l4, shallow, middle, deep, all
```

第一轮必须跑：

```text
fa.key.l0.high
fa.key.l4.high
fa.key.shallow.high
fa.key.middle.high
fa.key.deep.high
fa.key.all.high
fa.key.shallow.low
fa.key.middle.low
fa.key.deep.low
```

目的：判断 F3 中 `key_cosine_avg/shallow/deep` 全部得到相同 ATE 的原因，是 cue 本身相似、归一化抹平、还是 HMC 控制能量相同。

#### A2. Query response cue

Query response 定义为：

$$
r^Q_{l,t,i}=\frac{1}{H}\sum_h
\left\langle
\hat q_{l,h,t,i},
\frac{1}{P}\sum_{j=1}^{P}\hat k_{l,h,t,j}
\right\rangle
$$

当前 F3 显示 query cosine 接近 no-control，但仍要进入 library，因为它可以作为 negative control。第一轮只做诊断，不进入主控制矩阵：

```text
fa.query.shallow.high
fa.query.deep.high
fa.query.shallow_deep.decay
```

#### A3. Shallow-to-deep decay cue

VGGT4D 和 MUT3R 都提示浅层更容易显现 dynamic/semantic disturbance，而深层会逐渐形成稳定几何。因此可以定义浅层显著但深层消失的 cue：

$$
D^{K-decay}_{t,i}=\mathcal N_q\left(
R^K_{shallow,t,i}\cdot
\max(0,R^K_{shallow,t,i}-R^K_{deep,t,i})
\right)
$$

其中：

$$
R^K_{g,t,i}=\operatorname{Mean}_{l\in\mathcal L_g} r^K_{l,t,i}
$$

这个 cue 的预期含义是：浅层很显著、深层不再稳定支持的 token 可能是动态干扰或语义 distractor。它可能比单纯 `key_cosine_avg` 更接近 VGGT4D 的 layer trend。

第一轮候选：

```text
fa.key.shallow_deep.decay.high
fa.key.l0_deep.decay.high
fa.key.l4_deep.decay.high
```

#### A4. Inter-layer instability cue

如果一个 token 在不同层的 attention response 波动很大，说明该 token 的解释不稳定。定义：

$$
D^{K-layerVar}_{t,i}=\mathcal N_q\left(
\operatorname{Var}_{l\in\mathcal L}
\left(r^K_{l,t,i}\right)
\right)
$$

这个 cue 不一定对应运动物体，但可能对应几何不确定、边界、反光或动态干扰。它需要重点看与 $C_{unc}$、$C_{occ}$、$C_{anchor}$ 的相关性。

#### A5. Dense attention dispersion cue，可选但建议实现

如果工程上可以在少数层导出 frame-attention 的 dense logits 或 softmax map，建议不要用 row mean，因为 softmax row mean 理论上接近常数 $1/P$，信息量很低。应该记录 entropy、peak、top-k concentration 和 logit variance。

设：

$$
p_{l,h,t,i,j}=\operatorname{softmax}_j\left(\frac{q_{l,h,t,i}^{\top}k_{l,h,t,j}}{\sqrt d}\right)
$$

定义 normalized entropy：

$$
H_{l,t,i}= -\frac{1}{H}\sum_h\frac{1}{\log P}\sum_{j=1}^{P}p_{l,h,t,i,j}\log(p_{l,h,t,i,j}+\epsilon)
$$

定义 peakiness：

$$
P_{l,t,i}=\frac{1}{H}\sum_h\max_j p_{l,h,t,i,j}
$$

定义 top-k mass：

$$
T^k_{l,t,i}=\frac{1}{H}\sum_h\sum_{j\in\operatorname{TopK}(p_{l,h,t,i,:})}p_{l,h,t,i,j}
$$

候选：

$$
D^{entropy}_{g}=\mathcal N_q(\operatorname{Mean}_{l\in\mathcal L_g}H_l)
$$

$$
D^{peak}_{g}=\mathcal N_q(\operatorname{Mean}_{l\in\mathcal L_g}P_l)
$$

这部分会增加 memory 开销，因此只在 selected chunks 或 selected layers 上先实现，不能一开始 full dense 全层导出。

---

### 6.2 Family B：Global / Chunk Gram Cues，VGGT4D-inspired

当前代码已经有 `dyn4d_qq_mean_patch`、`dyn4d_qk_var_patch`、`dyn4d_kk_mean_patch` 和 `dyn4d_patch`，但 Phase F 的 naive `dyn4d_patch` / `gram4d` 没有超过 old_dyn。下一步必须把它拆成 layer-aware cue，不再只用平均后的单图。

设 global/chunk attention 层 $l$ 的 normalized vectors 为：

$$
\hat q_{l,t,i},\quad \hat k_{l,t,i}
$$

对当前帧 $t$，定义 temporal window：

$$
\mathcal W_t=\{s: |s-t|\le r, s\ne t\}
$$

#### B1. Query-Query Gram mean

$$
G^{QQ}_{l,t,s}(i,j)=\left\langle \hat q_{l,t,i},\hat q_{l,s,j}\right\rangle
$$

$$
\mu^{QQ}_{l,t,i}=\frac{1}{|\mathcal W_t|P}\sum_{s\in\mathcal W_t}\sum_{j=1}^{P}G^{QQ}_{l,t,s}(i,j)
$$

低 $\\mu^{QQ}$ 可能表示该 token 的 query representation 与邻近帧不一致：

$$
D^{QQ-low}_{g,t,i}=\mathcal N_q\left(1-\operatorname{Mean}_{l\in\mathcal L_g}\mu^{QQ}_{l,t,i}\right)
$$

#### B2. Key-Key Gram mean

$$
G^{KK}_{l,t,s}(i,j)=\left\langle \hat k_{l,t,i},\hat k_{l,s,j}\right\rangle
$$

$$
\mu^{KK}_{l,t,i}=\frac{1}{|\mathcal W_t|P}\sum_{s\in\mathcal W_t}\sum_{j=1}^{P}G^{KK}_{l,t,s}(i,j)
$$

$$
D^{KK-low}_{g,t,i}=\mathcal N_q\left(1-\operatorname{Mean}_{l\in\mathcal L_g}\mu^{KK}_{l,t,i}\right)
$$

#### B3. Query-Key temporal variance

$$
G^{QK}_{l,t,s}(i,j)=\left\langle \hat q_{l,t,i},\hat k_{l,s,j}\right\rangle
$$

$$
\sigma^{QK}_{l,t,i}=\operatorname{Var}_{s\in\mathcal W_t,j\in[1,P]}G^{QK}_{l,t,s}(i,j)
$$

$$
D^{QK-var}_{g,t,i}=\mathcal N_q\left(\operatorname{Mean}_{l\in\mathcal L_g}\sigma^{QK}_{l,t,i}\right)
$$

根据 VGGT4D 的解释，中层的 variance 更可能是真正 motion instability，因此第一轮必须重点测试：

```text
gg.qk.middle.var.robustq
gg.qk.shallow.var.robustq
gg.qk.deep.var.robustq
```

#### B4. VGGT4D-style three-factor cue

不要直接用全层平均。应构造 shallow / middle / deep 三因子：

$$
S_{sem}(t,i)=\mathcal N_q(D^{QQ-low}_{shallow,t,i})
$$

$$
M_{inst}(t,i)=\mathcal N_q(D^{QK-var}_{middle,t,i})
$$

$$
R_{deep}(t,i)=\mathcal N_q(1-D^{KK-low}_{deep,t,i})
$$

其中 $S_{sem}$ 是浅层显著性，$M_{inst}$ 是中层不稳定性，$R_{deep}$ 是深层空间先验/可靠性。然后定义：

$$
D^{G-SMD}_{t,i}=\mathcal N_q\left(
S_{sem}(t,i)^{\alpha}\cdot
M_{inst}(t,i)^{\beta}\cdot
R_{deep}(t,i)^{\gamma}
\right)
$$

第一轮设置：

$$
(\alpha,\beta,\gamma)\in\{(1,1,1),(0.5,1,1),(0,1,1),(1,1,0.5)\}
$$

这里最重要的是比较：

- 只用 middle instability 是否更强；
- 加 shallow saliency 是否引入 semantic false positive；
- 加 deep reliability 是否减少 fragmentation 和 anchor collision。

#### B5. Deep static rescue cue

深层 Gram similarity 可能更适合识别 static support，而不是 dynamic。定义：

$$
S^{G-deep}_{static}(t,i)=\mathcal N_q\left(
\operatorname{Mean}_{l\in\mathcal L_{deep}}
\frac{\mu^{QQ}_{l,t,i}+\mu^{KK}_{l,t,i}}{2}
\right)
$$

这个 cue 不直接作为 $D_{read}$，而用于保护 old_dyn 高但 deep-static 也高的区域：

$$
D^{old-veto-deepStatic}(t,i)=D^{old}(t,i)\cdot
\left(1-\lambda S^{G-deep}_{static}(t,i)\right)
$$

第一轮建议：

$$
\lambda\in\{0.25,0.50,0.75\}
$$

该分支的目标是降低 rotation / endpoint 损伤，不一定降低 ATE。

---

### 6.3 Family C：Frame Affinity / Cross-frame Support Cues

LoGeR 已经导出 `frame_attention_prior: [T,T]` 和 `attn_dynamic_patch`。当前 Stage B 使用它们辅助 support selection 和 implicit dynamic fusion。下一步要单独建立 frame-support cue。

设 frame affinity 为：

$$
A^{frame}_{t,s}\in[0,1]
$$

设 token-level cross-frame support 为：

$$
U_{t,i}=\sum_{s\ne t}\bar A^{frame}_{t,s}\cdot \operatorname{sim}(z_{t,i},c_s)
$$

其中 $c_s$ 是第 $s$ 帧的 frame/key centroid，$z_{t,i}$ 可以取 $q$ 或 $k$。

动态/不稳定 cue：

$$
D^{support-low}_{t,i}=\mathcal N_q(1-U_{t,i})
$$

temporal variance cue：

$$
D^{support-var}_{t,i}=\mathcal N_q\left(
\operatorname{Var}_{s\in\mathcal W_t}
\left(\operatorname{sim}(z_{t,i},c_s)\right)
\right)
$$

第一轮候选：

```text
support.q.low
support.k.low
support.q.var
support.k.var
support.old_addclip.low_lambda05
```

这类 cue 的风险是：在自车运动和大视差场景中，远处/边缘/低纹理区域也可能低 support。因此它必须重点看与 $C_{unc}$、confidence 和 sky/horizon 的相关性。

---

### 6.4 Family D：Layer Evolution / Representation Instability Cues

MUT3R 的 PCA 可视化说明，动态干扰会在浅层表现为 fragmented activation，随着层数推进才变得更稳定。LoGeR 可以用无需外部模型的方式定义 representation instability。

如果可以导出每个 layer 的 patch feature $x_{l,t,i}$，定义 normalized layer displacement：

$$
\Delta^{feat}_{l,t,i}=\frac{\|x_{l+1,t,i}-x_{l,t,i}\|_2}{\|x_{l,t,i}\|_2+\epsilon}
$$

定义 shallow instability：

$$
D^{feat-inst}_{t,i}=\mathcal N_q\left(
\operatorname{Mean}_{l\in\mathcal L_{shallow}}\Delta^{feat}_{l,t,i}
\right)
$$

定义 late stabilization：

$$
S^{feat-stable}_{t,i}=\mathcal N_q\left(
1-\operatorname{Mean}_{l\in\mathcal L_{deep}}\Delta^{feat}_{l,t,i}
\right)
$$

再定义 shallow-unstable but deep-stable 的 cue：

$$
D^{feat-evolve}_{t,i}=D^{feat-inst}_{t,i}\cdot S^{feat-stable}_{t,i}
$$

这部分不是第一批必须跑 full 的 cue，因为需要额外导出 features，可能增加内存；但它非常适合作为可视化诊断，帮助解释 attention cue 的层间变化。

---

### 6.5 Family E：TTT Residual / Memory-side Diagnostic Cues

虽然本阶段重点是 read cue，但 TTT residual 完全来自 LoGeR 内部，不属于 external cue，应该进入 cue library 的 diagnostic 部分，为第二优先级 TTT write policy 做准备。

对 TTT layer cache 中的 apply prediction 和 value target 定义：

$$
r^{TTT}_{l,i}=\frac{\|f_{W_m}(k_{l,i})-v_{l,i}\|_2}{\|v_{l,i}\|_2+\epsilon}
$$

聚合成 token map：

$$
D^{TTT-res}_{i}=\mathcal N_q\left(\operatorname{Mean}_{l\in\mathcal L_{TTT}}r^{TTT}_{l,i}\right)
$$

这个 cue 的解释不是“动态”，而是“当前 token 对现有 TTT memory 来说难以预测 / update-needed”。它可能更适合作为 write policy 的 $E_{write}$ 或 update strength cue，而不是 read suppression cue。第一阶段只记录、可视化和做 correlation，不作为主 read cue，除非它在 full-run audit 中明显表现好。

---

### 6.6 Family F：Attention × old_dyn 融合 / Reliability Cues

现阶段最可能突破的平台，不一定是让 attention cue 直接替代 old_dyn，而是让 attention cue 帮 old_dyn 判断 false positive / false negative。

设当前最强 old cue 为：

$$
D^{old}=D^{old\_dyn\_addclip}
$$

设某个 attention cue 为：

$$
D^{att}
$$

#### F1. Addclip fusion

$$
D^{mix-addclip}=\operatorname{clip}(D^{exp}+\lambda D^{att},0,1)
$$

这个与当前 `old_dyn_addclip` 思路一致，但要换不同 attention cue、不同 layer-aware cue。

#### F2. Agreement modulation

$$
D^{mix-agree}=\operatorname{clip}\left(
D^{old}\cdot \left(1+\lambda(2D^{att}-1)\right),0,1
\right)
$$

如果 attention cue 高，则增强 old_dyn；如果 attention cue 低，则减弱 old_dyn。

#### F3. Static rescue / veto

设 attention-derived static cue 为 $S^{att}_{static}$：

$$
D^{mix-veto}=D^{old}\cdot (1-\lambda S^{att}_{static})
$$

该 cue 的目标不是找到更多 dynamic，而是减少 old_dyn 对可靠 static anchor 的误伤。

#### F4. Reliability routing

定义 attention reliability：

$$
R^{att}=\operatorname{clip}\left(1-\left|D^{att}-D^{old}\right|,0,1\right)
$$

也可以用 layer agreement：

$$
R^{layer}=1-\operatorname{Var}_{g\in\{shallow,middle,deep\}}D^{att}_g
$$

融合：

$$
D^{mix-route}=R^{att}D^{old}+(1-R^{att})D^{exp}
$$

这个分支适合测试“attention 只做 reliability gate”的价值。

---

## 7. 必须记录的指标

### 7.1 每个 cue 的基础统计

每个 cue、每个 chunk、每个 frame 都必须记录：

```text
mean
std
min / max
q01 / q05 / q10 / q50 / q90 / q95 / q99
mass_D_gt_001
mass_D_gt_050
mass_D_gt_075
range_q90_q10
raw_mean
raw_std
calibrated_mean
```

其中：

$$
\operatorname{mass}_{\tau}=\frac{1}{TP}\sum_{t,i}\mathbf 1[D_{t,i}>\tau]
$$

### 7.2 Cue informativeness

要避免接近常数的 cue 进入控制。记录：

$$
I_{range}=Q_{0.90}(D)-Q_{0.10}(D)
$$

$$
I_{entropy}= -\frac{1}{N}\sum_i \left(D_i\log(D_i+\epsilon)+(1-D_i)\log(1-D_i+\epsilon)\right)
$$

通过建议：

```text
I_range > 0.10
mass_D_gt_050 不长期接近 0 或 1
至少 90% chunks 有非退化分布
```

### 7.3 Anchor collision

高动态 cue 如果撞到稳定 anchor，通常会伤害 rotation / endpoint。记录：

$$
\operatorname{AnchorCollision}(D)=
\frac{\sum_i D_i C_{anchor,i}}{\sum_i D_i+\epsilon}
$$

建议 gate：

```text
primary read cue: AnchorCollision <= 0.08
strict cue: AnchorCollision <= 0.05
```

### 7.4 Fragmentation

fragmentation 用来识别过碎的 cue。建议在 patch grid 上记录边界复杂度：

$$
\operatorname{Frag}(D)=
\frac{1}{T}\sum_t
\frac{\sum_{y,x}|B_{t,y,x}-B_{t,y+1,x}|+|B_{t,y,x}-B_{t,y,x+1}|}{H_{tok}W_{tok}}
$$

其中：

$$
B_{t,y,x}=\mathbf 1[D_{t,y,x}>0.5]
$$

建议 gate：

```text
primary read cue: Frag <= 0.06
relaxed diagnostic: Frag <= 0.10
```

### 7.5 与旧 cue / geometry cue 的关系

每个 attention cue 都要记录与以下 maps 的 Pearson / Spearman correlation：

```text
old_dyn_addclip
explicit_dyn_only
implicit_dyn_only
C_stat
C_dyn
C_occ
C_unc
C_anchor
G_write_geo
confidence
1-confidence
```

公式：

$$
\rho(D,X)=\frac{\operatorname{Cov}(D,X)}{\sigma_D\sigma_X+\epsilon}
$$

还要记录 old_dyn IoU：

$$
\operatorname{IoU}(D,D^{old})=
\frac{|\mathbf 1[D>0.5]\cap\mathbf 1[D^{old}>0.5]|}
{|\mathbf 1[D>0.5]\cup\mathbf 1[D^{old}>0.5]|+\epsilon}
$$

解释规则：

- IoU 很高且 ATE 没提升：只是 old_dyn 的变体，价值有限；
- IoU 中等且 ATE 提升：可能提供新信息；
- IoU 很低但质量 gate 通过：适合作为 reliability / rescue，而不一定直接作为 primary cue；
- 与 $C_{unc}$ 高相关但与 ATE 不一致：可能只是 uncertainty detector。

### 7.6 Temporal / overlap consistency

由于不用 external flow，temporal consistency 只能做粗检查。记录 same-grid 的相邻帧差异：

$$
\operatorname{TempDiff}(D)=\frac{1}{T-1}\sum_{t=2}^{T}\operatorname{Mean}_i|D_{t,i}-D_{t-1,i}|
$$

对 overlap frames，记录相邻 chunk 中同一全局 frame 的 cue 差异：

$$
\operatorname{OverlapDiff}=\operatorname{Mean}_i|D^{chunk\ m}_{t,i}-D^{chunk\ m+1}_{t',i}|
$$

这项对 HMC 很重要，因为 chunk boundary 的 cue 抖动会直接影响后续 memory。

### 7.7 HMC hook control energy

对每个 controlled run，记录 frame attention bias 能量。当前 pair mode 的 bias 可写成：

$$
B_{i,j}=\beta\log\left(\max\left(1-(1-D_i)D_j,\epsilon\right)\right)
$$

key mode：

$$
B_{i,j}=\beta\log\left(\max(1-D_j,\epsilon)\right)
$$

query mode 当前代码不是 softmax bias，而是 residual output gate：

$$
y_i = y_i^{before} + (y_i^{after}-y_i^{before})\cdot \operatorname{clip}(1-\beta D_i,0,1)
$$

必须记录：

```text
mean_abs_bias
max_abs_bias
mean_abs_query_gate_delta
max_abs_query_gate_delta
raw_frame_bias_energy
beta_effective
beta_was_clipped
active layer count
hook trace count
```

### 7.8 Read-only geometry metrics

每个 controlled full run 都记录：

```text
ATE RMSE
Rot RMSE
RPE t
RPE r
Sim(3) scale
final aligned error
50-frame mean ATE
100-frame mean ATE
200-frame mean ATE
worst 5 chunks by RMSE
best 5 chunks by delta
worst 5 chunks by delta
```

其中 window ATE 用于判断提升来自全局 scale 还是局部窗口稳定性。

### 7.9 Commit side-effect metrics

每个 read cue 都必须记录 controlled state 与 probe/native state 的 TTT side effect：

$$
\Delta W_l=\frac{\|W^{controlled}_{l,next}-W^{probe}_{l,next}\|_F}{\|W^{probe}_{l,next}\|_F+\epsilon}
$$

记录：

```text
mean_ttt_side_effect
max_ttt_side_effect
side_effect_by_layer
side_effect_by_branch
commit_mode
commit_source
```

如果某 cue read-only 变好但 side effect 巨大，后续只能在 `probe_native` / `probe_ttt_write` 中使用，不能直接 controlled commit。

---

## 8. 必须做的可视化

### 8.1 Per-frame cue grid

每个候选 cue 至少在固定 frames 上输出网格图：

```text
RGB
old_dyn_addclip
explicit_dyn_only
implicit_dyn_only
candidate_raw
candidate_calibrated
candidate_mask_D_gt_05
C_anchor
C_unc
confidence
```

每张图必须有同一 colormap range，不能每个 cue 自动拉伸后无法比较。建议同时显示 raw 和 calibrated。

### 8.2 Layer-wise attention panel

对 frame attention cue，输出：

```text
key_l0
key_l4
key_shallow
key_middle
key_deep
key_shallow_deep_decay
query_shallow
query_deep
layer_variance
```

对 global Gram cue，输出：

```text
QQ_low_shallow
QQ_low_middle
QQ_low_deep
KK_low_shallow
KK_low_middle
KK_low_deep
QK_var_shallow
QK_var_middle
QK_var_deep
G_SMD_product
Deep_static_rescue
```

这张图的目标是直接回答：LoGeR 的 layer trend 是否类似 VGGT4D / MUT3R。

### 8.3 Full-run timeline dashboard

每个 cue 输出 full sequence 时间线：

```text
chunk index vs mean_D
chunk index vs mass_D_gt_050
chunk index vs anchor_collision
chunk index vs fragmentation
chunk index vs corr_D_old_dyn
chunk index vs corr_D_unc
chunk index vs overlap_diff
chunk index vs local 100-frame ATE delta
```

把指标和 ATE delta 放在同一 dashboard，帮助看出：哪些 chunk 是 cue 过稀疏、过碎、撞 anchor 或与 old_dyn disagreement 导致失败。

### 8.4 Correlation heatmap

对所有 cue 输出一个 pairwise correlation heatmap：

```text
old_dyn_addclip
explicit_dyn_only
implicit_dyn_only
fa.key.*
fa.query.*
gg.qq.*
gg.kk.*
gg.qk.*
gg.smd.*
support.*
```

目的：去重。高度相关且指标没有差异的 cue 不需要重复进入 full-run 控制。

### 8.5 Good / bad chunk gallery

对每个 full controlled run，自动生成：

```text
Top-5 improved chunks
Top-5 hurt chunks
Top-5 high anchor collision chunks
Top-5 high fragmentation chunks
Top-5 high disagreement with old_dyn chunks
```

每个 chunk gallery 包含：

```text
RGB selected frames
candidate cue overlay
old_dyn overlay
C_anchor overlay
trajectory local segment
chunk-level metrics table
```

这一步非常关键，因为 Phase C/F 已经证明短程 gate 可能 false positive，只有看 full-run failure gallery 才能知道 cue 是怎么坏的。

### 8.6 Trajectory visualization

每个晋级 full run 输出：

```text
full KITTI trajectory: GT vs no-control vs old_best vs candidate
aligned error over frame index
rotation error over frame index
local 50/100/200 frame ATE curves
endpoint zoom-in
```

如果 ATE 变好但 rotation / endpoint 明显变坏，必须标记为 `ATE-only tradeoff`，不能直接作为主候选。

### 8.7 PCA / feature evolution visualization，可选

参考 MUT3R 的分析，对 selected chunks 导出 shallow/mid/deep feature PCA intensity map：

```text
layer 0 / 2 / 4 / 8 / 12 / 16 feature PCA intensity
candidate cue overlay
old_dyn overlay
```

这不是主指标，但能帮助解释：attention cue 是否真的在早期捕捉动态干扰，还是只是语义 saliency。

---

## 9. 实验阶段设计

### Phase 0：Correctness 与 export sanity

目的：确保新增 cue export 不改变模型输出。

必须跑：

```text
P0-00 no-control native
P0-01 identity hooks
P0-02 export_attention_cue_bank but no control
P0-03 old best F1_11 reproduction
```

通过条件：

```text
P0-00 ATE == 41.7502 m within numerical tolerance
P0-01 identity output byte-identical or max pose diff = 0
P0-02 no-control output unchanged
P0-03 F1_11 old_dyn_addclip b=1.25 reproducible around 39.3103 m
all hook trace counts nonzero where expected
no NaN in raw/caled cue maps
```

记录：

```text
hmc_state_hash.jsonl
hmc_hook_identity_check.json
exported_tensor_shapes.json
cue_card_sanity.json
runtime_memory_summary.json
```

### Phase 1：Full-run cue bank export，不做控制

目的：一次性导出 KITTI 01 全序列所有内部 attention cue，避免每个 cue 控制实验都重新 forward。

输出：

```text
cue_bank/chunk_XXXX/raw_maps.pt
cue_bank/chunk_XXXX/calibrated_maps.pt
cue_quality_per_chunk.jsonl
cue_cards.json
visualization index
```

每个 chunk 至少保存：

```text
old_dyn_addclip
explicit_dyn_only
implicit_dyn_only
fa.key.l0/l4/shallow/middle/deep/all
fa.query.shallow/deep/all
gg.qq.low.shallow/middle/deep
gg.kk.low.shallow/middle/deep
gg.qk.var.shallow/middle/deep
gg.smd.product variants
gg.deep.static_rescue
support.low / support.var
```

通过条件：

```text
100% chunks 有 cue records
每个 cue 有 raw + calibrated
每个 cue 有 per-frame q-stat
所有 raw tensor shape 与 patch_grid 对齐
cue_bank 可被后续 audit 脚本单独读取
```

### Phase 2：Cue quality audit，不跑 controlled full

目的：先剔除明显退化 cue，防止 Phase C v3/v4 那类短程 false positive。

对每个 cue 生成 summary：

```text
mean mass_D_gt_050
coverage of valid chunks
mean anchor_collision
mean fragmentation
corr with old_dyn
corr with C_unc / C_occ / C_anchor / confidence
non-degenerate chunk ratio
overlap consistency
```

Primary read cue 初选条件：

```text
0.05 <= mean mass_D_gt_050 <= 0.65
nondegenerate_chunk_ratio >= 0.90
mean anchor_collision <= 0.08
mean fragmentation <= 0.06
mean overlap_diff <= 0.25
not purely confidence/uncertainty: |corr_D_unc| <= 0.75 unless intentionally uncertainty cue
```

Reliability / rescue cue 初选条件可以不同：

```text
static_rescue cue: corr with C_anchor positive, dynamic_mass can be high or low
reliability cue: layer agreement stable, old_dyn disagreement localized
negative controls: allowed to fail but must be recorded
```

Phase 2 的输出不是 ATE，而是三张名单：

```text
primary_read_candidates.csv
reliability_candidates.csv
static_rescue_candidates.csv
rejected_cues.csv with reason
```

### Phase 3：Read-only HMC smoke，只用于工程，不作为晋级依据

目的：确认 cue 接入 HMC 不报错，hook energy 合理。

只跑 64 / 256 frames：

```text
read_path_only
commit_mode=probe_native
frame_bias_mode=pair/key
read_layer_mode=early
beta=1.0
```

注意：Phase C 已经证明 64/256 win 不可靠，因此这里不按 ATE 晋级，只按工程条件判断：

```text
no NaN
hook bias energy not zero unless cue intentionally zero
pose output finite
runtime acceptable
cue map correctly projected to D_tok
```

### Phase 4：Read-only full evaluation，主判断 cue 本身价值

这是本阶段最重要的实验。所有候选都使用：

```text
current output = controlled forward
future memory = probe_native commit
TTT write = disabled
```

固定主配置：

```text
hmc_mode=read_path_only
commit_mode=probe_native
enable_frame_read_control=true
enable_ttt_apply_control=false
enable_swa_read_control=false
enable_chunk_read_control=false
read_layer_mode=early
frame_bias_mode=pair and key
beta_frame in {0.75, 1.0, 1.25}
```

首先跑少量 anchor baselines：

| Run | 目的 |
|---|---|
| R0 no-control | LoGeR native reference |
| R1 old_dyn_calibrated_soft_or read-only | CM02 reference |
| R2 old_dyn_addclip read-only | 当前 best read cue 的 read-only 对照 |
| R3 key_cosine_avg read-only | F3 reference |
| R4 dyn4d_patch read-only | F2 reference |

然后跑 Phase 2 晋级候选。每个 cue 至少跑：

```text
pair beta=1.0
pair beta=1.25
key beta=1.0
```

如果一个 cue 的 pair/key 方向差异很大，再补：

```text
query beta=0.5
protected_pair beta=1.0
```

Read-only full 晋级条件：

```text
strong primary: ATE < old_dyn_addclip_readonly by >= 0.10 m
useful primary: ATE < 39.7820 m and cue quality better than key_cosine reference
reliability candidate: alone not strong but improves old_dyn fusion in Phase 5
rotation guard: Rot RMSE not worse than F1_11 by >0.30 deg
endpoint guard: final error not worse than F1_11 by >1.0 m
```

### Phase 5：Attention × old_dyn fusion full evaluation

如果 Phase 4 没有直接 primary cue 超过 old_dyn，这是预期内的；下一步测试 attention 作为 reliability / rescue。

固定：

```text
read cue base = old_dyn_addclip or explicit_dyn_only
commit_mode = probe_native for read-only fusion
write disabled in first round
```

组合公式只测试少量，不做大矩阵：

#### 5.1 Addclip

$$
D=\operatorname{clip}(D^{exp}+\lambda D^{att},0,1)
$$

$$
\lambda\in\{0.25,0.50,0.75\}
$$

#### 5.2 Agreement modulation

$$
D=\operatorname{clip}\left(D^{old}\left(1+\lambda(2D^{att}-1)\right),0,1\right)
$$

$$
\lambda\in\{0.25,0.50\}
$$

#### 5.3 Static rescue

$$
D=D^{old}(1-\lambda S^{att}_{static})
$$

$$
\lambda\in\{0.25,0.50,0.75\}
$$

#### 5.4 Reliability routing

$$
D=R^{att}D^{old}+(1-R^{att})D^{exp}
$$

第一轮只允许最多 12 个 fusion full runs，按 Phase 2/4 结果选择最有解释力的 cue，不做无目的扫参。

晋级条件：

```text
ATE < F1_11 39.3103 m by >= 0.05 m, or
ATE roughly equal but Rot RMSE improves by >= 0.15 deg, or
ATE roughly equal but final error improves by >= 0.5 m
```

其中 `roughly equal` 定义为：

```text
ATE within +0.05 m of F1_11
```

### Phase 6：Hybrid-safe evaluation，与 branch0 write 组合

只有 Phase 4/5 通过的 cue 才进入 hybrid-safe evaluation。使用：

```text
current output = controlled forward
future memory = probe_ttt_write
TTT write = branch0 safe write, same as D5/F1 reference
```

第一轮只跑：

```text
best_read_candidate + BL01 branch0 write beta_write=1.0
best_read_candidate + BL01 branch0 write beta_write=1.25
best_fusion_candidate + BL01 branch0 write beta_write=1.0
best_fusion_candidate + residual_reliability write beta_write=1.25
```

通过条件：

```text
main success: ATE < 39.0 m
useful success: ATE < 39.3103 m and rotation/final not worse
balanced success: ATE within 0.05 m of 39.3103 but Rot RMSE improves >= 0.20 deg
```

如果 Phase 6 没有过，不说明 cue 没价值，可能是 write policy 未匹配。此时把 cue 标记为 `read-useful/write-unresolved`，进入第二优先级 TTT write policy。

---

## 10. 第一批实验候选矩阵

第一批 full-run 不应该太大。建议从 cue bank 里筛选后最多跑 30 个 full reads。初始候选如下。

### 10.1 Anchor baselines

| Name | Formula / Source | Mode | Purpose |
|---|---|---|---|
| `old_dyn_calibrated_soft_or` | existing D5 reference | read-only | CM02 reference |
| `old_dyn_addclip` | existing F1 best cue | read-only + hybrid | current best cue reference |
| `explicit_dyn_only` | Stage-B geometry residual | read-only + hybrid | strong geometry diagnostic |
| `implicit_dyn_only` | existing implicit attention | read-only | weak reference |
| `key_cosine_avg` | existing F3 cue | read-only | attention positive reference |
| `dyn4d_patch` | existing F2 cue | read-only | global Gram reference |

### 10.2 Frame attention candidates

| Name | Expected role | First control |
|---|---|---|
| `fa.key.l0.high.robustq` | shallow saliency | pair beta=1.0 |
| `fa.key.l4.high.robustq` | shallow-mid saliency | pair beta=1.0 |
| `fa.key.middle.high.robustq` | possible instability | pair beta=1.0 |
| `fa.key.deep.high.robustq` | possible static or residual dynamic | pair beta=1.0 |
| `fa.key.shallow_deep.decay.robustq` | dynamic/semantic disturbance | pair beta=1.0/1.25 |
| `fa.key.layerVar.robustq` | instability | pair beta=1.0 |
| `fa.key.deep.low.robustq` | low deep support | key beta=1.0 |
| `fa.query.shallow.high.robustq` | negative diagnostic | pair beta=1.0 |

### 10.3 Global Gram candidates

| Name | Expected role | First control |
|---|---|---|
| `gg.qk.middle.var.robustq` | primary motion instability | pair beta=1.0 |
| `gg.qq.middle.low.robustq` | cross-frame inconsistency | pair beta=1.0 |
| `gg.kk.middle.low.robustq` | key inconsistency | key beta=1.0 |
| `gg.smd.product.a1b1g1.robustq` | VGGT4D-style combined cue | pair beta=1.0 |
| `gg.smd.product.a0b1g1.robustq` | middle+deep without shallow semantic | pair beta=1.0 |
| `gg.deep.static_rescue.robustq` | static protection | no direct read; fusion only |

### 10.4 Support / layer evolution candidates

| Name | Expected role | First control |
|---|---|---|
| `support.k.low.robustq` | low cross-frame support | pair beta=1.0 |
| `support.k.var.robustq` | support instability | pair beta=1.0 |
| `feat.shallow.instability.robustq` | optional diagnostic | no direct control first |
| `feat.evolve.robustq` | optional diagnostic | no direct control first |

### 10.5 Fusion candidates

| Name | Formula | Purpose |
|---|---|---|
| `mix.exp_add_fa_decay_l05` | $\operatorname{clip}(D^{exp}+0.5D^{fa-decay},0,1)$ | replace current implicit branch |
| `mix.old_agree_fa_decay_l05` | $D^{old}(1+0.5(2D^{fa-decay}-1))$ | attention agreement |
| `mix.old_veto_gg_deep_static_l05` | $D^{old}(1-0.5S^{gg-deep}_{static})$ | anchor rescue |
| `mix.old_route_gg_smd` | $R^{att}D^{old}+(1-R^{att})D^{exp}$ | reliability routing |
| `mix.exp_add_gg_qkvar_l05` | $\operatorname{clip}(D^{exp}+0.5D^{gg-qkvar},0,1)$ | VGGT4D middle instability |

---

## 11. 结果目录与文件规范

建议建立：

```text
results/kitti01_hmc_v2/attention_cue_library_v1/
  00_correctness/
  01_cue_bank_export/
  02_quality_audit/
  03_readonly_smoke/
  04_readonly_full/
  05_old_attention_fusion/
  06_hybrid_safe/
```

每个 run 目录必须包含：

```text
config.yaml
command.txt
git_status.txt or code_fingerprint.json
01.txt
kitti_benchmark.log
hmc_state_hash.jsonl
hmc_probe_summary.jsonl
hmc_control_summary.jsonl
cue_quality_per_chunk.jsonl
cue_quality_summary.json
hook_effect_summary.json
trajectory_diagnostics.json
window_ate_summary.json
visualizations/
```

每个 cue bank 目录必须包含：

```text
cue_cards.json
cue_bank_index.json
cue_quality_summary.csv
cue_correlation_summary.csv
rejected_cues.csv
primary_read_candidates.csv
reliability_candidates.csv
static_rescue_candidates.csv
```

---

## 12. 具体工程实现计划

### 12.1 新增模块

建议新增文件：

```text
loger/pipeline/attention_cue_library.py
```

职责：

```text
1. 从 GeometryOutput 和 CueOutput 中读取 raw attention / geometry maps
2. 构建所有 cue cards
3. 执行 raw map -> calibrated map
4. 计算质量指标
5. 保存 cue bank
6. 为 HMC 提供 read_cue_source = attnlib:<cue_name> 的查询接口
```

核心函数：

```python
build_attention_cue_bank(geo, cue, config) -> CueBankChunk
calibrate_cue(raw_map, method, params) -> calibrated_map
compute_cue_quality(cue_map, refs) -> dict
select_cue(cue_bank_chunk, name) -> torch.Tensor
```

### 12.2 修改 `geometry_backbone.py`

需要确保以下内容稳定导出：

```text
frame_attn_cosine_query_layers
frame_attn_cosine_key_layers
frame_attn_cosine_layer_ids
global_q_raw_patchvec_layers
global_k_raw_patchvec_layers
dyn4d_global_layer_ids
```

如果当前只导出 averaged maps，则要补充 per-layer raw maps，至少能区分 shallow/middle/deep。不要只保存 `shallow/deep/avg` 三张图，否则无法判断 layer trend。

### 12.3 修改 `hybrid_memory_controller.py`

新增 `read_cue_source` 解析：

```text
attnlib:<cue_name>
```

当 source 以 `attnlib:` 开头时，从当前 chunk 的 cue bank record 读取 calibrated patch map，并投影到 token：

```python
read_patch = attention_cue_bank.select(cue_name, chunk_id)
D_tok = token_from_patch_values(geo, read_patch, special_value=0.0)
```

所有现有质量指标继续记录，包括：

```text
anchor_collision
fragmentation
old_dyn_iou
old_dyn_coverage
old_dyn_recall
corr_D_unc
corr_D_occ
corr_D_conf
corr_D_old_dyn
```

### 12.4 新增工具脚本

建议新增：

```text
tools/build_attention_cue_bank.py
tools/audit_attention_cue_bank.py
tools/visualize_attention_cue_bank.py
tools/run_attention_cue_candidates.py
tools/summarize_attention_cue_runs.py
```

#### `build_attention_cue_bank.py`

功能：跑 full sequence probe/no-control，导出 raw 和 calibrated cue。

输入：

```text
--sequence kitti01
--result_dir .../01_cue_bank_export
--chunk_size 32
--overlap 3
--export_layers all
--cue_config configs/attention_cue_library_v1.yaml
```

输出：

```text
cue_bank/chunk_XXXX/*.pt
cue_cards.json
cue_quality_per_chunk.jsonl
```

#### `audit_attention_cue_bank.py`

功能：读取 cue bank，不跑模型，输出质量报告和晋级名单。

输出：

```text
cue_quality_summary.csv
cue_correlation_summary.csv
primary_read_candidates.csv
reliability_candidates.csv
static_rescue_candidates.csv
rejected_cues.csv
```

#### `visualize_attention_cue_bank.py`

功能：生成 cue grid、layer panel、timeline、failure gallery。

#### `run_attention_cue_candidates.py`

功能：根据 Phase 2 晋级名单自动生成 controlled read-only full runs。

#### `summarize_attention_cue_runs.py`

功能：汇总 full ATE、Rot、RPE、window ATE、hook energy、cue quality。

---

## 13. 实验命令模板

以下是计划中的命令模板，实际 flags 需按当前 `run_pipeline_abc_v2.py` 调整。

### 13.1 Correctness

```bash
python run_pipeline_abc_v2.py \
  --sequence 01 \
  --hmc_mode native \
  --result_dir results/kitti01_hmc_v2/attention_cue_library_v1/00_correctness/P0_native
```

```bash
python run_pipeline_abc_v2.py \
  --sequence 01 \
  --hmc_mode identity_hooks \
  --result_dir results/kitti01_hmc_v2/attention_cue_library_v1/00_correctness/P0_identity
```

### 13.2 Cue bank export

```bash
python tools/build_attention_cue_bank.py \
  --sequence 01 \
  --chunk_size 32 \
  --overlap 3 \
  --export_frame_attention_layers all \
  --export_global_gram_layers all \
  --cue_config configs/attention_cue_library_v1.yaml \
  --result_dir results/kitti01_hmc_v2/attention_cue_library_v1/01_cue_bank_export
```

### 13.3 Cue audit

```bash
python tools/audit_attention_cue_bank.py \
  --cue_bank results/kitti01_hmc_v2/attention_cue_library_v1/01_cue_bank_export \
  --output_dir results/kitti01_hmc_v2/attention_cue_library_v1/02_quality_audit
```

### 13.4 Read-only full run

```bash
python run_pipeline_abc_v2.py \
  --sequence 01 \
  --hmc_mode read_path_only \
  --hmc_commit_mode probe_native \
  --enable_frame_read_control \
  --frame_bias_mode pair \
  --read_layer_mode early \
  --read_cue_source attnlib:fa.key.shallow_deep.decay.high.robustq \
  --beta_frame 1.0 \
  --cue_bank results/kitti01_hmc_v2/attention_cue_library_v1/01_cue_bank_export \
  --result_dir results/kitti01_hmc_v2/attention_cue_library_v1/04_readonly_full/fa_key_decay_pair_b100
```

### 13.5 Hybrid-safe run

```bash
python run_pipeline_abc_v2.py \
  --sequence 01 \
  --hmc_mode hybrid \
  --hmc_commit_mode probe_ttt_write \
  --enable_frame_read_control \
  --frame_bias_mode pair \
  --read_layer_mode early \
  --read_cue_source attnlib:mix.old_veto_gg_deep_static_l05 \
  --beta_frame 1.25 \
  --hmc_write_score_source stage_d \
  --cue_bank results/kitti01_hmc_v2/attention_cue_library_v1/01_cue_bank_export \
  --result_dir results/kitti01_hmc_v2/attention_cue_library_v1/06_hybrid_safe/mix_old_veto_deepstatic_b125
```

---

## 14. Promotion gates 与停止规则

### 14.1 Cue library gate

一个 cue 被允许进入 read-only full run，需要满足：

```text
1. shape 正确，无 NaN/Inf
2. nondegenerate_chunk_ratio >= 0.90
3. full-run cue quality audit 通过对应类型 gate
4. 可视化上不是明显纯 sky/horizon/road edge false positive
5. 与已有 cue 的相关性不是完全重复，或重复但预期作为 sanity/reference
```

### 14.2 Read-only gate

一个 cue 被认为是 read-useful，需要满足至少一个：

```text
ATE < 39.7820 m
或 ATE 明显优于已有 key_cosine reference
或 ATE 接近 old_dyn_addclip_readonly 且 rotation/final 更好
```

如果 cue 单独 ATE 不强，但可视化和质量审计显示它能识别 old_dyn 的 false positive，则进入 reliability/fusion 分支。

### 14.3 Hybrid gate

一个 cue 被认为可以进入下一阶段 TTT write policy，需要满足至少一个：

```text
ATE < 39.3103 m
或 ATE within +0.05 m of 39.3103 且 Rot RMSE 改善 >= 0.20 deg
或 ATE within +0.05 m of 39.3103 且 final error 改善 >= 0.5 m
```

### 14.4 Stop rules

停止继续扫某个 cue family 的条件：

```text
1. 质量 audit 大面积失败：mass 退化、fragmentation 高、anchor collision 高；
2. 3 个代表 full runs 均明显差于 key_cosine reference；
3. 只在 64/256 smoke 上好，full-run fail；
4. 与 old_dyn 高度相关且没有 ATE/Rot/final 任一改善；
5. 需要过多特殊阈值才能看起来有效。
```

停止继续做无目的 fusion 的条件：

```text
1. 12 个 fusion full runs 后仍无 ATE/Rot/final 任一实质改善；
2. improvement 只来自加大 beta 或 mass，而非 cue 质量；
3. failure gallery 显示主要问题是同一类 false positive，且 attention cue 无法修复。
```

---

## 15. 预期分析方式

本阶段结束时，不应该只汇报一个 best ATE，而应该输出一份 cue library report，至少回答：

1. LoGeR 的 frame attention 是否存在类似 MUT3R 的 motionness signal？
2. 这个 signal 在 key-side 还是 query-side 更可靠？
3. 它主要出现在浅层、中层还是深层？
4. LoGeR 的 global/chunk Gram cue 是否存在类似 VGGT4D 的 shallow / middle / deep 分工？
5. 之前 `gram4d` 失败是因为公式不对、层组不对、归一化不对，还是该 cue 在 LoGeR/KITTI 上确实不强？
6. attention cue 更适合直接做 $D_{read}$，还是更适合做 $R_{cue}$ / $S_{static}$？
7. 哪些 cue 可以降低 `old_dyn_addclip` 的 anchor collision 或 fragmentation？
8. 哪些 cue 能改善 rotation / final error，即使 ATE 不变？
9. 哪些 cue 值得带入第二优先级 TTT write policy？

---

## 16. 风险与应对

### 16.1 风险：attention cue 主要是语义 saliency，不是 motion

应对：必须拆分 shallow saliency、middle instability、deep static prior；不要直接把 shallow high response 当 dynamic。

### 16.2 风险：Gram cue 计算太耗显存

应对：优先用 per-layer in-place / per-frame window aggregation，不 materialize 全局 $P\times P$ 矩阵；先在 selected layers 和 selected chunks 做 dense debug，full run 只保存压缩统计。

### 16.3 风险：target-mass calibration 人为制造提升

应对：每个 cue 同时保存 raw quality 和 calibrated quality；最终报告必须包含 hook bias energy，确保不是单纯控制能量差异导致指标变化。

### 16.4 风险：短序列结果误导

应对：64/256 只做工程 smoke，不做科学晋级。任何 cue 必须过 full-run quality audit 和 full KITTI01 read-only。

### 16.5 风险：ATE 改善但 rotation / endpoint 恶化

应对：promotion gate 必须同时检查 Rot RMSE、final error、window ATE。ATE-only tradeoff 不直接晋级 write policy。

### 16.6 风险：attention cue 与 old_dyn 高度重复

应对：重复 cue 可以保留为 sanity/reference，但不进入大规模 full matrix。真正有价值的是：低 IoU 但质量好，或能作为 static rescue / reliability gate 的 cue。

---

## 17. 第一轮结束后的决策树

第一轮实验完成后，按下面方式决策：

```text
Case A: 有 attention-only cue read-only 明显优于 old_dyn_addclip_readonly
    -> 进入 hybrid-safe evaluation
    -> 进入第二优先级 TTT write policy，探索 branch/layer-specific write

Case B: attention-only cue 不强，但 fusion / rescue 改善 F1_11
    -> 固定该 fusion 作为 new read cue
    -> 第二优先级只研究 TTT write policy，不再大扫 cue

Case C: attention cue 只能改善 rotation/final，ATE 不变
    -> 标记 balanced cue
    -> 后续 write policy 中作为 protection cue 使用

Case D: 所有 attention cue 均不超过 key_cosine reference
    -> 结论不是 attention 没用，而是 LoGeR 当前可导出的 attention statistic 不足
    -> 下一步再考虑更深的 feature evolution / dense attention entropy，而不是回到 external cue

Case E: Gram family 质量差但 frame-key family 有信号
    -> 保留 MUT3R-style 路线，暂停 VGGT4D-style Gram full run

Case F: Gram middle-var / deep-static 对 failure chunks 有解释力
    -> 不直接作为 primary read cue，转为 reliability/static rescue cue
```

---

## 18. 本阶段最小可交付物

本阶段完成后，至少应该交付：

1. `attention_cue_library.py`
2. `cue_cards.json`
3. `cue_quality_summary.csv`
4. `cue_correlation_summary.csv`
5. `primary_read_candidates.csv`
6. `reliability_candidates.csv`
7. `static_rescue_candidates.csv`
8. full-run visual dashboard
9. read-only full result table
10. hybrid-safe result table，仅包含晋级 cue
11. failure gallery
12. 一份最终分析报告，明确哪些 cue 进入第二优先级 TTT write policy

---

## 19. 推荐第一批运行顺序

为了避免跑太散，建议严格按下面顺序：

1. `P0` correctness：native / identity / export no-control / F1_11 reproduction。
2. `P1` cue bank export：先只导出 KITTI01 全序列内部 cue。
3. `P2` quality audit：生成 candidate lists，不跑 full control。
4. `P3` read-only smoke：每个 family 只选 1-2 个代表，确认工程无误。
5. `P4` read-only full：最多 30 个 full runs。
6. `P5` fusion full：最多 12 个 full runs。
7. `P6` hybrid-safe：最多 6 个 full runs。
8. 汇总报告：不按最高 ATE 排名，而按 `primary read / reliability / static rescue / rejected` 四类归档。

---

## 20. 最终一句话

本阶段要做的不是“再试几个 attention cue”，而是把 LoGeR 内部 attention 表征变成一个可复用的科学工具：每个 cue 都有来源、公式、层组、方向、归一化、质量审计、可视化、read-only 指标和 hybrid-safe 指标。只有这样，后续 TTT write policy 才不会再次变成盲目调参。

# KITTI 01 Hybrid Memory Controller Phase C v2：细粒度 Read-Path Gate 实验方案

> 目标：在新的 **Hybrid Memory Controller, HMC** pipeline 上，重新设计 Phase C 的 read-path-only gate。Phase C v1 已经说明 frame-attention early control 有真实但偏弱的信号，TTT-apply read control 当前不安全；因此 Phase C v2 不再简单扩大强度，也不急着进入 Phase D 组合实验，而是先把代码正确性、信号来源、层级调度、attention bias 方向、reference protection、SWA/chunk attention 可用性逐项拆开验证。
>
> 本文档使用 Typora 友好的公式格式，所有公式使用 `$...$` 或 `$$...$$`。

---

## 0. 当前结论与为什么需要 Phase C v2

Phase C v1 的结论不能简单说“完全失败”。更准确地说，它找到了一个弱信号，但这个信号还不够强、不够稳定，暂时不能支撑进入多控制器组合的 Phase D。

当前新的 HMC pipeline 已经完成了 Phase B / G2 工程级 identity-hook parity：frame attention、SWA read、TTT apply、chunk attention 的 identity hooks 在 no-control 条件下能够保持 A3 no-control trajectory 一致。这说明“hook 被挂上去以后是否会无意改变 LoGeR”这一点暂时通过。但 Phase C v1 的 read-path-only 结果显示，真正有用的控制仍然很有限。

Phase C v1 最重要的结果是 frame attention early control：

| Run | Control | Strength | Layer mode | ATE RMSE | Rot RMSE | Verdict |
|---|---|---:|---|---:|---:|---|
| A3 HMC v2 native | none | n/a | n/a | 41.7502 m | 8.9928 deg | reference |
| BL01 TTT write | branch0 dyn MP-01 | n/a | all | 41.3665 m | 8.9490 deg | previous best |
| RFR-025 | frame attention | beta=0.25 | early | 41.1524 m | 9.1499 deg | useful |
| RFR-050 | frame attention | beta=0.50 | early | 41.1323 m | 9.1113 deg | useful |
| RFR-100 | frame attention | beta=1.00 | early | 41.0733 m | 9.0158 deg | best Phase C |
| RFR-125 | frame attention | beta=1.25 | early | 41.2165 m | 9.0406 deg | over-strength starts hurting |
| RFR-150 | frame attention | beta=1.50 | early | 41.2125 m | 9.0046 deg | worse than beta=1.0 |
| RFR-200 | frame attention | beta=2.00 | early | 41.5707 m | 8.9167 deg | too strong |

这个结果说明：frame attention read-path control 确实比旧 TTT-only branch0 result 更进一步，从 41.3665 m 推到了 41.0733 m。但是它仍然没有过 Phase C 原计划中的 `<41.0 m` gate，而且更强的 beta 并没有继续改善，说明失败不是 under-strength，而是当前 cue、层级、bias 方向或者保护策略还不够准。

TTT-apply read control 则明显不安全：

| Run | Control | Strength | Layer mode | ATE RMSE | Rot RMSE | Verdict |
|---|---|---:|---|---:|---:|---|
| R-TTTA-010 | TTT apply | rho=0.10 | all | 42.6125 m | 8.6284 deg | bad ATE, better rotation |
| R-TTTA-020 | TTT apply | rho=0.20 | all | 43.5418 m | 8.0282 deg | bad ATE |
| R-TTTA-030 | TTT apply | rho=0.30 | all | 44.8062 m | 7.7155 deg | fails |

这组结果说明：TTT apply gate 当前又回到了旧实验中常见的模式——rotation 变好，但 translation / full ATE 明显变坏。也就是说，它不是简单地“更强抑制动态就更好”，而是在破坏 LoGeR 对全局尺度、平移连续性或 long-context memory 的使用。

所以 Phase C v2 的核心判断是：

$$
\text{Phase C v1 找到了 frame-attention read-path 的弱信号，但还没有找到足够可靠的 read-path control signal。}
$$

Phase C v2 需要更细，不是继续扫 beta，而是要回答下面五个问题：

1. 新 HMC pipeline 的 correctness 是否在 full KITTI 01 上彻底通过，包括 LoGeR 和 LoGeR*，且不能通过旧代码路径绕过？
2. RFR-100 的收益到底来自 frame attention 的哪个层段、哪种 bias 方向、哪种 dynamic map？
3. 当前 dynamic map 是不是太 dense / 太 noisy，导致 beta 强一点就误伤 road/building/guardrail 等 translation-useful token？
4. SWA read 和 chunk-wise bidirectional attention 是否真的没有信号，还是因为 v1 使用的控制图太粗，导致尚未进入有效区间？
5. TTT apply 是否应该暂时退出 Phase C，或者只保留非常保守的 reference-protected gate？

Phase C v2 的目标不是立即追最终主模型，而是建立一个足够可信的 read-path-only signal gate。只有当 read-path-only 控制能稳定超过 RFR-100，并且最好达到 `<40.5 m`，才值得进入 Phase D 的 read-path + TTT branch0 write 组合。

---

## 1. Phase C v2 的总体原则

### 1.1 不能再把 `<41.0 m` 当成“成功”

`<41.0 m` 只能证明 read-path control 不是噪声级别。它不能作为主模型成功标准，因为当前 LoGeR native / unity replay 在 KITTI 01 上已经是 41.6–41.8 m 量级，BL01 和 RFR-100 的改善都还只是 0.x m 级别。

Phase C v2 使用分层判断标准：

| 等级 | ATE 目标 | 含义 |
|---|---:|---|
| Correctness gate | 复现 LoGeR / LoGeR* | 证明新 pipeline 没有破坏 baseline |
| Minimal read-path signal | `<41.0 m` | 证明 read-path-only 确实有效，不只是旧 BL01 附近抖动 |
| Phase C v2 pass | `≤40.8 m` | 至少比 RFR-100 有 0.2 m 左右的明确提升 |
| Strong Phase C signal | `≤40.5 m` | 值得进入 Phase D 组合实验 |
| Main candidate prerequisite | `≤40.0 m` | read-path 本身已经足够强，组合后有希望进入 `<38.0 m` |

换句话说，Phase C v2 的正式目标不是“勉强低于 41.0”，而是尽量把 single-path read controller 推到：

$$
ATE_{PhaseC\ v2} \le 40.5\text{ m}
$$

如果所有细化后仍然只能在 41.0 m 附近徘徊，那么 read-path-only 信号的上限就很可能不足，后续必须引入更强的 internal motion cue、semantic protection、optical flow / epipolar cue，或者进入学习式 reliability gate。

### 1.2 Phase C v2 只做 read-path，不做 TTT write 组合

Phase C v2 仍然保持：

```text
hybrid_memory_mode = read_path_only
stage_c_mode = none
TTT write = native / unity equivalent
```

原因是我们必须先弄清楚 read-path 自己有没有强信号。如果 frame attention、SWA、chunk attention 的单路径信号都没有过 gate，就直接和 TTT branch0 write 组合，会让归因变混乱：一旦结果提升或下降，很难判断是 read-path 变好了，还是 TTT branch0 的旧信号在撑着。

### 1.3 Phase C v2 不再只扫强度

RFR-100 最好、RFR-125 / RFR-150 / RFR-200 变差，说明不是 under-strength。继续把 beta 从 1.0 扫到 3.0、4.0 只会重复同一个失败模式。Phase C v2 的改进维度应当是：

$$
\text{signal source} \times \text{layer schedule} \times \text{bias direction} \times \text{dynamic mass control} \times \text{reference protection}
$$

每一组实验都要能回答一个明确假设，而不是把很多参数混在一起。

---

## 2. Phase C v2-G：新 HMC pipeline correctness gate

Phase C v2 的第一步必须是重新验证代码正确性，而且验证对象必须是 **新 HMC pipeline**。不能通过旧 `geometry_eval_mode`、旧 exact external geometry orchestrator、旧 TTT-only replay path 或旧 run_pipeline_abc 逻辑绕过。

### 2.1 为什么要重新做 correctness gate

Phase C v2 要改 read path hook、dynamic map、SWA trace、chunk attention bias、reference protection 等。如果 baseline 本身没有在新 HMC pipeline 上复现 LoGeR / LoGeR*，后续任何 read-path 提升都不可信。

LoGeR 的原始 hybrid block 是：

$$
\text{per-frame attention}
\rightarrow
\text{SWA}
\rightarrow
\text{TTT apply/update}
\rightarrow
\text{chunk-wise bidirectional attention}
$$

所以新 HMC pipeline 的 baseline 复现必须真实经过这些 hook 的注册、两遍 chunk 协议和 controlled-pass 输出路径。只有这样才能证明 controller 没有暗中改变 LoGeR 的状态传递。

### 2.2 禁止的捷径

本 gate 明确禁止以下做法作为正式 baseline：

```text
1. 禁止用旧 geometry_eval_mode 直接输出 LoGeR trajectory 作为 HMC baseline。
2. 禁止用旧 TTTWriteController / unity_replay 结果冒充 HMC no-control。
3. 禁止只跑 Pass 1 probe，然后把 Pass 1 trajectory 当成 controlled output。
4. 禁止 identity hooks 没有真正注册到 frame/SWA/TTT/chunk 路径，只是在 CLI 上标记 identity。
5. 禁止 Pass 1 provisional memory 被提交给 Pass 2。
6. 禁止 LoGeR* 走旧 SE3 merge 路径而不经过 HMC two-pass controlled output。
```

正式结果必须满足：

```text
Pass 1: probe from committed H_m, collect traces, do not commit state
Pass 2: controlled pass from the same committed H_m, identity/no-control hooks active
Output: Pass 2 geometry only
Commit: Pass 2 memory only
Merge: HMC pipeline 的正式 output merge，不使用旧绕行结果
```

### 2.3 Gate G0：HMC two-pass no-control 复现 LoGeR

实验目的：证明新 HMC two-pass pipeline 在没有任何控制时，可以复现 LoGeR 在 KITTI 01 上的已知结果。

实验设置：

```text
checkpoint = ckpts/LoGeR/latest.pt
config = ckpts/LoGeR/original_config.yaml
chunk_size = 32
chunk_overlap = 3
window_size = 32
overlap_size = 3
reset_every = 5
geometry_edge_rtol = 0.0
stage_c_mode = none
hmc_mode = two_pass
control = none
output = Pass 2 only
commit = Pass 2 only
```

需要记录：

```text
ATE / Rot / RPE t / RPE r
trajectory row count
Pass 1 state hash before/after
Pass 2 input state hash
Pass 2 output state hash
whether probe provisional state was discarded
hook registry summary
```

判断标准：

| 检查项 | 通过标准 |
|---|---|
| KITTI frames | 1101 / 1101 matched |
| ATE vs previous HMC/A3 native reference | `|ATE - 41.7502| < 0.15 m` |
| ATE vs paper LoGeR 01 | `|ATE - 41.64| < 0.30 m` |
| trajectory diff vs known native trajectory | mean translation diff `<0.10 m`，max `<0.20 m` |
| Pass 1 memory commit | must be false |
| Pass 2 input memory | exactly committed $H_m$ before Pass 1 |
| hook registry | frame/SWA/TTT/chunk hooks either absent or identity-noop, no accidental bias |

如果 G0 失败，不能进入任何 Phase C v2 模型实验。优先检查 two-pass state isolation、reset boundary、overlap merge、SWA history 和 TTT state 是否被 Pass 1 污染。

### 2.4 Gate G1：HMC two-pass no-control 复现 LoGeR*

实验目的：证明新 HMC pipeline 对 LoGeR* 的 SE3 overlap alignment / reset 机制也没有破坏。

实验设置：

```text
checkpoint = ckpts/LoGeR_star/latest.pt
config = ckpts/LoGeR_star/original_config.yaml
se3 = true
chunk_size = 64
chunk_overlap = 3
window_size = 64
overlap_size = 3
reset_every = 5
geometry_edge_rtol = 0.0
stage_c_mode = none
hmc_mode = two_pass
control = none
output = Pass 2 only
commit = Pass 2 only
```

判断标准：

| 检查项 | 通过标准 |
|---|---|
| KITTI frames | 1101 / 1101 matched |
| ATE vs previous LoGeR* reproduction | `|ATE - 47.9793| < 0.20 m` |
| ATE vs paper LoGeR* 01 | `|ATE - 47.91| < 0.35 m` |
| reset / SE3 alignment logs | 与 LoGeR* 原始配置一致 |
| Pass 1 memory commit | must be false |
| output source | Pass 2 only |

LoGeR* 的 ATE 比 LoGeR 高，但 rotation 更好，这是原论文和既有复现都出现的现象。因此 LoGeR* 复现不是为了优化指标，而是为了确保新 pipeline 没有把 SE3 alignment / reset path 弄坏。

### 2.5 Gate G2：identity hooks 全路径等价性

实验目的：证明所有控制 hook 被真实挂载后，在 identity 参数下不会改变输出。

实验设置：

```text
frame_attn_hook = enabled, beta = 0
swa_read_hook = enabled, beta = 0
ttt_apply_hook = enabled, rho = 0 or gate = 1
chunk_attn_hook = enabled, beta = 0
reference_protection = enabled but no-op
D_tok = any synthetic map, but beta/rho makes it identity
```

对 LoGeR 和 LoGeR* 都要跑 full KITTI 01。

判断标准：

| 检查项 | 通过标准 |
|---|---|
| identity-hook ATE vs HMC no-control | `<0.05 m` preferred，hard limit `<0.10 m` |
| trajectory mean translation diff | `<0.03 m` preferred，hard limit `<0.08 m` |
| enabled layer count | 日志显示 hooks 真实启用 |
| attention bias norm | beta=0 时 exactly zero 或数值误差 `<1e-7` |
| feature delta norm | identity hook 前后 selected layer output diff `<1e-5` 相对误差 |

如果 G2 失败，不允许进入 Phase C v2 signal experiments。

### 2.6 Gate G3：synthetic control-map 局部正确性

实验目的：验证 hook 不只是 identity 正确，而且非 identity 时作用方向正确。这个 gate 不看最终 ATE，只看局部统计。

构造三个 synthetic dynamic maps：

1. `center_square`：每帧中心 20% patch 的 $D_i=1$，其他为 0。
2. `frame_ramp`：第 $t$ 帧所有 patch 的 $D(t)=t/(T-1)$。
3. `checkerboard`：patch grid 上交替 $D=0/1$。

对 frame attention hook，记录：

$$
M_{highD}^{before} = \sum_{q}\sum_{k:D_k>0.8} A_{qk}^{before}
$$

$$
M_{highD}^{after} = \sum_{q}\sum_{k:D_k>0.8} A_{qk}^{after}
$$

判断标准：

| Hook | synthetic map | 期望 |
|---|---|---|
| frame key-only | center_square | attention mass to center high-D keys decreases |
| frame query-only | center_square | high-D query output update norm decreases |
| SWA prev-key | frame_ramp | current chunk reads late/high-D prev tokens less |
| chunk key-only | checkerboard | high-D key mass decreases only in enabled layers |

如果 synthetic control map 的作用方向不正确，先修 hook 维度、token reshape、layer id、attention mask broadcast。

---

## 3. Phase C v2-A：Frame-attention fine gate

Phase C v1 里 frame attention early control 是唯一接近成功的 read-path-only 信号。因此 Phase C v2 的第一主线不是 SWA，也不是 TTT apply，而是更细地拆 frame attention。

### 3.1 核心假设

Phase C v1 的 RFR-100 没有过 `<41.0 m`，但已经明显优于 BL01。更强 beta 变差，说明 current dynamic map 有用但误伤。Phase C v2 的 frame-attention 假设是：

$$
\text{RFR-100 的瓶颈不是强度，而是 cue source、bias direction、layer schedule 和 reference protection。}
$$

我们要找的是：

$$
\text{哪一种 frame-attn control 能减少 static-to-dynamic contamination，同时不破坏 translation-useful static evidence。}
$$

### 3.2 需要比较的 bias 方向

Phase C v1 的 frame attention control 大概率使用了某种 pairwise self-attention gating。Phase C v2 需要把 bias 方向拆开，因为不同方向代表完全不同的语义。

#### A. key-only dynamic suppression

$$
B_{qk}^{key} = \beta \log(1-D_k+\epsilon)
$$

含义：所有 query 都少看 dynamic key。这个简单直接，但风险是 dynamic object 自己也无法看自己，可能造成动态区域 feature 变得异常，进而影响周围 token。

#### B. static-query to dynamic-key suppression

$$
B_{qk}^{pair} = \beta \log(1-(1-D_q)D_k+\epsilon)
$$

含义：只有 static query 少看 dynamic key；dynamic query 仍然可以看 dynamic key。这个更接近 MUT3R 的 self-attention gating 直觉，也更可能安全。

#### C. query-only dynamic weakening

$$
B_{qk}^{query} = \beta \log(1-D_q+\epsilon)
$$

含义：dynamic query 的 read/update 被弱化。这个可能减少动态 token 输出对后续模块的影响，但也可能让动态区域变成 bad feature source。它应当只作为诊断，不作为第一候选。

#### D. protected pairwise suppression

$$
B_{qk}^{prot} = \beta \log(1-(1-D_q^{prot})D_k^{prot}+\epsilon)
$$

其中：

$$
D_i^{prot} = D_i \cdot (1-P_{ref,i}) \cdot (1-P_{safe,i})
$$

$P_{ref}$ 保护 register / role / overlap / reset-boundary tokens，$P_{safe}$ 保护 high-confidence static structure-like tokens，例如高 $C_{anchor}$ 且低 $C_{dyn}$ 的 patch。

Phase C v2 最看好的不是 key-only，而是 protected pairwise suppression。

### 3.3 需要比较的 dynamic cue source

Phase C v2 不应该继续只用一个 calibrated-soft-or $C_{dyn}$。要至少比较四类 dynamic map。

#### Source 1: current calibrated $C_{dyn}$

这是 RFR-100 的 reference signal。保留它是为了可比性。

#### Source 2: reliability-filtered $C_{dyn}$

$$
D_i = C_{dyn,i} \cdot (1-C_{unc,i}) \cdot (1-C_{occ,i})
$$

动机：如果 high dynamic 其实来自遮挡或不确定性，直接拿来做 read-path suppression 会误伤。旧实验已经显示 `unc` 和 `occ` 不适合作为直接 token ranking，但可以作为 reliability filter。

#### Source 3: top-k dynamic mass map

每帧只保留 $D$ 最高的一小部分 patch：

$$
D_i^{topK}=D_i\cdot \mathbf{1}[D_i \in TopK_t(D)]
$$

建议：

$$
K \in \{5\%, 10\%, 20\%\}
$$

动机：RFR-125 / 150 / 200 开始伤害，可能是 dynamic map 太 diffuse。VGGT4D 也强调 dynamic mask 不应 full-mask 全层，而是早期、局部、精选地抑制 dynamic image tokens。

#### Source 4: internal attention motion map

从 LoGeR probe trace 中使用：

```text
attn_dynamic_patch
frame_attn_key_cosine_avg
frame_attn_key_cosine_shallow/deep
dyn4d_patch or Gram-lite Q/K statistics
```

如果已经有 `dyn4d_patch` 或 key-cosine maps，则优先用这些 patch-level internal motion cue。VGGT4D 的启发是：motion cue 往往已经隐含在 pretrained geometry transformer 的 attention / QK / Gram statistics 里，直接 pointmap residual 反而容易被视差和纹理误导。

### 3.4 需要比较的 layer schedule

Phase C v1 只用了 `early`。但 early 的范围可能仍然太粗。Phase C v2 要把 early 继续拆开。

建议层级设置：

| Layer schedule | 目的 |
|---|---|
| `early_all` | 复现 RFR-100 设置 |
| `early_quarter` | 只控制最浅 1/4 层，减少 OOD 风险 |
| `early_half_weak` | 早期 1/2 层但 beta 更弱 |
| `single_layer_0` | 看第一层是否已经足够 |
| `single_layer_1` | 看第二个 frame-attn hook 是否更安全 |
| `single_layer_2` | 看 motion cue 是否在稍后层更成熟 |
| `middle_control` | 诊断中层是否伤害 translation |

这里不要一上来 full KITTI 跑所有组合。应采用三级 gate：64-frame smoke、256-frame mid-run、full KITTI 01。

### 3.5 Frame-attention v2 实验矩阵

第一批只围绕 RFR-100 附近做最小矩阵。

#### 3.5.1 64-frame smoke

所有候选先跑 64-frame smoke，目标不是选最优，而是排除明显 hook 失效或轨迹崩溃。

| ID | Cue | Bias | Layer | Beta | 目的 |
|---|---|---|---|---:|---|
| F64-01 | calibrated $C_{dyn}$ | pair | early_all | 1.0 | 复现 RFR-100 smoke |
| F64-02 | calibrated $C_{dyn}$ | key-only | early_all | 0.5 | 测试 key-only 是否过强 |
| F64-03 | calibrated $C_{dyn}$ | protected-pair | early_all | 1.0 | 测试 reference protection |
| F64-04 | reliability-filtered $C_{dyn}$ | protected-pair | early_all | 1.0 | 测试 filter 是否减少误伤 |
| F64-05 | top10 $C_{dyn}$ | protected-pair | early_all | 1.0 | 测试 dynamic mass cap |
| F64-06 | internal motion | protected-pair | early_all | 1.0 | 测试 internal cue |
| F64-07 | top10 internal motion | protected-pair | early_quarter | 1.0 | 测试最保守主候选 |

64-frame 通过标准：

```text
1. 不崩溃。
2. ATE 不比 HMC no-control 64-frame差超过 0.25 m。
3. attention_mass_to_highD 在 enabled layers 有下降，下降幅度在 15%–70% 之间。
4. attention entropy 不出现大面积塌陷。
5. protected tokens 的平均 attention received 不下降超过 5%。
```

#### 3.5.2 256-frame mid-run

从 64-frame 中保留最多 6 个候选进入 256-frame。256-frame 主要看局部轨迹是否稳定，避免 full KITTI 浪费。

| ID | Cue | Bias | Layer | Beta |
|---|---|---|---|---:|
| F256-01 | RFR-100 reference | pair | early_all | 1.0 |
| F256-02 | reliability-filtered $C_{dyn}$ | protected-pair | early_all | 1.0 |
| F256-03 | top10 $C_{dyn}$ | protected-pair | early_all | 1.0 |
| F256-04 | top10 $C_{dyn}$ | protected-pair | early_quarter | 1.0 |
| F256-05 | internal motion | protected-pair | early_all | 1.0 |
| F256-06 | top10 internal motion | protected-pair | early_quarter | 1.0 |

256-frame 通过标准：

```text
1. ATE 相比 HMC no-control 256-frame至少改善 0.15 m，或者不差且 rotation / RPE 有明显改善。
2. 不能出现“rotation 好但 ATE 大幅变坏”的旧失败模式。
3. enabled layer 的 high-D attention mass 下降，但 protected/ref token attention 保持。
4. 如果某候选 256-frame 明显差于 RFR-100 reference，则不进入 full。
```

#### 3.5.3 Full KITTI 01

只允许最多 4 个候选进入 full KITTI 01。

建议 full candidates：

| ID | Candidate | 说明 |
|---|---|---|
| FC2-01 | RFR-100 reference | Phase C v1 best reference |
| FC2-02 | reliability-filtered + protected-pair + early_all | 检验 reliability filter |
| FC2-03 | top10 $C_{dyn}$ + protected-pair + early_all | 检验 dynamic mass cap |
| FC2-04 | top10 internal motion + protected-pair + early_quarter | 最保守 internal cue candidate |

Full 判断标准：

| 结果 | 解释 |
|---|---|
| `ATE > 41.0733 m` | 没超过 RFR-100，不算 Phase C v2 进展 |
| `41.0 m > ATE ≥ 40.8 m` | minimal pass，说明细化有效，但仍偏弱 |
| `40.8 m > ATE ≥ 40.5 m` | Phase C v2 pass，可考虑小规模 Phase D |
| `ATE < 40.5 m` | strong read-path signal，进入 Phase D 组合 |
| `ATE < 40.0 m` | read-path 本身已成主候选分支 |

如果所有 FC2 候选都没有超过 RFR-100，则说明 frame-attention 这一路的当前信号已经接近上限，需要转入更强 internal motion cue 或 external motion cue。

---

## 4. Phase C v2-B：SWA read gate 重新验证

Phase C v1 没有把 SWA-read 和 chunk-attention 当成有效结果行，因为 dense read-importance 和 chunk dynamic-mass control maps 还不够可靠。Phase C v2 需要重新给 SWA 设计一个更严格的 gate。

### 4.1 核心假设

LoGeR 的 SWA 是相邻 chunk 之间的 lossless local context highway。它帮助当前 chunk 使用上一 chunk 的高保真局部特征做相邻对齐。如果上一 chunk 的 moving car、遮挡边界或错误低置信 token 被 SWA 当作 local anchor，可能会污染当前 chunk 的局部对齐。这个污染不是 TTT write 能解决的。

Phase C v2 的 SWA 假设是：

$$
\text{SWA control 只有在“只控制 previous-chunk dynamic keys”且保护 overlap/reference tokens 时才可能安全。}
$$

因此 SWA gate 不应控制 current chunk tokens，也不应对 previous chunk 所有 token 做 dense suppression。

### 4.2 SWA control map

对 previous chunk token $j$ 构造：

$$
D_{prev,j}^{swa} = D_{prev,j}\cdot(1-P_{ref,j})\cdot(1-P_{safe,j})
$$

其中：

```text
D_prev: previous chunk 的 dynamic / instability score
P_ref: overlap frames, special tokens, reset-boundary tokens, high cumulative attention tokens
P_safe: high-anchor / low-dynamic / high-confidence structure-like tokens
```

SWA attention bias 只加在 previous keys 上：

$$
B_k^{swa}=\beta_{swa}\log(1-D_{prev,k}^{swa}+\epsilon)
$$

### 4.3 SWA gate 实验

先跑 64-frame / 256-frame，不直接 full。

| ID | Cue | Target | Beta | Protection | 目的 |
|---|---|---|---:|---|---|
| S64-01 | $C_{dyn}$ | prev-key only | 0.25 | on | smoke |
| S64-02 | $C_{dyn}$ | prev-key only | 0.50 | on | smoke |
| S64-03 | top10 $C_{dyn}$ | prev-key only | 0.50 | on | mass cap |
| S64-04 | internal motion | prev-key only | 0.50 | on | internal cue |
| S64-05 | top10 internal motion | prev-key only | 0.50 | on | conservative internal |

SWA 通过 64-frame 的条件：

```text
1. previous high-D token 的 attention received mass 下降 15%–60%。
2. overlap/reference token 的 attention received mass 不下降超过 5%。
3. current chunk self/local attention 不被改变。
4. trajectory 不崩。
```

256-frame 候选最多 3 个。Full KITTI 01 只跑满足下面条件的候选：

```text
1. 256-frame ATE 不差于 no-control。
2. SWA attention dashboard 显示 high-D previous keys 被降权。
3. Reference tokens 被保护。
```

Full 判断标准：

| 结果 | 解释 |
|---|---|
| `ATE < 41.0733 m` | SWA 有独立价值，超过 RFR-100 |
| `ATE < 41.0 m` | SWA read gate 进入 Phase C v2 minimal pass |
| `ATE < 40.8 m` | SWA 可进入 Phase D 与 frame/TTT 组合 |
| `ATE worse but local segments improve` | SWA 可能是 local alignment cue，但 global continuity 不稳，暂不组合 |

如果 SWA full 失败，但 dashboard 清楚显示它修复了某些局部片段，可以把它保留为 reliability-gated local schedule 的候选；否则不再继续扫 SWA beta。

---

## 5. Phase C v2-C：Chunk-wise bidirectional attention gate

Chunk-wise bidirectional attention 是当前 chunk 内最强的几何推理路径。它比 frame attention 更危险，因为它直接混合跨帧 token，如果 dynamic suppression 错了，很可能破坏当前 chunk 内的 pose 和 pointmap。

### 5.1 核心假设

Phase C v2 不应把 chunk attention 当成第一个优化对象。它只在 frame-attention v2 至少有一个候选过 minimal pass 后再试。因为如果 dynamic map 连 frame attention 都不能安全使用，放到 chunk attention 中风险更高。

chunk attention gate 的假设是：

$$
\text{dynamic tokens 的跨帧 key 可能污染 static tokens 的 chunk-wise reasoning，但控制必须非常浅层、轻量、protected。}
$$

### 5.2 Chunk attention bias

优先只测 protected-pair：

$$
B_{qk}^{chunk}=\beta_c\log(1-(1-D_q^{prot})D_k^{prot}+\epsilon)
$$

不测 query-only，不测 all-layer hard suppression。

### 5.3 Chunk attention 实验

只做最小矩阵：

| ID | Cue | Bias | Layer | Beta | 进入条件 |
|---|---|---|---|---:|---|
| C64-01 | best frame cue | protected-pair | first chunk-attn layer only | 0.25 | frame v2 minimal pass |
| C64-02 | best frame cue | protected-pair | early_quarter | 0.25 | frame v2 minimal pass |
| C64-03 | top10 best frame cue | protected-pair | early_quarter | 0.50 | frame v2 minimal pass |

64-frame 通过后，最多 2 个进入 256-frame。Full KITTI 01 只有在 256-frame 不差于 no-control 且 attention dashboard 合理时才跑。

Chunk attention full 判断标准：

```text
1. 必须超过 RFR-100：ATE < 41.0733 m。
2. 如果只是 rotation 变好但 ATE 变差，立刻停止 chunk attention route。
3. 如果达到 ATE < 40.8 m，进入 Phase D 组合候选。
```

---

## 6. Phase C v2-D：TTT apply gate 的保守重测

Phase C v1 的 TTT apply gate 明确不安全，rho=0.1 就已经把 ATE 拉到 42.6125 m，rho=0.3 更是 44.8062 m。Phase C v2 不应继续把 TTT apply 当成主路线。

但是为了确认失败不是因为“all layers + aggressive residual gate”，可以做一个非常保守的 rescue gate。

### 6.1 核心假设

TTT apply 是 long-range global memory 注入当前 chunk 的路径。LoGeR 依赖 TTT apply 稳住全局坐标和尺度。直接用 dynamic / residual gate 去抑制 TTT apply，很可能会破坏 translation continuity。

因此 Phase C v2 对 TTT apply 的假设是：

$$
\text{TTT apply 不能做强 suppression，只能做 reference-protected、high-D token 的极轻 read gate。}
$$

### 6.2 保守 gate 公式

对每个 token 的 TTT apply output：

$$
\tilde H_i = H_i + g_i^{read} f_{W_m}(LN(H_i))
$$

其中：

$$
g_i^{read} = 1 - \rho \cdot D_i^{prot}
$$

并强制：

$$
g_i^{read} \ge 0.9
$$

这和 v1 的 rho=0.1/0.2/0.3 不是一个量级。v2 的 TTT apply 只允许极轻改变。

### 6.3 实验矩阵

| ID | Cue | Layer | rho | min gate | 目的 |
|---|---|---|---:|---:|---|
| TA64-01 | top10 $C_{dyn}$ | late only | 0.05 | 0.95 | smoke |
| TA64-02 | top10 internal | late only | 0.05 | 0.95 | smoke |
| TA64-03 | best frame cue | single safest layer | 0.05 | 0.95 | smoke |

只有 64-frame 和 256-frame。如果任一候选在 256-frame 仍出现 ATE 明显变差，停止 TTT apply route。Full KITTI 01 只有在 256-frame 明确改善时才允许跑。

判断标准：

```text
1. TTT apply gate 不允许再出现 rotation 变好但 ATE 大幅变差。
2. 如果 full ATE 不能超过 RFR-100，不进入 Phase D。
3. 如果 TTT apply 仍然不安全，Phase D 只组合 frame/SWA/chunk + TTT branch0 write，不再包含 TTT apply read gate。
```

---

## 7. Phase C v2-E：信号质量 dashboard 和 failure diagnostics

Phase C v2 每个候选都必须输出 dashboard。否则即使 ATE 稍微变好，也不知道为什么变好。

### 7.1 每个 chunk 必须记录的统计

```text
mean(D_tok)
q90(D_tok) - q10(D_tok)
dynamic_mass@top10
mean(C_unc), mean(C_occ), mean(C_anchor)
protected_token_ratio
attention_mass_to_highD_before / after
attention_mass_to_ref_before / after
attention_entropy_before / after
enabled_layer_count
bias_norm_mean / max
feature_delta_norm at hook output
```

### 7.2 attention mass 判断

对高动态 key：

$$
M_D=\sum_q\sum_{k:D_k>\tau_D} A_{qk}
$$

要求：

$$
0.15 \le \frac{M_D^{before}-M_D^{after}}{M_D^{before}+\epsilon} \le 0.70
$$

如果下降小于 15%，说明 hook 几乎没有有效作用；如果下降大于 70%，说明 suppression 太强，可能把模型推到 OOD。

对 reference token：

$$
M_{ref}=\sum_q\sum_{k:P_{ref,k}=1} A_{qk}
$$

要求：

$$
\frac{M_{ref}^{after}}{M_{ref}^{before}+\epsilon} > 0.95
$$

如果 reference token attention 下降太多，可能会伤害全局连续性。

### 7.3 必须做可视化的片段

至少选择四段：

```text
[0, 200): early trajectory establishment
[300, 500): previous experiments had local wins
[400, 600): previous experiments often had failures
[800, 1000): long-horizon accumulated drift region
```

每段抽取 5–10 帧，保存：

```text
RGB
D_tok map
C_dyn / C_unc / C_occ / C_anchor
protected mask
attention mass to high-D before/after
frame-attn output delta norm
SWA prev-key importance if enabled
chunk-attn interaction if enabled
```

人工检查重点：

1. high-D 是否主要落在 moving cars / dynamic objects，而不是 road/building/guardrail。
2. road、building、guardrail、overlap reference 是否被保护。
3. sky / vegetation 是否占据大量 high-D 导致 attention 被错误重排。
4. RFR-100 失败片段里，D map 是否太 diffuse。
5. top-k / protection 是否确实减少误伤。

---

## 8. Phase C v2 的完整执行顺序

Phase C v2 不应该一次性并行跑所有 full KITTI。推荐按 gate 顺序推进。

### Step 1：新 pipeline correctness

先完成：

```text
G0: HMC two-pass no-control LoGeR full KITTI 01
G1: HMC two-pass no-control LoGeR* full KITTI 01
G2: HMC identity hooks LoGeR / LoGeR* full KITTI 01
G3: synthetic control-map hook sanity, 64 frames
```

只有全部通过，才进入模型实验。

### Step 2：frame attention v2 smoke

跑 F64-01 到 F64-07。保留最多 6 个进入 256-frame。

### Step 3：frame attention v2 mid-run

跑 F256-01 到 F256-06。保留最多 4 个进入 full KITTI 01。

### Step 4：frame attention v2 full

跑 FC2-01 到 FC2-04。若没有任何一个超过 RFR-100，停止 Phase C v2-A，转向 internal motion cue 或 semantic/flow signal。

### Step 5：SWA read gate

只有 frame attention v2 至少有 minimal pass，或者 SWA dashboard 在 smoke 阶段非常清晰，才进入 SWA 256/full。

### Step 6：chunk attention gate

只有 frame attention v2 过 minimal pass 后再试。chunk attention 不作为第一优先级。

### Step 7：TTT apply conservative rescue

只做 64/256。除非明显改善，否则不 full。

---

## 9. Phase C v2 的决策树

### 情况 A：frame attention v2 达到 `<40.5 m`

这说明 read-path signal 已经足够强，进入 Phase D：

```text
best frame-attn read control
+ TTT branch0 dyn/TTT-residual write control
+ optional sparse exact preserve
```

但 Phase D 必须保留 ablation：frame-only、TTT-only、combined。

### 情况 B：frame attention v2 只达到 40.8–41.0 m

这说明 read-path signal 有效但弱。可以进入小规模 Phase D，但不能大范围组合。优先组合当前已知最安全的 TTT branch0 BL01，而不是加入 SWA/chunk/TTT apply。

### 情况 C：frame attention v2 没超过 RFR-100

说明当前 frame-attn control 已接近上限。下一步不应继续扫 beta/layer，而应升级 motion cue：

```text
VGGT4D-style Gram-lite Q/K statistics
MUT3R-style multi-layer self-attention dispersion
semantic-protected dynamic maps
optical flow / epipolar consistency
```

### 情况 D：SWA read gate 独立超过 RFR-100

说明 local memory highway 是新的有效控制点。进入 Phase D 时优先组合：

```text
SWA read gate + TTT branch0 write
```

而不是先组合 chunk attention。

### 情况 E：TTT apply 仍然失败

TTT apply 从 Phase C/Phase D 中移除。它只保留 dashboard，不参与主模型。

---

## 10. Phase C v2 后的进入 Phase D 条件

Phase D 是组合实验，成本高，且容易归因混乱。因此必须满足下面条件之一：

$$
ATE_{best\ read-only} \le 40.5\text{ m}
$$

或者：

$$
ATE_{best\ read-only} < 41.0\text{ m}
\quad\text{and}\quad
\text{non-overlap segment diagnostics shows broad improvement}
$$

其中 broad improvement 定义为：

```text
100-frame non-overlap improved segments >= 7 / 11
50-frame non-overlap improved segments >= 14 / 22
full ATE improves over RFR-100
no large segment failure > +0.20 m vs no-control
```

如果只在少数 segment 有明显 local win，但 full ATE 没有过 gate，不进入 Phase D。那种情况需要先做 reliability-gated schedule，而不是直接组合控制器。

---

## 11. 预期结果和风险

Phase C v2 最可能成功的方向是：

$$
\text{top-k / reliability-filtered dynamic map}
+
\text{protected pairwise frame-attention bias}
+
\text{narrow early layer schedule}
$$

因为 Phase C v1 已经说明 frame attention 有信号，但 dense map + early_all + beta=1.0 仍然略弱。增加 protection 和 dynamic mass cap 有可能减少误伤，从而把 41.0733 m 推到 40.8 m 以下。

最可能失败的方向是 TTT apply。它已经在 rho=0.1 时明显伤害 ATE，所以除非非常保守，否则不值得继续投入。

SWA read gate 是不确定但值得验证的方向。LoGeR 原论文中 SWA 对 adjacent chunk local alignment 很关键，因此如果 previous dynamic keys 被当前 chunk 强读取，SWA gating 可能有意义。但它必须只作用在 previous keys，并保护 overlap/reference tokens。

chunk-wise bidirectional attention 是高风险方向。它直接影响当前 chunk 内强几何推理，一旦 cue 错，会比 frame attention 更伤。因此必须放在 frame-attn v2 之后，而不是优先 full sweep。

---

## 12. 最终总结

Phase C v1 并不是完全没有信号。它证明 frame attention early control 能把 full KITTI 01 从 A3 native 的 41.7502 m 和 BL01 的 41.3665 m 推到 41.0733 m。但它没有过 `<41.0 m`，更强 beta 反而退化，TTT apply read gate 明显失败。因此 Phase C v2 的重点不是继续加大强度，也不是马上进入 Phase D，而是把 read-path gate 拆得更细。

Phase C v2 的核心路线是：

$$
\boxed{
\text{HMC correctness}
\rightarrow
\text{frame-attn cue/source/bias/layer/protection fine gate}
\rightarrow
\text{SWA prev-key read gate}
\rightarrow
\text{chunk-attn cautious gate}
\rightarrow
\text{TTT-apply conservative rescue only if justified}
}
$$

正式进入 Phase D 的条件也要提高：单一路径至少要接近或低于 40.5 m，或者虽然不到 40.5 m，但 full ATE 和 segment diagnostics 都显示稳定、广泛的改善。否则继续组合只会把当前弱信号和旧 TTT branch0 信号混在一起，难以解释，也难以真正突破 LoGeR。


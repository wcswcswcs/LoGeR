# KITTI 01 主模型实验方案：Geometry-First Semantic-Light TTT Write Control

> 目标：在 KITTI 01 上建立一个主模型，显著超过 LoGeR 与 LoGeR\* 的 ATE 表现。截图中 01 序列长度为 1101 帧、约 2.5 km，LoGeR 为 41.64 m，LoGeR\* 为 47.91 m。因此本轮实验的硬目标设为：**ATE < 33 m**；理想目标设为：**ATE < 30 m**。如果某个配置低于 30 m，并且分段漂移没有明显局部崩坏，则作为第一版主模型候选。

本文设定的主模型暂命名为 **GSL-WC**：**Geometry-first, Semantic-light Write Controller**。它不是让语义前端主导写入，而是让 Dynamic Cue Extractor 给出主要的 geometry eligibility；Video Masklet Front-end 只作为参考，用来轻度压制明显可移动物体和低价值大区域。这个选择直接来自当前 KITTI 场景和现有语义前端状态：thing 会漏检/跟踪失败，stuff 边界粗糙，indoor prompt 不适合 KITTI。因此，主模型的风险控制要以几何为主，语义只能做低强度、负向、可回退的参考。

---

## 1. 先明确我们真正要验证的假设

KITTI 01 是高速、长距离、无闭环的驾驶序列。LoGeR 已经明显优于 Pi3-Chunk，说明 TTT memory 对长程几何有帮助；但 LoGeR 在 01 上仍有 41.64 m ATE，说明当前 memory 写入里仍然混进了会伤害后续位姿/几何的 token。我的核心假设不是“语义检测不够好所以要修语义”，而是：

**假设 H1：01 序列的主要改进空间来自 TTT 写入污染控制，而不是当前 chunk 的 feedforward 几何预测。**

也就是说，我们要让当前 chunk 写入 fast weights 的内容更干净。动态车辆、遮挡边界、远处天空/树木、低纹理或几何不确定区域，不应该和道路、建筑、护栏、稳定结构一样强地写入。Stage E 已经是 delayed write-back，因此我们有机会在 LoGeR forward 后用 Stage B/D 的 prior 重放 TTT update。

**假设 H2：对 KITTI 01，显式几何动态线索比隐式 attention 动态线索更可靠，但显式线索会偏边缘化，隐式线索能补充 object-level 或 attention-level 的风险区域。**

当前 Dynamic Cue Extractor 已经输出 `C_dyn_explicit` 和 `C_dyn_implicit`，并保存了 `max / soft_or / avg / addclip` 四种候选融合。当前代码默认使用 raw max：

$$
C_{dyn}=\max(D_{exp}, D_{imp})
$$

这个默认实现的好处是召回高，坏处是 implicit attention 如果在道路、天空、远处纹理上出现低频假响应，会直接压掉很多本来该写的静态结构。KITTI 01 的高速前向运动会使这种风险更大。因此本方案的主假设是：**raw max 可能不是最优，最佳融合应当是校准后的 weighted soft-or，而不是简单 max。**

**假设 H3：Semantic Prior Generator 在 KITTI 01 上应该保持 geometry eligibility 主导。**

当前 v2 设计已经把 geometry、semantic、mask trust 解耦：geometry 决定 eligibility，semantic 只决定 value，mask quality 决定 routing。我们要进一步限制语义对 KITTI 的影响：语义只做“可信时轻度压低”，不做“把某个区域强行升高”。具体说，structure 语义不会把低几何 eligibility 区域救回来；movable / low-value 语义只在 mask 有一定质量时乘上一个弱抑制。

**假设 H4：如果 replay 本身和 native LoGeR 不等价，任何 prior 调参都没有意义。**

因此第一个实验不是调 prior，而是验证 `unity_replay` 是否接近 native LoGeR。当前 Stage E 用 replay 写回 fast weights，且 `_replay_layer()` 里暂时用 `A_tok[:l]` 对齐 cached token。如果 cached token 顺序和 `A_tok` 的前缀不一致，token prior 会错位；这会让所有实验结论失效。所以必须先通过 parity gate。

---

## 2. 主模型的候选公式

### 2.1 显式动态线索

当前显式动态线索来自几何一致性和遮挡项：

$$
D_{exp}(t,u)=
\operatorname{clip}
\left(
\alpha_1\left(1-C_{stat}^{geom}(t,u)\right)-\alpha_3 C_{occ}(t,u),
0,1
\right)
$$

其中 $u=(h,w)$。这里保留 $-\alpha_3 C_{occ}$ 是必要的，因为遮挡边界不应该被完全等价地当成动态物体。对 KITTI 01，初始值使用：

$$
\alpha_1=0.8,\qquad \alpha_3=0.5
$$

`C_occ` 的作用不是让遮挡区域可写，而是避免它把 `C_dyn` 误推得过高；最终写入图里 `C_occ` 仍然会作为风险项被扣掉。

### 2.2 隐式动态线索校准

隐式动态来自 Stage A 的 attention / key-cosine dynamic patch map，记为 $D_{imp}^{raw}$。不建议直接 raw max。先做 chunk 内分位数校准：

$$
\tilde D_{imp}(t,u)=
\operatorname{clip}
\left(
\frac{D_{imp}^{raw}(t,u)-Q_{50}(D_{imp}^{raw})}
{Q_{95}(D_{imp}^{raw})-Q_{50}(D_{imp}^{raw})+\epsilon},
0,1
\right)
$$

这样做的目的不是改变 attention map 的相对结构，而是去掉 KITTI 场景中可能出现的全局偏置。若 $Q_{95}-Q_{50}$ 太小，说明 implicit map 没有有效动态对比度，本 chunk 应把 $\omega_{imp}$ 自动降到 0。

### 2.3 推荐的主融合：calibrated weighted soft-or

定义一个 explicit-only anchor：

$$
C_{anchor}^{exp}=C_{stat}\cdot (1-D_{exp})\cdot (1-C_{unc})
$$

隐式线索只在 explicit anchor 不强的位置充分发挥作用：

$$
g_{imp}(t,u)=
\operatorname{clip}
\left(
 g_0+(1-g_0)\left(1-C_{anchor}^{exp}(t,u)\right),
 g_0,1
\right)
$$

其中 $g_0=0.25$。最终动态融合为：

$$
C_{dyn}(t,u)=
1-
\left(1-D_{exp}(t,u)\right)
\left(1-\omega_{imp}\,g_{imp}(t,u)\,\tilde D_{imp}(t,u)\right)
$$

第一轮搜索：

$$
\omega_{imp}\in\{0.25,0.50,0.75\}
$$

我预期 $\omega_{imp}=0.50$ 最稳。如果 $\omega_{imp}=0.75$ 明显降低 mean write prior 或让道路区域变暗，说明 implicit 过强；如果 $\omega_{imp}=0.25$ 和 explicit-only 几乎没有区别，说明隐式线索确实有效但权重不足。

### 2.4 Geometry eligibility 仍然是主写入资格

Stage D 第一轮保持 `use_g_write_geo=True`，也就是直接使用 Stage B 的几何写入图作为 eligibility：

$$
Elig_{pix}(t,u)=G_{write\_geo}(t,u)
$$

其中：

$$
z_{geo}(t,u)=
\lambda_s C_{stat}(t,u)
+\lambda_a C_{anchor}(t,u)
-\lambda_d C_{dyn}(t,u)
-\lambda_o C_{occ}(t,u)
-\lambda_u C_{unc}(t,u)
$$

$$
G_{write\_geo}(t,u)=\sigma(z_{geo}(t,u))
$$

注意，由于代码当前没有 $b_{geo}$ 和 temperature，$G_{write\_geo}$ 很容易集中在 0.4 到 0.8 之间。第一阶段先不加新自由度；如果所有候选的 $A_{tok}$ 均值过高，动态区域仍写太多，再增加一个 post-calibration：

$$
Elig'_{pix}=\operatorname{clip}
\left(
0.5+s_{geo}\left(Elig_{pix}-0.5\right),
0,1
\right)
$$

其中 $s_{geo}\in\{1.0,1.5,2.0\}$。只有当动态风险分布被 sigmoid 压扁时才引入这个旋钮。

### 2.5 Semantic-light prior

语义 masklet 只做参考。对 masklet $j$，仍先计算 mask 内几何均值：

$$
\bar E(j,t)=
\frac{\sum_u M_j(t,u)Elig_{pix}(t,u)}
{\sum_u M_j(t,u)+\epsilon}
$$

但语义调制采用 group-specific strength，而不是单一 $\rho_{sem}$：

$$
A_{mask}(j,t)=
\bar E(j,t)\cdot
\left(1-\rho_{g_j}\left(1-v_{g_j}\right)\right)
$$

推荐第一版：

| 语义组 | $v_g$ | $\rho_g$ | 解释 |
|---|---:|---:|---|
| `STRUCTURE_ANCHOR` | 1.00 | 0.10 | 不提高，只基本保持 geometry |
| `STATIC_THING` | 0.70 | 0.15 | 轻微降权 |
| `LOW_VALUE_STUFF` | 0.35 | 0.25 | 天空/植被等轻度压低 |
| `MOVABLE_THING` | 0.10 | 0.45 | 车/人/骑行者较明显压低 |
| `UNCERTAIN_REGION` | 0.45 | 0.20 | 不确定区域轻度压低 |

mask trust 也做 group cap，避免粗糙 stuff 边界覆盖 geometry：

$$
R_{mask}(j,t)=\min(V_j(t)Q_j(t), r_{max,g_j})
$$

推荐 cap：

| 语义组 | $r_{max,g}$ |
|---|---:|
| `STRUCTURE_ANCHOR` | 0.25 |
| `STATIC_THING` | 0.35 |
| `LOW_VALUE_STUFF` | 0.30 |
| `MOVABLE_THING` | 0.60 |
| `UNCERTAIN_REGION` | 0.20 |

最终像素融合仍保持 soft routing：

$$
A_{pix}=R_{mask}A^{sem}_{pix}+(1-R_{mask})Elig_{pix}
$$

这保证语义不会直接推翻几何。特别是粗糙 road / building / sky stuff 边界，不会把大面积区域强行改写。

### 2.6 Chunk-level budget

chunk 预算只来自 geometry，不来自 semantic：

$$
B_{chunk\_geo}=\frac{1}{L_{patch}}\sum_i E_{patch,i}
$$

$$
\lambda_{write}=\lambda_{min}+(\lambda_{max}-\lambda_{min})B_{chunk\_geo}
$$

KITTI 01 是长高速序列，完全把 uncertain chunk 写入降到接近 0 可能会造成欠适配。因此第一轮主模型不建议 $\lambda_{min}=0$，推荐从：

$$
\lambda_{min}=0.15,\qquad \lambda_{max}=1.0
$$

开始。

---

## 3. 实验必须先过的 Implementation Gate

### Gate 0：native LoGeR 复现

先用同一 checkpoint、同一 config、同一 resolution、同一 window / overlap 跑 KITTI 01 native。目标不是创新，而是确认我们能复现截图量级。

判定标准：

$$
|ATE_{native}-41.64| < 2.0\text{ m}
$$

如果差距超过 2 m，先不要调 prior。优先检查：frame list 是否一致、是否用了同样的 resolution、是否启用了相同的 `se3/sim3`、`window_size`、`overlap_size`、`reset_every` 和评估对齐方式。

### Gate 1：unity replay parity

然后跑 `unity_replay`，即 replay TTT update 但 token prior 全为 1。这个结果应该接近 native，因为它不应该引入语义或几何抑制。

判定标准：

$$
|ATE_{unity}-ATE_{native}| < 3.0\text{ m}
$$

更严格地，也要检查每层 debug：

$$
\bar A_{tok}=1,
\qquad
\lambda_{write}=1
$$

如果 unity replay 与 native 明显不一致，先修 Stage E replay 或 token alignment。尤其要检查 cached token length $l$ 是否对应 `A_tok[:l]`。建议增加一个 debug：对每个 TTT layer 打印：

```text
layer_id, cached_l, L_tok, L_patch, num_special, first_20_token_type, assumed_alignment
```

如果 cached $l$ 只包含 patch tokens，就应使用 `A_patch_flat` 对齐，而不是 `A_tok` 前缀；如果 cached $l$ 是 per-frame `[special + patch]`，则必须按 `geo.token_type` 的真实顺序索引。这个 gate 是所有后续实验的前提。

---

## 4. 第一阶段：不跑语义，先找到 geometry-only 主方向

这一阶段建议新增一个轻量开关：`--stage_c_mode none` 或 `--semantic_reference_mode none`。当该开关启用时，不加载 SAM/YOLOE，直接构造空 `MaskletOutput`，让 Stage D 退化为：

$$
A_{pix}=Elig_{pix}=G_{write\_geo}
$$

这样可以快速搜索 Dynamic Cue 和 SPG/Stage E 参数，同时避免语义前端的不稳定性污染结论。

### 实验 1：Dynamic cue fusion 选择

这里的控制变量是 fusion 方式，其他参数固定为保守值：

$$
\lambda_s=1.0,\quad
\lambda_a=0.5,\quad
\lambda_d=0.8,\quad
\lambda_o=0.3,\quad
\lambda_u=0.5
$$

$$
\lambda_{min}=0.15,
\quad
\lambda_{max}=1.0,
\quad
A_{special}^{min}=0.5
$$

候选如下：

| Run | $C_{dyn}$ 融合 | 目的 | 预期 |
|---|---|---|---|
| F0 | native | 复现基线 | ATE 约 41.6 |
| F1 | unity replay | replay parity | 接近 F0 |
| F2 | $D_{exp}$ only | 只用几何显式风险 | ATE 35–40，稳定但改进有限 |
| F3 | $D_{imp}$ only | 验证 implicit 是否独立可靠 | 大概率不稳定，只用于诊断 |
| F4 | raw max | 当前代码默认 | 如果 attention 噪声小，可能 32–36；若过抑制，可能退化 |
| F5 | raw soft-or | 比 max 更 aggressive | 高风险，高概率过抑制 |
| F6 | raw avg | 保守融合 | 可能与 F2 接近，动态召回不足 |
| F7 | calibrated weighted soft-or, $\omega_{imp}=0.25$ | 推荐模型弱 implicit | 预计稳，但可能不够强 |
| F8 | calibrated weighted soft-or, $\omega_{imp}=0.50$ | 推荐主候选 | 预计最好，目标 30–34 |
| F9 | calibrated weighted soft-or, $\omega_{imp}=0.75$ | 测试 implicit 上限 | 若过抑制，分段 drift 会变差 |

每个 run 都要保存这些统计，而不只看 ATE：

$$
\operatorname{mean}(C_{dyn}),
\quad
\operatorname{mean}(C_{anchor}),
\quad
\operatorname{mean}(G_{write\_geo}),
\quad
\operatorname{corr}(C_{dyn}, A_{patch}),
\quad
\operatorname{corr}(C_{anchor}, A_{patch})
$$

以及分段 ATE / RPE：按 100 帧或 200 帧切片，观察是否某个段突然崩掉。

**判定逻辑：**

如果 F4 raw max 比 F2 明显更好，并且 $\operatorname{mean}(G_{write\_geo})$ 没有低于 0.35，说明 implicit cue 可直接增强风险召回；下一阶段用 F4/F8 同时进参数调优。  
如果 F4/F5 比 F2 差，而 F8 好，说明 implicit 有用但必须校准；主模型采用 calibrated weighted soft-or。  
如果 F7/F8/F9 都不如 F2，说明 KITTI 01 的 attention dynamic 对写入控制不可靠；主模型改成 explicit-only，并把精力放到 $\lambda_d/\lambda_u$ 和 chunk budget。  
如果所有 geometry prior 都不如 native，先不要调语义，检查 Stage E prior 注入是否导致整体欠写或 token 对齐是否错。

### 实验 2：support horizon 与 residual scale

KITTI 01 高速前向运动会放大 same-pixel world-space residual。`k_intra` 太大时，静态道路/护栏也可能被误判为不一致；`sigma_pt` 太小时，静态区域的 $C_{stat}$ 会整体偏低。

在上一实验最佳 fusion 上，搜索：

| Run | `k_intra` | `sigma_pt` | 预期 |
|---|---:|---:|---|
| S1 | 3 | 0.20 | 动态敏感，但静态误伤较多 |
| S2 | 3 | 0.25 | 当前较稳默认 |
| S3 | 5 | 0.25 | 可能更稳，推荐候选 |
| S4 | 5 | 0.35 | 更宽容静态，动态压制变弱 |
| S5 | 7 | 0.25 | 测试更长支持，可能误伤高速静态结构 |
| S6 | 7 | 0.35 | 宽容版长支持 |

需要额外记录静态候选区域的 $C_{stat}$ 分布。没有人工标注也可以用高置信、高 anchor、低 implicit 的区域作为 proxy。理想情况：

$$
Q_{50}(C_{stat}^{static\ proxy})>0.65,
\qquad
Q_{90}(C_{dyn}^{static\ proxy})<0.35
$$

如果 `k_intra=7` 的 ATE 好但 $C_{dyn}$ 大面积变高，说明它可能靠更少写入“侥幸”变好，后续语义和 budget 稍变就会崩。主模型优先选择 `k_intra=3` 或 `5` 的稳健解。

---

## 5. 第二阶段：调 Geometry Eligibility 与 Stage E 写入强度

第二阶段仍然不跑语义，目的是让 geometry-only 的主模型先超过 LoGeR。因为如果 geometry-only 都没有明显改善，语义前端在当前质量下很难可靠补救。

### 实验 3：$G_{write\_geo}$ 参数网格

以第一阶段最佳 fusion 和最佳 `k_intra/sigma_pt` 为固定条件，搜索以下 6 组：

| Run | $\lambda_s$ | $\lambda_a$ | $\lambda_d$ | $\lambda_o$ | $\lambda_u$ | 设计意图 |
|---|---:|---:|---:|---:|---:|---|
| G0 | 1.0 | 0.5 | 0.8 | 0.3 | 0.5 | 当前 baseline |
| G1 | 1.2 | 0.7 | 1.0 | 0.4 | 0.5 | balanced road-safe |
| G2 | 1.0 | 0.8 | 1.4 | 0.5 | 0.6 | 更强动态抑制 |
| G3 | 1.2 | 0.8 | 1.2 | 0.3 | 0.3 | 容忍不确定，避免欠写 |
| G4 | 1.0 | 0.8 | 1.8 | 0.6 | 0.8 | aggressive suppression |
| G5 | 0.8 | 1.2 | 1.2 | 0.4 | 0.5 | anchor-priority |

预期：G2 或 G3 更可能成为 KITTI 01 主候选。G4 可能在动态车辆多的片段表现好，但全局可能欠写，造成后半段 drift 增大。G5 如果有效，说明 $C_{anchor}$ 比 $C_{stat}$ 更适合控制 TTT memory。

**判定逻辑：**

如果 G2 比 G0 好 5 m 以上，并且后半段没有变差，说明动态污染确实是主要问题；继续增强 movable/implicit 抑制。  
如果 G3 最好，说明 KITTI 01 的关键问题不是动态车辆，而是不要把低纹理/远距区域过度归入不可写；后续降低 $\lambda_u$，保留道路/护栏/建筑写入。  
如果 G5 最好，说明 anchor 构造可靠；后续可以考虑 patch pooling 改成 anchor-priority 或 risk-p90 pooling。  
如果 G4 最好但 $\bar A_{tok}$ 很低，先不要直接作为主模型，要检查它是否只是通过“少写”降低短期误差。少写模型在其他序列上可能崩。

### 实验 4：Stage E chunk budget 与 special token

固定最佳 G 组，调 Stage E：

| Run | $\lambda_{min}$ | $\lambda_{max}$ | $A_{special}^{min}$ | 设计意图 |
|---|---:|---:|---:|---|
| E0 | 0.00 | 1.00 | 0.30 | 当前默认，可能欠写 special |
| E1 | 0.10 | 1.00 | 0.50 | 保守主候选 |
| E2 | 0.15 | 1.00 | 0.50 | 推荐初始主模型 |
| E3 | 0.20 | 1.00 | 0.60 | 防止高速长程欠适配 |
| E4 | 0.15 | 0.85 | 0.50 | 降低整体写入幅度 |
| E5 | 0.20 | 1.15 | 0.60 | 如果 prior 太保守，用上限补回 |

对 KITTI 01，我预期 $\lambda_{min}=0.15$ 或 $0.20$ 会优于 0。原因是高速驾驶中大多数 chunk 仍需要持续适配；如果遇到低纹理道路或天空占比大就把写入降得太低，memory 可能跟不上长距离轨迹变化。

需要记录：

$$
\bar A_{tok},\quad
B_{chunk\_geo},\quad
\lambda_{write},\quad
\left\|\Delta W_l^{(r)}\right\|_F
$$

主候选的健康区间建议为：

$$
0.35 < \bar A_{tok} < 0.75
$$

$$
0.25 < \lambda_{write} < 1.05
$$

如果最佳 ATE 出现在 $\bar A_{tok}<0.25$ 或长时间 $\lambda_{write}<0.2$，需要非常谨慎；这通常表示模型靠“少写”赢当前序列，泛化风险大。

---

## 6. 第三阶段：只在最佳 geometry 配置上加入 KITTI semantic reference

这阶段才跑 Video Masklet Front-end。我们不修 thing 漏检和 stuff 边界，只调整 prompt / taxonomy，并把语义影响控制在 reference 级别。

### 6.1 KITTI prompt bank

建议把当前 indoor prompt 改为 KITTI prompt。命令行可直接传：

```bash
--thing_prompts "car,van,truck,bus,tram,train,motorcycle,bicycle,cyclist,rider,pedestrian,person,traffic sign,traffic light,road sign,pole,guardrail,barrier"
--stuff_prompts "road,lane marking,sidewalk,curb,building,bridge,tunnel,wall,fence,sky,tree,vegetation,grass,bush,water,glass,reflection,shadow"
```

同时建议改 taxonomy：

| Label | Group | 备注 |
|---|---|---|
| car / van / truck / bus / tram / train | `MOVABLE_THING` | 即使短时静止，也不作为长期几何锚点 |
| cyclist / rider / pedestrian / person / motorcycle / bicycle | `MOVABLE_THING` | 强低价值写入 |
| road / lane marking / sidewalk / curb / building / bridge / tunnel / wall | `STRUCTURE_ANCHOR` | 但语义不提升 geometry，只保持 |
| guardrail / barrier / fence | `STRUCTURE_ANCHOR` 或 `STATIC_THING` | 若边界破碎，先放 `STATIC_THING` 更安全 |
| traffic sign / traffic light / road sign / pole | `STATIC_THING` | 静态但小，避免过度主导 |
| sky / tree / vegetation / grass / bush / water / glass / reflection | `LOW_VALUE_STUFF` | 只做轻度降权 |
| shadow | `LOW_VALUE_STUFF` 或忽略 | detector 若不稳定，不建议强用 |

### 6.2 Semantic-light 搜索

固定第二阶段最佳 geometry 配置，仅搜索语义强度：

| Run | 语义设置 | 目的 | 预期 |
|---|---|---|---|
| M0 | no semantic | geometry-only 主候选 | 第二阶段最佳 |
| M1 | KITTI prompt, $\rho_{mov}=0.30$, $r_{mov}=0.45$ | 弱语义参考 | 小幅提升或持平 |
| M2 | KITTI prompt, $\rho_{mov}=0.45$, $r_{mov}=0.60$ | 推荐主候选 | 车辆多时应提升 |
| M3 | KITTI prompt, $\rho_{mov}=0.60$, $r_{mov}=0.75$ | 测语义上限 | 漏检/错检会带来退化 |
| M4 | only movable semantics | 验证 car/person 是否贡献主要收益 | 若优于 full semantic，stuff 太噪 |
| M5 | only low-value stuff semantics | 验证 sky/tree 是否有用 | 可能小幅改善，也可能边界误伤 |
| M6 | indoor prompt 原版 | 验证 prompt 调整必要性 | 应不如 KITTI prompt |

**判定逻辑：**

如果 M2 比 M0 提升 2 m 以上，且没有片段崩坏，把 semantic-light 加入主模型。  
如果 M1/M2/M3 都不如 M0，主模型不要用语义，仅保留 KITTI prompt 作为可视化/诊断。  
如果 M4 好、M5 差，说明 thing semantics 有价值但 stuff 太粗糙；主模型只启用 movable suppression。  
如果 M5 好，说明 sky/vegetation 的大面积低价值压制有贡献；但仍要保持 $r_{max,stuff}\le 0.30$，不要提高边界不稳的 stuff trust。

---

## 7. 必做消融：证明主模型为什么有效

最终主模型不应该只报一个 ATE。至少要做下面这些消融，才能判断它真的是 write control 在起作用。

### 7.1 写入路径消融

| Run | 设置 | 解释 |
|---|---|---|
| A0 | native LoGeR | 原始模型 |
| A1 | unity replay | replay 本身控制 |
| A2 | token prior only, $\lambda_{write}=1$ | 看 token ranking 是否有效 |
| A3 | chunk budget only, $A_{tok}=1$ | 看整体写入强度是否有效 |
| A4 | token prior + chunk budget | 主模型 |

预期 A4 最好；如果 A3 接近 A4，说明收益主要来自少写，而非 token-level clean write；如果 A2 接近 A4，说明 token ranking 已经足够，chunk budget 可以减弱以提高泛化。

### 7.2 cue 消融

| Run | 设置 | 解释 |
|---|---|---|
| C0 | no $C_{dyn}$ | 动态风险是否必要 |
| C1 | explicit only | 显式几何贡献 |
| C2 | implicit only | 隐式 attention 是否可独立使用 |
| C3 | explicit + implicit 主融合 | 主模型 |
| C4 | no $C_{occ}$ penalty | 遮挡项是否误伤/必要 |
| C5 | no $C_{unc}$ penalty | 不确定项是否导致欠写 |
| C6 | no $C_{anchor}$ positive | anchor 是否真正提供正向写入价值 |

这里特别关注 C5。如果 no $C_{unc}$ 反而更好，说明 KITTI 01 上很多低纹理道路/天空被 `C_unc` 压得太狠，应该降低 $\lambda_u$ 或只让 `C_unc` 进入 chunk budget，不进入 token ranking。

### 7.3 semantic 消融

| Run | 设置 | 解释 |
|---|---|---|
| S0 | no semantic | geometry-only |
| S1 | movable only | 车/人/骑行者压制 |
| S2 | low-value stuff only | 天空/植被压制 |
| S3 | structure labels only | 验证 structure 不应主动 boost |
| S4 | full KITTI semantic-light | 主语义参考 |
| S5 | full KITTI semantic-heavy | 证明强语义会因前端噪声退化 |

预期 S1 有小幅收益，S2 可能视 stuff 质量决定，S3 应基本持平。如果 S3 明显提升，要警惕它是否是在粗糙 mask 下错误改变了 geometry prior；因为当前原则是不依赖语义提升结构写入。

### 7.4 patch pooling 消融

当前 patch prior 是 mean pooling。KITTI 中车辆边缘和遮挡边界可能被 patch mean 稀释，建议加一个 risk-pooling 候选：

$$
S_{pix}=1-A_{pix}
$$

$$
A_{patch}^{p90}=1-Q_{90}\left(S_{pix}\mid u\in\Pi_i\right)
$$

对比：

| Run | patch pooling | 预期 |
|---|---|---|
| P0 | mean | 当前默认，稳定 |
| P1 | risk p75 | 稍保守 |
| P2 | risk p90 | 强保守，可能更好地压边界 |
| P3 | 0.7 mean + 0.3 risk-p90 | 折中候选 |

如果 P2 好但 mean A_tok 太低，优先选 P3。

---

## 8. 每次实验应该保存的数据

为了让下一轮决策不是只靠 ATE，建议把 `run_pipeline_abc.py` 的 `output_pt` 从“只保存第一个 chunk”改成保存所有 chunk 的摘要。每个 chunk 至少保存：

```python
{
  "chunk_start": int,
  "chunk_end": int,
  "C_dyn_explicit_mean": float,
  "C_dyn_implicit_mean": float,
  "C_dyn_mean": float,
  "C_anchor_mean": float,
  "C_occ_mean": float,
  "C_unc_mean": float,
  "G_write_geo_mean": float,
  "A_tok_mean": float,
  "A_tok_q10": float,
  "A_tok_q90": float,
  "B_chunk_geo": float,
  "lambda_write_by_layer": list,
  "mean_prior_by_layer": list,
  "masklet_label_hist": dict,
  "masklet_group_hist": dict,
}
```

最终汇总时画出：

$$
ATE_{seg}(0{:}200),ATE_{seg}(200{:}400),\dots
$$

以及每个 chunk 的：

$$
\bar A_{tok},\quad B_{chunk\_geo},\quad \lambda_{write},\quad \bar C_{dyn},\quad \bar C_{anchor}
$$

如果一个配置 ATE 好，但在某些 chunk 的 $\bar A_{tok}$ 接近 0 或 $\lambda_{write}$ 长期低于 0.2，它不是优先主模型。

---

## 9. 推荐执行顺序与停止条件

### 第 0 天：复现与 replay gate

先跑 native 和 unity replay。只有二者通过，才进入正式搜索。

如果 native 没复现，停止，修评估协议。  
如果 unity replay 没对齐，停止，修 Stage E replay / token alignment。

### 第 1 天：geometry-only fusion 搜索

运行 F2 到 F9，不跑 Stage C。选择 2 个最佳 fusion 进入下一阶段。选择不是只看 ATE，还要看 prior 健康度。

期望结果：calibrated weighted soft-or 的 $\omega_{imp}=0.50$ 进入前二。如果 explicit-only 最好，则主模型改为 explicit-only。

### 第 2 天：support horizon 与 $G_{write\_geo}$ 搜索

在最佳 fusion 上跑 S1 到 S6，再跑 G0 到 G5。此时目标是 geometry-only ATE 先压到 35 m 以下。如果 geometry-only 已经低于 33 m，可以暂定为主模型候选，再做语义参考实验。

如果 geometry-only 仍高于 38 m，说明当前 prior 没有抓住 01 的主要误差来源；下一步不是加语义，而是检查：

1. $C_{dyn}$ 是否只在边缘响应；
2. $G_{write\_geo}$ 是否均值过高，动态车辆仍写入；
3. $G_{write\_geo}$ 是否均值过低，导致欠写；
4. TTT token alignment 是否错位。

### 第 3 天：Stage E 与 semantic-light

先跑 E0 到 E5，选出写入强度最稳的配置。然后只在最佳 1 到 2 个 geometry 配置上跑 M0 到 M6。

期望结果：M2 比 M0 小幅提升。如果没有提升，最终主模型就用 geometry-only，不强行带语义。

### 第 4 天：消融与主模型锁定

对最佳配置跑写入路径消融、cue 消融、semantic 消融和 patch pooling 消融。主模型锁定标准：

$$
ATE_{main}<33\text{ m}
$$

并且：

$$
ATE_{main}<ATE_{native}-8\text{ m}
$$

同时不能出现某个 200-frame segment 比 native 差超过 6 m。若达到 $ATE<30$ m，可直接作为第一版强主模型；若在 30–33 m，则作为可接受主模型，但要继续查后半段漂移。

---

## 10. 初始主模型配置建议

在还没有跑实验前，我建议的第一版主模型配置是：

```text
Dynamic Cue:
  fusion = calibrated_weighted_soft_or
  omega_imp = 0.50
  g0 = 0.25
  k_intra = 5
  sigma_pt = 0.25
  tau_occ = 0.05
  alpha_1 = 0.8
  alpha_3 = 0.5

G_write_geo:
  lambda_s = 1.0
  lambda_a = 0.8
  lambda_d = 1.4
  lambda_o = 0.5
  lambda_u = 0.6

Semantic Prior:
  use_g_write_geo = true
  semantic_mode = reference_only
  rho_structure = 0.10
  rho_static = 0.15
  rho_low_value = 0.25
  rho_movable = 0.45
  rho_uncertain = 0.20
  rmax_structure = 0.25
  rmax_static = 0.35
  rmax_low_value = 0.30
  rmax_movable = 0.60
  rmax_uncertain = 0.20
  a_min_special = 0.50

Stage E:
  lambda_min = 0.15
  lambda_max = 1.00
```

我对这个配置的预期是：geometry-only 版本能把 ATE 从 41.64 m 降到大约 32–36 m；加入 semantic-light 后，如果 KITTI prompt 能稳定抓到车辆和大面积天空/植被，可能再降 1–3 m。真正进入 <30 m 的概率取决于 implicit dynamic 是否可靠，以及 replay token alignment 是否完全正确。

---

## 11. 对下一步代码改动的最小建议

为了让上述实验可执行，建议先只加少量开关，不重构大模块：

1. `DynamicCueExtractor` 增加 `dyn_fusion_mode`：`explicit`、`implicit`、`max`、`avg`、`soft_or`、`calibrated_soft_or`。
2. 增加 `omega_imp`、`imp_q_low`、`imp_q_high`、`imp_anchor_gate_g0`。
3. `SemanticPriorGenerator` 增加 group-specific `rho_g` 和 `r_mask_cap_g`；如果不想改太多，先加 `semantic_strength_scale` 和 `mask_trust_cap`。
4. `run_pipeline_abc.py` 增加 `--stage_c_mode none/reference`，允许 geometry-only semantic write prior 不跑 Stage C。
5. `TTTWriteController` 增加 token alignment debug；在 `unity_replay` 和 synthetic prior 下确认 cached token 顺序。
6. `output_pt` 保存所有 chunk debug 摘要，而不是只存第一个 chunk。

这些改动都不改变 LoGeR apply path，只是让实验可控、可归因。

---

## 12. 最重要的实验判断原则

本轮实验的核心不是“哪个语义 prompt 看起来漂亮”，而是：

$$
\text{cleaner write} \Rightarrow \text{less long-horizon drift}
$$

如果一个配置降低了 $A_{tok}$，但没有降低 ATE，说明它只是少写，不是写得更干净。  
如果一个配置降低了车辆/遮挡边界的写入，同时保持道路/建筑/护栏的 anchor 写入，并且后半段 drift 下降，那才是我们要的主模型。  
如果语义前端不稳定，主模型宁愿退回 geometry-only，也不要让粗糙 masklet 主导 TTT memory。

# Semantic Prior Generator 设计文档（v1）

> 本文档定义 **Semantic Prior Generator** 模块。模块以 **Dynamic Cue Extractor** 输出的几何 cue 和 **Video Masklet Front-end** 输出的 semantic masklets 为输入，生成 patch token 与 special token 的 **write prior**，并把这份 prior 送入 **TTT Write Controller**。本文档重点说明：如何从语义类别、masklet 质量、几何静态性 / 动态性 / 不确定性中生成 `A_mask / A_pix / A_patch / A_tok`，以及如何让 sky/tree/water 等 low-value stuff 与 wall/floor/building 等 structure 区域在 write prior 上体现出明确差异。

---

## 0. 一句话定位

**Semantic Prior Generator 的职责，是把“几何上是否值得写”与“语义上是否值得写”统一到同一个 write-allow 表示中，并最终映射到 LoGeR token 空间。**

---

## 1. 模块目标

本模块在当前 pipeline 中承担三个核心任务：

1. 将每条 semantic masklet 的语义类别与质量，转成可比较的 **语义写入价值**；
2. 将几何 cue 按 masklet 聚合，再与语义写入价值结合，生成 **masklet-level write allow**；
3. 将 masklet-level prior 和 cue-only prior 融合为 **pixel-level** 与 **token-level** prior，供 TTT Write Controller 使用。

最终它要回答的是：

> 当前 chunk 中，每个 patch token 在多大程度上应该参与 fast weights 的写入？

---

## 2. 设计原则

### 2.1 输出的是 write-allow，不是类别真相

`A_tok` 的含义不是“这个 token 是背景还是前景”，而是“这个 token 在当前 chunk 中作为 memory write 来源的资格有多强”。

### 2.2 语义和几何在同一尺度上融合

几何 cue 提供“稳不稳”的证据，语义标签提供“值不值得写”的证据，两者必须共同决定 prior。

### 2.3 对被 masklet 覆盖的区域，语义优先；对未覆盖区域，几何回退

因为 masklet 提供更明确的语义解释；但前端不可能始终无缝覆盖整张图，所以必须为 uncovered 区域保留 cue-only 回退 prior。

### 2.4 thing 与 stuff 统一表示，但写入价值不同

例如：

- `wall / floor / building` 常常是高写入价值；
- `sky / tree / water / reflection` 常常是低写入价值；
- `person / vehicle / animal` 常常是动态风险源；
- `chair / box / table` 可能介于中间，需要结合几何 cue 决定。

### 2.5 token prior 应连续、可平滑、可解释

为了服务 TTT update，输出 prior 不能过于离散跳变。需要保留：

- masklet-level 分数
- pixel-level 分数
- patch token 分数

这样便于 debug 和 ablation。

---

## 3. 在整条 Pipeline 中的位置

```text
Dynamic Cue Extractor
        ↓
Video Masklet Front-end
        ↓
Semantic Prior Generator
        ↓
A_mask / A_pix / A_patch / A_tok
        ↓
TTT Write Controller
```

本模块是整个 pipeline 的融合核心：它第一次把几何证据和语义对象结果投到同一张 write prior 图上。

---

## 4. 输入定义

### 4.1 主输入

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| cue 张量 | `E_cue` | `[T, H, W, 5]` | `C_stat/C_dyn/C_occ/C_unc/C_anchor` |
| 几何写入图 | `G_write_geo` | `[T, H, W]` | cue-only write allow |
| semantic masklets | `M_mask` | `[J, T, H, W]` | 前端输出的 2D masklets |
| visibility | `V_mask` | `[J, T]` | masklet 可见性 |
| quality | `Q_mask` | `[J, T]` | masklet 每帧质量 |
| semantic label | `L_sem` | `[J]` 或 `[J,K]` | 离散标签或类别分布 |
| semantic group | `G_sem` | `[J]` | 一级语义组 |
| patch meta | `PatchMeta` | `[L_patch, 3]` | patch token 位置索引 |
| token type | `TokenType` | `[L_tok]` | patch / reg / role |

### 4.2 语义标签形式

支持两种形式：

#### A. 离散标签

$$
L_{sem}[j] \in \{1,\dots,K\}
$$

#### B. 类别分布

$$
\pi_j \in [0,1]^K, \qquad \sum_c \pi_j(c)=1
$$

当前模块内部统一把它们映射成 **语义基础权重**。

---

## 5. 输出定义

本模块输出三层 prior 表示。

### 5.1 masklet-level write allow

$$
A_{mask} \in [0,1]^{J \times T}
$$

表示第 `j` 条 masklet 在第 `t` 帧的写入允许程度。

### 5.2 pixel-level write allow

$$
A_{pix} \in [0,1]^{T \times H \times W}
$$

表示每个像素作为写入来源的允许程度。

### 5.3 token-level write allow

- patch token prior：
  $$
  A_{patch} \in [0,1]^{T \times H_{tok} \times W_{tok}}
  $$
- flatten 后的 patch prior：
  $$
  A_{patch-flat} \in [0,1]^{L_{patch}}
  $$
- special token prior：
  $$
  A_{special} \in [0,1]^{L_{special}}
  $$
- 最终 token prior：
  $$
  A_{tok} \in [0,1]^{L_{tok}}
  $$

### 5.4 suppression 表示

定义：

$$
S_{tok}=1-A_{tok}
$$

它更适合：

- 可视化当前 chunk 的高风险写入区域；
- 计算 chunk 的整体 suppress occupancy；
- 构造 TTT Write Controller 中的 block gain。

### 5.5 Debug 输出

推荐保留：

```text
PriorDebug = {
  semantic_base_weight,
  cue_aggregates_per_mask,
  overlap_resolution_map,
  uncovered_ratio,
  patch_pooling_stats,
  chunk_risk_summary,
}
```

---

## 6. 三层 prior 表示的作用

### 6.1 为什么要有 `A_mask`

因为语义标签、masklet 质量、thing/stuff 属性天然都定义在 masklet 粒度。

### 6.2 为什么还要有 `A_pix`

因为多个 masklet 会在像素级发生竞争，而未覆盖区域也必须被赋予回退 prior。

### 6.3 为什么最终一定要有 `A_tok`

因为 TTT update 的真正作用单元是 token，而不是像素。

因此，本模块的自然数据流是：

```text
semantic label + mask quality + geometry cues
        ↓
A_mask
        ↓
A_pix
        ↓
A_patch / A_special
        ↓
A_tok
```

---

## 7. 语义基础权重

### 7.1 一级语义组到基础权重的映射

建议定义一个语义基础写入权重表：

| 语义组 | 记号 | 默认权重 |
|---|---|---:|
| `STRUCTURE_ANCHOR` | `w_struct` | `1.00` |
| `STATIC_THING` | `w_staticthing` | `0.75` |
| `MOVABLE_THING` | `w_movable` | `0.25` |
| `LOW_VALUE_STUFF` | `w_lowstuff` | `0.10` |
| `UNCERTAIN_REGION` | `w_uncertain` | `0.15` |

这组默认值体现的是：

- 墙、地、建筑这类结构化区域优先写；
- 静态 thing 可中高权重；
- 可移动物体默认偏低；
- sky/tree/water 等 low-value stuff 默认很低；
- 明显不稳区域也偏低。

### 7.2 从离散标签得到基础权重

若使用离散标签：

$$
w_{sem}(j)=w_{L_{sem}[j]}
$$

### 7.3 从类别分布得到基础权重

若使用类别分布：

$$
w_{sem}(j)=\sum_{c=1}^{K} \pi_j(c)\, w_c
$$

### 7.4 语义熵修正（推荐）

如果类别分布不确定，可进一步引入熵惩罚：

$$
H(\pi_j)=-\sum_c \pi_j(c)\log(\pi_j(c)+\epsilon)
$$

$$
w^{\prime}_{sem}(j)=w_{sem}(j) \cdot \exp(-\tau_H H(\pi_j))
$$

这样语义越不确定，基础写入权重越低。

---

## 8. 几何 cue 按 masklet 聚合

对第 `j` 条 masklet 在第 `t` 帧，定义：

$$
\bar C_k(j,t)=
\frac{\sum_{h,w} M_{mask}(j,t,h,w)\, C_k(t,h,w)}{\sum_{h,w} M_{mask}(j,t,h,w)+\epsilon}
$$

其中：

$$
k \in \{stat, dyn, occ, unc, anchor\}
$$

此外，定义 masklet 面积占比：

$$
a(j,t)=\frac{\sum_{h,w} M_{mask}(j,t,h,w)}{H\cdot W}
$$

以及有效质量：

$$
q(j,t)=Q_{mask}(j,t)\cdot V_{mask}(j,t)
$$

---

## 9. `A_mask` 的构造

### 9.1 参数化版本

定义第 `j` 条 masklet 在第 `t` 帧的 logits：

$$
z_{j,t}=
 b_0
 + b_1 \bar C_{stat}(j,t)
 + b_2 \bar C_{anchor}(j,t)
 - b_3 \bar C_{dyn}(j,t)
 - b_4 \bar C_{occ}(j,t)
 - b_5 \bar C_{unc}(j,t)
 + b_6 q(j,t)
 + b_7 w_{sem}(j)
 + b_8 \log(a(j,t)+\epsilon)
$$

然后：

$$
A_{mask}(j,t)=\sigma(z_{j,t})
$$

这个形式的直觉非常清楚：

- 静态性、锚点资格、mask 质量、语义写入价值越高，`A_mask` 越高；
- 动态性、遮挡性、不确定性越高，`A_mask` 越低。

### 9.2 规则型版本

如果第一版不使用学习参数，可以用规则型版本：

$$
A_{mask}(j,t)=
\operatorname{clip}\Big(
 w_{sem}(j)
 \cdot (\lambda_s \bar C_{stat} + \lambda_a \bar C_{anchor} + \lambda_q q)
 \cdot (1-\lambda_d \bar C_{dyn})
 \cdot (1-\lambda_o \bar C_{occ})
 \cdot (1-\lambda_u \bar C_{unc}),
0,1\Big)
$$

### 9.3 stuff 类额外约束

对于 `LOW_VALUE_STUFF`，建议引入额外上限：

$$
A_{mask}(j,t) \leftarrow \min\big(A_{mask}(j,t), a^{max}_{lowstuff}\big)
$$

例如：

$$
a^{max}_{lowstuff}=0.25
$$

这样即使天空在几何上局部稳定，也不会在 write prior 中占据过高权重。

---

## 10. cue-only pixel prior

在没有任何 masklet 覆盖的区域，直接使用几何生成回退 prior。定义：

$$
z^{cue}(t,h,w)=
 c_0
 + c_1 C_{stat}(t,h,w)
 + c_2 C_{anchor}(t,h,w)
 - c_3 C_{dyn}(t,h,w)
 - c_4 C_{occ}(t,h,w)
 - c_5 C_{unc}(t,h,w)
$$

$$
A^{cue}_{pix}(t,h,w)=\sigma(z^{cue}(t,h,w))
$$

实际上，如果 Dynamic Cue Extractor 已经输出 `G_write_geo`，也可以直接令：

$$
A^{cue}_{pix}=G_{write-geo}
$$

---

## 11. 从 `A_mask` 到 `A_pix`

### 11.1 mask 驱动像素 prior

定义：

$$
A^{mask}_{pix}(t,h,w)=
\max_j \Big(M_{mask}(j,t,h,w)\cdot A_{mask}(j,t)\Big)
$$

### 11.2 覆盖指示图

$$
m_{cov}(t,h,w)=\max_j M_{mask}(j,t,h,w)
$$

### 11.3 融合公式

$$
A_{pix}(t,h,w)=
 m_{cov}(t,h,w) \cdot A^{mask}_{pix}(t,h,w)
 + (1-m_{cov}(t,h,w)) \cdot A^{cue}_{pix}(t,h,w)
$$

### 11.4 重叠区域竞争

如果多个 masklet 重叠，可采用以下优先级：

1. 更高 `A_mask`
2. 更高 `Q_mask`
3. 更高语义写入价值 `w_sem`
4. 更稳定的 semantic group（如 structure 优先于 movable）

可写成分数：

$$
Score_{ov}(j,t)=\mu_1 A_{mask}(j,t)+\mu_2 Q_{mask}(j,t)+\mu_3 w_{sem}(j)
$$

再选取得分最高者。

---

## 12. Pixel-to-token 映射

### 12.1 patch token prior

对每个 patch 区域 `\Pi(t,y_{tok},x_{tok})`：

$$
A_{patch}(t,y_{tok},x_{tok})=
\operatorname{Mean}_{(h,w)\in\Pi} A_{pix}(t,h,w)
$$

flatten 后：

$$
A_{patch-flat} \in [0,1]^{L_{patch}}
$$

### 12.2 风险 pooling 的可选版本

若希望更保守地压制 patch 内局部高风险区域，可定义：

$$
S_{patch}(t,y_{tok},x_{tok})=
\operatorname{p90}_{(h,w)\in\Pi}(1-A_{pix}(t,h,w))
$$

$$
A_{patch}=1-S_{patch}
$$

### 12.3 语义类别统计的 patch 输出（可选）

还可以输出 patch 的语义组直方图：

$$
H_{patch}(t,y_{tok},x_{tok},g)
$$

它对 debug 非常有帮助，例如可以观察：

- 哪些 token 主要来自 sky / tree / wall / floor；
- 哪些 token 被 movable thing 主导。

---

## 13. Special token prior

register / role tokens 不直接对应单个像素区域，需要单独定义。

### 13.1 chunk 级统计量

定义：

- 动态风险占比：
  $$
  \rho_{dyn} = \frac{1}{L_{patch}}\sum_i (1-A_{patch-flat}[i])\,\mathbf{1}[patch_i \text{来自动态风险主导}]
  $$

- low-value stuff 占比：
  $$
  \rho_{lowsem} = \frac{1}{L_{patch}}\sum_i \mathbf{1}[patch_i \text{来自 low-value stuff 主导}]
  $$

- anchor patch 占比：
  $$
  \rho_{anchor} = \frac{1}{L_{patch}}\sum_i \mathbf{1}[A_{patch-flat}[i] > \tau_{anchor}]
  $$

### 13.2 register token prior

可定义：

$$
A_{reg} = \operatorname{clip}
\big(1-\kappa_r \rho_{dyn}-\kappa_s \rho_{lowsem}+\kappa_a \rho_{anchor}, a^{min}_{reg}, 1\big)
$$

### 13.3 role token prior

由于 role token 更偏 chunk 内位置语义，可定义：

$$
A_{role} = \operatorname{clip}
\big(1-\kappa_{role} \rho_{dyn}, a^{min}_{role}, 1\big)
$$

### 13.4 最终拼接

$$
A_{tok} = \operatorname{concat}(A_{patch-flat}, A_{special})
$$

其中：

$$
A_{special}=[A_{reg}, \dots, A_{reg}, A_{role}, \dots]
$$

---

## 14. token prior 的时间平滑与稳定性

因为 `A_tok` 最终直接控制 TTT update，所以建议保留一定的时间平滑。

### 14.1 frame 内平滑

对 `A_pix` 可做局部平滑：

- 仅在同类语义区域内平滑；
- 避免跨语义边界抹平。

### 14.2 chunk 内平滑

对同一条 masklet 的 `A_mask(j,t)`，可做 EMA：

$$
\hat A_{mask}(j,t)=\beta \hat A_{mask}(j,t-1)+(1-\beta)A_{mask}(j,t)
$$

### 14.3 prior clipping

为避免极端值，建议最终 token prior 做 clipping：

$$
A_{tok} \leftarrow \operatorname{clip}(A_{tok}, a_{min}, a_{max})
$$

通常可取：

- `a_min = 0.02 ~ 0.05`
- `a_max = 0.95 ~ 1.0`

---

## 15. 与 TTT Write Controller 的接口

本模块对 Stage E 至少输出：

| 名称 | Shape | 用途 |
|---|---:|---|
| `A_tok` | `[L_tok]` | token-wise write allow |
| `A_patch-flat` | `[L_patch]` | patch token prior debug |
| `A_special` | `[L_special]` | special token prior |
| `S_tok` | `[L_tok]` | suppression occupancy 统计 |
| `PriorDebug` | 结构体 | 可视化与 ablation |

Stage E 会基于 `A_tok` 生成：

$$
p_{m,l,i}^{(r,h)}
$$

并把它乘到 LoGeR 当前已有的 `\eta` 上。

---

## 16. 推荐默认语义权重表

建议在第一版中显式写死一张表，便于 debug。

| 类别/语义组 | 默认权重 |
|---|---:|
| wall / floor / building / road / stair | 1.00 |
| door / window / ceiling | 0.90 |
| cabinet / fixed furniture / large static device | 0.75 |
| chair / table / sofa / box | 0.40 |
| person / rider / bicycle / motorcycle | 0.15 |
| car / bus / truck / train | 0.10 |
| animal | 0.10 |
| sky | 0.05 |
| tree / vegetation | 0.08 |
| water / reflection / smoke / screen | 0.05 |
| unknown / fragment / uncertain | 0.15 |

这张表不是最终真理，而是第一版可运行的先验起点。

---

## 17. 实现优先级

### Phase 1：最小可运行版本

先实现：

1. 语义组到基础权重的映射；
2. masklet 内 cue 聚合；
3. `A_mask -> A_pix -> A_patch -> A_tok`；
4. uncovered 区域 cue-only 回退。

### Phase 2：增强 special token prior

加入：

1. chunk risk summary；
2. `A_special`；
3. `S_tok` 统计；
4. 更稳定的 patch 风险 pooling。

### Phase 3：增强语义不确定性与时间平滑

加入：

1. 语义熵惩罚；
2. masklet temporal smoothing；
3. patch semantic histograms；
4. 更细类别的权重表。

---

## 18. 常见失败模式与缓解策略

### 18.1 sky / tree 未被显式分割，导致 prior 过高

**缓解**：前端必须显式输出 stuff masklets；未被 mask 覆盖时再退回 cue-only prior。

### 18.2 geometry 很稳，但 movable thing 被写得过多

**缓解**：语义基础权重应对 `MOVABLE_THING` 设明显压低；`A_mask` 中不能只看 `C_stat`。

### 18.3 大片背景未被前端覆盖，导致 prior 过度依赖语义分割

**缓解**：保留 `G_write_geo` 作为回退；确保 uncovered 背景也有 prior。

### 18.4 patch pooling 抹掉局部高风险边界

**缓解**：用 `p90` 风险 pooling 或混合 pooling。

### 18.5 语义分类不稳导致 prior 抖动

**缓解**：类别分布 + 熵惩罚 + temporal smoothing。

---

## 19. 最终接口摘要

### 输入

- `E_cue: [T,H,W,5]`
- `G_write_geo: [T,H,W]`
- `M_mask: [J,T,H,W]`
- `V_mask: [J,T]`
- `Q_mask: [J,T]`
- `L_sem: [J]` 或 `[J,K]`
- `G_sem: [J]`
- `PatchMeta: [L_patch,3]`
- `TokenType: [L_tok]`

### 输出

- `A_mask: [J,T]`
- `A_pix: [T,H,W]`
- `A_patch: [T,H_tok,W_tok]`
- `A_patch-flat: [L_patch]`
- `A_special: [L_special]`
- `A_tok: [L_tok]`
- `S_tok: [L_tok]`
- `PriorDebug`

### 一句话总结

**Semantic Prior Generator 的核心作用，是把“几何上是否稳定”和“语义上是否值得写”统一映射成 LoGeR token 的写入先验，使 wall/floor/building 等结构区域主导 memory write，而 sky/tree/water 及动态风险区域在 token 级别被系统性降权。**

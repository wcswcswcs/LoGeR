# Semantic Prior Generator Redesign v2

> 目标：将 Semantic Prior Generator 从“单一混合打分器”重构为一套 **职责分离（decoupled write policy）** 的写入策略模块。  
> 新版本明确拆分三类决策：  
> 1. **Eligibility（写入资格）**：当前区域从几何上是否适合写入；  
> 2. **Value（写入价值）**：在几何允许的前提下，语义上更值得写多少；  
> 3. **Budget（写入预算）**：当前 chunk 整体应当写多少。  
>
> 本设计用于替代 `Semantic_Prior_Generator_Design_v1.md` 中的旧版设计，并保持与 LoGeR TTT pipeline 的总体接口兼容。

---

## 0. One-line summary

**v2 的核心思想不是把所有信号塞进一个 prior score，而是给不同信号分配单一职责：几何决定资格，语义决定价值，mask 质量决定是否信任语义分支，chunk 级预算主要由几何决定。**

---

## 1. Why redesign v1?

v1 是一个能工作的设计，但它把多种职责揉在了一起，导致超参数耦合、归因困难，以及 write prior 的绝对意义不够稳定。

### 1.1 v1 中存在的主要问题

#### Problem A. Geometry / semantics / mask quality are mixed in the same score

v1 中 `A_mask(j,t)` 的 logit 同时包含：

- `\bar C_stat, \bar C_anchor`
- `\bar C_dyn, \bar C_occ, \bar C_unc`
- `Q_mask`
- `w_sem`
- 甚至可选面积项

它们一起进入同一个 logit：

\[
z_{j,t}^{v1} =
b_0
+ b_1 \bar C_{stat}
+ b_2 \bar C_{anchor}
- b_3 \bar C_{dyn}
- b_4 \bar C_{occ}
- b_5 \bar C_{unc}
+ b_6 Q_{mask}
+ b_7 w_{sem}
+ \cdots
\]

这种写法的好处是简单，但坏处也非常明显：

- **几何稳定性**、**语义价值**、**masklet 可信度** 被当成同类因素处理；
- 当一个区域被压低时，很难判断是 **几何风险** 起作用，还是 **语义类别** 起作用，还是 **mask 质量** 起作用；
- 参数一多以后，任何类别都可能被“双重压制”。

典型例子：

- `person` 的 `w_sem` 本来就低；
- 同时 `C_dyn` 的惩罚也强；
- 结果该区域被同时从“语义上”和“动态上”压两次。

这不一定错，但会让调参与归因变得非常困难。

#### Problem B. Mask quality is treated as a direct prior term

v1 把 `Q_mask` 直接加进 `A_mask` logit。  
但 `Q_mask` 的真正语义更接近：

> “我有多信这条 masklet 对应的语义解释”

而不是：

> “这个区域本身就应该多写或少写”

也就是说，低质量 masklet 不一定意味着该区域不值得写；它更可能意味着：

- 我不应该太相信 masklet 分支；
- 我应该更多退回 geometry cue 分支。

#### Problem C. The old controller partially normalizes away the absolute level of the prior

v1 在 Stage E 中使用：

\[
\beta_i = \eta_i \cdot A_{tok,i}
\]

\[
\gamma_i^{v1} =
\frac{\beta_i}
{\frac{1}{|C_m|}\sum_j \beta_j + \varepsilon}
\]

如果一个 chunk 中所有 token 的 prior 都被统一乘上一个常数 `\alpha`，即：

\[
A'_{tok,i} = \alpha A_{tok,i}
\]

则有：

\[
\gamma_i' \approx \gamma_i
\]

也就是说：

- **prior 的相对分布** 会保留下来；
- **prior 的绝对均值** 会被归一化掉。

这会带来一个实际问题：

> 如果整段 chunk 都是低价值区域，或者所有 token 都被一致性地压低，v1 的 token prior 在 pre-zeropower 聚合中未必真的会明显减小总写入。

于是 v1 需要再用 `\lambda_{write}` 从 chunk 级别补一次刹车。

#### Problem D. λ_write in v1 is entangled with semantics

v1 建议用 `\bar A_{patch}` 来构造 `\lambda_{write}`：

\[
\lambda_{write}^{v1}
=
\lambda_{min}
+
(\lambda_{max}-\lambda_{min})\cdot \bar A_{patch}
\]

而 `\bar A_{patch}` 已经混合了：

- geometry cue
- semantic value
- overlap resolution
- mask quality

结果是：

- token-level prior 已经压过一次；
- chunk-level budget 又根据同一套混合 prior 再压一次。

这就很容易出现 **重复抑制**，而且你很难分清楚究竟是：

- 局部 token ranking 错了；
- 还是 chunk 预算过低了。

#### Problem E. Too many knobs live at the same layer

v1 同时暴露了：

- `b_i`
- `c_i`
- overlap `\mu_i`
- special token `\kappa_i`
- `a_min`, `a_max`
- 语义表
- 可选 area / entropy / smoothing / layer schedule

这些自由度并不是都没用，而是太多参数都在 **同一层级** 上竞争解释权。

---

## 2. Design objectives of v2

v2 的目标不是让 prior 更复杂，而是让它**更可解释、更稳、更容易调试**。

### Objective 1. One signal, one primary responsibility

- Geometry cues: 决定 **eligibility**
- Semantic labels: 决定 **value**
- Mask quality: 决定 **trust / routing**
- Chunk budget: 决定 **overall write magnitude**

### Objective 2. Preserve the absolute meaning of the token prior

v2 希望 `A_tok` 不仅能表达“谁比谁更重要”，还要保留“这一整段 chunk 值不值得写很多”的绝对含义。

### Objective 3. Remove semantic-global double suppression

语义应该主要影响：

- 哪些 token 更值得主导写入；
- 低价值对象在 token 级别被降权。

但语义不应该在 token 级和 chunk 级用同样方式重复压制。

### Objective 4. Keep the first version small

v2 baseline 只保留少量核心旋钮，不在第一版就引入：

- layer schedule
- branch/head-specific policy
- complex special-token semantics
- area-dependent heuristics
- too many overlap rules

---

## 3. Core idea: decoupled write policy

v2 把 Semantic Prior Generator 拆成三条支路：

```text
E_cue
  ├── Geometry Eligibility Branch  -> Elig_pix
  ├── Semantic Value Branch        -> v_sem(j)
  └── Mask Trust Branch            -> r_mask(j,t)

Elig_pix + v_sem + r_mask
        ↓
A_mask
        ↓
A_pix
        ↓
A_patch / A_special / A_tok
        ↓
B_chunk_geo
```

这三条支路分别回答：

1. **Eligibility branch**  
   当前像素从几何上是否适合进入长期 memory。

2. **Semantic value branch**  
   在“可以写”的前提下，这个对象从语义上更值得写多少。

3. **Mask trust branch**  
   当前应当多大程度上信任 masklet 分支，而不是退回几何 prior。

---

## 4. New input / output contract

### 4.1 Inputs

保持与 v1 基本兼容：

| Name | Symbol | Shape | Meaning |
|---|---|---:|---|
| cue tensor | `E_cue` | `[T,H,W,5]` | `C_stat/C_dyn/C_occ/C_unc/C_anchor` |
| geometry write map (optional) | `G_write_geo` | `[T,H,W]` | 如果 Stage B 已产出可直接复用 |
| masklets | `M_mask` | `[J,T,H,W]` | semantic masklets |
| visibility | `V_mask` | `[J,T]` | visibility |
| quality | `Q_mask` | `[J,T]` | masklet quality |
| semantic label / distribution | `L_sem` or `\pi_j` | `[J]` or `[J,K]` | semantic info |
| patch meta | `PatchMeta` | `[L_patch,3]` | patch token mapping |
| token type | `TokenType` | `[L_tok]` | patch / special |

### 4.2 Outputs

v2 输出两类核心结果：

#### A. Token-level absolute write gate

- `A_mask: [J,T]`
- `A_pix: [T,H,W]`
- `A_patch: [T,H_tok,W_tok]`
- `A_patch_flat: [L_patch]`
- `A_special: [L_special]`
- `A_tok: [L_tok]`

#### B. Geometry-driven chunk budget statistic

- `B_chunk_geo: []`

这个标量将用于 Stage E 中构造全局写入预算 `\lambda_{write}`。

### 4.3 Key interface difference from v1

**v1 主要输出 token-wise prior，chunk budget 间接从 `A_patch` 推导。**  
**v2 明确输出两种不同信号：**

1. `A_tok`：token 级 absolute write gate  
2. `B_chunk_geo`：chunk 级几何预算统计

这使得“token 局部谁写”和“chunk 整体写多少”不再被同一个混合量隐式承担。

---

## 5. Branch 1: Geometry Eligibility Branch

这一支路的目标是只根据 geometry cue 生成 **eligibility**，不掺入语义类别。

### 5.1 Positive evidence and risk evidence

定义正向几何证据：

\[
P_{pos}(t,h,w)
=
\omega_{stat} C_{stat}(t,h,w)
+
\omega_{anchor} C_{anchor}(t,h,w)
\]

默认第一版固定：

\[
\omega_{stat}=0.5,\qquad \omega_{anchor}=0.5
\]

定义风险证据：

\[
P_{risk}(t,h,w)
=
\omega_{dyn} C_{dyn}(t,h,w)
+
\omega_{occ} C_{occ}(t,h,w)
+
\omega_{unc} C_{unc}(t,h,w)
\]

默认第一版固定：

\[
\omega_{dyn}=0.5,\qquad
\omega_{occ}=0.25,\qquad
\omega_{unc}=0.25
\]

### 5.2 Pixel eligibility

定义几何 eligibility：

\[
Elig_{pix}(t,h,w)
=
\sigma\Big(
k_{pos} P_{pos}(t,h,w)
-
k_{risk} P_{risk}(t,h,w)
+
b_{elig}
\Big)
\]

其中：

- `k_pos`：静态/锚点鼓励强度
- `k_risk`：动态/遮挡/不确定惩罚强度

如果 Stage B 已经输出 `G_write_geo` 且与上述意义一致，也可以直接令：

\[
Elig_{pix} = G_{write\_geo}
\]

### 5.3 Motivation

这一支路只回答一件事：

> “这里从几何上看，是否有资格进入长期 memory？”

它不回答“这是什么对象”，也不回答“我是否应该信任 masklet”。

---

## 6. Branch 2: Semantic Value Branch

这一支路只负责给语义类别分配 **write value**，不再参与几何风险计算。

### 6.1 Four-bucket semantic table

第一版建议压缩成 4 个语义桶：

| Bucket | Meaning | Default value |
|---|---|---:|
| `STRUCTURE` | wall / floor / building / road / stair / ceiling | `1.00` |
| `BACKGROUND` | stable generic background / fixed furniture / static device | `0.70` |
| `DISTRACTOR` | vegetation / screen / reflection / weakly useful stuff | `0.40` |
| `MOVABLE` | person / car / animal / bicycle / hand-held object | `0.10` |

若存在语义分布 `\pi_j(c)`，则：

\[
v_{sem}(j)=\sum_c \pi_j(c)\, v_c
\]

若是离散标签，则直接查表。

### 6.2 Optional semantic certainty correction

若使用类别分布，可选地加入语义熵修正：

\[
H(\pi_j)=-\sum_c \pi_j(c)\log(\pi_j(c)+\epsilon)
\]

\[
u_{sem}(j)=\exp(-\tau_H H(\pi_j))
\]

其中 `u_sem(j)` 不直接改变写入价值，而优先用于后面的 mask trust 支路。

### 6.3 Motivation

这一支路只回答：

> “在已经几何可写的前提下，这类对象从长期记忆价值角度，更值得写多少？”

因此：

- 语义更像 **value prior**
- 几何更像 **eligibility evidence**

---

## 7. Branch 3: Mask Trust Branch

这一支路只决定：

> “当前这个像素，要不要信 masklet 分支，而不是退回 geometry-only prior？”

### 7.1 Mask trust

定义：

\[
r_{mask}(j,t)
=
V_{mask}(j,t)\cdot Q_{mask}(j,t)\cdot u_{sem}(j)
\]

若没有类别分布，则令：

\[
u_{sem}(j)=1
\]

因此第一版最简单可写成：

\[
r_{mask}(j,t)=V_{mask}(j,t)\cdot Q_{mask}(j,t)
\]

### 7.2 Motivation

这和 v1 的根本区别是：

- v1：`Q_mask` 直接进入 prior score
- v2：`Q_mask` 主要用于决定 **masklet 分支可信度**

因此低质量 masklet 的效果是：

- 更少主导最终 `A_pix`
- 更多退回 `Elig_pix`

而不是直接把该区域当成“低价值区域”。

---

## 8. Masklet-level write gate

### 8.1 Aggregate eligibility inside each masklet

对第 `j` 条 masklet 在第 `t` 帧，定义：

\[
\bar E(j,t)
=
\frac{\sum_{h,w} M_{mask}(j,t,h,w)\, Elig_{pix}(t,h,w)}
{\sum_{h,w} M_{mask}(j,t,h,w)+\epsilon}
\]

### 8.2 Construct A_mask

v2 中 masklet-level write gate 定义为：

\[
A_{mask}(j,t)
=
\operatorname{clip}
\Big(
\bar E(j,t)\cdot\big((1-\rho_{sem})+\rho_{sem} v_{sem}(j)\big),
0,1
\Big)
\]

其中：

- `\bar E(j,t)`：该对象区域几何上有多“可写”
- `v_sem(j)`：该对象语义上有多“有价值”
- `\rho_sem \in [0,1]`：语义调制强度

### 8.3 Why this form?

这个公式的设计动机是：

1. **几何 eligibility 是前提**  
   如果几何上不稳定，则语义不应强行救回来。

2. **语义只做调制，不做绝对 veto**  
   `MOVABLE` 会被明显降权，但不会仅因类别就被打成零。

3. **可解释性强**  
   一个对象分数低，是因为：
   - 它几何上不可写；
   - 或者它几何可写但语义价值低；
   - 两者可以明确区分。

### 8.4 Comparison to v1

与 v1 的主要区别是：

- v1：所有 cue + 语义 + 质量一起进一个 logit
- v2：`A_mask` 只由 **eligibility × semantic value** 构成
- mask trust 不再直接进入 `A_mask`

---

## 9. Pixel fusion

### 9.1 Overlap winner

定义像素 `(t,h,w)` 的候选 winner 为：

\[
j^\*(t,h,w)
=
\arg\max_j
\Big(
M_{mask}(j,t,h,w)\cdot r_{mask}(j,t)\cdot A_{mask}(j,t)
\Big)
\]

若没有任何 mask 覆盖该像素，则记为“no winner”。

### 9.2 Semantic branch score and trust

若存在 winner，则定义：

\[
A^{sem}_{pix}(t,h,w)=A_{mask}(j^\*,t)
\]

\[
R^{mask}_{pix}(t,h,w)=r_{mask}(j^\*,t)
\]

若不存在 winner，则：

\[
A^{sem}_{pix}(t,h,w)=0,\qquad
R^{mask}_{pix}(t,h,w)=0
\]

### 9.3 Final pixel prior

最终像素级 absolute write gate 为：

\[
A_{pix}(t,h,w)
=
R^{mask}_{pix}(t,h,w)\cdot A^{sem}_{pix}(t,h,w)
+
\big(1-R^{mask}_{pix}(t,h,w)\big)\cdot Elig_{pix}(t,h,w)
\]

### 9.4 Interpretation

这个融合比 v1 的 hard switch 更稳：

- mask 质量高、语义可信时：semantic branch 主导
- mask 质量低、语义不稳时：自动退回 geometry branch
- uncovered 区域：天然退回 geometry branch

### 9.5 Why not use hard semantic priority?

v1 在覆盖区域里更接近“只要被 mask 覆盖，就优先用 mask 分支”。  
v2 不这样做，因为：

- 被覆盖不等于 masklet 就可信；
- 低质量 masklet 的正确处理方式应该是 **减小语义分支权重**，而不是强行替代 cue-only prior。

---

## 10. Pixel-to-token projection

### 10.1 Patch prior

对每个 patch 区域 `\Pi(t,y_{tok},x_{tok})`：

\[
A_{patch}(t,y_{tok},x_{tok})
=
\operatorname{Mean}_{(h,w)\in\Pi}
A_{pix}(t,h,w)
\]

flatten 后得到：

\[
A_{patch\_flat}\in[0,1]^{L_{patch}}
\]

### 10.2 Geometry patch summary

同时保留 geometry-only patch summary：

\[
E_{patch}(t,y_{tok},x_{tok})
=
\operatorname{Mean}_{(h,w)\in\Pi}
Elig_{pix}(t,h,w)
\]

flatten 后：

\[
E_{patch\_flat}\in[0,1]^{L_{patch}}
\]

这是 v2 很关键的新增量，因为它将用于 chunk budget，而不再依赖 `A_patch`。

### 10.3 Special token prior

v2 baseline 不再为 special tokens 设计复杂的语义规则。  
统一用 geometry-only chunk summary 生成：

\[
B_{chunk\_geo}
=
\frac{1}{L_{patch}}
\sum_{i=1}^{L_{patch}} E_{patch\_flat}[i]
\]

\[
A_{special}
=
\operatorname{clip}
\Big(
a^{min}_{special}
+
(1-a^{min}_{special})\cdot B_{chunk\_geo},
a^{min}_{special},
1
\Big)
\]

若有多个 special tokens，第一版直接共享同一个值。

### 10.4 Final token prior

\[
A_{tok}=\operatorname{concat}(A_{patch\_flat}, A_{special})
\]

---

## 11. New Stage E interface

这是 v2 相比 v1 的一个关键变化。

### 11.1 Why Stage E should change

如果仍使用 v1 的：

\[
\gamma_i^{v1} =
\frac{\eta_i A_{tok,i}}
{\operatorname{Mean}_j(\eta_j A_{tok,j})+\epsilon}
\]

则 `A_tok` 的均值会被大幅归一化掉。  
这与 v2 想让 `A_tok` 保留 **absolute write meaning** 的目标冲突。

### 11.2 Recommended coefficient in v2

v2 建议改为：

\[
\gamma_i^{v2}
=
\frac{\eta_i A_{tok,i}}
{\operatorname{Mean}_j(\eta_j)+\epsilon}
\]

它的含义是：

- `\eta_i`：LoGeR 原生的 token 权重
- `A_{tok,i}`：外部 prior 提供的 absolute gate
- 分母只用来稳定 `\eta` 的整体尺度，不再把 `A_tok` 的均值归一化掉

这样：

- 当整个 chunk 的 `A_tok` 都偏低时，整体写入会真的偏小；
- 当只有部分 token 的 `A_tok` 高时，它们会相对更主导写入。

### 11.3 Chunk-level write budget

v2 中：

\[
\lambda_{write}^{v2}
=
\lambda_{min}
+
(\lambda_{max}-\lambda_{min})\cdot B_{chunk\_geo}
\]

这里的预算只看 geometry eligibility 的 chunk 平均水平。

### 11.4 Final write-back

于是：

\[
\tilde G_m
=
\sum_{i\in C_m} \gamma_i^{v2}\, J_i
\]

\[
G_{app} = \mathcal U_{dir}(\tilde G_m)
\]

\[
W_{m+1}
=
\mathrm{Renorm}
\Big(
W_m
+
\lambda_{write}^{v2}\, G_{app}
\Big)
\]

### 11.5 Motivation

这一步的设计意图是：

- `A_tok` 负责 **token-level absolute gate**
- `\lambda_{write}` 负责 **chunk-level geometry budget**
- 语义不再在 token 级和 chunk 级重复压制
- 几何则在局部和全局两个尺度上共同做 safety control

---

## 12. Comparison with v1

### 12.1 High-level summary table

| Aspect | v1 | v2 |
|---|---|---|
| Core design | one mixed scoring pipeline | decoupled write policy |
| Geometry role | mixed with semantics in same score | decides eligibility |
| Semantics role | mixed with geometry in same score | decides value modulation |
| Mask quality role | direct prior term | trust / routing coefficient |
| Covered region fusion | mask-priority hard switch | reliability-weighted soft fusion |
| Patch prior meaning | relative-ish and mixed | absolute write gate |
| Stage E normalization | may normalize away prior mean | preserves prior absolute scale |
| Chunk budget | derived from mixed `A_patch` | derived from geometry-only `B_chunk_geo` |
| Special token logic | separate semantic/risk formulas | geometry-only baseline |
| Hyperparameter profile | many coupled knobs | few, role-separated knobs |

### 12.2 What is intentionally removed in v2 baseline

v2 baseline 暂时不引入：

- `b_i / c_i` 两套并行 logit 参数
- overlap `\mu_i` 的复杂规则
- area-dependent correction
- special token 的语义类别统计项
- layer schedule
- head / branch specific prior
- stuff cap / many per-class handcrafted exceptions

这些能力不是永远不要，而是**不应该成为第一版语义 prior 的基本结构**。

---

## 13. Recommended hyperparameters

v2 baseline 的调参空间有意压缩为少量主旋钮。

### 13.1 Fixed weights (recommended fixed in v2 baseline)

\[
\omega_{stat}=0.5,\quad
\omega_{anchor}=0.5
\]

\[
\omega_{dyn}=0.5,\quad
\omega_{occ}=0.25,\quad
\omega_{unc}=0.25
\]

### 13.2 Main knobs

| Knob | Meaning | Suggested range |
|---|---|---:|
| `k_pos` | geometry positive gain | `1.0 ~ 2.5` |
| `k_risk` | geometry risk penalty | `2.0 ~ 4.0` |
| `\rho_sem` | semantic modulation strength | `0.4 ~ 0.8` |
| semantic table | 4-bucket values | fixed first, then tune |
| `a^{min}_{special}` | minimum special-token gate | `0.2 ~ 0.4` |
| `\lambda_{min}` | minimum chunk budget | `0.0 ~ 0.2` |
| `\lambda_{max}` | maximum chunk budget | `1.0` |

### 13.3 Optional knob

| Knob | Meaning | Default |
|---|---|---:|
| `\tau_H` | semantic entropy trust sensitivity | `0` (off) |

---

## 14. Expected behavior on typical cases

### Case A. Stable wall / floor / building

- `Elig_pix` 高
- `v_sem` 高
- `r_mask` 若也高，则 `A_pix` 高
- `A_tok` 高
- 这类区域应稳定主导 memory write

### Case B. Person or vehicle with clear motion

- `Elig_pix` 往往不高，或不稳定
- `v_sem` 很低
- 即使局部几何看起来一时稳定，也会被语义调制明显降权
- 若 mask 质量差，则自动更多退回 geometry branch

### Case C. Large low-quality mask over a good background region

- `Elig_pix` 可能很高
- 但 `r_mask` 低
- 最终 `A_pix` 更接近 `Elig_pix`
- 这避免了“差 mask 把好背景带偏”

### Case D. Whole chunk dominated by low-value content

- 在 v1 中，这类 chunk 的 prior 均值有可能被 Stage E 的归一化部分吞掉；
- 在 v2 中，`A_tok` 的均值保留下来，因此整体写入会真实变小；
- 同时 `\lambda_{write}` 主要反映 geometry safety，而非语义重复压制。

---

## 15. Minimal implementation plan

### Phase 1. Replace the old core logic

1. 用 `Elig_pix` 替代旧的 cue-only mixed score；
2. 用 4-bucket semantic table 替代大而散的语义规则；
3. 用 `r_mask` 负责 mask trust，而不是把 `Q_mask` 直接加进 prior；
4. 用 `A_mask = \bar E \times semantic_modulation` 替代旧 logit；
5. 用 soft fusion 替代 covered-region hard switch。

### Phase 2. Change the Stage E interface

1. 增加 `B_chunk_geo` 输出；
2. 将 Stage E 的 prior 接入从

\[
\gamma_i^{v1}
=
\frac{\eta_i A_{tok,i}}
{\operatorname{Mean}(\eta A_{tok})+\epsilon}
\]

改为

\[
\gamma_i^{v2}
=
\frac{\eta_i A_{tok,i}}
{\operatorname{Mean}(\eta)+\epsilon}
\]

3. `\lambda_{write}` 改由 `B_chunk_geo` 构造。

### Phase 3. Add optional refinements only after baseline works

仅在 baseline 跑通并验证有效后，再考虑：

- semantic entropy trust
- temporal smoothing on `A_mask`
- class-distribution semantics
- per-special-token differentiation
- per-layer scheduling
- branch/head specialization

---

## 16. Debug and ablation checklist

v2 最重要的不是先看最终指标，而是先确认职责是否分离成功。

### 16.1 Must-have debug tensors

- `Elig_pix`
- `A_mask`
- `r_mask`
- `A_pix`
- `A_patch_flat`
- `B_chunk_geo`
- `A_tok`

### 16.2 Recommended ablations

#### Ablation A. No semantics
令：

\[
\rho_{sem}=0
\]

检查系统是否退化为 geometry-only write gating。

#### Ablation B. Perfect trust vs low trust
比较：

- `r_mask = 1`
- `r_mask = Q_mask`

检查 mask trust 是否真的只影响融合，而不是误伤 geometry branch。

#### Ablation C. Old vs new Stage E normalization
比较：

\[
\gamma^{v1}
\quad \text{vs} \quad
\gamma^{v2}
\]

检查 `A_tok` 的均值是否在新接口中真正保留下来。

#### Ablation D. Old lambda vs geometry-only lambda
比较：

- `\lambda` from `mean(A_patch)`
- `\lambda` from `B_chunk_geo`

检查是否减少了 semantic-global double suppression。

---

## 17. Final summary

v2 的本质不是“换一个公式”，而是把旧版 Semantic Prior Generator 从一个**混合打分器**重构为一套**职责分离的写入政策**：

- Geometry cues 只负责 **eligibility**
- Semantic labels 只负责 **value**
- Mask quality 只负责 **trust / routing**
- Chunk budget 主要由 **geometry-only summary** 决定
- Token prior 作为 **absolute write gate** 进入 Stage E，而不再被完全归一化为纯相对排序

因此，v2 相比 v1 的最大提升不是“更复杂”，而是：

1. **更容易解释**
2. **更容易调参**
3. **更少重复抑制**
4. **更符合长期几何 memory 的控制目标**

---

## Appendix A. Pseudocode

```python
# inputs:
# E_cue = {C_stat, C_dyn, C_occ, C_unc, C_anchor}
# M_mask, V_mask, Q_mask, semantic labels/distributions
# PatchMeta, TokenType

# 1) geometry eligibility
P_pos  = 0.5 * C_stat + 0.5 * C_anchor
P_risk = 0.5 * C_dyn  + 0.25 * C_occ + 0.25 * C_unc
Elig_pix = sigmoid(k_pos * P_pos - k_risk * P_risk + b_elig)

# 2) semantic value
v_sem = lookup_4_bucket_value(label_or_distribution)

# 3) mask trust
r_mask = V_mask * Q_mask * u_sem   # u_sem = 1 if no semantic entropy correction

# 4) masklet gate
E_bar = masked_mean(Elig_pix, M_mask)
A_mask = clip(E_bar * ((1 - rho_sem) + rho_sem * v_sem), 0, 1)

# 5) pixel fusion
j_star = argmax_j(M_mask[j] * r_mask[j] * A_mask[j])
A_sem_pix = A_mask[j_star] if covered else 0
R_mask_pix = r_mask[j_star] if covered else 0
A_pix = R_mask_pix * A_sem_pix + (1 - R_mask_pix) * Elig_pix

# 6) token projection
A_patch = patch_mean_pool(A_pix)
E_patch = patch_mean_pool(Elig_pix)
B_chunk_geo = mean(E_patch)
A_special = clip(a_special_min + (1 - a_special_min) * B_chunk_geo,
                 a_special_min, 1.0)
A_tok = concat(A_patch.flatten(), repeat(A_special, num_special_tokens))

# 7) Stage E (recommended)
gamma = eta * A_tok / (mean(eta) + eps)
lambda_write = lambda_min + (lambda_max - lambda_min) * B_chunk_geo
```

---

## Appendix B. Migration note from v1

如果你已经实现了 v1，可以按下面的映射迁移：

- old `A_pix^{cue}` -> `Elig_pix`
- old `A_mask` logit -> remove, replace with `E_bar * semantic_modulation`
- old `Q_mask` direct score term -> move to `r_mask`
- old `A_patch` -> keep, but add `E_patch`
- old `A_special` semantic/risk formulas -> replace by geometry-only baseline
- old `lambda_write(mean(A_patch))` -> replace with `lambda_write(B_chunk_geo)`
- old `gamma(beta / mean(beta))` -> replace with `gamma(eta * A_tok / mean(eta))`

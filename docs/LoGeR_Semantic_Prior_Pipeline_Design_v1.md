
# LoGeR + Dynamic Cue + Video Masklet + Semantic Prior Generator Pipeline 设计文档

> 本文档给出一套面向 **LoGeR TTT 写入控制** 的完整 pipeline。系统以 **chunk** 为运行单元，同时建立两条并行信息流：  
> 1. 从 LoGeR 提取 **几何动态线索**；  
> 2. 从视频前端提取 **带语义标签的 2D masklets**；  
> 最终将二者融合为 **token-wise write prior**，直接作用于 LoGeR 的 TTT update path，控制当前 chunk 中不同 token 对 fast weights 的写入贡献。

---

## 0. 问题定义

长视频场景中，LoGeR 的 TTT fast weights 负责跨 chunk 压缩历史几何信息。它的优势是记忆开销固定、可持续累积上下文；它的风险是：当前 chunk 中所有 token 默认都会以某种权重参与写入，因此动态区域、遮挡边界、低价值语义区域、几何不稳定区域，都可能持续污染长期几何记忆。

本文档的核心目标是定义一条完整而简洁的 pipeline，使系统在每个 chunk 上都能回答下面这个问题：

> 当前 chunk 中，哪些 token 更值得写入 TTT memory，哪些 token 应该被弱化，哪些 token 应该尽量不写入？

为此，系统引入三个紧密耦合的模块：

1. **Dynamic Cue Extractor**：从 LoGeR 的 pointmap / pose / confidence 中提取 chunk 内几何静态性、动态性和不确定性线索；
2. **Video Masklet Front-end**：在同一 chunk 内构建短时连续的 2D masklets，并为每条 masklet 赋予语义类别或区域属性；
3. **Semantic Prior Generator**：将几何 cues 与语义 masklets 融合，投影到 LoGeR patch tokens 上，生成 token-wise write prior，并进一步控制 TTT 的 update 过程。

这套设计的关键不是改变 LoGeR 的 apply path，而是控制它的 **update path**。也就是说，系统关注的是：

- 当前 chunk **向未来留下怎样的几何记忆**；
- 当前 chunk **哪些内容不应被压进 fast weights**；
- 当前 chunk **哪些区域应该在 memory write 中被保留为主导更新结构的 token**。

---

## 1. 总体流程

### 1.1 主流程图

```text
Input chunk X_m
    │
    ├── Stage A: LoGeR Geometry Backbone
    │       ├── pointmap / pose / confidence
    │       └── TTT apply using W_m
    │
    ├── Stage B: Dynamic Cue Extractor
    │       └── geometry cues:
    │           C_stat / C_dyn / C_occ / C_unc / C_anchor
    │
    ├── Stage C: Video Masklet Front-end
    │       └── 2D masklets + semantic labels + quality
    │
    ├── Stage D: Semantic Prior Generator
    │       ├── cue-only prior
    │       ├── masklet semantic prior
    │       ├── pixel-level write-allow map
    │       ├── patch-token prior
    │       └── special-token prior
    │
    └── Stage E: TTT Write Controller
            ├── token-wise contribution weighting
            ├── optional block-level write gain
            └── delayed write-back: W_m -> W_{m+1}
```

### 1.2 Pipeline 的核心数据流

设第 `m` 个 chunk 为：

$$
\mathcal{X}_m = \{I_t\}_{t=1}^{T}
$$

系统在当前 chunk 上同时产生三类中间结果：

1. **几何证据张量**
   $$
   E_{\text{cue}} \in \mathbb{R}^{T \times H \times W \times C_{\text{cue}}}
   $$

2. **语义 masklet 包**
   $$
   \mathcal{M}_m = \{M_j, V_j, L_j, Q_j\}_{j=1}^{J}
   $$

3. **token-wise write prior**
   $$
   A_{\text{tok}} \in [0,1]^{L_{\text{tok}}}
   $$

其中：

- `T`：chunk 内帧数
- `H, W`：图像分辨率
- `J`：当前 chunk 的 masklet 数
- `L_tok`：当前 chunk 在 LoGeR 某层对应的 token 数

最终，`A_tok` 不用于控制 LoGeR 的输出残差，而是直接参与 TTT 的 update 聚合链。

---

## 2. 记号、张量与运行单元约定

### 2.1 时间与窗口记号

本文默认系统以 chunk 为运行单元。设：

- `m`：chunk 索引
- `T`：当前 chunk 的帧数
- `t \in \{1,\dots,T\}`：chunk 内帧索引

当前 chunk 写作：

$$
\mathcal{W}_m = \{I_{m,1}, I_{m,2}, \dots, I_{m,T}\}
$$

### 2.2 分辨率与 token 网格

- 图像分辨率：`H × W`
- LoGeR pointmap 分辨率：`H_p × W_p`
- LoGeR patch token 网格：`H_{\text{tok}} × W_{\text{tok}}`

对应地：

$$
L_{\text{patch}} = T \cdot H_{\text{tok}} \cdot W_{\text{tok}}
$$

如果某层还有 special tokens，则总 token 数写为：

$$
L_{\text{tok}} = L_{\text{patch}} + L_{\text{special}}
$$

### 2.3 TTT 记号

对第 `l` 个 TTT 层、第 `h` 个 head、第 `r \in \{0,1,2\}` 个 fast-weight 分支，记：

- 当前 chunk 前的 fast weights：
  $$
  W_{m,l}^{(r,h)}
  $$

- 当前 token 的原始 update 贡献矩阵：
  $$
  J_{m,l,i}^{(r,h)}
  $$

- LoGeR 内部已有的 token-wise 系数：
  $$
  \eta_{m,l,i}^{(r,h)}
  $$

- 本文新增的 token prior：
  $$
  p_{m,l,i}^{(r,h)}
  $$

- 先验修正后的 token 系数：
  $$
  \beta_{m,l,i}^{(r,h)} = p_{m,l,i}^{(r,h)} \cdot \eta_{m,l,i}^{(r,h)}
  $$

---

## 3. Stage A：LoGeR Geometry Backbone

### 3.1 输入

当前 chunk 输入为：

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| RGB 序列 | `X_m` | `[T, H, W, 3]` | 当前 chunk 图像 |
| 相机内参 | `K_m` | `[T, 3, 3]` 或 `[3,3]` | 相机模型 |
| 旧 fast weights | `W_m` | 分层结构体 | 当前 chunk 开始前的 TTT memory |

### 3.2 LoGeR 输出

LoGeR 在读取 `W_m` 后，对当前 chunk 执行 geometry inference，输出：

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| pointmap | `P_cam` | `[T, H_p, W_p, 3]` | camera-space 点图 |
| pose | `T_w_c` | `[T, 4, 4]` | world-from-camera 位姿 |
| geometry confidence | `Conf_geo` | `[T, H_p, W_p]` | 点图可信度 |
| patch token meta | `PatchMeta` | `[L_patch, 3]` | 每个 patch token 对应 `(t, y_tok, x_tok)` |
| token type | `TokenType` | `[L_tok]` | patch / reg / role 等 |
| TTT write cache | `Cache_ttt` | 分层结构体 | update 所需中间量 |

### 3.3 世界坐标 lifting

对当前帧 `t` 中任意有效像素 `u=(x,y)`，其世界坐标点为：

$$
X_t(u) = T_{w \leftarrow c, t}\, P_{\text{cam},t}(u)
$$

后续所有几何动态性判断、静态一致性判断和局部锚点资格判断，都基于 `X_t(u)` 进行。

### 3.4 TTT 的读写语义

LoGeR 在当前 chunk 上遵循：

1. **先 apply**：用 `W_m` 读取 memory，参与当前 chunk 的 forward；
2. **后 update**：用当前 chunk 的 token 序列对 fast weights 形成新的写入。

因此，当前 pipeline 的控制目标不是当前 chunk 的 geometry prediction 输出，而是：

$$
W_m \rightarrow W_{m+1}
$$

这条 memory update 链。

---

## 4. Stage B：Dynamic Cue Extractor

### 4.1 目标

Dynamic Cue Extractor 的职责，是把 LoGeR 提供的几何信息转成一组可直接服务于 write control 的 cue map。它不是最终动态分割器，而是一个几何证据提取器。

### 4.2 输出通道设计

设当前输出为：

$$
E_{\text{cue}} \in \mathbb{R}^{T \times H \times W \times 5}
$$

5 个通道定义如下：

| 通道 | 名称 | 范围 | 含义 |
|---:|---|---|---|
| 0 | `C_stat` | `[0,1]` | 当前更像静态一致表面的程度 |
| 1 | `C_dyn` | `[0,1]` | 当前更像动态违背区域的程度 |
| 2 | `C_occ` | `[0,1]` | 当前更像遮挡/显露边界的程度 |
| 3 | `C_unc` | `[0,1]` | 当前几何不稳、支持不足或不确定的程度 |
| 4 | `C_anchor` | `[0,1]` | 当前适合作为长期几何写入锚点的程度 |

### 4.3 几何 cue 的计算主线

Dynamic Cue Extractor 的核心思路是：用 chunk 内多帧几何一致性，估计当前区域是否能被静态世界解释。

#### 4.3.1 静态一致性

对当前帧 `t` 的世界点 `X_t(u)`，投影到支持帧 `s`：

$$
\tilde u_{t \rightarrow s} = \pi\big(T_{c \leftarrow w, s} X_t(u)\big)
$$

若支持帧对应世界点为 `\tilde X_s(\tilde u)`，则点位残差可写为：

$$
r_{\text{pt}}(t,s,u) =
\frac{\|X_t(u)-\tilde X_s(\tilde u_{t \rightarrow s})\|_2}{\epsilon + \|X_t(u)\|_2}
$$

静态一致性可由多个支持帧的残差稳健聚合得到：

$$
C_{\text{stat}}(t,u) =
\operatorname{TrimMean}_{s \in \mathcal{N}(t)}
\Big[\exp(-r_{\text{pt}}(t,s,u)/\sigma_{\text{pt}})\Big]
$$

#### 4.3.2 动态违背性

一个简单而有效的动态性定义是：

$$
C_{\text{dyn}}(t,u) =
\operatorname{clip}\Big(
\alpha_1 (1-C_{\text{stat}}(t,u))
+ \alpha_2 C_{\text{hist}}(t,u)
- \alpha_3 C_{\text{occ}}(t,u),
0,1 \Big)
$$

这里 `C_hist` 表示当前观测与历史稳定背景缓存的冲突程度；在最小实现中若尚无背景缓存，可先省略该项，用 chunk 内一致性主导。

#### 4.3.3 遮挡/显露

对当前世界点投影到支持帧后的深度顺序，定义可见性冲突量：

$$
r_{\text{occ}}(t,s,u) =
\max\big(0, z_{\text{proj}}(t \rightarrow s,u)-\tilde z_s(\tilde u_{t \rightarrow s})-\tau_{\text{occ}}\big)
$$

再在多个支持帧上聚合：

$$
C_{\text{occ}}(t,u) =
\operatorname{Mean}_{s \in \mathcal{N}(t)}
\mathbf{1}[r_{\text{occ}}(t,s,u)>\tau_{\text{occ}}]
$$

#### 4.3.4 不确定性

不确定性由低 confidence、有效重投影不足、边界冲突等共同决定。可定义为：

$$
C_{\text{unc}}(t,u)=
1-\frac{\sum_{s\in\mathcal{N}(t)}w(t,s,u)\cdot \mathbf{1}[\text{valid}(t,s,u)]}{|\mathcal{N}(t)|+\epsilon}
$$

再与低 `Conf_geo` 区域叠加即可。

#### 4.3.5 锚点资格

写入控制最重要的正向通道是 `C_anchor`。一个实用定义是：

$$
C_{\text{anchor}}(t,u) =
C_{\text{stat}}(t,u)\cdot (1-C_{\text{dyn}}(t,u)) \cdot (1-C_{\text{unc}}(t,u))
$$

这意味着：静态一致、动态风险低、几何不确定性低的区域，才更适合作为 memory write 的正向来源。

### 4.4 输出接口

| 名称 | Shape | 含义 |
|---|---:|---|
| `E_cue` | `[T, H, W, 5]` | 五通道 cue |
| `G_write_geo` | `[T, H, W]` | 几何驱动的写入允许图，可由 `C_stat/C_anchor/C_dyn/C_unc` 导出 |
| `CueDebug` | 结构体 | 可视化残差、支持帧统计、有效投影比等 |

---

## 5. Stage C：Video Masklet Front-end

### 5.1 目标

Video Masklet Front-end 的目标，是在当前 chunk 内得到短时连续的 2D 对象/区域候选，并为每条 masklet 补充语义标签或区域属性。它输出的是当前 chunk 内的语义支撑，不承担长期 identity 判决。

### 5.2 前端主线

推荐采用：

```text
关键帧发现 -> 窗口内传播 -> trigger refresh -> chunk 边界交接
```

其中关键帧可以按固定步长加触发式调度，trigger 可以来自：

- 大面积未解释区域
- `C_dyn` / `C_occ` / `C_unc` 异常升高
- 传播置信度下降
- 边界 handoff 质量下降

### 5.3 输入

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| RGB 序列 | `X_m` | `[T, H, W, 3]` | 当前 chunk 图像 |
| 几何 cues（可选） | `E_cue` | `[T, H, W, 5]` | 用于 refresh 和 cue-guided discovery |
| overlap handoff（可选） | `H_{m-1\to m}` | 变长结构 | 上一个 chunk 的边界 masklet 状态 |

### 5.4 语义标签来源

设第 `j` 条 masklet 的语义标签为 `L_sem[j]`。它可以来自：

1. 关键帧上的图像语义分割器；
2. 开放词表 detector / segmentor；
3. 规则型区域属性分类器；
4. 已有语义模型对 mask 区域的 majority vote。

为了服务 TTT 写入控制，本文档不要求极细类别，只要求输出满足 write prior 设计需要的标签体系。实际工程中可以采用两种形式：

- **离散类别 id**
  $$
  L_{\text{sem}}[j] \in \{1,\dots,K\}
  $$

- **类别分布**
  $$
  \pi_j \in [0,1]^K,\quad \sum_c \pi_j(c)=1
  $$

若采用类别分布，后续可以更平滑地生成语义权重。

### 5.5 输出包

设当前 chunk 输出 `J` 条 masklets，则接口定义为：

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| masklets | `M_mask` | `[J, T, H, W]` | 每条 masklet 的 2D 支撑 |
| 可见性 | `V_mask` | `[J, T]` | 哪些帧可见 |
| bbox | `B_mask` | `[J, T, 4]` | 每帧 `xyxy` |
| 传播质量 | `Q_mask` | `[J, T]` | 前端传播或 refresh 的质量 |
| 语义标签 | `L_sem` | `[J]` 或 `[J, K]` | 离散标签或类别分布 |
| appearance feature | `F_mask` | `[J, T, C_a]` | mask 区域外观特征 |
| 2D 质心 | `C2d_mask` | `[J, T, 2]` | 图像平面质心 |

### 5.6 语义类别到写入价值的映射

为使本 pipeline 直接服务 TTT 写入，建议为每类语义指定一个基础权重：

$$
w_{\text{sem}}(j) =
\begin{cases}
w_{L_{\text{sem}}[j]}, & \text{离散标签} \\
\sum_{c=1}^K \pi_j(c)\, w_c, & \text{类别分布}
\end{cases}
$$

其中 `w_c` 表示语义类别 `c` 的基础写入价值。后文会给出默认表。

---

## 6. Stage D：Semantic Prior Generator

### 6.1 目标

Semantic Prior Generator 是整个 pipeline 的核心。它将 Stage B 的几何 cues 与 Stage C 的语义 masklets 融合，并将融合结果从像素空间映射到 LoGeR token 空间，得到可以直接作用于 TTT update 的 token prior。

### 6.2 输入

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| cue 张量 | `E_cue` | `[T, H, W, 5]` | `C_stat/C_dyn/C_occ/C_unc/C_anchor` |
| masklet 包 | `M_mask, V_mask, Q_mask, L_sem` | 见前文 | 语义对象候选 |
| patch meta | `PatchMeta` | `[L_patch, 3]` | patch token 对应位置 |
| token type | `TokenType` | `[L_tok]` | patch / special token |

### 6.3 三层 prior 表示

Semantic Prior Generator 内部维护三层表示：

1. **masklet-level prior**
   $$
   A_{\text{mask}} \in [0,1]^{J \times T}
   $$

2. **pixel-level prior**
   $$
   A_{\text{pix}} \in [0,1]^{T \times H \times W}
   $$

3. **token-level prior**
   $$
   A_{\text{tok}} \in [0,1]^{L_{\text{tok}}}
   $$

这里 `A` 统一表示 **write-allow**，值越大表示越允许当前区域主导写入。

---

## 7. Stage D-1：几何 cue 向 masklet 聚合

### 7.1 mask 内 cue 平均值

对第 `j` 条 masklet 在第 `t` 帧，定义各通道聚合统计：

$$
\bar C_k(j,t)=
\frac{\sum_{h,w} M_{\text{mask}}(j,t,h,w)\, C_k(t,h,w)}
{\sum_{h,w} M_{\text{mask}}(j,t,h,w)+\epsilon}
$$

其中 `k \in \{\text{stat}, \text{dyn}, \text{occ}, \text{unc}, \text{anchor}\}`。

### 7.2 masklet 级写入分数

定义第 `j` 条 masklet 在第 `t` 帧的写入允许分数：

$$
z_{j,t} =
b_0
+ b_1 \bar C_{\text{stat}}(j,t)
+ b_2 \bar C_{\text{anchor}}(j,t)
- b_3 \bar C_{\text{dyn}}(j,t)
- b_4 \bar C_{\text{occ}}(j,t)
- b_5 \bar C_{\text{unc}}(j,t)
+ b_6 Q_{\text{mask}}(j,t)
+ b_7 w_{\text{sem}}(j)
$$

然后经过 sigmoid：

$$
A_{\text{mask}}(j,t)=\sigma(z_{j,t})
$$

这个式子的含义非常直接：

- 静态一致性高、锚点资格高、传播质量高、语义写入价值高的 masklet，更应该允许写入；
- 动态性高、遮挡性高、不确定性高的 masklet，更应该抑制写入。

### 7.3 可选的规则型近似

如果第一版不想引入参数化 `b_i`，也可以用乘法型规则：

$$
A_{\text{mask}}(j,t)=
\operatorname{clip}\Big(
w_{\text{sem}}(j)\cdot
(\lambda_s \bar C_{\text{stat}} + \lambda_a \bar C_{\text{anchor}} + \lambda_q Q_{\text{mask}})
\cdot
(1-\lambda_d \bar C_{\text{dyn}})
\cdot
(1-\lambda_o \bar C_{\text{occ}})
\cdot
(1-\lambda_u \bar C_{\text{unc}}),
0,1\Big)
$$

这更适合作为无训练参数的原型实现。

---

## 8. Stage D-2：pixel-level write prior

### 8.1 cue-only prior

对于没有任何 masklet 覆盖的区域，也需要一个几何驱动的回退 prior。定义：

$$
z^{\text{cue}}(t,h,w)=
c_0
+ c_1 C_{\text{stat}}(t,h,w)
+ c_2 C_{\text{anchor}}(t,h,w)
- c_3 C_{\text{dyn}}(t,h,w)
- c_4 C_{\text{occ}}(t,h,w)
- c_5 C_{\text{unc}}(t,h,w)
$$

对应的 cue-only write allow 为：

$$
A_{\text{pix}}^{\text{cue}}(t,h,w)=\sigma(z^{\text{cue}}(t,h,w))
$$

### 8.2 mask-driven prior

由 masklets 生成的像素级 write allow 定义为：

$$
A_{\text{pix}}^{\text{mask}}(t,h,w)=
\max_j \Big(M_{\text{mask}}(j,t,h,w)\cdot A_{\text{mask}}(j,t)\Big)
$$

这样做的含义是：

- 若同一像素被多个 masklet 竞争覆盖，则保留更高写入价值的那个候选；
- 该像素是否允许写入，由覆盖它的语义对象解释来主导。

### 8.3 融合公式

设：

$$
m_{\text{cov}}(t,h,w)=\max_j M_{\text{mask}}(j,t,h,w)
$$

则最终像素级 write allow 为：

$$
A_{\text{pix}}(t,h,w)=
m_{\text{cov}}(t,h,w)\cdot A_{\text{pix}}^{\text{mask}}(t,h,w)
+
\big(1-m_{\text{cov}}(t,h,w)\big)\cdot A_{\text{pix}}^{\text{cue}}(t,h,w)
$$

这意味着：

- 被语义 masklet 覆盖的区域，优先采用语义+几何融合 prior；
- 未被 masklet 覆盖的区域，使用 cue-only prior 作为回退。

### 8.4 输出

| 名称 | Shape | 含义 |
|---|---:|---|
| `A_mask` | `[J, T]` | masklet 级 write allow |
| `A_pix` | `[T, H, W]` | 像素级 write allow |
| `S_pix` | `[T, H, W]` | 抑制图，`S_pix = 1 - A_pix` |

---

## 9. Stage D-3：pixel-to-token 映射

### 9.1 patch token prior

对每个 patch 区域 `\Pi(t,y_{\text{tok}},x_{\text{tok}})`，定义 patch 级 write allow：

$$
A_{\text{patch}}(t,y_{\text{tok}},x_{\text{tok}})
=
\operatorname{Mean}_{(h,w)\in\Pi(t,y_{\text{tok}},x_{\text{tok}})}
A_{\text{pix}}(t,h,w)
$$

flatten 后得到：

$$
A_{\text{patch-flat}} \in [0,1]^{L_{\text{patch}}}
$$

第一版默认采用 **mean pooling**。若希望更保守地抑制 patch 中的高风险局部区域，也可改用：

$$
S_{\text{patch}} = \operatorname{p90}(1-A_{\text{pix}})
,\qquad
A_{\text{patch}} = 1-S_{\text{patch}}
$$

### 9.2 special token prior

register / role / global tokens 不对应单独像素区域，因此采用 chunk-level 统计生成先验。定义当前 chunk 的 patch 抑制占比：

$$
\rho_{\text{suppr}}^{\text{chunk}}
=
\frac{1}{L_{\text{patch}}}
\sum_{i=1}^{L_{\text{patch}}}
\big(1-A_{\text{patch-flat}}[i]\big)
$$

则可定义 special tokens 的 write allow 为：

$$
A_{\text{special}} =
\operatorname{clip}\big(
1-\kappa_{\text{special}}\rho_{\text{suppr}}^{\text{chunk}},
a_{\min}^{\text{special}},1
\big)
$$

如果 special tokens 还细分为 `reg / role-prev / role-mid / role-next`，则可分别设置：

$$
A_{\text{reg}},\quad A_{\text{role-prev}},\quad A_{\text{role-mid}},\quad A_{\text{role-next}}
$$

### 9.3 层级调度

为避免所有 TTT 层一刀切地共享同一 prior，可引入 layer schedule：

$$
A_{m,l,i} =
\operatorname{clip}\big(\omega_l \cdot A_i + (1-\omega_l), a_{\min}, 1\big)
$$

其中：

- `A_i` 为基础 token prior
- `\omega_l \in [0,1]` 控制第 `l` 层对外部先验的敏感度

若第一版希望简单，直接令所有层 `\omega_l=1` 即可。

### 9.4 输出接口

| 名称 | Shape | 含义 |
|---|---:|---|
| `A_patch_flat` | `[L_patch]` | patch token write allow |
| `A_special` | `[L_special]` 或按类别分组 | special token write allow |
| `A_tok` | `[L_tok]` | 全 token 写入先验 |

---

## 10. Stage E：TTT Write Controller

### 10.1 TTT 的函数形式

LoGeR 中每个 TTT 层可写成一个小的 SwiGLU fast-weight module：

$$
f_W(x)=\big(\mathrm{SiLU}(xW^{(0)}) \odot (xW^{(2)})\big)W^{(1)}
$$

其中：

- `W^{(0)}`：gate 分支
- `W^{(1)}`：输出投影分支
- `W^{(2)}`：content 分支

当输入是 query `q` 时：

$$
o_{\text{apply}} = f_W(q)
$$

它对应 TTT 的 read / apply 路径。

当输入是 key `k` 时：

$$
y_{\text{pred}} = f_W(k)
$$

它对应 TTT update 时对 value 的预测。

### 10.2 单 token 的原始 update 贡献

定义第 `i` 个 token 在第 `l` 层、第 `h` 个 head、分支 `r` 上的原始 update 贡献矩阵为：

$$
J_{m,l,i}^{(r,h)}
$$

这些 `J` 在当前 chunk 内先按 token 聚合，再经过 zeropower 和 renorm，形成真正的 fast-weight update 方向。

### 10.3 token prior 接入方式

设 `A_{m,l,i}` 为当前 chunk 在第 `l` 层对第 `i` 个 token 的 write allow，则默认令该 prior 在三个分支上共享：

$$
p_{m,l,i}^{(0,h)}=
p_{m,l,i}^{(1,h)}=
p_{m,l,i}^{(2,h)}=
A_{m,l,i}
$$

然后与当前 LoGeR 内部已有的 token-wise 系数相乘：

$$
\beta_{m,l,i}^{(r,h)} =
p_{m,l,i}^{(r,h)} \cdot \eta_{m,l,i}^{(r,h)}
$$

### 10.4 结构归一化

为避免整体尺度失控，在 chunk 内对 `\beta` 做归一化：

$$
\gamma_{m,l,i}^{(r,h)}
=
\frac{\beta_{m,l,i}^{(r,h)}}
{\frac{1}{|C_m|}\sum_j \beta_{m,l,j}^{(r,h)}+\varepsilon}
$$

这里：

- `\gamma` 决定当前 chunk 中谁更主导更新结构；
- `\gamma` 不直接表示写入幅度，而表示 **相对结构权重**。

### 10.5 pre-zeropower 聚合

于是，当前 chunk 在 layer `l`、head `h`、分支 `r` 上的 pre-zeropower 聚合矩阵为：

$$
\tilde G_{m,l}^{(r,h)}
=
\sum_{i\in C_m}
\gamma_{m,l,i}^{(r,h)}\,
J_{m,l,i}^{(r,h)}
$$

随后通过 LoGeR 当前的 operator 得到 applied direction：

$$
G_{m,l,\mathrm{app}}^{(r,h)}
=
\mathcal U_{\mathrm{dir}}\big(\tilde G_{m,l}^{(r,h)}\big)
$$

### 10.6 block-level write gain

除了 token-level prior，还可以对整块 applied direction 再加一个 chunk 级别的写入增益。设：

$$
\bar A_{\text{patch}}=
\frac{1}{L_{\text{patch}}}\sum_{i=1}^{L_{\text{patch}}}A_{\text{patch-flat}}[i]
$$

则可令：

$$
\lambda_{m,l}^{(r,h)}
=
\lambda_{\min}
+
(\lambda_{\max}-\lambda_{\min})\cdot \bar A_{\text{patch}}
$$

如果某个 chunk 大面积由动态前景、低价值语义区域或不确定区域主导，则 `\bar A_{\text{patch}}` 变小，整块写入就更保守。

### 10.7 最终写回

最终 fast weights 更新为：

$$
W_{m+1,l}^{(r,h)}
=
\mathrm{Renorm}\Big(
W_{m,l}^{(r,h)}
+
\lambda_{m,l}^{(r,h)}\,
G_{m,l,\mathrm{app}}^{(r,h)}
\Big)
$$

整个 update 链可以写成：

$$
A_{\text{tok}}
\rightarrow
p
\rightarrow
\beta
\rightarrow
\gamma
\rightarrow
\tilde G
\rightarrow
G_{\mathrm{app}}
\rightarrow
W_{m+1}
$$

这条链正是本 pipeline 的数学核心。

---

## 11. 运行时序

### 11.1 单 chunk 协议

推荐采用如下运行协议：

1. 读取上一 chunk 更新后的 fast weights `W_m`；
2. 用 `W_m` 运行当前 chunk 的 LoGeR forward，得到几何输出与 TTT write cache；
3. 在几何输出上运行 Dynamic Cue Extractor，得到 `E_cue`；
4. 在当前 chunk 图像上运行 Video Masklet Front-end，得到 `M_mask, L_sem, Q_mask`；
5. 运行 Semantic Prior Generator，得到 `A_pix, A_patch_flat, A_tok`；
6. 将 `A_tok` 接入 TTT update path；
7. 执行 delayed write-back，得到 `W_{m+1}`；
8. 下一个 chunk 读取 `W_{m+1}`。

### 11.2 delayed write-back 的原因

该设计把 prior 的计算放在 LoGeR 几何推理之后、fast-weight 写回之前，原因是：

- 当前 chunk 的几何 cues 需要先由 LoGeR 产生；
- 当前 chunk 的 masklets 和语义标签也需要先由前端产生；
- token prior 只有在这两条支路都完成后才能构建；
- 因此最自然的方式是：先推理，后写回。

---

## 12. 接口总表

### 12.1 Stage A/B/C/D/E 总接口

| Stage | 输入 | 输出 | 关键 Shape |
|---|---|---|---|
| LoGeR | `X_m, K_m, W_m` | `P_cam, T_w_c, Conf_geo, PatchMeta, TokenType, Cache_ttt` | 见前文 |
| Dynamic Cue Extractor | `P_cam, T_w_c, Conf_geo` | `E_cue, G_write_geo` | `E_cue: [T,H,W,5]` |
| Video Masklet Front-end | `X_m, E_cue` | `M_mask, V_mask, L_sem, Q_mask, F_mask` | `M_mask: [J,T,H,W]` |
| Semantic Prior Generator | `E_cue, M_mask, L_sem, Q_mask, PatchMeta, TokenType` | `A_mask, A_pix, A_patch_flat, A_tok` | `A_tok: [L_tok]` |
| TTT Write Controller | `A_tok, Cache_ttt, \eta` | `W_{m+1}` | 分层 fast-weight state |

### 12.2 建议保留的 debug 输出

| 名称 | Shape | 作用 |
|---|---:|---|
| `A_mask` | `[J,T]` | 看每条 masklet 的写入价值 |
| `A_pix` | `[T,H,W]` | 看像素级 write allow |
| `S_pix` | `[T,H,W]` | 看抑制图 |
| `A_patch_flat` | `[L_patch]` | 看 patch token prior |
| `rho_suppr_chunk` | `[]` | 看 chunk 级抑制占比 |
| `lambda_write` | `[L_ttt,H_ttt,3]` 或简化版 | 看 block-level 写入增益 |

---

## 13. 默认语义权重表

为了让系统第一版可直接运行，建议给语义类别分配如下默认权重。该表不是理论常数，而是 write-control 导向的初始工程先验。

| 语义类别 / 区域属性 | 基础权重 `w_c` | 含义 |
|---|---:|---|
| 建筑、墙面、地面、天花板、固定结构 | 1.00 | 高写入价值 |
| 大型静态家具、门框、柜体、稳定室内结构 | 0.80 | 较高写入价值 |
| 普通背景物、弱纹理静态区域 | 0.65 | 中等写入价值 |
| 未知但几何稳定区域 | 0.55 | 保守保留 |
| 植被、树木、重复枝叶纹理 | 0.40 | 降权 |
| 天空、水面、反光强区域 | 0.20 | 强降权 |
| 人、车、动物、手持物、可移动物体 | 0.10 | 极低写入价值 |
| 低质量 masklet / 语义不确定区域 | 0.15 | 抑制写入 |

如果采用类别分布 `\pi_j(c)`，则：

$$
w_{\text{sem}}(j)=\sum_c \pi_j(c)\, w_c
$$

---

## 14. 默认超参数建议

### 14.1 chunk 与前端相关

| 参数 | 建议值 | 说明 |
|---|---:|---|
| `T` | 与 LoGeR chunk 一致，如 32 / 48 / 64 | chunk 长度 |
| `T_o` | 3～6 | overlap handoff 长度 |
| `s_k` | 4～8 | 前端关键帧间隔 |
| `J_max` | 32～128 | 单 chunk masklet 数上限 |

### 14.2 prior 融合相关

| 参数 | 建议值 | 说明 |
|---|---:|---|
| `b_1, b_2` | 正值 | 静态/锚点正向加权 |
| `b_3, b_4, b_5` | 正值 | 动态/遮挡/不确定抑制 |
| `b_6` | 正值 | masklet 质量加权 |
| `b_7` | 正值 | 语义价值加权 |
| `a_{\min}^{special}` | 0.2～0.4 | special tokens 最小写入允许 |
| `\kappa_{special}` | 0.5～1.0 | special token 对 chunk 抑制占比的敏感度 |

### 14.3 TTT 写入相关

| 参数 | 建议值 | 说明 |
|---|---:|---|
| `\lambda_{\min}` | 0.0～0.2 | 最低 block-level 写入增益 |
| `\lambda_{\max}` | 1.0 | 最高 block-level 写入增益 |
| `\omega_l` | 1.0 或随层递增 | layer prior 强度 |
| `\varepsilon` | `1e-6` | 归一化稳定项 |

---

## 15. 实现建议

### 15.1 第一阶段：规则型 prior baseline

第一版优先实现：

1. 基于 LoGeR 的 `E_cue`
2. 基于 masklet 语义标签的 `w_sem`
3. 规则型 `A_mask`
4. `A_pix -> A_patch -> A_tok`
5. `A_tok` 接入 `\eta`，形成 `\beta`

这一版不需要额外训练 Semantic Prior Generator，就可以直接验证：

- 动态区域是否更少污染 fast weights
- 语义低价值区域是否更少主导 update
- 长视频漂移是否减轻

### 15.2 第二阶段：轻量可学习 prior

在规则型 baseline 跑通后，可以把 `z_{j,t}` 和 `z^{cue}(t,h,w)` 改为小 MLP 或小卷积网络：

$$
A_{\text{mask}}(j,t)=g_{\theta}^{\text{mask}}(\bar C(j,t), Q_{\text{mask}}(j,t), \pi_j)
$$

$$
A_{\text{pix}}^{\text{cue}}(t,h,w)=g_{\phi}^{\text{pix}}(E_{\text{cue}}(t,h,w,:))
$$

这样可让系统学习更细粒度的 prior 融合策略。

### 15.3 第三阶段：branch/head/layer 特化

若后续需要更细控制，再考虑：

- 分支特定 prior：
  $$
  p^{(0)} \neq p^{(1)} \neq p^{(2)}
  $$
- head 特定 prior
- layer 深度调度
- 特定 special token policy

---

## 16. 建议的评估输出

虽然本文档重点是 pipeline，但为了让系统闭环可验证，建议同时记录以下指标：

1. **geometry stability**
   - 长视频累计漂移
   - 重投影误差统计
   - 背景区域几何一致性

2. **memory contamination**
   - 动态区域是否在后续 chunk 中留下 ghost-like geometry
   - 被抑制区域的 token 对 update 的贡献占比

3. **token prior behavior**
   - 各语义类别 token 的平均 `A_tok`
   - `C_dyn` 高区域与 `A_tok` 的相关性
   - `C_anchor` 高区域与 `A_tok` 的相关性

4. **TTT 更新统计**
   - `\tilde G` 的 Frobenius norm
   - `\lambda_write` 的分布
   - chunk 级 `rho_suppr_chunk`

---

## 17. 最终总结

这条 pipeline 的核心思想可以浓缩成下面一句话：

> 在每个 chunk 上，LoGeR 提供当前局部几何骨架，Dynamic Cue Extractor 从中提取静态/动态/遮挡/不确定/锚点线索；Video Masklet Front-end 生成带语义标签的 2D masklets；Semantic Prior Generator 将几何 cues 与语义 masklets 融合为像素级、patch 级和 token 级的 write prior；最终，这个 prior 直接作用于 TTT 的 update 聚合链，控制当前 chunk 中不同 token 对 fast weights 的写入贡献，从而使长期几何记忆更多由静态、稳定、语义上高价值的区域主导，而更少被动态、遮挡、不确定和低价值区域污染。

从实现角度看，这条主线最重要的优点有三点：

1. 它直接作用于 **TTT update path**，而不是停留在输出门控层；
2. 它把 **几何 cues** 和 **语义 masklets** 统一到了同一个 token prior 接口中；
3. 它保持 LoGeR 现有的 **apply / zeropower / renorm** 主结构不变，只在 token contribution weighting 与 block-level write gain 两个位置施加控制。

因此，这套设计既保持了结构清晰和实现可落地，又能直接回答本项目当前最核心的问题：  
**如何让 LoGeR 的长期几何记忆写得更干净。**

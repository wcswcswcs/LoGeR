# Dynamic Cue Extractor 设计文档（v2）

> 本文档定义 **Dynamic Cue Extractor** 模块。模块以 **LoGeR 的 chunk 级几何输出** 为输入，输出一组直接服务于 **Semantic Prior Generator** 和 **TTT Write Controller** 的几何 cue。本文档聚焦：如何从 LoGeR 的 pointmap / pose / confidence，以及可选的 decoder attention priors 中构造 `C_stat / C_dyn / C_occ / C_unc / C_anchor`，并进一步生成几何驱动的 write-allow map。

---

## 0. 一句话定位

**Dynamic Cue Extractor 的职责，是把 LoGeR 当前 chunk 的几何结果转化为一层面向 memory write 的几何证据场：哪些区域当前更像稳定静态表面，哪些区域更像动态违背、遮挡边界或几何不确定区域，哪些区域更适合成为长期几何写入锚点。**

---

## 1. 模块目标

在当前 pipeline 中，Dynamic Cue Extractor 需要回答下面五个问题：

1. 当前哪些区域更像 **静态一致表面**；
2. 当前哪些区域更像 **动态违背区域**；
3. 当前哪些区域更像 **遮挡 / 显露边界**；
4. 当前哪些区域更像 **几何不稳定或证据不足区域**；
5. 当前哪些区域更适合作为 **TTT memory write 的锚点来源**。

这里的输出不是最终的语义标签，也不是永久对象类别，而是一个 **几何证据层**。它直接服务于：

- Stage C 的 masklet 发现与 refresh trigger；
- Stage D 的语义先验融合；
- Stage E 的 token-wise 写入控制。

---

## 2. 设计原则

### 2.1 证据优先，避免把几何 cue 当成最终真相

模块输出的是当前 chunk 内的几何证据，而不是“这个区域永远是静态/动态”。

### 2.2 以当前 chunk 为主，但允许利用 overlap 和相邻 chunk 边界信息

最核心的静态/动态判断来自当前 chunk 内多帧几何一致性；若有 overlap、上一 chunk 边界缓存，或 Stage A 导出的 attention priors，则可额外利用这些时序先验提升稳健性。

### 2.3 attention prior 只作为先验，不直接替代几何证据

来自 decoder 的 `frame_attention_prior / attn_dynamic_patch` 更适合做：

- support 帧排序
- pairwise consistency 的权重调节
- `C_dyn` 的额外动态证据

而不应直接替代 pointmap / pose / confidence 本身。

### 2.4 明确区分动态违背、遮挡边界和几何不确定性

`C_dyn`、`C_occ` 和 `C_unc` 必须分开：

- `C_dyn` 表示静态世界假设在这里更解释不通；
- `C_occ` 表示更像 visibility change；
- `C_unc` 表示当前数据本身不适合做强判决。

### 2.5 `C_anchor` 不是 `C_stat` 的简单拷贝

一个区域当前静态一致，不等于它就适合写进长期记忆。`C_anchor` 必须额外考虑动态风险和不确定性。

### 2.6 面向写入控制的输出，应尽量稳定、连续、可 patch 化

因为下游最终需要的是 patch token prior，所以几何 cue 在空间上应平滑，在时间上应稳定，并且容易池化到 token 网格。

---

## 3. 在整条 Pipeline 中的位置

```text
Input chunk
    ↓
LoGeR Geometry Backbone
    ↓
Dynamic Cue Extractor
    ↓
C_stat / C_dyn / C_occ / C_unc / C_anchor
    ↓
Semantic Prior Generator
    ↓
TTT Write Controller
```

它与其他模块的接口关系是：

- 上游：LoGeR Geometry Backbone
- 同级协作：Video Masklet Front-end
- 下游：Semantic Prior Generator

---

## 4. 输入定义

下面默认讨论单条视频流、单个 chunk 的情况。若做 batch 版本，只需在最前面加 batch 维 `B`。

### 4.1 记号

- `T`：当前 chunk 的帧数
- `H, W`：图像分辨率
- `H_p, W_p`：LoGeR pointmap 分辨率
- `T_o`：chunk overlap 帧数（若可用）
- `K_s`：每像素用于支持集比较的支持帧数上限

### 4.2 主输入表

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| RGB 序列 | `X_m` | `[T, H, W, 3]` | 当前 chunk 图像 |
| camera-space pointmap | `P_cam` | `[T, H_p, W_p, 3]` | LoGeR 点图 |
| world-from-camera pose | `T_w_c` | `[T, 4, 4]` | 当前帧位姿 |
| geometry confidence | `Conf_geo` | `[T, H_p, W_p]` 或 `[T, H, W]` | 点图可信度 |
| frame attention prior（可选） | `A_frame` | `[T, T]` | decoder attention 导出的 frame-level 亲和度 |
| patch dynamic prior（可选） | `M_attn_patch` | `[T, H_tok, W_tok]` | decoder attention 导出的 patch-level dynamicness |
| patch token meta | `PatchMeta` | `[L_patch, 3]` | 每个 patch token 对应 `(t, y_tok, x_tok)` |
| 上一 chunk 边界缓存（可选） | `PrevOverlap` | 结构体 | 用于边界连续性检查 |
| 重投影残差图（可选） | `R_geo` | `[T, H, W]` | LoGeR 内部辅助残差 |

### 4.3 `PrevOverlap` 推荐字段

```text
PrevOverlap = {
  P_cam_prev: [T_o, H_p, W_p, 3],
  T_w_c_prev: [T_o, 4, 4],
  Conf_prev: [T_o, H_p, W_p],
}
```

若当前系统尚未启用 overlap 缓存，则可以先不使用跨 chunk 支持集，第一版只依赖 chunk 内一致性。

---

## 5. 输出定义

### 5.1 主输出：五通道 cue 张量

定义：

$$
E_{\text{cue}} \in \mathbb{R}^{T \times H \times W \times 5}
$$

对应通道为：

| 通道 | 名称 | 范围 | 含义 |
|---:|---|---|---|
| 0 | `C_stat` | `[0,1]` | 当前更像静态一致表面的程度 |
| 1 | `C_dyn` | `[0,1]` | 当前更像动态违背区域的程度 |
| 2 | `C_occ` | `[0,1]` | 当前更像遮挡/显露边界的程度 |
| 3 | `C_unc` | `[0,1]` | 当前更像几何不稳或证据不足区域 |
| 4 | `C_anchor` | `[0,1]` | 当前更适合成为长期 memory write 锚点的程度 |

### 5.2 几何写入允许图

定义：

$$
G_{\text{write-geo}} \in [0,1]^{T \times H \times W}
$$

它是一个由几何 cue 直接导出的 write-allow 图，用于：

- 当某像素没有任何语义 masklet 覆盖时，作为 Stage D 的回退 prior；
- debug 当前纯几何条件下的写入控制趋势。

### 5.3 可选输出：patch 级 cue

为方便下游做 token 对齐，可附带输出 patch pooling 后的版本：

$$
E_{\text{cue-patch}} \in \mathbb{R}^{T \times H_{tok} \times W_{tok} \times 5}
$$

第一版不是必须，但很适合做可视化和 ablation。

### 5.4 Debug 输出

推荐保留：

```text
CueDebug = {
  valid_support_ratio,
  reprojection_error_map,
  occlusion_conflict_map,
  pairwise_consistency_map,
  anchor_histogram,
}
```

---

## 6. 模块内部结构

推荐将模块拆成四个子单元：

1. **Geometry Alignment Unit**：分辨率对齐、世界坐标 lifting、局部法向估计；
2. **Support Set Builder**：构造 chunk 内与边界支持集；
3. **Pairwise Evidence Extractor**：计算点位残差、法向残差、遮挡残差与投影有效性；
4. **Cue Calibrator**：将多种残差转成五通道 cue，并生成几何写入图。

数据流如下：

```text
P_cam / pose / confidence / attention priors
      ↓
world lifting + normal estimation
      ↓
attention-aware support set construction
      ↓
pairwise reprojection & residuals
      ↓
attention-aware consistency weighting
      ↓
C_stat / C_dyn / C_occ / C_unc / C_anchor
      ↓
G_write_geo
```

---

## 7. 关键几何步骤

### 7.1 世界坐标 lifting

对第 `t` 帧像素 `u=(x,y)`，LoGeR 给出 camera-space 点：

$$
P_t(u) \in \mathbb{R}^3
$$

用位姿变到世界坐标：

$$
X_t(u) = T_{w\leftarrow c,t} \cdot P_t(u)
$$

后续所有静态一致性、动态违背性和锚点资格都基于 `X_t(u)`。

### 7.2 局部法向估计（推荐）

若 pointmap 分辨率和质量允许，建议在世界坐标点图上估计局部法向：

$$
N_t(u) \in \mathbb{R}^3
$$

它主要用于：

- 区分纯几何抖动与真实表面不一致；
- 提高静态一致性的判别力；
- 降低薄结构和边缘区域误报。

### 7.3 支持集构造

对当前帧 `t`，定义支持集：

#### A. chunk 内支持集

$$
\mathcal{N}_{\text{intra}}(t)
$$

例如取前后若干帧，或仅取时间上邻近的 `k_intra` 帧。

若 `A_frame / M_attn_patch` 可用，推荐把 support ranking 改成：

$$
S(t,s) = \lambda_t \cdot S_{time}(t,s) + \lambda_a \cdot A_{frame}(t,s) + \lambda_m \cdot O_{static}(t,s)
$$

其中：

- `S_time(t,s)`：时间距离衰减项
- `A_frame(t,s)`：frame attention prior
- `O_static(t,s)`：由 `1 - M_attn_patch` 计算出的静态区域 overlap

然后按 `S(t,s)` 选前 `k_intra` 个支持帧。

#### B. 边界支持集（可选）

若有 overlap，可定义：

$$
\mathcal{N}_{\text{bdry}}(t)
$$

它从上一 chunk 的尾部帧中取支持，用于检测边界连续性破坏。

#### C. 合并支持集

$$
\mathcal{N}(t)=\mathcal{N}_{\text{intra}}(t) \cup \mathcal{N}_{\text{bdry}}(t)
$$

---

## 8. Pairwise 几何残差

### 8.1 重投影

将当前世界点 `X_t(u)` 投影到支持帧 `s`：

$$
\tilde u_{t\to s} = \pi\big(T_{c\leftarrow w,s} X_t(u)\big)
$$

如果投影落在图像内且深度有效，则可在支持帧采样对应点与法向。

### 8.2 点位残差

$$
r_{\text{pt}}(t,s,u)=
\frac{\lVert X_t(u)-\tilde X_s(\tilde u_{t\to s})\rVert_2}{\epsilon + \lVert X_t(u)\rVert_2}
$$

### 8.3 法向残差（推荐）

$$
r_{\text{n}}(t,s,u)=1-\langle N_t(u), \tilde N_s(\tilde u_{t\to s}) \rangle
$$

### 8.4 遮挡 / 显露残差

$$
r_{\text{occ}}(t,s,u)=
\max\big(0, z_{\text{proj}}(t\to s,u)-\tilde z_s(\tilde u_{t\to s})-\tau_{\text{occ}}\big)
$$

### 8.5 投影权重

$$
w(t,s,u)=Conf_t(u) \cdot \tilde{Conf}_s(\tilde u_{t\to s})
$$

若系统里有视角夹角、边界 mask、valid flag 等，也可将它们乘进 `w`。

---

## 9. 从残差到五通道 cue

### 9.1 `C_stat`：静态一致性

先定义 pairwise consistency：

$$
c(t,s,u)=
 w(t,s,u)
 \cdot \exp\big(-r_{\text{pt}}/\sigma_{\text{pt}}\big)
 \cdot \exp\big(-r_{\text{n}}/\sigma_{\text{n}}\big)
$$

再在支持集上做稳健聚合：

$$
C_{\text{stat}}(t,u)=
\operatorname{TrimMean}_{s\in\mathcal{N}(t)} c(t,s,u)
$$

含义：若多帧共同支持这是同一静态表面，则 `C_stat` 高。

### 9.2 `C_occ`：遮挡 / 显露

$$
C_{\text{occ}}(t,u)=
\operatorname{Mean}_{s\in\mathcal{N}(t)}
\mathbf{1}[r_{\text{occ}}(t,s,u)>\tau_{\text{occ}}] \cdot w(t,s,u)
$$

含义：这里更像 visibility change，而不是物体本体运动。

### 9.3 `C_unc`：不确定性

可用投影有效性与支持度定义：

$$
C_{\text{unc}}(t,u)=
1-
\frac{\sum_{s\in\mathcal{N}(t)} w(t,s,u)\,\mathbf{1}[valid(t,s,u)]}{|\mathcal{N}(t)|+\epsilon}
$$

再与低 `Conf_geo` 区域叠加。

### 9.4 `C_dyn`：动态违背性

定义一个简单但稳定的版本：

$$
C_{\text{dyn}}(t,u)=
\operatorname{clip}\Big(
\alpha_1(1-C_{\text{stat}}(t,u))
+ \alpha_2 C_{\text{bdry-break}}(t,u)
- \alpha_3 C_{\text{occ}}(t,u),
0,1\Big)
$$

其中 `C_bdry-break` 是可选的边界连续性破坏项；若当前系统未使用上一 chunk 缓存，可暂时令该项为 0。

### 9.5 `C_anchor`：几何锚点资格

建议定义：

$$
C_{\text{anchor}}(t,u)=
C_{\text{stat}}(t,u)
\cdot (1-C_{\text{dyn}}(t,u))
\cdot (1-C_{\text{unc}}(t,u))
$$

这表示：静态一致、动态风险低、不确定性低的区域，更适合作为写入锚点。

---

## 10. 几何写入图 `G_write_geo`

本模块最终不仅要输出 cue，还要输出一个纯几何驱动的 write-allow 图。

定义：

$$
z_{\text{geo}}(t,u)=
\lambda_s C_{\text{stat}}(t,u)
+ \lambda_a C_{\text{anchor}}(t,u)
- \lambda_d C_{\text{dyn}}(t,u)
- \lambda_o C_{\text{occ}}(t,u)
- \lambda_u C_{\text{unc}}(t,u)
$$

然后：

$$
G_{\text{write-geo}}(t,u)=\sigma(z_{\text{geo}}(t,u))
$$

当语义 masklet 没有覆盖某个区域时，下游直接使用 `G_write_geo` 作为回退 prior。

---

## 11. Patch 对齐与可选 patch 输出

若希望本模块直接支持 token 对齐，可对每个 patch 区域 `\Pi(t,y_{tok},x_{tok})` 进行 pooling：

$$
E_{\text{cue-patch}}(t,y_{tok},x_{tok},k)=
\operatorname{Mean}_{(h,w)\in\Pi} E_{\text{cue}}(t,h,w,k)
$$

同理：

$$
G_{\text{write-geo-patch}}(t,y_{tok},x_{tok})=
\operatorname{Mean}_{(h,w)\in\Pi} G_{\text{write-geo}}(t,h,w)
$$

这部分不是 Semantic Prior Generator 的必须输入，但非常适合：

- 可视化每层 token 上的几何风险；
- 做 patch-level ablation；
- 提前发现几何 cue 是否与 token grid 错位。

---

## 12. 与 Video Masklet Front-end 的接口

Dynamic Cue Extractor 对视频前端提供三类直接帮助：

### 12.1 discovery supplement

当某些区域 `C_dyn` 或 `C_occ` 高，而当前前端没有 mask 覆盖时，可将这些区域转成：

- 点 prompt
- box prompt
- refresh trigger

### 12.2 propagation health check

若某条 masklet 长期落在高 `C_unc` 或高 `C_occ` 区域上，则可提示前端：

- 当前传播边界可能漂移；
- 需要在下一关键帧 refresh；
- 需要重新估计该 masklet 语义或区域边界。

### 12.3 质量评分辅助

前端可对每条 masklet 聚合 cue：

$$
\bar C_k(j,t)=
\frac{\sum M_j(t,h,w) C_k(t,h,w)}{\sum M_j(t,h,w)+\epsilon}
$$

作为 masklet 质量的一部分。

---

## 13. 与 Semantic Prior Generator 的接口

本模块是 Stage D 的几何输入来源。建议接口为：

| 输出 | Shape | 用途 |
|---|---:|---|
| `E_cue` | `[T,H,W,5]` | 与语义 masklet 融合 |
| `G_write_geo` | `[T,H,W]` | 作为无 mask 区域的回退 prior |
| `E_cue_patch`（可选） | `[T,H_tok,W_tok,5]` | 直接对 patch token 做 debug / pooling |

在 Stage D 中：

- `C_stat` 和 `C_anchor` 提供正向写入证据；
- `C_dyn`、`C_occ`、`C_unc` 提供抑制写入证据；
- `G_write_geo` 为未被任何 masklet 覆盖的背景区域提供写入允许度。

---

## 14. 实现优先级

### Phase 1：最小可运行版本

先实现：

1. world lifting；
2. chunk 内 pairwise point residual；
3. `C_stat / C_dyn / C_unc / C_anchor`；
4. `G_write_geo`。

此时可暂不实现：

- 法向残差；
- 边界支持集；
- `C_occ` 的复杂版本。

### Phase 2：加入遮挡与边界连续性

再补：

1. `C_occ`；
2. overlap 边界支持集；
3. 边界连续性破坏项。

### Phase 3：增强稳健性

进一步加入：

1. 法向残差；
2. 几何 smoothing；
3. patch-level debug 输出；
4. 与 masklet 前端的触发式接口。

---

## 15. 推荐默认超参数

| 参数 | 建议值 | 含义 |
|---|---:|---|
| `k_intra` | 2～4 | 每帧 chunk 内支持帧数 |
| `k_bdry` | 1～3 | 边界支持帧数 |
| `\sigma_pt` | 0.02～0.05 | 点位残差尺度 |
| `\sigma_n` | 0.1～0.2 | 法向残差尺度 |
| `\tau_occ` | 0.02～0.05 | 遮挡深度阈值 |
| `\alpha_1,\alpha_2,\alpha_3` | 手工设定 | `C_dyn` 组合权重 |
| `\lambda_s,\lambda_a,\lambda_d,\lambda_o,\lambda_u` | 手工设定 | `G_write_geo` 组合权重 |

---

## 16. 常见失败模式与缓解策略

### 16.1 把遮挡边界误判为动态区域

**原因**：只看点位残差，不看深度顺序。  
**缓解**：显式引入 `C_occ`，并在 `C_dyn` 中对 `C_occ` 做减项。

### 16.2 大面积低纹理区域不稳定，导致写入被过度抑制

**原因**：`Conf_geo` 低、支持帧有效投影少。  
**缓解**：将这类情况更多归入 `C_unc`，而不是直接推高 `C_dyn`。

### 16.3 动态物体在 chunk 内短暂停止，`C_stat` 虚高

**原因**：只看短时一致性。  
**缓解**：`C_anchor` 必须同时要求低 `C_dyn`、低 `C_unc`，且可选加入边界连续性项。

### 16.4 patch pooling 后边界被过度平均

**原因**：patch 内含静态背景与动态边界混合。  
**缓解**：第一版可用 mean pooling，若发现风险被稀释，可改用 `p90` 风险 pooling。

---

## 17. 最终接口摘要

### 输入

- `X_m: [T,H,W,3]`
- `P_cam: [T,H_p,W_p,3]`
- `T_w_c: [T,4,4]`
- `Conf_geo: [T,H_p,W_p]`
- `PatchMeta: [L_patch,3]`
- `PrevOverlap`（可选）

### 输出

- `E_cue: [T,H,W,5]`
- `G_write_geo: [T,H,W]`
- `E_cue_patch: [T,H_tok,W_tok,5]`（可选）
- `CueDebug`（可选）

### 五个 cue 通道

1. `C_stat`
2. `C_dyn`
3. `C_occ`
4. `C_unc`
5. `C_anchor`

### 一句话总结

**Dynamic Cue Extractor 的核心作用，是把 LoGeR 的几何结果转成一层可被直接用于 TTT write control 的几何证据场：哪些区域值得写、哪些区域应被抑制、哪些区域只是遮挡或不确定，而不是值得进入长期记忆的内容。**

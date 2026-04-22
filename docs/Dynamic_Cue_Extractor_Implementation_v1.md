# Dynamic Cue Extractor 实际实现说明（v1）

> 本文档描述当前仓库中 `loger/pipeline/dynamic_cue_extractor.py` 的**实际实现**，用于和设计文档区分。它回答的是“代码现在真正做了什么”，而不是“理想设计上希望做到什么”。

---

## 0. 一句话结论

当前版本的 `DynamicCueExtractor` 是一个**基于 chunk 内多帧几何一致性**的简化实现：

- 输入只使用 LoGeR Stage A 的 `world_points / local_points / camera_poses / confidence`
- 可选读取 Stage A 导出的 `frame_attention_prior / attn_dynamic_patch`
- 在当前 chunk 内为每帧构造 attention-aware 支持集
- 计算 `C_stat / C_occ / C_unc`
- 再结合 attention 动态先验得到 `C_dyn / C_anchor / G_write_geo`
- 可选输出 patch-level pooling 版本

它已经覆盖了设计文档中的主输出接口，但仍然属于一个偏 **Phase 1 / 1.5** 的实现，而不是完整的设计稿版本。

---

## 1. 代码位置

- 核心实现：`loger/pipeline/dynamic_cue_extractor.py`
- 独立运行与可视化入口：`inference_dynamic_cue_extractor.py`

---

## 2. 输入与输出

### 2.1 输入

`DynamicCueExtractor.run(geo)` 接收一个 `GeometryOutput`，实际使用的字段只有：

- `geo.world_points`：`[T, H_p, W_p, 3]`
- `geo.local_points`：`[T, H_p, W_p, 3]`
- `geo.camera_poses`：`[T, 4, 4]`
- `geo.confidence`：`[T, H_p, W_p]`
- `geo.frame_attention_prior`：`[T, T]`，可选
- `geo.attn_dynamic_patch`：`[T, H_tok, W_tok]`，可选
- `geo.patch_grid`：patch pooling 时使用

当前实现**没有**直接使用：

- RGB 图像
- patch token meta
- 上一 chunk overlap 缓存
- 法向图
- LoGeR 内部额外残差图

### 2.2 输出

返回 `CueOutput`，包含：

- `E_cue`：`[T, H_p, W_p, 5]`
- `G_write_geo`：`[T, H_p, W_p]`
- `E_cue_patch`：`[T, H_tok, W_tok, 5]`，可选
- `G_write_geo_patch`：`[T, H_tok, W_tok]`，可选
- `debug`：当前只保留少量调试信息

五个通道顺序固定为：

1. `C_stat`
2. `C_dyn`
3. `C_occ`
4. `C_unc`
5. `C_anchor`

---

## 3. 当前实现的数据流

```text
GeometryOutput
    ↓
提取 world/local points, pose, confidence, attention priors
    ↓
构造 attention-aware chunk 内支持集 N_intra(t)
    ↓
计算 pairwise point residual / depth-order occlusion / valid coverage
    ↓
attention-aware consistency weighting
    ↓
C_stat / C_occ / C_unc
    ↓
C_dyn(attn + geometry) / C_anchor
    ↓
G_write_geo
    ↓
可选 patch pooling
```

---

## 4. 支持集构造

当前实现使用 `_build_support_tensor(T, frame_attention_prior, attn_dynamic_patch)` 为每个时间帧 `t` 选择最多 `k_intra` 个支持帧：

- 如果没有 attention prior，按时间距离从近到远选取
- 如果有 attention prior，则按一个混合分数排序：
  - 时间接近性
  - `frame_attention_prior[t, s]`
  - `attn_dynamic_patch` 导出的静态区域 overlap
- 只在**当前 chunk 内部**选择支持帧

因此，当前实现中：

- 有 `N_intra(t)`
- 没有 `N_bdry(t)`
- 没有跨 chunk 边界连续性项

---

## 5. 几何残差与中间量

### 5.1 点位残差

当前实现的点位残差不是“投影后在支持帧采样对应点”的完整版本，而是更简化的：

```text
r_pt(t, s, u) = || X_t(u) - X_s(u) || / (eps + || X_t(u) ||)
```

也就是：

- 直接比较 `world_points` 在同一个像素位置上的差异
- 不做显式的 `u -> \tilde u_{t->s}` 采样

这是一个比设计稿更轻量的近似实现。

### 5.2 置信有效性

对于 `C_stat`，confidence 不直接乘进一致性分数，而是只作为有效性门限：

- 当前帧和支持帧都要求 `conf > conf_floor`
- 不满足时，该 pair 不参与 `C_stat` 聚合

### 5.3 遮挡检测

当前实现会把当前帧的世界点 `X_t` 变换到支持帧相机坐标系：

```text
X_in_s = R_s * X_t + t_s
z_proj = X_in_s[..., 2]
```

然后用一个简单的 depth-ordering 规则定义遮挡标记：

```text
occ_flag = 1[z_proj - depth_s > tau_occ]
```

这里的 `depth_s` 是支持帧的 `local_points[..., 2]`。

### 5.4 有效投影

当前实现没有显式二维投影与出界判断，而是用：

- `z_proj > 0`
- confidence 有效

来近似定义几何有效性。

---

## 6. 五个 cue 的实际定义

### 6.1 `C_stat`

先计算每个支持帧上的纯几何一致性：

```text
c = exp(-r_pt / sigma_pt)
```

然后对支持集做 trimmed mean，并在存在 attention prior 时额外计算一个 attention-aware 的加权一致性：

```text
C_stat_geom = TrimMean_s c(t, s, u)
C_stat_attn = WeightedMean_s c(t, s, u)
C_stat = (1-w_attn) * C_stat_geom + w_attn * C_stat_attn
```

实现细节：

- 不乘 confidence 权重
- 仅通过 `conf_floor` 排除低可信 pair
- 使用 `_trimmed_mean(...)` 做鲁棒聚合
- 若有 `attn_dynamic_patch`，则会把“当前帧和支持帧都更静态”的位置赋予更高的 consistency 权重

### 6.2 `C_occ`

`C_occ` 是 confidence 加权的遮挡冲突比例：

```text
w = conf_t * conf_s
C_occ = sum(occ_flag * w) / sum(w)
```

再裁剪到 `[0, 1]`。

### 6.3 `C_unc`

`C_unc` 由两部分线性混合：

1. 支持覆盖不足
2. 当前原始几何置信低

公式可写成：

```text
unc_support = 1 - valid_weighted_sum / (|N(t)| + eps)
unc_conf = 1 - conf
C_unc = (1 - unc_conf_weight) * unc_support + unc_conf_weight * unc_conf
```

### 6.4 `C_dyn`

当前实现与设计稿保持同一结构，但会额外融合 attention 动态证据：

```text
C_dyn_geom = clip(alpha_1 * (1 - C_stat) + alpha_2 * C_bdry - alpha_3 * C_occ, 0, 1)
C_dyn = (1-w_dyn_attn) * C_dyn_geom + w_dyn_attn * (M_attn * (1-C_unc))
```

但实际代码里：

- `alpha_2` 对应的边界项还没有实现
- `M_attn` 来自 `attn_dynamic_patch` 双线性上采样到 pixel grid 后的动态先验

因此它的含义更接近：

- “几何静态一致性不足”
- “并且 transformer attention 也觉得这里更像时序不稳定区域”
- 但“又不完全能被遮挡解释”

而不是一个纯粹的“运动物体显著图”。

### 6.5 `C_anchor`

当前实现：

```text
C_anchor = C_stat * (1 - C_dyn) * (1 - C_unc)
```

表示：

- 静态一致
- 动态风险低
- 不确定性低

的区域更适合做长期写入锚点。

---

## 7. `G_write_geo` 的实际定义

当前实现直接按线性组合后过 sigmoid：

```text
z_geo =
    lambda_s * C_stat
  + lambda_a * C_anchor
  - lambda_d * C_dyn
  - lambda_o * C_occ
  - lambda_u * C_unc

G_write_geo = sigmoid(z_geo)
```

它的作用是提供一个纯几何驱动的写入允许图。

---

## 8. Patch pooling

当 `compute_patch_cues=True` 且 `geo.patch_grid != (0, 0)` 时，当前实现会做**精确块平均**：

- `E_cue_patch`：对每个 cue 通道做 patch mean
- `G_write_geo_patch`：对写入图做 patch mean

实现不是基于 token meta 索引散射，而是直接按空间网格 reshape + mean。

---

## 9. 当前 debug 输出

当前 `debug` 中实际保留的内容较少，主要有：

- `support_count_per_frame`
- `mean_point_residual`
- `mean_point_residual_map`

还没有实现设计稿中更丰富的：

- `valid_support_ratio`
- `occlusion_conflict_map`
- `pairwise_consistency_map`
- `anchor_histogram`

---

## 10. 与设计文档相比，当前没有实现的部分

下面这些内容在设计稿里提到，但当前代码还没有实现：

- 使用 RGB 作为辅助线索
- 使用法向残差 `r_n`
- 跨 chunk overlap 支持集
- `C_bdry-break`
- 显式 2D reprojection 后在支持帧采样 `\tilde X_s(\tilde u)`
- 更完整的遮挡/显露建模
- 与 masklet discovery / refresh 的直接接口
- 更丰富的 debug 字段

---

## 11. 与设计文档相比，当前已经实现的核心部分

当前已经落实的部分有：

- 五通道 cue 输出接口
- 纯 chunk 内支持集
- 基于 point residual 的静态一致性
- 基于 depth ordering 的简化遮挡项
- 基于有效支持度和原始 confidence 的不确定性
- `C_dyn` 的一阶近似实现
- `C_anchor`
- `G_write_geo`
- patch-level pooling

---

## 12. 如何理解当前版本的定位

当前实现最适合被理解为：

**一个基于 chunk 内多帧几何一致性的一版简化 Dynamic Cue Extractor。**

它的优势是：

- 实现简单
- 计算稳定
- 与当前 pipeline 接口清晰
- 足以支持 Stage D / Stage E 的第一轮实验

它的局限是：

- `C_dyn` 更像“几何不一致性”而非“语义运动显著性”
- 缺少跨 chunk 边界建模
- 缺少法向与更完整的重投影几何

因此，在解释可视化结果时，应该把 `C_dyn` 理解为：

> “静态几何假设解释不通、且不完全由遮挡解释的程度”

而不应直接把它等同于“人在动的热力图”。

---

## 13. 推荐阅读顺序

如果想进一步对应代码，推荐按下面顺序阅读：

1. `DynamicCueExtractor.run(...)`
2. `_build_support_tensor(...)`
3. `_compute_pairwise_cues(...)`
4. `_trimmed_mean(...)`
5. `_patch_pool(...)`

若要看可视化如何渲染，再看：

- `inference_dynamic_cue_extractor.py`

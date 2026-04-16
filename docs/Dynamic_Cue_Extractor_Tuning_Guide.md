# Dynamic Cue Extractor 参数调整指南

> 本文档记录 Stage B 各超参数的含义、影响方向、调参方法和诊断手段，方便后续在不同场景上快速调参。

---

## 0. 快速参考：当前默认参数

| 参数 | 默认值 | 一句话含义 |
|---|---:|---|
| `k_intra` | 3 | 每帧取几个最近邻支持帧 |
| `sigma_pt` | 0.25 | 点位残差 → 一致性的衰减尺度 |
| `tau_occ` | 0.05 | 深度差超过多少判为遮挡 |
| `alpha_1` | 0.8 | C_dyn 对 (1-C_stat) 的放大系数 |
| `alpha_3` | 0.5 | C_dyn 中扣除 C_occ 的权重 |
| `conf_floor` | 0.1 | confidence 低于此值的像素排除出 C_stat 计算 |
| `unc_conf_weight` | 0.3 | C_unc 中 "低 confidence" 项的混合比例 |
| `lambda_s` | 1.0 | G_write_geo: C_stat 的正向权重 |
| `lambda_a` | 0.5 | G_write_geo: C_anchor 的正向权重 |
| `lambda_d` | 0.8 | G_write_geo: C_dyn 的负向权重 |
| `lambda_o` | 0.3 | G_write_geo: C_occ 的负向权重 |
| `lambda_u` | 0.5 | G_write_geo: C_unc 的负向权重 |

---

## 1. 核心公式回顾

```
C_stat       = TrimMean_k[ exp(-r_pt / sigma_pt) ]     （仅 conf > floor 的像素参与）
C_occ        = weighted_mean[ 1[z_proj - z_actual > tau_occ] ]
C_unc        = (1-w_unc) * unc_support + w_unc * (1-conf)
C_dyn        = clip( alpha_1 * (1-C_stat) - alpha_3 * C_occ,  0, 1 )
C_anchor     = C_stat * (1-C_dyn) * (1-C_unc)

z_geo        = λ_s*C_stat + λ_a*C_anchor - λ_d*C_dyn - λ_o*C_occ - λ_u*C_unc
G_write_geo  = sigmoid(z_geo)
```

其中 `r_pt = ||X_world[t,u] - X_world[s,u]|| / (ε + ||X_world[t,u]||)` 是同像素世界坐标相对残差。

---

## 2. 各参数详解与调整方向

### 2.1 `sigma_pt`（最关键的参数）

**含义**：控制 `exp(-r_pt / sigma_pt)` 的衰减速度。sigma_pt 越大，相同残差下的一致性越高。

**调参核心逻辑**：

- 同像素比较在相机运动时存在不可避免的 **视差基线残差**（即使全静态场景 r_pt 也不为 0）
- sigma_pt 必须显著大于这个基线残差，否则静态区域也会被判为不一致
- 经验法则：**sigma_pt ≈ 3 ~ 5 倍 × 静态场景高置信像素的 r_pt 中位数**

**调参方向**：

| 现象 | 调整 |
|---|---|
| 静态场景 C_stat 偏低 / C_dyn 偏高 | ↑ 增大 sigma_pt |
| 动态物体与静态背景 C_stat 无区分度 | ↓ 减小 sigma_pt |
| 不同场景基线残差差异大 | 考虑自适应 sigma（见后文） |

**如何获取场景的基线残差分布**：

```bash
# 在 inference 脚本的 debug 输出中查看
mean_point_residual : 0.084   # 所有支持对的平均残差
```

更精细的分位数分析可以用如下脚本（已在调参过程中使用）：

```python
# 在拿到 GeometryOutput 后：
for t in range(T):
    for s in [t-1, t+1]:
        diff = (world_pts[t] - world_pts[s]).norm(dim=-1)
        r_pt = diff / (1e-7 + world_pts[t].norm(dim=-1))
        mask = (conf[t] > 0.5) & (conf[s] > 0.5)
        # 收集 r_pt[mask] 的分位数
```

**参考值**（基于 LoGeR office 场景实测）：

| 分位数 | 值 | 含义 |
|---|---|---|
| P50 | 0.029 | 静态场景中位视差残差 |
| P75 | 0.048 | |
| P90 | 0.080 | |
| P95 | 0.118 | |

| sigma_pt | C_stat 均值 (静态场景) |
|---|---|
| 0.03 | ~0.31 （太低） |
| 0.10 | ~0.59 |
| 0.25 | ~0.75 （当前默认） |
| 0.50 | ~0.85 （对动态不敏感） |

### 2.2 `alpha_1`（C_dyn 的放大系数）

**含义**：`C_dyn = alpha_1 * (1 - C_stat) - alpha_3 * C_occ`，alpha_1 控制"不一致区域"被转化为"动态"判断的强度。

**调参方向**：

| 现象 | 调整 |
|---|---|
| 静态场景 C_dyn 偏高 | ↓ 减小 alpha_1（如 0.5~0.8） |
| 动态物体 C_dyn 不够高 | ↑ 增大 alpha_1（如 1.0~1.5） |
| C_dyn max 被 clip 到 1.0 太多 | ↓ 减小 alpha_1 |

**注意**：alpha_1 和 sigma_pt 共同决定 C_dyn。二者应联动调整，不要单独调。

### 2.3 `alpha_3`（C_dyn 中 C_occ 的扣减权重）

**含义**：遮挡边界不应被算作"动态"，alpha_3 控制从 C_dyn 中扣除遮挡的力度。

- 增大 → C_occ 区域的 C_dyn 更低（更少误判遮挡为动态）
- 减小 → 遮挡区域也可能有较高 C_dyn

当前默认 0.5 通常不需要调。

### 2.4 `tau_occ`（遮挡深度阈值）

**含义**：当 `z_proj - z_actual > tau_occ` 时判定为遮挡（当前帧的点在支持帧中被其他表面遮挡）。

| 现象 | 调整 |
|---|---|
| C_occ 偏高（误报） | ↑ 增大 tau_occ |
| 真实遮挡边界 C_occ 不明显 | ↓ 减小 tau_occ |

**参考值**（office 场景实测）：

| tau_occ | 遮挡像素占比 |
|---|---|
| 0.01 | 14.9% |
| 0.02 | 6.8% |
| 0.03 | 3.6% |
| 0.05 | 1.8% （当前默认） |
| 0.10 | 1.0% |

### 2.5 `k_intra`（支持帧数）

**含义**：每帧选 k 个最近邻帧做比较。

- 增大 → trimmed mean 更稳健，但远距离帧的视差残差更大（推高基线噪声）
- 减小 → 视差更小，但统计不够稳健

建议范围 2~4。当前默认 3 通常合适。

### 2.6 `lambda_*`（G_write_geo 的线性组合权重）

**含义**：`z_geo = λ_s*C_stat + λ_a*C_anchor - λ_d*C_dyn - λ_o*C_occ - λ_u*C_unc`，经 sigmoid 后得到 G_write_geo。

调参思路：
- 正项（λ_s, λ_a）控制"允许写入"的力度
- 负项（λ_d, λ_o, λ_u）控制"抑制写入"的力度
- sigmoid 的特性：z_geo=0 时 G=0.5，z_geo=1 时 G=0.73，z_geo=-1 时 G=0.27

| 现象 | 调整 |
|---|---|
| G_write_geo 整体偏高 | ↑ 增大负项权重或 ↓ 减小正项权重 |
| G_write_geo 整体偏低 | ↓ 减小负项权重或 ↑ 增大正项权重 |
| 动态区域写入没被有效抑制 | ↑ 增大 lambda_d |
| 静态区域写入被过度抑制 | ↓ 减小 lambda_u |

### 2.7 `conf_floor` 和 `unc_conf_weight`

- `conf_floor = 0.1`：confidence 低于此值的像素**排除**在 C_stat 计算之外（不参与而非惩罚）
- `unc_conf_weight = 0.3`：C_unc = 0.7×unc_support + 0.3×(1-conf)

这两个参数通常不需要调。如果发现低 confidence 区域的 C_stat 异常高（不应该），可以提高 conf_floor。

---

## 3. 典型场景调参策略

### 3.1 室内静态小运动（如 office）

目标：C_stat 高、C_dyn 低、G_write_geo 高。

```
sigma_pt = 0.25, alpha_1 = 0.8, tau_occ = 0.05
```

### 3.2 室外大运动 + 静态建筑

相机运动大，视差基线残差会显著增大。需要：

```
sigma_pt = 0.40~0.60    # 更宽容
alpha_1 = 0.8~1.0       # 保持
tau_occ = 0.08~0.15     # 大运动下深度差也更大
```

### 3.3 含动态物体的场景（行人、车辆）

需要保持对动态的敏感度。不能 sigma_pt 太大：

```
sigma_pt = 0.15~0.25    # 动态物体 r_pt 通常 0.2~0.5
alpha_1 = 1.0~1.2       # 适度放大动态信号
tau_occ = 0.05
```

### 3.4 长视频 / 大 chunk

k_intra=3 时支持帧跨度增大（帧 t±2），视差更大：

```
k_intra = 2             # 只用相邻帧
sigma_pt = 0.20         # 可以小一些因为基线更短
```

---

## 4. 自适应 sigma_pt（未来改进方向）

当前 sigma_pt 是全局固定值。如果不同场景基线残差差异很大，可以：

1. **基于当前 chunk 自适应**：取前几帧 r_pt 的 P75 作为 sigma_pt
2. **基于相机运动自适应**：相机平移量越大，sigma_pt 越大
3. **基于深度自适应**：远处区域 sigma_pt 更大（视差残差与深度正相关）

示例（概念代码）：

```python
# 自适应 sigma: 取高置信相邻帧残差的 P75 × 倍率
r_sample = compute_adjacent_residuals(world_pts, conf, floor=0.5)
sigma_pt = torch.quantile(r_sample, 0.75).item() * 3.5
```

---

## 5. 快速诊断 checklist

当输出不符合预期时，按以下顺序排查：

1. **看 `mean_point_residual`**：如果它远大于 sigma_pt，C_stat 一定偏低 → 增大 sigma_pt
2. **看 C_stat 的 min/max/mean**：
   - mean < 0.5 → sigma_pt 太小
   - max < 0.8 → sigma_pt 远小于场景基线
   - min = 0.0 很多 → 正常（低 conf 区域）
3. **看 C_dyn 的 max**：
   - max = 1.0（被 clip）→ alpha_1 偏大
   - max < 0.5 → alpha_1 可能偏小（对动态场景不够敏感）
4. **看 C_occ 的 mean**：
   - > 0.10 → tau_occ 偏小（太敏感）
   - ≈ 0 → tau_occ 偏大或场景无遮挡
5. **看 G_write_geo 的 mean**：
   - 静态场景应 > 0.55
   - 动态主导场景应 < 0.45

---

## 6. 命令行快速实验

```bash
# 默认参数
python inference_dynamic_cue_extractor.py \
    --input data/examples/office \
    --config ckpts/LoGeR/original_config.yaml \
    --checkpoint ckpts/LoGeR/latest.pt

# 调大 sigma_pt（宽松一致性判断）
python inference_dynamic_cue_extractor.py \
    --input data/examples/office \
    --config ckpts/LoGeR/original_config.yaml \
    --checkpoint ckpts/LoGeR/latest.pt \
    --sigma_pt 0.40 --alpha_1 0.6

# 调小 sigma_pt + 调大 alpha_1（更敏感）
python inference_dynamic_cue_extractor.py \
    --input data/examples/office \
    --config ckpts/LoGeR/original_config.yaml \
    --checkpoint ckpts/LoGeR/latest.pt \
    --sigma_pt 0.15 --alpha_1 1.2
```

所有 cue extractor 参数都可以通过命令行覆盖，无需改代码。

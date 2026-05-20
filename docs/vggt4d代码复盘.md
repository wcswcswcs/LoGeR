# VGGT4D 代码复盘：论文公式到实现的对应关系

日期：2026-05-21（Asia/Singapore）

阅读对象与范围：

```text
论文：docs/VGGT4D.pdf
代码：third_party/VGGT4D/
```

审计边界：

```text
本复盘只做论文公式与本地代码的静态对照。
没有运行 VGGT4D 推理，也没有生成新的实验数值。
不把论文描述中没有在代码里看到的功能当成已实现功能。
```

---

## 1. 整体流程

VGGT4D 的代码流程可以理解成三步：

1. 先跑一次不带动态 mask 的 VGGT4D，得到深度、相机位姿、特征 token，以及每一层 attention 里的 Query/Key 向量。
2. 用 Query/Key 的相似度构造动态区域分数图，再用特征聚类和阈值变成粗动态 mask。
3. 用粗动态 mask 再跑一次模型，在浅层 attention 中屏蔽动态区域 token，得到更稳的相机位姿；最后用深度、颜色和跨视角投影误差精修动态 mask。

主流程代码在：

```text
third_party/VGGT4D/demo_vggt4d.py:263-345
```

其中：

```text
Stage 1:
    inference(model, images)
    organize_qk_dict(...)
    extract_dyn_map(...)
    cluster_attention_maps(...)
    adaptive_multiotsu_variance(...)

Stage 2:
    inference(model, images, dyn_masks)

Stage 3:
    RefineDynMask(...).refine_masks()
```

输出文件在 README 中列为深度图、置信度、相机内参、相机位姿、refined dynamic masks 和 RGB 图像：

```text
third_party/VGGT4D/README.md:106-114
```

---

## 2. 模型入口和 token 结构

VGGT4D 模型入口：

```text
third_party/VGGT4D/vggt4d/models/vggt4d.py:24-97
```

`VGGTFor4D.forward()` 接收：

```text
images:
    输入图像序列。

dyn_masks:
    可选动态 mask。第一次推理为空；第二次推理传入粗动态 mask。
```

它调用：

```text
third_party/VGGT4D/vggt4d/models/aggregator.py:17-148
```

token 结构来自 VGGT 的 Aggregator：

```text
third_party/VGGT4D/vggt/models/aggregator.py:52-70
third_party/VGGT4D/vggt/models/aggregator.py:125-131
```

关键点：

```text
depth = 24
num_heads = 16
num_register_tokens = 4
patch_size = 14
patch_start_idx = 1 + num_register_tokens = 5
```

也就是说，每帧 token 前 5 个是特殊 token：

```text
1 个 camera token
4 个 register token
后面才是图像 patch token
```

这解释了为什么动态 mask 进入 attention 前会先 pad 5 个 False：

```text
third_party/VGGT4D/vggt4d/layers/attention.py:24-29
```

这些特殊 token 不会被当成动态 patch 屏蔽。

---

## 3. 公式到代码：动态 cue 提取

### 3.1 论文 Eq. 1：标准 QK attention

论文 Eq. 1 写的是标准注意力相似度：

```text
A_QK = Q K^T / sqrt(c)
```

基础 VGGT attention 的代码在：

```text
third_party/VGGT4D/vggt/layers/attention.py:50-72
```

VGGT4D 改写后的 attention 在：

```text
third_party/VGGT4D/vggt4d/layers/attention.py:63-95
```

实现中使用 PyTorch 的：

```text
F.scaled_dot_product_attention(q, k, v)
```

对应论文里的 `QK^T / sqrt(c)`、softmax 和乘以 `V`。

### 3.2 论文 Eq. 2：QQ 和 KK Gram similarity

论文 Eq. 2 改用同类向量之间的相似度：

```text
A_QQ = Q Q^T / sqrt(c)
A_KK = K K^T / sqrt(c)
```

代码里没有单独叫 `A_QQ` 或 `A_KK` 的变量，而是在 `dynamic_mask.py` 里直接做矩阵乘：

```text
Q-Q:
    attn_map = q_ref @ q_src.transpose(-2, -1)
    third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:40
    third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:86
    third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:132

K-K:
    attn_map = k_ref @ k_src.transpose(-2, -1)
    third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:178

Q-K:
    attn_map = q_ref @ k_src.transpose(-2, -1)
    third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:224
```

审计备注：

```text
代码中没有显式除以 sqrt(c)。
因为后面每张 map 都做 min-max normalize，常数缩放不会改变相对排序。
```

### 3.3 论文 Eq. 3 / Eq. 4：时间窗口和层区间上的均值 / 方差

论文 Eq. 3 和 Eq. 4 是：

```text
在 temporal window W(t) 里，对若干层的 Gram map 求均值 S 和方差 V。
```

代码里的 temporal window 是：

```text
window = torch.tensor([-6, -4, -2, 2, 4, 6])
```

对应位置：

```text
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:11
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:57
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:103
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:149
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:195
```

越界帧会被过滤：

```text
src_ids = src_ids[src_ids >= 0]
src_ids = src_ids[src_ids < n_img]
```

对应位置：

```text
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:18-19
```

均值统计对应：

```text
attn_map.mean(...)
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:45
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:137
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:183
```

方差/标准差统计对应：

```text
attn_map.mean(...).std(...)
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:91
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:229
```

审计备注：

```text
论文写 Var。
代码实际用了 std，也就是标准差。
```

### 3.4 论文 Eq. 5 - Eq. 8：动态分数 Dyn

论文把动态分数写成：

```text
Dyn = w_shallow * w_middle * w_deep
w_shallow = (1 - S_QK_shallow) * V_KK_shallow
w_middle  = 1 - S_QQ_middle
w_deep    = (1 - V_QQ_deep) * S_QQ_deep
```

代码中的核心组合在：

```text
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:248-255
```

实际实现是：

```text
mean1_map = extract_mean1_map(...)          # Q-Q, layer 3-7
mean2_map = extract_mean2_map(...)          # Q-Q, layer 17-21
mean3_map = extract_mean3_map(...)          # K-K, layer 0
var1_map  = extract_spacial_var1_map(...)   # Q-Q std, layer 18-19
var3_map  = extract_spacial_var3_map(...)   # Q-K std, layer 0

dyn_map = (1 - mean1_map) * (1 - var1_map) * mean2_map * (1 - mean3_map) * var3_map
```

对应函数和层区间：

```text
extract_mean1_map:
    layer_ids = torch.arange(3, 8)
    Q-Q mean
    third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:9-52

extract_spacial_var1_map:
    layer_ids = torch.arange(18, 20)
    Q-Q std
    third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:55-98

extract_mean2_map:
    layer_ids = torch.arange(17, 22)
    Q-Q mean
    third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:101-144

extract_mean3_map:
    layer_ids = torch.arange(0, 1)
    K-K mean
    third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:147-190

extract_spacial_var3_map:
    layer_ids = torch.arange(0, 1)
    Q-K std
    third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:193-236
```

审计备注：

```text
代码和论文公式不是逐字同名实现。
论文给的是三项抽象公式；代码落地成五个归一化 map 的乘积。
层号也按 0-index 写在代码中：
    layer 0 ~= 论文 Layer 1
    layer 3-7 ~= 论文 Layers 4-8
    layer 17-21 ~= 论文 Layers 18-22
```

### 3.5 论文中的阈值和 feature clustering

论文说先得到动态分数，再通过 feature clustering 和 Otsu 阈值变成 mask。

代码中：

```text
cluster_attention_maps(...)
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:303-350

adaptive_multiotsu_variance(...)
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:353-391
```

demo 里调用方式：

```text
feat_map = rearrange(enc_feat, ...)
norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)
upsampled_map = F.interpolate(...)
thres = adaptive_multiotsu_variance(...)
dyn_masks = upsampled_map > thres
```

对应位置：

```text
third_party/VGGT4D/demo_vggt4d.py:279-293
```

---

## 4. 公式到代码：mask refinement

### 4.1 论文 Eq. 9：投影深度残差

论文 Eq. 9 用投影到其他视角后的深度残差判断动态点：

```text
r_d = projected_depth - sampled_depth
```

代码里对应：

```text
sample_depths = grid_sample_depth(...)
depth_diff = pick_pts_proj[..., 2] - sample_depths
```

位置：

```text
third_party/VGGT4D/vggt4d/masks/refine_dyn_mask.py:141
third_party/VGGT4D/vggt4d/masks/refine_dyn_mask.py:172
```

它还会要求点可见、投影有效，并且投影位置不是已知动态区域：

```text
visible_mask = pick_pts_proj[..., 2] - 0.01 < sample_depths
loss_mask = visible_mask & (~sample_dyn_masks)
loss_mask = loss_mask & valid_proj
```

位置：

```text
third_party/VGGT4D/vggt4d/masks/refine_dyn_mask.py:158-162
```

### 4.2 论文 Eq. 10：跨视角聚合投影梯度

论文 Eq. 10 写的是带梯度的聚合项。

代码实现没有显式 autograd 求 `∇r_d`，而是用跨视角深度误差的平均值作为几何动态分数：

```text
valid_depth_diff = torch.abs(valid_depth_diff)
depth_loss = valid_depth_diff.sum() / loss_mask.sum()
```

位置：

```text
third_party/VGGT4D/vggt4d/masks/refine_dyn_mask.py:174-181
```

审计备注：

```text
论文称 projection gradient-aware。
当前代码看起来是 depth residual + RGB residual 的聚合版本，没有看到显式对 3D 点坐标求梯度的实现。
```

### 4.3 论文 Eq. 11 / Eq. 12：颜色残差和总分

论文 Eq. 11 是 photometric residual，Eq. 12 把几何和颜色加起来。

代码对应：

```text
rgb_diff = pick_rgb.unsqueeze(0) - sample_rgbs
valid_rgb_diff = torch.abs(valid_rgb_diff)
rgb_loss = valid_rgb_diff.sum() / loss_mask.sum()
total_loss = depth_loss + rgb_loss / 3
```

位置：

```text
third_party/VGGT4D/vggt4d/masks/refine_dyn_mask.py:173-183
```

然后按 cluster label 判断动态：

```text
thres = 0.1
selected_labels = [label for label, _, _, loss in label_losses if loss > thres]
refine_dyn_mask = torch.isin(pts_labels, selected_labels)
```

位置：

```text
third_party/VGGT4D/vggt4d/masks/refine_dyn_mask.py:226-231
```

精修前还做了点云离群点过滤和 KMeans 聚类：

```text
remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
KMeans(n_clusters=30, random_state=42)
```

位置：

```text
third_party/VGGT4D/vggt4d/masks/refine_dyn_mask.py:196-213
```

精修后还做形态学闭运算和膨胀：

```text
cv2.morphologyEx(...)
cv2.dilate(...)
```

位置：

```text
third_party/VGGT4D/vggt4d/masks/refine_dyn_mask.py:241-246
```

---

## 5. 动态区域 token 是怎么“跳过计算”的

这是本次复盘最关键的点。

结论：

```text
代码不是把动态 token 从整个网络里删掉。
代码仍然会为所有 token 做 patch embedding、QKV 线性投影、Query 输出和后续 MLP。
真正跳过的是 early layers attention 里的动态 token Key/Value 参与。
```

执行链如下。

### 5.1 像素级 mask 变成 patch 级 mask

输入的 `dyn_masks` 形状是：

```text
[B, S, H, W]
```

Aggregator 里用 patch 大小做 max pooling：

```text
dyn_masks = F.max_pool2d(dyn_masks.float(), kernel_size=self.patch_size, stride=self.patch_size)
dyn_masks = rearrange(dyn_masks, "b s h w -> b s (h w)") > 0.5
```

位置：

```text
third_party/VGGT4D/vggt4d/models/aggregator.py:48-51
```

含义：

```text
只要一个 14x14 patch 内有足够动态像素，这个 patch token 就被标为动态。
```

代码里曾经尝试把动态 patch token 直接置零，但注释说明效果不好，所以没有启用：

```text
# bad effect
# patch_tokens[...] = 0
```

位置：

```text
third_party/VGGT4D/vggt4d/models/aggregator.py:52-56
```

### 5.2 mask 传给 frame/global attention block

Aggregator 交替跑两种 attention：

```text
frame attention:
    每帧内部的 token 互相看。

global attention:
    所有帧的 token 合在一起互相看。
```

传入 mask 的位置：

```text
third_party/VGGT4D/vggt4d/models/aggregator.py:90-107
third_party/VGGT4D/vggt4d/models/aggregator.py:161-191
third_party/VGGT4D/vggt4d/models/aggregator.py:193-222
```

Block 继续把 `dyn_masks` 传给 attention：

```text
third_party/VGGT4D/vggt4d/layers/block.py:18-23
```

### 5.3 只在早期层开启动态 mask

VGGT4D attention 中的门控：

```text
need_mask_atten = dyn_masks is not None
need_mask_atten = need_mask_atten and layer_id in range(0, 5)
```

位置：

```text
third_party/VGGT4D/vggt4d/layers/attention.py:74-80
```

这里 `range(0, 5)` 是 Python 的 0-index 层号：

```text
0, 1, 2, 3, 4
```

对应论文中的：

```text
Layers 1-5
```

这与 PDF Sec. 7.4 的 early-stage masking 描述一致。

### 5.4 具体怎么跳过动态 token

核心代码：

```text
dyn_mask = dyn_masks[b]
non_dyn_idx = (~dyn_mask).nonzero(as_tuple=True)[0]

non_dyn_k = kb[..., non_dyn_idx, :].contiguous()
non_dyn_v = vb[..., non_dyn_idx, :].contiguous()

o = F.scaled_dot_product_attention(qb, non_dyn_k, non_dyn_v)
```

位置：

```text
third_party/VGGT4D/vggt4d/layers/attention.py:47-59
```

含义：

```text
Query:
    仍然保留所有 token，包括动态 token。

Key:
    只保留非动态 token。

Value:
    只保留非动态 token。

attention 输出:
    所有 Query 都只能从静态 Key/Value 中读取信息。
```

所以，动态 token 被跳过的准确说法是：

```text
动态 patch token 不再作为被其他 token 读取的 Key/Value。
它们不能向早期 attention 的上下文写入动态信息。
但它们自己仍作为 Query 产生输出，并继续经过后续网络。
```

这比“完全删除动态 token”更保守，也更接近论文的目标：防止早期动态区域污染几何估计。

审计备注：

```text
论文文字强调 suppress Key(K) vectors。
代码实现同时过滤了 K 和 V：
    non_dyn_k
    non_dyn_v

因此代码比论文文字描述更强一点：不仅不让动态 token 被注意到，也不让其 Value 被聚合。
```

### 5.5 frame attention 和 global attention 的 mask 形状

frame attention 中：

```text
dyn_masks = rearrange(dyn_masks, "b s n -> (b s) n")
```

位置：

```text
third_party/VGGT4D/vggt4d/layers/attention.py:31-35
```

也就是每帧单独做 mask。

global attention 中：

```text
dyn_masks = rearrange(dyn_masks, "b s n -> b (s n)")
```

位置：

```text
third_party/VGGT4D/vggt4d/layers/attention.py:37-45
```

也就是把整段序列的 token 拼到一起做 mask。

审计备注：

```text
attention.py:35-45 中的 cam_idx / rest_idx 变量被创建了，但没有参与后续计算。
真正生效的是 non_dyn_idx 对 K/V 的过滤。
```

---

## 6. 中间层 token 的内存优化

论文提到只保留预测头需要的层，例如 5、12、18、24。

代码中的 0-index 对应层是：

```text
[4, 11, 17, 23]
```

AggregatorFor4D 中：

```text
preserve_layer_idx = [4, 11, 17, 23]
```

位置：

```text
third_party/VGGT4D/vggt4d/models/aggregator.py:88
```

如果启用 memory saving，不在这些层的输出会被置空并删除：

```text
if i not in preserve_layer_idx:
    output_list[i * B + j] = None
```

位置：

```text
third_party/VGGT4D/vggt4d/models/aggregator.py:115-121
```

`model_utils.inference()` 里也再次清理非关键层：

```text
if i not in [4, 11, 17, 23]:
    agg_tokens_list[i] = None
```

位置：

```text
third_party/VGGT4D/vggt4d/utils/model_utils.py:39-43
```

DPTHead 默认确实只读取这些中间层：

```text
intermediate_layer_idx = [4, 11, 17, 23]
third_party/VGGT4D/vggt/heads/dpt_head.py:43-64

for layer_idx in self.intermediate_layer_idx:
    x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
third_party/VGGT4D/vggt/heads/dpt_head.py:205-207
```

审计备注：

```text
README 里 Long sequence implementation 仍标为 TODO：
third_party/VGGT4D/README.md:116-122

所以本地代码里能确认的是“中间层 token 内存清理”，不是完整的长序列 streaming 实现。
```

---

## 7. Q/K 是怎么被保存给动态 mask 使用的

VGGT4D attention 返回：

```text
x, q, k
```

位置：

```text
third_party/VGGT4D/vggt4d/layers/attention.py:95
```

BlockFor4D 继续返回：

```text
return x, attn_q, attn_k
```

位置：

```text
third_party/VGGT4D/vggt4d/layers/block.py:41-44
```

AggregatorFor4D 收集每层 frame/global 的 Q/K：

```text
frame_q_list.append(frame_q.detach().cpu())
frame_k_list.append(frame_k.detach().cpu())
global_q_list.append(global_q.detach().cpu())
global_k_list.append(global_k.detach().cpu())
```

位置：

```text
third_party/VGGT4D/vggt4d/models/aggregator.py:93-105
```

最后组成：

```text
qk_dict = {
    "global_q": global_q,
    "global_k": global_k,
    "frame_q": frame_q,
    "frame_k": frame_k
}
```

位置：

```text
third_party/VGGT4D/vggt4d/models/aggregator.py:123-136
```

`organize_qk_dict()` 再把特殊 token 和 patch token 分开：

```text
global_cam_q = global_q[..., 0:1, :]
global_reg_q = global_q[..., 1:patch_start_idx, :]
global_tok_q = global_q[..., patch_start_idx:, :]
```

位置：

```text
third_party/VGGT4D/vggt4d/utils/model_utils.py:57-109
```

`extract_dyn_map()` 只用 image patch token：

```text
global_q = qk_dict["global_tok_q"].to("cuda")
global_k = qk_dict["global_tok_k"].to("cuda")
```

位置：

```text
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py:239-246
```

审计备注：

```text
代码还读取了 global_cam_q，但后续没有使用。
```

---

## 8. 和论文描述不完全一致的地方

本地代码与论文总体方向一致，但有几个实现细节需要审计时注意：

1. 论文 Eq. 2 写了 `/ sqrt(c)`，代码里的 Gram map 没有显式除以 `sqrt(c)`。
   由于后面做 min-max normalize，这通常只影响绝对尺度，不影响排序。

2. 论文 Eq. 4 写 variance，代码中动态 map 的两个方差类统计实际用了 `std`。

3. 论文 Eq. 5-8 是三个因子，代码实际用五个归一化 map 相乘：
   `Q-Q middle/deep mean`、`Q-Q deep std`、`K-K shallow mean`、`Q-K shallow std`。

4. 论文说 suppress dynamic token 的 Key 向量，代码实际过滤了 dynamic token 的 Key 和 Value。

5. 论文 Sec. 7.3 说 projection gradient，代码没有看到显式 `∇r_d` 的 autograd 或解析梯度实现，而是用深度残差和颜色残差做 cluster-level 判断。

6. 论文提到 long-sequence inference / FastVGGT，当前 repo README 仍把 Long sequence implementation 标为 TODO。本地代码可确认的是非关键中间层 token 清理。

---

## 9. 一句话总结

VGGT4D 先从 VGGT attention 的 Q/K 向量中挖出动态区域，再把动态区域转成 patch mask；第二次推理时，只在前 5 层 attention 里把这些动态 patch 从 Key/Value 集合中拿掉，让模型早期只能从静态区域聚合信息。这个实现不是完全跳过动态 token 的所有计算，而是跳过动态 token 作为上下文来源的 attention 计算。

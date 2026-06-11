# 当前代码审计：KITTI projected GT 2D semantic 与 VGGT4D-style source-skip attention

日期：2026-05-24（Asia/Singapore）

审计目标：

```text
1. 检查当前代码使用的 KITTI / SemanticKITTI projected GT 2D semantic 是否正确。
2. 生成可人工检查的并列可视化：
       RGB original | VideoMasklet frontend | projected GT semantic
3. 解释并修复初版可视化中 VideoMasklet 与 GT 颜色不一致的问题。
4. 仔细对照 third_party/VGGT4D，检查当前 source-skip / 区域 token attention 是否真的模仿正确。
```

边界：

```text
Projected GT semantic 是 SemanticKITTI LiDAR label 经过 KITTI calibration 投影到 image_2 的稀疏点级 GT。
它不是 dense 2D semantic segmentation。

VideoMasklet frontend 是 dense video masklet semantic/predicted frontend。
二者可以几何对照，但不能把 VideoMasklet 当作 GT。
```

---

## 1. 可视化工具与命令

新增工具：

```text
tools/current_gt2d_semantic_masklet_visual_audit.py
```

用途：

```text
1. 读取原始 KITTI image_2。
2. 读取 Stage-C VideoMasklet masklet.pt。
3. 读取 v29C projected SemanticKITTI cache：
       *_sem_sparse.npy
       *_valid_mask.npy
4. 通过 GTSemanticProvider 重新加载同帧 GT，并与 projected cache 做 exact match。
5. 输出并列图：
       RGB | VideoMasklet frontend overlay | Sparse projected SemanticKITTI GT
```

执行命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/current_gt2d_semantic_masklet_visual_audit.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/current_gt2d_semantic_masklet_visual_audit.py \
    --frames 290,464,650 \
    --point-radius 1 \
    --out-dir results/kitti01_hmc_v2/current_code_semantic_vggt4d_audit/gt2d_semantic_visual_unified_palette
```

输出目录：

```text
results/kitti01_hmc_v2/current_code_semantic_vggt4d_audit/gt2d_semantic_visual_unified_palette/
```

关键输出：

```text
frame_000290_rgb_masklet_gtsem.png
frame_000464_rgb_masklet_gtsem.png
frame_000650_rgb_masklet_gtsem.png
gt2d_semantic_visual_audit.csv
gt2d_semantic_visual_audit_summary.json
```

---

## 2. 颜色为什么一开始不一致

初版图里颜色不一致的原因是：

```text
VideoMasklet frontend panel 使用的是内部 coarse group palette。
Projected GT panel 使用的是 SemanticKITTI label palette。
```

这会导致同一个语义类在两边颜色不同，例如 road / vegetation / fence 看起来不像同类，影响人工审计。

已修复：

```text
tools/current_gt2d_semantic_masklet_visual_audit.py

默认参数：
    --masklet-color-mode semantic_kitti

VideoMasklet fine label 映射到 SemanticKITTI ID / color：
    road -> 40
    sidewalk -> 48
    building / wall -> 50
    fence -> 51
    vegetation / tree -> 70
    terrain / grass -> 72
    car -> 10
    moving_car -> 252
```

特殊情况：

```text
sky -> sky(no-gt)
```

解释：

```text
SemanticKITTI LiDAR projection 没有 sky 类。
VideoMasklet 可以预测 sky，但 projected GT 不能验证 sky。
所以 sky 使用独立颜色，并在图中文字里标注为 sky(no-gt)。
```

当前 summary 已记录：

```json
{
  "frames_ok": 3,
  "frames_requested": [290, 464, 650],
  "masklet_color_mode": "semantic_kitti",
  "provider_cache_exact_match_all_ok": true,
  "note": "Projected GT is sparse SemanticKITTI LiDAR-to-image semantic, not dense 2D semantic."
}
```

---

## 3. 可视化检查结果

人工打开检查的统一 palette 图：

```text
frame_000290_rgb_masklet_gtsem.png
frame_000464_rgb_masklet_gtsem.png
frame_000650_rgb_masklet_gtsem.png
```

逐帧结果：

| Frame | Provider/cache exact match | Valid projected pixels | Coverage | Top projected labels |
|---:|---|---:|---:|---|
| 290 | true | 13607 | 0.0291610232 | road / fence / terrain / vegetation |
| 464 | true | 14504 | 0.0310833748 | road / fence / moving_car / terrain |
| 650 | true | 18664 | 0.0399986284 | road / vegetation / fence |

人工检查结论：

```text
1. road 的 projected GT 点落在可见道路区域，VideoMasklet road overlay 与 GT 颜色一致。
2. vegetation / terrain / fence 在道路两侧、护栏和植被区域位置合理。
3. frame 000464 中 moving_car projected 点与画面中左侧车辆位置一致。
4. 没看到全局左右翻转、上下翻转、明显平移或 calibration 错位。
5. GT panel 是稀疏点投影，因此 sky 和远处无 LiDAR 点区域没有 GT，并不是语义缺失错误。
```

结论：

```text
当前 GTSemanticProvider 加载的 GT 2D semantic 与 v29C landed projection cache exact match。
代表帧可视化支持 projected sparse GT 的几何正确性。
```

不能声称：

```text
不能声称它是 dense 2D semantic GT。
不能用 VideoMasklet predicted semantic 替代 GT。
不能用 sky 的 VideoMasklet mask 与 projected GT 做一致性判断。
```

---

## 4. VGGT4D 原始实现审计

读取文件：

```text
third_party/VGGT4D/vggt4d/layers/attention.py
third_party/VGGT4D/vggt4d/models/aggregator.py
third_party/VGGT4D/vggt4d/masks/dynamic_mask.py
```

VGGT4D attention source-skip 机制：

```text
AttentionFor4D.forward:
    只有 dyn_masks != None 且 layer_id in range(0,5) 时启用 masked attention。

attention_with_dynamic_mask:
    1. 给 5 个 special tokens padding False，保护它们不被当作 dynamic token 去掉。
    2. frame attention:
           dyn_masks: b s n -> (b s) n
    3. global attention:
           dyn_masks: b s n -> b (s n)
    4. 对每个 batch：
           non_dyn_idx = (~dyn_mask).nonzero(...)
           non_dyn_k = k[..., non_dyn_idx, :]
           non_dyn_v = v[..., non_dyn_idx, :]
           output = scaled_dot_product_attention(q, non_dyn_k, non_dyn_v)
```

关键点：

```text
VGGT4D 保留所有 query rows。
VGGT4D 只从 Key/Value source columns 中移除 dynamic tokens。
```

VGGT4D dynamic map 计算：

```text
mean1:
    q_ref @ q_src
    layers 3..7
    temporal offsets [-6,-4,-2,2,4,6]

var1:
    q_ref @ q_src
    layers 18..19
    spatial std

mean2:
    q_ref @ q_src
    layers 17..21

mean3:
    k_ref @ k_src
    layer 0

var3:
    q_ref @ k_src
    layer 0
    spatial std

dyn_map:
    (1 - mean1) * (1 - var1) * mean2 * (1 - mean3) * var3
```

然后 demo 里还会：

```text
1. 用 encoder feature 做 KMeans clustering。
2. 用 multi-Otsu / adaptive threshold 生成 dyn_masks。
3. 再把 dyn_masks 输入 VGGT4D inference。
```

---

## 5. 当前 LoGeR source-skip attention 审计

读取文件：

```text
loger/models/pi3.py
loger/models/layers/attention.py
tools/run_v24_candidate_rollout.sh
tools/run_attention_cue_experiment.sh
```

当前实现：

```text
loger/models/pi3.py:
    _make_context_source_skip_bias(...)

当 context_source_skip_impl = compact_kv 时：
    frame_attention:
        source_keep_mask shape = [B * frame_num, tokens_per_frame]
    chunk/global attention:
        source_keep_mask shape = [B, frame_num * tokens_per_frame]

loger/models/layers/attention.py:
    _compact_kv_sdpa(...)
        idx = nonzero(source_keep_mask[b])
        kb = k[b:b+1, :, idx, :]
        vb = v[b:b+1, :, idx, :]
        scaled_dot_product_attention(qb, kb, vb)
```

包装脚本：

```text
tools/run_v24_candidate_rollout.sh:
    enable_compact_role_skip(...)
        CONTEXT_SOURCE_SKIP_ENABLE=1
        CONTEXT_SOURCE_SKIP_IMPL=compact_kv
        CONTEXT_SOURCE_SKIP_MASK=semantic_role_negative
        CONTEXT_SOURCE_SKIP_LAYER_MODE=early
```

单元验证：

```json
{
  "compact_vs_manual_max_abs_diff": 0.0,
  "compact_vs_dense_hard_bias_max_abs_diff": 2.384185791015625e-07
}
```

输出：

```text
results/kitti01_hmc_v2/current_code_semantic_vggt4d_audit/source_skip_compact_kv_unit.json
```

结论：

```text
当前 compact_kv 的 attention application 与 VGGT4D 的核心 source-removal 语义一致：
    保留 query rows，
    只删除 selected K/V source tokens，
    frame/global reshape 方向也与 VGGT4D 对应。
```

重要边界：

```text
如果某个 batch 的 source_keep_mask 全 False，当前 _compact_kv_sdpa 会 fallback 到全 source。
这是为了避免 SDPA 空 K/V 崩溃，但严格来说不同于 VGGT4D。
因此 empty_source_events > 0 的实验 row 应视为 invalid / blocker，而不能当作有效 source-skip 结果。
```

---

## 6. 当前 LoGeR 的 VGGT4D region-value 计算是否正确

读取文件：

```text
loger/pipeline/hybrid_memory_controller.py
```

结论：

```text
当前 LoGeR 的 source-skip attention 执行方式是 VGGT4D-like 且正确。

但当前 LoGeR 中 v4d.mean1 / v4d.var1 / v4d.mean2 / v4d.mean3 / v4d.var3
这些 region-value 计算不是 VGGT4D dynamic_mask.py 的忠实复刻。
它们只能算 VGGT4D-inspired approximation。
```

主要不一致：

| Component | VGGT4D | 当前 LoGeR |
|---|---|---|
| mean1 | q_ref @ q_src, layers 3..7 | qq high, layer g0 |
| var1 | q_ref @ q_src, layers 18..19, spatial std | qq var, layers g0_2 |
| mean2 | q_ref @ q_src, layers 17..21 | qq low, layers g2_6 |
| mean3 | k_ref @ k_src, layer 0 | kk high, layers g13_17 |
| var3 | q_ref @ k_src, layer 0, spatial std | qk var, layers g13_15 |
| mask generation | KMeans + adaptive multi-Otsu | not implemented |
| attention map detail | patch-token to all source-token maps | head/centroid-style aggregate patch score |

因此：

```text
如果实验文档说“模仿 VGGT4D source-skip attention”，这个说法对 compact_kv source removal 是成立的。

如果说“VGGT4D 的区域 token attention / dynamic map 计算已经正确复现”，这个说法不成立。
```

需要修复的方向：

```text
1. 导出或保留足够的 global q/k patch-token 张量，而不是只用当前 aggregate proxy。
2. 按 VGGT4D layer windows 精确计算：
       mean1: q_ref @ q_src, layers 3..7
       var1: q_ref @ q_src, layers 18..19
       mean2: q_ref @ q_src, layers 17..21
       mean3: k_ref @ k_src, layer 0
       var3: q_ref @ k_src, layer 0
3. 按 VGGT4D 组合式生成 dyn_map。
4. 如要完全复现，还需要 KMeans + adaptive multi-Otsu 生成 binary dynamic mask。
5. 再把 binary patch mask 接入当前已经验证过的 compact_kv source removal。
```

---

## 7. 最终审计结论

```text
GT 2D semantic:
    当前使用的是 valid sparse projected SemanticKITTI GT。
    GTSemanticProvider 与 landed projection cache exact match。
    统一 palette 可视化后，代表帧几何上合理。

颜色问题:
    初版颜色不一致是因为 VideoMasklet 和 GT 使用了两套 palette。
    已改为 VideoMasklet label 映射 SemanticKITTI color。
    sky 仍单独标注 sky(no-gt)，因为 projected GT 无 sky。

VGGT4D-style source-skip:
    compact_kv attention source removal 正确，和 VGGT4D 保留 query / 删除 K,V source 的语义一致。

VGGT4D region-value / dynamic map:
    当前不是忠实复刻，只是近似。
    后续若要声称“按 VGGT4D 计算区域 token attention”，必须补 exact dynamic-map 计算。
```


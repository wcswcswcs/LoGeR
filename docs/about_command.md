# 各模块运行指令

> 所有命令默认在项目根目录 (`LoGeR/`) 下执行，并使用 conda 环境 `loger`。
inference_dynamic_cue_extractor.py ，run_geometry_backbone_inference.py，run_pipeline_abc.py，run_pipeline_abc.py，run_video_masklet_front_end.py
---
## 0. original LoGeR inference
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n loger python demo_viser.py \
    --input data/examples/taylor.mp4 \
    --config ckpts/LoGeR/original_config.yaml \
    --model_name  ckpts/LoGeR/latest.pt \
    --start_frame 0 \
    --end_frame 50 \
    --stride 1 \
    --window_size 32 \
    --overlap_size 3 \
    --subsample 2 \
    --share

---

## 1. Stage A: LoGeR Geometry Backbone

**入口文件：** `run_geometry_backbone_inference.py`

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n loger python run_geometry_backbone_inference.py \
    --input data/examples/office \
    --config ckpts/LoGeR/original_config.yaml \
    --checkpoint ckpts/LoGeR/latest.pt \
    --window_size 32 \
    --overlap_size 3 \
    --output results/office_geometry.pt
```

**新增参数：**

| 参数 | 说明 |
|------|------|
| `--cache_ttt` | 导出 `WriteCacheOutput`（TTT 更新原语），用于 delayed write-back |

```bash
# 导出 TTT cache
python run_geometry_backbone_inference.py \
    --input data/examples/office \
    --config ckpts/LoGeR/original_config.yaml \
    --checkpoint ckpts/LoGeR/latest.pt \
    --cache_ttt
```

**常用参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--resolution W H` | auto | 目标分辨率，须为 14 的倍数 |
| `--window_size` | 32 | LoGeR sliding window 大小 |
| `--overlap_size` | 3 | window 间重叠帧数 |
| `--reset_every` | 0 | 每 N 个 window 重置 TTT state (0=不重置) |
| `--se3` | auto | 强制 SE(3) 对齐 |
| `--start_frame` | 0 | 起始帧索引 |
| `--end_frame` | -1 | 结束帧索引 (-1=全部) |
| `--stride` | 1 | 帧步长 |
| `--viser` | - | 启动 3D 可视化 |
| `--save_vis DIR` | - | 保存 2D 可视化图片 |

---

## 2. Stage B: Dynamic Cue Extractor

**入口文件：** `inference_dynamic_cue_extractor.py`

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n loger python inference_dynamic_cue_extractor.py \
    --input data/examples/taylor.mp4 \
    --config ckpts/LoGeR/original_config.yaml \
    --checkpoint ckpts/LoGeR/latest.pt \
    --output results/office_cues.pt \
    --output_video results/taylor_cue.mp4
```

**Stage B 特有参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--k_intra` | 3 | 帧内支持集大小 |
| `--sigma_pt` | 0.25 | 点位残差尺度 |
| `--tau_occ` | 0.05 | 遮挡深度阈值 |

---

## 3. Stage C: Video Masklet Front-end

**入口文件：** `run_video_masklet_front_end.py`

### Grounding DINO 检测器

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n loger python run_video_masklet_front_end.py \
    --input data/examples/office \
    --sam2_checkpoint /home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt \
    --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --detector gdino \
    --gdino_config Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino_checkpoint /mnt/data/users/chengshun.wang/pjs/GroundingDINO/weights/groundingdino_swint_ogc.pth \
    --output_video results/office_masklets.mp4
```

### YOLOE 检测器

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n loger python run_video_masklet_front_end.py \
    --input data/examples/office \
    --sam2_checkpoint /home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt \
    --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --detector yoloe \
    --yoloe_model yoloe-11l-seg.pt \
    --output_video results/office_yoloe.mp4
```

**Stage C 特有参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--detector` | gdino | 检测器类型 (gdino / yoloe) |
| `--ann_frame_idx` | 0 | 标注帧索引 |
| `--max_thing_objects` | 15 | thing 类最大跟踪数量 |
| `--box_threshold` | 0.30 | 检测框置信度阈值 |
| `--text_threshold` | 0.25 | 文本匹配阈值 |
| `--thing_prompts` | 内置列表 | 自定义 thing 类提示词 (逗号分隔) |
| `--stuff_prompts` | 内置列表 | 自定义 stuff 类提示词 (逗号分隔) |

---

## 4. 完整五阶段 Pipeline (A→B→C→D→E)

**入口文件：** `run_pipeline_abc.py`

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n loger python run_pipeline_abc.py \
    --input data/examples/office \
    --config ckpts/LoGeR/original_config.yaml \
    --checkpoint ckpts/LoGeR/latest.pt \
    --sam2_checkpoint /home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt \
    --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --detector gdino \
    --gdino_config Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino_checkpoint /mnt/data/users/chengshun.wang/pjs/GroundingDINO/weights/groundingdino_swint_ogc.pth \
    --chunk_size 32 \
    --output_video results/office_full_pipeline.mp4
```

### 单 chunk 模式（所有帧作为一个 chunk）

```bash
# chunk_size=0 表示不分 chunk
python run_pipeline_abc.py \
    --input data/examples/office \
    --config ckpts/LoGeR/original_config.yaml \
    --checkpoint ckpts/LoGeR/latest.pt \
    --sam2_checkpoint /home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt \
    --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --detector gdino \
    --gdino_config ... \
    --gdino_checkpoint ... \
    --chunk_size 0 \
    --output_video results/office_single_chunk.mp4
```

**Pipeline 特有参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chunk_size` | 0 | 每个 chunk 的帧数 (0=全部帧作为一个 chunk) |
| `--chunk_overlap` | 0 | chunk 间重叠帧数 |
| `--lambda_min` | 0.0 | Stage E block-level 写入增益下界 |
| `--lambda_max` | 1.0 | Stage E block-level 写入增益上界 |

---

## 5. 模型权重路径汇总

| 模型 | 路径 |
|------|------|
| LoGeR checkpoint | `ckpts/LoGeR/latest.pt` |
| LoGeR config | `ckpts/LoGeR/original_config.yaml` |
| LoGeR★ checkpoint | `ckpts/LoGeR_star/latest.pt` |
| LoGeR★ config | `ckpts/LoGeR_star/original_config.yaml` |
| SAM 2.1 Hiera Large | `/home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt` |
| SAM 2.1 config | `configs/sam2.1/sam2.1_hiera_l.yaml` |
| Grounding DINO config | `Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py` |
| Grounding DINO weights | `/mnt/data/users/chengshun.wang/pjs/GroundingDINO/weights/groundingdino_swint_ogc.pth` |
| YOLOE model | `yoloe-11l-seg.pt` (自动下载) |

---

## 6. Pipeline 五阶段数据流

```
X_m (chunk images) + W_m (old fast weights)
    │
    ├─ Stage A: LoGeR Geometry Backbone
    │     → GeometryOutput + WriteCacheOutput
    │
    ├─ Stage B: Dynamic Cue Extractor
    │     → CueOutput (C_stat / C_dyn / C_occ / C_unc / C_anchor)
    │
    ├─ Stage C: Video Masklet Front-end
    │     → MaskletOutput (masklets + semantic labels)
    │
    ├─ Stage D: Semantic Prior Generator
    │     → PriorOutput (A_mask, A_pix, A_tok)
    │
    └─ Stage E: TTT Write Controller
          → WriteResult (W_{m+1})
              ↓
          Next chunk reads W_{m+1}
```

# ACL2 v6 初步实验计划：LoGeR 复现保护、Stage C/D 缓存回接、语义先验驱动 TTT 写入

日期：2026-05-07  
对象：LoGeR / Pipeline v2 / HMC / Video Masklet Front-end / Semantic Prior Generator v2  
主开发集：KITTI Odometry sequence 01  
当前读控制主线：`C23 past + frame-attention pair/all`  
当前重点病灶：`[200,300)` worst segment  
本轮新增模块：Pipeline v1 的 Stage C `VideoMaskletFrontend` 与 Stage D `SemanticPriorGenerator`

---

## 0. 本轮计划的一句话结论

本轮不能直接把语义模块接上后就开始刷 ATE。第一件事必须是证明：

```text
引入 Stage C/D 代码、缓存、语义 prior 之后，在 no-op / pass-through 设置下仍能复现原 LoGeR / 当前 HMC v2 baseline。
```

然后再做三件事：

```text
1. 把 [200,300) 当作核心病灶做 causal audit；
2. 用有缓存的 Stage C/D 回接 Pipeline v2，避免每次重跑 video masklet frontend；
3. 只把语义信息用于 TTT 写入 prior，探索语义能否定义长期 memory 的写入资格和写入价值。
```

这轮的关键不是“语义一定能让结果变好”，而是要回答：

> 语义类别、masklet trust、几何 eligibility 和 attention read cue 能否共同定义一个比 `stage_d` 或 `stage_d*sqrt(1-D_g)` 更适合 TTT 长期写入的 prior？

---

## 1. 当前代码状态和必须修正的接入原则

### 1.1 Pipeline v2 已经有 Stage C/D 接口，但默认接上不等于 no-op

当前 `run_pipeline_abc_v2.py` 已经导入并调用了：

```text
VideoMaskletFrontend / MaskletOutput
SemanticPriorGenerator / PriorOutput
HybridMemoryController.build_control_prior(...)
```

Pipeline v2 的主流程是：

```text
Pass 1 probe geometry
Stage B cue
Stage C masklet
Stage D semantic prior
Pass 2 controlled forward
commit by hmc_commit_mode
```

但要特别注意：只要 `needs_prior=True`，`prior_gen.run(...)` 产出的 `PriorOutput.A_tok` 就会进入 HMC 的 `P_ttt_write`。如果 Stage C 为空，Stage D 仍可能根据 `G_write_geo` 生成几何 prior；如果 `hmc_write_score_source=stage_d` 没有 override，`P_ttt_write` 仍可能不是 `None`。因此：

```text
--hmc_write_score_source stage_d
```

并不自动等价于：

```text
完全不使用 Stage D prior。
```

所以本轮必须新增明确的 semantic no-op / pass-through 模式，用于复现保护。

### 1.2 本轮新增语义模块后的第一原则

任何新模块必须满足：

```text
模块可以被构建、缓存、读取、debug，但在 no-op 模式下不得改变：
    1. probe output；
    2. controlled output；
    3. committed HMC state；
    4. TTT fast weights；
    5. SWA history；
    6. 最终 trajectory。
```

换句话说，Stage C/D 的接入要先证明“不会改变原系统”，再证明“有用”。

### 1.3 Stage C 必须缓存

Video masklet frontend 可能调用 detector、SAM/SAM3.1、视频传播、refresh、stuff/thing merging。这部分比 Stage B/D 慢得多，且对每个候选 TTT 写入实验来说，masklet 结果在相同输入、相同配置下应当复用。

因此 Stage C 必须设计成：

```text
第一次：compute + save cache
后续：load cache，不加载 SAM，不重跑 detector，不重新传播 masklet
```

缓存不是可选优化，而是本轮能否高效实验的前置工程。

---

## 2. 本轮整体目标

### 2.0 新增硬目标：复现 `ACL2V5_SWKS3_03` 历史 best

在进入任何语义 Stage C/D 实验之前，本轮还必须复现之前最好的 v5 结果：

```text
ACL2V5_SWKS3_03
read cue = C23 past
read path = frame-attention pair/all
beta = 4.75
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
SWA keep scope = both_overlap
SWA source replacement = source / kv / alpha0.50
RESET_EVERY = 5
KITTI01 ATE / Rot = 36.4153 / 6.6186
FinalErr = 5.687
50f mean = 29.734
100f mean = 30.383
200f mean = 30.436
100f worst = 77.83
200f worst = 57.06
YawRMSE = 3.765
Sim3Scale = 31.298748
```

这个目标独立于 LoGeR native parity。它回答的是：

> 新模块接入后，不仅不能破坏 no-control LoGeR，也不能破坏当前最强 HMC/SWA active protocol。

因此本轮 Phase 0 被拆成两个 hard gates：

```text
Phase 0A: LoGeR / HMC no-op parity
Phase 0B: ACL2V5_SWKS3_03 historical best reproduction
```

只有两个 gate 都通过，才允许进入 Stage C/D semantic prior 实验。

本轮有四个整体目标。

### 2.1 目标 A：复现保护

引入 Stage C/D 和缓存后，先证明当前 baseline 能复现。

至少要复现两类 baseline：

```text
A. LoGeR / HMC native 或当前 no-control parity；
B. 当前 v5 read 主线：C23 past + frame pair/all + probe_ttt_write。
```

复现保护分为两层：

```text
hard no-op parity:
    Stage C/D 代码路径存在，但 HMC 完全忽略语义 prior。

semantic pass-through parity:
    Stage D 输出 A_tok = 1, B_chunk_geo = 1，验证 write controller 的 pass-through 是否近似或完全等价。
```

### 2.2 目标 B：`[200,300)` causal audit

当前 `<30m` 的最大瓶颈集中在 `[200,300)`。本轮先把它作为核心病灶，判断它和 TTT / SWA / read cue / semantic distribution 的关系。

目标不是马上修掉它，而是回答：

```text
[200,300) 是否有特殊的语义组成？
是否 sky/tree/vegetation/road/building/car/person 的写入比例异常？
是否某些 chunk 的 TTT update norm / memory diff / write prior 出现异常？
是否 Stage C masklet 覆盖或语义误分集中发生在这个段？
```

### 2.3 目标 C：带缓存地回接 Stage C/D

把 Pipeline v1 的 Stage C/D 以 Pipeline v2/HMC 兼容方式重新接入，并产生可审计的 semantic prior 输出：

```text
MaskletOutput
PriorOutput
A_geo_patch
A_sem_patch
A_semgeo_patch
SemGroup_patch
SemLabel_patch
R_mask_patch
B_chunk_geo
```

这些输出必须能够按 chunk、frame、semantic group、patch token 聚合，供 TTT 写入分析使用。

### 2.4 目标 D：探索语义信息是否适合做 TTT 写入 prior

本轮只把语义用于 TTT 写入，不改变当前最强 frame-attention read cue。固定 read 主线后，探索：

```text
低价值 stuff：sky / tree / vegetation / water / reflection 是否应少写？
结构区域：road / building / wall / sidewalk 是否应多写？
movable thing：car / person / rider 是否应少写或短期写？
mask trust：低质量 masklet 是否应该退回 geometry-only prior？
semantic prior 是否只改善 Rot/Yaw/FinalErr，还是能真正降低 [200,300) ATE？
```

---

## 3. 本轮核心假设

### H0：Stage C/D 接入后可以 no-op 复现 LoGeR / HMC baseline

#### 假设

如果 Stage C/D 只是被构建、缓存或加载，但 HMC 不消费它们，输出应与原 pipeline 完全一致。

#### 实验

固定当前 baseline，跑以下配置：

```text
R0-native:
    stage_c_mode = none
    stage_d = none
    semantic_prior_mode = disabled

R1-stageC-cache-only:
    stage_c_mode = reference
    stage_c_cache_mode = readwrite
    stage_d = disabled
    hmc_ignore_semantic_prior = 1

R2-stageC-load-cache-only:
    stage_c_mode = reference
    stage_c_cache_mode = read
    stage_d = disabled
    hmc_ignore_semantic_prior = 1

R3-stageD-noop:
    stage_c_mode = reference
    stage_d_mode = noop
    PriorOutput.A_tok = ones
    B_chunk_geo = 1
    hmc_semantic_prior_mode = ignore

R4-stageD-pass-through:
    stage_c_mode = reference
    stage_d_mode = pass_through
    HMC consumes A_tok = ones
```

#### 记录指标

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
ATE_50 / ATE_100 / ATE_200
YawRMSE
Sim3Scale
pose_max_abs_diff_vs_R0
pointmap_max_abs_diff_vs_R0
conf_max_abs_diff_vs_R0
hmc_state_hash_before/after
TTT_state_hash_before/after
SWA_history_hash_before/after
cache_hit
cache_key
stage_c_runtime
stage_d_runtime
```

#### 成立标准

H0 hard no-op 通过标准：

```text
R1/R2/R3 vs R0:
    pose_max_abs_diff = 0 或 <= 1e-7
    trajectory ATE diff <= 1e-4m
    HMC committed state hash identical
    no extra TTT update path consumed semantic prior
```

H0 pass-through 通过标准：

```text
R4 vs R0:
    ATE diff <= 0.01m
    Rot diff <= 0.01deg
    TTT update norm 与 R0 近似一致
```

如果 R4 不通过，不代表 Stage C/D 失败，而说明 `A_tok=ones` 在当前 TTT write controller 中不是严格 no-op。后续必须使用 `hmc_ignore_semantic_prior` 做复现保护。

---

### H0B：引入新模块后必须复现 `ACL2V5_SWKS3_03`

#### 假设

如果 Stage C/D cache、semantic prior generator、masklet frontend 的代码接入是安全的，那么在语义模块处于 disabled / ignored 状态时，当前历史 best `ACL2V5_SWKS3_03` 应该可以复现。

这个 gate 比 LoGeR native parity 更严格，因为它覆盖了：

```text
1. C23 past read cue；
2. frame-attention pair/all read intervention；
3. probe_ttt_write commit；
4. TTT write score stage_d_x_dg_inv_sqrt；
5. SWA both_overlap keep-scope；
6. SWA source-side K/V replacement alpha0.50；
7. RESET_EVERY=5；
8. 当前 v5 active HMC/SWA 协议。
```

#### 实验

先跑三个配置。

```text
B0-SWKS3-reference-rerun:
    完全复现 ACL2V5_SWKS3_03，不启用 Stage C/D。

B1-SWKS3-stageC-cache-ignored:
    启用 Stage C cache compute/load 代码路径，但 hmc_ignore_semantic_prior=1，语义 prior 不进入 HMC。

B2-SWKS3-stageD-noop-ignored:
    启用 Stage C cache + Stage D noop，HMC 仍忽略 PriorOutput。

B3-SWKS3-pass-through-consumed:
    Stage D 输出 A_tok=ones，HMC 消费 pass-through prior。该项不是硬 parity gate，而是检查 write controller pass-through 是否安全。
```

#### 必须记录的指标

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
50f mean
100f mean
200f mean
100f worst
200f worst
YawRMSE
Sim3Scale
pose_max_abs_diff_vs_B0
hmc_state_hash_before/after
TTT_state_hash_before/after
SWA_history_hash_before/after
swa_keep_scope_effective
swa_source_replace_mode_effective
swa_source_replace_target_effective
swa_source_replace_alpha_effective
semantic_prior_present
semantic_prior_consumed
cache_hit_rate
```

#### 成立标准

`B0` 必须复现历史 best：

```text
|ATE_B0 - 36.4153| <= 0.03m
|Rot_B0 - 6.6186| <= 0.03deg
100f worst <= 78.00
200f worst <= 57.30
```

`B1/B2` 必须不改变 `B0`：

```text
|ATE_B1 - ATE_B0| <= 0.01m
|ATE_B2 - ATE_B0| <= 0.01m
|Rot_B1 - Rot_B0| <= 0.01deg
|Rot_B2 - Rot_B0| <= 0.01deg
HMC/SWA/TTT committed state hash 无异常变化
semantic_prior_consumed = false
```

`B3` 是 diagnostic：

```text
如果 B3 与 B0 完全一致或接近一致：
    A_tok=ones pass-through 安全，可用于后续调试。

如果 B3 与 B0 不一致：
    pass-through 不是 no-op；后续所有复现保护必须使用 hmc_ignore_semantic_prior=1，而不能用 A_tok=ones 假装 no-op。
```

#### 失败处理

如果 `B0` 不能复现 `36.4153 / 6.6186`，停止语义模块实验，先排查：

```text
1. run script 是否真的使用 pair/all；
2. beta 是否为 4.75；
3. write score 是否为 stage_d_x_dg_inv_sqrt；
4. TTT native mix / write alpha 是否与原 SWKS3_03 一致；
5. SWA keep_scope 是否为 both_overlap；
6. SWA source replace mode 是否为 source；
7. target 是否为 kv；
8. alpha 是否为 0.50；
9. RESET_EVERY 是否仍为 5；
10. Stage C/D 是否无意中改变了 image loading、chunk scheduling、state commit 或 cache path。
```

如果 `B1/B2` 不能复现 `B0`，说明 Stage C/D 接入虽然没有被显式消费，但已经改变了运行状态，常见原因包括：

```text
1. 额外加载大模型导致显存/CPU 内存行为改变，间接影响并发稳定性；
2. Stage C image loading 改变了输入 tensor 顺序或 resize；
3. Stage C/D 构建改变了 global random seed；
4. prior_output 非空后 HMC 某处默认消费；
5. cache load / save 中途改变 chunk loop timing 或 state swap。
```

---

### H1：Stage C cache 可以完全复用 masklets，并显著减少时间

#### 假设

相同输入、相同 Stage C 配置、相同模型权重下，`MaskletOutput` 应可通过 cache 完全复用；第二次运行不应重新初始化 SAM / detector / tracker。

#### 建议新增 CLI

```text
--stage_c_cache_dir PATH
--stage_c_cache_mode off|read|write|readwrite|refresh
--stage_c_cache_key_mode strict|relaxed
--stage_c_cache_require_hit 0|1
--stage_c_cache_validate 0|1
--stage_d_cache_dir PATH
--stage_d_cache_mode off|read|write|readwrite|refresh
```

#### Cache key 必须包含

```text
input sequence id
absolute or canonical image paths
frame_start / frame_end / stride
image file size + mtime 或 image sha1
image resolution used by Stage C
chunk_size / chunk_overlap
stage_c_mode
sam_backend
tracker_backend
sam checkpoint path + file size/mtime
sam2 model cfg
sam31 checkpoint path + file size/mtime
detector type
detector model path + file size/mtime
thing_prompts
stuff_prompts
all sam31_* prompt/budget/stride args
box_threshold / text_threshold
max_thing_objects
code_version_hash
cache_schema_version
```

#### 每个 chunk 的缓存目录

```text
stage_c_cache/
  cache_index.jsonl
  chunk_000/
    manifest.json
    masklet.pt
    debug.json
    preview_grid.png
    semantic_coverage.png
    cache_key.txt
  chunk_001/
    ...
```

#### `masklet.pt` 推荐字段

不要直接依赖 dataclass pickle 的长期兼容性，推荐保存版本化 dict：

```text
schema_version
M_mask
V_mask
B_mask
Q_mask
L_sem
G_sem
W_sem
A_ratio
num_masklets
num_frames
frame_height
frame_width
source_type
birth_frame
seed_global_track_idx
debug
```

所有 tensor 保存为 CPU tensor。第一版可以直接 `torch.save` bool/float tensor；后续如磁盘过大，再把 `M_mask` 做 `np.packbits` 压缩。

#### 成立标准

```text
第一次 readwrite：cache_miss 后成功保存所有 chunks。
第二次 read：cache_hit_rate = 100%。
第二次 Stage C runtime 降低 >= 80%。
cache load 后 MaskletOutput 与第一次 compute 的 tensor 完全一致。
cache read 模式下不应构建 SAM video predictor / detector。
```

如果 `--stage_c_cache_require_hit=1` 且某个 chunk cache miss，应直接报错停止，避免无意中重跑昂贵 Stage C。

---

### H2：`[200,300)` 的失败可以被 chunk-level memory / semantic / read-cue 统计解释

#### 假设

`[200,300)` 不是全局均匀误差的一部分，而是某些 chunk 的 memory state、read cue、SWA source、TTT update 或 semantic write distribution 发生异常。

#### 实验

固定以下候选做 audit：

```text
B0: C23 pair/all reference
B1: current TTT-only best
B2: current SWA-active tiny best
B3: freeze chunks 4/5/6 diagnostic
B4: semantic no-op cached run
```

对每个 run 输出：

```text
segment_50f.csv
segment_100f.csv
segment_200f.csv
per_chunk_memory.csv
per_chunk_cue.csv
per_chunk_semantic.csv
per_chunk_write.csv
```

每个 chunk 需要标注它与 `[200,300)` 的 overlap：

$$
	ext{overlap}_{c,[200,300)}=rac{|[s_c,e_c)\cap[200,300)|}{e_c-s_c}
$$

#### 必须记录的语义指标

按 semantic group 聚合：

```text
STRUCTURE_ANCHOR mass
STATIC_THING mass
MOVABLE_THING mass
LOW_VALUE_STUFF mass
UNCERTAIN_REGION mass
sky mass
tree / vegetation mass
road mass
building mass
car/person mass
mask trust mean
mask coverage
uncovered ratio
semantic entropy
```

按 TTT 写入聚合：

```text
write_score_mean_by_sem_group
write_score_q90_by_sem_group
TTT_update_norm_by_sem_group
branch0_update_norm_by_sem_group
branch1_update_norm_by_sem_group
branch2_update_norm_by_sem_group
selected_sparse_mass_by_sem_group
```

#### 可视化

```text
[200,300)_causal_audit/
  trajectory_xy_marked_200_300.png
  frame_error_curve_marked.png
  segment_100f_bar.png
  per_chunk_ttt_norm_vs_error.png
  per_chunk_semantic_mass_vs_error.png
  semantic_group_stack_by_chunk.png
  D_g_vs_semantic_overlay_grid.md
  worst_chunk_gallery/
      RGB
      D_g
      old_dyn
      explicit_dyn
      semantic group map
      A_sem
      A_write
      TTT write heatmap
```

#### 成立标准

H2 成立不要求立刻改善 ATE，但必须满足至少一项：

```text
1. [200,300) overlap chunks 的某个 memory 指标显著异常；
2. [200,300) overlap chunks 的某类 semantic mass 或 write mass 显著异常；
3. freeze diagnostic 改变 [200,300) 的同时，对应的 TTT/SWA/semantic统计出现可解释变化；
4. 能提出一个由数据支持的自动触发 risk score，而不是手工指定 chunk 4/5/6。
```

如果没有任何统计能区分 `[200,300)`，则不能继续做 chunk-trigger TTT 策略，只能转向 read cue / evaluation / global alignment 层面的诊断。

---

### H3：语义低价值区域不是天然“不写”，需要验证 soft veto / hard veto 的区别

#### 假设

`sky/tree/vegetation` 可能污染 TTT 长期 memory，但也可能提供 horizon、scale、orientation 或连续性信息。因此不能直接假设 hard veto 有益。

#### 语义 score 定义

定义 semantic write value：

$$
V_{sem}(i)=
\begin{cases}
1.00, & \text{road/building/wall/sidewalk/structure} \\
0.70, & \text{static thing/background} \\
0.40, & \text{tree/vegetation/sky/water/reflection} \\
0.10, & \text{person/car/rider/movable thing} \\
0.40, & \text{uncertain}
\end{cases}
$$

定义低价值 stuff veto：

$$
G_{lowstuff}(i)=1-\rho_{low}\mathbf{1}[sem(i)\in\{sky,tree,vegetation,water,reflection\}]
$$

定义 sky/tree 专项 veto：

$$
G_{skytree}(i)=1-\rho_{sky}\mathbf{1}[sem(i)\in\{sky,tree,vegetation\}]
$$

#### 实验设计

固定 read：

```text
read cue = C23 past
read path = frame pair/all
beta = 4.75
commit = probe_ttt_write
```

候选：

```text
S0 reference:
    S_write = stage_d * sqrt(1-D_g)

S1 semantic value only:
    S_write = V_sem

S2 stage_d x semantic:
    S_write = stage_d * V_sem

S3 current best x semantic:
    S_write = stage_d * sqrt(1-D_g) * V_sem

S4 sky/tree soft veto:
    S_write = stage_d * sqrt(1-D_g) * G_skytree, rho_sky=0.25

S5 sky/tree medium veto:
    rho_sky=0.50

S6 sky/tree hard-ish veto:
    rho_sky=0.75

S7 low-value stuff soft veto:
    rho_low=0.25

S8 low-value stuff medium veto:
    rho_low=0.50
```

#### 判断标准

如果 sky/tree 抑制：

```text
只改善 Rot/Yaw/FinalErr，但 [200,300) 不动：
    判定为 orientation regularizer，不是 ATE 主解。

使 [200,300) 100f ATE 下降 >= 10m，且 global ATE 不回退：
    判定 sky/tree 是长期 TTT memory 污染源候选。

hard veto 伤 ATE 或 Sim3Scale 异常：
    判定 sky/tree/horizon 仍提供 scale/orientation continuity，后续只允许 soft veto 或 layer-specific veto。
```

强通过标准：

```text
global ATE <= current reference - 0.30m
或 [200,300) ATE 降低 >= 10m 且其它 100f segment 最大回退 <= 3m
```

弱通过标准：

```text
global ATE 改善 >= 0.05m
且 Rot/Yaw/FinalErr 至少两项改善
且 [200,300) 不恶化
```

---

### H4：结构语义 boost 可能比低价值 veto 更适合 TTT 写入

#### 假设

TTT 长期 memory 需要的是稳定的 global anchor。与其少写 sky/tree，不如多写 road/building/wall/sidewalk 等结构区域。

#### score 定义

$$
G_{struct}(i)=1+\rho_{struct}\mathbf{1}[sem(i)\in STRUCTURE]
$$

候选：

```text
B1 structure soft boost:
    S_write = stage_d * sqrt(1-D_g) * G_struct, rho_struct=0.25

B2 structure medium boost:
    rho_struct=0.50

B3 structure + lowstuff soft veto:
    S_write = stage_d * sqrt(1-D_g) * G_struct * G_lowstuff

B4 road/building only boost:
    only road/building/wall/sidewalk boost, not all structure
```

#### 记录指标

```text
structure_write_mass
road_write_mass
building_write_mass
sky_write_mass
tree_write_mass
structure_update_norm_branch0
structure_update_norm_branch2
Sim3Scale
YawRMSE
[200,300) ATE
```

#### 成立标准

```text
global ATE 改善 >= 0.10m
或 [200,300) 改善 >= 10m
且 Sim3Scale 不异常
且 Rot/Yaw 不恶化
```

如果 structure boost 只改善 FinalErr，不改善 ATE，则保留为 endpoint regularizer，不作为主 TTT write prior。

---

### H5：movable thing veto 应只做温和写入降权，不应简单 hard mask

#### 假设

car/person/rider 等 movable thing 可能污染长期 memory，但它们也可能携带短时运动上下文。TTT 写入应区分：

```text
current chunk adaptation 可以使用；
commit 到未来 memory 时应降权。
```

#### 实验

```text
M1 movable soft veto:
    S_write = stage_d * sqrt(1-D_g) * G_movable, rho=0.25

M2 movable medium veto:
    rho=0.50

M3 movable hard-ish veto:
    rho=0.75

M4 commit-only movable veto:
    current controlled output 不变，只在 probe_ttt_write commit replay prior 上降权 movable tokens

M5 late-layer movable veto:
    只对 late TTT layers 使用 movable veto
```

#### 成立标准

```text
如果 soft veto 改善 Rot/Yaw 但 ATE 不动：
    movable semantic prior 只是姿态 regularizer。

如果 commit-only 比 full replay veto 更好：
    支持“动态信息对当前 chunk 有用，但不应长期传播”的假设。

如果 late-layer veto 比 all-layer veto 更好：
    说明语义污染主要发生在长期传播层，而不是所有 TTT 层。
```

---

### H6：mask trust 应作为 routing，而不是直接打分

#### 假设

低质量 masklet 不意味着区域低价值，而意味着不应信语义分支，应退回 geometry/read cue。

#### 公式

语义 branch：

$$
A_{sem}(i)=V_{sem}(i)
$$

mask trust：

$$
R_{mask}(i)=V_{mask}(i)Q_{mask}(i)
$$

最终 semantic prior：

$$
A_{write}(i)=R_{mask}(i)A_{sem}(i)+(1-R_{mask}(i))A_{geo}(i)
$$

其中 $A_{geo}$ 可以是：

```text
stage_d
sqrt(1-D_g)
stage_d * sqrt(1-D_g)
G_write_geo
```

#### 实验

```text
T1 trust routing with stage_d fallback
T2 trust routing with stage_d*sqrt(1-D_g) fallback
T3 no trust, direct semantic value
T4 trust threshold high only, Q_mask > 0.7 使用语义，否则 fallback
```

#### 成立标准

```text
trust routing > direct semantic value
且低 trust 区域不产生明显 ATE regression
```

如果 direct semantic 和 trust routing 都失败，说明 Stage C 语义质量暂时不足以进入 TTT 写入主线，应先做 masklet quality 改进或只用于分析。

---

## 4. 代码修改建议

### 4.1 新增 Stage C cache wrapper

当前 `_run_stage_c_lazy(...)` 会在没有 frontend instance 时构建 `VideoMaskletFrontend`，然后直接 `run(chunk_images, chunk_index=ci)`。建议改成：

```text
_run_stage_c_cached(
    build_kwargs,
    frontend_kwargs,
    chunk_images,
    chunk_idx,
    cache_dir,
    cache_mode,
    cache_key_mode,
    require_hit,
)
```

逻辑：

```text
1. build cache key；
2. 如果 cache_mode 包含 read 且命中：
       load MaskletOutput；
       return；
3. 如果 require_hit 且 miss：
       raise RuntimeError；
4. 否则调用 _run_stage_c_lazy；
5. 如果 cache_mode 包含 write：
       atomic save；
6. return MaskletOutput。
```

atomic save：

```text
write to chunk_xxx.tmp/
fsync manifest if needed
rename tmp -> final
```

### 4.2 新增 semantic no-op / pass-through mode

建议 CLI：

```text
--semantic_prior_mode disabled|noop|pass_through|spg_v2
--hmc_ignore_semantic_prior 0|1
--hmc_write_score_source semantic_stage_d|semantic_value|stage_d_x_sem|stage_d_x_sem_dg
```

模式定义：

```text
disabled:
    不构建 prior_gen，HMC P_ttt_write=None。

noop:
    构建 Stage C/D，但 HMC 忽略 PriorOutput。

pass_through:
    PriorOutput.A_tok=ones, B_chunk_geo=1，HMC 正常消费，用于测试 TTT controller pass-through。

spg_v2:
    正常运行 SemanticPriorGenerator v2。
```

### 4.3 扩展 PriorOutput debug 字段

建议新增：

```text
A_geo_patch_flat
A_sem_patch_flat
A_semgeo_patch_flat
R_mask_patch_flat
SemGroup_patch_flat
SemLabel_patch_flat
MaskId_patch_flat
semantic_group_hist
semantic_write_hist
semantic_trust_hist
```

最小实现可以不改变核心算法，只把现有中间量落盘。没有这些字段，后续无法判断 sky/tree 是否真的被压低，也无法解释 TTT 写入变化。

### 4.4 HMC write score 新增 semantic source

建议新增：

```text
semantic_prior
stage_d_x_sem
stage_d_x_sem_sqrt
stage_d_x_dg_sem
stage_d_x_dg_sem_sqrt
sem_skytree_veto
sem_lowstuff_veto
sem_structure_boost
sem_movable_veto
sem_trust_routing
```

每个 write source 都必须在 debug 中输出：

```text
hmc_write_score_source
semantic_write_group_mean
semantic_write_group_q90
corr_score_Dg
corr_score_exp_dyn
corr_score_old_dyn
corr_score_sem_lowstuff
corr_score_sem_structure
```

---

## 5. 执行顺序

### Phase 0：复现保护与 cache correctness

Phase 0 现在分成两部分：

```text
Phase 0A: LoGeR / HMC no-op parity
Phase 0B: ACL2V5_SWKS3_03 historical best reproduction
```

先跑短 smoke：

```text
END_FRAME=128
R0/R1/R2/R3/R4
```

通过后跑 KITTI01 full：

```text
R0 baseline
R1 stageC cache compute but ignored
R2 stageC cache load but ignored
R3 stageD noop ignored
R4 pass-through consumed
```

停止条件：

```text
如果 R1/R2/R3 不能复现 baseline，停止所有语义实验，先修 no-op path。
如果 cache load 不一致，停止所有 full semantic sweep，先修 cache serialization。
```

### Phase 1：`[200,300)` causal audit

使用当前 best / references 做 audit，不先改模型：

```text
C23 pair/all reference
TTT-only reference
SWA-active reference
freeze diagnostic
semantic cached no-op
```

输出 segment/memory/semantic dashboard。

### Phase 2：Stage C/D passive semantic audit

只生成 masklet + semantic prior，不改变 HMC 写入：

```text
semantic_prior_mode = spg_v2
hmc_ignore_semantic_prior = 1
```

目的：判断语义覆盖是否可用。

通过标准：

```text
mask coverage >= 0.70 或至少 structure+lowstuff 覆盖充分；
sky/tree/road/building/car/person 有合理分布；
[200,300) 的 semantic maps 可解释；
无明显全帧错误 masklet 主导。
```

### Phase 3：语义 TTT write 小矩阵

固定 read path，只改 TTT write score：

```text
read = C23 past pair/all beta 4.75
commit = probe_ttt_write
stage_c_cache_mode = read
semantic_prior_mode = spg_v2
```

第一批只跑 8-10 个 full candidates，不做大矩阵。

候选优先级：

```text
1. stage_d * sqrt(1-D_g) * semantic value
2. sky/tree soft veto
3. sky/tree medium veto
4. low-value stuff soft veto
5. structure soft boost
6. structure + lowstuff soft veto
7. movable soft veto
8. trust routing fallback
```

### Phase 4：通过候选的 branch/layer/commit 归因

只有 Phase 3 通过 weak gate 的候选才进入：

```text
branch0 only
branch0+2
late layers only
middle/late layers only
commit-only semantic veto
sparse top75 semantic static write
```

### Phase 5：cross-sequence sanity

只让 1-2 个候选进入：

```text
KITTI00 full
KITTI02 full
KITTI05 full
```

通过标准：

```text
KITTI01 改善不是以 KITTI02/05 明显退化换来的；
00/02/05 平均 ATE 或 Rot 不差于 current C23 pair/all reference；
没有任何序列 ATE regression > 3%。
```

---

## 6. 必须记录的指标

### 6.1 复现与 cache 指标

```text
cache_hit_rate
cache_miss_reason
stage_c_compute_runtime
stage_c_load_runtime
stage_d_runtime
cache_size_mb
cache_key
cache_schema_version
masklet_tensor_hash
prior_tensor_hash
pose_max_abs_diff
pointmap_max_abs_diff
hmc_state_hash_diff
```

### 6.2 标准轨迹指标

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
ATE_50_mean / ATE_50_worst
ATE_100_mean / ATE_100_worst
ATE_200_mean / ATE_200_worst
YawRMSE
Sim3Scale
```

### 6.3 病灶段指标

```text
ATE_[200,300)
Rot_[200,300)
Yaw_[200,300)
ScaleProxy_[200,300)
SegmentRank_[200,300)
Delta_[200,300)_vs_reference
```

### 6.4 masklet / semantic 指标

```text
num_masklets
coverage_total
coverage_by_group
coverage_by_label
uncovered_ratio
mean_Q_mask
Q_mask_p10/p50/p90
masklet_area_mean
masklet_area_by_group
semantic_group_hist
semantic_label_hist
```

### 6.5 semantic write 指标

```text
write_score_mean_by_group
write_score_q90_by_group
write_score_mass_gt_0.5_by_group
selected_sparse_mass_by_group
TTT_update_norm_by_group
TTT_update_cosine_by_group
branch0_update_norm_by_group
branch1_update_norm_by_group
branch2_update_norm_by_group
```

### 6.6 相关性指标

```text
Corr(write_score, D_g)
Corr(write_score, explicit_dyn)
Corr(write_score, old_dyn)
Corr(write_score, C_anchor)
Corr(write_score, LOW_VALUE_STUFF mask)
Corr(write_score, STRUCTURE mask)
Corr(write_score, MOVABLE mask)
Corr(group_mass, segment_ATE_delta)
```

---

## 7. 必须可视化

### 7.1 复现 dashboard

```text
repro_dashboard.md
  baseline vs stageC-cache-only trajectory overlay
  baseline vs stageD-noop trajectory overlay
  pose diff curve
  HMC state hash table
  cache hit/miss table
```

### 7.2 Stage C cache dashboard

```text
stage_c_cache_dashboard.md
  per chunk runtime compute vs load
  cache hit rate
  cache size
  masklet count by chunk
  semantic group coverage by chunk
```

### 7.3 `[200,300)` semantic causal dashboard

```text
segment_200_300_dashboard.md
  trajectory with [200,300) highlighted
  per-frame error curve
  semantic group stacked area by chunk
  TTT update norm by chunk
  write score by semantic group
  worst-frame gallery
```

### 7.4 semantic prior gallery

每个关键 chunk 输出：

```text
RGB
D_g
explicit_dyn
old_dyn
semantic group map
sky/tree mask
road/building mask
car/person mask
A_geo
A_sem
A_final_write
write_score_delta_vs_reference
```

### 7.5 TTT write attribution heatmap

```text
layer x branch update norm
layer x branch update cosine to reference
semantic group x branch update norm
semantic group x layer update norm
```

---

## 8. 第一批建议运行矩阵

### 8.1 Phase 0 smoke

```text
S0_00 baseline no Stage C/D
S0_01 Stage C compute cache, ignored
S0_02 Stage C load cache, ignored
S0_03 Stage D noop, ignored
S0_04 Stage D pass-through consumed
```

### 8.2 Phase 1 audit

```text
A1_00 current C23 pair/all reference
A1_01 TTT-only reference
A1_02 SWA-active reference
A1_03 freeze diagnostic reference
A1_04 semantic cached no-op
```

### 8.3 Phase 3 semantic TTT write first matrix

```text
SEM_00 reference: stage_d*sqrt(1-D_g)
SEM_01 stage_d*sqrt(1-D_g)*V_sem
SEM_02 skytree veto rho=0.25
SEM_03 skytree veto rho=0.50
SEM_04 lowstuff veto rho=0.25
SEM_05 structure boost rho=0.25
SEM_06 structure boost rho=0.50
SEM_07 structure boost + lowstuff veto
SEM_08 movable veto rho=0.25
SEM_09 trust routing fallback
```

### 8.4 Phase 4 attribution only for winners

```text
WIN_01 branch0 only
WIN_02 branch0+2
WIN_03 late TTT layers only
WIN_04 commit-only semantic prior
WIN_05 semantic sparse top75
```

---

## 9. 最重要的决策规则

### 9.1 语义 prior 成功标准

强成功：

```text
KITTI01 global ATE <= current reference - 0.30m
或 [200,300) ATE 下降 >= 10m 且其它段不崩
```

弱成功：

```text
KITTI01 global ATE 改善 >= 0.05m
且 Rot/Yaw/FinalErr 至少两项改善
且 [200,300) 不恶化
```

最终成功：

```text
KITTI01 ATE < 30m
```

### 9.2 何时停止某条语义假设

```text
如果 hard sky/tree veto 伤 ATE：
    停止 hard veto，只保留 soft/layer-specific。

如果 structure boost 只改善 FinalErr：
    保留为 endpoint regularizer，不当主线。

如果 movable veto 只改善 Rot：
    改成 commit-only / late-layer，不继续 all-layer hard veto。

如果 trust routing 不如 direct semantic：
    检查 Q_mask 是否无判别力；不继续调 trust 参数。

如果所有 semantic write 都不动 [200,300)：
    语义不是当前主病灶；返回 TTT objective / read cue / evaluation diagnosis。
```

---

## 10. 当前我对 sky/tree 示例的预期

我不会先假设 sky/tree 一定应该不写。更合理的预期是：

```text
sky/tree hard suppression 可能改善 Rot/Yaw，但也可能伤 scale/ATE；
sky/tree soft suppression 如果有效，应该表现为 [200,300) 或 FinalErr 改善；
road/building/structure boost 可能比 sky/tree veto 更适合 TTT；
语义价值必须经过 mask trust routing，否则 bad masklet 会制造新的 false positive。
```

因此 sky/tree 实验的解释必须分三类：

```text
1. ATE 和 [200,300) 明显改善：
       sky/tree 是长期 TTT 污染源候选。

2. 只改善 Rot/Yaw/FinalErr：
       sky/tree prior 是 orientation regularizer，不是主 ATE 解法。

3. ATE 变差：
       sky/tree/horizon 仍提供 scale 或 continuity，不能简单不写。
```

---

## 11. 本轮最终产物

本轮结束后应产出：

```text
1. 可复现的 Stage C cache 机制；
2. Stage C/D no-op 复现报告；
3. [200,300) causal audit dashboard；
4. semantic prior passive dashboard；
5. 第一批 semantic TTT write 结果表；
6. 每个 semantic group 的 write attribution；
7. 对 sky/tree、structure、movable、trust routing 四类假设的明确结论。
```

如果这些完成后语义 prior 没有改善 ATE，也仍然是有效结果：它会说明语义不适合作当前 TTT 写入主方向，或者当前 masklet/semantic quality 不足，需要改前端而不是继续调 write controller。

---

## 12. 执行记录

### 2026-05-07 Phase 0 启动

执行顺序按本计划的 hard gate 开始：

1. 先跑 `Phase 0B / B0-SWKS3-reference-rerun`，不启用 Stage C/D，复现历史 best `ACL2V5_SWKS3_03`。
2. 同步检查当前代码是否已经具备 Stage C cache、semantic no-op / pass-through、HMC ignore prior 等接口；若缺失，先补工程再跑 `B1-B3`。
3. 所有结果统一记入本节。

#### Phase 0B：SWKS3 historical best reproduction

固定目标：

```text
reference = ACL2V5_SWKS3_03
expected KITTI01 ATE / Rot = 36.4153 / 6.6186
gate = |ATE - 36.4153| <= 0.03m, |Rot - 6.6186| <= 0.03deg
```

| Run | 状态 | 关键配置 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| `ACL2V6_B0_SWKS3_reference_rerun` | done | C23 past + pair/all beta 4.75; `stage_d_x_dg_inv_sqrt`; `TTT_WRITE_NATIVE_MIX_SCALES=1.10,1.00,1.00`; `SWA_WRITE_KEEP_SCOPE=both_overlap`; source/KV replace alpha 0.50; `RESET_EVERY=5`; Stage C/D off | `36.4161` | `6.6128` | `92.4452` | `0.0082` | 通过复现 gate；与 `ACL2V5_SWKS3_03 = 36.4153 / 6.6186` 基本一致 |

Trajectory diagnostics：

输出目录：

```text
results/kitti01_hmc_v2/acl2_v6_phase0_repro/trajectory_diagnostics_b0/
```

| Run | ATE RMSE | Final error | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ACL2V6_B0_SWKS3_reference_rerun` | `36.4161` | `5.798` | `29.718 / 78.272` | `30.368 / 77.831` | `30.385 / 57.101` | `3.765` | `31.238916` |

B0 结论：

1. v6 当前代码在 Stage C/D disabled / ignored 状态下成功复现 v5 历史 best `ACL2V5_SWKS3_03`。
2. ATE 差值 `+0.0008m`，Rot 差值 `-0.0058deg`，均小于计划 gate 的 `0.03` 阈值。
3. 100f worst `77.831`、200f worst `57.101` 与 v5 记录 `77.83 / 57.06` 对齐；主灾区仍集中在 `[200,300)` / `[200,400)`。
4. 这说明 v6 的 Stage C cache / semantic prior 接线没有破坏当前最强 read+TTT+SWA 主线，可以继续进入 full no-op parity / cached Stage C 检查。

#### Phase 0 工程接线记录

本轮补齐 v6 计划所需的最小 no-op / cache 接口：

- `run_pipeline_abc_v2.py`
  - 新增 `--stage_c_cache_dir`、`--stage_c_cache_mode`、`--stage_c_cache_require_hit`、`--stage_c_cache_validate`。
  - 新增 `--semantic_prior_mode disabled/noop/pass_through/spg_v2`。
  - 新增 `--hmc_ignore_semantic_prior`。
  - `noop/pass_through` 会构造全 1 的 `PriorOutput`，用于 no-op / pass-through 安全性测试。
- `loger/pipeline/hybrid_memory_controller.py`
  - 修正 `prior_output=None` 时仍可走 HMC write-score override，保证 `hmc_ignore_semantic_prior=1` 不会关掉 `stage_d_x_dg_inv_sqrt` 这类几何/attention 写入策略。
- `tools/run_attention_cue_experiment.sh`
  - 透传 `STAGE_C_MODE`、`STAGE_C_CACHE_*`、`SEMANTIC_PRIOR_MODE`、`HMC_IGNORE_SEMANTIC_PRIOR`。

静态验证：

```text
python3 -m py_compile run_pipeline_abc_v2.py loger/pipeline/hybrid_memory_controller.py
bash -n tools/run_attention_cue_experiment.sh
```

均通过。

#### Phase 0 smoke：Stage C cache / Stage D no-op

固定：

```text
END_FRAME = 128
read = C23 past + pair/all beta 4.75
write = stage_d_x_dg_inv_sqrt
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA_WRITE_KEEP_SCOPE = both_overlap
SWA_OVERLAP_SOURCE_REPLACE = source / kv / alpha 0.50
RESET_EVERY = 5
```

| Run | Stage C cache | Stage D / HMC | ATE RMSE | Rot RMSE | RPE t | RPE r | pose max abs diff | cache | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `ACL2V6_SMOKE_B1_stageC_cache_ignored_e128` | `readwrite` compute/save | semantic disabled, HMC ignore | `1.354989` | `3.029616` | `95.055239` | `0.026470` | reference | wrote 5 chunks | Stage C cache 生成成功 |
| `ACL2V6_SMOKE_B2_stageC_cache_read_ignored_e128` | `read`, require hit | semantic disabled, HMC ignore | `1.355008` | `3.032874` | `95.051396` | `0.026456` | `0.00086` vs B1 | require-hit pass | cache load 成功 |
| `ACL2V6_SMOKE_B3_stageD_noop_ignored_e128` | `read`, require hit | `noop`, HMC ignore | `1.355008` | `3.032874` | `95.051396` | `0.026456` | `0.00086` vs B1 | require-hit pass | Stage D 构造但未消费，和 B2 完全一致 |
| `ACL2V6_SMOKE_B4_pass_through_consumed_e128` | `read`, require hit | `pass_through`, HMC consumed | `1.355008` | `3.032874` | `95.051396` | `0.026456` | `0.00086` vs B1 | require-hit pass | pass-through consumed 与 B2/B3 完全一致 |

补充观察：

- B2/B3/B4 的指标和 trajectory txt 完全一致。
- B1 首次 compute/save 与 B2 cache-load 存在极小数值差：ATE `+0.000019m`，Rot `+0.003258deg`，trajectory 最大元素差 `0.00086`。这个量级在 smoke gate 内，但 full no-op parity 仍以 B0/B1/B2 full 为准。
- Cache 目录已写出 5 个 chunk：
  - `chunk_000_000000_000032`
  - `chunk_001_000029_000061`
  - `chunk_002_000058_000090`
  - `chunk_003_000087_000119`
  - `chunk_004_000116_000128`

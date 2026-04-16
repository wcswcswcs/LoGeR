# Video Masklet Front-end 设计文档（v2）

> 本文档定义 **Video Masklet Front-end** 模块。模块以当前 chunk 的 RGB 视频为主输入，以 **Grounded-SAM2（SAM 2.2）风格的关键帧发现 + 视频传播** 为核心实现路径，输出覆盖整个 chunk 的 **2D semantic masklets**。本文档特别强调：前端必须覆盖 **chunk 中所有帧的 thing 与 stuff 区域**，而不是只从首帧初始化并跟踪少量对象；其中天空、墙面、地面、树木、水面等 **stuff 类区域** 也是关键输出对象，因为它们直接影响后续的 write prior 设计。

---

## 0. 一句话定位

**Video Masklet Front-end 的职责，是在当前 chunk 内构建一组带语义标签、覆盖所有帧、同时包含 thing 与 stuff 的 2D masklets，并将这些 masklets 作为语义先验的主要来源送入 Semantic Prior Generator。**

---

## 1. 模块目标

本模块需要完成下面四件事：

1. 在当前 chunk 的多个关键时刻发现 **所有当前可见的重要语义区域**；
2. 将这些发现组织成 **贯穿整个 chunk 的 masklets**，而不是只跟踪首帧对象；
3. 同时覆盖 **thing 类实例** 与 **stuff 类区域**；
4. 为每条 masklet 提供可直接服务 write prior 的语义标签、质量分数和时序可见性。

这里的“所有当前可见的重要语义区域”包括但不限于：

- thing：人、车、动物、自行车、椅子、箱子、桌子、包等；
- structural stuff：墙、地面、天花板、建筑、道路、台阶、窗户等；
- low-value / risky stuff：天空、树木/植被、水面、反光、屏幕、烟雾等。

这些区域中，有的在后续阶段应高写入，有的应低写入，但前端必须先把它们显式提出。

---

## 2. 设计原则

### 2.1 以 chunk 为运行单元，而不是首帧 tracking

系统的基本单元是一个 chunk：

$$
\mathcal{W}_m = \{I_{m,1}, \dots, I_{m,T}\}
$$

前端必须在整个 `\mathcal{W}_m` 上工作，允许对象在任意帧进入系统，而不是要求所有对象都在首帧被发现。

### 2.2 thing 与 stuff 同等重要

这版前端不是纯 object tracker，而是 **semantic masklet generator**。因此：

- thing 类通常对应 instance masklets；
- stuff 类通常对应 region masklets；
- 两者在输出接口上统一表示。

特别地，**天空** 虽然通常是低写入价值区域，但它必须被显式识别出来，因为这类标签正是后续压低 write prior 的关键。

### 2.3 采用 “周期性发现 + 全 chunk 传播 + trigger refresh”

Grounded-SAM2 / SAM 2.2 风格的实现不应退化成“首帧 grounding + 全程跟踪”。更合理的主线是：

```text
关键帧发现 -> 局部传播 -> 中途新区域插入 -> 触发式刷新 -> chunk 结束导出
```

### 2.4 语义标签应服务 write prior，而不是只追求精细分类

本模块的标签体系要优先满足：

- 哪些区域是长期结构锚点；
- 哪些区域是动态风险源；
- 哪些区域是低价值 stuff；
- 哪些区域是高不确定性或短时区域。

因此允许使用粗粒度语义组，而不要求复杂的开放词表层级树。

### 2.5 Grounded-SAM2 作为主干，实现上允许“借鉴式组合”

前端推荐以 **Grounded-SAM2（SAM 2.2）** 作为主实现模板，即：

- 用 grounding 式 discovery 在关键帧找语义候选；
- 用 SAM 2.2 风格视频传播把候选扩展到 chunk 其他帧；
- 在中途帧允许再次 grounding / refresh；
- 结合 cue-trigger 和 uncovered 区域发现，保证覆盖完整性。

---

## 3. 模块在整条 Pipeline 中的位置

```text
Input chunk
    ↓
LoGeR Geometry Backbone
    ↓
Dynamic Cue Extractor
    ↓
Video Masklet Front-end
    ↓
M_mask / V_mask / Q_mask / L_sem / F_mask
    ↓
Semantic Prior Generator
```

本模块的直接上游是：

- RGB chunk
- 可选的几何 cue（用于 refresh / discovery supplement）

本模块的直接下游是：

- Semantic Prior Generator

---

## 4. 语义类别体系

为直接服务 write prior，推荐采用两层语义体系。

### 4.1 一级语义组

| 一级组 | 含义 | 后续写入趋势 |
|---|---|---|
| `STRUCTURE_ANCHOR` | 墙、地面、建筑、道路、台阶、窗框等 | 高 |
| `MOVABLE_THING` | 人、车、动物、自行车、包、箱子、椅子等 | 低到中 |
| `STATIC_THING` | 柜体、大型固定家具、设备壳体等 | 中到高 |
| `LOW_VALUE_STUFF` | 天空、水面、植被、树木、反光、烟雾、屏幕等 | 低 |
| `UNCERTAIN_REGION` | 语义不稳、边界碎片、小区域 | 低 |

### 4.2 二级标签（可选）

在一级语义组下，可保留更细粒度标签，例如：

- `person / rider / bicycle / motorcycle / car / bus / truck`
- `chair / table / sofa / cabinet / box / bag`
- `wall / floor / ceiling / building / road / stair / door / window`
- `sky / tree / vegetation / water / glass-reflection / monitor`

若当前 grounding 模型能输出更细类别，可保留二级标签；若不能，一级语义组已足够支持 Stage D。

---

## 5. 输入定义

### 5.1 主输入

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| RGB chunk | `X_m` | `[T, H, W, 3]` | 当前 chunk 图像序列 |
| cue 张量（可选） | `E_cue` | `[T, H, W, 5]` | `C_stat/C_dyn/C_occ/C_unc/C_anchor` |
| overlap handoff（可选） | `H_{m-1\to m}` | 结构体列表 | 上一 chunk 的局部 masklet 状态 |

### 5.2 `H_{m-1\to m}` 推荐结构

```text
{
  local_id_prev,
  overlap_frame_indices,
  overlap_masks,
  overlap_boxes,
  visibility_scores,
  semantic_label,
  semantic_group,
  app_features,
  quality_scores,
}
```

这里只承载短时 continuity，不代表全局 identity。

---

## 6. 输出定义

设当前 chunk 最终输出 `J` 条 semantic masklets。

### 6.1 主输出张量

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| semantic masklets | `M_mask` | `[J, T, H, W]` | 每条 masklet 的 2D 支撑 |
| visibility | `V_mask` | `[J, T]` | 每帧是否可见 |
| bbox | `B_mask` | `[J, T, 4]` | 每帧 `xyxy` bbox |
| propagation quality | `Q_mask` | `[J, T]` | 传播与 refresh 质量 |
| semantic label | `L_sem` | `[J]` 或 `[J,K]` | 离散标签或类别分布 |
| semantic group | `G_sem` | `[J]` | 一级语义组 |
| appearance feature | `F_mask` | `[J, T, C_a]` | 区域外观特征 |
| 2D centroid | `C2d_mask` | `[J, T, 2]` | 图像质心 |
| area ratio | `A_ratio` | `[J, T]` | mask 面积占比 |

### 6.2 元信息输出

```text
Meta_mask = {
  local_track_id: [J],
  source_type: [J],            # grounding / auto / cue-guided / refresh / handoff
  anchor_frames: list[list[int]],
  birth_frame: [J],
  end_frame: [J],
  dormant_flag: [J],
  stuff_flag: [J],
  thing_flag: [J],
  replaced_by_refresh: [J],
}
```

### 6.3 chunk 边界导出包

输出给下一 chunk 的 handoff：

```text
H_{m->m+1}
```

只包含最后 `T_o` 帧仍活跃或最近被 refresh 的 masklets。

---

## 7. 内部状态设计

前端内部建议维护三类 masklet 状态。

### 7.1 Active masklets

当前仍可稳定传播，且语义标签和几何支撑相对可信。

### 7.2 Dormant masklets

近期短时不可见、被遮挡、或传播质量下降，但仍值得等待后续 refresh / rediscovery。

### 7.3 Ended masklets

已离开画面或已被更高质量 refresh 版本替代，不再继续传播。

对于 stuff 区域，可再增加一个标签：

### 7.4 Persistent stuff regions

例如天空、墙面、地面等，通常跨多帧稳定存在，但边界随视角可能缓慢变化。它们更适合通过周期性 refresh + 全 chunk propagation 管理，而不是短时 object tracking。

---

## 8. 核心算法主线

推荐采用下面的主流程：

```text
Stage A  Window initialization
Stage B  Keyframe scheduling
Stage C  Grounded discovery on keyframes
Stage D  Proposal merge and semantic assignment
Stage E  All-frame propagation across chunk
Stage F  Late-entry insertion and trigger refresh
Stage G  Quality scoring and pruning
Stage H  Overlap handoff export
```

---

## 9. Stage A：窗口初始化

### 9.1 对齐输入

- 对齐 RGB、cue map 分辨率；
- 若有 `H_{m-1\to m}`，将 overlap 区域已有 masklets 注入到当前窗口前几帧的初始状态；
- 初始化 `active / dormant / ended` 容器。

### 9.2 发现优先级图

定义当前帧 discovery priority：

$$
D_{\text{prio}}(t)=
\omega_1 \cdot \text{uncovered\_ratio}(t)
+ \omega_2 \cdot \overline{C_{dyn}}(t)
+ \omega_3 \cdot \overline{C_{occ}}(t)
+ \omega_4 \cdot \overline{C_{unc}}(t)
+ \omega_5 \cdot \text{propagation\_drop}(t)
$$

它用于决定哪些帧更值得作为关键帧。

---

## 10. Stage B：关键帧调度

采用 **固定步长 + 触发式插入**：

### 10.1 固定关键帧

$$
\mathcal{K}^{\text{base}}_m = \{1, 1+s_k, 1+2s_k, \dots\}
$$

### 10.2 触发式插入

若某帧满足以下任一条件，则加入关键帧集合：

1. 未解释区域比例过高；
2. `C_dyn / C_occ / C_unc` 高且当前无足够 mask 覆盖；
3. 传播质量明显下降；
4. overlap handoff 进入当前窗口后 continuity 变差。

最终关键帧集合为：

$$
\mathcal{K}_m = \mathcal{K}^{\text{base}}_m \cup \mathcal{K}^{\text{trigger}}_m
$$

---

## 11. Stage C：关键帧发现（Grounded Discovery）

Grounded-SAM2 / SAM 2.2 风格的发现建议分三条支路并行。

### 11.1 全帧语义 grounding

在关键帧 `t \in \mathcal{K}_m` 上，用预定义 prompt bank 对全图做 grounding：

- thing prompts
- structure prompts
- stuff prompts

例如 prompt bank 可写成：

```text
THING_PROMPTS = [
  "person", "car", "bus", "truck", "bicycle", "motorcycle", "animal",
  "chair", "table", "sofa", "box", "bag", "cabinet"
]

STUFF_PROMPTS = [
  "wall", "floor", "ceiling", "building", "road", "stair", "door", "window",
  "sky", "tree", "vegetation", "water", "screen", "reflection"
]
```

对每个 grounding proposal，用 SAM 2.2 风格的 mask decoder 生成精细 mask。

### 11.2 cue-guided discovery

对于 `C_dyn` / `C_occ` / `C_unc` 高的区域，若当前没有已知 mask 覆盖，则补充：

- positive point prompt
- box prompt
- optional negative prompts from static surroundings

这个分支特别适合发现：

- 中途进入画面的动态物体；
- 遮挡边界附近的新出现区域；
- 几何不一致导致普通 grounding 漏检的区域。

### 11.3 未覆盖区域 discovery

记当前所有 active masklets 在关键帧上的覆盖并集为：

$$
C_t(h,w)=\max_j M_j(t,h,w)
$$

定义未覆盖区域：

$$
U_t = 1 - \text{dilate}(C_t)
$$

对 `U_t` 上的高显著区域再做 discovery，保证：

- 中途新对象可以进入系统；
- 大面积 stuff 区域不会因为首帧缺失而永远缺失。

---

## 12. Stage D：proposal 融合与语义赋值

### 12.1 proposal 过滤

每个 proposal 至少检查：

- 面积下限与上限；
- SAM 置信度；
- 几何支撑率；
- 与已存在 masklet 的重复度；
- 语义类别稳定性。

### 12.2 语义类别分配

若 grounding 输出离散类别，则直接赋值。  
若输出的是类别分数，则保留：

$$
\pi_j(c) \in [0,1], \qquad \sum_c \pi_j(c)=1
$$

并可定义一级语义组：

$$
G_{sem}(j)=g\big(\arg\max_c \pi_j(c)\big)
$$

### 12.3 proposal 与已有 masklet 的关系

新 proposal 可能是：

1. 新 masklet；
2. 旧 masklet 的 refresh；
3. 被丢弃的重复候选。

采用以下 affinity：

$$
A(i,j)=
\lambda_{iou}\, IoU
+ \lambda_{app}\, \cos(F_i,F_j)
+ \lambda_{sem}\, S_{sem}(i,j)
+ \lambda_{geo}\, Q_{geo}
$$

若 affinity 高，则作为 refresh 或 merge；否则新建 masklet。

---

## 13. Stage E：全 chunk 传播

### 13.1 传播基本原则

**每个关键帧上的新发现或 refresh 结果，都会成为一个 anchor。**

从该 anchor 出发：

- 向后传播到 chunk 末尾；
- 向前传播到 chunk 开始；
- 必要时仅在相邻关键帧之间局部传播，再由多个 anchor 拼接成全 chunk masklet。

这保证了：

- chunk 中后半段才出现的对象，仍然能被覆盖到其出现之前后的所有相关帧；
- 不是只有首帧对象才有完整轨迹。

### 13.2 全帧覆盖要求

对任意帧 `t`，都要求系统维护：

- 当前已解释区域
- 未解释区域
- 当前 ակտիվ masklets 列表

因此，前端不是“先有对象再传播”，而是一个持续维护 chunk 全帧覆盖的过程。

### 13.3 thing 与 stuff 的传播差异

#### thing masklets

- 目标是 instance-level continuity；
- 更关注边界精确与局部运动一致性；
- 更可能进入 dormant / refresh。

#### stuff masklets

- 目标是 region-level continuity；
- 允许大面积、低频变化；
- 更关注类别一致和覆盖完整；
- 可由多个关键帧的区域 mask 进行 piecewise refresh。

天空、墙面、地面等通常属于这一类。

---

## 14. Stage F：late-entry insertion 与 trigger refresh

### 14.1 late-entry insertion

当前 chunk 中后期才进入画面的对象，必须能在该时刻的关键帧被纳入系统，而不是要求回到首帧。

因此系统需要支持：

- 任意关键帧新建 masklet；
- 新 masklet 从其出生帧开始向前/向后有限传播；
- 在 chunk 输出中显式记录 `birth_frame`。

### 14.2 trigger refresh

当出现以下情况时，需要触发 refresh：

- 传播质量下降；
- `C_occ` 持续升高；
- `C_unc` 持续升高；
- 当前 masklet 所在区域周围出现大量未解释区域。

refresh 的作用是：

- 用新的高质量 keyframe mask 替换旧传播结果；
- 把长期漂移限制在相邻关键帧之间。

---

## 15. Stage G：质量评分

为服务 Stage D 的 write prior，本模块需给每条 masklet 生成质量分数。

### 15.1 每帧质量

建议：

$$
Q_{mask}(j,t)=
\sigma\Big(
q_0
+ q_1 \cdot conf_{prop}(j,t)
+ q_2 \cdot Q_{geo}(j,t)
+ q_3 \cdot sem_{stab}(j,t)
- q_4 \cdot drift(j,t)
- q_5 \cdot frag(j,t)
\Big)
$$

其中：

- `conf_prop`：传播置信度；
- `Q_geo`：mask 内有效几何支撑比例；
- `sem_stab`：关键帧间语义一致性；
- `drift`：位置/形状异常变化；
- `frag`：边界破碎程度。

### 15.2 整条 masklet 质量摘要

可定义：

$$
\bar Q_{mask}(j)=
\frac{\sum_t V_{mask}(j,t) Q_{mask}(j,t)}{\sum_t V_{mask}(j,t)+\epsilon}
$$

作为后续排序、prune 和 handoff 的依据。

---

## 16. Stage H：overlap handoff

chunk 结束时，导出以下对象到下一 chunk：

- 最后 `T_o` 帧仍 active 的 masklets；
- 最近进入 dormant 但边界仍可信的 masklets；
- 刚被 refresh 过的高质量 masklets；
- 大面积 persistent stuff regions（如天空、墙、地面）。

### 16.1 handoff 结构建议

```text
H_{m->m+1} = {
  masklets_on_overlap,
  bboxes_on_overlap,
  semantic_labels,
  semantic_groups,
  quality_scores,
  app_features,
  source_type,
}
```

### 16.2 handoff 的用途

- 帮助下一 chunk 的窗口前端快速恢复局部连续性；
- 避免 chunk 边界处频繁重新发现造成的闪断；
- 但不承担全局 identity 语义。

---

## 17. 为什么天空和 stuff 类是关键

本模块必须显式纳入天空等 stuff 类，原因不是为了后续对象跟踪，而是为了后续 **写入控制**：

1. 若天空未被识别出来，Semantic Prior Generator 很难稳定地给这类区域低 write prior；
2. 大面积 sky / tree / water 等 region 往往占据大量 token，如果不显式分类，容易让 token prior 过于依赖几何 cue；
3. 像墙、地面、建筑这类 structure stuff 则往往是高写入价值区域，也必须被识别出来。

所以，前端必须把 **thing + stuff** 放在同等重要的位置，而不是只追求 COCO 风格实例目标。

---

## 18. 与 Semantic Prior Generator 的接口

本模块对 Stage D 的核心作用是提供：

- `M_mask`：区域覆盖；
- `V_mask`：可见性；
- `Q_mask`：masklet 质量；
- `L_sem / G_sem`：语义标签与语义组；
- `F_mask`：外观特征（可选）；
- `birth_frame / anchor_frames / source_type`：调试与时间线信息。

Stage D 将基于这些信息做：

- masklet 级语义写入价值计算；
- cue 按 mask 聚合；
- pixel-level prior 融合；
- patch token prior 生成。

---

## 19. 实现优先级

### Phase 1：最小 Grounded-SAM2 原型

- 关键帧 grounding discovery；
- chunk 内 forward/backward propagation；
- masklet 输出与质量打分；
- 支持中途新对象进入。

### Phase 2：补齐 stuff 类和全 chunk 覆盖

- 引入固定 prompt bank 的 stuff discovery；
- 未覆盖区域 discovery；
- sky / wall / floor / tree / water 等 persistent stuff masklets。

### Phase 3：加入 cue-guided refresh

- `C_dyn / C_occ / C_unc` 触发 refresh；
- 与 overlap handoff 联动；
- 与 Stage D 共同做 quality-aware fusion。

---

## 20. 推荐默认超参数

| 参数 | 建议值 | 含义 |
|---|---:|---|
| `T` | 与 LoGeR chunk 一致 | 当前窗口长度 |
| `T_o` | 不小于 LoGeR overlap | handoff 长度 |
| `s_k` | 4～8 | 基础关键帧间隔 |
| `\tau_{cov}` | 0.05～0.10 | 未覆盖区域触发阈值 |
| `\tau_{refresh}` | 0.3～0.5 | 传播质量下降触发阈值 |
| `\tau_{dup}` | 0.8～0.9 | proposal 去重阈值 |
| `A_{min}` | 0.001～0.01 | proposal 最小面积占比 |

---

## 21. 最终接口摘要

### 输入

- `X_m: [T,H,W,3]`
- `E_cue: [T,H,W,5]`（可选）
- `H_{m-1\to m}`（可选）

### 输出

- `M_mask: [J,T,H,W]`
- `V_mask: [J,T]`
- `B_mask: [J,T,4]`
- `Q_mask: [J,T]`
- `L_sem: [J]` 或 `[J,K]`
- `G_sem: [J]`
- `F_mask: [J,T,C_a]`
- `Meta_mask`
- `H_{m->m+1}`

### 一句话总结

**Video Masklet Front-end 的核心，不是从首帧启动几个对象然后一路跟踪，而是用 Grounded-SAM2（SAM 2.2）风格的关键帧发现、全 chunk 传播、late-entry insertion 和 stuff-aware 区域建模，在当前 chunk 的所有帧上构建一组覆盖完整、带语义标签、同时包含 thing 与 stuff 的 2D semantic masklets。**

---

## 22. 实验发现：C_dyn 的空间分布特征与正确使用方式

### 22.1 C_dyn 集中在物体边缘，而非物体内部

在 office 场景 50 帧实验中，我们观察到 Stage B 输出的 **C_dyn 通道（动态线索）主要激活在物体边缘和深度不连续处**，而非物体内部区域。这是因为：

1. **C_dyn 的计算原理**：$C_{\text{dyn}} = \alpha_1 (1 - C_{\text{stat}}) - \alpha_3 C_{\text{occ}}$，其中 $C_{\text{stat}}$ 基于 pairwise point residual（同一像素在不同帧的世界坐标差异）。
2. **深度边缘的视差效应**：在物体边缘，相邻帧之间由于相机运动产生视差（parallax），导致同一像素的世界坐标在不同帧之间变化较大，point residual 偏高。
3. **因此 C_dyn 高值区域 ≈ 深度不连续的边缘**，而非真正"在动"的物体中心。

### 22.2 C_dyn 不适合作为检测 prompt

基于上述发现，**将 C_dyn 高值区域直接作为检测器或 SAM2 的空间 prompt 是不合理的**：

- C_dyn 的高值 blob 对应的是物体边缘轮廓碎片，不是完整的物体区域；
- 这些碎片的 bounding box 不对应有意义的物体（例如一把椅子的边缘被拆成多个碎片）；
- 将这些碎片注册到 SAM2 会产生破碎的、无意义的 masklet。

### 22.3 正确使用方式：后处理 per-masklet 打分

C_dyn 的正确用法是作为 **masklet 级别的后处理打分信号**：

1. Stage C 正常运行：检测器提供语义标签，SAM2 负责跟踪 → 得到 masklet
2. 对每个 masklet $j$，计算其 mask 区域内的平均 C_dyn：

$$
\text{dyn\_score}(j) = \frac{1}{|\mathcal{V}_j|} \sum_{t \in \mathcal{V}_j} \frac{\sum_{(x,y) \in M_{j,t}} C_{\text{dyn}}(t, x, y)}{|M_{j,t}|}
$$

其中 $\mathcal{V}_j$ 是 masklet $j$ 可见的帧集合，$M_{j,t}$ 是第 $t$ 帧的 mask 区域。

3. **高 dyn\_score 的 masklet** → 该物体在几何上不一致（可能在运动或是几何不确定区域）→ 后续 TTT write controller 应降低写入权重
4. **低 dyn\_score 的 masklet** → 该物体几何稳定 → 适合作为 TTT 记忆锚点

这一信号将在 Stage D（Semantic Prior Generator）和 Stage E（TTT Write Controller）中被消费。

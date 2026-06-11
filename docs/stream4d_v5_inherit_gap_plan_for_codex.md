# Stream4D v5 Inherit / Fixed-Support Gap Plan for Codex

## 1. 为什么需要 v5

v4.1 已经确认一个关键事实：

```text
inherit_pre_points / fixed-support 下，pure Stream4D 仍没有超过 Stream3D。
```

当前 probe5 最好 pure Stream4D：

```text
stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
AP/AP50/AP25 = 0.281615 / 0.497583 / 0.690254
union in target = 0.957699
#pred = 415.6
```

同一 32f support 下 Stream3D：

```text
scannet_on_stream4d_32f_probe5
AP/AP50/AP25 = 0.399213 / 0.597171 / 0.742535
union in target = 0.985608
#pred = 128.2
```

差距：

```text
AP   ≈ 0.1176
AP50 ≈ 0.0996
AP25 ≈ 0.0523
```

这说明 v4.1 的后处理式修复已经接近上限。v5 不能继续只扫 WTA、NMS、score、threshold、low-confidence recall。v5 必须把改动前移到：

```text
proposal generation
multi-view boundary evidence
object identity / split-merge memory
candidate quality calibration
```

## 2. v4.1 已排除的方向

以下方向已有实验证据，不应作为 v5 主线继续堆阈值：

| 方向 | v4.1 证据 | 结论 |
|---|---|---|
| recompute/self support top-k | recompute 可高分，但 fixed support 崩 | 不能证明方法本体超过 Stream3D |
| point-level WTA | probe5 best WTA 约 `0.258648 / 0.473374 / 0.678121` | 只能改点归属，不能修 object quality |
| point merge / point NMS | scene0050 负例 | 会误删 recall 或过合并 |
| support-aware global ranking | probe5 负例 | support area 只能解释 coarse coverage |
| silhouette direct ranking/filtering | probe5 负例 | silhouette 不是通用 object quality |
| greedy support novelty | probe5 负例 | 新 support 点不一定属于正确实例 |
| object-level postprocess competition | current best 上 AP 降到 `0.129698-0.216250` | 无监督代表选择会错删 recall |
| 只靠 low-confidence recall layer | best 到 `0.281615 / 0.497583 / 0.690254` | 有小收益，但离 Stream3D 很远 |

## 3. v5 验收指标

### 3.1 必须报告的对照

所有 v5 指标必须至少报告以下行：

```text
1. Stream3D self-inherit / original。
2. Stream3D on Stream4D 32f support。
3. v4.1 best pure Stream4D on 32f support。
4. v5 candidate on same 32f support。
5. v5 candidate on scannet / Stream3D self-inherit support。
6. recompute support diagnostic。
```

不能只报告 recompute support。

### 3.2 probe5 gate

先在 probe5 上过 gate，再考虑 final/full ScanNet。

最低 gate：

```text
v5 pure Stream4D probe5 fixed-support AP > 0.32
```

强 gate：

```text
v5 pure Stream4D probe5 fixed-support AP50 >= 0.55
v5 pure Stream4D probe5 fixed-support AP25 >= 0.72
#pred <= 250
```

最终目标：

```text
v5 pure Stream4D probe5 fixed-support AP/AP50/AP25 接近或超过 Stream3D same 32f support：
0.399213 / 0.597171 / 0.742535
```

如果达不到，必须记录失败原因，不能改 claim。

### 3.3 final/full ScanNet gate

只有 probe5 达到最低 gate 后，才能跑 final/full。

final split 要报告：

```text
Stream3D final baseline
v4.1 pure Stream4D best
v5 pure Stream4D candidate
Stream3D on v5 support diagnostic
v5 on Stream3D support diagnostic
```

## 4. v5 核心假设

### H1：当前差距不是 coverage 不够，而是一对一实例质量不足

证据：

```text
high + 32f residual union in target = 0.982493
但 AP 约 0.277，几乎没有提升。
```

所以 v5 不能只补点。

### H2：boundary evidence 是有效信号，但必须进入 proposal generation

证据：

```text
self-discovered boundary refinement 把 AP 从 0.260386 提到 0.277338。
```

但作为后处理仍不够。v5 要在 proposal 形成时就使用 boundary evidence，而不是最后删点。

### H3：low-confidence recall layer 有有用候选，但需要 split/merge memory

证据：

```text
boundary high + compref recall:
0.281615 / 0.497583 / 0.690254
```

比 boundary high 更好。但低置信层太碎，不能直接变成高质量 output。

### H4：postprocess object competition 失败是因为缺少时间和边界证据

证据：

```text
object competition 把 #pred 降低，但 AP 大跌。
```

所以 v5 的 object competition 不能只看最终 3D overlap；它必须看：

```text
same-frame 2D mask exclusivity
multi-view boundary agreement
carrier/evidence temporal support
split/merge history
```

## 5. v5 模块设计

### Module A：MaskObservation Bank

目标：不要直接把 3D object 当作唯一单位。先保存每个 2D mask observation 的证据。

每条 observation 至少记录：

```text
scene_id
frame_id
mask_id
mask_area
carrier_ids
projected_point_ids
visible_point_ids
inside_mask_ratio
boundary_distance_mean
boundary_distance_quantiles
depth_consistency_ratio
3D centroid
3D bbox extent
appearance feature
```

注意：

```text
appearance feature 可以先用 RGB histogram，不许冒充 DINO/CLIP。
```

### Module B：Boundary-Aware Proposal Generation

目标：生成 object proposal 时直接利用 boundary evidence。

不要先生成大而脏的 3D mask，再后处理裁剪。改成：

```text
1. carrier seed 定位 object 在 2D mask 中的连通区域。
2. 对每个 observation 计算 boundary-safe core。
3. 只把多视角一致的 core 点作为 high-confidence support。
4. fringe 点单独保存，不直接并入 high-confidence mask。
```

输出两个 support：

```text
core_support
fringe_support
```

但 evaluator 只能吃二值 mask，所以 v5 必须学习一个无 GT 的二值化策略：

```text
core always keep
fringe only keep when:
  multi-view support >= threshold
  3D connected to core
  not claimed by stronger same-frame competitor
```

### Module C：Evidence Graph v2

v4.1 evidence graph 已证明方向有效，但仍依赖 scene0050 / cached carriers。

v5 graph 节点：

```text
MaskObservation
```

边：

```text
positive:
  shared carriers
  3D support overlap
  temporal consistency
  boundary-compatible projection

negative:
  same-frame different masks
  strong 3D overlap but different same-frame masks
  incompatible 3D centroid/extent
  boundary expansion too large
```

graph 输出不是直接 object，而是：

```text
object hypothesis candidates
```

每个 candidate 保留完整证据链，便于审计。

### Module D：Split / Merge Capable Object Memory

当前 memory 更像 merge/create。v5 要允许：

```text
pending object
quarantine evidence
split candidate
merge candidate
rollback recent observation
```

规则：

```text
如果新 observation 与一个 object 的一部分证据一致、与另一部分冲突，
不要立即合并。
先进入 pending split buffer。
等后续 frame 验证后，再决定 split 或 merge。
```

这是为了避免：

```text
早期错误合并后，object 越滚越脏。
```

### Module E：Calibrated Object Quality

v4.1 的 score/area/support ranking 都不够。v5 需要 object quality，但不能用 GT。

候选 quality 应由以下无监督信号组成：

```text
temporal observation count
same-frame conflict penalty
boundary-safe support ratio
3D compactness
core/fringe ratio
carrier support stability
appearance consistency
extent stability
support novelty relative to already accepted objects
```

输出时不应该固定 top-N，而应：

```text
1. 先接受 high-quality core objects。
2. 对冲突组做 local competition。
3. 对未解释 support 区域，才允许 low-confidence recall。
```

## 6. v5 实验路线

### Phase 0：审计与数据准备

必须先确认：

```text
probe5 所有 scenes 的 32f carrier cache 可用。
96f/128f cache 若不可用，必须记录 blocker。
```

如果 D4RT 不能稳定生成多窗口 cache，v5 先只做 32f evidence graph v2，不允许写多窗口收益。

### Phase 1：Observation Bank 导出

先在 probe5 生成 observation bank：

```text
outputs/stream4d_v5_observation_bank_probe5/<scene>.jsonl
```

每条记录可追踪到 frame/mask/carrier/point。

验收：

```text
每个 scene 至少输出：
  num_observations
  mean_boundary_distance
  mean_depth_consistency
  mean_carriers_per_observation
  same_frame_mask_count
```

### Phase 2：Boundary-Aware Proposal v2

基于 observation bank 生成：

```text
stream4d_v5_probe5_boundary_proposals
```

对照：

```text
v4.1 selfboundary best
v4.1 compref best
Stream3D same support
```

必须报告：

```text
AP/AP50/AP25
union in target
#pred
core points
fringe points
conflict ratio
```

### Phase 3：Evidence Graph v2

构建 observation graph，输出 object hypotheses。

扫描：

```text
negative edge strength
min temporal observations
core/fringe threshold
same-frame conflict penalty
```

禁止：

```text
用 GT 选择阈值。
```

可以：

```text
用 probe5 做开发 split，但 final 数字必须锁参数后再跑。
```

### Phase 4：Split / Merge Memory

只在 probe5 中两个场景先跑：

```text
scene0050_00
scene0011_00
```

记录：

```text
object count over time
pending split count
merge accepted count
merge rejected count
same-frame conflict count
```

如果 AP 没提升，但 object explosion 降低，也要记录。

### Phase 5：Probe5 fixed-support gate

锁参数后跑全 probe5：

```text
stream4d_v5_probe5_candidate_on_32f_support
```

必须和以下对照同表：

```text
Stream3D same 32f support
v4.1 best pure Stream4D
v5 candidate
v5 recompute diagnostic
v5 on scannet self-inherit support
```

### Phase 6：Final split

只有 Phase 5 达到最低 gate 后执行。

## 7. 日志要求

沿用 v4.1 要求。

执行日志必须写：

```text
完整命令
输入/输出 config
split 文件
support config
环境变量
summary 文件路径
```

复盘日志必须写：

```text
真实 AP/AP50/AP25
不能达成时的失败原因
每个修复做了什么
是否读取 GT
是否改变 evaluator
和 Stream3D 同 support 的差距
```

## 8. 允许和禁止的 claim

允许写：

```text
v5 在 probe5 fixed-support 上相对 v4.1 改善。
v5 的某个模块改善 boundary quality / object count / conflict ratio。
```

只有当实际超过时才允许写：

```text
v5 pure Stream4D 在 inherit/fixed support 下超过 Stream3D。
```

禁止写：

```text
用 recompute support 的高分证明 fixed-support 超越。
用 hybrid Stream3D-primary 证明 pure Stream4D 超越。
隐藏 Stream3D same support 对照。
把 GT-read-only oracle 当作方法结果。
```

## 9. v5 第一优先级

第一步不是继续跑后处理，而是实现：

```text
MaskObservation Bank + Boundary-Aware Proposal v2
```

原因：

```text
v4.1 已证明 boundary evidence 是最稳定的正信号；
但后处理边界裁剪只能到 0.281615 AP。
下一版必须让 boundary evidence 进入 proposal 生成阶段。
```

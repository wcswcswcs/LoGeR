# Stream4D v6 方法优先代码审计与并行实验计划书

面向 Codex 的执行文件。本文档基于重新解压并审阅以下材料后写成：

```text
/mnt/data/stream4d_v5_code_audit_packet_20260608.zip
/mnt/data/粘贴的 markdown (1)。md(422)
/mnt/data/Code_Stream3D.zip
/mnt/data/Open-d4rt-main.zip
```

审计解压路径：

```text
/mnt/data/audit_v5_20260608_new/stream4d_v5_code_audit_packet_20260608
/mnt/data/orig_code_stream3d_current/Code_Stream3D
/mnt/data/opend4rt_audit_current/Open-d4rt-main
```

本文档只使用 Typora 友好的公式格式，即 `$...$` 和 `$$...$$`，不使用 display 公式的方括号语法。

---

## 0. 这版计划的核心转向

v5 的主要失败不是工程速度，也不是 D4RT cache，而是算法本体没有形成正确 object。下一步不能继续做 top-k、NMS、简单 fusion、简单 score sweep、单帧 mask observation 直出、local proposal 直出。这些已经被实验否定。v6 必须把方法改成：

```text
mask observation -> typed carrier-tracklet evidence graph -> object hypothesis with core/fringe/reject support -> global object competition -> streaming 4D memory
```

换句话说，D4RT 的 carrier 不应该直接当最终 mask，也不应该只是扩大 support 的工具；它应该作为“物理 surface tracklet 的证据约束”，帮助判断哪些 2D masks 是同一个 object 的互补视角，哪些是冲突、污染、边界漂移或弱桥。

当前最重要的 scientific hypothesis 是：

$$
\text{Stream4D 的主要瓶颈} \neq \text{coverage 不足};
$$

而是：

$$
\text{Stream4D 的主要瓶颈}=\text{object formation + boundary purity + one-to-one assignment}.
$$

证据已经很明确：v5 observation bank 的 `pre/union` 达到 `55.2481%`，但 AP 只有 `0.001469`；local proposal bank 的 `pre/union` 达到 `35.7973%`、`#pred=173.20`，但 AP 只有 `0.112524`，并且 conflict rate 约 `0.616`。这说明“有点、有 mask、有 proposal”不等于“有正确 object”。

---

## 1. 本轮代码审计结论

### 1.1 我实际审阅了什么

本轮 zip 解压后共包含 27 个文件，主要是：

```text
Stream3D/evaluation/evaluate.py
Stream3D/stream4d/appearance_memory.py
Stream3D/stream4d/d4rt_adapter.py
Stream3D/stream4d/d4rt_worker.py
Stream3D/stream4d/export_scannet.py
Stream3D/stream4d/object_memory_v2.py
Stream3D/stream4d/reexport_scannet.py
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/run_scannet.py
Stream3D/tests/test_stream4d_protocol_fixes.py
Stream3D/tools/*.py
最新 v5 复盘和执行日志
```

我执行了：

```bash
cd /mnt/data/audit_v5_20260608_new/stream4d_v5_code_audit_packet_20260608/Stream3D
python -m py_compile evaluation/evaluate.py stream4d/*.py tools/*.py tests/*.py
```

结果：`py_compile` 通过。

但是这只说明单文件语法没有错，不能说明代码包可运行。因为该审计包缺少大量 `run_scannet.py` 和 `object_memory_v2.py` 需要 import 的模块。

### 1.2 P0：本次代码审计包不完整，不能作为完整可运行代码包

我在干净解压目录中执行：

```bash
python - <<'PY'
import sys
sys.path.insert(0, '.')
import stream4d.object_memory_v2
PY
```

实际失败：

```text
ModuleNotFoundError: No module named 'stream4d.local_4d_filter'
```

这不是算法结果问题，而是审计包问题。当前 zip 没有包含至少这些关键依赖：

```text
Stream3D/stream4d/carrier_sampler.py
Stream3D/stream4d/carrier_store.py
Stream3D/stream4d/diagnostics.py
Stream3D/stream4d/local_4d_filter.py
Stream3D/stream4d/mask_evidence.py
Stream3D/stream4d/object_memory.py
Stream3D/stream4d/scannet_stream.py
Stream3D/stream4d/memory_diagnostics.py
Stream3D/stream4d/motion_memory.py
Stream3D/stream4d/evidence_graph.py
Stream3D/stream4d/replay_evidence_graph.py
Stream3D/stream4d/replay_memory.py
```

因此，本次我能审核新增/修改文件、结果复盘与明显风险，但不能在这个 zip 内完成完整 import-level audit。Codex 下一次提交必须按第 12 节的代码审计包规范提交完整代码，否则不允许把“单测通过”写成“代码可复现”。

### 1.3 P0：目前没有发现直接虚假指标，但需要补强 evaluator-manifest guard

正向结论：

```text
1. v5 复盘显示 oracle output config 必须包含 oracle，否则工具拒绝输出。
2. evaluator 对 oracle-named prediction/TMP/output 默认拒绝，除非显式 --allow-oracle-eval。
3. reportable configs 的 scanner 显示 num_uses_gt_and_method_result=0。
4. metric integrity 显示 evaluator AP core hash 与原版 Stream3D evaluator 一致。
5. alignment source configs 的 object_dict/pred alignment mean/min IoU 为 1.0。
```

因此，我没有发现“非 oracle 方法读取 GT 后生成 prediction”或“直接改 AP core”的证据。

但仍有一个 P0 安全缺口：`evaluation/evaluate.py` 目前主要靠路径名里是否包含 `oracle` 来拒绝 oracle 评估。如果某个 GT-read diagnostic 被错误命名成不含 oracle 的 config，evaluator 本身不会读取 manifest 来拒绝。当前 scanner 能发现一部分问题，但 evaluator/report 脚本还不够强。

Codex 必须新增：

```text
1. evaluation/evaluate.py 可选读取 prediction_manifest.json。
2. 若 manifest.uses_gt=true 且未传 --allow-oracle-eval，则直接拒绝。
3. report 表格生成脚本必须 --require-manifest。
4. 没有 manifest 的 config 只能进入 diagnostic/audit，不可进入 method table。
```

### 1.4 P0：`export_object_dict_mask_backproject()` 的 score-mode 实现与 v5 local proposal 结果不完全相符

`tools/export_local_proposal_bank.py` 默认：

```text
--export-score-mode observations
```

但 `ScanNetExporter.export_object_dict_mask_backproject()` 构造 `object_records` 时只写了：

```python
{"object_id": ..., "point_ids": point_ids, "score": len(point_ids)}
```

没有写 `observations`、`area_score`、`reliability` 等字段。随后 `_write_outputs()` 在 `export_score_mode="observations"` 时会使用：

```python
record.get("observations", record.get("reliability", 0.0))
```

这意味着 local proposal bank 的 prediction score 很可能全是 0，而不是文档语义上的 observation count。这个 bug 不构成虚假指标，因为没有读 GT，也没有提高分数；但它说明 C2 的“observations score”实验没有真正按计划执行。Codex 必须修复并重跑 C2：

```text
mask_backproject object record 必须写：
observations = len(unique mask observations)
carrier_count = len(carrier_ids)
area_score = len(point_ids)
reliability = observations * sqrt(area)
```

然后比较：

```text
score=one
score=observations
score=area
score=reliability
```

如果重跑后 AP 仍低，才能确认 local proposal bank 失败不是 score-field bug。

### 1.5 P1：`fuse_prediction_configs.py` 的 support 使用需要进入 manifest

`fuse_prediction_configs.py` 可使用：

```text
--drop-overlap-pre-points-config
```

这不会读 GT，但会使用某个 config 的 support universe 来决定 primary/secondary 的 suppression overlap。如果这个 support 来自 `scannet` 或 Stream3D baseline，它就不是纯 Stream4D 方法内部可用信号，只能作为 diagnostic/hybrid。当前 manifest 只写了 `support_policy=fusion:<mode>`，没有明确记录 `drop_overlap_pre_points_config`。

Codex 必须修复：

```text
1. manifest.extra.drop_overlap_pre_points_config = value
2. 若 drop_overlap_pre_points_config 是 scannet 或 Stream3D baseline，则 is_diagnostic_only=true，除非实验明确是 hybrid diagnostic。
3. pure Stream4D method table 禁止使用 Stream3D/scannet support 作为 selection signal。
```

### 1.6 P1：`d4rt_nn` 仍未实现，不能 claim D4RT-native geometry

`ScanNetExporter.export_d4rt_nn()` 仍是：

```python
raise NotImplementedError(...)
```

所以当前 ScanNet 仍是 `rgbd_eval` bridge：D4RT 用于 carrier / UV / visibility / confidence，最终导出到 ScanNet mesh 仍依赖 RGB-D/pose。不能写成“Stream4D 已完成 D4RT-native geometry”。v6 必须新增专门实验回答：**Stream3D 如果使用 D4RT 几何，性能会下降多少？** 详见第 6 节。

---

## 2. 最新实验结果的独立判断

### 2.1 v5 有进展，但不是算法性能进展

v5 的真实进展在三方面：

```text
1. metric safety：manifest / oracle guard / scanner / metric integrity 通过。
2. D4RT cache：96f probe5 5/5 完成，128f 3/5 完成。
3. 诊断清楚了：observation/candidate 不缺，object formation 失败。
```

v5 没有达成 ScanNet probe5 第一阶段目标：

```text
目标：AP >= 0.32, AP50 >= 0.53, AP25 >= 0.70, #pred <= 300, union in target >= 0.94
新增 localprop：0.112524 / 0.285390 / 0.479773, #pred=173.20
v4.1 current best：0.281615 / 0.497583 / 0.690254, #pred=415.6
```

所以 v5 新 method path 不是“慢慢接近”，而是比 v4.1 best 退步很大。它的价值是证明简单 local proposal bank 不是答案。

### 2.2 目前离 Stream3D 还差多少

在 probe5 32f fixed support 上，关键对比是：

```text
Stream3D on Stream4D 32f support:
0.399213 / 0.597171 / 0.742535

Stream4D v4.1 current best:
0.281615 / 0.497583 / 0.690254

Stream4D v5 localprop:
0.112524 / 0.285390 / 0.479773
```

因此 v4.1 best 距 Stream3D 仍差：

$$
\Delta AP = 0.399213 - 0.281615 = 0.117598,
$$

$$
\Delta AP50 = 0.597171 - 0.497583 = 0.099588,
$$

$$
\Delta AP25 = 0.742535 - 0.690254 = 0.052281.
$$

v5 localprop 距 Stream3D 更远：

$$
\Delta AP = 0.399213 - 0.112524 = 0.286689.
$$

这说明 v5 不是“只差一点调参”，而是算法结构没有抓住 object formation 的关键。

### 2.3 当前路线哪些是正确的，哪些该停止

正确方向：

```text
1. evidence graph / typed evidence，而不是 raw local proposal。
2. carrier 作为 mask connected component 的 seed，而不是直接作为最终 support。
3. component densify + strict relative coverage gate。
4. fixed-support 诊断和 oracle diagnostic 继续保留，但不能作为方法结果。
5. D4RT cache 多窗口基础设施继续保留，因为 D4RT cache 不是主要 blocker 了。
```

应该停止或降级为 diagnostic 的方向：

```text
1. raw mask observation direct export。
2. local proposal bank min2/bestframe direct export。
3. 简单 top-k / NMS / point merge / point IoC merge。
4. 单纯扩大 densify 半径或增加 max masks per object。
5. 简单 score-mode sweep。
6. Stream3D-primary hybrid 作为主方法。
```

尤其要注意：hybrid diagnostic 已经证明 Stream4D recall layer 可以帮助 Stream3D，但这不是纯 Stream4D 胜利。它可以作为 motivation：D4RT carrier 证据确实有互补价值；但 paper 主方法不能把 Stream3D prediction 当 primary。

---

## 3. v6 方法主线：不是工程堆叠，而是 object formation 改写

### 3.1 核心表示

v6 的基本单位不是 2D mask，也不是 3D point mask，而是 mask-carrier tracklet evidence node：

$$
n_i = (t_i, m_i, C_i, U_i, X_i, q_i),
$$

其中：

```text
t_i: frame/time
m_i: 2D mask id
C_i: 支持该 mask 的 D4RT carrier set
U_i: mask 内 carrier seed / projected pixel support
X_i: 由 RGB-D 或 D4RT geometry 得到的 3D support summary
q_i: 不确定性与质量特征
```

object 不是由单个 mask 导出，而是图分割得到的 component：

$$
O_k = \{n_i \mid z_i = k\}.
$$

每个 object 维护三类 support：

$$
S_k = S_k^{core} \cup S_k^{fringe}, \quad S_k^{reject} \cap S_k = \varnothing.
$$

解释：

```text
core: 高置信、边界干净、由多视角一致支持的点。
fringe: 有召回价值但不确定，只有满足连通/多帧/低冲突才进入。
reject: 低 coverage mask、same-frame conflict、跨实例边界污染、D4RT drift 区域。
```

### 3.2 typed evidence graph

构建有类型边，而不是单一 carrier IoC：

$$
E = E^{+}_{track} \cup E^{+}_{comp} \cup E^{-}_{conflict} \cup E^{-}_{own} \cup E^{-}_{geom} \cup E^{weak}_{bridge}.
$$

边类型定义：

```text
positive_track:
  两个 node 共享稳定 carrier tracklet，且 cycle/self/visibility 质量高。

positive_complement:
  两个 node 来自不同视角，carrier overlap 不一定高，但 3D centroid/extent 兼容，且 mask view 互补。

negative_conflict:
  同一帧两个不同 mask，或者同一 carrier 同一时间支持不同 object。

negative_ownership:
  两个 node 回投到大量相同 3D points，但在 2D 中长期分离。

negative_geometry:
  3D bbox/centroid/extent 明显不兼容。

weak_bridge:
  低 coverage node 不能单独把两个 component 合并，只能附着到已有 strong component 上提供 fringe。
```

节点质量不用于硬删除，因为实验已经证明低 coverage node 可能是有用桥。节点质量只用于：

```text
1. edge ordering
2. component support selection
3. core/fringe/reject 分配
4. object-level competition score
```

### 3.3 graph partition 目标

v6 不直接做贪心 merge，而做带 cannot-link 的训练自由图分割。目标函数可以写成：

$$
\max_{z} \sum_{(i,j)\in E^+} w^+_{ij} \mathbf{1}[z_i=z_j]
- \sum_{(i,j)\in E^-} w^-_{ij} \mathbf{1}[z_i=z_j]
- \lambda K
- \mu \Omega(z),
$$

其中 $\Omega(z)$ 惩罚 object 内部不紧凑、过大、跨帧冲突或边界污染。

Codex 第一版不需要写 ILP，可做 greedy constrained correlation clustering：

```text
1. strong positive edges 从高到低排序。
2. 合并前检查 cannot-link closure。
3. weak_bridge 不能触发两个 strong components 合并，只能 attach 到一个 component。
4. 合并后计算 component quality；若 quality 下降过多则拒绝。
5. component 输出前做 split audit：若同一 object 内存在多个互相 cannot-link 的 frame clusters，则 split。
```

### 3.4 object 内 support selection

v4.1 的关键正信号是 component_densify + relative coverage gate。v6 需要把它系统化。

对 object $O_k$ 和 mask observation $n_i$，先计算：

$$
r_i = \alpha_1 q_i^{carrier} + \alpha_2 q_i^{boundary} + \alpha_3 q_i^{component} + \alpha_4 q_i^{view} - \alpha_5 q_i^{conflict}.
$$

然后做：

```text
1. 只用高 r_i 的 mask 生成 core。
2. 中等 r_i 的 mask 只能提供 fringe candidate。
3. 低 r_i 的 mask 进入 reject，不参与 support。
4. fringe 点必须与 core 在 3D 中连通，且不违反 ownership conflict。
5. 如果加入 fringe 会显著降低 compactness 或增加 same-frame conflict，则拒绝。
```

二值输出时不能直接输出 core+fringe 的并集。应使用 self-consistency threshold：

$$
S_k(\tau)=S_k^{core}\cup\{p\in S_k^{fringe}: q_{kp}\ge \tau_k\}.
$$

$\tau_k$ 不用 GT，而用无监督稳定性选择：

$$
\tau_k^* = \arg\max_\tau \left[Q_{view}(S_k(\tau)) + Q_{compact}(S_k(\tau)) - Q_{conflict}(S_k(\tau))\right].
$$

### 3.5 global one-to-one object competition

当前 pure Stream4D 的 AP/AP50 输给 Stream3D，不只是排序问题，而是多个 predictions 抢同一个 GT-like object，同时还有一些高 IoU object 生成不够。v6 输出前需要 global object competition，但不能用 GT。

将 object hypothesis 视作候选集合 $\mathcal{H}$，选择输出集合 $\mathcal{Y}$：

$$
\max_{\mathcal{Y}\subseteq\mathcal{H}} \sum_{h\in\mathcal{Y}} Q(h)
- \beta \sum_{h_i,h_j\in\mathcal{Y}} \text{OverlapConflict}(h_i,h_j)
+ \gamma \text{SupportNovelty}(h_i).
$$

约束：

```text
1. 同一 frame 内来自 cannot-link masks 的 hypotheses 不能同时大量覆盖同一 3D region。
2. 如果 h_i 几乎被 h_j 包含，保留 quality 更高者。
3. 如果 h_i/h_j 是同一 component 的 core/fringe variant，不允许作为两个 object 输出。
4. weak_bridge 产生的 fringe 不能独立成为 object。
```

Codex 可先实现 greedy set packing：按 $Q(h)$ 排序，每加入一个 hypothesis 都更新 occupied support 和 conflict ledger；但必须保存每一步拒绝原因，用于审计。

---

## 4. 关键实验总目标

v6 实验不追求马上 full ScanNet。目标是并行验证三个核心假设：

```text
H1: Stream4D 当前瓶颈是 object formation 和 high-IoU boundary，不是 coverage。
H2: typed evidence graph + core/fringe/reject support 能在 probe5 fixed support 上显著缩小与 Stream3D 的差距。
H3: Stream3D 若替换为 D4RT geometry，会出现可量化下降；该下降决定我们能否把 D4RT geometry 作为 ScanNet 主路径，还是只作为 4D tracking carrier。
```

只有 H1/H2 通过，才跑 full ScanNet。只有 H3 明确，才能在论文中诚实描述 D4RT geometry 的作用。

---

## 5. Phase A：代码审计与可复现提交规范

### A1. 目标

确保后续每次结果都能被重新审核，不再出现“py_compile 通过但审计包缺关键模块”的情况。

### A2. Codex 必须提交的代码包

每轮实验后，Codex 必须生成：

```text
stream4d_v6_code_review_packet_<YYYYMMDD_HHMM>.zip
stream4d_v6_code_review_packet_<YYYYMMDD_HHMM>.sha256
```

zip 必须包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/evaluate.py
Stream3D/evaluation/constants.py if modified
Stream3D/evaluation/utils_3d.py if modified
Stream3D/tests/*.py
Stream3D/configs/stream4d*.json
Stream3D/splits/scannet_v6_probe5.txt
Stream3D/splits/scannet_tune.txt
Stream3D/splits/scannet_final.txt
Stream3D/docs/<current plan>.md
Stream3D/docs/<execution log>.md
Stream3D/docs/<result recap>.md
git_status.txt
git_diff.patch
filelist.txt
import_smoke.log
unit_tests.log
metric_integrity.log
```

必须排除：

```text
large checkpoints
raw ScanNet data
raw Dynamic Replica data
large prediction npz unless explicitly requested
__pycache__
```

但必须包含小型 toy fixture：

```text
Stream3D/tests/fixtures/tiny_scene_prediction.npz
Stream3D/tests/fixtures/tiny_pre_points.npy
Stream3D/tests/fixtures/tiny_gt.txt
```

用于测试 evaluator、manifest、object_dict alignment 和 score-mode。

### A3. 必须通过的审计命令

Codex 在打包前必须运行：

```bash
python -m py_compile evaluation/evaluate.py stream4d/*.py tools/*.py tests/*.py
python - <<'PY'
import stream4d.run_scannet
import stream4d.object_memory_v2
import stream4d.export_scannet
import stream4d.reliable_densifier
import tools.verify_stream4d_metric_integrity
import tools.scan_reportable_configs
print('import smoke OK')
PY
python -m unittest discover -s tests
```

通过标准：

```text
1. import smoke 必须通过。
2. unit tests 必须通过。
3. filelist 中必须包含所有 import 依赖。
4. git_diff.patch 必须包含本轮修改。
5. result recap 中的每个 config 必须有 manifest。
```

如果不满足，不能报告新 AP。

### A4. 代码层 P0 修复列表

Codex 必须先修：

```text
P0-1: 审计包缺失依赖模块。
P0-2: evaluate.py 读取 manifest，拒绝 uses_gt=true 的非 allow-oracle 评估。
P0-3: export_object_dict_mask_backproject 写入 observations/area_score/reliability。
P0-4: fuse_prediction_configs manifest 记录 drop_overlap_pre_points_config，并区分 diagnostic/hybrid/pure。
P0-5: 所有 reportable configs 必须 manifest_exists=true, uses_gt=false, is_method_result=true。
```

---

## 6. Phase B：Stream3D 使用 D4RT 几何会不会下降

这是用户特别要求的新实验。它必须作为 v6 的核心实验，而不是附录。

### B1. 为什么要做

Stream3D 的原始强项来自 RGB-D/pose 和完整 reconstructed point cloud。它的局部 set-cover 和 manifold refining 都假设几何是稳定、metric、同一场景坐标系下的。如果把几何换成 D4RT feed-forward geometry，性能可能下降，原因包括：

```text
1. D4RT 坐标需要统一 reference，尺度和全局坐标不一定和 ScanNet mesh 一致。
2. Stream3D manifold threshold δ 是 metric-space threshold；D4RT 坐标若尺度不同，需要 normalized threshold。
3. D4RT 对薄结构、反光、遮挡、快速运动可能有 track/geometry drift。
4. Stream3D 的 2D mask -> 3D mask projection 对 depth/pose 精度非常敏感。
```

这个实验回答：**D4RT 几何本身是否足以替代 Stream3D 的 RGB-D/pose 几何？** 如果明显下降，论文主线就不能说“D4RT geometry 替代 GT geometry 后静态 ScanNet 仍同等强”；只能说 D4RT 主要提供 4D correspondence，RGB-D bridge 是 ScanNet evaluation adapter。

### B2. 三条几何路径

在同一 2D masks、同一 frame list、同一 Stream3D grouping 参数下比较：

```text
G0: Stream3D-RGBD
    原版 Stream3D，使用 ScanNet depth + pose + scene mesh。

G1: Stream3D-D4RT-Geometry-Internal
    用 D4RT 预测的 $x^{D4RT}$ 作为 Stream3D 内部 geometry，所有 set-cover / manifold / merging 在 D4RT 坐标中完成。
    方法内部不使用 ScanNet GT instance，也不使用 Sim3。
    δ 等 metric threshold 改成 D4RT normalized scale。

G2: Stream3D-D4RT-Sim3-Export-Diagnostic
    grouping 仍在 D4RT 坐标中完成；仅在 evaluation/export 阶段用 Sim3 把 D4RT object points 对齐到 ScanNet mesh。
    Sim3 不允许反馈到 grouping/memory/selection。
```

可选诊断：

```text
G3: Stream3D-D4RT-DepthPose-Bridge
    用 D4RT query 生成 per-frame depth/pose，再按 Stream3D 原始 projection 生成 3D masks。
    这是几何替换诊断，不作为主方法，因为它更容易把 Sim3/pose adapter 混入方法。
```

### B3. D4RT geometry export 设计

对每个 2D mask observation，在 mask 内采样 pixels $p=(u,v,t)$，查询：

$$
X_p^{D4RT}=D(u,v,t_{src}=t,t_{tgt}=t,t_{cam}=t_{ref}).
$$

对于同一 scene，选定 reference frame $t_{ref}$。推荐先用 `t_ref=0` 和 `t_ref=window center` 做对照。

D4RT 内部距离阈值不能直接用 `0.05m`，需要归一化：

$$
\delta^{D4RT}=\eta \cdot \text{medianNN}(X^{D4RT}_{anchors}),
$$

或：

$$
\delta^{D4RT}=\eta \cdot \text{medianDepth}(X^{D4RT}) / 100.
$$

这里 $\eta$ 只在 tune split 选，不看 final GT。

评估导出时估计 Sim3：

$$
T^* = \arg\min_{s,R,t}\sum_i \omega_i \|sR X_i^{D4RT}+t-X_i^{ScanNet}\|_2^2.
$$

anchor 来源必须记录：

```text
A0: D4RT self depth point 与 ScanNet RGB-D point 的同像素 anchor。
A1: 高 visibility/confidence carrier anchor。
A2: RANSAC filtered static/background anchors。
```

Sim3 只用于 evaluation/export，不能用于方法内部 object formation。

### B4. 实验设置

先跑：

```text
B-probe5: scene0011_00, scene0030_00, scene0050_00, scene0081_01, scene0591_00
B-tune30: 30 scenes
B-final: 仅当 probe5 gate 通过后运行
```

每个 scene 对比：

```text
G0 original Stream3D-RGBD
G1 Stream3D-D4RT internal geometry + Sim3 export
G1-noMR: 去掉 Stream3D manifold refining，检查 D4RT geometry 是否被 MR 阈值误伤
G1-normMR: 使用 normalized manifold threshold
G1-trackMR: 用 D4RT track consistency 替代 Euclidean MR
```

### B5. 必须记录的指标

几何质量：

```text
sim3_anchor_count
sim3_inlier_ratio
sim3_scale
sim3_rotation_det
sim3_residual_mean / median / p90
D4RT depth AbsRel against ScanNet depth on sampled pixels
D4RT point-to-ScanNet NN distance mean / p90
uv_in01_rate
visibility_mean
confidence_mean
```

Stream3D projection质量：

```text
2D mask count
valid projected mask count
projection hit rate
mask 3D point count mean/median
D4RT mask point count vs RGBD mask point count ratio
D4RT-vs-RGBD 3D mask IoU on same frame/mask after Sim3 export
```

最终 segmentation：

```text
AP/AP50/AP25, recompute
AP/AP50/AP25, fixed Stream3D support
pre_points %
union %
union in target %
#pred
GT crop/full
fragmentation proxy: predictions per GT oracle best
merge proxy: GT best IoU distribution
```

速度：

```text
D4RT encode/decode seconds
Stream3D projection seconds
Sim3 fitting seconds
```

### B6. 判断标准

最低可接受：

```text
G1-D4RT geometry 相比 G0-RGBD 的 AP drop <= 3.0，AP50 drop <= 5.0。
Sim3 residual median <= 0.08m 或相对 scene scale <= 2%。
projection hit rate >= 70%。
```

强成功：

```text
G1-D4RT geometry 在 AP25 接近 G0，AP/AP50 drop <= 2.0，并且在动态场景上 tracking 明显更好。
```

如果失败：

```text
1. 若 Sim3 residual 高：检查 frame indexing、t_cam reference、坐标轴、D4RT output normalization、anchor selection。
2. 若 residual 可接受但 AP 低：说明 D4RT geometry 可对齐，但 Stream3D manifold/merge 不适配 D4RT sparse/noisy support；尝试 G1-noMR 和 G1-trackMR。
3. 若 projection hit rate 低：提高 anchor/queries，改用 mask connected-component seeds，而不是 dense mask pixels。
4. 若 AP25 尚可但 AP/AP50 低：问题是边界精度，不是粗几何。
```

### B7. 可视化

每个 scene 保存：

```text
D4RT point cloud after Sim3 vs ScanNet mesh overlay
Sim3 residual heatmap
same 2D mask: RGBD 3D mask vs D4RT 3D mask overlay
D4RT geometry fail cases: high residual, out-of-frame, low visibility
Stream3D-RGBD vs Stream3D-D4RT prediction comparison
```

---

## 7. Phase C：Typed Evidence Graph v3

### C1. 目标

验证 typed graph 是否能在 probe5 fixed support 上显著缩小与 Stream3D 的差距。

### C2. 实验组

在 probe5 上并行跑：

```text
C0 current v4.1 best fixed32
C1 egraph old scalar edges
C2 typed graph: positive_track + negative_conflict
C3 C2 + weak_bridge attach-only
C4 C3 + positive_complement geometry compatibility
C5 C4 + component split audit
C6 C5 + object-level competition
```

### C3. 必须记录的指标

主指标：

```text
AP/AP50/AP25, own recompute
AP/AP50/AP25, fixed 32f support
AP/AP50/AP25, scannet self-inherit support diagnostic
#pred
union in target %
```

图指标：

```text
raw observations
nodes
edges by type
accepted edges by type
rejected edges by type
cannot-link violations prevented
weak_bridge attachments
components before split
components after split
mean nodes/component
mean frames/component
mean carriers/component
```

support 指标：

```text
core points
fringe candidate points
fringe kept points
reject points
core/fringe ratio
support conflict rate
3D compactness
mask relative coverage distribution
```

### C4. 成功标准

probe5 minimal gate：

```text
fixed32 AP >= 0.32
fixed32 AP50 >= 0.53
fixed32 AP25 >= 0.70
#pred <= 300
union in target >= 0.94
```

strong gate：

```text
fixed32 AP >= 0.36
fixed32 AP50 >= 0.57
fixed32 AP25 >= 0.73
```

full success：

```text
pure Stream4D fixed32 >= Stream3D on same support in AP50/AP25, and AP gap <= 0.03。
```

如果 C2/C3 提升 AP25 但 AP 不升，说明 coarse grouping 有效但 boundary 不够，进入 Phase D。如果 C6 降低 #pred 但 AP 降，说明 object competition 误删 recall，回退并输出 reject reason histogram。

### C5. 可视化

每个 probe scene 保存：

```text
graph component visualization: node timeline, edge types, cannot-link edges
object core/fringe/reject overlay on 2D frame
3D support colored by component and conflict
failure panels: duplicate, over-merge, under-merge, boundary pollution
```

---

## 8. Phase D：Object-internal core/fringe/reject support

### D1. 目标

解决 v4.1 的核心矛盾：strict component densify 提升 recompute，但 fixed support 崩；wide support 提升 AP25 但伤 AP/AP50。v6 要在同一个 object 内区分 core/fringe/reject，而不是输出多个 hypothesis 或简单并集。

### D2. 实验组

固定 typed graph best，比较：

```text
D0 carrier support
D1 maskbp core only
D2 component_densify strict relative coverage
D3 core + connected fringe, no confidence threshold
D4 core + fringe with self-consistency threshold
D5 D4 + ownership-aware WTA
D6 D5 + per-object adaptive threshold
```

### D3. 指标

```text
AP/AP50/AP25 own/fixed
core point count
fringe candidate count
fringe kept count
reject count
fringe accepted ratio
fringe conflict ratio
boundary distance mean/p10
3D connected component count per object
per-object compactness change from core to core+fringe
```

### D4. 成功标准

```text
AP50 不低于 strict core best - 0.02
AP25 提升至少 +0.05
fixed32 AP 提升至少 +0.03
conflict rate <= 0.25
```

如果 fringe 加入后 AP/AP50 崩：缩小 fringe source，只允许 multi-frame supported fringe。若 AP25 不升：说明 fringe 过保守，放宽 3D connectivity 半径或允许 low-risk bridge nodes。若 fixed support 仍崩：问题不是 support 内阈值，而是 object candidate generation，回到 Phase C。

---

## 9. Phase E：Global object competition and one-to-one assignment

### E1. 目标

减少重复 predictions 和多个 candidates 抢同一 object，同时不误删 rare/small objects。

### E2. 方法

构造 object-level conflict graph：

$$
G_O=(\mathcal{H}, E_O),
$$

其中节点是 object hypothesis，边表示包含、冲突、重复或 weak complement。

质量分：

$$
Q(h)=q_{core}+q_{view}+q_{track}+q_{compact}-q_{conflict}-q_{boundary}.
$$

选择准则：

```text
1. 高 quality 且提供 new support 的 object 优先。
2. 被高 quality object 大比例包含的 object 不输出。
3. 低 quality object 只有在提供 substantial new support 且不冲突时输出。
4. same-frame cannot-link violation 大的 object quarantine，不直接输出。
```

### E3. 实验组

```text
E0 no competition
E1 containment suppression
E2 containment + support novelty
E3 E2 + same-frame conflict penalty
E4 E3 + small-object rescue
E5 E4 + quarantine/pending buffer
```

### E4. 指标

```text
AP/AP50/AP25 own/fixed
#pred
duplicate prediction rate proxy
contained prediction count
rejected reason histogram
small object prediction count
small object AP proxy/oracle coverage
union in target
```

### E5. 成功标准

```text
#pred 从 400+ 降到 <=300
fixed32 AP 不下降，最好 +0.03
AP50 提升至少 +0.03
AP25 不下降超过 0.02
```

如果 #pred 降但 AP 降：competition 误删 recall，启用 small-object rescue。若 AP25 降很多：novelty 阈值过高。若 AP50 不升：duplicate 不是主因，回到 boundary/core support。

---

## 10. Phase F：Split/Merge-capable Object Memory

### F1. 目标

让更多窗口真的改善 object，而不是带来污染。v5 cache 已证明 96f/128f 可跑；问题是多窗口 object formation。

### F2. 方法

memory 不再只有 match/create。每个 object 有状态：

```text
active
pending
lost
reactivated
split_candidate
quarantine
merged
```

更新规则：

```text
1. 新窗口 evidence 先进入 pending buffer。
2. 如果与 active object positive_track 强且无 cannot-link，才 merge。
3. 如果同一 object 出现两个互相 cannot-link clusters，进入 split_candidate。
4. 如果新 evidence 只由 weak_bridge 支持，不直接 merge，只作为 fringe pending。
5. 若连续两个窗口验证稳定，再 commit。
```

### F3. 实验

先只在 probe5 做：

```text
32f old memory
96f old memory
96f v2 old Hungarian
96f split/merge memory
128f split/merge memory
```

### F4. 指标

```text
AP/AP50/AP25 own/fixed
num_windows
num_created
num_matched
num_pending
num_committed
num_quarantined
num_split_candidates
num_reactivated
final_num_objects
object_growth_rate
fragmentation proxy
merge conflict proxy
```

### F5. 成功标准

```text
96f split/merge AP50 >= 32f current AP50
96f fixed32 AP >= v4.1 fixed32 + 0.03
final_num_objects <= 32f current * 1.5
quarantine ratio 不超过 40%
```

若 96f 仍差：不要继续加窗口。先在 single-scene 可视化 pending/quarantine，检查是否把有效 masks 拒掉或把冲突 masks commit。

---

## 11. Phase G：Replica-Dynamic / Dynamic Replica

### G1. 目标

验证 Stream4D 的真正优势：dynamic object identity 和 4D tracking。ScanNet 是静态 benchmark，不能证明动态 4D semantic reconstruction。

### G2. 数据检查

先运行：

```bash
python -m tools.check_dynamic_replica_env \
  --root <dynamic_replica_root> \
  --split valid \
  --output outputs/audit/dynamic_replica_env_v6.md
```

只有满足：

```text
usable_scene_count > 0
RGB/depth/camera 可用
至少有 instance masks 或 object IDs 或 trajectories
```

才能报告 official metrics。否则只允许 qualitative，不允许写 IDF1/MOTA/4D IoU。

### G3. 实验组

```text
DR0 Stream3D-style static overlap baseline
DR1 Stream4D old memory
DR2 Stream4D typed graph + split/merge memory
DR3 DR2 + D4RT-native geometry Sim3 evaluation adapter
```

### G4. 指标

若有 GT instance IDs：

```text
IDF1
ID switches
MOTA/MOTP if applicable
track fragmentation
reactivation accuracy
4D IoU over time
per-frame AP/AP50/AP25
object birth/death accuracy
```

若有 trajectories：

```text
3D trajectory endpoint error
trajectory APD3D-like thresholds
occlusion reappearance consistency
```

若没有 GT IDs：

```text
qualitative only
cycle consistency
self-consistency under time reversal
object identity stability proxy
```

### G5. 成功标准

```text
动态场景中 IDF1 或 reactivation accuracy 明显超过 Stream3D-style overlap baseline。
ID switches 降低至少 30%。
移动物体的 track fragmentation 降低至少 30%。
静态背景 AP 不明显下降。
```

如果 Dynamic Replica 数据仍缺失，Codex 必须输出 blocker 清单，不得造指标。

---

## 12. 并行执行计划

### Lane 0：代码审计与 P0 修复

立即执行：

```text
1. 补全审计包缺失模块。
2. 修 evaluate manifest guard。
3. 修 mask_backproject score-mode bug。
4. 修 fusion manifest support 记录。
5. 生成完整 import smoke 和 unit tests。
```

完成标准：下一轮提交的 zip 在干净目录 import smoke 通过。

### Lane 1：Stream3D-D4RT geometry degradation

并行实现 D4RT geometry adapter。先只跑 probe5，不等 v6 graph 完成。

输出：

```text
Stream3D-RGBD vs Stream3D-D4RT geometry 表格
Sim3 residual 表格
projection quality 表格
可视化 overlay
```

如果 D4RT geometry drop 很大，立刻停止把 D4RT geometry 当 ScanNet 主卖点；论文主线转为 D4RT carrier correspondence + dynamic tracking。

### Lane 2：Typed evidence graph v3

在 scene0050_00 和 probe5 上实现 typed edges。不要先跑 full ScanNet。

最小提交：

```text
graph node schema
edge type stats
cannot-link closure
weak_bridge attach-only
component split audit
```

### Lane 3：Core/fringe/reject support

基于 Lane 2 best components，做 object 内 support。重点看 AP/AP50/AP25 tradeoff，而不是只最大化 recompute。

### Lane 4：Object competition

只有 Lane 2/3 产生足够候选后再做。不要在 raw/localprop 上做 competition，因为它们已经被证明候选质量不够。

### Lane 5：Dynamic Replica

先做数据环境，不做指标想象。如果数据就绪，优先跑 2-3 个短动态序列，验证 moving object re-ID。

---

## 13. Full ScanNet final gate

只有 probe5 同时满足以下条件，才允许启动 full ScanNet final：

```text
pure Stream4D fixed32 AP >= 0.32
pure Stream4D fixed32 AP50 >= 0.53
pure Stream4D fixed32 AP25 >= 0.70
#pred <= 300
union in target >= 0.94
metric integrity pass
no oracle config in reportable set
D4RT geometry degradation experiment completed or explicitly marked diagnostic/blocker
```

Full final 表必须包含：

```text
Method
Geometry source: RGBD / D4RT / D4RT+Sim3 eval-only
2D masks: Cropformer / SAM2
D4RT: yes/no
Eval support policy: recompute / inherit / fixed / Stream3D-style
AP/AP50/AP25
pre_points %
union %
union in target %
#pred
time per scene
Sim3 residual if D4RT geometry
```

如果只在 recompute 上强、fixed/inherit 崩，必须写成 observed-support diagnostic，不能写成全面超越。

---

## 14. 每次实验必须保存的结果

每个 config 必须保存：

```text
prediction manifest
command log
config json
result txt/csv/json
metric integrity report
summary json
per-scene diagnostics
visualization index html or markdown
failure case list
```

每个实验复盘必须回答：

```text
1. 本实验验证哪个假设？
2. 是否使用 GT？如果使用，是否 diagnostic-only？
3. 是否使用 Stream3D/scannet support 作为 selection signal？
4. 指标是否在 own/recompute 和 fixed/inherit 上都报告？
5. 是否有比当前 best 更强？
6. 如果失败，失败是 coverage、boundary、one-to-one、D4RT geometry、score calibration 还是 data blocker？
7. 下一步要改算法什么地方，而不是改哪个阈值？
```

---

## 15. 安全 claim 边界

当前可以安全说：

```text
v5 完成了 metric safety 和 D4RT cache 基础设施。
probe5 上 observation/candidate 并不缺，但 object formation 仍失败。
v4.1 evidence graph + component densify 在 scene0050_00 own recompute 上有强正信号。
```

当前不能说：

```text
Stream4D v5 超过 Stream3D。
pure Stream4D 在 fixed/inherit support 下超过 Stream3D。
D4RT-native geometry 已经验证。
Dynamic Replica 已经有 tracking 结果。
local proposal bank 是 v5 Proposal v2。
Typed Evidence Graph v2 已完成。
```

论文主线建议：

```text
如果 ScanNet 静态 benchmark 只能 comparable，不要强行围绕 ScanNet 讲故事。
真正新意应是：D4RT carrier 提供 4D dynamic correspondence，typed evidence graph 把 2D semantic masks 变成 streaming semantic 4D object memory。
ScanNet 用来证明静态兼容性；Dynamic Replica 用来证明 4D tracking 优势。
```

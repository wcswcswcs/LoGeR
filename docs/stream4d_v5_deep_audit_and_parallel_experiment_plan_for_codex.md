# Stream4D v5：深度代码审核、结果判断与并行实验计划书

面向 Codex 的执行文件。本文档基于重新解压并审核最新代码包和最新 v4.1 复盘文件后写成：

```text
/mnt/data/stream4d_v4_1_code_review_packet(1).zip
/mnt/data/粘贴的 markdown (1)。md(420)
```

审核路径：

```text
/mnt/data/audit_v4_1_latest/Stream3D
```

本文档只使用 Typora 友好的公式格式，即 `$...$` 和 `$$...$$`。不要把本文档中的任何公式改成方括号 display 公式。

---

## 0. 本文件的结论先行

这轮 v4.1 不是完全没有进展，但**没有达到目标**。进展主要体现在：从早期只靠 `top-k / recompute_pre_points` 的 sparse-support 高分，推进到了 `evidence graph + component densify + boundary refinement + tiered support` 这条更像算法的路线。它在 `scene0050_00` 上有明显信号，在 `probe5` fixed-support 上也从原始 Stream4D 32f 的 `0.144238 / 0.288344 / 0.464716` 提升到当前 pure Stream4D best 的 `0.281615 / 0.497583 / 0.690254`。

但是，它仍然明显低于同一 32f support 下的 Stream3D：

```text
Stream3D on Stream4D 32f support, probe5:
AP / AP50 / AP25 = 0.399213 / 0.597171 / 0.742535
#pred = 128.2
union in target = 0.985608

Current best pure Stream4D, probe5 fixed support:
AP / AP50 / AP25 = 0.281615 / 0.497583 / 0.690254
#pred = 415.6
union in target = 0.957699

Gap:
AP   = -0.1176
AP50 = -0.0996
AP25 = -0.0523
```

所以，当前最诚实的判断是：

```text
v4.1 找到了正确方向的一部分：carrier 不能直接当最终 mask；carrier 更适合作为 2D mask 连通域的种子，evidence graph 和 boundary-aware support 也确实比单纯后处理更有效。

但 v4.1 仍没有解决核心问题：Stream4D 不能把 low-confidence recall 中有用但碎片化的候选，组织成少量、高质量、一对一的 object instances。
```

下一步不能再把主要时间花在 final prediction 后处理，例如 NMS、point merge、score ranking、top-k、support novelty、简单 object competition。下一步必须前移到 object 形成过程：

```text
mask observation bank
+ boundary-aware proposal generation
+ typed evidence graph
+ split/merge-capable object memory
+ unsupervised one-to-one object assignment
+ dynamic scene evaluation
```

---

## 1. 最新代码审核结论

### 1.1 我实际审核了什么

已解压：

```bash
rm -rf /mnt/data/audit_v4_1_latest
mkdir -p /mnt/data/audit_v4_1_latest
unzip -q /mnt/data/stream4d_v4_1_code_review_packet\(1\).zip -d /mnt/data/audit_v4_1_latest
```

代码包根目录：

```text
/mnt/data/audit_v4_1_latest/Stream3D
```

该包包含：

```text
configs/
splits/
stream4d/
tests/
tools/
```

但它**不包含完整的 `evaluation/` 目录，也不包含 `data/prediction`、`data/TMP`、`data/scannet/gt` 或实际 ScanNet 数据**。因此，从这个 zip 包本身，我能完成的是源码静态审核、语法检查、单元测试检查、GT 泄漏扫描和计划/实现一致性审核；我不能仅靠该 zip 重新计算 AP，也不能仅靠该 zip 独立比较 `evaluation/evaluate.py` 的 AP core hash。复盘文件中报告的 evaluator hash 与 no-GT-leakage 结论需要在完整工程目录中再次运行 `tools.verify_stream4d_metric_integrity` 复核。

已运行：

```bash
cd /mnt/data/audit_v4_1_latest/Stream3D
python -m py_compile stream4d/*.py tools/*.py tests/*.py
python -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 6 tests in 0.002s
OK
```

这说明最新包的 Python 语法和现有 6 个单元测试通过，但这不是性能达标证明。

### 1.2 与计划/implementation 是否相符

总体相符，但有边界：

```text
已实现并可见：
- metric / protocol tools：evaluate_cross_prepoints、verify_stream4d_metric_integrity、audit_stream3d_eval_protocol。
- reliable densifier：seeded component、boundary erosion、seed distance cap、WTA、component_densify、core_fringe。
- evidence graph：mask observation graph、component stability、same-frame cannot-link、replay_evidence_graph。
- memory-v2 初版：appearance_memory、motion_memory、object_memory_v2、replay_memory。
- 多种后处理诊断：NMS、WTA、fuse、support-aware rank、object competition、silhouette consistency、boundary refine。

仍未完成或不能 claim：
- D4RT-native `d4rt_nn` / Sim3 export 仍未作为主路径完成。
- Dynamic Replica / Replica-Dynamic 实验没有跑。
- 多场景 96f/128f cache 没有完成，scene0050 不能代表多场景。
- 当前输出还不是完整 queryable semantic 4D field，仍主要是 object proposal / prediction mask pipeline。
```

### 1.3 是否有虚假指标或 GT 泄漏

#### 1.3.1 没发现非 oracle 方法脚本直接用 GT 生成 prediction

我扫描了 `stream4d/` 和 `tools/` 中明显的 GT 读取信号，例如：

```text
gt_path
data/scannet/gt
np.loadtxt
loadtxt(
evaluation.constants
SCANNET_IDS
```

结论：

```text
- method 主路径，如 rescore、reexport、replay_evidence_graph、reliable_densifier、object_memory_v2，没有发现直接读取 GT 来生成 prediction 的证据。
- evaluate_cross_prepoints、audit_stream3d_eval_protocol、verify_stream4d_metric_integrity 会读取 GT 或统计 GT crop/full，这是指标审计工具，不是方法输出工具。
```

#### 1.3.2 P0 风险：`oracle_candidate_upper_bound.py` 是 GT-read tool，而且可以写 prediction

`tools/oracle_candidate_upper_bound.py` 明确读取：

```text
data/scannet/gt/{scene}.txt
evaluation.constants.SCANNET_IDS
np.loadtxt(gt_path)
```

它的文件开头写了：

```text
GT-read-only candidate-pool upper-bound diagnostic.
This tool must not be used to produce a method result.
```

但是它有 `--output-config`，会把 GT oracle 选择出的候选写成 prediction。这是当前最需要防误用的指标风险。它不是已经发生的虚假指标，但如果没有硬 guard，后续很容易被误当作 method config 评估。

Codex 必须立刻加保护：

```text
1. 如果 `oracle_candidate_upper_bound.py` 使用 `--output-config`，output_config 必须包含 `oracle`。
2. 写出的 prediction 路径必须包含 `oracle`，并写 manifest：`uses_gt=true`、`result_type=oracle_diagnostic_only`。
3. `tools.verify_stream4d_metric_integrity.py` 必须扫描所有待报告 configs，如果 manifest 中 `uses_gt=true`，直接报错。
4. `evaluation.evaluate` 或 wrapper 脚本若发现 config 名含 oracle，必须要求 `--allow-oracle-eval`，否则拒绝。
5. 论文表格、主日志、final summary 禁止读取 oracle config。
```

#### 1.3.3 当前 packet 不完整，不能从 zip 独立确认 AP core hash

最新 zip 缺 `evaluation/evaluate.py`。复盘文件里报告过：

```text
phase0_pass=True
evaluator_ap_core_equal_by_hash=True
gt_files_read_by_rescore=False
object_dict 与 prediction 列对齐 mean IoU = 1.0
```

这可以作为已有工程日志的证据，但 Codex 下一轮必须在完整项目目录重新运行：

```bash
python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /path/to/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet.txt \
  --configs scannet,stream4d_scannet_32f_ioc075_fixmem,... \
  --output outputs/audit/stream4d_v5_metric_integrity.md
```

通过之前，不允许新增主结果表。

### 1.4 发现的代码 bug / 弱点

#### P0-1：oracle output 缺硬 guard

上面已说明。这是最重要的虚假指标风险。

#### P0-2：每个 prediction config 缺统一 manifest

很多 tools 会写 prediction/TMP，但没有强制写：

```text
config_manifest.json
uses_gt
source_configs
pre_points_policy
support_policy
command
code_git_hash_or_zip_hash
input_cache
output_config
is_method_result
is_diagnostic_only
```

没有 manifest，结果越来越多后很难保证不会误用 diagnostic/oracle/postprocess 输出。Codex 需要给所有 prediction-writing tools 增加统一 `write_prediction_manifest()`。

#### P1-1：`ReliableDensifier` 的 observation 诊断字段覆盖了 raw count

在 `reliable_densifier.py` 中，`densify_observations_considered` 先记录原始 observation 数，后面经过 `_select_observations()` 后又被覆盖成 selected observation 数。这会隐藏到底过滤了多少 observation。

应改成：

```text
densify_observations_raw
densify_observations_after_quality_filter
densify_observations_selected
densify_observations_used_for_export
```

#### P1-2：`run_scannet.py` CLI 与 exporter 能力不同步

`export_scannet.py` 已支持很多后续模式，例如 `component_densify`、relative coverage gate、更多 score diagnostics；但 `run_scannet.py` 中暴露的 `--export-support-mode` / `--export-score-mode` choices 仍偏少。很多实验只能通过 replay/reexport tools 跑，导致结果复现路径分裂。

Codex 应统一：

```text
run_scannet.py
replay_evidence_graph.py
replay_memory.py
reexport_scannet.py
```

四者共用一个 `ExportConfig` dataclass 和同一套 manifest。

#### P1-3：`D4RTAdapter` 对长视频仍只有硬报错，没有 chunk/persistent worker

当前如果输入窗口帧数超过 checkpoint 的 `clip_frames`，直接：

```text
raise ValueError(... checkpoint supports clip_frames=...)
```

这与当前实验 blocker 对应：probe5 里尝试给 `scene0011_00` 生成 96f cache 时，log 为 0 bytes，进程处于 D state，未进入 GPU compute。这个不是算法结果，而是 D4RT checkpoint/filesystem/I/O 或初始化层面的 blocker。

Codex 必须先解决工程吞吐问题，否则多场景多窗口永远无法验证：

```text
1. 运行前把 D4RT ckpt 复制到本地 fast scratch，例如 /tmp 或本地 NVMe。
2. 增加 preflight load：单独加载 ckpt、打印耗时、退出。
3. 增加 persistent D4RT worker：一次加载，多场景复用，不要每个 scene 重复 torch.load 大 checkpoint。
4. 增加 lock / timeout / heartbeat：如果 5 分钟内 log 仍为 0 bytes，自动 abort 并记录 filesystem blocker。
5. 对 96f/128f 继续使用 32-frame D4RT chunks + overlap stitching，而不是让单个 D4RT window 超过 clip limit。
```

#### P1-4：`appearance_memory` 缺失特征时可能给出正相似度

需要检查 `appearance_memory.py` 中 `cosine_similarity_01(None, None)` 或类似路径。如果缺失 appearance 被映射为 0.5，这会让 memory-v2 出现 appearance-only 或 missing-feature positive match。当前复盘已经显示 RGB histogram 容易让相似椅子过合并，missing feature 不能再贡献正分。

应改为：

```text
similarity_score, is_valid = compute_similarity(...)
if not is_valid:
    do not add this term to numerator or denominator
```

#### P1-5：same-frame cannot-link 过硬，可能伤 over-segmented 2D masks

Evidence graph 当前把 same-frame different mask 当 hard negative 是有价值的，它确实减少了大量错误合并。但 2D segmentation 经常把同一物体过分割成多个互补 mask。绝对 cannot-link 会阻止这些互补 mask 在 graph 内合成完整 object。

v5 不应简单删除 cannot-link，而要把边类型分开：

```text
positive co-carrier edge
negative same-frame-overlap edge
complement same-frame-disjoint edge
weak temporal bridge edge
conflict carrier-owner edge
```

same-frame different mask 只有在它们同帧显著重叠、或互相竞争同一 carrier/support 时才应成为 hard negative；如果它们空间互补且跨帧被同一 carrier tracklet 连接，应该允许 delayed merge。

---

## 2. 对这次实验结果的独立判断

### 2.1 有进展，但不是目标达成

有进展的证据：

```text
1. scene0050_00 上，evidence graph + component_densify + strict relative coverage gate 把 128f own recompute 提到 0.490385 / 0.605769 / 0.810897，超过 32f current 的 0.202698 / 0.445714 / 0.681429。
2. probe5 fixed support 上，32f cached evidence graph + component densify 从 Stream4D 32f self 的 0.144238 / 0.288344 / 0.464716 提升到 0.240665 / 0.447128 / 0.671741。
3. 最新 boundary high + comp low-confidence layer 把 pure Stream4D probe5 fixed support 提到 0.281615 / 0.497583 / 0.690254。
```

这说明 `evidence graph`、`component_densify`、`boundary-aware refinement` 不是无效工程，它们确实推动了 fixed-support 指标。

但没达标的证据更强：

```text
1. probe5 same 32f support 下，Stream3D 是 0.399213 / 0.597171 / 0.742535。
2. 当前 best pure Stream4D 是 0.281615 / 0.497583 / 0.690254。
3. Stream4D 需要 415.6 个 prediction 才达到这个结果，而 Stream3D 只需要 128.2 个 prediction。
4. Stream4D union in target 已经到 0.957699，接近 Stream3D 的 0.985608，但 AP 仍差 0.1176。
```

所以失败不能再归因于“coverage 不够”这么简单。coverage 现在已经很接近，问题变成：

```text
Stream4D 的 object proposals 仍然碎、重复、边界不准，不能形成 Stream3D 那种少量、一对一、高 IoU 的 instance masks。
```

### 2.2 进度为什么显得慢

慢不是因为完全没想法，而是因为前面很多轮一直在 final prediction 层补救：

```text
top-k
min area
point dilation
mask backprojection
point WTA
point NMS
fusion
low-confidence recall layer
support ranking
greedy novelty
object overlap competition
```

这些方法大多只能改变排序或删候选，不能改变候选形成过程。它们只能处理症状，不能解决候选本身的边界、归属和 split/merge 问题。

最新负例尤其关键：`object_competition_rank.py` 把 prediction 数从 415.6 降到 233.4，但 AP 从 0.281615 掉到 0.190065。这说明：

```text
不是“预测太多，删掉一些”就能解决。
低置信候选里有 recall；简单压缩会错删有用候选。
真正缺的是能把低置信 recall 变成更少、更准的 object proposal 的机制。
```

### 2.3 是否在正确道路上

方向一半正确，一半需要立刻纠偏。

正确的部分：

```text
1. 放弃 pure carrier support，转向 carrier-seeded mask connected component：正确。
2. 放弃简单 dense backprojection，使用 boundary / relative coverage gate：正确。
3. 放弃只看 own recompute，持续报告 fixed/inherit：正确。
4. 从 postprocess 走向 evidence graph：正确。
```

需要纠偏的部分：

```text
1. 不要继续在 final prediction 上做 WTA/NMS/ranking/merge。
2. 不要继续把 scene0050 的 96f/128f 结果当成多场景趋势。
3. 不要继续只在 ScanNet 静态上硬拼 Stream3D，而不跑动态场景。
4. 不要把 recompute own-support 高分作为主 claim。
```

### 2.4 离目标还差多远

按不同目标分开看。

#### ScanNet fixed-support 超过 Stream3D

当前还差较远：

```text
probe5 fixed-support:
AP gap   = 0.399213 - 0.281615 = 0.117598
AP50 gap = 0.597171 - 0.497583 = 0.099588
AP25 gap = 0.742535 - 0.690254 = 0.052281
```

这不是一个小阈值能补上的差距。需要改变 object proposal generation。

#### Full ScanNet final split

还没完成。当前 best probe5 还没过，full final 不能先跑成主结果。应该先 probe5 过门槛，再跑 final。

#### 多窗口 4D tracking

还没开始实质验证。scene0050 有 96f/128f cache，其它 probe5 场景没有；Dynamic Replica 也没有跑。论文主线如果要讲 4D reconstruction and tracking，必须补这块。

#### D4RT-native geometry / Sim3 export

还没完成。当前 ScanNet 仍然是 `rgbd_eval` bridge，不能 claim D4RT 几何已替代 GT 3D。Sim3 只能在 evaluation/export adapter 使用。

---

## 3. v5 的核心改进思路：不要再后处理，前移到 object 生成

### 3.1 问题本质

当前 pipeline 大致是：

```text
2D masks -> D4RT carrier evidence -> local proposals -> object memory / evidence graph -> export -> final prediction postprocess
```

v4.1 的错误主要已经不是最后一步能修的。真正错误发生在：

```text
一个 object hypothesis 是如何从多帧 mask observations 和 carrier tracklets 形成的？
哪些 mask observations 应该是同一个 object 的 core evidence？
哪些是 fringe evidence？
哪些是互斥反证？
什么时候应该 split？什么时候应该 merge？
```

Stream3D 的优势来自它在静态 ScanNet 上能借助 dense RGB-D point cloud 和 3D manifold prior 形成较少、较完整的 3D instance。Stream4D 目前虽然有 D4RT carrier，但输出仍然是大量碎片化 hypotheses。要超越，必须让 carrier 真正用于 object formation，而不是只用于候选筛选或后处理。

### 3.2 v5 方法主线

v5 主线定义为：

```text
Observation-level object formation with track-carried evidence.
```

具体说：

```text
1. 建立 MaskObservation Bank：每个 2D mask observation 不是最终候选，而是一个带有 carrier、boundary、3D、appearance、temporal 信息的证据单元。
2. 建立 Typed Evidence Graph：边不再只有 positive/cannot-link，而是区分 co-carrier、complement、conflict、temporal bridge、appearance support。
3. 做 Boundary-aware Proposal Generation：object 的 core/fringe 在生成时决定，不在 final mask 后处理时补救。
4. 做 Split/Merge-capable Memory：object 可以 pending、split_candidate、merge_candidate、quarantine，而不是只能 match/create。
5. 做 Unsupervised One-to-One Assignment Proxy：用 same-frame exclusivity、carrier ownership、3D compactness、mask boundary consistency 逼近一对一实例质量，不能用 GT。
```

一个 object $O$ 不应再是若干 final masks 的并集，而应是：

$$
O = (C_{core}, C_{fringe}, E^+, E^-, B, G, A, Q)
$$

其中：

```text
C_core: 高置信 carrier / support core。
C_fringe: 低置信但可解释的 recall fringe。
E^+: 正证据边，例如多帧同 carrier / same object mask evidence。
E^-: 反证据边，例如 same-frame strong conflict / carrier ownership conflict。
B: boundary evidence。
G: 3D geometry evidence，例如 centroid、extent、compactness。
A: appearance evidence。
Q: object quality and lifecycle。
```

最终二值 mask 不再由简单 union 得到，而由 object 内部的支持分数决定：

$$
S_O(p) = w_c S_{carrier}(p) + w_m S_{mask}(p) + w_b S_{boundary}(p) + w_g S_{geometry}(p) + w_t S_{temporal}(p) - w_x S_{conflict}(p)
$$

输出点集：

$$
P_O = \{p \mid S_O(p) > \tau_O\}
$$

这里 $\tau_O$ 不能用 GT 调；只能用 tune split 或无监督分位数策略。

---

## 4. 并行执行总览

为了加快实验，不要串行等一个大实验结束。按 4 条 lane 并行：

```text
Lane A: Metric Safety / Code Hygiene
Lane B: D4RT Cache / Multi-window Infrastructure
Lane C: ScanNet Probe5 Algorithm Core
Lane D: Replica-Dynamic / Dynamic Replica
```

### 4.1 Lane A：指标与代码安全

目标：任何主结果前，先确保没有 oracle 泄漏、没有 evaluator 误用、没有 manifest 缺失。

### 4.2 Lane B：D4RT cache 加速和多窗口 unblock

目标：解决 96f/128f 多场景 cache 生成失败，不再让 scene0050 成为唯一多窗口证据。

### 4.3 Lane C：ScanNet probe5 算法核心

目标：在不使用 GT 的情况下，让 pure Stream4D 在 `probe5 32f fixed support` 上接近或超过 Stream3D。先不跑 full 312 scenes。

### 4.4 Lane D：动态场景

目标：尽快在 Replica-Dynamic / Dynamic Replica 上建立最小动态 tracking 评估，不再只在静态 ScanNet 上证明 D4RT 价值。

---

## 5. Phase A：Metric Safety and Code Hygiene

### A0. 整体目标

保证后续所有结果可审计：不读 GT 生成 prediction，不误用 oracle，不混淆 recompute/fixed/inherit，不把 diagnostic 当 method。

### A1. 实验假设

$H_A$：当前非 oracle method scripts 没有 GT 泄漏；若添加 manifest 和 oracle guard，可以把虚假指标风险降到可控。

### A2. Codex 必做代码修改

#### A2.1 增加统一 manifest

新增：

```text
Stream3D/tools/prediction_manifest.py
```

所有会写 prediction/TMP/object_dict 的脚本都调用：

```text
write_prediction_manifest(output_config, payload)
```

manifest 必须包含：

```json
{
  "output_config": "...",
  "is_method_result": true,
  "is_diagnostic_only": false,
  "uses_gt": false,
  "gt_usage": "none",
  "source_configs": [],
  "pre_points_policy": "recompute|inherit|fixed_path|self|unknown",
  "support_policy": "own|fixed_32f|scannet|oracle|unknown",
  "command": "...",
  "code_packet_sha256": "...",
  "created_at": "...",
  "notes": "..."
}
```

#### A2.2 给 oracle 工具加硬 guard

修改：

```text
Stream3D/tools/oracle_candidate_upper_bound.py
```

要求：

```text
- 如果 `--output-config` 不包含 `oracle`，直接 raise ValueError。
- manifest 写 uses_gt=true, is_diagnostic_only=true。
- 输出文件目录名必须包含 oracle。
```

#### A2.3 增加 report config scanner

新增：

```text
Stream3D/tools/scan_reportable_configs.py
```

输入若干 config，输出：

```text
method-safe configs
oracle configs
diagnostic-only configs
missing-manifest configs
suspicious configs
```

如果出现：

```text
uses_gt=true 且 is_method_result=true
```

直接 fail。

### A3. 必须运行的检查

```bash
cd Stream3D
python -m py_compile stream4d/*.py tools/*.py tests/*.py
python -m unittest tests.test_stream4d_protocol_fixes
python -m tools.scan_reportable_configs \
  --configs scannet,stream4d_scannet_32f_ioc075_fixmem,... \
  --output outputs/audit/v5_reportable_config_scan.md
python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /path/to/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --configs ... \
  --output outputs/audit/v5_metric_integrity_probe5.md
```

### A4. 必须记录的指标

```text
evaluator_ap_core_equal_by_hash
has_pre_points_load_original
has_pre_points_load_current
gt_files_read_by_rescore
object_dict_pred_alignment_mean_iou
object_dict_pred_alignment_min_iou
alignment_failed_instances
num_configs_missing_manifest
num_oracle_configs
num_reportable_method_configs
num_suspicious_configs
```

### A5. 判断标准

A lane 通过条件：

```text
1. py_compile 和 unit tests 通过。
2. 非 oracle method configs 全部 `uses_gt=false`。
3. oracle configs 不能被标记为 method result。
4. 所有 reportable configs 有 manifest。
5. object_dict-pred alignment 无失败。
6. evaluator AP core hash 在完整工程中与原版 Stream3D 一致。
```

### A6. 不满足时 Codex 先尝试什么

```text
- 如果 evaluator 缺失，先补完整工程路径，不要继续跑 AP。
- 如果 config 缺 manifest，先补 manifest，不要把历史 config 写入主表。
- 如果 oracle output 能被正常 evaluator 评估，先加硬 guard。
- 如果 object_dict alignment 失败，停止所有 rescore/replay，先修 object_id_to_column mapping。
```

### A7. 可视化与输出

```text
outputs/audit/v5_metric_integrity_probe5.md
outputs/audit/v5_reportable_config_scan.md
outputs/audit/v5_manifest_table.csv
outputs/audit/v5_oracle_guard_test.log
```

---

## 6. Phase B：D4RT Cache / Multi-window Infrastructure

### B0. 整体目标

解决当前最大的实验吞吐 blocker：除 scene0050 外，probe5 其它场景没有 96f/128f 多窗口 carrier cache。没有这些 cache，就不能判断 evidence graph / component_densify 是否真的适合多窗口。

### B1. 实验假设

$H_B$：当前多窗口卡住主要是 checkpoint/filesystem/I/O 初始化问题，不是 D4RT 算法本身无法推理。通过 local checkpoint copy、persistent worker、timeout heartbeat，可以稳定生成 probe5 的 96f/128f carrier caches。

### B2. Codex 必做代码修改

#### B2.1 新增 D4RT preflight

新增：

```text
Stream3D/tools/d4rt_preflight.py
```

功能：

```text
1. 打印 checkpoint path、size、sha256 前 8 位。
2. 复制 checkpoint 到 local scratch，可选参数 `--local-ckpt-copy`。
3. 单独 torch.load 并计时。
4. build_model 并 load_state_dict 计时。
5. encode 一个 2-frame fake video，decode 16 queries。
6. 输出 GPU memory、load seconds、decode seconds。
```

#### B2.2 D4RT persistent worker

新增：

```text
Stream3D/stream4d/d4rt_worker.py
```

第一版可以简单：一个 Python process 一次加载 D4RT，按 scene list 依次调用 `run_scannet` 内部函数，不要每个 scene 重复 load checkpoint。

#### B2.3 Cache manifest

每个 carrier cache 写：

```text
carriers_window000.npz
cache_manifest.json
```

manifest 包含：

```text
seq_name
window_id
frame_indices
raw_frame_ids
num_frames
num_carriers
num_target_frames
checkpoint_sha256
clip_frames
query_chunk_size
seconds_d4rt_encode
seconds_d4rt_decode
uv_in01_rate
visibility_rate
confidence_mean
```

### B3. 实验设计

先 probe5：

```text
scene0050_00
scene0011_00
scene0030_00
scene0081_01
scene0591_00
```

每个 scene 生成：

```text
32f existing sanity
96f: max_frames=96, window_size=32, window_stride=16
128f: max_frames=128, window_size=32, window_stride=16
```

不要先跑 full ScanNet。

### B4. 必须记录的指标

```text
cache_success_rate
num_cache_files
num_windows
num_carriers_per_window
uv_in01_rate_mean/p10/p90
visibility_rate_mean/p10/p90
confidence_mean
seconds_load_ckpt
seconds_encode
seconds_decode
GPU memory peak
num_D_state_timeouts
num_zero_byte_logs
```

### B5. 判断标准

B lane 通过条件：

```text
1. probe5 中至少 5/5 scenes 生成 96f cache。
2. 至少 3/5 scenes 生成 128f cache。
3. 每个 run log 非 0 bytes。
4. 没有 D-state 卡死；若卡死，timeout 自动记录并退出。
5. cache manifest 完整。
```

### B6. 不满足时 Codex 先尝试什么

```text
- 如果 torch.load 卡住：先 cp checkpoint 到 /tmp 或 local scratch，再从 local load。
- 如果 build_model 卡住：单独 preflight，不进入 ScanNet loader。
- 如果 GPU memory OOM：降低 query_chunk_size，不降算法窗口。
- 如果 IO 卡在读取 images/depth：先只读 frame list，打印每帧路径和耗时。
- 如果 128f 慢：先只生成 96f，保证多场景 evidence graph 能跑。
```

### B7. 可视化

```text
outputs/cache_audit/probe5_cache_timeline.png
outputs/cache_audit/uv_in01_by_scene.png
outputs/cache_audit/visibility_by_scene.png
outputs/cache_audit/cache_success_table.md
```

---

## 7. Phase C：ScanNet Probe5 Algorithm Core

### C0. 整体目标

在 probe5 fixed 32f support 上，减少与 Stream3D 的差距。当前目标不是立刻 full ScanNet，而是先在多场景小集上证明 v5 object formation 方向有效。

当前 baseline：

```text
Stream3D on 32f support probe5:
0.399213 / 0.597171 / 0.742535

Current best pure Stream4D probe5 fixed:
0.281615 / 0.497583 / 0.690254
```

v5 第一阶段目标：

```text
AP   >= 0.32
AP50 >= 0.53
AP25 >= 0.70
#pred <= 300
union in target >= 0.94
```

v5 强目标：

```text
AP   >= 0.36
AP50 >= 0.57
AP25 >= 0.73
#pred <= 220
```

超过 Stream3D 的目标：

```text
AP   > 0.399213
AP50 > 0.597171
AP25 > 0.742535
```

### C1. C-Exp0：固定基线重跑

#### 假设

$H_{C0}$：所有后续改进必须在同一 probe5 fixed 32f support 上比较，否则不可解释。

#### 实验

重跑并保存以下表：

```text
Stream3D self-inherit probe5
Stream3D on 32f support probe5
Stream4D 32f self probe5
v3 adaptive on 32f probe5
v4.1 current best pure Stream4D on 32f probe5
```

#### 必须记录

```text
AP/AP50/AP25
union in target
#pred
mean mask area
median mask area
duplicate prediction rate
best-pred-per-GT diagnostic, GT-read-only, not method
```

#### 成功标准

指标必须复现到误差小于：

```text
AP absolute delta < 0.002
```

如果不满足，停止 C lane，先修复数据路径和 support config。

### C2. C-Exp1：MaskObservation Bank v2

#### 假设

$H_{C1}$：当前错误来自 object 形成阶段信息不足。把每个 mask observation 的边界、carrier、3D、appearance、temporal 统计显式化，能让后续 proposal 生成更可靠。

#### 代码修改

新增或扩展：

```text
Stream3D/tools/export_mask_observation_bank.py
Stream3D/stream4d/mask_observation_bank.py
```

每个 observation 一行 JSONL：

```json
{
  "scene": "scene0050_00",
  "frame_id": 120,
  "mask_id": 15,
  "mask_area_2d": 2300,
  "carrier_count": 18,
  "carrier_weight_sum": 12.4,
  "carrier_visibility_mean": 0.74,
  "carrier_boundary_distance_mean": 0.31,
  "projected_3d_count": 850,
  "projected_3d_centroid": [0.1, 1.2, 2.3],
  "projected_3d_extent": [0.4, 0.8, 0.5],
  "depth_valid_ratio": 0.92,
  "temporal_span": 3,
  "appearance_feature_type": "rgb_hist|dino|clip|none",
  "appearance_valid": true
}
```

#### 指标

```text
num_observations
num_observations_with_carriers
carrier_count histogram
boundary_distance histogram
projected_3d_count histogram
appearance_valid_rate
observation temporal span
```

#### 判断标准

通过条件：

```text
1. probe5 5/5 scenes 输出 observation bank。
2. 每个 observation 有完整 manifest。
3. 不读取 GT。
4. 至少 80% existing object candidates 能追溯到 observation bank。
```

#### 可视化

```text
每个 scene 10 帧：RGB + mask + carrier seeds + boundary heatmap。
每个 object：top evidence observations montage。
```

### C3. C-Exp2：Boundary-Aware Proposal v2

#### 假设

$H_{C2}$：当前 high-confidence layer 有效，是因为 boundary-refined core 更干净。若在 proposal 生成阶段就区分 core/fringe，而不是 final fusion，可以提高 AP/AP50 并减少 #pred。

#### 方法

对每个 potential object component，先构造：

```text
core observations: 高 carrier support、高 boundary distance、高 temporal stability。
fringe observations: 与 core 3D 连通、boundary 不太差、但 evidence 弱。
reject observations: same-frame conflict、低 depth validity、与 core 3D extent 不兼容。
```

observation 质量：

$$
q(o)=\alpha_c \log(1+n_c(o)) + \alpha_b d_b(o)+\alpha_v v(o)+\alpha_d d_{valid}(o)+\alpha_t t(o)-\alpha_x x(o)
$$

object support 点分数：

$$
S_O(p)=\max_{o\in core(O)} S_o(p) + \lambda_f \max_{o\in fringe(O)} S_o(p)-\lambda_x X_O(p)
$$

第一版不要训练 $\alpha$，用固定归一化。

#### 实验组

在 probe5 32f cache 上跑：

```text
P0 current best pure Stream4D
P1 boundary high only
P2 boundary high + naive fringe
P3 boundary high + component-connected fringe
P4 P3 + 3D compactness gate
P5 P4 + same-frame exclusivity gate
```

#### 必须记录

```text
AP/AP50/AP25 on 32f fixed support
own recompute AP/AP50/AP25
union in target
#pred
core point count
fringe point count
fringe/core ratio
conflict point rate
mean object compactness
median object compactness
same-frame conflict count
```

#### 判断标准

通过标准：

```text
P4 或 P5 相比 current best:
AP   +0.03 以上
AP50 +0.03 以上
AP25 不下降超过 0.02
#pred 不增加超过 10%
```

强标准：

```text
AP >= 0.32, AP50 >= 0.53, AP25 >= 0.70
```

#### 不满足时 Codex 尝试

```text
- 如果 AP25 掉太多：fringe 太少，降低 compactness gate，但只允许 fringe 接入 core 的同一 3D connected component。
- 如果 AP/AP50 掉：fringe 污染，增加 boundary threshold 或 same-frame exclusivity。
- 如果 #pred 增多：不要最后 NMS，回到 graph component formation，看是否 over-split。
```

#### 可视化

```text
object core/fringe/reject 三色 mesh overlay
2D mask view montage：green core, blue fringe, red reject
每个 scene 前 20 个 high-error object 的 observation evidence table
```

### C4. C-Exp3：Typed Evidence Graph v2

#### 假设

$H_{C3}$：当前 evidence graph 的 hard cannot-link 太粗，导致过分割与欠合并并存。typed edges 能在保留强反证的同时允许同一 object 的互补 mask 合并。

#### 方法

边类型：

```text
positive_track: shared carriers across frames。
positive_mask_overlap: compatible mask support。
positive_complement: same-frame disjoint but 3D/temporal compatible。
negative_conflict: same-frame overlapping masks competing for same carrier/support。
negative_ownership: same support point claimed by incompatible components。
weak_bridge: low-coverage node only as bridge, not as core.
```

边分数：

$$
S_{ij}=S^+_{ij}-\beta S^-_{ij}
$$

但不能把 $S^-_{ij}$ 只当 threshold；要记录并进入 component quality。

#### 实验组

```text
G0 current evidence graph
G1 typed positive + hard conflict
G2 typed positive + soft conflict penalty
G3 G2 + complement edge
G4 G3 + weak bridge node weighting
G5 G4 + split candidate quarantine
```

#### 必须记录

```text
num_nodes
num_edges_by_type
accepted_edges_by_type
rejected_edges_by_type
num_components
kept_components
dropped_components
component_size_histogram
mean_component_quality
num_split_candidates
num_quarantined_nodes
AP/AP50/AP25
#pred
union in target
```

#### 判断标准

通过标准：

```text
G3/G4 比 current best AP +0.03，且 #pred 不超过 current best。
不能只提高 AP25 而 AP 下降。
```

若失败：

```text
- complement edge 过合并：提高 3D extent compatibility。
- soft conflict 太弱：对 same-frame support overlap > 0.5 仍用 hard negative。
- weak bridge 伤 AP：允许 weak node 只参与 connectivity，不参与 final support。
```

### C5. C-Exp4：Split/Merge-Capable Object Memory

#### 假设

$H_{C4}$：当前 object memory 只能 match/create，错误一旦进入 object 会滚雪球。允许 pending/split/quarantine 可以减少碎片和重复。

#### 方法

object lifecycle：

```text
active
lost
reactivated
pending
split_candidate
merge_candidate
quarantined
```

新的 update 规则：

```text
1. 新 window 的 proposal 不立即写入 object。
2. 如果同一 object 在同一 window 出现多个互相 conflict 的 proposal，进入 split_candidate。
3. pending evidence 需要至少下一个 window 复核，或者满足高质量条件才 commit。
4. object merge 必须满足 typed graph positive + low conflict + 3D extent compatibility。
```

matching score：

$$
S(O,P)=w_c S_c+w_a S_a+w_g S_g+w_m S_m+w_b S_b-w_x S_x
$$

其中：

```text
S_c: carrier / tracklet overlap。
S_a: appearance valid similarity；missing feature 不贡献正分。
S_g: 3D centroid/extent compatibility。
S_m: D4RT motion continuity，动态场景优先。
S_b: boundary evidence compatibility。
S_x: same-frame and support conflict。
```

#### 实验

先在 scene0050、scene0011、scene0030 跑 96f；如果 B lane 还没完成，就先 scene0050 96/128f + probe5 32f replay。

对比：

```text
M0 old memory
M1 memory-v2 current
M2 split/merge memory without pending
M3 split/merge memory with pending
M4 M3 + typed graph edge input
```

#### 必须记录

```text
AP/AP50/AP25
num_objects_final
num_active/lost/reactivated/pending/split/quarantine
num_matches
num_creates
num_splits
num_merges
object_fragmentation_proxy
same-frame conflict per object
object support growth curve
per-window object count curve
```

#### 判断标准

通过标准：

```text
96f AP50 不低于 32f current，且 AP 不下降超过 0.03。
object count 不超过 32f 的 1.5x。
split/quarantine 能减少 conflict，不只是藏掉 object。
```

如果失败：

```text
- object count 暴涨：提高 create threshold，不要提高 match threshold。
- AP25 低：pending 太保守，允许 low-confidence fringe 不计入 high-confidence score。
- AP/AP50 低：commit 的 object 边界差，回到 C-Exp2 的 proposal support。
```

### C6. C-Exp5：GT-read-only diagnostic，不作为方法

#### 目标

用 GT 只回答：candidate pool 是否有足够的高 IoU 候选。不要生成 method prediction。

#### 规则

```text
只能使用 `oracle_candidate_upper_bound.py`。
输出 config 必须含 oracle。
主表禁止使用。
```

#### 指标

```text
mean best IoU per GT
GT covered at IoU >= 0.25/0.5/0.75/0.9
oracle one-to-one selected count
Stream4D candidate pool oracle AP
Stream3D candidate pool oracle AP
```

#### 判断

如果 Stream4D oracle 仍明显低于 Stream3D oracle，说明 candidate generation 本身缺高 IoU mask；优先做 boundary-aware proposal。

如果 Stream4D oracle 接近 Stream3D oracle，但 actual AP 低，说明 candidate selection/assignment 问题；优先做 object memory / quality calibration。

---

## 8. Phase D：Replica-Dynamic / Dynamic Replica

### D0. 整体目标

ScanNet 是静态 benchmark，不能证明 semantic 4D reconstruction and tracking。v5 必须在动态场景上跑最小实验，哪怕先是 3 到 5 个 sequences。

### D1. 数据检查

先运行：

```bash
python -m tools.check_dynamic_replica_env \
  --root data/dynamic-replica/v2 \
  --output outputs/audit/dynamic_replica_env_v5.md
```

需要确认：

```text
RGB frames
depth frames
camera poses / intrinsics
object instance masks
object IDs over time
3D trajectories / point tracks
semantic labels
```

### D2. 如果有 GT object IDs / instance masks

#### 假设

$H_D$：D4RT carrier-based object memory 在动态 object 上应优于 Stream3D-style static 3D overlap merging，特别是在移动、遮挡、重现和交叉运动时。

#### 实验组

```text
D0 Stream3D-style overlap baseline, if runnable。
D1 Stream4D 32f current。
D2 Evidence graph + boundary proposal。
D3 Split/merge object memory。
D4 D3 + D4RT geometry Sim3 evaluation adapter。
```

#### 指标

```text
tracking AP or mAP, if dataset supports
IDF1
MOTA / MOTP, if object tracks available
ID switches
fragmentation
reactivation success rate
time-consistent 4D IoU
per-frame instance AP
object trajectory endpoint error
occlusion reappearance accuracy
```

4D object IoU 可定义为：

$$
IoU_{4D}(P,G)=\frac{\sum_t |P_t\cap G_t|}{\sum_t |P_t\cup G_t|}
$$

ID switch rate：

$$
IDSW = \frac{\#\text{identity switches}}{\#\text{matched object time steps}}
$$

#### 成功标准

```text
1. IDF1 比 overlap baseline 提升至少 +5。
2. ID switches 降低至少 20%。
3. 4D IoU 提升至少 +3。
4. 可视化中 moving object 跨遮挡保持同一 ID。
```

### D3. 如果没有 GT object IDs

不能编造 official IDF1 / semantic AP。只能报告：

```text
pseudo consistency
qualitative tracking
D4RT cycle consistency
object persistence curves
manual visual panels
```

pseudo metrics：

```text
carrier identity consistency
temporal mask stability
track fragmentation proxy
object reactivation count
same-object appearance consistency
```

### D4. Sim3 规则

D4RT geometry 不进方法内部对齐。Sim3 只在 evaluation/export adapter：

$$
T^*=\arg\min_{s,R,t}\sum_i \omega_i \|sRx_i^{D4RT}+t-x_i^{eval}\|_2^2
$$

记录：

```text
sim3_anchor_count
sim3_inlier_ratio
sim3_scale
sim3_residual_mean
sim3_residual_p90
```

禁止：

```text
把 Sim3 后的坐标反馈给 object grouping / memory / selection。
```

### D5. 可视化

每个 dynamic sequence 保存：

```text
RGB video with object IDs overlay
D4RT carrier tracks colored by object
3D/4D object trajectory visualization
ID switch timeline
occlusion/reappearance panels
failure cases: merge, split, lost, hallucinated object
```

---

## 9. Full ScanNet final gate

只有当 probe5 通过以下门槛，才跑 full final：

```text
AP >= 0.32
AP50 >= 0.53
AP25 >= 0.70
#pred <= 300
manifest and metric audit pass
no oracle config in reportable set
```

Full final 需要固定参数，不得在 final 上继续搜索。

### Full final 必须报告

```text
Stream3D original/self-inherit
Stream3D on Stream4D support
Stream4D recompute own support, diagnostic only
Stream4D on 32f fixed support
Stream4D on Stream3D support, diagnostic
v5 selected config on locked final split
```

### 判断标准

主 claim 的最低标准：

```text
在 Stream3D-style evaluator 下，locked final AP/AP50/AP25 超过 Stream3D baseline。
同时 fixed/inherit diagnostic 不能灾难性低。
必须报告 pre_points ratio、union in target、#pred。
```

更强 claim：

```text
fixed support 下接近或超过 Stream3D。
```

如果 ScanNet 只 comparable，但 Dynamic Replica 明显优于 Stream3D/overlap baseline，论文主线应改成：

```text
Stream4D is not just a better static ScanNet segmenter; it is a feed-forward semantic 4D reconstruction and tracking framework. Static ScanNet is comparable, while dynamic tracking is the primary advantage.
```

---

## 10. 每次实验必须保存的内容

每次运行必须保存：

```text
1. command.sh
2. config.json
3. manifest.json
4. metrics.csv
5. summary.json
6. audit.md
7. visualization index.html or markdown
8. failure_cases.json
```

每个 summary 至少包含：

```text
AP/AP50/AP25
support_policy
pre_points_policy
union in target
prediction union ratio
#pred
mean mask area
median mask area
conflict rate
duplicate proxy
coverage proxy
runtime
cache source
uses_gt
```

---

## 11. 下一步 48 小时并行计划

### Day 0：立即执行

```text
A lane:
- 加 oracle guard。
- 加 manifest。
- 跑 metric integrity。

B lane:
- 写 d4rt_preflight。
- 尝试 local checkpoint copy。
- 跑 scene0011_00 2-frame / 32f smoke。

C lane:
- 重跑 probe5 fixed baseline table。
- 导出 MaskObservation Bank。

D lane:
- 跑 Dynamic Replica env checker。
```

### Day 1：算法并行

```text
C1:
- Boundary-aware Proposal v2：core/fringe/reject。

C2:
- Typed Evidence Graph v2：positive/complement/conflict/weak bridge。

B:
- probe5 96f cache generation。

D:
- 如果数据可用，跑 1 个 dynamic smoke sequence。
```

### Day 2：决策

```text
如果 probe5 AP < 0.32:
  继续 full ScanNet 没意义，回到 proposal generation。

如果 probe5 AP >= 0.32 且 AP50 >= 0.53:
  跑 tune30 / final split 小规模。

如果 Dynamic Replica 有明显 tracking improvement:
  论文主线转向 dynamic 4D tracking。
```

---

## 12. 禁止 claim

```text
禁止写：Stream4D v4.1 已在 inherit/fixed support 下超过 Stream3D。
禁止写：scene0050 的 96f/128f 结果代表多场景趋势。
禁止写：当前 ScanNet 结果证明 dynamic semantic 4D reconstruction。
禁止写：当前是 D4RT-native geometry 主结果。
禁止把 oracle config 放进主表。
禁止隐藏 `Stream3D on Stream4D support` 强于 Stream4D 的诊断。
禁止只报告 own recompute 而不报告 fixed/inherit。
禁止把 Dynamic Replica pseudo consistency 写成 official IDF1 / semantic AP。
```

---

## 13. 最终判断

v4.1 最大价值不是“已经接近投稿结果”，而是**把问题定位清楚了**：

```text
Stream4D 的瓶颈不是单纯 coverage，也不是最后删几个重复预测。
瓶颈是 object proposal formation：大量低置信候选包含有用 recall，但当前无监督方法不能把它们合成少量、高质量、一对一实例。
```

v5 必须以此为核心，围绕 observation-level evidence、typed graph、boundary-aware proposal 和 split/merge memory 设计实验。只有这条线在 probe5 fixed support 上过门槛，才值得跑 full ScanNet；只有 Dynamic Replica 跑通，才有资格 claim feed-forward semantic 4D reconstruction and tracking。

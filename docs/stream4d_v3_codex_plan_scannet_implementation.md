# Stream4D v3 ScanNet 复现实验实现说明

本文说明我为了执行 `docs/stream4d_v3_protocol_corrected_plan_for_codex.md` 做了什么。目标读者是不熟悉这个项目的人，所以我会先解释必要概念，再写代码改动、执行命令、输出文件和实验结论。

## 1. 这次任务要解决什么问题

原始任务是复现 Stream3D 论文里 ScanNet 的结果，并检查当前 Stream4D 结果是否真的比原版 Stream3D 好。

这里有一个容易误解的地方：Stream3D 的评价不是在整张三维网格上直接评价，而是先选出一批要评价的点，这批点保存在 `_pre_points.npy` 文件里。评价程序只看这些点。

所以同一个预测结果，如果换一批评价点，分数可能完全不同。v3 计划要求我把这个问题查清楚。

这次最终回答了两个问题：

1. 当前代码有没有使用和原版 Stream3D 一致的评价核心逻辑。
2. 当前 Stream4D 超过 Stream3D 的结果，是不是依赖缩小了评价点范围。

## 2. 几个关键概念

### 2.1 ScanNet

ScanNet 是一个室内三维场景数据集。一个场景可以想象成一间屋子的三维网格。网格上有很多点，每个点可能属于椅子、桌子、墙、地板等物体。

本项目里，ScanNet 验证集场景列表在：

```text
Stream3D/splits/scannet.txt
```

本轮使用了 312 个场景。

### 2.2 预测文件

每个场景的预测结果保存在 `.npz` 文件里。可以把它理解成一个压缩包，里面主要有三类数组：

```text
pred_masks
pred_score
pred_classes
```

含义如下：

```text
pred_masks: 每个预测物体覆盖了哪些三维点
pred_score: 每个预测物体的置信分数
pred_classes: 每个预测物体的类别
```

例如某个配置的预测目录是：

```text
Stream3D/data/prediction/stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute_class_agnostic
```

其中一个场景可能是：

```text
scene0050_00.npz
```

### 2.3 评价点集 pre_points

`pre_points` 是一个整数数组，里面每个数字是三维网格中的一个点编号。

例如：

```text
Stream3D/data/TMP/<配置名>/scene0050_00_pre_points.npy
```

评价程序会做类似这样的事情：

```python
gt_ids = gt_ids[pre_points]
pred_masks = pred_masks[pre_points]
```

这表示只在 `pre_points` 指定的点上比较真实答案和预测结果。

如果 `pre_points` 覆盖了很多点，评价会更接近完整场景。

如果 `pre_points` 覆盖很少的点，评价会变成“只在模型实际碰到的一小块地方评价”。

### 2.4 prediction union

`prediction union` 是当前所有预测物体覆盖点的并集。

用普通话说，就是：

```text
把当前保留下来的所有预测物体覆盖的点合在一起，去掉重复点，得到一批点。
```

代码里是：

```python
prediction_union = np.flatnonzero(pred_masks.any(axis=1))
```

如果一个后处理步骤删掉了很多预测物体，那么 `prediction union` 通常会变小。

### 2.5 recompute_pre_points

`recompute_pre_points` 的意思是：

```text
后处理筛完预测物体以后，重新用当前 prediction union 生成评价点集。
```

这和原版 Stream3D 的主评估方式一致：每个配置使用自己的预测结果和自己的 TMP 评价点文件。

这次和原版 Stream3D 一致的主结果是：

```text
stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute
20.3718 / 35.5222 / 55.0649
```

### 2.6 inherit_pre_points

`inherit_pre_points` 是我按照 v3 计划新增的诊断方式，不是原版 Stream3D 的主评估方式。

它的意思是：

```text
后处理筛完预测物体以后，不重新缩小评价点集，而是继承输入配置原本的 pre_points。
```

这样可以检查一个问题：

```text
如果评价范围不跟着预测物体一起缩小，当前方法还好吗？
```

结果表明不好：

```text
adaptive top-k recompute: 20.3718 / 35.5222 / 55.0649
adaptive top-k inherit:   12.2851 / 23.3147 / 41.6773
```

所以当前超过原版 Stream3D 的主结果是真实跑出来的，但它依赖原版那种每个配置使用自己 TMP 的评价方式。

## 3. 我改了哪些代码

### 3.1 `Stream3D/stream4d/rescore_scannet.py`

这个脚本的作用是：

```text
读取已有的 Stream4D 预测结果；
根据一些规则删掉一部分预测物体；
写出新的预测文件；
写出新的 pre_points 文件；
写出一份 summary 方便审计。
```

我加了这些命令行参数：

```text
--pre-points-policy recompute|inherit|fixed_path
--fixed-pre-points-root
--fixed-pre-points-config
```

三个 policy 的意思是：

```text
recompute:
  用当前保留下来的预测物体覆盖点重新生成 pre_points。

inherit:
  沿用输入配置的 pre_points。

fixed_path:
  从用户指定目录读取固定 pre_points。
```

我还让 summary 多记录这些字段：

```text
pre_points_policy
input_pre_points_path
input_pre_points_count
output_pre_points_count
prediction_union_count
pre_points_equals_prediction_union
prediction_union_subset_of_pre_points
```

这些字段用于回答：

```text
这个实验用的是哪种 pre_points 策略？
输入评价点有多少？
输出评价点有多少？
当前预测实际覆盖多少点？
输出评价点是否刚好等于预测覆盖点？
预测覆盖点是否完全包含在评价点里？
```

关键实现位置：

```python
def _pre_points_input_path(args: argparse.Namespace, seq_name: str) -> Path:
    if args.pre_points_policy == "fixed_path":
        return Path(args.fixed_pre_points_root) / args.fixed_pre_points_config / f"{seq_name}_pre_points.npy"
    return Path("data/TMP") / args.input_config / f"{seq_name}_pre_points.npy"
```

```python
def _write_pre_points(...):
    input_pre_points = np.load(tmp_in).astype(np.int64)
    prediction_union = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
    if args.pre_points_policy == "recompute":
        output_pre_points = prediction_union
    elif args.pre_points_policy in {"inherit", "fixed_path"}:
        output_pre_points = input_pre_points
    ...
```

为什么这个修改合理：

```text
原来脚本只能默认重新生成 pre_points。
v3 计划要求同一个预测结果同时比较 recompute 和 inherit。
所以必须把 pre_points 策略变成命令行参数。
```

### 3.2 `Stream3D/stream4d/local_4d_filter.py`

这个文件里有一个本地筛选器。它会根据一批二维 mask 观测，选择更可靠的候选物体。

原来的问题是权重可能对错 carrier。

原来代码的思路类似：

```python
for key, weight in zip(support, obs.weights.tolist()):
    ...
```

这里 `support` 是一个 set。set 没有可靠顺序。

但是 `obs.weights` 的顺序是和 `obs.carrier_ids` 对齐的。

举例：

```text
carrier_ids = [303, 101, 202]
weights     = [0.3, 0.1, 0.2]
```

这表示：

```text
303 的权重是 0.3
101 的权重是 0.1
202 的权重是 0.2
```

如果先把 carrier 放进 set，再和 weights 配对，顺序可能变成：

```text
support = {101, 202, 303}
```

这时权重可能会错配。

修复后，直接按 `carrier_ids` 和 `weights` 的原始顺序配对：

```python
for cid, weight in zip(obs.carrier_ids.tolist(), obs.weights.tolist()):
    key = (int(obs.frame_id), int(cid))
    weights[key] = max(weights.get(key, 0.0), float(weight))
```

我还抽出了一个小函数：

```python
@staticmethod
def _carrier_weights(observations: list[MaskObservation]) -> dict[tuple[int, int], float]:
    ...
```

这样测试可以直接检查权重是否对齐。

为什么这个修改合理：

```text
carrier_ids 和 weights 本来就是一一对应的数组。
不应该让 set 的随机顺序参与权重对应关系。
```

### 3.3 `Stream3D/stream4d/carrier_sampler.py`

这个文件负责从二维 mask 里采样 carrier 点。

潜在问题是：

```text
如果一个 mask 只有 2 个像素，但参数要求最少采 4 个点，实际只能采到 2 个。
```

原代码有些伴随字段仍可能按请求数量生成，而不是按实际采样数量生成。

我加入：

```python
actual_count = int(keep.shape[0])
```

然后所有伴随字段都使用 `actual_count`。

修复后这些数组长度保持一致：

```text
carrier_id
src_frame
src_frame_global
src_xy
src_uv
src_mask_id
```

为什么这个修改合理：

```text
一个 carrier 点应该在所有字段里都有一条对应记录。
如果某些字段长度不同，后面保存、读取、匹配时可能出错。
```

### 3.4 `Stream3D/tools/audit_stream3d_eval_protocol.py`

这是新增的审计工具。

它做四件事：

1. 读取原始 Stream3D 的 `evaluation/evaluate.py`。
2. 读取当前项目的 `evaluation/evaluate.py`。
3. 比较核心评价函数的哈希值。
4. 统计每个配置、每个场景的评价点数量和预测覆盖点数量。

它重点比较这两个函数：

```text
evaluate_matches
compute_averages
```

如果这两个函数完全一样，说明平均精度指标的核心计算逻辑没有被改。

它输出：

```text
outputs/audit/eval_protocol_audit.md
outputs/audit/eval_protocol_audit.json
outputs/audit/eval_protocol_audit_pre_points_ratio.png
outputs/audit/eval_protocol_audit_prediction_union_ratio.png
```

它统计的字段包括：

```text
scene_id
num_scene_vertices
num_pre_points
pre_points_ratio
num_prediction_union
prediction_union_ratio
pre_points_equals_prediction_union
prediction_union_subset_of_pre_points
num_pred_instances
num_gt_instances_in_pre_points
num_gt_instances_fullmesh
```

为什么这个工具必要：

```text
只看最终分数不够。
必须知道评价点集到底有多大，预测实际覆盖了多少点，真实物体有多少进入了评价范围。
```

### 3.5 `Stream3D/tools/make_scannet_stream4d_splits.py`

这是新增的 split 工具。

它做一件事：

```text
把 splits/scannet.txt 中的 312 个场景用固定随机种子打乱，
一半写到 scannet_tune.txt，
另一半写到 scannet_final.txt。
```

本轮使用：

```text
seed = 20260607
```

输出：

```text
Stream3D/splits/scannet_tune.txt   156 个场景
Stream3D/splits/scannet_final.txt  156 个场景
```

为什么要固定随机种子：

```text
以后别人重新生成 split，会得到完全相同的 tune/final 场景列表。
```

### 3.6 `Stream3D/tools/materialize_scannet_eval_subset.py`

原本的评价脚本会评价一个预测目录里的所有 `.npz` 文件。

如果想只评价 tune 或 final 的 156 个场景，就需要创建只包含这些场景的新目录。

这个工具会：

```text
读取一个 split 文件；
为 split 里的每个场景创建预测文件软链接；
为 split 里的每个场景创建 pre_points 文件软链接；
输出一个新的配置目录。
```

它默认用软链接，不复制大文件。

为什么这个工具必要：

```text
可以不改原始 evaluator，也能让它只评价 tune 或 final 子集。
```

### 3.7 `Stream3D/tests/test_stream4d_protocol_fixes.py`

新增两个测试。

第一个测试检查：

```text
carrier 权重是否按照 carrier_id 原始顺序对齐。
```

第二个测试检查：

```text
当 mask 实际像素数少于请求采样数时，所有 carrier 字段长度是否一致。
```

测试命令：

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_stream4d_protocol_fixes
```

实际结果：

```text
Ran 2 tests ... OK
```

## 4. 我怎么运行实验

### 4.1 环境

用户要求使用 conda 环境：

```text
loger
```

当前 shell 里直接输入 `conda` 找不到命令，所以我使用绝对路径：

```text
/mnt/data/users/chengshun.wang/miniconda3/bin/conda
```

也直接使用了环境内 Python：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
```

### 4.2 原始 Stream3D 代码

计划中写的路径是：

```text
/mnt/data/orig_stream3d/Code_Stream3D
```

这个目录没有写权限，所以我没有在那里解压。

我找到本机已有 zip：

```text
/mnt/data/users/chengshun.wang/pjs/Code_Stream3D.zip
```

然后解压到：

```text
/mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D
```

命令：

```bash
mkdir -p /mnt/data/users/chengshun.wang/pjs/orig_stream3d
unzip -q -o /mnt/data/users/chengshun.wang/pjs/Code_Stream3D.zip -d /mnt/data/users/chengshun.wang/pjs/orig_stream3d
```

### 4.3 审计 evaluator 是否和原版一致

命令：

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n loger python -m tools.audit_stream3d_eval_protocol \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --configs scannet,stream4d_scannet_32f_ioc075_fixmem,stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one \
  --seq-list splits/scannet.txt \
  --output outputs/audit/eval_protocol_audit.md \
  2>&1 | tee logs/stream4d_v3_s0_eval_protocol_audit.log
```

重要结果：

```text
原始 evaluate.py 存在: True
当前 evaluate.py 存在: True
原始 evaluate.py 读取 pre_points: True
当前 evaluate.py 读取 pre_points: True
evaluate_matches 函数哈希一致: True
compute_averages 函数哈希一致: True
```

结论：

```text
当前主评价程序的平均精度核心计算逻辑和原版 Stream3D 一致。
```

### 4.4 生成 adaptive top-k recompute 结果

这里的 adaptive top-k 是：

```text
根据每个场景原本有多少候选物体，按比例保留一部分；
至少保留 8 个；
最多保留 18 个；
排序依据是 mask_count，也就是一个候选物体被多少二维 mask 支持。
```

命令：

```bash
CFG=stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n loger python -m stream4d.rescore_scannet \
  --seq-list splits/scannet.txt --backbone Cropformer \
  --input-config stream4d_scannet_32f_ioc075_fixmem --output-config "$CFG" \
  --score-mode one --select-mode mask_count \
  --filter-max-instances-ratio 0.14 --filter-min-instances 8 --filter-max-instances 18 \
  --pre-points-policy recompute --debug-root outputs/stream4d_rescore_v3 \
  2>&1 | tee logs/${CFG}_rescore.log
```

评价命令：

```bash
CFG=stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/${CFG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet --no_class --tmp_config ${CFG} \
  --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
  2>&1 | tee logs/${CFG}_evaluate.log
```

结果：

```text
20.3718 / 35.5222 / 55.0649
```

这就是和原版 Stream3D 主评估方式一致的 Stream4D v3 主结果。

### 4.5 生成 adaptive top-k inherit 诊断结果

命令和上面基本一样，只改：

```text
--pre-points-policy inherit
```

配置名：

```text
stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit
```

结果：

```text
12.2851 / 23.3147 / 41.6773
```

这个不是原版 Stream3D 主评估方式，而是额外诊断。

### 4.6 生成 min250 对照结果

min250 的意思是：

```text
只保留覆盖点数至少 250 的候选物体。
```

recompute 结果：

```text
18.9498 / 32.5721 / 53.0585
```

inherit 结果：

```text
9.9167 / 18.9180 / 36.6473
```

这个对照说明：

```text
只靠删除小候选物体可以改善 recompute 分数，
但在 inherit 诊断下仍然失败。
```

### 4.7 生成 tune/final split

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.make_scannet_stream4d_splits \
  --input splits/scannet.txt \
  --tune-output splits/scannet_tune.txt \
  --final-output splits/scannet_final.txt \
  --seed 20260607 \
  2>&1 | tee logs/stream4d_v3_make_scannet_splits.log
```

结果：

```text
tune: 156 scenes
final: 156 scenes
```

### 4.8 让原 evaluator 评价 tune/final

先创建 split 子目录：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
for SPLIT in tune final; do
  for CFG in scannet stream4d_scannet_32f_ioc075_fixmem stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit; do
    OUT=${CFG}_v3_${SPLIT}
    $PY -m tools.materialize_scannet_eval_subset --root . --config "$CFG" --seq-list splits/scannet_${SPLIT}.txt --output-config "$OUT"
  done
done 2>&1 | tee logs/stream4d_v3_materialize_tune_final.log
```

再用原 evaluator 模板评价这些新目录。

结果：

```text
scannet_v3_tune:
20.8394 / 35.5954 / 50.8577

Stream4D MVP tune:
13.1056 / 24.2241 / 42.6161

adaptive recompute tune:
20.5105 / 35.3847 / 55.1521

adaptive inherit tune:
12.6566 / 23.6953 / 42.1901

scannet_v3_final:
19.4294 / 33.3989 / 49.6361

Stream4D MVP final:
12.4298 / 23.1572 / 41.8428

adaptive recompute final:
20.2401 / 35.6642 / 54.9907

adaptive inherit final:
11.9313 / 22.9523 / 41.1922
```

注意：这里没有做完整的参数搜索，只是把固定好的结果在 tune/final 上做诊断。

## 5. 最终结果怎么理解

### 5.1 和原版 Stream3D 一致的评估结果

如果只关心“哪个评估方式和原版 Stream3D 一致”，应看：

```text
recompute_pre_points / 每个配置自己的 TMP
```

对应结果：

```text
Stream3D-Cropformer baseline:
20.1139 / 34.4654 / 50.2268

Stream4D adaptive top-k recompute:
20.3718 / 35.5222 / 55.0649
```

### 5.2 inherit 诊断说明什么

inherit 结果说明：

```text
如果不让评价点集跟随 top-k 缩小，
同样的预测结果分数会从 20.3718 降到 12.2851。
```

这说明当前方法的强结果主要来自：

```text
删掉大量候选物体后，评价点集也跟着变小。
```

它不说明主结果是假的，因为原版 Stream3D 的主评估方式本来就是每个配置用自己的 TMP。

但它说明这个结论要写得很谨慎：

```text
可以写：在原版 Stream3D 风格的 cropped-TMP evaluator 下超过 baseline。
不能写：Stream4D 已经全面优于 Stream3D。
```

### 5.3 为什么后续不应该继续盲调 ratio

本轮发现：

```text
recompute 超过 baseline；
inherit 没有超过 baseline。
```

v3 计划的决策树说，这种情况下后续应该优先做：

```text
reliable densification
memory-v2
```

用普通话解释：

```text
现在的问题不是“应该保留 14% 还是 16% 的候选物体”。
真正的问题是模型覆盖到的三维点太少。
要提升 inherit 诊断分数，需要让预测物体真正覆盖更多正确点，而不是只调整删除规则。
```

## 6. 审阅代码时建议重点看哪里

### 6.1 `rescore_scannet.py`

重点看：

```text
_pre_points_input_path
_write_pre_points
_process_sequence 返回的 summary 字段
build_parser 里新增的参数
```

审阅问题：

```text
recompute 是否真的使用 prediction union？
inherit 是否真的沿用输入 pre_points？
fixed_path 是否从指定目录读取？
summary 是否足够证明每个 scene 的评价点关系？
```

### 6.2 `audit_stream3d_eval_protocol.py`

重点看：

```text
是否正确找到原始和当前 evaluate.py
是否正确提取 evaluate_matches 和 compute_averages
是否正确读取 prediction、pre_points、ground truth
是否正确统计 pre_points 和 prediction union
```

审阅问题：

```text
baseline 的 TMP 有两种可能位置：
  Stream3D/TMP/scannet
  Stream3D/data/TMP/scannet
工具是否正确 fallback？
```

### 6.3 `local_4d_filter.py`

重点看：

```text
_carrier_weights 是否直接按 carrier_ids 和 weights 配对。
```

审阅问题：

```text
是否完全避免了 set 顺序影响权重？
```

### 6.4 `carrier_sampler.py`

重点看：

```text
actual_count 是否用于所有伴随字段。
```

审阅问题：

```text
当实际 mask 像素数小于请求采样数时，所有输出数组长度是否一致？
```

### 6.5 `tests/test_stream4d_protocol_fixes.py`

重点看：

```text
测试是否覆盖了权重顺序错配。
测试是否覆盖了采样数量不足的边界情况。
```

## 7. 本次审阅包应包含哪些文件

代码文件：

```text
Stream3D/stream4d/rescore_scannet.py
Stream3D/stream4d/local_4d_filter.py
Stream3D/stream4d/carrier_sampler.py
Stream3D/tools/audit_stream3d_eval_protocol.py
Stream3D/tools/make_scannet_stream4d_splits.py
Stream3D/tools/materialize_scannet_eval_subset.py
Stream3D/tests/__init__.py
Stream3D/tests/test_stream4d_protocol_fixes.py
```

说明和证据文件：

```text
docs/stream4d_v3_codex_plan_scannet_implementation.md
docs/stream4d_v3_执行日志.md
docs/stream4d_v3_实验结果复盘.md
docs/stream4d_v3_protocol_corrected_plan_for_codex.md
```

这些文件会打包为：

```text
stream4d_v3_code_review_packet.zip
```

## 8. 没有做什么

本轮没有伪造或补写任何未运行结果。

本轮没有完成完整的 tune grid search，因为 `inherit_pre_points` 已经明确失败，继续大规模调 top-k 参数不能解决核心 coverage 问题。

本轮没有实现 reliable densification、memory-v2、D4RT Sim3 export、Replica-Dynamic 动态实验，因为这些是 v3 计划后续阶段，需要新的实现和更长实验。

本轮没有把 fullmesh evaluator 当作主指标。

## 9. 一句话结论

```text
和原版 Stream3D 一致的评估方式下，当前 Stream4D adaptive top-k recompute 结果是 20.3718 / 35.5222 / 55.0649，超过本地 Stream3D-Cropformer baseline 20.1139 / 34.4654 / 50.2268；但 inherit_pre_points 诊断失败，说明当前优势主要依赖 cropped-TMP 评价点集随预测结果缩小，后续应该提升三维点覆盖能力，而不是继续盲调删除比例。
```

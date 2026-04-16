# 关于LoGeR的TTT

LoGeR 里的 TTT 不是“测试时把整个模型拿去再训练一遍”，而是**嵌入在网络块内部的 fast-weight memory**。视频先被切成 chunk；在每个 hybrid block 里，核心逻辑是：

**per-frame attention → 相邻 chunk 的 SWA → chunk-wise TTT → chunk 内双向 attention**

其中：

- **SWA** 负责相邻 chunk 之间的无损局部对齐；
- **TTT** 负责把长程全局几何压缩进 fast weights，帮助稳住全局坐标系和尺度，减少长序列漂移；
- 论文里还明确说了这个 TTT 是 **apply-then-update**，并且 fast-weight module 用的是 **SwiGLU**，test-time update 用 **Muon**。附录又说明：**TTT 插在 18 个 residual blocks 里，SWA 只插 4 层**。

的**文中 TTT 层对应代码**。

## 1) 真正实现 TTT 层的文件

**文件：** `loger/models/ttt.py`

这里就是论文里 Eq. (5)(6) 那个 TTT 的真正实现。

最关键的几个位置：

- `TTTOperator`
  在 `loger/models/ttt.py:12`
  用来描述 TTT 的执行顺序：哪些 token 段做 `apply`，哪些做 `update`。
- `fast_weight_swish_glu_weight_norm_mini_batch_apply(...)`
  在 `loger/models/ttt.py:76-175`
  这是 **TTT 核心算子**。
  里面：
  - `118-165` 是 **update**
  - `167-170` 是 **apply**
- `FastWeightGluMLPMultihead`
  在 `loger/models/ttt.py:178-318`
  这是论文里那个 **fast-weight module (f_W)** 的代码化版本，也就是你真正可以称为“TTT layer”的类。

### 它和论文公式的对应关系

论文把 TTT 写成一个抽象的 ( f_{W^m}(\cdot) )，代码里具体落地成了一个 **multi-head SwiGLU fast-weight MLP**：

[
(\mathrm{silu}(x W_0)\odot (x W_2))W_1
]

对应代码就在 `ttt.py:93-95` 和 `170` 附近。
也就是说，论文里抽象写成一个 fast weight (W)，但代码里实际不是单个矩阵，而是**每个 head 有三组 fast weights：`w0 / w1 / w2`**。

所以如果你问“TTT 层到底是哪一个 class”，答案就是：

> **`FastWeightGluMLPMultihead`**

------

## 2) TTT 是怎么被插进 LoGeR 主干里的

**文件：** `loger/models/pi3.py`

这里不是实现 fast-weight 更新本身，而是把 TTT 作为 hybrid memory 的一部分插入 Pi3/LoGeR backbone。

最关键的位置：

- **实例化 TTT 层**
  `loger/models/pi3.py:224-237`
  这里创建了：

  ```python
  self.ttt_layers = nn.ModuleList([
      FastWeightGluMLPMultihead(...)
      for _ in self.ttt_insert_after
  ])
  ```

- **定义 apply→update 的顺序**
  `loger/models/pi3.py:248-251`

  ```python
  self.ttt_op_order = [
      TTTOperator(start=0, end=None, update=False, apply=True),
      TTTOperator(start=0, end=None, update=True, apply=False),
  ]
  ```

  这和论文说的 **apply-then-update** 完全对应。

- **真正把 TTT 插到 decoder 流程里**
  `loger/models/pi3.py:410-447`
  这里会：

  1. 取当前层输入 `tokens_in`

  2. 调用 `self.ttt_layers[layer_idx](tokens_in, info)`

  3. 得到 TTT 输出和更新后的 fast weights

  4. 用 gate 做残差注入：

     ```python
     update_term = ttt_output * gate_scale
     tokens_out = update_term + tokens_post
     ```

- **跨 window 维护 TTT state（w0/w1/w2）**
  `loger/models/pi3.py:643-660`
  这里初始化跨 chunk / window 保持的 fast weights：

  ```python
  w0 = [None] * len(self.ttt_insert_after)
  w1 = [None] * len(self.ttt_insert_after)
  w2 = [None] * len(self.ttt_insert_after)
  ```

- **长序列时周期性 reset TTT state**
  `loger/models/pi3.py:655-660` 和 `668-669`
  这对应论文里说的：TTT 虽然理论上无限上下文，但实践里会有长度泛化瓶颈，所以超长序列推理时会做 periodic reset。

- **每个 window 结束后，把更新后的 fast weights 写回去**
  `loger/models/pi3.py:731-735`
  这里把当前 chunk 更新后的 `w0/w1/w2` 存起来，供下一个 chunk 继续用。

------

## 3) 一个很关键的细节：TTT 插在哪里

从 checkpoint 配置看：

- `ckpts/LoGeR/original_config.yaml`
- `ckpts/LoGeR_star/original_config.yaml`

里面都有：

```yaml
ttt_insert_after:
  - 0
  - 2
  - 4
  ...
  - 34
```

也就是 **18 个位置**，正好对应论文附录说的 “TTT 插在 18 个 residual blocks”。

而且这段代码里：

- **偶数层** 是 frame attention
- **奇数层** 是 chunk/global attention

所以 `ttt_insert_after = [0,2,4,...,34]` 的真实含义不是“插在 global attention 后面”，而是：

> **插在每个 frame-attention 后面、下一个 global/chunk attention 前面**

这其实正好对应论文里写的 block 顺序：

> frame attention → SWA / TTT → chunk-wise bidirectional attention

这一点很重要，不然很容易看错。

------

## 4) 把“文中的 TTT 层”直接对应成这几处

最核心的映射我给你浓缩成一句话：

- **论文中的 TTT layer 本体**
  → `loger/models/ttt.py::FastWeightGluMLPMultihead`
- **论文中的 apply / update 机制**
  → `loger/models/ttt.py::fast_weight_swish_glu_weight_norm_mini_batch_apply`
- **论文中的 chunk-wise TTT 插入 backbone**
  → `loger/models/pi3.py:224-251` + `410-447`
- **论文中的跨 chunk 记忆状态 W^m**
  → `pi3.py` 里跨 window 保存的 `w0 / w1 / w2`

------

# 关于TTT的update

定位到了，**真正更新 TTT fast weights 的核心代码**在这里：

## 1) 真正“改写 fast weights”的位置

**文件：** `loger/models/ttt.py`
**函数：** `fast_weight_swish_glu_weight_norm_mini_batch_apply(...)`

最关键的是这段：

- **进入 update 分支**：`loger/models/ttt.py:118`

- **算当前 chunk 的中间量**：`125-134`

- **算三组 fast weights 的梯度**：`136-143`

  - `w1_grad`
  - `w0_grad`
  - `w2_grad`

- **叠加 momentum**：`146-153`

- **真正更新 fast weights**：`155-158`

  ```python
  w1_now = w1_now + w1_grad
  w0_now = w0_now + w0_grad
  w2_now = w2_now + w2_grad
  ```

- **更新后做 weight norm**：`160-163`

- **把更新后的权重写回状态**：`165`

  ```python
  w0, w1, w2 = w0_now, w1_now, w2_now
  ```

所以如果你问：

> “哪几行代码真的在更新 TTT fast weights？”

最直接的答案就是：

> **`loger/models/ttt.py:155-165`**
> 其中 `155-158` 是更新本体，`160-163` 是更新后的归一化，`165` 是写回新的 fast weights。

------

## 2) 谁调用了这段更新逻辑

还是在 **`loger/models/ttt.py`** 里，但更上层一点：

### `FastWeightGluMLPMultihead.forward(...)`

- **读取旧状态 / 若无则用初始化权重**：`286-294`

- **调用核心更新函数**：`296-302`

  ```python
  output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(...)
  ```

- **把更新后的 fast weights 返回出去**：`316-317`

  ```python
  return output, {
      "w0": w0, "w1": w1, "w2": w2,
  }
  ```

这说明：

- `ttt.py` 里的核心函数负责“算怎么更新”
- `FastWeightGluMLPMultihead.forward` 负责“把旧状态喂进去，再把新状态吐出来”

------

## 3) LoGeR 主干里怎么把更新后的 fast weights 往下一个 chunk 传

**文件：** `loger/models/pi3.py`

这里负责 **跨 chunk / window 维护 fast weights 状态**。

### 在 decode/block 内把旧的 fast weights 传进 TTT 层

```
loger/models/pi3.py:425-431
info = {
    "ttt_op_order": ttt_state.get("ttt_op_order", []),
    "w0": ttt_state["w0"][layer_idx],
    "w1": ttt_state["w1"][layer_idx],
    "w2": ttt_state["w2"][layer_idx],
}
ttt_output, output = self.ttt_layers[layer_idx](tokens_in, info)
```

### 把当前层更新后的 fast weights 存起来

```
loger/models/pi3.py:445-447
ttt_output_info["w0"][layer_idx] = output["w0"]
ttt_output_info["w1"][layer_idx] = output["w1"]
ttt_output_info["w2"][layer_idx] = output["w2"]
```

### 一个 window 结束后，把新状态写回全局 TTT state

```
loger/models/pi3.py:733-735
if self.ttt_layers is not None and ttt_output_info is not None:
    w0, w1, w2 = ttt_output_info["w0"], ttt_output_info["w1"], ttt_output_info["w2"]
```

所以完整链条是：

1. `pi3.py` 把旧的 `w0/w1/w2` 传给 TTT layer
2. `ttt.py` 里计算梯度并更新
3. `ttt.py` 返回新的 `w0/w1/w2`
4. `pi3.py` 再把这些新权重保存下来，供**下一 chunk**继续用

------

## 4) apply / update 的顺序也能对上论文

在 `loger/models/pi3.py:248-250`：

```python
self.ttt_op_order = [
    TTTOperator(start=0, end=None, update=False, apply=True),
    TTTOperator(start=0, end=None, update=True, apply=False),
]
```

也就是当前实现里明确指定了：

- 先 **apply**
- 再 **update**

这和论文方法部分写的 **apply-then-update** 是一致的。论文把 chunk-wise TTT 写成：先用 (f_{W^m}) 作用当前 chunk，再由当前 chunk 更新到 (W^{m+1})。

------

## 5) 一个很重要的观察

这里的 fast-weight 更新**不是**通过常规 `optimizer.step()` 做的。

而是**手写在 forward 路径里**：

- 先显式构造 `w0_grad / w1_grad / w2_grad`

- 再直接

  ```python
  w_now = w_now + grad
  ```

- 最后再做一次 norm 约束

也就是说，这是一种**在线 test-time fast-weight update**，不是普通训练参数更新。

------

## 6) 最该盯的几个位置

如果你要继续深挖，我建议按这个顺序看：

1. **`loger/models/ttt.py:118-165`**
   看 fast weights 到底怎么更新

2. **`loger/models/ttt.py:296-317`**
   看更新前后的状态是怎么进出 TTT 层的

3. **`loger/models/pi3.py:425-447`**
   看 block 内如何把 TTT 层接进主干

4. **`loger/models/pi3.py:697-735`**
   看跨 chunk / window 的状态传递

5. **`loger/models/pi3.py:654-660`**
   看 reset 时如何把 fast weights 清空
   这也对应论文里提到的 periodic state reset。

   



# LoGeR* 的overlap-based feedforward alignment 与 reset 机制 	

找到了，而且这部分代码和论文描述是能一一对上的。

论文里说 **LoGeR\*** 用的是“基于 overlap 的纯 feedforward SE(3) alignment”，同时长序列推理时会 **periodic reset TTT fast weights**，并且 **reset 时也做 pose alignment**。这一点在文中写得很明确。

对应代码主要都在：

**`loger/models/pi3.py`**

------

### 1) LoGeR* 开关本身

**文件：** `ckpts/LoGeR_star/original_config.yaml`

这里和普通 LoGeR 的关键区别是：

```yaml
model:
  se3: true
```

也就是 **LoGeR\*** 走的是 **SE(3) overlap alignment** 路径。普通 `LoGeR/original_config.yaml` 没这个开关。

------

### 2) forward 里读取 alignment / reset 相关参数

**位置：** `loger/models/pi3.py:582-593`

这里会读进来：

- `window_size`
- `overlap_size`
- `sim3`
- `se3`
- `reset_every`

其中最关键的是：

- `se3 = kwargs.pop('se3', False)` → LoGeR* 会开
- `reset_every = kwargs.pop('reset_every', 0)` → 控制每多少个 window reset 一次 TTT 状态

------

### 3) reset TTT fast weights 的代码

**位置：** `loger/models/pi3.py:643-669`

这里先准备跨 window 的 TTT 状态：

- `645-647`：初始化 `w0 / w1 / w2`
- `654-660`：定义 `reset_adaptive_states()`
- `668-669`：真正触发 reset

关键代码就是：

```python
def reset_adaptive_states():
    nonlocal w0, w1, w2
    if self.ttt_layers is not None:
        w0 = [None] * len(self.ttt_insert_after)
        w1 = [None] * len(self.ttt_insert_after)
        w2 = [None] * len(self.ttt_insert_after)
```

以及：

```python
if reset_every > 0 and window_idx > 0 and window_idx % reset_every == 0:
    reset_adaptive_states()
```

所以这就是 **“每隔 N 个 windows，把 TTT fast weights 清空”** 的实现。

这正对应论文里说的：

- TTT state 会 periodic reset
- 文中实验里是 **每 5 个 windows reset 一次**。

------

### 4) reset 之后怎么继续保持位姿连续：merge 阶段的 SE(3) alignment

**位置：** `loger/models/pi3.py:827-845`

这里是 window 级预测做全局拼接时的主分支选择：

```python
align_on_resets_without_explicit_pose = reset_every > 0 and not sim3 and not se3

if sim3:
    merged = self._merge_windowed_predictions_sim3(... allow_scale=True ...)
elif se3 or align_on_resets_without_explicit_pose:
    merged = self._merge_windowed_predictions_sim3(... allow_scale=False ...)
else:
    merged = self._merge_windowed_predictions(...)
```

对 **LoGeR\*** 来说，因为 `se3=True`，所以它会走：

```python
self._merge_windowed_predictions_sim3(... allow_scale=False ...)
```

注意名字虽然叫 `sim3`，但当 `allow_scale=False` 时，它实际上退化成了 **纯 SE(3) alignment**。

------

### 5) overlap-based feedforward alignment 的核心实现

**位置：** `loger/models/pi3.py:967-1264`

真正实现 LoGeR* alignment 的函数就是：

```python
def _merge_windowed_predictions_sim3(..., allow_scale=False, ...)
```

#### 5.1 估计相邻 windows 的相对变换

**位置：** `loger/models/pi3.py:1017-1132`

内部函数：

```python
def _estimate_relative_sim3(prev_aligned, curr_raw, overlap, current_allow_scale, ...)
```

这里的关键逻辑：

- `1021-1023`：取前一个 window 的已对齐相机、当前 window 的原始相机

- `1026-1030`：从 overlap 中选对应帧

  ```python
  prev_idx = max(prev_frames - overlap, 0)
  prev_pose = prev_cam[:, prev_idx]
  curr_pose = curr_cam[:, 0]
  ```

  也就是说：

  - 前一个 chunk/window 取 **overlap 区域的第一个重叠帧**
  - 当前 chunk/window 取 **第一个帧**

  当 `overlap_size=1` 时，这就和论文公式里那个重叠帧 (k) 完全一致。
  当 `overlap_size>1` 时，它实际上选的是“共享 overlap 区域里的第一个对应帧”。

- `1037`：计算相对旋转

  ```python
  relative_rot = torch.matmul(R_prev, R_curr.transpose(-1, -2))
  ```

- `1039-1042`：如果 `allow_scale=False`，则 scale 固定为 1
  这就是 **LoGeR\*** 的 rigid SE(3)，而不是 Pi3-Chunk 那种 SIM(3)

- `1129-1130`：计算相对平移

  ```python
  rotated_curr_centers = torch.matmul(relative_rot, t_curr.unsqueeze(-1)).squeeze(-1)
  relative_trans = t_prev - relative_scale.unsqueeze(-1) * rotated_curr_centers
  ```

这正是论文里
[
A_m = \tilde T_k^{(m-1)} (\hat T_k^{(m)})^{-1}
]
那种“用 overlap pose 算 SE(3) 对齐”的代码化版本。论文对 LoGeR* 的描述就是这个意思。

------

### 6) 把这个 SE(3) 施加到当前整个 chunk/window

**位置：** `loger/models/pi3.py:1175-1240`

先把当前 window 的变换写成 4×4：

- `1175-1178`

  ```python
  pose_mat[:, :3, :3] = current_rot
  pose_mat[:, :3, 3] = current_trans
  ```

然后把它应用到：

- **当前 window 的 camera poses**：`1197-1223`
- **当前 window 的 local points / world points**：`1185-1240`

特别是 camera 的那段很关键：

```python
rot_global = current_rot @ rot_local
trans_global = current_rot @ trans_local + current_trans
```

这就是把当前 raw chunk 变成 aligned chunk。

然后：

- `1249`：加入 `aligned_predictions`
- `1257`：再做 overlap-aware merge

------

### 7) reset 和 alignment 是怎样配合的

这个问题最关键的结论是：

**LoGeR\*** 的 reset 和 overlap-based feedforward alignment 不是两套互不相干的逻辑，而是配套的。**

具体来说：

1. 推理过程中，TTT fast weights 按 `reset_every` 周期性清空
   → `654-660`, `668-669`
2. 即使 TTT memory 被 reset，window 之间仍然通过 overlap pose 做 **SE(3) 对齐**
   → `827-845`, `967-1264`

所以 reset 只是在“清记忆”，但不会让全局轨迹直接断掉，因为还有 overlap-based feedforward alignment 在兜底。
这也正好对应论文里那句：

> “we also apply the feedforward pose alignment when doing a reset”

------

### 8) 一个额外但很重要的细节

还有一条特殊分支：

**位置：** `loger/models/pi3.py:830`, `837-843`, `1149-1170`

如果：

- `reset_every > 0`
- 但又 **没有显式开** `sim3` 或 `se3`

代码会走一个特殊 fallback：

```python
align_on_resets_without_explicit_pose = reset_every > 0 and not sim3 and not se3
```

然后用

```python
reuse_transform_within_reset_block=True
```

的方式，在 reset block 边界估一次变换，然后在这个 block 内复用。

不过这条更像是“reset-only 模式的补丁逻辑”，**不是 LoGeR\*** 的主路径。
LoGeR* 的主路径还是 `se3=True` 那条。

------

### 9) 最该看的几个位置

如果你要继续深挖，我建议直接盯这几段：

- **LoGeR\* 开关**
  `ckpts/LoGeR_star/original_config.yaml`
- **读取 se3 / reset_every**
  `loger/models/pi3.py:582-593`
- **TTT reset**
  `loger/models/pi3.py:654-660`
  `loger/models/pi3.py:668-669`
- **进入 SE(3) merge 路径**
  `loger/models/pi3.py:827-845`
- **overlap-based alignment 核心**
  `loger/models/pi3.py:1017-1132`
- **把变换应用到整个当前 window**
  `loger/models/pi3.py:1175-1240`


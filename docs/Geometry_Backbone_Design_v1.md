# Geometry Backbone 设计文档（v1）

> 本文档定义 **Geometry Backbone** 模块。模块以当前 chunk 的图像序列与上一轮已经提交的 TTT fast weights 为输入，在不提交新的 fast-weight write-back 的前提下，完成 LoGeR 的几何前向、TTT apply 读记忆，以及当前 chunk 的 raw update primitive 缓存。本文档重点说明：模块职责、运行时协议、输入输出接口、`WriteCache_m` 的完整字段设计、与当前 `ttt.py / pi3.py` 代码路径的对接方式，以及它和后续 **Dynamic Cue Extractor / Semantic Prior Generator / TTT Write Controller** 的边界。

---

## 0. 一句话定位

**Geometry Backbone 的职责，是在读取旧 memory \(W_m\) 的前提下，为当前 chunk 产出几何结果，并导出一份可供 delayed write-back 使用的 `WriteCache_m`；它不负责提交新的 fast weights。**

---

## 1. 模块目标

Geometry Backbone 在当前 pipeline 中承担四个核心任务：

1. 接收当前 chunk 与当前可读的 TTT fast weights \(W_m\)，作为 LoGeR geometry forward 的输入；
2. 使用 \(W_m\) 执行 TTT **apply**，让当前 chunk 在旧记忆的帮助下完成几何推理；
3. 输出当前 chunk 的几何结果，包括 pointmap、pose、geometry confidence 和 token 对齐信息；
4. 缓存当前 chunk 对 TTT **update** 所需的原始中间量，形成 `WriteCache_m`，供后续 **TTT Write Controller** 在获得 `A_tok` 之后完成真正的 write commit。

更具体地说，它要回答的是：

> 在不提前提交 \(W_{m+1}\) 的情况下，如何让 LoGeR 先用 \(W_m\) 做几何推理，同时又为后续的语义先验控制 write-back 预留完整接口？

---

## 2. 在整条 Pipeline 中的位置

当前 pipeline 可写成：

```text
X_m, W_m
   ↓
Geometry Backbone
   ↓
GeometryPackage_m + TokenMeta_m + WriteCache_m
   ├── Dynamic Cue Extractor
   ├── Video Masklet Front-end
   └── Semantic Prior Generator
                ↓
              A_tok
                ↓
        TTT Write Controller
                ↓
             W_{m+1}
```

其中：

- `Geometry Backbone` 负责 **read old memory + produce geometry + cache raw write signals**；
- `TTT Write Controller` 负责 **inject prior + commit new memory**。

因此，`Geometry Backbone` 与 `TTT Write Controller` 的职责边界非常明确：

- 前者不提交 \(W_{m+1}\)；
- 后者不重跑几何前向，而是在 `WriteCache_m` 上做 delayed write-back。

---

## 3. 运行时语义

### 3.1 当前 chunk 的因果顺序

处理第 \(m\) 个 chunk 时，推荐的语义是：

1. 用上一轮已经提交的 fast weights \(W_m\) 读取 memory；
2. 产出当前 chunk 的几何结果；
3. 记录当前 chunk 的 raw update primitives；
4. 等待后续模块产生 token-wise write prior；
5. 再由 `TTT Write Controller` 完成
   
$$
W_m \rightarrow W_{m+1}
$$

这意味着当前 chunk 的几何结果依赖的是：

$$
X_m,\; W_m
$$

而不是未来才会写出的 \(W_{m+1}\)。

### 3.2 apply 与 update 的分离

LoGeR 中的 TTT 本质上有两个动作：

- **apply**：读 memory
- **update**：写 memory

如果把 fast-weight module 记成 \(f_W(\cdot)\)，那么：

- apply 对应

$$
o_{\text{apply}} = f_{W_m}(q)
$$

- update 对应：用当前 chunk 的 token 序列去构造新的 fast-weight update

因此，Geometry Backbone 的正确职责不是“同时完成 apply 和最终 update commit”，而是：

- **完成 apply**
- **准备 update 所需缓存**
- **把 commit 延迟到 controller**

### 3.3 这个模块最终改变什么

它本身不改变 persistent memory state。它改变的是：

- 当前 chunk 的几何结果可用性；
- 后续模块能否拿到足够完整的 `WriteCache_m`；
- TTT Write Controller 能否在不重跑 geometry 的情况下，按先验控制当前 chunk 的 write-back。

---

## 4. 核心设计结论

### 4.1 输入状态必须是“已提交的旧 memory”

Geometry Backbone 的 TTT 状态输入应该是：

$$
W_m = \{W_{m,l}^{(0,h)}, W_{m,l}^{(1,h)}, W_{m,l}^{(2,h)}\}
$$

它表示在处理 chunk \(m\) 之前，已经被系统正式接受并可供读取的 fast weights。

### 4.2 当前模块不直接导出 \(W_{m+1}\)

本模块不应在 geometry forward 结束时直接把新的 fast weights 作为正式输出提交。原因很简单：

- `A_tok` 还没生成；
- token-level write prior 还没被注入 update chain；
- 当前 chunk 的语义和几何证据还没来得及共同决定“谁该写、谁不该写”。

因此，Geometry Backbone 的正式输出应该是：

- `GeometryPackage_m`
- `TokenMeta_m`
- `WriteCache_m`
- （可选）`ReadStateRef_m = W_m`

而不是最终的 \(W_{m+1}\)。

### 4.3 `WriteCache_m` 是 delayed write package，不是新 weights

`WriteCache_m` 的语义不是“未 update 的权重”本身，而是：

> 当前 chunk 对 TTT update 的原始证据包。

它要保存的是：

- token 对齐信息；
- token-wise coefficient；
- 构造单 token 原始更新贡献所需的 primitives；
- 旧 state 引用；
- update operator 的必要元信息。

### 4.4 token 顺序必须完全可追踪

后续 `Semantic Prior Generator` 会输出：

$$
A_{tok} \in [0,1]^{L_{tok}}
$$

如果 `A_tok` 的 token 顺序与 Geometry Backbone 在当前 chunk 中使用的 token 顺序不一致，TTT Write Controller 就会把 prior 错乘到别的 token 上，整个 delayed write-back 会失效。

因此，`TokenMeta_m` 与 `WriteCache_m` 中必须包含稳定且可重建的 token order 描述。

### 4.5 patch token 与 special token 需要显式区分

当前 LoGeR token 序列不仅包含 patch token，还包含 special tokens，例如：

- register tokens
- role token

这些 token 的来源、语义和后续 prior 策略都不一样，因此 Geometry Backbone 不能只导出 patch grid，而必须显式导出：

- `PatchMeta`
- `TokenType`
- `SpecialTokenMeta`

---

## 5. 记号与张量规模约定

下面默认讨论单条视频流、单个 current chunk。若做 batch，只需在所有张量最前面增加 batch 维 \(B\)。

### 5.1 主要记号

- \(T\)：当前 chunk 的帧数
- \(H, W\)：输入图像分辨率
- \(H_p, W_p\)：LoGeR pointmap 分辨率
- \(H_{tok}, W_{tok}\)：patch token 网格分辨率
- \(P = H_{tok}W_{tok}\)：单帧 patch token 数
- \(R\)：每帧 register token 数
- \(U\)：每帧 role token 数
- \(L_{patch} = T \cdot P\)：整个 chunk 的 patch token 数
- \(L_{special} = T \cdot (R+U)\)：整个 chunk 的 special token 数
- \(L_{tok} = L_{patch} + L_{special}\)：整个 chunk 总 token 数
- \(L_{ttt}\)：TTT 插入层数
- \(H_{ttt}\)：每个 TTT layer 的 head 数

### 5.2 当前 LoGeR 默认 special token 数

若沿用你当前整理的 LoGeR 代码语义，则每帧粗略可视为：

- 5 个 register tokens
- 1 个 role token

因此第一版可取：

$$
R=5,\qquad U=1
$$

于是每帧 token 数约为：

$$
P + 6
$$

整个 chunk 展平后总 token 数约为：

$$
L_{tok} = T(P+6)
$$

---

## 6. 输入定义

### 6.1 主输入

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| 当前 chunk RGB | `X_m` | `[T, H, W, 3]` | 当前 chunk 图像序列 |
| 相机内参 | `K_m` | `[T, 3, 3]` 或 `[3, 3]` | 当前 chunk 相机内参 |
| 当前可读 fast weights | `W_m` | per-layer struct | 上一轮已提交的 TTT fast weights |
| chunk 元信息 | `Meta_chunk` | struct | chunk id、overlap 位置、时间索引等 |

其中：

$$
W_m = \{W_{m,l}^{(0,h)}, W_{m,l}^{(1,h)}, W_{m,l}^{(2,h)}\}_{l=1..L_{ttt},\; h=1..H_{ttt}}
$$

### 6.2 可选输入

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| 上一 chunk 几何摘要 | `G_{m-1}` | struct | 仅用于 debug 或 fallback continuity |
| reset 控制 | `ResetFlag_m` | `[]` | 当前 chunk 是否清空 fast weights |
| camera meta | `CamMeta_m` | struct | 畸变、裁切、frame id 等 |

如果 `ResetFlag_m = True`，则 `W_m` 应在进入 forward 前被重置为默认空状态。

---

## 7. 输出定义

Geometry Backbone 的输出建议显式拆成四部分：

1. `GeometryPackage_m`
2. `TokenMeta_m`
3. `WriteCache_m`
4. `ReadStateRef_m`（可选）

### 7.1 GeometryPackage_m

这是直接供后续 **Dynamic Cue Extractor** 和其他几何消费者使用的结果。

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| camera-space pointmap | `P_cam_m` | `[T, H_p, W_p, 3]` | 当前 chunk 每帧 pointmap |
| pose | `T_w_c_m` | `[T, 4, 4]` | world-from-camera 位姿 |
| geometry confidence | `Conf_geo_m` | `[T, H_p, W_p]` 或 `[T, H, W]` | 每像素几何置信度 |
| optional world pointmap | `P_world_m` | `[T, H_p, W_p, 3]` | 若方便，可直接给世界坐标点图 |
| frame attention prior（可选） | `A_frame_m` | `[T, T]` | decoder attention 汇总得到的 frame-level 亲和度 |
| patch dynamic prior（可选） | `M_attn_patch_m` | `[T, H_tok, W_tok]` | decoder attention 汇总得到的 patch-level dynamicness |
| patch grid size | `PatchShape_m` | `[2]` | `H_tok, W_tok` |
| geometry debug bundle | `Dbg_geo_m` | struct | 可选 debug 信息 |

如果后续模块更习惯 world 坐标，也可以直接导出：

$$
P_{world,m}(t,u) = T_{w\leftarrow c,m}(t) \cdot P_{cam,m}(t,u)
$$

但从接口最小化角度，导出 `P_cam_m + T_w_c_m` 通常已经足够。

### 7.2 TokenMeta_m

这部分是后续 `Semantic Prior Generator` 与 `TTT Write Controller` 共同需要的对齐信息。

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| patch token 索引表 | `PatchMeta` | `[L_patch, 3]` | 每个 patch token 对应 `(t, y_tok, x_tok)` |
| token type | `TokenType` | `[L_tok]` | `patch / reg / role` |
| token frame id | `TokenFrameId` | `[L_tok]` | 每个 token 属于哪一帧 |
| token valid mask | `TokenValid` | `[L_tok]` | 若有 padding，标记哪些 token 有效 |
| special token meta | `SpecialTokenMeta` | `[L_special, 3]` 或 struct list | special token 的类型、本地索引、frame id |
| flatten order id | `TokenOrderId` | `[L_tok]` | 当前 chunk token 序列的稳定顺序标识 |

其中最关键的是：

- `PatchMeta`
- `TokenType`
- `TokenOrderId`

如果启用了 attention-prior 导出，那么 `A_frame_m / M_attn_patch_m` 也会成为 Stage B 的额外输入，用来做 support ranking 和动态证据融合；它们不会改变 token 对齐顺序，但会影响后续写入先验的估计质量。

这三项决定 `A_tok` 是否能和当前 chunk 的 token 序列严格对齐。

### 7.3 ReadStateRef_m（可选）

如果希望接口自洽，可以把旧 memory 的引用也作为输出之一：

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| current readable weights | `ReadStateRef_m` | per-layer struct | 本 chunk 真正读取过的 fast weights |

它通常就是输入的 `W_m` 原样透传，但在 delayed write-back 协议里，把它显式包含在输出中会让后续 controller 接口更完整。

### 7.4 WriteCache_m

这是本模块最重要的输出。它不是最终更新后的 weights，而是：

> 当前 chunk 对 TTT update 的 raw cache。

它至少应包含下面四组内容：

1. token 对齐信息
2. 每层的 raw update primitives
3. 旧 weights 快照或引用
4. update 执行元信息

完整 schema 见第 10 节。

---

## 8. Geometry Backbone 的内部结构

为了让模块职责清晰，建议把 Geometry Backbone 拆成 6 个子单元：

1. **State Loader**：装载 `W_m`
2. **Chunk Tokenizer**：把当前 chunk 转成 LoGeR token 序列
3. **TTT Apply Reader**：使用 `W_m` 做 apply
4. **Geometry Decoder**：输出 pointmap / pose / confidence
5. **Raw Update Recorder**：缓存当前 chunk 的 update primitives
6. **Cache Assembler**：打包 `GeometryPackage_m / TokenMeta_m / WriteCache_m`

其数据流可以概括为：

```text
X_m + W_m
   ↓
Chunk tokenization
   ↓
TTT apply with W_m
   ↓
Geometry decode
   ↓
Raw update primitive capture
   ↓
GeometryPackage_m + TokenMeta_m + WriteCache_m
```

---

## 9. 核心算法流程

### 9.1 Stage A：加载当前可读 memory

处理 chunk \(m\) 时，先装载：

$$
W_m
$$

如果当前 chunk 触发 reset，则把 `W_m` 置为空状态。

这一阶段不做任何新的 write commit，只是明确：

- 当前 chunk 将读取哪组 fast weights；
- 当前 chunk 若需要缓存 update，应围绕这组 `W_m` 展开。

### 9.2 Stage B：chunk tokenization

将当前 chunk 的每一帧编码成 LoGeR token 序列。若单帧 patch grid 为 \(H_{tok}\times W_{tok}\)，则 patch token 数为：

$$
P = H_{tok}W_{tok}
$$

每帧再拼接 special tokens 后，整个 chunk 展平成：

$$
L_{tok} = T(P + R + U)
$$

这一阶段必须同时记录：

- patch token 在 chunk 中的时空位置；
- special token 的类型和所属帧；
- flatten 后的全局顺序。

### 9.3 Stage C：TTT apply with old memory

对每个 TTT layer，Geometry Backbone 使用 \(W_m\) 做 apply。若 query 为 \(q\)，则：

$$
o_{\text{apply}} = f_{W_m}(q)
$$

这部分输出进入 LoGeR 主干残差路径，用于帮助当前 chunk 完成几何推理。

这里最重要的约束是：

- **允许读 memory**
- **不提交新的 persistent write**

### 9.4 Stage D：几何解码

在 apply 之后继续完成 LoGeR 的 geometry forward，输出：

- pointmap
- pose
- geometry confidence
- 其他几何头结果

也就是第 7 节定义的 `GeometryPackage_m`。

### 9.5 Stage E：记录 raw update primitives

当前 LoGeR 的 TTT update 本质上是对整个 chunk 的 token 序列做聚合式更新。为了让后续 controller 能在注入 prior 后继续完成 write-back，Geometry Backbone 在这一阶段需要记录每层 update 所需的 primitives。

设当前 layer/head 的单 token 原始更新贡献写成：

$$
J_{m,l,i}^{(1,h)} = U_{m,l,i}^{(h)\top} V_{m,l,i}^{(h)}
$$

$$
J_{m,l,i}^{(0,h)} = K_{m,l,i}^{(h)\top} A_{m,l,i}^{(h)}
$$

$$
J_{m,l,i}^{(2,h)} = K_{m,l,i}^{(h)\top} B_{m,l,i}^{(h)}
$$

这里：

- `U / A / B` 不是最终输出，而是 update path 的内部原语；
- 后续 `TTT Write Controller` 将用 `A_tok` 去改写每个 token 的 contribution weighting。

因此 Geometry Backbone 不应直接把当前 chunk 的 token 贡献聚合成已提交更新，而应保留：

- token-wise coefficient `eta`
- 构造 \(J_i\) 所需的 primitives
- 旧 state 引用

### 9.6 Stage F：组装 WriteCache_m

最后将所有 per-layer primitives 与 token meta 组装成：

$$
WriteCache_m
$$

供后续 `TTT Write Controller` 使用。至此，Geometry Backbone 的工作结束。

---

## 10. `WriteCache_m` 的完整 schema

### 10.1 顶层结构

推荐把 `WriteCache_m` 组织成：

```text
WriteCache_m = {
  chunk_id,
  token_meta_ref,
  cache_layers,
  read_state_ref,
  update_meta,
}
```

其中：

- `token_meta_ref` 可以直接引用 `TokenMeta_m`
- `cache_layers` 是每个 TTT layer 的 raw cache
- `read_state_ref` 是 `W_m` 的引用或快照
- `update_meta` 是 controller 重建 update 所需的执行元信息

### 10.2 token 对齐信息

虽然 `TokenMeta_m` 可以单独输出，但为避免后续模块接口复杂，建议 `WriteCache_m` 内部也持有引用：

| 名称 | Shape | 含义 |
|---|---:|---|
| `PatchMeta` | `[L_patch, 3]` | patch token 对应 `(t, y_tok, x_tok)` |
| `TokenType` | `[L_tok]` | `patch / reg / role` |
| `TokenFrameId` | `[L_tok]` | 每个 token 属于哪一帧 |
| `TokenValid` | `[L_tok]` | token 是否有效 |
| `TokenOrderId` | `[L_tok]` | 当前 token 顺序稳定标识 |

### 10.3 每层 raw update primitives

对每个 TTT layer \(l\)，建议缓存：

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| token-wise coefficient for `w0` | `eta0_l` | `[H_ttt, L_tok, 1]` | 当前代码中的 branch-0 token 系数 |
| token-wise coefficient for `w1` | `eta1_l` | `[H_ttt, L_tok, 1]` | 当前代码中的 branch-1 token 系数 |
| token-wise coefficient for `w2` | `eta2_l` | `[H_ttt, L_tok, 1]` | 当前代码中的 branch-2 token 系数 |
| key primitive | `K_l` | `[H_ttt, L_tok, d_k]` | update 所需 key-like 原语 |
| value primitive | `V_l` | `[H_ttt, L_tok, d_v]` | update 所需 value-like 原语 |
| hidden primitive for `w1` | `U_l` | `[H_ttt, L_tok, d_u]` | branch-1 原始更新所需 hidden 原语 |
| aux primitive for `w0` | `A_l` | `[H_ttt, L_tok, d_a]` | branch-0 原始更新所需辅助原语 |
| aux primitive for `w2` | `B_l` | `[H_ttt, L_tok, d_b]` | branch-2 原始更新所需辅助原语 |

这组张量足以让后续 controller 在获得 token prior 后，重建：

$$
J_i^{(1)} = U_i^\top V_i,
\qquad
J_i^{(0)} = K_i^\top A_i,
\qquad
J_i^{(2)} = K_i^\top B_i
$$

并进一步完成：

$$
A_{tok} \rightarrow p \rightarrow \beta \rightarrow \gamma \rightarrow \tilde G
$$

### 10.4 旧 state 快照或引用

为了让 delayed write-back 完整可复现，建议在每个 layer cache 中一起保留旧 state：

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| previous `w0` | `W_prev_l^(0)` | per-head matrix | 当前 chunk 真正读取过的 branch-0 weights |
| previous `w1` | `W_prev_l^(1)` | per-head matrix | 当前 chunk 真正读取过的 branch-1 weights |
| previous `w2` | `W_prev_l^(2)` | per-head matrix | 当前 chunk 真正读取过的 branch-2 weights |

如果工程实现上不想在 layer cache 内重复保存，也可以只在 `read_state_ref` 中全局保存一次，由 controller 统一访问。

### 10.5 update 执行元信息

| 名称 | Shape | 含义 |
|---|---:|---|
| `ttt_op_order` | struct | 当前是 apply-then-update |
| `zeropower_cfg` | struct | `zeropower` 近似计算所需配置 |
| `renorm_cfg` | struct | 写回前 renorm 相关配置 |
| `momentum_cfg` | struct | 若启用 momentum，需要导出其配置 |
| `layer_head_meta` | struct | 各层 head 数、各分支维度 |
| `chunk_shape_meta` | struct | `T/H_tok/W_tok/L_tok` 等规模信息 |

这部分的作用是保证 controller 后续做 write commit 时，仍然严格遵循 LoGeR 当前实现的 update 语义。

---

## 11. Geometry Backbone 与当前 LoGeR 代码的对接方式

### 11.1 推荐主实现：split apply and cache

当前 `ttt.py` 中，`FastWeightGluMLPMultihead.forward(...)` 会把 apply 和 update 打包在一起，默认返回更新后的 `w0 / w1 / w2`。如果要支持当前 pipeline，推荐的主实现方式是把这条路径拆成：

1. **apply path**：读取 \(W_m\)，给主干提供 memory readout
2. **cache path**：导出 update primitives，但不提交新的 persistent state

也就是说，在 Geometry Backbone 内部，TTT 层的语义应改成：

```text
forward_with_cache(tokens, W_m)
    -> apply_output
    -> raw_update_primitives
    -> no persistent state commit
```

### 11.2 fallback 方案：provisional execution + discard state mutation

如果第一版工程上不想大改 LoGeR 内部路径，也可以采用 fallback：

1. 让当前 `ttt.py` 暂时照常执行；
2. 把本次 forward 期间产生的 raw update primitives 缓存下来；
3. 不把内部 provisional updated weights 当作正式 `W_{m+1}`；
4. 仍以 `WriteCache_m + W_m` 为唯一合法写回来源。

这个方案工程风险较小，但接口语义不如主实现干净。

### 11.3 无论哪种实现，都必须满足的约束

无论采用哪一种改法，Geometry Backbone 都必须满足：

1. 当前 chunk 的 geometry 结果基于 \(W_m\) 读取 memory；
2. 当前 chunk 结束后不能直接把 provisional state 当成正式 \(W_{m+1}\)；
3. 后续只有 `TTT Write Controller` 才有权提交新的 fast weights。

---

## 12. 与后续模块的接口约定

### 12.1 给 Dynamic Cue Extractor 的接口

Dynamic Cue Extractor 至少需要：

- `P_cam_m`
- `T_w_c_m`
- `Conf_geo_m`

可选还可给：

- `P_world_m`
- patch grid 信息

### 12.2 给 Semantic Prior Generator 的接口

Semantic Prior Generator 至少需要：

- `TokenMeta_m`
- patch grid 尺度 `H_tok, W_tok`
- token flatten 顺序

这样它才能把：

- masklet-level prior
- pixel-level prior

正确映射到：

$$
A_{tok}
$$

### 12.3 给 TTT Write Controller 的接口

TTT Write Controller 至少需要：

- `WriteCache_m`
- `ReadStateRef_m`（若不已包含于 cache 内）

然后它才能用：

$$
A_{tok}
$$

去控制 token contribution weighting，最终得到：

$$
W_{m+1}
$$

---

## 13. 典型失败模式与缓解策略

### 13.1 失败模式：Geometry Backbone 误提交了新的 weights

**问题**：Stage A 结束时内部 state 已经变成新的 weights，导致 controller 的 delayed write 失去意义。  
**缓解**：明确区分 `provisional state` 与 `committed state`，并把 `TTT Write Controller` 设为唯一 commit 入口。

### 13.2 失败模式：token 顺序与 `A_tok` 不一致

**问题**：prior 被乘到了错误 token 上。  
**缓解**：`TokenMeta_m` 中显式输出 `TokenOrderId`，并把它作为 controller 的必检字段。

### 13.3 失败模式：只缓存了聚合后的更新矩阵，没缓存原始 primitives

**问题**：controller 只能在 prior 生成前就固定 update 结构，无法真正做到 token-wise reweight。  
**缓解**：缓存 `eta/K/V/U/A/B` 等 token-level primitives，而不是只缓存已聚合的 `\tilde G`。

### 13.4 失败模式：patch token 与 special token 未区分

**问题**：special token 被错误使用 patch prior。  
**缓解**：`TokenType + SpecialTokenMeta` 必须显式输出。

### 13.5 失败模式：`WriteCache_m` 不足以重建 update 语义

**问题**：controller 只能实现一个近似版 write-back，和 LoGeR 当前代码路径不一致。  
**缓解**：把 `ttt_op_order / zeropower_cfg / renorm_cfg / layer_head_meta` 一并纳入 `WriteCache_m`。

---

## 14. 最小可实现版本（MVP）建议

如果只做一个主线正确、工程复杂度可控的版本，建议按下面顺序实现。

### Phase 1：最小 delayed-write backbone

先实现：

1. `X_m + W_m -> GeometryPackage_m`
2. `TokenMeta_m`
3. per-layer `eta0/1/2 + K/V/U/A/B` cache
4. `ReadStateRef_m`

只要这一步打通，系统就已经具备：

> 当前 chunk 先做 geometry，再把 write commit 留给 controller

这一关键能力。

### Phase 2：补完整 cache meta

再加入：

1. `TokenValid / TokenOrderId`
2. `SpecialTokenMeta`
3. `zeropower_cfg / renorm_cfg / momentum_cfg`

### Phase 3：更精细的 debug / fallback 支持

最后再加：

1. `Dbg_geo_m`
2. optional provisional state for debug only
3. reset / fallback 相关记录

---

## 15. 一句话总结

> **Geometry Backbone 不是一个“顺手把 TTT update 也做完”的 LoGeR 包装器，而是当前 pipeline 中的 read-and-cache 前端：它用旧 memory \(W_m\) 完成当前 chunk 的几何推理，同时把本次 write 所需的 raw update primitives 缓存成 `WriteCache_m`，从而把真正的 fast-weight commit 留给后续基于语义和几何先验的 TTT Write Controller。**

---

## 16. 文档末尾的接口摘要（便于实现时查阅）

### 输入

- `X_m: [T,H,W,3]`
- `K_m: [T,3,3]` 或 camera meta
- `W_m`: previous committed fast weights
- `Meta_chunk`

### 输出

- `GeometryPackage_m = {P_cam_m, T_w_c_m, Conf_geo_m, optional P_world_m}`
- `TokenMeta_m = {PatchMeta, TokenType, TokenFrameId, TokenValid, SpecialTokenMeta, TokenOrderId}`
- `WriteCache_m`
- `ReadStateRef_m`（optional）

### `WriteCache_m` 关键字段

- `eta0_l, eta1_l, eta2_l`
- `K_l, V_l, U_l, A_l, B_l`
- `W_prev_l^(0/1/2)` 或全局 `read_state_ref`
- `ttt_op_order`
- `zeropower_cfg`
- `renorm_cfg`
- `momentum_cfg`
- `layer_head_meta`
- `chunk_shape_meta`

### 运行时语义

- 当前 chunk 读：\(W_m\)
- 当前 chunk 不直接写：\(W_{m+1}\)
- 当前 chunk 输出：geometry + cache
- 后续 controller 提交：\(W_m \rightarrow W_{m+1}\)

# Geometry Backbone 设计文档（v2）

> 本文档是 `Geometry_Backbone_Design_v1.md` 的大版本更新。v1 将 Geometry Backbone 定义为“读取旧 TTT fast weights，完成当前 chunk 几何 forward，并缓存 TTT update primitives”。v2 需要支持 **Hybrid Memory Controller**，因此 Geometry Backbone 不再只暴露 TTT write cache，而要暴露 LoGeR hybrid block 中 frame attention、SWA、TTT apply/update、chunk-wise bidirectional attention 的 probe traces，并支持 two-pass controlled forward。
>
> 旧的 `WriteCache_m` 会被保留为 `HybridMemoryCache_m` 的 TTT 子字段；旧的 TTT Write Controller 将作为 Hybrid Memory Controller 的子控制器继续使用。

---

## 0. One-line summary

**Geometry Backbone v2 的职责，是在读取 committed hybrid memory $\mathcal{H}_m$ 的前提下，支持两种运行：probe pass 收集当前 chunk 的几何输出、内部 attention/memory traces 和 TTT/SWA cache；controlled pass 从同一个 $\mathcal{H}_m$ 重新运行当前 chunk，并接收 Hybrid Memory Controller 的控制 bundle，最终输出 controlled geometry 和 $\mathcal{H}_{m+1}$。**

---

## 1. 为什么 Geometry Backbone 需要升级

### 1.1 v1 的边界

v1 的 Backbone 输出：

```text
GeometryOutput + TokenMeta + TTT WriteCache
```

下游 TTT Write Controller 用 `A_tok` 重放 update：

$$
W_m^{TTT}\rightarrow W_{m+1}^{TTT}
$$

这对 delayed TTT write control 足够，但对 Hybrid Memory Controller 不够。

### 1.2 v2 需要暴露 LoGeR 的 hybrid memory 全路径

LoGeR hybrid block 的顺序是：

```text
per-frame attention
    ↓
sliding-window attention over previous/current chunks
    ↓
chunk-wise TTT apply/update
    ↓
chunk-wise bidirectional attention
```

如果 controller 需要控制 frame attention、SWA read、TTT apply、TTT update 和 chunk bi-attn，就必须让 Backbone：

1. 在 probe pass 中导出这些路径的可诊断 traces；
2. 在 controlled pass 中接收 control bundle 并在对应模块插入 soft bias / gate；
3. 正确管理 committed state、probe provisional state 和 controlled state；
4. 保证 two-pass no-control 与 unity/native parity。

---

## 2. 新职责定义

Geometry Backbone v2 承担六个任务：

1. **State loading**：读取 committed hybrid memory $\mathcal{H}_m$；
2. **Probe forward**：无控制地运行当前 chunk，产出 geometry 和 traces，不提交 state；
3. **Trace / cache export**：导出 frame/SWA/TTT/chunk attention 与 update cache；
4. **Controlled forward**：从同一个 $\mathcal{H}_m$ 重新运行当前 chunk，并注入 control bundle；
5. **Controlled commit**：只提交 controlled pass 的 $\mathcal{H}_{m+1}$；
6. **Token/state alignment**：确保所有 token-level control fields 与真实 LoGeR token 顺序一致。

---

## 3. 运行模式

### 3.1 Native mode

用于 reproduction：

```text
run(images, state_m, mode="native")
```

行为：

- 按 LoGeR 原生方式运行；
- 如果启用 state update，则提交 native TTT/SWA state；
- 不保存完整 probe traces。

### 3.2 Probe mode

用于 first pass：

```text
probe = run_probe(images, state_m, collect_hybrid_trace=True)
```

行为：

- 从 committed state $\mathcal{H}_m$ 开始；
- 原生 forward；
- 采集 traces 和 caches；
- 不把 provisional update 提交给外部 persistent state。

### 3.3 Controlled mode

用于 second pass：

```text
result = run_controlled(images, state_m, control_bundle)
```

行为：

- 仍然从 committed state $\mathcal{H}_m$ 开始，而不是 probe provisional state；
- 在 selected frame/SWA/TTT/chunk paths 注入控制；
- controlled geometry 作为最终输出；
- controlled hybrid state 作为 $\mathcal{H}_{m+1}$ 提交。

### 3.4 Probe-only debug mode

用于离线分析：

```text
probe = run_probe(...)
save_probe_artifacts(probe)
```

不跑 controlled pass。

---

## 4. 状态定义

### 4.1 HybridMemoryState

v2 中，输入状态不是单纯 `w0/w1/w2`。

```text
HybridMemoryState_m = {
  ttt_state,
  swa_state,
  ref_state,
  prev_control_summary,
}
```

| 字段 | 含义 |
|---|---|
| `ttt_state` | TTT fast weights `w0/w1/w2` and optional history |
| `swa_state` | previous chunk local context / KV / token features, depending on LoGeR implementation |
| `ref_state` | protected reference tokens and reset/overlap metadata |
| `prev_control_summary` | previous chunk dynamic/reliability/control maps used by SWA read gating |

### 4.2 TTT state

沿用当前代码语义：

```text
ttt_state = {
  w0: list[tensor or None],
  w1: list[tensor or None],
  w2: list[tensor or None],
  history: optional SWA/KV-like history,
}
```

### 4.3 SWA state

SWA state 在当前代码中可能体现为 `history` 或 block 内部缓存。v2 需要显式区分：

- TTT fast weights；
- SWA / local attention history；
- previous chunk token metadata；
- previous chunk control summary。

第一版实现可以先把已有 `history` 继续挂在 `ttt_state["history"]`，但文档和接口上应迁移到：

```text
HybridMemoryState.swa_state
```

---

## 5. 输出结构总览

Probe pass 输出：

```text
ProbeOutput = {
  geometry,
  token_meta,
  hybrid_cache,
  probe_trace,
  native_provisional_state,
  debug,
}
```

Controlled pass 输出：

```text
ControlledOutput = {
  geometry,
  token_meta,
  hybrid_state_next,
  control_trace,
  debug,
}
```

---

## 6. GeometryOutput

沿用 v1 并保留内部 cue 字段。

| 名称 | Shape | 含义 |
|---|---:|---|
| `local_points` | `[T,H_p,W_p,3]` | camera-space pointmap |
| `world_points` | `[T,H_p,W_p,3]` | world-space pointmap |
| `camera_poses` | `[T,4,4]` | world-from-camera pose |
| `confidence` | `[T,H_p,W_p]` | geometry confidence |
| `patch_meta` | `[L_patch,3]` | patch token `(t,y,x)` |
| `token_type` | `[L_tok]` | register / role / patch |
| `frame_attention_prior` | `[T,T]` optional | frame-level affinity |
| `attn_dynamic_patch` | `[T,H_tok,W_tok]` optional | existing attention-derived dynamic patch |
| `dyn4d_patch` | `[T,H_tok,W_tok]` optional | VGGT4D-style internal dynamic cue |
| `frame_attn_key_cosine_*` | `[T,H_tok,W_tok]` optional | MUT3R-style / key-cosine cues |
| `raw_predictions` | dict | raw model outputs |

v2 不强制每个 cue 都实现，但 schema 应保留。

---

## 7. TokenMeta

TokenMeta 需要从 v1 的 `PatchMeta + TokenType` 扩展。

```text
TokenMeta = {
  patch_meta,
  token_type,
  token_frame_id,
  token_local_index,
  token_order_id,
  token_valid,
  special_token_meta,
  protected_default_mask,
}
```

### 7.1 字段说明

| 名称 | Shape | 含义 |
|---|---:|---|
| `patch_meta` | `[L_patch,3]` | patch token `(t,y_tok,x_tok)` |
| `token_type` | `[L_tok]` | `0=register,1=role,2=patch` |
| `token_frame_id` | `[L_tok]` | 每个 token 属于哪一帧 |
| `token_local_index` | `[L_tok]` | 每帧内部 token index |
| `token_order_id` | `[L_tok]` | 当前 flatten order 的稳定 id |
| `token_valid` | `[L_tok]` | padding / valid mask |
| `special_token_meta` | struct | register/role token 信息 |
| `protected_default_mask` | `[L_tok]` | special/overlap/reset 默认保护 mask |

### 7.2 token order 约束

如果 LoGeR 当前 per-frame token layout 是：

```text
[register tokens, role token, patch tokens]
```

则必须在 TokenMeta 中明确记录，避免 controller 默认 `A_tok[:l]` 时发生错位。

当前实验显示 `cache_l` 与 `L_tok` 对齐，但 v2 仍要求显式 `token_order_id`，为后续 frame/SWA/chunk control 做准备。

---

## 8. HybridMemoryCache

v2 的 `HybridMemoryCache` 包含四类 cache。

```text
HybridMemoryCache = {
  token_meta,
  ttt_cache,
  swa_cache,
  frame_attn_cache,
  chunk_attn_cache,
  state_refs,
  update_meta,
}
```

### 8.1 TTTCache

沿用当前 `WriteCacheOutput`。

```text
TTTCache = {
  layer_caches,
  w0_provisional,
  w1_provisional,
  w2_provisional,
  history_provisional,
  num_frames,
  patch_grid,
  num_ttt_layers,
}
```

每层：

| 名称 | Shape | 含义 |
|---|---:|---|
| `q` | `[B*H,L,d]` | apply query |
| `k` | `[B*H,L,d]` | update key |
| `v` | `[B*H,L,dv]` | update value |
| `lr0/lr1/lr2` | `[B*H,L,1]` | branch-wise eta |
| `w0_old/w1_old/w2_old` | tensors | old fast weights |
| `momentum` | optional | Muon / optimizer state |
| `ttt_op_order` | list | apply/update order |
| `muon_update_steps` | int | update operator meta |
| `ttt_update_steps` | int | update operator meta |

新增建议字段：

| 名称 | Shape | 含义 |
|---|---:|---|
| `token_indices` | `[L]` | cache token 对应全局 token index |
| `layer_id` | scalar | LoGeR block/layer id |
| `branch_names` | list | `w0/w1/w2` |

### 8.2 SWACache

用于 SWA read control。

```text
SWACache = {
  layer_traces,
  prev_token_meta,
  curr_token_meta,
  attention_importance,
}
```

每层 trace：

| 名称 | Shape | 含义 |
|---|---:|---|
| `attn_prev_to_curr` or `attn_qk` | compact | current query to previous/current keys |
| `q_stats` | `[L_curr,...]` optional | query summaries |
| `k_stats_prev` | `[L_prev,...]` optional | previous key summaries |
| `prev_token_indices` | `[L_prev]` | previous chunk token ids |
| `curr_token_indices` | `[L_curr]` | current chunk token ids |
| `layer_id` | scalar | SWA insertion layer |

如果完整 attention 太大，至少保存：

$$
I_j^{prev}=\sum_{l,h,q}A_{l,h,q\rightarrow j}^{swa}
$$

用于判断 previous tokens 的 SWA 重要性。

### 8.3 FrameAttentionCache

用于 internal motion cue 和 frame attention control。

```text
FrameAttentionCache = {
  layer_stats,
  qk_summary,
  key_cosine_maps,
  attention_entropy_maps,
}
```

可选字段：

| 名称 | Shape | 含义 |
|---|---:|---|
| `attn_entropy_patch` | `[T,H_tok,W_tok]` | self-attn dispersion |
| `key_cosine_patch` | `[T,H_tok,W_tok]` | key consistency cue |
| `qk_var_patch` | `[T,H_tok,W_tok]` | QK variance |
| `gram_qq/kk/qk_stats` | `[T,H_tok,W_tok]` | Gram-lite cue |
| `layer_ids` | list | 来源层 |

### 8.4 ChunkAttentionCache

用于 chunk-wise bidirectional attention control。

```text
ChunkAttentionCache = {
  layer_stats,
  qk_summary,
  cross_frame_attention_mass,
  dynamic_key_mass,
}
```

建议保存：

$$
A_{t\rightarrow s}^{chunk}
$$

的 frame-level 聚合，以及 patch-level dynamic key mass。

---

## 9. ProbeTrace

ProbeTrace 是从 HybridMemoryCache 中派生出的可直接用于 Stage B/D 的轻量摘要。

```text
ProbeTrace = {
  internal_dynamic_patch,
  ttt_residual_patch,
  swa_prev_importance,
  frame_attn_entropy_patch,
  chunk_attn_dynamic_mass,
  reference_importance,
}
```

### 9.1 internal_dynamic_patch

可由现有字段组合：

$$
D_{internal}=\operatorname{Fuse}(attn\_dynamic, key\_cosine, dyn4d, gram)
$$

### 9.2 ttt_residual_patch

对每个 TTT layer 可计算：

$$
e_i=\frac{\|f_{W_m}(k_i)-v_i\|_2}{\|v_i\|_2+\epsilon}
$$

然后聚合到 patch grid：

$$
E_{ttt}(t,y,x)=\operatorname{Mean}_{i\in\Pi(t,y,x)}e_i
$$

### 9.3 swa_prev_importance

Previous-token attention importance：

$$
I_j^{swa}=\sum_{h,q}A_{h,q\rightarrow j}^{swa}
$$

并映射到 previous chunk patch grid。

---

## 10. ControlBundle

Controlled pass 接收：

```text
ControlBundle = {
  mode,
  frame_attention_control,
  swa_control,
  ttt_apply_control,
  ttt_update_control,
  chunk_attention_control,
  reference_protection,
}
```

### 10.1 Frame attention control

```text
FrameAttentionControl = {
  enabled,
  layer_selector,
  D_tok,
  beta,
  bias_clip,
  protected_mask,
}
```

Attention bias：

$$
B_{qk}^{frame}=\beta\log(1-(1-D_q)D_k+\epsilon)
$$

### 10.2 SWA control

```text
SWAControl = {
  enabled,
  layer_selector,
  D_prev_tok,
  beta,
  protected_prev_mask,
}
```

Bias：

$$
B_{qk}^{swa}=\beta\log(1-D_k^{prev}+\epsilon)
$$

### 10.3 TTT apply control

```text
TTTApplyControl = {
  enabled,
  layer_selector,
  read_gate_tok,
}
```

Gate：

$$
H_i\leftarrow H_i+r_i^{read}f_{W_m}(LN(H_i))
$$

### 10.4 TTT update control

```text
TTTUpdateControl = {
  enabled,
  branch_mask,
  layer_selector,
  token_prior,
  eta_mean_preserve,
  sparse_route_mask,
}
```

### 10.5 Chunk attention control

```text
ChunkAttentionControl = {
  enabled,
  layer_selector,
  D_tok,
  beta,
  bias_clip,
  protected_mask,
}
```

---

## 11. Internal algorithms

### 11.1 `run_probe`

```python
def run_probe(images, state_m, *, collect_hybrid_trace=True):
    state_input = clone_state_to_device(state_m)
    raw = model(
        images,
        ttt_state_input=state_input.ttt_state,
        swa_state_input=state_input.swa_state,
        cache_ttt_primitives=True,
        collect_frame_attn_trace=True,
        collect_swa_trace=True,
        collect_chunk_attn_trace=True,
        return_ttt_state=True,
        commit_state=False,
    )
    geometry = postprocess_geometry(raw)
    token_meta = build_token_meta(raw, images)
    cache = build_hybrid_memory_cache(raw, token_meta)
    trace = build_probe_trace(raw, cache)
    return ProbeOutput(geometry, token_meta, cache, trace, raw_provisional_state)
```

### 11.2 `run_controlled`

```python
def run_controlled(images, state_m, control_bundle):
    state_input = clone_state_to_device(state_m)
    raw = model(
        images,
        ttt_state_input=state_input.ttt_state,
        swa_state_input=state_input.swa_state,
        hybrid_control_bundle=control_bundle,
        return_ttt_state=True,
        return_swa_state=True,
        commit_state=True,
    )
    geometry = postprocess_geometry(raw)
    state_next = extract_hybrid_state(raw)
    debug = extract_control_debug(raw)
    return ControlledOutput(geometry, state_next, debug)
```

### 11.3 no-control parity

`control_bundle=None` 时，`run_controlled` 必须等价于 native/unity route。

---

## 12. 与当前 Python 代码的迁移

### 12.1 当前类

当前 `geometry_backbone.py` 中已有：

- `GeometryOutput`
- `TTTLayerCache`
- `WriteCacheOutput`
- `LoGeRGeometryBackbone.run(...)`

v2 建议迁移为：

| 旧名称 | 新名称 | 说明 |
|---|---|---|
| `WriteCacheOutput` | `HybridMemoryCacheOutput` | 保留 TTT cache，增加 SWA/attn traces |
| `TTTLayerCache` | `TTTLayerCache` | 可保留 |
| `GeometryOutput` | `GeometryOutput` | 保留，扩展 trace fields |
| `LoGeRGeometryBackbone.run` | `run_native/run_probe/run_controlled` | 明确模式 |

### 12.2 最小兼容实现

第一步可以不立即改底层 LoGeR attention，而是：

1. `run_probe` 调用当前 `run(..., cache_ttt_primitives=True)`；
2. `HybridMemoryCacheOutput.ttt_cache = old WriteCacheOutput`；
3. `swa_cache/frame_attn_cache/chunk_attn_cache = None`；
4. `run_controlled` 先只复用旧 TTT replay controller；
5. 完成 two-pass parity。

然后逐步加 trace hooks。

### 12.3 需要底层 LoGeR 支持的新增 kwargs

```python
model(
  images,
  cache_ttt_primitives=True,
  collect_frame_attn_trace=False,
  collect_swa_trace=False,
  collect_chunk_attn_trace=False,
  hybrid_control_bundle=None,
  return_swa_state=False,
  commit_state=True,
)
```

---

## 13. Debug and validation

### 13.1 必备 debug

```text
GeometryBackboneDebug = {
  mode,
  pass_id,
  state_input_hash,
  ttt_state_input_present,
  swa_state_input_present,
  token_count,
  token_type_count,
  ttt_cache_l_by_layer,
  frame_trace_enabled,
  swa_trace_enabled,
  chunk_trace_enabled,
  control_enabled_paths,
}
```

### 13.2 Two-pass parity check

运行：

```text
probe pass + controlled pass with no control
```

输出 trajectory 应接近 unity replay。

判断：

$$
|ATE_{two-pass-no-control}-ATE_{unity}|<0.1m
$$

### 13.3 State double-write check

必须确认 controlled pass 不从 probe provisional state 开始。Debug 中记录：

```text
probe_state_hash
controlled_input_state_hash
controlled_output_state_hash
```

要求：

```text
controlled_input_state_hash == committed_state_hash
controlled_input_state_hash != probe_provisional_state_hash
```

除非 probe control explicitly disabled state update。

### 13.4 Token alignment check

继续保留：

- patch-only prior；
- special-only prior；
- frame ramp；
- reverse ramp；
- roll prior。

但 v2 还需要加：

- SWA previous-token prior ramp；
- chunk-attn key-only ramp；
- frame-attn key-only checkerboard。

---

## 14. 实现优先级

### Phase 1：接口重构，不改模型行为

目标：所有旧实验能通过新接口跑。

- 新建 `HybridMemoryState`；
- 新建 `HybridMemoryCacheOutput`；
- 新建 `run_probe` / `run_controlled` wrapper；
- `control_bundle=None` 时完全不改变输出。

### Phase 2：TTT 子路径兼容

目标：复现 BL-01 branch0 dyn MP-01。

- `HybridMemoryCacheOutput.ttt_cache` 对接旧 TTT replay；
- `HybridMemoryController` 只启用 TTT update；
- no-control 和 ttt-only parity 通过。

### Phase 3：trace collection

目标：先看信号，不控制。

- frame attention internal motion summary；
- SWA previous-token importance summary；
- chunk attention dynamic-key mass；
- TTT residual / alignment score。

### Phase 4：read-path control hooks

目标：测试 TTT-only 以外的收益。

- frame attention soft bias；
- SWA read soft bias；
- chunk bi-attn soft bias；
- TTT apply gate。

### Phase 5：sparse write and reference protection

目标：减少 dense reweighting 的长期干扰。

- branch0 sparse route；
- protected reference mask；
- overlap/reset tokens protection；
- update direction cosine diagnostics。

---

## 15. 风险与注意事项

### 15.1 Two-pass 成本

Two-pass 会约等于两倍 forward 成本。但这是研究阶段可接受的。若有效，后续再合并 probe cue extraction 到 single pass。

### 15.2 Attention trace 内存

完整 attention maps 可能过大。建议优先保存 compact stats：

- query entropy；
- key cumulative importance；
- frame-level attention mass；
- patch-level reduced maps。

### 15.3 Control OOD 风险

不要一开始 hard mask。先使用小 $\beta$ soft bias，并只在 selected early/mid layers 作用。

### 15.4 State reset / merge 语义

LoGeR 的 reset 与 overlap alignment 对 KITTI 结果非常敏感。Controlled pass 必须保持和 native path 一样的 reset-block-aware merge 语义。

---

## 16. 最终总结

Geometry Backbone v2 不再只是“LoGeR forward + TTT write cache”。它要成为 Hybrid Memory Controller 的双通道执行器：

```text
Probe mode:
  read committed memory, collect geometry + traces, do not commit

Controlled mode:
  read same committed memory, inject control, output geometry + commit H_{m+1}
```

这个升级是必要的，因为 LoGeR 的跨 chunk 记忆不是单一 TTT fast weights，而是由 SWA 的局部无损 memory 和 TTT 的长程压缩 memory 共同构成；当前 chunk 的几何输出还受到 frame attention 和 chunk-wise bidirectional attention 直接影响。只有 Geometry Backbone 提供这些 trace 和 control hooks，Hybrid Memory Controller 才能从 TTT-only future write control 扩展为真正的 current-read + future-write hybrid memory control。

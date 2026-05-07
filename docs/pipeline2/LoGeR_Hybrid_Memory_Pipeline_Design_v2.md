# LoGeR + Dynamic Cue + Video Masklet + Memory Control Pipeline 设计文档（v2）

> 本文档是 `LoGeR_Semantic_Prior_Pipeline_Design_v1.md` 的大版本更新。v1 的核心是 **TTT Write Controller**：只在当前 chunk 结束后控制 `W_m \rightarrow W_{m+1}` 的 TTT fast-weight 写入。v2 将其升级为 **Hybrid Memory Controller**：控制 LoGeR hybrid block 中与记忆读取、局部跨 chunk 传递、长程 fast-weight 写入和当前 chunk 内双向几何推理相关的多条路径。
>
> 本文档仍保留 Dynamic Cue Extractor、Video Masklet Front-end 和 Semantic / Memory Prior Generator 的接口，但重新定义它们的下游目标：不再只产生一个 `A_tok` 去调 TTT write，而是产生一组可作用于 **frame attention / SWA / TTT apply / TTT update / chunk-wise bidirectional attention** 的 memory-control fields。

---

## 0. One-line summary

**v2 的核心变化是：从“只控制 TTT 写入”升级为“控制 LoGeR hybrid memory 的读写全过程”。系统先对当前 chunk 做一次 probe forward，提取几何 cue、内部 attention/motion cue、SWA/TTT 记忆交互 cue，然后用这些 cue 再跑一次 controlled forward，同时控制当前 chunk 的 read path 和未来 chunk 的 write path。**

---

## 1. 为什么必须从 TTT Write Controller 升级

### 1.1 v1 的有效边界

v1 的 pipeline 是：

```text
Run current chunk with W_m
    ↓
extract geometry / semantic cues
    ↓
produce A_tok
    ↓
replay TTT update
    ↓
commit W_{m+1}
```

它有一个很清晰的优点：实现上不需要改 LoGeR 当前 chunk 的 forward，只需要在 delayed write-back 阶段重放 TTT update。因此，它适合验证：

$$
\text{当前 chunk 的哪些 token 不应该被写进 long-term fast weights?}
$$

但它也有一个结构性上限：

$$
A_{tok}\text{ 只影响 }W_{m+1}\text{，不影响当前 chunk 的 geometry output。}
$$

如果当前 chunk 的 pose / pointmap 已经在 frame attention、SWA、TTT apply 或 chunk-wise bidirectional attention 中受到动态区域、遮挡边界或不稳定 token 干扰，那么 v1 的 delayed TTT write control 无法修复当前输出。

### 1.2 LoGeR 不是单一 TTT memory

LoGeR 的 hybrid block 不是单纯的 recurrent TTT。单个 block 可以抽象成：

```text
per-frame attention
    ↓
sliding-window attention over previous/current chunks
    ↓
chunk-wise TTT apply/update
    ↓
chunk-wise bidirectional attention
```

其中：

- **frame attention** 提取每帧内部空间特征；
- **SWA** 在相邻 chunks 之间传递无压缩的局部高保真 context；
- **TTT** 用 fast weights 压缩长程全局几何 context；
- **chunk-wise bidirectional attention** 在当前 chunk 内做强几何推理。

因此，一个完整的 memory controller 不应只问：

$$
\text{谁应该写进 TTT fast weights?}
$$

还应该问：

$$
\text{当前 chunk 应该怎样读取 previous chunk 的 SWA memory?}
$$

$$
\text{当前 chunk 的 attention 是否应该避开动态/不稳定 token?}
$$

$$
\text{当前 TTT memory 是否应该对某些 query 少读或多读?}
$$

$$
\text{哪些 tokens 应该作为 reference 被保护，不能被 controller 改坏?}
$$

### 1.3 实验结论对设计的约束

当前 KITTI 01 实验已经说明：

1. native LoGeR reproduction、unity replay parity、variable prior alignment 基本通过；
2. 单纯的 geometry-only suppressive write control 提升极小；
3. eta-mean-preserving reweighting 证明问题不只是“整体写少了”；
4. `-rank(C_dyn)` 只在 TTT branch 0 上有弱但真实收益；
5. branch 1 / branch 2 / all-branch 控制容易伤害 full-sequence trajectory；
6. 显式几何 residual / uncertainty / occlusion 目前不能稳定作为 token ranking 主信号。

这些结论意味着：

$$
\text{TTT-only delayed write 是必要但不充分的控制面。}
$$

因此 v2 的目标不是继续把 `A_tok` 调得更好，而是把 pipeline 升级为：

$$
\boxed{\text{two-pass hybrid memory read/write control}}
$$

---

## 2. 新的问题定义

长视频 chunk-wise LoGeR 推理中，系统在第 $m$ 个 chunk 面对的是一组 hybrid memory state：

$$
\mathcal{H}_m = \{W_m^{TTT}, H_{m-1}^{SWA}, R_m^{ref}\}
$$

其中：

- $W_m^{TTT}$ 是 long-range compressed fast-weight memory；
- $H_{m-1}^{SWA}$ 是 previous chunk 的 local uncompressed context；
- $R_m^{ref}$ 是 register / role / overlap / reset-block reference 信息。

给定当前 chunk：

$$
\mathcal{X}_m=\{I_{m,t}\}_{t=1}^T
$$

v2 要回答的问题从单一写入控制扩展为：

> 当前 chunk 中哪些 token 可以作为可靠几何证据？哪些 token 应该参与当前 attention/read path？哪些 token 应该写入 long-term TTT fast weights？哪些 previous/local memory tokens 应该被 SWA 读作 anchor？哪些 reference tokens 应该保持不动？

最终目标是同时改善：

$$
\text{current chunk geometry output}
$$

和：

$$
\mathcal{H}_m \rightarrow \mathcal{H}_{m+1}
$$

---

## 3. v2 总体流程

### 3.1 Two-pass protocol

v2 推荐采用 two-pass chunk protocol。

```text
Input chunk X_m + committed hybrid state H_m
    │
    ├── Pass 1: Probe Geometry Backbone
    │       ├── run LoGeR normally with H_m
    │       ├── do not commit new memory
    │       ├── output geometry prediction
    │       ├── output TTT write cache
    │       ├── output SWA / attention traces
    │       └── output internal motion / alignment cues
    │
    ├── Stage B: Dynamic / Internal Cue Extractor
    │       ├── explicit geometry cues
    │       ├── internal attention / Gram / key-cosine cues
    │       ├── TTT residual / update-needed cues
    │       └── SWA read-importance cues
    │
    ├── Stage C: Video Masklet Front-end
    │       └── optional semantic reference / protection cues
    │
    ├── Stage D: Memory Control Prior Generator
    │       ├── token dynamic score D_tok
    │       ├── token reliability R_tok
    │       ├── TTT write prior P_ttt_write
    │       ├── TTT read prior P_ttt_read
    │       ├── SWA read prior P_swa_read
    │       ├── attention bias fields B_frame / B_chunk
    │       └── reference protection mask P_ref
    │
    └── Pass 2: Controlled Geometry Backbone + Hybrid Memory Controller
            ├── controlled frame attention
            ├── controlled SWA read / local memory handoff
            ├── controlled TTT apply
            ├── controlled TTT update commit
            ├── controlled chunk-wise bidirectional attention
            └── output geometry + committed H_{m+1}
```

### 3.2 为什么需要两遍

Pass 1 提供当前 chunk 的自诊断信息。Pass 2 用这些信息修正当前 chunk 的推理和写入。

这解决了 v1 的根本限制：

$$
\text{v1 cues only affect future chunks}
$$

而 v2 变成：

$$
\text{v2 cues affect current chunk and future chunks}
$$

### 3.3 是否仍然 chunk-causal

Two-pass 并不使用未来 chunk。Pass 1 和 Pass 2 都只访问当前 chunk $\mathcal{X}_m$、previous committed memory $\mathcal{H}_m$ 和当前 chunk 内部的双向信息。因此它仍然是 chunk-causal，而不是 offline global optimization。

---

## 4. Stage A：Probe / Controlled Geometry Backbone

Stage A 从单模式 forward 升级为三种模式。

### 4.1 Native mode

用于 baseline reproduction：

```text
run LoGeR normally
commit native TTT / SWA states
```

### 4.2 Probe mode

用于 Pass 1：

```text
run LoGeR normally
collect geometry outputs
collect hybrid memory traces
collect TTT write primitives
collect provisional native memory outputs
but do not commit them
```

Probe mode 的输出包括：

| 输出 | 含义 |
|---|---|
| `GeometryOutput_probe` | pointmap / pose / confidence |
| `TokenMeta` | token order / type / frame id / patch meta |
| `TTTWriteCache` | q/k/v/lr/w_old/momentum/op_order |
| `SWAProbeTrace` | SWA q/k/v/attention/previous-token mapping |
| `FrameAttnTrace` | frame-attention q/k or motion statistics |
| `ChunkAttnTrace` | chunk bi-attn q/k or motion statistics |
| `NativeProvisionalState` | native TTT/SWA update result for debug only |

### 4.3 Controlled mode

用于 Pass 2：

```text
run LoGeR from the same committed H_m
inject control bundle into selected memory/read/write paths
commit controlled H_{m+1}
```

Controlled mode 的输入新增：

$$
\mathcal{C}_m^{mem}
$$

也就是 Stage D 生成的 MemoryControlPrior。

Controlled mode 的输出是：

| 输出 | 含义 |
|---|---|
| `GeometryOutput_controlled` | 最终用于评估的几何结果 |
| `HybridMemoryState_next` | 提交给下一 chunk 的 TTT/SWA state |
| `HybridControlDebug` | 各路径控制统计 |

---

## 5. Stage B：Dynamic / Internal Cue Extractor

v2 中 Stage B 不再只输出显式几何 cue。它输出四类 cue。

### 5.1 Explicit geometry cues

沿用 v1：

$$
E_{geo}=\{C_{stat},C_{dyn},C_{occ},C_{unc},C_{anchor}\}
$$

但当前策略是：

- `C_dyn` 可作为弱动态信号；
- `C_unc` 和 `C_occ` 更适合做 reliability gate，不适合直接作为 token ranking；
- `C_stat/C_anchor` 可作为 structure protection，不宜与动态 score 粗暴混合。

### 5.2 Internal motion cues

从 probe trace 中提取 transformer 内部 motionness。

一个通用形式是：

$$
D_{internal}(t,i)=\operatorname{Norm}(\Phi_{attn}(t,i))
$$

其中 $i$ 是 patch token index，$\Phi_{attn}$ 可以来自：

- frame-attention self-attention dispersion；
- chunk-wise bi-attention Q/K feature variance；
- VGGT4D-style Gram similarity statistics；
- MUT3R-style multi-layer attention aggregation；
- LoGeR 已导出的 `attn_dynamic_patch`、`frame_attn_key_cosine_*`、`dyn4d_patch` 等。

### 5.3 TTT residual / update-needed cue

从 TTT write cache 里计算旧 memory 对当前 key/value 的解释误差。

对 TTT layer $l$、branch/head 展平 token $i$：

$$
\hat v_i = f_{W_m}(k_i)
$$

$$
e_i=\frac{\|\hat v_i-v_i\|_2}{\|v_i\|_2+\epsilon}
$$

直觉：

- $e_i$ 低：memory 已能解释当前 observation，不需要强写；
- $e_i$ 高：memory 没解释好，可能需要写；
- 但只有在 token 可靠时才应该写。

因此定义：

$$
U_i=\operatorname{RankNorm}(e_i)
$$

$$
R_i=(1-D_i)(1-C_{unc,i})(1-C_{occ,i})
$$

$$
S_i^{write}=U_i\cdot R_i
$$

### 5.4 SWA read importance cue

从 SWA trace 中估计 current chunk 是否过度读取 previous unstable tokens。

设 previous chunk token $j$ 收到当前 query 的 SWA attention 总量为：

$$
I_j^{swa}=\sum_{l,h,q} A_{l,h,q\rightarrow j}^{swa}
$$

如果 previous token 同时满足：

$$
I_j^{swa}\text{ high},\quad D_j^{prev}\text{ high}
$$

则它很可能是一个不稳定 local anchor。SWA controller 可对其 read 权重降权。

---

## 6. Stage C：Video Masklet Front-end 的新角色

v2 中，Stage C 暂时不作为主控制信号。它主要做三件事：

1. **structure protection**：road / building / sidewalk / guardrail 等可靠结构区域避免被误压；
2. **obvious movable cap**：person / moving car / rider 等在高动态 cue 同时成立时可轻度限制；
3. **debug reference**：解释某些 cue 是否落在语义合理区域。

由于当前 thing 漏检、tracking failure、stuff 边界粗糙仍未解决，语义不应直接覆盖 geometry/internal cue。

推荐规则：

$$
P_i \leftarrow \max(P_i,0.9),\quad i\in \text{reliable structure}
$$

$$
P_i \leftarrow \min(P_i,0.8),\quad i\in \text{movable}\cap\text{high dynamic}\cap\text{high quality mask}
$$

---

## 7. Stage D：Memory Control Prior Generator

Stage D 从 `Semantic Prior Generator` 的单一 `A_tok` 输出升级为 `MemoryControlPrior`。

### 7.1 输入

| 名称 | Shape | 来源 |
|---|---:|---|
| `E_geo` | `[T,H,W,5]` | Dynamic Cue Extractor |
| `D_internal_patch` | `[T,H_tok,W_tok]` | probe attention traces |
| `U_ttt_patch` | `[T,H_tok,W_tok]` 或 per-layer | TTT residual |
| `SWA_importance` | variable | SWA trace |
| `MaskletOutput` | variable | Stage C |
| `TokenMeta` | struct | Geometry Backbone |

### 7.2 输出

| 名称 | Shape | 用途 |
|---|---:|---|
| `D_tok` | `[L_tok]` | token dynamic / instability score |
| `R_tok` | `[L_tok]` | reliability score |
| `U_tok` | `[L_tok]` | update-needed score |
| `P_ttt_write` | `[L_ttt,3,L_tok]` 或 broadcastable | TTT update branch prior |
| `P_ttt_read` | `[L_ttt,L_tok]` | TTT apply read gate |
| `P_swa_read_prev` | previous-token shaped | SWA previous key read prior |
| `B_frame_attn` | callable / compact fields | frame attention bias |
| `B_chunk_attn` | callable / compact fields | chunk bi-attn bias |
| `P_ref` | `[L_tok]` | reference protection mask |
| `ControlDebug` | struct | diagnostics |

### 7.3 核心组合

基础动态分数：

$$
D_i=\operatorname{Fuse}(D_{internal,i}, C_{dyn,i}, D_{sem,i})
$$

可靠性：

$$
R_i=(1-C_{unc,i})(1-C_{occ,i})P_{ref,i}^{soft}
$$

写入必要性：

$$
U_i=\operatorname{RankNorm}(e_i)
$$

TTT 写入 score：

$$
S_i^{ttt-write}=U_i\cdot R_i\cdot (1-D_i)
$$

TTT branch0 dynamic-preserve score 的保守版本：

$$
S_i^{branch0}= -\operatorname{RankNorm}(D_i)
$$

### 7.4 Branch-specific prior

当前实验支持先从 branch 0 开始：

$$
P_{ttt-write}^{(0)}=\operatorname{MP}(S_i)
$$

$$
P_{ttt-write}^{(1)}=1
$$

$$
P_{ttt-write}^{(2)}=1
$$

其中 MP 是 eta-weighted mean-preserving mapping：

$$
\tilde p_i=\operatorname{clip}(1+\alpha\hat S_i,p_{min},p_{max})
$$

$$
p_i=\frac{\tilde p_i}{\frac{\sum_j\eta_j\tilde p_j}{\sum_j\eta_j+\epsilon}+\epsilon}
$$

### 7.5 Sparse route option

借鉴 selective memory update：

$$
M_i=\mathbf{1}[S_i\in\operatorname{TopK}(S)]
$$

$$
p_i=\frac{M_i}{\frac{\sum_j\eta_jM_j}{\sum_j\eta_j+\epsilon}+\epsilon}
$$

未选中 token exact preserve：

$$
p_i=0\quad\text{for unselected write tokens}
$$

但 read-path gate 不应直接为 0，避免当前 chunk OOD。

---

## 8. Stage E：Hybrid Memory Controller

Stage E 取代 v1 的 TTT Write Controller。

### 8.1 控制对象

| 控制对象 | 是否影响当前 chunk | 是否影响未来 chunk |
|---|---:|---:|
| Frame attention bias | yes | indirectly |
| SWA read gating | yes | yes, through local continuity |
| TTT apply read gate | yes | indirectly |
| TTT update/write control | no for current output, yes for next memory | yes |
| Chunk-wise bidirectional attention bias | yes | indirectly |
| SWA handoff / retention | no for current output | yes |

### 8.2 Frame attention control

Soft additive bias：

$$
\tilde A^{frame}=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}+B^{frame}\right)
$$

推荐的 static-to-dynamic suppression：

$$
B^{frame}_{qk}=\beta_f\log(1-(1-D_q)D_k+\epsilon)
$$

### 8.3 SWA read control

对 previous chunk 的 unstable keys 降权：

$$
B^{swa}_{qk}=\beta_{swa}\log(1-D_k^{prev}+\epsilon)
$$

reference tokens 保护：

$$
D_k^{prev}=0\quad\text{if }k\in\mathcal{R}_{protected}
$$

### 8.4 TTT apply control

对 TTT read output 做 query-wise gate：

$$
\tilde H_i = H_i + r_i^{ttt}\cdot f_{W_m}(LN(H_i))
$$

其中：

$$
r_i^{ttt}=\operatorname{clip}(1-\rho_rD_i,r_{min},1)
$$

第一版建议只做 diagnostic，不作为默认主控制，避免破坏 LoGeR 已学到的 global memory read。

### 8.5 TTT update/write control

沿用 delayed replay，但升级为 branch/layer/sparse-aware：

$$
\tilde G_{l}^{(r,h)}=\sum_i \eta_{l,i}^{(r,h)}p_{l,i}^{(r,h)}J_{l,i}^{(r,h)}
$$

branch 0 默认启用，branch 1/2 默认 unity：

$$
p^{(0)}=p_i,
\quad p^{(1)}=1,
\quad p^{(2)}=1
$$

### 8.6 Chunk-wise bidirectional attention control

同 frame attention，但只在 early/mid selected layers 做 soft bias：

$$
B^{chunk}_{qk}=\beta_c\log(1-(1-D_q)D_k+\epsilon)
$$

不建议全层 hard mask。

---

## 9. Reference protection

为了避免 controller 破坏 LoGeR 的全局连续性，以下 tokens 默认保护：

1. register tokens；
2. role tokens；
3. overlap frames；
4. reset block 的第一帧或第一窗口；
5. first chunk reference tokens；
6. 高累计 attention 的 stable tokens；
7. 高质量 structure semantic tokens。

保护方式：

$$
D_i\leftarrow0
$$

$$
P_{read,i}\leftarrow1
$$

$$
P_{write,i}\leftarrow1\quad\text{or no-control, depending on branch}
$$

注意：reference protection 不是让这些 token 强写，而是不让 controller 错误干扰它们。

---

## 10. 运行时状态

v2 的 committed state 不再只是 TTT fast weights。

$$
\mathcal{H}_m=\{W_m^{TTT}, H_{m-1}^{SWA}, M_m^{ref}, D_{m-1}^{prev}\}
$$

其中：

| 名称 | 含义 |
|---|---|
| `W_ttt` | TTT fast weights `w0/w1/w2` |
| `history` | SWA / local KV history if represented in model state |
| `ref_meta` | protected reference token metadata |
| `prev_control_summary` | previous chunk dynamic/reliability summary for SWA read |

---

## 11. Debug 输出

v2 必须输出更完整的 debug，否则无法定位控制是否有效。

```text
HybridControlDebug = {
  mode,
  pass_id,
  D_tok_stats,
  R_tok_stats,
  U_tok_stats,
  frame_attn_bias_stats,
  swa_read_bias_stats,
  ttt_read_gate_stats,
  ttt_write_branch_stats,
  chunk_attn_bias_stats,
  protected_token_count,
  protected_token_types,
  eta_mass_pre_post,
  sparse_route_selected_ratio,
  per_layer_control_enabled,
  per_branch_control_enabled,
}
```

还需要保存可视化：

- `D_tok` 映射回 patch grid；
- `R_tok` 映射回 patch grid；
- TTT residual map；
- SWA previous-key importance map；
- attention bias map；
- protected reference mask；
- controlled vs probe trajectory segment delta。

---

## 12. 实验 gate

### 12.1 Two-pass no-control parity

Pass 1 probe + Pass 2 no-control 必须复现 unity replay。

判断标准：

$$
|ATE_{two-pass-no-control}-ATE_{unity}| < 0.1m
$$

更理想：

$$
<0.05m
$$

### 12.2 Two-pass TTT-only parity

只启用原先最好设置：branch0 dynamic MP-01。

判断标准：

$$
|ATE_{two-pass-ttt-only}-41.3665|<0.15m
$$

### 12.3 Read-path control gate

只启用 frame / SWA / chunk read-path control，不启用 TTT write control。

如果：

$$
ATE<41.0m
$$

说明当前 chunk read path control 有实际价值。

如果：

$$
ATE<40.0m
$$

说明 Hybrid Memory Controller 方向明显优于 TTT-only。

### 12.4 Main-candidate target

新的主模型目标不应低估。建议：

| 级别 | KITTI 01 ATE |
|---|---:|
| continue-invest gate | `< 41.0 m` |
| weak candidate | `< 40.0 m` |
| main candidate | `< 38.0 m` |
| strong target | `< 35.0 m` |
| stretch target | `< 33.0 m` |

---

## 13. 实现迁移计划

### Phase 1：重命名和接口迁移

- `TTTWriteController` → `HybridMemoryController`
- `WriteResult` → `HybridMemoryResult`
- `WriteCacheOutput` → `HybridMemoryCacheOutput`
- `A_tok` → `MemoryControlPrior` 中的一个字段
- `ttt_write_mode` → `hybrid_memory_mode`

### Phase 2：Two-pass 框架

实现：

```text
probe_result = backbone.run_probe(chunk, state_m)
control_prior = prior_gen(probe_result, cues, masklets)
controlled_result = backbone.run_controlled(chunk, state_m, control_prior)
state_{m+1} = controlled_result.hybrid_state_next
```

### Phase 3：只迁移旧 TTT write path

在 HybridMemoryController 内部先复用旧的 delayed replay，实现 v1 功能等价。

### Phase 4：加入 SWA / attention traces

Probe pass 增加 frame/SWA/chunk attention trace hooks。

### Phase 5：加入 read-path control hooks

逐步实现：

1. frame attention soft bias；
2. SWA previous-key bias；
3. chunk bi-attn soft bias；
4. TTT apply read gate。

### Phase 6：加入 sparse write / reference protection

引入：

- branch0 sparse TTT route；
- protected reference token mask；
- SWA local memory protection。

---

## 14. 最终总结

v2 的设计核心可以浓缩为：

> **LoGeR 的 memory 不是只有 TTT fast weights。TTT 提供长程压缩记忆，SWA 提供短程无损局部记忆，frame attention 和 chunk-wise bidirectional attention 决定当前 chunk 的几何表示是否已被动态/不稳定 token 污染。因此，控制器必须从 TTT-only delayed write-back 升级为 two-pass Hybrid Memory Controller，同时控制当前 chunk 的 read path 和未来 chunk 的 write path。**

因此，新的 pipeline 不再是：

```text
cue -> A_tok -> TTT write
```

而是：

```text
probe LoGeR -> internal/geometric/semantic cues -> MemoryControlPrior
    -> controlled LoGeR read/write -> controlled geometry + H_{m+1}
```

这才有机会突破 TTT-only write control 在 KITTI 01 上目前约 41m 附近的上限。

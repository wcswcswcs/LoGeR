# Hybrid Memory Controller 设计文档（v1）

> 本文档替代 `TTT_Write_Controller_Design_v1.md`。旧模块只控制 LoGeR TTT fast-weight 的 delayed write-back；新模块升级为 **Hybrid Memory Controller**，用于控制 LoGeR hybrid memory block 中的多个读写路径：frame attention、SWA、TTT apply、TTT update、chunk-wise bidirectional attention，以及跨 chunk 的 reference / local memory handoff。
>
> 代码层面，旧 `TTTWriteController` 可以作为本模块的 TTT-write 子控制器继续复用，但对外模块名、输入输出、运行时协议和 debug 语义需要升级。

---

## 0. 一句话定位

**Hybrid Memory Controller 的职责，是把 probe pass 中提取的几何、内部 motion、memory alignment 和语义参考信号，转成真正作用在 LoGeR hybrid block 上的控制量：既控制当前 chunk 如何读取 memory 和进行 attention reasoning，也控制当前 chunk 如何写入下一轮 hybrid memory。**

---

## 1. 背景与升级动机

### 1.1 旧 TTT Write Controller 的职责

旧模块接收 `A_tok` 和 `WriteCacheOutput`，重放 LoGeR TTT update，并产生：

$$
W_m^{TTT}\rightarrow W_{m+1}^{TTT}
$$

它只作用于 TTT update path，不改变当前 chunk 的 apply/read path。

### 1.2 为什么不足

LoGeR 的 hybrid block 可以写成：

```text
Frame Attention
    ↓
SWA over previous/current chunks
    ↓
TTT apply / update
    ↓
Chunk-wise Bidirectional Attention
```

因此，当前 chunk 的 pose / pointmap 不只由 TTT fast weights 决定。以下路径都会影响当前 chunk 输出：

1. frame attention 是否把动态/不稳定区域编码进局部 token；
2. SWA 是否从 previous chunk 读取了错误 local anchors；
3. TTT apply 是否向当前 token 注入了不合适的 global memory；
4. chunk-wise bidirectional attention 是否让动态 token 污染静态几何推理。

旧 TTT Write Controller 只能在当前 chunk 结束后改变下一 chunk 的 fast weights，不能修复当前 chunk 内的上述 read-path 干扰。

### 1.3 新设计目标

Hybrid Memory Controller 要回答五个问题：

1. **Frame attention read**：当前帧内部哪些 query/key 交互应该被轻度抑制？
2. **SWA read**：当前 chunk 从 previous chunk 读取哪些 local memory tokens 是可靠的？
3. **TTT apply read**：当前 TTT memory 对哪些 token 的注入应该保留或降低？
4. **TTT update write**：当前 chunk 哪些 token 应该写入 TTT fast weights？哪些 branch / layer 应该被控制？
5. **Chunk attention read**：当前 chunk 内哪些 cross-frame interactions 会传播动态/不稳定干扰？

---

## 2. 设计原则

### 2.1 Two-pass rather than delayed-write-only

第一遍 probe forward 负责收集信号，不提交最终 memory。第二遍 controlled forward 从同一个初始 state 重新运行当前 chunk，并注入控制量。

```text
Probe pass:
  H_m + X_m -> geometry_probe + traces + caches

Controlled pass:
  H_m + X_m + ControlBundle_m -> geometry_final + H_{m+1}
```

这里 $H_m$ 是完整 hybrid memory state，不只是 TTT fast weights。

### 2.2 Soft control before hard masking

对 attention/read path，默认使用 additive logit bias 或 residual gate，而不是直接 hard mask token。原因是 LoGeR 是在无显式 dynamic mask 的分布下训练的，强 hard mask 很容易造成 OOD。

### 2.3 Branch-specific TTT control

当前实验显示同一个 dynamic score 在 TTT branch 0 上有正收益，但 branch 1 会伤害 full ATE，branch 2 只有弱收益。因此默认：

$$
p^{(0)}\neq 1,
\quad p^{(1)}=1,
\quad p^{(2)}=1
$$

之后再通过实验决定是否打开 branch 2 或特定 layer。

### 2.4 Preserve reference tokens

以下 tokens 不应被控制器随意压低：

- register tokens；
- role tokens；
- overlap frames；
- reset-block first window / first frame；
- high cumulative attention stable tokens；
- reliable structure semantic tokens。

这些 tokens 可能承担全局坐标、局部连续性或 chunk 内角色语义。

### 2.5 Separate signal roles

不同信号的职责不能混在一起：

| 信号 | 主要职责 |
|---|---|
| dynamic / motion score `D` | read-path soft suppression, write reliability |
| uncertainty `C_unc` | reliability gate，不直接作为 write ranking 主信号 |
| occlusion `C_occ` | reliability / SWA boundary gate，不直接作为主 ranking |
| TTT residual `U` | update-needed score |
| semantic structure | protection floor |
| semantic movable | mild cap only when dynamic cue also high |

---

## 3. 输入输出定义

### 3.1 输入

```text
HybridMemoryControllerInput = {
  chunk_images,
  committed_state_m,
  probe_geometry,
  probe_trace,
  token_meta,
  memory_cache,
  memory_control_prior,
  controller_config,
}
```

其中：

| 名称 | 含义 |
|---|---|
| `committed_state_m` | 当前 chunk 读取的 committed hybrid memory |
| `probe_geometry` | probe pass 的 pointmap / pose / confidence |
| `probe_trace` | frame/SWA/TTT/chunk attention traces |
| `token_meta` | token order / frame id / type / patch grid |
| `memory_cache` | TTT replay cache、SWA trace、attention trace |
| `memory_control_prior` | Stage D 生成的控制先验 |

### 3.2 MemoryControlPrior schema

```text
MemoryControlPrior = {
  D_tok,
  R_tok,
  U_tok,
  P_ref,
  P_ttt_write,
  P_ttt_read,
  P_swa_read_prev,
  B_frame_spec,
  B_chunk_spec,
  sparse_route_mask,
  semantic_protection,
  debug,
}
```

字段解释：

| 名称 | Shape | 含义 |
|---|---:|---|
| `D_tok` | `[L_tok]` | dynamic / instability score |
| `R_tok` | `[L_tok]` | reliability score |
| `U_tok` | `[L_tok]` | update-needed score |
| `P_ref` | `[L_tok]` | protected reference mask |
| `P_ttt_write` | broadcastable to `[L_ttt,3,H,L]` | TTT update prior |
| `P_ttt_read` | `[L_tok]` or per-layer | TTT apply output gate |
| `P_swa_read_prev` | previous-token shaped | SWA previous key read prior |
| `B_frame_spec` | compact bias spec | frame attention bias specification |
| `B_chunk_spec` | compact bias spec | chunk bi-attn bias specification |
| `sparse_route_mask` | `[L_tok]` or per-layer/branch | optional exact-preserve route |

### 3.3 输出

```text
HybridMemoryResult = {
  geometry_output,
  state_next,
  ttt_state_next,
  swa_state_next,
  debug,
}
```

| 名称 | 含义 |
|---|---|
| `geometry_output` | controlled pass 最终几何结果 |
| `state_next` | 下一 chunk 读取的完整 hybrid memory state |
| `ttt_state_next` | `w0/w1/w2/history` |
| `swa_state_next` | SWA local handoff state |
| `debug` | per-path control diagnostics |

---

## 4. Runtime protocol

### 4.1 Probe pass

Probe pass 从 committed state 开始：

$$
\mathcal{H}_m=\{W_m^{TTT},H_{m-1}^{SWA}\}
$$

执行普通 LoGeR forward：

$$
Y_m^{probe},\; \mathcal{T}_m^{probe},\; \mathcal{K}_m^{write}
=
F_{LoGeR}(X_m,\mathcal{H}_m;\text{collect}=1,\text{commit}=0)
$$

输出：

- probe geometry；
- TTT write cache；
- SWA attention trace；
- frame / chunk attention trace；
- native provisional state for debugging only。

### 4.2 Control prior generation

根据 probe 输出计算：

$$
\mathcal{P}_m^{mem}=G_{prior}(Y_m^{probe},\mathcal{T}_m^{probe},\mathcal{K}_m^{write})
$$

### 4.3 Controlled pass

从同一个 committed state 重新运行：

$$
Y_m^{ctrl},\mathcal{H}_{m+1}^{ctrl}
=
F_{LoGeR}(X_m,\mathcal{H}_m;\mathcal{P}_m^{mem},\text{commit}=1)
$$

注意：controlled pass 不能从 probe pass 的 provisional state 开始，否则会重复写入。

---

## 5. Control fields

### 5.1 Dynamic score

动态 score 可以来自多个来源：

$$
D_i=\operatorname{Fuse}(D_i^{geo},D_i^{internal},D_i^{semantic})
$$

默认建议：

$$
D_i=\operatorname{softor}(D_i^{internal}, C_{dyn,i})
$$

其中 `D_internal` 可来自 attention / Gram / key-cosine / dyn4d features。

### 5.2 Reliability score

可靠性不等同于静态性。推荐：

$$
R_i=(1-C_{unc,i})(1-C_{occ,i})\cdot P_i^{ref-soft}
$$

对于 protected tokens：

$$
P_i^{ref-soft}=1
$$

对于明显 bad regions：

$$
P_i^{ref-soft}\in[0,1]
$$

### 5.3 Update-needed score

TTT residual：

$$
\hat v_i=f_{W_m}(k_i)
$$

$$
e_i=\frac{\|\hat v_i-v_i\|_2}{\|v_i\|_2+\epsilon}
$$

归一化：

$$
U_i=\operatorname{RankNorm}(e_i)
$$

写入 score：

$$
S_i^{write}=U_i\cdot R_i\cdot(1-D_i)
$$

### 5.4 Branch0 dynamic preserve score

当前最小可用主线可保留：

$$
S_i^{branch0}=-\operatorname{RankNorm}(D_i)
$$

这相当于让 dynamic 高的 token 在 TTT branch0 上少主导 update。

---

## 6. Frame Attention Controller

### 6.1 作用范围

Frame attention controller 作用在每帧内部的 early / selected frame attention layers。

### 6.2 Soft bias 公式

设 token $q,k$ 的 dynamic score 分别为 $D_q,D_k$。对 self-attention logits：

$$
\tilde A=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}+B^{frame}\right)
$$

$$
B^{frame}_{qk}=\beta_f\log(1-(1-D_q)D_k+\epsilon)
$$

解释：

- static query attend dynamic key 时被压低；
- dynamic query 自身仍允许一定 dynamic interaction；
- 这是 soft bias，不是 hard mask。

### 6.3 默认参数

| 参数 | 推荐初值 |
|---|---:|
| `beta_frame` | `0.2, 0.5, 1.0` |
| `frame_layer_mode` | `early` |
| `min_bias_clip` | `-4` |
| `reference override` | protected tokens set `D=0` |

### 6.4 Debug

记录：

- bias mean / min / max；
- 被强负 bias 的 query-key pair 比例；
- protected token 比例；
- controlled vs native attention entropy。

---

## 7. SWA Controller

### 7.1 目标

SWA 是 previous chunk 和 current chunk 的短程无损局部 context highway。它的风险是：previous chunk 中的 dynamic / unstable / occlusion-boundary token 也会被作为 local anchor 传递。

### 7.2 SWA read bias

当 current token attend previous token 时：

$$
\tilde A^{swa}=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}+B^{swa}\right)
$$

$$
B^{swa}_{qk}=\beta_{swa}\log(1-D_k^{prev}+\epsilon)
$$

如果 previous token 是 protected reference：

$$
D_k^{prev}=0
$$

### 7.3 SWA handoff / retention

当前 chunk 输出给下一 chunk 的 local SWA state 时，生成：

$$
P_i^{swa-retain}=R_i(1-D_i)\lor P_i^{ref}
$$

第一版不直接删除 token，而是将该值存入 `prev_control_summary`，供下一 chunk 的 SWA read bias 使用。

### 7.4 Debug

- previous tokens attention importance；
- high-dynamic previous tokens 被 attend 的比例；
- SWA read entropy；
- overlap frame token contribution。

---

## 8. TTT Apply Controller

### 8.1 目标

TTT apply 是当前 chunk 从 global fast-weight memory 读取长程信息：

$$
H_i\leftarrow H_i+f_{W_m}(LN(H_i))
$$

如果 high dynamic / unreliable query 从 global memory 读出不稳定 residual，可能影响当前 chunk 表示。

### 8.2 Query-wise gate

$$
H_i\leftarrow H_i+r_i^{ttt-read}\cdot f_{W_m}(LN(H_i))
$$

$$
r_i^{ttt-read}=\operatorname{clip}(1-\rho_{read}D_i,r_{min},1)
$$

### 8.3 默认策略

第一版默认关闭 TTT apply gate，只作为 diagnostic path。因为 TTT apply 是 LoGeR 稳住 global coordinate / scale 的关键路径，过早压低可能造成 trajectory damage。

建议实验顺序：

1. no TTT apply gate；
2. dynamic query mild gate, `rho=0.1`；
3. only high-confidence dynamic gate；
4. only apply gate in selected layers。

---

## 9. TTT Update Controller

旧 TTT Write Controller 作为本节子模块保留。

### 9.1 Baseline replay

对每个 TTT layer/head/branch：

$$
\tilde G_l^{(r,h)}=\sum_i \eta_{l,i}^{(r,h)}J_{l,i}^{(r,h)}
$$

控制后：

$$
\tilde G_{l,ctrl}^{(r,h)}=\sum_i \eta_{l,i}^{(r,h)}p_{l,i}^{(r,h)}J_{l,i}^{(r,h)}
$$

### 9.2 Eta mean preservation

为了避免只改变总写入量：

$$
\tilde p_i=\operatorname{clip}(1+\alpha \hat S_i,p_{min},p_{max})
$$

$$
p_i=\frac{\tilde p_i}{\frac{\sum_j\eta_j\tilde p_j}{\sum_j\eta_j+\epsilon}+\epsilon}
$$

### 9.3 Branch-specific default

$$
p_{l,i}^{(0,h)}=p_i
$$

$$
p_{l,i}^{(1,h)}=1
$$

$$
p_{l,i}^{(2,h)}=1
$$

### 9.4 Sparse write route

借鉴 selective update：

$$
M_i=\mathbf{1}[S_i^{write}\in\operatorname{TopK}(S^{write})]
$$

$$
p_i=\frac{M_i}{\frac{\sum_j\eta_jM_j}{\sum_j\eta_j+\epsilon}+\epsilon}
$$

未选中 token 对该 branch exact no-write。

### 9.5 Reference protection

对于 protected tokens，默认不施加 dynamic suppression：

$$
p_i=1
$$

也可以选择让 protected tokens 参与 native update 而非强写。

### 9.6 Debug

每层记录：

- `m_eta_before` / `m_eta_after`；
- branch0/1/2 prior enabled；
- selected sparse ratio；
- update norm native vs controlled；
- cosine similarity between native and controlled update direction。

---

## 10. Chunk-wise Bidirectional Attention Controller

### 10.1 目标

Chunk-wise bi-attn 是当前 chunk 内强几何推理路径。它直接影响 current output，所以是 v2 的重点控制面之一。

### 10.2 Bias 公式

与 frame attention 类似：

$$
\tilde A^{chunk}=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}+B^{chunk}\right)
$$

$$
B^{chunk}_{qk}=\beta_c\log(1-(1-D_q)D_k+\epsilon)
$$

### 10.3 作用层策略

不建议全层开启。推荐：

- early chunk layers only；
- mid chunk layers only；
- TTT 后紧邻的 chunk bi-attn layer；
- selected insertion indices based on LoGeR block structure。

### 10.4 Debug

- controlled attention entropy；
- dynamic key attention mass；
- static-to-dynamic attention mass；
- per-layer bias stats。

---

## 11. Reference Protection Manager

### 11.1 Protected token sets

```text
ProtectedTokenSet = {
  register_tokens,
  role_tokens,
  overlap_frame_tokens,
  reset_block_anchor_tokens,
  first_chunk_reference_tokens,
  high_cumulative_attention_tokens,
  high_conf_structure_tokens,
}
```

### 11.2 Protection rules

For attention：

$$
D_i\leftarrow0
$$

For TTT write：

$$
p_i\leftarrow1
$$

For sparse write：

protected tokens 可以被强制 selected 或强制 native。第一版建议 **native**：不被 controller 改写。

### 11.3 保护原因

这些 token 可能承载：

- global coordinate anchoring；
- role / register summary；
- adjacent chunk alignment；
- reset 后 continuity；
- scene-scale consistency。

---

## 12. 模式定义

### 12.1 `hybrid_memory_mode=native`

完全使用 LoGeR native memory update。

### 12.2 `hybrid_memory_mode=unity_replay`

重放 TTT update，但所有 priors 为 1，用于 parity。

### 12.3 `hybrid_memory_mode=ttt_write_only`

只启用旧 TTT write controller 子路径，用于复现 v1 / BL-01。

### 12.4 `hybrid_memory_mode=read_path_only`

只控制 frame / SWA / chunk attention，不控制 TTT write。

### 12.5 `hybrid_memory_mode=hybrid`

同时启用 read-path control 和 TTT write control。

### 12.6 `hybrid_memory_mode=probe_only`

只跑 probe，保存 traces 和 dashboard，不输出 controlled result。

---

## 13. API 草案

### 13.1 Python class

```python
@dataclass
class HybridMemoryControlPrior:
    D_tok: torch.Tensor
    R_tok: torch.Tensor
    U_tok: torch.Tensor
    P_ref: torch.Tensor
    P_ttt_write: Optional[torch.Tensor] = None
    P_ttt_read: Optional[torch.Tensor] = None
    P_swa_read_prev: Optional[torch.Tensor] = None
    frame_bias_spec: Optional[dict] = None
    chunk_bias_spec: Optional[dict] = None
    sparse_route_mask: Optional[torch.Tensor] = None
    debug: dict = field(default_factory=dict)

@dataclass
class HybridMemoryResult:
    geometry_output: GeometryOutput
    state_next: dict
    ttt_state_next: dict
    swa_state_next: Optional[dict]
    debug: dict

class HybridMemoryController:
    def run_probe(self, backbone, images, state_m, **kwargs): ...
    def build_control_prior(self, probe_package, cues, masklets): ...
    def run_controlled(self, backbone, images, state_m, control_prior, **kwargs): ...
```

### 13.2 与旧接口的兼容

旧：

```python
wr = TTTWriteController().run(write_cache, A_tok, B_chunk_geo)
ttt_state = {"w0": wr.w0, "w1": wr.w1, "w2": wr.w2}
```

新：

```python
probe = backbone.run_probe(images, state_m, collect_hybrid_trace=True)
prior = prior_gen.run_memory_control(probe, cues, masklets)
result = hmc.run_controlled(backbone, images, state_m, prior)
state_next = result.state_next
```

兼容模式：

```python
result = hmc.run_ttt_write_only(write_cache, A_tok, B_chunk_geo)
```

---

## 14. 实验 gate

### 14.1 Two-pass no-control parity

$$
|ATE_{two-pass-no-control}-ATE_{unity}|<0.1m
$$

### 14.2 TTT-only compatibility

$$
|ATE_{two-pass-ttt-only}-ATE_{BL01}|<0.15m
$$

### 14.3 Read-path value gate

Read-path only：

$$
ATE<41.0m
$$

才说明当前 chunk read control 有价值。

### 14.4 Hybrid model gate

Hybrid mode：

$$
ATE<40.0m
$$

才算明确超过 TTT-only。

$$
ATE<38.0m
$$

才算主模型候选。

$$
ATE<35.0m
$$

才算强目标。

---

## 15. 实现优先级

### Phase 1：重命名和兼容旧 TTT write

- 创建 `hybrid_memory_controller.py`；
- 将旧 `TTTWriteController` 内嵌为 `TTTUpdateSubController`；
- 保留 `ttt_write_only` 模式；
- 确保旧实验可复现。

### Phase 2：Two-pass framework

- `run_probe()` 不提交 state；
- `run_controlled()` 从同一 state 重跑；
- no-control parity 通过。

### Phase 3：Trace collection

增加：

- frame attention motion stats；
- SWA attention stats；
- chunk attention stats；
- TTT residual stats。

### Phase 4：Read-path control hooks

依次实现：

1. frame attention soft bias；
2. SWA previous-key bias；
3. chunk bi-attn soft bias；
4. TTT apply gate。

### Phase 5：Sparse write and reference protection

- branch0 sparse route；
- protected token mask；
- update direction diagnostics。

---

## 16. 最终总结

Hybrid Memory Controller 不是简单把 TTT Write Controller 换个名字。它的本质变化是：

$$
\text{TTT-only future write control}
\rightarrow
\text{hybrid memory current-read + future-write control}
$$

旧模块只能影响：

$$
W_m^{TTT}\rightarrow W_{m+1}^{TTT}
$$

新模块要同时影响：

$$
\text{current chunk attention/read path}
$$

和：

$$
\mathcal{H}_m\rightarrow\mathcal{H}_{m+1}
$$

这使它能够处理 v1 无法解决的问题：当前 chunk 的 frame attention、SWA、TTT apply 和 chunk-wise bidirectional attention 已经被动态或不稳定 token 干扰时，单纯 delayed write-back 不足以修复。v2 的 two-pass controlled forward 才是更合理的下一阶段结构。

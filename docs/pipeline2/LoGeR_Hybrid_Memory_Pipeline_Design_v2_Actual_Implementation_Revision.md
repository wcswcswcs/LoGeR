# LoGeR Hybrid Memory Pipeline v2 实际实现修正版

日期：2026-05-27（Asia/Singapore）

对应旧设计文档：

```text
docs/pipeline2/LoGeR_Hybrid_Memory_Pipeline_Design_v2.md
```

本文件目的：

```text
按当前代码实现与已落盘实验结果，修正旧 v2 设计文档中的理想化描述。
旧文档仍可作为设计意图参考；本文件记录当前 repo 中真正实现、真正可运行、
以及当前实验已经证明或尚未证明的边界。
```

主要代码依据：

```text
run_pipeline_abc_v2.py
loger/pipeline/hybrid_memory_controller.py
loger/pipeline/ttt_write_controller.py
loger/pipeline/semantic_prior_generator.py
loger/pipeline/video_masklet_frontend.py
loger/models/pi3.py
loger/models/layers/attention.py
tools/run_attention_cue_experiment.sh
tools/run_v43_full_candidate.sh
```

---

## 0. 一句话修正

旧文档把 v2 描述为一个已经全面控制 frame attention / SWA / TTT apply /
TTT update / chunk attention 的统一 Hybrid Memory Controller。

当前代码中的真实 v2 是：

```text
一个 two-pass chunk-causal 推理框架已经实现；
多个 read/write hook 和 semantic policy surface 已经接入模型；
但当前最可靠、最成熟、真正支撑 C9 最佳结果的路径仍然是：

    probe_ttt_write commit
    + C9 locked read cue
    + TTT tri-replay
    + chunk-specific tri gamma map
    + commit EMA chunks

semantic READ / context source skip / SWA write / TTT apply 等路径存在，
但多数仍是实验性控制面，不能按旧设计文档当作已验证的统一控制器。
```

---

## 1. 旧设计 vs 当前实际

| 主题 | 旧 v2 设计写法 | 当前代码 / 实验事实 |
|---|---|---|
| 总体目标 | 全面控制 hybrid memory read/write | two-pass 框架已实现，但部署最佳仍由 C9 的 probe-TTT-write 机制主导 |
| memory state | `H_m = {W_TTT, H_SWA, R_ref}` 三类显式状态 | `HybridMemoryState.ttt_state` 是主提交状态，`ttt_state["history"]` 承载当前 SWA/local memory；`swa_state/ref_state` 更多是迁移/扩展占位 |
| Pass 1 / Pass 2 | Probe 诊断后 controlled forward，同时影响当前输出和未来记忆 | 已实现；但在 `probe_ttt_write` 模式下，未来 TTT 记忆从 Pass 1 native probe cache 构建，不是从 Pass 2 controlled tokens 直接写入 |
| Stage B | Dynamic/internal cue 必跑或核心依赖 | 实际按需要运行；很多 C9 / full-online 主线不依赖 Stage B |
| Stage C | VideoMasklet semantic prior 是主要语义来源 | Stage C 已实现且可缓存；但 C9 locked baseline 默认关闭 Stage C，semantic READ 候选才显式打开 |
| Stage D | 统一 MemoryControlPrior 生成多路径控制 | `HybridMemoryControlPrior` 字段很全，但成熟路径是 `P_ttt_write`；read-path 控制多为保守实验面 |
| frame/global read | 通过 attention bias / source skip 控制 | 已接入 `context_source_skip`，支持 bias 和 `compact_kv`；但有效性依候选而定，很多 semantic READ 行有大回归 |
| SWA | 读写两侧都有稳定控制 | SWA overlap/source replacement/write gate 已实现；v43 attribution 显示 C9 的 SWA overlap replacement 对 full ATE 近中性 |
| TTT apply | 当前 chunk 可受 TTT read gate 控制 | hook 存在，但不是当前最佳 C9 的核心解释 |
| TTT write | v1 延迟写入升级为多路径写入 | 当前最强证据仍是 TTT write；特别是 tri-replay 和 chunk gamma map |
| chunk-id | 旧文档倾向机制泛化 | 当前 C9 强依赖 chunk-specific maps；v43 dechunk 显著回归 |
| deployable best | 旧文档未区分成熟度 | 当前 best deployable 仍是 `C9_P0_R2`, ATE `33.7629421029m` |

---

## 2. 当前实际运行协议

当前 `run_pipeline_abc_v2.py` 的核心 chunk loop 可以概括为：

```text
for each chunk:
    optionally reset/load HMC state and merge/gauge state

    Pass 1:
        hmc.run_probe(backbone, chunk_loger, state)
        collect probe geometry, TTT write cache, optional debug/probe cache

    Stage B:
        run DynamicCueExtractor only when needed

    Stage C:
        if stage_c_mode != none and not ignored:
            run cached / inline VideoMasklet semantic frontend
        else:
            semantic prior is absent

    Stage D:
        build PriorOutput by SemanticPriorGenerator or pass-through/noop policy

    HMC:
        hmc.build_control_prior(...)
        hmc.run_controlled(...)

    Commit:
        commit according to hmc_commit_mode
        merge controlled geometry into trajectory output
```

关键点：

```text
1. Pass 1 和 Pass 2 都只使用当前 chunk 与已提交历史状态，仍是 chunk-causal。
2. controlled forward 可以影响当前 chunk 的输出。
3. future memory 如何提交取决于 hmc_commit_mode。
4. 当前 C9 locked 主线使用 probe_ttt_write，而不是单纯 controlled commit。
```

---

## 3. Commit modes 的真实边界

当前代码中重要的 commit mode：

| Mode | 行为摘要 | 当前地位 |
|---|---|---|
| `controlled` | 直接提交 Pass 2 controlled state | 实验可用 |
| `controlled_probe_only` | 运行 controlled 但不提交 | 诊断可用 |
| `probe_native` | 提交 Pass 1 native provisional state | 诊断/对照 |
| `split_ttt_native` | TTT 用 native，SWA/ref 用 controlled | 实验可用 |
| `probe_ttt_write` | 用 Pass 1 native probe cache 重放 TTT write，构建下个 TTT state | 当前 C9 主线 |

`probe_ttt_write` 是理解当前最佳结果的关键：

```text
Pass 2 controlled forward 可以给当前几何输出服务；
但下一步 TTT fast-weight memory 主要由 Pass 1 native probe cache 通过
TTTWriteController 重放/改写后提交。

因此不能把当前实现简单理解为：
    controlled tokens -> directly become future TTT memory

更准确的说法是：
    native probe cache -> TTTWriteController -> committed TTT state
```

相关代码：

```text
loger/pipeline/hybrid_memory_controller.py:
    build_probe_ttt_write_state(...)

loger/pipeline/ttt_write_controller.py:
    TTTWriteController.run(...)
```

---

## 4. 当前 HybridMemoryState

旧文档把 memory 写成：

```text
H_m = {W_m^TTT, H_{m-1}^SWA, R_m^ref}
```

当前代码中更贴近事实的抽象是：

```text
HybridMemoryState:
    ttt_state:
        w0 / w1 / w2
        history
        optional transient_delta
        debug / commit metadata

    swa_state:
        optional / migration placeholder

    ref_state:
        optional / migration placeholder

    meta:
        chunk / commit / debug metadata
```

重要修正：

```text
当前模型实际读取的 SWA/local history 主要在 ttt_state["history"] 中。
swa_state 和 ref_state 字段存在，但不是当前 C9 主线的主要 memory carrier。
```

因此旧文档里的三分法适合概念讲解，但不能作为当前 checkpoint / state 文件
结构的精确说明。

---

## 5. Stage B / C / D 的实际实现

### 5.1 Stage B：DynamicCueExtractor

Stage B 当前是可选诊断/特征阶段，不是所有主线都强依赖。

输出仍可提供：

```text
D_tok / old_dyn
reliability-like cue
geometry / residual / alignment cue
```

但当前最强 C9 路径不是由 Stage B 单独解释。

### 5.2 Stage C：VideoMasklet semantic frontend

Stage C 已实现，来源边界如下：

```text
fine_label_source = MaskletOutput.L_sem
semantic_group_source = MaskletOutput.G_sem
semantic_group_taxonomy = stage_c_coarse_5_groups
```

重要边界：

```text
C9 locked baseline 默认关闭 Stage C。
semantic READ / semantic memory candidates 才显式启用 Stage C。
运行时不使用 GT SemanticKITTI label 作为 action。
```

### 5.3 Stage D：SemanticPriorGenerator / PriorOutput

当前 `SemanticPriorGenerator.run(...)` 会把 cue、masklet、geometry 组合成：

```text
A_mask / A_pix / A_tok
eligibility
semantic fine / group labels
G_sem / L_sem / Q_sem / R_sem
R_frame / R_global / R_swa / R_ttt
```

这些字段可以进入 `HybridMemoryControlPrior`。

但是：

```text
字段存在 != 已经形成部署级成功机制。
v36-v43 多轮实验显示 semantic READ 经常只产生局部信号或轻微 full-online 信号。
```

---

## 6. HybridMemoryControlPrior 的成熟度

当前 dataclass 已经包含多路径控制字段，例如：

```text
D_tok / R_tok / U_tok / P_ref / S_tok
P_ttt_write
P_ttt_read
P_swa_read_prev
D_swa_write_tok
frame_bias_spec
semantic role streams
```

实际成熟度需要分层理解：

| 控制面 | 代码状态 | 实验成熟度 |
|---|---|---|
| `P_ttt_write` | 已接入 TTTWriteController | 最成熟，C9 主要依赖 |
| frame/global context source skip | 已接入 Pi3 attention | 有局部信号，但 full-online 容易回归 |
| `compact_kv` source removal | 已实现真实 K/V column compaction | 可审计，但不是 deployable best |
| SWA overlap/read/write controls | 已接入 | 多数 full-online 贡献有限或不稳定 |
| TTT apply gate | 已接入 | 当前不是 C9 主要正贡献 |
| semantic role paths | 已接入多种 policy | 仍需更强因果/空间证据 |

---

## 7. READ / context source skip 的实际机制

当前 READ 干预主要通过 Pi3 中的 HMC control dict 进入 attention。

典型路径：

```text
HybridMemoryController._build_model_hmc_control(...)
    -> Pi3._make_context_source_skip_bias(...)
        -> bias mask or compact_kv mask
        -> attention block
```

`context_source_skip_impl` 支持：

```text
bias:
    对被选中的 source K/V column 加 attention bias / hard mask。

compact_kv:
    保留所有 query；
    真正压缩 / 移除 source K/V columns；
    可记录 compact 前后的 sampled attention mass。
```

当前实现会保护一些 reference / special tokens，并记录诸如：

```text
context_source_skip_applied
context_source_skip_tokens
context_source_keep_ratio
context_empty_source_events
attention_mass_requested
```

重要边界：

```text
context_empty_source_events = 0 只能说明 source 没被删空；
不能自动证明语义 causality 或 full-online 改善。
```

---

## 8. semantic_action_active_chunks 的真实含义

v42 后新增了 health-selected semantic action gate：

```text
semantic_action_active_chunks
semantic_action_inactive_read_cue_source
```

实际语义：

```text
在 selected chunks 内：
    semantic_role_policy / semantic_memory_paths / context_source_skip 可以生效。

在 selected chunks 外：
    semantic action 关闭；
    read cue 可以回退到 C9 cue；
    Stage-C semantic prior 不应扰动非选中 chunk。
```

边界：

```text
这是一个 chunk gate，不是 learned persistence detector。
v42 的 full-online 结果显示，虽然 gate 工程上生效，但 health-selected
semantic READ rows 全部回归 C9。
```

---

## 9. TTT write 的当前真实核心

当前 C9 最强 evidence 指向 TTT write，而不是 semantic READ。

C9 locked 配置摘要：

```text
hybrid_memory_mode = hybrid
hmc_commit_mode = probe_ttt_write
hmc_write_score_source = stage_d_x_dg_inv_sqrt
read_cue_source = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
mp_alpha = 0.1
read_beta_frame_chunks = 5-9:4.85, 10-12/16:4.25
ttt_write_gradient_reversal_mode = tri_replay
ttt_write_gradient_reversal_chunk_gammas =
    5-9:0.005, 10-12:0.003, 16:0.0003
```

v43 component attribution 的关键落盘结论：

| 移除项 | ATE | Delta vs C9 | 解释 |
|---|---:|---:|---|
| remove TTT tri-replay | `36.2098947787m` | `+2.4469526758m` | 最大正贡献来自 tri-replay |
| flatten tri gamma chunk map | `34.7339675087m` | `+0.9710254058m` | chunk-specific gamma map 很重要 |
| disable commit EMA | `34.2513275668m` | `+0.4883854639m` | commit EMA 中等正贡献 |
| flatten read beta map | `33.7895226653m` | `+0.0265805624m` | read beta chunk map 近中性 |
| disable SWA overlap replace | `33.8191923480m` | `+0.0562502451m` | SWA overlap replacement 近中性 |
| native mix to 1.0 | `33.8546448528m` | `+0.0917027499m` | native mix 近中性 |

因此当前更准确的机制判断是：

```text
C9 的 full-online 表现主要来自 TTT tri-replay + chunk-specific gamma /
commit behavior，而不是旧 v2 文档强调的完整 semantic hybrid read/write
controller。
```

---

## 10. semantic READ 的当前状态

semantic READ 已经不是纯设想，代码和 full-online row 都已经跑通。

v43 的最佳 semantic READ：

```text
SEM_READ_03_C23_RESID_READ_ONLY
ATE = 33.4875667508m
delta vs C9 = -0.2753753521m
```

这是一个真实的 full-online 正信号。

但边界也必须写清楚：

```text
它没有达到 v43 minimum progress:
    required improvement >= 0.3m or ATE <= 33.3m

observed:
    improvement = 0.2753753521m
    ATE = 33.4875667508m

因此它是 diagnostic positive signal，不是 deployable online success。
```

多轮实验还显示：

```text
semantic / appearance / health-gated READ 经常在局部窗口改善，
但容易出现 h15 washout、downstream regression 或 full-online regression。
```

---

## 11. C9 dechunk 的当前结论

旧设计倾向把 chunk map 看成可被更通用机制替代的实现细节。

v43 已经证明：

```text
当前 C9 不能直接 dechunk。
```

落盘结果：

| Candidate | ATE | Delta vs C9 |
|---|---:|---:|
| `FLAT_01` | `35.2952180149m` | `+1.5322759119m` |
| `FLAT_02` | `35.5004971353m` | `+1.7375550324m` |
| `FLAT_03` | `35.3608497931m` | `+1.5979076901m` |
| `FLAT_04` | `36.5229452729m` | `+2.7600031699m` |

结论：

```text
C9 是有效但 chunk-map-heavy 的 diagnostic best。
如果未来要机制洁净化，需要先替代或解释 TTT tri-replay / tri gamma
chunk map，而不是直接抹平 chunk-id 结构。
```

---

## 12. 当前不应再声称的内容

以下说法不符合当前代码与实验边界：

```text
1. v2 已经是统一成熟的 hybrid read/write controller。
2. semantic prior 已经是当前 best deployable 的核心来源。
3. Stage C semantic label 在 C9 baseline 中默认参与 runtime action。
4. C9 可以去 chunk-id map 而不损失性能。
5. context source skip / compact_kv 的 action realism 等于 full-online success。
6. 局部 [200,300) 或 rolling100 改善可以直接当作 Target-30 证据。
7. proxy overlay / scalar attention mass 可以证明 per-label spatial causality。
8. `swa_state` 是当前主要 SWA memory carrier。
```

更准确的说法：

```text
v2 代码已经提供丰富的 hybrid memory control hook surface。
当前最佳部署证据仍是 C9 locked probe-TTT-write recipe。
semantic READ 出现了小的 full-online 正信号，但尚未达到部署边界。
```

---

## 13. 当前实现文件索引

核心 pipeline：

```text
run_pipeline_abc_v2.py
```

HMC state / control prior / two-pass wrapper：

```text
loger/pipeline/hybrid_memory_controller.py
```

TTT write / tri-replay / commit EMA / scale-state / native mix：

```text
loger/pipeline/ttt_write_controller.py
```

semantic prior：

```text
loger/pipeline/semantic_prior_generator.py
loger/pipeline/video_masklet_frontend.py
```

Pi3 hook sites：

```text
loger/models/pi3.py
loger/models/layers/attention.py
```

launcher / locked C9 recipe：

```text
tools/run_attention_cue_experiment.sh
tools/run_v43_full_candidate.sh
docs/C9_P0_R2_Pipelinev2_Configuration_Explainer.md
```

---

## 14. 建议的新设计方向

基于当前实际实现，后续文档不要再把目标写成“全面控制所有 memory path”。
更可审计的路线应拆成三条：

```text
1. C9 mechanism preservation:
       保留 TTT tri-replay / tri gamma / commit EMA 的正贡献，
       尝试把 chunk-id map 替换成可解释的 trajectory / scale / gauge state。

2. semantic READ minimal additive path:
       以 C9 locked 为 base，
       只允许像 SEM_READ_03 这种小扰动、无严重 downstream regression 的路径继续。

3. causality instrumentation:
       落地 per-selected-chunk spatial attention / affected-source maps /
       tensor-state snapshots，
       避免只靠 proxy overlay 或 scalar mass 解释 causality。
```

验收边界也应保持严格：

```text
short rollout / h10 / proxy attribution:
    diagnostic only

full-online KITTI01:
    minimum progress boundary 才允许组合或跨序列

Target-30:
    只有 full-online ATE <= 30m 才能声称
```

---

## 15. 修正版结论

当前 repo 中的 LoGeR Hybrid Memory Pipeline v2 应被描述为：

```text
一个已经实现 two-pass probe/control、并在 Pi3/HMC/TTT 层接入多种
read/write hook 的实验平台。

它的当前 deployable best 不是 semantic unified controller，而是
C9_P0_R2:
    probe_ttt_write
    TTT tri-replay
    chunk-specific tri gamma
    commit EMA
    C9 locked read cue

semantic READ 已经产生小的 full-online 正信号，但尚未越过部署或 Target-30
边界。
```

因此，旧 v2 设计文档应作为“目标架构/意图”阅读；本文件才是截至
2026-05-27 当前代码实现和实验边界的修正版说明。

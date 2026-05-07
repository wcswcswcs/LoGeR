# KITTI 01 Hybrid Memory Controller 新实验方案

> 版本：v1.0  
> 目标序列：KITTI Odometry 01  
> 当前方法名建议：**LoGeR Hybrid Memory Controller, HMC**  
> 核心实现方式：**two-pass chunk processing**，即同一个 chunk 先 probe，再从同一个 committed memory state 重新 controlled forward。  
> 约束：本方案的正确性验证必须走新的 HMC pipeline，不能用旧 `run_pipeline_abc.py --geometry_eval_mode native` 或旧 TTT-only controller 作为“捷径”产出最终 baseline。

---

## 0. 当前结论与为什么必须换实验范式

前几轮实验已经把很多低层风险排除了。原始 LoGeR 和 LoGeR* 在 KITTI 01 上都能复现论文表格附近的指标；unity replay 也能和 native LoGeR 接近；variable prior 的 patch、special、frame ramp、roll 测试没有发现明显 token 对齐灾难；branch/layer selective 进一步说明，当前 `-rank(C_dyn)` 信号只在 TTT branch 0 上有稳定价值。当前最好的有效结果是：

$$
ATE_{unity}=41.6193\text{ m},
$$

$$
ATE_{BL01}=41.3665\text{ m}.
$$

这个结果比 unity replay 好，但提升只有约 $0.25$ m，不能支撑“主模型”的说法。它更像一个诊断信号：**LoGeR 内部确实有 motion / update 相关信息，但只控制 delayed TTT write 的收益上限很低。**

原因很明确。LoGeR 的单个 hybrid block 不是单独的 TTT update，而是：

```text
per-frame attention
    → SWA over previous/current chunks
    → chunk-wise TTT apply/update
    → chunk-wise bidirectional attention
```

TTT fast weights 负责长程、压缩的全局记忆；SWA 负责相邻 chunk 的无损局部对齐；frame attention 和 chunk-wise bidirectional attention 则直接影响当前 chunk 的几何推理。旧 TTT Write Controller 只改变 $W_m \rightarrow W_{m+1}$，也就是只影响未来 chunk。它无法修复当前 chunk 中 frame attention、SWA read、TTT apply 或 chunk-wise attention 已经造成的错误传播。

因此，下一阶段的实验不应该再围绕 `lambda/floor/alpha` 做细扫，而要验证一个更强的中心假设：

$$
\boxed{
\text{KITTI 01 的主要改进空间不只在 TTT write，而在 LoGeR hybrid memory 的 read/write 全链路。}
}
$$

新的目标不是把 TTT Write Controller 调得更好，而是构建并验证一个 **Hybrid Memory Controller**：它先通过 probe pass 让 LoGeR 自己诊断当前 chunk 中的动态、SWA 读取、TTT 残差和 attention 交互，再在 controlled pass 中同时控制 frame attention、SWA read、TTT apply、TTT update 和 chunk-wise bidirectional attention。

---

## 1. 新 pipeline 的基本协议

新 pipeline 以 chunk 为单位。处理第 $m$ 个 chunk 时，输入为：

$$
X_m=\{I_{m,1},I_{m,2},\dots,I_{m,T}\},
$$

以及上一轮已经正式提交的 hybrid memory：

$$
\mathcal H_m=
\{W_m^{TTT}, H_m^{SWA}, R_m^{ref}, S_m^{summary}\}.
$$

其中 $W_m^{TTT}$ 包含 LoGeR TTT 的三组 fast weights：

$$
W_m^{TTT}=\{w0_m,w1_m,w2_m\},
$$

$H_m^{SWA}$ 表示相邻 chunk 的局部 feature / history 状态，$R_m^{ref}$ 表示 overlap、reset-boundary、special tokens 等 reference metadata，$S_m^{summary}$ 保存上一轮 controller 的 debug 统计。

新 pipeline 的核心运行协议如下：

```text
For chunk m:

  Input: X_m, committed memory H_m

  Pass 1: Probe forward
      - 从 H_m 开始跑 LoGeR 原生 forward
      - 允许计算 provisional update，但绝不提交
      - 收集 geometry / attention / SWA / TTT cache / residual traces

  Stage B: Hybrid cue extraction
      - 显式 geometry cue
      - internal motion cue
      - TTT residual / update-needed cue
      - SWA read / previous-token importance cue

  Stage C: optional semantic reference
      - 只做 protection / cap，不做主导控制

  Stage D: MemoryControlPrior generation
      - 生成 frame-attn bias, SWA read gate, TTT apply/read gate,
        TTT branch-wise update prior, chunk-bi-attn bias, reference protection

  Pass 2: Controlled forward
      - 从同一个 H_m 重新跑同一个 X_m
      - 注入 MemoryControlPrior
      - 只提交 Pass 2 的 geometry output 和 H_{m+1}

  Output: geometry_m^{ctrl}, committed memory H_{m+1}^{ctrl}
```

这里有一个绝对约束：

$$
\boxed{
\text{Pass 1 的 provisional memory 不允许成为 Pass 2 的输入。}
}
$$

Pass 2 必须从同一个 committed $\mathcal H_m$ 重新开始。否则同一个 chunk 等价于被写入两次，指标即使改善也没有意义。

---

## 2. 总体目标与判断标准

本轮实验的目标分为两层。第一层是工程正确性：新的 two-pass HMC pipeline 必须在 no-control / identity-control 情况下复现 LoGeR 和 LoGeR* 的 KITTI 01 指标。第二层才是模型有效性：HMC 必须显著超过 LoGeR，而不是只比 41.6 m 好几个厘米。

### 2.1 正确性 gate

LoGeR 原论文表格中，KITTI 01 上的 LoGeR ATE 约为：

$$
ATE_{LoGeR,paper}=41.64\text{ m}.
$$

我们之前用当前环境复现到：

$$
ATE_{LoGeR,prev}=41.7502\text{ m}.
$$

LoGeR* 在论文表格中 KITTI 01 约为：

$$
ATE_{LoGeR^*,paper}=47.91\text{ m},
$$

之前复现为：

$$
ATE_{LoGeR^*,prev}=47.9793\text{ m}.
$$

新的 HMC pipeline 必须重新复现这两个结果，并且不能通过旧的 geometry-only runner 或旧的 TTT-only controller 绕过。允许旧结果作为对照 reference，但最终 trajectory 必须由新 `HybridMemoryRunner` 产出。

正确性 gate 的通过标准如下：

| Gate | 要验证的内容 | 通过标准 |
|---|---|---:|
| G0 | Pass 1 不提交 memory | Pass 1 前后 committed $\mathcal H_m$ hash 完全不变，或 tensor diff 为 0 |
| G1 | two-pass no-control 单 chunk parity | Pass 1 output 和 Pass 2 no-control output 的 pose mean diff $<10^{-4}$ m，max diff $<10^{-3}$ m；若 bfloat16 有非确定性，可放宽到 mean $<10^{-3}$ m |
| G2 | identity hooks parity | frame/SWA/TTT/chunk hooks 全开但 bias 为 0、gate 为 1，full KITTI 01 ATE 与 no-control two-pass 差 $<0.05$ m，最多 $<0.10$ m |
| G3 | LoGeR full KITTI 01 reproduction | 新 pipeline 产出的 ATE 与 $41.7502$ m 差 $<0.5$ m，理想 $<0.2$ m |
| G4 | LoGeR* full KITTI 01 reproduction | 新 pipeline 产出的 ATE 与 $47.9793$ m 差 $<0.5$ m，理想 $<0.2$ m |
| G5 | HMC TTT-only reproduction | 新 HMC 的 TTT branch0-only 设置复现 $41.3665$ m，误差 $<0.15$ m |

如果 G0–G4 任一失败，不进入模型实验。G5 失败时，说明新 HMC 的 TTT 子控制器没有兼容旧最佳结果，不能把后续差异解释成 read-path controller 的收益。

### 2.2 模型有效性 gate

本轮实验不能再把 $41.25$ m 或 $41.0$ m 当作成功。它们最多是“机制有点信号”的 debug gate。新的分层标准如下：

| 等级 | KITTI 01 ATE 目标 | 含义 |
|---|---:|---|
| Debug signal | $<41.0$ m | 说明 HMC 比 BL-01 有实质进展，但仍不是主模型 |
| Weak candidate | $<40.0$ m | 说明读写全链路控制开始有效，值得继续消融 |
| Main candidate | $<38.0$ m | 相比 LoGeR 有约 9% 改善，可作为主线候选 |
| Strong target | $<35.0$ m | 有明显方法价值，接近“显著超越” |
| Stretch target | $<33.0$ m | 达到最初设想的强目标区间 |

模型实验里，只要一个设置 full KITTI 01 不能低于 $40.0$ m，就不能称为主方法，只能作为诊断方向。低于 $38.0$ m 才开始写成 “main candidate”。低于 $35.0$ m 才值得围绕它做完整 semantic / layer / branch / reset / robustness 扩展。

---

## 3. 无捷径正确性协议

为了避免“看起来复现了 LoGeR，其实绕回旧代码”的问题，本方案规定所有 baseline 和候选都必须通过新 runner 产生。

建议新增统一入口：

```bash
python run_hybrid_memory_pipeline.py \
  --input <KITTI01 image_2> \
  --checkpoint <LoGeR checkpoint> \
  --config <LoGeR config> \
  --hmc_mode no_control|identity_hooks|controlled \
  --two_pass 1 \
  --commit_source pass2 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --window_size 32 \
  --overlap_size 3 \
  --reset_every 5 \
  --output_txt <path>/01.txt
```

旧 `run_pipeline_abc.py` 可以继续作为历史 reference 或 debug 对照，但不能作为新 pipeline 的正式结果来源。尤其是：

1. 不能用旧 `--geometry_eval_mode` 直接生成 no-control baseline。
2. 不能用旧 `TTTWriteController` 直接生成 BL-01 结果后冒充 HMC 的 TTT subcontroller。
3. 不能在 LoGeR* 上额外做非论文设置的 pose post-alignment。LoGeR* 只允许使用原配置中的 SE(3) overlap alignment 和 reset 策略。
4. 不能提交 Pass 1 的 provisional memory。必须明确记录 `commit_source=pass2`。
5. identity hooks 必须真的穿过所有新 hook 位置：frame attention、SWA、TTT apply、TTT update、chunk-wise attention。不能因为 hook disabled 而通过 parity。

每个新 runner 输出目录必须包含：

```text
01.txt
hmc_config.yaml
hmc_state_hash.jsonl
hmc_probe_summary.jsonl
hmc_control_summary.jsonl
hmc_hook_identity_check.json
kitti_benchmark.log
```

其中 `hmc_state_hash.jsonl` 至少记录：

```text
chunk_id
hash_H_m_before_probe
hash_H_m_after_probe
hash_H_m_before_pass2
hash_H_m_after_commit
hash_H_{m+1}
```

期望关系为：

$$
hash(\mathcal H_m^{before\ probe})=hash(\mathcal H_m^{after\ probe})=hash(\mathcal H_m^{before\ pass2}).
$$

---

## 4. Phase A：新 pipeline 工程正确性验证

### 4.1 假设

本阶段验证的假设不是“controller 有效”，而是：

$$
\mathcal P_{HMC}^{no\ control}
\equiv
\mathcal P_{LoGeR}^{native}.
$$

也就是说，当所有控制量为 identity 时，新 pipeline 必须退化为 LoGeR 本身。这个 gate 一定要在 full KITTI 01 上做，因为之前多次发现 64/128-frame prefix 不能预测 full sequence 行为。

### 4.2 实验 A0：Pass 1 no-commit 单 chunk 验证

先用 KITTI 01 的前 32 或 64 帧进行单 chunk smoke。运行 two-pass，但 Pass 2 不开启任何控制。

需要记录：

$$
\Delta_{state}^{probe}=\|\mathcal H_m^{after\ probe}-\mathcal H_m^{before\ probe}\|.
$$

如果实现上 state 是 dict/list/tensor，建议对所有 tensor 做 hash 和 max abs diff。通过标准：

$$
\Delta_{state}^{probe}=0.
$$

如果某些模块因为临时 buffer 或 lazy init 导致 hash 不同，则必须证明 committed memory 的 `w0/w1/w2/history/SWA cache` 没有改变。

### 4.3 实验 A1：Pass 1 vs Pass 2 no-control geometry parity

同一个 chunk 从同一个 $\mathcal H_m$ 跑两次，Pass 2 不控制。比较：

$$
\Delta T=rac{1}{T}\sum_t\|t_t^{pass1}-t_t^{pass2}\|_2,
$$

$$
\Delta R=rac{1}{T}\sum_t d_R(R_t^{pass1},R_t^{pass2}).
$$

还要比较 pointmap：

$$
\Delta P=rac{1}{THW}\sum_{t,u}\|P_{t,u}^{pass1}-P_{t,u}^{pass2}\|_1.
$$

通过标准：pose mean translation diff $<10^{-4}$ m，pointmap mean diff $<10^{-5}$ 到 $10^{-4}$ 量级；如果 autocast 非确定性较大，可放宽但必须在 identity hooks 和 full-sequence指标中继续验证。

### 4.4 实验 A2：Identity hooks full KITTI 01 parity

开启所有新 hook，但全部设成 identity：

```text
frame attention bias = 0
SWA read bias = 0
TTT apply read gate = 1
TTT update branch priors = 1
chunk-wise attention bias = 0
reference protection no-op
```

这一步不能通过关闭 hook 完成，必须证明数据真的经过 hook interface。

运行 full KITTI 01，LoGeR 普通 checkpoint：

```text
hmc_mode = identity_hooks
checkpoint = ckpts/LoGeR/latest.pt
config = ckpts/LoGeR/original_config.yaml
chunk_size = 32
window_size = 32
overlap = 3
reset_every = 5
geometry_edge_rtol = 0.0
```

期望：

$$
|ATE_{identity}-ATE_{no\ control}|<0.05\text{ m},
$$

最大允许：

$$
<0.10\text{ m}.
$$

### 4.5 实验 A3：新 pipeline 复现 LoGeR

使用 `hmc_mode=no_control` 和 `commit_source=pass2` 跑 full KITTI 01。这个结果必须由新 runner 产生。

通过标准：

$$
|ATE_{HMC,no\ control}^{LoGeR}-41.7502|<0.5\text{ m},
$$

理想标准：

$$
<0.2\text{ m}.
$$

如果它和论文表格 $41.64$ m 接近但和之前复现 $41.7502$ m 差大，也要检查输入分辨率、edge suppression、merge、reset、chunk/window size 是否完全一致。

### 4.6 实验 A4：新 pipeline 复现 LoGeR*

LoGeR* 使用 SE(3) overlap alignment 路径，配置要和 LoGeR* checkpoint 一致：

```text
checkpoint = ckpts/LoGeR_star/latest.pt
config = ckpts/LoGeR_star/original_config.yaml
se3 = true
chunk_size = 64
window_size = 64
overlap = 3
reset_every = 5
geometry_edge_rtol = 0.0
```

通过标准：

$$
|ATE_{HMC,no\ control}^{LoGeR^*}-47.9793|<0.5\text{ m},
$$

理想标准：

$$
<0.2\text{ m}.
$$

LoGeR* 复现不是为了主模型指标，而是为了证明新 pipeline 对 LoGeR 的 reset / overlap / pose merge 逻辑没有破坏。后续 HMC 若只在 LoGeR 上优化，不代表代码可以跳过 LoGeR* 复现。

### 4.7 实验 A5：HMC TTT-only 复现旧 BL-01

在新 HMC pipeline 中只开启 TTT Update Subcontroller，其他 read-path hooks identity。设置：

```text
score = -rank(C_dyn)
prior_policy = eta_mean_preserving
alpha = 0.1
range = [0.8, 1.2]
branch_mask = 0
layer_mode = all
special = 1.0
lambda = 1.0
Stage C = none
```

期望复现旧结果：

$$
ATE_{BL01}=41.3665\text{ m}.
$$

通过标准：

$$
|ATE_{HMC,TTT-only}-41.3665|<0.15\text{ m}.
$$

如果这一步失败，说明 TTT Update Subcontroller 在新 HMC 中与旧实现不一致，后续 read-path 实验不能直接和旧 BL-01 比较。

---

## 5. Phase B：Probe trace 和控制信号仪表盘

### 5.1 假设

本阶段验证的假设是：

$$
\text{Pass 1 能提取出可解释、可对齐、可复现的 hybrid-memory signals。}
$$

在模型实验之前，必须先看这些 signal 是否有基本合理性。否则 read-path controller 失败时，无法判断是 cue 错、hook 错，还是控制公式错。

### 5.2 需要收集的 trace

Probe pass 至少要收集五类信息。

第一类是显式几何 cue：

```text
C_stat, C_dyn, C_occ, C_unc, C_anchor
G_write_geo
```

这些来自 pointmap、pose 和 confidence，作为旧 Stage B 的延续。

第二类是 internal motion cue：

```text
attn_dynamic_patch
frame_attn_key_cosine_shallow/deep/avg
frame_attn_cosine_query/key_layers
dyn4d_patch
global_q/k raw patch vectors if available
```

需要统一投影到 patch token：

$$
D_i\in[0,1],\quad i=1,\dots,L_{patch}.
$$

第三类是 TTT update-needed cue。对每个 TTT layer、branch、head，计算或近似：

$$
\hat v_i=f_{W_m}(k_i),
$$

$$
e_i=\frac{\|\hat v_i-v_i\|_2}{\|v_i\|_2+\epsilon}.
$$

如果直接复现 fast-weight SwiGLU forward 成本过高，第一版可以先对 cache 中的 branch0/1/2 update gradient norm 做替代：

$$
g_i^{(r)}=\|\eta_i^{(r)} J_i^{(r)}\|_F.
$$

第四类是 SWA trace：

```text
current token → previous chunk token attention
previous token cumulative read importance
previous token frame id / type / overlap status / D score
```

第五类是 chunk-wise bidirectional attention trace：

```text
query frame id
key frame id
attention mass across frame pairs
attention mass to high-D tokens
attention mass from static queries to dynamic keys
```

### 5.3 代表片段可视化

先不要只看 full ATE。必须固定几个代表片段做 dashboard：

```text
[0,200)       旧 MP 系列伤害段
[300,500)     旧 MP 系列收益段
[400,600)     旧 MP 系列明显伤害段
[800,1000)    后期 drift/伤害段
```

每个片段抽 5–10 帧，保存：

```text
RGB
C_dyn / C_unc / C_occ / C_anchor
D_internal
TTT residual e_i / update-needed U_i
SWA previous-token read importance
frame-attn dynamic interaction heatmap
chunk-attn dynamic interaction heatmap
final P_ref / P_ttt_write / P_swa_read / B_frame / B_chunk
```

### 5.4 判断标准

这个阶段没有 ATE 成功标准，但有“是否值得进入控制实验”的标准。

如果在 road/building/guardrail 上 $D_i$ 大面积高，而 moving car 上 $D_i$ 不明显，则 internal motion cue 不能直接用于 read-path gating。若 moving car / dynamic boundary 上 $D_i$ 高，但 road/building 低，则进入 Phase C。

如果 TTT residual $e_i$ 在 moving car 和可靠 road/building 上都高，但 reliability $R_i$ 能把 moving car 降下来，则 update-needed score 有希望。若 $e_i$ 完全无空间结构，先不要用 residual 做 sparse routing。

如果 SWA read importance 大量集中在 previous high-dynamic tokens，SWA Controller 值得优先做。若 SWA read 几乎都集中在 overlap static/reference tokens，SWA 可能不是主要瓶颈。

---

## 6. Phase C：read-path controller 单路径验证

### 6.1 中心假设

本阶段验证最关键的假设：

$$
\boxed{
\text{delayed TTT write-only 低收益，是因为当前 chunk read path 已经被污染。}
}
$$

如果这个假设成立，那么在 TTT update 保持 unity 的情况下，只控制 frame attention、SWA read、TTT apply 或 chunk-wise attention，应当能改善 full KITTI 01 ATE。反过来，如果 read-path 单独完全无效，说明我们的 HMC 大改不一定带来核心收益，需要回到更强 cue 或语义/flow。

### 6.2 Frame Attention Controller

Frame Attention Controller 的目标是防止静态结构 query 过度读取 dynamic key。Pass 2 中对 selected early frame-attention layers 加 soft bias：

$$
B_{qk}^{frame}
=
\beta_f\log\left(1-(1-D_q)D_k+\epsilon\right).
$$

这里 $D_q,D_k$ 是 probe pass 得到的 patch-level dynamic score。这个公式的含义是：静态 query 对 dynamic key 降权；dynamic query 仍然可以 attend dynamic key，不会彻底冻结动态区域。

第一版只在 early frame-attention blocks 使用，例如 LoGeR block depth 的前 $1/3$，或 frame-attention 插入层中前 4–6 个位置。初始强度：

$$
\beta_f\in\{0.25,0.5,1.0\}.
$$

实验命名：

```text
R-FR-025
R-FR-050
R-FR-100
```

设置：TTT update unity，SWA identity，TTT apply identity，chunk attention identity。

判断标准：

- 若任一设置 $ATE<41.0$ m，说明 read-path control 有真实信号。
- 若 $ATE<40.0$ m，frame attention controller 成为主候选路径。
- 若全部 $>41.6$ m 或局部 trajectory 崩，说明 $D_i$ 或 layer selection 不适合 frame attention。

### 6.3 SWA Controller

SWA Controller 的目标是控制 previous chunk local memory 的读取。SWA 是 LoGeR 中相邻 chunk 的 lossless high-fidelity pathway，因此它可能同时带来局部对齐收益和动态污染风险。

第一版只做 read gating：

$$
B_k^{swa}=\beta_{swa}\log(P_{prev,k}^{swa}+\epsilon),
$$

其中：

$$
P_{prev,k}^{swa}=\max(P_{ref,k},1-D_{prev,k}).
$$

$P_{ref,k}=1$ 的 token 永远不被压制，包括：

```text
special / register / role tokens
overlap frames
reset-block first window / first frame
high cumulative attention stable tokens
reliable structure semantic tokens, if available
```

初始强度：

$$
\beta_{swa}\in\{0.25,0.5,1.0\}.
$$

实验命名：

```text
R-SWA-025
R-SWA-050
R-SWA-100
```

设置：TTT update unity，其余 read-path identity。

判断标准：

- 如果 SWA-only 能进入 $<41.0$ m，说明 previous local memory 是重要瓶颈。
- 如果 SWA-only 不改善但不伤害，可以作为组合保护项。
- 如果 SWA-only 明显伤害 ATE，说明 SWA 主要提供有益 adjacent alignment，不应强控；后续只允许 reference protection，不允许 broad suppression。

### 6.4 TTT Apply Controller

TTT apply 是当前 chunk 读取 global compressed memory 的路径：

$$
\tilde H_i=H_i+f_{W_m}(LN(H_i)).
$$

第一版做 output gate：

$$
\tilde H_i^{ctrl}=H_i+g_i^{read}f_{W_m}(LN(H_i)),
$$

其中：

$$
g_i^{read}=1-\rho_{read}D_i,
$$

并且对 reference / special token 固定：

$$
g_i^{read}=1\quad\text{if }P_{ref,i}=1.
$$

初始强度：

$$
\rho_{read}\in\{0.1,0.2,0.3\}.
$$

实验命名：

```text
R-TTTA-010
R-TTTA-020
R-TTTA-030
```

判断标准：

- 如果 TTT apply-only 改善，说明 global memory read 对 dynamic/unstable tokens 有负面影响。
- 如果 TTT apply-only 伤害 ATE，说明 $W_m$ 对当前所有 token 的注入整体有益，不要在 apply path 做 broad gate。

### 6.5 Chunk-wise Bidirectional Attention Controller

Chunk-wise bidirectional attention 是当前 chunk 内强几何推理路径。它可能是最有效但也最危险的控制对象。

控制公式和 frame attention 类似：

$$
B_{qk}^{chunk}
=
\beta_c\log\left(1-(1-D_q)D_k+\epsilon\right).
$$

第一版只作用在 early/middle chunk-wise attention layers，不做 full-depth，也不做 hard mask。强度：

$$
\beta_c\in\{0.1,0.25,0.5\}.
$$

实验命名：

```text
R-CH-010
R-CH-025
R-CH-050
```

判断标准：

- 如果 $ATE<41.0$ m，说明 current chunk intra-window reasoning 确实需要 motion-aware correction。
- 如果 $ATE<40.0$ m，chunk-wise attention controller 是非常重要的主线。
- 如果 rotation 改善但 ATE 变差，说明它又落入旧 TTT-only trade-off，需要更强 reference protection 或更弱 layer selection。

### 6.6 Phase C 决策

Phase C 的结果会决定后面是否继续大改 read path。

如果所有 read-path-only 实验都不能超过 $41.0$ m，那么 HMC 的 read-path 控制没有明显信号，下一步应优先改 cue，而不是继续组合。

如果某条 read path 进入 $<41.0$ m，则进入 Phase D 组合实验。

如果任一 read path 进入 $<40.0$ m，则它成为 main read controller，优先与 TTT branch0 和 sparse write 组合。

---

## 7. Phase D：控制信号来源实验

### 7.1 中心假设

本阶段比较不同 cue 的价值：

$$
\text{explicit geometry residual cue}
\quad vs \quad
\text{internal transformer motion cue}
\quad vs \quad
\text{TTT update-needed cue}.
$$

之前显式几何 cue 的 `C_stat/C_anchor/C_unc/C_occ` 效果有限；`C_dyn` 在 branch0 上有弱信号。相关工作提示，更有效的动态信号可能来自模型内部 attention / QK / Gram statistics，而更合理的写入信号可能不是“静态性”，而是“memory 尚未解释好但当前 observation 可靠”。

### 7.2 Internal motion cue 候选

候选一是当前 `C_dyn`：

$$
D_i=C_{dyn,i}.
$$

候选二是 MUT3R-style attention motion：对 selected self-attention maps 做 head/key/layer 聚合，得到 query-level motionness：

$$
\bar A_i=\frac{1}{LHN_k}\sum_{l,h,k} A_{l,h,i,k},
$$

$$
D_i=\sigma(\operatorname{Norm}(\bar A_i)).
$$

候选三是 VGGT4D-style Gram-lite motion。若 LoGeR 能导出 selected $Q,K$ patch vectors，可以计算：

$$
A^{QQ}_{l,t,s}=\frac{Q_{l,t}Q_{l,s}^{\top}}{\sqrt c},
$$

$$
A^{KK}_{l,t,s}=\frac{K_{l,t}K_{l,s}^{\top}}{\sqrt c}.
$$

然后用 shallow/middle/deep 的近似统计：

$$
w_{shallow}=(1-S^{KK}_{shallow})\odot V^{QK}_{shallow},
$$

$$
w_{middle}=1-S^{QQ}_{middle},
$$

$$
w_{deep}=(1-V^{QQ}_{deep})\odot S^{QQ}_{deep},
$$

$$
D_i=\operatorname{Norm}(w_{shallow}\odot w_{middle}\odot w_{deep}).
$$

如果 LoGeR 当前只导出 `frame_attn_key_cosine_*` 和 `dyn4d_patch`，先用这些作为 Gram-lite proxy，不要一开始就重构所有 attention internals。

### 7.3 TTT residual / update-needed cue

候选四是 TTT residual：

$$
e_i=\frac{\|f_{W_m}(k_i)-v_i\|_2}{\|v_i\|_2+\epsilon}.
$$

候选五是可靠 update-needed score：

$$
U_i=\operatorname{rank}(e_i)\cdot (1-D_i)\cdot(1-C_{unc,i})\cdot(1-C_{occ,i}).
$$

候选六是 alignment confidence，借鉴 TTT3R：

$$
\beta_i^{align}=\sigma\left(\sum_m Q_{memory,i}K_{obs,m}^{\top}\right).
$$

在 LoGeR 里可以先用 TTT cache 中 $q,k$ 的聚合相似度或 SWA/current attention alignment 近似。

### 7.4 实验矩阵

这一阶段先不要组合太多控制路径。选择 Phase C 中最安全的 read-path controller 或直接用 TTT branch0-only 作为测试床。

| 实验 | Cue | 控制位置 | 目的 |
|---|---|---|---|
| IM-00 | $C_{dyn}$ | TTT branch0 MP-01 | 复现 BL-01 reference |
| IM-01 | MUT3R attention motion | TTT branch0 MP-01 | 看 attention motion 是否优于 $C_{dyn}$ |
| IM-02 | VGGT4D Gram-lite motion | TTT branch0 MP-01 | 看 Gram statistics 是否更稳定 |
| IM-03 | soft-or($C_{dyn}$, Gram-lite) | TTT branch0 MP-01 | 看显式/内部 cue 是否互补 |
| TR-01 | $e_i$ | TTT branch0 MP-01 | 测试 memory residual 是否有用 |
| TR-02 | $e_i(1-D_i)$ | TTT branch0 MP-01 | 防止 residual 写入动态污染 |
| TR-03 | $U_i=e_i(1-D_i)(1-C_{unc})(1-C_{occ})$ | TTT branch0 MP-01 | 测试可靠 update-needed score |
| TR-04 | alignment confidence | TTT branch0 MP-01 | 测试 TTT3R-style confidence |

判断标准：

- 若某 cue 在 TTT branch0-only 下不能超过 $41.3665$ m，则它暂时不能替代 $C_{dyn}$。
- 若某 cue 进入 $<41.0$ m，说明 cue 有明显进步。
- 若某 cue 与 Phase C 的 read-path controller 组合后进入 $<40.0$ m，进入 Phase E。
- 若所有 internal cue 都不如 $C_{dyn}$，说明 LoGeR 已导出的 internal motion 还不够可靠，下一步要么加更强 trace extraction，要么引入外部 flow/semantic。

---

## 8. Phase E：Hybrid Memory Controller 组合实验

### 8.1 中心假设

组合实验验证：

$$
\boxed{
\text{read-path correction + branch0 write control 可以突破 TTT-only 上限。}
}
$$

TTT-only branch0 最好是 $41.3665$ m。如果 read-path-only 有信号，但组合后仍然只在 41 m 附近，说明这些控制路径不是互补的，或者 cue 被重复使用造成过度 suppression。

### 8.2 组合规则

组合时不要一下子全开。每次只增加一个子控制器，并保持强度 mild。默认保护：

```text
special/register/role tokens: protected
overlap frames: protected
reset-boundary tokens: protected
reliable structure semantic tokens: protected if Stage C is enabled
```

组合实验：

| 实验 | Read-path control | TTT update control | 目的 |
|---|---|---|---|
| MC-01 | best FrameAttn | branch0 $C_{dyn}$ MP-01 | 验证 frame read + TTT write 是否互补 |
| MC-02 | best SWA | branch0 $C_{dyn}$ MP-01 | 验证 local memory read + global memory write 是否互补 |
| MC-03 | best ChunkAttn | branch0 $C_{dyn}$ MP-01 | 验证 intra-chunk reasoning + write 是否互补 |
| MC-04 | best TTT Apply | branch0 $C_{dyn}$ MP-01 | 验证 TTT read + TTT write 是否互补 |
| MC-05 | best two read paths | branch0 best cue MP-01 | 小组合，不全开 |
| MC-06 | best read path | branch0 update-needed score | 替换 $C_{dyn}$ 为 $U_i$ |

判断标准：

$$
ATE<40.0\text{ m}
$$

才算 HMC 组合有实质效果。

$$
ATE<38.0\text{ m}
$$

才进入 main candidate。

同时要求 RPE rotation 不明显恶化，且 non-overlap segment diagnostics 中坏段不能被放大。若 full ATE 变好但 local segment 只有少数段极端改善、其他段明显恶化，需要继续做 reliability gate，而不是直接作为主模型。

---

## 9. Phase F：Sparse write 与 exact preserve

### 9.1 中心假设

旧 dense reweighting 的问题是：即使 eta-mean-preserving，它仍然扰动所有 token。MeMix 的启发是：连续 gate 永远非零，会持续改变 state；sparse routing 可以让未选 token exactly preserve，从而减少长期干扰。

对应到 LoGeR TTT branch0，我们验证：

$$
\text{sparse branch0 update} > \text{dense branch0 MP}
$$

尤其是在 full KITTI 01 的 global trajectory continuity 上。

### 9.2 路由公式

给定写入 score $s_i$，选出需要写入的 token：

$$
M_i=\mathbf 1[s_i\in\operatorname{TopK}(s)].
$$

若使用 alignment score $r_i$，则采用 least-aligned bottom-k：

$$
M_i=\mathbf 1[r_i\in\operatorname{BottomK}(r)].
$$

branch0 的 effective prior 为：

$$
p_i^{(0)}=
\frac{M_i}{\frac{\sum_j \eta_j^{(0)}M_j}{\sum_j\eta_j^{(0)}+\epsilon}+\epsilon}.
$$

对 reference tokens：

$$
M_i=1\quad\text{if }P_{ref,i}=1.
$$

branch1 和 branch2 默认 unity。

### 9.3 实验矩阵

| 实验 | Score | Write ratio | Read-path setting | 目的 |
|---|---|---:|---|---|
| SR-01 | $-D_i$ | 95% | none | 极轻 sparse，验证 exact preserve 是否安全 |
| SR-02 | $-D_i$ | 85% | none | 中等 sparse |
| SR-03 | $-D_i$ | 70% | none | 强 sparse，测试上限和风险 |
| SR-04 | $U_i$ | 95% | best read path | update-needed sparse |
| SR-05 | $U_i$ | 85% | best read path | update-needed sparse 中等强度 |
| SR-06 | alignment bottom-k | 95% | best read path | TTT3R/MeMix-style route |

判断标准：

- 如果 SR-01 比 dense BL-01 好，说明 exact preserve 思想有效。
- 如果 SR-01 到 SR-03 越 sparse 越差，说明 LoGeR TTT 仍需要密集写入，后续只保留 mild sparse 或 dense MP。
- 如果 update-needed sparse 明显优于 $-D_i$ sparse，说明“哪里需要写”比“哪里动态”更适合 LoGeR TTT。
- 只有 $ATE<40.0$ m 的 sparse 设置才进入组合候选；$ATE<38.0$ m 才作为主模型 sparse policy。

---

## 10. Phase G：LoGeR* 与 reset / alignment 交互实验

### 10.1 为什么要测 LoGeR*

LoGeR* 在 KITTI 01 上 ATE 比 LoGeR 差，但 rotation 更好。这说明 feedforward SE(3) alignment 和 HMC 控制可能存在复杂交互：HMC 可能改善 LoGeR 的 global memory，也可能和 LoGeR* 的 overlap alignment 目标冲突。

本阶段不作为第一优先级，但一旦 LoGeR 上 HMC 进入 $<40.0$ m，就必须在 LoGeR* 上测试同样 controller 的 no-control parity 和 controlled result。

### 10.2 实验

| 实验 | Checkpoint | Controller | 目的 |
|---|---|---|---|
| STAR-G0 | LoGeR* | no-control HMC | 复现 $47.9793$ m |
| STAR-I0 | LoGeR* | identity hooks | 确认 hook 不破坏 SE(3) alignment |
| STAR-C1 | LoGeR* | best LoGeR controller | 看 controller 是否泛化到 LoGeR* |
| STAR-C2 | LoGeR* | read-path only | 看 LoGeR* 是否更适合 read-path control |
| STAR-C3 | LoGeR* | TTT branch0 sparse | 看 reset/alignment 下 sparse write 是否更安全 |

判断标准：

- STAR-G0 / STAR-I0 必须先通过正确性 gate。
- 若 LoGeR* controlled 不能明显改善，但 LoGeR controlled 改善，说明 controller 主要改善 LoGeR memory，而不是 alignment 后处理。
- 若 LoGeR* controlled 改善 rotation 但 ATE 仍差，说明 KITTI 01 上 SE(3) alignment 本身不是主目标，主模型仍以 LoGeR 为准。

---

## 11. Phase H：segment diagnostics 与 reliability gate

### 11.1 为什么还要做 segment diagnostics

之前已经出现过一个重要现象：某些配置在 local Sim(3) segment 上赢很多，但 full ATE 反而更差。典型例子是 occlusion score。它在很多 non-overlap segment 上局部更好，但 full ATE 伤害 global trajectory。说明 local segment ATE 不能代替 full KITTI 01 objective。

新 HMC 也必须同时看 full ATE 和 segment diagnostics，避免把局部修复误认为全局改善。

### 11.2 诊断协议

对每个 full run 生成：

```text
non-overlap 50-frame segments
non-overlap 100-frame segments
non-overlap 200-frame segments
overlap 200 stride 100 segments
```

记录：

$$
\Delta ATE_s=ATE_s(candidate)-ATE_s(unity).
$$

同时记录 segment 前后若干 chunk 的控制统计：

```text
mean D_tok
mean U_tok
mean P_swa_read
mean frame/chunk attention bias magnitude
mean TTT branch0 prior
sparse selected ratio
reference protection ratio
SWA high-dynamic read mass
chunk-attn static-to-dynamic mass
```

### 11.3 reliability gate

如果某个 controller 在 `[300,500)` 帮助明显，但在 `[0,200)` 或 `[400,600)` 伤害明显，不能继续调全局强度，而要加 chunk-level reliability gate：

$$
g_m\in[0,1].
$$

最终控制变为：

$$
B_{qk}^{ctrl}=g_m B_{qk},
$$

$$
p_i^{TTT}=1+g_m\alpha\hat s_i.
$$

初始 reliability 可以定义为：

$$
g_m=
\operatorname{clip}\left(
\frac{Q_{90}(s)-Q_{10}(s)-\tau_c}{\sigma_c},0,1
\right)
\cdot
\operatorname{clip}\left(
\frac{\tau_u-\operatorname{mean}(C_{unc})}{\sigma_u},0,1
\right)
\cdot
\operatorname{clip}\left(
\frac{\tau_h-\operatorname{SWAHighDynRead}}{\sigma_h},0,1
\right).
$$

如果 manual schedule 能显著改善 full ATE，再把它转成 automatic reliability gate。若 manual schedule 都无法把 ATE 推到 $<40.0$ m，则说明 local wins 不可组合，不应继续投入 reliability gate。

---

## 12. Stage C semantic 在新 pipeline 中的位置

Stage C 目前仍然不能主导控制。它有 thing 漏检、tracking failure、stuff 边界粗糙的问题。新 pipeline 中 Stage C 的角色应当是 protection/cap，而不是主 prior。

KITTI prompt 应该调整为：

```text
THING:
car, van, truck, bus, trailer, train,
person, pedestrian, rider, cyclist,
bicycle, motorcycle

STRUCTURE:
road, lane marking, sidewalk, building, building facade,
guardrail, fence, barrier, pole, traffic sign, traffic light,
bridge

LOW_VALUE_STUFF:
sky, tree, vegetation, grass, bush
```

使用规则：

可靠结构保护：

$$
P_{ref,i}=1
\quad\text{if semantic is reliable road/building/sidewalk/guardrail and }Q_{mask}>\tau_q.
$$

动态 thing 轻度 cap：

$$
D_i\leftarrow \max(D_i,D_i^{sem})
\quad\text{only if semantic movable and internal }D_i\text{ is already high.}
$$

不要因为 label 是 `car` 就直接强压，因为 KITTI 中 parked car 可能是有用 landmark。语义必须和 internal motion cue 一致时才参与 suppression。

语义实验只在 HMC 进入 $<40.0$ m 后进行。否则语义噪声会干扰我们判断 HMC read/write 控制本身是否有效。

---

## 13. 推荐执行顺序

这不是普通 sweep，而是逐级 gate。推荐按下面顺序执行。

第一步，完成新 pipeline 的 correctness gate：A0、A1、A2、A3、A4、A5。任何一步失败，都不要跑模型实验。尤其 LoGeR/LoGeR* reproduction 必须在新 runner 中完成。

第二步，做 Phase B 仪表盘。先看代表片段的 `D_internal`、`U_ttt`、SWA read、chunk-attn static-to-dynamic mass 是否合理。这个阶段不追 ATE。

第三步，做 read-path-only 实验。优先顺序是：Frame Attention Controller、SWA Controller、Chunk-wise Attention Controller、TTT Apply Controller。原因是我们现在最需要验证“TTT-only delayed write 是否结构性受限”。

第四步，做 cue source 对比。用当前最安全的控制位置测试 $C_{dyn}$、MUT3R-style attention motion、VGGT4D Gram-lite motion、TTT residual、update-needed score。只有超过 $41.3665$ m 的 cue 才继续组合。

第五步，组合最佳 read path 和 TTT branch0 write。目标不是 $41.0$ m，而是先冲 $<40.0$ m。如果组合不能低于 $40.0$ m，就不进入大规模语义/LoGeR*扩展。

第六步，引入 sparse write 和 exact preserve。这里的目标是减少 dense reweighting 对 global memory 的长期扰动，尤其关注 full ATE 而不是 local segment ATE。

第七步，在有效设置上做 LoGeR* 和 Stage C semantic protection。LoGeR* 是验证泛化和 reset/alignment 交互，Stage C 是保护结构和限制明显动态 thing，不是主控制信号。

---

## 14. 预期结果与如何解释失败

如果新 HMC no-control 无法复现 LoGeR / LoGeR*，说明新 runner、merge、reset 或 memory commit 逻辑有 bug。此时任何 controller 结果都不能解释。

如果 no-control 正确，但 identity hooks 失败，说明 hook 插入本身改变了模型分布。必须修 hook 的 no-op 行为，不能用“指标略好”作为成功，因为这可能是非预期扰动。

如果 HMC TTT-only 无法复现 $41.3665$ m，说明 TTT subcontroller 和旧 TTT write path 不一致。此时不能继续比较 read-path 组合。

如果 read-path-only 全部无效，但 TTT-only 仍能到 $41.36$ m，说明当前可用信号主要还是 delayed TTT branch0 write；HMC 大改的收益需要更强 cue，而不是继续改 hook。

如果 read-path-only 能到 $<41.0$ m，但组合不能到 $<40.0$ m，说明 read-path 和 TTT write 不是简单互补，可能重复压制同一类 dynamic token。此时要降低强度或加 reliability gate。

如果某个组合达到 $<40.0$ m，但达不到 $<38.0$ m，它是 weak candidate。应做 segment diagnostics 和 cue 可视化，而不是马上写成主模型。

如果某个组合达到 $<38.0$ m，就进入主模型候选，必须做完整消融：控制路径消融、cue source 消融、branch/layer消融、sparse/dense消融、reference protection消融、LoGeR*复现、Stage C semantic protection、runtime/memory分析。

如果达到 $<35.0$ m，就可以围绕它设计最终方法叙事，因为这已经是对 LoGeR 的明显改善。

---

## 15. 最终方法假设的当前版本

本方案最终要验证的完整假设是：

$$
\boxed{
\begin{aligned}
&\text{LoGeR 的 long-context 误差不是单纯 TTT 写入污染，}\\
&\text{而是 hybrid memory 中 read path 与 write path 共同传播不稳定 token。}
\end{aligned}
}
$$

因此，最小有效 HMC 不应该只输出一个 `A_tok`，而应输出：

```text
MemoryControlPrior = {
  D_tok,                         # dynamic / motionness
  U_tok,                         # update-needed score
  P_ref,                         # reference protection
  B_frame_attn,                  # frame attention soft bias
  P_swa_read_prev,               # previous chunk local memory read gate
  G_ttt_apply,                   # TTT memory read gate
  P_ttt_update_branch0,          # branch0 write prior / sparse route
  B_chunk_biattn,                # chunk-wise attention soft bias
  semantic_protection,           # optional protection/cap
  reliability_gate
}
```

如果这个设计成立，那么最先出现的有效结果应该不是“TTT branch0 又好一点”，而是：

$$
\text{read-path-only 有收益，且 read-path + branch0 write 能突破 }40\text{ m}.
$$

如果最终仍然停留在 $41$ m 附近，那就说明 training-free internal HMC 对 KITTI 01 的提升上限有限，需要引入更强外部 cue，例如 optical flow consistency、better semantic/motion mask、或者轻量学习式 reliability gate。

---

## 16. 最小实验清单

为了便于执行，下面给出最小必须跑的实验清单。这里的命名只是建议，具体 CLI 可以按实现调整。

```text
Correctness:
  HMC_A0_pass1_no_commit_64
  HMC_A1_pass1_pass2_no_control_64
  HMC_A2_identity_hooks_full_loger
  HMC_A3_no_control_full_loger
  HMC_A4_no_control_full_loger_star
  HMC_A5_ttt_branch0_bl01_repro_full

Trace dashboard:
  HMC_B_dashboard_segments_0_200
  HMC_B_dashboard_segments_300_500
  HMC_B_dashboard_segments_400_600
  HMC_B_dashboard_segments_800_1000

Read-path-only:
  HMC_R_FR_025_full
  HMC_R_FR_050_full
  HMC_R_FR_100_full
  HMC_R_SWA_025_full
  HMC_R_SWA_050_full
  HMC_R_SWA_100_full
  HMC_R_TTTA_010_full
  HMC_R_TTTA_020_full
  HMC_R_TTTA_030_full
  HMC_R_CH_010_full
  HMC_R_CH_025_full
  HMC_R_CH_050_full

Cue source:
  HMC_IM_Cdyn_branch0_full
  HMC_IM_MUT3R_branch0_full
  HMC_IM_GramLite_branch0_full
  HMC_IM_Cdyn_Gram_softor_branch0_full
  HMC_TR_residual_branch0_full
  HMC_TR_residual_motion_reliable_branch0_full
  HMC_TR_update_needed_branch0_full
  HMC_TR_alignment_conf_branch0_full

Combined:
  HMC_MC_bestFrame_branch0_full
  HMC_MC_bestSWA_branch0_full
  HMC_MC_bestChunk_branch0_full
  HMC_MC_bestRead_bestCue_branch0_full
  HMC_MC_bestRead_updateNeeded_branch0_full

Sparse:
  HMC_SR_dyn_95_full
  HMC_SR_dyn_85_full
  HMC_SR_updateNeeded_95_full
  HMC_SR_updateNeeded_85_full
  HMC_SR_alignment_bottomk_95_full
```

建议每个 full run 都同步生成：

```text
01.txt
kitti_benchmark.log
segment_ate_nonoverlap_50.csv
segment_ate_nonoverlap_100.csv
segment_delta_summary.json
hmc_control_summary.jsonl
hmc_trace_sample.mp4 or png grid
```

---

## 17. 结语

这轮新方案的重点不是“多跑几个控制组合”，而是先让新的 HMC pipeline 成为一个严格可验证的 LoGeR 等价超集。它必须在 no-control 和 identity-hooks 情况下复现 LoGeR 与 LoGeR*，再证明 TTT-only 子路径能复现旧 BL-01。只有这些 gate 都过了，read-path controller 和 sparse write 的结果才有解释价值。

如果 HMC 能在 read-path-only 或 read-path + branch0 write 下进入 $<40.0$ m，就说明我们终于突破了旧 TTT-only delayed write 的瓶颈。如果进一步进入 $<38.0$ m，它就可以成为新的主模型候选。如果仍然停在 $41$ m 附近，则应及时承认：training-free internal cue 不足以支撑 KITTI 01 的显著提升，下一步需要引入更强的外部 motion / semantic / flow signal 或学习式 reliability gate。

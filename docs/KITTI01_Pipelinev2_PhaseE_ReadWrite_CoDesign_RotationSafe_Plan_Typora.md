# KITTI 01 / HMC Phase E 实验计划

# Read/Write Co-Design with Rotation-Safe Protection

> 版本：Phase E v1  
> 目标序列：KITTI Odometry 01  
> 目标模型：LoGeR / HMC Pipeline v2  
> 当前最佳参考：D5-07，`probe_ttt_write`，`old_dyn` frame-attention pair early beta=1.25，branch0 BL01 write，ATE = 39.4903 m  
> 当前平衡参考：D5-04，`probe_ttt_write`，`old_dyn` frame-attention pair early beta=1.00，branch0 BL01 write，ATE = 39.5127 m  

---

## 0. 这份计划要解决什么问题

Phase C 和 Phase D 已经把项目从“TTT-only write control”推进到了“read correction + safe commit”。这是一条真正有效的方向：当前 chunk 的输出可以用 controlled read-path 修正，但下一 chunk 的 memory 不能直接继承 controlled forward 产生的 TTT state。Phase D v5 进一步证明，如果输出使用 Pass-2 controlled geometry，而下一 chunk 的 TTT memory 使用 Pass-1 probe cache 上显式执行的 branch0 TTT write，就可以把 full KITTI 01 ATE 从 read-only CM02 的 39.7820 m 推到 D5-07 的 39.4903 m。

但是当前结果还不是最终主模型。D5-07 的 ATE 是目前最好，但它的 rotation RMSE 和 final aligned error 都比 BL01 / read-only reference 更差。也就是说，当前方法改善了全序列 Sim(3)-aligned trajectory shape 和中段局部 ATE，却损伤了 orientation / endpoint / reference stability。Phase E 的核心任务不是继续盲目强化 `old_dyn` 或继续扫 beta，而是重新设计 read cue、write cue 与 protection cue 的分工：read cue 用来修当前输出，write cue 用来安全更新未来 memory，reference/protection cue 用来避免 orientation-critical token 被错误抑制或错误写入。

Phase E 的核心问题可以写成：

$$
\text{Can we keep the current ATE gain from controlled read, while making the update rotation-safe and endpoint-safe?}
$$

更具体地说，本阶段要回答四个问题：

1. `old_dyn` 是否可以通过 reference / overlap / structure protection 变得更 rotation-safe？
2. TTT write 是否应该从 `-rank(C_dyn)` 迁移到 memory update-needed signal，例如 TTT residual、alignment confidence 或 sparse routing？
3. read cue 和 write cue 是否必须分开设计，而不是使用同一张 dynamic map？
4. 在不触碰或少触碰 SWA / chunk attention 的情况下，是否能把当前 39.49 m 推到 39.0 m 以下，并降低 rotation / endpoint 损伤？

这份计划会先总结之前实验给出的可靠 insight，再提出 Phase E 的实验假设，最后围绕每条假设设计具体实验、晋级规则、失败解释和停止条件。计划会比较庞大，因为 Phase E 不再是单一参数 sweep，而是 read/write/protection 三类信号的系统性 co-design。

---

## 1. 之前实验已经给出的可靠 insight

### 1.1 LoGeR / HMC 基础正确性已经不是主要问题

最早的 native reproduction 已经验证当前输入、checkpoint、chunk/window/reset 设置、merge 和评估链路是可信的。LoGeR 在 KITTI 01 上复现到 41.7502 m，接近论文表格里的 41.64 m；LoGeR* 复现到 47.9793 m，接近论文表格里的 47.91 m。随后 unity replay 也复现到 41.6193 m，和 native 结果很接近。后续 HMC no-control / identity-hook pipeline 又复现了 LoGeR 和 LoGeR*，并且 no-control pass1/pass2 pose diff 为 0，hook bias 为 0。这说明当前 HMC pipeline 的基础工程链路已经足够可信。

因此，Phase E 不再把主要精力放在“是不是旧 pipeline 复现问题”上，但每一批新实验前仍然必须保留 HMC correctness gate。只要新增 hook、commit mode 或写入策略，就必须重新证明：无控制时 LoGeR / LoGeR* 结果不变，identity hook 不改变输出，Pass 1 和 Pass 2 在 no-control 情况下完全一致。

### 1.2 旧的纯 TTT write control 不够强，但 branch0 确实有信号

早期 GSL-WC 的主要假设是：动态/不稳定 token 少写入 TTT memory，就能减少 memory contamination。实验最终证明这条线太弱。最好的 suppressive geometry-only 结果只有 41.5765 m，比 unity replay 好 0.0428 m，收益很小。

后来做 eta-mean-preserving reweighting 后，整体写入量保持了，但 ATE 反而变差；这说明问题不是单纯“写少了”，而是 token ranking / branch coupling 不够好。score decomposition 找到第一个较明确的信号：`-rank(C_dyn)` 用 MP-01 作为 TTT 写入 score 可以得到 41.4742 m，比 unity 和 previous best 都好一点。进一步 branch selective 发现：这个动态 score 只在 branch0 上比较有效，branch1 在 full sequence 上不安全，branch2 只有弱收益。

因此 Phase E 必须继承这条结论：TTT write 不应该 all-branch；当前默认安全写入路径是 branch0 only。branch1 / branch2 只有在新的证据明确通过 full-sequence gate 后才允许引入。

### 1.3 read-path control 真正有效，但它产生的 memory side effect 不能直接 commit

Phase C 的关键突破是发现 frame-attention read-path control 有真实 full-sequence 信号。RFR-100 的具体配置是：

```text
cue = old C_dyn
path = frame attention
bias = pair
layers = early
beta = 1.0
mode = read-path-only
```

它在 controlled commit 下可以达到 41.0733 m，已经比 LoGeR native 和 BL01 branch0 write 更好。这个结果说明：旧 Stage-B `C_dyn` 不只是 TTT write ranking，它也可以作为当前 chunk forward 的 read-path cue，用来减弱 static query 对 dynamic key 的 attention。

但是 Phase C v5 的 commit isolation 给出更关键的结果。同样使用 RFR-100 controlled output，如果最终 commit Pass-2 controlled TTT state，结果是 41.0733 m；如果当前输出仍然使用 Pass-2 controlled geometry，但下一 chunk 的 TTT memory 改成 Pass-1 native/probe state，结果变成 39.7820 m。这个差异证明：read-path correction 是有效的，但 controlled forward 内部产生的 TTT update 不适合直接提交给未来 memory。

这一点是 Phase E 的基石：

$$
Y_m^{final}=Y_m^{controlled}, \qquad W_{m+1}^{final}\neq W_{m+1}^{controlled}.
$$

更安全的协议是：

$$
Y_m^{final}=Y_m^{controlled}, \qquad W_{m+1}^{final}=\operatorname{SafeWrite}(Cache_m^{probe}).
$$

也就是说，read correction 和 memory write 必须解耦。

### 1.4 Phase D v5 证明 read correction 与 branch0 write 可以互补

Phase D v5 的 `probe_ttt_write` 进一步证明：不提交 Pass-2 controlled memory，并不等于完全放弃 TTT write。正确方式是使用 Pass-1 native/probe cache，再在这个 cache 上显式执行 branch0 write controller。

核心对比是：

| 方法 | 当前输出 | 下一 chunk TTT memory | ATE RMSE |
|---|---|---|---:|
| BL01 | native output | branch0 dynamic write | 41.3665 |
| D5-03 / CM02 | Pass-2 controlled output | Pass-1 probe/native state | 39.7820 |
| D5-04 | Pass-2 controlled output | Pass-1 probe cache + branch0 write, beta=1.0 | 39.5127 |
| D5-07 | Pass-2 controlled output | Pass-1 probe cache + branch0 write, beta=1.25 | 39.4903 |

这个结果说明 read-path correction 和 branch0 TTT write 是互补的。read-only 已经能到 39.7820 m；在 commit-safe 协议下叠加 branch0 BL01 write，可以进一步改善到 39.49 m 左右。

因此 Phase E 不应该回退到 read-only，也不应该直接提交 controlled state。默认实验基准应为：

```text
output = Pass-2 controlled geometry
commit = Pass-1 probe cache + explicit branch0 write
```

### 1.5 当前最大副作用是 rotation / endpoint / reference stability

D5-07 的 ATE 最好，但 rotation 与 final endpoint 变差。D5-b125 的 ATE 是 39.4903 m，但 rotation RMSE 是 9.8299 deg，final error 是 6.311 m。相比之下，BL01 的 final error 只有 3.590 m，rotation RMSE 是 8.9490 deg。D5-04 的 ATE 只比 D5-07 差 0.0224 m，但 rotation 更好一些，因此它是更平衡的开发 reference。

Trajectory diagnostics 还显示，D5 的 50/100/200-frame mean ATE 都改善，但 final aligned error 变大。这说明组合方法改善了大部分中段局部形状和全序列 aligned shape，但可能牺牲了 orientation consistency、endpoint drift 或某些 reference token 的稳定性。

Phase E 的首要目标不是再做 beta sweep，而是修这个 trade-off：

$$
\text{Keep } ATE\approx 39.5\text{ m or better, while reducing rotation and final error damage.}
$$

### 1.6 新 read cue 不能直接替换 old_dyn，至少不能重复 v3/v4 的错误

Phase C v2/v3/v4 已经给了很强的负证据：一些 cue 在短序列、256-frame 或 stateful slice 上表现很好，但 full KITTI 01 失败。`flow_sem_veto` proxy 在 64-frame 和 256-frame 都很强，stateful slice 也有 6/7 slices improved，但 full KITTI 01 是 41.4333 m，比 RFR-100 更差。v4 的 full-run cue audit 发现 flow proxy calibration 即使修了 mass，也带来高 fragmentation；安全化之后短程收益基本消失。

因此 Phase E 不应该继续直接用 `flow_proxy`、`flow_sem_veto_proxy`、`gram4d` 替换 `old_dyn`。更合理的方式是让它们作为 modulator 或 veto：

- 当它们和 `old_dyn` 一致时，增强 dynamic suppression；
- 当它们显示某些区域是可靠静态结构时，对 `old_dyn` 做 static rescue；
- 当它们 fragment / sparse / coverage 不稳定时，不进入 full candidate。

换句话说，Phase E 的 read cue 策略是：

$$
D_{read}=\text{protected/modulated old\_dyn},
$$

而不是：

$$
D_{read}=\text{brand-new cue replacing old\_dyn}.
$$

### 1.7 相关工作给出的启发

LoGeR 原论文强调 hybrid memory 的双通道作用：SWA 是相邻 chunk 的 lossless local memory，TTT 是 long-range compressed global context；单个 block 的顺序是 per-frame attention、SWA、TTT apply/update、chunk-wise bidirectional attention。这个结构说明我们现在做的 read/write 分离是合理的：read path 修当前输出，TTT write path 决定未来 global memory。

VGGT4D 和 MUT3R 的共同启发是：动态 cue 最适合在 early-stage attention 里做 soft gating，而不是全层 hard mask。VGGT4D 还指出普通 attention 里动态和语义/纹理噪声混在一起，直接用 attention 替换 motion cue 不稳定；这与我们 Phase C v3/v4 看到的 full-run false positive 一致。

TTSA3R 和 TTT3R 的共同启发是：memory update 应该根据 observation-state alignment、feature divergence、temporal evolution 或 learning-rate confidence 来控制，而不是简单“动态少写，静态多写”。MeMix 进一步强调 dense soft gate 的不足：即使每个 token 只写一点，也会造成长期干扰；sparse routing / exact preserve 可以减少 state interference。这正是 Phase E 要探索 update-needed write 和 sparse write 的原因。

---

## 2. Phase E 的总目标与实验原则

### 2.1 Phase E 的总目标

Phase E 的目标不是重新证明 read-path control 有效，也不是继续追求单一 ATE 最小值。它要建立一个更完整的 read/write co-design：

1. read cue 负责当前 chunk output correction；
2. write cue 负责未来 TTT memory 的 safe update；
3. protection cue 负责保护 rotation / endpoint / reference stability；
4. commit protocol 继续保持 `probe_ttt_write`，不直接提交 Pass-2 controlled TTT state。

最终目标是形成一个比 D5 更稳的主候选：

$$
ATE < 38.0\text{ m}
$$

这是主模型候选标准。短期阶段性目标是：

$$
ATE < 39.0\text{ m}
$$

并且不能以严重 rotation / endpoint 损伤为代价。

### 2.2 Phase E 的指标标准

本阶段不再只看 full ATE。每个 full candidate 必须报告：

| 指标 | 含义 | 参考 |
|---|---|---|
| Full ATE RMSE | 主指标 | D5-07 = 39.4903 m，D5-04 = 39.5127 m |
| Rotation RMSE | orientation stability | D5-04 = 9.7345 deg，D5-07 = 9.8299 deg |
| Final aligned error | endpoint / long-horizon drift | D5-04 = 6.058 m，D5-07 = 6.311 m |
| 50/100/200-frame mean ATE | local/mid-segment shape | D5 已明显优于 BL01/CM02 |
| Worst chunk RMSE | localized failure | D5 worst chunks: 7–9, 15–16 |
| RPE t / RPE r | auxiliary | 不作为唯一决策，但必须记录 |

Phase E 的晋级标准分成四档：

| 等级 | 条件 | 解释 |
|---|---|---|
| E-debug pass | $ATE < 39.49$ 或 $ATE \approx 39.5$ 且 rotation/final 明显改善 | 比 D5 有实质改进或更平衡 |
| E-balanced pass | $ATE \le 39.5$ 且 $Rot \le 9.5^\circ$ 且 $FinalErr \le 5.5$ | 保住 ATE，修 rotation / endpoint |
| E-strong pass | $ATE < 39.0$ 且 $Rot \le 9.5^\circ$ | Phase E 明确优于 D5 |
| Main-candidate pass | $ATE < 38.0$ 且 rotation/final 不显著恶化 | 可作为主模型候选 |

如果某个配置 ATE 最低，但 rotation 超过 10.0 deg 或 final error 超过 6.8 m，则不作为主候选，只保留为 ATE diagnostic。

### 2.3 Phase E 的实验原则

Phase E 不能再大规模无约束 sweep。它要按下面原则推进。

第一，所有模型选择都必须基于 full KITTI 01。64-frame、256-frame 和 stateful slices 只能作为 smoke / debugging，不再作为晋级充分条件。

第二，read cue 和 write cue 分开设计。一个 cue 可以同时出现在两个路径里，但不能默认同一张 dynamic map 同时是 read gating map 和 write prior map。

第三，commit protocol 固定为 `probe_ttt_write`，除非特意做 ablation。也就是说，当前输出来自 Pass-2 controlled geometry，未来 TTT memory 来自 Pass-1 probe cache 上的显式 safe write。

第四，branch0 是默认可控 TTT branch。branch1 / branch2 暂时不进入主实验矩阵，除非某个 write cue 在 branch0 上已经明确有效，再做小规模 branch ablation。

第五，任何新的 flow cue 都必须是真正外部光流 residual，例如 RAFT/GMFlow。之前的 patch-match / reprojection proxy 不允许直接作为 full candidate；如果仍使用 proxy，只能用于 dashboard 或 smoke。

第六，semantic 只能作为 protection / veto，不作为主导 read or write cue。原因是当前 Video Masklet Frontend 仍有 thing 漏检、tracking failure 和 stuff 边界粗糙的问题。

---

## 3. Phase E 的核心假设

### 3.1 假设 H1：D5 的 rotation / final error 损伤来自 orientation-critical reference token 被 read bias 误伤

D5 的 read cue 是 `old_dyn`。它在 early frame attention 里降低 static query 对 dynamic key 的 attention。这个策略改善了 ATE，但可能会错误压制一些对 orientation / endpoint 有用的 token，例如 overlap frame、reset block 前几帧、high-anchor structure、远处建筑轮廓、道路/护栏边界等。这些区域可能被 `C_dyn` 误判为 dynamic 或 unstable，但它们对相机 orientation 和长程 reference 很重要。

如果 H1 成立，那么加入 reference protection 后，应该出现下面现象：

$$
ATE \approx 39.5\text{ m or better}, \qquad Rot \downarrow, \qquad FinalErr \downarrow.
$$

也就是说，ATE 可能不大幅下降，但 rotation 和 endpoint 会更稳。

### 3.2 假设 H2：read cue 应该以 old_dyn 为 backbone，其他 cue 只做 modulator / rescue

Phase C v3/v4 已经证明，`flow_sem_veto` proxy、flow calibration、gram cue 等直接替换 old_dyn 时，很容易短程好、full 失败。原因可能是它们太 sparse、fragmented、coverage 不稳定，或者没有保护 full-sequence trajectory continuity。

因此新的 read cue 不应直接替换 old_dyn，而应采用：

$$
D_{read}^{new}=\operatorname{clip}\big(D_{old}\cdot A_{agree}-R_{static},0,1\big).
$$

其中 $A_{agree}$ 是 motion agreement / reinforcement，$R_{static}$ 是 static rescue / reference protection。

如果 H2 成立，那么 modulated old_dyn 应该比 direct replacement 更稳定，至少不会重现 v3/v4 的 full-run collapse。

### 3.3 假设 H3：TTT write cue 应该是 update-needed，而不是 read dynamic cue 的简单复用

当前 D5 使用 branch0 BL01 write，即基于 `-rank(C_dyn)` 的 MP-01 写入。它能与 read correction 互补，但它并没有回答 memory 是否已经解释好了某个 token。更合理的 write cue 应该来自 Pass-1 probe cache，例如 TTT residual：

$$
e_i=\frac{\|f_{W_m}(k_i)-v_i\|_2}{\|v_i\|_2+\epsilon}.
$$

然后结合 observation reliability：

$$
S_{write,i}=\operatorname{rank}(e_i)\cdot (1-D_{read,i})\cdot (1-C_{unc,i})\cdot (1-C_{occ,i})\cdot (1-P_{ref,i}).
$$

这个 score 的含义是：只有当 memory 还没有解释好当前 observation，并且当前 observation 可靠、非动态、非 reference-sensitive 时，才应该写入未来 TTT memory。

如果 H3 成立，那么 update-needed write 应该在 D5 的基础上进一步降低 ATE，或者在保持类似 ATE 的同时改善 rotation/final error。

### 3.4 假设 H4：sparse exact-preserve 比 dense soft reweighting 更适合长期 TTT memory

MeMix 的核心启发是：dense gate 即使幅度很小，也会让每个 token 获得非零更新，长期累积后仍可能造成 memory interference。我们的 D5 虽然用 probe cache 避免了 Pass-2 controlled side effect，但 branch0 BL01 write 仍然是 dense MP-01。

Phase E 应该测试 sparse write：

$$
M_i=\mathbf{1}[S_{write,i}\in \operatorname{TopK}(S_{write})],
$$

然后只让 selected tokens 参与 branch0 update，未选 token exact preserve：

$$
p_i^{branch0}=\frac{M_i}{\frac{\sum_j\eta_jM_j}{\sum_j\eta_j+\epsilon}+\epsilon}.
$$

如果 H4 成立，sparse route 应该减少 final error 和 rotation damage，并可能进一步提升 ATE。

### 3.5 假设 H5：SWA / chunk attention 不是 Phase E 的首要控制路径，除非 dashboard 明确指向它们

LoGeR 的 SWA 与 chunk-wise bidirectional attention 当然重要。SWA 是相邻 chunk 的 lossless local memory，chunk-wise attention 是当前 chunk 内强几何推理。但现在已经有效的路径是 per-frame read + branch0 TTT write。贸然加入 SWA/chunk attention non-identity control 会增加不可解释性。

因此 Phase E 的默认策略是：先不碰 SWA/chunk dense control。只有当 worst-chunk dashboard 明确显示 failure 来自 chunk boundary local context 或 chunk-level cross-frame contamination，才启动 SWA/chunk rescue experiment。

---

## 4. 实验前工程与正确性 Gate

Phase E 引入了 reference protection、read cue modulation、TTT residual write、sparse routing 等新逻辑。这些都可能影响 HMC 的状态传递。因此每一组新代码进入 full KITTI 前必须通过 E0 correctness gate。

### 4.1 E0-A：HMC no-control LoGeR / LoGeR* 复现

必须在当前 Phase E 代码版本下，使用 HMC two-pass no-control 复现：

| Model | Expected ATE | Pass criterion |
|---|---:|---:|
| LoGeR | 41.7502 m | absolute diff $\le 0.10$ m |
| LoGeR* | 47.9793 m | absolute diff $\le 0.10$ m |

这里不能使用旧 `geometry_eval_mode` 作为正式结果，也不能绕回旧 TTT-only pipeline。必须走新 HMC pipeline，记录：

```text
probe_safe = true for all chunks
state_double_write_safe = true for all chunks
pass1/pass2 pose diff = 0 in no-control
all read hook max bias = 0 in no-control
```

如果任何一项失败，停止所有 Phase E 模型实验。

### 4.2 E0-B：D5 reference deterministic reproduction

Phase E 的主要 reference 是：

| Run | Setting | Expected ATE | Role |
|---|---|---:|---|
| D5-04R | beta=1.0, probe_ttt_write | 39.5127 m | balanced reference |
| D5-07R | beta=1.25, probe_ttt_write | 39.4903 m | best ATE reference |

Phase E 新代码必须能重复 D5-04 或 D5-07，允许误差：

$$
|ATE_{repeat}-ATE_{reference}|\le 0.05\text{ m}.
$$

如果 D5 repeat 失败，说明新代码或新 logging 改变了 baseline，不允许进行新的 model selection。

### 4.3 E0-C：new hook identity check

新增的 protection / modulation / residual / sparse write hook 必须支持 identity mode：

```text
read protection identity: P_ref = 0 or rho_ref = 0
read modulation identity: D_read_new = old_dyn
write residual identity: use BL01 write
sparse identity: selected_ratio = 1.0
```

这些 identity variants 应复现 D5 reference。允许 ATE 误差：

$$
\Delta ATE \le 0.05\text{ m}.
$$

只有 identity pass 后，才允许打开非 identity 控制。

---

## 5. Phase E 诊断 Dashboard

在跑新 full candidate 前，必须先建立 Phase E dashboard。这个 dashboard 不是为了美观，而是为了解释 D5 的 rotation/final 损伤，并决定哪些 protection 有必要进入 full runs。

### 5.1 必须重点诊断的 chunk

Phase D v5 给出的 D5-b125 worst chunks 是：

| Chunk | Frames | RMSE |
|---:|---|---:|
| 8 | `[232,264)` | 87.73 |
| 9 | `[261,293)` | 74.47 |
| 7 | `[203,235)` | 70.92 |
| 15 | `[435,467)` | 65.06 |
| 16 | `[464,496)` | 63.54 |

Phase E dashboard 必须覆盖这些 chunk，并额外覆盖若干正常 chunk 作为对照，例如 chunk 0、20、25、30。

### 5.2 每个 chunk 保存的可视化

对每个重点 chunk，保存以下图或视频帧：

```text
RGB
old_dyn / StageB C_dyn
C_stat / C_anchor / C_unc / C_occ
reference protection mask P_ref
overlap/reset/reference frame mask
read bias energy map E_B
frame-attention suppressed key mass
branch0 write prior map
TTT residual e_i
residual × reliability write score
sparse selected mask, if used
semantic protection map, if enabled
local trajectory overlay and per-frame pose error
```

### 5.3 每个 chunk 保存的数值统计

每个 chunk 记录：

$$
\operatorname{mean}(D_{read}),\quad Q_{90}(D_{read}),\quad \operatorname{mass}(D_{read}>0.5)
$$

$$
\operatorname{mean}(P_{ref}),\quad \operatorname{mass}(P_{ref}>0.5),\quad \operatorname{overlap}(D_{read},P_{ref})
$$

$$
E_B=\operatorname{Mean}(|B_{ij}|),\quad E_{B,ref}=\operatorname{Mean}(|B_{ij}|\cdot P_{ref,j})
$$

$$
M_\eta^{branch0}=\frac{\sum_i\eta_i p_i}{\sum_i\eta_i+\epsilon}
$$

$$
\operatorname{mean}(e_i),\quad Q_{90}(e_i),\quad \operatorname{Corr}(e_i,D_{read,i}),\quad \operatorname{Corr}(e_i,C_{unc,i})
$$

以及 trajectory side effect：

```text
chunk local RMSE
chunk endpoint delta
chunk rotation delta
boundary pose jump vs previous chunk
```

这些统计必须写入 `phaseE_dashboard.jsonl`，并用于后续选择 full candidates。

---

## 6. Experiment Block E1：Rotation-Safe Read Protection

### 6.1 实验目的

E1 的目标是验证 H1：D5 的 rotation / final error 损伤是否来自 read bias 误伤 reference / orientation-critical token。如果是，那么给 `old_dyn` 增加 reference protection 后，应该能在保持 ATE 的同时降低 rotation 和 final error。

E1 不改变 TTT write。所有 E1 实验都使用当前 D5 的 commit-safe branch0 write：

```text
commit = probe_ttt_write
write = branch0 BL01 dynamic MP-01
read path = frame attention
bias mode = pair
layer = early
```

唯一变化是 read cue 中引入 protection。

### 6.2 reference protection 的数学形式

定义保护图：

$$
P_{ref,i}=\max(P_{overlap,i},P_{reset,i},P_{anchor,i},P_{attn,i},P_{sem,i}).
$$

其中：

- $P_{overlap}$：chunk overlap frames 或与 overlap 对齐相关 token；
- $P_{reset}$：reset block 首个 chunk / 首几帧 token；
- $P_{anchor}$：高 $C_{anchor}$、低 $C_{dyn}$、高 confidence 的稳定结构 token；
- $P_{attn}$：probe pass 中被大量 attention 读取的 important key token；
- $P_{sem}$：road、building、guardrail、sidewalk 等可靠结构语义保护。

read bias 中使用 protected dynamic key：

$$
D_{key,i}^{prot}=D_{old,i}\cdot(1-\rho_{ref}P_{ref,i}).
$$

pair bias 变成：

$$
B_{ij}=\beta\log\left(1-(1-D_i)D_{key,j}^{prot}+\epsilon\right).
$$

第一版只保护 key，不保护 query。原因是当前 RFR pair bias 的主要作用是减少 static query 对 dynamic key 的读取；如果同时保护 query，可能削弱 read correction 本身。

### 6.3 E1 实验矩阵

E1 不应该一次性 full 跑所有组合。先跑 dashboard/smoke，再 full 推进。

#### E1 Smoke / dashboard

对 256 frames 和重点 stateful chunks 先跑：

| ID | Protection | beta | 目的 |
|---|---|---:|---|
| E1S-00 | none / D5 reference | 1.0 | reference |
| E1S-01 | overlap only | 1.0 | 看 overlap protection 是否降低 boundary jump |
| E1S-02 | high-anchor only | 1.0 | 看 high-anchor 是否被 old_dyn 误伤 |
| E1S-03 | reset/ref only | 1.0 | 看 reset/reference frame 是否影响 endpoint |
| E1S-04 | attention-importance only | 1.0 | 看 important key protection 是否有效 |
| E1S-05 | structure semantic only | 1.0 | 看 KITTI structure protection 是否有效 |
| E1S-06 | combined light | 1.0 | 综合保护轻量版 |

Smoke 不以 ATE 晋级为唯一条件，还要看 dashboard：

```text
reference-overlap bias energy should decrease
protected tokens should not be heavily suppressed
normal dynamic regions should still be suppressed
boundary pose jump should decrease in worst chunks
```

#### E1 Full candidates

从 smoke/dashboard 中选最多 4 个 full runs：

| ID | Protection | beta | Full 目的 |
|---|---|---:|---|
| E1F-00 | D5-04 reference | 1.0 | balanced baseline |
| E1F-01 | best single protection | 1.0 | 单保护是否足够 |
| E1F-02 | combined light | 1.0 | 是否修 rotation/final |
| E1F-03 | combined light | 1.25 | 是否保留 D5-07 ATE |
| E1F-04 | combined stronger | 1.0 | 若 light 不够，试更强保护 |

### 6.4 E1 判断标准

E1 成功不要求一定打破 39.49。它有两种成功形式。

第一种是 ATE 继续变好：

$$
ATE < 39.49\text{ m}.
$$

第二种是 balanced improvement：

$$
ATE \le 39.60\text{ m},\quad Rot \le 9.5^\circ,\quad FinalErr \le 5.5\text{ m}.
$$

如果 E1F 没有任何配置改善 rotation/final，说明 D5 的 rotation 损伤不主要来自 reference token 被 read bias 误伤，Phase E 应立即转向 write cue / sparse write，而不是继续 protection sweep。

---

## 7. Experiment Block E2：Read Cue Modulation Instead of Replacement

### 7.1 实验目的

E2 的目标是验证 H2：old_dyn 是目前最稳定的 read backbone，但可以通过其他信号做 reinforcement 或 static rescue。E2 不允许直接替换 old_dyn，除非新的 cue 通过 full-run cue audit。

E2 仍然使用 `probe_ttt_write` 和当前 best branch0 write。E2 的关键是改变 $D_{read}$。

### 7.2 read modulation 形式

基础形式是：

$$
D_{read}=\operatorname{clip}(D_{old}\cdot A_{agree}-R_{static},0,1).
$$

其中 motion agreement 可以是：

$$
A_{agree}=1+\lambda_g(\tilde D_{gram}-0.5)
$$

或 true-flow agreement：

$$
A_{agree}=1+\lambda_f(\tilde D_{flow}-0.5).
$$

static rescue 可以是：

$$
R_{static}=\lambda_s\cdot P_{ref}\cdot (1-C_{unc})\cdot C_{anchor}.
$$

若使用 semantic：

$$
R_{static}^{sem}=\lambda_{sem}\cdot \mathbf{1}[label\in\{road,building,sidewalk,guardrail\}]\cdot Q_{mask}.
$$

注意，semantic 只做 rescue，不直接把 car/person 判为 dynamic。对 car/person 只有在 `old_dyn` 高且 mask quality 高时才允许增强 dynamic suppression。

### 7.3 E2 候选 cue

E2 可考虑以下 modulator：

| Cue | 用法 | 备注 |
|---|---|---|
| Gram-lite / Gram4D | agreement / reinforcement | 不再直接替换 old_dyn |
| true RAFT / GMFlow residual | agreement or static rescue | 必须是真 flow，不允许 patch-match proxy |
| Semantic structure mask | static rescue | 只保护 road/building/guardrail/sidewalk |
| High-anchor / low-unc | static rescue | 不依赖 Stage C |
| old_dyn confidence collision | rescue | 若 old_dyn 与 high-confidence static 强冲突则降低 |

### 7.4 E2 实验矩阵

E2 先只 full 跑少量候选。若 true flow 没实现，则跳过 flow full，只做 Gram/anchor/semantic rescue。

| ID | Read cue | beta | Write | 目的 |
|---|---|---:|---|---|
| E2F-00 | D5 reference old_dyn | 1.0 / 1.25 | branch0 old_dyn MP01 | baseline |
| E2F-01 | old_dyn - high-anchor static rescue | 1.0 | branch0 old_dyn MP01 | 几何保护 |
| E2F-02 | old_dyn × Gram agreement | 1.0 | branch0 old_dyn MP01 | internal motion reinforcement |
| E2F-03 | old_dyn + semantic structure rescue | 1.0 | branch0 old_dyn MP01 | KITTI structure protection |
| E2F-04 | old_dyn × true-flow agreement | 1.0 | branch0 old_dyn MP01 | 只有 true flow 完成后运行 |
| E2F-05 | old_dyn - true-flow static rescue | 1.0 | branch0 old_dyn MP01 | 只有 true flow 完成后运行 |

### 7.5 E2 判断标准

E2 的重点是证明新 cue 作为 modulator 有价值，而不是证明它能直接替代 old_dyn。通过条件：

$$
ATE < 39.49\text{ m}
$$

或：

$$
ATE \le 39.60\text{ m},\quad Rot \le 9.5^\circ,\quad FinalErr \le 5.5\text{ m}.
$$

若所有 modulator 都不如 E1 best / D5 reference，则保留 old_dyn，进入 E3 write cue 设计。

---

## 8. Experiment Block E3：Update-Needed TTT Write

### 8.1 实验目的

E3 的目标是验证 H3：未来 memory write 不应该直接复用 read dynamic cue，而应该使用 Pass-1 probe cache 上的 update-needed signal。

D5 的 write cue 是 branch0 BL01，即大致使用 `-rank(C_dyn)` 作为 write score。这能工作，但它仍然只是“低动态更写”。更好的写入策略应该回答：memory 是否需要更新？当前 observation 是否可靠？这个 token 是否 reference-sensitive？

### 8.2 TTT residual 的定义

对 Pass-1 probe cache，branch0/TTT 层有 old fast weights $W_m$、key $k_i$、value $v_i$。定义 memory prediction：

$$
\hat v_i=f_{W_m}(k_i).
$$

残差：

$$
e_i=\frac{\|\hat v_i-v_i\|_2}{\|v_i\|_2+\epsilon}.
$$

然后定义 observation reliability：

$$
R_i^{obs}=(1-D_{read,i})(1-C_{unc,i})(1-C_{occ,i}).
$$

再加入 reference protection：

$$
S_{write,i}=\operatorname{rank}(e_i)\cdot R_i^{obs}\cdot (1-P_{ref,i}).
$$

解释：

- $e_i$ 高：当前 memory 没解释好这个 token；
- $R_i^{obs}$ 高：当前 token 可靠、非动态、非遮挡；
- $1-P_{ref}$ 高：不是不该扰动的 reference token。

### 8.3 alignment confidence 写入

参考 TTT3R，也可以计算 alignment confidence：

$$
\beta_i^{align}=\sigma\left(\sum_j Q_{memory,i}K_{obs,j}^\top\right).
$$

LoGeR 的具体 TTT cache 不是 CUT3R state-observation cross-attention，但可以用 probe cache 中的 $q/k$、frame/chunk attention summary 或 TTT residual proxy 构造类似 confidence。第一版可以把它作为辅助：

$$
S_{write,i}^{align}=\beta_i^{align}\cdot R_i^{obs}\cdot(1-P_{ref,i}).
$$

如果 alignment confidence 不稳定，则不进入 full candidate。

### 8.4 E3 写入策略

E3 固定 read path 使用 E1/E2 最好的 read cue。如果 E1/E2 还没有明确更好，则使用 D5-04 beta=1.0 作为 balanced read reference。

写入策略包括：

| ID | Write cue | Policy | Branch | 目的 |
|---|---|---|---|---|
| E3F-00 | probe_native | no controlled write | branch none | read-only reference |
| E3F-01 | old_dyn | dense MP-01 | branch0 | D5 write reference |
| E3F-02 | TTT residual $e_i$ | dense MP-01 | branch0 | memory residual 是否有效 |
| E3F-03 | residual × reliability | dense MP-01 | branch0 | update-needed 主候选 |
| E3F-04 | alignment confidence | dense MP-01 | branch0 | TTT3R-style confidence |
| E3F-05 | residual × reliability + ref protection | dense MP-01 | branch0 | rotation-safe write |

dense MP-01 保持：

$$
p_i=\operatorname{clip}(1+\alpha\hat S_i,0.8,1.2),\quad \alpha=0.1,
$$

并执行 branch0 eta-normalization。

### 8.5 E3 判断标准

E3 的通过条件是：

$$
ATE < \min(ATE_{D5}, ATE_{E1/E2\ best})
$$

并且 rotation/final 不能比 D5 明显更差。更具体：

| 条件 | 解释 |
|---|---|
| $ATE < 39.49$ | 写入策略比 D5 更好 |
| $ATE < 39.0$ | update-needed write 明确有效 |
| $ATE \approx 39.5$ 且 $Rot \le 9.5^\circ$ | 写入策略更 balanced |
| $FinalErr \le 5.5$ | endpoint 副作用被控制 |

如果 residual write 全部不如 old_dyn branch0 write，则说明当前 LoGeR TTT residual proxy 与 KITTI ATE 目标不一致，E4 只在 old_dyn write 上测试 sparse preserve。

---

## 9. Experiment Block E4：Sparse Exact-Preserve Write

### 9.1 实验目的

E4 验证 H4：相比 dense MP-01，sparse exact-preserve 是否更适合长期 TTT memory。D5 的 branch0 write 仍然是 dense reweighting，每个 token 都会对 branch0 update 有非零贡献。若长期 side effect 来自 dense perturbation，sparse route 应该改善 rotation/final stability。

### 9.2 sparse write 形式

给定 write score $S_i$，选出 TopK 写入 token：

$$
M_i=\mathbf{1}[S_i\in\operatorname{TopK}_r(S)].
$$

其中 $r$ 是 selected ratio，例如 95%、85%、70%。branch0 prior：

$$
p_i^{sparse}=\frac{M_i}{\frac{\sum_j\eta_jM_j}{\sum_j\eta_j+\epsilon}+\epsilon}.
$$

未选 token 在 branch0 update 中相当于 no-write / exact preserve。branch1 和 branch2 保持 unity。

可选 soft sparse：

$$
p_i=M_i\cdot p_i^{dense}+(1-M_i)\cdot 0.
$$

再做 eta normalization。

### 9.3 E4 实验矩阵

E4 固定 read path 为 E1/E2 best 或 D5-04 balanced read。写入 score 从 E3 里选两个：old_dyn 和 residual×reliability。

| ID | Write score | Selected ratio | Read | 目的 |
|---|---|---:|---|---|
| E4F-00 | dense D5 write | 100% | reference | D5 write reference |
| E4F-01 | old_dyn | 95% | fixed | 极轻 exact preserve |
| E4F-02 | old_dyn | 85% | fixed | 中等 sparse |
| E4F-03 | old_dyn | 70% | fixed | 强 sparse |
| E4F-04 | residual × reliability | 95% | fixed | update-needed sparse |
| E4F-05 | residual × reliability | 85% | fixed | update-needed sparse |
| E4F-06 | residual × reliability | 70% | fixed | update-needed sparse |

### 9.4 E4 判断标准

如果 95% sparse 比 dense 更好或 rotation/final 更稳，说明 exact preserve 有价值，后续可在 selected ratio 90–98% 之间细调。

如果 85% 或 70% 明显恶化，说明 LoGeR TTT 仍需要大部分 token 密集写入，sparse 只能作为轻度 preservation。

E4 成功条件：

$$
ATE < 39.49\text{ m}
$$

或：

$$
ATE \le 39.60\text{ m},\quad Rot \le 9.5^\circ,\quad FinalErr \le 5.5\text{ m}.
$$

如果 sparse write 没有任何 balanced gain，则停止 sparse 方向，不再做 selected-ratio sweep。

---

## 10. Experiment Block E5：Read/Write Pair Search

### 10.1 实验目的

E1–E4 分别测试 read protection、read modulation、update-needed write、sparse write。E5 把最优 read cue 与最优 write cue 成对组合，寻找真正的 Phase E 主候选。

E5 不是全组合大矩阵。它只使用前面阶段晋级的候选。

### 10.2 E5 组合规则

从 E1/E2 选最多两个 read variants：

```text
R_best_ATE
R_best_balanced
```

从 E3/E4 选最多三个 write variants：

```text
W_D5_reference
W_best_update_needed
W_best_sparse
```

组合时 beta 只选两个：

```text
beta = 1.0
beta = 1.25
```

如果 E1/E2 已经显示 beta=1.0 更 balanced，就只在最终 ATE candidate 上跑 beta=1.25。

### 10.3 E5 实验矩阵

| ID | Read variant | Write variant | beta | 目的 |
|---|---|---|---:|---|
| E5F-00 | D5 old_dyn | D5 old_dyn branch0 | 1.0/1.25 | reference repeat |
| E5F-01 | R_best_balanced | D5 old_dyn branch0 | 1.0 | read protection 是否已足够 |
| E5F-02 | R_best_balanced | W_update_needed | 1.0 | 主 co-design candidate |
| E5F-03 | R_best_balanced | W_sparse | 1.0 | sparse balanced candidate |
| E5F-04 | R_best_ATE | W_update_needed | 1.25 | 最强 ATE candidate |
| E5F-05 | R_best_ATE | W_sparse | 1.25 | 最强 sparse ATE candidate |

最多 6 个 full runs。若 E5F-02 或 E5F-03 已达到主模型候选，不继续跑后续低优先级组合。

### 10.4 E5 判断标准

E5 的目标比 E1–E4 更高：

| 条件 | 结论 |
|---|---|
| $ATE < 39.0$ | Phase E 明确优于 D5，可继续优化 |
| $ATE < 38.0$ | 主模型候选，进入跨序列验证 |
| $ATE \le 39.5$ 且 $Rot \le 9.3^\circ$ 且 $FinalErr \le 5.0$ | balanced candidate，适合进一步保护和验证 |
| $ATE$ 变好但 $Rot > 10^\circ$ 或 $FinalErr > 6.8$ | ATE diagnostic，不作为主候选 |

---

## 11. Experiment Block E6：Worst-Chunk Focused Rescue

### 11.1 实验目的

如果 E1–E5 得到的最好候选仍然卡在 39.3–39.6 m，且 worst chunks 仍集中在 `[203,293)` 和 `[435,496)`，则 E6 做 targeted rescue。E6 不是全序列 sweep，而是先对 worst chunks 做分析，再决定是否引入 schedule 或 localized protection。

### 11.2 分析内容

对 worst chunks 比较：

```text
D5-04
D5-07
E_best_balanced
E_best_ATE
```

记录：

- old_dyn 是否异常高；
- reference protection 是否覆盖到这些区域；
- branch0 write prior 是否过度集中；
- TTT residual 是否在这些 chunk 异常；
- beta=1.25 是否在这些 chunk 比 beta=1.0 更激进；
- endpoint drift 是否从这些 chunk 后开始扩大。

### 11.3 E6 可尝试的 rescue

若发现 read bias 在这些 chunk 太强，可用 per-chunk reliability gate：

$$
g_m=\operatorname{clip}\left(\frac{\tau_q-Q_m}{\sigma_q},0,1\right)
$$

其中 $Q_m$ 可以是：

```text
reference collision
bias energy on protected tokens
fragmentation
mean uncertainty
boundary jump predictor
```

read beta 改成：

$$
\beta_m=g_m\beta.
$$

若发现 write prior 过度集中，可用 write confidence gate：

$$
p_i^{write}=1+g_m\alpha\hat S_i.
$$

若发现问题来自 overlap / boundary，可增强 overlap protection。

### 11.4 E6 判断标准

E6 只有在 dashboard 明确定位局部 failure 时才运行 full。否则不允许用 manual schedule 直接追指标。

通过条件：

$$
ATE < E5\ best
$$

并且 worst-chunk RMSE 至少下降 10%。若 worst chunks 改善但 full ATE 不变，可以作为后续 reliability gate 的证据，但不作为主模型。

---

## 12. Optional E7：SWA / Chunk Attention Rescue

### 12.1 何时启动 E7

默认不启动 E7。只有当 dashboard 显示以下明确现象时才启动：

- SWA previous keys 中高 dynamic / high uncertainty token 被 current chunk 大量读取；
- chunk boundary pose jump 与 SWA read distribution 高相关；
- chunk-wise attention 中 moving object keys 被 static structure queries 高频读取，且 frame-attention protection 无法解决。

### 12.2 SWA read gate

如果启动 SWA，第一版只做 previous-key gate：

$$
B_k^{swa}=\beta_{swa}\log(1-D_{prev,k}^{prot}+\epsilon).
$$

必须保护 overlap / reference tokens：

$$
D_{prev,k}^{prot}=D_{prev,k}(1-P_{ref,k}).
$$

不做 dense query-key pair bias，不做 all-layer SWA control。

### 12.3 Chunk-wise attention gate

如果启动 chunk attention，第一版只做 early / light gate，并且沿用 protected old_dyn：

$$
B_{ij}^{chunk}=\beta_c\log(1-(1-D_i)D_j^{prot}+\epsilon).
$$

不允许 full-layer hard mask。

### 12.4 E7 判断标准

E7 成功必须明显超过 E5 best：

$$
ATE < E5\ best - 0.2\text{ m}
$$

否则说明 SWA/chunk control 增加复杂度但没有足够收益，应该回到 read/write branch0 co-design。

---

## 13. 推荐执行顺序

Phase E 推荐按下面顺序执行。

### Step 0：Correctness and reference repeat

先跑：

1. HMC no-control LoGeR repeat；
2. HMC no-control LoGeR* repeat；
3. D5-04 repeat；
4. D5-07 repeat；
5. new identity hooks repeat D5。

这一步通过后才能进入 E1。

### Step 1：Dashboard first

生成 Phase E dashboard，重点看 D5 worst chunks `[203,293)` 和 `[435,496)`。如果 dashboard 显示某类 reference token 明显被 dynamic bias 误伤，优先对应 protection。

### Step 2：E1 read protection

先做 protection smoke，然后选最多 4 个 full。目标是保住 ATE 同时修 rotation/final。若 E1 失败，说明 read protection 不是主要副作用来源，进入 E3。

### Step 3：E3 update-needed write

固定 best read，测试 residual / residual×reliability / alignment write。若 residual write 有效，再进入 E4 sparse。若无效，保留 old_dyn branch0 write。

### Step 4：E4 sparse write

只在 E3 或 D5 write 上做 selected ratio 95/85/70。不要做更多比例，除非 95% 明确有效。

### Step 5：E2 modulated read cue

如果 E1/E3/E4 仍不能达标，再尝试 old_dyn 的 Gram / semantic / true-flow modulation。不要直接替换 old_dyn。

### Step 6：E5 pair search

把最好的 read 和 write 组合。最多 6 个 full runs。

### Step 7：E6/E7 rescue

只有 E5 仍卡住且 dashboard 明确定位 failure 时启动。

---

## 14. 预期结果与分支决策

### 14.1 如果 E1 reference protection 有效

如果 E1 得到：

$$
ATE\approx 39.5\text{ m},\quad Rot<9.5^\circ,\quad FinalErr<5.5\text{ m},
$$

说明 D5 的副作用主要来自 read bias 误伤 reference token。下一步应固定 protected read，进入 E3/E4 改写入。此时不要继续探索 Gram/flow read modulator。

### 14.2 如果 E3 update-needed write 有效

如果 residual×reliability write 得到：

$$
ATE<39.0\text{ m},
$$

说明 old_dyn write 不是最佳 memory update cue。下一步应该做 sparse residual route 和 branch/layer ablation。如果同时 rotation/final 改善，则 update-needed write 成为 Phase E 主线。

### 14.3 如果 E4 sparse write 有效

如果 sparse 95% 或 85% selected 改善 final error/rotation，说明 dense MP-01 的长期扰动确实存在。下一步可以细调 selected ratio 90–98%，但不能扩展到过多组合。

### 14.4 如果只有 E2 modulation 有效

如果 Gram/true-flow/semantic modulation 才有效，说明 old_dyn backbone 仍有可塑性，但需要更稳的 motion/static discriminator。若 true flow 有明显收益，应考虑实现更完整的 flow/epipolar cue；若 semantic rescue 有收益，应完善 KITTI prompt 和 structure mask quality。

### 14.5 如果 E1–E5 都没有超过 D5

如果所有实验都不能超过 D5 或不能改善 rotation/final，那么 Phase E 应停止。当前方法可以定格为：

```text
D5-04 balanced candidate
D5-07 best ATE candidate
```

之后应转向更大改动：

- 真光流/语义/外部 motion cue；
- learned reliability gate；
- LoGeR* style alignment integration；
- SWA/chunk attention deeper controller；
- 或跨序列泛化验证而非继续 KITTI 01 调参。

---

## 15. 最终希望得到的模型形式

如果 Phase E 成功，最终模型不应该是“一个更强的 C_dyn”。它应该是一个 read/write/protection 分工明确的 HMC controller：

```text
Pass 1 native probe:
  - output native geometry
  - extract old_dyn / internal cues / TTT residual
  - build reference protection
  - cache native TTT update primitives

Pass 2 controlled read:
  - use protected/modulated read cue
  - inject early frame-attention pair bias
  - output controlled geometry

Commit:
  - discard Pass-2 controlled TTT state
  - replay safe branch0 write from Pass-1 probe cache
  - write cue uses update-needed / reliability / sparse preserve

Next chunk:
  - reads the safe committed TTT state
```

它的核心不是“动态区域少写”，而是：

$$
\boxed{
\text{Correct current output through read control, and update future memory through probe-cache safe write.}
}
$$

Phase E 的目标就是把这句话变成一个可复现、可解释、指标足够强的主模型。

---

## 16. Phase E 输出物

Phase E 每个 full run 必须保存：

```text
01.txt trajectory
kitti_benchmark.log
hmc_correctness_summary.json
control_trace.jsonl
phaseE_dashboard.jsonl
read_cue_summary.json
write_prior_summary.json
memory_side_effect_summary.json
trajectory_diagnostics.csv/json
worst_chunk_visualizations/
wandb run
```

每个晋级候选必须额外保存：

```text
repeat run
per-chunk read/write maps
D5 reference comparison plots
segment ATE comparison
rotation/final endpoint diagnostic
```

没有这些诊断的 full ATE 结果不作为模型选择依据。

---

## 17. 本阶段最重要的停止条件

Phase E 计划很大，但不能无限扩展。设置以下停止条件。

1. 如果 E1–E4 都不能超过 D5，也不能改善 rotation/final，则停止 Phase E，保留 D5。
2. 如果 E2 中所有 modulator 都不如 old_dyn，则不再尝试直接替换 old_dyn，除非有 true external flow。
3. 如果 E3 residual write 全失败，则不再做 residual sparse route。
4. 如果 E4 sparse 95% 都无效，则不再细扫 sparse selected ratio。
5. 如果 E5 组合仍不能达到 $ATE<39.0$ 或 balanced pass，则不再继续 KITTI 01 上手工规则搜索。
6. 如果某个实验只在 64/256/stateful slice 上好，但 full 失败，则不再因为短程结果晋级。

---

## 18. 当前最推荐的第一批 full runs

如果资源有限，我建议 Phase E 第一批 full 只跑下面 8 个：

| Priority | Run | 说明 |
|---:|---|---|
| 0 | D5-04R | beta=1.0 balanced repeat |
| 0 | D5-07R | beta=1.25 best ATE repeat |
| 1 | E1F-overlap-protect-b100 | 最小 reference protection |
| 2 | E1F-anchor-protect-b100 | 高 anchor 保护 |
| 3 | E1F-combined-light-b100 | 综合轻保护 |
| 4 | E3F-residual-reliability-b100 | update-needed write 主候选 |
| 5 | E4F-oldDyn-sparse95-b100 | sparse exact preserve 检查 |
| 6 | E5F-bestRead-bestWrite-b100 | 第一轮 co-design 组合 |

这 8 个 run 之后，基本就能判断 Phase E 是继续向 `<39` / `<38` 推进，还是当前 D5 已接近手工规则上限。

---


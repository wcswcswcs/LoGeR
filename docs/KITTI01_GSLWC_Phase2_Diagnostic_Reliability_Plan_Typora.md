# KITTI 01 GSL-WC 下一阶段实验方案：从全局写入控制转向“可用 cue、可用分支、可用场景”的识别

本文档是当前 KITTI 01 GSL-WC 实验之后的新一轮推进方案。它不再沿着“继续调 `floor / lambda / alpha / fusion weight`”这个方向走，而是把实验目标改成：找出当前 geometry prior 到底在哪些 cue、哪些 TTT branch/layer、哪些序列片段上有用，并把它们组合成一个只在可靠场景启用的 write controller。

当前最重要的事实是：基础工程链路已经可信，但 geometry-only write-control 还没有形成显著主模型。LoGeR native reproduction 和 unity replay parity 已通过；fixed-merge 后 unity replay baseline 是 $41.6193\text{ m}$，当前最好 G3/F8/k5/floor07 是 $41.5765\text{ m}$，只比 unity 好 $0.0428\text{ m}$。eta-weighted mean-preserving reweighting 也已经验证了“总写入量不足”不是主因：MP-01、MP-02、MP-03 都保持了 branch-wise post eta ratio 约等于 $1.0$，但 ATE 反而随重分配强度增大而变差，同时 rotation 单调变好。Segment diagnostics 显示 MP 并非全程无效，它在 $[300,500)$ 片段上能显著改善，但同样会放大 $[0,200)$、$[400,600)$ 和后段的失败。Stage-B reprojection proxy 的第一版也没有改善 MP-02 full ATE。

因此，下一阶段的核心判断不是“prior 有没有信号”，而是：

$$
\text{当前 geometry ranking 有局部信号，但没有可靠性模型。}
$$

我们要用下面的实验逐步回答三个问题：第一，哪些 cue 是安全的，哪些 cue 会误伤 translation-useful memory；第二，prior 是否只应该作用在某些 TTT branch 或 layer，而不是全层全分支一刀切；第三，局部 segment win 是否可以通过 manual schedule 或 automatic reliability gate 转化为 full-sequence ATE 收益。

---

## 1. 当前实验结论如何改变下一步策略

之前的实验已经排除了几个基础怀疑。`debug_prior_mode` 的 patch-only、special-only、frame ramp、reverse frame ramp、roll prior 表明 variable prior 的 token layout 没有明显错位。`cache_l` 与 `L_tok` 匹配，patch/special synthetic prior 也落在预期 token 类型上。all-one replay 和 unity replay 也已经通过，因此后续失败不应再简单归因于 replay 或 prefix alignment。

eta-mean-preserving 实验进一步排除了“只是整体写少了”的解释。对每个 branch 做 eta normalization 后，MP-01、MP-02、MP-03 的 post-normalized lr-weighted write ratio 都保持在 $1.0$ 附近，但 full KITTI 01 ATE 分别变成 $41.7479\text{ m}$、$41.8596\text{ m}$、$41.9070\text{ m}$。与此同时 rotation RMSE 从 unity 的 $8.9508^\circ$ 降到 MP-03 的 $8.5595^\circ$。这说明当前 ranking 改变了 token dominance，确实压掉了某些 rotation-noisy token，但也改变了 LoGeR 原生对 translation / scale / trajectory continuity 有用的写入结构。

Segment diagnostics 是下一阶段最关键的线索。MP-03 在 $[300,500)$ 的 200-frame segment 上可以做到 $-0.6326\text{ m}$ 的 local delta，但同一个 MP-03 在 $[400,600)$、$[0,200)$、$[800,1000)$ 又显著变差。这个现象说明：

$$
\text{同一个 score 和同一个 strength 在不同场景下有相反效果。}
$$

因此，继续扫 scalar 大概率只能在“局部收益”和“局部失败”之间移动，不会变成稳定主模型。下一阶段应该把实验从 scalar sweep 改成 diagnostic-driven model selection。

---

## 2. 新阶段的中心假设

本阶段围绕四个假设设计实验。每个假设都必须被实验支持后，才进入下一步。

第一个假设是 **mixed geometry score 的失败主要来自某些 cue 分量，而不是所有 geometry cue 都无效**。当前 G3 score 把 $C_{stat}$、$C_{anchor}$、$C_{dyn}$、$C_{occ}$、$C_{unc}$ 混在一起。如果 positive cue 比 mixed cue 更安全，说明风险项在误伤有用 token；如果 risk-only 只改善 rotation 而伤害 ATE，说明 risk cue 更适合作为 reliability signal，而不是直接作为 write penalty。

第二个假设是 **prior 不应该同时作用于所有 TTT branch 和所有 layer**。当前 controller 共享同一个 token prior 给所有 fast-weight branch 和 layer，但 TTT 的三条分支 $W^{(0)}$、$W^{(1)}$、$W^{(2)}$ 可能承担不同功能。同一个 ranking 可能对某些 branch 是 rotation denoising，对另一些 branch 是 translation damage。一刀切会把收益和伤害混合。

第三个假设是 **segment-level local wins 是可以被调度利用的**。如果 oracle manual schedule 只在已知好片段启用 MP，就能把 full ATE 明显推低，那么说明当前 prior 的问题不是“无效”，而是缺少可靠性判别。如果 manual schedule 也无法超过当前 best，那么 local wins 可能不是可组合收益，或者 write effect 的长期传播让局部启停无法稳定利用。

第四个假设是 **自动 reliability gate 可以规避失败片段**。当前 MP 的失败不是强度问题，而是“什么时候信这个 score”的问题。下一版 controller 应该写成：

$$
p_i = 1 + g_m \alpha \hat{s}_i,
$$

其中 $g_m\in[0,1]$ 是 chunk-level reliability gate。cue 可信时启用 reweighting，cue 不可信时回退 unity replay。

---

## 3. 统一实验协议与判断标准

所有 full-sequence model-selection 实验都使用 KITTI 01 全序列，固定 LoGeR checkpoint、chunk schedule 和 merge route：

```bash
--checkpoint ckpts/LoGeR/latest.pt
--config ckpts/LoGeR/original_config.yaml
--ttt_write_mode semantic
--stage_c_mode none
--geometry_edge_rtol 0.0
--chunk_size 32
--chunk_overlap 3
--window_size 32
--overlap_size 3
--reset_every 5
--lambda_min 1.0
--lambda_max 1.0
--a_min_special 1.0
```

除非特别说明，所有本阶段实验都先不启用 Stage C。语义分支只在 geometry reliability 形成之后作为 protection 使用，不作为主 prior。

本阶段的 reference metrics 固定为：

| Reference | ATE RMSE | Rotation RMSE | 用途 |
|---|---:|---:|---|
| Unity replay baseline | $41.6193\text{ m}$ | $8.9508^\circ$ | replay baseline |
| Current best G3/F8/k5/floor07 | $41.5765\text{ m}$ | $8.9914^\circ$ | 当前 GSL-WC best |
| MP-02 reference | $41.8596\text{ m}$ | $8.7987^\circ$ | mean-preserving mixed-score failed baseline |

判断标准分成三档。第一档是 **diagnostic useful**：虽然 full ATE 未必超过当前 best，但能清楚说明某个 cue、branch、layer 或 segment schedule 的作用。第二档是 **model useful**：full ATE 必须小于 $41.5765\text{ m}$，即超过当前 best。第三档是 **strong candidate**：full ATE 进入 $41.3\text{ m}$ 以下，同时 rotation 不明显劣化。只有 strong candidate 才值得进一步加 Stage C semantic reference 或更复杂的 reliability gate。

为了避免只看 full ATE 掩盖失败，本阶段所有 candidate 都必须生成以下诊断文件：

```text
prior_debug.jsonl
segment_ate_sim3_nonoverlap_100.csv
segment_ate_sim3_nonoverlap_50.csv
segment_delta_summary.json
```

每个 chunk 至少记录：

$$
\operatorname{mean}(C_{stat}),\quad
\operatorname{mean}(C_{anchor}),\quad
\operatorname{mean}(C_{dyn}),\quad
\operatorname{mean}(C_{occ}),\quad
\operatorname{mean}(C_{unc}),
$$

$$
Q_{90}(s)-Q_{10}(s),\quad
\operatorname{mean}(p_i),\quad
\operatorname{std}(p_i),\quad
\frac{\sum_i \eta_i p_i}{\sum_i \eta_i+\epsilon}
$$

并按 branch 分别记录 eta-normalized write ratio。long KITTI run 不再用 `tee` 长时间流式输出，统一使用：

```bash
> 01.log 2>&1
```

这是因为之前文件系统和 stdout back-pressure 都造成过不必要风险。

---

## 4. 实验一：Score Decomposition，拆清楚哪个 cue 在帮、哪个 cue 在害

### 4.1 实验目的

这个实验要回答：当前 G3 mixed score 的问题到底来自 positive cue 不准，还是 risk cue 误伤，或者两者都不可靠。

当前 mixed score 可写为：

$$
z_{geo}
=
\lambda_s C_{stat}
+
\lambda_a C_{anchor}
-
\lambda_d C_{dyn}
-
\lambda_o C_{occ}
-
\lambda_u C_{unc}.
$$

但这个混合量的 full ATE 收益很小，MP 版本甚至失败。因此我们要把它拆开，分别测试 positive-only、risk-only 和最小正负组合。

所有 score 都先池化到 patch token。对任意 patch score $x_i$，定义 percentile-rank normalization：

$$
r_i(x)=2\cdot \operatorname{PercentRank}(x_i)-1.
$$

正向 cue 使用 $r_i(x)$，风险 cue 使用 $-r_i(x)$，这样较大的 $s_i$ 始终表示“更应该主导写入”。最终 prior 使用最温和的 eta-mean-preserving policy：

$$
\tilde p_i=\operatorname{clip}(1+\alpha s_i,p_{min},p_{max}),
$$

$$
p_i=
\frac{\tilde p_i}
{\frac{\sum_j \eta_j\tilde p_j}{\sum_j \eta_j+\epsilon}+\epsilon}.
$$

第一批固定：

$$
\alpha=0.1,\qquad p_{min}=0.8,\qquad p_{max}=1.2,
$$

并且 special tokens 固定为 $1.0$。不用更强 MP-02/MP-03，因为前一轮已经说明强度越大越容易放大失败。

### 4.2 实验安排

| Run ID | Score $s_i$ | 实验含义 | 预期能回答的问题 |
|---|---|---|---|
| SD-01 | $r(C_{anchor})$ | 纯 anchor boost | 只强化长期锚点是否比 mixed score 更安全 |
| SD-02 | $r(0.5C_{stat}+0.5C_{anchor})$ | positive-only | 静态/锚点正向排序是否可用于 translation-safe reweighting |
| SD-03 | $-r(C_{dyn})$ | dynamic penalty only | 动态 cue 是否只是 rotation denoising，还是也有 ATE 收益 |
| SD-04 | $-r(C_{unc})$ | uncertainty penalty only | uncertainty 是否误伤低纹理道路、远处结构和 translation-useful tokens |
| SD-05 | $-r(C_{occ})$ | occlusion penalty only | 遮挡 cue 是否值得进入 write-control，还是只应做 reliability flag |
| SD-06 | $r(C_{anchor})-0.5r(C_{dyn})$ | 最小正负组合 | 最小 anchor + dynamic 是否优于完整 mixed score |

所有 SD runs 都跑 full KITTI 01，并生成 non-overlap 100/50 frame segment diagnostics。虽然 full runs 成本较高，但这些实验是当前阶段的核心诊断，不建议只跑 128-frame smoke 后下结论。128-frame 只能作为代码 sanity check。

### 4.3 判断标准

如果 SD-01 或 SD-02 的 full ATE 小于 $41.5765\text{ m}$，说明 positive cue 是当前最安全的主线，后续 branch/layer selective 和 reliability gate 都应基于 positive-only score 继续。此时 risk cue 不应继续作为 token-level penalty，而应转为 chunk-level reliability signal。

如果 SD-03、SD-04、SD-05 呈现“rotation 变好但 ATE 变差”，说明风险项确实具有 denoising signal，但不适合作为全局 write ranking。特别是如果 SD-04 明显伤害 ATE，则 $C_{unc}$ 应从 suppression term 中移除，改成 gate：当 uncertainty 大面积升高时直接回退 unity，而不是对不确定区域做强惩罚。

如果 SD-06 明显好于 SD-01/SD-02，说明轻微 dynamic penalty 可以和 anchor boost 共存；如果 SD-06 变差，则后续主 prior 只保留 positive cue。

如果所有 SD runs 都不能接近当前 best，同时 segment diagnostics 也没有比 MP 更清晰的局部规律，则说明当前五通道 cue 本身不足以成为主模型，需要提前转向 Stage-C protection 或外部 motion cue。

---

## 5. 实验二：Branch / Layer Selective Prior，验证 prior 是否接错作用位置

### 5.1 实验目的

前一轮 MP 失败并不一定说明 token score 完全无用，也可能是同一个 prior 同时作用于所有 branch/layer 太粗。LoGeR 的 TTT fast-weight module 有三条分支：

$$
f_W(x)=\big(\operatorname{SiLU}(xW^{(0)})\odot(xW^{(2)})\big)W^{(1)}.
$$

当前做法相当于对 $W^{(0)}$、$W^{(1)}$、$W^{(2)}$ 的 update 都使用同一个 $p_i$。但 rotation 和 translation 的 trade-off 可能来自 branch coupling：某些 branch 上的 reweighting 有利于 rotation，另一些 branch 上会伤害 translation。Layer 也可能类似，早层负责局部几何 adaptation，晚层负责更全局的几何记忆；一刀切会破坏 translation-useful adaptation。

### 5.2 需要加入的实现开关

建议加入两个参数：

```bash
--prior_branch_mask 0,1,2
--prior_layer_mode all|early|late|middle|single
```

`prior_branch_mask` 决定哪些 branch 使用 $p_i$，未选中的 branch 使用 unity prior。`prior_layer_mode` 决定哪些 TTT layers 启用 prior，未选中 layers 使用 unity replay。branch-wise eta normalization 只在启用 prior 的 branch 上执行。

### 5.3 实验安排

实验二不直接铺大矩阵，而是先用实验一中最安全的 score。如果实验一还没有完成，则临时使用 SD-02 positive-only score；不要用 MP-02 mixed score 做主实验，因为它已经 full 失败。

| Run ID | Branch | Layer | Score | Policy | 目的 |
|---|---|---|---|---|---|
| BL-01 | branch 0 only | all | best SD score | MP-01 | 测试 gate branch 是否可安全 reweight |
| BL-02 | branch 1 only | all | best SD score | MP-01 | 测试 output projection branch 是否是 ATE 伤害源 |
| BL-03 | branch 2 only | all | best SD score | MP-01 | 测试 content branch 是否可安全 reweight |
| BL-04 | all branches | late half | best SD score | MP-01 | 测试 prior 是否只适合高层 memory |
| BL-05 | all branches | early half | best SD score | MP-01 | 测试早层是否导致 translation damage |
| BL-06 | best branch | late half | best SD score | MP-01 | 组合最安全 branch 和 layer |

每个 BL run 先跑 128-frame smoke，确保没有实现错误和数值异常。只要 smoke 不崩，就跑 full，因为 128-frame prefix 在前一轮已经证明不能预测 full KITTI 01 行为。

### 5.4 判断标准

如果某个 branch-only run 满足：

$$
\Delta ATE \le 0
\quad\text{and}\quad
\Delta Rot < 0,
$$

相对于 unity replay 同时不伤 ATE 且改善 rotation，那么它就是后续 reliability gate 的首选接入位置。这里的 $\Delta ATE$ 和 $\Delta Rot$ 都相对于 unity replay。

如果某个 branch-only full ATE 小于 $41.5765\text{ m}$，说明 prior 本身可能没错，之前的问题主要是 branch coupling。若 BL-06 进一步进入 $41.3\text{ m}$ 以下，则可以暂时不升级 Stage B，直接进入 reliability-gated controller。

如果 late-only 比 early-only 稳定，后续默认只在 late half layers 使用 prior。如果 early-only 或 all-layer 全部伤害 ATE，说明 lower-level TTT adaptation 对 translation 很敏感，应尽量保留 native write-through。

如果所有 BL runs 都失败，并且失败模式仍是 rotation 变好、ATE 变差，则说明问题主要在 score 对 translation 目标不可靠，而不是 branch/layer 接入位置。

---

## 6. 实验三：Non-overlap Segment Diagnostics 与 Manual Schedule，验证局部收益是否可组合

### 6.1 实验目的

之前的 segment diagnostics 使用 200-frame stride 100 和 100-frame stride 50。它已经说明 MP 在 $[300,500)$ 有强 local win，但由于窗口重叠，仍然不能确定哪个更窄片段真正受益、哪个片段真正受害。下一步要做 non-overlap diagnostics，并用 manual schedule 测试 local wins 是否可转化为 full ATE。

这一步不是为了得到最终模型，而是为了验证一个因果问题：

$$
\text{如果只在局部有益区域启用 prior，full ATE 是否会下降？}
$$

如果答案是否定的，后续 automatic reliability gate 的价值会很有限。

### 6.2 诊断方法

对 unity、current best G3 floor07、MP-01/02/03、实验一和实验二中的最佳 candidate，生成 non-overlap segment ATE：

```text
100-frame non-overlap:
[0,100), [100,200), ..., [1000,1100)

50-frame non-overlap:
[0,50), [50,100), ..., [1050,1100)
```

对每个 segment 计算：

$$
\Delta ATE_s = ATE_s(\text{candidate})-ATE_s(\text{unity}).
$$

然后把 $\Delta ATE_s$ 与 segment 前 $1$ 到 $3$ 个 chunks 的 prior stats 关联，而不是只看同一 segment。原因是 TTT update 写给未来，chunk $m$ 的写入可能主要影响 $m+1$ 或 $m+2$。

### 6.3 Manual schedule 实验

先根据 non-overlap diagnostics 选出“好区域”和“坏区域”。现有 overlapping diagnostics 已经提示：$[300,500)$ 可能是好区域，$[0,200)$、$[400,600)$、$[800,1000)$ 可能是坏区域，但必须用 non-overlap diagnostics 重新确认。

然后设计三个 oracle schedule。这里的 oracle 只用于诊断，不作为最终模型汇报。

| Run ID | 策略 | 目的 |
|---|---|---|
| SCH-01 | 只在 confirmed good chunks 启用最强 local-win prior，其余 chunks 使用 unity | 测试 local win 是否可以被单独收集 |
| SCH-02 | 全局使用温和 prior，但在 confirmed bad chunks 回退 unity | 测试 bad-region avoidance 是否能恢复 full ATE |
| SCH-03 | 当前 best G3 floor07 作为默认，只在 confirmed good chunks 叠加 branch/layer selective MP | 测试 conservative baseline 能否叠加局部收益 |

schedule 的粒度使用 chunk index，而不是 frame index。chunk index 与 frame range 的对应关系由当前 chunk schedule 决定：

$$
\text{step}=\text{chunk\_size}-\text{chunk\_overlap}=29.
$$

所以一个 100-frame segment 大约覆盖 $3$ 到 $4$ 个 write chunks。manual schedule 应该从 segment 起点往前回看一到两个 chunks，避免错过 write-to-future effect。

### 6.4 判断标准

如果 SCH-01 或 SCH-02 的 full ATE 小于 $41.3\text{ m}$，说明局部收益是可组合的，下一步应该优先做 automatic reliability gate。

如果 manual schedule 只能达到 $41.57\text{ m}$ 附近，说明当前 local wins 很弱或被长期传播抵消，reliability gate 的上限有限。

如果 manual schedule 反而比 global MP 更差，说明局部 segment delta 不是直接 causal，可能由 Sim(3) alignment 或局部轨迹形状造成，不应该据此训练 gate。

---

## 7. 实验四：Reliability-gated Controller，从“固定强度”变成“可信时才启用”

### 7.1 实验目的

如果实验三说明 local wins 可以通过 schedule 利用，就需要把 oracle schedule 变成不使用 GT 的 automatic reliability gate。当前最合理的形式是：

$$
p_i=1+g_m\alpha s_i,
$$

其中 $g_m$ 是 chunk-level gate。$g_m=0$ 时退回 unity replay，$g_m=1$ 时使用完整 MP reweighting。

### 7.2 Gate 设计

初始 gate 不使用学习模型，只使用当前 chunk 的 cue statistics。定义 score contrast：

$$
c_m=Q_{90}(s_i)-Q_{10}(s_i).
$$

定义 anchor coverage：

$$
a_m=\operatorname{mean}(C_{anchor}).
$$

定义 uncertainty spread：

$$
u_m=\operatorname{mean}(C_{unc}).
$$

第一版 soft gate 使用：

$$
g_m=
\operatorname{clip}\left(\frac{c_m-\tau_c}{\sigma_c},0,1\right)
\cdot
\operatorname{clip}\left(\frac{a_m-\tau_a}{\sigma_a},0,1\right)
\cdot
\operatorname{clip}\left(\frac{\tau_u-u_m}{\sigma_u},0,1\right).
$$

阈值不要用 GT segment result 直接调，而是从 chunk statistics 的无监督分布取：

$$
\tau_c=Q_{50}(c_m),\quad
\sigma_c=Q_{75}(c_m)-Q_{50}(c_m)+\epsilon,
$$

$$
\tau_a=Q_{40}(a_m),\quad
\sigma_a=Q_{60}(a_m)-Q_{40}(a_m)+\epsilon,
$$

$$
\tau_u=Q_{60}(u_m),\quad
\sigma_u=Q_{60}(u_m)-Q_{40}(u_m)+\epsilon.
$$

这个 gate 的直觉是：score 没有空间对比度时不要信，anchor coverage 太低时不要信，uncertainty 大面积升高时不要信。

### 7.3 实验安排

| Run ID | Gate | Score | Branch/Layer | 目的 |
|---|---|---|---|---|
| RG-01 | contrast only | best SD score | best BL setting | 测试 score contrast 是否足够决定启用 prior |
| RG-02 | contrast × uncertainty | best SD score | best BL setting | 测试 high uncertainty 回退 unity 是否能减少 bad segments |
| RG-03 | contrast × anchor × uncertainty | best SD score | best BL setting | 完整 soft reliability gate |
| RG-04 | hard top-30% reliable chunks | best SD score | best BL setting | 测试只在最可信 chunks 启用 prior 的上限 |
| RG-05 | hard top-50% reliable chunks | best SD score | best BL setting | 测试更高覆盖率是否更好 |

如果实验一和实验二没有产出明确 best score / best branch，则 RG 系列暂缓，不要用已经失败的 MP-02 mixed score 强行做 gate。

### 7.4 判断标准

RG 系列必须同时看 full ATE 和 bad-segment delta。一个 gate 只有在满足下面条件时才算成功：

$$
ATE_{full}<41.5765\text{ m},
$$

并且相对于对应 non-gated prior，bad segments 的最大正 delta 至少下降 $50\%$。例如如果 MP-03 在某个 200-frame bad segment 上是 $+0.3012\text{ m}$，gated version 应该把同类 worst delta 压到 $+0.15\text{ m}$ 以下。

如果 RG full ATE 进入 $41.3\text{ m}$ 以下，说明当前 geometry cue 仍然可作为主模型的一部分。若 RG 只能接近当前 best，但无法显著超过，则 geometry heuristic 已接近上限，后续应该优先引入 semantic protection 或外部 motion cue。

---

## 8. 实验五：Cue Visualization Dashboard，决定是否继续升级 Stage B

### 8.1 实验目的

Stage-B reprojection 第一版没有解决问题，但这不等于 Stage-B 方向无效。下一步如果要继续改 proxy，必须先看 cue 本身，而不是直接把新 residual 接进 full KITTI 跑 ATE。

这个 dashboard 要回答：bad segments 中，prior 是否把 road / building / guardrail 这类 translation-useful static structure 压低了；good segments 中，prior 是否确实压掉了 moving cars、遮挡边界或几何不稳定区域；$C_{dyn}$ 是否仍然主要落在深度边缘，而不是完整动态物体。

### 8.2 可视化片段

固定抽取四类片段：

| Segment | 来源 | 用途 |
|---|---|---|
| $[0,200)$ | MP bad | 观察早期失败是否来自道路/建筑被误伤 |
| $[300,500)$ | MP good | 观察当前 score 为什么有效 |
| $[400,600)$ | MP bad | 观察好段后立即失败的原因 |
| $[800,1000)$ | late bad | 观察后期 drift 与 cue 分布关系 |

每个片段抽 $5$ 到 $10$ 帧，保存以下 panel：

```text
RGB
C_stat
C_dyn_explicit
C_dyn_implicit
C_occ
C_unc
C_anchor
G_write_geo
selected score rank
eta-weighted prior p_i
```

如果启用 branch/layer selective，还要保存每个 branch 的 effective multiplier summary。

### 8.3 判断标准

如果 bad segments 中 road、building、guardrail 被大量低 prior 覆盖，而 moving vehicles 反而没有被可靠识别，说明应该先做 structure protection，而不是继续强化 risk pooling。

如果 $C_{dyn}$ 在 good segments 里确实覆盖了 moving objects，而 bad segments 里主要覆盖深度边缘或低纹理区域，说明 dynamic cue 需要一个 reliability classifier，不适合全局使用。

如果 $C_{unc}$ 在低纹理道路和远处结构上大面积升高，且 SD-04 也伤害 ATE，那么 $C_{unc}$ 应永久移出 token-level penalty，只保留为 chunk-level gate。

如果 cue 可视化整体混乱，Stage-B heuristic 路线不应该继续投入太多 full-sequence sweep，应转向 external optical flow / semantic motion proxy 或学习式 controller。

---

## 9. Stage C 语义只作为 protection，不作为主 prior

只有当前面实验找到一个相对安全的 geometry score 或 branch/layer setting 后，才启用 Stage C。当前 Stage C 的 thing 漏检、tracking failure、stuff 边界粗糙仍然存在，所以语义不能主导 write policy，只能做 protection。

KITTI prompt bank 应该从 indoor prompt 改成 outdoor driving：

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

语义规则不改变 score 主体，只对 multiplier 做轻量修正。对 reliable structure mask：

$$
p_i\leftarrow\max(p_i,0.95),
$$

含义是不要让 geometry cue 把 road / building / sidewalk / guardrail 这类结构压得太低。对 movable thing 不直接压低，因为 KITTI 中 parked car 可能是有效 landmark。只有当 movable semantic 和 high dynamic cue 同时成立时，才做 mild cap：

$$
p_i\leftarrow\min(p_i,0.85).
$$

这个条件建议写成：

$$
Q_{mask}>0.7,
\quad
\operatorname{mean}_{u\in M_j}C_{dyn}^{nonocc}(u)>\tau_{dyn},
\quad
\operatorname{visibility\_ratio}(j)>0.5.
$$

语义实验只跑三组：structure protection only、structure protection + movable dynamic cap、low-value stuff mild cap。判断标准是：语义 protection 应该主要减少 bad segments 的 positive delta，而不能让 full ATE 比对应 geometry-only candidate 恶化超过 $0.1\text{ m}$。如果恶化超过这个范围，说明 Stage C 当前质量不足，语义应该继续停留在可视化/诊断层。

---

## 10. 推荐执行顺序

实际执行时不要同时开太多方向。推荐顺序如下。

第一步先完成 Score Decomposition。它会告诉我们当前五通道 cue 中哪些还能用。最优先看 SD-01 和 SD-02。如果 positive-only 已经比当前 best 更好，就立刻停止 risk-heavy 方向，把后续实验都建立在 positive-only 上。

第二步做 Branch / Layer Selective。它会告诉我们 prior 接入位置是否太粗。如果某个 branch 或 late-layer setting 能保留 rotation gain 且不伤 ATE，后续 reliability gate 就以这个 setting 为基础。

第三步做 non-overlap segment diagnostics 和 manual schedule。它会告诉我们 local wins 是否真能组合成 full-sequence gain。如果 oracle schedule 都无效，就不应该花大量时间做 automatic reliability gate。

第四步做 Reliability Gate。如果 RG 系列能稳定超过当前 best，并减少 bad-segment delta，就把它作为新的主模型候选。如果 RG 系列仍然只能在 $41.57\text{ m}$ 附近波动，则说明 geometry-only heuristic write-control 已到平台期。

第五步才加入 Stage C protection。Stage C 不应该在前四步之前进入主实验，否则很难判断收益或失败到底来自 geometry prior 还是语义 mask 质量。

---

## 11. 最终决策树

本阶段结束后按下面的规则决定项目方向。

如果 SD positive-only 或 BL selective 已经能达到：

$$
ATE<41.3\text{ m},
$$

则说明 geometry-only write-control 仍然有成为主模型的潜力，下一步应该集中强化 reliability gate 和 Stage-C structure protection。

如果只有 manual schedule 能达到 $41.3\text{ m}$ 以下，而 automatic RG 做不到，说明 local signal 存在但当前无监督 reliability feature 不够，需要考虑轻量学习式 gate。训练目标可以来自 segment outcome，但需要小心避免只对 KITTI 01 过拟合。

如果 SD、BL、SCH、RG 都无法超过 $41.5765\text{ m}$，则应承认当前 geometry-only heuristic prior 在 KITTI 01 上已经到达平台期。此时不应继续扫 scalar，而应转向两条更强路径：一是引入 external motion cue，例如 optical flow / epipolar residual / dynamic object track；二是把 write policy 从手工规则升级为学习式 controller。

如果 Stage C protection 能减少 bad-segment failure，但 full ATE 仍不显著改善，说明语义有辅助价值但不能单独解决问题。下一阶段应把 semantic protection 和 external motion cue 结合，而不是让 semantic masklets 主导所有 token prior。

---

## 12. 本阶段预期产出

本阶段不是只追求一个新 best number，而是要产出可解释结论。完成后应该能回答以下问题：

1. $C_{stat}$、$C_{anchor}$、$C_{dyn}$、$C_{occ}$、$C_{unc}$ 中，哪些适合作为 token-level ranking，哪些只能作为 reliability gate。
2. prior 是否应该只作用于某个 TTT branch 或 late layers。
3. MP 在 $[300,500)$ 的 local win 是否可以被 manual schedule 利用。
4. 不使用 GT 的 reliability gate 是否能规避 $[0,200)$、$[400,600)$、后段 bad regions。
5. Stage-C KITTI prompt 的语义结果是否足以做 structure protection。

如果这些问题都回答清楚，即使 full ATE 没有马上大幅降低，项目也会从“调不动”变成“知道下一步该换 signal 还是换 controller”。

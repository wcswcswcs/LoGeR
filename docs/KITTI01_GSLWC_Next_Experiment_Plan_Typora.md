# KITTI 01 GSL-WC 下一阶段实验方案：从“少写”转向“可信 token 重分配”

版本：2026-04-28  
适用对象：LoGeR + Dynamic Cue Extractor + Semantic Prior Generator + TTT Write Controller 在 KITTI odometry sequence 01 上的下一轮实验  
当前主目标：建立一个能在 KITTI 01 上稳定、可解释地超过 LoGeR / unity replay baseline 的主模型，而不是继续获得毫米级或厘米级的 fragile gain。

---

## 1. 当前实验给出的结论

当前实验已经完成了两个必要 gate。第一，native LoGeR 复现通过。LoGeR 在 KITTI 01 上得到 ATE RMSE 41.7502 m，和表格参考值 41.64 m 的差异只有 0.1102 m；LoGeR* 也复现到 47.9793 m，和参考值 47.91 m 基本一致。这说明 checkpoint、输入序列、窗口配置、reset、merge 和评估脚本没有明显基础错误。

第二，unity replay parity 通过。完整 KITTI 01 上，unity replay 得到 ATE RMSE 41.6193 m，和 native LoGeR 的 41.7502 m 很接近；轨迹 sanity check 中 translation diff mean 只有 0.0406 m，translation diff max 只有 0.0577 m。这说明在 $A_{tok}=1$ 的情况下，TTT replay 路径整体可信。

但这两个 gate 通过之后，geometry-only write control 的进展很弱。固定 merge 之后，当前最好结果是 `G3 + F8 calibrated soft-or + k_intra=5 + A_tok floor 0.7 + lambda=1`，ATE RMSE 为 41.5765 m，只比 unity replay baseline 41.6193 m 好 0.0428 m。这个提升只有 4.28 cm，不能被认为是显著超过 LoGeR。更重要的是，当降低 token floor 或恢复 chunk-level budget 时，ATE 反而变差。`NoFloor` 版本虽然把 rotation RMSE 从 8.9508 deg 降到 7.8645 deg，但 ATE RMSE 恶化到 43.5147 m。这个现象说明当前 prior 确实在影响 TTT memory，而且能压掉一部分旋转噪声；但它同时压掉了对平移、尺度或轨迹连续性有用的 token。

因此，当前被证伪的假设是：

$$
\text{只要少写动态/不确定区域，KITTI 01 ATE 就会显著下降。}
$$

当前实验更支持下面这个判断：

$$
\text{LoGeR 在 KITTI 01 上需要稳定写入；当前 prior 的 token ranking 不够准，强 suppression 会误伤 translation-useful tokens。}
$$

所以下一阶段不应该继续大范围扫 `lambda_min`、`lambda_max`、`a_token_floor`、`lambda_d`、`lambda_u` 这些全局或半全局旋钮。新的实验主线应该改成：先确认空间变化的 $A_{tok}$ 是否真的对齐到正确 token；然后把 write control 从 $[0,1]$ 的抑制型 gate 改成保持 LoGeR 原生有效写入量的 token 重分配；最后再评估 patch risk pooling 和更可靠的 static/dynamic proxy。

---

## 2. 下一阶段的中心假设

新的中心假设不是“写得越少越好”，而是：

$$
\text{保持 LoGeR 原生总写入强度，同时让稳定结构 token 更主导、风险 token 更少主导，才能改善 KITTI 01。}
$$

这里“总写入强度”不能简单理解为 $\operatorname{Mean}(A_{tok})$。LoGeR 的 TTT write 本来就有原生 token-wise update coefficient，记为 $\eta_i$。如果我们只保证 $\operatorname{Mean}(p_i)=1$，但把 $\eta_i$ 高的关键 token 压低了，有效写入仍然会下降。因此真正需要保持的是：

$$
M_\eta =
\frac{\sum_i \eta_i p_i}{\sum_i \eta_i + \epsilon}
\approx 1.
$$

其中 $p_i$ 是新的 token write multiplier。当前 suppressive policy 基本是：

$$
p_i=A_{tok,i}, \qquad p_i\in[0,1].
$$

下一阶段要改成：

$$
p_i\in[p_{min},p_{max}], \qquad p_{min}<1<p_{max},
$$

并且通过 $\eta$-weighted normalization 保证 $M_\eta\approx1$。这样系统不再主要删除信息，而是在 native write-through 附近重新分配 token 贡献。

这个中心假设带出三个必须检验的子假设。

第一，variable prior token alignment 必须成立。All-one replay parity 只能证明 $p_i=1$ 时 replay 等价，不能证明空间变化的 $p_i$ 打到了正确 token。如果 `A_tok[:l]` 这个截取假设错了，强 prior 越强，错位破坏越大。

第二，当前 geometry cue 有弱信号，但不能直接用作 hard gate。它应该先作为 ranking score，用于围绕 1 的重分配，而不是把 token 压到很低。

第三，patch 内风险聚合和更可靠的 reprojection consistency 可能提高 ranking，但只有在前两个假设成立后才值得大改。否则无法区分失败来自 token 错位、policy 错误，还是 cue 本身不准。

---

## 3. 实验总流程

下一阶段分四个连续阶段。每个阶段都有明确的进入条件和停止条件。

第一阶段是 variable prior alignment gate。它不是为了提高 ATE，而是为了确认 $A_{tok}$ 的 token 类型、帧顺序、patch 顺序真的和 TTT cache 中的 token 一致。如果这一阶段失败，后续所有模型实验都暂停，优先修 Stage E 的 token indexing。

第二阶段是 $\eta$-weighted mean-preserving reweighting。这个阶段使用当前最好的 geometry cue 配置，即 `G3 + F8 calibrated soft-or + k_intra=5`，但不再使用 suppressive floor gate，而是把 geometry score 转成围绕 1 的 multiplier。这个阶段的目标是验证“保持有效写入量，只改变 token 主导权”是否能消除当前 rotation/translation trade-off。

第三阶段是 patch risk ranking。这个阶段不把 risk pooling 直接变成 $[0,1]$ gate，而是把它作为 ranking score 输入第二阶段的 mean-preserving policy。它要验证的是：patch 内高风险像素的 quantile pooling 是否能提升 token ranking，而不是验证“更强压制是否有效”。

第四阶段是 diagnostic-driven proxy upgrade。如果第二、第三阶段都无法给出明显收益，则说明当前 static/dynamic proxy 的质量不足，需要把 Stage B 从 same-pixel world-space residual 升级为真正的 reprojection consistency，并更严格地区分 dynamic、occlusion 和 uncertainty。

这四个阶段的执行顺序不能打乱。尤其不要在 variable prior alignment 没有通过之前继续解释 GSL-WC 的 ATE 结果；也不要在 mean-preserving policy 没跑通之前直接加重 risk pooling，因为这很可能只是把当前的“误伤有用 token”放大。

---

## 4. Phase 0：Variable Prior Alignment Gate

### 4.1 实验目的

当前已经证明 all-one replay 和 native LoGeR 接近。但 all-one prior 无法暴露 token 对齐问题，因为无论 TTT cache 里取的是完整 token、patch-only token、某个窗口 token、还是重排后的 token，全 1 prior 都不会出错。

当前 Stage E 中有如下高风险假设：

```python
if A_tok.shape[0] >= l:
    prior_flat = A_tok[:l]
```

这个逻辑假设 TTT cache 的长度 $l$ 对应 `A_tok` 的前 $l$ 个 token。如果真实 TTT layer 只接收 patch tokens，或者 special tokens 和 patch tokens 交替排列，或者内部 windowing 重排过 token，那么 variable prior 会错位。当前 floor 越低 ATE 越差的现象，既可能说明 ranking 不准，也可能说明 ranking 被打到了错误 token。

Phase 0 的目标就是把这两种可能分开。

### 4.2 需要增加的 debug 能力

建议在 `run_pipeline_abc.py` 和 `TTTWriteController` 中增加一个参数：

```bash
--debug_prior_mode none|patch_only|special_only|frame_ramp|reverse_frame_ramp|checkerboard|roll
```

同时增加每个 chunk、每个 TTT layer 的 prior debug 输出，至少保存到 JSONL：

```text
chunk_idx
start_frame
end_frame
layer_idx
cache_l
L_tok
num_patch_tokens_in_A_tok
num_special_tokens_in_A_tok
mean_prior_flat
min_prior_flat
max_prior_flat
std_prior_flat
first_20_prior_values
first_20_token_type_if_available
mean_prior_patch_expected
mean_prior_special_expected
```

如果现有 cache 无法知道 `prior_flat` 对应的 token type，就要先记录 `cache_l` 和 `L_tok` 的关系。如果发现 `cache_l` 明显不是完整 `L_tok`，就不能再用 `A_tok[:l]` 作为默认对齐方式。长期正确做法是在 Stage A 构建 write cache 时保存每个 TTT layer 实际使用的 token index：

$$
I^{(l)}_{cache}\in\{0,\dots,L_{tok}-1\}^{l}.
$$

Stage E 应该使用 gather：

$$
prior\_flat = A_{tok}[I^{(l)}_{cache}],
$$

而不是使用前缀截取。

### 4.3 实验 A：patch-only 与 special-only prior

先在 64 或 128 帧 prefix 上跑，不需要 full sequence。构造两个 synthetic prior。

第一个 prior 只压 patch tokens：

$$
p_{patch}=0.7, \qquad p_{special}=1.0.
$$

第二个 prior 只压 special tokens：

$$
p_{patch}=1.0, \qquad p_{special}=0.7.
$$

实验命令保持当前 unity replay 设置，只把 write mode 改成 semantic synthetic prior。推荐公共设置如下：

```bash
CUDA_VISIBLE_DEVICES=1 python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --ttt_write_mode semantic \
  --stage_c_mode none \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --window_size 32 \
  --overlap_size 3 \
  --reset_every 5 \
  --end_frame 128 \
  --lambda_min 1.0 \
  --lambda_max 1.0 \
  --debug_prior_mode patch_only \
  --output_txt results/kitti01_gslwc/phase0_alignment/patch_only_128/01.txt \
  --output_pt results/kitti01_gslwc/phase0_alignment/patch_only_128/01.pt \
  > results/kitti01_gslwc/phase0_alignment/patch_only_128/01.log 2>&1
```

判断标准不是单看 ATE，而是看行为是否符合 token 类型。Patch-only suppression 应该主要影响几何细节和轨迹；special-only suppression 如果明显影响 translation，说明 special / role / register tokens 对全局轨迹很关键，后续主模型里应固定 $p_{special}=1$。如果 patch-only 和 special-only 的 `prior_flat` 统计几乎相同，或者两者 trajectory 影响异常接近，就说明 token type 对齐存在问题。

通过标准：

$$
\operatorname{Mean}(prior\_flat | debug=patch\_only)
$$

应该和构造的 patch/special 比例一致，而不是总是接近某个固定值。若 cache 中确实只有 patch tokens，则 patch-only 的 `prior_flat` 均值应接近 0.7，special-only 的 `prior_flat` 均值应接近 1.0。若 cache 中包含所有 tokens，则均值应与 patch/special 数量比例一致。

失败处理：如果统计不符合预期，不再跑 Phase 1，先修改 Stage A cache，显式保存 `cache_token_indices`。

### 4.4 实验 B：frame ramp 与 reverse frame ramp

构造按 chunk 内帧号变化的 prior：

$$
p(t)=0.7+0.3\frac{t}{T-1},
$$

以及反向 prior：

$$
p(t)=1.0-0.3\frac{t}{T-1}.
$$

这个实验验证帧维度是否对齐。如果 TTT cache 中 token 顺序和 `PatchMeta` 的帧顺序一致，`prior_flat` 应该呈现清晰的 ramp。forward ramp 和 reverse ramp 的 `first_20_prior_values`、chunk 内均值分布、trajectory 影响应该有可解释差异。

通过标准：每个 chunk 的 patch-token prior 应该能按 frame index 恢复出单调趋势。如果 `prior_flat` 的统计完全没有 ramp 特征，说明 Stage E 没有正确按帧使用 $A_{tok}$。如果 forward 和 reverse 的结果几乎完全相同，也要怀疑 prior 没有按预期作用到帧级 token。

### 4.5 实验 C：roll prior

使用当前 G3/F8/k5 生成的 geometry score，构造原始 prior 和 circular shift prior：

$$
p'_i = p_{(i+\Delta)\bmod L}.
$$

推荐在 128 帧上测试 $\Delta=L_{patch}/8$ 和 $\Delta=L_{patch}/4$。如果 shifted prior 和 unshifted prior 的行为几乎没有差异，说明当前空间 ranking 可能没有被有效利用；如果 shifted prior 反而更好，说明原始 token 对齐或 score 对齐高度可疑。

Phase 0 的最终 gate 是：synthetic prior 的 `prior_flat` 统计必须和构造方式一致，frame ramp 必须有可解释的时序形状，roll prior 不能系统性优于 unshifted prior。只有满足这些条件，才能进入 Phase 1。

---

## 5. Phase 1：$\eta$-weighted Mean-Preserving Reweighting

### 5.1 实验目的

当前 policy 把 $A_{tok}$ 作为 $[0,1]$ gate 使用，本质上是在减少写入。KITTI 01 的 floor sweep 已经说明，强减少写入会改善 rotation 但伤害 ATE。Phase 1 要验证新的假设：

$$
\text{不减少 LoGeR 的有效总写入，只改变 token 的相对主导权，能够保留 translation adaptation 并继承 rotation gain。}
$$

### 5.2 从 geometry score 到 multiplier

第一版不直接用 semantic。Stage C 继续使用 `--stage_c_mode none`，让实验专注于 geometry write control。

使用当前最好方向作为 base score：`G3 + F8 calibrated_soft_or + k_intra=5`。需要注意，Phase 1 不应该使用 `a_token_floor=0.7` 后的平坦 $A_{tok}$ 做 ranking，因为 floor 会抹掉弱信号。应该使用 floor 之前的 geometry score，例如 `G_write_geo_patch` 或未 floor 的 `A_patch_flat`。

为了减少绝对标定误差，建议先把 base score 转成 rank percentile：

$$
r_i=\operatorname{PercentileRank}(s_i)\in[0,1],
$$

然后转成中心化 score：

$$
\hat s_i = 2r_i-1.
$$

这样 $\hat s_i\in[-1,1]$，表示该 token 在当前 chunk 内相对更可写还是更不可写。再构造初始 multiplier：

$$
\tilde p_i = \operatorname{clip}(1+\alpha \hat s_i, p_{min}, p_{max}).
$$

第一版固定 special tokens：

$$
p_{special}=1.0.
$$

原因是 KITTI 01 当前主要坏在 translation / global trajectory，special、role、register tokens 可能承载窗口级全局信息。除非 Phase 0 明确证明 special suppression 是安全的，否则主模型先不动 special tokens。

### 5.3 $\eta$-weighted normalization

Stage E replay 中有三路 lr：`lr0/lr1/lr2`。为了不改变 LoGeR 原生每个分支的有效总写入，建议对每个 layer、每个 head-batch、每个 branch 分别计算：

$$
M_{\eta}^{(r)} =
\frac{\sum_i \eta_i^{(r)}\tilde p_i}{\sum_i \eta_i^{(r)}+\epsilon}.
$$

然后在 replay 时使用：

$$
p_i^{final}=\tilde p_i,
$$

并把 branch lr 乘上归一化系数：

$$
\eta_i^{(r,new)}=
\eta_i^{(r)}\cdot\frac{1}{M_\eta^{(r)}+\epsilon}.
$$

如果当前 `fast_weight_replay_update` 只能接收一个 shared `token_prior`，这个做法仍然可行：`token_prior` 传入 $\tilde p_i$，而 `lr0/lr1/lr2` 分别乘以各自的 $1/M_\eta^{(r)}$。这样有效写入满足：

$$
\frac{\sum_i \eta_i^{(r,new)}\tilde p_i}{\sum_i \eta_i^{(r)}+\epsilon}
\approx 1.
$$

这一步非常关键。若只做 $\operatorname{Mean}(p_i)=1$，仍然可能压低 LoGeR 原生高 $\eta_i$ 的关键 token。

### 5.4 Phase 1 实验设置

所有 Phase 1 实验都固定：

```text
--ttt_write_mode semantic
--stage_c_mode none
--dyn_fusion_mode calibrated_soft_or
--implicit_weight 0.50
--implicit_gate_floor 0.25
--k_intra 5
--sigma_pt 0.25
--lambda_s 1.2
--lambda_a 0.8
--lambda_d 1.2
--lambda_o 0.3
--lambda_u 0.3
--lambda_min 1.0
--lambda_max 1.0
--a_min_special 1.0
--chunk_size 32
--chunk_overlap 3
--window_size 32
--overlap_size 3
--reset_every 5
```

第一组实验如下：

| Run ID | Policy | $\alpha$ | Range | Special | 目的 |
|---|---|---:|---|---|---|
| M0 | unity replay | n/a | n/a | native | 固定 baseline，ATE 41.6193 m |
| M1 | 当前最好 G3/F8/k5/floor07 | n/a | $[0.7,1.0]$ | current | 当前 reference，ATE 41.5765 m |
| MP-01 | $\eta$-mean-preserving | 0.1 | $[0.8,1.2]$ | 1.0 | 最温和重分配，验证是否安全 |
| MP-02 | $\eta$-mean-preserving | 0.2 | $[0.7,1.3]$ | 1.0 | 主候选配置 |
| MP-03 | $\eta$-mean-preserving | 0.4 | $[0.7,1.4]$ | 1.0 | 强一点的重分配，测试上限 |

推荐先跑 128 帧 smoke，确认没有 OOM、没有明显 trajectory 崩坏、每层 $M_\eta$ 在合理范围内，再跑 full KITTI 01。

### 5.5 判断标准

Phase 1 的最低通过标准是 full KITTI 01 不劣于当前最好结果：

$$
ATE_{MP} \le 41.5765\text{ m}.
$$

但这只是最低标准。真正认为方向有效，需要满足：

$$
ATE_{MP} < 41.2\text{ m},
$$

同时 rotation RMSE 不显著差于 unity replay。这里设置 41.2 m 是因为当前 41.5765 m 的提升太小，无法支持“模型显著优于 LoGeR”的结论。若 MP 系列只能得到 41.55 m 附近，则说明 policy 变安全了，但 ranking 仍然很弱。

还要检查每个 layer 的有效写入比：

$$
M_\eta^{(r,l)}\in[0.98,1.02]
$$

作为理想范围。若某些 layer/head 长期偏离到 0.9 或 1.1 之外，说明 mean-preserving 实现不稳，不能解释最终 ATE。

如果 MP-01 安全但没有收益，MP-02 是主判断。若 MP-02 明显优于 M1，说明 suppressive gate 是主要瓶颈，可以继续围绕 reweighting 优化。若 MP-02 与 M1 接近但 rotation 更好，说明 policy 改对了但 ranking 不够强。若 MP-02 和 MP-03 都变差，先回到 Phase 0 检查 token 对齐和 $\eta$ normalization，再考虑 cue 质量。

---

## 6. Phase 2：Patch Risk Ranking，而不是 Patch Risk Gate

### 6.1 实验目的

当前 patch prior 大概率使用 mean pooling，这会把 patch 内的风险区域平均掉。KITTI 01 中，一个 patch 可能同时包含道路、车辆边缘、遮挡边界、建筑边缘和远处纹理。Mean pooling 的问题是：少量高风险像素可能被稀释，而低纹理或不确定区域又可能把整个 patch 过度拉低。

但是，当前实验已经说明强 suppression 会伤害 ATE。所以 Phase 2 不把 risk pooling 直接输出为 $[0,1]$ gate，而是把它作为 Phase 1 的 ranking score。也就是说，risk pooling 只决定谁在 mean-preserving reweighting 中略高或略低，不决定整体少写。

### 6.2 新的 patch score

对 patch $\Pi_i$，定义正向稳定证据：

$$
P_i =
\operatorname{Mean}_{u\in\Pi_i}
\left(0.5C_{stat}(u)+0.5C_{anchor}(u)\right).
$$

定义风险证据：

$$
R_i(q)=
Q_q{}_{u\in\Pi_i}
\left(
0.6C_{dyn}(u)+0.2C_{occ}(u)+0.2C_{unc}(u)
\right).
$$

其中 $Q_q$ 表示 patch 内像素风险的分位数，例如 $q=0.8$ 或 $q=0.9$。最终 ranking score 为：

$$
s_i = P_i - \beta R_i(q).
$$

再把 $s_i$ 送入 Phase 1 的 percentile-rank 和 $\eta$-weighted mean-preserving policy。

这个定义有两个好处。第一，它让 patch 内高风险像素有机会影响 ranking。第二，它不把风险直接解释为“少写整个 chunk”，避免重演 no-floor 实验中的 translation collapse。

### 6.3 Phase 2 实验设置

以 MP-02 作为默认 policy。如果 MP-02 在 Phase 1 中明显失败，则不要跑 Phase 2，直接进入 Phase 3/4 诊断和 Stage B proxy 改造。

| Run ID | Positive pooling | Risk pooling | $\beta$ | Policy | 目的 |
|---|---|---|---:|---|---|
| RP-01 | mean | q80 | 0.5 | MP-02 | 温和 risk ranking |
| RP-02 | mean | q90 | 0.5 | MP-02 | 主候选 risk pooling |
| RP-03 | mean | q90 | 0.8 | MP-02 | 更强 risk ranking |
| RP-04 | mean | q95 | 0.5 | MP-02 | 检查是否对少量风险过敏 |

### 6.4 判断标准

如果 RP-02 相对 MP-02 至少改善 0.2 m，并且没有明显恶化 rotation，则 patch risk ranking 是有效方向，可以继续细化 $R_i$ 的通道权重。

如果 RP 系列只改善 rotation，但 ATE 变差，说明 risk pooling 抓到的仍然是“会影响旋转的边界/不稳定区域”，但同时误伤了 translation-useful geometry。此时不要继续提高 $q$ 或 $\beta$，应该进入 Stage B v3，把 same-pixel residual 换成 reprojection residual。

如果 RP 系列完全没有变化，说明当前 prior 对 token ranking 的影响仍然太弱，或者 token 对齐/normalization 仍有问题。此时回查 Phase 0 和 $M_\eta$ debug，不要直接得出“risk pooling 无效”的结论。

---

## 7. Phase 3：分段诊断，用数据决定下一步改什么

### 7.1 实验目的

当前只看 full KITTI 01 的最终 ATE，信息量太低。一个配置可能前 300 帧有用，后 800 帧伤害；也可能在转弯、动态车辆密集、低纹理道路、高速直行等场景中表现完全不同。分段诊断的目的不是替代最终 benchmark，而是找出 prior 帮助和伤害 LoGeR 的具体场景。

### 7.2 需要新增的 per-chunk summary

建议在 full run 中保存一个 `chunk_stats.jsonl`，每个 chunk 一行，避免保存巨大的全量 map。字段包括：

```text
run_id
chunk_idx
start_frame
end_frame
mean_C_stat
mean_C_dyn
mean_C_occ
mean_C_unc
mean_C_anchor
mean_G_write_geo
mean_A_tok
std_A_tok
q10_A_tok
q50_A_tok
q90_A_tok
mean_patch_score
mean_risk_score
B_chunk_geo
lambda_write
mean_M_eta_layer0
mean_M_eta_all_layers
max_abs_M_eta_minus_1
```

如果使用 semantic reference，还要记录：

```text
num_masklets
coverage_ratio
movable_coverage_ratio
structure_coverage_ratio
low_value_stuff_coverage_ratio
mean_r_mask
```

现有 `run_pipeline_abc.py` 似乎更偏向保存第一段结果和最终轨迹。下一步应优先增加轻量 summary 输出，而不是只依赖终端日志。长 KITTI run 建议全部使用：

```bash
> 01.log 2>&1
```

不要用 `tee` 长时间流式输出，避免 stdout back-pressure。

### 7.3 Segment metric

把 KITTI 01 划分成长度 100 或 200 帧的 segment。推荐同时做两种：

$$
\mathcal{S}_{200,100}=\{[0,200),[100,300),[200,400),\dots\}
$$

和：

$$
\mathcal{S}_{100,50}=\{[0,100),[50,150),[100,200),\dots\}.
$$

对每个 segment 计算：

$$
ATE_s,\quad RPE^t_s,\quad RPE^r_s.
$$

分段 ATE 可以使用 segment 内 Sim(3) 或 SE(3) 对齐。它不需要和 full benchmark 的绝对值完全一致，因为它的用途是比较同一 segment 上不同 run 的差异。核心统计是：

$$
\Delta ATE_s = ATE_s(candidate)-ATE_s(unity).
$$

同时记录：

$$
\Delta RPE^t_s,
\qquad
\Delta RPE^r_s.
$$

### 7.4 Lagged correlation

TTT write 是写给未来 chunk 用的，所以不能只比较 chunk $m$ 的 prior 和 chunk $m$ 的误差。建议计算 lagged correlation：

$$
\operatorname{Corr}\left(\operatorname{meanRisk}_m, \Delta ATE_{m+k}\right),
\qquad k\in\{0,1,2,3\}.
$$

如果某个配置在 $k=1$ 或 $k=2$ 时出现明显负相关，说明当前 chunk 的 write policy 对后续轨迹有帮助。如果出现正相关，说明高风险 chunk 的处理反而伤害未来 memory。

### 7.5 如何根据分段诊断决策

如果 MP/RP 在转弯或动态车辆多的 segment 上变好，但高速直行、低纹理道路 segment 变差，说明 risk cue 对动态干扰有价值，但 uncertainty / edge risk 正在误伤静态结构。下一步应降低 $C_{unc}$ 权重，并引入 reprojection consistency 区分自运动视差和真实动态。

如果 MP/RP 主要改善 rotation 但几乎所有 segment 的 translation 都变差，说明 current policy 仍在有效减少 LoGeR 的 translation adaptation。此时检查 $M_\eta$ 是否真的接近 1，并确认 special tokens 是否固定为 1。

如果只有少数 segment 出现大幅 collapse，优先检查这些 segment 的 `A_tok q10/q50/q90`、`mean_C_unc`、`mean_C_dyn` 和 chunk 边界 reset 情况。不要因为 full ATE 差就直接否定整个方向。

如果所有 segment 都没有规律，优先怀疑 token 对齐或 prior 影响太弱；此时回到 Phase 0 的 roll prior 和 frame ramp 检查。

---

## 8. Phase 4：升级 Dynamic Cue Extractor 的 static/dynamic proxy

### 8.1 为什么要升级 Stage B

当前 Stage B Phase 1 实现主要是 same-pixel world-space comparison 加 depth-ordering occlusion check。这是一个工程上快速的近似，但在 KITTI 01 这种前向运动场景中，它容易把相机自运动带来的视差、深度边缘、道路/建筑边界误判为 dynamic 或 risk。

理想的静态一致性应该不是比较同一像素在不同帧的世界坐标，而是把当前世界点投影到支持帧，并在支持帧采样对应位置的世界点。

### 8.2 Reprojection consistency

对当前帧 $t$ 的像素 $u$，有世界点：

$$
X_t(u)=T_{w\leftarrow c,t}P_t(u).
$$

投影到支持帧 $s$：

$$
\tilde u_{t\rightarrow s}=\pi\left(T_{c\leftarrow w,s}X_t(u)\right).
$$

在支持帧上 bilinear sample 世界点：

$$
\tilde X_s=X_s(\tilde u_{t\rightarrow s}).
$$

点位残差改为：

$$
r_{pt}(t,s,u)=
\frac{\left\|X_t(u)-\tilde X_s\right\|_2}
{\epsilon+\left\|X_t(u)\right\|_2}.
$$

这比 same-pixel residual 更接近“静态世界是否解释得通”。如果这个改动能明显减少 road/building/guardrail 的 false risk，就有机会提升 token ranking。

### 8.3 更严格地区分 dynamic、occlusion、uncertainty

当前实验中 G3 比 G2 好，一个重要线索是降低 uncertainty penalty 后 ATE 更好。这说明 KITTI 01 上不能把不确定区域简单当成低写入区域。Stage B v3 应定义更保守的 dynamic：

$$
C_{dyn}^{nonocc}=C_{dyn}\cdot(1-C_{occ})\cdot(1-C_{unc}).
$$

risk score 改为：

$$
R = w_d C_{dyn}^{nonocc}+w_o C_{occ}+w_u C_{unc},
$$

并从较小的 uncertainty 权重开始：

$$
w_d=0.6,\qquad w_o=0.25,\qquad w_u=0.15.
$$

这个定义的思想是：只有当 residual 高、不是遮挡、也不是纯几何不确定时，才更强地解释为 dynamic。遮挡和不确定仍然可以影响 ranking，但不应该像真实 dynamic 一样强烈压制。

### 8.4 可选 optical flow / epipolar consistency

如果 reprojection consistency 仍然不能带来明显提升，可以加入 optical flow 或 epipolar residual 作为外部 motion proxy。定义 observed flow：

$$
F^{obs}_{t\rightarrow s}(u),
$$

由静态几何预测的 flow 为：

$$
F^{geo}_{t\rightarrow s}(u)=\tilde u_{t\rightarrow s}-u.
$$

flow residual：

$$
r_{flow}(t,s,u)=\left\|F^{obs}_{t\rightarrow s}(u)-F^{geo}_{t\rightarrow s}(u)\right\|_2.
$$

更可靠的 moving-object proxy 应该同时满足：

$$
r_{pt}\text{ high},\qquad r_{flow}\text{ high},\qquad C_{occ}\text{ low}.
$$

这样能减少把遮挡边界、低纹理道路或纯自运动视差误当作动态物体的概率。

### 8.5 Stage B v3 实验

只有当 Phase 0 通过、Phase 1/2 没有明显收益时，才进入下面实验：

| Run ID | Stage B 改动 | Policy | 判断 |
|---|---|---|---|
| B3-01 | reprojection residual 替换 same-pixel residual | MP-02 | 看是否减少 translation 损伤 |
| B3-02 | reprojection + nonocc dynamic | MP-02 | 看 dynamic/occ/unc 分离是否有效 |
| B3-03 | B3-02 + q90 risk ranking | MP-02 | 在更可靠 proxy 下重测 patch risk |

如果 B3-01 比 MP-02 明显改善，说明当前主要瓶颈是 same-pixel residual 误报。若 B3-02 再进一步改善，说明当前确实需要更干净地区分 dynamic、occlusion 和 uncertainty。若 B3-03 又恶化，说明即使 proxy 改善，patch quantile risk 仍然过敏，应该回到 mean-preserving geometry score。

---

## 9. Video Masklet Front-end 的使用策略

### 9.1 为什么暂时不让语义主导

当前 Video Masklet Front-end 的 thing 漏检、tracking 失败和 stuff 边界粗糙问题还没有解决。因此，语义结果不能作为主导 write policy。尤其在 KITTI 01 中，车辆既可能是 moving object，也可能是 parked landmark；如果只凭 semantic label 把所有 car 压低，会误伤大量对位姿有用的静态结构。

语义在下一阶段只作为 weak reference，用于保护明显结构类 token，或者轻微限制高置信动态物体。它不应该覆盖 geometry eligibility。

### 9.2 KITTI prompt bank 调整

当前 prompt 更偏 indoor，需要改成 KITTI 场景。推荐第一版 prompt bank：

```text
THING_PROMPTS = [
  "car", "van", "truck", "bus", "trailer", "train",
  "person", "pedestrian", "rider", "cyclist",
  "bicycle", "motorcycle"
]

STRUCTURE_PROMPTS = [
  "road", "lane marking", "sidewalk", "building", "building facade",
  "guardrail", "fence", "barrier", "pole", "traffic sign",
  "traffic light", "bridge"
]

LOW_VALUE_STUFF_PROMPTS = [
  "sky", "tree", "vegetation", "grass", "bush"
]
```

语义组建议映射为：

```text
STRUCTURE_ANCHOR:
  road, lane marking, sidewalk, building, building facade,
  guardrail, fence, barrier, pole, traffic sign, traffic light, bridge

MOVABLE_THING:
  car, van, truck, bus, trailer, train,
  person, pedestrian, rider, cyclist, bicycle, motorcycle

LOW_VALUE_STUFF:
  sky, tree, vegetation, grass, bush
```

### 9.3 语义只做 floor/cap protection

结构类只做 floor protection。如果某个 token 被高质量 structure mask 覆盖，并且 geometry score 没有极端风险，则：

$$
p_i\leftarrow\max(p_i,0.9).
$$

movable thing 只做 mild cap，而且必须同时满足高质量 mask 和高 dynamic proxy：

$$
Q_{mask}>0.7,\qquad C_{dyn}^{patch}>0.6.
$$

此时：

$$
p_i\leftarrow\min(p_i,0.85).
$$

如果只是 semantic label 是 car，但 geometry 上稳定，不应该强压。因为 parked cars 在 KITTI 中可能是有用 landmark。

low-value stuff，例如 sky / vegetation，只做轻微 cap：

$$
p_i\leftarrow\min(p_i,0.95),
$$

且只在 mask quality 足够高时启用。由于当前 stuff 边界粗糙，不能让 sky/tree mask 大面积覆盖后直接强压 token。

### 9.4 语义实验的进入条件

语义实验不应该在 Phase 1 之前启动。只有当 MP 或 RP 系列已经证明 geometry-only reweighting 至少不伤害 LoGeR，并且最好能达到 ATE < 41.2 m，才加入 semantic floor/cap。否则语义会增加噪声源，让失败归因更困难。

第一组语义实验建议：

| Run ID | Base | Semantic use | 判断 |
|---|---|---|---|
| SEM-01 | MP-02 | structure floor only | 看 road/building/guardrail 保护是否减少 translation 损伤 |
| SEM-02 | MP-02 | structure floor + movable dynamic cap | 看动态物体轻微 cap 是否有用 |
| SEM-03 | RP-02 | structure floor + movable dynamic cap | 在 risk ranking 下测试语义补偿 |

若 SEM-01 改善而 SEM-02 变差，说明结构保护有价值，但 movable semantic 仍不可靠。若 SEM-02 改善，说明 geometry+semantic 双条件 cap 有用，可以考虑扩大 KITTI prompt。若 SEM-01/02 都无收益，保持 geometry-only 主路径。

---

## 10. 推荐执行顺序和命名

为了避免再次出现无效 sweep 混入模型选择，所有新实验目录建议按 phase 命名：

```text
results/kitti01_gslwc/phase0_alignment/
results/kitti01_gslwc/phase1_mean_preserving/
results/kitti01_gslwc/phase2_patch_risk/
results/kitti01_gslwc/phase3_segment_diagnostics/
results/kitti01_gslwc/phase4_stageb_v3/
results/kitti01_gslwc/phase5_semantic_reference/
```

每个 run 必须保存：

```text
01.txt
01.pt
01.log
kitti_benchmark.log
chunk_stats.jsonl
config.yaml 或 args.txt
```

推荐每个 full run 的命名包含 policy、score、range 和 special 设置，例如：

```text
MP02_G3F8_k5_etaNorm_range07_13_special1_lam1
RP02_G3F8_k5_q90_beta05_etaNorm_range07_13_special1_lam1
B301_reproj_MP02_range07_13_special1_lam1
```

---

## 11. 最小实验矩阵

如果计算资源有限，下一阶段只跑下面这些实验即可。

首先做 Phase 0 的 128-frame debug：

| ID | Frames | 实验 | 通过后才能继续 |
|---|---:|---|---|
| D1 | 128 | patch-only prior | 验证 patch token prior 是否对齐 |
| D2 | 128 | special-only prior | 判断 special token 是否应固定为 1 |
| D3 | 128 | frame ramp / reverse ramp | 验证帧顺序对齐 |
| D4 | 128 | roll prior | 检查空间 prior 是否有效或错位 |

然后做 full KITTI 01 的 Phase 1：

| ID | 实验 | 预期 |
|---|---|---|
| M0 | unity replay baseline | reference，41.6193 m |
| M1 | 当前 G3/F8/k5/floor07 | reference，41.5765 m |
| MP-01 | $\eta$-mean-preserving, $\alpha=0.1$, $[0.8,1.2]$ | 应该非常安全 |
| MP-02 | $\eta$-mean-preserving, $\alpha=0.2$, $[0.7,1.3]$ | 主候选 |
| MP-03 | $\eta$-mean-preserving, $\alpha=0.4$, $[0.7,1.4]$ | 检查强重分配上限 |

如果 MP-02 不差，再跑 Phase 2：

| ID | 实验 | 预期 |
|---|---|---|
| RP-01 | q80 risk, $\beta=0.5$ | 温和风险 ranking |
| RP-02 | q90 risk, $\beta=0.5$ | 主候选 |
| RP-03 | q90 risk, $\beta=0.8$ | 判断风险权重是否过强 |

所有 full run 完成后，立刻做 Phase 3 分段诊断。不要等更多 sweep 完成后再分析，否则会继续堆积难以归因的 ATE 数字。

如果 MP/RP 都无法达到至少 0.2 m 级别提升，再进入 Phase 4：

| ID | 实验 | 目标 |
|---|---|---|
| B3-01 | reprojection residual + MP-02 | 验证 same-pixel residual 是否主要瓶颈 |
| B3-02 | reprojection + nonocc dynamic + MP-02 | 验证 cue 分离是否提升 ranking |
| B3-03 | B3-02 + q90 risk ranking | 在新 proxy 下重测 patch risk |

最后，只有当 geometry-only 主路径有明显收益后，才进入 semantic reference：

| ID | 实验 | 目标 |
|---|---|---|
| SEM-01 | KITTI prompts + structure floor | 保护 road/building 等稳定结构 |
| SEM-02 | SEM-01 + dynamic movable cap | 轻微限制高置信动态物体 |

---

## 12. 最终判断标准

下一阶段不要把“略微超过 41.6193 m”当作成功。当前最好已经达到 41.5765 m，但这只是 4.28 cm 的 gain。新的主模型至少需要满足下面三个层级的标准。

第一层，工程可信：

$$
\text{Phase 0 alignment gate pass},\qquad
M_\eta\in[0.98,1.02],
$$

并且 all-one replay 仍保持 unity parity。

第二层，模型有用：

$$
ATE < 41.2\text{ m},
$$

同时 rotation RMSE 不显著差于 unity replay。如果只得到 41.55 m 左右，说明方向可能安全，但还不能作为主模型。

第三层，值得扩展：

$$
ATE \le 40.5\text{ m}
$$

或者分段诊断显示大多数困难 segment 都有稳定收益，且没有少数 segment 大幅 collapse。达到这一层后，才值得继续调语义 prompt、扩展到 KITTI 00/02，或重新追求原计划中的 ATE < 33 m。

如果所有 mean-preserving 和 patch-risk 实验都无法超过 41.2 m，则结论不是“项目失败”，而是：当前 Dynamic Cue Extractor 的 static/dynamic proxy 不足以支撑 KITTI 01 的 token ranking。此时应停止 scalar sweep，集中改 Stage B v3 的 reprojection consistency 和 dynamic/occlusion/uncertainty 分离。

---

## 13. 本方案的核心变化

这份新方案和上一轮最大的区别是：它不再试图通过更强的 $A_{tok}$ floor、chunk budget 或 dynamic penalty 来找到一个幸运配置。上一轮已经说明，强 suppression 会改善 rotation 但伤害 ATE；安全配置又太接近 unity，只有厘米级收益。

新的主线是：

$$
\boxed{
\text{确认 variable prior 对齐}
\rightarrow
\eta\text{-weighted mean-preserving reweighting}
\rightarrow
\text{patch risk ranking}
\rightarrow
\text{分段诊断}
\rightarrow
\text{reprojection-based Stage B v3}
}
$$

这条路线直接针对当前实验暴露的真实瓶颈：不是整体写多写少，而是 token ranking 与 write policy 的耦合还不够正确。只有先把 write control 改成不破坏 LoGeR 原生有效写入的重分配，再去提高 ranking 质量，才有可能把当前 41.57 m 的 fragile gain 推到真正可见的改进区间。

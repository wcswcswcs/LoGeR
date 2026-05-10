# LoGeR TTT Cue Library v1：从 TTT 内部挖掘写入 Cue 的设计与实验计划

日期：2026-05-08  
对象：LoGeR / HMC Pipeline v2 / TTT fast-weight memory  
主开发集：KITTI Odometry Sequence 01  
当前固定 read 主线：`acl2.gg.qq.low.g2_3.past_only.headmean.robustq`，frame-attention `pair/all`，beta `4.75`  
当前固定安全协议：`probe_ttt_write`  
当前可复现参考：`B0_SWKS3 = 36.4161 / 6.6128`，`TTGRL_04 = 36.2957 / 6.6182`

---

## 0. 一句话结论

下一阶段不应该继续把 frame-attention cue `D_g` 当成所有 memory-control 的唯一来源。应该像之前系统挖掘 frame-attention cue 一样，建立一套 **TTT-native cue library**，从 LoGeR 的 TTT fast-weight update / apply / replay 机制内部直接挖掘：

```text
哪些 token 的 fast-weight update 会污染长期 memory；
哪些 token 只是当前 chunk 有用，但不应该长期 commit；
哪些 branch / layer 的 update direction 与 [200,300) 病灶强相关；
哪些 update 应该进入 long-term W，哪些只应该作为 short-term transient state。
```

TTSA3R 可以作为原则启发，但不能直接照搬。TTSA3R 的对象是显式 persistent state token，而 LoGeR 的 TTT 是 fast-weight matrix memory。对应地，LoGeR 里的 temporal cue 不是 state-token 差分，而是 fast-weight delta evolution；spatial/context cue 也不是 CUT3R cross-attention mask，而是 TTT pre-update prediction、update direction、apply readback、next-chunk influence 等量。

---

## 1. 背景与动机

### 1.1 当前已经成立的事实

当前实验已经证明，LoGeR 内部 frame-attention / global-query cue 是有效 read cue。主线已经从早期的 old-dyn family 推进到：

```text
D_g_locked = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path  = frame-attention pair/all
beta       = 4.75
```

v5 中 `pair/all` 把 ATE 从 v4 的 `38m` 级推到 `36m` 级，是当前最大收益来源。相比之下，TTT write-score、sparse write、explicit dyn veto、semantic scalar、TTGR 等写入侧尝试大多是 `0.01m-0.12m` 级别改善，且没有真正修掉 `[200,300)` 主病灶。

v7 的关键结果是：

```text
B0_SWKS3:
    ATE / Rot = 36.4161 / 6.6128
    [200,300) = 77.831

TTGRL_04:
    chunk5, w0, all-layer, gamma=0.025
    ATE / Rot = 36.2957 / 6.6182
    [200,300) = 77.644

freeze5:
    ATE = 38.9727
    [200,300) = 41.899

freeze56:
    ATE = 60.3998
    [200,300) = 26.102
```

这说明：

1. chunk5 / chunk6 附近的 TTT state 对 `[200,300)` 有强因果作用；
2. hard freeze 能救 `[200,300)`，但会毁掉全局 continuity；
3. 轻量 negative replay 能改善 overall ATE，但没有真正打到主病灶方向；
4. 当前 signed scalar prior / gamma / branch / layer 小扫已经进入平台。

### 1.2 为什么必须挖 TTT 自己的 cue

当前控制方式大多是：

```text
D_g 或 semantic prior -> token write multiplier -> TTT replay
```

但 `D_g` 的语义主要是 read-path harmful support：它告诉我们当前 frame attention 中哪些 patch 不适合作为 support。它不一定等价于 TTT write-path 的 harmful update source。

TTT 的问题更具体：

```text
某个 token 看起来动态，并不意味着它对 fast weights 的影响大；
某个 token D_g 不高，也可能通过 lr 或 branch gradient 造成巨大 fast-weight update；
少写某些 token 可能清 rotation，但损害 scale / continuity；
同一个 chunk 的 update 里可能混有 harmful direction 和 useful continuity direction。
```

因此，TTT write policy 不能只问：

```text
这个 token 是否动态？
```

而应该问：

```text
这个 token 对 W0/W1/W2 的更新贡献有多大？
这个贡献方向和 static / native / freeze-harmful direction 是否一致？
这个贡献写入后会怎样影响下一 chunk 的 TTT apply？
这个 contribution 应该进入 long-term W，还是 short-term transient W？
```

这就是 TTT Cue Library v1 的目标。

---

## 2. TTSA3R 的启发与不可照搬之处

### 2.1 TTSA3R 给出的原则

TTSA3R 的核心思想可以概括为两条：

1. **Temporal state evolution**：如果 state token 随时间变化小，说明已经稳定，应更多保留；如果变化大，说明需要更新或存在不稳定区域。
2. **Spatial observation-state alignment**：如果 observation 与历史 state 对齐且 feature 一致，应该保留；如果 attention 强但 feature 变化大，说明需要 refinement 或存在动态/错误更新。

这两个原则很有价值，因为它们不是单纯根据语义或 attention map 做 hard mask，而是同时考虑：

```text
状态自身是否稳定；
当前观测和历史状态是否一致；
更新是必要 refinement，还是错误覆盖。
```

### 2.2 为什么不能照搬

TTSA3R 的架构对象是 persistent state tokens，状态更新可以抽象为：

$$
S_t = \tilde S_t \odot M_t + S_{t-1} \odot (1-M_t)
$$

而 LoGeR 的 TTT 不是显式 state-token interpolation。LoGeR 的长期 memory 是 fast weights：

$$
W_{m+1} = W_m + \Delta W_m
$$

并且这个 $\Delta W_m$ 来自 TTT replay 的 fast-weight update。也就是说，LoGeR 中需要被分析的是：

```text
W0/W1/W2 的 fast-weight delta；
per-token replay contribution；
Muon / zeropower 后的 update direction；
fast-weight norm restoration 后的实际 committed direction；
当前 W 对 q/k/v 的 apply / prediction 行为。
```

所以，LoGeR 版 TTSA3R 不应该复制 TAUM / SCUM 的具体公式，而应该重建对应物：

```text
TAUM-LoGeR:
    fast-weight temporal evolution cue

SCUM-LoGeR:
    observation-memory prediction / update / readback alignment cue
```

---

## 3. LoGeR TTT 实现要点

### 3.1 Fast-weight function

当前 LoGeR 的 TTT fast-weight module 是 SwiGLU 形式。对输入 token $x$，fast-weight function 可以写作：

$$
f_W(x) = (\operatorname{SiLU}(xW_0) \odot xW_2)W_1
$$

其中三个 fast-weight branch 分别是：

```text
W0 / w0: gate branch
W1 / w1: output/value projection branch
W2 / w2: content branch
```

### 3.2 TTT update 的真实结构

在 `loger/models/ttt.py` 中，TTT update 会用当前 chunk 的 cached `k/v/lr0/lr1/lr2` 进行 replay。对一个 update segment，代码逻辑可以抽象为：

$$
g_i = k_i W_0
$$

$$
h_i^{pre} = k_i W_2
$$

$$
h_i = \operatorname{SiLU}(g_i) \odot h_i^{pre}
$$

然后三个 branch 的 pre-Muon update 聚合方向近似为：

$$
G_1 = \sum_i (h_i \cdot lr_{1,i})^T v_i
$$

$$
G_0 = \sum_i (k_i \cdot lr_{0,i})^T dgate_i
$$

$$
G_2 = \sum_i (k_i \cdot lr_{2,i})^T dhidden_i
$$

随后这些 $G_r$ 会经过 `zeropower_via_newtonschulz5`，再加到 fast weights：

$$
W_r' = W_r + \operatorname{ZeroPower}(G_r)
$$

最后还有 fast-weight norm restoration：

$$
W_r' \leftarrow \frac{W_r'}{\|W_r'\| + \epsilon}\|W_r\|
$$

### 3.3 为什么幅度控制经常不够

由于 TTT replay 包含：

```text
per-token lr scaling
outer-product aggregation
zeropower / Muon-like orthogonalization
fast-weight norm restoration
```

所以简单把 token prior 从 `1.0` 改成 `0.8` 或 `0.5`，未必能线性改变最终 committed direction。幅度信息会被 zeropower 和 norm restoration 部分折叠。

这解释了 v5-v7 的一个重要现象：

```text
少写 / sparse / focal / explicit veto 通常能改善 Rot、Yaw、FinalErr；
但 ATE 主病灶 [200,300) 基本不动。
```

因此 TTT cue 的核心不应该只是 `write strength`，而应该是：

```text
update direction
branch/layer conflict
future readback influence
long-term vs short-term lifetime
```

---

## 4. 实验整体目标

本阶段目标不是直接再跑一堆 TTT write variants，而是建立一个可解释的 TTT-native cue library，然后再用通过审计的 TTT cue 控制写入。

具体目标分成四层。

### 4.1 目标 A：建立 TTT passive trace 与 cue cache

对每个 chunk、TTT layer、branch、token，记录 TTT 内部可解释量：

```text
pre-update memory prediction error
apply readback mismatch
per-token update magnitude
per-token update direction alignment
fast-weight delta temporal evolution
pre/post zeropower direction变化
next-chunk readback influence
```

### 4.2 目标 B：解释 `[200,300)` 病灶

围绕 `B0_SWKS3`、`freeze5`、`freeze56`、`TTGRL_04`、`P3G_01`，回答：

```text
freeze5 删除了哪些 TTT direction？
TTGRL_04 修改了哪些 direction？
二者是否对齐？
chunk5/w0/all gamma=0.025 为什么改善 overall ATE，却没有修 [200,300)？
chunk6 为什么 hard freeze 会破坏入口 continuity？
```

### 4.3 目标 C：构造 TTT-native write cue

从被动审计中选出能解释病灶的 cue，用于定义：

```text
positive evidence:
    应该长期写入 W_long 的 token / direction

neutral evidence:
    当前有用，但不应强长期 commit 的 token / direction

negative evidence:
    应该反向抵消、弱写或只进入 transient memory 的 token / direction
```

### 4.4 目标 D：验证结构性写入策略

最终控制不再是单一 scalar multiplier，而是：

$$
G_{commit} = G_{pos} + \lambda G_{neu} - \gamma G_{neg}
$$

以及：

```text
W_long:
    只接收 positive + small neutral

W_transient:
    接收 short-term dynamic / negative correction
    apply K chunks 后衰减或丢弃
```

---

## 5. TTT Cue Library v1：候选 cue family

### 5.1 Family A：pre-update memory prediction error

对每个 token $i$，在 TTT update 之前，用当前 fast weights $W_m$ 预测 value：

$$
\hat v_i = f_{W_m}(k_i)
$$

定义：

$$
E_i^{pre} = \frac{\|\hat v_i - v_i\|_2}{\|v_i\|_2 + \epsilon}
$$

解释：

```text
E_pre 低：
    当前 fast weights 已经能解释这个 token，可能是 stable memory / preserve candidate。

E_pre 高：
    当前 observation 和 memory 不一致，可能需要更新，也可能是 harmful novelty。
```

这个 cue 不能单独决定正负，但它能区分：

```text
memory 已经知道的 token
vs
memory 解释不了的新 token
```

推荐组合解释：

```text
E_pre 高 + D_g 低 + semantic structure:
    positive refinement

E_pre 高 + D_g 高:
    risky / negative candidate

E_pre 低 + D_g 低:
    preserve / stable anchor

E_pre 低 + D_g 高:
    suspicious; 可能是 memory 已被污染，或 token 不应作为长期 anchor
```

### 5.2 Family B：per-token update magnitude

TTT 写入的真正影响取决于 token 对 $W_0/W_1/W_2$ 的 contribution，而不是 token map 的表面强度。

对每个 branch $r$，定义 per-token pre-Muon contribution $J_i^{(r)}$。

对于 $W_1$：

$$
J_i^{(1)} = (h_i \cdot lr_{1,i})^T v_i
$$

对于 $W_0$：

$$
J_i^{(0)} = (k_i \cdot lr_{0,i})^T dgate_i
$$

对于 $W_2$：

$$
J_i^{(2)} = (k_i \cdot lr_{2,i})^T dhidden_i
$$

定义归一化 update magnitude：

$$
U_i^{(r)} = \frac{\|J_i^{(r)}\|_F}{\operatorname{Mean}_j \|J_j^{(r)}\|_F + \epsilon}
$$

实现上不需要显式 materialize 每个 $J_i$ 矩阵。因为 outer product 的 Frobenius norm 可写为：

$$
\|a_i^T b_i\|_F = \|a_i\|_2 \|b_i\|_2
$$

因此可以高效记录：

```text
update_norm_w0
update_norm_w1
update_norm_w2
update_norm_total
branch_update_entropy
```

### 5.3 Family C：per-token update direction alignment

只是 magnitude 还不够。当前最核心的问题是 update direction。

对 chunk $m$、layer $l$、branch $r$，定义 native chunk update：

$$
\Delta W_{m,l}^{(r)} = W_{m+1,l}^{(r)} - W_{m,l}^{(r)}
$$

对 token contribution：

$$
A_{i,l}^{(r,native)} = \cos(\operatorname{vec}(J_{i,l}^{(r)}), \operatorname{vec}(\Delta W_{m,l}^{(r)}))
$$

更重要的是与 static / harmful direction 的对齐。

定义 static direction：

$$
\Delta W_{static,l}^{(r)} = \sum_{i \in \mathcal{S}_{static}} J_{i,l}^{(r)}
$$

其中 $\mathcal{S}_{static}$ 可先用 `low D_g + structure semantic + high confidence` 近似。

定义 static alignment：

$$
A_{i,l}^{(r,static)} = \cos(\operatorname{vec}(J_{i,l}^{(r)}), \operatorname{vec}(\Delta W_{static,l}^{(r)}))
$$

定义 static conflict：

$$
C_{i,l}^{(r,static)} = \operatorname{clip}(-A_{i,l}^{(r,static)}, 0, 1)
$$

如果 $C_i$ 高，说明该 token 的更新方向与 static update 相反或冲突，比单纯 high-D 更像 negative evidence。

实现优化：对于 outer product $J_i = a_i^T b_i$ 和聚合矩阵 $G$，点积可以写作：

$$
\langle J_i, G \rangle = a_i^T G b_i
$$

因此可以在不显式保存所有 $J_i$ 的情况下计算 token-matrix alignment。

### 5.4 Family D：fast-weight temporal evolution cue

这是 LoGeR 版 TAUM。

对每个 chunk、layer、branch，记录 update magnitude spike：

$$
M_{m,l}^{(r)} = \frac{\|\Delta W_{m,l}^{(r)}\|_F}{\operatorname{EMA}_{m'}\|\Delta W_{m',l}^{(r)}\|_F + \epsilon}
$$

记录 update direction flip：

$$
F_{m,l}^{(r)} = 1 - \cos(\operatorname{vec}(\Delta W_{m,l}^{(r)}), \operatorname{vec}(\Delta W_{m-1,l}^{(r)}))
$$

解释：

```text
M 高 + F 高：
    fast-weight update 幅度异常且方向突变，可能是 harmful overwrite / basin change。

M 高 + F 低：
    大幅但连续的 refinement，可能是有用 continuity。

M 低 + F 高：
    小幅但方向突变，可能是 subtle correction 或噪声。
```

对于当前问题，重点看：

```text
chunk5 / chunk6 的 w0/w1/w2 中，哪些 layer 出现 M/F spike？
这些 spike 是否和 freeze5 / freeze56 的效果对应？
TTGRL_04 是否作用在这些 spike 上？
```

### 5.5 Family E：TTT apply / readback cue

TTT 不只是写入，也会对当前 chunk apply：

$$
o_i = f_{W_m}(q_i)
$$

定义 apply magnitude：

$$
R_i^{apply} = \frac{\|f_{W_m}(q_i)\|_2}{\|x_i\|_2 + \epsilon}
$$

定义 apply-value mismatch：

$$
E_i^{apply} = \frac{\|f_{W_m}(q_i)-v_i\|_2}{\|v_i\|_2 + \epsilon}
$$

解释：

```text
apply magnitude 高 + mismatch 低 + D_g 低：
    memory readback 正确，可能是 stable anchor。

apply magnitude 高 + mismatch 高 + D_g 高：
    memory 正在强干预一个不稳定 token，可能是 harmful readback。

apply magnitude 低 + pre_error 高：
    memory 不知道当前 token，但也没有强错误读取，可能只是新区域。
```

这个 cue 回答的是：

```text
TTT memory 当前是否读对了？
```

而不是：

```text
当前 token 外观看起来是否动态？
```

### 5.6 Family F：next-chunk influence cue

这是最接近 TTT write 问题本质的 cue。

当前 chunk token $i$ 的 update 真正危险之处在于：它被写进 $W_{m+1}$ 后，会影响未来 chunk 的 TTT apply。

近似定义 token influence：

$$
I_i = \left\| f_{W + \epsilon J_i}(q^{next}) - f_W(q^{next}) \right\|_2
$$

进一步定义 influence alignment：

$$
A_i^{next} = \cos(f_{W+\epsilon J_i}(q^{next}) - f_W(q^{next}), \Delta o_{good}^{next})
$$

其中 $\Delta o_{good}^{next}$ 可以先用 static candidate / native-probe delta 近似。

实现上第一版不必对全 token 做昂贵 perturbation。可以采用三种近似：

```text
Option 1: layer/branch aggregate influence
    只比较 grouped update 对 next chunk apply 的影响。

Option 2: top-k token influence
    只对 update_norm 或 conflict top-k token 做 finite difference。

Option 3: first-order influence
    用 JVP / VJP 近似 token contribution 对 next apply 的线性影响。
```

这个 cue 的核心价值是回答：

```text
谁是真正 memory-propagating token？
```

### 5.7 Family G：zeropower / norm-restore distortion cue

由于 TTT replay 中存在 zeropower 和 norm restoration，pre-update direction 和最终 committed direction 可能不一致。

定义 pre-Muon aggregate：

$$
G_{pre,l}^{(r)} = \sum_i J_{i,l}^{(r)}
$$

定义 post-Muon delta：

$$
G_{post,l}^{(r)} = \operatorname{ZeroPower}(G_{pre,l}^{(r)})
$$

定义 distortion：

$$
Z_{l}^{(r)} = 1 - \cos(\operatorname{vec}(G_{pre,l}^{(r)}), \operatorname{vec}(G_{post,l}^{(r)}))
$$

定义 norm restoration ratio：

$$
N_{restore,l}^{(r)} = \frac{\|W_{new,l}^{(r)} - W_{old,l}^{(r)}\|_F}{\|G_{post,l}^{(r)}\|_F + \epsilon}
$$

如果某些 layer/branch 的 $Z$ 高，说明简单 token multiplier 很可能无法可靠控制最终方向；这些位置更适合做 explicit direction control，而不是 scalar gate。

---

## 6. 重点对照状态与实验对象

第一批 TTT Cue Library 不应全量扫所有 run，而应围绕已经有因果意义的状态进行 passive audit。

### 6.1 必须包含的参考 run

```text
B0_SWKS3:
    reproducible v5/v6 baseline

P3G_01:
    semantic + TTGR tiny best

TTGRL_04:
    chunk5 / w0 / all-layer / gamma=0.025 当前 v7 ATE best

freeze5:
    chunk5 hard freeze，大幅修 [200,300)，但全局回退

freeze56:
    chunk5+6 hard freeze，把 [200,300) 打到 26m，但全局崩坏
```

### 6.2 重点时间范围

必须重点导出：

```text
chunk5:
    主要 causal entry

chunk6:
    病灶入口 / continuity carrier

chunks 7-9:
    [200,300) 高误差传播区

chunk10:
    病灶出口
```

同时保留全序列 summary，但第一批可视化不要平均掉这些局部差异。

---

## 7. 核心假设与实验设计

## H1：TTT fast-weight temporal evolution 能定位 chunk5 / chunk6 的异常写入

### 假设

`[200,300)` 病灶不是随机轨迹评估错误，而是 chunks 5/6 附近某些 TTT layer/branch 的 fast-weight evolution 进入异常方向。该异常应体现在：

```text
delta norm spike
update direction flip
pre/post zeropower distortion
memory_state_rel_diff spike
```

### 实验

对 `B0_SWKS3` 的每个 chunk、layer、branch 记录：

$$
\|\Delta W_{m,l}^{(r)}\|_F
$$

$$
1-\cos(\Delta W_{m,l}^{(r)}, \Delta W_{m-1,l}^{(r)})
$$

$$
1-\cos(G_{pre,l}^{(r)}, G_{post,l}^{(r)})
$$

并和 `freeze5 / freeze56 / TTGRL_04` 的对应 delta 比较。

### 必须记录

```text
per_chunk_layer_branch_delta.csv
per_chunk_layer_branch_flip.csv
per_chunk_layer_branch_zeropower.csv
memory_state_rel_diff_by_chunk.csv
segment_error_by_chunk.csv
```

字段：

```text
run_id
chunk_id
frame_start
frame_end
layer_id
branch
pre_update_norm
post_update_norm
committed_delta_norm
delta_norm_ema_ratio
delta_cos_prev
pre_post_zeropower_cos
norm_restore_ratio
memory_state_rel_diff
segment_200_300_overlap
chunk_ate
```

### 成立标准

H1 通过条件：

```text
1. chunk5 或 chunk6 在某些 layer/branch 上是 delta_norm 或 delta_flip 的 top outlier；
2. outlier layer/branch 与 freeze5/freeze56 对 [200,300) 的因果效果一致；
3. TTGRL_04 的作用 scope 与 outlier 至少部分重合；
4. 这些指标比 D_g mass / semantic coverage 更能解释 chunk5 的特殊性。
```

如果 H1 不成立，说明当前可观测 TTT delta 还不够，需要加更底层 trace，而不是继续做 write control。

---

## H2：freeze5 删除的 direction 与 TTGRL_04 修改的 direction 不完全对齐

### 假设

`TTGRL_04` 能改善 overall ATE，但没有修 `[200,300)`，是因为它没有真正打到 freeze5 删除的 harmful direction，或者只打到其中很小一部分。

### 实验

定义 freeze5 removed direction：

$$
\Delta W_{removed,5,l}^{(r)} = \Delta W_{B0,5,l}^{(r)} - \Delta W_{freeze5,5,l}^{(r)}
$$

由于 freeze5 实际上丢弃 chunk5 commit，可近似为：

$$
\Delta W_{removed,5,l}^{(r)} \approx \Delta W_{B0,5,l}^{(r)}
$$

定义 TTGRL04 correction：

$$
\Delta W_{corr,5,l}^{(r)} = \Delta W_{TTGRL04,5,l}^{(r)} - \Delta W_{B0,5,l}^{(r)}
$$

计算：

$$
Cos_{corr,removed,l}^{(r)} = \cos(\Delta W_{corr,5,l}^{(r)}, \Delta W_{removed,5,l}^{(r)})
$$

以及：

$$
MagRatio_{corr,removed,l}^{(r)} = \frac{\|\Delta W_{corr,5,l}^{(r)}\|_F}{\|\Delta W_{removed,5,l}^{(r)}\|_F+\epsilon}
$$

### 必须记录

```text
freeze_ttgr_direction_alignment.csv
```

字段：

```text
layer_id
branch
cos_corr_removed
cos_corr_static
cos_removed_static
mag_corr
mag_removed
mag_ratio
sign_interpretation
```

### 成立标准

如果满足下面任一情况，则 H2 成立：

```text
1. cos_corr_removed 大部分接近 0 或为正，而不是强负，说明 TTGRL 并未反向抵消 freeze harmful direction；
2. 只有少数 layer/branch 有强负对齐，且 mag_ratio 很小，说明 TTGRL 打到了但强度不足；
3. TTGRL correction 与 static direction 也冲突，说明当前 TTGR correction 混合了 harmful 和 useful continuity；
4. freeze removed direction 与 static direction 本身混合，说明必须做 token-level decomposition。
```

H2 通过后，下一步不是调 gamma，而是做 token-level direction attribution。

---

## H3：per-token update conflict 比 `D_g` 更适合定义 TTT negative evidence

### 假设

`D_g` 是 read cue，不是最优 TTT write cue。真正的 TTT negative evidence 应该来自 per-token update direction 与 static / native / freeze direction 的冲突。

### 实验

对 chunk5 的每个 token，计算：

```text
D_g
semantic group
E_pre
update_norm_w0/w1/w2
static_align_w0/w1/w2
static_conflict_w0/w1/w2
apply_mismatch
lr stats
```

再按 token group 比较：

```text
high D_g vs low D_g
structure vs lowstuff
high update_norm vs low update_norm
high conflict vs low conflict
TTGR negative-selected vs not selected
freeze-sensitive proxy top-k vs rest
```

### 必须记录

```text
ttt_token_cue_chunk5.pt
ttt_token_cue_chunk6.pt
ttt_token_group_stats.csv
ttt_cue_overlap_with_Dg_semantic.csv
```

关键字段：

```text
token_id
frame_id
patch_y
patch_x
D_g
semantic_group
semantic_label
mask_trust
C_anchor
C_unc
E_pre
E_apply
R_apply
lr0
lr1
lr2
update_norm_w0
update_norm_w1
update_norm_w2
static_align_w0
static_align_w1
static_align_w2
static_conflict_w0
static_conflict_w1
static_conflict_w2
selected_by_TTGR_low_prior
selected_by_high_conflict
```

### 成立标准

H3 通过条件：

```text
1. high-conflict token group 与 TTGR low-prior group 不完全重合；
2. high-conflict token group 更集中于 chunk5 / 病灶入口；
3. high-conflict token group 与 freeze removed direction 更对齐；
4. high-conflict group 的 semantic 分布能解释 sky/vegetation/road 等不同角色；
5. high-conflict cue 比 D_g 单独更能区分 harmful direction。
```

如果 H3 成立，后续 TTT negative replay 应使用 `update_conflict`，而不是 `D_g` 或 semantic scalar。

---

## H4：TTT apply/readback mismatch 能区分 stable memory anchor 和 harmful memory influence

### 假设

有些 token 不一定产生最大 update，但当前 TTT memory 对它们的 apply/readback 已经错了。apply mismatch 能提示 memory 已污染或 readback 不可靠。

### 实验

对 chunks 5-10 计算：

$$
R_i^{apply} = \frac{\|f_W(q_i)\|_2}{\|x_i\|_2 + \epsilon}
$$

$$
E_i^{apply} = \frac{\|f_W(q_i)-v_i\|_2}{\|v_i\|_2 + \epsilon}
$$

并与：

```text
D_g
semantic group
chunk error
frame error
C_anchor
pre-error
```

做 correlation 和可视化。

### 必须记录

```text
apply_readback_cue_by_chunk.pt
apply_readback_summary.csv
apply_mismatch_error_corr.csv
```

### 成立标准

H4 通过条件：

```text
1. chunks 7-9 的 high-error 区域存在高 apply_mismatch；
2. chunk5 freeze 或 TTGRL_04 会改变后续 chunks 的 apply_mismatch 分布；
3. apply_mismatch 与 [200,300) per-frame error 相关性高于 D_g mass；
4. high apply_mismatch 不只是 sky/lowstuff coverage 的副作用。
```

---

## H5：next-chunk influence cue 能识别真正会传播错误的 token

### 假设

TTT 写入的核心不是当前 token 的分类，而是该 token 写入后对未来 chunk 的 apply 影响。next-chunk influence 能识别真正 memory-propagating harmful token。

### 实验

第一版只在 chunk5 做 top-k token influence，不全量跑。

选择候选 token 集合：

```text
top update_norm_w0
top static_conflict_w0
top E_pre
top D_g
semantic structure top
semantic lowstuff highD top
random control
```

对每个候选 token 或 token group，近似计算：

$$
I_i = \| f_{W + \epsilon J_i}(q^{next}) - f_W(q^{next}) \|_2
$$

如果 token-wise 太贵，可以先 group-wise：

$$
I_{group} = \| f_{W + \epsilon \sum_{i \in group} J_i}(q^{next}) - f_W(q^{next}) \|_2
$$

`q^{next}` 建议先取 chunk6 / chunk7 的 q cache。

### 必须记录

```text
next_influence_chunk5_groups.csv
next_influence_patchmaps.pt
```

字段：

```text
group_name
branch
layer_scope
epsilon
influence_norm_chunk6
influence_norm_chunk7
influence_align_static
influence_align_B0_to_freeze5
semantic_distribution
Dg_mean
conflict_mean
```

### 成立标准

H5 通过条件：

```text
1. high conflict group 对 next apply 的 influence 明显高于 random / high D_g only；
2. influence direction 能解释 freeze5 修 [200,300) 的方向；
3. structure positive group 与 lowstuff/highD negative group 对 next apply 的影响方向不同；
4. influence cue 能给出比 scalar TTGR 更精确的 negative set。
```

---

## H6：语义应该用于解释 TTT cue，而不是先验规定 TTT cue

### 假设

语义不是 TTT cue 的主来源，而是用于解释 TTT-native cue 的 role assignment。sky / vegetation / grass 不应默认 hard negative；road / building / fence 也不应默认 hard positive。是否 positive / negative 取决于 TTT cue 中的 update direction 与 memory alignment。

### 实验

把 TTT cue 按 semantic group 统计：

```text
STRUCTURE_ANCHOR:
    road/building/fence/wall

LOW_VALUE_STUFF:
    sky/vegetation/grass

MOVABLE_THING:
    当前 Mask2Former cache 中可能较少

UNCERTAIN_REGION:
    low trust / fragment / no mask
```

记录每个 group 的：

```text
update_norm
static_conflict
E_pre
E_apply
next_influence
selected_negative_mass
selected_positive_mass
```

### 成立标准

H6 通过条件：

```text
1. 语义 group 不能单独预测 harmful update；
2. lowstuff 内部存在 neutral / useful continuity token；
3. structure 内部也存在 high-conflict token；
4. semantic group 与 TTT cue 组合后能定义更合理的 positive/neutral/negative role。
```

---

## 8. 第一阶段实现计划：Instrumentation 与 Passive Audit

### 8.1 Phase T0：instrumentation no-op gate

新增 TTT trace 不允许改变任何输出。必须先跑：

```text
T0_01: B0_SWKS3 without TTT cue instrumentation
T0_02: B0_SWKS3 with TTT cue instrumentation, no control consumed
```

通过标准：

```text
|ATE_T0_02 - ATE_T0_01| <= 0.005m
|Rot_T0_02 - Rot_T0_01| <= 0.005deg
trajectory max abs diff == 0 或在已知浮点误差范围内
hmc_state_hash 一致或可解释
```

若不通过，所有 passive cue 都不能用于后续判断。

### 8.2 需要新增的代码接口

建议新增：

```text
loger/pipeline/ttt_cue_extractor.py
```

核心类：

```python
@dataclass
class TTTCueOutput:
    token_maps: Dict[str, torch.Tensor]
    layer_branch_stats: Dict[str, Any]
    group_stats: Dict[str, Any]
    debug: Dict[str, Any]

class TTTCueExtractor:
    def run_from_write_cache(...):
        ...
```

它应尽量复用 `WriteCacheOutput` 中已经保存的：

```text
k, v, lr0, lr1, lr2
w_old / w_provisional
layer_caches
ttt_ua_order
muon_update_steps
ttt_update_steps
```

不要在第一版重新跑完整 LoGeR forward。

### 8.3 最小 required outputs

第一版只要求落盘这些：

```text
per_layer_branch_stats.csv
per_token_ttt_cues_chunk5.pt
per_token_ttt_cues_chunk6.pt
per_token_ttt_cues_chunks7_9.pt
freeze_ttgr_alignment.csv
semantic_ttt_group_stats.csv
visualization_manifest.json
```

不要一开始就全序列全 token 全矩阵保存，避免 IO 过大。

---

## 9. 第二阶段实验：TTT Cue Passive Audit

### 9.1 固定 run set

```text
B0_SWKS3
TTGRL_04
P3G_01
freeze5
freeze56
```

如果时间允许，加：

```text
TTGR_03 global
TTGRST_03 transient
```

### 9.2 固定 chunk set

```text
chunk5
chunk6
chunk7
chunk8
chunk9
chunk10
```

其中 chunk5/chunk6 做最全 token cue，chunks7-10 可以先只做 apply/readback cue 和 summary。

### 9.3 输出目录规范

```text
results/kitti01_hmc_v2/ttt_cue_library_v1/
    T0_noop_gate/
    passive_audit_B0/
    passive_audit_TTGRL04/
    passive_audit_freeze5/
    passive_audit_freeze56/
    dashboards/
        chunk5/
        chunk6/
        chunks7_9/
    tables/
        per_layer_branch_stats.csv
        freeze_ttgr_alignment.csv
        token_group_stats.csv
        hypothesis_summary.md
```

---

## 10. 必须记录的指标

### 10.1 轨迹指标

每个 run 必须记录：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
YawRMSE
Sim3Scale
50f mean / worst
100f mean / worst
200f mean / worst
```

重点 segments：

```text
[100,200)
[200,250)
[200,300)
[200,400)
[400,500)
[400,600)
```

重点 chunks：

```text
chunk5
chunk6
chunk7
chunk8
chunk9
chunk10
```

### 10.2 TTT layer / branch 指标

每个 chunk、layer、branch 记录：

```text
pre_update_norm
post_zeropower_norm
committed_delta_norm
delta_norm_ema_ratio
delta_cos_prev
pre_post_zeropower_cos
norm_restore_ratio
memory_state_rel_diff
cos_to_B0_delta
cos_to_freeze_removed
cos_to_TTGR_correction
```

### 10.3 Token-level TTT cue 指标

每个 selected chunk 的 patch token 记录：

```text
D_g
semantic_group
semantic_label
mask_trust
C_anchor
C_unc
C_dyn / explicit_dyn if available
E_pre
E_apply
R_apply
lr0/lr1/lr2
lr_mean
lr_branch_entropy
update_norm_w0/w1/w2
update_norm_total
static_align_w0/w1/w2
static_conflict_w0/w1/w2
native_align_w0/w1/w2
freeze_align_w0/w1/w2
next_influence_score if computed
```

### 10.4 Group-level 指标

按 group 输出：

```text
group_name
num_tokens
area_mass
D_g_mean
E_pre_mean
E_apply_mean
update_norm_mean_w0/w1/w2
static_conflict_mean_w0/w1/w2
freeze_align_mean_w0/w1/w2
negative_candidate_mass
positive_candidate_mass
semantic_distribution
```

### 10.5 相关性指标

按 chunk 或 frame 计算：

```text
corr(D_g_mass, chunk_error)
corr(E_pre_p90, chunk_error)
corr(update_norm_w0_p90, chunk_error)
corr(static_conflict_w0_mass, chunk_error)
corr(apply_mismatch_p90, chunk_error)
corr(next_influence_mass, next_chunk_error)
```

这里的目标不是建立严肃统计模型，而是看 TTT cue 是否比现有 `D_g` / semantic scalar 更能解释 chunk5 / `[200,300)`。

---

## 11. 必须做的可视化

### 11.1 Chunk dashboard

对 chunk5、chunk6、chunk7、chunk8、chunk9 输出：

```text
RGB
D_g
semantic group
C_anchor / C_unc
TTT pre-error
TTT apply mismatch
TTT update_norm_w0
TTT update_norm_w1
TTT update_norm_w2
TTT static_conflict_w0
TTT freeze_align_w0
TTT next_influence if available
```

每个图必须按 frame grid 展示，不要只显示均值。

### 11.2 Layer / branch heatmap

输出：

```text
chunk × layer × branch delta_norm heatmap
chunk × layer × branch delta_flip heatmap
chunk × layer × branch pre_post_zeropower_distortion heatmap
freeze5_removed vs TTGRL04_correction cosine heatmap
```

重点看 chunk5 / chunk6 是否有异常 layer/branch。

### 11.3 Direction cosine matrix

对 B0、freeze5、freeze56、TTGRL04、P3G01 的 chunk5 delta，画：

```text
cosine matrix by branch/layer
```

至少包括：

```text
B0 delta
freeze5 removed
freeze56 removed
TTGRL04 correction
P3G01 correction
static group delta
lowstuff highD delta
```

### 11.4 Semantic overlay

按语义 group 输出箱线图或直方图：

```text
update_norm_w0 by semantic group
static_conflict_w0 by semantic group
E_pre by semantic group
apply_mismatch by semantic group
freeze_align by semantic group
```

目标是判断：

```text
sky / vegetation 是否真的 harmful；
road / building 是否真的 positive；
还是 TTT cue 与语义存在更复杂关系。
```

### 11.5 Error linkage plot

画：

```text
chunk index vs chunk ATE
chunk index vs TTT delta_norm spike
chunk index vs static_conflict mass
chunk index vs apply_mismatch p90
chunk index vs D_g mass
```

重点标注：

```text
chunk5
chunk6
[200,300)
```

---

## 12. 从 passive cue 到 control 的晋级标准

一个 TTT cue 只有满足以下条件之一，才进入 write-control 实验：

```text
1. 它在 chunk5 / chunk6 上显著异常，并且异常 layer/branch 与 freeze causal effect 对齐；
2. 它能区分 freeze-sensitive token group 和普通 high-D token group；
3. 它与 [200,300) 或 chunks7-9 error 的相关性明显高于 D_g mass；
4. 它能解释 TTGRL_04 为什么改善 overall ATE 却不修 [200,300)；
5. 它显示 positive continuity 与 harmful direction 分布在不同 token group 或不同 layer/branch。
```

不满足这些条件的 cue 只能保留为 diagnostic，不进入写入策略矩阵。

---

## 13. 第一批控制实验：TTT-native cue write policy

只有 passive audit 通过后才做控制。第一批只做小矩阵，不再大扫 gamma。

### 13.1 控制目标

从：

$$
G_{commit} = \sum_i a_i G_i
$$

升级到：

$$
G_{commit} = G_{pos} + \lambda G_{neu} - \gamma G_{neg}
$$

### 13.2 候选策略

#### TWO_01：static alignment positive + high conflict negative

```text
scope:
    chunk5, w0, all layers

positive:
    static_align_w0 high
    D_g low

negative:
    static_conflict_w0 high
```

#### TWO_02：semantic interpreted pos/neg

```text
positive:
    STRUCTURE_ANCHOR ∩ low D_g ∩ high static_align

neutral:
    LOW_VALUE_STUFF ∩ low D_g

negative:
    LOW_VALUE_STUFF ∩ high D_g ∩ high static_conflict
```

#### TWO_03：freeze-alignment targeted negative

```text
positive:
    high static_align

negative:
    high freeze_align_to_removed_direction
    but low static_align
```

#### TWO_04：layer-targeted version

```text
scope:
    top layers from freeze_ttgr_direction_alignment.csv

positive / negative:
    same as TWO_01 or TWO_03
```

### 13.3 控制成功标准

强成功：

```text
ATE <= 36.15
或 [200,300) 下降 >= 3m 且 overall ATE <= 36.40
```

弱成功：

```text
ATE <= 36.29
且 Rot / FinalErr / [200,300) 至少两项优于 TTGRL_04
```

停止条件：

```text
[200,300) 不动，只改善 Rot / FinalErr；
或 overall ATE 回到 36.5m 以上；
或 chunk6/chunk7 error 明显恶化。
```

---

## 14. 第二批控制实验：dual-lifetime TTT

如果 passive audit 显示某些 update 对短期有用、长期有害，则做 dual-lifetime。

### 14.1 思路

将 fast weights 拆成：

```text
W_long:
    长期 memory，只接收 positive + small neutral

W_transient:
    短期 correction，接收 risky / dynamic / negative evidence
    apply K chunks 后衰减，不长期 commit
```

最小实现可用 overlay：

$$
W_{apply,next} = W_{long} + \alpha_t W_{transient}
$$

$$
W_{commit,next} = W_{long}
$$

$$
W_{transient}^{m+k} = \rho^k W_{transient}^{m}
$$

### 14.2 实验矩阵

```text
DL_01:
    negative cue = high static_conflict_w0
    K = 1
    alpha_t = 0.50

DL_02:
    same cue
    K = 2
    alpha_t = 0.50

DL_03:
    same cue
    K = 3
    alpha_t = 0.25

DL_04:
    semantic interpreted negative
    K = 2
    alpha_t = 0.50
```

### 14.3 判断标准

dual-lifetime 成立：

```text
1. [200,300) 下降 >= 3m；
2. [400,600) 不明显恶化；
3. overall ATE <= TTGRL_04 + 0.05m；
4. FinalErr / Yaw 不明显崩。
```

如果 dual-lifetime 不能动 `[200,300)`，说明当前 negative cue 仍然没找对。

---

## 15. 第三批控制实验：risk-triggered TTT policy

在 chunk5 上做机制 discovery 是允许的，但最终不能 hard-code chunk5。需要从 TTT cue 中找 risk trigger。

定义 chunk risk：

$$
Risk_m = a M_m^{w0} + b F_m^{w0} + c C_m^{conflict} + d E_m^{apply} + e I_m^{next}
$$

其中各项来自 TTT cue summary。

第一版不训练权重，只做 rule-based diagnostic：

```text
risk trigger candidate:
    delta_norm_ema_ratio_w0 > q90
    or static_conflict_mass_w0 > q90
    or apply_mismatch_p90 > q90
```

判断：

```text
chunk5 是否被触发？
chunk6 是否被触发？
其它 chunk 是否大量误触发？
触发后的 policy 是否能泛化到 KITTI00/02/05？
```

---

## 16. 工程注意事项

### 16.1 先 passive，后 control

所有 TTT cue 都先 passive audit，不允许直接接入 HMC 控制。否则无法区分：

```text
cue 本身有解释力
vs
控制策略碰巧改变了轨迹
```

### 16.2 不要一次保存全量 per-token matrices

per-token outer product 太大。第一版保存：

```text
norm
alignment score
branch/layer summary
必要 top-k token 的 low-rank factors
```

只有 top-k influence 实验才 materialize 局部 $J_i$。

### 16.3 与 existing HMC debug 兼容

建议所有 TTT cue summary 写入：

```text
hybrid_debug_jsonl
prior_debug_jsonl
```

或单独：

```text
ttt_cue_debug_jsonl
```

但必须保留 `run_id / chunk_id / frame_range / config_hash`。

### 16.4 保持 Stage C cache policy

语义只用于 overlay 和 group statistics。Stage C 必须使用离线 cache + require-hit read，不允许 inline compute 进入 parity run。

### 16.5 资源策略

```text
KITTI01 full passive trace:
    2-4 并发，避免 IO 和 CPU 写 cache 过载

next influence top-k:
    单 GPU 或低并发

普通 control full run:
    可以沿用 4-6 并发，但必须确认 host RAM
```

---

## 17. 推荐执行顺序

### Step 1：T0 no-op instrumentation

```text
B0 without TTT cue trace
B0 with TTT cue trace
```

通过后进入 Step 2。

### Step 2：passive audit 五个参考 run

```text
B0_SWKS3
TTGRL_04
P3G_01
freeze5
freeze56
```

输出所有 layer/branch summary 和 chunk5/chunk6 token cue。

### Step 3：方向归因

计算：

```text
freeze5 removed direction
freeze56 removed direction
TTGRL04 correction direction
P3G01 correction direction
static group direction
lowstuff highD group direction
```

输出 cosine heatmap。

### Step 4：TTT cue 与 semantic / D_g overlay

输出 chunk5/chunk6 dashboard。

回答：

```text
high D_g 是否就是 high update conflict？
sky/vegetation 是否真的 negative？
structure 是否真的 positive？
```

### Step 5：top-k next influence

只对 chunk5 的 top-k groups 做。

### Step 6：第一批 two-replay control

只让通过 passive gate 的 cue 进入控制。

### Step 7：dual-lifetime diagnostic

若 two-replay 不能修 `[200,300)`，但 passive cue 显示 short-term/long-term 分离，则执行。

---

## 18. 最终判定逻辑

本阶段结束时要给出三类结论之一。

### 18.1 结论 A：TTT-native cue 成立，并能进入主线

条件：

```text
passive cue 能解释 chunk5 / [200,300)；
control 后 ATE <= 36.15 或 [200,300) 下降 >= 3m；
无明显 [400,600) 转移错误；
```

下一步：扩展到 cross-sequence，并系统做 TTT branch/layer/lifetime policy。

### 18.2 结论 B：TTT-native cue 有解释力，但当前控制形式不够

条件：

```text
passive cue 与 freeze/TTGR 方向强相关；
但 two-replay / dual-lifetime 仍无法显著动 [200,300)。
```

下一步：改 TTT replay objective 或实现更真实的 separate transient fast weights，而不是继续 scalar control。

### 18.3 结论 C：当前 trace 不足以挖出 TTT cue

条件：

```text
TTT cue 无法解释 chunk5；
也无法区分 freeze-sensitive 与普通 high-D token；
```

下一步：加强 instrumentation，例如记录更底层 q/k/v/lr/head-wise TTT cache、pre/post Muon SVD summary、per-head TTT contribution。

---

## 19. 本阶段不做什么

为了避免再次进入盲扫，本阶段暂时不做：

```text
1. 继续普通 gamma 细扫；
2. 继续 old_dyn / explicit_dyn scalar fusion；
3. 继续 semantic scalar value 大扫；
4. 继续 SWA 新策略；
5. controlled commit；
6. 大规模 cross-sequence，除非 KITTI01 通过强 gate；
7. 没有 passive attribution 的 write-control 组合。
```

---

## 20. 总结

这一阶段的核心不是再调一个 TTT prior，而是把 TTT 变成可观察、可归因、可解释的 memory system。

Frame-attention cue 告诉我们：

```text
当前 read path 中哪些 token 是 harmful support。
```

TTT-native cue 应该告诉我们：

```text
哪些 token 的 fast-weight update 会污染长期 memory；
哪些 token 是 positive continuity；
哪些 token 只应该短期存在；
哪些 layer/branch 是 [200,300) 病灶的真实传播通道。
```

只有把这两类 cue 分开，才能避免继续把 `D_g` 强行解释成 TTT write cue，也才能从当前 `36.3m` 平台进入真正的 TTT 写入机制改造。


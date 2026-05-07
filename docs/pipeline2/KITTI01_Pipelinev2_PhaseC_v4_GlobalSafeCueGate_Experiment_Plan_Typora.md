# KITTI 01 / HMC Phase C v4 Global-Safe Cue Gate 实验方案

> 版本：Phase C v4  
> 目标序列：KITTI Odometry 01  
> 当前 reference：`RFR-100 = old C_dyn + frame attention pair bias + early layers + beta=1.0`，full ATE = `41.0733 m`  
> 当前失败候选：`flow_sem_veto + frame key + beta=1.0`，64-frame ATE = `0.9163 m`，256-frame ATE = `3.8000 m`，stateful slice gate `6/7` improved，但 full KITTI 01 ATE = `41.4333 m`  
> 本阶段目的：不再用短序列或 local Sim(3) slice 作为主要筛选依据，而是建立 **full-sequence-safe cue gate**，只允许能够保护全局轨迹连续性的 read-path cue 进入 full KITTI 01。

---

## 0. 当前结论与为什么 Phase C v3 失败

Phase C v3 的结果不是“完全没有信号”。`flow_sem_veto` 在 64-frame、256-frame、stateful continuation slice 上都表现很好，尤其 256-frame ATE 从 RFR reference 的 `3.8666 m` 改到 `3.8000 m`，stateful slice 也达成 `6/7` improved。但它的 full KITTI 01 结果是 `41.4333 m`，比 RFR-100 的 `41.0733 m` 差 `+0.3600 m`。这说明 v3 的 gate 仍然太宽松：它能筛出局部 continuation 有用的 cue，却不能筛出 full-sequence trajectory 安全的 cue。

这轮失败最关键的诊断不是 ATE 数字本身，而是 cue 统计。`flow_sem_veto` full run 的 mean dynamic mass 只有 `0.010`，明显低于原计划希望的 `0.03–0.25` 区间，而且 `Corr(D, Conf)=0.680`。这意味着它并不是一个干净的 moving-object map，更像是一个非常稀疏的高置信 residual selector。这样的 cue 可以在局部 slice 上避免一些坏交互，但 full sequence 里它缺少足够稳定、足够连续的覆盖，不能形成可靠的全局 read-path regularization。

因此 Phase C v4 的核心判断是：

$$
\text{短序列 ATE 改善} \not\Rightarrow \text{full KITTI 01 ATE 改善}
$$

$$
\text{local Sim(3) slice 改善} \not\Rightarrow \text{global trajectory continuity 改善}
$$

Phase C v4 不应该继续简单扫 `beta`、`top-k`、`protected_pair` 或替换一个单独 cue。它要先回答：

> 一个 read-path cue 是否具有 **足够动态覆盖、足够低的结构误伤、足够低的全局漂移风险**？

只有通过这个 full-sequence-safe gate 的 cue，才值得跑 full KITTI 01。

---

## 1. Phase C v4 的中心假设

Phase C v4 围绕四个假设设计实验。

### 假设 C4-H1：RFR-100 的有效性来自“dense and smooth regularization”，不一定来自准确动态分割

RFR-100 使用 old `C_dyn`、pair bias、early frame attention，full ATE 达到 `41.0733 m`，是目前最好的 read-path-only reference。它没有过 `<41.0 m`，但比 A3 native、BL01 TTT branch0 write 都更好。相反，`flow_sem_veto` 在短序列更强，full 却更差。这个对比说明：full KITTI 01 上真正需要的不是最稀疏、最尖锐的动态点，而是一个能在全序列上稳定影响 attention 交互的 read-path regularizer。

所以 Phase C v4 不再把 `old C_dyn` 当作过时 baseline。它被视为一个 **dense regularization base**。新的 flow / Gram / semantic-veto cue 不直接替换它，而是作为 residual correction 或 reliability switch。

### 假设 C4-H2：v3 stateful slice gate 失败，是因为它只看 local Sim(3) ATE，没有约束全局 continuity

v3 stateful slice gate 用从 full no-control run 保存的 HMC state 分叉测试中后段 slice，这是比 prefix gate 更合理的；但它仍然用 local Sim(3)-aligned ATE 判断好坏。local Sim(3) alignment 会消除很多 scale / global-frame / continuity 错误，因此容易出现：

$$
\Delta ATE_{\text{local Sim(3)}} < 0
$$

但 full sequence 中：

$$
\Delta ATE_{\text{full}} > 0
$$

Phase C v4 必须给 slice 增加 global continuity 约束，包括 global-frame endpoint drift、path-length ratio、chunk boundary pose jump、overlap pose consistency。

### 假设 C4-H3：`flow_sem_veto` 失败不是因为 optical-flow 思路错，而是因为当前 proxy 太稀疏且不是真正外部 flow

当前 `flow_sem_veto` 实现使用的是 patch-match / Stage-B reprojection proxy，不是 RAFT 或 GMFlow。它比 same-pixel `C_dyn` 更强，但还不是严格的 optical-flow residual。真正的 dynamic motion cue 应该比较：

$$
F^{obs}_{t\rightarrow s}(u)
$$

和由 probe pose + pointmap 诱导出的 rigid flow：

$$
F^{rigid}_{t\rightarrow s}(u)
=
\pi(T_{c_s\leftarrow w} X_t(u)) - u
$$

动态残差为：

$$
r_{flow}(t,u)
=
\frac{
\lVert F^{obs}_{t\rightarrow s}(u)-F^{rigid}_{t\rightarrow s}(u)\rVert_2
}{
\lVert F^{obs}_{t\rightarrow s}(u)\rVert_2+\lVert F^{rigid}_{t\rightarrow s}(u)\rVert_2+\epsilon
}
$$

如果外部 flow 版本仍然失败，那么 read-path-only motion suppression 的上限就非常明确；但在没有 true flow 前，不应该把 v3 的 proxy 失败解释成“flow 方向无效”。

### 假设 C4-H4：Phase C 需要“cue calibration + fallback”，而不是单一 cue 的全序列强制注入

v3 的 full failure 说明同一个 cue 在所有 chunk 上全程使用是不安全的。Phase C v4 应该引入 chunk-level reliability：

$$
g_m \in [0,1]
$$

用于决定当前 chunk 是否相信新 cue。如果新 cue dynamic mass 太低、结构误伤太高、fragmentation 太强，controller 应该回退到 RFR-100 或 identity，而不是强制使用新 cue。

---

## 2. 代码正确性 Gate：必须在新 HMC pipeline 上复现 LoGeR 和 LoGeR*

Phase C v4 在任何非 identity control 之前，必须重新通过 correctness gate。原因是本阶段会新增 flow cue、cue calibration、global-slice metrics、chunk-level fallback，任何一处实现错误都可能制造假改进或假失败。

这里禁止用旧 `geometry_eval_mode`、旧 `run_pipeline_abc.py` 的 TTT-only 路径、旧 native shortcut 或旧 merge 逻辑来产出正式 baseline。所有正式 correctness 结果必须走新 HMC pipeline：

```text
two-pass HMC runner
probe pass
identity / no-control control package
controlled pass
HMC state commit
native reset-block-aware merge
KITTI evaluation
```

### 2.1 C4-0A：LoGeR no-control full reproduction

运行 LoGeR checkpoint，在 HMC two-pass no-control 模式下处理完整 KITTI 01。所有 hook 都必须被注册，但 bias / gate 必须为 identity。

期望结果：

$$
ATE_{\text{LoGeR,HMC-no-control}}
\approx 41.7502\text{ m}
$$

判断标准：

$$
|ATE_{\text{HMC-no-control}}-41.7502| < 0.15\text{ m}
$$

同时必须满足：

- `probe_no_commit_hash_equal_all = true`
- `state_double_write_safe_all = true`
- pass1/pass2 no-control pose translation diff max = `0.0 m`
- frame-attention hook count > 0，但 total bias = 0
- SWA hook count > 0，但 total bias = 0
- TTT apply hook count > 0，但 gate = 1
- chunk-attention hook count > 0，但 total bias = 0

如果这个 gate 不通过，Phase C v4 立即停止。

### 2.2 C4-0B：LoGeR* no-control full reproduction

运行 LoGeR* checkpoint，必须走 HMC two-pass no-control，并保留 SE(3) overlap alignment / reset 机制。

期望结果：

$$
ATE_{\text{LoGeR*,HMC-no-control}}
\approx 47.9793\text{ m}
$$

判断标准：

$$
|ATE_{\text{HMC-no-control}}-47.9793| < 0.20\text{ m}
$$

LoGeR* 的存在非常重要，因为它 rotation 更好但 ATE 更差。后续任何 read-path cue 如果只改善 rotation、破坏 translation，很可能是在复现 LoGeR* 的失败方向，而不是改善 LoGeR。

### 2.3 C4-0C：RFR-100 reference reproduction

Phase C v4 所有候选必须和 RFR-100 比，而不是只和 A3 native 或 BL01 比。先复现：

```text
read_cue_source = old_dyn
read_path = frame
frame_bias_mode = pair
read_layer_mode = early
beta = 1.0
```

期望结果：

$$
ATE_{\text{RFR-100}} \approx 41.0733\text{ m}
$$

判断标准：

$$
|ATE_{\text{RFR-100,reproduced}}-41.0733| < 0.10\text{ m}
$$

如果 RFR-100 复现失败，不允许进入新 cue 实验。

---

## 3. Phase C v4 的主要变化：从 SignalGate 改成 Global-Safe Cue Gate

Phase C v3 的 gate 是：

```text
64-frame smoke
→ 256-frame check
→ stateful local Sim(3) slice
→ full candidate
```

Phase C v4 改成：

```text
HMC correctness
→ full-run probe-only cue audit
→ cue calibration and fallback design
→ global-continuity stateful slice
→ full candidate, at most two
```

区别在于，v4 不是先跑短序列再筛 full，而是在完整 no-control probe 的所有 chunk 上先看 cue 是否像一个 full-sequence-safe signal。

---

## 4. C4-1：Full-run probe-only cue audit

这一阶段只跑 no-control probe，不注入任何 read-path control。目标是对所有候选 cue 做完整序列统计，决定哪些 cue 有资格进入 controlled experiment。

### 4.1 候选 cue

本阶段至少评估以下 cue：

| Cue 名称 | 来源 | 目的 |
|---|---|---|
| `old_dyn` | 当前 RFR-100 用的 `C_dyn` | dense regularization reference |
| `gram4d` | VGGT4D-style Gram / LoGeR internal QK statistics | 内部 motion cue |
| `flow_sem_veto_proxy` | v3 的 patch-match / reprojection proxy | 失败候选复检 |
| `flow_epi_raft` | RAFT observed flow vs LoGeR rigid flow | true external flow residual |
| `flow_epi_gmflow` | GMFlow observed flow vs LoGeR rigid flow | true external flow residual |
| `old_dyn_plus_flow` | old dyn base + flow residual | residual correction |
| `old_dyn_switch_flow` | chunk-level fallback / switch | reliability-gated cue |

如果 RAFT / GMFlow 尚未实现，Phase C v4 可以先完成 `old_dyn + calibrated flow proxy` 和 `global-continuity slice`，但不能把结果解释成 external flow 的结论。

### 4.2 Cue 统计指标

对每个 cue $D_m(t,i)$，其中 $i$ 是 patch token，记录以下 full-run 统计：

#### 4.2.1 Dynamic mass

$$
\rho_m =
\frac{1}{T P}
\sum_{t,i}
\mathbf{1}[D_m(t,i)>\tau_D]
$$

要求：

$$
0.03 \le \operatorname{Mean}_m(\rho_m) \le 0.20
$$

同时要求：

$$
\Pr_m(\rho_m > 0.01) \ge 0.80
$$

v3 的 `flow_sem_veto` mean mass = `0.010`，因此不合格。它太稀疏，不能直接全序列替换 RFR-100。

#### 4.2.2 Confidence correlation

记录：

$$
Corr(D, Conf)
$$

如果：

$$
Corr(D, Conf) > 0.60
$$

说明 cue 可能主要选择高置信区域，而不是动态区域。这类 cue 不允许作为主 dynamic mask，只能作为 residual 或 update-needed cue。

#### 4.2.3 Anchor collision

定义：

$$
\chi_{anchor}
=
\frac{
\sum_i D_i C_{anchor,i}
}{
\sum_i D_i+\epsilon
}
$$

如果：

$$
\chi_{anchor} > 0.35
$$

说明 cue 大量打在高 anchor 区域上，存在 suppress 静态结构的风险。该 cue 不能进入 full candidate。

#### 4.2.4 Fragmentation

把每帧 dynamic mask 连通域数量记为 $N_{cc}(t)$，dynamic mass 记为 $\rho(t)$。定义：

$$
Frag(t)=\frac{N_{cc}(t)}{\rho(t)TP+\epsilon}
$$

如果 fragmentation 过高，说明 cue 是边缘点 / residual speckles，不是对象或稳定区域。这样的 cue 只允许做 secondary residual，不允许作为 attention key suppression 主 cue。

#### 4.2.5 Overlap with old_dyn

记录：

$$
IoU(D_{\text{new}},D_{\text{old}})
$$

和：

$$
Coverage(D_{\text{new}}\mid D_{\text{old}})
=
\frac{
|D_{\text{new}}\cap D_{\text{old}}|
}{
|D_{\text{new}}|+\epsilon
}
$$

如果 new cue 几乎不和 old_dyn 重叠，但又没有独立 motion validation，它可能是另一个不稳定 proxy。

### 4.3 Cue audit 的判断标准

一个 cue 要进入 C4-2 controlled gate，必须满足：

| 指标 | 通过条件 |
|---|---|
| Mean dynamic mass | `0.03–0.20` |
| Chunk coverage | 至少 80% chunks 的 mass > 0.01 |
| Confidence correlation | `Corr(D,Conf) < 0.60`；若高于 0.60，只能做 residual |
| Anchor collision | `<0.35` |
| Fragmentation | 不得显著高于 old_dyn 的 2 倍 |
| Cue availability | 每个 reset block 内至少有 2 个有效 chunks |

如果没有任何 new cue 通过，Phase C v4 不进入 full candidate；只允许继续实现 true external flow 或转向 Phase D / learned gate。

---

## 5. C4-2：Cue calibration 与 fallback 设计

v3 的核心问题是 cue 太稀疏。Phase C v4 要把所有 cue 先校准到可控 dynamic mass，再做 read-path injection。

### 5.1 Per-frame adaptive quantile calibration

对 raw cue $D_{raw}(t,i)$，定义目标 mass $q$，例如：

$$
q \in \{0.03,0.06,0.10\}
$$

令：

$$
\theta_t = Q_{1-q}(D_{raw}(t,:))
$$

用 soft threshold 得到：

$$
D_{calib}(t,i)
=
\sigma
\left(
\frac{D_{raw}(t,i)-\theta_t}{\tau}
\right)
$$

其中 $\tau$ 控制软硬程度，先取：

$$
\tau \in \{0.05,0.10\}
$$

这样每帧都能获得近似目标 dynamic mass，而不是出现 v3 那种 full sequence mean mass `0.010` 的极稀疏 map。

### 5.2 Old-dyn base + residual correction

RFR-100 是当前 reference，所以 first-order candidate 不应替换 old_dyn，而应定义：

$$
D_{blend}
=
\operatorname{Norm}
\left(
(1-\lambda)D_{old}
+
\lambda D_{new,calib}
\right)
$$

其中：

$$
\lambda \in \{0.25,0.50\}
$$

这个设计的含义是：保留 old_dyn 的 dense full-sequence regularization，再用 flow / Gram cue 做局部修正。

### 5.3 Reliability switch / fallback

对每个 chunk 计算 cue quality：

$$
Q_m =
\mathbf{1}[0.03\le \rho_m \le 0.20]
\cdot
\mathbf{1}[\chi_{anchor,m}<0.35]
\cdot
\mathbf{1}[Frag_m<\tau_{frag}]
$$

然后定义：

$$
D_m =
Q_m D_{new,calib}
+
(1-Q_m)D_{old}
$$

也可以用连续 gate：

$$
D_m =
g_m D_{new,calib}
+
(1-g_m)D_{old}
$$

其中：

$$
g_m=
\operatorname{clip}
\left(
\frac{\rho_m-0.02}{0.04},
0,1
\right)
\cdot
\operatorname{clip}
\left(
\frac{0.35-\chi_{anchor,m}}{0.20},
0,1
\right)
$$

这个 fallback 机制是 Phase C v4 的关键。v3 的错误就是把一个 full-run quality 不合格的 sparse cue 强制用于所有 chunk。

### 5.4 Attention bias 保持 RFR-100 的 pair form 为默认

v3 的 best candidate 是 `flow_sem_veto + key bias`，但 full 失败。RFR-100 是 `old_dyn + pair bias`。Phase C v4 的默认不是 key bias，而是 pair bias：

$$
B_{qk}
=
\beta
\log
\left(
1-(1-D_q)D_k+\epsilon
\right)
$$

这保留了一个重要性质：dynamic query 本身不会被过度压制，主要抑制 static query attending dynamic key。key-only bias：

$$
B_k=\beta\log(1-D_k+\epsilon)
$$

会对所有 query 一视同仁地降低 dynamic key，可能过强，也可能破坏动态区域自身的内部一致性。v4 只在 pair bias 失败后才测试 key bias。

---

## 6. C4-3：Global-continuity stateful slice gate

Phase C v3 的 stateful slice gate 已经比 prefix gate 更强，但仍然 false positive。Phase C v4 将 slice gate 从 local Sim(3) gate 升级为 global-continuity gate。

### 6.1 Slice 设置

仍然使用 full no-control run 保存的 HMC state，在以下 chunk 分叉：

```text
chunk starts: 0, 5, 10, 15, 20, 25, 30
slice length: 128 frames
```

另外增加两个 256-frame slice：

```text
chunk starts: 10, 20
slice length: 256 frames
```

原因是 128-frame 仍然太短，容易被 local correction 误导；但 full runs 成本太高，所以 256-frame 中段 slice 是必要补充。

### 6.2 四个 slice 指标

#### 6.2.1 Local Sim(3) ATE

保留 v3 指标：

$$
ATE_{\text{local-sim3}}
$$

但它不再是唯一 gate。

#### 6.2.2 Global-frame ATE

使用 full no-control trajectory 的 global alignment transform，固定对 candidate slice 做评估，不允许每个 slice 重新 Sim(3) 对齐。记为：

$$
ATE_{\text{global-fixed}}
$$

通过条件：

$$
\Delta ATE_{\text{global-fixed}} \le 0
$$

在至少 `5 / 7` 个 128-frame slices 上成立。

#### 6.2.3 Endpoint drift

对 slice 结束帧：

$$
\Delta e_{end}
=
\lVert p^{cand}_{end}-p^{gt}_{end}\rVert
-
\lVert p^{base}_{end}-p^{gt}_{end}\rVert
$$

要求：

$$
\operatorname{Mean}(\Delta e_{end}) < 0
$$

且任一 slice：

$$
\Delta e_{end} < +0.15\text{ m}
$$

如果 candidate local ATE 变好但 endpoint drift 变差，说明它在局部形状上有效，但破坏 global trajectory。

#### 6.2.4 Path-length / scale continuity

定义 slice 内路径长度：

$$
L^{pred}=\sum_t \lVert p_{t+1}-p_t\rVert
$$

与 GT 或 base 比较：

$$
r_L=
\frac{L^{cand}}{L^{base}+\epsilon}
$$

要求：

$$
0.99 \le r_L \le 1.01
$$

如果 path-length ratio 偏离明显，full sequence ATE 很可能会失败。

### 6.3 C4-3 的通过标准

一个 candidate 必须同时满足：

| Gate | 通过标准 |
|---|---|
| Local Sim(3) slice | 至少 `5/7` 128-frame slices improved |
| Global fixed slice | 至少 `5/7` 128-frame slices improved |
| Endpoint drift | mean improved，任一 slice 不得 worse `>0.15 m` |
| Path-length ratio | 所有 128-frame slices 在 `[0.99,1.01]` 内 |
| 256-frame stateful | 两个 256-frame slices 至少一个 improved，另一个不得 worse `>0.10 m` |

这比 v3 gate 严很多。它的目标不是多筛出 candidate，而是少跑 false-positive full run。

---

## 7. C4-4：实验矩阵

Phase C v4 不再铺很大的 sweep。所有实验先过 C4-1/C4-3，再最多跑两个 full candidate。

### 7.1 Probe-only cue audit runs

这些只跑 no-control probe + cue computation，不注入 control：

| ID | Cue | 说明 |
|---|---|---|
| P-A | `old_dyn` | RFR reference cue |
| P-B | `gram4d` | internal Gram cue |
| P-C | `flow_sem_veto_proxy` | v3 failed cue |
| P-D | `flow_epi_raft` | true RAFT flow residual |
| P-E | `flow_epi_gmflow` | true GMFlow residual |
| P-F | `old_dyn_plus_flow_proxy` | blend candidate |
| P-G | `old_dyn_switch_flow_proxy` | fallback candidate |
| P-H | `old_dyn_plus_true_flow` | if true flow available |

通过 C4-1 才能进入 controlled run。

### 7.2 256-frame controlled gate

优先测试这些 256-frame candidates：

| ID | Cue | Calibration | Bias | Layer | Beta | 目的 |
|---|---|---|---|---|---:|---|
| G256-00 | old_dyn | native | pair | early | 1.0 | RFR reference |
| G256-01 | flow_proxy_calib | q=0.06 | pair | early | 0.75 | 测试 calibrated proxy 是否稳定 |
| G256-02 | flow_proxy_calib | q=0.06 | pair | early | 1.0 | 对齐 v3 beta 强度 |
| G256-03 | old_dyn + 0.25 flow_proxy | q=0.06 | pair | early | 1.0 | residual correction |
| G256-04 | old_dyn + 0.50 flow_proxy | q=0.06 | pair | early | 1.0 | 较强 blend |
| G256-05 | old_dyn switch flow_proxy | quality gate | pair | early | 1.0 | fallback |
| G256-06 | old_dyn + true_flow | q=0.06 | pair | early | 1.0 | true flow residual |
| G256-07 | true_flow_calib | q=0.06 | pair | early | 0.75 | true flow standalone |

这里不优先测试 key bias。只有 pair bias 的 best candidate 在 C4-3 过了，但 full 仍失败，才考虑 key。

### 7.3 Stateful global-continuity gate

对 G256 里最好的 2–3 个 candidate 跑 C4-3。只有通过 C4-3 的 candidate 才进入 full。

### 7.4 Full KITTI 01 candidates

最多跑两个 full candidate：

| ID | 条件 | 说明 |
|---|---|---|
| FC4-01 | C4-1/C4-3 综合分最高 | 主 candidate |
| FC4-02 | 与 FC4-01 cue 类型不同 | 例如一个 blend，一个 fallback；避免同质重复 |

Full 判断标准：

| 等级 | ATE 条件 | 解释 |
|---|---:|---|
| Minimum pass | `<41.0 m` | Phase C read-path-only 终于过最低门槛 |
| Formal pass | `<=40.8 m` | Phase C v4 成功，可以进入 Phase D 组合 |
| Strong pass | `<=40.5 m` | read-path cue 已有实质价值 |
| Main-candidate precondition | `<40.0 m` | 才有资格称为后续主模型组件 |

如果 FC4 仍停在 `41.0–41.3 m`，即便比 RFR-100 略好，也不算 Phase C 通过，只能作为 diagnostic signal。

---

## 8. C4-5：实现细节

### 8.1 True external flow residual

如果实现 RAFT 或 GMFlow，建议离线预计算相邻帧 flow：

```text
F_obs[t -> t+1]
F_obs[t+1 -> t]
forward-backward consistency
flow confidence
```

每个 chunk 加载 flow，不在 HMC 主循环里实时跑 flow 模型，避免实验时间不可控。

对于每个 pixel：

$$
F^{rigid}_{t\rightarrow t+1}(u)
=
\pi(T_{c_{t+1}\leftarrow w}X_t(u))-u
$$

残差：

$$
r_{flow}(t,u)
=
\frac{
\lVert F^{obs}_{t\rightarrow t+1}(u)-F^{rigid}_{t\rightarrow t+1}(u)\rVert_2
}{
\lVert F^{obs}_{t\rightarrow t+1}(u)\rVert_2+\lVert F^{rigid}_{t\rightarrow t+1}(u)\rVert_2+\epsilon
}
$$

flow reliability：

$$
c_{fb}(t,u)=
\mathbf{1}
\left[
\lVert F_{t\rightarrow t+1}(u)+F_{t+1\rightarrow t}(u+F_{t\rightarrow t+1}(u))\rVert
< \tau_{fb}
\right]
$$

最终：

$$
D_{flow}(t,u)
=
\sigma
\left(
\frac{r_{flow}(t,u)-\tau_r}{\sigma_r}
\right)
\cdot c_{fb}(t,u)
\cdot (1-C_{occ}(t,u))
$$

然后 pool 到 patch grid。

### 8.2 Structure protection

即使 Stage C 语义不可靠，也可以用轻量保护。结构保护不是主语义 prior，只是防止 cue 误伤稳定结构：

$$
D_i \leftarrow D_i(1-P_{struct,i})
$$

其中 $P_{struct}$ 可以来自：

- high `C_anchor` and low `C_dyn`
- reliable road/building/sidewalk mask if Stage C available
- overlap/reference token protection

第一版不要对 movable thing 强 cap，因为 parked cars 在 KITTI 里可能是有用 landmark。只在 flow residual 和 movable semantic 同时高时才强化 dynamic。

### 8.3 Global-slice evaluation implementation

为避免 v3 的 false positive，slice evaluation 要输出四个文件：

```text
slice_local_sim3.csv
slice_global_fixed.csv
slice_endpoint_drift.csv
slice_path_length_ratio.csv
```

其中 `slice_global_fixed.csv` 必须使用 full no-control alignment transform，不能每个 slice 重新优化 Sim(3)。

### 8.4 Full candidate logging

每个 full candidate 必须写：

```text
cue_quality_per_chunk.jsonl
cue_quality_summary.json
global_continuity_summary.json
hook_effect_summary.jsonl
hmc_correctness_summary.json
trajectory.txt
```

`cue_quality_summary.json` 至少包含：

```json
{
  "mean_dynamic_mass": ...,
  "chunk_dynamic_mass_p10": ...,
  "chunk_dynamic_mass_p90": ...,
  "corr_D_conf": ...,
  "anchor_collision": ...,
  "fragmentation_ratio": ...,
  "old_dyn_iou": ...,
  "fallback_rate": ...
}
```

如果 full candidate 失败，但这些统计不完整，该实验不作为有效结论。

---

## 9. 结果解释规则

Phase C v4 的关键是避免“差一点就过”的误读。

### 9.1 如果 calibrated flow / true flow 短程好但 full 差

如果新的 flow cue 仍然出现：

$$
ATE_{256} \text{ improves}
$$

但：

$$
ATE_{full} > ATE_{RFR}
$$

并且 global-continuity slice 已经有 warning，那么结论是：

> flow cue 可以做局部 motion detector，但不能作为 LoGeR frame-attention read-path controller 的全局 signal。

这时不要继续扫 beta。

### 9.2 如果 blend 比 old_dyn 好

如果：

$$
D_{blend}=(1-\lambda)D_{old}+\lambda D_{flow}
$$

full ATE 低于 RFR-100，说明 old_dyn 的 dense regularization 和 flow 的 precision 互补。下一步可以进入 Phase D，与 TTT branch0 write 组合。

### 9.3 如果 fallback 比 blend 好

如果 `old_dyn_switch_flow` 好于 fixed blend，说明 cue reliability 是关键。下一阶段应发展 chunk-level reliability model，而不是继续改 cue 本身。

### 9.4 如果 RFR-100 仍然是最优

如果所有 C4 full candidate 都不能超过 `41.0733 m`，则 Phase C read-path-only 基本到上限。此时结论不是“没有信号”，而是：

> 当前 LoGeR read-path frame attention gating 的 training-free motion signal 上限大约在 41.0 m 附近。进一步突破需要与 TTT branch0 write、SWA protection、或 learning-based reliability gate 组合。

这时不建议继续 Phase C v5。

---

## 10. 停止条件与下一步

Phase C v4 的停止条件非常明确。

### 10.1 Phase C v4 成功

若任一 full candidate 满足：

$$
ATE \le 40.8\text{ m}
$$

则 Phase C v4 通过，可以进入 Phase D：

```text
best read-path controller
+ TTT branch0 dynamic / update-needed write
+ sparse preserve
```

若达到：

$$
ATE \le 40.5\text{ m}
$$

说明 read-path signal 本身已经有较强价值，应优先做 ablation 和多序列验证。

### 10.2 Phase C v4 最低通过

若：

$$
40.8 < ATE < 41.0
$$

则算 minimum pass，但不能立刻称为主模型。可以进入 Phase D 组合，但必须保留 RFR-100 和 BL01 作为强 baseline。

### 10.3 Phase C v4 失败

若最优结果仍然：

$$
ATE \ge 41.0\text{ m}
$$

则停止 read-path-only tuning。后续不要再做：

```text
more beta sweep
more top-k around same cue
more short-prefix candidate promotion
more local Sim(3)-only slice gate
```

下一步应转向：

1. Phase D：RFR-100 read-path reference + TTT branch0 write control 的组合，但必须把它定义为 combination experiment，不再要求 Phase C 单独过关；
2. True external flow / semantic motion module，如果 C4 仍未实现真实 RAFT/GMFlow；
3. lightweight learned reliability gate，用已有 full/segment outcomes 监督 `g_m`，而不是继续手工规则。

---

## 11. 推荐执行顺序

本阶段建议按下面顺序执行，避免再浪费 full runs。

第一天只做 C4-0 correctness 和 C4-1 cue audit。不要跑任何 full controlled candidate。目标是确认新 instrumentation 不破坏 HMC，并且所有 candidate cue 的 full-run statistics 可用。

第二天做 C4-2 calibration 和 G256 candidate。只跑 256-frame，不跑 full。重点看 calibrated cue 是否仍然有 256-frame 改善，同时 dynamic mass、anchor collision、fragmentation 是否满足要求。

第三天做 C4-3 global-continuity stateful slices。这个 gate 会比 v3 更严格。如果 candidate 在 global fixed ATE / endpoint drift / path-length ratio 上失败，直接淘汰。

第四天最多跑两个 full candidates。任何没有通过 C4-1 和 C4-3 的 candidate 都不允许 full run。Full run 后如果没有 `<41.0 m`，Phase C read-path-only 停止。

---

## 12. 本阶段最可能的成功路线

基于 v2 / v3 的结果，我认为最有希望的不是 standalone `flow_sem_veto`，而是：

$$
D_{blend}=
\operatorname{Norm}
\left(
0.75D_{old}
+
0.25D_{flow,calib}
\right)
$$

配合：

```text
frame attention
pair bias
early layers
beta = 1.0
reference / high-anchor protection
```

原因是 RFR-100 的 old_dyn 已经证明 full-sequence safe，但不够强；flow_sem_veto 有短程和 slice signal，但 full 不安全。Blend 让 old_dyn 保持全局 regularization，flow 只提供局部 correction。

第二有希望的是：

$$
D_m =
g_mD_{flow,calib}
+
(1-g_m)D_{old}
$$

其中 $g_m$ 由 dynamic mass、anchor collision、fragmentation 控制。这个 fallback 版本可以避免 v3 那种 cue 太稀疏却全程强制使用的问题。

最不建议继续的是 standalone sparse flow key bias。v3 已经说明它可以在局部通过，但 full 会失败。

---

## 13. 总结

Phase C v3 失败的核心不是“flow_sem_veto 没有任何信号”，而是：

$$
\text{local continuation signal} \neq \text{full-sequence trajectory-safe signal}
$$

Phase C v4 因此要把问题从“哪个 cue 短程 ATE 最好”改成：

$$
\boxed{
\text{哪个 cue 在全序列上有足够 coverage，并且不破坏 global continuity}
}
$$

新的实验门槛是：

```text
HMC no-control correctness
→ full-run cue audit
→ calibration / fallback
→ global-continuity stateful slices
→ at most two full candidates
```

只有这样，后续 full KITTI 01 的结果才会有解释力。否则继续依赖 64/256 prefix 和 local Sim(3) slice，很容易再次筛出 v3 这种 false-positive candidate。

# TTT Write Controller 设计文档（v1）

> 本文档定义 **TTT Write Controller** 模块。模块以 **Semantic Prior Generator** 产生的 token-wise write prior 与 LoGeR 当前 chunk 的 TTT write cache 为输入，不改变 LoGeR 的 apply path，而是在 **update path** 中控制各 token 对 fast weights 的贡献，并最终完成 `W_m \rightarrow W_{m+1}` 的 memory write。本文档重点说明：运行时协议、update chain、token-wise weighting、block-level gain、special token 策略、以及与当前 `ttt.py / pi3.py` 代码路径的对接方式。

---

## 0. 一句话定位

**TTT Write Controller 的职责，是把 `A_tok` 转成真正作用在 LoGeR TTT update 链上的控制量，重新决定当前 chunk 中哪些 token 主导 fast weights 的写入，以及这一次 chunk 整体写入应有多保守。**

---

## 1. 模块目标

本模块需要完成下面四件事：

1. 接收 Semantic Prior Generator 输出的 token-wise write prior；
2. 将这份 prior 投影到 LoGeR 当前 TTT 的 token-wise coefficient 链上；
3. 以不改变 LoGeR 核心结构的方式，控制当前 chunk 的 update 聚合；
4. 产出新的 fast weights：

$$
W_m \rightarrow W_{m+1}
$$

其中，控制对象不是当前 chunk 的 apply 输出，而是：

- 当前 chunk 中 **哪些 token 更该写入 memory**；
- 当前 chunk 中 **哪些 token 应尽量不写入**；
- 当前 chunk **整体写入强度是否应降低或冻结**。

---

## 2. 设计原则

### 2.1 只控制 update，不改 apply

本模块不改变当前 chunk 读取旧 memory 的方式，也不改变 LoGeR 当前 chunk 的主干几何输出；它只作用在 update path。

### 2.2 尽量沿用 LoGeR 当前的 update 结构

不改动以下核心链：

- token raw contribution
- chunk 聚合
- zeropower / direction operator
- renorm

本模块只在以下两处插入控制：

1. **token-wise contribution weighting**
2. **block-level write gain / freeze**

### 2.3 先控制“谁主导更新结构”，再控制“整体写多少”

因此，本模块显式区分两类量：

- `\gamma_i`：结构权重，决定哪些 token 主导 pre-zeropower 聚合；
- `\lambda_write`：块级写入增益，决定 applied direction 最终偏转多少。

### 2.4 与 prior 的关系保持可解释

`A_tok` 是 write-allow，不直接等于最终 learning rate。需要经过：

$$
A_{tok} \rightarrow p \rightarrow \beta \rightarrow \gamma
$$

之后才进入 TTT update 聚合。

---

## 3. 代码接口与运行时位置

### 3.1 相关代码位置

结合当前 LoGeR 实现，本模块最相关的代码路径为：

- `loger/models/pi3.py`
- `loger/models/ttt.py`

### 3.2 现有语义

LoGeR 当前的 TTT 顺序是：

1. **apply**：用 `W_m` 读取 memory；
2. **update**：用当前 chunk 的 token 序列生成新的 fast-weight update。

因此，如果只关掉 `turn_off_ttt` 这一类 output gate，并不能阻止 memory 被写脏。真正的 write control 必须进入 update path。

### 3.3 运行时协议

建议采用 **cache-first, delayed-commit** 协议：

```text
chunk m
  1. LoGeR 用 W_m 跑 forward，并缓存 TTT update primitives
  2. Dynamic Cue Extractor 产生几何 cue
  3. Video Masklet Front-end 产生 semantic masklets
  4. Semantic Prior Generator 产生 A_tok
  5. TTT Write Controller 用 A_tok + cache 执行 delayed write-back
  6. 得到 W_{m+1}
chunk m+1 读取 W_{m+1}
```

这样保证：

- 当前 chunk 预测仍基于旧 memory；
- 当前 chunk 结束后，才由语义+几何先验决定它往未来留下怎样的记忆。

---

## 4. 输入定义

### 4.1 来自 Semantic Prior Generator 的输入

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| token write allow | `A_tok` | `[L_tok]` | 最终 token prior |
| patch prior | `A_patch-flat` | `[L_patch]` | patch token prior |
| special prior | `A_special` | `[L_special]` | special token prior |
| suppression | `S_tok` | `[L_tok]` | `1 - A_tok` |
| prior debug | `PriorDebug` | 结构体 | 可选诊断信息 |

### 4.2 来自 LoGeR 的 TTT write cache

对每个 TTT 层 `l`，缓存：

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| token type | `TokenType` | `[L_tok]` | patch / reg / role |
| `eta0_l` | `\Eta_l^{(0)}` | `[H_ttt, L_tok, 1]` | `w0` 分支 token 系数 |
| `eta1_l` | `\Eta_l^{(1)}` | `[H_ttt, L_tok, 1]` | `w1` 分支 token 系数 |
| `eta2_l` | `\Eta_l^{(2)}` | `[H_ttt, L_tok, 1]` | `w2` 分支 token 系数 |
| `K_l` | `K_l` | `[H_ttt, L_tok, d_k]` | key-like 原语 |
| `V_l` | `V_l` | `[H_ttt, L_tok, d_v]` | value-like 原语 |
| `U_l` | `U_l` | `[H_ttt, L_tok, d_u]` | `w1` 分支 hidden 原语 |
| `A_l` | `A_l` | `[H_ttt, L_tok, d_a]` | `w0` 分支反传原语 |
| `B_l` | `B_l` | `[H_ttt, L_tok, d_b]` | `w2` 分支反传原语 |
| `W_prev_l` | `W_{m,l}` | 结构体 | 当前 chunk 开始前 fast weights |

### 4.3 可选输入

| 名称 | 记号 | Shape | 含义 |
|---|---|---:|---|
| chunk risk summary | `R_chunk` | `[R]` | 例如动态占比、low-value stuff 占比 |
| layer gain prior | `g_l` | `[L_ttt]` | 层级保守性控制 |
| head gain prior | `g_{l,h}` | `[L_ttt, H_ttt]` | head 级控制 |

---

## 5. 输出定义

### 5.1 控制中间量

| 名称 | Shape | 含义 |
|---|---:|---|
| `Beta_write` | `[L_ttt, 3, H_ttt, L_tok]` | 先验修正后的 token 系数 |
| `Gamma_write` | `[L_ttt, 3, H_ttt, L_tok]` | 归一化后的结构权重 |
| `Lambda_write` | `[L_ttt, 3, H_ttt]` | block-level write gain |
| `Freeze_write` | `[L_ttt, 3, H_ttt]` | 是否冻结写入 |

### 5.2 更新结果

| 名称 | Shape | 含义 |
|---|---:|---|
| `W_next` | 分层结构体 | 更新后的 fast weights |
| `WriteDebug` | 结构体 | token / head / layer 级诊断 |

---

## 6. TTT update chain 的数学对象

### 6.1 单 token 原始更新贡献

对第 `m` 个 chunk、第 `l` 个层、第 `i` 个 token、第 `h` 个 head，记分支 `r \in \{0,1,2\}` 的单 token 原始更新矩阵为：

$$
J_{m,l,i}^{(r,h)}
$$

三条分支具体写为：

$$
J_{m,l,i}^{(1,h)} = u_{m,l,i}^{(h)\top} v_{m,l,i}^{(h)}
$$

$$
J_{m,l,i}^{(0,h)} = k_{m,l,i}^{(h)\top} a_{m,l,i}^{(h)}
$$

$$
J_{m,l,i}^{(2,h)} = k_{m,l,i}^{(h)\top} b_{m,l,i}^{(h)}
$$

### 6.2 LoGeR 当前已有 token 系数

当前代码已有：

$$
\eta_{m,l,i}^{(r,h)}
$$

它来自 `lr0/lr1/lr2`，控制当前 token 在 pre-zeropower 聚合中占多大权重。

### 6.3 当前 chunk 的 pre-zeropower 聚合矩阵

LoGeR 的 baseline 聚合是：

$$
\tilde G_{m,l}^{(r,h)}=
\sum_{i\in C_m} \eta_{m,l,i}^{(r,h)} J_{m,l,i}^{(r,h)}
$$

TTT Write Controller 的任务，就是把 `A_tok` 接到这条链上。

---

## 7. 从 `A_tok` 到 token-wise prior `p`

### 7.1 最简单版本

第一版可以直接令：

$$
p_{m,l,i}^{(r,h)} = A_{tok}[i]
$$

即：所有层、所有 head、所有分支共享同一份 token prior。

### 7.2 带层 / head / 分支偏置的版本

若想更灵活，可定义：

$$
p_{m,l,i}^{(r,h)} = A_{tok}[i] \cdot g_l \cdot g_{l,h} \cdot g_{r}
$$

其中：

- `g_l`：层级保守性；
- `g_{l,h}`：head 级偏置；
- `g_r`：分支级偏置。

第一版建议：

$$
g_l = g_{l,h} = g_r = 1
$$

---

## 8. token-wise contribution weighting

### 8.1 先验修正后的 token 系数

定义：

$$
\beta_{m,l,i}^{(r,h)} = p_{m,l,i}^{(r,h)} \cdot \eta_{m,l,i}^{(r,h)}
$$

这表示：

- 若 `A_tok[i]` 高，则当前 token 仍然保留对 update 的贡献；
- 若 `A_tok[i]` 低，则它在当前 chunk 的写入结构中被弱化。

### 8.2 chunk 内归一化

为了让控制更偏向“相对结构”而不是“统一缩放”，定义：

$$
\gamma_{m,l,i}^{(r,h)}=
\frac{\beta_{m,l,i}^{(r,h)}}{\frac{1}{|C_m|}\sum_j \beta_{m,l,j}^{(r,h)}+\epsilon}
$$

### 8.3 新的 pre-zeropower 聚合矩阵

于是有：

$$
\tilde G_{m,l}^{(r,h,ctrl)}=
\sum_{i\in C_m} \gamma_{m,l,i}^{(r,h)} J_{m,l,i}^{(r,h)}
$$

这一步的含义是：

**TTT Write Controller 首先决定的是：当前 chunk 中谁更主导 update direction 的结构。**

---

## 9. block-level write gain

token 级结构重加权之后，还需要决定“当前 chunk 整体写多少”。

### 9.1 suppression occupancy

定义当前 chunk 的 token 抑制占比：

$$
\rho_{suppr} = \frac{1}{L_{tok}}\sum_{i=1}^{L_{tok}} (1-A_{tok}[i])
$$

也可以只对 patch token 计算：

$$
\rho_{suppr}^{patch} = \frac{1}{L_{patch}}\sum_{i=1}^{L_{patch}} (1-A_{patch-flat}[i])
$$

### 9.2 chunk 风险项

再定义两个额外统计：

- high-dynamic risk 占比：`\rho_dyn`
- low-value stuff 占比：`\rho_lowsem`

### 9.3 write gain

可定义：

$$
\lambda_{write}(l,h,r)=
\operatorname{clip}\Big(
1 - \kappa_s \rho_{suppr}
    - \kappa_d \rho_{dyn}
    - \kappa_l \rho_{lowsem},
\lambda_{min},
1\Big)
$$

这样：

- 当前 chunk 越被高风险区域主导，整体写入越保守；
- 即使少量 token 仍然有高 `A_tok`，整个 chunk 的 applied direction 也不会过猛。

### 9.4 freeze 条件

若满足：

$$
\rho_{suppr} > \tau_{freeze}
$$

或语义 / 几何整体质量极差，则可直接：

$$
Freeze_{write}(l,h,r)=1
$$

这时：

$$
W_{m+1,l}^{(r,h)} = W_{m,l}^{(r,h)}
$$

即：当前 chunk 只读不写。

---

## 10. post-zeropower applied direction

控制器并不改变 LoGeR 的 direction operator。仍然沿用：

$$
G_{m,l,app}^{(r,h)} = \mathcal U_{dir}\big(\tilde G_{m,l}^{(r,h,ctrl)}\big)
$$

最后，若未冻结，则：

$$
W_{m+1,l}^{(r,h)}=
Renorm\Big(
W_{m,l}^{(r,h)} + \lambda_{write}(l,h,r)\, G_{m,l,app}^{(r,h)}
\Big)
$$

因此完整控制链为：

$$
A_{tok}
\rightarrow p
\rightarrow \beta
\rightarrow \gamma
\rightarrow \tilde G^{ctrl}
\rightarrow G_{app}
\rightarrow \lambda_{write}
\rightarrow W_{m+1}
$$

---

## 11. Special token 策略

### 11.1 patch token

直接使用 `A_patch-flat`。

### 11.2 register token

若 Semantic Prior Generator 已输出 `A_special`，则直接使用；否则采用 chunk 统计构造一个统一值：

$$
A_{reg}=\operatorname{clip}(1-\kappa_r\rho_{suppr}, a_{min}^{reg}, 1)
$$

### 11.3 role token

角色 token 通常承载 chunk 内位置语义，建议更保守：

$$
A_{role}=\operatorname{clip}(1-\kappa_{role}\rho_{dyn}, a_{min}^{role}, 1)
$$

### 11.4 special token 的理由

如果 patch token 被 suppress，但 special token 完全不 suppress，那么动态上下文仍可能通过 register / role token 写进 fast weights。因此 special tokens 不能默认全放行。

---

## 12. 与当前代码的对接建议

### 12.1 `pi3.py` 侧：缓存 update primitives

在当前 chunk forward 时，需要缓存：

- `eta0/1/2`
- `K / V / U / A / B`
- `W_prev`
- `TokenType`
- `PatchMeta`

### 12.2 `Semantic Prior Generator` 完成后

得到：

- `A_tok`
- `A_special`
- `A_patch-flat`

### 12.3 `ttt.py` 侧：替换聚合系数

当前 `eta` 不再直接用于聚合，而是：

1. 先与 `p` 相乘得到 `beta`；
2. 归一化成 `gamma`；
3. 用 `gamma` 替代原先的 token-wise 系数进入 batched matmul 聚合。

### 12.4 不建议的做法

不建议仅在 `pi3.py` 外层对 TTT residual 做 mask 或 gate，因为那并不会阻止 memory write。

---

## 13. Debug 与诊断输出

推荐输出：

```text
WriteDebug = {
  A_tok,
  Beta_write,
  Gamma_write,
  Lambda_write,
  Freeze_write,
  rho_suppr,
  rho_dyn,
  rho_lowsem,
  token_type_breakdown,
  layer_head_write_energy,
}
```

特别建议可视化：

1. `A_tok` 映射回 patch grid；
2. `Gamma_write` 的 patch 平均值；
3. 每层每头的 `Lambda_write`；
4. freeze 触发比例。

---

## 14. 实现优先级

### Phase 1：最小 write control 原型

- 共享 prior：`p = A_tok`
- `beta = p * eta`
- `gamma` 归一化
- 替换 pre-zeropower 聚合
- 不做 block gain
- 不做 freeze

### Phase 2：加入 block-level write gain

- 基于 `rho_suppr / rho_dyn / rho_lowsem` 计算 `lambda_write`
- 保留 patch / special token 区分

### Phase 3：加入 freeze 和层 / 头级策略

- `Freeze_write`
- `g_l / g_{l,h} / g_r`
- 更细粒度的 special token policy

---

## 15. 推荐默认超参数

| 参数 | 建议值 | 含义 |
|---|---:|---|
| `\lambda_{min}` | 0.0～0.2 | block gain 下限 |
| `\tau_{freeze}` | 0.85～0.95 | 冻结阈值 |
| `a_{min}^{reg}` | 0.2～0.5 | register token 最低放行 |
| `a_{min}^{role}` | 0.2～0.5 | role token 最低放行 |
| `\kappa_s,\kappa_d,\kappa_l` | 手工设定 | block gain 组合权重 |
| `\epsilon` | `1e-6` | 归一化稳定项 |

---

## 16. 常见失败模式与缓解策略

### 16.1 `A_tok` 太稀疏，导致整块几乎不写

**缓解**：先用 clipping 保证最小 prior；block gain 不要一开始就过低。

### 16.2 dynamic risk patch 被压了，但 special token 仍把上下文写进 memory

**缓解**：显式构造 `A_special`，不要默认 special token 全放行。

### 16.3 prior 抖动导致每个 chunk 写入结构大幅震荡

**缓解**：对 `A_tok` 或 `Lambda_write` 做 chunk 级平滑，或只在 Phase 2 再加入 `lambda_write`。

### 16.4 只做 output gate，却发现 memory 仍被污染

**缓解**：必须进入 update path；不要把 residual scaling 误认为 write control。

---

## 17. 最终接口摘要

### 输入

- `A_tok: [L_tok]`
- `A_patch-flat: [L_patch]`
- `A_special: [L_special]`
- `S_tok: [L_tok]`
- 每层 `TTT write cache`

### 输出

- `Beta_write`
- `Gamma_write`
- `Lambda_write`
- `Freeze_write`
- `W_next`
- `WriteDebug`

### 一句话总结

**TTT Write Controller 的核心作用，是把 Semantic Prior Generator 产生的 token-wise write prior 真正接到 LoGeR 的 update path 中：先重排当前 chunk 内谁更主导更新方向，再决定这一整块最终写多少，从而控制当前 chunk 往未来 fast weights 里留下怎样的记忆。**

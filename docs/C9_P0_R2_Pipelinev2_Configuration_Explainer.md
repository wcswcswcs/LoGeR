# C9_P0_R2 配方详解：从一段视频到 online TTT-write

日期：2026-05-26  
对象：只懂基础神经网络的读者。默认你知道“向量、attention、权重、softmax”这些概念，但不要求你了解 LoGeR、Pipeline v2、HMC、TTT 或 ACL2。

本文解释当前历史最好可部署 online TTT-write 配方：

```text
C9_P0_R2
ATE = 33.7629421029m
```

它不是 v41 新结果，而是 v15/v16 中锁定、复现、计数过的历史最佳在线 TTT 写入配方。

对应落盘 run：

```text
v15 repeat:
results/kitti01_hmc_v2/acl2_v15_ttt_repro_causal_sandbox_target25/
phase0_repro/V15_P0_A2_C9_REPEAT_no_state_save_SWKS3

v16 locked boundary:
results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25/
phase0_boundary/V16_P0_R2_C9_locked_exact_merge_input_SWKS3
```

核心配置：

```text
results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25/
phase0_boundary/V16_P0_R2_C9_locked_exact_merge_input_SWKS3/hmc_config.yaml
```

---

## 0. 先说人话：C9_P0_R2 到底在干什么

LoGeR 在 KITTI 01 上做的是视觉里程计：输入一段车载相机视频，输出每一帧相机的位置和朝向，最后形成一条轨迹。

问题是：视频里并不是所有图像区域都适合当“长期几何记忆”。

比如：

```text
适合记住：
    道路边界、建筑边缘、护栏、稳定地面纹理。

不适合强记住：
    移动车辆、遮挡边缘、低纹理天空、反光、远处模糊区域。
```

如果模型把不稳定区域也写进自己的在线记忆，后面窗口会继续读到这些坏记忆，轨迹误差会慢慢放大。

`C9_P0_R2` 做的事可以概括成三句话：

```text
1. 当前窗口推理时：
   少看那些不像过去稳定结构的 patch。

2. 当前窗口结束写记忆时：
   稳定 patch 多写，不稳定 patch 少写。

3. 如果某些 patch 的 TTT 更新方向和整体更新方向打架：
   把它们当成轻微 negative evidence，别让它们污染 fast weights。
```

这里的 patch 就是把图像切成小块后得到的视觉 token。每个 token 是一个向量。

---

## 1. C9_P0_R2 这个名字什么意思

```text
C9:
    v15/v16 中锁定的 C 系列第 9 个控制配方。
    当前作为历史 best deployable online TTT-write。

P0:
    Phase 0 boundary / reproducibility。
    这是复现和边界确认阶段，不是 v41 的新搜索阶段。

R2:
    第 2 个 locked repeat / locked boundary run。
```

所以：

```text
C9_P0_R2 = 历史 C9 配方，在 Phase 0 locked boundary 中第 2 次有效 repeat。
```

它的关键指标：

| Run | ATE | Rot | RPE_t | FinalErr | `[200,300)` | `[400,600)` | hmc rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `H9_P0_R2` | `34.1257769401` | `6.5414` | `92.4053` | `6.189399` | `74.409927` | `44.353638` | `38` |
| `C9_P0_R2` | `33.7629421029` | `6.5259` | `92.3871` | `5.666384` | `76.102136` | `41.896364` | `38` |

ATE 是 absolute trajectory error，整条估计轨迹和真值轨迹的整体差距，单位是米。越小越好。

对比 H9：

```text
全局 ATE:
    H9 = 34.1257769401m
    C9 = 33.7629421029m
    C9 改善约 0.3628m

[400,600) 段:
    H9 = 44.353638m
    C9 = 41.896364m
    C9 改善约 2.4573m

[200,300) 段:
    H9 = 74.409927m
    C9 = 76.102136m
    C9 回退约 1.6922m
```

所以 C9 是当前历史最好可部署 online TTT-write，但它不是完美解，也没有达到 Target-30。

---

## 2. 先建立四个基本概念

如果这四个概念清楚，后面配置就会顺很多。

### 2.1 Token：图像小块变成向量

模型不是直接处理“整张图片”，而是把图像切成许多小块，每个小块变成一个向量。

```text
图像 patch -> token 向量
```

一个 token 可能对应：

```text
道路上的一小块纹理
汽车边缘的一小块区域
天空的一小块区域
建筑窗户的一小块区域
```

### 2.2 Attention：一个 token 从别的 token 取信息

普通 attention 可以简化成：

```text
query:
    当前 token 发出的“我要找什么信息”的向量。

key:
    其他 token 的“我是什么信息”的向量。

value:
    其他 token 真正提供的信息内容。
```

attention 权重通常来自：

```text
score(q, k) = dot(q, k) / sqrt(d)
weight(q, k) = softmax(score(q, k))
```

如果某个 key 的 score 很高，当前 query 就会更多读取它的 value。

C9 的读控制就是在这个 score 上加一个负 bias，让稳定 token 少读高风险 token。

### 2.3 TTT fast weights：测试时临时写入的小记忆

TTT 是 test-time training / test-time tuning 的缩写。这里不要理解成“重新训练整个模型”。在 LoGeR 里，它更像是模型在测试视频上边跑边维护的一组快速权重。

C9 里关心三组 fast weights：

```text
w0, w1, w2
```

你可以把它们理解成：

```text
模型看完当前窗口后，写下的一份临时几何笔记。
下一窗口会带着这份笔记继续推理。
```

坏处是：如果笔记里写进了动态车辆、遮挡、天空这类不稳定东西，下一窗口也会被影响。

### 2.4 HMC：管理“看什么”和“写什么”的控制器

HMC 是 Hybrid Memory Controller，混合记忆控制器。

它主要管两件事：

```text
读：
    当前窗口推理时，哪些 token 可以被 attention 更多读取。

写：
    当前窗口结束时，哪些 token 可以更强地写入 TTT fast weights。
```

C9 的核心就是 HMC 的读写策略。

### 2.5 先把后面会出现的符号说清楚

后面会频繁出现几个名字。它们不是同一个东西：

| 符号 / 名字 | 它是什么 | 值越大表示什么 | 主要用在哪里 |
|---|---|---|---|
| `D_g` | C9 的 ACL2 read cue，也叫 read risk / read_patch | 当前 patch 越不像过去稳定结构，越可疑 | frame attention 读控制；TTT 写入反向项 |
| `D_tok` | 把 patch 级 `D_g` 展开到 token 级后的风险 | 当前 token 越可疑 | attention bias、SWA、TTT risk 输入 |
| `C_dyn` | Stage B 的几何/attention 动态风险 | 越像动态或不稳定几何 | Stage D 几何写入可靠度 |
| `C_stat` | Stage B 的静态结构置信度 | 越像稳定静态结构 | Stage D 几何写入可靠度 |
| `C_occ` | 遮挡风险 | 越可能是遮挡/不可见问题 | 降低写入可靠度 |
| `C_unc` | 几何不确定性 | 几何越不确定 | 降低写入可靠度 |
| `G_write_geo` | Stage B 输出的几何写入允许度 | 越适合写入 | Stage D 原始 eligibility |
| `stage_d_score` / `base_score` | HMC 写入 override 里使用的 Stage D 基础分数 | 越像低动态、可写的几何证据 | 和 `sqrt(1-D_g)` 相乘 |
| `A_tok` | 最终给 TTT replay 用的 token multiplier | token 写入学习率越高 | 乘到 TTT replay 的 `lr` 上 |

最容易混淆的是：

```text
D_g 不是 Stage B 的 C_dyn。

D_g:
    来自模型内部 query 和过去帧 query centroid 的相似度。
    它问的是“当前 patch 像不像过去稳定结构”。

C_dyn:
    来自 Stage B 的几何/attention 动态估计。
    它问的是“当前 patch 从几何和 attention 看像不像动态/不稳定”。
```

C9 的写入分数把二者合起来：

```text
低 C_dyn / 高 stage_d_score:
    说明几何上更像可写。

低 D_g:
    说明内部 query 上更像过去稳定结构。

二者都好，才更适合写入 TTT。
```

---

## 3. Pipeline v2 的完整时间线

LoGeR 不会一次性处理整段 KITTI 01。它把视频切成窗口：

```text
window_size = 32
chunk_size = 32
overlap_size = 3
chunk_overlap = 3
```

意思是：

```text
每个 chunk 看 32 帧；
相邻 chunk 之间共享 3 帧；
共享的 3 帧用来把两段轨迹接得更稳。
```

现在假设正在处理第 `c` 个 chunk。C9 的流程如下。

### 3.1 输入当前 chunk 和上一轮记忆

进入当前 chunk 时，pipeline 拿到：

```text
当前 32 帧图像
上一 chunk 提交的 TTT fast weights
上一 chunk 的 overlap / local history 信息
轨迹 merge state
```

C9 还有一个 reset 节奏：

```text
reset_every = 5
```

当全局 chunk 编号是 5、10、15 这类 5 的倍数时，pipeline 会清掉 TTT fast weights，但保留局部 history。

这不是用真值重置，也不是后处理改轨迹。它只是防止 fast weights 无限期积累旧错误。

### 3.2 Pass 1：probe，先正常跑一遍

第一遍叫 probe pass。

它先不施加 C9 的读控制，按当前已有记忆正常跑一遍 LoGeR，得到：

```text
1. 初始几何结果：
   当前窗口的相机位姿、patch 几何、confidence 等。

2. 模型内部特征：
   包括 query/key 等 attention 特征。

3. TTT write cache：
   记录这次窗口如果要更新 fast weights，需要用到的 k、v、lr、old weights。
```

probe pass 的作用是“先观察，不干预”。

### 3.3 构造风险分数：哪些 patch 不可靠

probe pass 后，HMC 会构造风险分数。

C9 里最重要的风险分数叫：

```text
D_g
```

可以先把它理解成：

```text
D_g 越接近 0:
    这个 patch 越像过去出现过的稳定结构。

D_g 越接近 1:
    这个 patch 越不像过去稳定结构，越可疑。
```

后面第 4 节会详细讲 `D_g` 怎么算。

同时，Stage B 也会构造另一套几何动态 cue：

```text
C_stat, C_dyn, C_occ, C_unc, C_anchor, G_write_geo
```

这套 cue 不直接等于 `D_g`。它主要负责回答：

```text
这个 patch 从几何上是否稳定、是否低动态、是否低遮挡、是否低不确定？
```

后面写入 TTT 时，C9 会同时用：

```text
Stage B / Stage D:
    决定“几何上适不适合写”。

D_g:
    决定“像不像过去稳定结构，是否应当避开”。
```

### 3.4 Pass 2：controlled，带着控制再跑一遍

第二遍叫 controlled pass。

这次会把 `D_g` 接入 frame attention。高风险 patch 会更难被稳定 patch 读取。

这一步影响的是当前窗口输出的轨迹。

### 3.5 当前窗口输出来自 controlled pass

C9 的当前窗口轨迹结果来自 controlled pass。

也就是说：

```text
当前输出:
    受 C9 读控制影响。
```

### 3.6 下一窗口记忆写入来自 probe cache

但 C9 的下一窗口 TTT memory 不是直接从 controlled pass 写。

它使用：

```text
hmc_commit_mode = probe_ttt_write
```

意思是：

```text
当前窗口输出:
    用 controlled pass 的结果。

下一窗口 TTT memory:
    用 probe pass 的原生 TTT cache，
    再应用 C9 的写入策略重新 replay 出来。
```

这点非常重要。它避免 controlled pass 的改动直接自我写入下一窗口，减少“控制信号自我放大”的风险。

---

## 4. `D_g` 怎么构造：C9 的 read cue

C9 的核心 read cue 配置是：

```text
read_cue_source = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
```

这个名字很长，不要背。它的意思是：

```text
用模型内部 global geometry attention 的 query 特征，
比较当前 patch 和过去帧的稳定结构像不像。
如果不像，就给它更高风险。
```

拆开看：

| 名字片段 | 含义 |
|---|---|
| `acl2` | 项目内部 cue family 名 |
| `gg` | 使用 global geometry attention 的内部特征 |
| `qq` | 用 query 和 query 比相似度 |
| `low` | 低相似度代表高风险 |
| `g2_3` | 使用第 2 到第 3 组 global 特征 |
| `past_only` | 只和过去帧比较，不看未来 |
| `headmean` | 对支持帧、层/head 统计做平均 |
| `robustq` | 用稳健分位数归一化到 0 到 1 |

下面讲具体公式。

### 4.1 当前 patch 的 query

对第 `t` 帧、第 `p` 个 patch、第 `l` 层，模型内部有一个 query 向量：

```text
q[t, l, p]
```

它表示：这个 patch 在这一层里“想找什么样的信息”。

这里的 query 不是最终输出轨迹，也不是语义标签。它是 LoGeR backbone 中 global geometry attention 的内部特征。C9 选择 query 而不是 RGB 像素，是因为 query 已经混入了模型对几何匹配、时序关系、局部结构的表示。

`g2_3` 的含义是使用一组中前层 global feature。它不是第 2 帧到第 3 帧，而是内部 global geometry 特征的 layer window。这个层段比纯浅层纹理更接近几何，又没有深层特征那么晚，因此适合作为“是否像过去结构”的 cue。

### 4.2 过去帧的 query centroid

对过去某一帧 `s`，把这一帧所有 patch 的 query 做平均：

```text
q_cent[s, l] = mean_p q[s, l, p]
```

`q_cent[s, l]` 可以理解成：

```text
过去第 s 帧在第 l 层的整体结构特征。
```

### 4.3 只和过去帧比较

因为 C9 是在线方法，处理第 `t` 帧时不能偷看未来。

所以支持集合是：

```text
S_t = {0, 1, 2, ..., t - 1}
```

如果 `t = 0`，没有过去帧，就退化成零风险或默认风险。

### 4.4 计算相似度

当前 patch 和过去帧 centroid 的相似度：

```text
sim[t, l, p, s] = dot(q[t, l, p], q_cent[s, l])
```

再把相似度映射到 0 到 1：

```text
sim01[t, l, p, s] = clamp((sim[t, l, p, s] + 1) / 2, 0, 1)
```

如果相似度高，说明当前 patch 像过去稳定结构。

如果相似度低，说明当前 patch 可疑。

### 4.5 低相似度变成高风险

配置里的 `low` 表示：

```text
低相似度 -> 高风险
```

所以原始风险可以写成：

```text
raw_D_g[t, p] = 1 - mean_{l, s in S_t} sim01[t, l, p, s]
```

然后做稳健分位数归一化：

```text
D_g[t, p] = RobustQuantile01(raw_D_g[t, p])
```

最终：

```text
D_g = 0:
    很像过去稳定结构。

D_g = 1:
    很不像过去稳定结构。
```

实现里这个 `D_g` 会以两种形态存在：

```text
read_patch:
    patch 级，长度约等于 T * H_patch * W_patch。
    这里 T 是当前 chunk 的帧数。

D_tok:
    token 级，把 read_patch 填回 LoGeR 的 token 序列。
    patch token 拿对应的 D_g；
    special / reference token 默认风险为 0 或被保护。
```

为什么要有两个形态：

```text
attention hook:
    需要 token 级 D_tok，因为 attention 是 token 对 token。

write score:
    先在 patch 级组合分数，再填回 token 级 A_tok。
```

### 4.6 为什么这个 cue 有用

车载视频里，稳定结构一般会在多个相邻帧里重复出现。道路、建筑、护栏不会突然消失，也不会像车一样快速换位置。

所以：

```text
如果一个 patch 的内部 query 特征一直和过去结构相似，
它更可能是稳定几何。

如果一个 patch 的 query 特征和过去结构差异很大，
它更可能是动态物体、遮挡、反光、低纹理或几何异常。
```

C9 不需要知道这个 patch 是 car、sky、road 还是 building。它只问：

```text
它像不像过去稳定结构？
```

还要注意：`D_g` 不是“动态物体概率”。一个 patch 的 `D_g` 高，可能是车，也可能是天空、反光、遮挡边缘、低纹理区域、或者模型内部 query 不稳定的地方。它表达的是“对几何记忆不可靠”，不是语义类别。

---

## 5. Stage B 的动态 cue：给写入可靠度做底座

除了 `D_g`，C9 还会用 Stage B 的几何动态估计。

Stage B 会估计几个量：

| 记号 | 含义 |
|---|---|
| `C_stat` | 像稳定静态结构的程度 |
| `C_occ` | 遮挡/不可见风险 |
| `C_unc` | 几何不确定性 |
| `D_exp` | 显式几何动态风险 |
| `D_imp` | attention 暗示的隐式动态风险 |
| `C_dyn` | 融合后的动态风险 |
| `C_anchor` | 稳定锚点置信度 |

显式动态风险简化成：

```text
D_exp = clamp(alpha_1 * (1 - C_stat_geom) - alpha_3 * C_occ, 0, 1)
```

C9 使用：

```text
alpha_1 = 0.8
alpha_3 = 0.5
```

直觉：

```text
越不像稳定几何结构，动态风险越高。
遮挡区域需要单独处理，避免把纯遮挡误当成可写动态证据。
```

隐式动态风险 `D_imp` 由模型内部 attention 支持。C9 使用 calibrated soft-or：

```text
D_imp_cal = clamp((D_imp - q50(D_imp)) / (q95(D_imp) - q50(D_imp) + eps), 0, 1)

C_anchor_exp = C_stat * (1 - D_exp) * (1 - C_unc)

g_imp = floor + (1 - floor) * (1 - C_anchor_exp)

D_imp_weighted = implicit_weight * g_imp * D_imp_cal

C_dyn = 1 - (1 - D_exp) * (1 - D_imp_weighted)
```

C9 参数：

```text
implicit_weight = 0.5
implicit_gate_floor = 0.25
```

这一步的作用是给“写入可靠度”提供底座：低动态、低不确定、稳定的 patch 更适合写入 fast weights。

### 5.1 `G_write_geo` 是怎么来的

Stage B 最后会把这些几何 cue 合成一个写入允许度：

```text
z_geo =
    lambda_s * C_stat
    + lambda_a * C_anchor
    - lambda_d * C_dyn
    - lambda_o * C_occ
    - lambda_u * C_unc

G_write_geo = sigmoid(z_geo)
```

C9 使用的权重是：

```text
lambda_s = 1.2
lambda_a = 0.8
lambda_d = 1.2
lambda_o = 0.3
lambda_u = 0.3
```

逐项解释：

```text
C_stat 高:
    像静态结构，加分。

C_anchor 高:
    像稳定锚点，加分。

C_dyn 高:
    像动态或不稳定，扣分。

C_occ 高:
    遮挡风险高，扣分。

C_unc 高:
    几何不确定，扣分。
```

所以 `G_write_geo` 不是语义分数，而是纯几何/attention 侧的“这个位置适不适合作为写入证据”。

### 5.2 C9 的 Stage C 关闭后，Stage D 还在做什么

C9 的配置是：

```text
stage_c_mode = none
semantic_prior_mode = spg_v2
spg_use_g_write_geo = 1
```

这容易让人误会：Stage C 关闭是不是 Stage D 就没东西可做？

实际不是。

当 `stage_c_mode=none` 时，pipeline 会构造一个空的 masklet output。也就是说：

```text
没有语义 mask；
没有 car / sky / road 的 runtime action；
没有 SemanticKITTI label。
```

但 Stage D 的 `SemanticPriorGenerator` 仍然可以使用 Stage B 的 `G_write_geo`：

```text
Elig_pix = G_write_geo
```

因为没有 masklet，语义分支退化成：

```text
A_pix = Elig_pix
V_sem_pix = 1
```

通俗讲：

```text
C9 的 Stage D 不是在用语义类别；
它是在把 Stage B 的几何写入允许度整理成 token prior 的初始形态。
```

后面 HMC 还会再用 `stage_d_x_dg_inv_sqrt` 覆盖一次写入分数，所以 Stage D 的结果是“底座”，不是最终 TTT multiplier。

---

## 6. 读控制：当前窗口少读高风险 patch

C9 的读路径配置：

```text
read_path = frame
enable_frame_read_control = 1
frame_bias_mode = pair
read_layer_mode = all
```

意思是：

```text
把 D_g 接入 frame attention；
所有相关 frame attention 层都使用这个控制；
控制方式是 query-key pair bias。
```

### 6.1 从 patch 风险到 token 风险

`D_g[t, p]` 是 patch 级风险。模型内部 attention 用的是 token，所以 HMC 会把它展开成 token 风险：

```text
D_tok[i]
```

其中 `i` 表示某个 token。

特殊 token / reference token 会被保护：

```text
read_protect_ref = 1
```

得到真正用于 attention bias 的风险：

```text
D = D_tok * (1 - P_ref)
```

`P_ref = 1` 的 token 风险会被压掉，避免 reference token 被误伤。

这里的 special / reference token 可以理解成模型内部用于全局汇总、相机、寄存器或结构控制的非图像 patch token。它们没有明确对应某个图像小块。如果把 patch 风险粗暴套到这些 token 上，容易伤到模型的全局组织能力，所以 C9 让它们保持低风险或直接保护。

### 6.2 pair bias 公式

对一个 query token `q` 和一个 key token `k`：

```text
D_q = query token 的风险
D_k = key token 的风险
```

C9 使用：

```text
keep(q, k) = 1 - (1 - D_q) * D_k

bias(q, k) = beta_frame * log(clamp(keep(q, k), 1e-4, 1))
```

然后：

```text
attention_score_controlled(q, k)
    = attention_score_native(q, k) + bias(q, k)
```

因为 `keep <= 1`，所以 `log(keep) <= 0`。也就是说这个 bias 只会降低 attention score，不会抬高它。

### 6.3 这个公式怎么理解

看几个情况。

情况 A：稳定 query 读高风险 key。

```text
D_q 低，D_k 高
keep = 1 - (1 - D_q) * D_k 变小
bias 变成明显负数
```

结果：

```text
稳定 token 不太会去读高风险 token。
```

情况 B：稳定 query 读稳定 key。

```text
D_q 低，D_k 低
keep 接近 1
bias 接近 0
```

结果：

```text
稳定 token 可以正常互相读取。
```

情况 C：高风险 query 自己在读东西。

```text
D_q 高
keep 接近 1
bias 较小
```

结果：

```text
系统不会强行阻止高风险 token 自己读取上下文。
重点是别让稳定结构被高风险 key 污染。
```

### 6.4 读控制强度

基础强度：

```text
beta_frame = 4.75
```

关键 chunk 微调：

```text
chunks 5,6,7,8,9:
    beta_frame = 4.85

chunks 10,11,12,16:
    beta_frame = 4.25

其他 chunk:
    beta_frame = 4.75
```

`beta_frame` 越大，高风险 key 被压得越强。

---

## 7. 写控制第一层：哪些 token 更适合写 TTT

当前窗口输出之后，C9 要决定下一窗口的 TTT fast weights 怎么写。

写入分数配置：

```text
hmc_write_score_source = stage_d_x_dg_inv_sqrt
```

这句话的意思是：

```text
写入分数 = Stage D 基础可靠度 * sqrt(1 - D_g)
```

这里要特别小心：文档里说的 `Stage D 基础可靠度`，在代码里不是单个变量名，而是几步之后进入 HMC 的 `base_score`。它来自 Stage D prior，但会经过 mean-preserving policy 和一次归一化。

更完整地看，C9 的写入分数链路是：

```text
Stage B:
    C_stat, C_dyn, C_occ, C_unc, C_anchor
        -> G_write_geo

Stage D:
    G_write_geo
        -> A_tok_stageD 初稿

mean-preserving policy:
    按 mp_score_source=dyn 调整 Stage D multiplier
        -> P_ttt_write

HMC write override:
    base_score = normalize(P_ttt_write on patch tokens)
    raw_score = base_score * sqrt(1 - D_g)
    final A_tok = 1 + 0.1 * rank(raw_score)
```

所以：

```text
stage_d_score / base_score:
    不是 D_g；
    不是 C_dyn；
    不是语义标签；
    它是 Stage D 几何写入 prior 经过策略转换后的基础写入分数。
```

### 7.1 Stage D 基础可靠度

Stage D 会给每个 token 一个基础写入可靠度。因为 C9 的：

```text
stage_c_mode = none
```

所以这里没有语义 masklet 作为 runtime action。基础可靠度主要来自几何动态、不确定性、稳定锚点这些信息。

可以简化记成：

```text
base_score[i] = StageD_reliability[i]
```

`base_score` 越高，说明这个 token 越像可靠几何证据。

更具体地说，C9 里有两层 Stage D 相关处理。

第一层，`SemanticPriorGenerator` 会从 `G_write_geo` 得到一个 patch prior：

```text
A_patch_flat ~= pool(G_write_geo)
```

如果没有语义 masklet，`A_patch_flat` 基本就是几何 eligibility 的 patch 版本。

第二层，pipeline 应用：

```text
prior_policy = eta_mean_preserving
mp_score_source = dyn
mp_alpha = 0.1
mp_min = 0.8
mp_max = 1.2
```

`mp_score_source=dyn` 的意思是：按 Stage B 的 `C_dyn` 排名，动态越高，multiplier 越低。

可以写成：

```text
centered_dyn_rank = centered_percentile_rank(C_dyn)

P_ttt_write_patch =
    clamp(1 - 0.1 * centered_dyn_rank, 0.8, 1.2)
```

这里有个负号，因为：

```text
C_dyn 高:
    更动态，不适合写，所以 multiplier 降低。

C_dyn 低:
    更静态，适合写，所以 multiplier 提高。
```

这个 `P_ttt_write_patch` 进入 HMC 后，会被归一化成：

```text
base_score = normalize01(P_ttt_write_patch)
```

这就是后面 `stage_d_x_dg_inv_sqrt` 里的 `stage_d` 基础项。

如果某些情况下 Stage D prior 完全没有区分度，HMC 会退回到同样的动态 rank 逻辑：

```text
base_patch = clamp(1 - 0.1 * centered_percentile_rank(C_dyn), 0.8, 1.2)
```

所以 C9 的 Stage D 底座可以稳定理解成：

```text
低动态 patch 更可写；
高动态 patch 更少写。
```

### 7.2 乘上 read cue 的反向项

`D_g` 高表示不稳定。写入时应该惩罚它。

C9 用：

```text
read_static_factor[i] = sqrt(1 - D_g[i])
```

如果：

```text
D_g = 0:
    sqrt(1 - D_g) = 1
    不惩罚。

D_g = 0.75:
    sqrt(1 - D_g) = 0.5
    写入分数减半。

D_g = 1:
    sqrt(1 - D_g) = 0
    强烈不建议写。
```

最终原始写入分数：

```text
write_score_raw[i] = base_score[i] * sqrt(1 - D_g[i])
```

这一步才是 `stage_d_x_dg_inv_sqrt` 的名字来源：

```text
stage_d:
    base_score，来自 Stage D / dynamic rank 的基础可写性。

x:
    相乘。

dg_inv_sqrt:
    sqrt(1 - D_g)，也就是 D_g 的反向平方根。
```

为什么用平方根而不是直接用 `1 - D_g`：

```text
1 - D_g:
    惩罚更强。

sqrt(1 - D_g):
    惩罚更温和。
```

C9 是保守配方，不希望单个 cue 过强地把 token 完全打掉，所以用了平方根。

### 7.3 从写入分数变成 token prior

TTT replay 需要的是一个乘到学习率上的 multiplier，记作：

```text
A_tok[i]
```

C9 把 `write_score_raw` 排名化，然后限制在 0.8 到 1.2：

```text
ranked[i] = centered_percentile_rank(write_score_raw[i])

A_tok[i] = clamp(1 + mp_alpha * ranked[i], mp_min, mp_max)
```

C9 参数：

```text
mp_alpha = 0.1
mp_min = 0.8
mp_max = 1.2
```

含义：

```text
A_tok > 1:
    这个 token 写入略微加强。

A_tok < 1:
    这个 token 写入略微减弱。

A_tok = 1:
    中性。
```

注意这里不是硬删除 token，而是 soft control。C9 的风格是保守调节，不是粗暴屏蔽。

还要注意：`write_score_raw` 本身不是最终学习率。它只决定 patch 在当前 chunk 内的排名。真正写入 TTT 的 multiplier 是 `A_tok`：

```text
lr_eff[i] = lr[i] * A_tok[i]
```

special token 默认保持：

```text
A_tok = 1
```

这样 C9 只轻微改变 patch token 的写入强弱，不去乱动模型内部特殊 token。

---

## 8. 为什么 C9 要用 `probe_ttt_write`

这是 C9 很关键的设计。

普通想法可能是：

```text
controlled pass 已经产生了更好的当前窗口结果，
那就直接用 controlled pass 的缓存写入下一窗口。
```

但这有风险。因为 controlled pass 的 token 已经被读控制影响过。如果直接把它们写入下一窗口，就可能发生：

```text
控制改变当前 token
    -> 改过的 token 写入 fast weights
    -> 下一窗口读到这个被控制过的记忆
    -> 控制影响被继续放大
```

C9 选择：

```text
hmc_commit_mode = probe_ttt_write
```

具体含义：

```text
1. 当前窗口最终轨迹：
   使用 controlled pass 的结果。

2. 下一窗口 TTT fast weights：
   不用 controlled pass 的 TTT cache；
   回到 probe pass 的 native TTT cache；
   只在 replay 时施加明确的写入 prior / tri replay / weak freeze。
```

这就把两个问题拆开了：

```text
当前怎么输出更好轨迹？
    用读控制。

下一窗口该写入什么记忆？
    用 probe cache + 写控制。
```

---

## 9. TTT 写入：从普通 replay 到 C9 replay

先讲普通 TTT update。

对每个 token，TTT cache 里有：

```text
k[i]:
    key 特征，决定这个 token 在 fast weights 里匹配什么。

v[i]:
    value 特征，决定写入什么内容。

lr[i]:
    这个 token 的学习率或更新步长。
```

普通写入可以粗略理解成：

```text
W_next = W_old + update(k, v, lr)
```

C9 把 `A_tok` 乘到学习率上：

```text
lr_eff[i] = lr[i] * A_tok[i]
```

并且主要作用在：

```text
prior_branch_mask = 0
```

也就是 TTT 的 `w0` 分支。

### 9.1 `A_tok` 怎么对齐到 TTT cache

TTT cache 里的 token 顺序必须和 HMC 构造的 token prior 对齐。代码里会先尝试按完整 token 长度对齐：

```text
A_tok length == cache token length:
    直接按 token index 使用。
```

如果某些诊断路径只缓存 patch token，它会再按 `token_type` 抽取 patch token 对齐。C9 的 full-online 主路径使用完整 token layout，所以可以理解成：

```text
第 i 个 token 的 A_tok[i]
    乘到第 i 个 replay token 的 lr[i]。
```

如果没有对齐，TTT 写入会变成“拿道路 token 的 prior 去控制天空 token”这种灾难，所以这是 replay 正确性的基础。

### 9.2 为什么 C9 主要动 `w0`

TTT fast weights 有：

```text
w0, w1, w2
```

在 C9 里：

```text
prior_branch_mask = 0
ttt_write_gradient_reversal_branch_mask = 0
```

含义是：

```text
w0:
    使用 C9 的 token prior 和 tri-replay negative 修正。

w1, w2:
    尽量保持原生写入逻辑。
```

这不是说 `w1/w2` 不重要，而是历史实验里同时大幅改变多个分支更容易不稳。C9 的策略是先控制最关键的 gate/update 方向分支 `w0`，让整体行为更保守。

---

## 10. `update_conflict_energy`：找出和整体更新打架的 token

只靠 `A_tok` 还不够。因为有些 token 看起来未必很高风险，但它们对 TTT 权重的更新方向可能和整体方向冲突。

C9 使用：

```text
ttt_write_gradient_reversal_risk_source = update_conflict_energy
```

它问的问题是：

```text
这个 token 对 w0 的更新方向，
是否和整个 chunk 聚合出来的 w0 更新方向冲突？

如果冲突，而且它更新能量很大，
它就是高风险 negative token。
```

### 10.1 先估计每个 token 对 w0 的更新方向

实现里会用 `k, v, lr, w0_old, w1_old, w2_old` 估计 token 对 `w0` gate 的贡献。

简化写成：

```text
gate_i = k_i * w0_old

hidden_i = k_i * w2_old

dhidden_i = v_i * w1_old^T

dgate_i = dhidden_i * hidden_i

sigma_i = sigmoid(gate_i)

dgate_before_act_i =
    dgate_i * sigma_i * (1 + gate_i * (1 - sigma_i))
```

不用纠结每一项的神经网络细节。关键是：

```text
dgate_before_act_i 表示 token i 对 w0 更新方向的局部贡献。
```

这里的几个量可以直观理解成：

```text
gate_i:
    这个 token 通过 w0 打开的门。

hidden_i:
    这个 token 通过 w2 得到的隐状态。

dhidden_i:
    value 经过 w1 反传/重放得到的 hidden 变化。

dgate_before_act_i:
    把上面这些因素合起来，得到 token 对 w0 的更新方向。
```

它不是正常训练里的完整反向传播，而是利用 TTT replay cache 做一个轻量的更新方向估计。

### 10.2 聚合整个 chunk 的更新方向

带上有效学习率：

```text
lr_eff_i = lr_i * A_tok_i
```

聚合更新：

```text
Aggregate = sum_i (k_i * lr_eff_i)^T * dgate_before_act_i
```

`Aggregate` 表示整个 chunk 想把 `w0` 往哪个方向推。

### 10.3 计算单个 token 是否和整体方向一致

用余弦相似度：

```text
cos_i =
    dot(k_i * Aggregate, dgate_before_act_i)
    / (||k_i|| * ||Aggregate|| * ||dgate_before_act_i|| + eps)
```

解释：

```text
cos_i 接近 1:
    token i 和整体更新方向一致。

cos_i 接近 0:
    token i 和整体方向关系弱。

cos_i 小甚至负:
    token i 在和整体更新方向打架。
```

### 10.4 再乘上更新能量

如果一个 token 方向冲突但能量很小，其实影响不大。C9 还看能量：

```text
energy_i = |lr_eff_i| * ||k_i|| * ||dgate_before_act_i||
```

最终风险：

```text
risk_i =
    Normalize01(((1 - cos_i) / 2) * Normalize01(energy_i))
```

所以高风险 token 必须同时满足：

```text
1. 更新方向和整体方向冲突；
2. 更新能量足够大。
```

这比单纯说“这个 patch 像动态物体”更直接，因为它看的是 TTT 写入本身会不会被这个 token 带歪。

还要注意：`risk_i` 和 `D_g` 是两种风险。

```text
D_g:
    进入读控制，也进入写入分数的 sqrt(1-D_g)。
    它来自 query 和过去帧结构的相似度。

risk_i:
    只用于 tri-replay 的 positive/neutral/negative 分组。
    它来自 TTT 更新方向冲突和能量。
```

一个 token 可能 `D_g` 不算特别高，但 TTT 更新方向很冲突，于是 `risk_i` 高。反过来，一个 token 也可能 `D_g` 高但更新能量很小，对 TTT 污染有限。

---

## 11. tri replay：positive / neutral / negative 三组分开写

C9 使用：

```text
ttt_write_gradient_reversal_mode = tri_replay
```

它不是一次性把所有 token 混在一起写，而是按 `risk_i` 分三组。

### 11.1 分组

默认：

```text
positive group:
    风险最低的 35% token。

negative group:
    风险最高的 12% token。

neutral group:
    剩下中间的 token。
```

chunk 16 稍微更保守：

```text
negative group:
    风险最高的 8% token。
```

### 11.2 三组分别 replay

对三组分别重放 TTT update：

```text
W_pos:
    只用 positive group 得到的 fast weights。

W_neu:
    只用 neutral group 得到的 fast weights。

W_neg:
    只用 negative group 得到的 fast weights。
```

代码里不是简单地把 token 删除，而是给三组构造不同的 replay vector：

```text
pos_vec[i] = A_tok[i] * positive_mask[i]

neu_vec[i] = A_tok[i] * neutral_mask[i]

neg_vec[i] = risk_i * negative_mask[i]
```

解释：

```text
positive / neutral:
    仍然使用 A_tok，因为它们是正常写入证据，只是分组不同。

negative:
    使用 risk_i，因为它不是正常正向写入；
    它的作用是估计高冲突 token 会把 w0 推向哪里，
    然后在合成时扣掉一点。
```

三次 replay 都从同一个 `W_old` 出发，所以 `W_pos`、`W_neu`、`W_neg` 是三条“如果只听这一组 token，fast weight 会怎么变”的候选方向。

### 11.3 组合公式

对 C9 主要生效的 `w0` 分支，组合方式是：

```text
W_candidate =
    Renorm(
        W_old
        + (W_pos - W_old)
        + neutral_lambda * (W_neu - W_old)
        - gamma * (W_neg - W_old)
    )
```

C9 参数：

```text
neutral_lambda = 0.85

gamma:
    chunks 5,6,7,8,9  -> 0.005
    chunks 10,11,12   -> 0.003
    chunk 16          -> 0.0003
    other chunks      -> 0
```

逐项解释：

```text
W_old:
    当前窗口开始前的 fast weight。

W_pos - W_old:
    低风险 token 给出的正向更新，完整保留。

0.85 * (W_neu - W_old):
    中性 token 也写，但打八五折。

- gamma * (W_neg - W_old):
    高冲突 token 的更新方向被轻微反向扣掉。
```

`gamma` 很小，说明 C9 不是激进地反训练，而是轻轻修正方向。

`Renorm` 表示把权重范数拉回合理范围，避免写完之后权重大小突然爆掉。

如果某个分支没有被 `branch_mask` 选中，代码会让它走正常 full positive candidate，而不是套用这个三段组合。C9 的 active branch 是 `w0`，所以你可以把上面的公式主要理解为 `w0` 的公式。

### 11.4 tri-replay 和普通“少写”有什么不同

普通少写只做：

```text
高风险 token 的 A_tok 低一点。
```

tri-replay 多做了一步：

```text
先单独估计高冲突 token 会把权重推向哪里，
再把这个方向以很小 gamma 反向扣掉。
```

所以 tri-replay 不只是 suppress，它还做 weak anti-update。因为 `gamma` 很小，这个 anti-update 是轻微方向修正，不是强反训练。

---

## 12. native mix：不要离原生写入太远

C9 还使用：

```text
ttt_write_native_mix_scales = 1.10,1.00,1.00
```

先得到两个候选：

```text
W_native:
    原生 TTT replay 的写入结果。

W_controlled:
    加了 C9 prior / tri replay 后的写入结果。
```

混合公式：

```text
W_mix = Renorm(W_native + scale * (W_controlled - W_native))
```

三个 scale 对应：

```text
w0 scale = 1.10
w1 scale = 1.00
w2 scale = 1.00
```

含义：

```text
w1, w2:
    使用 HMC-controlled candidate。

w0:
    从 native 结果往 HMC-controlled 结果方向走，
    并且略微多走 10%。
```

这和 `prior_branch_mask=0` 一致：C9 的主要控制点是 `w0`。

C9 没有设置 `ttt_write_native_mix_chunks`，所以这个 native mix 不只作用于某个 chunk，而是对所有 chunk 都可用。它放在 semantic/controlled replay 之后，commit EMA 之前。

可以把顺序理解成：

```text
1. 先得到原生 native replay。
2. 再得到带 C9 控制的 replay。
3. 用 native mix 把 controlled replay 拉回 native 附近。
4. 如果是 chunk 5/6，再做 weak freeze commit EMA。
```

---

## 13. weak freeze：chunk 5 和 6 的 w0 只写一半

C9 的另一个关键点：

```text
ttt_write_commit_ema_alpha = 0.5
ttt_write_commit_ema_branch_mask = 0
ttt_write_commit_ema_chunks = 5,6
```

最终提交时：

```text
W_commit = Renorm(W_old + alpha * (W_candidate - W_old))
```

在 chunk 5 和 6：

```text
alpha = 0.5
```

所以：

```text
只提交 w0 candidate 变化的一半。
```

这就是 weak freeze 的含义。

它不是完全不写：

```text
alpha = 0:
    完全冻结。

alpha = 0.5:
    写一半。

alpha = 1:
    完全采用 candidate。
```

C9 只在 chunk 5 和 6 做这件事，说明它不是全程保守，而是在特定早期窗口防止 w0 写入过猛。

这个 EMA 发生在 final committed fast weights 上，而不是发生在读控制上。也就是说：

```text
当前 chunk 的 controlled 输出:
    不因为 weak freeze 被改掉。

下一 chunk 会继承的 w0:
    在 chunk 5/6 只提交一半变化。
```

这也是为什么它叫 commit EMA，而不是 attention gate。

---

## 14. SWA overlap：处理相邻窗口的短期接缝

除了 TTT，C9 还控制 SWA local memory。

SWA 可以理解成相邻 chunk 之间的短期缓存，主要服务 overlap 的连续性。

C9 设置：

```text
enable_swa_overlap_source_replace = 1
swa_overlap_source_replace_alpha = 0.5
swa_overlap_source_replace_mode = source
swa_overlap_source_replace_target = kv
swa_overlap_source_replace_layer_mode = last
```

简化公式：

```text
score = D_prev_tail

alpha = 0.5 * score

K_prev_tail <- (1 - alpha) * K_prev_tail + alpha * K_cur_head
V_prev_tail <- (1 - alpha) * V_prev_tail + alpha * V_cur_head
```

解释：

```text
prev_tail:
    上一个窗口尾部 overlap token。

cur_head:
    当前窗口头部 overlap token。

K / V:
    attention 的 key / value 缓存。
```

如果上一窗口尾部 overlap token 风险高，就更多用当前窗口头部的新 K/V 替换它。

这里的 `D_prev_tail` 来自上一窗口留下的 `prev_control_summary`，本质上是上一窗口 overlap 区域的 `D_g` 摘要。它不是 TTT long memory，也不是语义标签。

所以 SWA overlap replace 的逻辑是：

```text
上一窗口尾部 overlap 如果高风险:
    用当前窗口头部的 K/V 多替换一些。

上一窗口尾部 overlap 如果低风险:
    保留它，维持窗口间连续性。
```

C9 还设置：

```text
enable_swa_write_control = 1
swa_write_mode = none
swa_write_keep_scope = both_overlap
swa_write_layer_mode = last
```

这里 `swa_write_mode=none` 表示没有额外分数门控；但 `keep_scope=both_overlap` 仍限制 SWA history 主要保留双方 overlap 范围。

直觉：

```text
SWA 是短期桥，不是长期笔记。
C9 让它更专注于相邻窗口接缝。
```

---

## 15. 用一个小例子串起来

假设当前窗口里有 5 类 patch：

```text
A: 道路边界
B: 建筑边缘
C: 护栏
D: 移动车辆
E: 天空/低纹理区域
```

### 15.1 构造 `D_g`

C9 比较每个 patch 和过去帧整体 query centroid 的相似度：

```text
A/B/C:
    多帧重复出现，query 和过去结构相似。
    D_g 低。

D:
    位置变化大，和过去稳定结构不一致。
    D_g 高。

E:
    低纹理，几何不稳定，query 可能也不稳定。
    D_g 中高。
```

### 15.2 controlled pass 读控制

道路 token 想读车辆 token：

```text
D_q 低，D_k 高
keep 小
bias 负
attention 降低
```

道路 token 想读建筑 token：

```text
D_q 低，D_k 低
keep 接近 1
bias 接近 0
attention 正常
```

### 15.3 TTT 写入 prior

写入分数：

```text
write_score = base_score * sqrt(1 - D_g)
```

所以：

```text
道路、建筑、护栏:
    写入 multiplier A_tok 可能略高。

车辆、天空:
    写入 multiplier A_tok 可能略低。
```

### 15.4 update conflict

如果某个车辆 token 的 TTT 更新方向和整体窗口方向冲突，而且能量大：

```text
risk_i 高
```

tri replay 会把它归入 negative group。

最终它对 `w0` 的影响会被轻微反向扣掉。

---

## 16. 一张完整流程图

```text
当前 32 帧图像 + 上一窗口 HMC state
        |
        v
如果 chunk 是 5 的倍数:
    清掉 TTT fast weights，保留 local history
        |
        v
Pass 1 probe:
    正常跑 LoGeR
    得到初始几何、q/k 特征、TTT write cache
        |
        v
构造风险:
    Stage B 几何动态风险
    ACL2 read cue D_g
        |
        v
构造控制 prior:
    读控制用 D_tok
    写控制用 A_tok = stage_d * sqrt(1 - D_g)
        |
        v
Pass 2 controlled:
    frame attention 加 pair bias
    SWA overlap 做 K/V replace
    输出当前窗口 final geometry
        |
        v
TTT commit:
    不用 controlled cache
    用 probe cache 重新 replay
    A_tok 调整学习率
    update_conflict_energy 找 negative token
    tri_replay 三组写入
    native mix
    chunk 5/6 weak freeze
        |
        v
提交下一窗口 HMC state
        |
        v
轨迹 merge，进入下一个 chunk
```

---

## 17. C9 为什么算 deployable online

C9 没有使用下面这些东西作为 runtime action：

```text
GT pose
SemanticKITTI 真值标签
后处理轨迹改写
离线训练出的 trigger / router / classifier
v41 的 health detector 结果
```

它使用的是：

```text
模型内部 q/k 特征
当前窗口几何一致性
attention 动态 proxy
TTT 更新方向冲突
overlap 局部缓存关系
```

这些都可以在在线运行时从输入视频和模型自身 forward 中得到。

所以它是 deployable online TTT-write，而不是 oracle。

---

## 18. C9 为什么有效

### 18.1 它同时管当前窗口和未来窗口

只做读控制，只能影响当前窗口。

只做写控制，当前窗口可能已经被坏 token 影响。

C9 同时做：

```text
当前窗口:
    frame attention 少读高风险 patch。

未来窗口:
    TTT fast weights 少写高风险/高冲突 patch。
```

这更符合在线漂移问题的本质：坏信息经常不是只伤害当前帧，而是被写进 memory 后继续影响后面。

### 18.2 它不是简单地“少写动态”

C9 的写入判断有两层：

```text
第一层:
    stage_d_x_dg_inv_sqrt
    几何稳定 + 像过去结构 -> 多写。

第二层:
    update_conflict_energy
    TTT 更新方向和整体方向冲突 -> negative。
```

第二层特别重要，因为它看的是 TTT 写入机制本身。

### 18.3 它很保守

C9 的控制强度都不大：

```text
A_tok 范围:
    0.8 到 1.2

negative gamma:
    0.005 / 0.003 / 0.0003

weak freeze:
    只在 chunk 5/6 的 w0 写一半
```

这说明 C9 不是强行改模型，而是在原生 LoGeR 的基础上做轻量、局部、可控的写入修正。

---

## 19. C9 的不足

C9 仍然不是最终解。

主要问题：

```text
1. ATE = 33.7629421029m，仍高于 Target-30。

2. [200,300) 病灶段比 H9 差：
       H9 = 74.409927m
       C9 = 76.102136m

3. C9 没有真实语义因果证据。
   stage_c_mode = none。
   它不是知道 sky/car/person 后做语义修复。

4. 它主要改善整体写入污染和一部分后段漂移，
   但没有解决所有局部 stress window。

5. v41 的 read-first family 虽然在 chunk10 有局部诊断信号，
   但 h15 没过，所以不能取代 C9 成为 deployable online success。
```

---

## 20. 配置速查

| 配置 | C9 值 | 行为 |
|---|---|---|
| `hybrid_memory_mode` | `hybrid` | 使用 HMC 混合记忆控制 |
| `two_pass` | `true` | 每个 chunk 跑 probe 和 controlled 两遍 |
| `window_size` / `chunk_size` | `32` | 每个窗口 32 帧 |
| `overlap_size` / `chunk_overlap` | `3` | 相邻窗口重叠 3 帧 |
| `reset_every` | `5` | 每 5 个 chunk 清掉 TTT fast weights |
| `stage_c_mode` | `none` | 不启用语义 masklet runtime action |
| `semantic_prior_mode` | `spg_v2` | Stage D 仍运行，但在 C9 中主要消费几何 `G_write_geo` |
| `spg_use_g_write_geo` | `1` | Stage D eligibility 直接使用 Stage B 的 `G_write_geo` |
| `read_cue_source` | `acl2.gg.qq.low.g2_3.past_only.headmean.robustq` | 用过去帧 query 相似度构造 `D_g` |
| `read_path` | `frame` | 读控制接到 frame attention |
| `enable_frame_read_control` | `1` | 开启 frame attention 控制 |
| `frame_bias_mode` | `pair` | 稳定 query 读取高风险 key 时加负 bias |
| `read_beta_frame_chunks` | `5-9:4.85, 10-12/16:4.25` | 对关键 chunk 微调读控制强度 |
| `hmc_write_score_source` | `stage_d_x_dg_inv_sqrt` | 写入分数乘 `sqrt(1 - D_g)` |
| `prior_policy` | `eta_mean_preserving` | Stage D prior 不改变整体 chunk 写入均值，只按排名重分配 |
| `mp_score_source` | `dyn` | Stage D 底座按 `C_dyn` 排名，低动态多写，高动态少写 |
| `mp_alpha` | `0.1` | 写入 multiplier 轻量调节 |
| `mp_min` / `mp_max` | `0.8` / `1.2` | 写入 multiplier 范围 |
| `hmc_write_sparse_mode` | `none` | 不做硬筛选，所有 patch token 都保留 soft multiplier |
| `hmc_write_sparse_ratio` | `1.0` | 不稀疏化 token |
| `prior_branch_mask` | `0` | 写入 prior 主要作用于 `w0` |
| `hmc_commit_mode` | `probe_ttt_write` | 当前输出用 controlled，下一窗口 TTT 用 probe cache replay |
| `ttt_write_gradient_reversal_mode` | `tri_replay` | positive / neutral / negative 三组 replay |
| `ttt_write_gradient_reversal_risk_source` | `update_conflict_energy` | 用 TTT 更新冲突能量找 negative token |
| `ttt_write_tri_replay_positive_frac` | `0.35` | 风险最低 35% 当 positive |
| `ttt_write_tri_replay_negative_frac` | `0.12` | 风险最高 12% 当 negative |
| `ttt_write_tri_replay_neutral_lambda` | `0.85` | neutral 写入打 0.85 倍 |
| `ttt_write_gradient_reversal_chunk_gammas` | `5-9:0.005, 10-12:0.003, 16:0.0003` | negative 反向扣减强度 |
| `ttt_write_native_mix_scales` | `1.10,1.00,1.00` | w0 相对 controlled candidate 多走 10% |
| `ttt_write_commit_ema_alpha` | `0.5` | EMA 弱冻结强度 |
| `ttt_write_commit_ema_chunks` | `5,6` | 只在 chunk 5/6 弱冻结 |
| `enable_swa_overlap_source_replace` | `1` | 开启 SWA overlap K/V 替换 |
| `swa_overlap_source_replace_alpha` | `0.5` | SWA 替换最大强度 |
| `swa_overlap_source_replace_target` | `kv` | 替换 K 和 V |
| `swa_write_keep_scope` | `both_overlap` | SWA history 主要保留 overlap 桥接 |

---

## 21. 最后再压缩成一句话

`C9_P0_R2` 是一个保守的在线记忆管理配方。

它先用过去帧 query 相似度构造 `D_g`，找出“不像稳定结构”的 patch，再让当前窗口少读这些 patch；窗口结束写 TTT fast weights 时，它用 Stage D 的低动态几何底座 `base_score` 乘上 `sqrt(1 - D_g)`，得到最终写入排名，再把排名变成 `A_tok` 学习率 multiplier；随后用 `update_conflict_energy` 找出和整体更新方向冲突的 token，并通过 tri replay 把低风险、中性、高冲突 token 分开写。

它的关键工程边界是：

```text
当前窗口输出:
    来自 controlled pass。

下一窗口 TTT 记忆:
    来自 probe cache 的受控 replay。

不使用:
    GT pose、SemanticKITTI 真值标签、后处理轨迹改写、learned runtime trigger。
```

这就是它能成为当前历史 best deployable online TTT-write 的原因；但它仍然没有达到 Target-30，也没有解决所有局部病灶段。

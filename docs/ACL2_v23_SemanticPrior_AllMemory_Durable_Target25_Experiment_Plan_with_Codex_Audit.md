# ACL2 v23 实验计划：Semantic Prior Generator 优先的 All-Memory Durable Write / Source Policy（含 Codex 代码自查版）

日期：2026-05-21  
更新：加入 Codex implementation self-audit / bug-check hard gate  
目标数据集：KITTI Odometry Sequence 01  
当前 deployable online TTT write best：`C9_P0_R2 = 33.7629421029m`  
最终目标：`KITTI01 ATE <= 25.0m`  
当前差距：约 `8.76m`

---

## 0. 本轮实验的核心判断

上一轮 v22 没有产生新的 deployable online Target-25 结果。最强结果仍然是旧的：

```text
C9_P0_R2
ATE = 33.7629421029m
counts_as_ttt_write = true
```

v22 的所有新结果仍然是 trusted short-rollout diagnostic，不是 full online success。它们说明一个非常重要的问题：**语义、source skip、skip-aware TTT、skip-aware SWA/global memory、durable commit、lifecycle split 都能产生短窗口修正，但这个修正没有稳定写进长期记忆。**

因此，本轮 v23 不再把重点放在普通 TTT 写入标量、普通 Dg 阈值、普通 compact K/V 阈值微扫上，而是把优先级切换到：

```text
Semantic Prior Generator -> all-memory role policy -> durable semantic memory correction
```

也就是说，语义不再只是 TTT 写入时的一个 scalar multiplier，而要变成所有 memory path 的角色分配器：

```text
frame attention memory:
    哪些语义区域能作为 K/V source

global / chunk attention memory:
    哪些语义区域能进入 global context

SWA memory:
    哪些 previous/current source cache 可以保留、替换、跳过

TTT memory:
    哪些 token 是 positive / neutral / negative write evidence
```

本轮要回答的核心问题不是“语义有没有用”，而是：

> 语义能不能把短期有效的 source/read correction 转成跨 h15、甚至 full online 仍然有效的 durable memory correction？

---

## 1. 对 v22 结果的独立分析

### 1.1 v22 不是完全没进展，但没有达成目标

v22 的工程进展是明确的：

```text
1. compact_kv / skip-aware path 跑通；
2. read-only compact blocker 被发现并修复；
3. exact coarse semantic group 已经进入 HMC / model control；
4. state attribution 已经能量化 h10 correction 被 h15 后续 state 覆盖的问题；
5. support / compact / semantic / TTT / SWA / durable commit / lifecycle branches 都实际跑完。
```

但策略进展仍然不够：

```text
1. 没有新的 deployable online result；
2. 没有启动 no-GT selector；
3. 没有启动 full online validation；
4. 没有产生 Target-25；
5. 最强 h10 信号仍然无法持久到 h15。
```

当前最好可计数 online 结果仍是 `C9_P0_R2 = 33.7629421029m`，距离目标 `25.0m` 仍差约：

$$
33.7629421029 - 25.0 = 8.7629421029
$$

这个差距很大，不能靠 `gamma`、`q80/q85/q90`、`compact_kv threshold`、`lowstuff scalar` 继续小扫来弥补。

---

### 1.2 v22 最强信号：TTT lifecycle 有 h10 作用，但 h15 durability 失败

v22 最强 h10 信号是：

```text
TTT_LIFE_04_SCALE_LONG_HIGHD_SHORT
h10 ATE delta       = -1.8034599870m
h10 [200,300) delta = -4.2566491136m
h15 ATE delta       = +0.2915878358m
durability ratio    = 0.1616824528
```

这个结果说明：

```text
1. 语义 / skip / scale / lifecycle 组合可以短期改善 [200,300)；
2. 但 correction 到 h15 之后基本被洗掉；
3. h15 ATE 甚至相对 H9 回退；
4. durability ratio 远低于 0.45 的最低要求。
```

因此，`TTT_LIFE_04` 不能进入 selector，也不能 full online validation。它只证明“短期有信号”，没有证明“长期 memory policy 成立”。

---

### 1.3 Phase C semantic role 有信号，但还不是合格语义策略

v22 的 semantic role 阶段有两个值得保留的结果：

```text
SEM_ROLE_01_STRUCTURE_RESCUE chunk6 h15:
    ATE delta = -0.8738393532m
    [200,300) delta = -1.5361619675m
    [400,600) delta = -0.9595408243m

SEM_ROLE_01_STRUCTURE_RESCUE chunk10 h10:
    ATE delta = -0.8190672795m
    [200,300) delta = -3.6277354286m
```

这说明 `STRUCTURE_RESCUE` 不完全没用：它在 chunk6 的 h15 上有稳定小收益，在 chunk10 的 h10 上对 disease window 有明显局部收益。

但是它不满足 gate：

```text
h10/h15 ATE delta <= -3m:
    未达到

[200,300) delta <= -5m:
    未达到

chunk10 h15:
    ATE delta = +0.7361069995m，发生回退
```

我的判断是：当前语义角色还没有错，但太浅。`STRUCTURE_RESCUE` 只作为一个 source/read 层面的 rescue，不足以进入所有 memory 的长期 commit。语义要继续推进，但不能继续只做 scalar 或单 path semantic mask。

---

### 1.4 Phase E/F 说明：skip-aware TTT / SWA / global source 都只是短期修正

v22 的 skip-aware TTT 和 skip-aware memory 结果为：

```text
KVC_TTT_04_SOURCE_KEEP_GATED_WRITE:
    h10 ATE delta       = -0.7324282608m
    h15 ATE delta       = +0.8391261607m
    h10 [200,300) delta = -3.4472737923m
    h15 [200,300) delta = -0.9390895264m

KVC_MEM_03_GLOBAL_CHUNK_SOURCE_SKIP:
    h10 ATE delta       = -0.7353211000m
    h15 ATE delta       = +0.5674474093m
    h10 [200,300) delta = -3.4871305431m
    h15 [200,300) delta = -1.4384892706m
```

这说明：

```text
1. source keep / source skip 能短期压 disease window；
2. 但进入 h15 后 ATE 回退；
3. 后续 memory update 会覆盖掉局部修正；
4. 只做 source filter 或单次 skip-aware write 不够。
```

因此，真正问题不是“source skip 有没有信号”，而是“source skip 后的修正如何进入长期 memory”。

---

### 1.5 State attribution 暴露了根本问题：h10 修正被 h15 后续状态覆盖

v22 的 state attribution 非常关键：

```text
HMC all overwrite ratio = 0.8731029807
HMC TTT overwrite ratio = 0.8731009793
merge overwrite ratio   = 0.5654160926
```

这说明：

```text
从 base 到 h10 产生了一个局部修正；
但 h10 到 h15 的后续 HMC/TTT state movement 几乎同样大；
后续 merge/gauge state movement 也很大；
所以 h10 修正不是稳定状态，而是被后续 state 更新洗掉。
```

这就是当前的核心瓶颈：

> 我们已经能做短期修正，但还不能把这个修正写成 durable memory state。

所以 v23 的实验目标必须从“找更强的 h10 short rollout”升级成：

```text
1. 找到 h10 -> h15 被覆盖的 memory path；
2. 让 semantic role 同时控制 read/source/write/cache/commit；
3. 让短期修正进入 W_long / SWA cache / global source / merge-safe state；
4. 在 h15 上仍然保持收益。
```

---

## 2. 本轮总目标

本轮 v23 的总目标是：

> 以 Semantic Prior Generator 为核心，把语义从单一 TTT write scalar 扩展为 all-memory role policy，验证语义是否能让 context skip / scale-state / TTT write 的短期修正变成长期有效的 durable correction。

具体目标分为五层。

### 2.1 工程目标

必须确保 semantic group 信息不再只停留在 `S_tok` scalar，而是完整进入所有 memory control path：

```text
G_sem_tok:
    coarse semantic group id

L_sem_tok:
    fine semantic label id or string id

Q_sem_tok:
    semantic trust / mask quality

V_sem_tok:
    semantic value

R_sem_tok:
    semantic role id: positive / neutral / negative / protect / skip
```

这些信息必须能被下面模块读取：

```text
frame attention source control
global / chunk attention source control
SWA source / cache control
TTT positive-neutral-negative replay
TTT long-short lifecycle commit
state attribution / debug logger
```

---

### 2.2 科学目标

验证下面这个假设：

> 不是所有 high-D 或 low-value semantic 区域都应该被跳过；真正有效的是 semantic role 与 geometry/attention risk 的组合。结构区域应该长期保留，低价值但低风险区域应该中性保留，高风险低价值区域才应该跳过或负写入。

换句话说，不能写成：

```text
sky / vegetation / lowstuff -> always skip
```

而应该写成：

```text
structure + lowD:
    source keep + positive long write

lowstuff + lowD:
    neutral keep, 保护 horizon / scale continuity

lowstuff + highD:
    source skip + short negative or no long write

movable + highD:
    source skip + negative / short-only

uncertain:
    trust low，fallback 到 geometry cue
```

---

### 2.3 实验目标

每个候选必须同时报告：

```text
h10 improvement
h15 improvement
durability ratio
[200,300) disease segment
[400,600) downstream regression
semantic group contribution
memory path overwrite
```

本轮不再把只改善 h10 的候选当作主要进展。只有 durable 候选才允许进入 selector / full online validation。

---

### 2.4 加速目标

本轮仍然使用 causal fork / short rollout 作为主筛选工具。原则是：

```text
1. 不直接 full-first；
2. 不把 short rollout 写成 deployable success；
3. 不通过 durability gate，不启动 selector；
4. 不通过 selector，不启动 full online；
5. 每个实验族最多先跑 4-8 条核心 rows；
6. 如果 4 条核心 rows 都没有 h10 信号，Codex 自动转方向。
```

---

### 2.5 成功目标

本轮分四级成功：

```text
Level 0: instrumentation success
    semantic group / role 信息准确进入所有 memory path，no-op parity 通过。

Level 1: local signal success
    h10 [200,300) delta <= -5m，且 [400,600) regression <= +1m。

Level 2: durability success
    h15 ATE delta <= -3m，或 h15 [200,300) delta <= -5m；
    durability ratio >= 0.45。

Level 3: full online success
    full KITTI01 ATE 明显低于 C9_P0_R2；
    最终目标为 ATE <= 25m。
```

---

### 2.6 Codex 代码实现自查目标（hard gate）

本轮计划必须新增一个前置 hard gate：**Codex 在跑任何 v23 主矩阵之前，必须先自查代码实现是否真的实现了计划中的语义 all-memory role policy。** 这不是形式检查，而是防止再次出现“配置写了、日志有了、但实际模型路径没有消费”的问题。

本轮最容易出 bug 的地方不是模型数学，而是下面这些实现链路：

```text
1. CLI / env 参数是否真的传到 run_pipeline_abc_v2.py；
2. run_pipeline_abc_v2.py 是否真的传给 HybridMemoryController；
3. HybridMemoryController 是否真的构建了语义 role masks；
4. pi3.py / attention.py 是否真的消费了 frame/global context source skip；
5. ttt_write_controller.py 是否真的消费了 semantic positive/neutral/negative role；
6. SWA cache 写入/读取是否真的使用 semantic source policy；
7. Stage C cache 在 sliced causal fork 里是否用 global frame start/end 命中正确 chunk；
8. pass-through / noop 是否真的不改变 baseline；
9. forced rerun 是否清理 stale run directory，避免旧 JSONL 污染新结果；
10. debug summary 是否能证明“控制实际生效”，而不是只证明“参数被请求”。
```

Codex 必须先生成一个 implementation audit report，再允许进入 Phase 1/2/3 的性能实验。报告建议命名为：

```text
implementation_audit/codex_self_check_report.md
implementation_audit/codex_self_check_summary.json
implementation_audit/codex_self_check_failures.jsonl
```

Hard gate：

```text
如果 self-check 没有通过：
    不允许启动 v23 semantic all-memory matrix；
    不允许启动 no-GT selector；
    不允许启动 full online validation；
    Codex 必须先修 bug 或降级为 smoke-only。
```

---


## 3. 核心假设与实验设计

---

## H1：当前失败不是语义无用，而是语义没有进入 all-memory role policy

### 假设

v22 的 semantic role 失败，不是因为语义对 LoGeR 无用，而是因为语义只被用于局部 source/rescue，没有系统控制 frame/global/SWA/TTT 的 memory path。

### 实验设计

先做 instrumentation-only 和 no-op 检查，不跑性能矩阵。

必须新增或确认：

```text
HybridMemoryControlPrior:
    G_sem_tok
    L_sem_tok
    Q_sem_tok
    V_sem_tok
    R_sem_tok

Model hmc_control:
    semantic_group_ids
    semantic_role_ids
    semantic_trust
    semantic_source_keep_mask
    semantic_ttt_role_mask

Debug outputs:
    semantic_group_summary.jsonl
    semantic_role_summary.jsonl
    semantic_memory_path_summary.jsonl
```

### 必须记录指标

```text
per_chunk_semantic_coverage
per_group_token_count
per_group_Dg_mean
per_group_Dg_p90
per_group_Qsem_mean
per_group_source_keep_ratio_frame
per_group_source_keep_ratio_global
per_group_source_keep_ratio_swa
per_group_ttt_pos_mass
per_group_ttt_neu_mass
per_group_ttt_neg_mass
per_group_ttt_long_mass
per_group_ttt_short_mass
```

### 可视化

```text
RGB frame
semantic group map
D_g map
semantic role map
source keep map
TTT pos/neu/neg map
overlap/cache keep map
```

每个图必须至少覆盖：

```text
chunk6
chunk10
chunk16
[200,300) focus frames
h10 endpoint frame
h15 endpoint frame
```

### 成立标准

H1 通过条件：

```text
1. Stage C cache require-hit；
2. semantic group projection coverage >= 0.85 in focus chunks；
3. G_sem_tok / R_sem_tok 非空且 shape 与 patch token 对齐；
4. no-op semantic pass-through 与 H9/C9 boundary 差异 <= 0.01m；
5. frame/global/SWA/TTT debug 都能按 semantic group 统计。
```

### 不满足时 Codex 自动尝试方向

```text
如果 semantic group coverage < 0.85:
    检查 Stage C cache chunk index 是否使用 global_start/global_end；
    检查 MaskletOutput.G_sem 是否为空；
    检查 masklet -> patch pooling 是否发生全 0；
    如果 coarse group 可用但 fine label 不可用，先用 coarse group，不阻塞。

如果 no-op parity 失败:
    强制 HMC_IGNORE_SEMANTIC_PRIOR=1；
    检查 pass-through A_tok 是否被归一化成非 no-op；
    回退到 B1G/B2G 语义忽略路径。

如果 semantic role summary 缺失:
    先不跑性能；
    只修 logger / control prior plumbing。
```

---

## H2：语义 role 必须区分 positive / neutral / negative，而不是 low-value hard skip

### 假设

`LOW_VALUE_STUFF` 不是天然负样本。sky / vegetation / grass 可能提供 horizon、scale、远景 continuity。只有当它们同时具有 high-D、high uncertainty、低 trust 时，才应该 source skip 或 negative write。

### Role 定义

本轮使用下面的基础 role 表：

| Semantic group | Geometry / attention condition | Role | Memory action |
|---|---|---|---|
| `STRUCTURE_ANCHOR` | lowD, high trust | positive | source keep, TTT long positive |
| `STRUCTURE_ANCHOR` | highD | protected neutral | source keep or weak skip, no strong negative |
| `LOW_VALUE_STUFF` | lowD | neutral | partial keep, no long strong write |
| `LOW_VALUE_STUFF` | highD | conditional negative | source skip, weak short negative |
| `MOVABLE_THING` | any highD | negative / short | source skip, no long write |
| `STATIC_THING` | lowD | positive or neutral | source keep, weak positive write |
| `UNCERTAIN_REGION` | low trust | fallback | downweight semantic branch, use geometry |

### 实验矩阵

先只跑 chunk6 / chunk10 / chunk16，h10 / h15。

```text
SEMROLE_01_STRUCTURE_POSITIVE_ONLY
    positive = STRUCTURE_ANCHOR & lowD
    neutral = all others
    negative = none

SEMROLE_02_LOWSTUFF_HIGHD_SKIP
    skip = LOW_VALUE_STUFF & highD
    structure = source keep

SEMROLE_03_LOWSTUFF_NEUTRAL_SKY_SAFE
    LOW_VALUE_STUFF lowD = neutral keep
    LOW_VALUE_STUFF highD = weak skip
    no negative write

SEMROLE_04_MOVABLE_NEGATIVE_ONLY
    MOVABLE_THING highD = negative short-only
    structure = source keep

SEMROLE_05_STRUCTURE_POS_LOWSTUFF_HIGHD_NEG
    positive = STRUCTURE_ANCHOR & lowD
    negative = LOW_VALUE_STUFF & highD
    neutral = LOW_VALUE_STUFF & lowD

SEMROLE_06_TRUST_ROUTED
    if Q_sem low: fallback to D_g only
    if Q_sem high: use semantic role table
```

### 关键公式

语义 role 不再直接生成一个最终 scalar，而是生成三组 mask：

$$
M_{pos}(i), M_{neu}(i), M_{neg}(i) \in \{0,1\}
$$

TTT 写入使用：

$$
G_{commit} = G_{pos} + \lambda_{neu}G_{neu} - \gamma_{neg}G_{neg}
$$

其中：

$$
G_{pos} = \sum_i M_{pos}(i) J_i
$$

$$
G_{neu} = \sum_i M_{neu}(i) J_i
$$

$$
G_{neg} = \sum_i M_{neg}(i) J_i
$$

默认参数：

```text
lambda_neu = 0.85
gamma_neg = 0.003 / 0.005 first pass
branch = w0 only
layer = all, with logger for per-layer effect
```

### 必须记录指标

```text
per_group_pos_mass
per_group_neu_mass
per_group_neg_mass
per_group_source_skip_mass
per_group_ttt_update_norm_w0
per_group_ttt_update_cos_to_H9
per_group_ttt_update_cos_to_C9
per_group_long_write_mass
per_group_short_write_mass
```

### 成立标准

```text
Local pass:
    h10 [200,300) delta <= -5m
    and [400,600) delta <= +1m

Durable pass:
    h15 ATE delta <= -3m
    or h15 [200,300) delta <= -5m
    and durability_ratio >= 0.45
```

### 不满足时 Codex 自动尝试方向

```text
如果 LOW_VALUE_STUFF highD hard skip h10 有效但 h15 回退:
    改为 weak skip + W_short，不进 W_long。

如果 STRUCTURE positive h15 有效但 h10 弱:
    保留 structure positive，组合 D_g compact_kv。

如果 MOVABLE_THING coverage 太低:
    不继续 movable 分支；转向 structure/lowstuff。

如果所有 semantic role h10 都弱:
    检查 semantic group map 是否过粗；
    尝试 fine label split；
    或把语义降级为 trust routing，而不是 action mask。
```

---

## H3：语义必须同时控制 frame/global/SWA/TTT，否则 h10 修正会被 h15 洗掉

### 假设

只控制 frame attention 或只控制 TTT 写入不足以产生 durable correction。必须让同一 semantic role 同时作用在：

```text
frame attention source K/V
global/chunk attention source K/V
SWA cache source keep/replace
TTT long/short write
```

### 实验矩阵

使用 H2 中表现最好的两个 semantic role policy，做 memory path ablation。

```text
ALLMEM_01_FRAME_ONLY
    semantic role controls frame attention source K/V only

ALLMEM_02_FRAME_GLOBAL
    semantic role controls frame + global/chunk source K/V

ALLMEM_03_FRAME_GLOBAL_SWA
    semantic role controls frame + global + SWA cache source

ALLMEM_04_FRAME_GLOBAL_TTT
    semantic role controls frame + global + TTT write

ALLMEM_05_ALL_MEMORY
    semantic role controls frame + global + SWA + TTT

ALLMEM_06_ALL_MEMORY_WITH_LONG_SHORT
    same as ALLMEM_05, but TTT uses W_long/W_short lifecycle
```

### 各 memory path 的具体动作

#### Frame attention / global attention

保留 query，不删除输出 token，只控制 source K/V：

```text
source keep:
    STRUCTURE_ANCHOR lowD
    LOW_VALUE_STUFF lowD partial keep
    special tokens always keep

source skip:
    LOW_VALUE_STUFF highD
    MOVABLE_THING highD
    UNCERTAIN low trust
```

#### SWA

SWA 不直接强删，而是控制 cache source role：

```text
structure source:
    keep into cache

lowstuff lowD:
    keep with decay or partial weight

lowstuff highD / movable highD:
    do not enter long cache; allow short cache only
```

#### TTT

TTT 分成 long / short：

```text
W_long:
    structure lowD positive
    static thing lowD weak positive

W_short:
    lowstuff highD correction
    movable highD correction
    uncertain short correction
```

应用时：

$$
W_{apply}^{m+1} = W_{long}^{m+1} + \alpha_{short} W_{short}^{m+1}
$$

提交时：

$$
W_{commit}^{m+1} = W_{long}^{m+1}
$$

默认：

```text
alpha_short = 0.25 / 0.50
short_decay = 1 or 2 chunks
```

### 必须记录指标

```text
frame_attn_per_group_source_mass
global_attn_per_group_source_mass
swa_cache_per_group_keep_mass
swa_cache_per_group_replace_mass
ttt_per_group_long_mass
ttt_per_group_short_mass
h10_to_h15_per_group_overwrite_ratio
memory_path_ablation_delta_table.csv
```

### 成立标准

H3 成立要求：

```text
ALLMEM_05 or ALLMEM_06 must outperform single-path variants on h15.

Minimum gate:
    h15 ATE delta <= -3m
    or h15 [200,300) delta <= -5m
    durability_ratio >= 0.45

Downstream safety:
    [400,600) regression <= +1m
```

### 不满足时 Codex 自动尝试方向

```text
如果 FRAME_ONLY h10 强但 h15 弱:
    转 ALLMEM_04 / ALLMEM_06，不再扫 frame threshold。

如果 SWA 加入后 h15 回退:
    只保留 structure source 进 SWA；
    lowstuff highD 不进入 SWA cache。

如果 TTT 加入后 h10/h15 都回退:
    检查 semantic TTT role 是否过强；
    将 negative 从 W_long 移到 W_short；
    降低 gamma_neg。

如果 ALL_MEMORY h15 仍弱:
    执行 state attribution，判断是 TTT overwrite、SWA overwrite、global source overwrite，还是 merge/gauge overwrite。
```

---

## H4：当前 h10 修正失效，是因为后续 TTT/merge state overwrite，而不是语义 role 本身错误

### 假设

v22 attribution 显示 h10->h15 的 HMC/TTT state movement 很大。v23 要进一步按 semantic group 和 memory path 拆开：到底是哪类语义、哪条 memory path 覆盖了 h10 修正。

### 实验设计

对下面候选保存 h10/h15 state：

```text
H9 baseline
TTTSSP_02 from v21/v22 reference
TTT_LIFE_04 from v22
ALLMEM best h10 candidate
ALLMEM best h15 candidate
```

保存 state：

```text
HMC state
TTT W0/W1/W2
SWA cache K/V
global/chunk source cache
merge/gauge state
semantic role summaries
```

计算：

$$
OverwriteRatio = \frac{\|S_{h15} - S_{h10}\|}{\|S_{h10} - S_{base}\| + \epsilon}
$$

并按 memory path 和 semantic group 统计：

```text
TTT structure overwrite
TTT lowstuff overwrite
SWA structure overwrite
SWA lowstuff overwrite
global source overwrite
merge/gauge overwrite
```

### 可视化

```text
h10->h15 overwrite waterfall
memory path × semantic group heatmap
group-wise TTT W0/W1/W2 update norm
SWA K/V source replacement map
trajectory segment overlay before/after h10/h15
```

### 成立标准

H4 成立标准不是 ATE，而是解释力：

```text
1. 找到主要 overwrite path；
2. 该 path 的 overwrite ratio >= 0.5；
3. 该 path 与 h15 regression candidate 对应；
4. 对该 path 做保护后，h15 durability ratio 提升 >= 0.20。
```

### 不满足时 Codex 自动尝试方向

```text
如果 HMC/TTT overwrite 最大:
    转 TTT long/short lifecycle and protect structure-positive W_long。

如果 SWA overwrite 最大:
    转 SWA semantic cache role，限制 lowstuff highD 进入 SWA history。

如果 global source overwrite 最大:
    转 global/chunk source semantic keep/drop。

如果 merge/gauge overwrite 最大:
    语义 memory 不足以解决，进入 explicit trajectory/scale-state module。
```

---

## H5：Semantic Prior Generator 需要重新输出 memory-role，而不是只输出 A_tok

### 假设

原来的 `A_tok` 是单一 write gate，不足以表达 all-memory role。Semantic Prior Generator 应该输出一个结构化对象：

```text
SemanticMemoryRoleOutput
```

### 建议输出结构

```text
A_tok:
    legacy token write gate

G_sem_tok:
    coarse semantic group id

L_sem_tok:
    fine semantic label id

Q_sem_tok:
    semantic trust

V_sem_tok:
    semantic value

Role_frame_src:
    keep / weak_keep / skip / protect

Role_global_src:
    keep / weak_keep / skip / protect

Role_swa_cache:
    long_keep / short_keep / skip / replace

Role_ttt_write:
    positive_long / neutral / negative_short / no_write

B_chunk_geo:
    geometry-only chunk write budget

B_chunk_sem:
    semantic diagnostic budget, not directly used unless explicitly enabled
```

### 关键原则

```text
1. geometry 决定 eligibility；
2. semantic 决定 value and role；
3. mask quality 决定 trust / fallback；
4. chunk budget 主要由 geometry 决定，不被 semantic low-value 过度拉低；
5. sky / vegetation 不默认 negative；
6. structure 不默认强写，必须 lowD / high trust。
```

### 成立标准

```text
1. Role output 与 patch token 对齐；
2. no-op mode bit-level safe；
3. 每个 memory path 能选择 consume 或 ignore role；
4. debug 能还原每个 token 的 role 来源；
5. role policy 变化不会隐式改变 legacy A_tok unless enabled。
```

---

## 4. 实验执行顺序

### Phase 0：语义 all-memory plumbing 与 no-op gate

目的：确保实验基础可信。

运行：

```text
P0_01_semantic_role_noop_ignored
P0_02_semantic_role_pass_through_consumed
P0_03_semantic_role_debug_only_all_memory
```

通过标准：

```text
ATE drift <= 0.01m
pose txt max diff == 0 or <= 1e-6
HMC hash unchanged
semantic role summaries non-empty
```

如果失败，不进入后续。

---

### Phase 1：Passive semantic all-memory audit

目的：不改模型，先看语义 role 与 memory path 的关系。

对象：

```text
H9
C9
TTTSSP_02
TTT_LIFE_04
KVC_MEM_03
SEM_ROLE_01
```

记录：

```text
per_group coverage
per_group source attention mass
per_group SWA cache mass
per_group TTT update norm
per_group h10->h15 overwrite
per_group relation to [200,300) delta
```

通过标准：

```text
至少一个 semantic group / role 与 h10 improvement 或 h15 overwrite 有可解释关系。
```

如果没有任何关系：暂停语义 action，先修 semantic taxonomy 或 mask quality。

---

### Phase 2：Semantic role single-path ablation

目的：分清语义在不同 memory path 上的作用。

候选：

```text
FRAME_SEM_01_STRUCTURE_KEEP
FRAME_SEM_02_LOWSTUFF_HIGHD_SKIP
GLOBAL_SEM_01_STRUCTURE_KEEP
SWA_SEM_01_STRUCTURE_LONG_KEEP
TTT_SEM_01_STRUCTURE_POSITIVE
TTT_SEM_02_LOWSTUFF_HIGHD_SHORT_NEG
```

每个候选只跑：

```text
chunks = 6, 10, 16
horizon = h10, h15
```

通过标准：

```text
single-path h10 [200,300) delta <= -3m
or h15 ATE delta <= -1m without downstream regression
```

不满足时 Codex 直接转下一 path，不在该 path 继续阈值微扫。

---

### Phase 3：Semantic all-memory role combination

目的：测试同一 semantic role 同时控制多个 memory path 是否能产生 durable correction。

候选：

```text
ALLSEM_01_FRAME_GLOBAL_STRUCTURE_KEEP
ALLSEM_02_FRAME_GLOBAL_LOWSTUFF_HIGHD_SKIP
ALLSEM_03_FRAME_GLOBAL_SWA_STRUCTURE_LONG_KEEP
ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG
ALLSEM_05_FRAME_GLOBAL_SWA_TTT_ALL_ROLE
ALLSEM_06_ALL_ROLE_LONG_SHORT
```

通过标准：

```text
h10 [200,300) delta <= -5m
h15 ATE delta <= -3m or h15 [200,300) delta <= -5m
durability_ratio >= 0.45
[400,600) regression <= +1m
```

---

### Phase 4：Durability attribution 与 targeted repair

目的：如果 h10 强、h15 弱，定位被谁覆盖。

候选：

```text
best h10 from Phase 3
best h15 from Phase 3
H9
C9
```

输出：

```text
state_attribution_by_memory_path.csv
state_attribution_by_semantic_group.csv
h10_to_h15_overwrite_heatmap.png
memory_path_decision.md
```

根据 attribution 自动修复：

```text
TTT overwrite:
    enable W_long protect structure-positive

SWA overwrite:
    enable semantic SWA cache long/short split

global source overwrite:
    enable semantic global source keep/drop

merge/gauge overwrite:
    route to explicit online trajectory/scale-state branch
```

---

### Phase 5：No-GT selector only after durability gate

只有 Phase 3 或 Phase 4 有候选满足：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
durability_ratio >= 0.45
```

才启动 no-GT selector。

Selector proxy 候选：

```text
semantic structure coverage change
lowstuff highD source skip mass
TTT W_long structure mass
SWA long cache structure mass
context keep ratio
pose-step EMA scale proxy
```

selector 通过标准：

```text
Spearman(proxy, h15 ATE delta) <= -0.45
selected candidate expected h15 ATE delta <= -3m
false positive selector rate <= 25%
```

---

### Phase 6：Full online validation

只有 selector 通过后，跑 full online KITTI01。

Full validation 必须报告：

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
[200,300]
[200,400]
[400,600]
YawRMSE
Sim3Scale
semantic role summary
memory overwrite summary
```

晋级标准：

```text
Strong:
    ATE <= 30m
    and [200,300) <= 50m

Target:
    ATE <= 25m

Weak:
    ATE <= 33.0m
    and [200,300) improves over H9
    and [400,600) does not regress over C9
```

如果 full online 只比 C9 小幅提升但 `[200,300)` 更差，不算成功。

---

## 5. 必须可视化的内容

### 5.1 Semantic role dashboard

每个关键候选输出：

```text
RGB
D_g
semantic group
semantic fine label
semantic trust
Role_frame_src
Role_global_src
Role_swa_cache
Role_ttt_write
```

重点帧：

```text
chunk6 selected frames
chunk10 selected frames
chunk16 selected frames
[200,300) high-error frames
```

---

### 5.2 Memory path contribution dashboard

输出：

```text
memory_path × semantic_group heatmap
source_keep_ratio over time
TTT pos/neu/neg mass over time
SWA cache semantic composition over time
global source semantic composition over time
h10->h15 overwrite ratio over time
```

---

### 5.3 Trajectory dashboard

必须比较：

```text
H9
C9
best v22 diagnostic
best v23 candidate
```

输出：

```text
XY trajectory
per-frame error curve
per-100f ATE curve
[200,300) zoom
[400,600) zoom
Sim3 scale over time
Yaw drift over time
```

---

## 6. 记录文件规范

每个 run 必须落盘：

```text
run_config.yaml
kitti_benchmark.log
01.txt
hmc_state_hash.jsonl
context_skip_summary.jsonl
semantic_group_summary.jsonl
semantic_role_summary.jsonl
semantic_memory_path_summary.jsonl
ttt_write_debug.jsonl
swa_cache_debug.jsonl
global_source_debug.jsonl
trajectory_diagnostics.json
candidate_boundary.json
```

候选汇总必须生成：

```text
candidate_vs_H9_delta_by_horizon.csv
candidate_vs_C9_delta_by_horizon.csv
semantic_role_metrics.csv
memory_path_metrics.csv
durability_gate_summary.json
selector_gate_summary.json
full_online_gate_summary.json
```

---

## 7. 并行执行策略

本轮建议 4 条 Codex track 并行。

### Track A：Semantic plumbing / no-op / role logging

目标：保证 semantic group / role 真正进入所有 memory path。

优先修：

```text
G_sem_tok / L_sem_tok / Q_sem_tok / R_sem_tok
semantic_role_summary.jsonl
memory path role debug
```

如果 role logging 不完整，不允许其它 track 跑性能。

---

### Track B：Frame/global semantic source policy

目标：测试 semantic K/V source role。

候选：

```text
FRAME_SEM_01
GLOBAL_SEM_01
FRAME_GLOBAL_SEM_01
```

如果 h10 没有 `-3m` 级别 `[200,300)` 改善，停止该 track。

---

### Track C：SWA / TTT semantic memory policy

目标：测试 semantic role 是否能进入 memory commit。

候选：

```text
SWA_SEM_01
TTT_SEM_01
TTT_SEM_02
SWA_TTT_SEM_01
```

如果 h10 有信号但 h15 回退，自动转 long/short split。

---

### Track D：Durability attribution / repair

目标：解释并修复 h10->h15 washout。

输入：Track B/C best candidate。

如果 attribution 显示 merge/gauge overwrite 最大，立刻触发 online trajectory/scale-state module 实现，不继续纯语义 memory 微扫。

---


## 8. Codex Implementation Self-Audit：代码自查与 bug 定位计划

本节是 v23 新增的强制前置步骤。它的目标是让 Codex 在执行性能矩阵之前，先证明“计划里的 semantic role policy 真的接到了模型路径上”。过去几轮已经多次出现过类似问题：read-only 模式没有转发 compact K/V 参数、Stage C cache 的 global frame index 不对、旧 run directory 混入新 JSONL、pass-through prior 导致 write score base 退化等。v23 必须把这类问题在 full run 之前拦住。

### 8.1 自查整体目标

Codex 需要回答下面五个问题。

第一，语义信号是否真的存在：

```text
G_sem_tok / L_sem_tok / Q_sem_tok / V_sem_tok / R_sem_tok
```

这些张量是否在每个 chunk 中有正确 shape、正确 token 对齐、正确语义 coverage，并且不是全零、全一或全 unknown。

第二，语义信号是否真的被所有 memory path 消费：

```text
frame attention source path
chunk/global attention source path
SWA cache/source path
TTT write/replay path
long-short lifecycle path
```

第三，每条 path 的 no-op / pass-through 是否安全。如果语义控制关闭，或者 semantic role 全部设为 neutral/keep，输出必须严格等价于 baseline。

第四，控制生效时 debug 是否能证明实际生效。例如 compact K/V 必须有非零 `num_context_source_skip_applied`，TTT semantic role 必须改变 per-role write mass，SWA semantic source policy 必须改变 source keep / replace 统计。

第五，Codex 是否能在失败时自动定位到具体层级：

```text
参数没传到 CLI；
HMC 构建了但 model 没消费；
model 消费了但 mask 全空；
mask 生效了但所有 token 都被 protected；
debug 没记录；
run 目录污染。
```

---

### 8.2 必查代码文件与责任链

Codex 必须逐文件检查下面的责任链。不是只 grep 参数名，而是要确认数据从入口到模型消费路径全链路闭合。

| 文件 | 必查内容 | 最常见 bug | 通过标准 |
|---|---|---|---|
| `tools/run_attention_cue_experiment.sh` | 环境变量是否转发到 Python CLI | readonly / smoke 模式漏传 `CONTEXT_SOURCE_SKIP_*` 或 `SEMANTIC_*` | bash trace 中能看到完整 CLI 参数 |
| `tools/run_v23_candidate_rollout.sh` | short-rollout launcher 是否转发 semantic all-memory 参数 | candidate 目录复用、stale JSONL 混入 | forced run 前旧目录移动到 `.INVALID_*` |
| `run_pipeline_abc_v2.py` | argparse choices、Stage C cache、semantic prior、HMC 参数透传 | 参数有 env 但 argparse 不接受；cache 用 local chunk index | `run_config.yaml` 与 expected config 完全一致 |
| `loger/pipeline/semantic_prior_generator.py` | semantic group / role / trust / value 的 token 投影 | 只输出 `S_tok`，没有 exact role id | `semantic_role_summary.jsonl` 中 group/role 非空且覆盖合理 |
| `loger/pipeline/hybrid_memory_controller.py` | `HybridMemoryControlPrior` 是否携带并路由语义 role | prior 构建了但 TTT/SWA/global path 没用 | debug 中每个 memory path 都有 semantic role consumption 字段 |
| `loger/models/pi3.py` | frame/global/chunk source skip 是否真正进入 attention | 请求了 skip 但 `context_source_skip_requested=false` | `num_context_source_skip_applied > 0` 且 keep ratio 合理 |
| `loger/models/layers/attention.py` | compact K/V 是否保留 query length、只压缩 source K/V | 错删 query token、special token、空 source | output token length 与 baseline 一致，无 empty source |
| `loger/pipeline/ttt_write_controller.py` | TTT positive/neutral/negative role 是否改变 replay/write | role 只进日志，不进 multiplier/replay | per-role write mass 与 baseline 不同，branch/layer mask 正确 |
| SWA 相关路径 | semantic source keep / replace / cache policy 是否生效 | 只改 read，不改 cache；或只改 cache 不改 read | SWA source summary 中 semantic group keep/replace 非零 |
| 诊断脚本 | h10/h15 delta 是否同帧 intersection 计算 | 不同帧区间比较导致假收益 | report 明确记录 matched frame count |

---

### 8.3 Codex 必跑静态检查

在任何 rollout 前，Codex 必须先跑这些静态检查。

```bash
python -m py_compile \
  run_pipeline_abc_v2.py \
  loger/pipeline/semantic_prior_generator.py \
  loger/pipeline/hybrid_memory_controller.py \
  loger/pipeline/ttt_write_controller.py \
  loger/models/pi3.py \
  loger/models/layers/attention.py

bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v23_candidate_rollout.sh
bash -n tools/run_v23_matrix.sh
```

Codex 还要做参数连通性 grep / diff 检查。建议检查项：

```text
CONTEXT_SOURCE_SKIP_IMPL
CONTEXT_SOURCE_SKIP_MASK
SEMANTIC_ROLE_POLICY
SEMANTIC_MEMORY_PATHS
SEMANTIC_GROUP_MODE
SEMANTIC_LOWSTUFF_POLICY
SEMANTIC_STRUCTURE_POLICY
TTT_SEMANTIC_ROLE_MODE
SWA_SEMANTIC_SOURCE_POLICY
GLOBAL_SEMANTIC_SOURCE_POLICY
FRAME_SEMANTIC_SOURCE_POLICY
STAGE_C_CACHE_MODE
STAGE_C_CACHE_REQUIRE_HIT
HMC_IGNORE_SEMANTIC_PRIOR
```

通过标准：每个关键参数至少在下面四层都能找到闭环：

```text
shell env / CLI
run_pipeline_abc_v2.py argparse
HybridMemoryController 或模型 control dict
实际消费位置 + debug summary
```

---

### 8.4 Codex 必跑 no-op / pass-through smoke

这一步的目标是验证“接入新模块本身不会改变结果”。

固定短序列：

```text
END_FRAME = 128
parent = H9 或 C9 locked protocol
Stage C cache = require-hit read
semantic prior = noop / pass_through
HMC ignore semantic prior = 1 for hard no-op
```

必须跑：

```text
NOOP_01_base_no_semantic
NOOP_02_stageC_cache_read_hmc_ignore
NOOP_03_semantic_noop_hmc_ignore
NOOP_04_pass_through_consumed
NOOP_05_all_neutral_roles_consumed
```

通过标准：

```text
NOOP_02/03/04/05 vs NOOP_01:
    max_abs_pose_diff <= 1e-6 on short smoke
    ATE delta <= 0.001m on short smoke
    HMC hash sequence identical unless explicitly expected otherwise
    prior_hmc_write_score_mean matches baseline when role is neutral/pass-through
```

如果 `pass_through` 改变输出，Codex 不允许继续性能实验。必须检查：

```text
A_tok=ones 是否被 normalize 成全 0；
stage_d base 是否被 semantic prior 覆盖；
HMC ignore 与 HMC consumed 两条路径是否处理一致；
semantic prior 的 all-neutral 是否真的等价于 keep-all。
```

---

### 8.5 Codex 必跑 semantic cache correctness smoke

语义 cache 是高风险模块，必须先保证 cache 命中和 frame index 正确。

必须检查：

```text
1. full sequence chunk cache 命中率；
2. sliced causal fork 中 global_start/global_end 是否正确；
3. 每个 chunk 的 semantic masklet count；
4. 每个 coarse group 的 token coverage；
5. unknown / uncertain / empty masklet 比例；
6. fine label 是否可用；如果 fine sky/vegetation 不可用，禁止声称已经测试 sky-specific skip。
```

必须落盘：

```text
semantic_cache_hit_summary.json
semantic_group_coverage_by_chunk.csv
semantic_group_coverage_by_token.csv
semantic_cache_frame_index_audit.jsonl
semantic_unknown_ratio.csv
```

通过标准：

```text
cache hit rate = 1.0 for required chunks
chunk coverage mean >= 0.90
focus chunks 6-10 coverage >= 0.90
coarse group id 非空
if sky/vegetation fine split unavailable:
    semantic_fine_split_available = false
    sky_skip candidates 自动降级为 lowstuff/structure coarse candidates
```

如果 cache miss：

```text
Codex 先检查 global frame start/end；
再检查 chunk filename convention；
再检查 Stage C cache mode 是否是 read/require-hit；
禁止 inline compute 作为 no-op benchmark。
```

---

### 8.6 Codex 必跑 compact K/V correctness smoke

VGGT4D-style source skip 的关键不是普通 attention bias，而是只让高风险区域退出 K/V source，query token 仍保留。Codex 必须验证 compact K/V 真的这样做。

必须跑：

```text
KV_SMOKE_01_frame_early_dg_q80_compact
KV_SMOKE_02_frame_global_dg_q80_compact
KV_SMOKE_03_sem_lowstuff_highD_compact
KV_SMOKE_04_structure_rescue_compact
```

必须记录：

```text
context_source_skip_requested
context_source_skip_impl
context_source_skip_mask
num_context_source_skip_applied
mean_context_source_keep_ratio
min_context_source_keep_ratio
max_context_source_skip_tokens
num_context_empty_source_events
query_length_before
query_length_after
kv_source_length_before
kv_source_length_after
num_protected_special_tokens
```

通过标准：

```text
context_source_skip_requested = true
context_source_skip_impl = compact_kv
num_context_source_skip_applied > 0
num_context_empty_source_events = 0
query_length_before == query_length_after
kv_source_length_after < kv_source_length_before for at least one controlled layer
special / camera / register tokens protected
mean_context_source_keep_ratio between 0.70 and 0.98 for q80 smoke
```

如果 `requested=false`：

```text
Codex 先检查 launcher 是否转发 CONTEXT_SOURCE_SKIP_*；
再检查 run_pipeline_abc_v2.py 是否透传；
再检查 HMC control dict 是否覆盖 readonly mode。
```

如果 keep ratio = 1.0：

```text
检查 mask 是否全 false；
检查 D_g threshold 是否错误；
检查 semantic role 是否未投影到 token；
检查 protected token mask 是否覆盖了所有 patch tokens。
```

如果 empty source events > 0：

```text
自动切换为 soft skip 或提高 keep floor；
强制保留 structure / special / same-frame minimal source。
```

---

### 8.7 Codex 必跑 all-memory consumption smoke

这一步验证 semantic role 不只是 frame attention 生效，而是真的进入所有 memory path。

候选：

```text
PATH_SMOKE_01_frame_only
PATH_SMOKE_02_global_only
PATH_SMOKE_03_swa_only
PATH_SMOKE_04_ttt_only
PATH_SMOKE_05_all_memory
```

每个 smoke 使用同一套 semantic role：

```text
STRUCTURE_ANCHOR + lowD -> positive / keep
LOW_VALUE_STUFF + lowD -> neutral / keep
LOW_VALUE_STUFF + highD -> skip / short negative
UNCERTAIN -> fallback geometry
```

必须记录：

```text
frame_semantic_source_consumed
chunk_global_semantic_source_consumed
swa_semantic_source_consumed
ttt_semantic_role_consumed
per_path_role_mass
per_path_keep_ratio
per_path_write_score_mean
per_path_update_norm
per_path_delta_vs_noop
```

通过标准：

```text
每个 path 单独开启时：
    只有对应 path 的 consumed flag 为 true；
    其它 path consumed flag 为 false；
    对应 path 的 role_mass 非零；
    输出不报错，empty source = 0。

all_memory 开启时：
    四条 path consumed flag 全为 true；
    per-path debug 均有有效 role_mass。
```

如果某条 path 只有 requested 没有 consumed：

```text
Codex 优先修 path 接线；
不允许进入该 path 的性能矩阵；
其它 path 可以并行继续。
```

---

### 8.8 Codex 必跑 support correctness smoke

因为 `past_only / full_chunk / full_chunk_no_overlap` 会影响主 read cue，Codex 需要先证明 support 实现正确。

必须落盘：

```text
support_index_summary.csv
support_index_by_frame.jsonl
support_future_mass.csv
support_overlap_exclusion_check.csv
```

通过标准：

```text
past_only:
    future_ratio_mean = 0
    weighted_future_mass = 0

full_chunk_true:
    future_ratio_mean ≈ 0.5 for symmetric chunk
    count mean = chunk_size - 1

full_chunk_no_overlap:
    overlap seam support = 0
    not equal to full_chunk_true

past_plus_future_light_real:
    support count can equal full_chunk, but weighted_future_mass < full_chunk_true
```

如果 `full_chunk_no_overlap` 等于 `full_chunk_true`：

```text
说明 overlap metadata 没接上；
Codex 必须修 overlap_frames / seam mask；
禁止声称 no-overlap support 已验证。
```

---

### 8.9 Codex 必跑 stale-run / contamination check

v23 任何强制重跑都必须避免旧数据混入。

规则：

```text
如果 run directory 已存在且 FORCE=1：
    移动旧目录到 .INVALID_RERUN_TIMESTAMP
    新建干净目录

如果同一个 run_id 出现两个 kitti_benchmark.log 或多个 summary.jsonl 来源：
    标记 INVALID_DUPLICATE_WRITE
    不纳入 report
```

必须落盘：

```text
run_dir_cleanliness_report.json
invalidated_run_dirs.jsonl
duplicate_write_check.csv
```

通过标准：

```text
所有 report 只聚合 DONE 且 clean 的 run；
INVALID / partial / failed / stale 目录不参与；
每条 row 有唯一 run_config hash。
```

---

### 8.10 Codex 自动修复方向

如果 self-check 不通过，Codex 按下面顺序自动尝试修复，不需要等待人工确认。

#### Case A：参数没有传到模型

现象：

```text
run_config 中有请求，但 model summary requested=false；
或 shell env 有请求，但 run_config 没有。
```

处理：

```text
1. 修 tools/run_attention_cue_experiment.sh / run_v23_candidate_rollout.sh；
2. 修 run_pipeline_abc_v2.py argparse；
3. 修 HybridMemoryController control dict；
4. 重跑对应 smoke。
```

#### Case B：semantic group 全 unknown 或 coverage 太低

处理：

```text
1. 检查 Stage C cache path 与 global frame index；
2. 检查 MaskletOutput.G_sem 是否保留；
3. 检查 semantic_prior_generator projection；
4. 若 fine label 不可用，降级 coarse group，不跑 sky-specific candidate。
```

#### Case C：compact K/V 请求了但没有实际压缩 source

处理：

```text
1. 检查 D_g / semantic mask 是否投影到 token；
2. 检查 protected mask 是否错误保护全部 patch；
3. 检查 attention.py 是否走 compact_kv path；
4. 若 hard compact 产生 empty source，自动改 soft skip 或 keep_min_source。
```

#### Case D：pass-through / neutral role 改变 baseline

处理：

```text
1. 检查 all-neutral role 是否误作 skip；
2. 检查 A_tok=ones 是否破坏 stage_d base；
3. 检查 write score normalization 是否把 base prior 归零；
4. 修复后必须重新过 no-op gate。
```

#### Case E：single-path smoke 生效，all-memory smoke 不生效

处理：

```text
1. 检查各 path 是否互相覆盖 control dict；
2. 检查 frame/global/SWA/TTT 是否共用同名字段导致后写覆盖前写；
3. 为每条 path 增加独立 consumed flag；
4. 重跑 all-memory smoke。
```

#### Case F：h10 强但 h15 失败

这不是代码 bug，但 Codex 应自动转向 durability 修复：

```text
1. 保存 h10/h15 HMC / TTT / SWA / merge states；
2. 计算 overwrite ratio；
3. 如果 TTT overwrite 最大，转 skip-aware TTT long-write；
4. 如果 SWA overwrite 最大，转 semantic SWA cache commit；
5. 如果 global/source overwrite 最大，转 semantic global source persistence；
6. 如果 merge/gauge overwrite 最大，转 explicit trajectory/scale-state module。
```

---

### 8.11 Codex self-check 通过后的输出格式

Codex 必须在 `implementation_audit/codex_self_check_report.md` 中给出下面的最终表格。

| Gate | Status | Evidence | If failed, fix attempted |
|---|---|---|---|
| py_compile / bash -n | pass/fail | command output | file path |
| CLI pass-through | pass/fail | config diff | launcher/parser fix |
| no-op parity | pass/fail | pose diff / ATE diff | prior normalization fix |
| Stage C cache | pass/fail | hit rate / coverage | global frame cache fix |
| compact K/V | pass/fail | keep ratio / applied count | path/threshold/protection fix |
| semantic group projection | pass/fail | group coverage | projection/cache fix |
| frame path consumed | pass/fail | role mass | HMC/model fix |
| global path consumed | pass/fail | role mass | HMC/model fix |
| SWA path consumed | pass/fail | source summary | SWA hook fix |
| TTT path consumed | pass/fail | write mass / branch norm | controller fix |
| support audit | pass/fail | support csv | overlap metadata fix |
| stale-run protection | pass/fail | invalid dirs report | launcher fix |

只有当下面条件同时满足时，才允许启动 v23 性能矩阵：

```text
no_op_parity_pass = true
stage_c_cache_pass = true
semantic_group_projection_pass = true
at_least_one_memory_path_consumed_pass = true
compact_kv_or_semantic_role_path_pass = true
stale_run_protection_pass = true
```

如果 all-memory consumed 没有全部通过，但至少一个单 path 通过，则只允许启动该单 path 的 Phase 2 ablation，不允许启动 Phase 3 all-memory combination。

---

## 9. 停止规则

本轮必须避免无限微扫。

```text
Rule 1:
    如果某个 semantic role family 连续 4 条 candidate h10 [200,300) 改善都 > -3m，停止该 family。

Rule 2:
    如果某个 family h10 强但 h15 durability ratio < 0.2，停止 threshold 微扫，转 lifecycle / overwrite attribution。

Rule 3:
    如果 lowstuff skip 连续导致 h15 或 [400,600) 回退，禁止 hard skip，改为 neutral / short-only。

Rule 4:
    如果 structure positive 在 h15 有稳定小收益，但 h10 弱，不丢弃；用于组合，不继续单独扫。

Rule 5:
    如果 all-memory semantic role 仍不能达到 h15 gate，且 attribution 指向 merge/gauge overwrite，则 TTT/SWA/global semantic memory 降级为 stabilizer，Target-25 主线转 explicit online trajectory/scale-state module。

Rule 6:
    任何 short rollout 未过 durability gate，不允许 full online validation。
```

---

## 10. 本轮预期结论形式

v23 结束时，不应该只说“ATE 有没有降”。必须给出下面三类结论之一。

### 结论 A：语义 all-memory 成立

条件：

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m
durability_ratio >= 0.45
```

后续：启动 selector + full online。

---

### 结论 B：语义只有短期 source/read 价值

条件：

```text
h10 有强信号
h15 被 wash out
attribution 指向 TTT/SWA/global overwrite
```

后续：继续做 memory lifecycle / overwrite protection，而不是阈值微扫。

---

### 结论 C：语义 memory 不足以支撑 Target-25

条件：

```text
all-memory semantic role h10/h15 都不过 gate
或 h15 durability 长期失败
或 attribution 指向 merge/gauge trajectory-state
```

后续：

```text
semantic memory 作为 regularizer / stabilizer；
Target-25 主线转 explicit online trajectory-state / scale-state module；
保留 semantic role 辅助 trajectory-state module 的 source reliability。
```

---

## 11. 一句话总结

v23 的目标不是继续证明“语义有一点帮助”，而是验证：

> Semantic Prior Generator 能否从单一 TTT write prior 升级为 frame/global/SWA/TTT all-memory role controller，并把 h10 短期修正变成 h15 仍然有效的 durable trajectory correction。

如果不能，就要明确承认：语义 memory 目前只能作为 stabilizer，Target-25 需要显式 online trajectory / scale-state 模块。

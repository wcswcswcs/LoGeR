# ACL2 v45：代码审计、C9 去 chunk-id 化、组件贡献归因、C23 support 与自适应 tri-replay 实验计划

日期：2026-06-07  
目标对象：LoGeR Pipeline v2 / Hybrid Memory Controller / READ path / SWA / TTT write / Semantic Prior Generator  
当前开发主序列：KITTI Odometry sequence 01  
当前最好可部署 baseline：`C9_P0_R2`，full-online ATE = `33.7629421029m`  
阶段目标：先把 `C9_P0_R2` 变成更干净、可解释、可迁移的 baseline，再在它上面推进 KITTI01 full-online ATE，最终目标是进入 `ATE < 30m`。

---

## 0. 本轮计划为什么必须重写

前几轮实验已经说明，继续在当前 C9 黑箱上叠加语义、SWA 或 TTT 小规则，很难产生稳定进展。当前问题不是单个候选不够强，而是实验基础本身有三类不清楚：

1. `C9_P0_R2` 仍然含有 absolute chunk-id policy，例如 `read_beta_frame_chunks`、tri-replay chunk gamma map、commit EMA chunk list。这些在 KITTI01 上能复现当前 best，但不能作为可迁移策略。
2. C9 中每个组件的贡献没有完全拆清楚。v43 已经做了第一轮 leave-one-out attribution，但这只能说明“移除某组件会怎样”，不能完整解释组件交互。
3. 当前主 read cue `acl2.gg.qq.low.g2_3.past_only.headmean.robustq` 可能还有增强空间。它只和过去帧比较，但 LoGeR chunk 内推理本来是 bidirectional 的，因此 `full_chunk`、fixed offset、near support 等都值得在当前 C9 体系下重新验证。

所以 v45 的核心不是继续做大矩阵，而是先把问题拆干净：

```text
先清理 C9 的不可迁移成分；
再量化每个组件贡献；
再验证 C23 support 是否还有 read cue 增强；
再把 TTT tri-replay 从固定百分比改成 training-free 自适应分组；
最后才做少量 full-online 组合。
```

---

## 1. 项目边界：禁止再次走偏

本项目是 training-free memory-control 工作，不是训练新模型，也不是为 KITTI01 拟合触发器。

本轮明确禁止：

```text
禁止训练 trigger / selector / classifier / learned role router。
禁止用 oracle ATE / GT pose / GT trajectory 作为 runtime action。
禁止用 absolute chunk id 作为最终 runtime policy。
禁止根据 KITTI01 或任何单一 sequence 专门调参。
禁止把 short rollout、fixed-window diagnostic、hook smoke、GT diagnostic 写成 deployable success。
禁止把 semantic cache 命中、no-op parity、action smoke 写成 ATE 成功。
```

允许：

```text
允许 fixed chunk / freeze / oracle 做 diagnostic，但必须标注不可部署。
允许用其他 sequence 做 sanity / failure-mode diagnosis，但不能为这些 sequence 调参。
允许用 SemanticKITTI sparse projection 做 offline trust calibration，但不能作为 runtime GT action。
允许使用 training-free unsupervised clustering / robust statistics 来决定 token groups，但不能从 ATE label 学规则。
```

---

## 2. 代码审计结论

### 2.1 审计范围

我独立解包并检查了 `core_code_audit_pack.zip`。包内包含 `README_for_new_programming_ai.md`、`CORE_CODE_INVENTORY.csv`、风险审核文档，以及 `files/` 下的核心代码。`package_manifest.json` 记录 mandatory file count 为 22，`MISSING_FILES.md` 写明没有 mandatory 文件缺失。

我对打包出的 Python 文件执行了 `py_compile`，包括：

```text
run_pipeline_abc.py
run_pipeline_abc_v2.py
loger/pipeline/hybrid_memory_controller.py
loger/pipeline/ttt_write_controller.py
loger/pipeline/dynamic_cue_extractor.py
loger/pipeline/semantic_prior_generator.py
loger/pipeline/video_masklet_frontend.py
loger/pipeline/gt_semantic_provider.py
loger/models/pi3.py
loger/models/layers/attention.py
tools/*.py
```

结果：语法编译通过。这只能说明文件没有明显语法错误，不代表逻辑正确。

### 2.2 打包是否和 v44 计划一致

总体上，代码包符合 v44 的最小 mandatory list：主入口、HMC、TTT write、READ/SWA hook、attention、Semantic Prior、Video Masklet、GT semantic provider、diagnostic tools 和 v43/v42 launchers 都在包中。

但它不是一个完整可运行仓库，也不是足够完整的深度审计包。以下依赖文件没有打包，虽然它们不在 v44 mandatory list 中，但新的编程 AI 做实际运行或完整追踪时会需要：

```text
loger/pipeline/geometry_backbone.py
loger/utils/rotation.py
run_geometry_backbone_inference.py
inference_dynamic_cue_extractor.py
配置文件和 checkpoint 配置样例
eval/long_eval_script/kitti_benchmark
v43_noop_gate_report.py
v42_full_online_report.py
部分 Phase report 生成脚本
```

因此，`MISSING_FILES.md` 的“无缺失”只表示 v44 最小清单无缺失；如果目标是让新 AI 能完整复现实验，下一版代码包需要补这些 supporting files。

### 2.3 已确认的高风险代码问题

#### 问题 A：absolute chunk-id policy 仍然在 C9 主配置中

`tools/run_v43_full_candidate.sh` 中，C9 locked default 直接写了：

```text
READ_BETA_FRAME_CHUNKS=5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25
TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=5:0.005,...,12:0.003,16:0.0003
TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=5:0.35/0.12/0.85,...,16:0.35/0.08/0.85
TTT_WRITE_COMMIT_EMA_CHUNKS=5,6
```

这解释了为什么 C9 可复现但不干净。后续必须去掉这些 absolute chunk-id policy。用户的判断是对的：这些参数相差并不大，第一步不应该换成不可靠的 health-driven rule，而应该固定成全局值，并审计固定化损失来自哪里。

#### 问题 B：v43 组件归因只是 leave-one-out，不是完整贡献分解

`tools/v43_registry_summarize.py` 的 component ledger 计算的是：

$$
E_i = ATE(C9 - i) - ATE(C9)
$$

如果 $E_i > 0$，说明移除组件 $i$ 会变差，因此组件 $i$ 有正贡献。这个账本有用，但不是完整 ablation，因为组件之间有交互。

例如，移除 TTT tri-replay 退化 `+2.4469m`，移除 tri gamma chunk map 退化 `+0.9710m`，但这不能简单相加成真实贡献；因为 tri-replay 是否有用可能依赖 read beta、commit EMA、native mix、SWA overlap replacement 等上下文。

#### 问题 C：TTT `two_replay` / `separate_replay` 路径疑似存在实际 bug

在 `loger/pipeline/ttt_write_controller.py` 的 `maybe_apply_two_replay_negative(...)` 中，`tri_replay` 分支内已经 `return out_w0, out_w1, out_w2`，后面保留了旧的 negative replay 代码块，属于不可达 dead code。更严重的是，函数入口允许：

```text
two_replay
separate_replay
pos_neg_replay
tri_replay
```

但当前代码在非 tri replay 模式下可能会跳过 `pos_w0 / neu_w0 / neg_w0` 的定义，然后继续执行依赖这些变量的逻辑。这意味着：

```text
C9 使用 tri_replay，所以 C9 不受这个 bug 直接影响。
但 two_replay / separate_replay / pos_neg_replay 相关历史或未来候选不可信，必须先修再测。
```

Codex 必须在新实验开始前修复或明确禁用这些模式，避免未来实验误用。

#### 问题 D：`full_chunk_no_overlap` 与旧 `overlap_excluded` 名字不等价

`hybrid_memory_controller.py` 里的 `_acl2_support_indices(...)` 对 `full_chunk_no_overlap` 有真实 seam 排除逻辑，但对旧的 `overlap_excluded / noovlp / ovlp_only` 有注释说明：由于当前 builder 不一定接收 chunk-boundary metadata，这些名字在该分支会 fallback 到 full support。

因此后续 C23 support sweep 必须使用明确名字：

```text
full_chunk
full_chunk_no_overlap
past_only
near12
off246
past_plus_future_light_real
```

不能把旧 `overlap_excluded` 当作真实 no-overlap。

#### 问题 E：C9 没有语义参与

C9 的 Stage C semantic 默认关闭，`stage_c_mode = none`。C9 的写入分数是：

$$
write\_score = stage_d \cdot \sqrt{1-D_g}
$$

其中 `stage_d` 是几何/动态 rank 产生的写入资格，`D_g` 是 C23 attention cue，不是 semantic label。C9 不是 Semantic Prior Generator 成功案例。

---

## 3. 当前实验结果的独立判断

v43 的数据说明：

```text
C9 locked repeat 精确复现：ATE = 33.76294210291885m。
flat no-chunk-id 最好仍然退化到 35.2952180149m，delta vs C9 = +1.5323m。
移除 TTT tri-replay 后 ATE = 36.2098947787m，delta vs C9 = +2.4470m。
移除 tri gamma chunk map 后 ATE = 34.7339675087m，delta vs C9 = +0.9710m。
移除 commit EMA 后 ATE = 34.2513275668m，delta vs C9 = +0.4884m。
移除 read beta chunk map、SWA overlap replacement、native mix 的 full ATE 影响都接近 neutral。
最好的 semantic READ full-online 候选是 SEM_READ_03_C23_RESID_READ_ONLY，ATE = 33.4875667508m，delta vs C9 = -0.2754m，未达到最低有效进展门槛。
```

这说明：

```text
1. C9 的主要贡献不是语义。
2. C9 的主要贡献来自 TTT tri-replay 及其 chunk-specific gamma / no-chunk-id interaction。
3. C9-flat 退化不代表去 chunk-id 不可行，而是现有 fixed-value 方案没有保留关键交互。
4. semantic READ 目前有小幅 full-online 收益，但还不足以成为 Target-30 主线。
```

当前离阶段目标仍然很远：

$$
33.7629421029 - 30.0 = 3.7629421029m
$$

v43 的最好 semantic READ 只推进了：

$$
33.7629421029 - 33.4875667508 = 0.2753753521m
$$

所以现在不能继续把希望放在一个小 semantic READ 候选上。必须先把 C9 机制拆清楚，再找真正能带来米级提升的方向。

---

## 4. 本轮实验整体目标

v45 的目标不是直接铺开大矩阵，而是让下一轮实验有三类切实推进：

1. **机制清理**：把 C9 中所有 absolute chunk-id policy 去掉或固定化，得到一个可迁移的 `C9-Clean` baseline。
2. **贡献归因**：用 full-online component attribution 和小规模 interaction attribution 解释 C9 的 ATE 下降来自哪里。
3. **可提升方向验证**：只验证两个最有可能带来提升的方向：C23 support 增强与 TTT tri-replay 自适应分组。

本轮不以 semantic all-memory 为主线。语义只保留为一个 minimal READ full-online branch，因为目前它只给出 `-0.275m` 的小幅收益，不足以支撑大规模探索。

---

## 5. 核心假设

### H1：C9 的 absolute chunk-id policy 可以被固定全局值替代，但必须保留 TTT tri-replay 这个组件

用户指出 C9 的 chunk-specific 参数相差不大，所以不应继续保留绝对 chunk 编号。v43 的 flat 退化很大，但它同时改了多个东西，不能直接推出“去 chunk-id 不可能”。

本轮要验证：

```text
把 read beta、tri gamma、commit EMA 从 chunk map 改成固定值后，是否能保持接近 C9 的 ATE。
```

成功标准：

```text
C9-Clean acceptable:
    ATE <= C9 + 0.30m

C9-Clean promising:
    ATE <= C9 + 0.10m

C9-Clean success:
    ATE <= C9
```

如果固定化后仍比 C9 差超过 `0.5m`，则说明 C9 的 chunk-specific schedule 是关键交互，不能简单平均，需要先做机制归因而不是继续调 fixed scalar。

### H2：C9 组件贡献需要从 leave-one-out 升级为 interaction-aware attribution

v43 的 leave-one-out 已经说明 TTT tri-replay 是最大正组件，但没有解释组件交互。

本轮要计算：

$$
E_i = ATE(C9-i)-ATE(C9)
$$

以及二阶交互：

$$
I_{ij} = ATE(C9-i-j)-ATE(C9)-E_i-E_j
$$

如果 $I_{ij}$ 很大，说明两个组件不能独立解释，后续不能用单组件 ablation 做结论。

### H3：C23 past 不是必然最优；chunk 内 full support 和固定 offset support 可能增强 read cue

C9 使用：

```text
acl2.gg.qq.low.g2_3.past_only.headmean.robustq
```

它只比较过去帧。这个选择符合 causal streaming 直觉，但 LoGeR 的 chunk 内推理本身是 bidirectional 的。由于每个 chunk 的输入帧在当前 chunk 内都已知，使用 `full_chunk` 或 `off246` support 不是外部未来泄漏，而是 chunk 内 bidirectional inference 的合理利用。

本轮要验证：

```text
C23 full_chunk / off246 / near12 / past+future 是否能在当前 C9-Clean 或 C9 原始协议上改善 READ path。
```

### H4：固定百分比 tri-replay 不是最优；training-free 自适应分组可能更稳

C9 使用固定：

```text
positive_frac = 0.35
negative_frac = 0.12
neutral_lambda = 0.85
```

这可能对 KITTI01 的 chunk5-12 有效，但不是普适机制。更合理的 training-free 思路是：根据当前 chunk 的 `update_conflict_energy` 分布自动划分 positive / neutral / negative。

本轮不训练任何 classifier，只测试 unsupervised rule：

```text
1D k-means K=3 on token risk
Otsu threshold on risk histogram
robust MAD threshold
risk-distribution quantile with adaptive mass cap
```

成功标准：

```text
adaptive tri-replay full ATE <= C9-Clean - 0.3m
或者在 C9 原始协议上 <= C9 - 0.3m
且 [400,600) regression <= +1m
```

### H5：跨场景验证必须做，但不能抢在 KITTI01 机制成立之前

跨 sequence 验证的意义是检查泛化，不是调参。只有当 KITTI01 上有明确 full-online 改善后，才在 KITTI00/02/05 做同配置 sanity。

---

## 6. 实验阶段设计

## Phase 0：代码和 baseline hard gate

### 目标

确保新一轮不是在漂移的 baseline 上做实验，并修复会污染结论的代码问题。

### 必做任务

1. 精确复现 C9：

```text
P0_C9_REPEAT
expected ATE = 33.7629421029m
tolerance = 0.03m
hmc_rows = 38
Stage C semantic = off
```

2. 修复或禁用 TTT `two_replay / separate_replay / pos_neg_replay` 路径。

如果不准备使用这些模式，本轮必须显式在 launcher 中禁止它们；如果准备测试，必须先加 smoke，确认非 tri mode 不会触发未定义变量或 dead-code 路径。

3. 生成 effective config diff：

```text
effective_config.yaml
effective_config_diff_vs_C9.json
chunk_id_policy_audit.json
stage_c_semantic_disabled_confirm.json
```

### 记录指标

```text
ATE / Rot / RPE_t / RPE_r / FinalErr
[200,300), [400,600)
hmc rows
stage_c_mode
read_beta_frame_chunks
tri gamma chunk map
commit_ema_chunks
semantic enabled flag
effective_config unexpected diff count
```

### 通过标准

```text
abs(ATE - 33.7629421029) <= 0.03m
hmc_rows = 38
unexpected_config_diff_count = 0
Stage C semantic = off
```

如果 Phase 0 不过，Codex 必须先修 baseline，不允许启动 Phase 1。

---

## Phase 1：C9 去 absolute chunk-id 化，建立 C9-Clean

### 目标

把 C9 的 chunk-specific schedule 改成固定全局值，并评估损失到底来自哪里。

### 设计原则

用户明确不接受把 chunk-specific 规则替换成尚未证明的 health-driven rule。本阶段只做固定值，不做 health-driven。

### 候选

以 C9 为起点，运行：

```text
D0_C9_REPEAT
    C9 locked reference

D1_FIXED_READ_BETA_ONLY
    read_beta_frame_chunks -> global beta = 4.75
    其他保持 C9

D2_FIXED_TRI_GAMMA_003
    tri gamma chunk map -> global gamma = 0.003
    其他保持 C9

D3_FIXED_TRI_GAMMA_004
    tri gamma chunk map -> global gamma = 0.004
    其他保持 C9

D4_FIXED_TRI_GAMMA_005
    tri gamma chunk map -> global gamma = 0.005
    其他保持 C9

D5_FIXED_COMMIT_EMA_OFF
    commit_ema_chunks -> disabled / alpha = 1.0
    其他保持 C9

D6_FIXED_COMMIT_EMA_GLOBAL_A08
    commit EMA global alpha = 0.8
    no chunk list
    其他保持 C9

D7_C9_CLEAN_BEST_FIXED
    使用 D1-D6 中最优固定组合
    不含任何 absolute chunk-id policy
```

这里的固定 gamma 不根据 KITTI01 重新拟合；它们只来自 C9 原始范围 `0.003-0.005` 的机制审计。

### 记录指标

```text
full ATE / Rot / RPE_t / RPE_r / FinalErr
[200,300), [200,400), [400,600)
rolling50/100/200 mean/p90/worst
boundary10f/20f mean/p90/worst
Sim3 scale / Yaw RMSE
hmc rows
chunk-id policy audit: read map / tri map / EMA chunks 是否为空
TTT tri-replay actual pos/neutral/neg mass
post-zp delta norm by branch/layer
```

### 判断标准

```text
C9-Clean acceptable:
    best no-chunk-id ATE <= C9 + 0.30m

C9-Clean promising:
    best no-chunk-id ATE <= C9 + 0.10m

C9-Clean success:
    best no-chunk-id ATE <= C9
```

如果所有 fixed candidates 都比 C9 差 `>0.5m`，Codex 不允许继续扩大 fixed gamma sweep。它必须进入 Phase 2 attribution，解释是哪个 chunk-map interaction 导致退化。

---

## Phase 2：C9 组件贡献与交互归因

### 目标

回答用户最关心的问题：C9 的 ATE 下降到底来自 READ、TTT tri-replay、SWA、commit EMA、native mix 中的哪一部分。

### 已有 v43 结论

v43 已经完成第一轮 leave-one-out：

```text
remove TTT tri-replay: +2.4469m
remove tri gamma chunk map: +0.9710m
remove commit EMA: +0.4884m
remove read beta map: +0.0266m
remove SWA overlap replacement: +0.0563m
remove native mix: +0.0917m
full no-chunk-id flat: +1.5323m
```

本阶段不重复这些，而是补交互项。

### 候选

只补最关键的 8 条：

```text
I1_NO_TRI_REPLAY_NO_EMA
I2_NO_TRI_REPLAY_NO_SWA
I3_NO_TRI_REPLAY_NATIVE_MIX_OFF
I4_FIXED_TRI_GAMMA_BEST_NO_EMA
I5_FIXED_TRI_GAMMA_BEST_NO_SWA
I6_FIXED_TRI_GAMMA_BEST_NATIVE_MIX_OFF
I7_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST
I8_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST_FIXED_EMA_BEST
```

这里 `FIXED_TRI_GAMMA_BEST` 与 `FIXED_EMA_BEST` 来自 Phase 1，不使用 chunk id。

### 计算方式

对于每个组件 $i$：

$$
E_i = ATE(C9-i) - ATE(C9)
$$

对于组件对 $i,j$：

$$
I_{ij}=ATE(C9-i-j)-ATE(C9)-E_i-E_j
$$

如果 $|I_{ij}| > 0.3m$，说明交互不可忽略。

### 记录指标

```text
component_removed
ATE delta vs C9
segment delta vs C9
rolling100 delta vs C9
post-zp delta norm change
tri-replay role mass change
commit EMA applied mass
SWA source replacement summary
component interaction I_ij
```

### 判断标准

Phase 2 成功不是要求 ATE 降低，而是要求形成清晰贡献账本：

```text
必须能回答：
    TTT tri-replay 是否仍是最大贡献？
    tri gamma chunk map 的贡献是否可由 fixed gamma 替代？
    commit EMA 是独立贡献还是依赖 tri gamma？
    READ/SWA/native mix 是否真 near-neutral？
```

如果贡献账本显示 TTT tri-replay + gamma map 是唯一关键，而其他 near-neutral，则后续优化优先进入 Phase 4 adaptive tri replay。

---

## Phase 3：C23 support 增强实验

### 目标

验证 `past_only` 是否真是当前最优 support，还是 `full_chunk` / fixed offset support 能增强 C23 read cue。

### 为什么值得做

当前 C9 使用：

```text
acl2.gg.qq.low.g2_3.past_only.headmean.robustq
```

但 LoGeR 每个 chunk 内是 bidirectional inference。对于 chunk 内 frames，使用所有输入帧的 support 是合理的，不是外部未来泄漏。因此 `full_chunk` 需要在当前 C9/C9-Clean 条件下重新测试。

### 候选 cue

```text
S0_C23_PAST:
    acl2.gg.qq.low.g2_3.past_only.headmean.robustq

S1_C23_FULL_CHUNK:
    acl2.gg.qq.low.g2_3.full_chunk.headmean.robustq

S2_C23_FULL_CHUNK_NO_OVERLAP:
    acl2.gg.qq.low.g2_3.full_chunk_no_overlap.headmean.robustq

S3_C23_OFF246:
    acl2.gg.qq.low.g2_3.off246.headmean.robustq

S4_C23_NEAR12:
    acl2.gg.qq.low.g2_3.near12.headmean.robustq

S5_C23_PAST_PLUS_FUTURE_LIGHT:
    acl2.gg.qq.low.g2_3.past_plus_future_light_real.headmean.robustq
```

注意：不能用旧 `overlap_excluded` 名字声称 no-overlap，因为当前代码中它可能 fallback 到 full。

### 实验流程

1. 先在 short causal fork 上跑：

```text
parents = C9 and C9-Clean if available
chunks = 6, 10, 16
horizon = h10
mode = read-only / probe_native first
```

2. 只把 top 2 support 变体推进 full-online：

```text
C9 + best support
C9-Clean + best support
```

### 记录指标

```text
D_g distribution by frame
D_g mass > 0.5
support count per token/frame
support past/future ratio
overlap excluded count
ATE/Rot/RPE/FinalErr
rolling100/200 mean/p90/worst
[200,300), [400,600)
source attention mass to high-D tokens
```

### 判断标准

```text
Short gate:
    h10 ATE delta <= -1.0m
    or rolling100 best delta <= -3m
    or [200,300) delta <= -5m with [400,600) regression <= +1m

Full-online progress:
    ATE <= C9 - 0.3m
    or ATE <= C9-Clean - 0.3m
```

如果 `full_chunk` 只在 short rollout 好、full online 差，说明它可能引入 chunk-internal leakage / low-frequency over-suppression，不作为主线。

---

## Phase 4：training-free 自适应 tri-replay

### 目标

替代手工固定百分比：

```text
positive_frac = 0.35
negative_frac = 0.12
neutral_lambda = 0.85
```

用当前 chunk 的 TTT-native risk distribution 自动划分 positive / neutral / negative token groups。

### 自适应方法

所有方法都是 training-free，不使用 ATE label，不训练模型。

#### A1：1D k-means K=3

对 token-level risk $r_i$ 做一维 k-means：

```text
cluster with lowest risk  -> positive
middle cluster            -> neutral
highest risk              -> negative
```

角色约束：

```text
positive mass capped to [0.20, 0.60]
negative mass capped to [0.03, 0.25]
if cluster collapses, fallback to fixed C9 fractions
```

#### A2：Otsu histogram thresholds

对 $r_i$ 的 histogram 做 3-class Otsu thresholding，分成 low/mid/high risk。

#### A3：Robust MAD rule

计算：

$$
z_i = \frac{r_i - median(r)}{MAD(r)+\epsilon}
$$

分组：

```text
positive: z_i <= -0.5
negative: z_i >= 1.5
neutral: otherwise
```

#### A4：adaptive quantile with risk spread

如果当前 chunk risk spread 小，则减少 negative mass；如果 risk spread 大，则增加 negative mass。

$$
spread = Q_{90}(r)-Q_{10}(r)
$$

```text
if spread < 0.15: negative mass = 0.05
if 0.15 <= spread < 0.30: negative mass = 0.12
if spread >= 0.30: negative mass = 0.18
positive mass = 0.35 fixed or adjusted by low-risk cluster mass
```

### 候选

```text
A0_FIXED_C9_TRI_REPLAY
A1_KMEANS3_TRI_REPLAY
A2_OTSU3_TRI_REPLAY
A3_MAD_TRI_REPLAY
A4_ADAPTIVE_QUANTILE_TRI_REPLAY
```

### 运行协议

先在 C9-Clean 上跑，如果 C9-Clean 不 acceptable，则也在 C9 original 上跑 top 2 自适应候选。

### 记录指标

```text
risk distribution histogram per chunk
positive/neutral/negative actual mass per chunk
role mass by branch/layer
post-zp delta norm by branch/layer
update_conflict_energy mean/p90
ATE/Rot/RPE/FinalErr
[200,300), [400,600)
rolling100/200 p90/worst
fixed-vs-adaptive role maps
```

### 判断标准

```text
Adaptive success:
    full ATE <= fixed-tri baseline - 0.3m

Strong success:
    full ATE <= 33.0m

Target success:
    full ATE <= 30.0m
```

如果 adaptive tri improves short but not full，Codex 必须输出 per-chunk role mass 和 post-zp delta，判断是否因为某些 chunks role collapse 或 negative mass 过大。

---

## Phase 5：最小 semantic READ 复测

### 目标

当前 semantic READ 最好 full-online 是：

```text
SEM_READ_03_C23_RESID_READ_ONLY
ATE = 33.4875667508m
improvement vs C9 = -0.2754m
```

它未达最低进展门槛，但这是当前少数 full-online 语义正信号。本阶段只做最小复测，不再扩 semantic 大矩阵。

### 候选

```text
SEM1_C23_RESID_READ_ONLY_ON_C9
SEM2_C23_RESID_READ_ONLY_ON_C9_CLEAN
SEM3_C23_RESID_PLUS_BEST_SUPPORT
SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST
```

只有 Phase 3/4 产生 best support 或 adaptive tri 时才运行 SEM3/SEM4。

### 记录指标

```text
Stage C cache hit rate
semantic policy summary
READ affected source mass
attention mass before/after
ATE/Rot/RPE/FinalErr
[200,300), [400,600)
rolling100 p90/worst
context empty source events
```

### 判断标准

```text
Semantic minimum progress:
    ATE <= 33.3m
    or improvement vs matched baseline >= 0.5m

Semantic stage success:
    ATE <= 33.0m
```

如果仍然只改善 `~0.2-0.3m`，语义保留为 auxiliary read branch，不再作为主线。

---

## Phase 6：跨 sequence sanity，低优先级但必须规划

### 启动条件

仅当 KITTI01 满足以下任一条件时启动：

```text
ATE <= 33.0m
or improvement vs C9 >= 0.5m
or C9-Clean acceptable and contribution map clarified
```

### Sequence

```text
KITTI00
KITTI02
KITTI05
```

### 规则

```text
不调参。
不改变 thresholds。
不根据 sequence id 调整 cue / gamma / semantic rule。
只记录同一配置的结果和 failure mode。
```

### 记录指标

```text
full ATE/Rot/RPE/FinalErr per sequence
relative delta vs native / C9-style baseline if available
rolling-window p90/worst
Sim3 scale
failure windows
cue/memory health distribution
```

### 判断标准

```text
Sanity pass:
    no sequence catastrophic regression > +1m ATE
    at least one non-KITTI01 sequence improves or remains neutral

Sanity fail:
    KITTI01 gain comes with multi-sequence systematic regression
```

Cross-sequence 失败不允许导致 KITTI01 专用调参，只能用于诊断 failure mode。

---

## 7. Codex 并行执行安排

### Codex A：代码修复与 hard gate

负责：

```text
修复或禁用 TTT two_replay / separate_replay bug。
确认 full_chunk_no_overlap 使用正确 alias。
确认 C9 Stage C semantic disabled。
确认 P0 C9 exact repeat。
```

如果 P0 drift：先修 launcher / config diff，不跑实验。

### Codex B：Phase 1 去 chunk-id

负责：

```text
运行 D1-D7。
生成 c9_clean_report.md。
输出 chunk_id_policy_removed_summary.json。
```

如果 fixed candidates 全部退化 `>0.5m`：停止 fixed sweep，进入 Phase 2 attribution。

### Codex C：Phase 2 贡献归因

负责：

```text
补 I1-I8 interaction runs。
生成 component_interaction_matrix.csv。
生成 contribution_waterfall.png。
生成 interaction_heatmap.png。
```

如果发现某个交互项 $|I_{ij}|>0.3m$，在报告里标为 non-additive，不允许用单组件贡献解释。

### Codex D：Phase 3 C23 support

负责：

```text
运行 S0-S5 short causal fork。
选择 top 2 full-online。
输出 support_effect_dashboard.md。
```

如果所有 support 变体都不超过 past：锁定 C23 past，不再做 support sweep。

### Codex E：Phase 4 adaptive tri-replay

负责：

```text
实现 kmeans3 / otsu3 / MAD / adaptive quantile role assignment。
落盘 role mass 和 risk histograms。
运行 A0-A4。
```

如果自适应方法 role mass collapse：自动 fallback 到 fixed fraction，并记录 collapse，不允许把 collapse row 写成有效候选。

### Codex F：Phase 5 semantic minimal

负责：

```text
只在 Phase 3/4 有实际候选时运行 SEM3/SEM4。
否则只复测 SEM1/SEM2。
```

如果 semantic 仍低于 `0.5m` improvement：降级为 auxiliary branch，不再扩展。

---

## 8. 必须生成的报告和可视化

```text
v45_code_audit_update.md
v45_c9_clean_report.md
v45_component_contribution_ledger.csv
v45_component_interaction_matrix.csv
v45_c23_support_dashboard.md
v45_adaptive_trireplay_report.md
v45_semantic_read_minimal_report.md
v45_final_decision.md
```

可视化：

```text
component_ate_waterfall.png
component_interaction_heatmap.png
c9_clean_metric_bar.png
support_variant_ate_bar.png
support_variant_dg_distribution.png
adaptive_tri_risk_histograms.png
adaptive_tri_role_mass_by_chunk.png
segment_delta_bar_200_300_400_600.png
rolling100_p90_worst_comparison.png
```

---

## 9. 最终判断规则

本轮结束时必须回答：

```text
1. C9 去 chunk-id 后能否保持接近原性能？
2. C9 的主要贡献到底来自哪些组件？
3. past_only 是否真是当前 C23 最优 support？
4. adaptive tri-replay 是否能替代固定百分比？
5. semantic READ 是否有超过 0.5m 的 full-online价值？
6. 是否值得启动跨 sequence sanity？
```

最终状态分类：

```text
Target-30 success:
    任一 deployable full-online row ATE <= 30.0m。

Strong progress:
    ATE <= 32.0m。

Stage progress:
    ATE <= 33.0m。

Minimum progress:
    improvement vs C9 >= 0.5m 或 ATE <= 33.3m。

Mechanism progress:
    没有明显 ATE 进展，但 C9 贡献与 chunk-id 机制被拆清楚。

Failure:
    C9-clean 不成立，component attribution 不清，support/adaptive tri/semantic 都没有 full-online收益。
```

如果只得到 mechanism progress，也不是白做；它会明确告诉下一轮应该继续优化 TTT tri-replay，还是回到 READ/C23 cue，还是停止语义主线。

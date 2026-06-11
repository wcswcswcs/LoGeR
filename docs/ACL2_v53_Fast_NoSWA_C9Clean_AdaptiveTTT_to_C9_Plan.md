# ACL2 v53：Fast No-SWA C9-Clean Adaptive TTT 写入计划

日期：2026-06-09  
目标序列：KITTI Odometry Sequence 01  
当前 reference：`C9_P0_R2`，full-online ATE = `33.7629421029m`  
本轮第一目标：在 **不使用 absolute chunk-id 手工策略**、**不使用手工 tri-replay 百分比**、**不探索 SWA** 的条件下，让 clean / adaptive TTT 版本尽量接近 `C9_P0_R2`。  
本轮硬效率目标：任意一次 KITTI01 full run wall time 不允许超过 **28 分钟**。

---

## 0. 本轮必须守住的项目边界

本轮不是继续冲 `ATE < 30m`，也不是继续做语义大矩阵。当前最重要的问题是先把 C9 的核心机制变成一个更干净、更可解释、更可迁移的 **training-free adaptive TTT writing policy**。

本轮有三个硬约束：

```text
1. 不允许 absolute chunk-id policy。
   禁止 chunk5-9 / chunk10-12 / chunk16 这类手工指定配置。
   禁止 read_beta_frame_chunks、tri_gamma_chunk_map、tri_replay_chunk_params、commit_ema_chunks 以绝对 chunk id 生效。

2. TTT writing 的 tri replay 不得使用手工指定 positive / neutral / negative percentage。
   禁止 fixed positive_frac=0.35 / negative_frac=0.12 / neutral_lambda=0.85 作为 role assignment。
   允许固定公式、robust statistics、energy matching、state-conditioned rule。
   不允许 learned trigger / learned selector / learned router。

3. 本轮先不探索 SWA。
   SWA 不做变量，不做 ablation，不做语义控制。
   如果为了复现当前运行协议必须保留 SWA，则 SWA 固定为 C9/SWKS3 默认，不允许成为解释变量。
```

本轮还有一个新的硬约束：

```text
任意 KITTI01 full run wall time <= 28min。
如果某个 full run 超过 28min，该 row 不能作为有效候选推进，Codex 必须先进入 efficiency repair，而不是继续开新实验。
```

这条效率约束不是附加要求，而是实验能否推进的前置条件。v52 已经显示部分本地 full run 可以在约 24.5--26.6min 完成，但 chunk mean 约 37--40s，说明 full run 已经贴近效率边界。后续不能继续把 full run 当默认筛选工具。

---

## 1. 当前研究进展对本轮的启发

### 1.1 C9 的贡献结构已经比以前清楚

v46B 做了 clean no-chunk 的 8-row factorial attribution。它不是 exact C9，但它清楚说明，在干净组件空间里：

```text
READ / frame attention 单独贡献很大；
TTT 单独贡献也很大；
SWA 单独几乎没有 full ATE 贡献；
READ + TTT 有最重要的非加性交互；
READ + SWA 几乎没有额外收益。
```

因此，本轮可以合理地先不探索 SWA，把主线收敛到：

```text
fixed READ cue + adaptive TTT writing
```

这里的 fixed READ cue 指：继续沿用 C9 的 `C23 past` cue，即：

```text
acl2.gg.qq.low.g2_3.past_only.headmean.robustq
```

但它不能继续使用 chunk-wise read beta map。read beta 应使用固定全局值，优先采用之前最接近 C9 的 fixed beta。若已有 artifact 已证明 `fixed read beta` 只造成极小退化，则本轮不再重新大扫 read beta。

### 1.2 C9-Clean 直接固定化失败，说明不能只是把 chunk map 改成常数

v45 / v52 已经显示，简单去掉 chunk id 并改成固定值会明显退化。C9 的强项不是某一个参数本身，而是下面几件事的组合：

```text
TTT tri-replay
tri gamma timing / strength schedule
commit EMA / filtered commit behavior
READ + TTT interaction
```

这意味着：

```text
错误方向：
    chunk-specific gamma map -> 一个全局 gamma 常数

正确方向：
    chunk-specific gamma map -> state-conditioned gamma rule
```

也就是说，我们不应该用 chunk id 决定什么时候强写入，而应该用当前 chunk 的内部状态决定写入强度。

### 1.3 adaptive TTT 的失败不是“自适应完全没用”，而是当前自适应只学到了表层分组

v47-v50 和 v52 的结果说明：

```text
fused replay 失败得很明显；
split replay 明显优于 fused；
residual_x_dg risk 比纯 d_tok 稍好；
energy-matched / cluster3D 仍然没有接近 C9；
当前 adaptive 主要在做 role assignment，却没有学到 C9 的 post-zeropower update energy、gamma timing 和 commit behavior。
```

因此，本轮不再继续盲扫：

```text
role threshold
risk threshold
gamma clamp
positive/negative mass shape
```

本轮要换成：

```text
C9 teacher behavior autopsy
    -> state-conditioned adaptive split TTT
    -> strict runtime gate
    -> 少量 full run 验证
```

### 1.4 之前被验证失败但仍可能因代码/设计问题保留希望的方向

本轮只保留与目标 1 直接相关的可能性：

```text
1. fused adaptive TTT 失败，但 split adaptive TTT 有明显提升。
   因此 fused 不再作为主线，split 是必要条件。

2. d_tok / robust writer 失败，但 residual_x_dg 有小幅改善。
   说明 risk proxy 不能只看单一动态强度，仍需要更接近 conflict/update 的轻量 proxy。

3. full update_conflict_energy 太慢，不能作为 full run 默认路径。
   但 selected-layer / sampled-token 的 ConflictLite 仍可能保留希望。

4. past_plus_future_light_real support 曾有 alias bug。
   这需要修复和小复核，但不是本轮主线；C23 past 继续锁定。

5. 语义直接控制 TTT 写入多次失败。
   这不说明语义方向死了，但说明目标 2 必须等目标 1 建立 clean adaptive TTT baseline 后再重新接入。
```

---

## 2. 本轮总目标和成功标准

### 2.1 总目标

本轮要产出一个新的 baseline，暂称：

```text
C9Clean-AdaptiveTTT-v1
```

它必须满足：

```text
1. 使用 C9 的 frame-attention read cue family，但不使用 chunk-wise read beta map。
2. 不使用 C9 的 absolute chunk-id gamma / replay params / commit EMA chunks。
3. TTT tri replay 的 positive / neutral / negative role 不由手工 percentage 决定。
4. TTT gamma / neutral / commit behavior 由当前 chunk 的内部状态自适应决定。
5. 每次 KITTI01 full run wall time <= 28min。
6. 性能尽量接近 C9_P0_R2。
```

### 2.2 性能 gate

当前 C9 reference：

```text
C9_P0_R2 ATE = 33.7629421029m
```

本轮不要求直接超过 C9，但必须明确分级：

```text
Close-pass:
    ATE <= 34.30m
    即距离 C9 不超过约 0.54m。

Soft-pass:
    ATE <= 34.60m
    即说明 no-chunk adaptive TTT 已经进入 C9 附近，可以进入下一步语义目标。

Progress-pass:
    ATE <= 35.30m
    即明显优于 v50/v52 当前约 35.97m 平台，但还不能认为接近 C9。

Fail:
    ATE > 35.30m。
```

### 2.3 效率 gate

每条 full KITTI01 run 必须同时满足：

```text
wall_time_min <= 28.0
chunk_total_seconds_mean <= 42.0
probe_ttt_write_seconds_mean <= 8.0
hmc_rows = 38
frames = 1101
status = DONE
```

如果 `wall_time_min > 28.0`，该 candidate 不能继续进入更多 full run。Codex 必须先做 efficiency repair。

---

## 3. 实验假设

### 假设 H1：C9 的 chunk-id map 不是“数字本身重要”，而是它近似编码了 state-conditioned 写入节奏

C9 之所以比简单 dechunk 强，不是因为 chunk 5、10、16 这些编号本身有意义，而是因为这些 chunk 处在特定的 state：

```text
risk spread 变化；
D_g mass 变化；
post-zp delta energy 变化；
READ 和 TTT 交互状态变化；
commit candidate 与 native candidate 的差异变化。
```

因此，本轮要把 chunk-id map 替换成状态规则，而不是简单替换成常数。

### 假设 H2：当前 adaptive TTT 只做了 role split，没有匹配 C9 的 post-zp update energy

当前 adaptive writer 主要解决：

```text
哪些 token 是 positive / neutral / negative？
```

但 C9 还隐含解决：

```text
每个 chunk 写入多强？
post-zp 后 fast-weight delta 多大？
commit 后保留多少 candidate update？
什么时候需要 EMA-like filtering？
```

所以本轮 adaptive TTT v2 必须记录并匹配：

```text
post_zp_delta_norm
branch_delta_norm
candidate_vs_native_delta_cosine
commit_delta_norm
neutral continuity energy
negative correction energy
```

### 假设 H3：split replay 是必要条件，fused replay 不再作为主线

v50 已经证明 split replay 明显优于 fused replay。因此所有主候选都必须使用 split replay。fused 只可作为 diagnostic，不再占用 full run。

### 假设 H4：不解决效率，研究无法推进

完整 KITTI01 full run 只能作为最后确认，不能作为默认筛选。候选必须先通过：

```text
96F smoke
runtime projection
selected-window short diagnostic
full-run budget approval
```

---

## 4. Phase 0：代码与运行效率 hard gate

Phase 0 的目标是确保本轮不会再次出现“跑完了但配置不干净 / runtime 太慢 / 结果不可解释”。

### 4.1 配置 clean gate

每个候选 run 必须落盘：

```text
effective_config.yaml
chunk_id_policy_audit.json
adaptive_ttt_audit.json
runtime_profile.json
reproduce_command.sh
```

`chunk_id_policy_audit.json` 必须显示：

```text
read_beta_frame_chunks_empty = true
ttt_gradient_reversal_chunk_gammas_empty = true
ttt_tri_replay_chunk_params_empty = true
ttt_commit_ema_chunks_empty = true
native_mix_chunks_empty = true
semantic_action_active_chunks_empty = true
```

`adaptive_ttt_audit.json` 必须显示：

```text
manual_positive_frac = 0
manual_negative_frac = 0
manual_neutral_lambda = 0
role_mode contains adaptive / state_conditioned / no_percentage
adaptive_split_debug_count > 0
adaptive_fused_debug_count = 0
```

如果任何一项失败，候选无效，不允许跑 full。

### 4.2 C9 reference 使用规则

本轮原则上复用最近一次已通过 hard gate 的 C9 repeat。只有当下面任一条件成立，才重新跑 C9：

```text
1. 代码修改触及 C9 路径；
2. run_pipeline_abc_v2.py / HMC / TTT controller 影响 C9 的默认逻辑；
3. 最近 C9 repeat 缺少 runtime_profile；
4. effective_config_diff_vs_C9 不为 0。
```

如果需要重新跑 C9，它也必须满足 28min runtime gate。

### 4.3 效率设置 hard rule

所有 v53 full run 必须默认：

```text
Stage C = off
output_video = ""
save_frames = off
save heavy tensor trace = off
save visualization = off
empty_cuda_cache_each_chunk = 0
C23 cue fast path = on
full update_conflict_energy risk = forbidden
one process per GPU
no more than 4 full KITTI01 runs launched at the same time unless memory audit allows
```

### 4.4 如果 Phase 0 失败，Codex 先修什么

如果 runtime 超过 28min：

```text
1. 检查是否误开 Stage C / output_video / visualization。
2. 检查是否开启 full update_conflict_energy。
3. 检查是否保存 post-zp large tensor traces。
4. 检查 empty_cuda_cache_each_chunk 是否误开。
5. 检查 report JSONL 是否每层每token过度落盘。
6. 先跑 96F + 384F runtime profile，不允许直接再跑 full。
```

如果 no-chunk audit 失败：

```text
1. 检查 launcher env 是否仍继承 C9 chunk map。
2. 检查 run_attention_cue_experiment.sh 是否把空字符串误替换成默认 C9 map。
3. 检查 run_pipeline_abc_v2.py 默认值是否不是空。
4. 检查 HMC 内部是否有 fallback 到 historical C9 chunks。
```

---

## 5. Phase 1：C9 teacher behavior autopsy，不先跑新大矩阵

Phase 1 不是为了得新 ATE，而是为了回答：当前 adaptive TTT 和 C9 的行为到底差在哪里。

### 5.1 对比对象

必须比较三条轨迹的 per-chunk 行为：

```text
Teacher:
    C9_P0_R2 或最新 locked repeat。

Student baseline:
    v50 adaptive_writer_robust_split AW111。

Student v52:
    EnergyMatched AW111 或当前 v52 best adaptive split。
```

如果已有 landed artifacts 足够，优先复用；如果字段缺失，只允许补跑 selected-window trace，不允许直接补 full。

### 5.2 必须记录的 per-chunk 指标

对每个 chunk 记录：

```text
chunk_idx
frame_start / frame_end
D_g mean / p90 / high-mass
stage_d mean / p90
role_positive_mass
role_neutral_mass
role_negative_mass
adaptive_gamma_mean
adaptive_neutral_lambda_mean
post_zp_delta_norm_w0 / w1 / w2
candidate_minus_native_delta_norm
candidate_vs_native_delta_cosine
commit_delta_norm
commit_ema_effective_alpha
risk_spread
risk_entropy
negative_energy
neutral_energy
positive_energy
probe_ttt_write_seconds
chunk_total_seconds
```

### 5.3 必须生成的可视化

```text
teacher_vs_student_role_mass_timeline.png
teacher_vs_student_gamma_timeline.png
teacher_vs_student_post_zp_delta_norm.png
candidate_vs_native_cosine_timeline.png
risk_spread_vs_delta_norm_scatter.png
runtime_per_chunk_timeline.png
```

### 5.4 Phase 1 判断标准

Phase 1 成功不是看 ATE，而是必须给出清晰 gap diagnosis：

```text
1. adaptive 是否 role mass 合理但 post-zp delta energy 不对？
2. adaptive 是否 gamma 太弱 / 太强？
3. adaptive 是否 neutral lambda 过强导致 correction 被冲淡？
4. adaptive 是否没有 commit EMA-like filtering？
5. adaptive 的 runtime 是否主要被 split replay、state hash、debug serialization 或 tensor clone 拖慢？
```

如果 Phase 1 无法给出这些诊断，不能进入 Phase 2。

---

## 6. Phase 2：实现三个 state-conditioned adaptive TTT v2 候选

Phase 2 只允许 3 个主候选。每个候选必须满足：

```text
no chunk id
no manual tri replay percentage
split replay only
runtime projected full <= 28min
```

### 6.1 候选 A：SC-GammaSplit

目的：保留 v50 robust split role assignment，但把 gamma 从简单 risk_gap 公式改成 post-zp energy-aware rule。

Role assignment：

$$
s_i = \operatorname{norm}(stage\_d_i) \cdot (1-r_i)
$$

$$
d_i = r_i \cdot (1-\operatorname{norm}(stage\_d_i))
$$

$$
m_i = z(s_i) - z(d_i)
$$

其中 $r_i$ 是 fast risk proxy，例如 `ttt_residual_x_dg` 或 ConflictLite risk。三类角色用当前 chunk 内的 robust margin 决定：

```text
positive: m_i > median(m) + 0.5 * MAD(m)
negative: m_i < median(m) - 0.5 * MAD(m)
neutral: otherwise
```

这里没有使用 top percentage。若 positive 或 negative 完全塌缩，该 chunk 的 role 状态标记为 collapsed，不进行 fallback percentage 修补。

Gamma：

$$
\gamma_{eff}=\operatorname{clip}\left(
\rho \cdot \frac{E_{native}}{E_{neg}+\epsilon} \cdot \sqrt{\frac{\operatorname{Var}(r)}{\operatorname{Var}(stage\_d)+\epsilon}},
\gamma_{min},\gamma_{max}
\right)
$$

其中 $E_{native}$ 是 native/probe update 的轻量 delta norm proxy，$E_{neg}$ 是 negative replay branch 的 pre/post delta proxy。$\rho$、$\gamma_{min}$、$\gamma_{max}$ 是全局常数，不随 chunk id 变化。

### 6.2 候选 B：SC-GammaCommit

目的：在候选 A 基础上加入 state-conditioned commit filtering，替代 C9 的 `commit_ema_chunks=5,6`。

它不使用 chunk id，而是根据 candidate update 与 native update 的距离决定是否进行 filtered commit：

$$
q = \frac{\|\Delta W_{candidate}-\Delta W_{native}\|}{\|\Delta W_{native}\|+\epsilon}
$$

$$
c = 1 - \cos(\Delta W_{candidate}, \Delta W_{native})
$$

当 $q$ 或 $c$ 超过当前 run 的 robust threshold 时，使用：

$$
W_{commit}=\alpha W_{candidate}+(1-\alpha)W_{native}
$$

其中：

$$
\alpha=\operatorname{clip}\left(1 - \frac{q}{q+1}, \alpha_{min}, 1\right)
$$

它相当于 “adaptive EMA”，但不是 chunk-specific EMA。

### 6.3 候选 C：ConflictLite-Split

目的：验证以前失败的 update-conflict 思路是否只是因为 full `update_conflict_energy` 太慢，而不是 conflict signal 本身没用。

做法：

```text
只在 selected TTT layers / branch w0 上计算 sampled-token conflict proxy；
采样比例固定为轻量上限，例如最多 2048 tokens；
不回 CPU；
不保存大 tensor；
risk source = conflict_lite_x_dg。
```

要求：

```text
probe_ttt_write_seconds_mean <= 8.0
chunk_total_seconds_mean <= 42.0
```

如果 ConflictLite 超时，立即停止该候选，不允许用 full update_conflict_energy 替代。

---

## 7. Phase 3：效率优先筛选

### 7.1 96F smoke

每个候选先跑 96F smoke。

必须通过：

```text
status = DONE
no_chunk_policy_pass = true
adaptive_writer_audit_pass = true
split_debug_count > 0
manual percentage = 0
chunk_total_seconds_mean <= 42.0
probe_ttt_write_seconds_mean <= 8.0
no role collapse on all chunks
```

如果 96F smoke 失败，Codex 不能把候选送入 full。

### 7.2 384F / 512F runtime-projection screen

通过 96F 后，跑一个 384F 或 512F screen，用于估算 full runtime 和早期 catastrophic failure。

记录：

```text
ATE_short
Rot_short
chunk_total_seconds_mean
probe_ttt_write_seconds_mean
role_mass_mean
post_zp_delta_norm_mean
candidate_vs_native_cosine_mean
projected_full_wall_time_min
```

通过条件：

```text
projected_full_wall_time_min <= 28.0
short ATE 不得比 corresponding clean baseline 回退超过 1.5m
role collapse rows <= 20%
```

这一步只用于拒绝明显差的候选，不用于 claim 方法有效。

---

## 8. Phase 4：full KITTI01 最小验证

最多允许以下 full rows：

```text
P0_C9_REFERENCE_REPEAT_IF_NEEDED
V53_SC_A_GAMMASPLIT_FULL
V53_SC_B_GAMMACOMMIT_FULL
V53_SC_C_CONFLICTLITE_FULL
```

如果 C9 reference 已有最近有效 repeat，且配置 hash 与当前代码无冲突，可以不重跑。

每个 full row 必须记录：

```text
ATE_full
Rot_full
RPE_t / RPE_r
FinalErr
[200,300]
[400,600]
rolling50_mean / p90 / worst
rolling100_mean / p90 / worst
rolling200_mean / p90 / worst
hmc_rows
frames
wall_time_min
chunk_total_seconds_mean
probe_ttt_write_seconds_mean
role mass timeline
post-zp delta timeline
commit filter activation rate
no_chunk_policy_audit
manual_percentage_audit
```

### 8.1 full run 判断

每个 candidate 判定：

```text
Close-pass:
    ATE <= 34.30m
    wall_time <= 28min
    no_chunk_policy_pass = true
    manual_percentage_audit_pass = true

Soft-pass:
    ATE <= 34.60m
    wall_time <= 28min

Progress-pass:
    ATE <= 35.30m
    wall_time <= 28min

Fail:
    ATE > 35.30m
    or wall_time > 28min
    or no_chunk/manual-percentage audit fails
```

如果两个候选都 soft-pass，优先选择：

```text
1. 更低 ATE；
2. 更低 runtime；
3. 更少 downstream [400,600] 回退；
4. 更稳定 role mass；
5. 更少 commit filtering emergency。
```

---

## 9. Phase 5：失败分流与 Codex 自动处理

### 9.1 如果所有候选都慢于 28min

Codex 必须暂停算法试验，进入 efficiency repair：

```text
1. 把每 chunk timing 拆成：probe forward / controlled forward / build_probe_ttt_write_state / split replay / debug serialization / evaluation。
2. 禁用非必要 jsonl 大字段。
3. 禁用所有大 tensor trace。
4. 确认 output_video 为空。
5. 确认 Stage C off。
6. 检查是否多次 torch.cuda.empty_cache。
7. 检查 TTT state deepcopy / hash 是否每层重复。
8. 若 split replay 是瓶颈，尝试只对 selected layers / branch w0 做 split。
```

修完后只允许重新跑 96F + 384F runtime screen，不能直接重跑 full。

### 9.2 如果角色分组 collapse

如果 positive 或 negative role 几乎全空 / 全满：

```text
不要用 fixed top percentage 补。
```

Codex 应该尝试：

```text
1. 改 robust margin 的标准化方式；
2. 对 safety/danger 分别做 rank-normalization；
3. 增加 deadband 而不是固定 percentage；
4. 若所有 chunk 都 collapse，则该 risk proxy 无效。
```

### 9.3 如果 post-zp delta energy 和 C9 teacher 差异大

如果 adaptive candidate 的 role mass 看起来合理，但 post-zp delta norm / cosine 和 C9 teacher 差异大：

```text
优先修 gamma / commit rule，不要继续调 role split。
```

Codex 应该输出：

```text
teacher_student_delta_gap_report.md
post_zp_energy_mismatch_by_chunk.csv
```

### 9.4 如果 SC-A 失败但 SC-B 明显更好

说明问题主要是 commit filtering / EMA，而不是 role assignment。下一轮要围绕 state-conditioned commit，不能回到 role threshold sweep。

### 9.5 如果 SC-C 超时但短窗口很好

说明 conflict-like risk 有价值但实现太慢。Codex 应该：

```text
1. 减少 sampled tokens；
2. 减少 selected layers；
3. 只在 w0 branch 计算；
4. 使用 cached apply residual；
5. 重新测 96F runtime。
```

不允许直接上 full update_conflict_energy。

### 9.6 如果所有候选都 fail 且 ATE > 35.30m

这说明当前 action space 不足。下一步不应继续 adaptive split 小改，而应重新审计：

```text
1. C9 是否依赖 commit EMA more than expected；
2. C9 是否依赖 native_mix / branch interaction；
3. 是否需要从 token role 转为 branch/layer-level action；
4. 是否需要显式 trajectory-state / scale-state controller。
```

---

## 10. 目标 2：语义真实改进几何重建的延后策略

本轮不把语义纳入主 run。只有当目标 1 达到 soft-pass，即：

```text
clean adaptive TTT ATE <= 34.60m
```

才允许进入目标 2。

目标 2 的初始方向不是 all-memory semantic prior，而是：

```text
semantic-conditioned C23 / READ residual
high-influence anomaly READ filtering
static structure rescue
```

不允许直接把语义接到 TTT adaptive writer。原因是：当前语义直接写 TTT 多次失败，且目标 1 本身还未稳定。

目标 2 启动时必须只做最小复测：

```text
AdaptiveTTT baseline + semantic C23 residual READ
AdaptiveTTT baseline + high-influence anomaly READ filtering
AdaptiveTTT baseline + semantic residual + static rescue
```

成功标准：

```text
semantic over clean adaptive baseline improvement >= 0.3m
no downstream [400,600] regression > +1m
no full runtime > 28min
```

如果这些不成立，语义仍保留为诊断工具，不进入 TTT writer。

---

## 11. 本轮最终交付物

Codex 必须交付：

```text
v53_phase0_efficiency_audit.md
v53_c9_teacher_student_autopsy.md
teacher_student_role_mass_timeline.png
teacher_student_gamma_timeline.png
teacher_student_post_zp_delta_norm.png
risk_spread_vs_delta_norm_scatter.png
v53_candidate_registry.csv
v53_candidate_registry.json
v53_full_metrics_summary.md
v53_runtime_profile_summary.csv
v53_runtime_profile_by_chunk.png
v53_no_chunk_policy_audit.json
v53_manual_percentage_audit.json
v53_failure_routing_report.md
```

其中 `v53_final_report.md` 必须明确回答：

```text
1. 是否产生了 no-chunk / no-manual-percentage adaptive TTT candidate？
2. 它离 C9_P0_R2 还差多少 ATE？
3. 它是否满足 28min runtime gate？
4. 它的 post-zp energy / gamma / commit behavior 和 C9 的主要差异是什么？
5. 下一步应该修 role、修 gamma、修 commit，还是放弃当前 TTT action space？
```

---

## 12. 本轮停止规则

本轮不允许无限探索。停止规则如下：

```text
1. 三个主候选全部 fail 且 ATE > 35.30m：停止算法尝试，做 failure report。
2. 任意两个 full run 超过 28min：停止 full run，做 efficiency repair。
3. ConflictLite 超时：停止 conflict line，不允许上 full update_conflict_energy。
4. role collapse 连续发生：停止该 risk proxy。
5. 若某候选达到 ATE <= 34.60m 且 runtime <= 28min：冻结为 C9Clean-AdaptiveTTT-v1，进入目标 2 前置评审。
```

---

## 13. 一句话总结

本轮 v53 的目标不是再跑一个大矩阵，而是把 C9 的手工 chunk 配方替换成一个 **无 chunk-id、无手工 tri percentage、可在 28 分钟内完成 KITTI01 full run 的 adaptive TTT writer**。如果这一步做不到，后续语义实验会继续叠在一个不干净、不稳定的 TTT baseline 上，无法形成可靠结论。

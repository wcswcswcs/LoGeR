## 2026-05-05 ACL2 v4：Early-Mid Window 因果确认启动

这轮按 `ACL2_v4_EarlyMidWindow_CausalValidation_and_Combination_Readiness_Plan.md` 执行。v4 的重点不是立刻做复杂 cue combination，而是先确认 `g2_3.full` / `g2_4.full` 相比 `g3.full` 的优势是否稳定，并把 ATE-oriented 与 balanced/Pareto 主线分清楚。

加速/资源记录：

- H1 首批 4 并发：GPU 0-3，`07:22:38 -> 07:50:00`，约 `27.4 min` 完成 4 个 full run。
- 中途尝试额外打开 GPU 4-7 补 H2，但 8 并发把系统内存压到只剩 `~2.5GB available`，因此主动停止了后一批未完成 run；这些 partial 目录不计入结果。
- 后续固定回到 4 并发，避免 swap / OOM；4 并发是当前 full-run 安全甜点位。
- H3 soft-g4 补跑 4 并发：GPU 0-3，`22:44:43 -> 23:11:17`，约 `26.6 min` 完成 4 个 full run；峰值内存约 `255GiB used`，继续说明 8 并发不是安全加速方式。

### H1：`C3/C23/C24` repeat 稳定性

固定协议：

- mode: hybrid full KITTI01
- commit: `probe_ttt_write`
- write score: `stage_d`
- read path: frame attention early layers
- bias: pair

候选定义：

- `C3 = acl2.gg.qq.low.g3.full.headmean.robustq`, beta `3.75`
- `C23 = acl2.gg.qq.low.g2_3.full.headmean.robustq`, beta `3.75`
- `C24 = acl2.gg.qq.low.g2_4.full.headmean.robustq`, beta `4.00`

| Run | Cue | Beta | ATE RMSE | Rot RMSE | RPE t | RPE r | SHA/复现结论 |
|---|---|---:|---:|---:|---:|---:|---|
| C3 original | `g3.full` | 3.75 | `38.4298` | `8.9846` | `92.3905` | `0.0090` | 历史 anchor |
| ACL2V4_H1_01 | `g3.full` | 3.75 | `38.4694` | `8.9902` | `92.3909` | `0.0090` | 未 byte-match；比原始 C3 差 `0.0396m` |
| C23 original | `g2_3.full` | 3.75 | `38.3847` | `8.7583` | `92.3928` | `0.0087` | 历史 Pareto |
| ACL2V4_H1_02 | `g2_3.full` | 3.75 | `38.3847` | `8.7583` | `92.3928` | `0.0087` | byte-level exact repeat |
| C24 original | `g2_4.full` | 4.00 | `38.3805` | `8.8707` | `92.3942` | `0.0089` | 历史 ATE best |
| ACL2V4_H1_03 | `g2_4.full` | 4.00 | `38.3805` | `8.8707` | `92.3942` | `0.0089` | byte-level exact repeat |
| ACL2V4_H1_04 | `g2_4.full` | 4.00 | `38.3805` | `8.8707` | `92.3942` | `0.0089` | byte-level exact repeat |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| C3 original | `38.4298` | `4.692` | `33.353` | `33.860` | `33.808` | `5.477` | `30.704494` |
| C3 H1 repeat | `38.4694` | `4.777` | `33.385` | `33.893` | `33.840` | `5.481` | `30.706068` |
| C23 original | `38.3847` | `3.677` | `33.279` | `33.789` | `33.746` | `5.291` | `30.720405` |
| C23 H1 repeat | `38.3847` | `3.677` | `33.279` | `33.789` | `33.746` | `5.291` | `30.720405` |
| C24 original | `38.3805` | `4.503` | `33.243` | `33.758` | `33.707` | `5.401` | `30.731105` |
| C24 H1 repeat1 | `38.3805` | `4.503` | `33.243` | `33.758` | `33.707` | `5.401` | `30.731105` |
| C24 H1 repeat2 | `38.3805` | `4.503` | `33.243` | `33.758` | `33.707` | `5.401` | `30.731105` |

H1 结论：

1. `C23` 和 `C24` 在当前代码下是 byte-level 稳定复现；`C24` 两次 repeat 完全一致，`C23` repeat 也与 v3 原始结果完全一致。
2. `C3` 当前 repeat 没有 byte-match 历史 `ACL2_B6_10`，ATE 差 `0.0396m`；但 `C23/C24` 仍稳定优于当前/历史 C3，因此不改变主线判断。
3. `C24` 仍是 ATE-oriented best：`38.3805 / 8.8707`。
4. `C23` 是更强 balanced/Pareto 候选：ATE 只比 `C24` 差 `0.0042m`，但 Rot 好 `0.1124deg`、FinalErr 好 `0.826m`、YawRMSE 好 `0.110`。
5. H1 后的主线判断：先把 `D_g_balanced = C23`，`D_g_ate = C24`，`C3` 只保留为 historical anchor；下一步补 `C23` hybrid beta 曲线和同 beta read-only/write-gain。

### H2 第一批：`C23` hybrid beta 曲线

固定 cue：

`acl2.gg.qq.low.g2_3.full.headmean.robustq`

固定协议：

- mode: hybrid full KITTI01
- commit: `probe_ttt_write`
- write score: `stage_d`
- read path: frame attention early layers
- bias: pair

| Run | Beta | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---|
| ACL2V4_H2R_01 | 3.25 | `38.4284` | `8.7310` | `92.3925` | `0.0087` | rotation 最好，但 ATE 回退 |
| ACL2V4_H2R_02 | 3.50 | `38.4275` | `8.7474` | `92.3928` | `0.0087` | ATE 不如 3.75 |
| ACL2V3_S2_12 / H1 repeat | 3.75 | `38.3847` | `8.7583` | `92.3928` | `0.0087` | **C23 ATE/segment best** |
| ACL2V4_H2R_03 | 4.00 | `38.4044` | `8.7619` | `92.3926` | `0.0087` | ATE 回退，但 endpoint 好 |
| ACL2V4_H2R_04 | 4.25 | `38.3984` | `8.7635` | `92.3929` | `0.0087` | final error 最好，但 ATE 不如 3.75 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| C24 b4.00 | `38.3805` | `4.503` | `33.243` | `33.758` | `33.707` | `5.401` | `30.731105` |
| C23 b3.25 | `38.4284` | `3.591` | `33.306` | `33.818` | `33.774` | `5.265` | `30.723349` |
| C23 b3.50 | `38.4275` | `3.620` | `33.317` | `33.828` | `33.785` | `5.278` | `30.723095` |
| C23 b3.75 | `38.3847` | `3.677` | `33.279` | `33.789` | `33.746` | `5.291` | `30.720405` |
| C23 b4.00 | `38.4044` | `3.506` | `33.311` | `33.820` | `33.781` | `5.291` | `30.718784` |
| C23 b4.25 | `38.3984` | `3.488` | `33.308` | `33.816` | `33.782` | `5.292` | `30.722028` |

H2 第一批结论：

1. `C23` hybrid ATE 最优仍是 beta `3.75`，没有被 `4.00/4.25` 超过。
2. beta `3.25/3.50` 的 rotation 更好，其中 `3.25` Rot=`8.7310`、Yaw=`5.265`，但 ATE 比 `3.75` 差约 `0.0437m`。
3. beta `4.25` final error 最好，`3.488m`，比 `3.75` 好 `0.189m`，但 ATE 差 `0.0137m`、Rot 略差。
4. 当前 `C23` 主候选仍锁 beta `3.75`；`b4.25` 可作为 endpoint diagnostic，`b3.25` 可作为 rotation/yaw diagnostic。
5. 下一步补 `C23` 同 beta read-only，计算 write gain 曲线。如果 safe write 在 `C23` 上仍稳定为正，`probe_ttt_write` 可继续作为主协议。

### H2 第二批：`C23` read-only / hybrid 同 beta write gain

固定 cue：

`acl2.gg.qq.low.g2_3.full.headmean.robustq`

| Beta | Read-only ATE / Rot | Hybrid ATE / Rot | Write gain ATE | Hybrid-Rot minus Read-Rot | 结论 |
|---:|---:|---:|---:|---:|---|
| 3.25 | `38.7077 / 8.7536` | `38.4284 / 8.7310` | `0.2793` | `-0.0226` | safe write 同时改善 ATE/Rot |
| 3.50 | `38.6614 / 8.7446` | `38.4275 / 8.7474` | `0.2339` | `+0.0028` | safe write 稳定有效 |
| 3.75 | `38.6319 / 8.7472` | `38.3847 / 8.7583` | `0.2472` | `+0.0111` | **C23 当前主点** |
| 4.00 | `38.6609 / 8.7576` | `38.4044 / 8.7619` | `0.2565` | `+0.0043` | safe write 稳定有效 |
| 4.25 | `38.6424 / 8.7767` | `38.3984 / 8.7635` | `0.2440` | `-0.0132` | endpoint diagnostic，ATE 不如 3.75 |

H2 第二批结论：

1. `C23` 的 `probe_ttt_write` gain 在所有 beta 上稳定为正，ATE gain 为 `0.2339-0.2793m`。
2. 这证明 `C23` 不是单纯 read-only trick；它和 safe branch0 write 仍然互补，可以继承 Pipeline v2 的主 commit 协议。
3. read-only 最好是 beta `3.75` 的 `38.6319 / 8.7472`；hybrid 最好仍是 beta `3.75` 的 `38.3847 / 8.7583`。
4. beta `3.25` 的 hybrid rotation/yaw 更干净，但 ATE 回退明显；beta `4.25` 的 final error 更好，但 ATE 仍不如 `3.75`。
5. `D_g_balanced` 先锁为 `C23 b3.75`；`D_g_ate` 保持 `C24 b4.00`。下一步应按 v4 plan 做 `g4` 贡献拆解和 `g2_3/g2_4` support sweep，而不是继续细扫 C23 beta。

### H3 第一批：`g4` contribution / layer-window read-only 拆解

固定协议：

- mode: read-only full KITTI01
- commit: `probe_native`
- beta: `3.75`
- read path: frame attention early layers
- bias: pair

本批用 4 并发运行，`08:47:10 -> 09:11:19`，约 `24.1 min` 完成 4 个 full run。

| Run | Cue | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| ACL2V3_S2_08 | `acl2.gg.qq.low.g2_3.full.headmean.robustq` | `38.6319` | `8.7472` | `92.3943` | `0.0087` | C23 read-only reference，当前 balanced base |
| ACL2V3_S2_04 | `acl2.gg.qq.low.g2_4.full.headmean.robustq` | `38.6137` | `8.8417` | `92.3954` | `0.0088` | C24 read-only reference，当前 ATE-oriented base |
| ACL2V3_S2_07 | `acl2.gg.qq.low.g3_4.full.headmean.robustq` | `38.6981` | `9.0334` | `92.3948` | `0.0091` | 加入 g4 后 rotation/endpoint 变差 |
| ACL2V4_H3_03 | `acl2.gg.qq.low.g1_3.full.headmean.robustq` | `39.1540` | `9.0104` | `92.3973` | `0.0089` | 有信号，但明显弱于 C23/C24 |
| ACL2V4_H3_04 | `acl2.gg.qq.low.g2_5.full.headmean.robustq` | `39.0393` | `9.2193` | `92.3953` | `0.0091` | 比 g1_3 好，但仍明显弱于 C23/C24 |
| ACL2V4_H3_02 | `acl2.gg.qq.low.g4.full.headmean.robustq` | `39.4285` | `9.5862` | `92.3968` | `0.0095` | g4 单层 ATE 有弱信号，但 orientation 代价很大 |
| ACL2V4_H3_01 | `acl2.gg.qq.low.g2.full.headmean.robustq` | `39.7765` | `8.7929` | `92.4021` | `0.0086` | rotation 尚可，ATE 太弱 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| C23 read b3.75 | `38.6319` | `3.273` | `33.543` | `34.051` | `33.986` | `5.258` | `30.733672` |
| C24 read b3.75 | `38.6137` | `4.094` | `33.486` | `34.002` | `33.924` | `5.353` | `30.743752` |
| g3_4 read b3.75 | `38.6981` | `5.117` | `33.580` | `34.095` | `33.999` | `5.525` | `30.743910` |
| g2 read b3.75 | `39.7765` | `2.370` | `34.631` | `35.153` | `35.091` | `5.289` | `30.818644` |
| g4 read b3.75 | `39.4285` | `6.606` | `34.326` | `34.845` | `34.713` | `5.958` | `30.761201` |
| g1_3 read b3.75 | `39.1540` | `3.983` | `34.117` | `34.629` | `34.536` | `5.451` | `30.767676` |
| g2_5 read b3.75 | `39.0393` | `4.001` | `34.066` | `34.573` | `34.503` | `5.666` | `30.743033` |

H3 第一批结论：

1. `g4.full` 单层不是可用主 cue：ATE `39.4285` 虽然比 no-control / old key-cosine 好，但 Rot=`9.5862`、FinalErr=`6.606`、Yaw=`5.958` 都明显差。
2. `g3_4.full` 的 ATE 接近 C23/C24，但 Rot/FinalErr/Yaw 全部恶化，说明 `g4` 更像 trade-off / contamination layer，而不是越多越好。
3. `g2_5.full` 和 `g1_3.full` 都没有超过 C23/C24；扩大 layer window 会稀释早中层有效信号。
4. `g2.full` 的 endpoint 很好，但 ATE 太弱，说明单层 g2 不足以支撑主 read correction。
5. 当前不建议马上做大规模 soft-g4 组合；如果要做，也只应做极小 λ 的 diagnostic。更高优先级是 H4：对 `C23/C24` 做 support sweep，确认 `full` 是否真是最优，以及是否存在更 causal / endpoint-safe 的 support。
6. v4 当前锁定：`D_g_balanced = g2_3.full b3.75`，`D_g_ate = g2_4.full b4.00`，`g4` 暂时只保留为解释 C24/C23 trade-off 的 diagnostic source。

### H3 第二批：soft-g4 map-level interpolation

固定协议：

- mode: read-only full KITTI01
- commit: `probe_native`
- beta: `3.75`
- read path: frame attention early layers
- bias: pair
- definition: `D = clip((1-lambda) * D23 + lambda * D4, 0, 1)`
- 结果目录：`results/kitti01_hmc_v2/acl2_v4_earlymid_validation/h3_soft_g4/`
- 汇总 CSV：`weighted_g4_metrics.csv`

| Run | Lambda | Cue | ATE RMSE | Rot RMSE | RPE t | RPE r | Mean D>0.5 | Anchor collide | Frag | Corr old_dyn | Gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ACL2V4_H3SG_01 | 0.10 | `mix.c23_g4_soft_l010` | `38.7295` | `8.8799` | `92.3943` | `0.0088` | 0.191 | 0.111 | 0.097 | -0.008 | fail |
| ACL2V4_H3SG_02 | 0.25 | `mix.c23_g4_soft_l025` | `38.8379` | `9.0426` | `92.3950` | `0.0090` | 0.182 | 0.112 | 0.096 | 0.004 | fail |
| ACL2V4_H3SG_03 | 0.50 | `mix.c23_g4_soft_l050` | `38.9879` | `9.2406` | `92.3955` | `0.0092` | 0.167 | 0.113 | 0.116 | 0.026 | fail |
| ACL2V4_H3SG_04 | 0.75 | `mix.c23_g4_soft_l075` | `39.2026` | `9.4617` | `92.3960` | `0.0094` | 0.179 | 0.114 | 0.156 | 0.048 | fail |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| C23 full read b3.75 | `38.6319` | `3.273` | `33.543` | `34.051` | `33.986` | `5.258` | `30.733672` |
| C24 full read b3.75 | `38.6137` | `4.094` | `33.486` | `34.002` | `33.924` | `5.353` | `30.743752` |
| g4 full read b3.75 | `39.4285` | `6.606` | `34.326` | `34.845` | `34.713` | `5.958` | `30.761201` |
| soft-g4 lambda 0.10 | `38.7295` | `3.541` | `33.659` | `34.169` | `34.097` | `5.367` | `30.733985` |
| soft-g4 lambda 0.25 | `38.8379` | `4.263` | `33.766` | `34.280` | `34.195` | `5.520` | `30.736729` |
| soft-g4 lambda 0.50 | `38.9879` | `5.240` | `33.898` | `34.414` | `34.308` | `5.688` | `30.742390` |
| soft-g4 lambda 0.75 | `39.2026` | `6.138` | `34.108` | `34.627` | `34.505` | `5.863` | `30.751413` |

H3 第二批结论：

1. soft-g4 没有找到比 C23/C24 更好的 weighted window；最小 lambda `0.10` 也比 C23 full read 回退 `0.0976m`，超过 H3 hybrid promotion 的 `+0.05m` gate。
2. 随着 lambda 从 `0.10` 增到 `0.75`，ATE、Rot、FinalErr、Yaw 基本单调变差，确认 `g4` 更像 contamination/trade-off source，而不是可软融合的正贡献层。
3. 不启动 H3 soft-g4 hybrid；后续组合 base 不使用 `g2_3 + soft-g4`，而使用 H4/H8 后锁定的 `g2_3.past_only`。
4. `weighted_g4_metrics.csv` 已保存；D23/D4 raw map overlap 没有补做，因为 soft-g4 未过 read-only gate，继续补 component-map logging 不影响主线决策。

### H4：`C23/C24` support sweep 与 hybrid-safe 验证

固定协议：

- read-only: `probe_native`
- hybrid-safe: `probe_ttt_write`
- read path: frame attention early layers
- bias: pair
- `C23 = acl2.gg.qq.low.g2_3.*.headmean.robustq`, beta `3.75`
- `C24 = acl2.gg.qq.low.g2_4.*.headmean.robustq`, beta `4.00`

运行时间记录：

- read-only batch1：`12:17:14 -> 12:41:32`，4 runs，约 `24.3 min`
- read-only batch2：`12:41:58 -> 13:06:28`，4 runs，约 `24.5 min`
- read-only batch3：`13:06:59 -> 13:31:58`，4 runs，约 `25.0 min`
- hybrid gate batch：`13:32:43 -> 13:58:39`，4 runs，约 `26.0 min`
- past-only repeat batch：`13:59:24 -> 14:24:52`，2 runs，约 `25.5 min`

#### H4 read-only support sweep

| Candidate | Support | Beta | ATE RMSE | Rot RMSE | RPE t | RPE r | Mean D>0.5 | Frag | Anchor | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C23 | full | 3.75 | `38.6319` | `8.7472` | `92.3943` | `0.0087` | 0.197 | 0.099 | 0.111 | reference |
| C23 | off246 | 3.75 | `38.6703` | `8.7539` | `92.3938` | `0.0087` | 0.197 | 0.100 | 0.111 | ATE 回退 |
| C23 | near12 | 3.75 | `38.6177` | `8.7119` | `92.3942` | `0.0086` | 0.196 | 0.100 | 0.111 | read-only 同时改善 ATE/Rot |
| C23 | near24 | 3.75 | `38.6335` | `8.7384` | `92.3940` | `0.0086` | 0.196 | 0.100 | 0.111 | 接近 full，Rot 略好 |
| C23 | past_only | 3.75 | `38.6421` | `8.6795` | `92.3944` | `0.0086` | 0.190 | 0.097 | 0.112 | rotation 明显更好，ATE 小回退 |
| C23 | future_only | 3.75 | `38.6907` | `8.4343` | `92.3941` | `0.0082` | 0.191 | 0.096 | 0.110 | rotation 最强，但 ATE 回退 |
| C23 | overlap_excluded | 3.75 | `38.6319` | `8.7472` | `92.3943` | `0.0087` | 0.197 | 0.099 | 0.111 | 与 full 等价 |
| C24 | full | 4.00 | `38.6157` | `8.8614` | `92.3957` | `0.0088` | 0.201 | 0.091 | 0.107 | reference |
| C24 | off246 | 4.00 | `38.6188` | `8.8814` | `92.3953` | `0.0088` | 0.201 | 0.092 | 0.107 | 接近 full，但不超过 |
| C24 | near12 | 4.00 | `38.6438` | `8.8798` | `92.3948` | `0.0088` | 0.201 | 0.092 | 0.107 | 回退 |
| C24 | near24 | 4.00 | `38.6179` | `8.8562` | `92.3948` | `0.0088` | 0.201 | 0.092 | 0.107 | 接近 full |
| C24 | past_only | 4.00 | `38.5767` | `8.7579` | `92.3961` | `0.0088` | 0.195 | 0.089 | 0.109 | read-only 新 best |
| C24 | future_only | 4.00 | `38.6138` | `8.5124` | `92.3945` | `0.0083` | 0.194 | 0.088 | 0.107 | ATE 接近 full，Rot 大幅改善 |
| C24 | overlap_excluded | 4.00 | `38.6157` | `8.8614` | `92.3957` | `0.0088` | 0.201 | 0.091 | 0.107 | 与 full 等价 |

Read-only gate 结论：

1. `full support` 不是 C23/C24 上的绝对最优。`C23 near12` read-only 同时改善 ATE/Rot；`C24 past_only` read-only 明显超过 C24 full。
2. `past_only` 很有价值：C23 上主要改善 rotation，C24 上同时改善 ATE 和 rotation。
3. `future_only` 带来很强 rotation 改善，尤其 C24 Rot 从 `8.8614` 到 `8.5124`，但它依赖 chunk 内未来帧，且 ATE 没有同步成为 best，因此只保留为 diagnostic。
4. `overlap_excluded` 当前与 full 完全一致，说明实现里的 overlap exclusion 在这个 support path 上没有产生有效差异，需要后续单独查 support mask。

#### H4 hybrid-safe gate

只让 read-only 通过 gate 的少量 support 进入 hybrid：

| Run | Candidate | Support | Beta | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| C23 full reference | C23 | full | 3.75 | `38.3847` | `8.7583` | `92.3928` | `0.0087` | old balanced best |
| ACL2V4_H4_13 | C23 | near12 | 3.75 | `38.4022` | `8.7410` | `92.3929` | `0.0087` | Rot 略好，但 ATE 回退 |
| ACL2V4_H4_14 | C23 | past_only | 3.75 | `38.3706` | `8.6694` | `92.3930` | `0.0086` | **new balanced best** |
| C24 full reference | C24 | full | 4.00 | `38.3805` | `8.8707` | `92.3942` | `0.0089` | old ATE best |
| ACL2V4_H4_15 | C24 | past_only | 4.00 | `38.3566` | `8.7660` | `92.3942` | `0.0088` | **new ATE best** |
| ACL2V4_H4_16 | C24 | future_only | 4.00 | `38.4096` | `8.5130` | `92.3937` | `0.0084` | rotation diagnostic，ATE 回退 |

Past-only repeat：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | SHA 结论 |
|---|---:|---:|---:|---:|---|
| C23 past original | `38.3706` | `8.6694` | `92.3930` | `0.0086` | reference |
| C23 past repeat | `38.3706` | `8.6694` | `92.3930` | `0.0086` | byte-level exact |
| C24 past original | `38.3566` | `8.7660` | `92.3942` | `0.0088` | reference |
| C24 past repeat | `38.3566` | `8.7660` | `92.3942` | `0.0088` | byte-level exact |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| C23 full hybrid b3.75 | `38.3847` | `3.677` | `33.279` | `33.789` | `33.746` | `5.291` | `30.720405` |
| C23 past hybrid b3.75 | `38.3706` | `3.403` | `33.259` | `33.769` | `33.735` | `5.227` | `30.725774` |
| C23 near12 hybrid b3.75 | `38.4022` | `3.512` | `33.301` | `33.811` | `33.765` | `5.269` | `30.717675` |
| C24 full hybrid b4.00 | `38.3805` | `4.503` | `33.243` | `33.758` | `33.707` | `5.401` | `30.731105` |
| C24 past hybrid b4.00 | `38.3566` | `4.244` | `33.207` | `33.723` | `33.675` | `5.325` | `30.735921` |
| C24 future hybrid b4.00 | `38.4096` | `2.870` | `33.322` | `33.828` | `33.783` | `5.063` | `30.727763` |

H4 结论：

1. H4 找到新的稳定 best：`C24 past_only + probe_ttt_write`，ATE/Rot=`38.3566 / 8.7660`，且 repeat byte-level exact。
2. 新 balanced/Pareto 候选是 `C23 past_only + probe_ttt_write`，ATE/Rot=`38.3706 / 8.6694`，比 C23 full 同时改善 ATE、Rot、FinalErr、Yaw 和 segment mean，并且 repeat exact。
3. `past_only` 支持 `full` 不是必须条件，且说明当前有效信号主要可由历史/当前 support 提供；这比 `future_only` 更适合后续 causal / streaming 版本。
4. `future_only` 明显改善 rotation 和 endpoint，但 ATE 回退；它说明未来帧 support 有强 orientation signal，但不能作为主线，因为非因果且主指标不够。
5. 现有 quality 指标仍不能直接 ranking：`D>0.5`、Frag、Anchor 在 full/near/past 间变化很小，但指标差异显著。因此 H5 quality recalibration 仍必要。
6. v4 当前锁定更新为：`D_g_ate = C24 past_only b4.00`，`D_g_balanced = C23 past_only b3.75`；原 C24/C23 full 降级为 support baseline。

### H5：cue quality 重标定 first pass

固定对 H4 的 C23/C24 full / past / near / future 候选做 chunk-level quality-error join，未再额外开 GPU。输出目录：

`results/kitti01_hmc_v2/acl2_v4_earlymid_validation/h5_quality_recalibration_h4_support/`

关键 correlation：

| Subset | Metric vs delta RMSE | Corr | n | 解释 |
|---|---|---:|---:|---|
| past candidates | `prior_anchor_collision` vs delta vs C23 | `-0.448` | 76 | AnchorCollide 不能作 hard reject，高 overlap 并不必然更差 |
| past candidates | `prior_old_dyn_coverage` vs delta vs C23 | `0.399` | 76 | old_dyn overlap 越高反而越容易 regression |
| past candidates | `prior_old_dyn_iou` vs delta vs C23 | `0.365` | 76 | 与 old_dyn 太像未必好 |
| past candidates | `prior_corr_D_unc` vs delta vs C23 | `0.327` | 76 | uncertainty 相关区域可能带来风险 |
| past candidates | `prior_corr_D_old_dyn` vs delta vs C23 | `0.302` | 76 | 旧 dyn 相关性有 diagnostic value |

Run-level quality / error summary：

| Run | Mean delta vs C23 | Mean delta vs C24 | D>0.5 | Frag | Anchor | OldIoU |
|---|---:|---:|---:|---:|---:|---:|
| C23 full hybrid b3.75 | `0.000` | `0.037` | 0.197 | 0.099 | 0.111 | 0.144 |
| C23 past hybrid b3.75 | `-0.023` | `0.014` | 0.190 | 0.097 | 0.112 | 0.139 |
| C23 near12 hybrid b3.75 | `0.021` | `0.059` | 0.196 | 0.100 | 0.111 | 0.144 |
| C24 full hybrid b4.00 | `-0.037` | `0.000` | 0.201 | 0.091 | 0.107 | 0.160 |
| C24 past hybrid b4.00 | `-0.077` | `-0.040` | 0.195 | 0.089 | 0.109 | 0.155 |
| C24 future hybrid b4.00 | `0.046` | `0.083` | 0.194 | 0.088 | 0.107 | 0.155 |

H5 结论：

1. `AnchorCollide` 不能再作为 hard reject；它在 past candidates 中与 delta RMSE 呈负相关。
2. 与 old_dyn 的 overlap / correlation 更像 regression warning：当前 ACL2 past-only cue 的优势恰恰不是更像 old_dyn。
3. `D>0.5`、Frag、Anchor 的 run mean 变化很小，不能直接 ranking；它们只适合做 reject / diagnostic。
4. 后续 cue combination 如果把 `D_g` 往 old_dyn/explicit routing 拉，需要先过 read-only gate，不能只看质量统计。

### H6：`D_g` 与 old/explicit/static rescue 小矩阵 gate

按 v4 plan 只做最小 routing family，并先跑 read-only gate。实现上新增了少量固定 cue source：

- `mix.c24past_old_route_lg100_lo025`
- `mix.c24past_old_route_lg100_lo050`
- `mix.c24past_exp_route_lg100_le050`
- `mix.c24past_static_rescue_a025`

固定协议：

- base: `C24 past_only`
- mode: read-only
- commit: `probe_native`
- beta: `4.00`

| Run | Cue | ATE RMSE | Rot RMSE | RPE t | RPE r | D>0.5 | Anchor | Corr old_dyn | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| C24 past read baseline | `acl2.gg.qq.low.g2_4.past_only.headmean.robustq` | `38.5767` | `8.7579` | `92.3961` | `0.0088` | 0.195 | 0.109 | 0.013 | reference |
| H6 old-route lo0.25 | `mix.c24past_old_route_lg100_lo025` | `39.4342` | `9.4605` | `92.3975` | `0.0095` | 0.210 | 0.082 | 0.286 | 明显 regression |
| H6 old-route lo0.50 | `mix.c24past_old_route_lg100_lo050` | `39.5968` | `9.7984` | `92.3955` | `0.0098` | 0.233 | 0.077 | 0.543 | 更差 |
| H6 exp-route le0.50 | `mix.c24past_exp_route_lg100_le050` | `39.5976` | `9.2676` | `92.4010` | `0.0092` | 0.241 | 0.063 | 0.438 | ATE 失败 |
| H6 static rescue a0.25 | `mix.c24past_static_rescue_a025` | `38.6688` | `8.7829` | `92.3957` | `0.0088` | 0.189 | 0.107 | 0.003 | 小幅回退 |

H6 结论：

1. 第一批真正 cue combination 没有通过 read-only gate，因此没有进入 hybrid。
2. old_dyn / explicit routing 明显破坏 ACL2 past-only cue，和 H5 的 old_dyn-overlap regression warning 一致。
3. static rescue 比 routing 温和，但仍没有超过 baseline；后续如继续做 rescue，应该先找更干净的 static cue，而不是加大 alpha。

### H7：intervention path 小测试

H7 目标是验证 `D_g_locked` 除了 frame-attention pair bias 以外，是否能作为更底层的 residual / SWA source gate。先用 query/residual approximation 做安全性测试；随后新增了真实 SWA previous-source soft gate hook，再跑计划中的 group C 小矩阵。

固定协议：

- cue: `C24 past_only`
- mode: read-only
- commit: `probe_native`
- beta/rho: `0.10, 0.20, 0.30`

| Run | Mode | Strength | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| C24 past pair baseline | pair bias | 4.00 | `38.5767` | `8.7579` | `92.3961` | `0.0088` | reference |
| H7 query r0.10 | query gate approx | 0.10 | `42.4656` | `9.4606` | `92.3944` | `0.0087` | 崩 |
| H7 query r0.20 | query gate approx | 0.20 | `43.1880` | `9.8717` | `92.3955` | `0.0089` | 更差 |
| H7 query r0.30 | query gate approx | 0.30 | `43.6837` | `10.1449` | `92.3967` | `0.0091` | 更差 |

真实 SWA previous-source soft gate 补充实现：

- `loger/models/pi3.py`: 在 SWA KV-cache read path 上对 previous source `V_cache` 施加 `g_src=clip(1-rho*D_prev,g_min,1)`；
- `run_pipeline_abc_v2.py`: `probe_native` commit 时保留上一 chunk 的 `D_patch` summary，不改变 native/probe TTT memory；
- `tools/run_attention_cue_experiment.sh`: 新增 `READ_PATH=swa`、`BETA_SWA`、`SWA_GATE_MIN`；
- smoke `ACL2V4_H7SWA_SMOKE2_C23past_r005_g085_e128` 通过，chunk 1 以后 `num_source_gate_applied=1`，说明 gate 真正进入 SWA cache path。

H7 group C 固定协议：

- cue: `C23 past_only`，即 v4 locked `D_g_locked`
- mode: read-only
- commit: `probe_native`
- read path: `swa`
- layer: `first_swa_only`
- source: previous source tail only

| Run | rho | g_min | ATE RMSE | Rot RMSE | RPE t | RPE r | Gate calls | Mean gate | Max gate delta | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H7C-01 | 0.05 | 0.85 | `41.7270` | `8.9823` | `92.3960` | `0.0084` | 37 | 0.997378 | 0.0508 | ATE 失败 |
| H7C-02 | 0.10 | 0.85 | `41.7213` | `9.0020` | `92.3956` | `0.0084` | 37 | 0.994753 | 0.1016 | ATE 失败 |
| H7C-03 | 0.20 | 0.85 | `41.7480` | `9.0197` | `92.3953` | `0.0084` | 37 | 0.990454 | 0.1484 | ATE 失败 |
| H7C-04 | 0.05 | 0.70 | `41.7270` | `8.9823` | `92.3960` | `0.0084` | 37 | 0.997378 | 0.0508 | 同 H7C-01 |
| H7C-05 | 0.10 | 0.70 | `41.7213` | `9.0020` | `92.3956` | `0.0084` | 37 | 0.994753 | 0.1016 | 同 H7C-02 |
| H7C-06 | 0.20 | 0.70 | `41.6927` | `9.0072` | `92.3947` | `0.0084` | 37 | 0.989550 | 0.1992 | H7C best，但仍接近 no-control |

Trajectory diagnostics:

| Run | ATE RMSE | Rot RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| C23 past read baseline | `38.6421` | n/a | `3.034` | `33.553` | `34.061` | `34.004` | `5.212` |
| C23 past hybrid baseline | `38.3706` | n/a | `3.403` | `33.259` | `33.769` | `33.735` | `5.227` |
| H7C-01 SWA r0.05 g0.85 | `41.7270` | `8.9823` | `3.884` | `37.092` | `37.549` | `37.533` | `5.343` |
| H7C-06 SWA r0.20 g0.70 | `41.6927` | `9.0072` | `3.808` | `37.073` | `37.531` | `37.511` | `5.361` |

H7 结论：

1. Query/residual gate approximation 非常不安全，不进入 hybrid。
2. 真实 SWA previous-source gate 已完成工程验证，hook 有效且 37/38 个非首 chunk 都实际施加 source gate。
3. SWA-only intervention 的 ATE 全部回到 `41.69-41.75m`，明显差于 C23 past read baseline `38.6421m`，也没有改善 FinalErr / local segment mean。
4. 因此 H7 group D 的 `FA bias + SWA soft gate` 不触发；`D_g_locked` 的有效 intervention path 仍锁定为 frame-attention pair bias。

### H8：cross-sequence sanity full run

H8 跑三个候选：

- C3 anchor: `g3.full + probe_ttt_write`, beta `3.75`
- C23 past: `g2_3.past_only + probe_ttt_write`, beta `3.75`
- C24 past: `g2_4.past_only + probe_ttt_write`, beta `4.00`

已完成：

- KITTI05 full sequence
- KITTI00 full sequence
- KITTI02 full sequence

资源记录：

- 一开始尝试 6 并发，host RAM available 从数百 GiB 一路降到 `24GiB`，随后 4 并发继续降到 `~7GiB`，有 OOM/swap 风险。
- 因此中断了 C24/02 partial，对 full 00/02 改成 2 并发分批跑完。
- 结论：这类 full long-sequence 任务瓶颈是 host RAM，而不是 GPU 显存；后续 full cross-sequence 应默认 2 并发。

#### KITTI05 full

| Candidate | ATE RMSE | Rot RMSE | RPE t | RPE r | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C3 g3.full | `38.3229` | `12.8591` | `54.8482` | `0.0232` | `133.227` | `31.410` | `33.160` | `33.214` | `9.467` | anchor |
| C23 past | `37.3343` | `12.4947` | `54.8518` | `0.0226` | `129.559` | `30.604` | `32.305` | `32.359` | `9.180` | best ATE/Rot/Yaw |
| C24 past | `37.4202` | `12.5121` | `54.8527` | `0.0227` | `129.540` | `30.724` | `32.422` | `32.474` | `9.192` | FinalErr marginal best |

#### KITTI00 full

| Candidate | ATE RMSE | Rot RMSE | RPE t | RPE r | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C3 g3.full | `49.8130` | `20.3285` | `58.5080` | `0.0233` | `34.482` | `44.871` | `45.114` | `45.621` | `15.382` | anchor |
| C23 past | `49.4601` | `20.0317` | `58.5088` | `0.0233` | `34.703` | `44.679` | `44.918` | `45.427` | `15.156` | stable improvement |
| C24 past | `49.0504` | `19.8193` | `58.5070` | `0.0234` | `32.850` | `44.542` | `44.804` | `45.234` | `14.991` | best on KITTI00 |

#### KITTI02 full

| Candidate | ATE RMSE | Rot RMSE | RPE t | RPE r | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C3 g3.full | `39.3212` | `10.4154` | `68.3005` | `0.0143` | `41.849` | `37.521` | `37.624` | `37.755` | `6.702` | anchor |
| C23 past | `38.4849` | `10.1936` | `68.2982` | `0.0144` | `42.294` | `36.640` | `36.757` | `36.861` | `6.542` | best ATE/Rot/Yaw |
| C24 past | `39.2674` | `10.4786` | `68.2964` | `0.0143` | `46.013` | `37.391` | `37.536` | `37.592` | `6.767` | near C3 ATE, Rot/Final worse |

#### Full-sequence average over KITTI00/02/05

| Candidate | Avg ATE | Avg Rot | Cross-seq conclusion |
|---|---:|---:|---|
| C3 g3.full | `42.4857` | `14.5343` | historical anchor |
| C23 past | `41.7598` | `14.2400` | **best average ATE/Rot; most stable locked candidate** |
| C24 past | `41.9127` | `14.2700` | best on KITTI00, but weaker on KITTI02/05 average |

H8 full 结论：

1. `C23 past` 在 KITTI00/02/05 full average 上最稳，平均 ATE=`41.7598`，平均 Rot=`14.2400`，同时在 KITTI02 和 KITTI05 是 ATE/Rot 主胜者。
2. `C24 past` 在 KITTI00 full 最强，ATE=`49.0504`，Rot=`19.8193`，FinalErr 也最好；但它在 KITTI02 full 明显不如 C23，且 KITTI05 也略弱于 C23。
3. 因此主线应锁为 `D_g_locked = C23 past_only b3.75`，它不是 KITTI01 单点 ATE 最低，但跨序列稳健性最好。
4. `C24 past_only b4.00` 保留为 KITTI01/KITTI00 ATE-oriented 对照，不再作为默认组合 base。
5. H8 满足 promotion sanity：C23 平均 ATE/Rot 优于 C3 anchor，且没有出现跨序列系统性崩坏。

---

## 2026-05-05 ACL2 v3：Cue Combination Validation 启动与 Stage 1 因果隔离

这轮按 `ACL2_v3_Cue_Combination_Validation_Experiment_Plan.md` 开始执行。为了控制时间，没有重复已经完成的 baseline/hybrid sweep，而是优先补最缺的同 beta read-only：这样可以把当前 best 拆成 read correction 与 safe write 两部分。

执行加速策略：

- 不再 8 卡全铺；默认 3-4 并发。
- `tools/run_attention_cue_experiment.sh` 已加 `TORCHINDUCTOR_COMPILE_THREADS=1`，降低启动阶段 CPU compile worker 争抢。
- `acl2` 的 `full` support mean/low/high 计算已改成 support centroid 均值等价形式，减少 g3/full 的 support einsum 开销。
- 新增 `END_FRAME` 环境变量用于短 smoke；新增 `near24/past_only/future_only/overlap_excluded` support 名字兼容。

### Stage 1：`g3.full` 同 beta read-only vs hybrid-safe

固定 cue：

`acl2.gg.qq.low.g3.full.headmean.robustq`

| Beta | Read-only ATE / Rot | Hybrid ATE / Rot | Write gain ATE | 结论 |
|---:|---:|---:|---:|---|
| 1.00 | `39.1170 / 8.8160` | `38.8598 / 8.7889` | `0.2572` | safe write 有效 |
| 1.50 | `38.9516 / 8.8456` | `38.6380 / 8.8205` | `0.3136` | safe write 有效 |
| 2.00 | `38.7950 / 8.8825` | `38.5487 / 8.8710` | `0.2463` | safe write 有效 |
| 2.50 | `38.7894 / 8.9290` | `38.4965 / 8.9072` | `0.2929` | safe write 有效 |
| 3.00 | `38.7381 / 8.9602` | `38.5105 / 8.9582` | `0.2276` | safe write 有效 |
| 3.50 | `38.7531 / 8.9967` | `38.4539 / 8.9772` | `0.2992` | safe write 有效 |
| 3.75 | `38.7322 / 9.0097` | `38.4298 / 8.9846` | `0.3024` | 当前 hybrid best |
| 4.00 | `38.7194 / 9.0221` | `38.4765 / 9.0092` | `0.2429` | read-only 最好，但 hybrid 已回退 |

Stage 1 结论：

1. 之前只用 beta `1.0` 报 read-only `39.1170 / 8.8160`，低估了 `g3.full` read cue 本身；高 beta read-only 可到 `38.7194 / 9.0221`。
2. `probe_ttt_write` 的同 beta ATE gain 稳定存在，约 `0.23-0.31m`，不是单点或 beta confound。
3. Hybrid best 仍是 beta `3.75` 的 `38.4298 / 8.9846`；read-only best 是 beta `4.0` 的 `38.7194 / 9.0221`。
4. 当前 best 的因果解释应修正为：强 read correction 是主贡献，safe branch0 write 提供稳定的额外 `~0.25-0.30m`。
5. 第一批 4 并发 read-only full 约 `24.5 min` 完成；第二批 3 并发约 `24.5 min` 完成。后续必须继续做小批关键实验，而不是全矩阵。

### Stage 2：`g3` support 与邻域单 cue 最小补全

固定协议：

- mode: read-only full KITTI01
- commit: `probe_native`
- beta: `3.75`
- read path: frame attention early layers
- bias: pair

| Run | Cue | ATE RMSE | Rot RMSE | RPE t | RPE r | Mean D>0.5 | Anchor collide | Frag | Corr old_dyn | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ACL2V3_S2_01 | `acl2.gg.qq.low.g3.off246.headmean.robustq` | `38.7532` | `8.9539` | `92.3918` | `0.0089` | 0.194 | 0.152 | 0.128 | -0.099 | 接近 g3.full，但未超过 |
| ACL2V3_S2_02 | `acl2.gg.qq.low.g3.near12.headmean.robustq` | `38.7940` | `8.9820` | `92.3912` | `0.0089` | 0.193 | 0.152 | 0.128 | -0.099 | 不如 full/off246 |
| ACL2V3_S2_03 | `acl2.gg.qq.low.g3.near24.headmean.robustq` | `38.7505` | `8.9604` | `92.3917` | `0.0089` | 0.194 | 0.152 | 0.127 | -0.099 | 接近 off246，未超过 full |
| ACL2V3_S2_04 | `acl2.gg.qq.low.g2_4.full.headmean.robustq` | `38.6137` | `8.8417` | `92.3954` | `0.0088` | 0.196 | 0.199 | 0.119 | -0.255 | **新 read-only best；single cue mining 仍未结束** |
| ACL2V3_S2_07 | `acl2.gg.qq.low.g3_4.full.headmean.robustq` | `38.6981` | `9.0334` | `92.3948` | `0.0091` | 0.190 | 0.153 | 0.148 | -0.086 | ATE 强，但 rotation 比 g2_4 差 |
| ACL2V3_S2_08 | `acl2.gg.qq.low.g2_3.full.headmean.robustq` | `38.6319` | `8.7472` | `92.3943` | `0.0087` | 0.188 | 0.205 | 0.091 | -0.312 | rotation 最好，值得 hybrid 验证 |

Hybrid-safe 验证：

| Run | Cue | Beta | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| ACL2V3_S2_05 | `acl2.gg.qq.low.g2_4.full.headmean.robustq` | 3.75 | `38.3991` | `8.8510` | `92.3937` | `0.0089` | 超过旧 `g3.full` best |
| ACL2V3_S2_06 | `acl2.gg.qq.low.g2_4.full.headmean.robustq` | 4.00 | `38.3805` | `8.8707` | `92.3942` | `0.0089` | **当前新 best，但 ATE 提升为边界级** |
| ACL2V3_S2_09 | `acl2.gg.qq.low.g2_4.full.headmean.robustq` | 4.25 | `38.4205` | `8.8797` | `92.3943` | `0.0089` | 不如 b4.00 |
| ACL2V3_S2_10 | `acl2.gg.qq.low.g2_4.full.headmean.robustq` | 4.50 | `38.4327` | `8.9199` | `92.3941` | `0.0089` | 过强回退 |
| ACL2V3_S2_12 | `acl2.gg.qq.low.g2_3.full.headmean.robustq` | 3.75 | `38.3847` | `8.7583` | `92.3928` | `0.0087` | ATE 近似 best，rotation 明显最好 |

Trajectory diagnostics：

| Run | ATE RMSE | Rot RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `g3.full` hybrid b3.75 | `38.4298` | `8.9846` | `4.692` | `33.353` | `33.860` | `33.808` | `5.477` | 旧 best |
| `g2_4.full` hybrid b4.00 | `38.3805` | `8.8707` | `4.503` | `33.243` | `33.758` | `33.707` | `5.401` | 当前 ATE best，segment mean 全部略优 |
| `g2_3.full` hybrid b3.75 | `38.3847` | `8.7583` | `3.677` | `33.279` | `33.789` | `33.746` | `5.291` | 当前 Pareto best，endpoint/yaw/rotation 最好 |

Stage 2 当前结论：

1. `g3` 的局部 support `off246/near12/near24` 都有效，但没有超过 `g3.full` 高 beta read-only 平台；`full support` 仍是当前 `g3` 最稳设置。
2. 邻域窗口 `g2_4.full` 明显强于 `g3.full` read-only：ATE 从 `38.7322` 降到 `38.6137`，rotation 从 `9.0097` 降到 `8.8417`。
3. `g2_4.full` 的 anchor collision 较高到 `0.199`，但 ATE/Rot 同时改善，说明现有 cue-quality 指标仍只能做 reject gate，不能直接 ranking。
4. `g2_4.full` hybrid beta 4.0 达到 `38.3805 / 8.8707`，比旧 `g3.full` best `38.4298 / 8.9846` 改善 `0.0493m / 0.1139deg`。这是当前 ATE 新 best，但 ATE gain 小于强突破标准。
5. `g2_4` beta `4.25/4.50` 均回退，说明当前局部 best 在 `4.0` 附近。
6. `g2_3.full` hybrid beta 3.75 达到 `38.3847 / 8.7583`，ATE 只比 `g2_4` best 差 `0.0042m`，但 rotation 再好 `0.1124deg`，是当前最强 balanced/Pareto 候选。
7. 轨迹诊断显示 `g2_3.full` 的 final error `3.677m` 明显优于 `g2_4.full` 的 `4.503m` 和旧 `g3.full` 的 `4.692m`，不是简单用 ATE 换 endpoint。
8. 当前 v3 判断：先继续 single-cue 邻域验证，不应立刻围绕旧 `g3.full` 做大规模 cue combination。

---

## 2026-05-05 ACL2 v2 第一轮：非比例 Global Layer Window / Support 深挖

这轮按 `LoGeR_Attention_Cue_Library_v2_Detailed_Experiment_Plan.md` 先补了最核心缺口：不再只用比例切出来的 `middle=[13,15,17,19,21,23]`，而是测试 LoGeR global stack 的非比例 singleton / VGGT4D-mapped window。当前实现是 **head-mean centroid/support 版本**，还不是 per-head strict VGGT4D cache；它用现有 `global_q_raw_patchvec_layers/global_k_raw_patchvec_layers`，在 HMC 内解析 `acl2.*` cue。

Batch 2 固定协议：

- mode: read-only full KITTI01
- commit: `probe_native`
- read path: frame attention early layers
- bias: pair
- beta: 1.0
- normalization: robustq
- 结果目录：`results/kitti01_hmc_v2/attention_cue_library_v2/`

| Run | Cue | ATE RMSE | Rot RMSE | RPE t | RPE r | Mean D>0.5 | Anchor collide | Frag | Corr old_dyn | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ACL2_B2_02 | `acl2.gg.qq.low.g3.full.headmean.robustq` | `39.1170` | `8.8160` | `92.3934` | `0.0087` | 0.191 | 0.098 | 0.123 | 0.086 | **当前最强 read-only attention cue；明显超过旧 `gg.qq.middle.low` read-only** |
| ACL2_B2_06 | `acl2.gg.qq.low.g2_6.off246.headmean.robustq` | `39.4048` | `9.0229` | `92.3959` | `0.0089` | 0.205 | 0.091 | 0.135 | 0.099 | VGGT4D-mapped mid/off246 有效，但不如 g3.full |
| ACL2_B2_01 | `acl2.gg.qq.low.g0.full.headmean.robustq` | `39.6773` | `8.9046` | `92.3981` | `0.0086` | 0.227 | 0.087 | 0.052 | 0.406 | 接近旧 `gg.qq.middle.low` read-only，rotation 很好 |
| ACL2_B2_03 | `acl2.gg.qq.low.g8.full.headmean.robustq` | `39.7868` | `8.9245` | `92.3989` | `0.0087` | 0.192 | 0.140 | 0.111 | -0.020 | 有弱信号，ATE 不够 |
| ACL2_B2_07 | `acl2.gg.qq.low.g12_17.off246.headmean.robustq` | `40.4152` | `9.4251` | `92.3986` | `0.0091` | 0.198 | 0.079 | 0.217 | 0.155 | 深层 off246 不适合作主 D_read |
| ACL2_B2_05 | `acl2.gg.qq.low.g17.full.headmean.robustq` | `40.6034` | `9.4244` | `92.3980` | `0.0091` | 0.203 | 0.094 | 0.102 | 0.045 | 深层 singleton 不推荐 |
| ACL2_B2_08 | `acl2.gg.qq.low.g13_15.off246.headmean.robustq` | `40.9028` | `9.5617` | `92.3978` | `0.0092` | 0.184 | 0.060 | 0.210 | 0.259 | deepvar-mapped window 失败 |
| ACL2_B2_04 | `acl2.gg.qq.low.g13.full.headmean.robustq` | `41.0846` | `9.6347` | `92.3983` | `0.0092` | 0.191 | 0.054 | 0.186 | 0.286 | 深层 g13 失败 |

关键结论：

1. 用户指出的“中层不能只按比例划分”是对的。当前最强 read-only cue 不是比例 middle，而是 global stack `g3`，即实际 decoder global layer id `7`。
2. `acl2.gg.qq.low.g3.full.headmean.robustq` 达到 `39.1170 / 8.8160`，比旧 `gg.qq.middle.low.robustq` read-only `39.6811 / 9.2540` 改善 `0.5641 m`，rotation 同时改善 `0.4380 deg`。
3. VGGT4D-mapped `g2_6.off246` 也有效，`39.4048 / 9.0229`，说明固定 offset support 有价值，但当前最优不是整个 `g2_6` window，而更像局部单层 `g3`。
4. 深层 `g12_17/g13_15/g13/g17` 作为 dynamic read cue 明显失败，更可能只适合作 static rescue / reliability，而不是直接 suppress。
5. 这轮 ACL2 比以前慢很多，因为当前实现把 q/k layer-window/support 统计放在 CPU 上做；同时 TorchInductor 启动阶段会开较多 compile workers，8 个 full run 并发会抢 CPU/IO。后续脚本已加 `TORCHINDUCTOR_COMPILE_THREADS=1` 默认值，并采用 2-4 并发。

### ACL2 Batch 6：`g3.full` hybrid-safe beta sweep

固定协议：

- read cue: `acl2.gg.qq.low.g3.full.headmean.robustq`
- mode: hybrid full KITTI01
- commit: `probe_ttt_write`
- write score: `stage_d`
- read path: frame attention early layers
- bias: pair

| Run | Beta | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---|
| ACL2_B6_01 | 1.00 | `38.8598` | `8.7889` | `92.3914` | `0.0087` | 已超过旧 ACL v1 best |
| ACL2_B6_02 | 1.50 | `38.6380` | `8.8205` | `92.3914` | `0.0088` | 稳定提升 |
| ACL2_B6_03 | 2.00 | `38.5487` | `8.8710` | `92.3906` | `0.0088` | 稳定提升 |
| ACL2_B6_04 | 2.50 | `38.4965` | `8.9072` | `92.3908` | `0.0089` | 明显优于旧 `gg.qq.middle.low` b2.50 |
| ACL2_B6_05 | 2.75 | `38.4901` | `8.9378` | `92.3907` | `0.0089` | 接近 best |
| ACL2_B6_06 | 3.00 | `38.5105` | `8.9582` | `92.3905` | `0.0089` | 略回退 |
| ACL2_B6_09 | 3.25 | `38.4754` | `8.9599` | `92.3903` | `0.0089` | 继续有效 |
| ACL2_B6_07 | 3.50 | `38.4539` | `8.9772` | `92.3902` | `0.0090` | 接近 best |
| ACL2_B6_10 | 3.75 | `38.4298` | `8.9846` | `92.3905` | `0.0090` | **当前 Pipeline v2 / ACL2 新最好** |
| ACL2_B6_08 | 4.00 | `38.4765` | `9.0092` | `92.3901` | `0.0090` | 过强后回退 |

诊断对比：

| Run | ATE RMSE | Rot RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 `gg.qq.middle.low` hybrid b2.50 | `38.9714` | `9.2084` | `5.441` | `33.851` | `34.370` | `34.280` | `5.696` |
| ACL2 `g3.full` hybrid b3.75 | `38.4298` | `8.9846` | `4.692` | `33.353` | `33.860` | `33.808` | `5.477` |

Batch 6 结论：

1. `acl2.gg.qq.low.g3.full.headmean.robustq + probe_ttt_write` 达到 `38.4298 / 8.9846`，比旧 `gg.qq.middle.low + probe_ttt_write` best `38.9714 / 9.2084` 再改善 `0.5416 m`，rotation 同时改善 `0.2238 deg`。
2. 相比 read-only `g3.full` 的 `39.1170 / 8.8160`，hybrid safe write 继续改善 `0.6872 m`，说明 branch0 write 与这个 ACL2 read cue 仍然互补。
3. 最佳强度在 beta `3.50-3.75` 附近；beta `4.00` 已有回退，下一步不应大范围粗扫，而应做局部细扫和 cue 结构 ablation。
4. 新 best 的 final error 从旧 best 的 `5.441 m` 降到 `4.692 m`，不是用 endpoint 损伤换 ATE；主要 50/100/200-frame segment mean 也同步改善。

### ACL2 Batch 7：`g3.full` beta 局部细扫

固定协议同 Batch 6，只在 beta `3.60-3.90` 细扫。

| Run | Beta | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---:|---:|---:|---:|---:|---|
| ACL2_B7_01 | 3.60 | `38.4815` | `8.9994` | `92.3904` | `0.0090` | 接近 B6 best，但未超过 |
| ACL2_B7_02 | 3.70 | `38.4663` | `8.9922` | `92.3901` | `0.0090` | B7 最好 |
| ACL2_B7_03 | 3.80 | `38.4703` | `9.0067` | `92.3905` | `0.0090` | 与 3.70 接近 |
| ACL2_B7_04 | 3.90 | `38.4775` | `9.0207` | `92.3905` | `0.0090` | 继续略回退 |

Batch 7 结论：

1. beta `3.60-3.90` 细扫没有超过 B6 的 beta `3.75` best `38.4298 / 8.9846`。
2. `3.60-3.90` 全部稳定在 `38.46-38.48 m`，说明 B6 的 `g3.full` hybrid gain 不是单点偶然。
3. 继续只扫 beta 的收益已经很小；下一步优先改 cue 结构：`g3` support 和邻域层窗。

下一步：

- 对 `g3` 附近做邻域窗口：`g2_3`, `g3_4`, `g2_4`, `g3_5`；
- 对 `g3` 做 support sweep：`full / off246 / near12 / past / future`；
- per-head cache 暂缓到第二批工程优化后再跑。

---

现在可以分三类看：

**真正可用**
| Cue / 组合 | 最好结果 ATE / Rot | 结论 |
|---|---:|---|
| `acl2.gg.qq.low.g2_3.past_only.headmean.robustq` + `probe_ttt_write` | `38.3706 / 8.6694` at beta 3.75 | **当前 locked main cue**；KITTI01 balanced/Pareto best，KITTI00/02/05 full 平均也最好 |
| `acl2.gg.qq.low.g2_4.past_only.headmean.robustq` + `probe_ttt_write` | `38.3566 / 8.7660` at beta 4.00 | KITTI01 ATE best；KITTI00 full 也最好，但 KITTI02/05 不如 C23，保留为 ATE-oriented 对照 |
| `acl2.gg.qq.low.g2_4.full.headmean.robustq` + `probe_ttt_write` | `38.3805 / 8.8707` at beta 4.00 | 旧 ATE best；现在被 `g2_4.past_only` 超过 |
| `acl2.gg.qq.low.g2_3.full.headmean.robustq` + `probe_ttt_write` | `38.3847 / 8.7583` at beta 3.75 | 旧 balanced/Pareto 候选；现在被 `g2_3.past_only` 超过 |
| `acl2.gg.qq.low.g2_4.past_only.headmean.robustq` read-only | `38.5767 / 8.7579` at beta 4.00 | read-only support sweep best；证明 full support 不是最优 |
| `acl2.gg.qq.low.g2_4.full.headmean.robustq` read-only | `38.6137 / 8.8417` at beta 3.75，`38.6157 / 8.8614` at beta 4.00 | 旧 full-support read-only reference；现在被 `g2_4.past_only` 超过 |
| `acl2.gg.qq.low.g2_3.near12.headmean.robustq` read-only | `38.6177 / 8.7119` at beta 3.75 | C23 read-only 最好，但 hybrid 后未超过 C23 past |
| `acl2.gg.qq.low.g2_3.full.headmean.robustq` read-only | `38.6319 / 8.7472` at beta 3.75 | old C23 read-only balanced reference |
| `acl2.gg.qq.low.g3.full.headmean.robustq` + `probe_ttt_write` | `38.4298 / 8.9846` at beta 3.75 | 旧 ACL2 best；现在被 `g2_4/g2_3.full` 超过 |
| `acl2.gg.qq.low.g3.full.headmean.robustq` read-only | `38.7194 / 9.0221` at beta 4.0，`39.1170 / 8.8160` at beta 1.0 | 高 beta read-only 比原先认识更强，但已弱于 `g2_4/g2_3` |
| `acl2.gg.qq.low.g2_6.off246.headmean.robustq` read-only | `39.4048 / 9.0229` at beta 1.0 | VGGT4D-mapped mid/off246 有效，但弱于 g3.full |
| `gg.qq.middle.low.robustq` + `probe_ttt_write` | `38.9714 / 9.2084` at beta 2.50 | 旧 ACL v1 best；现在被 ACL2 `g3.full` 超过 |
| `gg.qq.middle.low.robustq` read-only | `39.6811 / 9.2540` | 旧 internal-attention cue，read-only 已过 CM02/key-cosine gate，但弱于 ACL2 `g3.full` |
| `old_dyn_addclip` read | `39.3103 / 9.7097` | 旧 Phase F 最好 cue，现在被 attention cue 系统性超过 |
| `old_dyn_addclip` + `residual_reliability` write | `39.3149 / 9.7586` | 几乎追平最好，但没超过 |
| `old_dyn_addclip` + sparse95 resrel write | `39.3864 / 9.7821` | ATE 变差，不推荐主线 |
| `explicit_dyn_only` | `39.4191 / 9.5794` at beta 1.0，`39.4214 / 9.6600` at beta 1.25 | 很强，rotation 比 old_dyn 系更好，是最值得保留的分解 cue |
| `old_dyn_soft_or` | `39.4147 / 9.8046` | 强，但不如 addclip |
| `old_dyn_calibrated_soft_or` | `39.4903 / 9.8299` | D5 参考基线 |
| `residual_reliability` write on old_dyn | `39.4881 / 9.7984` | Phase E tiny best，可作为参考，不是突破 |

**有信号但不够强**
| Cue / 组合 | 最好结果 ATE / Rot | 结论 |
|---|---:|---|
| `key_cosine_avg/shallow/deep` read-only | `39.7820 / 9.7417` | 有效但低于 old_dyn 系 |
| `key_cosine_avg` + safe write | `39.8103 / 9.7032` | safe write 后反而差一点 |
| `alignment_confidence` write | `39.7311 / 9.6980` | rotation 尚可，ATE 不够 |
| `residual_reliability sparse95` on old_dyn | `39.7822 / 9.5632` | rotation 改善，ATE 明显损失 |
| `implicit_dyn_only` | `39.8656 / 9.6406` | 有信号，但不能单独替代 explicit/old_dyn |
| `dyn4d_patch` hybrid | `39.9985 / 9.5368` | 接近 40m，rotation 好一点但 ATE 不行 |
| `acl2.gg.qq.low.g2_4.future_only.headmean.robustq` + `probe_ttt_write` | `38.4096 / 8.5130` | rotation / endpoint 很强，但 ATE 回退且非因果 support，不作主线 |
| `acl2.gg.qq.low.g2_3.future_only.headmean.robustq` read-only | `38.6907 / 8.4343` | rotation 最强之一，但 ATE 回退，不进主线 |
| `acl2.gg.qq.low.g2_5.full.headmean.robustq` read-only | `39.0393 / 9.2193` | window 扩到 g5 后信号仍在，但明显弱于 `g2_3/g2_4` |
| `acl2.gg.qq.low.g1_3.full.headmean.robustq` read-only | `39.1540 / 9.0104` | 有信号，但加入 g1 后弱于 `g2_3` |
| `acl2.gg.qq.low.g4.full.headmean.robustq` read-only | `39.4285 / 9.5862` | g4 单层有弱 ATE 信号，但 rotation/endpoint 代价大 |
| `mix.c24past_static_rescue_a025` read-only | `38.6688 / 8.7829` | static rescue 较温和但仍回退，暂不晋级 |
| `mix.old_route_gg_smd` read-only / hybrid | `39.6804 / 9.6607` read-only，`39.4130 / 9.6559` hybrid | 有 ATE 信号，但 rotation 不如 `gg.qq.middle.low`，不作为主线 |
| `mix.exp_add_fa_decay` / `mix.old_agree_fa_decay` | `40.0357 / 9.6863`，`40.0681 / 9.5101` | fusion 后比纯 frame cue 好，但没有晋级 |
| `fa.key.middle.high.robustq` | `40.3812 / 9.3763` | 当前纯 frame-attention cue 最好，但不够强 |

**目前不推荐 / diagnostic**
| Cue | 最好结果 ATE / Rot | 结论 |
|---|---:|---|
| `gg.smd.product.a1b1g1.robustq` | `40.9064 / 9.0697` | rotation 最干净，但 ATE 太弱，可作 rotation diagnostic |
| `gg.qk.middle.var.robustq` | `40.7386 / 9.2043` | rotation 尚可，ATE 不够 |
| `gg.smd.product.a0b1g1.robustq` | `40.7450 / 9.1944` | rotation 尚可，ATE 不够 |
| `acl2.gg.qq.low.g2.full.headmean.robustq` read-only | `39.7765 / 8.7929` | endpoint / rotation 尚可，但 ATE 太弱；只能作 layer diagnostic |
| `acl2.gg.qq.low.g3_4.full.headmean.robustq` read-only | `38.6981 / 9.0334` | ATE 接近，但 g4 引入明显 Rot/FinalErr/Yaw 损伤，不推荐主线 |
| `mix.c23_g4_soft_l010/l025/l050/l075` read-only | best `38.7295 / 8.8799` | soft-g4 全部未过 read-only gate，lambda 越大越差，不进入 hybrid |
| `gg.kk.middle.low.robustq` | `41.3694 / 9.4865` | 不推荐 |
| `fa.key.l0/l4/shallow_deep/deep_low/layerVar` | best `40.5603 / 9.2535` among these | 有弱信号或 diagnostic value，但不能晋级 |
| `manual_implicit_dyn` read-only | `40.1474 / 9.5945` | 不够 |
| `qk_var` | `40.2060 / 9.5394` | 不够 |
| `qqkk_disagree` | `40.8638 / 9.4104` | ATE 太差 |
| `gram4d` | `41.1249 / 9.2537` read-only，`40.8677 / 9.2598` hybrid | rotation 好但 ATE 崩 |
| `entropy` | `41.0887 / 9.9225` | 不推荐 |
| `query_cosine_*` / `shallow_deep_disagree` | `41.7316 / 8.9914` | 基本像 no-control，不能作为主 cue |
| `mix.c24past_old_route_*` read-only | best `39.4342 / 9.4605` | old_dyn routing 破坏 ACL2 past cue，H6 gate failed |
| `mix.c24past_exp_route_*` read-only | `39.5976 / 9.2676` | explicit routing ATE 失败 |
| `C24 past query-gate approx` read-only | best `42.4656 / 9.4606` | query/residual approximation 崩，不进入 hybrid |
| `C23 past SWA previous-source gate` read-only | best `41.6927 / 9.0072` | 真 SWA hook 已验证生效，但 SWA-only intervention ATE 接近 no-control，不进入 FA+SWA 组合 |


一句话：当前 KITTI01 ATE best 仍是 `g2_4.past_only + probe_ttt_write = 38.3566 / 8.7660`；但 H8 full KITTI00/02/05 显示 `g2_3.past_only + probe_ttt_write = 38.3706 / 8.6694` 跨序列平均最好，因此 `D_g_locked` 应锁 C23 past。H3 soft-g4、H6 组合、H7 query/residual approximation 和 H7 真 SWA previous-source gate 都没有通过 gate；`docs/ACL2_v4_EarlyMidWindow_CausalValidation_and_Combination_Readiness_Plan.md` 中当前可执行的 v4 实验已收口，可以停止继续跑 v4。

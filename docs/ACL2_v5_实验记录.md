# ACL2 v5 实验记录

日期：2026-05-06  
计划文件：`docs/ACL2_v5_DgLocked_PerHead_TTT_SWA_Acceleration_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked/`  
主线 cue：`acl2.gg.qq.low.g2_3.past_only.headmean.robustq`  
v5 起始主线协议：frame-attention early-layer pair bias，beta `3.75`，`probe_ttt_write`  
当前最强 frame-attention 读控制协议：`g2_3.past_only` cue + `pair/all`，beta `4.75`

说明：从 v5 开始，实验数据和结论统一记录在本文件；`docs/cue.md` 只保留 v4 及以前的汇总状态，不再作为 v5 主记录文件。

## 关键代码文件索引

后续定位实验实现时，优先看这些文件：

| 文件 | 负责内容 | 相关实验/关键词 |
|---|---|---|
| `run_pipeline_abc_v2.py` | 主 pipeline CLI 和参数接线；`RESET_EVERY`、chunk/overlap、`probe_ttt_write`、TTT/SWA 控制参数都从这里进入模型与 HMC | reset 固定、`--hmc_write_score_source`、`--ttt_write_*`、`--swa_write_*` |
| `loger/pipeline/hybrid_memory_controller.py` | HMC 主逻辑；解析 ACL2 cue，生成 `D_g`，计算 cue quality，构造 frame-attention bias，改写 TTT write score/branch/layer/EMA/scope，并写 debug jsonl | `C23/C24`、`stage_d_x_dg_inv_sqrt`、`EXPV`、`TTEX`、`TTEMA`、`TTLB`、`TTFG` |
| `loger/pipeline/ttt_write_controller.py` | TTT delayed write-back / replay commit 的核心实现；负责 token prior、branch/layer policy、feature gate、hard token filter、native mix、commit EMA、native-delta gate、commit propagation filter | `TTCPF`、`TTEX`、`TTEMA`、`TTNDG`、`TTFG`、`TTFS`、`TTVM` |
| `loger/models/pi3.py` | LoGeR/PI3 模型执行和 attention/cache hook；SWA read/write、overlap source gate/replace、keep-scope selection、SWA cache blend 等底层路径主要在这里生效 | `SWAW`、`SWAC`、`SWOV`、`SWOVR`、`SWOC`、`SWKS`、`SWRP` |
| `tools/run_attention_cue_experiment.sh` | 标准实验启动脚本；把环境变量翻译成 `run_pipeline_abc_v2.py` 参数，控制 GPU、seq、mode、cue、beta、write score、默认 reset | 所有 `ACL2V5_*` full/smoke run |
| `tools/kitti_trajectory_diagnostics.py` | 轨迹诊断脚本；生成 Sim3 对齐、FinalErr、segment mean、Yaw RMSE、chunk error 图和 `diagnosis.md` | `Trajectory diagnostics` 表格 |
| `eval/long_eval_script/kitti_benchmark` | KITTI ATE/RPE 基础评测二进制 | `results_sim3/results_ate.txt`、`results_rpe.txt` |
| `docs/ACL2_v5_DgLocked_PerHead_TTT_SWA_Acceleration_Plan.md` | v5 原始计划书；用于看预设假设和阶段目标 | plan reference |
| `docs/ACL2_v5_ExplicitDyn_TTT_WriteVeto_补充实验计划.md` | explicit dyn 写入 veto 的补充计划；结果仍统一记在本文件 | `EXPV` |

定位建议：

- 想查 **TTT 写入有没有 bug / 写入规则怎么生效**：先看 `hybrid_memory_controller.py`，再看 `run_pipeline_abc_v2.py` 的参数接线。
- 想查 **SWA overlap / source cache / K/V replacement**：先看 `pi3.py`，再回到 `hybrid_memory_controller.py` 看控制信号和 debug。
- 想复现实验命令：看对应 section 的固定协议，再看 `tools/run_attention_cue_experiment.sh`。
- 想核对指标来源：看 run 目录下 `results_sim3/`，复杂诊断看对应 `trajectory_diagnostics_*` 目录。

## Cue 名字怎么读

v5 里最常见的 cue 名字长这样：

```text
acl2.gg.qq.low.g2_3.past_only.headmean.robustq
```

它不是一个抽象代号，可以按下面拆开：

| 片段 | 具体含义 |
|---|---|
| `acl2` | Attention Cue Library v2。直接用 LoGeR decoder global stack 导出的 patch-level Q/K 向量做动态/不稳定区域估计，不是外部语义分割，也不是 optical flow。 |
| `gg` | 使用 LoGeR 的 global attention / global decoder stack 特征。这里是历史命名，核心意思是“从 global Q/K patch vector 里算 cue”。 |
| `qq` | 当前 patch 的 query 向量和 support 帧的 query centroid 比相似度。对应地，`kk` 是 key-key，`qk` 是 query-key。当前最好的一支是 `qq`。 |
| `low` | 相似度低的地方给高分。换句话说，一个 patch 如果不像历史 support 帧里的稳定结构，它的 `D_g` 会高，更像动态/不稳定/不适合长期传播区域。 |
| `g2_3` | 使用 global stack 的第 2 到第 3 层，闭区间。`g2_4` 表示第 2 到第 4 层，`g3` 表示只用第 3 层。 |
| `past_only` | support 只看当前 chunk 里该帧之前的帧，不看未来帧。它比 `future_only` 更适合 causal / streaming 使用。 |
| `headmean` | 当前轻量实现：先做 head-mean / support centroid 统计，不是更重的 per-head strict attention-map cache。 |
| `robustq` | 用 robust quantile 把原始相似度分数归一化到 `[0,1]`，让不同 chunk/frame 的阈值更稳定。 |

这类 cue 最后输出的是一个 patch-level map，文档里通常记作 `D_g`：

- `D_g` 高：该 patch 更像动态、不稳定、跨帧不一致，读控制里应该少相信它；TTT/SWA 写入里通常也要更谨慎。
- `D_g` 低：该 patch 更像静态结构或可长期传播信息。

### 常用 cue 对照

| 简写 | 完整 cue | 它实际在问什么 | 当前角色 |
|---|---|---|---|
| `C23 past` | `acl2.gg.qq.low.g2_3.past_only.headmean.robustq` | 用 global 第 2-3 层 query 特征，只和过去帧比。如果当前 patch 不像过去帧里的结构，就认为它动态/不稳定。 | **当前主线 cue**。v4/H8 后锁定，v5 的 `pair/all` 读控制也用它。 |
| `C24 past` | `acl2.gg.qq.low.g2_4.past_only.headmean.robustq` | 比 `C23 past` 多看第 4 层。 | KITTI01 ATE-oriented 对照。v4 单点 ATE 很强，但 v5 `pair/all` 后 ATE 输给 C23。 |
| `C23 full` | `acl2.gg.qq.low.g2_3.full.headmean.robustq` | 用第 2-3 层，和当前 chunk 内所有其他帧比。 | 旧 balanced baseline，被 `C23 past` 超过。 |
| `C24 full` | `acl2.gg.qq.low.g2_4.full.headmean.robustq` | 用第 2-4 层，和所有其他帧比。 | 旧 KITTI01 ATE baseline，被 `C24 past` 超过。 |
| `C23 near12` | `acl2.gg.qq.low.g2_3.near12.headmean.robustq` | 只和相邻 1-2 帧比，包括前后帧。 | read-only 上不错，但 hybrid 后没有超过 `C23 past`。 |
| `C23 future` | `acl2.gg.qq.low.g2_3.future_only.headmean.robustq` | 只和未来帧比。 | rotation diagnostic 很强，但非因果且 ATE 回退，不作为主线。 |
| `C24 future` | `acl2.gg.qq.low.g2_4.future_only.headmean.robustq` | 第 2-4 层，只和未来帧比。 | rotation / endpoint diagnostic，不作为主线。 |
| `g3 full` | `acl2.gg.qq.low.g3.full.headmean.robustq` | 只用 global 第 3 层，和所有其他帧比。 | v2/v3 旧 attention cue anchor，后来被 `g2_3/g2_4` 系列超过。 |
| `g4 full` | `acl2.gg.qq.low.g4.full.headmean.robustq` | 只用 global 第 4 层。 | 有弱 ATE 信号，但 Rot/Final/Yaw 代价大，只做 diagnostic。 |

### 为什么主线叫 C23 past

`C23 past` 不是“第 23 个实验”的意思，而是：

```text
C23 = candidate using g2_3 layer window
past = support only uses previous frames
```

选择它作为主线的原因：

1. 在 KITTI01 上它不是最低单点 ATE，`C24 past` 曾经更低一点。
2. 但在 KITTI00/02/05 full-sequence 平均上，`C23 past` 的 ATE/Rot 最稳。
3. 进入 v5 的 `frame-attention pair/all` 后，`C23 past` 明显强于 `C24 past`：

| Cue + 读控制 | KITTI01 ATE / Rot | 结论 |
|---|---:|---|
| `C23 past + pair/all` beta `4.75` | `36.6803 / 6.3855` | 当前 TTT-only 主线 |
| `C24 past + pair/all` beta `4.50` | `37.2757 / 6.0611` | Rot 更干净，但 ATE 明显差 |

所以后续如果没有特别说明，文档里的 `D_g_locked`、`C23`、`C23 past` 都指：

```text
acl2.gg.qq.low.g2_3.past_only.headmean.robustq
```

## Run ID 与缩写说明

本文件保留原始 run id，是为了和 `results/...` 目录、日志、CSV 一一对应；不再把历史 run 重命名。阅读时可以按下面的规则拆：

```text
ACL2V5_<实验族缩写>_<编号>_<可选标签>
```

- `ACL2V5`：ACL2 v5 系列实验。
- `<实验族缩写>`：一组共享假设/实现路径的实验族，例如 `SWOVR`、`TTEMA`。
- `<编号>`：该实验族里的本地编号，只代表顺序，不代表强度大小。
- `<可选标签>`：给人看的简短提示，例如 `C23pairall`、`sourceV`、`a025`。
- 表格里如果写短名 `SWOVR_02`，默认就是 `ACL2V5_SWOVR_02`。

`SWOVR_02` 的完整含义：

`ACL2V5_SWOVR_02` = **SWA Overlap-source V Replacement 第 2 个实验**。它在当前 v5 active base 上，使用 `C23 past_only` + frame-attention `pair/all` + `probe_ttt_write` + `TTT_WRITE_NATIVE_MIX_SCALES=1.10,1.00,1.00`，并额外对 SWA overlap source 做 `V` 替换：

- `SWOVR`：SWA overlap source replacement，小范围测试 overlap 区域的 source cache 是否污染后续读。
- `_02`：该小矩阵第二个配置。
- `mode=source`：动态区域判断只用上一 chunk tail-overlap 的 `D_g`。
- `target=v`：只替换 SWA cache 里的 `V`，不动 `K`。
- `alpha=0.25`：替换强度 0.25。
- 结果：KITTI01 `36.5915 / 6.4307`，是当前 v5 tiny best，但仍远没达到成功标准 `ATE < 30`。

常用候选/协议缩写：

| 缩写 | 含义 |
|---|---|
| `C23` | `acl2.gg.qq.low.g2_3.past_only.headmean.robustq`，v4/H8 后锁定的主 `D_g_locked` cue |
| `C24` | `acl2.gg.qq.low.g2_4.past_only.headmean.robustq`，KITTI01 ATE-oriented 对照 |
| `D_g` | ACL2 attention 产生的动态区域图；值越高越倾向动态/应抑制 |
| `explicit dyn` / `E` | 旧 explicit dynamic cue，用来辅助判断动态区是否准确 |
| `stage_d` | 原始 safe TTT write score |
| `stage_d_x_dg_inv_sqrt` | `stage_d * sqrt(1-D_g)`，让 TTT 少写高 `D_g` 动态区域 |
| `pair/all` | frame-attention pair bias 打到所有 read layers |
| `probe_ttt_write` | 使用 probe TTT commit，并允许本轮写入策略修改 |
| `TTT` | test-time training memory/write path |
| `SWA` | sliding-window attention cache/read-write path |
| `KV` / `K` / `V` | attention cache 的 key/value |
| `branch0/1/2` 或 `w0/w1/w2` | TTT 写入的不同 branch/weight slot |
| `tail_overlap` | 当前 chunk 尾部 overlap 帧 |
| `head_overlap` | 当前 chunk 头部 overlap 帧，通常对应上一 chunk tail |
| `RESET_EVERY=5` | LoGeR 对齐要求，v5 后续不改 reset 机制 |

主要实验族缩写：

| 实验族 | 含义 |
|---|---|
| `S0` | smoke / correctness 检查 |
| `T1/T2/T3` | 早期 TTT write-score、强度、beta decouple 小矩阵 |
| `TF` | focal-loss-like TTT writing |
| `NR` | non-TTT read intervention gate |
| `PA/PB` | `pair/all` read intervention 及后续 policy-B |
| `EXPV` | explicit dynamic cue 写入 veto，测试 TTT 少写动态区域 |
| `RDSP` | read mask sparsification |
| `WRSP` | sparse write gate |
| `TTW` | TTT wide prior / 更宽动态先验 |
| `TTS` | suppressive TTT 写入 |
| `TTDYN` | TTT dynamic map accuracy variants |
| `TTDS` | TTT delta scale |
| `TTBR` | TTT branch delta scale |
| `TTBL` | TTT branch-layer delta |
| `TTBM` | TTT branch-mask / write-veto |
| `TTSC` | TTT scope / tail-overlap replay |
| `TTFL` | TTT floor for overlap scope |
| `TTVT` | TTT tail-overlap veto |
| `TTDO` / `TTDV` | TTT overlap drop / value-branch drop |
| `TTFG` | TTT replay feature gate |
| `TTFS` | TTT same-frame static replay target |
| `TTVM` | TTT value-memory (`w1`) replay target |
| `TTLB` | TTT layer/branch policy |
| `TTEX` | TTT native-mix / extrapolation，主要由 `TTT_WRITE_NATIVE_MIX_SCALES` 控制 |
| `TTEMA` | TTT commit EMA |
| `TTNDG` | TTT native-delta gate |
| `TTCPF` | TTT commit propagation filter：当前 chunk 的 TTT 使用不变，只在提交给下一 chunk 的 fast weights 上按 overlap 动态风险做遗忘/缩放 |
| `TTOVF` | TTT overlap forget：当前 chunk full TTT 不变，只在提交 replay 时遗忘 overlap 范围内的高动态 token |
| `TTOVFB` | TTT overlap forget blend：`TTOVF` 的软混合版本，commit fast weight 从 full replay 按 `blend` 部分拉向 filtered replay |
| `TTOVFL` | TTT overlap forget layer-specific：`TTOVFB/TTOVF` 只作用于指定 TTT layer 段，验证动态信息是否只应在传播层被遗忘 |
| `TTOVFLC` | TTT overlap/full-chunk forget layer-specific chunk sanitize：把 filtered replay 从 tail seam 扩到整个 chunk 的 per-frame static token，仍只在 commit 侧生效 |
| `TTCPFL` | TTT commit propagation filter layer-specific：`TTCPF` 只作用于指定 TTT layer 段，测试跨 chunk fast-weight 遗忘是否应按 layer 分配 |
| `TTCPF2` | 第二轮 TTT commit propagation filter：主要调温和度，避免 TTCPF/TTCPFL 的过强 commit decay 伤 ATE |
| `TTAD` | TTT aligned dynamic propagation：当前 chunk full TTT 不变，commit 时只保留与 static/filtered replay 更新方向一致的 dynamic delta |
| `TTDTTL` | TTT dynamic TTL：本 chunk 允许 dynamic residual 短期进入 TTT fast weights，但下一次 commit 从长期 fast weights 中扣除上一轮 dynamic residual |
| `TTPROJ` | TTT projected anti-dynamic commit：当前 chunk 允许 full dynamic update，但提交到下一 chunk 时只削弱 dynamic residual 中与 static update 不同向/正交的部分 |
| `TTALIGN` | TTT cache/token alignment bug check |
| `SWAW` | 第一批 SWA write gate |
| `SWAC` | centered SWA gate |
| `SWOV` | SWA overlap write / overlap K/V centering |
| `SWOVR` | SWA overlap-source replacement，例如 `SWOVR_02` |
| `SWOC` | SWA current-overlap source replacement |
| `SWKS` | SWA keep-scope structural source selection，当前在跑的结构性保留/丢弃 overlap 实验 |
| `SWTR` | SWA history truncation |
| `SWSC` | SWA score-source compare |
| `SWOD` | SWA overlap direct gate |
| `SWKP` | SWA keep/prune overlap history |
| `SWAQ2/SWAQ3` | SWA overlap read-side compact bias variants |
| `SWAS` | SWA aligned source gate |
| `SWBO` | SWA both/head overlap write |
| `SWRP` | SWA source replacement K/V |
| `SWCB` | SWA cache blend |
| `SWRS` | SWA residual centering |

指标缩写：

| 指标 | 含义 |
|---|---|
| `ATE RMSE` | Sim3 对齐后的轨迹位置 RMSE，主指标，越低越好 |
| `Rot RMSE` | KITTI rotation RMSE，越低越好 |
| `RPE t/r` | relative pose error 的平移/旋转项 |
| `FinalErr` | 最后一帧位置误差 |
| `50f/100f/200f mean` | 对固定长度片段的平均 ATE |
| `Yaw RMSE` | yaw 角误差 |
| `Sim3 scale` | Sim3 对齐尺度，异常漂移时作为诊断 |

---

## 0. v5 启动状态

v4 已经把主线锁到：

```text
D_g_locked = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
beta = 3.75
commit = probe_ttt_write
write score = stage_d
```

v4 主线参考：

| Candidate | ATE RMSE | Rot RMSE | Final error | 50f mean | 100f mean | 200f mean | Yaw RMSE | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `C23 past + stage_d` | `38.3706` | `8.6694` | `3.403` | `33.259` | `33.769` | `33.735` | `5.227` | v4 locked mainline |
| `C24 past + stage_d` | `38.3566` | `8.7660` | `4.244` | `33.207` | `33.723` | `33.675` | `5.325` | KITTI01 ATE-oriented shadow |

v5 的第一步先不做 per-head 大矩阵，而是固定 `D_g_locked` 做 TTT write-policy 小矩阵，验证 `D_g` 是否能从 read suppress cue 转成 write eligibility。

---

## 1. 工程加速与 correctness smoke

本轮先做了轻量加速，目标是减少不必要 cue 计算：

- 新增 `--fast_cue_eval`，只在 ACL2 headmean robustq cue 且无 read protection 时启用。
- fast path 只计算当前 `acl2.gg.*.headmean.robustq` 需要的 `D_patch`、基础 quality 和 frame bias。
- fast path 跳过 old_dyn fusion、flow proxy、frame-attention key maps、Gram/global extra diagnostics 等本轮 TTT write-policy 不需要的 cue。
- 新增 Dg-derived TTT write score：
  - `dg_inv` = `1-D_g`
  - `dg_inv_sqrt` = `sqrt(1-D_g)`
  - `dg_inv_sq` = `(1-D_g)^2`
  - `stage_d_x_dg_inv`
  - `stage_d_x_dg_inv_sqrt`
- `residual_reliability` 保持为旧 diagnostic，对照它的计算开销和效果。

Smoke:

| Run | END_FRAME | Write score | 结论 |
|---|---:|---|---|
| `ACL2V5_S0_SMOKE_C23past_fast_e128` | 128 | `stage_d` | 通过；`fast_cue_eval=True`，frame bias 生效，`probe_ttt_write` 生效 |
| `ACL2V5_S0_SMOKE_C23past_dginv_e128` | 128 | `dg_inv` | 通过；TTT write override 生效 |

资源策略：

- KITTI01 full 默认 4 并发；本轮 fast path 后尝试 6 并发，host RAM 仍保持安全。
- 6 并发峰值记录约 `145GiB available`，未触发 swap/OOM 风险。
- 不再尝试 8 并发；v4 已证明 8 full 并发会把 host RAM 压到危险区。

---

## 2. T1：固定 `D_g_locked` 的 TTT write-score 小矩阵

固定协议：

- read cue: `acl2.gg.qq.low.g2_3.past_only.headmean.robustq`
- read beta: `3.75`
- mode: hybrid full KITTI01
- commit: `probe_ttt_write`
- read path: frame attention early layers
- bias: pair
- fast cue eval: enabled

运行时间：

| Batch | Runs | GPU | Time | 备注 |
|---|---|---|---|---|
| T1 batch A | T1-01 到 T1-04 | 0-3 | `02:57:37 -> 03:23:44` | 4 full runs，约 26 min |
| T1 batch B | T1-05 到 T1-06 | 4-5 | `03:08:19 -> 03:34:17` | 2 full runs，约 26 min |
| T1 diagnostic | T1-07 | 0 | `03:26:42 -> 03:53:04` | residual write diagnostic，约 26 min |

### 2.1 Global metrics

| Run | Write score | ATE RMSE | Rot RMSE | RPE t | RPE r | Mean write score | Mean D>0.5 | Frag | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ACL2V5_T1_01` | `stage_d` | `38.3706` | `8.6694` | `92.3930` | `0.0086` | `1.0000` | `0.1904` | `0.0970` | baseline exact repeat |
| `ACL2V5_T1_02` | `1-D_g` | `38.3737` | `8.7215` | `92.3934` | `0.0087` | `0.7843` | `0.1903` | `0.0970` | ATE 接近，但 Rot 变差 |
| `ACL2V5_T1_03` | `sqrt(1-D_g)` | `38.3820` | `8.6301` | `92.3922` | `0.0086` | `0.8465` | `0.1904` | `0.0970` | Rot/Yaw 改善，ATE 小回退 |
| `ACL2V5_T1_04` | `stage_d*(1-D_g)` | `38.4613` | `8.6363` | `92.3934` | `0.0086` | `0.3901` | `0.1904` | `0.0969` | Final/Yaw 好，ATE 回退过大 |
| `ACL2V5_T1_05` | `(1-D_g)^2` | `38.3737` | `8.7215` | `92.3934` | `0.0087` | `0.7120` | `0.1903` | `0.0970` | 与 `1-D_g` 指标等价，未带来收益 |
| `ACL2V5_T1_06` | `stage_d*sqrt(1-D_g)` | `38.3803` | `8.6418` | `92.3945` | `0.0086` | `0.4207` | `0.1904` | `0.0971` | balanced 候选，ATE 近似 baseline |
| `ACL2V5_T1_07` | `residual_reliability` | `38.5878` | `8.7178` | `92.3943` | `0.0087` | `0.0528` | `0.1904` | `0.0970` | 明显失败，不进入后续 |

### 2.2 Trajectory diagnostics

诊断输出：

`results/kitti01_hmc_v2/acl2_v5_dglocked/t1_write_policy_trajectory_diagnostics_full/`

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stage_d` | `38.3706` | `3.403` | `33.259` | `33.769` | `33.735` | `5.227` | `30.725774` |
| `1-D_g` | `38.3737` | `3.861` | `33.231` | `33.740` | `33.686` | `5.265` | `30.723982` |
| `sqrt(1-D_g)` | `38.3820` | `3.421` | `33.256` | `33.768` | `33.716` | `5.200` | `30.716360` |
| `stage_d*(1-D_g)` | `38.4613` | `3.085` | `33.359` | `33.869` | `33.808` | `5.182` | `30.722062` |
| `(1-D_g)^2` | `38.3737` | `3.861` | `33.231` | `33.740` | `33.686` | `5.265` | `30.723982` |
| `stage_d*sqrt(1-D_g)` | `38.3803` | `3.095` | `33.268` | `33.776` | `33.734` | `5.189` | `30.734863` |
| `residual_reliability` | `38.5878` | `3.151` | `33.517` | `34.025` | `33.976` | `5.250` | `30.738883` |

### 2.3 T1 结论

1. `stage_d` 仍是 KITTI01 ATE 最优 TTT write score。它精确复现 v4 locked mainline：`38.3706 / 8.6694`。
2. 直接把 `D_g` 转成写入资格没有带来 ATE 突破：`1-D_g` 和 `(1-D_g)^2` 都是 `38.3737m`，只比 stage_d 差 `0.0031m`，但 rotation 退到 `8.7215deg`。
3. `sqrt(1-D_g)` 有 balanced 价值：ATE 只回退 `0.0114m`，Rot 从 `8.6694` 改善到 `8.6301`，Yaw 从 `5.227` 改善到 `5.200`。
4. `stage_d*sqrt(1-D_g)` 是本批最值得保留的 balanced write variant：ATE `38.3803`，Rot `8.6418`，FinalErr `3.095`，Yaw `5.189`。它没有超过主 ATE，但 FinalErr 比 stage_d 好 `0.308m`，Yaw 好 `0.038`。
5. `stage_d*(1-D_g)` 把 write score 压得太低，FinalErr/Yaw 变好，但 ATE 回退到 `38.4613`，不适合作主线。
6. `residual_reliability` 在 C23 past 上失败：ATE `38.5878`，segment mean 全部变差，不进入 T2/T3。
7. 当前判断：`D_g` 更适合作 read suppress cue；作为 TTT write eligibility 只能产生 balanced/endpoint trade-off，尚不能替代 `stage_d`。

### 2.4 T1 后续决策

进入下一步的候选：

| Candidate | 进入原因 | 下一步 |
|---|---|---|
| `stage_d` | ATE baseline 仍最优 | 作为 T2/T3 anchor |
| `sqrt(1-D_g)` | Rot/Yaw 改善，ATE 回退很小 | 可做 write strength 小扫 |
| `stage_d*sqrt(1-D_g)` | FinalErr/Yaw 改善明显，ATE 接近 baseline | 可做 write strength 小扫；优先级最高 |

不继续：

| Candidate | 停止原因 |
|---|---|
| `1-D_g` | ATE 接近但 Rot/FinalErr 变差 |
| `(1-D_g)^2` | 与 `1-D_g` 指标等价，没有额外价值 |
| `stage_d*(1-D_g)` | ATE 回退过大 |
| `residual_reliability` | ATE 和 segment mean 明显失败 |

推荐下一批：

```text
T2 write_strength sweep:
    read cue = C23 past
    read beta = 3.75
    write candidates = stage_d*sqrt(1-D_g), sqrt(1-D_g)
    write_strength = 0.50, 0.75, 1.25
```

如果 `stage_d*sqrt(1-D_g)` 在某个 strength 下保持 ATE <= `38.3706 + 0.03` 且 FinalErr / Rot 继续改善，则作为 weak balanced write-policy variant 保留；否则 TTT write 主线继续锁 `stage_d`。

---

## 3. 当前 v5 状态

已完成：

- fast cue eval 工程 smoke。
- `D_g_locked` TTT write-score T1 小矩阵。
- T1 trajectory diagnostics。

暂未执行：

- H1 passive interpretation dashboard。
- H2 per-head passive cache / read-only gate。
- T2 write strength sweep。
- H5 SWA FA+SWA local-memory matrix。

当前主线仍为：

```text
D_g_locked = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
beta = 3.75
commit = probe_ttt_write
write score = stage_d
KITTI01 = 38.3706 / 8.6694
```

当前 v5 weak balanced candidate：

```text
D_g_locked = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
beta = 3.75
commit = probe_ttt_write
write score = stage_d*sqrt(1-D_g)
KITTI01 = 38.3803 / 8.6418
FinalErr = 3.095
YawRMSE = 5.189
```

---

## 4. T2：write strength sweep

### 4.1 固定协议

```text
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read beta = 3.75
mode = hybrid-safe full KITTI01
commit = probe_ttt_write
read path = frame attention early layers
bias = pair
FAST_CUE_EVAL = 1
```

T2 只对 T1 保留下来的两个候选做 write alpha 小扫：

```text
sqrt(1-D_g)
stage_d * sqrt(1-D_g)
WRITE_ALPHA = 0.05 / 0.075 / 0.125
```

资源记录：

- 6 并发运行，GPU 0-5。
- 开始：`2026-05-06 03:56:56`。
- 完成时间范围：`04:21:53 -> 04:22:54`。
- 6 个 full run 总 walltime 约 `26.0 min`。
- 运行中 host RAM 最低观测 available 约 `104GiB`；收尾后恢复到约 `489GiB` available。6 并发对 KITTI01 full 可用，但已经会把内存推高，不建议继续加到 8 并发。

### 4.2 T2 benchmark 指标

| Run | Write score | WRITE_ALPHA | ATE RMSE | Rot RMSE | RPE t | RPE r | Mean write score | Mean D>0.5 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| T1 baseline | `stage_d` | 0.100 | `38.3706` | `8.6694` | `92.3930` | `0.0086` | n/a | 0.190 | v4 locked baseline |
| T2-01 | `sqrt(1-D_g)` | 0.050 | `38.5350` | `8.7331` | `92.3942` | `0.0087` | 0.846481 | 0.190353 | 写入太弱，回退 |
| T2-02 | `sqrt(1-D_g)` | 0.075 | `38.5181` | `8.7058` | `92.3949` | `0.0087` | 0.846479 | 0.190362 | 仍回退 |
| T2-03 | `sqrt(1-D_g)` | 0.125 | `38.3608` | `8.6580` | `92.3937` | `0.0086` | 0.846472 | 0.190370 | 小幅超过 baseline |
| T2-04 | `stage_d*sqrt(1-D_g)` | 0.050 | `38.5818` | `8.7640` | `92.3956` | `0.0087` | 0.420706 | 0.190364 | 写入太弱，明显回退 |
| T2-05 | `stage_d*sqrt(1-D_g)` | 0.075 | `38.3907` | `8.6302` | `92.3932` | `0.0086` | 0.420709 | 0.190344 | Rot/Yaw 强，ATE 小回退 |
| T2-06 | `stage_d*sqrt(1-D_g)` | 0.125 | `38.2958` | `8.5996` | `92.3926` | `0.0086` | 0.420704 | 0.190384 | **当前 v5 KITTI01 新 best** |

### 4.3 T2 trajectory diagnostics

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `stage_d` baseline | `38.3706` | `3.403` | `33.259` | `33.769` | `33.735` | `5.227` | `30.725774` | v4 locked baseline |
| T1 `stage_d*sqrt` a0.100 | `38.3803` | `3.095` | `33.268` | `33.776` | `33.734` | `5.189` | `30.734863` | endpoint/Yaw 好，ATE 未过 baseline |
| `sqrt` a0.050 | `38.5350` | `3.415` | `33.446` | `33.955` | `33.894` | `5.261` | `30.727792` | 回退 |
| `sqrt` a0.075 | `38.5181` | `3.261` | `33.411` | `33.920` | `33.877` | `5.235` | `30.739815` | 回退 |
| `sqrt` a0.125 | `38.3608` | `3.503` | `33.241` | `33.750` | `33.698` | `5.221` | `30.728110` | ATE/segment 小幅过 baseline，但 FinalErr 差 |
| `stage_d*sqrt` a0.050 | `38.5818` | `3.013` | `33.546` | `34.050` | `34.015` | `5.278` | `30.755048` | endpoint 好但 ATE 失败 |
| `stage_d*sqrt` a0.075 | `38.3907` | `3.297` | `33.283` | `33.794` | `33.733` | `5.180` | `30.723005` | Yaw 最强之一，ATE 小回退 |
| `stage_d*sqrt` a0.125 | `38.2958` | `3.385` | `33.177` | `33.685` | `33.628` | `5.156` | `30.717566` | **ATE/segment/Yaw 全部最好，FinalErr 与 baseline 接近** |

### 4.4 T2 结论

1. T2 找到 v5 第一处实质突破：`stage_d*sqrt(1-D_g)` 在 `WRITE_ALPHA=0.125` 达到 `38.2958 / 8.5996`，相比 v4 locked baseline `38.3706 / 8.6694` 改善 `0.0748m / 0.0698deg`。
2. 这个提升不是用 segment 损伤换来的：50/100/200-frame mean ATE 从 `33.259 / 33.769 / 33.735` 改善到 `33.177 / 33.685 / 33.628`，YawRMSE 从 `5.227` 改善到 `5.156`。
3. FinalErr `3.385m` 与 baseline `3.403m` 基本持平，虽然不如 T1 `stage_d*sqrt` a0.100 的 `3.095m`，但没有出现 endpoint regression。
4. `sqrt(1-D_g)` 单独作为 write score 只有在 `alpha=0.125` 才小幅超过 baseline，ATE=`38.3608`，但 FinalErr 变差到 `3.503m`，因此只作为次级候选。
5. `WRITE_ALPHA=0.05/0.075` 普遍偏弱，尤其 `stage_d*sqrt` a0.050 把 ATE 拉回 `38.5818`；这说明 Dg-modulated write 不是越保守越好，需要足够写入强度。
6. H3 在 KITTI01 上已经成立：Dg-based write eligibility 通过 `stage_d*sqrt(1-D_g)` 超过 stage_d baseline，并且达到 `ATE <= 38.30` 的 v5 H3 gate。

### 4.5 T2 后续决策

进入 T3 的候选：

| Candidate | WRITE_ALPHA | 进入原因 |
|---|---:|---|
| `stage_d*sqrt(1-D_g)` | 0.125 | 当前 v5 KITTI01 best；ATE/Rot/segment/Yaw 同步最好 |
| `sqrt(1-D_g)` | 0.125 | 次级候选；小幅超过 stage_d baseline，但 FinalErr 较差 |

T3 要补：

```text
read beta = 3.25 / 4.25
write alpha = 0.125
hybrid candidates = stage_d*sqrt(1-D_g), sqrt(1-D_g)
read-only anchors = C23 past read-only beta 3.25 / 4.25
```

目标是确认 T2 新 write policy 不是只适配 beta `3.75` 的单点，并计算同 beta write gain。

---

## 5. 当前 v5 状态更新

### 5.1 成功标准更新

用户在 `2026-05-06` 明确更新最终成功标准：

```text
KITTI01 ATE RMSE < 30m
```

因此，当前 T2 的 `38.2958 / 8.5996` 只能视为相对 v4 locked baseline 的阶段性改进，不能视为 v5 最终成功。后续所有 promotion / conclusion 需要分两层写：

```text
relative improvement:
    是否超过当前 locked baseline 或前一阶段 best

final success:
    KITTI01 ATE RMSE 是否低于 30m
```

补充执行原则：

```text
如果 TTT writing policy 系列，包括 D_g eligibility / stage_d 组合 / focal-style 写入，仍不能把 KITTI01 ATE 降到 30m 以下，则不继续在同一写入空间做大规模细扫。
后续应主动转向其他可能产生数量级改进的方向：
    1. read intervention 本身的强度 / 路径 / 层选择；
    2. per-head 或 head-group cue，避免 headmean 稀释；
    3. support / layer window 的重新组合；
    4. read cue 与已有强 cue 的最小非线性组合；
    5. 必要时重新检查 trajectory scale / chunk boundary / evaluation failure mode。
```

已完成：

- fast cue eval 工程 smoke。
- `D_g_locked` TTT write-score T1 小矩阵。
- T1 trajectory diagnostics。
- T2 write strength sweep。
- T2 trajectory diagnostics。

当前 v5 KITTI01 best：

```text
D_g_locked = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
beta = 3.75
commit = probe_ttt_write
write score = stage_d*sqrt(1-D_g)
WRITE_ALPHA = 0.125
KITTI01 = 38.2958 / 8.5996
FinalErr = 3.385
YawRMSE = 5.156
```

暂未执行：

- T3 read beta / write policy decoupling。
- H1 passive interpretation dashboard。
- H2 per-head passive cache / read-only gate。
- H5 SWA FA+SWA local-memory matrix。

---

## 6. T3：read beta / write policy decoupling

### 6.1 固定协议

```text
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
mode = KITTI01 full
FAST_CUE_EVAL = 1
write alpha = 0.125
```

T3 补两个 read beta：

```text
beta = 3.25 / 4.25
read-only anchor = probe_native
hybrid candidates = sqrt(1-D_g), stage_d*sqrt(1-D_g)
```

资源记录：

- 6 并发运行，GPU 0-5。
- 开始：`2026-05-06 04:25:49`。
- 完成时间范围：`04:47:53 -> 04:51:10`。
- 6 个 full run 总 walltime 约 `25.4 min`。
- 自动 benchmark 首次显示 `no estimated poses found`，但轨迹文件已正常保存；手动重跑 benchmark 后指标正常。该问题只影响评估日志生成，不影响 full run 输出。

### 6.2 T3 benchmark 指标

| Run | Mode / write | Beta | ATE RMSE | Rot RMSE | RPE t | RPE r | Write mean | Mean D>0.5 | Write gain ATE | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T3-01 | read-only | 3.25 | `38.6572` | `8.6654` | `92.3944` | `0.0086` | n/a | 0.190370 | n/a | read anchor |
| T3-02 | read-only | 4.25 | `38.6208` | `8.6868` | `92.3944` | `0.0086` | n/a | 0.190370 | n/a | read anchor |
| T3-03 | `sqrt(1-D_g)` | 3.25 | `38.3672` | `8.6390` | `92.3936` | `0.0086` | 0.846472 | 0.190370 | `0.2900` | stable gain |
| T3-04 | `sqrt(1-D_g)` | 4.25 | `38.3276` | `8.6719` | `92.3933` | `0.0087` | 0.846472 | 0.190370 | `0.2932` | stable gain |
| T3-05 | `stage_d*sqrt(1-D_g)` | 3.25 | `38.2992` | `8.5704` | `92.3927` | `0.0086` | 0.420704 | 0.190384 | `0.3580` | strong relative improvement |
| T3-06 | `stage_d*sqrt(1-D_g)` | 4.25 | `38.2658` | `8.6024` | `92.3927` | `0.0086` | 0.420704 | 0.190384 | `0.3550` | **current v5 best, but not final success** |

### 6.3 T3 trajectory diagnostics

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `stage_d` b3.75 baseline | `38.3706` | `3.403` | `33.259` | `33.769` | `33.735` | `5.227` | `30.725774` | v4 locked baseline |
| T2 best b3.75 | `38.2958` | `3.385` | `33.177` | `33.685` | `33.628` | `5.156` | `30.717566` | previous v5 best |
| read b3.25 | `38.6572` | `3.073` | `33.560` | `34.069` | `34.010` | `5.201` | `30.739069` | endpoint 好，ATE 弱 |
| read b4.25 | `38.6208` | `2.957` | `33.543` | `34.049` | `33.995` | `5.216` | `30.738358` | endpoint 最好，ATE 弱 |
| `sqrt` b3.25 | `38.3672` | `3.559` | `33.231` | `33.742` | `33.687` | `5.205` | `30.728876` | ATE 接近 baseline |
| `sqrt` b4.25 | `38.3276` | `3.446` | `33.226` | `33.732` | `33.689` | `5.230` | `30.726507` | ATE 好于 baseline，Yaw 无改善 |
| `stage_d*sqrt` b3.25 | `38.2992` | `3.422` | `33.162` | `33.673` | `33.610` | `5.134` | `30.718174` | Yaw 最好 |
| `stage_d*sqrt` b4.25 | `38.2658` | `3.301` | `33.161` | `33.667` | `33.616` | `5.163` | `30.716463` | **ATE / segment best** |

### 6.4 T3 结论

1. `stage_d*sqrt(1-D_g)` 的提升不是 beta `3.75` 单点：beta `3.25` 和 `4.25` 都稳定优于各自 read-only anchor，write gain 约 `0.355-0.358m`。
2. 当前 v5 relative best 更新为 `stage_d*sqrt(1-D_g), beta=4.25, alpha=0.125`，KITTI01 `38.2658 / 8.6024`。
3. 相比 v4 locked baseline `38.3706 / 8.6694`，当前 best 改善 `0.1048m / 0.0670deg`；50/100/200-frame mean ATE 也同步改善。
4. 但按用户更新的最终成功标准，`38.2658m` 仍远高于 `KITTI01 ATE < 30m`，所以 TTT writing policy 虽然相对成立，但没有达到最终实验目标。
5. 继续只围绕 `stage_d*sqrt` 做细扫不太可能带来 `8m+` 级别跃迁；下一步按用户要求增加 focal-style TTT write gate，然后主动转向非 TTT writing 方向。

### 6.5 Focal-style TTT writing 补充计划

用户要求额外尝试 focal loss 思路探索 TTT 写入方式。先做 3 个最小 gate：

| Candidate | 定义 | 直觉 |
|---|---|---|
| `stage_d_x_static_focal2` | `stage_d * (1-D_g)^2` | 强调高置信 static token，弱化 high-D harmful token |
| `stage_d_x_static_focal4` | `stage_d * (1-D_g)^4` | 更强 static focal，测试过度保守是否有利 |
| `stage_d_x_dg_boundary_focal2` | `stage_d * (1-D_g) * D_g^2` | 类 focal hard-example，关注 D_g 边界/难例 token |

固定：

```text
read cue = C23 past
beta = 4.25
WRITE_ALPHA = 0.125
commit = probe_ttt_write
```

若 focal-style 仍不能接近 `30m`，TTT writing 分支不再扩大矩阵，转向 read intervention / per-head / support-layer 方向。

---

## 7. Focal-style TTT writing gate

### 7.1 固定协议

```text
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
mode = KITTI01 full / hybrid
commit = probe_ttt_write
READ_LAYER_MODE = early
beta = 4.25
WRITE_ALPHA = 0.125
FAST_CUE_EVAL = 1
```

资源记录：

- 与 read-intervention gate 并行执行，6 个 full run 同批调度。
- focal 三个 run 完成时间：`2026-05-06 05:17:20 -> 05:18:41`。
- 自动 benchmark 因相对路径问题首次显示 `no estimated poses found`；轨迹文件完整，使用绝对路径重跑 evaluator 后指标有效。

### 7.2 Focal-style benchmark 指标

| Run | Write score | ATE RMSE | Rot RMSE | RPE t | RPE r | Write mean | Mean D>0.5 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| T3 best reference | `stage_d*sqrt(1-D_g)` | `38.2658` | `8.6024` | `92.3927` | `0.0086` | 0.420704 | 0.190384 | v5 TTT reference |
| ACL2V5_TF_01 | `stage_d*(1-D_g)^2` | `38.3447` | `8.6079` | `92.3926` | `0.0086` | 0.354552 | 0.190380 | endpoint/yaw 略好，但 ATE 回退 |
| ACL2V5_TF_02 | `stage_d*(1-D_g)^4` | `38.3824` | `8.6180` | `92.3938` | `0.0086` | 0.321195 | 0.190379 | 过度保守，ATE 继续回退 |
| ACL2V5_TF_03 | `stage_d*(1-D_g)*D_g^2` | `38.4861` | `8.7614` | `92.3939` | `0.0087` | 0.015284 | 0.190364 | boundary/hard-example 写入失败 |

### 7.3 Focal-style trajectory diagnostics

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| T3 best reference | `38.2658` | `3.301` | `33.161` | `33.667` | `33.616` | `5.163` | `30.716463` |
| focal static2 | `38.3447` | `2.922` | `33.258` | `33.763` | `33.725` | `5.147` | `30.717907` |
| focal static4 | `38.3824` | `3.016` | `33.290` | `33.793` | `33.754` | `5.179` | `30.729442` |
| focal boundary2 | `38.4861` | `3.098` | `33.413` | `33.917` | `33.881` | `5.282` | `30.730721` |

### 7.4 Focal-style 结论

1. focal-style TTT writing 没有达到最终成功标准 `KITTI01 ATE < 30m`，也没有超过 T3 best `38.2658m`。
2. `stage_d*(1-D_g)^2` 能改善 FinalErr 和 Yaw，但 ATE 回退 `0.0789m`，说明更强 static focal 可能减少 endpoint drift，却不足以改变全局 RMSE 主误差。
3. `stage_d*(1-D_g)^4` 进一步降低 write mean 到 `0.321195`，ATE 继续变差，过度保守不成立。
4. `stage_d*(1-D_g)*D_g^2` 的 write mean 只有 `0.015284`，几乎把 TTT 写入关掉，ATE/Rot 明显失败；hard-example/boundary 写入不是当前方向。
5. 结论：按用户要求，TTT writing policy 线已证明不能接近 `<30m`；后续不再扩大 TTT 写入细扫，主动转向 read intervention / per-head / support-window / evaluation-error source。

---

## 8. 非 TTT writing 主动探索 A：read intervention gate

### 8.1 固定协议

```text
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
mode = KITTI01 full / hybrid
commit = probe_ttt_write
write score = stage_d*sqrt(1-D_g)
WRITE_ALPHA = 0.125
beta = 4.25
FAST_CUE_EVAL = 1
```

本批只改 read intervention：

| Run | Intervention | ATE RMSE | Rot RMSE | RPE t | RPE r | Write mean | Mean D>0.5 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| T3 best reference | pair / early | `38.2658` | `8.6024` | `92.3927` | `0.0086` | 0.420704 | 0.190384 | previous best |
| ACL2V5_NR_01 | pair / early_half | `38.7572` | `8.0062` | `92.4096` | `0.0085` | 0.420704 | 0.190384 | Rot/Yaw 强，但 ATE 失败 |
| ACL2V5_NR_02 | pair / all | `36.7035` | `6.4052` | `92.4333` | `0.0078` | 0.420704 | 0.190384 | **large jump; active lead** |
| ACL2V5_NR_03 | key / early | `39.1577` | `9.1990` | `92.3868` | `0.0090` | 0.420704 | 0.190384 | key bias 失败 |

### 8.2 Trajectory diagnostics

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| T3 best reference | `38.2658` | `3.301` | `33.161` | `33.667` | `33.616` | `5.163` | `30.716463` |
| pair / early_half | `38.7572` | `4.162` | `32.977` | `33.553` | `33.483` | `4.712` | `30.899074` |
| pair / all | `36.7035` | `5.312` | `29.900` | `30.534` | `30.532` | `3.599` | `31.122736` |
| key / early | `39.1577` | `3.196` | `34.361` | `34.839` | `34.783` | `5.594` | `30.663529` |

### 8.3 Read intervention gate 结论

1. `pair / all` 是 v5 第一条大幅改善信号：KITTI01 从 T3 best `38.2658` 降到 `36.7035`，同时 Rot 从 `8.6024` 降到 `6.4052`，Yaw 从 `5.163` 降到 `3.599`。
2. `pair / all` 仍没有达到用户定义的成功标准 `ATE < 30m`，但它已经把 50-frame mean ATE 降到 `29.900`，说明 read intervention 覆盖层比 TTT writing policy 更接近数量级突破。
3. `pair / early_half` rotation/yaw 明显改善但全局 ATE 变差，说明只扩大到 early_half 会改变 orientation，但不足以修全局漂移。
4. `key / early` 明确失败；当前强信号仍是 pair bias，而不是 key-only bias。
5. 下一步立即围绕 `pair / all` 做小矩阵：补 read-only anchor、C23 beta 外推、C24 shadow，而不是继续 TTT writing。

---

## 9. 非 TTT writing 主动探索 B：pair/all 小矩阵

### 9.1 固定协议

```text
mode = KITTI01 full
READ_LAYER_MODE = all
FRAME_BIAS_MODE = pair
write score = stage_d*sqrt(1-D_g)
WRITE_ALPHA = 0.125
FAST_CUE_EVAL = 1
```

资源记录：

- 6 并发运行，GPU 0-5。
- 开始：`2026-05-06 05:25:59`。
- 完成：`05:49:15 -> 05:50:20`。
- walltime 约 `24.4 min`，6 并发仍可控；最低 host RAM available 约 `148GiB`。

### 9.2 Benchmark 指标

| Run | Cue | Mode | Beta | ATE RMSE | Rot RMSE | RPE t | RPE r | Write mean | Mean D>0.5 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ACL2V5_PA_01 | C23 past | read-only | 4.25 | `36.8807` | `6.4322` | `92.4349` | `0.0077` | n/a | 0.190370 | pair/all read-only anchor |
| ACL2V5_PA_02 | C23 past | hybrid | 3.75 | `36.7223` | `6.4458` | `92.4300` | `0.0077` | 0.420704 | 0.190384 | strong |
| ACL2V5_NR_02 | C23 past | hybrid | 4.25 | `36.7035` | `6.4052` | `92.4333` | `0.0078` | 0.420704 | 0.190384 | previous pair/all lead |
| ACL2V5_PA_03 | C23 past | hybrid | 4.75 | `36.6803` | `6.3855` | `92.4356` | `0.0078` | 0.420704 | 0.190384 | **current pair/all best** |
| ACL2V5_PA_04 | C23 past | hybrid | 5.25 | `36.6883` | `6.3506` | `92.4386` | `0.0078` | 0.420704 | 0.190384 | Rot/Yaw 更好，ATE 略回退 |
| ACL2V5_PA_05 | C24 past | hybrid | 4.00 | `37.3017` | `6.0605` | `92.4483` | `0.0076` | 0.422847 | 0.194833 | Rot 强，ATE 不如 C23 |
| ACL2V5_PA_06 | C24 past | hybrid | 4.50 | `37.2757` | `6.0611` | `92.4503` | `0.0077` | 0.422847 | 0.194833 | C24 shadow 不适合 KITTI01 ATE |

### 9.3 Trajectory diagnostics

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| C23 pair/all read b4.25 | `36.8806` | `4.943` | `30.115` | `30.757` | `30.785` | `3.619` | `31.143263` |
| C23 pair/all b3.75 | `36.7223` | `5.353` | `29.953` | `30.587` | `30.572` | `3.645` | `31.092788` |
| C23 pair/all b4.25 | `36.7035` | `5.312` | `29.900` | `30.534` | `30.532` | `3.599` | `31.122736` |
| C23 pair/all b4.75 | `36.6803` | `5.051` | `29.870` | `30.506` | `30.521` | `3.567` | `31.148976` |
| C23 pair/all b5.25 | `36.6883` | `4.927` | `29.852` | `30.492` | `30.527` | `3.528` | `31.182787` |
| C24 pair/all b4.00 | `37.3017` | `3.933` | `30.624` | `31.214` | `31.332` | `3.130` | `31.278774` |
| C24 pair/all b4.50 | `37.2757` | `4.126` | `30.625` | `31.210` | `31.346` | `3.121` | `31.305454` |

### 9.4 pair/all 结论

1. `pair/all` 的核心收益来自 read intervention 本身：C23 read-only b4.25 已达 `36.8807 / 6.4322`，比 early-layer T3 best `38.2658 / 8.6024` 有数量级更大的提升。
2. TTT safe write 在 `pair/all` 上仍有正 gain，但比 early-layer 小：b4.25 read-only `36.8807` -> hybrid `36.7035`，ATE gain `0.1772m`。
3. C23 beta 最优暂在 `4.75`：`36.6803 / 6.3855`。beta `5.25` 的 Rot/Yaw 更好，但 ATE 已略回退。
4. C24 shadow 的 Rot/Yaw 更干净，但 ATE 明显弱于 C23，说明 v4 的 cross-seq locked 判断仍更适合作主线。
5. 仍未达到最终成功标准 `KITTI01 ATE < 30m`。下一步继续沿 `pair/all` 主线做更强 read policy：高 beta 外推、`protected_pair`、`query/all` 与写入 policy 对照。

---

## 10. 非 TTT writing 主动探索 C：pair/all policy-B

### 10.1 固定协议

```text
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
mode = KITTI01 full / hybrid
commit = probe_ttt_write
READ_LAYER_MODE = all
WRITE_ALPHA = 0.125
FAST_CUE_EVAL = 1
```

资源记录：

- 6 并发运行，GPU 0-5。
- 开始：`2026-05-06 05:51:23`。
- 完成：`06:14:45 -> 06:15:47`。
- walltime 约 `24.4 min`；最低 host RAM available 约 `126GiB`。

### 10.2 Benchmark 指标

| Run | Bias | Beta | Write score | ATE RMSE | Rot RMSE | RPE t | RPE r | Write mean | 结论 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| PA best reference | pair | 4.75 | `stage_d*sqrt(1-D_g)` | `36.6803` | `6.3855` | `92.4356` | `0.0078` | 0.420704 | active reference |
| ACL2V5_PB_01 | pair | 5.75 | `stage_d*sqrt(1-D_g)` | `36.6942` | `6.3298` | `92.4415` | `0.0078` | 0.420704 | Rot 更好，ATE 回退 |
| ACL2V5_PB_02 | pair | 6.25 | `stage_d*sqrt(1-D_g)` | `36.7034` | `6.3259` | `92.4440` | `0.0078` | 0.420704 | ATE 回退 |
| ACL2V5_PB_03 | pair | 7.00 | `stage_d*sqrt(1-D_g)` | `36.7632` | `6.3156` | `92.4479` | `0.0079` | 0.420704 | 过强，ATE 明显回退 |
| ACL2V5_PB_04 | pair | 4.75 | `stage_d` | `36.7612` | `6.4020` | `92.4376` | `0.0078` | n/a | 不如 `stage_d*sqrt` |
| ACL2V5_PB_05 | protected_pair | 4.75 | `stage_d*sqrt(1-D_g)` | `36.6803` | `6.3855` | `92.4356` | `0.0078` | 0.420704 | 与 pair 等价 |
| ACL2V5_PB_06 | query | 4.75 | `stage_d*sqrt(1-D_g)` | `79.9402` | `15.5180` | `93.3475` | `0.0259` | 0.420704 | query/all 崩溃 |

### 10.3 Trajectory diagnostics

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| PB b5.75 | `36.6942` | `4.773` | `29.844` | `30.486` | `30.536` | `3.504` | `31.209866` |
| PB b6.25 | `36.7034` | `4.583` | `29.852` | `30.497` | `30.561` | `3.489` | `31.237243` |
| PB b7.00 | `36.7632` | `4.421` | `29.905` | `30.554` | `30.637` | `3.472` | `31.275248` |
| PB stageD b4.75 | `36.7612` | `4.950` | `29.964` | `30.605` | `30.649` | `3.576` | `31.164446` |
| PB protected b4.75 | `36.6803` | `5.051` | `29.870` | `30.506` | `30.521` | `3.567` | `31.148976` |
| PB query b4.75 | `79.9402` | `99.149` | `74.044` | `75.163` | `78.513` | `9.313` | `44.843560` |

### 10.4 PB 结论

1. C23 `pair/all` 的 beta sweet spot 已经很窄，`4.75` 仍是当前最好 ATE；继续增大 beta 只改善 Rot/Yaw，ATE 变差。
2. `stage_d*sqrt(1-D_g)` 仍优于原始 `stage_d`，说明 TTT 写入 eligibility 在 pair/all 上还有小幅价值。
3. `protected_pair/all` 与 `pair/all` 完全等价，说明当前保护逻辑没有改变这个路径。
4. `query/all` 完全崩溃，说明 all-layer intervention 只能用于 pair bias；query-only 会破坏尺度/轨迹，不能继续。
5. 仍未达到最终成功标准 `<30m`。下一步按用户新想法，写补充计划并测试 explicit dyn cue 对 TTT 写入的动态区域 veto。

---

## 11. explicit dyn TTT write-veto 补充启动

用户提出：

```text
结合 explicit dyn cue 让 TTT 少写入动态区域。
```

补充计划文件：

```text
docs/ACL2_v5_ExplicitDyn_TTT_WriteVeto_补充实验计划.md
```

本轮实现新增 write score：

| Write score | 定义 |
|---|---|
| `explicit_dyn_inv` | `1 - explicit_dyn` |
| `stage_d_x_exp_inv` | `stage_d * (1 - explicit_dyn)` |
| `stage_d_x_exp_inv_sqrt` | `stage_d * sqrt(1 - explicit_dyn)` |
| `stage_d_x_exp_inv_sq` | `stage_d * (1 - explicit_dyn)^2` |
| `stage_d_x_dg_exp_inv_sqrt` | `stage_d * sqrt(1 - D_g) * sqrt(1 - explicit_dyn)` |
| `stage_d_x_union_dyn_inv` | `stage_d * (1 - max(D_g, explicit_dyn))` |

固定第一批 gate：

```text
read cue = C23 past
read intervention = pair/all
beta = 4.75
WRITE_ALPHA = 0.125
success target = KITTI01 ATE < 30m
promotion local reference = 36.6803m
```

### 11.1 实现与资源

代码改动：

- `loger/pipeline/hybrid_memory_controller.py`：新增 explicit dyn write source，并在 fast cue path 中计算 `explicit_dyn_patch`。
- `run_pipeline_abc_v2.py`：新增 write score choices，并在 debug row 中输出 `prior_hmc_write_corr_score_exp_dyn`。
- `docs/ACL2_v5_ExplicitDyn_TTT_WriteVeto_补充实验计划.md`：记录本补充实验的假设、gate 和候选矩阵。

验证：

```text
python -m py_compile loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py
bash -n tools/run_attention_cue_experiment.sh
```

运行资源：

- 5 并发，GPU 0-4。
- 开始：`2026-05-06 06:19:50`。
- 完成：`06:43:13 -> 06:44:31`。
- walltime 约 `24.7 min`；host RAM 充足，无 OOM / swap 风险。

### 11.2 Benchmark 指标

固定协议：

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
WRITE_ALPHA = 0.125
FAST_CUE_EVAL = 1
```

| Run | Write score | ATE RMSE | Rot RMSE | RPE t | RPE r | Write mean | Corr score-expDyn | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA reference | `stage_d*sqrt(1-D_g)` | `36.6803` | `6.3855` | `92.4356` | `0.0078` | 0.420704 | n/a | active reference |
| ACL2V5_EXPV_01 | `stage_d*(1-E)` | `36.7822` | `6.4081` | `92.4366` | `0.0078` | 0.398852 | -0.930337 | ATE 回退 |
| ACL2V5_EXPV_02 | `stage_d*sqrt(1-E)` | `36.7173` | `6.3990` | `92.4370` | `0.0078` | 0.435735 | -0.924826 | 接近但未超过 reference |
| ACL2V5_EXPV_03 | `stage_d*(1-E)^2` | `36.7931` | `6.4479` | `92.4395` | `0.0078` | 0.360436 | -0.918621 | veto 过强，回退 |
| ACL2V5_EXPV_04 | `stage_d*sqrt(1-D_g)*sqrt(1-E)` | `36.7168` | `6.3915` | `92.4372` | `0.0078` | 0.366366 | -0.809437 | 未超过 reference |
| ACL2V5_EXPV_05 | `stage_d*(1-max(D_g,E))` | `36.7635` | `6.3672` | `92.4375` | `0.0078` | 0.320550 | -0.787473 | Rot 略好但 ATE 回退 |

### 11.3 Trajectory diagnostics

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| PA reference | `36.6803` | `5.051` | `29.870` | `30.506` | `30.521` | `3.567` | `31.148976` |
| EXPV-01 exp inv | `36.7822` | `5.150` | `29.993` | `30.636` | `30.676` | `3.581` | `31.162016` |
| EXPV-02 exp inv sqrt | `36.7173` | `4.962` | `29.947` | `30.587` | `30.620` | `3.591` | `31.162067` |
| EXPV-03 exp inv sq | `36.7932` | `5.186` | `30.006` | `30.648` | `30.687` | `3.636` | `31.190834` |
| EXPV-04 dg+exp inv sqrt | `36.7168` | `5.147` | `29.891` | `30.528` | `30.546` | `3.578` | `31.165937` |
| EXPV-05 union dyn inv | `36.7635` | `4.828` | `29.964` | `30.603` | `30.653` | `3.553` | `31.168835` |

### 11.4 explicit dyn write-veto 结论

1. explicit dyn write-veto 实现有效：write score 与 explicit dyn 的相关性明显为负，直接 veto 的 `Corr score-expDyn` 约 `-0.93`。
2. 但该思路没有改善 KITTI01 ATE。最好的 explicit-dyn 版本是 `stage_d*sqrt(1-D_g)*sqrt(1-E)`，ATE=`36.7168`，仍比 active reference `36.6803` 差 `0.0365m`。
3. `stage_d*(1-max(D_g,E))` 的 Rot/Yaw/FinalErr 略有优势，但 ATE 明显回退，因此不能作为主线。
4. 这不能证明“动态区域少写”方向错误，只能说明 `explicit_dyn` 单 map 和连续乘法 veto 不够准、不够强；后续应测试更直接的 sparse / hard write gate。
5. 仍未达到成功标准 `KITTI01 ATE < 30m`。下一步把 TTT writing 与 SWA writing 作为最高优先级，不再浅尝辄止。

---

## 12. 非 TTT writing 主动探索 D：C23 pair/all read mask 稀疏化

### 12.1 动机

截至 explicit dyn write-veto 结束，KITTI01 active best 仍是：

```text
C23 past + pair/all + beta 4.75 + stage_d_x_dg_inv_sqrt
ATE / Rot = 36.6803 / 6.3855
```

TTT write-side 的改动只带来 `~0.1m` 量级变化，达不到 `<30m` 所需的大跨越；而 `early` -> `all` read intervention 带来了 `~1.6m` 级收益。因此本轮不再新增 cue，而是在同一个 `D_g_locked` 上改变 read mask 的稀疏度：

- `read_topk_frac`：每帧只保留最高响应 patch，减少低置信动态区域的 read bias。
- `read_calib_mode=per_frame_quantile`：按每帧分位数把 read cue 重新校准到固定 target mass。

这组实验只复用已有 `C23 past` map，不额外计算新 cue，符合“减少不必要 cue 计算”的约束。

### 12.2 固定协议

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
FAST_CUE_EVAL = 1
success target = KITTI01 ATE < 30m
local reference = 36.6803m
```

第一批候选：

| Run | Read postprocess | 参数 | 目的 |
|---|---|---|---|
| ACL2V5_RDSP_01 | topk | `READ_TOPK_FRAC=0.08` | 强稀疏 |
| ACL2V5_RDSP_02 | topk | `READ_TOPK_FRAC=0.12` | 中强稀疏 |
| ACL2V5_RDSP_03 | topk | `READ_TOPK_FRAC=0.16` | 接近当前 D mass |
| ACL2V5_RDSP_04 | quantile calib | `target_mass=0.08, blend=1.0` | 软稀疏 |
| ACL2V5_RDSP_05 | quantile calib | `target_mass=0.12, blend=1.0` | 软中稀疏 |
| ACL2V5_RDSP_06 | quantile calib | `target_mass=0.16, blend=1.0` | 软接近 reference |

用户进一步指出：`让 TTT 少写入动态区域` 这个思路本身未必错，关键在于“怎么写入”和“动态区域是否准确”。因此本 RDSP read-mask 稀疏化批次暂缓，优先回到 TTT write path，直接测试真正减少动态区域写入的 sparse-write 策略。

---

## 13. TTT 动态区域少写：sparse static write gate

### 13.1 修正后的判断

前面的 `stage_d*sqrt(1-D_g)`、focal 和 explicit dyn veto，大多只是改变 write score 的排序；在默认 `hmc_write_sparse_ratio=1.0` 下，最终写入 prior 仍被压到 `0.8-1.2` 的窄范围，并没有真正禁止动态区域写入。

因此“少写动态区域”的更直接测试应该是：

- 用 `D_g_locked` 或 `explicit dyn` 参与 static eligibility ranking；
- 只允许排名靠前的一部分 token 写入；
- 对排名靠后的疑似动态 token 置零或强削弱，而不是只给一个温和的 percentile penalty。

### 13.2 固定协议

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
WRITE_ALPHA = 0.125
FAST_CUE_EVAL = 1
success target = KITTI01 ATE < 30m
local reference = 36.6803m
```

第一批 sparse-write gate：

| Run | Write score | Sparse ratio | Sparse mode | 目的 |
|---|---|---:|---|---|
| ACL2V5_WRSP_01 | `stage_d_x_dg_inv_sqrt` | 0.75 | soft | 底部 25% token 不写，保留 selected token 的 soft prior |
| ACL2V5_WRSP_02 | `stage_d_x_dg_inv_sqrt` | 0.50 | soft | 只写 top 50% 高置信静态 token |
| ACL2V5_WRSP_03 | `stage_d_x_dg_inv_sqrt` | 0.25 | soft | 强 sparsify，检验是否过度保守 |
| ACL2V5_WRSP_04 | `stage_d_x_dg_inv_sqrt` | 0.50 | hard | top 50% 二值写入，检验 soft prior 是否仍太弱 |
| ACL2V5_WRSP_05 | `stage_d_x_dg_exp_inv_sqrt` | 0.50 | soft | D_g 与 explicit dyn 双 map 共同排名 |
| ACL2V5_WRSP_06 | `stage_d_x_union_dyn_inv` | 0.50 | soft | 只写同时避开 D_g / explicit dyn 的 token |

### 13.3 第一批 sparse-write 结果

资源/时间：

- 6 并发 full KITTI01：GPU 0-5，`06:51:07 -> 07:15`，约 `24 min` 完成 6 个 full run。
- 首次 benchmark 因 `ATTN_CUE_BASE=results/...` 是相对路径，在 `eval/long_eval_script` 子目录下找不到 `01.txt`；已用绝对路径重跑 benchmark，并修正 `tools/run_attention_cue_experiment.sh` 后续自动转绝对路径。

| Run | Write score | Sparse ratio | Mode | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---|---:|---:|---:|---:|---|
| reference | `stage_d_x_dg_inv_sqrt` | 1.00 | none | `36.6803` | `6.3855` | `92.4356` | `0.0078` | v5 当前局部 reference |
| ACL2V5_WRSP_01 | `stage_d_x_dg_inv_sqrt` | 0.75 | soft | `36.6749` | `6.0432` | `92.4311` | `0.0075` | ATE 微幅新低，Rot 明显改善 |
| ACL2V5_WRSP_02 | `stage_d_x_dg_inv_sqrt` | 0.50 | soft | `36.7561` | `5.7937` | `92.4298` | `0.0074` | Rot 更好，但 ATE 回退 |
| ACL2V5_WRSP_03 | `stage_d_x_dg_inv_sqrt` | 0.25 | soft | `37.2803` | `5.7044` | `92.4363` | `0.0078` | 过度稀疏，ATE 明显损伤 |
| ACL2V5_WRSP_04 | `stage_d_x_dg_inv_sqrt` | 0.50 | hard | `36.7729` | `5.7713` | `92.4288` | `0.0074` | hard 写入更像 rotation diagnostic，不适合主线 |
| ACL2V5_WRSP_05 | `stage_d_x_dg_exp_inv_sqrt` | 0.50 | soft | `36.8425` | `5.8317` | `92.4303` | `0.0074` | explicit dyn 加入 ranking 后 ATE 变差 |
| ACL2V5_WRSP_06 | `stage_d_x_union_dyn_inv` | 0.50 | soft | `36.6859` | `5.8383` | `92.4268` | `0.0074` | 接近 reference，Rot 改善，但未超过 WRSP_01 |

### 13.4 Sparse-write 结论

1. 真正减少 TTT 动态区域写入后，rotation 有稳定收益：Rot 从 `6.3855` 降到 `6.04/5.79/5.77/5.84` 一带。
2. ATE 没有同步大幅下降，最好 `WRSP_01 = 36.6749m` 只比 reference 好 `0.0054m`，距离 `<30m` 仍差很远。
3. `0.25/0.50` 的强稀疏更像“牺牲 translation/scale 换 rotation”，说明 TTT 写入不能只简单少写；还要区分哪些 token 是 trajectory-continuity 必需的静态 token。
4. explicit dyn 单独/union 加入 sparse ranking 没有帮助，支持之前判断：explicit dyn 的连续 map 可能不够准，或者它和 `D_g_locked` 的动态定义不在同一类错误上。
5. 下一步不再继续只扫 sparse ratio；优先探索 SWA 写入策略，以及 SWA-write + TTT sparse 的组合。若组合仍不达标，再进一步放开 `mp_min/mp_max`、branch/layer 写入策略和更结构化的 static-token selection。

---

## 14. 写入侧最高优先级：SWA-write 第一批 gate

### 14.1 动机

用户要求后续把 TTT 写入策略与 SWA 写入策略作为最高优先级，不要浅尝辄止。这里先记录第一批已经完成的 SWA-write full gate。

这批不是 v4 的 SWA-read gate，而是在每个 chunk commit 前，对将进入下一 chunk 的 SWA `history` 做写入侧削弱：

```text
gate = clip(1 - rho * D_g, min_gate, 1)
mode = v / kv
sparse_ratio < 1 时，只保留 static-score top-k 的 SWA source token，其余压到 min_gate
```

固定协议：

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
FAST_CUE_EVAL = 1
local reference = WRSP_01 = 36.6749 / 6.0432
success target = KITTI01 ATE < 30m
```

### 20.54 计划：TTT projected anti-dynamic commit（停止 SWA，聚焦 TTT 写入）

用户的新判断：动态区域可能对当前 chunk 的 TTT 自适应仍然有用，直接在 TTT 写入时压制动态 token 会让当前 chunk 缺必要信息；真正的问题可能是这些动态 residual 被长期传到下一个 chunk。因此本轮不再继续 SWA，也不再做 reset 改动，固定 `RESET_EVERY=5` 与 LoGeR 对齐。

20.53 的 one-hop TTL 已验证“忘动态”能改善 Rot / FinalErr / Yaw，但 ATE 回退，说明在 fast-weight 空间直接扣 `full - filtered` 太粗。新的写法改为 projection：

```text
static_delta  = W_filtered_static - W_old
dynamic_delta = W_full - W_filtered_static
aligned_dyn   = positive_projection(dynamic_delta, static_delta)
anti_dyn      = dynamic_delta - aligned_dyn
W_commit      = W_full - alpha * anti_dyn
```

含义：

- 当前 chunk 仍然用 full TTT replay 的 dynamic 信息；
- 跨 chunk commit 时保留与 static update 同向的 dynamic residual；
- 只削弱不支持 static update 的 anti / orthogonal dynamic residual；
- `alpha=0` 等于 `TTEX_01` full commit，`alpha=1` 是最强 anti-dynamic projection。

实现：

- 新增 `TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE=project_anti_dynamic`；
- `run_pipeline_abc_v2.py` CLI 已接线；
- `tools/run_attention_cue_experiment.sh` 复用已有 `TTT_WRITE_REPLAY_TOKEN_FILTER_*` 环境变量；
- SWA 全部关闭：`ENABLE_SWA_WRITE_CONTROL=0`、`ENABLE_SWA_OVERLAP_SOURCE_GATE=0`、`ENABLE_SWA_OVERLAP_SOURCE_REPLACE=0`、`ENABLE_SWA_OVERLAP_BIAS=0`。

固定协议：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
PRIOR_LAYER_MODE = late
RESET_EVERY = 5
SWA = off
success target = KITTI01 ATE < 30m
```

第一批小矩阵：

| Run | Filter source | Scope | Branch | Alpha | 目的 |
|---|---|---|---|---:|---|
| `ACL2V5_TTPROJ_01` | per-frame static topk `0.75` | full chunk | `w0` | `0.25` | 温和削掉 anti-dynamic residual，保留多数 full update |
| `ACL2V5_TTPROJ_02` | per-frame static topk `0.75` | full chunk | `w0` | `0.50` | 检查 projection 强度是否存在 ATE sweet spot |
| `ACL2V5_TTPROJ_03` | per-frame static topk `0.75` | full chunk | `w0,w2` | `0.25` | 测试 w2 是否只需要 anti-dynamic 过滤，而不是 hard forget |
| `ACL2V5_TTPROJ_04` | scoped static topk `0.75` | tail overlap | `w0` | `0.50` | 只在跨 chunk seam 做 projection，验证长期传播假设 |

Gate：

- 若达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若超过 `TTEX_01 = 36.5932 / 6.4327`，保留为新的 TTT-only base；
- 若只改善 Rot/Yaw/FinalErr 但 ATE 回退，则说明 fast-weight post-commit sanitation 仍然不足，下一步需要改 TTT replay loss / update target 本身。

#### 20.54 实验结果：TTT projected anti-dynamic commit

运行记录：

- smoke `ACL2V5_TTPROJ_SMOKE_C23pairall_fullTopk075_w0_projAnti_a025_late`：`2026-05-07 11:00:32 -> 11:03:38`，约 `3.1 min`。
- smoke debug 确认 `ttt_replay_token_filter_blend_mode=project_anti_dynamic`，`swa_write_enabled=False`，SWA overlap source gate / replace / bias 均未启用。
- full batch 4 并发 GPU0-3：`11:04:43 -> 11:28:25`，约 `23.7 min` 完成 4 个 KITTI01 full run。
- 固定 `RESET_EVERY=5`，未改 LoGeR 对齐的 reset 机制；本批全部停止 SWA，只验证 TTT 写入传播。

Global metrics：

| Run | 具体含义 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `TTEX_01` | TTT-only reference：native mix `w0=1.10` | `36.5932` | `6.4327` | n/a | n/a | 当前 TTT-only reference |
| `TTDTTL_02` | one-hop dynamic TTL，full topk `0.75`，subtract `0.5` | `36.7036` | `6.2600` | `92.4350` | `0.0077` | 上一批 TTL best |
| `ACL2V5_TTPROJ_01` | full per-frame static topk `0.75`，`w0`，project alpha `0.25` | `36.7076` | `6.3306` | `92.4362` | `0.0077` | ATE 回退 |
| `ACL2V5_TTPROJ_02` | full per-frame static topk `0.75`，`w0`，project alpha `0.50` | `36.6532` | `6.2574` | `92.4342` | `0.0077` | full projection best，但未过 TTEX |
| `ACL2V5_TTPROJ_03` | full per-frame static topk `0.75`，`w0,w2`，project alpha `0.25` | `36.6971` | `6.3037` | `92.4363` | `0.0077` | 加 `w2` 未带来 ATE gain |
| `ACL2V5_TTPROJ_04` | tail-overlap static topk `0.75`，`w0`，project alpha `0.50` | `36.6477` | `6.2472` | `92.4352` | `0.0076` | 本批 best；Rot/Final/Yaw 改善，但 ATE 仍回退 |

Trajectory diagnostics：

输出目录：

`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_projected_anti_dynamic/trajectory_diagnostics/`

| Run | ATE RMSE | Final error | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `TTEX_01` | `36.5932` | `5.120` | `29.809 / 78.815` | `30.445 / 78.750` | `30.519 / 57.753` | `3.596` | `31.219899` |
| `TTDTTL_02` | `36.7036` | `4.009` | `29.915 / 79.139` | `30.549 / 78.824` | `30.613 / 57.867` | `3.448` | `31.144162` |
| `TTPROJ_01` | `36.7076` | `4.863` | `29.896 / 79.137` | `30.531 / 78.847` | `30.558 / 57.884` | `3.525` | `31.159350` |
| `TTPROJ_02` | `36.6532` | `4.167` | `29.848 / 79.036` | `30.479 / 78.687` | `30.535 / 57.782` | `3.452` | `31.132732` |
| `TTPROJ_03` | `36.6971` | `4.357` | `29.911 / 78.787` | `30.547 / 78.592` | `30.599 / 57.724` | `3.494` | `31.156741` |
| `TTPROJ_04` | `36.6477` | `3.546` | `29.924 / 79.124` | `30.551 / 78.719` | `30.640 / 57.780` | `3.425` | `31.156886` |

20.54 结论：

1. 本批没有达到成功标准 `KITTI01 ATE < 30m`，也没有超过 TTT-only reference `TTEX_01 = 36.5932 / 6.4327`。
2. `project_anti_dynamic` 比 raw TTL 更温和：best `TTPROJ_04 = 36.6477 / 6.2472`，比 `TTDTTL_02 = 36.7036 / 6.2600` ATE/Rot 都略好；FinalErr 从 `5.120` 降到 `3.546`，Yaw 从 `3.596` 降到 `3.425`。
3. 但它仍是 orientation / endpoint regularizer，不是 ATE 突破。worst `[200,300)` / `[200,400)` 主灾区基本没有被修掉，说明主要 ATE bottleneck 不在“commit 侧遗忘 dynamic residual”这个层面。
4. `TTPROJ_02` 的 full projection 和 `TTPROJ_04` 的 tail projection 接近，tail 版本 Final/Yaw 更好但 segment mean 更差；这支持“跨 chunk seam 的长期传播确实有污染”，但该污染不是 KITTI01 ATE 主因。
5. 当前判断：TTT 写入策略不能继续只做 fast-weight post-commit sanitation。下一步应改 TTT replay objective / loss 权重本身，例如 focal-like token weighting：当前 chunk 允许 dynamic token 参与更新，但把更新目标按 high-confidence static / overlap-consistent / low-residual token 做重加权，而不是事后过滤 fast weights。

#### 20.49 实验结果：TTT overlap soft forget

工程记录：

- 新增 `TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND`，语义是把目标 branch 的 full replay fast weight 软拉向 filtered replay fast weight；
- smoke `ACL2V5_TTOVFB_SMOKE_C23pairall_tailTopk075_w0_b025` 通过，debug 确认 `ttt_replay_token_filter_branch_isolated=True`、`ttt_replay_token_filter_blend=0.25`、`swa_write_enabled=False`；
- full batch：4 并发 GPU0-3，`2026-05-07 08:15:48 -> 08:39:55`，约 `24.1 min`；
- 固定 `RESET_EVERY=5`，SWA 全部关闭。

Global metrics：

| Run | Filter | Branch | Blend | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `TTEX_01` | TTT-only reference | - | - | `36.5932` | `6.4327` | n/a | n/a | 当前 TTT-only reference |
| `TTOVF_03` | tail static topk `0.75` | `w0` | `1.00` | `36.7257` | `5.9924` | `92.4326` | `0.0074` | hard forget reference |
| `TTOVF2_03` | tail dynamic veto `1.00` | `w2` | `1.00` | `36.8408` | `6.2291` | `92.4344` | `0.0076` | branch hard forget reference |
| `ACL2V5_TTOVFB_01` | tail static topk `0.75` | `w0` | `0.25` | `36.8202` | `6.2645` | `92.4366` | `0.0077` | 不如 TTEX / hard w0 |
| `ACL2V5_TTOVFB_02` | tail static topk `0.75` | `w0` | `0.50` | `36.7151` | `6.1588` | `92.4354` | `0.0075` | 本批 best；仍未超过 TTEX |
| `ACL2V5_TTOVFB_03` | tail dynamic veto `1.00` | `w2` | `0.25` | `36.7889` | `6.3367` | `92.4361` | `0.0077` | w2 soft 不如 w0 soft |
| `ACL2V5_TTOVFB_04` | tail dynamic veto `1.00` | `w2` | `0.50` | `36.8418` | `6.3033` | `92.4358` | `0.0077` | 回退 |

Trajectory diagnostics：

| Run | ATE RMSE | FinalErr | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `TTEX_01` | `36.5932` | `5.120` | `29.809 / 78.815` | `30.445 / 78.750` | `30.519 / 57.753` | `3.596` | `31.219899` |
| `TTOVF_03` | `36.7257` | `1.391` | `30.053 / 79.171` | `30.665 / 78.814` | `30.822 / 57.852` | `3.152` | `31.125078` |
| `TTOVF2_03` | `36.8408` | `1.703` | `30.174 / 80.066` | `30.798 / 79.579` | `30.985 / 58.393` | `3.429` | `31.160979` |
| `TTOVFB_01` | `36.8202` | `3.696` | `30.070 / 79.137` | `30.703 / 78.898` | `30.770 / 57.927` | `3.444` | `31.170031` |
| `TTOVFB_02` | `36.7151` | `2.569` | `30.041 / 79.053` | `30.667 / 78.726` | `30.795 / 57.787` | `3.330` | `31.154522` |
| `TTOVFB_03` | `36.7889` | `4.204` | `30.031 / 79.197` | `30.668 / 78.903` | `30.729 / 57.940` | `3.533` | `31.160378` |
| `TTOVFB_04` | `36.8418` | `3.273` | `30.110 / 79.636` | `30.742 / 79.332` | `30.857 / 58.222` | `3.494` | `31.165894` |

20.49 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `TTEX_01 = 36.5932 / 6.4327`。
2. soft forget 的最好点是 `TTOVFB_02 = 36.7151 / 6.1588`，比 TTEX 回退 `+0.1219m`。
3. soft forget 能保留一部分 hard forget 的 orientation / endpoint 收益：`TTOVFB_02` FinalErr=`2.569`、Yaw=`3.330`，明显好于 TTEX 的 FinalErr=`5.120`、Yaw=`3.596`，但 segment mean 和 ATE 没有同步改善。
4. `w2` soft forget 没有超过 `w0` soft forget；value/third branch 的传播遗忘仍只像 endpoint regularizer。
5. 这批支持用户的判断：动态区域对当前 chunk 的 TTT 更新有用，直接 hard/soft token subset 都容易伤 ATE；下一步应把遗忘限制到更可能负责跨 chunk 传播的 layer 段，而不是全层统一忘。

### 20.50 计划：TTT layer-specific propagation / forget

用户新判断：

> 动态区域可能也为 TTT 提供了当前 chunk 内必要信息；这些动态区域对长期传播有风险，所以关键不是当前 chunk 禁用它们，而是传到下一个 chunk 时怎样遗忘。

已有结果对应：

- `TTCPF` 已经做了 “当前 chunk full TTT 不动，只在 commit 到下个 chunk 时缩放 fast weight”，但它是 branch 级标量，太粗；
- `TTOVF/TTOVFB` 做了 token-level replay forget，但之前是 all-layer，容易把当前 chunk 需要的更新方向也抹掉；
- 20.30 的 layer split 显示 late TTT layers 最能保留 Rot/FinalErr 收益，因此下一批只让 late/middle 层承担传播遗忘。

固定协议：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
RESET_EVERY = 5
SWA = off
success target = KITTI01 ATE < 30m
```

首批小矩阵：

| Run | 简写含义 | Layer mode | 写入策略 | 目的 |
|---|---|---|---|---|
| `ACL2V5_TTOVFL_01` | overlap forget layer-specific | late | tail static topk `0.75`，`w0`，blend `0.50` | 只在 late 层做 soft token forget，验证是否保留 endpoint/yaw 但减少 ATE 回退 |
| `ACL2V5_TTOVFL_02` | overlap forget layer-specific | middle | tail static topk `0.75`，`w0`，blend `0.50` | 对比 middle 是否才是跨 chunk 污染层 |
| `ACL2V5_TTCPFL_01` | commit propagation filter layer-specific | late | `native_to_candidate_by_risk`，tail `q90`，`w0` | 当前 chunk full replay 完整保留，只在 late commit 按 tail 高风险动态强度拉回 native |
| `ACL2V5_TTCPFL_02` | commit propagation filter layer-specific | late | `old_decay_by_risk`，tail `mean`，`w0` | 只在 late commit 缩短 dynamic risky update 的寿命，检查能否保留 TTCPF_03 的 Rot/FinalErr 而减小 ATE 伤害 |

Gate：

- 若任一 run 达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若任一 run 超过 `TTEX_01 = 36.5932` 或 `SWOVR_02 = 36.5915`，升级为 active base；
- 若仍只改善 FinalErr/Yaw 而 ATE 卡在 `36m+`，说明当前 fast-weight 传播遗忘还不够，需要改写 TTT replay objective 或 fast-weight delta 的方向估计，而不是继续扫 topk/EMA。

#### 20.50 实验结果：TTT layer-specific propagation / forget

运行记录：

- smoke `ACL2V5_TTOVFL_SMOKE_C23pairall_tailTopk075_w0_blend050_late`：`08:46:07 -> 08:49:06`，约 `3.0 min`；
- smoke 确认 `PRIOR_LAYER_MODE=late` 下非 late layer 为 `ttt_replay_token_filter_layer_disabled=True`，late layer `ttt_replay_token_filter_applied=True`，`blend=0.5`，`swa_write_enabled=False`；
- full batch：4 并发 GPU0-3，`08:49:54 -> 09:14:38`，约 `24.7 min`；
- 固定 `RESET_EVERY=5`，SWA 全部关闭。

Global metrics：

| Run | Layer mode | Propagation strategy | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| `SWOVR_02` reference | - | SWA historical tiny best | `36.5915` | `6.4307` | `92.4416` | `0.0078` | 参考；本轮已停止 SWA |
| `TTEX_01` reference | - | TTT-only native mix | `36.5932` | `6.4327` | n/a | n/a | 当前 TTT-only reference |
| `TTOVFB_02` reference | all | tail static topk `0.75`，`w0`，blend `0.50` | `36.7151` | `6.1588` | `92.4354` | `0.0075` | all-layer soft forget reference |
| `ACL2V5_TTOVFL_01` | late | tail static topk `0.75`，`w0`，blend `0.50` | `36.7368` | `6.1510` | `92.4355` | `0.0075` | late-only 保留 Rot/Final，但 ATE 更差 |
| `ACL2V5_TTOVFL_02` | middle | tail static topk `0.75`，`w0`，blend `0.50` | `36.7064` | `6.4114` | `92.4384` | `0.0078` | ATE 比 late 稍好，但未过 TTEX |
| `ACL2V5_TTCPFL_01` | late | `native_to_candidate_by_risk`，tail `q90`，`w0` | `36.6632` | `6.3558` | `92.4365` | `0.0077` | 本批 ATE best，但仍回退 |
| `ACL2V5_TTCPFL_02` | late | `old_decay_by_risk`，tail `mean`，`w0` | `37.3165` | `5.9113` | `92.4418` | `0.0074` | Yaw/Final 好，ATE 明显失败 |

Trajectory diagnostics：

| Run | ATE RMSE | FinalErr | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808 / 78.809` | `30.443 / 78.755` | `30.519 / 57.755` | `3.593` | `31.213845` |
| `TTEX_01` | `36.5932` | `5.120` | `29.809 / 78.815` | `30.445 / 78.750` | `30.519 / 57.753` | `3.596` | `31.219899` |
| `TTOVFB_02` | `36.7151` | `2.569` | `30.041 / 79.053` | `30.667 / 78.726` | `30.795 / 57.787` | `3.330` | `31.154522` |
| `TTOVFL_01` | `36.7368` | `2.669` | `30.024 / 79.209` | `30.647 / 78.824` | `30.766 / 57.867` | `3.335` | `31.158351` |
| `TTOVFL_02` | `36.7064` | `5.085` | `29.914 / 79.043` | `30.556 / 78.816` | `30.605 / 57.855` | `3.583` | `31.183112` |
| `TTCPFL_01` | `36.6632` | `4.723` | `29.886 / 79.111` | `30.521 / 78.740` | `30.568 / 57.812` | `3.536` | `31.160813` |
| `TTCPFL_02` | `37.3165` | `3.459` | `30.979 / 78.990` | `31.567 / 78.744` | `31.906 / 57.868` | `3.026` | `31.220267` |

20.50 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `TTEX_01 / SWOVR_02`。
2. `TTCPFL_01` 是本批 ATE best：`36.6632 / 6.3558`，比 TTEX 回退 `+0.0700m`，但比 `TTOVFB_02` 稍好。
3. `old_decay_by_risk` 的方向依然只改善 orientation / endpoint：`TTCPFL_02` Yaw=`3.026`、FinalErr=`3.459`，但 50/100/200 mean 全部明显回退，ATE 失败。
4. late-only token forget 并没有减少 ATE 伤害；middle-only 更接近 TTEX，但几乎失去 endpoint/yaw 收益。
5. 只看 `tail_overlap` seam 仍不够。TTT fast weights 是全局矩阵，非 overlap 动态 token 也会通过 replay update 写进下一 chunk 的 fast weights；下一批改成 **全 chunk / per-frame static replay sanitize**，但仍只在 commit 侧做，不影响当前 chunk 使用动态信息。

### 20.51 计划：TTT full-chunk propagation sanitize 与温和 commit decay

动机：

20.50 证明只在 tail overlap 做传播遗忘太窄。用户的假设仍成立的一种解释是：当前 chunk 内所有动态 token 都可以参与本 chunk 的 TTT 推理，但这些 token 对全局 fast-weight delta 的影响不应完整传到下一 chunk。

因此下一批不再只看 `tail_overlap`，而是：

- token replay sanitize 用 `per_frame_static_topk`，对整个 chunk 每帧保留高 static/write-prior token；
- commit decay 用更温和的 `old_decay_by_risk`，避免 20.50/20.46 的过强回旧状态导致 segment drift；
- SWA 继续关闭，reset 继续固定 `RESET_EVERY=5`。

小矩阵：

| Run | Strategy | Layer mode | Key params | 目的 |
|---|---|---|---|---|
| `ACL2V5_TTOVFLC_01` | full-chunk soft token sanitize | late | `per_frame_static_topk 0.75`, `w0`, blend `0.25` | 20.30 late hard top75 有信号；测试更温和传播 sanitize |
| `ACL2V5_TTOVFLC_02` | full-chunk soft token sanitize | late | `per_frame_static_topk 0.90`, `w0`, blend `0.25` | 更温和，避免误删当前有用动态 |
| `ACL2V5_TTCPF2_01` | all-layer native-to-candidate commit | all | tail `q90`, gain `1.0`, max `1.0` | 复查 TTCPF_02 是否因 `gain=1.5/max=1.25` 过冲 |
| `ACL2V5_TTCPF2_02` | all-layer mild old-decay commit | all | tail `mean`, gain `0.25`, min `0.85` | 保留 old-decay 的 Yaw/FinalErr 信号，但降低 ATE 损伤 |

Gate：

- 任一 run 达到 `<30m` 立即 repeat；
- 任一 run 超过 `TTEX_01/SWOVR_02`，升级为 TTT active base；
- 若仍未过 `36.59m`，说明靠 post-hoc fast-weight sanitize 仍不足，需要进入下一类：修改 TTT replay objective 的目标/残差方向，而不是只删 token 或缩放 commit delta。

工程实现：

- 新增 `TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND` / `--ttt_write_replay_token_filter_blend`。
- 涉及代码：`loger/pipeline/ttt_write_controller.py`、`loger/pipeline/hybrid_memory_controller.py`、`run_pipeline_abc_v2.py`、`tools/run_attention_cue_experiment.sh`。
- smoke：`ACL2V5_TTOVFB_SMOKE_C23pairall_tailTopk075_w0_b025`，`END_FRAME=128` 通过；`hmc_state_hash.jsonl` 确认 `ttt_replay_token_filter_branch_isolated=True`、`ttt_replay_token_filter_blend=0.25`、`swa_write_enabled=False`。
- `python3 -m py_compile loger/pipeline/ttt_write_controller.py loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py` 通过。

#### 20.51 实验结果：full-chunk propagation sanitize 与温和 commit decay

运行记录：

- full batch：GPU 0-3，`2026-05-07 09:18:58 -> 09:43:06`，约 `24.1 min` 完成 4 个 KITTI01 full run。
- reset：继续固定 `RESET_EVERY=5`，没有改 LoGeR 对齐的 reset 机制。
- SWA：本批全部关闭 SWA write / overlap source gate / source replace，只看 TTT commit propagation。
- debug：`TTOVFLC_01` 确认 `ttt_replay_token_filter_mode=per_frame_static_topk`，late 层启用、非 late 层 disabled，`ttt_replay_token_filter_blend=0.25`，`swa_write_enabled=False`。

Benchmark：

| Run | 具体含义 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `SWOVR_02` reference | v5 tiny TTT+SWA reference，SWA 现已暂停继续扩展 | `36.5915` | `6.4307` | n/a | n/a | 历史参考 |
| `TTEX_01` reference | TTT-only reference：native mix `w0=1.10` | `36.5932` | `6.4327` | n/a | n/a | 当前 TTT-only best |
| `TTCPFL_01` reference | late-only `native_to_candidate_by_risk`，tail q90，`w0` | `36.6632` | `6.3558` | `92.4365` | `0.0077` | 上批 best |
| `ACL2V5_TTOVFLC_01` | full-chunk per-frame static topk `0.75`，late `w0`，blend `0.25` | `36.6883` | `6.3583` | `92.4353` | `0.0077` | 比 TTCPFL_01 回退 |
| `ACL2V5_TTOVFLC_02` | full-chunk per-frame static topk `0.90`，late `w0`，blend `0.25` | `36.6993` | `6.3380` | `92.4362` | `0.0077` | 更温和仍未过 |
| `ACL2V5_TTCPF2_01` | all-layer `native_to_candidate_by_risk`，tail q90，gain `1.0` | `36.6949` | `6.3787` | `92.4366` | `0.0078` | 温和过冲修正未过 |
| `ACL2V5_TTCPF2_02` | all-layer mild `old_decay_by_risk`，tail mean，gain `0.25`，min `0.85` | `36.7691` | `6.2789` | `92.4386` | `0.0077` | endpoint/yaw 有信号，ATE 回退 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean/worst | 100f mean/worst | 200f mean/worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808 / 78.809` | `30.443 / 78.755` | `30.519 / 57.755` | `3.593` | `31.213845` |
| `TTEX_01` | `36.5932` | `5.120` | `29.809 / 78.815` | `30.445 / 78.750` | `30.519 / 57.753` | `3.596` | `31.219899` |
| `TTCPFL_01` | `36.6632` | `4.723` | `29.886 / 79.111` | `30.521 / 78.740` | `30.568 / 57.812` | `3.536` | `31.160813` |
| `TTOVFLC_01` | `36.6883` | `4.650` | `29.888 / 78.991` | `30.525 / 78.737` | `30.567 / 57.808` | `3.540` | `31.148968` |
| `TTOVFLC_02` | `36.6993` | `4.527` | `29.895 / 78.989` | `30.531 / 78.730` | `30.585 / 57.815` | `3.515` | `31.157391` |
| `TTCPF2_01` | `36.6949` | `4.642` | `29.914 / 79.058` | `30.550 / 78.781` | `30.601 / 57.836` | `3.564` | `31.166453` |
| `TTCPF2_02` | `36.7691` | `3.531` | `30.053 / 79.064` | `30.682 / 78.861` | `30.785 / 57.883` | `3.450` | `31.189959` |

20.51 结论：

1. 全 chunk per-frame static sanitize 没有超过 `TTEX_01` / `SWOVR_02`，也没有达到成功标准 `KITTI01 ATE < 30`。
2. `TTOVFLC_01/02` 比 `TTOVFB_02` 的 endpoint/yaw 信号更弱，说明“把所有疑似动态 token 的传播都拉向 static replay”仍然会伤害局部 segment mean。
3. `old_decay_by_risk` 的温和版仍是同一类 trade-off：FinalErr/Yaw 改善，但 ATE 与 50/100/200 mean 回退。
4. 机制判断：动态区域不应被简单过滤；它们在当前 chunk 内提供了必要更新。下一步改为 **aligned dynamic commit**：filtered/static replay 作为主方向，dynamic replay 只在和 static 更新方向一致时允许传到下一 chunk。

### 20.52 计划：TTT aligned dynamic commit

用户假设更新：

> 动态区域可能也为 TTT 提供当前 chunk 的必要信息；问题不是当前 chunk 不该用动态，而是这些动态更新不应无条件传播到下一个 chunk。

因此新增 `TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE=aligned_dynamic`：

- `base replay` = full replay，包含当前 chunk 全部动态/静态 token 的 TTT 更新；
- `filtered replay` = static/topk replay，代表更可信的长期传播方向；
- `dynamic delta = base - filtered`；
- 只保留与 `filtered - W_m` 同向的 dynamic delta；如果 dynamic delta 与 static replay 方向冲突，则在 commit 时忘掉；
- 仍只作用于下一 chunk 的 fast weights，不改变当前 chunk 的 TTT 使用；
- SWA 继续关闭，reset 继续固定 `RESET_EVERY=5`。

小矩阵：

| Run | Strategy | Key params | 目的 |
|---|---|---|---|
| `ACL2V5_TTAD_01` | aligned dynamic full-chunk commit | `per_frame_static_topk 0.75`, late `w0`, blend `0.25` | 基准 aligned-dynamic，允许少量同向动态传播 |
| `ACL2V5_TTAD_02` | aligned dynamic full-chunk commit | `per_frame_static_topk 0.75`, late `w0`, blend `0.50` | 更强 forgetting，检查 endpoint/yaw 与 ATE trade-off |
| `ACL2V5_TTAD_03` | aligned dynamic tail commit | tail static topk `0.75`, late `w0`, blend `0.25` | 对比只处理 tail seam 是否足够 |
| `ACL2V5_TTAD_04` | aligned dynamic middle commit | `per_frame_static_topk 0.75`, middle `w0`, blend `0.25` | 检查传播污染是否主要来自 middle TTT 层 |

Gate：

- 任一 run 达到 `KITTI01 ATE < 30` 立即 repeat；
- 任一 run 超过 `TTEX_01/SWOVR_02`，升级为 TTT active base 并补 repeat；
- 若仍未过 `36.59m`，说明 TTT fast-weight commit 的改善不能靠 replay token 后处理，需要进入更大结构：两阶段 replay / objective-level weighting / branch-specific learned update rule。

工程实现：

- 新增 `TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE` / `--ttt_write_replay_token_filter_blend_mode`。
- 相关代码：`loger/pipeline/ttt_write_controller.py`、`loger/pipeline/hybrid_memory_controller.py`、`run_pipeline_abc_v2.py`、`tools/run_attention_cue_experiment.sh`。
- `python3 -m py_compile loger/pipeline/ttt_write_controller.py loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py` 通过。

#### 20.52 实验结果：TTT aligned dynamic commit

运行记录：

- smoke `ACL2V5_TTAD_SMOKE2_C23pairall_fullTopk075_w0_alignDyn_b025_late`：`09:49:26 -> 09:52:22`，约 `3.0 min`；
- smoke 确认 `ttt_replay_token_filter_blend_mode=aligned_dynamic` 进入 late TTT layer，且记录 `ttt_replay_token_filter_w0_align_cos_mean` / `ttt_replay_token_filter_w0_dyn_keep_mean`；
- full batch：GPU 0-3，`2026-05-07 09:53:21 -> 10:17:16`，约 `23.9 min` 完成 4 个 KITTI01 full run；
- reset：继续固定 `RESET_EVERY=5`；
- SWA：本批全部关闭 SWA write / overlap bias / source gate / source replace。

Benchmark：

| Run | 具体含义 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `SWOVR_02` reference | v5 tiny TTT+SWA reference，SWA 现已暂停继续扩展 | `36.5915` | `6.4307` | n/a | n/a | 历史参考 |
| `TTEX_01` reference | TTT-only reference：native mix `w0=1.10` | `36.5932` | `6.4327` | n/a | n/a | 当前 TTT-only best |
| `TTCPFL_01` reference | late-only `native_to_candidate_by_risk`，tail q90，`w0` | `36.6632` | `6.3558` | `92.4365` | `0.0077` | 上批 best |
| `ACL2V5_TTAD_01` | aligned dynamic：full per-frame static topk `0.75`，late `w0`，blend `0.25` | `36.6863` | `6.0820` | `92.4317` | `0.0075` | Rot/Final 好，ATE 未过 |
| `ACL2V5_TTAD_02` | aligned dynamic：full per-frame static topk `0.75`，late `w0`，blend `0.50` | `36.6758` | `6.1260` | `92.4308` | `0.0075` | 本批 ATE best，但仍回退 |
| `ACL2V5_TTAD_03` | aligned dynamic：tail static topk `0.75`，late `w0`，blend `0.25` | `36.7586` | `6.0823` | `92.4350` | `0.0075` | FinalErr/Yaw 最好，ATE 回退 |
| `ACL2V5_TTAD_04` | aligned dynamic：full per-frame static topk `0.75`，middle `w0`，blend `0.25` | `36.7772` | `6.3905` | `92.4385` | `0.0078` | middle 失败 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean/worst | 100f mean/worst | 200f mean/worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808 / 78.809` | `30.443 / 78.755` | `30.519 / 57.755` | `3.593` | `31.213845` |
| `TTEX_01` | `36.5932` | `5.120` | `29.809 / 78.815` | `30.445 / 78.750` | `30.519 / 57.753` | `3.596` | `31.219899` |
| `TTCPFL_01` | `36.6632` | `4.723` | `29.886 / 79.111` | `30.521 / 78.740` | `30.568 / 57.812` | `3.536` | `31.160813` |
| `TTAD_01` | `36.6863` | `3.477` | `29.874 / 78.865` | `30.502 / 78.603` | `30.574 / 57.734` | `3.301` | `31.105240` |
| `TTAD_02` | `36.6758` | `3.195` | `29.954 / 78.821` | `30.581 / 78.482` | `30.663 / 57.643` | `3.327` | `31.096341` |
| `TTAD_03` | `36.7586` | `2.053` | `30.094 / 79.268` | `30.716 / 78.867` | `30.858 / 57.894` | `3.259` | `31.149964` |
| `TTAD_04` | `36.7772` | `5.145` | `29.953 / 79.180` | `30.596 / 78.992` | `30.630 / 57.986` | `3.581` | `31.182514` |

20.52 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `TTEX_01 / SWOVR_02`。
2. aligned-dynamic 机制显著改善 Rot / FinalErr / Yaw：`TTAD_03` FinalErr=`2.053`、Yaw=`3.259`，但 ATE 和 segment mean 回退，仍是 orientation/endpoint trade-off。
3. full-chunk aligned dynamic 比 tail-only 更适合 ATE：`TTAD_02` ATE=`36.6758`，但仍比 `TTEX_01` 差 `+0.0825m`。
4. middle-only 不成立；能改善 orientation 的主要还是 late commit。
5. 机制判断更新：动态 token 不能简单删，也不能只靠“同向才保留”修补。更合理的模型是 **动态更新可以短期影响下一 chunk，但不应进入长期 fast-weight 累积**。下一步改成 TTT transient / TTL commit：显式估计本轮 dynamic residual，让它只活一跳，在下一次 commit 时从长期 fast weights 中扣掉。

### 20.53 计划：TTT transient dynamic TTL commit

新假设：

- TTT 的动态区域确实可能给相邻 chunk 提供必要 continuity；
- 失败点在于这些 dynamic residual 被写进 `W_{m+1}` 后，继续叠到 `W_{m+2}, W_{m+3}`，形成长期污染；
- 因此不是“少写动态区域”，而是“动态区域只短期传播，下一次 commit 要忘记上一轮动态残差”。

拟实现：

- 在 `TTTWriteController` 中新增 transient commit 模式；
- 当前 commit 同时计算：
  - `base replay`：full replay；
  - `filtered/static replay`：static/topk replay；
  - `dynamic residual = base - filtered`；
- `W_{m+1}` 允许保留一部分本轮 dynamic residual；
- 到下一次 commit 时，从 candidate fast weights 里扣除上一轮保存的 dynamic residual，近似实现 one-hop TTL；
- 只作用 TTT，SWA 继续关闭，reset 继续固定 `RESET_EVERY=5`。

工程实现：

- `TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE=ttl_dynamic`：当前 commit 仍使用 full replay，把 `base replay - filtered/static replay` 记录为 transient dynamic delta；
- `TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE=ttl_aligned_dynamic`：只保留与 filtered/static replay 更新方向一致的 dynamic delta，并记录实际保留的 transient delta；
- `TTT_WRITE_TRANSIENT_DELTA_SUBTRACT_SCALE`：下一次 TTT commit 结束时从 fast weights 中扣除上一轮 transient delta，默认 `0` 关闭；
- transient delta 存在 `ttt_state["transient_delta"]`，只给下一次 HMC commit 使用，不传入 LoGeR model；`RESET_EVERY=5` 的 reset 逻辑不改，reset 后 transient delta 会自然丢弃；
- SWA 在本批保持关闭：`ENABLE_SWA_WRITE_CONTROL=0`，不再新增 SWA 任务。

Smoke：

- `ACL2V5_TTDTTL_SMOKE_C23pairall_fullTopk075_w0_ttlDyn_s100_late`：`10:25:32 -> 10:28:30`，约 `3.0 min`；
- 日志确认 `ttt_replay_token_filter_blend_mode=ttl_dynamic`、`ttt_transient_delta_stored=True`，并且第二个 chunk 后出现 `ttt_transient_delta_prev_subtract_applied=True`；
- smoke 只用于工程验证，不参与指标结论。

Full KITTI01 小矩阵：

| Run | 机制 | 关键设置 | 目的 |
|---|---|---|---|
| `ACL2V5_TTDTTL_01` | full-chunk one-hop dynamic TTL | per-frame static topk `0.75`，late `w0`，subtract `1.0` | 动态完整进入下一 chunk，但只活一跳 |
| `ACL2V5_TTDTTL_02` | full-chunk soft one-hop dynamic TTL | per-frame static topk `0.75`，late `w0`，subtract `0.5` | 检查扣除过强是否伤 ATE |
| `ACL2V5_TTDTTL_03` | aligned-dynamic one-hop TTL | per-frame static topk `0.75`，late `w0`，aligned blend `0.25`，subtract `1.0` | 同向动态允许短期传播，下一跳忘记 |
| `ACL2V5_TTDTTL_04` | tail-only one-hop dynamic TTL | tail-overlap static topk `0.75`，late `w0`，subtract `1.0` | 只处理 chunk seam 的动态传播 |

Gate：

- 任一 run 达到 `KITTI01 ATE < 30` 立即 repeat；
- 任一 run 超过 `TTEX_01/SWOVR_02`，升级为 TTT active base；
- 如果 TTL 也只改善 endpoint/yaw 而不降 ATE，说明 v5 当前 TTT 写入瓶颈不在 propagation forget，而在 TTT update objective 本身，需要探索 replay loss / branch objective，而不是继续 commit 后处理。

#### 20.53 实验结果：TTT transient dynamic TTL commit

资源/时间：

- smoke `ACL2V5_TTDTTL_SMOKE_C23pairall_fullTopk075_w0_ttlDyn_s100_late`：`10:25:32 -> 10:28:30`，约 `3.0 min`；
- full batch：GPU 0-3，`10:29:09 -> 10:53:49`，4 runs，约 `24.7 min`；
- SWA 全部关闭，reset 仍固定 `RESET_EVERY=5`。

主指标：

| Run | 机制 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `SWOVR_02` | 当前 v5 tiny best reference | `36.5915` | `6.4307` | n/a | n/a | reference，含 SWA source-V replacement，后续 SWA 暂停 |
| `TTEX_01` | TTT native-mix reference | `36.5932` | n/a | n/a | n/a | TTT-only reference |
| `ACL2V5_TTDTTL_01` | full per-frame static topk `0.75`，late `w0`，TTL subtract `1.0` | `36.7732` | `6.1927` | `92.4331` | `0.0076` | Rot 改善，ATE 回退 |
| `ACL2V5_TTDTTL_02` | full per-frame static topk `0.75`，late `w0`，TTL subtract `0.5` | `36.7036` | `6.2600` | `92.4350` | `0.0077` | 本批 ATE best，但未过 reference |
| `ACL2V5_TTDTTL_03` | aligned-dynamic TTL，blend `0.25`，subtract `1.0` | `36.7593` | `6.1417` | `92.4314` | `0.0075` | Rot 好，ATE 回退 |
| `ACL2V5_TTDTTL_04` | tail-overlap static topk `0.75`，late `w0`，TTL subtract `1.0` | `36.7589` | `6.1295` | `92.4342` | `0.0076` | endpoint/yaw 好，ATE 回退 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808 / 78.809` | `30.443 / 78.755` | `30.519 / 57.755` | `3.593` | `31.213845` |
| `TTEX_01` | `36.5932` | `5.120` | `29.809 / 78.815` | `30.445 / 78.750` | `30.519 / 57.753` | `3.596` | `31.219899` |
| `TTCPFL_01` | `36.6632` | `4.723` | `29.886 / 79.111` | `30.521 / 78.740` | `30.568 / 57.812` | `3.536` | `31.160813` |
| `TTAD_02` | `36.6758` | `3.195` | `29.954 / 78.821` | `30.581 / 78.482` | `30.663 / 57.643` | `3.327` | `31.096341` |
| `TTDTTL_01` | `36.7732` | `3.675` | `29.956 / 79.165` | `30.588 / 78.854` | `30.650 / 57.899` | `3.378` | `31.121183` |
| `TTDTTL_02` | `36.7036` | `4.009` | `29.915 / 79.139` | `30.549 / 78.824` | `30.613 / 57.867` | `3.448` | `31.144162` |
| `TTDTTL_03` | `36.7593` | `3.273` | `29.982 / 78.950` | `30.610 / 78.615` | `30.685 / 57.761` | `3.355` | `31.102712` |
| `TTDTTL_04` | `36.7589` | `2.251` | `30.052 / 79.224` | `30.673 / 78.836` | `30.796 / 57.883` | `3.308` | `31.140530` |

20.53 结论：

1. 没有达到成功标准 `KITTI01 ATE < 30`；也没有超过 `SWOVR_02` / `TTEX_01`。
2. TTL 机制方向有效地改善 Rot / FinalErr / Yaw：`TTDTTL_04` FinalErr=`2.251`、Yaw=`3.308`，但 ATE/segment mean 回退，仍然是 orientation/endpoint trade-off。
3. subtract `0.5` 比 `1.0` 更利于 ATE，说明“忘记上一轮动态 residual”如果过强，会损伤当前 TTT 需要的连续几何信息。
4. full-chunk TTL 比 tail-only TTL 更利于 ATE；tail-only 更像 endpoint/yaw diagnostic。
5. 机制判断：动态区域确实不能简单长期传播，但在 fast-weight 空间用 `base-filtered` 残差直接扣除太粗，会把一部分有用静态/几何更新一起扣掉。下一步 TTT 优先探索 **更新目标本身**：把动态区作为短期自适应信号保留在当前 chunk，但只让更可靠的 branch/layer/objective 进入长期 fast weights，而不是继续做单纯 commit 后处理。

### 20.49 实验结果：TTT overlap soft forget

运行记录：

- full batch：GPU 0-3，`2026-05-07 08:15:48 -> 08:39:55`，约 `24.1 min` 完成 4 个 KITTI01 full run。
- 资源：4 并发稳定；中途 host RAM available 仍有数百 GiB，未触发 swap/OOM 风险。
- reset：继续固定 `RESET_EVERY=5`，没有改 LoGeR 对齐的 reset 机制。
- SWA：本批全部关闭 SWA write / overlap source gate / source replace，只看 TTT commit propagation。

Benchmark：

| Run | 具体含义 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `SWKS3_03` | 历史 SWA-active tiny best，当前已暂停 SWA 继续扩展 | `36.4153` | `6.6186` | n/a | n/a | 全局参考，不作为当前 TTT-only 目标 |
| `TTEX_01` | TTT-only 当前 reference：native mix `w0=1.10` | `36.5932` | `6.4327` | n/a | n/a | 当前 TTT-only best |
| `TTOVF_03` | hard forget：tail static topk `0.75`，branch `w0`，blend=`1.0` | `36.7257` | `5.9924` | n/a | n/a | FinalErr/Yaw 强，但 ATE 回退 |
| `TTOVF2_03` | hard forget branch 拆解：tail dynamic veto `1.00`，branch `w2`，blend=`1.0` | `36.8408` | `6.2291` | n/a | n/a | branch hard filter 里较稳，但仍回退 |
| `ACL2V5_TTOVFB_01` | soft forget：tail static topk `0.75`，branch `w0`，blend=`0.25` | `36.8202` | `6.2645` | `92.4366` | `0.0077` | 不如 TTEX / hard w0 |
| `ACL2V5_TTOVFB_02` | soft forget：tail static topk `0.75`，branch `w0`，blend=`0.50` | `36.7151` | `6.1588` | `92.4354` | `0.0075` | 本批 best；仍未超过 TTEX |
| `ACL2V5_TTOVFB_03` | soft forget：tail dynamic veto `1.00`，branch `w2`，blend=`0.25` | `36.7889` | `6.3367` | `92.4361` | `0.0077` | w2 soft 不如 w0 soft |
| `ACL2V5_TTOVFB_04` | soft forget：tail dynamic veto `1.00`，branch `w2`，blend=`0.50` | `36.8418` | `6.3033` | `92.4358` | `0.0077` | 回退 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean / worst | 100f mean / worst | 200f mean / worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SWKS3_03` | `36.4153` | `5.687` | `29.734 / 78.996` | `30.383 / 77.833` | `30.436 / 57.063` | `3.765` | `31.298748` |
| `TTEX_01` | `36.5932` | `5.120` | `29.809 / 78.815` | `30.445 / 78.750` | `30.519 / 57.753` | `3.596` | `31.219899` |
| `TTOVF_03` | `36.7257` | `1.391` | `30.053 / 79.171` | `30.665 / 78.814` | `30.822 / 57.852` | `3.152` | `31.125078` |
| `TTOVF2_03` | `36.8408` | `1.703` | `30.174 / 80.066` | `30.798 / 79.579` | `30.985 / 58.393` | `3.429` | `31.160979` |
| `TTOVFB_01` | `36.8202` | `3.696` | `30.070 / 79.137` | `30.703 / 78.898` | `30.770 / 57.927` | `3.444` | `31.170031` |
| `TTOVFB_02` | `36.7151` | `2.569` | `30.041 / 79.053` | `30.667 / 78.726` | `30.795 / 57.787` | `3.330` | `31.154522` |
| `TTOVFB_03` | `36.7889` | `4.204` | `30.031 / 79.197` | `30.668 / 78.903` | `30.729 / 57.940` | `3.533` | `31.160378` |
| `TTOVFB_04` | `36.8418` | `3.273` | `30.110 / 79.636` | `30.742 / 79.332` | `30.857 / 58.222` | `3.494` | `31.165894` |

20.49 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过当前 TTT-only reference `TTEX_01 = 36.5932 / 6.4327`。
2. `w0` soft forget 的最佳点是 `blend=0.50`，ATE=`36.7151`，比 hard `w0` 的 `36.7257` 略好，但仍比 TTEX 差 `+0.1219m`。
3. soft forget 能保留一部分 hard forget 的 orientation / endpoint 收益：`TTOVFB_02` FinalErr=`2.569`、Yaw=`3.330`，明显好于 TTEX 的 FinalErr=`5.120`、Yaw=`3.596`，但 segment mean 和 ATE 没有同步改善。
4. `w2` soft forget 不如 `w0` soft forget；branch `w2` 更像弱 regularizer，不像主要污染源。
5. 这批支持用户的机制判断：动态区域在当前 chunk / full replay 中仍然有必要信息，不能硬删；但“replay token 子集 soft 混合”仍不足以解决 ATE 主误差。
6. 下一步 TTT 优先方向应转到 **commit delta 级传播门控**：先让当前 chunk 使用完整 TTT/full replay，再在提交给下一 chunk 的 fast-weight delta 上按 overlap 动态风险做连续缩放、方向筛选或 EMA，而不是改 replay token 集合。

资源记录：

- 6 并发 full KITTI01：GPU 0-5，`07:18:45 -> 07:44:14`，约 `25.5 min` 完成。
- 同时短暂尝试在 GPU 6-7 开 `TTW` 宽幅写入补跑，但 8 并发再次把 host RAM 压到危险区，因此主动中止 `TTW_01/02` partial；这两个 partial 不计入有效结果。

### 14.2 SWA-write benchmark 指标

| Run | SWA mode | rho | min_gate | sparse_ratio | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| WRSP_01 reference | off | 0.00 | n/a | n/a | `36.6749` | `6.0432` | `92.4311` | `0.0075` | 当前写入侧 reference |
| ACL2V5_SWAW_01 | v | 0.50 | 0.50 | 1.00 | `37.4663` | `6.5262` | `92.4029` | `0.0078` | 明显 ATE 回退 |
| ACL2V5_SWAW_02 | v | 1.00 | 0.00 | 1.00 | `38.8317` | `6.9506` | `92.3542` | `0.0082` | 强 gate 失败 |
| ACL2V5_SWAW_03 | v | 1.00 | 0.00 | 0.50 | `42.0464` | `6.7636` | `92.3261` | `0.0114` | sparse SWA-V 崩 |
| ACL2V5_SWAW_04 | kv | 1.00 | 0.00 | 0.50 | `39.9532` | `9.4209` | `92.3890` | `0.0124` | KV 一起 gate 不安全 |
| ACL2V5_SWAW_05 | v | 1.00 | 0.00 | 0.25 | `49.2052` | `15.1105` | `92.4955` | `0.0236` | 过强稀疏，严重失败 |
| ACL2V5_SWAW_06 | kv | 0.50 | 0.50 | 1.00 | `37.3603` | `6.4861` | `92.4259` | `0.0090` | 比 SWAW_01 略好，但仍失败 |

### 14.3 SWA-write 结论

1. 当前 SWA-write hook 是生效的；`hmc_state_hash.jsonl` 中记录了 `swa_write_enabled=True`、`swa_write_applied_layers=4` 和 gate 统计。
2. 但第一批策略全部未过 gate：最温和的 `kv r0.50 min0.50` 也只有 `37.3603m`，比 WRSP_01 差 `0.6854m`。
3. 强 SWA sparse 写入非常危险，`v sparse0.25` 直接退到 `49.2052m`；这说明 SWA local memory 里有大量 trajectory-continuity 必需 token，不能按 D_g 全局粗暴裁掉。
4. 下一步 SWA 不继续加大 rho / 稀疏率，而要改成更局部、更晚层、更温和的写入策略，例如只 gate late SWA 层、只 gate value、`min_gate>=0.85`，或者只把 SWA 写入当作 rotation/endpoint diagnostic。
5. SWA-write 仍未达到最终成功标准 `<30m`，但它给出明确负面边界：不要用 D_g 对 SWA history 做全局强 veto。

---

## 15. 写入侧继续加压计划：TTT wide / branch-layer / local SWA

当前写入侧最强是：

```text
C23 past + pair/all + beta 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT sparse = 0.75 soft
KITTI01 = 36.6749 / 6.0432
```

离 `<30m` 仍差很远，因此后续写入侧不做无意义小细扫，改成三条高优先级：

1. **TTT wide prior**：放开 `mp_min/mp_max` 到 `[0,1.5]` / `[0,2.0]`，确认之前 `0.8-1.2` 是否过窄。
2. **TTT branch/layer policy**：验证 branch0 是否仍是唯一合理写入分支，以及 early/middle/late TTT 层是否有不同作用。
3. **SWA local-write policy**：放弃强全局 SWA veto，后续只做温和 value gate 或 late-layer gate。

下一批先执行 TTT wide，因为它不需要新增工程，且能直接测试“写入幅度是不是不够大胆”。

工程补充：

- 新增 `--swa_write_layer_mode all/first/last/early/middle/late/single`。
- 新增 `--swa_write_single_layer`。
- `tools/run_attention_cue_experiment.sh` 暴露 `SWA_WRITE_LAYER_MODE` / `SWA_WRITE_SINGLE_LAYER`。

这样下一批 SWA 不再只能 gate 全部 4 个 SWA history layer，可以直接测试 `last` / `late` 的温和写入策略。

---

## 16. TTT writing 第二批：wide prior 与 branch policy

### 16.1 本批目的

用户把成功标准明确为：

```text
KITTI01 ATE < 30m
```

因此这一批不再只做微小 beta / sparse ratio 调整，而是直接测试两个更底层的问题：

1. `0.8-1.2` 的写入 prior 范围是否太保守；
2. branch0 是否仍应是唯一被 prior 控制的写入分支。

固定协议：

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
FAST_CUE_EVAL = 1
success target = KITTI01 ATE < 30m
previous best = WRSP_01 = 36.6749 / 6.0432
```

资源记录：

- 起初尝试 8 并发，其中 GPU 6-7 加跑 `TTL early/late`。运行到 `28-29` 个 chunk 时 host RAM available 只剩 `~24GiB`，存在 OOM / swap 风险。
- 为保住更关键且接近完成的 `TTW/TTB` 结果，主动中止 `ACL2V5_TTL_01/02` partial；这两条不计入有效结果。
- 有效完成 6 条 full run，`07:50/07:51 -> 08:17`，约 `26-28 min`。
- 结论：KITTI01 full 仍可短时 6 并发；8 并发不稳定，除非只跑短 smoke。

### 16.2 Benchmark 指标

| Run | 写入策略 | Branch prior | Sparse | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| WRSP_01 reference | `alpha=0.125, [0.8,1.2]` | 0 | 0.75 soft | `36.6749` | `6.0432` | `92.4311` | `0.0075` | previous best |
| ACL2V5_TTW_05 | `alpha=0.35, [0,1.5]` | 0 | 1.00 | `36.6566` | `6.2878` | `92.4353` | `0.0077` | **v5 当前 ATE 新低，但 Rot 回退** |
| ACL2V5_TTW_06 | `alpha=0.35, [0,1.5]` | 0 | 0.75 soft | `36.7176` | `6.0010` | `92.4318` | `0.0075` | endpoint/yaw 强，ATE 回退 |
| ACL2V5_TTW_07 | `alpha=0.50, [0,1.5]` | 0 | 1.00 | `36.6691` | `6.2463` | `92.4343` | `0.0077` | 接近 TTW_05，未超过 |
| ACL2V5_TTW_08 | `alpha=0.50, [0,1.5]` | 0 | 0.75 soft | `36.6894` | `5.9848` | `92.4306` | `0.0074` | Rot / endpoint 好，ATE 不如 TTW_05 |
| ACL2V5_TTB_01 | `alpha=0.125, [0.8,1.2]` | 0,1 | 0.75 soft | `37.1356` | `5.7656` | `92.4354` | `0.0082` | branch1 加入后 ATE/endpoint 崩 |
| ACL2V5_TTB_02 | `alpha=0.125, [0.8,1.2]` | 0,2 | 0.75 soft | `36.9080` | `6.0754` | `92.4345` | `0.0075` | branch2 加入也回退 |

### 16.3 Trajectory diagnostics

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTW_05 | `36.6566` | `4.618` | `29.826` | `30.462` | `30.506` | `3.480` | `31.148541` |
| TTW_06 | `36.7176` | `2.147` | `30.001` | `30.619` | `30.744` | `3.182` | `31.112862` |
| TTW_07 | `36.6691` | `4.381` | `29.831` | `30.464` | `30.501` | `3.441` | `31.133683` |
| TTW_08 | `36.6894` | `1.880` | `30.003` | `30.617` | `30.742` | `3.174` | `31.101056` |
| TTB_01 | `37.1356` | `11.559` | `31.506` | `32.049` | `32.725` | `2.868` | `31.175014` |
| TTB_02 | `36.9080` | `1.855` | `30.426` | `31.038` | `31.258` | `3.221` | `31.137442` |

### 16.4 结论

1. 放宽 TTT write prior 幅度有小幅正收益：`TTW_05` 从 `36.6749` 降到 `36.6566`，是当前 v5 ATE 新低，但只是 `0.0183m` 量级，仍远未达到 `<30m`。
2. `alpha=0.35` 比 `alpha=0.50` 略好；继续单纯加大 alpha 不是主突破方向。
3. sparse + wide 明显改善 endpoint/yaw/rotation，但 ATE 不升反降，说明“少写动态区域”确实影响 orientation/endpoint，但 translation / global shape 还需要保留足够静态连续性。
4. branch0 仍应保持默认主线；把 prior 扩到 branch1 或 branch2 都没有帮助。尤其 branch1 导致 FinalErr=`11.559m`，风险很高。
5. 8 并发在 KITTI01 full 上再次触发内存危险；后续默认 6 并发上限，除非明确是短 smoke。
6. 下一步需要测试真正 suppressive TTT：当前 `eta_mean_preserving` 会把动态区少写的学习率质量补偿给静态 token，可能导致过度集中；`suppressive` 才是真正减少总写入。

---

## 17. 写入侧第三批启动：suppressive TTT / 动态准确性 / centered SWA

### 17.1 工程补充

为了围绕“动态区域是否准确、怎么写入”继续探索，新增/暴露：

- `tools/run_attention_cue_experiment.sh` 暴露 `PRIOR_POLICY`，可选 `eta_mean_preserving` / `suppressive`。
- `tools/run_attention_cue_experiment.sh` 暴露 `LAMBDA_MIN` / `LAMBDA_MAX`。
- 新增 write score `stage_d_x_dg_high_inv(_sqrt)`：只惩罚高置信 `D_g > 0.5` 区域，避免低置信 D_g 全图误伤。
- 新增 write score `stage_d_x_dg_exp_inter_inv(_sqrt)`：只在 `D_g` 与 `explicit_dyn` consensus 高时降低写入，测试更保守的动态区域定义。
- 新增 SWA write mode `v_centered` / `kv_centered`：用 centered rank 对 SWA history 做近似 mean-preserving 重权重，而不是只做 `<=1` 的全局削弱。

验证：

```text
python -m py_compile loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py
bash -n tools/run_attention_cue_experiment.sh
```

### 17.2 正在运行的候选

固定协议仍为：

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
FAST_CUE_EVAL = 1
success target = KITTI01 ATE < 30m
```

| Run | 类型 | 关键参数 | 目的 |
|---|---|---|---|
| ACL2V5_TTS_01 | suppressive TTT | `stage_d_x_dg_inv_sqrt`, sparse0.75 | 真正减少总写入，而非 eta 补偿 |
| ACL2V5_TTS_02 | suppressive TTT | `stage_d_x_dg_inv_sqrt`, sparse0.50 | 更强 suppressive gate |
| ACL2V5_TTS_03 | suppressive TTT | `alpha=0.35, [0,1.5]`, no sparse | 对照 TTW_05，测试 eta-preserve 是否必要 |
| ACL2V5_TTDYN_01 | high-only D_g | `stage_d_x_dg_high_inv_sqrt`, sparse0.75 | 只 veto 高置信 D_g，减少误伤 |
| ACL2V5_TTDYN_02 | consensus dyn | `stage_d_x_dg_exp_inter_inv_sqrt`, sparse0.75 | 用 D_g ∩ explicit_dyn 提高动态区域准确性 |
| ACL2V5_SWAC_01 | centered SWA | `v_centered`, rho0.10, min0.85, last layer | 不削弱总 SWA continuity，只做温和重权重 |

### 17.3 第三批结果

固定协议同 17.2。6 并发 full KITTI01 完成，GPU 0-5，约 `08:20 -> 08:47`，没有触发 8 并发那类 host RAM 危险。

| Run | 类型 | 关键参数 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| ACL2V5_TTS_01 | suppressive TTT | sparse0.75 soft | `36.7114` | `6.1176` | `92.4348` | `0.0075` | 少写总量后 ATE 回退 |
| ACL2V5_TTS_02 | suppressive TTT | sparse0.50 soft | `36.8946` | `5.9368` | `92.4272` | `0.0074` | Rot 好，但 ATE 明显回退 |
| ACL2V5_TTS_03 | suppressive TTT | `alpha=0.35,[0,1.5]`, no sparse | `36.6517` | `6.3281` | `92.4351` | `0.0077` | TTT-only 新低，略优于 TTW_05 |
| ACL2V5_TTDYN_01 | high-only D_g | `stage_d_x_dg_high_inv_sqrt`, sparse0.75 | `36.7318` | `6.0237` | `92.4313` | `0.0075` | 高置信 D_g 不够 |
| ACL2V5_TTDYN_02 | consensus dyn | `stage_d_x_dg_exp_inter_inv_sqrt`, sparse0.75 | `36.7354` | `6.0297` | `92.4336` | `0.0075` | D_g ∩ explicit_dyn 没过 gate |
| ACL2V5_SWAC_01 | centered SWA | `v_centered`, rho0.10, min0.85, last | `36.6436` | `6.2450` | `92.4366` | `0.0077` | v5 当前 overall best，但仍远离 `<30m` |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTS_01 | `36.7114` | `2.197` | `30.016` | `30.635` | `30.764` | `3.266` | `31.148915` |
| TTS_02 | `36.8946` | `2.144` | `30.282` | `30.901` | `31.072` | `3.091` | `31.062058` |
| TTS_03 | `36.6517` | `4.625` | `29.857` | `30.489` | `30.533` | `3.511` | `31.144264` |
| TTDYN_01 | `36.7318` | `2.318` | `30.030` | `30.650` | `30.765` | `3.227` | `31.105222` |
| TTDYN_02 | `36.7354` | `2.024` | `30.050` | `30.668` | `30.801` | `3.203` | `31.122347` |
| SWAC_01 | `36.6436` | `4.650` | `29.821` | `30.452` | `30.486` | `3.441` | `31.160792` |

### 17.4 第三批结论

1. `suppressive + sparse` 没有帮助：真正减少总写入后，rotation / endpoint 仍可变好，但 ATE 回退，说明写入不能只按动态 veto 做稀疏化。
2. `suppressive + wide no-sparse` 有小幅收益，`36.6517m` 比 eta-preserving 的 `36.6566m` 略好；但这个收益仍是厘米级，不能解释 `<30m` 目标。
3. `D_g > 0.5` high-only 和 `D_g ∩ explicit_dyn` 都没改善，说明“动态区域是否准确”仍未解决；explicit dyn 与 ACL2 D_g 的交集过于保守或不对齐。
4. `v_centered` SWA last-layer 写入给出当前 v5 overall best `36.6436 / 6.2450`，但只比 TTS_03 好 `0.0081m`。SWA 写入有一点信号，但不是突破。
5. 当前仍没有任何写入策略接近 `KITTI01 ATE < 30m`，必须继续探索 TTT 更新机制本身，而不是只调 cue map。

---

## 18. 用户新假设：overlap-SWA 与 TTT 更新机制

用户指出两点：

1. SWA 可能应该关注 overlap 帧的动态区域，而不是整个 chunk 的动态区域。
2. 需要深入理解 TTT 更新机制，从机制上寻找写入策略，而不是只做浅层 gate。

### 18.1 机制分析

当前 SWA-write 的问题是空间/时间对象错位：`history` 写入后主要作为下一 chunk 的 previous source，而下一 chunk 与上一 chunk 的强几何关系集中在 tail overlap。之前 `SWA_WRITE_MODE=v_centered` 用整 chunk `D_patch` 重权重，可能把非 overlap 帧的动态判断错误施加到真正需要连续性的 overlap source 上。

当前 TTT-write 的问题是更新幅度控制不一定生效：TTT replay 里的梯度经过 `zeropower_via_newtonschulz5`，会先按梯度矩阵范数归一化。因此单纯缩小 `lambda` 或全局 lr 不一定能缩小 fast-weight 更新步长；token prior 更像改变更新方向，而不是稳定改变更新幅度。

### 18.2 工程补充

新增两个写入机制开关：

- `SWA_WRITE_SCOPE=tail_overlap/head_overlap/both_overlap/all`：只对 overlap 范围内的 SWA history token 施加 write gate，默认仍为 `all`。
- `TTT_WRITE_DELTA_SCALE`：在 TTT replay 完成后，对真实 fast-weight delta `W_new - W_old` 做缩放并重归一化，绕过 zeropower 对全局幅度的吞噬。

验证：

```text
python -m py_compile loger/pipeline/hybrid_memory_controller.py loger/pipeline/ttt_write_controller.py run_pipeline_abc_v2.py
bash -n tools/run_attention_cue_experiment.sh
```

### 18.3 下一批计划

固定 base：

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
success target = KITTI01 ATE < 30m
```

优先跑 6 条：

| Run | 方向 | 参数 | 目的 |
|---|---|---|---|
| ACL2V5_SWOV_01 | overlap SWA | `v_centered`, `scope=tail_overlap`, last, rho0.10 | 验证只看 tail overlap 是否优于整 chunk SWAC_01 |
| ACL2V5_SWOV_02 | overlap SWA | `v_centered`, `scope=tail_overlap`, late, rho0.10 | 多一点 late SWA history，但仍只 gate overlap |
| ACL2V5_SWOV_03 | overlap SWA | `kv_centered`, `scope=tail_overlap`, last, rho0.10 | 测 K/V 同调是否在 overlap 上安全 |
| ACL2V5_TTDS_01 | TTT delta scale | `delta_scale=0.75`, TTS_03 setting | 直接缩小 fast-weight delta |
| ACL2V5_TTDS_02 | TTT delta scale | `delta_scale=0.50`, TTS_03 setting | 更强缩小 delta，测试是否少写动态仍保 continuity |
| ACL2V5_TTDS_03 | TTT delta scale | `delta_scale=1.25`, TTS_03 setting | 反向试探：若欠更新，放大真实 delta |

### 18.4 overlap-SWA / TTT delta-scale 结果

固定协议同 18.3。6 并发 full KITTI01 完成，GPU 0-5，`08:57 -> 09:23`，约 `26 min`。本批继续以 `KITTI01 ATE < 30m` 为成功标准。

| Run | 类型 | 关键参数 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| SWAC_01 reference | centered SWA | `v_centered`, whole chunk, last, rho0.10 | `36.6436` | `6.2450` | `92.4366` | `0.0077` | 上一批 best |
| ACL2V5_SWOV_01 | overlap SWA | `v_centered`, `tail_overlap`, last, rho0.10 | `36.7190` | `6.2841` | `92.4356` | `0.0077` | 比 whole-chunk 回退 |
| ACL2V5_SWOV_02 | overlap SWA | `v_centered`, `tail_overlap`, late, rho0.10 | `36.9965` | `6.3458` | `92.4235` | `0.0077` | late 多层回退明显 |
| ACL2V5_SWOV_03 | overlap SWA | `kv_centered`, `tail_overlap`, last, rho0.10 | `36.6063` | `6.2809` | `92.4378` | `0.0077` | **v5 当前 overall ATE best** |
| ACL2V5_TTDS_01 | TTT delta scale | all branch delta `0.75` | `42.6703` | `10.8762` | `92.8380` | `0.0160` | 崩 |
| ACL2V5_TTDS_02 | TTT delta scale | all branch delta `0.50` | `58.7568` | `26.8513` | `93.3323` | `0.0316` | 严重崩 |
| ACL2V5_TTDS_03 | TTT delta scale | all branch delta `1.25` | `38.0019` | `14.1968` | `92.0854` | `0.0174` | ATE/Rot 均失败 |

Trajectory diagnostics:

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWAC_01 | `36.6436` | `4.650` | `29.821` | `30.452` | `30.486` | `3.441` | `31.160792` |
| TTS_03 | `36.6517` | `4.625` | `29.857` | `30.489` | `30.533` | `3.511` | `31.144264` |
| SWOV_01 | `36.7190` | `4.535` | `29.902` | `30.540` | `30.597` | `3.475` | `31.154632` |
| SWOV_02 | `36.9965` | `3.597` | `30.241` | `30.870` | `30.958` | `3.466` | `31.053300` |
| SWOV_03 | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| TTDS_01 | `42.6703` | `50.024` | `36.652` | `37.507` | `40.002` | `7.686` | `35.663736` |
| TTDS_02 | `58.7568` | `128.468` | `50.259` | `52.001` | `58.404` | `19.788` | `43.092633` |
| TTDS_03 | `38.0019` | `42.242` | `32.268` | `32.810` | `34.528` | `10.003` | `27.996394` |

### 18.5 结论

1. 用户的 overlap-SWA 判断有信号：`tail_overlap + kv_centered + last` 从 SWAC_01 的 `36.6436` 降到 `36.6063`，改善 `0.0372m`，并且 50/100-frame segment mean 也略好。
2. 但 overlap 不是简单替代 whole-chunk：`v_centered tail_overlap` 回退，`late` 多层回退更明显。当前有效点是 **overlap 范围内 K/V 一起 centered，而且只作用 last SWA history layer**。
3. 这支持一个更精确的解释：SWA local memory 不是要全局 veto 动态，而是要在 chunk 接缝附近保持 previous-source 的 K/V 几何一致性；过多层或只改 V 都会破坏 continuity。
4. TTT 全分支 delta-scale 失败非常明确。`0.75/0.50` 直接破坏轨迹，`1.25` 也造成 Rot/Yaw 大幅恶化，说明问题不是“整体写入步长太大/太小”。
5. TTT 的更新机制更像三分支 fast-weight GLU 的方向重构：`zeropower_via_newtonschulz5` 会归一化梯度方向，且每次 update 后按旧权重范数重归一化。因此所有分支一起缩放 delta 会破坏 learned fast-weight manifold。
6. 下一步不再做 all-branch scalar delta sweep，而是做 **分支级 delta / 分支级写入**：优先验证 `w0` gate 分支、`w1` output 分支、`w2` hidden/value 分支谁更适合动态抑制。

---

## 19. TTT/SWA 写入机制深挖计划

### 19.1 新机制开关

为了验证 TTT 更新机制，新增：

```text
TTT_WRITE_DELTA_SCALES=w0,w1,w2
```

它在 TTT replay 后分别缩放 `w0/w1/w2` 的真实 fast-weight delta，并保持各分支旧权重范数。这样可以避免 `TTT_WRITE_DELTA_SCALE` 把三个分支一起扰乱。

机制假设：

- `w0` 更像 gate / dynamic selection 分支，可能适合少写动态。
- `w1` 直接连接 hidden 到 value/output，可能对轨迹尺度和 translation 连续性最敏感，不宜轻易缩放。
- `w2` 参与 hidden-before-mul，可能影响 static geometry feature 的稳定性，强缩放也有风险。

### 19.2 下一批优先候选

固定 base:

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
success target = KITTI01 ATE < 30m
```

| Run | 方向 | 参数 | 目的 |
|---|---|---|---|
| ACL2V5_SWOV_04 | overlap SWA | `kv_centered`, `tail_overlap`, last, rho0.05, min0.90 | 更温和的 overlap KV centered |
| ACL2V5_SWOV_05 | overlap SWA | `kv_centered`, `tail_overlap`, last, rho0.15, min0.80 | 稍强一点确认局部曲线 |
| ACL2V5_SWOV_06 | overlap SWA | `kv_centered`, `both_overlap`, last, rho0.10, min0.85 | 测 head+tail overlap 是否比只 tail 更稳 |
| ACL2V5_SWOV_07 | overlap SWA + TTT | `kv_centered tail last` + TTS_03 suppressive wide | 测 best SWA 是否能和 TTT-only best 叠加 |
| ACL2V5_TTBR_01 | TTT branch delta | `delta_scales=0.75,1,1` | 只缩小 `w0` gate 分支更新 |
| ACL2V5_TTBR_02 | TTT branch delta | `delta_scales=1,0.75,1` | 只缩小 `w1` output 分支更新 |
| ACL2V5_TTBR_03 | TTT branch delta | `delta_scales=1,1,0.75` | 只缩小 `w2` hidden 分支更新 |
| ACL2V5_TTBR_04 | TTT branch delta | `delta_scales=1.25,1,1` | 只放大 `w0`，验证是否欠 gate 更新 |

先跑 1 个 branch-delta smoke，确认新开关生效；full batch 仍控制在 6 并发以内，避免 8 并发 host RAM 风险。

### 19.3 第一批结果：overlap-SWA 曲线与 TTT branch delta

资源记录：

- 6 并发 full KITTI01：GPU 0-5，`09:31:05 -> 09:57:41`，约 `26.6 min`。
- 峰值内存安全但不宽裕：后段 host RAM available 约 `138GiB`，再次确认不要开 8 并发。

Benchmark:

| Run | 类型 | 关键参数 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | overlap SWA | `kv_centered`, `tail_overlap`, last, rho0.10, min0.85 | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_SWOV_04 | overlap SWA | rho0.05, min0.90 | `36.7555` | `6.2878` | `92.4358` | `0.0077` | 太弱，回退 |
| ACL2V5_SWOV_05 | overlap SWA | rho0.15, min0.80 | `36.6228` | `6.3159` | `92.4447` | `0.0077` | 接近 best，但未超过 |
| ACL2V5_SWOV_06 | overlap SWA | `both_overlap`, rho0.10, min0.85 | `36.6963` | `6.3009` | `92.4412` | `0.0077` | head+tail 不如 tail-only |
| TTS_03 reference | TTT suppressive | all branch native delta | `36.6517` | `6.3281` | `92.4351` | `0.0077` | TTT-only reference |
| ACL2V5_TTBR_01 | TTT branch delta | `w0=0.75,w1=1,w2=1` | `37.3276` | `5.8797` | `92.4410` | `0.0075` | Rot 好，但 ATE 回退 |
| ACL2V5_TTBR_02 | TTT branch delta | `w0=1,w1=0.75,w2=1` | `40.2128` | `8.2678` | `92.8175` | `0.0130` | `w1` 不能缩，严重破坏 |
| ACL2V5_TTBR_03 | TTT branch delta | `w0=1,w1=1,w2=0.75` | `37.3360` | `5.5832` | `92.4596` | `0.0075` | Rot 最好，但 ATE 回退 |

Trajectory diagnostics:

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| SWOV_04 | `36.7555` | `4.605` | `29.926` | `30.561` | `30.615` | `3.472` | `31.157250` |
| SWOV_05 | `36.6228` | `3.970` | `29.866` | `30.498` | `30.615` | `3.475` | `31.251147` |
| SWOV_06 | `36.6964` | `4.173` | `29.946` | `30.579` | `30.679` | `3.475` | `31.215951` |
| TTS_03 | `36.6517` | `4.625` | `29.857` | `30.489` | `30.533` | `3.511` | `31.144264` |
| TTBR_01 | `37.3276` | `4.431` | `31.079` | `31.659` | `32.044` | `2.974` | `31.219394` |
| TTBR_02 | `40.2128` | `32.445` | `34.311` | `35.114` | `36.685` | `5.538` | `35.413140` |
| TTBR_03 | `37.3360` | `3.724` | `30.657` | `31.261` | `31.558` | `2.752` | `31.405620` |

结论：

1. SWA overlap 的最优点仍是 `tail_overlap + kv_centered + last + rho0.10/min0.85`。rho 太弱不够，rho 稍强只接近但未过，`both_overlap` 会回退。
2. 这进一步支持“只处理 previous chunk tail overlap”比 head+tail 更合理；SWA history 在下一 chunk 中最关键的是前后 chunk 接缝。
3. TTT 分支级 delta 给出清楚机制信号：`w1` 是最敏感的 output/value 分支，不能随意少写；`w0/w2` 少写会明显改善 rotation/yaw，但 ATE 回退约 `0.7m`。
4. 因此 TTT 的动态少写不是错，而是 **不能全层强缩放**。下一步要做更温和的 `0.90` 分支 delta，以及只在 late TTT 层对 `w0/w2` 做 delta，看看能否保住 ATE 同时拿到 Rot/Yaw 收益。
5. 本批仍未接近 `<30m`，当前 overall best 保持 `SWOV_03 = 36.6063 / 6.2809`。

### 19.4 下一批计划

继续只做 TTT/SWA 写入侧，不回到 read cue 大扫。

| Run | 方向 | 参数 | 目的 |
|---|---|---|---|
| ACL2V5_SWOV_07 | SWA+TTT 组合 | `SWOV_03` + suppressive wide TTT | 测 best overlap-SWA 是否能叠加 TTS_03 |
| ACL2V5_SWOV_08 | SWA+TTT sparse | `SWOV_03` + eta wide sparse0.75 soft | 测 endpoint/rotation 型 sparse 是否能和 overlap-SWA 互补 |
| ACL2V5_TTBR_04 | TTT branch delta | `w0=0.90,w1=1,w2=1` | 温和 gate 分支少写 |
| ACL2V5_TTBR_05 | TTT branch delta | `w0=1,w1=1,w2=0.90` | 温和 hidden 分支少写 |
| ACL2V5_TTBL_01 | TTT branch-layer delta | `w2=0.75`, late layers only | 验证 Rot 收益是否来自 late TTT |
| ACL2V5_TTBL_02 | TTT branch-layer delta | `w0=0.75`, late layers only | 验证 gate 分支 late-only 是否更安全 |

### 19.5 第二批结果：SWA+TTT 叠加与温和 branch delta

资源记录：

- 6 并发 full KITTI01：GPU 0-5，`10:01:08 -> 10:26:13`，约 `25.1 min`。
- 本批继续只探索 TTT/SWA 写入侧；成功标准仍是 `KITTI01 ATE < 30m`。

Benchmark:

| Run | 类型 | 关键参数 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | overlap SWA | `kv_centered`, `tail_overlap`, last, rho0.10, min0.85 | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| TTS_03 reference | TTT suppressive | `alpha=0.35,[0,1.5]`, no sparse | `36.6517` | `6.3281` | `92.4351` | `0.0077` | TTT-only reference |
| ACL2V5_SWOV_07 | SWA+TTT | `SWOV_03` + suppressive wide TTT | `36.6956` | `6.3372` | `92.4401` | `0.0078` | 叠加失败 |
| ACL2V5_SWOV_08 | SWA+TTT sparse | `SWOV_03` + eta wide sparse0.75 soft | `36.6442` | `6.0283` | `92.4341` | `0.0075` | endpoint/Rot 强，但 ATE 未过 |
| ACL2V5_TTBR_04 | TTT branch delta | `w0=0.90,w1=1,w2=1` | `36.8600` | `6.1386` | `92.4364` | `0.0075` | 温和 w0 少写仍回退 |
| ACL2V5_TTBR_05 | TTT branch delta | `w0=1,w1=1,w2=0.90` | `36.8478` | `5.9748` | `92.4441` | `0.0074` | Rot/endpoint 好，ATE 回退 |
| ACL2V5_TTBL_01 | TTT late branch delta | late-only `w2=0.75` | `37.2314` | `5.5618` | `92.4590` | `0.0075` | late-only 保留 Rot 收益但 ATE 差 |
| ACL2V5_TTBL_02 | TTT late branch delta | late-only `w0=0.75` | `37.3773` | `5.8803` | `92.4406` | `0.0075` | late-only ATE 仍差 |

Trajectory diagnostics:

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| SWOV_07 | `36.6956` | `4.632` | `29.897` | `30.532` | `30.592` | `3.507` | `31.202280` |
| SWOV_08 | `36.6442` | `1.933` | `29.989` | `30.607` | `30.756` | `3.194` | `31.137054` |
| TTBR_04 | `36.8600` | `2.267` | `30.212` | `30.830` | `30.962` | `3.297` | `31.156082` |
| TTBR_05 | `36.8478` | `2.375` | `30.062` | `30.686` | `30.793` | `3.169` | `31.237630` |
| TTBL_01 | `37.2314` | `3.970` | `30.591` | `31.193` | `31.502` | `2.730` | `31.399108` |
| TTBL_02 | `37.3773` | `4.630` | `31.157` | `31.734` | `32.124` | `2.973` | `31.211327` |

结论：

1. `SWOV_03` 与 suppressive TTT 没有叠加，说明这两条都在修同一类接缝/动态污染，直接相乘会过约束。
2. `SWOV_08` 的 FinalErr=`1.933m` 和 Rot=`6.0283` 很强，但 ATE/segment mean 回退；它是 endpoint/rotation diagnostic，不是主线。
3. 温和 branch delta `0.90` 仍没有保住 ATE；`w0/w2` 少写持续改善 endpoint/Rot，但会牺牲全局 ATE。
4. late-only `w0/w2` 缩放说明 rotation 收益确实可来自 TTT 后层，但 ATE 回退更明显，不能作为默认写入策略。
5. 当前 overall best 仍是 `SWOV_03 = 36.6063 / 6.2809`，离 `<30m` 还很远。

---

## 20. 用户新思考：overlap history 与 TTT 更新机制再挖

用户提出：

1. SWA 可能要关注 overlap 帧的动态区域，而不是 chunk 的动态区域。
2. 需要深入理解 TTT 更新机制，看是否能从机制上设计写入方式。

### 20.1 机制判断

SWA 之前的 `SWA_WRITE_SCOPE=tail_overlap` 只是在当前 chunk 的 SWA history token 上调 K/V 权重，但 **非 overlap token 仍然完整写进下一 chunk 的 history**。如果污染源是“下一 chunk 读取了上一 chunk 非接缝帧的动态 KV”，那么只 gate overlap 不够，需要直接控制 history 写入范围。

因此新增：

```text
SWA_WRITE_KEEP_SCOPE=all/tail_overlap/head_overlap/both_overlap
```

语义：

- `SWA_WRITE_SCOPE`：对哪些 token 做 K/V 权重 gate；
- `SWA_WRITE_KEEP_SCOPE`：哪些 token 真正保留进下一 chunk 的 SWA KV history。

TTT 机制方面，`fast_weight_replay_update` 不是普通 SGD：

- `lr * token_prior` 先进入每个分支的梯度聚合；
- 梯度随后经过 `zeropower_via_newtonschulz5`，幅度被近似归一化为方向更新；
- `w0/w1/w2` 每次更新后都被拉回旧权重范数；
- 因此 prior 主要改变 **方向、分支耦合和 token 集合**，不是简单改变步长。

这解释了当前现象：

- all-branch delta scale 会破坏 fast-weight manifold；
- `w1` 作为 output/value 分支极敏感，不能少写；
- `w0/w2` 少写能改善 Rot/Yaw，但会伤害 ATE，说明 orientation 与 translation/global scale 的写入需求不同。

### 20.2 工程验证

新增参数：

```text
--swa_write_keep_scope
SWA_WRITE_KEEP_SCOPE
```

验证命令：

```text
python3 -m py_compile loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py loger/pipeline/ttt_write_controller.py
bash -n tools/run_attention_cue_experiment.sh
```

### 20.3 下一批计划：真正裁剪 SWA history

固定 base:

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
success target = KITTI01 ATE < 30m
```

| Run | 方向 | 参数 | 目的 |
|---|---|---|---|
| ACL2V5_SWTR_01 | SWA history truncate | `keep=tail_overlap`, no K/V gate, all SWA layers | 直接验证只写 overlap history 是否有帮助 |
| ACL2V5_SWTR_02 | SWA history truncate | `keep=tail_overlap`, no K/V gate, last SWA layer | 对齐 SWOV_03 的 last-layer 假设 |
| ACL2V5_SWTR_03 | SWA truncate + gate | `keep=tail_overlap`, `kv_centered`, rho0.10/min0.85, last | 验证“只写 tail + tail 内动态 K/V 调制” |
| ACL2V5_SWTR_04 | SWA truncate + gate | `keep=tail_overlap`, `kv_centered`, rho0.10/min0.85, all | 测多层裁剪是否过强 |
| ACL2V5_SWTR_05 | SWA truncate + sparse TTT | SWTR_03 + eta wide sparse0.75 soft | 测 endpoint/rotation 型 TTT 是否与裁剪互补 |
| ACL2V5_SWTR_06 | SWA truncate + suppressive TTT | SWTR_03 + suppressive wide TTT | 测 TTS_03 是否与真正 overlap history 互补 |

本批如果仍然无法接近 `<30m`，下一步不要继续小幅调 rho，而应转向 TTT replay token 集合本身：例如只让 overlap/static tokens replay TTT、对 `w0/w2` 使用不同 token 集合、或把 explicit dyn cue 用作 TTT token-level hard veto 的辅助但保留 `w1` native。

### 20.4 第一批结果：真正裁剪 SWA history

资源记录：

- smoke `END_FRAME=128` 已验证 `SWA_WRITE_KEEP_SCOPE=tail_overlap` 生效：`swa_write_history_tokens_before=15120`，`after=3834`，下一 chunk 的 `max_history_tokens` 降到 overlap 量级。
- full batch 用 GPU 0-3，`10:36:16 -> 11:00:52`，约 `24.6 min` 完成 4 个 KITTI01 full run。
- GPU 4-7 当时被外部任务占用，因此没有继续加并发。

Benchmark:

| Run | 类型 | 关键参数 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | overlap SWA gate | `keep=all`, `kv_centered tail_overlap last`, rho0.10/min0.85 | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_SWTR_01 | SWA history truncate | `keep=tail_overlap`, no K/V gate, all layers | `38.4759` | `8.3414` | `92.4286` | `0.0083` | 明显回退 |
| ACL2V5_SWTR_02 | SWA history truncate | `keep=tail_overlap`, no K/V gate, last layer | `38.1725` | `8.8541` | `92.4212` | `0.0089` | 本批 ATE 最好，但仍大幅差于 SWOV_03 |
| ACL2V5_SWTR_03 | SWA truncate + gate | `keep=tail_overlap`, `kv_centered`, rho0.10/min0.85, last | `38.2174` | `8.8803` | `92.4243` | `0.0089` | gate 后没有补回 ATE |
| ACL2V5_SWTR_04 | SWA truncate + gate | `keep=tail_overlap`, `kv_centered`, rho0.10/min0.85, all | `38.8543` | `8.0553` | `92.4186` | `0.0082` | Rot 相对好，但 ATE 最差 |

Trajectory diagnostics:

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| SWTR_01 | `38.4759` | `2.505` | `33.478` | `34.002` | `34.068` | `4.964` | `31.085491` |
| SWTR_02 | `38.1725` | `3.704` | `33.207` | `33.717` | `33.779` | `5.390` | `30.999550` |
| SWTR_03 | `38.2174` | `3.854` | `33.253` | `33.766` | `33.868` | `5.425` | `31.037216` |
| SWTR_04 | `38.8543` | `3.271` | `34.003` | `34.505` | `34.722` | `4.623` | `31.006935` |

结论：

1. “只保留 overlap history”这个强裁剪失败：四个 run 都回到 `38.17-38.85m`，比 SWOV_03 差 `1.57-2.25m`。
2. 这说明 SWA 非 overlap history 仍携带重要几何上下文；污染源不是“所有非 overlap 历史”，而更像是 overlap 接缝里的动态 token 与下一 chunk 反复交互。
3. 用户提出“关注 overlap 帧动态区域而不是 chunk 动态区域”是对的，但实现上应是 **保留完整 history，只对 overlap source 的动态 K/V 做软调制**，而不是删掉非 overlap history。
4. SWTR_03 没有通过 read/write gate，因此不继续跑 SWTR_05/06，避免再花两轮 full run 验证明显失败的 base。
5. 下一步转向 TTT 更新机制：当前脚本默认 `PRIOR_BRANCH_MASK=0`，也就是说 `stage_d_x_dg_inv_sqrt` 的动态少写主要作用在 `w0` replay；之前的 branch-delta 是 replay 后缩放，不等价于让 `w2` 的梯度聚合避开动态 token。因此应直接测试 `w2` 或 `w0+w2` token prior，并用 explicit dyn / union dyn 做更准确的动态 veto。

### 20.5 下一批计划：TTT replay token 集合与 explicit dyn 写入 veto

固定 base:

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
success target = KITTI01 ATE < 30m
```

机制假设：

- `w1` 必须基本保持 native replay，因为它是 output/value 分支，少写会破坏 translation/scale。
- `w0` 的动态少写已验证有一定效果，但单独压 `w0` 不够。
- `w2` 不能只做 post-delta scale；要让 token prior 进入 `lr2 * prior` 的梯度聚合，才是真正改变 TTT 更新方向。
- explicit dyn 可以作为动态 veto 辅助，但不能完全替代 C23 D_g；优先测 `D_g`、`explicit_dyn`、`max(D_g, explicit_dyn)` 三类写入 mask。

| Run | 方向 | 参数 | 目的 |
|---|---|---|---|
| ACL2V5_TTBM_01 | TTT branch mask | `PRIOR_BRANCH_MASK=2`, write=`stage_d_x_dg_inv_sqrt` | 只让 `w2` 避开 D_g 动态 token，保留 `w0/w1` native |
| ACL2V5_TTBM_02 | TTT branch mask | `PRIOR_BRANCH_MASK=0,2`, write=`stage_d_x_dg_inv_sqrt` | 同时约束 gate 分支和 hidden 分支 |
| ACL2V5_TTBM_03 | explicit dyn veto | `PRIOR_BRANCH_MASK=2`, write=`stage_d_x_exp_inv_sqrt` | 验证 explicit dyn 是否比 D_g 更适合保护 `w2` |
| ACL2V5_TTBM_04 | explicit dyn veto | `PRIOR_BRANCH_MASK=0,2`, write=`stage_d_x_exp_inv_sqrt` | explicit dyn 同时作用 `w0/w2` |
| ACL2V5_TTBM_05 | union dyn veto | `PRIOR_BRANCH_MASK=2`, write=`stage_d_x_union_dyn_inv` | D_g 和 explicit dyn 任一认为动态就少写 `w2` |
| ACL2V5_TTBM_06 | union dyn veto | `PRIOR_BRANCH_MASK=0,2`, write=`stage_d_x_union_dyn_inv` | 更强 union veto，确认是否过约束 |

调度：

- 先跑 4 并发 `TTBM_01-04`；如果没有明显接近 SWOV_03 或 `<30m`，再跑 `TTBM_05-06`。
- 如果 `TTBM_01/03/05` 中有一个明显优于 SWOV_03，再把同一写法叠加 `SWOV_03` 的 overlap K/V soft gate；否则不做 SWA+TTT 叠加。

### 20.6 TTBM 结果：branch token-prior 与 explicit / union dyn veto

固定协议：

```text
seq = KITTI01 full
read cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
success target = KITTI01 ATE < 30m
```

Benchmark:

| Run | Branch prior | Write score | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | default | `stage_d_x_dg_inv_sqrt` + SWA tail KV | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_TTBM_01 | `w2` | `stage_d_x_dg_inv_sqrt` | `38.5226` | `8.7884` | `92.3944` | `0.0087` | ATE 明显回退 |
| ACL2V5_TTBM_02 | `w0+w2` | `stage_d_x_dg_inv_sqrt` | `38.4123` | `8.7005` | `92.3958` | `0.0087` | 仍回退 |
| ACL2V5_TTBM_03 | `w2` | `stage_d_x_exp_inv_sqrt` | `38.4615` | `8.6950` | `92.3941` | `0.0086` | explicit dyn 没有帮到 ATE |
| ACL2V5_TTBM_04 | `w0+w2` | `stage_d_x_exp_inv_sqrt` | `38.4648` | `8.7300` | `92.3931` | `0.0087` | endpoint 尚可但 ATE 回退 |
| ACL2V5_TTBM_05 | `w2` | `stage_d_x_union_dyn_inv` | `38.5047` | `8.8546` | `92.3940` | `0.0088` | union dyn 更差 |
| ACL2V5_TTBM_06 | `w0+w2` | `stage_d_x_union_dyn_inv` | `38.4906` | `8.7431` | `92.3938` | `0.0087` | union dyn 仍失败 |

Trajectory diagnostics:

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTBM_01 | `38.5226` | `3.492` | `33.473` | `33.977` | `33.924` | `5.305` | `30.736129` |
| TTBM_02 | `38.4123` | `3.477` | `33.319` | `33.821` | `33.776` | `5.229` | `30.744998` |
| TTBM_03 | `38.4615` | `3.238` | `33.360` | `33.866` | `33.823` | `5.225` | `30.730084` |
| TTBM_04 | `38.4648` | `3.049` | `33.403` | `33.905` | `33.878` | `5.257` | `30.727775` |
| TTBM_05 | `38.5047` | `3.590` | `33.482` | `33.986` | `33.943` | `5.369` | `30.730167` |
| TTBM_06 | `38.4906` | `3.212` | `33.441` | `33.948` | `33.899` | `5.285` | `30.726276` |

结论：

1. 直接把 dynamic/static prior 换到 `w2` 或 `w0+w2` 的 token learning-rate 上，没有产生突破；所有 TTBM 都回到 `38.41-38.52m`。
2. explicit dyn / union dyn 作为 TTT 写入 veto 没有优于 C23 D_g，说明显式动态区域要么不够准，要么它描述的是 object-motion，不等价于 TTT fast-weight 更新里真正 harmful 的 token。
3. `w1` 仍应尽量保持 native replay；涉及 `w1` 的后续实验只作为强 diagnostic，不作为默认主线。
4. 这一批不触发与 SWOV_03 的叠加；下一步转向 TTT token 集合本身，而不是继续换 dynamic mask。

### 20.7 TTSC 结果：hard tail-overlap TTT replay

工程变更：

- `TTT_WRITE_TOKEN_SCOPE=tail_overlap/head_overlap/both_overlap/all`
- `TTT_WRITE_TOKEN_SCOPE_FLOOR`

`tail_overlap` 的含义是：只让当前 chunk 的最后 overlap 帧 token 参与 enabled branch 的 prior-weighted replay。smoke 显示 hard tail 时 scope mass 只有 `0.09375`，也就是 32 帧 chunk 中仅 3 帧参与 enabled branch 的 TTT 更新。

Benchmark:

| Run | Token scope | Branch prior | Prior policy | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---|---:|---:|---:|---:|---|
| C23 past reference | all | default branch0 | eta mean preserving | `38.3706` | `8.6694` | `92.3930` | `0.0086` | v4 locked |
| SWOV_03 reference | all + SWA tail KV | default branch0 | eta mean preserving | `36.6063` | `6.2809` | `92.4378` | `0.0077` | v5 overall best |
| ACL2V5_TTSC_01 | hard `tail_overlap` | `w0+w2` | suppressive | `43.1690` | `7.5000` | `92.6465` | `0.0109` | ATE 崩，scale/final 明显坏 |
| ACL2V5_TTSC_02 | hard `tail_overlap` | `w0+w1+w2` | suppressive | `80.8536` | `31.5660` | `93.7970` | `0.0400` | 严重崩坏，确认不能少写 `w1` |

Trajectory diagnostics:

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTSC_01 | `43.1690` | `30.168` | `37.951` | `38.572` | `40.569` | `3.539` | `33.567347` |
| TTSC_02 | `80.8536` | `148.848` | `72.123` | `74.565` | `80.682` | `23.034` | `53.019766` |

结论：

1. “只用 overlap token 做 TTT 写入”太硬，直接破坏全局 scale / translation；TTT fast weights 仍需要 chunk 内非 overlap token 提供低频几何背景。
2. `w1` 不能少写这一点被再次确认：branchAll hard tail 直接崩到 `80.85m`。
3. 但这个实验也给出启发：TTT 的关键不是简单“少写动态”，而是 **让 overlap/static token 提供方向，同时保留非 overlap 的弱更新背景**。
4. 因此新增 `TTT_WRITE_TOKEN_SCOPE_FLOOR`，下一批不再 hard zero 非 overlap，而测 floor `0.25/0.50`。

### 20.8 下一批：TTT floor 与 SWA overlap 动态源

当前新假设：

1. TTT 写入应避免 hard tail，只对非 overlap token 降权而不是清零；
2. SWA 写入应关注 overlap source 的动态区域，但动态源未必应该是 C23 D_g，需测试 explicit dyn / union dyn 是否更准；
3. 如果 floor 或 explicit/union SWA 仍不能接近 SWOV_03，则说明当前瓶颈不是动态 mask 精度，而是 TTT/SWA 写入形式本身。

新增工程开关：

```text
TTT_WRITE_TOKEN_SCOPE_FLOOR
SWA_WRITE_SCORE_SOURCE=read/explicit_dyn/old_dyn/union_dyn/intersection
```

运行矩阵：

| Run | 方向 | 参数 | 目的 |
|---|---|---|---|
| ACL2V5_TTFL_01 | TTT soft scope | `tail_overlap`, floor `0.25`, branch `w0+w2`, suppressive | 验证弱保留非 overlap 是否避免 hard tail 崩坏 |
| ACL2V5_TTFL_02 | TTT soft scope | `tail_overlap`, floor `0.50`, branch `w0+w2`, suppressive | 测更温和 floor |
| ACL2V5_SWSC_01 | SWA overlap score source | `kv_centered tail_overlap last`, score=`explicit_dyn` | 验证 explicit dyn 是否更适合作 SWA overlap 动态源 |
| ACL2V5_SWSC_02 | SWA overlap score source | `kv_centered tail_overlap last`, score=`union_dyn` | 验证 D_g 与 explicit dyn 的 union 是否更安全 |

### 20.9 TTFL / SWSC 结果：floor TTT 与 explicit/union SWA source

执行偏差说明：

- 这一批 run 名称里写了 `C23pairall`，但实际启动环境缺少 `READ_LAYER_MODE=all / FRAME_BIAS_MODE=pair`，因此走的是脚本默认 early-layer read path。
- 这批结果只作为机制 diagnostic，不作为 v5 pair/all 主线排名。

Benchmark:

| Run | 方向 | 关键参数 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| ACL2V5_TTFL_01 | TTT floor | `tail_overlap`, floor `0.25`, branch `w0+w2`, suppressive | `37.5105` | `9.3898` | `92.4042` | `0.0095` | 比 hard tail 安全，但仍明显不如 SWOV_03 |
| ACL2V5_TTFL_02 | TTT floor | `tail_overlap`, floor `0.50`, branch `w0+w2`, suppressive | `37.9296` | `9.1874` | `92.4005` | `0.0093` | 更温和 floor 仍回退 |
| ACL2V5_SWSC_01 | SWA source | `tail_overlap KV centered`, score=`explicit_dyn` | `38.1502` | `8.6332` | `92.3964` | `0.0086` | explicit dyn source 未超过 D_g/SWOV |
| ACL2V5_SWSC_02 | SWA source | `tail_overlap KV centered`, score=`union_dyn` | `38.2132` | `8.5590` | `92.3967` | `0.0086` | union dyn 也没有帮助 ATE |

Trajectory diagnostics:

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTFL_01 | `37.5105` | `9.145` | `32.154` | `32.652` | `32.525` | `5.841` | `30.768185` |
| TTFL_02 | `37.9296` | `7.558` | `32.654` | `33.170` | `33.036` | `5.670` | `30.762074` |
| SWSC_01 | `38.1502` | `3.580` | `33.040` | `33.551` | `33.522` | `5.259` | `30.751194` |
| SWSC_02 | `38.2132` | `3.204` | `33.112` | `33.620` | `33.601` | `5.181` | `30.755075` |

结论：

1. TTT floor 比 hard tail 好很多，但它仍然是在“突出 tail overlap token、削弱非 overlap 几何背景”，方向不完全符合当前假设。
2. explicit dyn / union dyn 没有直接解决 SWA overlap 动态源精度问题；显式动态更像 object-motion cue，不等价于 SWA/TTT 写入里真正 harmful 的 token。
3. 当前更合理的 TTT 思路不是 `tail_overlap` floor，而是 **tail-overlap veto**：非 overlap token 保持 native replay，只在接缝 overlap 动态 token 上少写。
4. 当前更合理的 SWA 思路不是删 history，也不是整块 chunk gate，而是 **current head-overlap query 读取 previous tail-overlap source** 时抑制动态 source。

### 20.10 TTT/SWA 写入机制重排：overlap veto 与低内存 SWA overlap bias

用户补充判断：

1. SWA 应关注 overlap 帧的动态区域，而不是整个 chunk 的动态区域。
2. TTT 要从更新机制出发：`w0/w1/w2` 的 fast-weight replay 更新经过 `zeropower` 与 branch norm restoration，token prior 主要改变更新方向，而不是简单缩小更新幅度。

机制分析：

- `w1` 仍应保持 native replay；此前 hard tail 的 `w0+w1+w2` 直接崩到 `80.8536m`，说明 value/output 分支不能少写。
- `w0/w2` 可以探索，但应保留非 overlap 的低频几何 token；真正可疑的是 overlap seam 里重复进入相邻 chunk 的动态 token。
- TTT 的 `apply` 发生在 `update` 之前：当前 chunk 的几何输出使用旧 fast weights，写入策略主要影响 **下一 chunk** 的 fast weights，而不是当前 chunk 立即生效。
- chunk merge 使用 tail-trim：非末 chunk 的 tail overlap 输出会被丢掉，但这些 tail overlap token 仍参与 TTT update / SWA history commit，因此它们是最可疑的“只写入不出图”的污染入口。
- head overlap 则参与当前 chunk 对齐与输出前段，更多是读上一 chunk history 的 seam；它更适合作 SWA read-bias 或 query-side diagnostic，不应先用 hard drop 破坏。
- 因此新增 `tail_overlap_veto / head_overlap_veto / both_overlap_veto`：非 overlap prior 固定为 `1.0`，只在 overlap 范围内按动态 prior 少写。
- SWA overlap bias 第一版构造完整 `[current_tokens, history+current]` bias，KITTI01 full 首段需要额外约 `12GB` 显存，`SWAQ_01/02` OOM。已改成 compact descriptor：先跑 native full attention，再只对 overlap query rows 分块重算并替换输出，默认 block=`128`。

当前正在运行的 pair/all full run：

| Run | 方向 | 关键参数 | 状态 |
|---|---|---|---|
| ACL2V5_TTVT_01 | TTT tail-overlap veto | branch `w0`, `eta_mean_preserving`, beta `4.75` | running |
| ACL2V5_TTVT_02 | TTT tail-overlap veto | branch `w0`, suppressive wide, beta `4.75` | running |
| ACL2V5_SWAQ2_01 | SWA overlap compact bias | `pair` mode, last SWA layer, beta `1.0`, min_keep `0.70` | running |
| ACL2V5_SWAQ2_02 | SWA overlap compact bias | `source` mode, last SWA layer, beta `1.0`, min_keep `0.70` | running |

继续规则：

1. 若 `TTVT/SWAQ2` 未接近或超过 SWOV_03，则继续探索 TTT/SWA 写入形式，不因单批失败停止。
2. 若 SWA overlap compact bias 有正信号，再与 SWOV_03 的 `tail_overlap KV centered` 写入 gate 叠加。
3. 若 TTT tail-overlap veto 有正信号，再测试 `head_overlap_veto / both_overlap_veto` 与 `w0+w2` 的温和版本。
4. 成功标准仍是 `KITTI01 ATE < 30m`；当前 overall best `SWOV_03 = 36.6063 / 6.2809`，尚未达标。

### 20.11 TTVT / SWAQ2 结果：轻 veto 与 read-side overlap bias 未过 gate

这一批继续只看 TTT/SWA 写入机制，不再回到 read cue 大扫。注意：`TTVT/SWAQ2` 启动时没有显式覆盖 `WRITE_ALPHA`，因此使用脚本默认 `0.10`，本批作为机制 diagnostic；后续主线 pair/all 对照统一显式设回 `WRITE_ALPHA=0.125`。

#### TTT tail-overlap veto

设计目的：非 overlap token 保持 native replay，只在 tail overlap 动态 token 上少写，避免此前 hard tail scope 把低频几何背景清掉。

| Run | TTT 写入策略 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | overlap SWA `kv_centered tail last` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_TTVT_01 | `tail_overlap_veto`, branch `w0`, eta mean-preserving | `36.7654` | `6.3741` | `92.4379` | `0.0078` | 回退 |
| ACL2V5_TTVT_02 | `tail_overlap_veto`, branch `w0`, suppressive wide | `36.7501` | `6.2917` | `92.4367` | `0.0077` | Rot 接近，但 ATE 未过 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| TTVT_01 | `4.829` | `29.969` | `30.609` | `30.663` | `3.550` | `31.178002` |
| TTVT_02 | `3.932` | `29.989` | `30.626` | `30.705` | `3.468` | `31.168318` |

结论：

1. tail-overlap veto 比 hard tail replay 安全得多，但它对 harmful TTT 写入的抑制不够，ATE 仍比 SWOV_03 差 `0.14-0.16m`。
2. `TTVT_02` final error 更好，但 segment mean 全部回退，说明单纯轻 veto 更像 endpoint/rotation diagnostic，不是主线突破。
3. 下一步需要更直接地测试“overlap token 不写入 TTT update”：新增 `tail_overlap_drop / head_overlap_drop / both_overlap_drop`，非 overlap 保持 native replay，overlap token 按 `TTT_WRITE_TOKEN_SCOPE_FLOOR` 降到 0 或小值。

#### SWA overlap read-side bias

设计目的：只在 current head-overlap query 读取 previous tail-overlap source 时，对动态 source 加 attention bias，而不是改整块 chunk 或裁剪 history。

第一版 dense bias 因为要构造 `[current_tokens, history+current]` 浮点 mask，KITTI01 full 首段额外分配约 `12GB`，`SWAQ_01/02` OOM。已改为 compact descriptor：先跑 native full attention，再只重算 overlap query rows，默认 block size `128`。

| Run | SWA read-side bias | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | write-side `kv_centered tail last` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_SWAQ2_01 | compact overlap bias, `pair`, beta `1.0`, min keep `0.70` | `36.6989` | `6.3814` | `92.4377` | `0.0078` | 未过 gate |
| ACL2V5_SWAQ2_02 | compact overlap bias, `source`, beta `1.0`, min keep `0.70` | `36.6984` | `6.3810` | `92.4374` | `0.0078` | 与 pair 几乎等价 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| SWAQ2_01 | `4.933` | `29.903` | `30.541` | `30.583` | `3.566` | `31.168721` |
| SWAQ2_02 | `4.930` | `29.903` | `30.541` | `30.583` | `3.565` | `31.168768` |

结论：

1. compact overlap-bias 工程路径可用，解决了 dense OOM，但 read-side SWA overlap bias 没有超过 write-side SWOV_03。
2. `pair` 与 `source` 几乎重合，说明这一路主要受 source 动态项控制；query-side overlap 动态并不是主要增益来源。
3. 暂不把 SWAQ2 与 SWOV_03 叠加；下一步优先继续写入侧：TTT overlap drop，以及 SWA tail-overlap source 的 K/V 写入形态。

#### 当前补跑：TTT overlap drop

这批按用户最新判断继续验证“overlap 帧动态区域少写”，但不清掉非 overlap 几何背景。

| Run | TTT 写入策略 | 关键参数 | 状态 |
|---|---|---|---|
| ACL2V5_TTDO_01b | `tail_overlap_drop` | branch `w0`, eta mean-preserving, `WRITE_ALPHA=0.125`, beta `4.75`, floor `0.0` | done |
| ACL2V5_TTDO_02b | `both_overlap_drop` | branch `w0`, eta mean-preserving, `WRITE_ALPHA=0.125`, beta `4.75`, floor `0.0` | done |

如果 `TTDO` 仍不接近 SWOV_03，下一批不要再小调 veto/drop 强度，而转向 SWA 写入侧更大胆的 overlap-source K/V 方案：`tail_overlap` source 上直接 `kv` gate、`kv_centered` sparse gate、以及 read cue 与 explicit dyn 的 intersection gate。

补充 TTT 机制判断：

- 当前 `TTDO_01b/02b` 先测 branch `w0`，因为它改 gate 分支，风险较低。
- 但从更新式看，`w1_grad=(hidden * lr1)^T @ v` 是最直接把 token value 写进 fast weight 的分支；如果 tail overlap 动态 token 是污染入口，只动 `w0` 未必够。
- 早先 `w1` hard-tail 崩坏不等于 `w1` 不能碰，而是不能清掉非 overlap 的 `w1` 几何背景。因此若 `w0 tail/both drop` 未过 gate，下一批应测：
  - `tail_overlap_drop + PRIOR_BRANCH_MASK=1`：只 drop tail overlap 的 `w1`，保留非 overlap native；
  - `tail_overlap_drop + PRIOR_BRANCH_MASK=0,1`：同时控制 gate 与 value/output 写入；
  - 若 branch1 有正信号，再小心测试 `0,1,2`，否则不碰全分支。

TTDO 结果：

| Run | TTT 写入策略 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | overlap SWA `kv_centered tail last` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_TTDO_01b | `tail_overlap_drop`, branch `w0`, eta | `37.7686` | `5.7000` | `92.4637` | `0.0081` | Rot 强，但 ATE 大幅回退 |
| ACL2V5_TTDO_02b | `both_overlap_drop`, branch `w0`, eta | `37.9466` | `5.9178` | `92.4400` | `0.0088` | ATE 更差 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| TTDO_01b | `9.707` | `31.580` | `32.165` | `32.804` | `2.803` | `31.463937` |
| TTDO_02b | `15.728` | `31.851` | `32.410` | `33.329` | `3.011` | `31.235254` |

TTDO 结论：

1. 只 drop overlap seam 的 branch `w0` 仍明显破坏 ATE / segment mean，虽然 rotation/yaw 更干净。这说明 gate 分支少写会改变 fast-weight 更新方向，但不能解决主漂移。
2. `both_overlap_drop` 更差，进一步支持 tail seam 是重点，head overlap 不宜先做 hard drop。
3. 下一批已经切到 branch `w1` 与 `w0+w1` tail-overlap drop：如果 value/output 分支的 tail 动态污染是主因，应比 branch `w0` 更接近 SWOV_03；如果仍失败，则 TTT drop 方向判为不适合主线。

TTDV 结果：

| Run | TTT 写入策略 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| ACL2V5_TTDV_01 | `tail_overlap_drop`, branch `w1`, eta | `56.8369` | `23.1532` | `93.6345` | `0.0274` | value/output tail hard drop 崩 |
| ACL2V5_TTDV_02 | `tail_overlap_drop`, branch `w0+w1`, eta | `60.7833` | `26.7913` | `93.6446` | `0.0308` | 更差 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| TTDV_01 | `89.545` | `50.352` | `52.324` | `56.232` | `17.169` | `49.431247` |
| TTDV_02 | `107.026` | `53.736` | `55.870` | `60.605` | `19.810` | `49.567174` |

TTDV 结论：

1. `w1` tail-overlap hard drop 是灾难级失败，说明 tail seam 的 value/output 更新虽然可疑，但不能直接移除；它仍承载跨 chunk continuity。
2. `w0+w1` 更差，确认 TTT overlap hard-drop 方向不适合作主线。
3. TTT 写入侧下一步不再做 overlap hard drop；如果继续动 TTT，应转向更结构化的 update mixing / replay curriculum，而不是 token hard veto。当前优先级转回 SWA 写入集合和 overlap-source K/V 细化。

#### 并行补跑：SWA overlap-source 写入

根据“**SWA 可能要关注 overlap 帧的动态区域，而不是 chunk 的动态区域**”这一判断，GPU2/3 并行启动两条不依赖 TTDO 结果的写入侧实验。它们都保留完整 SWA history，只改 previous tail-overlap source 的 K/V 写入形态。

| Run | SWA 写入策略 | 关键参数 | 状态 |
|---|---|---|---|
| ACL2V5_SWOD_01 | `tail_overlap` source 直接 `kv` gate | `SWA_WRITE_MODE=kv`, rho `0.10`, min gate `0.85`, last layer, score=`read` | done |
| ACL2V5_SWOD_02 | `tail_overlap` source `kv_centered` sparse gate | `SWA_WRITE_MODE=kv_centered`, sparse `0.75`, rho `0.10`, min gate `0.85`, last layer, score=`read` | done |

判定规则：若 `SWOD_01` 过 `SWOV_03`，说明 centered 重权重不是必要条件，直接少写 dynamic K/V 更有效；若 `SWOD_02` 过，说明 SWOV_03 的有效性来自 overlap-source 内更稀疏的 static token 保留；若两者都不行，下一步再测 `intersection(read, explicit_dyn)`，避免 explicit dyn 的 false positive 过多破坏几何 history。

SWOD 第一批结果：

| Run | SWA 写入策略 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | `kv_centered tail_overlap`, sparse `1.0`, score=`read` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_SWOD_01 | direct `kv`, tail overlap, score=`read` | `36.8064` | `6.3377` | `92.4364` | `0.0078` | 直接削 K/V 过硬，回退 |
| ACL2V5_SWOD_02 | `kv_centered`, sparse `0.75`, tail overlap, score=`read` | `36.6702` | `6.3814` | `92.4414` | `0.0078` | 比 direct kv 好，但仍未过 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| SWOD_01 | `4.841` | `29.996` | `30.636` | `30.675` | `3.511` | `31.156572` |
| SWOD_02 | `4.589` | `29.903` | `30.538` | `30.627` | `3.544` | `31.219202` |

SWOD 第一批结论：

1. 直接 `kv` gate 明显不如 centered；SWA history 不能简单少写 K/V，否则 continuity 变差。
2. sparse `0.75` 没有超过 SWOV_03，说明 SWOV_03 的增益不是靠更稀疏的 static token，而是靠温和的 centered 重权重。
3. 已启动第二批：
   - `SWOD_03`: tail overlap `k`-only gate，保留 V 内容，只弱化动态 source 的可检索性；
   - `SWOD_04`: tail overlap `kv_centered`，score=`intersection(read, explicit_dyn)`，减少 explicit dyn false positive。

SWOD 第二批结果：

| Run | SWA 写入策略 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| ACL2V5_SWOD_03 | `k`-only, tail overlap, score=`read` | `36.7486` | `6.3289` | `92.4356` | `0.0078` | 只弱化可检索性仍回退 |
| ACL2V5_SWOD_04 | `kv_centered`, tail overlap, score=`intersection` | `36.7068` | `6.4538` | `92.4417` | `0.0079` | intersection 更保守但未过 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| SWOD_03 | `4.921` | `29.935` | `30.572` | `30.597` | `3.518` | `31.151883` |
| SWOD_04 | `5.200` | `29.932` | `30.579` | `30.634` | `3.617` | `31.217197` |

SWOD 第二批结论：

1. `k`-only 没有解决问题，说明仅降低 dynamic source 被检索到的概率不足以改善 ATE。
2. `intersection(read, explicit_dyn)` 也没有超过 read-only source，说明 explicit dyn 在 SWA 写入侧更像保守过滤器，不能提供更准的 harmful-source 定位。
3. SWOD 四条都未过 SWOV_03；SWA 写入侧当前仍以 `kv_centered tail_overlap score=read` 为局部最优。
4. 下一步转到 SWKP：不再调 K/V 权重，而是测试 overlap history 去重。

#### SWA history overlap 去重补跑

由于 chunk merge 会裁掉非末 chunk tail overlap，但 SWA history 会继续写入 overlap source，新增 `SWA_WRITE_KEEP_SCOPE=exclude_*`，测试不是“只保留 overlap”，而是“从完整 history 中移除重复 overlap slice”：

| Run | SWA 写入集合策略 | 关键参数 | 状态 |
|---|---|---|---|
| ACL2V5_SWKP_01 | exclude head overlap | `SWA_WRITE_MODE=none`, `SWA_WRITE_KEEP_SCOPE=exclude_head_overlap`, last layer | running |
| ACL2V5_SWKP_02 | exclude tail overlap | `SWA_WRITE_MODE=none`, `SWA_WRITE_KEEP_SCOPE=exclude_tail_overlap`, last layer | running |
| ACL2V5_SWKP_03 | exclude head overlap + SWOV gate | `SWA_WRITE_MODE=kv_centered`, `SWA_WRITE_SCOPE=tail_overlap`, score=`read`, last layer | running |
| ACL2V5_SWKP_04 | exclude head overlap + intersection gate | `SWA_WRITE_MODE=kv_centered`, `SWA_WRITE_SCOPE=tail_overlap`, score=`intersection`, last layer | running |

判定规则：如果 `exclude_head_overlap` 有正信号，说明重复写入当前 chunk head seam 会污染后续 SWA history；如果 `exclude_tail_overlap` 崩，说明 tail seam 虽然输出被裁掉，但仍是下一 chunk continuity 所必需。

SWKP 结果：

| Run | SWA 写入集合策略 | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | `kv_centered tail_overlap`, keep all history | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_SWKP_01 | exclude head overlap, no K/V gate | `36.7643` | `6.3970` | `92.4483` | `0.0078` | 明显回退 |
| ACL2V5_SWKP_02 | exclude tail overlap, no K/V gate | `36.6239` | `6.2634` | `92.4306` | `0.0077` | ATE 未过 SWOV_03，Rot 略好 |
| ACL2V5_SWKP_03 | exclude head overlap + tail `kv_centered` read gate | `36.6590` | `6.3847` | `92.4507` | `0.0078` | 比 SWKP_01 好，但仍未过 |
| ACL2V5_SWKP_04 | exclude head overlap + tail `kv_centered` intersection gate | `36.6925` | `6.4046` | `92.4524` | `0.0078` | 更差 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| SWKP_01 | `5.122` | `29.902` | `30.552` | `30.627` | `3.592` | `31.293488` |
| SWKP_02 | `3.570` | `29.913` | `30.525` | `30.606` | `3.449` | `31.107005` |
| SWKP_03 | `4.186` | `29.848` | `30.487` | `30.621` | `3.547` | `31.322036` |
| SWKP_04 | `4.940` | `29.850` | `30.499` | `30.611` | `3.592` | `31.336610` |

SWKP 结论：

1. overlap history 去重没有超过 SWOV_03；`exclude_head_overlap` 明显破坏 ATE/Rot，说明当前 chunk head seam 的 SWA history 不能直接删。
2. `exclude_tail_overlap` 是本批最好，Rot=`6.2634`、Yaw=`3.449` 略优于 SWOV_03，但 ATE=`36.6239` 仍回退 `0.0176m`，segment mean 也没有超过 SWOV_03。
3. `exclude_head + tail kv_centered` 能追回一部分 ATE，但仍不如 keep-all 的 SWOV_03；这说明 SWOV_03 的有效性主要来自温和重权重，而不是删除重复 overlap token。
4. 结论：SWA history 的 overlap source 需要 **保留 continuity**，不能做 hard de-dup；后续重点转向 overlap read 时的 previous-tail / current-head 动态对齐，而不是继续扩大 keep-scope。

### 20.12 最新机制判断：SWA overlap 对齐动态区与 TTT update continuity

用户补充两点判断：

1. SWA 可能要关注 **overlap 帧的动态区域**，而不是 whole chunk 的动态区域。
2. TTT 需要从 update 机制本身理解，不能只把它当成普通 token mask。

当前代码/实验给出的机制约束：

- TTT 的 `apply` 在当前 chunk 使用旧 fast weights；`probe_ttt_write` 的写入主要影响下一 chunk。
- `w1_grad=(hidden * lr1)^T @ v` 是最直接把 token value/output 内容写进 fast weights 的分支；`TTDV` 的 `w1 tail_overlap_drop` 已经灾难级失败，说明 overlap seam 的 value continuity 不能硬删。
- `w0/w2` 少写能改善 Rot/Yaw，但会牺牲 ATE/segment mean，说明 TTT 不是简单“少写动态就好”，而是三分支 fast-weight 更新方向被重排。
- 因此 TTT 后续不再做 hard drop；若继续探索，应做更结构化的 update mixing/curriculum：保留 `w1` continuity，同时只对 overlap 动态区域的 `w0/w2` 做温和方向调整。
- SWA 的 history 是显式 KV cache，当前 chunk tail overlap 输出会被 merge tail-trim 丢掉，但它仍会写入下一 chunk 的 SWA source；这是比 TTT 更直接的 overlap-source 污染入口。
- 已新增 `SWA_WRITE_KEEP_SCOPE=exclude_*` 测试 overlap history 去重；当前 `SWKP` 正在跑。
- 为了更贴合“overlap 帧动态区域”，新增 `swa_overlap_bias_mode=union/intersection`：
  - `union`: previous tail source 与 current head query 任一端高动态就压制；
  - `intersection`: 只压制两端都高动态的位置；
  - 二者都走 compact overlap-bias 路径，不构造 dense `[current, history+current]` mask。

下一步执行顺序：

1. 等 `SWKP_01-04` 完成并记录结果。
2. 若 SWKP 未过 `SWOV_03`，启动 `SWAQ3`：在 `SWOV_03` 写入侧基础上叠加 read-side overlap `union/intersection` bias，验证 overlap 对齐动态区域是否与 write-side KV centered 互补。
3. 若仍未接近 `KITTI01 ATE < 30m`，停止小幅调参，转向更大的结构改动：TTT update mixing / two-phase replay 或 SWA overlap-source replacement，而不是继续扩展普通 cue 名字。

### 20.13 SWAQ3 / TTVT branch02：overlap read 对齐与 w0+w2 tail veto

本批按 20.12 的判断继续只探索 TTT/SWA 写入机制，不回到 cue 大扫。固定主线：

```text
read cue = C23 past_only
read intervention = frame pair/all
beta = 4.75
write score = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
success target = KITTI01 ATE < 30m
```

资源记录：

- `SWAQ3_01`: `14:19:03 -> 14:43:50`，约 `24.8 min`
- `SWAQ3_02`: `14:20:51 -> 14:46:20`，约 `25.5 min`
- `TTVT_03`: `14:27:24 -> 14:51:59`，约 `24.6 min`
- `TTVT_04`: `14:27:36 -> 14:52:35`，约 `25.0 min`

#### SWAQ3：SWOV 写入侧 + overlap read-side union/intersection

SWAQ3 在当前 best `SWOV_03` 写入侧基础上叠加 compact overlap read bias：

- `union`: previous tail source 与 current head query 任一端高动态就压制；
- `intersection`: 只压制两端都高动态的位置。

| Run | Read overlap mode | SWA write | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | off | `kv_centered tail_overlap last` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_SWAQ3_01 | union | `kv_centered tail_overlap last` | `36.6519` | `6.3576` | `92.4416` | `0.0078` | 回退 |
| ACL2V5_SWAQ3_02 | intersection | `kv_centered tail_overlap last` | `36.6561` | `6.3626` | `92.4414` | `0.0078` | 回退 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| SWAQ3_01 union | `4.410` | `29.901` | `30.538` | `30.624` | `3.533` | `31.216903` |
| SWAQ3_02 intersection | `4.414` | `29.905` | `30.542` | `30.628` | `3.538` | `31.217322` |

SWAQ3 结论：

1. overlap read-side dynamic alignment 工程路径有效，但没有超过 SWOV_03。
2. union / intersection 都改善不了 segment mean，说明当前瓶颈不是“读下一 chunk head-overlap 时再压一下 previous tail source”。
3. 只在 overlap read path 加动态关系 bias 会牺牲 ATE/Rot；当前 SWA 主点仍是写入侧 `kv_centered tail_overlap last`。

#### TTVT branch02：只在 tail-overlap seam 对 w0+w2 做 dynamic veto

TTVT_03/04 直接测试用户强调的“动态区域少写入”，但保留 `w1` native continuity：

| Run | TTT write scope | Branch | SWA write | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | all | branch0 default | `kv_centered tail_overlap last` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_TTVT_03 | `tail_overlap_veto` | `w0+w2` | off | `36.6862` | `6.3703` | `92.4378` | `0.0078` | 回退 |
| ACL2V5_TTVT_04 | `tail_overlap_veto` | `w0+w2` | SWOV_03 | `36.7237` | `6.4110` | `92.4418` | `0.0078` | 叠加更差 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| TTVT_03 | `4.726` | `29.893` | `30.530` | `30.588` | `3.555` | `31.182129` |
| TTVT_04 | `4.664` | `29.945` | `30.582` | `30.661` | `3.574` | `31.223454` |

TTVT branch02 结论：

1. `w0+w2` tail-overlap veto 比 hard drop 安全，但仍没有过 SWOV_03。
2. 叠加 SWOV 后反而更差，说明 TTT seam-veto 与 SWA overlap-KV centered 不是简单互补。
3. 当前 evidence 支持：TTT 的 overlap seam 需要 continuity；“少写动态”如果只落到 TTT replay token prior，容易改变 fast-weight update 方向但不能修全局 ATE。

### 20.14 TTNM：native replay anchored update mixing

根据 TTT update 机制新增 `TTT_WRITE_NATIVE_MIX_SCALES`：

```text
W_commit = W_native + gamma * (W_semantic - W_native)
```

这与之前 `ttt_write_delta_scales` 不同：旧方法是相对旧权重缩放 replay delta；native-mix 以原生 replay 后的 fast weight 为 continuity anchor，只把 semantic replay 当 correction。目标是避免 TTT update 被 D_g prior 过度重排。

本批固定叠加 SWOV_03 写入侧，只测 native-mix 是否能超过当前 best：

| Run | Native mix scales | Branch prior | SWA write | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | `1,1,1` | default branch0 | `kv_centered tail_overlap last` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_TTNM_01 | `0.5,1,0.5` | `w0+w2` | SWOV_03 | `36.7278` | `6.3935` | `92.4414` | `0.0078` | 回退 |
| ACL2V5_TTNM_02 | `0.25,1,0.25` | `w0+w2` | SWOV_03 | `36.7159` | `6.3760` | `92.4416` | `0.0078` | 回退但略好于 0.5 |
| ACL2V5_TTNM_03 | `0.5,1,1` | `w0` | SWOV_03 | `36.7185` | `6.3750` | `92.4410` | `0.0078` | branch0 correction 变轻也回退 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| TTNM_01 | `4.840` | `29.942` | `30.583` | `30.647` | `3.575` | `31.214989` |
| TTNM_02 | `4.696` | `29.934` | `30.572` | `30.650` | `3.551` | `31.219682` |
| TTNM_03 | `4.642` | `29.960` | `30.599` | `30.677` | `3.546` | `31.210044` |

TTNM 结论：

1. native-anchored mixing 没有超过 SWOV_03；semantic replay 变轻后 ATE/Rot/segment mean 全部回退。
2. 这说明当前 branch0 semantic write correction 虽然小，但不是简单过强；把它拉回 native 会损失当前 read+SWA 主线收益。
3. `w0+w2` 即使保留 `w1` native continuity，也没有改善；`w2` hidden/value-gate 分支参与 semantic correction 仍不适合作主线。
4. TTT update-mixing 这条目前没有形成 `<30m` 级别突破信号；后续若继续动 TTT，应考虑更底层的 replay objective / loss，而不是只在 replay output state 上插值。

### 20.15 SWRS：SWA overlap-source residual centering

根据“overlap 帧动态区域”假设新增 SWA 写入形态：

```text
kv_resid_centered:
    对 tail_overlap source 中的 K/V，不直接乘 gate；
    而是以该 overlap source 的均值为 continuity anchor：
        KV_new = mean(KV_scope) + gate * (KV_old - mean(KV_scope))
```

意图：动态 token 不被硬删、不被乘到接近零，而是去掉局部偏差并保留 seam 均值连续性。

| Run | SWA write mode | Scope | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| SWOV_03 reference | `kv_centered` | `tail_overlap` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | 当前 overall best |
| ACL2V5_SWRS_01 | `kv_resid_centered` | `tail_overlap` | `36.7312` | `6.3891` | `92.4370` | `0.0078` | 回退 |

Trajectory diagnostics：

| Run | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| SWRS_01 | `4.582` | `29.980` | `30.616` | `30.675` | `3.574` | `31.167644` |

SWRS 结论：

1. residual centering 保留了 final error，但 ATE/Rot/segment mean 明显不如 SWOV_03。
2. 这说明 SWOV_03 的有效性不是简单“消掉 overlap 动态局部偏差”，而更像对 overlap K/V 的温和幅度重权重；直接改变 K/V 几何分布会破坏 local memory。
3. SWA 写入侧当前仍锁 `kv_centered tail_overlap score=read rho0.10 min0.85 last`。

### 20.16 当前 v5 收口判断

截至本批，KITTI01 best 仍是：

```text
ACL2V5_SWOV_03
C23 past_only + pair/all beta 4.75
write = stage_d_x_dg_inv_sqrt
SWA write = kv_centered tail_overlap last, rho=0.10, min_gate=0.85
ATE / Rot = 36.6063 / 6.2809
```

距离用户最终成功标准：

```text
KITTI01 ATE < 30m
```

仍差约：

```text
36.6063 - 30.0000 = 6.6063m
```

本批新增机制判断：

1. SWA overlap read-side union/intersection bias 没有帮助，说明 read seam 再压制不是主瓶颈。
2. TTT `w0+w2` tail-overlap dynamic veto 没有帮助，说明 TTT seam 的 fast-weight continuity 不能只靠 token prior 解决。
3. TTT native-anchored semantic correction 没有帮助，说明当前 branch0 correction 不是简单过强。
4. SWA residual centering 没有帮助，说明直接改变 overlap K/V 分布比温和 centered rescale 风险更大。
5. 当前 v5 的数量级瓶颈已经不像普通 TTT/SWA 写入策略能解决；这些写入策略最多带来 `0.02-0.1m` 级别变化，距离 `<30m` 需要 `6m+` 级跃迁。

下一步若继续追 `<30m`，不应再扩大普通写入参数矩阵。优先级应转为：

1. 回到 read/intervention 的结构性突破：例如 pair/all 已把 50f mean 拉到 `<30m`，但 100/200f 与全局 Sim3 仍卡住；
2. 做 per-head / per-layer D_g cache，寻找比 headmean 更强的 read cue，而不是继续用同一个 D_g 控写入；
3. 若坚持写入侧，必须改变目标函数或 replay objective，而不是继续改 `A_tok`、scope、rho、gate mode。

### 20.17 新机制计划：SWA overlap-source aligned gate 与 TTT update 解释

用户补充两点：

1. SWA 可能要关注 overlap 帧的动态区域，而不是 chunk 的动态区域。
2. TTT 需要深入理解更新机制，看是否能得到新的写入策略启发。

当前代码机制拆解：

- SWA 的 cache 是显式 `history["k"], history["v"]`。上一 chunk 的 tail overlap 会作为下一 chunk 的 previous source tail，被下一 chunk head overlap 直接读取；因此 SWA 的风险对象应优先是 **previous tail-overlap source token**，不是 whole chunk。
- 之前 `SWOV_03` 是写入侧 `kv_centered tail_overlap last`，已经说明温和改写 tail-overlap source K/V 有正信号。
- 之前 `SWAQ3 union/intersection` 改的是 overlap query-source pair 的 attention logits；它没有改 source K/V 本体，因此仍可能漏掉“source token 自身污染”。
- 新增 `SWA overlap-source aligned gate`：在 SWA KV-cache read path 中，只对 previous tail-overlap source token 按一对一对齐的动态分数 gate K/V：

```text
source token = previous chunk tail overlap
query token  = current chunk head overlap
score modes:
    source       = D_prev_tail
    current      = D_cur_head
    union        = max(D_prev_tail, D_cur_head)
    intersection = min(D_prev_tail, D_cur_head)
    disagreement = abs(D_prev_tail - D_cur_head)
gate = clamp(1 - rho * score, min_gate, 1)
target = v / kv
```

这条与 `SWAQ3` 的区别是：它不是 pairwise logit bias，而是直接改变 overlap source K/V 的贡献，且只作用在物理 overlap 对齐区域。

TTT 机制判断：

- TTT replay 不是普通 SGD 或 focal loss。`token_prior` 乘到 `lr0/lr1/lr2` 后，分支梯度会进入 `zeropower_via_newtonschulz5`，并且每次 update 后恢复旧 fast-weight norm。
- 因此 “少写动态区域” 不等价于线性缩小写入幅度；它主要是在改变 fast-weight update direction。
- `w1` 负责 output/value 内容写入，hard drop 已经崩坏；`w0/w2` 可以改善 Rot/Yaw 但常损伤 ATE，说明 TTT 需要保留全 chunk 低频几何背景和 overlap continuity。
- 本批不再做 TTT hard/drop/veto 参数小扫；先测试更贴合 SWA 机制的新 overlap-source gate。如果仍远离 `<30m`，下一步应转向更大的 TTT replay objective 设计，而不是继续改普通 token prior。

#### 20.17 计划矩阵

固定主线：

```text
seq = KITTI01 full
cue = C23 past_only
read intervention = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
SWA write baseline = SWOV_03: kv_centered tail_overlap last, rho=0.10, min=0.85
success target = KITTI01 ATE < 30m
```

| Run | 新增机制 | Gate mode | Target | rho/min | 目的 |
|---|---|---|---|---|---|
| ACL2V5_SWAS_01 | overlap-source aligned gate + SWOV | source | `v` | `0.10 / 0.85` | 只按上一 tail overlap D 抑制 V source |
| ACL2V5_SWAS_02 | overlap-source aligned gate + SWOV | current | `v` | `0.10 / 0.85` | 按当前 head overlap D 修正 source |
| ACL2V5_SWAS_03 | overlap-source aligned gate + SWOV | union | `v` | `0.10 / 0.85` | 任一端动态就压 V source |
| ACL2V5_SWAS_04 | overlap-source aligned gate + SWOV | source | `kv` | `0.05 / 0.85` | 更温和地同时压 K/V source |

Promotion gate：

- 第一目标仍是 `KITTI01 ATE < 30m`。
- 若未达标但超过 `SWOV_03 = 36.6063 / 6.2809`，保留为新 SWA 主线。
- 若全部回退，则说明 overlap source 的“读时 K/V gate”也不能解决 6m 级差距，应停止普通 SWA gate 小扫。

#### 20.17 实验结果：SWA overlap-source aligned gate

运行记录：

- 4 并发，GPU 0-3；
- start `15:32:05`；
- finish：`SWAS_03 15:55:22`，`SWAS_01 15:56:36`，`SWAS_02 15:56:27`，`SWAS_04 15:56:47`；
- 约 `23.3-24.7 min` 完成 4 个 KITTI01 full run。

说明：中途补了脚本参数，已启动的 bash 在 benchmark 后读到新脚本尾部导致 `line 215: 2: command not found`，但该错误发生在 `kitti_benchmark.log` 生成和 `DONE` 记录之后，不影响结果文件。

| Run | Mode | Target | rho/min | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOV_03 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| SWOV_03 reference | tail-overlap write `kv_centered` | K/V write | `0.10 / 0.85` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | reference | 当前 v5 best |
| ACL2V5_SWAS_01 | overlap-source read gate `source` | `v` | `0.10 / 0.85` | `36.6552` | `6.3522` | `92.4405` | `0.0078` | `+0.0489` | 回退 |
| ACL2V5_SWAS_02 | overlap-source read gate `current` | `v` | `0.10 / 0.85` | `36.6576` | `6.3613` | `92.4405` | `0.0078` | `+0.0513` | 回退 |
| ACL2V5_SWAS_03 | overlap-source read gate `union` | `v` | `0.10 / 0.85` | `36.6565` | `6.3561` | `92.4404` | `0.0078` | `+0.0502` | 回退 |
| ACL2V5_SWAS_04 | overlap-source read gate `source` | `kv` | `0.05 / 0.85` | `36.6515` | `6.3568` | `92.4410` | `0.0078` | `+0.0452` | 本批最好但仍回退 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| SWAS_01 sourceV | `36.6552` | `4.396` | `29.898` | `30.535` | `30.619` | `3.530` | `31.209946` |
| SWAS_02 currentV | `36.6576` | `4.387` | `29.905` | `30.542` | `30.626` | `3.539` | `31.210670` |
| SWAS_03 unionV | `36.6565` | `4.369` | `29.902` | `30.539` | `30.622` | `3.533` | `31.206912` |
| SWAS_04 sourceKV | `36.6515` | `4.377` | `29.900` | `30.537` | `30.623` | `3.531` | `31.215772` |

SWAS 结论：

1. 用户关于 SWA 关注 overlap source 的判断是合理机制假设，但直接在 read path gate previous tail-overlap source K/V 没有超过 SWOV_03。
2. 这批确实改善 FinalErr `0.19-0.22m`，但 ATE、Rot、50/100/200 segment mean 和 Yaw 全部回退；这不是可晋级主线。
3. `source/current/union` 三种 overlap 对齐分数差异很小，说明当前瓶颈不是“读下一 chunk head-overlap 时再压 previous tail source”。
4. 停止继续 SWA overlap-source 小 gate；下一步转入 TTT replay feature gate，直接改变 TTT replay 的 K/V 残差方向，而不是继续改普通 lr prior。

#### 20.17 预备实现：TTT replay feature centered gate

为了避免继续浅扫普通 `token_prior / rho / scope`，已补一个更贴近 TTT 更新机制的可选写入策略，等待 SWAS 结果后决定是否启动：

```text
--ttt_write_replay_feature_gate_mode = v_centered / k_centered / kv_centered
--ttt_write_replay_feature_gate_rho
--ttt_write_replay_feature_gate_min
```

机制：

- 当前 `token_prior` 只重加权 lr，之后会被 `zeropower_via_newtonschulz5` 近似归一化，因此它主要改变 update direction，不能可靠缩小 update magnitude。
- 新策略不再只改 lr，而是在 TTT replay 前对低 prior token 的 cached `k/v` residual 做 centered gate：

```text
risk = normalize(max(prior) - prior)
gate = clamp(1 - rho * risk, min_gate, 1)
x' = center_static(x) + gate * (x - center_static(x))
x = k / v / kv
```

这相当于把动态/低 prior token 对 fast-weight 梯度方向的影响拉回静态中心，而不是简单少写。它更接近“focal loss / robust target”的思路：保留全 chunk 的静态几何背景，同时降低低 prior token 的特征残差方向影响。

若 SWAS 全部未过 `SWOV_03`，下一批优先只跑 2-4 个 TTT replay feature gate，而不是继续 SWA overlap-source 参数小扫。

#### 20.18 启动：TTT replay feature centered gate

由于 SWAS 全部未过 `SWOV_03`，启动 TTT replay feature gate 小批。固定主线：

```text
seq = KITTI01 full
cue = C23 past_only
read intervention = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
SWA write = SWOV_03: kv_centered tail_overlap last, rho=0.10, min=0.85
success target = KITTI01 ATE < 30m
```

| Run | TTT replay feature mode | rho/min | 目的 |
|---|---|---:|---|
| ACL2V5_TTRF_01 | `v_centered` | `0.25 / 0.75` | 主要改 `w1` value/content 方向 |
| ACL2V5_TTRF_02 | `k_centered` | `0.25 / 0.75` | 主要改 `w0/w2` key/gate 方向 |
| ACL2V5_TTRF_03 | `kv_centered` | `0.25 / 0.75` | 同时改 K/V，测试动态 residual 方向是否为主污染 |
| ACL2V5_TTRF_04 | `kv_centered` | `0.50 / 0.50` | 强版本，判断这个机制有没有大信号 |

Promotion gate：

- 第一目标：`KITTI01 ATE < 30m`；
- 若未达标但超过 `SWOV_03 = 36.6063 / 6.2809`，保留为新 TTT 写入主线；
- 若全部只出现 `0.05-0.1m` 级波动或回退，应判断普通 TTT/SWA 写入机制已到平台，下一步必须转向 per-head / per-layer read cue 或真正 replay objective 重构。

#### 20.18 实验结果：TTT replay feature centered gate

运行记录：

- 4 并发，GPU 0-3；
- start `15:59:20`；
- finish：`TTRF_02 16:24:29`，`TTRF_03 16:25:04`，`TTRF_01 16:25:07`，`TTRF_04 16:25:35`；
- 约 `25.2-26.3 min` 完成 4 个 KITTI01 full run。

| Run | Feature gate | rho/min | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOV_03 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SWOV_03 reference | none | - | `36.6063` | `6.2809` | `92.4378` | `0.0077` | reference | 当前 overall best |
| ACL2V5_TTRF_01 | `v_centered` | `0.25 / 0.75` | `36.9507` | `5.8465` | `92.4519` | `0.0076` | `+0.3444` | yaw/rot 改善但 ATE 回退 |
| ACL2V5_TTRF_02 | `k_centered` | `0.25 / 0.75` | `37.7875` | `6.1522` | `92.5513` | `0.0094` | `+1.1812` | scale/final 开始坏 |
| ACL2V5_TTRF_03 | `kv_centered` | `0.25 / 0.75` | `38.4357` | `6.7634` | `92.5735` | `0.0112` | `+1.8294` | 明显回退 |
| ACL2V5_TTRF_04 | `kv_centered` | `0.50 / 0.50` | `45.1307` | `12.3507` | `92.7291` | `0.0194` | `+8.5244` | 强 gate 崩坏 |

TTT feature gate 实际强度：

| Run | Gate target | Applied layer records | Mean gate | Max delta | Risk mean |
|---|---|---:|---:|---:|---:|
| TTRF_01 | `v` | 684 | `0.875` | `0.25` | `0.50` |
| TTRF_02 | `k` | 684 | `0.875` | `0.25` | `0.50` |
| TTRF_03 | `k,v` | 684 | `0.875` | `0.25` | `0.50` |
| TTRF_04 | `k,v` | 684 | `0.750` | `0.50` | `0.50` |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| TTRF_01 v_centered | `36.9507` | `4.496` | `30.687` | `31.275` | `31.694` | `2.952` | `31.322531` |
| TTRF_02 k_centered | `37.7875` | `18.805` | `32.474` | `33.069` | `34.269` | `2.885` | `32.394961` |
| TTRF_03 kv_centered | `38.4357` | `29.573` | `33.320` | `33.917` | `35.671` | `3.560` | `32.641489` |
| TTRF_04 kv_centered strong | `45.1307` | `75.082` | `39.880` | `40.561` | `44.474` | `8.573` | `34.390775` |

TTRF 结论：

1. TTT replay feature gate 没有达到 `KITTI01 ATE < 30m`，也没有超过 SWOV_03。
2. `v_centered` 的 yaw / rotation 变干净，但 ATE、local segment mean 和 scale 都回退，说明 value/content residual gate 不是主线。
3. `k_centered` 与 `kv_centered` 对 final error / scale 非常敏感；强 KV gate 直接把 trajectory 推坏。
4. 这进一步确认 TTT replay 不能粗暴改全分支 K/V。当前 feature gate 在 replay 前改的是共享 `k/v`，会同时影响 branch0/1/2 的 native replay 连续性；这和 safe write 的机制冲突。
5. 下一步如果继续 TTT 写入侧，必须做 **branch-isolated replay feature gate**：只让 gated K/V 影响某些最终分支，例如 `w0` 或 `w0+w2`，同时保留其他分支的 native replay 输出，避免破坏 `w1` value/scale 通道。

### 20.19 下一批计划：overlap 动态 source 与 branch-isolated TTT replay

用户补充判断：

```text
1. SWA 可能要关注 overlap 帧的动态区域，而不是 chunk 的动态区域；
2. TTT 需要深入理解更新机制，看是否能得到新的写入策略启发。
```

当前判断：

- SWA line：`SWOV_03` 的正信号来自 tail-overlap source 写入侧，而 `SWAS` 的 read-time aligned source gate 没有晋级。下一步不再做 source/current/union 小扫，而只测试是否应把 **head+tail overlap** 都纳入 SWA write，因为当前 SWA history 保留多 chunk source，head overlap 虽然不是下一 chunk 的最近 tail source，但会作为 older source 留在 history。
- TTT line：`token_prior` 乘 lr 后被 zeropower 与 weight-norm restoration 吃掉幅度信息，主要改变 update direction；TTRF 证明共享 K/V feature gate 会污染 native branch。因此下一步要把 feature gate 与最终 branch 输出隔离。

计划矩阵：

| Run | 机制 | 配置 | 目的 |
|---|---|---|---|
| ACL2V5_SWBO_01 | SWA write both overlap | `kv_centered both_overlap last rho0.10 min0.85` | 检查 SWA history 中 head+tail overlap 动态 source 是否都应温和重权重 |
| ACL2V5_SWBO_02 | SWA write head overlap | `kv_centered head_overlap last rho0.10 min0.85` | 判断 head overlap older-source 污染是否存在 |
| ACL2V5_TTBI_01 | branch-isolated feature gate | `k_centered rho0.25 min0.75 -> gated branch w0 only` | 保留 w1/w2 native，测试只改 branch0 gate direction |
| ACL2V5_TTBI_02 | branch-isolated feature gate | `k_centered rho0.25 min0.75 -> gated branch w0+w2` | 保留 w1 value/scale，测试 gate+hidden branch 是否可清 Rot/Yaw 而不坏 ATE |

Promotion gate：

- 第一目标仍是 `KITTI01 ATE < 30m`；
- 若未达标但超过 SWOV_03，保留为新写入主线；
- 若全部回退，停止普通 TTT/SWA 写入策略小扫，转向真正 replay objective 重构或 per-head read cue。

#### 20.19 实验结果：SWA both/head overlap 与 branch-isolated TTT replay

运行记录：

- 4 并发，GPU 0-3；
- start `16:30:51`；
- finish：`SWBO_01 16:54:29`，`SWBO_02 16:54:36`，`TTBI_02 16:55:29`，`TTBI_01 16:56:58`；
- SWBO 约 `23.6 min`，TTBI 约 `24.6-26.1 min`。

| Run | 机制 | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOV_03 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| SWOV_03 reference | SWA `kv_centered tail_overlap` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | reference | 当前 overall best |
| ACL2V5_SWBO_01 | SWA `kv_centered both_overlap` | `36.6179` | `6.3787` | `92.4428` | `0.0078` | `+0.0116` | 很接近但未超过 |
| ACL2V5_SWBO_02 | SWA `kv_centered head_overlap` | `36.6768` | `6.3580` | `92.4357` | `0.0078` | `+0.0705` | 回退 |
| ACL2V5_TTBI_01 | `k_centered -> gated w0 only` | `36.7974` | `6.3295` | `92.4547` | `0.0078` | `+0.1911` | endpoint/yaw 好，但 ATE 回退 |
| ACL2V5_TTBI_02 | `k_centered -> gated w0+w2` | `37.0805` | `6.2281` | `92.4657` | `0.0078` | `+0.4742` | rotation/yaw 更好，ATE 回退更大 |

TTBI hook 验证：

| Run | Feature gate | Branch mask | Isolated | Mean gate |
|---|---|---|---|---:|
| TTBI_01 | `k_centered` | `[0]` | true | `0.875` |
| TTBI_02 | `k_centered` | `[0,2]` | true | `0.875` |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| SWBO_01 both overlap | `36.6179` | `4.870` | `29.871` | `30.506` | `30.587` | `3.545` | `31.231604` |
| SWBO_02 head overlap | `36.6768` | `4.839` | `29.906` | `30.544` | `30.587` | `3.544` | `31.146322` |
| TTBI_01 branch0 | `36.7974` | `2.449` | `30.056` | `30.686` | `30.911` | `3.324` | `31.368753` |
| TTBI_02 branch0+2 | `37.0805` | `2.710` | `30.741` | `31.333` | `31.798` | `3.130` | `31.500577` |

20.19 结论：

1. SWA both-overlap 与 head-overlap 没有超过 tail-overlap；这把用户的 overlap 假设进一步收窄：当前有效 SWA 写入区域主要是 **tail overlap source**，也就是下一 chunk 会直接复用的最近 previous-source overlap。
2. head overlap 作为 older source 留在 history 不是主要污染源；把 head overlap 一起重权重会损伤 ATE/Rot。
3. branch-isolated TTT feature gate 证明了机制方向：保护 `w1` 后，`k_centered` 不再像 TTRF_02 那样把 FinalErr 推到 `18.805m`，而是把 FinalErr 降到 `2.449m`。
4. 但 TTBI 仍损伤 ATE / segment mean / scale，说明它修的是 endpoint/yaw，不是全局 trajectory drift。
5. 当前仍未接近 `KITTI01 ATE < 30m`。下一步如果继续 TTT/SWA，不能再只做 gate 范围小扫；要测试 **overlap source replacement / consistency** 或 **two-replay objective** 这种结构改动。

### 20.20 启动：SWA overlap-source replacement 与 TTT value-branch isolated replay

用户新判断：

```text
1. SWA 应优先关注 overlap 帧的动态区域，而不是 whole-chunk 动态区域；
2. TTT 需要从更新机制理解，而不是继续只调 token prior / sparse ratio。
```

机制分析：

- SWA 的真正 seam source 是上一 chunk 的 tail-overlap KV cache。`SWOV_03` 已证明在写入侧对 tail-overlap K/V 做温和 centered rescale 有正信号，但 `SWAS` 的读时 gate 只是削弱 source，没有修正上一 tail source 与当前 head-overlap 的 K/V 对齐关系。
- 新增 `SWA overlap-source replacement`：在 SWA read path 中，对上一 chunk tail-overlap source token，按 `D_prev_tail` 与 `D_current_head` 的 aligned dynamic score，把 source K/V 轻微 blend 到当前 head-overlap K/V：

```text
K_src' = (1-alpha) * K_src + alpha * K_cur_head
V_src' = (1-alpha) * V_src + alpha * V_cur_head
alpha = alpha_max * score(D_prev_tail, D_current_head)
```

- TTT 方面，`token_prior` 乘进 `lr0/lr1/lr2` 后会经过 `zeropower` 和 fast-weight norm restoration，因此它更像改变 update direction，而不是可靠减少写入幅度。TTRF 显示 `v_centered` 会改善 rotation/yaw 但伤 ATE；因此补测 **只让 v-centered replay 影响最终 `w1`**，保留 `w0/w2` native replay，验证 value/content 分支是否能单独清理 orientation 而不破坏 key/gate trajectory continuity。

固定主线：

```text
seq = KITTI01 full
success target = KITTI01 ATE < 30m
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
SWA baseline = SWOV_03: kv_centered tail_overlap last rho0.10 min0.85
```

计划矩阵：

| Run | 机制 | 参数 | 目的 |
|---|---|---|---|
| ACL2V5_SWRP_01 | SWA overlap-source replacement | `union, target=kv, alpha=0.25` | 任一端动态时，把上一 tail source K/V 温和对齐到当前 head-overlap K/V |
| ACL2V5_SWRP_02 | SWA overlap-source replacement | `mismatch, target=kv, alpha=0.50` | 只修 previous/current dynamic disagreement 区域，测试动态定义是否要看 overlap 两端差异 |
| ACL2V5_TTVB_01 | TTT value-branch isolated replay | `v_centered rho0.25 min0.75 -> gated w1 only` | 只改 value/content branch，测试能否保留 ATE 同时改善 Rot/Yaw |
| ACL2V5_TTVB_02 | TTT value-branch isolated replay | `v_centered rho0.10 min0.90 -> gated w1 only` | 更温和版本，避免 TTRF_01 的 scale/segment 回退 |

Promotion gate：

- 第一目标仍为 `KITTI01 ATE < 30m`；
- 若未达标但超过 `SWOV_03 = 36.6063 / 6.2809`，保留为新 TTT/SWA 写入主线；
- 若仍只在 `0.05-0.3m` 内波动或回退，则说明当前 TTT/SWA 写入机制的可用空间仍是小修小补，`<30m` 需要转向 read/per-head/intervention 或更底层 TTT objective 重构。

#### 20.20 实验结果：SWA overlap-source replacement 与 TTT value-branch isolated replay

运行记录：

- 4 并发，GPU 0-3；
- SWRP start `17:03:52`，finish：`SWRP_01 17:27:59`，`SWRP_02 17:29:53`，约 `24.1-26.0 min`；
- TTVB start `17:06:03`，finish：`TTVB_01 17:31:10`，`TTVB_02 17:31:50`，约 `25.1-25.8 min`；
- host RAM 余量充足；继续不动 GPU 4-7，避免影响已有任务。

| Run | 机制 | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOV_03 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| SWOV_03 reference | SWA `kv_centered tail_overlap` + TTT `stage_d_x_dg_inv_sqrt` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | reference | 当前 overall best |
| ACL2V5_SWRP_01 | SWA source replacement `union, kv, alpha=0.25` | `36.6510` | `6.3633` | `92.4405` | `0.0078` | `+0.0447` | FinalErr 略好但 ATE/Rot 回退 |
| ACL2V5_SWRP_02 | SWA source replacement `mismatch, kv, alpha=0.50` | `36.6530` | `6.3669` | `92.4406` | `0.0078` | `+0.0467` | 与 union 类似，未晋级 |
| ACL2V5_TTVB_01 | TTT `v_centered -> gated w1 only`, rho/min `0.25/0.75` | `36.8969` | `6.0297` | `92.4534` | `0.0075` | `+0.2906` | Rot/endpoint 改善但 ATE 回退 |
| ACL2V5_TTVB_02 | TTT `v_centered -> gated w1 only`, rho/min `0.10/0.90` | `36.6248` | `6.2969` | `92.4447` | `0.0077` | `+0.0185` | 几乎 neutral，但未超过 SWOV_03 |

TTVB hook 验证：

| Run | Feature gate | Branch mask | Isolated | Applied layer records | Mean gate | Max delta |
|---|---|---|---|---:|---:|---:|
| TTVB_01 | `v_centered` | `[1]` | true | 684 | `0.875` | `0.25` |
| TTVB_02 | `v_centered` | `[1]` | true | 684 | `0.950` | `0.10` |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| SWRP_01 source replace union | `36.6510` | `4.354` | `29.904` | `30.540` | `30.629` | `3.534` | `31.210740` |
| SWRP_02 source replace mismatch | `36.6530` | `4.347` | `29.910` | `30.545` | `30.635` | `3.538` | `31.210300` |
| TTVB_01 value branch strong | `36.8969` | `2.303` | `30.575` | `31.179` | `31.509` | `3.148` | `31.337643` |
| TTVB_02 value branch mild | `36.6248` | `2.991` | `29.984` | `30.609` | `30.755` | `3.452` | `31.244804` |

20.20 结论：

1. 本批仍没有达到成功标准 `KITTI01 ATE < 30m`，也没有超过 `SWOV_03`。
2. SWA overlap-source replacement 在 `union/mismatch` 两种动态定义下都只改善 FinalErr，ATE/Rot/scale 均回退。直接把 previous tail source K/V blend 到 current head K/V 可能太几何化，破坏了 SWA history 的连续性。
3. TTT value-branch isolated replay 提供了有价值信号：强 `w1` value gate 把 FinalErr 从 `4.587m` 降到 `2.303m`，Rot 从 `6.2809` 降到 `6.0297`，但 ATE 和 segment mean 回退，说明它主要修 endpoint/yaw，不修全局长弧 ATE。
4. 结合 TTT 更新机制：`lr` prior 会进入 `zeropower` 和 fast-weight norm restoration，普通“少写动态区域”不是可靠的幅度控制；它更像改变梯度方向。真正要做的是限制 **哪里写**、**哪条 branch 写**、以及 **写入目标是否只作用在 seam/overlap**。
5. 用户指出“动态区域是否准确、SWA 应关注 overlap 帧动态区域”是关键。当前 SWA 的有效区域已收窄为 tail-overlap source，但 source replacement 没通过；下一步重点转向 TTT overlap-scoped feature gate，即只在会影响下一 chunk 的 overlap seam 上少写动态区域。

### 20.21 追加计划：overlap-scoped TTT 写入与 seam 动态验证

新的工作假设：

- TTT replay 的全 chunk feature gate 会把非 seam 的正常几何也一起改掉，导致 scale/segment 回退；
- 真正跨 chunk 污染主要来自 overlap seam，因为 tail-overlap 的 TTT fast-weight / SWA source 会直接影响下一 chunk；
- 因此“少写动态区域”应先限制在 `tail_overlap_veto` token scope，再看 branch/feature gate 是否保留 endpoint/yaw 收益且不伤 ATE。

固定主线：

```text
seq = KITTI01 full
success target = KITTI01 ATE < 30m
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
SWA baseline = SWOV_03: kv_centered tail_overlap last rho0.10 min0.85
TTT_WRITE_TOKEN_SCOPE = tail_overlap_veto
```

计划矩阵：

| Run | 机制 | 参数 | 目的 |
|---|---|---|---|
| ACL2V5_TTOVG_01 | TTT overlap-scoped value branch gate | `v_centered branch1 rho0.10 min0.90` | 温和版，保留 TTVB_02 的 endpoint 收益但只作用 seam |
| ACL2V5_TTOVG_02 | TTT overlap-scoped value branch gate | `v_centered branch1 rho0.25 min0.75` | 强版，验证 TTVB_01 的 Rot/FinalErr 收益是否可不伤 ATE |
| ACL2V5_TTOVG_03 | TTT overlap-scoped key branch gate | `k_centered branch0 rho0.25 min0.75` | 只改 branch0 gate direction，测试动态区域写入对 TTT gate 分支的影响 |
| ACL2V5_TTOVG_04 | TTT overlap-scoped key branch gate | `k_centered branch0+2 rho0.25 min0.75` | 保留 w1 value/scale，测试 gate+hidden 分支是否能修 yaw |

Promotion gate：

- 第一目标仍为 `KITTI01 ATE < 30m`；
- 若未达标但超过 `SWOV_03 = 36.6063 / 6.2809`，保留为新 TTT 写入主线；
- 若全部回退，则停止普通 gate 小扫，转向 two-replay objective / per-head read cue / TTT replay loss 重构。

#### 20.21 实验结果：overlap-scoped TTT feature gate

运行记录：

- 4 并发，GPU 0-3；
- start `17:36:50`，finish `18:02:55-18:03:23`，约 `26.1-26.6 min`；
- 成功标准仍为 `KITTI01 ATE < 30m`。

| Run | TTT feature gate | Branch | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOV_03 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| SWOV_03 reference | none | branch0 semantic full | `36.6063` | `6.2809` | `92.4378` | `0.0077` | reference | 当前 v5 best |
| ACL2V5_TTOVG_01 | `v_centered rho0.10 min0.90`, `tail_overlap_veto` | `[1]` | `36.9000` | `6.2539` | `92.4476` | `0.0076` | `+0.2937` | Rot 略好但 ATE 回退 |
| ACL2V5_TTOVG_02 | `v_centered rho0.25 min0.75`, `tail_overlap_veto` | `[1]` | `37.3067` | `6.1302` | `92.4589` | `0.0075` | `+0.7004` | FinalErr/Rot 好，ATE 明显回退 |
| ACL2V5_TTOVG_03 | `k_centered rho0.25 min0.75`, `tail_overlap_veto` | `[0]` | `36.8146` | `6.3019` | `92.4559` | `0.0077` | `+0.2083` | 本批 ATE 最接近，但未过 |
| ACL2V5_TTOVG_04 | `k_centered rho0.25 min0.75`, `tail_overlap_veto` | `[0,2]` | `37.1832` | `6.2215` | `92.4664` | `0.0078` | `+0.5769` | Yaw/FinalErr 改善，ATE 回退 |

Hook 验证：

| Run | Applied layer records | Gate mean | Gate min | Max delta | Scope mass | Branch isolated |
|---|---:|---:|---:|---:|---:|---|
| TTOVG_01 | 684 | `0.951078` | `0.900000` | `0.100000` | `0.094102` | true |
| TTOVG_02 | 684 | `0.877680` | `0.750000` | `0.250000` | `0.094102` | true |
| TTOVG_03 | 684 | `0.877605` | `0.750000` | `0.250000` | `0.094102` | true |
| TTOVG_04 | 684 | `0.877577` | `0.750000` | `0.250000` | `0.094102` | true |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| TTOVG_01 | `36.9000` | `2.658` | `30.221` | `30.848` | `31.003` | `3.372` | `31.277953` |
| TTOVG_02 | `37.3067` | `1.626` | `30.835` | `31.448` | `31.723` | `3.211` | `31.393947` |
| TTOVG_03 | `36.8146` | `1.822` | `30.096` | `30.720` | `30.964` | `3.292` | `31.380728` |
| TTOVG_04 | `37.1832` | `2.899` | `30.842` | `31.437` | `31.896` | `3.146` | `31.501086` |

20.21 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOV_03`。
2. `tail_overlap_veto` 的机制不是“full prior 里少写 overlap”，而是把非 overlap 全部退回 native、只在 tail seam 应用 semantic prior。因此它删除了 full-chunk TTT semantic continuity，ATE/scale 回退是合理的。
3. TTT feature gate 再次确认：endpoint / yaw / Rot 可以通过 overlap-seam 分支调节改善，但全局 ATE 依赖 full-chunk fast-weight continuity；不能只盯 seam。
4. 下一步修正实验机制：保留 full-chunk semantic TTT 写入，只在 tail-overlap seam 局部保护或去掉 static boost，验证“动态区域少写”是否应该是 **full prior + seam protection**，而不是 **tail-only replay**。

### 20.22 机制修正计划：full-prior TTT seam protection

新代码：

- `TTT_WRITE_TOKEN_SCOPE=tail_overlap_native/head_overlap_native/both_overlap_native`：保留全 chunk semantic prior，只把指定 overlap seam 的 prior 拉回 native；`TTT_WRITE_TOKEN_SCOPE_FLOOR` 控制残留 semantic residual，`0.0` 表示完全 native，`0.50` 表示半保留。
- `TTT_WRITE_TOKEN_SCOPE=tail_overlap_no_boost/head_overlap_no_boost/both_overlap_no_boost`：保留 overlap seam 内低于 1 的 dynamic suppression，但去掉高于 1 的 static boost；用于测试“少写动态，但不要过度写静态 seam”。

固定主线：

```text
seq = KITTI01 full
success target = KITTI01 ATE < 30m
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
SWA baseline = SWOV_03: kv_centered tail_overlap last rho0.10 min0.85
```

计划矩阵：

| Run | TTT scope | Floor | 目的 |
|---|---|---:|---|
| ACL2V5_TTSP_01 | `tail_overlap_native` | 0.00 | 保留 full prior，tail seam 完全 native，保护下一 chunk overlap continuity |
| ACL2V5_TTSP_02 | `tail_overlap_native` | 0.50 | tail seam 半保留 semantic prior，避免完全取消局部 static/dynamic 写入 |
| ACL2V5_TTSP_03 | `tail_overlap_no_boost` | 0.00 | overlap 只允许 dynamic suppression，不允许 static boost |
| ACL2V5_TTSP_04 | `both_overlap_no_boost` | 0.00 | 同时保护 head/tail overlap 的 boost，检查 head older-source 污染是否存在 |

Promotion gate：

- 第一目标仍为 `KITTI01 ATE < 30m`；
- 若未达标但超过 `SWOV_03 = 36.6063 / 6.2809`，保留为新 TTT 写入主线；
- 若仍全部回退，则说明 TTT 的普通 prior/scope 空间已经基本收口，需要转向更大结构：two-phase replay objective、per-head dynamic map，或把 overlap 静态/动态判断前移到模型内部 cache construction。

#### 20.22 实验结果：full-prior TTT seam protection

运行记录：

- 4 并发，GPU 0-3；
- start `18:13:04`，finish `18:38:16-18:38:46`，约 `25.2-25.7 min`；
- 成功标准仍为 `KITTI01 ATE < 30m`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_seam_protection/`。

| Run | TTT scope | Floor | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOV_03 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SWOV_03 reference | none | - | `36.6063` | `6.2809` | `92.4378` | `0.0077` | reference | 当前 v5 best |
| ACL2V5_TTSP_01 | `tail_overlap_native` | 0.00 | `36.6404` | `6.4012` | `92.4404` | `0.0078` | `+0.0341` | 最接近，但仍回退 |
| ACL2V5_TTSP_02 | `tail_overlap_native` | 0.50 | `36.7418` | `6.4031` | `92.4412` | `0.0078` | `+0.1355` | 半保留 semantic residual 更差 |
| ACL2V5_TTSP_03 | `tail_overlap_no_boost` | 0.00 | `36.6589` | `6.3940` | `92.4405` | `0.0078` | `+0.0526` | no-boost 没有过 SWOV |
| ACL2V5_TTSP_04 | `both_overlap_no_boost` | 0.00 | `36.6975` | `6.4366` | `92.4419` | `0.0079` | `+0.0912` | 同时保护 head/tail 更差 |

Hook 验证：

| Run | Applied layer records | Scope mass | Scope tokens | Prior mean before | Prior mean after |
|---|---:|---:|---:|---:|---:|
| TTSP_01 | 684 | `0.094102` | `3780.0` | `1.000000` | `1.000091` |
| TTSP_02 | 684 | `0.094102` | `3780.0` | `1.000000` | `1.000046` |
| TTSP_03 | 684 | `0.094102` | `3780.0` | `1.000000` | `0.997097` |
| TTSP_04 | 684 | `0.188205` | `7560.0` | `1.000000` | `0.994408` |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| TTSP_01 | `36.6404` | `5.152` | `29.843` | `30.478` | `30.529` | `3.580` | `31.197029` |
| TTSP_02 | `36.7418` | `5.009` | `29.951` | `30.589` | `30.645` | `3.572` | `31.208434` |
| TTSP_03 | `36.6589` | `5.020` | `29.855` | `30.492` | `30.562` | `3.572` | `31.205814` |
| TTSP_04 | `36.6975` | `5.277` | `29.925` | `30.565` | `30.613` | `3.611` | `31.212404` |

20.22 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOV_03`。
2. 保留 full-chunk semantic prior 后，tail seam 回 native 的回退幅度变小，说明 20.21 的主要问题确实是 tail-only replay 删除了 full-chunk continuity；但 seam protection 本身仍没有正收益。
3. `tail_overlap_native floor=0.0` 最接近 SWOV，仅回退 `0.0341m`，但 Rot/FinalErr/Yaw 全部变差，说明 tail seam 不是当前 TTT 写入污染的唯一入口。
4. `tail/both no_boost` 都没有晋级；仅去掉 overlap seam 的 static boost 不够，且 both overlap 更差，说明 head older-source 污染不是主要矛盾。
5. 继续做普通 token-scope 小扫收益很低。下一步按 20.23 进入 post-replay `native-delta gate`：在 zeropower / norm restoration 之后约束 semantic correction 的方向与幅度。

### 20.23 预案：TTT native-delta gate（若 20.22 未晋级）

用户补充思考：

1. SWA 更可能应该关注 overlap 帧的动态区域，而不是 whole-chunk 动态区域；
2. TTT 要从更新机制本身找启发，而不是继续只调表层 gate。

当前机制判断：

- SWA 侧已经把有效区域收窄到 `tail_overlap` source：`SWOV_03` 优于 head/both overlap，也优于读时 source gate / replacement。因此若继续做 SWA，不应再做大范围 head/current/union 小扫，而应考虑更结构性的 “当前 head overlap 反向修正 previous tail source” 或 two-pass source correction。
- TTT 侧更关键：`A_tok` 乘在 `lr0/lr1/lr2` 上，但 replay 更新随后经过 `zeropower` 与 fast-weight norm restoration。也就是说，低 prior 不等价于稳定地减少写入幅度；它更像改变 branch update direction。这解释了为什么很多“少写动态区域”实验能改善 FinalErr/Yaw/Rot，却伤 ATE/scale。

新代码 hook：

- `TTT_WRITE_NATIVE_DELTA_GATE_MODE=cosine/cosine_soft/cap/cosine_cap`
- `TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK=...`
- `TTT_WRITE_NATIVE_DELTA_GATE_MIN_COS`
- `TTT_WRITE_NATIVE_DELTA_GATE_FALLBACK`
- `TTT_WRITE_NATIVE_DELTA_GATE_CAP_RATIO`

它在 semantic replay 完成后，以 native provisional replay 为连续性锚点，对每层每 branch/head 计算：

```text
native_delta   = W_native - W_old
semantic_delta = W_semantic - W_old
correction     = W_semantic - W_native
```

然后只在 `semantic_delta` 与 `native_delta` 方向相容时保留 semantic correction，或按 native delta norm 对 correction 限幅。这个策略比 token prior 更接近真正的 TTT 写入控制，因为它发生在 zeropower / norm restoration 之后。

若 20.22 未达成 `KITTI01 ATE < 30m` 且未超过 `SWOV_03`，优先启动以下 4-run 小矩阵：

| Run | native-delta gate | Branch | 目的 |
|---|---|---|---|
| ACL2V5_TTDG_01 | `cosine min_cos=0.95 fallback=0.00` | `0` | branch0 只接受与 native 连续性高度同向的 semantic gate correction |
| ACL2V5_TTDG_02 | `cosine_soft min_cos=0.90 fallback=0.25` | `0` | 温和版，按 cosine 连续缩放，避免 hard gate 过猛 |
| ACL2V5_TTDG_03 | `cap cap_ratio=0.50` | `0` | branch0 correction norm 不超过 native update 的 50% |
| ACL2V5_TTDG_04 | `cosine_cap min_cos=0.90 fallback=0.25 cap_ratio=0.75` | `0,2` | 同时限制 gate/hidden 两条非 value branch，保留 w1 value continuity |

Smoke：

- `ACL2V5_TTDG_SMOKE_C23pairall_SWOV_cos_b0_e128` 通过；
- 发现 `min_cos=0.00` 在早期 chunk 基本不触发，`ttt_write_native_delta_gate_scale_mean=1.0`，因此 full 矩阵改为上表更高阈值。

固定主线仍沿用当前 best context：

```text
seq = KITTI01 full
success target = KITTI01 ATE < 30m
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
SWA = SWOV_03: kv_centered tail_overlap last rho0.10 min0.85
```

Promotion gate：

- 第一目标仍为 `KITTI01 ATE < 30m`；
- 若未达标但超过 `SWOV_03 = 36.6063 / 6.2809`，保留为新 TTT 写入主线；
- 若仍全部回退，则 TTT 写入策略需要进入更大结构：two-pass / two-replay objective，而不是继续做 prior/gate 小扫。

#### 20.23 实验结果：TTT native-delta gate

运行记录：

- 4 并发，GPU 0-3；
- start `18:44:55`，finish `19:08:47-19:09:26`，约 `24.0-24.5 min`；
- 成功标准仍为 `KITTI01 ATE < 30m`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/native_delta_gate/`。

| Run | Native-delta gate | Branch | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOV_03 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| SWOV_03 reference | none | default branch0 | `36.6063` | `6.2809` | `92.4378` | `0.0077` | reference | 当前 v5 best |
| ACL2V5_TTDG_01 | `cosine min_cos=0.95 fallback=0.00` | `0` | `36.6997` | `6.3744` | `92.4410` | `0.0078` | `+0.0934` | hard gate 回退 |
| ACL2V5_TTDG_02 | `cosine_soft min_cos=0.90 fallback=0.25` | `0` | `36.7340` | `6.4089` | `92.4419` | `0.0078` | `+0.1277` | soft gate 更差 |
| ACL2V5_TTDG_03 | `cap cap_ratio=0.50` | `0` | `36.7058` | `6.4280` | `92.4405` | `0.0078` | `+0.0995` | correction norm 限幅回退 |
| ACL2V5_TTDG_04 | `cosine_cap min_cos=0.90 fallback=0.25 cap=0.75` | `0,2` | `36.6423` | `6.3716` | `92.4419` | `0.0078` | `+0.0360` | 最接近，但仍未超过 |

Native-delta gate hook 统计：

| Run | Branch | Applied layer records | Scale mean | Cos mean | 解释 |
|---|---|---:|---:|---:|---|
| TTDG_01 | `w0` | 684 | `0.565` | `0.929` | high-cos hard gate 确实大量收缩 correction |
| TTDG_02 | `w0` | 684 | `0.648` | `0.929` | soft gate 更温和，但 ATE 更差 |
| TTDG_03 | `w0` | 684 | `0.964` | `0.928` | norm cap 多数时候不强，但仍破坏 ATE |
| TTDG_04 | `w0` | 684 | `0.884` | `0.929` | gate branch 有中等收缩 |
| TTDG_04 | `w2` | 684 | `0.940` | `0.971` | hidden branch 与 native 更同向，但限制后仍回退 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| TTDG_01 | `36.6997` | `4.649` | `29.912` | `30.547` | `30.628` | `3.552` | `31.214797` |
| TTDG_02 | `36.7340` | `4.687` | `29.980` | `30.615` | `30.696` | `3.580` | `31.221355` |
| TTDG_03 | `36.7058` | `4.979` | `29.928` | `30.565` | `30.627` | `3.584` | `31.202936` |
| TTDG_04 | `36.6423` | `5.014` | `29.858` | `30.498` | `30.577` | `3.534` | `31.220908` |

20.23 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOV_03`。
2. 这个实验有机制价值：post-replay 约束确实发生在 zeropower / norm restoration 之后，且 `w0` semantic/native delta cosine 均值约 `0.93`，说明 semantic replay 与 native replay 方向总体并不冲突。
3. 但只要把 semantic correction 往 native 拉回去，ATE/Rot/segment/scale 基本都回退。这和之前 `native_mix < 1` 的结果一致：当前最佳语义 TTT 写入不像是 correction 过强，更像是 correction 已经偏保守或需要更精确地放大。
4. 因此继续做“少写/限幅/拉回 native”的小扫收益很低。下一步切换到反向假设：在 post-replay 层面对 semantic correction 做小幅外推，验证是否能突破 `36.6063` 平台。
5. SWA 的下一步也不能再只用 whole-chunk D 或 read-time source gate；如果 post-replay 外推失败，应实现 two-pass overlap source correction：用当前 chunk head-overlap 的动态判断去修正 previous tail-overlap source，而不是只看上一 chunk 自己的 D。

### 20.24 计划：post-replay semantic correction extrapolation

当前 TTT 更新机制启发：

- replay 中 `A_tok` 乘在 `lr0/lr1/lr2` 上；
- 但更新会经过 `zeropower`，随后 fast weight 又被 renorm 到旧权重范数；
- 所以 token prior 不等价于线性减少写入幅度，它更像在改变更新方向；
- `native_mix < 1` 和 `native_delta_gate` 都回退，提示 post-replay semantic correction 不是简单过强。

新假设：

```text
W_semantic = W_native + semantic_correction
```

如果 `semantic_correction` 的方向整体有效但幅度偏保守，那么 `native_mix_scales > 1` 的外推可能比继续做 token-prior gate 更接近真正的 TTT 写入控制。这个实验直接作用在 replay 完成后的 fast weight correction 上，避开 `A_tok -> zeropower -> renorm` 的非线性混淆。

固定主线：

```text
seq = KITTI01 full
success target = KITTI01 ATE < 30m
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
SWA = SWOV_03: kv_centered tail_overlap last rho0.10 min0.85
```

计划矩阵：

| Run | `TTT_WRITE_NATIVE_MIX_SCALES` | 目的 |
|---|---|---|
| ACL2V5_TTEX_01 | `1.10,1.00,1.00` | branch0 semantic correction 小幅外推 |
| ACL2V5_TTEX_02 | `1.25,1.00,1.00` | branch0 中等外推 |
| ACL2V5_TTEX_03 | `1.50,1.00,1.00` | branch0 强外推，测试是否快速过冲 |
| ACL2V5_TTEX_04 | `1.25,1.00,1.25` | gate/hidden 两条非 value branch 同时外推，保留 value branch native continuity |

Promotion gate：

- 第一目标仍为 `KITTI01 ATE < 30m`；
- 若未达标但超过 `SWOV_03 = 36.6063 / 6.2809`，保留为新 TTT 写入主线；
- 若全部回退，则当前 single-pass TTT 写入空间基本收口，下一步必须转向结构实验：two-pass SWA overlap source correction 或 per-head/overlap-frame D map，而不是继续做标量小扫。

#### 20.24 实验结果：post-replay semantic correction extrapolation

运行记录：

- 4 并发，GPU 0-3；
- start `19:15:08`，finish `19:39:02-19:39:36`，约 `23.9-24.5 min`；
- 成功标准仍为 `KITTI01 ATE < 30m`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/native_mix_extrapolate/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_24_diagnostics/`。

| Run | `TTT_WRITE_NATIVE_MIX_SCALES` | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOV_03 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| SWOV_03 reference | `1.00,1.00,1.00` | `36.6063` | `6.2809` | `92.4378` | `0.0077` | reference | 旧 v5 best |
| ACL2V5_TTEX_01 | `1.10,1.00,1.00` | `36.5932` | `6.4327` | `92.4423` | `0.0078` | `-0.0131` | **ATE 微弱新 best，但 Rot/endpoint 回退** |
| ACL2V5_TTEX_02 | `1.25,1.00,1.00` | `36.7143` | `6.4098` | `92.4417` | `0.0078` | `+0.1080` | 过强回退 |
| ACL2V5_TTEX_03 | `1.50,1.00,1.00` | `36.6581` | `6.3534` | `92.4412` | `0.0078` | `+0.0518` | 仍回退，未单调 |
| ACL2V5_TTEX_04 | `1.25,1.00,1.25` | `36.6993` | `6.4044` | `92.4411` | `0.0078` | `+0.0930` | branch0+2 外推回退 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOV_03 reference | `36.6063` | `4.587` | `29.790` | `30.424` | `30.483` | `3.456` | `31.174512` |
| TTEX_01 | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` |
| TTEX_02 | `36.7143` | `5.017` | `29.937` | `30.576` | `30.641` | `3.584` | `31.213323` |
| TTEX_03 | `36.6581` | `4.847` | `29.862` | `30.495` | `30.555` | `3.540` | `31.210038` |
| TTEX_04 | `36.6993` | `4.909` | `29.934` | `30.572` | `30.629` | `3.583` | `31.208842` |

20.24 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`。
2. `TTEX_01` 以 `36.5932 / 6.4327` 成为 ATE 微弱新 best，比 `SWOV_03` 只好 `0.0131m`，但 Rot、FinalErr、Yaw 和 50/100/200 segment mean 都回退。
3. 这说明 post-replay semantic correction 外推有一点 ATE 信号，但不是强突破方向；`1.25/1.50` 都不能继续带来收益。
4. 更关键的机制判断：TTT 写入通过标量 prior / post-replay scale 已经进入平台区。`<30m` 不可能靠继续在 `36.6m` 附近做小幅 gate/scale 得到，需要改写 SWA overlap 或 TTT replay 结构。
5. 下一步按用户提示重新回到 SWA overlap 动态准确性：当前 `SWRP` 已用 `D_prev_tail` 与 `D_current_head` 做 read-time source replacement，但之前只测了 `union/mismatch` 的 K/V replacement，容易破坏 K 几何。下一批改成 **V-only overlap-source replacement**，并比较 `source/current/intersection/mismatch`，判断到底是上一 tail 的 D 准、当前 head 的 D 准，还是只应在二者一致/不一致区域修正。

### 20.25 计划：SWA overlap-frame dynamic V-only source correction

新假设：

- `SWRP_01/02` 的 K/V replacement 可能太强，破坏了 SWA source key geometry；
- 但用户指出的核心仍可能成立：SWA 应关注 overlap 帧动态，而不是 whole chunk 动态；
- 因此先只修 `V_cache`，保留 `K_cache` 的 source continuity；
- 以 `TTEX_01` 作为当前 ATE best 的 TTT 写入基底，检查 SWA overlap 动态定义是否还能给 ATE 带来增益。

固定主线：

```text
seq = KITTI01 full
success target = KITTI01 ATE < 30m
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85
SWA read-time overlap source replacement target = v
```

计划矩阵：

| Run | Replace mode | Alpha | Target | 目的 |
|---|---|---:|---|---|
| ACL2V5_SWOVR_01 | `current` | 0.25 | `v` | 只信当前 head-overlap D，测试当前帧动态是否更准 |
| ACL2V5_SWOVR_02 | `source` | 0.25 | `v` | 只信上一 tail-overlap D，测试已写入 source 的 D 是否足够 |
| ACL2V5_SWOVR_03 | `intersection` | 0.50 | `v` | 只在两端都认为动态时强修，减少误杀静态 overlap |
| ACL2V5_SWOVR_04 | `mismatch` | 0.25 | `v` | 只修两端不一致区域，测试 dynamic disagreement 是否是 seam 污染源 |

Promotion gate：

- 第一目标仍为 `KITTI01 ATE < 30m`；
- 若未达标但超过 `TTEX_01 = 36.5932 / 6.4327`，保留为新 v5 best；
- 若只改善 Rot/FinalErr 但 ATE 回退，则 SWA overlap source correction 暂停，下一步转向 per-head/overlap-frame D map 或 TTT replay loss/objective 改写。

#### 20.25 实验结果：SWA overlap-frame dynamic V-only source correction

运行记录：

- 4 并发，GPU 0-3；
- start `19:42:48`，finish `20:06:08-20:07:49`，约 `23.3-25.0 min`；
- 成功标准仍为 `KITTI01 ATE < 30m`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/swa_overlap_v_replace_ttex/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_25_diagnostics/`。

| Run | Replace mode | Alpha | Target | ATE RMSE | Rot RMSE | RPE t | RPE r | vs TTEX_01 | 结论 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| TTEX_01 reference | none | 0.00 | none | `36.5932` | `6.4327` | `92.4423` | `0.0078` | reference | 当前 ATE tiny best reference |
| ACL2V5_SWOVR_01 | `current` | 0.25 | `v` | `36.5952` | `6.4333` | `92.4413` | `0.0078` | `+0.0020` | 基本等价，略差 |
| ACL2V5_SWOVR_02 | `source` | 0.25 | `v` | `36.5915` | `6.4307` | `92.4416` | `0.0078` | `-0.0017` | ATE 微弱最好，但量级不可视为突破 |
| ACL2V5_SWOVR_03 | `intersection` | 0.50 | `v` | `36.5930` | `6.4311` | `92.4416` | `0.0078` | `-0.0002` | 完全平台内 |
| ACL2V5_SWOVR_04 | `mismatch` | 0.25 | `v` | `36.5922` | `6.4315` | `92.4418` | `0.0078` | `-0.0010` | 完全平台内 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTEX_01 reference | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` |
| SWOVR_01 current-V | `36.5952` | `5.124` | `29.812` | `30.447` | `30.522` | `3.596` | `31.215242` |
| SWOVR_02 source-V | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` |
| SWOVR_03 intersection-V | `36.5930` | `5.128` | `29.807` | `30.442` | `30.517` | `3.595` | `31.214161` |
| SWOVR_04 mismatch-V | `36.5922` | `5.120` | `29.809` | `30.444` | `30.520` | `3.595` | `31.215116` |

20.25 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`。
2. `source/current/intersection/mismatch` 四种 overlap dynamic 定义全部落在 `36.5915-36.5952m`，与 `TTEX_01=36.5932m` 几乎等价；best `SWOVR_02` 只好 `0.0017m`，不能作为真实突破。
3. 轨迹指标也几乎重合：FinalErr 仍约 `5.12m`，Yaw 仍约 `3.59deg`，50/100/200 segment mean 没有明显改善。
4. 因此当前 **read-time overlap source V replacement** 没有解决平台问题。用户提出的“关注 overlap 帧动态区域”是对的，但当前做法只是临时 blend previous-tail source V；它没有改变 SWA cache 写入本身，也没有改变 TTT 更新目标。
5. 下一步不要继续围绕 `SWA_OVERLAP_SOURCE_REPLACE_ALPHA` 小扫。更值得做的结构方向：
   - **SWA delayed write / overlap-consistency write**：当 chunk `t+1` 来时，用 `D_prev_tail` 与 `D_current_head` 的一致性去修正/重写上一 chunk tail 的 SWA source cache，或者控制当前 chunk head/tail 的历史保留，而不是只在 read 时临时 blend；
   - **TTT replay objective 改写**：`token_prior` 只乘学习率会被 zeropower 与 fast-weight norm restoration 折叠，后续应更多改变 replay feature/target direction，例如对高动态 token 的 `k/v` residual 做 stop-gradient/centering/静态原型替换，或按 branch/head 选择性更新；
   - **per-head dynamic reliability**：现在 D 是 head-mean patch map，SWA/TTT 都按同一 patch prior 操作；若要打破 `36.6m` 平台，需要知道哪些 head/layer 的动态判断真的对应 harmful write，而不是继续用全局 scalar D。

### 20.26 计划：TTT fast-weight lifetime / reset schedule

新假设：

- 目前几乎所有 TTT 写入策略都在默认 `reset_every=5` 下测试；
- TTT fast weight 的风险不只来自“写哪些 token”，也来自“写入保留多久”；
- 如果 reset 太频繁，长程几何纠正可能被切断；如果 reset 太少，动态污染可能累计；
- 这是比继续调 `0.8-1.2` prior 或 `SWA_OVERLAP_SOURCE_REPLACE_ALPHA` 更结构化的 TTT writing/lifetime 实验。

工程变更：

- `tools/run_attention_cue_experiment.sh` 新增 `RESET_EVERY` 环境变量，默认仍为 `5`；
- `run_status.txt` 会记录实际 `reset_every`。

固定主线：

```text
seq = KITTI01 full
success target = KITTI01 ATE < 30m
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85
```

计划矩阵：

| Run | RESET_EVERY | 目的 |
|---|---:|---|
| ACL2V5_TTRS_01 | 0 | 不做外部 TTT reset，测试长程 semantic write 是否积累有效几何纠正 |
| ACL2V5_TTRS_02 | 2 | 更频繁 reset，测试是否减少动态污染 |
| ACL2V5_TTRS_03 | 10 | 比默认更长保留，测试默认 5 是否过短 |
| ACL2V5_TTRS_04 | 20 | 接近整段少量 reset，测试长期写入是否崩坏或突破 |

Promotion gate：

- 第一目标仍为 `KITTI01 ATE < 30m`；
- 若未达标但超过 `TTEX_01 = 36.5932 / 6.4327`，保留为新 v5 TTT lifetime 主线；
- 若 `RESET_EVERY=0/20` 崩而 `2` 接近或更好，说明主要问题是写入污染积累；
- 若 `10/20` 好，说明默认 reset 切断了有效 TTT 纠正。

#### 20.26 实验结果：TTT fast-weight lifetime / reset schedule

运行记录：

- 4 并发，GPU 0-3；
- start `20:11:03`，finish `20:36:36`，约 `25.6 min`；
- 成功标准仍为 `KITTI01 ATE < 30m`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_reset_schedule/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_26_diagnostics/`。

| Run | RESET_EVERY | ATE RMSE | Rot RMSE | RPE t | RPE r | vs TTEX_01 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| TTEX_01 reference | 5 | `36.5932` | `6.4327` | `92.4423` | `0.0078` | reference | 默认 reset reference |
| ACL2V5_TTRS_01 | 0 | `692.7380` | `69.8326` | `95.0943` | `0.0559` | `+656.1448` | 完全崩坏 |
| ACL2V5_TTRS_02 | 2 | `44.1060` | `5.5812` | `92.5473` | `0.0084` | `+7.5128` | ATE 明显回退，Rot 好但不可用 |
| ACL2V5_TTRS_03 | 10 | `69.1357` | `7.0643` | `93.0051` | `0.0122` | `+32.5425` | 长生命周期污染明显 |
| ACL2V5_TTRS_04 | 20 | `300.1047` | `19.0350` | `94.1183` | `0.0291` | `+263.5115` | 接近长程无 reset，崩坏 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTEX_01 reference | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` |
| TTRS_01 reset0 | `692.7380` | `902.710` | `630.252` | `633.567` | `666.316` | `56.557` | `53.262121` |
| TTRS_02 reset2 | `44.1060` | `22.485` | `36.293` | `37.098` | `37.228` | `3.306` | `32.212464` |
| TTRS_03 reset10 | `69.1357` | `70.490` | `62.152` | `64.011` | `65.657` | `4.790` | `38.386380` |
| TTRS_04 reset20 | `300.1047` | `324.249` | `265.123` | `271.188` | `280.893` | `13.313` | `68.732537` |

20.26 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有任何接近 `TTEX_01` 的设置。
2. TTT fast-weight 生命周期是高风险轴：`reset_every=10/20/0` 会快速积累污染并导致尺度/轨迹崩坏，说明当前 TTT 写入不是可以长期保留的稳定几何记忆。
3. `reset_every=2` 虽然 Rot 更好，但 ATE 从 `36.5932` 退到 `44.1060`，FinalErr 从 `5.120` 退到 `22.485`，说明过频 reset 会切断当前有效的短程 correction。
4. 默认 `reset_every=5` 目前仍是唯一安全点；后续不继续细扫 reset schedule。
5. 对 TTT 更新机制的判断更明确：当前 token prior / native mix / focal prior 这类策略只能在短生命周期内轻微改变 fast-weight 更新方向，一旦写入累积，动态/错误 token 会污染 fast weights。下一步必须改 **写入目标与更新内容**，而不是继续调 fast-weight 保留时间。

### 20.27 计划：SWA overlap source cache 写入内容修正

用户补充判断：

1. SWA 可能应关注 **overlap 帧的动态区域**，而不是整段 current chunk 的动态区域；
2. TTT 的写入机制需要深入看“写入目标/更新内容”，不能只做 prior 强度小调参。

机制分析：

- SWA history 写入当前实现存的是每个 SWA layer 的 current chunk `x_in` 计算出的 `K/V` cache；下一 chunk 会把这段 history 当 previous source 读。
- 之前 H7/SWOV/SWOVR 多数实验是在 read-time 对 previous source gate/replace，这只能临时改变读，不改变当前 chunk 写给下一 chunk 的 source cache。
- 对 overlap source 来说，更直接的结构实验是：在当前 chunk 已经完成 SWA 更新后，同时计算 `x_post = x_in + gate * swa_output` 的 post-SWA `K/V`，commit 时只在 overlap tail 的动态/静态 tokens 上把写入 cache 从 pre-SWA 向 post-SWA blend。
- 这相当于测试：“下一 chunk 读 overlap source 时，是否应该读到被当前 chunk 局部修正后的 overlap 表示”，而不是读原始 pre-SWA 表示。

工程变更：

- `loger/models/pi3.py`：当 `swa_write_cache_store_post=1` 时，在 SWA history entry 中额外存 `k_post/v_post`；
- `geometry_backbone.py`：probe write-cache 保留 `k_post/v_post`；
- `hybrid_memory_controller.py`：commit SWA history 时支持 `SWA_WRITE_CACHE_BLEND_ALPHA/MODE/TARGET`：
  - `MODE=dynamic`：按 `D` 越动态越偏向 post-cache；
  - `MODE=static`：按 `1-D` 越静态越偏向 post-cache；
  - `MODE=all`：scope 内常数 blend；
  - `TARGET=v/k/kv` 控制改写 V、K 或 K/V。

固定主线：

```text
seq = KITTI01 full
success target = KITTI01 ATE < 30m
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
SWA cache blend scope = tail_overlap
```

首批矩阵：

| Run | Cache blend mode | Target | Alpha | 目的 |
|---|---|---|---:|---|
| ACL2V5_SWCB_01 | dynamic | v | 0.25 | 动态 overlap source V 改成 post-SWA 表示 |
| ACL2V5_SWCB_02 | dynamic | v | 0.50 | 加强动态 V post-cache 写入 |
| ACL2V5_SWCB_03 | dynamic | kv | 0.25 | 同时改 K/V，测试 source attention key 是否也需修正 |
| ACL2V5_SWCB_04 | static | v | 0.25 | 反向对照：只让静态 overlap 更偏向 post-cache |

Promotion gate：

- 第一目标仍是 `KITTI01 ATE < 30m`；
- 若未达标但超过当前平台 `SWOVR_02 = 36.5915 / 6.4307` 或 `TTEX_01 = 36.5932 / 6.4327`，保留为 SWA 写入内容主线；
- 若 dynamic 明显差而 static 好，说明 D 动态区域不够准，post-cache 应用于静态 anchor 更有价值；
- 若 `kv` 比 `v` 差，说明改 K 会破坏 source attention 定位，后续只保留 V-side cache write。

#### 20.27 实验结果：SWA overlap source post-cache write blend

运行记录：

- 4 并发，GPU 0-3；
- start `22:03:10`，finish `22:29:23-22:30:05`，约 `26.2-27.0 min`；
- 成功标准仍为 `KITTI01 ATE < 30m`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/swa_cache_post_blend/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_27_diagnostics/`；
- 本批由于额外存/算 post-SWA `K/V`，单批比普通 full run 慢约 `2-3 min`。

| Run | Cache blend mode | Target | Alpha | ATE RMSE | Rot RMSE | RPE t | RPE r | vs TTEX_01 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| TTEX_01 reference | none | none | 0.00 | `36.5932` | `6.4327` | `92.4423` | `0.0078` | reference | 当前 TTT extrapolate reference |
| SWOVR_02 reference | source replace | v | 0.25 | `36.5915` | `6.4307` | `92.4416` | `0.0078` | `-0.0017` | 当前 tiny best reference |
| ACL2V5_SWCB_01 | dynamic | v | 0.25 | `36.7398` | `6.3707` | `92.4422` | `0.0078` | `+0.1466` | ATE 明显回退 |
| ACL2V5_SWCB_02 | dynamic | v | 0.50 | `36.6813` | `6.4209` | `92.4422` | `0.0078` | `+0.0881` | 仍回退 |
| ACL2V5_SWCB_03 | dynamic | kv | 0.25 | `36.6764` | `6.3591` | `92.4411` | `0.0078` | `+0.0832` | 本批最好，但未过 gate |
| ACL2V5_SWCB_04 | static | v | 0.25 | `36.6936` | `6.3974` | `92.4404` | `0.0078` | `+0.1004` | static 也不行 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTEX_01 reference | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` |
| SWCB_01 dynamic-V a0.25 | `36.7398` | `4.785` | `29.947` | `30.585` | `30.654` | `3.543` | `31.226218` |
| SWCB_02 dynamic-V a0.50 | `36.6813` | `4.884` | `29.901` | `30.536` | `30.605` | `3.601` | `31.224378` |
| SWCB_03 dynamic-KV a0.25 | `36.6764` | `4.940` | `29.871` | `30.509` | `30.583` | `3.540` | `31.216977` |
| SWCB_04 static-V a0.25 | `36.6936` | `5.001` | `29.893` | `30.527` | `30.593` | `3.576` | `31.204716` |

20.27 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `TTEX_01/SWOVR_02` 平台。
2. post-cache write blend 的确改变了 SWA source history：`SWCB_01/02/03/04` 的 FinalErr 均比 TTEX_01 小，但 ATE 和 50/100/200 segment mean 全部回退，说明它修的是局部 endpoint/yaw 类现象，不是主 RMSE 漂移。
3. `dynamic` 与 `static` 都失败，不能简单归因于 “D 动态区域不准，应改写静态 anchor”；`dynamic-KV` 的 Rot/Yaw 最好但 ATE 仍差，说明改 K 没有马上崩，但也没有带来有效 source attention 定位收益。
4. 这批每 run 额外增加 post-SWA cache 计算和保存，时间成本比普通 full run 高；既然 read/hybrid 指标不过 gate，后续不继续扫 `SWA_WRITE_CACHE_BLEND_ALPHA/MODE/TARGET`。
5. 对用户关于 SWA overlap 的判断更新为：关注 overlap 帧动态区域仍是合理的，但目前 read-time replacement 与 write-time post-cache blend 都只在 `36.6m` 平台附近微扰。要继续 SWA，需要更大胆地改 **source selection / overlap frame alignment / per-head SWA cache**，而不是继续做同一 cache 的线性 blend。
6. 下一步把最高优先级切回 TTT 更新机制：测试 commit-level EMA 与 replay hard token subset。原因是 soft prior 会经过 `zeropower` 与 fast-weight norm restoration，可能无法真正减少 harmful dynamic token 的更新贡献；hard token subset 和 commit EMA 分别改变 “更新矩阵由哪些 token 生成” 与 “最终 fast-weight 走多远”。

### 20.28 计划：TTT commit EMA 与 replay hard token subset

用户新要求：

- TTT 写入策略作为第一优先级；
- 重点思考 EMA 是否可行；
- 目标仍是 `KITTI01 ATE < 30m`，如果计划内写入策略不达标，需要主动探索其他 TTT 写入方式。

TTT 更新机制判断：

1. 当前 TTT semantic write 的核心路径是：用 token prior 改 `lr0/lr1/lr2`，重放 `fast_weight_replay_update`，再经过 Muon / `zeropower_via_newtonschulz5` 和 fast-weight norm restoration。
2. 因此连续 soft prior 不是纯粹的 “少写动态区域”：它会改变梯度方向，但更新幅度会被正交化/归一化部分折叠。这解释了前面 focal / explicit dyn / sparse soft gate 大多只能带来 `0.0x-0.1m` 级微扰。
3. EMA 应该放在 replay / native-delta / native-mix 之后，对最终 `W_candidate - W_old` 做 commit-level damping，而不是只缩放 token lr。这样可以保留 replay 方向，但避免单 chunk 错误 token 让 fast weight 过冲。
4. hard token subset 比 soft prior 更直接：在 Muon/zeropower 之前删除低 prior token，改变用于构造更新矩阵的 token 集合。这才是真正意义上的 “TTT 少写入动态区域”。
5. 动态区域准确性仍是关键。首批先用当前 best `D_g` static ranking 做 hard filter，若不达标，下一批用 explicit dyn / union dyn 做 hard-filter ranking，而不是再做连续乘法 veto。

工程变更：

- `ttt_write_controller.py` 新增 `TTT_WRITE_COMMIT_EMA_ALPHA`：
  - `1.0` = 原行为；
  - `<1.0` 时，最终 commit 变成 `W_old + alpha * (W_candidate - W_old)` 并重新做 norm restoration。
- `ttt_write_controller.py` 新增 `TTT_WRITE_REPLAY_TOKEN_FILTER_MODE/RATIO/THRESHOLD`：
  - `per_frame_static_topk`：每帧保留 top-ratio 静态 token，避免某几帧 token 被全删；
  - `static_topk`：整 chunk top-ratio；
  - `dynamic_veto`：按阈值硬过滤。
- 这些 hard filter 在 `fast_weight_replay_update` 前执行，和之前的 write sparse / focal prior 不同。

固定主线：

```text
seq = KITTI01 full
success target = KITTI01 ATE < 30m
cue = C23 past_only
read = frame pair/all
beta = 4.75
write = stage_d_x_dg_inv_sqrt
WRITE_ALPHA = 0.125
PRIOR_BRANCH_MASK = 0
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
```

首批矩阵：

| Run | 机制 | EMA alpha | Hard filter | Ratio | 目的 |
|---|---|---:|---|---:|---|
| ACL2V5_TTEMA_01 | commit EMA | 0.75 | none | 1.00 | 轻度 EMA，测试是否降低 TTEX 的 over-update |
| ACL2V5_TTEMA_02 | commit EMA | 0.50 | none | 1.00 | 中等 EMA，测试是否牺牲写入换稳定 |
| ACL2V5_TTFILT_01 | hard replay subset | 1.00 | per_frame_static_topk | 0.75 | 每帧删 bottom 25% 动态/低置信 token |
| ACL2V5_TTFILT_02 | hard replay subset | 1.00 | per_frame_static_topk | 0.50 | 每帧只保留 top 50% 静态 token，测试强 hard veto |

Promotion gate：

- 第一目标仍是 `KITTI01 ATE < 30m`；
- 若未达标但超过 `SWOVR_02 = 36.5915 / 6.4307`，保留为新 TTT 主线；
- 若 EMA 改善 ATE/segment，下一批做 `alpha=0.60/0.85` 与 EMA+hard filter；
- 若 hard filter 改善 ATE，下一批比较 `D_g` vs `explicit_dyn` vs `union_dyn` 的 hard filter 排名；
- 若 hard filter 只改善 Rot/Yaw 但 ATE 回退，说明当前 dynamic/static ranking 仍不够准，需要改 hard filter 的 token source，而不是继续调 ratio。

#### 20.28 实验结果：TTT commit EMA / replay hard token subset

运行记录：

- 4 并发，GPU 0-3；
- start `21:25:31`，finish `21:50:01-21:51:16`，约 `24.5-25.8 min`；
- 成功标准：`KITTI01 ATE < 30m`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_ema_filter/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_28_diagnostics/`；
- short smoke 已确认 `ttt_write_commit_ema_applied=True`、`ttt_replay_token_filter_applied=True`，`per_frame_static_topk` ratio `0.75` 会把 `40320` tokens 压到 `30240` tokens。

| Run | 机制 | EMA alpha | Hard filter | Ratio | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| SWOVR_02 reference | SWA replace | 1.00 | none | 1.00 | `36.5915` | `6.4307` | `92.4416` | `0.0078` | 当前 tiny best reference |
| TTEX_01 reference | native mix | 1.00 | none | 1.00 | `36.5932` | `6.4327` | `92.4423` | `0.0078` | 当前 TTT extrapolate reference |
| ACL2V5_TTEMA_01 | commit EMA all branches | 0.75 | none | 1.00 | `42.6928` | `10.7492` | `92.8376` | `0.0159` | scale / endpoint 明显崩 |
| ACL2V5_TTEMA_02 | commit EMA all branches | 0.50 | none | 1.00 | `58.7134` | `26.8000` | `93.3290` | `0.0317` | 更强 EMA 更崩 |
| ACL2V5_TTFILT_01 | replay hard subset all branches | 1.00 | per-frame static topk | 0.75 | `37.1650` | `5.9450` | `92.4407` | `0.0081` | Rot/Yaw 好，但 ATE 回退 |
| ACL2V5_TTFILT_02 | replay hard subset all branches | 1.00 | per-frame static topk | 0.50 | `68.2380` | `7.7885` | `92.9489` | `0.0122` | token 删除过强，ATE 崩 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOVR_02 reference | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` |
| TTEX_01 reference | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` |
| TTEMA_01 EMA a0.75 | `42.6928` | `49.308` | `36.649` | `37.511` | `39.984` | `7.577` | `35.665564` |
| TTEMA_02 EMA a0.50 | `58.7134` | `128.219` | `50.183` | `51.945` | `58.344` | `19.753` | `43.039253` |
| TTFILT_01 top75 | `37.1650` | `11.068` | `31.433` | `31.964` | `32.691` | `3.010` | `31.231935` |
| TTFILT_02 top50 | `68.2380` | `44.517` | `59.737` | `60.958` | `63.904` | `5.388` | `37.556554` |

20.28 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02/TTEX_01` 平台。
2. commit EMA 不能以 all-branch 形式使用。当前 `PRIOR_BRANCH_MASK=0` 只让 semantic prior 作用在 branch0，但 `TTT_WRITE_COMMIT_EMA_ALPHA` 是对 `w0/w1/w2` 全部 damp，因此它实际把 native branch1/branch2 的正常更新也一起压掉；`alpha=0.50` 的 scale 从 `31.22` 飙到 `43.04`，证明这不是温和正则，而是破坏 TTT 更新平衡。
3. replay hard subset 的方向有信号但实现太粗：`top75` 把 Rot 从 `6.43` 降到 `5.95`、Yaw 从 `3.60` 降到 `3.01`，说明少写低静态/高动态 token 确实能清理 orientation；但 ATE、FinalErr、segment mean 全部回退，说明全分支共享 hard token subset 会伤到负责几何/尺度的 branch。
4. `top50` 崩到 `68.2380m`，说明 TTT replay 不能简单大比例删 token。动态区域是否准确是一方面，更关键的是 token 删除要只作用在需要抑制动态污染的 branch/layer，而不是把所有 branch 的更新样本一起砍掉。
5. 结合用户提出的 “不同层数不同 branch 采用不同策略” 判断：之前没有系统做过这一类矩阵；20.28 的失败正好说明下一步必须做 **branch-isolated / layer-aware TTT writing**。

### 20.29 计划：TTT branch / layer 分离写入优先实验

目标：

- 第一成功标准仍是 `KITTI01 ATE < 30m`；
- 若未达标，至少要搞清楚 TTT 三个 branch 与 layer 范围中，哪些能安全使用 dynamic/static cue，哪些必须保留 native；
- 不再继续 all-branch EMA 或 all-branch hard token subset。

核心假设：

1. `w0` 更适合作 semantic / dynamic suppression：此前 `PRIOR_BRANCH_MASK=0` 一直是稳定主线。
2. `w1/w2` 更可能承担尺度、几何或 residual 稳定性；对它们做全局 EMA 或 hard token 删除会造成 scale/final error 崩坏。
3. 早层/中层 TTT 写入可能更接近局部几何，深层写入更容易把 dynamic cue 放大成全局 drift；因此 branch 策略必须和 layer mode 绑定。
4. EMA 仍值得试，但应改成 `w0-only` 或指定 branch mask；hard token subset 也应改成 branch-isolated，即 filtered replay 只替换目标 branch 的 fast weight，其他 branch 使用 full-token replay。

工程补充：

- 新增 `TTT_WRITE_COMMIT_EMA_BRANCH_MASK`；
- 新增 `TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK`；
- hard filter 实现从 “全分支共用 filtered tokens” 改成 “full replay 与 filtered replay 双路径，按 branch mask 选择 `w_i`”；
- CLI / script 透传上述参数；
- smoke 必须检查：
  - `TTT_WRITE_COMMIT_EMA_BRANCH_MASK=0` 时只出现 `ttt_write_commit_ema_w0_alpha < 1`，`w1/w2` 保持 `1.0`；
  - `TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK=0` 时 debug 中标记 branch-isolated，并确认 filtered token 数只用于目标 branch。

首批 full 矩阵：

| Run | Branch 策略 | Layer mode | EMA | Hard filter | Ratio | 目的 |
|---|---|---|---:|---|---:|---|
| ACL2V5_TTBI_01 | w0-only hard subset | all | 1.00 | per-frame static topk | 0.75 | 验证 top75 的 Rot/Yaw 信号是否能保留，同时不伤 w1/w2 |
| ACL2V5_TTBI_02 | w0-only hard subset | all | 1.00 | per-frame static topk | 0.90 | 更温和 hard veto，测试是否降低 ATE 回退 |
| ACL2V5_TTBI_03 | w0-only EMA | all | 0.75 | none | 1.00 | 验证 EMA 失败是否来自误伤 w1/w2 |
| ACL2V5_TTBI_04 | w0-only hard subset + EMA | all | 0.85 | per-frame static topk | 0.90 | 组合温和 hard veto 与轻 EMA |

第二批候选（视首批结果触发）：

- 如果 `TTBI_01/02` ATE 改善但仍未达标，做 layer split：`PRIOR_LAYER_MODE=early/mid/late/last`；
- 如果 `w0-only` 明显优于 all-branch，探索 `w0+w2` 或 `w0 early + w2 last`；
- 如果 `D_g` static topk 只改善 Rot/Yaw，换 ranking：`explicit_dyn` / `old_dyn` / `D_g ∪ explicit_dyn` hard veto；
- 如果所有 TTT branch/layer hard filter 仍卡在 `36m` 平台，说明 KITTI01 `<30m` 需要更底层 memory 内容或 pose/scale correction，而不是继续微调 TTT 写入权重。

#### 20.29 实验结果：TTT branch-isolated w0-only 写入

工程验证：

- `ACL2V5_TTBI_SMOKE_C23pairall_w0filter090_ema085_e128` short smoke 通过；
- `TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK=0` 时 debug 显示：
  - `ttt_replay_token_filter_branch_isolated=true`
  - `ttt_replay_token_filter_branch_mask=[0]`
  - tokens `40320 -> 36288` for ratio `0.90`
- `TTT_WRITE_COMMIT_EMA_BRANCH_MASK=0` 时 debug 显示：
  - `ttt_write_commit_ema_w0_alpha=0.85`
  - `ttt_write_commit_ema_w1_alpha=1.0`
  - `ttt_write_commit_ema_w2_alpha=1.0`

运行记录：

- 4 并发，GPU 0-3；
- start `22:00:21`，finish `22:24:15-22:26:31`，约 `23.9-26.2 min`；
- 成功标准：`KITTI01 ATE < 30m`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_branch_isolated/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_29_diagnostics/`。

| Run | Branch 策略 | EMA | Hard filter | Ratio | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| SWOVR_02 reference | reference | 1.00 | none | 1.00 | `36.5915` | `6.4307` | `92.4416` | `0.0078` | 当前 tiny best |
| TTEX_01 reference | reference | 1.00 | none | 1.00 | `36.5932` | `6.4327` | `92.4423` | `0.0078` | TTT extrapolate reference |
| ACL2V5_TTBI_01 | w0-only | 1.00 | per-frame static topk | 0.75 | `36.6813` | `6.1168` | `92.4355` | `0.0075` | Rot/Yaw/Final 明显改善，ATE 回退 |
| ACL2V5_TTBI_02 | w0-only | 1.00 | per-frame static topk | 0.90 | `36.6985` | `6.3044` | `92.4400` | `0.0077` | 更温和但 ATE 仍回退 |
| ACL2V5_TTBI_03 | w0-only | 0.75 | none | 1.00 | `37.2723` | `5.9028` | `92.4454` | `0.0075` | EMA w0-only 不崩 scale，但 ATE 明显回退 |
| ACL2V5_TTBI_04 | w0-only | 0.85 | per-frame static topk | 0.90 | `37.0393` | `6.0681` | `92.4440` | `0.0075` | FinalErr 很好，但主指标回退 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOVR_02 reference | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` |
| TTEX_01 reference | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` |
| TTBI_01 w0 top75 | `36.6813` | `2.511` | `29.979` | `30.602` | `30.745` | `3.286` | `31.151060` |
| TTBI_02 w0 top90 | `36.6985` | `4.299` | `29.927` | `30.559` | `30.643` | `3.471` | `31.199518` |
| TTBI_03 w0 EMA0.75 | `37.2723` | `4.065` | `31.028` | `31.610` | `32.006` | `2.997` | `31.263684` |
| TTBI_04 w0 top90+EMA0.85 | `37.0393` | `1.252` | `30.547` | `31.149` | `31.403` | `3.183` | `31.247058` |

20.29 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02/TTEX_01`。
2. branch isolation 是必要的：20.28 all-branch EMA/filter 会让 scale/final error 崩坏，而 20.29 w0-only 全部把 Sim3 scale 稳定在 `31.15-31.26`。这证明 `w1/w2` 不能被粗暴 EMA 或 hard token subset 误伤。
3. “TTT 少写动态区域”仍然有正信号：w0-only top75 把 Rot 从 `6.4307` 降到 `6.1168`，FinalErr 从 `5.118` 降到 `2.511`，Yaw 从 `3.593` 降到 `3.286`。但它让 50/100/200 segment mean 回退，导致 ATE 仍差。
4. w0-only EMA 不再出现 20.28 的 scale disaster，但 ATE 回退到 `37.2723`，说明 EMA 不是当前主突破口；它更像 endpoint/yaw regularizer，不是全局 RMSE 修正。
5. 当前关键问题已经从 “branch 是否要分离” 变成 “哪些 TTT layer 可以被 w0 dynamic/static cue 控制”。下一步立刻做 layer split，不再继续扫 all-layer ratio / EMA。

### 20.30 计划：TTT layer-aware w0-only 写入拆解

固定主线：

```text
cue = C23 past_only
beta = 4.75
write = stage_d_x_dg_inv_sqrt
PRIOR_BRANCH_MASK = 0
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK = 0
```

首批 layer split：

| Run | Layer mode | Hard filter | Ratio | 目的 |
|---|---|---|---:|---|
| ACL2V5_TTLAYER_01 | early | per-frame static topk | 0.75 | 测试早层 w0 dynamic suppression 是否保留 Rot/Final 收益并减少 ATE 回退 |
| ACL2V5_TTLAYER_02 | middle | per-frame static topk | 0.75 | 测试中层是否是真正污染/收益来源 |
| ACL2V5_TTLAYER_03 | late | per-frame static topk | 0.75 | 测试深层 w0 是否主要导致 segment drift |
| ACL2V5_TTLAYER_04 | single layer 6 | per-frame static topk | 0.75 | 细查当前 early/mid 边界附近是否存在单层有效点 |

触发逻辑：

- 若某个 layer mode 明显优于 all-layer w0Top075，再围绕该 layer 做 ratio `0.60/0.90` 与 `explicit_dyn` ranking；
- 若 layer split 仍全部卡在 `36m+`，说明 TTT 写入的 token/layer/rank 调整只能改善姿态/endpoint，不能独立打到 `<30m`，下一步必须把 SWA overlap source selection 或尺度校正纳入主线。

#### 20.30 实验结果：TTT layer-aware w0-only 写入拆解

实现检查：

- 在跑 layer split 前发现并修复了一个真实实现缺陷：hard replay token filter 原先没有完全尊重 `PRIOR_LAYER_MODE`，被禁用的 layer 仍可能按 `prior_flat=ones` 走 hard filter 逻辑；
- 修复后，disabled layer 的 debug 显示 `ttt_replay_token_filter_applied=false`、`ttt_replay_token_filter_layer_disabled=true`，active layer 才显示 tokens `40320 -> 30240`；
- `TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK=0` 仍正常触发 branch-isolated 双 replay，active layer 只替换 `w0`。

运行记录：

- 4 并发，GPU 0-3；
- start `22:31:51`，finish `22:56:32-22:56:59`，约 `24.7-25.1 min`；
- 成功标准：`KITTI01 ATE < 30m`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_layer_split/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_30_diagnostics/`。

| Run | Layer mode | Hard filter | Ratio | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| SWOVR_02 reference | reference | none | 1.00 | `36.5915` | `6.4307` | `92.4416` | `0.0078` | 当前 tiny best |
| TTBI_01 reference | all | per-frame static topk | 0.75 | `36.6813` | `6.1168` | `92.4355` | `0.0075` | all-layer w0 top75 reference |
| ACL2V5_TTLAYER_01 | early | per-frame static topk | 0.75 | `36.7381` | `6.4357` | `92.4416` | `0.0078` | early-only 基本失去 Rot/Final 收益 |
| ACL2V5_TTLAYER_02 | middle | per-frame static topk | 0.75 | `36.6834` | `6.4006` | `92.4417` | `0.0078` | 接近 reference，但未改善 |
| ACL2V5_TTLAYER_03 | late | per-frame static topk | 0.75 | `36.6422` | `6.1506` | `92.4348` | `0.0076` | 本批最好；保留部分 Rot/Final 收益，但 ATE 仍回退 |
| ACL2V5_TTLAYER_04 | single layer 6 | per-frame static topk | 0.75 | `36.6773` | `6.4251` | `92.4426` | `0.0078` | 单层 6 没有独立有效 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWOVR_02 reference | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` |
| TTEX_01 reference | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` |
| TTBI_01 all-layer w0 top75 | `36.6813` | `2.511` | `29.979` | `30.602` | `30.745` | `3.286` | `31.151060` |
| TTLAYER_01 early | `36.7381` | `5.072` | `29.946` | `30.589` | `30.657` | `3.602` | `31.215582` |
| TTLAYER_02 middle | `36.6834` | `4.686` | `29.924` | `30.563` | `30.650` | `3.567` | `31.224567` |
| TTLAYER_03 late | `36.6422` | `2.948` | `29.919` | `30.544` | `30.672` | `3.321` | `31.144459` |
| TTLAYER_04 single6 | `36.6773` | `5.156` | `29.892` | `30.531` | `30.593` | `3.602` | `31.226095` |

20.30 结论：

1. 本批仍没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02/TTEX_01`。
2. layer split 证明 all-layer w0 hard filter 的收益主要来自 late TTT layers：`late` 保留了 Rot/Final 的大部分改善，ATE=`36.6422` 也是本批最好。
3. early/middle/single6 都没有成为有效突破点，说明简单的 “只让某一段 layer 少写动态区域” 不能独立解决 KITTI01 全局 ATE。
4. 但 TTT 写入思路并非完全失败：late-only 把 FinalErr 从 reference `5.118` 降到 `2.948`，Yaw 从 `3.593` 降到 `3.321`，只是 segment mean 仍回退。
5. 下一步不应继续只扫 `D_g` topk ratio；需要换写入规则：用 explicit dyn cue 做更强 dynamic veto / focal-style weighting，并考虑 TTT 的更新数学，尽量改变 update direction 而不是只改 token lr mass。

### 20.31：TTT 写入实现检查与加速修复

用户问题：`loger/pipeline/hybrid_memory_controller.py` 里的 TTT 写入实现是否有 bug。

检查结论：

1. HMC 的主 commit 链路是正确的：`P_ttt_write` 会从 control prior 进入 `TTTWriteController.run(...)`，并且输出的 `w0/w1/w2/history` 会成为下一 chunk 的 `ttt_state`。
2. 真实 correctness bug 不在 HMC 主体，而在 `TTTWriteController`：hard replay token filter 曾经没有完全尊重 layer-disable，这会污染 layer-aware TTT 实验。已修复并通过 20.30 debug 验证。
3. HMC 里发现一个重要效率问题：`hmc_commit_mode=probe_ttt_write` 时，controlled pass 会先跑一次 controlled-cache semantic TTT replay，但最后真正提交的是 probe/native cache 的第二次 replay；第一遍 replay 被丢弃，只用于 debug，浪费时间。
4. 已新增 `skip_ttt_write_replay` 参数：当 commit mode 是 `probe_ttt_write` 时，controlled pass 跳过这次被丢弃的 replay，仍然保留 controlled geometry/read path；真正的 probe/native TTT write commit 不变。
5. 语法检查已通过：`py_compile loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py`。

后续影响：

- 这个修复不应改变实验结果，只减少后续 `probe_ttt_write` 实验耗时；
- short smoke `ACL2V5_HMC_SKIP_SMOKE2_C23past_b475_e032` 已通过：log 显示 `controlled TTT write replay: skipped`，随后仍打印 `probe/native TTT write commit` 并提交 state hash `5bfc5b8acfda0060`。

### 20.32 计划：explicit / union dynamic hard-token replay ranking

启动时间：`2026-05-06 23:07:29 +08`

当前状态：

- v5 计划内已完成大量 TTT/SWA 写入策略，但仍未达到用户更新后的成功标准 `KITTI01 ATE < 30m`；
- 当前 full KITTI01 最好仍在 `36.59m` 左右，远高于目标；
- 20.30 修复了 hard token filter 的 layer-disable bug，并确认 `probe_ttt_write` 下可跳过被丢弃的 controlled semantic replay；
- 因此下一批优先验证“动态区域是否准确”：不用连续 soft prior，而是在 TTT replay 前直接改变参与构造更新矩阵的 token 集合。

固定 base：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
PRIOR_BRANCH_MASK = 0
TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK = 0
TTT_WRITE_REPLAY_TOKEN_FILTER_MODE = per_frame_static_topk
success target = KITTI01 ATE < 30m
```

本批只跑 4 个 full run，保持 4 并发安全甜点位：

| Run | Layer | Hard filter ranking | Ratio | 目的 |
|---|---|---|---:|---|
| `ACL2V5_TTHR_01` | late | `stage_d_x_exp_inv_sqrt` | 0.75 | 用 explicit dyn 替代 D_g 做 hard token ranking，测试显式动态是否更准 |
| `ACL2V5_TTHR_02` | late | `stage_d_x_dg_exp_inv_sqrt` | 0.75 | D_g 与 explicit dyn 都作为 static eligibility，测试双 cue 是否更干净 |
| `ACL2V5_TTHR_03` | late | `stage_d_x_union_dyn_inv` | 0.75 | 任一 cue 认为动态就排除，测试强 veto 是否过约束 |
| `ACL2V5_TTHR_04` | late | `stage_d_x_dg_exp_inv_sqrt` | 0.90 | 更温和 hard filter，测试 0.75 是否删掉过多几何 token |

Gate：

- 若任何 run 达到 `ATE < 30m`，立即停止 v5 当前方向并做 repeat / cross-sequence；
- 若未达标但超过 `SWOVR_02 = 36.5915 / 6.4307`，保留为新写入候选；
- 若全部回退，则 hard-token dynamic ranking 线基本收口，后续必须转向更结构化的 TTT replay objective / per-head dynamic reliability，而不是继续扫同类 mask。

#### 20.32 实验结果：explicit / union dynamic hard-token replay ranking

运行记录：

- 4 并发，GPU 0-3；
- start `23:08:18`，finish `23:31:12-23:32:17`，约 `23.0-24.0 min`；
- host RAM 安全，没有触发 swap / OOM；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_hard_exp_union/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_32_diagnostics/`。

| Run | Layer | Hard filter ranking | Ratio | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| SWOVR_02 reference | reference | none | 1.00 | `36.5915` | `6.4307` | `92.4416` | `0.0078` | 当前 tiny best |
| TTLAYER_03 reference | late | D_g static topk | 0.75 | `36.6422` | `6.1506` | `92.4348` | `0.0076` | late-only reference |
| ACL2V5_TTHR_01 | late | `stage_d_x_exp_inv_sqrt` | 0.75 | `36.7582` | `6.0567` | `92.4376` | `0.0075` | explicit dyn 排序改善 Rot，但 ATE 回退 |
| ACL2V5_TTHR_02 | late | `stage_d_x_dg_exp_inv_sqrt` | 0.75 | `36.7105` | `6.1650` | `92.4344` | `0.0076` | 双 cue 比 pure explicit 好，但仍回退 |
| ACL2V5_TTHR_03 | late | `stage_d_x_union_dyn_inv` | 0.75 | `36.6367` | `6.0739` | `92.4361` | `0.0075` | 本批 ATE 最好，但仍未超过 SWOVR_02 |
| ACL2V5_TTHR_04 | late | `stage_d_x_dg_exp_inv_sqrt` | 0.90 | `36.6685` | `6.2811` | `92.4402` | `0.0077` | 更温和保留 token 没有解决 ATE |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTHR_01 | `36.7582` | `2.523` | `30.027` | `30.647` | `30.781` | `3.228` | `31.164917` |
| TTHR_02 | `36.7105` | `3.076` | `29.972` | `30.593` | `30.708` | `3.332` | `31.141160` |
| TTHR_03 | `36.6367` | `3.079` | `29.881` | `30.504` | `30.617` | `3.272` | `31.154121` |
| TTHR_04 | `36.6685` | `4.516` | `29.887` | `30.521` | `30.588` | `3.465` | `31.200317` |

20.32 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02 = 36.5915 / 6.4307`。
2. explicit dyn cue 对 “少写动态区域” 有方向性帮助：Rot 最好的是 pure explicit ranking 的 `6.0567`，FinalErr 也降到 `2.523m`；但 ATE 明显回退，说明 explicit dyn 更像 orientation / endpoint filter，而不是全局 ATE filter。
3. `union_dyn_inv` 是本批最稳的 hard ranking，ATE=`36.6367`、Rot=`6.0739`，接近 TTLAYER_03，但仍没有突破 reference。
4. ratio `0.90` 没有把 ATE 拉回 reference，说明问题不是简单 “删太多 token”；hard-token dynamic ranking 这条线基本收口。
5. 后续 TTT 最高优先级应从 token selection 转向 replay update direction：动态 token 不能只是删掉或降权，还要测试把动态 token 的 `K/V` 写入目标拉向静态原型、EMA teacher 或分支/层专属目标。

### 20.33 计划：TTT normalized anti-dynamic replay prior

启动时间：`2026-05-06 23:42:00 +08`

动机：

- 20.32 说明 hard-token dynamic ranking 没有突破，且 ratio 变化不是关键；
- 检查 TTT replay 机制后，关键点仍是 `token_prior * lr` 先进入 update aggregate，随后经过 `zeropower` 和 fast-weight norm restoration，普通 `[0,1]` 降权很容易被折叠；
- 因此新增 `TTT_WRITE_PRIOR_TRANSFORM_MODE`，允许在 TTT replay 前把本层 prior range 归一化后构造 signed / anti-dynamic prior，让低静态 eligibility token 产生小的负 lr contribution，测试动态区域是否应该被 “反写 / unlearn”，而不只是少写。

实现：

```text
anti_dynamic_norm:
    p_norm = (p - min(p)) / (max(p) - min(p))
    p_ttt  = p_norm - anti_scale * (1 - p_norm)
```

smoke 结论：

- `ACL2V5_SMOKE_ANTI_NORM_C23pairall_late_a025_e032` 通过；
- active late layer 出现 `ttt_write_prior_negative_mass ~= 0.199`，说明低 prior token 确实进入了负 lr contribution；
- disabled layer 虽然也记录 transform debug，但 `branch*_prior_enabled=false`，不会参与更新。

固定 base：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read intervention = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
PRIOR_BRANCH_MASK = 0
success target = KITTI01 ATE < 30m
```

本批只跑 4 个 full run：

| Run | Layer | Write score | Transform | Anti scale | 目的 |
|---|---|---|---|---:|---|
| `ACL2V5_TTANTI_01` | late | `stage_d_x_union_dyn_inv` | `anti_dynamic_norm` | 0.25 | 在 TTHR_03 最稳 ranking 上测试轻度反写 |
| `ACL2V5_TTANTI_02` | late | `stage_d_x_union_dyn_inv` | `anti_dynamic_norm` | 0.50 | 测更强 dynamic anti update 是否有大信号 |
| `ACL2V5_TTANTI_03` | late | `stage_d_x_dg_exp_inv_sqrt` | `anti_dynamic_norm` | 0.25 | 测 D_g+explicit eligibility 是否比 union 更稳 |
| `ACL2V5_TTANTI_04` | all | `stage_d_x_union_dyn_inv` | `anti_dynamic_norm` | 0.25 | 测 anti update 是否只适合 late，还是全层有用 |

Gate：

- 若达到 `ATE < 30m`，立即 repeat；
- 若超过 `SWOVR_02 = 36.5915` 或至少超过 `TTHR_03 = 36.6367`，保留并做局部曲线；
- 若全部回退，说明负 lr / anti-write 不是当前主突破方向，下一步转向更具体的 per-branch/layer target replacement 或 debug worst chunks。

#### 20.33 实验结果：normalized anti-dynamic replay prior

运行记录：

- 4 并发，GPU 0-3；
- start `23:42:16`，finish `00:05:15-00:06:06`，约 `23.0-23.8 min`；
- host RAM 峰值安全，未触发 OOM / swap；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_anti_dynamic_prior/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_33_diagnostics/`。

| Run | Layer | Write score | Anti scale | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| SWOVR_02 reference | reference | stage_d | 0.00 | `36.5915` | `6.4307` | `92.4416` | `0.0078` | 当前 tiny best |
| TTHR_03 reference | late | `stage_d_x_union_dyn_inv` hard top75 | 0.00 | `36.6367` | `6.0739` | `92.4361` | `0.0075` | 上批 best hard ranking |
| ACL2V5_TTANTI_01 | late | `stage_d_x_union_dyn_inv` | 0.25 | `36.7331` | `5.7736` | `92.4321` | `0.0078` | Rot/Yaw 明显改善，但 ATE 回退 |
| ACL2V5_TTANTI_02 | late | `stage_d_x_union_dyn_inv` | 0.50 | `37.8443` | `8.9071` | `92.4585` | `0.0136` | 反写过强，trajectory 破坏 |
| ACL2V5_TTANTI_03 | late | `stage_d_x_dg_exp_inv_sqrt` | 0.25 | `36.9786` | `5.8978` | `92.4348` | `0.0078` | 比 union 更差 |
| ACL2V5_TTANTI_04 | all | `stage_d_x_union_dyn_inv` | 0.25 | `36.8152` | `5.7780` | `92.4321` | `0.0078` | all-layer 不优于 late |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTANTI_01 | `36.7330` | `6.151` | `30.385` | `30.966` | `31.412` | `2.805` | `31.114262` |
| TTANTI_02 | `37.8443` | `39.262` | `32.453` | `33.002` | `34.971` | `5.589` | `31.322407` |
| TTANTI_03 | `36.9786` | `6.131` | `30.703` | `31.283` | `31.728` | `2.912` | `31.147827` |
| TTANTI_04 | `36.8152` | `5.925` | `30.491` | `31.074` | `31.509` | `2.818` | `31.116517` |

20.33 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02` 或 `TTHR_03`。
2. normalized anti-dynamic prior 的机制确实生效：smoke 中 active late layer 有约 `19.9%` token 进入负 lr contribution；但 full run 显示它主要改善 orientation/yaw，不能改善 global ATE。
3. `TTANTI_01` 的 Rot=`5.7736`、Yaw=`2.805` 是很干净的姿态信号，但 FinalErr=`6.151` 和 segment mean 回退，说明 “反写动态” 会牺牲全局连续性。
4. `anti=0.50` 明显过强，FinalErr=`39.262m`，这条线不能继续加大强度。
5. all-layer anti-write 不如 late-only，说明负更新更像高层姿态 regularizer，而不是全层 TTT memory policy。
6. 结论更新：动态区域少写/反写这个粗方向有一定姿态收益，但不是当前 `<30m` 的关键；下一步应把重点转到最坏区间 `[200,300)` 的 chunk-level diagnosis，确认到底是 read cue 失败、scale/Sim3 drift、还是 KITTI eval 对 early segment 的系统误差主导。

### 20.34 计划：KITTI01 ATE<30 bottleneck 与定点 TTT 因果测试

启动时间：`2026-05-07 00:09 +08`

目标仍按用户更新后的成功标准：

```text
KITTI01 ATE RMSE < 30m
```

先做低成本 trajectory diagnosis，把 v4 locked、当前 v5 tiny best、最近 TTT 写入变体放进同一 Sim3 诊断目录：

```text
results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_34_bottleneck_diagnostics/
```

对比对象：

| Run | 角色 |
|---|---|
| `C23past_v4` | v4 locked baseline |
| `SWOVR_02` | 当前 v5 tiny best |
| `TTEX_01` | native mix reference |
| `TTHR_03` | hard dynamic token ranking best |
| `TTANTI_01` | anti-dynamic signed update best orientation diagnostic |

诊断初步发现：

| Run | ATE RMSE | Worst 100f segment | Worst 100f ATE | 如果 `[200,300)` 修到 30m 的全局 ATE |
|---|---:|---|---:|---:|
| `C23past_v4` | `38.3706` | `[200,300)` | `74.7324` | `32.35` |
| `SWOVR_02` | `36.5915` | `[200,300)` | `78.7547` | `29.28` |
| `TTEX_01` | `36.5932` | `[200,300)` | `78.7501` | `29.28` |
| `TTHR_03` | `36.6367` | `[200,300)` | `78.6034` | `29.37` |
| `TTANTI_01` | `36.7330` | `[200,300)` | `77.8628` | `29.67` |

关键判断：

1. 当前 `<30m` 的主要瓶颈高度集中在 `[200,300)`；不是全序列均匀差一点。
2. `SWOVR_02` 如果只把 `[200,300)` 这 100 帧压到 `30m`，全局 ATE 理论上可到 `29.28m`，刚好过成功标准。
3. 过去多种 TTT/SWA 写入策略主要改善 yaw/final/后段，但 `[200,300)` 基本仍在 `78m` 左右，说明继续均匀扫写入权重收益很低。
4. 原本想用 reset 做定点因果测试，但用户指出 reset 机制必须和 LoGeR 原生对齐，因此 **取消所有新增 reset 变量**，不再把 reset 作为 v5 实验方向。
5. 下一步只做“不改变 reset 机制”的 TTT 写入策略诊断：冻结 / 丢弃指定 chunk 的 TTT write commit，判断进入 `[200,300)` 前的写入是否污染 fast weights。

为此只保留一个写入侧诊断开关：

```text
--ttt_freeze_chunks    # 指定 chunk 的 TTT write commit 丢弃，SWA/ref 等其它 state 保留
```

脚本环境变量：

```text
TTT_FREEZE_CHUNKS
```

执行修正：

- `ACL2V5_TTLOC_01 / 02 / 04` 已启动但涉及 `RESET_AT_CHUNKS`，按用户要求立即停止，不计入结果；
- `--reset_at_chunks` 与 `RESET_AT_CHUNKS` 已从代码和脚本撤掉；
- `ACL2V5_TTLOC_03` 只使用 `TTT_FREEZE_CHUNKS=4,5,6`，不改 LoGeR reset 机制，因此保留继续跑。

固定 base 使用当前 tiny best 附近的配置：

```text
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
success target = KITTI01 ATE < 30m
```

本批先跑 4 个 full run：

| Run | Intervention | 目的 |
|---|---|---|
| `ACL2V5_TTLOC_03` | `TTT_FREEZE_CHUNKS=4,5,6` | 保留原生 reset / history，只丢弃进入最坏段前 3 个 chunk 的 TTT 写入 |

后续若 `TTLOC_03` 有信号，再补只改写入、不改 reset 的小矩阵：

| Candidate | Intervention | 目的 |
|---|---|---|
| `freeze6` | `TTT_FREEZE_CHUNKS=6` | 只冻结最坏段前一 chunk 的写入 |
| `freeze7_9` | `TTT_FREEZE_CHUNKS=7,8,9` | 不重置，只禁止最坏段内继续写入 |
| `freeze4_9` | `TTT_FREEZE_CHUNKS=4,5,6,7,8,9` | 判断 `[150,300)` 一整段写入是否污染后续 |

Gate：

- 若任一 run 达到 `ATE < 30m`，立即 repeat；
- 若 `[200,300)` 明显下降但全局未达标，则继续做可泛化的 chunk-quality-triggered freeze / write suppression；
- 若 `[200,300)` 基本不动，则说明当前 TTT fast-weight writing 不是该瓶颈主因，必须转向 read cue / local pose estimation / scale drift 方向。

#### 20.34 实验结果：不改 reset 的 TTT write freeze 诊断

运行记录：

- `RESET_EVERY=5` 保持 LoGeR 原生对齐；所有涉及 `RESET_AT_CHUNKS` 的 partial run 已停止，不计入结果；
- `--reset_at_chunks` / `RESET_AT_CHUNKS` 已从代码和脚本撤掉；
- 4 并发，GPU 0-3；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_local_freeze/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_34_freeze_diagnostics/`。

| Run | Intervention | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `SWOVR_02` reference | none | `36.5915` | `6.4307` | `92.4416` | `0.0078` | 当前 tiny best |
| `ACL2V5_TTLOC_03` | freeze chunks `4,5,6` | `60.8135` | `8.4670` | `92.7027` | `0.0107` | 全局崩，但 `[200,300)` 大幅下降 |
| `ACL2V5_TTLOC_05` | freeze chunk `6` | `43.2789` | `7.5080` | `92.5615` | `0.0096` | 全局明显回退 |
| `ACL2V5_TTLOC_06` | freeze chunks `7,8,9` | `71.3949` | `7.4678` | `92.7034` | `0.0094` | 最坏段更差 |
| `ACL2V5_TTLOC_07` | freeze chunks `4-9` | `100.6355` | `7.5328` | `92.9075` | `0.0095` | 全局崩 |

Segment diagnostics：

| Run | Overall ATE | `[0,200)` | `[200,300)` | `[200,400)` | `[400,600)` | FinalErr | Yaw RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SWOVR_02` | `36.5915` | `31.34` | `78.75` | `57.75` | `38.70` | `5.118` | `3.593` |
| `TTLOC_03 freeze4_6` | `60.8135` | `90.13` | `25.07` | `31.79` | `86.10` | `21.283` | `5.645` |
| `TTLOC_05 freeze6` | `43.2789` | `57.20` | `42.57` | `34.88` | `65.00` | `13.076` | `4.633` |
| `TTLOC_06 freeze7_9` | `71.3949` | `90.60` | `105.92` | `85.46` | `84.29` | `20.085` | `4.673` |
| `TTLOC_07 freeze4_9` | `100.6355` | `145.09` | `59.67` | `49.96` | `144.66` | `29.735` | `4.836` |

20.34 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，且所有 freeze run 全局都明显差于 `SWOVR_02`。
2. `freeze4_6` 把核心瓶颈 `[200,300)` 从 `78.75m` 压到 `25.07m`，说明进入最坏段前后的 TTT 写入确实有强因果影响。
3. 但当前 `TTT_FREEZE_CHUNKS` 是粗诊断开关：它把 `state_next.ttt_state` 回滚到上一 chunk，等于同时丢弃 LoGeR native TTT 更新和 semantic write；这会破坏全局连续性，不能作为最终写入策略。
4. 因此下一步不能继续改 reset，也不能继续用整块 freeze 当策略；应改成 **保留 native LoGeR TTT，只缩放 semantic write delta**。这更贴近“少写动态区域/少写风险 chunk”的真实问题。

### 20.35 计划：native-preserving semantic TTT write delta scaling

启动时间：`2026-05-07`

代码修正：

- 新增 `--ttt_semantic_write_scale_chunks` / `TTT_SEMANTIC_WRITE_SCALE_CHUNKS`；
- 格式为 `CHUNK:SCALE`，例如 `4:0.25,5:0.25,6:0.25`；
- 只在 `probe_ttt_write` commit 后，把 semantic TTT state 按

```text
W_commit = W_native_provisional + scale * (W_semantic - W_native_provisional)
```

并做 fast-weight row norm restoration；
- `scale=0` 表示该 chunk 只保留 native LoGeR TTT，不提交 semantic write delta；
- `scale=1` 等价当前 semantic write；
- reset 机制完全不变，仍使用 LoGeR 对齐的 `RESET_EVERY=5`。

固定 base 延续 `SWOVR_02`：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
RESET_EVERY = 5
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
SWA overlap source replace = source/v last alpha0.25
success target = KITTI01 ATE < 30m
```

本批只跑 4 个 full run：

| Run | Semantic write scale | 目的 |
|---|---|---|
| `ACL2V5_TTSEM_01` | chunks `4,5,6 -> 0.00` | 最接近 freeze4_6，但保留 native TTT |
| `ACL2V5_TTSEM_02` | chunks `4,5,6 -> 0.25` | 测轻度 semantic write 是否兼顾 `[200,300)` 与全局连续性 |
| `ACL2V5_TTSEM_03` | chunks `4,5,6 -> 0.50` | 测中度 semantic write |
| `ACL2V5_TTSEM_04` | chunk `6 -> 0.00` | 最小定点抑制，只处理最坏段前一 chunk |

空闲 GPU 补充矩阵：

| Run | Semantic write scale | 目的 |
|---|---|---|
| `ACL2V5_TTSEM_05` | chunks `4,5,6 -> 0.10` | 在 `0.00/0.25` 之间补轻度 scale |
| `ACL2V5_TTSEM_06` | chunks `4,5,6 -> 0.75` | 判断是否只需很小幅弱写 |
| `ACL2V5_TTSEM_07` | chunks `4,5 -> 0.00` | 定位 chunk6 是否必要 |
| `ACL2V5_TTSEM_08` | chunks `5,6 -> 0.00` | 定位 chunk4 是否主要造成前段崩坏 |

Gate：

- 若任一 run 达到 `ATE < 30m`，立即 repeat；
- 若 `[200,300)` 降低且全局 ATE 不崩，继续做 chunk-quality-triggered scaling；
- 若全局仍崩或 `[200,300)` 不动，说明不是 “semantic delta 过强” 单因子，需要转向更细的 branch/layer/chunk 规则。

#### 20.35 实验结果：native-preserving semantic TTT write delta scaling

运行记录：

- `RESET_EVERY=5` 保持 LoGeR 原生对齐，reset 机制没有作为变量；
- 4 并发完成首批 `TTSEM_01-04`，随后用空闲 GPU 补 `TTSEM_05-08`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_semantic_chunk_scale/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_35_semantic_scale_diagnostics/`。

| Run | Semantic write scale | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `SWOVR_02` reference | none | `36.5915` | `6.4307` | `92.4416` | `0.0078` | 当前 tiny best |
| `ACL2V5_TTSEM_01` | chunks `4,5,6 -> 0.00` | `36.6783` | `6.4151` | `92.4417` | `0.0078` | 不达标，ATE 回退 |
| `ACL2V5_TTSEM_02` | chunks `4,5,6 -> 0.25` | `36.7132` | `6.4030` | `92.4412` | `0.0078` | ATE 继续回退 |
| `ACL2V5_TTSEM_03` | chunks `4,5,6 -> 0.50` | `36.5976` | `6.3988` | `92.4409` | `0.0078` | 最接近 reference，但仍未超过 |
| `ACL2V5_TTSEM_04` | chunk `6 -> 0.00` | `36.6660` | `6.4041` | `92.4420` | `0.0078` | 单 chunk 抑制无效 |
| `ACL2V5_TTSEM_05` | chunks `4,5,6 -> 0.10` | `36.6842` | `6.4093` | `92.4415` | `0.0078` | 不如 reference |
| `ACL2V5_TTSEM_06` | chunks `4,5,6 -> 0.75` | `36.6972` | `6.3738` | `92.4407` | `0.0078` | Rot 最好，但 ATE 回退 |
| `ACL2V5_TTSEM_07` | chunks `4,5 -> 0.00` | `36.7044` | `6.4209` | `92.4407` | `0.0078` | 不如 reference |
| `ACL2V5_TTSEM_08` | chunks `5,6 -> 0.00` | `36.6783` | `6.4151` | `92.4417` | `0.0078` | 与 `TTSEM_01` 等价 |

Segment diagnostics：

| Run | Overall ATE | `[0,100)` | `[100,200)` | `[200,300)` | `[300,400)` | `[400,500)` | `[500,600)` | Worst 100f |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SWOVR_02` | `36.5915` | `36.08` | `34.04` | `78.75` | `21.66` | `55.92` | `36.99` | `[200,300)=78.75` |
| `TTSEM_01` | `36.6783` | `35.93` | `34.28` | `78.92` | `21.74` | `55.97` | `37.15` | `[200,300)=78.92` |
| `TTSEM_02` | `36.7132` | `35.83` | `34.26` | `78.83` | `21.75` | `56.13` | `37.48` | `[200,300)=78.83` |
| `TTSEM_03` | `36.5976` | `35.98` | `34.10` | `78.70` | `21.71` | `56.04` | `37.01` | `[200,300)=78.70` |
| `TTSEM_04` | `36.6660` | `36.06` | `34.13` | `78.97` | `21.77` | `56.05` | `37.04` | `[200,300)=78.97` |
| `TTSEM_05` | `36.6842` | `35.87` | `34.27` | `78.82` | `21.69` | `56.15` | `37.26` | `[200,300)=78.82` |
| `TTSEM_06` | `36.6972` | `35.77` | `34.27` | `78.85` | `21.73` | `56.15` | `37.36` | `[200,300)=78.85` |
| `TTSEM_07` | `36.7044` | `35.79` | `34.27` | `78.81` | `21.78` | `56.22` | `37.39` | `[200,300)=78.81` |
| `TTSEM_08` | `36.6783` | `35.93` | `34.28` | `78.92` | `21.74` | `55.97` | `37.15` | `[200,300)=78.92` |

Trajectory summary：

| Run | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|
| `SWOVR_02` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` |
| `TTSEM_01` | `5.019` | `29.909` | `30.543` | `30.602` | `3.581` | `31.219839` |
| `TTSEM_02` | `4.938` | `29.950` | `30.588` | `30.653` | `3.575` | `31.207553` |
| `TTSEM_03` | `4.694` | `29.821` | `30.453` | `30.536` | `3.559` | `31.208102` |
| `TTSEM_04` | `4.968` | `29.862` | `30.495` | `30.554` | `3.578` | `31.221088` |
| `TTSEM_05` | `4.789` | `29.916` | `30.552` | `30.621` | `3.571` | `31.211826` |
| `TTSEM_06` | `4.785` | `29.928` | `30.565` | `30.631` | `3.540` | `31.204932` |
| `TTSEM_07` | `5.009` | `29.931` | `30.570` | `30.620` | `3.588` | `31.204857` |
| `TTSEM_08` | `5.019` | `29.909` | `30.543` | `30.602` | `3.581` | `31.219839` |

20.35 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02`。
2. native-preserving semantic delta scaling 比 20.34 的粗 freeze 安全得多：全局不会崩，FinalErr / Yaw 甚至有小幅改善。
3. 但核心瓶颈 `[200,300)` 仍稳定在 `78.7-79.0m`，说明 20.34 的 `[200,300)` 大幅下降不是来自“少提交 semantic delta”本身，而更像来自完全擦掉 native TTT continuity / fast-weight state。
4. 因此不能继续沿 reset 或整块 freeze 走；reset 需要和 LoGeR 对齐。下一步应保持 reset 不变，转向更底层的 TTT 写入规则：post-replay native-direction gate、correction cap，以及 overlap token scope。

### 20.36 计划：不改 reset 的 TTT native-direction / overlap 写入小矩阵

启动时间：`2026-05-07`

固定 base 延续 `SWOVR_02`，reset 机制继续保持：

```text
RESET_EVERY = 5
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
beta = 4.75
commit = probe_ttt_write
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
SWA overlap source replace = source/v last alpha0.25
success target = KITTI01 ATE < 30m
```

本批先跑 4 个 full run：

| Run | Intervention | 目的 |
|---|---|---|
| `ACL2V5_TTNDG_01` | `native_delta_gate=cosine_soft`, min cos `0.25`, fallback `0.25`, branch0 | 只保留与 native TTT 更新方向相容的 semantic correction |
| `ACL2V5_TTNDG_02` | `native_delta_gate=cap`, cap ratio `0.25`, branch0 | 防止 semantic correction 相对 native 更新幅度过大 |
| `ACL2V5_TTNDG_03` | `native_delta_gate=cosine_cap`, min cos `0.00`, fallback `0.00`, cap ratio `0.50`, branch0 | 同时剔除反向 correction 并限幅 |
| `ACL2V5_TTOVL_01` | `TTT_WRITE_TOKEN_SCOPE=tail_overlap_no_boost` | 验证 TTT 写入是否应更关注 overlap seam 的动态区域 |

Gate：

- 若任一 run 达到 `ATE < 30m`，立即 repeat；
- 若 ATE 未达标但 `[200,300)` 明显下降，继续做 chunk-aware / overlap-aware 写入；
- 若继续停在 `36m` 平台，说明当前 pair/read + TTT 写入规则无法打穿瓶颈，需要重新审视 TTT 更新机制本身或换更准确动态 cue。

#### 20.36 实验结果：不改 reset 的 TTT native-direction / overlap 写入小矩阵

运行记录：

- `RESET_EVERY=5` 固定保持 LoGeR 原生对齐，reset 机制不作为实验变量；
- 4 并发完成 `TTNDG_01-03` 与 `TTOVL_01`；
- 结果目录：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_native_direction_overlap/`；
- trajectory diagnostics：`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/20_36_native_direction_overlap_diagnostics/`。

| Run | Intervention | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---:|---:|---:|---:|---|
| `SWOVR_02` reference | none | `36.5915` | `6.4307` | `92.4416` | `0.0078` | 当前 tiny best |
| `ACL2V5_TTNDG_01` | native delta `cosine_soft`, min cos `0.25`, fallback `0.25`, branch0 | `36.6476` | `6.3864` | `92.4413` | `0.0078` | Rot/Yaw 小幅改善，ATE 回退 |
| `ACL2V5_TTNDG_02` | native delta `cap`, cap ratio `0.25`, branch0 | `36.6840` | `6.3842` | `92.4398` | `0.0078` | FinalErr 更好，ATE 回退 |
| `ACL2V5_TTNDG_03` | native delta `cosine_cap`, min cos `0.00`, fallback `0.00`, cap ratio `0.50`, branch0 | `36.6768` | `6.3854` | `92.4404` | `0.0078` | ATE 回退 |
| `ACL2V5_TTOVL_01` | `TTT_WRITE_TOKEN_SCOPE=tail_overlap_no_boost` | `36.6667` | `6.3562` | `92.4413` | `0.0078` | 本批 Rot 最好，但 ATE 回退 |

Trajectory summary：

| Run | ATE RMSE | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | Sim3 scale | Worst 100f |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` | `[200,300)=78.75` |
| `TTNDG_01` | `36.6477` | `4.946` | `29.879` | `30.517` | `30.587` | `3.572` | `31.216122` | `[200,300)=78.89` |
| `TTNDG_02` | `36.6840` | `4.678` | `29.911` | `30.545` | `30.612` | `3.549` | `31.197725` | `[200,300)=78.82` |
| `TTNDG_03` | `36.6768` | `4.718` | `29.919` | `30.556` | `30.630` | `3.565` | `31.203627` | `[200,300)=78.81` |
| `TTOVL_01` | `36.6667` | `4.848` | `29.851` | `30.490` | `30.564` | `3.533` | `31.218201` | `[200,300)=78.99` |

20.36 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02`。
2. post-replay native-direction gate / cap 主要起到 regularization：Rot、Yaw、FinalErr 有小幅收益，但 ATE 与最坏 `[200,300)` 段没有改善。
3. `tail_overlap_no_boost` 支持“TTT 写入更关注 overlap seam 会改善 orientation”的判断，但它没有降低 `[200,300)`，因此不是当前 ATE 瓶颈的主因。
4. 和 20.35 一致，保留 native LoGeR TTT 连续性后，各种“缩小/限幅 semantic correction”的策略都无法复现 20.34 freeze 的 `[200,300)` 大幅下降。20.34 的信号更像是 fast-weight state 结构被整体改写后的副作用，不能通过改变 reset 或粗 freeze 作为最终方案。
5. 后续继续探索时，reset 机制固定不变；下一步只允许修改 TTT/SWA 的写入内容、写入 token/branch/layer 规则、或 overlap-frame 动态估计方式。

### 20.37 更正：TTT replay prior 对齐不是 patch-token bug

启动时间：`2026-05-07`

用户约束：

- reset 机制必须和 LoGeR 对齐；
- 因此本批继续固定 `RESET_EVERY=5`，不新增、不修改 reset 行为。

中途检查时一度怀疑 TTT write prior 存在 patch-token 对齐偏移：

```text
GeometryOutput.token_type / A_tok 是 full token layout:
    [register tokens, role token, patch tokens] * T

初步 debug:
    A_tok/token_type = 40320 tokens
    special tokens = 192
    patch tokens = 40128
    TTT replay cache l = 40320
```

但继续沿 `loger/models/pi3.py` 和 `loger/models/ttt.py` 深入检查后，确认正常 Pi3 TTT replay cache 使用的是 **full decoder token layout**，不是 patch-only：

- `decode` 先拼接 register/role/patch token，再进入 decoder；
- TTT 模块以 `hidden.view(B, N, hw, -1)` 形式接收 token，其中 `hw` 包含 special tokens；
- `loger/models/ttt.py` 的 replay `k/v/lr` 维度来自 `b t l d -> b (t l) d`，因此 `cache_l=40320` 与 full token prior 长度相同。

因此：

- 原本 direct `A_tok[:cache_l]` / exact-length 对齐是正常主路径；
- 只在 cache 本身变成 patch-only 的 diagnostic path 中才应该按 `token_type == PATCH` 抽取；
- `ACL2V5_TTALIGN_01-04` patch-token 对齐重测在运行中被主动中止，partial 目录不计入结果；
- 没有 TTALIGN 指标进入 v5 表格，也不作为结论依据；
- `loger/pipeline/ttt_write_controller.py` 已改成更稳健的 direct-length-first alignment：
  1. `prior.numel() == cache_l` 时使用 `direct_length`；
  2. 只有 replay cache 长度等于 patch-token 数时，才使用 `patch_token_type`；
  3. 其余情况回退 legacy prefix/pad。

20.37 结论：

1. 当前 HMC main path 没有发现 TTT prior/token alignment bug。
2. TTALIGN patch-token 版本不符合 LoGeR 正常 TTT cache 语义，因此中止是正确的。
3. reset 机制没有变，仍固定为 LoGeR 对齐的 `RESET_EVERY=5`。
4. 后续继续探索时，不再沿 patch-token alignment 方向消耗 full run；重点转向 TTT 写入目标 / replay feature 内容 / branch-layer 策略。

### 20.38 计划：不改 reset 的 TTT replay feature 写入目标小矩阵

启动时间：`2026-05-07`

动机：

- 20.35 / 20.36 说明“只缩放 semantic write delta / 限幅 / native-direction gate”无法复现 20.34 freeze diagnostic 的 `[200,300)` 大幅下降；
- soft lr prior 会被 zeropower / norm restoration 部分吸收，可能不够改变 fast-weight 更新方向；
- 下一步优先改变 TTT replay 的 **K/V 更新内容**：对低静态 prior 的 token，把 replay K/V residual 向静态 token centroid 收缩，而不是只降低 lr。

固定不变量：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
RESET_EVERY = 5
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
SWA overlap source replace = source/v last alpha0.25
success target = KITTI01 ATE < 30m
```

本批只改 TTT replay feature gate，其他保持 `SWOVR_02` active base：

| Run | Write score | TTT feature gate | Layer/branch | 目的 |
|---|---|---|---|---|
| `ACL2V5_TTFG_01` | `stage_d_x_dg_inv_sqrt` | `kv_centered rho0.10 min0.85` | late / branch0 | 同时收缩 K/V dynamic residual，测试是否改变 update direction |
| `ACL2V5_TTFG_02` | `stage_d_x_dg_inv_sqrt` | `kv_centered rho0.20 min0.70` | late / branch0 | 更强 K/V 内容门控 |
| `ACL2V5_TTFG_03` | `stage_d_x_dg_inv_sqrt` | `v_centered rho0.10 min0.85` | late / branch0 | 只改 value 写入目标 |
| `ACL2V5_TTFG_04` | `stage_d_x_dg_inv_sqrt` | `k_centered rho0.10 min0.85` | late / branch0 | 只改 key/update matching 方向 |

Gate：

- 若任一 run 达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若超过 `SWOVR_02 = 36.5915`，保留为新 active base 并围绕它继续 TTT/SWA 写入；
- 若全部回退，说明简单 centroid feature shrink 不够，下一步需要更大胆的 branch/layer 分策略或 TTT replay 目标替换，而不是继续调 lr 权重。

#### 20.38 实验结果：TTT replay feature 写入目标小矩阵

运行记录：

- smoke `ACL2V5_TTFG_SMOKE_C23pairall_lateW0_kvcenter_r010`：`02:19:13 -> 02:22:12`，约 `3.0 min`；
- full batch：4 并发 GPU0-3，`02:23:09 -> 02:47:30`，约 `24.4 min` 完成 4 个 full run；
- 峰值 host RAM 约 `239GiB used / 260GiB available`，4 并发安全；
- `RESET_EVERY=5` 固定，reset 机制未改；
- smoke / full debug 确认 late 层使用 `ttt_prior_alignment_mode=direct_length`，feature gate 在 branch0 生效。

主指标：

| Run | TTT feature gate | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOVR_02 ATE | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `SWOVR_02` reference | none | `36.5915` | `6.4307` | `92.4416` | `0.0078` | `0.0000` | 当前 v5 tiny best |
| `ACL2V5_TTFG_01` | `kv_centered rho0.10 min0.85` | `36.7791` | `6.3020` | `92.4434` | `0.0077` | `+0.1876` | Rot/Final/Yaw 好，但 ATE 回退 |
| `ACL2V5_TTFG_02` | `kv_centered rho0.20 min0.70` | `36.8258` | `6.2185` | `92.4460` | `0.0077` | `+0.2343` | Rot/Final 最好，但 ATE 更差 |
| `ACL2V5_TTFG_03` | `v_centered rho0.10 min0.85` | `36.7822` | `6.3061` | `92.4375` | `0.0077` | `+0.1907` | 与 TTFG_01 接近，ATE 回退 |
| `ACL2V5_TTFG_04` | `k_centered rho0.10 min0.85` | `36.8715` | `6.3634` | `92.4450` | `0.0078` | `+0.2800` | 最差，K-only 不适合作主线 |

Trajectory diagnostics：

输出目录：

`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_replay_feature_target/trajectory_diagnostics_ttf_gate/`

| Run | ATE RMSE | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | Sim3 scale | Worst 100f |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` | `[200,300)=78.75` |
| `ACL2V5_TTFG_01` | `36.7791` | `3.445` | `30.036` | `30.668` | `30.819` | `3.412` | `31.249932` | `[200,300)=79.43` |
| `ACL2V5_TTFG_02` | `36.8258` | `2.056` | `30.068` | `30.690` | `30.903` | `3.244` | `31.278012` | `[200,300)=79.83` |
| `ACL2V5_TTFG_03` | `36.7822` | `4.192` | `30.035` | `30.673` | `30.776` | `3.481` | `31.173245` | `[200,300)=78.97` |
| `ACL2V5_TTFG_04` | `36.8715` | `4.133` | `30.058` | `30.695` | `30.812` | `3.479` | `31.264808` | `[200,300)=79.61` |

20.38 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02=36.5915`。
2. replay feature gate 的方向很明确：它能系统性改善 Rot/Yaw/FinalErr，尤其 `kv_centered rho0.20` 把 FinalErr 从 `5.118m` 降到 `2.056m`，Yaw 从 `3.593` 降到 `3.244`。
3. 但它会让 50/100/200 segment mean 和最坏 `[200,300)` 同步回退，因此仍然不是 ATE 主病灶的解法。
4. 这与前面 hard-token / explicit-dyn / anti-dynamic 的观察一致：**少写动态区域或收缩动态 residual 可以清姿态与 endpoint，但会牺牲全局 ATE/scale continuity**。
5. 后续不要继续围绕简单 `k/v/kv centered` 调 rho；下一步应尝试更结构化的 TTT 写入：不同 layer / branch 用不同策略，或直接改 replay 目标而不是全局 centroid shrink。

### 20.39 计划：reset 固定下的 TTT layer × branch policy

用户明确要求：reset 机制需要和 LoGeR 对齐，**不要变**。因此本批继续固定：

```text
RESET_EVERY = 5
```

本批只新增 TTT 写入策略自由度，不改 reset、不做 chunk-local reset、不改 LoGeR 原生 reset path。

新增实现：

- `TTT_WRITE_LAYER_BRANCH_POLICY` / `--prior_layer_branch_policy`
- 语法示例：`0-5:all;6-11:0;12-17:none`
- 含义：同一次 TTT replay 内，不同 TTT layer 可使用不同 prior branch mask；未命中的 layer 回退到 `PRIOR_BRANCH_MASK`。
- reset 相关代码未修改；`python3 -m py_compile loger/pipeline/ttt_write_controller.py loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py` 已通过。

动机：

1. 前面 all-branch EMA / hard token subset 会破坏 `w1/w2` 的 native 几何/尺度通道；
2. w0-only branch isolation 能保持 scale，但全层少写动态主要改善 Rot/Yaw/FinalErr，ATE 仍卡在 `36.6m` 平台；
3. 20.30 layer split 显示 late layer 更像姿态 regularizer，early/middle 保留 native continuity 可能更重要；
4. 因此需要测试 “early/middle/late layer 采用不同 branch 写入规则”，而不是继续全层同一 branch mask。

固定 active base：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
SWA overlap source replace = source/v last alpha0.25
success target = KITTI01 ATE < 30m
```

首批小矩阵：

| Run | Layer × branch policy | 额外策略 | 目的 |
|---|---|---|---|
| `ACL2V5_TTLB_01` | `0-5:all;6-11:0;12-17:0` | none | early native/semantic 全分支，middle/late 只控 w0，测试早层几何写入是否需要 full branch |
| `ACL2V5_TTLB_02` | `0-5:all;6-11:all;12-17:0` | none | 只把 late 限到 w0，保护 late w1/w2 |
| `ACL2V5_TTLB_03` | `0-5:all;6-11:0;12-17:none` | none | early 全分支 + middle w0，late 完全 native，测试 late dynamic prior 是否主要造成 ATE 回退 |
| `ACL2V5_TTLB_04` | `0-5:all;6-11:0;12-17:0` | `commit_ema_alpha=0.85, branch0` | 只对目标 branch0 做温和 EMA，避免 all-branch EMA 的尺度破坏 |

Gate：

- 若任一 run 达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若超过 `SWOVR_02 = 36.5915`，保留为新 active base；
- 若全部仍在 `36m+` 且只改善 Rot/Yaw/FinalErr，则判断普通 branch/layer 写入规则无法解决 ATE 主病灶，下一步转向更大胆的 TTT replay objective 或 per-head/per-layer dynamic reliability。

#### 20.39 实验结果：TTT layer × branch policy

运行记录：

- smoke `ACL2V5_TTLB_SMOKE_C23pairall_policy_earlyAll_midLateW0`：`02:53:32 -> 02:56:34`，约 `3.0 min`；
- smoke 确认 `prior_layer_branch_policy=0-5:all;6-11:0;12-17:0` 生效：
  - layer `0-5`: `prior_branch_mask=[0,1,2]`
  - layer `6-17`: `prior_branch_mask=[0]`
  - `ttt_prior_alignment_mode=direct_length`
- full batch：4 并发 GPU0-3，`02:57:59 -> 03:21:41`，约 `22.6-23.7 min` 完成 4 个 full run；
- `RESET_EVERY=5` 固定，reset 机制未改。

主指标：

| Run | Layer × branch policy | Extra | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOVR_02 ATE | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `SWOVR_02` reference | default branch0 | none | `36.5915` | `6.4307` | `92.4416` | `0.0078` | `0.0000` | 当前 v5 tiny best |
| `ACL2V5_TTLB_01` | `0-5:all;6-11:0;12-17:0` | none | `36.6597` | `6.4568` | `92.4414` | `0.0079` | `+0.0682` | 接近但未超过 |
| `ACL2V5_TTLB_02` | `0-5:all;6-11:all;12-17:0` | none | `36.6329` | `6.4202` | `92.4409` | `0.0079` | `+0.0414` | 本批 ATE 最接近，仍回退 |
| `ACL2V5_TTLB_03` | `0-5:all;6-11:0;12-17:none` | none | `36.7193` | `6.3803` | `92.4416` | `0.0078` | `+0.1278` | late native 改善 Rot/Final，但 ATE 回退 |
| `ACL2V5_TTLB_04` | `0-5:all;6-11:0;12-17:0` | branch0 commit EMA `0.85` | `37.0045` | `6.0668` | `92.4447` | `0.0075` | `+0.4130` | FinalErr/Yaw 强，但 ATE 明显差 |

Trajectory diagnostics：

输出目录：

`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_layer_branch_policy/trajectory_diagnostics_ttlb/`

| Run | ATE RMSE | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | Sim3 scale | Worst 100f |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` | `[200,300)=78.75` |
| `TTLB_01` | `36.6597` | `5.112` | `29.864` | `30.500` | `30.574` | `3.615` | `31.213784` | `[200,300)=78.95` |
| `TTLB_02` | `36.6329` | `5.137` | `29.847` | `30.484` | `30.552` | `3.581` | `31.207037` | `[200,300)=78.78` |
| `TTLB_03` | `36.7193` | `4.940` | `29.901` | `30.537` | `30.608` | `3.569` | `31.218233` | `[200,300)=79.05` |
| `TTLB_04` | `37.0045` | `1.108` | `30.517` | `31.123` | `31.373` | `3.195` | `31.254640` | `[200,300)=78.84` |

20.39 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02=36.5915`。
2. 真正的 layer × branch policy 没有产生突破；`TTLB_02` 只比 reference 差 `0.0414m`，说明保护 late `w1/w2` 是安全的，但不是 ATE 主病灶。
3. `TTLB_04` 的 branch0 EMA 把 FinalErr 从 `5.118m` 降到 `1.108m`、Yaw 从 `3.593` 降到 `3.195`，但全局 ATE 和 segment mean 全面回退，进一步说明写入侧很容易清 orientation/endpoint，却不修 `[200,300)` 主漂移。
4. `[200,300)` worst segment 几乎没动：`78.75 -> 78.78/78.95/79.05/78.84`。因此继续调同类 branch/layer mask 价值很低。
5. 下一步应转向更底层：SWA overlap 写入应显式使用 overlap 帧动态区域；TTT 侧则需要改变 replay objective / source target，而不是继续同一个 scalar D 只换 branch/layer。

### 20.40 计划：reset 固定下的 TTT 同帧静态 replay target

启动时间：`2026-05-07 03:27 +08`

用户最新约束：

```text
reset 机制不要变，需要和 LoGeR 对齐。
```

因此本批继续把 reset 视为固定系统条件：

```text
RESET_EVERY = 5
```

本批不修改 reset 代码、不新增 reset 变量、不使用 reset 结果做推进；只改 TTT replay 写入目标。

动机：

1. TTT 更新里 `v` 直接进入 `w1` 梯度，也通过 `dhidden` 影响 `w0/w2`；改变 replay 的 K/V target 比单纯缩放 lr/prior 更直接。
2. 20.38 的 `v_centered/kv_centered` 用的是整段 static-weight centroid，能改善 Rot/Yaw/FinalErr，但 ATE 回退，可能因为全局 centroid 抹掉了帧内几何结构。
3. 新增 `*_frame_static_center` 模式：对 patch token，使用同一帧内高静态 prior patch 的 centroid 作为 target，只把低静态 token 的 residual 向同帧静态背景收缩。
4. 这更贴近“动态区域不要污染 TTT 写入，但保留每帧静态几何”的假设。

新增实现：

- 扩展 `TTT_WRITE_REPLAY_FEATURE_GATE_MODE`：
  - `v_frame_static_center`
  - `k_frame_static_center`
  - `kv_frame_static_center` / `frame_static_center`
- 使用 `token_type == PATCH` 限定主要作用对象；
- centroid 在每帧内部计算，不跨帧混合；
- branch isolation 继续沿用 `TTT_WRITE_REPLAY_FEATURE_GATE_BRANCH_MASK`；
- reset path 未改，`python3 -m py_compile loger/pipeline/ttt_write_controller.py loger/pipeline/hybrid_memory_controller.py run_pipeline_abc_v2.py` 已通过。

固定 active base：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
RESET_EVERY = 5
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
SWA overlap source replace = source/v last alpha0.25
success target = KITTI01 ATE < 30m
```

首批小矩阵：

| Run | TTT frame-static target | Layer/branch | 目的 |
|---|---|---|---|
| `ACL2V5_TTFS_01` | `v_frame_static_center rho0.25 min0.50` | late / branch0 | 只改 value target，直接影响 TTT 更新方向 |
| `ACL2V5_TTFS_02` | `v_frame_static_center rho0.50 min0.25` | late / branch0 | 更强 value target replacement |
| `ACL2V5_TTFS_03` | `kv_frame_static_center rho0.25 min0.50` | late / branch0 | K/V 同时局部静态化，测试是否比全局 `kv_centered` 稳 |
| `ACL2V5_TTFS_04` | `v_frame_static_center rho0.25 min0.50` | all layers / branch0 | 判断只 late 是否太弱 |

Gate：

- 若任一 run 达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若超过 `SWOVR_02 = 36.5915`，保留为新 active base；
- 若只改善 Rot/Yaw/FinalErr 但不动 `[200,300)`，说明“动态区域少写”方向主要清姿态，仍需要寻找更准确的 dynamic cue 或更靠近 LoGeR TTT loss 的目标函数。

#### 20.40 实验结果：TTT 同帧静态 replay target

运行记录：

- smoke `ACL2V5_TTFS_SMOKE_C23pairall_lateW0_vFrameStatic_r025`：`03:29:12 -> 03:32:12`，约 `3.0 min`；
- smoke 确认 `v_frame_static_center` 生效：
  - `ttt_replay_feature_gate_frame_static=True`
  - `ttt_replay_feature_gate_targets=['v']`
  - `ttt_replay_feature_gate_tokens_per_frame=1260`
  - `ttt_replay_feature_gate_patch_tokens=40128`
  - active late layer 使用 `ttt_prior_alignment_mode=direct_length`
- full batch：4 并发 GPU0-3，`03:32:51 -> 03:57:49`，约 `22.8-25.0 min` 完成 4 个 full run；
- `RESET_EVERY=5` 固定，reset 机制未改。

主指标：

| Run | TTT frame-static target | Layer/branch | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOVR_02 ATE | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `SWOVR_02` reference | none | branch0 all | `36.5915` | `6.4307` | `92.4416` | `0.0078` | `0.0000` | 当前 v5 tiny best |
| `ACL2V5_TTFS_01` | `v_frame_static_center rho0.25 min0.50` | late / branch0 | `36.6757` | `6.2567` | `92.4367` | `0.0077` | `+0.0842` | Rot/Final/Yaw 好，但 ATE 回退 |
| `ACL2V5_TTFS_02` | `v_frame_static_center rho0.50 min0.25` | late / branch0 | `36.7503` | `6.1103` | `92.4342` | `0.0076` | `+0.1588` | 更强 gate 继续清姿态，但 ATE 更差 |
| `ACL2V5_TTFS_03` | `kv_frame_static_center rho0.25 min0.50` | late / branch0 | `36.7507` | `6.1477` | `92.4420` | `0.0076` | `+0.1592` | FinalErr 最好，但 ATE 回退 |
| `ACL2V5_TTFS_04` | `v_frame_static_center rho0.25 min0.50` | all / branch0 | `36.6914` | `6.2781` | `92.4370` | `0.0077` | `+0.0999` | all-layer 不如 reference |

Trajectory diagnostics：

输出目录：

`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_frame_static_target/trajectory_diagnostics_ttfs/`

| Run | ATE RMSE | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | Sim3 scale | Worst 100f |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` | `[200,300)=78.75` |
| `TTFS_01` | `36.6757` | `4.150` | `29.891` | `30.523` | `30.619` | `3.451` | `31.167388` | `[200,300)=79.00` |
| `TTFS_02` | `36.7503` | `3.392` | `29.966` | `30.593` | `30.713` | `3.297` | `31.141771` | `[200,300)=79.27` |
| `TTFS_03` | `36.7507` | `2.775` | `30.083` | `30.703` | `30.853` | `3.273` | `31.224814` | `[200,300)=79.00` |
| `TTFS_04` | `36.6914` | `3.928` | `29.918` | `30.549` | `30.650` | `3.465` | `31.172712` | `[200,300)=79.03` |

20.40 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02=36.5915`。
2. 同帧静态 centroid 比全局 centroid 更符合动态区域假设，但结果模式仍一致：Rot/Yaw/FinalErr 系统性改善，ATE 和 segment mean 回退。
3. `TTFS_03` 把 FinalErr 从 `5.118m` 降到 `2.775m`，Yaw 从 `3.593` 降到 `3.273`，但 100f mean 从 `30.443` 升到 `30.703`，说明该类 TTT replay target 更像姿态/endpoint regularizer，不修全局 ATE 主漂移。
4. `[200,300)` worst segment 仍在 `79m` 左右，甚至略差于 reference；继续在 TTT replay K/V centroid target 上调 rho / min gate 价值不高。
5. 至此，“少写动态区域”这个方向对 TTT 写入的主要收益已经比较清楚：能清 orientation，但没有让 KITTI01 ATE 突破。下一步需要转到更像 LoGeR TTT 更新机制本身的策略，例如只改变 `w1` value-memory 分支、或者直接让 write score 学会保护 `[200,300)` 前的静态尺度连续性；SWA 侧也应继续聚焦 overlap-source 的真实动态/静态对齐，而不是普通 whole-chunk gate。

### 20.41 计划：reset 固定下的 TTT `w1` value-memory replay target

启动时间：`2026-05-07 04:02 +08`

用户最新约束再次确认：

```text
reset 机制不要变，需要和 LoGeR 对齐。
```

因此本批继续固定：

```text
RESET_EVERY = 5
```

本批不修改 reset 代码、不新增 reset 变量、不使用 reset schedule 做推进。只改变 TTT replay 写入目标和 branch isolation。

动机：

1. LoGeR TTT update 中 `w1_grad = zeropower((hidden * lr1)^T @ v)`，`v` 是直接进入 value/output memory 分支的目标。
2. 20.40 的同帧静态 target 只作用在 branch0，能改善 Rot/Yaw/FinalErr，但 ATE 和 `[200,300)` 不动。
3. 如果动态区域污染主要发生在 value-memory 写入，而不是 gate branch0，那么应该用 branch-isolated feature gate 只替换 `w1` replay 的 `v` target，保留 `w0/w2` 原生几何/尺度通道。
4. 早期 all-branch / hard-tail `w1` 实验会破坏尺度，因此本批只做温和到中等的 `w1` value target，不做 hard drop，不动 reset。

固定 active base：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
RESET_EVERY = 5
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
SWA overlap source replace = source/v last alpha0.25
success target = KITTI01 ATE < 30m
```

首批小矩阵：

| Run | TTT feature target | Branch | Layer | 目的 |
|---|---|---|---|---|
| `ACL2V5_TTVM_01` | `v_frame_static_center rho0.10 min0.75` | `w1` only | late | 温和 value-memory 静态化，避免尺度通道被硬破坏 |
| `ACL2V5_TTVM_02` | `v_frame_static_center rho0.25 min0.50` | `w1` only | late | 中等强度，和 20.40 branch0 对齐比较 |
| `ACL2V5_TTVM_03` | `v_centered rho0.10 min0.75` | `w1` only | late | 对照同帧 centroid vs 全局 centroid |
| `ACL2V5_TTVM_04` | `v_frame_static_center rho0.10 min0.75` | `w0+w1` | late | 判断 branch0 gate 与 branch1 value 是否需要协同 |

Gate：

- 若任一 run 达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若超过 `SWOVR_02 = 36.5915`，保留为新 active base；
- 若只改善 Rot/Yaw/FinalErr 但 `[200,300)` 仍不动，则说明 TTT value-memory target 也不是主漂移来源，下一步优先转向 SWA overlap-source 的结构性 source selection / per-head alignment。

#### 20.41 实验结果：TTT `w1` value-memory replay target

运行记录：

- smoke `ACL2V5_TTVM_SMOKE_C23pairall_lateW1_vFrameStatic_r010`：`04:03:13 -> 04:06:14`，约 `3.0 min`；
- smoke / full config 均确认 `reset_every: 5`，reset 机制未改；
- full batch：4 并发 GPU0-3，`04:08:06 -> 04:32:51`，约 `23.3-24.8 min` 完成 4 个 full run；
- 输出目录：
  `results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_w1_value_memory/`

主指标：

| Run | TTT feature target | Branch | Layer | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOVR_02 ATE | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `SWOVR_02` reference | none | branch0 | all | `36.5915` | `6.4307` | `92.4416` | `0.0078` | `0.0000` | 当前 v5 tiny best |
| `ACL2V5_TTVM_01` | `v_frame_static_center rho0.10 min0.75` | `w1` only | late | `36.7089` | `6.2121` | `92.4392` | `0.0076` | `+0.1174` | Rot/Final/Yaw 好，ATE 回退 |
| `ACL2V5_TTVM_02` | `v_frame_static_center rho0.25 min0.50` | `w1` only | late | `36.8283` | `5.8820` | `92.4404` | `0.0075` | `+0.2368` | rotation 最强，但 ATE 更差 |
| `ACL2V5_TTVM_03` | `v_centered rho0.10 min0.75` | `w1` only | late | `36.6946` | `6.2544` | `92.4443` | `0.0076` | `+0.1031` | 本批 ATE 最接近，但未超过 |
| `ACL2V5_TTVM_04` | `v_frame_static_center rho0.10 min0.75` | `w0+w1` | late | `36.7426` | `6.1502` | `92.4389` | `0.0076` | `+0.1511` | branch0+w1 协同未带来 ATE gain |

Trajectory diagnostics：

输出目录：

`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/ttt_w1_value_memory/trajectory_diagnostics_ttvm/`

| Run | ATE RMSE | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | Sim3 scale | Worst 100f |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` | `[200,300)=78.75` |
| `TTVM_01` | `36.7089` | `2.733` | `30.046` | `30.674` | `30.831` | `3.363` | `31.194508` | `[200,300)=78.68` |
| `TTVM_02` | `36.8283` | `3.078` | `30.408` | `31.012` | `31.374` | `2.995` | `31.211223` | `[200,300)=78.57` |
| `TTVM_03` | `36.6946` | `2.635` | `30.076` | `30.698` | `30.857` | `3.405` | `31.240001` | `[200,300)=78.51` |
| `TTVM_04` | `36.7426` | `2.539` | `30.079` | `30.701` | `30.859` | `3.311` | `31.190309` | `[200,300)=78.90` |

20.41 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02=36.5915`。
2. 只改 `w1` value-memory target 的方向非常明确：Rot/Yaw/FinalErr 系统性改善，尤其 `TTVM_02` Rot 降到 `5.8820`、Yaw 降到 `2.995`；但全局 ATE 和 segment mean 全部回退。
3. `[200,300)` worst 100f 从 `78.75` 最多只到 `78.51`，幅度太小，不足以解释 `<30` 目标所需的大幅 ATE 降低。
4. `w1` 不是无效分支，它能管 orientation/value memory；但它仍不是 KITTI01 ATE 主病灶。继续在 `v_centered/v_frame_static_center` 上加大强度价值不高。
5. reset 固定后，TTT 写入侧目前最稳定的认识是：少写动态区域 / 静态 target / branch-layer policy 都更像姿态正则器；要突破 ATE，下一步优先转向 SWA overlap-source 的真实 overlap 动态/静态对齐，以及更结构化的 source selection。

### 20.42 计划：reset 固定下的 SWA current-overlap source replacement

启动时间：`2026-05-07 04:39 +08`

用户提出的关键假设：

```text
SWA 可能要关注 overlap 帧的动态区域，而不是整个 chunk 的动态区域。
```

现有 best `SWOVR_02` 使用：

```text
SWA_OVERLAP_SOURCE_REPLACE_MODE = source
SWA_OVERLAP_SOURCE_REPLACE_TARGET = v
SWA_OVERLAP_SOURCE_REPLACE_ALPHA = 0.25
```

也就是用 previous tail-overlap 的 `D_prev` 决定替换强度。本批改成 current head-overlap 或 current/source mismatch，验证真正 overlap 对齐区域是否更重要。

固定 active base：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
RESET_EVERY = 5
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
success target = KITTI01 ATE < 30m
```

小矩阵：

| Run | SWA overlap source replace | Target | 目的 |
|---|---|---|---|
| `ACL2V5_SWOC_01` | `mode=current alpha0.25` | `v` | 用 current head-overlap D 控制 previous source V 替换 |
| `ACL2V5_SWOC_02` | `mode=current alpha0.50` | `v` | 更强 current-overlap replacement |
| `ACL2V5_SWOC_03` | `mode=current alpha0.25` | `kv` | K/V 同时对齐 overlap source |
| `ACL2V5_SWOC_04` | `mode=disagreement alpha0.50` | `v` | 只在 current/source overlap D 不一致时替换 |

Gate：

- 若任一 run 达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若超过 `SWOVR_02 = 36.5915`，保留为新 active base；
- 若只是改善 Rot/FinalErr 或微调 `[200,300)`，继续寻找更结构化的 source selection，而不是加大普通 alpha。

#### 20.42 实验结果：SWA current-overlap source replacement

运行记录：

- full batch：4 并发 GPU0-3，`04:36:19 -> 05:00:11`，约 `23.9 min` 完成 4 个 full run；
- `RESET_EVERY=5` 固定，reset 机制未改；
- 输出目录：
  `results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/swa_overlap_current_replace/`

主指标：

| Run | SWA overlap replace | Target | ATE RMSE | Rot RMSE | RPE t | RPE r | vs SWOVR_02 ATE | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `SWOVR_02` reference | `mode=source alpha0.25` | `v` | `36.5915` | `6.4307` | `92.4416` | `0.0078` | `0.0000` | 当前 v5 tiny best |
| `ACL2V5_SWOC_01` | `mode=current alpha0.25` | `v` | `36.6693` | `6.4179` | `92.4410` | `0.0078` | `+0.0778` | ATE 回退 |
| `ACL2V5_SWOC_02` | `mode=current alpha0.50` | `v` | `36.6683` | `6.4213` | `92.4403` | `0.0078` | `+0.0768` | 与 0.25 几乎等价 |
| `ACL2V5_SWOC_03` | `mode=current alpha0.25` | `kv` | `36.6705` | `6.4218` | `92.4405` | `0.0078` | `+0.0790` | K/V 同改仍回退 |
| `ACL2V5_SWOC_04` | `mode=disagreement alpha0.50` | `v` | `36.6651` | `6.4193` | `92.4402` | `0.0078` | `+0.0736` | 本批最接近，但未超过 |

SWA replace debug：

| Run | Applied chunks | Mean alpha | Mean score | 说明 |
|---|---:|---:|---:|---|
| `SWOC_01` | `37/38` | `0.035963` | `0.143851` | current overlap gate 实际生效 |
| `SWOC_02` | `37/38` | `0.071925` | `0.143851` | alpha 加倍但指标几乎不变 |
| `SWOC_03` | `37/38` | `0.035963` | `0.143851` | K/V target 实际生效 |
| `SWOC_04` | `37/38` | `0.071169` | `0.142341` | disagreement gate 实际生效 |

Trajectory diagnostics：

输出目录：

`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/swa_overlap_current_replace/trajectory_diagnostics_swoc/`

| Run | ATE RMSE | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | Sim3 scale | Worst 100f |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` | `[200,300)=78.75` |
| `SWOC_01` | `36.6693` | `5.247` | `29.896` | `30.537` | `30.580` | `3.594` | `31.202579` | `[200,300)=78.70` |
| `SWOC_02` | `36.6683` | `5.250` | `29.896` | `30.536` | `30.580` | `3.594` | `31.197780` | `[200,300)=78.70` |
| `SWOC_03` | `36.6705` | `5.208` | `29.903` | `30.543` | `30.587` | `3.593` | `31.202861` | `[200,300)=78.70` |
| `SWOC_04` | `36.6651` | `5.221` | `29.895` | `30.535` | `30.581` | `3.592` | `31.197175` | `[200,300)=78.70` |

20.42 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `SWOVR_02=36.5915`。
2. current-overlap / disagreement gate 确实进入 SWA path，37 个非首 chunk 都有非零 replacement，因此不是 hook 没生效。
3. 只把 source overlap 的 K/V 做线性 replacement 太弱：worst `[200,300)` 从 `78.75` 只到约 `78.70`，但全局 ATE 和 segment mean 反而回退。
4. 用户关于 overlap 区域的判断仍然值得继续：普通 alpha blend 不够，下一步要做更结构化的 SWA source selection，例如只保留 tail-overlap source、只保留 both-overlap source、或者直接丢弃 tail-overlap source 来验证 source memory 的污染位置。
5. reset 仍固定为 `5`；后续不再把 reset 作为实验变量。

### 20.43 计划：reset 固定下的 SWA overlap source 结构性 selection

启动时间：`2026-05-07 05:08 +08`

本批不再只是对 overlap source 做小幅线性替换，而是直接改变写入到 SWA history 的 source token 范围。目标是验证：

```text
SWA 的关键污染是否来自非 overlap source history，或来自 tail-overlap source 本身。
```

固定 active base：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
RESET_EVERY = 5
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA write = kv_centered tail_overlap last rho0.10 min0.85 score=read
success target = KITTI01 ATE < 30m
```

小矩阵：

| Run | `SWA_WRITE_KEEP_SCOPE` | Extra source replace | 目的 |
|---|---|---|---|
| `ACL2V5_SWKS_01` | `tail_overlap` | none | 只把当前 chunk 的 tail-overlap 写进 SWA source，测试非 overlap 历史是否污染 |
| `ACL2V5_SWKS_02` | `both_overlap` | none | 保留 head+tail overlap，测试 seam-only source 是否更稳 |
| `ACL2V5_SWKS_03` | `exclude_tail_overlap` | none | 反向测试：如果 tail-overlap 本身有毒，丢弃它应改善 |
| `ACL2V5_SWKS_04` | `tail_overlap` | `current/v alpha0.25` | 结构性 tail-only source 加 current overlap replacement |

Gate：

- 若任一 run 达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若超过 `SWOVR_02 = 36.5915`，保留为新 active base；
- 若 tail-only / both-only 大幅变差，说明 SWA 仍需要完整 source context，后续应回到 TTT update / write score；
- 若 exclude-tail 明显变好，说明 overlap tail source 自身是污染点，下一步围绕 tail-overlap hard filtering / high-risk token drop 展开。

#### 20.43 实验结果：SWA overlap source 结构性 selection

运行记录：

- smoke：`ACL2V5_SWKS_SMOKE_C23pairall_keepTailOverlap` 通过；debug 显示 last SWA layer history 从 `40320` tokens 缩到 `3954` tokens，`keep_mass≈0.09375`，说明 `SWA_WRITE_KEEP_SCOPE` 确实进入 source cache 写入路径；
- full batch：4 并发 GPU0-3，`05:07:48 -> 05:31:21`，约 `23.6 min` 完成 4 个 full run；
- `RESET_EVERY=5` 固定，reset 机制未改；
- 输出目录：
  `results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/swa_keep_scope_structural/`

主指标：

| Run | `SWA_WRITE_KEEP_SCOPE` | Extra replace | ATE RMSE | Rot RMSE | RPE t | RPE r | vs `SWOVR_02` ATE | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `SWOVR_02` reference | full source + source-V replace | `source/v alpha0.25` | `36.5915` | `6.4307` | `92.4416` | `0.0078` | `0.0000` | previous tiny best |
| `ACL2V5_SWKS_01` | `tail_overlap` | none | `36.4796` | `6.7810` | `92.4676` | `0.0083` | `-0.1119` | ATE 明显改善，但 Rot 回退 |
| `ACL2V5_SWKS_02` | `both_overlap` | none | `36.4494` | `6.5917` | `92.4524` | `0.0081` | `-0.1421` | **new v5 ATE best** |
| `ACL2V5_SWKS_03` | `exclude_tail_overlap` | none | `36.5012` | `6.2434` | `92.4337` | `0.0077` | `-0.0903` | ATE 改善，Rot/Final/Yaw 最好 |
| `ACL2V5_SWKS_04` | `tail_overlap` | `current/v alpha0.25` | `36.4747` | `6.7847` | `92.4664` | `0.0083` | `-0.1169` | 接近 tail-only；current-V replace 没有额外收益 |

Trajectory diagnostics：

输出目录：

`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/swa_keep_scope_structural/trajectory_diagnostics_swks/`

| Run | ATE RMSE | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | Sim3 scale | Worst 100f |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `3.593` | `31.213845` | `[200,300)=78.75` |
| `TTEX_01` | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` | `[200,300)=78.75` |
| `SWKS_01` | `36.4796` | `5.704` | `29.739` | `30.396` | `30.517` | `3.886` | `31.483346` | `[200,300)=78.59` |
| `SWKS_02` | `36.4494` | `5.640` | `29.752` | `30.404` | `30.447` | `3.744` | `31.306871` | `[200,300)=77.88` |
| `SWKS_03` | `36.5012` | `4.390` | `29.744` | `30.361` | `30.437` | `3.427` | `31.134881` | `[200,300)=78.54` |
| `SWKS_04` | `36.4747` | `5.703` | `29.736` | `30.391` | `30.513` | `3.885` | `31.473897` | `[200,300)=78.59` |

20.43 结论：

1. 本批仍没有达到 `KITTI01 ATE < 30m`，但第一次明确超过 `SWOVR_02/TTEX_01` 平台；`SWKS_02` 把 ATE 推到 `36.4494m`，相对 `SWOVR_02` 改善 `0.1421m`。
2. 结构性 source selection 比普通 overlap alpha blend 更有效。20.42 的 current/source replacement 只在 `36.66m` 附近回退，而本批直接改 source cache 范围后全部优于 `SWOVR_02`。
3. `both_overlap` 是当前 ATE 最好的 source scope，说明 SWA 写入不应该保留完整非 overlap history；只保留 seam 附近 source 反而减少污染。
4. `exclude_tail_overlap` 虽然 ATE 不如 `both_overlap`，但 Rot=`6.2434`、FinalErr=`4.390`、Yaw=`3.427` 最好，说明 tail-overlap 不是纯粹“好 source”；它带 ATE 信息，也带 orientation/endpoint 污染。
5. `tail_overlap + current-V replace` 没有超过 plain `tail_overlap`，说明当前 overlap replacement 的 alpha blend 不是关键；更关键的是 source cache 的结构性保留范围。
6. 新 active base 更新为 `SWKS_02 = both_overlap keep-scope + C23 pair/all + TTEX w0=1.10 + stage_d_x_dg_inv_sqrt`。下一步围绕它做 repeat 和局部组合，不再把 `SWOVR_02` 当 active best。

### 20.44 计划：围绕 `SWKS_02` 的 repeat 与局部 source-replace 组合

启动时间：`2026-05-07 05:36 +08`

当前新 best：

```text
SWKS_02 = C23 pair/all + beta 4.75 + stage_d_x_dg_inv_sqrt
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
SWA_WRITE_KEEP_SCOPE = both_overlap
RESET_EVERY = 5
ATE/Rot = 36.4494 / 6.5917
```

本批目的：

1. 先做 exact repeat，确认 `SWKS_02` 不是单次波动；
2. 在 `both_overlap` source scope 上补最小 source replacement 组合，确认是否能同时拿到 `SWKS_02` 的 ATE 和 `SWKS_03` 的 Rot/Final/Yaw。

小矩阵：

| Run | Keep scope | Source replace | Target | 目的 |
|---|---|---|---|---|
| `ACL2V5_SWKS2_01` | `both_overlap` | none | none | exact repeat of `SWKS_02` |
| `ACL2V5_SWKS2_02` | `both_overlap` | `source alpha0.25` | `v` | 把 previous overlap D 的 source-V replacement 加到 both-overlap scope |
| `ACL2V5_SWKS2_03` | `both_overlap` | `current alpha0.25` | `v` | 用 current overlap D 控制 replacement |
| `ACL2V5_SWKS2_04` | `both_overlap` | `source alpha0.25` | `kv` | 测试 both-overlap 下 K/V 同改是否能进一步修正 source cache |

Gate：

- 若任一 run 达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若 repeat 稳定，`SWKS_02` 升为 v5 active base；
- 若 replacement 组合超过 `36.4494`，保留为新 active base；
- 若 replacement 全部回退，说明 SWA source selection 的主贡献来自结构性 token scope，而不是 K/V alpha blend，后续转向按 chunk-risk/overlap-risk 自适应选择 `both_overlap` vs `exclude_tail_overlap`。

#### 20.44 实验结果：`SWKS_02` repeat 与 both-overlap replacement

运行记录：

- full batch：4 并发 GPU0-3，`05:33:51 -> 05:57:56`，约 `24.1 min` 完成 4 个 full run；
- `RESET_EVERY=5` 固定，reset 机制未改；
- 输出目录：
  `results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/swa_keep_scope_refine/`

主指标：

| Run | Keep scope | Source replace | Target | ATE RMSE | Rot RMSE | RPE t | RPE r | vs `SWKS_02` ATE | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `SWKS_02` reference | `both_overlap` | none | none | `36.4494` | `6.5917` | `92.4524` | `0.0081` | `0.0000` | 20.43 best |
| `ACL2V5_SWKS2_01` | `both_overlap` | none | none | `36.4494` | `6.5917` | `92.4524` | `0.0081` | `0.0000` | byte-level exact repeat |
| `ACL2V5_SWKS2_02` | `both_overlap` | `source alpha0.25` | `v` | `36.4404` | `6.5908` | `92.4503` | `0.0081` | `-0.0090` | 小幅改善 |
| `ACL2V5_SWKS2_03` | `both_overlap` | `current alpha0.25` | `v` | `36.4481` | `6.5937` | `92.4511` | `0.0081` | `-0.0013` | 基本等价 |
| `ACL2V5_SWKS2_04` | `both_overlap` | `source alpha0.25` | `kv` | `36.4276` | `6.6081` | `92.4516` | `0.0082` | `-0.0218` | **new v5 ATE best** |

Trajectory diagnostics：

输出目录：

`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/swa_keep_scope_refine/trajectory_diagnostics_swks2/`

| Run | ATE RMSE | FinalErr | 50f worst | 100f worst | 200f worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SWOVR_02` | `36.5915` | `5.118` | `78.81` | `78.75` | `57.75` | `3.593` | `31.213845` |
| `SWKS_02` | `36.4494` | `5.640` | `79.08` | `77.88` | `57.12` | `3.744` | `31.306871` |
| `SWKS2_01` | `36.4494` | `5.640` | `79.08` | `77.88` | `57.12` | `3.744` | `31.306871` |
| `SWKS2_02` | `36.4404` | `5.654` | `79.04` | `77.89` | `57.12` | `3.738` | `31.294842` |
| `SWKS2_03` | `36.4481` | `5.656` | `79.05` | `77.89` | `57.12` | `3.744` | `31.297874` |
| `SWKS2_04` | `36.4276` | `5.682` | `79.03` | `77.86` | `57.09` | `3.756` | `31.300070` |

20.44 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，但继续刷新 v5 ATE best 到 `36.4276m`。
2. `SWKS_02` exact repeat 完全一致，说明 `both_overlap` keep-scope 的收益不是单次波动。
3. 在 `both_overlap` source scope 上，`source/v alpha0.25` 只有很小收益，`current/v alpha0.25` 基本无效；`source/kv alpha0.25` 才带来新 best，说明 source-side overlap 的 K/V 一起对齐更接近有效机制。
4. 新 best 的主要收益仍来自 `[200,300)` 和 `[200,400)` worst segment 的轻微降低：100f worst 从 `78.75` 到 `77.86`，200f worst 从 `57.75` 到 `57.09`；这是真改善，但距离 `<30` 需要的量级还很远。
5. 代价是 Rot/FinalErr/Yaw 比 `SWOVR_02` 和 `SWKS_02` 略差；这条更像 ATE-oriented active base，不是 balanced best。
6. 下一步需要对 `source/kv` 做 alpha 曲线和 repeat：如果 `alpha0.25` 稳定且 `0.50` 继续降低 ATE，可继续围绕 SWA source-KV correction；如果 alpha 曲线平台化，则说明 source scope 已经吃到主要收益，需要转向 chunk-risk 自适应 source scope。

### 20.45 计划：`both_overlap + source/KV replacement` alpha 曲线

启动时间：`2026-05-07 06:02 +08`

当前 ATE best：

```text
SWKS2_04 = both_overlap keep-scope + source/kv alpha0.25
ATE/Rot = 36.4276 / 6.6081
```

本批只做最小 alpha 曲线和 repeat，不引入新 reset 或大矩阵。

小矩阵：

| Run | Keep scope | Source replace | Target | 目的 |
|---|---|---|---|---|
| `ACL2V5_SWKS3_01` | `both_overlap` | `source alpha0.25` | `kv` | repeat `SWKS2_04` |
| `ACL2V5_SWKS3_02` | `both_overlap` | `source alpha0.10` | `kv` | 测试更小 K/V correction 是否减少 Rot/Final 代价 |
| `ACL2V5_SWKS3_03` | `both_overlap` | `source alpha0.50` | `kv` | 测试更强 K/V correction 是否继续压 ATE |
| `ACL2V5_SWKS3_04` | `both_overlap` | `source alpha0.25` | `k` | 分离 K-only 贡献，判断 ATE 改善是否主要来自 source key 对齐 |

Gate：

- 若任一 run 达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若 `alpha0.25` repeat 稳定，`SWKS2_04/SWKS3_01` 升为 active ATE base；
- 若 `alpha0.50` 更好，继续做 `0.75/1.00`；
- 若 `k-only` 接近 `kv`，说明 source K 是关键；若明显回退，说明 V correction 仍必要。

#### 20.45 实验结果：`both_overlap + source/KV replacement` alpha 曲线

运行记录：

- full batch：4 并发 GPU0-3，`05:59:35 -> 06:24:05`，约 `24.5 min` 完成 4 个 full run；
- `RESET_EVERY=5` 固定，reset 机制未改；
- 输出目录：
  `results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/swa_keep_scope_alpha/`

主指标：

| Run | Keep scope | Source replace | Target | ATE RMSE | Rot RMSE | RPE t | RPE r | vs `SWKS2_04` ATE | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `SWKS2_04` reference | `both_overlap` | `source alpha0.25` | `kv` | `36.4276` | `6.6081` | `92.4516` | `0.0082` | `0.0000` | previous v5 ATE best |
| `ACL2V5_SWKS3_01` | `both_overlap` | `source alpha0.25` | `kv` | `36.4276` | `6.6081` | `92.4516` | `0.0082` | `0.0000` | byte-level exact repeat |
| `ACL2V5_SWKS3_02` | `both_overlap` | `source alpha0.10` | `kv` | `36.4385` | `6.6007` | `92.4516` | `0.0082` | `+0.0109` | 小 alpha 略回退，Rot 略好 |
| `ACL2V5_SWKS3_03` | `both_overlap` | `source alpha0.50` | `kv` | `36.4153` | `6.6186` | `92.4509` | `0.0082` | `-0.0123` | **new v5 ATE best** |
| `ACL2V5_SWKS3_04` | `both_overlap` | `source alpha0.25` | `k` | `36.4313` | `6.6095` | `92.4518` | `0.0082` | `+0.0037` | K-only 接近但不如 K/V，V correction 仍有小贡献 |

Trajectory diagnostics：

输出目录：

`results/kitti01_hmc_v2/acl2_v5_dglocked_perhead_ttt_swa_accel/swa_keep_scope_alpha/trajectory_diagnostics_swks3/`

| Run | ATE RMSE | FinalErr | 50f mean | 100f mean | 200f mean | 100f worst | 200f worst | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SWOVR_02` | `36.5915` | `5.118` | `29.808` | `30.443` | `30.519` | `78.75` | `57.75` | `3.593` | `31.213845` |
| `SWKS2_04` | `36.4276` | `5.682` | `29.737` | `30.387` | `30.436` | `77.86` | `57.09` | `3.756` | `31.300070` |
| `SWKS3_01` | `36.4276` | `5.682` | `29.737` | `30.387` | `30.436` | `77.86` | `57.09` | `3.756` | `31.300070` |
| `SWKS3_02` | `36.4385` | `5.667` | `29.743` | `30.395` | `30.440` | `77.87` | `57.11` | `3.750` | `31.302870` |
| `SWKS3_03` | `36.4153` | `5.687` | `29.734` | `30.383` | `30.436` | `77.83` | `57.06` | `3.765` | `31.298748` |
| `SWKS3_04` | `36.4313` | `5.696` | `29.738` | `30.389` | `30.435` | `77.86` | `57.09` | `3.760` | `31.306271` |

20.45 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，但把 v5 ATE best 小幅刷新到 `36.4153m`。
2. `SWKS2_04` exact repeat 完全一致，确认 `both_overlap + source/KV alpha0.25` 稳定。
3. `alpha0.50` 比 `0.25` 再低 `0.0123m`，说明 source-side overlap K/V correction 还有一点单调收益；但收益已经是厘米级，不可能单靠这条达到 `<30m`。
4. `K-only alpha0.25` 接近 K/V 但略弱，说明 source key 对齐是主要贡献，value correction 仍提供很小的附加收益。
5. 代价继续集中在 Rot / FinalErr / Yaw：`SWKS3_03` 的 ATE best 同时有 `FinalErr=5.687`、`Yaw=3.765`，明显不如 `SWOVR_02`。
6. 按用户最新判断，本轮结束后 **SWA 先停止**。SWA source selection 已证明有真实信号，但收益平台很浅；后续最高优先级转为 TTT 写入/提交策略。

### 20.46 计划：TTT commit propagation filter（当前 chunk 保留动态信息，跨 chunk 传播时遗忘）

用户提出的新判断：

```text
动态区域可能也为 TTT 提供当前 chunk 内必要信息；
直接压制这些区域，会让 TTT 在当前 chunk 内无法提供必要适应；
但这些动态区域不一定应该长期传到下一个 chunk；
所以关键不是简单“少写动态区域”，而是怎么在 commit / 传播阶段忘记动态污染。
```

对照当前实现：

- `probe_ttt_write` 的受控输出来自 pass2；commit state 由 native probe cache 重新 replay 生成；
- 现有 `stage_d_x_dg_inv_sqrt` / sparse / feature gate / token scope 等策略，大多在 replay 时改变写入 prior 或 replay K/V；
- 这仍然是在“生成下一状态的 replay objective”里改 token，而不是先允许 native/full TTT 更新吸收当前 chunk 信息，再对传给下一 chunk 的状态做动态风险过滤。

因此新增一个更直接的 TTT 提交态过滤方向，暂命名：

```text
TTCPF = TTT Commit Propagation Filter
```

含义：不改 reset，不改当前 chunk pass2 的 TTT 使用；只在 `probe_ttt_write` 生成 `state_next.ttt_state` 时，对最终提交的 fast weights 做风险自适应 post-filter。

第一批只做两类机制：

| 机制 | 直觉 | 具体形式 |
|---|---|---|
| `native_to_candidate_by_risk` | 当前 chunk native TTT 可用；只有 tail-overlap 动态风险高时，才把 native commit 往 semantic/static candidate 拉 | `W_commit = W_native + s(D_tail) * (W_candidate - W_native)` |
| `old_decay_by_risk` | 高风险 chunk 的 fast-weight 生命周期应更短；动态风险高时，把 commit 往上一 chunk `W_old` 缩 | `W_commit = W_old + a(D_tail) * (W_candidate - W_old)` |

第一批 TTT 小矩阵（SWA 停止）：

| Run | 机制 | Scope / stat | Branch | 目的 |
|---|---|---|---|---|
| `ACL2V5_TTCPF_01` | `native_to_candidate_by_risk` | `tail_overlap / mean` | `w0` | 高 tail 动态风险时更强动态遗忘，低风险保留 native 连续性 |
| `ACL2V5_TTCPF_02` | `native_to_candidate_by_risk` | `tail_overlap / q90` | `w0` | 用高风险尾部 token 触发更强遗忘 |
| `ACL2V5_TTCPF_03` | `old_decay_by_risk` | `tail_overlap / mean` | `w0` | 高风险时缩短本 chunk semantic fast-weight 寿命 |
| `ACL2V5_TTCPF_04` | `native_to_candidate_by_risk` | `both_overlap / mean` | `w0` | 对比只看 tail 是否太窄，验证 overlap 双端传播风险 |

固定协议：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
RESET_EVERY = 5
SWA write / source replace = off
success target = KITTI01 ATE < 30m
```

Gate：

- 若任一 run 达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若超过 TTT-only / SWA-active best，保留为 active TTT write base；
- 若 ATE 不动但 `[200,300)` worst 明显下降，继续围绕风险函数做 chunk-aware refinement；
- 若仍只改善 Rot/Final/Yaw，不继续浅层扫同类参数，转向更靠近 TTT loss/fast-weight state decomposition 的写入目标。

#### 20.46 实验结果：TTT commit propagation filter

运行记录：

- smoke `ACL2V5_TTCPF_SMOKE_C23pairall_tailmean_native2cand` 通过；debug 显示 `ttt_write_commit_filter_applied=True`、`risk_source=d_tok`、`scope=tail_overlap`、`stat=mean`、`branch_mask=[0]`，并且 `RESET_EVERY=5`、`swa_write_enabled=False`。
- full batch：4 并发，GPU 0-3，`2026-05-07 06:34:36 -> 06:59:46`，约 `25.2 min` 完成 4 个 KITTI01 full run。
- 固定协议同上，SWA 全部关闭；本批只改变 TTT 提交态传播过滤。

Global metrics：

| Run | Commit filter | Scope/stat | Branch | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---|---:|---:|---:|---:|---|
| `SWKS3_03` reference | SWA-active best | n/a | n/a | `36.4153` | `6.6186` | `92.4495` | `0.0077` | 当前 v5 ATE best，但 SWA 线已暂停 |
| `TTEX_01` reference | TTT-only native mix | n/a | n/a | `36.5932` | `6.4327` | `92.4390` | `0.0078` | 当前 TTT-only reference |
| `ACL2V5_TTCPF_01` | `native_to_candidate_by_risk` | tail / mean | `w0` | `36.7629` | `6.4050` | `92.4377` | `0.0078` | ATE 回退，Rot 略好 |
| `ACL2V5_TTCPF_02` | `native_to_candidate_by_risk` | tail / q90 | `w0` | `36.6308` | `6.4140` | `92.4375` | `0.0078` | 本批 ATE 最好，但仍弱于 `TTEX_01` |
| `ACL2V5_TTCPF_03` | `old_decay_by_risk` | tail / mean | `w0` | `37.2756` | `5.9597` | `92.4410` | `0.0074` | Rot/endpoint 很强，但 ATE 明显失败 |
| `ACL2V5_TTCPF_04` | `native_to_candidate_by_risk` | both / mean | `w0` | `36.7548` | `6.3872` | `92.4375` | `0.0078` | both overlap 未优于 tail q90 |

Trajectory diagnostics：

| Run | ATE RMSE | FinalErr | 50f mean | 100f mean | 200f mean | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SWKS3_03` | `36.4153` | `5.687` | `29.734` | `30.383` | `30.436` | `3.765` | `31.298748` |
| `TTEX_01` | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` |
| `TTCPF_01` | `36.7629` | `5.382` | `29.944` | `30.587` | `30.606` | `3.592` | `31.171910` |
| `TTCPF_02` | `36.6308` | `5.102` | `29.854` | `30.491` | `30.520` | `3.596` | `31.167796` |
| `TTCPF_03` | `37.2756` | `2.965` | `30.956` | `31.543` | `31.858` | `3.085` | `31.213325` |
| `TTCPF_04` | `36.7548` | `4.651` | `29.979` | `30.618` | `30.663` | `3.572` | `31.175444` |

20.46 结论：

1. 本批没有达到成功标准 `KITTI01 ATE < 30m`，也没有超过 TTT-only reference `TTEX_01 = 36.5932 / 6.4327`。
2. `native_to_candidate_by_risk` 的 tail q90 是本批最好，但 ATE 仍回退 `+0.0376m`；说明只用 overlap 风险标量缩放整个 branch fast-weight delta 太粗。
3. `old_decay_by_risk` 把 FinalErr 从 `5.120` 降到 `2.965`、Yaw 从 `3.596` 降到 `3.085`、Rot 从 `6.4327` 降到 `5.9597`，证明“缩短高风险 fast-weight 寿命”确实会改变传播污染；但它把 50/100/200-frame mean 全部拉差，不能作为主线。
4. 最坏 `[200,300)` 片段仍在 `78m` 左右，没有被 TTCPF 打掉；TTCPF 只在 endpoint / yaw 侧有信号。
5. 因此当前用户假设需要更细实现：**当前 chunk 允许动态 token 参与 TTT；只在向下一 chunk replay/commit 时，对 overlap 范围内的高动态 token 做 token-level 遗忘**。下一批从标量 fast-weight filter 转为 overlap-scoped hard replay token filter，SWA 继续关闭。

### 20.47 计划：TTT overlap forget（当前 chunk full TTT，提交 replay 只遗忘 overlap 动态 token）

本节继续执行用户判断：

```text
动态区域可能对当前 chunk TTT 有用；
不能在当前 chunk 内简单压制动态区域；
但动态区域不应该长期传到下一 chunk；
因此要在跨 chunk commit / replay 传播阶段遗忘它们。
```

20.46 的 TTCPF 是 fast-weight 标量过滤，太粗。20.47 改成 token-level replay 过滤：

- 当前 chunk 的 controlled pass / TTT 使用不变；
- 只在 `probe_ttt_write` 生成下一 chunk 的 `ttt_state` 时改 replay token；
- 非 overlap token 一律保留，继续提供 chunk 内低频几何背景；
- 只在 `tail_overlap` 或 `both_overlap` 里，对低 write-prior / 高动态 token 做 hard filter；
- reset 机制固定 `RESET_EVERY=5`，SWA 全部关闭。

新增实现：

```text
TTT_WRITE_REPLAY_TOKEN_FILTER_MODE = scoped_dynamic_veto / scoped_static_topk
TTT_WRITE_REPLAY_TOKEN_FILTER_SCOPE = tail_overlap / both_overlap
```

含义：

- `scoped_dynamic_veto`：scope 外 token 全保留；scope 内只保留 `prior >= threshold` 的静态/可写 token。
- `scoped_static_topk`：scope 外 token 全保留；scope 内只保留 static-score top ratio。

第一批小矩阵：

| Run | Filter | Scope | 参数 | Branch | 目的 |
|---|---|---|---|---|---|
| `ACL2V5_TTOVF_01` | `scoped_dynamic_veto` | `tail_overlap` | threshold `1.00` | `w0` | 只遗忘 tail seam 高动态 token |
| `ACL2V5_TTOVF_02` | `scoped_dynamic_veto` | `tail_overlap` | threshold `0.98` | `w0` | 比 01 更温和，避免误删可用动态信息 |
| `ACL2V5_TTOVF_03` | `scoped_static_topk` | `tail_overlap` | ratio `0.75` | `w0` | 只在 tail seam 内保留前 75% 静态 token |
| `ACL2V5_TTOVF_04` | `scoped_dynamic_veto` | `both_overlap` | threshold `1.00` | `w0` | 验证只看 tail 是否太窄 |

固定协议：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
RESET_EVERY = 5
SWA write / source replace = off
success target = KITTI01 ATE < 30m
```

Gate：

- 若达到 `KITTI01 ATE < 30m`，立即 repeat；
- 若超过 `TTEX_01 = 36.5932 / 6.4327`，保留为新 TTT-only base；
- 若只改善 Rot/Yaw/FinalErr 但 ATE 回退，则说明 overlap token 遗忘仍主要是姿态正则器，下一步要改 TTT replay objective / branch value-memory target，而不是继续浅扫 threshold。

### 20.47 实验结果：TTT overlap forget

运行记录：

- smoke `ACL2V5_TTOVF_SMOKE_C23pairall_tailScopedVeto` 通过，`END_FRAME=128`，约 `3.1 min`。
- smoke debug 显示 `ttt_replay_token_filter_mode=scoped_dynamic_veto`、`scope=tail_overlap`、`branch_mask=[0]`、`applied=True`。
- full batch 4 并发 GPU0-3：`2026-05-07 07:11:48 -> 07:37:03`，约 `25.3 min` 完成 4 个 full run。
- 固定 `RESET_EVERY=5`，未改 reset 机制；SWA 全部关闭。

Global metrics：

| Run | Filter | Scope | Branch | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---|---:|---:|---:|---:|---|
| `SWKS3_03` | SWA-active reference | - | - | `36.4153` | `6.6186` | n/a | n/a | v5 历史 best，但本阶段 SWA 暂停 |
| `TTEX_01` | TTT-only reference | - | - | `36.5932` | `6.4327` | n/a | n/a | 当前 TTT-only reference |
| `TTOVF_01` | scoped dynamic veto, thr `1.00` | tail_overlap | `w0` | `36.8998` | `5.7826` | `92.4355` | `0.0074` | Rot/Yaw 强，但 ATE 回退 |
| `TTOVF_02` | scoped dynamic veto, thr `0.98` | tail_overlap | `w0` | `36.8282` | `5.8306` | `92.4333` | `0.0074` | 比 01 温和，ATE 仍回退 |
| `TTOVF_03` | scoped static topk, ratio `0.75` | tail_overlap | `w0` | `36.7257` | `5.9924` | `92.4326` | `0.0074` | 本批 ATE 最好，但未超过 TTEX |
| `TTOVF_04` | scoped dynamic veto, thr `1.00` | both_overlap | `w0` | `36.8989` | `5.6643` | `92.4261` | `0.0075` | Rot 最好，ATE 失败 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SWKS3_03` | `36.4153` | `5.687` | `29.734` | `30.383` | `30.436` | `3.765` | `31.298748` |
| `TTEX_01` | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` |
| `TTOVF_01` | `36.8998` | `2.611` | `30.396` | `30.996` | `31.303` | `2.943` | `31.160512` |
| `TTOVF_02` | `36.8282` | `1.784` | `30.278` | `30.879` | `31.142` | `2.962` | `31.141631` |
| `TTOVF_03` | `36.7257` | `1.391` | `30.053` | `30.665` | `30.822` | `3.152` | `31.125078` |
| `TTOVF_04` | `36.8989` | `4.140` | `30.411` | `31.012` | `31.350` | `2.819` | `31.057543` |

Worst segment check：

| Run | Worst 50f | Worst 100f | Worst 200f | 解释 |
|---|---:|---:|---:|---|
| `TTEX_01` | `78.815` | `78.750` | `57.753` | reference；最坏区间仍是 `[200,300)` 附近 |
| `TTOVF_01` | `78.997` | `78.926` | `57.912` | 未打掉最坏 ATE 段 |
| `TTOVF_02` | `79.085` | `78.969` | `57.938` | 未打掉最坏 ATE 段 |
| `TTOVF_03` | `79.171` | `78.814` | `57.852` | 100/200f 接近 reference，但 ATE 仍回退 |
| `TTOVF_04` | `78.519` | `78.394` | `57.590` | worst segment 略好，但全局 ATE 仍回退 |

20.47 结论：

1. 本批没有达到成功标准 `KITTI01 ATE < 30m`，也没有超过 TTT-only reference `TTEX_01 = 36.5932 / 6.4327`。
2. 用户判断“动态区域不能在当前 chunk 内简单压制”得到支持：当前 chunk full TTT、只在 commit replay 忘 overlap 动态 token，仍没有带来 ATE 突破，说明 ATE 主瓶颈不只是动态 token 被传到下一 chunk。
3. 但“动态区域长期传播会污染状态”也得到部分支持：`TTOVF_03` 把 FinalErr 从 `5.120` 降到 `1.391`，`TTOVF_04` 把 Rot 从 `6.4327` 降到 `5.6643`，Yaw 从 `3.596` 降到 `2.819`。
4. 这说明 overlap forgetting 是强姿态/endpoint 正则器，但还没有修掉 `[200,300)` ATE 主灾区；继续浅扫 threshold 意义不大。
5. 下一步停止 SWA，优先探索 TTT 写入传播的 branch-specific 机制：动态信息可能应该在 `w0/w1/w2` 中被不同方式遗忘，而不是只在 `w0` 上做统一 token filter。

### 20.48 计划：TTT overlap forget branch 拆解

20.47 只在 `w0` 上做 replay token filter，结果是 Rot/Yaw/FinalErr 强改善但 ATE 回退。这说明 `w0` 很可能更像姿态/几何正则分支；如果动态区域对当前 chunk 的 TTT 有用，但会污染跨 chunk 传播，那么真正需要遗忘的可能不是 `w0`，而是 value / hidden 分支的长期 fast-weight 记忆。

本节不碰 reset：`RESET_EVERY=5` 固定。SWA 继续全部关闭。

固定协议仍沿用 20.47：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
RESET_EVERY = 5
SWA write / source replace = off
success target = KITTI01 ATE < 30m
```

第一批 branch 拆解：

| Run | Filter | Scope | 参数 | Branch | 目的 |
|---|---|---|---|---|---|
| `ACL2V5_TTOVF2_01` | `scoped_dynamic_veto` | `tail_overlap` | threshold `1.00` | `w1` | 只遗忘 value 分支的 tail 动态传播 |
| `ACL2V5_TTOVF2_02` | `scoped_static_topk` | `tail_overlap` | ratio `0.75` | `w1` | value 分支温和保留 top static token |
| `ACL2V5_TTOVF2_03` | `scoped_dynamic_veto` | `tail_overlap` | threshold `1.00` | `w2` | 测试 hidden/output 分支是否是长期污染源 |
| `ACL2V5_TTOVF2_04` | `scoped_dynamic_veto` | `tail_overlap` | threshold `1.00` | `w0,w1,w2` | 强遗忘对照，验证 all-branch 是否彻底破坏 ATE |

预期判读：

- 如果 `w1` filter 改善 ATE，说明动态区域应在当前 chunk 使用，但 value-memory 不能跨 chunk 留存；
- 如果只有 `w0` 改善 Rot/FinalErr，`w1/w2` 改善 ATE，则下一步做 branch-aware mixed policy；
- 如果 all-branch 更差，说明动态信息确实不能被无差别清除；
- 若全部不超过 `TTEX_01`，下一步转向 TTT replay objective：让动态 token 参与当前 update，但对 commit delta 做 branch/layer-aware EMA 或 projection，而不是 hard token drop。

### 20.48 实验结果：TTT overlap forget branch 拆解

运行记录：

- 4 并发 GPU0-3：`2026-05-07 07:42:08 -> 08:06:28`，约 `24.3 min`。
- 实际输出目录：`results/kitti01_hmc_v2/attention_cue_library_v1/ACL2V5_TTOVF2_*`。
- 固定 `RESET_EVERY=5`，未改 reset 机制；SWA 全部关闭。

Global metrics：

| Run | Filter | Branch | ATE RMSE | Rot RMSE | RPE t | RPE r | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| `TTEX_01` | TTT-only reference | - | `36.5932` | `6.4327` | n/a | n/a | 当前 TTT-only reference |
| `TTOVF_03` | tail static topk `0.75` | `w0` | `36.7257` | `5.9924` | `92.4326` | `0.0074` | 20.47 hard forget best |
| `TTOVF2_01` | tail dynamic veto `1.00` | `w1` | `37.3193` | `5.7067` | `92.4401` | `0.0087` | Rot/Yaw 强，但 final/ATE 崩 |
| `TTOVF2_02` | tail static topk `0.75` | `w1` | `36.9656` | `6.2057` | `92.4381` | `0.0076` | 温和 w1 仍回退 |
| `TTOVF2_03` | tail dynamic veto `1.00` | `w2` | `36.8408` | `6.2291` | `92.4344` | `0.0076` | branch 拆解 ATE 最好，但未过 TTEX |
| `TTOVF2_04` | tail dynamic veto `1.00` | `w0,w1,w2` | `38.5165` | `6.2397` | `92.4358` | `0.0107` | all-branch hard forget 明显失败 |

Trajectory diagnostics：

| Run | ATE RMSE | Final error | 50f mean ATE | 100f mean ATE | 200f mean ATE | Yaw RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SWKS3_03` | `36.4153` | `5.687` | `29.734` | `30.383` | `30.436` | `3.765` | `31.298748` |
| `TTEX_01` | `36.5932` | `5.120` | `29.809` | `30.445` | `30.519` | `3.596` | `31.219899` |
| `TTOVF_03` | `36.7257` | `1.391` | `30.053` | `30.665` | `30.822` | `3.152` | `31.125078` |
| `TTOVF2_01` | `37.3193` | `14.612` | `31.558` | `32.127` | `32.981` | `2.863` | `31.190796` |
| `TTOVF2_02` | `36.9656` | `2.928` | `30.755` | `31.358` | `31.649` | `3.371` | `31.176393` |
| `TTOVF2_03` | `36.8408` | `1.703` | `30.174` | `30.798` | `30.985` | `3.429` | `31.160979` |
| `TTOVF2_04` | `38.5165` | `26.727` | `33.031` | `33.583` | `35.044` | `3.445` | `31.178117` |

20.48 结论：

1. 本批没有达到 `KITTI01 ATE < 30m`，也没有超过 `TTEX_01 = 36.5932 / 6.4327`。
2. `w1` hard dynamic veto 会严重伤 FinalErr 和 ATE，说明 value 分支不能简单删 overlap 动态 token。
3. `w2` hard dynamic veto 是 branch 拆解里最稳的 ATE 点，但仍比 TTEX 回退 `+0.2476m`；它更像 endpoint regularizer，不是 ATE 突破。
4. all-branch hard forget 直接失败，强力支持用户判断：动态区域确实给 TTT 当前/短期更新提供重要信息，不能无差别清除。
5. 下一步改成 **soft forgetting / replay blend**：当前 chunk 与 full replay 都保留，commit 时只把目标 branch 的 fast weight 软拉向 overlap-static replay，而不是完全替换成 static-only replay。

### 20.49 计划：TTT overlap soft forget（full replay -> static replay 软混合）

20.47/20.48 都是 hard replacement：目标 branch 直接采用 filtered replay 的 fast weight。它证明“忘动态”方向有 Rot/FinalErr 信号，但 hard 替换太粗，ATE 回退。

新的写法：

```text
W_commit_branch = W_full_branch + alpha * (W_filtered_branch - W_full_branch)
```

其中：

- `alpha=0` 等于原始 full replay；
- `alpha=1` 等于 20.47/20.48 的 hard forget；
- `0<alpha<1` 表示动态信息仍主要保留，只在跨 chunk commit 时逐步遗忘。

计划新增参数：

```text
TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND = 0.0-1.0
```

第一批小矩阵：

| Run | Filter | Branch | Blend alpha | 目的 |
|---|---|---|---:|---|
| `ACL2V5_TTOVFB_01` | tail static topk `0.75` | `w0` | `0.25` | 20.47 最强 endpoint 信号的温和版 |
| `ACL2V5_TTOVFB_02` | tail static topk `0.75` | `w0` | `0.50` | 检查 w0 soft forget 是否存在 ATE/Rot trade-off sweet spot |
| `ACL2V5_TTOVFB_03` | tail dynamic veto `1.00` | `w2` | `0.25` | 20.48 branch-best 的温和版 |
| `ACL2V5_TTOVFB_04` | tail dynamic veto `1.00` | `w2` | `0.50` | 检查 w2 soft forget 是否比 hard forget 更稳 |

固定协议继续：

```text
seq = KITTI01 full
cue = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
read path = frame pair/all
beta = 4.75
mode = hybrid
commit = probe_ttt_write
write score = stage_d_x_dg_inv_sqrt
TTT_WRITE_NATIVE_MIX_SCALES = 1.10,1.00,1.00
RESET_EVERY = 5
SWA = off
success target = KITTI01 ATE < 30m
```

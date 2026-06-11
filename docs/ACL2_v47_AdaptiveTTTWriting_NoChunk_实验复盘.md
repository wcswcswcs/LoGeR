# ACL2 v47 Adaptive TTT Writing No-Chunk 实验复盘

日期：2026-06-08（Asia/Singapore）  
执行日志：`docs/ACL2_v47_AdaptiveTTTWriting_NoChunk_执行日志.md`  
结果根目录：`results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk/`

本复盘只记录真实运行得到的数据。没有运行完成的实验标注为 `not_run`，没有落盘指标不补写。

## 当前状态

```text
代码准备已完成。
语法检查已通过。
v47 adaptive TTT writer 96F smoke 已完成。
三条 d_tok adaptive writer full KITTI01 run 已完成。
最终结论：当前 adaptive TTT writing 是真自适应，但性能显著低于 v46B fixed-percentage TTT 与 C9/P0。
```

## 用户问题的精确定义

用户所说的 adaptive TTT writing 不是：

```text
只把 fixed tri replay 的 role assignment 改成 adaptive_quantile。
```

而是：

```text
1. TTT 写入策略本身自适应。
2. 不依赖手工置顶百分比做 positive / neutral / negative triple replay。
3. 不依赖 C9/P0 那种 absolute chunk-id gamma / replay params。
4. 同时要解决 v46B 暴露出的速度问题，不能把更多计算堆到 CPU 上。
```

## 已完成代码修改审计

| 文件 | 修改 | 目的 | 当前验证 |
|---|---|---|---|
| `loger/pipeline/ttt_write_controller.py` | 新增 `adaptive_writer_fused` / `no_percentage_fused` 等 role modes | 用在线 risk 分布自适应决定正/中/负写入角色和负写入强度，不使用手工百分比 | `py_compile` pass |
| `loger/pipeline/ttt_write_controller.py` | adaptive writer 支持 `gamma=0` 仍激活 tri replay | 使 external gamma 不再是人工强度参数 | `py_compile` pass |
| `loger/pipeline/ttt_write_controller.py` | fused replay：把 pos/neutral/neg 合成一次 replay | 降低旧 tri replay 多次 replay 的速度开销 | `py_compile` pass，待 timing 验证 |
| `run_pipeline_abc_v2.py` | CLI choices 增加 adaptive writer modes | 允许 launcher 选择新策略 | `py_compile` pass |
| `tools/run_v47_adaptive_ttt_writer_candidate.sh` | 新增 no-chunk adaptive TTT writer launcher | 运行 AW010/AW110/AW111，并写审计 artifact | `bash -n` pass |
| `tools/v47_adaptive_ttt_writer_report.py` | 新增 landed-artifact report | 汇总 ATE、debug、timing，并对比 C9/P0 与 v46B fixed TTT rows | `py_compile` pass |

## Smoke 结果

运行：

```text
V47_SMOKE_AW010_96F
row = AW010_ADAPTIVE_TTT_ONLY
frames = 96
```

审计结论：

```text
adaptive_writer_fused 实际进入 controller debug。
manual tri replay percentages 在 launcher 中均为 0。
chunk-id maps 均为空。
```

关键 debug：

```text
role_modes_seen = adaptive_writer_fused
role_sources_seen = adaptive_writer_otsu3
adaptive_writer_fused_debug_count = 72
adaptive_gamma_mean = 0.001319117669481784
```

速度字段：

```text
probe_ttt_write_seconds_mean = 3.450065791606903
chunk_total_seconds_mean = 32.96995633840561
```

说明：

```text
96F smoke 的 ATE=0.9102923573955306 不是 full KITTI01 ATE，不能和 C9/P0 的 33.76294210291885 比。
它只用于证明路径、debug 和 timing 有效。
```

Report 修复：

```text
1. tools/v47_adaptive_ttt_writer_report.py 修正 _load_kitti_gt 三返回值。
2. 修正 01.log regex 中 raw string 的 \s 写法，以抓取 adaptive_gamma / role mode / fused count。
```

这些修复只影响 report 解析，不改变模型运行或实验结果。

## 与旧 adaptive_quantile 的区别

旧 `adaptive_quantile`：

```text
positive fraction 仍固定为 0.35；
negative fraction 只是在 0.05 / 0.12 / 0.18 间切换；
仍然是多次 replay 的 tri replay；
不能算用户要求的 fully adaptive writing。
```

新 `adaptive_writer_fused`：

```text
positive / neutral / negative masks 来自当前 risk histogram 的 Otsu-style 三段划分；
不使用手工 top percentage；
negative gamma 由 risk_gap、negative mass、token count 自适应计算；
pos/neutral/neg 合并成一个 signed replay vector，只跑一次 replay。
```

## 实验矩阵

| Row | 定义 | 对比对象 |
|---|---|---|
| `AW010_ADAPTIVE_TTT_ONLY` | adaptive TTT writer only，无 frame read，无 SWA | v46B `F010_ONLY_TTT` |
| `AW110_FRAME_ADAPTIVE_TTT` | frame read + adaptive TTT writer | v46B `F110_FRAME_ATTN_TTT` |
| `AW111_FRAME_ADAPTIVE_TTT_SWA` | frame read + adaptive TTT writer + SWA | v46B `F111_ALL_THREE` |

共同约束：

```text
READ_BETA_FRAME_CHUNKS=""
TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
TTT_WRITE_COMMIT_EMA_CHUNKS=""
TTT_WRITE_GRADIENT_REVERSAL_GAMMA=0.0
TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC=0.0
TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC=0.0
TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA=0.0
TTT_WRITE_TRI_REPLAY_ROLE_MODE=adaptive_writer_fused
```

## 当前必须回答的问题

### Q1：ttt 写入策略现在是自适应的吗？

当前答案：

```text
是，96F smoke 与 full KITTI01 run 均证明 adaptive_writer_fused 在完整 pipeline 中实际生效。
它不再使用手工 positive/negative/neutral 百分比。
它也不使用 C9/P0 absolute chunk-id gamma / replay params。
但它没有达到性能目标。
```

### Q2：改成自适应之后对比以前性能下降多少？

当前答案：

```text
相对 v46B fixed-percentage TTT 对应行，性能下降：
AW010 vs F010: +1.8657988665103034m
AW110 vs F110: +1.6575598992521208m
AW111 vs F111: +1.671403777497659m
```

对比方式已经固定：

```text
AW010 - F010_ONLY_TTT
AW110 - F110_FRAME_ATTN_TTT
AW111 - F111_ALL_THREE
```

## Gate

用户要求：

```text
目标逼近 C9/P0 ATE = 33.76294210291885；
约 0.3m 容忍区间可视为 acceptable。
```

因此：

```text
ATE <= 34.06294210291885 为 acceptable。
```

当前：

```text
best = AW111, ATE = 38.322144227956436
38.322144227956436 > 34.06294210291885
acceptable = false
```

## speed blocker 诊断更新

### 发现的问题

初始 `V47_SMOKE_AW010_96F` 跑得很快：

```text
probe_ttt_write_seconds_mean = 3.450065791606903
```

但它后来被证明不是有效的 update-conflict risk 测试。原因是：

```text
adaptive writer 设定 external gamma=0；
旧逻辑在 gamma=0 时没有真正构造 update_conflict_energy risk，
导致 requested risk source 被 cheap prior-like/fallback 路径替代。
```

这说明最初 smoke 只证明了 adaptive writer 路径可运行，不能证明 update-conflict 方案速度可接受。

### 修复与真实 update-conflict 速度

修复：

```text
loger/pipeline/ttt_write_controller.py
  adaptive_writer/no_percentage role 下，gamma=0 仍允许 requested risk source 生效。
```

真实 update-conflict smoke：

| Run | risk source | probe_ttt_write mean | chunk total mean | 结论 |
|---|---|---:|---:|---|
| `V47_RISKFIX_SMOKE_AW010_96F` | `update_conflict_energy` | `69.28046560287476` | `105.64987802505493` | 太慢 |
| `V47_GPUFIX_SMOKE_AW010_96F` | `update_conflict_energy` + GPU/log 优化尝试 | `75.27440929412842` | `113.34883093833923` | 仍太慢 |

分析：

```text
CPU clone / debug vector 不是主因。
主要速度拖累来自 update_conflict_energy 的风险构造本身。
```

### d_tok adaptive writer smoke

为保留“自适应写入”定义，同时避免 update-conflict 的计算灾难，改用当前 chunk 已有的 `d_tok` risk：

```text
V47_DTOK_SMOKE_AW010_96F
frames = 96
hmc_rows = 4
```

关键证据：

```text
ttt_gradient_reversal_risk_source = d_tok
ttt_gradient_reversal_risk_source_applied = true
ttt_tri_replay_role_mode = adaptive_writer_fused
ttt_tri_replay_role_source = adaptive_writer_otsu3
manual positive/negative/neutral frac = 0
external gamma = 0
chunk-id maps = empty
```

速度：

```text
probe_ttt_write_seconds_mean = 3.504811644554138
chunk_total_seconds_mean = 33.28179794549942
adaptive_gamma_mean = 0.0006464838952524588
```

对比：

```text
update_conflict_energy: 69.28-75.27s/chunk probe TTT write
d_tok adaptive writer:  3.50s/chunk probe TTT write
```

结论：

```text
当前主要速度 blocker 不是 adaptive_writer_fused，也不是 fused replay；
是 update_conflict_energy risk source。
d_tok 是当前可运行的 adaptive TTT writer 候选，因为它不使用手工百分比/手工 gamma/chunk map，同时速度可接受。
```

### 重要边界

`V47_DTOK_SMOKE_AW010_96F` 的 96F ATE：

```text
ATE = 0.9392492203205386
```

不能和 C9/P0 full KITTI01 ATE 比较。它只是 smoke 和速度数据。

## full KITTI01 最终结果

三条 `d_tok` adaptive writer full run 均已完成：

```text
frames = 1101
hmc_rows = 38
status = done
```

结果表：

| Run | Row | ATE_full | delta vs C9/P0 | v46B fixed reference | delta vs fixed TTT | probe_ttt mean | chunk mean |
|---|---|---:|---:|---|---:|---:|---:|
| `V47_DTOK_AW010_ADAPTIVE_TTT_ONLY` | TTT only | `41.38490315066118` | `+7.621961047742332` | `F010_ONLY_TTT=39.51910428415088` | `+1.8657988665103034` | `3.489066061220671` | `38.025220808229946` |
| `V47_DTOK_AW110_FRAME_ADAPTIVE_TTT` | frame + TTT | `38.326470740397355` | `+4.563528637478505` | `F110_FRAME_ATTN_TTT=36.668910841145234` | `+1.6575598992521208` | `3.464312277342144` | `37.73894845184527` |
| `V47_DTOK_AW111_FRAME_ADAPTIVE_TTT_SWA` | frame + TTT + SWA | `38.322144227956436` | `+4.5592021250375865` | `F111_ALL_THREE=36.65074045045878` | `+1.671403777497659` | `3.633767454247726` | `39.45773262099216` |

审计字段：

```text
no_chunk_policy_pass = True
adaptive_writer_audit_pass = True
role_modes_seen = adaptive_writer_fused
role_sources_seen = adaptive_writer_otsu3
manual positive/negative/neutral frac = 0
external gamma = 0
adaptive_gamma_mean = 0.0005682417494857585
ttt_positive_mass_mean = 0.665234426134511
ttt_neutral_mass_mean = 0.20001342618151716
ttt_negative_mass_mean = 0.13475215238960167
```

### 对用户问题的直接回答

Q：TTT 写入策略现在是自适应的吗？

```text
是。v47 d_tok full runs 使用 adaptive_writer_fused：
positive / neutral / negative role 由当前 chunk risk histogram 的 Otsu-style 三段划分决定；
negative gamma 由 risk_gap、negative mass、token count 自适应计算；
manual tri replay percentages 全为 0；
C9/P0 chunk-id maps 全为空。
```

Q：改成自适应之后对比以前性能下降多少？

```text
相对 v46B fixed-percentage TTT 对应行，性能下降：
AW010 vs F010: +1.8657988665103034m
AW110 vs F110: +1.6575598992521208m
AW111 vs F111: +1.671403777497659m
```

相对 C9/P0：

```text
AW010: +7.621961047742332m
AW110: +4.563528637478505m
AW111: +4.5592021250375865m
```

因此：

```text
当前 adaptive TTT writing 已经是真自适应，但性能不合格。
它解决了速度，不解决准确率。
```

### 分析

1. **速度 blocker 已定位并绕开。**  
   `update_conflict_energy` risk source 的 probe TTT write 是 `69-75s/chunk`；`d_tok` adaptive writer 是 `3.46-3.63s/chunk`。所以真正拖慢的是 update-conflict risk 构造，不是 adaptive writer/fused replay 本身。

2. **d_tok risk 太弱或太粗，不能替代 fixed TTT 写入策略。**  
   三条 full run 都比 v46B fixed 对应行差约 `+1.66m` 到 `+1.87m`，说明用 D_tok histogram 做 Otsu 三分虽然自适应，但没有捕捉到 C9/fixed tri replay 中有效的 token 写入结构。

3. **frame read 和 SWA 仍有辅助作用，但不足以修复 adaptive writer。**  
   `AW010` 是 `41.3849`，加入 frame read 后 `AW110` 到 `38.3265`，改善约 `3.0584m`；再加 SWA 后 `AW111` 只到 `38.3221`，额外收益约 `0.0043m`，基本中性。

4. **adaptive mass 分布与 fixed TTT 差异很大。**  
   v47 full 的平均 mass 是：

   ```text
   pos / neutral / neg = 0.6652 / 0.2000 / 0.1348
   ```

   v46B fixed TTT 是：

   ```text
   pos / neutral / neg ≈ 0.35 / 0.53 / 0.12
   ```

   这表明 Otsu/D_tok 自适应分组把过多 token 放进 positive replay，可能稀释了稳定写入或放大了错误写入。

5. **adaptive gamma 量级偏小。**  
   v47 full 的 `adaptive_gamma_mean=0.000568`，而 C9 手工 gamma map 最高到 `0.005`、常用 `0.003-0.005`。这可能解释为什么负写入抑制不足，但不能简单手动调大 gamma，否则又回到手工策略。

### 结论

可以写：

```text
v47 实现并验证了 no-chunk/no-manual-percentage adaptive TTT writing。
速度瓶颈来自 update_conflict_energy risk source；d_tok risk source 可以把 probe TTT write 降回约 3.5s/chunk。
但 d_tok adaptive writer 的 full KITTI01 ATE 明显差于 v46B fixed TTT 和 C9/P0。
```

不能写：

```text
不能写 adaptive TTT writing 已经替代 C9 手工 tri replay。
不能写自适应策略性能接近 C9/P0。
不能把速度修复等同于算法成功。
```

### 下一步 insight

后续如果继续，不应回到手工百分比，而应改进自适应信号：

```text
1. 不用 update_conflict_energy 的重计算版本，尝试缓存/近似的 update-conflict proxy。
2. 对 Otsu/D_tok role mass 加入自适应温度或熵约束，但不能固定 top percentage。
3. 让 adaptive gamma 使用 sequence-agnostic 统计校准，例如由 risk distribution 的 robust scale 自动决定，而不是手工 gamma map。
4. 优先修正 positive mass 过高的问题，因为当前 0.665 positive mass 明显偏离 v46B fixed TTT 的有效分布。
```

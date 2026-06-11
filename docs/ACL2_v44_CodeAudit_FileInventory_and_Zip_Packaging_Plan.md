# ACL2 v44：核心代码审核清单与代码包打包计划

日期：2026-05-26  
目标读者：新接手的编程 AI / Codex / 研究助理  
项目：LoGeR Pipeline v2 / Hybrid Memory Controller / Semantic Prior Generator / KITTI01 Target-30

---

## 0. 这份计划要解决什么问题

当前项目已经进行了大量实验，代码里同时存在很多历史功能、诊断功能、失败路线、实验性 hook 和仍然有效的主线逻辑。新的编程 AI 如果直接看整个仓库，很容易被以下内容误导：

```text
1. 历史实验开关很多，难以判断哪些是当前主线。
2. 有些代码是 diagnostic，只用于定位问题，不应该被当作 deployable 策略。
3. 有些参数按 absolute chunk-id 生效，短期能复现 KITTI01，但不具备迁移意义。
4. 语义、SWA、TTT、READ path 的控制逻辑分散在多个文件里。
5. 实验 launcher 的环境变量和 Python CLI 参数之间容易出现接线错误。
6. 有些失败实验不是科学失败，而是 hook 没生效、配置继承污染、cache 缺失、baseline drift 或 run dir 污染。
```

因此，本计划要求 Codex 先做一件很具体的事：

> **列出项目中最核心、最需要审核的代码文件，说明每个文件为什么重要、应该检查什么，然后把这些核心文件、必要脚本和最小说明文档打包成一个 zip，交给新的编程 AI 快速理解项目。**

这不是新实验计划，不跑 KITTI full run，不改模型逻辑。它是一次 **代码地图 + 审核包生成任务**。

---

## 1. 当前项目目标，必须写在代码包 README 顶部

新的编程 AI 必须先理解项目目标，否则容易继续走偏。

### 1.1 项目主目标

当前目标是：

```text
在 training-free、不训练新模型、不使用 GT runtime action、不针对 KITTI01 过拟合的前提下，
通过 LoGeR 内部 cue、语义/几何/外观健康信号和 memory control，
把 KITTI01 full-online ATE 从当前最好 C9_P0_R2 = 33.7629m 推进到 30m 以下。
```

### 1.2 当前最好可部署基线

当前最好可部署 online TTT-write baseline 是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

但 C9 有两个重要问题：

```text
1. 它不是一个干净的最终策略，因为内部还存在 absolute chunk-id policy。
2. 它的组件贡献还没有完全拆清楚。
```

最新 v43 实验证明：

```text
C9 locked repeat 可以精确复现：ATE = 33.76294210291885m。
flat dechunk 后明显退化：best flat ATE = 35.29521801485317m。
移除 TTT tri-replay 退化最大：ATE = 36.2098947787m，delta vs C9 = +2.4469526758m。
最好的 semantic READ full-online 候选到 ATE = 33.4875667508m，delta vs C9 = -0.2753753521m，但仍未达到最低有效进展门槛。
```

### 1.3 当前短期目标

当前短期目标不是继续扩大语义大矩阵，而是：

```text
1. 清理 C9 中的 absolute chunk-id policy。
2. 解释 C9 组件贡献。
3. 找到一个能在 C9 full-online 上真实降低 ATE 的 semantic / geometry READ policy。
4. 优先先把 C9 从 33.7629m 推到 33m 内，再冲 30m。
```

---

## 2. 严禁再次走偏的项目边界

Codex 在做代码审核和打包时必须在 README 里写明这些边界。

```text
禁止训练 trigger / selector / classifier / role router。
禁止用 oracle label 拟合规则。
禁止用 absolute chunk id 作为最终 runtime policy。
禁止针对 KITTI01 或任何单一 sequence 调专用参数。
禁止使用 GT semantic / GT trajectory / GT pose 作为 runtime action。
禁止把 short rollout / fixed-window diagnostic / GT diagnostic 写成 deployable online success。
禁止把 semantic cache 命中、no-op parity、hook smoke 写成 ATE 成功。
禁止把 failed / partial / config-mismatch run 写入成功结果。
```

允许的事情：

```text
允许用其他 sequence 做 sanity / failure-mode diagnosis。
允许用 SemanticKITTI 3D projection 做 offline trust calibration。
允许用 fixed chunk / freeze / oracle 做 diagnostic，但必须标注不可部署。
允许用 health metrics 诊断 bad chunk，但不能用 chunk id 本身作为策略。
```

---

## 3. 本次任务的最终交付物

Codex 需要生成以下文件：

```text
code_audit_pack/
    README_for_new_programming_ai.md
    CORE_CODE_INVENTORY.csv
    CORE_CODE_INVENTORY.md
    CODE_DEPENDENCY_MAP.md
    CONFIG_AND_RUNTIME_FLOW.md
    HIGH_RISK_AUDIT_CHECKLIST.md
    ABSOLUTE_CHUNK_ID_POLICY_AUDIT.md
    SEMANTIC_MEMORY_PATH_AUDIT.md
    TTT_SWA_READ_HOOK_AUDIT.md
    PACKAGED_FILES.sha256
    package_manifest.json
    files/
        <核心代码文件按原始相对路径复制到这里>

core_code_audit_pack.zip
core_code_audit_pack.sha256
```

其中：

```text
README_for_new_programming_ai.md:
    面向没有项目背景的人解释项目目标、当前主线、不要走偏的边界、从哪里开始看代码。

CORE_CODE_INVENTORY.csv:
    每一行一个文件，至少包含 path, category, priority, reason, key_symbols, audit_questions。

CODE_DEPENDENCY_MAP.md:
    解释 run_pipeline_abc_v2.py -> HMC -> pi3.py / attention.py / TTTWriteController / SemanticPriorGenerator 的数据流。

HIGH_RISK_AUDIT_CHECKLIST.md:
    列出最容易出 bug 的地方和检查方法。

ABSOLUTE_CHUNK_ID_POLICY_AUDIT.md:
    专门列出所有 chunk-id 控制参数、代码位置、是否仍在 runtime 主线中。

PACKAGED_FILES.sha256:
    zip 内所有文件的 sha256。
```

---

## 4. 核心代码文件清单：必须审核并打包

下面是最小必打包清单。Codex 必须先检查这些文件是否存在。如果文件不存在，不能静默跳过，必须在 `MISSING_FILES.md` 中说明。

### 4.1 Pipeline 入口与主运行逻辑

| 优先级 | 文件 | 为什么重要 | 必查内容 |
|---|---|---|---|
| P0 | `run_pipeline_abc_v2.py` | Pipeline v2 主入口，所有 HMC、TTT、SWA、语义、chunk、reset、full-online 参数都从这里进入 | CLI 参数是否完整；环境变量是否正确转发；chunk/global offset 是否正确；Stage C cache 是否安全；C9 默认是否可复现 |
| P1 | `run_pipeline_abc.py` | Pipeline v1 / Stage A-E 的旧接口和工具函数，v2 复用大量 helper | helper 是否会对 no-op 产生副作用；Stage C lazy/cache 逻辑；chunk merge 函数 |
| P1 | `tools/run_attention_cue_experiment.sh` | 标准实验 launcher，把环境变量转成 Python CLI | C9 配置是否完整；semantic/SWA/TTT 参数是否正确转发；是否存在隐式默认值污染 |

### 4.2 Hybrid Memory Controller 核心

| 优先级 | 文件 | 为什么重要 | 必查内容 |
|---|---|---|---|
| P0 | `loger/pipeline/hybrid_memory_controller.py` | HMC 核心：D_g、read control、semantic roles、SWA/TTT control、write score 都在这里组织 | `stage_d_x_dg_inv_sqrt`；semantic role 是否真的进 path；absolute chunk id gate；C9 组件；debug JSONL 是否足够 |
| P0 | `loger/pipeline/ttt_write_controller.py` | TTT 写入控制核心：tri-replay、positive/negative/neutral、update conflict、commit EMA、native mix | tri-replay 贡献；chunk gamma map；no-long-write / short-negative 是否真影响 post-zp delta；risk source 是否正确 |
| P1 | `loger/pipeline/dynamic_cue_extractor.py` | 几何 cue 实现，输出 C_dyn/C_stat/C_anchor/G_write_geo 等 | 当前几何 cue 是否只是简化实现；stage_d 是否由这里或 HMC 内部重建；跨 chunk 信息是否缺失 |

### 4.3 模型 hook：READ / SWA / attention

| 优先级 | 文件 | 为什么重要 | 必查内容 |
|---|---|---|---|
| P0 | `loger/models/pi3.py` | LoGeR 模型执行和 READ/SWA/context source hook 主要位置 | frame/global source skip 是否真进入 K/V；SWA read/write hook 是否真生效；compact_kv 是否保 query；context empty events |
| P0 | `loger/models/layers/attention.py` | attention 层与 compact_kv / source skip 相关实现 | source removal 是否是 K/V compaction；attention mass instrumentation 是否可用；是否保留 query token |

### 4.4 语义前端与 Semantic Prior Generator

| 优先级 | 文件 | 为什么重要 | 必查内容 |
|---|---|---|---|
| P0 | `loger/pipeline/semantic_prior_generator.py` | Semantic Prior Generator，负责语义 role、fine/coarse labels、masklet projection | fine label 是否稳定；G_sem/L_sem/R_frame/R_swa/R_ttt 是否对齐；语义是否仍是 coarse fallback |
| P0 | `loger/pipeline/video_masklet_frontend.py` | Video Masklet Front-end，Stage C 输出语义 masklets | cache hit；masklet temporal consistency；label instability；coverage；是否包含 sky/vegetation/road/building |
| P1 | `loger/pipeline/gt_semantic_provider.py` | GT semantic / SemanticKITTI / KITTI-STEP / KITTI-360 provider | 只做 offline diagnostic；禁止 predicted fallback 冒充 GT；SemanticKITTI projection ignore pixels |

### 4.5 Evaluation / diagnostics / result aggregation

| 优先级 | 文件 | 为什么重要 | 必查内容 |
|---|---|---|---|
| P1 | `tools/kitti_trajectory_diagnostics.py` | 生成 segment、rolling、FinalErr、Yaw、Sim3 scale 等诊断 | segment 定义是否清楚；rolling metrics 是否可用于 scene-agnostic diagnosis |
| P1 | `tools/v43_registry_summarize.py` | v43 组件贡献归因汇总 | attribution delta 是否相对正确 baseline；是否避免 partial/invalid rows |
| P1 | `tools/v41_health_detector_report.py` | v41 health detector，曾选出 chunk10 | 是否用 ATE；是否用固定 chunk/segment；health components 如何聚合 |
| P1 | `tools/v41_read_gate_report.py` | READ h10/h15 gate report | h15 gate 是否过保守；stress window 是否只做 diagnostic |
| P1 | `tools/v39_semantic_appearance_atlas.py` | 语义/外观异常 atlas | 是否落盘 per-label spatial maps；sky causality 是否可证明 |
| P1 | `tools/v36b_context_skip_summary.py` | context source skip 与 attention mass summary | skipped source 是否真的有 attention mass；compact_kv 是否真实 |
| P1 | `tools/v36b_h0c_action_smoke_report.py` | action distinguishability smoke | 不同 semantic/SWA/TTT action 是否实际不同 |

### 4.6 关键 run launcher / phase scripts

| 优先级 | 文件 | 为什么重要 | 必查内容 |
|---|---|---|---|
| P0 | `tools/run_v43_full_candidate.sh` | 最新 v43 full-online C9/dechunk/attribution/semantic READ launcher | C9 locked defaults；flat/dechunk 候选；component attribution；semantic READ 是否 no chunk-id |
| P1 | `tools/run_v42_full_candidate.sh` | v42 health-gated full online launcher | semantic_action_active_chunks 是否导致过拟合；baseline drift 修复 |
| P1 | `tools/run_v24_candidate_rollout.sh` | 多轮短 rollout 候选 launcher，被 v24-v41 多次复用 | aliases 是否污染；read_path / hybrid / probe_native 是否按候选正确设置 |
| P1 | `tools/run_v36b_snapshot_generation.sh` | H9/C9 parent snapshots 生成 | causal fork parent state 是否正确；snapshot chunk 是否完整 |

---

## 5. 审核重点：Codex 必须重点回答的问题

### 5.1 C9 是否干净？

Codex 必须列出 C9 的所有关键控制项：

```text
read cue
read beta
read_beta_frame_chunks
hmc_write_score_source
TTT tri-replay 参数
tri-replay gamma chunk map
native_mix
commit_ema_chunks
SWA overlap source replacement
Stage C / semantic 是否关闭
```

并回答：

```text
哪些是全局固定值？
哪些是 absolute chunk-id policy？
哪些只是 diagnostic？
哪些必须在下一步清理或固定化？
```

### 5.2 组件贡献是否清楚？

Codex 必须基于 v43 attribution 的实现和结果，说明：

```text
remove TTT tri-replay 为什么退化最大？
remove tri gamma chunk map 为什么也退化？
read / SWA / native_mix / commit EMA 各自贡献是多少？
是否存在组件交互，不能用单独 remove 简单相加？
```

### 5.3 语义是否真的参与 C9？

必须明确写出：

```text
C9_P0_R2 不使用 Stage C semantic。
C9 的 write score 使用 stage_d_x_dg_inv_sqrt。
这里的 stage_d / D_g 都不是语义标签。
Semantic Prior Generator 是后续实验模块，不是 C9 成功的来源。
```

### 5.4 READ / SWA / TTT hook 是否真的改变计算？

不要只看候选名字。必须检查：

```text
READ:
    被 skip/attenuate 的 source token 数量；
    source attention mass before/after；
    compact_kv 是否真的移除 K/V source；
    query 是否保留。

SWA:
    K/V keep mask 是否不同；
    overlap vs non-overlap 是否区分；
    boundary 10f / 20f 是否恶化；
    source cache 是否真的变化。

TTT:
    positive/negative/no-write token mass；
    per-branch update norm；
    post-zp delta norm；
    commit EMA / native mix 是否真正改变 fast weights。
```

### 5.5 Stage C / VideoMasklet cache 是否可信？

必须检查：

```text
cache require-hit 是否开启；
no-op parity 是否通过；
fine labels 是否稳定；
coarse/fine semantic 是否混用；
SemanticKITTI projection 是否只做 offline trust calibration；
是否存在 predicted fallback 冒充 GT。
```

---

## 6. 必须搜索的高风险关键词

Codex 应在 repo 根目录运行以下 grep，并把结果整理到 `HIGH_RISK_AUDIT_CHECKLIST.md`：

```bash
grep -R "read_beta_frame_chunks" -n run_pipeline_abc_v2.py loger tools || true
grep -R "commit_ema_chunks" -n run_pipeline_abc_v2.py loger tools || true
grep -R "ttt_write_gradient_reversal_chunks" -n run_pipeline_abc_v2.py loger tools || true
grep -R "TTT_WRITE_GRADIENT_REVERSAL_CHUNKS" -n tools loger || true
grep -R "semantic_action_active_chunks" -n run_pipeline_abc_v2.py loger tools || true
grep -R "stage_d_x_dg_inv_sqrt" -n run_pipeline_abc_v2.py loger tools || true
grep -R "stage_d_x_sem" -n run_pipeline_abc_v2.py loger tools || true
grep -R "context_source_skip" -n run_pipeline_abc_v2.py loger tools || true
grep -R "compact_kv" -n loger tools || true
grep -R "update_conflict_energy" -n loger tools || true
grep -R "probe_ttt_write" -n run_pipeline_abc_v2.py loger tools || true
grep -R "Stage C" -n run_pipeline_abc_v2.py loger/pipeline tools || true
grep -R "SemanticKITTI" -n loger tools docs || true
```

Codex 必须给每类关键词一个总结：

```text
出现在哪些文件？
是否是 runtime 主路径？
是否有 absolute chunk-id 风险？
是否有 semantic / non-semantic 混淆风险？
是否需要进入 zip？
```

---

## 7. 自动生成 inventory 的建议脚本

Codex 可以把下面脚本保存为 `tools/build_core_code_inventory.py`，或直接在临时脚本中运行。

```python
#!/usr/bin/env python3
from pathlib import Path
import csv
import hashlib

ROOT = Path.cwd()
MANDATORY = [
    "run_pipeline_abc_v2.py",
    "run_pipeline_abc.py",
    "loger/pipeline/hybrid_memory_controller.py",
    "loger/pipeline/ttt_write_controller.py",
    "loger/pipeline/dynamic_cue_extractor.py",
    "loger/pipeline/semantic_prior_generator.py",
    "loger/pipeline/video_masklet_frontend.py",
    "loger/pipeline/gt_semantic_provider.py",
    "loger/models/pi3.py",
    "loger/models/layers/attention.py",
    "tools/run_attention_cue_experiment.sh",
    "tools/run_v43_full_candidate.sh",
    "tools/run_v42_full_candidate.sh",
    "tools/run_v24_candidate_rollout.sh",
    "tools/kitti_trajectory_diagnostics.py",
    "tools/v43_registry_summarize.py",
    "tools/v41_health_detector_report.py",
    "tools/v41_read_gate_report.py",
    "tools/v39_semantic_appearance_atlas.py",
    "tools/v36b_context_skip_summary.py",
    "tools/v36b_h0c_action_smoke_report.py",
]

CATEGORY = {
    "run_pipeline_abc_v2.py": "entrypoint",
    "run_pipeline_abc.py": "legacy_entrypoint_helpers",
    "hybrid_memory_controller.py": "hmc_core",
    "ttt_write_controller.py": "ttt_write_core",
    "dynamic_cue_extractor.py": "geometry_cue",
    "semantic_prior_generator.py": "semantic_prior",
    "video_masklet_frontend.py": "video_masklet",
    "gt_semantic_provider.py": "gt_semantic_offline",
    "pi3.py": "model_hooks",
    "attention.py": "attention_hooks",
}

REASONS = {
    "run_pipeline_abc_v2.py": "main Pipeline v2 CLI and runtime orchestration",
    "run_pipeline_abc.py": "shared Stage C / merge / helper functions reused by v2",
    "loger/pipeline/hybrid_memory_controller.py": "central HMC prior, D_g, READ/SWA/TTT semantic control wiring",
    "loger/pipeline/ttt_write_controller.py": "TTT tri-replay, update conflict, commit EMA, native mix, post-zp write actions",
    "loger/pipeline/dynamic_cue_extractor.py": "geometry cue implementation and write eligibility support",
    "loger/pipeline/semantic_prior_generator.py": "semantic roles, masklet projection, fine/coarse label routing",
    "loger/pipeline/video_masklet_frontend.py": "Stage C VideoMasklet frontend producing semantic masklets",
    "loger/pipeline/gt_semantic_provider.py": "offline GT/SemanticKITTI provider; must not be runtime fallback",
    "loger/models/pi3.py": "model-level READ/SWA/context source hooks and compact_kv controls",
    "loger/models/layers/attention.py": "attention / compact K/V source removal implementation",
    "tools/run_attention_cue_experiment.sh": "standard launcher converting env vars into CLI args",
    "tools/run_v43_full_candidate.sh": "latest C9/dechunk/attribution/semantic READ full-online launcher",
    "tools/run_v42_full_candidate.sh": "health-selected semantic READ full-online launcher and active chunk gate",
    "tools/run_v24_candidate_rollout.sh": "trusted short rollout alias launcher reused across many semantic experiments",
    "tools/kitti_trajectory_diagnostics.py": "segment/rolling/final/yaw trajectory diagnostics",
    "tools/v43_registry_summarize.py": "latest component attribution and semantic READ summary",
    "tools/v41_health_detector_report.py": "training-free health detector used for bad chunk diagnosis",
    "tools/v41_read_gate_report.py": "READ h10/h15 gate logic",
    "tools/v39_semantic_appearance_atlas.py": "semantic/appearance anomaly audit",
    "tools/v36b_context_skip_summary.py": "context source skip attention-mass and compact_kv evidence",
    "tools/v36b_h0c_action_smoke_report.py": "action distinguishability smoke summary",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def line_count(path: Path) -> int:
    try:
        return len(path.read_text(errors="replace").splitlines())
    except Exception:
        return -1

rows = []
missing = []
for rel in MANDATORY:
    p = ROOT / rel
    if not p.exists():
        missing.append(rel)
        continue
    key = p.name
    rows.append({
        "path": rel,
        "category": CATEGORY.get(key, "tool_or_support"),
        "priority": "P0" if rel in MANDATORY[:10] or rel.endswith("run_v43_full_candidate.sh") else "P1",
        "lines": line_count(p),
        "sha256": sha256(p),
        "reason": REASONS.get(rel, "core audit file"),
    })

out = ROOT / "CORE_CODE_INVENTORY.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "category", "priority", "lines", "sha256", "reason"])
    writer.writeheader()
    writer.writerows(rows)

if missing:
    (ROOT / "MISSING_FILES.md").write_text("# Missing mandatory files\n\n" + "\n".join(f"- `{m}`" for m in missing) + "\n")
    raise SystemExit(2)

print(f"Wrote {out} with {len(rows)} rows")
```

---

## 8. 打包 zip 的具体要求

### 8.1 不允许打包的内容

Codex 必须排除：

```text
ckpts/
data/
results/
third_party/
__pycache__/
*.pt
*.pth
*.ckpt
*.bin
*.label
*.mp4
*.avi
*.png
*.jpg
*.jpeg
*.zip
*.tar
*.tar.gz
```

例外：如果某个小的 `.png` 是 report 图，默认也不打包。代码审核包只打代码、脚本、少量说明文档。

### 8.2 推荐 zip 目录结构

```text
code_audit_pack/
    README_for_new_programming_ai.md
    CORE_CODE_INVENTORY.csv
    CORE_CODE_INVENTORY.md
    CODE_DEPENDENCY_MAP.md
    CONFIG_AND_RUNTIME_FLOW.md
    HIGH_RISK_AUDIT_CHECKLIST.md
    ABSOLUTE_CHUNK_ID_POLICY_AUDIT.md
    SEMANTIC_MEMORY_PATH_AUDIT.md
    TTT_SWA_READ_HOOK_AUDIT.md
    PACKAGED_FILES.sha256
    package_manifest.json
    files/
        run_pipeline_abc_v2.py
        run_pipeline_abc.py
        loger/pipeline/...
        loger/models/...
        tools/...
```

### 8.3 打包命令模板

Codex 可以使用下面的 bash 模板。

```bash
set -euo pipefail

PACK_ROOT="code_audit_pack"
ZIP_NAME="core_code_audit_pack.zip"
rm -rf "$PACK_ROOT" "$ZIP_NAME" core_code_audit_pack.sha256
mkdir -p "$PACK_ROOT/files"

# 1. 先生成 inventory
python tools/build_core_code_inventory.py
cp CORE_CODE_INVENTORY.csv "$PACK_ROOT/CORE_CODE_INVENTORY.csv"

# 2. 按 inventory 复制文件，保留相对路径
python - <<'PY'
from pathlib import Path
import csv, shutil
pack = Path('code_audit_pack/files')
with open('CORE_CODE_INVENTORY.csv', newline='') as f:
    for row in csv.DictReader(f):
        src = Path(row['path'])
        dst = pack / row['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
PY

# 3. 复制人工说明文档
for f in \
  README_for_new_programming_ai.md \
  CORE_CODE_INVENTORY.md \
  CODE_DEPENDENCY_MAP.md \
  CONFIG_AND_RUNTIME_FLOW.md \
  HIGH_RISK_AUDIT_CHECKLIST.md \
  ABSOLUTE_CHUNK_ID_POLICY_AUDIT.md \
  SEMANTIC_MEMORY_PATH_AUDIT.md \
  TTT_SWA_READ_HOOK_AUDIT.md \
  package_manifest.json
  do
    if [ -f "$f" ]; then cp "$f" "$PACK_ROOT/$f"; fi
  done

# 4. 生成 sha256
(
  cd "$PACK_ROOT"
  find . -type f | sort | xargs sha256sum > PACKAGED_FILES.sha256
)

# 5. 打包
zip -r "$ZIP_NAME" "$PACK_ROOT" >/tmp/core_code_audit_zip.log
sha256sum "$ZIP_NAME" > core_code_audit_pack.sha256

# 6. 最终检查
unzip -l "$ZIP_NAME" > core_code_audit_pack_filelist.txt
python - <<'PY'
from pathlib import Path
zip_path = Path('core_code_audit_pack.zip')
size_mb = zip_path.stat().st_size / 1024 / 1024
print(f'zip_size_mb={size_mb:.2f}')
if size_mb > 50:
    raise SystemExit('zip too large; remove non-code artifacts')
PY
```

---

## 9. Codex 需要生成的 README 内容结构

`README_for_new_programming_ai.md` 必须包含以下小节。

```text
1. Project goal in plain language
2. Current best baseline: C9_P0_R2
3. What C9 does and what C9 does not do
4. Why absolute chunk-id policy is a problem
5. Where READ / SWA / TTT live in code
6. What Semantic Prior Generator is supposed to do
7. What went wrong in previous semantic routes
8. Which files to read first
9. Which files are dangerous to modify
10. How to run no-op / smoke / full-online safely
11. What not to claim as success
```

每个缩写必须第一次出现时解释。例如：

```text
TTT = Test-Time Training fast-weight memory, the compressed long-term memory used by LoGeR.
SWA = Sliding Window Attention, the local short-term memory for adjacent chunk alignment.
READ path = the current chunk's attention/source-reading path, before writing future memory.
D_g = internal attention cue map, high value means unstable / dynamic / unreliable patch.
C23 = acl2.gg.qq.low.g2_3.past_only.headmean.robustq, current main internal attention cue.
Stage C = Video Masklet Front-end semantic cache.
SPG = Semantic Prior Generator.
```

---

## 10. 审核判断标准

这次代码审核包生成任务的成功标准不是 ATE，而是信息完整性。

### 10.1 必须通过的 gate

```text
1. 所有 P0 mandatory files 都存在并被打包。
2. CORE_CODE_INVENTORY.csv 至少包含 path/category/priority/sha256/reason。
3. README_for_new_programming_ai.md 能让新 AI 在 10 分钟内理解项目目标与代码入口。
4. ABSOLUTE_CHUNK_ID_POLICY_AUDIT.md 列出所有 chunk-id 相关代码位置。
5. HIGH_RISK_AUDIT_CHECKLIST.md 包含 grep 结果和风险解释。
6. zip 不包含 checkpoint/data/results/large tensor/video/image artifacts。
7. zip 大小建议小于 50MB。
8. core_code_audit_pack.sha256 存在。
```

### 10.2 如果 gate 不过怎么办

```text
如果 P0 文件缺失：
    先用 find / grep 搜索替代路径，仍找不到则生成 MISSING_FILES.md 并停止打包。

如果 zip 超过 50MB：
    删除非代码 artifact，不删除 P0 源码。

如果发现 checkpoint/data/results 被打包：
    立即删除 zip，修正 exclude，再重新打包。

如果 grep 发现新的 chunk-id policy：
    记录到 ABSOLUTE_CHUNK_ID_POLICY_AUDIT.md，不要现场改代码。

如果 README 缺少缩写解释：
    不允许交付。
```

---

## 11. 给新编程 AI 的最短读代码顺序

Codex 必须在 README 中建议新 AI 按下面顺序读代码：

```text
1. run_pipeline_abc_v2.py
   先理解主 pipeline、chunk、reset、Stage C、HMC 参数怎么进入。

2. loger/pipeline/hybrid_memory_controller.py
   再理解 D_g、write score、semantic role、READ/SWA/TTT 控制如何组成 prior。

3. loger/models/pi3.py + loger/models/layers/attention.py
   看 READ source skip、compact_kv、SWA cache hook 是否真的改变模型计算。

4. loger/pipeline/ttt_write_controller.py
   看 TTT tri-replay、update_conflict_energy、commit EMA、native mix 等写入逻辑。

5. loger/pipeline/semantic_prior_generator.py + video_masklet_frontend.py
   看语义从 masklet 到 token 的投影和 role 生成。

6. tools/run_attention_cue_experiment.sh + tools/run_v43_full_candidate.sh
   看实验参数如何被拼成 full run。

7. tools/v43_registry_summarize.py + kitti_trajectory_diagnostics.py
   看结果如何被评估、汇总和归因。
```

---

## 12. 这次打包之后，下一步代码审核应该优先看什么

这次 zip 交付后，新编程 AI 的第一轮代码审核不应该大范围改代码，而应该先回答四个问题：

```text
Q1. C9 中所有 absolute chunk-id policy 在哪里？
Q2. C9 的各组件是否能用固定全局值替代 chunk map？
Q3. v43 component attribution 的实现是否正确？
Q4. semantic READ 候选 SEM_READ_03 为什么能改善 0.275m，但没有达到 0.5m？
```

如果这四个问题没回答清楚，不要继续开新的大矩阵。

---

## 13. 最终提醒

这次任务的目标是让新的编程 AI 快速理解项目，不是让它马上追 ATE。

最容易犯的错误是：

```text
看到很多开关，就继续扫参数；
看到语义接口，就继续跑 all-memory semantic matrix；
看到 freeze5 局部有效，就尝试 hard reset TTT；
看到 chunk10 有病灶，就写死 chunk10；
看到短 rollout 变好，就宣布 deployable success。
```

这些都不允许。

正确路线是：

```text
先理解代码；
先清理不可迁移的 chunk-id policy；
先拆清 C9 贡献；
再只做极少量 full-online 候选验证。
```


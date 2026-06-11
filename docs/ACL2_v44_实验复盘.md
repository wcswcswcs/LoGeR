# ACL2 v44 实验复盘：CodeAudit FileInventory and Zip Packaging

日期：2026-05-27（Asia/Singapore）  
计划文件：`docs/ACL2_v44_CodeAudit_FileInventory_and_Zip_Packaging_Plan.md`  
执行日志：`docs/ACL2_v44_执行日志.md`  
交付目录：`code_audit_pack/`  
交付 zip：`core_code_audit_pack.zip`

本轮原则：v44 是代码地图与审核包生成任务，不是新实验。不跑 KITTI full run，不修改模型逻辑，不产生 ATE success claim。

---

## 0. 当前结论

```text
v44 已按计划完成代码审核包生成与 zip 打包。

已完成：
    1. 阅读 v44 计划。
    2. 检查 mandatory 核心文件。
    3. 确认所有 mandatory 文件存在：
           missing_files = []
           packaged_code_file_count = 22
    4. 生成 CORE_CODE_INVENTORY.csv / md。
    5. 生成 README_for_new_programming_ai.md。
    6. 生成 dependency/config/risk/chunk-id/semantic/hook audit 文档。
    7. 复制核心代码文件到 code_audit_pack/files/，保留原始相对路径。
    8. 生成 PACKAGED_FILES.sha256 和 package_manifest.json。
    9. 打包 core_code_audit_pack.zip。
    10. 生成 core_code_audit_pack.sha256 和 core_code_audit_pack_filelist.txt。
    11. 验证 zip 不包含禁入内容。

Final:
    package_gate_pass = true
    zip_size = 320K
    zip_sha256 =
        bf75b3fb9fa58d2149177bf9e748b226710bfe37d51255a40f30e9d4148f7047
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

v44 没有产生新 ATE、没有新 deployable success，也没有改动 runtime 逻辑。

---

## 1. 交付物

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
    MISSING_FILES.md
    PACKAGED_FILES.sha256
    package_manifest.json
    files/
        run_pipeline_abc_v2.py
        run_pipeline_abc.py
        loger/pipeline/hybrid_memory_controller.py
        loger/pipeline/ttt_write_controller.py
        loger/pipeline/dynamic_cue_extractor.py
        loger/pipeline/semantic_prior_generator.py
        loger/pipeline/video_masklet_frontend.py
        loger/pipeline/gt_semantic_provider.py
        loger/models/pi3.py
        loger/models/layers/attention.py
        tools/run_attention_cue_experiment.sh
        tools/run_v43_full_candidate.sh
        tools/run_v42_full_candidate.sh
        tools/run_v24_candidate_rollout.sh
        tools/run_v36b_snapshot_generation.sh
        tools/kitti_trajectory_diagnostics.py
        tools/v43_registry_summarize.py
        tools/v41_health_detector_report.py
        tools/v41_read_gate_report.py
        tools/v39_semantic_appearance_atlas.py
        tools/v36b_context_skip_summary.py
        tools/v36b_h0c_action_smoke_report.py

core_code_audit_pack.zip
core_code_audit_pack.sha256
core_code_audit_pack_filelist.txt
```

---

## 2. Inventory Summary

| Metric | Value |
|---|---:|
| `mandatory_file_count` | `22` |
| `packaged_code_file_count` | `22` |
| `missing_files` | `[]` |
| `CORE_CODE_INVENTORY.csv lines` | `23` |
| `PACKAGED_FILES.sha256 lines` | `33` |
| `zip entries including dirs` | `41` |
| `zip size` | `320K` |

`CORE_CODE_INVENTORY.csv` includes:

```text
path
category
priority
lines
sha256
reason
key_symbols
audit_questions
```

---

## 3. Risk Audit Summary

Generated：

```text
HIGH_RISK_AUDIT_CHECKLIST.md
ABSOLUTE_CHUNK_ID_POLICY_AUDIT.md
SEMANTIC_MEMORY_PATH_AUDIT.md
TTT_SWA_READ_HOOK_AUDIT.md
```

Key findings recorded for the next programming AI：

```text
1. C9_P0_R2 remains best deployable:
       ATE = 33.7629421029m

2. C9 is not mechanism-clean:
       read_beta_frame_chunks
       ttt_write_gradient_reversal_chunks
       commit_ema_chunks
   remain absolute chunk-id policy surfaces.

3. v43 component attribution says C9 depends strongly on:
       TTT tri-replay
       tri gamma chunk map
       no-chunk-id interactions

4. C9 does not use Stage C semantic labels as success source.
   Stage C / SPG / SemanticKITTI paths must remain clearly separated from C9.

5. Semantic READ best lead is diagnostic only:
       SEM_READ_03_C23_RESID_READ_ONLY
       ATE = 33.4875667508m
       delta vs C9 = -0.2753753521m
       deployable_success = false
```

---

## 4. Blocker / 修复记录

### Blocker 1：`python` 命令不存在

现象：

```text
/bin/bash: python: command not found
```

修复：

```text
使用项目 conda Python：
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
```

结果：

```text
code_audit_pack generated
core_code_audit_pack.zip generated
```

Boundary：

```text
这是打包脚本运行环境问题，不涉及模型逻辑或实验结果。
```

---

## 5. Gate Check

| Gate | Status |
|---|---|
| All P0/P1 mandatory files exist | pass |
| Inventory has required fields | pass |
| README explains goal, C9, boundaries, acronyms, read order | pass |
| Absolute chunk-id audit present | pass |
| High-risk keyword checklist present | pass |
| Zip excludes forbidden artifacts | pass |
| Zip size < 50MB | pass |
| `core_code_audit_pack.sha256` exists | pass |

Forbidden artifact check：

```text
No forbidden artifact matched.
```

Zip sha：

```text
bf75b3fb9fa58d2149177bf9e748b226710bfe37d51255a40f30e9d4148f7047  core_code_audit_pack.zip
```

---

## 6. Final Decision

```text
v44 package_gate_pass = true.

The code audit pack is ready for a new programming AI / Codex / research
assistant to inspect the core LoGeR Pipeline v2, HMC, READ/SWA/TTT hooks,
semantic path, launchers, evaluation scripts, and high-risk chunk-id policy
surfaces.

No ATE experiment was run.
No deployable online success is claimed.
```

Next recommended reader flow：

```text
1. Open code_audit_pack/README_for_new_programming_ai.md.
2. Read CORE_CODE_INVENTORY.md.
3. Read ABSOLUTE_CHUNK_ID_POLICY_AUDIT.md and TTT_SWA_READ_HOOK_AUDIT.md.
4. Inspect files/run_pipeline_abc_v2.py.
5. Inspect files/loger/pipeline/hybrid_memory_controller.py.
6. Inspect files/loger/pipeline/ttt_write_controller.py.
7. Only then consider any new code changes or experiments.
```

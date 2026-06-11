# ACL2 v44 执行日志：CodeAudit FileInventory and Zip Packaging

日期：2026-05-27（Asia/Singapore）  
计划文件：`docs/ACL2_v44_CodeAudit_FileInventory_and_Zip_Packaging_Plan.md`  
交付目录：`code_audit_pack/`  
交付 zip：`core_code_audit_pack.zip`

本轮原则：只做代码地图、核心文件清单、风险审计说明与打包；不跑 full-online 实验；不改模型逻辑；不把任何诊断或历史实验写成 deployable success。

---

## 1. 读取计划与 mandatory 文件检查

读取计划：

```bash
sed -n '1,260p' docs/ACL2_v44_CodeAudit_FileInventory_and_Zip_Packaging_Plan.md
sed -n '260,620p' docs/ACL2_v44_CodeAudit_FileInventory_and_Zip_Packaging_Plan.md
sed -n '620,980p' docs/ACL2_v44_CodeAudit_FileInventory_and_Zip_Packaging_Plan.md
```

检查 mandatory 文件：

```bash
for f in run_pipeline_abc_v2.py run_pipeline_abc.py \
  loger/pipeline/hybrid_memory_controller.py \
  loger/pipeline/ttt_write_controller.py \
  loger/pipeline/dynamic_cue_extractor.py \
  loger/pipeline/semantic_prior_generator.py \
  loger/pipeline/video_masklet_frontend.py \
  loger/pipeline/gt_semantic_provider.py \
  loger/models/pi3.py \
  loger/models/layers/attention.py \
  tools/run_attention_cue_experiment.sh \
  tools/run_v43_full_candidate.sh \
  tools/run_v42_full_candidate.sh \
  tools/run_v24_candidate_rollout.sh \
  tools/kitti_trajectory_diagnostics.py \
  tools/v43_registry_summarize.py \
  tools/v41_health_detector_report.py \
  tools/v41_read_gate_report.py \
  tools/v39_semantic_appearance_atlas.py \
  tools/v36b_context_skip_summary.py \
  tools/v36b_h0c_action_smoke_report.py; do
    if [ -f "$f" ]; then printf 'OK %s\n' "$f"; else printf 'MISSING %s\n' "$f"; fi
done
```

结果：

```text
All mandatory files exist.
Missing mandatory files = 0
```

---

## 2. 高风险关键词审计

执行等价关键词扫描：

```bash
rg -n "read_beta_frame_chunks|commit_ema_chunks|ttt_write_gradient_reversal_chunks|TTT_WRITE_GRADIENT_REVERSAL_CHUNKS|semantic_action_active_chunks|stage_d_x_dg_inv_sqrt|stage_d_x_sem|context_source_skip|compact_kv|update_conflict_energy|probe_ttt_write|Stage C|SemanticKITTI" \
  run_pipeline_abc_v2.py loger tools docs \
  --glob '!**/__pycache__/**' \
  --glob '!*.png' --glob '!*.jpg' --glob '!*.zip'
```

处理：

```text
原始匹配超过 1000 行；未把完整噪声塞入包内。
已在 code_audit_pack/HIGH_RISK_AUDIT_CHECKLIST.md 中按 keyword 汇总：
    hit count
    main files
    risk summary
    immediate audit priorities
```

---

## 3. 打包生成

首次尝试使用 `python` 失败：

```text
/bin/bash: python: command not found
```

修复：

```text
改用项目 conda Python:
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
```

生成命令：

```bash
PYBIN=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

$PYBIN - <<'PY'
# 机械生成 code_audit_pack:
#   CORE_CODE_INVENTORY.csv/md
#   README_for_new_programming_ai.md
#   CODE_DEPENDENCY_MAP.md
#   CONFIG_AND_RUNTIME_FLOW.md
#   HIGH_RISK_AUDIT_CHECKLIST.md
#   ABSOLUTE_CHUNK_ID_POLICY_AUDIT.md
#   SEMANTIC_MEMORY_PATH_AUDIT.md
#   TTT_SWA_READ_HOOK_AUDIT.md
#   MISSING_FILES.md
#   package_manifest.json
#   files/<mandatory files copied with relative paths>
#   PACKAGED_FILES.sha256
PY

zip -r core_code_audit_pack.zip code_audit_pack >/tmp/core_code_audit_zip.log
sha256sum core_code_audit_pack.zip > core_code_audit_pack.sha256
unzip -l core_code_audit_pack.zip > core_code_audit_pack_filelist.txt
```

结果：

```text
generated code_audit_pack with 22 code files and 33 checksummed files
zip_size_mb = 0.31
```

---

## 4. Gate 检查

检查 zip sha：

```bash
cat core_code_audit_pack.sha256
```

结果：

```text
bf75b3fb9fa58d2149177bf9e748b226710bfe37d51255a40f30e9d4148f7047  core_code_audit_pack.zip
```

检查 inventory / packaged sha 行数：

```bash
wc -l code_audit_pack/CORE_CODE_INVENTORY.csv code_audit_pack/PACKAGED_FILES.sha256
```

结果：

```text
23 code_audit_pack/CORE_CODE_INVENTORY.csv
33 code_audit_pack/PACKAGED_FILES.sha256
```

检查 zip 大小：

```bash
du -h core_code_audit_pack.zip
```

结果：

```text
320K core_code_audit_pack.zip
```

检查禁入内容：

```bash
unzip -Z1 core_code_audit_pack.zip | \
  rg '(^|/)(ckpts|data|results|third_party|__pycache__)/|\.(pt|pth|ckpt|bin|label|mp4|avi|png|jpg|jpeg|zip|tar|gz)$' || true
```

结果：

```text
No forbidden artifact matched.
```

检查 zip file count：

```bash
unzip -Z1 core_code_audit_pack.zip | wc -l
```

结果：

```text
41
```

---

## 5. 最终交付物

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
        <22 mandatory/core files copied with original relative paths>

core_code_audit_pack.zip
core_code_audit_pack.sha256
core_code_audit_pack_filelist.txt
```

Final gate：

```text
P0/P1 mandatory files present = true
CORE_CODE_INVENTORY.csv has required fields = true
README includes project goal, C9 baseline, boundaries, acronyms, read order = true
absolute chunk-id audit present = true
high-risk keyword checklist present = true
zip excludes forbidden artifacts = true
zip size < 50MB = true
core_code_audit_pack.sha256 exists = true
```

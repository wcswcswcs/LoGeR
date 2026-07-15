#!/usr/bin/env python3
"""Build a compact ACL2 v106R LingBot core-code plus artifact audit packet."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "code_audit_pack"
PACK_PREFIX = "acl2_v106r_lingbot_memory_operation_core_artifact_audit"
RESULT_ROOT = ROOT / "results/acl2_v106r_lingbot_memory_operation_discovery_semantic_aware_control"
SELF = Path("tools/build_v106r_lingbot_memory_operation_core_artifact_audit_pack.py")

DOC_FILES = [
    "docs/ACL2_v106R_LingBot_MemoryOperationDiscovery_SemanticAwareReadoutUpdateRetention_ExperimentPlan.md",
    "docs/ACL2_v106R_LingBot_MemoryOperationDiscovery_SemanticAwareReadoutUpdateRetention_执行日志.md",
    "docs/ACL2_v106R_LingBot_MemoryOperationDiscovery_SemanticAwareReadoutUpdateRetention_实验结果复盘.md",
]

V106R_EXPERIMENT_CODE = [
    "tools/build_v106r_stage0_v105_evidence_freeze.py",
    "tools/build_v106r_stage1_memory_operation_map.py",
    "tools/build_v106r_stage1_targeted_trace_configs.py",
    "tools/run_v106r_stage1_targeted_trace_manifest.py",
    "tools/build_v106r_stage1_targeted_trace_summary.py",
    "tools/finalize_v106r_stage1_no_memory_lever.py",
    "tools/build_v105tf_lingbot_stage3_oracle.py",
]

LINGBOT_CORE_FILES = [
    "third_party/lingbot-map/README.md",
    "third_party/lingbot-map/benchmark/README.md",
    "third_party/lingbot-map/benchmark/README_zh.md",
    "third_party/lingbot-map/benchmark/prepare.py",
    "third_party/lingbot-map/benchmark/run_worker.py",
    "third_party/lingbot-map/benchmark/methods/lingbot_map.py",
    "third_party/lingbot-map/benchmark/io/image.py",
    "third_party/lingbot-map/benchmark/io/intrinsics.py",
    "third_party/lingbot-map/lingbot_map/heads/camera_head.py",
    "third_party/lingbot-map/lingbot_map/layers/attention.py",
    "third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py",
    "third_party/lingbot-map/lingbot_map/models/gct_base.py",
    "third_party/lingbot-map/lingbot_map/models/gct_stream_window_v2.py",
]

PACKER_FILES = [SELF.as_posix()]

RESULT_ARTIFACTS = [
    "final_decision/final_decision.json",
    "final_decision/final_decision.md",
    "stage0_v105_evidence_freeze/available_artifact_manifest.csv",
    "stage0_v105_evidence_freeze/forbidden_repeat_list.md",
    "stage0_v105_evidence_freeze/stage0_freeze_report.md",
    "stage0_v105_evidence_freeze/stage0_summary.json",
    "stage0_v105_evidence_freeze/v105_action_surface_summary.csv",
    "stage0_v105_evidence_freeze/v105_headlocal_selected_manifest.csv",
    "stage0_v105_evidence_freeze/v105_known_facts.json",
    "stage0_v105_evidence_freeze/v105_platform_decision_copy.md",
    "stage1_memory_operation_map/memory_lever_rank.csv",
    "stage1_memory_operation_map/memory_lever_report.md",
    "stage1_memory_operation_map/memory_lever_sequence_split.csv",
    "stage1_memory_operation_map/memory_operation_rows.csv",
    "stage1_memory_operation_map/non_readout_operation_observability_report.md",
    "stage1_memory_operation_map/stage1_no_memory_lever_found.md",
    "stage1_memory_operation_map/stage1_summary.json",
    "stage1_memory_operation_map/targeted_trace/config_generation_summary.json",
    "stage1_memory_operation_map/targeted_trace/no_action_parity_rows.csv",
    "stage1_memory_operation_map/targeted_trace/pre_safe_l3_selection_fix/run_results.csv",
    "stage1_memory_operation_map/targeted_trace/pre_safe_l3_selection_fix/run_results.jsonl",
    "stage1_memory_operation_map/targeted_trace/pre_safe_l3_selection_fix/target_manifest.csv",
    "stage1_memory_operation_map/targeted_trace/pre_safe_l3_selection_fix/targeted_memory_lever_rank.csv",
    "stage1_memory_operation_map/targeted_trace/pre_safe_l3_selection_fix/targeted_trace_summary.json",
    "stage1_memory_operation_map/targeted_trace/run_manifest.csv",
    "stage1_memory_operation_map/targeted_trace/run_results.csv",
    "stage1_memory_operation_map/targeted_trace/run_results.jsonl",
    "stage1_memory_operation_map/targeted_trace/run_results_bad_global_idx.csv",
    "stage1_memory_operation_map/targeted_trace/run_results_bad_global_idx.jsonl",
    "stage1_memory_operation_map/targeted_trace/target_manifest.csv",
    "stage1_memory_operation_map/targeted_trace/targeted_context_role_token_rows.csv",
    "stage1_memory_operation_map/targeted_trace/targeted_frame_semantic_geometry_rows.csv",
    "stage1_memory_operation_map/targeted_trace/targeted_headlocal_frame_head_features.csv",
    "stage1_memory_operation_map/targeted_trace/targeted_kv_cache_provenance_rows.csv",
    "stage1_memory_operation_map/targeted_trace/targeted_memory_lever_rank.csv",
    "stage1_memory_operation_map/targeted_trace/targeted_memory_lever_report.md",
    "stage1_memory_operation_map/targeted_trace/targeted_memory_lever_sequence_split.csv",
    "stage1_memory_operation_map/targeted_trace/targeted_memory_operation_rows.csv",
    "stage1_memory_operation_map/targeted_trace/targeted_trace_summary.json",
    "stage1_memory_operation_map/targeted_trace_requirement.md",
    "stage2_semantic_increment_map/semantic_increment_not_run_due_stage1_no_memory_lever.md",
    "stage3_role_disambiguation/role_assignment_not_run_due_stage1_no_memory_lever.md",
    "stage4_action_surface_screen/action_surface_screen_blocked_by_stage1_no_memory_lever.md",
    "stage5_runtime_pilot_or_blocked/runtime_failure_report.md",
    "stage6_full_validation_or_blocked/full_validation_blocked.md",
]

EXCLUDED_ARTIFACTS = [
    "stage1_memory_operation_map/targeted_trace/raw_trace/",
    "stage1_memory_operation_map/targeted_trace/workspace/",
    "stage1_memory_operation_map/targeted_trace/pre_safe_l3_selection_fix/raw_trace/",
    "stage1_memory_operation_map/targeted_trace/targeted_gca_context_trace_rows.csv",
    "stage1_memory_operation_map/targeted_trace/targeted_trace_semantic_key_rows.csv",
]

EXCLUDED_SUFFIXES = {
    ".avi",
    ".bin",
    ".ckpt",
    ".exr",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".tgz",
}

EXCLUDED_NAMES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "checkpoints",
    "code_audit_pack",
    "raw_trace",
    "wandb",
    "workspace",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "output", "")
        return f"{type(exc).__name__}: {exc}\n{output}"


def should_exclude_archive(path: Path) -> bool:
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return bool(set(path.parts) & EXCLUDED_NAMES)


def add_file(
    src_rel: str,
    pack_dir: Path,
    manifest: list[dict[str, Any]],
    *,
    category: str,
    archive_rel: str | None = None,
) -> None:
    src = ROOT / src_rel
    archive = archive_rel or src_rel
    if not src.is_file():
        manifest.append(
            {
                "category": category,
                "source_path": src_rel,
                "archive_path": archive,
                "size_bytes": 0,
                "sha256": "",
                "status": "missing",
            }
        )
        return
    if should_exclude_archive(Path(archive)):
        raise ValueError(f"excluded archive path requested: {archive}")
    dst = pack_dir / archive
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "category": category,
            "source_path": src_rel,
            "archive_path": archive,
            "size_bytes": dst.stat().st_size,
            "sha256": sha256(dst),
            "status": "included",
        }
    )


def collect_sources() -> list[tuple[str, str, str | None]]:
    sources: list[tuple[str, str, str | None]] = []
    for item in DOC_FILES:
        sources.append(("v106r_doc", item, None))
    for item in V106R_EXPERIMENT_CODE:
        sources.append(("v106r_experiment_code", item, None))
    for item in LINGBOT_CORE_FILES:
        sources.append(("lingbot_core", item, None))
    for item in PACKER_FILES:
        sources.append(("packaging_code", item, None))
    for item in RESULT_ARTIFACTS:
        src_rel = (RESULT_ROOT / item).relative_to(ROOT).as_posix()
        archive_rel = f"artifacts/{src_rel}"
        sources.append(("v106r_result_artifact", src_rel, archive_rel))
    for item in sorted((RESULT_ROOT / "configs").rglob("*.yaml")) if (RESULT_ROOT / "configs").is_dir() else []:
        src_rel = item.relative_to(ROOT).as_posix()
        archive_rel = f"artifacts/{src_rel}"
        sources.append(("v106r_runtime_config", src_rel, archive_rel))
    return sources


def write_manifest(pack_dir: Path, manifest: list[dict[str, Any]]) -> None:
    keys = ["category", "source_path", "archive_path", "size_bytes", "sha256", "status"]
    rows = ["\t".join(keys)]
    for row in sorted(manifest, key=lambda item: (str(item["category"]), str(item["source_path"]))):
        rows.append("\t".join(str(row.get(key, "")) for key in keys))
    (pack_dir / "PACK_MANIFEST.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def result_root_file_count() -> int:
    if not RESULT_ROOT.exists():
        return 0
    return sum(1 for item in RESULT_ROOT.rglob("*") if item.is_file())


def du_text(path: Path) -> str:
    try:
        return subprocess.check_output(["du", "-sh", str(path)], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def observed_v106r_state(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    targeted_summary = RESULT_ROOT / "stage1_memory_operation_map/targeted_trace/targeted_trace_summary.json"
    final_decision = RESULT_ROOT / "final_decision/final_decision.json"
    summary_payload = json.loads(targeted_summary.read_text(encoding="utf-8")) if targeted_summary.is_file() else {}
    final_payload = json.loads(final_decision.read_text(encoding="utf-8")) if final_decision.is_file() else {}
    included_artifacts = [
        row["source_path"]
        for row in manifest
        if row["status"] == "included" and row["category"].startswith("v106r_")
    ]
    return {
        "schema": "acl2_v106r_pack_observed_state_v1",
        "result_root": RESULT_ROOT.relative_to(ROOT).as_posix(),
        "result_root_exists": RESULT_ROOT.exists(),
        "result_root_du": du_text(RESULT_ROOT),
        "result_file_count": result_root_file_count(),
        "included_artifact_count": len(included_artifacts),
        "excluded_artifacts": EXCLUDED_ARTIFACTS,
        "targeted_trace_summary": summary_payload,
        "final_decision": final_payload,
        "note": "Compact code plus audit-artifact packet. Raw trace/workspace and large expanded trace tables are intentionally excluded; summarized evidence and derived ranking tables are included.",
    }


def write_metadata(pack_dir: Path, manifest: list[dict[str, Any]]) -> None:
    metadata_dir = pack_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    included_archive = sorted(row["archive_path"] for row in manifest if row.get("status") == "included")
    included_source = sorted(row["source_path"] for row in manifest if row.get("status") == "included")
    source_for_git = [item for item in included_source if not item.startswith("results/")]
    lingbot_included = sorted(
        item.removeprefix("third_party/lingbot-map/")
        for item in source_for_git
        if item.startswith("third_party/lingbot-map/")
    )
    (metadata_dir / "SOURCE_GIT_HEAD.txt").write_text(run_text(["git", "rev-parse", "HEAD"]), encoding="utf-8")
    (metadata_dir / "SOURCE_GIT_STATUS_RELEVANT.txt").write_text(
        run_text(["git", "status", "--short", "--", *source_for_git]),
        encoding="utf-8",
    )
    (metadata_dir / "SOURCE_GIT_DIFF_RELEVANT.patch").write_text(
        run_text(["git", "diff", "--", *source_for_git]),
        encoding="utf-8",
    )
    (metadata_dir / "LINGBOT_GIT_HEAD.txt").write_text(
        run_text(["git", "-C", "third_party/lingbot-map", "rev-parse", "HEAD"]),
        encoding="utf-8",
    )
    (metadata_dir / "LINGBOT_GIT_STATUS_RELEVANT.txt").write_text(
        run_text(["git", "-C", "third_party/lingbot-map", "status", "--short", "--", *lingbot_included]),
        encoding="utf-8",
    )
    (metadata_dir / "LINGBOT_GIT_DIFF_RELEVANT.patch").write_text(
        run_text(["git", "-C", "third_party/lingbot-map", "diff", "--", *lingbot_included]),
        encoding="utf-8",
    )
    (metadata_dir / "OBSERVED_V106R_STATE.json").write_text(
        json.dumps(observed_v106r_state(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exclusion_report = {
        "schema": "acl2_v106r_pack_exclusion_report_v1",
        "excluded_artifacts": EXCLUDED_ARTIFACTS,
        "reason": "Excluded raw traces, worker output workspace, and oversized expanded trace tables. Included summaries, parity rows, operation rows, lever ranks, run manifests/results, final decision, and logs.",
    }
    (metadata_dir / "ARTIFACT_EXCLUSION_REPORT.json").write_text(
        json.dumps(exclusion_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    commands = [
        "python -m py_compile tools/build_v106r_lingbot_memory_operation_core_artifact_audit_pack.py",
        "python tools/build_v106r_lingbot_memory_operation_core_artifact_audit_pack.py",
        "cd code_audit_pack && sha256sum -c <archive>.zip.sha256",
        "unzip -tq code_audit_pack/<archive>.zip",
        "cd <extracted-packet-root> && sha256sum -c PAYLOAD_SHA256SUMS.txt",
    ]
    (metadata_dir / "BUILD_AND_VERIFY_COMMANDS.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    missing = [row["source_path"] for row in manifest if row.get("status") != "included"]
    included_by_category: dict[str, int] = {}
    size_by_category: dict[str, int] = {}
    for row in manifest:
        if row.get("status") == "included":
            included_by_category[row["category"]] = included_by_category.get(row["category"], 0) + 1
            size_by_category[row["category"]] = size_by_category.get(row["category"], 0) + int(row["size_bytes"])
    readme = f"""# ACL2 v106R LingBot Memory Operation Core+Artifact Audit Packet

Scope:
- v106R plan, execution log, and retrospective log.
- v106R Stage0/Stage1/finalizer source code plus the v105 semantic loader dependency.
- LingBot-Map core files needed to audit SDPA trace, KV cache provenance, and fixed cache append/eviction behavior.
- v106R final/blocking artifacts, Stage0 freeze artifacts, Stage1 summaries/ranks/operation rows, targeted run manifests/results, no-action parity rows, and runtime configs.
- Packaging script, manifests, relevant git status/diff, payload hashes, and validation sidecars.

Audit boundary:
- This is a compact reviewer packet, not a whole result-tree dump.
- Raw datasets, checkpoints, tensor dumps, EXR/depth outputs, worker workspaces, previous audit packs, raw trace JSONL, and two oversized expanded trace CSVs are excluded.
- Excluded raw/expanded artifacts are listed in `metadata/ARTIFACT_EXCLUSION_REPORT.json`.

Final v106R status in included artifacts:
- taxonomy: `{observed_v106r_state(manifest).get("final_decision", {}).get("taxonomy", "")}`
- action_stages_allowed: `{observed_v106r_state(manifest).get("final_decision", {}).get("action_stages_allowed", "")}`
- targeted_stage1_discovery_pass: `{observed_v106r_state(manifest).get("targeted_trace_summary", {}).get("targeted_stage1_discovery_pass", "")}`
- targeted_trace_parity_pass: `{observed_v106r_state(manifest).get("targeted_trace_summary", {}).get("targeted_trace_parity_pass", "")}`

Included file counts by category:
{json.dumps(included_by_category, ensure_ascii=False, indent=2, sort_keys=True)}

Included byte counts by category:
{json.dumps(size_by_category, ensure_ascii=False, indent=2, sort_keys=True)}

Missing requested source/artifact files:
{json.dumps(missing, ensure_ascii=False, indent=2)}

Validation files outside the zip:
- `<archive>.zip.sha256`
- `<archive>.zip.sha256_check.txt`
- `<archive>.zip.payload_sha256_check.txt`
- `<archive>.zip.unzip_test.txt`
- `<archive>.zip.entry_parity.txt`
- `<archive>.zip.exclusion_check.txt`
- `<archive>.zip.summary.json`
"""
    (pack_dir / "PACK_README.md").write_text(readme, encoding="utf-8")
    (metadata_dir / "ARCHIVE_ENTRY_PREVIEW.txt").write_text("\n".join(included_archive[:500]) + "\n", encoding="utf-8")


def write_payload_hashes(pack_dir: Path) -> list[str]:
    filelist_path = pack_dir / "PAYLOAD_FILELIST.txt"
    hash_path = pack_dir / "PAYLOAD_SHA256SUMS.txt"
    hash_path.write_text("", encoding="utf-8")
    files = sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())
    filelist_path.write_text("\n".join(files) + "\n", encoding="utf-8")
    files = sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())
    hash_files = [name for name in files if name != "PAYLOAD_SHA256SUMS.txt"]
    hash_path.write_text(
        "\n".join(f"{sha256(pack_dir / name)}  {name}" for name in hash_files) + "\n",
        encoding="utf-8",
    )
    return sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())


def create_zip(pack_dir: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for src in sorted(p for p in pack_dir.rglob("*") if p.is_file()):
            arcname = src.relative_to(pack_dir).as_posix()
            archive.write(src, arcname)
            entries.append(arcname)
    return entries


def check_payload_hashes(zip_path: Path, sidecar_path: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="acl2_v106r_pack_verify_") as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmpdir)
        proc = subprocess.run(
            ["sha256sum", "-c", "PAYLOAD_SHA256SUMS.txt"],
            cwd=tmpdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        sidecar_path.write_text(proc.stdout, encoding="utf-8")
        return proc.returncode


def check_archive_sha(sha_sidecar: Path, sidecar_path: Path) -> int:
    proc = subprocess.run(
        ["sha256sum", "-c", sha_sidecar.name],
        cwd=PACK_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    sidecar_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode


def check_unzip(zip_path: Path, sidecar_path: Path) -> int:
    proc = subprocess.run(
        ["unzip", "-tq", str(zip_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    sidecar_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode


def check_entry_parity(payload_files: list[str], zip_entries: list[str], sidecar_path: Path) -> int:
    payload_set = set(payload_files)
    zip_set = set(zip_entries)
    data = {
        "payload_file_count": len(payload_files),
        "zip_entry_count": len(zip_entries),
        "missing_from_zip": sorted(payload_set - zip_set),
        "extra_in_zip": sorted(zip_set - payload_set),
    }
    data["entry_parity_ok"] = not data["missing_from_zip"] and not data["extra_in_zip"]
    sidecar_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if data["entry_parity_ok"] else 1


def check_exclusions(zip_entries: list[str], sidecar_path: Path) -> int:
    violations: list[str] = []
    for entry in zip_entries:
        path = Path(entry)
        if should_exclude_archive(path):
            violations.append(entry)
        if entry.startswith("results/") or entry.startswith("code_audit_pack/"):
            violations.append(entry)
        if "/raw_trace/" in entry or "/workspace/" in entry:
            violations.append(entry)
        if entry.endswith("targeted_gca_context_trace_rows.csv") or entry.endswith("targeted_trace_semantic_key_rows.csv"):
            violations.append(entry)
    data = {
        "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
        "excluded_names": sorted(EXCLUDED_NAMES),
        "violation_count": len(sorted(set(violations))),
        "violations": sorted(set(violations)),
        "exclusion_check_ok": not violations,
    }
    sidecar_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if data["exclusion_check_ok"] else 1


def build() -> dict[str, Any]:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = os.environ.get("V106R_PACK_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = PACK_ROOT / f"{PACK_PREFIX}_{timestamp}.zip"
    with tempfile.TemporaryDirectory(prefix=f"{PACK_PREFIX}_") as tmp:
        pack_dir = Path(tmp) / f"{PACK_PREFIX}_{timestamp}"
        pack_dir.mkdir(parents=True)
        manifest: list[dict[str, Any]] = []
        for category, src_rel, archive_rel in collect_sources():
            add_file(src_rel, pack_dir, manifest, category=category, archive_rel=archive_rel)
        write_manifest(pack_dir, manifest)
        write_metadata(pack_dir, manifest)
        payload_files = write_payload_hashes(pack_dir)
        zip_entries = create_zip(pack_dir, zip_path)

    sha_sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha_sidecar.write_text(f"{sha256(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    sha_check = zip_path.with_suffix(zip_path.suffix + ".sha256_check.txt")
    payload_check = zip_path.with_suffix(zip_path.suffix + ".payload_sha256_check.txt")
    unzip_test = zip_path.with_suffix(zip_path.suffix + ".unzip_test.txt")
    entry_parity = zip_path.with_suffix(zip_path.suffix + ".entry_parity.txt")
    exclusion_check = zip_path.with_suffix(zip_path.suffix + ".exclusion_check.txt")
    sha_rc = check_archive_sha(sha_sidecar, sha_check)
    payload_rc = check_payload_hashes(zip_path, payload_check)
    unzip_rc = check_unzip(zip_path, unzip_test)
    parity_rc = check_entry_parity(payload_files, zip_entries, entry_parity)
    exclusion_rc = check_exclusions(zip_entries, exclusion_check)
    summary = {
        "schema": "acl2_v106r_core_artifact_audit_pack_summary_v1",
        "archive": zip_path.relative_to(ROOT).as_posix(),
        "archive_size_bytes": zip_path.stat().st_size,
        "archive_sha256": sha256(zip_path),
        "payload_file_count": len(payload_files),
        "zip_entry_count": len(zip_entries),
        "archive_sha256_check_returncode": sha_rc,
        "payload_sha256_check_returncode": payload_rc,
        "unzip_test_returncode": unzip_rc,
        "entry_parity_returncode": parity_rc,
        "exclusion_check_returncode": exclusion_rc,
        "validation_ok": sha_rc == 0 and payload_rc == 0 and unzip_rc == 0 and parity_rc == 0 and exclusion_rc == 0,
        "sidecars": [
            sha_sidecar.relative_to(ROOT).as_posix(),
            sha_check.relative_to(ROOT).as_posix(),
            payload_check.relative_to(ROOT).as_posix(),
            unzip_test.relative_to(ROOT).as_posix(),
            entry_parity.relative_to(ROOT).as_posix(),
            exclusion_check.relative_to(ROOT).as_posix(),
        ],
        "observed_v106r_state": observed_v106r_state(manifest),
    }
    summary_path = zip_path.with_suffix(zip_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (PACK_ROOT / ".latest_acl2_v106r_pack_tag").write_text(zip_path.stem + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["validation_ok"]:
        raise SystemExit(1)
    return summary


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())

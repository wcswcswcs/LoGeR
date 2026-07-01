#!/usr/bin/env python3
"""Build a compact ACL2 v93 final No-Go code/artifact audit packet."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = REPO_ROOT / "code_audit_pack"
V92_ROOT = REPO_ROOT / "results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery"
V93_ROOT = REPO_ROOT / "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier"

TAG_PREFIX = "acl2_v93tf_semantic_object_identity_final_no_go_core_audit"

CORE_PATHS = [
    "tools/build_v93_code_audit_pack.py",
    "docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_ExperimentPlan.md",
    "docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_执行日志.md",
    "docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_实验结果复盘.md",
    "tools/v93_semantic_object_identity_utils.py",
    "tools/build_v93_v92_evidence_lock.py",
    "tools/build_v93_object_identity_row_join.py",
    "tools/audit_v93_object_identity_source_coverage.py",
    "tools/build_v93_object_topology_policy.py",
    "tools/audit_v93_object_topology_policy_specificity.py",
    "tools/run_v93_merge_gauge_boundary_trace_smoke.py",
    "tools/audit_v93_merge_gauge_trace_availability.py",
    "tools/audit_v93_merge_gauge_trace_hidden_fields.py",
    "tools/audit_v93_merge_gauge_carrier_alignment.py",
    "tools/audit_v93_merge_gauge_counterfactual_upper_bound.py",
    "tools/build_v93_swa_secondary_route_masks.py",
    "tools/run_v93_swa_secondary_route_audit.py",
    "tools/audit_v93_swa_secondary_carrier.py",
    "tools/build_v93_final_decision.py",
    "tools/build_v70_radio_sidecar_cache.py",
    "tools/v70_radio_sidecar_common.py",
    "tools/run_v68_phaseE_merge_multichunk.py",
    "results/kitti_preprocess/00/radio_sidecar_chunks_v93_labelled",
    "results/kitti_preprocess/01/radio_sidecar_chunks_v93_labelled",
    "results/kitti_preprocess/02/radio_sidecar_chunks_v93_labelled",
    "results/kitti_preprocess/05/radio_sidecar_chunks_v93_labelled",
    "results/kitti_preprocess/00/radio_sidecar_chunks_v93_labelled.build.log",
    "results/kitti_preprocess/01/radio_sidecar_chunks_v93_labelled.build.log",
    "results/kitti_preprocess/02/radio_sidecar_chunks_v93_labelled.build.log",
    "results/kitti_preprocess/05/radio_sidecar_chunks_v93_labelled.build.log",
    "docs/ACL2_v92TF_SemanticPolicyCarrier_MergeGaugeBoundaryDiscovery_ExperimentPlan.md",
    "docs/ACL2_v92TF_SemanticPolicyCarrier_MergeGaugeBoundaryDiscovery_执行日志.md",
    "docs/ACL2_v92TF_SemanticPolicyCarrier_MergeGaugeBoundaryDiscovery_实验结果复盘.md",
    "tools/v86_soft_latent_utils.py",
    "tools/v91_semantic_regime_utils.py",
    "tools/build_v91_external_mask_materialization.py",
    "tools/v92_semantic_policy_carrier_utils.py",
    "tools/build_v92_evidence_lock.py",
    "tools/build_v92_semantic_policy_row_bank.py",
    "tools/audit_v92_semantic_policy_row_bank.py",
    "tools/build_v92_boundary_trace_ledger.py",
    "tools/audit_v92_boundary_trace_availability.py",
    "tools/audit_v92_boundary_trace_hidden_fields.py",
    "tools/audit_v92_noop_trace_smoke.py",
    "tools/build_v92_swa_policy_route_masks.py",
    "tools/run_v92_swa_semantic_policy_route_audit.py",
    "tools/audit_v92_swa_semantic_policy_carrier.py",
    "tools/audit_v92_semantic_source_expansion_candidates.py",
    "tools/build_v92_radio_tracklet_sidecar_if_available.py",
    "tools/audit_v92_expanded_semantic_policy.py",
    "tools/build_v92_final_decision.py",
    "tools/run_v78_phase9_swa_cache_value_carryover.py",
    "loger/models/pi3.py",
    "run_pipeline_abc_v2.py",
]

EXPECTED_V93_PATHS = [
    "docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_执行日志.md",
    "docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_实验结果复盘.md",
    "tools/v93_semantic_object_identity_utils.py",
    "tools/build_v93_v92_evidence_lock.py",
    "tools/build_v93_object_identity_row_join.py",
    "tools/audit_v93_object_identity_source_coverage.py",
    "tools/build_v93_object_topology_policy.py",
    "tools/audit_v93_object_topology_policy_specificity.py",
    "tools/run_v93_merge_gauge_boundary_trace_smoke.py",
    "tools/audit_v93_merge_gauge_trace_availability.py",
    "tools/audit_v93_merge_gauge_trace_hidden_fields.py",
    "tools/audit_v93_merge_gauge_carrier_alignment.py",
    "tools/audit_v93_merge_gauge_counterfactual_upper_bound.py",
    "tools/build_v93_swa_secondary_route_masks.py",
    "tools/run_v93_swa_secondary_route_audit.py",
    "tools/audit_v93_swa_secondary_carrier.py",
    "tools/build_v93_final_decision.py",
    "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier",
    "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/report_final/final_decision.json",
    "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/phase5_merge_gauge_counterfactual_upper_bound/counterfactual_upper_bound_summary.json",
    "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/phase7_swa_secondary_carrier/carrier_audit/phase7_swa_secondary_summary.json",
]

LIGHT_EXTS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".patch",
    ".txt",
    ".yaml",
    ".yml",
}

EXCLUDED_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".mp4",
    ".mov",
    ".avi",
    ".zip",
    ".gz",
    ".tar",
}

EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    "code_audit_pack",
    "data",
    "checkpoints",
    "weights",
}


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as exc:
        return f"COMMAND_NOT_FOUND: {cmd[0]} ({exc})\n"
    return proc.stdout


def should_copy_file(path: Path) -> bool:
    rel_parts = set(path.resolve().relative_to(REPO_ROOT).parts)
    if rel_parts & EXCLUDED_PARTS:
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.suffix.lower() in LIGHT_EXTS


def copy_one(src_rel: str, packet_dir: Path, copied: list[str], missing: list[str]) -> None:
    src = REPO_ROOT / src_rel
    if not src.exists():
        missing.append(src_rel)
        return
    if src.is_file():
        if should_copy_file(src):
            dst = packet_dir / src_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(src_rel)
        return
    for path in sorted(src.rglob("*")):
        if path.is_file() and should_copy_file(path):
            rel = repo_rel(path)
            dst = packet_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied.append(rel)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit packet should expose malformed evidence.
        return {"read_error": str(exc)}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def packet_files(packet_dir: Path) -> list[Path]:
    return sorted(path for path in packet_dir.rglob("*") if path.is_file())


def write_filelists(packet_dir: Path) -> dict[str, int]:
    files = packet_files(packet_dir)
    filelist = packet_dir / "PAYLOAD_FILELIST.txt"
    hashes = packet_dir / "PAYLOAD_SHA256SUMS.txt"
    write_text(filelist, "\n".join(path.relative_to(packet_dir).as_posix() for path in files) + "\n")
    lines = []
    for path in packet_files(packet_dir):
        rel = path.relative_to(packet_dir)
        if rel.as_posix() in {"PAYLOAD_FILELIST.txt", "PAYLOAD_SHA256SUMS.txt"}:
            continue
        lines.append(f"{sha256_file(path)}  {rel.as_posix()}")
    write_text(hashes, "\n".join(lines) + "\n")
    return {"payload_file_count": len(packet_files(packet_dir)), "payload_hash_rows": len(lines)}


def write_git_context(packet_dir: Path) -> None:
    write_text(packet_dir / "GIT_STATUS_SHORT.txt", run_text(["git", "status", "--short"]))
    diff_paths = [path for path in CORE_PATHS if (REPO_ROOT / path).exists()]
    diff = run_text(["git", "diff", "--", *diff_paths]) if diff_paths else ""
    write_text(packet_dir / "SCOPED_GIT_DIFF.patch", diff)


def write_compile_check(packet_dir: Path) -> None:
    py_paths = [path for path in CORE_PATHS if path.endswith(".py") and (REPO_ROOT / path).is_file()]
    if not py_paths:
        write_text(packet_dir / "PY_COMPILE_CHECK.txt", "no python files selected\n")
        return
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", *py_paths],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    write_text(
        packet_dir / "PY_COMPILE_CHECK.txt",
        f"$ {sys.executable} -m py_compile {' '.join(py_paths)}\n"
        f"returncode={proc.returncode}\n\n{proc.stdout}",
    )


def write_readme(packet_dir: Path, tag: str, missing: list[str], v93_missing: list[str]) -> None:
    v92_final_path = V92_ROOT / "report_final/final_decision.json"
    v92_final = read_json(v92_final_path) if v92_final_path.exists() else {}
    v93_final_path = V93_ROOT / "report_final/final_decision.json"
    v93_final = read_json(v93_final_path) if v93_final_path.exists() else {}
    readme = f"""# {tag}

ACL2 v93 final No-Go core code and audit artifact packet.

## Scope

This packet includes the v93 plan, execution log, retrospective log, v93 tools,
final No-Go decision artifacts, and the v92 predecessor evidence that v93 locks.
It intentionally excludes raw data, tensor caches, checkpoints, and media.

## Current Verified Status

- v93 final status: `{v93_final.get("final_status", "missing")}`
- v93 blocker: `{v93_final.get("blocker", "missing")}`
- v93 runtime action allowed: `{v93_final.get("runtime_action_allowed", "missing")}`
- v93 TTT allowed: `{v93_final.get("ttt_allowed", "missing")}`
- v93 result artifacts found: `{str(V93_ROOT.exists()).lower()}`
- v92 final status: `{v92_final.get("final_status", "missing")}`
- v92 blocker: `{v92_final.get("blocker", "missing")}`
- v93 final decision path: `results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/report_final/final_decision.json`
- v92 final decision path: `results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery/report_final/final_decision.json`

## Start Here

- v93 plan: `docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_ExperimentPlan.md`
- v93 execution log: `docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_执行日志.md`
- v93 retrospective: `docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_实验结果复盘.md`
- v93 final decision: `results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/report_final/final_decision.json`
- v92 execution log: `docs/ACL2_v92TF_SemanticPolicyCarrier_MergeGaugeBoundaryDiscovery_执行日志.md`
- v92 retrospective: `docs/ACL2_v92TF_SemanticPolicyCarrier_MergeGaugeBoundaryDiscovery_实验结果复盘.md`
- v92 final decision: `results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery/report_final/final_decision.json`

## Validation Sidecars

- `PAYLOAD_FILELIST.txt`
- `PAYLOAD_SHA256SUMS.txt`
- `PY_COMPILE_CHECK.txt`
- external sidecars beside the zip: `.zip.sha256`, `.unzip_test.txt`,
  `.payload_sha256_check.txt`, `.entry_diff.txt`, `.build_summary.json`

## Missing v93 Expected Paths

```text
{chr(10).join(v93_missing) if v93_missing else "none"}
```

## Missing Selected Core Paths

```text
{chr(10).join(missing) if missing else "none"}
```

## Exclusions

Raw data, checkpoints, tensor caches (`.pt/.npy/.npz`), media, old audit zips,
and the local `code_audit_pack` directory are excluded.
"""
    write_text(packet_dir / "PACKET_README.md", readme)


def write_manifest(packet_dir: Path, tag: str, copied: list[str], missing: list[str], v93_missing: list[str]) -> None:
    v92_final_path = V92_ROOT / "report_final/final_decision.json"
    v92_final = read_json(v92_final_path) if v92_final_path.exists() else {}
    v93_final_path = V93_ROOT / "report_final/final_decision.json"
    v93_final = read_json(v93_final_path) if v93_final_path.exists() else {}
    manifest = {
        "tag": tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "ACL2 v93 final No-Go core code and audit artifacts with v92 predecessor evidence",
        "status": "final_no_go_packet",
        "v93_objective_complete": False,
        "v93_final_status": v93_final.get("final_status"),
        "v93_blocker": v93_final.get("blocker"),
        "v93_runtime_action_allowed": v93_final.get("runtime_action_allowed"),
        "v93_ttt_allowed": v93_final.get("ttt_allowed"),
        "v93_execution_artifacts_found": V93_ROOT.exists(),
        "v93_missing_expected_paths": v93_missing,
        "v92_final_status": v92_final.get("final_status"),
        "v92_blocker": v92_final.get("blocker"),
        "copied_path_count_before_manifest": len(copied),
        "missing_selected_core_paths": missing,
        "key_artifacts": [
            "docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_ExperimentPlan.md",
            "docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_执行日志.md",
            "docs/ACL2_v93TF_SemanticObjectIdentity_MergeGaugeBoundaryCarrier_实验结果复盘.md",
            "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/report_final/final_decision.json",
            "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/phase3_merge_gauge_trace_audit/phase3_trace_availability_summary.json",
            "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/phase7_swa_secondary_carrier/carrier_audit/phase7_swa_secondary_summary.json",
            "results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery/report_final/final_decision.json",
            "results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery/phase1_semantic_policy_row_bank/phase1_gate_summary.json",
            "results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery/phase2_boundary_trace_ledger/phase2_gate_summary.json",
            "results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery/phase7_data_source_expansion/expanded_semantic_policy_summary.json",
        ],
        "exclusions": [
            "raw data",
            "*.pt/*.npy/*.npz tensor artifacts",
            "checkpoints/weights",
            "media files",
            "old code_audit_pack archives",
            "__pycache__",
        ],
    }
    write_text(packet_dir / "PACKET_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def zip_packet(packet_dir: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in packet_files(packet_dir):
            arcname = Path(packet_dir.name) / path.relative_to(packet_dir)
            zf.write(path, arcname.as_posix())
            entries.append(arcname.as_posix())
    return sorted(entries)


def strip_top(entry: str) -> str:
    parts = Path(entry).parts
    if len(parts) <= 1:
        return ""
    return Path(*parts[1:]).as_posix()


def validate_zip(zip_path: Path, packet_dir: Path, tag: str, entries: list[str]) -> dict[str, Any]:
    unzip_proc = subprocess.run(
        ["unzip", "-t", zip_path.name],
        cwd=PACK_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    unzip_test_path = PACK_ROOT / f"{tag}.unzip_test.txt"
    write_text(unzip_test_path, f"$ unzip -t {zip_path.name}\nreturncode={unzip_proc.returncode}\n\n{unzip_proc.stdout}")

    payload_proc = subprocess.run(
        ["sha256sum", "-c", "PAYLOAD_SHA256SUMS.txt"],
        cwd=packet_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    payload_check_path = PACK_ROOT / f"{tag}.payload_sha256_check.txt"
    write_text(payload_check_path, f"$ sha256sum -c PAYLOAD_SHA256SUMS.txt\nreturncode={payload_proc.returncode}\n\n{payload_proc.stdout}")

    payload = [path.relative_to(packet_dir).as_posix() for path in packet_files(packet_dir)]
    archive_payload = sorted(strip_top(entry) for entry in entries if strip_top(entry))
    missing_in_zip = sorted(set(payload) - set(archive_payload))
    extra_in_zip = sorted(set(archive_payload) - set(payload))
    entry_diff_path = PACK_ROOT / f"{tag}.entry_diff.txt"
    write_text(
        entry_diff_path,
        "missing_in_zip:\n"
        + ("\n".join(missing_in_zip) if missing_in_zip else "none")
        + "\n\nextra_in_zip:\n"
        + ("\n".join(extra_in_zip) if extra_in_zip else "none")
        + "\n",
    )

    sha_path = PACK_ROOT / f"{tag}.zip.sha256"
    write_text(sha_path, f"{sha256_file(zip_path)}  {zip_path.name}\n")

    return {
        "zip_path": zip_path.as_posix(),
        "zip_sha256": sha256_file(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_entry_count": len(entries),
        "unzip_test_returncode": unzip_proc.returncode,
        "payload_sha256_check_returncode": payload_proc.returncode,
        "entry_diff_missing_count": len(missing_in_zip),
        "entry_diff_extra_count": len(extra_in_zip),
        "sidecars": [
            unzip_test_path.as_posix(),
            payload_check_path.as_posix(),
            entry_diff_path.as_posix(),
            sha_path.as_posix(),
        ],
    }


def main() -> None:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    tag = f"{TAG_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    packet_dir = PACK_ROOT / tag
    if packet_dir.exists():
        raise FileExistsError(packet_dir)
    packet_dir.mkdir(parents=True)

    copied: list[str] = []
    missing: list[str] = []
    for rel in CORE_PATHS:
        copy_one(rel, packet_dir, copied, missing)
    if V92_ROOT.exists():
        copy_one(repo_rel(V92_ROOT), packet_dir, copied, missing)
    if V93_ROOT.exists():
        copy_one(repo_rel(V93_ROOT), packet_dir, copied, missing)

    v93_missing = [rel for rel in EXPECTED_V93_PATHS if not (REPO_ROOT / rel).exists()]

    write_git_context(packet_dir)
    write_compile_check(packet_dir)
    write_readme(packet_dir, tag, missing, v93_missing)
    write_manifest(packet_dir, tag, copied, missing, v93_missing)
    payload_counts = write_filelists(packet_dir)

    zip_path = PACK_ROOT / f"{tag}.zip"
    entries = zip_packet(packet_dir, zip_path)
    validation = validate_zip(zip_path, packet_dir, tag, entries)

    summary = {
        "tag": tag,
        "packet_dir": packet_dir.as_posix(),
        "copied_path_count_before_manifest": len(copied),
        "missing_selected_core_paths": missing,
        "v93_missing_expected_paths": v93_missing,
        **payload_counts,
        **validation,
    }
    summary_path = PACK_ROOT / f"{tag}.build_summary.json"
    write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_text(PACK_ROOT / ".latest_acl2_v93_pack_tag", tag + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

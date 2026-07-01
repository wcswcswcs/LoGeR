#!/usr/bin/env python3
"""Build a compact Stream4D v85 code/artifact audit packet."""

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
AUDIT_ROOT = REPO_ROOT / "Stream3D/outputs/audit"

TAG_PREFIX = "stream4d_v85_persistent_affinity_field_l2h_current_core_audit"

DOC_PATHS = [
    "docs/stream4d_v85_persistent_affinity_field_l2h_experiment_plan.md",
    "docs/stream4d_v85_执行日志.md",
    "docs/stream4d_v85_实验结果复盘.md",
    "docs/stream4d_v84_l2h_strong_materialization_experiment_plan.md",
    "docs/stream4d_v84_执行日志.md",
    "docs/stream4d_v84_实验结果复盘.md",
    "docs/stream4d_v83_local2history_evidence_ledger_experiment_plan.md",
    "docs/stream4d_v83_执行日志.md",
    "docs/stream4d_v83_实验结果复盘.md",
    "docs/stream4d_v82_revised_causal_tracklet_memory_plan.md",
    "docs/stream4d_v82_执行日志.md",
    "docs/stream4d_v82_实验结果复盘.md",
    "docs/stream4d_v80_cmap_af_l2h_revised_critical_plan.md",
    "docs/stream4d_v80_执行日志.md",
    "docs/stream4d_v80_实验结果复盘.md",
    "docs/stream4d_v79_cmap_af_l2h_experiment_plan.md",
    "docs/stream4d_v79_执行日志.md",
    "docs/stream4d_v79_实验结果复盘.md",
]

CODE_PATHS = [
    "tools/build_stream4d_v85_code_audit_pack.py",
    "Stream3D/tools/run_v85_persistent_affinity_field_l2h.py",
    "Stream3D/tools/run_v84_l2h_strong_materialization.py",
    "Stream3D/tools/run_v83_local2history_evidence_ledger.py",
    "Stream3D/tools/run_v82_revised_causal_tracklet_memory.py",
    "Stream3D/tools/run_v81_history_anchored_cmap_af_l2h_pipeline.py",
    "Stream3D/tools/run_v80_cmap_af_l2h_pipeline.py",
    "Stream3D/tools/run_v79_cmap_af_l2h_pipeline.py",
]

V85_EXPECTED_DIRS = [
    "Stream3D/outputs/audit/v85_phase0_fact_lock",
    "Stream3D/outputs/audit/v85_phase1_local_affinity_feature",
    "Stream3D/outputs/audit/v85_phase2_local_clustering",
    "Stream3D/outputs/audit/v85_phase3_slot_descriptor",
    "Stream3D/outputs/audit/v85_phase4_tracklet_descriptor",
    "Stream3D/outputs/audit/v85_phase5_history_object_feature",
    "Stream3D/outputs/audit/v85_phase6_history_query",
    "Stream3D/outputs/audit/v85_phase7_renderable_materializer",
    "Stream3D/outputs/audit/v85_phase8_strong_controls",
    "Stream3D/outputs/audit/v85_phase9_holdout",
    "Stream3D/outputs/audit/v85_phase10_casebook",
    "Stream3D/outputs/audit/v85_config",
]

V85_DECLARED_DATA_MANIFESTS = [
    "Stream3D/data/prediction/v85_paf_l2h_frame_mask_diag_class_agnostic/config_manifest.json",
    "Stream3D/data/TMP/v85_paf_l2h_frame_mask_diag/config_manifest.json",
]

PREDECESSOR_ARTIFACT_DIRS = [
    "Stream3D/outputs/audit/v84_config",
    "Stream3D/outputs/audit/v84_phase0_fact_lock",
    "Stream3D/outputs/audit/v84_phase1_graph_build",
    "Stream3D/outputs/audit/v84_phase2_id_only_stitching",
    "Stream3D/outputs/audit/v84_phase3_cross_chunk_materializer",
    "Stream3D/outputs/audit/v84_phase4_conflict_split",
    "Stream3D/outputs/audit/v84_phase5_fragmentation_merge",
    "Stream3D/outputs/audit/v84_phase6_controls",
    "Stream3D/outputs/audit/v84_phase7_holdout_input_generation",
    "Stream3D/outputs/audit/v84_phase8_frozen_holdout",
    "Stream3D/outputs/audit/v84_phase9_casebook",
    "Stream3D/outputs/audit/v84_holdout_replay_v82_phase10_casebook",
    "Stream3D/outputs/audit/v84_holdout_replay_v82_phase7_final_local",
    "Stream3D/outputs/audit/v83_config_repair11_structural_edges",
    "Stream3D/outputs/audit/v83_phase2_evidence_ledger_repair8_antihijack_extreme_bound",
    "Stream3D/outputs/audit/v83_phase3_state_machine_repair10_safe_topk_coverage",
    "Stream3D/outputs/audit/v83_phase4_conflict_memory_repair11_structural_edges",
    "Stream3D/outputs/audit/v83_phase5_weak_l2h_repair10_safe_topk_coverage",
    "Stream3D/outputs/audit/v83_phase6_controls_repair10_safe_topk_coverage",
    "Stream3D/outputs/audit/v83_phase7_strong_history_repair11_structural_edges",
    "Stream3D/outputs/audit/v83_phase8_frozen_eval_repair11_structural_edges",
    "Stream3D/outputs/audit/v83_phase9_casebook_repair11_structural_edges",
    "Stream3D/outputs/audit/v82_phase1_local_b0",
    "Stream3D/outputs/audit/v80_phase1_streaming_affinity_features_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard",
    "Stream3D/outputs/audit/v80_phase2_signed_affinity_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard",
    "Stream3D/outputs/audit/v80_phase6_control_audit_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard",
    "Stream3D/outputs/audit/v79_phase1_affinity_features",
    "Stream3D/outputs/audit/v79_phase2_neighbor_graph",
]

LIGHT_EXTS = {".csv", ".json", ".jsonl", ".log", ".md", ".patch", ".py", ".txt", ".yaml", ".yml"}
EXCLUDED_SUFFIXES = {".avi", ".bin", ".ckpt", ".gz", ".jpg", ".jpeg", ".mov", ".mp4", ".npy", ".npz", ".pickle", ".pkl", ".png", ".pt", ".pth", ".tar", ".zip"}
EXCLUDED_PARTS = {".git", "__pycache__", "code_audit_pack", "checkpoints", "data", "weights"}
MAX_FILE_BYTES = 100 * 1024 * 1024


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit packets should expose malformed evidence.
        return {"read_error": str(exc)}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_text(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError as exc:
        return f"$ {' '.join(cmd)}\nCOMMAND_NOT_FOUND: {exc}\n"
    return f"$ {' '.join(cmd)}\nreturncode={proc.returncode}\n\n{proc.stdout}"


def should_copy_file(path: Path) -> tuple[bool, str]:
    rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    if rel in V85_DECLARED_DATA_MANIFESTS:
        return True, ""
    rel_parts = set(path.resolve().relative_to(REPO_ROOT).parts)
    if rel_parts & EXCLUDED_PARTS:
        return False, "excluded_path_part"
    suffix = path.suffix.lower()
    if suffix in EXCLUDED_SUFFIXES:
        return False, "excluded_suffix"
    if suffix not in LIGHT_EXTS:
        return False, "non_light_extension"
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        return False, f"size_gt_{MAX_FILE_BYTES}"
    return True, ""


def copy_file(src: Path, packet_dir: Path, copied: list[str], excluded: list[dict[str, Any]]) -> None:
    allowed, reason = should_copy_file(src)
    rel = repo_rel(src)
    if not allowed:
        excluded.append({"path": rel, "reason": reason, "size_bytes": src.stat().st_size})
        return
    dst = packet_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(rel)


def copy_path(src_rel: str, packet_dir: Path, copied: list[str], missing: list[str], excluded: list[dict[str, Any]]) -> None:
    src = REPO_ROOT / src_rel
    if not src.exists():
        missing.append(src_rel)
        return
    if src.is_file():
        copy_file(src, packet_dir, copied, excluded)
        return
    for path in sorted(src.rglob("*")):
        if path.is_file():
            copy_file(path, packet_dir, copied, excluded)


def packet_files(packet_dir: Path) -> list[Path]:
    return sorted(path for path in packet_dir.rglob("*") if path.is_file())


def write_git_context(packet_dir: Path) -> None:
    write_text(packet_dir / "GIT_STATUS_SHORT.txt", run_text(["git", "status", "--short"]))
    diff_paths = [p for p in [*DOC_PATHS, *CODE_PATHS] if (REPO_ROOT / p).exists()]
    write_text(packet_dir / "SCOPED_GIT_DIFF.patch", run_text(["git", "diff", "--", *diff_paths]) if diff_paths else "")


def write_scope(packet_dir: Path, copied: list[str], missing: list[str], excluded: list[dict[str, Any]]) -> None:
    v85_present = [p for p in V85_EXPECTED_DIRS if (REPO_ROOT / p).exists()]
    v85_missing = [p for p in V85_EXPECTED_DIRS if not (REPO_ROOT / p).exists()]
    artifact_note = (
        "- note: all expected v85 output dirs were present at package build time."
        if not v85_missing
        else "- note: missing v85 output dirs mean no formal v85 phase artifact was present at package build time; this packet does not fabricate those outputs."
    )
    lines = [
        "# Stream4D v85 Audit Scope",
        "",
        f"generated_at_utc: {datetime.utcnow().isoformat(timespec='seconds')}Z",
        "",
        "Purpose: package the current Stream4D v85 Persistent Affinity Field L2H core code, logs, and auditable predecessor evidence.",
        "",
        "Current v85 artifact status:",
        f"- present_v85_output_dirs: {len(v85_present)}",
        f"- missing_v85_output_dirs: {len(v85_missing)}",
        artifact_note,
        "",
        "Included scope:",
        "- v85 plan/logs/runner and this pack builder",
        "- v84 strong materialization closure and holdout boundary artifacts",
        "- v83 repair11/safe-topk L2H evidence ledger artifacts used by v85",
        "- v82 local slot/descriptor replay artifacts",
        "- v80/v79 affinity feature and neighbor graph predecessor artifacts used by the v85 runner",
        "- v85 diagnostic npz/TMP config manifests only; diagnostic .npz/.npy payloads remain excluded",
        "",
        "Excluded by rule:",
        "- caches, checkpoints, raw data, weights, nested code_audit_pack contents",
        "- binary model/data/media/archive outputs such as .pt/.pth/.npz/.npy/.png/.mp4/.zip",
        "",
        f"copied_file_count: {len(copied)}",
        f"missing_declared_path_count: {len(missing)}",
        f"excluded_file_count: {len(excluded)}",
        "",
    ]
    write_text(packet_dir / "AUDIT_SCOPE.md", "\n".join(lines))
    write_text(packet_dir / "MISSING_DECLARED_PATHS.txt", "\n".join(missing) + ("\n" if missing else ""))
    write_text(packet_dir / "EXCLUDED_FILES.json", json.dumps(excluded, indent=2, sort_keys=True) + "\n")


def write_phase_snapshot(packet_dir: Path) -> None:
    summaries = {
        "v85_final": AUDIT_ROOT / "v85_phase10_casebook/final_decision.json",
        "v85_phase6": AUDIT_ROOT / "v85_phase6_history_query/q_summary.json",
        "v85_phase7": AUDIT_ROOT / "v85_phase7_renderable_materializer/materializer_summary.json",
        "v84_final": AUDIT_ROOT / "v84_phase9_casebook/final_decision.json",
        "v84_holdout": AUDIT_ROOT / "v84_phase8_frozen_holdout/summary.json",
        "v83_phase6": AUDIT_ROOT / "v83_phase6_controls_repair10_safe_topk_coverage/summary.json",
        "v83_phase7": AUDIT_ROOT / "v83_phase7_strong_history_repair11_structural_edges/summary.json",
    }
    payload = {}
    for key, path in summaries.items():
        payload[key] = {"path": repo_rel(path), "exists": path.exists(), "content": read_json(path) if path.exists() else None}
    write_text(packet_dir / "PHASE_DECISION_SNAPSHOT.json", json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_filelists(packet_dir: Path) -> dict[str, int]:
    lines = []
    for path in packet_files(packet_dir):
        rel = path.relative_to(packet_dir).as_posix()
        if rel in {"PAYLOAD_FILELIST.txt", "PAYLOAD_SHA256SUMS.txt"}:
            continue
        lines.append(f"{sha256_file(path)}  {rel}")
    write_text(packet_dir / "PAYLOAD_SHA256SUMS.txt", "\n".join(lines) + "\n")
    files = [path.relative_to(packet_dir).as_posix() for path in packet_files(packet_dir)]
    write_text(packet_dir / "PAYLOAD_FILELIST.txt", "\n".join(files) + "\n")
    return {"payload_file_count": len(files), "payload_hash_rows": len(lines)}


def zip_packet(packet_dir: Path, zip_path: Path) -> list[str]:
    entries = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in packet_files(packet_dir):
            rel = path.relative_to(packet_dir).as_posix()
            arc = f"{packet_dir.name}/{rel}"
            zf.write(path, arc)
            entries.append(arc)
    return entries


def validate_payload(packet_dir: Path, zip_path: Path, entries: list[str], tag: str) -> dict[str, Any]:
    actual = sorted(zipfile.ZipFile(zip_path).namelist())
    expected = sorted(entries)
    diff = []
    if expected != actual:
        diff.extend([f"missing_in_zip {x}" for x in sorted(set(expected) - set(actual))])
        diff.extend([f"unexpected_in_zip {x}" for x in sorted(set(actual) - set(expected))])
    hash_lines = (packet_dir / "PAYLOAD_SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    hash_status = []
    ok_hashes = 0
    for line in hash_lines:
        digest, rel = line.split("  ", 1)
        path = packet_dir / rel
        if path.exists() and sha256_file(path) == digest:
            ok_hashes += 1
            hash_status.append(f"OK  {rel}")
        else:
            hash_status.append(f"FAIL  {rel}")
    test_zip = zipfile.ZipFile(zip_path).testzip()
    zip_sha = sha256_file(zip_path)
    side = PACK_ROOT / tag
    write_text(Path(f"{side}.zip.sha256"), f"{zip_sha}  {zip_path.name}\n")
    write_text(Path(f"{side}.zip_entries.txt"), "\n".join(actual) + "\n")
    write_text(Path(f"{side}.entry_diff.txt"), "\n".join(diff) + ("\n" if diff else ""))
    write_text(Path(f"{side}.payload_sha256_check.txt"), "\n".join(hash_status) + "\n")
    write_text(Path(f"{side}.unzip_test.txt"), "OK\n" if test_zip is None else f"FAIL {test_zip}\n")
    summary = {
        "tag": tag,
        "zip_path": repo_rel(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": zip_sha,
        "zip_entry_count": len(actual),
        "entry_parity_ok": not diff,
        "payload_hash_ok_count": ok_hashes,
        "payload_hash_total": len(hash_lines),
        "payload_hashes_ok": ok_hashes == len(hash_lines),
        "zip_test_ok": test_zip is None,
    }
    write_text(Path(f"{side}.validation_summary.txt"), "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n")
    return summary


def build() -> dict[str, Any]:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    tag = f"{TAG_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    packet_dir = PACK_ROOT / tag
    zip_path = PACK_ROOT / f"{tag}.zip"
    if packet_dir.exists():
        shutil.rmtree(packet_dir)
    packet_dir.mkdir(parents=True)

    copied: list[str] = []
    missing: list[str] = []
    excluded: list[dict[str, Any]] = []
    for src in [*DOC_PATHS, *CODE_PATHS, *V85_EXPECTED_DIRS, *V85_DECLARED_DATA_MANIFESTS, *PREDECESSOR_ARTIFACT_DIRS]:
        copy_path(src, packet_dir, copied, missing, excluded)

    write_git_context(packet_dir)
    write_phase_snapshot(packet_dir)
    write_scope(packet_dir, copied, missing, excluded)
    counts = write_filelists(packet_dir)
    entries = zip_packet(packet_dir, zip_path)
    validation = validate_payload(packet_dir, zip_path, entries, tag)
    build_summary = {
        "schema": "stream4d_v85_code_audit_pack_v1",
        "tag": tag,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "pack_root": repo_rel(PACK_ROOT),
        "packet_dir": repo_rel(packet_dir),
        "zip_path": repo_rel(zip_path),
        "copied_file_count_before_metadata": len(copied),
        "missing_declared_paths": missing,
        "excluded_file_count": len(excluded),
        **counts,
        **validation,
    }
    write_text(PACK_ROOT / f"{tag}.build_summary.json", json.dumps(build_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    write_text(PACK_ROOT / ".latest_stream4d_v85_pack_tag", tag + "\n")
    return build_summary


def main() -> None:
    summary = build()
    json.dump(summary, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

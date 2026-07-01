#!/usr/bin/env python3
"""Build a compact Stream4D v96 code/artifact audit packet."""

from __future__ import annotations

import argparse
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


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = REPO_ROOT / "code_audit_pack"
AUDIT_ROOT = REPO_ROOT / "Stream3D/outputs/audit"
TAG_PREFIX = "stream4d_v96_d4rt_micro_primitive_core_audit"
MAX_FILE_BYTES = 32 * 1024 * 1024

DOC_PATHS = [
    "docs/stream4d_v96_d4rt_micro_primitive_affinity_field_plan.md",
    "docs/stream4d_v96_执行日志.md",
    "docs/stream4d_v96_实验结果复盘.md",
]

CODE_PATHS = [
    "tools/build_stream4d_v96_code_audit_pack.py",
    "Stream3D/tools/build_v96_phase0_fact_lock.py",
    "Stream3D/tools/build_v96_phase1_query_planner.py",
    "Stream3D/tools/build_v96_phase2_d4rt_micro_tracks.py",
    "Stream3D/tools/build_v96_phase2_segment_aggregate.py",
    "Stream3D/tools/build_v96_phase3_triton_incidence.py",
    "Stream3D/tools/build_v96_phase4_affinity_features.py",
    "Stream3D/tools/build_v96_phase5_object_birth.py",
    "Stream3D/tools/build_v96_phase5_semantic_spatial_d4rt_merge_repair.py",
    "Stream3D/tools/build_v96_phase5_object_specific_core_repair.py",
    "Stream3D/tools/build_v96_phase6_render_snap.py",
    "Stream3D/tools/build_v96_phase7_control_summary.py",
    "Stream3D/tools/build_v96_phase7_materialize_no_temporal_control.py",
    "Stream3D/tools/build_v96_phase7_materialize_required_controls.py",
    "Stream3D/tools/build_v96_phase7_materialize_shuffled_d4rt_control.py",
    "Stream3D/tools/build_v96_phase9_error_decomposition.py",
    "Stream3D/tools/build_v96_phase10_dev_decision.py",
    # Supporting evaluator/provenance scripts.
    "Stream3D/tools/run_v65_scene_multiview_ap.py",
    "Stream3D/tools/build_v95_phase1_physical_source_registry.py",
    "Stream3D/tools/build_v91_radio_mask_features.py",
    "Stream3D/tools/merge_v91_radio_mask_feature_stores.py",
]

ARTIFACT_DIRS = [
    # Current final chain.
    "Stream3D/outputs/audit/v96_phase5_object_specific_core_k512_s010_h1",
    "Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k512_s010_h1_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase7_control_summary_object_core_k512_s010_h1_R32_all_controls",
    "Stream3D/outputs/audit/v96_phase9_error_decomposition_object_core_k512_s010_h1_R32",
    "Stream3D/outputs/audit/v96_phase10_dev_decision_object_core_k512_s010_h1_R32_all_controls",
    # Full object-specific-core family and ranking probes.
    "Stream3D/outputs/audit/v96_phase5_object_specific_core_k64_s030_h2",
    "Stream3D/outputs/audit/v96_phase5_object_specific_core_k128_s020_h2",
    "Stream3D/outputs/audit/v96_phase5_object_specific_core_k256_s015_h1",
    "Stream3D/outputs/audit/v96_phase5_object_specific_core_k1024_s005_h1",
    "Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k64_s030_h2_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k128_s020_h2_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k256_s015_h1_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k256_s015_h1_frame_count_x_masklet_wta005",
    "Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k256_s015_h1_rank_frame_count",
    "Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k256_s015_h1_rank_frame_count_x_best_support_iou",
    "Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k256_s015_h1_rank_masklet_score",
    "Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k256_s015_h1_rank_qid_frame_support",
    "Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k1024_s005_h1_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase7_control_summary_object_core_k256_s015_h1_R32_all_controls",
    "Stream3D/outputs/audit/v96_phase9_error_decomposition_object_core_k256_s015_h1_R32",
    "Stream3D/outputs/audit/v96_phase10_dev_decision_object_core_k256_s015_h1_R32_all_controls",
    # Previous best and repair-family evidence referenced by the logs.
    "Stream3D/outputs/audit/v96_phase5_object_birth_w0020_segmented_r4_D3_repair5_overlap090_sceneoffset",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_sceneoffset_fullscope_C_best_support_snap_wta_objnms_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase7_control_summary_w0020_segmented_r4_D3_fullscope_best_support_snap_wta_objnms_all_controls",
    "Stream3D/outputs/audit/v96_phase9_error_decomposition_w0020_segmented_r4_D3_fullscope_C_best_support_snap_wta_objnms_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase10_dev_decision_fullscope_best_support_snap_wta_objnms_all_controls",
    "Stream3D/outputs/audit/v96_phase5_object_birth_sem_sp_d4rt_merge_m2_j002_d018_a4_g35_v2",
    "Stream3D/outputs/audit/v96_phase5_object_birth_sem_sp_d4rt_merge_m4_j005_d012_a3_g20",
    "Stream3D/outputs/audit/v96_phase5_object_birth_sem_sp_d4rt_merge_m6_j008_d010_a2_g15",
    "Stream3D/outputs/audit/v96_phase6_render_snap_sem_sp_d4rt_merge_m2_j002_d018_a4_g35_v2_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase6_render_snap_sem_sp_d4rt_merge_m4_j005_d012_a3_g20_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase6_render_snap_sem_sp_d4rt_merge_m6_j008_d010_a2_g15_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_repair1_overlapC_best_support_wta_objnms_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_repair2_overlap050_best_support_wta_objnms_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_repair3_overlap070_best_support_wta_objnms_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_repair4_overlap050_shared8_best_support_wta_objnms_frame_count_x_masklet",
    # Required controls.
    "Stream3D/outputs/audit/v96_phase5_control_C0_semantic_only_sceneoffset",
    "Stream3D/outputs/audit/v96_phase5_control_C1_mask_area_risk_sceneoffset",
    "Stream3D/outputs/audit/v96_phase5_control_C2_shuffled_D4RT_sceneoffset",
    "Stream3D/outputs/audit/v96_phase5_control_C3_no_temporal_sceneoffset",
    "Stream3D/outputs/audit/v96_phase5_control_C4_random_micro_primitives_sceneoffset_dedup",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_sceneoffset_fullscope_A",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_C0_semantic_only_masklet_score",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_C1_mask_area_risk_masklet_score",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_C2_shuffled_D4RT_frame_count_x_masklet",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_C3_no_temporal_masklet_score",
    "Stream3D/outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_C4_random_micro_primitives_dedup_frame_count_x_masklet",
    # Upstream compact provenance.
    "Stream3D/outputs/audit/v96_phase0_fact_lock",
    "Stream3D/outputs/audit/v96_phase1_query_planner",
    "Stream3D/outputs/audit/v96_phase2_d4rt_micro_tracks_w0020_segmented_r4_aggregate",
    "Stream3D/outputs/audit/v96_phase3_triton_incidence_w0020_segmented_r4_D3_repair1",
    "Stream3D/outputs/audit/v96_phase4_affinity_features_w0020_segmented_r4_D3",
    "Stream3D/outputs/audit/v95_phase1_physical_source_registry",
    "Stream3D/outputs/audit/v81_dino_feature_json_scene0011_scene0050",
]

LIGHT_EXTS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".patch",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}

EXCLUDED_SUFFIXES = {
    ".avi",
    ".bin",
    ".ckpt",
    ".gz",
    ".jpg",
    ".jpeg",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".tar",
    ".zip",
}

EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    "cache",
    "checkpoints",
    "code_audit_pack",
    "data",
    "generated_masks",
    "tmp",
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


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit packets should expose malformed evidence.
        return {"read_error": str(exc)}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        return f"$ {' '.join(cmd)}\nCOMMAND_NOT_FOUND: {exc}\n"
    return f"$ {' '.join(cmd)}\nreturncode={proc.returncode}\n\n{proc.stdout}"


def should_copy_file(path: Path) -> tuple[bool, str]:
    rel_path = path.resolve().relative_to(REPO_ROOT)
    if set(rel_path.parts) & EXCLUDED_PARTS:
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


def copy_file(
    src: Path,
    payload_dir: Path,
    copied: list[str],
    copied_seen: set[str],
    excluded: list[dict[str, Any]],
) -> None:
    rel = repo_rel(src)
    allowed, reason = should_copy_file(src)
    if not allowed:
        excluded.append({"path": rel, "reason": reason, "size_bytes": src.stat().st_size})
        return
    if rel in copied_seen:
        return
    dst = payload_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(rel)
    copied_seen.add(rel)


def copy_path(
    src_rel: str,
    payload_dir: Path,
    copied: list[str],
    copied_seen: set[str],
    missing: list[str],
    excluded: list[dict[str, Any]],
) -> None:
    src = REPO_ROOT / src_rel
    if not src.exists():
        missing.append(src_rel)
        return
    if src.is_file():
        copy_file(src, payload_dir, copied, copied_seen, excluded)
        return
    for dirpath, dirnames, filenames in os.walk(src):
        root = Path(dirpath)
        kept_dirs = []
        for dirname in dirnames:
            candidate = root / dirname
            if set(candidate.resolve().relative_to(REPO_ROOT).parts) & EXCLUDED_PARTS:
                excluded.append({"path": repo_rel(candidate), "reason": "excluded_directory_pruned", "size_bytes": None})
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in sorted(filenames):
            path = root / filename
            if path.is_file():
                copy_file(path, payload_dir, copied, copied_seen, excluded)


def payload_hashes(payload_dir: Path, copied: list[str]) -> list[tuple[str, str]]:
    rows = []
    for rel in sorted(copied):
        if rel == "PACK_PAYLOAD_SHA256SUMS.txt":
            continue
        rows.append((sha256_file(payload_dir / rel), rel))
    return rows


def write_payload_sidecars(payload_dir: Path, copied: list[str], excluded: list[dict[str, Any]], missing: list[str]) -> None:
    write_text(payload_dir / "PACK_FILELIST.txt", "\n".join(sorted(copied)) + "\n")
    write_text(payload_dir / "PACK_MISSING.txt", "\n".join(sorted(missing)) + ("\n" if missing else ""))
    write_text(payload_dir / "PACK_EXCLUDED.json", json.dumps(excluded, indent=2, sort_keys=True) + "\n")
    sha_rows = payload_hashes(payload_dir, copied)
    write_text(payload_dir / "PACK_PAYLOAD_SHA256SUMS.txt", "".join(f"{digest}  {rel}\n" for digest, rel in sha_rows))


def write_readme(payload_dir: Path, tag: str) -> None:
    final_decision = read_json(AUDIT_ROOT / "v96_phase10_dev_decision_object_core_k512_s010_h1_R32_all_controls/final_dev_decision.json")
    best = final_decision.get("phase6_c_best_variant", {}) if isinstance(final_decision, dict) else {}
    lines = [
        f"# Stream4D v96 D4RT Micro-Primitive Core Audit Pack",
        "",
        f"tag: `{tag}`",
        "",
        "## Scope",
        "",
        "Compact reviewer-facing bundle for Stream4D v96. It includes the plan, execution log, retrospective log, core v96 code, evaluator/support scripts, latest K512 object-specific-core evidence chain, controls, and compact provenance artifacts.",
        "",
        "Excluded by design: raw datasets, checkpoints/weights, caches, old audit packs, image/video files, numpy tensors, and oversized generated CSVs that are documented in `PACK_EXCLUDED.json`.",
        "",
        "## Latest Decision",
        "",
        f"decision: `{final_decision.get('decision')}`",
        f"holdout_allowed: `{final_decision.get('holdout_allowed')}`",
        f"local2history_allowed: `{final_decision.get('local2history_allowed')}`",
        f"reason: `{final_decision.get('reason')}`",
        "",
        "## Latest Best Row",
        "",
        f"phase6_root: `Stream3D/outputs/audit/v96_phase6_render_snap_object_core_k512_s010_h1_frame_count_x_masklet`",
        f"variant: `{best.get('readout_variant')}`",
        f"MV_AP: `{best.get('MV_AP_window')}`",
        f"AP50: `{best.get('MV_AP50_window')}`",
        f"AP25: `{best.get('MV_AP25_window')}`",
        f"SF50: `{best.get('ScoreFreeMatch50_window')}`",
        f"same_frame_collision_count: `{best.get('same_frame_collision_count')}`",
        f"pixel_collision_count: `{best.get('pixel_collision_count')}`",
        f"missing_mask_raster_count: `{best.get('missing_mask_raster_count')}`",
        "",
        "## Validation",
        "",
        "The archive is accompanied by external sidecars: `.sha256`, `.validation.json`, `.unzip_test.txt`, and `.payload_sha256_check.txt`.",
        "The payload also includes `PACK_FILELIST.txt`, `PACK_PAYLOAD_SHA256SUMS.txt`, `PACK_EXCLUDED.json`, `PACK_MISSING.txt`, `GIT_STATUS_RELEVANT.txt`, and `GIT_DIFF_RELEVANT.patch`. `PACK_PAYLOAD_SHA256SUMS.txt` covers payload files except itself.",
        "",
    ]
    write_text(payload_dir / "README_AUDIT.md", "\n".join(lines))


def write_git_sidecars(payload_dir: Path) -> None:
    relevant = DOC_PATHS + CODE_PATHS
    write_text(payload_dir / "GIT_STATUS_RELEVANT.txt", run_text(["git", "status", "--short", "--", *relevant]))
    write_text(payload_dir / "GIT_DIFF_RELEVANT.patch", run_text(["git", "diff", "--", *relevant]))
    compile_targets = [path for path in CODE_PATHS if (REPO_ROOT / path).exists() and path.endswith(".py")]
    cmd = [sys.executable, "-m", "py_compile", *compile_targets]
    write_text(payload_dir / "PY_COMPILE_RELEVANT.txt", run_text(cmd))


def make_zip(payload_dir: Path, archive_path: Path, tag: str) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(payload_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = f"{tag}/{path.relative_to(payload_dir).as_posix()}"
            zf.write(path, arcname)
            entries.append(arcname)
    return entries


def validate_archive(archive_path: Path, payload_dir: Path, tag: str) -> dict[str, Any]:
    unzip_test = run_text(["unzip", "-tq", archive_path.name], cwd=archive_path.parent)
    write_text(archive_path.with_suffix(archive_path.suffix + ".unzip_test.txt"), unzip_test)
    with zipfile.ZipFile(archive_path) as zf:
        zip_entries = sorted(name for name in zf.namelist() if not name.endswith("/"))
    payload_entries = sorted(f"{tag}/{path.relative_to(payload_dir).as_posix()}" for path in payload_dir.rglob("*") if path.is_file())
    missing_in_zip = sorted(set(payload_entries) - set(zip_entries))
    extra_in_zip = sorted(set(zip_entries) - set(payload_entries))
    with tempfile.TemporaryDirectory(prefix=f"{tag}_verify_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(tmp_path)
        extracted_payload = tmp_path / tag
        payload_check = run_text(["sha256sum", "-c", "PACK_PAYLOAD_SHA256SUMS.txt"], cwd=extracted_payload)
    write_text(archive_path.with_suffix(archive_path.suffix + ".payload_sha256_check.txt"), payload_check)
    archive_sha = sha256_file(archive_path)
    sha_sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    write_text(sha_sidecar, f"{archive_sha}  {archive_path.name}\n")
    sidecar_check = run_text(["sha256sum", "-c", sha_sidecar.name], cwd=archive_path.parent)
    write_text(archive_path.with_suffix(archive_path.suffix + ".sha256_check.txt"), sidecar_check)
    validation = {
        "archive": archive_path.name,
        "archive_path": repo_rel(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha,
        "zip_entry_count": len(zip_entries),
        "payload_file_count": len(payload_entries),
        "entry_parity_ok": not missing_in_zip and not extra_in_zip,
        "entry_missing_in_zip": missing_in_zip,
        "entry_extra_in_zip": extra_in_zip,
        "zip_test_ok": "No errors detected" in unzip_test or "returncode=0" in unzip_test,
        "payload_hash_check_ok": "FAILED" not in payload_check and "returncode=0" in payload_check,
        "sha256_sidecar_ok": "returncode=0" in sidecar_check,
    }
    write_text(archive_path.with_suffix(archive_path.suffix + ".validation.json"), json.dumps(validation, indent=2, sort_keys=True) + "\n")
    return validation


def build(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{TAG_PREFIX}_{timestamp}"
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    stage_dir = PACK_ROOT / f".stage_{tag}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    payload_dir = stage_dir / tag
    payload_dir.mkdir(parents=True)
    copied: list[str] = []
    copied_seen: set[str] = set()
    missing: list[str] = []
    excluded: list[dict[str, Any]] = []
    for rel in DOC_PATHS + CODE_PATHS + ARTIFACT_DIRS:
        copy_path(rel, payload_dir, copied, copied_seen, missing, excluded)
    write_readme(payload_dir, tag)
    write_git_sidecars(payload_dir)
    sidecar_rel_paths = [
        "README_AUDIT.md",
        "GIT_STATUS_RELEVANT.txt",
        "GIT_DIFF_RELEVANT.patch",
        "PY_COMPILE_RELEVANT.txt",
    ]
    for rel in sidecar_rel_paths:
        if rel not in copied_seen and (payload_dir / rel).exists():
            copied.append(rel)
            copied_seen.add(rel)
    for rel in ["PACK_FILELIST.txt", "PACK_MISSING.txt", "PACK_EXCLUDED.json", "PACK_PAYLOAD_SHA256SUMS.txt"]:
        if rel not in copied_seen:
            copied.append(rel)
            copied_seen.add(rel)
    write_payload_sidecars(payload_dir, copied, excluded, missing)
    archive_path = PACK_ROOT / f"{tag}.zip"
    if archive_path.exists():
        raise FileExistsError(f"archive already exists: {archive_path}")
    make_zip(payload_dir, archive_path, tag)
    validation = validate_archive(archive_path, payload_dir, tag)
    shutil.rmtree(stage_dir)
    write_text(PACK_ROOT / ".latest_stream4d_v96_pack_tag", tag + "\n")
    result = {
        "tag": tag,
        "archive_path": repo_rel(archive_path),
        "archive_sha256": validation["archive_sha256"],
        "archive_size_bytes": validation["archive_size_bytes"],
        "payload_file_count": validation["payload_file_count"],
        "zip_entry_count": validation["zip_entry_count"],
        "entry_parity_ok": validation["entry_parity_ok"],
        "zip_test_ok": validation["zip_test_ok"],
        "payload_hash_check_ok": validation["payload_hash_check_ok"],
        "sha256_sidecar_ok": validation["sha256_sidecar_ok"],
        "missing_declared_paths": missing,
        "excluded_count": len(excluded),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default="")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()

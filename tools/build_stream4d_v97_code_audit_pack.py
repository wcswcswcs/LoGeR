#!/usr/bin/env python3
"""Build a compact Stream4D v97 code/artifact audit packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = REPO_ROOT / "code_audit_pack"
AUDIT_ROOT = REPO_ROOT / "Stream3D/outputs/audit"
TAG_PREFIX = "stream4d_v97_d4rt_micro_primitive_semantic_core_audit"
MAX_FILE_BYTES = 32 * 1024 * 1024

DOC_PATHS = [
    "docs/stream4d_v97_d4rt_micro_primitive_semantic_affinity_field_plan.md",
    "docs/stream4d_v97_执行日志.md",
    "docs/stream4d_v97_实验结果复盘.md",
]

CODE_PATHS = [
    "tools/build_stream4d_v97_code_audit_pack.py",
    "Stream3D/tools/build_v97_phase0_fact_lock.py",
    "Stream3D/tools/build_v97_phase1_query_planner.py",
    "Stream3D/tools/build_v97_phase1_overlap_chunk_query_views.py",
    "Stream3D/tools/build_v97_phase1_relative_segment_query_views.py",
    "Stream3D/tools/build_v97_phase2_d4rt_micro_tracks.py",
    "Stream3D/tools/build_v97_phase2_full_aggregate.py",
    "Stream3D/tools/build_v97_phase2_overlap_stitch_micro_tracks.py",
    "Stream3D/tools/build_v97_phase2_source_preserving_query_repair.py",
    "Stream3D/tools/diagnose_v97_phase2_stratum_support.py",
    "Stream3D/tools/build_v97_phase3_triton_incidence.py",
    "Stream3D/tools/build_v97_phase4_micro_affinity_feature.py",
    "Stream3D/tools/build_v97_phase5_object_birth.py",
    "Stream3D/tools/build_v97_phase6_render_splat.py",
    "Stream3D/tools/build_v97_phase7_support_iou_readout.py",
    "Stream3D/tools/build_v97_phase9_failure_decomposition.py",
    # Reused decode/evaluation/provenance code that v97 depends on or cites.
    "Stream3D/tools/build_v96_phase2_d4rt_micro_tracks.py",
    "Stream3D/tools/build_v96_phase3_triton_incidence.py",
    "Stream3D/tools/build_v96_phase4_affinity_features.py",
    "Stream3D/tools/build_v96_phase5_object_birth.py",
    "Stream3D/tools/build_v96_phase6_render_snap.py",
    "Stream3D/tools/build_v95_phase1_physical_source_registry.py",
    "Stream3D/tools/build_v91_radio_mask_features.py",
    "Stream3D/tools/merge_v91_radio_mask_feature_stores.py",
    "Stream3D/tools/run_v65_scene_multiview_ap.py",
]

STATIC_ARTIFACT_DIRS = [
    # Upstream compact provenance referenced by the v97 scripts/logs.
    "Stream3D/outputs/audit/v95_phase1_physical_source_registry",
    "Stream3D/outputs/audit/v93_phase1_source_edge_registry",
    "Stream3D/outputs/audit/v93_phase3_region_edge_graph",
    "Stream3D/outputs/audit/v96_phase10_dev_decision_object_core_k512_s010_h1_R32_all_controls",
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
    "carrier_batches",
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
    except Exception as exc:  # noqa: BLE001 - expose malformed/missing audit inputs.
        return {"read_error": str(exc), "path": repo_rel(path) if path.exists() else path.as_posix()}


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


def artifact_dirs() -> list[str]:
    """Package the current v97 audit frontier plus compact upstream provenance."""
    discovered: list[str] = []
    if AUDIT_ROOT.exists():
        discovered = [
            f"Stream3D/outputs/audit/{path.name}"
            for path in AUDIT_ROOT.iterdir()
            if path.is_dir() and path.name.startswith("v97_phase")
        ]
    out: list[str] = []
    seen: set[str] = set()
    for rel in sorted(discovered) + STATIC_ARTIFACT_DIRS:
        if rel in seen:
            continue
        seen.add(rel)
        out.append(rel)
    return out


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
        for dirname in sorted(dirnames):
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
        path = payload_dir / rel
        if path.exists() and path.is_file():
            rows.append((sha256_file(path), rel))
    return rows


def dedupe_excluded(excluded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any]] = set()
    out: list[dict[str, Any]] = []
    for row in excluded:
        key = (row.get("path"), row.get("reason"), row.get("size_bytes"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_payload_sidecars(payload_dir: Path, copied: list[str], excluded: list[dict[str, Any]], missing: list[str]) -> None:
    excluded = dedupe_excluded(excluded)
    write_text(payload_dir / "PACK_FILELIST.txt", "\n".join(sorted(copied)) + "\n")
    write_text(payload_dir / "PACK_MISSING.txt", "\n".join(sorted(missing)) + ("\n" if missing else ""))
    write_text(payload_dir / "PACK_EXCLUDED.json", json.dumps(excluded, indent=2, sort_keys=True) + "\n")
    sha_rows = payload_hashes(payload_dir, copied)
    write_text(payload_dir / "PACK_PAYLOAD_SHA256SUMS.txt", "".join(f"{digest}  {rel}\n" for digest, rel in sha_rows))


def summarize_artifact_roots() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in artifact_dirs():
        root = REPO_ROOT / rel
        if not root.exists():
            rows.append({"path": rel, "exists": False})
            continue
        file_count = 0
        total_bytes = 0
        largest: list[dict[str, Any]] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            file_count += 1
            size = path.stat().st_size
            total_bytes += size
            largest.append({"path": repo_rel(path), "size_bytes": size})
        largest = sorted(largest, key=lambda item: int(item["size_bytes"]), reverse=True)[:8]
        rows.append(
            {
                "path": rel,
                "exists": True,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "summary_json": read_json(root / "summary.json") if (root / "summary.json").exists() else None,
                "largest_files": largest,
            }
        )
    return rows


def write_build_context(payload_dir: Path, tag: str) -> None:
    context = {
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cwd": str(REPO_ROOT),
        "argv": sys.argv,
        "python": sys.executable,
        "max_file_bytes": MAX_FILE_BYTES,
        "doc_paths": DOC_PATHS,
        "code_paths": CODE_PATHS,
        "artifact_dirs": artifact_dirs(),
        "artifact_root_summaries": summarize_artifact_roots(),
    }
    write_text(payload_dir / "PACK_BUILD_CONTEXT.json", json.dumps(context, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_readme(payload_dir: Path, tag: str) -> None:
    phase0 = read_json(AUDIT_ROOT / "v97_phase0_fact_lock/summary.json")
    phase1 = read_json(AUDIT_ROOT / "v97_phase1_query_planner/summary.json")
    phase2_full = read_json(AUDIT_ROOT / "v97_phase2_d4rt_micro_tracks_full_D1_D3_aggregate/summary.json")
    phase2_diag = read_json(
        AUDIT_ROOT / "v97_phase2_d4rt_micro_tracks_relseg_size7_D3_source_preserve2048_weighted_aggregate/summary.json"
    )
    phase3 = read_json(
        AUDIT_ROOT / "v97_phase3_triton_incidence_D3_source_preserve2048_relseg_size7_500k_gpu6/summary.json"
    )
    phase4 = read_json(
        AUDIT_ROOT / "v97_phase4_micro_affinity_feature_D3_source_preserve2048_region_proxy_500k_gpu6/summary.json"
    )
    phase5 = read_json(AUDIT_ROOT / "v97_phase5_object_birth_region_proxy_500k/summary.json")
    phase5_best = phase5.get("best_variant", {}) if isinstance(phase5.get("best_variant", {}), dict) else {}
    phase6_initial = read_json(AUDIT_ROOT / "v97_phase6_render_splat_C0_region_proxy_500k_gpu7/summary.json")
    phase6 = read_json(AUDIT_ROOT / "v97_phase6_render_splat_C0_region_proxy_500k_gpu7_sparsefix/summary.json")
    phase6_quality = phase6.get("quality", {}) if isinstance(phase6.get("quality", {}), dict) else {}
    phase2_q512 = read_json(AUDIT_ROOT / "v97_phase2_d4rt_micro_tracks_overlap48_48clip_all4_q512_gpu7/summary.json")
    phase2_q512_stitch = read_json(AUDIT_ROOT / "v97_phase2_d4rt_micro_tracks_overlap48_48clip_all4_q512_stitched/summary.json")
    phase5_q512 = read_json(AUDIT_ROOT / "v97_phase5_object_birth_overlap48_48clip_q512_stitched_500k/summary.json")
    phase5_q512_best = phase5_q512.get("best_variant", {}) if isinstance(phase5_q512.get("best_variant", {}), dict) else {}
    phase6_q512_b4 = read_json(AUDIT_ROOT / "v97_phase6_render_splat_B4_overlap48_48clip_q512_stitched_500k_gpu7/summary.json")
    phase6_q512_b4_quality = phase6_q512_b4.get("quality", {}) if isinstance(phase6_q512_b4.get("quality", {}), dict) else {}
    phase7_dual = read_json(AUDIT_ROOT / "v97_phase7_support_iou_readout_overlap48_48clip_q512_B4_stitched_gpu7_sigma2_window_scene/summary.json")
    phase7_dual_best = phase7_dual.get("best_variant", {}) if isinstance(phase7_dual.get("best_variant", {}), dict) else {}
    phase9 = read_json(AUDIT_ROOT / "v97_phase9_failure_decomposition/blocker_summary.json")
    lines = [
        "# Stream4D v97 D4RT Micro-Primitive Semantic Core Audit Pack",
        "",
        f"tag: `{tag}`",
        "",
        "## Scope",
        "",
        "Compact reviewer-facing bundle for Stream4D v97. It includes the v97 plan, execution log, retrospective log, core v97 code, reused D4RT/Triton dependencies, and compact Phase0-Phase7 artifact sidecars.",
        "",
        "Excluded by design: raw datasets, checkpoints/weights, caches, old audit packs, image/video files, numpy tensors, carrier batch directories, and oversized generated CSVs. Excluded paths and sizes are listed in `PACK_EXCLUDED.json`; copied payload files are covered by `PACK_PAYLOAD_SHA256SUMS.txt`.",
        "",
        "## Current Decision",
        "",
        f"phase0_decision: `{phase0.get('decision')}`",
        f"phase1_decision: `{phase1.get('decision')}`",
        f"phase2_full_decision: `{phase2_full.get('decision')}`",
        f"phase2_full_dev_gate_pass: `{phase2_full.get('full_dev_gate_pass')}`",
        f"phase2_diagnostic_decision: `{phase2_diag.get('decision')}`",
        f"phase2_diagnostic_can_enter_phase3: `{phase2_diag.get('can_enter_phase3')}`",
        f"phase2_diagnostic_scope: `{phase2_diag.get('can_enter_phase3_scope')}`",
        f"phase2_diagnostic_full_dev_gate_pass: `{phase2_diag.get('full_dev_gate_pass')}`",
        f"phase3_decision: `{phase3.get('decision')}`",
        f"phase3_diagnostic_scope: `{phase3.get('diagnostic_scope')}`",
        f"phase3_selected_track_rows: `{phase3.get('selected_track_rows')}`",
        f"phase3_runtime_incidence_sec: `{phase3.get('runtime_incidence_sec')}`",
        f"phase3_uses_gt_for_prediction: `{phase3.get('uses_gt_for_prediction')}`",
        f"phase3_uses_future: `{phase3.get('uses_future')}`",
        f"phase4_decision: `{phase4.get('decision')}`",
        f"phase4_semantic_source: `{phase4.get('semantic_source')}`",
        f"phase4_full_semantic_gate_pass: `{phase4.get('full_semantic_gate_pass')}`",
        f"phase4_can_enter_phase5_diagnostic: `{phase4.get('can_enter_phase5_diagnostic')}`",
        f"phase5_decision: `{phase5.get('decision')}`",
        f"phase5_best_variant: `{phase5_best.get('variant_id')}`",
        f"phase5_can_enter_phase6_diagnostic: `{phase5.get('can_enter_phase6_diagnostic')}`",
        f"phase6_initial_decision: `{phase6_initial.get('decision')}`",
        f"phase6_initial_render_runtime_ms: `{phase6_initial.get('quality', {}).get('render_runtime_ms') if isinstance(phase6_initial.get('quality', {}), dict) else None}`",
        f"phase6_sparsefix_decision: `{phase6.get('decision')}`",
        f"phase6_sparsefix_emitted_object_frame_count: `{phase6.get('emitted_object_frame_count')}`",
        f"phase6_sparsefix_render_runtime_ms: `{phase6_quality.get('render_runtime_ms')}`",
        f"phase6_sparsefix_triton_cpu_parity_error_max: `{phase6.get('triton_splat_parity', {}).get('cpu_vs_triton_abs_error_max') if isinstance(phase6.get('triton_splat_parity', {}), dict) else None}`",
        f"phase6_can_enter_phase7_diagnostic: `{phase6.get('can_enter_phase7_diagnostic')}`",
        f"phase6_can_enter_phase7_full: `{phase6.get('can_enter_phase7_full')}`",
        "",
        "## Current Overlap-Stitch + 48CLIP Rerun Frontier",
        "",
        f"q512_phase2_decision: `{phase2_q512.get('decision')}`",
        f"q512_phase2_query_count: `{phase2_q512.get('best_variant', {}).get('query_count') if isinstance(phase2_q512.get('best_variant', {}), dict) else phase2_q512.get('query_count')}`",
        f"q512_phase2_uv_in01_rate: `{phase2_q512.get('best_variant', {}).get('uv_in01_rate') if isinstance(phase2_q512.get('best_variant', {}), dict) else None}`",
        f"q512_overlap_stitch_decision: `{phase2_q512_stitch.get('decision')}`",
        f"q512_overlap_stitch_micro_track_row_count: `{phase2_q512_stitch.get('micro_track_row_count')}`",
        f"q512_overlap_stitch_edge_count: `{phase2_q512_stitch.get('overlap_stitch_edge_count')}`",
        f"q512_phase5_decision: `{phase5_q512.get('decision')}`",
        f"q512_phase5_best_variant: `{phase5_q512_best.get('variant_id')}`",
        f"q512_phase5_best_keypoint_coverage_rate: `{phase5_q512_best.get('keypoint_coverage_rate')}`",
        f"q512_phase6_B4_decision: `{phase6_q512_b4.get('decision')}`",
        f"q512_phase6_B4_support_area_ratio_mean: `{phase6_q512_b4_quality.get('support_area_ratio_mean')}`",
        f"q512_phase6_B4_triton_cpu_parity_error_max: `{phase6_q512_b4.get('triton_splat_parity', {}).get('cpu_vs_triton_abs_error_max') if isinstance(phase6_q512_b4.get('triton_splat_parity', {}), dict) else None}`",
        f"q512_phase7_dualmetric_decision: `{phase7_dual.get('decision')}`",
        f"q512_phase7_dualmetric_best_variant: `{phase7_dual_best.get('variant_id')}`",
        f"q512_phase7_dualmetric_best_MV_AP_window: `{phase7_dual_best.get('MV_AP_window')}`",
        f"q512_phase7_dualmetric_best_MV_AP_scene: `{phase7_dual_best.get('MV_AP_scene')}`",
        f"q512_phase7_dualmetric_scene_comparator_available: `{phase7_dual.get('scene_comparator_available')}`",
        f"q512_phase7_dualmetric_scene_comparator_note: `{phase7_dual.get('scene_comparator_note')}`",
        f"phase9_decision: `{phase9.get('decision')}`",
        f"phase9_target_achieved: `{phase9.get('target_achieved')}`",
        f"phase9_primary_blockers: `{phase9.get('primary_blockers')}`",
        f"phase9_best_MV_AP_window: `{phase9.get('best_MV_AP_window')}`",
        f"phase9_best_MV_AP_scene: `{phase9.get('best_MV_AP_scene')}`",
        "",
        "## Key Evidence",
        "",
        "Full-dev active_sparse D3 failed the UV and source-support gates while passing boundary/competing support. The source-preserving Q3_source_preserve2048 segmented diagnostic repaired Phase2 quality gates and allowed a diagnostic Phase3 run, but it is still segment_diagnostic and not full_dev.",
        "",
        "Phase3 used the v96 Triton incidence kernel through the v97 wrapper and passed bounded 500k-track diagnostic parity/quality gates. Phase4 built a GPU-assisted region-proxy micro-affinity field and passed proxy-quality gates, but the full semantic gate failed because no dense semantic tensor was available.",
        "",
        "Phase5 object birth selected C0_cover_seed_plus_affinity_expand as the best diagnostic variant with zero cannot-link/same-frame violations. Phase6 initially failed runtime, then the sparsefix kept the Triton splat parity error at 1.1920928955078125e-07 while reducing render_runtime_ms below budget and passing the diagnostic gate.",
        "",
        "The corrected overlap-stitch rerun used D4RT-only overlap/self-stitch in the method path. Final GT-Sim3 remains diagnostic-only for visualization/geometry quality and is not a method input.",
        "",
        "The latest q512+B4 Phase7 readout explicitly reports the locked main metric as MV_AP_window and the local2history/scene diagnostic as MV_AP_scene. MV_AP_window is 0.0 for every readout variant, so v97 remains No-Go. Phase0 locks MV_AP_window; missing scene comparator fields are not treated as a primary blocker.",
        "",
        "Phase9 failure decomposition preserves the same MV_AP_window No-Go conclusion and groups the remaining evidence into full-dev scope, D4RT geometry, semantic feature, object-birth coverage, and render/readout support alignment blockers. Ranking/score calibration is not the primary blocker because score-free AP is also zero.",
        "",
        "## Validation",
        "",
        "External sidecars are written next to the archive: `.sha256`, `.sha256_check.txt`, `.unzip_test.txt`, `.payload_sha256_check.txt`, and `.validation.json`.",
        "The payload includes `PACK_FILELIST.txt`, `PACK_PAYLOAD_SHA256SUMS.txt`, `PACK_EXCLUDED.json`, `PACK_MISSING.txt`, `PACK_BUILD_CONTEXT.json`, `GIT_STATUS_RELEVANT.txt`, `GIT_DIFF_RELEVANT.patch`, and `PY_COMPILE_RELEVANT.txt`.",
        "",
    ]
    write_text(payload_dir / "README_AUDIT.md", "\n".join(lines))


def write_git_sidecars(payload_dir: Path) -> None:
    relevant = DOC_PATHS + CODE_PATHS
    write_text(payload_dir / "GIT_STATUS_RELEVANT.txt", run_text(["git", "status", "--short", "--", *relevant]))
    write_text(payload_dir / "GIT_DIFF_RELEVANT.patch", run_text(["git", "diff", "--", *relevant]))
    compile_targets = [path for path in CODE_PATHS if (REPO_ROOT / path).exists() and path.endswith(".py")]
    write_text(payload_dir / "PY_COMPILE_RELEVANT.txt", run_text([sys.executable, "-m", "py_compile", *compile_targets]))


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
        "zip_test_ok": "returncode=0" in unzip_test,
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
    try:
        for rel in DOC_PATHS + CODE_PATHS + artifact_dirs():
            copy_path(rel, payload_dir, copied, copied_seen, missing, excluded)
        write_build_context(payload_dir, tag)
        write_readme(payload_dir, tag)
        write_git_sidecars(payload_dir)
        sidecar_rel_paths = [
            "PACK_BUILD_CONTEXT.json",
            "README_AUDIT.md",
            "GIT_STATUS_RELEVANT.txt",
            "GIT_DIFF_RELEVANT.patch",
            "PY_COMPILE_RELEVANT.txt",
            "PACK_FILELIST.txt",
            "PACK_MISSING.txt",
            "PACK_EXCLUDED.json",
            "PACK_PAYLOAD_SHA256SUMS.txt",
        ]
        for rel in sidecar_rel_paths:
            if rel not in copied_seen:
                copied.append(rel)
                copied_seen.add(rel)
        excluded = dedupe_excluded(excluded)
        write_payload_sidecars(payload_dir, copied, excluded, missing)
        archive_path = PACK_ROOT / f"{tag}.zip"
        if archive_path.exists():
            raise FileExistsError(f"archive already exists: {archive_path}")
        make_zip(payload_dir, archive_path, tag)
        validation = validate_archive(archive_path, payload_dir, tag)
    finally:
        if stage_dir.exists() and not args.keep_stage:
            shutil.rmtree(stage_dir)

    write_text(PACK_ROOT / ".latest_stream4d_v97_pack_tag", tag + "\n")
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
    parser.add_argument("--keep-stage", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()

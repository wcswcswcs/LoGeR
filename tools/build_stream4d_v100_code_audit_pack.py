#!/usr/bin/env python3
"""Build a compact Stream4D v100 code-focused audit packet."""

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
TAG_PREFIX = "stream4d_v100_f2_gpu_history_memory_code_only_audit"
MAX_FILE_BYTES = 32 * 1024 * 1024

DOC_PATHS = [
    "docs/stream4d_v100_f2_gpu_history_memory_plan.md",
    "docs/stream4d_v100_执行日志.md",
    "docs/stream4d_v100_实验结果复盘.md",
]

CORE_CODE_GLOBS = [
    "Stream3D/tools/build_v100_phase*.py",
]

EXTRA_CODE_PATHS = [
    "tools/build_stream4d_v100_code_audit_pack.py",
    "Stream3D/tools/build_v99_phase1_f2_base_reproduction.py",
    "Stream3D/tools/build_v99_phase4_f2_da3_link_verifier.py",
    "Stream3D/tools/build_v99_phase10_holdout_final_decision.py",
    "Stream3D/tools/build_v99_phase10ag_prefix_da3_d4rt_sim3_alignment.py",
    "Stream3D/tools/build_v99_phase10ah_prefix_sim3_aligned_anchor_scene_stitch.py",
    "Stream3D/tools/build_v99_phase10ai_prefix_sim3_d4rt_semantic_scene_repair.py",
    "Stream3D/tools/build_v99_phase10k_holdout_chunk_object_birth_sweep.py",
    "Stream3D/tools/build_v99_phase10o_overlap3_scene_stitch_repair.py",
    "Stream3D/tools/build_v99_phase10r_geometry_provider_contract_audit.py",
    "Stream3D/tools/build_v99_phase10s_da3_holdout_chunk32o3_provider_audit.py",
    "Stream3D/tools/build_v99_phase10x_d4rt_holdout_chunk32o3_provider_audit.py",
    "Stream3D/tools/build_v99_phase10y_d4rt_anchor_holdout_scene_stitch.py",
    "Stream3D/tools/build_v99_phase10z_d4rt_verifier_semantic_scene_repair.py",
    "Stream3D/tools/check_mv_ap_contract.py",
    "Stream3D/tools/run_v65_scene_multiview_ap.py",
    "Stream3D/stream4d_native/v65_ap_contract.py",
    "Stream3D/stream4d_native/v65_final_eval.py",
]

LIGHT_EXTS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".parquet",
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
    except Exception as exc:  # noqa: BLE001 - audit packet should expose malformed/missing inputs.
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


def code_paths() -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for pattern in CORE_CODE_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if path.is_file():
                rel = repo_rel(path)
                if rel not in seen:
                    paths.append(rel)
                    seen.add(rel)
    for rel in EXTRA_CODE_PATHS:
        if rel not in seen:
            paths.append(rel)
            seen.add(rel)
    return paths


def artifact_dirs(*, include_artifacts: bool) -> list[str]:
    if not include_artifacts:
        return []
    if not AUDIT_ROOT.exists():
        return []
    return sorted(
        f"Stream3D/outputs/audit/{path.name}"
        for path in AUDIT_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("v100_")
    )


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
            rel_parts = candidate.resolve().relative_to(REPO_ROOT).parts
            if set(rel_parts) & EXCLUDED_PARTS:
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


def summarize_artifact_roots(*, include_artifacts: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in artifact_dirs(include_artifacts=include_artifacts):
        root = REPO_ROOT / rel
        file_count = 0
        copied_candidate_count = 0
        total_bytes = 0
        largest: list[dict[str, Any]] = []
        excluded_reason_counts: dict[str, int] = {}
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            file_count += 1
            size = path.stat().st_size
            total_bytes += size
            largest.append({"path": repo_rel(path), "size_bytes": size})
            allowed, reason = should_copy_file(path)
            if allowed:
                copied_candidate_count += 1
            else:
                excluded_reason_counts[reason] = excluded_reason_counts.get(reason, 0) + 1
        rows.append(
            {
                "path": rel,
                "exists": root.exists(),
                "file_count": file_count,
                "copied_candidate_count": copied_candidate_count,
                "total_bytes": total_bytes,
                "summary_json": read_json(root / "summary.json") if (root / "summary.json").exists() else None,
                "largest_files": sorted(largest, key=lambda item: int(item["size_bytes"]), reverse=True)[:8],
                "excluded_reason_counts": excluded_reason_counts,
            }
        )
    return rows


def write_build_context(payload_dir: Path, tag: str, selected_code_paths: list[str], *, include_artifacts: bool) -> None:
    context = {
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cwd": str(REPO_ROOT),
        "argv": sys.argv,
        "python": sys.executable,
        "max_file_bytes": MAX_FILE_BYTES,
        "doc_paths": DOC_PATHS,
        "code_paths": selected_code_paths,
        "include_artifacts": include_artifacts,
        "artifact_dirs": artifact_dirs(include_artifacts=include_artifacts),
        "artifact_root_summaries": summarize_artifact_roots(include_artifacts=include_artifacts),
        "exclusion_policy": {
            "light_exts": sorted(LIGHT_EXTS),
            "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
            "excluded_parts": sorted(EXCLUDED_PARTS),
        },
    }
    write_text(payload_dir / "PACK_BUILD_CONTEXT.json", json.dumps(context, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_readme(payload_dir: Path, tag: str) -> None:
    phase2c = read_json(AUDIT_ROOT / "v100_phase2c_overlap3_local_repair/summary.json")
    phase4r = read_json(AUDIT_ROOT / "v100_phase4r_position_union_repair/summary.json")
    phase5c = read_json(AUDIT_ROOT / "v100_phase5c_da3_broad_split_repair/summary.json")
    phase5d = read_json(AUDIT_ROOT / "v100_phase5d_da3_surface_component_split_repair/summary.json")
    phase8e = read_json(AUDIT_ROOT / "v100_phase8e_final_decision_freeze/summary.json")
    best_scene = phase8e.get("best_current_scene_attempt", {}) if isinstance(phase8e.get("best_current_scene_attempt"), dict) else {}
    lines = [
        "# Stream4D v100 F2 GPU History Memory Code-Only Audit Pack",
        "",
        f"tag: `{tag}`",
        "",
        "## Scope",
        "",
        "Compact reviewer-facing code bundle for Stream4D v100. It includes the v100 plan, execution log, retrospective log, v100 phase code, and direct evaluator/provider helper code. It does not copy Stream3D/outputs/audit artifacts unless --include-artifacts is explicitly passed.",
        "",
        "Excluded by design: Stream3D/outputs/audit artifact payloads, raw datasets, checkpoints/weights, caches, old audit packs, image/video files, generated mask rasters, numpy tensors, model/object-tube binaries, and oversized files. Excluded paths and sizes are listed in `PACK_EXCLUDED.json`; copied payload files are covered by `PACK_PAYLOAD_SHA256SUMS.txt`.",
        "",
        "## Final Decision",
        "",
        f"phase8e_decision: `{phase8e.get('decision')}`",
        f"full_goal_achieved: `{phase8e.get('full_goal_achieved')}`",
        f"local_claim_allowed: `{phase8e.get('local_claim_allowed')}`",
        f"scene_claim_allowed: `{phase8e.get('scene_claim_allowed')}`",
        f"uses_gt_for_prediction: `{phase8e.get('uses_gt_for_prediction')}`",
        "",
        "## Local Phase2c Claim",
        "",
        f"phase2c_decision: `{phase2c.get('decision')}`",
        f"phase2c_pass: `{phase2c.get('phase2c_pass')}`",
        f"formal_claim_allowed: `{phase2c.get('formal_claim_allowed')}`",
        f"dev_MV_AP_window: `{phase2c.get('dev_MV_AP_window')}`",
        f"dev_MV_AP50_window: `{phase2c.get('dev_MV_AP50_window')}`",
        f"holdout_MV_AP_window: `{phase2c.get('holdout_MV_AP_window')}`",
        f"holdout_MV_AP50_window: `{phase2c.get('holdout_MV_AP50_window')}`",
        "",
        "## Best Scene Attempt",
        "",
        f"source_id: `{best_scene.get('source_id')}`",
        f"variant_id: `{best_scene.get('variant_id')}`",
        f"dev_MV_AP_scene: `{best_scene.get('dev_MV_AP_scene')}` gate `{best_scene.get('dev_scene_gate')}`",
        f"dev_MV_AP50_scene: `{best_scene.get('dev_MV_AP50_scene')}` gate `{best_scene.get('dev_ap50_gate')}`",
        f"holdout_MV_AP_scene: `{best_scene.get('holdout_MV_AP_scene')}` gate `{best_scene.get('holdout_scene_gate')}`",
        f"holdout_MV_AP50_scene: `{best_scene.get('holdout_MV_AP50_scene')}` gate `{best_scene.get('holdout_ap50_gate')}`",
        "",
        "## Continued Repair Evidence",
        "",
        f"phase4r_decision: `{phase4r.get('decision')}`",
        f"phase4r_best_variant: `{phase4r.get('best_variant_id')}`",
        f"phase4r_best_holdout_MV_AP_scene: `{phase4r.get('best_holdout_MV_AP_scene')}`",
        f"phase5c_decision: `{phase5c.get('decision')}`",
        f"phase5c_best_variant: `{phase5c.get('best_variant_id')}`",
        f"phase5c_split_component_row_count: `{phase5c.get('split_component_row_count')}`",
        f"phase5d_decision: `{phase5d.get('decision')}`",
        f"phase5d_best_variant: `{phase5d.get('best_variant_id')}`",
        f"phase5d_best_split_variant: `{phase5d.get('best_split_variant_id')}`",
        f"phase5d_best_split_holdout_MV_AP_scene: `{phase5d.get('best_split_holdout_MV_AP_scene')}`",
        f"phase5d_split_component_row_count: `{phase5d.get('split_component_row_count')}`",
        "",
        "## Audit Notes",
        "",
        "Phase5c initially had an evaluator scope bug; the invalid output was preserved separately and the fixed run reproduced Phase2c local metrics before method conclusions were drawn.",
        "Phase5c median-depth split and Phase5d connected surface-component split both failed to beat the no-split baseline. The final blocker is a missing non-GT cross-chunk identity witness, not per-frame DA3 mask splitting.",
        "",
        "## Validation",
        "",
        "External sidecars are written next to the archive: `.sha256`, `.sha256_check.txt`, `.unzip_test.txt`, `.payload_sha256_check.txt`, and `.validation.json`.",
        "The payload includes `PACK_FILELIST.txt`, `PACK_PAYLOAD_SHA256SUMS.txt`, `PACK_EXCLUDED.json`, `PACK_MISSING.txt`, `PACK_BUILD_CONTEXT.json`, `GIT_STATUS_RELEVANT.txt`, `GIT_DIFF_RELEVANT.patch`, and `PY_COMPILE_RELEVANT.txt`.",
        "",
    ]
    write_text(payload_dir / "README_AUDIT.md", "\n".join(lines))


def write_git_sidecars(payload_dir: Path, selected_code_paths: list[str]) -> None:
    relevant = DOC_PATHS + selected_code_paths
    write_text(payload_dir / "GIT_STATUS_RELEVANT.txt", run_text(["git", "status", "--short", "--", *relevant]))
    write_text(payload_dir / "GIT_DIFF_RELEVANT.patch", run_text(["git", "diff", "--", *relevant]))
    compile_targets = [path for path in selected_code_paths if (REPO_ROOT / path).exists() and path.endswith(".py")]
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

    selected_code_paths = code_paths()
    copied: list[str] = []
    copied_seen: set[str] = set()
    missing: list[str] = []
    excluded: list[dict[str, Any]] = []
    try:
        for rel in DOC_PATHS + selected_code_paths + artifact_dirs(include_artifacts=args.include_artifacts):
            copy_path(rel, payload_dir, copied, copied_seen, missing, excluded)
        write_build_context(payload_dir, tag, selected_code_paths, include_artifacts=args.include_artifacts)
        write_readme(payload_dir, tag)
        write_git_sidecars(payload_dir, selected_code_paths)
        for rel in [
            "PACK_BUILD_CONTEXT.json",
            "README_AUDIT.md",
            "GIT_STATUS_RELEVANT.txt",
            "GIT_DIFF_RELEVANT.patch",
            "PY_COMPILE_RELEVANT.txt",
            "PACK_FILELIST.txt",
            "PACK_MISSING.txt",
            "PACK_EXCLUDED.json",
            "PACK_PAYLOAD_SHA256SUMS.txt",
        ]:
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

    write_text(PACK_ROOT / ".latest_stream4d_v100_pack_tag", tag + "\n")
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
        "code_file_count": len(selected_code_paths),
        "artifact_dir_count": len(artifact_dirs(include_artifacts=args.include_artifacts)),
        "include_artifacts": args.include_artifacts,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--keep-stage", action="store_true")
    parser.add_argument("--include-artifacts", action="store_true", help="Also copy compact Stream3D/outputs/audit/v100_* sidecars.")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a compact Stream4D v92 code/artifact audit packet."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
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

TAG_PREFIX = "stream4d_v92_diagnostic_affinity_core_audit"

DOC_PATHS = [
    "docs/stream4d_v92_diagnostic_first_affinity_readout_detailed_plan.md",
    "docs/stream4d_v92_执行日志.md",
    "docs/stream4d_v92_实验结果复盘.md",
]

CODE_PATHS = [
    "tools/build_stream4d_v92_code_audit_pack.py",
    "Stream3D/tools/build_v92_phase0_mv_ap_contract.py",
    "Stream3D/tools/build_v92_phase1_source_container_registry.py",
    "Stream3D/tools/build_v92_phase2_d4rt_sufficiency.py",
    "Stream3D/tools/build_v92_phase3_d4rt_highres_config_audit.py",
    "Stream3D/tools/run_v92_d4rt_window_highres_recompute.py",
    "Stream3D/tools/build_v92_phase3_highres_bridge.py",
    "Stream3D/tools/run_v92_phase3_hr1_same_readout_adaptive.py",
    "Stream3D/tools/run_v92_phase3c_uncertainty_readout.py",
    "Stream3D/tools/build_v92_phase4_semantic_region_affinity.py",
    "Stream3D/tools/build_v92_phase4b_region_granularity_repair.py",
    "Stream3D/tools/build_v92_phase5_source_container_field.py",
    "Stream3D/tools/build_v92_phase5d_score_calibration.py",
    "Stream3D/tools/build_v92_phase6_attribution.py",
    "Stream3D/tools/build_v92_phase9_final_decision.py",
    "Stream3D/tools/build_v92_common_artifact_closure.py",
    # Supporting evaluator/materializer code imported by the v92 scripts.
    "Stream3D/tools/run_v65_scene_multiview_ap.py",
    "Stream3D/tools/run_v65_d4rt_stride_overlap_geometry.py",
    "Stream3D/tools/run_v89_mv_ap_stream3d_local_baseline.py",
    "Stream3D/tools/run_v89_recalc_point_projected_mv_ap.py",
    "Stream3D/tools/run_v90_carrier_supported_carving.py",
    "Stream3D/tools/run_v90_dev_extent_score_cross_audit.py",
    "Stream3D/tools/run_v90_geo_semantic_witness_cover.py",
    "Stream3D/tools/run_v90_mv_ap_window_contract.py",
    "Stream3D/tools/run_v91_mv_ap_window_affinity_readout_lock.py",
    "Stream3D/tools/run_v91_phase4_adaptive_uncertainty_materialization.py",
    "Stream3D/tools/run_v91_phase4_ap50_control_repair.py",
    "Stream3D/tools/run_v91_phase4_radius_sweep.py",
    "Stream3D/tools/build_v91_final_decision.py",
    "Stream3D/tools/build_v91_radio_mask_features.py",
    "Stream3D/tools/merge_v91_radio_mask_feature_stores.py",
    "Stream3D/tools/diagnose_v91_radio_feature_store_quality.py",
]

V92_ARTIFACT_DIRS = [
    "Stream3D/outputs/audit/v92_phase0_mv_ap_contract",
    "Stream3D/outputs/audit/v92_phase1_source_container_registry",
    "Stream3D/outputs/audit/v92_phase2_d4rt_sufficiency",
    "Stream3D/outputs/audit/v92_phase3_d4rt_highres",
    "Stream3D/outputs/audit/v92_phase3_d4rt_highres_hr2_grid16",
    "Stream3D/outputs/audit/v92_phase3_d4rt_highres_recompute",
    "Stream3D/outputs/audit/v92_phase3_hr1_same_readout_adaptive_materialization",
    "Stream3D/outputs/audit/v92_phase3_hr2_same_readout_adaptive_materialization",
    "Stream3D/outputs/audit/v92_phase3c_hr2_uncertainty_readout",
    "Stream3D/outputs/audit/v92_phase4_semantic_region_affinity",
    "Stream3D/outputs/audit/v92_phase4_semantic_region_affinity_scene0011",
    "Stream3D/outputs/audit/v92_phase4_semantic_region_affinity_scene0050",
    "Stream3D/outputs/audit/v92_phase4_semantic_region_affinity_smoke_scene0011",
    "Stream3D/outputs/audit/v92_phase4b_region_granularity_coarse2",
    "Stream3D/outputs/audit/v92_phase5_source_container_field",
    "Stream3D/outputs/audit/v92_phase5b_source_container_edge_field",
    "Stream3D/outputs/audit/v92_phase5c_tight_field_repair",
    "Stream3D/outputs/audit/v92_phase5d_score_calibration",
    "Stream3D/outputs/audit/v92_phase5e_coarse2_tight_field",
    "Stream3D/outputs/audit/v92_phase6_attribution",
    "Stream3D/outputs/audit/v92_phase9_casebook",
]

BASELINE_ARTIFACT_DIRS = [
    "Stream3D/outputs/audit/v89_phase0_mv_ap_contract",
    "Stream3D/outputs/audit/v90_phase0_mv_ap_contract",
    "Stream3D/outputs/audit/v91_phase0_mv_ap_contract",
    "Stream3D/outputs/audit/v91_phase8_dev_selection",
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
    "checkpoints",
    "code_audit_pack",
    "data",
    "generated_masks",
    "weights",
}

EXCLUDED_BASENAMES = {
    "field_region_membership_rows.csv",
    "region_edge_rows.csv",
    "region_node_rows.csv",
}

MAX_FILE_BYTES = 32 * 1024 * 1024


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
    rel_parts = set(rel_path.parts)
    if rel_parts & EXCLUDED_PARTS:
        return False, "excluded_path_part"
    if path.name in EXCLUDED_BASENAMES:
        return False, "excluded_large_table"
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
    packet_dir: Path,
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
    dst = packet_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(rel)
    copied_seen.add(rel)


def copy_path(
    src_rel: str,
    packet_dir: Path,
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
        copy_file(src, packet_dir, copied, copied_seen, excluded)
        return

    for dirpath, dirnames, filenames in os.walk(src):
        root = Path(dirpath)
        kept_dirs: list[str] = []
        for dirname in dirnames:
            candidate = root / dirname
            rel_parts = set(candidate.resolve().relative_to(REPO_ROOT).parts)
            if rel_parts & EXCLUDED_PARTS:
                excluded.append({"path": repo_rel(candidate), "reason": "excluded_directory_pruned", "size_bytes": None})
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            path = root / filename
            if path.is_file():
                copy_file(path, packet_dir, copied, copied_seen, excluded)


def packet_files(packet_dir: Path) -> list[Path]:
    return sorted(path for path in packet_dir.rglob("*") if path.is_file())


def write_git_context(packet_dir: Path) -> None:
    write_text(packet_dir / "GIT_STATUS_SHORT.txt", run_text(["git", "status", "--short"]))
    diff_paths = [path for path in [*DOC_PATHS, *CODE_PATHS] if (REPO_ROOT / path).exists()]
    if diff_paths:
        write_text(packet_dir / "SCOPED_GIT_DIFF.patch", run_text(["git", "diff", "--", *diff_paths]))
    else:
        write_text(packet_dir / "SCOPED_GIT_DIFF.patch", "no selected diff paths exist\n")


def write_compile_check(packet_dir: Path) -> None:
    py_paths = [path for path in CODE_PATHS if path.endswith(".py") and (REPO_ROOT / path).is_file()]
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


def selected_json(path: str) -> Any:
    full = AUDIT_ROOT / path
    return {"path": repo_rel(full), "exists": full.exists(), "content": read_json(full) if full.exists() else None}


def write_phase_snapshot(packet_dir: Path) -> None:
    snapshot = {
        "v92_final_decision": selected_json("v92_phase9_casebook/final_decision.json"),
        "v92_phase0": selected_json("v92_phase0_mv_ap_contract/summary.json"),
        "v92_phase2": selected_json("v92_phase2_d4rt_sufficiency/summary.json"),
        "v92_phase3_hr2": selected_json("v92_phase3_d4rt_highres_hr2_grid16/summary.json"),
        "v92_phase3c": selected_json("v92_phase3c_hr2_uncertainty_readout/summary.json"),
        "v92_phase4": selected_json("v92_phase4_semantic_region_affinity/summary.json"),
        "v92_phase4b": selected_json("v92_phase4b_region_granularity_coarse2/summary.json"),
        "v92_phase5b": selected_json("v92_phase5b_source_container_edge_field/summary.json"),
        "v92_phase5c": selected_json("v92_phase5c_tight_field_repair/summary.json"),
        "v92_phase5d": selected_json("v92_phase5d_score_calibration/summary.json"),
        "v92_phase5e": selected_json("v92_phase5e_coarse2_tight_field/summary.json"),
        "v92_phase6": selected_json("v92_phase6_attribution/summary.json"),
        "v91_phase8": selected_json("v91_phase8_dev_selection/summary.json"),
    }
    write_text(packet_dir / "PHASE_DECISION_SNAPSHOT.json", json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_scope(
    packet_dir: Path,
    copied: list[str],
    missing: list[str],
    excluded: list[dict[str, Any]],
) -> None:
    final_path = AUDIT_ROOT / "v92_phase9_casebook/final_decision.json"
    final = read_json(final_path) if final_path.exists() else {}
    present_v92_dirs = [path for path in V92_ARTIFACT_DIRS if (REPO_ROOT / path).exists()]
    missing_v92_dirs = [path for path in V92_ARTIFACT_DIRS if not (REPO_ROOT / path).exists()]
    lines = [
        "# Stream4D v92 Audit Scope",
        "",
        f"generated_at_local: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Purpose: package the Stream4D v92 diagnostic-first affinity readout core code, logs, final No-Go decision, and compact evidence artifacts.",
        "",
        "Verified status copied from final decision artifact:",
        f"- final_decision: `{final.get('final_decision', 'missing')}`",
        f"- primary_blocker: `{final.get('primary_blocker', 'missing')}`",
        f"- secondary_blocker: `{final.get('secondary_blocker', 'missing')}`",
        f"- can_claim_local_method_success: `{final.get('can_claim_local_method_success', 'missing')}`",
        f"- can_enter_local2history: `{final.get('can_enter_local2history', 'missing')}`",
        f"- holdout_decision: `{final.get('holdout_decision', 'missing')}`",
        f"- da3_branch_decision: `{final.get('da3_branch_decision', 'missing')}`",
        f"- final_decision_path: `{repo_rel(final_path)}`",
        "",
        "Included scope:",
        "- v92 plan, execution log, and retrospective log",
        "- v92 phase builders/runners and support evaluator/materializer scripts imported by those builders",
        "- v92 lightweight artifacts: summaries, decisions, gate/config/metric/failure/casebook CSVs, logs, and hash manifests",
        "- v90/v91 lightweight baseline artifacts used by v92 Phase0 and final-decision comparisons",
        "",
        "Excluded by rule:",
        "- raw data, checkpoints, weights, tensor caches, binary model/data/media/archive outputs",
        "- generated mask directories",
        "- full region graph tables and field-region membership tables",
        "- any selected file larger than 32 MiB",
        "- nested code_audit_pack contents",
        "",
        f"present_v92_artifact_dirs: {len(present_v92_dirs)}",
        f"missing_v92_artifact_dirs: {len(missing_v92_dirs)}",
        f"copied_file_count_before_metadata: {len(copied)}",
        f"missing_declared_path_count: {len(missing)}",
        f"excluded_file_count: {len(excluded)}",
        "",
    ]
    write_text(packet_dir / "AUDIT_SCOPE.md", "\n".join(lines))
    write_text(packet_dir / "MISSING_DECLARED_PATHS.txt", "\n".join(missing) + ("\n" if missing else ""))
    write_text(packet_dir / "MISSING_V92_ARTIFACT_DIRS.txt", "\n".join(missing_v92_dirs) + ("\n" if missing_v92_dirs else ""))
    write_text(packet_dir / "EXCLUDED_FILES.json", json.dumps(excluded, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_manifest(
    packet_dir: Path,
    tag: str,
    copied: list[str],
    missing: list[str],
    excluded: list[dict[str, Any]],
) -> None:
    final_path = AUDIT_ROOT / "v92_phase9_casebook/final_decision.json"
    final = read_json(final_path) if final_path.exists() else {}
    manifest = {
        "schema": "stream4d_v92_code_audit_pack_v1",
        "tag": tag,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Stream4D v92 diagnostic-first affinity readout No-Go code and compact artifact audit packet",
        "status": "final_no_go_packet",
        "final_decision": final.get("final_decision"),
        "primary_blocker": final.get("primary_blocker"),
        "secondary_blocker": final.get("secondary_blocker"),
        "can_claim_local_method_success": final.get("can_claim_local_method_success"),
        "can_enter_local2history": final.get("can_enter_local2history"),
        "holdout_decision": final.get("holdout_decision"),
        "da3_branch_decision": final.get("da3_branch_decision"),
        "key_metrics": final.get("key_metrics"),
        "copied_file_count_before_metadata": len(copied),
        "missing_declared_paths": missing,
        "excluded_file_count": len(excluded),
        "max_file_bytes": MAX_FILE_BYTES,
        "excluded_basenames": sorted(EXCLUDED_BASENAMES),
        "excluded_path_parts": sorted(EXCLUDED_PARTS),
        "selected_doc_paths": DOC_PATHS,
        "selected_code_paths": CODE_PATHS,
        "selected_v92_artifact_dirs": V92_ARTIFACT_DIRS,
        "selected_baseline_artifact_dirs": BASELINE_ARTIFACT_DIRS,
        "key_start_here": [
            "docs/stream4d_v92_diagnostic_first_affinity_readout_detailed_plan.md",
            "docs/stream4d_v92_执行日志.md",
            "docs/stream4d_v92_实验结果复盘.md",
            "Stream3D/outputs/audit/v92_phase9_casebook/final_decision.json",
            "PHASE_DECISION_SNAPSHOT.json",
            "AUDIT_SCOPE.md",
        ],
    }
    write_text(packet_dir / "PACKET_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_filelists(packet_dir: Path) -> dict[str, int]:
    filelist = packet_dir / "PAYLOAD_FILELIST.txt"
    hashes = packet_dir / "PAYLOAD_SHA256SUMS.txt"
    write_text(filelist, "")
    write_text(hashes, "")

    files = [path.relative_to(packet_dir).as_posix() for path in packet_files(packet_dir)]
    write_text(filelist, "\n".join(files) + "\n")

    lines = []
    for path in packet_files(packet_dir):
        rel = path.relative_to(packet_dir).as_posix()
        if rel == "PAYLOAD_SHA256SUMS.txt":
            continue
        lines.append(f"{sha256_file(path)}  {rel}")
    write_text(hashes, "\n".join(lines) + "\n")
    return {"payload_file_count": len(packet_files(packet_dir)), "payload_hash_rows": len(lines)}


def zip_packet(packet_dir: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in packet_files(packet_dir):
            rel = path.relative_to(packet_dir).as_posix()
            arc = f"{packet_dir.name}/{rel}"
            zf.write(path, arc)
            entries.append(arc)
    return sorted(entries)


def strip_top(entry: str) -> str:
    parts = Path(entry).parts
    if len(parts) <= 1:
        return ""
    return Path(*parts[1:]).as_posix()


def validate_payload(packet_dir: Path, zip_path: Path, entries: list[str], tag: str) -> dict[str, Any]:
    zip_entries = sorted(zipfile.ZipFile(zip_path).namelist())
    expected_entries = sorted(entries)
    entry_diff: list[str] = []
    if expected_entries != zip_entries:
        entry_diff.extend([f"missing_in_zip {path}" for path in sorted(set(expected_entries) - set(zip_entries))])
        entry_diff.extend([f"unexpected_in_zip {path}" for path in sorted(set(zip_entries) - set(expected_entries))])

    unzip_proc = subprocess.run(
        ["unzip", "-t", zip_path.name],
        cwd=PACK_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    extract_dir = PACK_ROOT / f"{tag}_extract_check"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    extract_proc = subprocess.run(
        ["unzip", "-q", zip_path.name, "-d", extract_dir.name],
        cwd=PACK_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    extracted_packet = extract_dir / packet_dir.name
    payload_proc = subprocess.run(
        ["sha256sum", "-c", "PAYLOAD_SHA256SUMS.txt"],
        cwd=extracted_packet if extracted_packet.exists() else extract_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    shutil.rmtree(extract_dir)

    bad_entries: list[str] = []
    bad_suffixes = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".pickle", ".bin", ".png", ".jpg", ".jpeg", ".mp4", ".mov", ".avi", ".zip", ".gz", ".tar"}
    bad_basenames = set(EXCLUDED_BASENAMES)
    for entry in zip_entries:
        rel = strip_top(entry)
        parts = set(Path(rel).parts)
        suffix = Path(rel).suffix.lower()
        if (parts & EXCLUDED_PARTS) or Path(rel).name in bad_basenames or suffix in bad_suffixes:
            bad_entries.append(entry)

    zip_sha = sha256_file(zip_path)
    side = PACK_ROOT / tag
    write_text(Path(f"{side}.zip.sha256"), f"{zip_sha}  {zip_path.name}\n")
    write_text(Path(f"{side}.zip_entries.txt"), "\n".join(zip_entries) + "\n")
    write_text(Path(f"{side}.entry_diff.txt"), "\n".join(entry_diff) + ("\n" if entry_diff else ""))
    write_text(
        Path(f"{side}.unzip_test.txt"),
        f"$ unzip -t {zip_path.name}\nreturncode={unzip_proc.returncode}\n\n{unzip_proc.stdout}",
    )
    write_text(
        Path(f"{side}.payload_sha256_check.txt"),
        f"$ unzip -q {zip_path.name} -d {extract_dir.name}\nreturncode={extract_proc.returncode}\n\n"
        f"$ sha256sum -c PAYLOAD_SHA256SUMS.txt\nreturncode={payload_proc.returncode}\n\n{payload_proc.stdout}",
    )
    write_text(
        Path(f"{side}.exclusion_check.txt"),
        "bad_entries:\n" + ("\n".join(bad_entries) if bad_entries else "none") + "\n",
    )

    summary = {
        "tag": tag,
        "zip_path": repo_rel(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": zip_sha,
        "zip_entry_count": len(zip_entries),
        "entry_parity_ok": not entry_diff,
        "entry_diff_count": len(entry_diff),
        "unzip_test_returncode": unzip_proc.returncode,
        "extract_returncode": extract_proc.returncode,
        "payload_sha256_check_returncode": payload_proc.returncode,
        "exclusion_check_bad_entry_count": len(bad_entries),
    }
    write_text(Path(f"{side}.validation_summary.txt"), "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n")
    return summary


def build(tag_override: str | None = None) -> dict[str, Any]:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    tag = tag_override or f"{TAG_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not tag.startswith(f"{TAG_PREFIX}_"):
        raise ValueError(f"tag must start with {TAG_PREFIX}_")
    packet_dir = PACK_ROOT / tag
    zip_path = PACK_ROOT / f"{tag}.zip"
    if packet_dir.exists():
        shutil.rmtree(packet_dir)
    if zip_path.exists():
        zip_path.unlink()
    packet_dir.mkdir(parents=True)

    copied: list[str] = []
    copied_seen: set[str] = set()
    missing: list[str] = []
    excluded: list[dict[str, Any]] = []

    for src in [*DOC_PATHS, *CODE_PATHS, *V92_ARTIFACT_DIRS, *BASELINE_ARTIFACT_DIRS]:
        copy_path(src, packet_dir, copied, copied_seen, missing, excluded)

    write_git_context(packet_dir)
    write_compile_check(packet_dir)
    write_phase_snapshot(packet_dir)
    write_scope(packet_dir, copied, missing, excluded)
    write_manifest(packet_dir, tag, copied, missing, excluded)
    counts = write_filelists(packet_dir)
    entries = zip_packet(packet_dir, zip_path)
    validation = validate_payload(packet_dir, zip_path, entries, tag)

    build_summary = {
        "schema": "stream4d_v92_code_audit_pack_v1",
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
    write_text(PACK_ROOT / ".latest_stream4d_v92_pack_tag", tag + "\n")
    return build_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Optional fixed output tag for reproducible package naming.")
    args = parser.parse_args()
    summary = build(args.tag)
    json.dump(summary, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

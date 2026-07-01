#!/usr/bin/env python3
"""Build a compact Stream4D v93 code/artifact audit packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

TAG_PREFIX = "stream4d_v93_boundary_affinity_core_audit"

DOC_PATHS = [
    "docs/stream4d_v93_boundary_aware_affinity_field_readout_plan.md",
    "docs/stream4d_v93_执行日志.md",
    "docs/stream4d_v93_实验结果复盘.md",
]

CODE_PATHS = [
    "tools/build_stream4d_v93_code_audit_pack.py",
    "Stream3D/tools/build_v93_phase0_contract.py",
    "Stream3D/tools/build_v93_phase1_source_edge_registry.py",
    "Stream3D/tools/build_v93_phase2_d4rt_edge_sampling_diagnostic.py",
    "Stream3D/tools/build_v93_phase3_region_edge_graph.py",
    "Stream3D/tools/build_v93_phase4_cue_isolation.py",
    "Stream3D/tools/build_v93_phase4_edge_only_materialization.py",
    "Stream3D/tools/build_v93_phase5_boundary_affinity_field.py",
    "Stream3D/tools/build_v93_phase5b_unknown_background_field.py",
    "Stream3D/tools/verify_v93_phase5_triton_kernels.py",
    "Stream3D/tools/run_v93_d4rt_window_adaptive_recompute.py",
    "Stream3D/tools/build_v93_phase7_adaptive_d4rt_sampling_summary.py",
    "Stream3D/tools/diagnose_v93_phase7_density_readout_gap.py",
    "Stream3D/tools/build_v93_da3_conditional_readiness.py",
    "Stream3D/tools/build_v93_final_decision.py",
    # Supporting code used by v93 for MV_AP and v92-backed cue rows.
    "Stream3D/tools/run_v65_scene_multiview_ap.py",
    "Stream3D/tools/run_v65_d4rt_stride_overlap_geometry.py",
    "Stream3D/tools/run_v92_d4rt_window_highres_recompute.py",
    "Stream3D/tools/build_v92_phase3_highres_bridge.py",
    "Stream3D/tools/run_v92_phase3_hr1_same_readout_adaptive.py",
    "Stream3D/tools/build_v92_phase5_source_container_field.py",
    "Stream3D/tools/build_v92_phase5d_score_calibration.py",
    "Stream3D/tools/build_v92_phase6_attribution.py",
    "Stream3D/tools/build_v92_common_artifact_closure.py",
    "Stream3D/tools/run_v91_phase4_adaptive_uncertainty_materialization.py",
    "Stream3D/tools/build_v91_final_decision.py",
    "Stream3D/tools/build_v91_radio_mask_features.py",
    "Stream3D/tools/merge_v91_radio_mask_feature_stores.py",
    "Stream3D/tools/diagnose_v91_radio_feature_store_quality.py",
]

V93_ARTIFACT_DIRS = [
    "Stream3D/outputs/audit/v93_phase0_contract",
    "Stream3D/outputs/audit/v93_phase1_source_edge_registry",
    "Stream3D/outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic",
    "Stream3D/outputs/audit/v93_phase3_region_edge_graph",
    "Stream3D/outputs/audit/v93_phase4_cue_isolation",
    "Stream3D/outputs/audit/v93_phase4_edge_only_materialization",
    "Stream3D/outputs/audit/v93_phase5_boundary_affinity_field",
    "Stream3D/outputs/audit/v93_phase5b_unknown_background_field",
    "Stream3D/outputs/audit/v93_phase5_triton_torch_equivalence_check",
    "Stream3D/outputs/audit/v93_phase5_triton_kernel_validation",
    "Stream3D/outputs/audit/v93_phase7_adaptive_d4rt_recompute/A512_adaptive_edge_conflict",
    "Stream3D/outputs/audit/v93_phase7_adaptive_d4rt_sampling",
    "Stream3D/outputs/audit/v93_phase7_A512_same_readout_adaptive_materialization",
    "Stream3D/outputs/audit/v93_phase7_density_readout_gap",
    "Stream3D/outputs/audit/v93_phase5_boundary_affinity_field_A512",
    "Stream3D/outputs/audit/v93_da3_conditional_readiness",
    "Stream3D/outputs/audit/v93_final_decision",
]

SUPPORT_ARTIFACT_DIRS = [
    "Stream3D/outputs/audit/v92_phase5_source_container_field",
    "Stream3D/outputs/audit/v92_phase5b_source_container_edge_field",
    "Stream3D/outputs/audit/v92_phase5c_tight_field_repair",
    "Stream3D/outputs/audit/v92_phase5d_score_calibration",
    "Stream3D/outputs/audit/v92_phase6_attribution",
    "Stream3D/outputs/audit/v91_phase0_mv_ap_contract",
    "Stream3D/outputs/audit/v91_phase8_dev_selection",
    "Stream3D/outputs/audit/v90_phase0_mv_ap_contract",
    "Stream3D/outputs/audit/v89_phase0_mv_ap_contract",
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
    "adaptive_radius_rows.csv",
    "d4rt_edge_band_support_rows.csv",
    "d4rt_observation_rows.csv",
    "d4rt_support_rows.csv",
    "d4rt_source_support_rows.csv",
    "feature_availability_rows.csv",
    "field_region_membership_rows.csv",
    "generated_mask_rows.csv",
    "highres_carrier_observation_rows.csv",
    "highres_incidence_rows.csv",
    "highres_native_carrier_support_rows.csv",
    "highres_quality_proxy_rows.csv",
    "mask_edge_hypothesis_rows.csv",
    "mv_object_frame_mask_rows.csv",
    "mv_object_rows.csv",
    "object_container_link_rows.csv",
    "pre_filter_eval_rows.csv",
    "region_edge_rows.csv",
    "region_feature_rows.csv",
    "region_node_rows.csv",
    "sampling_query_rows.csv",
    "scored_frame_mask_rows.csv",
    "selected_masklet_rows.csv",
    "source_container_rows.csv",
    "source_selection_rows.csv",
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
    except Exception as exc:  # noqa: BLE001 - malformed evidence should be visible.
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
                excluded.append(
                    {"path": repo_rel(candidate), "reason": "excluded_directory_pruned", "size_bytes": None}
                )
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            path = root / filename
            if path.is_file():
                copy_file(path, packet_dir, copied, copied_seen, excluded)


def packet_files(packet_dir: Path) -> list[Path]:
    return sorted(path for path in packet_dir.rglob("*") if path.is_file())


def selected_json(path: str) -> Any:
    full = AUDIT_ROOT / path
    return {"path": repo_rel(full), "exists": full.exists(), "content": read_json(full) if full.exists() else None}


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


def write_phase_snapshot(packet_dir: Path) -> None:
    snapshot = {
        "v93_phase0": selected_json("v93_phase0_contract/summary.json"),
        "v93_phase1": selected_json("v93_phase1_source_edge_registry/summary.json"),
        "v93_phase2": selected_json("v93_phase2_d4rt_edge_sampling_diagnostic/summary.json"),
        "v93_phase3": selected_json("v93_phase3_region_edge_graph/summary.json"),
        "v93_phase4": selected_json("v93_phase4_cue_isolation/summary.json"),
        "v93_phase4_edge_only": selected_json("v93_phase4_edge_only_materialization/summary.json"),
        "v93_phase5_hr2_field": selected_json("v93_phase5_boundary_affinity_field/summary.json"),
        "v93_phase5b_unknown_background_field": selected_json("v93_phase5b_unknown_background_field/summary.json"),
        "v93_phase5_triton_torch_equivalence": selected_json("v93_phase5_triton_torch_equivalence_check/report.json"),
        "v93_phase5_triton_kernel_validation": selected_json("v93_phase5_triton_kernel_validation/report.json"),
        "v93_phase7_adaptive_d4rt_sampling": selected_json("v93_phase7_adaptive_d4rt_sampling/summary.json"),
        "v93_phase7_density_readout_gap": selected_json("v93_phase7_density_readout_gap/summary.json"),
        "v93_phase5_a512_field": selected_json("v93_phase5_boundary_affinity_field_A512/summary.json"),
        "v93_da3_conditional_readiness": selected_json("v93_da3_conditional_readiness/summary.json"),
        "v93_final_decision": selected_json("v93_final_decision/summary.json"),
        "v92_phase5_source_container_field": selected_json("v92_phase5_source_container_field/summary.json"),
        "v92_phase5b_source_container_edge_field": selected_json("v92_phase5b_source_container_edge_field/summary.json"),
        "v92_phase5c_tight_field_repair": selected_json("v92_phase5c_tight_field_repair/summary.json"),
        "v92_phase6_attribution": selected_json("v92_phase6_attribution/summary.json"),
        "v91_phase8_dev_selection": selected_json("v91_phase8_dev_selection/summary.json"),
    }
    write_text(packet_dir / "PHASE_DECISION_SNAPSHOT.json", json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def current_final_status() -> dict[str, Any]:
    final_path = AUDIT_ROOT / "v93_final_decision/summary.json"
    final = read_json(final_path) if final_path.exists() else {}
    if final:
        return {
            "decision": final.get("decision"),
            "goal_achieved": final.get("goal_achieved"),
            "dev_gate_pass": final.get("dev_gate_pass"),
            "freeze_config_written": final.get("freeze_config_written"),
            "holdout_executed": final.get("holdout_executed"),
            "required_MV_AP_window": final.get("required_MV_AP_window"),
            "required_MV_AP50_window": final.get("required_MV_AP50_window"),
            "best_attempt": final.get("best_attempt"),
            "triton_validated": final.get("triton_validated"),
            "summary_path": repo_rel(final_path),
        }

    phase4_path = AUDIT_ROOT / "v93_phase4_cue_isolation/summary.json"
    failure_path = AUDIT_ROOT / "v93_phase4_cue_isolation/variant_failure_rows.csv"
    phase4 = read_json(phase4_path) if phase4_path.exists() else {}
    failure_rows: list[dict[str, str]] = []
    if failure_path.exists():
        with failure_path.open("r", encoding="utf-8", newline="") as handle:
            failure_rows = list(csv.DictReader(handle))
    missing_variants = sorted({row.get("variant_id", "") for row in failure_rows if row.get("variant_id")})
    repair_directions = sorted({row.get("repair_direction", "") for row in failure_rows if row.get("repair_direction")})
    return {
        "decision": phase4.get("decision"),
        "phase4_complete": phase4.get("phase4_complete"),
        "missing_edge_only_variant_count": phase4.get("missing_edge_only_variant_count"),
        "missing_edge_only_variants": missing_variants,
        "available_cue_variant_count": phase4.get("available_cue_variant_count"),
        "best_control_MV_AP_window": phase4.get("best_control_MV_AP_window"),
        "whole_source_MV_AP_window": phase4.get("whole_source_MV_AP_window"),
        "D4RT_plus_RADIO_MV_AP_window": phase4.get("D4RT_plus_RADIO_MV_AP_window"),
        "repair_direction": phase4.get("repair_direction") or (repair_directions[0] if len(repair_directions) == 1 else None),
        "repair_direction_source": repo_rel(failure_path) if repair_directions else None,
        "summary_path": repo_rel(phase4_path),
    }


def write_scope(
    packet_dir: Path,
    copied: list[str],
    missing: list[str],
    excluded: list[dict[str, Any]],
) -> None:
    status = current_final_status()
    present_v93_dirs = [path for path in V93_ARTIFACT_DIRS if (REPO_ROOT / path).exists()]
    missing_v93_dirs = [path for path in V93_ARTIFACT_DIRS if not (REPO_ROOT / path).exists()]
    present_support_dirs = [path for path in SUPPORT_ARTIFACT_DIRS if (REPO_ROOT / path).exists()]
    missing_support_dirs = [path for path in SUPPORT_ARTIFACT_DIRS if not (REPO_ROOT / path).exists()]
    lines = [
        "# Stream4D v93 Audit Scope",
        "",
        f"generated_at_local: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Purpose: package the current Stream4D v93 boundary-aware affinity-field readout core code, logs, final No-Go decision, and compact evidence artifacts.",
        "",
        "Current verified status copied from final decision artifact when available:",
        f"- decision: `{status.get('decision', 'missing')}`",
        f"- goal_achieved: `{status.get('goal_achieved', 'missing')}`",
        f"- dev_gate_pass: `{status.get('dev_gate_pass', 'missing')}`",
        f"- freeze_config_written: `{status.get('freeze_config_written', 'missing')}`",
        f"- holdout_executed: `{status.get('holdout_executed', 'missing')}`",
        f"- required_MV_AP_window: `{status.get('required_MV_AP_window', 'missing')}`",
        f"- required_MV_AP50_window: `{status.get('required_MV_AP50_window', 'missing')}`",
        f"- best_attempt: `{status.get('best_attempt', 'missing')}`",
        f"- triton_validated: `{status.get('triton_validated', 'missing')}`",
        f"- final_summary_path: `{status.get('summary_path', 'missing')}`",
        "",
        "Included scope:",
        "- v93 plan, execution log, and retrospective log",
        "- v93 Phase0-Phase7 builders, GPU/Triton Phase5 code, Triton validation code, and direct evaluator/materializer support code",
        "- v93 lightweight artifacts: summaries, gate/config/metric/failure CSVs, logs, and hash manifests",
        "- v93 final decision artifact and Phase5/Phase7 repair evidence",
        "- v92/v91/v90/v89 lightweight support artifacts used by v93 Phase0/Phase4/Phase7 comparisons",
        "",
        "Excluded by rule:",
        "- raw data, checkpoints, weights, tensor caches, binary model/data/media/archive outputs",
        "- generated mask directories",
        "- full edge/region/source mega tables explicitly listed in EXCLUDED_BASENAMES",
        "- any selected file larger than 32 MiB",
        "- nested code_audit_pack contents",
        "",
        "Important status note:",
        "- This packet is a No-Go audit packet, not a success packet. Phase5/Phase7 repairs were executed and did not meet dev local/control gates.",
        "",
        f"present_v93_artifact_dirs: {len(present_v93_dirs)}",
        f"missing_v93_artifact_dirs: {len(missing_v93_dirs)}",
        f"present_support_artifact_dirs: {len(present_support_dirs)}",
        f"missing_support_artifact_dirs: {len(missing_support_dirs)}",
        f"copied_file_count_before_metadata: {len(copied)}",
        f"missing_declared_path_count: {len(missing)}",
        f"excluded_file_count: {len(excluded)}",
        "",
    ]
    write_text(packet_dir / "AUDIT_SCOPE.md", "\n".join(lines))
    write_text(packet_dir / "MISSING_DECLARED_PATHS.txt", "\n".join(missing) + ("\n" if missing else ""))
    write_text(packet_dir / "MISSING_V93_ARTIFACT_DIRS.txt", "\n".join(missing_v93_dirs) + ("\n" if missing_v93_dirs else ""))
    write_text(packet_dir / "MISSING_SUPPORT_ARTIFACT_DIRS.txt", "\n".join(missing_support_dirs) + ("\n" if missing_support_dirs else ""))
    write_text(packet_dir / "EXCLUDED_FILES.json", json.dumps(excluded, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_manifest(
    packet_dir: Path,
    tag: str,
    copied: list[str],
    missing: list[str],
    excluded: list[dict[str, Any]],
) -> None:
    status = current_final_status()
    manifest = {
        "schema": "stream4d_v93_code_audit_pack_v2",
        "tag": tag,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Stream4D v93 boundary-aware affinity-field readout final No-Go code and compact artifact audit packet",
        "status": "no_go_v93_dev_gate_failed_after_phase5_phase7_repairs",
        "current_final_status": status,
        "copied_file_count_before_metadata": len(copied),
        "missing_declared_paths": missing,
        "excluded_file_count": len(excluded),
        "max_file_bytes": MAX_FILE_BYTES,
        "excluded_basenames": sorted(EXCLUDED_BASENAMES),
        "excluded_path_parts": sorted(EXCLUDED_PARTS),
        "selected_doc_paths": DOC_PATHS,
        "selected_code_paths": CODE_PATHS,
        "selected_v93_artifact_dirs": V93_ARTIFACT_DIRS,
        "selected_support_artifact_dirs": SUPPORT_ARTIFACT_DIRS,
        "key_start_here": [
            "docs/stream4d_v93_boundary_aware_affinity_field_readout_plan.md",
            "docs/stream4d_v93_执行日志.md",
            "docs/stream4d_v93_实验结果复盘.md",
            "Stream3D/outputs/audit/v93_final_decision/summary.json",
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
    bad_suffixes = set(EXCLUDED_SUFFIXES)
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

    for src in [*DOC_PATHS, *CODE_PATHS, *V93_ARTIFACT_DIRS, *SUPPORT_ARTIFACT_DIRS]:
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
        "schema": "stream4d_v93_code_audit_pack_v2",
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
    write_text(PACK_ROOT / ".latest_stream4d_v93_pack_tag", tag + "\n")
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

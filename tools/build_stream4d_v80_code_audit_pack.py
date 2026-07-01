#!/usr/bin/env python3
"""Build a compact Stream4D v80 CMAP-AF-L2H code/artifact audit pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


DOC_FILES = [
    Path("docs/stream4d_v80_cmap_af_l2h_revised_critical_plan.md"),
    Path("docs/stream4d_v80_执行日志.md"),
    Path("docs/stream4d_v80_实验结果复盘.md"),
]

CODE_FILES = [
    Path("Stream3D/tools/run_v80_cmap_af_l2h_pipeline.py"),
    Path("tools/build_stream4d_v80_code_audit_pack.py"),
]

BASELINE_FILES = [
    Path("Stream3D/outputs/audit/v71_semantic_features/semantic_summary.json"),
    Path("Stream3D/outputs/audit/v77_final_decision/final_decision.json"),
    Path("Stream3D/outputs/audit/v79_repair_sweep_summary/sweep_summary.json"),
]

LIGHT_EXTS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}

FORBIDDEN_EXTS = {
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}

FORBIDDEN_PARTS = {
    "__pycache__",
    ".git",
    "code_audit_pack",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_cmd(cmd: list[str], cwd: Path, out_file: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    write_text(
        out_file,
        f"$ {' '.join(cmd)}\n"
        f"returncode={proc.returncode}\n\n"
        f"[stdout]\n{proc.stdout}\n\n"
        f"[stderr]\n{proc.stderr}\n",
    )
    return proc


def should_copy_artifact(path: Path, max_artifact_bytes: int, include_large_csv: bool) -> tuple[bool, str]:
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        return False, "forbidden_path_part"
    if path.suffix in FORBIDDEN_EXTS:
        return False, "forbidden_extension"
    if path.suffix not in LIGHT_EXTS:
        return False, "non_light_extension"
    size = path.stat().st_size
    if path.suffix == ".csv" and include_large_csv:
        return True, ""
    if size > max_artifact_bytes:
        return False, f"size_gt_{max_artifact_bytes}"
    return True, ""


def copy_one(src: Path, packet_root: Path, copied: set[Path]) -> bool:
    if not src.is_file() or src in copied:
        return False
    dst = packet_root / src
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.add(src)
    return True


def iter_v80_artifact_files(repo: Path) -> list[Path]:
    audit_root = repo / "Stream3D/outputs/audit"
    if not audit_root.exists():
        return []
    files: list[Path] = []
    for root in sorted(audit_root.glob("v80*")):
        if root.is_dir():
            files.extend(sorted(path for path in root.rglob("*") if path.is_file()))
        elif root.is_file():
            files.append(root)
    return [path.relative_to(repo) for path in files]


def build_run_matrix(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((repo / "Stream3D/outputs/audit").glob("v80_cmap_af_l2h_pipeline_dev_*/pipeline_summary.json")):
        data = read_json(path)
        summaries = data.get("summaries", {})
        final = summaries.get("final", {})
        p1 = summaries.get("phase1", {})
        p2 = summaries.get("phase2", {})
        p3 = summaries.get("phase3", {})
        p4 = summaries.get("phase4", {})
        p5 = summaries.get("phase5", {})
        p6 = summaries.get("phase6", {})
        tag = path.parent.name.replace("v80_cmap_af_l2h_pipeline_dev_", "")
        rows.append(
            {
                "run": tag,
                "pipeline_summary": str(path.relative_to(repo)),
                "final_decision": final.get("final_decision", ""),
                "primary_blocker": final.get("primary_blocker", ""),
                "can_enter_local2history": final.get("can_enter_local2history", ""),
                "best_dev_local_SF50": final.get("best_dev_local_SF50", ""),
                "phase1_topk_recall_under_sketch": p1.get("topk_recall_under_sketch", ""),
                "phase1_cosine_error_p95": p1.get("cosine_error_p95", ""),
                "phase1_broad_collision_mass_ratio": p1.get("broad_collision_mass_ratio", ""),
                "phase1_method_motion_feature_weight": p1.get("method_motion_feature_weight", ""),
                "phase1_semantic_positive_guard": p1.get("semantic_positive_guard", ""),
                "phase1_semantic_disagreement_penalty": p1.get("semantic_disagreement_penalty", ""),
                "phase1_semantic_profile_coverage": p1.get("semantic_profile_coverage", ""),
                "phase2_decision": p2.get("decision", ""),
                "phase2_largest_connected_component_ratio": p2.get("largest_connected_component_ratio", ""),
                "phase2_component_cannot_link_violation_count": p2.get("component_cannot_link_violation_count", ""),
                "phase2_positive_bridge_removed_rate": p2.get("positive_bridge_removed_rate", ""),
                "phase2_semantic_positive_guard": p2.get("semantic_positive_guard", ""),
                "phase2_semantic_positive_rejected_edge_count": p2.get("semantic_positive_rejected_edge_count", ""),
                "phase2_semantic_disagreement_penalized_edge_count": p2.get(
                    "semantic_disagreement_penalized_edge_count", ""
                ),
                "phase3_decision": p3.get("decision", ""),
                "phase3_within_semantic_affinity_minus_semantic_AUC": p3.get(
                    "within_semantic_affinity_minus_semantic_AUC", ""
                ),
                "phase3_motion_visibility_affinity_minus_semantic_AUC": p3.get(
                    "motion_visibility_affinity_minus_semantic_AUC", ""
                ),
                "phase3_secondary_blocker": p3.get("secondary_blocker", ""),
                "phase4_decision": p4.get("decision", ""),
                "phase4_cluster_count_fine": p4.get("cluster_count_fine", ""),
                "phase4_cluster_count_object": p4.get("cluster_count_object", ""),
                "phase4_cluster_count_coarse": p4.get("cluster_count_coarse", ""),
                "phase4_largest_cluster_ratio_object": p4.get("largest_cluster_ratio_object", ""),
                "phase4_object_cluster_temporal_span_mean": p4.get("object_cluster_temporal_span_mean", ""),
                "phase4_single_frame_cluster_rate_object": p4.get("single_frame_cluster_rate_object", ""),
                "phase4_cross_scale_inclusion_rate": p4.get("cross_scale_inclusion_rate", ""),
                "phase4_cross_scale_relation_pass_rate": p4.get("cross_scale_relation_pass_rate", ""),
                "phase4_scale_conflict_rate": p4.get("scale_conflict_rate", ""),
                "phase4_object_parent_merge_applied_count": p4.get("object_parent_merge_applied_count", ""),
                "phase4_object_parent_merge_attempt_count": p4.get("object_parent_merge_attempt_count", ""),
                "phase4_object_parent_merge_pre_count_mean": p4.get("object_parent_merge_pre_count_mean", ""),
                "phase4_object_parent_merge_post_count_mean": p4.get("object_parent_merge_post_count_mean", ""),
                "phase4_object_parent_merge_broad_parent_reject_count": p4.get(
                    "object_parent_merge_broad_parent_reject_count", ""
                ),
                "phase4_object_parent_merge_max_parent_child_count": p4.get(
                    "object_parent_merge_max_parent_child_count", ""
                ),
                "phase4_semantic_positive_guard": p4.get("semantic_positive_guard", ""),
                "phase4_semantic_positive_rejected_edge_count": p4.get("semantic_positive_rejected_edge_count", ""),
                "phase4_semantic_disagreement_penalized_edge_count": p4.get(
                    "semantic_disagreement_penalized_edge_count", ""
                ),
                "phase4_object_mask_ownership_mode": p4.get("object_mask_ownership_mode", ""),
                "phase4_object_mask_ownership_candidate_frame_mask_count": p4.get(
                    "object_mask_ownership_candidate_frame_mask_count", ""
                ),
                "phase4_object_mask_ownership_ambiguous_frame_mask_count": p4.get(
                    "object_mask_ownership_ambiguous_frame_mask_count", ""
                ),
                "phase4_object_mask_ownership_ambiguous_rate": p4.get("object_mask_ownership_ambiguous_rate", ""),
                "phase4_object_mask_ownership_resolved_frame_mask_count": p4.get(
                    "object_mask_ownership_resolved_frame_mask_count", ""
                ),
                "phase4_object_mask_ownership_unresolved_frame_mask_count": p4.get(
                    "object_mask_ownership_unresolved_frame_mask_count", ""
                ),
                "phase4_object_mask_ownership_allowed_candidate_count": p4.get(
                    "object_mask_ownership_allowed_candidate_count", ""
                ),
                "phase4_object_mask_ownership_rejected_candidate_count": p4.get(
                    "object_mask_ownership_rejected_candidate_count", ""
                ),
                "phase4_secondary_blocker": p4.get("secondary_blocker", ""),
                "phase5_decision": p5.get("decision", ""),
                "phase5_adapter_score_mode": p5.get("adapter_score_mode", ""),
                "phase5_adapter_render_kernel": p5.get("adapter_render_kernel", ""),
                "phase5_object_mask_ownership_mode": p5.get("object_mask_ownership_mode", ""),
                "phase5_local_SF50_rendered_adapter": p5.get("local_SF50_rendered_adapter", ""),
                "phase5_local_AP50": p5.get("local_AP50", ""),
                "phase5_local_AP25": p5.get("local_AP25", ""),
                "phase5_GT_best_IoU_mean": p5.get("GT_best_IoU_mean", ""),
                "phase5_v79_best_replay_SF50": p5.get("v79_best_replay_SF50", ""),
                "phase5_adapter_threshold_sensitivity_SF50_range": p5.get(
                    "adapter_threshold_sensitivity_SF50_range", ""
                ),
                "phase5_carrier_F1_vs_pixel_F1_spearman": p5.get("carrier_F1_vs_pixel_F1_spearman", ""),
                "phase5_carrier_pixel_F1_gap_p95": p5.get("carrier_pixel_F1_gap_p95", ""),
                "phase5_projected_support_density_mean": p5.get("projected_support_density_mean", ""),
                "phase5_adapter_density_correction_power": p5.get("adapter_density_correction_power", ""),
                "phase5_adapter_density_reference": p5.get("adapter_density_reference", ""),
                "phase5_adapter_min_projected_density": p5.get("adapter_min_projected_density", ""),
                "phase5_adapter_max_carrier_pixel_f1_gap": p5.get("adapter_max_carrier_pixel_f1_gap", ""),
                "phase5_adapter_ambiguous_mask_policy": p5.get("adapter_ambiguous_mask_policy", ""),
                "phase5_adapter_candidate_frame_mask_count": p5.get("adapter_candidate_frame_mask_count", ""),
                "phase5_adapter_ambiguous_frame_mask_count": p5.get("adapter_ambiguous_frame_mask_count", ""),
                "phase5_adapter_ambiguous_candidate_count": p5.get("adapter_ambiguous_candidate_count", ""),
                "phase5_adapter_ambiguous_rejected_count": p5.get("adapter_ambiguous_rejected_count", ""),
                "phase5_adapter_candidate_frame_mask_conflict_rate": p5.get(
                    "adapter_candidate_frame_mask_conflict_rate", ""
                ),
                "phase5_duplicate_frame_mask_conflict_rate": p5.get("duplicate_frame_mask_conflict_rate", ""),
                "phase5_adapter_fine_child_support": p5.get("adapter_fine_child_support", ""),
                "phase5_adapter_fine_child_min_inclusion": p5.get("adapter_fine_child_min_inclusion", ""),
                "phase5_adapter_fine_child_min_frames": p5.get("adapter_fine_child_min_frames", ""),
                "phase5_adapter_fine_child_min_carriers": p5.get("adapter_fine_child_min_carriers", ""),
                "phase5_contained_fine_child_support_rate": p5.get("contained_fine_child_support_rate", ""),
                "phase5_broad_adapter_rate": p5.get("broad_adapter_rate", ""),
                "phase5_secondary_blocker": p5.get("secondary_blocker", ""),
                "phase6_decision": p6.get("decision", ""),
                "phase6_control_SF50": p6.get("control_SF50", ""),
                "phase6_method_SF50": p6.get("method_SF50", ""),
                "phase6_control_rows_source_diagnostic_only": p6.get("control_rows_source_diagnostic_only", ""),
                "phase6_control_uses_GT_for_prediction": p6.get("control_uses_GT_for_prediction", ""),
                "phase6_control_uses_eval_metric_for_selection": p6.get(
                    "control_uses_eval_metric_for_selection", ""
                ),
                "phase6_control_uses_future_information": p6.get("control_uses_future_information", ""),
                "phase6_secondary_blocker": p6.get("secondary_blocker", ""),
                "known_issue": "phase3_same_semantic_filter_bug_pre_fix" if tag == "r1" else "",
            }
        )
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_text(path, "")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_packet_summary(packet_root: Path, run_matrix: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    best = None
    for row in run_matrix:
        val = row.get("best_dev_local_SF50")
        if isinstance(val, (int, float)) and (best is None or val > best.get("best_dev_local_SF50", -1)):
            best = row
    lines = [
        "# Stream4D v80 CMAP-AF-L2H Audit Pack",
        "",
        "Status: current run did not achieve the v80 objective.",
        "",
        "Key facts:",
        "",
        "- local2history was not entered in method mode for any packaged dev run.",
        "- Holdout was blocked by dev local quality; no frozen holdout success is claimed.",
        "- Phase5 adapter/local quality remains the main blocker in the best dev run.",
        "- r1 is retained for provenance but marked as pre-fix for the phase3 same-semantic filtering bug.",
    ]
    if best is not None:
        lines.extend(
            [
                "",
                "Best packaged dev run by measured local_SF50:",
                "",
                f"- run: {best['run']}",
                f"- final_decision: {best['final_decision']}",
                f"- best_dev_local_SF50: {best['best_dev_local_SF50']}",
                f"- phase5_local_AP50: {best['phase5_local_AP50']}",
                f"- phase5_GT_best_IoU_mean: {best['phase5_GT_best_IoU_mean']}",
                f"- primary_blocker: {best['primary_blocker']}",
            ]
        )
    lines.extend(
        [
            "",
            "Included evidence:",
            "",
            "- v80 plan, execution log, and recap log.",
            "- canonical v80 pipeline runner and this pack builder.",
            "- v80 JSON/CSV/TXT artifacts up to the configured artifact size limit.",
            "- baseline summary files from v71 semantic features, v77 final decision, and v79 repair sweep.",
            "- run matrix extracted from packaged pipeline_summary.json files.",
            "",
            "Large skipped artifacts:",
            "",
            f"- {len(skipped)} files were skipped; see EXCLUDED_PATHS.txt for exact path, size, and reason.",
        ]
    )
    write_text(packet_root / "PACK_SUMMARY.md", "\n".join(lines) + "\n")


def list_payload_files(packet_root: Path) -> list[Path]:
    return sorted(path for path in packet_root.rglob("*") if path.is_file())


def write_payload_sidecars(packet_root: Path) -> dict[str, int]:
    filelist = packet_root / "PAYLOAD_FILELIST.txt"
    hashes = packet_root / "PAYLOAD_SHA256SUMS.txt"
    write_text(filelist, "")
    write_text(hashes, "")
    files = list_payload_files(packet_root)
    write_text(filelist, "\n".join(str(path.relative_to(packet_root)) for path in files) + "\n")
    lines = []
    for path in files:
        rel = path.relative_to(packet_root)
        if rel in {Path("PAYLOAD_FILELIST.txt"), Path("PAYLOAD_SHA256SUMS.txt")}:
            continue
        lines.append(f"{sha256_file(path)}  {rel}")
    write_text(hashes, "\n".join(lines) + "\n")
    return {"payload_files": len(files), "payload_hash_rows": len(lines)}


def zip_packet(packet_root: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in list_payload_files(packet_root):
            arcname = Path(packet_root.name) / path.relative_to(packet_root)
            zf.write(path, arcname)
            entries.append(str(arcname))
    return sorted(entries)


def strip_top(entry: str) -> str:
    parts = Path(entry).parts
    return str(Path(*parts[1:])) if len(parts) > 1 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="code_audit_pack")
    parser.add_argument("--tag", default="")
    parser.add_argument("--max-artifact-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--include-large-csv", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    output_dir = repo / args.output_dir
    tag = args.tag or f"stream4d_v80_cmap_af_l2h_current_state_core_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    packet_root = output_dir / tag
    zip_path = output_dir / f"{tag}.zip"
    if packet_root.exists() or zip_path.exists():
        raise SystemExit(f"Refusing to overwrite existing packet: {tag}")
    packet_root.mkdir(parents=True, exist_ok=False)

    copied: set[Path] = set()
    missing: list[str] = []
    skipped: list[dict[str, Any]] = []

    for rel in DOC_FILES + CODE_FILES + BASELINE_FILES:
        if not (repo / rel).is_file():
            missing.append(str(rel))
            continue
        copy_one(rel, packet_root, copied)

    for rel in iter_v80_artifact_files(repo):
        src = repo / rel
        ok, reason = should_copy_artifact(src, args.max_artifact_bytes, args.include_large_csv)
        if ok:
            copy_one(rel, packet_root, copied)
        else:
            skipped.append({"path": str(rel), "size_bytes": src.stat().st_size, "reason": reason})

    run_matrix = build_run_matrix(repo)
    write_text(packet_root / "stream4d_v80_dev_run_matrix.json", json.dumps(run_matrix, indent=2, sort_keys=True) + "\n")
    write_csv_rows(packet_root / "stream4d_v80_dev_run_matrix.csv", run_matrix)
    write_packet_summary(packet_root, run_matrix, skipped)

    write_text(
        packet_root / "EXCLUDED_PATHS.txt",
        "\n".join(f"{row['size_bytes']}\t{row['reason']}\t{row['path']}" for row in skipped) + ("\n" if skipped else ""),
    )
    write_text(packet_root / "MISSING_PATHS.txt", "\n".join(missing) + ("\n" if missing else ""))
    run_cmd(["git", "status", "--short"], repo, packet_root / "git_status_short.txt")

    manifest = {
        "schema": "stream4d_v80_cmap_af_l2h_code_audit_pack_v1",
        "tag": tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "build_command": " ".join([sys.executable, *sys.argv]),
        "build_cwd": str(repo),
        "status": "current_state_no_go_review_packet",
        "objective_complete": False,
        "local2history_method_mode_entered": False,
        "holdout_success_claimed": False,
        "run_count": len(run_matrix),
        "copied_file_count_before_sidecars": len(copied),
        "skipped_file_count": len(skipped),
        "missing_path_count": len(missing),
        "max_artifact_bytes": args.max_artifact_bytes,
        "include_large_csv": args.include_large_csv,
    }
    write_text(packet_root / "MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sidecar_counts = write_payload_sidecars(packet_root)
    entries = zip_packet(packet_root, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        zip_entries = sorted(info.filename for info in zf.infolist() if not info.is_dir())
    if bad:
        raise SystemExit(f"zip integrity test failed at {bad}")

    expected = sorted(f"{packet_root.name}/{path.relative_to(packet_root)}" for path in list_payload_files(packet_root))
    missing_entries = sorted(set(expected) - set(zip_entries))
    extra_entries = sorted(set(zip_entries) - set(expected))
    entry_diff_path = output_dir / f"{tag}.entry_diff.txt"
    write_text(
        entry_diff_path,
        "missing_entries:\n"
        + "\n".join(missing_entries)
        + "\nextra_entries:\n"
        + "\n".join(extra_entries)
        + "\n",
    )

    payload_check_path = output_dir / f"{tag}.payload_sha256_check.txt"
    lines = []
    hash_file = packet_root / "PAYLOAD_SHA256SUMS.txt"
    for raw in hash_file.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected_hash, rel = raw.split("  ", 1)
        current = sha256_file(packet_root / rel)
        lines.append(f"{'OK' if current == expected_hash else 'FAIL'}  {rel}")
    write_text(payload_check_path, "\n".join(lines) + "\n")

    unzip_test_path = output_dir / f"{tag}.unzip_test.txt"
    write_text(unzip_test_path, f"zipfile.testzip={bad}\nentry_count={len(zip_entries)}\n")
    zip_sha_path = output_dir / f"{tag}.zip.sha256"
    write_text(zip_sha_path, f"{sha256_file(zip_path)}  {zip_path.name}\n")

    build_summary = {
        **manifest,
        **sidecar_counts,
        "archive": str(zip_path.relative_to(repo)),
        "archive_sha256": sha256_file(zip_path),
        "entry_count": len(entries),
        "entry_diff_file": str(entry_diff_path.relative_to(repo)),
        "payload_sha256_check_file": str(payload_check_path.relative_to(repo)),
        "unzip_test_file": str(unzip_test_path.relative_to(repo)),
        "zip_sha256_file": str(zip_sha_path.relative_to(repo)),
    }
    write_text(output_dir / f"{tag}.build_summary.json", json.dumps(build_summary, indent=2, sort_keys=True) + "\n")
    write_text(output_dir / ".latest_stream4d_v80_pack_tag", tag + "\n")
    print(json.dumps(build_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

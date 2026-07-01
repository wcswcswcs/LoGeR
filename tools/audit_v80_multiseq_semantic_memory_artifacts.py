#!/usr/bin/env python3
"""Phase 0 multi-sequence artifact audit for ACL2 v80-TF.

This tool is audit-only. It checks existing artifacts for KITTI good/bad case
mining and memory-body action eligibility. It never synthesizes missing metrics
or treats an unavailable path as usable.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_SEQS = ("00", "01", "02", "05", "08")
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_KITTI_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset")
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase0_multiseq_artifact_audit"
)

STABLE_WORDS = (
    "building",
    "house",
    "wall",
    "fence",
    "handrail_or_fence",
    "pole",
    "traffic sign",
    "traffic light",
    "bridge",
    "construction",
    "billboard",
    "pillar",
    "stair",
)
DYNAMIC_WORDS = ("car", "person", "rider", "bicycle", "motorcycle", "bus", "truck", "train", "dog")
LOWTRUST_WORDS = ("tree", "grass", "vegetation", "mountain", "terrain", "void", "unknown", "plant")
CONTEXT_WORDS = ("sky", "road", "ground", "sidewalk", "path", "vegetation")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_seqs(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return [part.strip().zfill(2) for part in text.split(",") if part.strip()]
    return [str(part).zfill(2) for part in text]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = list(fieldnames or [])
    if not keys:
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def positive_number(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def ttt_action_evidence_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if "ttt" in str(row.get("hmc_commit_mode", "")).lower():
        tags.append("hmc_commit_mode")
    if boolish(row.get("prior_ttt_write_present")):
        tags.append("prior_ttt_write_present")
    for key in (
        "memory_ttt_max_rel_diff",
        "memory_ttt_mean_rel_diff",
        "memory_ttt_w0_max_rel_diff",
        "memory_ttt_w0_mean_rel_diff",
        "memory_ttt_w1_max_rel_diff",
        "memory_ttt_w1_mean_rel_diff",
        "memory_ttt_w2_max_rel_diff",
        "memory_ttt_w2_mean_rel_diff",
    ):
        if positive_number(row.get(key)):
            tags.append(key)
            break
    side_effect = row.get("memory_side_effect")
    if isinstance(side_effect, dict):
        ttt_diff = side_effect.get("ttt_state_diff")
        if isinstance(ttt_diff, dict) and (
            positive_number(ttt_diff.get("max_abs_diff"))
            or positive_number(ttt_diff.get("max_rel_diff"))
            or positive_number(ttt_diff.get("mean_abs_diff"))
            or positive_number(ttt_diff.get("mean_rel_diff"))
        ):
            tags.append("memory_side_effect.ttt_state_diff")
    return tags


def scan_ttt_action_evidence(
    files: Sequence[Path],
    repo: Path,
    max_files: int = 48,
    max_lines_per_file: int = 16,
) -> dict[str, Any]:
    files_scanned = 0
    lines_scanned = 0
    evidence_lines = 0
    evidence_files: list[Path] = []
    evidence_tags: list[str] = []
    for path in files[:max_files]:
        files_scanned += 1
        file_has_evidence = False
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with handle:
            for idx, line in enumerate(handle):
                if idx >= max_lines_per_file:
                    break
                lines_scanned += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                tags = ttt_action_evidence_tags(data)
                if not tags:
                    continue
                evidence_lines += 1
                file_has_evidence = True
                for tag in tags:
                    if tag not in evidence_tags:
                        evidence_tags.append(tag)
        if file_has_evidence:
            evidence_files.append(path)
    return {
        "files_scanned": files_scanned,
        "lines_scanned": lines_scanned,
        "evidence_line_count": evidence_lines,
        "evidence_file_count": len(evidence_files),
        "evidence_files_sample": limited_paths(evidence_files, repo),
        "evidence_tags": ";".join(evidence_tags),
    }


def count_files(path: Path, patterns: Iterable[str] = ("*",)) -> int:
    if not path.exists():
        return 0
    total = 0
    for pattern in patterns:
        total += sum(1 for p in path.rglob(pattern) if p.is_file())
    return total


def image_frame_count(kitti_root: Path, seq: str) -> int:
    image_dir = kitti_root / "sequences" / seq / "image_2"
    if not image_dir.is_dir():
        return 0
    return sum(1 for path in image_dir.iterdir() if path.is_file())


def label_role_counts(label_names: Sequence[Any]) -> dict[str, int]:
    names = [str(name).lower() for name in label_names]

    def count(words: Sequence[str]) -> int:
        return sum(1 for name in names if any(word in name for word in words))

    return {
        "semantic_label_count": len(names),
        "stable_role_label_count": count(STABLE_WORDS),
        "dynamic_role_label_count": count(DYNAMIC_WORDS),
        "lowtrust_role_label_count": count(LOWTRUST_WORDS),
        "context_role_label_count": count(CONTEXT_WORDS),
    }


def seq_preprocess_dir(preprocess_root: Path, seq: str) -> Path:
    return preprocess_root / seq


def stage_cache_dir(preprocess_root: Path, seq: str) -> Path:
    return seq_preprocess_dir(preprocess_root, seq) / "stage_c_cache_semantic_chunks"


def stage_chunk_count(stage_dir: Path) -> int:
    if not stage_dir.is_dir():
        return 0
    return sum(1 for path in stage_dir.glob("chunk_*") if path.is_dir())


def discover_radio_dirs(preprocess_root: Path, seq: str) -> list[Path]:
    root = seq_preprocess_dir(preprocess_root, seq)
    out: list[Path] = []
    for pattern in ("radio_sidecar_chunks*", "radseg_sidecar_chunks*"):
        out.extend(path for path in root.glob(pattern) if path.is_dir())
    return sorted(dict.fromkeys(out))


def is_under_code_audit(path: Path) -> bool:
    return "code_audit_pack" in path.parts


def limited_paths(paths: Sequence[Path], repo: Path, limit: int = 8) -> str:
    rels: list[str] = []
    for path in paths[:limit]:
        try:
            rels.append(str(path.relative_to(repo)))
        except ValueError:
            rels.append(str(path))
    return ";".join(rels)


def discover_baseline_txts(results_root: Path, seq: str) -> list[Path]:
    candidates = [
        path
        for path in results_root.rglob(f"{seq}.txt")
        if path.is_file() and not is_under_code_audit(path)
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)


def discover_run_dirs_with_seq_txt(results_root: Path, seq: str) -> list[Path]:
    return [path.parent for path in discover_baseline_txts(results_root, seq)]


def discover_run_child_dirs(run_dirs: Sequence[Path], child_name: str) -> list[Path]:
    out = [run_dir / child_name for run_dir in run_dirs if (run_dir / child_name).is_dir()]
    return sorted(dict.fromkeys(out), key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)


def discover_run_child_files(run_dirs: Sequence[Path], file_name: str) -> list[Path]:
    out = [run_dir / file_name for run_dir in run_dirs if (run_dir / file_name).is_file()]
    return sorted(dict.fromkeys(out), key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)


def seq_audit_row(repo: Path, args: argparse.Namespace, seq: str) -> dict[str, Any]:
    preprocess_root: Path = args.preprocess_root
    results_root: Path = args.results_root
    kitti_root: Path = args.kitti_root

    seq_root = seq_preprocess_dir(preprocess_root, seq)
    sparse_pt = seq_root / "sparse_masklets_with_semantic.pt"
    semantic_metrics = load_json(seq_root / "sparse_masklets_with_semantic.metrics.json")
    stage_dir = stage_cache_dir(preprocess_root, seq)
    stage_summary = load_json(stage_dir / "conversion_summary.json")
    stage_chunks = stage_chunk_count(stage_dir)
    labels = semantic_metrics.get("label_names", [])
    if not isinstance(labels, list):
        labels = []
    role_counts = label_role_counts(labels)
    radio_dirs = discover_radio_dirs(preprocess_root, seq)
    radio_file_count = sum(count_files(path, ("*.pt", "*.json", "*.jsonl", "*.png")) for path in radio_dirs)

    frame_count = image_frame_count(kitti_root, seq)
    gt_path = kitti_root / "poses" / f"{seq}.txt"
    baseline_txts = discover_baseline_txts(results_root, seq)
    run_dirs = discover_run_dirs_with_seq_txt(results_root, seq)
    per_chunk_dirs = discover_run_child_dirs(run_dirs, "per_chunk_geometry")
    overlap_dirs = discover_run_child_dirs(run_dirs, "overlap_pairs")
    merge_trace_files = discover_run_child_files(run_dirs, "merge_state_trace.jsonl")
    hmc_state_hash_files = discover_run_child_files(run_dirs, "hmc_state_hash.jsonl")
    postmerge_files = discover_run_child_files(run_dirs, "postmerge_global_pose.jsonl")
    ttt_post_delta_dirs = discover_run_child_dirs(run_dirs, "ttt_spatial_post_delta_maps")
    read_dump_dirs = discover_run_child_dirs(run_dirs, "read_cue_patch_dumps")
    ttt_action_evidence = scan_ttt_action_evidence(hmc_state_hash_files, repo)

    semantic_label_maps_available = boolish(semantic_metrics.get("semantic_format")) or boolish(
        stage_summary.get("has_semantic_segmentation")
    )
    semantic_confidence_available = boolish(semantic_metrics.get("has_confidence_maps")) or boolish(
        stage_summary.get("has_semantic_confidence")
    )
    stage_c_cache_available = stage_dir.is_dir() and stage_chunks > 0 and boolish(
        stage_summary.get("has_semantic_segmentation")
    )
    enough_frames_for_chunking = frame_count >= int(args.min_frames)
    baseline_trajectory_available = len(baseline_txts) > 0
    per_chunk_geometry_available = len(per_chunk_dirs) > 0
    overlap_pairs_available = len(overlap_dirs) > 0
    merge_state_trace_available = len(merge_trace_files) > 0
    hmc_state_hash_available = len(hmc_state_hash_files) > 0
    ttt_post_delta_available = len(ttt_post_delta_dirs) > 0
    ttt_trace_available = hmc_state_hash_available or ttt_post_delta_available
    ttt_action_trace_available = ttt_post_delta_available or ttt_action_evidence["evidence_file_count"] > 0
    pca_feature_dumps_available = len(read_dump_dirs) > 0
    radio_available = len(radio_dirs) > 0 and radio_file_count > 0

    phase1_basic_case_mining_allowed = (
        semantic_label_maps_available and baseline_trajectory_available and enough_frames_for_chunking
    )
    semantic_confidence_action_allowed = phase1_basic_case_mining_allowed and semantic_confidence_available
    radio_action_allowed = phase1_basic_case_mining_allowed and radio_available
    mid_swa_action_allowed = phase1_basic_case_mining_allowed and overlap_pairs_available
    long_ttt_action_allowed = phase1_basic_case_mining_allowed and ttt_action_trace_available
    visual_confirmation_ready = phase1_basic_case_mining_allowed and (
        pca_feature_dumps_available or per_chunk_geometry_available or overlap_pairs_available or ttt_trace_available
    )

    blockers = [
        name
        for name, ok in (
            ("semantic_label_maps_available", semantic_label_maps_available),
            ("baseline_trajectory_available", baseline_trajectory_available),
            ("enough_frames_for_chunking", enough_frames_for_chunking),
        )
        if not ok
    ]
    diagnostic_only_reasons = [
        name
        for name, ok in (
            ("confidence_maps_unavailable", semantic_confidence_available),
            ("radio_radseg_unavailable", radio_available),
            ("overlap_pairs_unavailable", overlap_pairs_available),
            ("ttt_action_trace_unavailable", ttt_action_trace_available),
            ("pca_read_dump_unavailable", pca_feature_dumps_available),
        )
        if not ok
    ]

    return {
        "seq": seq,
        "image_frame_count": frame_count,
        "gt_pose_path": str(gt_path),
        "gt_pose_exists": gt_path.is_file(),
        "semantic_sparse_pt": str(sparse_pt),
        "semantic_sparse_pt_exists": sparse_pt.is_file(),
        "semantic_metrics_json": str(seq_root / "sparse_masklets_with_semantic.metrics.json"),
        "semantic_metrics_json_exists": (seq_root / "sparse_masklets_with_semantic.metrics.json").is_file(),
        "semantic_label_maps_available": semantic_label_maps_available,
        "semantic_confidence_available": semantic_confidence_available,
        "semantic_num_frames": semantic_metrics.get("num_frames", stage_summary.get("num_frames")),
        "semantic_num_labels": semantic_metrics.get("num_labels", stage_summary.get("semantic_num_labels")),
        "stage_c_cache_dir": str(stage_dir),
        "stage_c_cache_exists": stage_dir.is_dir(),
        "stage_c_chunk_dirs": stage_chunks,
        "stage_c_cache_available": stage_c_cache_available,
        "baseline_trajectory_available": baseline_trajectory_available,
        "baseline_txt_candidate_count": len(baseline_txts),
        "baseline_txt_latest": limited_paths(baseline_txts, repo, 1),
        "baseline_txt_candidates_sample": limited_paths(baseline_txts, repo),
        "per_chunk_geometry_available": per_chunk_geometry_available,
        "per_chunk_geometry_dir_count": len(per_chunk_dirs),
        "per_chunk_geometry_file_count": sum(count_files(path, ("chunk_*.pt",)) for path in per_chunk_dirs),
        "per_chunk_geometry_dirs_sample": limited_paths(per_chunk_dirs, repo),
        "overlap_pairs_available": overlap_pairs_available,
        "overlap_pairs_dir_count": len(overlap_dirs),
        "overlap_pair_file_count": sum(count_files(path, ("*.pt", "*.json", "*.csv")) for path in overlap_dirs),
        "overlap_pairs_dirs_sample": limited_paths(overlap_dirs, repo),
        "merge_state_trace_available": merge_state_trace_available,
        "merge_state_trace_count": len(merge_trace_files),
        "merge_state_trace_sample": limited_paths(merge_trace_files, repo),
        "hmc_state_hash_available": hmc_state_hash_available,
        "hmc_state_hash_count": len(hmc_state_hash_files),
        "hmc_state_hash_sample": limited_paths(hmc_state_hash_files, repo),
        "ttt_post_delta_available": ttt_post_delta_available,
        "ttt_post_delta_dir_count": len(ttt_post_delta_dirs),
        "ttt_post_delta_dirs_sample": limited_paths(ttt_post_delta_dirs, repo),
        "ttt_trace_available": ttt_trace_available,
        "ttt_action_trace_available": ttt_action_trace_available,
        "ttt_action_evidence_file_count": ttt_action_evidence["evidence_file_count"],
        "ttt_action_evidence_line_count": ttt_action_evidence["evidence_line_count"],
        "ttt_action_evidence_files_scanned": ttt_action_evidence["files_scanned"],
        "ttt_action_evidence_lines_scanned": ttt_action_evidence["lines_scanned"],
        "ttt_action_evidence_tags": ttt_action_evidence["evidence_tags"],
        "ttt_action_evidence_files_sample": ttt_action_evidence["evidence_files_sample"],
        "read_dump_dir_count": len(read_dump_dirs),
        "read_dump_dirs_sample": limited_paths(read_dump_dirs, repo),
        "pca_feature_dumps_available": pca_feature_dumps_available,
        "radio_sidecar_available": radio_available,
        "radio_sidecar_dir_count": len(radio_dirs),
        "radio_sidecar_file_count": radio_file_count,
        "radio_sidecar_dirs_sample": limited_paths(radio_dirs, repo),
        "phase1_basic_case_mining_allowed": phase1_basic_case_mining_allowed,
        "semantic_confidence_action_allowed": semantic_confidence_action_allowed,
        "radio_action_allowed": radio_action_allowed,
        "short_read_action_allowed": phase1_basic_case_mining_allowed,
        "mid_swa_action_allowed": mid_swa_action_allowed,
        "long_ttt_action_allowed": long_ttt_action_allowed,
        "visual_confirmation_ready": visual_confirmation_ready,
        "required_blockers": ";".join(blockers),
        "diagnostic_only_reasons": ";".join(diagnostic_only_reasons),
        **role_counts,
    }


def subset_rows(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    return [{key: row.get(key, "") for key in keys} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqs", default=",".join(DEFAULT_SEQS))
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-frames", type=int, default=160)
    args = parser.parse_args()

    repo = Path.cwd()
    seqs = parse_seqs(args.seqs)
    rows = [seq_audit_row(repo, args, seq) for seq in seqs]

    semantic_keys = (
        "seq",
        "semantic_sparse_pt_exists",
        "semantic_metrics_json_exists",
        "semantic_label_maps_available",
        "semantic_confidence_available",
        "semantic_num_frames",
        "semantic_num_labels",
        "stage_c_cache_available",
        "stage_c_chunk_dirs",
        "semantic_label_count",
        "stable_role_label_count",
        "dynamic_role_label_count",
        "lowtrust_role_label_count",
        "context_role_label_count",
    )
    geometry_keys = (
        "seq",
        "image_frame_count",
        "gt_pose_exists",
        "baseline_trajectory_available",
        "baseline_txt_candidate_count",
        "baseline_txt_latest",
        "per_chunk_geometry_available",
        "per_chunk_geometry_dir_count",
        "per_chunk_geometry_file_count",
        "overlap_pairs_available",
        "overlap_pairs_dir_count",
        "overlap_pair_file_count",
        "merge_state_trace_available",
        "merge_state_trace_count",
    )
    radio_keys = (
        "seq",
        "radio_sidecar_available",
        "radio_sidecar_dir_count",
        "radio_sidecar_file_count",
        "radio_action_allowed",
        "radio_sidecar_dirs_sample",
    )
    ttt_keys = (
        "seq",
        "ttt_trace_available",
        "ttt_action_trace_available",
        "hmc_state_hash_available",
        "hmc_state_hash_count",
        "ttt_post_delta_available",
        "ttt_post_delta_dir_count",
        "ttt_action_evidence_file_count",
        "ttt_action_evidence_line_count",
        "ttt_action_evidence_files_scanned",
        "ttt_action_evidence_lines_scanned",
        "ttt_action_evidence_tags",
        "long_ttt_action_allowed",
        "hmc_state_hash_sample",
        "ttt_post_delta_dirs_sample",
        "ttt_action_evidence_files_sample",
    )
    allowed = {
        row["seq"]: {
            "phase1_basic_case_mining_allowed": bool(row["phase1_basic_case_mining_allowed"]),
            "semantic_confidence_action_allowed": bool(row["semantic_confidence_action_allowed"]),
            "radio_action_allowed": bool(row["radio_action_allowed"]),
            "short_read_action_allowed": bool(row["short_read_action_allowed"]),
            "mid_swa_action_allowed": bool(row["mid_swa_action_allowed"]),
            "long_ttt_action_allowed": bool(row["long_ttt_action_allowed"]),
            "visual_confirmation_ready": bool(row["visual_confirmation_ready"]),
            "required_blockers": [x for x in str(row["required_blockers"]).split(";") if x],
            "diagnostic_only_reasons": [x for x in str(row["diagnostic_only_reasons"]).split(";") if x],
        }
        for row in rows
    }
    summary = {
        "schema": "acl2_v80tf_phase0_multiseq_artifact_audit_v1",
        "created_at_utc": utc_now(),
        "seqs": seqs,
        "phase0_gate_pass": sum(1 for row in rows if row["phase1_basic_case_mining_allowed"]) >= 3,
        "phase1_basic_case_mining_allowed_seqs": [
            row["seq"] for row in rows if row["phase1_basic_case_mining_allowed"]
        ],
        "semantic_confidence_action_allowed_seqs": [
            row["seq"] for row in rows if row["semantic_confidence_action_allowed"]
        ],
        "radio_action_allowed_seqs": [row["seq"] for row in rows if row["radio_action_allowed"]],
        "mid_swa_action_allowed_seqs": [row["seq"] for row in rows if row["mid_swa_action_allowed"]],
        "long_ttt_action_allowed_seqs": [row["seq"] for row in rows if row["long_ttt_action_allowed"]],
        "visual_confirmation_ready_seqs": [row["seq"] for row in rows if row["visual_confirmation_ready"]],
        "blockers_by_seq": {
            row["seq"]: [x for x in str(row["required_blockers"]).split(";") if x]
            for row in rows
            if row["required_blockers"]
        },
        "diagnostic_only_reasons_by_seq": {
            row["seq"]: [x for x in str(row["diagnostic_only_reasons"]).split(";") if x]
            for row in rows
            if row["diagnostic_only_reasons"]
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "artifact_status_by_seq.csv", rows)
    write_csv(args.out_dir / "semantic_status_by_seq.csv", subset_rows(rows, semantic_keys), semantic_keys)
    write_csv(args.out_dir / "geometry_status_by_seq.csv", subset_rows(rows, geometry_keys), geometry_keys)
    write_csv(args.out_dir / "radio_status_by_seq.csv", subset_rows(rows, radio_keys), radio_keys)
    write_csv(args.out_dir / "ttt_status_by_seq.csv", subset_rows(rows, ttt_keys), ttt_keys)
    write_json(args.out_dir / "allowed_case_mining_by_seq.json", allowed)
    write_json(args.out_dir / "phase0_artifact_audit_summary.json", summary)

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "phase0_gate_pass": summary["phase0_gate_pass"],
                "phase1_basic_case_mining_allowed_seqs": summary[
                    "phase1_basic_case_mining_allowed_seqs"
                ],
                "mid_swa_action_allowed_seqs": summary["mid_swa_action_allowed_seqs"],
                "long_ttt_action_allowed_seqs": summary["long_ttt_action_allowed_seqs"],
                "diagnostic_only_reasons_by_seq": summary["diagnostic_only_reasons_by_seq"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

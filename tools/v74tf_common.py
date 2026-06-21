#!/usr/bin/env python3
"""Shared helpers for ACL2 v74-TF training-free semantic-memory diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from v73_semantic_memory_common import (
    TARGET_CHUNKS,
    finite_median,
    finite_quantile,
    finite_values,
    load_json,
    read_csv,
    safe_float,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


REPORT_ROOT = Path("results/kitti01_hmc_v2/acl2_v74tf_training_free_semantic_memory_control/report_final")
V73_REPORT_ROOT = Path("results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final")
H35_TRACE_RUN = Path(
    "results/kitti01_hmc_v2/acl2_v67_dense_semantic_reconstruction/"
    "phaseO2_h35_trace_geom_merge_full/rollouts/V67S_H35_TRACE_GEOM_MERGE_FULL_H35_PARITY"
)
V53_H35_FULL = Path(
    "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_fix_full/rollouts/V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075"
)
V74TF_ROOT = Path("results/kitti01_hmc_v2/acl2_v74tf_training_free_semantic_memory_control")
GT_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")
PREPROCESS_ROOT = Path("results/kitti_preprocess")
DEFAULT_SEQS = ["01", "09", "00", "02"]

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
)
DYNAMIC_WORDS = ("car", "person", "rider", "bicycle", "motorcycle", "bus", "truck", "train")
LOWTRUST_WORDS = ("tree", "grass", "vegetation", "mountain", "terrain", "void", "unknown", "plant")
ROAD_WORDS = ("road", "ground", "crosswalk", "sidewalk", "path")
SKY_WORDS = ("sky",)


def parse_seqs(text: str | Sequence[str] | None) -> list[str]:
    if text is None:
        return list(DEFAULT_SEQS)
    if isinstance(text, (list, tuple)):
        return [str(x).zfill(2) for x in text]
    return [part.strip().zfill(2) for part in str(text).split(",") if part.strip()]


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def seq_preprocess_dir(preprocess_root: Path, seq: str) -> Path:
    return preprocess_root / str(seq).zfill(2)


def stage_cache_dir(preprocess_root: Path, seq: str) -> Path:
    return seq_preprocess_dir(preprocess_root, seq) / "stage_c_cache_semantic_chunks"


def stage_chunk_dirs(stage_dir: Path) -> list[Path]:
    return sorted(path for path in stage_dir.glob("chunk_*") if path.is_dir())


def read_stage_summary(preprocess_root: Path, seq: str) -> dict[str, Any]:
    path = stage_cache_dir(preprocess_root, seq) / "conversion_summary.json"
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def read_semantic_metrics(preprocess_root: Path, seq: str) -> dict[str, Any]:
    path = seq_preprocess_dir(preprocess_root, seq) / "sparse_masklets_with_semantic.metrics.json"
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def label_role_ids(label_names: Sequence[Any], words: Sequence[str]) -> list[int]:
    lowered = [str(name).lower() for name in label_names]
    out: list[int] = []
    for idx, name in enumerate(lowered):
        if any(word in name for word in words):
            out.append(idx)
    return out


def label_role_coverage(metrics: Mapping[str, Any]) -> dict[str, Any]:
    names = metrics.get("label_names", [])
    if not isinstance(names, list):
        names = []
    return {
        "label_count": len(names),
        "stable_role_label_count": len(label_role_ids(names, STABLE_WORDS)),
        "dynamic_role_label_count": len(label_role_ids(names, DYNAMIC_WORDS)),
        "lowtrust_role_label_count": len(label_role_ids(names, LOWTRUST_WORDS)),
        "road_role_label_count": len(label_role_ids(names, ROAD_WORDS)),
        "sky_role_label_count": len(label_role_ids(names, SKY_WORDS)),
        "label_names": ",".join(str(x) for x in names),
    }


def discover_radio_dirs(preprocess_root: Path, seq: str) -> list[Path]:
    root = seq_preprocess_dir(preprocess_root, seq)
    dirs: list[Path] = []
    for pattern in ("radio_sidecar_chunks*", "radseg_sidecar_chunks*"):
        dirs.extend(path for path in root.glob(pattern) if path.is_dir())
    return sorted(dict.fromkeys(dirs))


def directory_file_count(path: Path, patterns: Sequence[str] = ("*.pt", "*.json", "*.jsonl")) -> int:
    if not path.exists():
        return 0
    count = 0
    for pattern in patterns:
        count += sum(1 for _ in path.rglob(pattern))
    return count


def discover_v74tf_prefix_run(seq: str) -> Path | None:
    seq = str(seq).zfill(2)
    candidates = sorted(
        V74TF_ROOT.glob(f"phase{seq}_h35_artifact_repair/rollouts/V74TF_{seq}_H35_PREFIX*"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for run_dir in candidates:
        if (run_dir / f"{seq}.txt").exists() and (run_dir / "per_chunk_geometry").is_dir():
            return run_dir
    return None


def discover_geometry_for_seq(seq: str) -> dict[str, Any]:
    seq = str(seq).zfill(2)
    gt_path = GT_ROOT / f"{seq}.txt"
    out: dict[str, Any] = {
        "seq": seq,
        "gt_path": str(gt_path),
        "gt_exists": gt_path.exists(),
        "baseline_run_dir": "",
        "baseline_txt": "",
        "postmerge_global_pose_jsonl": "",
        "merge_state_trace_jsonl": "",
        "hmc_state_hash_jsonl": "",
        "per_chunk_geometry_dir": "",
        "per_chunk_geometry_count": 0,
        "overlap_pairs_dir": "",
        "overlap_pair_count": 0,
        "baseline_trajectory_available": False,
        "merge_trace_available": False,
        "per_chunk_geometry_available": False,
        "overlap_pairs_available": False,
        "geometry_artifact_status": "missing_baseline_geometry",
    }
    if seq == "01":
        run_dir = H35_TRACE_RUN
        out.update(
            {
                "baseline_run_dir": str(run_dir),
                "baseline_txt": str(V53_H35_FULL / "01.txt") if (V53_H35_FULL / "01.txt").exists() else "",
                "postmerge_global_pose_jsonl": str(run_dir / "postmerge_global_pose.jsonl"),
                "merge_state_trace_jsonl": str(run_dir / "merge_state_trace.jsonl"),
                "hmc_state_hash_jsonl": str(run_dir / "hmc_state_hash.jsonl"),
                "per_chunk_geometry_dir": str(run_dir / "per_chunk_geometry"),
                "overlap_pairs_dir": str(run_dir / "overlap_pairs"),
            }
        )
        out["baseline_trajectory_available"] = (run_dir / "postmerge_global_pose.jsonl").exists()
        out["merge_trace_available"] = (run_dir / "merge_state_trace.jsonl").exists()
        geom_dir = run_dir / "per_chunk_geometry"
        out["per_chunk_geometry_count"] = sum(1 for _ in geom_dir.glob("chunk_*.pt")) if geom_dir.exists() else 0
        out["per_chunk_geometry_available"] = out["per_chunk_geometry_count"] > 0
        overlap_dir = run_dir / "overlap_pairs"
        out["overlap_pair_count"] = sum(1 for _ in overlap_dir.glob("*.pt")) if overlap_dir.exists() else 0
        out["overlap_pairs_available"] = out["overlap_pair_count"] > 0
        if out["baseline_trajectory_available"] and out["merge_trace_available"] and out["overlap_pairs_available"]:
            out["geometry_artifact_status"] = "h35_trace_geometry_available"
    else:
        run_dir = discover_v74tf_prefix_run(seq)
        if run_dir is not None:
            geom_dir = run_dir / "per_chunk_geometry"
            overlap_dir = run_dir / "overlap_pairs"
            out.update(
                {
                    "baseline_run_dir": str(run_dir),
                    "baseline_txt": str(run_dir / f"{seq}.txt"),
                    "merge_state_trace_jsonl": str(run_dir / "merge_state_trace.jsonl"),
                    "hmc_state_hash_jsonl": str(run_dir / "hmc_state_hash.jsonl"),
                    "per_chunk_geometry_dir": str(geom_dir),
                    "per_chunk_geometry_count": sum(1 for _ in geom_dir.glob("chunk_*.pt")),
                    "overlap_pairs_dir": str(overlap_dir),
                    "overlap_pair_count": sum(1 for _ in overlap_dir.glob("*.pt")) if overlap_dir.exists() else 0,
                }
            )
            out["baseline_trajectory_available"] = (run_dir / f"{seq}.txt").exists()
            out["merge_trace_available"] = (run_dir / "merge_state_trace.jsonl").exists()
            out["per_chunk_geometry_available"] = out["per_chunk_geometry_count"] > 0
            out["overlap_pairs_available"] = out["overlap_pair_count"] > 0
            if out["baseline_trajectory_available"] and out["merge_trace_available"] and out["per_chunk_geometry_available"]:
                out["geometry_artifact_status"] = "v74tf_h35_prefix_geometry_available"
            return out
        candidates = sorted(Path("results").glob(f"kitti_acl2_v4_cross_sequence_*/*/{seq}.txt"))
        if candidates:
            out["baseline_txt"] = str(candidates[0])
            out["baseline_trajectory_available"] = True
            out["geometry_artifact_status"] = "trajectory_txt_only_no_h35_trace"
    return out


def finite_norm(value: Any, denom: float | None) -> float | None:
    val = safe_float(value)
    if val is None or denom is None or not math.isfinite(denom) or abs(denom) <= 1e-12:
        return None
    return float(val / denom)


def add_j_scale(rows: list[dict[str, Any]]) -> None:
    for seq in sorted({str(row.get("seq", "")) for row in rows}):
        seq_rows = [row for row in rows if str(row.get("seq", "")) == seq]
        denoms = {
            "future_after_overlap": finite_quantile((row.get("future_after_overlap") for row in seq_rows), 0.75),
            "head_to_tail": finite_quantile((row.get("head_to_tail") for row in seq_rows), 0.75),
            "scale_cv": finite_quantile((row.get("scale_cv") for row in seq_rows), 0.75),
            "boundary_jump": finite_quantile((row.get("boundary_jump") for row in seq_rows), 0.75),
        }
        for row in seq_rows:
            parts = {
                key: finite_norm(row.get(key), denom)
                for key, denom in denoms.items()
            }
            if any(value is None for value in parts.values()):
                row["J_scale"] = None
                row["J_scale_status"] = "missing_required_metric"
                continue
            row["J_scale"] = float(
                0.35 * parts["future_after_overlap"]
                + 0.30 * parts["head_to_tail"]
                + 0.25 * parts["scale_cv"]
                + 0.10 * parts["boundary_jump"]
            )
            row["J_scale_status"] = "computed_from_seq_p75_normalized_terms"


def target_chunks_for_seq(seq: str, available_chunks: Sequence[int], max_chunks: int = 12) -> list[int]:
    seq = str(seq).zfill(2)
    chunks = sorted(int(x) for x in available_chunks)
    if seq == "01":
        return [chunk for chunk in TARGET_CHUNKS if chunk in set(chunks)] or list(TARGET_CHUNKS)
    if seq == "09":
        return chunks[:max(8, min(max_chunks, len(chunks)))]
    return chunks[: min(max_chunks, len(chunks))]


def summarize_metric_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    required = ("local_sim3_ate", "global_chunk_ate", "head_to_tail", "future_after_overlap", "scale_cv", "boundary_jump")
    for seq in sorted({str(row.get("seq", "")) for row in rows}):
        seq_rows = [row for row in rows if str(row.get("seq", "")) == seq]
        finite_by_key = {key: sum(safe_float(row.get(key)) is not None for row in seq_rows) for key in required}
        complete = sum(all(safe_float(row.get(key)) is not None for key in required) for row in seq_rows)
        out[seq] = {
            "rows": len(seq_rows),
            "complete_required_metric_rows": complete,
            "finite_by_key": finite_by_key,
            "median_J_scale": finite_median(row.get("J_scale") for row in seq_rows),
        }
    return out


def copy_selected_fields(row: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: row.get(key) for key in keys}


def write_markdown_table(path: Path, title: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines += ["No rows.", ""]
    else:
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join("---" for _ in fields) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
        lines.append("")
    write_text(path, "\n".join(lines))


def json_loads_maybe(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def improvement_ratio(before: Any, after: Any) -> float | None:
    b = safe_float(before)
    a = safe_float(after)
    if b is None or a is None or abs(b) <= 1e-12:
        return None
    return float((b - a) / abs(b))


def median(values: Iterable[Any]) -> float | None:
    vals = finite_values(values)
    return float(np.median(vals)) if vals else None

#!/usr/bin/env python3
"""Build fresh 03/04 semantic support rows and token tensors for R47 validation."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import build_v107r_semantic_memory_decision_cue_operation_control as v107r  # noqa: E402


STAGE1_SCRIPT = ROOT / "tools/build_v118tf_stage1_causal_object_track_sidecar.py"
spec = importlib.util.spec_from_file_location("acl2_v118tf_stage1_sidecar", STAGE1_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"unable to load Stage1 sidecar script: {STAGE1_SCRIPT}")
stage1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage1)

RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"


def parse_seq_env(name: str, default: str) -> tuple[str, ...]:
    seqs = tuple(part.strip().zfill(2) for part in os.environ.get(name, default).replace(";", ",").split(",") if part.strip())
    return seqs or tuple(part.strip().zfill(2) for part in default.split(",") if part.strip())


STAGE_TAG = os.environ.get("ACL2_V118_FRESH_TOKEN_TAG", "r50").strip().lower() or "r50"
STAGE = RESULT_ROOT / os.environ.get("ACL2_V118_FRESH_TOKEN_STAGE_SLUG", "stage4_r50_lingbot_ar_fresh_support_token_tensors")
SUMMARY_DIR = STAGE / "summary"
TOKEN_ROOT = STAGE / "token_semantics"
SEM_ROOT = ROOT / "results/kitti_preprocess"
FRESH_SEQS = parse_seq_env("ACL2_V118_FRESH_TOKEN_SEQS", "04,03")
SEQ_LABEL = "/".join(FRESH_SEQS)
GRID = {
    "target_width": 504,
    "target_height": 280,
    "patch_size": 14,
    "patch_grid_h": 20,
    "patch_grid_w": 36,
    "inferred_patch_count": 720,
    "patch_start_idx": 6,
    "special_token_count": 6,
}
CHUNK_RE = re.compile(r"_(\d{6})_(\d{6})$")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def top_counts(values: list[Any], limit: int = 5) -> str:
    counts = Counter("" if pd.isna(value) else str(value) for value in values)
    counts.pop("", None)
    return ";".join(f"{key}:{value}" for key, value in counts.most_common(limit))


def frame_support_rows(prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not prefix_rows:
        return []
    df = pd.DataFrame(prefix_rows)
    rows: list[dict[str, Any]] = []
    for (seq, frame_id), group in df.groupby(["seq", "frame_id"], sort=True):
        persistence = pd.to_numeric(group["semantic_persistence_prefix"], errors="coerce")
        confidence = pd.to_numeric(group["semantic_confidence_prefix"], errors="coerce")
        area = pd.to_numeric(group["current_area_ratio"], errors="coerce")
        mask_quality = pd.to_numeric(group["current_mask_quality"], errors="coerce")
        best_idx = persistence.fillna(-1).idxmax()
        best = group.loc[best_idx]
        rows.append(
            {
                "schema": "acl2_v118tf_stage4_r50_fresh_frame_semantic_support_row_v1",
                "seq": str(seq).zfill(2),
                "frame_id": int(frame_id),
                "visible_track_rows": int(len(group)),
                "unique_track_count": int(group["track_id"].nunique()),
                "max_semantic_persistence_prefix": safe_float(persistence.max()),
                "mean_semantic_persistence_prefix": safe_float(persistence.mean()),
                "max_semantic_confidence_prefix": safe_float(confidence.max()),
                "mean_semantic_confidence_prefix": safe_float(confidence.mean()),
                "sum_current_area_ratio": safe_float(area.sum()),
                "mean_current_mask_quality": safe_float(mask_quality.mean()),
                "best_track_id_by_semantic_persistence": str(best.get("track_id", "")),
                "best_track_role": str(best.get("current_role", "")),
                "best_track_label": str(best.get("current_label", "")),
                "top_roles": top_counts(group["current_role"].tolist()),
                "top_labels": top_counts(group["current_label"].tolist()),
            }
        )
    return rows


def frame_count(seq: str) -> int:
    conversion = read_json(SEM_ROOT / seq / "stage_c_cache_semantic_chunks/conversion_summary.json")
    if conversion.get("num_frames"):
        return int(conversion["num_frames"])
    image_dir = ROOT / "data/kitti/dataset/sequences" / seq / "image_2"
    return sum(1 for path in image_dir.glob("*.png"))


def role_channels(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, int]:
    role = str(row.get("semantic_role", ""))
    trust = max(0.0, min(1.0, float(row.get("semantic_trust", 0.0))))
    boundary = max(0.0, min(1.0, float(row.get("semantic_boundary_risk", 0.0))))
    confidence = max(0.0, min(1.0, float(row.get("semantic_confidence", 0.0))))
    dynamic = trust if role == "dynamic_transient" else 0.0
    stable = trust if role == "stable_structure" else 0.0
    weak = trust if role in {"vegetation_weak_context", "ground_or_road_weak", "sky_or_lowobs"} else 0.0
    lowtrust = max(1.0 - trust, trust if role == "unknown_lowtrust" else 0.0)
    role_id = {
        "dynamic_transient": 1,
        "semantic_boundary": 2,
        "unknown_lowtrust": 3,
        "vegetation_weak_context": 4,
        "ground_or_road_weak": 5,
        "sky_or_lowobs": 6,
        "stable_structure": 7,
        "road_boundary_or_layout": 8,
    }.get(role, 0)
    return dynamic, boundary, lowtrust, weak, stable, confidence, role_id


def assign_frames_to_chunks(seq: str, length: int) -> tuple[dict[Path, list[int]], list[dict[str, Any]]]:
    chunks = v107r.build_chunk_index(seq)
    chunk_to_frames: dict[Path, list[int]] = defaultdict(list)
    missing: list[dict[str, Any]] = []
    for frame_id in range(length):
        chunk = v107r.find_chunk(chunks, frame_id)
        if chunk is None:
            missing.append({"seq": seq, "frame_id": frame_id, "reason": "no_stage_c_chunk_covering_frame"})
            continue
        _start, _end, path = chunk
        chunk_to_frames[path].append(frame_id)
    return chunk_to_frames, missing


def build_token_tensors(seq: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    length = frame_count(seq)
    patch_count = int(GRID["inferred_patch_count"])
    arrays = {
        "dynamic": np.zeros((length, patch_count), dtype=np.float32),
        "boundary": np.zeros((length, patch_count), dtype=np.float32),
        "lowtrust": np.zeros((length, patch_count), dtype=np.float32),
        "weak": np.zeros((length, patch_count), dtype=np.float32),
        "stable": np.zeros((length, patch_count), dtype=np.float32),
        "confidence": np.zeros((length, patch_count), dtype=np.float32),
        "role_id": np.zeros((length, patch_count), dtype=np.uint8),
        "filled": np.zeros((length, patch_count), dtype=np.bool_),
    }
    chunk_to_frames, missing = assign_frames_to_chunks(seq, length)
    duplicate_count = 0
    processed_frames = 0
    role_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for chunk_path, frames in sorted(chunk_to_frames.items(), key=lambda item: item[0].as_posix()):
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        sem = payload.get("semantic_segmentation", {})
        label_maps = sem.get("label_maps")
        conf_maps = sem.get("confidence_maps")
        label_names = list(sem.get("label_names", []))
        label_to_id = sem.get("label_to_id", {}) or {}
        if label_maps is None or conf_maps is None:
            for frame_id in frames:
                missing.append({"seq": seq, "frame_id": frame_id, "reason": "chunk_missing_label_or_confidence_maps"})
            continue
        match = CHUNK_RE.search(chunk_path.parent.name)
        global_start = int(sem.get("global_start_frame", match.group(1) if match else 0))
        void_id = int(label_to_id.get("void", 0))
        for frame_id in frames:
            local_idx = frame_id - global_start
            if local_idx < 0 or local_idx >= int(label_maps.shape[0]):
                missing.append({"seq": seq, "frame_id": frame_id, "reason": "frame_not_in_loaded_chunk"})
                continue
            label_proj = v107r.cover_fit_resize_2d(label_maps[local_idx], GRID["target_height"], GRID["target_width"], "nearest").long()
            conf_proj = v107r.cover_fit_resize_2d(conf_maps[local_idx], GRID["target_height"], GRID["target_width"], "bilinear").float()
            patch_rows, _summary = v107r.patchify_projected_frame(label_proj, conf_proj, GRID, label_names, void_id)
            for prow in patch_rows:
                patch_idx = int(prow["token_id"]) - int(GRID["patch_start_idx"])
                if bool(arrays["filled"][frame_id, patch_idx]):
                    duplicate_count += 1
                dynamic, boundary, lowtrust, weak, stable, confidence, role_id = role_channels(prow)
                arrays["dynamic"][frame_id, patch_idx] = dynamic
                arrays["boundary"][frame_id, patch_idx] = boundary
                arrays["lowtrust"][frame_id, patch_idx] = lowtrust
                arrays["weak"][frame_id, patch_idx] = weak
                arrays["stable"][frame_id, patch_idx] = stable
                arrays["confidence"][frame_id, patch_idx] = confidence
                arrays["role_id"][frame_id, patch_idx] = role_id
                arrays["filled"][frame_id, patch_idx] = True
                role_counts[str(prow.get("semantic_role", ""))] += 1
                label_counts[str(prow.get("label_name", ""))] += 1
            processed_frames += 1
    output_files: dict[str, str] = {}
    TOKEN_ROOT.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        path = TOKEN_ROOT / f"seq{seq}_{name}.npy"
        np.save(path, arr)
        output_files[name] = rel(path)
    filled = arrays["filled"]
    coverage = {
        "seq": seq,
        "frame_count": int(length),
        "patch_count": int(patch_count),
        "processed_frame_count": int(processed_frames),
        "missing_frame_count": int(len(missing)),
        "filled_token_count": int(filled.sum()),
        "expected_token_count": int(filled.size),
        "token_coverage": float(filled.mean()) if filled.size else 0.0,
        "all_patch_frame_coverage": float(np.mean(filled.all(axis=1))) if filled.shape[0] else 0.0,
        "duplicate_token_assignment_count": int(duplicate_count),
        "dynamic_mean": float(arrays["dynamic"].mean()),
        "boundary_mean": float(arrays["boundary"].mean()),
        "lowtrust_mean": float(arrays["lowtrust"].mean()),
        "weak_mean": float(arrays["weak"].mean()),
        "stable_mean": float(arrays["stable"].mean()),
        "top_roles": dict(role_counts.most_common(12)),
        "top_labels": dict(label_counts.most_common(12)),
        "outputs": output_files,
    }
    return coverage, missing


def compact_prefix_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Keep summary JSON compact; detailed rows live in CSV artifacts."""
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, list):
            out[f"{key}_count"] = len(value)
            continue
        out[key] = value
    return out


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    all_support: list[dict[str, Any]] = []
    all_parity: list[dict[str, Any]] = []
    all_violations: list[dict[str, Any]] = []
    support_summaries: list[dict[str, Any]] = []
    token_coverages: list[dict[str, Any]] = []
    missing_frames: list[dict[str, Any]] = []

    for seq in FRESH_SEQS:
        obs_by_track, obs_meta = stage1.build_observations(seq)
        prefix_rows, running_summary_rows, prefix_meta = stage1.build_prefix_rows(seq, obs_by_track)
        parity_rows, violation_rows = stage1.prefix_parity_rows(seq, obs_by_track, prefix_rows)
        support_rows = frame_support_rows(prefix_rows)
        all_support.extend(support_rows)
        all_parity.extend(parity_rows)
        all_violations.extend(violation_rows)
        support_summaries.append(
            {
                "seq": seq,
                "track_count": obs_meta["track_count"],
                "chunk_count": obs_meta["chunk_count"],
                "prefix_row_count": len(prefix_rows),
                "running_summary_row_count": len(running_summary_rows),
                "frame_support_row_count": len(support_rows),
                "future_leakage_violation_count": len(violation_rows),
                "prefix_meta": compact_prefix_meta(prefix_meta),
            }
        )
        coverage, missing = build_token_tensors(seq)
        token_coverages.append(coverage)
        missing_frames.extend(missing)

    write_csv(SUMMARY_DIR / "stage4_r50_fresh_frame_semantic_support_rows.csv", all_support)
    write_csv(SUMMARY_DIR / "stage4_r50_fresh_prefix_parity_rows.csv", all_parity)
    write_csv(SUMMARY_DIR / "stage4_r50_fresh_prefix_parity_violations.csv", all_violations)
    write_csv(SUMMARY_DIR / "stage4_r50_fresh_token_coverage_rows.csv", token_coverages)
    write_csv(SUMMARY_DIR / "stage4_r50_fresh_missing_token_frames.csv", missing_frames)

    support_ready = bool(all_support) and all(item["future_leakage_violation_count"] == 0 for item in support_summaries)
    token_ready = bool(token_coverages) and all(float(row["token_coverage"]) >= 0.999 for row in token_coverages)
    decision_key = f"stage4_{STAGE_TAG}_decision"
    summary = {
        "schema": "acl2_v118tf_stage4_r50_fresh_support_token_tensor_summary_v1",
        decision_key: (
            "FRESH_SUPPORT_AND_TOKEN_TENSORS_READY_FOR_R47_CONFIGS"
            if support_ready and token_ready
            else "NO_GO_FRESH_SUPPORT_OR_TOKEN_TENSOR_INCOMPLETE"
        ),
        "global_goal_achieved": False,
        "sequences": list(FRESH_SEQS),
        "support_ready": support_ready,
        "token_ready": token_ready,
        "support_summaries": support_summaries,
        "token_coverages": token_coverages,
        "missing_token_frame_count": len(missing_frames),
        "outputs": {
            "support_rows": rel(SUMMARY_DIR / "stage4_r50_fresh_frame_semantic_support_rows.csv"),
            "token_coverage": rel(SUMMARY_DIR / "stage4_r50_fresh_token_coverage_rows.csv"),
            "token_root": rel(TOKEN_ROOT),
            "summary": rel(SUMMARY_DIR / "stage4_r50_fresh_support_token_tensor_summary.json"),
            "report": rel(SUMMARY_DIR / "STAGE4_R50_FRESH_SUPPORT_TOKEN_TENSORS_REPORT.md"),
        },
        "boundary": (
            f"{STAGE_TAG.upper()} builds fresh {SEQ_LABEL} semantic support and token tensors only. It does not run the R47 "
            "candidate, opposite-polarity control, or random control."
        ),
    }
    write_json(SUMMARY_DIR / "stage4_r50_fresh_support_token_tensor_summary.json", summary)
    lines = [
        f"# ACL2 v118 Stage4-{STAGE_TAG.upper()} Fresh Support And Token Tensors",
        "",
        f"- decision: `{summary[decision_key]}`",
        f"- support_ready: `{summary['support_ready']}`",
        f"- token_ready: `{summary['token_ready']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        "",
        "| seq | support rows | parity violations | token coverage | all-frame coverage | stable mean | lowtrust mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    support_by_seq = {row["seq"]: row for row in support_summaries}
    for coverage in token_coverages:
        seq = str(coverage["seq"])
        support = support_by_seq[seq]
        lines.append(
            f"| {seq} | {support['frame_support_row_count']} | {support['future_leakage_violation_count']} | "
            f"{coverage['token_coverage']} | {coverage['all_patch_frame_coverage']} | "
            f"{coverage['stable_mean']} | {coverage['lowtrust_mean']} |"
        )
    lines += ["", "## Boundary", "", summary["boundary"]]
    (SUMMARY_DIR / "STAGE4_R50_FRESH_SUPPORT_TOKEN_TENSORS_REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

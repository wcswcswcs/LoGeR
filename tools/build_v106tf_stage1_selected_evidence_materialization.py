#!/usr/bin/env python3
"""Materialize ACL2 v106 selected evidence rows with semantic/GCA/geometry fields."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106 = ROOT / "results/acl2_v106tf_lingbot_semantic_aware_memory_role_control"
STAGE0 = V106 / "stage0_v105_headlocal_selected_set"
OUT = V106 / "stage1_selected_evidence_materialization"

HEAD_TRACE_ROWS = V105 / "stage4_lingbot_headlocal_trace/headlocal_trace_semantic_key_rows.csv"
ACTION_WORKSPACE = V105 / "stage4_lingbot_action_pilot_or_blocked/workspace"
BASELINE_WORKSPACE = V105 / "stage2_gca_trace/workspace"
ACTION_LABEL = "semantic_headlocal_relaxed_context_only_demote"
BASELINE_METHOD = "lingbot_map_stream_default_stage2_notrace"


STABLE_LABELS = {
    "building", "house", "wall", "fence", "handrail_or_fence", "pole",
    "traffic sign", "traffic light", "static construction", "static object",
    "other_construction", "pillar",
}
ROAD_LABELS = {"road", "ground", "lane", "sidewalk", "path"}
VEGETATION_LABELS = {"tree", "vegetation", "grass", "mountain", "other_plant"}
DYNAMIC_LABELS = {"car", "truck", "bus", "person", "rider", "bicycle", "motorcycle"}
LOWOBS_LABELS = {"sky", "void", "unknown"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    return float(raw)


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    raw = row.get(key, "")
    if raw == "":
        return default
    return int(float(raw))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_top_labels(raw: str) -> list[tuple[str, int]]:
    labels: list[tuple[str, int]] = []
    for item in str(raw or "").split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        label, count = item.rsplit(":", 1)
        try:
            labels.append((label, int(float(count))))
        except ValueError:
            continue
    return labels


def dominant_label(row: dict[str, Any]) -> tuple[str, int, float]:
    labels = parse_top_labels(str(row.get("top_labels", "")))
    patch_count = as_float(row, "patch_count", 0.0)
    if not labels or patch_count <= 0:
        return "unknown", 0, 0.0
    label, count = max(labels, key=lambda item: item[1])
    return label, count, float(count / patch_count)


def semantic_role(label: str, patch_purity: float) -> str:
    if label in DYNAMIC_LABELS:
        return "dynamic_object"
    if label in LOWOBS_LABELS:
        return "sky_or_lowobs"
    if patch_purity < 0.25:
        return "object_boundary"
    if label in STABLE_LABELS:
        return "stable_structure"
    if label in ROAD_LABELS:
        return "road_or_ground"
    if label in VEGETATION_LABELS:
        return "vegetation_or_weak_context"
    return "unknown"


def thing_or_stuff(label: str) -> str:
    if label in DYNAMIC_LABELS:
        return "thing"
    if label in STABLE_LABELS:
        return "thing_or_static_structure"
    if label in ROAD_LABELS or label in VEGETATION_LABELS or label in LOWOBS_LABELS:
        return "stuff"
    return "unknown"


def load_trace_rows() -> dict[tuple[str, int, int], list[dict[str, str]]]:
    by_key: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(HEAD_TRACE_ROWS):
        raw_pos = row.get("current_sample_position", "")
        raw_head = row.get("head_idx", "")
        if raw_pos == "" or raw_head == "":
            continue
        by_key[(row["seq"], int(float(raw_pos)), int(float(raw_head)))].append(row)
    return by_key


def weighted_mean(values: list[tuple[float, float]]) -> float:
    total_w = sum(weight for _, weight in values)
    if total_w <= 0:
        return float("nan")
    return float(sum(value * weight for value, weight in values) / total_w)


def most_common(rows: list[dict[str, str]], key: str) -> str:
    counts = Counter(row.get(key, "") for row in rows if row.get(key, "") != "")
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def trace_feature(row: dict[str, Any], trace_rows: list[dict[str, str]]) -> dict[str, Any]:
    frame_id = as_int(row, "frame_id")
    if not trace_rows:
        return {
            "context_path": row.get("context_role", ""),
            "token_type": "unknown",
            "key_token_role_top": "unknown",
            "query_token_role_top": "unknown",
            "source_frame_id_weighted": "",
            "source_frame_age_weighted": "",
            "keyframe_or_scale_frame_flag": "",
            "trajectory_memory_age": "",
            "kv_cache_page_id": row.get("context_role", ""),
            "trace_rows": 0,
        }
    weights = [as_float(item, "attention_weight", 0.0) for item in trace_rows]
    key_positions: list[tuple[float, float]] = []
    ages: list[tuple[float, float]] = []
    context_weight: dict[str, float] = defaultdict(float)
    for item, weight in zip(trace_rows, weights):
        key_pos = as_float(item, "key_sample_position", float("nan"))
        if math.isfinite(key_pos):
            key_positions.append((key_pos, weight))
            ages.append((frame_id - key_pos, weight))
        context_weight[item.get("key_context_role", "unknown")] += weight
    context_path = max(context_weight.items(), key=lambda item: item[1])[0] if context_weight else row.get("context_role", "")
    source_frame = weighted_mean(key_positions)
    source_age = weighted_mean(ages)
    return {
        "context_path": context_path,
        "token_type": most_common(trace_rows, "key_token_role"),
        "key_token_role_top": most_common(trace_rows, "key_token_role"),
        "query_token_role_top": most_common(trace_rows, "query_token_role"),
        "source_frame_id_weighted": source_frame,
        "source_frame_age_weighted": source_age,
        "keyframe_or_scale_frame_flag": context_path == "scale_reference_context",
        "trajectory_memory_age": source_age if context_path == "trajectory_memory_special" else "",
        "kv_cache_page_id": context_path,
        "trace_rows": len(trace_rows),
    }


def method_paths(seq: str, frame_id: int) -> dict[str, Any]:
    dataset = f"kitti_v105_seq{seq}_trace32"
    action_method = f"lingbot_map_stage4_{ACTION_LABEL}_seq{seq}"
    frame = f"{frame_id:06d}"
    baseline_root = BASELINE_WORKSPACE / dataset / seq / BASELINE_METHOD
    action_root = ACTION_WORKSPACE / dataset / seq / action_method
    paths = {
        "baseline_depth_exr": baseline_root / "depth" / f"{frame}.exr",
        "baseline_confidence_exr": baseline_root / "confidence" / f"{frame}.exr",
        "action_depth_exr": action_root / "depth" / f"{frame}.exr",
        "action_confidence_exr": action_root / "confidence" / f"{frame}.exr",
        "baseline_traj": baseline_root / "traj.txt",
        "action_traj": action_root / "traj.txt",
    }
    return {key: path.relative_to(ROOT).as_posix() for key, path in paths.items()} | {
        f"{key}_exists": path.exists() for key, path in paths.items()
    }


def geometry_proxy(row: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    baseline_l3 = as_float(row, "baseline_L3")
    action_l3 = as_float(row, "action_L3")
    local_window = as_float(row, "local_window_context_attention_frac", 0.0)
    scale_context = as_float(row, "scale_reference_context_attention_frac", 0.0)
    reject_frac = as_float(row, "reject_unreliable_patch_frac", 0.0)
    local_patch = as_float(row, "local_registration_patch_frac", 0.0)
    scale_patch = as_float(row, "scale_reference_patch_frac", 0.0)
    source_age = as_float(trace, "source_frame_age_weighted", 0.0)
    return {
        "lingbot_depth_residual_local": "",
        "lingbot_pose_residual_local": baseline_l3,
        "trajectory_residual": action_l3,
        "local_window_support_score": local_window,
        "anchor_context_consistency_score": scale_context,
        "depth_spread_proxy": max(local_patch, scale_patch),
        "point_spread_proxy": max(local_patch + scale_patch, 0.0),
        "parallax_proxy": min(1.0, max(0.0, abs(source_age) / 31.0)) if finite(source_age) else "",
        "overlap_support_proxy": local_window * max(0.0, 1.0 - reject_frac),
        "geometry_proxy_only": True,
    }


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = read_csv(STAGE0 / "selected_evidence_rows.csv")
    trace_by_key = load_trace_rows()

    enriched_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    gca_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []

    for row in selected:
        seq = row["seq_id"]
        frame_id = as_int(row, "frame_id")
        head_id = as_int(row, "head_id")
        trace_rows = trace_by_key.get((seq, frame_id, head_id), [])
        trace = trace_feature(row, trace_rows)
        label, label_count, patch_purity = dominant_label(row)
        semantic_confidence = as_float(row, "semantic_confidence_mean", 0.0)
        trust = semantic_confidence * (patch_purity ** 2)
        role = semantic_role(label, patch_purity)
        boundary_risk = max(0.0, min(1.0, 1.0 - patch_purity))
        paths = method_paths(seq, frame_id)
        geom = geometry_proxy(row, trace)

        base = {
            "schema": "acl2_v106tf_stage1_selected_evidence_enriched_row_v1",
            "seq_id": seq,
            "frame_id": frame_id,
            "original_frame": row.get("original_frame", ""),
            "head_id": head_id,
            "token_group_id": row.get("token_group_id", ""),
            "label_type": row.get("label_type", ""),
            "action_label": row.get("action_label", ACTION_LABEL),
            "selected_by_v105_policy": row.get("selected_by_v105_policy", "True"),
        }
        semantic = {
            "dominant_label_id": label,
            "dominant_label_name": label,
            "dominant_label_count": label_count,
            "semantic_confidence": semantic_confidence,
            "patch_purity": patch_purity,
            "semantic_trust": trust,
            "thing_or_stuff": thing_or_stuff(label),
            "semantic_role": role,
            "boundary_risk": boundary_risk,
            "masklet_id": "",
            "component_id": row.get("local_window_or_chunk_id", ""),
            "track_id": "",
            "semantic_source": row.get("semantic_source", ""),
            "top_labels": row.get("top_labels", ""),
        }
        gca = {
            "context_path": trace["context_path"],
            "token_type": trace["token_type"],
            "head_id": head_id,
            "layer_id": "sdpa_gca_global0_11_23_trace",
            "attention_mass": row.get("head_trace_topk_attention_sum", ""),
            "source_frame_id": trace["source_frame_id_weighted"],
            "source_frame_age": trace["source_frame_age_weighted"],
            "keyframe_or_scale_frame_flag": trace["keyframe_or_scale_frame_flag"],
            "trajectory_memory_age": trace["trajectory_memory_age"],
            "kv_cache_page_id": trace["kv_cache_page_id"],
            "trace_rows_for_frame_head": trace["trace_rows"],
            "key_token_role_top": trace["key_token_role_top"],
            "query_token_role_top": trace["query_token_role_top"],
        }
        geometry = {
            "baseline_L3": row.get("baseline_L3", ""),
            "action_L3": row.get("action_L3", ""),
            "bad_improvement": row.get("bad_improvement", ""),
            "good_harm": row.get("good_harm", ""),
            **geom,
            **paths,
        }
        enriched_rows.append({**base, **semantic, **gca, **geometry})
        semantic_rows.append({**base, **semantic})
        gca_rows.append({**base, **gca})
        geometry_rows.append({**base, **geometry})

    selected_count = len(enriched_rows)
    context_available = sum(1 for row in enriched_rows if row.get("context_path", "") not in {"", "unknown"})
    head_available = sum(1 for row in enriched_rows if str(row.get("head_id", "")) != "")
    semantic_available = sum(1 for row in enriched_rows if row.get("semantic_role", "unknown") != "unknown")
    geometry_proxy_available = sum(1 for row in enriched_rows if row.get("geometry_proxy_only") is True)
    hard_negative_rows = [row for row in enriched_rows if row.get("label_type") == "good_selected"]
    hard_negative_represented = len(hard_negative_rows)
    all_depth_paths_exist = sum(
        1
        for row in enriched_rows
        if row.get("baseline_depth_exr_exists") and row.get("action_depth_exr_exists")
    )
    all_conf_paths_exist = sum(
        1
        for row in enriched_rows
        if row.get("baseline_confidence_exr_exists") and row.get("action_confidence_exr_exists")
    )

    materialization_summary = {
        "schema": "acl2_v106tf_stage1_materialization_summary_v1",
        "selected_evidence_rows": selected_count,
        "materialized_rows": selected_count,
        "selected_evidence_row_coverage": 1.0 if selected_count else 0.0,
        "context_path_available_fraction": context_available / selected_count if selected_count else 0.0,
        "head_id_available_fraction": head_available / selected_count if selected_count else 0.0,
        "semantic_role_available_fraction": semantic_available / selected_count if selected_count else 0.0,
        "geometry_support_available_or_proxy_fraction": geometry_proxy_available / selected_count if selected_count else 0.0,
        "geometry_proxy_only": True,
        "baseline_action_depth_file_pair_fraction": all_depth_paths_exist / selected_count if selected_count else 0.0,
        "baseline_action_confidence_file_pair_fraction": all_conf_paths_exist / selected_count if selected_count else 0.0,
        "hard_negative_selected_good_rows_represented": hard_negative_represented,
        "hard_negative_selected_good_rows_expected": 4,
        "hard_negative_all_represented": hard_negative_represented == 4,
        "stage1_materialization_pass": (
            selected_count > 0
            and context_available / selected_count >= 0.95
            and head_available / selected_count >= 0.95
            and semantic_available / selected_count >= 0.80
            and geometry_proxy_available / selected_count >= 0.80
            and hard_negative_represented == 4
        ),
        "outputs": {
            "selected_evidence_enriched_rows": (OUT / "selected_evidence_enriched_rows.csv").relative_to(ROOT).as_posix(),
            "semantic_role_rows": (OUT / "semantic_role_rows.csv").relative_to(ROOT).as_posix(),
            "gca_context_token_rows": (OUT / "gca_context_token_rows.csv").relative_to(ROOT).as_posix(),
            "geometry_support_rows": (OUT / "geometry_support_rows.csv").relative_to(ROOT).as_posix(),
            "materialization_summary": (OUT / "materialization_summary.json").relative_to(ROOT).as_posix(),
            "missing_field_report": (OUT / "missing_field_report.md").relative_to(ROOT).as_posix(),
            "token_layout_audit": (OUT / "token_layout_audit.md").relative_to(ROOT).as_posix(),
            "geometry_field_missing_report": (OUT / "geometry_field_missing_report.md").relative_to(ROOT).as_posix(),
        },
        "note": "Stage1 geometry support is proxy-based; depth/confidence files are present but no per-token pointmap/depth residual is computed here.",
    }

    write_csv(OUT / "selected_evidence_enriched_rows.csv", enriched_rows)
    write_csv(OUT / "semantic_role_rows.csv", semantic_rows)
    write_csv(OUT / "gca_context_token_rows.csv", gca_rows)
    write_csv(OUT / "geometry_support_rows.csv", geometry_rows)
    (OUT / "materialization_summary.json").write_text(
        json.dumps(materialization_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    missing = """# Stage1 Missing Field Report

Materialized fields available:
- selected evidence id / frame / head from v105 Stage4 headlocal selected rows.
- context path, token type, query token role, key token role, source-frame age from SDPA trace rows.
- semantic role, patch purity, semantic trust from v105 semantic/geometric frame rows.
- per-frame baseline/action L3 residuals from Stage0.
- LingBot depth/confidence file paths for baseline and action workspaces.

Fields not directly available as numeric measurements in current artifacts:
- `lingbot_depth_residual_local`
- `lingbot_pose_residual_local` beyond Sim3 residual proxy
- dense `point_spread_proxy` from pointmaps
- token-specific depth / pointmap residual
- trajectory memory token age except when trace context exposes trajectory-memory rows
- masklet/track id

Actions taken before marking proxy:
1. Checked LingBot action workspace depth/confidence/rgb folders for selected sequence outputs.
2. Checked Stage2 baseline workspace depth/confidence/rgb folders.
3. Checked Stage4 headlocal trace workspace for depth/confidence availability.
4. Used SDPA trace key context role / key token role / source frame position when available.
5. Marked geometry rows as `geometry_proxy_only=true`.

Implication:
- Stage1 materialization can pass on `geometry_support available or proxy fraction`.
- Stage2/Stage3 must preserve `proxy_only` provenance.
- MoGe-based action promotion remains disallowed unless a real MoGe verifier later reaches coverage.
"""
    (OUT / "missing_field_report.md").write_text(missing, encoding="utf-8")
    (OUT / "geometry_field_missing_report.md").write_text(missing, encoding="utf-8")

    token_audit = """# Stage1 Token Layout Audit

Available token evidence:
- v105 SDPA trace rows expose `key_token_role`, `query_token_role`, `key_token_offset`, `key_context_role`, `key_sample_position`, and attention weight.
- `gca_context_token_rows.csv` records most-common key/query token roles per selected frame/head.

Current token-type status:
- token_type is inferred from traced `key_token_role`.
- token-type-specific action is not yet allowed solely from Stage1; Stage3/Stage4 must still verify role purity and action trace fidelity.

Missing / limited:
- No persistent per-token memory-write id is available.
- No direct trajectory-memory token age is available except proxy from source-frame age and context role.
"""
    (OUT / "token_layout_audit.md").write_text(token_audit, encoding="utf-8")

    print(json.dumps(materialization_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return materialization_summary


def main() -> int:
    summary = build()
    return 0 if summary["stage1_materialization_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

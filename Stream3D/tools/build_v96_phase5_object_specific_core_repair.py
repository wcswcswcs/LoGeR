#!/usr/bin/env python3
"""Materialize v96 object-specific micro-query core repair artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v96_phase5_object_specific_core_repair"
RUN_ID = "v96_phase5_object_specific_core_repair"
DEFAULT_PHASE5 = ROOT / "outputs/audit/v96_phase5_object_birth_w0020_segmented_r4_D3_repair5_overlap090_sceneoffset"
DEFAULT_INCIDENCE = ROOT / "outputs/audit/v96_phase3_triton_incidence_w0020_segmented_r4_D3_repair1"
DEFAULT_FEATURE = ROOT / "outputs/audit/v96_phase4_affinity_features_w0020_segmented_r4_D3"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_features(root: Path, decode_variant: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in _read_csv(root / "micro_feature_rows.csv"):
        if row.get("decode_variant") != decode_variant:
            continue
        qid = row.get("query_id", "")
        if not qid:
            continue
        out[qid] = {
            "event_count": _num(row.get("event_count")),
            "positive_rate": _num(row.get("positive_rate")),
            "dominant_mask_ratio": _num(row.get("dominant_mask_ratio")),
            "multi_mask_rate": _num(row.get("multi_mask_rate")),
            "boundary_near_rate": _num(row.get("boundary_near_rate")),
            "distinct_mask_count_mean": _num(row.get("distinct_mask_count_mean")),
            "frame_span": _num(row.get("frame_span")),
            "feature_norm": _num(row.get("feature_norm")),
        }
    return out


def _load_incidence(root: Path, decode_variant: str) -> tuple[dict[tuple[str, str, int, int], set[str]], dict[str, list[dict[str, str]]]]:
    node_to_qids: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    events_by_query: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(root / "incidence_event_rows.csv"):
        if row.get("decode_variant") != decode_variant:
            continue
        qid = row.get("query_id", "")
        if not qid:
            continue
        events_by_query[qid].append(row)
        mask_id = int(_num(row.get("center_mask_id")))
        if mask_id > 0:
            key = (
                row.get("scene_id", ""),
                row.get("window_id", ""),
                int(_num(row.get("target_frame_id"))),
                mask_id,
            )
            node_to_qids[key].add(qid)
    return node_to_qids, events_by_query


def _score_query_for_object(
    qid: str,
    selected_nodes: set[tuple[str, str, int, int]],
    selected_frame_count: int,
    events_by_query: dict[str, list[dict[str, str]]],
    features: dict[str, dict[str, float]],
) -> dict[str, Any]:
    events = events_by_query.get(qid, [])
    event_count = len(events)
    selected_hits = 0
    hit_frames: set[tuple[str, str, int]] = set()
    positive_events = 0
    for row in events:
        mask_id = int(_num(row.get("center_mask_id")))
        if mask_id > 0:
            positive_events += 1
        key = (
            row.get("scene_id", ""),
            row.get("window_id", ""),
            int(_num(row.get("target_frame_id"))),
            mask_id,
        )
        if key in selected_nodes:
            selected_hits += 1
            hit_frames.add((key[0], key[1], key[2]))
    feat = features.get(qid, {})
    object_hit_ratio = float(selected_hits / max(1, event_count))
    frame_coverage = float(len(hit_frames) / max(1, selected_frame_count))
    positive_rate = float(feat.get("positive_rate", positive_events / max(1, event_count)))
    multi_mask_rate = float(feat.get("multi_mask_rate", 0.0))
    boundary_near_rate = float(feat.get("boundary_near_rate", 0.0))
    dominant_mask_ratio = float(feat.get("dominant_mask_ratio", 0.0))
    stability = max(0.0, 1.0 - 0.55 * multi_mask_rate - 0.35 * boundary_near_rate)
    diversity_guard = max(0.35, 1.0 - 0.35 * dominant_mask_ratio)
    core_score = object_hit_ratio * math.sqrt(frame_coverage) * positive_rate * stability * diversity_guard
    return {
        "query_id": qid,
        "core_score": float(core_score),
        "object_hit_ratio": float(object_hit_ratio),
        "selected_hit_count": int(selected_hits),
        "selected_frame_hit_count": int(len(hit_frames)),
        "selected_frame_coverage": float(frame_coverage),
        "event_count": int(event_count),
        "positive_rate": float(positive_rate),
        "multi_mask_rate": float(multi_mask_rate),
        "boundary_near_rate": float(boundary_near_rate),
        "dominant_mask_ratio": float(dominant_mask_ratio),
        "feature_norm": float(feat.get("feature_norm", 0.0)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    phase5_root = _project(args.phase5_root)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_source = [
        row for row in _read_csv(phase5_root / "selected_masklet_rows.csv") if row.get("family") == args.source_family
    ]
    if not selected_source:
        raise ValueError(f"no selected masklet rows for family {args.source_family}")
    node_to_qids, events_by_query = _load_incidence(_project(args.incidence_root), args.decode_variant)
    features = _load_features(_project(args.feature_root), args.decode_variant)

    source_to_output: dict[str, str] = {}
    for idx, source_object_id in enumerate(sorted({row.get("object_id", "") for row in selected_source}), start=1):
        source_to_output[source_object_id] = f"{args.output_family}_obj_{idx:06d}"

    source_rows_by_object: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_source:
        source_rows_by_object[row.get("object_id", "")].append(row)

    selected_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    object_micro_rows: list[dict[str, Any]] = []
    core_metric_rows: list[dict[str, Any]] = []
    dropped_by_score = 0
    fallback_selected_count = 0
    empty_core_object_count = 0
    for source_object_id, rows in sorted(source_rows_by_object.items()):
        output_object_id = source_to_output[source_object_id]
        selected_nodes = {
            (
                row.get("scene_id", ""),
                row.get("window_id", ""),
                int(_num(row.get("frame_id"))),
                int(_num(row.get("selected_mask_id"))),
            )
            for row in rows
        }
        candidate_qids: set[str] = set()
        for node in selected_nodes:
            candidate_qids.update(node_to_qids.get(node, set()))
        scored = [
            _score_query_for_object(qid, selected_nodes, len(rows), events_by_query, features)
            for qid in sorted(candidate_qids)
        ]
        scored.sort(
            key=lambda item: (
                -float(item["core_score"]),
                -int(item["selected_frame_hit_count"]),
                -int(item["selected_hit_count"]),
                str(item["query_id"]),
            )
        )
        filtered = [
            item
            for item in scored
            if float(item["core_score"]) >= float(args.min_core_score)
            and int(item["selected_hit_count"]) >= int(args.min_selected_hit_count)
            and int(item["selected_frame_hit_count"]) >= int(args.min_frame_hit_count)
        ]
        dropped_by_score += max(0, len(scored) - len(filtered))
        if len(filtered) < int(args.min_core_qids):
            fallback = scored[: int(args.min_core_qids)]
            fallback_selected_count += max(0, len(fallback) - len(filtered))
            filtered = fallback
        selected_core = filtered[: int(args.max_qids_per_object)]
        if not selected_core:
            empty_core_object_count += 1

        support_sum = 0
        for row in rows:
            out = dict(row)
            out["family"] = args.output_family
            out["object_id"] = output_object_id
            out["source_family"] = args.source_family
            out["source_object_id"] = source_object_id
            out["selection_status"] = "selected_with_explicit_object_specific_core"
            selected_rows.append(out)
            support_sum += int(_num(row.get("masklet_support_query_count")))

        for rank, item in enumerate(selected_core, start=1):
            object_micro_rows.append(
                {
                    "schema_version": "stream4d_v96_object_micro_query_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "family": args.output_family,
                    "object_id": output_object_id,
                    "source_family": args.source_family,
                    "source_object_id": source_object_id,
                    "query_id": item["query_id"],
                    "core_rank": int(rank),
                    "core_score": item["core_score"],
                    "object_hit_ratio": item["object_hit_ratio"],
                    "selected_hit_count": item["selected_hit_count"],
                    "selected_frame_hit_count": item["selected_frame_hit_count"],
                    "selected_frame_coverage": item["selected_frame_coverage"],
                    "event_count": item["event_count"],
                    "positive_rate": item["positive_rate"],
                    "multi_mask_rate": item["multi_mask_rate"],
                    "boundary_near_rate": item["boundary_near_rate"],
                    "dominant_mask_ratio": item["dominant_mask_ratio"],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

        scores = [float(item["core_score"]) for item in selected_core]
        object_rows.append(
            {
                "schema_version": "stream4d_v96_object_candidate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "family": args.output_family,
                "object_id": output_object_id,
                "source_family": args.source_family,
                "source_object_id": source_object_id,
                "group_id": rows[0].get("group_id", ""),
                "micro_query_count": int(len(selected_core)),
                "candidate_micro_query_count": int(len(scored)),
                "selected_frame_count_before_collision_resolution": int(len(rows)),
                "selected_frame_count": int(len(rows)),
                "masklet_support_query_count_sum": int(support_sum),
                "object_score": float(sum(scores) / max(1, len(scores))),
                "core_score_max": float(max(scores)) if scores else 0.0,
                "core_score_min": float(min(scores)) if scores else 0.0,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        core_metric_rows.append(
            {
                "schema_version": "stream4d_v96_object_specific_core_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "family": args.output_family,
                "object_id": output_object_id,
                "source_object_id": source_object_id,
                "candidate_micro_query_count": int(len(scored)),
                "selected_core_query_count": int(len(selected_core)),
                "selected_frame_count": int(len(rows)),
                "core_score_mean": float(sum(scores) / max(1, len(scores))),
                "core_score_max": float(max(scores)) if scores else 0.0,
                "core_score_p10": float(sorted(scores)[max(0, int(0.1 * (len(scores) - 1)))]) if scores else 0.0,
                "fallback_used": len(filtered) < int(args.min_core_qids),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    _write_csv(output_root / "selected_masklet_rows.csv", selected_rows)
    _write_csv(output_root / "object_candidate_rows.csv", object_rows)
    _write_csv(output_root / "object_micro_query_rows.csv", object_micro_rows)
    _write_csv(output_root / "object_specific_core_metric_rows.csv", core_metric_rows)
    summary = {
        "schema": "stream4d_v96_object_specific_core_repair_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "MATERIALIZED_V96_OBJECT_SPECIFIC_CORE_REPAIR",
        "source_phase5_root": _rel(phase5_root),
        "source_family": args.source_family,
        "output_root": _rel(output_root),
        "output_family": args.output_family,
        "decode_variant": args.decode_variant,
        "object_count": len(object_rows),
        "selected_masklet_count": len(selected_rows),
        "object_micro_query_count": len(object_micro_rows),
        "empty_core_object_count": int(empty_core_object_count),
        "dropped_by_score_or_hit_count": int(dropped_by_score),
        "fallback_selected_count": int(fallback_selected_count),
        "max_qids_per_object": int(args.max_qids_per_object),
        "min_core_qids": int(args.min_core_qids),
        "min_core_score": float(args.min_core_score),
        "min_selected_hit_count": int(args.min_selected_hit_count),
        "min_frame_hit_count": int(args.min_frame_hit_count),
        "runtime_total_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "object_count": summary["object_count"], "object_micro_query_count": summary["object_micro_query_count"], "output_root": summary["output_root"]}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize v96 object-specific micro-query core repair.")
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5))
    parser.add_argument("--source-family", default="C_hybrid_cover_cluster")
    parser.add_argument("--output-family", default="C_object_specific_core")
    parser.add_argument("--incidence-root", default=str(DEFAULT_INCIDENCE))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE))
    parser.add_argument("--decode-variant", default="D3_adaptive1024")
    parser.add_argument("--max-qids-per-object", type=int, default=128)
    parser.add_argument("--min-core-qids", type=int, default=16)
    parser.add_argument("--min-core-score", type=float, default=0.20)
    parser.add_argument("--min-selected-hit-count", type=int, default=2)
    parser.add_argument("--min-frame-hit-count", type=int, default=2)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

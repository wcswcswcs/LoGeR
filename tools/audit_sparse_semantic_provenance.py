#!/usr/bin/env python3
"""Audit sparse masklet semantic/provenance fields.

This is a CPU-only inspection tool. It does not run detectors, SAM, VOS, or
postprocessing. The goal is to make it explicit whether a sparse track came
from the main YOLOE->SAM3 chain, proposal fallback, stuff, or a later append
tool that reused the generic ``thing_tracked`` source_type.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit sparse_masklets.pt semantic provenance.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--focus_labels", default="traffic sign,pole,car,person")
    parser.add_argument("--metrics_json", default="")
    return parser.parse_args()


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _load_metrics(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {"missing_metrics_json": str(metrics_path)}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"failed_metrics_json": str(metrics_path), "error": str(exc)}


def _frames(track: Dict[str, Any]) -> List[int]:
    return [int(v) for v in track.get("frames", [])]


def _tensor_values(value: Any) -> List[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        arr = value.detach().cpu().reshape(-1).tolist()
        return [float(v) for v in arr]
    try:
        return [float(v) for v in value]
    except Exception:
        return []


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / max(len(vals), 1))


def _infer_origin(
    track_index: int,
    track: Dict[str, Any],
    debug: Dict[str, Any],
) -> Dict[str, Any]:
    source_type = str(track.get("source_type", ""))
    tracking_source = str(track.get("tracking_source", ""))
    proposal_source = str(track.get("proposal_source", ""))
    mask_source = str(track.get("mask_source", ""))
    label_source = str(track.get("label_source", ""))
    semantic_resolver = str(track.get("semantic_resolver", ""))
    sam3_status_field = str(track.get("sam3_status", ""))
    track_metadata = debug.get("track_metadata", {}) if isinstance(debug.get("track_metadata", {}), dict) else {}
    meta = track_metadata.get(str(track_index), {})
    refine_status = str(meta.get("refine_status", track.get("_refine_status", "")))

    append_debug = debug.get("append_yoloe_static_thing_tracks")
    append_input_tracks: Optional[int] = None
    if isinstance(append_debug, dict) and "input_tracks" in append_debug:
        try:
            append_input_tracks = int(append_debug["input_tracks"])
        except Exception:
            append_input_tracks = None

    if tracking_source == "yoloe_static_frame_assoc":
        origin = "postprocess_yoloe_static_append"
        reason = "tracking_source=yoloe_static_frame_assoc"
    elif tracking_source == "sam31_multiplex_refinement":
        origin = "main_yoloe_sam31_multiplex"
        reason = "tracking_source=sam31_multiplex_refinement"
    elif tracking_source == "sam3_refinement":
        origin = "main_yoloe_sam3"
        reason = "tracking_source=sam3_refinement"
    elif tracking_source == "yoloe_proposal_tracklet":
        origin = "main_yoloe_proposal_only"
        reason = "tracking_source=yoloe_proposal_tracklet"
    elif tracking_source == "yoloe_proposal_tracklet_after_sam3_failed":
        origin = "main_yoloe_sam3_failed"
        reason = "tracking_source=yoloe_proposal_tracklet_after_sam3_failed"
    elif tracking_source == "stuff_backend":
        origin = "stuff_static"
        reason = "tracking_source=stuff_backend"
    elif append_input_tracks is not None and track_index >= append_input_tracks:
        origin = "postprocess_yoloe_static_append"
        reason = f"track_index >= append_input_tracks ({append_input_tracks})"
    elif source_type == "stuff_static":
        origin = "stuff_static"
        reason = "source_type=stuff_static"
    elif refine_status == "sam3_refined":
        origin = "main_yoloe_sam3"
        reason = "track_metadata.refine_status=sam3_refined"
    elif refine_status == "proposal_only":
        origin = "main_yoloe_proposal_only"
        reason = "track_metadata.refine_status=proposal_only"
    elif refine_status == "sam3_failed":
        origin = "main_yoloe_sam3_failed"
        reason = "track_metadata.refine_status=sam3_failed"
    elif source_type == "thing_proposal_fallback":
        origin = "postprocess_thing_proposal_fallback"
        reason = "source_type=thing_proposal_fallback"
    elif source_type == "thing_tracked":
        origin = "thing_tracked_unknown"
        reason = "source_type=thing_tracked without preserved refine metadata"
    else:
        origin = "unknown"
        reason = f"source_type={source_type}"

    return {
        "origin": origin,
        "origin_reason": reason,
        "refine_status": refine_status,
        "append_input_tracks": append_input_tracks,
        "tracking_source": tracking_source,
        "proposal_source": proposal_source,
        "mask_source": mask_source,
        "label_source": label_source,
        "semantic_resolver": semantic_resolver,
        "sam3_status": sam3_status_field,
    }


def _track_row(
    track_index: int,
    track: Dict[str, Any],
    debug: Dict[str, Any],
) -> Dict[str, Any]:
    frames = _frames(track)
    scores = _tensor_values(track.get("scores"))
    areas = _tensor_values(track.get("area_ratio"))
    origin = _infer_origin(track_index, track, debug)
    return {
        "track_index": int(track_index),
        "label": str(track.get("L_sem", "")),
        "group": int(track.get("G_sem", -1)),
        "source_type": str(track.get("source_type", "")),
        "provenance_origin": origin["origin"],
        "origin_reason": origin["origin_reason"],
        "refine_status": origin["refine_status"],
        "mask_source": origin["mask_source"],
        "proposal_source": origin["proposal_source"],
        "tracking_source": origin["tracking_source"],
        "label_source": origin["label_source"],
        "semantic_resolver": origin["semantic_resolver"],
        "sam3_status": origin["sam3_status"],
        "frames_count": int(len(frames)),
        "start_frame": int(min(frames)) if frames else "",
        "end_frame": int(max(frames)) if frames else "",
        "mean_score": _mean(scores),
        "max_score": max(scores) if scores else 0.0,
        "mean_area_ratio": _mean(areas),
        "max_area_ratio": max(areas) if areas else 0.0,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _counter_dict(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}


def build_report(payload: Dict[str, Any], rows: List[Dict[str, Any]], focus_labels: List[str], metrics: Dict[str, Any]) -> Dict[str, Any]:
    debug = payload.get("debug") or {}
    label_counts = Counter(row["label"] for row in rows)
    source_counts = Counter(row["source_type"] for row in rows)
    origin_counts = Counter(row["provenance_origin"] for row in rows)
    label_origin_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (label, origin), count in Counter((row["label"], row["provenance_origin"]) for row in rows).items():
        label_origin_counts[str(label)][str(origin)] = int(count)

    append_debug = debug.get("append_yoloe_static_thing_tracks")
    append_present = isinstance(append_debug, dict)
    append_input_tracks = None
    if append_present and "input_tracks" in append_debug:
        try:
            append_input_tracks = int(append_debug["input_tracks"])
        except Exception:
            append_input_tracks = None

    focus: Dict[str, Any] = {}
    for label in focus_labels:
        label_rows = [row for row in rows if row["label"] == label]
        focus[label] = {
            "track_count": int(len(label_rows)),
            "origin_counts": _counter_dict(Counter(row["provenance_origin"] for row in label_rows)),
            "source_type_counts": _counter_dict(Counter(row["source_type"] for row in label_rows)),
            "main_yoloe_sam3_tracks": int(sum(1 for row in label_rows if row["provenance_origin"] == "main_yoloe_sam3")),
            "postprocess_append_tracks": int(sum(1 for row in label_rows if row["provenance_origin"] == "postprocess_yoloe_static_append")),
            "unknown_thing_tracks": int(sum(1 for row in label_rows if row["provenance_origin"] == "thing_tracked_unknown")),
        }

    warnings: List[str] = []
    if append_present:
        warnings.append("append_yoloe_static_thing_tracks debug is present; source_type=thing_tracked is not enough to prove SAM3 tracking.")
    if append_present and append_input_tracks is None:
        warnings.append("append debug exists but input_tracks is missing/unparseable; appended-track boundary is ambiguous.")
    if any(row["provenance_origin"] == "thing_tracked_unknown" for row in rows):
        warnings.append("Some thing_tracked rows lack preserved refine metadata; SAM3/proposal/postprocess attribution is incomplete.")

    metrics_view = {
        key: metrics.get(key)
        for key in (
            "thing_prompts",
            "num_raw_proposals",
            "num_filtered_proposals",
            "num_tracklets_total",
            "num_tracklets_confirmed",
            "num_sam3_attempted",
            "num_sam3_success",
            "num_output_tracks",
        )
        if key in metrics
    }

    return {
        "input_format": payload.get("format"),
        "num_tracks": int(len(rows)),
        "num_frames": int(payload.get("num_frames", 0)),
        "frame_height": int(payload.get("frame_height", 0)),
        "frame_width": int(payload.get("frame_width", 0)),
        "debug_keys": sorted(str(k) for k in debug.keys()) if isinstance(debug, dict) else [],
        "append_debug_present": bool(append_present),
        "append_input_tracks": append_input_tracks,
        "label_counts": _counter_dict(label_counts),
        "source_type_counts": _counter_dict(source_counts),
        "provenance_origin_counts": _counter_dict(origin_counts),
        "label_origin_counts": {k: {str(ok): int(ov) for ok, ov in sorted(v.items())} for k, v in label_origin_counts.items()},
        "focus_labels": focus,
        "metrics_view": _jsonable(metrics_view),
        "warnings": warnings,
    }


def write_markdown(path: Path, input_pt: str, summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Sparse Semantic Provenance Audit")
    lines.append("")
    lines.append(f"input_pt: `{input_pt}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- format: `{summary['input_format']}`")
    lines.append(f"- tracks: `{summary['num_tracks']}`")
    lines.append(f"- frames: `{summary['num_frames']}`")
    lines.append(f"- debug_keys: `{', '.join(summary['debug_keys'])}`")
    lines.append(f"- append_debug_present: `{summary['append_debug_present']}`")
    lines.append(f"- append_input_tracks: `{summary['append_input_tracks']}`")
    lines.append("")
    lines.append("## Provenance Origin Counts")
    lines.append("")
    for key, value in summary["provenance_origin_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Label Counts")
    lines.append("")
    for key, value in summary["label_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Focus Labels")
    lines.append("")
    for label, item in summary["focus_labels"].items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- tracks: `{item['track_count']}`")
        lines.append(f"- main_yoloe_sam3_tracks: `{item['main_yoloe_sam3_tracks']}`")
        lines.append(f"- postprocess_append_tracks: `{item['postprocess_append_tracks']}`")
        lines.append(f"- unknown_thing_tracks: `{item['unknown_thing_tracks']}`")
        lines.append(f"- origin_counts: `{json.dumps(item['origin_counts'], ensure_ascii=False)}`")
        lines.append("")
    if summary["metrics_view"]:
        lines.append("## Metrics View")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(summary["metrics_view"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    if summary["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_pt = Path(args.input_pt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(input_pt, map_location="cpu", weights_only=False)
    if payload.get("format") != "sparse_masklets_v1":
        raise SystemExit(f"unsupported sparse format: {payload.get('format')}")
    debug = payload.get("debug") or {}
    rows = [_track_row(idx, track, debug) for idx, track in enumerate(payload.get("tracks", []))]
    metrics = _load_metrics(args.metrics_json)
    summary = build_report(payload, rows, _split_csv(args.focus_labels), metrics)
    summary["input_pt"] = str(input_pt)
    summary["metrics_json"] = str(args.metrics_json or "")

    (out_dir / "provenance_summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_dir / "track_rows.csv", rows)
    write_markdown(out_dir / "provenance_report.md", str(input_pt), summary)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _spec_name(semantic_threshold: float, p4_max: float) -> str:
    sem = f"{semantic_threshold:g}".replace(".", "p")
    p4 = f"{p4_max:g}".replace(".", "p")
    return f"derived_semantic_le_{sem}_p4lt_{p4}"


def _parse_derived_negative_specs(raw: str) -> dict[str, tuple[float, float]]:
    specs: dict[str, tuple[float, float]] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 2:
            semantic_threshold = float(parts[0])
            p4_max = float(parts[1])
            name = _spec_name(semantic_threshold, p4_max)
        elif len(parts) == 3:
            name = parts[0]
            semantic_threshold = float(parts[1])
            p4_max = float(parts[2])
        else:
            raise ValueError(f"invalid derived negative spec: {item}")
        specs[name] = (semantic_threshold, p4_max)
    return specs


def _is_negative_row(row: dict[str, Any], negative_key: str, derived_specs: dict[str, tuple[float, float]]) -> bool:
    if negative_key in derived_specs:
        semantic_threshold, p4_max = derived_specs[negative_key]
        return bool(
            _parse_float(row.get("P6_feature_only")) <= float(semantic_threshold)
            and _parse_float(row.get("P4_vc_q_temporal")) < float(p4_max)
        )
    return _parse_bool(row.get(negative_key))


def _read_rows(input_root: Path, scene: str) -> list[dict[str, Any]]:
    path = input_root / "raw_visual_semantic_edge_rows.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if row.get("scene") == scene]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p75": None, "p90": None, "max": None}
    ordered = sorted(values)

    def q(frac: float) -> float:
        idx = min(len(ordered) - 1, int(round(frac * (len(ordered) - 1))))
        return float(ordered[idx])

    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": float(ordered[-1]),
    }


def _edge_brief(row: dict[str, Any], *, input_root: Path, positive_key: str, negative_key: str | None = None) -> dict[str, Any]:
    out = {
        "input_root": str(input_root),
        "input_root_name": input_root.name,
        "scene": row.get("scene"),
        "left_node_id": row.get("left_node_id"),
        "right_node_id": row.get("right_node_id"),
        "left_frame_id": row.get("left_frame_id"),
        "right_frame_id": row.get("right_frame_id"),
        "observation_frame_gap": row.get("observation_frame_gap"),
        "left_gt": row.get("left_gt"),
        "right_gt": row.get("right_gt"),
        "diagnostic_same_gt": row.get("diagnostic_same_gt"),
        "shared_carrier_jaccard": row.get("shared_carrier_jaccard"),
        "P1_adjacent_temporal": row.get("P1_adjacent_temporal"),
        "P2_raw_view_consensus": row.get("P2_raw_view_consensus"),
        "P3_view_consensus_q": row.get("P3_view_consensus_q"),
        "P4_vc_q_temporal": row.get("P4_vc_q_temporal"),
        positive_key: row.get(positive_key),
        "P6_feature_only": row.get("P6_feature_only"),
    }
    if negative_key is not None:
        out["negative_key"] = negative_key
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="v46 edge construction failure autopsy from raw edge rows.")
    parser.add_argument("--input-roots", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--positive-key", default="P5_p4_semantic_boost_capped")
    parser.add_argument("--negative-keys", default="")
    parser.add_argument("--derived-negative-specs", default="")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    input_roots = [Path(item) for item in str(args.input_roots).split(",") if item]
    negative_keys = [item for item in str(args.negative_keys).split(",") if item]
    derived_specs = _parse_derived_negative_specs(str(args.derived_negative_specs))
    positive_key = str(args.positive_key)
    top_k = int(args.top_k)
    out = Path(args.output_root)

    summary_rows: list[dict[str, Any]] = []
    top_false_positive_rows: list[dict[str, Any]] = []
    top_false_negative_rows: list[dict[str, Any]] = []
    negative_false_cut_rows: list[dict[str, Any]] = []
    negative_summary_rows: list[dict[str, Any]] = []

    for input_root in input_roots:
        rows = _read_rows(input_root, str(args.scene))
        same_rows = [row for row in rows if _parse_bool(row.get("diagnostic_same_gt"))]
        diff_rows = [row for row in rows if not _parse_bool(row.get("diagnostic_same_gt"))]
        summary_rows.append(
            {
                "input_root": str(input_root),
                "input_root_name": input_root.name,
                "scene": str(args.scene),
                "edge_count": len(rows),
                "same_gt_edge_count": len(same_rows),
                "diff_gt_edge_count": len(diff_rows),
                "same_gt_positive_score": _quantiles([_parse_float(row.get(positive_key)) for row in same_rows]),
                "diff_gt_positive_score": _quantiles([_parse_float(row.get(positive_key)) for row in diff_rows]),
                "same_gt_p4": _quantiles([_parse_float(row.get("P4_vc_q_temporal")) for row in same_rows]),
                "diff_gt_p4": _quantiles([_parse_float(row.get("P4_vc_q_temporal")) for row in diff_rows]),
                "same_gt_semantic": _quantiles([_parse_float(row.get("P6_feature_only")) for row in same_rows]),
                "diff_gt_semantic": _quantiles([_parse_float(row.get("P6_feature_only")) for row in diff_rows]),
            }
        )
        top_false_positive_rows.extend(
            _edge_brief(row, input_root=input_root, positive_key=positive_key)
            for row in sorted(diff_rows, key=lambda r: _parse_float(r.get(positive_key)), reverse=True)[:top_k]
        )
        top_false_negative_rows.extend(
            _edge_brief(row, input_root=input_root, positive_key=positive_key)
            for row in sorted(same_rows, key=lambda r: _parse_float(r.get(positive_key)))[:top_k]
        )
        for negative_key in negative_keys:
            flagged = [row for row in rows if _is_negative_row(row, negative_key, derived_specs)]
            false_same = [row for row in flagged if _parse_bool(row.get("diagnostic_same_gt"))]
            true_diff = [row for row in flagged if not _parse_bool(row.get("diagnostic_same_gt"))]
            negative_summary_rows.append(
                {
                    "input_root": str(input_root),
                    "input_root_name": input_root.name,
                    "scene": str(args.scene),
                    "negative_key": negative_key,
                    "negative_edge_count": len(flagged),
                    "false_same_gt_count": len(false_same),
                    "true_diff_gt_count": len(true_diff),
                    "negative_precision": None if not flagged else float(len(true_diff) / len(flagged)),
                    "false_same_p4": _quantiles([_parse_float(row.get("P4_vc_q_temporal")) for row in false_same]),
                    "true_diff_p4": _quantiles([_parse_float(row.get("P4_vc_q_temporal")) for row in true_diff]),
                    "false_same_positive_score": _quantiles([_parse_float(row.get(positive_key)) for row in false_same]),
                    "true_diff_positive_score": _quantiles([_parse_float(row.get(positive_key)) for row in true_diff]),
                    "false_same_observation_gap": _quantiles([_parse_float(row.get("observation_frame_gap")) for row in false_same]),
                    "true_diff_observation_gap": _quantiles([_parse_float(row.get("observation_frame_gap")) for row in true_diff]),
                }
            )
            negative_false_cut_rows.extend(
                _edge_brief(row, input_root=input_root, positive_key=positive_key, negative_key=negative_key)
                for row in sorted(
                    false_same,
                    key=lambda r: (_parse_float(r.get("P4_vc_q_temporal")), _parse_float(r.get(positive_key))),
                    reverse=True,
                )[:top_k]
            )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v46_edge_failure_autopsy",
        "input_roots": [str(root) for root in input_roots],
        "scene": str(args.scene),
        "positive_key": positive_key,
        "negative_keys": negative_keys,
        "derived_negative_specs": {
            name: {"semantic_threshold": spec[0], "p4_max": spec[1]} for name, spec in derived_specs.items()
        },
        "top_k": top_k,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "summary_rows": summary_rows,
        "negative_summary_rows": negative_summary_rows,
    }
    _write_json(out / "edge_failure_autopsy.json", payload)
    _write_csv(out / "edge_failure_summary_rows.csv", summary_rows)
    _write_csv(out / "negative_summary_rows.csv", negative_summary_rows)
    _write_csv(out / "top_false_positive_edges.csv", top_false_positive_rows)
    _write_csv(out / "top_false_negative_edges.csv", top_false_negative_rows)
    _write_csv(out / "negative_false_cut_edges.csv", negative_false_cut_rows)
    print(json.dumps({"summary": str(out / "edge_failure_autopsy.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

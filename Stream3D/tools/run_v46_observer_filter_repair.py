from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    pos = sum(1 for label, _score in pairs if label)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    pairs.sort(key=lambda item: item[1])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][1] == pairs[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum += avg_rank * sum(1 for label, _score in pairs[idx:end] if label)
        idx = end
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _precision(rows: list[dict[str, Any]], key: str, k: int) -> float | None:
    ranked = sorted(rows, key=lambda row: float(row.get(key) or 0.0), reverse=True)[: min(int(k), len(rows))]
    if not ranked:
        return None
    return float(sum(1 for row in ranked if str(row.get("diagnostic_same_gt")) == "True") / len(ranked))


def _mean(values: list[float]) -> float | None:
    nums = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(nums)) if nums else None


def _median(values: list[float]) -> float | None:
    nums = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(nums)) if nums else None


def _summarize(rows: list[dict[str, Any]], *, root_name: str, min_observer_count: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    scenes = sorted({str(row["scene"]) for row in rows})
    variants = sorted({str(row["variant"]) for row in rows})
    for scene in scenes:
        for variant in variants:
            subset = [
                row
                for row in rows
                if str(row["scene"]) == scene
                and str(row["variant"]) == variant
                and int(float(row.get("observer_count") or 0)) >= int(min_observer_count)
            ]
            if not subset:
                continue
            labels = [str(row.get("diagnostic_same_gt")) == "True" for row in subset]
            pos_rate = float(sum(labels) / max(len(labels), 1))
            observer_counts = [float(row.get("observer_count") or 0.0) for row in subset]
            metric_rows: dict[str, dict[str, Any]] = {}
            for key in ["raw_view_consensus", "shared_carrier_jaccard", "shuffled_view_consensus"]:
                scores = [float(row.get(key) or 0.0) for row in subset]
                metric_rows[key] = {
                    "input_root": root_name,
                    "scene": scene,
                    "variant": variant,
                    "min_observer_count": int(min_observer_count),
                    "score_key": key,
                    "edge_count": len(subset),
                    "positive_label_rate": pos_rate,
                    "mean_observer_count": _mean(observer_counts),
                    "median_observer_count": _median(observer_counts),
                    "edge_same_gt_AUC": _auc(labels, scores),
                    "edge_precision@top1k": _precision(subset, key, 1000),
                    "edge_precision@top5k": _precision(subset, key, 5000),
                }
            raw = metric_rows["raw_view_consensus"]
            shared = metric_rows["shared_carrier_jaccard"]
            shuffled = metric_rows["shuffled_view_consensus"]
            raw_auc = raw["edge_same_gt_AUC"]
            shared_auc = shared["edge_same_gt_AUC"]
            shuffled_auc = shuffled["edge_same_gt_AUC"]
            raw_p5 = raw["edge_precision@top5k"]
            shared_p5 = shared["edge_precision@top5k"]
            raw["real_minus_shared_edge_AUC"] = None if raw_auc is None or shared_auc is None else float(raw_auc - shared_auc)
            raw["real_minus_shuffled_edge_AUC"] = None if raw_auc is None or shuffled_auc is None else float(raw_auc - shuffled_auc)
            raw["precision_top5k_minus_shared"] = None if raw_p5 is None or shared_p5 is None else float(raw_p5 - shared_p5)
            raw["gate_pass"] = bool(
                raw["real_minus_shared_edge_AUC"] is not None
                and raw["real_minus_shared_edge_AUC"] >= 0.08
                and raw["real_minus_shuffled_edge_AUC"] is not None
                and raw["real_minus_shuffled_edge_AUC"] >= 0.10
                and raw["precision_top5k_minus_shared"] is not None
                and raw["precision_top5k_minus_shared"] >= 0.10
            )
            out.extend(metric_rows.values())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc observer-count repair audit for v46 raw view-consensus edges.")
    parser.add_argument("--input-roots", required=True, help="Comma-separated audit roots from run_v46_raw_carrier_incidence_repair.py")
    parser.add_argument("--min-observer-counts", default="0,1,2,5,10,18")
    parser.add_argument("--output-root", default="outputs/audit/v46_observer_filter_repair")
    args = parser.parse_args()

    rows_out: list[dict[str, Any]] = []
    input_roots = [ROOT / item.strip() for item in str(args.input_roots).split(",") if item.strip()]
    mins = [int(item.strip()) for item in str(args.min_observer_counts).split(",") if item.strip()]
    for root in input_roots:
        edge_rows = _read_csv(root / "raw_view_consensus_edge_rows.csv")
        root_name = str(root.relative_to(ROOT) if root.is_relative_to(ROOT) else root)
        for min_observer_count in mins:
            rows_out.extend(_summarize(edge_rows, root_name=root_name, min_observer_count=int(min_observer_count)))
    raw_rows = [row for row in rows_out if row["score_key"] == "raw_view_consensus"]
    gate = {
        "any_scene_gate_pass": any(bool(row.get("gate_pass")) for row in raw_rows),
        "all_scene_gate_pass": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    for root_name in sorted({str(row["input_root"]) for row in raw_rows}):
        root_rows = [row for row in raw_rows if row["input_root"] == root_name]
        for variant in sorted({str(row["variant"]) for row in root_rows}):
            for min_observer_count in sorted({int(row["min_observer_count"]) for row in root_rows}):
                selected = [row for row in root_rows if str(row["variant"]) == variant and int(row["min_observer_count"]) == min_observer_count]
                if selected and all(bool(row.get("gate_pass")) for row in selected):
                    gate["all_scene_gate_pass"] = True
    gate["pass"] = bool(gate["all_scene_gate_pass"])
    payload = {
        "phase": "v46_observer_filter_repair",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_roots": [str(root.relative_to(ROOT) if root.is_relative_to(ROOT) else root) for root in input_roots],
        "min_observer_counts": mins,
        "summary_rows": rows_out,
        "gate": gate,
    }
    out = ROOT / str(args.output_root)
    _write_json(out / "observer_filter_repair.json", payload)
    _write_csv(out / "observer_filter_summary_rows.csv", rows_out)
    print(json.dumps({"summary": str(out / "observer_filter_repair.json"), "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

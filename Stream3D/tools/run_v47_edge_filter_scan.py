from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stream4d_native.v47_common import ROOT, write_csv, write_json


def _parse_int_grid(spec: str) -> list[int]:
    spec = str(spec).strip()
    if ":" in spec:
        parts = [int(float(part)) for part in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"int grid must be start:end:step, got {spec!r}")
        start, end, step = parts
        if step <= 0:
            raise ValueError(f"grid step must be positive, got {spec!r}")
        return list(range(start, end + 1, step))
    return [int(float(part)) for part in spec.split(",") if part.strip()]


def _parse_float_grid(spec: str) -> list[float]:
    spec = str(spec).strip()
    if ":" in spec:
        raw = [float(part) for part in spec.split(":")]
        if len(raw) != 3:
            raise ValueError(f"float grid must be start:end:step, got {spec!r}")
        start, end, step = raw
        if step <= 0.0:
            raise ValueError(f"grid step must be positive, got {spec!r}")
        values: list[float] = []
        current = start
        while current <= end + (step * 0.5):
            values.append(round(float(current), 6))
            current += step
        return values
    return [float(part) for part in spec.split(",") if part.strip()]


def _safe_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    label_values = labels.astype(bool).to_numpy()
    if label_values.sum() == 0 or label_values.sum() == label_values.size:
        return None
    return float(roc_auc_score(label_values, scores.astype(float).to_numpy()))


def _top_metrics(frame: pd.DataFrame, score_key: str) -> dict[str, float | None]:
    if frame.empty:
        return {"edge_precision@top1_per_node": None, "edge_recall@top3_per_node": None}
    group_cols = ["scene", "src_node_id"]
    idx = frame.groupby(group_cols, sort=False)[score_key].idxmax()
    top1 = float(frame.loc[idx, "diagnostic_same_gt"].astype(bool).mean()) if len(idx) else None

    positive_groups = frame.groupby(group_cols, sort=False)["diagnostic_same_gt"].any()
    positive_group_keys = set(positive_groups[positive_groups].index)
    if not positive_group_keys:
        top3 = None
    else:
        ranked = frame.sort_values(group_cols + [score_key], ascending=[True, True, False])
        top3_rows = ranked.groupby(group_cols, sort=False).head(3)
        top3_hits = top3_rows.groupby(group_cols, sort=False)["diagnostic_same_gt"].any()
        top3 = float(np.mean([bool(top3_hits.get(key, False)) for key in positive_group_keys]))
    return {"edge_precision@top1_per_node": top1, "edge_recall@top3_per_node": top3}


def _filter_frame(frame: pd.DataFrame, min_fwd: int, min_bwd: int, max_visible_outside: float) -> pd.DataFrame:
    return frame[
        (frame["forward_visible_carrier_count"] >= int(min_fwd))
        & (frame["backward_visible_carrier_count"] >= int(min_bwd))
        & (frame["visible_outside"] <= float(max_visible_outside))
    ]


def _rank_value(row: dict[str, Any]) -> tuple[float, float, float]:
    margin_shuffled = float(row.get("real_minus_shuffled_edge_AUC") or -999.0)
    margin_no_temporal = float(row.get("real_minus_no_temporal_edge_AUC") or -999.0)
    auc = float(row.get("edge_AUC") or -999.0)
    return (min(margin_shuffled - 0.10, margin_no_temporal - 0.08), margin_no_temporal, auc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan v47 global edge-filter repairs for the adjacent-edge audit.")
    parser.add_argument("--edge-table", default="outputs/audit/v47_adjacent_edges_adjacent_only/temporal_candidate_edge_table.csv")
    parser.add_argument("--output-root", default="outputs/audit/v47_edge_filter_scan_adjacent_only")
    parser.add_argument("--score-keys", default="A3_d4rt_symmetric,A4_d4rt_visible_veto,A5_d4rt_semantic_confirmation")
    parser.add_argument("--min-forward-grid", default="0:30:2")
    parser.add_argument("--min-backward-grid", default="0:30:2")
    parser.add_argument(
        "--max-visible-outside-grid",
        default="1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4",
    )
    parser.add_argument("--min-edge-count", type=int, default=500)
    parser.add_argument("--topn", type=int, default=100)
    args = parser.parse_args()

    edge_path = ROOT / str(args.edge_table)
    out_root = ROOT / str(args.output_root)
    score_keys = [item.strip() for item in str(args.score_keys).split(",") if item.strip()]
    min_forward_grid = _parse_int_grid(str(args.min_forward_grid))
    min_backward_grid = _parse_int_grid(str(args.min_backward_grid))
    max_visible_outside_grid = _parse_float_grid(str(args.max_visible_outside_grid))

    frame = pd.read_csv(edge_path)
    numeric_cols = [
        "src_node_id",
        "forward_visible_carrier_count",
        "backward_visible_carrier_count",
        "visible_outside",
        "A7_shuffled_D4RT",
        "A8_no_temporal_control",
        *score_keys,
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["diagnostic_same_gt"] = frame["diagnostic_same_gt"].astype(str).str.lower().isin({"true", "1", "yes", "y"})

    records: list[dict[str, Any]] = []
    for min_fwd in min_forward_grid:
        for min_bwd in min_backward_grid:
            for max_visible_outside in max_visible_outside_grid:
                sub = _filter_frame(frame, min_fwd, min_bwd, max_visible_outside)
                if len(sub) < int(args.min_edge_count):
                    continue
                labels = sub["diagnostic_same_gt"]
                shuffled_auc = _safe_auc(labels, sub["A7_shuffled_D4RT"])
                no_temporal_auc = _safe_auc(labels, sub["A8_no_temporal_control"])
                if shuffled_auc is None or no_temporal_auc is None:
                    continue
                for score_key in score_keys:
                    edge_auc = _safe_auc(labels, sub[score_key])
                    if edge_auc is None:
                        continue
                    records.append(
                        {
                            "score_key": score_key,
                            "edge_count": int(len(sub)),
                            "min_forward_visible_carrier_count": int(min_fwd),
                            "min_backward_visible_carrier_count": int(min_bwd),
                            "max_visible_outside": float(max_visible_outside),
                            "edge_AUC": edge_auc,
                            "shuffled_edge_AUC": shuffled_auc,
                            "no_temporal_edge_AUC": no_temporal_auc,
                            "real_minus_shuffled_edge_AUC": float(edge_auc - shuffled_auc),
                            "real_minus_no_temporal_edge_AUC": float(edge_auc - no_temporal_auc),
                            "edge_precision@top1_per_node": None,
                            "edge_recall@top3_per_node": None,
                            "gate_pass": False,
                        }
                    )

    records.sort(key=_rank_value, reverse=True)
    auc_passing = [
        row
        for row in records
        if float(row["real_minus_shuffled_edge_AUC"]) >= 0.10 and float(row["real_minus_no_temporal_edge_AUC"]) >= 0.08
    ]
    rows_for_top_metrics = records[: max(int(args.topn), 0)]
    seen = {
        (
            row["score_key"],
            row["min_forward_visible_carrier_count"],
            row["min_backward_visible_carrier_count"],
            row["max_visible_outside"],
        )
        for row in rows_for_top_metrics
    }
    for row in auc_passing:
        key = (
            row["score_key"],
            row["min_forward_visible_carrier_count"],
            row["min_backward_visible_carrier_count"],
            row["max_visible_outside"],
        )
        if key not in seen:
            rows_for_top_metrics.append(row)
            seen.add(key)

    for row in rows_for_top_metrics:
        sub = _filter_frame(
            frame,
            int(row["min_forward_visible_carrier_count"]),
            int(row["min_backward_visible_carrier_count"]),
            float(row["max_visible_outside"]),
        )
        top = _top_metrics(sub, str(row["score_key"]))
        row.update(top)
        row["gate_pass"] = bool(
            (float(row.get("edge_precision@top1_per_node") or 0.0) >= 0.80)
            and (float(row.get("edge_recall@top3_per_node") or 0.0) >= 0.55)
            and (float(row.get("real_minus_shuffled_edge_AUC") or 0.0) >= 0.10)
            and (float(row.get("real_minus_no_temporal_edge_AUC") or 0.0) >= 0.08)
        )

    passing = [row for row in rows_for_top_metrics if bool(row.get("gate_pass"))]
    top_rows = rows_for_top_metrics[: int(args.topn)]
    summary = {
        "phase": "v47_edge_filter_scan",
        "edge_table": str(edge_path),
        "output_root": str(out_root),
        "score_keys": score_keys,
        "min_forward_grid": min_forward_grid,
        "min_backward_grid": min_backward_grid,
        "max_visible_outside_grid": max_visible_outside_grid,
        "min_edge_count": int(args.min_edge_count),
        "config_score_rows": int(len(records)),
        "auc_passing_rows": int(len(auc_passing)),
        "gate_passing_rows": int(len(passing)),
        "best_row": top_rows[0] if top_rows else None,
        "best_gate_passing_row": passing[0] if passing else None,
        "gate": {"pass": bool(passing)},
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    write_csv(out_root / "filter_scan_top_rows.csv", top_rows)
    write_csv(out_root / "filter_scan_auc_passing_rows.csv", auc_passing)
    write_csv(out_root / "filter_scan_gate_passing_rows.csv", passing)
    write_json(out_root / "filter_scan_summary.json", summary)
    print({"summary": str(out_root / "filter_scan_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()

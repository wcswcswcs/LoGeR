from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from stream4d_native.v47_common import ROOT, read_csv, write_csv, write_json
from stream4d_native.v47_tracklet_builder import build_tracklets


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _summary_fields(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_selected_edge_count": summary.get("selected_edge_count"),
        f"{prefix}_tracklet_count": summary.get("tracklet_count"),
        f"{prefix}_tracklet_length_mean": summary.get("tracklet_length_mean"),
        f"{prefix}_tracklet_purity": summary.get("tracklet_purity"),
        f"{prefix}_tracklet_completeness": summary.get("tracklet_completeness"),
        f"{prefix}_tracklet_ARI": summary.get("tracklet_ARI"),
        f"{prefix}_scene0081_tracklet_purity": summary.get("scene0081_tracklet_purity"),
        f"{prefix}_scene0591_tracklet_purity": summary.get("scene0591_tracklet_purity"),
        f"{prefix}_gate_pass": summary.get("gate", {}).get("pass"),
    }


def _candidate_rows(frame: pd.DataFrame, max_configs: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["_original_rank"] = np.arange(len(frame))
    if max_configs <= 0 or len(frame) <= max_configs:
        return frame
    top_rank = frame.head(max_configs // 2)
    top_edge_count = frame.sort_values(["edge_count", "real_minus_no_temporal_edge_AUC"], ascending=[False, False]).head(
        max_configs - len(top_rank)
    )
    merged = pd.concat([top_rank, top_edge_count], ignore_index=True)
    key_cols = ["score_key", "min_forward_visible_carrier_count", "min_backward_visible_carrier_count", "max_visible_outside"]
    return merged.drop_duplicates(key_cols).sort_values("_original_rank")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan v47 Phase2-passing edge filters at the tracklet-control layer.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables_metricfix")
    parser.add_argument("--edge-table", default="outputs/audit/v47_adjacent_edges_adjacent_only/temporal_candidate_edge_table.csv")
    parser.add_argument("--filter-rows", default="outputs/audit/v47_edge_filter_scan_adjacent_only_coarse/filter_scan_gate_passing_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v47_tracklet_filter_scan_adjacent_only")
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--edge-types", default="adjacent")
    parser.add_argument("--max-configs", type=int, default=400)
    parser.add_argument("--control-ignore-edge-accept-candidate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    obs_root = ROOT / str(args.observation_root)
    edge_path = ROOT / str(args.edge_table)
    filter_path = ROOT / str(args.filter_rows)
    out_root = ROOT / str(args.output_root)
    edge_types = {item.strip() for item in str(args.edge_types).split(",") if item.strip()}

    mask_rows = read_csv(obs_root / "mask_observation_table.csv")
    raw_edge_rows = read_csv(edge_path)
    edge_frame = pd.read_csv(edge_path)
    filter_frame = pd.read_csv(filter_path)
    filter_frame = _candidate_rows(filter_frame, int(args.max_configs))

    for col in ["forward_visible_carrier_count", "backward_visible_carrier_count", "visible_outside"]:
        edge_frame[col] = pd.to_numeric(edge_frame[col], errors="coerce").fillna(0.0)

    results: list[dict[str, Any]] = []
    for idx, row in filter_frame.iterrows():
        min_fwd = int(_as_float(row["min_forward_visible_carrier_count"]))
        min_bwd = int(_as_float(row["min_backward_visible_carrier_count"]))
        max_outside = _as_float(row["max_visible_outside"])
        score_key = str(row["score_key"])
        keep = (
            (edge_frame["forward_visible_carrier_count"] >= min_fwd)
            & (edge_frame["backward_visible_carrier_count"] >= min_bwd)
            & (edge_frame["visible_outside"] <= max_outside)
        ).to_numpy()
        selected_edge_rows = [raw_edge_rows[i] for i in np.flatnonzero(keep)]
        if not selected_edge_rows:
            continue
        real = build_tracklets(
            mask_rows=mask_rows,
            edge_rows=selected_edge_rows,
            score_key=score_key,
            min_score=float(args.min_score),
            edge_types=edge_types,
            respect_edge_accept_candidate=True,
        )["summary"]
        shuffled = build_tracklets(
            mask_rows=mask_rows,
            edge_rows=selected_edge_rows,
            score_key="A7_shuffled_D4RT",
            min_score=float(args.min_score),
            edge_types=edge_types,
            respect_edge_accept_candidate=not bool(args.control_ignore_edge_accept_candidate),
        )["summary"]
        no_temporal = build_tracklets(
            mask_rows=mask_rows,
            edge_rows=selected_edge_rows,
            score_key="A8_no_temporal_control",
            min_score=float(args.min_score),
            edge_types=edge_types,
            respect_edge_accept_candidate=not bool(args.control_ignore_edge_accept_candidate),
        )["summary"]
        real_ari = _as_float(real.get("tracklet_ARI"))
        shuffled_ari = _as_float(shuffled.get("tracklet_ARI"))
        no_temporal_ari = _as_float(no_temporal.get("tracklet_ARI"))
        out = {
            "candidate_index": int(idx),
            "score_key": score_key,
            "edge_count": int(len(selected_edge_rows)),
            "min_forward_visible_carrier_count": min_fwd,
            "min_backward_visible_carrier_count": min_bwd,
            "max_visible_outside": max_outside,
            "edge_AUC": _as_float(row.get("edge_AUC")),
            "edge_real_minus_shuffled_AUC": _as_float(row.get("real_minus_shuffled_edge_AUC")),
            "edge_real_minus_no_temporal_AUC": _as_float(row.get("real_minus_no_temporal_edge_AUC")),
            **_summary_fields(real, "real"),
            **_summary_fields(shuffled, "shuffled"),
            **_summary_fields(no_temporal, "no_temporal"),
            "real_minus_shuffled_tracklet_ARI": float(real_ari - shuffled_ari),
            "real_minus_no_temporal_tracklet_ARI": float(real_ari - no_temporal_ari),
        }
        out["tracklet_control_gate_pass"] = bool(
            bool(real.get("gate", {}).get("pass"))
            and out["real_minus_shuffled_tracklet_ARI"] >= 0.20
            and out["real_minus_no_temporal_tracklet_ARI"] >= 0.10
        )
        results.append(out)

    results.sort(
        key=lambda item: (
            bool(item.get("tracklet_control_gate_pass")),
            float(item.get("real_minus_no_temporal_tracklet_ARI") or -999.0),
            float(item.get("real_tracklet_ARI") or -999.0),
        ),
        reverse=True,
    )
    passing = [row for row in results if bool(row.get("tracklet_control_gate_pass"))]
    summary = {
        "phase": "v47_tracklet_filter_scan",
        "edge_table": str(edge_path),
        "filter_rows": str(filter_path),
        "output_root": str(out_root),
        "candidate_rows_available": int(len(pd.read_csv(filter_path))),
        "candidate_rows_scanned": int(len(filter_frame)),
        "result_rows": int(len(results)),
        "gate_passing_rows": int(len(passing)),
        "control_ignore_edge_accept_candidate": bool(args.control_ignore_edge_accept_candidate),
        "best_row": results[0] if results else None,
        "best_gate_passing_row": passing[0] if passing else None,
        "gate": {"pass": bool(passing)},
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    write_csv(out_root / "tracklet_filter_scan_rows.csv", results)
    write_csv(out_root / "tracklet_filter_scan_gate_passing_rows.csv", passing)
    write_json(out_root / "tracklet_filter_scan_summary.json", summary)
    print({"summary": str(out_root / "tracklet_filter_scan_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()

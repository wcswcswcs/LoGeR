#!/usr/bin/env python3
"""Build a diagnostic five-chunk regime-shift signal for v78 TTT follow-up.

This tool consumes already generated bad-vs-reference geometry-regime deltas.
It does not run LoGeR, does not use labels beyond the mined bad/reference pairs,
and does not claim a method gate.  The goal is to make the TTT visual/metric
follow-up focus on long-window appearance and geometry regime shifts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_DELTAS_CSV = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "bad_good_case_contrast/v2_unique_scenes_top5/geometry_regime_audit/"
    "geometry_regime_feature_deltas.csv"
)
DEFAULT_VISUAL_MANIFEST_CSV = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "bad_good_case_contrast/v2_unique_scenes_top5/visual_artifact_manifest.csv"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "bad_good_case_contrast/v2_unique_scenes_top5/ttt_five_chunk_regime_signal_v1"
)


SIGNAL_FEATURES = [
    {
        "feature": "dark_frac_std",
        "delta": "delta_dark_frac_std",
        "direction": 1.0,
        "reason": "bad window has more within-window darkness variation",
    },
    {
        "feature": "dark_frac_temporal_range",
        "delta": "delta_dark_frac_temporal_range",
        "direction": 1.0,
        "reason": "bad window has larger shadow/dark temporal range",
    },
    {
        "feature": "dark_frac_mean",
        "delta": "delta_dark_frac_mean",
        "direction": 1.0,
        "reason": "bad window is darker on average",
    },
    {
        "feature": "luminance_mean_mean",
        "delta": "delta_luminance_mean_mean",
        "direction": -1.0,
        "reason": "bad window has lower average luminance",
    },
    {
        "feature": "road_center_range_std",
        "delta": "delta_road_center_range_std",
        "direction": 1.0,
        "reason": "bad window has less stable road/corridor geometry",
    },
    {
        "feature": "road_center_range_temporal_range",
        "delta": "delta_road_center_range_temporal_range",
        "direction": 1.0,
        "reason": "bad window has larger road/corridor shift over time",
    },
    {
        "feature": "road_edge_confidence_mean_mean",
        "delta": "delta_road_edge_confidence_mean_mean",
        "direction": -1.0,
        "reason": "bad window has lower road-edge confidence",
    },
    {
        "feature": "semantic_boundary_density_temporal_range",
        "delta": "delta_semantic_boundary_density_temporal_range",
        "direction": 1.0,
        "reason": "bad window has more changing object/wall/road-edge boundaries",
    },
    {
        "feature": "vegetation_frac_temporal_range",
        "delta": "delta_vegetation_frac_temporal_range",
        "direction": 1.0,
        "reason": "bad window has more changing tree/vegetation context",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas-csv", type=Path, default=DEFAULT_DELTAS_CSV)
    parser.add_argument(
        "--visual-manifest-csv", type=Path, default=DEFAULT_VISUAL_MANIFEST_CSV
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _visual_by_rank(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("family") != "five_chunk":
            continue
        rank = row.get("contrast_rank", "")
        if rank and rank not in out:
            out[rank] = row
    return out


def _scales(rows: list[dict[str, str]]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for spec in SIGNAL_FEATURES:
        values = []
        for row in rows:
            raw = _float(row, spec["delta"])
            if raw is None:
                continue
            values.append(abs(raw * float(spec["direction"])))
        scales[spec["delta"]] = mean(values) if values and mean(values) > 0 else 1.0
    return scales


def _build_rows(
    delta_rows: list[dict[str, str]], visual_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    five_rows = [row for row in delta_rows if row.get("family") == "five_chunk"]
    scales = _scales(five_rows)
    visual = _visual_by_rank(visual_rows)
    out: list[dict[str, Any]] = []
    for row in five_rows:
        vote_count = 0
        components: dict[str, float | None] = {}
        normalized: dict[str, float | None] = {}
        for spec in SIGNAL_FEATURES:
            raw = _float(row, spec["delta"])
            signed = None if raw is None else raw * float(spec["direction"])
            components[f"{spec['feature']}_signed_delta"] = signed
            norm = None
            if signed is not None:
                norm = signed / scales[spec["delta"]]
                if signed > 0:
                    vote_count += 1
            normalized[f"{spec['feature']}_norm"] = norm
        valid_norms = [v for v in normalized.values() if v is not None]
        score = mean(valid_norms) if valid_norms else None
        rank = row.get("contrast_rank", "")
        visual_row = visual.get(rank, {})
        out_row: dict[str, Any] = {
            "family": row.get("family", ""),
            "contrast_rank": rank,
            "bad_case": row.get("bad_case", ""),
            "reference_case": row.get("reference_case", ""),
            "metric": row.get("metric", ""),
            "bad_metric_value": row.get("bad_metric_value", ""),
            "reference_metric_value": row.get("reference_metric_value", ""),
            "reference_strategy": row.get("reference_strategy", ""),
            "expected_direction_vote_count": vote_count,
            "expected_direction_vote_rate": vote_count / len(SIGNAL_FEATURES),
            "ttt_regime_shift_score": score,
            "visual_file": visual_row.get("visual_file", ""),
            "visual_sha256": visual_row.get("sha256", ""),
            "diagnostic_interpretation": (
                "higher score means the bad five-chunk window more strongly matches "
                "the expected shadow/exposure/corridor/road-edge regime-shift pattern"
            ),
        }
        out_row.update(components)
        out_row.update(normalized)
        out.append(out_row)
    out.sort(
        key=lambda r: (
            float(r["ttt_regime_shift_score"])
            if r.get("ttt_regime_shift_score") is not None
            else -999.0
        ),
        reverse=True,
    )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    delta_rows = _read_csv(args.deltas_csv)
    visual_rows = _read_csv(args.visual_manifest_csv)
    rows = _build_rows(delta_rows, visual_rows)
    out_rows = args.out_dir / "ttt_five_chunk_regime_signal_rows.csv"
    out_summary = args.out_dir / "ttt_five_chunk_regime_signal_summary.json"
    _write_csv(out_rows, rows)
    summary = {
        "schema": "acl2_v78_ttt_five_chunk_regime_signal_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "deltas_csv": str(args.deltas_csv),
        "visual_manifest_csv": str(args.visual_manifest_csv),
        "num_rows": len(rows),
        "signal_features": SIGNAL_FEATURES,
        "top_rows": rows[:3],
        "interpretation": [
            "This signal ranks five-chunk bad/reference windows for TTT visual review.",
            "It is derived from existing mined bad/reference feature deltas.",
            "It should not be used as a runtime TTT update rule without a held-out gate.",
        ],
        "next_required_evidence": [
            "Inspect the top-ranked panels for shadow/exposure and corridor/road-edge regime shifts.",
            "Check whether a no-GT long-window shift detector predicts failures outside these mined rows.",
            "Only then test a TTT write/update/freeze gate.",
        ],
        "rows_csv": str(out_rows),
    }
    _write_json(out_summary, summary)
    print(json.dumps({"rows": str(out_rows), "summary": str(out_summary)}, indent=2))


if __name__ == "__main__":
    main()

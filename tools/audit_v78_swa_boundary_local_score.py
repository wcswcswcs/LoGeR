#!/usr/bin/env python3
"""Audit a diagnostic SWA boundary-local geometry/visibility score.

The score is fitted from existing bad/reference geometry-regime case features.
It is diagnostic-only: it does not run LoGeR and does not claim a method gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_v78_geometry_regime_contrast import (  # noqa: E402
    _aggregate_frame_features,
    _frame_features,
)


DEFAULT_RGB_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")


SCORE_FEATURES: list[tuple[str, float, str]] = [
    ("dark_frac_std", 1.0, "bad windows have more within-window darkness variation"),
    ("dark_frac_temporal_range", 1.0, "bad windows have larger temporal dark-region change"),
    ("dark_frac_mean", 1.0, "bad windows are darker on average"),
    ("vegetation_frac_mean", 1.0, "bad windows have more vegetation/tree/mountain context"),
    ("road_center_range_std", 1.0, "bad windows have less stable road-center geometry"),
    ("road_center_range_temporal_range", 1.0, "bad windows have larger road/corridor shift over time"),
    ("road_edge_confidence_mean_mean", -1.0, "bad windows have lower road-edge confidence"),
    ("luminance_mean_mean", -1.0, "bad windows have lower average luminance"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-features-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--rgb-root", type=Path, default=DEFAULT_RGB_ROOT)
    parser.add_argument(
        "--extra-window",
        action="append",
        default=[],
        help=(
            "NAME=LABEL:SEQ:START:BOUNDARY:END. Example: "
            "P9_34_KITTI01=weak_positive:01:145:174:206"
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fit_scaler(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    scaler: dict[str, dict[str, float]] = {}
    for feature, _, _ in SCORE_FEATURES:
        vals = np.asarray(
            [float(row[feature]) for row in rows if _finite(row.get(feature)) is not None],
            dtype=np.float64,
        )
        if vals.size == 0:
            scaler[feature] = {"mean": 0.0, "std": 1.0, "n": 0}
            continue
        std = float(np.std(vals))
        scaler[feature] = {
            "mean": float(np.mean(vals)),
            "std": std if std > 1e-12 else 1.0,
            "n": int(vals.size),
        }
    return scaler


def _score_row(row: dict[str, Any], scaler: dict[str, dict[str, float]]) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    for feature, direction, _ in SCORE_FEATURES:
        value = _finite(row.get(feature))
        stats = scaler[feature]
        if value is None:
            continue
        components[feature] = float(direction) * ((value - stats["mean"]) / stats["std"])
    if not components:
        return float("nan"), {}
    return float(np.mean(list(components.values()))), components


def _parse_extra_window(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise ValueError(f"Expected NAME=LABEL:SEQ:START:BOUNDARY:END, got {spec!r}")
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    if len(parts) != 5:
        raise ValueError(f"Expected LABEL:SEQ:START:BOUNDARY:END, got {rest!r}")
    label, seq, start, boundary, end = parts
    return {
        "name": name,
        "label": label,
        "sequence": str(seq).zfill(2),
        "start": int(start),
        "boundary": int(boundary),
        "end": int(end),
    }


def _index_rows(preprocess_root: Path, seq: str) -> list[dict[str, Any]]:
    path = preprocess_root / str(seq).zfill(2) / "stage_c_cache_semantic_chunks" / "cache_index.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _chunk_path_for_frame(preprocess_root: Path, seq: str, frame: int, index_rows: list[dict[str, Any]]) -> tuple[Path, int]:
    for row in index_rows:
        if int(row["start_frame"]) <= int(frame) < int(row["end_frame"]):
            return (
                preprocess_root / str(seq).zfill(2) / "stage_c_cache_semantic_chunks" / row["chunk"] / "masklet.pt",
                int(frame) - int(row["start_frame"]),
            )
    raise FileNotFoundError(f"No stage-C chunk cache for seq={seq} frame={frame}")


def _rgb_path(rgb_root: Path, seq: str, frame: int) -> Path:
    base = rgb_root / str(seq).zfill(2) / "image_2"
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = base / f"{int(frame):06d}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No RGB image for seq={seq} frame={frame}")


def _window_frames(start: int, boundary: int, end: int) -> list[int]:
    frames = [int(start), int(boundary) - 1, int(boundary), int(end) - 1]
    out: list[int] = []
    for frame in frames:
        if int(start) <= frame < int(end) and frame not in out:
            out.append(frame)
    return out


def _extra_window_features(
    *,
    spec: dict[str, Any],
    preprocess_root: Path,
    rgb_root: Path,
    chunk_cache: dict[Path, Any],
) -> dict[str, Any]:
    seq = str(spec["sequence"]).zfill(2)
    index = _index_rows(preprocess_root, seq)
    frame_rows: list[dict[str, Any]] = []
    source_assets: list[dict[str, Any]] = []
    for frame in _window_frames(int(spec["start"]), int(spec["boundary"]), int(spec["end"])):
        chunk_path, local = _chunk_path_for_frame(preprocess_root, seq, frame, index)
        asset = {
            "frame": int(frame),
            "rgb_path": str(_rgb_path(rgb_root, seq, frame)),
            "path": str(chunk_path),
            "local_frame": int(local),
        }
        features = _frame_features(asset, chunk_cache)
        features.update({"frame": int(frame), "valid": bool(features.get("valid"))})
        frame_rows.append(features)
        source_assets.append(asset)
    agg = _aggregate_frame_features(frame_rows)
    out: dict[str, Any] = {
        "source": "extra_window",
        "family": "adjacent_pair",
        "contrast_rank": "",
        "role": spec["label"],
        "run": spec["name"],
        "sequence": seq,
        "case_id": f"{int(spec['start'])}-{int(spec['end'])}",
        "metric": "",
        "case_metric_value": "",
        "reference_strategy": "",
        "window_start": int(spec["start"]),
        "boundary_frame": int(spec["boundary"]),
        "window_end": int(spec["end"]),
        "window_key": f"{seq}:{int(spec['start'])}:{int(spec['boundary'])}:{int(spec['end'])}",
        "sample_frames": json.dumps(_window_frames(int(spec["start"]), int(spec["boundary"]), int(spec["end"]))),
        "source_assets": json.dumps(source_assets, ensure_ascii=False),
    }
    out.update(agg)
    return out


def _pairwise_summary(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scored_rows:
        if row.get("source") != "case_features":
            continue
        key = (str(row.get("family")), str(row.get("contrast_rank")))
        pairs.setdefault(key, {})[str(row.get("role"))] = row
    rows: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, int]] = {}
    for (family, rank), pair in sorted(pairs.items()):
        bad = pair.get("bad")
        ref = pair.get("reference")
        if not bad or not ref:
            continue
        bad_score = _finite(bad.get("boundary_local_score"))
        ref_score = _finite(ref.get("boundary_local_score"))
        if bad_score is None or ref_score is None:
            continue
        win = bad_score > ref_score
        stats = by_family.setdefault(family, {"pairs": 0, "bad_score_gt_reference": 0})
        stats["pairs"] += 1
        stats["bad_score_gt_reference"] += int(win)
        rows.append(
            {
                "family": family,
                "contrast_rank": rank,
                "bad_case": bad.get("case_id", ""),
                "reference_case": ref.get("case_id", ""),
                "bad_score": bad_score,
                "reference_score": ref_score,
                "score_margin": bad_score - ref_score,
                "bad_score_gt_reference": win,
                "bad_metric": bad.get("case_metric_value", ""),
                "reference_metric": ref.get("case_metric_value", ""),
            }
        )
    total_pairs = sum(v["pairs"] for v in by_family.values())
    total_wins = sum(v["bad_score_gt_reference"] for v in by_family.values())
    return {
        "rows": rows,
        "by_family": by_family,
        "total_pairs": total_pairs,
        "total_bad_score_gt_reference": total_wins,
        "total_win_rate": (total_wins / total_pairs) if total_pairs else None,
    }


def _extra_duplicate_windows(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scored_rows:
        if row.get("source") == "extra_window":
            grouped.setdefault(str(row.get("window_key", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        labels = sorted({str(row.get("role")) for row in rows})
        if len(labels) <= 1:
            continue
        scores = sorted({_finite(row.get("boundary_local_score")) for row in rows if _finite(row.get("boundary_local_score")) is not None})
        out.append(
            {
                "window_key": key,
                "labels": ",".join(labels),
                "num_rows": len(rows),
                "unique_scores": ",".join(f"{float(s):.9g}" for s in scores),
                "interpretation": (
                    "same geometry window has multiple action outcome labels; "
                    "scene-level score cannot distinguish action variants"
                ),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_rows = _read_csv(args.case_features_csv)
    for row in case_rows:
        row["source"] = "case_features"
        row["window_key"] = f"{row.get('sequence')}:{row.get('case_id')}"
    scaler = _fit_scaler(case_rows)
    chunk_cache: dict[Path, Any] = {}
    extra_rows = [
        _extra_window_features(
            spec=_parse_extra_window(spec),
            preprocess_root=args.preprocess_root,
            rgb_root=args.rgb_root,
            chunk_cache=chunk_cache,
        )
        for spec in args.extra_window
    ]

    scored_rows: list[dict[str, Any]] = []
    for row in [*case_rows, *extra_rows]:
        score, components = _score_row(row, scaler)
        out = dict(row)
        out["boundary_local_score"] = score
        out["score_components"] = json.dumps(components, sort_keys=True)
        scored_rows.append(out)

    pair_summary = _pairwise_summary(scored_rows)
    duplicate_extra = _extra_duplicate_windows(scored_rows)
    _write_csv(args.out_dir / "boundary_local_score_rows.csv", scored_rows)
    _write_csv(args.out_dir / "boundary_local_score_pairwise.csv", pair_summary["rows"])
    _write_csv(args.out_dir / "boundary_local_score_duplicate_extra_windows.csv", duplicate_extra)

    summary = {
        "schema": "acl2_v78_swa_boundary_local_score_audit_v1",
        "diagnostic_only": True,
        "case_features_csv": str(args.case_features_csv),
        "out_dir": str(args.out_dir),
        "score_features": [
            {"feature": feature, "direction": direction, "reason": reason}
            for feature, direction, reason in SCORE_FEATURES
        ],
        "scaler": scaler,
        "num_case_feature_rows": len(case_rows),
        "num_extra_windows": len(extra_rows),
        "pairwise_bad_vs_reference": {
            "total_pairs": pair_summary["total_pairs"],
            "total_bad_score_gt_reference": pair_summary["total_bad_score_gt_reference"],
            "total_win_rate": pair_summary["total_win_rate"],
            "by_family": pair_summary["by_family"],
        },
        "duplicate_extra_windows": duplicate_extra,
        "interpretation": (
            "The score measures scene-level geometry/visibility risk. If the same window has "
            "both weak-positive and weak-negative action outcomes, this score cannot by itself "
            "choose the runtime action."
        ),
    }
    _write_json(args.out_dir / "boundary_local_score_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
